#!/usr/bin/env python3
"""Strictly merge model-specific ESTAR collector shards in source-qid order."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_qids_sha256(qids: list[str]) -> str:
    payload = json.dumps(
        qids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {constant!r}")
                ),
            )
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def validate_static_contract(
    config: dict[str, Any],
    *,
    model_id: str,
    prompt_protocol: str,
    tensor_parallel_size: int,
) -> None:
    if config.get("model") != model_id:
        raise ValueError(f"model identity mismatch: {config.get('model')!r}")
    prompt = config.get("prompt") or {}
    engine = config.get("engine") or {}
    probe = config.get("probe") or {}
    if prompt.get("protocol") != prompt_protocol:
        raise ValueError(f"prompt protocol mismatch: {prompt}")
    if int(engine.get("tensor_parallel_size", -1)) != tensor_parallel_size:
        raise ValueError(f"tensor-parallel contract mismatch: {engine}")
    if probe.get("checkpoints") != [index / 10.0 for index in range(1, 11)]:
        raise ValueError("collector must use the audited ten-decile probe schedule")
    if int(probe.get("max_tokens", -1)) != 64 or probe.get("qa_mode") != "openqa":
        raise ValueError(f"probe contract mismatch: {probe}")


def merge(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_jsonl(args.input)
    source_qids = [str(row.get("unique_id") or "") for row in source_rows]
    if (
        len(source_rows) != args.expected_qids
        or any(not qid for qid in source_qids)
        or len(set(source_qids)) != len(source_qids)
    ):
        raise ValueError("source qid contract mismatch")
    source_sha = file_sha256(args.input)
    source_qids_sha = ordered_qids_sha256(source_qids)

    by_qid: dict[str, dict[str, Any]] = {}
    reference_without_selection: str | None = None
    worker_summaries: list[dict[str, Any]] = []
    canonical_template: dict[str, Any] | None = None
    for worker_index in range(args.shard_count):
        output = args.workers_dir / f"worker_{worker_index}.jsonl"
        errors = args.workers_dir / f"worker_{worker_index}.errors.jsonl"
        if not output.is_file() or not errors.is_file():
            raise FileNotFoundError(f"missing worker artifact for shard {worker_index}")
        if errors.stat().st_size:
            raise ValueError(f"worker {worker_index} recorded errors: {errors}")
        rows = load_jsonl(output)
        expected_qids = source_qids[worker_index :: args.shard_count]
        actual_qids: list[str] = []
        for row_number, row in enumerate(rows, 1):
            qid = str(row.get("qid") or "")
            if not qid or qid in by_qid:
                raise ValueError(
                    f"{output}:{row_number}: empty or duplicate qid {qid!r}"
                )
            config = row.get("generation_config")
            fingerprint = row.get("generation_config_sha256")
            if not isinstance(config, dict) or not isinstance(fingerprint, str):
                raise TypeError(f"{output}:{row_number}: invalid generation config")
            if config_sha256(config) != fingerprint:
                raise ValueError(f"{output}:{row_number}: generation fingerprint mismatch")
            source = config.get("source") or {}
            selection = config.get("selection") or {}
            if source != {
                "input_sha256": source_sha,
                "row_count": len(source_qids),
                "ordered_qids_sha256": source_qids_sha,
            }:
                raise ValueError(f"{output}:{row_number}: source contract mismatch")
            if selection != {
                "shard_index": worker_index,
                "shard_count": args.shard_count,
                "limit": 0,
                "row_count": len(expected_qids),
                "ordered_qids_sha256": ordered_qids_sha256(expected_qids),
            }:
                raise ValueError(f"{output}:{row_number}: selection contract mismatch")
            validate_static_contract(
                config,
                model_id=args.model_id,
                prompt_protocol=args.prompt_protocol,
                tensor_parallel_size=args.tensor_parallel_size,
            )
            without_selection = copy.deepcopy(config)
            without_selection.pop("selection", None)
            serialized = canonical_json(without_selection)
            if reference_without_selection is None:
                reference_without_selection = serialized
                canonical_template = copy.deepcopy(config)
            elif serialized != reference_without_selection:
                raise ValueError(f"{output}:{row_number}: mixed generation contracts")
            if len(row.get("probes") or []) != 10:
                raise ValueError(f"{output}:{row_number}: expected exactly ten probes")
            by_qid[qid] = row
            actual_qids.append(qid)
        if set(actual_qids) != set(expected_qids) or len(actual_qids) != len(expected_qids):
            missing = sorted(set(expected_qids) - set(actual_qids))[:20]
            extra = sorted(set(actual_qids) - set(expected_qids))[:20]
            raise ValueError(
                f"worker {worker_index} qid coverage mismatch; missing={missing}, extra={extra}"
            )
        worker_summaries.append(
            {
                "worker_index": worker_index,
                "rows": len(rows),
                "output": str(output.resolve()),
                "output_sha256": file_sha256(output),
                "errors_sha256": file_sha256(errors),
            }
        )

    if set(by_qid) != set(source_qids) or canonical_template is None:
        raise ValueError("merged workers do not exactly cover the source")
    canonical_config = copy.deepcopy(canonical_template)
    canonical_config["selection"] = {
        "shard_index": 0,
        "shard_count": 1,
        "limit": 0,
        "row_count": len(source_qids),
        "ordered_qids_sha256": source_qids_sha,
    }
    canonical_fingerprint = config_sha256(canonical_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for qid in source_qids:
                row = copy.deepcopy(by_qid[qid])
                row["generation_config"] = canonical_config
                row["generation_config_sha256"] = canonical_fingerprint
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "format_version": 1,
        "status": "complete",
        "input": str(args.input.resolve()),
        "input_sha256": source_sha,
        "input_qids_sha256": source_qids_sha,
        "output": str(args.output.resolve()),
        "output_rows": len(source_qids),
        "output_sha256": file_sha256(args.output),
        "canonical_generation_config_sha256": canonical_fingerprint,
        "model_id": args.model_id,
        "prompt_protocol": args.prompt_protocol,
        "tensor_parallel_size": args.tensor_parallel_size,
        "shard_count": args.shard_count,
        "workers": worker_summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workers-dir", required=True, type=Path)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--prompt-protocol", required=True, choices=("qwen3_system", "deepseek_user")
    )
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--expected-qids", type=int, default=38918)
    args = parser.parse_args()
    if args.shard_count < 1 or args.tensor_parallel_size < 1:
        parser.error("shard count and tensor-parallel size must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(merge(parse_args()), ensure_ascii=False, sort_keys=True))
