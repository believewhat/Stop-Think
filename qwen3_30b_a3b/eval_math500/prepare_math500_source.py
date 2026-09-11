#!/usr/bin/env python3
"""Strictly adapt three audited Math500 baseline repeats for every-50 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
SCHEMA = "qwen30a3b_math500_repeat3_replay_source_v1"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"
EXPECTED_SEEDS = {"repeat1": 20260903, "repeat2": 20260904, "repeat3": 20260905}
EXPECTED_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 20,
    "repetition_penalty": 1.2,
    "max_model_len": 40960,
    "max_tokens": 32768,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def values_sha256(values: list[str]) -> str:
    return sha256_bytes(canonical_json(values).encode("utf-8"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_repeat(path: Path, repeat_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    all_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            all_rows += 1
            if row.get("eval_dataset") != "math500":
                continue
            config = row.get("run_config")
            if not isinstance(config, dict):
                raise TypeError(f"{path}:{line_number}: missing run_config")
            declared_fingerprint = str(row.get("run_fingerprint") or "")
            observed_fingerprint = sha256_bytes(canonical_json(config).encode("utf-8"))
            if declared_fingerprint != observed_fingerprint:
                raise ValueError(f"{path}:{line_number}: run fingerprint mismatch")
            expected = {
                "prompt_version": "qwen3_system_user_explicit_think_v1",
                "system_prompt": SYSTEM_PROMPT,
                "enable_thinking": True,
                "explicit_think_prefix": True,
                "seed": EXPECTED_SEEDS[repeat_id],
                **EXPECTED_SAMPLING,
            }
            drift = {key: {"expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value}
            if drift:
                raise ValueError(f"{path}:{line_number}: prompt/sampling drift: {drift}")
            if Path(str(config.get("model") or "")).name != "Qwen3-30B-A3B":
                raise ValueError(f"{path}:{line_number}: wrong backbone model")
            for field in ("qid", "problem", "gold_answer", "prediction", "is_correct", "answer_valid", "response_tokens", "response_text"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            if not str(row["qid"]) or not str(row["problem"]) or not isinstance(row["response_text"], str):
                raise ValueError(f"{path}:{line_number}: invalid qid/problem/response_text")
            selected.append(row)
    if len(selected) != 500:
        raise ValueError(f"{path}: expected exactly 500 Math500 rows, got {len(selected)}")
    qids = [str(row["qid"]) for row in selected]
    if len(set(qids)) != 500:
        raise ValueError(f"{path}: duplicate Math500 qid")
    return selected, {"path": str(path.resolve()), "sha256": file_sha256(path), "all_rows": all_rows, "math500_rows": 500}


def build_source(repeats: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    reference = {str(row["qid"]): row for row in repeats["repeat1"]}
    for repeat_id in ("repeat2", "repeat3"):
        current = {str(row["qid"]): row for row in repeats[repeat_id]}
        if set(current) != set(reference):
            raise ValueError(f"{repeat_id}: Math500 qid set differs from repeat1")
        for qid, first in reference.items():
            other = current[qid]
            if (str(other["problem"]), str(other["gold_answer"])) != (str(first["problem"]), str(first["gold_answer"])):
                raise ValueError(f"{repeat_id}/{qid}: problem or gold answer differs")

    result: list[dict[str, Any]] = []
    source_index = 0
    for repeat_id in ("repeat1", "repeat2", "repeat3"):
        for row in repeats[repeat_id]:
            original_qid = str(row["qid"])
            config = dict(row["run_config"])
            config_fingerprint = sha256_bytes(canonical_json(config).encode("utf-8"))
            result.append({
                "qid": f"{repeat_id}::{original_qid}",
                "source_index": source_index,
                "sample_rank": repeat_id,
                "sample_stratum": "math500",
                "problem": str(row["problem"]),
                "gold_answer": str(row["gold_answer"]),
                "generation_config": config,
                "generation_config_sha256": config_fingerprint,
                "main": {
                    "text": str(row["response_text"]),
                    "repeat_id": repeat_id,
                    "original_qid": original_qid,
                    "eval_dataset": "math500",
                    "prediction": str(row["prediction"]),
                    "reported_is_correct": bool(row["is_correct"]),
                    "answer_valid": bool(row["answer_valid"]),
                    "response_tokens": int(row["response_tokens"]),
                    "baseline_run_fingerprint": str(row["run_fingerprint"]),
                },
            })
            source_index += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat1", required=True, type=Path)
    parser.add_argument("--repeat2", required=True, type=Path)
    parser.add_argument("--repeat3", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    repeats: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for repeat_id in ("repeat1", "repeat2", "repeat3"):
        repeats[repeat_id], sources[repeat_id] = read_repeat(getattr(args, repeat_id), repeat_id)
    model_config_hashes = {str(row["run_config"].get("model_config_sha256") or "") for rows in repeats.values() for row in rows}
    if len(model_config_hashes) != 1 or "" in model_config_hashes:
        raise ValueError(f"model content identity differs or is absent: {model_config_hashes}")

    rows = build_source(repeats)
    content = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    content_sha = sha256_bytes(content)
    if args.output.exists() and file_sha256(args.output) != content_sha:
        raise ValueError(f"refusing to overwrite mismatched source: {args.output}")
    if not args.output.exists():
        atomic_write(args.output, content)
    manifest = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA,
        "status": "complete",
        "sources": sources,
        "expected_seeds": EXPECTED_SEEDS,
        "sampling": EXPECTED_SAMPLING,
        "prompt": {"system_prompt": SYSTEM_PROMPT, "prompt_version": "qwen3_system_user_explicit_think_v1", "enable_thinking": True, "explicit_think_prefix": True},
        "model": {"basename": "Qwen3-30B-A3B", "config_sha256": next(iter(model_config_hashes))},
        "selection": {"rows": 1500, "repeats": 3, "rows_per_repeat": 500, "qid_prefix": "{repeat_id}::{original_qid}", "ordered_qids_sha256": values_sha256([row["qid"] for row in rows])},
        "output": {"path": str(args.output.resolve()), "sha256": content_sha},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if args.manifest.exists() and args.manifest.read_bytes() != manifest_bytes:
        raise ValueError(f"refusing to overwrite mismatched manifest: {args.manifest}")
    if not args.manifest.exists():
        atomic_write(args.manifest, manifest_bytes)
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
