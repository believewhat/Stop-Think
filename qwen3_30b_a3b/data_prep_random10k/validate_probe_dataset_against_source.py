#!/usr/bin/env python3
"""Strictly bind merged probe records back to the frozen random-10k source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}: expected non-negative integer, got {value!r}")
    return value


def finite_features(value: Any, location: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{location}: features must be an object")
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{location}.{key}: expected finite numeric value")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_jsonl(args.input)
    probe_rows = load_jsonl(args.probes)
    if len(source_rows) != args.expected_qids or len(probe_rows) != args.expected_qids:
        raise ValueError(
            f"row mismatch input={len(source_rows)} probes={len(probe_rows)} "
            f"expected={args.expected_qids}"
        )

    source_qids = [str(row.get("unique_id") or "") for row in source_rows]
    probe_qids = [str(row.get("qid") or "") for row in probe_rows]
    if any(not qid for qid in source_qids + probe_qids):
        raise ValueError("empty qid")
    if len(set(source_qids)) != len(source_qids):
        raise ValueError("duplicate input qid")
    if len(set(probe_qids)) != len(probe_qids):
        raise ValueError("duplicate probe qid")
    if probe_qids != source_qids:
        raise ValueError("merged probe qid order differs from frozen random-draw order")

    input_sha = file_sha256(args.input)
    common_fingerprint: str | None = None
    total_probes = 0
    for row_number, (source, record) in enumerate(zip(source_rows, probe_rows), 1):
        location = f"{args.probes}:{row_number} qid={probe_qids[row_number - 1]!r}"
        if str(record.get("problem") or "") != str(source.get("problem") or ""):
            raise ValueError(f"{location}: problem differs from source")
        if str(record.get("gold_answer") or "") != str(source.get("answer") or ""):
            raise ValueError(f"{location}: gold_answer differs from source")
        if int(record.get("source_index", -1)) != int(source.get("source_index", -1)):
            raise ValueError(f"{location}: source_index differs from source")
        for field in ("sample_rank", "sample_stratum"):
            if str(record.get(field, "")) != str(source.get(field, "")):
                raise ValueError(f"{location}: {field} differs from source")

        config = record.get("generation_config")
        if not isinstance(config, dict):
            raise TypeError(f"{location}: generation_config must be an object")
        if str(config.get("model")) != args.model_id:
            raise ValueError(f"{location}: model identity mismatch")
        source_contract = config.get("source")
        if not isinstance(source_contract, dict):
            raise TypeError(f"{location}: missing generation source contract")
        if source_contract.get("input_sha256") != input_sha:
            raise ValueError(f"{location}: generation input SHA mismatch")
        if int(source_contract.get("row_count", -1)) != args.expected_qids:
            raise ValueError(f"{location}: generation source row count mismatch")
        fingerprint = str(record.get("generation_config_sha256") or "")
        if not fingerprint:
            raise ValueError(f"{location}: empty generation fingerprint")
        if common_fingerprint is None:
            common_fingerprint = fingerprint
        elif fingerprint != common_fingerprint:
            raise ValueError(f"{location}: mixed canonical generation fingerprints")

        max_model_len = nonnegative_int(config.get("max_model_len"), f"{location}.max_model_len")
        max_main_tokens = nonnegative_int(
            (config.get("main_sampling") or {}).get("max_tokens"),
            f"{location}.main_sampling.max_tokens",
        )
        max_probe_tokens = nonnegative_int(
            (config.get("probe") or {}).get("max_tokens"),
            f"{location}.probe.max_tokens",
        )
        main = record.get("main")
        if not isinstance(main, dict):
            raise TypeError(f"{location}: main must be an object")
        prompt_tokens = nonnegative_int(main.get("prompt_tokens"), f"{location}.main.prompt_tokens")
        output_tokens = nonnegative_int(main.get("output_tokens"), f"{location}.main.output_tokens")
        cot_tokens = nonnegative_int(main.get("cot_tokens"), f"{location}.main.cot_tokens")
        cached_tokens = nonnegative_int(main.get("num_cached_tokens"), f"{location}.main.num_cached_tokens")
        if output_tokens > max_main_tokens or prompt_tokens + output_tokens > max_model_len:
            raise ValueError(f"{location}: main token counts exceed generation contract")
        if cot_tokens > output_tokens or cached_tokens > prompt_tokens:
            raise ValueError(f"{location}: inconsistent main token counts")

        probes = record.get("probes")
        if not isinstance(probes, list) or len(probes) != 10:
            raise ValueError(f"{location}: expected exactly ten probes")
        if [probe.get("probe_index") for probe in probes] != list(range(1, 11)):
            raise ValueError(f"{location}: probe indices are not 1..10")
        previous_step = 0
        for probe in probes:
            probe_index = int(probe["probe_index"])
            probe_location = f"{location}.probe[{probe_index}]"
            step_tokens = nonnegative_int(probe.get("step_tokens"), f"{probe_location}.step_tokens")
            probe_prompt = nonnegative_int(
                probe.get("probe_prompt_tokens"), f"{probe_location}.probe_prompt_tokens"
            )
            probe_output = nonnegative_int(
                probe.get("probe_output_tokens"), f"{probe_location}.probe_output_tokens"
            )
            probe_cached = nonnegative_int(
                probe.get("num_cached_tokens"), f"{probe_location}.num_cached_tokens"
            )
            probe_forked = nonnegative_int(
                probe.get("num_direct_forked_tokens"),
                f"{probe_location}.num_direct_forked_tokens",
            )
            if step_tokens < previous_step or step_tokens > max(cot_tokens, 1):
                raise ValueError(f"{probe_location}: invalid CoT cut position")
            if probe_output > max_probe_tokens or probe_prompt + probe_output > max_model_len:
                raise ValueError(f"{probe_location}: token counts exceed generation contract")
            if probe_cached > probe_prompt or probe_forked > probe_prompt:
                raise ValueError(f"{probe_location}: cached/forked tokens exceed prompt")
            finite_features(probe.get("features"), probe_location)
            previous_step = step_tokens
            total_probes += 1

    return {
        "schema": "qwen30_random10k_source_binding_v1",
        "input": str(args.input),
        "input_sha256": input_sha,
        "probes": str(args.probes),
        "probe_qids": len(probe_rows),
        "probe_rows": total_probes,
        "generation_config_sha256": common_fingerprint,
        "model_id": args.model_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--expected-qids", type=int, default=10_000)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
