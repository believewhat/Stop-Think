#!/usr/bin/env python3
"""Merge and summarize one arbitrarily sharded AIME2024 baseline repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero rows")
    tokens = [int(row["response_tokens"]) for row in rows]
    correct = sum(bool(row["is_correct"]) for row in rows)
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "answer_valid_rate": sum(bool(row["answer_valid"]) for row in rows) / len(rows),
        "close_think_rate": sum(bool(row["close_think_found"]) for row in rows) / len(rows),
        "box_closed_rate": sum(bool(row["box_closed"]) for row in rows) / len(rows),
        "length_capped": sum(
            row["finish_reason"] in {"length", "max_tokens"} for row in rows
        ),
        "mean_response_tokens": statistics.fmean(tokens),
        "median_response_tokens": statistics.median(tokens),
        "p90_response_tokens": percentile(tokens, 0.90),
        "p95_response_tokens": percentile(tokens, 0.95),
        "max_response_tokens": max(tokens),
    }


def invariant_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("shard_index", None)
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--merged", required=True, type=Path)
    args = parser.parse_args()

    if len({str(path.resolve()) for path in args.shard}) != len(args.shard):
        raise ValueError("Duplicate --shard path")

    rows: list[dict[str, Any]] = []
    shard_metadata: list[dict[str, Any]] = []
    configs_by_index: dict[int, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for path in args.shard:
        shard_rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {
                    "run_fingerprint",
                    "run_config",
                    "eval_dataset",
                    "qid",
                    "is_correct",
                    "answer_valid",
                    "close_think_found",
                    "box_closed",
                    "finish_reason",
                    "response_tokens",
                }
                missing = sorted(required - row.keys())
                if missing:
                    raise ValueError(f"{path}:{line_number} missing fields: {missing}")
                if row["eval_dataset"] != "aime2024":
                    raise ValueError(f"{path}:{line_number} is not an AIME2024 row")
                shard_rows.append(row)
        if not shard_rows:
            raise ValueError(f"Empty shard is not allowed: {path}")

        shard_fingerprints = {str(row["run_fingerprint"]) for row in shard_rows}
        if len(shard_fingerprints) != 1:
            raise ValueError(f"{path} contains multiple run fingerprints")
        fingerprint = next(iter(shard_fingerprints))
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate run fingerprint across shards: {path}")
        fingerprints.add(fingerprint)

        shard_configs = {
            json.dumps(row["run_config"], sort_keys=True, separators=(",", ":"))
            for row in shard_rows
        }
        if len(shard_configs) != 1:
            raise ValueError(f"{path} contains multiple run configs")
        config = dict(shard_rows[0]["run_config"])
        shard_index = int(config["shard_index"])
        if shard_index in configs_by_index:
            raise ValueError(f"Duplicate shard_index={shard_index}")
        configs_by_index[shard_index] = config
        rows.extend(shard_rows)
        shard_metadata.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "shard_index": shard_index,
                "rows": len(shard_rows),
                "run_fingerprint": fingerprint,
            }
        )

    shard_count = len(args.shard)
    if set(configs_by_index) != set(range(shard_count)):
        raise ValueError(
            f"Expected shard indices 0..{shard_count - 1}, got {sorted(configs_by_index)}"
        )
    configs = list(configs_by_index.values())
    if any(int(config.get("shard_count", -1)) != shard_count for config in configs):
        raise ValueError("Declared shard_count does not match number of --shard inputs")
    reference_config = invariant_config(configs[0])
    if any(invariant_config(config) != reference_config for config in configs[1:]):
        raise ValueError("Shard run configs differ beyond shard_index")
    observed_sampling = {
        key: reference_config.get(key) for key in EXPECTED_SAMPLING
    }
    if observed_sampling != EXPECTED_SAMPLING:
        raise ValueError(
            f"Sampling mismatch: expected={EXPECTED_SAMPLING} observed={observed_sampling}"
        )
    if reference_config.get("max_model_len") != 40960:
        raise ValueError("max_model_len must be 40960")
    if reference_config.get("max_tokens") != 32768:
        raise ValueError("max_tokens must be 32768")

    qids = [str(row["qid"]) for row in rows]
    if len(rows) != 30 or len(set(qids)) != 30:
        raise ValueError(
            f"Expected 30 unique AIME2024 rows, got rows={len(rows)} unique={len(set(qids))}"
        )
    rows.sort(key=lambda row: str(row["qid"]))

    merged_text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    atomic_write_text(args.merged, merged_text)
    payload = {
        "experiment": "Qwen3-30B-A3B AIME2024 temperature-0.6 baseline",
        "model": reference_config["model"],
        "prompt_version": reference_config["prompt_version"],
        "seed": int(reference_config["seed"]),
        "max_model_len": 40960,
        "max_tokens": 32768,
        "sampling": EXPECTED_SAMPLING,
        "aime2024": metrics(rows),
        "merged_output": str(args.merged),
        "merged_sha256": sha256(args.merged),
        "shards": sorted(shard_metadata, key=lambda item: item["shard_index"]),
    }
    atomic_write_text(
        args.output,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
