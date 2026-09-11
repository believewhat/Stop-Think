#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


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
    tokens = [int(row["response_tokens"]) for row in rows]
    return {
        "rows": len(rows),
        "correct": sum(bool(row["is_correct"]) for row in rows),
        "accuracy": sum(bool(row["is_correct"]) for row in rows) / len(rows),
        "answer_valid_rate": sum(bool(row["answer_valid"]) for row in rows) / len(rows),
        "close_think_rate": sum(bool(row["close_think_found"]) for row in rows) / len(rows),
        "box_closed_rate": sum(bool(row["box_closed"]) for row in rows) / len(rows),
        "length_capped": sum(row["finish_reason"] in {"length", "max_tokens"} for row in rows),
        "mean_response_tokens": statistics.fmean(tokens),
        "median_response_tokens": statistics.median(tokens),
        "p90_response_tokens": percentile(tokens, 0.90),
        "p95_response_tokens": percentile(tokens, 0.95),
        "max_response_tokens": max(tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--merged", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for path in args.shard:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    fingerprints.add(str(row["run_fingerprint"]))
                    rows.append(row)
    if len(fingerprints) != len(args.shard):
        raise ValueError("Expected one distinct fingerprint per shard")
    qids = [str(row["qid"]) for row in rows]
    if len(rows) != 530 or len(set(qids)) != 530:
        raise ValueError(f"Expected 530 unique rows, got {len(rows)} rows/{len(set(qids))} qids")
    rows.sort(key=lambda row: (row["eval_dataset"], row["qid"]))

    args.merged.parent.mkdir(parents=True, exist_ok=True)
    temporary_merged = args.merged.with_suffix(args.merged.suffix + ".tmp")
    with temporary_merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary_merged, args.merged)

    math_rows = [row for row in rows if row["eval_dataset"] == "math500"]
    aime_rows = [row for row in rows if row["eval_dataset"] == "aime2024"]
    if len(math_rows) != 500 or len(aime_rows) != 30:
        raise ValueError("Dataset split mismatch")
    payload = {
        "experiment": "Qwen3-30B-A3B native-long-context baseline",
        "model": rows[0]["run_config"]["model"],
        "prompt_version": rows[0]["run_config"]["prompt_version"],
        "max_model_len": 40960,
        "max_tokens": 32768,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "repetition_penalty": 1.2},
        "math500": metrics(math_rows),
        "aime2024": metrics(aime_rows),
        "merged_output": str(args.merged),
        "merged_sha256": sha256(args.merged),
        "shards": [{"path": str(path), "sha256": sha256(path)} for path in args.shard],
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
