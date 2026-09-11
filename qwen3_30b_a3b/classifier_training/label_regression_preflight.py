#!/usr/bin/env python3
"""Emit deterministic every-50 labels for an old/new trainer regression.

The relaunch script executes this helper in two separate Python processes with
different ``--trainer-root`` values.  That keeps modules with the same name
isolated and makes byte-for-byte output comparison meaningful.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("serial", "parallel"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    trainer_root = args.trainer_root.resolve()
    sys.path.insert(0, str(trainer_root))
    trainer = importlib.import_module("train_every50_classifier")
    observed_path = Path(trainer.__file__).resolve()
    expected_path = trainer_root / "train_every50_classifier.py"
    if observed_path != expected_path:
        raise RuntimeError(
            f"trainer import isolation failed: observed={observed_path} expected={expected_path}"
        )

    if args.mode == "serial":
        equivalence = trainer.MathEquivalenceCache(timeout_seconds=0.25)
    else:
        if args.cache is None:
            raise ValueError("parallel mode requires --cache")
        equivalence = trainer.ParallelMathEquivalenceCache(
            cache_path=args.cache,
            timeout_seconds=0.25,
            workers=args.workers,
            checkpoint_every=16,
            hard_timeout_grace_seconds=0.75,
            worker_retries=1,
            start_method="fork",
        )

    try:
        steps, qids, _metadata = trainer.load_records(
            [args.input.resolve()], equivalence=equivalence
        )
        qid_rows = [
            {
                "qid": str(row.qid),
                "full_correct_gold": int(row.full_correct_gold),
            }
            for row in qids.sort_values("qid").itertuples(index=False)
        ]
        step_rows = [
            {
                "qid": str(row.qid),
                "probe_index": int(row.probe_index),
                "label_final_consistency": int(row.label_final_consistency),
                "label_gold_safe": int(row.label_gold_safe),
            }
            for row in steps.sort_values(["qid", "probe_index"]).itertuples(
                index=False
            )
        ]
        payload = {"qids": qid_rows, "probes": step_rows}
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
    finally:
        close = getattr(equivalence, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
