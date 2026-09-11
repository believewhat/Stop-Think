#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent


def load_runner():
    math_verify = types.ModuleType("math_verify")
    math_verify.parse = lambda value: [value]
    math_verify.verify = lambda left, right: left == right
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    vllm = types.ModuleType("vllm")
    vllm.LLM = object
    vllm.SamplingParams = object
    with mock.patch.dict(
        sys.modules,
        {"math_verify": math_verify, "transformers": transformers, "vllm": vllm},
    ):
        spec = importlib.util.spec_from_file_location(
            "aime06_runner", ROOT / "run_aime06_baseline_shard.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module


def load_summarizer():
    spec = importlib.util.spec_from_file_location(
        "aime06_summary", ROOT / "summarize_aime06_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_strict_answer_suffix(self) -> None:
        accepted = self.runner.derive_answer(
            "scratch \\boxed{wrong}</think> final \\boxed{17}", "stop"
        )
        self.assertTrue(accepted["answer_valid"])
        self.assertEqual(accepted["prediction"], "17")
        nested = self.runner.derive_answer("x</think>\\boxed{1+{2}}", "stop")
        self.assertEqual(nested["prediction"], "1+{2}")
        self.assertFalse(
            self.runner.derive_answer("\\boxed{17} but no close tag", "stop")[
                "answer_valid"
            ]
        )
        self.assertFalse(
            self.runner.derive_answer("x</think>\\boxed{17}", "length")[
                "answer_valid"
            ]
        )

    def test_load_exactly_30_unique_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aime.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {"unique_id": f"aime-{index}", "problem": "p", "answer": "1"}
                    )
                    + "\n"
                    for index in range(30)
                ),
                encoding="utf-8",
            )
            rows = self.runner.load_rows(path)
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row["eval_dataset"] == "aime2024" for row in rows))

    def test_resume_rejects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "run_fingerprint": "old",
                        "eval_dataset": "aime2024",
                        "qid": "q1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                self.runner.existing_qids(path, "new")


class SummarizerTests(unittest.TestCase):
    def test_three_shards_merge_to_30_unique_rows(self) -> None:
        summarizer = load_summarizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_paths = [root / f"shard-{index}.jsonl" for index in range(3)]
            base_config = {
                "model": "/models/Qwen3-30B-A3B",
                "prompt_version": "qwen3_system_user_explicit_think_v1",
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.2,
                "max_model_len": 40960,
                "max_tokens": 32768,
                "seed": 123,
                "shard_count": 3,
            }
            for shard_index, path in enumerate(shard_paths):
                config = {**base_config, "shard_index": shard_index}
                shard_rows = []
                for qid_index in range(shard_index, 30, 3):
                    shard_rows.append(
                        {
                            "run_fingerprint": f"fingerprint-{shard_index}",
                            "run_config": config,
                            "eval_dataset": "aime2024",
                            "qid": f"aime-{qid_index:02d}",
                            "is_correct": qid_index % 2 == 0,
                            "answer_valid": True,
                            "close_think_found": True,
                            "box_closed": True,
                            "finish_reason": "stop",
                            "response_tokens": 100 + qid_index,
                        }
                    )
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in shard_rows),
                    encoding="utf-8",
                )

            summary_path = root / "summary.json"
            merged_path = root / "merged.jsonl"
            argv = ["summarize_aime06_baseline.py"]
            for path in shard_paths:
                argv.extend(["--shard", str(path)])
            argv.extend(["--output", str(summary_path), "--merged", str(merged_path)])
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                summarizer.main()

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["aime2024"]["rows"], 30)
            self.assertEqual(summary["aime2024"]["correct"], 15)
            self.assertEqual(summary["sampling"]["temperature"], 0.6)
            self.assertEqual(
                len(merged_path.read_text(encoding="utf-8").splitlines()), 30
            )


if __name__ == "__main__":
    unittest.main()
