from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prepare_aime", ROOT / "prepare_aime2024_source.py")
prepare = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(prepare)


def row(repeat_id: str, index: int) -> dict:
    config = {
        "model": "/models/Qwen3-30B-A3B", "model_config_sha256": "a" * 64,
        "prompt_version": "qwen3_system_user_explicit_think_v1",
        "system_prompt": prepare.SYSTEM_PROMPT, "enable_thinking": True,
        "explicit_think_prefix": True, "seed": prepare.EXPECTED_SEEDS[repeat_id],
        **prepare.EXPECTED_SAMPLING, "shard_index": index % 2, "shard_count": 2,
    }
    return {"run_config": config, "run_fingerprint": hashlib.sha256(prepare.canonical_json(config).encode()).hexdigest(),
            "eval_dataset": "aime2024", "qid": f"aime2024_{index}", "problem": f"P{index}",
            "gold_answer": str(index), "prediction": str(index), "is_correct": True,
            "answer_valid": True, "response_tokens": 1000, "response_text": "reasoning</think>"}


def write(path: Path, repeat_id: str, temperature: float = 0.6) -> None:
    rows = [row(repeat_id, index) for index in range(30)]
    rows[0]["run_config"]["temperature"] = temperature
    rows[0]["run_fingerprint"] = hashlib.sha256(prepare.canonical_json(rows[0]["run_config"]).encode()).hexdigest()
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_three_repeats_become_90_prefixed_rows(tmp_path: Path) -> None:
    repeats = {}
    for repeat_id in prepare.EXPECTED_SEEDS:
        path = tmp_path / f"{repeat_id}.jsonl"; write(path, repeat_id)
        repeats[repeat_id], _ = prepare.read_repeat(path, repeat_id)
    source = prepare.build_source(repeats)
    assert len(source) == 90
    assert source[0]["qid"] == "repeat1::aime2024_0"
    assert source[30]["qid"] == "repeat2::aime2024_0"
    assert source[-1]["main"]["eval_dataset"] == "aime2024"


def test_temperature_zero_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "repeat1.jsonl"; write(path, "repeat1", 0.0)
    with pytest.raises(ValueError, match="prompt/sampling drift"):
        prepare.read_repeat(path, "repeat1")
