from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_math500_source")


class FakeArray:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.values[item[0]][item[1]]
        return self.values[item]

    def all(self):
        return all(all(row) if isinstance(row, list) else row for row in self.values)


fake_numpy = types.ModuleType("numpy")
fake_numpy.asarray = lambda values, dtype=None: FakeArray([[float(x) for x in row] for row in values])
fake_numpy.isfinite = lambda array: FakeArray([[True for _ in row] for row in array.values])
fake_joblib = types.ModuleType("joblib")
fake_joblib.load = lambda _path: None
fake_math_verify = types.ModuleType("math_verify")
fake_math_verify.parse = lambda value: value
fake_math_verify.verify = lambda left, right: left == right
sys.modules.setdefault("numpy", fake_numpy)
sys.modules.setdefault("joblib", fake_joblib)
sys.modules.setdefault("math_verify", fake_math_verify)
evaluate = load("evaluate_dual_classifiers")


def baseline_row(repeat_id: str, index: int) -> dict:
    config = {
        "model": "/models/Qwen3-30B-A3B",
        "model_config_sha256": "a" * 64,
        "prompt_version": "qwen3_system_user_explicit_think_v1",
        "system_prompt": prepare.SYSTEM_PROMPT,
        "enable_thinking": True,
        "explicit_think_prefix": True,
        "seed": prepare.EXPECTED_SEEDS[repeat_id],
        **prepare.EXPECTED_SAMPLING,
        "shard_index": index % 2,
        "shard_count": 2,
    }
    return {
        "format_version": 1,
        "run_config": config,
        "run_fingerprint": hashlib.sha256(prepare.canonical_json(config).encode()).hexdigest(),
        "eval_dataset": "math500",
        "qid": f"math/{index}",
        "problem": f"Compute {index}",
        "gold_answer": str(index),
        "prediction": str(index),
        "is_correct": True,
        "answer_valid": True,
        "response_tokens": 100,
        "response_text": f"reasoning {index}</think>\\boxed{{{index}}}",
    }


def write_repeat(path: Path, repeat_id: str, mutate=None) -> None:
    rows = [baseline_row(repeat_id, index) for index in range(500)]
    if mutate:
        mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_exact_three_repeat_source(tmp_path: Path) -> None:
    inputs = {}
    for repeat_id in prepare.EXPECTED_SEEDS:
        inputs[repeat_id] = tmp_path / f"{repeat_id}.jsonl"
        write_repeat(inputs[repeat_id], repeat_id)
    repeats, _ = zip(*(prepare.read_repeat(inputs[r], r) for r in prepare.EXPECTED_SEEDS))
    source = prepare.build_source(dict(zip(prepare.EXPECTED_SEEDS, repeats)))
    assert len(source) == 1500
    assert source[0]["qid"] == "repeat1::math/0"
    assert source[500]["qid"] == "repeat2::math/0"
    assert source[-1]["main"]["repeat_id"] == "repeat3"


def test_prepare_rejects_sampling_drift(tmp_path: Path) -> None:
    path = tmp_path / "repeat1.jsonl"
    def mutate(rows):
        rows[0]["run_config"]["temperature"] = 0.6
        rows[0]["run_fingerprint"] = hashlib.sha256(prepare.canonical_json(rows[0]["run_config"]).encode()).hexdigest()
    write_repeat(path, "repeat1", mutate)
    with pytest.raises(ValueError, match="prompt/sampling drift"):
        prepare.read_repeat(path, "repeat1")


class DummyModel:
    classes_ = [0, 1]
    n_features_in_ = 22

    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)

    def predict_proba(self, _vector):
        positive = next(self.probabilities)
        return [[1.0 - positive, positive]]


def probe(index: int, *, eligible=True) -> dict:
    return {
        "probe_index": index,
        "step_tokens": index * 50,
        "probe_answer": "7",
        "eligible": eligible,
        "features": {name: float(index) for name in evaluate.EXPECTED_FEATURES},
    }


def test_policy_uses_earliest_hit_and_think_only(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "safe_grade", lambda prediction, gold: prediction == gold)
    row = {
        "qid": "repeat1::q",
        "gold_answer": "7",
        "main": {"prediction": "8"},
        "replay": {"reasoning_tokens": 999},
        "probes": [probe(1), probe(2), probe(3)],
    }
    result = evaluate.select_result(row, DummyModel([0.2, 0.95, 0.99]), list(evaluate.EXPECTED_FEATURES), 1)
    assert result["stopped"] is True
    assert result["stop_step_tokens"] == 100
    assert result["primary_think_tokens"] == 100


def test_policy_no_hit_falls_back_to_full_baseline(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "safe_grade", lambda prediction, gold: prediction == gold)
    row = {"qid": "repeat1::q", "gold_answer": "7", "main": {"prediction": "7"}, "replay": {"reasoning_tokens": 999}, "probes": [probe(1)]}
    result = evaluate.select_result(row, DummyModel([0.949]), list(evaluate.EXPECTED_FEATURES), 1)
    assert result["stopped"] is False
    assert result["prediction"] == "7"
    assert result["primary_think_tokens"] == 999


def test_run_level_ci_is_t_interval() -> None:
    result = evaluate.ci([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_sd"] == 1.0
    assert result["ci95_low"] < 0 < result["ci95_high"]
