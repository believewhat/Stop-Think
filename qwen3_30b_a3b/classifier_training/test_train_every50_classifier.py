from __future__ import annotations

import json
import os
import sqlite3
import time

import pandas as pd
import pytest

from math_cluster_features import FEATURES
from math_equivalence_frozen import grade_answer_sympy, mathd_normalize_answer
from train_every50_classifier import (
    MathEquivalenceCache,
    ParallelMathEquivalenceCache,
    config_sha256,
    evaluate_policy,
    load_records,
    parallel_verify_pairs,
    stable_qid_split,
)


def parallel_test_verifier(
    prediction: str, reference: str, _timeout_seconds: float
) -> bool:
    """Top-level helper so the Windows spawn context can pickle it."""
    if prediction == "hang":
        # Deliberately ignores the supplied timeout.  The parent must kill this
        # process and then successfully reuse the slot for the following pair.
        time.sleep(10.0)
    if prediction == "late":
        time.sleep(0.10)
        return True
    if prediction == "explode":
        raise KeyboardInterrupt("unexpected worker failure")
    return prediction.removeprefix("equal:") == reference


def pre_ready_exit_worker(_slot, connection, _verifier, _timeout) -> None:
    connection.close()
    os._exit(17)


def pre_ready_hang_worker(_slot, _connection, _verifier, _timeout) -> None:
    time.sleep(10.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.5", r"\frac{1}{2}"),
        ("1/2", r"\frac{1}{2}"),
        (r"\left\frac12\right", r"\frac{1}{2}"),
        (r"\text{ seven }", "seven"),
        ("x = 7", "7"),
    ],
)
def test_frozen_mathd_normalizer_historical_parity(raw, expected) -> None:
    assert mathd_normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("0.5", "1/2", True),
        (r"\frac{1}{2}", "0.5", True),
        ("100%", "100", True),
        ("(1,2)", "[1,2]", False),
        ("2", "3", False),
    ],
)
def test_frozen_sympy_grader_historical_parity(
    prediction, reference, expected
) -> None:
    assert grade_answer_sympy(prediction, reference) is expected


def generation_config() -> dict[str, object]:
    return {
        "model": "/models/Qwen3-30B-A3B",
        "probe_stride_tokens": 50,
        "feature_order": list(FEATURES),
        "selection": {"shard_index": 0, "shard_count": 1},
    }


def probe(index: int, *, reverse_features: bool = False) -> dict[str, object]:
    names = list(reversed(FEATURES)) if reverse_features else list(FEATURES)
    return {
        "probe_index": index,
        "step_tokens": 50 * index,
        "probe_answer": "2",
        "answer_key": "2",
        "eligible": True,
        "box_closed": True,
        "features": {name: float(offset) for offset, name in enumerate(names)},
    }


def record(*, probes: list[dict[str, object]]) -> dict[str, object]:
    config = generation_config()
    return {
        "qid": "q1",
        "problem": "1+1?",
        "gold_answer": "2",
        "main": {
            "full_answer": "2",
            "full_answer_key": "2",
            "full_valid": True,
            "close_think_found": True,
            "box_closed": True,
            "finish_reason": "stop",
            "cot_tokens": 170,
        },
        "probes": probes,
        "generation_config": config,
        "generation_config_sha256": config_sha256(config),
    }


def write_record(tmp_path, value: dict[str, object]):
    path = tmp_path / "merged.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def exact_only_cache() -> MathEquivalenceCache:
    return MathEquivalenceCache(verifier=lambda _a, _b, _timeout: False)


def test_loader_accepts_strict_every50_and_ordered_cluster22(tmp_path) -> None:
    path = write_record(tmp_path, record(probes=[probe(1), probe(2), probe(3)]))
    steps, qids, metadata = load_records([path], equivalence=exact_only_cache())
    assert steps.step_tokens.tolist() == [50, 100, 150]
    assert steps.label_final_consistency.tolist() == [1, 1, 1]
    assert steps.label_gold_safe.tolist() == [1, 1, 1]
    assert qids.full_correct_gold.tolist() == [1]
    assert metadata["inputs"][0]["probes"] == 3


def test_loader_rejects_non50_grid(tmp_path) -> None:
    bad = probe(2)
    bad["step_tokens"] = 110
    path = write_record(tmp_path, record(probes=[probe(1), bad]))
    with pytest.raises(ValueError, match="every-50 grid violation"):
        load_records([path], equivalence=exact_only_cache())


def test_loader_rejects_reordered_features(tmp_path) -> None:
    path = write_record(tmp_path, record(probes=[probe(1, reverse_features=True)]))
    with pytest.raises(ValueError, match="feature order mismatch"):
        load_records([path], equivalence=exact_only_cache())


def test_two_pass_loader_uses_parallel_prepared_pair_universe(tmp_path) -> None:
    value = record(probes=[probe(1), probe(2)])
    value["main"]["full_answer_key"] = "equal:2"
    value["probes"][0]["probe_answer"] = "different"
    value["probes"][0]["answer_key"] = "different"
    value["probes"][1]["probe_answer"] = "equal:2"
    value["probes"][1]["answer_key"] = "equal:2"
    path = write_record(tmp_path, value)
    cache = ParallelMathEquivalenceCache(
        cache_path=tmp_path / "loader.sqlite3",
        workers=2,
        checkpoint_every=1,
        timeout_seconds=1.0,
        verifier=parallel_test_verifier,
    )
    steps, qids, metadata = load_records([path], equivalence=cache)
    assert qids.full_correct_gold.tolist() == [1]
    assert steps.label_final_consistency.tolist() == [0, 1]
    assert steps.label_gold_safe.tolist() == [0, 1]
    assert metadata["pair_universe_count"] == 3
    assert cache.computed_this_run == 3
    cache.close()


def test_equivalence_cache_reuses_normalized_pair() -> None:
    calls: list[tuple[str, str]] = []

    def verifier(prediction: str, reference: str, _timeout: float) -> bool:
        calls.append((prediction, reference))
        return False

    cache = MathEquivalenceCache(verifier=verifier)
    assert not cache.equivalent("x", "y", prediction_key="x", reference_key="y")
    assert not cache.equivalent("x again", "y again", prediction_key="x", reference_key="y")
    assert calls == [("x", "y")]
    assert cache.diagnostics()["cache_hits"] == 1


def test_parallel_cache_globally_deduplicates_and_resumes(tmp_path) -> None:
    cache_path = tmp_path / "equivalence.sqlite3"
    pairs = [
        ("equal:2", "2"),
        ("different", "2"),
        ("equal:2", "2"),
        ("same", "same"),
    ]
    cache = ParallelMathEquivalenceCache(
        cache_path=cache_path,
        workers=2,
        checkpoint_every=1,
        timeout_seconds=1.0,
        hard_timeout_grace_seconds=0.5,
        worker_retries=1,
        verifier=parallel_test_verifier,
    )
    cache.prepare_pairs(pairs, universe_metadata={"input_sha256": "fixture-v1"})
    assert cache.values == {("different", "2"): False, ("equal:2", "2"): True}
    assert cache.computed_this_run == 2
    cache.close()

    resumed = ParallelMathEquivalenceCache(
        cache_path=cache_path,
        workers=2,
        checkpoint_every=1,
        timeout_seconds=1.0,
        hard_timeout_grace_seconds=0.5,
        worker_retries=1,
        verifier=parallel_test_verifier,
    )
    resumed.prepare_pairs(
        list(reversed(pairs)), universe_metadata={"input_sha256": "fixture-v1"}
    )
    assert resumed.resumed_pairs == 2
    assert resumed.computed_this_run == 0
    assert resumed.diagnostics()["parallel_run"]["scheduled_pairs"] == 0
    progress = json.loads(resumed.progress_path.read_text(encoding="utf-8"))
    assert progress["complete"] is True
    assert progress["state"] == "complete"
    assert progress["total_unique_nonexact_pairs"] == 2
    resumed.close()


def test_parent_hard_timeout_kills_worker_and_reuses_slot() -> None:
    observed: dict[tuple[str, str], tuple[bool, str]] = {}

    summary = parallel_verify_pairs(
        [("hang", "x"), ("equal:x", "x")],
        verifier=parallel_test_verifier,
        timeout_seconds=0.05,
        workers=1,
        hard_timeout_grace_seconds=0.05,
        worker_retries=1,
        worker_max_tasks=1_000,
        worker_start_timeout_seconds=2.0,
        start_method="auto",
        on_result=lambda pair, value, status, _elapsed: observed.__setitem__(
            pair, (value, status)
        ),
    )
    assert observed[("hang", "x")] == (False, "hard_timeout")
    assert observed[("equal:x", "x")] == (True, "ok")
    assert summary["status_hard_timeout"] == 1
    assert summary["status_ok"] == 1


def test_result_after_semantic_deadline_is_forced_false() -> None:
    observed: dict[tuple[str, str], tuple[bool, str]] = {}
    summary = parallel_verify_pairs(
        [("late", "x")],
        verifier=parallel_test_verifier,
        timeout_seconds=0.05,
        workers=1,
        hard_timeout_grace_seconds=0.50,
        worker_retries=1,
        worker_max_tasks=1_000,
        worker_start_timeout_seconds=2.0,
        start_method="auto",
        on_result=lambda pair, value, status, _elapsed: observed.__setitem__(
            pair, (value, status)
        ),
    )
    assert observed[("late", "x")] == (False, "deadline_exceeded")
    assert summary["status_deadline_exceeded"] == 1


@pytest.mark.parametrize(
    "entrypoint", [pre_ready_exit_worker, pre_ready_hang_worker]
)
def test_pre_ready_failure_retries_then_fails_closed(entrypoint) -> None:
    observed = []
    with pytest.raises(RuntimeError, match="failed before ready"):
        parallel_verify_pairs(
            [("equal:x", "x")],
            verifier=parallel_test_verifier,
            timeout_seconds=0.05,
            workers=1,
            hard_timeout_grace_seconds=0.05,
            worker_retries=1,
            worker_max_tasks=1_000,
            worker_start_timeout_seconds=0.10,
            start_method="auto",
            on_result=lambda *value: observed.append(value),
            worker_entrypoint=entrypoint,
        )
    assert observed == []


def test_unexpected_worker_error_fails_without_caching_false(tmp_path) -> None:
    cache_path = tmp_path / "failure.sqlite3"
    cache = ParallelMathEquivalenceCache(
        cache_path=cache_path,
        workers=1,
        checkpoint_every=1,
        timeout_seconds=1.0,
        hard_timeout_grace_seconds=0.5,
        worker_retries=1,
        verifier=parallel_test_verifier,
    )
    with pytest.raises(RuntimeError, match="Unexpected equivalence worker failure"):
        cache.prepare_pairs(
            [("explode", "x")],
            universe_metadata={"input_sha256": "failure-fixture"},
        )
    with sqlite3.connect(cache_path) as connection:
        cached_rows = connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    assert cached_rows == 0
    progress = json.loads(cache.progress_path.read_text(encoding="utf-8"))
    assert progress["complete"] is False
    assert progress["state"] == "failed"
    assert progress["pending_pairs"] == 1
    cache.close()


def test_resume_fails_closed_on_input_universe_change(tmp_path) -> None:
    cache_path = tmp_path / "bound.sqlite3"
    cache = ParallelMathEquivalenceCache(
        cache_path=cache_path,
        workers=1,
        checkpoint_every=1,
        timeout_seconds=1.0,
        verifier=parallel_test_verifier,
    )
    cache.prepare_pairs(
        [("equal:2", "2")], universe_metadata={"input_sha256": "input-a"}
    )
    cache.close()

    resumed = ParallelMathEquivalenceCache(
        cache_path=cache_path,
        workers=1,
        checkpoint_every=1,
        timeout_seconds=1.0,
        verifier=parallel_test_verifier,
    )
    with pytest.raises(ValueError, match="pair/input universe mismatch"):
        resumed.prepare_pairs(
            [("equal:2", "2")], universe_metadata={"input_sha256": "input-b"}
        )
    resumed.close()


def test_resume_rejects_semantically_corrupt_cache_row(tmp_path) -> None:
    cache_path = tmp_path / "corrupt-status.sqlite3"
    cache = ParallelMathEquivalenceCache(
        cache_path=cache_path,
        workers=1,
        checkpoint_every=1,
        timeout_seconds=1.0,
        verifier=parallel_test_verifier,
    )
    cache.prepare_pairs(
        [("equal:2", "2")], universe_metadata={"input_sha256": "input-a"}
    )
    cache.close()
    with sqlite3.connect(cache_path) as connection:
        connection.execute("UPDATE results SET status = 'infrastructure_failure'")
        connection.commit()
    with pytest.raises(ValueError, match="Invalid equivalence cache status"):
        ParallelMathEquivalenceCache(
            cache_path=cache_path,
            workers=1,
            checkpoint_every=1,
            timeout_seconds=1.0,
            verifier=parallel_test_verifier,
        )


def test_stable_split_is_exact_qid_level_and_repeatable() -> None:
    qids = [f"q{index}" for index in range(20)]
    train1, validation1 = stable_qid_split(qids, seed=42, validation_fraction=0.10)
    train2, validation2 = stable_qid_split(reversed(qids), seed=42, validation_fraction=0.10)
    assert (train1, validation1) == (train2, validation2)
    assert len(train1) == 18
    assert len(validation1) == 2
    assert not train1 & validation1


def test_policy_uses_earliest_hit_and_full_answer_fallback() -> None:
    steps = pd.DataFrame(
        [
            {
                "qid": "q1",
                "probe_index": 1,
                "step_tokens": 50,
                "eligible": 1,
                "prob": 0.80,
                "label_final_consistency": 0,
                "label_gold_safe": 0,
                "probe_answer": "bad",
            },
            {
                "qid": "q1",
                "probe_index": 2,
                "step_tokens": 100,
                "eligible": 1,
                "prob": 0.96,
                "label_final_consistency": 1,
                "label_gold_safe": 1,
                "probe_answer": "good",
            },
            {
                "qid": "q1",
                "probe_index": 3,
                "step_tokens": 150,
                "eligible": 1,
                "prob": 0.99,
                "label_final_consistency": 0,
                "label_gold_safe": 0,
                "probe_answer": "too late",
            },
            {
                "qid": "q2",
                "probe_index": 1,
                "step_tokens": 50,
                "eligible": 1,
                "prob": 0.94,
                "label_final_consistency": 0,
                "label_gold_safe": 0,
                "probe_answer": "no hit",
            },
        ]
    )
    qids = pd.DataFrame(
        [
            {
                "qid": "q1",
                "full_answer": "good",
                "full_valid": 1,
                "full_correct_gold": 1,
                "cot_tokens": 200,
            },
            {
                "qid": "q2",
                "full_answer": "also good",
                "full_valid": 1,
                "full_correct_gold": 1,
                "cot_tokens": 300,
            },
        ]
    )
    metrics, decisions = evaluate_policy(
        steps,
        qids,
        target="gold_safe",
        probability_column="prob",
        threshold=0.95,
    )
    q1 = decisions.set_index("qid").loc["q1"]
    q2 = decisions.set_index("qid").loc["q2"]
    assert q1.probe_index == 2
    assert q1.selected_tokens == 100
    assert q2.stopped == 0
    assert q2.selected_tokens == 300
    assert metrics["coverage"] == 0.5
    assert metrics["policy_accuracy_gold"] == 1.0
