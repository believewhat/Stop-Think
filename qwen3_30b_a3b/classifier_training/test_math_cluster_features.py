from __future__ import annotations

import math

import numpy as np

from math_cluster_features import (
    FEATURES,
    ClusterEvidenceTracker,
    answer_logprob_stats,
)


def step(*values: float):
    return [(f"tok{i}", value) for i, value in enumerate(values)]


def test_answer_logprob_stats_use_top1_and_length_normalize() -> None:
    result = answer_logprob_stats([step(-0.2, -2.0), step(-0.4, -1.0)])
    assert result["ans_len"] == 2
    assert np.isclose(result["mean_logprob"], -0.3)
    assert np.isclose(result["min_logprob"], -0.4)
    assert np.isclose(result["seq_logprob_per_sqrt_len"], -0.6 / math.sqrt(2))


def test_repeated_candidate_accumulates_medqa_style_support() -> None:
    tracker = ClusterEvidenceTracker()
    stats = {
        "mean_logprob": -0.1,
        "min_logprob": -0.2,
        "var_logprob": 0.01,
        "ans_len": 2,
        "seq_logprob_per_sqrt_len": -0.2 / math.sqrt(2),
    }
    first = tracker.update(answer_key="1/2", logprob_stats=stats, probe_index=1)
    second = tracker.update(answer_key="1/2", logprob_stats=stats, probe_index=2)
    third = tracker.update(answer_key="2/3", logprob_stats=stats, probe_index=3)
    assert first["new_cluster"] == 1
    assert second["run_len"] == 2
    assert second["top1_share"] == 1
    assert third["new_cluster"] == 1
    assert third["flips"] == 1
    assert third["n_clusters"] == 2
    assert third["current_rank"] == 2
    assert third["top1_share"] > third["top2_share"]


def test_invalid_answer_updates_time_without_creating_cluster() -> None:
    tracker = ClusterEvidenceTracker()
    zeros = answer_logprob_stats([])
    result = tracker.update(answer_key=None, logprob_stats=zeros, probe_index=1)
    assert result["n_clusters"] == 0
    assert result["run_len"] == 0
    assert result["current_is_top"] == 0
    assert tuple(result) == FEATURES


def test_probe_indices_must_be_causal_and_consecutive() -> None:
    tracker = ClusterEvidenceTracker()
    try:
        tracker.update(answer_key="1", logprob_stats=answer_logprob_stats([]), probe_index=2)
    except ValueError as error:
        assert "consecutive" in str(error)
    else:
        raise AssertionError("non-consecutive probe index was accepted")
