"""Shared causal A70 feature contract for online and offline ESTAR.

This module is intentionally dependency-light and mirrors the frozen
``medqa_train_triple_t07_classifier_v1`` A70 column order.  The online path
uses it for the same two constrained boundary probes that the offline
collector records: legacy ``<final_answer>`` + bare A-D and canonical
``<final_answer`` + ``>A/>B/>C/>D``.  No future trajectory length is used.
"""

from __future__ import annotations

import math
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np


CHOICES = ("A", "B", "C", "D")
REASONING_BUDGET = 5_000

CAUSAL30_FEATURES = (
    "cum_A", "cum_B", "cum_C", "cum_D", "cum_margin", "run_len", "flips",
    "delta_recent", "slope_recent", "prefix_rate", "inst_sA", "inst_sB",
    "inst_sC", "inst_sD", "inst_pA", "inst_pB", "inst_pC", "inst_pD",
    "curv_margin2", "curv_cum_A2", "curv_cum_B2", "curv_cum_C2",
    "curv_cum_D2", "path_velocity_norm", "path_acceleration_norm",
    "path_curvature", "fisher_trace", "fisher_lmax", "fisher_offdiag_fro",
    "fisher_entropy",
)

A70_FEATURES = (
    *CAUSAL30_FEATURES,
    "position_tokens", "log1p_position_tokens", "interval_tokens",
    "log1p_interval_tokens", "interval_prefix_rate", "probe_index",
    "log1p_probe_index", "instant_top1_confidence", "instant_top12_margin",
    "instant_entropy_normalized", "delta_top1_confidence",
    "delta_top12_margin", "js_current_previous", "js_current_recent_mean",
    "boundary_answer_agree", "legacy_top1_confidence", "legacy_top12_margin",
    "legacy_entropy_normalized", "legacy_support_canonical",
    "canonical_support_legacy", "boundary_js", "boundary_tv",
    "consensus_min_support", "consensus_geomean_support",
    "average_ensemble_margin", "geometric_ensemble_margin",
    "average_ensemble_agrees_canonical", "geometric_ensemble_agrees_canonical",
    "legacy_p_A", "legacy_p_B", "legacy_p_C", "legacy_p_D",
    "average_ensemble_p_A", "average_ensemble_p_B", "average_ensemble_p_C",
    "average_ensemble_p_D", "geometric_ensemble_p_A", "geometric_ensemble_p_B",
    "geometric_ensemble_p_C", "geometric_ensemble_p_D",
)
A70_FEATURE_SCHEMA_VERSION = "medqa_train_triple_t07_A70_v1"
A70_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "schema_version": A70_FEATURE_SCHEMA_VERSION,
            "feature_names": list(A70_FEATURES),
            "feature_count": len(A70_FEATURES),
            "reasoning_budget": REASONING_BUDGET,
            "boundary_contract": "legacy_bare_AD_vs_canonical_gt_AD",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _softmax(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("A70 scores must be four finite values")
    values = values - float(values.max())
    probabilities = np.exp(values)
    probabilities /= float(probabilities.sum())
    return probabilities


def _top2(values: np.ndarray) -> tuple[int, int]:
    order = np.argsort(-values, kind="stable")
    return int(order[0]), int(order[1])


def _entropy(values: np.ndarray) -> float:
    return float(-np.sum(values * np.log(np.clip(values, 1e-300, None))) / math.log(4.0))


def _js(left: np.ndarray, right: np.ndarray) -> float:
    middle = (left + right) * 0.5
    return float(
        0.5 * np.sum(left * np.log(np.clip(left / middle, 1e-300, None)))
        + 0.5 * np.sum(right * np.log(np.clip(right / middle, 1e-300, None)))
    )


def _geometric_pool(distributions: Sequence[np.ndarray]) -> np.ndarray:
    values = np.exp(np.mean(np.stack([np.log(np.clip(p, 1e-300, None)) for p in distributions]), axis=0))
    return values / float(values.sum())


def boundary_features(canonical_scores: Sequence[float], legacy_scores: Sequence[float]) -> np.ndarray:
    """Return the frozen 26 boundary features in exact A70 order."""

    canonical = _softmax(canonical_scores)
    legacy = _softmax(legacy_scores)
    ci, _ = _top2(canonical)
    li, lr = _top2(legacy)
    average = (canonical + legacy) * 0.5
    geometric = _geometric_pool((canonical, legacy))
    ai, ar = _top2(average)
    gi, gr = _top2(geometric)
    values = [
        float(ci == li),
        float(legacy[li]),
        float(legacy[li] - legacy[lr]),
        _entropy(legacy),
        float(legacy[ci]),
        float(canonical[li]),
        _js(canonical, legacy),
        float(np.abs(canonical - legacy).sum() * 0.5),
        float(min(canonical[ci], legacy[ci])),
        float(math.sqrt(canonical[ci] * legacy[ci])),
        float(average[ai] - average[ar]),
        float(geometric[gi] - geometric[gr]),
        float(ai == ci),
        float(gi == ci),
        *legacy.tolist(),
        *average.tolist(),
        *geometric.tolist(),
    ]
    if len(values) != 26 or not np.isfinite(values).all():
        raise ValueError("invalid A70 boundary feature vector")
    return np.asarray(values, dtype=np.float32)


class A70Featurizer:
    """Stateful exact A70 featurizer using only current/past probes."""

    feature_names = A70_FEATURES

    def __init__(self, *, reasoning_budget: int = REASONING_BUDGET, recent_window: int = 5):
        if int(reasoning_budget) <= 0 or int(recent_window) <= 0:
            raise ValueError("reasoning_budget and recent_window must be positive")
        self.reasoning_budget = int(reasoning_budget)
        self.recent_window = int(recent_window)
        # The frozen A70 state is Base30 (five-probe history) plus the
        # MinimalCausalFeaturizer's JS augmentation (three-probe history).
        self.minimal_recent_window = 3
        self.cumulative = np.zeros(4, dtype=np.float64)
        self.previous_winner: int | None = None
        self.run_length = 0
        self.flips = 0
        self.margin_history: list[float] = []
        self.cumulative_history: list[np.ndarray] = []
        self.probability_history: list[np.ndarray] = []
        self.previous_top_confidence: float | None = None
        self.previous_top_margin: float | None = None

    @staticmethod
    def _second(values: Sequence[float]) -> float:
        if len(values) < 3:
            return 0.0
        y = np.asarray(values, dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        return float(2.0 * np.polyfit(x, y, deg=2)[0])

    @staticmethod
    def _path(history: Sequence[np.ndarray]) -> tuple[float, float, float]:
        if len(history) < 3:
            return 0.0, 0.0, 0.0
        newest, previous, older = history[-1], history[-2], history[-3]
        velocity = newest - previous
        acceleration = newest - 2.0 * previous + older
        denom = float(np.dot(velocity, velocity)) + 1e-9
        parallel = float(np.dot(acceleration, velocity)) / denom * velocity
        return float(np.linalg.norm(velocity)), float(np.linalg.norm(acceleration)), float(np.linalg.norm(acceleration - parallel) / denom)

    @staticmethod
    def _fisher(history: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
        if not history:
            return 0.0, 0.0, 0.0, 0.0
        fisher = np.zeros((4, 4), dtype=np.float64)
        entropy = 0.0
        for p in history:
            fisher += np.diag(p) - np.outer(p, p)
            entropy += _entropy(p) * math.log(4.0)
        fisher /= len(history)
        eig = np.linalg.eigvalsh((fisher + fisher.T) * 0.5)
        offdiag = fisher.copy()
        np.fill_diagonal(offdiag, 0.0)
        return float(np.trace(fisher)), float(np.max(eig)), float(np.linalg.norm(offdiag, ord="fro")), float(entropy / len(history))

    def step(
        self,
        *,
        position_tokens: int,
        interval_tokens: int,
        probe_index: int,
        canonical_scores: Sequence[float],
        legacy_scores: Sequence[float],
    ) -> np.ndarray:
        position = int(position_tokens)
        interval = int(interval_tokens)
        index = int(probe_index)
        # Offline formal rows are bounded by the 5k generation budget.  The
        # online DAPO controller may reserve a larger response tensor; those
        # late positions remain valid causal observations and deliberately
        # receive a prefix_rate > 1 rather than crashing the rollout.
        if position < 0 or interval < 0 or index < 1:
            raise ValueError("invalid causal A70 position/interval/index")
        canonical_raw = np.asarray(canonical_scores, dtype=np.float64)
        probabilities = _softmax(canonical_raw)
        self.cumulative += np.log(np.clip(probabilities, 1e-12, 1.0))
        order = np.argsort(self.cumulative)[::-1]
        winner, runner = int(order[0]), int(order[1])
        margin = float(self.cumulative[winner] - self.cumulative[runner])
        if self.previous_winner is None or winner != self.previous_winner:
            if self.previous_winner is not None:
                self.flips += 1
            self.previous_winner = winner
            self.run_length = 1
        else:
            self.run_length += 1
        previous_margins = list(self.margin_history)
        self.margin_history.append(margin)
        self.cumulative_history.append(self.cumulative.copy())
        self.probability_history.append(probabilities.copy())
        self.margin_history = self.margin_history[-self.recent_window:]
        self.cumulative_history = self.cumulative_history[-self.recent_window:]
        self.probability_history = self.probability_history[-self.recent_window:]
        if previous_margins:
            anchor = previous_margins[max(0, len(previous_margins) - self.recent_window)]
            delta_recent = margin - anchor
            slope_recent = (margin - previous_margins[0]) / max(1, len(previous_margins))
        else:
            delta_recent = slope_recent = 0.0
        top, second = _top2(probabilities)
        confidence = float(probabilities[top])
        instant_margin = float(probabilities[top] - probabilities[second])
        previous_probability = self.probability_history[-2] if len(self.probability_history) >= 2 else None
        recent = self.probability_history[:-1]
        recent_mean = (
            np.mean(np.stack(recent[-self.minimal_recent_window:]), axis=0)
            if recent else None
        )
        delta_confidence = 0.0 if self.previous_top_confidence is None else confidence - self.previous_top_confidence
        delta_margin = 0.0 if self.previous_top_margin is None else instant_margin - self.previous_top_margin
        self.previous_top_confidence = confidence
        self.previous_top_margin = instant_margin
        velocity, acceleration, path_curvature = self._path(self.cumulative_history)
        fisher = self._fisher(self.probability_history)
        base = [
            *self.cumulative.tolist(), margin, float(self.run_length), float(self.flips),
            delta_recent, slope_recent, position / float(self.reasoning_budget),
            *canonical_raw.tolist(), *probabilities.tolist(),
            self._second(self.margin_history),
            *[self._second([row[i] for row in self.cumulative_history]) for i in range(4)],
            velocity, acceleration, path_curvature, *fisher,
        ]
        appended = [
            float(position), math.log1p(position), float(interval), math.log1p(interval),
            interval / max(1.0, float(position)), float(index), math.log1p(index),
            confidence, instant_margin, _entropy(probabilities), delta_confidence,
            delta_margin,
            0.0 if previous_probability is None else _js(probabilities, previous_probability),
            0.0 if recent_mean is None else _js(probabilities, recent_mean),
        ]
        result = np.concatenate((np.asarray(base, dtype=np.float32), np.asarray(appended, dtype=np.float32), boundary_features(canonical_raw, legacy_scores)))
        if result.size != 70 or not np.isfinite(result).all():
            raise ValueError(f"A70 feature contract violated: size={result.size}")
        return result.astype(np.float32, copy=False)


def assert_a70_schema(feature_names: Sequence[str]) -> None:
    if tuple(feature_names) != A70_FEATURES:
        raise ValueError("A70 feature schema mismatch")
