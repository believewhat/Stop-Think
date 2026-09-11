"""Policy-aligned open-math features inspired by the MedQA tracker.

MedQA can accumulate evidence for four fixed choices.  Open math has no fixed
candidate set, so this module treats each normalized boxed answer as a stable
candidate cluster and accumulates confidence-weighted votes for those clusters.
The tracker is causal: a row at probe ``t`` only uses probes ``<= t``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FEATURES = (
    "mean_logprob",
    "min_logprob",
    "var_logprob",
    "ans_len",
    "seq_logprob_per_sqrt_len",
    "top1_share",
    "top2_share",
    "vote_margin",
    "vote_entropy",
    "current_share",
    "current_is_top",
    "current_rank",
    "run_len",
    "flips",
    "new_cluster",
    "n_clusters",
    "agree_last3",
    "agree_last5",
    "delta_margin",
    "slope_margin",
    "curv_margin2",
    "log1p_probe_index",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _top1_logprob(step: Any) -> float | None:
    """Extract the highest finite logprob from one normalized/raw vLLM step."""
    values: list[float] = []
    if isinstance(step, Mapping):
        iterable = step.values()
    elif isinstance(step, Sequence) and not isinstance(step, (str, bytes)):
        iterable = step
    else:
        iterable = ()
    for item in iterable:
        if isinstance(item, Mapping):
            value = item.get("logprob")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            value = item[1]
        else:
            value = getattr(item, "logprob", None)
        number = _finite(value, default=float("nan"))
        if math.isfinite(number):
            values.append(number)
    return max(values) if values else None


def answer_logprob_stats(steps: Sequence[Any]) -> dict[str, float]:
    """Return length-normalized confidence statistics for a decoded answer."""
    logprobs = [value for step in steps for value in [_top1_logprob(step)] if value is not None]
    if not logprobs:
        return {
            "mean_logprob": 0.0,
            "min_logprob": 0.0,
            "var_logprob": 0.0,
            "ans_len": 0.0,
            "seq_logprob_per_sqrt_len": 0.0,
        }
    array = np.asarray(logprobs, dtype=float)
    length = int(array.size)
    return {
        "mean_logprob": float(array.mean()),
        "min_logprob": float(array.min()),
        "var_logprob": float(array.var()),
        "ans_len": float(length),
        "seq_logprob_per_sqrt_len": float(array.sum() / math.sqrt(length)),
    }


def _second_derivative(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    y = np.asarray(values, dtype=float)
    x = np.arange(y.size, dtype=float)
    return float(2.0 * np.polyfit(x, y, deg=2)[0])


class ClusterEvidenceTracker:
    """Causal confidence-weighted voting over normalized answer strings."""

    def __init__(self, *, recent_window: int = 5, delta_window: int = 3):
        if recent_window < 3 or delta_window < 2:
            raise ValueError("recent_window>=3 and delta_window>=2 are required")
        self.recent_window = int(recent_window)
        self.delta_window = int(delta_window)
        self.support: defaultdict[str, float] = defaultdict(float)
        self.keys: list[str | None] = []
        self.margin_history: list[float] = []
        self.previous_key: str | None = None
        self.run_len = 0
        self.flips = 0

    def update(
        self,
        *,
        answer_key: str | None,
        logprob_stats: Mapping[str, Any],
        probe_index: int,
    ) -> dict[str, float]:
        if probe_index != len(self.keys) + 1:
            raise ValueError(
                f"probe_index must be consecutive: expected={len(self.keys)+1} actual={probe_index}"
            )
        key = str(answer_key).strip() if answer_key is not None else ""
        key = key or None
        is_new = bool(key is not None and key not in self.support)

        if key is None:
            self.run_len = 0
        elif self.previous_key is None:
            self.run_len = 1
        elif key == self.previous_key:
            self.run_len += 1
        else:
            self.flips += 1
            self.run_len = 1
        if key is not None:
            self.previous_key = key

        mean_logprob = _finite(logprob_stats.get("mean_logprob"), 0.0)
        vote_weight = math.exp(min(0.0, max(-50.0, mean_logprob))) if key is not None else 0.0
        if key is not None:
            self.support[key] += vote_weight
        self.keys.append(key)

        ordered = sorted(self.support.items(), key=lambda item: (-item[1], item[0]))
        total = float(sum(value for _, value in ordered))
        top1_value = ordered[0][1] if ordered else 0.0
        top2_value = ordered[1][1] if len(ordered) > 1 else 0.0
        top1_key = ordered[0][0] if ordered else None
        top1_share = top1_value / total if total else 0.0
        top2_share = top2_value / total if total else 0.0
        margin = top1_share - top2_share

        probabilities = np.asarray(
            [value / total for _, value in ordered], dtype=float
        ) if total else np.asarray([], dtype=float)
        if len(probabilities) <= 1:
            entropy = 0.0
        else:
            entropy = float(
                -(probabilities * np.log(probabilities + 1e-12)).sum()
                / math.log(len(probabilities))
            )

        if key is None or not total:
            current_share = 0.0
            current_rank = float(len(ordered) + 1)
            current_is_top = 0.0
        else:
            current_share = self.support[key] / total
            current_rank = float(next(i for i, (candidate, _) in enumerate(ordered, 1) if candidate == key))
            current_is_top = float(key == top1_key)

        def agreement(window: int) -> float:
            if key is None:
                return 0.0
            recent = self.keys[-window:]
            return float(sum(item == key for item in recent) / len(recent))

        self.margin_history.append(float(margin))
        if len(self.margin_history) >= 2:
            start = max(0, len(self.margin_history) - self.delta_window)
            delta_margin = margin - self.margin_history[start]
            slope_margin = (
                (margin - self.margin_history[0])
                / max(1, len(self.margin_history) - 1)
            )
        else:
            delta_margin = 0.0
            slope_margin = 0.0
        curvature = _second_derivative(self.margin_history[-self.recent_window :])

        result = {
            "mean_logprob": mean_logprob,
            "min_logprob": _finite(logprob_stats.get("min_logprob"), 0.0),
            "var_logprob": _finite(logprob_stats.get("var_logprob"), 0.0),
            "ans_len": _finite(logprob_stats.get("ans_len"), 0.0),
            "seq_logprob_per_sqrt_len": _finite(
                logprob_stats.get("seq_logprob_per_sqrt_len"), 0.0
            ),
            "top1_share": float(top1_share),
            "top2_share": float(top2_share),
            "vote_margin": float(margin),
            "vote_entropy": float(entropy),
            "current_share": float(current_share),
            "current_is_top": float(current_is_top),
            "current_rank": float(current_rank),
            "run_len": float(self.run_len),
            "flips": float(self.flips),
            "new_cluster": float(is_new),
            "n_clusters": float(len(ordered)),
            "agree_last3": agreement(3),
            "agree_last5": agreement(5),
            "delta_margin": float(delta_margin),
            "slope_margin": float(slope_margin),
            "curv_margin2": float(curvature),
            "log1p_probe_index": float(math.log1p(probe_index)),
        }
        missing = [name for name in FEATURES if name not in result]
        vector = np.asarray([result[name] for name in FEATURES], dtype=float)
        if missing or vector.shape != (len(FEATURES),) or not np.isfinite(vector).all():
            raise ValueError(f"invalid cluster feature vector; missing={missing}")
        return result
