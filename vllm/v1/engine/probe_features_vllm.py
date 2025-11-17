# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping, Iterable

import numpy as np

# ================================================================
# Token utilities
# ================================================================

_STRIP = set(' \t\n\rĠ▁<>"\'`()[]{}.,;:|/\\!?*-_+=~')

def _first_letter_bucket_from_decoded(tok: Optional[str]) -> int:
    """Map the first non-space-like character to bucket 0..3 for A/B/C/D, else -1."""
    if not tok:
        return -1
    i, L = 0, len(tok)
    while i < L and tok[i] in _STRIP:
        i += 1
    if i >= L:
        return -1
    c = tok[i].lower()
    if   c == 'a': return 0
    elif c == 'b': return 1
    elif c == 'c': return 2
    elif c == 'd': return 3
    return -1

def _lse_add(a: float, b: float) -> float:
    """Stable log-sum-exp of two terms, using -1e9 as sentinel for -inf."""
    if a == -1e9: return b
    if b == -1e9: return a
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))

# ================================================================
# Normalization: vLLM step top-k -> [(token_text, logprob)] sorted by rank/logprob
# ================================================================

def _pairs_from_step(step: Any) -> List[Tuple[str, float]]:
    """
    Normalize ONE step's top-k into a sorted list of (decoded_token, logprob).
    Accepts:
      - Mapping[int -> obj] where obj has attributes: decoded_token/token, logprob, rank
      - Iterable of dict/obj/tuple each containing token/logprob/(optional rank)
      - Already [(token, logprob)] pairs

    Sorting rule:
      1) ascending by rank if present; else
      2) descending by logprob.
    """
    rows: List[Tuple[str, float, Optional[int]]] = []

    def _push(tok: Any, lp: Any, rk: Any = None):
        if lp is None:
            return
        t = str(tok) if tok is not None else ""
        try:
            lpf = float(lp)
        except Exception:
            return
        rki = int(rk) if rk is not None else None
        rows.append((t, lpf, rki))

    # Case A: Mapping[int -> Logprob-like object]
    if isinstance(step, Mapping):
        for _id, obj in step.items():  # pyright: ignore[reportUnknownArgumentType]
            if isinstance(obj, Mapping):
                tok = obj.get("decoded_token") or obj.get("token") or obj.get("text") or ""
                lp  = obj.get("logprob")
                rk  = obj.get("rank")
            else:
                tok = getattr(obj, "decoded_token", None) or getattr(obj, "token", None) or getattr(obj, "text", None) or ""
                lp  = getattr(obj, "logprob", None)
                rk  = getattr(obj, "rank", None)
            _push(tok, lp, rk)
    # Case B: Iterable
    elif isinstance(step, (list, tuple)):
        if step and isinstance(step[0], (list, tuple)) and len(step[0]) >= 2 and isinstance(step[0][0], (str, bytes)):
            # Already pairs [(token, logprob)] (maybe with extra rank in pos 2)
            for it in step:
                tok = it[0]
                lp  = it[1]
                rk  = it[2] if len(it) > 2 else None
                _push(tok, lp, rk)
        else:
            for it in step:
                if isinstance(it, Mapping):
                    tok = it.get("decoded_token") or it.get("token") or it.get("text") or ""
                    lp  = it.get("logprob")
                    rk  = it.get("rank")
                elif isinstance(it, (list, tuple)):
                    tok = it[0] if len(it) > 0 else ""
                    lp  = it[1] if len(it) > 1 else None
                    rk  = it[2] if len(it) > 2 else None
                else:
                    tok = getattr(it, "decoded_token", None) or getattr(it, "token", None) or getattr(it, "text", None) or ""
                    lp  = getattr(it, "logprob", None)
                    rk  = getattr(it, "rank", None)
                _push(tok, lp, rk)
    else:
        # Unknown; return empty.
        pass

    # Sort: by rank if given; otherwise by logprob descending
    rows.sort(key=lambda x: (x[2] if x[2] is not None else float("inf"), -x[1]))
    return [(t, lp) for (t, lp, _rk) in rows]

# ================================================================
# Slot aggregation
# ================================================================

def _slot_lse_vec_and_tail_from_pairs(
    pairs: List[Tuple[str, float]], topk: int
) -> Tuple[np.ndarray, float, str, str]:
    """
    Given pairs [(token, logprob)], compute LSE aggregation for A/B/C/D buckets.
    Returns:
      vals: np.array([A,B,C,D]) with -1e9 as initial -inf; caller will fallback-fill.
      tail: the smallest finite logprob encountered among used pairs (for fallback)
      top_dec: top-1 token text
      next_dec: top-2 token text (may help regex letter hints)
    """
    items = pairs[:max(1, int(topk))]
    top_dec = items[0][0] if items else ""
    next_dec = items[1][0] if len(items) > 1 else ""
    A = B = C = D = -1e9
    tail = float('inf')
    for tok, lp in items:
        try_lp = float(lp)
        if np.isfinite(try_lp) and try_lp < tail:
            tail = try_lp
        b = _first_letter_bucket_from_decoded(tok)
        if   b == 0: A = _lse_add(A, try_lp)
        elif b == 1: B = _lse_add(B, try_lp)
        elif b == 2: C = _lse_add(C, try_lp)
        elif b == 3: D = _lse_add(D, try_lp)
    vals = np.array([A, B, C, D], dtype=float)
    if not np.isfinite(tail):
        tail = -100.0
    return vals, tail, top_dec, next_dec

def _slot_lse_vec_and_tail_from_steps(steps: Sequence[Any], topk: int):
    """
    Iterate multiple steps (each possibly a Mapping[int->Logprob] or list of entries),
    convert with `_pairs_from_step`, and keep the step with the best coverage.
    Coverage = number of buckets with finite LSE (> -1e8). Tie-breaker: later step.
    """
    pre: List[Tuple[int, np.ndarray, float, str, str, int]] = []
    for idx, step in enumerate(steps or []):
        pairs = _pairs_from_step(step)
        vals, tail, top_dec, next_dec = _slot_lse_vec_and_tail_from_pairs(pairs, topk)
        cov = int((vals > -1e8).sum())
        pre.append((idx, vals, tail, top_dec, next_dec, cov))

    if not pre:
        # Empty -> default
        return np.full((4,), -100.0), -100.0, "", ""

    # Try 1) earliest step whose top-1 visibly points to ABCD
    for (idx, vals, tail, top_dec, next_dec, cov) in pre:
        if _first_letter_bucket_from_decoded(top_dec) in (0, 1, 2, 3):
            return vals, tail, top_dec, next_dec

    # Else 2) best coverage; tie -> later step
    best_key = None
    best_pack = None
    for (idx, vals, tail, top_dec, next_dec, cov) in pre:
        key = (cov, idx)
        if best_key is None or key > best_key:
            best_key = key
            best_pack = (vals, tail, top_dec, next_dec)
    assert best_pack is not None
    return best_pack

# ================================================================
# Public API
# ================================================================

def compute_probe_slot_from_vllm_steps(
    *,
    steps_logprobs: Sequence[Any],
    topk: int,
    prefer_letter: Optional[str],
    probe_text: str = "",
) -> Dict[str, Any]:
    """
    Robustly compute A/B/C/D slot scores from vLLM V1 probe outputs.

    Parameters
    ----------
    steps_logprobs:
        A sequence where each element is ONE decoding step's top-k.
        Supported forms per step:
          - dict[token_id -> Logprob{decoded_token/token, logprob, rank}]
          - list[dict/obj] each having decoded_token/token/logprob/(rank)
          - list[(token_text, logprob)] or list[(token_text, logprob, rank)]
    topk:
        Use the top-k entries from each step when aggregating.
    prefer_letter:
        Optional current cum-top hint ("A"/"B"/"C"/"D") to break ties later.
        (This function only uses it when probe_text yields no direct letter.)
    probe_text:
        The raw decoded probe text (e.g., from the short forced answer).

    Returns
    -------
    dict with:
        - "vals": np.array([A,B,C,D]) log-scores after fallback fill
        - "probe_letter": Optional[str] (A/B/C/D if detected)
        - "early_stop_elig": bool  (True if a letter is detected)
    """
    vals, tail, top_dec, next_dec = _slot_lse_vec_and_tail_from_steps(steps_logprobs, topk)

    # Fallback for missing buckets
    fallback = (tail if np.isfinite(tail) else -100.0) - 2.0
    vals = np.where(vals <= -1e8, fallback, vals)
    if not np.all(np.isfinite(vals)):
        base = np.nanmin(vals) if np.isfinite(np.nanmin(vals)) else -1000.0
        vals = np.where(np.isfinite(vals), vals, base - 10.0)

    # Regex detectors for explicit letter in the probe text or top candidates
    _PAT_STRONG = re.compile(r"(?i)(?:^|\b(?:answer|ans)\s*[:\-]?\s*)([ABCD])(?:\s*[:\.\)]|\s|$)")
    _PAT_SOFT   = re.compile(r"(?i)(?<![A-Za-z])([ABCD])(?![A-Za-z])")

    def _pick_letter(s: str) -> Optional[str]:
        m = _PAT_STRONG.search(s)
        if m: return m.group(1).upper()
        m2 = _PAT_SOFT.search(s)
        return m2.group(1).upper() if m2 else None

    probe_letter = (
        _pick_letter(top_dec) or
        _pick_letter(next_dec) or
        _pick_letter(probe_text or "") or
        (prefer_letter if isinstance(prefer_letter, str) and prefer_letter in "ABCD" else None)
    )
    elig = probe_letter is not None

    return {"vals": vals, "probe_letter": probe_letter, "early_stop_elig": bool(elig)}

# ================================================================
# Online sequential features (unchanged public interface)
# ================================================================

def _quad_second_derivative(y):
    y = np.asarray(y, dtype=float)
    if y.size < 3: return 0.0
    t = np.arange(y.size, dtype=float)
    coef = np.polyfit(t, y, deg=2)
    return float(2.0 * coef[0])

class SeqFeatureTracker:
    """
    Maintains cumulative log-prob tracks over A/B/C/D and exposes online features.
    cum is kept in log-space. Each update expects `vals` in log-score space.
    """
    def __init__(self, W:int=5, K_recent:int=3):
        self.cum = np.zeros(4, dtype=float)
        self.prev_top: Optional[int] = None
        self.run_len = 0
        self.flips = 0
        self.cum_margin_seq: List[float] = []
        self.hist_margin: List[float] = []
        self.hist_cum_vecs: List[np.ndarray] = []
        self.W = W
        self.K_recent = K_recent
        self.EPS = 1e-9

    def current_cum_top_letter(self) -> Optional[str]:
        order = np.argsort(self.cum)[::-1]
        return "ABCD"[int(order[0])]

    def update_with_step_vals(self, vals: np.ndarray) -> Dict[str, float]:
        # Convert step log-scores to normalized probs (softmax), add to cum (in log-space)
        m = float(np.max(vals))
        z = np.exp(vals - m)
        p = z / (z.sum() + 1e-12)
        self.cum += np.log(p + self.EPS)

        order = np.argsort(self.cum)[::-1]
        top, sec = int(order[0]), int(order[1])
        cum_top_letter = "ABCD"[top]
        cum_margin = float(self.cum[top] - self.cum[sec])

        if self.prev_top is None or self.prev_top != top:
            self.flips += (0 if self.prev_top is None else 1)
            self.run_len = 1
            self.prev_top = top
        else:
            self.run_len += 1

        self.hist_margin.append(cum_margin)
        self.hist_cum_vecs.append(self.cum.copy())
        if len(self.hist_margin) > self.W:
            self.hist_margin.pop(0); self.hist_cum_vecs.pop(0)

        self.cum_margin_seq.append(cum_margin)
        seq = np.array(self.cum_margin_seq, dtype=float)
        if len(seq) >= 2:
            delta_recent = float(seq[-1] - seq[max(0, len(seq)-self.K_recent)])
            slope_recent = float((seq[-1] - seq[0]) / max(1, len(seq)-1))
        else:
            delta_recent = 0.0; slope_recent = 0.0

        curv_margin2 = _quad_second_derivative(self.hist_margin)
        curv_cum_A2 = _quad_second_derivative([v[0] for v in self.hist_cum_vecs])
        curv_cum_B2 = _quad_second_derivative([v[1] for v in self.hist_cum_vecs])
        curv_cum_C2 = _quad_second_derivative([v[2] for v in self.hist_cum_vecs])
        curv_cum_D2 = _quad_second_derivative([v[3] for v in self.hist_cum_vecs])

        return {
            "cum_A": float(self.cum[0]), "cum_B": float(self.cum[1]),
            "cum_C": float(self.cum[2]), "cum_D": float(self.cum[3]),
            "cum_top": cum_top_letter,
            "cum_margin": float(cum_margin),
            "run_len": int(self.run_len),
            "flips": int(self.flips),
            "delta_recent": float(delta_recent),
            "slope_recent": float(slope_recent),
            "inst_sA": float(vals[0]), "inst_sB": float(vals[1]),
            "inst_sC": float(vals[2]), "inst_sD": float(vals[3]),
            "curv_margin2": float(curv_margin2),
            "curv_cum_A2": float(curv_cum_A2),
            "curv_cum_B2": float(curv_cum_B2),
            "curv_cum_C2": float(curv_cum_C2),
            "curv_cum_D2": float(curv_cum_D2),
        }

def cum_top_onehot(cum_top: Optional[str]) -> Dict[str, float]:
    oh = {"cum_top_A": 0.0, "cum_top_B": 0.0, "cum_top_C": 0.0, "cum_top_D": 0.0}
    if isinstance(cum_top, str) and cum_top in "ABCD":
        oh[f"cum_top_{cum_top}"] = 1.0
    return oh

# ================================================================
# OpenQA (e.g., math) features: 1D L_sum trajectory over probes
# ================================================================

def _math_L_stats_from_steps(
    steps: Sequence[Any],
    topk: int,
) -> Tuple[float, int, float, float]:
    """
    从 vLLM 每步 top-k 分布中，构造 openQA 的 1D 统计：
      - 只看每步 top-1 的 logprob，近似当前 forced answer 的路径 logprob；
      - L_sum = 所有步 top-1 logprob 之和；
      - ans_len = 步数（token 数）；
      - mean_logprob / var_logprob = 逐 token 的均值与方差。

    注意：steps 的每个元素可以是 vLLM 的 Mapping / 对象 / 已经变成 [(tok, lp)] 的列表；
    我们用 _pairs_from_step 做统一。
    """
    lps: List[float] = []
    for step in steps or []:
        pairs = _pairs_from_step(step)
        if not pairs:
            continue
        lp = float(pairs[0][1])  # top-1 logprob
        if np.isfinite(lp):
            lps.append(lp)

    if not lps:
        return 0.0, 0, 0.0, 0.0

    arr = np.asarray(lps, dtype=float)
    L_sum = float(np.sum(arr))
    ans_len = int(arr.size)
    mean_lp = float(np.mean(arr))
    var_lp = float(np.var(arr))
    return L_sum, ans_len, mean_lp, var_lp


def compute_openqa_slot_from_vllm_steps(
    *,
    steps_logprobs: Sequence[Any],
    topk: int,
) -> Dict[str, Any]:
    """
    openQA/math 用的 probe 聚合：
      - 不再做 ABCD 桶，只做 1 维 L_sum 统计；
      - 提供 L_sum / ans_len / mean_logprob / var_logprob / neg_ppl；
      - 不需要 probe_letter，early_stop_elig 一律 True（由上层 classifier 控制早停）。
    """
    L_sum, ans_len, mean_lp, var_lp = _math_L_stats_from_steps(
        steps_logprobs,
        topk=topk,
    )
    neg_ppl = -mean_lp  # 简单 proxy：越大越“自信”

    return {
        "L_sum": float(L_sum),
        "ans_len": int(ans_len),
        "mean_logprob": float(mean_lp),
        "var_logprob": float(var_lp),
        "neg_ppl": float(neg_ppl),
        "early_stop_elig": True,    # openQA 不依赖 ABCD letter，统一允许被 classifier 早停
        "probe_letter": None,       # 占位，便于上层代码统一访问
    }


class OpenSeqFeatureTracker:
    """
    openQA/math 版本的在线特征跟踪器（1D L_sum 轨迹）。

    设计目标：尽量对齐你 train 脚本里的特征：
      - L_sum, S_es, H_es, ans_len
      - run_len, flips, changed_prev
      - delta_recent_L, slope_recent_L, curv_L2, vel_L, acc_L
      - mean_logprob, var_logprob, neg_ppl

    其中 run_len / flips / changed_prev 用“当前 answer_key 是否变化”来刻画，
    answer_key 由调用方根据 probe_text 提供（例如 extract <final_answer> 后简单归一化）。
    """
    def __init__(self, W: int = 5, K_recent: int = 3):
        self.L_hist: List[float] = []
        self.S_hist: List[float] = []
        self.recent_L: List[float] = []
        self.prev_key: Optional[str] = None
        self.run_len: int = 0
        self.flips: int = 0
        self.W = int(W)
        self.K_recent = int(K_recent)

    def update_with_slot(
        self,
        slot: Mapping[str, Any],
        answer_key: Optional[str],
    ) -> Dict[str, float]:
        """
        参数：
          - slot: compute_openqa_slot_from_vllm_steps 的输出 dict；
          - answer_key: 当前 probe 的“答案 key”（例如简单归一化后的 \boxed{...} 文本）。
        返回：
          - 一整套 openQA 用特征（字段名与 train 代码保持一致）。
        """
        L_t = float(slot.get("L_sum", 0.0))
        ans_len = float(slot.get("ans_len", 0.0))
        mean_lp = float(slot.get("mean_logprob", 0.0))
        var_lp = float(slot.get("var_logprob", 0.0))
        neg_ppl = float(slot.get("neg_ppl", -mean_lp))

        if self.L_hist:
            S_t = float(L_t - self.L_hist[-1])
            H_t = float(S_t - (self.S_hist[-1] if self.S_hist else 0.0))
        else:
            S_t = 0.0
            H_t = 0.0

        # key 稳定性：answer_key 相同则 run_len++，否则 flips++
        if self.prev_key is None:
            changed_prev = 0
            self.run_len = 1
        elif (answer_key is not None) and (answer_key != self.prev_key):
            changed_prev = 1
            self.flips += 1
            self.run_len = 1
        else:
            changed_prev = 0
            self.run_len += 1

        if answer_key is not None:
            self.prev_key = answer_key

        # 更新历史轨迹
        self.L_hist.append(L_t)
        self.S_hist.append(S_t)
        self.recent_L.append(L_t)
        if len(self.recent_L) > self.W:
            self.recent_L.pop(0)

        # delta_recent_L / slope_recent_L
        if len(self.L_hist) >= 2:
            idx0 = max(0, len(self.L_hist) - self.K_recent)
            delta_recent_L = float(L_t - self.L_hist[idx0])
            slope_recent_L = float(
                (L_t - self.L_hist[0]) / max(1, len(self.L_hist) - 1)
            )
        else:
            delta_recent_L = 0.0
            slope_recent_L = 0.0

        # 二次拟合曲率（最近 W 步）
        if len(self.recent_L) >= 3:
            curv_L2 = _quad_second_derivative(self.recent_L)
        else:
            curv_L2 = 0.0

        # 速度 / 加速度
        if len(self.L_hist) >= 2:
            vel_L = float(L_t - self.L_hist[-2])
        else:
            vel_L = 0.0

        if len(self.L_hist) >= 3:
            acc_L = float(
                L_t - 2.0 * self.L_hist[-2] + self.L_hist[-3]
            )
        else:
            acc_L = 0.0

        return {
            "L_sum": float(L_t),
            "S_es": float(S_t),
            "H_es": float(H_t),
            "ans_len": float(ans_len),
            "run_len": int(self.run_len),
            "flips": int(self.flips),
            "changed_prev": int(changed_prev),
            "delta_recent_L": float(delta_recent_L),
            "slope_recent_L": float(slope_recent_L),
            "curv_L2": float(curv_L2),
            "vel_L": float(vel_L),
            "acc_L": float(acc_L),
            "mean_logprob": float(mean_lp),
            "var_logprob": float(var_lp),
            "neg_ppl": float(neg_ppl),
        }
