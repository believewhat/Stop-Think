"""Reward calculation for full, unmodified AdaptThink policy responses."""

from __future__ import annotations

import math
import os
import re
from numbers import Integral, Real
from typing import Any

from math_verify import parse, verify

try:
    # Normal execution loads this file from the AdaptThink repository root.
    from verl.src.medqa_utils import (
        extract_medqa_choice,
        extract_strict_terminal_choice,
    )
except ModuleNotFoundError:
    # Keep direct execution from verl/src usable for focused CPU tests.
    from medqa_utils import extract_medqa_choice, extract_strict_terminal_choice


ACCURACY_WEIGHT = 0.50
FORMAT_WEIGHT = 0.25
STOP_ACTION_WEIGHT = 1.0
# Kept as an import-compatible alias for preflights and downstream telemetry.
# This weight is never part of the sequence score: it is consumed only by the
# single verified ``<stop>`` action loss in the actor.
STOP_WEIGHT = STOP_ACTION_WEIGHT

assert ACCURACY_WEIGHT >= FORMAT_WEIGHT, (
    "terminal accuracy must dominate the sequence-level format reward"
)

_ASCII_WHITESPACE = r"[ \t\r\n\f\v]"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_STOP = "<stop>"
_FINAL_OPEN = "<final_answer>"
_FINAL_CLOSE = "</final_answer>"
_THINK_PREFIX_RE = re.compile(rf"\A{_ASCII_WHITESPACE}*<think>")
_ASCII_WHITESPACE_ONLY_RE = re.compile(rf"{_ASCII_WHITESPACE}*\Z")


def hf_math_rm(solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """Grade MedQA with the raw strict parser and retain math_verify fallback."""

    del extra_info
    gold_choice = extract_medqa_choice(ground_truth)
    if gold_choice is not None:
        pred_choice = extract_strict_terminal_choice(solution_str)
        acc = pred_choice == gold_choice
        return {
            "score": acc,
            "acc": acc,
            "pred": pred_choice or "ERROR: Answer Extraction Failed",
            "strict_final_valid": int(pred_choice is not None),
        }

    model_solution = solution_str.strip()[-500:]
    preds = parse(model_solution)
    gold = parse(ground_truth)
    acc = verify(gold, preds)
    if preds is None or preds == []:
        pred = "ERROR: Answer Extraction Failed"
    else:
        pred = str(preds[0])
    assert isinstance(pred, str), preds
    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
        "strict_final_valid": 0,
    }


def _well_formed_response_format(solution_str: str) -> int:
    """Check the exact think/terminal-final structure.

    ``<stop>`` is an optional action-level signal.  When it is present, every
    visible stop must still be inside the thinking region and must agree with
    the authoritative token-level metadata checked by ``compute_score``.
    Absence of a stop must not invalidate an otherwise complete response.
    """

    if not isinstance(solution_str, str):
        return 0
    text = solution_str
    if extract_strict_terminal_choice(text) is None:
        return 0
    if text.count(_THINK_OPEN) != 1 or text.count(_THINK_CLOSE) != 1:
        return 0

    think_prefix = _THINK_PREFIX_RE.match(text)
    if think_prefix is None:
        return 0
    think_open_end = think_prefix.end()
    think_close_start = text.find(_THINK_CLOSE)
    if think_close_start < think_open_end:
        return 0
    think_close_end = think_close_start + len(_THINK_CLOSE)

    stop_matches = list(re.finditer(re.escape(_STOP), text))
    if any(
        match.start() < think_open_end or match.end() > think_close_start
        for match in stop_matches
    ):
        return 0

    final_open_start = text.find(_FINAL_OPEN)
    if final_open_start < think_close_end:
        return 0
    between = text[think_close_end:final_open_start]
    if _ASCII_WHITESPACE_ONLY_RE.fullmatch(between) is None:
        return 0
    return 1


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _required_int(name: str, value: Any) -> int:
    value = _scalar(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{name} must be an integer scalar, got {value!r}")


def _required_flag(name: str, value: Any) -> bool:
    integer = _required_int(name, value)
    if integer not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {integer}")
    return bool(integer)


def _required_float(name: str, value: Any) -> float:
    value = _scalar(value)
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _required_stop_kind(value: Any) -> str:
    value = _scalar(value)
    if not isinstance(value, str):
        raise ValueError(f"selected_stop_kind must be a string, got {value!r}")
    if value not in {"none", "gold_correct", "final_consistent"}:
        raise ValueError(f"invalid selected_stop_kind: {value!r}")
    return value


def _validated_stop_metadata(
    *,
    verified_stop: Any,
    verified_stop_count: Any,
    verified_stop_score: Any,
    verified_stop_position: Any,
    raw_stop_count: Any,
    raw_atomic_stop_count: Any,
    illegal_surface_stop_count: Any,
    stop_fragment_count: Any,
    selected_stop_ordinal: Any,
    eligible_stop_count: Any,
    probed_stop_count: Any,
    max_stop_count: Any,
    stop_reward_half_life_tokens: Any,
    min_verified_stop_separation_tokens: Any,
    selected_stop_kind: Any = None,
    selected_stop_credit: Any = None,
    final_consistency_credit: Any = 1.0,
) -> tuple:
    verified = _required_flag("verified_stop", verified_stop)
    verified_count = _required_int("verified_stop_count", verified_stop_count)
    verified_score = _required_float("verified_stop_score", verified_stop_score)
    position = _required_int("verified_stop_position", verified_stop_position)
    raw_count = _required_int("raw_stop_count", raw_stop_count)
    raw_atomic_count = _required_int(
        "raw_atomic_stop_count", raw_atomic_stop_count
    )
    illegal_surface_count = _required_int(
        "illegal_surface_stop_count", illegal_surface_stop_count
    )
    fragment_count = _required_int("stop_fragment_count", stop_fragment_count)
    ordinal = _required_int("selected_stop_ordinal", selected_stop_ordinal)
    eligible_count = _required_int("eligible_stop_count", eligible_stop_count)
    probed_count = _required_int("probed_stop_count", probed_stop_count)
    cap = _required_int("max_stop_count", max_stop_count)
    half_life = _required_float(
        "stop_reward_half_life_tokens", stop_reward_half_life_tokens
    )
    min_separation = _required_int(
        "min_verified_stop_separation_tokens",
        min_verified_stop_separation_tokens,
    )
    kind = _required_stop_kind(
        ("gold_correct" if verified else "none")
        if selected_stop_kind is None
        else selected_stop_kind
    )
    credit = _required_float(
        "selected_stop_credit",
        (1.0 if verified else 0.0)
        if selected_stop_credit is None
        else selected_stop_credit,
    )
    consistency_credit = _required_float(
        "final_consistency_credit", final_consistency_credit
    )
    if not 0.0 < consistency_credit <= 1.0:
        raise ValueError(
            "final_consistency_credit must be in (0, 1], got "
            f"{consistency_credit}"
        )

    if raw_count < 0:
        raise ValueError(f"raw_stop_count must be non-negative, got {raw_count}")
    if raw_atomic_count < 0:
        raise ValueError(
            "raw_atomic_stop_count must be non-negative, got "
            f"{raw_atomic_count}"
        )
    if illegal_surface_count < 0:
        raise ValueError(
            "illegal_surface_stop_count must be non-negative, got "
            f"{illegal_surface_count}"
        )
    if fragment_count < 0:
        raise ValueError(
            f"stop_fragment_count must be non-negative, got {fragment_count}"
        )
    if raw_count != raw_atomic_count + illegal_surface_count:
        raise ValueError(
            "raw_stop_count must equal atomic plus illegal surface events: "
            f"{raw_count} != {raw_atomic_count} + {illegal_surface_count}"
        )
    if fragment_count > illegal_surface_count:
        raise ValueError(
            "stop_fragment_count cannot exceed illegal_surface_stop_count: "
            f"{fragment_count} > {illegal_surface_count}"
        )
    if eligible_count < 0:
        raise ValueError(
            f"eligible_stop_count must be non-negative, got {eligible_count}"
        )
    if probed_count < 0:
        raise ValueError(f"probed_stop_count must be non-negative, got {probed_count}")
    if cap < 1:
        raise ValueError(f"max_stop_count must be positive, got {cap}")
    if half_life <= 0:
        raise ValueError(
            "stop_reward_half_life_tokens must be finite and positive, got "
            f"{half_life}"
        )
    if min_separation < 1:
        raise ValueError(
            "min_verified_stop_separation_tokens must be positive, got "
            f"{min_separation}"
        )
    if eligible_count > raw_atomic_count:
        raise ValueError(
            "eligible_stop_count cannot exceed raw_atomic_stop_count: "
            f"{eligible_count} > {raw_atomic_count}"
        )
    # Surface events consume the same global attempt budget but are never
    # eligible for an oracle probe.  The rollout therefore may legitimately
    # probe fewer than min(eligible_count, cap) atomic proposals.
    max_probed = min(eligible_count, cap, raw_atomic_count)
    if probed_count > max_probed:
        raise ValueError(
            "probed_stop_count cannot exceed the eligible/raw stop budget: "
            f"probed={probed_count}, eligible={eligible_count}, "
            f"raw={raw_count}, cap={cap}"
        )
    if verified_count not in (0, 1):
        raise ValueError(
            "earliest-only credit requires verified_stop_count in {0, 1}, "
            f"got {verified_count}"
        )
    if verified != (verified_count > 0):
        raise ValueError(
            "verified_stop must equal int(verified_stop_count > 0): "
            f"verified_stop={int(verified)}, "
            f"verified_stop_count={verified_count}"
        )
    if not verified:
        if kind != "none" or credit != 0.0:
            raise ValueError(
                "unselected rows require selected_stop_kind='none' and "
                f"selected_stop_credit=0, got kind={kind!r} credit={credit}"
            )
    elif kind == "gold_correct":
        if credit != 1.0:
            raise ValueError(
                "gold_correct selected stops require credit=1, got "
                f"{credit}"
            )
    elif kind == "final_consistent":
        if not math.isclose(
            credit,
            consistency_credit,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "final_consistent selected stops require credit equal to "
                "final_consistency_credit: "
                f"{credit} != {consistency_credit}"
            )
    else:
        raise ValueError(
            "selected rows require kind gold_correct or final_consistent, got "
            f"{kind!r}"
        )
    if verified_count > probed_count:
        raise ValueError(
            "a verified stop must correspond to a probed stop: "
            f"verified_stop_count={verified_count}, "
            f"probed_stop_count={probed_count}"
        )

    if verified:
        if position < 0:
            raise ValueError(
                "verified_stop_position must be non-negative for a verified stop"
            )
        # The selected ordinal is measured in the unified atomic-plus-surface
        # event stream.  Surface events can consume earlier budget slots
        # without being probeable, so this ordinal may exceed probed_count.
        if ordinal < 1 or ordinal > cap or ordinal > raw_count:
            raise ValueError(
                "selected_stop_ordinal must identify a legal global stop event "
                "when verified: "
                f"ordinal={ordinal}, raw_stop_count={raw_count}, cap={cap}"
            )
    elif position != -1 or ordinal != -1:
        raise ValueError(
            "unverified stops require verified_stop_position=-1 and "
            "selected_stop_ordinal=-1"
        )
    if not verified:
        if verified_score != 0.0:
            raise ValueError(
                "unverified rollouts require verified_stop_score=0, got "
                f"{verified_score}"
            )
    else:
        earliness = float(2.0 ** (-float(position) / half_life))
        expected_score = earliness
        tolerance = 1e-6
        if (
            verified_score <= 0.0
            or verified_score > 1.0 + tolerance
            or not math.isclose(
                verified_score,
                expected_score,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(
                "verified_stop_score must equal the shared exponential "
                "earliness score: "
                f"score={verified_score}, position={position}, "
                f"half_life={half_life}, kind={kind}, "
                f"final_consistency_credit={consistency_credit}, "
                f"expected={expected_score}"
            )

    return (
        verified,
        verified_count,
        verified_score,
        position,
        raw_count,
        raw_atomic_count,
        illegal_surface_count,
        fragment_count,
        ordinal,
        eligible_count,
        probed_count,
        cap,
        half_life,
        min_separation,
        kind,
        credit,
        consistency_credit,
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    *,
    verified_stop,
    verified_stop_count,
    verified_stop_score,
    verified_stop_position,
    raw_stop_count,
    raw_atomic_stop_count,
    illegal_surface_stop_count,
    stop_fragment_count,
    selected_stop_ordinal,
    eligible_stop_count,
    probed_stop_count,
    max_stop_count,
    stop_reward_half_life_tokens,
    min_verified_stop_separation_tokens,
    selected_stop_kind=None,
    selected_stop_credit=None,
    final_consistency_credit=1.0,
    extra_info=None,
    tokenizer=None,
):
    """Score the supplied raw response without truncation or answer rewriting."""

    del data_source, tokenizer
    (
        verified,
        verified_count,
        verified_score,
        stop_position,
        raw_count,
        raw_atomic_count,
        illegal_surface_count,
        fragment_count,
        stop_ordinal,
        eligible_count,
        probed_count,
        stop_cap,
        half_life,
        min_separation,
        stop_kind,
        stop_credit,
        consistency_credit,
    ) = _validated_stop_metadata(
        verified_stop=verified_stop,
        verified_stop_count=verified_stop_count,
        verified_stop_score=verified_stop_score,
        verified_stop_position=verified_stop_position,
        raw_stop_count=raw_stop_count,
        raw_atomic_stop_count=raw_atomic_stop_count,
        illegal_surface_stop_count=illegal_surface_stop_count,
        stop_fragment_count=stop_fragment_count,
        selected_stop_ordinal=selected_stop_ordinal,
        eligible_stop_count=eligible_stop_count,
        probed_stop_count=probed_stop_count,
        max_stop_count=max_stop_count,
        stop_reward_half_life_tokens=stop_reward_half_life_tokens,
        min_verified_stop_separation_tokens=(
            min_verified_stop_separation_tokens
        ),
        selected_stop_kind=selected_stop_kind,
        selected_stop_credit=selected_stop_credit,
        final_consistency_credit=final_consistency_credit,
    )

    accuracy_result = hf_math_rm(
        solution_str,
        ground_truth,
        extra_info=extra_info,
    )
    strict_raw_acc = int(bool(accuracy_result["acc"]))
    strict_final_valid = int(bool(accuracy_result["strict_final_valid"]))
    structural_format_score = _well_formed_response_format(solution_str)
    visible_stop_count = (
        solution_str.count(_STOP) if isinstance(solution_str, str) else 0
    )
    stop_surface_count_matches_raw = visible_stop_count == raw_atomic_count
    all_atomic_stops_eligible = eligible_count == raw_atomic_count
    no_illegal_surface_stops = illegal_surface_count == 0
    eligible_atomic_stop_present = int(eligible_count >= 1)
    medqa_example = extract_medqa_choice(ground_truth) is not None
    # A terminal MedQA answer remains the only sequence-level requirement.
    # ``<stop>`` is optional here and is optimized exclusively through its
    # dedicated action objective.  Missing it therefore cannot reverse the
    # accuracy/format gradient of an otherwise complete response.
    required_output_valid = int(not medqa_example or strict_final_valid)
    output_requirement_gated = int(medqa_example and not required_output_valid)
    # Decoding is not injective: ordinary vocabulary tokens can render the same
    # literal ``<stop>`` as the atomic rollout token, and malformed ``stop>``
    # fragments may not appear in a literal-string count at all.  Token-level
    # ByteLevel event metadata is authoritative for both cases.
    stop_presence_format_score = int(raw_atomic_count >= 1)
    format_score = int(
        structural_format_score
        and stop_surface_count_matches_raw
        and all_atomic_stops_eligible
        and no_illegal_surface_stops
    )
    stop_earliness = (
        float(2.0 ** (-float(stop_position) / half_life)) if verified else 0.0
    )
    selected_stop_weight = float(verified_score) if verified else 0.0
    stop_over_limit = raw_count > stop_cap

    accuracy_reward = strict_raw_acc * ACCURACY_WEIGHT
    format_reward = format_score * FORMAT_WEIGHT
    # This is action-level telemetry.  The actor applies the same value only to
    # the single oracle-verified ``<stop>`` token.  Keeping it out of ``score``
    # prevents GRPO from broadcasting stop earliness to the response tail.
    stop_reward = selected_stop_weight * STOP_ACTION_WEIGHT
    # Overflow no longer invalidates the sequence reward.  Accuracy and format
    # remain properties of the completed response, while every atomic stop
    # after the legal budget is penalized separately at the sampled action.
    # A missing strict terminal answer remains fail-closed. Missing ``<stop>``
    # only yields zero stop-action reward and does not gate sequence reward.
    reward_hard_gated = bool(output_requirement_gated)
    if reward_hard_gated:
        # For MedQA, a missing strict terminal answer invalidates every positive
        # component. Overflow telemetry remains independent and still routes
        # every post-budget stop to the direct negative action loss.
        accuracy_reward = 0.0
        format_reward = 0.0
        stop_reward = 0.0
    total_reward = accuracy_reward + format_reward

    if os.environ.get("ADAPTTHINK_REWARD_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(
            f"format_reward:{format_reward:.3f} "
            f"stop_earliness:{stop_earliness:.3f} "
            f"stop_reward:{stop_reward:.3f} "
            f"accuracy_reward:{accuracy_reward:.3f} "
            f"verified_stop:{int(verified)} "
            f"verified_stop_count:{verified_count} "
            f"raw_stop_count:{raw_count} "
            f"raw_atomic_stop_count:{raw_atomic_count} "
            f"illegal_surface_stop_count:{illegal_surface_count} "
            f"stop_fragment_count:{fragment_count} "
            f"visible_stop_count:{visible_stop_count} "
            f"stop_surface_count_matches_raw:{int(stop_surface_count_matches_raw)} "
            f"strict_final_valid:{strict_final_valid} "
            f"eligible_atomic_stop_present:{eligible_atomic_stop_present} "
            f"output_requirement_gated:{output_requirement_gated} "
            f"stop_over_limit:{int(stop_over_limit)}"
        )

    return {
        "score": float(total_reward),
        "acc": strict_raw_acc,
        "strict_raw_acc": strict_raw_acc,
        "raw_strict_acc": strict_raw_acc,
        "pred": accuracy_result["pred"],
        "strict_final_valid": strict_final_valid,
        "eligible_atomic_stop_present": eligible_atomic_stop_present,
        "required_output_valid": required_output_valid,
        "output_requirement_gated": output_requirement_gated,
        "format_score": int(format_score),
        "stop_presence_format_score": int(stop_presence_format_score),
        "structural_format_score": int(structural_format_score),
        "format_reward": float(format_reward),
        "stop_earliness": float(stop_earliness),
        "stop_reward": float(stop_reward),
        "stop_action_reward": float(stop_reward),
        "sequence_stop_reward": 0.0,
        "accuracy_reward": float(accuracy_reward),
        "verified_stop": int(verified),
        "verified_stop_count": verified_count,
        "verified_stop_score": float(verified_score),
        "verified_stop_position": stop_position,
        "selected_stop_kind": stop_kind,
        "selected_stop_credit": float(stop_credit),
        "selected_stop_weight": selected_stop_weight,
        "final_consistency_credit": float(consistency_credit),
        "raw_stop_count": raw_count,
        "raw_atomic_stop_count": raw_atomic_count,
        "illegal_surface_stop_count": illegal_surface_count,
        "stop_fragment_count": fragment_count,
        "ordinary_surface_stop_count": illegal_surface_count - fragment_count,
        "visible_stop_count": visible_stop_count,
        "stop_surface_count_matches_raw": int(stop_surface_count_matches_raw),
        "legacy_surface_stop_count": illegal_surface_count - fragment_count,
        "selected_stop_ordinal": stop_ordinal,
        "eligible_stop_count": eligible_count,
        "all_raw_stops_eligible": int(
            all_atomic_stops_eligible and no_illegal_surface_stops
        ),
        "all_atomic_stops_eligible": int(all_atomic_stops_eligible),
        "no_illegal_surface_stops": int(no_illegal_surface_stops),
        "probed_stop_count": probed_count,
        "max_stop_count": stop_cap,
        "stop_reward_half_life_tokens": half_life,
        "min_verified_stop_separation_tokens": min_separation,
        "stop_over_limit": int(stop_over_limit),
        "auxiliary_reward_gated": int(reward_hard_gated),
        "reward_gated": int(reward_hard_gated),
        "stop_positive_reward_suppressed": int(reward_hard_gated),
        "overflow_action_penalty_required": int(stop_over_limit),
        "negative_stop_action_penalty_required": int(
            stop_over_limit or illegal_surface_count > 0
        ),
    }
