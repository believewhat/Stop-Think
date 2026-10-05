"""vLLM rollout with controller-terminal ESTAR training.

The policy first produces a complete ordinary rollout. Tokenizer-level atomic
``<stop>`` tokens, ordinary-token surface spellings, and bracket-adjacent
stop-like fragments share one ordered attempt budget. Only legal atomic events
inside the first thinking region are probed. Surface events are always negative;
atomic events become negative after the budget.  The earliest oracle-correct
atomic proposal is a real reasoning terminal: the sampled suffix is discarded,
the controller appends a canonical final answer from the accepted probe, and
only the sampled prefix participates in trajectory GRPO.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from .medqa_utils import extract_medqa_choice, extract_strict_terminal_choice
from .estar_a70_shared import (
    A70_FEATURES,
    A70_FEATURE_SCHEMA_SHA256,
    A70_FEATURE_SCHEMA_VERSION,
    A70Featurizer,
    REASONING_BUDGET,
)


CHOICES: Tuple[str, ...] = ("A", "B", "C", "D")
DEFAULT_FINAL_CONSISTENCY_CREDIT = 1.0

_ATOMIC_STOP_SENTINEL = 256
_OPAQUE_ADDED_TOKEN_SENTINEL = 257
_ASCII_LT = ord("<")
_ASCII_GT = ord(">")
_ASCII_STOP = tuple(b"stop")
_ASCII_WORD_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
)
_ASCII_FORMAT_WHITESPACE = " \t\r\n\f\v"


@dataclass(frozen=True)
class _StopEvent:
    """One atomic or surface-level stop attempt in policy byte order."""

    start_unit: int
    end_unit: int
    start_token: int
    end_token: int
    kind: str

    @property
    def is_atomic(self) -> bool:
        return self.kind == "atomic"

    @property
    def is_surface(self) -> bool:
        return not self.is_atomic


@dataclass(frozen=True)
class _ProbeRecord:
    """One stop-position probe with both frozen A70 boundary observations.

    Iteration/indexing intentionally preserve the historical three-field
    ``(answer, canonical_scores, canonical_probabilities)`` interface used by
    controller selection and auxiliary-loss code.  The legacy boundary is
    retained as explicit fields for the shared 70-feature featurizer.
    """

    answer: Optional[str]
    canonical_scores: np.ndarray
    canonical_probabilities: np.ndarray
    legacy_scores: np.ndarray
    legacy_probabilities: np.ndarray

    def __iter__(self):
        yield self.answer
        yield self.canonical_scores
        yield self.canonical_probabilities

    def __getitem__(self, index: int):
        values = (
            self.answer,
            self.canonical_scores,
            self.canonical_probabilities,
            self.legacy_scores,
            self.legacy_probabilities,
        )
        return values[index]


def _ascii_lower(value: int) -> int:
    value = int(value)
    if ord("A") <= value <= ord("Z"):
        return value + (ord("a") - ord("A"))
    return value


class _ByteLevelStopScanner:
    """Map Qwen ByteLevel token IDs to exact bytes and find stop-like events.

    Token IDs are never decoded and re-tokenized.  Each ordinary vocabulary
    piece is inverted through the same byte-to-unicode table used by the
    ByteLevel tokenizer, so a match spanning arbitrary BPE boundaries retains
    an exact byte-to-policy-action map.  Added tokens are opaque boundaries;
    the configured atomic stop is represented by its own out-of-byte sentinel.
    """

    def __init__(self, tokenizer, stop_token_id: int):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if decoder is None or "ByteLevel" not in repr(decoder):
            raise RuntimeError(
                "ESTAR surface-stop scanning requires a fast ByteLevel tokenizer"
            )
        self.tokenizer = tokenizer
        self.stop_token_id = int(stop_token_id)
        self.byte_decoder = {
            unicode_value: byte_value
            for byte_value, unicode_value in bytes_to_unicode().items()
        }
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        added_vocab = get_added_vocab() if callable(get_added_vocab) else {}
        self.added_token_ids = {int(value) for value in added_vocab.values()}
        self._piece_cache: Dict[int, Tuple[int, ...]] = {}

    def _token_units(self, token_id: int) -> Tuple[int, ...]:
        token_id = int(token_id)
        if token_id == self.stop_token_id:
            return (_ATOMIC_STOP_SENTINEL,)
        if token_id in self.added_token_ids:
            return (_OPAQUE_ADDED_TOKEN_SENTINEL,)
        cached = self._piece_cache.get(token_id)
        if cached is not None:
            return cached
        piece = self.tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(piece, str) or not piece:
            raise RuntimeError(
                f"tokenizer returned no ByteLevel vocabulary piece for id {token_id}"
            )
        try:
            units = tuple(self.byte_decoder[character] for character in piece)
        except KeyError as error:
            raise RuntimeError(
                "non-ByteLevel character in ordinary vocabulary piece "
                f"for id {token_id}: {piece!r}"
            ) from error
        if not units:
            raise RuntimeError(f"empty ByteLevel payload for token id {token_id}")
        self._piece_cache[token_id] = units
        return units

    def scan(self, token_ids: Sequence[int]) -> List[_StopEvent]:
        units: List[int] = []
        owners: List[int] = []
        events: List[_StopEvent] = []

        for token_index, raw_token_id in enumerate(token_ids):
            token_id = int(raw_token_id)
            start = len(units)
            piece = self._token_units(token_id)
            units.extend(piece)
            owners.extend([token_index] * len(piece))
            if token_id == self.stop_token_id:
                events.append(
                    _StopEvent(
                        start_unit=start,
                        end_unit=start,
                        start_token=token_index,
                        end_token=token_index,
                        kind="atomic",
                    )
                )

        # Match the ASCII word "stop" case-insensitively, but require an
        # adjacent angle bracket.  This catches malformed control fragments
        # without treating ordinary English "stop" as a control action.
        for stop_start in range(max(0, len(units) - len(_ASCII_STOP) + 1)):
            if tuple(
                _ascii_lower(value)
                for value in units[stop_start : stop_start + len(_ASCII_STOP)]
            ) != _ASCII_STOP:
                continue
            stop_end = stop_start + len(_ASCII_STOP)
            left = units[stop_start - 1] if stop_start > 0 else None
            right = units[stop_end] if stop_end < len(units) else None
            has_left_angle = left in (_ASCII_LT, _ASCII_GT)
            has_right_angle = right in (_ASCII_LT, _ASCII_GT)
            if not (has_left_angle or has_right_angle):
                continue
            if not has_left_angle and left in _ASCII_WORD_BYTES:
                continue
            if not has_right_angle and right in _ASCII_WORD_BYTES:
                continue

            span_start = stop_start - 1 if has_left_angle else stop_start
            span_end = stop_end if has_right_angle else stop_end - 1
            exact_lowercase_full = (
                left == _ASCII_LT
                and right == _ASCII_GT
                and tuple(units[stop_start:stop_end]) == _ASCII_STOP
            )
            events.append(
                _StopEvent(
                    start_unit=span_start,
                    end_unit=span_end,
                    start_token=owners[span_start],
                    end_token=owners[span_end],
                    kind=(
                        "ordinary_surface_stop"
                        if exact_lowercase_full
                        else "stop_fragment"
                    ),
                )
            )

        events.sort(
            key=lambda event: (
                event.start_unit,
                event.end_unit,
                0 if event.is_atomic else 1,
            )
        )
        return events


def _stop_event_action_metadata(
    stop_events: Sequence[_StopEvent],
    response_length: int,
    max_stop_count: int,
) -> Dict[str, Any]:
    """Build globally budgeted negative-action metadata for one rollout.

    Every non-atomic surface event is illegal even among the first eight
    attempts.  Atomic stops are illegal only when their ordinal in the merged
    atomic-plus-surface event stream exceeds the budget.  Multiplicity is
    retained when one sampled token completes more than one textual event, then
    normalized so each rollout contributes unit negative-stop mass at most.
    """

    length = int(response_length)
    cap = int(max_stop_count)
    if length < 0:
        raise ValueError(f"response_length must be non-negative, got {length}")
    if cap < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")

    stop_event_multiplicity = [0] * length
    surface_stop_event_multiplicity = [0] * length
    negative_multiplicity = [0] * length
    overflow_stop_mask = [0] * length
    atomic_ordinals: Dict[int, int] = {}
    raw_atomic_count = 0
    ordinary_surface_count = 0
    fragment_count = 0
    overflow_atomic_count = 0

    previous_start = -1
    for ordinal, event in enumerate(stop_events, start=1):
        if event.start_unit < previous_start:
            raise ValueError("stop_events must be ordered by policy byte position")
        previous_start = event.start_unit
        anchor = int(event.end_token)
        if anchor < 0 or anchor >= length:
            raise ValueError(
                f"stop event anchor {anchor} is outside response length {length}"
            )
        if event.is_atomic:
            stop_event_multiplicity[anchor] += 1
            raw_atomic_count += 1
            if anchor in atomic_ordinals:
                raise ValueError(f"duplicate atomic stop action at token {anchor}")
            atomic_ordinals[anchor] = ordinal
            if ordinal > cap:
                overflow_atomic_count += 1
                overflow_stop_mask[anchor] = 1
                negative_multiplicity[anchor] += 1
        else:
            stop_event_multiplicity[anchor] += 1
            surface_stop_event_multiplicity[anchor] += 1
            if event.kind == "ordinary_surface_stop":
                ordinary_surface_count += 1
            elif event.kind == "stop_fragment":
                fragment_count += 1
            else:
                raise ValueError(f"unknown stop event kind: {event.kind!r}")
            negative_multiplicity[anchor] += 1

    illegal_surface_count = ordinary_surface_count + fragment_count
    negative_event_count = illegal_surface_count + overflow_atomic_count
    if sum(negative_multiplicity) != negative_event_count:
        raise RuntimeError(
            "negative stop event multiplicity disagrees with event counts"
        )
    negative_mask = [int(value > 0) for value in negative_multiplicity]
    if negative_event_count:
        negative_weight = [
            float(value) / float(negative_event_count)
            for value in negative_multiplicity
        ]
    else:
        negative_weight = [0.0] * length

    return {
        "raw_stop_count": len(stop_events),
        "raw_atomic_stop_count": raw_atomic_count,
        "ordinary_surface_stop_count": ordinary_surface_count,
        "stop_fragment_count": fragment_count,
        "illegal_surface_stop_count": illegal_surface_count,
        "overflow_atomic_stop_count": overflow_atomic_count,
        "negative_stop_event_count": negative_event_count,
        "stop_event_multiplicity": stop_event_multiplicity,
        "surface_stop_event_multiplicity": surface_stop_event_multiplicity,
        "negative_stop_event_multiplicity": negative_multiplicity,
        "negative_stop_aux_mask": negative_mask,
        "negative_stop_aux_weight": negative_weight,
        "overflow_stop_mask": overflow_stop_mask,
        "atomic_event_ordinals": atomic_ordinals,
    }


def _post_accepted_tail_auxiliary(
    response_ids: Sequence[int],
    accepted_stop_position: Optional[int],
    think_close_ids: Sequence[int],
    negative_stop_aux_mask: Sequence[int],
    atomic_stop_mask: Sequence[int],
    incorrect_stop_aux_mask: Optional[Sequence[int]] = None,
    normalizer_tokens: int = 256,
) -> Tuple[List[int], List[float]]:
    """Penalize only excess thinking after an oracle-accepted stop.

    Required ``</think>`` and final-answer tokens are never penalized.  If the
    close tag is absent, the excess tail extends to the physical response end.
    Every atomic stop action is excluded: later recognized-correct and
    unrecognized probe stops are neutral, while recognized-wrong and
    generalized-negative stop anchors are handled only by their dedicated
    objectives.  For L selected non-stop reasoning actions, total row weight
    is ``min(L / normalizer_tokens, 1)``.
    """

    length = len(response_ids)
    mask = [0] * length
    weight = [0.0] * length
    if len(negative_stop_aux_mask) != length:
        raise ValueError(
            "negative_stop_aux_mask and response_ids must have the same length"
        )
    if any(int(value) not in (0, 1) for value in negative_stop_aux_mask):
        raise ValueError("negative_stop_aux_mask must be binary")
    if len(atomic_stop_mask) != length:
        raise ValueError(
            "atomic_stop_mask and response_ids must have the same length"
        )
    if any(int(value) not in (0, 1) for value in atomic_stop_mask):
        raise ValueError("atomic_stop_mask must be binary")
    if incorrect_stop_aux_mask is None:
        incorrect_stop_aux_mask = [0] * length
    if len(incorrect_stop_aux_mask) != length:
        raise ValueError(
            "incorrect_stop_aux_mask and response_ids must have the same length"
        )
    if any(int(value) not in (0, 1) for value in incorrect_stop_aux_mask):
        raise ValueError("incorrect_stop_aux_mask must be binary")
    normalizer = int(normalizer_tokens)
    if normalizer < 1:
        raise ValueError(
            f"normalizer_tokens must be positive, got {normalizer_tokens}"
        )
    if accepted_stop_position is None:
        return mask, weight
    accepted = int(accepted_stop_position)
    if accepted < 0 or accepted >= length:
        raise ValueError(
            f"accepted stop position {accepted} is outside response length {length}"
        )
    if not int(atomic_stop_mask[accepted]):
        raise ValueError("accepted_stop_position must identify an atomic stop")
    if not think_close_ids:
        raise ValueError("think_close_ids must not be empty")

    tail_start = accepted + 1
    close_start = _find_token_subsequence(response_ids, think_close_ids, tail_start)
    tail_end = close_start if close_start is not None else length
    selected = [
        position
        for position in range(tail_start, tail_end)
        if not int(atomic_stop_mask[position])
        and not int(negative_stop_aux_mask[position])
        and not int(incorrect_stop_aux_mask[position])
    ]
    if not selected:
        return mask, weight
    total_mass = min(float(len(selected)) / float(normalizer), 1.0)
    per_action_weight = total_mass / float(len(selected))
    for position in selected:
        mask[position] = 1
        weight[position] = per_action_weight
    return mask, weight


def _pre_process_inputs(pad_token_id: int, ids: torch.Tensor) -> List[int]:
    values = ids.tolist()
    first = 0
    while first < len(values) and values[first] == pad_token_id:
        first += 1
    return values[first:]


def _object_array_1d(values: Sequence[Any]) -> np.ndarray:
    """Build a batch-shaped object array without NumPy expanding nested values."""

    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


def _classifier_schema_metadata(batch_size: int) -> Dict[str, np.ndarray]:
    """Return batch-shaped A70 provenance accepted by :class:`DataProto`.

    ``DataProto.non_tensor_batch`` does not accept scalar strings, integers,
    or tuples: every value must be a NumPy array whose leading dimension is
    the rollout batch size.  Keep even constant schema provenance row-shaped
    so concatenation, reordering, and consistency checks remain fail-closed.
    """

    batch_size = int(batch_size)
    if batch_size < 0:
        raise ValueError("classifier schema batch_size must be non-negative")
    names = tuple(A70_FEATURES)
    return {
        "classifier_feature_schema": _object_array_1d(["A70"] * batch_size),
        "classifier_feature_schema_version": _object_array_1d(
            [A70_FEATURE_SCHEMA_VERSION] * batch_size
        ),
        "classifier_feature_schema_sha256": _object_array_1d(
            [A70_FEATURE_SCHEMA_SHA256] * batch_size
        ),
        "classifier_feature_dim": np.full(
            batch_size, len(names), dtype=np.int64
        ),
        "classifier_feature_names": _object_array_1d(
            [names for _ in range(batch_size)]
        ),
    }


def _repeat_by_indices(value: Any, indices: Sequence[int]) -> Any:
    """Index a batch-shaped tensor/array/list with candidate-to-base indices."""

    if isinstance(value, torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, np.ndarray):
        return value[np.asarray(indices, dtype=np.int64)]
    if isinstance(value, list):
        return _object_array_1d([value[i] for i in indices])
    return value


def _atomic_token_id(tokenizer, text: str, configured_id: Optional[int] = None) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            f"{text!r} must be one tokenizer token for ESTAR, got {ids}. "
            "Start RL from the repaired SFT checkpoint/tokenizer."
        )
    token_id = int(ids[0])
    if configured_id is not None and int(configured_id) != token_id:
        raise RuntimeError(
            f"configured stop_token_id={configured_id} disagrees with tokenizer "
            f"encoding {text!r} -> {token_id}"
        )
    return token_id


def _choice_token_ids(tokenizer) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"MedQA choice {choice!r} must be one token, got {ids}")
        result[choice] = int(ids[0])
    if len(set(result.values())) != len(CHOICES):
        raise RuntimeError(f"MedQA choice token ids are not distinct: {result}")
    return result


def _canonical_choice_token_ids(tokenizer) -> Dict[str, int]:
    """Return the fixed-prefix ``>A/>B/>C/>D`` one-token IDs.

    A70 is defined from the legacy/canonical boundary pair.  Silently
    falling back to bare A-D would make online features incomparable with the
    offline contract, so the runtime fails closed when a tokenizer lacks the
    canonical single-token spelling.
    """

    result: Dict[str, int] = {}
    for choice in CHOICES:
        ids = tokenizer.encode(">" + choice, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"canonical MedQA choice >{choice!r} must be one token, got {ids}"
            )
        result[choice] = int(ids[0])
    if len(set(result.values())) != len(CHOICES):
        raise RuntimeError(f"canonical choice token ids are not distinct: {result}")
    return result


def _make_probe_sampling_params(choice_ids: Dict[str, int]) -> SamplingParams:
    """Create the fixed-cost, closed-vocabulary MedQA probe configuration."""

    return SamplingParams(
        n=1,
        best_of=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        min_tokens=1,
        # The local vLLM sampler computes constrained-request logprobs after
        # applying allowed_token_ids, so top-4 is exactly the A-D bucket.
        logprobs=4,
        detokenize=False,
        ignore_eos=True,
        allowed_token_ids=list(choice_ids.values()),
    )


def _atomic_stop_positions(
    token_ids: Sequence[int],
    stop_token_id: int,
    max_proposals: Optional[int] = None,
) -> List[int]:
    """Return token indices of genuine atomic stop proposals.

    A legacy three-token spelling that merely decodes to ``<stop>`` does not
    contain ``stop_token_id`` and is deliberately ignored.
    """

    positions = [i for i, token_id in enumerate(token_ids) if int(token_id) == stop_token_id]
    if max_proposals is None:
        return positions
    return positions[: max(0, int(max_proposals))]


def _count_atomic_stop_tokens(token_ids: Sequence[int], stop_token_id: int) -> int:
    """Count every policy-emitted atomic ``<stop>`` in the full rollout."""

    return sum(int(token_id) == int(stop_token_id) for token_id in token_ids)


def _truncate_response_at_first_eos(
    token_ids: Sequence[int],
    eos_token_id: int | Sequence[int],
) -> Tuple[List[int], int]:
    """Keep the first EOS and discard any invalid post-EOS token IDs.

    VERL's response attention mask ends at the first configured EOS.  A vLLM
    output containing IDs after that point must be normalized before reward,
    stop telemetry, and actor masks are built; otherwise those consumers see
    different trajectories and the actor mask can escape the valid response.
    """

    if isinstance(eos_token_id, (int, np.integer)):
        eos_ids = {int(eos_token_id)}
    elif torch.is_tensor(eos_token_id):
        eos_ids = {int(value) for value in eos_token_id.detach().cpu().reshape(-1).tolist()}
    else:
        eos_ids = {int(value) for value in eos_token_id}
    if not eos_ids:
        raise ValueError("eos_token_id must contain at least one token ID")

    response = [int(token_id) for token_id in token_ids]
    for position, token_id in enumerate(response):
        if token_id in eos_ids:
            kept = response[: position + 1]
            return kept, len(response) - len(kept)
    return response, 0


def _response_attention_mask_from_lengths(
    response_lengths: Sequence[int],
    max_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the authoritative valid-token mask for normalized responses.

    The configured EOS set may also contain the tokenizer's padding token.  An
    EOS scan over an already padded tensor then counts the first padding token
    as an inclusive EOS for a max-token response that emitted no real EOS.  At
    this point every response has already been truncated through its first real
    configured EOS and controller suffixes have been spliced as exact token-ID
    lists, so the pre-padding list lengths are the only unambiguous boundary.
    """

    width = int(max_length)
    if width < 0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    normalized_lengths = [int(length) for length in response_lengths]
    if any(length < 0 or length > width for length in normalized_lengths):
        raise ValueError(
            "response length is outside the padded tensor width: "
            f"lengths={normalized_lengths} width={width}"
        )
    lengths = torch.as_tensor(
        normalized_lengths, dtype=torch.long, device=device
    )
    positions = torch.arange(width, dtype=torch.long, device=device).unsqueeze(0)
    return positions.lt(lengths.unsqueeze(1)).to(dtype=dtype)


def _controller_terminal_response(
    candidate_ids: Sequence[int],
    *,
    accepted_stop_position: int,
    accepted_probe_answer: str,
    stop_token_id: int,
    forced_suffix_ids_by_answer: Dict[str, Sequence[int]],
    max_response_length: int,
) -> Tuple[List[int], int, int]:
    """Replace a sampled post-stop suffix with the controller's final answer.

    The returned response contains the sampled policy prefix through the
    accepted atomic ``<stop>`` followed by a forced canonical
    ``</think><final_answer>...`` suffix.  The suffix is part of the valid
    decoded response used by the rule reward, but callers must keep it outside
    every policy/action mask because the controller, not the policy, supplied
    those actions.

    Returns ``(effective_ids, discarded_sampled_tokens, forced_suffix_tokens)``.
    Token IDs are spliced directly; no decode/re-tokenize round trip is used.
    """

    response = [int(token_id) for token_id in candidate_ids]
    position = int(accepted_stop_position)
    answer = str(accepted_probe_answer).upper()
    max_length = int(max_response_length)
    if position < 0 or position >= len(response):
        raise ValueError(
            "accepted_stop_position is outside the sampled response: "
            f"{position} not in [0, {len(response)})"
        )
    if response[position] != int(stop_token_id):
        raise ValueError(
            "accepted_stop_position must identify the atomic stop token"
        )
    if answer not in CHOICES or answer not in forced_suffix_ids_by_answer:
        raise ValueError(f"invalid accepted probe answer: {accepted_probe_answer!r}")
    if max_length < 1:
        raise ValueError(
            f"max_response_length must be positive, got {max_response_length}"
        )

    sampled_prefix = response[: position + 1]
    forced_suffix = [
        int(token_id) for token_id in forced_suffix_ids_by_answer[answer]
    ]
    if not forced_suffix:
        raise ValueError("controller forced suffix must contain at least one token")
    effective = sampled_prefix + forced_suffix
    if len(effective) > max_length:
        raise ValueError(
            "controller-terminal response exceeds max_response_length: "
            f"{len(effective)} > {max_length}"
        )
    return effective, len(response) - len(sampled_prefix), len(forced_suffix)


def _controller_prefix_is_wrap_safe_text(text: object) -> bool:
    """Return whether a sampled prefix can receive one canonical final block.

    Token-subsequence searches are insufficient for this check because BPE can
    merge the closing ``>`` of ``<final_answer>`` with its neighbouring text.
    The controller therefore validates the decoded policy prefix that it will
    actually retain.  Requiring the reward-compatible open thinking structure
    and no pre-existing close/final tags guarantees that appending
    ``</think><final_answer>X</final_answer>`` creates exactly one terminal
    answer instead of duplicating a textual tag that used different token
    boundaries.
    """

    if not isinstance(text, str):
        return False
    stripped = text.lstrip(_ASCII_FORMAT_WHITESPACE)
    return (
        stripped.startswith("<think>")
        and text.count("<think>") == 1
        and text.count("</think>") == 0
        and text.count("<final_answer>") == 0
        and text.count("</final_answer>") == 0
    )


def _controller_wrap_safe_stop_positions(
    tokenizer,
    token_ids: Sequence[int],
    stop_positions: Sequence[int],
    stop_token_id: int,
) -> List[int]:
    """Filter atomic proposals to prefixes that the controller can terminate.

    Decoding is validation-only: the returned trajectory still splices the
    original token IDs directly and never decode/re-tokenizes policy actions.
    """

    response = [int(token_id) for token_id in token_ids]
    positions = [int(position) for position in stop_positions]
    if any(
        position < 0
        or position >= len(response)
        or response[position] != int(stop_token_id)
        for position in positions
    ):
        raise ValueError(
            "controller stop positions must identify atomic stop tokens"
        )
    if any(current <= previous for previous, current in zip(positions, positions[1:])):
        raise ValueError(
            "controller stop positions must be unique and strictly increasing"
        )
    if not positions:
        return []

    prefix_texts = tokenizer.batch_decode(
        [response[: position + 1] for position in positions],
        skip_special_tokens=True,
    )
    if len(prefix_texts) != len(positions):
        raise RuntimeError(
            "tokenizer returned a misaligned controller-prefix decode batch"
        )
    return [
        position
        for position, text in zip(positions, prefix_texts)
        if _controller_prefix_is_wrap_safe_text(text)
    ]


def _find_token_subsequence(
    token_ids: Sequence[int], pattern: Sequence[int], start: int = 0
) -> Optional[int]:
    """Return the first exact token-subsequence start, or ``None``."""

    pattern = list(pattern)
    if not pattern:
        raise ValueError("token subsequence pattern must not be empty")
    first = max(0, int(start))
    last = len(token_ids) - len(pattern)
    for index in range(first, last + 1):
        if all(int(token_ids[index + offset]) == int(value) for offset, value in enumerate(pattern)):
            return index
    return None


def _eligible_stop_positions(
    token_ids: Sequence[int],
    stop_token_id: int,
    think_open_ids: Sequence[int],
    think_close_ids: Sequence[int],
    final_answer_open_ids: Sequence[int],
) -> List[int]:
    """Return stop proposals from the first legal thinking region only.

    Stops before ``<think>`` or after ``</think>``/``<final_answer>`` are not
    termination proposals.  An unclosed first thinking region remains
    eligible up to the end of the rollout.  Probe budgeting is deliberately a
    separate operation so this function always reports the uncapped count.
    """

    think_start = _find_token_subsequence(token_ids, think_open_ids)
    if think_start is None:
        return []
    region_start = think_start + len(think_open_ids)
    boundaries = [
        boundary
        for boundary in (
            _find_token_subsequence(token_ids, think_close_ids, region_start),
            _find_token_subsequence(token_ids, final_answer_open_ids, region_start),
        )
        if boundary is not None
    ]
    region_end = min(boundaries) if boundaries else len(token_ids)
    positions = [
        index
        for index in range(region_start, region_end)
        if int(token_ids[index]) == int(stop_token_id)
    ]
    return positions


def _budgeted_eligible_stop_positions(
    raw_stop_positions: Sequence[int],
    eligible_stop_positions: Sequence[int],
    max_stop_count: int,
    raw_stop_events: Optional[Sequence[_StopEvent]] = None,
) -> List[int]:
    """Keep eligible proposals whose ordinal in the raw stop stream is legal."""

    cap = int(max_stop_count)
    if cap < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")
    raw_positions = [int(position) for position in raw_stop_positions]
    eligible_positions = [int(position) for position in eligible_stop_positions]
    raw_set = set(raw_positions)
    if len(raw_set) != len(raw_positions) or any(
        current <= previous
        for previous, current in zip(raw_positions, raw_positions[1:])
    ):
        raise ValueError(
            "raw_stop_positions must be unique and strictly increasing"
        )
    if any(position not in raw_set for position in eligible_positions):
        raise ValueError(
            "eligible_stop_positions must be a subset of raw_stop_positions"
        )
    if raw_stop_events is None:
        budgeted_positions = set(raw_positions[:cap])
    else:
        atomic_ordinals = {
            int(event.end_token): ordinal
            for ordinal, event in enumerate(raw_stop_events, start=1)
            if event.is_atomic
        }
        if set(atomic_ordinals) != raw_set:
            raise ValueError(
                "raw_stop_events atomic actions disagree with raw_stop_positions"
            )
        budgeted_positions = {
            position
            for position, ordinal in atomic_ordinals.items()
            if ordinal <= cap
        }
    return [
        position
        for position in eligible_positions
        if position in budgeted_positions
    ]


def _exponential_stop_earliness(stop_position: int, half_life_tokens: float) -> float:
    """Score absolute stop position without consulting the rollout tail length."""

    position = int(stop_position)
    half_life = float(half_life_tokens)
    if position < 0:
        raise ValueError(f"stop_position must be non-negative, got {position}")
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError(
            f"stop_reward_half_life_tokens must be finite and positive, got {half_life}"
        )
    return float(2.0 ** (-float(position) / half_life))


def _incorrect_stop_action_metadata(
    probed_stop_positions: Sequence[int],
    probe_records: Sequence[Tuple[Optional[str], np.ndarray, np.ndarray]],
    gold_choice: Optional[str],
    *,
    response_length: int,
    max_stop_count: int,
    half_life_tokens: float,
    positive_override_position: Optional[int] = None,
) -> Dict[str, Any]:
    """Build independent negative-action metadata for wrong oracle probes.

    Every recognized, legal, budgeted atomic proposal whose probe answer is
    different from the gold choice receives its own negative PPO action term.
    Unrecognized probes (``answer is None``) are deliberately neutral.  A
    recognized-wrong probe selected by the final-consistency fallback is also
    neutral in this negative channel because the same action receives a lower
    positive credit through the selected-stop objective.  The
    per-action weights use the same absolute-position earliness function as
    verified-stop credit and are *not* normalized by the number of mistakes in
    the rollout.  Consequently, adding another wrong proposal increases the
    row's total negative mass without weakening earlier penalties.
    """

    length = int(response_length)
    cap = int(max_stop_count)
    if length < 0:
        raise ValueError(
            f"response_length must be non-negative, got {response_length}"
        )
    if cap < 1:
        raise ValueError(
            f"max_stop_count must be positive, got {max_stop_count}"
        )
    if len(probed_stop_positions) != len(probe_records):
        raise ValueError(
            "probe position/record count mismatch: "
            f"{len(probed_stop_positions)} != {len(probe_records)}"
        )
    if len(probed_stop_positions) > cap:
        raise ValueError(
            "probed stop count exceeds the configured attempt budget: "
            f"{len(probed_stop_positions)} > {cap}"
        )

    positions = [int(position) for position in probed_stop_positions]
    if any(position < 0 or position >= length for position in positions):
        raise ValueError(
            "probed stop position is outside the response: "
            f"positions={positions}, response_length={length}"
        )
    if any(
        current <= previous
        for previous, current in zip(positions, positions[1:])
    ):
        raise ValueError(
            "probed stop positions must be unique and strictly increasing: "
            f"{positions}"
        )
    if gold_choice is not None and gold_choice not in CHOICES:
        raise ValueError(f"invalid gold choice: {gold_choice!r}")
    override_position = (
        None
        if positive_override_position is None
        else int(positive_override_position)
    )
    if override_position is not None and override_position not in positions:
        raise ValueError(
            "positive_override_position must identify a probed stop: "
            f"{override_position} not in {positions}"
        )

    mask = [0] * length
    weight = [0.0] * length
    incorrect_positions: List[int] = []
    recognized_correct_positions: List[int] = []
    neutral_positions: List[int] = []
    for position, record in zip(positions, probe_records):
        answer = record[0]
        if answer is None or gold_choice is None:
            neutral_positions.append(position)
            continue
        if answer not in CHOICES:
            raise ValueError(f"invalid recognized probe answer: {answer!r}")
        if answer == gold_choice:
            if position == override_position:
                raise ValueError(
                    "positive_override_position is reserved for a recognized "
                    "non-gold final-consistent probe"
                )
            recognized_correct_positions.append(position)
        elif position == override_position:
            neutral_positions.append(position)
        else:
            mask[position] = 1
            weight[position] = _exponential_stop_earliness(
                position, half_life_tokens
            )
            incorrect_positions.append(position)

    count = len(incorrect_positions)
    if sum(mask) != count or count > cap:
        raise RuntimeError("incorrect-stop action count invariant failed")
    row_weight = float(sum(weight))
    if row_weight < 0.0 or row_weight > float(count) + 1e-12:
        raise RuntimeError(
            "incorrect-stop row weight must be the unnormalized sum of at "
            "most one earliness unit per action"
        )
    recognized_correct_count = len(recognized_correct_positions)
    neutral_count = len(neutral_positions)
    if recognized_correct_count + count + neutral_count != len(positions):
        raise RuntimeError(
            "correct/incorrect/neutral probe counts must partition every "
            "probed stop"
        )
    return {
        "recognized_correct_probe_count": recognized_correct_count,
        "recognized_correct_probe_positions": recognized_correct_positions,
        "incorrect_stop_count": count,
        "incorrect_stop_positions": incorrect_positions,
        "neutral_probe_stop_count": neutral_count,
        "neutral_probe_stop_positions": neutral_positions,
        "incorrect_stop_aux_mask": mask,
        "incorrect_stop_aux_weight": weight,
        "incorrect_stop_aux_weight_sum": row_weight,
    }


def _mask_unverified_stop_actions(
    actor_mask: Sequence[int],
    response_ids: Sequence[int],
    stop_token_id: int,
    verified_stop_position: Optional[int],
    max_stop_count: Optional[int] = None,
    cut_accepted_stop_tail: bool = False,
    negative_stop_aux_mask: Optional[Sequence[int]] = None,
) -> List[int]:
    """Exclude every atomic stop from ordinary trajectory GRPO.

    Verified and overflow actions are optimized only by their independent PPO
    losses.  Failed legal proposals are ignored.  Under controller
    termination, every action after an accepted stop is externally supplied
    and excluded from policy optimization; sequence GRPO is broadcast only to
    sampled ordinary tokens in the accepted prefix.
    """

    if max_stop_count is not None and int(max_stop_count) < 1:
        raise ValueError(
            f"max_stop_count must be positive when provided, got {max_stop_count}"
        )
    if len(actor_mask) != len(response_ids):
        raise ValueError(
            "actor_mask and response_ids must have the same length: "
            f"{len(actor_mask)} != {len(response_ids)}"
        )

    result = list(actor_mask)
    if verified_stop_position is not None:
        verified_position = int(verified_stop_position)
        if verified_position < 0 or verified_position >= len(response_ids):
            raise ValueError(
                "verified_stop_position is outside the response: "
                f"{verified_position} not in [0, {len(response_ids)})"
            )
        if int(response_ids[verified_position]) != int(stop_token_id):
            raise ValueError(
                "verified_stop_position must identify an atomic stop token: "
                f"response_ids[{verified_position}]="
                f"{response_ids[verified_position]}"
            )
    for position, token_id in enumerate(response_ids):
        if int(token_id) == int(stop_token_id):
            result[position] = 0
    if negative_stop_aux_mask is not None:
        if len(negative_stop_aux_mask) != len(response_ids):
            raise ValueError(
                "negative_stop_aux_mask and response_ids must have the same length"
            )
        for position, value in enumerate(negative_stop_aux_mask):
            if int(value) not in (0, 1):
                raise ValueError("negative_stop_aux_mask must be binary")
            if int(value):
                result[position] = 0
    if cut_accepted_stop_tail and verified_stop_position is not None:
        position = int(verified_stop_position)
        result[position + 1 :] = [0] * (len(result) - position - 1)
    return result


def _normalize_verified_stop_positions(
    verified_stop_positions: Optional[Sequence[int] | int],
) -> List[int]:
    """Normalize a scalar/sequence compatibility input to ordered positions."""

    if verified_stop_positions is None:
        return []
    if isinstance(verified_stop_positions, Integral):
        positions = [int(verified_stop_positions)]
    else:
        positions = [int(position) for position in verified_stop_positions]
    if len(positions) > 1:
        raise ValueError(
            "restored v3 credit is earliest-only; at most one verified stop "
            f"position is allowed, got {positions}"
        )
    if any(current <= previous for previous, current in zip(positions, positions[1:])):
        raise ValueError(
            "verified_stop_positions must be unique and strictly increasing, got "
            f"{positions}"
        )
    return positions


def _verified_stop_action_mask(
    response_ids: Sequence[int],
    stop_token_id: int,
    verified_stop_positions: Optional[Sequence[int] | int],
) -> List[int]:
    """Mark every credited oracle-verified stop action."""

    result = [0] * len(response_ids)
    for position in _normalize_verified_stop_positions(verified_stop_positions):
        if position < 0 or position >= len(response_ids):
            raise ValueError(
                "verified stop position is outside the response: "
                f"{position} not in [0, {len(response_ids)})"
            )
        if int(response_ids[position]) != int(stop_token_id):
            raise ValueError(
                "verified stop position must identify an atomic stop token: "
                f"response_ids[{position}]={response_ids[position]}"
            )
        result[position] = 1
    return result


def _overflow_stop_action_mask(
    response_ids: Sequence[int],
    stop_token_id: int,
    max_stop_count: int,
) -> List[int]:
    """Mark only policy stop actions whose ordinal exceeds the legal budget."""

    cap = int(max_stop_count)
    if cap < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")

    result = [0] * len(response_ids)
    stop_ordinal = 0
    for position, token_id in enumerate(response_ids):
        if int(token_id) != int(stop_token_id):
            continue
        stop_ordinal += 1
        if stop_ordinal > cap:
            result[position] = 1
    return result


def _resolve_stop_auxiliary_action_masks(
    response_ids: Sequence[int],
    stop_token_id: int,
    verified_stop_positions: Optional[Sequence[int] | int],
    max_stop_count: int,
) -> Tuple[List[int], List[int], int]:
    """Build disjoint telemetry masks without suppressing overflow diagnostics."""

    verified_mask = _verified_stop_action_mask(
        response_ids, stop_token_id, verified_stop_positions
    )
    overflow_mask = _overflow_stop_action_mask(
        response_ids, stop_token_id, max_stop_count
    )
    if any(
        verified and overflow
        for verified, overflow in zip(verified_mask, overflow_mask)
    ):
        raise ValueError("verified and overflow stop masks must be disjoint")
    return verified_mask, overflow_mask, 0


def _build_full_response(
    candidate_ids: Sequence[int],
    stop_token_id: int,
    verified_stop_position: Optional[int],
    max_stop_count: Optional[int] = None,
    cut_accepted_stop_tail: bool = False,
    negative_stop_aux_mask: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Return the effective response and its sampled-policy action mask."""

    response = list(candidate_ids)
    actor_mask = _mask_unverified_stop_actions(
        [1] * len(response),
        response,
        stop_token_id,
        verified_stop_position,
        max_stop_count=max_stop_count,
        cut_accepted_stop_tail=cut_accepted_stop_tail,
        negative_stop_aux_mask=negative_stop_aux_mask,
    )
    return response, actor_mask


def _extract_base_gold(non_tensor_batch: Dict[str, Any], batch_size: int) -> List[str]:
    reward_model = non_tensor_batch.get("reward_model")
    if reward_model is None:
        return [""] * batch_size
    if isinstance(reward_model, np.ndarray):
        reward_model = reward_model.tolist()
    if (
        isinstance(reward_model, list)
        and len(reward_model) == 1
        and isinstance(reward_model[0], (list, np.ndarray))
    ):
        reward_model = list(reward_model[0])

    gold: List[str] = []
    for i in range(batch_size):
        item = reward_model[i] if isinstance(reward_model, list) and i < len(reward_model) else None
        if isinstance(item, dict):
            value = item.get("ground_truth", "")
        else:
            value = ""
        gold.append(value if isinstance(value, str) else "")
    return gold


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def _probe_answer_and_scores(
    request_output: Any, tokenizer, choice_ids: Dict[str, int]
) -> Tuple[Optional[str], np.ndarray, np.ndarray]:
    """Read a constrained one-token probe and its A-D distribution."""

    output = request_output.outputs[0] if request_output.outputs else None
    scores = np.full(len(CHOICES), -100.0, dtype=np.float64)
    generated_answer: Optional[str] = None
    if output is not None:
        generated_ids = list(getattr(output, "token_ids", []) or [])
        if generated_ids:
            inverse = {token_id: choice for choice, token_id in choice_ids.items()}
            generated_answer = inverse.get(int(generated_ids[0]))
            # Fail closed on the generated token ID.  The constrained probe is
            # only authoritative when vLLM actually sampled one of the four
            # configured singleton A-D IDs; decoding an unrelated token and
            # interpreting its text as a choice would break the controller's
            # token-level contract.

        logprobs = getattr(output, "logprobs", None)
        first_step = logprobs[0] if logprobs else None
        if isinstance(first_step, dict):
            for i, choice in enumerate(CHOICES):
                token_id = choice_ids[choice]
                value = first_step.get(token_id)
                if value is None:
                    value = first_step.get(str(token_id))
                if value is not None:
                    scores[i] = _logprob_value(value)

    finite = np.isfinite(scores) & (scores > -99.0)
    # A partial bucket is not a valid probe. In particular, do not reuse a
    # previous answer or award a proposal from an incomplete/unrecognized
    # diagnostic probe.  With constrained logprobs=4, all four choices are
    # normally present; the probe never changes the completed raw rollout.
    answer: Optional[str] = None
    if finite.all() and generated_answer is not None:
        # The probe's actual generated A-D token is the terminal answer.
        # Log-probabilities are classifier features only and must not silently
        # replace a generated token if a backend ever returns a mismatch.
        answer = generated_answer
    if finite.any():
        floor = float(scores[finite].min() - 20.0)
        scores[~finite] = floor
    else:
        # Treat a numerically invalid probe exactly like an unrecognized one,
        # while keeping downstream classifier features finite.
        scores.fill(0.0)
        probabilities = np.full(len(CHOICES), 1.0 / len(CHOICES), dtype=np.float64)
        return None, scores, probabilities
    max_score = float(np.max(scores))
    probabilities = np.exp(scores - max_score)
    probabilities /= float(probabilities.sum() + 1e-12)
    return answer, scores, probabilities


class EarlyStopFeaturizer:
    """Historical 30-feature helper retained for compatibility tests only.

    Production dual-classifier rollouts use :class:`A70Featurizer` below.
    """

    def __init__(self, recent_window: int = 5):
        self.recent_window = recent_window
        self.cumulative = np.zeros(4, dtype=np.float64)
        self.previous_winner: Optional[int] = None
        self.run_length = 0
        self.flips = 0
        self.margin_history: List[float] = []
        self.cumulative_history: List[np.ndarray] = []
        self.probability_history: List[np.ndarray] = []

    @staticmethod
    def _second_derivative(values: Sequence[float]) -> float:
        if len(values) < 3:
            return 0.0
        y = np.asarray(values, dtype=np.float64)
        t = np.arange(y.size, dtype=np.float64)
        return float(2.0 * np.polyfit(t, y, deg=2)[0])

    @staticmethod
    def _path_stats(history: Sequence[np.ndarray]) -> Tuple[float, float, float]:
        if len(history) < 3:
            return 0.0, 0.0, 0.0
        newest, previous, older = history[-1], history[-2], history[-3]
        velocity = newest - previous
        acceleration = newest - 2.0 * previous + older
        velocity_sq = float(np.dot(velocity, velocity)) + 1e-9
        parallel = (float(np.dot(acceleration, velocity)) / velocity_sq) * velocity
        perpendicular = acceleration - parallel
        return (
            float(np.linalg.norm(velocity)),
            float(np.linalg.norm(acceleration)),
            float(np.linalg.norm(perpendicular) / velocity_sq),
        )

    @staticmethod
    def _fisher_stats(probabilities: Sequence[np.ndarray]) -> Tuple[float, float, float, float]:
        if not probabilities:
            return 0.0, 0.0, 0.0, 0.0
        ps = np.asarray(probabilities, dtype=np.float64)
        ps = np.clip(ps, 1e-12, 1.0)
        ps /= ps.sum(axis=1, keepdims=True)
        fisher = np.zeros((4, 4), dtype=np.float64)
        entropy = 0.0
        for p in ps:
            fisher += np.diag(p) - np.outer(p, p)
            entropy -= float(np.sum(p * np.log(p)))
        fisher /= float(len(ps))
        eigenvalues = np.linalg.eigvalsh((fisher + fisher.T) * 0.5)
        off_diagonal = fisher.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        return (
            float(np.trace(fisher)),
            float(np.max(eigenvalues)),
            float(np.linalg.norm(off_diagonal, ord="fro")),
            float(entropy / len(ps)),
        )

    def step(
        self,
        log_scores: np.ndarray,
        step: int,
        total_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Create features using only evidence available at this probe.

        ``total_steps`` is retained for source compatibility but deliberately
        ignored: the final number of probes is future information at inference
        time.  Feature 9 is instead the causal winner-change indicator from the
        ESTAR-LITE stability family.
        """
        scores = np.asarray(log_scores, dtype=np.float64)
        max_score = float(np.max(scores))
        probabilities = np.exp(scores - max_score)
        probabilities /= float(probabilities.sum() + 1e-12)
        self.cumulative += np.log(np.clip(probabilities, 1e-12, 1.0))

        order = np.argsort(self.cumulative)[::-1]
        winner, runner_up = int(order[0]), int(order[1])
        margin = float(self.cumulative[winner] - self.cumulative[runner_up])
        changed_prev = float(
            self.previous_winner is not None and winner != self.previous_winner
        )
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
        self.margin_history = self.margin_history[-5:]
        self.cumulative_history = self.cumulative_history[-5:]
        self.probability_history = self.probability_history[-5:]

        if previous_margins:
            recent_anchor = previous_margins[max(0, len(previous_margins) - self.recent_window)]
            delta_recent = margin - recent_anchor
            slope_recent = (margin - previous_margins[0]) / max(1, len(previous_margins))
        else:
            delta_recent = 0.0
            slope_recent = 0.0

        curvature_margin = self._second_derivative(self.margin_history)
        curvature_cumulative = [
            self._second_derivative([row[i] for row in self.cumulative_history]) for i in range(4)
        ]
        velocity, acceleration, path_curvature = self._path_stats(self.cumulative_history)
        fisher_trace, fisher_lmax, fisher_offdiag, fisher_entropy = self._fisher_stats(
            self.probability_history
        )
        return np.asarray(
            [
                *self.cumulative.tolist(),
                margin,
                float(self.run_length),
                float(self.flips),
                delta_recent,
                slope_recent,
                changed_prev,
                *scores.tolist(),
                *probabilities.tolist(),
                curvature_margin,
                *curvature_cumulative,
                velocity,
                acceleration,
                path_curvature,
                fisher_trace,
                fisher_lmax,
                fisher_offdiag,
                fisher_entropy,
            ],
            dtype=np.float32,
        )


def build_earlystop_training_data(
    probe_records: Sequence[Sequence[Tuple[Optional[str], np.ndarray, np.ndarray]]],
    physical_final_choices: Sequence[Optional[str]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build one feature/label array per candidate.

    Labels preserve the untouched natural full-rollout answer, matching the
    classifier target in the paper.  Gold answers and controller-rewritten
    effective answers are deliberately not accepted by this function.
    """

    if len(probe_records) != len(physical_final_choices):
        raise ValueError(
            "probe_records and physical_final_choices must have equal length: "
            f"{len(probe_records)} != {len(physical_final_choices)}"
        )

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for records, physical_final_choice in zip(probe_records, physical_final_choices):
        featurizer = EarlyStopFeaturizer()
        rows: List[np.ndarray] = []
        labels: List[int] = []
        for step, (answer, log_scores, _probabilities) in enumerate(records, start=1):
            feature = featurizer.step(log_scores, step)
            if physical_final_choice is not None:
                rows.append(feature)
                labels.append(
                    int(answer is not None and answer == physical_final_choice)
                )
        all_x.append(np.vstack(rows) if rows else np.zeros((0, 30), dtype=np.float32))
        all_y.append(np.asarray(labels, dtype=np.int64))
    return all_x, all_y


def build_dual_earlystop_training_data(
    physical_probe_records: Sequence[
        Sequence[Tuple[Optional[str], np.ndarray, np.ndarray]]
    ],
    physical_final_choices: Sequence[Optional[str]],
    gold_choices: Sequence[str],
    physical_probe_positions: Optional[Sequence[Sequence[int]]] = None,
    reasoning_budget: int = REASONING_BUDGET,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Build both A70 classifier datasets from every untouched physical probe.

    The final-consistency target is defined only when the untouched natural
    response has a strict terminal answer.  The gold-or-final target remains
    defined for every physical probe because each training row has a gold
    answer.  Controller-truncated/effective probe lists must never be passed to
    this function. Production passes physical positions and paired canonical /
    legacy records; the no-position compatibility branch is only for old CPU
    regression tests.
    """

    candidate_count = len(physical_probe_records)
    if len(physical_final_choices) != candidate_count or len(gold_choices) != candidate_count:
        raise ValueError(
            "physical_probe_records, physical_final_choices, and gold_choices "
            "must have equal length"
        )

    final_x: List[np.ndarray] = []
    final_y: List[np.ndarray] = []
    gold_or_final_x: List[np.ndarray] = []
    gold_or_final_y: List[np.ndarray] = []
    valid_choices = {"A", "B", "C", "D"}
    legacy_mode = physical_probe_positions is None and not any(
        isinstance(record, _ProbeRecord)
        for candidate_records in physical_probe_records
        for record in candidate_records
    )
    if physical_probe_positions is None:
        # Compatibility fallback for old unit callers.  Production online
        # rollout always supplies actual token positions; using probe index
        # here is intentionally not allowed to masquerade as physical A70.
        physical_probe_positions = [
            list(range(1, len(records) + 1)) for records in physical_probe_records
        ]
    if len(physical_probe_positions) != candidate_count:
        raise ValueError("physical_probe_positions must align with candidates")
    if int(reasoning_budget) <= 0:
        raise ValueError("reasoning_budget must be positive")
    for records, physical_final_choice, gold_choice, positions in zip(
        physical_probe_records, physical_final_choices, gold_choices, physical_probe_positions
    ):
        if len(positions) != len(records):
            raise ValueError("physical probe positions and records must have equal length")
        if gold_choice not in valid_choices:
            raise ValueError(f"invalid gold choice for classifier labels: {gold_choice!r}")
        if physical_final_choice is not None and physical_final_choice not in valid_choices:
            raise ValueError(
                "invalid untouched physical final choice for classifier labels: "
                f"{physical_final_choice!r}"
            )

        featurizer = (
            EarlyStopFeaturizer()
            if legacy_mode
            else A70Featurizer(reasoning_budget=int(reasoning_budget))
        )
        physical_rows: List[np.ndarray] = []
        labels_final: List[int] = []
        labels_gold_or_final: List[int] = []
        previous_position = 0
        for step, (position, record) in enumerate(zip(positions, records), start=1):
            answer, log_scores, _probabilities = record
            if isinstance(record, _ProbeRecord):
                legacy_scores = record.legacy_scores
            else:
                # A legacy three-tuple is accepted only as a compatibility
                # path; production A70 records always carry both boundaries.
                legacy_scores = log_scores
            if legacy_mode:
                feature = featurizer.step(log_scores, step)
            else:
                feature = featurizer.step(
                    position_tokens=int(position),
                    interval_tokens=int(position) - int(previous_position),
                    probe_index=step,
                    canonical_scores=log_scores,
                    legacy_scores=legacy_scores,
                )
            previous_position = int(position)
            physical_rows.append(feature)
            if physical_final_choice is not None:
                labels_final.append(
                    int(answer is not None and answer == physical_final_choice)
                )
            labels_gold_or_final.append(
                int(
                    answer is not None
                    and (
                        answer == gold_choice
                        or (
                            physical_final_choice is not None
                            and answer == physical_final_choice
                        )
                    )
                )
            )

        all_physical_x = (
            np.vstack(physical_rows)
            if physical_rows
            else np.zeros(
                (0, 30 if legacy_mode else 70), dtype=np.float32
            )
        )
        if physical_final_choice is None:
            final_x.append(
                np.zeros((0, 30 if legacy_mode else 70), dtype=np.float32)
            )
        else:
            final_x.append(all_physical_x.copy())
        final_y.append(np.asarray(labels_final, dtype=np.int64))
        gold_or_final_x.append(all_physical_x)
        gold_or_final_y.append(np.asarray(labels_gold_or_final, dtype=np.int64))

    return final_x, final_y, gold_or_final_x, gold_or_final_y


def _earliest_safe_probe(
    positions: Sequence[int],
    records: Sequence[Tuple[Optional[str], np.ndarray, np.ndarray]],
    gold_choice: Optional[str],
) -> Optional[Tuple[int, str]]:
    """Return ``(token_position, probe_answer)`` for the earliest verified proposal."""

    verified = _spaced_safe_probes(
        positions,
        records,
        gold_choice,
        min_separation_tokens=1,
    )
    return verified[0] if verified else None


@dataclass(frozen=True)
class _SelectedStop:
    """The sole stop action receiving positive auxiliary credit."""

    position: int
    answer: str
    kind: str
    credit: float
    earliness: float
    weight: float


def _select_training_stop(
    positions: Sequence[int],
    probe_answers: Sequence[Optional[str]],
    *,
    gold_choice: Optional[str],
    final_choice: Optional[str],
    response_length: int,
    max_stop_count: int,
    half_life_tokens: float,
    final_consistency_credit: float = DEFAULT_FINAL_CONSISTENCY_CREDIT,
) -> Optional[_SelectedStop]:
    """Select the earliest gold-correct or natural-final-consistent stop.

    ``final_choice`` must be extracted from the untouched physical response.
    Looking at the controller-wrapped response here would make consistency
    circular because the controller writes the selected probe answer itself.
    The supplied positions have already passed v18's unified surface-plus-
    atomic attempt budget; their global event ordinal remains a caller concern.
    """

    if len(positions) != len(probe_answers):
        raise ValueError(
            "probe position/answer count mismatch: "
            f"{len(positions)} != {len(probe_answers)}"
        )
    length = int(response_length)
    cap = int(max_stop_count)
    partial = float(final_consistency_credit)
    if length < 1:
        raise ValueError(
            f"response_length must be positive, got {response_length}"
        )
    if cap < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")
    if len(positions) > cap:
        raise ValueError(
            "probed stop count exceeds the configured unified attempt budget: "
            f"{len(positions)} > {cap}"
        )
    if not math.isfinite(partial) or not 0.0 < partial <= 1.0:
        raise ValueError(
            "final_consistency_credit must be finite in (0, 1], got "
            f"{final_consistency_credit}"
        )
    _exponential_stop_earliness(0, half_life_tokens)

    normalized_positions = [int(position) for position in positions]
    if any(position < 0 or position >= length for position in normalized_positions):
        raise ValueError(
            "probe positions must be inside the physical response: "
            f"positions={normalized_positions}, response_length={length}"
        )
    if any(
        current <= previous
        for previous, current in zip(
            normalized_positions, normalized_positions[1:]
        )
    ):
        raise ValueError(
            "probe positions must be unique and strictly increasing, got "
            f"{normalized_positions}"
        )
    for name, choice in (("gold_choice", gold_choice), ("final_choice", final_choice)):
        if choice is not None and choice not in CHOICES:
            raise ValueError(f"{name} must be one of {CHOICES} or None, got {choice!r}")
    normalized_answers: List[Optional[str]] = []
    for answer in probe_answers:
        if answer is not None and answer not in CHOICES:
            raise ValueError(
                f"probe answers must be one of {CHOICES} or None, got {answer!r}"
            )
        normalized_answers.append(answer)

    selected: Optional[Tuple[int, str, str, float]] = None
    for position, answer in zip(normalized_positions, normalized_answers):
        if answer is None:
            continue
        if gold_choice is not None and answer == gold_choice:
            selected = (position, str(answer), "gold_correct", 1.0)
            break
        if final_choice is not None and answer == final_choice:
            selected = (position, str(answer), "final_consistent", partial)
            break
    if selected is None:
        return None

    position, answer, kind, credit = selected
    earliness = _exponential_stop_earliness(position, half_life_tokens)
    # Both accepted semantic classes receive exactly the same bounded score.
    # A short half-life configured by the launcher makes the difference
    # between an early and a late stop deliberately large.
    weight = earliness
    return _SelectedStop(
        position=position,
        answer=answer,
        kind=kind,
        credit=credit,
        earliness=earliness,
        weight=weight,
    )


def _spaced_safe_probes(
    positions: Sequence[int],
    records: Sequence[Tuple[Optional[str], np.ndarray, np.ndarray]],
    gold_choice: Optional[str],
    *,
    min_separation_tokens: int,
) -> List[Tuple[int, str]]:
    """Return all credited correct probes with token-position de-duplication.

    Only a correct probe can advance ``last_credited_position``.  Therefore an
    incorrect or unrecognized proposal never blocks a later correct proposal,
    while every raw atomic stop still consumes the global stop budget.
    """

    separation = int(min_separation_tokens)
    if separation < 1:
        raise ValueError(
            "min_separation_tokens must be positive, got "
            f"{min_separation_tokens}"
        )
    if len(positions) != len(records):
        raise ValueError(
            "probe position/record count mismatch: "
            f"{len(positions)} != {len(records)}"
        )
    normalized_positions = [int(position) for position in positions]
    if any(
        current <= previous
        for previous, current in zip(
            normalized_positions, normalized_positions[1:]
        )
    ):
        raise ValueError(
            "probe positions must be unique and strictly increasing, got "
            f"{normalized_positions}"
        )
    if gold_choice is None:
        return []

    credited: List[Tuple[int, str]] = []
    last_credited_position: Optional[int] = None
    for position, record in zip(normalized_positions, records):
        answer = record[0]
        if answer is None or answer != gold_choice:
            continue
        if (
            last_credited_position is not None
            and position - last_credited_position < separation
        ):
            continue
        credited.append((position, answer))
        last_credited_position = position
    return credited


def _gate_safe_probes_for_overflow(
    safe_probes: Sequence[Tuple[int, str]],
    *,
    raw_stop_count: int,
    max_stop_count: int,
) -> Tuple[List[Tuple[int, str]], int]:
    """Validate overflow metadata without suppressing a legal verified stop.

    Only proposals whose raw ordinal is within ``max_stop_count`` are probed,
    so a verified action passed here is necessarily legal.  Later overflow
    stops receive their own negative action loss and do not revoke that credit.
    """

    raw_count = int(raw_stop_count)
    cap = int(max_stop_count)
    if raw_count < 0:
        raise ValueError(f"raw_stop_count must be non-negative, got {raw_stop_count}")
    if cap < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")
    normalized = [(int(position), str(answer)) for position, answer in safe_probes]
    return normalized, 0


def _configured_max_num_seqs(config: DictConfig) -> int:
    """Resolve and validate the scheduler concurrency passed to vLLM."""

    max_num_seqs = int(config.get("max_num_seqs", 256))
    if max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
    return max_num_seqs


def _required_stop_configuration(config: DictConfig) -> Tuple[int, float, int, float]:
    """Return the stop budget, earliness scale, spacing, and fallback credit."""

    configured_max_stop_count = config.get("max_stop_count", None)
    if configured_max_stop_count is None:
        raise ValueError("max_stop_count must be configured explicitly")
    if config.get("max_stop_proposals", None) is not None:
        raise ValueError(
            "max_stop_proposals was removed; max_stop_count is the single probe budget"
        )
    max_stop_count = int(configured_max_stop_count)
    if max_stop_count < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")

    configured_half_life = config.get("stop_reward_half_life_tokens", None)
    if configured_half_life is None:
        raise ValueError("stop_reward_half_life_tokens must be configured explicitly")
    half_life = float(configured_half_life)
    _exponential_stop_earliness(0, half_life)

    configured_separation = config.get(
        "min_verified_stop_separation_tokens", None
    )
    if configured_separation is None:
        raise ValueError(
            "min_verified_stop_separation_tokens must be configured explicitly"
        )
    min_separation = int(configured_separation)
    if min_separation < 1:
        raise ValueError(
            "min_verified_stop_separation_tokens must be positive, got "
            f"{min_separation}"
        )
    final_consistency_credit = float(
        config.get(
            "final_consistency_credit",
            DEFAULT_FINAL_CONSISTENCY_CREDIT,
        )
    )
    if (
        not math.isfinite(final_consistency_credit)
        or not 0.0 < final_consistency_credit <= 1.0
    ):
        raise ValueError(
            "final_consistency_credit must be finite in (0, 1], got "
            f"{final_consistency_credit}"
        )
    return max_stop_count, half_life, min_separation, final_consistency_credit


class StopvLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__()
        self.config = config
        if not config.enforce_eager and config.free_cache_engine:
            raise AssertionError("disable CUDA graph when free_cache_engine is enabled")

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        if tensor_parallel_size > torch.distributed.get_world_size():
            raise AssertionError("tensor parallel size must be <= distributed world size")
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 9192)
        max_num_seqs = _configured_max_num_seqs(self.config)
        self.tokenizer = tokenizer
        configured_stop_id = config.get("stop_token_id", None)
        self.stop_token_id = _atomic_token_id(tokenizer, "<stop>", configured_stop_id)
        self.stop_surface_scanner = _ByteLevelStopScanner(
            tokenizer, self.stop_token_id
        )
        # The canonical boundary is not interchangeable with the legacy
        # ``<final_answer>`` boundary: A70 deliberately compares both.
        self.choice_ids = _choice_token_ids(tokenizer)
        self.canonical_choice_ids = _canonical_choice_token_ids(tokenizer)
        (
            self.max_stop_count,
            self.stop_reward_half_life_tokens,
            self.min_verified_stop_separation_tokens,
            self.final_consistency_credit,
        ) = _required_stop_configuration(config)
        configured_tail_normalizer = config.get(
            "post_accepted_tail_penalty_normalization_tokens", None
        )
        if configured_tail_normalizer is None:
            raise ValueError(
                "post_accepted_tail_penalty_normalization_tokens must be "
                "configured explicitly"
            )
        self.post_accepted_tail_penalty_normalization_tokens = int(
            configured_tail_normalizer
        )
        if self.post_accepted_tail_penalty_normalization_tokens < 1:
            raise ValueError(
                "post_accepted_tail_penalty_normalization_tokens must be "
                "positive, got "
                f"{configured_tail_normalizer}"
            )
        cut_accepted_stop_tail = config.get("cut_accepted_stop_tail", False)
        if isinstance(cut_accepted_stop_tail, str):
            normalized = cut_accepted_stop_tail.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    "cut_accepted_stop_tail must be a boolean, got "
                    f"{cut_accepted_stop_tail!r}"
                )
            cut_accepted_stop_tail = normalized == "true"
        self.cut_accepted_stop_tail = bool(cut_accepted_stop_tail)
        if not self.cut_accepted_stop_tail:
            raise ValueError(
                "controller-terminal rollout requires cut_accepted_stop_tail=true"
            )
        terminate_on_stop_overflow = config.get(
            "terminate_on_stop_overflow", False
        )
        if isinstance(terminate_on_stop_overflow, str):
            normalized = terminate_on_stop_overflow.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    "terminate_on_stop_overflow must be a boolean, got "
                    f"{terminate_on_stop_overflow!r}"
                )
            terminate_on_stop_overflow = normalized == "true"
        self.terminate_on_stop_overflow = bool(terminate_on_stop_overflow)
        if self.terminate_on_stop_overflow:
            raise ValueError(
                "terminate_on_stop_overflow must be false: every stop after "
                "max_stop_count needs to remain in the rollout for its own "
                "negative action loss"
            )
        self.legacy_probe_prefix_ids = tokenizer.encode(
            "</think>\n<final_answer>", add_special_tokens=False
        )
        self.canonical_probe_prefix_ids = tokenizer.encode(
            "</think>\n<final_answer", add_special_tokens=False
        )
        if not self.legacy_probe_prefix_ids or not self.canonical_probe_prefix_ids:
            raise RuntimeError("A70 probe boundary prefixes must tokenize non-empty")
        # Retain the historical attribute for callers that only need a
        # conservative context-length estimate; all actual probes below use
        # the two explicit prefixes.
        self.probe_prefix_ids = self.legacy_probe_prefix_ids

        if kwargs.get("train_tp") is not None:
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs["train_tp"]
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=train_tp // tensor_parallel_size,
            )

        required_model_len = (
            int(config.prompt_length)
            + int(config.response_length)
            + max(len(self.legacy_probe_prefix_ids), len(self.canonical_probe_prefix_ids))
            + 1
        )
        if model_hf_config.max_position_embeddings < required_model_len:
            raise AssertionError("model context length must cover prompt + response + probe")
        max_model_len = int(
            self.config.max_model_len
            if self.config.max_model_len
            else required_model_len
        )
        if max_model_len < required_model_len:
            raise ValueError(
                f"max_model_len={max_model_len} is smaller than required probe context "
                f"length {required_model_len}"
            )
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "enable_chunked_prefill requires max_num_batched_tokens >= max_model_len"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # vLLM 0.17.1 in the Unity image has no disable_mm_preprocessor_cache
            # EngineArgs field; leaving this legacy Gamma-only kwarg out preserves
            # the same rollout behavior while allowing the image API to initialize.
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )
        self.inference_engine.sleep(level=1)

        main_kwargs: Dict[str, Any] = {
            "n": 1,
            "logprobs": 0,
            "max_tokens": config.response_length,
        }
        if vllm_version != "0.3.1":
            main_kwargs["detokenize"] = False
        for key in config.keys():
            if hasattr(SamplingParams(), str(key)):
                main_kwargs[key] = config.get(key)

        extra_args = dict(main_kwargs.get("extra_args") or {})
        # v12 must retain the complete natural rollout: every stop after the
        # budget receives a negative action loss.  Strip stale v11 termination
        # hooks even if they arrived through SamplingParams.extra_args.
        extra_args.pop("estar_atomic_stop_token_id", None)
        extra_args.pop("estar_atomic_stop_max_count", None)
        if extra_args:
            main_kwargs["extra_args"] = extra_args
        else:
            main_kwargs.pop("extra_args", None)
        self.sampling_params = SamplingParams(**main_kwargs)
        self.think_open_ids = tokenizer.encode("<think>", add_special_tokens=False)
        self.think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.final_answer_open_ids = tokenizer.encode(
            "<final_answer>", add_special_tokens=False
        )
        self.final_answer_close_ids = tokenizer.encode(
            "</final_answer>", add_special_tokens=False
        )
        if (
            not self.think_open_ids
            or not self.think_close_ids
            or not self.final_answer_open_ids
            or not self.final_answer_close_ids
        ):
            raise RuntimeError("ESTAR format tags must tokenize to non-empty sequences")

        controller_eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(controller_eos_token_id, Integral):
            raise RuntimeError(
                "controller-terminal wrapping requires one integer EOS token ID"
            )
        self.controller_eos_token_id = int(controller_eos_token_id)
        self.controller_forced_suffix_ids_by_answer = {}
        for choice in CHOICES:
            suffix_text = (
                f"</think>\n<final_answer>{choice}</final_answer>"
            )
            suffix_without_eos = tokenizer.encode(
                suffix_text, add_special_tokens=False
            )
            decoded_suffix = tokenizer.decode(
                suffix_without_eos, skip_special_tokens=True
            )
            if (
                decoded_suffix != suffix_text
                or extract_strict_terminal_choice(decoded_suffix) != choice
            ):
                raise RuntimeError(
                    "controller canonical final suffix failed tokenizer "
                    f"round-trip for choice {choice}: {decoded_suffix!r}"
                )
            self.controller_forced_suffix_ids_by_answer[choice] = (
                list(suffix_without_eos) + [self.controller_eos_token_id]
            )
        if any(
            not suffix
            for suffix in self.controller_forced_suffix_ids_by_answer.values()
        ):
            raise RuntimeError("controller-terminal suffix tokenization is empty")
        self.controller_max_forced_suffix_tokens = max(
            len(suffix)
            for suffix in self.controller_forced_suffix_ids_by_answer.values()
        )
        self.controller_main_max_tokens = (
            int(self.config.response_length)
            - self.controller_max_forced_suffix_tokens
        )
        if self.controller_main_max_tokens < 1:
            raise RuntimeError(
                "response_length cannot reserve the controller final suffix"
            )
        # Reserve room inside the fixed VERL response tensor.  This affects all
        # physical main rollouts by only the short canonical suffix length and
        # guarantees that even a stop at the last sampled token can be wrapped
        # without dropping sampled prefix actions.
        self.sampling_params.max_tokens = self.controller_main_max_tokens

        self.legacy_probe_sampling_params = _make_probe_sampling_params(self.choice_ids)
        self.canonical_probe_sampling_params = _make_probe_sampling_params(
            self.canonical_choice_ids
        )
        # Historical callers may inspect this name.  It now aliases the
        # legacy half only; generate_sequences always invokes both params.
        self.probe_sampling_params = self.legacy_probe_sampling_params
        self.classifier_reasoning_budget = int(
            config.get("classifier_reasoning_budget", REASONING_BUDGET)
        )
        if self.classifier_reasoning_budget <= 0:
            raise ValueError("classifier_reasoning_budget must be positive")
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        previous: Dict[str, Any] = {}
        try:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    previous[key] = getattr(self.sampling_params, key)
                    setattr(self.sampling_params, key, value)
            yield
        finally:
            for key, value in previous.items():
                setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        base_batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch

        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = _object_array_1d(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(base_batch_size)]
            )
        if base_batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vLLM sharding manager returned a misaligned prompt batch")
        base_prompt_ids = [
            value.tolist() if isinstance(value, np.ndarray) else list(value)
            for value in non_tensor_batch["raw_prompt_ids"]
        ]
        base_gold_texts = _extract_base_gold(non_tensor_batch, base_batch_size)

        if "multi_modal_data" in non_tensor_batch:
            raw_prompts = non_tensor_batch.pop("raw_prompt_ids")
            multimodal = non_tensor_batch.pop("multi_modal_data")
            vllm_inputs = [
                {
                    "prompt_token_ids": rp.tolist() if isinstance(rp, np.ndarray) else list(rp),
                    "multi_modal_data": mm,
                }
                for rp, mm in zip(raw_prompts, multimodal)
            ]
        else:
            raw_prompts = non_tensor_batch.pop("raw_prompt_ids")
            vllm_inputs = [
                {"prompt_token_ids": rp.tolist() if isinstance(rp, np.ndarray) else list(rp)}
                for rp in raw_prompts
            ]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            generation_overrides = {
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0.0,
                "n": 1,
            }
        elif is_validate:
            generation_overrides = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            generation_overrides = {}

        with self.update_sampling_params(**generation_overrides):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
            )

        cand_base_idx = [base for base, output in enumerate(outputs) for _ in output.outputs]
        cand_rollout_idx = [
            rollout_index
            for output in outputs
            for rollout_index, _ in enumerate(output.outputs)
        ]
        candidate_samples = [sample for output in outputs for sample in output.outputs]
        raw_candidate_ids = [list(sample.token_ids) for sample in candidate_samples]
        candidate_count = len(raw_candidate_ids)
        if candidate_count == 0:
            raise RuntimeError("vLLM returned no rollout candidates")
        normalized_candidates = [
            _truncate_response_at_first_eos(ids, eos_token_id)
            for ids in raw_candidate_ids
        ]
        physical_candidate_ids = [result[0] for result in normalized_candidates]
        eos_trimmed_token_counts = [result[1] for result in normalized_candidates]
        if any(not ids for ids in physical_candidate_ids):
            raise RuntimeError(
                "vLLM returned an empty physical response; controller/reward "
                "terminal indexing requires at least one generated token"
            )

        configured_eos_ids = (
            {int(eos_token_id)}
            if isinstance(eos_token_id, (int, np.integer))
            else {
                int(value)
                for value in (
                    eos_token_id.detach().cpu().reshape(-1).tolist()
                    if torch.is_tensor(eos_token_id)
                    else eos_token_id
                )
            }
        )
        if self.controller_eos_token_id not in configured_eos_ids:
            raise RuntimeError(
                "controller EOS token is not recognized by the response mask: "
                f"{self.controller_eos_token_id} not in {configured_eos_ids}"
            )
        for choice, suffix_ids in self.controller_forced_suffix_ids_by_answer.items():
            if (
                int(suffix_ids[-1]) != self.controller_eos_token_id
                or any(int(token_id) in configured_eos_ids for token_id in suffix_ids[:-1])
            ):
                raise RuntimeError(
                    "controller forced suffix must contain a configured EOS only "
                    f"at its final token: choice={choice} suffix={suffix_ids}"
                )

        # Build one policy-ordered event stream.  Surface spellings/fragments
        # consume the same proposal budget as atomic stops and are always
        # negative actions; only atomic events remain probe candidates.  These
        # physical streams are used only to issue probes.  Once a probe is
        # accepted, all later sampled events are outside the deployed
        # controller trajectory and are rebuilt from the effective response.
        physical_raw_stop_events = [
            self.stop_surface_scanner.scan(ids) for ids in physical_candidate_ids
        ]
        physical_raw_stop_positions = [
            _atomic_stop_positions(ids, self.stop_token_id)
            for ids in physical_candidate_ids
        ]
        physical_stop_event_metadata = [
            _stop_event_action_metadata(
                events,
                response_length=len(ids),
                max_stop_count=self.max_stop_count,
            )
            for events, ids in zip(
                physical_raw_stop_events, physical_candidate_ids
            )
        ]
        physical_raw_stop_counts = [
            int(metadata["raw_stop_count"])
            for metadata in physical_stop_event_metadata
        ]
        candidate_gold_texts = [base_gold_texts[base] for base in cand_base_idx]
        gold_choices = [extract_medqa_choice(text) for text in candidate_gold_texts]
        physical_token_eligible_stop_positions = [
            _eligible_stop_positions(
                ids,
                self.stop_token_id,
                self.think_open_ids,
                self.think_close_ids,
                self.final_answer_open_ids,
            )
            for ids in physical_candidate_ids
        ]
        physical_budgeted_token_eligible_stop_positions = []
        for raw_positions, token_eligible_positions, events in zip(
            physical_raw_stop_positions,
            physical_token_eligible_stop_positions,
            physical_raw_stop_events,
        ):
            physical_budgeted_token_eligible_stop_positions.append(
                _budgeted_eligible_stop_positions(
                    raw_positions,
                    token_eligible_positions,
                    self.max_stop_count,
                    raw_stop_events=events,
                )
            )
        physical_eligible_stop_positions = [
            _controller_wrap_safe_stop_positions(
                self.tokenizer,
                ids,
                positions,
                self.stop_token_id,
            )
            for ids, positions in zip(
                physical_candidate_ids,
                physical_budgeted_token_eligible_stop_positions,
            )
        ]
        controller_wrap_rejected_stop_counts = [
            len(budgeted_token_eligible) - len(wrap_safe)
            for budgeted_token_eligible, wrap_safe in zip(
                physical_budgeted_token_eligible_stop_positions,
                physical_eligible_stop_positions,
            )
        ]
        # Probe only eligible atomic stops whose ordinal in the unified event
        # stream is within the first max_stop_count attempts.
        physical_probed_stop_positions = [
            list(positions) for positions in physical_eligible_stop_positions
        ]

        flat_owners: List[int] = []
        probe_inputs: List[Dict[str, List[int]]] = []
        for candidate, positions in enumerate(physical_probed_stop_positions):
            base = cand_base_idx[candidate]
            for position in positions:
                causal_prefix = (
                    base_prompt_ids[base]
                    + physical_candidate_ids[candidate][: position + 1]
                )
                # Keep legacy then canonical ordering stable so the two
                # responses can be paired without relying on vLLM request
                # metadata.  Both are one-token constrained probes with
                # logprobs=4; neither changes the physical rollout.
                flat_owners.extend((candidate, candidate))
                probe_inputs.extend(
                    [
                        {"prompt_token_ids": causal_prefix + self.legacy_probe_prefix_ids},
                        {"prompt_token_ids": causal_prefix + self.canonical_probe_prefix_ids},
                    ]
                )

        records_by_candidate: List[List[_ProbeRecord]] = [
            [] for _ in range(candidate_count)
        ]
        if probe_inputs:
            # vLLM accepts one SamplingParams object per request only through
            # separate calls.  Split the interleaved list into the two fixed
            # boundary batches and then pair by candidate/position index.
            legacy_indices = list(range(0, len(probe_inputs), 2))
            canonical_indices = list(range(1, len(probe_inputs), 2))
            legacy_outputs = self.inference_engine.generate(
                prompts=[probe_inputs[i] for i in legacy_indices],
                sampling_params=self.legacy_probe_sampling_params,
                use_tqdm=False,
            )
            canonical_outputs = self.inference_engine.generate(
                prompts=[probe_inputs[i] for i in canonical_indices],
                sampling_params=self.canonical_probe_sampling_params,
                use_tqdm=False,
            )
            expected = len(legacy_indices)
            if len(legacy_outputs) != expected or len(canonical_outputs) != expected:
                raise RuntimeError(
                    "A70 dual-boundary probe output count mismatch: "
                    f"legacy={len(legacy_outputs)}/{expected}, "
                    f"canonical={len(canonical_outputs)}/{expected}"
                )
            for pair_index, (legacy_output, canonical_output) in enumerate(
                zip(legacy_outputs, canonical_outputs)
            ):
                owner = flat_owners[2 * pair_index]
                if flat_owners[2 * pair_index + 1] != owner:
                    raise RuntimeError("A70 probe owner pairing drift")
                legacy = _probe_answer_and_scores(
                    legacy_output, self.tokenizer, self.choice_ids
                )
                canonical = _probe_answer_and_scores(
                    canonical_output, self.tokenizer, self.canonical_choice_ids
                )
                records_by_candidate[owner].append(
                    _ProbeRecord(
                        answer=canonical[0],
                        canonical_scores=canonical[1],
                        canonical_probabilities=canonical[2],
                        legacy_scores=legacy[1],
                        legacy_probabilities=legacy[2],
                    )
                )
        # Immutable logical snapshot for classifier supervision.  The lists
        # below are never shortened when the controller materializes its
        # effective trajectory.
        physical_records_by_candidate = [
            list(records) for records in records_by_candidate
        ]

        # Resolve the policy's strict natural final before selecting a stop.
        # This must use the untouched physical response: the controller closure
        # below deliberately copies the selected probe answer and therefore
        # cannot be used as independent consistency evidence.
        physical_response_lengths = [len(ids) for ids in physical_candidate_ids]
        physical_raw_atomic_stop_counts = [
            int(metadata["raw_atomic_stop_count"])
            for metadata in physical_stop_event_metadata
        ]
        physical_illegal_surface_stop_counts = [
            int(metadata["illegal_surface_stop_count"])
            for metadata in physical_stop_event_metadata
        ]
        physical_candidate_texts = self.tokenizer.batch_decode(
            physical_candidate_ids, skip_special_tokens=True
        )
        physical_final_choices = [
            extract_strict_terminal_choice(text)
            for text in physical_candidate_texts
        ]
        physical_policy_acc = [
            int(final_choice is not None and final_choice == gold_choice)
            for final_choice, gold_choice in zip(
                physical_final_choices, gold_choices
            )
        ]
        # Freeze classifier examples before the controller can truncate the
        # effective trajectory.  Both classifiers see every physical probe;
        # only final-consistency rows with an undefined strict physical final
        # are omitted because that target is undefined.
        (
            classifier_final_feature_x,
            classifier_final_feature_y,
            classifier_gold_or_final_feature_x,
            classifier_gold_or_final_feature_y,
        ) = build_dual_earlystop_training_data(
            physical_records_by_candidate,
            physical_final_choices,
            gold_choices,
            physical_probe_positions=physical_probed_stop_positions,
            reasoning_budget=self.classifier_reasoning_budget,
        )

        early_stop_scores = [0.0] * candidate_count
        # Controller acceptance is the terminal transition.  The earliest
        # proposal matching either gold or the independent natural final wins;
        # both semantic classes receive the same earliness score.
        oracle_accepted_positions: List[Optional[int]] = [None] * candidate_count
        oracle_accepted_answers: List[Optional[str]] = [None] * candidate_count
        oracle_accepted_stop_ordinals = [-1] * candidate_count
        selected_positions: List[Optional[int]] = [None] * candidate_count
        selected_answers: List[Optional[str]] = [None] * candidate_count
        selected_kinds = ["none"] * candidate_count
        selected_credits = [0.0] * candidate_count
        credited_positions: List[List[int]] = [
            [] for _ in range(candidate_count)
        ]
        verified_stops = [0] * candidate_count
        verified_stop_counts = [0] * candidate_count
        selected_stop_ordinals = [-1] * candidate_count
        first_probe_correct = [0] * candidate_count
        output_suppressed_verified_stop_counts = [0] * candidate_count

        for candidate, (
            positions,
            records,
            gold_choice,
            final_choice,
            raw_stop_count,
        ) in enumerate(
            zip(
                physical_probed_stop_positions,
                records_by_candidate,
                gold_choices,
                physical_final_choices,
                physical_raw_stop_counts,
            )
        ):
            if records:
                first_probe_correct[candidate] = int(
                    records[0][0] is not None and records[0][0] == gold_choice
                )
            selection = _select_training_stop(
                positions,
                [record[0] for record in records],
                gold_choice=gold_choice,
                final_choice=final_choice,
                response_length=len(physical_candidate_ids[candidate]),
                max_stop_count=self.max_stop_count,
                half_life_tokens=self.stop_reward_half_life_tokens,
                final_consistency_credit=self.final_consistency_credit,
            )
            if selection is None:
                continue

            safe_probes, suppressed_count = _gate_safe_probes_for_overflow(
                [(selection.position, selection.answer)],
                raw_stop_count=raw_stop_count,
                max_stop_count=self.max_stop_count,
            )
            oracle_position, oracle_answer = safe_probes[0]

            # Validate the exact token-ID splice before granting any stop
            # credit.  Prefix filtering above should make this a no-op, but
            # this candidate-local fallback prevents one malformed response
            # from aborting the entire rollout batch if a tokenizer exposes a
            # new context-sensitive tag segmentation.
            proposed_effective_ids, _, _ = _controller_terminal_response(
                physical_candidate_ids[candidate],
                accepted_stop_position=oracle_position,
                accepted_probe_answer=oracle_answer,
                stop_token_id=self.stop_token_id,
                forced_suffix_ids_by_answer=(
                    self.controller_forced_suffix_ids_by_answer
                ),
                max_response_length=self.config.response_length,
            )
            proposed_text = self.tokenizer.decode(
                proposed_effective_ids, skip_special_tokens=True
            )
            if extract_strict_terminal_choice(proposed_text) != oracle_answer:
                controller_wrap_rejected_stop_counts[candidate] += 1
                continue

            oracle_accepted_positions[candidate] = oracle_position
            oracle_accepted_answers[candidate] = oracle_answer
            oracle_accepted_stop_ordinals[candidate] = int(
                physical_stop_event_metadata[candidate]["atomic_event_ordinals"][
                    oracle_position
                ]
            )

            stop_position, probe_answer = safe_probes[0]
            selected_positions[candidate] = stop_position
            selected_answers[candidate] = probe_answer
            selected_kinds[candidate] = selection.kind
            selected_credits[candidate] = selection.credit
            credited_positions[candidate] = [stop_position]
            verified_stops[candidate] = 1
            verified_stop_counts[candidate] = 1
            selected_stop_ordinals[candidate] = int(
                physical_stop_event_metadata[candidate]["atomic_event_ordinals"][
                    stop_position
                ]
            )
            early_stop_scores[candidate] = selection.weight

        # Materialize the authoritative controller trajectory.  Accepted rows
        # retain sampled IDs through the accepted stop and receive a forced,
        # masked canonical final answer.  Non-accepted rows remain byte-for-byte
        # the physical policy rollout.
        controller_terminated: List[int] = []
        controller_discarded_token_counts: List[int] = []
        controller_forced_suffix_token_counts: List[int] = []
        candidate_ids: List[List[int]] = []
        for candidate, physical_ids in enumerate(physical_candidate_ids):
            accepted_position = oracle_accepted_positions[candidate]
            accepted_answer = oracle_accepted_answers[candidate]
            if accepted_position is None:
                candidate_ids.append(list(physical_ids))
                controller_terminated.append(0)
                controller_discarded_token_counts.append(0)
                controller_forced_suffix_token_counts.append(0)
                continue
            if accepted_answer is None:
                raise RuntimeError(
                    "controller accepted a stop without a probe answer"
                )
            effective_ids, discarded_count, forced_count = (
                _controller_terminal_response(
                    physical_ids,
                    accepted_stop_position=accepted_position,
                    accepted_probe_answer=accepted_answer,
                    stop_token_id=self.stop_token_id,
                    forced_suffix_ids_by_answer=(
                        self.controller_forced_suffix_ids_by_answer
                    ),
                    max_response_length=self.config.response_length,
                )
            )
            candidate_ids.append(effective_ids)
            controller_terminated.append(1)
            controller_discarded_token_counts.append(discarded_count)
            controller_forced_suffix_token_counts.append(forced_count)

        # All policy/reward metadata below is rebuilt from the effective
        # trajectory.  Physical telemetry is retained under explicit names so
        # late async tokens can never leak into action penalties or format
        # checks.
        raw_stop_events = [
            self.stop_surface_scanner.scan(ids) for ids in candidate_ids
        ]
        raw_stop_positions = [
            _atomic_stop_positions(ids, self.stop_token_id)
            for ids in candidate_ids
        ]
        stop_event_metadata = [
            _stop_event_action_metadata(
                events,
                response_length=len(ids),
                max_stop_count=self.max_stop_count,
            )
            for events, ids in zip(raw_stop_events, candidate_ids)
        ]
        raw_stop_counts = [
            int(metadata["raw_stop_count"]) for metadata in stop_event_metadata
        ]
        eligible_stop_positions = [
            _eligible_stop_positions(
                ids,
                self.stop_token_id,
                self.think_open_ids,
                self.think_close_ids,
                self.final_answer_open_ids,
            )
            for ids in candidate_ids
        ]
        probed_stop_positions: List[List[int]] = []
        effective_records_by_candidate: List[
            List[Tuple[Optional[str], np.ndarray, np.ndarray]]
        ] = []
        for candidate, (positions, records) in enumerate(
            zip(physical_probed_stop_positions, physical_records_by_candidate)
        ):
            accepted_position = oracle_accepted_positions[candidate]
            keep_count = (
                len(positions)
                if accepted_position is None
                else sum(
                    int(position) <= int(accepted_position)
                    for position in positions
                )
            )
            effective_positions = list(positions[:keep_count])
            effective_records = list(records[:keep_count])
            if any(
                position not in eligible_stop_positions[candidate]
                for position in effective_positions
            ):
                raise RuntimeError(
                    "effective probed stop is not eligible after controller wrap"
                )
            probed_stop_positions.append(effective_positions)
            effective_records_by_candidate.append(effective_records)
        records_by_candidate = effective_records_by_candidate

        candidate_texts = self.tokenizer.batch_decode(
            candidate_ids, skip_special_tokens=True
        )
        final_choices = [
            extract_strict_terminal_choice(text) for text in candidate_texts
        ]
        raw_policy_acc = [
            int(final_choice is not None and final_choice == gold_choice)
            for final_choice, gold_choice in zip(final_choices, gold_choices)
        ]
        for candidate, terminated in enumerate(controller_terminated):
            if not terminated:
                continue
            if (
                final_choices[candidate] is None
                or final_choices[candidate]
                != oracle_accepted_answers[candidate]
            ):
                raise RuntimeError(
                    "controller final wrapper did not preserve the accepted "
                    "probe answer as a strict terminal answer"
                )
            selected_kind = selected_kinds[candidate]
            if selected_kind == "gold_correct":
                if raw_policy_acc[candidate] != 1:
                    raise RuntimeError(
                        "gold-correct controller stop must produce a strict "
                        "correct final answer"
                    )
            elif selected_kind == "final_consistent":
                if (
                    physical_final_choices[candidate] is None
                    or oracle_accepted_answers[candidate]
                    != physical_final_choices[candidate]
                    or final_choices[candidate]
                    != physical_final_choices[candidate]
                    or raw_policy_acc[candidate]
                    != physical_policy_acc[candidate]
                ):
                    raise RuntimeError(
                        "final-consistent controller stop must preserve the "
                        "strict natural final answer and its accuracy"
                    )
            else:
                raise RuntimeError(
                    "controller-terminated row has invalid selected stop kind: "
                    f"{selected_kind!r}"
                )
        probe_probabilities: List[List[List[float]]] = [
            [record[2].astype(float).tolist() for record in records]
            for records in records_by_candidate
        ]
        probe_canonical_probabilities = probe_probabilities
        probe_legacy_probabilities: List[List[List[float]]] = [
            [
                (
                    record.legacy_probabilities.astype(float).tolist()
                    if isinstance(record, _ProbeRecord)
                    else record[2].astype(float).tolist()
                )
                for record in records
            ]
            for records in records_by_candidate
        ]

        response_token_lists: List[List[int]] = []
        actor_loss_lists: List[List[int]] = []
        atomic_stop_lists: List[List[int]] = []
        verified_stop_lists: List[List[int]] = []
        verified_stop_aux_weight_lists: List[List[float]] = []
        overflow_stop_lists: List[List[int]] = []
        stop_event_multiplicity_lists: List[List[int]] = []
        surface_stop_event_multiplicity_lists: List[List[int]] = []
        negative_stop_event_multiplicity_lists: List[List[int]] = []
        negative_stop_aux_mask_lists: List[List[int]] = []
        negative_stop_aux_weight_lists: List[List[float]] = []
        recognized_correct_probe_counts: List[int] = []
        recognized_correct_probe_positions: List[List[int]] = []
        incorrect_stop_counts: List[int] = []
        incorrect_stop_positions: List[List[int]] = []
        neutral_probe_stop_counts: List[int] = []
        neutral_probe_stop_positions: List[List[int]] = []
        incorrect_stop_aux_mask_lists: List[List[int]] = []
        incorrect_stop_aux_weight_lists: List[List[float]] = []
        post_accepted_tail_aux_mask_lists: List[List[int]] = []
        post_accepted_tail_aux_weight_lists: List[List[float]] = []
        verified_stop_aux_suppressed: List[int] = []
        for candidate, ids in enumerate(candidate_ids):
            if len(ids) > self.config.response_length:
                raise RuntimeError(
                    "effective controller response exceeds response_length: "
                    f"{len(ids)} > {self.config.response_length}"
                )
            metadata = stop_event_metadata[candidate]
            negative_stop_aux_mask_list = list(
                metadata["negative_stop_aux_mask"]
            )
            negative_stop_aux_weight_list = list(
                metadata["negative_stop_aux_weight"]
            )
            incorrect_metadata = _incorrect_stop_action_metadata(
                probed_stop_positions[candidate],
                records_by_candidate[candidate],
                gold_choices[candidate],
                response_length=len(ids),
                max_stop_count=self.max_stop_count,
                half_life_tokens=self.stop_reward_half_life_tokens,
                positive_override_position=(
                    selected_positions[candidate]
                    if selected_kinds[candidate] == "final_consistent"
                    else None
                ),
            )
            incorrect_stop_aux_mask_list = list(
                incorrect_metadata["incorrect_stop_aux_mask"]
            )
            incorrect_stop_aux_weight_list = list(
                incorrect_metadata["incorrect_stop_aux_weight"]
            )
            response_ids, action_mask = _build_full_response(
                ids,
                self.stop_token_id,
                selected_positions[candidate],
                max_stop_count=self.max_stop_count,
                cut_accepted_stop_tail=self.cut_accepted_stop_tail,
                negative_stop_aux_mask=negative_stop_aux_mask_list,
            )
            response_token_lists.append(response_ids)
            actor_loss_lists.append(action_mask)
            atomic_stop_mask_list = [
                int(int(token_id) == int(self.stop_token_id))
                for token_id in response_ids
            ]
            if sum(atomic_stop_mask_list) != int(
                metadata["raw_atomic_stop_count"]
            ):
                raise RuntimeError(
                    "raw atomic-stop telemetry disagrees with response token IDs"
                )
            atomic_stop_lists.append(atomic_stop_mask_list)
            verified_stop_mask_list = _verified_stop_action_mask(
                response_ids,
                self.stop_token_id,
                credited_positions[candidate],
            )
            overflow_stop_mask_list = list(metadata["overflow_stop_mask"])
            if any(
                verified and negative
                for verified, negative in zip(
                    verified_stop_mask_list, negative_stop_aux_mask_list
                )
            ):
                raise RuntimeError(
                    "verified and negative stop auxiliary actions must be disjoint"
                )
            if any(
                verified and incorrect
                for verified, incorrect in zip(
                    verified_stop_mask_list,
                    incorrect_stop_aux_mask_list,
                )
            ):
                raise RuntimeError(
                    "verified and incorrect stop auxiliary actions must be disjoint"
                )
            if any(
                negative and incorrect
                for negative, incorrect in zip(
                    negative_stop_aux_mask_list,
                    incorrect_stop_aux_mask_list,
                )
            ):
                raise RuntimeError(
                    "generalized-negative and incorrect stop actions must be disjoint"
                )
            if any(
                incorrect and not atomic
                for incorrect, atomic in zip(
                    incorrect_stop_aux_mask_list,
                    atomic_stop_mask_list,
                )
            ):
                raise RuntimeError(
                    "incorrect stop actions must be atomic stop actions"
                )
            verified_stop_aux_suppressed.append(
                output_suppressed_verified_stop_counts[candidate]
            )
            verified_stop_lists.append(verified_stop_mask_list)
            verified_stop_aux_weight_lists.append(
                [
                    (
                        early_stop_scores[candidate]
                        if verified_stop_mask_list[position]
                        else 0.0
                    )
                    for position in range(len(response_ids))
                ]
            )
            overflow_stop_lists.append(overflow_stop_mask_list)
            stop_event_multiplicity_lists.append(
                list(metadata["stop_event_multiplicity"])
            )
            surface_stop_event_multiplicity_lists.append(
                list(metadata["surface_stop_event_multiplicity"])
            )
            negative_stop_event_multiplicity_lists.append(
                list(metadata["negative_stop_event_multiplicity"])
            )
            negative_stop_aux_mask_lists.append(negative_stop_aux_mask_list)
            negative_stop_aux_weight_lists.append(
                negative_stop_aux_weight_list
            )
            incorrect_stop_counts.append(
                int(incorrect_metadata["incorrect_stop_count"])
            )
            recognized_correct_probe_counts.append(
                int(incorrect_metadata["recognized_correct_probe_count"])
            )
            recognized_correct_probe_positions.append(
                list(incorrect_metadata["recognized_correct_probe_positions"])
            )
            incorrect_stop_positions.append(
                list(incorrect_metadata["incorrect_stop_positions"])
            )
            neutral_probe_stop_counts.append(
                int(incorrect_metadata["neutral_probe_stop_count"])
            )
            neutral_probe_stop_positions.append(
                list(incorrect_metadata["neutral_probe_stop_positions"])
            )
            incorrect_stop_aux_mask_lists.append(
                incorrect_stop_aux_mask_list
            )
            incorrect_stop_aux_weight_lists.append(
                incorrect_stop_aux_weight_list
            )
            # Controller termination replaces the sampled tail; no sampled
            # post-accepted action exists in the effective MDP.  Keep zero
            # schema tensors for fail-closed trainer/audit compatibility.
            tail_mask = [0] * len(response_ids)
            tail_weight = [0.0] * len(response_ids)
            post_accepted_tail_aux_mask_lists.append(tail_mask)
            post_accepted_tail_aux_weight_lists.append(tail_weight)
        response = pad_2d_list_to_length(
            response_token_lists,
            self.pad_token_id,
            max_length=self.config.response_length,
        ).to(idx.device)
        actor_loss_mask = pad_2d_list_to_length(
            actor_loss_lists, 0, max_length=self.config.response_length
        ).to(idx.device)
        atomic_stop_mask = pad_2d_list_to_length(
            atomic_stop_lists, 0, max_length=self.config.response_length
        ).to(idx.device)
        verified_stop_mask = pad_2d_list_to_length(
            verified_stop_lists, 0, max_length=self.config.response_length
        ).to(idx.device)
        verified_stop_aux_weight = pad_2d_list_to_length(
            verified_stop_aux_weight_lists,
            0.0,
            max_length=self.config.response_length,
        ).to(idx.device)
        overflow_stop_mask = pad_2d_list_to_length(
            overflow_stop_lists, 0, max_length=self.config.response_length
        ).to(idx.device)
        stop_event_multiplicity = pad_2d_list_to_length(
            stop_event_multiplicity_lists,
            0,
            max_length=self.config.response_length,
        ).to(idx.device)
        surface_stop_event_multiplicity = pad_2d_list_to_length(
            surface_stop_event_multiplicity_lists,
            0,
            max_length=self.config.response_length,
        ).to(idx.device)
        negative_stop_event_multiplicity = pad_2d_list_to_length(
            negative_stop_event_multiplicity_lists,
            0,
            max_length=self.config.response_length,
        ).to(idx.device)
        negative_stop_aux_mask = pad_2d_list_to_length(
            negative_stop_aux_mask_lists,
            0,
            max_length=self.config.response_length,
        ).to(idx.device)
        negative_stop_aux_weight = pad_2d_list_to_length(
            negative_stop_aux_weight_lists,
            0.0,
            max_length=self.config.response_length,
        ).to(idx.device)
        incorrect_stop_aux_mask = pad_2d_list_to_length(
            incorrect_stop_aux_mask_lists,
            0,
            max_length=self.config.response_length,
        ).to(idx.device)
        incorrect_stop_aux_weight = pad_2d_list_to_length(
            incorrect_stop_aux_weight_lists,
            0.0,
            max_length=self.config.response_length,
        ).to(idx.device)
        post_accepted_tail_aux_mask = pad_2d_list_to_length(
            post_accepted_tail_aux_mask_lists,
            0,
            max_length=self.config.response_length,
        ).to(idx.device)
        post_accepted_tail_aux_weight = pad_2d_list_to_length(
            post_accepted_tail_aux_weight_lists,
            0.0,
            max_length=self.config.response_length,
        ).to(idx.device)

        candidate_index = torch.as_tensor(cand_base_idx, dtype=torch.long, device=idx.device)
        idx = idx.index_select(0, candidate_index)
        attention_mask = attention_mask.index_select(0, candidate_index)
        position_ids = position_ids.index_select(0, candidate_index)

        for key, value in list(non_tensor_batch.items()):
            try:
                if len(value) == base_batch_size:
                    non_tensor_batch[key] = _repeat_by_indices(value, cand_base_idx)
            except TypeError:
                pass
        non_tensor_batch["classifier_final_feature_x"] = _object_array_1d(
            classifier_final_feature_x
        )
        non_tensor_batch["classifier_final_feature_y"] = _object_array_1d(
            classifier_final_feature_y
        )
        non_tensor_batch["classifier_gold_or_final_feature_x"] = _object_array_1d(
            classifier_gold_or_final_feature_x
        )
        non_tensor_batch["classifier_gold_or_final_feature_y"] = _object_array_1d(
            classifier_gold_or_final_feature_y
        )
        non_tensor_batch.update(_classifier_schema_metadata(candidate_count))
        non_tensor_batch["classifier_physical_record_count"] = _object_array_1d(
            [len(records) for records in physical_records_by_candidate]
        )
        non_tensor_batch["physical_probe_answers"] = _object_array_1d(
            [
                [record[0] for record in records]
                for records in physical_records_by_candidate
            ]
        )
        non_tensor_batch["probe_answers"] = _object_array_1d(
            [[record[0] for record in records] for records in records_by_candidate]
        )
        non_tensor_batch["probe_probabilities"] = _object_array_1d(probe_probabilities)
        non_tensor_batch["probe_canonical_probabilities"] = _object_array_1d(
            probe_canonical_probabilities
        )
        non_tensor_batch["probe_legacy_probabilities"] = _object_array_1d(
            probe_legacy_probabilities
        )
        non_tensor_batch["selected_stop_position"] = _object_array_1d(selected_positions)
        non_tensor_batch["selected_probe_answer"] = _object_array_1d(selected_answers)
        non_tensor_batch["selected_stop_kind"] = _object_array_1d(selected_kinds)
        non_tensor_batch["selected_stop_credit"] = _object_array_1d(selected_credits)
        non_tensor_batch["final_consistency_credit"] = _object_array_1d(
            [self.final_consistency_credit] * candidate_count
        )
        non_tensor_batch["oracle_accepted_stop_positions"] = _object_array_1d(
            oracle_accepted_positions
        )
        non_tensor_batch["oracle_accepted_probe_answer"] = _object_array_1d(
            oracle_accepted_answers
        )
        non_tensor_batch["verified_stop_positions"] = _object_array_1d(
            credited_positions
        )
        non_tensor_batch["incorrect_stop_positions"] = _object_array_1d(
            incorrect_stop_positions
        )
        non_tensor_batch["recognized_correct_probe_positions"] = _object_array_1d(
            recognized_correct_probe_positions
        )
        non_tensor_batch["neutral_probe_stop_positions"] = _object_array_1d(
            neutral_probe_stop_positions
        )
        non_tensor_batch["eos_trimmed_token_count"] = _object_array_1d(
            eos_trimmed_token_counts
        )
        non_tensor_batch["physical_final_choice"] = _object_array_1d(
            physical_final_choices
        )
        non_tensor_batch["effective_final_choice"] = _object_array_1d(
            final_choices
        )

        sequence = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(candidate_count, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(candidate_count, 1, -1).expand(
                candidate_count, 3, -1
            )
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        explicit_effective_response_lengths = [
            len(response_ids) for response_ids in response_token_lists
        ]
        response_attention_mask = _response_attention_mask_from_lengths(
            explicit_effective_response_lengths,
            response_length,
            device=response.device,
            dtype=attention_mask.dtype,
        )
        if response_attention_mask.shape != response.shape:
            raise RuntimeError(
                "explicit response mask shape disagrees with the padded response: "
                f"mask={tuple(response_attention_mask.shape)} "
                f"response={tuple(response.shape)}"
            )
        attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
        actor_loss_mask = actor_loss_mask.to(dtype=attention_mask.dtype)
        atomic_stop_mask = atomic_stop_mask.to(dtype=attention_mask.dtype)
        verified_stop_mask = verified_stop_mask.to(dtype=attention_mask.dtype)
        verified_stop_aux_weight = verified_stop_aux_weight.to(
            dtype=torch.float32
        )
        overflow_stop_mask = overflow_stop_mask.to(dtype=attention_mask.dtype)
        stop_event_multiplicity = stop_event_multiplicity.to(dtype=torch.long)
        surface_stop_event_multiplicity = (
            surface_stop_event_multiplicity.to(dtype=torch.long)
        )
        negative_stop_event_multiplicity = (
            negative_stop_event_multiplicity.to(dtype=torch.long)
        )
        negative_stop_aux_mask = negative_stop_aux_mask.to(
            dtype=attention_mask.dtype
        )
        negative_stop_aux_weight = negative_stop_aux_weight.to(
            dtype=torch.float32
        )
        incorrect_stop_aux_mask = incorrect_stop_aux_mask.to(
            dtype=attention_mask.dtype
        )
        incorrect_stop_aux_weight = incorrect_stop_aux_weight.to(
            dtype=torch.float32
        )
        post_accepted_tail_aux_mask = post_accepted_tail_aux_mask.to(
            dtype=attention_mask.dtype
        )
        post_accepted_tail_aux_weight = post_accepted_tail_aux_weight.to(
            dtype=torch.float32
        )
        if torch.any(actor_loss_mask > response_attention_mask):
            violations = torch.nonzero(
                actor_loss_mask > response_attention_mask,
                as_tuple=False,
            )
            first_row, first_position = [int(value) for value in violations[0].tolist()]
            raise RuntimeError(
                "actor_loss_mask must be inside the valid response after EOS "
                f"normalization: violations={int(violations.size(0))} "
                f"first_row={first_row} first_position={first_position} "
                f"response_token={int(response[first_row, first_position])} "
                f"eos_token_id={eos_token_id!r}"
            )
        if torch.any(verified_stop_mask > response_attention_mask):
            raise RuntimeError(
                "verified stop actions must be inside the valid response"
            )
        if torch.any(overflow_stop_mask > response_attention_mask):
            raise RuntimeError(
                "overflow stop actions must be inside the valid response"
            )
        if torch.any(negative_stop_aux_mask > response_attention_mask):
            raise RuntimeError(
                "negative stop actions must be inside the valid response"
            )
        if torch.any(incorrect_stop_aux_mask > response_attention_mask):
            raise RuntimeError(
                "incorrect stop actions must be inside the valid response"
            )
        if torch.any(post_accepted_tail_aux_mask > response_attention_mask):
            raise RuntimeError(
                "post-accepted tail actions must be inside the valid response"
            )
        if torch.any(atomic_stop_mask > response_attention_mask):
            raise RuntimeError(
                "atomic stop actions must be inside the valid response"
            )
        if torch.any(verified_stop_mask > atomic_stop_mask):
            raise RuntimeError(
                "verified_stop_mask must be a subset of atomic_stop_mask"
            )
        if torch.any(overflow_stop_mask > atomic_stop_mask):
            raise RuntimeError(
                "overflow_stop_mask must be a subset of atomic_stop_mask"
            )
        if torch.any(incorrect_stop_aux_mask > atomic_stop_mask):
            raise RuntimeError(
                "incorrect_stop_aux_mask must be a subset of atomic_stop_mask"
            )
        if torch.any((verified_stop_mask > 0) & (overflow_stop_mask > 0)):
            raise RuntimeError(
                "verified and overflow stop action masks must be disjoint"
            )
        if torch.any((verified_stop_mask > 0) & (negative_stop_aux_mask > 0)):
            raise RuntimeError(
                "verified and negative stop action masks must be disjoint"
            )
        if torch.any((verified_stop_mask > 0) & (incorrect_stop_aux_mask > 0)):
            raise RuntimeError(
                "verified and incorrect stop action masks must be disjoint"
            )
        if torch.any(
            (negative_stop_aux_mask > 0) & (incorrect_stop_aux_mask > 0)
        ):
            raise RuntimeError(
                "generalized-negative and incorrect stop masks must be disjoint"
            )
        if torch.any(
            (post_accepted_tail_aux_mask > 0)
            & (negative_stop_aux_mask > 0)
        ):
            raise RuntimeError(
                "tail and negative stop auxiliary masks must be disjoint"
            )
        if torch.any(
            (post_accepted_tail_aux_mask > 0)
            & (incorrect_stop_aux_mask > 0)
        ):
            raise RuntimeError(
                "tail and incorrect stop auxiliary masks must be disjoint"
            )
        if torch.any((negative_stop_aux_weight > 0) != (negative_stop_aux_mask > 0)):
            raise RuntimeError(
                "negative stop auxiliary mask and weight support disagree"
            )
        if torch.any(
            (incorrect_stop_aux_weight > 0)
            != (incorrect_stop_aux_mask > 0)
        ):
            raise RuntimeError(
                "incorrect stop auxiliary mask and weight support disagree"
            )
        if torch.any(
            (negative_stop_event_multiplicity > 0)
            != (negative_stop_aux_mask > 0)
        ):
            raise RuntimeError(
                "negative stop multiplicity and auxiliary mask support disagree"
            )
        if torch.any(
            (post_accepted_tail_aux_weight > 0)
            != (post_accepted_tail_aux_mask > 0)
        ):
            raise RuntimeError(
                "tail auxiliary mask and weight support disagree"
            )
        expected_overflow_counts = torch.as_tensor(
            [
                int(metadata["overflow_atomic_stop_count"])
                for metadata in stop_event_metadata
            ],
            dtype=torch.long,
            device=idx.device,
        )
        observed_overflow_counts = overflow_stop_mask.sum(dim=-1).to(
            dtype=torch.long
        )
        if not torch.equal(
            observed_overflow_counts, expected_overflow_counts
        ):
            raise RuntimeError(
                "overflow stop mask must contain every atomic stop whose "
                "global event ordinal is greater than max_stop_count: observed="
                f"{observed_overflow_counts.tolist()} expected="
                f"{expected_overflow_counts.tolist()}"
            )
        enabled_atomic_stops = (
            (actor_loss_mask > 0) & (atomic_stop_mask > 0)
        )
        if torch.any(enabled_atomic_stops):
            raise RuntimeError(
                "every atomic stop action must be excluded from trajectory GRPO"
            )
        enabled_negative_stops = (
            (actor_loss_mask > 0) & (negative_stop_aux_mask > 0)
        )
        if torch.any(enabled_negative_stops):
            raise RuntimeError(
                "negative stop actions must be excluded from trajectory GRPO"
            )
        enabled_incorrect_stops = (
            (actor_loss_mask > 0) & (incorrect_stop_aux_mask > 0)
        )
        if torch.any(enabled_incorrect_stops):
            raise RuntimeError(
                "incorrect stop actions must be excluded from trajectory GRPO"
            )
        verified_stop_position_tensor = torch.as_tensor(
            [
                position if position is not None else -1
                for position in selected_positions
            ],
            dtype=torch.long,
            device=idx.device,
        )
        oracle_accepted_stop_position_tensor = torch.as_tensor(
            [
                position if position is not None else -1
                for position in oracle_accepted_positions
            ],
            dtype=torch.long,
            device=idx.device,
        )
        raw_response_lengths = response_attention_mask.sum(dim=-1).to(
            dtype=torch.long
        )
        explicit_effective_response_length_tensor = torch.as_tensor(
            explicit_effective_response_lengths,
            dtype=torch.long,
            device=idx.device,
        )
        if not torch.equal(
            raw_response_lengths, explicit_effective_response_length_tensor
        ):
            raise RuntimeError(
                "explicit response attention mask changed an authoritative "
                "effective token-list length"
            )
        accepted_prefix_lengths = torch.where(
            verified_stop_position_tensor >= 0,
            verified_stop_position_tensor + 1,
            raw_response_lengths,
        )
        if torch.any(accepted_prefix_lengths > raw_response_lengths):
            raise RuntimeError(
                "accepted stop prefix extends beyond the valid raw response"
            )
        controller_terminated_tensor = torch.as_tensor(
            controller_terminated, dtype=torch.long, device=idx.device
        )
        controller_wrap_rejected_stop_count_tensor = torch.as_tensor(
            controller_wrap_rejected_stop_counts,
            dtype=torch.long,
            device=idx.device,
        )
        if torch.any(controller_wrap_rejected_stop_count_tensor < 0):
            raise RuntimeError(
                "controller wrap rejection counts must be non-negative"
            )
        controller_discarded_token_count_tensor = torch.as_tensor(
            controller_discarded_token_counts,
            dtype=torch.long,
            device=idx.device,
        )
        controller_forced_suffix_token_count_tensor = torch.as_tensor(
            controller_forced_suffix_token_counts,
            dtype=torch.long,
            device=idx.device,
        )
        physical_response_length_tensor = torch.as_tensor(
            physical_response_lengths, dtype=torch.long, device=idx.device
        )
        expected_controller_terminated = (
            verified_stop_position_tensor >= 0
        ).to(dtype=torch.long)
        if not torch.equal(
            controller_terminated_tensor, expected_controller_terminated
        ):
            raise RuntimeError(
                "controller termination must exactly match oracle-verified stop credit"
            )
        expected_effective_lengths = torch.where(
            controller_terminated_tensor > 0,
            accepted_prefix_lengths + controller_forced_suffix_token_count_tensor,
            physical_response_length_tensor,
        )
        if not torch.equal(raw_response_lengths, expected_effective_lengths):
            raise RuntimeError(
                "effective response length disagrees with the controller splice: "
                f"observed={raw_response_lengths.tolist()} expected="
                f"{expected_effective_lengths.tolist()}"
            )
        expected_physical_lengths = torch.where(
            controller_terminated_tensor > 0,
            accepted_prefix_lengths + controller_discarded_token_count_tensor,
            raw_response_lengths,
        )
        if not torch.equal(
            physical_response_length_tensor, expected_physical_lengths
        ):
            raise RuntimeError(
                "physical response length disagrees with discarded-tail telemetry: "
                f"observed={physical_response_length_tensor.tolist()} expected="
                f"{expected_physical_lengths.tolist()}"
            )
        if torch.any(
            (controller_terminated_tensor == 0)
            & (
                (controller_discarded_token_count_tensor != 0)
                | (controller_forced_suffix_token_count_tensor != 0)
            )
        ):
            raise RuntimeError(
                "non-terminated responses cannot report a discarded tail or forced suffix"
            )
        if torch.any(
            (controller_terminated_tensor > 0)
            & (controller_forced_suffix_token_count_tensor <= 0)
        ):
            raise RuntimeError(
                "controller-terminated responses require a non-empty forced suffix"
            )
        # Compatibility name retained for downstream dashboards.  It now has
        # the only defensible meaning: sampled physical tokens discarded by
        # the external controller, never the forced canonical closure.
        tail_cut_tokens = controller_discarded_token_count_tensor
        post_accepted_active_tokens = torch.zeros_like(raw_response_lengths)
        for row_index, position in enumerate(selected_positions):
            if position is None:
                continue
            post_accepted_active_tokens[row_index] = actor_loss_mask[
                row_index, int(position) + 1 :
            ].sum()
        if self.cut_accepted_stop_tail:
            if torch.any(post_accepted_active_tokens != 0):
                raise RuntimeError(
                    "ordinary actor gradient leaked after an accepted stop"
                )
        if torch.any(post_accepted_tail_aux_mask != 0) or torch.any(
            post_accepted_tail_aux_weight != 0
        ):
            raise RuntimeError(
                "controller-terminal rollout cannot retain post-accepted tail loss"
            )
        for row_index, position in enumerate(selected_positions):
            if position is None:
                continue
            suffix_start = int(position) + 1
            suffix_end = int(raw_response_lengths[row_index])
            if torch.any(actor_loss_mask[row_index, suffix_start:suffix_end] != 0):
                raise RuntimeError(
                    "controller-forced final suffix must have zero actor gradient"
                )
        early_stop_score_tensor = torch.as_tensor(
            early_stop_scores, dtype=torch.float32, device=idx.device
        )
        observed_verified_stop_counts = verified_stop_mask.sum(dim=-1).to(
            dtype=torch.long
        )
        expected_verified_stop_counts = torch.as_tensor(
            verified_stop_counts,
            dtype=torch.long,
            device=idx.device,
        )
        if not torch.equal(
            observed_verified_stop_counts, expected_verified_stop_counts
        ):
            raise RuntimeError(
                "verified stop action mask disagrees with credited stop count"
            )
        observed_stop_scores = verified_stop_aux_weight.sum(dim=-1)
        if not torch.allclose(
            observed_stop_scores,
            early_stop_score_tensor,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "verified stop weights disagree with earliest-only stop score"
            )
        incorrect_stop_count_tensor = torch.as_tensor(
            incorrect_stop_counts,
            dtype=torch.long,
            device=idx.device,
        )
        observed_incorrect_stop_counts = incorrect_stop_aux_mask.sum(
            dim=-1
        ).to(dtype=torch.long)
        if not torch.equal(
            observed_incorrect_stop_counts, incorrect_stop_count_tensor
        ):
            raise RuntimeError(
                "incorrect stop action mask disagrees with incorrect probe count"
            )
        if torch.any(incorrect_stop_count_tensor > self.max_stop_count):
            raise RuntimeError(
                "incorrect stop action count exceeds the probe budget"
            )
        recognized_correct_probe_count_tensor = torch.as_tensor(
            recognized_correct_probe_counts,
            dtype=torch.long,
            device=idx.device,
        )
        neutral_probe_stop_count_tensor = torch.as_tensor(
            neutral_probe_stop_counts,
            dtype=torch.long,
            device=idx.device,
        )
        expected_probed_stop_count_tensor = torch.as_tensor(
            [len(positions) for positions in probed_stop_positions],
            dtype=torch.long,
            device=idx.device,
        )
        if not torch.equal(
            recognized_correct_probe_count_tensor
            + incorrect_stop_count_tensor
            + neutral_probe_stop_count_tensor,
            expected_probed_stop_count_tensor,
        ):
            raise RuntimeError(
                "correct/incorrect/neutral probe counts must sum to "
                "probed_stop_count"
            )
        response_positions = torch.arange(
            response.size(1),
            dtype=torch.float32,
            device=idx.device,
        ).unsqueeze(0)
        expected_incorrect_stop_weight = (
            torch.pow(
                torch.tensor(2.0, dtype=torch.float32, device=idx.device),
                -response_positions / float(self.stop_reward_half_life_tokens),
            )
            * incorrect_stop_aux_mask.to(dtype=torch.float32)
        )
        if not torch.allclose(
            incorrect_stop_aux_weight,
            expected_incorrect_stop_weight,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "incorrect stop weights must equal unnormalized per-action "
                "earliness weights"
            )
        expected_raw_event_counts = torch.as_tensor(
            raw_stop_counts, dtype=torch.long, device=idx.device
        )
        expected_surface_event_counts = torch.as_tensor(
            [
                int(metadata["illegal_surface_stop_count"])
                for metadata in stop_event_metadata
            ],
            dtype=torch.long,
            device=idx.device,
        )
        expected_negative_event_counts = torch.as_tensor(
            [
                int(metadata["negative_stop_event_count"])
                for metadata in stop_event_metadata
            ],
            dtype=torch.long,
            device=idx.device,
        )
        if not torch.equal(
            stop_event_multiplicity.sum(dim=-1), expected_raw_event_counts
        ):
            raise RuntimeError("stop event multiplicity disagrees with raw count")
        if not torch.equal(
            surface_stop_event_multiplicity.sum(dim=-1),
            expected_surface_event_counts,
        ):
            raise RuntimeError(
                "surface stop multiplicity disagrees with surface count"
            )
        if not torch.equal(
            negative_stop_event_multiplicity.sum(dim=-1),
            expected_negative_event_counts,
        ):
            raise RuntimeError(
                "negative stop multiplicity disagrees with negative count"
            )
        negative_row_weight = negative_stop_aux_weight.sum(dim=-1)
        expected_negative_row_weight = torch.as_tensor(
            [
                1.0 if int(metadata["negative_stop_event_count"]) else 0.0
                for metadata in stop_event_metadata
            ],
            dtype=torch.float32,
            device=idx.device,
        )
        if not torch.allclose(
            negative_row_weight,
            expected_negative_row_weight,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "negative stop auxiliary weights must be normalized per rollout"
            )
        tail_lengths = post_accepted_tail_aux_mask.sum(dim=-1).to(
            dtype=torch.float32
        )
        expected_tail_weight = torch.clamp(
            tail_lengths
            / float(self.post_accepted_tail_penalty_normalization_tokens),
            max=1.0,
        )
        if not torch.allclose(
            post_accepted_tail_aux_weight.sum(dim=-1),
            expected_tail_weight,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "post-accepted tail weights have incorrect per-rollout mass"
            )

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": sequence,
                "early_stop_score": early_stop_score_tensor,
                "raw_stop_count": torch.as_tensor(
                    raw_stop_counts, dtype=torch.long, device=idx.device
                ),
                "raw_atomic_stop_count": torch.as_tensor(
                    [
                        int(metadata["raw_atomic_stop_count"])
                        for metadata in stop_event_metadata
                    ],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "ordinary_surface_stop_count": torch.as_tensor(
                    [
                        int(metadata["ordinary_surface_stop_count"])
                        for metadata in stop_event_metadata
                    ],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "stop_fragment_count": torch.as_tensor(
                    [
                        int(metadata["stop_fragment_count"])
                        for metadata in stop_event_metadata
                    ],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "illegal_surface_stop_count": torch.as_tensor(
                    [
                        int(metadata["illegal_surface_stop_count"])
                        for metadata in stop_event_metadata
                    ],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "overflow_atomic_stop_count": torch.as_tensor(
                    [
                        int(metadata["overflow_atomic_stop_count"])
                        for metadata in stop_event_metadata
                    ],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "negative_stop_event_count": torch.as_tensor(
                    [
                        int(metadata["negative_stop_event_count"])
                        for metadata in stop_event_metadata
                    ],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "raw_policy_acc": torch.as_tensor(
                    raw_policy_acc, dtype=torch.long, device=idx.device
                ),
                "effective_response_length": raw_response_lengths,
                "controller_terminated": controller_terminated_tensor,
                "controller_wrap_rejected_stop_count": (
                    controller_wrap_rejected_stop_count_tensor
                ),
                "controller_discarded_token_count": (
                    controller_discarded_token_count_tensor
                ),
                "controller_forced_suffix_token_count": (
                    controller_forced_suffix_token_count_tensor
                ),
                "physical_response_length": physical_response_length_tensor,
                "physical_raw_stop_count": torch.as_tensor(
                    physical_raw_stop_counts,
                    dtype=torch.long,
                    device=idx.device,
                ),
                "physical_raw_atomic_stop_count": torch.as_tensor(
                    physical_raw_atomic_stop_counts,
                    dtype=torch.long,
                    device=idx.device,
                ),
                "physical_illegal_surface_stop_count": torch.as_tensor(
                    physical_illegal_surface_stop_counts,
                    dtype=torch.long,
                    device=idx.device,
                ),
                "physical_policy_acc": torch.as_tensor(
                    physical_policy_acc, dtype=torch.long, device=idx.device
                ),
                "verified_stop": torch.as_tensor(
                    verified_stops, dtype=torch.long, device=idx.device
                ),
                "verified_stop_count": expected_verified_stop_counts,
                "verified_stop_score": early_stop_score_tensor,
                "verified_stop_position": verified_stop_position_tensor,
                "oracle_accepted_stop_position": (
                    oracle_accepted_stop_position_tensor
                ),
                "oracle_accepted_stop_ordinal": torch.as_tensor(
                    oracle_accepted_stop_ordinals,
                    dtype=torch.long,
                    device=idx.device,
                ),
                "accepted_stop_prefix_length": accepted_prefix_lengths,
                "tail_cut_tokens": tail_cut_tokens,
                "accepted_stop_tail_cut_enabled": torch.full_like(
                    raw_response_lengths,
                    int(self.cut_accepted_stop_tail),
                ),
                "post_accepted_active_token_count": (
                    post_accepted_active_tokens
                ),
                "verified_stop_aux_suppressed": torch.as_tensor(
                    verified_stop_aux_suppressed,
                    dtype=torch.long,
                    device=idx.device,
                ),
                "selected_stop_ordinal": torch.as_tensor(
                    selected_stop_ordinals, dtype=torch.long, device=idx.device
                ),
                "eligible_stop_count": torch.as_tensor(
                    [len(positions) for positions in eligible_stop_positions],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "probed_stop_count": torch.as_tensor(
                    [len(positions) for positions in probed_stop_positions],
                    dtype=torch.long,
                    device=idx.device,
                ),
                "recognized_correct_probe_count": (
                    recognized_correct_probe_count_tensor
                ),
                "incorrect_stop_count": incorrect_stop_count_tensor,
                "neutral_probe_stop_count": neutral_probe_stop_count_tensor,
                "first_probe_correct": torch.as_tensor(
                    first_probe_correct, dtype=torch.long, device=idx.device
                ),
                # Preserve vLLM's original candidate ordinal across the
                # trainer's subsequent length-balancing permutation.
                "rollout_index": torch.as_tensor(
                    cand_rollout_idx, dtype=torch.long, device=idx.device
                ),
                "actor_loss_mask": actor_loss_mask,
                "atomic_stop_mask": atomic_stop_mask,
                "verified_stop_mask": verified_stop_mask,
                "verified_stop_aux_weight": verified_stop_aux_weight,
                "overflow_stop_mask": overflow_stop_mask,
                "stop_event_multiplicity": stop_event_multiplicity,
                "surface_stop_event_multiplicity": (
                    surface_stop_event_multiplicity
                ),
                "negative_stop_event_multiplicity": (
                    negative_stop_event_multiplicity
                ),
                "negative_stop_aux_mask": negative_stop_aux_mask,
                "negative_stop_aux_weight": negative_stop_aux_weight,
                "incorrect_stop_aux_mask": incorrect_stop_aux_mask,
                "incorrect_stop_aux_weight": incorrect_stop_aux_weight,
                "post_accepted_tail_aux_mask": post_accepted_tail_aux_mask,
                "post_accepted_tail_aux_weight": post_accepted_tail_aux_weight,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=candidate_count,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
