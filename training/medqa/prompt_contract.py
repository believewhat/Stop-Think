#!/usr/bin/env python3
"""Single MedQA prompt/output contract for bootstrap, SFT, DAPO and eval.

Qwen3's actual template pre-fills an *empty* thinking block when
``enable_thinking=False``.  ESTAR's reward and stop controller inspect only
response tokens and require the opening tag there.  Every caller therefore
uses ``enable_thinking=True`` without appending a manual prefix: the rendered
prompt ends at the assistant boundary and the policy owns all four tags.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


STOP_TOKEN = "<stop>"
STOP_TOKEN_ID = 151_669
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FINAL_OPEN = "<final_answer>"
FINAL_CLOSE = "</final_answer>"
LETTERS = ("A", "B", "C", "D")

PROMPT_CONTRACT_VERSION = "qwen30a3b_medqa_stop_reasoning_point_v3"
SYSTEM_PROMPT = "Solve the medical multiple-choice question carefully. Return exactly one assistant response using this structure:\n<think>\nA complete reasoning point.<stop>\nContinue reasoning with another complete point.<stop>\nFinish the reasoning.\n</think>\n<final_answer>X</final_answer>\n\nFormat rules:\n1. Put all reasoning inside exactly one <think>...</think> block.\n2. Inside that block, use <stop> to mark the end of a completed reasoning point. Place it after a complete sentence or thought, never inside a word or an unfinished sentence. You may mark multiple completed reasoning points.\n3. <stop> is an internal reasoning marker. It does NOT mean the response is finished, does NOT close the thinking block, and does NOT mean you should stop generating. After an intermediate <stop>, continue the remaining reasoning within the same <think> block.\n4. Never place <stop> before <think>, after </think>, or inside <final_answer>.\n5. Once the reasoning is complete, write </think> and then <final_answer>X</final_answer>. Replace X with exactly one uppercase letter A, B, C, or D.\n6. A response ending after an intermediate reasoning point or a <stop> marker is incomplete. Always provide the final_answer block before ending the response. Do not put anything after </final_answer>."

_STRICT_RESPONSE = re.compile(
    r"\A[ \t\r\n\f\v]*<think>\n?(?P<reasoning>.*?)"
    r"</think>[ \t\r\n\f\v]*<final_answer>(?P<letter>[A-D])"
    r"</final_answer>[ \t\r\n\f\v]*\Z",
    flags=re.DOTALL,
)


def normalize_choices(choices: Mapping[str, Any] | Sequence[Any]) -> dict[str, str]:
    if isinstance(choices, Mapping):
        result = {letter: str(choices[letter]).strip() for letter in LETTERS}
    else:
        if len(choices) != 4:
            raise ValueError("MedQA choices must contain exactly four values")
        result = {letter: str(value).strip() for letter, value in zip(LETTERS, choices)}
    if any(not value for value in result.values()):
        raise ValueError("MedQA choices must all be non-empty")
    return result


def render_user_content(question: str, choices: Mapping[str, Any] | Sequence[Any]) -> str:
    question = str(question).strip()
    if not question:
        raise ValueError("question must be non-empty")
    if STOP_TOKEN in question:
        raise ValueError("question leaks the ESTAR stop action")
    normalized = normalize_choices(choices)
    return "\n".join(
        [question, "", *(f"{letter}. {normalized[letter]}" for letter in LETTERS)]
    )


def canonical_messages(
    question: str, choices: Mapping[str, Any] | Sequence[Any]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_content(question, choices)},
    ]


def normalize_serialized_messages(messages: Any) -> list[dict[str, Any]]:
    """Restore a strict list-of-dicts after pandas/Arrow nested serialization."""
    if not isinstance(messages, (list, tuple)):
        converter = getattr(messages, "tolist", None)
        if callable(converter):
            messages = converter()
    if isinstance(messages, tuple):
        messages = list(messages)
    if not isinstance(messages, list):
        raise ValueError("serialized messages are not a sequence")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            converter = getattr(message, "as_py", None)
            if callable(converter):
                message = converter()
        if not isinstance(message, Mapping):
            raise ValueError("serialized message is not a mapping")
        normalized.append(dict(message))
    return normalized


def validate_messages(messages: Any) -> tuple[str, dict[str, str]]:
    messages = normalize_serialized_messages(messages)
    if (
        len(messages) != 2
        or messages[0].get("role") != "system"
        or messages[0].get("content") != SYSTEM_PROMPT
        or messages[1].get("role") != "user"
        or not isinstance(messages[1].get("content"), str)
    ):
        raise ValueError("messages do not satisfy the canonical MedQA prompt contract")
    body = messages[1]["content"]
    if STOP_TOKEN in body:
        raise ValueError("user content leaks the ESTAR stop action")
    lines = body.splitlines()
    choice_indices: list[int] = []
    parsed: dict[str, str] = {}
    for index, line in enumerate(lines):
        if len(line) >= 4 and line[:2] in {f"{letter}." for letter in LETTERS}:
            letter = line[0]
            if letter in parsed:
                raise ValueError(f"duplicate choice {letter}")
            parsed[letter] = line[2:].strip()
            choice_indices.append(index)
    if tuple(parsed) != LETTERS or any(not parsed[letter] for letter in LETTERS):
        raise ValueError("canonical user content has invalid A-D choices")
    if choice_indices != list(range(choice_indices[0], choice_indices[0] + 4)):
        raise ValueError("canonical choices are not contiguous")
    question = "\n".join(lines[: choice_indices[0]]).strip()
    if not question:
        raise ValueError("canonical question is empty")
    return question, parsed


def render_generation_prompt(tokenizer: Any, messages: Any) -> str:
    messages = normalize_serialized_messages(messages)
    validate_messages(messages)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("tokenizer returned an empty generation prompt")
    assistant_boundary = "<|im_start|>assistant\n"
    if not rendered.endswith(assistant_boundary) or rendered.count(assistant_boundary) != 1:
        raise RuntimeError(
            "Qwen3 generation prompt must end at exactly one bare assistant boundary"
        )
    # The Qwen3 false branch inserts an empty thinking block.  Any thinking
    # tag after the final assistant boundary therefore means policy actions
    # would no longer own the required opening tag.
    tail = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    if THINK_OPEN in tail or THINK_CLOSE in tail:
        raise RuntimeError("chat template prefilled a thinking tag")
    return rendered


def generation_prompt_ids(tokenizer: Any, messages: Any) -> list[int]:
    return [
        int(token_id)
        for token_id in tokenizer.encode(
            render_generation_prompt(tokenizer, messages), add_special_tokens=False
        )
    ]


def parse_strict_response(text: Any) -> tuple[str, str] | None:
    if not isinstance(text, str):
        return None
    match = _STRICT_RESPONSE.fullmatch(text)
    if match is None:
        return None
    if any(text.count(tag) != 1 for tag in (THINK_OPEN, THINK_CLOSE, FINAL_OPEN, FINAL_CLOSE)):
        return None
    reasoning = match.group("reasoning")
    if any(tag in reasoning for tag in (THINK_OPEN, THINK_CLOSE, FINAL_OPEN, FINAL_CLOSE)):
        return None
    return reasoning, match.group("letter")


def strip_generated_im_end(text: Any) -> Any:
    """Remove the single Qwen chat terminator retained by raw token decoding."""
    if not isinstance(text, str):
        return text
    stripped = text.rstrip()
    suffix = "<|im_end|>"
    if stripped.endswith(suffix):
        return stripped[: -len(suffix)].rstrip()
    return text


def canonical_completion(reasoning: str, letter: str) -> str:
    if letter not in LETTERS:
        raise ValueError(f"invalid MedQA choice: {letter!r}")
    reasoning = str(reasoning).strip()
    if not reasoning:
        raise ValueError("reasoning must be non-empty")
    if any(tag in reasoning for tag in (THINK_OPEN, THINK_CLOSE, FINAL_OPEN, FINAL_CLOSE)):
        raise ValueError("reasoning contains reserved format tags")
    return f"<think>\n{reasoning}\n</think>\n<final_answer>{letter}</final_answer>"


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def prompt_contract_metadata() -> dict[str, Any]:
    return {
        "version": PROMPT_CONTRACT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "enable_thinking": True,
        "manual_think_prefix": False,
        "expected_generation_tail": "<|im_start|>assistant\\n",
        "assistant_must_generate_think_open": True,
        "strict_response": "<think>...</think><final_answer>[A-D]</final_answer>",
        "stop_token": STOP_TOKEN,
        "stop_token_id": STOP_TOKEN_ID,
    }
