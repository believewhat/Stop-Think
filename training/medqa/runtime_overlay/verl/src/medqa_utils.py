"""Strict multiple-choice answer parsing for MedQA-style A-D tasks.

The parser intentionally does not scan arbitrary prose for isolated letters.  In
particular, words such as ``and`` and ``because`` must never become choices just
because they start with A or B.
"""

from __future__ import annotations

import re
from typing import Optional


_FINAL_TAG_RE = re.compile(
    r"<final_answer>\s*([A-D])\s*</final_answer>", re.IGNORECASE
)
_EXPLICIT_LABEL_RE = re.compile(
    r"\b(?:the\s+)?(?:correct|final)\s+answer\s*"
    r"(?::|=|-|\bis\b)\s*(?:option\s+|choice\s+)?"
    r"(?:[([]\s*)?([A-D])\b(?:\s*[)\]])?",
    re.IGNORECASE,
)
_BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?"
    r"([A-D])\s*\}?\s*\}",
    re.IGNORECASE,
)
_SINGLE_CHOICE_RE = re.compile(
    r"^\s*(?:[([]\s*)?([A-D])(?:\s*[)\]])?\s*[.]?\s*$",
    re.IGNORECASE,
)
# vLLM probes can be prompted with a prefix ending in ``\\boxed{``.  Their
# continuation is then exactly e.g. ``A}``; accepting that entire fragment is
# strict and does not introduce prose/bucket false positives.
_BOX_CONTINUATION_RE = re.compile(r"^\s*([A-D])\s*}\s*$", re.IGNORECASE)

_ASCII_WHITESPACE = r"[ \t\r\n\f\v]"
_STRICT_TERMINAL_FINAL_RE = re.compile(
    rf"<final_answer>{_ASCII_WHITESPACE}*([A-D])"
    rf"{_ASCII_WHITESPACE}*</final_answer>{_ASCII_WHITESPACE}*\Z"
)


def _last_group(pattern: re.Pattern[str], text: str) -> Optional[str]:
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).upper() if matches else None


def extract_medqa_choice(text: object) -> Optional[str]:
    """Return a strict A-D choice, or ``None`` when no answer is explicit.

    Accepted forms, in priority order, are ``<final_answer>A</final_answer>``,
    an explicit ``correct/final answer`` label, ``\\boxed{A}``, and a response
    consisting only of one choice letter (with harmless wrappers/punctuation).
    The last match within a structured form wins.
    """

    if not isinstance(text, str):
        return None

    for pattern in (_FINAL_TAG_RE, _EXPLICIT_LABEL_RE, _BOXED_RE):
        choice = _last_group(pattern, text)
        if choice is not None:
            return choice

    match = _SINGLE_CHOICE_RE.fullmatch(text)
    if match is None:
        match = _BOX_CONTINUATION_RE.fullmatch(text)
    return match.group(1).upper() if match is not None else None


def extract_strict_terminal_choice(text: object) -> Optional[str]:
    """Return the one uppercase A-D choice from an exact terminal final block.

    This is the policy-accuracy parser.  It deliberately rejects every legacy
    fallback accepted by :func:`extract_medqa_choice`, including bare letters,
    prose labels, boxed answers, lowercase payloads, duplicate tags, and any
    non-ASCII whitespace or non-whitespace text after the closing tag.
    """

    if not isinstance(text, str):
        return None
    if text.count("<final_answer>") != 1 or text.count("</final_answer>") != 1:
        return None
    match = _STRICT_TERMINAL_FINAL_RE.search(text)
    return match.group(1) if match is not None else None


# Descriptive alias retained for call sites that include the dataset name.
extract_strict_terminal_medqa_choice = extract_strict_terminal_choice
