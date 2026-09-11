from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from collect_replay_every50 import (
    FORMAT_VERSION,
    RECORD_SCHEMA_VERSION,
    completed_qids,
    expected_steps_for_main,
    find_subsequence,
    parse_probe_text,
    trim_probe,
)
from math_cluster_features import FEATURES


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens
        return "".join(chr(int(token)) for token in token_ids)


def test_parse_and_trim_stop_at_first_balanced_outer_brace() -> None:
    answer, boundary = parse_probe_text(r"\frac{1}{2}}ignored")
    assert answer == r"\frac{1}{2}"
    assert boundary == len(r"\frac{1}{2}}")

    tokenizer = CharacterTokenizer()
    text = r"\frac{1}{2}}ignored"
    candidate = SimpleNamespace(
        token_ids=tokenizer.encode(text),
        logprobs=[[{"logprob": -0.1}]] * len(text),
    )
    trimmed = trim_probe(tokenizer, candidate)
    assert trimmed["answer"] == r"\frac{1}{2}"
    assert trimmed["text"] == r"\frac{1}{2}}"
    assert len(trimmed["steps"]) == len(trimmed["text"])
    assert trimmed["generated_output_tokens"] == len(text)


def test_retokenized_positions_are_only_complete_every_50_boundaries() -> None:
    tokenizer = CharacterTokenizer()
    close_ids = tokenizer.encode("</think>")
    text = "x" * 125 + "</think>" + "final"
    positions, reasoning_end, close_found, total = expected_steps_for_main(
        tokenizer, text, close_ids
    )
    assert positions == [50, 100]
    assert reasoning_end == 125
    assert close_found
    assert total == len(text)
    assert find_subsequence(tokenizer.encode(text), close_ids) == 125


def test_resume_contract_allows_a_valid_zero_probe_qid(tmp_path: Path) -> None:
    fingerprint = "f" * 64
    row = {
        "format_version": FORMAT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "generation_config_sha256": fingerprint,
        "qid": "short-qid",
        "probes": [],
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert completed_qids(path, fingerprint, {"short-qid"}) == {"short-qid"}


def test_resume_contract_enforces_feature_order(tmp_path: Path) -> None:
    fingerprint = "a" * 64
    features = {name: 0.0 for name in FEATURES}
    row = {
        "format_version": FORMAT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "generation_config_sha256": fingerprint,
        "qid": "qid",
        "probes": [
            {
                "probe_index": 1,
                "step_tokens": 50,
                "features": features,
            }
        ],
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert completed_qids(path, fingerprint, {"qid"}) == {"qid"}
