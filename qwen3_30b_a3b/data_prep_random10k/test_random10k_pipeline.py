from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen30a3b_random10k_builder", ROOT / "build_random10k_source.py"
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def write_source(path: Path, count: int) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {"unique_id": f"q{index:03d}", "problem": f"p{index}"},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_positions_are_draw_order_not_hash_order() -> None:
    observed = BUILDER.historical_positions(100, 20, 20260827)
    expected = random.Random(20260827).sample(range(100), 20)
    assert observed == expected
    assert observed != sorted(observed)


def test_build_verify_and_refuse_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "sample.jsonl"
    manifest = tmp_path / "sample.manifest.json"
    source_sha = write_source(source, 50)
    args = dict(
        source=source,
        output=output,
        manifest=manifest,
        source_rows_expected=50,
        source_sha256_expected=source_sha,
        sample_rows=10,
        seed=20260827,
    )
    built = BUILDER.build_or_verify(**args, verify_only=False)
    checked = BUILDER.build_or_verify(**args, verify_only=True)
    assert built == checked
    rows = BUILDER.load_jsonl(output)
    positions = random.Random(20260827).sample(range(50), 10)
    assert [row["random_source_position"] for row in rows] == positions
    assert [row["random_draw_rank"] for row in rows] == list(range(10))
    assert built["hash_used_for_selection"] is False

    rows[0]["problem"] = "tampered"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="historical draw"):
        BUILDER.build_or_verify(**args, verify_only=True)


def test_partial_publication_is_not_repaired(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "sample.jsonl"
    manifest = tmp_path / "sample.manifest.json"
    source_sha = write_source(source, 20)
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial publication"):
        BUILDER.build_or_verify(
            source=source,
            output=output,
            manifest=manifest,
            source_rows_expected=20,
            source_sha256_expected=source_sha,
            sample_rows=5,
            seed=20260827,
            verify_only=True,
        )
