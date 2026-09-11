#!/usr/bin/env python3
"""Build and strictly verify the historical deterministic/no-hash math 10k.

The historical contract is the draw order returned by
``random.Random(20260827).sample(range(38918), 10000)`` on CPython 3.10.
Selection is by frozen source position, never by qid hash.  File hashes below
are integrity checks and do not participate in sample selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "deepscaler_uniform_random_10k_v1"
SELECTION_ALGORITHM = "random.Random(seed).sample(range(38918),10000)"
SELECTION_METADATA = (
    "random_draw_rank",
    "random_source_position",
    "sampling_method",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def ordered_qids_sha256(qids: list[str]) -> str:
    payload = json.dumps(qids, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {token!r}")
                ),
            )
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def source_qids(rows: list[dict[str, Any]], path: Path) -> list[str]:
    qids = [str(row.get("unique_id") or "") for row in rows]
    if any(not qid for qid in qids) or len(set(qids)) != len(qids):
        raise ValueError(f"{path}: empty or duplicate unique_id")
    return qids


def historical_positions(total_rows: int, sample_rows: int, seed: int) -> list[int]:
    if total_rows < 1 or not 0 < sample_rows <= total_rows:
        raise ValueError("require 0 < sample_rows <= total_rows")
    return random.Random(seed).sample(range(total_rows), sample_rows)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            # Match the historical builder's JSON serialization.
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_projection(
    source_rows: list[dict[str, Any]], *, sample_rows: int, seed: int
) -> tuple[list[int], list[dict[str, Any]]]:
    positions = historical_positions(len(source_rows), sample_rows, seed)
    selected: list[dict[str, Any]] = []
    for draw_rank, source_position in enumerate(positions):
        source_row = source_rows[source_position]
        conflicts = sorted(set(SELECTION_METADATA) & set(source_row))
        if conflicts:
            raise ValueError(
                f"source row {source_position} already has selection metadata {conflicts}"
            )
        row = dict(source_row)
        row["random_draw_rank"] = draw_rank
        row["random_source_position"] = source_position
        row["sampling_method"] = "python_random_sample_without_replacement"
        selected.append(row)
    return positions, selected


def expected_manifest(
    *,
    source: Path,
    output: Path,
    source_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    positions: list[int],
    seed: int,
) -> dict[str, Any]:
    source_ids = source_qids(source_rows, source)
    selected_ids = source_qids(selected_rows, output)
    return {
        "schema": SCHEMA,
        "selection_algorithm": SELECTION_ALGORITHM,
        "hash_used_for_selection": False,
        "seed": seed,
        "python_contract": "CPython 3.10 random.Random.sample",
        "source": str(source.resolve()),
        "source_rows": len(source_rows),
        "source_sha256": file_sha256(source),
        "source_ordered_qids_sha256": ordered_qids_sha256(source_ids),
        "selected_source_positions_sha256": object_sha256(positions),
        "output": str(output.resolve()),
        "output_rows": len(selected_rows),
        "output_sha256": file_sha256(output),
        "output_ordered_qids_sha256": ordered_qids_sha256(selected_ids),
        "unique_qids": len(set(selected_ids)),
    }


def _require_python_contract(required: str) -> None:
    observed = f"{sys.version_info.major}.{sys.version_info.minor}"
    if required and observed != required:
        raise RuntimeError(
            f"historical selector requires CPython {required}; observed {observed}"
        )


def build_or_verify(
    *,
    source: Path,
    output: Path,
    manifest: Path,
    source_rows_expected: int,
    source_sha256_expected: str,
    sample_rows: int,
    seed: int,
    verify_only: bool,
) -> dict[str, Any]:
    source_rows = load_jsonl(source)
    if len(source_rows) != source_rows_expected:
        raise ValueError(
            f"source row mismatch: {len(source_rows)} != {source_rows_expected}"
        )
    observed_source_sha = file_sha256(source)
    if observed_source_sha != source_sha256_expected:
        raise ValueError(
            f"source SHA mismatch: {observed_source_sha} != {source_sha256_expected}"
        )
    source_qids(source_rows, source)
    positions, expected_rows = expected_projection(
        source_rows, sample_rows=sample_rows, seed=seed
    )

    output_exists = output.exists()
    manifest_exists = manifest.exists()
    if output_exists != manifest_exists:
        raise RuntimeError("output/manifest partial publication; refusing repair or overwrite")
    if verify_only and not output_exists:
        raise FileNotFoundError("10k output and manifest are not published")
    if not output_exists:
        _atomic_write_jsonl(output, expected_rows)
        payload = expected_manifest(
            source=source,
            output=output,
            source_rows=source_rows,
            selected_rows=expected_rows,
            positions=positions,
            seed=seed,
        )
        _atomic_write_json(manifest, payload)

    actual_rows = load_jsonl(output)
    if actual_rows != expected_rows:
        raise ValueError("published 10k JSONL does not match the historical draw exactly")
    payload = expected_manifest(
        source=source,
        output=output,
        source_rows=source_rows,
        selected_rows=actual_rows,
        positions=positions,
        seed=seed,
    )
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    if stored != payload:
        raise ValueError("selection manifest does not match the verified 10k artifact")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-rows", type=int, default=38_918)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_827)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-python-major-minor", default="3.10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_python_contract(args.require_python_major_minor)
    result = build_or_verify(
        source=args.source,
        output=args.output,
        manifest=args.manifest,
        source_rows_expected=args.source_rows,
        source_sha256_expected=args.source_sha256,
        sample_rows=args.sample_rows,
        seed=args.seed,
        verify_only=args.verify_only,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
