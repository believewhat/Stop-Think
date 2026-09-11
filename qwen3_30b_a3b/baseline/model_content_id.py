#!/usr/bin/env python3
"""Create a fast, explicitly non-bytewise identity for a HF model tree.

Small files that control architecture/tokenization are byte-hashed. Large weight
shards are identified by their sorted filename/size manifest; their bytes are not
read. This is a provenance identity, not a cryptographic proof of weight equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


SMALL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class IdentityError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_identity(model: Path) -> dict[str, Any]:
    if not model.is_dir():
        raise IdentityError(f"Model directory is missing: {model}")
    missing = [name for name in SMALL_FILES if not (model / name).is_file()]
    if missing:
        raise IdentityError(f"Required model identity files are missing: {missing}")
    small_hashes = {name: sha256(model / name) for name in SMALL_FILES}
    weights = [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(model.glob("*.safetensors"), key=lambda item: item.name)
        if path.is_file()
    ]
    if len(weights) != 16:
        raise IdentityError(f"Expected 16 safetensors shards, found {len(weights)}")
    weight_canonical = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
    weight_manifest_sha = hashlib.sha256(weight_canonical).hexdigest()
    identity_basis = {
        "small_file_sha256": small_hashes,
        "weight_filename_size_manifest": weights,
        "weight_filename_size_manifest_sha256": weight_manifest_sha,
    }
    canonical = json.dumps(identity_basis, sort_keys=True, separators=(",", ":")).encode()
    return {
        "format_version": 2,
        "identity_algorithm": (
            "sha256(canonical small-file SHA256 values plus weight filename/size manifest)"
        ),
        "composite_identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "small_file_sha256": small_hashes,
        "weight_filename_size_manifest": weights,
        "weight_filename_size_manifest_sha256": weight_manifest_sha,
        "weight_shard_count": len(weights),
        "weight_total_bytes": sum(item["size"] for item in weights),
        "weights_byte_hashed": False,
        "limitation": (
            "Weight shard bytes were not hashed; equality is supported by matching "
            "architecture/tokenizer/index hashes and the 16-shard filename/size manifest."
        ),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host-role", required=True, choices=("unity_repeat1", "gamma_repeat23"))
    parser.add_argument("--expected-composite-identity-sha256")
    args = parser.parse_args()

    payload = build_identity(args.model)
    payload.update(
        {
            "host_role": args.host_role,
            "model_path": str(args.model),
            "created_unix": time.time(),
        }
    )
    expected = args.expected_composite_identity_sha256
    if expected and payload["composite_identity_sha256"] != expected.lower():
        raise IdentityError(
            "Model identity mismatch: "
            f"expected={expected.lower()} actual={payload['composite_identity_sha256']}"
        )
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
