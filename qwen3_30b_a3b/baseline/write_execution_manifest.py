#!/usr/bin/env python3
"""Write immutable-style execution provenance for the Gamma repeat workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-identity", required=True, type=Path)
    parser.add_argument("--math", required=True, type=Path)
    parser.add_argument("--aime", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--summarizer", required=True, type=Path)
    args = parser.parse_args()

    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    identity = json.loads(args.model_identity.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "format_version": 1,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("vllm", "transformers", "torch", "math-verify")
        },
        "gpu_inventory_csv": gpu_csv.splitlines(),
        "gpu_mapping": {
            "repeat2/shard0": 0,
            "repeat2/shard1": 1,
            "repeat3/shard0": 2,
            "repeat3/shard1": 4,
        },
        "model": str(args.model),
        "model_composite_identity_sha256": identity["composite_identity_sha256"],
        "weights_byte_hashed": identity["weights_byte_hashed"],
        "inputs": {
            "math500": {"path": str(args.math), "sha256": sha256(args.math)},
            "aime2024": {"path": str(args.aime), "sha256": sha256(args.aime)},
        },
        "code": {
            "runner": {"path": str(args.runner), "sha256": sha256(args.runner)},
            "summarizer": {
                "path": str(args.summarizer),
                "sha256": sha256(args.summarizer),
            },
        },
        "requested_sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "repetition_penalty": 1.2,
            "max_model_len": 40960,
            "max_tokens": 32768,
            "batch_size": 8,
        },
        "seeds": {"repeat2": 20260904, "repeat3": 20260905},
        "created_unix": time.time(),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
