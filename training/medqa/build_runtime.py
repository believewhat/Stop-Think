"""Assemble an isolated runtime; never overwrite installed packages or legacy code."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-package", type=Path, required=True,
                        help="Installed vLLM 0.9.0 package directory, including CUDA extensions")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    if importlib.metadata.version("torch").split("+")[0] != "2.7.0":
        raise RuntimeError("This runtime requires the validated PyTorch 2.7.0 ABI")
    if importlib.metadata.version("vllm") != "0.9.0":
        raise RuntimeError("Install vLLM 0.9.0, then supply its package directory")
    if not (args.vllm_package / "__init__.py").is_file():
        raise ValueError("Not a vLLM package directory")
    if not list(args.vllm_package.glob("_C*.so")):
        raise RuntimeError("Missing vLLM CUDA extensions; this source tree alone is insufficient")
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
    shutil.copytree(repo / "EarlyStop/verl", args.output / "verl", ignore=ignore)
    # Begin with the matching binary wheel; preserve its generated version and extensions.
    shutil.copytree(args.vllm_package, args.output / "vllm", ignore=ignore)
    shutil.copytree(repo / "vllm", args.output / "vllm", dirs_exist_ok=True, ignore=ignore)
    shutil.copytree(here / "runtime_overlay", args.output, dirs_exist_ok=True, ignore=ignore)
    (args.output / "runtime_manifest.json").write_text(json.dumps({
        "source": "Stop-Think training/medqa runtime_overlay",
        "torch": importlib.metadata.version("torch"),
        "vllm": importlib.metadata.version("vllm"),
        "legacy_tree_modified": False,
        "gpu_training_smoke_test_required": True,
    }, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
