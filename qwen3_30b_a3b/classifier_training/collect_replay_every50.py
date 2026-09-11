#!/usr/bin/env python3
"""Replay saved Qwen3-30B-A3B CoTs and collect one probe every 50 tokens.

This collector intentionally does *not* regenerate the expensive full CoT.  It
reads the audited 10k DeepScaleR JSONL, retokenizes ``main.text`` with the Qwen
tokenizer, and asks the same forced-box probe used by the deployed ESTAR-Lite
evaluator at token positions 50, 100, ... inside the THINK span.

Rows are committed one qid at a time.  Re-running the identical command safely
skips committed qids after validating the full generation fingerprint and the
probe/feature schema.  Raw vLLM logprob objects are used transiently to compute
the 22 cluster features and are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from math_cluster_features import FEATURES, ClusterEvidenceTracker, answer_logprob_stats


FORMAT_VERSION = 1
RECORD_SCHEMA_VERSION = "qwen30a3b_deepscaler_replay_every50_cluster22_v1"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"

# These constants and the helper implementations below are deliberately kept
# identical to qwen30a3b_math_aime_estar_lite_repeat3_gamma_v1/
# run_estar_lite_shard.py.  In particular, the leading/trailing newlines here
# are part of the classifier contract.
PROBE_SUFFIX = "\n</think>\n\n\\boxed{"
PROBE_TOPK = 20
PROBE_MAX_TOKENS = 64
TOKEN_STEP = 50
DEFAULT_SEED = 20260827


def json_safe(value: Any) -> Any:
    """Convert common numeric containers to strict JSON-compatible values."""
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover - numpy is required by the features
        pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qids_sha256(qids: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(qids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_probe_text(text: str) -> tuple[str | None, int | None]:
    """Parse text generated after the already-open outer ``\\boxed{``."""
    depth = 1
    characters: list[str] = []
    for index, character in enumerate(text):
        cursor = index - 1
        slash_count = 0
        while cursor >= 0 and text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        escaped = bool(slash_count % 2)
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            depth -= 1
            if depth == 0:
                answer = "".join(characters).strip()
                return answer or None, index + 1
        characters.append(character)
    return None, None


def trim_probe(tokenizer: Any, candidate: Any) -> dict[str, Any]:
    """Exact forced-box trimming contract used by the deployed evaluator."""
    token_ids = [int(token) for token in candidate.token_ids]
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    answer, close_character = parse_probe_text(decoded)
    token_end = len(token_ids)
    if close_character is not None:
        for index in range(1, len(token_ids) + 1):
            _answer, boundary = parse_probe_text(
                tokenizer.decode(token_ids[:index], skip_special_tokens=False)
            )
            if boundary is not None:
                token_end = index
                break
    return {
        "answer": answer,
        "box_closed": close_character is not None,
        "steps": list(candidate.logprobs or [])[:token_end],
        "output_tokens": token_end,
        "generated_output_tokens": len(token_ids),
        "text": tokenizer.decode(token_ids[:token_end], skip_special_tokens=False),
    }


def find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    first = needle[0]
    for index in range(len(values) - len(needle) + 1):
        if values[index] == first and values[index : index + len(needle)] == needle:
            return index
    return None


def build_prompt_ids(tokenizer: Any, problem: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) + "<think>\n"
    return [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]


def _strip_math_string(string: str) -> str:
    """Frozen mathd normalization used by the existing Q30 evaluator."""

    def fix_fractions(value: str) -> str:
        parts = value.split("\\frac")
        new_value = parts[0]
        for part in parts[1:]:
            new_value += "\\frac"
            if part and part[0] == "{":
                new_value += part
                continue
            if len(part) < 2:
                return value
            numerator, denominator = part[0], part[1]
            if denominator != "{":
                suffix = part[2:] if len(part) > 2 else ""
                new_value += "{" + numerator + "}{" + denominator + "}" + suffix
            else:
                suffix = part[2:] if len(part) > 2 else ""
                new_value += "{" + numerator + "}" + denominator + suffix
        return new_value

    def fix_slash_fraction(value: str) -> str:
        if len(value.split("/")) != 2:
            return value
        left, right = value.split("/")
        try:
            numerator, denominator = int(left), int(right)
            if value != f"{numerator}/{denominator}":
                return value
            return f"\\frac{{{numerator}}}{{{denominator}}}"
        except Exception:
            return value

    def fix_square_roots(value: str) -> str:
        if "\\sqrt" not in value:
            return value
        parts = value.split("\\sqrt")
        new_value = parts[0]
        for part in parts[1:]:
            if part and part[0] != "{":
                new_value += "\\sqrt{" + part[0] + "}" + part[1:]
            else:
                new_value += "\\sqrt" + part
        return new_value

    string = (string or "").replace("\n", "")
    string = string.replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    if "\\text{ " in string:
        string = string.split("\\text{ ", 1)[0]
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_square_roots(string).replace(" ", "")
    string = fix_fractions(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return fix_slash_fraction(string)


def normalized_answer_key(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        match = re.search(r"^\\text\{(?P<text>.+?)\}$", text)
        if match is not None:
            text = match.group("text").strip()
        normalized = _strip_math_string(text)
    except Exception:
        normalized = text
    return str(normalized).strip() if str(normalized).strip() else None


def load_source(path: Path, expected_qids: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: row is not an object")
            qid = str(row.get("qid") or "")
            if not qid or qid in seen:
                raise ValueError(f"{path}:{line_number}: empty/duplicate qid={qid!r}")
            for field in ("problem", "gold_answer", "main", "generation_config_sha256"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            main = row["main"]
            if not isinstance(main, dict) or not isinstance(main.get("text"), str):
                raise TypeError(f"{path}:{line_number}: main.text is required")
            old_config = row.get("generation_config")
            if not isinstance(old_config, dict):
                raise TypeError(f"{path}:{line_number}: generation_config is required")
            declared_old_fingerprint = str(row["generation_config_sha256"])
            observed_old_fingerprint = hashlib.sha256(
                canonical_json(old_config).encode("utf-8")
            ).hexdigest()
            if declared_old_fingerprint != observed_old_fingerprint:
                raise ValueError(
                    f"{path}:{line_number}: source generation fingerprint mismatch"
                )
            model = str(old_config.get("model") or "")
            if Path(model).name != "Qwen3-30B-A3B":
                raise ValueError(
                    f"{path}:{line_number}: source was not generated by Qwen3-30B-A3B: {model!r}"
                )
            rows.append(row)
            seen.add(qid)
    if expected_qids > 0 and len(rows) != expected_qids:
        raise ValueError(
            f"Source qid count mismatch: expected={expected_qids} actual={len(rows)}"
        )
    if not rows:
        raise ValueError("Source JSONL is empty")
    return rows


def expected_steps_for_main(tokenizer: Any, main_text: str, close_think_ids: list[int]) -> tuple[list[int], int, bool, int]:
    generated_ids = [
        int(token) for token in tokenizer.encode(main_text, add_special_tokens=False)
    ]
    close_position = find_subsequence(generated_ids, close_think_ids)
    reasoning_end = close_position if close_position is not None else len(generated_ids)
    positions = list(range(TOKEN_STEP, reasoning_end + 1, TOKEN_STEP))
    return positions, reasoning_end, close_position is not None, len(generated_ids)


def validate_resume_record(
    row: Any,
    *,
    path: Path,
    line_number: int,
    fingerprint: str,
    allowed_qids: set[str],
) -> str:
    if not isinstance(row, dict):
        raise TypeError(f"{path}:{line_number}: resume row is not an object")
    qid = str(row.get("qid") or "")
    if (
        row.get("format_version") != FORMAT_VERSION
        or row.get("record_schema_version") != RECORD_SCHEMA_VERSION
        or row.get("generation_config_sha256") != fingerprint
        or qid not in allowed_qids
    ):
        raise ValueError(f"{path}:{line_number}: resume fingerprint/schema/qid mismatch")
    probes = row.get("probes")
    if not isinstance(probes, list):
        raise TypeError(f"{path}:{line_number}: probes must be a list")
    expected_step = TOKEN_STEP
    for probe_index, probe in enumerate(probes, 1):
        if not isinstance(probe, dict):
            raise TypeError(f"{path}:{line_number}: malformed probe {probe_index}")
        features = probe.get("features")
        if (
            probe.get("probe_index") != probe_index
            or probe.get("step_tokens") != expected_step
            or not isinstance(features, dict)
            or tuple(features) != FEATURES
            or any(not isinstance(features[name], (int, float)) for name in FEATURES)
        ):
            raise ValueError(f"{path}:{line_number}: probe/feature contract mismatch")
        expected_step += TOKEN_STEP
    return qid


def completed_qids(path: Path, fingerprint: str, allowed_qids: set[str]) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank resume line")
            qid = validate_resume_record(
                json.loads(line),
                path=path,
                line_number=line_number,
                fingerprint=fingerprint,
                allowed_qids=allowed_qids,
            )
            if qid in done:
                raise ValueError(f"{path}:{line_number}: duplicate resume qid={qid}")
            done.add(qid)
    return done


def model_identity(model: Path) -> dict[str, str]:
    if model.name != "Qwen3-30B-A3B":
        raise ValueError(f"Expected model directory Qwen3-30B-A3B, received {model}")
    identity: dict[str, str] = {"path": str(model)}
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        candidate = model / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing model identity file: {candidate}")
        identity[f"{name}_sha256"] = file_sha256(candidate)
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--expected-qids", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--token-step", type=int, default=TOKEN_STEP)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid --shard-index/--shard-count")
    if args.expected_qids < 0:
        parser.error("--expected-qids cannot be negative")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.token_step != TOKEN_STEP:
        parser.error("this contract only permits --token-step 50")
    if args.probe_batch_size < 1:
        parser.error("--probe-batch-size must be positive")
    if args.max_model_len <= PROBE_MAX_TOKENS:
        parser.error("--max-model-len is too small")
    return args


def progress_payload(
    *,
    state: str,
    args: argparse.Namespace,
    assigned: int,
    completed: int,
    fingerprint: str,
    started: float,
    completed_now: int,
    current_qid: str | None = None,
) -> dict[str, Any]:
    elapsed = max(time.time() - started, 1e-9)
    rate = completed_now / elapsed if completed_now else 0.0
    remaining = assigned - completed
    return {
        "state": state,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned": assigned,
        "completed": completed,
        "remaining": remaining,
        "current_qid": current_qid,
        "samples_per_second": rate,
        "eta_seconds": remaining / rate if rate else None,
        "generation_config_sha256": fingerprint,
        "updated_unix": time.time(),
    }


def main() -> None:
    args = parse_args()
    rows_all = load_source(args.input, args.expected_qids)
    qids_all = [str(row["qid"]) for row in rows_all]
    selected = rows_all[args.shard_index :: args.shard_count]
    if args.limit:
        selected = selected[: args.limit]
    selected_qids = [str(row["qid"]) for row in selected]
    allowed_qids = set(selected_qids)

    feature_module = Path(__file__).with_name("math_cluster_features.py")
    old_fingerprints = sorted(
        {str(row["generation_config_sha256"]) for row in rows_all}
    )
    generation_config: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "implementation": "gamma_vllm09_public_generate_saved_cot_replay_every50_v1",
        "model": model_identity(args.model),
        "source": {
            "path": str(args.input),
            "sha256": file_sha256(args.input),
            "qid_count": len(rows_all),
            "ordered_qids_sha256": qids_sha256(qids_all),
            "source_generation_config_sha256_values": old_fingerprints,
            "main_reused_without_regeneration": True,
        },
        "selection": {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "limit": args.limit,
            "qid_count": len(selected),
            "ordered_qids_sha256": qids_sha256(selected_qids),
        },
        "prompt": {
            "template": "qwen3_chat_plus_explicit_think_v1",
            "system_prompt": SYSTEM_PROMPT,
            "enable_thinking": True,
            "explicit_think_open": True,
        },
        "probe": {
            "schedule": "every_50_main_think_tokens_after_text_retokenization",
            "token_step": TOKEN_STEP,
            "suffix": PROBE_SUFFIX,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "repetition_penalty": 1.2,
            "max_tokens": PROBE_MAX_TOKENS,
            "logprobs": PROBE_TOPK,
            "seed": args.seed,
            "balanced_outer_box_boundary": True,
        },
        "features": {
            "implementation": "ClusterEvidenceTracker_answer_logprob_stats_v1",
            "module_sha256": file_sha256(feature_module),
            "order": list(FEATURES),
            "count": len(FEATURES),
            "raw_logprobs_saved": False,
        },
        "engine": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "probe_batch_size": args.probe_batch_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": True,
            "enforce_eager": True,
        },
    }
    fingerprint = hashlib.sha256(
        canonical_json(generation_config).encode("utf-8")
    ).hexdigest()

    done = completed_qids(args.output, fingerprint, allowed_qids)
    pending = [row for row in selected if str(row["qid"]) not in done]
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    args.errors.touch(exist_ok=True)
    if args.errors.stat().st_size:
        raise RuntimeError(
            f"Refusing ambiguous resume with non-empty error ledger: {args.errors}"
        )

    if not pending:
        atomic_json(
            args.progress,
            progress_payload(
                state="complete",
                args=args,
                assigned=len(selected),
                completed=len(selected),
                fingerprint=fingerprint,
                started=started,
                completed_now=0,
            ),
        )
        return

    # Keep the heavy runtime imports below all source/resume contract checks.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    close_think_ids = [
        int(token) for token in tokenizer.encode("</think>", add_special_tokens=False)
    ]
    suffix_ids = [
        int(token) for token in tokenizer.encode(PROBE_SUFFIX, add_special_tokens=False)
    ]
    if not close_think_ids or not suffix_ids:
        raise RuntimeError("Tokenizer produced empty close-think/probe-suffix ids")

    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.probe_batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=20,
        repetition_penalty=1.2,
        max_tokens=PROBE_MAX_TOKENS,
        logprobs=PROBE_TOPK,
        seed=args.seed,
    )

    completed_now = 0
    with args.output.open("a", encoding="utf-8") as output_handle:
        for source_row in pending:
            qid = str(source_row["qid"])
            main_record = dict(source_row["main"])
            main_text = str(main_record["text"])
            positions, reasoning_tokens, close_found, reencoded_tokens = (
                expected_steps_for_main(tokenizer, main_text, close_think_ids)
            )
            generated_ids = [
                int(token)
                for token in tokenizer.encode(main_text, add_special_tokens=False)
            ]
            prompt_ids = build_prompt_ids(tokenizer, str(source_row["problem"]))
            tracker = ClusterEvidenceTracker(recent_window=5, delta_window=3)
            probes: list[dict[str, Any]] = []

            for batch_start in range(0, len(positions), args.probe_batch_size):
                batch_positions = positions[
                    batch_start : batch_start + args.probe_batch_size
                ]
                prompt_token_ids = [
                    prompt_ids + generated_ids[:position] + suffix_ids
                    for position in batch_positions
                ]
                if prompt_token_ids and (
                    max(map(len, prompt_token_ids)) + PROBE_MAX_TOKENS
                    > args.max_model_len
                ):
                    raise RuntimeError(f"{qid}: probe exceeds max_model_len")
                outputs = llm.generate(
                    [{"prompt_token_ids": ids} for ids in prompt_token_ids],
                    sampling,
                    use_tqdm=False,
                )
                if len(outputs) != len(batch_positions):
                    raise RuntimeError(f"{qid}: vLLM probe output count mismatch")

                for position, request_ids, request_output in zip(
                    batch_positions, prompt_token_ids, outputs
                ):
                    candidate = request_output.outputs[0]
                    trimmed = trim_probe(tokenizer, candidate)
                    answer = trimmed["answer"]
                    answer_key = normalized_answer_key(answer)
                    stats = answer_logprob_stats(trimmed["steps"])
                    eligible = bool(
                        answer_key
                        and trimmed["box_closed"]
                        and int(stats["ans_len"]) > 0
                    )
                    features = tracker.update(
                        answer_key=answer_key,
                        logprob_stats=stats,
                        probe_index=len(probes) + 1,
                    )
                    if tuple(features) != FEATURES:
                        raise RuntimeError(f"{qid}: cluster feature order drift")
                    probe_record = {
                        "probe_index": len(probes) + 1,
                        "step_tokens": int(position),
                        "probe_answer": str(answer or ""),
                        "answer_key": str(answer_key or ""),
                        "eligible": eligible,
                        "box_closed": bool(trimmed["box_closed"]),
                        "probe_text": str(trimmed["text"]),
                        "probe_prompt_tokens": len(request_ids),
                        "probe_output_tokens": int(trimmed["output_tokens"]),
                        "probe_generated_output_tokens": int(
                            trimmed["generated_output_tokens"]
                        ),
                        "probe_answer_logprob_tokens": int(stats["ans_len"]),
                        "features": {
                            name: float(features[name]) for name in FEATURES
                        },
                    }
                    probes.append(probe_record)

            expected_positions = [probe["step_tokens"] for probe in probes]
            if expected_positions != positions:
                raise RuntimeError(f"{qid}: incomplete or reordered every-50 probes")
            record = {
                "format_version": FORMAT_VERSION,
                "record_schema_version": RECORD_SCHEMA_VERSION,
                "generation_config": generation_config,
                "generation_config_sha256": fingerprint,
                "qid": qid,
                "source_index": int(source_row.get("source_index", -1)),
                "sample_rank": str(source_row.get("sample_rank", "")),
                "sample_stratum": str(source_row.get("sample_stratum", "")),
                "problem": str(source_row["problem"]),
                "gold_answer": str(source_row["gold_answer"]),
                "main": main_record,
                "replay": {
                    "response_text_reencoded_tokens": reencoded_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "close_think_found": close_found,
                    "probe_count": len(probes),
                },
                "probes": probes,
            }
            # Preserve FEATURES insertion order in the artifact.  Fingerprints
            # use canonical_json above, but records intentionally do not sort
            # nested keys so a reader can enforce the exact feature vector.
            output_handle.write(
                json.dumps(
                    json_safe(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())
            completed_now += 1
            atomic_json(
                args.progress,
                progress_payload(
                    state="running",
                    args=args,
                    assigned=len(selected),
                    completed=len(done) + completed_now,
                    fingerprint=fingerprint,
                    started=started,
                    completed_now=completed_now,
                    current_qid=qid,
                ),
            )

    final_done = completed_qids(args.output, fingerprint, allowed_qids)
    if final_done != allowed_qids:
        raise RuntimeError(
            f"Incomplete replay shard: completed={len(final_done)} expected={len(allowed_qids)}"
        )
    atomic_json(
        args.progress,
        progress_payload(
            state="complete",
            args=args,
            assigned=len(selected),
            completed=len(selected),
            fingerprint=fingerprint,
            started=started,
            completed_now=completed_now,
        ),
    )


if __name__ == "__main__":
    main()
