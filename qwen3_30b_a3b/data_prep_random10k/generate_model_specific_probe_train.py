#!/usr/bin/env python3
"""Generate ESTAR open-QA classifier records from DeepScaleR.

For every full sampled CoT, force an answer at 10%, ..., 100% of the exact
reasoning token IDs.  The exact IDs are passed back to vLLM, so forced probes
can reuse prefix-cache blocks without a decode/encode round trip.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.probe_features_vllm import (
    OpenSeqFeatureTracker,
    compute_openqa_slot_from_vllm_steps,
)

from inference_short_math_deep import (
    extract_boxed_answer,
    last_boxed_only_string,
    mathd_normalize_answer,
)


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"
PROMPT_PROTOCOLS = ("qwen3_system", "deepseek_user")
FORMAT_VERSION = 2
RECORD_SCHEMA_VERSION = "deepscaler_probe_train_v2"
FULL_ANSWER_BOUNDARY_VERSION = "after_first_exact_close_think_v1"
PROBE_BRACE_BOUNDARY_VERSION = "first_balanced_outer_brace_token_boundary_v1"
CLASSIFIER_FEATURES = [
    "L_sum",
    "S_es",
    "H_es",
    "ans_len",
    "run_len",
    "flips",
    "changed_prev",
    "mean_logprob",
    "var_logprob",
    "neg_ppl",
]


def stable_seed(base_seed: int, qid: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{qid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_qids_sha256(qids: list[str]) -> str:
    canonical = json.dumps(
        qids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_generation_config(
    args: argparse.Namespace,
    *,
    input_sha256: str,
    all_qids: list[str],
    selected_qids: list[str],
) -> dict[str, Any]:
    """Return the run-wide, qid-independent generation contract."""
    return {
        "format_version": FORMAT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "full_answer_boundary_version": FULL_ANSWER_BOUNDARY_VERSION,
        "probe_brace_boundary_version": PROBE_BRACE_BOUNDARY_VERSION,
        # The load path can differ across hosts even when the model files are
        # byte-identical.  Keep the semantic model identity stable so a
        # checkpoint can pass strict resume validation after migration.
        "model": str(args.generation_model_id or args.model),
        "dtype": str(args.dtype),
        "seed": int(args.seed),
        "source": {
            "input_sha256": input_sha256,
            "row_count": len(all_qids),
            "ordered_qids_sha256": ordered_qids_sha256(all_qids),
        },
        "selection": {
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "limit": int(args.limit),
            "row_count": len(selected_qids),
            "ordered_qids_sha256": ordered_qids_sha256(selected_qids),
        },
        "prompt": {
            "template": (
                "qwen3_chat_plus_explicit_think_v1"
                if args.prompt_protocol == "qwen3_system"
                else "deepseek_single_user_plus_explicit_think_v1"
            ),
            "protocol": str(args.prompt_protocol),
            "system_prompt": (
                SYSTEM_PROMPT if args.prompt_protocol == "qwen3_system" else None
            ),
            "user_prefix": (
                SYSTEM_PROMPT if args.prompt_protocol == "deepseek_user" else None
            ),
            "enable_thinking": True,
            "explicit_think_open": True,
        },
        "main_sampling": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.topk),
            "repetition_penalty": float(args.repetition_penalty),
            "max_tokens": int(args.max_main_tokens),
        },
        "probe": {
            "qa_mode": "openqa",
            "max_tokens": int(args.probe_max_tokens),
            "top_k_logprobs": int(args.topk),
            "checkpoints": [index / 10.0 for index in range(1, 11)],
            "tracker_window": 5,
            "tracker_recent": 3,
        },
        "max_model_len": int(args.max_model_len),
        "engine": {
            "enable_prefix_caching": True,
            "enforce_eager": True,
            "engine_max_num_seqs": int(args.engine_max_num_seqs),
            "max_batched_tokens": int(args.max_batched_tokens),
            "max_concurrency": int(args.max_concurrency),
            "max_inflight_rows": int(args.max_inflight_rows),
            "max_probe_concurrency": int(args.max_probe_concurrency),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "tensor_parallel_size": int(args.tensor_parallel_size),
        },
        "exact_token_id_prefix": True,
        "save_raw_logprobs": bool(args.save_raw_logprobs),
    }


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        json_safe(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def split_after_first_close_think(text: str) -> tuple[str, bool]:
    """Return only text after the first exact ``</think>`` marker."""
    marker = "</think>"
    position = text.find(marker)
    if position < 0:
        return "", False
    return text[position + len(marker):], True


def parse_openqa_probe_answer(probe_text: str) -> tuple[str | None, int | None]:
    """Parse text generated after an already-open ``\\boxed{``."""
    if not isinstance(probe_text, str):
        return None, None
    depth = 1
    chars: list[str] = []
    for index, char in enumerate(probe_text):
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and probe_text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        escaped = bool(slash_count % 2)
        if not escaped and char == "{":
            depth += 1
        elif not escaped and char == "}":
            depth -= 1
            if depth == 0:
                answer = "".join(chars).strip()
                return answer or None, index + 1
        chars.append(char)
    return None, None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def source_qids(rows: list[dict[str, Any]], path: Path) -> list[str]:
    """Return source qids after rejecting empty or duplicate identifiers."""
    qids: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        qid = str(row.get("unique_id") or "")
        if not qid:
            raise ValueError(f"{path}: row {row_number} has an empty unique_id")
        if qid in seen:
            raise ValueError(f"{path}: duplicate unique_id {qid!r}")
        seen.add(qid)
        qids.append(qid)
    return qids


def select_assigned_rows(
    rows: list[dict[str, Any]],
    qids: list[str],
    *,
    shard_index: int,
    shard_count: int,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select one shard and apply a smoke-test limit to rows and qids together."""
    if len(rows) != len(qids):
        raise ValueError("rows/qids length mismatch")
    selected = [
        (row, qid)
        for position, (row, qid) in enumerate(zip(rows, qids))
        if position % shard_count == shard_index
    ]
    if limit > 0:
        selected = selected[:limit]
    return [row for row, _ in selected], [qid for _, qid in selected]


def _require_nonnegative_int(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}: expected a non-negative integer, got {value!r}")


def _finite_number(value: Any, *, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location}: expected a finite number, got boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: expected a number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location}: expected a finite number, got {value!r}")
    return result


def validate_completed_record(
    obj: dict[str, Any],
    *,
    location: str,
    allowed_qids: set[str],
    expected_config: dict[str, Any],
    expected_fingerprint: str,
) -> str:
    """Validate that a resume record exactly matches the current run contract."""
    if obj.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{location}: incompatible format_version={obj.get('format_version')!r}; "
            f"expected {FORMAT_VERSION} ({RECORD_SCHEMA_VERSION})"
        )

    qid = obj.get("qid")
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"{location}: empty or non-string qid")
    if qid not in allowed_qids:
        raise ValueError(f"{location}: qid {qid!r} is not assigned to this run")

    config = obj.get("generation_config")
    fingerprint = obj.get("generation_config_sha256")
    if not isinstance(config, dict):
        raise TypeError(f"{location}: generation_config is not an object")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"{location}: missing generation_config_sha256")
    actual_fingerprint = config_sha256(config)
    if fingerprint != actual_fingerprint:
        raise ValueError(
            f"{location}: generation_config_sha256 does not match the stored config"
        )
    if fingerprint != expected_fingerprint or config != expected_config:
        raise ValueError(
            f"{location}: generation configuration mismatch; existing={fingerprint} "
            f"expected={expected_fingerprint}. Refusing incompatible resume data."
        )

    main = obj.get("main")
    if not isinstance(main, dict):
        raise TypeError(f"{location}: main is not an object")
    required_main = {
        "text",
        "final_text_after_think",
        "full_answer",
        "full_answer_key",
        "prompt_tokens",
        "output_tokens",
        "cot_tokens",
        "close_think_found",
        "finish_reason",
        "box_closed",
        "num_cached_tokens",
        "request_seed",
    }
    missing_main = sorted(required_main - set(main))
    if missing_main:
        raise ValueError(f"{location}: main is missing fields {missing_main}")
    for field in ("text", "final_text_after_think", "full_answer", "full_answer_key"):
        if not isinstance(main[field], str):
            raise TypeError(f"{location}: main.{field} must be a string")
    for field in (
        "prompt_tokens",
        "output_tokens",
        "cot_tokens",
        "num_cached_tokens",
        "request_seed",
    ):
        _require_nonnegative_int(main[field], location=f"{location}: main.{field}")
    for field in ("close_think_found", "box_closed"):
        if type(main[field]) is not bool:
            raise TypeError(f"{location}: main.{field} must be boolean")
    if main["finish_reason"] is not None and not isinstance(main["finish_reason"], str):
        raise TypeError(f"{location}: main.finish_reason must be a string or null")
    expected_request_seed = stable_seed(int(expected_config["seed"]), qid)
    if main["request_seed"] != expected_request_seed:
        raise ValueError(
            f"{location}: main.request_seed={main['request_seed']} does not match "
            f"the deterministic qid seed {expected_request_seed}"
        )

    derived_final_text, derived_close_found = split_after_first_close_think(main["text"])
    if main["close_think_found"] is not derived_close_found:
        raise ValueError(f"{location}: main.close_think_found disagrees with main.text")
    if main["final_text_after_think"] != derived_final_text:
        raise ValueError(
            f"{location}: main.final_text_after_think is not the first </think> suffix"
        )
    derived_answer = extract_boxed_answer(derived_final_text) or ""
    derived_box_closed = last_boxed_only_string(derived_final_text) is not None
    if main["full_answer"] != derived_answer:
        raise ValueError(
            f"{location}: main.full_answer was not extracted from final_text_after_think"
        )
    if main["box_closed"] is not derived_box_closed:
        raise ValueError(
            f"{location}: main.box_closed disagrees with final_text_after_think"
        )
    if main["full_answer_key"] != (normalized_answer_key(derived_answer) or ""):
        raise ValueError(f"{location}: main.full_answer_key is inconsistent")

    probes = obj.get("probes")
    if not isinstance(probes, list) or len(probes) != 10:
        raise ValueError(f"{location}: probes must be a list of exactly 10 records")
    probe_indices: list[int] = []
    probe_steps: dict[int, int] = {}
    for position, probe in enumerate(probes, 1):
        if not isinstance(probe, dict):
            raise TypeError(f"{location}: probe {position} is not an object")
        required_probe = {
            "probe_index",
            "slice_fraction",
            "step_tokens",
            "probe_text",
            "probe_answer",
            "box_closed",
            "answer_key",
            "eligible",
            "slot",
            "features",
            "num_cached_tokens",
            "num_direct_forked_tokens",
            "probe_prompt_tokens",
            "probe_output_tokens",
            "probe_answer_logprob_tokens",
            "probe_finish_reason",
        }
        missing_probe = sorted(required_probe - set(probe))
        if missing_probe:
            raise ValueError(
                f"{location}: probe {position} is missing fields {missing_probe}"
            )
        probe_index = probe.get("probe_index")
        if isinstance(probe_index, bool) or not isinstance(probe_index, int):
            raise TypeError(f"{location}: probe {position} has invalid probe_index")
        probe_indices.append(probe_index)
        probe_location = f"{location}: probe_index={probe_index}"
        _require_nonnegative_int(
            probe["step_tokens"], location=f"{probe_location}.step_tokens"
        )
        if probe["step_tokens"] < 1:
            raise ValueError(f"{probe_location}: step_tokens must be positive")
        probe_steps[probe_index] = probe["step_tokens"]
        slice_fraction = _finite_number(
            probe["slice_fraction"], location=f"{probe_location}.slice_fraction"
        )
        if not math.isclose(
            slice_fraction, probe_index / 10.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{probe_location}: slice_fraction is inconsistent")
        for field in ("probe_text", "probe_answer", "answer_key"):
            if not isinstance(probe[field], str):
                raise TypeError(f"{probe_location}.{field} must be a string")
        for field in ("box_closed", "eligible"):
            if type(probe[field]) is not bool:
                raise TypeError(f"{probe_location}.{field} must be boolean")
        for field in (
            "num_cached_tokens",
            "num_direct_forked_tokens",
            "probe_prompt_tokens",
            "probe_output_tokens",
            "probe_answer_logprob_tokens",
        ):
            _require_nonnegative_int(
                probe[field], location=f"{probe_location}.{field}"
            )
        if probe["probe_answer_logprob_tokens"] > probe["probe_output_tokens"]:
            raise ValueError(
                f"{probe_location}: answer logprob tokens exceed output tokens"
            )
        probe_max_tokens = int(expected_config["probe"]["max_tokens"])
        if probe["probe_output_tokens"] > probe_max_tokens:
            raise ValueError(
                f"{probe_location}: output tokens exceed configured probe maximum"
            )
        if (
            probe["probe_finish_reason"] is not None
            and not isinstance(probe["probe_finish_reason"], str)
        ):
            raise TypeError(
                f"{probe_location}.probe_finish_reason must be a string or null"
            )

        derived_probe_answer, close_char_end = parse_openqa_probe_answer(
            probe["probe_text"]
        )
        derived_box_closed = close_char_end is not None
        if close_char_end is not None and close_char_end != len(probe["probe_text"]):
            raise ValueError(f"{probe_location}: probe_text extends past first outer brace")
        if probe["box_closed"] is not derived_box_closed:
            raise ValueError(f"{probe_location}: box_closed is inconsistent")
        if probe["probe_answer"] != (derived_probe_answer or ""):
            raise ValueError(f"{probe_location}: probe_answer is inconsistent")
        derived_answer_key = normalized_answer_key(derived_probe_answer) or ""
        if probe["answer_key"] != derived_answer_key:
            raise ValueError(f"{probe_location}: answer_key is inconsistent")

        slot = probe["slot"]
        if not isinstance(slot, dict):
            raise TypeError(f"{probe_location}.slot must be an object")
        required_slot = {
            "L_sum",
            "ans_len",
            "mean_logprob",
            "var_logprob",
            "neg_ppl",
            "early_stop_elig",
            "probe_letter",
        }
        missing_slot = sorted(required_slot - set(slot))
        if missing_slot:
            raise ValueError(f"{probe_location}.slot is missing {missing_slot}")
        for field in ("L_sum", "mean_logprob", "var_logprob", "neg_ppl"):
            _finite_number(slot[field], location=f"{probe_location}.slot.{field}")
        _require_nonnegative_int(
            slot["ans_len"], location=f"{probe_location}.slot.ans_len"
        )
        if slot["ans_len"] > probe["probe_answer_logprob_tokens"]:
            raise ValueError(
                f"{probe_location}: slot ans_len exceeds answer logprob tokens"
            )
        if type(slot["early_stop_elig"]) is not bool:
            raise TypeError(f"{probe_location}.slot.early_stop_elig must be boolean")
        if slot["probe_letter"] is not None:
            raise ValueError(f"{probe_location}.slot.probe_letter must be null")

        expected_eligible = bool(derived_answer_key and slot["ans_len"] > 0)
        if probe["eligible"] is not expected_eligible:
            raise ValueError(f"{probe_location}: eligible is inconsistent")
        if slot["early_stop_elig"] is not expected_eligible:
            raise ValueError(f"{probe_location}: slot eligibility is inconsistent")

        features = probe["features"]
        if not isinstance(features, dict):
            raise TypeError(f"{probe_location}.features must be an object")
        missing_features = sorted(set(CLASSIFIER_FEATURES) - set(features))
        if missing_features:
            raise ValueError(
                f"{probe_location}.features is missing {missing_features}"
            )
        for field in CLASSIFIER_FEATURES:
            _finite_number(
                features[field], location=f"{probe_location}.features.{field}"
            )
        if "step" not in features:
            raise ValueError(f"{probe_location}.features is missing step")
        feature_step = _finite_number(
            features["step"], location=f"{probe_location}.features.step"
        )
        if feature_step != probe["step_tokens"]:
            raise ValueError(f"{probe_location}: feature step is inconsistent")
        for field in ("L_sum", "ans_len", "mean_logprob", "var_logprob", "neg_ppl"):
            if not math.isclose(
                float(features[field]),
                float(slot[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{probe_location}: feature {field} disagrees with slot"
                )
    if set(probe_indices) != set(range(1, 11)) or len(set(probe_indices)) != 10:
        raise ValueError(f"{location}: probes must contain indices 1..10 exactly once")
    cot_tokens = main["cot_tokens"]
    for probe_index in range(1, 11):
        expected_step = min(
            cot_tokens,
            max(1, math.ceil(cot_tokens * probe_index / 10)),
        )
        if probe_steps[probe_index] != expected_step:
            raise ValueError(
                f"{location}: probe_index={probe_index} step_tokens is inconsistent"
            )
    return qid


def completed_qids(
    path: Path,
    allowed_qids: set[str],
    *,
    expected_config: dict[str, Any],
    expected_fingerprint: str,
) -> set[str]:
    """Strictly validate an existing JSONL output and return its qids."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL record")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_number}: record is not an object")
            qid = validate_completed_record(
                obj,
                location=f"{path}:{line_number}",
                allowed_qids=allowed_qids,
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )
            if qid in done:
                raise ValueError(f"{path}:{line_number}: duplicate qid {qid!r}")
            done.add(qid)
    return done


def find_first_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    first = needle[0]
    for start in range(len(values) - len(needle) + 1):
        if values[start] == first and values[start : start + len(needle)] == needle:
            return start
    return None


def normalized_answer_key(answer: str) -> str | None:
    if not isinstance(answer, str) or not answer.strip():
        return None
    normalized = mathd_normalize_answer(answer)
    if normalized is None or not str(normalized).strip():
        return None
    return str(normalized).strip()


def build_prompt(tokenizer: Any, problem: str, prompt_protocol: str) -> str:
    if prompt_protocol == "qwen3_system":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(problem)},
        ]
    elif prompt_protocol == "deepseek_user":
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{problem}"},
        ]
    else:
        raise ValueError(f"Unsupported prompt protocol: {prompt_protocol!r}")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # Make the forced-stop prefix well formed even at the 0/10% boundary.
    return prompt + "<think>\n"


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def run(args: argparse.Namespace) -> None:
    started = time.time()
    input_fingerprint = file_sha256(args.input)
    all_rows = load_jsonl(args.input)
    if file_sha256(args.input) != input_fingerprint:
        raise RuntimeError("Input JSONL changed while it was being loaded")
    all_source_qids = source_qids(all_rows, args.input)
    shard_rows, shard_qids = select_assigned_rows(
        all_rows,
        all_source_qids,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit=args.limit,
    )
    generation_config = build_generation_config(
        args,
        input_sha256=input_fingerprint,
        all_qids=all_source_qids,
        selected_qids=shard_qids,
    )
    generation_config_fingerprint = config_sha256(generation_config)
    assigned_qids = set(shard_qids)
    already_done = (
        completed_qids(
            args.output,
            assigned_qids,
            expected_config=generation_config,
            expected_fingerprint=generation_config_fingerprint,
        )
        if args.resume
        else set()
    )
    rows = [row for row in shard_rows if str(row.get("unique_id", "")) not in already_done]

    if args.validate_resume_only:
        print(
            json.dumps(
                {
                    "status": "resume_validation_passed",
                    "assigned": len(assigned_qids),
                    "completed": len(already_done),
                    "pending": len(rows),
                    "generation_config_sha256": generation_config_fingerprint,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return

    if already_done == assigned_qids:
        write_progress(
            args.progress,
            {
                "status": "complete",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "assigned": len(assigned_qids),
                "resumed": len(already_done),
                "pending_at_start": 0,
                "completed_this_run": 0,
                "failures_this_run": 0,
                "output_records": len(already_done),
                "missing_records": 0,
                "generation_config_sha256": generation_config_fingerprint,
                "elapsed_seconds": max(time.time() - started, 0.0),
                "updated_unix": time.time(),
            },
        )
        print(
            f"[COMPLETE] all {len(assigned_qids)} assigned qids already pass "
            "strict resume validation",
            flush=True,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    close_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if not close_think_ids:
        raise RuntimeError("Tokenizer produced no IDs for </think>")

    print(
        f"[DATA] all={len(all_rows)} shard={args.shard_index}/{args.shard_count} "
        f"assigned={len(shard_rows)} resumed={len(already_done)} pending={len(rows)}",
        flush=True,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.engine_max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=args.max_model_len,
        disable_log_requests=True,
        disable_log_stats=True,
        enforce_eager=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    main_sem = asyncio.Semaphore(args.max_concurrency)
    probe_sem = asyncio.Semaphore(args.max_probe_concurrency)

    async def one_probe(_parent_request_id: str, prefix_ids: list[int]) -> dict[str, Any]:
        async with probe_sem:
            return await engine._probe_once(
                base_prompt="",
                think_accum="",
                suffix="",
                probe_max_steps=args.probe_max_tokens,
                topk=args.topk,
                qa_mode="openqa",
                # The full rollout has completed, so its request is no longer
                # resident.  Pass exact IDs without a parent hint: ordinary
                # APC reuses the saved full blocks and avoids noisy/direct-fork
                # fallback warnings.
                direct_fork_parent_request_id=None,
                direct_fork_prefix_token_ids=prefix_ids,
            )

    async def one_row(row: dict[str, Any]) -> dict[str, Any]:
        qid = str(row.get("unique_id") or "")
        if not qid:
            raise ValueError("Every sampled DeepScaleR row must have a non-empty unique_id")
        prompt = build_prompt(
            tokenizer,
            str(row.get("problem", "")),
            args.prompt_protocol,
        )
        request_id = f"train-main-{qid.replace('/', '-') }"
        main_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_main_tokens,
            seed=stable_seed(args.seed, qid),
        )
        async with main_sem:
            main = await engine._run_once(
                prompt=prompt,
                sampling_params=main_params,
                request_id=request_id,
                stop_on_first_finish=True,
            )
        prompt_ids = [int(token) for token in main.get("prompt_token_ids") or []]
        output_ids = [int(token) for token in main.get("output_token_ids") or []]
        if not prompt_ids or not output_ids:
            raise RuntimeError(
                f"Missing exact token IDs: prompt={len(prompt_ids)} output={len(output_ids)}"
            )

        close_position = find_first_subsequence(output_ids, close_think_ids)
        if close_position is None:
            cot_ids = output_ids
        else:
            cot_ids = output_ids[:close_position]
        if not cot_ids:
            raise RuntimeError("Generated reasoning span is empty")

        cuts = [
            min(len(cot_ids), max(1, math.ceil(len(cot_ids) * fraction / 10)))
            for fraction in range(1, 11)
        ]
        probe_tasks = [
            asyncio.create_task(
                one_probe(request_id, prompt_ids + cot_ids[:cut])
            )
            for cut in cuts
        ]
        raw_probes = await asyncio.gather(*probe_tasks)

        tracker = OpenSeqFeatureTracker(W=5, K_recent=3)
        probes: list[dict[str, Any]] = []
        for probe_index, (cut, raw_probe) in enumerate(zip(cuts, raw_probes), 1):
            slot = compute_openqa_slot_from_vllm_steps(
                steps_logprobs=raw_probe.get("steps_logprobs") or [],
                topk=args.topk,
            )
            probe_text = str(raw_probe.get("probe_text") or "")
            probe_answer, probe_close_char_end = parse_openqa_probe_answer(
                probe_text
            )
            engine_probe_answer = engine._extract_openqa_probe_answer(probe_text)
            if probe_answer != engine_probe_answer:
                raise RuntimeError(
                    "Generator and engine openQA brace parsers disagree"
                )
            if (
                probe_close_char_end is not None
                and probe_close_char_end != len(probe_text)
            ):
                raise RuntimeError(
                    "Probe text was not truncated at the first closed outer brace"
                )
            raw_box_closed = raw_probe.get("box_closed")
            if type(raw_box_closed) is not bool:
                raise RuntimeError("openQA probe did not return boolean box_closed")
            if raw_box_closed is not (probe_close_char_end is not None):
                raise RuntimeError("Probe parser and raw box_closed disagree")
            answer_key = normalized_answer_key(probe_answer)
            eligible = bool(answer_key is not None and int(slot.get("ans_len", 0)) > 0)
            slot["early_stop_elig"] = eligible
            if answer_key is None:
                tracker.prev_key = None
                tracker.run_len = 0
            features = tracker.update_with_slot(slot, answer_key=answer_key)
            if answer_key is None:
                features["run_len"] = 0
            features["step"] = int(cut)
            record = {
                "probe_index": probe_index,
                "slice_fraction": probe_index / 10.0,
                "step_tokens": int(cut),
                "probe_text": probe_text,
                "probe_answer": probe_answer or "",
                "box_closed": raw_box_closed,
                "answer_key": answer_key or "",
                "eligible": eligible,
                "slot": json_safe(slot),
                "features": json_safe(features),
                "num_cached_tokens": int(raw_probe.get("num_cached_tokens") or 0),
                "num_direct_forked_tokens": int(
                    raw_probe.get("num_direct_forked_tokens") or 0
                ),
                "probe_prompt_tokens": int(raw_probe.get("probe_prompt_tokens") or 0),
                "probe_output_tokens": int(
                    raw_probe.get("probe_output_tokens") or 0
                ),
                "probe_answer_logprob_tokens": int(
                    raw_probe.get("probe_answer_logprob_tokens") or 0
                ),
                "probe_finish_reason": raw_probe.get("probe_finish_reason"),
            }
            if args.save_raw_logprobs:
                record["steps_logprobs"] = json_safe(raw_probe.get("steps_logprobs") or [])
            probes.append(record)

        full_text = str(main.get("text") or "")
        final_text_after_think, text_close_think_found = (
            split_after_first_close_think(full_text)
        )
        token_close_think_found = close_position is not None
        if token_close_think_found != text_close_think_found:
            raise RuntimeError(
                "The exact-token and decoded-text </think> boundaries disagree"
            )
        # Never accept a boxed expression from the reasoning region as the
        # full answer.  Only the suffix after the first </think> is final.
        full_answer = extract_boxed_answer(final_text_after_think) or ""
        full_box_closed = (
            last_boxed_only_string(final_text_after_think) is not None
        )
        request_seed = stable_seed(args.seed, qid)
        return {
            "format_version": FORMAT_VERSION,
            "qid": qid,
            "source_index": int(row.get("source_index", -1)),
            "sample_rank": str(row.get("sample_rank", "")),
            "sample_stratum": str(row.get("sample_stratum", "")),
            "problem": str(row.get("problem", "")),
            "gold_answer": str(row.get("answer", "")),
            "generation_config": generation_config,
            "generation_config_sha256": generation_config_fingerprint,
            "main": {
                "text": full_text,
                "final_text_after_think": final_text_after_think,
                "full_answer": full_answer,
                "full_answer_key": normalized_answer_key(full_answer) or "",
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "cot_tokens": len(cot_ids),
                "close_think_found": text_close_think_found,
                "finish_reason": main.get("finish_reason"),
                "box_closed": full_box_closed,
                "num_cached_tokens": int(main.get("num_cached_tokens") or 0),
                "request_seed": request_seed,
            },
            "probes": probes,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.resume else "w"
    output_handle = args.output.open(file_mode, encoding="utf-8")
    error_handle = args.errors.open(file_mode, encoding="utf-8")

    total_pending = len(rows)
    completed = 0
    failures = 0
    total_cot_tokens = 0
    total_cached_tokens = 0
    total_probe_prompt_tokens = 0
    async def tagged_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, Exception | None]:
        try:
            return row, await one_row(row), None
        except Exception as exc:  # keep the source qid attached to failures
            return row, None, exc

    # Keep only a bounded window of row tasks alive.  A 10k-item task list
    # retains each completed result until the entire list is released and can
    # consume many GB when every result contains a long rollout and 10 probes.
    row_iterator = iter(rows)
    pending: set[asyncio.Task[Any]] = set()

    def schedule_next_row() -> bool:
        try:
            row = next(row_iterator)
        except StopIteration:
            return False
        pending.add(asyncio.create_task(tagged_row(row)))
        return True

    for _ in range(min(args.max_inflight_rows, total_pending)):
        schedule_next_row()
    try:
        while pending:
            finished, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for future in finished:
                row, result, exc = future.result()
                if exc is not None:
                    failures += 1
                    qid = str(row.get("unique_id", "unknown"))
                    error_handle.write(
                        json.dumps(
                            {"qid": qid, "error": f"{type(exc).__name__}: {exc}"},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    error_handle.flush()
                else:
                    assert result is not None
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    completed += 1
                    total_cot_tokens += int(result["main"]["cot_tokens"])
                    for probe in result["probes"]:
                        total_cached_tokens += int(probe["num_cached_tokens"])
                        total_probe_prompt_tokens += int(probe["probe_prompt_tokens"])

                observed = completed + failures
                if (
                    observed == 1
                    or observed % args.progress_every == 0
                    or observed == total_pending
                ):
                    elapsed = max(time.time() - started, 1e-6)
                    rate = observed / elapsed
                    remaining = total_pending - observed
                    payload = {
                        "status": (
                            "running"
                            if remaining
                            else ("failed" if failures else "complete")
                        ),
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "assigned": len(shard_rows),
                        "resumed": len(already_done),
                        "pending_at_start": total_pending,
                        "completed_this_run": completed,
                        "failures_this_run": failures,
                        "observed_this_run": observed,
                        "remaining": remaining,
                        "elapsed_seconds": elapsed,
                        "samples_per_second": rate,
                        "eta_seconds": remaining / rate if rate else None,
                        "mean_cot_tokens": (
                            total_cot_tokens / completed if completed else None
                        ),
                        "probe_prefix_cache_hit_ratio": (
                            total_cached_tokens / total_probe_prompt_tokens
                            if total_probe_prompt_tokens
                            else None
                        ),
                        "generation_config_sha256": generation_config_fingerprint,
                        "updated_unix": time.time(),
                    }
                    write_progress(args.progress, payload)
                    print(
                        f"[PROGRESS] observed={observed}/{total_pending} ok={completed} "
                        f"failed={failures} rate={rate:.4f}/s "
                        f"ETA={payload['eta_seconds']:.0f}s "
                        f"mean_cot={payload['mean_cot_tokens'] or 0:.1f} "
                        f"cache_hit={payload['probe_prefix_cache_hit_ratio'] or 0:.3f}",
                        flush=True,
                    )
                schedule_next_row()
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        output_handle.close()
        error_handle.close()
        # Predicate-stopped probes can leave one already-dispatched GPU step
        # finishing after their abort acknowledgement.  Round-trip through
        # EngineCore before shutdown so that normal teardown does not signal
        # the worker in the middle of that harmless final step.
        try:
            await asyncio.wait_for(engine.debug_kv_fork_state([]), timeout=5.0)
        except Exception:
            pass
        engine.shutdown()

    # A generation command is successful only when its output is a strict,
    # one-record-per-assigned-qid dataset.  Row exceptions are recorded above,
    # but must also make the process fail so a shell pipeline cannot continue
    # to classifier training with partial data.
    final_qids = completed_qids(
        args.output,
        assigned_qids,
        expected_config=generation_config,
        expected_fingerprint=generation_config_fingerprint,
    )
    missing_qids = assigned_qids - final_qids
    if failures or final_qids != assigned_qids:
        write_progress(
            args.progress,
            {
                "status": "failed",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "assigned": len(assigned_qids),
                "resumed": len(already_done),
                "completed_this_run": completed,
                "failures_this_run": failures,
                "output_records": len(final_qids),
                "missing_records": len(missing_qids),
                "missing_qid_examples": sorted(missing_qids)[:20],
                "generation_config_sha256": generation_config_fingerprint,
                "elapsed_seconds": max(time.time() - started, 0.0),
                "updated_unix": time.time(),
            },
        )
        raise RuntimeError(
            "Generation incomplete: "
            f"assigned={len(assigned_qids)} output={len(final_qids)} "
            f"failures_this_run={failures} missing={len(missing_qids)}"
        )

    write_progress(
        args.progress,
        {
            "status": "complete",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "assigned": len(assigned_qids),
            "resumed": len(already_done),
            "completed_this_run": completed,
            "failures_this_run": failures,
            "output_records": len(final_qids),
            "missing_records": 0,
            "generation_config_sha256": generation_config_fingerprint,
            "elapsed_seconds": max(time.time() - started, 0.0),
            "updated_unix": time.time(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-protocol",
        choices=PROMPT_PROTOCOLS,
        default="qwen3_system",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--generation-model-id",
        default=None,
        help=(
            "Stable model identity stored in the generation contract when the "
            "same byte-identical model is mounted at a different local path."
        ),
    )
    parser.add_argument("--seed", type=int, default=20_260_722)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-resume-only",
        action="store_true",
        help="Validate existing records and their generation contract without loading a model.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N assigned rows (smoke tests).")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-main-tokens", type=int, default=20_000)
    parser.add_argument("--probe-max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=24_576)
    parser.add_argument("--max-batched-tokens", type=int, default=32_768)
    parser.add_argument("--engine-max-num-seqs", type=int, default=64)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-inflight-rows", type=int, default=32)
    parser.add_argument("--max-probe-concurrency", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-raw-logprobs", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard_count >= 1 and 0 <= shard_index < shard_count")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.validate_resume_only and not args.resume:
        parser.error("--validate-resume-only requires --resume")
    if args.max_concurrency < 1 or args.max_probe_concurrency < 1:
        parser.error("Concurrency limits must be positive")
    if args.max_inflight_rows < args.max_concurrency:
        parser.error("--max-inflight-rows must be >= --max-concurrency")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
