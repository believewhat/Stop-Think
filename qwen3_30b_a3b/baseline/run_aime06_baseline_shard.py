#!/usr/bin/env python3
"""Run one Qwen3-30B-A3B AIME2024 baseline shard at temperature 0.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

from math_verify import parse as math_parse, verify as math_verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"
PROMPT_VERSION = "qwen3_system_user_explicit_think_v1"
INVALID_FINISH_REASONS = {"length", "max_tokens"}
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.2


class GradeTimeout(Exception):
    """Raised when symbolic answer verification exceeds its per-row budget."""


def grade_timeout(_signum: int, _frame: Any) -> None:
    raise GradeTimeout()


def canonical(text: str) -> str:
    value = str(text or "").strip().replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\s+", "", value)
    return value.strip("$")


def last_boxed(text: str) -> tuple[str, bool]:
    """Return the content of the last balanced boxed expression in ``text``."""
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return "", False
    brace = text.find("{", start)
    depth = 0
    chars: list[str] = []
    for char in text[brace:]:
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip(), True
            chars.append(char)
        else:
            chars.append(char)
    return "", False


def derive_answer(text: str, finish_reason: str) -> dict[str, Any]:
    """Strictly accept a balanced boxed answer only after ``</think>``."""
    close = text.find("</think>")
    suffix = text[close + len("</think>") :] if close >= 0 else ""
    prediction, box_closed = last_boxed(suffix)
    finish_valid = finish_reason not in INVALID_FINISH_REASONS
    return {
        "prediction": prediction,
        "close_think_found": close >= 0,
        "box_closed": box_closed,
        "finish_valid": finish_valid,
        "answer_valid": bool(close >= 0 and box_closed and prediction and finish_valid),
    }


def safe_grade(prediction: str, gold: str, timeout_seconds: float = 2.0) -> bool:
    if not prediction.strip() or not gold.strip():
        return False
    old_handler = signal.signal(signal.SIGALRM, grade_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        if canonical(prediction) == canonical(gold):
            return True
        parsed_prediction = math_parse(f"\\boxed{{{prediction}}}")
        parsed_gold = math_parse(f"\\boxed{{{gold}}}")
        return bool(
            parsed_prediction
            and parsed_gold
            and math_verify(parsed_prediction, parsed_gold)
        )
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_rows(aime_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with aime_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item.get("unique_id") or "").strip()
            problem = str(item.get("problem") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not qid or not problem or not answer:
                raise ValueError(
                    f"{aime_path}:{line_number} missing unique_id/problem/answer"
                )
            rows.append({**item, "eval_dataset": "aime2024"})
    qids = [str(row["unique_id"]) for row in rows]
    if len(rows) != 30 or len(set(qids)) != 30:
        raise ValueError(
            f"Expected 30 unique AIME2024 rows, got rows={len(rows)} unique={len(set(qids))}"
        )
    return rows


def existing_qids(path: Path, fingerprint: str) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("run_fingerprint") != fingerprint:
                raise ValueError(f"{path}:{line_number} fingerprint mismatch")
            if item.get("eval_dataset") != "aime2024":
                raise ValueError(f"{path}:{line_number} is not an AIME2024 row")
            qid = str(item.get("qid") or "")
            if not qid or qid in completed:
                raise ValueError(f"{path}:{line_number} empty or duplicate qid")
            completed.add(qid)
    return completed


def write_progress(
    path: Path,
    *,
    state: str,
    shard_index: int,
    shard_count: int,
    rows_in_shard: int,
    completed: int,
    fingerprint: str,
    samples_per_second: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "rows_in_shard": rows_in_shard,
        "completed": completed,
        "remaining": rows_in_shard - completed,
        "run_fingerprint": fingerprint,
        "updated_unix": time.time(),
    }
    if samples_per_second is not None:
        payload["samples_per_second_this_process"] = samples_per_second
    atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aime", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_model_len != 40960 or args.max_tokens != 32768:
        raise ValueError("This baseline is pinned to 40,960 context and 32,768 output cap")

    rows = load_rows(args.aime)
    selected = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    sampling_config = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
    }
    config = {
        "model": str(args.model),
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "aime_sha256": file_sha256(args.aime),
        "dataset": "aime2024",
        "dataset_rows": 30,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "enable_thinking": True,
        "explicit_think_prefix": True,
        **sampling_config,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    done = existing_qids(args.output, fingerprint)
    selected_qids = {str(row["unique_id"]) for row in selected}
    if not done <= selected_qids:
        raise ValueError("Output contains qids outside this shard")
    pending = [row for row in selected if str(row["unique_id"]) not in done]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=True)
    if not pending:
        if done != selected_qids:
            raise RuntimeError(f"Incomplete shard: {len(done)}/{len(selected_qids)}")
        write_progress(
            args.progress,
            state="complete",
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            rows_in_shard=len(selected),
            completed=len(done),
            fingerprint=fingerprint,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=65536,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    started = time.time()
    completed_now = 0
    with args.output.open("a", encoding="utf-8") as output_handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts: list[str] = []
            prompt_tokens: list[int] = []
            for row in batch:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["problem"])},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                ) + "<think>\n"
                token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
                if token_count + args.max_tokens > args.max_model_len:
                    raise ValueError(
                        f"Prompt for {row['unique_id']} leaves less than "
                        f"{args.max_tokens} output tokens: prompt_tokens={token_count}"
                    )
                prompts.append(prompt)
                prompt_tokens.append(token_count)

            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned the wrong output count")
            for row, prompt_count, request_output in zip(batch, prompt_tokens, outputs):
                candidate = request_output.outputs[0]
                text = str(candidate.text or "")
                finish_reason = str(candidate.finish_reason or "")
                fields = derive_answer(text, finish_reason)
                gold = str(row["answer"])
                record = {
                    "format_version": 1,
                    "run_fingerprint": fingerprint,
                    "run_config": config,
                    "sampling": sampling_config,
                    "eval_dataset": "aime2024",
                    "qid": str(row["unique_id"]),
                    "problem": str(row["problem"]),
                    "gold_answer": gold,
                    "prediction": fields["prediction"],
                    "is_correct": bool(
                        fields["answer_valid"] and safe_grade(fields["prediction"], gold)
                    ),
                    "close_think_found": fields["close_think_found"],
                    "box_closed": fields["box_closed"],
                    "finish_valid": fields["finish_valid"],
                    "answer_valid": fields["answer_valid"],
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_count,
                    "response_tokens": len(candidate.token_ids or []),
                    "response_text": text,
                }
                output_handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                output_handle.flush()
                completed_now += 1

            observed = len(done) + completed_now
            elapsed = max(time.time() - started, 1e-6)
            write_progress(
                args.progress,
                state="running",
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                rows_in_shard=len(selected),
                completed=observed,
                fingerprint=fingerprint,
                samples_per_second=completed_now / elapsed,
            )

    final_done = existing_qids(args.output, fingerprint)
    if final_done != selected_qids:
        raise RuntimeError(f"Incomplete shard: {len(final_done)}/{len(selected_qids)}")
    write_progress(
        args.progress,
        state="complete",
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        rows_in_shard=len(selected),
        completed=len(final_done),
        fingerprint=fingerprint,
    )


if __name__ == "__main__":
    main()
