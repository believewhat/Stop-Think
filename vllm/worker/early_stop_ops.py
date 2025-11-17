# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import Any, Dict, List, Tuple

# 这些函数会被 EngineCore 的 RPC 分发调用，传入 worker 上下文/引用。
# 约定：core 调用时会把 `runner`（ModelRunner实例）与 `seq_manager` 一并传入。

def _get_seq_by_request_id(seq_manager, request_id: str):
    # vLLM 0.9.x 的常见路径
    if hasattr(seq_manager, "get_seq_by_request_id"):
        return seq_manager.get_seq_by_request_id(request_id)
    if hasattr(seq_manager, "sequences_by_request_id"):
        return seq_manager.sequences_by_request_id.get(request_id)
    raise RuntimeError("Sequence manager API not found")

def kv_pin_checkpoint(runner, seq_manager, request_id: str) -> None:
    seq = _get_seq_by_request_id(seq_manager, request_id)
    if seq is None:
        raise RuntimeError(f"Sequence not found for request_id={request_id}")
    runner.kv_pin_checkpoint(request_id, seq.block_table)

def kv_release(runner, seq_manager, request_id: str) -> None:
    runner.kv_release(request_id)

def decode_continuous(runner, seq_manager, request_id: str, steps: int, sampling_params) -> Dict[str, Any]:
    seq = _get_seq_by_request_id(seq_manager, request_id)
    if seq is None:
        raise RuntimeError(f"Sequence not found for request_id={request_id}")
    text, finished = runner.decode_continuous(seq, int(steps), sampling_params)
    return {"text": text, "finished": bool(finished)}

def probe_from_checkpoint(runner, seq_manager, request_id: str,
                          suffix_ids: List[int], max_steps: int, topk: int) -> Dict[str, Any]:
    seq = _get_seq_by_request_id(seq_manager, request_id)
    if seq is None:
        raise RuntimeError(f"Sequence not found for request_id={request_id}")
    steps_logprobs, probe_text = runner.run_probe_branch(seq, list(map(int, suffix_ids)), int(max_steps), int(topk))
    # 标准化为 JSON 友好结构
    steps = [[(tok, float(lp)) for tok, lp in step] for step in (steps_logprobs or [])]
    return {"steps_logprobs": steps, "probe_text": probe_text}
