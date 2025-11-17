# vllm/engine/earlystop_mixin.py
from __future__ import annotations
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import numpy as np

from vllm.v1.sequence import Sequence
from probe_features_vllm import (
    compute_probe_slot_from_vllm_steps, SeqFeatureTracker, cum_top_onehot
)

_FA_BLOCK   = re.compile(r"<\s*final_answer\s*>(.*?)<\s*/\s*final_answer\s*>", flags=re.I|re.S)
_PAT_STRONG = re.compile(r"(?i)(?:^|\b(?:answer|ans)\s*[:\-]?\s*)([ABCD])(?:\s*[:\.\)]|\s|$)")
_PAT_SOFT   = re.compile(r"(?i)(?<![A-Za-z])([ABCD])(?![A-Za-z])")

class EarlyStopEngineMixin:
    """要求宿主类(AsyncLLMEngine/LLMEngine)已有：
       - self._add_request_from_token_ids(...)
       - self._decode_steps_for(seq, steps, sampling_params, need_logprobs_k)
       - self._kv_cache_manager
       - self._greedy_logprobs_params(topk)  # 返回 T=0, top_p=1 的采样参数
    """

    @contextmanager
    def forked_view(self, kv_mgr, parent_bt):
        child_bt = parent_bt.fork_view(kv_mgr)  # 引用+写时复制
        try:
            yield child_bt
        finally:
            child_bt.release_all(kv_mgr)         # 结束立刻 decref

    def decode_continuous(self, seq: Sequence, steps: int, sampling_params) -> str:
        """主 THINK 连续推进 N 步；不取 topk。返回新增文本以检测硬停。"""
        outs = self._decode_steps_for(seq, int(steps), sampling_params, need_logprobs_k=None)
        # 兼容不同回传结构，获取新增文本
        chunk = ""
        if outs and getattr(outs[0], "decoded_tokens", None) is not None:
            chunk = "".join([str(t) for t in (outs[0].decoded_tokens or [])])
        elif outs and getattr(outs[0], "text", None) is not None:
            chunk = str(outs[0].text or "")
        return chunk

    def run_probe_branch(self,
                         parent_seq: Sequence,
                         probe_suffix_ids: List[int],
                         max_steps: int,
                         topk: int,
                         prefer_letter: Optional[str] = None) -> Dict[str, Any]:
        kv_mgr = self._kv_cache_manager
        with self.forked_view(kv_mgr, parent_seq.block_table) as child_bt:
            # 1) 从现有 KV 造子序列（不重复 prefill）
            child = parent_seq.spawn_child_from_block_table(
                new_seq_id=f"{parent_seq.seq_id}::probe",
                block_table=child_bt,
            )
            # 2) 追加 </think><final_answer> 作为极短 suffix prompt
            if probe_suffix_ids:
                if getattr(child_bt, "_block_ids", None):
                    kv_mgr.ensure_writable_block(child_bt,
                                                 logical_idx=len(child_bt._block_ids) - 1)
                child.tokens.extend(probe_suffix_ids)

            # 3) 以 T=0, top_p=1 连续 <=max_steps，并采 topk logprobs
            outs = self._decode_steps_for(
                child, int(max_steps),
                self._greedy_logprobs_params(int(topk)),
                need_logprobs_k=int(topk),
            )
            steps_logprobs = getattr(outs[0], "logprobs", None) if outs else []
            # 解析为“槽位 + A/B/C/D LSE + 可否早停”
            slot = compute_probe_slot_from_vllm_steps(
                steps_logprobs=steps_logprobs,
                topk=int(topk),
                prefer_letter=prefer_letter,
                probe_text="".join([str(t) for t in getattr(outs[0], "decoded_tokens", [])]) if outs else "",
            )
            return slot

    def generate_with_checks(self,
                             *,
                             prompt_token_ids: List[int],
                             think_sampling_params: Any,
                             probe_suffix_token_ids: List[int],
                             classifier_pack: Dict[str, Any],
                             threshold: float,
                             check_interval: int = 50,
                             probe_max_steps: int = 10,
                             topk: int = 20,
                             request_id: Optional[str] = None) -> Dict[str, Any]:
        """主序列一口气生成；每到 interval 原地 fork 做 PROBE，阈值过则直接早停。"""
        clf = classifier_pack["model"]
        feats_order = classifier_pack["feats"]

        seq = self._add_request_from_token_ids(
            prompt_token_ids, think_sampling_params, request_id=request_id)
        tracker = SeqFeatureTracker(W=5, K_recent=3)

        # Round-0：空 think 的一次 probe（训练/评估对齐）
        slot0 = self.run_probe_branch(seq, probe_suffix_token_ids, int(probe_max_steps), int(topk))
        feats0 = tracker.update_with_step_vals(slot0["vals"])
        feats0.update(cum_top_onehot(feats0.get("cum_top")))

        rounds = 0
        total_gen = 0
        think_accum = ""

        while True:  # 注意：这是 Engine 内部隐含推进；外层调用只有一次 generate_with_checks
            rounds += 1
            chunk = self.decode_continuous(seq, int(check_interval), think_sampling_params)
            total_gen += int(check_interval)
            think_accum += (chunk or "")

            # 硬停：THINK 中出现完整 <final_answer>...</final_answer>
            mfin = _FA_BLOCK.search(think_accum)
            if mfin:
                inside = mfin.group(1)
                m = _PAT_STRONG.search(inside or "") or _PAT_SOFT.search(inside or "")
                letter = (m.group(1).upper() if m else None)
                return {"final_text": f"<final_answer>{inside}</final_answer>",
                        "final_cause": "think_final",
                        "final_letter": letter, "rounds": rounds, "tokens_generated": total_gen}

            # 主序列自然结束（没给出 final）
            if getattr(seq, "finished", False):
                return {"final_text": "",
                        "final_cause": "finished",
                        "final_letter": None,
                        "rounds": rounds, "tokens_generated": total_gen}

            # 原地 PROBE：用累计 top 作为 prefer hint
            prefer = tracker.current_cum_top_letter()
            slot = self.run_probe_branch(seq, probe_suffix_token_ids,
                                         int(probe_max_steps), int(topk),
                                         prefer_letter=prefer)

            feats = tracker.update_with_step_vals(slot["vals"])
            feats.update(cum_top_onehot(feats.get("cum_top")))
            feats["step"] = int(rounds * check_interval)

            X = np.asarray([[feats.get(c, 0.0) for c in feats_order]], dtype=float)
            prob = (float(clf.predict_proba(X)[:, 1][0])
                    if hasattr(clf, "predict_proba") else float(clf.predict(X)[0]))

            if slot["early_stop_elig"] and prob >= float(threshold):
                letter = slot.get("probe_letter")
                return {"final_text": (f"<final_answer>{letter}</final_answer>" if letter else ""),
                        "final_cause": "early_stop",
                        "final_letter": letter, "rounds": rounds, "tokens_generated": total_gen}
