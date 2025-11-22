# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import os
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
import numpy as np
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

def _find_subsequence(seq, pat):
    """在 seq(list[int]) 里找 pat(list[int]) 的第一次出现；找不到返回 None。"""
    n, m = len(seq), len(pat)
    if m == 0 or n < m:
        return None
    for i in range(n - m + 1):
        ok = True
        for j in range(m):
            if seq[i + j] != pat[j]:
                ok = False
                break
        if ok:
            return i
    return None


def _tok_text(tokenizer, tid: int) -> str:
    # 与 vLLM 的 logprobs key 一致的 token 文本（保留空格/前缀）
    return tokenizer.decode([tid], skip_special_tokens=False)


def _lp_dict_and_min(lp_entry):
    """
    统一把一个位置的 logprobs 条目整理为：
      - lp_map: dict[str->float]  (token_text -> logprob)
      - min_lp: float  (该位置返回条目中的最小 logprob，用作“第20名”后备)
    兼容两种常见结构：
      1) dict[token_text] = logprob
      2) list[TopLogprob-like]，元素有 .text/.token 和 .logprob
    """
    lp_map = {}
    min_lp = None


    if isinstance(lp_entry, dict):
        for k, v in lp_entry.items():
            try:
                lp = float(v)
            except Exception:
                continue
            lp_map[str(k)] = lp
            min_lp = lp if (min_lp is None or lp < min_lp) else min_lp
        return lp_map, min_lp


    if isinstance(lp_entry, (list, tuple)):
        for it in lp_entry:
            text = getattr(it, "text", None) or getattr(it, "token", None)
            lp   = getattr(it, "logprob", None)
            if text is None or lp is None:
                continue
            try:
                lp = float(lp)
            except Exception:
                continue
            lp_map[str(text)] = lp
            min_lp = lp if (min_lp is None or lp < min_lp) else min_lp
        return lp_map, min_lp


    # 兜底：未知结构
    return {}, None


def hard_prob_from_sample(sample_output, tokenizer, topk_expected: int = 20):
    """
    返回 (prob, hard_lp, easy_lp, hard_pos)
      - prob 按你的定义：hard_lp / (hard_lp + easy_lp)
      - 如果没找到 Hard 的起始 token，直接返回 (0.5, None, None, None)
      - 如果 Hard/Easy 不在 top-20，就用该步“最小 logprob”替代
    """
    # 1) 找到 "Hard" 的 token 序列（取其首 token 所在步）
    hard_ids = tokenizer.encode("Hard", add_special_tokens=False)
    if not hard_ids:
        return 0.5, None, None, None


    start = _find_subsequence(sample_output.token_ids, hard_ids)
    if start is None:
        # 没出现“Hard”（也就无 <Hard>），按 0.5
        return 0.5, None, None, None


    t = start  # 在第 t 步生成了 Hard 的首 token
    if not hasattr(sample_output, "logprobs") or sample_output.logprobs is None:
        return 0.5, None, None, t
    if t >= len(sample_output.logprobs):
        return 0.5, None, None, t


    lp_entry = sample_output.logprobs[t]


    # 2) 把该步的 logprobs 整理成 map，并拿到“第 20 名”的后备（最小值）
    lp_map, min_lp = _lp_dict_and_min(lp_entry)
    if min_lp is None:
        return 0.5, None, None, t


    # 3) 目标 token 文本（只取 Hard / Easy 的首 token 文本作为对比单位）
    hard_tok_text = _tok_text(tokenizer, hard_ids[0])


    easy_ids = tokenizer.encode("Easy", add_special_tokens=False)
    if not easy_ids:
        # 没法对比，就按 0.5
        return 0.5, None, None, t
    easy_tok_text = _tok_text(tokenizer, easy_ids[0])


    # 4) 取 Hard / Easy 的 logprob；不在 top-20 用最小 logprob 兜底
    hard_lp = lp_map.get(hard_tok_text, min_lp)
    easy_lp = lp_map.get(easy_tok_text, min_lp)


    denom = hard_lp + easy_lp
    if denom == 0:
        prob = 0.5
    else:
        prob = hard_lp / denom


    # 安全裁剪到 [0,1]（按你的公式一般会在 0~1 之间，这里只是保险）
    if prob < 0.0:
        prob = 0.0
    elif prob > 1.0:
        prob = 1.0


    return float(prob), float(hard_lp), float(easy_lp), int(t)

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

import re
from typing import Optional, List
from itertools import accumulate


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_FINAL_ANY_RE   = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.IGNORECASE | re.DOTALL)
_FIRST_LETTER_RE = re.compile(r"([ABCD])", re.IGNORECASE)

import numpy as np
from itertools import accumulate


# ====== 把 vLLM logprobs -> 单步 A/B/C/D 的 log 分数 ======
_BPE_MARKERS = {"Ġ", "▁"}
_PUNCT = set(' \t\n\r<>"\'`()[]{}.,;:|/\\!?*-_+=~')


def _canon_token_text(s: str) -> str:
    if s is None: return ""
    t = str(s)
    for m in _BPE_MARKERS: t = t.replace(m, "")
    t = t.strip()
    i, j = 0, len(t)
    while i < j and t[i] in _PUNCT: i += 1
    while j > i and t[j-1] in _PUNCT: j -= 1
    return t[i:j]


def _text_to_letter(s: str):
    t = _canon_token_text(s)
    if not t: return None
    ch = t[0].upper()
    return ch if ch in {"A","B","C","D"} else None


def _lp_entry_to_text_lp_and_min(lp_entry, tokenizer):
    """
    统一成 {token_text: logprob} 和该步最小 logprob（作缺失兜底）。
    兼容 vLLM 的 dict[token_id]->TopLogprob / dict[text]->float / list[TopLogprob]
    """
    text_lp = {}
    min_lp = None
    if isinstance(lp_entry, dict):
        for k, v in lp_entry.items():
            if hasattr(v, "logprob"):
                lp = float(v.logprob)
                tok_text = getattr(v, "decoded_token", None)
                if tok_text is None:
                    try:
                        tok_id = int(k)
                        tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
                    except Exception:
                        tok_text = str(k)
            else:
                try: lp = float(v)
                except Exception: continue
                tok_text = str(k)
            text_lp[str(tok_text)] = lp
            min_lp = lp if (min_lp is None or lp < min_lp) else min_lp
        return text_lp, min_lp
    if isinstance(lp_entry, (list, tuple)):
        for it in lp_entry:
            tok_text = getattr(it, "decoded_token", None) or getattr(it, "text", None) or getattr(it, "token", None)
            lp = getattr(it, "logprob", None)
            if tok_text is None or lp is None: 
                continue
            try: lp = float(lp)
            except Exception: continue
            text_lp[str(tok_text)] = lp
            min_lp = lp if (min_lp is None or lp < min_lp) else min_lp
        return text_lp, min_lp
    return {}, None


def _letter_lps_at_step(lp_entry, tokenizer, fallback_bias=-2.0):
    """
    单步聚合 A/B/C/D 的 log 概率（log-sum-exp）。缺失用 min_lp + bias 兜底。
    返回 (lpA, lpB, lpC, lpD, present_letters_set)
    """
    text_lp, min_lp = _lp_entry_to_text_lp_and_min(lp_entry, tokenizer)
    if not text_lp or min_lp is None:
        base = -100.0
        return base, base, base, base, set()


    buckets = {"A": [], "B": [], "C": [], "D": []}
    present = set()
    for tok_text, lp in text_lp.items():
        L = _text_to_letter(tok_text)
        if L:
            buckets[L].append(float(lp))
            present.add(L)


    fb = float(min_lp) + float(fallback_bias)
    def _lse(xs):
        xs = [x for x in xs if np.isfinite(x)]
        if not xs: return fb
        a = max(xs)
        return float(a + np.log(np.sum(np.exp(np.array(xs) - a))))
    return _lse(buckets["A"]), _lse(buckets["B"]), _lse(buckets["C"]), _lse(buckets["D"]), present


def _inst_scores_from_eval_item(item, tokenizer):
    """
    在一个 eval 的 outputs[0].logprobs 里，挑“含 A/B/C/D 最多”的那一步（平手取更靠后），
    返回这个“答案步”的 (sA,sB,sC,sD)（log-space）。
    """
    if (item is None) or (not hasattr(item, "logprobs")) or (item.logprobs is None) or (len(item.logprobs)==0):
        return np.array([-100.0,-100.0,-100.0,-100.0], dtype=float)
    best_idx, best_score, best_cnt = 0, None, -1
    for t, lp_entry in enumerate(item.logprobs):
        lpA, lpB, lpC, lpD, present = _letter_lps_at_step(lp_entry, tokenizer)
        cnt = len(present)
        score = (lpA, lpB, lpC, lpD)
        if (cnt > best_cnt) or (cnt == best_cnt and t > best_idx):
            best_idx, best_score, best_cnt = t, score, cnt
    return np.array(best_score if best_score else [-100.0,-100.0,-100.0,-100.0], dtype=float)


def _first_letter_from_ids(ids, tokenizer):
    txt = tokenizer.decode(ids or [], skip_special_tokens=False)
    m = re.search(r'([ABCD])', txt, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


# ====== 在线特征器：把 1..k 的“前缀评测”转成与你离线一致的特征 ======
class EarlyStopFeaturizer:
    def __init__(self, W=5, K_recent=3):
        self.W = W
        self.K_recent = K_recent
        self.reset()


    def reset(self):
        self.cum = np.zeros(4, dtype=float)
        self.prev_top = None
        self.run_len = 0
        self.flips = 0
        self.cum_margin_seq = []
        self.hist_margin = []
        self.hist_cum_vecs = []
        self.hist_probs = []


    @staticmethod
    def _path_curvature(cum_hist):
        eps = 1e-9
        if len(cum_hist) < 3: return 0.0, 0.0, 0.0
        x2, x1, x0 = np.asarray(cum_hist[-1]), np.asarray(cum_hist[-2]), np.asarray(cum_hist[-3])
        v = x2 - x1
        a = x2 - 2.0*x1 + x0
        v2 = float(np.dot(v, v)) + eps
        v_norm = float(np.sqrt(max(v2 - eps, 0.0)))
        a_norm = float(np.linalg.norm(a))
        a_para = (float(np.dot(a, v)) / v2) * v
        a_perp = a - a_para
        kappa = float(np.linalg.norm(a_perp) / (v2))
        return v_norm, a_norm, kappa


    @staticmethod
    def _quad_second_derivative(y):
        y = np.asarray(y, dtype=float)
        if y.size < 3: return 0.0
        t = np.arange(y.size, dtype=float)
        a = float(np.polyfit(t, y, deg=2)[0])
        return 2.0 * a


    @staticmethod
    def _fisher_stats(Ps):
        eps = 1e-12
        if len(Ps)==0: return 0.0,0.0,0.0,0.0
        Ps = np.asarray(Ps, dtype=float)
        Ps = np.clip(Ps, eps, 1.0)
        Ps = Ps / Ps.sum(axis=1, keepdims=True)
        F_sum = np.zeros((4,4), dtype=float); ent_sum=0.0
        for p in Ps:
            D = np.diag(p); F = D - np.outer(p,p)
            F_sum += F
            ent_sum += -float(np.sum(p*np.log(p+eps)))
        F_bar = F_sum / float(Ps.shape[0])
        fi_trace = float(np.trace(F_bar))
        evals = np.linalg.eigvalsh((F_bar + F_bar.T)*0.5)
        fi_lmax = float(np.max(evals))
        F_no = F_bar.copy(); np.fill_diagonal(F_no, 0.0)
        fi_offdiag_fro = float(np.linalg.norm(F_no, ord="fro"))
        fi_entropy = float(ent_sum/float(Ps.shape[0]))
        return fi_trace, fi_lmax, fi_offdiag_fro, fi_entropy


    def step(self, inst_scores_log, k, m):
        """
        inst_scores_log: 当前前缀评测的 (sA..sD)（log-space）
        k/m: 进度（1..m）
        返回：一个与你训练时 feats 顺序一致的特征向量
        """
        s = np.asarray(inst_scores_log, dtype=float)
        # 概率
        s_max = float(np.max(s)); exps = np.exp(s - s_max); p = exps / (np.sum(exps) + 1e-12)


        # 累计 log 概率
        self.cum += np.log(np.clip(p, 1e-12, 1.0))
        order = np.argsort(self.cum)[::-1]
        top, sec = int(order[0]), int(order[1])
        cum_margin = float(self.cum[top] - self.cum[sec])


        # run_len / flips
        if self.prev_top is None or self.prev_top != top:
            self.flips += (0 if self.prev_top is None else 1)
            self.run_len = 1
            self.prev_top = top
        else:
            self.run_len += 1


        # 序列派生
        self.cum_margin_seq.append(cum_margin)
        if len(self.cum_margin_seq) >= 2:
            delta_recent = float(self.cum_margin_seq[-1] - self.cum_margin_seq[max(0, len(self.cum_margin_seq)-self.K_recent)])
            slope_recent = float((self.cum_margin_seq[-1] - self.cum_margin_seq[0]) / max(1, len(self.cum_margin_seq)-1))
        else:
            delta_recent = 0.0; slope_recent = 0.0


        step_ratio = float(k / max(1, m))


        # 窗口缓存
        self.hist_margin.append(cum_margin)
        self.hist_cum_vecs.append(self.cum.copy())
        self.hist_probs.append(p.copy())
        if len(self.hist_margin) > 5:
            self.hist_margin.pop(0); self.hist_cum_vecs.pop(0); self.hist_probs.pop(0)


        # 曲率/Hessian 代理
        curv_margin2 = self._quad_second_derivative(self.hist_margin)
        curv_cum = [self._quad_second_derivative([v[i] for v in self.hist_cum_vecs]) for i in range(4)]
        vel_norm, acc_norm, kappa_curv = self._path_curvature(self.hist_cum_vecs)
        fi_trace, fi_lmax, fi_offdiag_fro, fi_entropy = self._fisher_stats(self.hist_probs)


        # 与离线训练一致的 30 维特征
        feats = np.array([
            self.cum[0], self.cum[1], self.cum[2], self.cum[3],
            cum_margin, self.run_len, self.flips,
            delta_recent, slope_recent, step_ratio,
            s[0], s[1], s[2], s[3],
            p[0], p[1], p[2], p[3],
            curv_margin2, curv_cum[0], curv_cum[1], curv_cum[2], curv_cum[3],
            vel_norm, acc_norm, kappa_curv,
            fi_trace, fi_lmax, fi_offdiag_fro, fi_entropy
        ], dtype=float)
        return feats


# ====== 主函数：把一个 eval 批（所有前缀） -> (X, y, meta) ======
def build_earlystop_training_data(eval_outs, group_sizes, final_letters, tokenizer):
    """
    参数：
      - eval_outs: vLLM 返回的列表；与 eval_inputs 一一对应
      - group_sizes: 每个 candidate 有多少个前缀（k=1..m）；len==num_candidates
      - final_letters: 每个 candidate 的“无停时最终答案字母”（A/B/C/D）
      - tokenizer: vLLM 的 tokenizer


    返回：
      - X: (N, 30) 特征矩阵（与你离线训练一致）
      - y: (N,) 0/1 标签；1 表示“该前缀预测的字母 == 最终不 stop 的字母”
      - meta: [(c, k, m, gold_final, pred_at_k)] 方便你后处理/调试
    """
    X_rows, y_rows, meta = [], [], []
    offsets = [0] + list(accumulate(group_sizes))


    for c, m in enumerate(group_sizes):
        if m <= 0: 
            continue
        featzer = EarlyStopFeaturizer()
        start, end = offsets[c], offsets[c+1]
        gold_final = (final_letters[c] or "").upper()


        for local_k, j in enumerate(range(start, end), start=1):
            out_item = eval_outs[j].outputs[0] if eval_outs[j].outputs else None


            # 单步 (sA..sD)
            if out_item is None:
                inst_s = np.array([-100.0,-100.0,-100.0,-100.0], dtype=float)
                pred_k = None
            else:
                inst_s = _inst_scores_from_eval_item(out_item, tokenizer)
                pred_k = _first_letter_from_ids(
                    out_item.token_ids if hasattr(out_item, "token_ids") else [], tokenizer
                )


            # 累计到 k/m → 特征
            xk = featzer.step(inst_s, local_k, m)
            X_rows.append(xk)
            y_rows.append(1 if (pred_k and gold_final and pred_k.upper()==gold_final) else 0)
            meta.append((c, local_k, m, gold_final, pred_k))


    X = np.vstack(X_rows) if len(X_rows)>0 else np.zeros((0, 30), dtype=float)
    y = np.asarray(y_rows, dtype=int)
    return X, y, meta




def _decode(ids: List[int], tokenizer) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def _extract_think_and_final(text: str):
    """返回 (think_str, final_text, final_letter 或 None)"""
    m_th = _THINK_BLOCK_RE.search(text or "")
    think_str = (m_th.group(1).strip() if m_th else "")
    m_fa = _FINAL_ANY_RE.search(text or "")
    final_text = (m_fa.group(1).strip() if m_fa else "")
    mL = _FIRST_LETTER_RE.search(final_text)
    final_letter = mL.group(1).upper() if mL else None
    return think_str, final_text, final_letter


def _split_by_stop(think_str: str) -> List[str]:
    if not think_str.strip():
        return []
    parts = re.split(r"\s*<stop>\s*", think_str.strip())
    return [p.strip() for p in parts if p.strip()]


def _join_with_stop(segs: List[str]) -> str:
    return " <stop> ".join(segs)


def _build_eval_suffix(prefix_text: str) -> str:
    # 用于前缀评测：拼在“原始 prompt token_ids”后
    return f"<think> {prefix_text} </think><final_answer>"


def _first_letter_from_ids(ids: List[int], tokenizer) -> Optional[str]:
    txt = _decode(ids, tokenizer)
    m = _FIRST_LETTER_RE.search(txt)
    return m.group(1).upper() if m else None

class StopvLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 9192)
        self.tokenizer = tokenizer
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')
        
        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=False,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )

        


        
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        kwargs['max_tokens'] = 20
        kwargs['logprobs'] = 20
        self.sampling_params2 = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # 1) vLLM cache & 取基础张量
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()


        idx = prompts.batch['input_ids']          # (bs, prompt_len)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)


        # 2) 准备 base prompt ids（左侧去 padding）
        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)
        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')


        base_prompt_ids_list = [rp.tolist() if isinstance(rp, np.ndarray) else list(rp)
                            for rp in non_tensor_batch['raw_prompt_ids']]


        # 3) 首次生成：得到原始 candidates
        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, mmd in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                    non_tensor_batch.pop('multi_modal_data')):
                raw_prompt_ids = raw_prompt_ids.tolist() if isinstance(raw_prompt_ids, np.ndarray) else list(raw_prompt_ids)
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': mmd})
        else:
            vllm_inputs = [{'prompt_token_ids': (rp.tolist() if isinstance(rp, np.ndarray) else list(rp))}
                       for rp in non_tensor_batch.pop('raw_prompt_ids')]


        for it in vllm_inputs:
            if isinstance(it['prompt_token_ids'], np.ndarray):
                it['prompt_token_ids'] = it['prompt_token_ids'].tolist()
            elif not isinstance(it['prompt_token_ids'], list):
                raise TypeError(f"prompt_token_ids must be list, got {type(it['prompt_token_ids'])}")


        do_sample   = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            gen_overrides = {'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1}
        elif is_validate:
            gen_overrides = {'top_k': self.config.val_kwargs.top_k,
                         'top_p': self.config.val_kwargs.top_p,
                         'temperature': self.config.val_kwargs.temperature,
                         'n': 1}
        else:
            gen_overrides = {}


        with self.update_sampling_params(**gen_overrides):
            outs = self.inference_engine.generate(prompts=vllm_inputs,
                                              sampling_params=self.sampling_params,
                                              use_tqdm=False)


            # 把所有 candidate 摊平到单个批（仅数据结构处理，不再调 vLLM）
            cand_base_idx  = [i for i, o in enumerate(outs) for _ in o.outputs]
            cand_samples   = [o.outputs[j] for o in outs for j in range(len(o.outputs))]
            cand_token_ids = [s.token_ids for s in cand_samples]


            # hard_prob（保持你原逻辑）
            #prob_hards = [hard_prob_from_sample(s, self.tokenizer, 20)[0] for s in cand_samples]


            # 先把原 response 备好（后面若命中再替换）
            response_token_lists = [list(t) for t in cand_token_ids]


            # 批量解析 think & final
            cand_texts                 = self.tokenizer.batch_decode(cand_token_ids, skip_special_tokens=False)
            parsed                     = [_extract_think_and_final(t) for t in cand_texts]
            think_list, final_texts, final_letters = zip(*parsed) if parsed else ([], [], [])


            # 4) 预处理所有“前缀输入”——一次性打包给 vLLM
            segs_list       = [_split_by_stop(th) if th and fl else [] for th, fl in zip(think_list, final_letters)]
            prefix_groups   = [[_join_with_stop(segs[:k]) for k in range(1, len(segs)+1)] for segs in segs_list]
            group_sizes     = [len(g) for g in prefix_groups]
            total_prefixes  = sum(group_sizes)


            early_stop_scores = [0.0] * len(prefix_groups)   # 结果分数：1 - (k/m)
            hit_k_list       = [None] * len(prefix_groups)   # 命中的最短前缀 k（1..m）；None=无命中

            X = np.array([])
            y = np.array([])
            if total_prefixes > 0:
                # 把所有前缀后缀 token_ids 与各自 base prompt ids 合并，形成一个“大批”
                suffix_ids   = [self.tokenizer.encode(_build_eval_suffix(ptxt), add_special_tokens=False)
                        for group in prefix_groups for ptxt in group]

                # prefix_groups: 每个 candidate 的前缀文本列表（长度 = num_cands）
                group_sizes = [len(g) for g in prefix_groups]  # 每个 candidate 有多少个前缀


                # owner_idx_flat[p] = 这个“扁平前缀 p”属于哪个 candidate（0..num_cands-1）
                # 用 numpy 写法更简洁，也可以用纯 Python 展开

                owner_idx_flat = np.repeat(np.arange(len(prefix_groups)), group_sizes).tolist()


                # 为每个“扁平前缀”找到对应的 base prompt 索引
                prefix_base_idx = [cand_base_idx[c] for c in owner_idx_flat]


                # 一次性组装 vLLM 的批输入
                eval_inputs = [
                    {'prompt_token_ids': base_prompt_ids_list[b] + suf}
                    for b, suf in zip(prefix_base_idx, suffix_ids)
                ]


                # 5) 单次 vLLM 批量评测所有前缀
                eval_outs = self.inference_engine.generate(prompts=eval_inputs,
                                            sampling_params=self.sampling_params2,
                                            use_tqdm=False)

                
                eval_pred_letters = [_first_letter_from_ids(eo.outputs[0].token_ids if eo.outputs else [], self.tokenizer)
                             for eo in eval_outs]
                # 你已有的变量：
                #   group_sizes: 每个 candidate 的前缀数量 m
                #   final_letters: 每个 candidate 的最终（不提前停）答案字母
                #   self.tokenizer: vLLM tokenizer
                X, y, meta = build_earlystop_training_data(eval_outs, group_sizes, final_letters, self.tokenizer)

                # 6) 后处理：对每个 candidate 在自己的切片内找“最短命中前缀”
                #    构造每个组在扁平 eval 里的起止索引，避免写多重 for
                offsets = [0] + list(accumulate(group_sizes))  # 长度 = num_cands + 1
                for c in range(len(prefix_groups)):
                    m = group_sizes[c]
                    if m == 0:
                        early_stop_scores[c] = 0.0
                        hit_k_list[c] = None
                        continue
                    start, end = offsets[c], offsets[c+1]
                    origL = (final_letters[c] or "").upper()
                    # 在 [start, end) 里寻找首个等于 origL 的预测字母
                    k_list = []
                    num_k = 0
                    k_index = []
                    for j in range(start, end):
                        pred = eval_pred_letters[j]
                        if pred and origL and pred.upper() == origL:
                            k = (j - start) + 1
                            k_index.append(k)
                            k_list.append(float(1.0 - (k / m)))   # 转为 1..m
                        num_k += 1
                    # 分数：1 - (k/m)；若无命中，按“必须到末尾”处理 → k=m → score=0
                    if len(k_list) == 0:
                        hit_k_list[c] = None
                        k_list.append(0.0)
                    else:
                        hit_k_list[c] = k_index[0]
                    early_stop_scores[c] = k_list[0]


                # 7) 根据最短命中（若不是最后一段）重写 response
                new_texts = []
                for c, k in enumerate(hit_k_list):
                    segs = segs_list[c]
                    if not segs or k is None or k >= len(segs):
                        # 无命中或命中在最后一段：保持原样
                        continue
                    prefix_txt = _join_with_stop(segs[:k])
                    new_text = f"<think> {prefix_txt} <stop> </think> <final_answer> {final_texts[c]} </final_answer><|im_end|>"
                    new_texts.append(new_text)
                    response_token_lists[c] = self.tokenizer.encode(new_text, add_special_tokens=True)
                
            # 8) 后续与原逻辑一致：pad、拼接位置与 mask，并把分数放进 batch
            response = pad_2d_list_to_length(response_token_lists, self.pad_token_id,
                                     max_length=self.config.response_length).to(idx.device)
            

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                X = np.array([{'value': X}])
                y = np.array([{'value': y}])
                non_tensor_batch['feature_x'] = _repeat_interleave(X, idx.shape[0])
                non_tensor_batch['feature_y'] = _repeat_interleave(y, idx.shape[0])
                non_tensor_batch['solution'] = _repeat_interleave(non_tensor_batch['solution'], self.sampling_params.n)
                non_tensor_batch['reward_model'] = _repeat_interleave(non_tensor_batch['reward_model'], self.sampling_params.n)
                non_tensor_batch['correct_think'] = _repeat_interleave(non_tensor_batch['correct_think'], self.sampling_params.n)
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                        self.sampling_params.n)


            seq = torch.cat([idx, response], dim=-1)


        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)


        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                eos_token=eos_token_id,
                                                dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                #'hard_prob': prob_hards,              # 你原有的 hard 指标
                'early_stop_score': early_stop_scores,  # 新增：按 <stop> 段比例得到的分数
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size)

        """
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        """
        #try:    self.inference_engine.sleep(level=2)
        #except: self.inference_engine.sleep(level=1)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)




