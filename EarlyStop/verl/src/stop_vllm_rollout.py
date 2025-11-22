# Copyright 2024 Bytedance Ltd. ...
# (license header unchanged)


"""
Open-domain (math) early-stop rollout with vLLM.


改动要点：
- \boxed{...}/\fbox{...}：用“花括号配平”完整抽取。
- THINK：支持缺失 </think>，匹配到文本末尾。
- 早停：命中“最终生成答案(且该答案非空)”或“正确答案(GT)”任一即可提前中断。
- 若最终答案为空字符串，则**只能**与 GT 匹配才能早停。
- 正确答案来源：strictly 从 `non_tensor_batch['reward_model'][i]['ground_truth']`。
- 重写响应：若由 GT 触发早停，`\boxed{...}` 用 GT 文本；否则用最终生成答案文本。
"""


import os
import re
import numpy as np
from typing import List, Any, Union, Tuple, Dict
from contextlib import contextmanager
from itertools import accumulate


from omegaconf import DictConfig


import torch
import torch.distributed
from torch import nn
from tensordict import TensorDict


from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout


from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version


from math_verify import parse, verify


# ========================
# 基础小工具
# ========================
def _pre_process_inputs(pad_token_id: int, ids: torch.Tensor) -> List[int]:
    ids = ids.tolist()
    i = 0
    n = len(ids)
    while i < n and ids[i] == pad_token_id:
        i += 1
    return ids[i:]


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _safe_strip(s):
    return s.strip() if isinstance(s, str) else ""


# ========================
# \boxed 提取（花括号配平）
# ========================
BOXED_START_RE = re.compile(r"\\(?:boxed|fbox)\s*\{", flags=re.IGNORECASE)


def _find_last_boxed_balanced(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    m = None
    for m in BOXED_START_RE.finditer(s):
        pass
    if not m:
        return ""
    start = m.end()
    depth = 1
    i = start
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i].strip()
        i += 1
    return ""


def _last_boxed(s: str) -> str:
    return _find_last_boxed_balanced(s)


# ========================
# THINK 提取：容忍缺失 </think>
# ========================
THINK_BLOCK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", flags=re.DOTALL | re.IGNORECASE)
STOP_SPLIT_RE  = re.compile(r"\s*<stop>\s*", flags=re.IGNORECASE)


def _extract_think_and_final(full_text: str) -> Tuple[str, str, Union[str, Any]]:
    txt = full_text if isinstance(full_text, str) else ""
    m = THINK_BLOCK_RE.search(txt)
    think_text = m.group(1).strip() if m else ""


    final_raw = ""
    final_key = ""
    try:
        final_key = parse(txt)
        if isinstance(final_key, str):
            final_raw = final_key
        else:
            br = _last_boxed(txt)
            final_raw = br if br else ""
    except Exception:
        pass


    if not final_key:
        br = _last_boxed(txt)
        if final_raw == "" and br:
            final_raw = br
        try:
            if br:
                final_key = parse(br)
        except Exception:
            final_key = ""


    return think_text, final_raw, final_key


def _split_by_stop(think_text: str):
    if not isinstance(think_text, str) or not think_text.strip():
        return []
    parts = [p.strip() for p in STOP_SPLIT_RE.split(think_text)]
    return [p for p in parts if p]


def _join_with_stop(segs: List[str]) -> str:
    if not segs:
        return ""
    return (" <stop> ").join([s.strip() for s in segs])


def _build_eval_suffix(prefix_think_text: str) -> str:
    prefix = (prefix_think_text or "").strip()
    return f"<think> {prefix} <stop> </think>\n$$\n\\boxed{{"


def _answer_from_ids(token_ids, tokenizer) -> Tuple[Union[str, Any], str]:
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return "", ""
    text = (text or "").strip()


    try:
        pred_key = parse(text)
        if pred_key:
            pred_raw = _last_boxed(text) or text.splitlines()[0]
            return pred_key, pred_raw
    except Exception:
        pass


    boxed = _last_boxed(text)
    if boxed:
        try:
            pred_key = parse(boxed)
            return pred_key, boxed
        except Exception:
            return "", boxed


    head = re.split(r"[\n\r;。]", text, maxsplit=1)[0].strip() if text else ""
    try:
        pred_key = parse(head)
        return pred_key, head
    except Exception:
        return "", head


def _normalize_segs(think_text: str) -> List[str]:
    if not isinstance(think_text, str) or not think_text.strip():
        return []


    if STOP_SPLIT_RE.search(think_text):
        segs = [p.strip() for p in STOP_SPLIT_RE.split(think_text)]
        return [p for p in segs if p]


    s = think_text.strip()
    n = len(s)
    if n <= 4:
        return [s]


    i1 = max(1, n // 3)
    i2 = max(i1 + 1, (2 * n) // 3)
    i3 = max(i2 + 1, n - 1)


    segs = [s[:i1].strip(), s[i1:i2].strip(), s[i2:i3].strip(), s[i3:].strip()]
    return [p for p in segs if p]


def _window_prefix_by_stops(segs: List[str], k: int, max_stops: int = 5) -> str:
    if not segs or k is None or k <= 0:
        return ""
    k = min(k, len(segs))
    max_segs = max_stops + 1
    start = max(0, k - max_segs)
    if start == 0:
        return _join_with_stop(segs[start:k])
    else:
        return ' '.join(segs[:start]) + _join_with_stop(segs[start:k])


def build_earlystop_training_data(eval_outs, group_sizes, final_keys, tokenizer):
    return np.array([]), np.array([]), {}


# =============================================================================
# vLLM Rollout（open-domain / math）
# =============================================================================
class StopvLLMRollout(BaseRollout):


    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"


        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be <= world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 9192)
        self.tokenizer = tokenizer
        if kwargs.get('train_tp', None) is not None:
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)


        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be >= total sequence length"


        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill but max_num_batched_tokens < max_model_len. '
                             'Increase max_num_batched_tokens or disable chunked prefill.')


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


        self.inference_engine.sleep(level=1)


        kwargs_sp = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        if vllm_version != '0.3.1':
            kwargs_sp['detokenize'] = False
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs_sp[k] = config.get(k)


        self.sampling_params = SamplingParams(**kwargs_sp)


        kwargs_eval = dict(kwargs_sp)
        kwargs_eval['max_tokens'] = 20
        kwargs_eval['logprobs'] = 20
        self.sampling_params2 = SamplingParams(**kwargs_eval)


        self.pad_token_id = tokenizer.pad_token_id


    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()


        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)


        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)
        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager not working properly.')


        base_prompt_ids_list = [rp.tolist() if isinstance(rp, np.ndarray) else list(rp)
                                for rp in non_tensor_batch['raw_prompt_ids']]


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


            cand_base_idx  = [i for i, o in enumerate(outs) for _ in o.outputs]
            cand_samples   = [o.outputs[j] for o in outs for j in range(len(o.outputs))]
            cand_token_ids = [s.token_ids for s in cand_samples]


            response_token_lists = [list(t) for t in cand_token_ids]


            # 解析 think + 最终答案
            cand_texts = self.tokenizer.batch_decode(cand_token_ids, skip_special_tokens=False)
            parsed = [_extract_think_and_final(t) for t in cand_texts]
            think_list, final_texts, final_keys = zip(*parsed) if parsed else ([], [], [])


            # ========= 读取正确答案(GT)自 non_tensor_batch['reward_model'] =========
            def _extract_gold_from_reward_model(nontb, need_len):
                lst = nontb.get('reward_model', None)
                if lst is None:
                    return [""] * need_len
                if isinstance(lst, np.ndarray):
                    try:
                        lst = lst.tolist()
                    except Exception:
                        pass
                if isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], (list, np.ndarray)):
                    try:
                        lst = lst[0].tolist() if isinstance(lst[0], np.ndarray) else list(lst[0])
                    except Exception:
                        lst = list(lst[0])


                gold_texts = []
                for i in range(need_len):
                    gt = ""
                    if isinstance(lst, list) and i < len(lst) and isinstance(lst[i], dict):
                        gt = lst[i].get("ground_truth", "") or ""
                    gold_texts.append(gt if isinstance(gt, str) else "")
                return gold_texts


            gold_text_list = _extract_gold_from_reward_model(non_tensor_batch, len(cand_texts))
            gold_keys = []
            for gt in gold_text_list:
                if isinstance(gt, str) and gt.strip():
                    try:
                        gold_keys.append(parse(gt))
                    except Exception:
                        gold_keys.append("")
                else:
                    gold_keys.append("")


            # 4) 构造所有前缀
            segs_list = [_normalize_segs(th) if th else [] for th in think_list]
            prefix_groups = [[_join_with_stop(segs[:k]) for k in range(1, len(segs)+1)] for segs in segs_list]
            group_sizes   = [len(g) for g in prefix_groups]
            total_prefixes = sum(group_sizes)


            early_stop_scores = [0.0] * len(prefix_groups)
            hit_k_list        = [None] * len(prefix_groups)
            hit_source_list   = [None] * len(prefix_groups)   # 'final' 或 'gold'


            X = np.array([]); y = np.array([])
            if total_prefixes > 0:
                suffix_ids = [self.tokenizer.encode(_build_eval_suffix(ptxt), add_special_tokens=False)
                              for group in prefix_groups for ptxt in group]


                owner_idx_flat   = np.repeat(np.arange(len(prefix_groups)), group_sizes).tolist()
                prefix_base_idx  = [cand_base_idx[c] for c in owner_idx_flat]


                eval_inputs = [
                    {'prompt_token_ids': base_prompt_ids_list[b] + suf}
                    for b, suf in zip(prefix_base_idx, suffix_ids)
                ]


                # 5) 评测所有前缀
                eval_outs = self.inference_engine.generate(prompts=eval_inputs,
                                                           sampling_params=self.sampling_params2,
                                                           use_tqdm=False)


                eval_pred_keys = []
                for eo in eval_outs:
                    if eo.outputs:
                        tk = eo.outputs[0].token_ids
                    else:
                        tk = []
                    pk, _ = _answer_from_ids(tk, self.tokenizer)
                    eval_pred_keys.append(pk)


                # 6) 在各自切片内找“最短命中”
                offsets = [0] + list(accumulate(group_sizes))
                for c in range(len(prefix_groups)):
                    m = group_sizes[c]
                    if m == 0:
                        early_stop_scores[c] = 0.0
                        hit_k_list[c] = None
                        hit_source_list[c] = None
                        continue
                    start, end = offsets[c], offsets[c+1]


                    # —— 关键：final 仅在“文本非空”时才允许被用于早停判定 —— #
                    final_text_nonempty = bool((final_texts[c] or "").strip())
                    fkey = final_keys[c] if final_text_nonempty else ""  # final 空则禁用


                    gkey = gold_keys[c] if gold_keys else ""


                    k_best = None
                    source = None
                    for j in range(start, end):
                        pkey = eval_pred_keys[j]
                        if not pkey:
                            continue
                        ok_final = False
                        ok_gold  = False
                        if fkey:
                            try:
                                ok_final = verify(pkey, fkey)
                            except Exception:
                                ok_final = False
                        if (not ok_final) and gkey:
                            try:
                                ok_gold = verify(pkey, gkey)
                            except Exception:
                                ok_gold = False
                        if ok_final or ok_gold:
                            k_best = (j - start) + 1
                            source = 'final' if ok_final else 'gold'
                            break


                    if k_best is None:
                        early_stop_scores[c] = 0.0
                        hit_k_list[c] = None
                        hit_source_list[c] = None
                    else:
                        hit_k_list[c] = k_best
                        hit_source_list[c] = source
                        early_stop_scores[c] = float(1.0 - (k_best / m))


                # 7) 命中则重写 response（截断到最短前缀）
                for c, k in enumerate(hit_k_list):
                    segs = segs_list[c]
                    if not segs or k is None or k >= len(segs):
                        continue
                    prefix_txt = _window_prefix_by_stops(segs, k, max_stops=5)


                    # 选择 \boxed 内文本：
                    if hit_source_list[c] == 'gold':
                        chosen_box = gold_text_list[c]
                    else:
                        # 命中 final；若 final 实际为空，回落到 GT
                        if (final_texts[c] or "").strip():
                            chosen_box = final_texts[c]
                        else:
                            chosen_box = gold_text_list[c]


                    new_text = f"<think> {prefix_txt} <stop> </think> $$\n\\boxed{{{chosen_box}}}\n$$ <|im_end|>"
                    response_token_lists[c] = self.tokenizer.encode(
                        new_text, add_special_tokens=True
                    )[:self.config.response_length]


            # 8) pad & 拼接
            response = pad_2d_list_to_length(response_token_lists, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                non_tensor_batch['reward_model'] = _repeat_interleave(non_tensor_batch['reward_model'], self.sampling_params.n)
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)


            seq = torch.cat([idx, response], dim=-1)


        # 位置/注意力
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
                'early_stop_score': early_stop_scores,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size)


        out_batch = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        return out_batch





