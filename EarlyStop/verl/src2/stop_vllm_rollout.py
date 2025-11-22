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
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)
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
            enable_prefix_caching=False,
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
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            prob_hards = []
            for i, output in enumerate(outputs):              
                mean_hard = []
                for sample_id in range(len(output.outputs)):
                    prob_hard, lp_hard, lp_easy, pos = hard_prob_from_sample(output.outputs[sample_id], self.tokenizer, topk_expected=20)
                    response.append(output.outputs[sample_id].token_ids)
                    prob_hards.append(prob_hard)
                    mean_hard.append(prob_hard)
            
            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                
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

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'hard_prob': prob_hards,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)
        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
