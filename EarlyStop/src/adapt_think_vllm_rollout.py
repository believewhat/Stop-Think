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
vLLM rollout with <answer>-aware evaluation (sentence-quantile insertion for correct_think).


responses 的组成（严格行优先、样本内相邻）：
 对每个样本 i：
   - 先追加其 n_rollout 个 vLLM 生成候选 (i, r=0..n_rollout-1)
   - 再追加其“完整 correct_think（按句子分位数最多 10 个插桩点）”，
     这些插桩点上的 <answer>LETTER</answer> 的 LETTER 必须由 “前缀推理段 + prompt” 经 LLM 推断得到；
     并在并入 responses 时，<think> 内部首部加入 <Hard>/<Easy>，尾部追加 <final_answer>GT</final_answer>。
总长度：batch_size * (n_rollout + 1)


评分：
 - 原始候选：直接从候选文本中提取（最多前 10 个）形如 <answer>X</answer> 的标签（且仅接受位于句末的标签），
   以权重 w = 1 - l/L（l: 该标签前可见字符数；L: 全文可见字符总长）做加权正确率：
       score = (∑ w * 1_{X==GT}) / (∑ w)
   若无有效标签，则分=0、count=0。
 - correct_think：先按【句子分位数】构造最多 10 个“前缀思考段”，
   将每一段与 prompt 拼接后交给 LLM 推断 <final_answer>LETTER</final_answer>；
   由于评估 prompt 以 '<final_answer>' 结尾，模型应直接续写 'A' 或 'A</final_answer>' 等；
   若未输出规范的选项字母则跳过该段；对成功段，在对应句末插入 <answer>LETTER</answer>。
   最后用与上面相同的加权方式打分（仅统计成功插入的标签），并回填到该样本的 CT 候选位置。
"""


import os
import re
import numpy as np
from typing import List, Any, Union, Tuple, Dict
from contextlib import contextmanager


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




# =========================
# Helpers
# =========================


# =========================
# Helpers: final_answer（仅接受闭合标签）
# =========================
import re
from typing import Union


# 仅匹配闭合形式：<final_answer>D</final_answer>
_FINAL_ANS_RE = re.compile(
    r"<final_answer>\s*([ABCD])\s*</final_answer>",
    re.IGNORECASE
)



ANSWER_ANY_RE = re.compile(
    r"\(\s*<answer>\s*([ABCD])\s*</answer>\s*\)|<answer>\s*([ABCD])\s*</answer>",
    re.IGNORECASE,
)

_EVAL_LETTER_RE = re.compile(r"^\s*(?:</?final_answer>\s*)*([ABCD])", re.IGNORECASE)



_END_PUNCT = set(list(".?!。！？"))


def _strip_tags(s: str) -> str:
    # 去掉 <think> 与 <final_answer>；answer 标签在计分里单独处理
    return re.sub(r"</?think>|</?final_answer>", "", s or "")


def _prev_non_space_char(s: str, idx: int) -> str:
    k = idx - 1
    while k >= 0 and s[k].isspace():
        k -= 1
    return s[k] if k >= 0 else ""


def _remove_answer_tags(s: str) -> str:
    # 移除两种 answer 形式（带括号或不带）
    return ANSWER_ANY_RE.sub("", s or "")

def _extract_eval_letter(text: str) -> Union[str, None]:
    """从以 '<final_answer>' 结尾的 prompt 的续写中抽取首个选项字母。"""
    m = _EVAL_LETTER_RE.search(text or "")
    if m:
        return m.group(1).upper()
    return None



def extract_answers_and_weights(
    text_with_answers: str, max_answers: int = 10
) -> Tuple[List[str], List[float], List[bool]]:
    """
    提取最多 max_answers 个按出现顺序的 <answer> 标签（两种形态皆可）。
    - "有效标签" 定义：带括号 *且* 句末（<answer> 之前的最近非空白字符是句末标点或换行）
    - 返回:
        answers: 该标签的字母（A-D）
        weights: 线性权重 w = 1 - l/L（l/L 用“移除所有标签后的可见字符”计）
        valids: 该位置是否“有效标签”（True/False）
    说明：
      * 不论是否有效，都会计入“标签总数”（用于分母）；无效标签得分按 0 计。
    """
    core = text_with_answers or ""


    # L: 全文可见字符总长（移除所有 answer 标签 + 去掉 think/final 标记）
    core_no_ans = _remove_answer_tags(core)
    visible_all = _strip_tags(core_no_ans)
    L = len(visible_all)
    if L <= 0:
        return [], [], []


    answers: List[str] = []
    weights: List[float] = []
    valids: List[bool] = []


    for m in ANSWER_ANY_RE.finditer(core):
        if len(answers) >= max_answers:
            break


        # 判断是否带括号（group(1) 命中为带括号，group(2) 为不带）
        letter = (m.group(1) or m.group(2)).upper()
        has_paren = m.group(1) is not None


        # 句末判定：标签起点之前的非空白字符必须是句末标点或换行
        prev_ch = _prev_non_space_char(core, m.start())
        is_sentence_final = (prev_ch in _END_PUNCT) or (prev_ch == "\n")


        # 有效性：带括号 且 句末
        is_valid = has_paren and is_sentence_final


        # l: 标签之前的“可见字符”数
        pre = core[:m.start()]
        pre_no_ans = _remove_answer_tags(pre)
        visible_pre = _strip_tags(pre_no_ans)
        l = len(visible_pre)
        w = max(0.0, min(1.0, 1.0 - (l / L)))


        answers.append(letter)
        weights.append(w)
        valids.append(is_valid)


    return answers, weights, valids


def _score_by_answers(
    text: str, gt_letter: Union[str, None], max_answers: int = 10
) -> Tuple[float, int]:
    """
    将文本中的 (<answer>X</answer>) 和 <answer>X</answer> 统统计入“标签个数”（最多 max_answers 个）。
    - 只有“带括号且句末”的标签才参与得分（正确得 w，错误得 0）
    - 无括号 / 非句末：计入个数但得分=0
    - 返回: (score, count)，其中 score = sum_i (w_i * 1_{valid_i and correct}) / count
    """
    if gt_letter is None:
        return 0.0, 0


    answers, weights, valids = extract_answers_and_weights(text, max_answers=max_answers)
    cnt = len(answers)
    if cnt == 0:
        return 0.0, 0


    total = 0.0
    for a, w, v in zip(answers, weights, valids):
        if v and (a == gt_letter):
            total += float(w)


    score = total / float(cnt)
    return score, cnt





def _prev_non_space_char(s: str, idx: int) -> str:
    """返回 s 中 idx 之前的第一个非空白字符；若无返回空串。"""
    k = idx - 1
    while k >= 0 and s[k].isspace():
        k -= 1
    return s[k] if k >= 0 else ""


def _remove_answer_tags(s: str) -> str:
    return ANSWER_ANY_RE.sub("", s or "")





_SENT_SPLIT_RE = re.compile(r'([\.?!。！？]+[\s]*)')  # 保留终止符


def _split_sentences_preserve_punct(raw: str) -> List[str]:
    """按句子切分，保留终止符到句子末尾；忽略空句。"""
    parts = _SENT_SPLIT_RE.split(raw or "")
    sents: List[str] = []
    for i in range(0, len(parts), 2):
        a = parts[i]
        if not a or a.strip() == "":
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        sents.append(a + punct)
    return sents


def _choose_sentence_quantiles(n_sent: int, m: int) -> List[int]:
    """
    选择 m 个分位的句子索引（单调不降且尽可能均匀）。
    """
    if n_sent <= 0 or m <= 0:
        return []
    m = min(m, n_sent)
    chosen_idx: List[int] = []
    # 使用 (m+1) 分隔，取 1..m 个分割点的左侧
    for i in range(1, m + 1):
        idx = int(np.ceil(i * n_sent / (m + 1))) - 1  # 0-based
        idx = max(0, min(n_sent - 1, idx))
        if len(chosen_idx) == 0 or idx > chosen_idx[-1]:
            chosen_idx.append(idx)
        else:
            # 若重复，尝试向后挪动一位
            next_idx = min(n_sent - 1, chosen_idx[-1] + 1)
            if next_idx > chosen_idx[-1]:
                chosen_idx.append(next_idx)
    # 去重（保持顺序）
    out: List[int] = []
    last = -1
    for x in chosen_idx:
        if x != last:
            out.append(x)
            last = x
    return out


def build_eval_prompt(base_prompt_text: str, segment_text: str) -> str:
    """
    评估时拼接：prompt + <think>{segment}</think> + 以 '<final_answer>' 结尾，诱导模型直接续写选项字母。
    （按你的新约定，这里不再写任何说明或闭合标签）
    """
    return f"{base_prompt_text}\n<think>{segment_text}</think><final_answer>"


def _extract_gt_letter(gt_str: str) -> Union[str, None]:
    m = re.match(r"\s*([ABCD])", str(gt_str).strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None



# =========================
# vLLM Rollout
# =========================


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)




class AdaptThinkvLLMRollout(BaseRollout):


    def __init__(self, model_path: str, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"


        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)


        if kwargs.get('train_tp', None) is not None:
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)


        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"


        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + max(config.response_length, self.config.val_kwargs.max_tokens)
        max_model_len = int(max_model_len)


        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens或 disable chunked prefill')


        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format
        self.actor_module = actor_module
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=self.config.dtype,
            enforce_eager=self.config.enforce_eager,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=False,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=self.config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )


        print(f"[RANK {os.environ['RANK']}] vLLM initialized.")
        print(f"Model path: {model_path}")


        self.inference_engine.sleep(level=1)


        # 生成参数
        kwargs_sp = dict(
            n=1,
            logprobs=0,
            max_tokens=self.config.response_length,
        )
        if vllm_version != '0.3.1':
            kwargs_sp['detokenize'] = False
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs_sp[k] = self.config.get(k)
        print(f"kwargs: {kwargs_sp}")
        self.sampling_params = SamplingParams(**kwargs_sp)


        # 二次评估（用于段内推断 <final_answer>LETTER</final_answer>）
        self.sampling_params2 = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, max_tokens=64, n=1,
        )


        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len


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
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()


        idx = prompts.batch['input_ids']          # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)


        # 去左 pad -> vLLM 输入
        idx_list: List[List[int]] = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))


        # 采样控制
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs_sp = {'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1}
        elif is_validate:
            kwargs_sp = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }
        else:
            kwargs_sp = {}


        with self.update_sampling_params(**kwargs_sp):
            
            outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False
            )
            


            # rollout 数
            n_rollout = len(outputs[0].outputs) if len(outputs) > 0 else 1


            # === 将要返回的列表（严格同序） ===
            responses: List[List[int]] = []
            correct_answers: List[List[int]] = []
            hard_prob: List[float] = []
            sample_of: List[int] = []  # 与 responses 一一对应，记录候选来源样本索引


            # === 一维分数/计数（与 responses 一一对应）===
            total_cands = batch_size * (n_rollout + 1)
            stop_resp_scores = torch.zeros(total_cands, dtype=torch.float32, device=idx.device)
            stop_count = torch.zeros(total_cands, dtype=torch.int32, device=idx.device)


            # ===== per-sample 计算 hard/easy 概率（用 revised_think） =====
            """
            hard_id = self.tokenizer.convert_tokens_to_ids('Hard')
            easy_id = self.tokenizer.convert_tokens_to_ids('Easy')
            per_sample_normprob: List[float] = []
            """
            per_sample_normprob: List[float] = []
            for i in range(batch_size):
                """
                think_raw = prompts.non_tensor_batch['correct_think'][i].replace('->', ' ')
                think_raw = think_raw.replace('<think>', '').replace('</think>', '')
                correct_answer_str = prompts.non_tensor_batch['reward_model'][i]['ground_truth']
                if prompts.non_tensor_batch['solution'][i][0] != correct_answer_str[0]:
                    #revised_think = f"<think> <Hard> {think_raw}</think><final_answer>{correct_answer_str}</final_answer>"
                    revised_think = f"<think>{think_raw}</think><final_answer>{correct_answer_str}</final_answer>"
                else:
                    #revised_think = f"<think> <Easy> {think_raw}</think><final_answer>{correct_answer_str}</final_answer>"
                    revised_think = f"<think> {think_raw}</think><final_answer>{correct_answer_str}</final_answer>"
                
                revised_ids = self.tokenizer.encode(revised_think, add_special_tokens=False)

                
                manual_input_ids = torch.tensor([revised_ids]).to(idx.device)
                hard_attention_mask = (manual_input_ids != self.pad_token_id).long()
                hard_position_ids = torch.arange(manual_input_ids.size(1), device=manual_input_ids.device).unsqueeze(0)
                
                out = self.actor_module(
                    input_ids=manual_input_ids,
                    attention_mask=hard_attention_mask,
                    position_ids=hard_position_ids,
                    use_cache=False
                )
                logits = out.logits[:, :10, :]
                
                tag_start = None
                for t in range(1, manual_input_ids.size(1)):
                    if manual_input_ids[0, t].item() == hard_id:
                        tag_start = t
                        break
                """
                normprob = 1.0
                """
                if tag_start is not None:
                    try:
                        prob_hard = logits[0, tag_start, hard_id].item()
                        prob_easy = logits[0, tag_start, easy_id].item()
                        normprob = (prob_hard + 1e-8) / (prob_hard + prob_easy + 1e-8)
                    except:
                        normprob = 1.0
                """
                per_sample_normprob.append(normprob)


            base_prompts: List[str] = [self.tokenizer.decode(toks, skip_special_tokens=False) for toks in idx_list]


            # ===== 逐样本：先候选，后 CT（保证顺序） =====
            for i in range(batch_size):
                req_out = outputs[i]
                gt_str = prompts.non_tensor_batch['reward_model'][i]['ground_truth']
                gt_ids = self.tokenizer.encode(gt_str)
                gt_letter = _extract_gt_letter(gt_str)


                # 1) 本样本的 n_rollout 候选
                for out_one in req_out.outputs:
                    cand_ids = out_one.token_ids
                    responses.append(cand_ids)
                    correct_answers.append(gt_ids)
                    hard_prob.append(per_sample_normprob[i])
                    sample_of.append(i)


                    k = len(responses) - 1
                    cand_text = self.tokenizer.decode(cand_ids, skip_special_tokens=False)
                    score, cnt = _score_by_answers(cand_text, gt_letter, max_answers=10)
                    stop_resp_scores[k] = score
                    stop_count[k] = cnt


                # 2) 本样本的 correct_think —— 用“句子分位数前缀 + LLM 推断答案”插入 <answer>
                ct_raw = prompts.non_tensor_batch['correct_think'][i].replace('->', ' ')
                ct_raw = ct_raw.replace('<think>', '').replace('</think>', '')


                sents = _split_sentences_preserve_punct(ct_raw)
                chosen_idx = _choose_sentence_quantiles(len(sents), 10)


                # 组装该样本所有“前缀段”的评估 prompts，一次性请求 LLM
                seg_prompts_token_ids: List[List[int]] = []
                seg_owner_indices: List[int] = []  # 对应 chosen_idx 的序号
                for j, sent_idx in enumerate(chosen_idx):
                    prefix_text = "".join(sents[:sent_idx + 1])  # 累积到该句末
                    prompt_text = build_eval_prompt(base_prompts[i], prefix_text)
                    toks = self.tokenizer.encode(prompt_text, add_special_tokens=False)[: self.max_model_len - 1]
                    seg_prompts_token_ids.append(toks)
                    seg_owner_indices.append(j)


                # 运行 LLM 得到每个前缀段的 'A' / 'A</final_answer>' 等
                idx_to_pred_letter: Dict[int, str] = {}
                if len(seg_prompts_token_ids) > 0:
                    outputs_eval = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params2,
                        prompt_token_ids=seg_prompts_token_ids,
                        use_tqdm=False
                    )
                    for jj, outj in enumerate(outputs_eval):
                        text_pred = self.tokenizer.decode(outj.outputs[0].token_ids, skip_special_tokens=False)
                        letter = _extract_eval_letter(text_pred)
                        if letter is None or letter[0] != gt_str[0]:
                            continue  # 该段不插入
                        owner_pos = seg_owner_indices[jj]
                        idx_to_pred_letter[owner_pos] = letter


                # 构造带 <answer> 的 think（只在成功段插入；严格插到句末）
                chunks: List[str] = []
                for pos, sent in enumerate(sents):
                    chunks.append(sent)
                    # 如果该句是被选中的插桩点，且 LLM 成功输出了规范答案，则插入标签
                    try:
                        chosen_pos = chosen_idx.index(pos)
                    except ValueError:
                        chosen_pos = -1
                    if chosen_pos != -1 and (chosen_pos in idx_to_pred_letter):
                        letter = idx_to_pred_letter[chosen_pos]
                        chunks.append(f"(<answer>{letter}</answer>)")
                ct_with_answers = "<think>" + "".join(chunks) + "</think>"


                # 并入 responses 的版本：在 <think> 内首部加 <Hard>/<Easy>，尾部加 <final_answer>GT</final_answer>
                #tag = "Hard" if prompts.non_tensor_batch['solution'][i][0] != gt_str[0] else "Easy"
                inner = ct_with_answers.replace("<think>", "").replace("</think>", "")
                #ct_resp_text = f"<think> <{tag}> {inner}</think><final_answer>{gt_str}</final_answer>"
                ct_resp_text = f"<think> {inner}</think><final_answer>{gt_str}</final_answer>"
                ct_ids = self.tokenizer.encode(ct_resp_text, add_special_tokens=False)[: self.max_model_len - 1]
                """
                if len(ct_ids) > 7000:
                    responses.append(ct_ids[:6000] + ct_ids[-1000:])
                else:
                    responses.append(ct_ids)
                """
                #correct_answers.append(gt_ids)
                #hard_prob.append(per_sample_normprob[i])
                #sample_of.append(i)


                k_ct = len(responses) - 1
                # 基于已插入的 <answer> 标签计算分数（仅统计成功插入的标签）
                ct_score, ct_cnt = _score_by_answers(ct_resp_text, gt_letter, max_answers=10)
                #stop_resp_scores[k_ct] = ct_score
                #stop_count[k_ct] = ct_cnt  # <= 10，取决于句子与成功插入的数量


            # ====== 断言顺序/长度一致性 ======
            
            if self.sampling_params.n > 1 and do_sample:
                idx_exp = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask_exp = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids_exp = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n

            
            # responses -> pad 统一长度
            responses = pad_2d_list_to_length(
                responses, self.pad_token_id, max_length=self.config.val_kwargs.max_tokens
            ).to(idx.device)


            # 拼接 input_ids
            seq = torch.cat([idx_exp, responses], dim=-1)

        # ===== 位置与 mask =====
        response_length = responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids_exp[:, -1:] + delta_position_id
        position_ids_exp = torch.cat([position_ids_exp, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=eos_token_id, dtype=attention_mask_exp.dtype
        )
        attention_mask_exp = torch.cat((attention_mask_exp, response_attention_mask), dim=-1)


        # ===== 返回（仅包含你会用到的 key；顺序严格一致） =====
        batch_td = TensorDict(
            {
                'prompts': idx_exp,
                'responses': responses,                 # 长度 = batch_size * (n_rollout + 1)
                'input_ids': seq,
                'attention_mask': attention_mask_exp,
                'position_ids': position_ids_exp,


                'correct_answer': correct_answers,      # 与 responses 一一对应（ct 用本样本 GT）
                #'hard_prob': hard_prob,                 # 与 responses 一一对应


                # 基于 <answer> 标签的得分/计数（最多 10；对原始候选为文本中已有标签；对 CT 为我们插入的标签）
                #'stop_resp_scores': stop_resp_scores,   # 1D, 长度 = batch_size*(n_rollout+1)
                #'stop_count': stop_count,               # 1D, 同上；有效 <answer> 数
            },
            batch_size=batch_size
        )


        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()


        return DataProto(batch=batch_td)






