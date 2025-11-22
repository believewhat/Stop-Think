import torch
import numpy as np
from collections import defaultdict

class AdaptThinkRewardManager:
    """
    Reward Manager for RayPPO/GRPO, 完全无ref baseline相关逻辑。
    """
    def __init__(self, tokenizer, num_examine=5, compute_score=None,
                 reward_fn_key='data_source', is_training=True):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.is_training = is_training

    def __call__(self, data, return_dict=False):
        """
        data: DataProto，必须包含 batch['responses']、batch['prompts']、batch['attention_mask']，
              non_tensor_batch 必须有 reward_model/ground_truth。
        return_dict: True 返回 {'reward_tensor', 'reward_extra_info'}, False 返回 reward_tensor
        """
        responses = data.batch['responses']
        prompts = data.batch.get('prompts', None)
        attn_mask = data.batch.get('attention_mask', None)

        non_tensor = data.non_tensor_batch
        #hard_prob = data.batch['hard_prob']
        ground_truths = data.batch['correct_answer']

        #stop_resp_scores = data.batch['stop_resp_scores']
        #stop_count = data.batch['stop_count']

        

        batch_size = responses.shape[0]
        reward_tensor = torch.zeros(batch_size, responses.shape[1], dtype=torch.float32, device=responses.device)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        for i in range(batch_size):
            prompt_ids = prompts[i] if prompts is not None else None
            response_ids = responses[i]
            valid_prompt_length = attn_mask[i][:len(prompt_ids)].sum().item() if (attn_mask is not None and prompt_ids is not None) else len(prompt_ids) if prompt_ids is not None else 0
            valid_response_length = attn_mask[i][len(prompt_ids):].sum().item() if (attn_mask is not None and prompt_ids is not None) else len(response_ids)
            valid_prompt_ids = prompt_ids[-int(valid_prompt_length):] if prompt_ids is not None else None
            valid_response_ids = response_ids[:int(valid_response_length)]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True) if valid_prompt_ids is not None else ""
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = self.tokenizer.decode(ground_truths[i])
            data_source = non_tensor[self.reward_fn_key][i] if self.reward_fn_key in non_tensor else "unknown"
            extra_info = non_tensor.get('extra_info', [None]*batch_size)[i]
            # === Custom reward逻辑 ===
            #score_lens = data.batch['score_len']
            #split_scores = data.batch['split_scores']
            if self.compute_score is not None:
                score = self.compute_score(
                    data_source=data_source,
                    #hard_prob=hard_prob[i],
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    tokenizer=self.tokenizer,
                    options=non_tensor['options'][i],
                    #stop_resp_scores=stop_resp_scores[i],
                    #stop_count=stop_count[i],
                    #split_scores=split_scores[i, :].sum(),
                )
            else:
                score = {"acc": int(ground_truth == response_str), "reward": int(ground_truth == response_str)}

            reward = score.get("score", 0.0)

            # 奖励给最后一个token
            if valid_response_length > 0:
                reward_tensor[i, int(valid_response_length) - 1] = reward
            else:
                reward_tensor[i, -1] = reward
            """
            for i in range(score_lens.shape[0]):
                for j in range(score_lens.shape[1]):
                    reward_tensor[i][min(int(score_lens[i][j]), int(valid_response_length)-1)] += split_scores[i][j] / 4
            """
            # logging
            print_key = f"source_{data_source}"
            if already_print_data_sources.get(print_key, 0) < self.num_examine:
                already_print_data_sources[print_key] = already_print_data_sources.get(print_key, 0) + 1
                print(f'\n\n[data_source]{print_key}')
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in score.items():
                    print(f"[{key}]", value)

            for k, v in score.items():
                reward_extra_info[k].append(v)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
