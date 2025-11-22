from verl.trainer.ray_ppo_trainer import RayPPOTrainer
from verl import DataProto
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re


def extract_chain_label2(text, text2, answer=None):
    is_correct = False
    answer_tag = ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        parsed_answer = match.group(1).strip()
        answer_tag = match.group(0)
        if answer is not None and len(answer) > 0:
            gold = str(answer[0]).strip().upper()
            pred = str(parsed_answer).strip().upper()
            is_correct = pred.startswith(gold)
    else:
        answer_tag = ""

    def difficulty_replace():
        return "<Easy>" if is_correct else "<Hard>"

    return f"<think> {difficulty_replace()} {text2}</think>{answer_tag}"


class RayGRPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def postprocess_batch(self, batch: DataProto):
        tokenizer = self.tokenizer
        if "responses" not in batch.batch:
            return batch

        completion_ids = batch.batch["responses"]
        answers = batch.non_tensor_batch.get("answer", [None] * len(completion_ids))
        correct_thinks = batch.non_tensor_batch.get("correct_think", [None] * len(completion_ids))

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        structured_completions = []

        for pred, think, ans in zip(completions, correct_thinks, answers):
            label_str = extract_chain_label2(pred, think, ans)
            structured_completions.append(label_str)

        input_ids_list = tokenizer(structured_completions, add_special_tokens=False)["input_ids"]
        input_ids_tensor_list = [torch.tensor(ids) for ids in input_ids_list]
        padded_input_ids = pad_sequence(input_ids_tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        batch.batch["structured_completion_ids"] = padded_input_ids
        return batch

    def compute_loss(self, model, inputs, *args, **kwargs):
        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        rl_loss = self._compute_loss(model, inputs)

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        answers = self.processing_class.batch_decode(
            inputs['answer'].tolist(), skip_special_tokens=True
        )
        correct_think = self.processing_class.batch_decode(
            inputs['correct_think'], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        correct_think = [x.split('!!!')[0] for x in correct_think]

        batch_size = prompt_ids.size(0)
        prompt_ids_list, chosen_ids_list, rejected_ids_list = [], [], []

        for i in range(int(batch_size / 2)):
            cur_prompt_ids = prompt_ids[i]
            completion_text = self.processing_class.decode(completion_ids[i], skip_special_tokens=True)
            label_str = extract_chain_label2(completion_text, correct_think[i], answers[i])
            if not label_str:
                continue
            chosen_ids = torch.tensor(
                self.processing_class.encode(label_str, add_special_tokens=False),
                dtype=cur_prompt_ids.dtype, device=cur_prompt_ids.device
            )
            input_ids_chosen = torch.cat([cur_prompt_ids, chosen_ids], dim=0)
            input_ids_rejected = torch.cat([cur_prompt_ids, completion_ids[i]], dim=0)
            prompt_ids_list.append(cur_prompt_ids)
            chosen_ids_list.append(input_ids_chosen)
            rejected_ids_list.append(input_ids_rejected)

        if len(chosen_ids_list) == 0:
            return rl_loss

        chosen_batch = pad_sequence(chosen_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id)
        rejected_batch = pad_sequence(rejected_ids_list, batch_first=True, padding_value=self.processing_class.pad_token_id)
        chosen_attention_mask = (chosen_batch != self.processing_class.pad_token_id).long()
        rejected_attention_mask = (rejected_batch != self.processing_class.pad_token_id).long()

        beta = 0.1

        def get_completion_logps(model, input_ids, attention_mask, prompt_lens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            labels = input_ids[:, 1:]
            log_probs = log_probs[:, :-1, :]
            mask = attention_mask[:, 1:]
            chosen_logprobs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) * mask
            B, Lm1 = chosen_logprobs.size()
            idxs = torch.arange(Lm1, device=chosen_logprobs.device).unsqueeze(0).expand(B, Lm1)
            comp_mask = idxs >= (torch.as_tensor(prompt_lens, device=chosen_logprobs.device).unsqueeze(1) - 1)
            comp_logprobs = chosen_logprobs * comp_mask
            return comp_logprobs.sum(dim=1)

        prompt_len = len(prompt_ids[0])
        B = chosen_batch.size(0)
        prompt_lens = torch.full((B,), prompt_len, device=chosen_batch.device)
        policy_chosen_logps = get_completion_logps(model, chosen_batch, chosen_attention_mask, prompt_lens)
        policy_rejected_logps = get_completion_logps(model, rejected_batch, rejected_attention_mask, prompt_lens)

        with torch.no_grad():
            ref_chosen_logps = get_completion_logps(self.ref_model, chosen_batch, chosen_attention_mask, prompt_lens)
            ref_rejected_logps = get_completion_logps(self.ref_model, rejected_batch, rejected_attention_mask, prompt_lens)

        delta = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        dpo_loss = -F.logsigmoid(beta * delta).mean()
        total_loss = rl_loss + dpo_loss * 5
        return total_loss

    def postprocess_metrics(self, metrics, batch: DataProto):
        return metrics
