
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from verl.workers.actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import compute_policy_loss, agg_loss

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']

class AdaptThinkDataParallelPPOActor(DataParallelPPOActor):

    def __init__(
        self,
        config,
        tokenizer,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        ref_model: nn.Module = None,  # 新增参数：reference model
        dpo_beta: float = 0.1,
        dpo_weight: float = 5.0,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.ref_model = ref_model
        self.dpo_beta = dpo_beta
        self.dpo_weight = dpo_weight
        self.tokenizer = tokenizer
    
    """
    def get_completion_logps(self, model, input_ids, attention_mask, prompt_lens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits  # [B, L, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:]
        log_probs = log_probs[:, :-1, :]
        mask = attention_mask[:, 1:]
        chosen_logprobs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) * mask  # [B, L-1]
        B, Lm1 = chosen_logprobs.size()
        device = chosen_logprobs.device
        idxs = torch.arange(Lm1, device=device).unsqueeze(0).expand(B, Lm1)
        comp_mask = idxs >= (prompt_lens.unsqueeze(1) - 1)
        comp_logprobs = chosen_logprobs * comp_mask
        return comp_logprobs.sum(dim=1)  # [B]
    """
    """
    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info['temperature']
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'position_ids',
            'old_log_probs', 'advantages', 'prompt_ids', 'chosen_ids', 'rejected_ids'
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        adapt_think_adjust_old_log_prob=self.adapt_think_adjust_old_logprobs)
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    # === DPO Loss部分 ===
                    dpo_loss = 0.0
                    # 只要有prompt_ids, chosen_ids, rejected_ids才计算DPO
                    if (
                        "prompt_ids" in data and "chosen_ids" in data and "rejected_ids" in data
                        and self.ref_model is not None
                    ):
                        prompt_ids = data["prompt_ids"]
                        chosen_ids = data["chosen_ids"]
                        rejected_ids = data["rejected_ids"]
                        attention_mask_chosen = (chosen_ids != self.actor_module.config.pad_token_id).long()
                        attention_mask_rejected = (rejected_ids != self.actor_module.config.pad_token_id).long()
                        prompt_lens = torch.full((chosen_ids.shape[0],), prompt_ids.shape[1], device=chosen_ids.device)
                        # 当前policy logp
                        policy_chosen_logps = self.get_completion_logps(self.actor_module, chosen_ids, attention_mask_chosen, prompt_lens)
                        policy_rejected_logps = self.get_completion_logps(self.actor_module, rejected_ids, attention_mask_rejected, prompt_lens)
                        # ref logp
                        with torch.no_grad():
                            ref_chosen_logps = self.get_completion_logps(self.ref_model, chosen_ids, attention_mask_chosen, prompt_lens)
                            ref_rejected_logps = self.get_completion_logps(self.ref_model, rejected_ids, attention_mask_rejected, prompt_lens)
                        delta = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
                        dpo_loss = -torch.nn.functional.logsigmoid(self.dpo_beta * delta).mean()

                    # === policy loss + dpo loss ===
                    policy_loss = pg_loss - entropy_loss * entropy_coeff + dpo_loss * self.dpo_weight

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/dpo_loss': float(dpo_loss) if isinstance(dpo_loss, torch.Tensor) else dpo_loss,
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
    """