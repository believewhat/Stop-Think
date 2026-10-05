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
Single Process Actor
"""

import itertools
import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


def _get_response_loss_mask(batch):
    """Select the actor mask while keeping forced suffix tokens scoreable."""
    responses = batch['responses']
    if 'actor_loss_mask' in batch.keys():
        actor_loss_mask = batch['actor_loss_mask']
        if actor_loss_mask.shape != responses.shape:
            raise ValueError(
                'actor_loss_mask must have the same shape as responses: '
                f'{tuple(actor_loss_mask.shape)} != {tuple(responses.shape)}')
        return actor_loss_mask
    response_length = responses.size(1)
    return batch['attention_mask'][:, -response_length:]


def _loss_aggregation_denominator(response_mask, loss_agg_mode):
    """Return the additive denominator for one policy-loss batch.

    ``agg_loss`` returns a mean local to the tensor passed to it.  To recover
    the same result after splitting a PPO mini-batch into arbitrary dynamic
    micro-batches, each local mean must be weighted by the fraction of the
    *global mini-batch denominator* represented by that micro-batch.  DAPO's
    token-level objective uses active actor tokens as that denominator; the
    two legacy sequence-level objectives use rollout rows.

    The actor mask is intentional here.  In controller-terminal training it
    excludes atomic stop actions, the canonical forced answer suffix, and the
    discarded physical tail, so none of those tokens can dilute ordinary
    GRPO.  Stop-action auxiliary objectives retain their separate rollout-row
    normalization below.
    """

    if not isinstance(response_mask, torch.Tensor):
        raise TypeError('response_mask must be a torch.Tensor')
    if response_mask.ndim != 2:
        raise ValueError(
            'response_mask must be rank two, got '
            f'{tuple(response_mask.shape)}')
    if torch.any((response_mask != 0) & (response_mask != 1)):
        raise ValueError('response_mask must be binary')

    if loss_agg_mode == 'token-mean':
        return float(response_mask.detach().sum().item())
    if loss_agg_mode in ('seq-mean-token-sum', 'seq-mean-token-mean'):
        return float(response_mask.size(0))
    raise ValueError(f'Invalid loss_agg_mode: {loss_agg_mode}')


def _micro_batch_loss_scale(
    response_mask,
    *,
    loss_agg_mode,
    mini_batch_denominator,
):
    """Scale a micro-batch mean into an exact PPO mini-batch mean."""

    total = float(mini_batch_denominator)
    if not math.isfinite(total) or total < 0:
        raise ValueError(
            'mini_batch_denominator must be finite and non-negative, got '
            f'{mini_batch_denominator}')
    local = _loss_aggregation_denominator(response_mask, loss_agg_mode)
    tolerance = max(1e-6, abs(total) * 1e-12)
    if local > total + tolerance:
        raise ValueError(
            'micro-batch denominator cannot exceed its PPO mini-batch '
            f'denominator: {local} > {total}')
    if total == 0:
        if local != 0:
            raise ValueError(
                'a zero-denominator PPO mini-batch cannot contain active '
                'micro-batch loss elements')
        # A controller can terminate every row before any ordinary GRPO token.
        # Keep ordinary policy/entropy/KL at exactly zero while still allowing
        # the separately normalized stop-action losses to train.
        return 0.0
    return local / total


def _compute_weighted_negative_policy_loss(
    old_log_prob,
    log_prob,
    response_mask,
    action_mask,
    action_weight,
    clip_ratio,
    clip_ratio_low,
    clip_ratio_high,
    clip_ratio_c,
    *,
    objective_name='weighted_negative',
    max_row_weight_sum: Optional[float] = 1.0,
):
    """Return a clipped negative-action PPO loss and its normalization stats.

    ``response_mask`` is the physical valid-response mask.  The auxiliary
    action mask is deliberately independent of the ordinary GRPO actor mask:
    post-accepted tail actions, for example, retain their sequence-level
    accuracy/format gradient while also receiving this bounded local penalty.

    The returned loss is averaged over selected actions by
    :func:`compute_policy_loss`.  Callers multiply it by ``action_count`` and
    divide by the PPO mini-batch row count, recovering the sum of per-action
    weights per rollout instead of diluting it by response length or dynamic
    micro-batch packing.
    """

    name = str(objective_name)
    if not name:
        raise ValueError('objective_name must not be empty')
    for tensor_name, tensor in (
        ('action_mask', action_mask),
        ('action_weight', action_weight),
    ):
        if tensor.shape != log_prob.shape:
            raise ValueError(
                f'{name} {tensor_name} must have the same shape as log_prob: '
                f'{tuple(tensor.shape)} != {tuple(log_prob.shape)}')

    selected_actions = action_mask.to(
        device=log_prob.device, dtype=log_prob.dtype)
    weights = action_weight.to(
        device=log_prob.device, dtype=log_prob.dtype)
    if torch.any((selected_actions != 0) & (selected_actions != 1)):
        raise ValueError(f'{name} action_mask must be binary')
    if not torch.all(torch.isfinite(weights)):
        raise ValueError(f'{name} action_weight must be finite')
    if torch.any(weights < 0):
        raise ValueError(f'{name} action_weight must be non-negative')
    if torch.any(weights * (1.0 - selected_actions) != 0):
        raise ValueError(
            f'{name} action_weight must be zero outside action_mask')
    if torch.any((selected_actions > 0) & (weights <= 0)):
        raise ValueError(
            f'{name} selected actions must have strictly positive weight')

    valid_response_mask = response_mask.to(
        device=log_prob.device, dtype=log_prob.dtype)
    if valid_response_mask.shape != log_prob.shape:
        raise ValueError(
            f'{name} response_mask must have the same shape as log_prob: '
            f'{tuple(valid_response_mask.shape)} != '
            f'{tuple(log_prob.shape)}')
    if torch.any(
            (valid_response_mask != 0) & (valid_response_mask != 1)):
        raise ValueError(f'{name} response_mask must be binary')
    if torch.any(selected_actions > valid_response_mask):
        raise ValueError(
            f'{name} actions must be inside the valid response mask')

    row_weight_sums = weights.sum(dim=-1)
    if max_row_weight_sum is not None:
        max_sum = float(max_row_weight_sum)
        if not math.isfinite(max_sum) or max_sum <= 0:
            raise ValueError(
                'max_row_weight_sum must be finite and positive when set, '
                f'got {max_row_weight_sum}')
        # Float32 construction of normalized event weights can accumulate a
        # few ulps above one.  The tolerance accepts only that numerical noise,
        # not an accidentally unbounded per-rollout objective.
        tolerance = max(1e-6, 8.0 * torch.finfo(weights.dtype).eps * max_sum)
        if torch.any(row_weight_sums > max_sum + tolerance):
            raise ValueError(
                f'{name} action_weight sum must be <= {max_sum} per row')

    action_count = selected_actions.sum()
    weight_sum = weights.sum()
    if action_count.item() == 0:
        if weight_sum.item() != 0:
            raise ValueError(
                f'{name} has non-zero weight without selected actions')
        zero = log_prob.sum() * 0.0
        return zero, action_count, weight_sum

    negative_advantages = -weights
    negative_pg_loss, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=negative_advantages,
        response_mask=selected_actions,
        cliprange=clip_ratio,
        cliprange_low=clip_ratio_low,
        cliprange_high=clip_ratio_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode='token-mean',
    )
    return negative_pg_loss, action_count, weight_sum


def _compute_overflow_stop_policy_loss(
    old_log_prob,
    log_prob,
    response_mask,
    overflow_stop_mask,
    clip_ratio,
    clip_ratio_low,
    clip_ratio_high,
    clip_ratio_c,
):
    """Return PPO loss averaged only over illegal stop actions and their count.

    ``response_mask`` is the physical valid-response mask, not the ordinary
    GRPO actor mask.  Overflow stops are excluded from GRPO and receive only
    this separately normalized negative auxiliary loss.
    """

    # Preserve the v12 unit penalty and public helper signature.  v13 routes
    # generalized stop surfaces through the weighted objective below and sets
    # this legacy coefficient to zero, but old checkpoints/launchers remain
    # executable and retain their original per-overflow-action strength.
    overflow_mask = overflow_stop_mask.to(
        device=log_prob.device, dtype=log_prob.dtype)
    overflow_pg_loss, overflow_action_count, _ = (
        _compute_weighted_negative_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            action_mask=overflow_mask,
            action_weight=overflow_mask,
            clip_ratio=clip_ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            objective_name='overflow_stop',
            max_row_weight_sum=None,
        )
    )
    return overflow_pg_loss, overflow_action_count


def _compute_verified_stop_policy_loss(
    old_log_prob,
    log_prob,
    response_mask,
    verified_stop_mask,
    verified_stop_aux_weight,
    clip_ratio,
    clip_ratio_low,
    clip_ratio_high,
    clip_ratio_c,
):
    """Return earliness-weighted PPO loss over oracle-verified stop actions."""

    if verified_stop_mask.shape != log_prob.shape:
        raise ValueError(
            'verified_stop_mask must have the same shape as log_prob: '
            f'{tuple(verified_stop_mask.shape)} != {tuple(log_prob.shape)}')
    if verified_stop_aux_weight.shape != log_prob.shape:
        raise ValueError(
            'verified_stop_aux_weight must have the same shape as log_prob: '
            f'{tuple(verified_stop_aux_weight.shape)} != '
            f'{tuple(log_prob.shape)}')

    verified_stop_mask = verified_stop_mask.to(
        device=log_prob.device, dtype=log_prob.dtype)
    verified_stop_aux_weight = verified_stop_aux_weight.to(
        device=log_prob.device, dtype=log_prob.dtype)
    if torch.any((verified_stop_mask != 0) & (verified_stop_mask != 1)):
        raise ValueError('verified_stop_mask must be binary')
    if not torch.all(torch.isfinite(verified_stop_aux_weight)):
        raise ValueError('verified_stop_aux_weight must be finite')
    if torch.any(verified_stop_aux_weight < 0):
        raise ValueError('verified_stop_aux_weight must be non-negative')
    if torch.any(
            verified_stop_aux_weight
            * (1.0 - verified_stop_mask) != 0):
        raise ValueError(
            'verified_stop_aux_weight must be zero outside verified_stop_mask')
    if torch.any(
            (verified_stop_mask > 0) & (verified_stop_aux_weight <= 0)):
        raise ValueError(
            'verified stop actions must have strictly positive auxiliary weight')

    valid_response_mask = response_mask.to(
        device=log_prob.device, dtype=log_prob.dtype)
    if valid_response_mask.shape != log_prob.shape:
        raise ValueError(
            'response_mask must have the same shape as log_prob: '
            f'{tuple(valid_response_mask.shape)} != {tuple(log_prob.shape)}')
    if torch.any(
            (valid_response_mask != 0) & (valid_response_mask != 1)):
        raise ValueError('response_mask must be binary')
    if torch.any(verified_stop_mask > valid_response_mask):
        raise ValueError(
            'verified stop actions must be inside the valid response mask')

    verified_action_count = verified_stop_mask.sum()
    if verified_action_count.item() == 0:
        return log_prob.sum() * 0.0, verified_action_count
    verified_pg_loss, _, _, _ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=verified_stop_aux_weight,
        response_mask=verified_stop_mask,
        cliprange=clip_ratio,
        cliprange_low=clip_ratio_low,
        cliprange_high=clip_ratio_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode='token-mean',
    )
    return verified_pg_loss, verified_action_count


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            if self.config.get(
                    'empty_cache_before_optimizer_step', False):
                # Adam creates its FP32 state lazily on the first update.  With
                # two FSDP ranks, releasing inactive allocator blocks here
                # avoids turning harmless fragmentation into an optimizer-step
                # OOM without changing parameters, gradients, or Adam math.
                torch.cuda.empty_cache()
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        overflow_stop_aux_loss_coef = float(
            self.config.get('overflow_stop_aux_loss_coef', 0.0))
        if (not math.isfinite(overflow_stop_aux_loss_coef)
                or overflow_stop_aux_loss_coef < 0):
            raise ValueError(
                'overflow_stop_aux_loss_coef must be finite and non-negative, '
                f'got {overflow_stop_aux_loss_coef}')
        verified_stop_aux_loss_coef = float(
            self.config.get('verified_stop_aux_loss_coef', 0.0))
        if (not math.isfinite(verified_stop_aux_loss_coef)
                or verified_stop_aux_loss_coef < 0):
            raise ValueError(
                'verified_stop_aux_loss_coef must be finite and non-negative, '
                f'got {verified_stop_aux_loss_coef}')
        negative_stop_aux_loss_coef = float(
            self.config.get('negative_stop_aux_loss_coef', 0.0))
        if (not math.isfinite(negative_stop_aux_loss_coef)
                or negative_stop_aux_loss_coef < 0):
            raise ValueError(
                'negative_stop_aux_loss_coef must be finite and non-negative, '
                f'got {negative_stop_aux_loss_coef}')
        incorrect_stop_aux_loss_coef = float(
            self.config.get('incorrect_stop_aux_loss_coef', 0.0))
        if (not math.isfinite(incorrect_stop_aux_loss_coef)
                or incorrect_stop_aux_loss_coef < 0):
            raise ValueError(
                'incorrect_stop_aux_loss_coef must be finite and '
                f'non-negative, got {incorrect_stop_aux_loss_coef}')
        post_accepted_tail_aux_loss_coef = float(
            self.config.get('post_accepted_tail_aux_loss_coef', 0.0))
        if (not math.isfinite(post_accepted_tail_aux_loss_coef)
                or post_accepted_tail_aux_loss_coef < 0):
            raise ValueError(
                'post_accepted_tail_aux_loss_coef must be finite and '
                f'non-negative, got {post_accepted_tail_aux_loss_coef}')
        if (negative_stop_aux_loss_coef > 0
                and overflow_stop_aux_loss_coef > 0):
            raise ValueError(
                'negative_stop_aux_loss_coef and '
                'overflow_stop_aux_loss_coef cannot both be positive: the '
                'generalized negative-stop objective already includes every '
                'legacy overflow action')
        legacy_stop_aux_loss_enabled = (
            verified_stop_aux_loss_coef > 0
            or overflow_stop_aux_loss_coef > 0
        )
        atomic_stop_aux_schema_enabled = (
            legacy_stop_aux_loss_enabled
            or incorrect_stop_aux_loss_coef > 0
        )

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if 'actor_loss_mask' in data.batch.keys():
            select_keys.append('actor_loss_mask')
        if atomic_stop_aux_schema_enabled:
            if 'atomic_stop_mask' not in data.batch.keys():
                raise KeyError(
                    'atomic_stop_mask is required when stop auxiliary loss is '
                    'enabled')
            select_keys.append('atomic_stop_mask')
            required_stop_keys = {
                'overflow_stop_mask', 'verified_stop_mask'}
            if verified_stop_aux_loss_coef > 0:
                required_stop_keys.add('verified_stop_aux_weight')
            missing_stop_keys = sorted(
                required_stop_keys.difference(data.batch.keys()))
            if missing_stop_keys:
                raise KeyError(
                    'stop auxiliary loss is missing batch keys: '
                    f'{missing_stop_keys}')
            select_keys.extend(sorted(required_stop_keys))
        weighted_negative_objectives = (
            (
                'negative_stop_aux',
                negative_stop_aux_loss_coef,
                'negative_stop_aux_mask',
                'negative_stop_aux_weight',
            ),
            (
                'incorrect_stop_aux',
                incorrect_stop_aux_loss_coef,
                'incorrect_stop_aux_mask',
                'incorrect_stop_aux_weight',
            ),
            (
                'post_accepted_tail_aux',
                post_accepted_tail_aux_loss_coef,
                'post_accepted_tail_aux_mask',
                'post_accepted_tail_aux_weight',
            ),
        )
        for objective_name, coefficient, mask_key, weight_key in (
                weighted_negative_objectives):
            if coefficient <= 0:
                continue
            missing_keys = [
                key for key in (mask_key, weight_key)
                if key not in data.batch.keys()
            ]
            if missing_keys:
                raise KeyError(
                    f'{objective_name} is missing batch keys: {missing_keys}')
            select_keys.extend((mask_key, weight_key))
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                mini_batch_row_count = int(
                    mini_batch.batch.batch_size[0]
                    if isinstance(mini_batch, DataProto)
                    else mini_batch.batch_size[0]
                )
                loss_agg_mode = self.config.loss_agg_mode
                mini_batch_tensors = (
                    mini_batch.batch
                    if isinstance(mini_batch, DataProto)
                    else mini_batch
                )
                mini_batch_response_mask = _get_response_loss_mask(
                    mini_batch_tensors)
                mini_batch_loss_denominator = (
                    _loss_aggregation_denominator(
                        mini_batch_response_mask,
                        loss_agg_mode,
                    )
                )
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_mask = _get_response_loss_mask(data)
                    valid_response_mask = data['attention_mask'][
                        :, -responses.size(1):]
                    if atomic_stop_aux_schema_enabled:
                        atomic_stop_mask = data['atomic_stop_mask'].to(
                            dtype=response_mask.dtype)
                        if atomic_stop_mask.shape != response_mask.shape:
                            raise ValueError(
                                'atomic_stop_mask must have the same shape as '
                                'the response loss mask')
                        if torch.any(
                            (atomic_stop_mask != 0)
                            & (atomic_stop_mask != 1)
                        ):
                            raise ValueError('atomic_stop_mask must be binary')
                        if torch.any(atomic_stop_mask > valid_response_mask):
                            raise ValueError(
                                'atomic stop actions must be inside the valid '
                                'response mask')
                        verified_stop_mask = data[
                            'verified_stop_mask'].to(
                                dtype=response_mask.dtype)
                        overflow_stop_mask = data[
                            'overflow_stop_mask'].to(
                                dtype=response_mask.dtype)
                        if torch.any(
                            verified_stop_mask > atomic_stop_mask
                        ):
                            raise ValueError(
                                'verified_stop_mask must be a subset of '
                                'atomic_stop_mask')
                        if torch.any(
                            overflow_stop_mask > atomic_stop_mask
                        ):
                            raise ValueError(
                                'overflow_stop_mask must be a subset of '
                                'atomic_stop_mask')
                        if torch.any(
                            (verified_stop_mask > 0)
                            & (overflow_stop_mask > 0)
                        ):
                            raise ValueError(
                                'verified and overflow stop masks must be '
                                'disjoint')
                        if post_accepted_tail_aux_loss_coef > 0:
                            tail_stop_mask = data[
                                'post_accepted_tail_aux_mask'].to(
                                    dtype=response_mask.dtype)
                            if torch.any(
                                (tail_stop_mask > 0)
                                & (atomic_stop_mask > 0)
                            ):
                                raise ValueError(
                                    'post-accepted tail must exclude every '
                                    'atomic stop action')
                        if incorrect_stop_aux_loss_coef > 0:
                            incorrect_stop_mask = data[
                                'incorrect_stop_aux_mask'].to(
                                    dtype=response_mask.dtype)
                            if torch.any(
                                incorrect_stop_mask > atomic_stop_mask
                            ):
                                raise ValueError(
                                    'incorrect_stop_aux_mask must be a subset '
                                    'of atomic_stop_mask')
                            if torch.any(
                                (incorrect_stop_mask > 0)
                                & (verified_stop_mask > 0)
                            ):
                                raise ValueError(
                                    'incorrect and verified stop masks must be '
                                    'disjoint')
                            if torch.any(
                                (incorrect_stop_mask > 0)
                                & (response_mask > 0)
                            ):
                                raise ValueError(
                                    'incorrect stop actions must be excluded '
                                    'from trajectory GRPO')
                            if (
                                negative_stop_aux_loss_coef > 0
                                and torch.any(
                                    (incorrect_stop_mask > 0)
                                    & (data['negative_stop_aux_mask'] > 0)
                                )
                            ):
                                raise ValueError(
                                    'incorrect and generalized-negative stop '
                                    'masks must be disjoint')
                            if (
                                post_accepted_tail_aux_loss_coef > 0
                                and torch.any(
                                    (incorrect_stop_mask > 0)
                                    & (data['post_accepted_tail_aux_mask'] > 0)
                                )
                            ):
                                raise ValueError(
                                    'incorrect stop and post-accepted tail '
                                    'masks must be disjoint')
                        enabled_atomic_stops = (
                            (atomic_stop_mask > 0) & (response_mask > 0)
                        )
                        if torch.any(enabled_atomic_stops):
                            raise ValueError(
                                'every atomic stop action must be excluded '
                                'from trajectory GRPO')
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    ordinary_loss_scale = _micro_batch_loss_scale(
                        response_mask,
                        loss_agg_mode=loss_agg_mode,
                        mini_batch_denominator=(
                            mini_batch_loss_denominator
                        ),
                    )
                    micro_batch_active_token_count = (
                        _loss_aggregation_denominator(
                            response_mask,
                            'token-mean',
                        )
                    )
                    if micro_batch_active_token_count > 0:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        )
                        # compute entropy loss from entropy
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        # Masked means over an empty actor trajectory are not
                        # guaranteed to be finite.  Keep the forward pass (its
                        # log-probs are still needed by stop-action auxiliary
                        # objectives), but construct graph-connected exact
                        # zeros without evaluating PG/entropy masked means.
                        ordinary_zero = log_prob.sum() * 0.0
                        pg_loss = ordinary_zero
                        entropy_loss = ordinary_zero
                        ppo_kl = ordinary_zero
                        pg_clipfrac = ordinary_zero
                        pg_clipfrac_lower = ordinary_zero

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        if micro_batch_active_token_count > 0:
                            ref_log_prob = data['ref_log_prob']
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = agg_loss(
                                loss_mat=kld,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            kl_loss = ordinary_zero

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    # ``policy_loss`` is a mean over this micro-batch.  Weight
                    # it by its share of the complete PPO mini-batch
                    # denominator.  In token-mean mode this is active-token
                    # mass, not row count; therefore arbitrary dynamic packing
                    # reconstructs exactly the unsplit DAPO objective.
                    loss = policy_loss * ordinary_loss_scale

                    verified_stop_aux_loss = log_prob.sum() * 0.0
                    verified_stop_action_count = log_prob.new_zeros(())
                    verified_stop_aux_contribution = log_prob.sum() * 0.0
                    overflow_stop_aux_loss = log_prob.sum() * 0.0
                    overflow_stop_action_count = log_prob.new_zeros(())
                    overflow_stop_aux_contribution = log_prob.sum() * 0.0
                    negative_stop_aux_loss = log_prob.sum() * 0.0
                    negative_stop_action_count = log_prob.new_zeros(())
                    negative_stop_aux_weight_sum = log_prob.new_zeros(())
                    negative_stop_aux_contribution = log_prob.sum() * 0.0
                    incorrect_stop_aux_loss = log_prob.sum() * 0.0
                    incorrect_stop_action_count = log_prob.new_zeros(())
                    incorrect_stop_aux_weight_sum = log_prob.new_zeros(())
                    incorrect_stop_aux_contribution = log_prob.sum() * 0.0
                    post_accepted_tail_aux_loss = log_prob.sum() * 0.0
                    post_accepted_tail_action_count = log_prob.new_zeros(())
                    post_accepted_tail_aux_weight_sum = log_prob.new_zeros(())
                    post_accepted_tail_aux_contribution = (
                        log_prob.sum() * 0.0)
                    if verified_stop_aux_loss_coef > 0:
                        verified_stop_aux_loss, verified_stop_action_count = (
                            _compute_verified_stop_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                response_mask=valid_response_mask,
                                verified_stop_mask=data['verified_stop_mask'],
                                verified_stop_aux_weight=data[
                                    'verified_stop_aux_weight'],
                                clip_ratio=clip_ratio,
                                clip_ratio_low=clip_ratio_low,
                                clip_ratio_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                            )
                        )
                        # Sum the weighted verified-action losses across
                        # micro-batches, then average per rollout row.  Stop
                        # strength is independent of response length.
                        verified_stop_aux_contribution = (
                            verified_stop_aux_loss
                            * verified_stop_action_count
                            / float(mini_batch_row_count)
                            * verified_stop_aux_loss_coef
                        )
                        loss = loss + verified_stop_aux_contribution
                    if overflow_stop_aux_loss_coef > 0:
                        overflow_stop_aux_loss, overflow_stop_action_count = (
                            _compute_overflow_stop_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                response_mask=valid_response_mask,
                                overflow_stop_mask=data['overflow_stop_mask'],
                                clip_ratio=clip_ratio,
                                clip_ratio_low=clip_ratio_low,
                                clip_ratio_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                            )
                        )
                        # Sum every equal-strength post-budget stop loss across
                        # micro-batches, then average over rollout rows in this
                        # PPO mini-batch.  A rollout with k illegal stops therefore
                        # receives k negative action terms, independent of length.
                        overflow_stop_aux_contribution = (
                            overflow_stop_aux_loss
                            * overflow_stop_action_count
                            / float(mini_batch_row_count)
                            * overflow_stop_aux_loss_coef
                        )
                        loss = loss + overflow_stop_aux_contribution
                    if negative_stop_aux_loss_coef > 0:
                        (
                            negative_stop_aux_loss,
                            negative_stop_action_count,
                            negative_stop_aux_weight_sum,
                        ) = _compute_weighted_negative_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            response_mask=valid_response_mask,
                            action_mask=data['negative_stop_aux_mask'],
                            action_weight=data['negative_stop_aux_weight'],
                            clip_ratio=clip_ratio,
                            clip_ratio_low=clip_ratio_low,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            objective_name='negative_stop_aux',
                            max_row_weight_sum=1.0,
                        )
                        negative_stop_aux_contribution = (
                            negative_stop_aux_loss
                            * negative_stop_action_count
                            / float(mini_batch_row_count)
                            * negative_stop_aux_loss_coef
                        )
                        loss = loss + negative_stop_aux_contribution
                    if incorrect_stop_aux_loss_coef > 0:
                        (
                            incorrect_stop_aux_loss,
                            incorrect_stop_action_count,
                            incorrect_stop_aux_weight_sum,
                        ) = _compute_weighted_negative_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            response_mask=valid_response_mask,
                            action_mask=data['incorrect_stop_aux_mask'],
                            action_weight=data['incorrect_stop_aux_weight'],
                            clip_ratio=clip_ratio,
                            clip_ratio_low=clip_ratio_low,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            objective_name='incorrect_stop_aux',
                            max_row_weight_sum=8.0,
                        )
                        # Each recognized wrong probe retains its own
                        # earliness-weighted negative term.  Multiplying by
                        # action_count reverses only the helper's token mean;
                        # division by the complete PPO mini-batch row count
                        # makes the result invariant to dynamic packing.
                        incorrect_stop_aux_contribution = (
                            incorrect_stop_aux_loss
                            * incorrect_stop_action_count
                            / float(mini_batch_row_count)
                            * incorrect_stop_aux_loss_coef
                        )
                        loss = loss + incorrect_stop_aux_contribution
                    if post_accepted_tail_aux_loss_coef > 0:
                        (
                            post_accepted_tail_aux_loss,
                            post_accepted_tail_action_count,
                            post_accepted_tail_aux_weight_sum,
                        ) = _compute_weighted_negative_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            response_mask=valid_response_mask,
                            action_mask=data['post_accepted_tail_aux_mask'],
                            action_weight=data['post_accepted_tail_aux_weight'],
                            clip_ratio=clip_ratio,
                            clip_ratio_low=clip_ratio_low,
                            clip_ratio_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            objective_name='post_accepted_tail_aux',
                            max_row_weight_sum=1.0,
                        )
                        post_accepted_tail_aux_contribution = (
                            post_accepted_tail_aux_loss
                            * post_accepted_tail_action_count
                            / float(mini_batch_row_count)
                            * post_accepted_tail_aux_loss_coef
                        )
                        loss = loss + post_accepted_tail_aux_contribution
                    if self.config.get(
                            'empty_cache_before_backward', False):
                        # The largest 12K responses can leave several GiB in
                        # inactive allocator blocks after the forward pass.
                        # Release only cached, non-live blocks before backward;
                        # parameters, activations, loss, and gradients are
                        # unchanged.
                        torch.cuda.empty_cache()
                    loss.backward()

                    data = {
                        'actor/entropy': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                        'actor/verified_stop_aux_loss': (
                            verified_stop_aux_loss.detach().item()),
                        'actor/verified_stop_aux_contribution': (
                            verified_stop_aux_contribution.detach().item()),
                        'actor/verified_stop_action_count': (
                            verified_stop_action_count.detach().item()),
                        'actor/overflow_stop_aux_loss': (
                            overflow_stop_aux_loss.detach().item()),
                        'actor/overflow_stop_aux_contribution': (
                            overflow_stop_aux_contribution.detach().item()),
                        'actor/overflow_stop_action_count': (
                            overflow_stop_action_count.detach().item()),
                        'actor/negative_stop_aux_loss': (
                            negative_stop_aux_loss.detach().item()),
                        'actor/negative_stop_aux_contribution': (
                            negative_stop_aux_contribution.detach().item()),
                        'actor/negative_stop_action_count': (
                            negative_stop_action_count.detach().item()),
                        'actor/negative_stop_aux_weight_sum': (
                            negative_stop_aux_weight_sum.detach().item()),
                        'actor/incorrect_stop_aux_loss': (
                            incorrect_stop_aux_loss.detach().item()),
                        'actor/incorrect_stop_aux_contribution': (
                            incorrect_stop_aux_contribution.detach().item()),
                        'actor/incorrect_stop_action_count': (
                            incorrect_stop_action_count.detach().item()),
                        'actor/incorrect_stop_aux_weight_sum': (
                            incorrect_stop_aux_weight_sum.detach().item()),
                        'actor/post_accepted_tail_aux_loss': (
                            post_accepted_tail_aux_loss.detach().item()),
                        'actor/post_accepted_tail_aux_contribution': (
                            post_accepted_tail_aux_contribution.detach().item()),
                        'actor/post_accepted_tail_action_count': (
                            post_accepted_tail_action_count.detach().item()),
                        'actor/post_accepted_tail_aux_weight_sum': (
                            post_accepted_tail_aux_weight_sum.detach().item()),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
