#!/usr/bin/env python3
"""CPU-only regression tests for the stop trainer/classifier actor mask."""

import os
import json
import pickle
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from verl import DataProto
from verl.src.stop_ray_trainer import (
    AdvantageEstimator,
    StopClassifierState,
    STOP_CLASSIFIER_SCHEMA_VERSION,
    STOP_CLASSIFIER_TARGET,
    STOP_CLASSIFIER_TARGET_FINAL,
    STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL,
    append_rollout_stop_audit,
    center_stop_action_weights_by_prompt_in_place,
    compute_advantage,
    compute_probe_metrics,
    compute_reward_component_metrics,
    compute_response_mask,
    maybe_update_stop_classifier,
    dedup_by_X,
)
from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import (
    _compute_overflow_stop_policy_loss,
    _compute_verified_stop_policy_loss,
    _compute_weighted_negative_policy_loss,
    _get_response_loss_mask,
)
from verl.src.stop_vllm_rollout import (
    _incorrect_stop_action_metadata,
    _post_accepted_tail_auxiliary,
)


class _FakeData:
    def __init__(self, batch):
        self.batch = batch


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f'expected {exc_type.__name__}')


def test_prompt_group_centered_stop_action_weights_and_gradients():
    """Only mixed-label prompt groups retain centered stop-action losses."""

    row_count = 12
    response_length = 4
    verified_mask = torch.zeros(
        (row_count, response_length), dtype=torch.long)
    verified_weight = torch.zeros(
        (row_count, response_length), dtype=torch.float32)
    incorrect_mask = torch.zeros_like(verified_mask)
    incorrect_weight = torch.zeros_like(verified_weight)

    # The rows are deliberately interleaved.  The mixed group has one correct
    # and two wrong labeled actions, so y_bar=1/3 regardless of row order or a
    # third rollout that contains no recognized stop action.
    verified_mask[0, 0] = 1
    verified_weight[0, 0] = 0.9
    incorrect_mask[4, 1:3] = 1
    incorrect_weight[4, 1] = 0.6
    incorrect_weight[4, 2] = 0.3

    # A three-action all-correct group and a four-action all-wrong group must
    # both become completely neutral after centering.
    verified_mask[1, 0] = 1
    verified_weight[1, 0] = 0.8
    verified_mask[5, 3] = 1
    verified_weight[5, 3] = 0.4
    verified_mask[9, 2] = 1
    verified_weight[9, 2] = 0.2
    incorrect_mask[2, [0, 3]] = 1
    incorrect_weight[2, 0] = 1.0
    incorrect_weight[2, 3] = 0.5
    incorrect_mask[6, 1] = 1
    incorrect_weight[6, 1] = 0.25
    incorrect_mask[10, 2] = 1
    incorrect_weight[10, 2] = 0.75

    ordinary_advantages = torch.arange(
        row_count * response_length, dtype=torch.float32
    ).reshape(row_count, response_length)
    tail_weight = torch.full_like(verified_weight, 1.0 / 256.0)
    untouched_advantages = ordinary_advantages.clone()
    untouched_tail_weight = tail_weight.clone()
    proto = DataProto.from_dict(
        tensors={
            'verified_stop_mask': verified_mask,
            'verified_stop_aux_weight': verified_weight,
            'incorrect_stop_aux_mask': incorrect_mask,
            'incorrect_stop_aux_weight': incorrect_weight,
            'advantages': ordinary_advantages,
            'post_accepted_tail_aux_weight': tail_weight,
        },
        non_tensors={
            'uid': np.asarray(
                [
                    'mixed', 'all-correct', 'all-wrong', 'empty',
                    'mixed', 'all-correct', 'all-wrong', 'empty',
                    'mixed', 'all-correct', 'all-wrong', 'empty',
                ],
                dtype=object,
            ),
        },
    )

    centered_metrics = center_stop_action_weights_by_prompt_in_place(
        proto,
        verified_stop_aux_loss_coef=0.25,
        incorrect_stop_aux_loss_coef=0.25,
        expected_group_size=3,
    )
    centered_verified_mask = proto.batch['verified_stop_mask']
    centered_verified_weight = proto.batch['verified_stop_aux_weight']
    centered_incorrect_mask = proto.batch['incorrect_stop_aux_mask']
    centered_incorrect_weight = proto.batch['incorrect_stop_aux_weight']

    assert centered_verified_mask[0, 0] == 1
    assert torch.allclose(
        centered_verified_weight[0, 0], torch.tensor(0.6))
    assert torch.equal(
        centered_incorrect_mask[4], torch.tensor([0, 1, 1, 0]))
    assert torch.allclose(
        centered_incorrect_weight[4],
        torch.tensor([0.0, 0.2, 0.1, 0.0]),
    )
    assert torch.count_nonzero(centered_verified_mask[[1, 5, 9]]).item() == 0
    assert torch.count_nonzero(centered_verified_weight[[1, 5, 9]]).item() == 0
    assert torch.count_nonzero(centered_incorrect_mask[[2, 6, 10]]).item() == 0
    assert torch.count_nonzero(centered_incorrect_weight[[2, 6, 10]]).item() == 0
    assert torch.count_nonzero(centered_verified_mask[[3, 7, 11]]).item() == 0
    assert torch.count_nonzero(centered_incorrect_mask[[3, 7, 11]]).item() == 0
    assert torch.equal(proto.batch['advantages'], untouched_advantages)
    assert torch.equal(
        proto.batch['post_accepted_tail_aux_weight'], untouched_tail_weight)
    assert centered_metrics['stop_centering/group_count'] == 4.0
    assert centered_metrics['stop_centering/mixed_group_count'] == 1.0
    assert centered_metrics['stop_centering/all_correct_group_count'] == 1.0
    assert centered_metrics['stop_centering/all_wrong_group_count'] == 1.0
    assert centered_metrics['stop_centering/empty_group_count'] == 1.0
    assert centered_metrics['stop_centering/positive_action_count'] == 1.0
    assert centered_metrics['stop_centering/negative_action_count'] == 2.0
    assert centered_metrics['stop_centering/zero_action_count'] == 7.0
    assert abs(
        centered_metrics['stop_centering/signed_weight_mass'] - 0.3
    ) < 1e-6
    assert abs(
        centered_metrics['stop_centering/absolute_weight_mass'] - 0.9
    ) < 1e-6

    # The existing actor helpers consume the centered magnitudes with their
    # original signs: correct actions increase probability, wrong actions
    # decrease it, and degenerate groups have exactly zero stop gradient.
    old_log_prob = torch.zeros((row_count, response_length))
    log_prob = torch.zeros_like(old_log_prob, requires_grad=True)
    physical_mask = torch.ones_like(old_log_prob)
    verified_loss, verified_count = _compute_verified_stop_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        response_mask=physical_mask,
        verified_stop_mask=centered_verified_mask.float(),
        verified_stop_aux_weight=centered_verified_weight,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )
    incorrect_loss, incorrect_count, _ = (
        _compute_weighted_negative_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=physical_mask,
            action_mask=centered_incorrect_mask.float(),
            action_weight=centered_incorrect_weight,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            objective_name='incorrect_stop_aux',
            max_row_weight_sum=8.0,
        )
    )
    total_loss = (
        verified_loss * verified_count / row_count * 0.25
        + incorrect_loss * incorrect_count / row_count * 0.25
    )
    total_loss.backward()
    assert float(verified_count) == 1.0
    assert float(incorrect_count) == 2.0
    assert log_prob.grad[0, 0] < 0
    assert log_prob.grad[4, 1] > 0
    assert log_prob.grad[4, 2] > 0
    assert torch.count_nonzero(log_prob.grad).item() == 3


def test_prompt_group_centering_fails_closed():
    def make_proto(uids):
        return DataProto.from_dict(
            tensors={
                'verified_stop_mask': torch.tensor([[1, 0], [0, 0]]),
                'verified_stop_aux_weight': torch.tensor(
                    [[0.8, 0.0], [0.0, 0.0]]),
                'incorrect_stop_aux_mask': torch.tensor([[0, 0], [0, 1]]),
                'incorrect_stop_aux_weight': torch.tensor(
                    [[0.0, 0.0], [0.0, 0.6]]),
            },
            non_tensors={'uid': np.asarray(uids, dtype=object)},
        )

    _assert_raises(
        ValueError,
        lambda: center_stop_action_weights_by_prompt_in_place(
            make_proto(['p', 'p']),
            verified_stop_aux_loss_coef=0.25,
            incorrect_stop_aux_loss_coef=0.5,
            expected_group_size=2,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: center_stop_action_weights_by_prompt_in_place(
            make_proto(['p', 'p']),
            verified_stop_aux_loss_coef=0.0,
            incorrect_stop_aux_loss_coef=0.0,
            expected_group_size=2,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: center_stop_action_weights_by_prompt_in_place(
            make_proto(['p', 'p']),
            verified_stop_aux_loss_coef=0.25,
            incorrect_stop_aux_loss_coef=0.25,
            expected_group_size=3,
        ),
    )
    for invalid_uid in (None, ''):
        _assert_raises(
            ValueError,
            lambda invalid_uid=invalid_uid: (
                center_stop_action_weights_by_prompt_in_place(
                    make_proto(['valid', invalid_uid]),
                    verified_stop_aux_loss_coef=0.25,
                    incorrect_stop_aux_loss_coef=0.25,
                    expected_group_size=1,
                )
            ),
        )


def test_actor_loss_mask_selection_and_gradient():
    responses = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    attention_mask = torch.ones((2, 7), dtype=torch.long)
    actor_loss_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
    batch = {
        'responses': responses,
        'attention_mask': attention_mask,
        'actor_loss_mask': actor_loss_mask,
    }

    assert torch.equal(compute_response_mask(_FakeData(batch)), actor_loss_mask)
    assert torch.equal(_get_response_loss_mask(batch), actor_loss_mask)
    fallback_batch = {'responses': responses, 'attention_mask': attention_mask}
    assert torch.equal(
        compute_response_mask(_FakeData(fallback_batch)), attention_mask[:, -4:])
    assert torch.equal(_get_response_loss_mask(fallback_batch), attention_mask[:, -4:])

    malformed = dict(batch)
    malformed['actor_loss_mask'] = torch.ones((2, 3), dtype=torch.long)
    _assert_raises(ValueError, lambda: compute_response_mask(_FakeData(malformed)))
    _assert_raises(ValueError, lambda: _get_response_loss_mask(malformed))

    # Exercise both ordinary TensorDict splitting and the dynamic-batch
    # rearranger: the selected mask must follow each example in both paths.
    proto = DataProto.from_single_dict({
        'responses': responses,
        'input_ids': torch.tensor([[1, 2, 3, 11, 12, 13, 14],
                                   [1, 2, 3, 21, 22, 23, 24]]),
        'attention_mask': attention_mask,
        'position_ids': torch.arange(7).repeat(2, 1),
        'old_log_probs': torch.zeros((2, 4)),
        'advantages': torch.ones((2, 4)),
        'actor_loss_mask': actor_loss_mask,
        'negative_stop_aux_mask': torch.tensor(
            [[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.long),
        'negative_stop_aux_weight': torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        'post_accepted_tail_aux_mask': torch.tensor(
            [[0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.long),
        'post_accepted_tail_aux_weight': torch.tensor(
            [[0.0, 0.0, 0.25, 0.0], [0.0, 0.5, 0.0, 0.0]]),
    })
    select_keys = [
        'responses', 'input_ids', 'attention_mask', 'position_ids',
        'old_log_probs', 'advantages', 'actor_loss_mask',
        'negative_stop_aux_mask', 'negative_stop_aux_weight',
        'post_accepted_tail_aux_mask', 'post_accepted_tail_aux_weight',
    ]
    selected = proto.select(batch_keys=select_keys).batch
    ordinary = selected.split(1)
    assert all(
        all(key in micro.keys() for key in select_keys)
        for micro in ordinary
    )
    dynamic, _ = rearrange_micro_batches(selected, max_token_len=100)
    assert all(
        all(key in micro.keys() for key in select_keys)
        for micro in dynamic
    )

    # The actual PPO policy loss must have exactly zero gradient on a forced
    # environment suffix when actor_loss_mask is zero there.
    old_log_prob = torch.zeros((1, 4))
    log_prob = torch.full((1, 4), -0.1, requires_grad=True)
    advantages = torch.ones((1, 4))
    policy_mask = torch.tensor([[1, 1, 0, 0]])
    pg_loss, *_ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=policy_mask,
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    pg_loss.backward()
    assert torch.all(log_prob.grad[:, :2] != 0)
    assert torch.equal(log_prob.grad[:, 2:], torch.zeros_like(log_prob.grad[:, 2:]))


def test_weighted_negative_action_loss_has_correct_sign_and_bounded_strength():
    old_log_prob = torch.zeros((2, 4))
    log_prob = torch.zeros((2, 4), requires_grad=True)
    physical_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)
    action_mask = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.float32)
    action_weight = torch.tensor(
        [[0.0, 0.25, 0.75, 0.0], [0.5, 0.0, 0.0, 0.0]])

    aux_loss, action_count, weight_sum = (
        _compute_weighted_negative_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=physical_mask,
            action_mask=action_mask,
            action_weight=action_weight,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            objective_name='post_accepted_tail_aux',
            max_row_weight_sum=1.0,
        )
    )
    assert float(action_count) == 3.0
    assert torch.isclose(weight_sum, torch.tensor(1.5))
    # At ratio=1, undoing compute_policy_loss's action mean exactly recovers
    # the sum of bounded per-action weights.  Averaging over two rollout rows
    # therefore cannot exceed the configured coefficient.
    assert torch.isclose(aux_loss * action_count, weight_sum)
    coefficient = 0.25
    contribution = aux_loss * action_count / 2.0 * coefficient
    assert 0.0 < float(contribution) <= coefficient
    contribution.backward()
    assert log_prob.grad[0, 1] > 0
    assert log_prob.grad[0, 2] > log_prob.grad[0, 1] > 0
    assert log_prob.grad[1, 0] > 0
    assert torch.count_nonzero(log_prob.grad).item() == 3

    def call_with(mask, weight, response_mask=physical_mask):
        return _compute_weighted_negative_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=torch.zeros_like(old_log_prob),
            response_mask=response_mask,
            action_mask=mask,
            action_weight=weight,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            objective_name='negative_stop_aux',
            max_row_weight_sum=1.0,
        )

    over_bound_weight = action_weight.clone()
    over_bound_weight[0, 1] = 0.5
    _assert_raises(
        ValueError, lambda: call_with(action_mask, over_bound_weight))
    outside_weight = action_weight.clone()
    outside_weight[0, 0] = 0.1
    _assert_raises(ValueError, lambda: call_with(action_mask, outside_weight))
    zero_selected_weight = action_weight.clone()
    zero_selected_weight[0, 1] = 0.0
    _assert_raises(
        ValueError, lambda: call_with(action_mask, zero_selected_weight))
    non_finite_weight = action_weight.clone()
    non_finite_weight[0, 1] = float('nan')
    _assert_raises(
        ValueError, lambda: call_with(action_mask, non_finite_weight))
    outside_physical_mask = action_mask.clone()
    outside_physical_mask[1, 3] = 1
    outside_physical_weight = action_weight.clone()
    outside_physical_weight[1, 3] = 0.1
    _assert_raises(
        ValueError,
        lambda: call_with(outside_physical_mask, outside_physical_weight),
    )

    empty_mask = torch.zeros_like(action_mask)
    empty_weight = torch.zeros_like(action_weight)
    empty_log_prob = torch.zeros_like(old_log_prob, requires_grad=True)
    empty_loss, empty_count, empty_sum = (
        _compute_weighted_negative_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=empty_log_prob,
            response_mask=physical_mask,
            action_mask=empty_mask,
            action_weight=empty_weight,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
        )
    )
    empty_loss.backward()
    assert float(empty_count) == 0.0
    assert float(empty_sum) == 0.0
    assert torch.count_nonzero(empty_log_prob.grad).item() == 0


def test_incorrect_probe_routing_is_per_action_and_none_is_neutral():
    positions = [1, 3, 5, 7]
    scores = np.zeros(4, dtype=np.float64)
    probabilities = np.full(4, 0.25, dtype=np.float64)
    records = [
        ('A', scores, probabilities),
        ('C', scores, probabilities),
        (None, scores, probabilities),
        ('B', scores, probabilities),
    ]
    metadata = _incorrect_stop_action_metadata(
        positions,
        records,
        'C',
        response_length=10,
        max_stop_count=8,
        half_life_tokens=1024.0,
    )
    # The recognized wrong proposal after the earliest correct proposal is
    # still penalized.  The unrecognized proposal is neutral.
    assert metadata['incorrect_stop_positions'] == [1, 7]
    assert metadata['incorrect_stop_count'] == 2
    assert metadata['recognized_correct_probe_count'] == 1
    assert metadata['neutral_probe_stop_count'] == 1
    assert (
        metadata['recognized_correct_probe_count']
        + metadata['incorrect_stop_count']
        + metadata['neutral_probe_stop_count']
    ) == len(positions)
    assert metadata['incorrect_stop_aux_mask'] == [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    expected_weights = [
        2.0 ** (-1.0 / 1024.0),
        2.0 ** (-7.0 / 1024.0),
    ]
    assert np.allclose(
        [metadata['incorrect_stop_aux_weight'][1],
         metadata['incorrect_stop_aux_weight'][7]],
        expected_weights,
    )
    assert abs(
        metadata['incorrect_stop_aux_weight_sum'] - sum(expected_weights)
    ) < 1e-12
    assert metadata['incorrect_stop_aux_weight_sum'] > 1.0

    # After the accepted stop at position 3, the unrecognized stop at 5, the
    # recognized-wrong stop at 7, and every other atomic stop are excluded from
    # the tail objective.  Thus None remains strictly gradient-neutral.
    atomic_mask = [0] * 10
    for position in positions:
        atomic_mask[position] = 1
    tail_mask, tail_weight = _post_accepted_tail_auxiliary(
        list(range(10)),
        accepted_stop_position=3,
        think_close_ids=[9],
        negative_stop_aux_mask=[0] * 10,
        atomic_stop_mask=atomic_mask,
        incorrect_stop_aux_mask=metadata['incorrect_stop_aux_mask'],
        normalizer_tokens=256,
    )
    assert [index for index, value in enumerate(tail_mask) if value] == [4, 6, 8]
    assert tail_mask[5] == 0
    assert tail_mask[7] == 0
    assert abs(sum(tail_weight) - 3.0 / 256.0) < 1e-12


def test_incorrect_stop_loss_sign_mass_and_dynamic_pack_invariance():
    coefficient = 0.25
    weights = torch.tensor([[0.9, 0.7, 0.4]], dtype=torch.float32)
    mask = torch.ones_like(weights)
    log_prob = torch.zeros_like(weights, requires_grad=True)
    loss, count, weight_sum = _compute_weighted_negative_policy_loss(
        old_log_prob=torch.zeros_like(weights),
        log_prob=log_prob,
        response_mask=torch.ones_like(weights),
        action_mask=mask,
        action_weight=weights,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        objective_name='incorrect_stop_aux',
        max_row_weight_sum=8.0,
    )
    contribution = loss * count * coefficient
    contribution.backward()
    assert float(count) == 3.0
    assert torch.isclose(weight_sum, torch.tensor(2.0))
    assert torch.allclose(
        log_prob.grad,
        torch.tensor([[0.225, 0.175, 0.1]]),
    )
    assert torch.isclose(log_prob.grad.sum(), torch.tensor(0.5))

    # Adding the third wrong stop must not renormalize or weaken the first two.
    first_two_log_prob = torch.zeros((1, 2), requires_grad=True)
    first_two_loss, first_two_count, _ = _compute_weighted_negative_policy_loss(
        old_log_prob=torch.zeros((1, 2)),
        log_prob=first_two_log_prob,
        response_mask=torch.ones((1, 2)),
        action_mask=torch.ones((1, 2)),
        action_weight=weights[:, :2],
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        objective_name='incorrect_stop_aux',
        max_row_weight_sum=8.0,
    )
    (first_two_loss * first_two_count * coefficient).backward()
    assert torch.allclose(first_two_log_prob.grad, log_prob.grad[:, :2])

    # The chosen stop probability must fall after an actual optimizer step;
    # this catches a scalar-loss sign inversion that a loss-value assertion
    # alone would miss.
    logits = torch.zeros((1, 2), requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.1)
    selected_log_prob = torch.log_softmax(logits, dim=-1)[:, :1]
    selected_old_log_prob = selected_log_prob.detach().clone()
    probability_before = float(torch.softmax(logits.detach(), dim=-1)[0, 0])
    categorical_loss, categorical_count, _ = (
        _compute_weighted_negative_policy_loss(
            old_log_prob=selected_old_log_prob,
            log_prob=selected_log_prob,
            response_mask=torch.ones_like(selected_log_prob),
            action_mask=torch.ones_like(selected_log_prob),
            action_weight=torch.tensor([[0.8]]),
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            objective_name='incorrect_stop_aux',
            max_row_weight_sum=8.0,
        )
    )
    (categorical_loss * categorical_count * coefficient).backward()
    optimizer.step()
    probability_after = float(torch.softmax(logits.detach(), dim=-1)[0, 0])
    assert probability_after < probability_before

    # Dynamic micro-batch splitting must not change the sum or any gradient.
    packed_weights = torch.tensor(
        [[0.9, 0.0, 0.0], [0.8, 0.6, 0.0],
         [0.7, 0.5, 0.3], [0.4, 0.0, 0.0]],
        dtype=torch.float32,
    )
    packed_mask = (packed_weights > 0).float()
    mini_batch_rows = packed_weights.size(0)

    def gradients_for_partitions(partitions):
        packed_log_prob = torch.zeros_like(
            packed_weights, requires_grad=True)
        total = packed_log_prob.sum() * 0.0
        for row_indices in partitions:
            rows = torch.as_tensor(row_indices, dtype=torch.long)
            micro_loss, micro_count, _ = (
                _compute_weighted_negative_policy_loss(
                    old_log_prob=torch.zeros_like(packed_weights[rows]),
                    log_prob=packed_log_prob[rows],
                    response_mask=torch.ones_like(packed_weights[rows]),
                    action_mask=packed_mask[rows],
                    action_weight=packed_weights[rows],
                    clip_ratio=0.2,
                    clip_ratio_low=0.2,
                    clip_ratio_high=0.2,
                    clip_ratio_c=3.0,
                    objective_name='incorrect_stop_aux',
                    max_row_weight_sum=8.0,
                )
            )
            total = total + (
                micro_loss * micro_count / mini_batch_rows * coefficient
            )
        total.backward()
        return total.detach(), packed_log_prob.grad.detach()

    full_loss, full_grad = gradients_for_partitions([[0, 1, 2, 3]])
    split_13_loss, split_13_grad = gradients_for_partitions([[0], [1, 2, 3]])
    split_22_loss, split_22_grad = gradients_for_partitions([[0, 1], [2, 3]])
    assert torch.allclose(full_loss, split_13_loss)
    assert torch.allclose(full_loss, split_22_loss)
    assert torch.allclose(full_grad, split_13_grad)
    assert torch.allclose(full_grad, split_22_grad)


def test_action_isolated_verified_and_overflow_stop_losses_are_audited():
    incorrect_weight = 2.0 ** (-1.0 / 1024.0)
    tensors = {
        'responses': torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]]),
        'attention_mask': torch.ones((2, 7), dtype=torch.long),
        'response_mask': torch.tensor(
            [[1, 0, 0, 1], [1, 0, 1, 0]], dtype=torch.long),
        'token_level_rewards': torch.zeros((2, 4), dtype=torch.float32),
        'token_level_scores': torch.zeros((2, 4), dtype=torch.float32),
        'atomic_stop_mask': torch.tensor(
            [[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.long),
        'verified_stop_mask': torch.tensor(
            [[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.long),
        'verified_stop_aux_weight': torch.tensor(
            [[0, 0.75, 0, 0], [0, 0, 0, 0]], dtype=torch.float32),
        'overflow_stop_mask': torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.long),
        'raw_atomic_stop_count': torch.tensor([2, 2]),
        'illegal_surface_stop_count': torch.tensor([0, 0]),
        'ordinary_surface_stop_count': torch.tensor([0, 0]),
        'stop_fragment_count': torch.tensor([0, 0]),
        'overflow_atomic_stop_count': torch.tensor([1, 1]),
        'negative_stop_event_count': torch.tensor([1, 1]),
        'stop_event_multiplicity': torch.tensor(
            [[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.long),
        'surface_stop_event_multiplicity': torch.zeros(
            (2, 4), dtype=torch.long),
        'negative_stop_aux_mask': torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.long),
        'negative_stop_aux_weight': torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        'incorrect_stop_count': torch.tensor([0, 1]),
        'incorrect_stop_aux_mask': torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.long),
        'incorrect_stop_aux_weight': torch.tensor(
            [[0.0, 0.0, 0.0, 0.0],
             [0.0, incorrect_weight, 0.0, 0.0]], dtype=torch.float32),
        'post_accepted_tail_aux_mask': torch.tensor(
            [[0, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.long),
        'post_accepted_tail_aux_weight': torch.tensor(
            [[0.0, 0.0, 0.0, 1.0 / 256.0], [0.0, 0.0, 0.0, 0.0]]),
        'raw_stop_count': torch.tensor([2, 2]),
        'early_stop_score': torch.tensor([0.75, 0.0]),
        'verified_stop_count': torch.tensor([1, 0]),
        'verified_stop_score': torch.tensor([0.75, 0.0]),
        'eligible_stop_count': torch.tensor([2, 2]),
        'probed_stop_count': torch.tensor([1, 1]),
        'recognized_correct_probe_count': torch.tensor([1, 0]),
        'neutral_probe_stop_count': torch.tensor([0, 0]),
        'verified_stop_aux_suppressed': torch.tensor([0, 0]),
        'verified_stop_position': torch.tensor([1, -1]),
        'oracle_accepted_stop_position': torch.tensor([1, -1]),
        'accepted_stop_prefix_length': torch.tensor([4, 4]),
        'tail_cut_tokens': torch.tensor([0, 0]),
        'accepted_stop_tail_cut_enabled': torch.tensor([0, 0]),
        'post_accepted_active_token_count': torch.tensor([1, 0]),
        'rollout_index': torch.tensor([0, 1]),
    }
    recognized_correct_positions = np.empty(2, dtype=object)
    recognized_correct_positions[:] = [[1], []]
    incorrect_positions = np.empty(2, dtype=object)
    incorrect_positions[:] = [[], [1]]
    neutral_positions = np.empty(2, dtype=object)
    neutral_positions[:] = [[], []]
    proto = DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            'uid': np.asarray(['same-prompt', 'same-prompt'], dtype=object),
            'recognized_correct_probe_positions': recognized_correct_positions,
            'incorrect_stop_positions': incorrect_positions,
            'neutral_probe_stop_positions': neutral_positions,
            'strict_final_valid': np.asarray([1, 0]),
            'format_score': np.asarray([1, 0]),
            'stop_presence_format_score': np.asarray([1, 1]),
            'structural_format_score': np.asarray([1, 0]),
            'format_reward': np.asarray([0.25, 0.0]),
            'accuracy_reward': np.asarray([0.5, 0.0]),
            'stop_earliness': np.asarray([0.75, 0.0]),
            'stop_reward': np.asarray([0.1875, 0.0]),
            'reward_gated': np.asarray([0, 1]),
            'max_stop_count': np.asarray([1, 1]),
            'stop_reward_half_life_tokens': np.asarray([1024.0, 1024.0]),
        },
    )
    proto = compute_advantage(
        proto,
        adv_estimator=AdvantageEstimator.GRPO,
        verified_stop_aux_loss_coef=0.25,
        overflow_stop_aux_loss_coef=0.0,
        negative_stop_aux_loss_coef=1.0,
        incorrect_stop_aux_loss_coef=0.25,
        post_accepted_tail_aux_loss_coef=0.25,
    )
    assert torch.equal(
        proto.batch['advantages'], torch.zeros_like(tensors['token_level_rewards']))
    assert torch.equal(
        proto.batch['returns'], torch.zeros_like(tensors['token_level_rewards']))
    assert torch.equal(
        proto.batch['verified_stop_aux_bonus'],
        tensors['verified_stop_aux_weight'] * 0.25,
    )
    assert torch.equal(
        proto.batch['overflow_stop_aux_penalty'],
        torch.zeros_like(tensors['overflow_stop_mask'].float()),
    )
    assert torch.equal(
        proto.batch['negative_stop_aux_penalty'],
        tensors['negative_stop_aux_weight'],
    )
    assert torch.equal(
        proto.batch['incorrect_stop_aux_penalty'],
        tensors['incorrect_stop_aux_weight'] * 0.25,
    )
    assert torch.equal(
        proto.batch['post_accepted_tail_aux_penalty'],
        tensors['post_accepted_tail_aux_weight'] * 0.25,
    )

    # Ordinary GRPO excludes every atomic stop.  Non-stop tail tokens remain
    # active for accuracy/format training.
    old_log_prob = torch.zeros((2, 4))
    log_prob = torch.full((2, 4), -0.1, requires_grad=True)
    policy_loss, *_ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=torch.ones((2, 4)),
        response_mask=tensors['response_mask'],
        cliprange=0.2,
        cliprange_low=0.2,
        cliprange_high=0.2,
        clip_ratio_c=3.0,
    )
    policy_loss.backward()
    assert log_prob.grad[0, 1] == 0  # earliest verified stop is action-only
    assert log_prob.grad[0, 2] == 0  # overflow stop is action-only
    assert log_prob.grad[1, 1] == 0  # failed legal stop
    assert log_prob.grad[1, 3] == 0  # overflow stop is action-only
    assert log_prob.grad[0, 3] != 0  # post-stop answer tail remains active

    # The verified action receives only its positive, earliness-weighted PPO
    # loss.  The gradient is localized to that single action.
    log_prob = torch.zeros((2, 4), requires_grad=True)
    verified_loss, verified_count = _compute_verified_stop_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        response_mask=torch.ones_like(tensors['response_mask']),
        verified_stop_mask=tensors['verified_stop_mask'].float(),
        verified_stop_aux_weight=tensors['verified_stop_aux_weight'],
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )
    verified_contribution = verified_loss * verified_count / 2.0 * 0.25
    verified_contribution.backward()
    assert float(verified_count) == 1.0
    assert log_prob.grad[0, 1] < 0
    assert torch.count_nonzero(log_prob.grad).item() == 1

    # Both overflow stops receive the length-independent negative auxiliary
    # loss, including the one sharing a rollout with the verified stop.
    log_prob = torch.zeros((2, 4), requires_grad=True)
    overflow_mask = tensors['overflow_stop_mask'].float()
    overflow_loss, overflow_count = _compute_overflow_stop_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        response_mask=torch.ones_like(tensors['response_mask']),
        overflow_stop_mask=overflow_mask,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )
    contribution = overflow_loss * overflow_count / 2.0
    contribution.backward()
    assert float(overflow_count) == 2.0
    assert log_prob.grad[0, 2] > 0
    assert log_prob.grad[1, 3] > 0
    assert torch.count_nonzero(log_prob.grad).item() == 2

    hard_gated_tensors = dict(tensors)
    hard_gated_tensors['verified_stop_mask'] = tensors[
        'verified_stop_mask'].clone()
    hard_gated_tensors['verified_stop_aux_weight'] = tensors[
        'verified_stop_aux_weight'].clone()
    _assert_raises(
        ValueError,
        lambda: compute_advantage(
            DataProto.from_dict(
                tensors=hard_gated_tensors,
                non_tensors={
                    'uid': np.asarray(
                        ['same-prompt', 'same-prompt'], dtype=object),
                    'reward_gated': np.asarray([1, 1]),
                    'stop_reward': np.asarray([0.0, 0.0]),
                    'max_stop_count': np.asarray([1, 1]),
                    'stop_reward_half_life_tokens': np.asarray(
                        [1024.0, 1024.0]),
                },
            ),
            adv_estimator=AdvantageEstimator.GRPO,
            verified_stop_aux_loss_coef=0.25,
            overflow_stop_aux_loss_coef=0.0,
            negative_stop_aux_loss_coef=1.0,
            incorrect_stop_aux_loss_coef=0.25,
            post_accepted_tail_aux_loss_coef=0.25,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        audit_path = os.path.join(tmp, 'rollout_stop_counts.jsonl')
        metrics = append_rollout_stop_audit(audit_path, 1, proto)
        with open(audit_path, encoding='utf-8') as audit_file:
            records = [json.loads(line) for line in audit_file]
    assert [record['raw_stop_count'] for record in records] == [2, 2]
    assert [record['rollout_index'] for record in records] == [0, 1]
    assert [record['verified_stop_action_count'] for record in records] == [1, 0]
    assert [record['verified_stop_count'] for record in records] == [1, 0]
    assert [record['verified_stop_score'] for record in records] == [0.75, 0.0]
    assert [record['verified_stop_aux_weight_sum'] for record in records] == [0.75, 0.0]
    assert [record['overflow_stop_action_count'] for record in records] == [1, 1]
    assert [record['negative_stop_event_count'] for record in records] == [1, 1]
    assert [record['negative_stop_aux_action_count'] for record in records] == [1, 1]
    assert [record['negative_stop_aux_weight_sum'] for record in records] == [1.0, 1.0]
    assert [record['negative_stop_aux_penalty_sum'] for record in records] == [1.0, 1.0]
    assert [record['incorrect_stop_count'] for record in records] == [0, 1]
    assert [
        record['recognized_correct_probe_count'] for record in records
    ] == [1, 0]
    assert [record['neutral_probe_stop_count'] for record in records] == [0, 0]
    assert [
        record['recognized_correct_probe_positions'] for record in records
    ] == [[1], []]
    assert [record['incorrect_stop_positions'] for record in records] == [[], [1]]
    assert [record['neutral_probe_stop_positions'] for record in records] == [[], []]
    assert [
        record['incorrect_stop_aux_action_count'] for record in records
    ] == [0, 1]
    assert np.allclose(
        [record['incorrect_stop_aux_weight_sum'] for record in records],
        [0.0, incorrect_weight],
    )
    assert np.allclose(
        [record['incorrect_stop_aux_penalty_sum'] for record in records],
        [0.0, incorrect_weight * 0.25],
    )
    assert [
        record['post_accepted_tail_aux_action_count'] for record in records
    ] == [1, 0]
    assert [
        record['post_accepted_tail_aux_weight_sum'] for record in records
    ] == [1.0 / 256.0, 0.0]
    assert [
        record['post_accepted_tail_aux_penalty_sum'] for record in records
    ] == [1.0 / 1024.0, 0.0]
    assert [record['base_policy_stop_action_count'] for record in records] == [0, 0]
    assert [record['accepted_stop_prefix_length'] for record in records] == [4, 4]
    assert [record['tail_cut_tokens'] for record in records] == [0, 0]
    assert [record['effective_actor_token_count'] for record in records] == [2, 2]
    assert [
        record['post_accepted_active_token_count'] for record in records
    ] == [1, 0]
    assert [record['strict_final_valid'] for record in records] == [1, 0]
    assert [record['format_score'] for record in records] == [1, 0]
    assert [
        record['stop_presence_format_score'] for record in records
    ] == [1, 1]
    assert [record['structural_format_score'] for record in records] == [1, 0]
    assert [record['format_reward'] for record in records] == [0.25, 0.0]
    assert [record['accuracy_reward'] for record in records] == [0.5, 0.0]
    assert [record['stop_earliness'] for record in records] == [0.75, 0.0]
    assert [record['stop_reward'] for record in records] == [0.1875, 0.0]
    assert all(record['step'] == 1 for record in records)
    assert metrics['reward/raw_stop_count_p50'] == 2.0
    assert metrics['reward/verified_stop_action_mean'] == 0.5
    assert metrics['reward/verified_stop_count_mean'] == 0.5
    assert metrics['reward/verified_stop_score_mean'] == 0.375
    assert metrics['reward/verified_stop_aux_weight_mean'] == 0.375
    assert metrics['reward/verified_stop_aux_bonus_mean'] == 0.09375
    assert metrics['reward/overflow_stop_aux_penalty_mean'] == 0.0
    assert metrics['reward/negative_stop_aux_penalty_mean'] == 1.0
    assert metrics['reward/incorrect_stop_count_mean'] == 0.5
    assert metrics['reward/recognized_correct_probe_count_mean'] == 0.5
    assert metrics['reward/neutral_probe_stop_count_mean'] == 0.0
    assert metrics['reward/incorrect_stop_aux_action_mean'] == 0.5
    assert abs(
        metrics['reward/incorrect_stop_aux_weight_mean']
        - incorrect_weight / 2.0
    ) < 1e-7
    assert abs(
        metrics['reward/incorrect_stop_aux_penalty_mean']
        - incorrect_weight / 8.0
    ) < 1e-7
    assert metrics['reward/post_accepted_tail_aux_weight_mean'] == 1.0 / 512.0
    assert metrics['reward/post_accepted_tail_aux_penalty_mean'] == 1.0 / 2048.0
    assert metrics['reward/tail_cut_row_rate'] == 0.0
    assert metrics['reward/tail_cut_tokens_mean'] == 0.0


def _multi_overflow_proto(*, omit_last_overflow=False):
    """Build two cap-eight rollouts containing twelve atomic stop actions."""

    response_length = 13
    atomic_stop_mask = torch.tensor(
        [[0] + [1] * 12, [0] + [1] * 12], dtype=torch.long)
    overflow_stop_mask = torch.tensor(
        [[0] + [0] * 8 + [1] * 4, [0] + [0] * 8 + [1] * 4],
        dtype=torch.long,
    )
    if omit_last_overflow:
        overflow_stop_mask[0, -1] = 0
    verified_stop_mask = torch.zeros((2, response_length), dtype=torch.long)
    verified_stop_mask[0, 1] = 1
    verified_stop_aux_weight = torch.zeros(
        (2, response_length), dtype=torch.float32)
    verified_stop_aux_weight[0, 1] = 1.0
    token_level_rewards = torch.zeros(
        (2, response_length), dtype=torch.float32)
    token_level_rewards[:, -1] = torch.tensor([0.75, 0.25])

    return DataProto.from_dict(
        tensors={
            'responses': torch.arange(response_length).repeat(2, 1),
            'attention_mask': torch.ones(
                (2, response_length + 3), dtype=torch.long),
            'response_mask': torch.tensor(
                [[1] + [0] * 12, [1] + [0] * 12], dtype=torch.long),
            'token_level_rewards': token_level_rewards,
            'atomic_stop_mask': atomic_stop_mask,
            'verified_stop_mask': verified_stop_mask,
            'verified_stop_aux_weight': verified_stop_aux_weight,
            'overflow_stop_mask': overflow_stop_mask,
            'raw_stop_count': torch.tensor([12, 12]),
        },
        non_tensors={
            'uid': np.asarray(['same-prompt', 'same-prompt'], dtype=object),
            'reward_gated': np.asarray([0, 0]),
            'stop_reward': np.asarray([0.25, 0.0]),
            'max_stop_count': np.asarray([8, 8]),
        },
    )


def test_all_four_overflow_actions_per_rollout_are_penalized():
    proto = compute_advantage(
        _multi_overflow_proto(),
        adv_estimator=AdvantageEstimator.GRPO,
        verified_stop_aux_loss_coef=0.25,
        overflow_stop_aux_loss_coef=1.0,
    )
    assert torch.equal(
        proto.batch['overflow_stop_aux_penalty'].sum(dim=-1),
        torch.tensor([4.0, 4.0]),
    )
    assert torch.equal(
        proto.batch['verified_stop_aux_bonus'].sum(dim=-1),
        torch.tensor([0.25, 0.0]),
    )
    assert torch.count_nonzero(
        proto.batch['verified_stop_mask']
        & proto.batch['overflow_stop_mask']
    ).item() == 0

    old_log_prob = torch.zeros((1, 13))
    log_prob = torch.zeros((1, 13), requires_grad=True)
    overflow_mask = proto.batch['overflow_stop_mask'][:1].float()
    overflow_loss, overflow_count = _compute_overflow_stop_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        response_mask=torch.ones_like(old_log_prob),
        overflow_stop_mask=overflow_mask,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )
    (overflow_loss * overflow_count).backward()
    self_grad = log_prob.grad[0]
    assert float(overflow_count) == 4.0
    assert torch.all(self_grad[9:13] > 0)
    assert torch.count_nonzero(self_grad).item() == 4


def test_missing_any_post_budget_stop_penalty_fails_closed():
    _assert_raises(
        ValueError,
        lambda: compute_advantage(
            _multi_overflow_proto(omit_last_overflow=True),
            adv_estimator=AdvantageEstimator.GRPO,
            verified_stop_aux_loss_coef=0.25,
            overflow_stop_aux_loss_coef=1.0,
        ),
    )


def _object_payload(X, y):
    feature_x = np.empty(1, dtype=object)
    feature_y = np.empty(1, dtype=object)
    feature_x[0] = X
    feature_y[0] = y
    return {'feature_x': feature_x, 'feature_y': feature_y}


def test_classifier_online_fit_is_disabled_without_side_effects():
    params = {
        'objective': 'binary',
        'n_estimators': 2,
        'random_state': 7,
        'n_jobs': 1,
        'verbosity': -1,
    }
    state = StopClassifierState(params)
    X = np.zeros((2, 30), dtype=np.float32)
    X[:, 0] = [-1.0, 1.0]
    payload = _object_payload(X, np.asarray([0, 1], dtype=np.int64))

    metrics = maybe_update_stop_classifier(state, payload, enabled=False)
    assert metrics['classifier/online_training_enabled'] == 0
    assert metrics['classifier/updated'] == 0
    assert metrics['classifier/cumulative_samples'] == 0
    assert len(state.feature_y) == 0
    assert state.model is None
    assert state.update_count == 0
    _assert_raises(
        TypeError,
        lambda: maybe_update_stop_classifier(state, payload, enabled='false'),
    )


def test_classifier_collect_train_and_checkpoint():
    params = {
        'objective': 'binary',
        'n_estimators': 30,
        'learning_rate': 0.2,
        'num_leaves': 7,
        'min_child_samples': 1,
        'min_data_in_bin': 1,
        'random_state': 7,
        'n_jobs': 1,
        'verbosity': -1,
    }
    state = StopClassifierState(params)

    empty_metrics = state.update({})
    assert empty_metrics['classifier/cumulative_samples'] == 0
    assert empty_metrics['classifier/updated'] == 0
    assert not state.is_trained

    X0 = np.zeros((12, 30), dtype=np.float32)
    X0[:, 0] = np.linspace(-2.0, -0.1, 12)
    y0 = np.zeros(12, dtype=np.int64)
    one_class_metrics = state.update(_object_payload(X0, y0))
    assert one_class_metrics['classifier/class_0_samples'] == 12
    assert one_class_metrics['classifier/class_1_samples'] == 0
    assert one_class_metrics['classifier/updated'] == 0
    assert not state.is_trained
    _assert_raises(RuntimeError, lambda: state.predict(X0[:1]))

    X1 = np.zeros((13, 30), dtype=np.float32)
    X1[:12, 0] = np.linspace(0.1, 2.0, 12)
    X1[12] = X1[0]  # duplicate is audited but remains a physical record
    y1 = np.ones(13, dtype=np.int64)
    train_metrics = state.update(_object_payload(X1, y1))
    assert train_metrics['classifier/batch_samples_raw'] == 13
    assert train_metrics['classifier/batch_unique_feature_rows'] == 12
    assert train_metrics['classifier/batch_duplicate_rows_observed'] == 1
    assert train_metrics['classifier/cumulative_samples'] == 25
    assert train_metrics['classifier/class_0_samples'] == 12
    assert train_metrics['classifier/class_1_samples'] == 13
    assert train_metrics['classifier/updated'] == 1
    assert state.is_trained
    assert train_metrics['classifier/prequential_rows'] == 0

    probes = np.zeros((2, 30), dtype=np.float32)
    probes[:, 0] = [-1.0, 1.0]
    before = state.predict(probes)
    assert before.tolist() == [0, 1]
    prequential_metrics = state.update(
        _object_payload(probes, np.asarray([0, 1], dtype=np.int64))
    )
    assert prequential_metrics['classifier/prequential_rows'] == 2
    assert 0.0 <= prequential_metrics['classifier/prequential_auc'] <= 1.0
    assert 0.0 <= prequential_metrics['classifier/prequential_brier'] <= 1.0
    update_count = state.update_count
    trained_empty_metrics = state.update({})
    assert trained_empty_metrics['classifier/updated'] == 0
    assert trained_empty_metrics['classifier/trained'] == 1
    assert state.update_count == update_count

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, 'stop_classifier.pkl')
        state.save(path)
        restored = StopClassifierState({'objective': 'binary'})
        restored.load(path)
        after = restored.predict(probes)
        assert np.array_equal(before, after)
        assert restored.update_count == state.update_count
        assert np.array_equal(restored.feature_x, state.feature_x)
        assert np.array_equal(restored.feature_y, state.feature_y)
        with open(path, 'rb') as checkpoint:
            payload = pickle.load(checkpoint)
        assert payload['schema_version'] == STOP_CLASSIFIER_SCHEMA_VERSION
        assert payload['target'] == STOP_CLASSIFIER_TARGET
        assert payload['source_records'] == 'all_physical_records_by_candidate'


def test_dual_classifier_states_use_independent_physical_keys_and_targets():
    params = {
        'objective': 'binary',
        'n_estimators': 2,
        'random_state': 7,
        'n_jobs': 1,
        'verbosity': -1,
    }
    final_state = StopClassifierState(
        params,
        target=STOP_CLASSIFIER_TARGET_FINAL,
        key_x='classifier_final_feature_x',
        key_y='classifier_final_feature_y',
        metric_prefix='classifier_final',
    )
    union_state = StopClassifierState(
        params,
        target=STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL,
        key_x='classifier_gold_or_final_feature_x',
        key_y='classifier_gold_or_final_feature_y',
        metric_prefix='classifier_gold_or_final',
    )
    X = np.zeros((3, 30), dtype=np.float32)
    X[:, 0] = [-1.0, 0.0, 1.0]
    payload = {}
    for prefix, labels in (
        ('classifier_final', [0, 0, 1]),
        ('classifier_gold_or_final', [1, 0, 1]),
    ):
        feature_x = np.empty(1, dtype=object)
        feature_y = np.empty(1, dtype=object)
        feature_x[0] = X
        feature_y[0] = np.asarray(labels, dtype=np.int64)
        payload[f'{prefix}_feature_x'] = feature_x
        payload[f'{prefix}_feature_y'] = feature_y

    final_metrics = maybe_update_stop_classifier(final_state, payload, enabled=True)
    union_metrics = maybe_update_stop_classifier(union_state, payload, enabled=True)
    final_labels_by_x = dict(zip(final_state.feature_x[:, 0], final_state.feature_y))
    union_labels_by_x = dict(zip(union_state.feature_x[:, 0], union_state.feature_y))
    assert final_labels_by_x == {-1.0: 0, 0.0: 0, 1.0: 1}
    assert union_labels_by_x == {-1.0: 1, 0.0: 0, 1.0: 1}
    assert final_metrics['classifier_final/source_is_all_physical_records'] == 1
    assert union_metrics['classifier_gold_or_final/source_is_all_physical_records'] == 1
    assert not set(final_metrics).intersection(union_metrics)

    with tempfile.TemporaryDirectory() as tmp_dir:
        final_path = os.path.join(tmp_dir, 'final.pkl')
        union_path = os.path.join(tmp_dir, 'union.pkl')
        final_state.save(final_path)
        union_state.save(union_path)
        _assert_raises(ValueError, lambda: union_state.load(final_path))
        _assert_raises(ValueError, lambda: final_state.load(union_path))


def test_classifier_dedup_majority_and_tie_contract():
    X = np.zeros((6, 30), dtype=np.float32)
    X[3:, 0] = 1.0
    y = np.asarray([0, 1, 1, 1, 0, 0], dtype=np.int64)
    X_unique, y_unique, info = dedup_by_X(X, y, strategy='majority')
    assert X_unique.shape == (2, 30)
    assert y_unique.tolist() == [1, 0]
    assert info['duplicates_removed'] == 4
    assert info['clash_groups'] == 2


def test_probe_metrics_with_empty_and_variable_rows():
    probe_answers = np.empty(3, dtype=object)
    probe_answers[0] = []
    probe_answers[1] = ['A', None]
    probe_answers[2] = ['D']
    selected = np.empty(3, dtype=object)
    selected[:] = [None, 17, None]
    metrics = compute_probe_metrics({
        'probe_answers': probe_answers,
        'selected_stop_position': selected,
    })
    assert metrics['probe/candidates'] == 3
    assert metrics['probe/candidates_with_proposals'] == 2
    assert metrics['probe/proposals'] == 3
    assert metrics['probe/recognized'] == 2
    assert abs(metrics['probe/recognized_ratio'] - 2 / 3) < 1e-12
    assert metrics['probe/selected_stops'] == 1


def test_reward_component_metrics_report_strict_raw_accuracy():
    metrics = compute_reward_component_metrics({
        'strict_raw_acc': [1, 1, 0, 1],
        'raw_policy_acc': [1, 1, 0, 1],
        'strict_final_valid': [1, 1, 0, 1],
        'format_score': [1, 0, 0, 1],
        'stop_presence_format_score': [1, 0, 1, 1],
        'structural_format_score': [1, 0, 0, 1],
        'verified_stop': [1, 0, 0, 1],
        'verified_stop_count': [2, 0, 0, 1],
        'verified_stop_score': [0.8, 0.0, 0.0, 0.4],
        'first_probe_correct': [1, 0, 0, 0],
        'stop_earliness': [0.8, 0.0, 0.0, 0.4],
        'stop_reward': [0.2, 0.0, 0.0, 0.1],
        'format_reward': [0.25, 0.0, 0.0, 0.25],
        'accuracy_reward': [0.5, 0.5, 0.0, 0.5],
        'raw_stop_count': [1, 0, 9, 3],
        'stop_over_limit': [0, 0, 1, 0],
        'overflow_action_penalty_required': [0, 0, 1, 0],
        'auxiliary_reward_gated': [0, 0, 1, 0],
        'eligible_stop_count': [1, 0, 0, 3],
        'probed_stop_count': [1, 0, 0, 3],
        'verified_stop_position': [100, -1, -1, 500],
        'selected_stop_ordinal': [1, -1, -1, 3],
    })
    assert metrics['reward/strict_raw_acc'] == 0.75
    assert metrics['reward/raw_policy_acc'] == 0.75
    assert metrics['reward/strict_final_parse_rate'] == 0.75
    assert metrics['reward/format_valid_rate'] == 0.5
    assert metrics['reward/stop_presence_format_rate'] == 0.75
    assert metrics['reward/structural_format_rate'] == 0.5
    assert metrics['reward/format_component'] == 0.125
    assert metrics['reward/verified_stop_count_mean'] == 0.75
    assert metrics['reward/auxiliary_reward_gated_rate'] == 0.25
    assert metrics['reward/overflow_action_penalty_required_rate'] == 0.25
    assert metrics['reward/probed_stop_count_mean'] == 1.0
    assert metrics['reward/verified_stop_rate'] == 0.5
    assert abs(metrics['reward/stop_earliness_verified'] - 0.6) < 1e-12
    assert metrics['reward/verified_stop_position_mean'] == 300.0
    assert metrics['reward/selected_stop_ordinal_mean'] == 2.0
    assert metrics['reward/raw_stop_count_max'] == 9.0

    try:
        compute_reward_component_metrics({
            'strict_raw_acc': [1, 0],
            'raw_policy_acc': [0, 0],
        })
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched strict raw accuracy was accepted')


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_prompt_group_centered_stop_action_weights_and_gradients()
    test_prompt_group_centering_fails_closed()
    test_actor_loss_mask_selection_and_gradient()
    test_weighted_negative_action_loss_has_correct_sign_and_bounded_strength()
    test_incorrect_probe_routing_is_per_action_and_none_is_neutral()
    test_incorrect_stop_loss_sign_mass_and_dynamic_pack_invariance()
    test_action_isolated_verified_and_overflow_stop_losses_are_audited()
    test_all_four_overflow_actions_per_rollout_are_penalized()
    test_missing_any_post_budget_stop_penalty_fails_closed()
    test_classifier_online_fit_is_disabled_without_side_effects()
    test_classifier_collect_train_and_checkpoint()
    test_dual_classifier_states_use_independent_physical_keys_and_targets()
    test_classifier_dedup_majority_and_tie_contract()
    test_probe_metrics_with_empty_and_variable_rows()
    test_reward_component_metrics_report_strict_raw_accuracy()
    print(
        'PASS: centered stop actions, actor mask, reward metrics, '
        'classifier update, and checkpoint reload'
    )
