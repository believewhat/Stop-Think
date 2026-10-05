#!/usr/bin/env python3
"""CPU-only tests for exact DAPO token-mean micro-batch aggregation.

The production actor imports CUDA-only flash-attention modules.  These tests
extract the two pure aggregation helpers from the sibling source file, so the
regression remains runnable on a CPU-only preflight host while still testing
the exact functions that are deployed.
"""

import ast
import math
from pathlib import Path

import torch


DEPLOYED_ACTOR_PATH = (
    Path(__file__).resolve().parents[1] / 'workers' / 'actor' / 'dp_actor.py'
)
# The staging directory keeps the actor beside this test, while the deployed
# repository uses verl/workers/actor.  Prefer the exact deployed source so the
# server preflight cannot accidentally test a stale copy.
ACTOR_PATH = (
    DEPLOYED_ACTOR_PATH
    if DEPLOYED_ACTOR_PATH.is_file()
    else Path(__file__).with_name('dp_actor.py')
)
ACTOR_SOURCE = ACTOR_PATH.read_text(encoding='utf-8')


def _load_aggregation_helpers():
    tree = ast.parse(ACTOR_SOURCE, filename=str(ACTOR_PATH))
    wanted = {
        '_loss_aggregation_denominator',
        '_micro_batch_loss_scale',
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'math': math, 'torch': torch}
    exec(compile(module, str(ACTOR_PATH), 'exec'), namespace)
    return (
        namespace['_loss_aggregation_denominator'],
        namespace['_micro_batch_loss_scale'],
    )


_loss_aggregation_denominator, _micro_batch_loss_scale = (
    _load_aggregation_helpers()
)


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f'expected {exc_type.__name__}')


def _masked_token_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def _called_function_names(node):
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def _update_policy_ast():
    tree = ast.parse(ACTOR_SOURCE, filename=str(ACTOR_PATH))
    actor_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'DataParallelPPOActor'
    )
    return next(
        node
        for node in actor_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'update_policy'
    )


def test_dynamic_micro_batches_reconstruct_exact_token_mean_and_gradient():
    # The three rollouts have 1, 3, and 6 active actor tokens.  Deliberately
    # pack the shortest and longest together so row weighting is visibly
    # different from DAPO token weighting.
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    packs = (torch.tensor([0, 2]), torch.tensor([1]))
    total_denominator = _loss_aggregation_denominator(mask, 'token-mean')
    assert total_denominator == 10.0

    split_values = torch.tensor(
        [
            [1.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
        ],
        requires_grad=True,
    )
    split_loss = split_values.sum() * 0.0
    local_means = []
    for indices in packs:
        local_mask = mask[indices]
        local_mean = _masked_token_mean(split_values[indices], local_mask)
        local_means.append(local_mean)
        scale = _micro_batch_loss_scale(
            local_mask,
            loss_agg_mode='token-mean',
            mini_batch_denominator=total_denominator,
        )
        split_loss = split_loss + local_mean * scale
    split_loss.backward()

    direct_values = split_values.detach().clone().requires_grad_(True)
    direct_loss = _masked_token_mean(direct_values, mask)
    direct_loss.backward()

    assert torch.allclose(split_loss, direct_loss, rtol=0, atol=1e-7)
    assert torch.allclose(
        split_values.grad,
        direct_values.grad,
        rtol=0,
        atol=1e-7,
    )

    # This is the removed behavior: dynamic micro-batches were weighted by
    # rollout rows (2/3 and 1/3), despite containing 7/10 and 3/10 tokens.
    old_row_weighted = local_means[0] * (2.0 / 3.0) + local_means[1] / 3.0
    assert not torch.allclose(old_row_weighted, direct_loss, rtol=0, atol=1e-3)


def test_controller_mask_excludes_forced_suffix_and_discarded_tail_mass():
    actor_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    physical_or_forced_mask = torch.ones_like(actor_mask)
    assert _loss_aggregation_denominator(actor_mask, 'token-mean') == 5.0
    assert (
        _loss_aggregation_denominator(
            physical_or_forced_mask,
            'token-mean',
        )
        == 12.0
    )
    # The production call must consume response_mask (the actor mask), not the
    # physical attention mask used by custom stop-action objectives.
    assert '_micro_batch_loss_scale(\n                        response_mask,' in ACTOR_SOURCE


def test_zero_ordinary_token_batch_keeps_stop_auxiliary_gradient_available():
    empty_actor_mask = torch.zeros((3, 4), dtype=torch.float32)
    denominator = _loss_aggregation_denominator(
        empty_actor_mask,
        'token-mean',
    )
    scale = _micro_batch_loss_scale(
        empty_actor_mask,
        loss_agg_mode='token-mean',
        mini_batch_denominator=denominator,
    )
    assert denominator == 0.0
    assert scale == 0.0

    ordinary_parameter = torch.tensor(2.0, requires_grad=True)
    stop_parameter = torch.tensor(3.0, requires_grad=True)
    ordinary_policy_loss = ordinary_parameter.square()
    stop_aux_contribution = 0.25 * stop_parameter.square()
    total_loss = ordinary_policy_loss * scale + stop_aux_contribution
    total_loss.backward()
    assert ordinary_parameter.grad.item() == 0.0
    assert stop_parameter.grad.item() == 1.5

    # All custom stop terms must remain normalized by rollout rows and be
    # added after the ordinary token-level scaling; the DAPO scale must never
    # be applied to these contributions.
    for contribution in (
        'verified_stop_aux_contribution',
        'overflow_stop_aux_contribution',
        'negative_stop_aux_contribution',
        'incorrect_stop_aux_contribution',
        'post_accepted_tail_aux_contribution',
    ):
        assert f'loss = loss + {contribution}' in ACTOR_SOURCE
    assert ACTOR_SOURCE.count('/ float(mini_batch_row_count)') >= 5


def test_source_bypasses_empty_masked_means_before_they_are_evaluated():
    update_policy = _update_policy_ast()
    active_token_guards = []
    for node in ast.walk(update_policy):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        names = {
            child.id
            for child in ast.walk(node.test)
            if isinstance(child, ast.Name)
        }
        if 'micro_batch_active_token_count' in names:
            active_token_guards.append(node)
    active_token_guards.sort(key=lambda node: node.lineno)
    assert len(active_token_guards) == 2

    ordinary_guard, kl_guard = active_token_guards
    ordinary_calls = _called_function_names(
        ast.Module(body=ordinary_guard.body, type_ignores=[]))
    ordinary_empty_calls = _called_function_names(
        ast.Module(body=ordinary_guard.orelse, type_ignores=[]))
    assert 'compute_policy_loss' in ordinary_calls
    assert 'agg_loss' in ordinary_calls
    assert 'compute_policy_loss' not in ordinary_empty_calls
    assert 'agg_loss' not in ordinary_empty_calls

    policy_call = next(
        child
        for child in ast.walk(ordinary_guard)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == 'compute_policy_loss'
    )
    keyword_values = {keyword.arg: keyword.value for keyword in policy_call.keywords}
    assert 'loss_agg_mode' in keyword_values
    assert isinstance(keyword_values['loss_agg_mode'], ast.Name)
    assert keyword_values['loss_agg_mode'].id == 'loss_agg_mode'

    empty_assignments = {
        target.id
        for statement in ordinary_guard.orelse
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    assert {
        'ordinary_zero',
        'pg_loss',
        'entropy_loss',
        'ppo_kl',
        'pg_clipfrac',
        'pg_clipfrac_lower',
    }.issubset(empty_assignments)

    kl_calls = _called_function_names(
        ast.Module(body=kl_guard.body, type_ignores=[]))
    kl_empty_calls = _called_function_names(
        ast.Module(body=kl_guard.orelse, type_ignores=[]))
    assert 'kl_penalty' in kl_calls
    assert 'agg_loss' in kl_calls
    assert 'kl_penalty' not in kl_empty_calls
    assert 'agg_loss' not in kl_empty_calls

    update_source = ast.get_source_segment(ACTOR_SOURCE, update_policy)
    assert update_source is not None
    assert 'len(data) / self.config.ppo_mini_batch_size' not in update_source
    assert 'policy_loss / self.gradient_accumulation' not in update_source


def test_sequence_level_modes_keep_row_denominator():
    mask = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]],
        dtype=torch.float32,
    )
    for mode in ('seq-mean-token-sum', 'seq-mean-token-mean'):
        total = _loss_aggregation_denominator(mask, mode)
        assert total == 3.0
        assert _micro_batch_loss_scale(
            mask[:2],
            loss_agg_mode=mode,
            mini_batch_denominator=total,
        ) == 2.0 / 3.0
        assert _micro_batch_loss_scale(
            mask[2:],
            loss_agg_mode=mode,
            mini_batch_denominator=total,
        ) == 1.0 / 3.0


def test_aggregation_contract_fails_closed():
    _assert_raises(
        ValueError,
        lambda: _loss_aggregation_denominator(
            torch.tensor([[1.0, 0.5]]),
            'token-mean',
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _loss_aggregation_denominator(
            torch.ones(2, 3),
            'not-a-mode',
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _micro_batch_loss_scale(
            torch.ones(2, 3),
            loss_agg_mode='token-mean',
            mini_batch_denominator=5.0,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _micro_batch_loss_scale(
            torch.zeros(1, 2),
            loss_agg_mode='token-mean',
            mini_batch_denominator=-1.0,
        ),
    )


if __name__ == '__main__':
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'PASS all {len(tests)} DAPO token-mean CPU tests')
