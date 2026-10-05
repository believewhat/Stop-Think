#!/usr/bin/env python3
"""Focused CPU regressions for v18 DAPO sampling and overlong shaping."""

import numpy as np
import torch

from verl import DataProto
from verl.src.stop_ray_trainer import (
    _accumulate_scalar_metrics,
    _finalize_scalar_metrics,
    apply_dapo_soft_overlong_penalty,
    dapo_dynamic_sampling_needs_more,
    select_dapo_nonconstant_uid_groups,
)


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_dynamic_sampling_filters_complete_uid_groups_only():
    uids = np.asarray(
        [
            "all-wrong",
            "mixed",
            "all-correct",
            "mixed",
            "all-wrong",
            "all-correct",
        ],
        dtype=object,
    )
    raw_policy_acc = torch.tensor([0, 0, 1, 1, 0, 1])
    result = select_dapo_nonconstant_uid_groups(
        uids,
        raw_policy_acc,
        expected_group_size=2,
    )

    assert result["prompt_count"] == 3
    assert result["kept_prompt_count"] == 1
    assert result["discarded_prompt_count"] == 2
    assert result["kept_prompt_uids"] == ["mixed"]
    assert result["discarded_prompt_uids"] == [
        "all-wrong",
        "all-correct",
    ]
    assert result["kept_indices"].tolist() == [1, 3]
    assert result["row_keep_mask"].tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
    ]

    _assert_raises(
        ValueError,
        lambda: select_dapo_nonconstant_uid_groups(
            np.asarray(["broken", "other"], dtype=object),
            torch.tensor([0, 1]),
            expected_group_size=2,
        ),
    )
    _assert_raises(
        ValueError,
        lambda: select_dapo_nonconstant_uid_groups(
            np.asarray(["x", "x"], dtype=object),
            torch.tensor([0.0, 0.5]),
            expected_group_size=2,
        ),
    )


def _make_overlong_proto(*, break_physical_equation=False):
    max_response_length = 8
    effective_lengths = torch.tensor([5, 7, 8, 8], dtype=torch.long)
    controller_terminated = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    # Terminated rows contain forced canonical suffixes of length 3 and 1.
    # Their sampled actor horizons are therefore 5 and 7, respectively.
    accepted_prefix_lengths = torch.tensor([0, 0, 5, 7], dtype=torch.long)
    forced_suffix_counts = torch.tensor([0, 0, 3, 1], dtype=torch.long)
    discarded_counts = torch.tensor([0, 0, 3, 0], dtype=torch.long)
    physical_lengths = torch.tensor([5, 7, 8, 7], dtype=torch.long)
    if break_physical_equation:
        physical_lengths[-1] = 8

    response_attention = torch.zeros(
        (4, max_response_length),
        dtype=torch.long,
    )
    for row, length in enumerate(effective_lengths.tolist()):
        response_attention[row, :length] = 1
    prompt_attention = torch.ones((4, 2), dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros(
                (4, max_response_length),
                dtype=torch.long,
            ),
            "attention_mask": torch.cat(
                [prompt_attention, response_attention],
                dim=-1,
            ),
            "effective_response_length": effective_lengths,
            "controller_terminated": controller_terminated,
            "accepted_stop_prefix_length": accepted_prefix_lengths,
            "controller_forced_suffix_token_count": forced_suffix_counts,
            "controller_discarded_token_count": discarded_counts,
            "physical_response_length": physical_lengths,
        }
    )


def test_soft_overlong_uses_sampled_prefix_not_suffix_or_physical_tail():
    proto = _make_overlong_proto()
    base_reward = torch.zeros((4, 8), dtype=torch.float32)
    penalized, penalties, overlong_lengths = (
        apply_dapo_soft_overlong_penalty(
            proto,
            base_reward,
            max_response_length=8,
            buffer_len=2,
            penalty_factor=1.0,
        )
    )

    # Expected length is 6.  The terminated row with effective length 8 but
    # accepted prefix 5 receives no penalty; its three forced tokens are not
    # actor actions.  The last row deliberately has physical<effective, which
    # is valid because one forced token replaced no discarded physical tail.
    assert overlong_lengths.tolist() == [5, 7, 5, 7]
    assert torch.allclose(
        penalties,
        torch.tensor([0.0, -0.5, 0.0, -0.5]),
    )
    expected = torch.zeros_like(base_reward)
    expected[1, 6] = -0.5
    expected[3, 7] = -0.5
    assert torch.equal(penalized, expected)
    assert torch.equal(base_reward, torch.zeros_like(base_reward))


def test_dynamic_sampling_max_generation_guard_is_fail_closed():
    assert dapo_dynamic_sampling_needs_more(
        kept_prompt_count=9,
        target_prompt_count=10,
        num_gen_batches=2,
        max_num_gen_batches=3,
    )
    assert not dapo_dynamic_sampling_needs_more(
        kept_prompt_count=10,
        target_prompt_count=10,
        num_gen_batches=3,
        max_num_gen_batches=3,
    )
    _assert_raises(
        RuntimeError,
        lambda: dapo_dynamic_sampling_needs_more(
            kept_prompt_count=9,
            target_prompt_count=10,
            num_gen_batches=3,
            max_num_gen_batches=3,
        ),
    )


def test_soft_overlong_fails_closed_on_controller_accounting_mismatch():
    proto = _make_overlong_proto(break_physical_equation=True)
    _assert_raises(
        ValueError,
        lambda: apply_dapo_soft_overlong_penalty(
            proto,
            torch.zeros((4, 8), dtype=torch.float32),
            max_response_length=8,
            buffer_len=2,
            penalty_factor=1.0,
        ),
    )


def test_generated_metric_accumulator_preserves_means_sums_and_maxima():
    state = {}
    _accumulate_scalar_metrics(
        state,
        {"x_mean": 1.0, "x_sum": 2.0, "x_max": 3.0},
        2,
    )
    _accumulate_scalar_metrics(
        state,
        {"x_mean": 3.0, "x_sum": 5.0, "x_max": 2.0},
        1,
    )
    result = _finalize_scalar_metrics(state)
    assert abs(result["x_mean"] - (5.0 / 3.0)) < 1e-12
    assert result["x_sum"] == 7.0
    assert result["x_max"] == 3.0


if __name__ == "__main__":
    tests = [
        test_dynamic_sampling_filters_complete_uid_groups_only,
        test_dynamic_sampling_max_generation_guard_is_fail_closed,
        test_soft_overlong_uses_sampled_prefix_not_suffix_or_physical_tail,
        test_soft_overlong_fails_closed_on_controller_accounting_mismatch,
        test_generated_metric_accumulator_preserves_means_sums_and_maxima,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
