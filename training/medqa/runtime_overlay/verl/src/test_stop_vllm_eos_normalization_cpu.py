import unittest

import torch

from verl.src.stop_vllm_rollout import (
    _count_atomic_stop_tokens,
    _truncate_response_at_first_eos,
)


class EosNormalizationTests(unittest.TestCase):
    def test_scalar_eos_is_kept_and_suffix_is_removed(self):
        response, trimmed = _truncate_response_at_first_eos(
            [10, 20, 2, 30, 2],
            2,
        )
        self.assertEqual(response, [10, 20, 2])
        self.assertEqual(trimmed, 2)

    def test_every_configured_eos_id_is_honored(self):
        response, trimmed = _truncate_response_at_first_eos(
            [10, 151643, 30],
            [151645, 151643],
        )
        self.assertEqual(response, [10, 151643])
        self.assertEqual(trimmed, 1)

    def test_tensor_eos_configuration_is_supported(self):
        response, trimmed = _truncate_response_at_first_eos(
            [10, 151645, 30],
            torch.tensor([151645, 151643]),
        )
        self.assertEqual(response, [10, 151645])
        self.assertEqual(trimmed, 1)

    def test_no_eos_leaves_response_unchanged(self):
        response, trimmed = _truncate_response_at_first_eos(
            [10, 20, 30],
            [151645, 151643],
        )
        self.assertEqual(response, [10, 20, 30])
        self.assertEqual(trimmed, 0)

    def test_post_eos_stop_cannot_reach_reward_or_action_masks(self):
        stop_token_id = 151669
        response, trimmed = _truncate_response_at_first_eos(
            [151643, stop_token_id, 30],
            [151645, 151643],
        )
        self.assertEqual(response, [151643])
        self.assertEqual(trimmed, 2)
        self.assertEqual(_count_atomic_stop_tokens(response, stop_token_id), 0)

    def test_empty_eos_configuration_fails_closed(self):
        with self.assertRaises(ValueError):
            _truncate_response_at_first_eos([10], [])


if __name__ == "__main__":
    unittest.main()
