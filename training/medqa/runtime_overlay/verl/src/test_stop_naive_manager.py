import unittest

import torch

from stop_naive_manager import StopNaiveRewardManager


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        if token_ids.tolist() == [1]:
            return "prompt"
        return (
            "<think>x" + "<stop>x" * 9 +
            "</think><final_answer>A</final_answer>"
        )


class _Item:
    def __init__(self):
        self.batch = {
            "prompts": torch.tensor([1]),
            "responses": torch.arange(2, 12),
            "attention_mask": torch.ones(11, dtype=torch.long),
            "raw_stop_count": torch.tensor(9),
            "raw_atomic_stop_count": torch.tensor(9),
            "illegal_surface_stop_count": torch.tensor(0),
            "stop_fragment_count": torch.tensor(0),
            "negative_stop_event_count": torch.tensor(1),
            "atomic_stop_mask": torch.tensor([0] + [1] * 9),
            "stop_event_multiplicity": torch.tensor([0] + [1] * 9),
            "surface_stop_event_multiplicity": torch.zeros(10, dtype=torch.long),
            "negative_stop_event_multiplicity": torch.tensor(
                [0] * 9 + [1]),
            "verified_stop": torch.tensor(0),
            "verified_stop_count": torch.tensor(0),
            "verified_stop_score": torch.tensor(0.0),
            "verified_stop_mask": torch.zeros(10, dtype=torch.long),
            "verified_stop_aux_weight": torch.zeros(10),
            "raw_policy_acc": torch.tensor(1),
            "verified_stop_position": torch.tensor(-1),
            "selected_stop_ordinal": torch.tensor(-1),
            "eligible_stop_count": torch.tensor(9),
            "probed_stop_count": torch.tensor(8),
            "recognized_correct_probe_count": torch.tensor(0),
            "incorrect_stop_count": torch.tensor(0),
            "neutral_probe_stop_count": torch.tensor(8),
            "incorrect_stop_aux_mask": torch.zeros(10, dtype=torch.long),
            "first_probe_correct": torch.tensor(0),
        }
        self.non_tensor_batch = {
            "reward_model": {"ground_truth": "A"},
            "data_source": "medicalqa_with_correct_think",
        }


class _Data:
    def __init__(self):
        self.items = [_Item()]
        self.batch = {"responses": torch.zeros((1, 10), dtype=torch.long)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _VerifiedItem:
    def __init__(self):
        self.batch = {
            "prompts": torch.tensor([1]),
            "responses": torch.tensor([2, 3, 0]),
            "attention_mask": torch.tensor([1, 1, 1, 0]),
            "raw_stop_count": torch.tensor(1),
            "raw_atomic_stop_count": torch.tensor(1),
            "illegal_surface_stop_count": torch.tensor(0),
            "stop_fragment_count": torch.tensor(0),
            "negative_stop_event_count": torch.tensor(0),
            "atomic_stop_mask": torch.tensor([0, 1, 0]),
            "stop_event_multiplicity": torch.tensor([0, 1, 0]),
            "surface_stop_event_multiplicity": torch.tensor([0, 0, 0]),
            "negative_stop_event_multiplicity": torch.tensor([0, 0, 0]),
            "verified_stop": torch.tensor(1),
            "verified_stop_count": torch.tensor(1),
            "verified_stop_score": torch.tensor(0.5),
            "verified_stop_mask": torch.tensor([0, 1, 0]),
            "verified_stop_aux_weight": torch.tensor([0.0, 0.5, 0.0]),
            "raw_policy_acc": torch.tensor(1),
            "verified_stop_position": torch.tensor(1024),
            "selected_stop_ordinal": torch.tensor(1),
            "eligible_stop_count": torch.tensor(1),
            "probed_stop_count": torch.tensor(1),
            "recognized_correct_probe_count": torch.tensor(1),
            "incorrect_stop_count": torch.tensor(0),
            "neutral_probe_stop_count": torch.tensor(0),
            "incorrect_stop_aux_mask": torch.tensor([0, 0, 0]),
            "first_probe_correct": torch.tensor(1),
        }
        self.non_tensor_batch = {
            "reward_model": {"ground_truth": "A"},
            "data_source": "medicalqa_with_correct_think",
        }


class _VerifiedData:
    def __init__(self):
        self.items = [_VerifiedItem()]
        self.batch = {"responses": torch.zeros((1, 3), dtype=torch.long)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class StopNaiveManagerRawCountTest(unittest.TestCase):
    def test_manager_forwards_raw_stop_count(self):
        seen = {}

        def compute_score(**kwargs):
            seen.update(kwargs)
            return {
                "score": 0.0,
                "acc": 1,
                "strict_raw_acc": 1,
                "raw_stop_count": kwargs["raw_stop_count"],
                "stop_over_limit": 1,
            }

        manager = StopNaiveRewardManager(
            tokenizer=_Tokenizer(),
            num_examine=0,
            compute_score=compute_score,
            max_stop_count=8,
            stop_reward_half_life_tokens=1024.0,
            min_verified_stop_separation_tokens=50,
        )
        result = manager(_Data(), return_dict=True)
        self.assertEqual(seen["raw_stop_count"], 9)
        self.assertEqual(seen["verified_stop"], 0)
        self.assertEqual(seen["verified_stop_count"], 0)
        self.assertEqual(seen["verified_stop_score"], 0.0)
        self.assertEqual(seen["eligible_stop_count"], 9)
        self.assertEqual(seen["probed_stop_count"], 8)
        self.assertEqual(seen["max_stop_count"], 8)
        self.assertEqual(seen["stop_reward_half_life_tokens"], 1024.0)
        self.assertEqual(seen["min_verified_stop_separation_tokens"], 50)
        self.assertEqual(float(result["reward_tensor"].sum()), 0.0)
        self.assertEqual(result["reward_extra_info"]["raw_stop_count"], [9])
        self.assertEqual(result["reward_extra_info"]["stop_over_limit"], [1])
        self.assertEqual(result["reward_extra_info"]["raw_policy_acc"], [1])
        self.assertEqual(result["reward_extra_info"]["verified_stop_position"], [-1])
        self.assertEqual(
            result["reward_extra_info"]["recognized_correct_probe_count"],
            [0],
        )
        self.assertEqual(result["reward_extra_info"]["incorrect_stop_count"], [0])
        self.assertEqual(
            result["reward_extra_info"]["neutral_probe_stop_count"], [8]
        )

    def test_manager_accepts_unnormalized_verified_stop_score(self):
        seen = {}

        def compute_score(**kwargs):
            seen.update(kwargs)
            return {
                "score": 0.75,
                "acc": 1,
                "strict_raw_acc": 1,
                "raw_stop_count": kwargs["raw_stop_count"],
                "stop_over_limit": 0,
            }

        manager = StopNaiveRewardManager(
            tokenizer=_Tokenizer(),
            num_examine=0,
            compute_score=compute_score,
            max_stop_count=8,
            stop_reward_half_life_tokens=1024.0,
            min_verified_stop_separation_tokens=50,
        )
        result = manager(_VerifiedData(), return_dict=True)
        self.assertEqual(seen["verified_stop"], 1)
        self.assertEqual(seen["verified_stop_count"], 1)
        self.assertEqual(seen["verified_stop_score"], 0.5)
        self.assertEqual(seen["verified_stop_position"], 1024)
        self.assertAlmostEqual(float(result["reward_tensor"].sum()), 0.75)
        self.assertEqual(result["reward_extra_info"]["verified_stop_score"], [0.5])
        self.assertEqual(
            result["reward_extra_info"]["recognized_correct_probe_count"],
            [1],
        )


if __name__ == "__main__":
    unittest.main()
