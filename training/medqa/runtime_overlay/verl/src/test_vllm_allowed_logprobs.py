import unittest

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


class AllowedTokenLogprobTests(unittest.TestCase):
    def test_constrained_row_returns_every_allowed_token_in_mixed_batch(self):
        logits = torch.tensor(
            [
                [100.0, 1.0, 90.0, 2.0, 80.0, 3.0, 70.0, 4.0],
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        allowed = {1, 3, 5, 7}
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0] = True
        mask[0, list(allowed)] = False

        metadata = SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            min_p=None,
            generators={},
            max_num_logprobs=4,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(2),
            presence_penalties=torch.zeros(2),
            repetition_penalties=torch.ones(2),
            output_token_ids=[[], []],
            min_tokens={},
            logit_bias=[None, None],
            allowed_token_ids_mask=mask,
            bad_words_token_ids={},
        )

        output = Sampler()(logits, metadata)
        self.assertEqual(int(output.sampled_token_ids[0, 0]), 7)
        constrained_ids = set(output.logprobs_tensors.logprob_token_ids[0].tolist())
        self.assertTrue(allowed.issubset(constrained_ids))
        constrained = output.logprobs_tensors.logprobs[0]
        self.assertTrue(torch.isfinite(constrained[:5]).all())

        # The unconstrained row in the same batch retains full-vocabulary
        # top-k semantics (plus the duplicated sampled token entry).
        unconstrained_ids = set(output.logprobs_tensors.logprob_token_ids[1].tolist())
        self.assertTrue({0, 1, 2, 3}.issubset(unconstrained_ids))


if __name__ == "__main__":
    unittest.main()
