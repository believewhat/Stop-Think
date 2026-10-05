import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto


class RawStopDataProtoAlignmentTest(unittest.TestCase):
    def test_reorder_and_concat_keep_response_and_count_aligned(self):
        first = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([[101], [102], [103]]),
                    "raw_stop_count": torch.tensor([1, 9, 4]),
                    "verified_stop": torch.tensor([1, 0, 1]),
                    "raw_policy_acc": torch.tensor([0, 1, 0]),
                },
                batch_size=3,
            ),
            non_tensor_batch={
                "candidate_id": np.asarray(["a", "b", "c"], dtype=object),
            },
        )
        first.reorder(torch.tensor([2, 0, 1]))
        second = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([[104]]),
                    "raw_stop_count": torch.tensor([8]),
                    "verified_stop": torch.tensor([0]),
                    "raw_policy_acc": torch.tensor([1]),
                },
                batch_size=1,
            ),
            non_tensor_batch={
                "candidate_id": np.asarray(["d"], dtype=object),
            },
        )
        joined = DataProto.concat([first, second])
        triples = list(
            zip(
                joined.non_tensor_batch["candidate_id"].tolist(),
                joined.batch["responses"][:, 0].tolist(),
                joined.batch["raw_stop_count"].tolist(),
                joined.batch["verified_stop"].tolist(),
                joined.batch["raw_policy_acc"].tolist(),
            )
        )
        self.assertEqual(
            triples,
            [
                ("c", 103, 4, 1, 0),
                ("a", 101, 1, 1, 0),
                ("b", 102, 9, 0, 1),
                ("d", 104, 8, 0, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
