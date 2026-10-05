import unittest
from unittest import mock

import numpy as np
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from verl.src.medqa_utils import extract_strict_terminal_choice
from verl.src.stop_vllm_rollout import (
    _atomic_stop_positions,
    _atomic_token_id,
    _ByteLevelStopScanner,
    _budgeted_eligible_stop_positions,
    _build_full_response,
    _choice_token_ids,
    _count_atomic_stop_tokens,
    _configured_max_num_seqs,
    _eligible_stop_positions,
    _earliest_safe_probe,
    _exponential_stop_earliness,
    _gate_safe_probes_for_overflow,
    _make_probe_sampling_params,
    _mask_unverified_stop_actions,
    _object_array_1d,
    _overflow_stop_action_mask,
    _post_accepted_tail_auxiliary,
    _probe_answer_and_scores,
    _repeat_by_indices,
    _required_stop_configuration,
    _resolve_stop_auxiliary_action_masks,
    _stop_event_action_metadata,
    _verified_stop_action_mask,
    EarlyStopFeaturizer,
    build_dual_earlystop_training_data,
    build_earlystop_training_data,
)


class DummyTokenizer:
    mapping = {
        "<stop>": [151669],
        "<think>": [151670],
        "</think>": [151671],
        "<final_answer>": [151672],
        "A": [10],
        "B": [11],
        "C": [12],
        "D": [13],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self.mapping[text])

    def decode(self, token_ids, skip_special_tokens=False):
        inverse = {value[0]: key for key, value in self.mapping.items()}
        return "".join(inverse.get(int(token_id), "?") for token_id in token_ids)


class _DummyByteLevelDecoder:
    def __repr__(self):
        return "ByteLevel(add_prefix_space=False, trim_offsets=False)"


class _DummyBackendTokenizer:
    decoder = _DummyByteLevelDecoder()


class DummyByteLevelTokenizer:
    """Tiny ByteLevel vocabulary with several non-canonical stop segmentations."""

    backend_tokenizer = _DummyBackendTokenizer()

    def __init__(self):
        byte_encoder = bytes_to_unicode()

        def piece(payload):
            return "".join(byte_encoder[value] for value in payload)

        self.pieces = {
            27: piece(b"<"),
            29: piece(b">"),
            267: piece(b"st"),
            453: piece(b"op"),
            3481: piece(b"top"),
            9495: piece(b"stop"),
            44047: piece(b"<s"),
            50000: piece(b" stop."),
            50001: piece(b"<stopwatch>"),
            50002: piece(b"stop>stop>"),
            50003: piece(b"nonstop>"),
            50004: piece(b"<STOP>"),
            50005: piece(b"Stop>"),
            50006: piece(b"stopping>"),
            151671: piece(b"</think>"),
        }

    def get_added_vocab(self):
        return {"<stop>": 151669}

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[int(token_id)]


class FakeLogprob:
    def __init__(self, value):
        self.logprob = value


class FakeProbeOutput:
    def __init__(self, token_id, logprobs):
        item = type("Item", (), {})()
        item.token_ids = [token_id]
        item.logprobs = [{key: FakeLogprob(value) for key, value in logprobs.items()}]
        self.outputs = [item]


class OpenQATokenizer:
    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, token_ids, skip_special_tokens=False):
        del token_ids, skip_special_tokens
        return self.decoded


def record(answer, scores):
    scores = np.asarray(scores, dtype=np.float64)
    probabilities = np.exp(scores - np.max(scores))
    probabilities /= probabilities.sum()
    return answer, scores, probabilities


class StopRolloutPureFunctionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_atomic_stop_rejects_legacy_surface_tokens(self):
        stop_id = _atomic_token_id(self.tokenizer, "<stop>", 151669)
        self.assertEqual(_atomic_stop_positions([27, 9495, 29], stop_id), [])
        self.assertEqual(_atomic_stop_positions([7, stop_id, 8, stop_id], stop_id), [1, 3])
        with self.assertRaises(RuntimeError):
            _atomic_token_id(self.tokenizer, "<stop>", 999)

    def test_bytelevel_scanner_finds_all_exact_surface_segmentations(self):
        scanner = _ByteLevelStopScanner(DummyByteLevelTokenizer(), 151669)
        response = [
            151669,
            27,
            9495,
            29,
            44047,
            3481,
            29,
            27,
            267,
            453,
            29,
        ]
        events = scanner.scan(response)
        self.assertEqual(
            [event.kind for event in events],
            ["atomic"] + ["ordinary_surface_stop"] * 3,
        )
        self.assertEqual([event.end_token for event in events], [0, 3, 6, 10])

    def test_bytelevel_scanner_catches_fragments_not_ordinary_english(self):
        scanner = _ByteLevelStopScanner(DummyByteLevelTokenizer(), 151669)
        cases = [
            ([27, 9495], 1),       # <stop
            ([9495, 29], 1),       # stop>
            ([29, 9495, 27], 2),   # >stop<
        ]
        for response, expected_anchor in cases:
            with self.subTest(response=response):
                events = scanner.scan(response)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].kind, "stop_fragment")
                self.assertEqual(events[0].end_token, expected_anchor)
        self.assertEqual(scanner.scan([50000]), [])  # ordinary " stop."
        self.assertEqual(scanner.scan([50001]), [])  # <stopwatch>
        self.assertEqual(scanner.scan([50003]), [])  # nonstop>
        self.assertEqual(scanner.scan([50006]), [])  # stopping>
        for token_id in (50004, 50005):
            events = scanner.scan([token_id])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].kind, "stop_fragment")

    def test_surface_events_consume_global_budget_and_are_always_negative(self):
        scanner = _ByteLevelStopScanner(DummyByteLevelTokenizer(), 151669)
        response = [27, 9495, 29] + [151669] * 8
        events = scanner.scan(response)
        metadata = _stop_event_action_metadata(events, len(response), 8)
        atomic_positions = list(range(3, 11))
        self.assertEqual(metadata["raw_stop_count"], 9)
        self.assertEqual(metadata["raw_atomic_stop_count"], 8)
        self.assertEqual(metadata["illegal_surface_stop_count"], 1)
        self.assertEqual(metadata["overflow_atomic_stop_count"], 1)
        self.assertEqual(metadata["negative_stop_event_count"], 2)
        self.assertEqual(
            _budgeted_eligible_stop_positions(
                atomic_positions,
                atomic_positions,
                8,
                raw_stop_events=events,
            ),
            atomic_positions[:7],
        )
        self.assertEqual(metadata["negative_stop_aux_mask"][2], 1)
        self.assertEqual(metadata["negative_stop_aux_mask"][10], 1)
        self.assertAlmostEqual(metadata["negative_stop_aux_weight"][2], 0.5)
        self.assertAlmostEqual(metadata["negative_stop_aux_weight"][10], 0.5)
        self.assertEqual(metadata["overflow_stop_mask"][10], 1)

    def test_one_token_multiple_surface_events_preserves_multiplicity(self):
        scanner = _ByteLevelStopScanner(DummyByteLevelTokenizer(), 151669)
        events = scanner.scan([50002])
        metadata = _stop_event_action_metadata(events, 1, 8)
        self.assertEqual(len(events), 2)
        self.assertEqual(metadata["negative_stop_event_multiplicity"], [2])
        self.assertEqual(metadata["negative_stop_aux_mask"], [1])
        self.assertEqual(metadata["negative_stop_aux_weight"], [1.0])

    def test_post_accepted_tail_stops_before_close_and_has_bounded_mass(self):
        response = [100, 151669, 101, 102, 103, 104, 151671, 200]
        negative = [0, 0, 0, 1, 0, 0, 0, 0]
        mask, weight = _post_accepted_tail_auxiliary(
            response,
            accepted_stop_position=1,
            think_close_ids=[151671],
            negative_stop_aux_mask=negative,
            atomic_stop_mask=[0, 1, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(mask, [0, 0, 1, 0, 1, 1, 0, 0])
        self.assertAlmostEqual(sum(weight), 3.0 / 256.0)
        self.assertEqual(weight[3], 0.0)
        self.assertEqual(weight[6], 0.0)

        long_response = list(range(300))
        long_mask, long_weight = _post_accepted_tail_auxiliary(
            long_response,
            accepted_stop_position=0,
            think_close_ids=[9999],
            negative_stop_aux_mask=[0] * len(long_response),
            atomic_stop_mask=[1] + [0] * (len(long_response) - 1),
        )
        self.assertEqual(sum(long_mask), 299)
        self.assertAlmostEqual(sum(long_weight), 1.0, places=7)

    def test_raw_stop_count_is_not_capped_by_probe_limit(self):
        stop_id = _atomic_token_id(self.tokenizer, "<stop>", 151669)
        token_ids = [7] + [stop_id, 8] * 9
        self.assertEqual(
            _atomic_stop_positions(token_ids, stop_id, max_proposals=5),
            [1, 3, 5, 7, 9],
        )
        self.assertEqual(
            _atomic_stop_positions(token_ids, stop_id),
            [1, 3, 5, 7, 9, 11, 13, 15, 17],
        )
        self.assertEqual(_count_atomic_stop_tokens(token_ids, stop_id), 9)

    def test_only_stops_inside_first_thinking_region_are_proposals(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        think_open = self.tokenizer.mapping["<think>"]
        think_close = self.tokenizer.mapping["</think>"]
        final_open = self.tokenizer.mapping["<final_answer>"]
        token_ids = [
            stop_id,
            think_open[0],
            7,
            stop_id,
            8,
            stop_id,
            think_close[0],
            stop_id,
            final_open[0],
            stop_id,
        ]
        self.assertEqual(
            _eligible_stop_positions(
                token_ids,
                stop_id,
                think_open,
                think_close,
                final_open,
            ),
            [3, 5],
        )
        self.assertEqual(
            _eligible_stop_positions(
                [think_open[0], 7, stop_id, 8],
                stop_id,
                think_open,
                think_close,
                final_open,
            ),
            [2],
        )
        nine_inside = [think_open[0]] + [stop_id, 7] * 9 + [think_close[0]]
        eligible = _eligible_stop_positions(
            nine_inside,
            stop_id,
            think_open,
            think_close,
            final_open,
        )
        self.assertEqual(len(eligible), 9)
        self.assertEqual(len(eligible[:5]), 5)
        self.assertEqual(_count_atomic_stop_tokens(nine_inside, stop_id), 9)

    def test_probe_is_one_output_one_token_closed_vocabulary(self):
        choice_ids = _choice_token_ids(self.tokenizer)
        params = _make_probe_sampling_params(choice_ids)
        self.assertEqual(params.n, 1)
        self.assertEqual(params.best_of, 1)
        self.assertEqual(params.max_tokens, 1)
        self.assertEqual(set(params.allowed_token_ids), set(choice_ids.values()))
        self.assertEqual(params.logprobs, 4)

    def test_openqa_boxed_boundary_ends_stop_eligibility(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        think_open = [100]
        think_close = [101]
        boxed_open = [102, 103]
        token_ids = [
            think_open[0],
            7,
            stop_id,
            8,
            boxed_open[0],
            boxed_open[1],
            stop_id,
        ]
        self.assertEqual(
            _eligible_stop_positions(
                token_ids,
                stop_id,
                think_open,
                think_close,
                boxed_open,
            ),
            [2],
        )

    def test_vllm_scheduler_concurrency_is_validated(self):
        self.assertEqual(_configured_max_num_seqs({"max_num_seqs": 4}), 4)
        self.assertEqual(_configured_max_num_seqs({}), 256)
        with self.assertRaises(ValueError):
            _configured_max_num_seqs({"max_num_seqs": 0})

    def test_stop_budget_and_half_life_are_required_single_source_config(self):
        valid = {
            "max_stop_count": 8,
            "stop_reward_half_life_tokens": 1024.0,
            "min_verified_stop_separation_tokens": 50,
        }
        self.assertEqual(_required_stop_configuration(valid), (8, 1024.0, 50, 1.0))
        with self.assertRaises(ValueError):
            _required_stop_configuration(
                {
                    "stop_reward_half_life_tokens": 1024.0,
                    "min_verified_stop_separation_tokens": 50,
                }
            )
        with self.assertRaises(ValueError):
            _required_stop_configuration(
                {
                    "max_stop_count": 8,
                    "min_verified_stop_separation_tokens": 50,
                }
            )
        with self.assertRaises(ValueError):
            _required_stop_configuration(
                {
                    "max_stop_count": 8,
                    "stop_reward_half_life_tokens": 1024.0,
                }
            )
        with self.assertRaises(ValueError):
            _required_stop_configuration({**valid, "max_stop_proposals": 8})

    def test_raw_stop_budget_precedes_eligibility_filter(self):
        raw_positions = [1, 10, 20, 30, 40, 50, 60, 70, 80]
        # Position 1 is malformed/outside think but still consumes attempt 1.
        eligible_positions = raw_positions[1:]
        self.assertEqual(
            _budgeted_eligible_stop_positions(
                raw_positions,
                eligible_positions,
                max_stop_count=8,
            ),
            [10, 20, 30, 40, 50, 60, 70],
        )

    def test_candidate_to_base_mapping_for_repeated_rollouts(self):
        base_gold = np.asarray(["A", "D"], dtype=object)
        candidate_base = [0, 0, 0, 1, 1, 1]
        mapped = _repeat_by_indices(base_gold, candidate_base).tolist()
        self.assertEqual(mapped, ["A", "A", "A", "D", "D", "D"])

    def test_variable_length_non_tensor_batches_remain_one_dimensional(self):
        all_empty = _object_array_1d([[], []])
        mixed = _object_array_1d([[None], [None, None]])
        self.assertEqual(all_empty.shape, (2,))
        self.assertEqual(mixed.shape, (2,))
        joined = np.concatenate([all_empty, mixed], axis=0)
        self.assertEqual(joined.shape, (4,))
        self.assertEqual(joined[0], [])
        self.assertEqual(joined[3], [None, None])

    def test_no_atomic_stop_means_no_safe_probe(self):
        positions = _atomic_stop_positions([1, 2, 3], 151669)
        self.assertEqual(positions, [])
        self.assertIsNone(_earliest_safe_probe(positions, [], "A"))

    def test_incomplete_probe_never_reuses_or_triggers_an_answer(self):
        choice_ids = _choice_token_ids(self.tokenizer)
        incomplete = FakeProbeOutput(choice_ids["A"], {choice_ids["A"]: -0.1})
        answer, _scores, _probabilities = _probe_answer_and_scores(
            incomplete, self.tokenizer, choice_ids
        )
        self.assertIsNone(answer)

        complete = FakeProbeOutput(
            choice_ids["B"],
            {
                choice_ids["A"]: -3.0,
                choice_ids["B"]: -0.2,
                choice_ids["C"]: -2.0,
                choice_ids["D"]: -4.0,
            },
        )
        answer, _scores, probabilities = _probe_answer_and_scores(
            complete, self.tokenizer, choice_ids
        )
        self.assertEqual(answer, "B")
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=8)

    def test_nonfinite_probe_is_rejected_without_poisoning_features(self):
        choice_ids = _choice_token_ids(self.tokenizer)
        invalid = FakeProbeOutput(
            choice_ids["A"],
            {token_id: float("nan") for token_id in choice_ids.values()},
        )
        answer, scores, probabilities = _probe_answer_and_scores(
            invalid, self.tokenizer, choice_ids
        )
        self.assertIsNone(answer)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities, np.full(4, 0.25))

    def test_probe_uses_the_generated_choice_not_logprob_argmax(self):
        choice_ids = _choice_token_ids(self.tokenizer)
        mismatch = FakeProbeOutput(
            choice_ids["D"],
            {
                choice_ids["A"]: -0.1,
                choice_ids["B"]: -2.0,
                choice_ids["C"]: -3.0,
                choice_ids["D"]: -4.0,
            },
        )
        answer, _scores, _probabilities = _probe_answer_and_scores(
            mismatch, self.tokenizer, choice_ids
        )
        self.assertEqual(answer, "D")

    def test_first_correct_probe_ignores_wrong_and_unrecognized_history(self):
        records = [
            record("A", [-0.1, -2, -3, -4]),
            record(None, [0, 0, 0, 0]),
            record("C", [-3, -2, -0.1, -4]),
            record("C", [-3, -2, -0.2, -4]),
        ]
        self.assertEqual(
            _earliest_safe_probe([10, 20, 30, 40], records, "C"),
            (30, "C"),
        )
        self.assertIsNone(
            _earliest_safe_probe([10, 20], records[:2], "C")
        )

    def test_only_earliest_correct_stop_is_selected(self):
        records = [
            record("C", [-3, -2, -0.1, -4]),
            record("C", [-3, -2, -0.1, -4]),
            record("C", [-3, -2, -0.1, -4]),
            record("A", [-0.1, -2, -3, -4]),
            record("C", [-3, -2, -0.1, -4]),
        ]
        self.assertEqual(
            _earliest_safe_probe(
                [100, 149, 150, 170, 200],
                records,
                "C",
            ),
            (100, "C"),
        )

    def test_absolute_earliness_is_tail_invariant_and_strictly_monotone(self):
        # The helper deliberately has no raw-rollout-length argument.
        at_100_short_tail = _exponential_stop_earliness(100, 1024)
        at_100_long_tail = _exponential_stop_earliness(100, 1024)
        at_500 = _exponential_stop_earliness(500, 1024)
        self.assertEqual(at_100_short_tail, at_100_long_tail)
        self.assertGreater(at_100_short_tail, at_500)
        self.assertAlmostEqual(_exponential_stop_earliness(1024, 1024), 0.5)
        with self.assertRaises(ValueError):
            _exponential_stop_earliness(10, 0)

    def test_all_atomic_stops_are_excluded_from_trajectory_mask(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        response = [1, stop_id, 2, stop_id, 3, 4]
        masked = _mask_unverified_stop_actions(
            [1, 1, 1, 1, 0, 0],
            response,
            stop_id,
            verified_stop_position=3,
        )
        self.assertEqual(masked, [1, 0, 1, 0, 0, 0])
        all_invalid = _mask_unverified_stop_actions(
            [1] * len(response),
            response,
            stop_id,
            verified_stop_position=None,
        )
        self.assertEqual(all_invalid, [1, 0, 1, 0, 1, 1])
        with self.assertRaises(ValueError):
            _mask_unverified_stop_actions(
                [1] * len(response),
                response,
                stop_id,
                verified_stop_position=None,
                max_stop_count=0,
            )
        with self.assertRaises(ValueError):
            _mask_unverified_stop_actions(
                [1],
                response,
                stop_id,
                verified_stop_position=None,
            )

    def test_overflow_does_not_revoke_legal_verified_stop_credit(self):
        for raw_stop_count in (8, 9, 10, 12):
            with self.subTest(raw_stop_count=raw_stop_count):
                safe, suppressed = _gate_safe_probes_for_overflow(
                    [(10, "A")],
                    raw_stop_count=raw_stop_count,
                    max_stop_count=8,
                )
                self.assertEqual(safe, [(10, "A")])
                self.assertEqual(suppressed, 0)

    def test_default_restored_mask_preserves_the_full_tail(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        raw_response = [101, stop_id, 102, 103, stop_id, 104]
        response, action_mask = _build_full_response(
            candidate_ids=raw_response,
            stop_token_id=stop_id,
            verified_stop_position=1,
            max_stop_count=8,
            cut_accepted_stop_tail=False,
        )
        self.assertEqual(response, raw_response)
        self.assertEqual(action_mask, [1, 0, 1, 1, 0, 1])

        no_acceptance_response, no_acceptance_mask = _build_full_response(
            candidate_ids=raw_response,
            stop_token_id=stop_id,
            verified_stop_position=None,
            max_stop_count=8,
            cut_accepted_stop_tail=False,
        )
        self.assertEqual(no_acceptance_response, raw_response)
        self.assertEqual(no_acceptance_mask, [1, 0, 1, 1, 0, 1])

    def test_accepted_stop_tail_cut_validates_selected_position(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        response = [101, stop_id, 102]
        with self.assertRaises(ValueError):
            _build_full_response(
                candidate_ids=response,
                stop_token_id=stop_id,
                verified_stop_position=2,
                cut_accepted_stop_tail=True,
            )
        with self.assertRaises(ValueError):
            _build_full_response(
                candidate_ids=response,
                stop_token_id=stop_id,
                verified_stop_position=3,
                cut_accepted_stop_tail=True,
            )

    def test_verified_stop_mask_is_single_hot_and_validated(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        response = [1, stop_id, 2, stop_id, 3]
        self.assertEqual(
            _verified_stop_action_mask(response, stop_id, 3),
            [0, 0, 0, 1, 0],
        )
        with self.assertRaises(ValueError):
            _verified_stop_action_mask(response, stop_id, [1, 3])
        self.assertEqual(
            _verified_stop_action_mask(response, stop_id, None),
            [0] * len(response),
        )
        with self.assertRaises(ValueError):
            _verified_stop_action_mask(response, stop_id, 2)
        with self.assertRaises(ValueError):
            _verified_stop_action_mask(response, stop_id, len(response))

    def test_every_stop_action_beyond_the_budget_is_overflow(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        for raw_stop_count in (8, 9, 10, 12):
            with self.subTest(raw_stop_count=raw_stop_count):
                response = []
                for index in range(raw_stop_count):
                    response.extend([100 + index, stop_id])
                overflow_mask = _overflow_stop_action_mask(
                    response,
                    stop_token_id=stop_id,
                    max_stop_count=8,
                )
                stop_positions = list(range(1, 2 * raw_stop_count, 2))
                expected_overflow_positions = stop_positions[8:]
                self.assertTrue(
                    all(
                        overflow_mask[position] == 0
                        for position in stop_positions[:8]
                    )
                )
                self.assertTrue(
                    all(
                        overflow_mask[position] == 1
                        for position in expected_overflow_positions
                    )
                )
                self.assertEqual(
                    sum(overflow_mask), max(raw_stop_count - 8, 0)
                )

    def test_verified_probe_preserves_response_and_trains_non_stop_tokens(self):
        records = [record("B", [-2, -0.1, -3, -4]), record("B", [-3, -0.2, -4, -5])]
        safe = _earliest_safe_probe([2, 5], records, "B")
        self.assertEqual(safe, (2, "B"))
        raw_response = [101, 102, 151669, 104, 105, 151669, 999]
        response, action_mask = _build_full_response(
            candidate_ids=raw_response,
            stop_token_id=151669,
            verified_stop_position=safe[0],
        )
        self.assertEqual(response, raw_response)
        self.assertEqual(action_mask, [1, 1, 0, 1, 1, 0, 1])
        self.assertEqual(action_mask[-1], 1)

        cut_response, cut_action_mask = _build_full_response(
            candidate_ids=raw_response,
            stop_token_id=151669,
            verified_stop_position=safe[0],
            cut_accepted_stop_tail=True,
        )
        self.assertEqual(cut_response, raw_response)
        self.assertEqual(cut_action_mask, [1, 1, 0, 0, 0, 0, 0])

    def test_multi_overflow_response_is_never_rewritten_or_truncated(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        raw_response = [101]
        for index in range(12):
            raw_response.extend([stop_id, 200 + index])
        raw_response.extend([301, 302, 303])

        response, action_mask = _build_full_response(
            candidate_ids=raw_response,
            stop_token_id=stop_id,
            verified_stop_position=1,
            max_stop_count=8,
        )
        self.assertEqual(response, raw_response)
        self.assertEqual(len(response), len(raw_response))
        self.assertEqual(_count_atomic_stop_tokens(response, stop_id), 12)
        self.assertEqual(action_mask[1], 0)
        self.assertTrue(all(action_mask[position] == 0 for position in range(3, 16, 2)))
        # Every stop after the eighth is excluded from trajectory GRPO and
        # receives only the separately normalized negative auxiliary loss.
        self.assertTrue(
            all(action_mask[position] == 0 for position in range(17, 24, 2))
        )
        self.assertEqual(action_mask[-1], 1)

    def test_verified_and_multiple_overflow_actions_coexist_disjointly(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        response = [stop_id] * 12
        verified_mask, overflow_mask, suppressed = (
            _resolve_stop_auxiliary_action_masks(
                response,
                stop_token_id=stop_id,
                verified_stop_positions=0,
                max_stop_count=8,
            )
        )
        self.assertEqual(verified_mask, [1] + [0] * 11)
        self.assertEqual(overflow_mask, [0] * 8 + [1] * 4)
        self.assertEqual(sum(verified_mask), 1)
        self.assertEqual(sum(overflow_mask), 4)
        self.assertFalse(
            any(
                verified and overflow
                for verified, overflow in zip(verified_mask, overflow_mask)
            )
        )
        self.assertEqual(suppressed, 0)

        with self.assertRaises(ValueError):
            _resolve_stop_auxiliary_action_masks(
                response,
                stop_token_id=stop_id,
                verified_stop_positions=8,
                max_stop_count=8,
            )

    def test_full_raw_final_answer_uses_shared_strict_parser(self):
        raw_text = (
            "<think>reasoning<stop>more reasoning</think>"
            "<final_answer>C</final_answer>"
        )
        self.assertEqual(extract_strict_terminal_choice(raw_text), "C")
        self.assertIsNone(extract_strict_terminal_choice("reasoning contains A and B"))
        self.assertIsNone(extract_strict_terminal_choice(raw_text + " trailing text"))
        self.assertIsNone(
            extract_strict_terminal_choice(
                raw_text + "<final_answer>A</final_answer>"
            )
        )

    def test_length_limited_unverified_final_stop_remains_ignored(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        response, action_mask = _build_full_response(
            candidate_ids=[101, 102, stop_id],
            stop_token_id=stop_id,
            verified_stop_position=None,
        )
        self.assertEqual(response, [101, 102, stop_id])
        self.assertEqual(action_mask, [1, 1, 0])

    def test_all_terminal_stops_after_eight_use_negative_auxiliary_mask(self):
        stop_id = self.tokenizer.mapping["<stop>"][0]
        candidate_ids = [stop_id] * 12
        response, action_mask = _build_full_response(
            candidate_ids=candidate_ids,
            stop_token_id=stop_id,
            verified_stop_position=None,
            max_stop_count=8,
        )
        overflow_mask = _overflow_stop_action_mask(
            response,
            stop_token_id=stop_id,
            max_stop_count=8,
        )
        self.assertEqual(action_mask, [0] * 12)
        self.assertEqual(overflow_mask, [0] * 8 + [1] * 4)

    def test_classifier_labels_preserve_full_rollout_not_ground_truth(self):
        records = [[record("A", [-0.1, -2, -3, -4]), record("B", [-2, -0.1, -3, -4])]]
        x, y = build_earlystop_training_data(
            records, physical_final_choices=["B"]
        )
        self.assertEqual(x[0].shape, (2, 30))
        self.assertEqual(y[0].tolist(), [0, 1])

    def test_dual_classifier_uses_every_untouched_physical_probe(self):
        # The second and third rows represent probes that an oracle controller
        # would have hidden after accepting the first stop.  They must remain
        # present in both physical classifier datasets.
        records = [[
            record("A", [-0.1, -2, -3, -4]),
            record("C", [-2, -3, -0.1, -4]),
            record("B", [-2, -0.1, -3, -4]),
        ]]
        final_x, final_y, union_x, union_y = build_dual_earlystop_training_data(
            records,
            physical_final_choices=["B"],
            gold_choices=["A"],
        )
        self.assertEqual(final_x[0].shape, (3, 30))
        self.assertEqual(union_x[0].shape, (3, 30))
        self.assertEqual(final_y[0].tolist(), [0, 0, 1])
        self.assertEqual(union_y[0].tolist(), [1, 0, 1])

    def test_gold_or_final_still_uses_physical_probes_without_strict_final(self):
        records = [[
            record("A", [-0.1, -2, -3, -4]),
            record("D", [-2, -3, -4, -0.1]),
        ]]
        final_x, final_y, union_x, union_y = build_dual_earlystop_training_data(
            records,
            physical_final_choices=[None],
            gold_choices=["A"],
        )
        self.assertEqual(final_x[0].shape, (0, 30))
        self.assertEqual(final_y[0].shape, (0,))
        self.assertEqual(union_x[0].shape, (2, 30))
        self.assertEqual(union_y[0].tolist(), [1, 0])

    def test_classifier_features_are_causal_and_report_winner_change(self):
        scores = [
            np.asarray([-0.01, -4.0, -4.0, -4.0]),
            np.asarray([-4.0, 2.0, -4.0, -4.0]),
        ]
        short = EarlyStopFeaturizer()
        long = EarlyStopFeaturizer()
        short_rows = [short.step(row, i + 1, 2) for i, row in enumerate(scores)]
        long_rows = [long.step(row, i + 1, 200) for i, row in enumerate(scores)]
        np.testing.assert_allclose(short_rows, long_rows)
        self.assertEqual(float(short_rows[0][9]), 0.0)
        self.assertEqual(float(short_rows[1][9]), 1.0)


if __name__ == "__main__":
    unittest.main()
