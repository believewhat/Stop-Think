import contextlib
import io
import math
import unittest

from medqa_utils import extract_medqa_choice, extract_strict_terminal_choice
from reward_loss import _well_formed_response_format, compute_score, hf_math_rm


def stop_metadata(
    *,
    verified=False,
    verified_count=None,
    verified_score=None,
    position=-1,
    raw=0,
    raw_atomic=None,
    illegal_surface=0,
    fragments=0,
    ordinal=-1,
    eligible=0,
    probed=None,
    cap=8,
    half_life=256.0,
    min_separation=50,
):
    if raw_atomic is None:
        raw_atomic = raw - illegal_surface
    if probed is None:
        probed = min(eligible, cap)
    if verified_count is None:
        verified_count = int(bool(verified))
    if verified_score is None:
        verified_score = (
            math.pow(2.0, -float(position) / float(half_life))
            if verified
            else 0.0
        )
    return {
        "verified_stop": verified,
        "verified_stop_count": verified_count,
        "verified_stop_score": verified_score,
        "verified_stop_position": position,
        "raw_stop_count": raw,
        "raw_atomic_stop_count": raw_atomic,
        "illegal_surface_stop_count": illegal_surface,
        "stop_fragment_count": fragments,
        "selected_stop_ordinal": ordinal,
        "eligible_stop_count": eligible,
        "probed_stop_count": probed,
        "max_stop_count": cap,
        "stop_reward_half_life_tokens": half_life,
        "min_verified_stop_separation_tokens": min_separation,
    }


def score(solution, gold="A", **metadata):
    return compute_score(
        data_source="medicalqa_with_correct_think",
        solution_str=solution,
        ground_truth=gold,
        **stop_metadata(**metadata),
    )


class MedQAParserTest(unittest.TestCase):
    def test_flexible_parser_is_only_for_probes_and_gold(self):
        cases = [
            ("A", "A"),
            (" b ", "B"),
            (r"Therefore, \boxed{D}.", "D"),
            ("The correct answer is (C).", "C"),
            ("B}", "B"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_medqa_choice(raw), expected)

    def test_strict_terminal_parser_requires_one_uppercase_terminal_block(self):
        self.assertEqual(
            extract_strict_terminal_choice(
                "<think>x</think>\n<final_answer> A </final_answer>\t"
            ),
            "A",
        )
        rejected = [
            "The final answer is A.",
            r"\boxed{A}",
            "A",
            "<final_answer>a</final_answer>",
            "<final_answer>A</final_answer> trailing",
            "<final_answer>A</final_answer><final_answer>A</final_answer>",
            "<final_answer>\u00a0A</final_answer>",
        ]
        for raw in rejected:
            with self.subTest(raw=raw):
                self.assertIsNone(extract_strict_terminal_choice(raw))

    def test_policy_accuracy_does_not_use_probe_fallbacks(self):
        self.assertTrue(hf_math_rm("<final_answer>A</final_answer>", "A")["acc"])
        for raw in ("A", "The final answer is A.", r"\boxed{A}"):
            with self.subTest(raw=raw):
                self.assertFalse(hf_math_rm(raw, "A")["acc"])


class StrictFormatTest(unittest.TestCase):
    def test_accepts_complete_ordered_response(self):
        valid = (
            " \n<think>reasoning<stop>more<stop></think>\r\n"
            "<final_answer> A </final_answer>\t"
        )
        self.assertEqual(_well_formed_response_format(valid), 1)

    def test_accepts_complete_ordered_response_without_stop(self):
        valid = "<think>reasoning</think><final_answer>A</final_answer>"
        self.assertEqual(_well_formed_response_format(valid), 1)

    def test_rejects_every_incomplete_or_misordered_structure(self):
        invalid = [
            "<final_answer>A</final_answer>",
            "<think>x<stop><final_answer>A</final_answer>",
            "<think>x<stop></think>",
            "<think>x</think><stop><final_answer>A</final_answer>",
            "<think>x<stop></think>junk<final_answer>A</final_answer>",
            "<think>x<stop></think><final_answer>a</final_answer>",
            "<think>x<stop></think><final_answer>A</final_answer>junk",
            (
                "<think>x<stop></think><final_answer>A</final_answer>"
                "<final_answer>A</final_answer>"
            ),
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertEqual(_well_formed_response_format(raw), 0)

    def test_surface_stop_event_never_receives_format_credit(self):
        solution = (
            "<think>x<stop>stop>y</think>"
            "<final_answer>A</final_answer>"
        )
        result = score(
            solution,
            verified=True,
            position=8,
            ordinal=1,
            raw=2,
            raw_atomic=1,
            illegal_surface=1,
            fragments=1,
            eligible=1,
            probed=1,
        )
        self.assertEqual(result["format_score"], 0)
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["raw_stop_count"], 2)
        self.assertEqual(result["raw_atomic_stop_count"], 1)
        self.assertEqual(result["negative_stop_action_penalty_required"], 1)

    def test_complete_non_atomic_surface_stop_is_illegal(self):
        solution = (
            "<think>x<stop>y<stop></think>"
            "<final_answer>A</final_answer>"
        )
        result = score(
            solution,
            verified=True,
            position=8,
            ordinal=1,
            raw=2,
            raw_atomic=1,
            illegal_surface=1,
            fragments=0,
            eligible=1,
            probed=1,
        )
        self.assertEqual(result["visible_stop_count"], 2)
        self.assertEqual(result["legacy_surface_stop_count"], 1)
        self.assertEqual(result["format_score"], 0)


class RestoredV3RewardTest(unittest.TestCase):
    formatted_correct = (
        "<think>reasoning<stop></think><final_answer>A</final_answer>"
    )
    formatted_wrong = (
        "<think>reasoning<stop></think><final_answer>B</final_answer>"
    )

    def test_exact_accuracy_and_format_components(self):
        correct = score(self.formatted_correct, raw=1, eligible=1)
        wrong = score(self.formatted_wrong, raw=1, eligible=1)
        self.assertEqual(correct["accuracy_reward"], 0.5)
        self.assertEqual(correct["format_reward"], 0.25)
        self.assertEqual(correct["stop_reward"], 0.0)
        self.assertEqual(correct["score"], 0.75)
        self.assertEqual(wrong["score"], 0.25)

    def test_one_verified_stop_is_not_divided_by_cap(self):
        result = score(
            self.formatted_correct,
            verified=True,
            position=1024,
            raw=1,
            ordinal=1,
            eligible=1,
        )
        self.assertEqual(result["verified_stop_score"], 0.0625)
        self.assertEqual(result["stop_earliness"], 0.0625)
        self.assertEqual(result["stop_reward"], 0.0625)
        self.assertEqual(result["stop_action_reward"], 0.0625)
        self.assertEqual(result["sequence_stop_reward"], 0.0)
        self.assertEqual(result["score"], 0.75)

    def test_verified_ordinal_is_global_when_surface_consumes_first_slot(self):
        result = score(
            self.formatted_correct,
            verified=True,
            position=128,
            raw=2,
            raw_atomic=1,
            illegal_surface=1,
            ordinal=2,
            eligible=1,
            probed=1,
        )
        self.assertEqual(result["selected_stop_ordinal"], 2)
        self.assertEqual(result["verified_stop_count"], 1)
        self.assertEqual(result["negative_stop_action_penalty_required"], 1)

    def test_earlier_verified_stop_scores_strictly_more(self):
        common = {
            "verified": True,
            "raw": 1,
            "ordinal": 1,
            "eligible": 1,
        }
        at_zero = score(self.formatted_correct, position=0, **common)
        early = score(self.formatted_correct, position=128, **common)
        late = score(self.formatted_correct, position=2048, **common)
        self.assertEqual(at_zero["score"], early["score"])
        self.assertEqual(early["score"], late["score"])
        self.assertEqual(late["score"], 0.75)
        self.assertGreater(at_zero["stop_reward"], early["stop_reward"])
        self.assertGreater(early["stop_reward"], late["stop_reward"])

    def test_cap_does_not_scale_single_stop_credit(self):
        common = {
            "verified": True,
            "position": 1024,
            "raw": 1,
            "ordinal": 1,
            "eligible": 1,
        }
        cap_two = score(self.formatted_correct, cap=2, **common)
        cap_eight = score(self.formatted_correct, cap=8, **common)
        self.assertEqual(cap_two["stop_reward"], 0.0625)
        self.assertEqual(cap_two["score"], cap_eight["score"])

    def test_only_one_verified_stop_can_be_credited(self):
        with self.assertRaises(ValueError):
            score(
                self.formatted_correct,
                verified=True,
                verified_count=2,
                verified_score=1.0,
                position=0,
                raw=2,
                ordinal=1,
                eligible=2,
            )

    def test_atomic_surface_and_eligibility_must_match_for_format(self):
        decoded_mismatch = score(
            (
                "<think>atomic<stop>legacy<stop></think>"
                "<final_answer>A</final_answer>"
            ),
            raw=1,
            eligible=1,
        )
        ineligible = score(
            self.formatted_correct,
            raw=1,
            eligible=0,
            probed=0,
        )
        self.assertEqual(decoded_mismatch["strict_raw_acc"], 1)
        self.assertEqual(decoded_mismatch["format_reward"], 0.0)
        self.assertEqual(decoded_mismatch["score"], 0.5)
        self.assertEqual(ineligible["strict_raw_acc"], 1)
        self.assertEqual(ineligible["format_reward"], 0.0)
        self.assertEqual(ineligible["reward_gated"], 0)
        self.assertEqual(ineligible["score"], 0.5)

    def test_incomplete_structure_keeps_accuracy_but_not_format(self):
        result = score(
            "<stop><final_answer>A</final_answer>",
            raw=1,
            eligible=1,
        )
        self.assertEqual(result["strict_raw_acc"], 1)
        self.assertEqual(result["accuracy_reward"], 0.5)
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["score"], 0.5)

    def test_unverified_stop_gets_no_stop_component(self):
        result = score(self.formatted_correct, raw=1, eligible=1)
        self.assertEqual(result["verified_stop"], 0)
        self.assertEqual(result["stop_reward"], 0.0)
        self.assertEqual(result["score"], 0.75)

    def test_probe_answer_never_rewrites_raw_terminal_accuracy(self):
        result = score(
            self.formatted_wrong,
            verified=True,
            position=0,
            raw=1,
            ordinal=1,
            eligible=1,
        )
        self.assertEqual(result["pred"], "B")
        self.assertEqual(result["strict_raw_acc"], 0)
        self.assertEqual(result["accuracy_reward"], 0.0)
        self.assertEqual(result["stop_reward"], 1.0)
        self.assertEqual(result["score"], 0.25)

    def test_stops_beyond_eight_keep_valid_sequence_and_verified_rewards(self):
        for raw_stop_count in (8, 9, 10, 12):
            with self.subTest(raw_stop_count=raw_stop_count):
                response = (
                    "<think>x" + "<stop>" * raw_stop_count
                    + "</think><final_answer>A</final_answer>"
                )
                result = score(
                    response,
                    verified=True,
                    position=1024,
                    raw=raw_stop_count,
                    ordinal=1,
                    eligible=raw_stop_count,
                    probed=min(raw_stop_count, 8),
                )
                self.assertEqual(result["strict_raw_acc"], 1)
                self.assertEqual(result["verified_stop_score"], 0.0625)
                self.assertEqual(result["accuracy_reward"], 0.5)
                self.assertEqual(result["format_reward"], 0.25)
                self.assertEqual(result["stop_reward"], 0.0625)
                self.assertEqual(result["score"], 0.75)
                self.assertEqual(result["reward_gated"], 0)
                self.assertEqual(
                    result["stop_over_limit"], int(raw_stop_count > 8)
                )
                self.assertEqual(
                    result["overflow_action_penalty_required"],
                    int(raw_stop_count > 8),
                )

    def test_missing_final_still_hard_gates_overflow_rollout(self):
        raw_stop_count = 12
        result = score(
            "<think>x" + "<stop>" * raw_stop_count + "</think>",
            verified=True,
            position=1024,
            raw=raw_stop_count,
            ordinal=1,
            eligible=raw_stop_count,
            probed=8,
        )
        self.assertEqual(result["strict_final_valid"], 0)
        self.assertEqual(result["output_requirement_gated"], 1)
        self.assertEqual(result["reward_gated"], 1)
        self.assertEqual(result["stop_over_limit"], 1)
        self.assertEqual(result["overflow_action_penalty_required"], 1)
        for component in (
            "accuracy_reward",
            "format_reward",
            "stop_reward",
            "score",
        ):
            self.assertEqual(result[component], 0.0)

    def test_metadata_is_fail_closed(self):
        bad_cases = [
            {"verified": True, "position": -1, "raw": 1, "ordinal": 1, "eligible": 1},
            {"verified": True, "position": 0, "raw": 1, "ordinal": -1, "eligible": 1},
            {"verified": True, "position": 0, "raw": 1, "ordinal": 2, "eligible": 1},
            {"verified": False, "position": 0, "raw": 1, "eligible": 1},
            {"raw": -1},
            {"cap": 0},
            {"half_life": 0},
            {"min_separation": 0},
            {
                "verified": True,
                "verified_score": 0.5,
                "position": 1024,
                "raw": 1,
                "ordinal": 1,
                "eligible": 1,
            },
        ]
        for metadata in bad_cases:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    score(self.formatted_correct, **metadata)

    def test_debug_output_is_quiet_by_default(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            score(self.formatted_correct, raw=1, eligible=1)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
