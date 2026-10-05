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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
import math
from numbers import Integral, Real


def _scalar(value):
    item = getattr(value, "item", None)
    return item() if callable(item) else value


class StopNaiveRewardManager:
    """The reward manager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='data_source',
        *,
        max_stop_count,
        stop_reward_half_life_tokens,
        min_verified_stop_separation_tokens,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_stop_count = int(max_stop_count)
        self.stop_reward_half_life_tokens = float(stop_reward_half_life_tokens)
        self.min_verified_stop_separation_tokens = int(
            min_verified_stop_separation_tokens
        )
        if self.max_stop_count < 1:
            raise ValueError(
                f"max_stop_count must be positive, got {self.max_stop_count}"
            )
        if (
            not math.isfinite(self.stop_reward_half_life_tokens)
            or self.stop_reward_half_life_tokens <= 0
        ):
            raise ValueError(
                "stop_reward_half_life_tokens must be finite and positive, got "
                f"{self.stop_reward_half_life_tokens}"
            )
        if self.min_verified_stop_separation_tokens < 1:
            raise ValueError(
                "min_verified_stop_separation_tokens must be positive, got "
                f"{self.min_verified_stop_separation_tokens}"
            )

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            #hard_prob = data_item.batch['hard_prob']
            raw_stop_count = int(data_item.batch['raw_stop_count'].item())
            raw_atomic_stop_count = int(
                data_item.batch['raw_atomic_stop_count'].item()
            )
            illegal_surface_stop_count = int(
                data_item.batch['illegal_surface_stop_count'].item()
            )
            stop_fragment_count = int(
                data_item.batch['stop_fragment_count'].item()
            )
            negative_stop_event_count = int(
                data_item.batch['negative_stop_event_count'].item()
            )
            verified_stop = int(data_item.batch['verified_stop'].item())
            verified_stop_count = int(
                data_item.batch['verified_stop_count'].item()
            )
            verified_stop_score = float(
                data_item.batch['verified_stop_score'].item()
            )
            selected_stop_kind = str(
                _scalar(
                    data_item.non_tensor_batch.get(
                        'selected_stop_kind',
                        'gold_correct' if verified_stop else 'none',
                    )
                )
            )
            selected_stop_credit = float(
                _scalar(
                    data_item.non_tensor_batch.get(
                        'selected_stop_credit',
                        1.0 if verified_stop else 0.0,
                    )
                )
            )
            final_consistency_credit = float(
                _scalar(
                    data_item.non_tensor_batch.get(
                        'final_consistency_credit', 1.0
                    )
                )
            )
            if (
                not math.isfinite(final_consistency_credit)
                or not 0.0 < final_consistency_credit <= 1.0
            ):
                raise RuntimeError(
                    "final_consistency_credit must be finite in (0, 1], got "
                    f"{final_consistency_credit}"
                )
            if verified_stop:
                if selected_stop_kind == 'gold_correct':
                    if selected_stop_credit != 1.0:
                        raise RuntimeError(
                            "gold_correct selected stop requires credit=1"
                        )
                elif selected_stop_kind == 'final_consistent':
                    if not math.isclose(
                        selected_stop_credit,
                        final_consistency_credit,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            "final_consistent selected stop credit must equal "
                            "final_consistency_credit"
                        )
                else:
                    raise RuntimeError(
                        "selected stop requires kind gold_correct or "
                        f"final_consistent, got {selected_stop_kind!r}"
                    )
            elif selected_stop_kind != 'none' or selected_stop_credit != 0.0:
                raise RuntimeError(
                    "unselected row requires kind='none' and credit=0"
                )
            raw_policy_acc = int(data_item.batch['raw_policy_acc'].item())
            raw_controller_terminated = data_item.batch[
                'controller_terminated'
            ].item()
            if isinstance(raw_controller_terminated, Integral):
                controller_terminated = int(raw_controller_terminated)
            elif (
                isinstance(raw_controller_terminated, Real)
                and math.isfinite(float(raw_controller_terminated))
                and float(raw_controller_terminated).is_integer()
            ):
                controller_terminated = int(raw_controller_terminated)
            else:
                raise RuntimeError(
                    "controller_terminated must be an integer scalar, got "
                    f"{raw_controller_terminated!r}"
                )
            raw_controller_wrap_rejected_stop_count = data_item.batch[
                'controller_wrap_rejected_stop_count'
            ].item()
            if isinstance(raw_controller_wrap_rejected_stop_count, Integral):
                controller_wrap_rejected_stop_count = int(
                    raw_controller_wrap_rejected_stop_count
                )
            elif (
                isinstance(raw_controller_wrap_rejected_stop_count, Real)
                and math.isfinite(float(raw_controller_wrap_rejected_stop_count))
                and float(raw_controller_wrap_rejected_stop_count).is_integer()
            ):
                controller_wrap_rejected_stop_count = int(
                    raw_controller_wrap_rejected_stop_count
                )
            else:
                raise RuntimeError(
                    "controller_wrap_rejected_stop_count must be an integer "
                    f"scalar, got {raw_controller_wrap_rejected_stop_count!r}"
                )
            controller_discarded_token_count = int(
                data_item.batch['controller_discarded_token_count'].item()
            )
            controller_forced_suffix_token_count = int(
                data_item.batch[
                    'controller_forced_suffix_token_count'
                ].item()
            )
            physical_response_length = int(
                data_item.batch['physical_response_length'].item()
            )
            physical_raw_stop_count = int(
                data_item.batch['physical_raw_stop_count'].item()
            )
            physical_raw_atomic_stop_count = int(
                data_item.batch['physical_raw_atomic_stop_count'].item()
            )
            physical_illegal_surface_stop_count = int(
                data_item.batch[
                    'physical_illegal_surface_stop_count'
                ].item()
            )
            physical_policy_acc = int(
                data_item.batch['physical_policy_acc'].item()
            )
            verified_stop_position = int(
                data_item.batch['verified_stop_position'].item()
            )
            selected_stop_ordinal = int(
                data_item.batch['selected_stop_ordinal'].item()
            )
            eligible_stop_count = int(data_item.batch['eligible_stop_count'].item())
            probed_stop_count = int(data_item.batch['probed_stop_count'].item())
            recognized_correct_probe_count = int(
                data_item.batch['recognized_correct_probe_count'].item()
            )
            incorrect_stop_count = int(
                data_item.batch['incorrect_stop_count'].item()
            )
            neutral_probe_stop_count = int(
                data_item.batch['neutral_probe_stop_count'].item()
            )
            first_probe_correct = int(
                data_item.batch['first_probe_correct'].item()
            )
            observed_verified_count = int(
                data_item.batch['verified_stop_mask'].sum().item()
            )
            observed_atomic_count = int(
                data_item.batch['atomic_stop_mask'].sum().item()
            )
            if observed_atomic_count != raw_atomic_stop_count:
                raise RuntimeError(
                    "atomic stop mask/count mismatch: "
                    f"{observed_atomic_count} != {raw_atomic_stop_count}"
                )
            if raw_stop_count != (
                raw_atomic_stop_count + illegal_surface_stop_count
            ):
                raise RuntimeError(
                    "generalized stop count mismatch: "
                    f"{raw_stop_count} != {raw_atomic_stop_count} + "
                    f"{illegal_surface_stop_count}"
                )
            observed_event_count = int(
                data_item.batch['stop_event_multiplicity'].sum().item()
            )
            observed_surface_count = int(
                data_item.batch[
                    'surface_stop_event_multiplicity'
                ].sum().item()
            )
            observed_negative_count = int(
                data_item.batch[
                    'negative_stop_event_multiplicity'
                ].sum().item()
            )
            if observed_event_count != raw_stop_count:
                raise RuntimeError(
                    "stop event multiplicity/count mismatch: "
                    f"{observed_event_count} != {raw_stop_count}"
                )
            if observed_surface_count != illegal_surface_stop_count:
                raise RuntimeError(
                    "surface stop multiplicity/count mismatch: "
                    f"{observed_surface_count} != "
                    f"{illegal_surface_stop_count}"
                )
            if observed_negative_count != negative_stop_event_count:
                raise RuntimeError(
                    "negative stop multiplicity/count mismatch: "
                    f"{observed_negative_count} != "
                    f"{negative_stop_event_count}"
                )
            if observed_verified_count != verified_stop_count:
                raise RuntimeError(
                    "verified stop mask/count mismatch: "
                    f"{observed_verified_count} != {verified_stop_count}"
                )
            if controller_terminated != verified_stop:
                raise RuntimeError(
                    "controller termination must exactly match verified stop: "
                    f"{controller_terminated} != {verified_stop}"
                )
            if controller_terminated not in (0, 1):
                raise RuntimeError(
                    "controller_terminated must be binary, got "
                    f"{controller_terminated}"
                )
            if controller_wrap_rejected_stop_count < 0:
                raise RuntimeError(
                    "controller_wrap_rejected_stop_count must be non-negative, "
                    f"got {controller_wrap_rejected_stop_count}"
                )
            if physical_raw_stop_count != (
                physical_raw_atomic_stop_count
                + physical_illegal_surface_stop_count
            ):
                raise RuntimeError(
                    "physical generalized stop count mismatch: "
                    f"{physical_raw_stop_count} != "
                    f"{physical_raw_atomic_stop_count} + "
                    f"{physical_illegal_surface_stop_count}"
                )
            observed_incorrect_stop_count = int(
                data_item.batch['incorrect_stop_aux_mask'].sum().item()
            )
            if observed_incorrect_stop_count != incorrect_stop_count:
                raise RuntimeError(
                    "incorrect stop mask/count mismatch: "
                    f"{observed_incorrect_stop_count} != {incorrect_stop_count}"
                )
            if (
                recognized_correct_probe_count
                + incorrect_stop_count
                + neutral_probe_stop_count
                != probed_stop_count
            ):
                raise RuntimeError(
                    "correct/incorrect/neutral probe counts must sum to "
                    "probed_stop_count: "
                    f"{recognized_correct_probe_count} + "
                    f"{incorrect_stop_count} + {neutral_probe_stop_count} != "
                    f"{probed_stop_count}"
                )
            if (
                selected_stop_kind == 'gold_correct'
                and recognized_correct_probe_count < verified_stop_count
            ):
                raise RuntimeError(
                    "gold-correct stop count cannot exceed recognized-correct "
                    "probe count"
                )
            observed_verified_score = float(
                data_item.batch['verified_stop_aux_weight'].sum().item()
            )
            if not math.isclose(
                observed_verified_score,
                verified_stop_score,
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise RuntimeError(
                    "verified stop weights/score mismatch: "
                    f"{observed_verified_score} != {verified_stop_score}"
                )
            if verified_stop:
                earliness = 2.0 ** (
                    -float(verified_stop_position)
                    / self.stop_reward_half_life_tokens
                )
                expected_verified_score = earliness
                if not math.isclose(
                    verified_stop_score,
                    expected_verified_score,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError(
                        "selected stop score violates the shared exponential "
                        "earliness contract: "
                        f"{verified_stop_score} != {expected_verified_score}"
                    )

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(
                data_item.batch['attention_mask'][prompt_length:].sum().item()
            )
            valid_response_ids = response_ids[:valid_response_length]

            effective_response_length = int(
                data_item.batch['effective_response_length'].item()
            )
            accepted_prefix_length = int(
                data_item.batch['accepted_stop_prefix_length'].item()
            )
            if valid_response_length != effective_response_length:
                raise RuntimeError(
                    "attention/effective response length mismatch: "
                    f"{valid_response_length} != {effective_response_length}"
                )
            if controller_terminated:
                if verified_stop_position < 0:
                    raise RuntimeError(
                        "controller-terminated row requires a verified position"
                    )
                if accepted_prefix_length != verified_stop_position + 1:
                    raise RuntimeError(
                        "accepted prefix must end at the verified stop action"
                    )
                if controller_forced_suffix_token_count <= 0:
                    raise RuntimeError(
                        "controller-terminated row requires a forced final suffix"
                    )
                if effective_response_length != (
                    accepted_prefix_length
                    + controller_forced_suffix_token_count
                ):
                    raise RuntimeError(
                        "controller effective length does not match prefix + suffix"
                    )
                if physical_response_length != (
                    accepted_prefix_length
                    + controller_discarded_token_count
                ):
                    raise RuntimeError(
                        "physical length does not match prefix + discarded tail"
                    )
            else:
                if (
                    controller_discarded_token_count != 0
                    or controller_forced_suffix_token_count != 0
                ):
                    raise RuntimeError(
                        "non-terminated row cannot contain controller splice tokens"
                    )
                if effective_response_length != physical_response_length:
                    raise RuntimeError(
                        "non-terminated physical/effective lengths must match"
                    )
                if raw_policy_acc != physical_policy_acc:
                    raise RuntimeError(
                        "non-terminated physical/effective accuracy must match"
                    )

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                verified_stop=verified_stop,
                verified_stop_count=verified_stop_count,
                verified_stop_score=verified_stop_score,
                raw_stop_count=raw_stop_count,
                raw_atomic_stop_count=raw_atomic_stop_count,
                illegal_surface_stop_count=illegal_surface_stop_count,
                stop_fragment_count=stop_fragment_count,
                verified_stop_position=verified_stop_position,
                selected_stop_ordinal=selected_stop_ordinal,
                eligible_stop_count=eligible_stop_count,
                probed_stop_count=probed_stop_count,
                max_stop_count=self.max_stop_count,
                stop_reward_half_life_tokens=self.stop_reward_half_life_tokens,
                min_verified_stop_separation_tokens=(
                    self.min_verified_stop_separation_tokens
                ),
                selected_stop_kind=selected_stop_kind,
                selected_stop_credit=selected_stop_credit,
                final_consistency_credit=final_consistency_credit,
                extra_info=extra_info,
                tokenizer=self.tokenizer,
            )
            strict_raw_acc = int(score.get('strict_raw_acc', score.get('acc', 0)))
            if strict_raw_acc != raw_policy_acc:
                raise RuntimeError(
                    "rollout/reward strict raw accuracy mismatch: "
                    f"rollout={raw_policy_acc}, reward={strict_raw_acc}"
                )
            if controller_terminated:
                if selected_stop_kind == 'gold_correct':
                    if strict_raw_acc != 1:
                        raise RuntimeError(
                            "gold-correct controller stop must produce a "
                            "strict correct final"
                        )
                elif selected_stop_kind == 'final_consistent':
                    physical_final_choice = _scalar(
                        data_item.non_tensor_batch.get('physical_final_choice')
                    )
                    effective_final_choice = _scalar(
                        data_item.non_tensor_batch.get('effective_final_choice')
                    )
                    selected_probe_answer = _scalar(
                        data_item.non_tensor_batch.get('selected_probe_answer')
                    )
                    if (
                        physical_final_choice is None
                        or effective_final_choice != physical_final_choice
                        or selected_probe_answer != physical_final_choice
                        or strict_raw_acc != physical_policy_acc
                    ):
                        raise RuntimeError(
                            "final-consistent controller stop must preserve "
                            "the strict natural final answer and accuracy"
                        )
            score.update({
                'response_length': valid_response_length,
                'effective_response_length': effective_response_length,
                'raw_policy_acc': raw_policy_acc,
                'controller_terminated': controller_terminated,
                'controller_wrap_rejected_stop_count': (
                    controller_wrap_rejected_stop_count
                ),
                'controller_discarded_token_count': (
                    controller_discarded_token_count
                ),
                'controller_forced_suffix_token_count': (
                    controller_forced_suffix_token_count
                ),
                'physical_response_length': physical_response_length,
                'physical_raw_stop_count': physical_raw_stop_count,
                'physical_raw_atomic_stop_count': (
                    physical_raw_atomic_stop_count
                ),
                'physical_illegal_surface_stop_count': (
                    physical_illegal_surface_stop_count
                ),
                'physical_policy_acc': physical_policy_acc,
                'raw_stop_count': raw_stop_count,
                'raw_atomic_stop_count': raw_atomic_stop_count,
                'illegal_surface_stop_count': illegal_surface_stop_count,
                'stop_fragment_count': stop_fragment_count,
                'negative_stop_event_count': negative_stop_event_count,
                'verified_stop_count': verified_stop_count,
                'verified_stop_score': verified_stop_score,
                'verified_stop_position': verified_stop_position,
                'selected_stop_ordinal': selected_stop_ordinal,
                'selected_stop_kind': selected_stop_kind,
                'selected_stop_credit': selected_stop_credit,
                'final_consistency_credit': final_consistency_credit,
                'eligible_stop_count': eligible_stop_count,
                'probed_stop_count': probed_stop_count,
                'recognized_correct_probe_count': (
                    recognized_correct_probe_count
                ),
                'incorrect_stop_count': incorrect_stop_count,
                'neutral_probe_stop_count': neutral_probe_stop_count,
                'first_probe_correct': first_probe_correct,
            })

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
