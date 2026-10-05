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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import pickle
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
WorkerType = Type[Worker]
import lightgbm as lgb

from .estar_a70_shared import (
    A70_FEATURES,
    A70_FEATURE_SCHEMA_SHA256,
    A70_FEATURE_SCHEMA_VERSION,
)

STOP_CLASSIFIER_FEATURE_NAMES = tuple(A70_FEATURES)
STOP_CLASSIFIER_FEATURE_DIM = len(STOP_CLASSIFIER_FEATURE_NAMES)
STOP_CLASSIFIER_SCHEMA_VERSION = 4
STOP_CLASSIFIER_TARGET_FINAL = 'probe_matches_untouched_physical_final'
STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL = (
    'probe_matches_gold_or_untouched_physical_final'
)
# Backward-compatible import name for offline tools that inspect the original
# final-consistency classifier.
STOP_CLASSIFIER_TARGET = STOP_CLASSIFIER_TARGET_FINAL
STOP_CLASSIFIER_TARGETS = frozenset((
    STOP_CLASSIFIER_TARGET_FINAL,
    STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL,
))

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    response_mask = compute_response_mask(data)

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Select the policy-loss mask, excluding any forced response suffix."""
    responses = data.batch['responses']
    response_length = responses.size(1)
    if 'actor_loss_mask' in data.batch.keys():
        actor_loss_mask = data.batch['actor_loss_mask']
        if actor_loss_mask.shape != responses.shape:
            raise ValueError(
                'actor_loss_mask must have the same shape as responses: '
                f'{tuple(actor_loss_mask.shape)} != {tuple(responses.shape)}')
        return actor_loss_mask
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def _selected_stop_metadata_arrays(
    data: DataProto,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return and validate selected kind, quality credit, and tier boundary."""

    raw_kinds = data.non_tensor_batch.get("selected_stop_kind")
    raw_credits = data.non_tensor_batch.get("selected_stop_credit")
    if (raw_kinds is None) != (raw_credits is None):
        raise KeyError(
            "selected_stop_kind and selected_stop_credit must be provided together"
        )
    if raw_kinds is None:
        # Keep legacy synthetic fixtures import-compatible.  Production v18r2
        # rollout always emits the explicit fields.
        if "verified_stop_count" not in data.batch:
            raise KeyError(
                "selected stop metadata requires verified_stop_count"
            )
        selected_counts = (
            data.batch["verified_stop_count"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        kinds = np.where(selected_counts > 0, "gold_correct", "none").astype(
            object
        )
        credits = np.where(selected_counts > 0, 1.0, 0.0).astype(np.float64)
    else:
        kinds = np.asarray(raw_kinds, dtype=object)
        credits = np.asarray(raw_credits, dtype=np.float64)
    if kinds.ndim != 1 or kinds.size != row_count:
        raise ValueError(
            "selected_stop_kind must contain one value per rollout: "
            f"shape={kinds.shape}, rows={row_count}"
        )
    if credits.ndim != 1 or credits.size != row_count:
        raise ValueError(
            "selected_stop_credit must contain one value per rollout: "
            f"shape={credits.shape}, rows={row_count}"
        )
    kinds = np.asarray([str(value) for value in kinds.tolist()], dtype=object)
    raw_consistency = data.non_tensor_batch.get("final_consistency_credit")
    if raw_consistency is None:
        consistency = np.full(row_count, 1.0, dtype=np.float64)
    else:
        consistency = np.asarray(raw_consistency, dtype=np.float64)
        if consistency.ndim == 0:
            consistency = np.full(
                row_count, float(consistency.item()), dtype=np.float64
            )
    if consistency.ndim != 1 or consistency.size != row_count:
        raise ValueError(
            "final_consistency_credit must contain one value per rollout: "
            f"shape={consistency.shape}, rows={row_count}"
        )
    if (
        not np.all(np.isfinite(credits))
        or not np.all(np.isfinite(consistency))
        or np.any(consistency <= 0.0)
        or np.any(consistency > 1.0)
    ):
        raise ValueError("selected stop credits must be finite and shared credit in (0,1]")
    if not np.allclose(consistency, consistency[0], rtol=0.0, atol=0.0):
        raise ValueError("final_consistency_credit must be constant across a batch")

    if "verified_stop_count" not in data.batch:
        raise KeyError("selected stop metadata requires verified_stop_count")
    selected_counts = (
        data.batch["verified_stop_count"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    if selected_counts.ndim != 1 or selected_counts.size != row_count:
        raise ValueError("verified_stop_count must contain one value per rollout")
    none_rows = kinds == "none"
    gold_rows = kinds == "gold_correct"
    fallback_rows = kinds == "final_consistent"
    if np.any(~(none_rows | gold_rows | fallback_rows)):
        raise ValueError(f"invalid selected_stop_kind values: {kinds.tolist()}")
    if np.any(none_rows & ((selected_counts != 0) | (credits != 0.0))):
        raise ValueError("kind=none requires no selected stop and zero credit")
    if np.any(gold_rows & ((selected_counts != 1) | (credits != 1.0))):
        raise ValueError("kind=gold_correct requires one selected stop and credit=1")
    if np.any(fallback_rows & (selected_counts != 1)) or not np.allclose(
        credits[fallback_rows],
        consistency[fallback_rows],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "kind=final_consistent requires one selected stop and shared credit"
        )
    return kinds, credits, consistency


def center_stop_action_weights_by_prompt_in_place(
    data: DataProto,
    *,
    verified_stop_aux_loss_coef: float,
    incorrect_stop_aux_loss_coef: float,
    expected_group_size: int,
) -> dict[str, float]:
    """Center labeled stop-action weights within each prompt rollout group.

    The rollout constructs one positive mask for an oracle-verified stop action
    and one negative mask for every recognized-wrong legal probe.  Immediately
    before actor dispatch, replace their raw earliness weights ``e(t)`` with the
    prompt-group-centered action advantages

    ``A_stop(t) = e(t) * (y - y_bar)``.

    Here ``y`` is one for an action in ``verified_stop_mask`` and zero for an
    action in ``incorrect_stop_aux_mask``.  ``y_bar`` is the mean over all
    labeled stop actions belonging to the same ``uid``; neutral probes do not
    enter its denominator.  Consequently, correct actions retain positive
    weights ``e(t) * (1 - y_bar)`` and wrong actions retain negative-objective
    magnitudes ``e(t) * y_bar``.  A group containing only one label class has
    zero centered advantage, so both action masks and weights are cleared to
    satisfy the actor's strictly-positive-weight contract.

    This operation is intentionally destructive and actor-only.  Call it after
    raw rollout validation/auditing and immediately before ``update_actor``.
    Sequence rewards, ordinary GRPO advantages, generalized stop-like losses,
    and post-accepted-tail losses are left unchanged.
    """

    verified_coef = float(verified_stop_aux_loss_coef)
    incorrect_coef = float(incorrect_stop_aux_loss_coef)
    for name, coefficient in (
        ("verified_stop_aux_loss_coef", verified_coef),
        ("incorrect_stop_aux_loss_coef", incorrect_coef),
    ):
        if not np.isfinite(coefficient) or coefficient <= 0:
            raise ValueError(
                f"{name} must be finite and strictly positive when prompt "
                f"group centering is enabled, got {coefficient}"
            )
    if verified_coef != incorrect_coef:
        raise ValueError(
            "prompt-group-centered stop actions require equal positive and "
            "negative coefficients: verified_stop_aux_loss_coef="
            f"{verified_coef}, incorrect_stop_aux_loss_coef={incorrect_coef}"
        )
    if (
        isinstance(expected_group_size, bool)
        or int(expected_group_size) != expected_group_size
        or int(expected_group_size) < 1
    ):
        raise ValueError(
            "expected_group_size must be a positive integer, got "
            f"{expected_group_size}"
        )
    expected_group_size = int(expected_group_size)

    required_batch_keys = {
        "verified_stop_mask",
        "verified_stop_aux_weight",
        "incorrect_stop_aux_mask",
        "incorrect_stop_aux_weight",
    }
    missing_batch_keys = sorted(required_batch_keys.difference(data.batch.keys()))
    if missing_batch_keys:
        raise KeyError(
            "prompt-centered stop actions are missing batch keys: "
            f"{missing_batch_keys}"
        )
    if "uid" not in data.non_tensor_batch:
        raise KeyError("prompt-centered stop actions require uid")

    verified_mask = data.batch["verified_stop_mask"]
    verified_weight = data.batch["verified_stop_aux_weight"]
    incorrect_mask = data.batch["incorrect_stop_aux_mask"]
    incorrect_weight = data.batch["incorrect_stop_aux_weight"]
    reference_shape = verified_mask.shape
    if len(reference_shape) != 2:
        raise ValueError(
            "stop action masks must be rank-two tensors, got "
            f"{tuple(reference_shape)}"
        )
    for name, tensor in (
        ("verified_stop_aux_weight", verified_weight),
        ("incorrect_stop_aux_mask", incorrect_mask),
        ("incorrect_stop_aux_weight", incorrect_weight),
    ):
        if tensor.shape != reference_shape:
            raise ValueError(
                f"{name} must have shape {tuple(reference_shape)}, got "
                f"{tuple(tensor.shape)}"
            )
    for name, mask in (
        ("verified_stop_mask", verified_mask),
        ("incorrect_stop_aux_mask", incorrect_mask),
    ):
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError(f"{name} must be binary")
    if torch.any((verified_mask > 0) & (incorrect_mask > 0)):
        raise ValueError(
            "verified and incorrect stop action masks must be disjoint"
        )
    for name, mask, weight in (
        ("verified_stop_aux_weight", verified_mask, verified_weight),
        ("incorrect_stop_aux_weight", incorrect_mask, incorrect_weight),
    ):
        if not torch.is_floating_point(weight):
            raise ValueError(f"{name} must be floating point")
        if not torch.all(torch.isfinite(weight)):
            raise ValueError(f"{name} must be finite")
        if torch.any(weight < 0):
            raise ValueError(f"{name} must be non-negative")
        if torch.any(weight * (1 - mask.to(dtype=weight.dtype)) != 0):
            raise ValueError(f"{name} must be zero outside its action mask")
        if torch.any((mask > 0) & (weight <= 0)):
            raise ValueError(
                f"{name} must be strictly positive inside its action mask"
            )

    row_count = int(reference_shape[0])
    uid_values = np.asarray(data.non_tensor_batch["uid"], dtype=object)
    if uid_values.ndim != 1 or int(uid_values.size) != row_count:
        raise ValueError(
            "uid must contain one prompt-group identifier per rollout row: "
            f"shape={uid_values.shape}, rows={row_count}"
        )

    group_rows: dict[str, list[int]] = defaultdict(list)
    for row_index, uid in enumerate(uid_values.tolist()):
        if uid is None or not str(uid).strip():
            raise ValueError(
                "uid values must be non-empty prompt-group identifiers: "
                f"row={row_index}, uid={uid!r}"
            )
        group_rows[str(uid)].append(row_index)
    invalid_group_sizes = {
        uid: len(rows)
        for uid, rows in group_rows.items()
        if len(rows) != expected_group_size
    }
    if invalid_group_sizes:
        raise ValueError(
            "each uid must contain exactly rollout.n rows before centered "
            f"stop actor dispatch: expected={expected_group_size}, "
            f"observed={invalid_group_sizes}"
        )

    correct_counts = verified_mask.sum(dim=-1).detach().cpu().tolist()
    wrong_counts = incorrect_mask.sum(dim=-1).detach().cpu().tolist()
    group_scales: list[tuple[list[int], float, float]] = []
    mixed_group_count = 0
    all_correct_group_count = 0
    all_wrong_group_count = 0
    empty_group_count = 0
    for rows in group_rows.values():
        correct_count = sum(int(correct_counts[row]) for row in rows)
        wrong_count = sum(int(wrong_counts[row]) for row in rows)
        labeled_count = correct_count + wrong_count
        if labeled_count == 0:
            empty_group_count += 1
            correct_scale = 0.0
            wrong_scale = 0.0
        elif wrong_count == 0:
            all_correct_group_count += 1
            correct_scale = 0.0
            wrong_scale = 0.0
        elif correct_count == 0:
            all_wrong_group_count += 1
            correct_scale = 0.0
            wrong_scale = 0.0
        else:
            mixed_group_count += 1
            y_bar = float(correct_count) / float(labeled_count)
            correct_scale = 1.0 - y_bar
            wrong_scale = y_bar
        group_scales.append((rows, correct_scale, wrong_scale))

    for rows, correct_scale, wrong_scale in group_scales:
        row_indices = torch.as_tensor(
            rows, device=verified_mask.device, dtype=torch.long
        )
        if correct_scale == 0.0:
            verified_mask[row_indices] = 0
            verified_weight[row_indices] = 0.0
        else:
            verified_weight[row_indices] = (
                verified_weight[row_indices] * correct_scale
            )
        if wrong_scale == 0.0:
            incorrect_mask[row_indices] = 0
            incorrect_weight[row_indices] = 0.0
        else:
            incorrect_weight[row_indices] = (
                incorrect_weight[row_indices] * wrong_scale
            )

    centered_positive_count = int(verified_mask.sum().detach().cpu().item())
    centered_negative_count = int(incorrect_mask.sum().detach().cpu().item())
    original_action_count = sum(int(value) for value in correct_counts) + sum(
        int(value) for value in wrong_counts
    )
    centered_zero_count = (
        original_action_count - centered_positive_count - centered_negative_count
    )
    positive_mass = float(verified_weight.sum().detach().cpu().item())
    negative_mass = float(incorrect_weight.sum().detach().cpu().item())
    return {
        "stop_centering/group_count": float(len(group_rows)),
        "stop_centering/mixed_group_count": float(mixed_group_count),
        "stop_centering/all_correct_group_count": float(
            all_correct_group_count
        ),
        "stop_centering/all_wrong_group_count": float(all_wrong_group_count),
        "stop_centering/empty_group_count": float(empty_group_count),
        "stop_centering/positive_action_count": float(centered_positive_count),
        "stop_centering/negative_action_count": float(centered_negative_count),
        "stop_centering/zero_action_count": float(centered_zero_count),
        "stop_centering/signed_weight_mass": positive_mass - negative_mass,
        "stop_centering/absolute_weight_mass": positive_mass + negative_mass,
    }

import numpy as np


def _get_val(x):
    # 兼容你现在的结构：elem 是 {'value': ndarray}
    return x['value'] if isinstance(x, dict) and 'value' in x else x


def collect_xy_from_non_tensor(
    non_tensor_batch,
    key_x='feature_x',
    key_y='feature_y',
    *,
    feature_dim=30,
):
    """Collect variable-length per-candidate feature matrices and labels."""
    feature_dim = int(feature_dim)
    if feature_dim <= 0:
        raise ValueError('feature_dim must be positive')
    if key_x not in non_tensor_batch or key_y not in non_tensor_batch:
        return np.zeros((0, feature_dim), np.float32), np.zeros((0,), np.int64)
    fx = non_tensor_batch[key_x]
    fy = non_tensor_batch[key_y]
    assert len(fx) == len(fy), f"len mismatch: {len(fx)} vs {len(fy)}"


    X_parts, y_parts = [], []
    for i in range(len(fx)):
        Xi = _get_val(fx[i])
        yi = _get_val(fy[i])
        if Xi is None or yi is None:
            continue
        Xi = np.asarray(Xi)
        yi = np.asarray(yi).reshape(-1)  # y 一维化
        if Xi.size == 0 or yi.size == 0:
            continue
        if Xi.ndim == 1:
            if Xi.size != feature_dim:
                raise ValueError(f'feature_x[{i}] must have {feature_dim} columns, got shape {Xi.shape}')
            Xi = Xi.reshape(1, feature_dim)
        if Xi.ndim != 2 or Xi.shape[1] != feature_dim:
            raise ValueError(f'feature_x[{i}] must have shape (N, {feature_dim}), got {Xi.shape}')
        # 兜底对齐（有时某卡 X/y 可能不等长）
        n = min(Xi.shape[0], yi.shape[0])
        if n == 0:
            continue
        Xi = Xi[:n].astype(np.float32, copy=False)
        yi = yi[:n].astype(np.int64, copy=False)
        valid = np.isin(yi, (0, 1))
        if np.any(valid):
            X_parts.append(Xi[valid])
            y_parts.append(yi[valid])


    if not X_parts:
        return np.zeros((0, feature_dim), np.float32), np.zeros((0,), np.int64)


    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    return X_all, y_all


def dedup_by_X(X, y, decimals=8, strategy='majority'):
    """
    只按 X 去重（行完全相等）。decimals 用来做数值容忍（先 round 再判等）。
    strategy:
      - 'first'    : 保留该 X 的第一条 y
      - 'majority' : 二分类取多数票；平票取第一条
    返回: X_uniq, y_uniq, info(dict)
    """
    if X.size == 0:
        return X, y, {'n_before': 0, 'n_after': 0, 'duplicates_removed': 0, 'clash_groups': 0}


    # 容错：先四舍五入再做唯一化，避免浮点微抖动造成“看起来一样但比特不同”
    X_key = np.round(X, decimals=decimals) if decimals is not None else X


    # 把每行视作字节串做唯一化（比 pandas.drop_duplicates 更轻）
    keys = np.ascontiguousarray(X_key).view(
        np.dtype((np.void, X_key.dtype.itemsize * X_key.shape[1]))
    ).reshape(-1)
    _, idx, inv, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)


    X_uniq = X[idx]
    # The first label is already correct for every singleton group.  Only
    # duplicate groups need a scan, avoiding an O(unique_rows * rows) loop for
    # the overwhelmingly common all-unique online batches.
    y_uniq = y[idx].copy()


    clash_groups = 0
    for g in np.flatnonzero(counts > 1):
        mask = (inv == g)
        vals = y[mask]
        if strategy == 'first':
            y_uniq[g] = vals[0]
        elif strategy == 'majority':
            ones = int(vals.sum())
            zeros = int(mask.sum() - ones)
            if ones > zeros:
                y_uniq[g] = 1
            elif ones < zeros:
                y_uniq[g] = 0
            else:
                y_uniq[g] = vals[0]  # 平票取第一条
            if np.any(vals != vals[0]):
                clash_groups += 1
        else:
            raise ValueError("strategy must be 'first' or 'majority'")


    info = {
        'n_before': int(len(y)),
        'n_after': int(len(y_uniq)),
        'duplicates_removed': int(len(y) - len(y_uniq)),
        'clash_groups': int(clash_groups),
    }
    return X_uniq, y_uniq, info


class StopClassifierState:
    """Cumulative, checkpointable LightGBM state updated once per rollout batch.

    The target and physical-record feature keys are explicit per instance.
    Every physical probe row is retained, including duplicate feature vectors.
    This object is not part of the RL stop decision, reward, or gradient path.
    Online fitting is disabled by default and must be explicitly opted into
    with trainer.train_stop_classifier_online=true.
    """

    def __init__(
        self,
        params,
        feature_dim=30,
        *,
        target=STOP_CLASSIFIER_TARGET_FINAL,
        key_x='feature_x',
        key_y='feature_y',
        metric_prefix='classifier',
    ):
        self.params = dict(params)
        self.feature_dim = int(feature_dim)
        self.feature_names = (
            STOP_CLASSIFIER_FEATURE_NAMES
            if self.feature_dim == STOP_CLASSIFIER_FEATURE_DIM
            else tuple(f"legacy_feature_{index}" for index in range(self.feature_dim))
        )
        if target not in STOP_CLASSIFIER_TARGETS:
            raise ValueError(f'unsupported stop-classifier target: {target!r}')
        if not key_x or not key_y or not metric_prefix:
            raise ValueError('classifier keys and metric_prefix must be non-empty')
        self.target = str(target)
        self.key_x = str(key_x)
        self.key_y = str(key_y)
        self.metric_prefix = str(metric_prefix).rstrip('/')
        self.feature_x = np.zeros((0, self.feature_dim), dtype=np.float32)
        self.feature_y = np.zeros((0,), dtype=np.int64)
        self.model = None
        self.update_count = 0
        self.generation_batch_count = 0

    @property
    def is_trained(self):
        return self.model is not None

    def update(self, non_tensor_batch):
        X_batch, y_batch = collect_xy_from_non_tensor(
            non_tensor_batch,
            key_x=self.key_x,
            key_y=self.key_y,
            feature_dim=self.feature_dim,
        )
        _batch_unique_x, _batch_unique_y, batch_info = dedup_by_X(
            X_batch, y_batch, decimals=8, strategy='majority')

        prequential = self._prequential_metrics(X_batch, y_batch)

        if len(y_batch):
            X_combined = np.concatenate((self.feature_x, X_batch), axis=0)
            y_combined = np.concatenate((self.feature_y, y_batch), axis=0)
            _cumulative_unique_x, _cumulative_unique_y, cumulative_info = dedup_by_X(
                X_combined, y_combined, decimals=8, strategy='majority')
            # Duplicate feature vectors are distinct physical probe records.
            # Retain every one for fitting; deduplication is telemetry only.
            self.feature_x = X_combined
            self.feature_y = y_combined
        else:
            cumulative_info = {
                'n_before': int(len(self.feature_y)),
                'n_after': int(len(self.feature_y)),
                'duplicates_removed': 0,
                'clash_groups': 0,
            }

        class_counts = (np.bincount(self.feature_y, minlength=2)
                        if len(self.feature_y) else np.zeros(2, dtype=int))
        updated = False
        # One-class LightGBM models cannot make a meaningful stop decision.
        # Keep accumulating samples and fit only after both classes appear.
        if len(y_batch) and np.unique(self.feature_y).size == 2:
            fit_started_at = time.monotonic()
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(self.feature_x, self.feature_y)
            fit_seconds = time.monotonic() - fit_started_at
            self.update_count += 1
            updated = True
        else:
            fit_seconds = 0.0
        self.generation_batch_count += 1

        p = self.metric_prefix
        metrics = {
            f'{p}/batch_samples_raw': int(batch_info['n_before']),
            f'{p}/batch_unique_feature_rows': int(batch_info['n_after']),
            f'{p}/batch_duplicate_rows_observed': int(batch_info['duplicates_removed']),
            f'{p}/cumulative_samples': int(len(self.feature_y)),
            f'{p}/cumulative_unique_feature_rows': int(cumulative_info['n_after']),
            f'{p}/cumulative_duplicate_rows_observed': int(
                cumulative_info['duplicates_removed']
            ),
            f'{p}/class_0_samples': int(class_counts[0]),
            f'{p}/class_1_samples': int(class_counts[1]),
            f'{p}/updated': int(updated),
            f'{p}/trained': int(self.is_trained),
            f'{p}/update_count': int(self.update_count),
            f'{p}/generation_batch_count': int(self.generation_batch_count),
            f'{p}/batch_positive_rate': (
                float(np.mean(y_batch)) if len(y_batch) else 0.0
            ),
            f'{p}/fit_seconds': float(fit_seconds),
        }
        metrics.update(prequential)
        return metrics

    def _prequential_metrics(self, feature_x, feature_y):
        """Evaluate the incoming batch before fitting it (no train/eval leakage)."""
        p = self.metric_prefix
        metrics = {
            f'{p}/prequential_rows': 0,
            f'{p}/prequential_available': 0,
            f'{p}/prequential_auc_defined': 0,
            f'{p}/prequential_auc': 0.0,
            f'{p}/prequential_brier': 0.0,
            f'{p}/prequential_logloss': 0.0,
        }
        for threshold in (0.90, 0.95, 0.99):
            suffix = str(threshold).replace('.', '')
            metrics[f'{p}/prequential_trigger_rate_{suffix}'] = 0.0
            metrics[f'{p}/prequential_precision_{suffix}'] = 0.0
            metrics[f'{p}/prequential_recall_{suffix}'] = 0.0
        if self.model is None or not len(feature_y):
            return metrics

        y = np.asarray(feature_y, dtype=np.int64)
        probabilities = np.asarray(
            self.model.predict_proba(feature_x)[:, 1], dtype=np.float64
        )
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        metrics[f'{p}/prequential_rows'] = int(len(y))
        metrics[f'{p}/prequential_available'] = 1
        metrics[f'{p}/prequential_brier'] = float(np.mean((probabilities - y) ** 2))
        metrics[f'{p}/prequential_logloss'] = float(
            -np.mean(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))
        )
        positives = int(np.sum(y == 1))
        negatives = int(np.sum(y == 0))
        if positives and negatives:
            order = np.argsort(probabilities, kind='mergesort')
            sorted_scores = probabilities[order]
            ranks = np.empty(len(y), dtype=np.float64)
            cursor = 0
            while cursor < len(y):
                end = cursor + 1
                while end < len(y) and sorted_scores[end] == sorted_scores[cursor]:
                    end += 1
                ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
                cursor = end
            positive_rank_sum = float(np.sum(ranks[y == 1]))
            metrics[f'{p}/prequential_auc'] = (
                positive_rank_sum - positives * (positives + 1) / 2.0
            ) / (positives * negatives)
            metrics[f'{p}/prequential_auc_defined'] = 1
        for threshold in (0.90, 0.95, 0.99):
            suffix = str(threshold).replace('.', '')
            predicted = probabilities >= threshold
            triggered = int(np.sum(predicted))
            true_positive = int(np.sum(predicted & (y == 1)))
            metrics[f'{p}/prequential_trigger_rate_{suffix}'] = (
                triggered / len(y)
            )
            metrics[f'{p}/prequential_precision_{suffix}'] = (
                true_positive / triggered if triggered else 0.0
            )
            metrics[f'{p}/prequential_recall_{suffix}'] = (
                true_positive / positives if positives else 0.0
            )
        return metrics

    def predict(self, feature_x):
        if self.model is None:
            raise RuntimeError('stop classifier has not been trained on both classes yet')
        feature_x = np.asarray(feature_x, dtype=np.float32)
        if feature_x.ndim == 1:
            feature_x = feature_x.reshape(1, -1)
        if feature_x.ndim != 2 or feature_x.shape[1] != self.feature_dim:
            raise ValueError(
                f'feature_x must have shape (N, {self.feature_dim}), got {feature_x.shape}')
        return self.model.predict(feature_x)

    def predict_proba(self, feature_x):
        if self.model is None:
            raise RuntimeError('stop classifier has not been trained on both classes yet')
        feature_x = np.asarray(feature_x, dtype=np.float32)
        if feature_x.ndim == 1:
            feature_x = feature_x.reshape(1, -1)
        if feature_x.ndim != 2 or feature_x.shape[1] != self.feature_dim:
            raise ValueError(
                f'feature_x must have shape (N, {self.feature_dim}), got {feature_x.shape}')
        return self.model.predict_proba(feature_x)

    def save(self, path):
        state = {
            'schema_version': STOP_CLASSIFIER_SCHEMA_VERSION,
            'target': self.target,
            'source_records': 'all_physical_records_by_candidate',
            'params': self.params,
            'feature_dim': self.feature_dim,
            'feature_x': self.feature_x,
            'feature_y': self.feature_y,
            'model': self.model,
            'update_count': self.update_count,
            'generation_batch_count': self.generation_batch_count,
            'feature_names': self.feature_names,
            'feature_schema_version': (
                A70_FEATURE_SCHEMA_VERSION
                if self.feature_dim == STOP_CLASSIFIER_FEATURE_DIM
                else 'legacy_30_compat'
            ),
            'feature_schema_sha256': (
                A70_FEATURE_SCHEMA_SHA256
                if self.feature_dim == STOP_CLASSIFIER_FEATURE_DIM
                else None
            ),
        }
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        if int(state.get('schema_version', -1)) != STOP_CLASSIFIER_SCHEMA_VERSION:
            raise ValueError(f'unsupported stop-classifier schema at {path}')
        if state.get('target') != self.target:
            raise ValueError(f'stop-classifier target mismatch at {path}')
        if state.get('source_records') != 'all_physical_records_by_candidate':
            raise ValueError(f'stop-classifier physical-record contract mismatch at {path}')
        feature_dim = int(state['feature_dim'])
        expected_names = (
            STOP_CLASSIFIER_FEATURE_NAMES
            if feature_dim == STOP_CLASSIFIER_FEATURE_DIM
            else tuple(f"legacy_feature_{index}" for index in range(feature_dim))
        )
        if tuple(state.get('feature_names', ())) != expected_names:
            raise ValueError(f'stop-classifier feature schema mismatch at {path}')
        if feature_dim == STOP_CLASSIFIER_FEATURE_DIM and (
            state.get('feature_schema_version') != A70_FEATURE_SCHEMA_VERSION
            or state.get('feature_schema_sha256') != A70_FEATURE_SCHEMA_SHA256
        ):
            raise ValueError(f'stop-classifier A70 schema fingerprint mismatch at {path}')
        feature_x = np.asarray(state['feature_x'], dtype=np.float32)
        feature_y = np.asarray(state['feature_y'], dtype=np.int64).reshape(-1)
        if (feature_x.ndim != 2 or feature_x.shape[1] != feature_dim
                or len(feature_x) != len(feature_y)):
            raise ValueError(f'invalid stop-classifier checkpoint at {path}')
        self.params = dict(state['params'])
        self.feature_dim = feature_dim
        self.feature_names = expected_names
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.model = state['model']
        self.update_count = int(state.get('update_count', int(self.model is not None)))
        self.generation_batch_count = int(state.get('generation_batch_count', 0))
        return self


def maybe_update_stop_classifier(classifier_state, non_tensor_batch, enabled):
    """Run the online A70 classifier fit only when explicitly enabled.

    The default RL path must not collect cumulative classifier samples or call
    LightGBM. Probe metrics remain independent and are computed by the caller.
    """
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError(
            'trainer.train_stop_classifier_online must be a boolean, '
            f'got {type(enabled).__name__}: {enabled!r}'
        )
    if not enabled:
        p = classifier_state.metric_prefix
        return {
            f'{p}/online_training_enabled': 0,
            f'{p}/cumulative_samples': int(len(classifier_state.feature_y)),
            f'{p}/updated': 0,
            f'{p}/trained': int(classifier_state.is_trained),
            f'{p}/update_count': int(classifier_state.update_count),
        }

    metrics = classifier_state.update(non_tensor_batch)
    p = classifier_state.metric_prefix
    metrics[f'{p}/online_training_enabled'] = 1
    metrics[f'{p}/target_is_final_consistency'] = int(
        classifier_state.target == STOP_CLASSIFIER_TARGET_FINAL
    )
    metrics[f'{p}/target_is_gold_or_final'] = int(
        classifier_state.target == STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL
    )
    metrics[f'{p}/source_is_all_physical_records'] = 1
    metrics[f'{p}/feeds_dapo_reward_or_termination'] = 0
    return metrics


def compute_probe_metrics(non_tensor_batch):
    """Summarize variable-length probe metadata for rollout observability."""

    probe_answers = non_tensor_batch.get('probe_answers')
    if probe_answers is None:
        rows = []
    elif isinstance(probe_answers, np.ndarray):
        rows = list(probe_answers.reshape(-1)) if probe_answers.ndim else [probe_answers.item()]
    else:
        rows = list(probe_answers)

    proposal_count = 0
    recognized_count = 0
    candidates_with_proposals = 0
    for row in rows:
        if isinstance(row, np.ndarray):
            row = row.tolist()
        elif row is None:
            row = []
        elif not isinstance(row, (list, tuple)):
            row = [row]
        if row:
            candidates_with_proposals += 1
        proposal_count += len(row)
        recognized_count += sum(
            isinstance(answer, str) and answer in ('A', 'B', 'C', 'D')
            for answer in row)

    selected = non_tensor_batch.get('selected_stop_position', [])
    if isinstance(selected, np.ndarray):
        selected = list(selected.reshape(-1))
    else:
        selected = list(selected)
    selected_count = sum(value is not None for value in selected)
    candidate_count = len(rows)

    return {
        'probe/candidates': candidate_count,
        'probe/candidates_with_proposals': candidates_with_proposals,
        'probe/candidate_proposal_coverage': (
            candidates_with_proposals / candidate_count if candidate_count else 0.0),
        'probe/proposals': proposal_count,
        'probe/recognized': recognized_count,
        'probe/recognized_ratio': (
            recognized_count / proposal_count if proposal_count else 0.0),
        'probe/selected_stops': selected_count,
        'probe/selected_stop_ratio': (
            selected_count / candidate_count if candidate_count else 0.0),
    }


def compute_reward_component_metrics(reward_extra_info):
    """Expose strict raw-policy reward components and stop-probe telemetry."""

    def numeric(key):
        result = []
        for value in reward_extra_info.get(key, []):
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                result.append(number)
        return np.asarray(result, dtype=np.float64)

    metrics = {}
    mean_names = {
        'strict_raw_acc': 'reward/strict_raw_acc',
        'raw_policy_acc': 'reward/raw_policy_acc',
        'controller_wrap_rejected_stop_count': (
            'reward/controller_wrap_rejected_stop_count_mean'),
        'strict_final_valid': 'reward/strict_final_parse_rate',
        'eligible_atomic_stop_present': (
            'reward/eligible_atomic_stop_presence_rate'),
        'required_output_valid': 'reward/required_output_valid_rate',
        'output_requirement_gated': (
            'reward/output_requirement_gated_rate'),
        'reward_gated': 'reward/reward_hard_gated_rate',
        'format_score': 'reward/format_valid_rate',
        'stop_presence_format_score': 'reward/stop_presence_format_rate',
        'structural_format_score': 'reward/structural_format_rate',
        'verified_stop': 'reward/verified_stop_rate',
        'verified_stop_count': 'reward/verified_stop_count_mean',
        'verified_stop_score': 'reward/verified_stop_score_mean',
        'first_probe_correct': 'reward/first_probe_correct_rate',
        'stop_earliness': 'reward/stop_earliness_all',
        'stop_reward': 'reward/stop_component',
        'format_reward': 'reward/format_component',
        'accuracy_reward': 'reward/accuracy_component',
        'raw_stop_count': 'reward/raw_stop_count_mean',
        'raw_atomic_stop_count': 'reward/raw_atomic_stop_count_mean',
        'illegal_surface_stop_count': (
            'reward/illegal_surface_stop_count_mean'),
        'stop_fragment_count': 'reward/stop_fragment_count_mean',
        'ordinary_surface_stop_count': (
            'reward/ordinary_surface_stop_count_mean'),
        'no_illegal_surface_stops': (
            'reward/no_illegal_surface_stops_rate'),
        'stop_over_limit': 'reward/stop_over_limit_rate',
        'overflow_action_penalty_required': (
            'reward/overflow_action_penalty_required_rate'),
        'negative_stop_action_penalty_required': (
            'reward/negative_stop_action_penalty_required_rate'),
        'auxiliary_reward_gated': 'reward/auxiliary_reward_gated_rate',
        'eligible_stop_count': 'reward/eligible_stop_count_mean',
        'probed_stop_count': 'reward/probed_stop_count_mean',
        'dapo_overlong_penalty': 'reward/dapo_overlong_penalty_mean',
        'dapo_overlong': 'reward/dapo_overlong_rate',
        'dapo_overlong_length': (
            'reward/dapo_overlong_length_mean'
        ),
    }
    for source, target in mean_names.items():
        values = numeric(source)
        if values.size:
            metrics[target] = float(values.mean())

    raw_stops = numeric('raw_stop_count')
    if raw_stops.size:
        metrics['reward/raw_stop_count_max'] = float(raw_stops.max())
    controller_wrap_rejected = numeric(
        'controller_wrap_rejected_stop_count'
    )
    if controller_wrap_rejected.size:
        if np.any(
            (controller_wrap_rejected < 0)
            | (controller_wrap_rejected != np.floor(controller_wrap_rejected))
        ):
            raise ValueError(
                "controller_wrap_rejected_stop_count must contain "
                "non-negative integers"
            )
        metrics['reward/controller_wrap_rejected_stop_count_sum'] = float(
            controller_wrap_rejected.sum()
        )
        metrics['reward/controller_wrap_rejected_stop_row_rate'] = float(
            np.mean(controller_wrap_rejected > 0)
        )
    for source, target in (
        ('raw_atomic_stop_count', 'reward/raw_atomic_stop_count_max'),
        ('illegal_surface_stop_count',
         'reward/illegal_surface_stop_count_max'),
        ('stop_fragment_count', 'reward/stop_fragment_count_max'),
    ):
        values = numeric(source)
        if values.size:
            metrics[target] = float(values.max())

    verified = numeric('verified_stop')
    earliness = numeric('stop_earliness')
    if verified.size and verified.size == earliness.size:
        accepted = earliness[verified > 0.5]
        metrics['reward/stop_earliness_verified'] = (
            float(accepted.mean()) if accepted.size else 0.0
        )

    positions = numeric('verified_stop_position')
    if positions.size:
        accepted_positions = positions[positions >= 0]
        metrics['reward/verified_stop_position_mean'] = (
            float(accepted_positions.mean()) if accepted_positions.size else -1.0
        )

    ordinals = numeric('selected_stop_ordinal')
    if ordinals.size:
        accepted_ordinals = ordinals[ordinals >= 1]
        metrics['reward/selected_stop_ordinal_mean'] = (
            float(accepted_ordinals.mean()) if accepted_ordinals.size else -1.0
        )

    raw_acc = numeric('raw_policy_acc')
    strict_acc = numeric('strict_raw_acc')
    if raw_acc.size and raw_acc.size == strict_acc.size:
        if not np.array_equal(raw_acc, strict_acc):
            raise ValueError(
                "rollout raw_policy_acc disagrees with reward strict_raw_acc"
            )
    return metrics


def select_dapo_nonconstant_uid_groups(
    uids,
    raw_policy_acc,
    *,
    expected_group_size: int,
):
    """Select complete prompt groups whose raw accuracy has nonzero variance.

    DAPO dynamic sampling is a prompt-group operation.  A group is useful only
    when at least one rollout is correct and at least one is wrong.  The
    returned row mask deliberately keeps or rejects *every* rollout sharing a
    uid; filtering individual trajectories would corrupt GRPO normalization.
    """

    uid_values = np.asarray(uids, dtype=object).reshape(-1)
    if isinstance(raw_policy_acc, torch.Tensor):
        acc_values = raw_policy_acc.detach().cpu().numpy().reshape(-1)
    else:
        acc_values = np.asarray(raw_policy_acc).reshape(-1)
    if uid_values.size != acc_values.size:
        raise ValueError(
            "uid and raw_policy_acc must contain the same number of rows: "
            f"{uid_values.size} != {acc_values.size}"
        )
    if uid_values.size == 0:
        raise ValueError("DAPO dynamic sampling received an empty rollout batch")
    if not isinstance(expected_group_size, (int, np.integer)):
        raise TypeError("expected_group_size must be an integer")
    expected_group_size = int(expected_group_size)
    if expected_group_size <= 1:
        raise ValueError(
            "DAPO dynamic sampling requires rollout.n > 1, got "
            f"{expected_group_size}"
        )
    try:
        acc_values = acc_values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_policy_acc must be numeric") from exc
    if not np.all(np.isfinite(acc_values)):
        raise ValueError("raw_policy_acc must contain only finite values")
    if np.any((acc_values != 0.0) & (acc_values != 1.0)):
        raise ValueError(
            "raw_policy_acc must be binary for DAPO group filtering"
        )

    uid_to_rows = {}
    for row_index, uid in enumerate(uid_values.tolist()):
        if uid is None:
            raise ValueError("DAPO group uid must not be None")
        try:
            uid_to_rows.setdefault(uid, []).append(row_index)
        except TypeError as exc:
            raise ValueError("DAPO group uid must be hashable") from exc

    malformed = {
        uid: len(rows)
        for uid, rows in uid_to_rows.items()
        if len(rows) != expected_group_size
    }
    if malformed:
        preview = list(malformed.items())[:5]
        raise ValueError(
            "every DAPO uid group must contain exactly rollout.n rows; "
            f"expected={expected_group_size}, malformed={preview}"
        )

    kept_prompt_uids = []
    discarded_prompt_uids = []
    row_keep_mask = np.zeros(uid_values.size, dtype=np.bool_)
    for uid, rows in uid_to_rows.items():
        group_acc = acc_values[np.asarray(rows, dtype=np.int64)]
        if float(np.std(group_acc)) > 0.0:
            kept_prompt_uids.append(uid)
            row_keep_mask[rows] = True
        else:
            discarded_prompt_uids.append(uid)

    return {
        "kept_indices": np.flatnonzero(row_keep_mask).astype(np.int64),
        "row_keep_mask": row_keep_mask,
        "kept_prompt_uids": kept_prompt_uids,
        "discarded_prompt_uids": discarded_prompt_uids,
        "prompt_count": len(uid_to_rows),
        "kept_prompt_count": len(kept_prompt_uids),
        "discarded_prompt_count": len(discarded_prompt_uids),
    }


def dapo_dynamic_sampling_needs_more(
    *,
    kept_prompt_count: int,
    target_prompt_count: int,
    num_gen_batches: int,
    max_num_gen_batches: int,
):
    """Return whether to resample, failing closed at the configured guard."""

    values = {
        "kept_prompt_count": kept_prompt_count,
        "target_prompt_count": target_prompt_count,
        "num_gen_batches": num_gen_batches,
        "max_num_gen_batches": max_num_gen_batches,
    }
    for name, value in values.items():
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        values[name] = int(value)
    if values["kept_prompt_count"] < 0:
        raise ValueError("kept_prompt_count must be non-negative")
    if values["target_prompt_count"] <= 0:
        raise ValueError("target_prompt_count must be positive")
    if values["num_gen_batches"] <= 0:
        raise ValueError("num_gen_batches must be positive")
    if values["max_num_gen_batches"] <= 0:
        raise ValueError("max_num_gen_batches must be positive")
    if values["kept_prompt_count"] >= values["target_prompt_count"]:
        return False
    if values["num_gen_batches"] >= values["max_num_gen_batches"]:
        raise RuntimeError(
            "DAPO dynamic sampling exhausted max_num_gen_batches before "
            "filling the optimizer batch: "
            f"{values['num_gen_batches']} batches produced "
            f"{values['kept_prompt_count']}/"
            f"{values['target_prompt_count']} non-constant "
            "raw_policy_acc groups"
        )
    return True


def apply_dapo_soft_overlong_penalty(
    data: DataProto,
    reward_tensor: torch.Tensor,
    *,
    max_response_length: int,
    buffer_len: int = 2048,
    penalty_factor: float = 1.0,
):
    """Add DAPO's linear overlong penalty using the effective trajectory.

    Controller-terminated rollouts may have both a discarded physical async
    tail and a controller-forced answer suffix.  Neither is a sampled actor
    action.  Their overlong length is therefore ``accepted_stop_prefix_length``;
    ordinary rows use ``effective_response_length``.  The penalty is written to
    the final valid response token so all pre-existing stop, format, and
    accuracy reward components remain intact.
    """

    if not isinstance(max_response_length, (int, np.integer)):
        raise TypeError("max_response_length must be an integer")
    if not isinstance(buffer_len, (int, np.integer)):
        raise TypeError("overlong buffer_len must be an integer")
    max_response_length = int(max_response_length)
    buffer_len = int(buffer_len)
    try:
        penalty_factor = float(penalty_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlong penalty_factor must be numeric") from exc
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    if buffer_len <= 0 or buffer_len > max_response_length:
        raise ValueError(
            "overlong buffer_len must be in [1, max_response_length], got "
            f"{buffer_len} for max_response_length={max_response_length}"
        )
    if not np.isfinite(penalty_factor) or penalty_factor < 0.0:
        raise ValueError(
            "overlong penalty_factor must be finite and non-negative"
        )
    if reward_tensor.ndim != 2:
        raise ValueError("reward_tensor must have shape [batch, response_length]")
    responses = data.batch["responses"]
    if reward_tensor.shape != responses.shape:
        raise ValueError(
            "reward_tensor must have the same shape as responses: "
            f"{tuple(reward_tensor.shape)} != {tuple(responses.shape)}"
        )
    if int(responses.size(1)) != max_response_length:
        raise ValueError(
            "configured max_response_length must equal the padded response "
            f"width: {max_response_length} != {int(responses.size(1))}"
        )
    if "effective_response_length" not in data.batch:
        raise KeyError(
            "effective_response_length is required so DAPO never penalizes "
            "the discarded physical async tail"
        )

    effective_lengths = data.batch["effective_response_length"]
    if effective_lengths.ndim != 1 or int(effective_lengths.size(0)) != len(data):
        raise ValueError(
            "effective_response_length must contain one value per rollout"
        )
    if torch.is_floating_point(effective_lengths):
        if not torch.all(torch.isfinite(effective_lengths)) or torch.any(
            effective_lengths != torch.floor(effective_lengths)
        ):
            raise ValueError(
                "effective_response_length must contain finite integers"
            )
    effective_lengths = effective_lengths.to(
        device=reward_tensor.device,
        dtype=torch.long,
    )
    if torch.any(effective_lengths <= 0) or torch.any(
        effective_lengths > max_response_length
    ):
        raise ValueError(
            "effective_response_length must be in "
            f"[1, {max_response_length}]"
        )

    attention_response_mask = data.batch["attention_mask"][
        :, -max_response_length:
    ]
    attention_lengths = attention_response_mask.sum(dim=-1).to(
        device=reward_tensor.device,
        dtype=torch.long,
    )
    if not torch.equal(attention_lengths, effective_lengths):
        raise ValueError(
            "effective_response_length disagrees with the controller-visible "
            "response attention mask"
        )
    if "controller_terminated" not in data.batch:
        raise KeyError(
            "controller_terminated is required to select the sampled actor "
            "length for DAPO overlong shaping"
        )
    controller_terminated = data.batch["controller_terminated"].to(
        device=reward_tensor.device,
        dtype=torch.long,
    )
    if controller_terminated.shape != effective_lengths.shape or torch.any(
        (controller_terminated != 0) & (controller_terminated != 1)
    ):
        raise ValueError("controller_terminated must be one binary value per row")
    if "accepted_stop_prefix_length" not in data.batch:
        raise KeyError(
            "accepted_stop_prefix_length is required for controller-terminated "
            "DAPO rows"
        )
    accepted_prefix_lengths = data.batch[
        "accepted_stop_prefix_length"
    ].to(device=reward_tensor.device, dtype=torch.long)
    if accepted_prefix_lengths.shape != effective_lengths.shape:
        raise ValueError(
            "accepted_stop_prefix_length must contain one value per rollout"
        )
    terminated_rows = controller_terminated.bool()
    if torch.any(
        terminated_rows
        & (
            (accepted_prefix_lengths <= 0)
            | (accepted_prefix_lengths > effective_lengths)
        )
    ):
        raise ValueError(
            "controller-terminated accepted_stop_prefix_length must be in "
            "[1, effective_response_length]"
        )
    controller_accounting = {}
    for key in (
        "physical_response_length",
        "controller_discarded_token_count",
        "controller_forced_suffix_token_count",
    ):
        if key not in data.batch:
            raise KeyError(
                f"{key} is required to validate controller overlong lengths"
            )
        values = data.batch[key].to(
            device=reward_tensor.device,
            dtype=torch.long,
        )
        if values.shape != effective_lengths.shape or torch.any(values < 0):
            raise ValueError(
                f"{key} must be one non-negative integer per rollout"
            )
        controller_accounting[key] = values
    physical_lengths = controller_accounting["physical_response_length"]
    discarded_counts = controller_accounting[
        "controller_discarded_token_count"
    ]
    forced_suffix_counts = controller_accounting[
        "controller_forced_suffix_token_count"
    ]
    if torch.any(
        terminated_rows
        & (
            physical_lengths
            != accepted_prefix_lengths + discarded_counts
        )
    ):
        raise ValueError(
            "terminated controller rows require physical_response_length == "
            "accepted_stop_prefix_length + controller_discarded_token_count"
        )
    if torch.any(
        terminated_rows
        & (
            effective_lengths
            != accepted_prefix_lengths + forced_suffix_counts
        )
    ):
        raise ValueError(
            "terminated controller rows require effective_response_length == "
            "accepted_stop_prefix_length + "
            "controller_forced_suffix_token_count"
        )
    penalty_lengths = torch.where(
        terminated_rows,
        accepted_prefix_lengths,
        effective_lengths,
    )

    expected_length = max_response_length - buffer_len
    exceed = (penalty_lengths - expected_length).clamp(
        min=0,
        max=buffer_len,
    )
    penalties = -(
        exceed.to(dtype=reward_tensor.dtype)
        / float(buffer_len)
        * penalty_factor
    )
    penalized_reward = reward_tensor.clone()
    row_indices = torch.arange(
        len(data), device=reward_tensor.device, dtype=torch.long
    )
    terminal_indices = effective_lengths - 1
    penalized_reward[row_indices, terminal_indices] += penalties
    return penalized_reward, penalties, penalty_lengths


def _accumulate_scalar_metrics(state, values, row_count: int):
    """Accumulate scalar telemetry across all generated DAPO batches."""

    if row_count <= 0:
        raise ValueError("metric row_count must be positive")
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        if key.endswith("_sum"):
            entry = state.setdefault(key, ["sum", 0.0, 0.0])
            entry[1] += number
        elif key.endswith("_max"):
            entry = state.setdefault(key, ["max", number, 0.0])
            entry[1] = max(entry[1], number)
        else:
            entry = state.setdefault(key, ["mean", 0.0, 0.0])
            entry[1] += number * row_count
            entry[2] += row_count


def _finalize_scalar_metrics(state):
    result = {}
    for key, (mode, value, weight) in state.items():
        if mode == "mean":
            result[key] = value / weight if weight else 0.0
        else:
            result[key] = value
    return result


def resolve_dapo_runtime_config(config):
    """Resolve the small DAPO surface used by this legacy trainer."""

    filter_cfg = config.algorithm.get("filter_groups", None)
    filter_enabled = bool(
        filter_cfg is not None and filter_cfg.get("enable", False)
    )
    filter_metric = (
        str(filter_cfg.get("metric", "raw_policy_acc"))
        if filter_cfg is not None
        else "raw_policy_acc"
    )
    if filter_enabled and filter_metric != "raw_policy_acc":
        raise ValueError(
            "DAPO dynamic sampling must filter on raw_policy_acc, got "
            f"{filter_metric!r}"
        )
    max_num_gen_batches = int(
        filter_cfg.get("max_num_gen_batches", 10)
        if filter_cfg is not None
        else 10
    )
    if filter_enabled and max_num_gen_batches <= 0:
        raise ValueError(
            "algorithm.filter_groups.max_num_gen_batches must be positive; "
            "this trainer fails closed instead of generating forever"
        )

    reward_model_cfg = config.get("reward_model", None)
    overlong_cfg = (
        reward_model_cfg.get("overlong_buffer", None)
        if reward_model_cfg is not None
        else None
    )
    overlong_enabled = bool(
        overlong_cfg.get("enable", filter_enabled)
        if overlong_cfg is not None
        else filter_enabled
    )
    overlong_buffer_len = int(
        overlong_cfg.get("len", 2048)
        if overlong_cfg is not None
        else 2048
    )
    overlong_penalty_factor = float(
        overlong_cfg.get("penalty_factor", 1.0)
        if overlong_cfg is not None
        else 1.0
    )
    return {
        "filter_enabled": filter_enabled,
        "filter_metric": filter_metric,
        "max_num_gen_batches": max_num_gen_batches,
        "overlong_enabled": overlong_enabled,
        "overlong_buffer_len": overlong_buffer_len,
        "overlong_penalty_factor": overlong_penalty_factor,
    }


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    verified_stop_aux_loss_coef=0.0,
    overflow_stop_aux_loss_coef=0.0,
    negative_stop_aux_loss_coef=0.0,
    incorrect_stop_aux_loss_coef=0.0,
    post_accepted_tail_aux_loss_coef=0.0,
):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            response_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError

    responses = data.batch["responses"]
    valid_response_mask = data.batch["attention_mask"][
        :, -responses.size(1):
    ].to(dtype=data.batch["advantages"].dtype)
    if valid_response_mask.shape != data.batch["advantages"].shape:
        raise ValueError(
            "valid response mask must have the same shape as advantages: "
            f"{tuple(valid_response_mask.shape)} != "
            f"{tuple(data.batch['advantages'].shape)}"
        )
    if torch.any(
        (valid_response_mask != 0) & (valid_response_mask != 1)
    ):
        raise ValueError("valid response mask must be binary")
    if torch.any(
        (data.batch["response_mask"] != 0)
        & (data.batch["response_mask"] != 1)
    ):
        raise ValueError("trajectory response_mask must be binary")
    if torch.any(data.batch["response_mask"] > valid_response_mask):
        raise ValueError(
            "trajectory response_mask must be inside the valid response mask"
        )
    if "atomic_stop_mask" not in data.batch.keys():
        raise KeyError("atomic_stop_mask is required for stop-loss isolation")
    atomic_stop_mask = data.batch["atomic_stop_mask"].to(
        dtype=data.batch["advantages"].dtype
    )
    if atomic_stop_mask.shape != data.batch["advantages"].shape:
        raise ValueError(
            "atomic_stop_mask must have the same shape as advantages"
        )
    if torch.any((atomic_stop_mask != 0) & (atomic_stop_mask != 1)):
        raise ValueError("atomic_stop_mask must be binary")
    if torch.any(atomic_stop_mask > valid_response_mask):
        raise ValueError(
            "atomic stop actions must be inside the valid response mask"
        )
    event_count_keys = {
        "raw_stop_count",
        "raw_atomic_stop_count",
        "illegal_surface_stop_count",
        "ordinary_surface_stop_count",
        "stop_fragment_count",
        "overflow_atomic_stop_count",
        "negative_stop_event_count",
    }
    event_multiplicity_keys = {
        "stop_event_multiplicity",
        "surface_stop_event_multiplicity",
    }
    v13_event_keys = event_count_keys | event_multiplicity_keys
    v13_only_event_keys = v13_event_keys.difference({"raw_stop_count"})
    present_v13_only_event_keys = v13_only_event_keys.intersection(
        data.batch.keys()
    )
    if present_v13_only_event_keys and not v13_event_keys.issubset(
        data.batch.keys()
    ):
        raise KeyError(
            "v13 stop-event telemetry is incomplete: missing "
            f"{sorted(v13_event_keys.difference(data.batch.keys()))}"
        )
    has_v13_event_schema = v13_event_keys.issubset(data.batch.keys())
    if (
        "negative_stop_event_multiplicity" in data.batch
        and not has_v13_event_schema
    ):
        raise KeyError(
            "negative_stop_event_multiplicity requires the complete v13 "
            "stop-event schema"
        )

    event_counts = {}
    stop_event_multiplicity = None
    surface_stop_event_multiplicity = None
    if has_v13_event_schema:
        row_count = int(data.batch["advantages"].size(0))
        for name in sorted(event_count_keys):
            values = data.batch[name]
            if values.ndim != 1 or int(values.size(0)) != row_count:
                raise ValueError(
                    f"{name} must contain one scalar per rollout row"
                )
            if torch.is_complex(values):
                raise ValueError(f"{name} must be real-valued")
            if torch.is_floating_point(values) and (
                not torch.all(torch.isfinite(values))
                or torch.any(values != torch.floor(values))
            ):
                raise ValueError(f"{name} must contain finite integers")
            if torch.any(values < 0):
                raise ValueError(f"{name} must contain non-negative integers")
            event_counts[name] = values.to(
                device=atomic_stop_mask.device, dtype=torch.long
            )

        for name in sorted(event_multiplicity_keys):
            values = data.batch[name]
            if values.shape != data.batch["advantages"].shape:
                raise ValueError(
                    f"{name} must have the same shape as advantages"
                )
            if torch.is_complex(values):
                raise ValueError(f"{name} must be real-valued")
            if torch.is_floating_point(values) and (
                not torch.all(torch.isfinite(values))
                or torch.any(values != torch.floor(values))
            ):
                raise ValueError(f"{name} must contain finite integers")
            if torch.any(values < 0):
                raise ValueError(f"{name} must contain non-negative integers")
            values = values.to(
                device=atomic_stop_mask.device, dtype=torch.long
            )
            if torch.any((values > 0) & (valid_response_mask <= 0)):
                raise ValueError(
                    f"{name} must be zero outside the valid response"
                )
            if name == "stop_event_multiplicity":
                stop_event_multiplicity = values
            else:
                surface_stop_event_multiplicity = values

        if torch.any(
            surface_stop_event_multiplicity > stop_event_multiplicity
        ):
            raise ValueError(
                "surface_stop_event_multiplicity must be a subset of the "
                "unified stop-event multiplicity"
            )
        expected_event_multiplicity = (
            atomic_stop_mask.to(dtype=torch.long)
            + surface_stop_event_multiplicity
        )
        if not torch.equal(
            stop_event_multiplicity, expected_event_multiplicity
        ):
            raise ValueError(
                "stop_event_multiplicity must equal atomic events plus "
                "surface-event multiplicity"
            )
        if not torch.equal(
            stop_event_multiplicity.sum(dim=-1),
            event_counts["raw_stop_count"],
        ):
            raise ValueError(
                "stop_event_multiplicity disagrees with raw_stop_count"
            )
        if not torch.equal(
            surface_stop_event_multiplicity.sum(dim=-1),
            event_counts["illegal_surface_stop_count"],
        ):
            raise ValueError(
                "surface_stop_event_multiplicity disagrees with "
                "illegal_surface_stop_count"
            )
        observed_atomic_stop_count = atomic_stop_mask.sum(dim=-1).to(
            dtype=torch.long
        )
        if not torch.equal(
            observed_atomic_stop_count,
            event_counts["raw_atomic_stop_count"],
        ):
            raise ValueError(
                "atomic_stop_mask disagrees with raw_atomic_stop_count"
            )
        if not torch.equal(
            event_counts["raw_stop_count"],
            event_counts["raw_atomic_stop_count"]
            + event_counts["illegal_surface_stop_count"],
        ):
            raise ValueError(
                "raw_stop_count must equal atomic plus illegal surface events"
            )
        if not torch.equal(
            event_counts["illegal_surface_stop_count"],
            event_counts["ordinary_surface_stop_count"]
            + event_counts["stop_fragment_count"],
        ):
            raise ValueError(
                "illegal_surface_stop_count must equal ordinary surface "
                "events plus stop fragments"
            )
        if not torch.equal(
            event_counts["negative_stop_event_count"],
            event_counts["illegal_surface_stop_count"]
            + event_counts["overflow_atomic_stop_count"],
        ):
            raise ValueError(
                "negative_stop_event_count must equal illegal surface "
                "events plus overflow atomic events"
            )
    elif "raw_stop_count" in data.batch.keys():
        # v12 compatibility: before unified stop-like events, raw_stop_count
        # meant exactly the number of atomic stop token IDs.
        observed_stop_count = atomic_stop_mask.sum(dim=-1).to(
            dtype=data.batch["raw_stop_count"].dtype
        )
        if not torch.equal(observed_stop_count, data.batch["raw_stop_count"]):
            raise ValueError(
                "atomic_stop_mask disagrees with legacy raw_stop_count telemetry"
            )

    required_stop_masks = {"verified_stop_mask", "overflow_stop_mask"}
    missing_stop_masks = sorted(
        required_stop_masks.difference(data.batch.keys())
    )
    if missing_stop_masks:
        raise KeyError(
            "restored stop-loss isolation is missing batch keys: "
            f"{missing_stop_masks}"
        )
    verified_stop_mask = data.batch["verified_stop_mask"].to(
        dtype=data.batch["advantages"].dtype
    )
    overflow_stop_mask = data.batch["overflow_stop_mask"].to(
        dtype=data.batch["advantages"].dtype
    )
    for name, stop_mask in (
        ("verified_stop_mask", verified_stop_mask),
        ("overflow_stop_mask", overflow_stop_mask),
    ):
        if stop_mask.shape != data.batch["advantages"].shape:
            raise ValueError(
                f"{name} must have the same shape as advantages"
            )
        if torch.any((stop_mask != 0) & (stop_mask != 1)):
            raise ValueError(f"{name} must be binary")
        if torch.any(stop_mask > atomic_stop_mask):
            raise ValueError(f"{name} must be a subset of atomic_stop_mask")
    if torch.any(
        (verified_stop_mask > 0) & (overflow_stop_mask > 0)
    ):
        raise ValueError(
            "verified and overflow stop action masks must be disjoint"
        )
    enabled_atomic_stops = (
        (atomic_stop_mask > 0) & (data.batch["response_mask"] > 0)
    )
    if torch.any(enabled_atomic_stops):
        raise ValueError(
            "every atomic stop action must be excluded from trajectory GRPO"
        )
    expected_overflow_mask = None
    max_stop_count_tensor = None
    if "raw_stop_count" in data.batch.keys():
        # The reward manager exports one configured cap per rollout.  v13
        # computes atomic overflow against the unified event stream, so every
        # earlier surface event consumes budget even though it cannot be
        # probed.  The v12 fallback retains atomic-only ordinals.
        max_stop_count_values = data.non_tensor_batch.get("max_stop_count")
        if max_stop_count_values is not None:
            try:
                max_stop_count_array = np.asarray(
                    max_stop_count_values, dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_stop_count telemetry must be numeric"
                ) from exc
            row_count = int(atomic_stop_mask.size(0))
            if (
                max_stop_count_array.ndim != 1
                or max_stop_count_array.size != row_count
            ):
                raise ValueError(
                    "max_stop_count telemetry must contain one value per "
                    f"rollout: shape={max_stop_count_array.shape}, "
                    f"rows={row_count}"
                )
            if (
                not np.all(np.isfinite(max_stop_count_array))
                or np.any(max_stop_count_array < 1)
                or np.any(max_stop_count_array != np.floor(max_stop_count_array))
            ):
                raise ValueError(
                    "max_stop_count telemetry must contain positive integers"
                )
            max_stop_count_tensor = torch.as_tensor(
                max_stop_count_array,
                device=atomic_stop_mask.device,
                dtype=torch.long,
            )
            event_ordinals = torch.cumsum(
                stop_event_multiplicity
                if has_v13_event_schema
                else atomic_stop_mask.to(dtype=torch.long),
                dim=-1,
            )
            expected_overflow_mask = (
                (atomic_stop_mask > 0)
                & (
                    event_ordinals
                    > max_stop_count_tensor.unsqueeze(-1)
                )
            )
            if not torch.equal(
                overflow_stop_mask.to(dtype=torch.bool),
                expected_overflow_mask,
            ):
                raise ValueError(
                    "overflow_stop_mask must mark every atomic stop whose "
                    "unified event ordinal exceeds max_stop_count"
                )
            observed_overflow_count = overflow_stop_mask.sum(dim=-1).to(
                dtype=torch.long
            )
            expected_overflow_count = expected_overflow_mask.sum(dim=-1).to(
                dtype=torch.long
            )
            if not torch.equal(observed_overflow_count, expected_overflow_count):
                raise ValueError(
                    "overflow stop count disagrees with the global event "
                    "ordinal mask"
                )
            if has_v13_event_schema and not torch.equal(
                observed_overflow_count,
                event_counts["overflow_atomic_stop_count"],
            ):
                raise ValueError(
                    "overflow_stop_mask disagrees with "
                    "overflow_atomic_stop_count"
                )

    verified_aux_coef = float(verified_stop_aux_loss_coef)
    if not np.isfinite(verified_aux_coef) or verified_aux_coef < 0:
        raise ValueError(
            "verified_stop_aux_loss_coef must be finite and non-negative, "
            f"got {verified_stop_aux_loss_coef}"
        )
    if verified_aux_coef != 0:
        if "verified_stop_aux_weight" not in data.batch.keys():
            raise KeyError(
                "verified_stop_aux_weight is required when "
                "verified_stop_aux_loss_coef is enabled"
            )
        verified_stop_aux_weight = data.batch[
            "verified_stop_aux_weight"
        ].to(dtype=data.batch["advantages"].dtype)
        if verified_stop_aux_weight.shape != data.batch["advantages"].shape:
            raise ValueError(
                "verified_stop_aux_weight must have the same shape as advantages"
            )
        if not torch.all(torch.isfinite(verified_stop_aux_weight)):
            raise ValueError("verified_stop_aux_weight must be finite")
        if torch.any(verified_stop_aux_weight < 0):
            raise ValueError("verified_stop_aux_weight must be non-negative")
        if torch.any(
            verified_stop_aux_weight * (1.0 - verified_stop_mask) != 0
        ):
            raise ValueError(
                "verified_stop_aux_weight must be zero outside verified_stop_mask"
            )
        if torch.any(
            (verified_stop_mask > 0) & (verified_stop_aux_weight <= 0)
        ):
            raise ValueError(
                "verified stop actions must have strictly positive auxiliary weight"
            )
        reward_gated = data.non_tensor_batch.get("reward_gated")
        if reward_gated is None:
            raise KeyError(
                "reward_gated telemetry is required for verified-stop fail-closed gating"
            )
        reward_gated_tensor = torch.as_tensor(
            np.asarray(reward_gated, dtype=np.int64),
            device=verified_stop_mask.device,
            dtype=torch.bool,
        )
        if reward_gated_tensor.ndim != 1 or reward_gated_tensor.size(0) != verified_stop_mask.size(0):
            raise ValueError(
                "reward_gated telemetry must contain one value per rollout"
            )
        if torch.any(
            verified_stop_mask[reward_gated_tensor] > 0
        ) or torch.any(verified_stop_aux_weight[reward_gated_tensor] > 0):
            raise ValueError(
                "hard-gated rollouts must not retain positive verified-stop credit"
            )
        data.batch["verified_stop_aux_bonus"] = (
            verified_stop_aux_weight * verified_aux_coef
        )
        reported_stop_reward = data.non_tensor_batch.get("stop_reward")
        if reported_stop_reward is None:
            raise KeyError(
                "stop_reward telemetry is required to verify action-reward routing"
            )
        reported_stop_reward_tensor = torch.as_tensor(
            np.asarray(reported_stop_reward, dtype=np.float64),
            device=verified_stop_mask.device,
            dtype=data.batch["verified_stop_aux_bonus"].dtype,
        )
        if reported_stop_reward_tensor.ndim != 1 or reported_stop_reward_tensor.size(0) != verified_stop_mask.size(0):
            raise ValueError(
                "stop_reward telemetry must contain one value per rollout"
            )
        if not torch.allclose(
            data.batch["verified_stop_aux_bonus"].sum(dim=-1),
            reported_stop_reward_tensor,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "verified-stop action bonus disagrees with stop_reward telemetry"
            )
    else:
        data.batch["verified_stop_aux_bonus"] = torch.zeros_like(
            data.batch["advantages"]
        )

    overflow_aux_coef = float(overflow_stop_aux_loss_coef)
    if not np.isfinite(overflow_aux_coef) or overflow_aux_coef < 0:
        raise ValueError(
            "overflow_stop_aux_loss_coef must be finite and non-negative, "
            f"got {overflow_stop_aux_loss_coef}"
        )
    if overflow_aux_coef > 0:
        if "overflow_stop_mask" not in data.batch.keys():
            raise KeyError(
                "overflow_stop_mask is required when "
                "overflow_stop_aux_loss_coef is enabled"
            )
        overflow_stop_mask = data.batch["overflow_stop_mask"].to(
            dtype=data.batch["advantages"].dtype
        )
        if overflow_stop_mask.shape != data.batch["advantages"].shape:
            raise ValueError(
                "overflow_stop_mask must have the same shape as advantages: "
                f"{tuple(overflow_stop_mask.shape)} != "
                f"{tuple(data.batch['advantages'].shape)}"
            )
        if torch.any((overflow_stop_mask != 0) & (overflow_stop_mask != 1)):
            raise ValueError("overflow_stop_mask must be binary")
        if torch.any(overflow_stop_mask > valid_response_mask):
            raise ValueError(
                "overflow stop actions must be inside the valid response mask"
            )
        if torch.any(overflow_stop_mask > atomic_stop_mask):
            raise ValueError(
                "overflow_stop_mask must be a subset of atomic_stop_mask"
            )
        if torch.any(
            (overflow_stop_mask > 0) & (data.batch["response_mask"] > 0)
        ):
            raise ValueError(
                "overflow stop actions must be excluded from trajectory GRPO"
            )

        # The actor consumes this mask in a separately normalized, clipped PPO
        # auxiliary loss.  Do not mix it into GRPO advantages: doing so would
        # make its strength depend on response length and micro-batch packing.
        data.batch["overflow_stop_aux_penalty"] = (
            overflow_stop_mask * overflow_aux_coef
        )
    else:
        data.batch["overflow_stop_aux_penalty"] = torch.zeros_like(
            data.batch["advantages"]
        )

    def validate_weighted_negative_auxiliary(
        objective_name,
        coefficient,
        mask_key,
        weight_key,
        penalty_key,
        max_row_weight_sum=1.0,
    ):
        coefficient = float(coefficient)
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError(
                f"{objective_name} coefficient must be finite and "
                f"non-negative, got {coefficient}"
            )
        present = {key for key in (mask_key, weight_key) if key in data.batch}
        if present and len(present) != 2:
            raise KeyError(
                f"{objective_name} is missing batch keys: "
                f"{sorted({mask_key, weight_key}.difference(data.batch.keys()))}"
            )
        if not present:
            if coefficient > 0:
                raise KeyError(
                    f"{objective_name} requires {mask_key} and {weight_key}"
                )
            zero = torch.zeros_like(data.batch["advantages"])
            data.batch[penalty_key] = zero
            return zero, zero

        mask = data.batch[mask_key].to(
            dtype=data.batch["advantages"].dtype
        )
        weight = data.batch[weight_key].to(
            dtype=data.batch["advantages"].dtype
        )
        for name, tensor in ((mask_key, mask), (weight_key, weight)):
            if tensor.shape != data.batch["advantages"].shape:
                raise ValueError(
                    f"{name} must have the same shape as advantages"
                )
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError(f"{mask_key} must be binary")
        if torch.any(mask > valid_response_mask):
            raise ValueError(
                f"{mask_key} must be inside the valid response mask"
            )
        if not torch.all(torch.isfinite(weight)):
            raise ValueError(f"{weight_key} must be finite")
        if torch.any(weight < 0):
            raise ValueError(f"{weight_key} must be non-negative")
        if torch.any(weight * (1.0 - mask) != 0):
            raise ValueError(f"{weight_key} must be zero outside {mask_key}")
        if torch.any((mask > 0) & (weight <= 0)):
            raise ValueError(
                f"{weight_key} must be strictly positive inside {mask_key}"
            )
        max_weight = float(max_row_weight_sum)
        if not np.isfinite(max_weight) or max_weight <= 0:
            raise ValueError(
                f"{objective_name} max_row_weight_sum must be finite and "
                f"positive, got {max_row_weight_sum}"
            )
        tolerance = max(
            1e-6,
            8.0 * torch.finfo(weight.dtype).eps * max_weight,
        )
        if torch.any(weight.sum(dim=-1) > max_weight + tolerance):
            raise ValueError(
                f"{weight_key} must sum to at most {max_weight} per rollout"
            )
        data.batch[penalty_key] = weight * coefficient
        return mask, weight

    negative_stop_aux_mask, negative_stop_aux_weight = (
        validate_weighted_negative_auxiliary(
            "negative_stop_aux",
            negative_stop_aux_loss_coef,
            "negative_stop_aux_mask",
            "negative_stop_aux_weight",
            "negative_stop_aux_penalty",
        )
    )
    incorrect_stop_aux_mask, incorrect_stop_aux_weight = (
        validate_weighted_negative_auxiliary(
            "incorrect_stop_aux",
            incorrect_stop_aux_loss_coef,
            "incorrect_stop_aux_mask",
            "incorrect_stop_aux_weight",
            "incorrect_stop_aux_penalty",
            max_row_weight_sum=8.0,
        )
    )
    post_accepted_tail_aux_mask, post_accepted_tail_aux_weight = (
        validate_weighted_negative_auxiliary(
            "post_accepted_tail_aux",
            post_accepted_tail_aux_loss_coef,
            "post_accepted_tail_aux_mask",
            "post_accepted_tail_aux_weight",
            "post_accepted_tail_aux_penalty",
        )
    )
    if "controller_terminated" in data.batch:
        if float(post_accepted_tail_aux_loss_coef) != 0.0:
            raise ValueError(
                "controller-terminal rollouts require "
                "post_accepted_tail_aux_loss_coef=0"
            )
        if torch.any(post_accepted_tail_aux_mask != 0) or torch.any(
            post_accepted_tail_aux_weight != 0
        ):
            raise ValueError(
                "controller-terminal rollouts cannot optimize a sampled "
                "post-accepted tail"
            )

    if "negative_stop_aux_mask" in data.batch:
        if not has_v13_event_schema:
            raise KeyError(
                "negative_stop_aux requires the complete v13 stop-event schema"
            )
        if expected_overflow_mask is None:
            raise KeyError(
                "negative_stop_aux requires max_stop_count telemetry for "
                "global event-ordinal validation"
            )
        expected_negative_multiplicity = (
            surface_stop_event_multiplicity
            + expected_overflow_mask.to(dtype=torch.long)
        )
        if "negative_stop_event_multiplicity" in data.batch:
            reported_negative_multiplicity = data.batch[
                "negative_stop_event_multiplicity"
            ]
            if (
                reported_negative_multiplicity.shape
                != data.batch["advantages"].shape
            ):
                raise ValueError(
                    "negative_stop_event_multiplicity must have the same "
                    "shape as advantages"
                )
            if torch.is_complex(reported_negative_multiplicity):
                raise ValueError(
                    "negative_stop_event_multiplicity must be real-valued"
                )
            if torch.is_floating_point(reported_negative_multiplicity) and (
                not torch.all(torch.isfinite(reported_negative_multiplicity))
                or torch.any(
                    reported_negative_multiplicity
                    != torch.floor(reported_negative_multiplicity)
                )
            ):
                raise ValueError(
                    "negative_stop_event_multiplicity must contain finite "
                    "integers"
                )
            reported_negative_multiplicity = (
                reported_negative_multiplicity.to(
                    device=atomic_stop_mask.device,
                    dtype=torch.long,
                )
            )
            if torch.any(reported_negative_multiplicity < 0):
                raise ValueError(
                    "negative_stop_event_multiplicity must be non-negative"
                )
            if torch.any(
                (reported_negative_multiplicity > 0)
                & (valid_response_mask <= 0)
            ):
                raise ValueError(
                    "negative_stop_event_multiplicity must be zero outside "
                    "the valid response"
                )
            if not torch.equal(
                reported_negative_multiplicity,
                expected_negative_multiplicity,
            ):
                raise ValueError(
                    "negative_stop_event_multiplicity must equal surface "
                    "multiplicity plus globally overflowing atomic stops"
                )
        expected_negative_mask = expected_negative_multiplicity > 0
        if not torch.equal(
            negative_stop_aux_mask.to(dtype=torch.bool),
            expected_negative_mask,
        ):
            raise ValueError(
                "negative_stop_aux_mask must be the union of every surface "
                "event anchor and every globally overflowing atomic stop"
            )
        if not torch.equal(
            expected_negative_multiplicity.sum(dim=-1),
            event_counts["negative_stop_event_count"],
        ):
            raise ValueError(
                "negative stop multiplicity disagrees with "
                "negative_stop_event_count"
            )
        negative_denominator = event_counts[
            "negative_stop_event_count"
        ].clamp(min=1).unsqueeze(-1)
        expected_negative_weight = (
            expected_negative_multiplicity.to(
                dtype=negative_stop_aux_weight.dtype
            )
            / negative_denominator.to(
                dtype=negative_stop_aux_weight.dtype
            )
        )
        if not torch.allclose(
            negative_stop_aux_weight,
            expected_negative_weight,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "negative_stop_aux_weight must normalize event multiplicity "
                "to unit total weight per negative rollout"
            )
        if torch.any(
            (negative_stop_aux_mask > 0)
            & (data.batch["response_mask"] > 0)
        ):
            raise ValueError(
                "negative stop action anchors must be excluded from "
                "trajectory GRPO"
            )
        if torch.any(
            (negative_stop_aux_mask > 0) & (verified_stop_mask > 0)
        ):
            raise ValueError(
                "verified and negative stop auxiliary masks must be disjoint"
            )
        if torch.any(overflow_stop_mask > negative_stop_aux_mask):
            raise ValueError(
                "every overflow atomic stop must be included in "
                "negative_stop_aux_mask"
            )

    if "incorrect_stop_aux_mask" in data.batch:
        tri_state_count_keys = {
            "recognized_correct_probe_count",
            "incorrect_stop_count",
            "neutral_probe_stop_count",
        }
        missing_tri_state_keys = tri_state_count_keys.difference(
            data.batch.keys()
        )
        if missing_tri_state_keys:
            raise KeyError(
                "incorrect_stop_aux requires complete probe tri-state "
                f"telemetry: missing {sorted(missing_tri_state_keys)}"
            )
        if max_stop_count_tensor is None:
            raise KeyError(
                "incorrect_stop_aux requires max_stop_count telemetry"
            )
        incorrect_counts = data.batch["incorrect_stop_count"]
        row_count = int(data.batch["advantages"].size(0))
        selected_kinds, _, _ = _selected_stop_metadata_arrays(data, row_count)
        if incorrect_counts.ndim != 1 or int(incorrect_counts.size(0)) != row_count:
            raise ValueError(
                "incorrect_stop_count must contain one scalar per rollout"
            )
        if torch.is_complex(incorrect_counts) or (
            torch.is_floating_point(incorrect_counts)
            and (
                not torch.all(torch.isfinite(incorrect_counts))
                or torch.any(incorrect_counts != torch.floor(incorrect_counts))
            )
        ):
            raise ValueError("incorrect_stop_count must contain finite integers")
        incorrect_counts = incorrect_counts.to(
            device=atomic_stop_mask.device, dtype=torch.long
        )
        tri_state_counts = {}
        for count_name in (
            "recognized_correct_probe_count",
            "neutral_probe_stop_count",
        ):
            values = data.batch[count_name]
            if values.ndim != 1 or int(values.size(0)) != row_count:
                raise ValueError(
                    f"{count_name} must contain one scalar per rollout"
                )
            if torch.is_complex(values) or (
                torch.is_floating_point(values)
                and (
                    not torch.all(torch.isfinite(values))
                    or torch.any(values != torch.floor(values))
                )
            ):
                raise ValueError(f"{count_name} must contain finite integers")
            values = values.to(
                device=atomic_stop_mask.device, dtype=torch.long
            )
            if torch.any(values < 0):
                raise ValueError(f"{count_name} must be non-negative")
            tri_state_counts[count_name] = values
        observed_incorrect_counts = incorrect_stop_aux_mask.sum(dim=-1).to(
            dtype=torch.long
        )
        if torch.any(incorrect_counts < 0) or not torch.equal(
            incorrect_counts, observed_incorrect_counts
        ):
            raise ValueError(
                "incorrect_stop_count must equal incorrect_stop_aux_mask sum"
            )
        if torch.any(incorrect_counts > max_stop_count_tensor) or torch.any(
            incorrect_counts > 8
        ):
            raise ValueError(
                "incorrect stop count exceeds the configured eight-attempt budget"
            )
        if "probed_stop_count" in data.batch and torch.any(
            incorrect_counts
            > data.batch["probed_stop_count"].to(
                device=incorrect_counts.device, dtype=torch.long
            )
        ):
            raise ValueError(
                "incorrect stop count cannot exceed probed stop count"
            )
        if "probed_stop_count" not in data.batch:
            raise KeyError(
                "probe tri-state validation requires probed_stop_count"
            )
        probed_counts = data.batch["probed_stop_count"].to(
            device=incorrect_counts.device, dtype=torch.long
        )
        if not torch.equal(
            tri_state_counts["recognized_correct_probe_count"]
            + incorrect_counts
            + tri_state_counts["neutral_probe_stop_count"],
            probed_counts,
        ):
            raise ValueError(
                "recognized-correct + incorrect + neutral probe counts must "
                "equal probed_stop_count"
            )
        if "verified_stop_count" in data.batch:
            selected_counts = data.batch["verified_stop_count"].to(
                device=incorrect_counts.device, dtype=torch.long
            )
            gold_rows = torch.as_tensor(
                selected_kinds == "gold_correct",
                device=incorrect_counts.device,
                dtype=torch.bool,
            )
            fallback_rows = torch.as_tensor(
                selected_kinds == "final_consistent",
                device=incorrect_counts.device,
                dtype=torch.bool,
            )
            if torch.any(
                gold_rows
                & (
                    selected_counts
                    > tri_state_counts["recognized_correct_probe_count"]
                )
            ):
                raise ValueError(
                    "gold-correct selected count cannot exceed recognized-correct "
                    "probe count"
                )
        if torch.any(incorrect_stop_aux_mask > atomic_stop_mask):
            raise ValueError(
                "incorrect_stop_aux_mask must be a subset of atomic_stop_mask"
            )
        if torch.any(
            (incorrect_stop_aux_mask > 0) & (verified_stop_mask > 0)
        ):
            raise ValueError(
                "incorrect and verified stop auxiliary masks must be disjoint"
            )
        if torch.any(
            (incorrect_stop_aux_mask > 0) & (negative_stop_aux_mask > 0)
        ):
            raise ValueError(
                "incorrect and generalized-negative stop masks must be disjoint"
            )
        if torch.any(
            (incorrect_stop_aux_mask > 0)
            & (data.batch["response_mask"] > 0)
        ):
            raise ValueError(
                "incorrect stop actions must be excluded from trajectory GRPO"
            )

        half_life_values = data.non_tensor_batch.get(
            "stop_reward_half_life_tokens"
        )
        if half_life_values is None:
            raise KeyError(
                "incorrect_stop_aux requires stop_reward_half_life_tokens telemetry"
            )
        try:
            half_life_array = np.asarray(half_life_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stop_reward_half_life_tokens telemetry must be numeric"
            ) from exc
        if (
            half_life_array.ndim != 1
            or half_life_array.size != row_count
            or not np.all(np.isfinite(half_life_array))
            or np.any(half_life_array <= 0)
        ):
            raise ValueError(
                "stop_reward_half_life_tokens must contain one positive "
                "finite value per rollout"
            )
        half_life_tensor = torch.as_tensor(
            half_life_array,
            device=incorrect_stop_aux_weight.device,
            dtype=incorrect_stop_aux_weight.dtype,
        ).unsqueeze(-1)
        positions = torch.arange(
            responses.size(1),
            device=incorrect_stop_aux_weight.device,
            dtype=incorrect_stop_aux_weight.dtype,
        ).unsqueeze(0)
        expected_incorrect_weight = (
            torch.pow(
                torch.tensor(
                    2.0,
                    device=incorrect_stop_aux_weight.device,
                    dtype=incorrect_stop_aux_weight.dtype,
                ),
                -positions / half_life_tensor,
            )
            * incorrect_stop_aux_mask
        )
        if not torch.allclose(
            incorrect_stop_aux_weight,
            expected_incorrect_weight,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "incorrect_stop_aux_weight must equal the unnormalized "
                "earliness weight of every wrong stop action"
            )
        if torch.any(
            incorrect_stop_aux_weight.sum(dim=-1)
            > incorrect_counts.to(dtype=incorrect_stop_aux_weight.dtype)
            + 1e-6
        ):
            raise ValueError(
                "incorrect stop row weight cannot exceed its action count"
            )

    if "post_accepted_tail_aux_mask" in data.batch:
        if "oracle_accepted_stop_position" not in data.batch:
            raise KeyError(
                "post_accepted_tail_aux requires "
                "oracle_accepted_stop_position"
            )
        oracle_positions = data.batch[
            "oracle_accepted_stop_position"
        ]
        row_count = int(data.batch["advantages"].size(0))
        if oracle_positions.ndim != 1 or int(oracle_positions.size(0)) != row_count:
            raise ValueError(
                "oracle_accepted_stop_position must contain one value per row"
            )
        if torch.is_complex(oracle_positions):
            raise ValueError(
                "oracle_accepted_stop_position must be real-valued"
            )
        if torch.is_floating_point(oracle_positions) and (
            not torch.all(torch.isfinite(oracle_positions))
            or torch.any(oracle_positions != torch.floor(oracle_positions))
        ):
            raise ValueError(
                "oracle_accepted_stop_position must contain finite integers"
            )
        oracle_positions = oracle_positions.to(
            device=atomic_stop_mask.device, dtype=torch.long
        )
        if torch.any(oracle_positions < -1) or torch.any(
            oracle_positions >= responses.size(1)
        ):
            raise ValueError(
                "oracle_accepted_stop_position is outside the response"
            )
        accepted_rows = oracle_positions >= 0
        if "oracle_accepted_stop_ordinal" in data.batch:
            oracle_ordinals = data.batch["oracle_accepted_stop_ordinal"]
            if (
                oracle_ordinals.ndim != 1
                or int(oracle_ordinals.size(0)) != row_count
            ):
                raise ValueError(
                    "oracle_accepted_stop_ordinal must contain one value per "
                    "row"
                )
            if torch.is_complex(oracle_ordinals):
                raise ValueError(
                    "oracle_accepted_stop_ordinal must be real-valued"
                )
            if torch.is_floating_point(oracle_ordinals) and (
                not torch.all(torch.isfinite(oracle_ordinals))
                or torch.any(oracle_ordinals != torch.floor(oracle_ordinals))
            ):
                raise ValueError(
                    "oracle_accepted_stop_ordinal must contain finite integers"
                )
            oracle_ordinals = oracle_ordinals.to(
                device=atomic_stop_mask.device,
                dtype=torch.long,
            )
            if torch.any(oracle_ordinals < -1):
                raise ValueError(
                    "oracle_accepted_stop_ordinal must be -1 or positive"
                )
            if torch.any(
                accepted_rows != (oracle_ordinals >= 1)
            ):
                raise ValueError(
                    "oracle accepted stop position and ordinal must be "
                    "present on exactly the same rows"
                )
            if torch.any(accepted_rows):
                accepted_row_indices = torch.nonzero(
                    accepted_rows, as_tuple=False
                ).squeeze(-1)
                if has_v13_event_schema:
                    cumulative_event_counts = torch.cumsum(
                        stop_event_multiplicity,
                        dim=-1,
                    )
                    accepted_positions = oracle_positions[
                        accepted_row_indices
                    ]
                    accepted_upper_ordinals = cumulative_event_counts[
                        accepted_row_indices,
                        accepted_positions,
                    ]
                    accepted_lower_ordinals = (
                        accepted_upper_ordinals
                        - stop_event_multiplicity[
                            accepted_row_indices,
                            accepted_positions,
                        ]
                    )
                    accepted_ordinals = oracle_ordinals[
                        accepted_row_indices
                    ]
                    if torch.any(
                        (accepted_ordinals <= accepted_lower_ordinals)
                        | (accepted_ordinals > accepted_upper_ordinals)
                    ):
                        raise ValueError(
                            "oracle_accepted_stop_ordinal is inconsistent "
                            "with the unified event stream at its token anchor"
                        )
                if max_stop_count_tensor is not None and torch.any(
                    oracle_ordinals[accepted_row_indices]
                    > max_stop_count_tensor[accepted_row_indices]
                ):
                    raise ValueError(
                        "oracle-accepted stop ordinal exceeds max_stop_count"
                    )
        if torch.any(accepted_rows):
            accepted_row_indices = torch.nonzero(
                accepted_rows, as_tuple=False
            ).squeeze(-1)
            accepted_tokens = atomic_stop_mask[
                accepted_row_indices,
                oracle_positions[accepted_row_indices],
            ]
            if torch.any(accepted_tokens <= 0):
                raise ValueError(
                    "oracle_accepted_stop_position must identify an atomic stop"
                )
        if "verified_stop_position" in data.batch:
            verified_positions = data.batch["verified_stop_position"].to(
                device=oracle_positions.device, dtype=torch.long
            )
            verified_rows = verified_positions >= 0
            if torch.any(
                verified_rows
                & (verified_positions != oracle_positions)
            ):
                raise ValueError(
                    "a gate-retained verified stop must equal the oracle "
                    "accepted stop position"
                )
        response_positions = torch.arange(
            responses.size(1), device=oracle_positions.device
        ).unsqueeze(0)
        if torch.any(
            (post_accepted_tail_aux_mask > 0)
            & (
                (~accepted_rows).unsqueeze(-1)
                | (response_positions <= oracle_positions.unsqueeze(-1))
            )
        ):
            raise ValueError(
                "post_accepted_tail_aux_mask must contain only actions after "
                "an oracle-accepted stop"
            )
        if torch.any(
            (post_accepted_tail_aux_mask > 0)
            & (atomic_stop_mask > 0)
        ):
            raise ValueError(
                "post-accepted tail must exclude every atomic stop action"
            )
        if torch.any(
            (post_accepted_tail_aux_mask > 0)
            & (negative_stop_aux_mask > 0)
        ):
            raise ValueError(
                "post-accepted tail and negative-stop masks must be disjoint"
            )
        if torch.any(
            (post_accepted_tail_aux_mask > 0)
            & (incorrect_stop_aux_mask > 0)
        ):
            raise ValueError(
                "post-accepted tail and incorrect-stop masks must be disjoint"
            )
        tail_action_counts = post_accepted_tail_aux_mask.sum(
            dim=-1
        ).to(dtype=torch.long)
        tail_denominator = torch.maximum(
            tail_action_counts,
            torch.full_like(tail_action_counts, 256),
        ).clamp(min=1).unsqueeze(-1)
        expected_tail_weight = (
            post_accepted_tail_aux_mask
            / tail_denominator.to(dtype=post_accepted_tail_aux_weight.dtype)
        )
        if not torch.allclose(
            post_accepted_tail_aux_weight,
            expected_tail_weight,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "post_accepted_tail_aux_weight must be uniform with row sum "
                "min(tail_action_count / 256, 1)"
            )
    return data


def append_rollout_stop_audit(path: str, step: int, data: DataProto) -> dict[str, float]:
    """Append one JSON record per rollout and return count-distribution metrics."""

    if not path:
        raise ValueError("rollout stop audit path must be non-empty")
    required_batch_keys = {
        "raw_stop_count",
        "raw_atomic_stop_count",
        "illegal_surface_stop_count",
        "ordinary_surface_stop_count",
        "stop_fragment_count",
        "overflow_atomic_stop_count",
        "negative_stop_event_count",
        "stop_event_multiplicity",
        "surface_stop_event_multiplicity",
        "early_stop_score",
        "verified_stop_count",
        "verified_stop_score",
        "eligible_stop_count",
        "probed_stop_count",
        "recognized_correct_probe_count",
        "neutral_probe_stop_count",
        "responses",
        "attention_mask",
        "response_mask",
        "token_level_scores",
        "advantages",
        "atomic_stop_mask",
        "verified_stop_mask",
        "verified_stop_aux_weight",
        "verified_stop_aux_bonus",
        "overflow_stop_mask",
        "overflow_stop_aux_penalty",
        "negative_stop_aux_mask",
        "negative_stop_aux_weight",
        "negative_stop_aux_penalty",
        "incorrect_stop_count",
        "incorrect_stop_aux_mask",
        "incorrect_stop_aux_weight",
        "incorrect_stop_aux_penalty",
        "post_accepted_tail_aux_mask",
        "post_accepted_tail_aux_weight",
        "post_accepted_tail_aux_penalty",
        "rollout_index",
        "verified_stop_aux_suppressed",
        "verified_stop_position",
        "oracle_accepted_stop_position",
        "accepted_stop_prefix_length",
        "tail_cut_tokens",
        "accepted_stop_tail_cut_enabled",
        "post_accepted_active_token_count",
        "effective_response_length",
        "controller_terminated",
        "controller_wrap_rejected_stop_count",
        "controller_discarded_token_count",
        "controller_forced_suffix_token_count",
        "physical_response_length",
        "physical_raw_stop_count",
        "physical_raw_atomic_stop_count",
        "physical_illegal_surface_stop_count",
        "physical_policy_acc",
        "raw_policy_acc",
    }
    missing = sorted(required_batch_keys.difference(data.batch.keys()))
    if missing:
        raise KeyError(f"rollout stop audit is missing batch keys: {missing}")
    if "uid" not in data.non_tensor_batch:
        raise KeyError("rollout stop audit requires uid")

    row_count = len(data)
    selected_kinds, selected_credits, final_consistency_credits = (
        _selected_stop_metadata_arrays(data, row_count)
    )
    raw_counts = data.batch["raw_stop_count"].detach().cpu().numpy().astype(np.int64)
    raw_atomic_counts = (
        data.batch["raw_atomic_stop_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    illegal_surface_counts = (
        data.batch["illegal_surface_stop_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    ordinary_surface_counts = (
        data.batch["ordinary_surface_stop_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    fragment_counts = (
        data.batch["stop_fragment_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    overflow_atomic_counts = (
        data.batch["overflow_atomic_stop_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    negative_event_counts = (
        data.batch["negative_stop_event_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    if np.any(raw_counts != raw_atomic_counts + illegal_surface_counts):
        raise ValueError(
            "raw_stop_count must equal atomic plus illegal surface events"
        )
    if np.any(
        illegal_surface_counts != ordinary_surface_counts + fragment_counts
    ):
        raise ValueError(
            "illegal surface count must equal ordinary surfaces plus fragments"
        )
    if np.any(
        negative_event_counts != illegal_surface_counts + overflow_atomic_counts
    ):
        raise ValueError(
            "negative event count must equal illegal surfaces plus atomic overflow"
        )
    stop_event_multiplicity = (
        data.batch["stop_event_multiplicity"]
        .detach().cpu().numpy().astype(np.int64)
    )
    surface_stop_event_multiplicity = (
        data.batch["surface_stop_event_multiplicity"]
        .detach().cpu().numpy().astype(np.int64)
    )
    atomic_stop_event_multiplicity = (
        data.batch["atomic_stop_mask"]
        .detach().cpu().numpy().astype(np.int64)
    )
    if np.any(
        stop_event_multiplicity
        != atomic_stop_event_multiplicity
        + surface_stop_event_multiplicity
    ):
        raise ValueError(
            "stop event multiplicity must equal atomic plus surface events"
        )
    stop_event_anchor_counts = (stop_event_multiplicity > 0).sum(
        axis=-1
    ).astype(np.int64)
    surface_stop_event_anchor_counts = (
        surface_stop_event_multiplicity > 0
    ).sum(axis=-1).astype(np.int64)
    if np.any(stop_event_multiplicity.sum(axis=-1) != raw_counts):
        raise ValueError(
            "stop event multiplicity sum disagrees with raw_stop_count"
        )
    if np.any(
        surface_stop_event_multiplicity.sum(axis=-1)
        != illegal_surface_counts
    ):
        raise ValueError(
            "surface event multiplicity sum disagrees with surface count"
        )
    eligible_counts = (
        data.batch["eligible_stop_count"].detach().cpu().numpy().astype(np.int64)
    )
    probed_counts = (
        data.batch["probed_stop_count"].detach().cpu().numpy().astype(np.int64)
    )
    recognized_correct_probe_counts = (
        data.batch["recognized_correct_probe_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    neutral_probe_stop_counts = (
        data.batch["neutral_probe_stop_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    overflow_stop_masks = (
        data.batch["overflow_stop_mask"]
        .detach().cpu().numpy().astype(np.int64)
    )
    overflow_action_counts = overflow_stop_masks.sum(axis=-1).astype(
        np.int64
    )
    if np.any(overflow_action_counts != overflow_atomic_counts):
        raise ValueError(
            "overflow_stop_mask disagrees with overflow_atomic_stop_count"
        )
    expected_negative_stop_event_multiplicity = (
        surface_stop_event_multiplicity + overflow_stop_masks
    )
    if "negative_stop_event_multiplicity" in data.batch:
        negative_stop_event_multiplicity = (
            data.batch["negative_stop_event_multiplicity"]
            .detach().cpu().numpy().astype(np.int64)
        )
        if np.any(
            negative_stop_event_multiplicity
            != expected_negative_stop_event_multiplicity
        ):
            raise ValueError(
                "negative stop event multiplicity must equal surface events "
                "plus atomic overflow"
            )
    else:
        # Compatibility for pre-v13 synthetic audit fixtures.  Production v13
        # rollouts export this tensor and take the strict branch above.
        negative_stop_event_multiplicity = (
            expected_negative_stop_event_multiplicity
        )
    if np.any(
        negative_stop_event_multiplicity.sum(axis=-1)
        != negative_event_counts
    ):
        raise ValueError(
            "negative stop event multiplicity sum disagrees with count"
        )
    negative_stop_event_anchor_counts = (
        negative_stop_event_multiplicity > 0
    ).sum(axis=-1).astype(np.int64)
    negative_action_counts = (
        data.batch["negative_stop_aux_mask"]
        .detach().sum(dim=-1).cpu().numpy().astype(np.int64)
    )
    negative_aux_weights = (
        data.batch["negative_stop_aux_weight"]
        .detach().sum(dim=-1).cpu().numpy()
    )
    negative_aux_penalties = (
        data.batch["negative_stop_aux_penalty"]
        .detach().sum(dim=-1).cpu().numpy()
    )
    incorrect_counts = (
        data.batch["incorrect_stop_count"]
        .detach().cpu().numpy().astype(np.int64)
    )
    incorrect_stop_masks = (
        data.batch["incorrect_stop_aux_mask"]
        .detach().cpu().numpy().astype(np.int64)
    )
    incorrect_action_counts = incorrect_stop_masks.sum(axis=-1).astype(np.int64)
    incorrect_aux_weights = (
        data.batch["incorrect_stop_aux_weight"]
        .detach().sum(dim=-1).cpu().numpy()
    )
    incorrect_aux_penalties = (
        data.batch["incorrect_stop_aux_penalty"]
        .detach().sum(dim=-1).cpu().numpy()
    )
    if np.any(incorrect_counts != incorrect_action_counts):
        raise ValueError(
            "incorrect stop count disagrees with its action mask"
        )
    if np.any(
        recognized_correct_probe_counts
        + incorrect_counts
        + neutral_probe_stop_counts
        != probed_counts
    ):
        raise ValueError(
            "recognized-correct + incorrect + neutral probe counts must "
            "equal probed_stop_count"
        )
    if np.any(incorrect_counts < 0) or np.any(incorrect_counts > 8):
        raise ValueError("incorrect stop count is outside the eight-attempt budget")
    if np.any(incorrect_aux_weights < 0) or np.any(
        incorrect_aux_weights > incorrect_counts + 1e-6
    ):
        raise ValueError(
            "incorrect stop weights must be an unnormalized per-action sum"
        )
    post_accepted_tail_action_counts = (
        data.batch["post_accepted_tail_aux_mask"]
        .detach().sum(dim=-1).cpu().numpy().astype(np.int64)
    )
    post_accepted_tail_aux_weights = (
        data.batch["post_accepted_tail_aux_weight"]
        .detach().sum(dim=-1).cpu().numpy()
    )
    post_accepted_tail_aux_penalties = (
        data.batch["post_accepted_tail_aux_penalty"]
        .detach().sum(dim=-1).cpu().numpy()
    )
    expected_negative_weight_sums = (negative_event_counts > 0).astype(
        np.float64
    )
    if not np.allclose(
        negative_aux_weights,
        expected_negative_weight_sums,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "negative stop auxiliary weights must sum to one exactly on "
            "rows with negative events"
        )
    expected_tail_weight_sums = np.minimum(
        post_accepted_tail_action_counts / 256.0,
        1.0,
    )
    if not np.allclose(
        post_accepted_tail_aux_weights,
        expected_tail_weight_sums,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "post-accepted tail auxiliary weight sum disagrees with "
            "min(action_count / 256, 1)"
        )
    verified_action_counts = (
        data.batch["verified_stop_mask"]
        .detach()
        .sum(dim=-1)
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    reported_verified_counts = (
        data.batch["verified_stop_count"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    if np.any(reported_verified_counts != verified_action_counts):
        raise ValueError(
            "verified stop action mask disagrees with verified_stop_count"
        )
    gold_selected_rows = selected_kinds == "gold_correct"
    fallback_selected_rows = selected_kinds == "final_consistent"
    if np.any(
        gold_selected_rows
        & (reported_verified_counts > recognized_correct_probe_counts)
    ):
        raise ValueError(
            "gold-correct selected count cannot exceed recognized-correct probes"
        )
    verified_aux_weights = (
        data.batch["verified_stop_aux_weight"]
        .detach()
        .sum(dim=-1)
        .cpu()
        .numpy()
    )
    reported_verified_scores = (
        data.batch["verified_stop_score"].detach().cpu().numpy()
    )
    rollout_verified_scores = (
        data.batch["early_stop_score"].detach().cpu().numpy()
    )
    if not np.allclose(
        reported_verified_scores,
        rollout_verified_scores,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "verified_stop_score disagrees with early_stop_score"
        )
    verified_aux_bonuses = (
        data.batch["verified_stop_aux_bonus"]
        .detach()
        .sum(dim=-1)
        .cpu()
        .numpy()
    )
    verified_aux_suppressed = (
        data.batch["verified_stop_aux_suppressed"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    auxiliary_penalties = (
        data.batch["overflow_stop_aux_penalty"]
        .detach()
        .sum(dim=-1)
        .cpu()
        .numpy()
    )
    rewards = (
        data.batch["token_level_scores"].detach().sum(dim=-1).cpu().numpy()
    )
    responses = data.batch["responses"]
    response_length = responses.size(1)
    response_lengths = (
        data.batch["attention_mask"][:, -response_length:]
        .detach()
        .sum(dim=-1)
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    def _integer_row_values(key: str) -> np.ndarray:
        tensor = data.batch[key]
        if tensor.ndim != 1 or int(tensor.size(0)) != row_count:
            raise ValueError(
                f"{key} must contain exactly one value per rollout row"
            )
        if torch.is_complex(tensor):
            raise ValueError(f"{key} must be real-valued")
        if torch.is_floating_point(tensor) and (
            not torch.all(torch.isfinite(tensor))
            or torch.any(tensor != torch.floor(tensor))
        ):
            raise ValueError(f"{key} must contain finite integers")
        return tensor.detach().cpu().numpy().astype(np.int64)

    effective_response_lengths = _integer_row_values(
        "effective_response_length"
    )
    controller_terminated = _integer_row_values("controller_terminated")
    controller_wrap_rejected_stop_counts = _integer_row_values(
        "controller_wrap_rejected_stop_count"
    )
    controller_discarded_token_counts = _integer_row_values(
        "controller_discarded_token_count"
    )
    controller_forced_suffix_token_counts = _integer_row_values(
        "controller_forced_suffix_token_count"
    )
    physical_response_lengths = _integer_row_values(
        "physical_response_length"
    )
    physical_raw_counts = _integer_row_values("physical_raw_stop_count")
    physical_raw_atomic_counts = _integer_row_values(
        "physical_raw_atomic_stop_count"
    )
    physical_illegal_surface_counts = _integer_row_values(
        "physical_illegal_surface_stop_count"
    )
    physical_policy_acc = _integer_row_values("physical_policy_acc")
    effective_policy_acc = _integer_row_values("raw_policy_acc")
    accepted_prefix_lengths = (
        data.batch["accepted_stop_prefix_length"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    tail_cut_tokens = (
        data.batch["tail_cut_tokens"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    tail_cut_enabled = (
        data.batch["accepted_stop_tail_cut_enabled"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    post_accepted_active_token_counts = (
        data.batch["post_accepted_active_token_count"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    verified_stop_positions = (
        data.batch["verified_stop_position"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    selected_rows_for_score = verified_stop_positions >= 0
    expected_selected_scores = np.zeros(row_count, dtype=np.float64)
    raw_half_lives = data.non_tensor_batch.get("stop_reward_half_life_tokens")
    if raw_half_lives is None:
        stop_half_lives = np.full(row_count, 1024.0, dtype=np.float64)
    else:
        stop_half_lives = np.asarray(raw_half_lives, dtype=np.float64)
        if stop_half_lives.ndim == 0:
            stop_half_lives = np.full(
                row_count, float(stop_half_lives.item()), dtype=np.float64
            )
    if (
        stop_half_lives.ndim != 1
        or stop_half_lives.size != row_count
        or not np.all(np.isfinite(stop_half_lives))
        or np.any(stop_half_lives <= 0.0)
    ):
        raise ValueError(
            "stop_reward_half_life_tokens must contain one finite positive "
            "value per rollout"
        )
    if np.any(selected_rows_for_score):
        selected_earliness = np.power(
            2.0,
            -verified_stop_positions[selected_rows_for_score].astype(np.float64)
            / stop_half_lives[selected_rows_for_score],
        )
        expected_selected_scores[selected_rows_for_score] = selected_earliness
    if not np.allclose(
        reported_verified_scores,
        expected_selected_scores,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "selected stop scores violate the shared exponential earliness "
            "contract"
        )
    oracle_accepted_stop_positions = (
        data.batch["oracle_accepted_stop_position"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    oracle_accepted_rows = oracle_accepted_stop_positions >= 0
    oracle_accepted_stop_ordinals_available = (
        "oracle_accepted_stop_ordinal" in data.batch
    )
    if oracle_accepted_stop_ordinals_available:
        oracle_accepted_stop_ordinals = (
            data.batch["oracle_accepted_stop_ordinal"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        if np.any(
            oracle_accepted_rows != (oracle_accepted_stop_ordinals >= 1)
        ) or np.any(oracle_accepted_stop_ordinals < -1):
            raise ValueError(
                "oracle accepted stop position and ordinal telemetry disagree"
            )
        accepted_indices = np.flatnonzero(oracle_accepted_rows)
        if accepted_indices.size:
            accepted_positions = oracle_accepted_stop_positions[
                accepted_indices
            ]
            if np.any(accepted_positions >= stop_event_multiplicity.shape[1]):
                raise ValueError(
                    "oracle accepted stop position exceeds event telemetry"
                )
            cumulative_event_counts = np.cumsum(
                stop_event_multiplicity,
                axis=-1,
            )
            accepted_upper_ordinals = cumulative_event_counts[
                accepted_indices,
                accepted_positions,
            ]
            accepted_lower_ordinals = (
                accepted_upper_ordinals
                - stop_event_multiplicity[
                    accepted_indices,
                    accepted_positions,
                ]
            )
            accepted_ordinals = oracle_accepted_stop_ordinals[
                accepted_indices
            ]
            if np.any(
                (accepted_ordinals <= accepted_lower_ordinals)
                | (accepted_ordinals > accepted_upper_ordinals)
            ):
                raise ValueError(
                    "oracle accepted stop ordinal is inconsistent with the "
                    "unified event stream"
                )
    else:
        oracle_accepted_stop_ordinals = np.full(
            row_count,
            -1,
            dtype=np.int64,
        )
    if np.any(
        (verified_stop_positions >= 0)
        & (verified_stop_positions != oracle_accepted_stop_positions)
    ):
        raise ValueError(
            "verified stop position must agree with the oracle-accepted "
            "position whenever positive credit survives the output gate"
        )
    policy_mask = data.batch["response_mask"].detach().bool()
    atomic_stop_mask = data.batch["atomic_stop_mask"].detach().bool()
    negative_stop_mask = data.batch["negative_stop_aux_mask"].detach().bool()
    incorrect_stop_mask = data.batch[
        "incorrect_stop_aux_mask"
    ].detach().bool()
    post_accepted_tail_mask = data.batch[
        "post_accepted_tail_aux_mask"
    ].detach().bool()
    base_policy_stop_action_counts = (
        (policy_mask & atomic_stop_mask)
        .sum(dim=-1)
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    if np.any(base_policy_stop_action_counts != 0):
        raise ValueError(
            "trajectory GRPO must exclude every atomic stop action"
        )
    base_policy_negative_action_counts = (
        (policy_mask & negative_stop_mask)
        .sum(dim=-1)
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    if np.any(base_policy_negative_action_counts != 0):
        raise ValueError(
            "trajectory GRPO must exclude every negative stop action anchor"
        )
    if torch.any(negative_stop_mask & post_accepted_tail_mask):
        raise ValueError(
            "negative stop and post-accepted tail masks must be disjoint"
        )
    if torch.any(incorrect_stop_mask & negative_stop_mask):
        raise ValueError(
            "incorrect and generalized-negative stop masks must be disjoint"
        )
    if torch.any(incorrect_stop_mask & post_accepted_tail_mask):
        raise ValueError(
            "incorrect stop and post-accepted tail masks must be disjoint"
        )
    if torch.any(
        incorrect_stop_mask
        & data.batch["verified_stop_mask"].detach().bool()
    ):
        raise ValueError(
            "incorrect and verified stop masks must be disjoint"
        )
    base_policy_incorrect_stop_action_counts = (
        (policy_mask & incorrect_stop_mask)
        .sum(dim=-1).cpu().numpy().astype(np.int64)
    )
    if np.any(base_policy_incorrect_stop_action_counts != 0):
        raise ValueError(
            "trajectory GRPO must exclude every incorrect stop action"
        )
    actor_token_counts = (
        policy_mask.sum(dim=-1).cpu().numpy().astype(np.int64)
    )
    if np.any(effective_response_lengths != response_lengths):
        raise ValueError(
            "effective_response_length disagrees with the response attention mask"
        )
    for name, values in (
        (
            "controller_wrap_rejected_stop_count",
            controller_wrap_rejected_stop_counts,
        ),
        ("controller_discarded_token_count", controller_discarded_token_counts),
        (
            "controller_forced_suffix_token_count",
            controller_forced_suffix_token_counts,
        ),
        ("physical_response_length", physical_response_lengths),
        ("physical_raw_stop_count", physical_raw_counts),
        ("physical_raw_atomic_stop_count", physical_raw_atomic_counts),
        (
            "physical_illegal_surface_stop_count",
            physical_illegal_surface_counts,
        ),
    ):
        if np.any(values < 0):
            raise ValueError(f"{name} must be non-negative")
    for name, values in (
        ("controller_terminated", controller_terminated),
        ("physical_policy_acc", physical_policy_acc),
        ("raw_policy_acc", effective_policy_acc),
    ):
        if np.any((values != 0) & (values != 1)):
            raise ValueError(f"{name} must be binary")
    if np.any(
        physical_raw_counts
        != physical_raw_atomic_counts + physical_illegal_surface_counts
    ):
        raise ValueError(
            "physical_raw_stop_count must equal physical atomic plus illegal "
            "surface events"
        )
    if np.any(accepted_prefix_lengths < 0):
        raise ValueError("accepted stop prefix length must be non-negative")
    if np.any(accepted_prefix_lengths > response_lengths):
        raise ValueError("accepted stop prefix exceeds effective response length")
    if np.any((tail_cut_enabled != 0) & (tail_cut_enabled != 1)):
        raise ValueError("accepted-stop tail-cut flag must be binary")
    selected_rows = verified_stop_positions >= 0
    if np.any(controller_terminated != selected_rows.astype(np.int64)):
        raise ValueError(
            "controller termination must exactly match the verified stop row"
        )
    if np.any(controller_terminated != oracle_accepted_rows.astype(np.int64)):
        raise ValueError(
            "controller termination must exactly match oracle acceptance"
        )
    if np.any(
        (controller_terminated == 1)
        & (tail_cut_enabled != 1)
    ):
        raise ValueError(
            "controller-terminated rows require accepted-stop tail cutting"
        )
    if np.any(
        selected_rows
        & (accepted_prefix_lengths != verified_stop_positions + 1)
    ):
        raise ValueError("accepted prefix does not end at selected stop")
    if np.any(
        (~selected_rows) & (accepted_prefix_lengths != response_lengths)
    ):
        raise ValueError(
            "non-terminated rollout prefix must equal its effective response length"
        )
    expected_effective_lengths = np.where(
        controller_terminated == 1,
        accepted_prefix_lengths + controller_forced_suffix_token_counts,
        physical_response_lengths,
    )
    if np.any(effective_response_lengths != expected_effective_lengths):
        raise ValueError(
            "effective response length disagrees with the controller-forced suffix"
        )
    expected_physical_lengths = np.where(
        controller_terminated == 1,
        accepted_prefix_lengths + controller_discarded_token_counts,
        effective_response_lengths,
    )
    if np.any(physical_response_lengths != expected_physical_lengths):
        raise ValueError(
            "physical response length disagrees with the discarded sampled tail"
        )
    if np.any(tail_cut_tokens != controller_discarded_token_counts):
        raise ValueError(
            "tail_cut_tokens must mean sampled physical tokens discarded by "
            "the external controller"
        )
    nonterminated_rows = controller_terminated == 0
    if np.any(
        nonterminated_rows
        & (
            (controller_discarded_token_counts != 0)
            | (controller_forced_suffix_token_counts != 0)
            | (tail_cut_tokens != 0)
        )
    ):
        raise ValueError(
            "non-terminated rollouts cannot report discarded or forced tokens"
        )
    if np.any(
        (controller_terminated == 1)
        & (controller_forced_suffix_token_counts <= 0)
    ):
        raise ValueError(
            "controller-terminated rollouts require a non-empty forced suffix"
        )
    if np.any(
        nonterminated_rows
        & (
            (physical_raw_counts != raw_counts)
            | (physical_raw_atomic_counts != raw_atomic_counts)
            | (physical_illegal_surface_counts != illegal_surface_counts)
            | (physical_policy_acc != effective_policy_acc)
        )
    ):
        raise ValueError(
            "non-terminated physical and effective rollout telemetry must match"
        )
    terminated_rows = controller_terminated == 1
    if np.any(
        terminated_rows
        & (
            (physical_raw_counts < raw_counts)
            | (physical_raw_atomic_counts < raw_atomic_counts)
            | (physical_illegal_surface_counts < illegal_surface_counts)
        )
    ):
        raise ValueError(
            "controller cropping cannot create new stop events"
        )
    if np.any(terminated_rows & gold_selected_rows & (effective_policy_acc != 1)):
        raise ValueError(
            "gold-correct controller final answers must be oracle-correct"
        )
    fallback_terminated_rows = terminated_rows & fallback_selected_rows
    if np.any(fallback_terminated_rows):
        final_choice_rows = {}
        for key in (
            "physical_final_choice",
            "effective_final_choice",
            "selected_probe_answer",
        ):
            raw_values = data.non_tensor_batch.get(key)
            if raw_values is None:
                raise KeyError(
                    f"final-consistent controller audit requires {key}"
                )
            values = np.asarray(raw_values, dtype=object)
            if values.ndim != 1 or values.size != row_count:
                raise ValueError(
                    f"{key} must contain one value per rollout"
                )
            final_choice_rows[key] = values
        fallback_indices = np.flatnonzero(fallback_terminated_rows)
        for row_index in fallback_indices.tolist():
            physical_choice = final_choice_rows["physical_final_choice"][row_index]
            effective_choice = final_choice_rows["effective_final_choice"][row_index]
            selected_answer = final_choice_rows["selected_probe_answer"][row_index]
            if (
                physical_choice is None
                or effective_choice != physical_choice
                or selected_answer != physical_choice
                or effective_policy_acc[row_index] != physical_policy_acc[row_index]
            ):
                raise ValueError(
                    "final-consistent controller stop must preserve the strict "
                    "natural final answer and accuracy"
                )
    if np.any(post_accepted_active_token_counts != 0):
        raise ValueError("ordinary actor gradient leaked after accepted stop")
    if np.any(post_accepted_tail_action_counts != 0) or not np.allclose(
        post_accepted_tail_aux_weights, 0.0, rtol=0.0, atol=0.0
    ) or not np.allclose(
        post_accepted_tail_aux_penalties, 0.0, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "controller-terminal rollouts must disable every post-accepted "
            "tail auxiliary action, weight, and penalty"
        )
    oracle_response_suffix_token_counts = np.where(
        oracle_accepted_rows,
        response_lengths - oracle_accepted_stop_positions - 1,
        0,
    )
    if np.any(oracle_response_suffix_token_counts < 0):
        raise ValueError("oracle accepted stop extends beyond the response")
    if np.any(
        oracle_response_suffix_token_counts
        != controller_forced_suffix_token_counts
    ):
        raise ValueError(
            "effective post-stop suffix must be exactly the controller-forced "
            "canonical closure"
        )
    controller_net_token_savings = (
        controller_discarded_token_counts
        - controller_forced_suffix_token_counts
    )
    advantages = data.batch["advantages"].detach()
    verified_mask = data.batch["verified_stop_mask"].detach().bool()
    overflow_mask = data.batch["overflow_stop_mask"].detach().bool()
    rollout_indices = (
        data.batch["rollout_index"].detach().cpu().numpy().astype(np.int64)
    )
    uids = [str(value) for value in data.non_tensor_batch["uid"]]
    if len(uids) != row_count:
        raise ValueError(f"uid count mismatch: {len(uids)} != {row_count}")

    def _probe_position_rows(key, expected_counts):
        if key not in data.non_tensor_batch:
            raise KeyError(f"rollout stop audit requires {key}")
        values = data.non_tensor_batch[key]
        if len(values) != row_count:
            raise ValueError(
                f"{key} row count mismatch: {len(values)} != {row_count}"
            )
        normalized = []
        for row_index, raw_positions in enumerate(values):
            if isinstance(raw_positions, np.ndarray):
                raw_positions = raw_positions.tolist()
            if not isinstance(raw_positions, (list, tuple)):
                raise ValueError(f"{key}[{row_index}] must be a position list")
            positions = [int(position) for position in raw_positions]
            if len(positions) != int(expected_counts[row_index]):
                raise ValueError(
                    f"{key}[{row_index}] count mismatch: {len(positions)} != "
                    f"{int(expected_counts[row_index])}"
                )
            if any(
                current <= previous
                for previous, current in zip(positions, positions[1:])
            ):
                raise ValueError(
                    f"{key}[{row_index}] must be unique and strictly increasing"
                )
            if any(
                position < 0 or position >= int(response_lengths[row_index])
                for position in positions
            ):
                raise ValueError(
                    f"{key}[{row_index}] contains a position outside the response"
                )
            if any(
                atomic_stop_event_multiplicity[row_index, position] != 1
                for position in positions
            ):
                raise ValueError(
                    f"{key}[{row_index}] must contain only atomic stop positions"
                )
            normalized.append(positions)
        return normalized

    recognized_correct_probe_position_rows = _probe_position_rows(
        "recognized_correct_probe_positions",
        recognized_correct_probe_counts,
    )
    incorrect_stop_position_rows = _probe_position_rows(
        "incorrect_stop_positions",
        incorrect_counts,
    )
    neutral_probe_stop_position_rows = _probe_position_rows(
        "neutral_probe_stop_positions",
        neutral_probe_stop_counts,
    )
    for row_index in range(row_count):
        expected_incorrect_positions = np.flatnonzero(
            incorrect_stop_masks[row_index]
        ).astype(np.int64).tolist()
        if incorrect_stop_position_rows[row_index] != expected_incorrect_positions:
            raise ValueError(
                "incorrect_stop_positions disagrees with incorrect_stop_aux_mask"
            )
        classified_positions = (
            recognized_correct_probe_position_rows[row_index]
            + incorrect_stop_position_rows[row_index]
            + neutral_probe_stop_position_rows[row_index]
        )
        if len(set(classified_positions)) != int(probed_counts[row_index]):
            raise ValueError(
                "correct/incorrect/neutral probe position lists must be disjoint "
                "and cover every probed stop"
            )

    optional_keys = (
        "stop_over_limit",
        "verified_stop",
        "verified_stop_count",
        "verified_stop_score",
        "selected_stop_ordinal",
        "selected_stop_kind",
        "selected_stop_credit",
        "final_consistency_credit",
        "strict_raw_acc",
        "strict_final_valid",
        "eligible_atomic_stop_present",
        "required_output_valid",
        "output_requirement_gated",
        "reward_gated",
        "format_score",
        "stop_presence_format_score",
        "structural_format_score",
        "stop_surface_count_matches_raw",
        "all_atomic_stops_eligible",
        "no_illegal_surface_stops",
        "negative_stop_action_penalty_required",
        "format_reward",
        "accuracy_reward",
        "stop_earliness",
        "stop_reward",
        "score",
        "dapo_filter_enabled",
        "dapo_group_kept",
        "dapo_generation_batch",
        "dapo_overlong_penalty",
        "dapo_overlong",
        "dapo_overlong_length",
    )
    optional_values = {
        key: data.non_tensor_batch.get(key) for key in optional_keys
    }
    run_id = os.environ.get("ADAPTTHINK_RUN_ID", "")
    records = []
    for row_index in range(row_count):
        uid = uids[row_index]
        active_advantages = advantages[row_index][policy_mask[row_index]]
        verified_advantages = advantages[row_index][verified_mask[row_index]]
        overflow_advantages = advantages[row_index][overflow_mask[row_index]]
        negative_stop_advantages = advantages[row_index][
            negative_stop_mask[row_index]
        ]
        incorrect_stop_advantages = advantages[row_index][
            incorrect_stop_mask[row_index]
        ]
        post_accepted_tail_advantages = advantages[row_index][
            post_accepted_tail_mask[row_index]
        ]
        record = {
            "run_id": run_id,
            "step": int(step),
            "row_index": int(row_index),
            "uid": uid,
            "rollout_index": int(rollout_indices[row_index]),
            "raw_stop_count": int(raw_counts[row_index]),
            "raw_atomic_stop_count": int(raw_atomic_counts[row_index]),
            "illegal_surface_stop_count": int(
                illegal_surface_counts[row_index]
            ),
            "ordinary_surface_stop_count": int(
                ordinary_surface_counts[row_index]
            ),
            "stop_fragment_count": int(fragment_counts[row_index]),
            "overflow_atomic_stop_count": int(
                overflow_atomic_counts[row_index]
            ),
            "negative_stop_event_count": int(
                negative_event_counts[row_index]
            ),
            "stop_event_anchor_count": int(
                stop_event_anchor_counts[row_index]
            ),
            "stop_event_multiplicity_sum": int(raw_counts[row_index]),
            "surface_stop_event_anchor_count": int(
                surface_stop_event_anchor_counts[row_index]
            ),
            "surface_stop_event_multiplicity_sum": int(
                illegal_surface_counts[row_index]
            ),
            "negative_stop_event_anchor_count": int(
                negative_stop_event_anchor_counts[row_index]
            ),
            "negative_stop_event_multiplicity_sum": int(
                negative_event_counts[row_index]
            ),
            "eligible_stop_count": int(eligible_counts[row_index]),
            "probed_stop_count": int(probed_counts[row_index]),
            "recognized_correct_probe_count": int(
                recognized_correct_probe_counts[row_index]
            ),
            "recognized_correct_probe_positions": (
                recognized_correct_probe_position_rows[row_index]
            ),
            "neutral_probe_stop_count": int(
                neutral_probe_stop_counts[row_index]
            ),
            "neutral_probe_stop_positions": (
                neutral_probe_stop_position_rows[row_index]
            ),
            "verified_stop_action_count": int(
                verified_action_counts[row_index]
            ),
            "verified_stop_count": int(
                reported_verified_counts[row_index]
            ),
            "verified_stop_score": float(
                reported_verified_scores[row_index]
            ),
            "selected_stop_kind": str(selected_kinds[row_index]),
            "selected_stop_credit": float(selected_credits[row_index]),
            "final_consistency_credit": float(
                final_consistency_credits[row_index]
            ),
            "verified_stop_aux_weight_sum": float(
                verified_aux_weights[row_index]
            ),
            "verified_stop_aux_bonus_sum": float(
                verified_aux_bonuses[row_index]
            ),
            "verified_stop_aux_suppressed": int(
                verified_aux_suppressed[row_index]
            ),
            "base_policy_stop_action_count": int(
                base_policy_stop_action_counts[row_index]
            ),
            "overflow_stop_action_count": int(
                overflow_action_counts[row_index]
            ),
            "overflow_stop_aux_penalty_sum": float(
                auxiliary_penalties[row_index]
            ),
            "negative_stop_aux_action_count": int(
                negative_action_counts[row_index]
            ),
            "negative_stop_aux_weight_sum": float(
                negative_aux_weights[row_index]
            ),
            "negative_stop_aux_penalty_sum": float(
                negative_aux_penalties[row_index]
            ),
            "base_policy_negative_stop_action_count": int(
                base_policy_negative_action_counts[row_index]
            ),
            "incorrect_stop_count": int(incorrect_counts[row_index]),
            "incorrect_stop_positions": incorrect_stop_position_rows[row_index],
            "incorrect_stop_aux_action_count": int(
                incorrect_action_counts[row_index]
            ),
            "incorrect_stop_aux_weight_sum": float(
                incorrect_aux_weights[row_index]
            ),
            "incorrect_stop_aux_penalty_sum": float(
                incorrect_aux_penalties[row_index]
            ),
            "base_policy_incorrect_stop_action_count": int(
                base_policy_incorrect_stop_action_counts[row_index]
            ),
            "post_accepted_tail_aux_action_count": int(
                post_accepted_tail_action_counts[row_index]
            ),
            "post_accepted_tail_aux_weight_sum": float(
                post_accepted_tail_aux_weights[row_index]
            ),
            "post_accepted_tail_aux_penalty_sum": float(
                post_accepted_tail_aux_penalties[row_index]
            ),
            "reward": float(rewards[row_index]),
            # ``response_length`` remains as a compatibility alias for the
            # controller-visible/effective trajectory stored in DataProto.
            "response_length": int(response_lengths[row_index]),
            "effective_response_length": int(
                effective_response_lengths[row_index]
            ),
            "physical_response_length": int(
                physical_response_lengths[row_index]
            ),
            "controller_terminated": int(
                controller_terminated[row_index]
            ),
            "controller_wrap_rejected_stop_count": int(
                controller_wrap_rejected_stop_counts[row_index]
            ),
            "controller_discarded_token_count": int(
                controller_discarded_token_counts[row_index]
            ),
            "controller_forced_suffix_token_count": int(
                controller_forced_suffix_token_counts[row_index]
            ),
            "controller_net_token_savings": int(
                controller_net_token_savings[row_index]
            ),
            "effective_raw_stop_count": int(raw_counts[row_index]),
            "effective_raw_atomic_stop_count": int(
                raw_atomic_counts[row_index]
            ),
            "effective_illegal_surface_stop_count": int(
                illegal_surface_counts[row_index]
            ),
            "effective_policy_acc": int(effective_policy_acc[row_index]),
            "physical_raw_stop_count": int(
                physical_raw_counts[row_index]
            ),
            "physical_raw_atomic_stop_count": int(
                physical_raw_atomic_counts[row_index]
            ),
            "physical_illegal_surface_stop_count": int(
                physical_illegal_surface_counts[row_index]
            ),
            "physical_policy_acc": int(physical_policy_acc[row_index]),
            "accepted_stop_position": int(
                verified_stop_positions[row_index]
            ),
            "verified_stop_position": int(
                verified_stop_positions[row_index]
            ),
            "oracle_accepted_stop": int(
                oracle_accepted_rows[row_index]
            ),
            "oracle_accepted_stop_position": int(
                oracle_accepted_stop_positions[row_index]
            ),
            "oracle_accepted_stop_ordinal": (
                int(oracle_accepted_stop_ordinals[row_index])
                if oracle_accepted_stop_ordinals_available
                else None
            ),
            "oracle_response_suffix_token_count": int(
                oracle_response_suffix_token_counts[row_index]
            ),
            "effective_controller_suffix_token_count": int(
                oracle_response_suffix_token_counts[row_index]
            ),
            "accepted_stop_tail_cut_enabled": int(
                tail_cut_enabled[row_index]
            ),
            "accepted_stop_prefix_length": int(
                accepted_prefix_lengths[row_index]
            ),
            "tail_cut_tokens": int(tail_cut_tokens[row_index]),
            "effective_actor_token_count": int(
                actor_token_counts[row_index]
            ),
            "post_accepted_active_token_count": int(
                post_accepted_active_token_counts[row_index]
            ),
            "policy_advantage_mean": (
                float(active_advantages.mean().cpu())
                if active_advantages.numel()
                else 0.0
            ),
            "verified_stop_grpo_advantage_mean": (
                float(verified_advantages.mean().cpu())
                if verified_advantages.numel()
                else 0.0
            ),
            "overflow_advantage_mean": (
                float(overflow_advantages.mean().cpu())
                if overflow_advantages.numel()
                else 0.0
            ),
            "negative_stop_grpo_advantage_mean": (
                float(negative_stop_advantages.mean().cpu())
                if negative_stop_advantages.numel()
                else 0.0
            ),
            "incorrect_stop_grpo_advantage_mean": (
                float(incorrect_stop_advantages.mean().cpu())
                if incorrect_stop_advantages.numel()
                else 0.0
            ),
            "post_accepted_tail_grpo_advantage_mean": (
                float(post_accepted_tail_advantages.mean().cpu())
                if post_accepted_tail_advantages.numel()
                else 0.0
            ),
        }
        for key, values in optional_values.items():
            if values is None or len(values) != row_count:
                continue
            value = values[row_index]
            item = getattr(value, "item", None)
            value = item() if callable(item) else value
            if isinstance(value, (np.integer, int, bool)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = float(value)
            record[key] = value
        records.append(record)

    audit_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(audit_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as audit_file:
        audit_file.write(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            )
        )
        audit_file.flush()
        os.fsync(audit_file.fileno())

    percentiles = np.percentile(raw_counts, [50, 90, 95, 99])
    physical_stop_percentiles = np.percentile(
        physical_raw_counts, [50, 90, 95, 99]
    )
    effective_length_percentiles = np.percentile(
        effective_response_lengths, [50, 90, 95, 99]
    )
    physical_length_percentiles = np.percentile(
        physical_response_lengths, [50, 90, 95, 99]
    )
    unique_counts, frequencies = np.unique(raw_counts, return_counts=True)
    histogram = {
        str(int(count)): int(frequency)
        for count, frequency in zip(unique_counts, frequencies)
    }
    print(
        "[rollout_stop_audit] "
        f"step={int(step)} rows={row_count} path={path} "
        f"controller_termination_rate={controller_terminated.mean():.6f} "
        f"wrap_rejected_sum={controller_wrap_rejected_stop_counts.sum()} "
        f"effective_length_mean={effective_response_lengths.mean():.2f} "
        f"physical_length_mean={physical_response_lengths.mean():.2f} "
        f"discarded_mean={controller_discarded_token_counts.mean():.2f} "
        f"histogram={json.dumps(histogram, sort_keys=True)}"
    )
    return {
        "reward/raw_stop_count_p50": float(percentiles[0]),
        "reward/raw_stop_count_p90": float(percentiles[1]),
        "reward/raw_stop_count_p95": float(percentiles[2]),
        "reward/raw_stop_count_p99": float(percentiles[3]),
        "reward/physical_raw_stop_count_p50": float(
            physical_stop_percentiles[0]
        ),
        "reward/physical_raw_stop_count_p90": float(
            physical_stop_percentiles[1]
        ),
        "reward/physical_raw_stop_count_p95": float(
            physical_stop_percentiles[2]
        ),
        "reward/physical_raw_stop_count_p99": float(
            physical_stop_percentiles[3]
        ),
        "reward/physical_raw_stop_count_mean": float(
            physical_raw_counts.mean()
        ),
        "reward/physical_raw_atomic_stop_count_mean": float(
            physical_raw_atomic_counts.mean()
        ),
        "reward/physical_illegal_surface_stop_count_mean": float(
            physical_illegal_surface_counts.mean()
        ),
        "reward/raw_atomic_stop_count_mean": float(
            raw_atomic_counts.mean()
        ),
        "reward/raw_atomic_stop_count_max": float(
            raw_atomic_counts.max()
        ),
        "reward/illegal_surface_stop_count_mean": float(
            illegal_surface_counts.mean()
        ),
        "reward/illegal_surface_stop_count_max": float(
            illegal_surface_counts.max()
        ),
        "reward/illegal_surface_stop_row_rate": float(
            np.mean(illegal_surface_counts > 0)
        ),
        "reward/ordinary_surface_stop_count_mean": float(
            ordinary_surface_counts.mean()
        ),
        "reward/stop_fragment_count_mean": float(fragment_counts.mean()),
        "reward/stop_fragment_count_max": float(fragment_counts.max()),
        "reward/stop_fragment_row_rate": float(
            np.mean(fragment_counts > 0)
        ),
        "reward/stop_event_anchor_count_mean": float(
            stop_event_anchor_counts.mean()
        ),
        "reward/surface_stop_event_anchor_count_mean": float(
            surface_stop_event_anchor_counts.mean()
        ),
        "reward/negative_stop_event_anchor_count_mean": float(
            negative_stop_event_anchor_counts.mean()
        ),
        "reward/overflow_atomic_stop_count_mean": float(
            overflow_atomic_counts.mean()
        ),
        "reward/negative_stop_event_count_mean": float(
            negative_event_counts.mean()
        ),
        "reward/negative_stop_event_row_rate": float(
            np.mean(negative_event_counts > 0)
        ),
        "reward/verified_stop_action_mean": float(
            verified_action_counts.mean()
        ),
        "reward/gold_correct_stop_rate": float(gold_selected_rows.mean()),
        "reward/final_consistent_stop_rate": float(
            fallback_selected_rows.mean()
        ),
        "reward/selected_stop_credit_mean": float(selected_credits.mean()),
        "reward/gold_correct_stop_score_mean": float(
            reported_verified_scores[gold_selected_rows].mean()
            if np.any(gold_selected_rows)
            else 0.0
        ),
        "reward/final_consistent_stop_score_mean": float(
            reported_verified_scores[fallback_selected_rows].mean()
            if np.any(fallback_selected_rows)
            else 0.0
        ),
        "reward/recognized_correct_probe_count_mean": float(
            recognized_correct_probe_counts.mean()
        ),
        "reward/neutral_probe_stop_count_mean": float(
            neutral_probe_stop_counts.mean()
        ),
        "reward/verified_stop_count_mean": float(
            reported_verified_counts.mean()
        ),
        "reward/verified_stop_score_mean": float(
            reported_verified_scores.mean()
        ),
        "reward/verified_stop_aux_weight_mean": float(
            verified_aux_weights.mean()
        ),
        "reward/verified_stop_aux_bonus_mean": float(
            verified_aux_bonuses.mean()
        ),
        "reward/verified_stop_aux_suppressed_mean": float(
            verified_aux_suppressed.mean()
        ),
        "reward/base_policy_stop_action_mean": float(
            base_policy_stop_action_counts.mean()
        ),
        "reward/overflow_stop_action_mean": float(
            overflow_action_counts.mean()
        ),
        "reward/overflow_stop_aux_penalty_mean": float(
            auxiliary_penalties.mean()
        ),
        "reward/negative_stop_aux_action_mean": float(
            negative_action_counts.mean()
        ),
        "reward/negative_stop_aux_weight_mean": float(
            negative_aux_weights.mean()
        ),
        "reward/negative_stop_aux_penalty_mean": float(
            negative_aux_penalties.mean()
        ),
        "reward/base_policy_negative_stop_action_mean": float(
            base_policy_negative_action_counts.mean()
        ),
        "reward/incorrect_stop_count_mean": float(incorrect_counts.mean()),
        "reward/incorrect_stop_aux_action_mean": float(
            incorrect_action_counts.mean()
        ),
        "reward/incorrect_stop_aux_weight_mean": float(
            incorrect_aux_weights.mean()
        ),
        "reward/incorrect_stop_aux_penalty_mean": float(
            incorrect_aux_penalties.mean()
        ),
        "reward/incorrect_stop_aux_row_rate": float(
            np.mean(incorrect_action_counts > 0)
        ),
        "reward/base_policy_incorrect_stop_action_mean": float(
            base_policy_incorrect_stop_action_counts.mean()
        ),
        "reward/oracle_accepted_stop_rate": float(
            oracle_accepted_rows.mean()
        ),
        "reward/oracle_accepted_stop_position_mean": float(
            oracle_accepted_stop_positions[oracle_accepted_rows].mean()
            if np.any(oracle_accepted_rows)
            else -1.0
        ),
        "reward/oracle_accepted_stop_ordinal_mean": float(
            oracle_accepted_stop_ordinals[oracle_accepted_rows].mean()
            if (
                oracle_accepted_stop_ordinals_available
                and np.any(oracle_accepted_rows)
            )
            else -1.0
        ),
        "reward/oracle_response_suffix_token_count_mean": float(
            oracle_response_suffix_token_counts.mean()
        ),
        "reward/controller_termination_rate": float(
            controller_terminated.mean()
        ),
        "reward/controller_wrap_rejected_stop_count_mean": float(
            controller_wrap_rejected_stop_counts.mean()
        ),
        "reward/controller_wrap_rejected_stop_row_rate": float(
            np.mean(controller_wrap_rejected_stop_counts > 0)
        ),
        "reward/controller_wrap_rejected_stop_count_sum": float(
            controller_wrap_rejected_stop_counts.sum()
        ),
        "reward/effective_response_length_mean": float(
            effective_response_lengths.mean()
        ),
        "reward/effective_response_length_p50": float(
            effective_length_percentiles[0]
        ),
        "reward/effective_response_length_p90": float(
            effective_length_percentiles[1]
        ),
        "reward/effective_response_length_p95": float(
            effective_length_percentiles[2]
        ),
        "reward/effective_response_length_p99": float(
            effective_length_percentiles[3]
        ),
        "reward/physical_response_length_mean": float(
            physical_response_lengths.mean()
        ),
        "reward/physical_response_length_p50": float(
            physical_length_percentiles[0]
        ),
        "reward/physical_response_length_p90": float(
            physical_length_percentiles[1]
        ),
        "reward/physical_response_length_p95": float(
            physical_length_percentiles[2]
        ),
        "reward/physical_response_length_p99": float(
            physical_length_percentiles[3]
        ),
        "reward/controller_discarded_token_count_mean": float(
            controller_discarded_token_counts.mean()
        ),
        "reward/controller_discarded_token_count_accepted_mean": float(
            controller_discarded_token_counts[terminated_rows].mean()
            if np.any(terminated_rows)
            else 0.0
        ),
        "reward/controller_forced_suffix_token_count_mean": float(
            controller_forced_suffix_token_counts.mean()
        ),
        "reward/controller_forced_suffix_token_count_accepted_mean": float(
            controller_forced_suffix_token_counts[terminated_rows].mean()
            if np.any(terminated_rows)
            else 0.0
        ),
        "reward/controller_net_token_savings_mean": float(
            controller_net_token_savings.mean()
        ),
        "reward/controller_net_token_savings_accepted_mean": float(
            controller_net_token_savings[terminated_rows].mean()
            if np.any(terminated_rows)
            else 0.0
        ),
        "reward/effective_policy_acc": float(effective_policy_acc.mean()),
        "reward/physical_policy_acc": float(physical_policy_acc.mean()),
        "reward/post_accepted_tail_aux_action_mean": float(
            post_accepted_tail_action_counts.mean()
        ),
        "reward/post_accepted_tail_aux_weight_mean": float(
            post_accepted_tail_aux_weights.mean()
        ),
        "reward/post_accepted_tail_aux_penalty_mean": float(
            post_accepted_tail_aux_penalties.mean()
        ),
        "reward/post_accepted_tail_aux_row_rate": float(
            np.mean(post_accepted_tail_action_counts > 0)
        ),
        "reward/tail_cut_row_rate": float(
            np.mean(tail_cut_tokens > 0)
        ),
        "reward/tail_cut_tokens_mean": float(tail_cut_tokens.mean()),
        "reward/accepted_stop_prefix_length_mean": float(
            accepted_prefix_lengths.mean()
        ),
        "reward/effective_actor_token_count_mean": float(
            actor_token_counts.mean()
        ),
        "reward/post_accepted_active_token_count_sum": float(
            post_accepted_active_token_counts.sum()
        ),
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


@contextmanager
def _accum_timer(name: str, timing_raw: Dict[str, float]):
    """Timer variant that includes every DAPO re-sampling batch."""

    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.0) + timer.last


class StopRayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        train_stop_classifier_online = self.config.trainer.get(
            'train_stop_classifier_online', False
        )
        if not isinstance(train_stop_classifier_online, (bool, np.bool_)):
            raise TypeError(
                'trainer.train_stop_classifier_online must be a boolean, '
                f'got {type(train_stop_classifier_online).__name__}: '
                f'{train_stop_classifier_online!r}'
            )
        self.train_stop_classifier_online = bool(train_stop_classifier_online)
        print(
            'Online A70 stop-classifier training: '
            f'{"ENABLED (explicit opt-in)" if self.train_stop_classifier_online else "DISABLED"}'
        )

        self.params_clf = dict(
            objective="binary",
            n_estimators=400,
            learning_rate=0.07,
            num_leaves=63,          # roughly like max_depth=6 in XGB
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=16,
            verbosity=-1
        )
        self.classifier_final_state = StopClassifierState(
            self.params_clf,
            feature_dim=STOP_CLASSIFIER_FEATURE_DIM,
            target=STOP_CLASSIFIER_TARGET_FINAL,
            key_x='classifier_final_feature_x',
            key_y='classifier_final_feature_y',
            metric_prefix='classifier_final',
        )
        self.classifier_gold_or_final_state = StopClassifierState(
            self.params_clf,
            feature_dim=STOP_CLASSIFIER_FEATURE_DIM,
            target=STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL,
            key_x='classifier_gold_or_final_feature_x',
            key_y='classifier_gold_or_final_feature_y',
            metric_prefix='classifier_gold_or_final',
        )
        self.stop_classifier_audit_path = str(
            self.config.trainer.get('stop_classifier_audit_path', '') or ''
        )
        self.stop_classifier_final_live_checkpoint_path = str(
            self.config.trainer.get(
                'stop_classifier_final_live_checkpoint_path', ''
            ) or ''
        )
        self.stop_classifier_gold_or_final_live_checkpoint_path = str(
            self.config.trainer.get(
                'stop_classifier_gold_or_final_live_checkpoint_path', ''
            ) or ''
        )
        if self.train_stop_classifier_online:
            for name, path in (
                ('trainer.stop_classifier_audit_path', self.stop_classifier_audit_path),
                (
                    'trainer.stop_classifier_final_live_checkpoint_path',
                    self.stop_classifier_final_live_checkpoint_path,
                ),
                (
                    'trainer.stop_classifier_gold_or_final_live_checkpoint_path',
                    self.stop_classifier_gold_or_final_live_checkpoint_path,
                ),
            ):
                if not path or not os.path.isabs(path):
                    raise ValueError(f'{name} must be a non-empty absolute path')
        # Backward-compatible handle for any offline inspection code. It is
        # never sent to rollout generation or used for an online GT decision.
        self.clf = self.classifier_gold_or_final_state.model
        self.train_flag = 0

    def _record_stop_classifier_batch(self, metrics):
        """Persist a live model and one prequential audit row per generation batch."""
        if not self.train_stop_classifier_online:
            return
        audit_parent = os.path.dirname(self.stop_classifier_audit_path)
        for path in (
            self.stop_classifier_final_live_checkpoint_path,
            self.stop_classifier_gold_or_final_live_checkpoint_path,
        ):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(audit_parent, exist_ok=True)
        self.classifier_final_state.save(
            self.stop_classifier_final_live_checkpoint_path
        )
        self.classifier_gold_or_final_state.save(
            self.stop_classifier_gold_or_final_live_checkpoint_path
        )

        def json_value(value):
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        payload = {
            'timestamp_unix': time.time(),
            'global_step': int(self.global_steps),
            'generation_batch': int(
                self.classifier_final_state.generation_batch_count
            ),
            'schema_version': STOP_CLASSIFIER_SCHEMA_VERSION,
            'targets': [
                STOP_CLASSIFIER_TARGET_FINAL,
                STOP_CLASSIFIER_TARGET_GOLD_OR_FINAL,
            ],
            'source_records': 'all_physical_records_by_candidate',
            'classifier_affects_dapo': False,
        }
        payload.update({key: json_value(value) for key, value in metrics.items()})
        with open(self.stop_classifier_audit_path, 'a', encoding='utf-8') as audit:
            audit.write(json.dumps(payload, sort_keys=True) + '\n')
            audit.flush()
            os.fsync(audit.fileno())

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type
        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                                f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        elif self.config.trainer.resume_mode == "resume_path":
            # In epoch-bounded continuation mode the configured epoch count is
            # additional loop budget after loading the checkpoint.  Keep the
            # progress/scheduler horizon above the restored global step; the
            # dataloader boundary, rather than this conservative upper bound,
            # terminates the continuation.
            resume_path = str(self.config.trainer.resume_from_path or "")
            resume_name = os.path.basename(os.path.normpath(resume_path))
            if not resume_name.startswith("global_step_"):
                raise ValueError(
                    "epoch-bounded resume requires a global_step_* checkpoint"
                )
            try:
                resume_step = int(resume_name.removeprefix("global_step_"))
            except ValueError as exc:
                raise ValueError(
                    f"invalid resume checkpoint step: {resume_name}"
                ) from exc
            total_training_steps += resume_step

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if var_name == core_var and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "std", "maj", "best"]) and f"@{n_max}/" in metric_name:
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        os.makedirs(local_global_step_folder, exist_ok=True)

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if self.train_stop_classifier_online:
            classifier_final_local_path = os.path.join(
                local_global_step_folder, 'stop_classifier_final_consistency.pkl')
            classifier_gold_or_final_local_path = os.path.join(
                local_global_step_folder, 'stop_classifier_gold_or_final.pkl')
            self.classifier_final_state.save(classifier_final_local_path)
            self.classifier_gold_or_final_state.save(
                classifier_gold_or_final_local_path
            )
            print(
                'Saved dual stop classifiers: '
                f'final={classifier_final_local_path} '
                f'(samples={len(self.classifier_final_state.feature_y)}, '
                f'updates={self.classifier_final_state.update_count}); '
                f'gold_or_final={classifier_gold_or_final_local_path} '
                f'(samples={len(self.classifier_gold_or_final_state.feature_y)}, '
                f'updates={self.classifier_gold_or_final_state.update_count})')
        else:
            print('Skipped stop-classifier checkpoint: online training is disabled')

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        if self.train_stop_classifier_online:
            classifier_paths = (
                (
                    self.classifier_final_state,
                    os.path.join(
                        global_step_folder,
                        'stop_classifier_final_consistency.pkl',
                    ),
                ),
                (
                    self.classifier_gold_or_final_state,
                    os.path.join(
                        global_step_folder,
                        'stop_classifier_gold_or_final.pkl',
                    ),
                ),
            )
            missing = [path for _state, path in classifier_paths if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    'dual classifier resume requires both checkpoints; missing: '
                    + ', '.join(missing)
                )
            for classifier_state, classifier_path in classifier_paths:
                classifier_state.load(classifier_path)
                print(
                    f'Loaded stop classifier from {classifier_path} '
                    f'(target={classifier_state.target}, '
                    f'samples={len(classifier_state.feature_y)}, '
                    f'updates={classifier_state.update_count})')
            self.clf = self.classifier_gold_or_final_state.model
            self.train_flag = int(
                self.classifier_gold_or_final_state.is_trained
            )
        else:
            print('Skipped stop-classifier restore: online training is disabled')

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """Run PPO/GRPO with optional DAPO dynamic sampling and soft overlong."""
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        rollout_stop_audit_path = os.environ.get(
            "ADAPTTHINK_ROLLOUT_STOP_AUDIT_PATH", ""
        ).strip()
        require_rollout_stop_audit = bool(
            self.config.trainer.get("require_rollout_stop_audit", False)
        )
        if require_rollout_stop_audit and not rollout_stop_audit_path:
            raise RuntimeError(
                "ADAPTTHINK_ROLLOUT_STOP_AUDIT_PATH must be configured so every "
                "rollout is audited starting at step 1"
            )
        if rollout_stop_audit_path:
            print(f"rollout stop audit path: {rollout_stop_audit_path}")

        dapo_cfg = resolve_dapo_runtime_config(self.config)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        target_prompt_count = int(self.config.data.train_batch_size)
        max_response_length = int(self.config.data.max_response_length)
        if target_prompt_count <= 0:
            raise ValueError("data.train_batch_size must be positive")
        if dapo_cfg["filter_enabled"]:
            if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                raise ValueError(
                    "DAPO dynamic sampling in the stop trainer requires GRPO"
                )
            if rollout_n <= 1:
                raise ValueError("DAPO dynamic sampling requires rollout.n > 1")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError(
                "the v18 DAPO path removes KL-in-reward; disable "
                "algorithm.use_kl_in_reward"
            )
        print(
            "DAPO runtime: "
            f"filter_enabled={dapo_cfg['filter_enabled']} "
            f"metric={dapo_cfg['filter_metric']} "
            f"max_num_gen_batches={dapo_cfg['max_num_gen_batches']} "
            f"overlong_enabled={dapo_cfg['overlong_enabled']} "
            f"overlong_buffer_len={dapo_cfg['overlong_buffer_len']} "
            f"overlong_penalty_factor={dapo_cfg['overlong_penalty_factor']} "
            f"target_prompts={target_prompt_count} rollout_n={rollout_n}"
        )

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )
        self.global_steps += 1
        last_completed_step = self.global_steps - 1
        last_val_metrics = None

        # State is reset only after an optimizer update.  A rejected generation
        # batch consumes data and GPU work but must not advance global_steps.
        pending_batch = None
        pending_prompt_count = 0
        num_gen_batches = 0
        generated_prompt_count = 0
        generated_row_count = 0
        kept_prompt_count = 0
        discarded_prompt_count = 0
        scalar_metric_state = {}
        metrics = {}
        timing_raw = {}
        step_started_at = time.monotonic()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                source_batch: DataProto = DataProto.from_single_dict(batch_dict)
                source_prompt_count = len(source_batch)
                num_gen_batches += 1

                if "multi_modal_inputs" in source_batch.non_tensor_batch:
                    gen_batch = source_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = source_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "reward_model"],
                    )

                with _accum_timer("gen", timing_raw):
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(
                        gen_batch
                    )

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with _accum_timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = (
                            self.actor_rollout_wg.generate_sequences(
                                gen_baseline_batch
                            )
                        )
                        source_batch = source_batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(source_batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                        source_batch.pop(
                            batch_keys=list(gen_baseline_output.batch.keys())
                        )
                        source_batch.batch["reward_baselines"] = (
                            reward_baseline_tensor
                        )
                        del gen_baseline_batch, gen_baseline_output

                source_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(source_prompt_count)],
                    dtype=object,
                )
                generated_batch = source_batch.repeat(
                    repeat_times=rollout_n,
                    interleave=True,
                )
                generated_batch = generated_batch.union(gen_batch_output)
                generated_rows = len(generated_batch)
                expected_rows = source_prompt_count * rollout_n
                if generated_rows != expected_rows:
                    raise ValueError(
                        "rollout output row count disagrees with prompt_count * "
                        f"rollout.n: {generated_rows} != {expected_rows}"
                    )

                probe_metrics = compute_probe_metrics(
                    generated_batch.non_tensor_batch
                )
                metrics.update(probe_metrics)
                classifier_final_metrics = maybe_update_stop_classifier(
                    self.classifier_final_state,
                    generated_batch.non_tensor_batch,
                    self.train_stop_classifier_online,
                )
                classifier_gold_or_final_metrics = maybe_update_stop_classifier(
                    self.classifier_gold_or_final_state,
                    generated_batch.non_tensor_batch,
                    self.train_stop_classifier_online,
                )
                if (
                    self.classifier_final_state.generation_batch_count
                    != self.classifier_gold_or_final_state.generation_batch_count
                ):
                    raise RuntimeError(
                        'dual classifier generation-batch counters diverged'
                    )
                classifier_metrics = dict(classifier_final_metrics)
                overlap = set(classifier_metrics).intersection(
                    classifier_gold_or_final_metrics
                )
                if overlap:
                    raise RuntimeError(
                        f'dual classifier metric-key collision: {sorted(overlap)}'
                    )
                classifier_metrics.update(classifier_gold_or_final_metrics)
                self.clf = self.classifier_gold_or_final_state.model
                self.train_flag = int(
                    self.classifier_gold_or_final_state.is_trained
                )
                metrics.update(classifier_metrics)
                self._record_stop_classifier_batch(classifier_metrics)
                if "response_mask" not in generated_batch.batch:
                    generated_batch.batch["response_mask"] = (
                        compute_response_mask(generated_batch)
                    )

                with _accum_timer("adv", timing_raw):
                    component_metric_keys = set()
                    if self.use_rm:
                        rm_scores = self.rm_wg.compute_rm_score(generated_batch)
                        generated_batch = generated_batch.union(rm_scores)

                    reward_extra_infos_dict: dict[str, list]
                    try:
                        reward_result = self.reward_fn(
                            generated_batch,
                            return_dict=True,
                        )
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_infos_dict = dict(
                            reward_result["reward_extra_info"]
                        )
                    except Exception as exc:
                        print(f"Error in reward_fn: {exc}")
                        reward_tensor = self.reward_fn(generated_batch)
                        reward_extra_infos_dict = {}

                    if dapo_cfg["overlong_enabled"]:
                        (
                            reward_tensor,
                            overlong_penalties,
                            overlong_lengths,
                        ) = apply_dapo_soft_overlong_penalty(
                            generated_batch,
                            reward_tensor,
                            max_response_length=max_response_length,
                            buffer_len=dapo_cfg["overlong_buffer_len"],
                            penalty_factor=dapo_cfg[
                                "overlong_penalty_factor"
                            ],
                        )
                        penalty_values = (
                            overlong_penalties.detach().cpu().numpy()
                        )
                        overlong_length_values = (
                            overlong_lengths.detach().cpu().numpy()
                        )
                        reward_extra_infos_dict["dapo_overlong_penalty"] = (
                            penalty_values.tolist()
                        )
                        reward_extra_infos_dict["dapo_overlong"] = (
                            (penalty_values < 0.0).astype(np.int64).tolist()
                        )
                        reward_extra_infos_dict[
                            "dapo_overlong_length"
                        ] = overlong_length_values.tolist()

                    generated_batch.batch["token_level_scores"] = reward_tensor
                    generated_batch.batch["token_level_rewards"] = reward_tensor
                    print(f"{list(reward_extra_infos_dict.keys())=}")
                    if reward_extra_infos_dict:
                        generated_batch.non_tensor_batch.update(
                            {
                                key: np.asarray(values)
                                for key, values in reward_extra_infos_dict.items()
                            }
                        )
                        component_metrics = compute_reward_component_metrics(
                            reward_extra_infos_dict
                        )
                        component_metric_keys = set(component_metrics)
                        _accumulate_scalar_metrics(
                            scalar_metric_state,
                            component_metrics,
                            generated_rows,
                        )

                    if dapo_cfg["filter_enabled"]:
                        filter_result = select_dapo_nonconstant_uid_groups(
                            generated_batch.non_tensor_batch["uid"],
                            generated_batch.batch["raw_policy_acc"],
                            expected_group_size=rollout_n,
                        )
                        row_keep_mask = filter_result["row_keep_mask"]
                    else:
                        row_keep_mask = np.ones(
                            generated_rows,
                            dtype=np.bool_,
                        )
                        filter_result = {
                            "kept_indices": np.arange(
                                generated_rows,
                                dtype=np.int64,
                            ),
                            "prompt_count": source_prompt_count,
                            "kept_prompt_count": source_prompt_count,
                            "discarded_prompt_count": 0,
                        }
                    generated_batch.non_tensor_batch[
                        "dapo_filter_enabled"
                    ] = np.full(
                        generated_rows,
                        int(dapo_cfg["filter_enabled"]),
                        dtype=np.int64,
                    )
                    generated_batch.non_tensor_batch["dapo_group_kept"] = (
                        row_keep_mask.astype(np.int64)
                    )
                    generated_batch.non_tensor_batch[
                        "dapo_generation_batch"
                    ] = np.full(
                        generated_rows,
                        num_gen_batches,
                        dtype=np.int64,
                    )

                    generated_batch = compute_advantage(
                        generated_batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=rollout_n,
                        verified_stop_aux_loss_coef=(
                            self.config.actor_rollout_ref.actor.get(
                                "verified_stop_aux_loss_coef", 0.0
                            )
                        ),
                        overflow_stop_aux_loss_coef=(
                            self.config.actor_rollout_ref.actor.get(
                                "overflow_stop_aux_loss_coef", 0.0
                            )
                        ),
                        negative_stop_aux_loss_coef=(
                            self.config.actor_rollout_ref.actor.get(
                                "negative_stop_aux_loss_coef", 0.0
                            )
                        ),
                        incorrect_stop_aux_loss_coef=(
                            self.config.actor_rollout_ref.actor.get(
                                "incorrect_stop_aux_loss_coef", 0.0
                            )
                        ),
                        post_accepted_tail_aux_loss_coef=(
                            self.config.actor_rollout_ref.actor.get(
                                "post_accepted_tail_aux_loss_coef", 0.0
                            )
                        ),
                    )
                    if rollout_stop_audit_path:
                        audit_metrics = append_rollout_stop_audit(
                            rollout_stop_audit_path,
                            self.global_steps,
                            generated_batch,
                        )
                        _accumulate_scalar_metrics(
                            scalar_metric_state,
                            {
                                key: value
                                for key, value in audit_metrics.items()
                                if key not in component_metric_keys
                            },
                            generated_rows,
                        )

                generated_prompt_count += filter_result["prompt_count"]
                generated_row_count += generated_rows
                kept_prompt_count += filter_result["kept_prompt_count"]
                discarded_prompt_count += filter_result[
                    "discarded_prompt_count"
                ]
                kept_indices = filter_result["kept_indices"].tolist()
                if kept_indices:
                    kept_batch = generated_batch[kept_indices]
                    pending_batch = (
                        kept_batch
                        if pending_batch is None
                        else DataProto.concat([pending_batch, kept_batch])
                    )
                    pending_prompt_count += filter_result[
                        "kept_prompt_count"
                    ]

                needs_more_generation = False
                if dapo_cfg["filter_enabled"]:
                    needs_more_generation = dapo_dynamic_sampling_needs_more(
                        kept_prompt_count=pending_prompt_count,
                        target_prompt_count=target_prompt_count,
                        num_gen_batches=num_gen_batches,
                        max_num_gen_batches=dapo_cfg[
                            "max_num_gen_batches"
                        ],
                    )
                if needs_more_generation:
                    print(
                        "[dapo_dynamic_sampling] "
                        f"step={self.global_steps} "
                        f"gen_batches={num_gen_batches} "
                        f"kept_prompts={pending_prompt_count}/"
                        f"{target_prompt_count}"
                    )
                    continue

                if pending_batch is None:
                    raise RuntimeError(
                        "no rollout rows are available for the optimizer update"
                    )
                if dapo_cfg["filter_enabled"]:
                    target_rows = target_prompt_count * rollout_n
                    if len(pending_batch) < target_rows:
                        raise RuntimeError(
                            "DAPO pending batch is smaller than the configured "
                            "optimizer batch"
                        )
                    batch = pending_batch[:target_rows]
                else:
                    batch = pending_batch

                is_last_step = self.global_steps >= self.total_training_steps
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1
                ).tolist()

                with _accum_timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)
                if self.use_reference_policy:
                    with _accum_timer("ref", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                            batch
                        )
                        batch = batch.union(ref_log_prob)
                if self.use_critic:
                    with _accum_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                    with _accum_timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(
                        reduce_metrics(critic_output.meta_info["metrics"])
                    )

                if self.config.trainer.critic_warmup <= self.global_steps:
                    center_stop_actions = (
                        self.config.actor_rollout_ref.actor.get(
                            "prompt_group_centered_stop_action_advantage",
                            False,
                        )
                    )
                    if not isinstance(center_stop_actions, (bool, np.bool_)):
                        raise ValueError(
                            "prompt_group_centered_stop_action_advantage must "
                            f"be boolean, got {center_stop_actions!r}"
                        )
                    if center_stop_actions:
                        metrics.update(
                            center_stop_action_weights_by_prompt_in_place(
                                batch,
                                verified_stop_aux_loss_coef=(
                                    self.config.actor_rollout_ref.actor.get(
                                        "verified_stop_aux_loss_coef", 0.0
                                    )
                                ),
                                incorrect_stop_aux_loss_coef=(
                                    self.config.actor_rollout_ref.actor.get(
                                        "incorrect_stop_aux_loss_coef", 0.0
                                    )
                                ),
                                expected_group_size=rollout_n,
                            )
                        )
                    with _accum_timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(
                        reduce_metrics(actor_output.meta_info["metrics"])
                    )

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (
                        is_last_step
                        or self.global_steps
                        % self.config.trainer.test_freq
                        == 0
                    )
                ):
                    with _accum_timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _accum_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                timing_raw["step"] = time.monotonic() - step_started_at
                metrics.update(_finalize_scalar_metrics(scalar_metric_state))
                metrics.update(
                    {
                        "dapo/gen_batches": float(num_gen_batches),
                        "dapo/generated_prompts": float(
                            generated_prompt_count
                        ),
                        "dapo/generated_rollouts": float(generated_row_count),
                        "dapo/kept_prompts_before_trim": float(
                            kept_prompt_count
                        ),
                        "dapo/discarded_constant_prompts": float(
                            discarded_prompt_count
                        ),
                        "dapo/filter_keep_rate": float(
                            kept_prompt_count / generated_prompt_count
                            if generated_prompt_count
                            else 0.0
                        ),
                        "dapo/surplus_kept_prompts": float(
                            max(pending_prompt_count - target_prompt_count, 0)
                            if dapo_cfg["filter_enabled"]
                            else 0
                        ),
                    }
                )
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=n_gpus,
                    )
                )
                logger.log(data=metrics, step=self.global_steps)
                last_completed_step = self.global_steps

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                pending_batch = None
                pending_prompt_count = 0
                num_gen_batches = 0
                generated_prompt_count = 0
                generated_row_count = 0
                kept_prompt_count = 0
                discarded_prompt_count = 0
                scalar_metric_state = {}
                metrics = {}
                timing_raw = {}
                step_started_at = time.monotonic()

        if self.config.trainer.total_training_steps is None:
            # DAPO dynamic sampling can consume more than one generation batch
            # per optimizer update, so an epoch cannot be converted exactly to
            # a fixed optimizer-step count in advance.  At the requested
            # dataloader boundary, persist the last completed update and finish
            # successfully instead of treating normal epoch exhaustion as an
            # error.
            self.global_steps = last_completed_step
            latest_marker = os.path.join(
                self.config.trainer.default_local_dir,
                "latest_checkpointed_iteration.txt",
            )
            latest_saved_step = None
            if os.path.isfile(latest_marker):
                with open(latest_marker, "r", encoding="utf-8") as handle:
                    latest_saved_step = handle.read().strip()
            if latest_saved_step != str(last_completed_step):
                self._save_checkpoint()
            progress_bar.close()
            pprint(
                "Epoch-bounded training completed at dataloader exhaustion: "
                f"global_step={last_completed_step}"
            )
            return

        progress_bar.close()
        raise RuntimeError(
            "training dataloader exhausted before trainer.total_training_steps; "
            "increase trainer.total_epochs when data.gen_batch_size exceeds "
            "data.train_batch_size"
        )
