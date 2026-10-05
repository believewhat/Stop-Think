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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from ..src.stop_ray_trainer import StopRayPPOTrainer

import os
import ray
import hydra


STOP_REWARD_CONFIG_KEYS = (
    "max_stop_count",
    "stop_reward_half_life_tokens",
    "min_verified_stop_separation_tokens",
)


def get_stop_reward_config(config):
    """Read rollout/reward policy from one required Hydra configuration path."""

    rollout_config = config.actor_rollout_ref.rollout
    missing = [key for key in STOP_REWARD_CONFIG_KEYS if rollout_config.get(key) is None]
    if missing:
        raise ValueError(
            "missing required actor_rollout_ref.rollout configuration: "
            + ", ".join(missing)
        )

    max_stop_count = int(rollout_config.max_stop_count)
    stop_reward_half_life_tokens = float(
        rollout_config.stop_reward_half_life_tokens
    )
    min_verified_stop_separation_tokens = int(
        rollout_config.min_verified_stop_separation_tokens
    )
    if max_stop_count < 1:
        raise ValueError(f"max_stop_count must be positive, got {max_stop_count}")
    if not 0.0 < stop_reward_half_life_tokens < float("inf"):
        raise ValueError(
            "stop_reward_half_life_tokens must be finite and positive, got "
            f"{stop_reward_half_life_tokens}"
        )
    if min_verified_stop_separation_tokens < 1:
        raise ValueError(
            "min_verified_stop_separation_tokens must be positive, got "
            f"{min_verified_stop_separation_tokens}"
        )

    legacy_probe_cap = rollout_config.get("max_stop_proposals")
    if legacy_probe_cap is not None:
        raise ValueError(
            "max_stop_proposals was removed; max_stop_count is the single probe budget"
        )

    values = {
        "max_stop_count": max_stop_count,
        "stop_reward_half_life_tokens": stop_reward_half_life_tokens,
        "min_verified_stop_separation_tokens": (
            min_verified_stop_separation_tokens
        ),
    }
    for path, kwargs in (
        (
            "custom_reward_function.reward_kwargs",
            dict((config.get("custom_reward_function") or {}).get("reward_kwargs", {})),
        ),
        (
            "reward_model.reward_kwargs",
            dict(config.reward_model.get("reward_kwargs", {})),
        ),
    ):
        conflicts = sorted(set(kwargs).intersection(values))
        if conflicts:
            raise ValueError(
                f"{path} must not override rollout-owned values: {conflicts}"
            )
    return values


def get_custom_reward_fn(config):
    import importlib.util, sys
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        runtime_env_vars = {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'VLLM_LOGGING_LEVEL': 'WARN'
        }
        run_id = os.environ.get('ADAPTTHINK_RUN_ID')
        if run_id:
            # Propagate the supervisor's ownership marker to every Ray/vLLM
            # worker so the GPU guard never mistakes this run for a foreign
            # CUDA process.
            runtime_env_vars['ADAPTTHINK_RUN_ID'] = run_id
        ray_init_kwargs = {
            'runtime_env': {
                'env_vars': runtime_env_vars
            }
        }
        ray_tmpdir = os.environ.get('ADAPTTHINK_RAY_TMPDIR')
        if ray_tmpdir:
            # Keep this job independent from any concurrently running local
            # Ray cluster and make ownership/auditing deterministic.
            ray_init_kwargs['_temp_dir'] = ray_tmpdir
        # this is for local ray cluster
        # SWITCH94_RAY_RESOURCES: independent three-GPU job, no attachment to other clusters.
        ray_init_kwargs.update(num_gpus=int(config.trainer.n_gpus_per_node),
            num_cpus=24, object_store_memory=4 * 1024**3, include_dashboard=False)
        ray.init(**ray_init_kwargs)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)
        stop_reward_config = get_stop_reward_config(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from ..src.stop_fsdp_workers import StopActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from ..src.stop_ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(StopActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        #use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(StopActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from ..src.stop_naive_manager import StopNaiveRewardManager
            reward_manager_cls = StopNaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == 'batch':
            from verl.workers.reward_manager import BatchRewardManager
            reward_manager_cls = BatchRewardManager
        elif reward_manager_name == 'dapo':
            from verl.workers.reward_manager import DAPORewardManager
            reward_manager_cls = DAPORewardManager
        else:

            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        if reward_manager_name == 'naive':
            reward_kwargs.update(stop_reward_config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                       num_examine=1,
                                       compute_score=compute_score,
                                       reward_fn_key=config.data.reward_fn_key,
                                       **reward_kwargs)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                           num_examine=1,
                                           compute_score=compute_score,
                                           reward_fn_key=config.data.reward_fn_key,
                                           **(
                                               stop_reward_config
                                               if reward_manager_name == 'naive'
                                               else {}
                                           ))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = StopRayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
