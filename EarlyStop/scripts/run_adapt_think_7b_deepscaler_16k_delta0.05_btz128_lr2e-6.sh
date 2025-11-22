#!/usr/bin/env bash
set -euo pipefail
set -x


# ===== NCCL / 通信建议 =====
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=600
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN


# ===== 路径配置（按需修改）=====
#MODEL_PATH=/home/azureuser/cloudfiles/code/Users/dongxu.zhang/models/Qwen3-8B
MODEL_PATH=/home/azureuser/cloudfiles/code/Users/junda.wang/qwen3_output/AdaptThink/sft_out_qwen
RESULT_DIR=./checkpoints/medqa_qwen3_8B_GRPO2

train_dataset=medicalqa
TRAIN_PARQUET="/home/azureuser/cloudfiles/code/Users/junda.wang/project/AdaptThink/data/train/preprocessed_data/${train_dataset}.parquet"

max_prompt_length=$((1200 * 1))
max_response_length=$((1024 * 5))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

# ===== 训练 =====
python -m verl.trainer.main_ppo_stop \
  algorithm.adv_estimator=grpo \
  \
  data.train_files=${TRAIN_PARQUET} \
  data.val_files=${TRAIN_PARQUET} \
  data.train_batch_size=256 \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  +actor_rollout_ref.rollout.repetition_penalty=1.2 \
  actor_rollout_ref.rollout.free_cache_engine=False \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward_model.enable=False \
  reward_model.reward_manager=naive \
  custom_reward_function.path="./verl/src/reward_loss.py" \
  custom_reward_function.name="compute_score" \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name='GRPO_revise' \
  trainer.experiment_name='search_with_custom_reward' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.default_local_dir=${RESULT_DIR} \
  trainer.default_hdfs_dir=null \
  trainer.save_freq=10 \
  trainer.test_freq=1000 \
  trainer.total_epochs=1


# ===== 合并权重为 HF 权重（按需）=====
#python verl/scripts/model_merger.py merge \
##  --backend fsdp \
#  --local_dir ${RESULT_DIR}/global_step_8/actor \
#  --target_dir ${RESULT_DIR}_hf/medqa_qwen3_8B_DAPO_V1_hf_step8






