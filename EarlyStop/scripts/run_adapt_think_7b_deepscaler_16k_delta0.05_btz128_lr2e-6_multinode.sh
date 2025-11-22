# export HYDRA_FULL_ERROR=1

if [ -n "$MLP_WORKER_NUM" ]; then
    echo "MLP_WORKER_NUM: $MLP_WORKER_NUM"
else
    MLP_WORKER_NUM=1
fi

train_dataset=deepscaler
train_files="./data/train/preprocessed_data/${train_dataset}.parquet"
val_files="['./data/test/preprocessed_data/gsm8k.parquet','./data/test/preprocessed_data/math.parquet','./data/test/preprocessed_data/aime*16.parquet']"

batch_size=128
n_rollout=16
max_response_length=16384
LR=2e-6

nothinking_ratio=0.5
nothinking_max_response_length=4096
end_of_think_token_id=151649 # </think>
non_end_of_think_token_id=71486 # "Alright"
nothinking_bonus=0.05
adjust_old_logprobs=True
ref_result_file="./data/train/ref_results/DeepSeek-R1-Distill-Qwen-7B_deepscaler_K16_len16384.json"

PROJECT_NAME="adapt_think_verl"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # path to your download HF model
EXP_NAME="adapt_think_ds7b_${train_dataset}_btz${batch_size}_n${n_rollout}_nr${nothinking_ratio}-sl${max_response_length}-fl${adapt_think_max_response_length}-nb${nothinking_bonus}-lr${LR}-${MLP_WORKER_NUM}node"
# EXP_NAME="adapt_think_debug"
CKPT_DIR="./ckpts/${PROJECT_NAME}/${EXP_NAME}"

# Train over a single node, 8 A100-80GB GPUs.
RUN_SCRIPT="
python3 -m src.main_ppo \
    algorithm.adv_estimator=naive \
    reward_model.reward_manager=adapt_think \
    reward_model.reward_kwargs.nothinking_bonus=$nothinking_bonus \
    reward_model.reward_kwargs.ref_result_file=$ref_result_file \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.adapt_think.nothinking_ratio=$nothinking_ratio \
    actor_rollout_ref.adapt_think.nothinking_max_response_length=$nothinking_max_response_length \
    actor_rollout_ref.adapt_think.eot_token_id=$end_of_think_token_id \
    actor_rollout_ref.adapt_think.non_eot_token_id=$non_end_of_think_token_id \
    actor_rollout_ref.adapt_think.adjust_old_logprobs=$adjust_old_logprobs \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$MLP_WORKER_NUM \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    custom_reward_function.path="./src/adapt_think_rm.py" \
    custom_reward_function.name="adapt_think_rm" \
"