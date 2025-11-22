#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1

train_files="./data/train/preprocessed_data/deepscaler.parquet"
val_files="['./data/test/preprocessed_data/gsm8k.parquet','./data/test/preprocessed_data/math.parquet','./data/test/preprocessed_data/aime*16.parquet', './data/test/preprocessed_data/mmlu.parquet']"

batch_size=128
max_response_length=16384
rollout=1

end_of_think_token_id=151649 # </think>
non_end_of_think_token_id=71486 # "Alright"

PROJECT_NAME="adapt_think_verl-eval"

CKPT_DIR_LIST=("./ckpts/adapt_think_ds1.5b_deepscaler_btz128_n16_nr0.5-sl16384-fl4096-nb0.05-lr2e-6")
STEPS=(300)
for CKPT_DIR in "${CKPT_DIR_LIST[@]}"; do
    for STEP in "${STEPS[@]}"; do
        CKPT_PATH=${CKPT_DIR}/global_step_${STEP}
        HF_MODEL_PATH=${CKPT_PATH}/HF
        RESULT_SAVE_DIR=${CKPT_PATH}/eval_results_HF_len${max_response_length}_n${rollout}
        EXP_NAME="eval-HF-len${max_response_length}-n${rollout}-${CKPT_DIR##*/}-step${STEP}"
        echo "EVAL CKPT: ${CKPT_PATH}"
        echo "MAX_LENGTH: ${max_response_length}"
        echo "EXP_NAME: ${EXP_NAME}"

        # Train over a single node, 8 A100-80GB GPUs.
        python3 -m src.main_ppo \
            algorithm.adv_estimator=naive \
            reward_model.reward_manager=adapt_think \
            data.train_files="$train_files" \
            data.val_files="$val_files" \
            data.train_batch_size=$batch_size \
            data.val_batch_size=512 \
            data.max_prompt_length=4096 \
            data.max_response_length=$max_response_length \
            actor_rollout_ref.adapt_think.eot_token_id=$end_of_think_token_id \
            actor_rollout_ref.adapt_think.non_eot_token_id=$non_end_of_think_token_id \
            actor_rollout_ref.model.path=$HF_MODEL_PATH  \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.rollout.val_kwargs.n=$rollout \
            actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
            actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
            actor_rollout_ref.rollout.enable_chunked_prefill=True \
            trainer.logger=['console','wandb'] \
            trainer.project_name=$PROJECT_NAME \
            trainer.experiment_name=$EXP_NAME \
            trainer.val_before_train=True \
            +trainer.val_only=True \
            +trainer.val_result_save_dir=$RESULT_SAVE_DIR \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=1 \
            trainer.default_local_dir=$CKPT_DIR \
            custom_reward_function.path="./src/adapt_think_rm.py" \
            custom_reward_function.name="adapt_think_rm" 
    done
done