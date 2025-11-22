#!/usr/bin/env bash


MODEL="/home/azureuser/cloudfiles/code/Users/dongxu.zhang/models/Qwen3-8B" 
DATA="earliest_or_fallback_trl.json"   # 会话式 prompt+completion
OUTDIR="sft_out_qwen_math"
GPUS=8


export NCCL_DEBUG=warn


torchrun --nproc_per_node=${GPUS} --master_port=12345 train_sft.py \
    --model_name_or_path "${MODEL}" \
    --train_file "${DATA}" \
    --output_dir "${OUTDIR}" \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --packing \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --lr_scheduler_type cosine \
    --deepspeed ds_config.json \

