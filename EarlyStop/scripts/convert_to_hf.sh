HF_MODEL_PATH=/home/azureuser/cloudfiles/code/Users/dongxu.zhang/models/Qwen3-8B # path to your download HF model
CKPT_PATH=/mnt/azmnt/users/zhichao_yang/rlfac/checkpoints/medqa_qwen3_8B_GRPO_MATH/global_step_10/actor
SAVE_PATH=/home/azureuser/cloudfiles/code/Users/junda.wang/project/our/RL/AdaptThink/ckpts/adapt_think_verl/HF_10_MATH
python src/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CKPT_PATH --target_dir $SAVE_PATH

TOKENIZER_FILES=("tokenizer_config.json" "tokenizer.json")
for FILE in "${TOKENIZER_FILES[@]}"; do
    cp $HF_MODEL_PATH/$FILE $SAVE_PATH
done




HF_MODEL_PATH=/home/azureuser/cloudfiles/code/Users/dongxu.zhang/models/Qwen3-8B # path to your download HF model
CKPT_PATH=/home/azureuser/cloudfiles/code/Users/junda.wang/qwen3_output/AdaptThink/checkpoints/medqa_qwen3_8B_GRPO_MATH/global_step_80/actor
SAVE_PATH=/mnt/azureblobshare/users/zhichao_yang/rlfac/checkpoints/AdaptThink/medqa_qwen3_8B_GRPO_MATH_hf
python src/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CKPT_PATH --target_dir $SAVE_PATH

TOKENIZER_FILES=("tokenizer_config.json" "tokenizer.json")
for FILE in "${TOKENIZER_FILES[@]}"; do
    cp $HF_MODEL_PATH/$FILE $SAVE_PATH
done