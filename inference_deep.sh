#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

########################################
# 可配置区域
########################################

# 使用多少块 GPU（比如 2、4、8）
NGPU=${NGPU:-2}

# 数据 & 模型路径
INPUT="AdaptThink/data/train/deepscaler.json"
MODEL="/data/data_user_alpha/public_models/Qwen3/Qwen3-8B"

# vLLM / 采样参数
DTYPE="bfloat16"
THRESH=0.95
TOKEN_STEP=50
PROBE_MAX=30
TOPK=20

# 输出前缀（每个 shard 会在后面加 _shardX）
OUT_PREFIX="online_earlystop_results_deepscaler"
PROBE_PREFIX="online_earlystop_probe_records_deepscaler"

# 如果你有 classifier，就在这里写路径；没有就留空
CLS_MODEL_PATH=""
ENABLE_EARLY_STOP_FLAG=""

if [[ -n "${CLS_MODEL_PATH}" ]]; then
  ENABLE_EARLY_STOP_FLAG="--enable_early_stop --cls_model_path ${CLS_MODEL_PATH}"
fi

########################################
# 启动各个 GPU 的 shard 进程
########################################

echo "[INFO] Launching ${NGPU} shards for ${INPUT} on ${NGPU} GPUs"

for gid in $(seq 0 $((NGPU-1))); do
  echo "[LAUNCH] GPU ${gid} -> shard ${gid}/${NGPU}"

  CUDA_VISIBLE_DEVICES=${gid} \
  python inference_short_math_deep.py \
    --input_path "${INPUT}" \
    --model_path "${MODEL}" \
    --dtype "${DTYPE}" \
    --tp 1 \
    --threshold "${THRESH}" \
    --token_step "${TOKEN_STEP}" \
    --probe_max "${PROBE_MAX}" \
    --topk "${TOPK}" \
    --out_csv "${OUT_PREFIX}_shard${gid}.csv" \
    --probe_jsonl "${PROBE_PREFIX}_shard${gid}.jsonl" \
    --shard_idx ${gid} \
    --shard_count ${NGPU} \
    ${ENABLE_EARLY_STOP_FLAG} &
done

# 等所有子进程结束
wait
echo "[INFO] All shards finished, merging CSVs..."

########################################
# 合并所有 shard 的 CSV
########################################

python - << 'PY'
import pandas as pd, glob

prefix = "online_earlystop_results_deepscaler_shard"
files = sorted(glob.glob(prefix + "*.csv"))
if not files:
    raise SystemExit(f"No shard CSVs found matching {prefix}*.csv")

dfs = [pd.read_csv(p) for p in files]
out = pd.concat(dfs, ignore_index=True)
out_path = "online_earlystop_results_deepscaler_merged.csv"
out.to_csv(out_path, index=False)
print(f"[MERGE] merged {len(files)} shards -> {out.shape[0]} rows -> {out_path}")
PY

echo "[DONE]"

NGPU=2 ./inference_deep.sh
