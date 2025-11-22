#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="tmp_input_json"
OUTPUT_DIR="tmp_output_json"
WS_NAME="cse51muz9j6fgvfopenai"          # 替换成你真实 Azure OpenAI 资源名
DEPLOYMENT_ID="gpt-4.1-mini-ols"        # 你的部署名
API_VERSION="2024-12-01-preview"

mkdir -p "$OUTPUT_DIR"

# 并行度
JOBS=${JOBS:-8}

find "$INPUT_DIR" -type f -name "*.json" -print0 \
| xargs -0 -P "$JOBS" -I{} bash -c '
  in="{}"
  bn="$(basename "$in")"
  out="'"$OUTPUT_DIR"'/$bn"

  # 跳过已存在且非空的文件
  if [[ -s "$out" ]]; then
      echo "[skip] $out already exists"
      exit 0
  fi

  echo "[proc] $in -> $out"
  python qwen_deal.py \
    --input "$in" \
    --output "$out" \
    --ws-name "'"$WS_NAME"'" \
    --deployment-id "'"$DEPLOYMENT_ID"'" \
    --api-version "'"$API_VERSION"'"

