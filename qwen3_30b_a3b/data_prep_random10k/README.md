# DeepScaleR random-10K preparation

`build_random10k_source.py` reproduces the historical selection exactly with
`random.Random(20260827).sample(range(38918), 10000)` under CPython 3.10. The
source SHA256 is mandatory and is used only for integrity, never selection.

```bash
python build_random10k_source.py \
  --source /path/to/deepscaler_clean_full_38918.jsonl \
  --source-sha256 SOURCE_SHA256 \
  --output /path/to/deepscaler_random_10000.jsonl \
  --manifest /path/to/deepscaler_random_10000.manifest.json
```

Generate Qwen3-30B-A3B full trajectories and the original model-specific probes
with `generate_model_specific_probe_train.py`. It imports the probe support from
the repository's vendored vLLM and the math extraction helpers from
`inference_short_math_deep.py`.

```bash
PYTHONPATH="$PWD:$PWD/vllm" CUDA_VISIBLE_DEVICES=0 \
python qwen3_30b_a3b/data_prep_random10k/generate_model_specific_probe_train.py \
  --input /path/to/deepscaler_random_10000.jsonl \
  --model /path/to/Qwen3-30B-A3B \
  --generation-model-id /path/to/Qwen3-30B-A3B \
  --prompt-protocol qwen3_system \
  --output /path/to/worker0.jsonl \
  --errors /path/to/worker0.errors.jsonl \
  --progress /path/to/worker0.progress.json \
  --shard-index 0 --shard-count 1 --resume \
  --dtype bfloat16 --temperature 0.6 --top-p 0.95 --topk 20 \
  --repetition-penalty 1.2 --max-main-tokens 20000 \
  --probe-max-tokens 64 --max-model-len 24576
```

Use `merge_model_specific_probe_shards.py` and
`validate_probe_dataset_against_source.py` before every-50 replay. The replay
trainer uses the full `main.text`; it does not use the original ten-position
classifier features.
