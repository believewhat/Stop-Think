# Every-50 cluster classifier training

Run `collect_replay_every50.py` on each shard of the audited 10K full-trajectory
JSONL. Run it from this directory so the local feature module is importable.

```bash
cd qwen3_30b_a3b/classifier_training
CUDA_VISIBLE_DEVICES=0 python collect_replay_every50.py \
  --input /path/to/model_specific_10k.jsonl \
  --model /path/to/Qwen3-30B-A3B \
  --output /path/to/every50/worker0.jsonl \
  --errors /path/to/every50/worker0.errors.jsonl \
  --progress /path/to/every50/worker0.progress.json \
  --shard-index 0 --shard-count 1 --expected-qids 10000 \
  --token-step 50 --probe-batch-size 64 --max-model-len 40960 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 65536
```

Train both classifiers from one or more completed replay shards:

```bash
python train_every50_classifier.py \
  --inputs /path/to/every50/worker0.jsonl \
  --output-dir /path/to/classifier \
  --seed 20260722 --validation-fraction 0.1 --threshold 0.95 \
  --n-jobs 16 --equivalence-workers 64 --expected-qids 10000 \
  --backbone-model /path/to/Qwen3-30B-A3B
```

The trainer writes both `.joblib` files, `feature_columns.json`, `metrics.json`,
validation decisions, and probe predictions. Labels use cached exact/symbolic
math equivalence for probe-to-full consistency and probe-to-gold safety.
