# Long-context baseline

`run_baseline_shard.py` runs the deterministic MATH500 baseline. It accepts a
500-row MATH500 JSONL and a 30-row AIME2024 JSONL and writes both datasets, but
the released MATH500 evaluation selects only its MATH rows. Each row must
contain `unique_id`, `problem`, and `answer`. The script is pinned to a
40,960-token model context and a 32,768-token response cap.

Run one shard:

```bash
CUDA_VISIBLE_DEVICES=0 python run_baseline_shard.py \
  --math /path/to/math500.jsonl \
  --aime /path/to/aime2024.jsonl \
  --model /path/to/Qwen3-30B-A3B \
  --output /path/to/repeat1/shard0.jsonl \
  --progress /path/to/repeat1/shard0.progress.json \
  --shard-index 0 --shard-count 1 --seed 20260903
```

Repeat with seeds `20260904` and `20260905`. Multiple GPU workers can use the
same seed and distinct `--shard-index` values. Merge and summarize a repeat:

```bash
python summarize_baseline.py \
  --shard /path/to/shard0.jsonl \
  --shard /path/to/shard1.jsonl \
  --output /path/to/summary.json \
  --merged /path/to/trajectories.jsonl
```

`model_content_id.py` and `write_execution_manifest.py` record model and runtime
provenance without hashing multi-gigabyte weight contents.

The released AIME2024 result uses stochastic decoding instead. Run
`run_aime06_baseline_shard.py` with temperature 0.6/top-p 0.95 (frozen in the
script), then merge with `summarize_aime06_baseline.py`:

```bash
CUDA_VISIBLE_DEVICES=0 python run_aime06_baseline_shard.py \
  --aime /path/to/aime2024.jsonl \
  --model /path/to/Qwen3-30B-A3B \
  --output /path/to/repeat1/shard0.jsonl \
  --progress /path/to/repeat1/shard0.progress.json \
  --shard-index 0 --shard-count 1 --seed 20260903
```
