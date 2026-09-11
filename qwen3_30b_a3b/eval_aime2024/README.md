# AIME2024 three-repeat evaluation

`prepare_aime2024_source.py` validates the three complete 30-question baseline
repeats and emits 90 repeat-prefixed rows. Replay that source with
`../classifier_training/collect_replay_every50.py`, using one or more modulo
shards and `--expected-qids 90`.

```bash
python evaluate_dual_classifiers.py \
  --source /path/to/aime2024_repeat3_source.jsonl \
  --shard /path/to/worker0.jsonl \
  --classifier-final ../classifiers/classifier_final_consistency.joblib \
  --classifier-gold ../classifiers/classifier_gold_safe.joblib \
  --output /path/to/decisions.jsonl \
  --summary-root /path/to/summaries \
  --aggregate /path/to/aggregate.json
```

The aggregate reports Student-t confidence intervals over three runs (df=2),
not pseudo-replication over 90 rows. Probe compute is excluded from the primary
THINK-token metric.
