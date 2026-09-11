# MATH500 three-repeat evaluation

`prepare_math500_source.py` verifies the three deterministic baseline repeats
and emits 1,500 repeat-prefixed rows. Replay that source with
`../classifier_training/collect_replay_every50.py`, using one or more modulo
shards and `--expected-qids 1500`.

Evaluate completed replay shards with the released classifiers:

```bash
python evaluate_dual_classifiers.py \
  --source /path/to/math500_repeat3_source.jsonl \
  --shard /path/to/worker0.jsonl \
  --classifier-final ../classifiers/classifier_final_consistency.joblib \
  --classifier-gold ../classifiers/classifier_gold_safe.joblib \
  --output /path/to/decisions.jsonl \
  --summary-root /path/to/summaries \
  --aggregate /path/to/aggregate.json
```

`summarize_math500_dual_strict.py` is the final reporting contract. It regrades
answers using normalized equality, `math_verify`, and an independent SymPy
fallback. The older summarizer is retained for regression comparison.
