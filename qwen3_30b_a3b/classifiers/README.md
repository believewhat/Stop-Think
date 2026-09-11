# Released classifiers

Both artifacts are LightGBM binary classifiers packaged with their feature
order, schema hashes, training provenance, threshold, and target label.

| File | Target | Threshold | Intended behavior |
| --- | --- | ---: | --- |
| `classifier_final_consistency.joblib` | Probe answer equals the model's eventual full answer | 0.95 | Higher coverage and approximately 24% fewer THINK tokens |
| `classifier_gold_safe.joblib` | Probe answer is equivalent to the gold answer | 0.95 | Low-coverage, accuracy-preserving early stop |

Training used 10,000 DeepScaleR questions and 1,258,639 every-50-token probes.
The serialized models require `joblib==1.5.0`, `lightgbm==4.6.0`, and
`scikit-learn==1.6.1` for the closest compatibility with the training runtime.

The public copies differ from the Gamma artifacts only in provenance path
strings: absolute server paths were replaced with portable model and worker
names. LightGBM model state and all predictive parameters are unchanged.
Checksums for the public files are in `SHA256SUMS`.

Original Gamma artifact checksums, before metadata-path cleanup:

```text
371db5fe91f8ebf36a8cd1237dc35efa4516a63e98903f51df1a56ebe90178f1  classifier_final_consistency.joblib
657e7e1725d3efaa750ca2e73ba7f2ccd191fde00d901500c2e44b61459c1eca  classifier_gold_safe.joblib
```

Only load joblib files obtained from a trusted source. Joblib/pickle is not a
safe format for untrusted input.
