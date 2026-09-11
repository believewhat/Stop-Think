# Stop-Think

Code and classifier artifacts for adaptive reasoning termination.

## Qwen3-30B-A3B (MoE)

The latest Qwen3-30B-A3B math release is in
[`qwen3_30b_a3b/`](qwen3_30b_a3b/README.md). It contains:

- long-context MATH500/AIME2024 baseline inference;
- deterministic DeepScaleR 10K sampling and model-specific probe generation;
- every-50-THINK-token replay with the 22-feature answer-cluster state;
- dual LightGBM classifier training and strict evaluation code;
- the trained `final_consistency` and `gold_safe` classifiers used in the
  three-repeat MATH500 and AIME2024 experiments.

The older implementation remains at the repository root and under
`EarlyStop/` and `vllm/`.
