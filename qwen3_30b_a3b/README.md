# Qwen3-30B-A3B ESTAR-Lite

This directory is the portable release of the latest Qwen3-30B-A3B (MoE)
math pipeline run on the Gamma server on 2026-09-07/08. It replaces the older
single-probe classifier with a probe every 50 THINK tokens and a 22-feature
answer-cluster state. Two classifiers are released: an aggressive
`final_consistency` policy and a conservative `gold_safe` policy.

## Experiment contract

- Model: `Qwen/Qwen3-30B-A3B` (`qwen3_moe`).
- System prompt: `Please reason step by step, and put your final answer within \boxed{}`.
- Chat template: `enable_thinking=True`, followed by the explicit assistant
  prefix `<think>\n`.
- MATH500 baseline decoding: temperature 0, top-p 1, top-k 20, repetition
  penalty 1.2.
- AIME2024 baseline decoding: temperature 0.6, top-p 0.95, top-k 20,
  repetition penalty 1.2.
- Both baselines use model length 40,960 and response cap 32,768.
- Probe suffix: `\n</think>\n\n\boxed{`.
- Probe schedule: positions 50, 100, ... in the re-tokenized THINK span.
- Classifier threshold: 0.95.
- Primary length metric: THINK tokens only. Probe computation and forced suffix
  tokens are excluded.
- No-hit behavior: return the original full answer and full THINK length.

The 22 features are answer log-probability statistics plus online answer-cluster
agreement, margin, entropy, run length, flips, and first/second-order margin
changes. Their exact order is frozen in
[`classifiers/feature_columns.json`](classifiers/feature_columns.json).

## Released results

Each number below is the mean of three independent complete runs. MATH500 has
1,500 evaluated trajectories; AIME2024 has 90.

| Dataset | Policy | Accuracy | Accuracy delta | Mean THINK tokens | Token reduction | Stop coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MATH500 | Full baseline | 96.53% | — | 3,679 | — | — |
| MATH500 | Final consistency | 96.27% | -0.27 pp | 2,780 | 24.44% | 67.87% |
| MATH500 | Gold safe | 96.53% | 0.00 pp | 3,628 | 1.40% | 9.40% |
| AIME2024 | Full baseline | 82.22% | — | 11,064 | — | — |
| AIME2024 | Final consistency | 81.11% | -1.11 pp | 8,320 | 24.81% | 88.89% |
| AIME2024 | Gold safe | 82.22% | 0.00 pp | 10,824 | 2.26% | 6.67% |

AIME2024 contains only 30 questions per run, so its run-level confidence
intervals are wide. Full validation metrics, hashes, and label counts are in
[`classifiers/metrics.json`](classifiers/metrics.json).

## Directory layout

- `baseline/`: deterministic long-context MATH500+AIME2024 inference.
- `data_prep_random10k/`: historical DeepScaleR 10K sample, full-CoT/probe
  generation, shard merge, and validation.
- `classifier_training/`: every-50 replay, feature extraction, exact math
  equivalence labels, dual LightGBM training, and CPU tests.
- `eval_math500/`, `eval_aime2024/`: strict three-repeat source preparation,
  classifier policy evaluation, and tests.
- `classifiers/`: trained models, model card, metrics, feature order, and
  checksums.

## Environment

The recorded Gamma environment used Python 3.10 with:

```text
vllm==0.9.0
transformers==4.53.3
torch==2.7.0
numpy==2.2.6
pandas==2.2.3
scipy==1.15.3
math-verify==0.7.0
pylatexenc==2.10
sympy==1.14.0
joblib==1.5.0
lightgbm==4.6.0
scikit-learn==1.6.1
```

Install the pinned non-CUDA dependencies with:

```bash
python -m pip install -r qwen3_30b_a3b/requirements.txt
```

Install a CUDA-compatible PyTorch and vLLM build separately. The repository's
vendored vLLM contains the probe feature support used by model-specific data
generation.

## Minimal workflow

1. Generate three long baselines with seeds `20260903`, `20260904`, and
   `20260905`; see [`baseline/README.md`](baseline/README.md).
2. Build the historical DeepScaleR 10K and generate the model-specific full
   trajectories; see
   [`data_prep_random10k/README.md`](data_prep_random10k/README.md).
3. Replay each full trajectory every 50 THINK tokens and train both
   classifiers; see
   [`classifier_training/README.md`](classifier_training/README.md).
4. Prepare a three-repeat evaluation source and run the corresponding
   `evaluate_dual_classifiers.py` script. Dataset-specific commands are in the
   MATH500 and AIME2024 READMEs.

All scripts use resumable JSONL artifacts and validate fingerprints, row counts,
probe schedules, feature schemas, and model identity before accepting results.
