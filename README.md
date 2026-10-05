# Stop-Think / ESTAR

Code for adaptive reasoning termination: **ESTAR-LITE** uses a model-specific
classifier at inference time; **ESTAR** also trains the reasoning policy.

## Start here

- **[Qwen3-30B-A3B MedQA SFT and DAPO examples](training/medqa/README.md)**:
  the current SWITCH-inspired LoRA SFT recipe, shared prompt contract, strict
  reward, verified-stop action loss, and online dual-classifier fitting.
  Fill in your own model, dataset, GPU, and output paths.
- **[Qwen3-30B-A3B math ESTAR-LITE](qwen3_30b_a3b/README.md)**:
  inference, data preparation, classifier training, and MATH-500/AIME2024 evaluation.
- **[Released Qwen3-30B-A3B math classifiers](qwen3_30b_a3b/classifiers/)**:
  final-consistency and gold-safe classifiers, feature order, and model cards.
  Only Qwen3-30B-A3B classifier weights are distributed in the current tree.
- **[Three-run statistics](docs/results.md)**: accuracy and output-token means,
  SDs, and 95% confidence intervals for Qwen3-30B-A3B.

The new MedQA runtime is an **opt-in overlay**. It does not replace the existing
math pipeline or the legacy code under `EarlyStop/` and `vllm/`.
Legacy classifier binaries formerly under `model_classifier/` are no longer
included. Legacy inference examples require your own compatible classifier;
do not substitute the 30B math weights into a different model or feature schema.
MedQA DAPO fits its own online classifiers; medical classifier weights are not
part of this release.

## Paper-reported main results

Transcribed from **Table 2 of the updated manuscript**. Each cell is
**accuracy (%) / mean output tokens**; lower output length is better.
These main-table entries are not the three-run means in the appendix.
For run variability, use the [separate statistical table](docs/results.md).
The PDF itself, training data, and policy checkpoints are not redistributed here.

### Qwen3-8B

| Method | MedQA / USMLE | JAMA | MATH-500 | AIME2024 |
| --- | ---: | ---: | ---: | ---: |
| GRPO | 78.1 / 2,814 | 56.9 / 2,910 | 95.2 / 4,618 | 76.7 / 16,996 |
| O1-Pruner | 76.6 / 1,472 | 53.2 / 1,981 | 91.6 / 2,856 | 66.7 / 7,841 |
| FlashThink | 76.8 / 1,693 | 55.8 / 2,161 | 94.8 / 4,050 | 76.7 / 13,507 |
| No-Thinking | 66.2 / 315 | 48.2 / 368 | 85.0 / 1,139 | 26.7 / 2,493 |
| AdaptThink | 76.4 / 987 | 55.8 / 1,102 | 94.6 / 2,628 | 73.3 / 9,877 |
| Length-Penalty | 76.6 / 1,325 | 55.4 / 1,872 | 92.4 / 3,190 | 66.7 / 7,324 |
| DEER | 75.7 / 1,063 | 56.5 / 1,001 | 94.0 / 3,227 | 73.3 / 12,465 |
| ESTAR-LITE | 77.1 / 772 | 56.5 / 754 | 94.8 / 2,690 | 76.7 / 9,199 |
| ESTAR | 77.3 / 558 | 56.5 / 601 | 95.2 / 2,217 | 76.7 / 7,788 |

### Qwen3-30B-A3B MoE (32k)

| Method | MedQA / USMLE | JAMA | MATH-500 | AIME2024 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 84.2 / 2,516 | 60.4 / 2,581 | 96.4 / 4,679 | 83.3 / 15,063 |
| DEER | 83.6 / 1,151 | 59.1 / 1,245 | 95.2 / 3,212 | 76.7 / 10,772 |
| ESTAR-LITE | 84.2 / 437 | 59.9 / 553 | 96.2 / 2,779 | 80.0 / 8,320 |
| ESTAR | 84.3 / 420 | 60.1 / 529 | 96.4 / 2,660 | 83.3 / 7,396 |

All AIME entries above refer to **AIME2024 (30 questions)**. The updated
8B GRPO/DEER/ESTAR-LITE and 30B group use the extended-thinking prompt and a
32,768-output-token cap described in the manuscript. Token ratios are **not
wall-clock speedups**: probing and classifier overhead must also be measured.

The older [math release](qwen3_30b_a3b/README.md) reports **THINK-only tokens**
(excluding probe work and forced suffixes). Those numbers use a different
length accounting convention and must not be substituted for the output-token
columns above.

## Release status and reproducibility

The tables above report the supplied manuscript, not a fresh benchmark run
performed for this code release. The current MedQA training example is a
subsequent development recipe: LoRA plus trainable vocabulary modules for SFT,
then partial-parameter DAPO with online A70 classifiers. It is not an assertion
that this exact recipe reproduces every manuscript table.

The current SFT run reached step 94; the four-GPU DAPO development run completed
seven optimizer updates before an out-of-memory failure. A completed two-epoch
30B DAPO reproduction is **not** included. Hardware-specific CUDA compatibility
and a bounded smoke run are required before a long training job.

For the exact configuration, checkpoint limitations, data split, and reward
semantics, see [the training notes](training/medqa/README.md).

## Attribution

This repository includes modified code from vLLM, veRL, and AdaptThink.
Detaching the GitHub fork relationship does not remove upstream attribution or
license obligations. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
the retained source headers.
