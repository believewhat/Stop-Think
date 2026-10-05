# Released Qwen3-30B-A3B MedQA classifier

[`early_stop_cls_medqa_qwen30a3b.joblib`](early_stop_cls_medqa_qwen30a3b.joblib)
is the **offline ESTAR-LITE medical classifier** trained from scratch on
Qwen3-30B-A3B's own MedQA-train trajectories. The same weight was used for JAMA;
it is not an 8B classifier and was not trained on JAMA test questions.

| Item | Value |
| --- | --- |
| Backbone | `Qwen/Qwen3-30B-A3B` |
| Training questions / probe rows | 10,178 / 244,136 |
| Classifier | XGBoost binary logistic; 500 trees, maximum depth 6 |
| Input | 22 ordered A/B/C/D evidence and history features |
| Target | Probe letter matches the same trajectory's eventual full-answer letter |
| Deployment threshold | 0.95 |
| Reuse | MedQA and JAMA with the same backbone and feature contract |

**The score is not a calibrated probability of gold-answer correctness.**
This is a final-answer-consistency classifier, not a gold-safe classifier.
The training recipe was shared with the earlier 8B experiment, but these
weights were fitted anew using 30B-generated data.

## Load and score existing probe features

Install the recorded CPU dependencies in your own environment:

```bash
python -m pip install -r qwen3_30b_a3b/classifiers_medqa/requirements.txt
```

Load only trusted joblib/pickle files: deserialization can execute code.
The release retains the original serialized weight unchanged. The separate
[manifest](classifier_manifest.json) uses portable paths instead of server paths.

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

directory = Path("qwen3_30b_a3b/classifiers_medqa")
pack = joblib.load(directory / "early_stop_cls_medqa_qwen30a3b.joblib")
features = json.loads((directory / "feature_columns.json").read_text())
assert pack["feats"] == features
assert pack["model"].n_features_in_ == len(features) == 22

# Supply probe-feature records produced with the matching legacy collector.
# Each JSON object must contain all 22 named features, in addition to any IDs.
steps = pd.DataFrame(json.loads(Path("/path/to/probe_features.json").read_text()))
missing = set(features) - set(steps.columns)
if missing:
    raise ValueError(f"Missing features: {sorted(missing)}")
X = steps[features].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
scores = pack["model"].predict_proba(X)[:, 1]
above_threshold = scores >= pack["deployment_threshold"]
```

Threshold crossing alone is insufficient: the recorded policy also requires a
valid decoded probe letter A/B/C/D and `probe_has_letter=True`. For each question,
select the first eligible crossing in `(step_tokens, round)` order. With no hit,
keep the full trajectory's answer and length. Reset feature history for each
question; do not substitute cumulative-top letters for decoded probe answers.

## Feature and evaluation contract

The exact order is in [feature_columns.json](feature_columns.json), and is also
stored as `pack["feats"]`. `changed_prev` is deliberately excluded.

- `inst_sA` through `inst_sD`: current probe's bucket-aggregated letter log scores.
- `cum_A` through `cum_D`: accumulated log normalized letter evidence, with
  the collector's numerical smoothing.
- `cum_margin`: largest minus second-largest cumulative evidence.
- `run_len` and `flips`: consecutive cumulative-top run length and historical
  cumulative-top changes.
- `delta_recent`: recent cumulative-margin change (window 3);
  `slope_recent`: first-to-current margin change per probe interval.
- `curv_*2`: quadratic second derivatives over the last at most 5 probes
  (zero with fewer than 3 probes).
- `cum_top_A` through `cum_top_D`: one-hot current cumulative-top letter.

These names do not license replacing the original log-probability bucket/slot
extraction or smoothing with a different definition. The release is a weight
and scoring example, not a new end-to-end benchmark runner.

The original medical protocol used probes every 50 main THINK tokens,
including the initial round-zero probe, and at most 100 THINK rounds. THINK
decoding used temperature 0.1, top-p 0.95, and repetition penalty 1.2; probes
used temperature 0, top-p 1, at most 10 generated tokens, and top-20 logprobs.
It used the legacy handwritten MedQA prompt with `<think>` and `<final_answer>`
tags, complete stitched trajectories, and offline first-hit evaluation.
Its length accounting excludes probing and forced suffixes.

This legacy protocol is **not** the latest 32k manuscript evaluation or the
current SFT/DAPO prompt contract. Publishing this weight does not establish
reproduction of the current main-result table.

## Training provenance and compatibility

A 10% question-group holdout (`random_state=42`) was used for classifier
diagnostics. At threshold 0.95, the holdout classifier had precision 0.9864164,
recall 0.5748693, and ROC AUC 0.8875119 for **probe/full-answer consistency**.
These are not MedQA question-answering accuracy or stop coverage. The released
deployment classifier was subsequently refitted on all 244,136 training rows.
Counts, positive-label rate, and tree parameters are in the manifest.

There are three distinct contracts in this repository:

- This medical classifier: 22 A/B/C/D features, XGBoost, offline ESTAR-LITE.
- [Math classifiers](../classifiers/README.md): different 22-feature schema,
  LightGBM, final-consistency and gold-safe targets.
- [MedQA DAPO](../../training/medqa/README.md): online dual classifiers with
  A70 features, fitted during training; their weights are not this artifact.

They are not interchangeable even when the feature counts happen to match.
Training trajectories, benchmark questions, and policy checkpoints are not
included in this classifier release.
