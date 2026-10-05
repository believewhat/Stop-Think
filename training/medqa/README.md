# Qwen3-30B-A3B MedQA: SFT and DAPO examples

These are the current MedQA development recipes, with private paths and job
supervision removed. **Fill in all paths and allocated GPU IDs yourself.**
Nothing automatically starts evaluation, another training stage, or GPU keepers.

## 1. Environment

Use a dedicated **Linux / Python 3.10** CUDA environment. Install PyTorch 2.7.0,
vLLM 0.9.0, and FlashAttention 2.8.0.post2 built for matching CUDA/PyTorch ABIs,
then install [requirements.txt](requirements.txt). Do not blindly upgrade:
the FSDP traversal optimization is explicitly version-guarded for torch 2.7.0.

The pins describe the existing environment, not a validated B200 environment.
Check CUDA architecture support for your hardware and run a bounded smoke test.
A 30B MoE's active-parameter count does not remove its full-weight memory cost.

## 2. SFT example (three GPUs)

Run from the repository root. Replace the example paths and GPU IDs:

```bash
export BASE_MODEL=/path/to/Qwen3-30B-A3B
export SFT_DATA=/path/to/prepared_medqa_sft
export SFT_OUTPUT=/path/to/new_sft_output
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_DISABLE=1

torchrun --standalone --nproc_per_node=3 training/medqa/train_sft.py \
  --model "$BASE_MODEL" --data "$SFT_DATA" --output "$SFT_OUTPUT" \
  --learning-rate 2e-5
```

This exact recipe expects **3,085 train + 254 eval** examples from the
30B model's own correct MedQA-training trajectories, with **14,594 / 1,210**
gold-safe stop labels. The eval split is held out from those training
trajectories, not the official MedQA test set. No gold-hint augmentation is
used in this current recipe.

The input directory contains `train.jsonl`, `eval.jsonl`, and `manifest.json`.
Each JSONL row has this structure (illustrative only, not a medical benchmark
example or a certified safe label):

```json
{
  "qid": "train-example",
  "prompt": [
    {"role": "system", "content": "<exact SYSTEM_PROMPT from prompt_contract.py>"},
    {"role": "user", "content": "<question>\n\nA. <option>\nB. <option>\nC. <option>\nD. <option>"}
  ],
  "completion": [
    {"role": "assistant", "content": "<think>\nA completed reasoning sentence.<stop> Further reasoning.\n</think>\n<final_answer>A</final_answer>"}
  ],
  "stop_label_count": 1
}
```

Required manifest fields:

```json
{
  "training_ready": true,
  "split_rows": {"train": 3085, "eval": 254},
  "all_stops_inside_think": true,
  "all_stops_at_sentence_ends": true
}
```

Do not turn these flags on for unverified data. In the experiment, each stop
location was moved to a complete sentence boundary **inside the think block**
and re-probed with the original 30B base model. Only gold-correct locations
were kept (1-5 per example). The original CoT, answer, and split membership
were retained. The preparatory trajectories and probe evidence are not included
in this repository; these examples assume that prepared dataset.

Training details:

- One epoch, 94 optimizer steps; batch 1 per GPU, accumulation 11, global batch 33.
- LR 2e-5, 3% warmup, cosine schedule; ordinary completion-only CE, stop weight 1.
- LoRA r32/alpha64/dropout0.05 on attention and expert projections; MoE routers frozen.
- Full input embedding and LM head remain trainable via PEFT `modules_to_save`.
- New ordinary AddedToken `<stop>` has ID 151669; it is **not EOS**.
- Both new rows copy the existing `<<` token's row. This current SWITCH-inspired
  initialization is **not** the earlier vocabulary-mean experiment.
- Whole-MoE-layer FSDP wrapping, CPU offload, BF16; maximum sequence length 10,000.
- Completion is the sole assistant message's `content`, never `str(list/dict)`.
  Overlong sequences fail explicitly; no silent CoT truncation.
- Save every 10 steps and at 94. These are adapters **plus full vocabulary
  modules**, not standalone base models and not optimizer-resumable checkpoints.

To export a completed checkpoint for inference/DAPO:

```bash
python training/medqa/merge_sft.py \
  --checkpoint "$SFT_OUTPUT/checkpoint-94" --expected-step 94 \
  --output /path/to/new_merged_sft94
```

The base location is read from the checkpoint metadata; it must remain
accessible. Merge is CPU-only and needs substantial host RAM and >90 GiB free
disk space. It verifies that the trained embedding/head were actually loaded.

## 3. MedQA DAPO example (four GPUs)

Assemble the current runtime without altering the legacy repository or installed
packages. Supply the directory of the matching installed vLLM package (including
its compiled CUDA extensions):

```bash
python training/medqa/build_runtime.py \
  --vllm-package /path/to/env/lib/python3.10/site-packages/vllm \
  --output /path/to/new_estar_runtime

python training/medqa/prepare_dapo_data.py \
  --source /path/to/canonical_medqa_train.parquet \
  --model /path/to/new_merged_sft94 \
  --output /path/to/new_dapo_data

export RUNTIME_DIR=/path/to/new_estar_runtime
export MODEL_PATH=/path/to/new_merged_sft94
export DATA_DIR=/path/to/new_dapo_data
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_ROOT=/path/to/new_smoke_output
bash training/medqa/run_dapo.sh smoke

# After inspecting the one-update smoke output, start a separate formal run:
export RUN_ROOT=/path/to/new_formal_output
bash training/medqa/run_dapo.sh formal
```

The source parquet must contain the canonical **10,178-row MedQA train split**
with `prompt`, `reward_model.ground_truth` (A-D), and
`extra_info.{qid,split}`. Preparation preserves the user question, options,
gold, and IDs and installs the **same system prompt as SFT**. It reserves
context for both probe suffixes plus the probe answer; no official test data is
used. Validation is disabled in this launcher; its val-file setting is not a
held-out accuracy evaluation.

Formal configuration:

| Component | Setting |
| --- | --- |
| Policy initialization | Merged SFT step 94 |
| Updates | Two epochs; GRPO advantages with DAPO group filtering by raw policy accuracy |
| Prompt batches | Train 120; generation 360; 8 responses per prompt |
| Actor | Last two full MoE layers + final norm + LM head; embeddings frozen |
| Trainable parameters | 1,557,408,256 of 30,532,122,624 |
| Actor LR / PPO mini-batch | 1e-6 / 30 |
| Precision / placement | BF16, FSDP parameter and optimizer CPU offload |
| Rollout | TP1; temperature 0.6, top-p 0.95, top-k 20, repetition penalty 1.2 |
| Response cap | 12,288 tokens; context includes the maximum prompt and probe allowance |
| Probes | At most 8 eligible stop candidates, minimum separation 50 tokens |
| Online classifiers | A70 schema; final-consistency and gold-or-final targets |
| Saving | Every 14 updates, keep one actor checkpoint; model + HF model only |

This is the last four-GPU development configuration, **not a claim that four
GPUs are sufficient for all batches**. It reached seven updates before OOM.
A three-B200 configuration is not validated by this release; do not just relabel
GPU counts. The planned two-epoch run has not been completed.

## Prompt, reward, and classifier semantics

The complete format instruction lives in [prompt_contract.py](prompt_contract.py).
The assistant generates `<think>...<stop>...</think><final_answer>A-D</final_answer>`.
`<stop>` marks a completed reasoning point; by itself it does **not** end the
response. It belongs only inside `<think>...</think>`.

In the released DAPO implementation:

- Terminal accuracy comes from the **raw policy's own exact terminal answer**.
  A probe answer never replaces a wrong raw final answer for accuracy.
- Sequence reward separates accuracy (0.50) and valid format (0.25).
  Verified-stop credit is used by an action-level auxiliary loss, not added
  again to the sequence score. Its earliness half-life is 256 tokens.
- Negative/incorrect stop-action coefficients are 1; overflow coefficient is 0;
  action advantages are prompt-group centered.
- `cut_accepted_stop_tail=True` restricts actor training after the accepted
  boundary. This is **not evidence of physical generation-time early stopping**
  or a measured training wall-clock speedup.
- Online dual-classifier fitting uses A70 causal features and all physical
  candidate records. `classifier_affects_dapo=False`: classifier predictions do
  not accept/reject training stops; ground-truth verification does.
- The released [math classifiers](../../qwen3_30b_a3b/classifiers/README.md)
  and [offline MedQA classifier](../../qwen3_30b_a3b/classifiers_medqa/README.md)
  have distinct 22-feature contracts. Neither can replace these A70 online
  classifiers, even though the offline medical weight also uses the 30B backbone.

Classifier snapshots and batch audits are written under each run's `logs/`.
Actor checkpoints omit optimizer/scheduler/RNG state; they support evaluation
or a separately specified restart, **not lossless training resumption**.
No completed DAPO policy or online A70 classifier weight is published here.
The separately released offline MedQA weight is for ESTAR-LITE, not this DAPO
training recipe.

## Tests and provenance

`runtime_overlay/` contains only the differences required by the current
MedQA runtime against the repository's legacy veRL/vLLM trees, plus regression
tests. Upstream licenses and headers are retained.

```bash
python -m unittest discover -s training/medqa/tests -v
# Additional reward regressions in an environment with math-verify installed:
python -m unittest discover \
  -s training/medqa/runtime_overlay/verl/src -p test_reward_loss_medqa.py -v
```

`smoke_switch_3gpu.py` is an opt-in two-step tiny-MoE GPU test, not a benchmark.
Full CUDA training and inference still require hardware validation. These
current training examples and the manuscript-reported result tables are
distinct artifacts; this release does not assert exact reproduction of every
reported result.
