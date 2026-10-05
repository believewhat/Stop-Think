#!/usr/bin/env bash
# Public, explicit launcher for the current four-GPU MedQA recipe.
# Run from an allocated Linux GPU job; no remote login, keeper, or global kill.
set -Eeuo pipefail
MODE=${1:-}
[[ "$MODE" == smoke || "$MODE" == formal ]] || { echo 'usage: bash run_dapo.sh smoke|formal'; exit 2; }
: "${RUNTIME_DIR:?Set RUNTIME_DIR to the isolated build_runtime.py output}"
: "${MODEL_PATH:?Set MODEL_PATH to the merged SFT checkpoint-94}"
: "${DATA_DIR:?Set DATA_DIR to prepare_dapo_data.py output}"
: "${RUN_ROOT:?Set RUN_ROOT to a fresh output directory}"
: "${CUDA_VISIBLE_DEVICES:?Set the four GPUs assigned to this job}"
IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} -eq 4 ]] || { echo 'This recipe expects four allocated GPUs'; exit 2; }
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$RUNTIME_DIR")"
MODEL_PATH="$(realpath "$MODEL_PATH")"
DATA_DIR="$(realpath "$DATA_DIR")"
RUN_ROOT="$(realpath -m "$RUN_ROOT")"
[[ ! -e "$RUN_ROOT" ]] || { echo 'Refusing to reuse a run directory'; exit 2; }
PY=${PYTHON:-python}
TRAIN_PARQUET="$DATA_DIR/train.parquet"
RESULT_DIR="$RUN_ROOT/checkpoints"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"
if [[ "$MODE" == smoke ]]; then
  TRAIN_BATCH=6 GEN_BATCH=12 PPO_MINI=3 TOTAL_EPOCHS=1 TOTAL_STEPS=1
  SAVE_FREQ=1 MAX_KEEP=1 MIN_FREE_GIB=70
else
  TRAIN_BATCH=120 GEN_BATCH=360 PPO_MINI=30 TOTAL_EPOCHS=2 TOTAL_STEPS=null
  SAVE_FREQ=14 MAX_KEEP=1 MIN_FREE_GIB=170
fi
export PYTHONPATH="$PROJECT_ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=20260907
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export ADAPTTHINK_RUN_ID="estar_medqa_${MODE}_$$"
export ADAPTTHINK_RAY_TMPDIR="$(mktemp -d /tmp/esray.XXXXXX)"
export ADAPTTHINK_ROLLOUT_STOP_AUDIT_PATH="$LOG_ROOT/rollout_stop_counts.jsonl"
export ADAPTTHINK_STOP_CLASSIFIER_AUDIT_PATH="$LOG_ROOT/classifier_batches.jsonl"
export ADAPTTHINK_STOP_CLASSIFIER_FINAL_LIVE_CHECKPOINT_PATH="$LOG_ROOT/stop_classifier_final_consistency.latest.pkl"
export ADAPTTHINK_STOP_CLASSIFIER_GOLD_OR_FINAL_LIVE_CHECKPOINT_PATH="$LOG_ROOT/stop_classifier_gold_or_final.latest.pkl"
"${PY}" -B - "${PROJECT_ROOT}" "${TRAIN_PARQUET}" "${RESULT_DIR}" "${MIN_FREE_GIB}" "${MODEL_PATH}" <<'PY'
import json,pathlib,shutil,sys
import pandas as pd
from prompt_contract import PROMPT_CONTRACT_VERSION,SYSTEM_PROMPT,validate_messages
from verl.src.estar_a70_shared import A70_FEATURES,A70_FEATURE_SCHEMA_VERSION
from verl.src.stop_ray_trainer import STOP_CLASSIFIER_FEATURE_DIM,STOP_CLASSIFIER_FEATURE_NAMES
from transformers import AutoTokenizer,AutoConfig
project,parquet,out=map(pathlib.Path,sys.argv[1:4]); min_free=int(sys.argv[4]); model=pathlib.Path(sys.argv[5])
m=json.loads((model/'merge_complete.json').read_text())
assert m['state']=='complete' and m['checkpoint_step']==94
assert json.loads((model/'prompt_contract.json').read_text())['system_prompt']==SYSTEM_PROMPT
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
config=AutoConfig.from_pretrained(model,local_files_only=True)
assert config.model_type=='qwen3_moe' and config.vocab_size==151936 and config.num_hidden_layers==48
assert tokenizer.encode('<stop>',add_special_tokens=False)==[151669] and tokenizer.eos_token_id!=151669
assert getattr(config,'stop_token_id',None) is None
assert all((model/f['name']).stat().st_size==f['bytes'] for f in m['files'])
frame=pd.read_parquet(parquet)
assert len(frame)==10178 and frame.extra_info.map(lambda x:str(x['qid'])).nunique()==10178
for messages in frame.prompt:validate_messages(messages)
assert len(A70_FEATURES)==STOP_CLASSIFIER_FEATURE_DIM==70
assert tuple(A70_FEATURES)==tuple(STOP_CLASSIFIER_FEATURE_NAMES)
assert A70_FEATURE_SCHEMA_VERSION=='medqa_train_triple_t07_A70_v1'
assert "'classifier_affects_dapo': False" in (project/'verl/src/stop_ray_trainer.py').read_text()
out.parent.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(out.parent).free>=min_free*2**30
print(json.dumps({'rows':len(frame),'prompt_contract':PROMPT_CONTRACT_VERSION,'initial_sft_step':94,'a70_dim':70,'classifier_affects_dapo':False}))
PY
read -r MAX_PROMPT MAX_MODEL < <("${PY}" -B - "${DATA_DIR}/manifest.json" "${MODEL_PATH}" <<'PY'
import json,sys
from transformers import AutoTokenizer
m=json.load(open(sys.argv[1]))
tok=AutoTokenizer.from_pretrained(sys.argv[2],local_files_only=True)
extra=max(len(tok.encode(p,add_special_tokens=False)) for p in ('</think>\n<final_answer>','</think>\n<final_answer'))+1
assert m['max_model_len']>=m['max_prompt_length']+m['max_response_length']+extra,'probe context allowance missing'
print(m['max_prompt_length'],m['max_model_len'])
PY
)


[[ "${PREFLIGHT_ONLY:-0}" == 1 ]] && exit 0
cd "$PROJECT_ROOT"
config_args=()
[[ "${CONFIG_ONLY:-0}" == 1 ]] && config_args=(--cfg job)
"${PY}" -B -m verl.trainer.main_ppo_stop "${config_args[@]}" \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  +algorithm.filter_groups.enable=True +algorithm.filter_groups.metric=raw_policy_acc \
  +algorithm.filter_groups.max_num_gen_batches=10 +data.seed=20260907 \
  data.train_files="${TRAIN_PARQUET}" data.val_files="${TRAIN_PARQUET}" \
  data.train_batch_size="${TRAIN_BATCH}" +data.gen_batch_size="${GEN_BATCH}" \
  data.filter_overlong_prompts_workers=4 \
  data.max_prompt_length="${MAX_PROMPT}" data.max_response_length=12288 +data.trust_remote_code=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True +actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=1e-6 +actor_rollout_ref.actor.optim.foreach=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI}" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL}" \
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.use_torch_compile=False \
  +actor_rollout_ref.actor.empty_cache_before_optimizer_step=True +actor_rollout_ref.actor.empty_cache_before_backward=True \
  +actor_rollout_ref.actor.verified_stop_aux_loss_coef=1.0 +actor_rollout_ref.actor.overflow_stop_aux_loss_coef=0.0 \
  +actor_rollout_ref.actor.negative_stop_aux_loss_coef=1.0 +actor_rollout_ref.actor.incorrect_stop_aux_loss_coef=1.0 \
  +actor_rollout_ref.actor.prompt_group_centered_stop_action_advantage=True \
  +actor_rollout_ref.actor.post_accepted_tail_aux_loss_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  +actor_rollout_ref.actor.fsdp_config.partial_parameter_training="{enabled: true, mode: qwen3_moe_last2_full_layers_lm_head_norm, train_last_n_layers: 2, train_embed_tokens: false, expected_num_hidden_layers: 48, expected_total_numel: 30532122624, expected_trainable_numel: 1557408256}" \
  actor_rollout_ref.actor.checkpoint.contents="['model','hf_model']" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.max_model_len="${MAX_MODEL}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=24576 actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.gpu_memory_utilization=0.78 \
  actor_rollout_ref.rollout.temperature=0.6 actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=20 \
  +actor_rollout_ref.rollout.repetition_penalty=1.2 +actor_rollout_ref.rollout.stop_token_id=151669 \
  +actor_rollout_ref.rollout.max_stop_count=8 +actor_rollout_ref.rollout.stop_reward_half_life_tokens=256.0 \
  +actor_rollout_ref.rollout.final_consistency_credit=1.0 +actor_rollout_ref.rollout.min_verified_stop_separation_tokens=50 \
  +actor_rollout_ref.rollout.classifier_reasoning_budget=5000 +actor_rollout_ref.rollout.terminate_on_stop_overflow=False \
  +actor_rollout_ref.rollout.cut_accepted_stop_tail=True \
  +actor_rollout_ref.rollout.post_accepted_tail_penalty_normalization_tokens=256 \
  actor_rollout_ref.rollout.free_cache_engine=False actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True reward_model.enable=False reward_model.reward_manager=naive \
  +reward_model.overlong_buffer.enable=True +reward_model.overlong_buffer.len=2048 \
  +reward_model.overlong_buffer.penalty_factor=1.0 \
  custom_reward_function.path="${PROJECT_ROOT}/verl/src/reward_loss.py" custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 trainer.logger="['console']" trainer.project_name=DAPO_medqa \
  trainer.experiment_name="q30_switch94_dapo4_${MODE}" trainer.n_gpus_per_node=4 trainer.nnodes=1 \
  trainer.val_before_train=False +trainer.require_rollout_stop_audit=True +trainer.train_stop_classifier_online=True \
  +trainer.stop_classifier_audit_path="${ADAPTTHINK_STOP_CLASSIFIER_AUDIT_PATH}" \
  +trainer.stop_classifier_final_live_checkpoint_path="${ADAPTTHINK_STOP_CLASSIFIER_FINAL_LIVE_CHECKPOINT_PATH}" \
  +trainer.stop_classifier_gold_or_final_live_checkpoint_path="${ADAPTTHINK_STOP_CLASSIFIER_GOLD_OR_FINAL_LIVE_CHECKPOINT_PATH}" \
  trainer.default_local_dir="${RESULT_DIR}" trainer.default_hdfs_dir=null trainer.resume_mode=disable \
  trainer.save_freq="${SAVE_FREQ}" trainer.max_actor_ckpt_to_keep="${MAX_KEEP}" trainer.test_freq=-1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" trainer.total_training_steps="${TOTAL_STEPS}" \
  2>&1 | tee "${LOG_ROOT}/train.log"

# Do not infer successful two-epoch training merely from a process exit.
# Review train.log, checkpoints/global_step_*/actor, and classifier_batches.jsonl.
echo "Trainer exited successfully; inspect checkpoint and dual-classifier artifacts in $RUN_ROOT"
