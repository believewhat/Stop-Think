"""SWITCH-inspired marker SFT for Qwen3-MoE; no latent reasoning or RL.

Freeze backbone weights, train LoRA plus full embedding/LM-head modules.
Reuse the existing completion extraction, prompt contract and nontruncating data.
"""
from __future__ import annotations

import argparse
import collections
from datetime import timedelta
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tokenizers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from sft_data import Collator, STOP, STOP_ID, TokenDataset, prepare, read_jsonl
import prompt_contract as contract
from fsdp_prefix_index import install_fsdp_prefix_index

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def initialize_anchor(model, anchor_id: int, stop_id: int):
    audit = {'strategy': 'switch_anchor_copy', 'anchor_token': '<<', 'anchor_id': anchor_id,
             'stop_id': stop_id, 'matrices': {}}
    with torch.no_grad():
        for name, layer in [('input_embedding', model.get_input_embeddings()),
                            ('output_lm_head', model.get_output_embeddings())]:
            assert not layer.weight.is_meta
            layer.weight[stop_id].copy_(layer.weight[anchor_id])
            error = (layer.weight[stop_id].float() - layer.weight[anchor_id].float()).abs().max().item()
            assert error == 0
            audit['matrices'][name] = {'copy_error': error, 'norm': layer.weight[stop_id].float().norm().item()}
    return audit


def add_lora(model, toy=False):
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, target_modules=TARGET_MODULES,
        modules_to_save=['embed_tokens', 'lm_head'], bias='none', task_type='CAUSAL_LM'))
    # PEFT promotes adapter weights to FP32 by default, but FSDP must flatten
    # each whole sparse layer in one dtype. Accelerate then maintains the
    # sharded FP32 master parameters under its BF16 mixed-precision policy.
    model.to(dtype=torch.bfloat16)
    counts = collections.Counter()
    matched = collections.Counter()
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            matched[name.split('.')[-1]] += 1
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            category = 'lora' if '.lora_' in name else 'vocab' if '.modules_to_save.' in name else 'unexpected'
            counts[category] += parameter.numel()
            assert category != 'unexpected', name
        elif '.mlp.gate.' in name:
            counts['frozen_router'] += parameter.numel()
    assert counts['lora'] > 0 and counts['vocab'] > 0 and counts['frozen_router'] > 0
    assert set(matched) == set(TARGET_MODULES), matched
    if not toy:
        assert all(matched[n] == 48 for n in TARGET_MODULES[:4]), matched
        assert all(matched[n] == 48 * 128 for n in TARGET_MODULES[4:]), matched
    return model, {'parameters': dict(counts), 'matched_lora_modules': dict(matched),
                   'modules_to_save': ['embed_tokens', 'lm_head'], 'router_frozen': True,
                   'r': 32, 'alpha': 64, 'dropout': 0.05}


def adapter_inventory(path: Path):
    from safetensors import safe_open
    config = json.loads((path / 'adapter_config.json').read_text())
    assert set(config['modules_to_save']) == {'embed_tokens', 'lm_head'}
    files = sorted(path.glob('adapter_model*.safetensors'))
    assert files, f'No adapter weights in {path}'
    keys = []
    for file in files:
        with safe_open(file, framework='pt', device='cpu') as handle:
            keys.extend(handle.keys())
    assert any('lora_A' in k for k in keys) and any('lora_B' in k for k in keys)
    assert any(k.endswith('embed_tokens.weight') for k in keys)
    assert any(k.endswith('lm_head.weight') for k in keys)
    assert (path / 'tokenizer_config.json').is_file()
    return {'files': [{'name': f.name, 'bytes': f.stat().st_size} for f in files], 'tensor_count': len(keys)}


class SwitchTrainer(Trainer):
    """Ordinary completion CE; stop diagnostics do not change its gradient."""
    def __init__(self, *args, audit=None, stop_id=STOP_ID, **kwargs):
        install_fsdp_prefix_index()
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.audit = audit or {}
        self.stop_id = stop_id
        self.diag = None

    def _fsdp_qlora_plugin_updates(self):
        # PEFT's default policy wraps trainable Linear leaves individually.
        # Sparse experts execute in different orders on different ranks: their
        # individual FSDP collectives would deadlock. Keep each whole MoE layer
        # in one unit, with use_orig_params=True supporting frozen/trainable mix.
        if self.is_fsdp_enabled:
            from functools import partial
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
            plugin = self.accelerator.state.fsdp_plugin
            assert plugin.use_orig_params
            plugin.auto_wrap_policy = partial(transformer_auto_wrap_policy,
                transformer_layer_cls={Qwen3MoeDecoderLayer})

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous().to(logits.device)
        flat_logits, flat_labels = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        stats = torch.zeros(5, device=logits.device, dtype=torch.float32)
        for start in range(0, flat_labels.numel(), 64):
            target = flat_labels[start:start + 64]
            scores = flat_logits[start:start + 64].float()
            losses = F.cross_entropy(scores, target, reduction='none', ignore_index=-100)
            total = total + losses.sum()
            with torch.no_grad():
                stop = target.eq(self.stop_id)
                valid = target.ne(-100)
                ordinary = valid & ~stop
                stats[0] += losses[stop].sum()
                stats[1] += stop.sum()
                stats[2] += losses[ordinary].sum()
                stats[3] += ordinary.sum()
                stats[4] += (scores.argmax(-1).eq(self.stop_id) & stop).sum()
        loss = total / flat_labels.ne(-100).sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite ordinary completion CE')
        if model.training:
            self.diag = stats if self.diag is None else self.diag + stats
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if 'loss' in logs and self.diag is not None:
            values = self.diag.detach().cpu().tolist()
            # These are explicitly local-rank diagnostics, not global metrics.
            logs = dict(logs)
            logs['diag_rank0_stop_ce'] = values[0] / max(values[1], 1)
            logs['diag_rank0_other_ce'] = values[2] / max(values[3], 1)
            logs['diag_rank0_stop_top1'] = values[4] / max(values[1], 1)
            logs['diag_rank0_stop_labels'] = int(values[1])
            self.diag = None
        return super().log(logs, start_time)

    def _save(self, output_dir=None, state_dict=None):
        output = Path(output_dir or self.args.output_dir)
        print(f'CHECKPOINT_SAVE_BEGIN step={self.state.global_step} path={output}', flush=True)
        super()._save(str(output), state_dict)
        if self.args.should_save:
            (output / 'prompt_contract.json').write_text(json.dumps(self.audit['format_contract'], indent=2) + '\n')
            (output / 'run_config.json').write_text(json.dumps(self.audit, indent=2) + '\n')
            info = adapter_inventory(output)
            (output / 'adapter_complete.json').write_text(json.dumps(info, indent=2) + '\n')

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            output = Path(self._get_output_dir(trial)) / f'checkpoint-{self.state.global_step}'
            info = adapter_inventory(output)
            assert json.loads((output / 'trainer_state.json').read_text())['global_step'] == self.state.global_step
            (output / 'checkpoint_complete.json').write_text(json.dumps({
                'state': 'complete', 'global_step': self.state.global_step, 'epoch': self.state.epoch,
                'artifact': 'peft_adapter_plus_full_embedding_and_lm_head',
                'base_model_required': self.audit['model'], 'model_only': True,
                'optimizer_resume_available': False, **info}, indent=2) + '\n')
            print(f'CHECKPOINT_SAVE_COMPLETE step={self.state.global_step} path={output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), 'Fresh output required; do not silently restart'
    assert args.learning_rate == 2e-5, 'User-selected LR is retained for this experiment'
    set_seed(42)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    assert world == 3, world
    manifest = json.loads((args.data / 'manifest.json').read_text())
    assert manifest['training_ready'] and manifest['split_rows'] == {'train': 3085, 'eval': 254}
    assert manifest['all_stops_inside_think'] and manifest['all_stops_at_sentence_ends']
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    assert len(tokenizer) == STOP_ID
    anchor = tokenizer.encode('<<', add_special_tokens=False)
    assert len(anchor) == 1 and anchor[0] != tokenizer.eos_token_id
    # Ordinary AddedToken preserves the existing visible-marker decoding contract.
    # Atomicity does not require the decode-time special-token removal flag.
    added = tokenizer.add_tokens([AddedToken(STOP, normalized=False, special=False)], special_tokens=False)
    assert added == 1 and tokenizer.encode(STOP, add_special_tokens=False) == [STOP_ID]
    assert tokenizer.eos_token_id != STOP_ID
    tokenizer.pad_token = tokenizer.eos_token
    train_rows, eval_rows = read_jsonl(args.data / 'train.jsonl'), read_jsonl(args.data / 'eval.jsonl')
    for row in train_rows + eval_rows:
        contract.validate_messages(row['prompt'])
    train, train_audit = prepare(train_rows, tokenizer, 10000, 'train')
    _, eval_audit = prepare(eval_rows, tokenizer, 10000, 'eval')
    assert train_audit['rows'] == 3085 and train_audit['stop_labels'] == 14594
    assert eval_audit['rows'] == 254 and eval_audit['stop_labels'] == 1210
    updates = math.ceil(len(train) / 33)
    assert updates == 94
    audit = {'schema': 'q30_switch_inspired_lora_v1', 'model': args.model, 'data': str(args.data),
             'train': train_audit, 'eval': eval_audit, 'learning_rate': args.learning_rate,
             'epochs': 1, 'steps': updates, 'effective_batch': 33, 'gradient_accumulation_steps': 11,
             'loss': 'ordinary_completion_ce', 'stop_loss_weight': 1, 'stop_id': STOP_ID,
             'anchor_id': anchor[0], 'save_steps': 10, 'save_only_model': True,
             'keeper_policy': 'disabled', 'automatic_evaluation': False, 'automatic_dapo': False,
             'format_contract': {**contract.prompt_contract_metadata(), 'selection': 'stop_view_v3',
                                 'system_prompt': contract.SYSTEM_PROMPT},
             'switch_adaptations': ['Qwen3-30B-A3B MoE, not dense 8B', 'LR 2e-5 and one epoch retained',
                                   'existing prompt/data and visible AddedToken contract retained',
                                   'MoE routers frozen; expert projections adapted', 'no latent computation']}
    if args.preflight_only:
        print(json.dumps(audit, indent=2))
        return
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=1800))
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=11, learning_rate=args.learning_rate, warmup_ratio=0.03,
        lr_scheduler_type='cosine', weight_decay=0.0, logging_steps=1, logging_first_step=True,
        eval_strategy='no', save_strategy='steps', save_steps=10, save_total_limit=None,
        save_only_model=True, bf16=True, tf32=True, optim='adamw_torch',
        remove_unused_columns=False, report_to=[], dataloader_num_workers=0,
        ddp_timeout=1800, fsdp='full_shard auto_wrap offload',
        fsdp_config={'transformer_layer_cls_to_wrap': ['Qwen3MoeDecoderLayer'],
                     'backward_prefetch': 'backward_pre', 'forward_prefetch': False,
                     'use_orig_params': True, 'sync_module_states': True,
                     'cpu_ram_efficient_loading': True, 'activation_checkpointing': True,
                     'limit_all_gathers': True}, seed=42, data_seed=42)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='flash_attention_2', low_cpu_mem_usage=True)
    assert model.config.model_type == 'qwen3_moe'
    assert sum(p.numel() for p in model.parameters()) == 30_532_122_624
    assert model.get_input_embeddings().weight.shape[0] == 151936
    if torch.distributed.get_rank() == 0:
        audit['initialization'] = initialize_anchor(model, anchor[0], STOP_ID)
        print('STOP_ROW_INITIALIZED ' + json.dumps(audit['initialization']), flush=True)
    model.config.use_cache = False
    model, lora_audit = add_lora(model)
    audit['lora'] = lora_audit
    print('LORA_PARAMETER_AUDIT ' + json.dumps(lora_audit), flush=True)
    trainer = SwitchTrainer(model=model, args=training_args, train_dataset=TokenDataset(train),
                            data_collator=Collator(tokenizer.pad_token_id),
                            processing_class=tokenizer, audit=audit)
    if trainer.is_world_process_zero():
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'preflight_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    result = trainer.train()
    assert trainer.state.global_step == 94 and abs(trainer.state.epoch - 1.0) < 1e-6
    trainer.save_model(str(args.output))
    if trainer.is_world_process_zero():
        info = adapter_inventory(args.output)
        for step in list(range(10, 91, 10)) + [94]:
            mark = json.loads((args.output / f'checkpoint-{step}' / 'checkpoint_complete.json').read_text())
            assert mark['global_step'] == step and mark['state'] == 'complete'
        final = {**audit, 'state': 'complete', 'global_step': 94, 'epoch': trainer.state.epoch,
                 'train_metrics': result.metrics, 'adapter_inventory': info}
        (args.output / 'train_complete.json').write_text(json.dumps(final, indent=2) + '\n')
        print('SFT_COMPLETE ' + json.dumps(result.metrics), flush=True)


if __name__ == '__main__':
    main()
