"""Bounded 3-GPU tiny-MoE training/save/reload test; never loads 30B weights."""
import json
import os
from pathlib import Path
import tempfile
import torch
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, PreTrainedTokenizerFast, TrainingArguments, set_seed
from train_sft import SwitchTrainer, initialize_anchor, add_lora, adapter_inventory
from sft_data import TokenDataset, Collator

set_seed(42)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.distributed.init_process_group('nccl')
root = Path(os.environ['SWITCH_TEST_OUTPUT'])
args = TrainingArguments(output_dir=str(root), max_steps=2, per_device_train_batch_size=1,
    gradient_accumulation_steps=2, learning_rate=1e-3, logging_steps=1, save_steps=1,
    save_only_model=True, bf16=True, report_to=[], remove_unused_columns=False,
    fsdp='full_shard auto_wrap offload', fsdp_config={
        'transformer_layer_cls_to_wrap': ['Qwen3MoeDecoderLayer'], 'use_orig_params': True,
        'sync_module_states': True, 'activation_checkpointing': True, 'cpu_ram_efficient_loading': False},
    seed=42, dataloader_num_workers=0)
config = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
    moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
    max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
base = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
base.config.use_cache = False
initialize_anchor(base, 3, 80)
model, audit = add_lora(base, toy=True)
vocab = {f't{i}': i for i in range(256)}
raw = Tokenizer(WordLevel(vocab, unk_token='t0'))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='t0', eos_token='t2', pad_token='t0')
rows = []
for i in range(12):
    ids = [1] + [4 + (i + j) % 60 for j in range(6)] + [80] + [12, 15, 19, 2]
    rows.append({'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': [-100]*3+ids[3:]})
trainer = SwitchTrainer(model=model, args=args, train_dataset=TokenDataset(rows),
    data_collator=Collator(0), processing_class=tokenizer, stop_id=80,
    audit={'model': 'tiny-test-only', 'format_contract': {}, 'lora': audit})
result = trainer.train()
assert trainer.state.global_step == 2
state = trainer.accelerator.get_state_dict(trainer.model)
if trainer.is_world_process_zero():
    for step in (1, 2):
        adapter_inventory(root / f'checkpoint-{step}')
    lora_b = [t for n,t in state.items() if 'lora_B' in n]
    assert any(t.float().abs().max().item() > 0 for t in lora_b), 'LoRA did not update'
    for part in ['embed_tokens', 'lm_head']:
        old = next(t for n,t in state.items() if part in n and 'original_module.weight' in n)
        new = next(t for n,t in state.items() if part in n and 'modules_to_save.default.weight' in n)
        assert not torch.equal(old, new), f'{part} did not update'
    restored = PeftModel.from_pretrained(Qwen3MoeForCausalLM(config), root / 'checkpoint-2', local_files_only=True)
    assert restored.get_input_embeddings().weight.shape == (256, 64)
    print('SWITCH_TINY_MOE_3GPU_PASS ' + json.dumps(result.metrics), flush=True)
torch.distributed.barrier()
torch.distributed.destroy_process_group()
