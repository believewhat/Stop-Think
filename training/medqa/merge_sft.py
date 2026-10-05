"""CPU-only immutable export of a completed SWITCH-style PEFT checkpoint."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, default=10)
    args = parser.parse_args()
    assert not args.output.exists(), 'Fresh export directory required'
    marker = json.loads((args.checkpoint / 'checkpoint_complete.json').read_text())
    state = json.loads((args.checkpoint / 'trainer_state.json').read_text())
    assert args.expected_step > 0
    assert marker['state'] == 'complete' and marker['global_step'] == state['global_step'] == args.expected_step
    assert marker['artifact'] == 'peft_adapter_plus_full_embedding_and_lm_head'
    base = marker['base_model_required']
    # base_model_required is recorded by this run, not a server-specific path.
    assert all((args.checkpoint / f['name']).stat().st_size == f['bytes'] for f in marker['files'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.output.parent).free > 90 * 2**30
    torch.set_num_threads(8)
    started = time.monotonic()
    print('MERGE_LOAD_BASE_CPU', base, flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 'cpu'}, low_cpu_mem_usage=True,
        attn_implementation='eager')
    print('MERGE_LOAD_ADAPTER_AND_FULL_VOCAB', flush=True)
    model = PeftModel.from_pretrained(model, args.checkpoint, local_files_only=True,
        is_trainable=False, autocast_adapter_dtype=True)
    assert model.peft_config['default'].modules_to_save == ['embed_tokens', 'lm_head']
    expected_vocab = {}
    for f in marker['files']:
        with safe_open(args.checkpoint / f['name'], framework='pt', device='cpu') as handle:
            for part in ('embed_tokens', 'lm_head'):
                keys = [k for k in handle.keys() if k.endswith(part + '.weight')]
                if keys:
                    assert len(keys) == 1
                    expected_vocab[part] = handle.get_tensor(keys[0]).to(torch.bfloat16)
    assert set(expected_vocab) == {'embed_tokens', 'lm_head'}
    assert torch.equal(model.get_input_embeddings().weight, expected_vocab['embed_tokens'])
    assert torch.equal(model.get_output_embeddings().weight, expected_vocab['lm_head'])
    print('MERGE_FULL_VOCAB_LOAD_VERIFIED; MERGE_DELTAS_BEGIN', flush=True)
    model = model.merge_and_unload(safe_merge=True)
    model.to(torch.bfloat16)
    assert not any('lora_' in n or 'modules_to_save' in n for n, _ in model.named_parameters())
    assert sum(p.numel() for p in model.parameters()) == 30_532_122_624
    assert torch.equal(model.get_input_embeddings().weight, expected_vocab['embed_tokens'])
    assert torch.equal(model.get_output_embeddings().weight, expected_vocab['lm_head'])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    assert tokenizer.encode('<stop>', add_special_tokens=False) == [151669]
    assert tokenizer.eos_token_id != 151669
    eos = model.generation_config.eos_token_id
    assert 151669 not in (eos if isinstance(eos, list) else [eos])
    model.config.use_cache = True
    model.config.torch_dtype = torch.bfloat16
    args.output.mkdir(parents=True)
    print('MERGE_SAVE_BF16_BEGIN', str(args.output), flush=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(args.output)
    for name in ('prompt_contract.json', 'run_config.json'):
        shutil.copy2(args.checkpoint / name, args.output / name)
    index = json.loads((args.output / 'model.safetensors.index.json').read_text())
    assert index['metadata']['total_size'] == 2 * 30_532_122_624
    shards = [args.output / n for n in sorted(set(index['weight_map'].values()))]
    assert all(p.is_file() and p.stat().st_size > 0 for p in shards)
    result = {'state': 'complete', 'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_step': args.expected_step,
        'base_model': base, 'export_dtype': 'bfloat16', 'format': 'merged_full_model',
        'full_embedding_and_lm_head_verified': True, 'stop_id': 151669,
        'stop_is_eos': False, 'elapsed_seconds': time.monotonic() - started,
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'files': [{'name': p.name, 'bytes': p.stat().st_size} for p in shards]}
    (args.output / 'merge_complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print('MERGE_COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__': main()
