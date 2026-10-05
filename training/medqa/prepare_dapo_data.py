"""Apply the SFT prompt contract to canonical MedQA TRAIN parquet, preserving gold/qids."""
import argparse
import json
from pathlib import Path
from prompt_contract import (
    SYSTEM_PROMPT, PROMPT_CONTRACT_VERSION, normalize_serialized_messages,
    validate_messages, render_generation_prompt,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Canonical MedQA train parquet with prompt/reward_model/extra_info")
    parser.add_argument("--model", type=Path, required=True, help="Merged SFT step-94 export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    contract = json.loads((args.model / "prompt_contract.json").read_text())
    assert contract["version"] == PROMPT_CONTRACT_VERSION and contract["system_prompt"] == SYSTEM_PROMPT
    marker = json.loads((args.model / "merge_complete.json").read_text())
    assert marker["state"] == "complete" and marker["checkpoint_step"] == 94
    frame = pd.read_parquet(args.source)
    assert len(frame) == 10178
    qids = [str(x["qid"]) for x in frame.extra_info]
    assert len(set(qids)) == len(frame)
    assert all(x["split"] == "train" for x in frame.extra_info), "Only MedQA train is allowed"
    assert all(x["ground_truth"] in "ABCD" and len(x["ground_truth"]) == 1 for x in frame.reward_model)
    gold = [x["ground_truth"] for x in frame.reward_model]
    prompts = []
    for old in frame.prompt:
        messages = normalize_serialized_messages(old)
        assert len(messages) == 2 and [x["role"] for x in messages] == ["system", "user"]
        user = messages[1]["content"]
        messages[0]["content"] = SYSTEM_PROMPT
        validate_messages(messages)
        assert messages[1]["content"] == user
        prompts.append(messages)
    frame["prompt"] = prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert tokenizer.encode("<stop>", add_special_tokens=False) == [151669]
    lengths = [len(tokenizer.encode(render_generation_prompt(tokenizer, p), add_special_tokens=False))
               for p in prompts]
    max_prompt = max(1200, ((max(lengths) + 127) // 128) * 128)
    extra = max(len(tokenizer.encode(s, add_special_tokens=False))
                for s in ("</think>\n<final_answer>", "</think>\n<final_answer")) + 1
    max_model = ((max_prompt + 12288 + extra + 127) // 128) * 128
    args.output.mkdir(parents=True)
    path = args.output / "train.parquet"
    frame.to_parquet(path, index=False)
    reread = pd.read_parquet(path)
    assert [str(x["qid"]) for x in reread.extra_info] == qids
    assert [x["ground_truth"] for x in reread.reward_model] == gold
    audit = {
        "state": "complete", "rows": len(frame), "source": str(args.source.resolve()),
        "source_split": "train", "official_test_used": False,
        "prompt_contract": PROMPT_CONTRACT_VERSION, "changed_only_system_prompt": True,
        "max_prompt_tokens_observed": max(lengths), "max_prompt_length": max_prompt,
        "max_response_length": 12288, "probe_context_extra_tokens": extra,
        "max_model_len": max_model,
    }
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
