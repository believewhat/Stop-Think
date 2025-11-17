#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, argparse
import numpy as np
import pandas as pd

# ——保持和你环境一致的一些 env（可按需删减）——
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from vllm import LLM, SamplingParams

# ---------- 提取 <final_answer> 的字母 ----------
_FA_BLOCK = re.compile(r"<\s*final_answer\s*>(.*?)<\s*/\s*final_answer\s*>", flags=re.I | re.S)
_PAT_STRONG = re.compile(r"(?i)(?:^|\b(?:answer|ans)\s*[:\-]?\s*)([ABCD])(?:\s*[:\.\)]|\s|$)")
_PAT_SOFT   = re.compile(r"(?i)(?<![A-Za-z])([ABCD])(?![A-Za-z])")

def pick_letter(s: str):
    if not isinstance(s, str) or not s:
        return None
    mfa = _FA_BLOCK.search(s)
    seg = mfa.group(1) if mfa else s
    m = _PAT_STRONG.search(seg)
    if m:
        return m.group(1).upper()
    m2 = _PAT_SOFT.search(seg)
    return m2.group(1).upper() if m2 else None

def map_gold(x):
    s = str(x).strip()
    if s.isdigit():
        v = int(s)
        if 0 <= v <= 3:
            return "ABCD"[v]
    m = re.search(r'[A-D]', s, flags=re.I)
    return m.group(0).upper() if m else None

# ---------- 构造提示（和你在线 early-stop 版本对齐） ----------
TASK = "Please output a single best option from Choices as your final answer."
CUST = ("Solve them in a step-by-step fashion, starting by summarizing the available information. "
        "Please output your all reasoning into <think> thoughts </think>.")
SYS  = (
    "You are a medical professional. Your job is to assist with medical questions.\n"
    f"{TASK}\n{CUST}\n"
    "Please wrap your final answer in <final_answer> your answer </final_answer>."
)

def build_user(question, choices):
    return (f"**Question**:\n{question}\n**Choices**:\n"
            f"A: {choices['A']}\nB: {choices['B']}\nC: {choices['C']}\nD: {choices['D']}\n"
            "Now based on the above reasoning, you must give the correct option in the form "
            "<final_answer>your answer(A,B,C,D,only one letter)</final_answer>. You must give an option.")

def build_base_prompt(question, choices):
    # 这里和你之前 online early-stop 的 system/user/assistant 模式保持一致
    return f"<system>{SYS}</system>\nuser\n{build_user(question, choices)}\nassistant\n<think> "

# ---------- 主流程：vanilla 同步推理 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out_csv", default="vanilla_inference_results.csv")

    # 采样参数（主序列）
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--max_tokens", type=int, default=8000)

    # vLLM 配置
    ap.add_argument("--gpu_util", type=float, default=0.8)
    ap.add_argument("--max_model_len", type=int, default=8000)
    ap.add_argument("--max_num_seqs", type=int, default=128)

    # 批大小（同步 + 批推理）
    ap.add_argument("--batch_size", type=int, default=16)

    # 某些 Qwen 需要 trust_remote_code
    ap.add_argument("--trust_remote_code", action="store_true")

    args = ap.parse_args()

    # ---------- 读取数据 ----------
    df = pd.read_parquet(args.input_parquet)
    df["qid"] = df["qid"].astype(str)

    prompts = []
    qids = []
    golds = []

    for _, row in df.iterrows():
        qid = row["qid"]
        q   = f"{row['sent1']}{row['sent2']}"
        ch  = {
            "A": row["ending0"],
            "B": row["ending1"],
            "C": row["ending2"],
            "D": row["ending3"],
        }
        gold = row.get("answer_idx", None)

        base_prompt = build_base_prompt(q, ch)
        prompts.append(base_prompt)
        qids.append(qid)
        golds.append(gold)

    N = len(prompts)
    print(f"[INIT] samples={N} | batch_size={args.batch_size}", flush=True)

    # ---------- 初始化 vLLM LLM ----------
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
    )

    sampling_params = SamplingParams(
        temperature=float(args.temp),
        top_p=float(args.top_p),
        repetition_penalty=float(args.rep),
        max_tokens=int(args.max_tokens),
        # vanilla：这里不早停，不加 stop / stop_token_ids
        # 如果你想保证生成到 </final_answer> 就停，可以设置 stop=["</final_answer>"]
        # 但为了和你 online 版本“max_tokens 上限”更可比，这里先留空
    )

    # ---------- 同步批推理 ----------
    t0 = time.time()
    rows = []
    total_gen_tokens = 0

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        batch_prompts = prompts[start:end]
        batch_qids    = qids[start:end]
        batch_golds   = golds[start:end]

        print(f"[BATCH] {start}-{end} / {N}", flush=True)

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"[ERROR] vLLM batch generate failed: {e}", flush=True)
            # 每个样本写一个空结果，避免中断
            for qid, gold in zip(batch_qids, batch_golds):
                rows.append(dict(
                    qid=qid,
                    final_text="",
                    gold=gold,
                    gen_tokens=None,
                    error=str(e),
                ))
            continue

        for out, qid, gold in zip(outputs, batch_qids, batch_golds):
            try:
                # vLLM 每个请求通常只有一个 output
                text = out.outputs[0].text if out.outputs else ""
                gen_tok = len(out.outputs[0].token_ids) if out.outputs else None
                total_gen_tokens += (gen_tok or 0)
                rows.append(dict(
                    qid=qid,
                    final_text=text,
                    gold=gold,
                    gen_tokens=gen_tok,
                    error="",
                ))
            except Exception as e:
                rows.append(dict(
                    qid=qid,
                    final_text="",
                    gold=gold,
                    gen_tokens=None,
                    error=f"postprocess_error:{e}",
                ))

    elapsed = time.time() - t0
    print(f"[TIME] total_elapsed={elapsed:.1f}s", flush=True)

    # ---------- 写结果 + 统计 ACC ----------
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv} rows={len(out)}", flush=True)

    out["pred_letter"] = out["final_text"].map(pick_letter)
    out["gold_letter"] = out["gold"].map(map_gold)

    valid_mask = out["gold_letter"].notna()
    if valid_mask.any():
        acc = (out.loc[valid_mask, "pred_letter"] == out.loc[valid_mask, "gold_letter"]).mean()
    else:
        acc = float("nan")

    avg_gen_tok = pd.to_numeric(out["gen_tokens"], errors="coerce").dropna().mean()

    print(f"[SUMMARY] ACC={acc*100:.2f}% "
          f"| avg_gen_tokens={avg_gen_tok:.1f} "
          f"| elapsed={elapsed:.1f}s "
          f"| tok/s={total_gen_tokens/elapsed if elapsed>0 else 0:.1f}", flush=True)

if __name__ == "__main__":
    main()

"""
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=1 python -u inference_vanilla_baseline.py \
  --input_parquet /home/jwang/Project/qwen3_output/data/MedQA/MedQA-USMLE-4-options-parquet/test.parquet \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --out_csv vanilla_inference_results.csv \
  --temp 0.6 \
  --top_p 0.95 \
  --rep 1.2 \
  --max_tokens 10000 \
  --batch_size 64 \
  --gpu_util 0.8 \
  --max_model_len 10000 \
  --max_num_seqs 128 \
  --trust_remote_code
"""