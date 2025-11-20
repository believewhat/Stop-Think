#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, argparse
import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from vllm import LLM, SamplingParams

# ---------- 提取 <final_answer> 的字母（和你现在脚本一样） ----------
_FA_BLOCK = re.compile(
    r"<\s*final_answer\s*>(.*?)<\s*/\s*final_answer\s*>",
    flags=re.I | re.S,
)
_PAT_STRONG = re.compile(
    r"(?i)(?:^|\b(?:answer|ans)\s*[:\-]?\s*)([ABCD])(?:\s*[:\.\)]|\s|$)"
)
_PAT_SOFT = re.compile(r"(?i)(?<![A-Za-z])([ABCD])(?![A-Za-z])")


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
    m = re.search(r"[A-D]", s, flags=re.I)
    return m.group(0).upper() if m else None


# ---------- 和在线脚本一致的系统 prompt & user 构造 ----------

TASK = "Please output a single best option from Choices as your final answer."
CUST = (
    "Solve them in a step-by-step fashion, starting by summarizing the available information. "
    "Please output your all reasoning into <think> thoughts </think>."
)
SYS = (
    "You are a medical professional. Your job is to assist with medical questions.\n"
    f"{TASK}\n{CUST}\n"
    "Please wrap your final answer in <final_answer> your answer </final_answer>."
)


def build_user(question, choices):
    return (
        f"**Question**:\n{question}\n**Choices**:\n"
        f"A: {choices['A']}\nB: {choices['B']}\nC: {choices['C']}\nD: {choices['D']}\n"
        "Now based on the above reasoning, you must give the correct option in the form "
        "<final_answer>your answer(A,B,C,D,only one letter)</final_answer>. You must give an option."
    )


def clean_think_text(raw: str) -> str:
    """
    用在 second-pass：
    - 去掉已有的 <final_answer>...</final_answer>
    - 去掉可能残留的 <think> / </think>
    留下纯 reasoning 文字，用作固定 <think> 块。
    """
    if not isinstance(raw, str):
        return ""
    s = raw

    # 删掉整个 final_answer block
    s = _FA_BLOCK.sub("", s)
    # 再把零散的 <final_answer> / </final_answer> 清掉
    s = re.sub(r"<\s*/?\s*final_answer[^>]*>", "", s, flags=re.I)

    # 删掉旧的 think 标签（有些流可能已经自己补了 </think>）
    s = re.sub(r"<\s*/?\s*think[^>]*>", "", s, flags=re.I)

    # 收尾空白
    return s.strip()


def build_prompt_with_think(question: str, choices: dict, think_clean: str) -> str:
    """
    按你的需求构造：

    输入 = question + <think> think </think>
    也就是：

    <system>SYS</system>
    user
    ...question + choices...
    assistant
    <think> {think_clean} </think>

    从 </think> 后面开始生成 COT（不手动加 <final_answer>）。
    """
    u = build_user(question, choices)
    return (
        f"<system>{SYS}</system>\n"
        f"user\n{u}\n"
        f"assistant\n<think> {think_clean} </think>\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True,
                    help="原始 MedQA parquet（含 sent1/sent2/ending0..3/answer_idx/qid）")
    ap.add_argument("--think_csv", required=True,
                    help="上一步 online early-stop 写出的 CSV（要有 qid, think_text）")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu_util", type=float, default=0.8)
    ap.add_argument("--max_model_len", type=int, default=10000)
    ap.add_argument("--max_num_seqs", type=int, default=128)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--out_csv", default="think2cot_results.csv")

    args = ap.parse_args()

    t0 = time.time()

    # 1) 读原始 MedQA & think_csv
    df_data = pd.read_parquet(args.input_parquet)
    if "qid" not in df_data.columns:
        df_data["qid"] = df_data.index.astype(str)
    df_data["qid"] = df_data["qid"].astype(str)

    df_think = pd.read_csv(args.think_csv)
    df_think["qid"] = df_think["qid"].astype(str)

    if "think_text" not in df_think.columns:
        raise ValueError("think_csv 里没有 'think_text' 列，请确认上一步脚本已写入。")

    # 只取需要的列
    cols_need = ["qid", "sent1", "sent2", "ending0", "ending1", "ending2", "ending3", "answer_idx"]
    for c in cols_need:
        if c not in df_data.columns:
            raise ValueError(f"input_parquet 中缺少列 {c!r}")

    df_merged = pd.merge(
        df_think,
        df_data[cols_need],
        on="qid",
        how="inner",
        suffixes=("", "_data"),
    )

    print(f"[INIT] merged rows={len(df_merged)} (有 think_text 且在原数据集中找得到)", flush=True)

    # 2) 准备 vLLM
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
    )

    # 这里是“生成 COT 长尾”的设置：
    # - 不再手动加 <final_answer>
    # - 不用 stop=["</final_answer>"]，让模型自由输出，直到 max_tokens / EOS
    sampling_params = SamplingParams(
        temperature=float(args.temp),
        top_p=float(args.top_p),
        repetition_penalty=float(args.rep),
        max_tokens=int(args.max_tokens),
        stop=[],          # 不人为截断在 </final_answer>
        ignore_eos=True,  # 和你在线脚本一样，主要靠 max_tokens 控长度
    )

    # 3) 组 batch prompt
    prompts = []
    qids = []
    golds = []
    thinks_raw = []
    thinks_clean = []

    for _, row in df_merged.iterrows():
        qid = row["qid"]
        q = f"{row['sent1']}{row['sent2']}"
        ch = {
            "A": row["ending0"],
            "B": row["ending1"],
            "C": row["ending2"],
            "D": row["ending3"],
        }
        gold = row.get("answer_idx", None)
        think_raw = row["think_text"] if isinstance(row["think_text"], str) else ""
        think_cln = clean_think_text(think_raw)

        prompt = build_prompt_with_think(q, ch, think_cln)
        prompts.append(prompt)
        qids.append(qid)
        golds.append(gold)
        thinks_raw.append(think_raw)
        thinks_clean.append(think_cln)

    N = len(prompts)
    print(f"[GEN] samples={N} | batch_size={args.batch_size}", flush=True)

    # 4) 同步批量生成
    rows = []
    total_gen_tokens = 0

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        batch_prompts = prompts[start:end]
        batch_qids = qids[start:end]
        batch_golds = golds[start:end]
        batch_thinks_raw = thinks_raw[start:end]
        batch_thinks_clean = thinks_clean[start:end]

        print(f"[BATCH] {start}-{end} / {N}", flush=True)

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"[ERROR] vLLM batch generate failed: {e}", flush=True)
            for qid, gold, think_raw, think_cln in zip(
                batch_qids, batch_golds, batch_thinks_raw, batch_thinks_clean
            ):
                rows.append(
                    dict(
                        qid=qid,
                        think_text_raw=think_raw,
                        think_text_clean=think_cln,
                        cot_tail="",
                        full_text="",
                        gold=gold,
                        gen_tokens=None,
                        error=str(e),
                    )
                )
            continue

        for out, qid, gold, think_raw, think_cln in zip(
            outputs, batch_qids, batch_golds, batch_thinks_raw, batch_thinks_clean
        ):
            try:
                # 生成的是 </think> 后面的整段 COT（可能带 final_answer，也可能不带）
                tail = out.outputs[0].text if out.outputs else ""
                gen_tok = len(out.outputs[0].token_ids) if out.outputs else None
                total_gen_tokens += (gen_tok or 0)

                # full_text = 固定的 <think>clean</think> + tail（方便之后统一 eval）
                full_text = f"<think> {think_cln} </think>{tail}"

                rows.append(
                    dict(
                        qid=qid,
                        think_text_raw=think_raw,
                        think_text_clean=think_cln,
                        cot_tail=tail,
                        full_text=full_text,
                        gold=gold,
                        gen_tokens=gen_tok,
                        error="",
                    )
                )
            except Exception as e:
                rows.append(
                    dict(
                        qid=qid,
                        think_text_raw=think_raw,
                        think_text_clean=think_cln,
                        cot_tail="",
                        full_text="",
                        gold=gold,
                        gen_tokens=None,
                        error=f"postprocess_error:{e}",
                    )
                )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv} rows={len(out)}", flush=True)

    # 5) 如果你仍然想看准确率：从 full_text 里找 <final_answer>（如果模型自己写了的话）
    out["pred_letter"] = out["full_text"].map(pick_letter)
    out["gold_letter"] = out["gold"].map(map_gold)

    valid_mask = out["gold_letter"].notna()
    if valid_mask.any():
        acc = (
            out.loc[valid_mask, "pred_letter"]
            == out.loc[valid_mask, "gold_letter"]
        ).mean()
    else:
        acc = float("nan")

    avg_gen_tok = pd.to_numeric(out["gen_tokens"], errors="coerce").dropna().mean()
    elapsed = time.time() - t0

    print(
        f"[SUMMARY] think 固定 + 继续生成 COT 策略 | ACC(若有final)= {acc*100:.2f}% "
        f"| avg_gen_tokens={avg_gen_tok:.1f} "
        f"| elapsed={elapsed:.1f}s "
        f"| tok/s={total_gen_tokens/elapsed if elapsed>0 else 0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
"""
CUDA_VISIBLE_DEVICES=0 python inference_cot.py \
  --input_parquet /home/jwang/Project/qwen3_output/data/MedQA/MedQA-USMLE-4-options-parquet/test.parquet \
  --think_csv online_earlystop_results_sync_es.csv \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --temp 0.6 \
  --top_p 0.95 \
  --rep 1.2 \
  --max_tokens 4000 \
  --batch_size 64 \
  --gpu_util 0.8 \
  --max_model_len 10000 \
  --max_num_seqs 128 \
  --trust_remote_code \
  --out_csv think2cot_results.csv

"""