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

# ============================================================
# 工具：加载数据（自动识别 parquet / jsonl / json）
# ============================================================

def load_df(path: str) -> pd.DataFrame:
    """
    - *.parquet  → MedQA 格式（sent1/sent2/ending0..3/answer_idx）
    - *.jsonl    → MATH-500 JSONL
    - *.json     → MATH-500 JSON array
    """
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if lower.endswith(".json"):
        return pd.read_json(path)
    # 默认尝试 parquet
    return pd.read_parquet(path)

# ============================================================
# 共同：MATH 提取 \boxed{...} & 简单 canonicalization
# ============================================================

_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", flags=re.S)

def extract_boxed_answer(s: str) -> str:
    """从字符串中抽取最后一个 \\boxed{...} 的内容；没有则返回空串。"""
    if not isinstance(s, str):
        return ""
    m = _BOXED_RE.findall(s)
    return m[-1].strip() if m else ""

def canon_simple(s: str) -> str:
    """轻量 canonicalization，用于便宜的等价判断（不走 sympy）。"""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = t.replace("–", "-").replace("−", "-")
    t = (t.replace("\\,", "")
           .replace("\\;", "")
           .replace("\\!", "")
           .replace("\\left", "")
           .replace("\\right", ""))
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = t.replace("tfrac", "frac").replace("dfrac", "frac")
    t = t.replace("^{\\circ}", "").replace("^\\circ", "")
    t = t.replace(" ", "").strip("$")
    return t

def grade_math_simple(pred: str, gold: str) -> bool:
    """简单 EM：canonicalization 后字符串相等。"""
    if gold is None:
        return False
    cp = canon_simple(pred)
    cg = canon_simple(gold)
    if not cg:
        return False
    return cp == cg

# ============================================================
# MedQA: 提取 <final_answer> 的字母
# ============================================================

_FA_BLOCK = re.compile(
    r"<\s*final_answer\s*>(.*?)<\s*/\s*final_answer\s*>",
    flags=re.I | re.S
)
_PAT_STRONG = re.compile(
    r"(?i)(?:^|\b(?:answer|ans)\s*[:\-]?\s*)([ABCD])(?:\s*[:\.\)]|\s|$)"
)
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
    m = re.search(r"[A-D]", s, flags=re.I)
    return m.group(0).upper() if m else None

# ============================================================
# Prompt 构造：MedQA 多选题
# ============================================================

TASK = "Please output a single best option from Choices as your final answer."
CUST = (
    "Solve them in a step-by-step fashion, starting by summarizing the available "
    "information. Please output your all reasoning into <think> thoughts </think>."
)
SYS_MEDQA  = (
    "You are a medical professional. Your job is to assist with medical questions.\n"
    f"{TASK}\n{CUST}\n"
    "Please wrap your final answer in <final_answer> your answer </final_answer>."
)

def build_user_medqa(question, choices):
    return (
        f"**Question**:\n{question}\n**Choices**:\n"
        f"A: {choices['A']}\nB: {choices['B']}\nC: {choices['C']}\nD: {choices['D']}\n"
        "Now based on the above reasoning, you must give the correct option in the form "
        "<final_answer>your answer(A,B,C,D,only one letter)</final_answer>. You must give an option."
    )

def build_base_prompt_medqa(question, choices):
    # 与你之前 online early-stop 的 system/user/assistant 模式保持一致
    return (
        f"<system>{SYS_MEDQA}</system>\n"
        f"user\n{build_user_medqa(question, choices)}\n"
        "assistant\n<think> "
    )

# ============================================================
# Prompt 构造：MATH-500 openQA 版本
# ============================================================

SYS_MATH = (
    "You are an expert math problem solver.\n"
    "Please solve the problem step by step. "
    "First, write your detailed reasoning inside <think>...</think>. "
    "Then, provide your final answer inside \\boxed{}."
)

def build_user_math(problem: str) -> str:
    return (
        "Here is a math problem:\n"
        f"{problem}\n\n"
        "Please provide detailed reasoning inside <think>...</think> and then "
        "output your final answer inside \\boxed{}."
    )

def build_base_prompt_math(problem: str) -> str:
    # 与之前 math early-stop 版本对齐：assistant\n<think> 开场
    return (
        f"<system>{SYS_MATH}</system>\n"
        f"user\n{build_user_math(problem)}\n"
        "assistant\n<think> "
    )

# ============================================================
# 主流程：vanilla 同步推理（支持 MedQA + MATH-500）
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_parquet",
        required=True,
        help="MedQA: *.parquet | MATH-500: *.jsonl / *.json"
    )
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

    # ---------- 读取数据（自动识别格式） ----------
    df = load_df(args.input_parquet)

    # 判定数据类型
    if {"sent1", "sent2", "ending0", "ending1", "ending2", "ending3"}.issubset(df.columns):
        dataset_type = "medqa"
    elif "problem" in df.columns:
        dataset_type = "math500"
    else:
        raise ValueError(
            "Unsupported dataset format: expected MedQA (sent1/sent2/ending0..3) "
            "or MATH-500 (problem/solution[/answer])."
        )

    print(f"[INFO] Detected dataset_type = {dataset_type}", flush=True)

    # 给所有样本一个 qid
    if "qid" in df.columns:
        df["qid"] = df["qid"].astype(str)
    elif "unique_id" in df.columns:
        df["qid"] = df["unique_id"].astype(str)
    else:
        df["qid"] = df.index.astype(str)

    prompts = []
    qids = []
    golds = []
    extra_cols = []  # 用来存一些原始字段，方便写回 csv（主要给 math500）

    if dataset_type == "medqa":
        # ---------- MedQA: 组 prompt ----------
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

            base_prompt = build_base_prompt_medqa(q, ch)
            prompts.append(base_prompt)
            qids.append(qid)
            golds.append(gold)
            extra_cols.append({})  # 占位
    else:
        # ---------- MATH-500: 组 prompt ----------
        # 约定：gold = 从 solution 的最后一个 \boxed{} 抽；若没有，则用 answer 字段
        for _, row in df.iterrows():
            qid = row["qid"]
            problem  = row.get("problem", "")
            solution = row.get("solution", "")
            answer   = row.get("answer", "")

            base_prompt = build_base_prompt_math(problem)
            prompts.append(base_prompt)
            qids.append(qid)

            gt_from_solution = extract_boxed_answer(solution)
            gt_text = gt_from_solution if gt_from_solution else (answer if isinstance(answer, str) else "")
            golds.append(gt_text)

            extra_cols.append({
                "problem": problem,
                "solution": solution,
                "answer_raw": answer,
                "gt_text": gt_text,
            })

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
        batch_extra   = extra_cols[start:end]

        print(f"[BATCH] {start}-{end} / {N}", flush=True)

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"[ERROR] vLLM batch generate failed: {e}", flush=True)
            # 每个样本写一个空结果，避免中断
            for qid, gold, extra in zip(batch_qids, batch_golds, batch_extra):
                row_out = dict(
                    qid=qid,
                    final_text="",
                    gold=gold,
                    gen_tokens=None,
                    error=str(e),
                )
                row_out.update(extra)
                rows.append(row_out)
            continue

        for out, qid, gold, extra in zip(outputs, batch_qids, batch_golds, batch_extra):
            try:
                # vLLM 每个请求通常只有一个 output
                text = out.outputs[0].text if out.outputs else ""
                gen_tok = len(out.outputs[0].token_ids) if out.outputs else None
                total_gen_tokens += (gen_tok or 0)
                row_out = dict(
                    qid=qid,
                    final_text=text,
                    gold=gold,
                    gen_tokens=gen_tok,
                    error="",
                )
                row_out.update(extra)
                rows.append(row_out)
            except Exception as e:
                row_out = dict(
                    qid=qid,
                    final_text="",
                    gold=gold,
                    gen_tokens=None,
                    error=f"postprocess_error:{e}",
                )
                row_out.update(extra)
                rows.append(row_out)

    elapsed = time.time() - t0
    print(f"[TIME] total_elapsed={elapsed:.1f}s", flush=True)

    # ---------- 写结果 ----------
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv} rows={len(out)}", flush=True)

    # ---------- 统计指标 ----------
    if dataset_type == "medqa":
        out["pred_letter"] = out["final_text"].map(pick_letter)
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

        print(
            f"[SUMMARY] (MedQA) ACC={acc*100:.2f}% "
            f"| avg_gen_tokens={avg_gen_tok:.1f} "
            f"| elapsed={elapsed:.1f}s "
            f"| tok/s={total_gen_tokens/elapsed if elapsed>0 else 0:.1f}",
            flush=True,
        )
    else:
        # MATH-500: openQA EM
        out["pred_boxed"] = out["final_text"].map(extract_boxed_answer)
        # gold 已经是 canonical GT（来自 solution 的 boxed 或 answer）
        out["gt_text"] = out["gold"].astype(str).fillna("")

        mask_eval = out["gt_text"].astype(str).str.len() > 0
        if mask_eval.any():
            out.loc[mask_eval, "is_correct"] = out.loc[mask_eval].apply(
                lambda r: grade_math_simple(r["pred_boxed"], r["gt_text"]), axis=1
            )
            em = float(out.loc[mask_eval, "is_correct"].mean())
        else:
            out["is_correct"] = None
            em = float("nan")

        avg_gen_tok = pd.to_numeric(out["gen_tokens"], errors="coerce").dropna().mean()

        print(
            f"[SUMMARY] (MATH-500) EM={em*100:.2f}% "
            f"| avg_gen_tokens={avg_gen_tok:.1f} "
            f"| elapsed={elapsed:.1f}s "
            f"| tok/s={total_gen_tokens/elapsed if elapsed>0 else 0:.1f}",
            flush=True,
        )

if __name__ == "__main__":
    main()

"""
MedQA 示例：

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=1 python -u inference_vanilla_baseline.py \
  --input_parquet /home/jwang/Project/qwen3_output/data/MedQA/MedQA-USMLE-4-options-parquet/test.parquet \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --out_csv vanilla_inference_results_medqa.csv \
  --temp 0.6 \
  --top_p 0.95 \
  --rep 1.2 \
  --max_tokens 10000 \
  --batch_size 64 \
  --gpu_util 0.8 \
  --max_model_len 10000 \
  --max_num_seqs 128 \
  --trust_remote_code

MATH-500 示例（JSONL）：

CUDA_VISIBLE_DEVICES=2 python -u inference_vanilla_baseline.py \
  --input_parquet /home/jwang/Project/qwen3_output/data/MATH-500/test.jsonl \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --out_csv vanilla_inference_results_math500.csv \
  --temp 0.6 \
  --top_p 0.95 \
  --rep 1.2 \
  --max_tokens 8000 \
  --batch_size 32 \
  --gpu_util 0.8 \
  --max_model_len 8000 \
  --max_num_seqs 128 \
  --trust_remote_code
"""
