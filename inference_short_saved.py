#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, argparse, asyncio, json
import numpy as np
import pandas as pd
import joblib

# ——强制使用 v1 与多进程 spawn（和你服务端补丁一致）——
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # 进一步禁用 torch.compile

# 引擎导入（优先 v1；若你本地做了别名也兼容）
try:
    from vllm.v1.engine.async_llm import AsyncLLM as Engine
except Exception:
    from vllm.engine.async_llm_engine import AsyncLLMEngine as Engine  # 兼容别名
from vllm.engine.arg_utils import AsyncEngineArgs

# ---------- 提取 <final_answer> 的字母 ----------
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


# ---------- 构造提示 ----------
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


def build_base_prompt(question, choices):
    # 按约定：以 assistant\n<think> 开场
    return (
        f"<system>{SYS}</system>\nuser\n"
        f"{build_user(question, choices)}\nassistant\n<think> "
    )


# ---------- 用 joblib 模型构造 classifier_callable ----------
def make_classifier_callable(joblib_pack):
    """
    引擎在每次 probe 后会把“在线特征字典 feats_dict”传进来（键名与 feats_order 对齐）。
    这里把它排成 joblib 模型需要的顺序，输出 (prob, letter_hint)。
    - prob: 模型的正类概率
    - letter_hint: 用 feats_dict['cum_top'] 作为字母提示（最终仍以 probe 正则到的字母为准）
    """
    clf = joblib_pack["model"]
    feats_order = joblib_pack["feats"]

    def _call(feats_dict):
        row = [float(feats_dict.get(k, 0.0)) for k in feats_order]
        X = np.asarray([row], dtype=float)
        if hasattr(clf, "predict_proba"):
            prob = float(clf.predict_proba(X)[0, 1])
        else:
            prob = float(clf.predict(X))
        letter_hint = feats_dict.get("cum_top", None)
        if not (isinstance(letter_hint, str) and letter_hint in "ABCD"):
            letter_hint = None
        return prob, letter_hint

    return _call


# ---------- 辅助：把 numpy 等转成可 JSON 序列化 ----------
def to_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


# ---------- 主流程 ----------
async def amain(args):
    t0 = time.time()

    # ============ 分类器加载逻辑 ============
    # - 若传了 cls_model_path：无论 enable_early_stop 与否都加载，并传给引擎用于 logging/早停
    # - 若没传 cls_model_path：
    #     * enable_early_stop=True -> 报错（早停必须有分类器）
    #     * enable_early_stop=False -> 不加载分类器（纯主流 + probe，无 classifier prob）
    clf_callable = None
    if args.cls_model_path:
        clf_pack = joblib.load(args.cls_model_path)
        clf_callable = make_classifier_callable(clf_pack)
    elif args.enable_early_stop:
        raise ValueError(
            "enable_early_stop=True 但未提供 --cls_model_path；"
            "早停模式必须加载分类器。"
        )

    # 启动 vLLM v1 异步引擎（重要：开启 prefix cache）
    eng_args = AsyncEngineArgs(
        model=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.engine_max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=args.max_model_len,
        disable_log_requests=True,
        disable_log_stats=True,
        enforce_eager=True,
    )
    engine = Engine.from_engine_args(eng_args)

    # THINK 采样参数（主序列）
    think_params = dict(
        temperature=float(args.think_temp),
        top_p=float(args.think_top_p),
        repetition_penalty=float(args.think_rep),
        max_tokens=int(args.think_max_tokens),
        # 对 MedQA closeQA，我们仍然依赖 <final_answer> 硬停 + max_tokens
        # 保留 ignore_eos=True / stop=[] 的旧设定
        ignore_eos=True,
        stop=[],
        stop_token_ids=[],
    )

    # 客户端并发限流：不超过引擎 max_num_seqs
    client_conc = min(int(args.max_concurrency),
                      int(args.engine_max_num_seqs))
    sem = asyncio.Semaphore(client_conc)

    async def _one_job(idx: int, qid: str, gold, base_prompt: str):
        """
        返回一个四元组：
        (idx, qid, gold, result_dict)
        """
        async with sem:
            try:
                out = await engine.generate_with_checks(
                    prompt=base_prompt,
                    sampling_params_main=think_params,
                    probe_max_steps=int(args.probe_max),
                    check_interval=int(args.token_step),
                    topk=int(args.topk),
                    classifier_callable=clf_callable,
                    threshold=float(args.threshold),
                    request_id=qid,
                    enable_early_stop=args.enable_early_stop,
                    qa_mode="closeqa",
                )
                return idx, qid, gold, out
            except Exception as e:
                # 避免异常把 as_completed 流程打断
                err = {
                    "final_text": "",
                    "final_cause": f"error:{e}",
                    "step_tokens": None,
                    "probe_prob": None,
                    "probe_records": [],
                }
                return idx, qid, gold, err

    # 如果要保存 probe 记录，提前打开文件句柄
    probe_fh = None
    if args.probe_jsonl:
        probe_path = args.probe_jsonl
        os.makedirs(os.path.dirname(probe_path) or ".", exist_ok=True)
        probe_fh = open(probe_path, "w", encoding="utf-8")

    try:
        # ===== 读数据（MedQA parquet）=====
        df = pd.read_parquet(args.input_parquet)
        df["qid"] = df["qid"].astype(str)

        jobs = []
        for i, (_, row) in enumerate(df.iterrows()):
            qid = row["qid"]
            q = f"{row['sent1']}{row['sent2']}"
            ch = {
                "A": row["ending0"],
                "B": row["ending1"],
                "C": row["ending2"],
                "D": row["ending3"],
            }
            gold = row.get("answer_idx", None)
            base_prompt = build_base_prompt(q, ch)
            jobs.append((i, qid, gold, base_prompt))

        N = len(jobs)
        print(
            f"[INIT] samples={N} | client_concurrency={client_conc} | "
            f"engine_max_num_seqs={args.engine_max_num_seqs}",
            flush=True,
        )

        # ===== 启动任务，as_completed 顺序消费 =====
        tasks = [
            asyncio.create_task(_one_job(i, qid, gold, bp))
            for (i, qid, gold, bp) in jobs
        ]

        results = [None] * N
        done = 0

        def _should_print(d, total):
            step = max(1, total // 20)  # 每 ~5% 打一次
            return (d == 1) or (d % step == 0) or (d == total)

        for fut in asyncio.as_completed(tasks):
            idx, qid, gold, res = await fut
            results[idx] = res
            done += 1

            if _should_print(done, N):
                elapsed = time.time() - t0
                rate = (done / elapsed) if elapsed > 0 else 0.0
                remain = N - done
                eta = (remain / rate) if rate > 0 else float("inf")
                print(
                    f"[PROGRESS] {done}/{N} ({done / N:.1%}) | "
                    f"{rate:.2f} samp/s | ETA={eta:.1f}s",
                    flush=True,
                )

        # ===== 写结果 =====
        rows = []
        for (i, qid, gold, _), res in zip(jobs, results):
            res = res or {}
            final_text = res.get("final_text") or res.get("text") or ""
            final_cause = res.get("final_cause") or res.get("finish_reason") or ""
            step_tokens = (
                res.get("step_tokens")
                or res.get("num_output_tokens")
                or res.get("tokens")
                or None
            )
            prob_match = res.get("probe_prob") or res.get("prob") or None

            rows.append(
                dict(
                    qid=qid,
                    final_text=final_text,
                    final_cause=final_cause,
                    step_tokens=step_tokens,
                    probe_prob=prob_match,
                    gold=gold,
                )
            )

            # 保存 probe 记录
            if probe_fh is not None:
                probe_records = res.get("probe_records") or []
                for j, rec in enumerate(probe_records):
                    rec_safe = to_serializable(rec)
                    line = {
                        "qid": qid,
                        "idx": int(i),
                        "probe_idx": int(j),
                        "final_cause": final_cause,
                        "step_tokens_final": step_tokens,
                        "probe_record": rec_safe,
                    }
                    probe_fh.write(json.dumps(line, ensure_ascii=False) + "\n")

        out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"[WRITE] {args.out_csv} rows={len(out)}")

        # 简要评估（closeQA）
        out["pred_letter"] = out["final_text"].map(pick_letter)
        out["gold_letter"] = out["gold"].map(map_gold)
        acc = (out["pred_letter"] == out["gold_letter"]).mean()
        avg_tok = pd.to_numeric(out["step_tokens"], errors="coerce").dropna().mean()
        mode_str = "EARLY-STOP" if args.enable_early_stop else "NO-STOP(LOG-ONLY)"
        print(
            f"[SUMMARY] MODE={mode_str} | ACC={acc*100:.2f}% | "
            f"avg_stop_tokens={avg_tok:.1f} | elapsed={time.time()-t0:.1f}s"
        )

    finally:
        if probe_fh is not None:
            probe_fh.close()
        engine.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp", type=int, default=1)

    # 不再强制 required；是否加载由逻辑 + enable_early_stop 决定
    ap.add_argument("--cls_model_path", default="", help="Path to joblib classifier.")

    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--token_step", type=int, default=50)
    ap.add_argument("--probe_max", type=int, default=10)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--think_temp", type=float, default=0.6)
    ap.add_argument("--think_top_p", type=float, default=0.95)
    ap.add_argument("--think_rep", type=float, default=1.2)
    ap.add_argument("--out_csv", default="online_earlystop_results_sync.csv")

    # 并发/引擎限制
    ap.add_argument("--max_concurrency", type=int, default=64)
    ap.add_argument("--engine_max_num_seqs", type=int, default=128)
    ap.add_argument("--gpu_util", type=float, default=0.8)
    ap.add_argument("--max_batched_tokens", type=int, default=8000)
    ap.add_argument("--max_model_len", type=int, default=8000)
    ap.add_argument("--think_max_tokens", type=int, default=8000)

    # 保存 probe logprobs+feats 的 JSONL 路径
    ap.add_argument(
        "--probe_jsonl",
        default="online_earlystop_probe_records.jsonl",
        help="Path to save per-probe logprobs & features in JSONL format.",
    )

    # 新增：是否真正启用 classifier 早停
    ap.add_argument(
        "--enable_early_stop",
        action="store_true",
        help="Enable classifier-based early stopping (requires --cls_model_path).",
    )

    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()

"""
示例：

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# 仅记录 probe，不早停（可以不传 cls_model_path，也可以传用来观察 prob）
CUDA_VISIBLE_DEVICES=1 python inference_short_saved.py \
  --input_parquet /home/jwang/Project/qwen3_output/data/MedQA/MedQA-USMLE-4-options-parquet/train.parquet \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --threshold 0.95 \
  --token_step 50 \
  --probe_max 10 \
  --topk 20 \
  --think_temp 0.6 \
  --think_top_p 0.95 \
  --think_rep 1.2 \
  --out_csv online_earlystop_results_sync_train.csv \
  --probe_jsonl online_earlystop_probe_records_train.jsonl

# 启用真实 early-stop（必须提供 cls_model_path）
CUDA_VISIBLE_DEVICES=0 python inference_short_saved.py \
  --input_parquet /home/jwang/Project/qwen3_output/data/MedQA/MedQA-USMLE-4-options-parquet/test.parquet \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --cls_model_path model_classifier/early_stop_cls.joblib \
  --threshold 0.95 \
  --token_step 50 \
  --probe_max 10 \
  --topk 20 \
  --think_temp 0.6 \
  --think_top_p 0.95 \
  --think_rep 1.2 \
  --out_csv online_earlystop_results_sync_es.csv \
  --probe_jsonl online_earlystop_probe_records_es.jsonl \
  --enable_early_stop
"""
