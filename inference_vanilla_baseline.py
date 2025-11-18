#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, argparse
import numpy as np
import pandas as pd

# 额外依赖：用于 MATH grader
import sympy
from sympy.parsing import sympy_parser
from pylatexenc import latex2text

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
# MATH 提取 \boxed{...} & 全套 grader（沿用 async 脚本）
# ============================================================

def last_boxed_only_string(string: str):
    """Return the substring '\\boxed{...}' of the last boxed, including braces (brace-balanced)."""
    if not isinstance(string, str):
        return None
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        ch = string[i]
        if ch == "{":
            num_left_braces_open += 1
        if ch == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(s: str) -> str:
    """Extract inside of the last \\boxed{...} from a LaTeX string (robust)."""
    if not isinstance(s, str):
        return ""
    chunk = last_boxed_only_string(s)
    return remove_boxed(chunk) if chunk is not None else ""


def canon_simple(s: str) -> str:
    """A very light canonicalization for cheap equality checks."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = t.replace("–", "-").replace("−", "-")
    t = (t.replace("\\,", "").replace("\\;", "").replace("\\!", "")
           .replace("\\left", "").replace("\\right", ""))
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = t.replace("tfrac", "frac").replace("dfrac", "frac")
    t = t.replace("^{\\circ}", "").replace("^\\circ", "")
    t = t.replace(" ", "").strip("$")
    return t


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def mathd_normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr and substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a); b = int(b)
            assert string == "{}/{}".format(a, b)
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
        except Exception:
            return string

    def _remove_right_units(string):
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            return splits[0]
        return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split and split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    string = (string or "").replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # mixed numbers
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = (expr.replace("√", "sqrt")
                 .replace("π", "pi")
                 .replace("∞", "inf")
                 .replace("∪", "U")
                 .replace("·", "*")
                 .replace("×", "*"))
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num); return True
    except Exception:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)
    return step


def _normalize(expr: str) -> str:
    if expr is None:
        return None
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = (expr.replace("\\%", "%")
                .replace("\\$", "$")
                .replace("$", "")
                .replace("%", "")
                .replace(" or ", " , ")
                .replace(" and ", " , "))
    for unit in [
        "degree","cm","centimeter","meter","mile","second","minute","hour","day","week",
        "month","year","foot","feet","inch","yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def should_allow_eval(expr: str):
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            return simplified == 0
    except Exception:
        pass
    return False


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    given = _normalize(given_answer)
    gold  = _normalize(ground_truth)
    if gold is None:
        return False
    if gold == given:
        return True
    if not given:
        return False
    gold_elems = split_tuple(gold)
    given_elems = split_tuple(given)
    if len(gold_elems) > 1 and (gold[0] != given[0] or gold[-1] != given[-1]):
        return False
    if len(gold_elems) != len(given_elems):
        return False
    for ge, pe in zip(gold_elems, given_elems):
        if _is_frac(ge) and _is_frac(pe):
            ok = (ge == pe)
        elif _str_is_int(ge) != _str_is_int(pe):
            ok = False
        else:
            ok = are_equal_under_sympy(ge, pe)
        if not ok:
            return False
    return True


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    gold = mathd_normalize_answer(ground_truth)
    pred = mathd_normalize_answer(given_answer)
    return (gold == pred)


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """final grade (shape → MathD → SymPy)"""
    if given_answer is None or ground_truth is None:
        return False
    if canon_simple(given_answer) == canon_simple(ground_truth):
        return True
    if grade_answer_mathd(given_answer, ground_truth):
        return True
    return grade_answer_sympy(given_answer, ground_truth)

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
# Prompt 构造：MedQA 多选题（恢复 <final_answer>）
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
    return (
        f"<system>{SYS_MEDQA}</system>\n"
        f"user\n{build_user_medqa(question, choices)}\n"
        "assistant\n<think> "
    )

# ============================================================
# Prompt 构造：MATH-500 openQA 版本（与你 async 脚本一致）
# ============================================================

SYS_MATH = (
    "You are an expert math problem solver.\n"
    "Please solve the problem step by step. "
    "First, write your detailed reasoning inside <think>...</think>. "
    "Then, provide your final answer inside \\boxed{}"
)

def build_user_math(problem: str) -> str:
    return (
        "Here is a math problem:\n"
        f"{problem}\n\n"
        "Please provide detailed reasoning inside <think>...</think> and then "
        "output your final answer inside \\boxed{}"
    )

def build_base_prompt_math(problem: str) -> str:
    # 与 generate_with_checks 的约定：以 assistant\n<think> 开场
    return f"<system>{SYS_MATH}</system>\nuser\n{build_user_math(problem)}\nassistant\n<think> "

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
        # gold 在后面用 solution/answer_raw 再重新算，这里先占位
        for _, row in df.iterrows():
            qid = row["qid"]
            problem  = row.get("problem", "")
            solution = row.get("solution", "")
            answer   = row.get("answer", "")

            base_prompt = build_base_prompt_math(problem)
            prompts.append(base_prompt)
            qids.append(qid)

            # gold 先留空；真正 GT 用 solution/answer_raw 再推
            golds.append(None)

            extra_cols.append({
                "unique_id": row.get("unique_id", ""),
                "subject": row.get("subject", ""),
                "level": row.get("level", ""),
                "problem": problem,
                "solution": solution,
                "answer_raw": answer,
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
        # MATH-500: 用和 async early-stop 一致的 EM 计算
        out["gt_from_solution_box"] = out["solution"].apply(extract_boxed_answer)
        out["gt_text"] = out.apply(
            lambda r: (
                r["gt_from_solution_box"]
                if isinstance(r["gt_from_solution_box"], str)
                and len(r["gt_from_solution_box"]) > 0
                else (r["answer_raw"] if isinstance(r["answer_raw"], str) else "")
            ),
            axis=1,
        )

        out["pred_boxed"] = out["final_text"].map(extract_boxed_answer)

        mask_eval = out["gt_text"].astype(str).str.len() > 0
        if mask_eval.any():
            out.loc[mask_eval, "is_correct"] = out.loc[mask_eval].apply(
                lambda r: grade_answer(r["pred_boxed"], r["gt_text"]), axis=1
            )
            em = float(out.loc[mask_eval, "is_correct"].mean())
            print(f"[Eval] Evaluated {int(mask_eval.sum())}/{len(out)} rows | EM = {em:.4f}")
        else:
            out["is_correct"] = None
            print("[Eval] No GT available (neither parsable \\boxed{...} in solution nor `answer`).")

        avg_gen_tok = pd.to_numeric(out["gen_tokens"], errors="coerce").dropna().mean()
        print(
            f"[SUMMARY] (MATH-500) avg_gen_tokens={avg_gen_tok:.1f} "
            f"| elapsed={elapsed:.1f}s "
            f"| tok/s={total_gen_tokens/elapsed if elapsed>0 else 0:.1f}",
            flush=True,
        )

        # 简单 sanity check 表（和 async 版保持风格）
        dbg = out.loc[mask_eval, ["qid", "pred_boxed", "gt_text"]].head(12).copy()
        if len(dbg) > 0:
            dbg["pred_simple"] = dbg["pred_boxed"].map(canon_simple)
            dbg["gt_simple"]   = dbg["gt_text"].map(canon_simple)
            dbg["eq_simple"]   = (dbg["pred_simple"] == dbg["gt_simple"])
            dbg["eq_mathd"]    = dbg.apply(
                lambda r: grade_answer_mathd(r["pred_boxed"], r["gt_text"]), axis=1
            )
            dbg["eq_sympy"]    = dbg.apply(
                lambda r: grade_answer_sympy(r["pred_boxed"], r["gt_text"]), axis=1
            )
            dbg["is_correct"]  = dbg.apply(
                lambda r: grade_answer(r["pred_boxed"], r["gt_text"]), axis=1
            )
            print("Sanity check (first 12 with GT):")
            print(dbg)

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
  --max_tokens 20000 \
  --batch_size 32 \
  --gpu_util 0.8 \
  --max_model_len 20000 \
  --max_num_seqs 128 \
  --trust_remote_code
"""
