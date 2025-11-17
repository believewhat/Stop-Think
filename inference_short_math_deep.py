#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, argparse, asyncio, json
import numpy as np
import pandas as pd
import joblib

# 额外依赖：用于 MATH / DeepScaleR grader
import sympy
from sympy.parsing import sympy_parser
from pylatexenc import latex2text

# ——强制使用 v1 与多进程 spawn——
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # 禁用 torch.compile

# 引擎导入（优先 v1；若你本地做了别名也兼容）
try:
    from vllm.v1.engine.async_llm import AsyncLLM as Engine
except Exception:
    from vllm.engine.async_llm_engine import AsyncLLMEngine as Engine  # 兼容别名
from vllm.engine.arg_utils import AsyncEngineArgs


# ============================================================
# MATH / DeepScaleR 提取 \boxed{...} & Grader
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
    t = (
        t.replace("\\,", "")
         .replace("\\;", "")
         .replace("\\!", "")
         .replace("\\left", "")
         .replace("\\right", "")
    )
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
    expr = (
        expr.replace("√", "sqrt")
            .replace("π", "pi")
            .replace("∞", "inf")
            .replace("∪", "U")
            .replace("·", "*")
            .replace("×", "*")
    )
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
    expr = (
        expr.replace("\\%", "%")
            .replace("\\$", "$")
            .replace("$", "")
            .replace("%", "")
            .replace(" or ", " , ")
            .replace(" and ", " , ")
    )
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
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
    gold = _normalize(ground_truth)
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


# ======================
# Prompt 构造（openQA 版本：<think> + \boxed{}）
# ======================

SYS = (
    "You are an expert math problem solver.\n"
    "Please solve the problem step by step. "
    "First, write your detailed reasoning inside <think>...</think>. "
    "Then, provide your final answer inside \\boxed{}"
)


def build_user(problem: str) -> str:
    return (
        "Here is a math problem:\n"
        f"{problem}\n\n"
        "Please provide detailed reasoning inside <think>...</think> and then "
        "output your final answer inside \\boxed{}"
    )


def build_base_prompt(problem: str) -> str:
    # 与 generate_with_checks 的约定：以 assistant\n<think> 开场
    return f"<system>{SYS}</system>\nuser\n{build_user(problem)}\nassistant\n<think> "


# ---------- 用 joblib 模型构造 classifier_callable ----------
def make_classifier_callable(joblib_pack):
    """
    引擎在每次 probe 后会把“在线特征字典 feats_dict”传进来（键名与 feats_order 对齐）。
    这里把它排成 joblib 模型需要的顺序，输出 (prob, letter_hint)。
    openQA 下 letter_hint 可以忽略（返回 None 即可）。
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
        # openQA 没有 ABCD，这里直接给 None
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


# ---------- 新增：加载输入数据，兼容 JSONL 和 JSON 数组 ----------
def load_input_df(path: str) -> pd.DataFrame:
    """
    - 若文件以 '['（忽略前导空白）开头，则视为 JSON 数组（如 deepscaler.json），用 pd.read_json(path)
    - 否则视为 JSONL（如 MATH-500/test.jsonl），用 pd.read_json(path, lines=True)
    """
    first_char = None
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break

    if first_char == "[":
        # JSON 数组: [{"problem":..., ...}, {...}, ...]
        df = pd.read_json(path)
    else:
        # JSONL: 每行一个 JSON 对象
        df = pd.read_json(path, lines=True)
    return df


# ---------- 主流程 ----------
async def amain(args):
    t0 = time.time()

    # ========== 分类器（可选）==========
    clf_callable = None
    if args.cls_model_path:
        print(f"[CLS] Loading classifier from {args.cls_model_path}", flush=True)
        clf_pack = joblib.load(args.cls_model_path)
        clf_callable = make_classifier_callable(clf_pack)
    else:
        print(
            "[CLS] No --cls_model_path provided; classifier will NOT be loaded.",
            flush=True,
        )

    # 如果启用 early-stop，但没有 classifier，直接报错（和 Engine 的约定一致）
    if args.enable_early_stop and clf_callable is None:
        raise ValueError(
            "enable_early_stop=True 但未提供 --cls_model_path；"
            "若不需要分类器早停，请去掉 --enable_early_stop。"
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
        stop=[],
        stop_token_ids=[],
    )

    client_conc = min(int(args.max_concurrency), int(args.engine_max_num_seqs))
    sem = asyncio.Semaphore(client_conc)

    async def _one_job(idx: int, qid: str, base_prompt: str):
        """
        返回 (idx, qid, result_dict)
        """
        async with sem:
            try:
                out = await engine.generate_with_checks(
                    prompt=base_prompt,
                    sampling_params_main=think_params,
                    probe_max_steps=int(args.probe_max),
                    check_interval=int(args.token_step),
                    topk=int(args.topk),
                    classifier_callable=clf_callable,  # 可能为 None
                    threshold=float(args.threshold),
                    request_id=qid,
                    enable_early_stop=bool(args.enable_early_stop),
                    qa_mode="openqa",  # ★ 关键：openQA 模式（只看 \\boxed{}）
                )
                return idx, qid, out
            except Exception as e:
                err = {
                    "final_text": "",
                    "final_cause": f"error:{e}",
                    "step_tokens": None,
                    "probe_prob": None,
                    "probe_records": [],
                }
                return idx, qid, err

    # 如果要保存 probe 记录，提前打开文件句柄
    probe_fh = None
    if args.probe_jsonl:
        probe_path = args.probe_jsonl
        os.makedirs(os.path.dirname(probe_path) or ".", exist_ok=True)
        probe_fh = open(probe_path, "w", encoding="utf-8")

    try:
        # ===== 读数据: 支持 MATH-500 JSONL & DeepScaleR/deepscaler.json =====
        df = load_input_df(args.input_path)

        total = len(df)
        print(f"[DATA] loaded {total} rows from {args.input_path}", flush=True)

        # ===== 分片：通过 shard_idx / shard_count 把数据切成多份（多 GPU / 多进程）=====
        if args.shard_count > 1:
            if not (0 <= args.shard_idx < args.shard_count):
                raise ValueError(
                    f"Invalid shard_idx={args.shard_idx}, shard_count={args.shard_count}"
                )
            shard_size = (total + args.shard_count - 1) // args.shard_count
            start = args.shard_idx * shard_size
            end = min(start + shard_size, total)
            df = df.iloc[start:end]  # 不 reset_index，保留全局行号用于 qid
            print(
                f"[SHARD] shard_idx={args.shard_idx}/{args.shard_count} | "
                f"rows=[{start}:{end}) -> {len(df)}",
                flush=True,
            )
        else:
            print(f"[SHARD] single shard (all {total} rows).", flush=True)

        # 补全缺失列
        for col in ["problem", "solution", "answer", "subject", "level", "unique_id"]:
            if col not in df.columns:
                df[col] = ""

        # qid：优先用 unique_id，否则用“全局行号字符串”
        if "unique_id" in df.columns and df["unique_id"].notna().all():
            df["qid"] = df["unique_id"].astype(str)
        else:
            df["qid"] = df.index.astype(str)

        jobs = []
        for i, (_, row) in enumerate(df.iterrows()):
            qid = row["qid"]
            problem = row.get("problem", "")
            base_prompt = build_base_prompt(problem)
            jobs.append((i, qid, base_prompt))

        N = len(jobs)
        print(
            f"[INIT] shard_rows={N} | client_concurrency={client_conc} | "
            f"engine_max_num_seqs={args.engine_max_num_seqs}",
            flush=True,
        )

        # ===== 启动任务 =====
        tasks = [
            asyncio.create_task(_one_job(i, qid, bp))
            for (i, qid, bp) in jobs
        ]

        results = [None] * N
        done = 0

        def _should_print(d, total):
            step = max(1, total // 20)
            return (d == 1) or (d % step == 0) or (d == total)

        for fut in asyncio.as_completed(tasks):
            idx, qid, res = await fut
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

        # ===== 写结果 & 评估 =====
        rows = []
        for (i, qid, _), res in zip(jobs, results):
            src = df.iloc[i]
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

            # 提取 boxed 预测
            pred_boxed = extract_boxed_answer(final_text)

            row_out = dict(
                qid=qid,
                unique_id=src.get("unique_id", ""),
                subject=src.get("subject", ""),
                level=src.get("level", ""),
                problem=src.get("problem", ""),
                solution=src.get("solution", ""),
                answer=src.get("answer", ""),
                final_text=final_text,
                final_cause=final_cause,
                step_tokens=step_tokens,
                probe_prob=prob_match,
                pred_boxed=pred_boxed,
            )
            rows.append(row_out)

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

        # ===== EM 评估：GT 优先使用字段 "answer"，其次尝试从 solution 的 \boxed{} 提取 =====
        out["gt_from_solution_box"] = out["solution"].apply(extract_boxed_answer)

        def _get_gt_text(r):
            ans = r.get("answer", "")
            if isinstance(ans, str) and ans.strip():
                return ans.strip()
            box = r.get("gt_from_solution_box", "")
            if isinstance(box, str) and box.strip():
                return box.strip()
            return ""

        out["gt_text"] = out.apply(_get_gt_text, axis=1)

        mask_eval = out["gt_text"].astype(str).str.len() > 0
        if mask_eval.any():
            out.loc[mask_eval, "is_correct"] = out.loc[mask_eval].apply(
                lambda r: grade_answer(r["pred_boxed"], r["gt_text"]), axis=1
            )
            em = float(out.loc[mask_eval, "is_correct"].mean())
            print(
                f"[Eval] Evaluated {int(mask_eval.sum())}/{len(out)} rows | EM = {em:.4f}"
            )
        else:
            out["is_correct"] = None
            print(
                "[Eval] No GT available (no `answer` and no parsable "
                "\\boxed{...} in solution)."
            )

        # 平均 step token
        avg_tok = pd.to_numeric(out["step_tokens"], errors="coerce").dropna().mean()
        print(
            f"[SUMMARY] avg_stop_tokens={avg_tok:.1f} | "
            f"elapsed={time.time()-t0:.1f}s"
        )

        # 简单 sanity check 表
        dbg = out.loc[mask_eval, ["qid", "unique_id", "pred_boxed", "gt_text"]].head(12).copy()
        if len(dbg) > 0:
            dbg["pred_simple"] = dbg["pred_boxed"].map(canon_simple)
            dbg["gt_simple"] = dbg["gt_text"].map(canon_simple)
            dbg["eq_simple"] = (dbg["pred_simple"] == dbg["gt_simple"])
            dbg["eq_mathd"] = dbg.apply(
                lambda r: grade_answer_mathd(r["pred_boxed"], r["gt_text"]), axis=1
            )
            dbg["eq_sympy"] = dbg.apply(
                lambda r: grade_answer_sympy(r["pred_boxed"], r["gt_text"]), axis=1
            )
            dbg["is_correct"] = dbg.apply(
                lambda r: grade_answer(r["pred_boxed"], r["gt_text"]), axis=1
            )
            print("Sanity check (first 12 with GT):")
            print(dbg)

    finally:
        if probe_fh is not None:
            probe_fh.close()
        engine.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_path",
        required=True,
        help="Path to MATH-500 JSONL or DeepScaleR JSON array",
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp", type=int, default=1)

    ap.add_argument(
        "--cls_model_path",
        default=None,
        help="Path to joblib classifier; if omitted, classifier will not be loaded.",
    )
    ap.add_argument(
        "--enable_early_stop",
        action="store_true",
        help="Use classifier-based early stop. Requires --cls_model_path.",
    )
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--token_step", type=int, default=50)
    ap.add_argument(
        "--probe_max",
        type=int,
        default=30,  # openQA 建议 30
        help="Max tokens for each probe generation (openQA).",
    )
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--think_temp", type=float, default=0.6)
    ap.add_argument("--think_top_p", type=float, default=0.95)
    ap.add_argument("--think_rep", type=float, default=1.2)
    ap.add_argument(
        "--out_csv",
        default="online_earlystop_results_math500.csv",
    )
    # 并发/引擎限制
    ap.add_argument("--max_concurrency", type=int, default=64)
    ap.add_argument("--engine_max_num_seqs", type=int, default=128)
    ap.add_argument("--gpu_util", type=float, default=0.8)
    ap.add_argument("--max_batched_tokens", type=int, default=20000)
    ap.add_argument("--max_model_len", type=int, default=20000)
    ap.add_argument("--think_max_tokens", type=int, default=20000)
    ap.add_argument(
        "--probe_jsonl",
        default="online_earlystop_probe_records_math500.jsonl",
        help="Path to save per-probe logprobs & features in JSONL format.",
    )
    # 新增：分片参数，用于多 GPU / 多进程数据并行
    ap.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Shard index (0-based). Use together with --shard_count.",
    )
    ap.add_argument(
        "--shard_count",
        type=int,
        default=1,
        help="Total number of shards (for data-parallel runs).",
    )

    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()

"""
单卡（无分片）示例：

CUDA_VISIBLE_DEVICES=0 python inference_short_math_deep.py \
  --input_path AdaptThink/data/train/deepscaler.json \
  --model_path /data/data_user_alpha/public_models/Qwen3/Qwen3-8B \
  --dtype bfloat16 \
  --tp 1 \
  --threshold 0.95 \
  --token_step 50 \
  --probe_max 30 \
  --topk 20 \
  --out_csv online_earlystop_results_deepscaler.csv \
  --probe_jsonl online_earlystop_probe_records_deepscaler.jsonl

"""
