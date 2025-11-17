#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train & evaluate an early-stop classifier directly FROM probe JSONL logs
(for MATH / openQA, 也兼容 closeQA)。

标签定义（通用）：
- 对于同一个 qid：
    * 找到该 qid 所有 probe 中 step_tokens 最大的那一行（最后一个 step），
      记它的“答案字符串” probe_answer 为 final_letter_full。
        - closeQA: probe_answer = 选项字母 A/B/C/D
        - openQA(MATH): probe_answer = probe_text（即当前答案字符串，如 \\sqrt{3}/2 ）
    * 对这个 qid 的每一行 probe：
        y_match_final = 1  当且仅当
          probe_has_answer == True 且
          final_letter_full 非空 且
          probe_answer == final_letter_full
      否则 y_match_final = 0
"""

import os
import json
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

LETTER_IDX = {0: "A", 1: "B", 2: "C", 3: "D"}


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        return s in {"1", "true", "t", "yes", "y"}
    return False


def load_steps_from_jsonl(json_path: str) -> pd.DataFrame:
    """从 probe JSONL 构造 steps DataFrame。"""
    if not os.path.exists(json_path):
        raise ValueError(f"[load_steps_from_jsonl] file not found: {json_path}")

    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            qid = str(obj.get("qid", "")).strip()
            if not qid:
                continue

            rec = obj.get("probe_record") or {}
            feats = rec.get("feats") or {}
            slot = rec.get("slot") or {}
            qa_mode = str(rec.get("qa_mode", "closeqa")).lower()

            # step_tokens: 优先 feats["step"]，否则 probe_record["step"]
            step_val = feats.get("step", rec.get("step", 0))
            try:
                step_tokens = int(step_val)
            except Exception:
                step_tokens = 0

            # ===== 定义“当前答案字符串” probe_answer =====
            if qa_mode == "openqa":
                # MATH / openQA：用 probe_text 作为答案字符串
                ans_str = rec.get("probe_text") or ""
                ans_str = str(ans_str).strip()
                probe_answer = ans_str
                probe_has_answer = len(ans_str) > 0
            else:
                # closeQA：slot["probe_letter"] = A/B/C/D
                pl = slot.get("probe_letter") or ""
                pl = str(pl).strip().upper()
                probe_answer = pl
                probe_has_answer = pl in "ABCD"

            base = {
                "qid": qid,
                "qa_mode": qa_mode,
                "step_tokens": step_tokens,
                "probe_answer": probe_answer,
                "probe_has_answer": int(probe_has_answer),
            }

            # 展开 feats
            for k, v in feats.items():
                if isinstance(v, (np.floating, float)):
                    base[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    base[k] = int(v)
                elif isinstance(v, bool):
                    base[k] = int(v)
                else:
                    base[k] = v

            rows.append(base)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"[load_steps_from_jsonl] no valid rows parsed from {json_path}")

    df["qid"] = df["qid"].astype(str)
    df["step_tokens"] = pd.to_numeric(df["step_tokens"], errors="coerce").fillna(0).astype(int)
    df["probe_has_answer"] = df["probe_has_answer"].apply(to_bool).astype(int)
    df["probe_answer"] = df["probe_answer"].fillna("").astype(str)

    # ===== 构造 final_answer_full（同 qid 最后一步的答案字符串）=====
    if "qa_mode" in df.columns:
        mask_open = df["qa_mode"].astype(str).str.lower().eq("openqa")
        mask_close = ~mask_open

        valid_open = mask_open & (df["probe_answer"].str.len() > 0)
        valid_close = mask_close & df["probe_answer"].str.upper().isin(list("ABCD"))
        valid = df[valid_open | valid_close].copy()
    else:
        valid = df[df["probe_answer"].str.upper().isin(list("ABCD"))].copy()

    if not valid.empty:
        last_rows = (
            valid.sort_values(["qid", "step_tokens"])
            .groupby("qid", as_index=False).tail(1)
        )
        final_map = last_rows[["qid", "probe_answer"]].rename(
            columns={"probe_answer": "final_answer_full"}
        )
        df = df.merge(final_map, on="qid", how="left")
    else:
        df["final_answer_full"] = ""

    df["final_answer_full"] = df["final_answer_full"].fillna("").astype(str)

    # ===== y_match_final：是否与最终答案字符串一致 =====
    df["y_match_final"] = (
        (df["probe_has_answer"] == 1)
        & (df["final_answer_full"].str.len() > 0)
        & (df["probe_answer"] == df["final_answer_full"])
    ).astype(int)

    return df


def load_feature_list(json_path: Optional[str], columns: List[str]) -> List[str]:
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            feats = json.load(f)
        feats = [c for c in feats if c in columns]
        if len(feats) == 0:
            raise ValueError("feature_columns.json yielded empty intersection with columns.")
        return feats

    # 优先挑你 MATH 用到的特征，如果存在就用
    default_feats = [
        # openQA / MATH 特征
        "L_sum", "S_es", "H_es", "ans_len",
        "run_len", "flips", "changed_prev",
        "delta_recent_L", "slope_recent_L",
        "curv_L2", "vel_L", "acc_L",
        "mean_logprob", "var_logprob", "neg_ppl",
        # closeQA 历史特征，如果有也一起用
        "cum_A", "cum_B", "cum_C", "cum_D",
        "cum_margin", "delta_recent", "slope_recent",
        "inst_sA", "inst_sB", "inst_sC", "inst_sD",
        "curv_margin2", "curv_cum_A2", "curv_cum_B2",
        "curv_cum_C2", "curv_cum_D2",
        "cum_top_A", "cum_top_B", "cum_top_C", "cum_top_D",
    ]
    feats = [c for c in default_feats if c in columns]
    if len(feats) == 0:
        # 兜底：用所有 numeric 特征
        exclude = {
            "qid", "qa_mode", "step_tokens",
            "probe_answer", "probe_has_answer",
            "final_answer_full", "y_match_final",
        }
        numeric = [
            c for c in columns
            if c not in exclude and pd.api.types.is_numeric_dtype(columns_map.get(c, float))
        ]
        feats = numeric
    return feats


def train_model(X, y):
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=400,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
        )
        clf.fit(X, y)
        return clf, "xgboost"
    except Exception:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, n_jobs=1)
        clf.fit(X, y)
        return clf, "logreg"


def predict_proba_any(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 1:
            return p
    if hasattr(clf, "decision_function"):
        from scipy.special import expit
        s = clf.decision_function(X)
        return expit(s)
    pred = clf.predict(X)
    return pred.astype(float)


def earliest_hits(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Return earliest rows where pred_match==1 per qid (may be empty)."""
    mask = (df_pred["pred_match"] == 1)
    if not mask.any():
        return df_pred.iloc[0:0].copy()
    return (
        df_pred[mask]
        .sort_values(["qid", "step_tokens"])
        .groupby("qid", as_index=False).head(1)
    )


def baseline_last_step(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["qid", "step_tokens"])
          .groupby("qid", as_index=False).tail(1)
    )


def eval_vs_reference(df_pred: pd.DataFrame, ref_col: str, ref_name: str):
    """
    df_pred 必须含有: qid, step_tokens, probe_answer, pred_match, 以及 ref_col。
    对 MATH(openQA)：就是字符串 exact match。
    """
    X = df_pred.copy()
    X = X[X["qid"].notna()].copy()
    X["qid"] = X["qid"].astype(str)

    if ref_col not in X.columns:
        print(f"[SKIP vs {ref_name}] no {ref_col} column.")
        return

    valid = X[
        (X["probe_answer"].str.len() > 0)
        & (X[ref_col].str.len() > 0)
    ].copy()
    total_q = valid["qid"].nunique()
    if total_q == 0:
        print(f"[SKIP vs {ref_name}] no valid qids with answers.")
        return

    # baseline: 最后一 step 的答案字符串 vs ref
    last_rows = baseline_last_step(valid)
    baseline_acc = (last_rows["probe_answer"] == last_rows[ref_col]).mean()
    print(
        f"[BASELINE LAST-STEP (TEXT) ACC vs {ref_name}] "
        f"acc={baseline_acc*100:.2f}% | qids={total_q}"
    )

    # earliest early-stop hits
    hits = earliest_hits(valid)
    covered_q = hits["qid"].nunique()
    coverage = covered_q / total_q
    if covered_q > 0:
        acc_when_stop = (hits["probe_answer"] == hits[ref_col]).mean()
        avg_stop = float(hits["step_tokens"].mean())
    else:
        acc_when_stop, avg_stop = float("nan"), float("nan")
    print(
        f"[EARLY-STOP @qid-TEST vs {ref_name}] "
        f"coverage={coverage*100:.1f}% | "
        f"acc_when_stop={acc_when_stop*100:.2f}% | "
        f"avg_stop_tok={avg_stop:.2f} | qids={total_q}"
    )

    # blended: earliest hit else last-step
    forced = last_rows[["qid", ref_col, "probe_answer"]].rename(
        columns={"probe_answer": "forced_top"}
    ).copy()
    if covered_q > 0:
        f_idx = forced.set_index("qid")
        e_idx = hits.set_index("qid")[["probe_answer"]].rename(
            columns={"probe_answer": "forced_top"}
        )
        f_idx.loc[e_idx.index, "forced_top"] = e_idx["forced_top"]
        forced = f_idx.reset_index()

    final_acc = (forced["forced_top"] == forced[ref_col]).mean()
    print(
        f"[FINAL ACC after Early-Stop vs {ref_name}] "
        f"acc={final_acc*100:.2f}%"
    )

    if np.isfinite(baseline_acc) and np.isfinite(acc_when_stop):
        exp_blend = coverage * acc_when_stop + (1 - coverage) * baseline_acc
        print(
            f"[CHECK] expected blended ACC ≈ {exp_blend*100:.2f}% "
            f"(= {coverage*100:.1f}%*{acc_when_stop*100:.2f}% "
            f"+ {(1-coverage)*100:.1f}%*{baseline_acc*100:.2f}%)"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_probe_jsonl", required=True,
                    help="TRAIN probe JSONL from inference --probe_jsonl")
    ap.add_argument("--test_probe_jsonl", required=True,
                    help="TEST probe JSONL from inference --probe_jsonl")

    ap.add_argument("--feature_cols_json",
                    help="Optional feature list JSON; if omitted, auto-detect.")
    ap.add_argument("--model_out", default="early_stop_cls_math.joblib")
    ap.add_argument("--out_pred_csv", default="early_stop_cls_step_preds_math.csv")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="probability threshold for pred=1 (stop).")
    ap.add_argument("--force_retrain", action="store_true")
    ap.add_argument("--drop_last_in_train", action="store_true",
                    help="Drop each qid's last step in training.")
    args = ap.parse_args()

    # 1) load train/test
    Tr = load_steps_from_jsonl(args.train_probe_jsonl)
    Te = load_steps_from_jsonl(args.test_probe_jsonl)

    # 2) feature list
    global columns_map
    columns_map = {c: Tr[c].dtype for c in Tr.columns}
    feats = load_feature_list(args.feature_cols_json, Tr.columns.tolist())
    print(f"[INFO] using {len(feats)} feature cols: {feats}")

    # 3) optional: drop last step per qid (正则化)
    if args.drop_last_in_train:
        max_step = Tr.groupby("qid")["step_tokens"].transform("max")
        before = len(Tr)
        Tr = Tr[Tr["step_tokens"] < max_step].copy()
        after = len(Tr)
        print(f"[TRAIN] drop last step per qid: removed {before-after} rows")
    else:
        print("[TRAIN] keep last step per qid (default)")

    # 4) train / load model
    need_train, model_obj = True, None
    if (not args.force_retrain) and args.model_out and os.path.exists(args.model_out) and joblib is not None:
        try:
            model_obj = joblib.load(args.model_out)
            need_train = False
            print(f"[MODEL] loaded from {args.model_out}")
        except Exception as e:
            print(f"[WARN] load failed: {e}. Will retrain.")

    if need_train:
        Xtr = Tr[feats].values
        ytr = Tr["y_match_final"].values
        print(f"[TRAIN] rows={len(Tr)} | pos_rate={ytr.mean():.3f}")
        clf, tag = train_model(Xtr, ytr)
        model_obj = {"model": clf, "feats": feats, "tag": tag}
        if args.model_out and joblib is not None:
            os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
            joblib.dump(model_obj, args.model_out)
            print(f"[MODEL] saved -> {args.model_out} ({tag})")
    else:
        tag = model_obj["tag"]
        clf = model_obj["model"]
        feats = model_obj["feats"]

    # 5) test
    Xte = Te[feats].values
    yte = Te["y_match_final"].values
    pte = predict_proba_any(clf, Xte)
    pred = (pte >= float(args.threshold)).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score
    step_acc = (pred == yte).mean()
    prec = precision_score(yte, pred, zero_division=0)
    rec  = recall_score(yte, pred, zero_division=0)
    f1   = f1_score(yte, pred, zero_division=0)
    print(
        f"[STEP-TEST] clf={tag} | thr={args.threshold:.2f} | "
        f"acc={step_acc*100:.2f}%  prec={prec*100:.2f}%  "
        f"rec={rec*100:.2f}%  f1={f1*100:.2f}%  "
        f"| pos_rate(Test)={yte.mean()*100:.1f}%"
    )

    # 6) 写出 step 预测
    Te_out = Te.copy()
    Te_out["prob_match"] = pte
    Te_out["pred_match"] = pred
    os.makedirs(os.path.dirname(args.out_pred_csv) or ".", exist_ok=True)
    Te_out.to_csv(args.out_pred_csv, index=False)
    print(f"[SAVE] step preds -> {args.out_pred_csv} (rows={len(Te_out)})")

    # 7) coverage / avg_stop_tok / vs FINAL
    hits = earliest_hits(Te_out)
    total_q = Te_out["qid"].nunique()
    covered_q = hits["qid"].nunique()
    coverage = covered_q / max(1, total_q)
    avg_stop = float(hits["step_tokens"].mean() if covered_q > 0 else np.nan)
    print(
        f"[COVERAGE] coverage={coverage*100:.1f}% | "
        f"avg_stop_tok={avg_stop:.2f} | qids={total_q}"
    )

    eval_vs_reference(Te_out, ref_col="final_answer_full", ref_name="FINAL/ALL")


if __name__ == "__main__":
    main()

"""
python online_classifier_train.py \
  --train_probe_jsonl online_earlystop_probe_records_train.jsonl \
  --test_probe_jsonl online_earlystop_probe_records.jsonl \
  --model_out model_classifier/early_stop_cls.joblib \
  --out_pred_csv early_stop_cls_step_preds.csv \
  --threshold 0.95 \
  --force_retrain

"""