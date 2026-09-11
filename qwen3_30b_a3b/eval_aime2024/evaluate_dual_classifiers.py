#!/usr/bin/env python3
"""Apply both cluster22 classifiers offline to one complete 90-row AIME replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from math_verify import parse as math_parse, verify as math_verify


FORMAT_VERSION = 1
SCHEMA = "qwen30a3b_aime2024_every50_cluster22_dual095_repeat3_eval_v1"
COLLECTOR_SCHEMA = "qwen30a3b_deepscaler_replay_every50_cluster22_v1"
TARGETS = ("final_consistency", "gold_safe")
THRESHOLD = 0.95
T95_DF2 = 4.302652729911275
TRAINER_SCHEMA = "qwen30a3b_math_every50_cluster22_dual_lgbm_v1"
EXPECTED_FEATURES = (
    "mean_logprob", "min_logprob", "var_logprob", "ans_len",
    "seq_logprob_per_sqrt_len", "top1_share", "top2_share", "vote_margin",
    "vote_entropy", "current_share", "current_is_top", "current_rank",
    "run_len", "flips", "new_cluster", "n_clusters", "agree_last3",
    "agree_last5", "delta_margin", "slope_margin", "curv_margin2",
    "log1p_probe_index",
)


class GradeTimeout(Exception):
    pass


def _grade_timeout(_signum: int, _frame: Any) -> None:
    raise GradeTimeout()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_answer(value: Any) -> str:
    text = str(value or "").strip().replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", text).strip("$")


def safe_grade(prediction: str, gold: str) -> bool:
    """The same bounded math_verify grade used by the prior Q30 evaluator."""
    if not prediction.strip() or not gold.strip():
        return False
    if canonical_answer(prediction) == canonical_answer(gold):
        return True
    can_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    old_handler = signal.signal(signal.SIGALRM, _grade_timeout) if can_alarm else None
    if can_alarm:
        signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        parsed_prediction = math_parse(f"\\boxed{{{prediction}}}")
        parsed_gold = math_parse(f"\\boxed{{{gold}}}")
        return bool(parsed_prediction and parsed_gold and math_verify(parsed_prediction, parsed_gold))
    except Exception:
        return False
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def validate_pack(path: Path, target: str) -> tuple[Any, list[str], int, dict[str, Any]]:
    pack = joblib.load(path)
    if not isinstance(pack, dict) or not callable(getattr(pack.get("model"), "predict_proba", None)):
        raise TypeError(f"{path}: invalid classifier package")
    if (
        pack.get("tag") != TRAINER_SCHEMA
        or pack.get("target") != target
        or float(pack.get("threshold", -1)) != THRESHOLD
        or float(pack.get("deployment_threshold", -1)) != THRESHOLD
    ):
        raise ValueError(f"{path}: target/threshold mismatch")
    if pack.get("feature_schema") != "math_cluster22_every50_v1" or int(pack.get("probe_stride_tokens", -1)) != 50:
        raise ValueError(f"{path}: feature/stride contract mismatch")
    if Path(str(pack.get("backbone_model") or "")).name != "Qwen3-30B-A3B":
        raise ValueError(f"{path}: wrong classifier backbone")
    raw_features = pack.get("feats")
    if isinstance(raw_features, (str, bytes)) or not raw_features:
        raise TypeError(f"{path}: missing ordered features")
    features = [str(name) for name in raw_features]
    if tuple(features) != EXPECTED_FEATURES:
        raise ValueError(f"{path}: exact cluster22 feature order mismatch")
    model = pack["model"]
    classes = list(getattr(model, "classes_", []))
    normalized_classes = [int(item) for item in classes]
    if len(classes) != 2 or set(normalized_classes) != {0, 1}:
        raise ValueError(f"{path}: classifier classes are not exactly 0/1")
    if int(getattr(model, "n_features_in_", len(features))) != len(features):
        raise ValueError(f"{path}: model feature count mismatch")
    return model, features, normalized_classes.index(1), pack


def load_collected(paths: list[Path], source_path: Path) -> list[dict[str, Any]]:
    source_rows = read_jsonl(source_path)
    if len(source_rows) != 90:
        raise ValueError(f"source must contain 90 rows, got {len(source_rows)}")
    source_qids = [str(row.get("qid") or "") for row in source_rows]
    if len(set(source_qids)) != 90 or any(not qid for qid in source_qids):
        raise ValueError("source has empty/duplicate qids")
    expected_by_shard = {index: set(source_qids[index::4]) for index in range(4)}
    collected: dict[str, dict[str, Any]] = {}
    semantic_fingerprints: set[str] = set()
    for shard_index, path in enumerate(paths):
        shard_rows = read_jsonl(path)
        expected_rows = len(expected_by_shard[shard_index])
        if len(shard_rows) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} rows, got {len(shard_rows)}")
        shard_qids: set[str] = set()
        fingerprint: str | None = None
        for line_number, row in enumerate(shard_rows, 1):
            qid = str(row.get("qid") or "")
            if row.get("format_version") != 1 or row.get("record_schema_version") != COLLECTOR_SCHEMA:
                raise ValueError(f"{path}:{line_number}: collector schema mismatch")
            current_fingerprint = str(row.get("generation_config_sha256") or "")
            config = row.get("generation_config")
            if not isinstance(config, dict) or hashlib.sha256(canonical_json(config).encode()).hexdigest() != current_fingerprint:
                raise ValueError(f"{path}:{line_number}: collector fingerprint mismatch")
            if fingerprint is None:
                fingerprint = current_fingerprint
                semantic = dict(config)
                selection = dict(semantic.pop("selection"))
                if selection.get("shard_index") != shard_index or selection.get("shard_count") != 4 or selection.get("qid_count") != expected_rows:
                    raise ValueError(f"{path}:{line_number}: shard selection mismatch")
                semantic_fingerprints.add(hashlib.sha256(canonical_json(semantic).encode()).hexdigest())
            elif fingerprint != current_fingerprint:
                raise ValueError(f"{path}: mixed collector fingerprints")
            probes = row.get("probes")
            if not isinstance(probes, list):
                raise TypeError(f"{path}:{line_number}: probes missing")
            if [int(p["step_tokens"]) for p in probes] != list(range(50, int(row["replay"]["reasoning_tokens"]) + 1, 50)):
                raise ValueError(f"{path}:{line_number}: incomplete every-50 grid")
            if qid in collected or qid in shard_qids:
                raise ValueError(f"duplicate collected qid: {qid}")
            shard_qids.add(qid)
            collected[qid] = row
        if shard_qids != expected_by_shard[shard_index]:
            raise ValueError(f"{path}: qid shard assignment mismatch")
    if len(semantic_fingerprints) != 1 or set(collected) != set(source_qids):
        raise ValueError("collector semantic configs or qid coverage differ")
    return [collected[qid] for qid in source_qids]


def select_result(row: dict[str, Any], model: Any, features: list[str], positive_index: int) -> dict[str, Any]:
    hit: dict[str, Any] | None = None
    scored = 0
    for probe in row["probes"]:
        eligible = bool(probe.get("eligible"))
        vector_values = probe.get("features")
        if not isinstance(vector_values, dict) or tuple(vector_values) != tuple(features):
            raise ValueError(f"{row['qid']}: classifier/collector feature order mismatch")
        vector = np.asarray([[float(vector_values[name]) for name in features]], dtype=float)
        if vector.shape != (1, 22) or not np.isfinite(vector).all():
            raise ValueError(f"{row['qid']}: invalid feature vector")
        probability = 0.0
        if eligible:
            probabilities = np.asarray(model.predict_proba(vector), dtype=float)
            if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
                raise ValueError(f"{row['qid']}: invalid predict_proba output")
            probability = float(probabilities[0, positive_index])
            scored += 1
        if eligible and probability >= THRESHOLD:
            hit = {"probe_index": int(probe["probe_index"]), "step_tokens": int(probe["step_tokens"]), "prediction": str(probe["probe_answer"]), "probability": probability}
            break
    main = row["main"]
    baseline_prediction = str(main["prediction"])
    if hit is None:
        prediction = baseline_prediction
        primary_tokens = int(row["replay"]["reasoning_tokens"])
    else:
        prediction = str(hit["prediction"])
        primary_tokens = int(hit["step_tokens"])
    return {
        "stopped": hit is not None,
        "stop_probe_index": None if hit is None else hit["probe_index"],
        "stop_step_tokens": None if hit is None else hit["step_tokens"],
        "stop_probability": None if hit is None else hit["probability"],
        "prediction": prediction,
        "is_correct": safe_grade(prediction, str(row["gold_answer"])),
        "primary_think_tokens": primary_tokens,
        "eligible_probes_scored_to_decision": scored,
    }


def method_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = [row[method] for row in rows]
    tokens = [int(value["primary_think_tokens"]) for value in values]
    baseline_tokens = [int(row["baseline"]["primary_think_tokens"]) for row in rows]
    mean_tokens = statistics.fmean(tokens)
    mean_baseline = statistics.fmean(baseline_tokens)
    accuracy = statistics.fmean(float(bool(value["is_correct"])) for value in values)
    baseline_accuracy = statistics.fmean(float(bool(row["baseline"]["is_correct"])) for row in rows)
    return {
        "rows": len(rows),
        "correct": sum(bool(value["is_correct"]) for value in values),
        "accuracy": accuracy,
        "accuracy_delta": accuracy - baseline_accuracy,
        "accuracy_delta_pp": 100.0 * (accuracy - baseline_accuracy),
        "stopped": sum(bool(value["stopped"]) for value in values),
        "coverage": statistics.fmean(float(bool(value["stopped"])) for value in values),
        "mean_primary_think_tokens": mean_tokens,
        "median_primary_think_tokens": statistics.median(tokens),
        "mean_token_reduction": mean_baseline - mean_tokens,
        "relative_token_reduction": 1.0 - mean_tokens / mean_baseline if mean_baseline else None,
        "correctness_wins_vs_baseline": sum(int(value["is_correct"]) > int(row["baseline"]["is_correct"]) for row, value in zip(rows, values)),
        "correctness_losses_vs_baseline": sum(int(value["is_correct"]) < int(row["baseline"]["is_correct"]) for row, value in zip(rows, values)),
    }


def baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["baseline"] for row in rows]
    tokens = [int(value["primary_think_tokens"]) for value in values]
    return {"rows": len(rows), "correct": sum(bool(v["is_correct"]) for v in values), "accuracy": statistics.fmean(float(bool(v["is_correct"])) for v in values), "mean_primary_think_tokens": statistics.fmean(tokens), "median_primary_think_tokens": statistics.median(tokens), "reported_vs_regraded_disagreements": sum(bool(v["reported_is_correct"]) != bool(v["is_correct"]) for v in values)}


def ci(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError("run-level CI requires exactly three repeats")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    margin = T95_DF2 * sd / math.sqrt(3)
    return {"n_runs": 3, "mean": mean, "sample_sd": sd, "ci95_low": mean - margin, "ci95_high": mean + margin, "method": "two-sided Student-t run-level CI, df=2"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--shard", required=True, action="append", type=Path)
    parser.add_argument("--classifier-final", required=True, type=Path)
    parser.add_argument("--classifier-gold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-root", required=True, type=Path)
    parser.add_argument("--aggregate", required=True, type=Path)
    args = parser.parse_args()
    if len(args.shard) != 4:
        raise ValueError("exactly four --shard paths are required")

    models: dict[str, tuple[Any, list[str], int, dict[str, Any]]] = {
        "final_consistency": validate_pack(args.classifier_final, "final_consistency"),
        "gold_safe": validate_pack(args.classifier_gold, "gold_safe"),
    }
    if models["final_consistency"][1] != models["gold_safe"][1]:
        raise ValueError("dual classifiers do not share the same ordered feature contract")
    collected = load_collected(args.shard, args.source)
    decisions: list[dict[str, Any]] = []
    for row in collected:
        main = row["main"]
        repeat_id = str(main.get("repeat_id") or "")
        if repeat_id not in ("repeat1", "repeat2", "repeat3") or row["qid"] != f"{repeat_id}::{main.get('original_qid')}":
            raise ValueError(f"{row['qid']}: repeat/original qid binding mismatch")
        baseline_prediction = str(main["prediction"])
        baseline = {
            "prediction": baseline_prediction,
            "reported_is_correct": bool(main["reported_is_correct"]),
            "is_correct": safe_grade(baseline_prediction, str(row["gold_answer"])),
            "answer_valid": bool(main["answer_valid"]),
            "primary_think_tokens": int(row["replay"]["reasoning_tokens"]),
            "original_full_response_tokens": int(main["response_tokens"]),
        }
        result: dict[str, Any] = {"format_version": FORMAT_VERSION, "schema": SCHEMA, "qid": row["qid"], "original_qid": str(main["original_qid"]), "repeat_id": repeat_id, "gold_answer": str(row["gold_answer"]), "baseline": baseline}
        for target in TARGETS:
            model, features, positive_index, _pack = models[target]
            result[target] = select_result(row, model, features, positive_index)
        decisions.append(result)

    atomic_jsonl(args.output, decisions)
    repeat_summaries: dict[str, dict[str, Any]] = {}
    for repeat_id in ("repeat1", "repeat2", "repeat3"):
        subset = [row for row in decisions if row["repeat_id"] == repeat_id]
        if len(subset) != 30 or len({row["original_qid"] for row in subset}) != 30:
            raise ValueError(f"{repeat_id}: incomplete decision set")
        summary = {
            "format_version": FORMAT_VERSION,
            "schema": SCHEMA,
            "repeat_id": repeat_id,
            "threshold": THRESHOLD,
            "primary_token_definition": "re-tokenized baseline response IDs before first </think>; early stop uses selected 50-token position; all probe/suffix/output tokens excluded",
            "no_hit_policy": "original full baseline prediction and full baseline THINK length",
            "grading": "bounded math_verify(parse(boxed prediction), parse(boxed gold)); canonical exact fast path; 2-second SIGALRM",
            "baseline": baseline_metrics(subset),
            "final_consistency": method_metrics(subset, "final_consistency"),
            "gold_safe": method_metrics(subset, "gold_safe"),
        }
        repeat_summaries[repeat_id] = summary
        atomic_json(args.summary_root / repeat_id / "summary.json", summary)
        atomic_jsonl(args.summary_root / repeat_id / "decisions.jsonl", subset)

    aggregate: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA,
        "threshold": THRESHOLD,
        "repeats": ["repeat1", "repeat2", "repeat3"],
        "classifier_artifacts": {"final_consistency": {"path": str(args.classifier_final.resolve()), "sha256": file_sha256(args.classifier_final)}, "gold_safe": {"path": str(args.classifier_gold.resolve()), "sha256": file_sha256(args.classifier_gold)}},
        "collector_shards": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.shard],
        "decisions": {"path": str(args.output.resolve()), "sha256": file_sha256(args.output)},
        "run_level_ci95": {},
    }
    metric_names = {
        "baseline": ("accuracy", "mean_primary_think_tokens"),
        "final_consistency": ("accuracy", "accuracy_delta", "coverage", "mean_primary_think_tokens", "mean_token_reduction", "relative_token_reduction"),
        "gold_safe": ("accuracy", "accuracy_delta", "coverage", "mean_primary_think_tokens", "mean_token_reduction", "relative_token_reduction"),
    }
    for method, names in metric_names.items():
        aggregate["run_level_ci95"][method] = {name: ci([float(repeat_summaries[repeat_id][method][name]) for repeat_id in ("repeat1", "repeat2", "repeat3")]) for name in names}
    atomic_json(args.aggregate, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
