#!/usr/bin/env python3
"""Strict offline Math500 evaluation for the two every-50 Qwen30 classifiers.

The collector outputs already contain the replayed, causal cluster22 features.
This program does not run inference.  It verifies the four completed collector
shards, applies both classifier packages at their fixed 0.95 deployment
threshold, and writes repeat-level decisions/summaries plus run-level t CIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


FORMAT_VERSION = 1
OUTPUT_SCHEMA = "qwen30a3b_math500_every50_cluster22_dual095_summary_v1"
COLLECTOR_SCHEMA = "qwen30a3b_deepscaler_replay_every50_cluster22_v1"
CLASSIFIER_SCHEMA = "qwen30a3b_math_every50_cluster22_dual_lgbm_v1"
FEATURE_SCHEMA = "math_cluster22_every50_v1"
LABEL_VERSION = "cached_math_equivalence_probe_to_full_and_gold_v1"
TARGETS = ("final_consistency", "gold_safe")
REPEATS = ("repeat1", "repeat2", "repeat3")
THRESHOLD = 0.95
PROBE_STRIDE = 50
ROWS_PER_REPEAT = 500
TOTAL_ROWS = len(REPEATS) * ROWS_PER_REPEAT
SHARD_COUNT = 4
ROWS_PER_SHARD = TOTAL_ROWS // SHARD_COUNT
T95_DF2 = 4.302652729911275
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"
PROBE_SUFFIX = "\n</think>\n\n\\boxed{"
COLLECTOR_IMPLEMENTATION = "gamma_vllm09_public_generate_saved_cot_replay_every50_v1"
COLLECTOR_SEED = 20260907
FEATURE_MODULE_SHA256 = "213ad3652080b58eaba2c38ad3ba95ade038db4eb3fef75421cf7776d117b630"

# Deliberately copied as a literal contract.  Merely checking for 22 columns
# would allow the collector and classifier to drift together unnoticed.
FEATURES = (
    "mean_logprob",
    "min_logprob",
    "var_logprob",
    "ans_len",
    "seq_logprob_per_sqrt_len",
    "top1_share",
    "top2_share",
    "vote_margin",
    "vote_entropy",
    "current_share",
    "current_is_top",
    "current_rank",
    "run_len",
    "flips",
    "new_cluster",
    "n_clusters",
    "agree_last3",
    "agree_last5",
    "delta_margin",
    "slope_margin",
    "curv_margin2",
    "log1p_probe_index",
)
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(list(FEATURES), separators=(",", ":")).encode("utf-8")
).hexdigest()
QID_PATTERN = re.compile(r"^(repeat[123])::(.+)$")


class GradeTimeout(Exception):
    """A symbolic Math500 grade exceeded its bounded wall-clock budget."""


def _raise_grade_timeout(_signum: int, _frame: Any) -> None:
    raise GradeTimeout()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qids_sha256(qids: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(qids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, rendered)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rendered = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    _atomic_bytes(path, rendered)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _require_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location}: expected integer >= {minimum}, got {value!r}")
    return value


def _require_finite_features(value: Any, *, location: str) -> list[float]:
    if not isinstance(value, dict) or tuple(value.keys()) != FEATURES:
        raise ValueError(f"{location}: features are not exact ordered cluster22")
    vector: list[float] = []
    for name in FEATURES:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{location}: feature {name} is not numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{location}: feature {name} is not finite")
        vector.append(number)
    return vector


@dataclass(frozen=True)
class PreparedSource:
    path: Path
    rows: tuple[dict[str, Any], ...]
    by_qid: Mapping[str, dict[str, Any]]
    sha256: str
    ordered_qids_sha256: str
    generation_fingerprints: tuple[str, ...]


def load_prepared_source(path: Path) -> PreparedSource:
    rows = _read_jsonl(path)
    if len(rows) != TOTAL_ROWS:
        raise ValueError(f"{path}: prepared source must contain exactly 1500 rows")
    seen: set[str] = set()
    repeat_originals: dict[str, set[str]] = {item: set() for item in REPEATS}
    generation_fingerprints: set[str] = set()
    for source_index, row in enumerate(rows):
        location = f"{path}:{source_index + 1}"
        qid = row.get("qid")
        match = QID_PATTERN.fullmatch(qid) if isinstance(qid, str) else None
        if match is None or qid in seen:
            raise ValueError(f"{location}: invalid/duplicate prefixed qid")
        repeat_id, original_qid = match.groups()
        if row.get("source_index") != source_index:
            raise ValueError(f"{location}: source_index must equal source row offset")
        if row.get("sample_rank") != repeat_id or row.get("sample_stratum") != "math500":
            raise ValueError(f"{location}: prepared source repeat metadata mismatch")
        if not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError(f"{location}: empty problem")
        if not isinstance(row.get("gold_answer"), str) or not row["gold_answer"].strip():
            raise ValueError(f"{location}: empty gold_answer")
        main = row.get("main")
        if (
            not isinstance(main, dict)
            or main.get("repeat_id") != repeat_id
            or str(main.get("original_qid") or "") != original_qid
            or not isinstance(main.get("text"), str)
        ):
            raise ValueError(f"{location}: prepared source main/qid binding mismatch")
        if (
            main.get("eval_dataset") != "math500"
            or not isinstance(main.get("prediction"), str)
            or not isinstance(main.get("reported_is_correct"), bool)
            or not isinstance(main.get("answer_valid"), bool)
            or isinstance(main.get("response_tokens"), bool)
            or not isinstance(main.get("response_tokens"), int)
            or int(main["response_tokens"]) < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(main.get("baseline_run_fingerprint") or ""))
            is None
        ):
            raise ValueError(f"{location}: prepared source main contract mismatch")
        old_config = row.get("generation_config")
        if not isinstance(old_config, dict):
            raise TypeError(f"{location}: prepared source generation_config missing")
        observed = hashlib.sha256(canonical_json(old_config).encode("utf-8")).hexdigest()
        if row.get("generation_config_sha256") != observed:
            raise ValueError(f"{location}: prepared source generation fingerprint mismatch")
        if Path(str(old_config.get("model") or "")).name != "Qwen3-30B-A3B":
            raise ValueError(f"{location}: prepared source backbone mismatch")
        generation_fingerprints.add(observed)
        repeat_originals[repeat_id].add(original_qid)
        seen.add(qid)
    expected_repeat_at_offset = [item for item in REPEATS for _ in range(ROWS_PER_REPEAT)]
    observed_repeat_at_offset = [QID_PATTERN.fullmatch(str(row["qid"])).group(1) for row in rows]  # type: ignore[union-attr]
    if observed_repeat_at_offset != expected_repeat_at_offset:
        raise ValueError("prepared source must be ordered repeat1, repeat2, repeat3")
    if any(len(repeat_originals[item]) != ROWS_PER_REPEAT for item in REPEATS):
        raise ValueError("prepared source must have 500 unique qids per repeat")
    if any(repeat_originals[item] != repeat_originals[REPEATS[0]] for item in REPEATS[1:]):
        raise ValueError("prepared source repeats have different original qid sets")
    ordered_qids = [str(row["qid"]) for row in rows]
    return PreparedSource(
        path=path.resolve(),
        rows=tuple(rows),
        by_qid={str(row["qid"]): row for row in rows},
        sha256=file_sha256(path),
        ordered_qids_sha256=qids_sha256(ordered_qids),
        generation_fingerprints=tuple(sorted(generation_fingerprints)),
    )


def _validate_generation_config(
    config: Any,
    *,
    path: Path,
    line_number: int,
    source: PreparedSource,
) -> tuple[int, str]:
    location = f"{path}:{line_number}"
    if not isinstance(config, dict):
        raise TypeError(f"{location}: generation_config must be an object")
    expected_top_level = {
        "format_version",
        "record_schema_version",
        "implementation",
        "model",
        "source",
        "selection",
        "prompt",
        "probe",
        "features",
        "engine",
    }
    if set(config) != expected_top_level:
        raise ValueError(f"{location}: generation_config key contract mismatch")
    if (
        config.get("format_version") != FORMAT_VERSION
        or config.get("record_schema_version") != COLLECTOR_SCHEMA
        or config.get("implementation") != COLLECTOR_IMPLEMENTATION
    ):
        raise ValueError(f"{location}: collector implementation contract mismatch")
    model = config.get("model")
    if not isinstance(model, dict) or set(model) != {
        "path",
        "config.json_sha256",
        "tokenizer_config.json_sha256",
        "generation_config.json_sha256",
    }:
        raise ValueError(f"{location}: model identity contract mismatch")
    if Path(str(model.get("path") or "")).name != "Qwen3-30B-A3B":
        raise ValueError(f"{location}: collector backbone mismatch")
    for key in ("config.json_sha256", "tokenizer_config.json_sha256", "generation_config.json_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(model.get(key) or "")) is None:
            raise ValueError(f"{location}: invalid model identity digest {key}")
    source_contract = config.get("source")
    expected_source = {
        "path": str(source.path),
        "sha256": source.sha256,
        "qid_count": TOTAL_ROWS,
        "ordered_qids_sha256": source.ordered_qids_sha256,
        "source_generation_config_sha256_values": list(source.generation_fingerprints),
        "main_reused_without_regeneration": True,
    }
    if not isinstance(source_contract, dict) or source_contract != expected_source:
        raise ValueError(f"{location}: prepared source identity contract mismatch")

    selection = config.get("selection")
    probe = config.get("probe")
    features = config.get("features")
    if not isinstance(selection, dict):
        raise TypeError(f"{location}: missing selection contract")
    shard_index = _require_int(
        selection.get("shard_index"), location=f"{location}/selection.shard_index"
    )
    if (
        shard_index >= SHARD_COUNT
        or selection.get("shard_count") != SHARD_COUNT
        or selection.get("qid_count") != ROWS_PER_SHARD
        or selection.get("limit") != 0
    ):
        raise ValueError(f"{location}: invalid four-shard selection contract")
    expected_shard_qids = [str(row["qid"]) for row in source.rows[shard_index::SHARD_COUNT]]
    expected_selection = {
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "limit": 0,
        "qid_count": ROWS_PER_SHARD,
        "ordered_qids_sha256": qids_sha256(expected_shard_qids),
    }
    if selection != expected_selection:
        raise ValueError(f"{location}: shard qid selection contract mismatch")
    expected_prompt = {
        "template": "qwen3_chat_plus_explicit_think_v1",
        "system_prompt": SYSTEM_PROMPT,
        "enable_thinking": True,
        "explicit_think_open": True,
    }
    if config.get("prompt") != expected_prompt:
        raise ValueError(f"{location}: collector prompt contract mismatch")
    expected_probe = {
        "schedule": "every_50_main_think_tokens_after_text_retokenization",
        "token_step": PROBE_STRIDE,
        "suffix": PROBE_SUFFIX,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "repetition_penalty": 1.2,
        "max_tokens": 64,
        "logprobs": 20,
        "seed": COLLECTOR_SEED,
        "balanced_outer_box_boundary": True,
    }
    if not isinstance(probe, dict) or probe != expected_probe:
        raise ValueError(f"{location}: every-50 probe contract mismatch")
    expected_features = {
        "implementation": "ClusterEvidenceTracker_answer_logprob_stats_v1",
        "module_sha256": FEATURE_MODULE_SHA256,
        "order": list(FEATURES),
        "count": len(FEATURES),
        "raw_logprobs_saved": False,
    }
    if not isinstance(features, dict) or features != expected_features:
        raise ValueError(f"{location}: generation feature contract mismatch")
    expected_engine = {
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 40960,
        "probe_batch_size": 64,
        "max_num_batched_tokens": 65536,
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": True,
        "enforce_eager": True,
    }
    if config.get("engine") != expected_engine:
        raise ValueError(f"{location}: collector engine contract mismatch")
    semantic = dict(config)
    semantic.pop("selection")
    return shard_index, hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()


def _validate_collector_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    declared_shard: int,
    source: PreparedSource,
) -> tuple[str, str, str]:
    location = f"{path}:{line_number}"
    if (
        row.get("format_version") != FORMAT_VERSION
        or row.get("record_schema_version") != COLLECTOR_SCHEMA
    ):
        raise ValueError(f"{location}: collector schema mismatch")
    config = row.get("generation_config")
    observed_fingerprint = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if row.get("generation_config_sha256") != observed_fingerprint:
        raise ValueError(f"{location}: generation_config fingerprint mismatch")
    shard_index, semantic_fingerprint = _validate_generation_config(
        config, path=path, line_number=line_number, source=source
    )
    if shard_index != declared_shard:
        raise ValueError(f"{location}: mixed shard indices")

    qid = row.get("qid")
    match = QID_PATTERN.fullmatch(qid) if isinstance(qid, str) else None
    if match is None:
        raise ValueError(f"{location}: qid must be repeat[1-3]::original_qid")
    repeat_id, original_qid = match.groups()
    source_index = _require_int(row.get("source_index"), location=f"{location}/source_index")
    if source_index >= TOTAL_ROWS or source_index % SHARD_COUNT != shard_index:
        raise ValueError(f"{location}: source_index/shard binding mismatch")
    if row.get("sample_rank") != repeat_id or row.get("sample_stratum") != "math500":
        raise ValueError(f"{location}: Math500 repeat metadata mismatch")
    if not isinstance(row.get("problem"), str) or not row["problem"].strip():
        raise ValueError(f"{location}: empty problem")
    if not isinstance(row.get("gold_answer"), str) or not row["gold_answer"].strip():
        raise ValueError(f"{location}: empty gold_answer")

    main = row.get("main")
    replay = row.get("replay")
    probes = row.get("probes")
    if not isinstance(main, dict) or not isinstance(replay, dict) or not isinstance(probes, list):
        raise TypeError(f"{location}: missing main/replay/probes objects")
    if main.get("repeat_id") != repeat_id or str(main.get("original_qid") or "") != original_qid:
        raise ValueError(f"{location}: prefixed qid does not match main metadata")
    for field in ("prediction", "reported_is_correct", "answer_valid", "response_tokens"):
        if field not in main:
            raise ValueError(f"{location}: main missing {field}")
    if not isinstance(main["prediction"], str):
        raise TypeError(f"{location}: main prediction must be text")
    if not isinstance(main["reported_is_correct"], bool) or not isinstance(main["answer_valid"], bool):
        raise TypeError(f"{location}: main correctness/validity flags must be bool")
    _require_int(main["response_tokens"], location=f"{location}/main.response_tokens")

    reasoning_tokens = _require_int(
        replay.get("reasoning_tokens"), location=f"{location}/replay.reasoning_tokens"
    )
    if replay.get("probe_count") != len(probes):
        raise ValueError(f"{location}: replay probe_count mismatch")
    expected_steps = list(range(PROBE_STRIDE, reasoning_tokens + 1, PROBE_STRIDE))
    if len(probes) != len(expected_steps):
        raise ValueError(f"{location}: incomplete every-50 probe grid")
    for probe_offset, (probe, expected_step) in enumerate(zip(probes, expected_steps), 1):
        probe_location = f"{location}/probe[{probe_offset}]"
        if not isinstance(probe, dict):
            raise TypeError(f"{probe_location}: probe must be an object")
        if probe.get("probe_index") != probe_offset or probe.get("step_tokens") != expected_step:
            raise ValueError(f"{probe_location}: probe index/step mismatch")
        if not isinstance(probe.get("eligible"), bool):
            raise TypeError(f"{probe_location}: eligible must be bool")
        if not isinstance(probe.get("probe_answer"), str):
            raise TypeError(f"{probe_location}: probe_answer must be text")
        _require_finite_features(probe.get("features"), location=probe_location)
        answer_key = probe.get("answer_key")
        box_closed = probe.get("box_closed")
        if not isinstance(answer_key, str) or not isinstance(box_closed, bool):
            raise TypeError(f"{probe_location}: answer_key/box_closed type mismatch")
        logprob_tokens = _require_int(
            probe.get("probe_answer_logprob_tokens"),
            location=f"{probe_location}/probe_answer_logprob_tokens",
        )
        derived_eligible = bool(answer_key and box_closed and logprob_tokens > 0)
        if probe["eligible"] is not derived_eligible:
            raise ValueError(f"{probe_location}: eligible derivation mismatch")

    source_row = source.by_qid.get(qid)
    if source_row is None:
        raise ValueError(f"{location}: qid absent from prepared source")
    for field in (
        "qid",
        "source_index",
        "sample_rank",
        "sample_stratum",
        "problem",
        "gold_answer",
        "main",
    ):
        if row.get(field) != source_row.get(field):
            raise ValueError(f"{location}: collector/source field differs: {field}")
    return repeat_id, original_qid, semantic_fingerprint


def load_collector_shards(paths: Sequence[Path], source_path: Path) -> list[dict[str, Any]]:
    if len(paths) != SHARD_COUNT:
        raise ValueError("exactly four collector --input paths are required")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != SHARD_COUNT:
        raise ValueError("collector input paths must be distinct")

    source = load_prepared_source(source_path)
    merged: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    semantic_fingerprints: set[str] = set()
    for path in paths:
        rows = _read_jsonl(path)
        if len(rows) != ROWS_PER_SHARD:
            raise ValueError(f"{path}: expected {ROWS_PER_SHARD} rows, got {len(rows)}")
        first_config = rows[0].get("generation_config") if rows else None
        if not isinstance(first_config, dict) or not isinstance(first_config.get("selection"), dict):
            raise TypeError(f"{path}: missing shard selection")
        declared_shard = first_config["selection"].get("shard_index")
        if isinstance(declared_shard, bool) or not isinstance(declared_shard, int):
            raise ValueError(f"{path}: invalid shard index")
        if declared_shard in seen_shards:
            raise ValueError(f"duplicate declared shard index {declared_shard}")
        seen_shards.add(declared_shard)
        path_fingerprint: str | None = None
        for line_number, row in enumerate(rows, 1):
            _, _, semantic = _validate_collector_row(
                row,
                path=path,
                line_number=line_number,
                declared_shard=declared_shard,
                source=source,
            )
            current_full = str(row["generation_config_sha256"])
            if path_fingerprint is None:
                path_fingerprint = current_full
            elif current_full != path_fingerprint:
                raise ValueError(f"{path}: mixed generation fingerprints")
            semantic_fingerprints.add(semantic)
            merged.append(row)
    if seen_shards != set(range(SHARD_COUNT)):
        raise ValueError(f"declared shard set mismatch: {sorted(seen_shards)}")
    if len(semantic_fingerprints) != 1:
        raise ValueError("collector shards have different semantic generation configs")

    qids = [str(row["qid"]) for row in merged]
    source_indices = [int(row["source_index"]) for row in merged]
    if len(merged) != TOTAL_ROWS or len(set(qids)) != TOTAL_ROWS:
        raise ValueError("collector outputs must contain exactly 1500 unique prefixed qids")
    if set(source_indices) != set(range(TOTAL_ROWS)):
        raise ValueError("collector source_index coverage must be exactly 0..1499")
    merged.sort(key=lambda row: int(row["source_index"]))
    if [str(row["qid"]) for row in merged] != [str(row["qid"]) for row in source.rows]:
        raise ValueError("collector ordered qids differ from prepared source")

    repeat_originals: dict[str, set[str]] = {repeat_id: set() for repeat_id in REPEATS}
    reference_payload: dict[str, tuple[str, str]] = {}
    for row in merged:
        repeat_id, original_qid = QID_PATTERN.fullmatch(str(row["qid"])).groups()  # type: ignore[union-attr]
        repeat_originals[repeat_id].add(original_qid)
        payload = (str(row["problem"]), str(row["gold_answer"]))
        if original_qid in reference_payload and reference_payload[original_qid] != payload:
            raise ValueError(f"{original_qid}: problem/gold differs across repeats")
        reference_payload[original_qid] = payload
    for repeat_id in REPEATS:
        if len(repeat_originals[repeat_id]) != ROWS_PER_REPEAT:
            raise ValueError(f"{repeat_id}: expected 500 unique original qids")
    if any(repeat_originals[item] != repeat_originals[REPEATS[0]] for item in REPEATS[1:]):
        raise ValueError("the three repeats do not contain the same 500 original qids")
    return merged


@dataclass(frozen=True)
class ClassifierBundle:
    target: str
    model: Any
    positive_index: int
    path: Path


def _default_pack_loader(path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("joblib is required to load classifier packages") from exc
    return joblib.load(path)


def validate_classifier_pack(
    path: Path,
    target: str,
    *,
    loader: Callable[[Path], Any] | None = None,
) -> ClassifierBundle:
    pack = (loader or _default_pack_loader)(path)
    if not isinstance(pack, dict):
        raise TypeError(f"{path}: classifier package must be a dictionary")
    if pack.get("tag") != CLASSIFIER_SCHEMA:
        raise ValueError(f"{path}: classifier training schema mismatch")
    if pack.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"{path}: classifier feature schema mismatch")
    if pack.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError(f"{path}: classifier feature schema digest mismatch")
    if pack.get("label_version") != LABEL_VERSION or pack.get("target") != target:
        raise ValueError(f"{path}: classifier target/label contract mismatch")
    if (
        float(pack.get("threshold", -1.0)) != THRESHOLD
        or float(pack.get("deployment_threshold", -1.0)) != THRESHOLD
        or pack.get("probe_stride_tokens") != PROBE_STRIDE
    ):
        raise ValueError(f"{path}: threshold/stride mismatch")
    if tuple(pack.get("feats") or ()) != FEATURES:
        raise ValueError(f"{path}: classifier features are not exact ordered cluster22")
    if "feature_names" in pack and tuple(pack["feature_names"]) != FEATURES:
        raise ValueError(f"{path}: classifier feature_names mismatch")
    if Path(str(pack.get("backbone_model") or "")).name != "Qwen3-30B-A3B":
        raise ValueError(f"{path}: classifier was not trained for Qwen3-30B-A3B")
    model = pack.get("model")
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(f"{path}: model has no predict_proba")
    classes = list(getattr(model, "classes_", ()))
    try:
        normalized_classes = [int(item) for item in classes]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid classifier classes") from exc
    if len(normalized_classes) != 2 or set(normalized_classes) != {0, 1}:
        raise ValueError(f"{path}: classes must be exactly binary 0/1")
    if int(getattr(model, "n_features_in_", -1)) != len(FEATURES):
        raise ValueError(f"{path}: classifier feature count mismatch")
    return ClassifierBundle(
        target=target,
        model=model,
        positive_index=normalized_classes.index(1),
        path=path,
    )


def _probability(bundle: ClassifierBundle, vector: list[float], *, qid: str) -> float:
    raw = bundle.model.predict_proba([vector])
    values = raw.tolist() if callable(getattr(raw, "tolist", None)) else raw
    if (
        not isinstance(values, Sequence)
        or len(values) != 1
        or not isinstance(values[0], Sequence)
        or len(values[0]) != 2
    ):
        raise ValueError(f"{qid}/{bundle.target}: predict_proba must return shape (1, 2)")
    probabilities = [float(item) for item in values[0]]
    if (
        any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    ):
        raise ValueError(f"{qid}/{bundle.target}: invalid probability vector")
    return probabilities[bundle.positive_index]


def canonical_answer(value: Any) -> str:
    text = str(value or "").strip().replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", text).strip("$")


def safe_math_grade(prediction: str, gold: str, *, timeout_seconds: float = 2.0) -> bool:
    """Bounded Math500 grade: canonical exact fast path, then math_verify."""
    if not prediction.strip() or not gold.strip():
        return False
    if canonical_answer(prediction) == canonical_answer(gold):
        return True
    try:
        from math_verify import parse as math_parse, verify as math_verify
    except ImportError as exc:  # fail closed rather than silently corrupt metrics
        raise RuntimeError("math_verify is required for non-exact Math500 grading") from exc

    can_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    previous_handler = signal.signal(signal.SIGALRM, _raise_grade_timeout) if can_alarm else None
    if can_alarm:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        parsed_prediction = math_parse(f"\\boxed{{{prediction}}}")
        parsed_gold = math_parse(f"\\boxed{{{gold}}}")
        return bool(parsed_prediction and parsed_gold and math_verify(parsed_prediction, parsed_gold))
    except Exception:
        return False
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)


def _policy_decision(
    row: dict[str, Any],
    bundle: ClassifierBundle,
    *,
    grade: Callable[[str, str], bool],
) -> dict[str, Any]:
    qid = str(row["qid"])
    hit: dict[str, Any] | None = None
    eligible_scored = 0
    for probe in row["probes"]:
        if not probe["eligible"]:
            continue
        vector = _require_finite_features(
            probe["features"], location=f"{qid}/probe[{probe['probe_index']}]"
        )
        probability = _probability(bundle, vector, qid=qid)
        eligible_scored += 1
        if probability >= THRESHOLD:
            hit = {
                "probe_index": int(probe["probe_index"]),
                "step_tokens": int(probe["step_tokens"]),
                "probability": probability,
                "prediction": str(probe["probe_answer"]),
            }
            break
    stopped = hit is not None
    prediction = str(hit["prediction"] if stopped else row["main"]["prediction"])
    think_tokens = int(
        hit["step_tokens"] if stopped else row["replay"]["reasoning_tokens"]
    )
    return {
        "stopped": stopped,
        "stop_probe_index": int(hit["probe_index"]) if stopped else None,
        "stop_step_tokens": int(hit["step_tokens"]) if stopped else None,
        "stop_probability": float(hit["probability"]) if stopped else None,
        "prediction": prediction,
        "is_correct": bool(grade(prediction, str(row["gold_answer"]))),
        "primary_think_tokens": think_tokens,
        "eligible_probes_scored_to_decision": eligible_scored,
    }


def build_decisions(
    rows: Sequence[dict[str, Any]],
    bundles: Mapping[str, ClassifierBundle],
    *,
    grader: Callable[[str, str], bool] = safe_math_grade,
) -> list[dict[str, Any]]:
    grade_cache: dict[tuple[str, str], bool] = {}

    def grade(prediction: str, gold: str) -> bool:
        key = (prediction, gold)
        if key not in grade_cache:
            grade_cache[key] = bool(grader(prediction, gold))
        return grade_cache[key]

    decisions: list[dict[str, Any]] = []
    for row in rows:
        baseline_prediction = str(row["main"]["prediction"])
        baseline = {
            "prediction": baseline_prediction,
            "reported_is_correct": bool(row["main"]["reported_is_correct"]),
            "is_correct": grade(baseline_prediction, str(row["gold_answer"])),
            "answer_valid": bool(row["main"]["answer_valid"]),
            "primary_think_tokens": int(row["replay"]["reasoning_tokens"]),
            "original_full_response_tokens": int(row["main"]["response_tokens"]),
        }
        repeat_id, original_qid = QID_PATTERN.fullmatch(str(row["qid"])).groups()  # type: ignore[union-attr]
        decision: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "schema": OUTPUT_SCHEMA,
            "qid": str(row["qid"]),
            "original_qid": original_qid,
            "repeat_id": repeat_id,
            "gold_answer": str(row["gold_answer"]),
            "baseline": baseline,
        }
        for target in TARGETS:
            decision[target] = _policy_decision(
                row, bundles[target], grade=grade
            )
        decisions.append(decision)
    return decisions


def _baseline_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [row["baseline"] for row in rows]
    tokens = [int(value["primary_think_tokens"]) for value in values]
    return {
        "rows": len(rows),
        "correct": sum(bool(value["is_correct"]) for value in values),
        "accuracy": statistics.fmean(float(bool(value["is_correct"])) for value in values),
        "mean_primary_think_tokens": statistics.fmean(tokens),
        "median_primary_think_tokens": statistics.median(tokens),
        "reported_vs_regraded_disagreements": sum(
            bool(value["reported_is_correct"]) != bool(value["is_correct"])
            for value in values
        ),
    }


def _policy_metrics(rows: Sequence[dict[str, Any]], target: str) -> dict[str, Any]:
    values = [row[target] for row in rows]
    tokens = [int(value["primary_think_tokens"]) for value in values]
    baseline_tokens = [int(row["baseline"]["primary_think_tokens"]) for row in rows]
    mean_tokens = statistics.fmean(tokens)
    mean_baseline = statistics.fmean(baseline_tokens)
    accuracy = statistics.fmean(float(bool(value["is_correct"])) for value in values)
    baseline_accuracy = statistics.fmean(
        float(bool(row["baseline"]["is_correct"])) for row in rows
    )
    accuracy_delta_pp = 100.0 * (accuracy - baseline_accuracy)
    mean_token_reduction = statistics.fmean(
        baseline - selected for baseline, selected in zip(baseline_tokens, tokens)
    )
    relative_token_reduction = (
        mean_token_reduction / mean_baseline if mean_baseline else None
    )
    wins = sum(
        int(value["is_correct"]) > int(row["baseline"]["is_correct"])
        for row, value in zip(rows, values)
    )
    losses = sum(
        int(value["is_correct"]) < int(row["baseline"]["is_correct"])
        for row, value in zip(rows, values)
    )
    return {
        "rows": len(rows),
        "correct": sum(bool(value["is_correct"]) for value in values),
        "accuracy": accuracy,
        "accuracy_delta_pp": accuracy_delta_pp,
        "stopped": sum(bool(value["stopped"]) for value in values),
        "coverage": statistics.fmean(float(bool(value["stopped"])) for value in values),
        "mean_primary_think_tokens": mean_tokens,
        "median_primary_think_tokens": statistics.median(tokens),
        "mean_token_reduction": mean_token_reduction,
        "relative_token_reduction": relative_token_reduction,
        "correctness_wins_vs_baseline": wins,
        "correctness_losses_vs_baseline": losses,
        "paired_vs_baseline": {
            "accuracy_delta_pp": accuracy_delta_pp,
            "mean_primary_think_token_reduction": mean_token_reduction,
            "relative_mean_primary_think_token_reduction": relative_token_reduction,
            "correctness_wins": wins,
            "correctness_losses": losses,
        },
    }


def student_t_ci_df2(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(REPEATS):
        raise ValueError("Student-t run-level CI requires exactly three repeats")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("Student-t run-level CI received non-finite input")
    mean = statistics.fmean(numbers)
    sample_sd = statistics.stdev(numbers)
    margin = T95_DF2 * sample_sd / math.sqrt(len(numbers))
    return {
        "n_runs": 3,
        "df": 2,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "method": "two-sided Student-t run-level CI (df=2)",
    }


def summarize(
    *,
    source: Path,
    inputs: Sequence[Path],
    classifier_final: Path,
    classifier_gold: Path,
    output_dir: Path,
    pack_loader: Callable[[Path], Any] | None = None,
    grader: Callable[[str, str], bool] = safe_math_grade,
) -> dict[str, Any]:
    rows = load_collector_shards(inputs, source)
    bundles = {
        "final_consistency": validate_classifier_pack(
            classifier_final, "final_consistency", loader=pack_loader
        ),
        "gold_safe": validate_classifier_pack(
            classifier_gold, "gold_safe", loader=pack_loader
        ),
    }
    decisions = build_decisions(rows, bundles, grader=grader)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "decisions.jsonl"
    atomic_jsonl(combined_path, decisions)

    repeat_summaries: dict[str, dict[str, Any]] = {}
    output_files: dict[str, Any] = {
        "combined_decisions": {
            "path": str(combined_path.resolve()),
            "sha256": file_sha256(combined_path),
        }
    }
    for repeat_id in REPEATS:
        subset = [row for row in decisions if row["repeat_id"] == repeat_id]
        if len(subset) != ROWS_PER_REPEAT:
            raise ValueError(f"{repeat_id}: incomplete decisions")
        repeat_dir = output_dir / repeat_id
        decisions_path = repeat_dir / "decisions.jsonl"
        summary_path = repeat_dir / "summary.json"
        atomic_jsonl(decisions_path, subset)
        summary = {
            "format_version": FORMAT_VERSION,
            "schema": OUTPUT_SCHEMA,
            "repeat_id": repeat_id,
            "threshold": THRESHOLD,
            "rows": ROWS_PER_REPEAT,
            "primary_token_definition": (
                "baseline re-tokenized THINK tokens before first </think>; early hit uses "
                "its every-50 step_tokens; probe prompt/output/suffix tokens are excluded"
            ),
            "no_hit_policy": "use original full baseline prediction and baseline THINK length",
            "grading": (
                "canonical exact fast path, otherwise bounded math_verify on boxed prediction/gold"
            ),
            "baseline": _baseline_metrics(subset),
            "final_consistency": _policy_metrics(subset, "final_consistency"),
            "gold_safe": _policy_metrics(subset, "gold_safe"),
        }
        atomic_json(summary_path, summary)
        repeat_summaries[repeat_id] = summary
        output_files[repeat_id] = {
            "decisions": {
                "path": str(decisions_path.resolve()),
                "sha256": file_sha256(decisions_path),
            },
            "summary": {
                "path": str(summary_path.resolve()),
                "sha256": file_sha256(summary_path),
            },
        }

    metric_names = {
        "baseline": ("accuracy", "mean_primary_think_tokens"),
        "final_consistency": (
            "accuracy",
            "accuracy_delta_pp",
            "coverage",
            "mean_primary_think_tokens",
            "mean_token_reduction",
            "relative_token_reduction",
        ),
        "gold_safe": (
            "accuracy",
            "accuracy_delta_pp",
            "coverage",
            "mean_primary_think_tokens",
            "mean_token_reduction",
            "relative_token_reduction",
        ),
    }
    aggregate: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "schema": OUTPUT_SCHEMA,
        "threshold": THRESHOLD,
        "probe_stride_tokens": PROBE_STRIDE,
        "feature_schema": FEATURE_SCHEMA,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "repeats": list(REPEATS),
        "rows": TOTAL_ROWS,
        "classifier_artifacts": {
            "final_consistency": {
                "path": str(classifier_final.resolve()),
                "sha256": file_sha256(classifier_final),
            },
            "gold_safe": {
                "path": str(classifier_gold.resolve()),
                "sha256": file_sha256(classifier_gold),
            },
        },
        "collector_shards": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in inputs
        ],
        "prepared_source": {
            "path": str(source.resolve()),
            "sha256": file_sha256(source),
        },
        "outputs": output_files,
        "run_level_ci95": {},
    }
    for method, names in metric_names.items():
        aggregate["run_level_ci95"][method] = {
            name: student_t_ci_df2(
                [repeat_summaries[repeat_id][method][name] for repeat_id in REPEATS]
            )
            for name in names
        }
    aggregate_path = output_dir / "aggregate.json"
    atomic_json(aggregate_path, aggregate)
    return aggregate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        type=Path,
        help="one completed collect_replay_every50 JSONL shard; pass exactly four",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--classifier-final", required=True, type=Path)
    parser.add_argument("--classifier-gold", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.input) != SHARD_COUNT:
        parser.error("exactly four --input values are required")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    aggregate = summarize(
        source=args.source,
        inputs=args.input,
        classifier_final=args.classifier_final,
        classifier_gold=args.classifier_gold,
        output_dir=args.output_dir,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
