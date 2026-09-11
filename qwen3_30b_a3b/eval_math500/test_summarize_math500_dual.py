from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import summarize_math500_dual as subject


class FakeModel:
    classes_ = [0, 1]
    n_features_in_ = 22

    def __init__(self, probability_by_probe: dict[int, float]) -> None:
        self.probability_by_probe = probability_by_probe

    def predict_proba(self, rows):
        probe_index = int(round(math.expm1(float(rows[0][-1]))))
        positive = self.probability_by_probe.get(probe_index, 0.1)
        return [[1.0 - positive, positive]]


def generation_config(shard_index: int) -> dict:
    return {
        "format_version": 1,
        "record_schema_version": subject.COLLECTOR_SCHEMA,
        "selection": {
            "shard_index": shard_index,
            "shard_count": 4,
            "limit": 0,
            "qid_count": 375,
        },
        "probe": {
            "schedule": "every_50_main_think_tokens_after_text_retokenization",
            "token_step": 50,
        },
        "features": {
            "implementation": "ClusterEvidenceTracker_answer_logprob_stats_v1",
            "order": list(subject.FEATURES),
            "count": 22,
            "raw_logprobs_saved": False,
        },
        "engine": {"model": "Qwen3-30B-A3B"},
    }


def features(probe_index: int) -> dict[str, float]:
    values = {name: 0.0 for name in subject.FEATURES}
    values["log1p_probe_index"] = math.log1p(probe_index)
    return values


def record(source_index: int, shard_index: int) -> dict:
    repeat_number = source_index // 500 + 1
    original_index = source_index % 500
    repeat_id = f"repeat{repeat_number}"
    original_qid = f"q{original_index:03d}"
    config = generation_config(shard_index)
    return {
        "format_version": 1,
        "record_schema_version": subject.COLLECTOR_SCHEMA,
        "generation_config": config,
        "generation_config_sha256": hashlib.sha256(
            subject.canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "qid": f"{repeat_id}::{original_qid}",
        "source_index": source_index,
        "sample_rank": repeat_id,
        "sample_stratum": "math500",
        "problem": f"problem {original_qid}",
        "gold_answer": str(original_index),
        "main": {
            "repeat_id": repeat_id,
            "original_qid": original_qid,
            "prediction": str(original_index),
            "reported_is_correct": True,
            "answer_valid": True,
            "response_tokens": 130,
        },
        "replay": {"reasoning_tokens": 120, "probe_count": 2},
        "probes": [
            {
                "probe_index": 1,
                "step_tokens": 50,
                "probe_answer": str(original_index),
                "eligible": True,
                "features": features(1),
            },
            {
                "probe_index": 2,
                "step_tokens": 100,
                "probe_answer": "wrong",
                "eligible": True,
                "features": features(2),
            },
        ],
    }


def write_shards(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    for shard_index in range(4):
        path = tmp_path / f"shard{shard_index}.jsonl"
        rows = [record(index, shard_index) for index in range(shard_index, 1500, 4)]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        paths.append(path)
    return paths


def classifier_pack(target: str, model: FakeModel) -> dict:
    return {
        "model": model,
        "feats": list(subject.FEATURES),
        "feature_names": tuple(subject.FEATURES),
        "tag": subject.CLASSIFIER_SCHEMA,
        "feature_schema": subject.FEATURE_SCHEMA,
        "feature_schema_sha256": subject.FEATURE_SCHEMA_SHA256,
        "target": target,
        "label_version": subject.LABEL_VERSION,
        "threshold": 0.95,
        "deployment_threshold": 0.95,
        "probe_stride_tokens": 50,
        "backbone_model": "/models/Qwen3-30B-A3B",
    }


def test_end_to_end_writes_repeat_decisions_and_t_ci(tmp_path: Path) -> None:
    paths = write_shards(tmp_path)
    final_path = tmp_path / "final.joblib"
    gold_path = tmp_path / "gold.joblib"
    final_path.write_bytes(b"final")
    gold_path.write_bytes(b"gold")
    packs = {
        final_path: classifier_pack("final_consistency", FakeModel({1: 0.96})),
        gold_path: classifier_pack("gold_safe", FakeModel({1: 0.94, 2: 0.97})),
    }
    output = tmp_path / "output"
    aggregate = subject.summarize(
        inputs=paths,
        classifier_final=final_path,
        classifier_gold=gold_path,
        output_dir=output,
        pack_loader=lambda path: packs[path],
        grader=lambda prediction, gold: prediction == gold,
    )

    assert aggregate["rows"] == 1500
    assert aggregate["run_level_ci95"]["final_consistency"]["coverage"]["mean"] == 1.0
    assert aggregate["run_level_ci95"]["final_consistency"]["coverage"]["df"] == 2
    assert aggregate["run_level_ci95"]["final_consistency"]["coverage"]["sample_sd"] == 0.0
    for repeat_id in subject.REPEATS:
        decisions = (output / repeat_id / "decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        summary = json.loads((output / repeat_id / "summary.json").read_text())
        assert len(decisions) == 500
        assert summary["baseline"]["accuracy"] == 1.0
        assert summary["final_consistency"]["coverage"] == 1.0
        assert summary["final_consistency"]["mean_primary_think_tokens"] == 50.0
        assert summary["final_consistency"]["paired_vs_baseline"] == {
            "accuracy_delta_pp": 0.0,
            "mean_primary_think_token_reduction": 70.0,
            "relative_mean_primary_think_token_reduction": 70.0 / 120.0,
            "correctness_wins": 0,
            "correctness_losses": 0,
        }
        assert summary["gold_safe"]["coverage"] == 1.0
        assert summary["gold_safe"]["accuracy"] == 0.0
        assert summary["gold_safe"]["mean_primary_think_tokens"] == 100.0
    assert (
        aggregate["run_level_ci95"]["final_consistency"]["mean_token_reduction"]["mean"]
        == 70.0
    )
    assert (
        aggregate["run_level_ci95"]["final_consistency"]["accuracy_delta_pp"]["mean"]
        == 0.0
    )


def test_no_hit_falls_back_to_baseline_answer_and_think_length(tmp_path: Path) -> None:
    row = record(0, 0)
    bundle = subject.ClassifierBundle(
        target="gold_safe",
        model=FakeModel({1: 0.94, 2: 0.949}),
        positive_index=1,
        path=tmp_path / "unused",
    )
    decision = subject._policy_decision(
        row, bundle, grade=lambda prediction, gold: prediction == gold
    )
    assert decision["stopped"] is False
    assert decision["prediction"] == row["main"]["prediction"]
    assert decision["primary_think_tokens"] == 120
    assert decision["eligible_probes_scored_to_decision"] == 2


def test_earliest_hit_skips_ineligible_probe(tmp_path: Path) -> None:
    row = record(0, 0)
    row["probes"][0]["eligible"] = False
    bundle = subject.ClassifierBundle(
        target="final_consistency",
        model=FakeModel({1: 0.99, 2: 0.96}),
        positive_index=1,
        path=tmp_path / "unused",
    )
    decision = subject._policy_decision(
        row, bundle, grade=lambda prediction, gold: prediction == gold
    )
    assert decision["stop_probe_index"] == 2
    assert decision["stop_step_tokens"] == 100
    assert decision["eligible_probes_scored_to_decision"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda pack: pack.update(tag="wrong"), "training schema"),
        (lambda pack: pack.update(threshold=0.94), "threshold/stride"),
        (lambda pack: pack.update(deployment_threshold=0.94), "threshold/stride"),
        (lambda pack: pack.update(probe_stride_tokens=20), "threshold/stride"),
        (lambda pack: pack.update(feats=list(reversed(subject.FEATURES))), "exact ordered"),
    ],
)
def test_classifier_pack_rejects_contract_drift(tmp_path: Path, mutation, match: str) -> None:
    pack = classifier_pack("gold_safe", FakeModel({}))
    mutation(pack)
    with pytest.raises(ValueError, match=match):
        subject.validate_classifier_pack(
            tmp_path / "pack.joblib", "gold_safe", loader=lambda _path: pack
        )


def test_positive_class_index_follows_model_classes(tmp_path: Path) -> None:
    model = FakeModel({})
    model.classes_ = [1, 0]
    pack = classifier_pack("final_consistency", model)
    bundle = subject.validate_classifier_pack(
        tmp_path / "pack.joblib", "final_consistency", loader=lambda _path: pack
    )
    assert bundle.positive_index == 0


def test_loader_rejects_joint_collector_feature_drift(tmp_path: Path) -> None:
    paths = write_shards(tmp_path)
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    drifted = list(subject.FEATURES)
    drifted[-1] = "not_cluster22"
    rows[0]["generation_config"]["features"]["order"] = drifted
    rows[0]["generation_config_sha256"] = hashlib.sha256(
        subject.canonical_json(rows[0]["generation_config"]).encode()
    ).hexdigest()
    paths[0].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="generation feature contract"):
        subject.load_collector_shards(paths)


def test_loader_rejects_duplicate_prefixed_qid(tmp_path: Path) -> None:
    paths = write_shards(tmp_path)
    rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
    rows[0]["qid"] = "repeat1::q000"
    rows[0]["main"]["repeat_id"] = "repeat1"
    rows[0]["main"]["original_qid"] = "q000"
    rows[0]["sample_rank"] = "repeat1"
    paths[1].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="1500 unique"):
        subject.load_collector_shards(paths)


def test_student_t_df2_interval() -> None:
    result = subject.student_t_ci_df2([0.8, 0.9, 1.0])
    expected_margin = subject.T95_DF2 * 0.1 / math.sqrt(3)
    assert result["mean"] == pytest.approx(0.9)
    assert result["sample_sd"] == pytest.approx(0.1)
    assert result["ci95_low"] == pytest.approx(0.9 - expected_margin)
    assert result["ci95_high"] == pytest.approx(0.9 + expected_margin)
