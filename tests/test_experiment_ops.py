from pathlib import Path

import yaml

from scripts.validate_experiment import (
    validate_config,
    validate_record,
)
from scripts.summarize_experiment import build_summary


def make_config():
    return {
        "experiment_id": "session06_user_eval_v1",
        "session": 6,
        "experiment_type": "user_evaluation",
        "random_seed": 42,
        "dataset": {
            "name": "RobinSta/SynthPAI",
            "revision": "TBD",
            "split": "validation",
        },
        "rewriter": {
            "name": "Qwen3-1.7B",
            "checkpoint": "TBD",
        },
        "user_study": {
            "core_scenarios": ["S-01", "S-02", "S-05"],
            "optional_scenarios": ["S-03", "S-04", "S-06"],
            "required_fields": [
                "participant_id",
                "scenario_id",
                "experiment_id",
                "session",
                "timestamp",
                "risk_understanding_correct",
                "risk_explanation",
                "requested_rewrite",
                "rewrite_decision",
                "usefulness",
                "would_use_before_sending",
                "time_to_decision_seconds",
                "qualitative_comment",
            ],
        },
        "metrics": {},
        "output": {
            "root": "experiments/session06_user_eval_v1",
        },
        "validation": {
            "scan_for_possible_pii": True,
        },
    }


def make_good_record():
    return {
        "participant_id": "P01",
        "scenario_id": "S-01",
        "experiment_id": "session06_user_eval_v1",
        "session": 6,
        "timestamp": "2026-09-29T12:00:00",
        "risk_understanding_correct": True,
        "risk_explanation": "The warning describes inferential privacy risk.",
        "requested_rewrite": True,
        "rewrite_decision": "would_use",
        "usefulness": 5,
        "would_use_before_sending": True,
        "time_to_decision_seconds": 30,
        "qualitative_comment": "Clear and useful.",
        "risk_before": 0.8,
        "risk_after": 0.3,
        "utility_score": 0.85,
        "cosine_similarity": 0.90,
        "nli_forward": 0.82,
        "nli_backward": 0.80,
        "contradiction": 0.05,
        "latency_s": 1.2,
    }


def test_config_passes_with_warnings():
    result = validate_config(make_config())

    assert result.ok
    assert len(result.errors) == 0

    warning_text = " ".join(result.warnings)

    assert "Dataset revision is still TBD" in warning_text
    assert "Rewriter checkpoint is still TBD" in warning_text


def test_valid_record_passes():
    config = make_config()
    record = make_good_record()

    result = validate_record(
        record=record,
        config=config,
        source="test.jsonl",
        index=1,
    )

    assert result.ok
    assert result.errors == []


def test_bad_participant_id_fails():
    config = make_config()
    record = make_good_record()
    record["participant_id"] = "Bingqi"

    result = validate_record(
        record=record,
        config=config,
        source="test.jsonl",
        index=1,
    )

    assert not result.ok

    assert any(
        "participant_id" in error
        for error in result.errors
    )


def test_unknown_scenario_fails():
    config = make_config()
    record = make_good_record()
    record["scenario_id"] = "S-99"

    result = validate_record(
        record=record,
        config=config,
        source="test.jsonl",
        index=1,
    )

    assert not result.ok

    assert any(
        "unknown scenario_id" in error
        for error in result.errors
    )


def test_session_mismatch_fails():
    config = make_config()
    record = make_good_record()
    record["session"] = 5

    result = validate_record(
        record=record,
        config=config,
        source="test.jsonl",
        index=1,
    )

    assert not result.ok

    assert any(
        "session mismatch" in error
        for error in result.errors
    )


def test_invalid_score_range_fails():
    config = make_config()
    record = make_good_record()
    record["risk_before"] = 1.2

    result = validate_record(
        record=record,
        config=config,
        source="test.jsonl",
        index=1,
    )

    assert not result.ok

    assert any(
        "outside expected [0, 1] range" in error
        for error in result.errors
    )


def test_negative_latency_fails():
    config = make_config()
    record = make_good_record()
    record["latency_s"] = -0.2

    result = validate_record(
        record=record,
        config=config,
        source="test.jsonl",
        index=1,
    )

    assert not result.ok

    assert any(
        "cannot be negative" in error
        for error in result.errors
    )


def test_summary_metrics_are_correct():
    config = make_config()

    record1 = make_good_record()

    record2 = make_good_record()
    record2["participant_id"] = "P02"
    record2["scenario_id"] = "S-02"
    record2["risk_understanding_correct"] = False
    record2["rewrite_decision"] = "would_not_use"
    record2["risk_before"] = 0.6
    record2["risk_after"] = 0.7
    record2["failure_mode"] = "F02"

    summary = build_summary(
        records=[record1, record2],
        config=config,
    )

    assert summary["record_count"] == 2
    assert summary["participant_count"] == 2

    decisions = summary["usability"]["decisions"]

    assert decisions["acceptance_rate"] == 0.5
    assert decisions["rejection_rate"] == 0.5

    comprehension = summary["usability"]["comprehension"]

    assert comprehension["comprehension_rate"] == 0.5

    privacy = summary["privacy"]

    # record1 delta = 0.5
    # record2 delta = -0.1
    # mean = 0.2
    assert privacy["risk_delta_mean"] == 0.2
    assert privacy["risk_inversion_count"] == 1
    assert privacy["risk_inversion_rate"] == 0.5

    failures = summary["reliability"]

    assert failures["records_with_failure"] == 1
    assert failures["failure_counts"]["F02"] == 1


def test_session06_config_file_loads():
    config_path = Path("configs/experiments/session06.yaml")

    assert config_path.exists()

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["experiment_id"] == "session06_user_eval_v1"
    assert config["session"] == 6
    assert config["experiment_type"] == "user_evaluation"
