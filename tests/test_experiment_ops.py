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


# ---- Session 06 v2 schema-alignment tests ----

def make_session06_v2_config():
    return {
        "experiment_id": "session06_user_eval_v1",
        "session": 6,
        "experiment_type": "user_evaluation",
        "schema_version": "session06_v2",
        "random_seed": 42,
        "dataset": {
            "name": "RobinSta/SynthPAI",
            "revision": "TBD",
            "split": "validation",
        },
        "rewriter": {
            "name": "Qwen3-1.7B",
            "checkpoint": "artifacts/rewriter_qlora",
        },
        "user_study": {
            "participant_id_regex": r"^P\d{2,}$",
            "core_scenarios": [
                "S-01",
                "S-02",
                "S-05",
            ],
            "optional_scenarios": [
                "S-03",
                "S-04",
                "S-06",
            ],
            "scenario_modes": {
                "S-01": "generative",
                "S-02": "controlled",
                "S-03": "generative",
                "S-04": "generative",
                "S-05": "controlled",
                "S-06": "staged_multiturn",
            },
            "valid_system_states": [
                "risk_positive_rewrite",
                "risk_positive_utility_conflict",
                "risk_negative_passthrough",
            ],
            "valid_warning_comprehension": [
                "correct",
                "partial",
                "incorrect",
                "echo",
            ],
            "valid_rewrite_decisions": [
                "accept_as_is",
                "accept_with_edits",
                "reject",
                "not_applicable",
            ],
            "required_fields": [
                "participant_id",
                "scenario_id",
                "scenario_mode",
                "experiment_id",
                "session",
                "date",
                "facilitator",
                "artifact",
                "anchors_available",
                "anchors_used",
                "initial_query",
                "query_formulation_seconds",
                "evaluator_intervention_pre_submit",
                "detected_risk",
                "risk_score",
                "system_state",
                "warning_in_their_words",
                "warning_comprehension",
                "requested_rewrite",
                "rewrite_changed_text",
                "rewrite_decision",
                "usefulness",
                "would_use_before_sending",
                "adoption_reason",
                "time_to_decision_seconds",
                "comments",
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


def make_session06_v2_record():
    return {
        "participant_id": "P01",
        "scenario_id": "S-01",
        "scenario_mode": "generative",
        "experiment_id": "session06_user_eval_v1",
        "session": 6,
        "date": "2026-10-05",
        "facilitator": "test-user",
        "artifact": "user_studies/session06/P01.mp4",

        "anchors_available": [
            "education",
            "location",
            "commute",
            "co_op",
        ],
        "anchors_used": [
            "education",
            "commute",
        ],

        "initial_query": (
            "I am finishing my degree and have a long commute. "
            "Should I move before my next placement?"
        ),
        "query_formulation_seconds": 18,

        "evaluator_intervention_pre_submit": False,

        "detected_risk": True,
        "risk_score": 0.70,
        "system_state": "risk_positive_rewrite",

        "warning_in_their_words": (
            "It says my education and commute details "
            "could reveal information about me."
        ),
        "warning_comprehension": "correct",

        "requested_rewrite": True,
        "rewrite_changed_text": True,
        "rewrite_decision": "accept_as_is",

        "usefulness": 7,
        "would_use_before_sending": True,
        "adoption_reason": "It reduces unnecessary personal detail.",
        "time_to_decision_seconds": 30,
        "comments": "",

        # Existing model-evaluation fields remain supported.
        "risk_before": 0.80,
        "risk_after": 0.30,
        "privacy_risk_delta": 0.50,
        "utility_score": 0.85,
        "cosine_similarity": 0.90,
        "nli_forward": 0.82,
        "nli_backward": 0.80,
        "contradiction": 0.05,
        "latency_s": 1.2,
    }


def test_session06_v2_valid_record_passes():
    config = make_session06_v2_config()
    record = make_session06_v2_record()

    result = validate_record(
        record=record,
        config=config,
        source="session06.json",
        index=1,
    )

    assert result.ok
    assert result.errors == []


def test_participant_regex_is_configurable():
    config = make_session06_v2_config()

    config["user_study"][
        "participant_id_regex"
    ] = r"^anonymous-\d{2}$"

    record = make_session06_v2_record()
    record["participant_id"] = "anonymous-01"

    result = validate_record(
        record=record,
        config=config,
        source="session06.json",
        index=1,
    )

    assert result.ok


def test_anchor_subset_violation_fails():
    config = make_session06_v2_config()
    record = make_session06_v2_record()

    record["anchors_used"].append(
        "unknown_anchor"
    )

    result = validate_record(
        record=record,
        config=config,
        source="session06.json",
        index=1,
    )

    assert not result.ok

    assert any(
        "anchors_used contains anchors"
        in error
        for error in result.errors
    )


def test_session06_invalid_enum_values_fail():
    config = make_session06_v2_config()
    record = make_session06_v2_record()

    record["warning_comprehension"] = "mostly"
    record["system_state"] = "maybe_safe"
    record["rewrite_decision"] = "probably_accept"

    result = validate_record(
        record=record,
        config=config,
        source="session06.json",
        index=1,
    )

    assert not result.ok

    error_text = " ".join(
        result.errors
    )

    assert (
        "invalid warning_comprehension"
        in error_text
    )
    assert (
        "invalid system_state"
        in error_text
    )
    assert (
        "invalid rewrite_decision"
        in error_text
    )


def test_zero_risk_passthrough_semantics_enforced():
    config = make_session06_v2_config()
    record = make_session06_v2_record()

    record["detected_risk"] = False
    record["risk_score"] = 0.01
    record["system_state"] = (
        "risk_negative_passthrough"
    )
    record["requested_rewrite"] = False
    record["rewrite_changed_text"] = True
    record["rewrite_decision"] = (
        "not_applicable"
    )

    result = validate_record(
        record=record,
        config=config,
        source="session06.json",
        index=1,
    )

    assert not result.ok

    assert any(
        "rewrite_changed_text=true"
        in error
        for error in result.errors
    )


def test_protocol_deviation_requires_metadata():
    config = make_session06_v2_config()
    record = make_session06_v2_record()

    record["protocol_deviation"] = True

    result = validate_record(
        record=record,
        config=config,
        source="session06.json",
        index=1,
    )

    assert not result.ok

    error_text = " ".join(
        result.errors
    )

    assert "deviation_type" in error_text
    assert "description" in error_text


def test_session06_summary_metrics():
    config = make_session06_v2_config()

    # Generative: ADR = 2 / 4 = 0.50
    record1 = make_session06_v2_record()

    # Controlled benchmark:
    # must not enter the ADR denominator.
    record2 = make_session06_v2_record()
    record2["participant_id"] = "P02"
    record2["scenario_id"] = "S-02"
    record2["scenario_mode"] = "controlled"
    record2["anchors_available"] = [
        "occupation",
        "career_stage",
        "education",
    ]
    record2["anchors_used"] = [
        "occupation",
        "career_stage",
        "education",
    ]
    record2["system_state"] = (
        "risk_positive_utility_conflict"
    )
    record2["warning_comprehension"] = (
        "partial"
    )
    record2["rewrite_decision"] = "reject"
    record2[
        "would_use_before_sending"
    ] = False
    record2["protocol_deviation"] = True
    record2["deviation_type"] = (
        "system_malfunction"
    )
    record2["description"] = (
        "Rewrite was not displayed correctly."
    )

    # Generative State C:
    # ADR = 1 / 4 = 0.25
    record3 = make_session06_v2_record()
    record3["participant_id"] = "P03"
    record3["scenario_id"] = "S-03"
    record3["scenario_mode"] = "generative"
    record3["anchors_available"] = [
        "age",
        "medication",
        "location",
        "provider_context",
    ]
    record3["anchors_used"] = [
        "medication",
    ]
    record3["detected_risk"] = False
    record3["risk_score"] = 0.01
    record3["system_state"] = (
        "risk_negative_passthrough"
    )
    record3["warning_comprehension"] = (
        "echo"
    )
    record3["requested_rewrite"] = False
    record3["rewrite_changed_text"] = False
    record3["rewrite_decision"] = (
        "not_applicable"
    )
    record3[
        "would_use_before_sending"
    ] = False
    record3["risk_before"] = 0.01
    record3["risk_after"] = 0.01
    record3["privacy_risk_delta"] = 0.0

    # Explicitly coded by researcher.
    record3["failure_mode"] = "F08"

    summary = build_summary(
        records=[
            record1,
            record2,
            record3,
        ],
        config=config,
    )

    # Controlled S-02 excluded:
    # mean ADR = (0.50 + 0.25) / 2
    assert (
        summary["disclosure"][
            "eligible_record_count"
        ]
        == 2
    )

    assert (
        summary["disclosure"][
            "anchor_disclosure_rate_mean"
        ]
        == 0.375
    )

    decisions = summary[
        "usability"
    ]["decisions"]

    # P03 not_applicable is not counted in
    # acceptance/rejection denominator.
    assert (
        decisions["total_decisions"]
        == 2
    )
    assert (
        decisions["not_applicable_count"]
        == 1
    )
    assert (
        decisions["acceptance_rate"]
        == 0.5
    )
    assert (
        decisions["rejection_rate"]
        == 0.5
    )

    comprehension = summary[
        "usability"
    ]["comprehension"]

    assert (
        comprehension["schema"]
        == "session06_categorical"
    )
    assert (
        comprehension["correct_count"]
        == 1
    )
    assert (
        comprehension["partial_count"]
        == 1
    )
    assert (
        comprehension["echo_count"]
        == 1
    )
    assert (
        comprehension["comprehension_rate"]
        == 0.3333
    )

    adoption = summary[
        "usability"
    ]["adoption"]

    assert (
        adoption["adoption_rate"]
        == 0.3333
    )

    reliability = summary[
        "reliability"
    ]

    assert (
        reliability[
            "zero_risk_passthrough_count"
        ]
        == 1
    )
    assert (
        reliability[
            "zero_risk_passthrough_rate"
        ]
        == 0.3333
    )

    assert (
        reliability[
            "protocol_deviation_count"
        ]
        == 1
    )
    assert (
        reliability[
            "protocol_deviation_rate"
        ]
        == 0.3333
    )

    assert (
        reliability[
            "trust_calibration_failure_count"
        ]
        == 1
    )


def test_session06_multiturn_summary():
    config = make_session06_v2_config()
    record = make_session06_v2_record()

    record["participant_id"] = "P04"
    record["scenario_id"] = "S-06"
    record["scenario_mode"] = (
        "staged_multiturn"
    )

    record["turns"] = [
        {
            "turn_level_risk": 0.10,
            "cumulative_risk": 0.10,
        },
        {
            "turn_level_risk": 0.30,
            "cumulative_risk": 0.35,
        },
        {
            "turn_level_risk": 0.45,
            "cumulative_risk": 0.60,
        },
        {
            "turn_level_risk": 0.50,
            "cumulative_risk": 0.80,
        },
    ]

    summary = build_summary(
        records=[record],
        config=config,
    )

    multiturn = summary["multiturn"]

    assert multiturn["record_count"] == 1
    assert (
        multiturn["turn_level_risk_max"]
        == 0.5
    )
    assert (
        multiturn["cumulative_risk_max"]
        == 0.8
    )


def test_actual_session06_config_matches_v2_contract():
    config_path = Path(
        "configs/experiments/session06.yaml"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    assert (
        config["schema_version"]
        == "session06_v2"
    )

    assert (
        config["rewriter"]["checkpoint"]
        == "artifacts/rewriter_qlora"
    )

    required = config[
        "user_study"
    ]["required_fields"]

    assert "date" in required
    assert "timestamp" not in required
    assert "anchors_available" in required
    assert "anchors_used" in required
    assert "system_state" in required
    assert "warning_comprehension" in required

    result = validate_config(config)

    assert result.ok

    warning_text = " ".join(
        result.warnings
    )

    assert (
        "Dataset revision is still TBD"
        in warning_text
    )

    assert (
        "Rewriter checkpoint is still TBD"
        not in warning_text
    )
