"""Tests for Issue 3: Ingestion Validation and Pydantic Schema Contracts.

Verifies Session06RecordSchema validation, cross-field integrity rules,
and atomic write-time validation for JSON and JSONL study records.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

pytest.importorskip(
    "pydantic",
    reason="needs the optional pydantic dependency; see tests/README.md",
)
from pydantic import ValidationError

from src.schemas.user_study import (
    Session06RecordSchema,
    validate_session_record,
    write_session_record,
)


@pytest.fixture
def valid_record_dict():
    return {
        "participant_id": "P01",
        "scenario_id": "S-01",
        "scenario_mode": "controlled",
        "session_id": "01",
        "experiment_id": "session06_user_eval_v1",
        "session": 6,
        "date": "2026-10-04",
        "facilitator": "Yiwei Jin",
        "artifact": "docs/session06-log/01.json",
        "anchors_available": ["hiking", "Boulder", "grad student"],
        "anchors_used": ["hiking", "Boulder"],
        "initial_query": "I love hiking near Boulder as a grad student.",
        "query_formulation_seconds": 12.5,
        "evaluator_intervention_pre_submit": False,
        "detected_risk": True,
        "risk_score": 0.72,
        "system_state": "risk_positive_rewrite",
        "warning_in_their_words": "It warned me that my school and location could identify me.",
        "warning_comprehension": "correct",
        "requested_rewrite": True,
        "rewrite_changed_text": True,
        "rewrite_decision": "accept_as_is",
        "usefulness": 8,
        "would_use_before_sending": True,
        "adoption_reason": "Helps protect privacy without losing meaning.",
        "time_to_decision_seconds": 4.2,
        "comments": "Smooth participant interaction.",
        "protocol_deviation": False,
    }


def test_valid_record_instantiation(valid_record_dict):
    """Verify valid record dictionary validates cleanly into Session06RecordSchema."""
    rec = validate_session_record(valid_record_dict)
    assert rec.participant_id == "P01"
    assert rec.scenario_id == "S-01"
    assert rec.risk_score == 0.72
    assert rec.system_state == "risk_positive_rewrite"


def test_invalid_participant_id_rejected(valid_record_dict):
    """Verify participant_id not matching ^P\\d{2,}$ is rejected."""
    valid_record_dict["participant_id"] = "P1"  # Needs at least 2 digits
    with pytest.raises(ValidationError) as exc:
        validate_session_record(valid_record_dict)
    assert "participant_id" in str(exc.value)

    valid_record_dict["participant_id"] = "user_42"
    with pytest.raises(ValidationError):
        validate_session_record(valid_record_dict)


def test_invalid_system_state_rejected(valid_record_dict):
    """Verify non-conforming system state string is rejected."""
    valid_record_dict["system_state"] = "unknown_state_xyz"
    with pytest.raises(ValidationError) as exc:
        validate_session_record(valid_record_dict)
    assert "system_state" in str(exc.value)


def test_invalid_comprehension_rejected(valid_record_dict):
    """Verify invalid warning comprehension rating is rejected."""
    valid_record_dict["warning_comprehension"] = "partially_understood"
    with pytest.raises(ValidationError) as exc:
        validate_session_record(valid_record_dict)
    assert "warning_comprehension" in str(exc.value)


def test_anchors_used_subset_validation(valid_record_dict):
    """Verify anchors_used containing unknown anchors raises ValidationError."""
    valid_record_dict["anchors_available"] = ["Boulder", "hiking"]
    valid_record_dict["anchors_used"] = ["Boulder", "skiing_not_in_available"]
    with pytest.raises(ValidationError) as exc:
        validate_session_record(valid_record_dict)
    assert "anchors_used contains anchors not present" in str(exc.value)


def test_passthrough_state_constraints(valid_record_dict):
    """Verify risk_negative_passthrough prohibits detected_risk=True or rewrite changes."""
    valid_record_dict["system_state"] = "risk_negative_passthrough"
    valid_record_dict["detected_risk"] = True
    with pytest.raises(ValidationError) as exc:
        validate_session_record(valid_record_dict)
    assert "risk_negative_passthrough cannot have detected_risk=True" in str(exc.value)

    valid_record_dict["detected_risk"] = False
    valid_record_dict["rewrite_changed_text"] = True
    with pytest.raises(ValidationError) as exc:
        validate_session_record(valid_record_dict)
    assert "risk_negative_passthrough cannot have rewrite_changed_text=True" in str(exc.value)


def test_atomic_write_session_record_json(valid_record_dict):
    """Verify atomic write and append to JSON array file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "records.json"

        # First write
        saved1 = write_session_record(valid_record_dict, json_file)
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["participant_id"] == "P01"

        # Second write
        rec2 = dict(valid_record_dict)
        rec2["participant_id"] = "P02"
        saved2 = write_session_record(rec2, json_file)
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[1]["participant_id"] == "P02"


def test_atomic_write_session_record_jsonl(valid_record_dict):
    """Verify write to JSONL format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_file = Path(tmpdir) / "records.jsonl"

        write_session_record(valid_record_dict, jsonl_file)
        lines = [line.strip() for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["participant_id"] == "P01"


def test_write_session_record_rejects_malformed_before_write(valid_record_dict):
    """Verify write_session_record aborts without modifying file when payload is invalid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "records.json"
        valid_record_dict["risk_score"] = 5.0  # Invalid > 1.0

        with pytest.raises(ValidationError):
            write_session_record(valid_record_dict, json_file)

        assert not json_file.exists()


def test_committed_session06_record_conforms_to_schema():
    """Verify docs/userstudy/session06_record.json satisfies Session06RecordSchema."""
    log_file = Path("docs/userstudy/session06_record.json")
    assert log_file.exists()
    records = json.loads(log_file.read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert len(records) >= 3

    for idx, raw in enumerate(records):
        validated = validate_session_record(raw)
        assert validated.participant_id.startswith("P")
        assert validated.system_state in {
            "risk_positive_rewrite",
            "risk_positive_utility_conflict",
            "risk_negative_passthrough",
        }
