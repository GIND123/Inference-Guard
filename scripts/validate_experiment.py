"""
Validation utilities for InferenceGuard experiment artifacts.

This script validates:
1. experiment configuration
2. user-study JSON / JSONL records
3. consistency between records and the experiment config
4. basic privacy and numerical sanity checks

Usage:

    # Validate only the experiment configuration
    python scripts/validate_experiment.py \
        --config configs/experiments/session06.yaml \
        --config-only

    # Validate one JSONL file
    python scripts/validate_experiment.py \
        --config configs/experiments/session06.yaml \
        --input experiments/session06_user_eval_v1/raw/P01.jsonl

    # Validate all JSON / JSONL files in a directory
    python scripts/validate_experiment.py \
        --config configs/experiments/session06.yaml \
        --input experiments/session06_user_eval_v1/raw/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


PARTICIPANT_RE = re.compile(r"^P\d{2,}$")

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}"
)

VALID_REWRITE_DECISIONS = {
    "use_as_is",
    "accept",
    "accept_with_edits",
    "use_with_edits",
    "reject",
    "would_use",
    "would_use_with_edits",
    "would_not_use",
}

SCORE_FIELDS = {
    "risk_before",
    "risk_after",
    "privacy_risk_delta",
    "utility_score",
    "cosine_similarity",
    "nli_forward",
    "nli_backward",
    "contradiction",
}

NONNEGATIVE_FIELDS = {
    "latency",
    "latency_s",
    "time_to_decision_seconds",
}


class ValidationResult:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")

    return data


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    result = ValidationResult()

    required_top_level = {
        "experiment_id",
        "session",
        "experiment_type",
        "random_seed",
        "metrics",
        "output",
        "validation",
    }

    for key in sorted(required_top_level):
        if key not in config:
            result.error(f"Config missing required field: {key}")

    experiment_id = config.get("experiment_id")
    session = config.get("session")

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        result.error("experiment_id must be a non-empty string.")

    if not isinstance(session, int) or session <= 0:
        result.error("session must be a positive integer.")

    # User-evaluation-specific validation
    if config.get("experiment_type") == "user_evaluation":
        user_cfg = config.get("user_study")

        if not isinstance(user_cfg, dict):
            result.error(
                "user_study section is required for user_evaluation experiments."
            )
        else:
            core = user_cfg.get("core_scenarios", [])
            optional = user_cfg.get("optional_scenarios", [])
            required_fields = user_cfg.get("required_fields", [])

            if not core:
                result.error("user_study.core_scenarios cannot be empty.")

            all_scenarios = core + optional

            duplicates = sorted(
                {x for x in all_scenarios if all_scenarios.count(x) > 1}
            )
            if duplicates:
                result.error(
                    f"Scenario IDs appear more than once: {duplicates}"
                )

            if not required_fields:
                result.error(
                    "user_study.required_fields cannot be empty."
                )

    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        if dataset.get("revision") == "TBD":
            result.warning(
                "Dataset revision is still TBD. "
                "Do not run final evaluation until it is pinned."
            )

    rewriter = config.get("rewriter", {})
    if isinstance(rewriter, dict):
        if rewriter.get("checkpoint") == "TBD":
            result.warning(
                "Rewriter checkpoint is still TBD."
            )

    return result


def allowed_scenarios(config: Dict[str, Any]) -> set[str]:
    user_cfg = config.get("user_study", {})

    core = user_cfg.get("core_scenarios", [])
    optional = user_cfg.get("optional_scenarios", [])

    return set(core + optional)


def required_record_fields(config: Dict[str, Any]) -> List[str]:
    user_cfg = config.get("user_study", {})
    return list(user_cfg.get("required_fields", []))


def iter_text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value

    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_text_values(v)

    elif isinstance(value, list):
        for v in value:
            yield from iter_text_values(v)


def contains_possible_pii(record: Dict[str, Any]) -> List[str]:
    findings: List[str] = []

    for text in iter_text_values(record):
        if EMAIL_RE.search(text):
            findings.append("possible email address")

        if PHONE_RE.search(text):
            findings.append("possible US phone number")

    return sorted(set(findings))


def validate_score(
    field: str,
    value: Any,
    result: ValidationResult,
    prefix: str,
) -> None:
    if not isinstance(value, (int, float)):
        result.error(f"{prefix}{field} must be numeric.")
        return

    # Delta may legitimately be negative when risk increases.
    if field == "privacy_risk_delta":
        if not -1.0 <= float(value) <= 1.0:
            result.error(
                f"{prefix}{field}={value} outside expected [-1, 1] range."
            )
        return

    if not 0.0 <= float(value) <= 1.0:
        result.error(
            f"{prefix}{field}={value} outside expected [0, 1] range."
        )


def validate_record(
    record: Dict[str, Any],
    config: Dict[str, Any],
    source: str,
    index: int,
) -> ValidationResult:
    result = ValidationResult()
    prefix = f"{source} record {index}: "

    for field in required_record_fields(config):
        if field not in record:
            result.error(f"{prefix}missing required field '{field}'.")

    participant_id = record.get("participant_id")

    if participant_id is not None:
        if not isinstance(participant_id, str):
            result.error(
                f"{prefix}participant_id must be a string."
            )
        elif not PARTICIPANT_RE.fullmatch(participant_id):
            result.error(
                f"{prefix}participant_id '{participant_id}' "
                "must follow a pseudonymous format such as P01."
            )

    scenario_id = record.get("scenario_id")

    if scenario_id is not None:
        allowed = allowed_scenarios(config)

        if scenario_id not in allowed:
            result.error(
                f"{prefix}unknown scenario_id '{scenario_id}'. "
                f"Allowed: {sorted(allowed)}"
            )

    expected_experiment = config.get("experiment_id")
    actual_experiment = record.get("experiment_id")

    if (
        actual_experiment is not None
        and actual_experiment != expected_experiment
    ):
        result.error(
            f"{prefix}experiment_id mismatch: "
            f"'{actual_experiment}' != '{expected_experiment}'."
        )

    expected_session = config.get("session")
    actual_session = record.get("session")

    if (
        actual_session is not None
        and actual_session != expected_session
    ):
        result.error(
            f"{prefix}session mismatch: "
            f"{actual_session} != {expected_session}."
        )

    rewrite_decision = record.get("rewrite_decision")

    if (
        rewrite_decision is not None
        and rewrite_decision not in VALID_REWRITE_DECISIONS
    ):
        result.warning(
            f"{prefix}unrecognized rewrite_decision "
            f"'{rewrite_decision}'."
        )

    for field in SCORE_FIELDS:
        if field in record and record[field] is not None:
            validate_score(
                field,
                record[field],
                result,
                prefix,
            )

    for field in NONNEGATIVE_FIELDS:
        if field in record and record[field] is not None:
            value = record[field]

            if not isinstance(value, (int, float)):
                result.error(
                    f"{prefix}{field} must be numeric."
                )

            elif float(value) < 0:
                result.error(
                    f"{prefix}{field} cannot be negative."
                )

    if config.get("validation", {}).get(
        "scan_for_possible_pii", False
    ):
        pii_findings = contains_possible_pii(record)

        for finding in pii_findings:
            result.warning(
                f"{prefix}{finding} detected. "
                "Confirm that this comes only from an approved synthetic scenario."
            )

    return result


def read_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError(
        f"{path} must contain a JSON object or list of objects."
    )


def read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: "
                    "each JSONL line must contain an object."
                )

            records.append(record)

    return records


def load_records(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    files: List[Path]

    if path.is_dir():
        files = sorted(
            [
                p
                for p in path.iterdir()
                if p.suffix.lower() in {".json", ".jsonl"}
            ]
        )
    else:
        files = [path]

    if not files:
        raise ValueError(
            f"No JSON or JSONL experiment files found in {path}."
        )

    output: List[Tuple[str, Dict[str, Any]]] = []

    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            records = read_jsonl_file(file_path)
        else:
            records = read_json_file(file_path)

        for record in records:
            output.append((str(file_path), record))

    return output


def print_result(
    title: str,
    result: ValidationResult,
) -> None:
    print(f"\n=== {title} ===")

    if result.ok:
        print("PASS")
    else:
        print("FAIL")

    for warning in result.warnings:
        print(f"WARNING: {warning}")

    for error in result.errors:
        print(f"ERROR: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate InferenceGuard experiment artifacts."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config.",
    )

    parser.add_argument(
        "--input",
        help="JSON/JSONL file or directory containing experiment records.",
    )

    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate only the experiment configuration.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 2

    try:
        config = load_yaml(config_path)
    except Exception as exc:
        print(f"ERROR: Failed to load config: {exc}")
        return 2

    config_result = validate_config(config)
    print_result("Experiment config", config_result)

    overall_ok = config_result.ok

    if args.config_only:
        return 0 if overall_ok else 1

    if not args.input:
        print(
            "\nERROR: --input is required unless --config-only is used."
        )
        return 2

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"\nERROR: Input path not found: {input_path}")
        return 2

    try:
        records = load_records(input_path)
    except Exception as exc:
        print(f"\nERROR: Failed to load records: {exc}")
        return 2

    print(f"\nLoaded {len(records)} experiment record(s).")

    total_errors = 0
    total_warnings = 0

    seen_participant_scenarios: set[Tuple[str, str]] = set()

    for i, (source, record) in enumerate(records, start=1):
        record_result = validate_record(
            record=record,
            config=config,
            source=source,
            index=i,
        )

        participant = record.get("participant_id")
        scenario = record.get("scenario_id")

        if participant and scenario:
            pair = (participant, scenario)

            if pair in seen_participant_scenarios:
                record_result.warning(
                    f"{source} record {i}: duplicate "
                    f"participant/scenario pair {pair}."
                )

            seen_participant_scenarios.add(pair)

        total_errors += len(record_result.errors)
        total_warnings += len(record_result.warnings)

        for warning in record_result.warnings:
            print(f"WARNING: {warning}")

        for error in record_result.errors:
            print(f"ERROR: {error}")

    print("\n=== Validation summary ===")
    print(f"Records: {len(records)}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warnings}")

    if total_errors == 0 and overall_ok:
        print("PASS: Experiment artifacts passed validation.")
        return 0

    print("FAIL: Fix validation errors before analysis.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
