"""
Summarize validated InferenceGuard experiment records.

Supports both the original experiment-operations schema and the
Session 06 goal-driven user-study schema.

Produces:
- summary.json
- metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


ACCEPT_DECISIONS = {
    "accept",
    "use_as_is",
    "would_use",
    "accept_as_is",
}

EDIT_DECISIONS = {
    "accept_with_edits",
    "use_with_edits",
    "would_use_with_edits",
}

REJECT_DECISIONS = {
    "reject",
    "would_not_use",
}

NOT_APPLICABLE_DECISIONS = {
    "not_applicable",
}

COMPREHENSION_LABELS = {
    "correct",
    "partial",
    "incorrect",
    "echo",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path} does not contain a YAML mapping."
        )

    return data


def read_json_file(
    path: Path,
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError(
        f"{path} must contain a JSON object or list of objects."
    )


def read_jsonl_file(
    path: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: "
                    "each JSONL line must contain an object."
                )

            records.append(record)

    return records


def load_records(
    path: Path,
) -> List[Dict[str, Any]]:
    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.suffix.lower()
            in {".json", ".jsonl"}
        )
    else:
        files = [path]

    if not files:
        raise ValueError(
            f"No JSON or JSONL files found in {path}."
        )

    records: List[Dict[str, Any]] = []

    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            records.extend(
                read_jsonl_file(file_path)
            )
        else:
            records.extend(
                read_json_file(file_path)
            )

    return records


def numeric_values(
    records: Iterable[Dict[str, Any]],
    field: str,
) -> List[float]:
    values: List[float] = []

    for record in records:
        value = record.get(field)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            values.append(float(value))

    return values


def safe_mean(
    values: List[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.mean(values),
        4,
    )


def safe_median(
    values: List[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.median(values),
        4,
    )


def rate(
    count: int,
    total: int,
) -> float | None:
    if total == 0:
        return None

    return round(
        count / total,
        4,
    )


def summarize_decisions(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    counts: Counter[str] = Counter()

    for record in records:
        decision = record.get(
            "rewrite_decision"
        )

        if decision in ACCEPT_DECISIONS:
            counts["accept"] += 1

        elif decision in EDIT_DECISIONS:
            counts["accept_with_edits"] += 1

        elif decision in REJECT_DECISIONS:
            counts["reject"] += 1

        elif decision in NOT_APPLICABLE_DECISIONS:
            counts["not_applicable"] += 1

        elif decision is not None:
            counts["other"] += 1

    applicable_total = (
        counts["accept"]
        + counts["accept_with_edits"]
        + counts["reject"]
        + counts["other"]
    )

    observed_total = (
        applicable_total
        + counts["not_applicable"]
    )

    return {
        "total_decisions": applicable_total,
        "observed_decisions": observed_total,
        "accept_count": counts["accept"],
        "accept_with_edits_count":
            counts["accept_with_edits"],
        "reject_count": counts["reject"],
        "not_applicable_count":
            counts["not_applicable"],
        "other_count": counts["other"],
        "acceptance_rate": rate(
            counts["accept"]
            + counts["accept_with_edits"],
            applicable_total,
        ),
        "accept_as_is_rate": rate(
            counts["accept"],
            applicable_total,
        ),
        "edit_rate": rate(
            counts["accept_with_edits"],
            applicable_total,
        ),
        "rejection_rate": rate(
            counts["reject"],
            applicable_total,
        ),
    }


def summarize_comprehension(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Prefer Session 06 categorical labels.

    Fall back to the legacy
    risk_understanding_correct boolean when no
    categorical labels are available.
    """
    labels: List[str] = []

    for record in records:
        value = record.get(
            "warning_comprehension"
        )

        if (
            isinstance(value, str)
            and value in COMPREHENSION_LABELS
        ):
            labels.append(value)

    if labels:
        counts = Counter(labels)
        total = len(labels)

        return {
            "schema": "session06_categorical",
            "evaluated_count": total,
            "correct_count": counts["correct"],
            "partial_count": counts["partial"],
            "incorrect_count": counts["incorrect"],
            "echo_count": counts["echo"],
            "comprehension_rate": rate(
                counts["correct"],
                total,
            ),
            "distribution": {
                label: {
                    "count": counts[label],
                    "rate": rate(
                        counts[label],
                        total,
                    ),
                }
                for label in sorted(
                    COMPREHENSION_LABELS
                )
            },
        }

    legacy_values = [
        record.get(
            "risk_understanding_correct"
        )
        for record in records
        if isinstance(
            record.get(
                "risk_understanding_correct"
            ),
            bool,
        )
    ]

    if not legacy_values:
        return {
            "schema": None,
            "evaluated_count": 0,
            "correct_count": 0,
            "comprehension_rate": None,
            "distribution": {},
        }

    correct = sum(legacy_values)

    return {
        "schema": "legacy_boolean",
        "evaluated_count":
            len(legacy_values),
        "correct_count": correct,
        "comprehension_rate": rate(
            correct,
            len(legacy_values),
        ),
        "distribution": {
            "correct": {
                "count": correct,
                "rate": rate(
                    correct,
                    len(legacy_values),
                ),
            },
            "incorrect": {
                "count":
                    len(legacy_values) - correct,
                "rate": rate(
                    len(legacy_values) - correct,
                    len(legacy_values),
                ),
            },
        },
    }


def summarize_disclosure(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute Anchor Disclosure Rate (ADR).

    Per protocol, ADR is primarily meaningful for
    generative or staged scenarios, not controlled
    benchmark prompts.
    """
    adr_values: List[float] = []
    per_record: List[Dict[str, Any]] = []

    for record in records:
        mode = record.get("scenario_mode")

        if mode == "controlled":
            continue

        available = record.get(
            "anchors_available"
        )
        used = record.get(
            "anchors_used"
        )

        if (
            not isinstance(available, list)
            or not isinstance(used, list)
        ):
            continue

        available_set = {
            str(x)
            for x in available
        }

        used_set = {
            str(x)
            for x in used
        }

        if not available_set:
            continue

        disclosed = (
            used_set & available_set
        )

        adr = (
            len(disclosed)
            / len(available_set)
        )

        adr_values.append(adr)

        per_record.append(
            {
                "participant_id":
                    record.get(
                        "participant_id"
                    ),
                "scenario_id":
                    record.get(
                        "scenario_id"
                    ),
                "anchors_available":
                    len(available_set),
                "anchors_used":
                    len(disclosed),
                "anchor_disclosure_rate":
                    round(adr, 4),
            }
        )

    return {
        "eligible_record_count":
            len(adr_values),
        "anchor_disclosure_rate_mean":
            safe_mean(adr_values),
        "anchor_disclosure_rate_median":
            safe_median(adr_values),
        "per_record": per_record,
    }


def summarize_detection(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    detected_values = [
        record.get("detected_risk")
        for record in records
        if isinstance(
            record.get("detected_risk"),
            bool,
        )
    ]

    detected_count = sum(
        detected_values
    )

    risk_scores = numeric_values(
        records,
        "risk_score",
    )

    return {
        "evaluated_count":
            len(detected_values),
        "detected_risk_count":
            detected_count,
        "detected_risk_rate": rate(
            detected_count,
            len(detected_values),
        ),
        "risk_score_mean":
            safe_mean(risk_scores),
        "risk_score_median":
            safe_median(risk_scores),
    }


def summarize_system_states(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    counter: Counter[str] = Counter()

    for record in records:
        state = record.get(
            "system_state"
        )

        if isinstance(state, str):
            counter[state] += 1

    total = sum(counter.values())

    return {
        "evaluated_count": total,
        "counts": dict(
            sorted(counter.items())
        ),
        "rates": {
            state: rate(
                count,
                total,
            )
            for state, count
            in sorted(counter.items())
        },
        "zero_risk_passthrough_count":
            counter[
                "risk_negative_passthrough"
            ],
        "zero_risk_passthrough_rate":
            rate(
                counter[
                    "risk_negative_passthrough"
                ],
                total,
            ),
    }


def summarize_adoption(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    values = [
        record.get(
            "would_use_before_sending"
        )
        for record in records
        if isinstance(
            record.get(
                "would_use_before_sending"
            ),
            bool,
        )
    ]

    yes = sum(values)

    return {
        "evaluated_count": len(values),
        "would_use_count": yes,
        "would_not_use_count":
            len(values) - yes,
        "adoption_rate": rate(
            yes,
            len(values),
        ),
    }


def summarize_privacy(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    before = numeric_values(
        records,
        "risk_before",
    )

    after = numeric_values(
        records,
        "risk_after",
    )

    explicit_delta = numeric_values(
        records,
        "privacy_risk_delta",
    )

    paired_deltas: List[float] = []

    for record in records:
        b = record.get("risk_before")
        a = record.get("risk_after")

        if (
            isinstance(b, (int, float))
            and not isinstance(b, bool)
            and isinstance(a, (int, float))
            and not isinstance(a, bool)
        ):
            paired_deltas.append(
                float(b) - float(a)
            )

    deltas = (
        explicit_delta
        or paired_deltas
    )

    risk_inversions = sum(
        1
        for value in paired_deltas
        if value < 0
    )

    return {
        "risk_before_mean":
            safe_mean(before),
        "risk_after_mean":
            safe_mean(after),
        "risk_delta_mean":
            safe_mean(deltas),
        "risk_delta_median":
            safe_median(deltas),
        "risk_inversion_count":
            risk_inversions,
        "risk_inversion_rate":
            rate(
                risk_inversions,
                len(paired_deltas),
            ),
    }


def summarize_utility(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fields = [
        "utility_score",
        "cosine_similarity",
        "nli_forward",
        "nli_backward",
        "contradiction",
    ]

    output: Dict[str, Any] = {}

    for field in fields:
        values = numeric_values(
            records,
            field,
        )

        output[
            f"{field}_mean"
        ] = safe_mean(values)

        output[
            f"{field}_median"
        ] = safe_median(values)

    return output


def summarize_latency(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latency = numeric_values(
        records,
        "latency_s",
    )

    if not latency:
        latency = numeric_values(
            records,
            "latency",
        )

    if not latency:
        latency = numeric_values(
            records,
            "client_latency_seconds",
        )

    query_time = numeric_values(
        records,
        "query_formulation_seconds",
    )

    decision_time = numeric_values(
        records,
        "time_to_decision_seconds",
    )

    return {
        "latency_mean_seconds":
            safe_mean(latency),
        "latency_median_seconds":
            safe_median(latency),
        "query_formulation_mean_seconds":
            safe_mean(query_time),
        "query_formulation_median_seconds":
            safe_median(query_time),
        "decision_time_mean_seconds":
            safe_mean(decision_time),
        "decision_time_median_seconds":
            safe_median(decision_time),
    }


def summarize_usefulness(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    values = numeric_values(
        records,
        "usefulness",
    )

    return {
        "usefulness_count":
            len(values),
        "usefulness_mean":
            safe_mean(values),
        "usefulness_median":
            safe_median(values),
    }


def summarize_protocol_deviations(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    deviations = [
        record
        for record in records
        if record.get(
            "protocol_deviation"
        ) is True
    ]

    interventions = [
        record
        for record in records
        if record.get(
            "evaluator_intervention_pre_submit"
        ) is True
    ]

    deviation_types: Counter[str] = Counter()

    for record in deviations:
        value = record.get(
            "deviation_type"
        )

        if value:
            deviation_types[
                str(value)
            ] += 1

    total = len(records)

    return {
        "protocol_deviation_count":
            len(deviations),
        "protocol_deviation_rate":
            rate(
                len(deviations),
                total,
            ),
        "pre_submit_intervention_count":
            len(interventions),
        "pre_submit_intervention_rate":
            rate(
                len(interventions),
                total,
            ),
        "deviation_type_counts":
            dict(
                sorted(
                    deviation_types.items()
                )
            ),
    }


def normalize_failure_labels(
    value: Any,
) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if item is not None
        ]

    return [str(value)]


def summarize_failures(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    records_with_failure = 0

    for record in records:
        labels = normalize_failure_labels(
            record.get(
                "failure_mode"
            )
        )

        if not labels:
            labels = normalize_failure_labels(
                record.get(
                    "failure_modes"
                )
            )

        if labels:
            records_with_failure += 1
            counter.update(labels)

    total = len(records)

    return {
        "records_with_failure":
            records_with_failure,
        "failure_record_rate":
            rate(
                records_with_failure,
                total,
            ),
        "failure_counts":
            dict(
                sorted(counter.items())
            ),
        "trust_calibration_failure_count":
            counter["F08"],
        "trust_calibration_failure_rate":
            rate(
                counter["F08"],
                total,
            ),
    }


def summarize_multiturn(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize multi-turn risk fields when present.

    This does not assume a specific nested turn
    representation. It supports top-level turn and
    cumulative risk values and an optional `turns`
    list if future records use one.
    """
    turn_risk: List[float] = []
    cumulative_risk: List[float] = []

    records_with_multiturn = 0

    for record in records:
        found = False

        value = record.get(
            "turn_level_risk"
        )

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            turn_risk.append(
                float(value)
            )
            found = True

        value = record.get(
            "cumulative_risk"
        )

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            cumulative_risk.append(
                float(value)
            )
            found = True

        turns = record.get("turns")

        if isinstance(turns, list):
            for turn in turns:
                if not isinstance(
                    turn,
                    dict,
                ):
                    continue

                tr = turn.get(
                    "turn_level_risk"
                )

                cr = turn.get(
                    "cumulative_risk"
                )

                if (
                    isinstance(
                        tr,
                        (int, float),
                    )
                    and not isinstance(
                        tr,
                        bool,
                    )
                ):
                    turn_risk.append(
                        float(tr)
                    )
                    found = True

                if (
                    isinstance(
                        cr,
                        (int, float),
                    )
                    and not isinstance(
                        cr,
                        bool,
                    )
                ):
                    cumulative_risk.append(
                        float(cr)
                    )
                    found = True

        if found:
            records_with_multiturn += 1

    return {
        "record_count":
            records_with_multiturn,
        "turn_level_risk_mean":
            safe_mean(turn_risk),
        "turn_level_risk_max":
            (
                round(
                    max(turn_risk),
                    4,
                )
                if turn_risk
                else None
            ),
        "cumulative_risk_mean":
            safe_mean(
                cumulative_risk
            ),
        "cumulative_risk_max":
            (
                round(
                    max(cumulative_risk),
                    4,
                )
                if cumulative_risk
                else None
            ),
    }


def summarize_by_scenario(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    grouped: Dict[
        str,
        List[Dict[str, Any]]
    ] = defaultdict(list)

    for record in records:
        scenario = str(
            record.get(
                "scenario_id",
                "unknown",
            )
        )

        grouped[scenario].append(
            record
        )

    result: Dict[str, Any] = {}

    for scenario, rows in sorted(
        grouped.items()
    ):
        result[scenario] = {
            "count": len(rows),
            "disclosure":
                summarize_disclosure(
                    rows
                ),
            "detection":
                summarize_detection(
                    rows
                ),
            "system_states":
                summarize_system_states(
                    rows
                ),
            "decisions":
                summarize_decisions(
                    rows
                ),
            "comprehension":
                summarize_comprehension(
                    rows
                ),
            "adoption":
                summarize_adoption(
                    rows
                ),
            "privacy":
                summarize_privacy(
                    rows
                ),
            "utility":
                summarize_utility(
                    rows
                ),
            "usefulness":
                summarize_usefulness(
                    rows
                ),
            "latency":
                summarize_latency(
                    rows
                ),
            "protocol":
                summarize_protocol_deviations(
                    rows
                ),
            "failures":
                summarize_failures(
                    rows
                ),
            "multiturn":
                summarize_multiturn(
                    rows
                ),
        }

    return result


def build_summary(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    participants = sorted(
        {
            str(
                record["participant_id"]
            )
            for record in records
            if record.get(
                "participant_id"
            )
        }
    )

    scenarios = sorted(
        {
            str(
                record["scenario_id"]
            )
            for record in records
            if record.get(
                "scenario_id"
            )
        }
    )

    failures = summarize_failures(
        records
    )

    system_states = (
        summarize_system_states(
            records
        )
    )

    protocol = (
        summarize_protocol_deviations(
            records
        )
    )

    reliability = {
        **failures,
        "zero_risk_passthrough_count":
            system_states[
                "zero_risk_passthrough_count"
            ],
        "zero_risk_passthrough_rate":
            system_states[
                "zero_risk_passthrough_rate"
            ],
        "protocol_deviation_count":
            protocol[
                "protocol_deviation_count"
            ],
        "protocol_deviation_rate":
            protocol[
                "protocol_deviation_rate"
            ],
        "pre_submit_intervention_count":
            protocol[
                "pre_submit_intervention_count"
            ],
        "pre_submit_intervention_rate":
            protocol[
                "pre_submit_intervention_rate"
            ],
    }

    return {
        "experiment_id":
            config.get(
                "experiment_id"
            ),
        "session":
            config.get(
                "session"
            ),
        "experiment_type":
            config.get(
                "experiment_type"
            ),
        "schema_version":
            config.get(
                "schema_version"
            ),
        "record_count":
            len(records),
        "participant_count":
            len(participants),
        "participants":
            participants,
        "scenario_count":
            len(scenarios),
        "scenarios":
            scenarios,

        "disclosure":
            summarize_disclosure(
                records
            ),

        "detection":
            summarize_detection(
                records
            ),

        "system_states":
            system_states,

        "privacy":
            summarize_privacy(
                records
            ),

        "utility":
            summarize_utility(
                records
            ),

        "usability": {
            "decisions":
                summarize_decisions(
                    records
                ),
            "comprehension":
                summarize_comprehension(
                    records
                ),
            "adoption":
                summarize_adoption(
                    records
                ),
            "usefulness":
                summarize_usefulness(
                    records
                ),
            "latency":
                summarize_latency(
                    records
                ),
        },

        "protocol":
            protocol,

        "reliability":
            reliability,

        "multiturn":
            summarize_multiturn(
                records
            ),

        "by_scenario":
            summarize_by_scenario(
                records
            ),
    }


def flatten_summary(
    summary: Dict[str, Any],
) -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []

    def walk(
        prefix: str,
        value: Any,
    ) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = (
                    f"{prefix}.{key}"
                    if prefix
                    else key
                )

                walk(
                    next_prefix,
                    child,
                )

        elif isinstance(value, list):
            rows.append(
                (
                    prefix,
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    ),
                )
            )

        else:
            rows.append(
                (
                    prefix,
                    value,
                )
            )

    walk(
        "",
        summary,
    )

    return rows


def write_outputs(
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    metrics_path = (
        output_dir
        / "metrics.csv"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        for metric, value in flatten_summary(
            summary
        ):
            writer.writerow(
                [
                    metric,
                    value,
                ]
            )

    print(
        f"Wrote {summary_path}"
    )

    print(
        f"Wrote {metrics_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize validated "
            "InferenceGuard experiment records."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Experiment YAML config.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSON/JSONL file or directory "
            "containing experiment records."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Directory for derived outputs."
        ),
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: "
            f"{config_path}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: "
            f"{input_path}"
        )

    config = load_yaml(
        config_path
    )

    records = load_records(
        input_path
    )

    if not records:
        raise ValueError(
            "No experiment records "
            "were loaded."
        )

    summary = build_summary(
        records=records,
        config=config,
    )

    write_outputs(
        summary=summary,
        output_dir=output_dir,
    )

    print(
        "\n=== Experiment summary ==="
    )

    print(
        f"Experiment: "
        f"{summary['experiment_id']}"
    )

    print(
        f"Records: "
        f"{summary['record_count']}"
    )

    print(
        f"Participants: "
        f"{summary['participant_count']}"
    )

    print(
        "Mean ADR:",
        summary[
            "disclosure"
        ][
            "anchor_disclosure_rate_mean"
        ],
    )

    print(
        "Detection rate:",
        summary[
            "detection"
        ][
            "detected_risk_rate"
        ],
    )

    print(
        "Acceptance rate:",
        summary[
            "usability"
        ][
            "decisions"
        ][
            "acceptance_rate"
        ],
    )

    print(
        "Comprehension rate:",
        summary[
            "usability"
        ][
            "comprehension"
        ][
            "comprehension_rate"
        ],
    )

    print(
        "Adoption rate:",
        summary[
            "usability"
        ][
            "adoption"
        ][
            "adoption_rate"
        ],
    )

    print(
        "Zero-risk passthrough rate:",
        summary[
            "reliability"
        ][
            "zero_risk_passthrough_rate"
        ],
    )

    print(
        "Mean risk delta:",
        summary[
            "privacy"
        ][
            "risk_delta_mean"
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
