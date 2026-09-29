"""
Summarize validated InferenceGuard experiment records.

This script reads JSON / JSONL experiment records and produces:
- summary.json
- metrics.csv

It is intended to run after validate_experiment.py.

Example:

python scripts/summarize_experiment.py \
    --config configs/experiments/session06.yaml \
    --input experiments/session06_user_eval_v1/raw \
    --output experiments/session06_user_eval_v1/derived
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


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping.")

    return data


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


def load_records(path: Path) -> List[Dict[str, Any]]:
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
            f"No JSON or JSONL files found in {path}."
        )

    records: List[Dict[str, Any]] = []

    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            records.extend(read_jsonl_file(file_path))
        else:
            records.extend(read_json_file(file_path))

    return records


def numeric_values(
    records: Iterable[Dict[str, Any]],
    field: str,
) -> List[float]:
    values: List[float] = []

    for record in records:
        value = record.get(field)

        if isinstance(value, (int, float)):
            values.append(float(value))

    return values


def safe_mean(values: List[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 4)


def safe_median(values: List[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 4)


def rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(count / total, 4)


def summarize_decisions(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    counts = Counter()

    for record in records:
        decision = record.get("rewrite_decision")

        if decision in ACCEPT_DECISIONS:
            counts["accept"] += 1

        elif decision in EDIT_DECISIONS:
            counts["accept_with_edits"] += 1

        elif decision in REJECT_DECISIONS:
            counts["reject"] += 1

        elif decision is not None:
            counts["other"] += 1

    total = sum(counts.values())

    return {
        "total_decisions": total,
        "accept_count": counts["accept"],
        "accept_with_edits_count": counts["accept_with_edits"],
        "reject_count": counts["reject"],
        "other_count": counts["other"],
        "acceptance_rate": rate(
            counts["accept"] + counts["accept_with_edits"],
            total,
        ),
        "accept_as_is_rate": rate(
            counts["accept"],
            total,
        ),
        "edit_rate": rate(
            counts["accept_with_edits"],
            total,
        ),
        "rejection_rate": rate(
            counts["reject"],
            total,
        ),
    }


def summarize_comprehension(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    values = [
        record.get("risk_understanding_correct")
        for record in records
        if isinstance(
            record.get("risk_understanding_correct"),
            bool,
        )
    ]

    if not values:
        return {
            "evaluated_count": 0,
            "correct_count": 0,
            "comprehension_rate": None,
        }

    correct = sum(values)

    return {
        "evaluated_count": len(values),
        "correct_count": correct,
        "comprehension_rate": rate(
            correct,
            len(values),
        ),
    }


def summarize_privacy(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    before = numeric_values(records, "risk_before")
    after = numeric_values(records, "risk_after")

    explicit_delta = numeric_values(
        records,
        "privacy_risk_delta",
    )

    paired_deltas: List[float] = []

    for record in records:
        b = record.get("risk_before")
        a = record.get("risk_after")

        if isinstance(b, (int, float)) and isinstance(
            a,
            (int, float),
        ):
            paired_deltas.append(float(b) - float(a))

    deltas = explicit_delta or paired_deltas

    risk_inversions = sum(
        1 for value in paired_deltas if value < 0
    )

    return {
        "risk_before_mean": safe_mean(before),
        "risk_after_mean": safe_mean(after),
        "risk_delta_mean": safe_mean(deltas),
        "risk_delta_median": safe_median(deltas),
        "risk_inversion_count": risk_inversions,
        "risk_inversion_rate": rate(
            risk_inversions,
            len(paired_deltas),
        ),
    }


def summarize_utility(
    records: List[Dict[str, Any]]
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
        values = numeric_values(records, field)

        output[f"{field}_mean"] = safe_mean(values)
        output[f"{field}_median"] = safe_median(values)

    return output


def summarize_latency(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    latency = numeric_values(records, "latency_s")

    if not latency:
        latency = numeric_values(records, "latency")

    decision_time = numeric_values(
        records,
        "time_to_decision_seconds",
    )

    return {
        "latency_mean_seconds": safe_mean(latency),
        "latency_median_seconds": safe_median(latency),
        "decision_time_mean_seconds": safe_mean(
            decision_time
        ),
        "decision_time_median_seconds": safe_median(
            decision_time
        ),
    }


def summarize_usefulness(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    values = numeric_values(records, "usefulness")

    return {
        "usefulness_count": len(values),
        "usefulness_mean": safe_mean(values),
        "usefulness_median": safe_median(values),
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
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    counter: Counter[str] = Counter()

    records_with_failure = 0

    for record in records:
        labels = normalize_failure_labels(
            record.get("failure_mode")
        )

        if not labels:
            labels = normalize_failure_labels(
                record.get("failure_modes")
            )

        if labels:
            records_with_failure += 1
            counter.update(labels)

    total = len(records)

    return {
        "records_with_failure": records_with_failure,
        "failure_record_rate": rate(
            records_with_failure,
            total,
        ),
        "failure_counts": dict(
            sorted(counter.items())
        ),
    }


def summarize_by_scenario(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(
        list
    )

    for record in records:
        scenario = str(
            record.get("scenario_id", "unknown")
        )
        grouped[scenario].append(record)

    result: Dict[str, Any] = {}

    for scenario, rows in sorted(grouped.items()):
        result[scenario] = {
            "count": len(rows),
            "decisions": summarize_decisions(rows),
            "comprehension": summarize_comprehension(
                rows
            ),
            "privacy": summarize_privacy(rows),
            "usefulness": summarize_usefulness(rows),
            "latency": summarize_latency(rows),
            "failures": summarize_failures(rows),
        }

    return result


def build_summary(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    participants = sorted(
        {
            str(record["participant_id"])
            for record in records
            if record.get("participant_id")
        }
    )

    scenarios = sorted(
        {
            str(record["scenario_id"])
            for record in records
            if record.get("scenario_id")
        }
    )

    return {
        "experiment_id": config.get("experiment_id"),
        "session": config.get("session"),
        "experiment_type": config.get(
            "experiment_type"
        ),
        "record_count": len(records),
        "participant_count": len(participants),
        "participants": participants,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "privacy": summarize_privacy(records),
        "utility": summarize_utility(records),
        "usability": {
            "decisions": summarize_decisions(records),
            "comprehension": summarize_comprehension(
                records
            ),
            "usefulness": summarize_usefulness(records),
            "latency": summarize_latency(records),
        },
        "reliability": summarize_failures(records),
        "by_scenario": summarize_by_scenario(records),
    }


def flatten_summary(
    summary: Dict[str, Any]
) -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = (
                    f"{prefix}.{key}"
                    if prefix
                    else key
                )
                walk(next_prefix, child)

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
            rows.append((prefix, value))

    walk("", summary)
    return rows


def write_outputs(
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_dir / "summary.json"

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

    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for metric, value in flatten_summary(summary):
            writer.writerow([metric, value])

    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize validated InferenceGuard "
            "experiment records."
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
            "JSON/JSONL file or directory containing "
            "experiment records."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for derived outputs.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}"
        )

    config = load_yaml(config_path)
    records = load_records(input_path)

    if not records:
        raise ValueError(
            "No experiment records were loaded."
        )

    summary = build_summary(
        records=records,
        config=config,
    )

    write_outputs(
        summary=summary,
        output_dir=output_dir,
    )

    print("\n=== Experiment summary ===")
    print(f"Experiment: {summary['experiment_id']}")
    print(f"Records: {summary['record_count']}")
    print(
        f"Participants: {summary['participant_count']}"
    )

    decisions = summary["usability"]["decisions"]

    print(
        "Acceptance rate:",
        decisions["acceptance_rate"],
    )

    print(
        "Rejection rate:",
        decisions["rejection_rate"],
    )

    comprehension = summary["usability"][
        "comprehension"
    ]

    print(
        "Comprehension rate:",
        comprehension["comprehension_rate"],
    )

    print(
        "Mean risk delta:",
        summary["privacy"]["risk_delta_mean"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
