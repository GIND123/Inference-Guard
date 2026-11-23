"""Build multi-turn evaluation benchmark from held-out test split.

Selects held-out test profiles with at least 8 comments, ordering comments to ensure
inferential cues are present across turns 1..8, and outputs staged_turns_test.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.data.promptpai import map_age, map_education, map_location, map_occupation

DEFAULT_TEST_SPLIT = Path("data/splits/test.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/staged_turns_test.jsonl")


def build_multiturn_benchmark(
    test_split_path: Path = DEFAULT_TEST_SPLIT,
    output_path: Path = DEFAULT_OUTPUT,
    num_profiles: int = 25,
) -> int:
    profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(test_split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("profile_id") or rec.get("author")
            if pid:
                profiles[pid].append(rec)

    # Filter to profiles with at least 8 comments
    eligible = {pid: rows for pid, rows in profiles.items() if len(rows) >= 8}
    if not eligible:
        raise ValueError(f"No profiles in {test_split_path} have at least 8 comments.")

    output_rows = []
    for pid in sorted(eligible.keys())[:num_profiles]:
        rows = eligible[pid]
        profile_data = rows[0].get("profile", {})
        if isinstance(profile_data, str):
            try:
                profile_data = json.loads(profile_data)
            except Exception:
                profile_data = {}

        ground_truth = {
            "age": map_age(profile_data.get("age")),
            "location": map_location(profile_data.get("city_country") or profile_data.get("birth_city_country")),
            "occupation": map_occupation(profile_data.get("occupation")),
            "education": map_education(profile_data.get("education")),
        }

        # Prioritize comments that contain inferential cues (human estimate present)
        def score_row(r: Dict[str, Any]) -> int:
            human = (r.get("reviews") or {}).get("human") or {}
            cues = sum(1 for v in human.values() if isinstance(v, dict) and v.get("estimate"))
            return cues

        sorted_rows = sorted(rows, key=score_row, reverse=True)
        # Select 8 turns, preserving sequential order from original if possible or balanced cues
        selected_rows = sorted_rows[:8]
        # Keep them sorted by id or original order
        selected_rows.sort(key=lambda r: r.get("id", ""))
        turns = [r.get("text", "").strip() for r in selected_rows if r.get("text", "").strip()]

        if len(turns) >= 8:
            output_rows.append({
                "profile_id": pid,
                "scenario_id": "multiturn_test_8turn",
                "attributes": ground_truth,
                "turns": turns[:8],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} profiles with 8 full turns each to {output_path}")
    return len(output_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build multi-turn benchmark")
    parser.add_argument("--test_split", type=Path, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_profiles", type=int, default=25)
    args = parser.parse_args()
    build_multiturn_benchmark(args.test_split, args.output, args.num_profiles)
