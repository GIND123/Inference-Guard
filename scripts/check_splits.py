"""Check that artifacts/profile_splits.json is a valid profile-disjoint split.

Needs only the standard library, so it runs anywhere (no `datasets` or
`scikit-learn`, unlike scripts/make_artifacts.py, which regenerates the file):

    python scripts/check_splits.py
    python scripts/check_splits.py --path artifacts/profile_splits.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "profile_splits.json"
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_SEED = 42


def check_splits(splits: Dict[str, Any]) -> List[str]:
    """Return a list of problems with a profile split; an empty list means valid."""
    problems: List[str] = []

    for name in SPLIT_NAMES:
        ids = splits.get(name)
        if not isinstance(ids, list) or not ids:
            problems.append(f"'{name}' is missing or empty")
        elif len(set(ids)) != len(ids):
            problems.append(f"'{name}' contains duplicate profile ids")

    if splits.get("seed") != EXPECTED_SEED:
        problems.append(f"seed is {splits.get('seed')!r}, expected {EXPECTED_SEED}")

    if problems:
        return problems

    sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                problems.append(f"'{a}' and '{b}' share {len(shared)} profile(s)")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a profile-disjoint split file.")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    problems = check_splits(splits)
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    sizes = " / ".join(str(len(splits[n])) for n in SPLIT_NAMES)
    print(f"OK: {args.path} train/val/test = {sizes}, seed {splits['seed']}, no overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
