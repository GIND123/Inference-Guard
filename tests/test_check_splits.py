import json
from pathlib import Path

from scripts.check_splits import DEFAULT_PATH, check_splits


def make_splits():
    return {"train": ["p1", "p2", "p3"], "val": ["p4"], "test": ["p5"], "seed": 42}


def test_valid_split_has_no_problems():
    assert check_splits(make_splits()) == []


def test_overlap_between_splits_is_reported():
    splits = make_splits()
    splits["test"] = ["p1"]
    problems = check_splits(splits)
    assert problems == ["'train' and 'test' share 1 profile(s)"]


def test_wrong_seed_is_reported():
    splits = make_splits()
    splits["seed"] = 7
    assert any("seed" in p for p in check_splits(splits))


def test_missing_or_empty_split_is_reported():
    splits = make_splits()
    del splits["val"]
    splits["test"] = []
    problems = check_splits(splits)
    assert "'val' is missing or empty" in problems
    assert "'test' is missing or empty" in problems


def test_duplicate_ids_within_a_split_are_reported():
    splits = make_splits()
    splits["train"] = ["p1", "p1", "p2"]
    assert "'train' contains duplicate profile ids" in check_splits(splits)


def test_committed_artifact_is_valid():
    with open(Path(DEFAULT_PATH), "r", encoding="utf-8") as f:
        assert check_splits(json.load(f)) == []
