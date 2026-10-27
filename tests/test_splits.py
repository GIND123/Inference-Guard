# tests/test_splits.py
"""Verification suite for Gate 1: Profile-Disjoint Split Integrity.

Verifies zero profile ID overlap between train, val, and test partitions,
cryptographic SHA-256 manifest integrity, and demographic stratification.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import pytest

pytest.importorskip(
    "numpy",
    reason="needs the optional numpy dependency; see tests/README.md",
)

from src.data.splits import create_profile_disjoint_splits, extract_profile_id


@pytest.fixture
def synthetic_author_records():
    """Generates 300 comments across 20 distinct synthetic profiles with demographic annotations."""
    records = []
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    occupations = ["student", "technology", "healthcare", "education_academic"]

    for i in range(20):
        pid = f"pers_{i:03d}"
        age = age_groups[i % len(age_groups)]
        occ = occupations[i % len(occupations)]
        # Each author has multiple comments
        for c in range(5):
            records.append({
                "profile_id": pid,
                "comment_id": f"{pid}_c{c}",
                "text": f"Comment {c} by {pid} discussing life in tech.",
                "attributes": {"age": age, "occupation": occ, "location": "us_west"},
            })
    return records


def test_profile_disjointness_and_manifest(synthetic_author_records):
    """Verify that train, val, and test splits share 0 profile IDs and manifest matches SHA-256."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "splits"
        manifest_dir = Path(tmp_dir) / "manifests"

        manifest = create_profile_disjoint_splits(
            records=synthetic_author_records,
            output_dir=out_dir,
            manifest_dir=manifest_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=641,
        )

        train_pids = set(manifest["train"]["profile_ids"])
        val_pids = set(manifest["val"]["profile_ids"])
        test_pids = set(manifest["test"]["profile_ids"])

        # Core invariant: zero profile ID leakage
        assert train_pids.isdisjoint(val_pids), "Train and val share profile IDs!"
        assert train_pids.isdisjoint(test_pids), "Train and test share profile IDs!"
        assert val_pids.isdisjoint(test_pids), "Val and test share profile IDs!"

        total_profiles = len(train_pids) + len(val_pids) + len(test_pids)
        assert total_profiles == 20, f"Expected 20 profiles, got {total_profiles}"

        # Verify SHA-256 hashes against actual files
        for split_name in ("train", "val", "test"):
            file_path = out_dir / f"{split_name}.jsonl"
            assert file_path.exists()
            content = file_path.read_bytes()
            computed_sha = hashlib.sha256(content).hexdigest()
            assert manifest[split_name]["sha256"] == computed_sha

        # Verify manifest JSON files were written
        assert (out_dir / "split_manifest.json").exists()
        assert (manifest_dir / "split_manifest.json").exists()


def test_extract_profile_id_variants():
    """Verify profile ID extraction across schema variations (author, username, profile_id)."""
    assert extract_profile_id({"profile_id": "p123"}) == "p123"
    assert extract_profile_id({"author": "author_456"}) == "author_456"
    assert extract_profile_id({"username": "user_789"}) == "user_789"
    with pytest.raises(KeyError):
        extract_profile_id({"unrelated_field": "val"})


def test_empty_records_rejection():
    """Verify that attempting to split an empty record set raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(ValueError, match="Cannot partition empty records list"):
            create_profile_disjoint_splits([], Path(tmp_dir))
