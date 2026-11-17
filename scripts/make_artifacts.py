"""Reproduce artifacts/taxonomy.json and artifacts/profile_splits.json on CPU.

Mirrors the notebook's logic exactly (InferenceGuard_Implementation.ipynb):
  - taxonomy dict from "Attribute Taxonomy Finalization"
  - profile-disjoint split from "Profile Disjoint Data Splits" (seed 42)

Requires only `datasets` and `scikit-learn`:
    pip install datasets scikit-learn
    python scripts/make_artifacts.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import sys

from src.data.dataset_pin import DATASET_ID, pinned_revision

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
SEED = 42

# Single source of truth: derive the serialized taxonomy from
# src/risk_model/taxonomy.ATTRIBUTE_TAXONOMY so artifacts/taxonomy.json can
# never drift from the code the risk model and attacker actually use
# (decision #7, docs/methodology-review.md). This supersedes the old hand-kept
# dict that had drifted (missing the location key, free-text education labels,
# and an extra age "unknown" bucket).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY  # noqa: E402

taxonomy = {attr.value: list(labels) for attr, labels in ATTRIBUTE_TAXONOMY.items()}


def get_profile_id(ex):
    if not isinstance(ex, dict):
        return None
    for key in ["author","username"]:
        if key in ex and isinstance(ex[key], str):
            return str(ex[key])
    return None


def main():
    # Pinned. An unpinned load takes whatever the hub serves today, and the
    # split is a function of dataset iteration order as well as the seed --
    # so the committed artifact would stop reproducing without anything
    # failing. See issue #21.
    ds = load_dataset(DATASET_ID, revision=pinned_revision())

    profile_to_examples = defaultdict(list)
    for split in ds.keys():
        for idx in range(len(ds[split])):
            ex = ds[split][idx]
            pid = get_profile_id(ex)
            if pid:
                profile_to_examples[pid].append((split, idx))

    unique_profiles = list(profile_to_examples.keys())

    random.seed(SEED)
    np.random.seed(SEED)
    train_p, temp_p = train_test_split(unique_profiles, test_size=0.30, random_state=SEED)
    val_p, test_p = train_test_split(temp_p, test_size=0.50, random_state=SEED)

    assert len(unique_profiles) == 300, f"expected 300 unique profiles, got {len(unique_profiles)}"
    assert (len(train_p), len(val_p), len(test_p)) == (210, 45, 45), \
        f"expected 210/45/45, got {len(train_p)}/{len(val_p)}/{len(test_p)}"
    tr, va, te = set(train_p), set(val_p), set(test_p)
    assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap"
    assert tr | va | te == set(unique_profiles), "splits do not cover all profiles"

    ARTIFACTS.mkdir(exist_ok=True)
    # The revision belongs in the artifact, not only in the config: the file
    # has to say what it was built from, or a future reader cannot tell
    # whether a regenerated split should match it.
    splits = {
        "train": train_p,
        "val": val_p,
        "test": test_p,
        "seed": SEED,
        "dataset_id": DATASET_ID,
        "dataset_revision": pinned_revision(),
    }
    with open(ARTIFACTS / "profile_splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    with open(ARTIFACTS / "taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    print(f"Unique profiles: {len(unique_profiles)}")
    print(f"Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)} | overlap: none")
    for name, ids in [("train", train_p), ("val", val_p), ("test", test_p)]:
        print(f"first 3 {name}: {ids[:3]}")
    print(f"Wrote {ARTIFACTS / 'profile_splits.json'} and {ARTIFACTS / 'taxonomy.json'}")


if __name__ == "__main__":
    main()
