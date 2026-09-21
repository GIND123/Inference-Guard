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

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
SEED = 42

# Identical to the notebook's taxonomy dict
taxonomy = {
    "age": ["18-24","25-34","35-44","45-54","55+","unknown"],
    "education": ["high_school","undergraduate","graduate_professional","other_unknown"],
    "occupation": ["student","technology","healthcare","education","business_finance","service","other"],
    "location": ["USA_Northeast","USA_South","USA_Midwest","USA_West","Europe","Other","unknown"]
}


def get_profile_id(ex):
    if not isinstance(ex, dict):
        return None
    for key in ["author","username"]:
        if key in ex and isinstance(ex[key], str):
            return str(ex[key])
    return None


def main():
    ds = load_dataset("RobinSta/SynthPAI")

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
    splits = {"train": train_p, "val": val_p, "test": test_p, "seed": SEED}
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
