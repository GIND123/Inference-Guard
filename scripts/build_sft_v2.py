"""Build clean single-turn Rewriter SFT dataset (teacher_v2).

Combines:
  1. Accepted natural SynthPAI rewrites from candidates_train.jsonl, with quality
     filtering on number preservation and forward entailment / token cosine.
  2. 2,000 position-controlled synthetic rows from data/sft/augmented_v1/augmented.jsonl.
  3. ~15% identity no-op rows (comments with no inferable attributes).

Outputs:
  data/sft/teacher_v2/train.jsonl
  data/sft/teacher_v2/val.jsonl
  data/sft/teacher_v2/candidates_train.jsonl
  data/sft/teacher_v2/manifest_train.json
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set

from src.rewriter.augment import PersonaComposer
from src.rewriter.prompting import build_rewrite_messages
from src.rewriter.teacher_sft import (
    ATTRIBUTES,
    build_augmented_rows,
    build_row,
    inferable_attributes,
    sha256_file,
)

CANDIDATES_SOURCE = Path("data/sft/teacher_v1/candidates_train.jsonl")
AUGMENTED_SOURCE = Path("data/sft/augmented_v1/augmented.jsonl")
TRAIN_SPLIT = Path("data/splits/train.jsonl")
VAL_SPLIT = Path("data/splits/val.jsonl")
OUT_DIR = Path("data/sft/teacher_v2")

SEED = 42


def extract_task_numbers(text: str, attrs: List[str]) -> Set[str]:
    text_clean = text.lower()
    if "age" in attrs:
        text_clean = re.sub(r"\b\d{1,2}[-\s]*(?:years?|yrs?|yo)\b", "", text_clean)
        text_clean = re.sub(r"\b(?:born|class of)\s+\d{4}\b", "", text_clean)
        text_clean = re.sub(r"\b(?:in my|in their|early|mid|late)\s+\d{2}s\b", "", text_clean)
    found = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", text_clean))
    return {n for n in found if float(n.replace(",", "")) not in (0, 1)}


def build_sft_v2() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # 1. Copy candidates cache
    target_cand_path = OUT_DIR / "candidates_train.jsonl"
    if CANDIDATES_SOURCE.exists() and not target_cand_path.exists():
        shutil.copyfile(CANDIDATES_SOURCE, target_cand_path)

    # Read candidates
    with open(CANDIDATES_SOURCE, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    accepted_synthpai: List[Dict[str, Any]] = []
    dropped_numbers = 0
    dropped_drift = 0

    for c in candidates:
        if not c.get("accepted"):
            continue

        f_ent = c.get("nli_forward_entailment")
        t_cos = c.get("token_cosine")
        if f_ent is not None and t_cos is not None:
            if f_ent < 0.50 and t_cos < 0.65:
                dropped_drift += 1
                continue

        orig_nums = extract_task_numbers(c["original"], c.get("attributes", []))
        cand_nums = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", c["candidate"].lower()))
        if orig_nums - cand_nums:
            dropped_numbers += 1
            continue

        accepted_synthpai.append(c)

    print(f"Accepted SynthPAI: {len(accepted_synthpai)} (dropped drift: {dropped_drift}, dropped numbers: {dropped_numbers})")

    # Format SynthPAI rows
    train_rows: List[Dict[str, Any]] = [
        build_row(c["original"], c["candidate"], {
            "id": c["id"],
            "profile_id": c["profile_id"],
            "split": "train",
            "source": "teacher_synthpai",
            "attributes": c["attributes"],
            "identity_noop": False,
            "nli_forward_entailment": c.get("nli_forward_entailment"),
        })
        for c in accepted_synthpai
    ]

    # 2. Add augmented rows
    with open(AUGMENTED_SOURCE, "r", encoding="utf-8") as f:
        augmented_records = [json.loads(line) for line in f if line.strip()]

    aug_rows, aug_stats = build_augmented_rows(augmented_records, None, None, mode="deterministic")
    print(f"Loaded {len(aug_rows)} augmented rows")
    train_rows.extend(aug_rows)

    # 3. Add identity no-ops from train split
    train_records: List[Dict[str, Any]] = []
    with open(TRAIN_SPLIT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    noop_pool = [
        r for r in train_records
        if not inferable_attributes(r) and len(r.get("text", "").split()) >= 5
    ]
    noop_pool.sort(key=lambda r: r["id"])
    rng.shuffle(noop_pool)

    # ~15% of total dataset should be noops
    target_noops = round((len(train_rows) * 0.15) / 0.85)
    selected_noops = noop_pool[:target_noops]
    print(f"Adding {len(selected_noops)} identity no-op rows")

    for r in selected_noops:
        train_rows.append(build_row(r["text"], r["text"], {
            "id": r["id"],
            "profile_id": r.get("profile_id") or r.get("author"),
            "split": "train",
            "source": "identity_noop",
            "attributes": [],
            "identity_noop": True,
            "nli_forward_entailment": 1.0,
        }))

    rng.shuffle(train_rows)

    train_out_path = OUT_DIR / "train.jsonl"
    with open(train_out_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_rows)} SFT training rows to {train_out_path}")

    # Build validation SFT file (val.jsonl)
    val_records: List[Dict[str, Any]] = []
    with open(VAL_SPLIT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    from src.rewriter.augment import compose_corpus
    val_examples = compose_corpus(300, seed=999)
    val_augmented = [
        {
            "id": ex.id,
            "input": ex.input,
            "target": ex.target,
            "position": ex.position,
            "attributes": ex.attributes,
            "injected": [{"attribute": f.attribute, "surface": f.surface, "variants": f.variants} for f in ex.injected],
        }
        for ex in val_examples
    ]
    val_aug_rows, _ = build_augmented_rows(val_augmented, None, None, mode="deterministic")

    val_noop_pool = [
        r for r in val_records
        if not inferable_attributes(r) and len(r.get("text", "").split()) >= 5
    ]
    val_noops = val_noop_pool[:60]
    val_rows: List[Dict[str, Any]] = list(val_aug_rows)
    for r in val_noops:
        val_rows.append(build_row(r["text"], r["text"], {
            "id": r["id"],
            "profile_id": r.get("profile_id") or r.get("author"),
            "split": "val",
            "source": "identity_noop",
            "attributes": [],
            "identity_noop": True,
        }))

    rng.shuffle(val_rows)
    val_out_path = OUT_DIR / "val.jsonl"
    with open(val_out_path, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(val_rows)} SFT validation rows to {val_out_path}")

    # Write manifest
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "split": "train",
        "rows": len(train_rows),
        "synthpai_accepted": len(accepted_synthpai),
        "augmented_rows": len(aug_rows),
        "identity_noops": len(selected_noops),
        "val_rows": len(val_rows),
        "outputs": {
            "train": {"path": str(train_out_path), "sha256": sha256_file(train_out_path)},
            "val": {"path": str(val_out_path), "sha256": sha256_file(val_out_path)},
        },
    }
    with open(OUT_DIR / "manifest_train.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    build_sft_v2()
