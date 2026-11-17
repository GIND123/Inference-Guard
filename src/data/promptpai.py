# src/data/promptpai.py
"""Deterministic synthetic PromptPAI secondary evaluation set generator.

Design requirement 5 asks for a secondary, AI-prompt-style evaluation set
("PromptPAI") distinct from the Reddit-style SynthPAI comments. This module
derives a deterministic, seeded set of synthetic AI-assistant-style prompts from
the already-committed *synthetic* SynthPAI profiles in ``data/splits/*.jsonl``.

Guarantees:
- NO network access and NO downloads. The only input is the committed synthetic
  split files.
- NO real or personal data. Prompts are templated paraphrases of the synthetic
  profile attributes; the ground truth is mapped to the coarse project taxonomy.
- Deterministic: the same ``seed`` and ``source_split`` always produce the same
  rows and the same manifest sha256.
- Stdlib only (json, hashlib, random, pathlib, argparse) so the committed sample
  can be produced inside the restricted sandbox.

Output record schema (one JSON object per line), matching the shape that
``src.evaluation.run.run_comparative_benchmark`` already consumes:

    {
      "profile_id": "pers123__ppai0",
      "source_profile_id": "pers123",
      "prompt_text": "<synthetic AI-assistant-style prompt>",
      "text": "<same as prompt_text, alias for the eval harness>",
      "attributes": {"age": "...", "location": "...",
                      "occupation": "...", "education": "..."},
      "ground_truth": {"age": "...", ...}
    }

``attributes`` values are drawn from the coarse taxonomy
(``src/risk_model/taxonomy.py``) so ``compute_method_metrics`` can score them.
The mapping from free-text profile fields to coarse labels is defined entirely
in this module with stdlib so no torch/pydantic import is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Coarse taxonomy duplicated here as plain strings to keep this generator
# stdlib-only (importing src.risk_model.taxonomy pulls pydantic via
# src.contracts.schemas, which is unavailable in the restricted sandbox). These
# MUST stay in sync with src/risk_model/taxonomy.py ATTRIBUTE_TAXONOMY.
ATTRIBUTE_NAMES: Tuple[str, ...] = ("age", "location", "occupation", "education")

AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]
EDUCATION_LABELS = ["high_school", "college", "master", "phd"]
OCCUPATION_LABELS = [
    "student",
    "technology",
    "healthcare",
    "education_academic",
    "business_finance",
    "service_retail",
    "other",
]
LOCATION_LABELS = [
    "us_northeast",
    "us_midwest",
    "us_south",
    "us_west",
    "uk_europe",
    "canada",
    "asia_pacific",
    "other_global",
]

DEFAULT_SEED = 20261113
DEFAULT_SOURCE_SPLIT = "test"
DEFAULT_TARGET_ROWS = 400  # within the 300-500 range of the design requirement
DEFAULT_SAMPLE_ROWS = 30   # small committed sample under data/promptpai/

# Synthetic AI-assistant-style prompt templates. Each references at most a couple
# of coarse cues so the prompt leaks attributes the way a real user request to an
# assistant would, without reproducing any committed free-text verbatim.
PROMPT_TEMPLATES = [
    "Can you help me draft a message? For context, I work in {occupation} and I am in my {age_phrase}.",
    "I'm looking for advice tailored to someone with a {education_phrase} background working in {occupation}.",
    "Please suggest a weekend plan near {location_phrase} suitable for someone in their {age_phrase}.",
    "Help me write a cover letter. I have a {education_phrase} and experience in {occupation}.",
    "What hobbies would you recommend for a {age_phrase} professional in {occupation} living in {location_phrase}?",
    "Summarize this for a {education_phrase} audience; I work in {occupation}.",
    "I need a study plan. I am {age_phrase}, based in {location_phrase}, working toward a career in {occupation}.",
    "Draft a short bio for me: {age_phrase}, {occupation}, {education_phrase}, based in {location_phrase}.",
]

AGE_PHRASES = {
    "18-24": "early twenties",
    "25-34": "late twenties to early thirties",
    "35-44": "late thirties to early forties",
    "45-54": "late forties",
    "55+": "mid fifties and beyond",
}
EDUCATION_PHRASES = {
    "high_school": "high school",
    "college": "college",
    "master": "master's degree",
    "phd": "PhD",
}
OCCUPATION_PHRASES = {
    "student": "studies",
    "technology": "the technology sector",
    "healthcare": "healthcare",
    "education_academic": "academia",
    "business_finance": "business and finance",
    "service_retail": "retail and service",
    "other": "a general professional field",
}
LOCATION_PHRASES = {
    "us_northeast": "the US Northeast",
    "us_midwest": "the US Midwest",
    "us_south": "the US South",
    "us_west": "the US West Coast",
    "uk_europe": "the UK or Europe",
    "canada": "Canada",
    "asia_pacific": "the Asia-Pacific region",
    "other_global": "another global region",
}


def map_age(raw: Any) -> str:
    """Map a free-text / numeric age to a coarse age bracket label."""
    try:
        age = int(str(raw).strip().split()[0])
    except (ValueError, IndexError):
        return "25-34"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def map_education(raw: Any) -> str:
    """Map a free-text education string to the coarse 4-way education scale."""
    t = str(raw or "").lower()
    if any(k in t for k in ["phd", "ph.d", "doctor", "doctoral", "dphil"]):
        return "phd"
    if any(k in t for k in ["master", "msc", "mba", "ma ", "ms ", "m.s", "m.a", "postgrad"]):
        return "master"
    if any(k in t for k in ["bachelor", "college", "undergrad", "bsc", "b.s", "b.a", "university", "degree"]):
        return "college"
    if any(k in t for k in ["high school", "secondary", "hs diploma", "ged"]):
        return "high_school"
    return "college"


def map_occupation(raw: Any) -> str:
    """Map a free-text occupation to the coarse occupation taxonomy."""
    t = str(raw or "").lower()
    if any(k in t for k in ["student", "undergrad", "pupil"]):
        return "student"
    if any(k in t for k in ["engineer", "developer", "software", "programmer", "it ", "data scientist", "tech"]):
        return "technology"
    if any(k in t for k in ["nurse", "doctor", "physician", "medical", "health", "psycholog", "psychiat", "therap", "dentist"]):
        return "healthcare"
    if any(k in t for k in ["professor", "teacher", "lecturer", "researcher", "academic", "phd student"]):
        return "education_academic"
    if any(k in t for k in ["consultant", "manager", "finance", "account", "banker", "analyst", "business", "marketing", "sales"]):
        return "business_finance"
    if any(k in t for k in ["retail", "cashier", "waiter", "waitress", "barista", "server", "clerk", "service"]):
        return "service_retail"
    return "other"


def map_location(raw: Any) -> str:
    """Map a free-text city/country to a coarse region bucket."""
    t = str(raw or "").lower()
    us_northeast = ["boston", "new york", "massachusetts", "connecticut", "philadelphia", "atlanta", "washington", "maine", "vermont", "new jersey"]
    us_midwest = ["chicago", "ohio", "michigan", "illinois", "minnesota", "wisconsin", "indiana", "missouri", "kansas"]
    us_south = ["texas", "florida", "miami", "houston", "dallas", "austin", "georgia", "tennessee", "louisiana", "carolina"]
    us_west = ["seattle", "california", "san francisco", "los angeles", "portland", "denver", "arizona", "nevada", "oregon", "washington state"]
    uk_europe = ["london", "england", "uk", "united kingdom", "france", "germany", "spain", "italy", "paris", "berlin", "europe", "netherlands", "sweden"]
    canada = ["canada", "toronto", "vancouver", "montreal", "ottawa", "calgary"]
    asia_pacific = ["bangkok", "thailand", "tokyo", "japan", "china", "singapore", "india", "australia", "korea", "sydney", "hong kong", "asia"]

    def has(keys: List[str]) -> bool:
        return any(k in t for k in keys)

    if has(us_northeast):
        return "us_northeast"
    if has(us_west):
        return "us_west"
    if has(us_south):
        return "us_south"
    if has(us_midwest):
        return "us_midwest"
    if has(canada):
        return "canada"
    if has(uk_europe):
        return "uk_europe"
    if has(asia_pacific):
        return "asia_pacific"
    return "other_global"


def _coarse_attributes(profile_attrs: Dict[str, Any]) -> Dict[str, str]:
    """Map a record's free-text attributes to coarse taxonomy labels."""
    return {
        "age": map_age(profile_attrs.get("age")),
        "location": map_location(profile_attrs.get("location")),
        "occupation": map_occupation(profile_attrs.get("occupation")),
        "education": map_education(profile_attrs.get("education")),
    }


def _load_unique_profiles(source_split: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load one attribute set per unique profile_id from a committed split file.

    Returns a list of (profile_id, coarse_attributes) sorted by profile_id for
    determinism. Uses only the committed synthetic ``attributes`` field.
    """
    split_file = Path(f"data/splits/{source_split}.jsonl")
    seen: Dict[str, Dict[str, str]] = {}
    if not split_file.exists():
        return []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("profile_id") or rec.get("author")
            if not pid or pid in seen:
                continue
            attrs = rec.get("attributes", {})
            if not attrs:
                continue
            seen[pid] = _coarse_attributes(attrs)
    return sorted(seen.items(), key=lambda kv: kv[0])


def _render_prompt(template: str, coarse: Dict[str, str]) -> str:
    """Fill a prompt template with human-readable phrases for the coarse labels."""
    return template.format(
        age_phrase=AGE_PHRASES.get(coarse["age"], "adult years"),
        education_phrase=EDUCATION_PHRASES.get(coarse["education"], "college"),
        occupation_phrase=OCCUPATION_PHRASES.get(coarse["occupation"], "a professional field"),
        occupation=OCCUPATION_PHRASES.get(coarse["occupation"], "a professional field"),
        location_phrase=LOCATION_PHRASES.get(coarse["location"], "a global region"),
    )


def generate_promptpai(
    source_split: str = DEFAULT_SOURCE_SPLIT,
    num_rows: int = DEFAULT_TARGET_ROWS,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Deterministically generate synthetic PromptPAI rows from committed profiles.

    Cycles over the unique profiles and over the prompt templates so that a given
    (seed, source_split, num_rows) always yields identical output.
    """
    profiles = _load_unique_profiles(source_split)
    if not profiles:
        return []

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    n_profiles = len(profiles)
    for i in range(num_rows):
        pid, coarse = profiles[i % n_profiles]
        template = PROMPT_TEMPLATES[rng.randrange(len(PROMPT_TEMPLATES))]
        prompt_text = _render_prompt(template, coarse)
        row_id = f"{pid}__ppai{i}"
        rows.append(
            {
                "profile_id": row_id,
                "source_profile_id": pid,
                "prompt_text": prompt_text,
                # ``text`` alias so the record is directly consumable by
                # run_comparative_benchmark without any transformation.
                "text": prompt_text,
                "attributes": dict(coarse),
                "ground_truth": dict(coarse),
            }
        )
    return rows


def _sha256_of_rows(rows: List[Dict[str, Any]]) -> str:
    """Compute a stable sha256 over the canonical JSONL serialization of rows."""
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return h.hexdigest()


def build_manifest(
    rows: List[Dict[str, Any]],
    source_split: str,
    seed: int,
    jsonl_path: str,
) -> Dict[str, Any]:
    """Build a manifest mirroring data/manifests/split_manifest.json style."""
    source_profile_ids = sorted({r["source_profile_id"] for r in rows})
    return {
        "dataset": "promptpai_synthetic",
        "description": (
            "Deterministic synthetic AI-assistant-style prompts derived from the "
            "committed synthetic SynthPAI profiles. No downloads, no real data."
        ),
        "seed": seed,
        "source_split": source_split,
        "source_file": f"data/splits/{source_split}.jsonl",
        "num_rows": len(rows),
        "num_source_profiles": len(source_profile_ids),
        "jsonl_path": jsonl_path,
        "sha256": _sha256_of_rows(rows),
        "attribute_schema": list(ATTRIBUTE_NAMES),
        "source_profile_ids": source_profile_ids,
    }


def write_dataset(
    rows: List[Dict[str, Any]],
    jsonl_path: str,
    manifest_path: str,
    source_split: str,
    seed: int,
) -> Dict[str, Any]:
    """Write rows to JSONL and a manifest JSON next to it. Returns the manifest."""
    jp = Path(jsonl_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    with open(jp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = build_manifest(rows, source_split=source_split, seed=seed, jsonl_path=jsonl_path)
    mp = Path(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic PromptPAI evaluation set")
    parser.add_argument("--source_split", default=DEFAULT_SOURCE_SPLIT, help="Committed split to derive profiles from")
    parser.add_argument("--num_rows", type=int, default=DEFAULT_TARGET_ROWS, help="Rows to generate (design target 300-500)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed")
    parser.add_argument("--out", default="data/promptpai/promptpai.jsonl", help="Output JSONL path")
    parser.add_argument("--manifest", default="data/promptpai/manifest.json", help="Output manifest path")
    parser.add_argument("--sample", action="store_true", help="Produce the small committed sample instead of the full set")
    args = parser.parse_args(argv)

    if args.sample:
        num_rows = DEFAULT_SAMPLE_ROWS
        out = "data/promptpai/sample.jsonl"
        manifest = "data/promptpai/manifest_sample.json"
    else:
        num_rows = args.num_rows
        out = args.out
        manifest = args.manifest

    rows = generate_promptpai(source_split=args.source_split, num_rows=num_rows, seed=args.seed)
    if not rows:
        raise SystemExit(f"No rows generated; source split '{args.source_split}' not found or empty.")
    manifest_obj = write_dataset(rows, out, manifest, source_split=args.source_split, seed=args.seed)
    print(f"Wrote {len(rows)} PromptPAI rows to {out}")
    print(f"Manifest: {manifest} (sha256={manifest_obj['sha256']})")


if __name__ == "__main__":
    main()
