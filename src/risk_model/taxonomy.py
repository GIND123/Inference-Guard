# src/risk_model/taxonomy.py
"""Demographic attribute taxonomy and integer identifier mappings.

Defines coarse classification categories for age, education, occupation, and location,
ensuring balanced class support and consistent integer encoding across risk models.

Two reconciliation decisions are recorded here (see docs/methodology-review.md
dated note 2026-11, decisions #4 and #7):

- EDUCATION adopts the SynthPAI paper's four-way scale (high_school / college /
  master / phd) for comparability with the paper. It replaces the previous ad
  hoc split (high_school / undergraduate / graduate_professional / other_unknown).
- LOCATION keeps the coarse regional buckets ONLY as an auxiliary/diagnostic
  signal for the multi-task heads. The reportable location metric is the
  free-text generate-and-score path in src/evaluation/attacker.py
  (Phi4MiniAttacker.evaluate_success), which scores a guessed free-text
  city/state/region token-wise against the ground truth. The Europe-vs-Other
  style collapse makes any coarse location accuracy number near-meaningless, so
  it must not be reported as the headline location result.
"""

import re
from typing import Dict, List
from src.contracts.schemas import AttributeName

ATTRIBUTE_TAXONOMY: Dict[AttributeName, List[str]] = {
    AttributeName.AGE: [
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55+",
    ],
    # SynthPAI paper's four-way education scale (high school / college / master /
    # PhD; paper support college 38.3% / master 37.7% / PhD 15.7% / high school
    # 8.3%). Canonical snake_case keys keep comparability with the paper while
    # matching the project's label-key style. The paper scale has no "other"
    # bucket, so map_to_taxonomy_id in train.py defaults unmatched text to
    # "high_school" (the least-attainment, lowest-information class). See
    # docs/methodology-review.md dated note (2026-11) for the rationale.
    AttributeName.EDUCATION: [
        "high_school",
        "college",
        "master",
        "phd",
    ],
    AttributeName.OCCUPATION: [
        "student",
        "technology",
        "healthcare",
        "education_academic",
        "business_finance",
        "service_retail",
        "other",
    ],
    AttributeName.LOCATION: [
        "us_northeast",
        "us_midwest",
        "us_south",
        "us_west",
        "uk_europe",
        "canada",
        "asia_pacific",
        "other_global",
    ],
}

LABEL_TO_ID: Dict[AttributeName, Dict[str, int]] = {
    attr: {label: idx for idx, label in enumerate(labels)}
    for attr, labels in ATTRIBUTE_TAXONOMY.items()
}

ID_TO_LABEL: Dict[AttributeName, Dict[int, str]] = {
    attr: {idx: label for idx, label in enumerate(labels)}
    for attr, labels in ATTRIBUTE_TAXONOMY.items()
}


def normalize_to_taxonomy_label(attr: AttributeName, raw_val: object) -> str:
    """Map a free-text attribute value to its coarse taxonomy label.

    SynthPAI profile/attributes ground truth is free text ("41",
    "Bangkok, Thailand", "Masters in Business Administration",
    "business consultant"), while the held-out adversary and the categorical
    heads predict coarse taxonomy classes ("35-44", "asia_pacific", "master",
    "business_finance"). This is the single source of truth that collapses the
    free text into the taxonomy label space so the two can be compared on the
    same scale. ``map_to_taxonomy_id`` in ``src/risk_model/train.py`` delegates
    here so the training and evaluation mappings never drift apart.

    An empty / missing value maps to "UNKNOWN" (an abstention on the ground
    truth side) rather than to a default class, so a blank ground truth is not
    silently scored as a concrete label.
    """
    val_str = str(raw_val or "").strip().lower()
    if not val_str:
        return "UNKNOWN"

    if attr == AttributeName.AGE:
        m = re.search(r"\d+", val_str)
        if m is None:
            return "UNKNOWN"
        num = int(m.group(0))
        if num < 25:
            return "18-24"
        elif num < 35:
            return "25-34"
        elif num < 45:
            return "35-44"
        elif num < 55:
            return "45-54"
        return "55+"

    elif attr == AttributeName.EDUCATION:
        if any(w in val_str for w in ["phd", "ph.d", "doctor", "doctoral", "postdoc"]):
            return "phd"
        elif any(w in val_str for w in ["master", "msc", "mba", "postgraduate", "jd", "md"]):
            return "master"
        elif any(w in val_str for w in ["bachelor", "undergraduate", "college", "university", "bs", "ba"]):
            return "college"
        elif any(w in val_str for w in ["high school", "secondary", "ged"]):
            return "high_school"
        return "high_school"

    elif attr == AttributeName.OCCUPATION:
        if any(w in val_str for w in ["student", "intern", "co-op"]):
            return "student"
        elif any(w in val_str for w in ["software", "developer", "engineer", "tech", "programmer", "data", "it"]):
            return "technology"
        elif any(w in val_str for w in ["doctor", "nurse", "physician", "psycholog", "clinic", "health"]):
            return "healthcare"
        elif any(w in val_str for w in ["teacher", "professor", "academic", "educat", "school"]):
            return "education_academic"
        elif any(w in val_str for w in ["finance", "bank", "account", "manager", "business", "analyst"]):
            return "business_finance"
        elif any(w in val_str for w in ["retail", "service", "barista", "waiter", "sales"]):
            return "service_retail"
        return "other"

    elif attr == AttributeName.LOCATION:
        if any(w in val_str for w in ["canada", "montreal", "toronto", "vancouver"]):
            return "canada"
        elif any(w in val_str for w in ["uk", "london", "england", "europe", "germany", "france", "berlin", "paris"]):
            return "uk_europe"
        elif any(w in val_str for w in ["asia", "japan", "tokyo", "china", "pacific", "australia", "thailand", "bangkok", "singapore", "india"]):
            return "asia_pacific"
        elif any(w in val_str for w in ["ny", "new york", "boston", "massachusetts", "northeast"]):
            return "us_northeast"
        elif any(w in val_str for w in ["wa", "seattle", "ca", "california", "colorado", "denver", "west"]):
            return "us_west"
        elif any(w in val_str for w in ["chicago", "illinois", "midwest", "ohio"]):
            return "us_midwest"
        elif any(w in val_str for w in ["tx", "texas", "austin", "atlanta", "georgia", "south"]):
            return "us_south"
        return "other_global"

    return "UNKNOWN"
