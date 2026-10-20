# src/risk_model/taxonomy.py
"""Demographic attribute taxonomy and integer identifier mappings.

Defines coarse classification categories for age, education, occupation, and location,
ensuring balanced class support and consistent integer encoding across risk models.
"""

from typing import Any, Dict, List
import re
from src.contracts.schemas import AttributeName

ATTRIBUTE_TAXONOMY: Dict[AttributeName, List[str]] = {
    AttributeName.AGE: [
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55+",
    ],
    AttributeName.EDUCATION: [
        "high_school",
        "undergraduate",
        "graduate_professional",
        "other_unknown",
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


def map_raw_to_taxonomy_label(attr: AttributeName, raw_val: Any) -> str:
    """Map raw profile string or integer to coarse demographic taxonomy label.

    Handles fine-grained SynthPAI profile strings (e.g. city/state, exact degree,
    exact age) as well as existing coarse category labels and abbreviations.
    """
    if raw_val is None:
        return "UNKNOWN"

    val_str = str(raw_val).strip().lower()
    if not val_str or val_str in ("unknown", "none", "n/a", "na"):
        return "UNKNOWN"

    valid_labels = ATTRIBUTE_TAXONOMY[attr]
    if val_str in valid_labels:
        return val_str

    if attr == AttributeName.AGE:
        try:
            m = re.search(r"\d+", val_str)
            num = int(m.group(0)) if m else 30
            if num < 25:
                return "18-24"
            elif num < 35:
                return "25-34"
            elif num < 45:
                return "35-44"
            elif num < 55:
                return "45-54"
            else:
                return "55+"
        except Exception:
            return "25-34"

    elif attr == AttributeName.EDUCATION:
        if any(w in val_str for w in ["phd", "doctor", "master", "postgraduate", "jd", "md"]):
            return "graduate_professional"
        elif any(w in val_str for w in ["bachelor", "undergraduate", "college", "bs", "ba"]):
            return "undergraduate"
        elif any(w in val_str for w in ["high school", "secondary", "ged"]):
            return "high_school"
        return "other_unknown"

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
        elif any(w in val_str for w in ["asia", "japan", "tokyo", "china", "pacific", "australia"]):
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

    return valid_labels[0]


def map_to_taxonomy_id(attr: AttributeName, raw_val: Any) -> int:
    """Map raw profile string to coarse integer class ID."""
    label = map_raw_to_taxonomy_label(attr, raw_val)
    return LABEL_TO_ID[attr].get(label, 0)
