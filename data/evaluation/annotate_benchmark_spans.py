"""Annotate the 50-case benchmark with PII spans and invariant spans.

Adds two reviewable fields to each case:

  pii_spans:      [{"text": "Johns Hopkins", "attribute": "occupation"}, ...]
                  personal-identity surfaces in the input that must NOT survive
                  the rewrite (matched against data/rewriter/personas banks plus
                  transparent age/salary patterns).
  invariant_spans: ["California", "ASCE 7-22", ...]
                  domain entities that MUST survive the rewrite (jurisdictions
                  when they are the subject of the question, standards, clinical
                  terms).

This is annotation tooling driven by the same data banks the rest of the
system uses - not a per-case hardcoded checklist. The annotated file is what
scripts/evaluate_benchmark_50.py scores against; review the output before
trusting it (it prints anything it could not annotate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rewriter.persona_banks import STATUTE_FOLLOWERS, THIRD_PARTY_FOLLOWERS, default_matcher

# Domain entities that must survive: standards/code designators (ASCE 7-22,
# IBC 2021, GDPR, HIPAA, ISO 9001), plus jurisdictions when the case flags
# them as required. Clinical subjects ("a 4-year-old patient") are third
# parties, not author attributes, and are preserved as invariants too.
_STANDARD_PATTERN = re.compile(
    r"\b(?:ASCE\s?7-?\d{2}|IBC\s?\d{4}|ACI\s?318|AISC\s?360|GDPR|HIPAA|CCPA|"
    r"ISO\s?\d{4,5}|IEC\s?\d{4,5}|UL\s?\d{3,4}|NFPA\s?\d{2,3}|HER2(?:-positive)?|"
    r"Schengen|JLPT\s?N\d|Cauchy'?s Integral Formula)\b",
    re.IGNORECASE,
)
_THIRD_PARTY_PATTERN = re.compile(
    r"\b(?:a\s+)?\d{1,2}-(?:year|month|week)-old\s+patient\b", re.IGNORECASE
)


# A matched city/age immediately followed by these markers belongs to a statute
# reference or a third-party subject, not to the author. (Shared guards from
# src.rewriter.persona_banks, so the annotator and the fallback agree.)
_STATUTE_FOLLOWERS = STATUTE_FOLLOWERS
_THIRD_PARTY_FOLLOWERS = THIRD_PARTY_FOLLOWERS


def annotate_case(case: Dict, matcher) -> Dict:
    text = case["input"]
    pii = []
    invariants: List[str] = []
    seen = set()
    for m in matcher.scan(text):
        key = (m.surface.lower(), m.attribute)
        if key in seen:
            continue
        seen.add(key)
        tail = text[m.end:m.end + 40]
        if _STATUTE_FOLLOWERS.match(tail):
            # "Washington State RCW 59.18": a statute jurisdiction, not the author's location.
            if m.surface not in invariants:
                invariants.append(m.surface)
            continue
        if m.attribute == "age" and _THIRD_PARTY_FOLLOWERS.match(tail):
            # "an 84-year-old patient": a third party's age, preserved as an invariant.
            surface = (m.surface + _THIRD_PARTY_FOLLOWERS.match(tail).group(0)).strip()
            if surface not in invariants:
                invariants.append(surface)
            continue
        pii.append({"text": m.surface, "attribute": m.attribute})

    if case.get("jurisdiction_or_entity_required"):
        for entity in ("California", "Texas", "New York", "Florida"):
            if re.search(rf"\b{entity}\b", text) and entity not in invariants:
                invariants.append(entity)
        if re.search(r"\bpatient\b", text, re.IGNORECASE) and "patient" not in invariants:
            invariants.append("patient")
    for pat in (_STANDARD_PATTERN, _THIRD_PARTY_PATTERN):
        for m2 in pat.finditer(text):
            surface = m2.group(0)
            if surface not in invariants:
                invariants.append(surface)

    out = dict(case)
    out["pii_spans"] = pii
    out["invariant_spans"] = invariants
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/evaluation/benchmark_test_50.jsonl")
    parser.add_argument("--output", default="data/evaluation/benchmark_test_50_annotated.jsonl")
    args = parser.parse_args(argv)

    matcher = default_matcher()
    cases = [json.loads(l) for l in Path(args.input).open(encoding="utf-8") if l.strip()]
    annotated = [annotate_case(c, matcher) for c in cases]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in annotated:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_spanless = sum(1 for c in annotated if not c["pii_spans"])
    print(f"annotated {len(annotated)} cases -> {out_path}")
    print(f"cases with zero pii_spans (review these): {n_spanless}")
    for c in annotated:
        if not c["pii_spans"]:
            print(f"  {c['id']}: {c['input'][:100]}")
    total = sum(len(c["pii_spans"]) for c in annotated)
    print(f"total pii spans: {total}; total invariant spans: "
          f"{sum(len(c['invariant_spans']) for c in annotated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
