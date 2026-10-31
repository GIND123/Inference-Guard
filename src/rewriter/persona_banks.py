"""Persona banks: externalized, data-driven PII value inventories.

Replaces the hardcoded regex lists (10 US cities, 10 job titles, 5 universities)
that used to live in ``src/rewriter/generate_training_data.py``. A regex list in
code fixes the *surface forms* the system can recognise, so the rewriter trained
on its output can only generalise over those exact strings - the opposite of
what a generalizing rewriter needs. Here the inventories are data files under
``data/rewriter/personas/`` that can be grown (by hand or with an LLM) without
touching code, and every consumer - the augmentation composer, the acceptance
gate, the no-adapter fallback and the benchmark annotator - reads the same
banks.

Torch-free and dependency-free: only the standard library.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BANKS_DIR = Path(__file__).resolve().parents[2] / "data" / "rewriter" / "personas"

# Banks whose entries are plain values (matched verbatim, case-insensitively).
VALUE_BANKS = ("cities", "occupations", "local_references", "education")
# Banks whose entries are regex patterns (age, income).
PATTERN_BANKS = ("age_expressions", "income_expressions")


@dataclass
class Match:
    """One recognized PII surface form inside a text."""

    attribute: str          # location / occupation / age / income / local_reference / education
    value: str              # canonical bank value or the matched surface form
    surface: str            # what actually appears in the text
    start: int              # character span
    end: int
    generalizations: List[str] = field(default_factory=list)


def _load_bank(name: str, banks_dir: Optional[Path] = None) -> Dict:
    path = (banks_dir or BANKS_DIR) / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_value_bank(name: str, banks_dir: Optional[Path] = None) -> List[Dict]:
    """Return the entry list of a value bank, normalized."""
    bank = _load_bank(name, banks_dir)
    return bank.get("entries", [])


def _phrase_pattern(surface: str) -> re.Pattern:
    """Case-insensitive pattern matching `surface` as a whole phrase."""
    escaped = re.escape(surface.strip())
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)


# Context guards shared by the fallback neutralizer and the benchmark annotator:
# a matched value followed by these markers belongs to a statute reference or a
# third-party subject, not to the author, and must NOT be neutralized.
STATUTE_FOLLOWERS = re.compile(
    # A capitalized statute/court reference starting right after the match:
    # "New York State General Business Law", "Tokyo Metropolitan Building
    # Ordinance", "Washington State RCW", "New York Commercial Division".
    r"\s+(?:State\s+)?[A-Z][\w\s]{0,30}?(?:Code|Act|Laws?|Statute|Ordinance|Regulation|"
    r"Chapter|Title|Section|Division|Court|RCW)\b"
)
THIRD_PARTY_FOLLOWERS = re.compile(
    r"\s+(?:patient|client|plaintiff|defendant|infant|child|female|male|woman|man|girl|boy)\b",
    re.IGNORECASE,
)
PROSPECTIVE_TARGET_HEAD = re.compile(
    r"\b(?:get into|getting into|apply to|applying to|admission to|admitted to|enrolling in|enrolled in|pursue|pursuing|target|targetting|acceptance into)\s+(?:an?\s+)?$",
    re.IGNORECASE,
)
TARGET_PROGRAM_INST = re.compile(
    r"\b(?:program at|degree at|school at|studies at|master(?:'s)? at|mba at|phd at)\s+$",
    re.IGNORECASE,
)


class PersonaMatcher:
    """Finds persona/PII surface forms from the banks inside arbitrary text.

    One matcher instance compiles the banks once and is reused across texts.
    """

    def __init__(self, banks_dir: Optional[Path] = None, banks: Optional[Sequence[str]] = None) -> None:
        self.banks_dir = Path(banks_dir) if banks_dir else BANKS_DIR
        self._value_patterns: List[Tuple[str, str, re.Pattern, List[str]]] = []
        self._pattern_banks: List[Tuple[str, re.Pattern]] = []
        self._education_entries: List[Dict] = []
        chosen = list(banks) if banks else list(VALUE_BANKS) + list(PATTERN_BANKS)
        for name in chosen:
            if name == "education":
                bank = _load_bank("education", self.banks_dir)
                attr = bank.get("attribute", "education")
                gens = bank.get("generalizations", [])
                for entry in bank.get("institutions", []):
                    surfaces = [entry["value"]] + entry.get("variants", [])
                    entry_gens = entry.get("generalizations") or gens
                    for surface in surfaces:
                        if not surface or not surface.strip():
                            continue
                        self._value_patterns.append(
                            (attr, entry["value"], _phrase_pattern(surface), list(entry_gens))
                        )
                # Degrees are weak signals; matched, but flagged as education.
                for degree in bank.get("degrees", []):
                    self._value_patterns.append(("education", degree, _phrase_pattern(degree), ["degree"]))
            elif name in VALUE_BANKS:
                bank = _load_bank(name, self.banks_dir)
                attr = bank.get("attribute", name)
                gens = bank.get("generalizations", [])
                for entry in bank.get("entries", []):
                    surfaces = [entry["value"]] + entry.get("variants", [])
                    entry_gens = entry.get("generalizations") or gens
                    for surface in surfaces:
                        if not surface or not surface.strip():
                            continue
                        self._value_patterns.append(
                            (attr, entry["value"], _phrase_pattern(surface), list(entry_gens))
                        )
            elif name in PATTERN_BANKS:
                bank = _load_bank(name, self.banks_dir)
                attr = bank.get("attribute", name)
                for entry in bank.get("entries", []):
                    self._pattern_banks.append((attr, re.compile(entry["pattern"], re.IGNORECASE)))

    @property
    def value_pattern_count(self) -> int:
        return len(self._value_patterns)

    def scan(self, text: str) -> List[Match]:
        """All bank matches in `text`, sorted by position, longest-first at ties."""
        matches: List[Match] = []
        for attr, value, pattern, gens in self._value_patterns:
            for m in pattern.finditer(text):
                matches.append(Match(attr, value, m.group(0), m.start(), m.end(), list(gens)))
        for attr, pattern in self._pattern_banks:
            for m in pattern.finditer(text):
                matches.append(Match(attr, m.group(0), m.group(0), m.start(), m.end(), []))
        # Drop overlapping matches (partial or complete), preferring longer matches
        matches.sort(key=lambda x: (-(x.end - x.start), x.start))
        kept: List[Match] = []
        for m in matches:
            if any(max(m.start, k.start) < min(m.end, k.end) for k in kept):
                continue
            kept.append(m)
        kept.sort(key=lambda x: x.start)
        return kept

    def attributes_in(self, text: str) -> List[str]:
        """Unique attributes present in `text` according to the banks."""
        return sorted({m.attribute for m in self.scan(text)})

    def contains_any(self, text: str, surfaces: Iterable[str]) -> List[str]:
        """Which of `surfaces` appear in `text` (case-insensitive, whole phrase)."""
        found = []
        for surface in surfaces:
            if _phrase_pattern(surface).search(text):
                found.append(surface)
        return found


_DEFAULT_MATCHER: Optional[PersonaMatcher] = None


def default_matcher() -> PersonaMatcher:
    """Process-wide cached matcher over all banks."""
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = PersonaMatcher()
    return _DEFAULT_MATCHER


def neutralize_text(text: str, matcher: Optional[PersonaMatcher] = None) -> str:
    """Replace every bank match with a safe generalization; remove age/income.

    Used by the no-adapter heuristic fallback (so it generalizes beyond any
    hardcoded city list) and to build deterministic targets for augmented
    training rows. Age and income matches are removed by policy; value matches
    are replaced by their first generalization (leading articles are stripped
    so the surrounding determiner survives).

    Context guards: matches that are part of a statute reference ("Tokyo
    Metropolitan Building Ordinance") or a third-party subject ("a 3-year-old
    patient") are left untouched - they are not facts about the author.
    """
    matcher = matcher or default_matcher()
    out = text
    for m in reversed(matcher.scan(text)):
        head = text[max(0, m.start - 50):m.start]
        tail = text[m.end:m.end + 40]
        if STATUTE_FOLLOWERS.match(tail) or (
            m.attribute in ("age", "income") and THIRD_PARTY_FOLLOWERS.match(tail)
        ):
            continue
        if m.attribute == "education":
            if PROSPECTIVE_TARGET_HEAD.search(head) or TARGET_PROGRAM_INST.search(head):
                continue
        if m.generalizations:
            replacement = re.sub(r"^(?:a|an|the)\s+", "", m.generalizations[0], flags=re.IGNORECASE)
        else:
            replacement = ""
        out = out[: m.start] + replacement + out[m.end :]
    # Tidy grammar artifacts left by mechanical substitution.
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"(?<!\w),\s*,", ",", out)
    out = re.sub(r"^\s*[,;:]\s*", "", out)
    # "as a my role" / "a a service job" -> keep only the later determiner.
    out = re.sub(r"\b(?:a|an|the)\s+(?=(?:a|an|the|my)\b)", "", out, flags=re.IGNORECASE)
    # Fix a/an agreement on the word that now follows the determiner.
    out = re.sub(r"\ba\s+(?=[aeiouAEIOU]|(?:MBA|MD|MS|MA|HR|LLC)\b)", "an ", out)
    out = re.sub(r"\ban\s+(?=[^aeiouAEIOU\s])(?!(?:MBA|MD|MS|MA|HR|LLC)\b)", "a ", out)
    return out.strip()
