"""Microsoft Presidio explicit redaction baseline and hardness stratification.

Compares explicit PII scrubbing against inferential privacy protection,
stratifying inputs by inference hardness: direct vs indirect vs complicated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


_DURATION_RE = re.compile(
    r"^(?:about|around|nearly|over|under|almost|for|past|last|every|this|next|coming|following|previous|until)?\s*"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|many|a\s+few|a\s+couple(?:\s+of)?)?"
    r"\s*[-\s]?(?:seconds?|minutes?|hours?|days?|weeks?|weekends?|months?|years?|decades?|mornings?|afternoons?|evenings?|nights?|semesters?|quarters?|terms?|finals?|years?[-\s]*old)$",
    re.IGNORECASE,
)

_NON_IDENTIFYING_MONTH_OR_DAY_RE = re.compile(
    r"^(?:in\s+|by\s+|until\s+|before\s+|after\s+|since\s+)?"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|"
    r"spring|summer|fall|autumn|winter|today|tomorrow|yesterday|now|early|late|Q[1-4])s?$",
    re.IGNORECASE,
)

_QUARTER_RE = re.compile(r"^Q[1-4]$", re.IGNORECASE)
_ORDINAL_PHRASE_RE = re.compile(
    r"^\d+(?:st|nd|rd|th)[-\s]*(?:year|grade|floor|circuit|amendment|quarter|edition)s?$",
    re.IGNORECASE,
)
_STATUTE_NUM_RE = re.compile(
    r"^(?:\d+-\d+(?:-\d+)*|\d+\([a-z0-9]+\))$",
    re.IGNORECASE,
)
_STATUTE_PREFIX_MARKERS = ("§", "sec.", "section", "u.s.c.", "c.f.r.", "o.c.g.a.")

_SHARED_ANALYZER: Optional[Any] = None
_SHARED_ANONYMIZER: Optional[Any] = None


def get_shared_analyzer() -> Optional[Any]:
    """Return a process-wide cached Presidio AnalyzerEngine (avoids duplicate ~560MB spaCy loads)."""
    global _SHARED_ANALYZER
    if _SHARED_ANALYZER is None:
        try:
            from presidio_analyzer import AnalyzerEngine

            _SHARED_ANALYZER = AnalyzerEngine()
        except Exception as exc:
            logger.info("Presidio AnalyzerEngine unavailable (%s).", exc)
    return _SHARED_ANALYZER


def get_shared_anonymizer() -> Optional[Any]:
    """Return a process-wide cached Presidio AnonymizerEngine."""
    global _SHARED_ANONYMIZER
    if _SHARED_ANONYMIZER is None:
        try:
            from presidio_anonymizer import AnonymizerEngine

            _SHARED_ANONYMIZER = AnonymizerEngine()
        except Exception:
            pass
    return _SHARED_ANONYMIZER


def is_duration_phrase(span_text: str) -> bool:
    """Return True if span_text is a non-identifying duration/relative time/standalone month phrase."""
    s = span_text.strip()
    if not s:
        return False
    return bool(_DURATION_RE.match(s) or _NON_IDENTIFYING_MONTH_OR_DAY_RE.match(s))


def is_non_identifying_presidio_span(
    entity_type: str,
    span_text: str,
    full_text: str = "",
    start: int = 0,
) -> bool:
    """Return True when a Presidio span is a false positive (duration, ordinal, statute, or generic US)."""
    s = span_text.strip()
    if entity_type == "DATE_TIME":
        if is_duration_phrase(s):
            return True
        if _ORDINAL_PHRASE_RE.match(s) or _STATUTE_NUM_RE.match(s):
            return True
        if full_text and start > 0:
            prefix = full_text[max(0, start - 12) : start].lower()
            if any(marker in prefix for marker in _STATUTE_PREFIX_MARKERS):
                return True
        return False
    if entity_type == "LOCATION" and (_QUARTER_RE.match(s) or s in {"US", "U.S."}):
        return True
    return False


@dataclass(frozen=True)
class RedactionResult:
    """Result of baseline redaction."""

    original: str
    redacted: str
    entities_found: List[str]
    hardness: str  # "direct", "indirect", or "complicated"


class PresidioBaseline:
    """Baseline explicit PII redactor with Presidio and fallback regex engine."""

    def __init__(self, use_presidio_if_available: bool = True, analyzer: Optional[Any] = None) -> None:
        self.analyzer = None
        self.anonymizer = None

        if use_presidio_if_available:
            self.analyzer = analyzer if analyzer is not None else get_shared_analyzer()
            self.anonymizer = get_shared_anonymizer()
            if self.analyzer is None or self.anonymizer is None:
                logger.info("Presidio engines not available. Using built-in explicit PII engine.")

        # Explicit regex patterns for direct PII
        self.patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "URL": r"https?://\S+|www\.\S+",
            "DATE": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
            "PERSON": r"\b(?:My name is|I am|This is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            "LOCATION": r"\b(?:living in|moved to|commute to|located in|from|in)\s+([A-Z][a-z]+(?:,\s*[A-Z]{2})?)\b",
        }

    def redact(self, text: str) -> str:
        """Redact explicit PII entities from text.

        Returns:
            Sanitized text with explicit entities replaced by placeholder tags.
        """
        if not text or not text.strip():
            return ""

        if self.analyzer is not None and self.anonymizer is not None:
            try:
                results = self.analyzer.analyze(
                    text=text,
                    language="en",
                    entities=["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "DATE_TIME"],
                )
                results = [
                    r for r in results
                    if not is_non_identifying_presidio_span(
                        r.entity_type, text[r.start:r.end], full_text=text, start=r.start
                    )
                ]
                anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized.text
            except Exception as e:
                logger.warning(f"Presidio error during anonymization: {e}. Falling back to regex.")

        # Fallback regex redaction
        redacted = text
        for entity_type, pat in self.patterns.items():
            if entity_type in ("PERSON", "LOCATION"):
                # Replace group 1 if present
                def replacer(match: re.Match) -> str:
                    full = match.group(0)
                    g1 = match.group(1)
                    return full.replace(g1, f"<{entity_type}>")

                redacted = re.sub(pat, replacer, redacted)
            else:
                redacted = re.sub(pat, f"<{entity_type}>", redacted)

        return redacted

    def extract_entities(self, text: str) -> List[str]:
        """Extract list of detected explicit entity types."""
        entities: List[str] = []
        for entity_type, pat in self.patterns.items():
            if re.search(pat, text):
                entities.append(entity_type)
        return entities

    def classify_hardness(self, text: str) -> str:
        """Classify inference hardness into 'direct', 'indirect', or 'complicated'.

        Direct: Contains explicit PII (names, emails, phones, exact addresses).
        Indirect: Free of direct PII, but contains dialect, cultural, transit,
                  or demographic inferential cues.
        Complicated: Combines both explicit and subtle inferential cues.
        """
        explicit_entities = self.extract_entities(text)
        has_direct_pii = len(explicit_entities) > 0

        # Inferential cues (indirect markers)
        indirect_patterns = [
            r"\b(?:altitude|mile high|front range|rockies|red rocks)\b",  # Denver indirect
            r"\b(?:space needle|rainier|sound transit|pike place)\b",     # Seattle indirect
            r"\b(?:silicon valley|bart|caltrain|bay area)\b",             # Bay Area indirect
            r"\b(?:the tube|oyster card|borough|m25)\b",                  # London indirect
            r"\b(?:junior year|undergrad|dissertation|advisor|postdoc)\b",# Education indirect
            r"\b(?:on call|attending physician|rounds|scrubs)\b",         # Medical indirect
            r"\b(?:pull request|code review|sprint|standup|prod deployment)\b", # Tech indirect
            r"\b(?:in my 20s|when I was a kid in the 90s|retirement plan)\b",   # Age indirect
        ]

        text_lower = text.lower()
        has_indirect_cues = any(re.search(pat, text_lower) for pat in indirect_patterns)

        if has_direct_pii and has_indirect_cues:
            return "complicated"
        elif has_direct_pii:
            return "direct"
        elif has_indirect_cues:
            return "indirect"
        else:
            return "direct"

    def analyze_sample(self, text: str) -> RedactionResult:
        """Process a text sample with redaction, entity detection, and hardness classification."""
        redacted = self.redact(text)
        entities = self.extract_entities(text)
        hardness = self.classify_hardness(text)
        return RedactionResult(
            original=text,
            redacted=redacted,
            entities_found=entities,
            hardness=hardness,
        )

    def evaluate_stratified(
        self,
        samples: Sequence[Dict[str, Any]],
        inference_detector: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Evaluate baseline performance stratified across direct, indirect, and complicated samples."""
        stratified: Dict[str, List[Dict[str, Any]]] = {
            "direct": [],
            "indirect": [],
            "complicated": [],
        }

        for sample in samples:
            text = sample.get("text", "")
            res = self.analyze_sample(text)
            stratified[res.hardness].append({
                "sample": sample,
                "redaction": res,
            })

        summary: Dict[str, Any] = {}
        for hardness_level, items in stratified.items():
            count = len(items)
            summary[hardness_level] = {
                "count": count,
                "explicit_redaction_rate": (
                    sum(1 for it in items if it["redaction"].entities_found) / max(count, 1)
                ),
            }

        return summary
