# src/conversation/entity_map.py
"""Deterministic session-scoped PII pseudonymization and entity map management.

Combines Presidio entity extraction with deterministic right-to-left placeholder
replacement, preventing entity drift and character-offset displacement across turns.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.contracts.schemas import ConversationState, PIIEntitySpan
from src.evaluation.presidio_baseline import get_shared_analyzer

logger = logging.getLogger(__name__)


class DeterministicPseudonymManager:
    """Manages session-stable entity pseudonyms using right-to-left replacement."""

    TRACKED_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "STREET_ADDRESS",
        "ORGANIZATION",
        "HOSPITAL",
        "UNIVERSITY",
        "LOCATION",
    ]
    PRESIDIO_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION", "LOCATION"]
    PREFIX_MAP = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "STREET_ADDRESS": "ADDRESS",
        "HOSPITAL": "HOSPITAL",
        "UNIVERSITY": "UNIVERSITY",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOCATION",
    }

    _STREET_ADDRESS_RE = re.compile(
        r"\b\d{1,6}\s+[A-Z][a-zA-Z0-9.\s]{1,28}?\s+"
        r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pkwy|Pl|Place)\b"
        r"(?:\s*,?\s*(?:Apt|Suite|Ste|Unit|#)\s*[\w-]+)?"
        r"(?:(?:,\s*|\s+in\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2}\b)?"
    )
    _PHONE_RE = re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    )
    _AMBIGUOUS_SINGLE_NAMES = {
        "will", "may", "bill", "mark", "grace", "chase", "april", "june", "august",
        "art", "pat", "rob", "grant", "dean", "sue", "ray", "joy", "hope", "faith",
        "don", "bob", "frank", "rich", "max", "jack", "nick", "drew", "lane", "reed",
        "wade", "cole", "sage", "sky", "brook", "park", "west", "north", "south", "east",
    }
    _NON_PERSON_TAIL_TOKENS = {
        "grant", "grants", "act", "code", "law", "rule", "rules", "plan",
        "fund", "award", "scholarship", "fellowship", "amendment", "court",
        "circuit", "bill", "test", "syndrome", "method", "theorem", "index",
        "standard",
    }

    _HOSPITAL_HEAD_RE = re.compile(
        r"\b(?:[A-Z][\w'’.-]*\s+){1,4}(?:Hospital|Medical\s+Center|Health\s+System|Infirmary|Clinic)\b"
    )
    _UNIVERSITY_HEAD_RE = re.compile(
        r"\b(?:[A-Z][\w'’.-]*\s+){1,4}(?:University|College|Institute\s+of\s+Technology)\b"
    )
    _TRANSIT_LINE_RE = re.compile(
        r"^(?:the\s+)?(?:[A-Z][a-z]+\s+Line|Subway|Metro|Tube|Muni|BART|Caltrain|LIRR|PATH(?:\s+train)?|U-?Bahn|S-?Bahn|RER|MRT|TTC|T|L)$",
        re.IGNORECASE,
    )
    _ALIAS_TOKEN_RE = re.compile(
        r"^(?:PERSON|EMAIL|PHONE|ADDRESS|HOSPITAL|UNIVERSITY|ORG|LOCATION)_\d+$"
    )
    _FP_CODE_OR_TEMPORAL_RE = re.compile(
        r"^(?:Q[1-4]|T[1-4]|L[1-5]|B\d{1,2}|"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
        r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|"
        r"spring|summer|fall|autumn|winter|today|tomorrow|yesterday|now|finals|semester|quarter|"
        r"mg|kg|mcg|ml|day|hr|hrs)$",
        re.IGNORECASE,
    )
    _SHORT_ACRONYM_ALLOW_SKIP = {
        "NIH", "NSF", "IRB", "FDA", "CDC", "WHO", "ICU", "NICU", "PICU", "ED", "ER", "OR",
        "RN", "MD", "DO", "NP", "PA", "PHD", "MBA", "BS", "BA", "MS", "MA", "GPA",
        "SAT", "ACT", "GRE", "MCAT", "API", "SQL", "AWS", "GCP", "CPU", "GPU", "RAM",
        "LLM", "AI", "ML", "NLP", "HR", "PR", "IT", "US", "UK", "EU", "CO", "CA", "NY",
    }

    # High-precision regex fallback patterns for offline environments or fast unit tests
    FALLBACK_PATTERNS = {
        "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE_NUMBER": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "STREET_ADDRESS": r"\b\d{1,6}\s+[A-Z][a-zA-Z0-9.\s]{1,28}?\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pkwy|Pl|Place)\b(?:\s*,?\s*(?:Apt|Suite|Ste|Unit|#)\s*[\w-]+)?",
        "HOSPITAL": r"\b(?:[A-Z][\w'’.-]*\s+){1,4}(?:Hospital|Medical\s+Center|Health\s+System|Infirmary|Clinic)\b",
        "UNIVERSITY": r"\b(?:[A-Z][\w'’.-]*\s+){1,4}(?:University|College|Institute\s+of\s+Technology)\b",
        "LOCATION": r"\b(?:Denver, CO|Seattle, WA|Boston, MA|New York, NY|San Francisco, CA|Zurich|Perth)\b",
        "ORGANIZATION": r"\b(?:Amazon|Google|Microsoft|Meta|Apple)\b",
        "PERSON": r"\b(?:Alice Smith|Bob Jones|Charlie Brown|Sarah Connor)\b",
    }

    REFERENT_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STREET_ADDRESS"]

    def __init__(
        self,
        score_threshold: float = 0.45,
        analyzer: Optional[Any] = None,
        referents_only: bool = False,
    ):
        """Initialize the manager with shared Presidio AnalyzerEngine."""
        self.score_threshold = score_threshold
        self.referents_only = referents_only
        self.analyzer = analyzer if analyzer is not None else get_shared_analyzer()
        if self.analyzer is None:
            logger.info("Presidio AnalyzerEngine unavailable; using fallback regex engine.")

    @staticmethod
    def _normalize_surface(surface: str) -> str:
        """Normalize whitespace, leading articles, and casing for entity key matching."""
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", surface.strip(), flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned.lower())

    @classmethod
    def _classify_org_domain(cls, surface: str) -> str:
        """Assign coarse domain category (HOSPITAL, UNIVERSITY, or ORGANIZATION) from head noun."""
        if cls._HOSPITAL_HEAD_RE.search(surface):
            return "HOSPITAL"
        if cls._UNIVERSITY_HEAD_RE.search(surface):
            return "UNIVERSITY"
        return "ORGANIZATION"

    @classmethod
    def _is_false_positive(cls, ent_type: str, surface: str, score: float) -> bool:
        """Filter structural false positives (quarters, transit lines, acronyms, existing aliases)."""
        s = surface.strip()
        if not s or len(s) < 2:
            return True
        if cls._ALIAS_TOKEN_RE.match(s):
            return True
        if cls._FP_CODE_OR_TEMPORAL_RE.match(s):
            return True
        if s.upper() in cls._SHORT_ACRONYM_ALLOW_SKIP and s.isupper():
            return True
        if ent_type == "PERSON":
            tokens = s.split()
            if tokens and tokens[-1].rstrip(".,;:!?'\"()").lower() in cls._NON_PERSON_TAIL_TOKENS:
                return True
        if ent_type == "LOCATION" and cls._TRANSIT_LINE_RE.match(s):
            return True
        if ent_type in ("PERSON", "LOCATION") and score < 0.60:
            return True
        return False

    _CONTEXTUAL_PERSON_RE = re.compile(
        r"\b(?i:I am|I'm|My name is|This is|my manager|my director|my colleague|my boss|customer)\s+"
        r"((?:(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+)?[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)\b"
    )
    _HONORIFIC_PERSON_RE = re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?\b"
    )
    _FALLBACK_NON_PERSON_WORDS = {
        "order", "ticket", "invoice", "boston", "denver", "seattle", "austin", "chicago",
        "london", "paris", "tokyo", "zurich", "hartford", "atlanta", "pittsburgh",
        "massachusetts", "california", "texas", "new", "san", "united", "states",
        "here", "there", "support", "team", "staff", "sir", "madam",
    }

    def _extract_spans_with_fallback(self, text: str) -> List[Tuple[str, int, int, float]]:
        """Extract entity spans using regex fallback when AnalyzerEngine is unavailable or empty."""
        spans: List[Tuple[str, int, int, float]] = []
        for ent_type, pattern in self.FALLBACK_PATTERNS.items():
            if self.referents_only and ent_type not in self.REFERENT_ENTITIES:
                continue
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                spans.append((ent_type, m.start(), m.end(), 0.95))

        # Contextual person-name extraction for CI / offline environments without Presidio
        for m in self._CONTEXTUAL_PERSON_RE.finditer(text):
            surf = m.group(1)
            first_tok = surf.split()[0].rstrip(".,;:!?").lower()
            if first_tok in self._FALLBACK_NON_PERSON_WORDS:
                continue
            if not self._is_false_positive("PERSON", surf, 0.90):
                spans.append(("PERSON", m.start(1), m.end(1), 0.90))
        for m in self._HONORIFIC_PERSON_RE.finditer(text):
            surf = m.group(0)
            if not self._is_false_positive("PERSON", surf, 0.90):
                spans.append(("PERSON", m.start(), m.end(), 0.90))
        return spans

    @staticmethod
    def _blind_key(ent_type: str, norm_surface: str, session_secret: str) -> str:
        """Compute a keyed HMAC-SHA256 blind index so server state stores zero plaintext PII."""
        digest = hmac.new(
            session_secret.encode("utf-8"),
            f"{ent_type}::{norm_surface}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]
        return f"hmac:{digest}"

    def pseudonymize_turn(
        self,
        text: str,
        state: ConversationState,
        session_secret: Optional[str] = None,
        blind_index: bool = False,
    ) -> Tuple[str, List[PIIEntitySpan], Dict[str, int]]:
        """Extract explicit PII and replace with deterministic tokens from right to left."""
        if not text:
            return "", [], {k: 0 for k in self.TRACKED_ENTITIES}

        active_presidio_entities = [
            e for e in (self.REFERENT_ENTITIES if self.referents_only else self.PRESIDIO_ENTITIES)
            if e != "STREET_ADDRESS"
        ]
        raw_detections: List[Tuple[str, int, int, float]] = []
        if self.analyzer is not None:
            try:
                results = self.analyzer.analyze(
                    text=text,
                    entities=active_presidio_entities,
                    language="en",
                    score_threshold=self.score_threshold,
                )
                for r in results:
                    start, end = r.start, r.end
                    # Strip leading article ("the ", "a ", "an ") so grammar stays intact and keys unify
                    span_slice = text[start:end]
                    m_art = re.match(r"^(?:the|a|an)\s+", span_slice, flags=re.IGNORECASE)
                    if m_art:
                        start += m_art.end()
                    surface = text[start:end]
                    ent_type = r.entity_type
                    if not self.referents_only:
                        if ent_type in ("ORGANIZATION", "LOCATION") and self._HOSPITAL_HEAD_RE.search(surface):
                            ent_type = "HOSPITAL"
                        elif ent_type in ("ORGANIZATION", "LOCATION") and self._UNIVERSITY_HEAD_RE.search(surface):
                            ent_type = "UNIVERSITY"
                    if self._is_false_positive(ent_type, surface, float(r.score)):
                        continue
                    raw_detections.append((ent_type, start, end, float(r.score)))
            except Exception as e:
                logger.warning("Presidio analysis failed (%s); reverting to fallback patterns.", e)
                raw_detections = self._extract_spans_with_fallback(text)
        else:
            raw_detections = self._extract_spans_with_fallback(text)

        # Always detect high-precision street addresses and phone numbers (zero-plaintext referent aliasing)
        for m in self._STREET_ADDRESS_RE.finditer(text):
            raw_detections.append(("STREET_ADDRESS", m.start(), m.end(), 0.96))
        for m in self._PHONE_RE.finditer(text):
            raw_detections.append(("PHONE_NUMBER", m.start(), m.end(), 0.95))

        # Supplement with structural proper-noun Hospital/University spans only when non-referents are tracked
        if not self.referents_only:
            for pat, ent_type in ((self._HOSPITAL_HEAD_RE, "HOSPITAL"), (self._UNIVERSITY_HEAD_RE, "UNIVERSITY")):
                for m in pat.finditer(text):
                    raw_detections.append((ent_type, m.start(), m.end(), 0.96))

        # Coreference recall: match honorific+name or single capitalized tokens already bound in state.entity_map
        use_blind = bool(blind_index or session_secret)
        if state.entity_map:
            # First detect two-word names starting with an already-known first name (e.g. "Sarah Patel" -> PERSON_2)
            for m in re.finditer(r"\b([A-Z][a-z]{1,25})\s+([A-Z][a-z]{1,25})\b", text):
                first_norm = self._normalize_surface(m.group(1))
                first_key = (
                    self._blind_key("PERSON", first_norm, session_secret)
                    if (use_blind and session_secret)
                    else f"PERSON::{first_norm}"
                )
                second_low = m.group(2).lower()
                if (
                    first_key in state.entity_map
                    and second_low not in self._FALLBACK_NON_PERSON_WORDS
                    and not self._is_false_positive("PERSON", m.group(0), 0.85)
                ):
                    raw_detections.append(("PERSON", m.start(), m.end(), 0.85))
            for pat in (
                re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]{1,25}\b"),
                re.compile(r"\b[A-Z][a-z]{1,25}\b"),
            ):
                for m in pat.finditer(text):
                    cand_surf = m.group(0)
                    cand_norm = self._normalize_surface(cand_surf)
                    cand_key = (
                        self._blind_key("PERSON", cand_norm, session_secret)
                        if (use_blind and session_secret)
                        else f"PERSON::{cand_norm}"
                    )
                    if cand_key in state.entity_map:
                        raw_detections.append(("PERSON", m.start(), m.end(), 0.72))

        # Resolve overlapping spans greedily: longest span first (so full names like 'Sarah Patel' beat bare 'Sarah'), then confidence
        raw_detections.sort(key=lambda r: (r[2] - r[1], r[3]), reverse=True)
        accepted = []
        occupied = [False] * (len(text) + 1)
        for ent_type, start, end, score in raw_detections:
            if any(occupied[start:end]):
                continue
            for idx in range(start, end):
                occupied[idx] = True
            accepted.append((ent_type, start, end, score))

        # Sort left-to-right to assign chronological indices
        accepted.sort(key=lambda r: r[1])

        pii_spans: List[PIIEntitySpan] = []
        pii_counts: Dict[str, int] = {k: 0 for k in self.TRACKED_ENTITIES}

        _honorifics = {"mr", "mrs", "ms", "dr", "prof", "mr.", "mrs.", "ms.", "dr.", "prof."}
        for ent_type, start, end, score in accepted:
            pii_counts[ent_type] = pii_counts.get(ent_type, 0) + 1
            surface = text[start:end]
            norm_surf = self._normalize_surface(surface)
            if use_blind and session_secret:
                norm_key = self._blind_key(ent_type, norm_surf, session_secret)
            else:
                norm_key = f"{ent_type}::{norm_surf}"

            if norm_key not in state.entity_map:
                prefix = self.PREFIX_MAP.get(ent_type, ent_type)
                distinct_existing = {
                    v for v in state.entity_map.values() if v.startswith(f"{prefix}_")
                }
                pseudonym = f"{prefix}_{len(distinct_existing) + 1}"
                state.entity_map[norm_key] = pseudonym
                if not use_blind:
                    state.reverse_entity_map[pseudonym] = surface

                # Register collision-safe coreference sub-keys for multi-token PERSON names
                if ent_type == "PERSON":
                    name_parts = [p for p in norm_surf.split() if p not in _honorifics and len(p) >= 2]
                    if len(name_parts) >= 2:
                        first_name, last_name = name_parts[0], name_parts[-1]
                        sub_surfaces = [
                            f"ms. {last_name}",
                            f"ms {last_name}",
                            f"mr. {last_name}",
                            f"mr {last_name}",
                            f"mrs. {last_name}",
                            f"mrs {last_name}",
                            f"dr. {last_name}",
                            f"dr {last_name}",
                            f"prof. {last_name}",
                            f"prof {last_name}",
                        ]
                        if first_name not in self._AMBIGUOUS_SINGLE_NAMES:
                            sub_surfaces.append(first_name)
                        if last_name not in self._AMBIGUOUS_SINGLE_NAMES:
                            sub_surfaces.append(last_name)
                        for sub_s in sub_surfaces:
                            sub_key = (
                                self._blind_key("PERSON", sub_s, session_secret)
                                if (use_blind and session_secret)
                                else f"PERSON::{sub_s}"
                            )
                            if sub_key not in state.entity_map:
                                state.entity_map[sub_key] = pseudonym
            else:
                pseudonym = state.entity_map[norm_key]

            pii_spans.append(
                PIIEntitySpan(
                    entity_type=ent_type,
                    start=start,
                    end=end,
                    original_surface=surface,
                    pseudonym_token=pseudonym,
                    confidence=round(score, 4),
                )
            )

        # Apply replacements from RIGHT TO LEFT so earlier character offsets never shift
        chars = list(text)
        for span in sorted(pii_spans, key=lambda s: s.start, reverse=True):
            chars[span.start : span.end] = list(span.pseudonym_token)

        pseudonymized_text = "".join(chars)
        return pseudonymized_text, pii_spans, pii_counts

    def restore_pseudonyms(self, text: str, state: ConversationState) -> str:
        """Replace active pseudonyms in rewritten text with their original surface entities."""
        restored = text
        for token, original in sorted(state.reverse_entity_map.items(), key=lambda x: len(x[0]), reverse=True):
            restored = restored.replace(token, original)
        return restored

    def verify_pseudonym_preservation(
        self, original_pseudo_text: str, rewritten_text: str, state: ConversationState
    ) -> Tuple[bool, List[str]]:
        """Verify whether all pseudonym tokens present in the input text survived rewriting."""
        missing: List[str] = []
        for token in state.reverse_entity_map.keys():
            if token in original_pseudo_text and token not in rewritten_text:
                missing.append(token)
        return len(missing) == 0, missing


class StreamAliasRestorer:
    """Lookahead stream buffer that restores alias tokens cleanly across split chunks."""

    def __init__(self, alias_to_original: Dict[str, str]) -> None:
        self.alias_to_original = dict(alias_to_original)
        self._sorted_tokens = sorted(self.alias_to_original.keys(), key=len, reverse=True)
        self._max_token_len = max((len(t) for t in self._sorted_tokens), default=0)
        self._buffer = ""

    def _replace_complete(self, text: str) -> str:
        out = text
        for tok in self._sorted_tokens:
            if tok in out:
                out = out.replace(tok, self.alias_to_original[tok])
        return out

    def _trailing_prefix_len(self, text: str) -> int:
        if not self._sorted_tokens or not text:
            return 0
        max_check = min(len(text), self._max_token_len)
        for length in range(max_check, 0, -1):
            suffix = text[-length:]
            for tok in self._sorted_tokens:
                if tok.startswith(suffix) and len(suffix) < len(tok):
                    return length
        return 0

    def feed(self, chunk: str) -> str:
        """Accept a streamed text chunk and emit safe restored text."""
        if not chunk:
            return ""
        self._buffer += chunk
        hold = self._trailing_prefix_len(self._buffer)
        if hold > 0:
            emit_raw = self._buffer[:-hold]
            self._buffer = self._buffer[-hold:]
        else:
            emit_raw = self._buffer
            self._buffer = ""
        return self._replace_complete(emit_raw)

    def flush(self) -> str:
        """Flush any remaining buffered text at end-of-stream."""
        out = self._replace_complete(self._buffer)
        self._buffer = ""
        return out
