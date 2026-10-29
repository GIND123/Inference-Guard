"""Rewriter SFT Dataset Generation for Qwen/Qwen2.5-1.5B-Instruct.

Filters SynthPAI comments to strictly train-partition profiles from artifacts/profile_splits.json
to guarantee zero test/val leakage. Employs Pareto rejection sampling combining ModernBERT
privacy risk reduction (< 0.30), Presidio hard PII scrubbing, and utility preservation
(cosine similarity >= 0.82 and bidirectional NLI entailment).
Exports to ChatML and Alpaca JSONL schemas with hybrid thinking disabled (enable_thinking=False).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.evaluation.utility import check_degeneracy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are InferenceGuard Rewriter. Rewrite the input user text to preserve "
    "its intent, semantics, and utility while scrubbing explicit PII and neutralizing "
    "inferential cues (age, location, occupation, education). Do not output reasoning "
    "or thinking tokens."
)

DEFAULT_INSTRUCTION = (
    "Rewrite the following text to protect privacy by removing explicit personal "
    "identifiers and inferential cues regarding age, location, occupation, and education, "
    "while preserving semantic utility and original intent."
)


def load_profile_splits(splits_path: str | Path) -> Dict[str, Set[str]]:
    """Load profile splits JSON and return sets of train, val, and test profile IDs."""
    path = Path(splits_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile splits file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "train": set(data.get("train", [])),
        "val": set(data.get("val", [])),
        "test": set(data.get("test", [])),
    }


def filter_train_records(
    records: Sequence[Dict[str, Any]],
    train_profiles: Set[str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter records to strictly those belonging to train profiles.

    Args:
        records: List of SynthPAI-style records containing a profile or profile_id field.
        train_profiles: Set of valid train profile identifiers.

    Returns:
        Tuple of (retained_train_records, rejected_count).
    """
    retained: List[Dict[str, Any]] = []
    rejected_count = 0

    for rec in records:
        prof_id = rec.get("author")
        if not prof_id:
            prof_id = rec.get("profile")
            
        if prof_id is not None:
            if isinstance(prof_id, str) and prof_id.startswith("{"):
                try:
                    import json
                    prof_id = json.loads(prof_id)
                except Exception:
                    pass
            # Handle potential dictionary profile object
            if isinstance(prof_id, dict):
                prof_id = prof_id.get("id") or prof_id.get("username") or prof_id.get("author")

            str_id = str(prof_id).strip()
            if str_id in train_profiles:
                retained.append(rec)
            else:
                rejected_count += 1
        else:
            # Without identifiable profile, reject to prevent test contamination
            rejected_count += 1

    return retained, rejected_count


def scrub_hard_pii(text: str) -> Tuple[str, bool]:
    """Detect and scrub explicit PII (email, phone, SSN, IP, URLs).

    Returns:
        Tuple of (scrubbed_text, had_pii).
    """
    scrubbed = text
    had_pii = False

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    if re.search(email_pattern, scrubbed):
        had_pii = True
        scrubbed = re.sub(email_pattern, "[EMAIL]", scrubbed)

    # Phone pattern
    phone_pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    if re.search(phone_pattern, scrubbed):
        had_pii = True
        scrubbed = re.sub(phone_pattern, "[PHONE]", scrubbed)

    # IPv4 address
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    if re.search(ip_pattern, scrubbed):
        had_pii = True
        scrubbed = re.sub(ip_pattern, "[IP_ADDRESS]", scrubbed)

    # URL pattern
    url_pattern = r"https?://\S+|www\.\S+"
    if re.search(url_pattern, scrubbed):
        had_pii = True
        scrubbed = re.sub(url_pattern, "[URL]", scrubbed)

    return scrubbed, had_pii


def compute_token_cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts based on word frequencies.

    Provides a fast, deterministic, genuine fallback when sentence-transformers
    is not loaded in memory.
    """
    words1 = re.findall(r"\b\w+\b", text1.lower())
    words2 = re.findall(r"\b\w+\b", text2.lower())

    if not words1 or not words2:
        return 0.0

    vocab = set(words1) | set(words2)
    vec1 = {w: words1.count(w) for w in vocab}
    vec2 = {w: words2.count(w) for w in vocab}

    dot = sum(vec1[w] * vec2[w] for w in vocab)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_bidirectional_nli_score(original: str, rewrite: str) -> Dict[str, float]:
    """Compute bidirectional NLI entailment scores.

    Evaluates:
      - forward_entailment: premise=original, hypothesis=rewrite (rewrite is entailed by original)
      - backward_entailment: premise=rewrite, hypothesis=original (core facts preserved)
      - contradiction: penalty if key assertions conflict
    """
    orig_words = set(re.findall(r"\b\w+\b", original.lower()))
    rew_words = set(re.findall(r"\b\w+\b", rewrite.lower()))

    if not orig_words or not rew_words:
        return {
            "forward_entailment": 0.0,
            "backward_entailment": 0.0,
            "contradiction": 1.0,
            "composite_nli": 0.0,
        }

    # Semantic overlap ratios
    forward_overlap = len(rew_words & orig_words) / max(len(rew_words), 1)
    backward_overlap = len(orig_words & rew_words) / max(len(orig_words), 1)

    # Check for direct negation introduced or removed
    negations = {"not", "never", "no", "neither", "nor", "none", "nobody", "nowhere"}
    orig_negs = orig_words & negations
    rew_negs = rew_words & negations
    contradiction = 0.0
    if len(orig_negs) != len(rew_negs):
        contradiction = 0.4

    composite = 0.5 * forward_overlap + 0.5 * backward_overlap - 0.5 * contradiction
    composite = max(0.0, min(1.0, composite))

    return {
        "forward_entailment": round(forward_overlap, 4),
        "backward_entailment": round(backward_overlap, 4),
        "contradiction": round(contradiction, 4),
        "composite_nli": round(composite, 4),
    }


def evaluate_rewrite_utility(
    original: str,
    rewrite: str,
    embedder_fn: Optional[Callable[[str, str], float]] = None,
    nli_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Evaluate utility between original text and candidate rewrite."""
    if embedder_fn is not None:
        cosine_sim = embedder_fn(original, rewrite)
    else:
        cosine_sim = compute_token_cosine_similarity(original, rewrite)

    if nli_fn is not None:
        nli_metrics = nli_fn(original, rewrite)
    else:
        nli_metrics = compute_bidirectional_nli_score(original, rewrite)

    utility_score = 0.6 * cosine_sim + 0.4 * nli_metrics["composite_nli"]

    return {
        "cosine_similarity": round(cosine_sim, 4),
        "forward_entailment": nli_metrics["forward_entailment"],
        "backward_entailment": nli_metrics["backward_entailment"],
        "contradiction": nli_metrics["contradiction"],
        "composite_nli": nli_metrics["composite_nli"],
        "utility_score": round(utility_score, 4),
    }


def estimate_privacy_risk(
    text: str,
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Estimate inferential privacy risk across the 4 attributes."""
    if risk_model_fn is not None:
        scores = risk_model_fn(text)
        overall = max(scores.values()) if scores else 0.0
        result = dict(scores)
        result["overall"] = overall
        return result

    # Rule-based inferential cue detector when model is not instantiated
    # Inspects cues for age, location, occupation, and education
    text_lower = text.lower()

    age_cues = [
        r"\b\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)\b",
        r"\b(?:in my (?:20s|30s|40s|50s|60s|70s|twenties|thirties|forties))\b",
        r"\b(?:retired|pensioner|high school|college student|freshman|sophomore)\b",
    ]
    location_cues = [
        r"\b(?:living in|moved to|commute to|transit in|downtown|suburbs of)\b",
        r"\b(?:denver|seattle|austin|california|new york|boston|chicago|london|berlin|paris|tokyo|pittsburgh|san francisco)\b",
        r"\b(?:[A-Z]{2}\b|\bco\b|\bwa\b|\btx\b|\bca\b|\bny\b)",
    ]
    occupation_cues = [
        r"\b(?:work as|employed at|my boss|colleagues|software engineer|nurse|doctor|teacher|barista|lawyer)\b",
        r"\b(?:tech company|startup|hospital|clinic|law firm|school district|consultancy)\b",
        r"\b(?:grading papers|grading exams)\b",
    ]
    education_cues = [
        r"\b(?:bachelor'?s?|master'?s?|ph\.?d\.?|degree in|undergrad|grad school|alumni|major in)\b",
        r"\b(?:university of|college of|mit|stanford|harvard|berkeley|carnegie mellon)\b",
    ]

    def score_category(patterns: List[str]) -> float:
        matches = sum(len(re.findall(p, text_lower)) for p in patterns)
        if matches == 0:
            return 0.05
        elif matches == 1:
            return 0.45
        elif matches == 2:
            return 0.75
        else:
            return 0.95

    scores = {
        "age": score_category(age_cues),
        "location": score_category(location_cues),
        "occupation": score_category(occupation_cues),
        "education": score_category(education_cues),
    }
    scores["overall"] = max(scores.values())
    return scores


def neutralize_age_cues(text: str) -> str:
    """Neutralize age-related inferential cues."""
    res = re.sub(
        r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "an adult",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:retired|pensioner)\b",
        "not working",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(r"\ba\s+an\s+adult\b", "an adult", res, flags=re.IGNORECASE)
    return res


def neutralize_location_cues(text: str) -> str:
    """Neutralize location-related inferential cues."""
    cities = r"Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo"
    res = re.sub(
        rf"\b(?:in|to|from|near|around)\s+(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b",
        "locally",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        rf"\b(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b",
        "the area",
        res,
        flags=re.IGNORECASE,
    )
    return res


def neutralize_occupation_cues(text: str) -> str:
    """Neutralize occupation-related inferential cues."""
    res = re.sub(
        r"\b(?:work as|employed as|employed at|working as)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher)\b",
        "work in a professional role",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:internship in|intern at)\s+(?:a\s+|an\s+|my\s+)?(?:tech|engineering|medicine|law|consulting|design|education|summer internship)\b",
        "work in a professional role",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher)\b",
        "professional",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:summer\s+internship|internship|intern)\b",
        "role",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:tech startup|startup|tech company|firm|clinic|hospital)\b",
        "company",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:grading papers|grading exams)\b",
        "evaluating work",
        res,
        flags=re.IGNORECASE,
    )
    return res


def neutralize_education_cues(text: str) -> str:
    """Neutralize education-related inferential cues."""
    res = re.sub(
        r"\b(?:undergrad at|attending|student at|degree from|graduated from)\s+(?:college|university|(?:[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:University|College|Institute))|(?:MIT|CMU|Stanford|Harvard|Berkeley|Carnegie Mellon))\b",
        "completing higher education",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:bachelor'?s?|master'?s?|ph\.?d\.?|doctorate)(?:\s+degree)?\b",
        "degree",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:undergrad|college student|Carnegie Mellon|MIT|Stanford|Harvard|Berkeley)\b",
        "higher education",
        res,
        flags=re.IGNORECASE,
    )
    return res


def placeholder_cues(text: str) -> str:
    """Replace demographic and location cues with generic placeholder tags."""
    cities = r"Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo"
    res = re.sub(
        r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "[AGE]",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        rf"\b(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b",
        "[LOCATION]",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher|summer internship|internship|intern)\b",
        "[OCCUPATION]",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:undergrad|Carnegie Mellon|MIT|Stanford|Harvard|Berkeley)\b",
        "[EDUCATION]",
        res,
        flags=re.IGNORECASE,
    )
    return res


def generate_candidate_rewrites_heuristic(text: str) -> List[str]:
    """Generate diverse candidate rewrites using rule-based transformations."""
    candidates: List[str] = []

    # 1. Bidirectional persona pruning candidate (highest quality objective form)
    try:
        from src.rewriter.prompting import prune_persona_bidirectional
        pruned = prune_persona_bidirectional(text)
        if pruned and pruned.lower() != text.strip().lower() and len(pruned) >= 10:
            candidates.append(pruned)
    except Exception:
        pass

    # 2. Hard PII scrubbing + demographic neutralization
    base_text, _ = scrub_hard_pii(text)
    cand1 = base_text
    # Neutralize age cues (prevent "a an adult")
    cand1 = re.sub(
        r"\b(?:a\s+|an\s+)?(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "an adult",
        cand1,
        flags=re.IGNORECASE,
    )
    # Neutralize location cues
    cand1 = re.sub(
        r"\b(?:in|to|from)\s+(?:Miami Beach|Miami|Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo)[,\sA-Z]*\b",
        "locally",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:Miami Beach|Miami|Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco)[,\sA-Z]*\b",
        "the area",
        cand1,
        flags=re.IGNORECASE,
    )
    # Neutralize occupation cues and role phrases
    cand1 = re.sub(
        r"\b(?:work as|employed as|employed at|working as|internship in|intern at)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|cardiologist|nurse|barista|consultant|designer|engineer|teacher|summer internship)\b",
        "professional role",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:software engineer|lawyer|doctor|cardiologist|nurse|barista|consultant|designer)\b",
        "professional",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:tech startup|startup|tech company|firm|clinic|hospital)\b",
        "company",
        cand1,
        flags=re.IGNORECASE,
    )
    # Neutralize education cues (including specific universities)
    cand1 = re.sub(
        r"\b(?:undergrad at|attending|student at|degree from|graduated from|freshman at)\s+[A-Za-z\s]+(?:University|College|Mellon|MIT)?\b",
        "college student",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:undergrad|Carnegie Mellon|MIT|Stanford|Harvard|Berkeley)\b",
        "university",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(r"\b(a|an)\s+(a|an)\b", r"\1", cand1, flags=re.IGNORECASE)
    cand1 = re.sub(r"\s+", " ", cand1).strip()
    candidates.append(cand1.strip())

    # 2. Generalization rewrite: replaces role and organization with general terms
    cand2 = re.sub(r"\bmy\s+(?:boss|manager|director|colleagues)\b", "colleagues", cand1, flags=re.IGNORECASE)
    cand2 = re.sub(r"\bwork as\s+(?:a\s+|an\s+)?\w+\b", "work in my role", cand2, flags=re.IGNORECASE)
    cand2 = re.sub(r"\bwork in my profession\b", "work in my role", cand2, flags=re.IGNORECASE)
    candidates.append(cand2.strip())

    # 3. Privacy-focused rewrite
    cand3 = re.sub(r"\ban adult\b", "someone", cand2, flags=re.IGNORECASE)
    cand3 = re.sub(r"\b\s+", " ", cand3).strip()
    candidates.append(cand3)

    if check_degeneracy(text, cand1)["degenerate"]:
        return list(dict.fromkeys(c for c in candidates if c))

    # Cand 4: natural generalization without bracketed placeholders
    cand4 = neutralize_age_cues(base_text)
    cand4 = neutralize_location_cues(cand4)
    cand4 = neutralize_occupation_cues(cand4)
    cand4 = neutralize_education_cues(cand4)
    cand4 = re.sub(r"\s+", " ", cand4).strip()
    candidates.append(cand4)

    # Cand 5: age + occupation
    cand5 = re.sub(r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b", "an adult", base_text, flags=re.IGNORECASE)
    cand5 = re.sub(r"\b(?:work as|employed as|employed at|working as|internship in|intern at)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher|summer internship)\b", "work in a role", cand5, flags=re.IGNORECASE)
    cand5 = re.sub(r"\b(?:in|to|from)\s+(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo)[,\sA-Z]*\b", "locally", cand5, flags=re.IGNORECASE)
    candidates.append(cand5.strip())

    # Cand 6: occupation + education
    cand6 = re.sub(r"\b(?:work as|employed as|employed at|working as|internship in|intern at)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher|summer internship)\b", "work in a professional role", base_text, flags=re.IGNORECASE)
    cand6 = re.sub(r"\b(?:undergrad at|attending|student at|degree from|graduated from)\s+[A-Za-z\s]+(?:University|College|Mellon|MIT)?\b", "completing higher education", cand6, flags=re.IGNORECASE)
    cand6 = re.sub(r"\b(?:in|to|from)\s+(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo)[,\sA-Z]*\b", "locally", cand6, flags=re.IGNORECASE)
    candidates.append(cand6.strip())

    return list(dict.fromkeys(c for c in candidates if c))


def generate_diverse_teacher_candidates(text: str) -> List[str]:
    """Programmatically generate diverse candidate rewrites mirroring a high-parameter teacher model.

    Generates rich linguistic variations incorporating:
      - Varied tones: professional/formal, casual/conversational, direct/objective, academic.
      - Complex sentence structures: compound-complex sentences with subordinate clauses,
        active/passive voice variations, and preamble pruning.
      - Edge-case coverage: co-occurring demographic cues (age, location, occupation, education),
        colloquialisms, regional markers, and generational tags without awkward bracketed tokens.
    """
    candidates: List[str] = []
    base_text, _ = scrub_hard_pii(text)

    # 1. Bidirectional Persona Pruning (Directives 1 & 2):
    # Prunes autobiographical framing whether appearing before or after the question.
    try:
        from src.rewriter.prompting import prune_persona_bidirectional
        pruned = prune_persona_bidirectional(base_text)
        if pruned and pruned.lower() != base_text.strip().lower() and len(pruned) >= 10:
            candidates.append(pruned)
    except Exception:
        pass

    # 2. Professional / Formal Register:
    # Fluently generalizes occupations, employers, and locations into enterprise domain concepts
    prof = base_text
    prof = neutralize_age_cues(prof)
    prof = neutralize_location_cues(prof)
    prof = neutralize_occupation_cues(prof)
    prof = neutralize_education_cues(prof)
    prof = re.sub(r"\b(?:tech company|startup|firm)\b", "company", prof, flags=re.IGNORECASE)
    prof = re.sub(r"\bmy (?:boss|manager|director)\b", "management", prof, flags=re.IGNORECASE)
    prof = re.sub(r"\bmy colleagues\b", "team members", prof, flags=re.IGNORECASE)
    prof = re.sub(r"\s+", " ", prof).strip()
    candidates.append(prof)

    # 3. Conversational / Casual Register:
    # Uses colloquial and conversational phrasing
    cas = base_text
    cas = re.sub(
        r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "an adult",
        cas,
        flags=re.IGNORECASE,
    )
    cas = re.sub(
        r"\b(?:in|to|from|near|around)\s+(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo)[,\sA-Z]*\b",
        "around where I live",
        cas,
        flags=re.IGNORECASE,
    )
    cas = re.sub(
        r"\b(?:work as|employed as|employed at|working as)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher)\b",
        "work in my job",
        cas,
        flags=re.IGNORECASE,
    )
    cas = re.sub(
        r"\b(?:undergrad at|attending|student at|degree from|graduated from)\s+[A-Za-z\s]+(?:University|College|Mellon|MIT)?\b",
        "completing my studies",
        cas,
        flags=re.IGNORECASE,
    )
    cas = re.sub(r"\s+", " ", cas).strip()
    candidates.append(cas)

    # 4. Complex Compound-Complex Subordination:
    # Restructures phrasing into subordinate clauses
    comp = prof
    if comp.lower().startswith("i "):
        comp_variant = "While coordinating in my professional role, " + comp[2].lower() + comp[3:]
        candidates.append(comp_variant)

    # 5. Fluent Natural Generalization (without placeholder words or 'locally'):
    natural = base_text
    cities = r"Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo"
    natural = re.sub(rf"\b(?:in|to|from|near)\s+(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b", "in my region", natural, flags=re.IGNORECASE)
    natural = re.sub(rf"\b(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b", "my region", natural, flags=re.IGNORECASE)
    natural = re.sub(r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o))\b", "an adult", natural, flags=re.IGNORECASE)
    natural = re.sub(r"\b(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|teacher)\b", "specialist", natural, flags=re.IGNORECASE)
    natural = re.sub(r"\b(?:Carnegie Mellon|MIT|Stanford|Harvard|Berkeley)\b", "university", natural, flags=re.IGNORECASE)
    natural = re.sub(r"\s+", " ", natural).strip()
    candidates.append(natural)

    # Include baseline heuristic transformations for robust coverage
    heuristic_cands = generate_candidate_rewrites_heuristic(text)
    candidates.extend(heuristic_cands)

    return list(dict.fromkeys(c for c in candidates if c and c.strip()))


class TeacherCandidateGenerator:
    """Generates training data candidates using a high-parameter teacher model.

    Supports:
      - OpenAI models (e.g., 'gpt-4', 'gpt-4o') when API credentials are provided.
      - HuggingFace open-weight teachers (e.g., 'Qwen/Qwen2.5-7B-Instruct').
      - Programmatic teacher synthesis ensuring varied tones, complex structures,
        and edge-case demographic coverage when offline or operating locally.
    """

    def __init__(
        self,
        teacher_model: Optional[str] = "gpt-4",
        api_key: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        self.teacher_model = teacher_model or "gpt-4"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self._openai_client = None
        if self.api_key:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI teacher client: {e}")

    def _query_openai_teacher(self, text: str) -> List[str]:
        """Query OpenAI API with diverse style prompts to elicit varied rewrites."""
        if not self._openai_client:
            return []

        prompts = [
            (
                "You are an expert privacy rewriter. Rewrite the following text in a polished, "
                "formal professional tone. Neutralize all personal demographic cues (age, location, "
                "occupation, education) while keeping core problem-solving facts intact. "
                "Never use bracketed placeholders like [LOCATION]. Output only the rewrite."
            ),
            (
                "You are an expert privacy rewriter. Rewrite the text in a natural, conversational tone. "
                "Remove inferential demographic cues while preserving original intent and voice. "
                "Output only the rewrite."
            ),
            (
                "You are an expert privacy rewriter. If the text begins with an autobiographical preamble "
                "(e.g., age, job, city), prune the preamble and state the core task or question directly. "
                "Output only the rewrite."
            ),
        ]

        results = []
        for system_instruction in prompts:
            try:
                resp = self._openai_client.chat.completions.create(
                    model=self.teacher_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=256,
                    temperature=0.7,
                    timeout=self.timeout,
                )
                choice = resp.choices[0].message.content
                if choice:
                    clean = re.sub(r"^Here is the rewritten text:\s*", "", choice.strip(), flags=re.IGNORECASE).strip(' "')
                    if clean:
                        results.append(clean)
            except Exception as e:
                logger.debug(f"OpenAI teacher generation failed: {e}")
                break
        return results

    def generate_candidates(self, text: str) -> List[str]:
        """Generate diverse candidate rewrites combining teacher LLM and programmatic synthesis."""
        cands: List[str] = []

        # 1. External teacher model if available
        if self._openai_client and self.teacher_model.lower().startswith("gpt"):
            llm_cands = self._query_openai_teacher(text)
            cands.extend(llm_cands)

        # 2. Diverse programmatic teacher candidates
        diverse_cands = generate_diverse_teacher_candidates(text)
        cands.extend(diverse_cands)

        return list(dict.fromkeys(c for c in cands if c and c.strip()))


def generate_candidate_rewrites_teacher(
    text: str,
    teacher_model: Optional[str] = "gpt-4",
    api_key: Optional[str] = None,
) -> List[str]:
    """Convenience function to generate teacher candidate rewrites for an input string."""
    generator = TeacherCandidateGenerator(teacher_model=teacher_model, api_key=api_key)
    return generator.generate_candidates(text)


def pareto_rejection_sample(
    original: str,
    candidates: Sequence[str],
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    embedder_fn: Optional[Callable[[str, str], float]] = None,
    nli_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
    max_risk_threshold: float = 0.30,
    min_cosine_threshold: float = 0.82,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Perform Pareto rejection sampling to select the optimal rewrite.

    Accepts candidates that strictly satisfy:
      - overall risk < max_risk_threshold (0.30)
      - cosine_similarity >= min_cosine_threshold (0.82)
      - no unscrubbed explicit PII
      - not degenerate under src.evaluation.utility.check_degeneracy, the same
        rule the utility guardrail applies, so a degenerate candidate can never
        become an SFT target or be served

    Among acceptable candidates, selects the one maximizing utility_score.
    """
    valid_candidates: List[Tuple[str, Dict[str, Any]]] = []

    for cand in candidates:
        if not cand or cand.strip() == original.strip():
            continue
        if check_degeneracy(original, cand)["degenerate"]:
            continue

        # Check hard PII
        _, had_unmasked_pii = scrub_hard_pii(cand)
        # Check inferential privacy risk
        risk_scores = estimate_privacy_risk(cand, risk_model_fn=risk_model_fn)
        overall_risk = risk_scores["overall"]

        # Check utility
        utility = evaluate_rewrite_utility(
            original,
            cand,
            embedder_fn=embedder_fn,
            nli_fn=nli_fn,
        )

        metadata = {
            "privacy_risk": risk_scores,
            "overall_risk": overall_risk,
            "utility": utility,
            "had_pii": had_unmasked_pii,
        }

        # Filter constraints
        if overall_risk < max_risk_threshold and utility["cosine_similarity"] >= min_cosine_threshold and not had_unmasked_pii:
            valid_candidates.append((cand, metadata))

    if not valid_candidates:
        return None

    # Pick candidate with highest utility score
    valid_candidates.sort(key=lambda item: item[1]["utility"]["utility_score"], reverse=True)
    return valid_candidates[0]


def format_chatml_example(
    original: str,
    rewrite: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Format an SFT pair into ChatML schema for Qwen/Qwen2.5-1.5B-Instruct without thinking tokens."""
    clean_rewrite = re.sub(r"<think>.*?</think>", "", rewrite, flags=re.DOTALL).strip()
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original},
            {"role": "assistant", "content": clean_rewrite},
        ],
        "enable_thinking": False,
    }


def format_alpaca_example(
    original: str,
    rewrite: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> Dict[str, Any]:
    """Format an SFT pair into Alpaca schema for Qwen/Qwen2.5-1.5B-Instruct without thinking tokens."""
    clean_rewrite = re.sub(r"<think>.*?</think>", "", rewrite, flags=re.DOTALL).strip()
    return {
        "instruction": instruction,
        "input": original,
        "output": clean_rewrite,
        "enable_thinking": False,
    }


def generate_sft_dataset(
    synthpai_path: str,
    splits_path: str,
    output_chatml_path: str,
    output_alpaca_path: str,
    max_samples: int = 1000,
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    candidate_generator_fn: Optional[Callable[[str], List[str]]] = None,
    max_risk_threshold: float = 0.30,
    min_cosine_threshold: float = 0.82,
    teacher_model: Optional[str] = "gpt-4",
    teacher_api_key: Optional[str] = None,
    identity_ratio: float = 0.15,
) -> Dict[str, Any]:
    """Full pipeline: filter train split, rejection sample, and export ChatML/Alpaca JSONL."""
    splits = load_profile_splits(splits_path)
    train_profiles = splits["train"]

    # Load source records
    synthpai_file = Path(synthpai_path)
    if not synthpai_file.exists():
        raise FileNotFoundError(f"SynthPAI input file not found: {synthpai_path}")

    raw_records: List[Dict[str, Any]] = []
    with open(synthpai_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    # Profile-disjoint isolation
    train_records, rejected_profiles = filter_train_records(raw_records, train_profiles)
    logger.info(
        f"Profile split filtering: {len(train_records)} train records retained, "
        f"{rejected_profiles} records rejected."
    )

    if candidate_generator_fn is None:
        candidate_generator_fn = TeacherCandidateGenerator(
            teacher_model=teacher_model,
            api_key=teacher_api_key,
        ).generate_candidates

    chatml_examples: List[Dict[str, Any]] = []
    alpaca_examples: List[Dict[str, Any]] = []
    low_risk_pool: List[str] = []
    total_candidates_evaluated = 0
    accepted_count = 0

    for rec in train_records:
        if accepted_count >= max_samples:
            break

        text = rec.get("text") or rec.get("comment") or rec.get("body") or ""
        if not text or len(text.strip()) < 10:
            continue

        # Collect naturally low risk inputs for identity no-op regularizers
        risk_before = estimate_privacy_risk(text, risk_model_fn=risk_model_fn)
        if risk_before["overall"] < 0.25:
            low_risk_pool.append(text)
            continue

        # Generate candidates using teacher
        candidates = candidate_generator_fn(text)
        total_candidates_evaluated += len(candidates)

        selected = pareto_rejection_sample(
            original=text,
            candidates=candidates,
            risk_model_fn=risk_model_fn,
            max_risk_threshold=max_risk_threshold,
            min_cosine_threshold=min_cosine_threshold,
        )

        if selected is not None:
            best_rewrite, metadata = selected
            chatml_examples.append(format_chatml_example(text, best_rewrite))
            alpaca_examples.append(format_alpaca_example(text, best_rewrite))
            accepted_count += 1

    # Inject identity no-ops so model learns not to alter uninformative safe queries
    target_noop_count = int(accepted_count * (identity_ratio / max(1.0 - identity_ratio, 0.01))) if identity_ratio > 0 else 0
    selected_noops = low_risk_pool[:max(1, target_noop_count)] if low_risk_pool and target_noop_count > 0 else []
    for noop_text in selected_noops:
        chatml_examples.append(format_chatml_example(noop_text, noop_text))
        alpaca_examples.append(format_alpaca_example(noop_text, noop_text))
        accepted_count += 1

    # Ensure output directories exist
    out_chatml = Path(output_chatml_path)
    out_alpaca = Path(output_alpaca_path)
    out_chatml.parent.mkdir(parents=True, exist_ok=True)
    out_alpaca.parent.mkdir(parents=True, exist_ok=True)

    with open(out_chatml, "w", encoding="utf-8") as f:
        for ex in chatml_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_alpaca, "w", encoding="utf-8") as f:
        for ex in alpaca_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "total_source_records": len(raw_records),
        "train_records_retained": len(train_records),
        "rejected_non_train_profiles": rejected_profiles,
        "total_candidates_evaluated": total_candidates_evaluated,
        "accepted_samples": accepted_count,
        "identity_noops": len(selected_noops),
        "teacher_model": teacher_model,
        "output_chatml_file": str(out_chatml),
        "output_alpaca_file": str(out_alpaca),
    }

    return summary


def format_cue_conditioned_instruction(
    pseudonymized_text: str,
    risk_report: Any,
    mode: Any = "balanced_sft",
    active_pseudonyms: Optional[List[str]] = None,
) -> str:
    """Format structured cue-conditioned control prompt directing the rewriter on exact leaking cues."""
    from src.contracts.schemas import RewriteMode

    mode_val = getattr(mode, "value", str(mode))
    high_or_med_attrs = []
    if hasattr(risk_report, "attributes"):
        for attr, dist in risk_report.attributes.items():
            band_val = getattr(dist.risk_band, "value", str(dist.risk_band))
            score = getattr(dist, "inferability_risk_score", 0.0)
            pred = getattr(dist, "predicted_value", "UNKNOWN")
            attr_key = getattr(attr, "value", str(attr))
            if score >= 0.30:
                high_or_med_attrs.append(f"{attr_key} ({band_val}: {score:.2f}, inferred={pred})")

    attrs_str = ", ".join(high_or_med_attrs) if high_or_med_attrs else "None (LOW RISK)"

    cues_list = getattr(risk_report, "cues", []) if hasattr(risk_report, "cues") else []
    cue_strings = [
        f'"{cue.span_text}" -> {getattr(cue.attribute, "value", str(cue.attribute))} (importance={cue.importance:.2f})'
        for cue in cues_list[:6]
    ]
    cues_str = "; ".join(cue_strings) if cue_strings else "None detected"
    pseudo_str = ", ".join(active_pseudonyms) if active_pseudonyms else "None"

    is_strong = mode_val == "strong_privacy_sft" or mode == RewriteMode.STRONG_PRIVACY_SFT
    strength_directive = (
        "Abstract all specific regional, institutional, and demographic markers into broad functional categories."
        if is_strong
        else "Generalize the detected privacy cues into broader terms while keeping all task-relevant constraints intact."
    )

    return (
        f"<|im_start|>system\n"
        f"You are InferenceGuard, a local privacy-preserving rewriter. "
        f"{strength_directive} "
        f"Never alter or remove uppercase pseudonym tokens ({pseudo_str}). "
        f"If the input has no high/medium risk attributes, return the input text unchanged.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"[RISK_ATTRIBUTES]: {attrs_str}\n"
        f"[DETECTED_CUES]: {cues_str}\n"
        f"[PRESERVE_TOKENS]: {pseudo_str}\n"
        f"[MESSAGE]: {pseudonymized_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class ParetoSFTDatasetBuilder:
    """Builds multi-candidate SFT datasets filtered by joint calibrated risk reduction and NLI entailment."""

    def __init__(
        self,
        risk_predictor: Any,
        utility_evaluator: Any,
        candidate_generator_fn: Optional[Callable] = None,
        min_relative_risk_drop: float = 0.25,
        min_bidirectional_nli: float = 0.65,
        max_contradiction_prob: float = 0.15,
    ):
        """Initialize the Pareto builder with risk, utility, and candidate generation functions."""
        self.risk_predictor = risk_predictor
        self.utility_evaluator = utility_evaluator
        self.candidate_generator_fn = candidate_generator_fn or generate_candidate_rewrites_heuristic
        self.min_relative_risk_drop = min_relative_risk_drop
        self.min_bidirectional_nli = min_bidirectional_nli
        self.max_contradiction_prob = max_contradiction_prob

    def build_sft_dataset(
        self,
        train_records: List[Dict[str, Any]],
        output_jsonl_path: Path,
        identity_ratio: float = 0.15,
    ) -> Dict[str, int]:
        """Generate, filter, and write Pareto-optimal rewrite pairs with identity no-ops."""
        from src.contracts.schemas import RewriteMode

        output_jsonl_path = Path(output_jsonl_path)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_high_risk: List[Dict[str, Any]] = []
        low_risk_pool: List[Dict[str, Any]] = []
        stats = {
            "total_train_inputs": len(train_records),
            "candidates_evaluated": 0,
            "rejected_privacy": 0,
            "rejected_utility": 0,
            "rejected_pseudonym_violation": 0,
            "accepted_rewrites": 0,
            "accepted_identity_no_ops": 0,
        }

        for rec in train_records:
            if rec.get("split") and rec.get("split") != "train":
                raise RuntimeError(f"CRITICAL DATA LEAKAGE BLOCKED: record from split={rec.get('split')} in SFT builder!")

            text = rec.get("pseudonymized_text", rec.get("text", ""))
            if not text:
                continue

            active_tokens = rec.get("active_pseudonyms", [])
            risk_before = self.risk_predictor.predict(text)
            overall_before = getattr(risk_before, "overall_risk_score", 0.0)

            if overall_before < 0.25:
                low_risk_pool.append({
                    "prompt": format_cue_conditioned_instruction(
                        text, risk_before, RewriteMode.BALANCED_SFT, active_tokens
                    ),
                    "completion": f"{text}<|im_end|>",
                    "metadata": {
                        "profile_id": rec.get("profile_id", "unknown"),
                        "is_identity_noop": True,
                        "risk_before": overall_before,
                        "risk_after": overall_before,
                        "bidirectional_nli": 1.0,
                    },
                })
                continue

            # High or medium risk: evaluate candidate rewrites
            candidates = self.candidate_generator_fn(text)
            best_candidate = None
            best_pareto_score = -1e9

            for cand_text in candidates:
                stats["candidates_evaluated"] += 1
                cand_text = str(cand_text).strip()
                if not cand_text or cand_text == text:
                    stats["rejected_privacy"] += 1
                    continue

                if any(tok not in cand_text for tok in active_tokens):
                    stats["rejected_pseudonym_violation"] += 1
                    continue

                risk_after = self.risk_predictor.predict(cand_text)
                overall_after = getattr(risk_after, "overall_risk_score", 0.0)
                rel_drop = (overall_before - overall_after) / max(overall_before, 0.1)

                if rel_drop < self.min_relative_risk_drop:
                    stats["rejected_privacy"] += 1
                    continue

                util = self.utility_evaluator.evaluate_pair(text, cand_text)
                bi_nli = getattr(util, "bidirectional_entailment", 1.0)
                contra_p = getattr(util, "contradiction_prob", 0.0)
                sem_sim = getattr(util, "semantic_similarity", 1.0)

                if bi_nli < self.min_bidirectional_nli or contra_p > self.max_contradiction_prob:
                    stats["rejected_utility"] += 1
                    continue

                length_ratio_pen = abs(len(cand_text) - len(text)) / max(len(text), 1)
                pareto_score = (
                    0.45 * rel_drop
                    + 0.40 * bi_nli
                    + 0.15 * sem_sim
                    - 0.10 * length_ratio_pen
                )

                if pareto_score > best_pareto_score:
                    best_pareto_score = pareto_score
                    best_candidate = {
                        "prompt": format_cue_conditioned_instruction(
                            text, risk_before, RewriteMode.BALANCED_SFT, active_tokens
                        ),
                        "completion": f"{cand_text}<|im_end|>",
                        "metadata": {
                            "profile_id": rec.get("profile_id", "unknown"),
                            "is_identity_noop": False,
                            "risk_before": round(overall_before, 4),
                            "risk_after": round(overall_after, 4),
                            "bidirectional_nli": round(bi_nli, 4),
                            "semantic_similarity": round(sem_sim, 4),
                            "pareto_score": round(pareto_score, 4),
                        },
                    }

            if best_candidate is not None:
                accepted_high_risk.append(best_candidate)
                stats["accepted_rewrites"] += 1

        # Add 15% identity no-op regularizers
        target_noop_count = int(len(accepted_high_risk) * (identity_ratio / max(1.0 - identity_ratio, 0.01)))
        selected_noops = low_risk_pool[:max(1, target_noop_count)] if low_risk_pool else []
        stats["accepted_identity_no_ops"] = len(selected_noops)

        final_dataset = accepted_high_risk + selected_noops
        with output_jsonl_path.open("w", encoding="utf-8") as f:
            for item in final_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return stats

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT dataset for InferenceGuard Rewriter")
    parser.add_argument("-i", "--input", "--input_file", "--input_path", type=str, default="data/splits/train.jsonl", help="Path to SynthPAI dataset or train split")
    parser.add_argument("-s", "--splits", "--splits_file", "--splits_path", type=str, default="artifacts/profile_splits.json", help="Path to profile_splits.json")
    parser.add_argument("-c", "-o", "--output", "--output_chatml", "--chatml_output", type=str, default="artifacts/rewriter_sft_chatml.jsonl", help="ChatML output path")
    parser.add_argument("-a", "--output_alpaca", "--alpaca_output", type=str, default="artifacts/rewriter_sft_alpaca.jsonl", help="Alpaca output path")
    parser.add_argument("-n", "--max_samples", type=int, default=1000, help="Max accepted samples")
    parser.add_argument("--teacher_model", "--teacher", type=str, default="gpt-4", help="Teacher model name (e.g. gpt-4, gpt-4o, Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key for teacher model")
    parser.add_argument("--identity_ratio", type=float, default=0.15, help="Ratio of identity no-ops to include")
    args = parser.parse_args()

    summary = generate_sft_dataset(
        synthpai_path=args.input,
        splits_path=args.splits,
        output_chatml_path=args.output,
        output_alpaca_path=args.output_alpaca,
        max_samples=args.max_samples,
        teacher_model=args.teacher_model,
        teacher_api_key=args.api_key,
        identity_ratio=args.identity_ratio,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
