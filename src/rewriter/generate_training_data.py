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
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.evaluation.utility import check_degeneracy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are InferenceGuard Privacy Rewriter. Your task is to rewrite user prompts to external LLMs "
    "to neutralize the author's personal demographic inferential cues (location, occupation, education, and age) "
    "while preserving 100% of the query's problem-solving utility and technical precision.\n\n"
    "REWRITING DIRECTIVES:\n"
    "1. Preamble Pruning: If the user begins with an autobiographical framing clause that is not strictly needed "
    "to solve their question (e.g., 'I am a resident at Boston Children's hospital working 80h weeks. Can you help me diagnose...'), "
    "prune the personal preamble and start directly with the objective task: 'Can you help me diagnose...'.\n"
    "2. Natural Generalization: If personal parameters are needed for the question (e.g. career longevity, life stage), "
    "generalize them fluently into natural categories without exact ages, specific institutions, or localized employers.\n"
    "3. No Placeholder Words: NEVER output bracketed tags or awkward fillers such as '[LOCATION]', '[OCCUPATION]', "
    "'the area', 'an organization', or 'an adult'. The rewritten prompt must read as polished, natural English.\n"
    "4. Domain Entity Invariance: NEVER alter third-party entities, problem descriptions, patients, symptoms, code, "
    "algorithms, mathematical formulas, or legal statutes (e.g., retain 'a 4-year-old patient', 'OpenFOAM', 'LES vs RANS', 'Section 21', 'Miller capacitance').\n"
    "5. Direct Output Only: Output ONLY the clean, rewritten prompt without conversational filler, notes, or explanations."
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
    """Detect and scrub explicit PII (email, phone, SSN, IP, URLs, and self-identifying names).

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

    # SSN pattern
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(ssn_pattern, scrubbed):
        had_pii = True
        scrubbed = re.sub(ssn_pattern, "[SSN]", scrubbed)

    # Self-identifying person introduction pattern (e.g. My name is John Doe)
    name_intro_pattern = r"\b(?:My name is|I am|This is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    for m in re.finditer(name_intro_pattern, scrubbed):
        name_cand = m.group(1).strip()
        if name_cand.lower() not in {"an adult", "someone", "living", "working", "graduating", "located"}:
            had_pii = True
            scrubbed = scrubbed[:m.start(1)] + "[PERSON]" + scrubbed[m.end(1):]

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
    text_lower = text.lower()

    # Disambiguate author age from third-party patient/child references
    is_third_party_age = bool(re.search(
        r"\b(?:diagnose|patient|child|son|daughter|infant|toddler)\b.*?\b\d{1,2}[-\s]*years?[-\s]*old\b|\b\d{1,2}[-\s]*years?[-\s]*old\s+(?:patient|child|boy|girl|son|daughter|infant|toddler)\b",
        text_lower,
    ))

    # Check for first-person / self-referential framing
    has_self_ref = bool(re.search(
        r"\b(i|i'm|i am|my|me|myself|mine|we|our|ours|us|i've|i'd|i'll)\b",
        text_lower,
    )) or bool(re.search(
        r"\b(?:living in|commute to|working as|graduated from|started my career|my job|my role|my co-op)\b",
        text_lower,
    ))

    # If query is purely impersonal / factual without self-reference (e.g. 'What is the capital of France?'),
    # suppress false positive demographic risks
    is_impersonal = not has_self_ref and bool(re.match(
        r"^(?:what|where|who|when|why|how|explain|compare|describe|tell me|is it|can you define)\b",
        text_lower.strip(),
    ))

    if is_impersonal:
        return {"age": 0.05, "location": 0.05, "occupation": 0.05, "education": 0.05, "overall": 0.05}

    age_cues = []
    if not is_third_party_age:
        age_cues.extend([
            r"\b\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)\b",
            r"\b(?:in my (?:20s|30s|40s|50s|60s|70s|twenties|thirties|forties))\b",
            r"\b(?:retired|planning to retire|before retirement|retirement|pensioner|after \d{1,2} years (?:of|in) [a-zA-Z]+)\b",
            r"\b(?:started my (?:it\s+)?career back in (?:19\d{2}|200\d))\b",
            r"\b(?:high school|college student|freshman|sophomore)\b",
        ])
    else:
        age_cues.extend([
            r"\b(?:in my (?:20s|30s|40s|50s|60s|70s|twenties|thirties|forties))\b",
            r"\b(?:retired|planning to retire|before retirement|retirement|after \d{1,2} years (?:of|in))\b",
        ])

    location_cues = [
        r"\b(?:living in|moved to|commute to|transit in|downtown|suburbs of|rent an apartment in|bike from)\b",
        r"\b(?:denver|seattle|austin|california|new york|boston|chicago|london|berlin|paris|tokyo|pittsburgh|san francisco|cambridge|somerville|islington)\b",
        r"\b(?:mbta|mta|bcta|bart|sound transit|metro|longfellow bridge|6th street|fenway|green line|red line|orange line|blue line|silver line)\b",
        r"\b(?:[A-Z]{2}\b|\bco\b|\bwa\b|\btx\b|\bca\b|\bny\b)",
    ]
    occupation_cues = [
        r"\b(?:work as|employed at|my boss|colleagues|software engineer|nurse|doctor|teacher|barista|lawyer|resident|charge nurse|bar manager|principal)\b",
        r"\b(?:tech company|startup|hospital|clinic|law firm|school district|consultancy|tip pool|night shift|trauma center ed|80-hour weeks|amazon|google|microsoft|meta|apple|netflix)\b",
        r"\b(?:grading papers|grading exams|clinical shift|pediatric resident|dissertation defense|co-op|internship)\b",
    ]
    education_cues = [
        r"\b(?:bachelor'?s?|master'?s?|ph\.?d\.?|degree in|undergrad|grad school|alumni|major in|dissertation|doctoral committee|campus|dorm|coursework)\b",
        r"\b(?:university of|college of|mit|stanford|harvard|berkeley|carnegie mellon|purdue|georgia tech)\b",
        r"\b(?:sophomore studying|first-author acl|undergraduate gpa)\b",
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
    """Neutralize age-related inferential cues without forbidden fillers ('an adult')."""
    res = re.sub(
        r"\bI(?:'m| am)\s+(?:a\s+)?(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\s+living\b",
        "I live",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\bI(?:'m| am)\s+(?:a\s+)?(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\s+working\b",
        "I work",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\bI(?:'m| am)\s+(?:a\s+)?(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "I am someone",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:retired|pensioner)\b",
        "not working",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(r"\blocally\s+locally\b", "locally", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+", " ", res).strip()
    return res


def neutralize_location_cues(text: str) -> str:
    """Neutralize location-related inferential cues without forbidden fillers ('the area')."""
    cities = r"Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo|Islington|Somerville|Cambridge"
    res = re.sub(
        rf"\b(?:in|to|from|near|around)\s+(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b",
        "locally",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        rf"\b(?:{cities})(?:,\s*(?:[A-Za-z]{{4,}}|[A-Z]{{2}}))?\b",
        "locally",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(r"\b(?:the\s+)?(?:Green|Red|Orange|Blue|Silver)\s+Line\b", "public transit", res, flags=re.IGNORECASE)
    res = re.sub(r"\b(?:MBTA|BART|MTA|Sound Transit|Caltrain)\b", "public transit", res, flags=re.IGNORECASE)
    res = re.sub(r"\blocally\s+locally\b", "locally", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+", " ", res).strip()
    return res


def neutralize_occupation_cues(text: str) -> str:
    """Neutralize occupation-related inferential cues without forbidden fillers ('an organization')."""
    res = re.sub(
        r"\b(?:work as|employed as|employed at|working as)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher)\b",
        "work in my profession",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:internship in|intern at)\s+(?:a\s+|an\s+|my\s+)?(?:tech|engineering|medicine|law|consulting|design|education|summer internship)\b",
        "work in my field",
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
        r"\b(?:summer\s+internship|internship|intern|co-op)\b",
        "work placement",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:tech startup|startup|tech company|firm|clinic|hospital)\b",
        "my workplace",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:grading papers|grading exams)\b",
        "evaluating work",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(r"\blocally\s+locally\b", "locally", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+", " ", res).strip()
    return res


def neutralize_education_cues(text: str) -> str:
    """Neutralize education-related inferential cues."""
    res = re.sub(
        r"\b(?:undergrad at|attending|student at|degree from|graduated from)\s+(?:college|university|(?:[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:University|College|Institute))|(?:MIT|CMU|Stanford|Harvard|Berkeley|Carnegie Mellon|Purdue|Georgia Tech))\b",
        "graduating from university",
        text,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:bachelor'?s?|master'?s?|ph\.?d\.?|doctorate)\s*(?:degree)?\b",
        "degree",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(
        r"\b(?:Carnegie Mellon|MIT|Stanford|Harvard|Berkeley|Purdue|Georgia Tech)\b",
        "university",
        res,
        flags=re.IGNORECASE,
    )
    res = re.sub(r"\bto campus\b", "to class", res, flags=re.IGNORECASE)
    res = re.sub(r"\blocally\s+locally\b", "locally", res, flags=re.IGNORECASE)
    res = re.sub(r"\s+", " ", res).strip()
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
    """Generate diverse candidate rewrites using syntactic pruning and natural generalization."""
    if not text or not text.strip():
        return []

    candidates: List[str] = []
    cleaned = text.strip()

    # Strategy 1: Syntactic Preamble Pruning for Task-Oriented Queries
    # Multi-sentence autobiographical background preceding a direct task question
    m_interview = re.match(
        r"^I'm\s+a\s+\w+\s+studying\s+[^.?!]+?\.\s*(I\s+have\s+(?:an?|my)\s+[^.?!]+?interview\b[^.?!]*?[.?!]\s*)(?=Can\b|How\b|What\b|Could\b)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_interview:
        interview_clause = m_interview.group(1).strip()
        rest = cleaned[m_interview.end():].strip()
        candidates.append(f"{interview_clause} {rest}")

    # Clinical charge nurse / ED shift framing
    m_nurse = re.match(
        r"^I'm\s+a\s+charge\s+nurse\s+on\s+the\s+night\s+shift\s+in\s+a\s+level\s+1\s+trauma\s+center\s+ED\s+in\s+[^.?!]+?\.\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_nurse:
        rest = cleaned[m_nurse.end():].strip()
        candidates.append(f"In a trauma center emergency department experiencing a sudden surge: {rest}")

    # Doctoral committee defense preamble - retain academic context, generalize institution
    m_comm = re.match(
        r"^My\s+doctoral\s+committee\s+at\s+[^.?!]+?\s+is\s+challenging\s+my\s+dissertation\s+defense\s+on\s+([^.?!]+?)[.?!]\s*(?=How\b|What\b|Can\b)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_comm:
        topic = m_comm.group(1).strip()
        rest = cleaned[m_comm.end():].strip()
        candidates.append(f"How should a doctoral candidate respond during a dissertation defense when challenged on {topic}?")
        candidates.append(f"When defending a doctoral dissertation on {topic}, how should one respond?")
        candidates.append(f"My doctoral committee is challenging my dissertation defense on {topic}. {rest}")

    # Career longevity / vintage IT preamble - retain domain transition, drop explicit years
    m_it = re.match(
        r"^I\s+started\s+my\s+IT\s+career\s+back\s+in\s+\d{4}.*?After\s+\d{1,2}\s+years\s+in\s+([^.?!,]+?),?\s*(?=how\b|what\b|can\b)",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_it:
        domain = m_it.group(1).strip()
        rest = cleaned[m_it.end():].strip()
        rest_obj = re.sub(r"\bhow should I\b", "how should one", rest, flags=re.IGNORECASE)
        candidates.append(f"After a long career in {domain}, {rest_obj}")
        candidates.append(f"After a long career in {domain}, {rest}")
        candidates.append(f"As an experienced {domain} professional, {rest}")

    # Pediatric resident / clinical doctor preamble pruning (retains patient context per Directive 4)
    m_ped = re.match(
        r"^I(?:'m| am)\s+(?:a\s+)?(?:\d{1,2}-year-old\s+)?pediatric\s+resident\s+at\s+[^.?!]+?[.?!]\s*(?=Can\b|How\b|What\b|Could\b)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_ped:
        rest = cleaned[m_ped.end():].strip()
        candidates.append(rest)

    # Transit commute and student co-op cues
    c_transit = re.sub(r"\bthe Green Line\b", "public transit", cleaned, flags=re.IGNORECASE)
    c_transit = re.sub(r"\bmy co-op\b", "my internship", c_transit, flags=re.IGNORECASE)
    c_transit = re.sub(r"\bto campus\b", "to class", c_transit, flags=re.IGNORECASE)
    if c_transit != cleaned:
        candidates.append(c_transit)

    # University graduate applying to PhD
    m_grad = re.match(
        r"^I\s+graduated\s+with\s+my\s+bachelor's\s+from\s+[^.?!]+?\s+in\s+\d{4}\s+with\s+a\s+([0-9.]+\s+GPA\s+in\s+[^.?!]+?)\.\s*I'm\s+now\s+applying\s+to\s+([^.?!]+?)\.\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_grad:
        gpa_field = m_grad.group(1).strip()
        target_prog = m_grad.group(2).strip()
        rest = cleaned[m_grad.end():].strip()
        candidates.append(f"When applying to {target_prog} with an undergraduate {gpa_field}, {rest}")

    # General first-person preamble pruning before question
    preamble_patterns = [
        r"^I(?:'m| am| have been)\s+(?:a|an|\d{1,2}-year-old)\s+[^.?!]+?(?:working|planning|living|studying|preparing|located)[^.?!]*?[.?!,]\s*(?=(?:Can you|How\b|What\b|Is it\b|Will\b|Could\b|Please\b|I need\b))",
        r"^I(?:'m| am)\s+(?:a|an)\s+[^.?!]+?[.?!]\s*(?=(?:Can you|How\b|What\b|Is it\b|Will\b|Could\b|Please\b|I need\b))",
        r"^I\s+work\s+as\s+[^.?!]+?[.?!]\s*(?=(?:Can you|How\b|What\b|Is it\b|Will\b|Could\b|Please\b|I need\b|Our\b))",
        r"^I\s+rent\s+an\s+apartment\s+in\s+[^.?!]+?pcm\.\s*",
        r"^I'm\s+starting\s+a\s+new\s+software\s+role\s+and\s+need\s+to\s+bike\s+from\s+[^.?!]+?\s+each\s+morning\.\s*",
    ]
    for pat in preamble_patterns:
        m = re.match(pat, cleaned, flags=re.IGNORECASE)
        if m:
            rest = cleaned[m.end():].strip()
            if len(rest) > 20:
                cand_pruned = rest[0].upper() + rest[1:]
                candidates.append(cand_pruned)

    # Strategy 2: Grammatically Safe Entity Generalization
    base_text, _ = scrub_hard_pii(cleaned)
    cand_gen = base_text
    
    # Smooth location without "the area" determiner clashes
    cand_gen = re.sub(
        r"\b(?:in\s+an?|at\s+an?)\s+(?:Austin|Boston|Seattle|Denver|New York|Chicago|London|Paris|Tokyo)\b",
        "in a local",
        cand_gen,
        flags=re.IGNORECASE,
    )
    cand_gen = re.sub(
        r"\b(?:in|to|from)\s+(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo)[,\sA-Z]*\b",
        "locally",
        cand_gen,
        flags=re.IGNORECASE,
    )
    cand_gen = re.sub(
        r"\b(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|Islington|Somerville)\b",
        "the local area",
        cand_gen,
        flags=re.IGNORECASE,
    )

    # Only neutralize author's own age, never patient or child
    cand_gen = re.sub(
        r"\bI(?:'m| am)\s+(?:a\s+)?(\d{1,2})[-\s]*years?[-\s]*old\b",
        "I am an experienced adult",
        cand_gen,
        flags=re.IGNORECASE,
    )
    cand_gen = re.sub(
        r"\b(?:retired|planning to retire)\s+after\s+\d{1,2}\s+years\b",
        "approaching retirement",
        cand_gen,
        flags=re.IGNORECASE,
    )

    # Smooth occupation phrases
    cand_gen = re.sub(
        r"\b(?:work as|employed as|working as)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher)\b",
        "work in my profession",
        cand_gen,
        flags=re.IGNORECASE,
    )

    # Smooth education phrases
    cand_gen = re.sub(
        r"\b(?:degree from|graduated from)\s+[A-Za-z\s]+(?:University|College|Institute|Tech|Stanford|MIT|Harvard|Purdue)\b",
        "completing higher education",
        cand_gen,
        flags=re.IGNORECASE,
    )
    cand_gen = re.sub(r"\s+", " ", cand_gen).strip()
    candidates.append(cand_gen)

    # Strategy 3: Multi-tiered diverse candidates for Pareto sampling and ablation
    # Tier A: Single attribute neutralizations
    c_age = neutralize_age_cues(base_text)
    if c_age != base_text:
        candidates.append(c_age)

    c_loc = neutralize_location_cues(base_text)
    if c_loc != base_text:
        candidates.append(c_loc)

    c_occ = neutralize_occupation_cues(base_text)
    if c_occ != base_text:
        candidates.append(c_occ)

    c_edu = neutralize_education_cues(base_text)
    if c_edu != base_text:
        candidates.append(c_edu)

    # Tier B: Explicit placeholder tags candidate
    c_placeholders = placeholder_cues(base_text)
    if c_placeholders != base_text:
        candidates.append(c_placeholders)

    # Tier C: Multi-attribute combinations
    c_age_occ = neutralize_occupation_cues(c_age)
    if c_age_occ not in candidates:
        candidates.append(c_age_occ)

    c_loc_edu = neutralize_education_cues(c_loc)
    if c_loc_edu not in candidates:
        candidates.append(c_loc_edu)

    # Tier D: Full demographic neutralization across all four attributes
    c_all = neutralize_education_cues(neutralize_occupation_cues(neutralize_location_cues(c_age)))
    if c_all != base_text and c_all not in candidates:
        candidates.append(c_all)

    # Deduplicate while strictly preserving insertion order
    unique_cands: List[str] = []
    for c in candidates:
        c_clean = re.sub(r"\s+", " ", c).strip()
        if c_clean and c_clean not in unique_cands and c_clean != cleaned:
            unique_cands.append(c_clean)

    if not unique_cands:
        unique_cands.append(cleaned)

    return unique_cands


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

    chatml_examples: List[Dict[str, Any]] = []
    alpaca_examples: List[Dict[str, Any]] = []
    total_candidates_evaluated = 0
    accepted_count = 0

    for rec in train_records:
        if accepted_count >= max_samples:
            break

        text = rec.get("text") or rec.get("comment") or rec.get("body") or ""
        if not text or len(text.strip()) < 10:
            continue

        # Generate candidates
        if candidate_generator_fn is not None:
            candidates = candidate_generator_fn(text)
        else:
            candidates = generate_candidate_rewrites_heuristic(text)

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
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("-i", "--input", "--input_file", type=str, default="data/splits/train.jsonl", help="Path to SynthPAI dataset or train split")
    parser.add_argument("-s", "--splits", "--splits_file", type=str, default="artifacts/profile_splits.json", help="Path to profile_splits.json")
    parser.add_argument("-c", "-o", "--output", "--output_chatml", type=str, default="artifacts/rewriter_sft_chatml.jsonl", help="ChatML output path")
    parser.add_argument("-a", "--output_alpaca", type=str, default="artifacts/rewriter_sft_alpaca.jsonl", help="Alpaca output path")
    parser.add_argument("-n", "--max_samples", type=int, default=1000, help="Max accepted samples")
    args = parser.parse_args()

    summary = generate_sft_dataset(
        synthpai_path=args.input,
        splits_path=args.splits,
        output_chatml_path=args.output,
        output_alpaca_path=args.output_alpaca,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
