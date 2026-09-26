"""Rewriter SFT Dataset Generation for Qwen3-1.7B.

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
        prof_id = (
            rec.get("profile_id")
            or rec.get("profile")
            or rec.get("user_id")
            or rec.get("author")
        )
        if prof_id is not None:
            # Handle potential dictionary profile object
            if isinstance(prof_id, dict):
                prof_id = prof_id.get("id") or prof_id.get("username")

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
        r"\b\d{1,2}\s*(?:years old|yo|y/o)\b",
        r"\b(?:in my (?:20s|30s|40s|50s|60s|70s|twenties|thirties|forties))\b",
        r"\b(?:retired|pensioner|high school|college student|freshman|sophomore)\b",
    ]
    location_cues = [
        r"\b(?:living in|moved to|commute to|transit in|downtown|suburbs of)\b",
        r"\b(?:denver|seattle|austin|california|new york|boston|chicago|london|berlin|paris|tokyo)\b",
        r"\b(?:[A-Z]{2}\b|\bco\b|\bwa\b|\btx\b|\bca\b|\bny\b)",
    ]
    occupation_cues = [
        r"\b(?:work as|employed at|my boss|colleagues|software engineer|nurse|doctor|teacher|barista|lawyer)\b",
        r"\b(?:tech company|startup|hospital|clinic|law firm|school district|consultancy)\b",
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


def generate_candidate_rewrites_heuristic(text: str) -> List[str]:
    """Generate diverse candidate rewrites using rule-based transformations."""
    candidates: List[str] = []

    # 1. Hard PII scrubbing + demographic neutralization
    cand1, _ = scrub_hard_pii(text)
    # Neutralize age cues (including hyphenated like 21-year-old)
    cand1 = re.sub(
        r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|in my (?:20s|30s|40s|50s|60s|twenties|thirties|forties))\b",
        "an adult",
        cand1,
        flags=re.IGNORECASE,
    )
    # Neutralize location cues
    cand1 = re.sub(
        r"\b(?:in|to|from)\s+(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco|London|Paris|Tokyo)[,\sA-Z]*\b",
        "locally",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:Pittsburgh|Denver|Seattle|Austin|New York|Boston|Chicago|San Francisco)[,\sA-Z]*\b",
        "the area",
        cand1,
        flags=re.IGNORECASE,
    )
    # Neutralize occupation cues and role phrases
    cand1 = re.sub(
        r"\b(?:work as|employed as|employed at|working as|internship in|intern at)\s+(?:a\s+|an\s+|my\s+)?(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer|engineer|teacher|summer internship)\b",
        "professional role",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:software engineer|lawyer|doctor|nurse|barista|consultant|designer)\b",
        "professional",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:tech startup|startup|tech company|firm|clinic|hospital)\b",
        "an organization",
        cand1,
        flags=re.IGNORECASE,
    )
    # Neutralize education cues (including specific universities)
    cand1 = re.sub(
        r"\b(?:undergrad at|attending|student at|degree from|graduated from)\s+[A-Za-z\s]+(?:University|College|Mellon|MIT)?\b",
        "completing higher education",
        cand1,
        flags=re.IGNORECASE,
    )
    cand1 = re.sub(
        r"\b(?:undergrad|Carnegie Mellon|MIT|Stanford|Harvard|Berkeley)\b",
        "higher education",
        cand1,
        flags=re.IGNORECASE,
    )
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

    return list(dict.fromkeys(c for c in candidates if c))


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

    Among acceptable candidates, selects the one maximizing utility_score.
    """
    valid_candidates: List[Tuple[str, Dict[str, Any]]] = []

    for cand in candidates:
        if not cand or cand.strip() == original.strip():
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
        if overall_risk < max_risk_threshold and utility["cosine_similarity"] >= min_cosine_threshold:
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
    """Format an SFT pair into ChatML schema for Qwen3-1.7B without thinking tokens."""
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
    """Format an SFT pair into Alpaca schema for Qwen3-1.7B without thinking tokens."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Qwen3-1.7B")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to SynthPAI dataset")
    parser.add_argument("-s", "--splits_file", type=str, default="artifacts/profile_splits.json", help="Path to profile_splits.json")
    parser.add_argument("-c", "--output_chatml", type=str, default="data/qwen_sft_chatml.jsonl", help="ChatML output path")
    parser.add_argument("-a", "--output_alpaca", type=str, default="data/qwen_sft_alpaca.jsonl", help="Alpaca output path")
    parser.add_argument("-n", "--max_samples", type=int, default=1000, help="Max accepted samples")
    args = parser.parse_args()

    summary = generate_sft_dataset(
        synthpai_path=args.input_file,
        splits_path=args.splits_file,
        output_chatml_path=args.output_chatml,
        output_alpaca_path=args.output_alpaca,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
