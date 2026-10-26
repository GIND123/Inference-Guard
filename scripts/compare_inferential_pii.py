"""Comparative Diagnostic Tool for Inferred PII Detection and Rewriting in LLM User Queries.

Focuses on the core problem:
When a user queries an LLM to solve a problem (coding, medicine, finance, career, legal)
and includes context with inferential demographic PII (Location, Occupation, Education, Age),
InferenceGuard neutralizes the PII while ensuring the downstream LLM can still solve the problem.

Compares:
1. User Query (Original task query containing inferential demographic context)
2. Presidio Baseline (demonstrating why explicit NER fails on inferential cues)
3. InferenceGuard Risk Model (detection of Age, Location, Occupation, Education)
4. InferenceGuard Rewriter (neutralization of inferential cues while maintaining task solvability)

Usage:
    # Run audit on task-oriented LLM user queries (Default)
    python scripts/compare_inferential_pii.py --dataset user_queries --limit 10

    # Run audit on SynthPAI validation split comments
    python scripts/compare_inferential_pii.py --dataset synthpai --split val --limit 10

    # Run with neural QLoRA adapter in Colab or GPU
    python scripts/compare_inferential_pii.py --dataset user_queries --neural --adapter_dir artifacts/rewriter_qlora
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.presidio_baseline import PresidioBaseline
from src.rewriter.generate_training_data import (
    estimate_privacy_risk,
    generate_candidate_rewrites_heuristic,
)


def sanitize_generated_output(text: str) -> str:
    """Clean model generation by stripping ChatML artifacts, meta-notes, and runaway loops."""
    if not text:
        return ""
    # Stop delimiters
    for stop_pat in [
        r"<\|im_end\|>",
        r"<\|endoftext\|>",
        r"<\|im_start\|>",
        r"<tool_call>",
        r"\[REDACTED\](?:\s*\[REDACTED\]){2,}",
        r"(?:\n|\A|\s*)(?:\*\*)?Note(?:\*\*)?:\s*This rewrite",
        r"(?:\n|\A|\s*)(?:\*\*)?Note(?:\*\*)?:",
        r";?\s*Sure!\s*I'm here to help",
        r"#include\s*<iostream>",
        r"istringstream\b",
        r"\}\}\}\}\}",
        r"<form\b",
        r"Login failed\.\s*Please try again",
    ]:
        m = re.search(stop_pat, text, flags=re.IGNORECASE)
        if m:
            text = text[:m.start()]

    # Strip ChatML control tokens
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)

    # Strip trailing hashtags and runaway emoji bursts
    text = re.sub(r"(\s*#\w+.*)$", "", text)
    text = re.sub(r"(\s*[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]){2,}.*$", "", text)
    text = re.sub(r"([.!?]\s*)(?:[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\s.!?]+)$", r"\1", text)
    return text.strip()


def extract_user_query_samples(file_path: Path, max_samples: int = 10) -> List[Dict[str, Any]]:
    """Extract realistic LLM user queries containing inferential demographic context."""
    if not file_path.exists():
        return get_curated_edge_cases()

    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            records.append({
                "profile_id": rec.get("id", "user_query"),
                "task_domain": rec.get("task_domain", "general_query"),
                "text": rec.get("query", "").strip(),
                "problem_to_solve": rec.get("problem_to_solve", "General task resolution"),
                "leaked_attributes": rec.get("inferred_attributes", {}),
                "ideal_sanitized_query": rec.get("ideal_sanitized_query", ""),
            })

            if len(records) >= max_samples:
                break

    return records


def extract_synthpai_samples(split_path: Path, max_samples: int = 10) -> List[Dict[str, Any]]:
    """Extract records from SynthPAI where human reviewers identified actual leaking cues."""
    """Extract records from SynthPAI where human reviewers identified actual leaking cues."""
    leaking_records: List[Dict[str, Any]] = []

    if not split_path.exists():
        print(f"Warning: {split_path} not found. Using curated diagnostic edge cases.")
        return get_curated_edge_cases()

    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            reviews = rec.get("reviews", {}).get("human", {})
            leaked_attrs = {}

            # Check 4 target demographic attributes
            for attr in ("age", "city_country", "education", "occupation"):
                rev_attr = reviews.get(attr, {})
                cert = rev_attr.get("certainty", 0)
                estimate = rev_attr.get("estimate")
                if cert > 0 or estimate:
                    canonical_name = "location" if attr == "city_country" else attr
                    leaked_attrs[canonical_name] = {
                        "estimate": estimate,
                        "certainty": cert,
                        "hardness": rev_attr.get("hardness", 0),
                    }

            if leaked_attrs:
                leaking_records.append({
                    "profile_id": rec.get("profile_id") or rec.get("author") or "unknown",
                    "task_domain": "social_comment",
                    "text": rec.get("text", "").strip(),
                    "problem_to_solve": "Opinion/Discussion context",
                    "leaked_attributes": leaked_attrs,
                    "ideal_sanitized_query": "",
                })

            if len(leaking_records) >= max_samples:
                break

    return leaking_records


# Backward compatibility alias
extract_leaking_samples = extract_synthpai_samples


def get_curated_edge_cases() -> List[Dict[str, Any]]:
    """Curated canonical benchmark cases covering the 4 target inferential attributes in LLM queries."""
    return [
        {
            "profile_id": "curated_med",
            "task_domain": "medical_case_study",
            "text": "I'm a 3rd-year pediatric resident at Boston Children's Hospital working 80-hour weeks. Can you help me differential-diagnose a 4-year-old presenting with high fever for 6 days, conjunctival injection, and strawberry tongue?",
            "problem_to_solve": "Differential diagnosis for pediatric Kawasaki disease evaluation",
            "leaked_attributes": {
                "occupation": {"estimate": "pediatrician / resident", "certainty": 5, "hardness": 1},
                "location": {"estimate": "Boston, MA", "certainty": 5, "hardness": 1},
                "education": {"estimate": "MD / Medical Resident", "certainty": 5, "hardness": 1},
            },
            "ideal_sanitized_query": "Can you help me differential-diagnose a 4-year-old presenting with high fever for 6 days, conjunctival injection, and strawberry tongue for clinical rounds?",
        },
        {
            "profile_id": "curated_retire",
            "task_domain": "retirement_finance",
            "text": "I'm a 58-year-old high school physics teacher in upstate New York planning to retire in two years. What are the best strategies to coordinate my NY state teacher pension with my 403b savings?",
            "problem_to_solve": "Retirement pension and 403b tax coordination advice",
            "leaked_attributes": {
                "age": {"estimate": "55-64", "certainty": 5, "hardness": 1},
                "occupation": {"estimate": "teacher / educator", "certainty": 5, "hardness": 1},
                "location": {"estimate": "New York", "certainty": 4, "hardness": 1},
            },
            "ideal_sanitized_query": "What are the best strategies to coordinate a public educator state pension with supplemental 403(b) retirement accounts?",
        },
    ]


def run_comparison(
    samples: List[Dict[str, Any]],
    use_neural_rewriter: bool = False,
    adapter_path: Optional[str] = "artifacts/rewriter_qlora",
) -> List[Dict[str, Any]]:
    """Execute side-by-side comparison across Presidio, Risk Model, and Rewriter."""
    presidio = PresidioBaseline(use_presidio_if_available=True)

    neural_rewriter = None
    if use_neural_rewriter:
        try:
            print("Loading neural rewriter...")
            neural_rewriter = QwenRewriterInference(adapter_path=adapter_path)
        except Exception as exc:
            print(f"Neural rewriter load note: {exc}. Operating with heuristic rewriter.")

    results = []

    for i, s in enumerate(samples, 1):
        orig_text = s["text"]
        leaked = s["leaked_attributes"]
        problem = s.get("problem_to_solve", "")

        # 1. Presidio Baseline Redaction
        pres_res = presidio.analyze_sample(orig_text)
        presidio_redacted = pres_res.redacted
        presidio_entities = pres_res.entities_found

        # 2. InferenceGuard Inferred Risk Estimation
        risk_scores = estimate_privacy_risk(orig_text)

        # 3. InferenceGuard Rewrite
        if neural_rewriter and getattr(neural_rewriter, "is_adapter_loaded", False):
            raw_rewrite = neural_rewriter.rewrite(orig_text)
            rewrite_text = sanitize_generated_output(raw_rewrite)
            rewrite_engine = "qlora_sft (neural)"
        else:
            candidates = generate_candidate_rewrites_heuristic(orig_text)
            raw_c = candidates[0] if candidates else orig_text
            rewrite_text = sanitize_generated_output(raw_c)
            rewrite_engine = "heuristic regex"

        presidio_changed = presidio_redacted != orig_text
        rewrite_changed = rewrite_text != orig_text

        results.append({
            "index": i,
            "profile_id": s["profile_id"],
            "task_domain": s.get("task_domain", "query"),
            "problem_to_solve": problem,
            "input_text": orig_text,
            "target_inferred_pii": leaked,
            "presidio_redacted": presidio_redacted,
            "presidio_entities_found": presidio_entities,
            "presidio_changed_text": presidio_changed,
            "inferred_risk_scores": risk_scores,
            "rewritten_text": rewrite_text,
            "rewriter_engine": rewrite_engine,
            "rewriter_changed_text": rewrite_changed,
            "ideal_sanitized_query": s.get("ideal_sanitized_query", ""),
        })

    return results


def format_markdown_report(results: List[Dict[str, Any]], dataset_name: str) -> str:
    """Format comparison results into a clean, legible markdown report."""
    md = []
    md.append(f"# InferenceGuard: LLM Query Inferential PII Audit ({dataset_name})\n")
    md.append(
        "> **Objective**: Protect user privacy when submitting queries to an LLM. "
        "The system must neutralize **inferred demographic context** (Location, Occupation, Education, Age) "
        "while ensuring the downstream LLM retains full semantic utility to solve the user's problem.\n"
    )

    for r in results:
        md.append(f"## Query {r['index']} [{r['task_domain']}] (ID: `{r['profile_id']}`)\n")
        md.append(f"**Original User Query to LLM:**\n> {r['input_text']}\n")
        md.append(f"**Problem the User is Trying to Solve:**\n* {r['problem_to_solve']}\n")

        md.append("**Inferred Demographic PII Leaked in Prompt Context:**")
        for attr, details in r["target_inferred_pii"].items():
            if isinstance(details, dict):
                est = details.get("estimate") or details.get("target") or "leaked"
                cues = details.get("cues")
                cues_str = f" (cues: {cues})" if cues else ""
                md.append(f"- **{attr.upper()}**: Inferred as `\"{est}\"`{cues_str}")
            else:
                md.append(f"- **{attr.upper()}**: `{details}`")
        md.append("")

        md.append("**1. Presidio Baseline Output:**")
        md.append(f"> {r['presidio_redacted']}")
        if not r["presidio_changed_text"]:
            md.append("*(Entities Detected: None — Presidio was 100% blind to this inferential context)*\n")
        else:
            md.append(f"*(Entities Detected: {r['presidio_entities_found']})*\n")

        md.append("**2. InferenceGuard Inferred Risk Scores:**")
        scores = r["inferred_risk_scores"]
        md.append(
            f"- Age: `{scores.get('age', 0.0):.2f}` | Location: `{scores.get('location', 0.0):.2f}` | "
            f"Occupation: `{scores.get('occupation', 0.0):.2f}` | Education: `{scores.get('education', 0.0):.2f}` | "
            f"**Overall Risk:** `{scores.get('overall', 0.0):.2f}`\n"
        )

        md.append(f"**3. InferenceGuard Sanitized Query ({r['rewriter_engine']}):**")
        md.append(f"> {r['rewritten_text']}")
        if not r["rewriter_changed_text"]:
            md.append("*(Cues Sanitized: NO — Query remained identical to input. Model needs targeted improvements.)*\n")
        else:
            md.append("*(Cues Sanitized: YES — Model abstracted the demographic context)*\n")

        if r.get("ideal_sanitized_query"):
            md.append(f"**Target Reference Sanitization:**\n> {r['ideal_sanitized_query']}\n")

        md.append("---\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Inspect and compare inferred PII detection in LLM user queries.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="user_queries",
        choices=["user_queries", "synthpai"],
        help="Dataset type: 'user_queries' for task-oriented prompts, 'synthpai' for Reddit comments",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split for synthpai (val or test)")
    parser.add_argument("--input_file", type=str, default=None, help="Custom jsonl input file")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--neural", action="store_true", help="Attempt loading QLoRA neural rewriter")
    parser.add_argument("--adapter_dir", type=str, default="artifacts/rewriter_qlora", help="Path to adapter")
    parser.add_argument("--output", type=str, default=None, help="Save report to markdown file")
    args = parser.parse_args()

    if args.input_file:
        query_path = Path(args.input_file)
        samples = extract_user_query_samples(query_path, max_samples=args.limit)
        dataset_name = query_path.name
    elif args.dataset == "user_queries":
        query_path = ROOT / "data" / "user_llm_queries_inferential_pii.jsonl"
        samples = extract_user_query_samples(query_path, max_samples=args.limit)
        dataset_name = "Realistic LLM User Queries"
    else:
        split_file = ROOT / "data" / "splits" / f"{args.split}.jsonl"
        samples = extract_synthpai_samples(split_file, max_samples=args.limit)
        dataset_name = f"SynthPAI {args.split} split"

    print(f"Loaded {len(samples)} samples from '{dataset_name}'.")

    results = run_comparison(samples, use_neural_rewriter=args.neural, adapter_path=args.adapter_dir)
    report = format_markdown_report(results, dataset_name)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_p}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
