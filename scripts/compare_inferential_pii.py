"""Comparative Diagnostic Tool for Inferred PII Detection and Rewriting.

Compares:
1. Input Text (with known ground-truth inferred demographic leakage)
2. Presidio Baseline (demonstrating why explicit NER fails on inferential cues)
3. InferenceGuard Risk Model (detection of Age, Location, Occupation, Education)
4. InferenceGuard Rewriter (neutralization of inferential cues while maintaining utility)

Usage:
    python scripts/compare_inferential_pii.py --limit 10
    python scripts/compare_inferential_pii.py --split val --output reports/inferential_pii_comparison.md
"""

from __future__ import annotations

import argparse
import json
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


def extract_leaking_samples(split_path: Path, max_samples: int = 10) -> List[Dict[str, Any]]:
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
                    "text": rec.get("text", "").strip(),
                    "leaked_attributes": leaked_attrs,
                    "ground_truth_profile": {
                        "age": rec.get("profile", {}).get("age"),
                        "location": rec.get("profile", {}).get("city_country"),
                        "education": rec.get("profile", {}).get("education"),
                        "occupation": rec.get("profile", {}).get("occupation"),
                    },
                })

            if len(leaking_records) >= max_samples:
                break

    return leaking_records


def get_curated_edge_cases() -> List[Dict[str, Any]]:
    """Curated canonical benchmark cases covering the 4 target inferential attributes."""
    return [
        {
            "profile_id": "curated_occ",
            "text": "Roles getting fuzzy isn't new on my radar; had clients push beyond boundaries solely pegged by society norms. Saw them thrive when they ditched 'typical.'",
            "leaked_attributes": {"occupation": {"estimate": "therapist / psychologist", "certainty": 3, "hardness": 3}},
            "ground_truth_profile": {"occupation": "psychologist"},
        },
        {
            "profile_id": "curated_edu",
            "text": "Definitely felt that traditional mold pressing in during grad school days - psych wasn't always seen as 'masculine'. Times change though!",
            "leaked_attributes": {"education": {"estimate": "master's degree in psychology", "certainty": 5, "hardness": 2}},
            "ground_truth_profile": {"education": "PHD in Psychology"},
        },
        {
            "profile_id": "curated_loc",
            "text": "I take the Green Line to Fenway after my clinical shift every Tuesday evening.",
            "leaked_attributes": {"location": {"estimate": "Boston, MA", "certainty": 5, "hardness": 1}},
            "ground_truth_profile": {"location": "Boston, United States"},
        },
        {
            "profile_id": "curated_age",
            "text": "Watching the 1986 Challenger disaster live in homeroom left an indelible impression on our entire generation.",
            "leaked_attributes": {"age": {"estimate": "50-55", "certainty": 4, "hardness": 2}},
            "ground_truth_profile": {"age": 52},
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
            from src.rewriter.inference import QwenRewriterInference
            print("Loading neural rewriter...")
            neural_rewriter = QwenRewriterInference(adapter_path=adapter_path)
        except Exception as exc:
            print(f"Neural rewriter load note: {exc}. Operating with heuristic rewriter.")

    results = []

    for i, s in enumerate(samples, 1):
        orig_text = s["text"]
        leaked = s["leaked_attributes"]

        # 1. Presidio Baseline Redaction
        pres_res = presidio.analyze_sample(orig_text)
        presidio_redacted = pres_res.redacted
        presidio_entities = pres_res.entities_found

        # 2. InferenceGuard Inferred Risk Estimation
        risk_scores = estimate_privacy_risk(orig_text)

        # 3. InferenceGuard Rewrite
        if neural_rewriter and getattr(neural_rewriter, "is_adapter_loaded", False):
            rewrite_text = neural_rewriter.rewrite(orig_text)
            rewrite_engine = "qlora_sft (neural)"
        else:
            candidates = generate_candidate_rewrites_heuristic(orig_text)
            rewrite_text = candidates[0] if candidates else orig_text
            rewrite_engine = "heuristic regex"

        # Check if inferential cues survived
        presidio_leaked_cues_scrubbed = presidio_redacted != orig_text
        rewrite_cues_scrubbed = rewrite_text != orig_text

        results.append({
            "index": i,
            "profile_id": s["profile_id"],
            "input_text": orig_text,
            "target_inferred_pii": leaked,
            "presidio_redacted": presidio_redacted,
            "presidio_entities_found": presidio_entities,
            "presidio_changed_text": presidio_leaked_cues_scrubbed,
            "inferred_risk_scores": risk_scores,
            "rewritten_text": rewrite_text,
            "rewriter_engine": rewrite_engine,
            "rewriter_changed_text": rewrite_cues_scrubbed,
        })

    return results


def format_markdown_report(results: List[Dict[str, Any]]) -> str:
    """Format comparison results into a clean, legible markdown report."""
    md = []
    md.append("# InferenceGuard: Inferential PII Detection & Redaction Audit\n")
    md.append(
        "This report compares raw input containing **inferred personal cues** (Age, Location, "
        "Occupation, Education) against **Presidio explicit redaction** and **InferenceGuard rewriting**.\n"
    )

    for r in results:
        md.append(f"## Sample {r['index']} (Profile: `{r['profile_id']}`)\n")
        md.append(f"**Original Text:**\n> {r['input_text']}\n")

        md.append("**Target Inferred Demographic Leakage (Human Annotations):**")
        for attr, details in r["target_inferred_pii"].items():
            md.append(f"- **{attr.upper()}**: Inferred as `\"{details['estimate']}\"` (certainty: {details['certainty']}/5, hardness: {details['hardness']})")
        md.append("")

        md.append("**1. Presidio Baseline Output:**")
        md.append(f"> {r['presidio_redacted']}")
        if not r["presidio_changed_text"]:
            md.append("*(Entities Detected: None — Presidio was 100% blind to these inferential cues)*\n")
        else:
            md.append(f"*(Entities Detected: {r['presidio_entities_found']})*\n")

        md.append("**2. InferenceGuard Inferred Risk Scores:**")
        scores = r["inferred_risk_scores"]
        md.append(
            f"- Age: `{scores.get('age', 0.0):.2f}` | Location: `{scores.get('location', 0.0):.2f}` | "
            f"Occupation: `{scores.get('occupation', 0.0):.2f}` | Education: `{scores.get('education', 0.0):.2f}` | "
            f"**Overall Risk:** `{scores.get('overall', 0.0):.2f}`\n"
        )

        md.append(f"**3. InferenceGuard Rewritten Output ({r['rewriter_engine']}):**")
        md.append(f"> {r['rewritten_text']}")
        if not r["rewriter_changed_text"]:
            md.append("*(Cues Sanitized: NO — Output remained identical to input. Model needs targeted improvements.)*\n")
        else:
            md.append("*(Cues Sanitized: YES — Model altered input text)*\n")

        md.append("---\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Inspect and compare inferred PII detection and redaction.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test)")
    parser.add_argument("--limit", type=int, default=5, help="Number of leaking samples to compare")
    parser.add_argument("--neural", action="store_true", help="Attempt loading QLoRA neural rewriter")
    parser.add_argument("--adapter_dir", type=str, default="artifacts/rewriter_qlora", help="Path to adapter")
    parser.add_argument("--output", type=str, default=None, help="Save report to markdown file")
    args = parser.parse_args()

    split_file = Path(f"data/splits/{args.split}.jsonl")
    samples = extract_leaking_samples(split_file, max_samples=args.limit)
    print(f"Extracted {len(samples)} samples containing confirmed inferential PII.")

    results = run_comparison(samples, use_neural_rewriter=args.neural, adapter_path=args.adapter_dir)
    report = format_markdown_report(results)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_p}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
