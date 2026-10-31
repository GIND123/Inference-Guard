"""Span-annotated benchmark evaluation for the InferenceGuard rewriter.

v2 - replaces the hardcoded five-string checklist ("Johns Hopkins", "Miami
Beach", "450k", "Chicago", "Mountain View") that reported 50/50 PASS while
tc_09 and tc_18 leaked full personas by relocating them into a trailing
clause. A keyword list can only catch the leaks its author imagined; span
recall catches any annotated surface that survives, wherever it ends up.

Scores each case against data/evaluation/benchmark_test_50_annotated.jsonl
(produced by data/evaluation/annotate_benchmark_spans.py, driven by the same
persona banks the rest of the system uses):

  span_removal_rate : fraction of annotated pii_spans absent from the rewrite
                      (case-insensitive, whole phrase). The privacy metric.
  invariant_rate    : fraction of invariant_spans (jurisdictions, standards,
                      third-party subjects) still present. The utility metric.
  copy_through      : rewrite == input although pii_spans exist (the v1
                      student's failure mode: token_edit_ratio 0.0).
  added_pii         : banked persona values present in the rewrite but not in
                      the input - catches "rewrote into a different leak".

Reports per-category (position-stratified: preamble vs postamble vs domain
cases) and overall. No per-case strings are hardcoded anywhere.

Usage:
    python scripts/evaluate_benchmark_50.py --adapter_path artifacts/rewriter_qlora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK = "data/evaluation/benchmark_test_50_annotated.jsonl"


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Case-insensitive whole-phrase containment."""
    import re

    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack, re.IGNORECASE) is not None


def evaluate_rewrite(
    case: Dict[str, Any],
    rewritten: str,
    new_value_scanner: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, Any]:
    """Score one rewrite against a case's annotations."""
    pii_spans = case.get("pii_spans", [])
    invariant_spans = case.get("invariant_spans", [])

    leaked = [s["text"] for s in pii_spans if _contains_phrase(rewritten, s["text"])]
    missing_invariants = [s for s in invariant_spans if not _contains_phrase(rewritten, s)]
    removal_rate = 1.0 - (len(leaked) / len(pii_spans)) if pii_spans else None
    invariant_rate = (1 - len(missing_invariants) / len(invariant_spans)) if invariant_spans else None
    copy_through = bool(pii_spans) and rewritten.strip() == case["input"].strip()

    added: List[str] = []
    if new_value_scanner is not None:
        added = [v for v in new_value_scanner(rewritten) if not _contains_phrase(case["input"], v)]

    passed = True
    if leaked:
        passed = False
    if missing_invariants:
        passed = False
    if copy_through:
        passed = False

    return {
        "leaked_spans": leaked,
        "span_removal_rate": round(removal_rate, 4) if removal_rate is not None else None,
        "missing_invariants": missing_invariants,
        "invariant_rate": round(invariant_rate, 4) if invariant_rate is not None else None,
        "copy_through": copy_through,
        "added_pii": added,
        "passed": passed,
    }


def run_benchmark(
    adapter_path: str = "artifacts/rewriter_qlora",
    benchmark_file: str = DEFAULT_BENCHMARK,
    output_report: str = "reports/benchmark_50_results.json",
    truncate_display: bool = False,
    rewriter_fn: Optional[Callable[[str], str]] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> Dict[str, Any]:
    """Evaluate the rewriter on the annotated benchmark.

    ``rewriter_fn`` overrides adapter-based rewriting (used by tests and by
    the bank-driven fallback baseline).
    """
    bench_path = Path(benchmark_file)
    if not bench_path.exists():
        annotated = bench_path.with_name(bench_path.name.replace("_annotated", ""))
        raise FileNotFoundError(
            f"{bench_path} not found. Generate it with "
            f"`python data/evaluation/annotate_benchmark_spans.py --input {annotated}`"
        )
    with open(bench_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    rewriter = None
    if rewriter_fn is None:
        from src.rewriter.inference import QwenRewriterInference

        print(f"Loading rewriter with adapter: {adapter_path}...")
        rewriter = QwenRewriterInference(model_name_or_path=model_name, adapter_path=adapter_path,
                                         load_in_4bit=False)
        print(f"Adapter active: {rewriter._is_adapter_loaded} (Base: {rewriter.resolved_base_model})")

    new_value_scanner = None
    try:
        from src.rewriter.persona_banks import default_matcher

        matcher = default_matcher()

        def _scan_new(text: str) -> List[str]:
            return [m.surface for m in matcher.scan(text)]

        new_value_scanner = _scan_new
    except Exception:
        pass

    print(f"Evaluating {len(cases)} test cases...\n")
    results: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = {}

    for idx, tc in enumerate(cases, 1):
        cat = tc["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0, "passed": 0, "removal_rates": [], "invariant_rates": [], "copy_through": 0,
            }
        category_stats[cat]["total"] += 1

        inp = tc["input"]
        if rewriter_fn is not None:
            rewritten = rewriter_fn(inp)
        else:
            rewritten = rewriter.rewrite(inp)
        if isinstance(rewritten, tuple):
            rewritten = rewritten[0]

        scores = evaluate_rewrite(tc, rewritten, new_value_scanner)
        if scores["span_removal_rate"] is not None:
            category_stats[cat]["removal_rates"].append(scores["span_removal_rate"])
        if scores["invariant_rate"] is not None:
            category_stats[cat]["invariant_rates"].append(scores["invariant_rate"])
        if scores["copy_through"]:
            category_stats[cat]["copy_through"] += 1
        if scores["passed"]:
            category_stats[cat]["passed"] += 1

        results.append({
            "id": tc["id"],
            "category": cat,
            "input": inp,
            "rewritten": rewritten,
            **scores,
        })

        status_sym = "[PASS]" if scores["passed"] else "[FAIL]"
        print(f"{status_sym} #{tc['id']} ({cat}):")
        if truncate_display:
            print(f"    ORIGINAL:  {inp[:80]}...")
            print(f"    REWRITTEN: {rewritten[:80]}...")
        else:
            print(f"    ORIGINAL:  {inp}")
            print(f"    REWRITTEN: {rewritten}")
        if scores["leaked_spans"]:
            print(f"    LEAKED:    {scores['leaked_spans']}")
        if scores["missing_invariants"]:
            print(f"    LOST:      {scores['missing_invariants']}")
        if scores["copy_through"]:
            print("    COPY-THROUGH: rewrite is identical to input despite annotated PII")
        print("-" * 70)

    all_removal = [r["span_removal_rate"] for r in results if r["span_removal_rate"] is not None]
    all_invariant = [r["invariant_rate"] for r in results if r["invariant_rate"] is not None]
    summary_by_cat = {}
    for cat, s in category_stats.items():
        removals = [r["span_removal_rate"] for r in results
                    if r["category"] == cat and r["span_removal_rate"] is not None]
        invariants = [r["invariant_rate"] for r in results
                      if r["category"] == cat and r["invariant_rate"] is not None]
        summary_by_cat[cat] = {
            "total": s["total"],
            "passed": s["passed"],
            "mean_span_removal_rate": round(sum(removals) / len(removals), 4) if removals else None,
            "mean_invariant_rate": round(sum(invariants) / len(invariants), 4) if invariants else None,
            "copy_through": s["copy_through"],
        }

    passed_count = sum(1 for r in results if r["passed"])
    summary = {
        "benchmark_file": str(bench_path),
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "pass_rate": round(passed_count / len(cases), 4),
        "mean_span_removal_rate": round(sum(all_removal) / len(all_removal), 4) if all_removal else None,
        "mean_invariant_rate": round(sum(all_invariant) / len(all_invariant), 4) if all_invariant else None,
        "copy_through_cases": sum(1 for r in results if r["copy_through"]),
        "total_leaked_spans": sum(len(r["leaked_spans"]) for r in results),
        "cases_with_added_pii": sum(1 for r in results if r["added_pii"]),
        "category_breakdown": summary_by_cat,
        "results": results,
    }

    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("BENCHMARK 50 EVALUATION SUMMARY (span-annotated, v2):")
    print(f"Overall Pass Rate: {passed_count}/{len(cases)} ({summary['pass_rate'] * 100:.1f}%)")
    print(f"Mean PII span removal rate: {summary['mean_span_removal_rate']}")
    print(f"Mean invariant preservation rate: {summary['mean_invariant_rate']}")
    print(f"Copy-through failures: {summary['copy_through_cases']}")
    for c, s in summary_by_cat.items():
        print(f"  - {c}: pass {s['passed']}/{s['total']}, removal={s['mean_span_removal_rate']}, "
              f"invariants={s['mean_invariant_rate']}, copy_through={s['copy_through']}")
    print(f"Saved full report to: {output_report}")
    print("=" * 50)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the span-annotated 50-case benchmark")
    parser.add_argument("--adapter_path", type=str, default="artifacts/rewriter_qlora")
    parser.add_argument("--benchmark_file", type=str, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output_report", type=str, default="reports/benchmark_50_results.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--fallback_baseline", action="store_true",
                        help="Score the bank-driven heuristic fallback instead of the adapter")
    parser.add_argument("--truncate_display", action="store_true")
    args = parser.parse_args()

    rewriter_fn = None
    if args.fallback_baseline:
        from src.rewriter.persona_banks import neutralize_text
        from src.rewriter.prompting import prune_persona_bidirectional

        def rewriter_fn(text: str) -> str:
            pruned = prune_persona_bidirectional(text)
            return neutralize_text(pruned if pruned else text)

    run_benchmark(
        adapter_path=args.adapter_path,
        benchmark_file=args.benchmark_file,
        output_report=args.output_report,
        truncate_display=args.truncate_display,
        rewriter_fn=rewriter_fn,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
