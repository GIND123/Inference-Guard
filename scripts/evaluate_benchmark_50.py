"""Benchmark evaluation runner for 50 diverse test cases across all privacy and utility dimensions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure repository root is in sys.path regardless of working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_benchmark(
    adapter_path: str = "artifacts/rewriter_qlora",
    benchmark_file: str = "data/evaluation/benchmark_test_50.jsonl",
    output_report: str = "reports/benchmark_50_results.json",
    truncate_display: bool = False,
) -> Dict[str, Any]:
    from src.rewriter.inference import QwenRewriterInference

    print(f"Loading rewriter with adapter: {adapter_path}...")
    rewriter = QwenRewriterInference(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=adapter_path,
        load_in_4bit=False,
    )
    print(f"Adapter active: {rewriter._is_adapter_loaded} (Base: {rewriter.resolved_base_model})")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    print(f"Evaluating {len(cases)} test cases...\n")
    results: List[Dict[str, Any]] = []

    passed_count = 0
    category_stats: Dict[str, Dict[str, int]] = {}

    for idx, tc in enumerate(cases, 1):
        cat = tc["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}
        category_stats[cat]["total"] += 1

        inp = tc["input"]
        rewritten = rewriter.rewrite(inp)
        if isinstance(rewritten, tuple):
            rewritten = rewritten[0]

        # Verification checks
        is_passed = True
        notes = []

        # Check 1: Invariant locations/laws/patients preserved if required
        if tc["jurisdiction_or_entity_required"]:
            # Check key entities from input are preserved
            if "California" in inp and "California" not in rewritten:
                is_passed = False
                notes.append("Failed to preserve 'California'")
            if "Texas" in inp and "Texas" not in rewritten:
                is_passed = False
                notes.append("Failed to preserve 'Texas'")
            if "New York" in inp and "New York" not in rewritten:
                is_passed = False
                notes.append("Failed to preserve 'New York'")
            if "Florida" in inp and "Florida" not in rewritten:
                is_passed = False
                notes.append("Failed to preserve 'Florida'")
            if "patient" in inp and "patient" not in rewritten.lower():
                is_passed = False
                notes.append("Failed to preserve clinical 'patient' presentation")

        # Check 2: Personal cues successfully stripped
        if "Johns Hopkins" in inp and "Johns Hopkins" in rewritten:
            is_passed = False
            notes.append("Leaked 'Johns Hopkins'")
        if "Miami Beach" in inp and "Miami Beach" in rewritten:
            is_passed = False
            notes.append("Leaked 'Miami Beach'")
        if "450k" in inp and "450k" in rewritten:
            is_passed = False
            notes.append("Leaked salary '450k'")
        if "Chicago" in inp and cat != "essential_jurisdiction_law" and "Chicago" in rewritten:
            is_passed = False
            notes.append("Leaked personal city 'Chicago'")
        if "Mountain View" in inp and "Mountain View" in rewritten:
            is_passed = False
            notes.append("Leaked personal city 'Mountain View'")

        if is_passed:
            passed_count += 1
            category_stats[cat]["passed"] += 1

        results.append({
            "id": tc["id"],
            "category": cat,
            "input": inp,
            "rewritten": rewritten,
            "passed": is_passed,
            "notes": notes,
        })

        status_sym = "[PASS]" if is_passed else "[FAIL]"
        print(f"{status_sym} #{tc['id']} ({cat}):")
        if truncate_display:
            print(f"    ORIGINAL:  {inp[:80]}...")
            print(f"    REWRITTEN: {rewritten[:80]}...")
        else:
            print(f"    ORIGINAL:  {inp}")
            print(f"    REWRITTEN: {rewritten}")
        if notes:
            print(f"    ISSUES:    {', '.join(notes)}")
        print("-" * 70)

    summary = {
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "pass_rate": round(passed_count / len(cases), 4),
        "category_breakdown": category_stats,
        "results": results,
    }

    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("BENCHMARK 50 EVALUATION SUMMARY:")
    print(f"Overall Pass Rate: {passed_count}/{len(cases)} ({summary['pass_rate'] * 100:.1f}%)")
    for c, s in category_stats.items():
        print(f"  - {c}: {s['passed']}/{s['total']} ({s['passed']/s['total']*100:.1f}%)")
    print(f"Saved full report to: {output_report}")
    print("=" * 50)

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run 50-Example Benchmark Evaluation")
    parser.add_argument("--adapter_path", type=str, default="artifacts/rewriter_qlora")
    parser.add_argument("--benchmark_file", type=str, default="data/evaluation/benchmark_test_50.jsonl")
    parser.add_argument("--output_report", type=str, default="reports/benchmark_50_results.json")
    parser.add_argument("--truncate_display", action="store_true", help="Truncate displayed text to 80 chars (default: False, prints full text)")
    args = parser.parse_args()

    run_benchmark(
        adapter_path=args.adapter_path,
        benchmark_file=args.benchmark_file,
        output_report=args.output_report,
        truncate_display=args.truncate_display,
    )


if __name__ == "__main__":
    main()
