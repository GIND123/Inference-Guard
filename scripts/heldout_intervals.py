"""Confidence intervals for a comparative held-out benchmark, from its committed JSON.

src/evaluation/run.py reports one attack success rate (ASR) per method, but not
how uncertain it is. At n=50 samples x 4 attributes that uncertainty is large
enough to change which claims the numbers support, so this script attaches a
95% Wilson interval to each method's overall and per-attribute ASR and says,
for each pair of methods, whether the intervals overlap.

The benchmark JSON stores rates, not per-sample outcomes, so the intervals
treat each (sample, attribute) attack as an independent trial and the
comparison is unpaired. Both choices make the test conservative relative to a
paired test on the same samples, which needs per-sample outputs the benchmark
does not commit.

Usage:
    python scripts/heldout_intervals.py reports/promptpai/comparative_benchmark_test.json
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

Z95 = 1.959964


def wilson(rate: float, n: int, z: float = Z95) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion observed as `rate` over `n` trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    denom = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denom
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def intervals(bench: Dict[str, Any]) -> Dict[str, Any]:
    n = int(bench["sample_count"])
    per_attr = bench.get("per_attribute_attack_success", {})
    methods = list(bench["methods"])
    out: Dict[str, Any] = {"sample_count": n, "methods": {}, "pairs": []}
    for m in methods:
        overall = float(bench["methods"][m]["overall_attack_success_rate"])
        attrs = per_attr.get(m, {})
        trials = n * max(1, len(attrs))
        lo, hi = wilson(overall, trials)
        out["methods"][m] = {
            "asr": overall, "trials": trials, "ci95": [round(lo, 4), round(hi, 4)],
            "per_attribute": {a: {"asr": r, "ci95": [round(x, 4) for x in wilson(float(r), n)]}
                              for a, r in attrs.items()},
        }
    for a, b in combinations(methods, 2):
        ca, cb = out["methods"][a]["ci95"], out["methods"][b]["ci95"]
        out["pairs"].append({"a": a, "b": b, "asr_a": out["methods"][a]["asr"], "asr_b": out["methods"][b]["asr"],
                             "intervals_overlap": ca[0] <= cb[1] and cb[0] <= ca[1]})
    return out


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", help="comparative benchmark JSON written by src/evaluation/run.py")
    parser.add_argument("--out", help="where to write the intervals JSON")
    args = parser.parse_args(argv)

    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    result = intervals(bench)
    result["source"] = args.benchmark
    for key in ("split", "attacker_model_id", "attacker_mode", "reportable_as_held_out",
                "risk_model_id", "rewriter_model_id"):
        result[key] = bench.get(key)

    print(f"{args.benchmark}: n={result['sample_count']} samples")
    print(f"{'method':<18}{'ASR':>7}  95% CI")
    for m, v in result["methods"].items():
        print(f"{m:<18}{v['asr']:>7.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  ({v['trials']} attacks)")
    for p in result["pairs"]:
        verdict = "overlap (not distinguishable)" if p["intervals_overlap"] else "separate"
        print(f"  {p['a']} vs {p['b']}: {p['asr_a']:.3f} vs {p['asr_b']:.3f} -> {verdict}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
