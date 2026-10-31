"""Generate contrastive preference pairs (prompt, chosen, rejected) for DPO.

Produces pairs covering the primary failure modes:
1. Truncation / Over-scrubbing: cuts off valid downstream questions or requests.
2. Leak / Under-scrubbing: leaves identifying author cues in the rewritten text.
3. Degenerate / Broken syntax: leaves bracketed placeholders or damaged grammar.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rewriter.augment import compose_corpus, AugmentedExample
from src.rewriter.prompting import build_rewrite_messages


def make_truncated_negative(target: str, input_text: str) -> Optional[str]:
    """Simulate greedy pruning by dropping downstream clauses."""
    parts = re.split(r",\s*(?:and\s+|can\s+|could\s+|please\s+|help\s+)?", target)
    if len(parts) > 1 and len(parts[0].split()) >= 4:
        cand = parts[0].strip()
        if not cand.endswith(("?", ".")):
            cand += "?" if "?" in target else "."
        if cand != target:
            return cand
    words = target.split()
    if len(words) >= 12:
        return " ".join(words[: len(words) // 2]) + ("?" if "?" in target else ".")
    return None


def make_leaky_negative(ex: AugmentedExample) -> Optional[str]:
    """Simulate under-scrubbing by preserving the author persona cue."""
    if not ex.injected:
        return None
    raw = ex.input.strip()
    if raw != ex.target.strip():
        return raw
    return None


def make_degenerate_negative(target: str, ex: AugmentedExample) -> str:
    """Simulate placeholder substitution or syntax clobbering."""
    if ex.injected:
        attr = ex.injected[0].attribute.upper()
        res = re.sub(r"\bmy (?:work|role|area|studies)\b", f"[{attr}]", target)
        if res.strip() != target.strip():
            return res.strip()
        return f"[{attr}]: {target.strip()}"
    return f"[ANONYMIZED]: {target.strip()}"


def generate_dpo_records(
    examples: List[AugmentedExample],
    rng: random.Random,
    format_messages: bool = True,
) -> List[Dict]:
    """Convert augmented examples into DPO preference records with strictly distinct chosen/rejected."""
    records: List[Dict] = []
    for ex in examples:
        if ex.position == "none" or not ex.injected:
            continue
        prompt = build_rewrite_messages(ex.input) if format_messages else ex.input
        chosen = ex.target.strip()

        negatives: List[tuple[str, str]] = []
        trunc = make_truncated_negative(chosen, ex.input)
        if trunc and trunc.strip() != chosen:
            negatives.append((trunc.strip(), "truncation_over_scrub"))
        leak = make_leaky_negative(ex)
        if leak and leak.strip() != chosen:
            negatives.append((leak.strip(), "pii_leak_under_scrub"))
        degen = make_degenerate_negative(chosen, ex)
        if degen and degen.strip() != chosen:
            negatives.append((degen.strip(), "syntax_placeholder_degenerate"))

        if not negatives:
            if ex.input.strip() != chosen:
                negatives.append((ex.input.strip(), "pii_leak_verbatim"))
            else:
                continue

        rejected, failure_mode = rng.choice(negatives)
        if rejected == chosen:
            continue

        records.append({
            "id": f"dpo_{ex.id}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "failure_mode": failure_mode,
                "position": ex.position,
                "attributes": ex.attributes,
                "source_carrier": ex.carrier_id,
            },
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs.")
    parser.add_argument("--n", type=int, default=1500, help="Number of seed augmented examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=str, default="data/dpo/train_dpo.jsonl", help="Output path.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Composing {args.n} balanced examples...")
    examples = compose_corpus(args.n, seed=args.seed)
    records = generate_dpo_records(examples, rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} verified DPO preference pairs to {out_path}")


if __name__ == "__main__":
    main()
