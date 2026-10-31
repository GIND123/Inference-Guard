"""Train/serve prompt parity and memorization guard for rewriter SFT files.

Run after building any SFT dataset and before training:

    python scripts/check_prompt_parity.py data/sft/teacher_v1/train.jsonl

Fails (exit 1) when any of the root causes of the v1 generalization failure
reappear:

  1. PROMPT DRIFT - a row whose messages are not exactly
     [system, user(instruction + text)] from src.rewriter.prompting
     (the committed v1 file predates the current builder: 1749/1749 rows
     mismatched the served prompt, so the adapter was served prompts it had
     never seen in training).

  2. TURN DUPLICATION - any (user, assistant) turn appearing in more than
     ``--dup-threshold`` of rows. The v1 file embedded one fixed two-shot
     primer in every row: 3,498 of 5,247 supervised turns (66.7%) were the
     same two completions, each receiving 1,749 gradient updates - the
     strongest memorization signal in the dataset.

Also reports the position distribution of the first changed token (v1 was 69%
start / 21% middle / 10% end - a positional prior instead of cue detection),
input length stats, and per-attribute coverage, so distribution regressions
are visible before training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rewriter.prompting import REWRITER_SYSTEM_PROMPT, REWRITE_INSTRUCTION, build_rewrite_messages


def _extract_user_text(user_content: str) -> Optional[str]:
    """Recover the raw text from a REWRITE_INSTRUCTION-framed user turn."""
    if not user_content.startswith(REWRITE_INSTRUCTION):
        return None
    return user_content[len(REWRITE_INSTRUCTION):]


def _first_change_position(original: str, rewrite: str) -> Optional[str]:
    wa, wb = original.split(), rewrite.split()
    i = 0
    while i < min(len(wa), len(wb)) and wa[i] == wb[i]:
        i += 1
    if i >= len(wa) and i >= len(wb):
        return None  # identical
    frac = i / max(len(wa), 1)
    return "start" if frac < 0.34 else ("middle" if frac < 0.67 else "end")


def check_file(path: Path, dup_threshold: float = 0.01) -> Dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    problems: List[str] = []
    turn_counts: Counter = Counter()
    positions: Counter = Counter()
    cue_positions: Counter = Counter()  # where the cue actually sat (from metadata)
    attr_counts: Counter = Counter()
    lengths: List[int] = []
    noop = 0

    for i, row in enumerate(rows):
        messages = row.get("messages", [])

        # --- turn statistics (collected for every row, parity or not) --------
        for j in range(1, len(messages) - 1, 2):
            if messages[j].get("role") == "user" and messages[j + 1].get("role") == "assistant":
                turn_counts[(messages[j]["content"], messages[j + 1]["content"])] += 1
        user_text = _extract_user_text(messages[1]["content"]) if len(messages) >= 3 else None
        if user_text is not None and messages[-1].get("role") == "assistant":
            original, rewrite = user_text, messages[-1]["content"]
            lengths.append(len(original.split()))
            if rewrite.strip() == original.strip():
                noop += 1
            else:
                pos = _first_change_position(original, rewrite)
                if pos:
                    positions[pos] += 1
        meta_position = row.get("metadata", {}).get("position")
        if meta_position:
            cue_positions[meta_position] += 1
        for a in row.get("metadata", {}).get("attributes", []):
            attr_counts[a] += 1

        # --- check 1: prompt parity with the served prompt ------------------
        if len(messages) < 3:
            problems.append(f"row {i}: fewer than 3 messages (system/user/assistant)")
            continue
        expected = build_rewrite_messages(user_text) if user_text is not None else None
        if (
            messages[0]["role"] != "system"
            or messages[0]["content"] != REWRITER_SYSTEM_PROMPT
            or messages[1]["role"] != "user"
            or expected is None
            or [m["content"] for m in messages[:2]] != [m["content"] for m in expected]
        ):
            problems.append(
                f"row {i}: prompt does not match build_rewrite_messages - "
                "train/serve drift (the served adapter would see an unseen prompt)"
            )
            continue
        if len(messages) != 3:
            problems.append(
                f"row {i}: {len(messages)} messages, expected 3 (single-turn). "
                "Multi-turn rows with a fixed primer caused 66.7% duplicated "
                "supervised turns in the v1 dataset."
            )
            continue
        if messages[2]["role"] != "assistant":
            problems.append(f"row {i}: last message is not the assistant target")

    # --- check 2: turn duplication ------------------------------------------
    n = len(rows)
    duplicated = [(t, c) for t, c in turn_counts.most_common() if c > max(1, dup_threshold * n)]
    for (user, assistant), count in duplicated:
        snippet = user[-90:].replace("\n", " ")
        problems.append(
            f"DUPLICATED TURN x{count} ({count / n:.1%} of rows): ...{snippet!r} -> {assistant[:50]!r}"
        )
    lengths.sort()
    stats = {
        "file": str(path),
        "rows": n,
        "prompt_parity_violations": sum(1 for p in problems if "drift" in p or "messages" in p),
        "duplication_violations": len(duplicated),
        "problems": problems[:20],
        "identity_noops": noop,
        "noop_ratio": round(noop / n, 3) if n else None,
        "cue_position_distribution": dict(cue_positions),
        "first_change_position": dict(positions),
        "position_shares": {
            k: round(v / max(1, sum(positions.values())), 3) for k, v in positions.items()
        },
        "input_words": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "attribute_coverage": dict(attr_counts.most_common()),
    }
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sft_files", nargs="+", type=Path, help="SFT JSONL files to check")
    parser.add_argument("--dup-threshold", type=float, default=0.01,
                        help="Max fraction of rows that may share one (user, assistant) turn")
    args = parser.parse_args(argv)

    failed = False
    for path in args.sft_files:
        stats = check_file(path, dup_threshold=args.dup_threshold)
        print(json.dumps(stats, indent=2))
        if stats["problems"]:
            failed = True
            print(f"\nFAIL {path}: {len(stats['problems'])} problem(s) listed above", file=sys.stderr)
        else:
            print(f"OK {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
