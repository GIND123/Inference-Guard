"""Multi-turn leakage on SynthPAI author conversations.

A per-message privacy check sees one message. A reader sees the conversation.
This measures the gap, on data where humans labelled what is inferable.

Conversation
    One author's comments within one thread, in split-file order (SynthPAI has
    no timestamps). Conversations with fewer than --min-turns are skipped.

Ground truth (human, never our model)
    An attribute is *exposed* at turn t if a SynthPAI annotator could infer it
    from any comment up to and including t. This is a lower bound: labels are
    per comment, so cues that only identify someone in combination are not
    counted.

What is reported, at each conversation's last turn
    1. data      how often an exposed attribute is exposed only by earlier
                 turns, so a check of the last message alone cannot see it
    2. detection for every (conversation, attribute):
                 recall on exposed pairs and false-alarm rate on unexposed
                 pairs, for the last-turn score vs the tracker's cumulative
                 score (both thresholded at the MEDIUM band edge)
    3. policy    over every turn: share handled (not passed through) by a
                 turn-only rule vs the cumulative-aware rule web/api.py now
                 uses, and how many are handled only because of it

The estimator is the rule-based fallback unless --risk-checkpoint points at a
trained ModernBERT; the output says which. Only train/val are accepted, the
test split is kept for the held-out attacker.

Usage:
    python scripts/eval_multiturn.py --split val
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conversation.state import ConversationTracker, contributing_attributes, estimate_turn_scores  # noqa: E402
from src.product.risk_bands import ATTRIBUTES, MEDIUM_EDGE  # noqa: E402
from src.rewriter.teacher_sft import inferable_attributes, load_split  # noqa: E402

Estimator = Callable[[str], Dict[str, float]]


def build_conversations(records: Sequence[Dict[str, Any]], min_turns: int = 2) -> List[List[Dict[str, Any]]]:
    """Group records by (author, thread), keeping file order."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(rec.get("profile_id") or rec.get("author"), rec.get("thread_id"))].append(rec)
    return [turns for turns in groups.values() if len(turns) >= min_turns]


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def evaluate(conversations: Sequence[Sequence[Dict[str, Any]]], estimator: Estimator) -> Dict[str, Any]:
    exposed_pairs = earlier_only = 0
    hit = {"last_turn": 0, "cumulative": 0}
    false_alarm = {"last_turn": 0, "cumulative": 0}
    unexposed_pairs = 0
    turns = handled_turn = handled_cumulative = handled_only_cumulative = 0
    per_attr = {a: {"exposed": 0, "earlier_only": 0} for a in ATTRIBUTES}

    for conv in conversations:
        tracker = ConversationTracker(risk_estimator_fn=estimator)
        exposed_so_far: set = set()
        record = None
        for rec in conv:
            record = tracker.add_turn(rec["text"], session_id="eval")
            exposed_so_far |= set(inferable_attributes(rec))
            turns += 1
            turn_flag = max(record.turn_scores.values()) >= MEDIUM_EDGE
            cum_flag = turn_flag or bool(contributing_attributes(record))
            handled_turn += turn_flag
            handled_cumulative += cum_flag
            handled_only_cumulative += cum_flag and not turn_flag

        last_inferable = set(inferable_attributes(conv[-1]))
        for attr in ATTRIBUTES:
            last_flag = record.turn_scores.get(attr, 0.0) >= MEDIUM_EDGE
            cum_flag = record.cumulative_scores.get(attr, 0.0) >= MEDIUM_EDGE
            if attr in exposed_so_far:
                exposed_pairs += 1
                per_attr[attr]["exposed"] += 1
                if attr not in last_inferable:
                    earlier_only += 1
                    per_attr[attr]["earlier_only"] += 1
                hit["last_turn"] += last_flag
                hit["cumulative"] += cum_flag
            else:
                unexposed_pairs += 1
                false_alarm["last_turn"] += last_flag
                false_alarm["cumulative"] += cum_flag

    return {
        "conversations": len(conversations),
        "turns": turns,
        "data": {
            "exposed_pairs": exposed_pairs,
            "exposed_only_by_earlier_turns": earlier_only,
            "share_invisible_to_last_message": _rate(earlier_only, exposed_pairs),
            "by_attribute": per_attr,
        },
        "detection": {
            "recall_last_turn": _rate(hit["last_turn"], exposed_pairs),
            "recall_cumulative": _rate(hit["cumulative"], exposed_pairs),
            "unexposed_pairs": unexposed_pairs,
            "false_alarm_last_turn": _rate(false_alarm["last_turn"], unexposed_pairs),
            "false_alarm_cumulative": _rate(false_alarm["cumulative"], unexposed_pairs),
        },
        "policy": {
            "handled_share_turn_only": _rate(handled_turn, turns),
            "handled_share_cumulative_aware": _rate(handled_cumulative, turns),
            "turns_handled_only_because_of_cumulative": handled_only_cumulative,
        },
    }


def load_estimator(checkpoint: Optional[str]) -> tuple:
    if not checkpoint:
        return estimate_turn_scores, "rule-based estimate_privacy_risk (no trained checkpoint)"
    import torch
    from transformers import AutoTokenizer

    from src.risk_model.model import ModernBertRiskClassifier
    from web.api import make_risk_estimator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernBertRiskClassifier.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    return make_risk_estimator(model, tokenizer, device), f"web.api.make_risk_estimator over {checkpoint}"


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--split-manifest", default="data/manifests/split_manifest.json")
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--risk-checkpoint", default=None)
    parser.add_argument("--out-dir", default="artifacts/multiturn")
    args = parser.parse_args(argv)

    records = load_split(args.split, ROOT / f"data/splits/{args.split}.jsonl", ROOT / args.split_manifest)
    conversations = build_conversations(records, args.min_turns)
    estimator, estimator_desc = load_estimator(args.risk_checkpoint)
    result = evaluate(conversations, estimator)
    result.update({
        "split": args.split,
        "min_turns": args.min_turns,
        "estimator": estimator_desc,
        "threshold": f"MEDIUM band edge {MEDIUM_EDGE}",
        "command": "python scripts/eval_multiturn.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    })

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "checkpoint" if args.risk_checkpoint else "rules"
    out = out_dir / f"{args.split}_{tag}_{dt.date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    d, det, pol = result["data"], result["detection"], result["policy"]
    print(f"{args.split}: {result['conversations']} conversations, {result['turns']} turns; estimator: {estimator_desc}")
    print(f"exposed (conversation, attribute) pairs: {d['exposed_pairs']}; "
          f"only visible through earlier turns: {d['exposed_only_by_earlier_turns']} "
          f"({d['share_invisible_to_last_message']})")
    print(f"recall    last-turn {det['recall_last_turn']}  cumulative {det['recall_cumulative']}")
    print(f"false alarm last-turn {det['false_alarm_last_turn']}  cumulative {det['false_alarm_cumulative']}")
    print(f"turns handled  turn-only {pol['handled_share_turn_only']}  cumulative-aware "
          f"{pol['handled_share_cumulative_aware']}  (+{pol['turns_handled_only_because_of_cumulative']})")
    print(f"wrote {out.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()
