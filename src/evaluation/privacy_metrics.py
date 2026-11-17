# src/evaluation/privacy_metrics.py
"""Comparative privacy evaluation metrics and abstention-aware attacker scoring.

Tracks attacker accuracy, macro F1, and explicit attacker abstention rate (UNKNOWN %)
alongside semantic utility preservation and latency benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import f1_score

from src.contracts.schemas import AttributeName
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY, normalize_to_taxonomy_label


def compute_method_metrics(
    records: List[Dict[str, Any]],
    predictions_by_attr: Dict[AttributeName, List[str]],
    ground_truth_by_attr: Dict[AttributeName, List[str]],
    utility_scores: List[Dict[str, float]],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    """Compute comprehensive comparative benchmark metrics for a privacy-preserving method."""
    per_attribute: Dict[str, Dict[str, float]] = {}
    attack_accs: List[float] = []
    macro_f1s: List[float] = []
    abstention_rates: List[float] = []

    for attr, valid_labels in ATTRIBUTE_TAXONOMY.items():
        preds = predictions_by_attr.get(attr, [])
        trues = ground_truth_by_attr.get(attr, [])

        # Ground truth arrives as free text ("41", "Bangkok, Thailand",
        # "Masters in Business Administration"), but predictions are coarse
        # taxonomy labels ("35-44", "asia_pacific", "master"). Collapse the
        # ground truth into the same taxonomy label space before comparing, so a
        # correct coarse inference is actually counted. Without this the
        # membership filter below (t in valid_labels) dropped every pair, which
        # zeroed attack accuracy for every method including raw. Values already
        # equal to a taxonomy label (or "UNKNOWN") pass through unchanged.
        trues = [
            t if t in valid_labels or t == "UNKNOWN" else normalize_to_taxonomy_label(attr, t)
            for t in trues
        ]

        valid_pairs = [(p, t) for p, t in zip(preds, trues) if t in valid_labels]
        if not valid_pairs:
            continue

        p_list, t_list = zip(*valid_pairs)
        acc = float(np.mean([1.0 if p == t else 0.0 for p, t in valid_pairs]))
        abstention = float(np.mean([1.0 if p == "UNKNOWN" else 0.0 for p in p_list]))
        mac_f1 = float(f1_score(t_list, p_list, labels=valid_labels, average="macro", zero_division=0))

        per_attribute[attr.value] = {
            "attack_accuracy": round(acc, 4),
            "macro_f1": round(mac_f1, 4),
            "abstention_rate": round(abstention, 4),
        }
        attack_accs.append(acc)
        macro_f1s.append(mac_f1)
        abstention_rates.append(abstention)

    mean_bi_entail = (
        float(np.mean([u.get("bidirectional_entailment", 1.0) for u in utility_scores]))
        if utility_scores
        else 1.0
    )
    mean_sem_sim = (
        float(np.mean([u.get("semantic_similarity", 1.0) for u in utility_scores]))
        if utility_scores
        else 1.0
    )
    contra_rate = (
        float(np.mean([1.0 if u.get("contradiction_prob", 0.0) > 0.20 else 0.0 for u in utility_scores]))
        if utility_scores
        else 0.0
    )
    p95_lat = (
        float(np.percentile(latencies_ms, 95))
        if latencies_ms
        else 0.0
    )

    return {
        "overall_attack_success_rate": round(float(np.mean(attack_accs)), 4) if attack_accs else 0.0,
        "overall_attacker_macro_f1": round(float(np.mean(macro_f1s)), 4) if macro_f1s else 0.0,
        "overall_attacker_abstention_rate": round(float(np.mean(abstention_rates)), 4) if abstention_rates else 0.0,
        "per_attribute": per_attribute,
        "mean_bidirectional_entailment": round(mean_bi_entail, 4),
        "mean_semantic_similarity": round(mean_sem_sim, 4),
        "contradiction_rate": round(contra_rate, 4),
        "p95_latency_ms": round(p95_lat, 2),
    }
