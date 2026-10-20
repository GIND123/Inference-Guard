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
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY, map_raw_to_taxonomy_label


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

        valid_pairs: List[tuple[str, str]] = []
        for p, t in zip(preds, trues):
            t_str = str(t or "").strip()
            if t_str not in valid_labels:
                t_str = map_raw_to_taxonomy_label(attr, t_str)
            if t_str in valid_labels:
                p_str = str(p or "").strip()
                if p_str not in valid_labels and p_str != "UNKNOWN":
                    p_str = map_raw_to_taxonomy_label(attr, p_str)
                valid_pairs.append((p_str, t_str))

        if not valid_pairs:
            per_attribute[attr.value] = {
                "attack_accuracy": 0.0,
                "macro_f1": 0.0,
                "abstention_rate": 0.0,
            }
            continue

        p_list, t_list = zip(*valid_pairs)
        acc = float(np.mean([1.0 if (p == t and p != "UNKNOWN") else 0.0 for p, t in valid_pairs]))
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
