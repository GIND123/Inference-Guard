# src/evaluation/multiturn.py
"""Multi-turn cumulative leakage curve harness for InferenceGuard.

Drives :class:`src.conversation.leakage.BayesianConversationLeakageTracker` over
cumulative 1/2/4/8-turn prefixes of a staged scenario (the S-06 staged protocol
in docs/experiment_protocol.md section 16) and emits a structured leakage curve.

CRITICAL: this module does NOT invent leakage numbers. When no trained risk
model / held-out adversary is loaded, the harness runs in the existing
``load_model=False`` fallback mode and every emitted curve is tagged
``reportable_as_held_out = False`` with ``attacker_mode`` recorded, mirroring the
convention in ``src/evaluation/run.py`` and ``src/evaluation/attacker.py``. The
only numbers in the output are whatever the leakage tracker actually computed
from the fallback adversary; they are explicitly non-reportable until the
north-star GPU run (P0 #1) produces a trained risk model + held-out adversary.

Turn counts evaluated are 1, 2, 4, 8, each clamped to the number of turns
actually available for a profile (the staged S-06 protocol has 4 stages, so with
the committed sample the 8-turn point collapses onto the 4-turn prefix and is
reported as such via ``turns_used``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.attacker import HeldOutPhi4Adversary

logger = logging.getLogger(__name__)

# Cumulative prefix lengths requested by backlog item #8.
DEFAULT_TURN_COUNTS: Sequence[int] = (1, 2, 4, 8)

_STAGED_TEST = Path("data/evaluation/staged_turns_test.jsonl")
DEFAULT_STAGED_INPUT = str(_STAGED_TEST) if _STAGED_TEST.exists() else "data/evaluation/staged_turns_sample.jsonl"


def _load_staged_records(input_file: Optional[str]) -> List[Dict[str, Any]]:
    """Read staged multi-turn records from committed JSONL input.

    Each record must provide ``profile_id``, ``turns`` (an ordered list of
    message strings) and optional ``attributes`` ground truth. The turns are the
    only source of conversation text; nothing is synthesised here.
    """
    path = Path(input_file) if input_file else Path(DEFAULT_STAGED_INPUT)
    records: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("Staged input %s not found; no multi-turn records loaded.", path)
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("turns"):
                records.append(rec)
    return records


def _build_turn_risk_report(adversary: "HeldOutPhi4Adversary", messages: Sequence[str]):
    """Build a per-turn ``RiskReport`` from the adversary's inferred attributes.

    The leakage tracker consumes an ``AttributeRiskDistribution`` per attribute.
    We derive a conservative distribution from the held-out adversary's top-1
    guess over the cumulative messages: probability mass concentrates on the
    predicted label (if any) and spreads uniformly otherwise. No metric values
    are hardcoded; everything flows from the adversary output, which in fallback
    mode is non-reportable by construction.
    """
    from src.contracts.schemas import (
        AttributeRiskDistribution,
        AttributeName,
        RiskBand,
        RiskReport,
    )
    from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY

    inferred = adversary.infer_attributes(list(messages))

    distributions: Dict[AttributeName, AttributeRiskDistribution] = {}
    for attr, labels in ATTRIBUTE_TAXONOMY.items():
        k = len(labels)
        predicted = inferred.get(attr, "UNKNOWN")
        if predicted in labels:
            # Concentrate a modest amount of mass on the guessed label. The
            # concentration is a fixed evidence prior, not a fabricated metric:
            # the reportable flag records that this came from a non-model run.
            hi = 0.60
            lo = (1.0 - hi) / max(k - 1, 1)
            class_probs = {lbl: (hi if lbl == predicted else lo) for lbl in labels}
            top1 = hi
            margin = round(hi - lo, 4)
            inferability = hi
            band = RiskBand.MEDIUM
            pred_value = predicted
        else:
            # No evidence: uniform distribution, zero inferability.
            uniform = 1.0 / k
            class_probs = {lbl: uniform for lbl in labels}
            top1 = uniform
            margin = 0.0
            inferability = 0.0
            band = RiskBand.LOW
            pred_value = "UNKNOWN"

        distributions[attr] = AttributeRiskDistribution(
            attribute=attr,
            predicted_value=pred_value,
            calibrated_confidence=round(top1, 4),
            top2_margin=margin,
            normalized_entropy=0.0,
            inferability_risk_score=round(inferability, 4),
            risk_band=band,
            class_probabilities=class_probs,
        )

    overall = max((d.inferability_risk_score for d in distributions.values()), default=0.0)
    primary = max(distributions, key=lambda a: distributions[a].inferability_risk_score, default=None)
    return RiskReport(
        attributes=distributions,
        overall_risk_score=round(overall, 4),
        overall_risk_band=RiskBand.MEDIUM if overall >= 0.30 else RiskBand.LOW,
        primary_risk_attribute=primary,
    )


def _make_tracker():
    """Construct a leakage tracker, or return ``None`` if heavy deps are absent.

    The tracker needs ``AttributeTemperatureScaler`` (torch/numpy). In the
    INTEGRATIONS_ONLY sandbox those are unavailable, so the import may fail; the
    caller then degrades to a structurally-valid but empty curve rather than
    fabricating leakage numbers.
    """
    try:
        from src.conversation.leakage import BayesianConversationLeakageTracker
        from src.risk_model.calibrate import AttributeTemperatureScaler

        return BayesianConversationLeakageTracker(AttributeTemperatureScaler())
    except Exception as exc:  # pragma: no cover - depends on optional heavy deps
        logger.warning(
            "Leakage tracker unavailable (%s); multi-turn curve will be marked "
            "unavailable rather than fabricated.",
            exc,
        )
        return None


def run_multiturn_leakage_curve(
    input_file: Optional[str] = None,
    turn_counts: Sequence[int] = DEFAULT_TURN_COUNTS,
    use_heavy_models: bool = False,
    output_dir: str = "reports",
    report_filename: str = "multiturn_leakage_curve.json",
    adversary: Optional["HeldOutPhi4Adversary"] = None,
) -> Dict[str, Any]:
    """Drive the leakage tracker over cumulative 1/2/4/8-turn prefixes.

    Returns a structured curve payload. All numbers originate from the leakage
    tracker fed by the held-out adversary; when the adversary and/or tracker run
    without a trained model the payload is tagged ``reportable_as_held_out =
    False`` and numbers are explicitly pending P0 #1.

    ``adversary`` lets the caller inject an already-constructed adversary so the
    multi-turn curve reuses the SAME held-out model that produced the per-sample
    benchmark rows instead of loading a second copy of the 7GB weights. When it
    is ``None`` the harness constructs its own, loading the real model only when
    ``use_heavy_models`` is set (strict, so a load failure surfaces rather than
    silently masquerading as a reportable run). The emitted ``attacker_mode`` /
    ``reportable_as_held_out`` always reflect the ACTUAL mode of the adversary
    that produced the curve: a heuristic/fallback adversary is never tagged
    reportable, so the note and the flags can no longer disagree.
    """
    from src.contracts.schemas import ConversationState

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = _load_staged_records(input_file)
    if adversary is None:
        adversary = HeldOutPhi4Adversary(
            load_model=use_heavy_models,
            strict=use_heavy_models,
        )
    attacker_mode = getattr(adversary, "mode", "heuristic")
    reportable = attacker_mode == "model"
    tracker = _make_tracker()

    requested = sorted({int(t) for t in turn_counts if int(t) >= 1})

    profiles_curves: List[Dict[str, Any]] = []
    for rec in records:
        turns: List[str] = list(rec.get("turns", []))
        if not turns:
            continue
        profile_id = rec.get("profile_id", "unknown")
        curve_points: List[Dict[str, Any]] = []

        if tracker is not None:
            # A fresh conversation state per requested prefix keeps each point an
            # independent cumulative measurement over the first N turns.
            for n in requested:
                turns_used = min(n, len(turns))
                prefix = turns[:turns_used]
                state = ConversationState(conversation_id=f"multiturn_{profile_id}_{n}")
                cumulative = 0.0
                per_attr: Dict[str, float] = {}
                for i in range(turns_used):
                    report = _build_turn_risk_report(adversary, prefix[: i + 1])
                    cumulative, _delta, per_attr_risk = tracker.compute_cumulative_leakage(
                        state, report, commit_update=True
                    )
                    per_attr = {a.value: round(float(v), 4) for a, v in per_attr_risk.items()}
                curve_points.append(
                    {
                        "requested_turns": n,
                        "turns_used": turns_used,
                        "cumulative_leakage": round(float(cumulative), 4),
                        "per_attribute_risk": per_attr,
                    }
                )

        profiles_curves.append(
            {
                "profile_id": profile_id,
                "scenario_id": rec.get("scenario_id"),
                "available_turns": len(turns),
                "ground_truth": rec.get("attributes", {}),
                "curve": curve_points,
            }
        )

    if reportable:
        note = (
            "Curve numbers come from the held-out Phi-4-mini adversary "
            "(attacker_mode='model') run over cumulative turn prefixes and are "
            "reportable as held-out results."
        )
    else:
        note = (
            "Curve numbers are NON-REPORTABLE: the held-out Phi-4-mini adversary "
            "resolved to attacker_mode='heuristic' (the real model did not load), "
            "so these values come from the load_model=False fallback and are "
            "pending the P0 #1 GPU run."
        )

    payload: Dict[str, Any] = {
        "harness": "multiturn_leakage_curve",
        "turn_counts": list(requested),
        "attacker_model_id": getattr(adversary, "model_name", "microsoft/Phi-4-mini-instruct"),
        "attacker_mode": attacker_mode,
        "reportable_as_held_out": reportable,
        "tracker_available": tracker is not None,
        "num_profiles": len(profiles_curves),
        "input_file": input_file or DEFAULT_STAGED_INPUT,
        "note": note,
        "profiles": profiles_curves,
    }

    report_file = out_path / report_filename
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Multi-turn leakage curve written to %s", report_file)

    return payload


def render_multiturn_curve(
    payload: Dict[str, Any],
    output_dir: str = "reports",
    png_filename: str = "multiturn_leakage_curve.png",
) -> Dict[str, Any]:
    """Render the 1/2/4/8-turn leakage curve to PNG when matplotlib is present.

    Degrades gracefully: the JSON payload is always returned/available; the PNG is
    only produced when matplotlib imports successfully (mirrors the guard in
    ``src.risk_model.calibrate.generate_reliability_diagram``). No placeholder
    numbers are ever emitted - only whatever the harness actually computed.
    """
    result = {"png_written": False, "png_path": None, "reason": None}
    profiles = payload.get("profiles", [])
    has_points = any(p.get("curve") for p in profiles)
    if not has_points:
        result["reason"] = "no curve points to plot (tracker unavailable in this environment)"
        return result

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        result["reason"] = f"matplotlib unavailable: {exc}"
        return result

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for prof in profiles:
        pts = prof.get("curve", [])
        if not pts:
            continue
        xs = [pt["turns_used"] for pt in pts]
        ys = [pt["cumulative_leakage"] for pt in pts]
        ax.plot(xs, ys, marker="o", label=str(prof.get("profile_id", "profile")))
    ax.set_xlabel("Cumulative turns disclosed")
    ax.set_ylabel("Cumulative conversation leakage")
    suffix = "" if payload.get("reportable_as_held_out") else " (NON-REPORTABLE: pending P0 #1)"
    ax.set_title("Multi-turn cumulative leakage curve" + suffix)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    png_path = out_path / png_filename
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    result["png_written"] = True
    result["png_path"] = str(png_path)
    return result
