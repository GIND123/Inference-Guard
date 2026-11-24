"""Staged Evaluation Runner for InferenceGuard.

Coordinates sequential execution across ModernBERT risk estimator, Presidio baseline,
Qwen rewriter, utility evaluators, and Phi-4-mini adversary.
Enforces explicit GPU memory cleanup (gc.collect + torch.cuda.empty_cache) between stages
to avoid VRAM exhaustion on 16 GB T4 instances.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.attacker import Phi4MiniAttacker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.utility import UtilityEvaluator

logger = logging.getLogger(__name__)

# Hosted model ids for the north-star held-out benchmark. The risk scorer
# defaults to the hosted ModernBERT checkpoint (configs/models.yaml
# risk_estimator.chosen = answerdotai/ModernBERT-base, hosted at the id below);
# load_risk_classifier auto-detects RoBERTa vs ModernBERT so the serving path
# (web/api.py / scripts/score_risk_model.py default to RoBERTa) is untouched.
DEFAULT_RISK_MODEL_ID = "AT707/inferenceguard-modernbert-risk"
DEFAULT_REWRITER_MODEL_ID = "AT707/inferenceguard-rewriter-v1"
DEFAULT_ADVERSARY_MODEL_ID = "microsoft/Phi-4-mini-instruct"


def clean_gpu_memory() -> None:
    """Explicitly clean GPU VRAM and run garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


class StagedEvaluator:
    """Orchestrates multi-stage evaluation pipeline respecting 16 GB VRAM limits."""

    def __init__(
        self,
        output_dir: str = "reports",
        use_heavy_models: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_heavy_models = use_heavy_models

    def stage_1_baseline_and_rewrite(
        self,
        samples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 1: Generate Presidio baseline and Privacy rewrites for all samples."""
        logger.info("Executing Stage 1: Baseline redaction and rewriter generation.")
        presidio = PresidioBaseline()

        logger.info("Loading Qwen Rewriter model...")
        from src.rewriter import DEFAULT_BASE_MODEL, QwenRewriterInference

        qwen_rewriter = QwenRewriterInference(
            model_name_or_path=DEFAULT_BASE_MODEL,
            adapter_path="artifacts/rewriter_qlora"
        )

        processed: List[Dict[str, Any]] = []

        for sample in samples:
            text = sample.get("text", "")
            gt = sample.get("ground_truth", {})
            prof_id = sample.get("profile_id", "unknown")

            redaction_res = presidio.analyze_sample(text)
            rewrite_text = qwen_rewriter.rewrite(text)

            processed.append({
                "profile_id": prof_id,
                "original": text,
                "presidio_redacted": redaction_res.redacted,
                "rewritten": rewrite_text,
                "hardness": redaction_res.hardness,
                "ground_truth": gt,
            })

        clean_gpu_memory()
        return processed

    def stage_2_utility_evaluation(
        self,
        processed_samples: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 2: Evaluate semantic similarity and NLI utility preservation."""
        logger.info("Executing Stage 2: Utility evaluation.")
        utility_eval = UtilityEvaluator(load_models=self.use_heavy_models)

        for item in processed_samples:
            orig = item["original"]
            rew = item["rewritten"]
            presidio_text = item["presidio_redacted"]

            item["utility_rewrite"] = utility_eval.compute_metrics(orig, rew)
            item["utility_presidio"] = utility_eval.compute_metrics(orig, presidio_text)

        del utility_eval
        clean_gpu_memory()
        return list(processed_samples)

    def stage_3_adversary_probing(
        self,
        processed_samples: Sequence[Dict[str, Any]],
        attributes: Sequence[str] = ("age", "location", "occupation", "education"),
    ) -> Dict[str, Any]:
        """Stage 3: Run independent Phi-4-mini adversary against Original, Presidio, and Rewrite."""
        logger.info("Executing Stage 3: Held-out adversary probing.")
        attacker = Phi4MiniAttacker(
            load_model=self.use_heavy_models,
            strict=self.use_heavy_models,
        )
        modes_seen = set()

        orig_success_count = 0
        presidio_success_count = 0
        rewrite_success_count = 0
        total_evaluations = 0

        hardness_stats = {
            "direct": {"orig": 0, "presidio": 0, "rewrite": 0, "total": 0},
            "indirect": {"orig": 0, "presidio": 0, "rewrite": 0, "total": 0},
            "complicated": {"orig": 0, "presidio": 0, "rewrite": 0, "total": 0},
        }

        for item in processed_samples:
            orig = item["original"]
            presidio_text = item["presidio_redacted"]
            rew = item["rewritten"]
            gt = item["ground_truth"]
            hardness = item.get("hardness", "direct")

            for attr in attributes:
                if attr not in gt:
                    continue

                attr_gt = str(gt[attr])
                total_evaluations += 1
                hardness_stats[hardness]["total"] += 1

                orig_pred = attacker.predict_attribute(orig, attr)
                orig_succ = attacker.evaluate_success(orig_pred["prediction"], attr_gt)
                if orig_succ:
                    orig_success_count += 1
                    hardness_stats[hardness]["orig"] += 1

                presidio_pred = attacker.predict_attribute(presidio_text, attr)
                presidio_succ = attacker.evaluate_success(presidio_pred["prediction"], attr_gt)
                if presidio_succ:
                    presidio_success_count += 1
                    hardness_stats[hardness]["presidio"] += 1

                rewrite_pred = attacker.predict_attribute(rew, attr)
                rewrite_succ = attacker.evaluate_success(rewrite_pred["prediction"], attr_gt)
                if rewrite_succ:
                    rewrite_success_count += 1
                    hardness_stats[hardness]["rewrite"] += 1

        attacker_model_id = getattr(attacker, "model_name", "microsoft/Phi-4-mini-instruct")
        if not modes_seen:
            modes_seen.add(getattr(attacker, "mode", "heuristic"))
        attacker_mode = modes_seen.pop() if len(modes_seen) == 1 else "mixed"

        del attacker
        clean_gpu_memory()

        asr_summary = {
            "attacker_model_id": attacker_model_id,
            "attacker_mode": attacker_mode,
            "reportable_as_held_out": attacker_mode == "model",
            "total_probes": total_evaluations,
            "original_asr": round(orig_success_count / max(total_evaluations, 1), 4),
            "presidio_asr": round(presidio_success_count / max(total_evaluations, 1), 4),
            "inferenceguard_asr": round(rewrite_success_count / max(total_evaluations, 1), 4),
            "hardness_breakdown": hardness_stats,
        }
        return asr_summary

    def run_pipeline(
        self,
        samples: Sequence[Dict[str, Any]],
        report_filename: str = "evaluation_report.json",
    ) -> Dict[str, Any]:
        """Execute all stages sequentially with memory cleanup and export final report."""
        logger.info(f"Starting staged evaluation on {len(samples)} samples.")

        processed = self.stage_1_baseline_and_rewrite(samples)
        processed = self.stage_2_utility_evaluation(processed)
        adversary_summary = self.stage_3_adversary_probing(processed)

        rewrite_cosines = [it["utility_rewrite"]["cosine_similarity"] for it in processed]
        rewrite_utilities = [it["utility_rewrite"]["utility_score"] for it in processed]
        presidio_cosines = [it["utility_presidio"]["cosine_similarity"] for it in processed]

        final_report = {
            "sample_count": len(samples),
            "adversary_evaluation": adversary_summary,
            "utility_summary": {
                "rewrite_mean_cosine": round(sum(rewrite_cosines) / max(len(rewrite_cosines), 1), 4),
                "rewrite_mean_utility": round(sum(rewrite_utilities) / max(len(rewrite_utilities), 1), 4),
                "presidio_mean_cosine": round(sum(presidio_cosines) / max(len(presidio_cosines), 1), 4),
            },
            "samples": processed,
        }

        report_path = self.output_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Staged evaluation report saved to {report_path}")
        return final_report


def run_comparative_benchmark(
    split: str = "val",
    methods: Sequence[str] = ("raw", "presidio", "prompt_baseline", "qlora_sft"),
    input_file: Optional[str] = None,
    output_dir: str = "reports",
    limit: Optional[int] = None,
    include_conversation_aware: bool = False,
    multiturn_input_file: Optional[str] = None,
    use_heavy_models: bool = False,
    risk_model_id: Optional[str] = None,
    rewriter_model_id: Optional[str] = None,
    adversary_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the comparative evaluation benchmark across privacy-preserving methods.

    The four per-sample methods (raw / presidio / prompt_baseline / qlora_sft)
    map to the design-matrix rows Raw / Presidio / Prompt-only / Fine-tuned. Set
    ``include_conversation_aware=True`` to additionally attach the
    Conversation-aware row (driven by the multi-turn leakage tracker) so a single
    command produces the full five-row matrix. No metric values are fabricated;
    the conversation-aware row carries the reportable flags from the multi-turn
    harness.

    Model wiring: ``risk_model_id`` / ``rewriter_model_id`` / ``adversary_model_id``
    default (in order) to the explicit argument, the corresponding environment
    variable (``RISK_MODEL_ID`` / ``REWRITER_MODEL_ID`` / ``ADVERSARY_MODEL_ID``),
    then the hosted ids (ModernBERT risk, rewriter v1, Phi-4-mini adversary).

    When ``use_heavy_models=True`` the held-out adversary, the rewriter, and the
    calibrated ModernBERT risk predictor are actually loaded, and the function
    RAISES if the adversary resolves to ``mode == 'heuristic'`` (i.e. the real
    model failed to load) rather than writing a report that could be mistaken for
    held-out results. When ``use_heavy_models=False`` the existing
    ``load_model=False`` fallback is kept and every payload is flagged
    ``reportable_as_held_out = False``. No metric values are ever fabricated.
    """
    from src.contracts.schemas import AttributeName, ConversationState, RewriteMode
    from src.conversation.entity_map import DeterministicPseudonymManager
    from src.evaluation.attacker import HeldOutPhi4Adversary
    from src.evaluation.privacy_metrics import compute_method_metrics
    from src.evaluation.utility_metrics import BidirectionalUtilityEvaluator
    from src.rewriter.inference import QwenPrivacyRewriterService

    risk_model_id = risk_model_id or os.getenv("RISK_MODEL_ID") or DEFAULT_RISK_MODEL_ID
    rewriter_model_id = (
        rewriter_model_id or os.getenv("REWRITER_MODEL_ID") or DEFAULT_REWRITER_MODEL_ID
    )
    adversary_model_id = (
        adversary_model_id or os.getenv("ADVERSARY_MODEL_ID") or DEFAULT_ADVERSARY_MODEL_ID
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    split_file = Path(input_file) if input_file else Path(f"data/splits/{split}.jsonl")
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if limit is not None and limit > 0:
            records = records[:limit]
    else:
        # High-coverage fallback samples conforming to SynthPAI scenarios
        records = [
            {
                "profile_id": "pers1",
                "text": "I take the Green Line to campus every morning for my undergraduate co-op in Boston.",
                "attributes": {"age": "18-24", "location": "us_northeast", "occupation": "student", "education": "college"},
            },
            {
                "profile_id": "pers2",
                "text": "Moving to Seattle, WA to start as a software engineer at Amazon, commuting via Sound Transit.",
                "attributes": {"age": "25-34", "location": "us_west", "occupation": "technology", "education": "college"},
            },
            {
                "profile_id": "pers3",
                "text": "Finished my clinical rotation at Massachusetts General Hospital before defending my PhD thesis.",
                "attributes": {"age": "25-34", "location": "us_northeast", "occupation": "healthcare", "education": "phd"},
            },
        ]

    pseudonym_mgr = DeterministicPseudonymManager()

    # Device safety gate (opt-in): under --use_heavy_models, refuse to run the
    # long held-out benchmark on CPU when IG_REQUIRE_CUDA is set truthy.
    if use_heavy_models and os.getenv("IG_REQUIRE_CUDA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "IG_REQUIRE_CUDA=1 but torch.cuda.is_available() is False; "
                "refusing to run the heavy held-out benchmark on CPU - fix the "
                "torch/CUDA install (or unset IG_REQUIRE_CUDA to allow CPU)."
            )

    from tqdm.auto import tqdm

    risk_model_loaded = False
    if use_heavy_models:
        from src.risk_model.predict import CalibratedRiskPredictor
        risk_predictor = CalibratedRiskPredictor.from_pretrained(risk_model_id)
        risk_model_loaded = risk_predictor is not None
        del risk_predictor
        clean_gpu_memory()

    # Pass 1: Text transformation (load rewriter alone, then free VRAM)
    if use_heavy_models:
        adapter_dir = os.getenv("REWRITER_ADAPTER_DIR", "artifacts/rewriter_qlora")
        rewriter = QwenPrivacyRewriterService(
            base_model_name=rewriter_model_id,
            adapter_dir=adapter_dir,
            load_model=True,
        )
    else:
        rewriter = QwenPrivacyRewriterService(base_model_name=rewriter_model_id)

    transformed_by_method: Dict[str, List[str]] = {}
    latencies_by_method: Dict[str, List[float]] = {}
    for method in methods:
        logger.info(f"Pass 1 (Transform) for method: {method} across {len(records)} samples")
        print(f"[InferenceGuard] Pass 1 (Transform): {method} ({len(records)} samples)", flush=True)
        texts_out: List[str] = []
        lats_out: List[float] = []
        for rec in tqdm(records, desc=f"Transform {method}", unit="sample"):
            orig_text = rec.get("text", "")
            state = ConversationState(conversation_id=f"eval_{method}")
            t0 = time.perf_counter()
            if method == "raw":
                transformed = orig_text
            elif method == "presidio":
                transformed, _, _ = pseudonym_mgr.pseudonymize_turn(orig_text, state)
            elif method == "prompt_baseline":
                pseudo, _, _ = pseudonym_mgr.pseudonymize_turn(orig_text, state)
                transformed, _ = rewriter.protect(pseudo, None, state, mode=RewriteMode.PROMPT_BASELINE)
            elif method == "qlora_sft":
                pseudo, _, _ = pseudonym_mgr.pseudonymize_turn(orig_text, state)
                transformed, _ = rewriter.protect(pseudo, None, state, mode=RewriteMode.BALANCED_SFT)
            else:
                transformed = orig_text
            lats_out.append((time.perf_counter() - t0) * 1000.0)
            texts_out.append(transformed)
        transformed_by_method[method] = texts_out
        latencies_by_method[method] = lats_out

    del rewriter
    clean_gpu_memory()

    # Pass 2: Utility evaluation (load NLI + embedder alone, then free VRAM)
    utility_eval = BidirectionalUtilityEvaluator(load_models=use_heavy_models)
    utility_by_method: Dict[str, List[Dict[str, float]]] = {}
    for method in methods:
        print(f"[InferenceGuard] Pass 2 (Utility): {method} ({len(records)} samples)", flush=True)
        u_scores: List[Dict[str, float]] = []
        for rec, transformed in zip(records, transformed_by_method[method]):
            util = utility_eval.evaluate_pair(rec.get("text", ""), transformed)
            u_scores.append({
                "bidirectional_entailment": util.bidirectional_entailment,
                "semantic_similarity": util.semantic_similarity,
                "contradiction_prob": util.contradiction_prob,
            })
        utility_by_method[method] = u_scores

    del utility_eval
    clean_gpu_memory()

    # Pass 3: Held-out adversarial inference (load Phi-4-mini alone)
    adversary = HeldOutPhi4Adversary(
        model_name=adversary_model_id,
        load_model=use_heavy_models,
        strict=use_heavy_models,
    )
    attacker_mode = getattr(adversary, "mode", "heuristic")
    if use_heavy_models and attacker_mode != "model":
        raise RuntimeError(
            "use_heavy_models=True but the held-out adversary "
            f"'{adversary_model_id}' resolved to mode='{attacker_mode}' "
            "(the real model did not load). Refusing to write a "
            "non-reportable report masquerading as a held-out run."
        )

    comparative_results: Dict[str, Any] = {}
    for method in methods:
        print(f"[InferenceGuard] Pass 3 (Adversary): {method} ({len(records)} samples)", flush=True)
        method_predictions: Dict[AttributeName, List[str]] = {attr: [] for attr in AttributeName}
        ground_truth: Dict[AttributeName, List[str]] = {attr: [] for attr in AttributeName}
        for rec, transformed in tqdm(
            zip(records, transformed_by_method[method]),
            total=len(records),
            desc=f"Adversary {method}",
            unit="sample",
        ):
            inferred = adversary.infer_attributes([transformed])
            attrs_gt = rec.get("attributes", {})
            for attr in AttributeName:
                method_predictions[attr].append(inferred.get(attr, "UNKNOWN"))
                ground_truth[attr].append(attrs_gt.get(attr.value, "UNKNOWN"))

        metrics = compute_method_metrics(
            records=records,
            predictions_by_attr=method_predictions,
            ground_truth_by_attr=ground_truth,
            utility_scores=utility_by_method[method],
            latencies_ms=latencies_by_method[method],
        )
        comparative_results[method] = metrics

    # Per-attribute attack-success matrix across methods and an ASR-reduction
    # block measuring each method against the "raw" baseline. Values are taken
    # verbatim from compute_method_metrics (per_attribute.attack_accuracy and
    # overall_attack_success_rate); nothing is fabricated.
    per_attribute_asr: Dict[str, Dict[str, float]] = {}
    for method, metrics in comparative_results.items():
        per_attribute_asr[method] = {
            attr: vals.get("attack_accuracy", 0.0)
            for attr, vals in metrics.get("per_attribute", {}).items()
        }

    asr_reduction: Dict[str, Any] = {}
    raw_metrics = comparative_results.get("raw")
    if raw_metrics is not None:
        raw_overall = raw_metrics.get("overall_attack_success_rate", 0.0)
        raw_per_attr = per_attribute_asr.get("raw", {})
        for method, metrics in comparative_results.items():
            if method == "raw":
                continue
            method_overall = metrics.get("overall_attack_success_rate", 0.0)
            per_attr_delta = {
                attr: round(raw_per_attr.get(attr, 0.0) - val, 4)
                for attr, val in per_attribute_asr.get(method, {}).items()
            }
            asr_reduction[method] = {
                "overall_asr_reduction_vs_raw": round(raw_overall - method_overall, 4),
                "per_attribute_asr_reduction_vs_raw": per_attr_delta,
            }

    reportable_as_held_out = attacker_mode == "model"
    benchmark_payload = {
        "split": split,
        "sample_count": len(records),
        # Held-out provenance: only mode=='model' may be reported as a held-out
        # adversary number. In the load_model=False fallback this is False.
        "attacker_model_id": adversary_model_id,
        "attacker_mode": attacker_mode,
        "reportable_as_held_out": reportable_as_held_out,
        "risk_model_id": risk_model_id,
        "rewriter_model_id": rewriter_model_id,
        "risk_model_loaded": risk_model_loaded,
        "methods": comparative_results,
        "per_attribute_attack_success": per_attribute_asr,
        "asr_reduction": asr_reduction,
    }
    if not reportable_as_held_out:
        benchmark_payload["note"] = (
            "attacker_mode='heuristic': the held-out Phi-4-mini adversary did not "
            "load, so these numbers are NON-REPORTABLE as held-out results and are "
            "pending the GPU run of P0 #1. Run with --use_heavy_models on a GPU "
            "host to produce a reportable benchmark."
        )

    # Conversation-aware matrix row: wire the multi-turn cumulative leakage
    # tracker result in so that all five design-matrix rows (Raw / Presidio /
    # Prompt-only / Fine-tuned / Conversation-aware) are produced by one command.
    # Numbers are whatever the leakage harness actually computed; in the
    # load_model=False fallback they are tagged non-reportable, never fabricated.
    if include_conversation_aware:
        try:
            from src.evaluation.multiturn import run_multiturn_leakage_curve

            mt_payload = run_multiturn_leakage_curve(
                input_file=multiturn_input_file,
                use_heavy_models=use_heavy_models,
                output_dir=output_dir,
                adversary=adversary,
            )
            benchmark_payload["conversation_aware"] = {
                "attacker_mode": mt_payload.get("attacker_mode"),
                "reportable_as_held_out": mt_payload.get("reportable_as_held_out", False),
                "tracker_available": mt_payload.get("tracker_available", False),
                "turn_counts": mt_payload.get("turn_counts", []),
                "note": mt_payload.get("note"),
                "multiturn_report": "multiturn_leakage_curve.json",
                "profiles": mt_payload.get("profiles", []),
            }
        except Exception as exc:  # pragma: no cover - heavy deps optional
            logger.warning("Conversation-aware row unavailable: %s", exc)
            benchmark_payload["conversation_aware"] = {
                "reportable_as_held_out": False,
                "tracker_available": False,
                "error": str(exc),
            }

    report_file = out_path / f"comparative_benchmark_{split}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    return benchmark_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged InferenceGuard evaluation harness")
    parser.add_argument("-i", "--input_file", type=str, help="Optional evaluation dataset JSONL")
    parser.add_argument("-o", "--output_dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--split", type=str, default="val", help="Data split to evaluate (val, test, train)")
    parser.add_argument("--methods", nargs="+", default=["raw", "presidio", "prompt_baseline", "qlora_sft"], help="Methods to compare")
    parser.add_argument("--limit", "--max_samples", type=int, default=None, dest="limit", help="Max samples to evaluate")
    parser.add_argument("--use_heavy_models", action="store_true", help="Load heavy models; fail instead of falling back")
    parser.add_argument(
        "--dataset",
        choices=["splits", "promptpai"],
        default="splits",
        help="Eval input source: committed splits (default) or the synthetic PromptPAI sample",
    )
    parser.add_argument(
        "--multiturn",
        action="store_true",
        help="Also emit the multi-turn 1/2/4/8-turn leakage curve and the Conversation-aware matrix row",
    )
    parser.add_argument(
        "--multiturn_input",
        type=str,
        default=None,
        help="Optional staged turns JSONL for the multi-turn curve (defaults to data/evaluation/staged_turns_sample.jsonl)",
    )
    parser.add_argument(
        "--matrix_markdown",
        action="store_true",
        help="Also write the per-method comparison matrix as a markdown table",
    )
    parser.add_argument(
        "--risk_model_id",
        type=str,
        default=None,
        help=(
            "Risk scorer id (defaults to $RISK_MODEL_ID or the hosted ModernBERT "
            f"'{DEFAULT_RISK_MODEL_ID}')"
        ),
    )
    parser.add_argument(
        "--rewriter_model_id",
        type=str,
        default=None,
        help=(
            "Rewriter id (defaults to $REWRITER_MODEL_ID or the hosted "
            f"'{DEFAULT_REWRITER_MODEL_ID}')"
        ),
    )
    parser.add_argument(
        "--adversary_model_id",
        type=str,
        default=None,
        help=(
            "Held-out adversary id (defaults to $ADVERSARY_MODEL_ID or "
            f"'{DEFAULT_ADVERSARY_MODEL_ID}')"
        ),
    )
    args = parser.parse_args()

    # --dataset promptpai selects the synthetic PromptPAI eval file as input,
    # unless an explicit --input_file was already given.
    input_file = args.input_file
    if args.dataset == "promptpai" and not input_file:
        full_ppai = Path("data/promptpai/promptpai.jsonl")
        input_file = str(full_ppai) if full_ppai.exists() else "data/promptpai/sample.jsonl"

    print(f"Executing comparative benchmark on split '{args.split}' (dataset={args.dataset}) for methods: {args.methods}")
    if args.use_heavy_models:
        print(
            "  Heavy-model (held-out) path: will FAIL rather than emit a "
            "non-reportable report if the adversary does not load."
        )
    results = run_comparative_benchmark(
        split=args.split,
        methods=args.methods,
        input_file=input_file,
        output_dir=args.output_dir,
        limit=args.limit,
        include_conversation_aware=args.multiturn,
        multiturn_input_file=args.multiturn_input,
        use_heavy_models=args.use_heavy_models,
        risk_model_id=args.risk_model_id,
        rewriter_model_id=args.rewriter_model_id,
        adversary_model_id=args.adversary_model_id,
    )
    print("\n" + "=" * 78)
    print(f"{'Method':<18} | {'Attacker Acc (v)':<16} | {'Abstain % (^)':<13} | {'Bi-NLI (^)':<11} | {'p95 Lat (ms)':<12}")
    print("-" * 78)
    for m, data in results["methods"].items():
        print(
            f"{m:<18} | {data['overall_attack_success_rate']:<16.4f} | "
            f"{data['overall_attacker_abstention_rate']:<13.4f} | "
            f"{data['mean_bidirectional_entailment']:<11.4f} | "
            f"{data['p95_latency_ms']:<12.2f}"
        )
    print("=" * 78)
    held_out = results.get("reportable_as_held_out", False)
    print(
        f"attacker_model_id={results.get('attacker_model_id')} "
        f"attacker_mode={results.get('attacker_mode')} "
        f"reportable_as_held_out={held_out}"
    )
    if not held_out:
        print(
            "  NOTE: this run is NON-REPORTABLE as a held-out result (pending the "
            "GPU run of P0 #1). Re-run with --use_heavy_models on a GPU host."
        )

    if args.multiturn:
        ca = results.get("conversation_aware", {})
        reportable = ca.get("reportable_as_held_out", False)
        print(
            "Conversation-aware row attached (attacker_mode="
            f"{ca.get('attacker_mode')}, reportable_as_held_out={reportable}). "
            "See reports/multiturn_leakage_curve.json."
        )
        if not reportable:
            print("  NOTE: conversation-aware leakage numbers are NON-REPORTABLE until P0 #1.")
        # Attempt to render the curve PNG; degrades gracefully if matplotlib absent.
        try:
            from src.evaluation.multiturn import render_multiturn_curve

            mt_report_path = Path(args.output_dir) / "multiturn_leakage_curve.json"
            if mt_report_path.exists():
                with open(mt_report_path, "r", encoding="utf-8") as f:
                    mt_payload = json.load(f)
                render_res = render_multiturn_curve(mt_payload, output_dir=args.output_dir)
                if render_res.get("png_written"):
                    print(f"  Curve PNG: {render_res['png_path']}")
                else:
                    print(f"  Curve PNG skipped: {render_res.get('reason')}")
        except Exception as exc:  # pragma: no cover
            logger.warning("Curve rendering skipped: %s", exc)

    if args.matrix_markdown:
        md_path = write_matrix_markdown(results, args.output_dir)
        print(f"Matrix markdown written to {md_path}")


def write_matrix_markdown(results: Dict[str, Any], output_dir: str) -> str:
    """Write the comparison matrix as a markdown table.

    Only values actually computed by the harness are written; the
    Conversation-aware row reports its reportable flag rather than a fabricated
    accuracy number.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    row_labels = {
        "raw": "Raw",
        "presidio": "Presidio",
        "prompt_baseline": "Prompt-only",
        "qlora_sft": "Fine-tuned",
    }
    # The per-method rows share the benchmark's held-out reportability: only a
    # mode=='model' run counts. The conversation-aware row carries its own flag
    # from the multi-turn payload.
    benchmark_reportable = results.get("reportable_as_held_out", False)
    asr_reduction = results.get("asr_reduction", {})
    lines = [
        f"# Comparison matrix (split={results.get('split')}, n={results.get('sample_count')})",
        "",
        f"- attacker_model_id: `{results.get('attacker_model_id')}`",
        f"- attacker_mode: `{results.get('attacker_mode')}`",
        f"- reportable_as_held_out: `{benchmark_reportable}`",
        f"- risk_model_id: `{results.get('risk_model_id')}`",
        f"- rewriter_model_id: `{results.get('rewriter_model_id')}`",
        "",
        "| Row | Attacker Acc | ASR reduction vs Raw | Abstain % | Bi-NLI | p95 Lat (ms) | Reportable |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method, data in results.get("methods", {}).items():
        label = row_labels.get(method, method)
        if method == "raw":
            reduction_cell = "-"
        else:
            reduction = asr_reduction.get(method, {}).get("overall_asr_reduction_vs_raw")
            reduction_cell = f"{reduction:.4f}" if reduction is not None else "-"
        lines.append(
            f"| {label} | {data['overall_attack_success_rate']:.4f} | "
            f"{reduction_cell} | "
            f"{data['overall_attacker_abstention_rate']:.4f} | "
            f"{data['mean_bidirectional_entailment']:.4f} | "
            f"{data['p95_latency_ms']:.2f} | {benchmark_reportable} |"
        )
    ca = results.get("conversation_aware")
    if ca is not None:
        reportable = ca.get("reportable_as_held_out", False)
        status = "pending P0 #1 (non-reportable)" if not reportable else "reportable"
        lines.append(f"| Conversation-aware | {status} | - | - | - | - | {reportable} |")
    lines.append("")
    if not benchmark_reportable:
        lines.append(
            "> Numbers are pending the north-star GPU run (P0 #1: trained ModernBERT "
            "risk model + LoRA adapter + held-out Phi-4-mini adversary). Rows produced "
            "in the load_model=False fallback (attacker_mode != 'model') are NOT "
            "reportable as held-out results."
        )
    else:
        lines.append(
            "> Held-out run: attacker_mode='model' / reportable_as_held_out=true. "
            "Numbers are the real held-out Phi-4-mini adversary results."
        )
    md_path = out_path / f"comparison_matrix_{results.get('split')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(md_path)


if __name__ == "__main__":
    main()
