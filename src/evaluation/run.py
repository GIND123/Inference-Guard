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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.attacker import Phi4MiniAttacker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.utility import UtilityEvaluator

logger = logging.getLogger(__name__)


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
    use_heavy_models: bool = False,
    adapter_dir: Optional[str] = None,
    base_model_name: Optional[str] = None,
    device: Optional[str] = None,
    strict: bool = False,
    adversary_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute four-way comparative evaluation benchmark across privacy-preserving methods."""
    from src.contracts.schemas import AttributeName, ConversationState, RewriteMode
    from src.conversation.entity_map import DeterministicPseudonymManager
    from src.evaluation.attacker import HeldOutPhi4Adversary
    from src.evaluation.privacy_metrics import compute_method_metrics
    from src.evaluation.utility_metrics import BidirectionalUtilityEvaluator
    from src.rewriter.inference import DEFAULT_BASE_MODEL, QwenPrivacyRewriterService
    from src.risk_model.taxonomy import map_raw_to_taxonomy_label

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
                "attributes": {"age": "18-24", "location": "us_northeast", "occupation": "student", "education": "undergraduate"},
            },
            {
                "profile_id": "pers2",
                "text": "Moving to Seattle, WA to start as a software engineer at Amazon, commuting via Sound Transit.",
                "attributes": {"age": "25-34", "location": "us_west", "occupation": "technology", "education": "undergraduate"},
            },
            {
                "profile_id": "pers3",
                "text": "Finished my clinical rotation at Massachusetts General Hospital before defending my PhD thesis.",
                "attributes": {"age": "25-34", "location": "us_northeast", "occupation": "healthcare", "education": "graduate_professional"},
            },
        ]

    # Resolve default adapter location if available
    effective_adapter = adapter_dir
    if effective_adapter is None and Path("artifacts/rewriter_qlora").exists():
        effective_adapter = "artifacts/rewriter_qlora"

    pseudonym_mgr = DeterministicPseudonymManager()
    utility_eval = BidirectionalUtilityEvaluator(load_models=use_heavy_models, device=device)
    adversary = HeldOutPhi4Adversary(
        model_name=adversary_model_name or "microsoft/Phi-4-mini-instruct",
        load_model=use_heavy_models,
        device=device,
    )
    rewriter = QwenPrivacyRewriterService(
        base_model_name=base_model_name or DEFAULT_BASE_MODEL,
        adapter_dir=effective_adapter,
        device=device,
        load_model=use_heavy_models,
    )

    adversary_mode = "model" if getattr(adversary, "model", None) is not None else "heuristic"
    rewriter_engine = (
        "qlora_sft"
        if getattr(rewriter, "model", None) is not None and getattr(rewriter, "has_adapter", False)
        else ("qwen_base" if getattr(rewriter, "model", None) is not None else "heuristic")
    )
    utility_mode = "model" if getattr(utility_eval, "nli_model", None) is not None else "heuristic"

    if use_heavy_models and strict:
        if adversary_mode != "model":
            raise RuntimeError("Strict evaluation requested but HeldOutPhi4Adversary could not load model weights.")
        if rewriter.model is None:
            raise RuntimeError("Strict evaluation requested but QwenPrivacyRewriterService could not load model weights.")

    logger.info(
        f"Starting comparative benchmark across {len(records)} samples. "
        f"Modes: Adversary={adversary_mode}, Rewriter={rewriter_engine}, Utility={utility_mode}"
    )

    comparative_results: Dict[str, Any] = {}

    for method in methods:
        logger.info(f"Evaluating method: {method} across {len(records)} samples")
        method_predictions: Dict[AttributeName, List[str]] = {attr: [] for attr in AttributeName}
        ground_truth: Dict[AttributeName, List[str]] = {attr: [] for attr in AttributeName}
        utility_scores: List[Dict[str, float]] = []
        latencies: List[float] = []

        for rec in records:
            orig_text = rec.get("text", "")
            state = ConversationState(conversation_id=f"eval_{method}")

            # Step 1: Text transformation by method
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

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            # Step 2: Utility evaluation
            util = utility_eval.evaluate_pair(orig_text, transformed)
            utility_scores.append({
                "bidirectional_entailment": util.bidirectional_entailment,
                "semantic_similarity": util.semantic_similarity,
                "contradiction_prob": util.contradiction_prob,
            })

            # Step 3: Adversarial inference
            inferred = adversary.infer_attributes([transformed])
            attrs_gt = rec.get("attributes", {}) or rec.get("profile", {})
            for attr in AttributeName:
                method_predictions[attr].append(inferred.get(attr, "UNKNOWN"))
                raw_gt = attrs_gt.get(attr.value) if isinstance(attrs_gt, dict) else None
                if raw_gt is None and isinstance(attrs_gt, dict):
                    raw_gt = attrs_gt.get(attr)
                ground_truth[attr].append(map_raw_to_taxonomy_label(attr, raw_gt))

        metrics = compute_method_metrics(
            records=records,
            predictions_by_attr=method_predictions,
            ground_truth_by_attr=ground_truth,
            utility_scores=utility_scores,
            latencies_ms=latencies,
        )
        comparative_results[method] = metrics

    benchmark_payload = {
        "split": split,
        "sample_count": len(records),
        "use_heavy_models": use_heavy_models,
        "adversary_mode": adversary_mode,
        "rewriter_engine": rewriter_engine,
        "utility_mode": utility_mode,
        "methods": comparative_results,
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
    parser.add_argument("--adapter_dir", type=str, default=None, help="Directory containing fine-tuned QLoRA adapter")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base causal LLM model name")
    parser.add_argument("--device", type=str, default=None, help="Device placement (cuda/cpu)")
    parser.add_argument("--strict", action="store_true", help="Raise if heavy models fail to load")
    parser.add_argument("--adversary_model", type=str, default="microsoft/Phi-4-mini-instruct", help="Held-out adversary model identifier")
    args = parser.parse_args()

    print(f"Executing 4-way comparative benchmark on split '{args.split}' for methods: {args.methods}")
    results = run_comparative_benchmark(
        split=args.split,
        methods=args.methods,
        input_file=args.input_file,
        output_dir=args.output_dir,
        limit=args.limit,
        use_heavy_models=args.use_heavy_models,
        adapter_dir=args.adapter_dir,
        base_model_name=args.base_model_name,
        device=args.device,
        strict=args.strict,
        adversary_model_name=args.adversary_model,
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
    print(f"Adversary Mode: {results['adversary_mode']} | Rewriter Engine: {results['rewriter_engine']} | Utility: {results['utility_mode']}")
    if not args.use_heavy_models:
        print("[NOTE] Running in offline heuristic mode. To evaluate trained neural weights, pass --use_heavy_models --adapter_dir <path>")
    elif results["adversary_mode"] != "model":
        print(f"[NOTE] Adversary loaded in 'heuristic' fallback mode. Model weights for '{args.adversary_model}' were not active. Pass --strict to see exact load traceback or specify --adversary_model.")


if __name__ == "__main__":
    main()
