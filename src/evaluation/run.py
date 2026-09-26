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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.evaluation.attacker import Phi4MiniAttacker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.utility import UtilityEvaluator
from src.rewriter.generate_training_data import generate_candidate_rewrites_heuristic, scrub_hard_pii

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
        processed: List[Dict[str, Any]] = []

        for sample in samples:
            text = sample.get("text", "")
            gt = sample.get("ground_truth", {})
            prof_id = sample.get("profile_id", "unknown")

            # Presidio explicit baseline
            redaction_res = presidio.analyze_sample(text)

            # Rewriter candidate generation
            candidates = generate_candidate_rewrites_heuristic(text)
            rewrite_text = candidates[0] if candidates else text

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
        attacker = Phi4MiniAttacker(load_model=self.use_heavy_models)

        results = {
            "original_probes": [],
            "presidio_probes": [],
            "rewrite_probes": [],
        }

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

                # Probe original
                orig_pred = attacker.predict_attribute(orig, attr)
                orig_succ = attacker.evaluate_success(orig_pred["prediction"], attr_gt)
                if orig_succ:
                    orig_success_count += 1
                    hardness_stats[hardness]["orig"] += 1

                # Probe presidio
                presidio_pred = attacker.predict_attribute(presidio_text, attr)
                presidio_succ = attacker.evaluate_success(presidio_pred["prediction"], attr_gt)
                if presidio_succ:
                    presidio_success_count += 1
                    hardness_stats[hardness]["presidio"] += 1

                # Probe rewrite
                rewrite_pred = attacker.predict_attribute(rew, attr)
                rewrite_succ = attacker.evaluate_success(rewrite_pred["prediction"], attr_gt)
                if rewrite_succ:
                    rewrite_success_count += 1
                    hardness_stats[hardness]["rewrite"] += 1

        del attacker
        clean_gpu_memory()

        asr_summary = {
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

        # Stage 1
        processed = self.stage_1_baseline_and_rewrite(samples)

        # Stage 2
        processed = self.stage_2_utility_evaluation(processed)

        # Stage 3
        adversary_summary = self.stage_3_adversary_probing(processed)

        # Aggregate utility
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged InferenceGuard evaluation harness")
    parser.add_argument("-i", "--input_file", type=str, help="Optional evaluation dataset JSONL")
    parser.add_argument("-o", "--output_dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    evaluator = StagedEvaluator(output_dir=args.output_dir)

    # Benchmark test set
    demo_samples = [
        {
            "profile_id": "pers1",
            "text": "I am 24 years old and work as a software engineer in Denver near Red Rocks.",
            "ground_truth": {"age": "24", "location": "Denver", "occupation": "Software Engineer"},
        },
        {
            "profile_id": "pers2",
            "text": "Taking sound transit to Pike Place after my clinical shift at the hospital.",
            "ground_truth": {"location": "Seattle", "occupation": "Nurse"},
        },
    ]

    report = evaluator.run_pipeline(demo_samples)
    print(json.dumps(report["adversary_evaluation"], indent=2))


if __name__ == "__main__":
    main()
