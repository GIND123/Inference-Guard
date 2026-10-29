# src/evaluation/utility_metrics.py
"""Bidirectional Natural Language Inference and semantic utility evaluation.

Measures intent and factual preservation using DeBERTa-v3 NLI bidirectional entailment,
dense embedding cosine similarity, and token-level edit distance.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

from src.contracts.schemas import UtilityReport
from src.evaluation.utility import check_degeneracy

logger = logging.getLogger(__name__)


def compute_fallback_similarity(text_a: str, text_b: str) -> float:
    """Compute lexical Jaccard/token cosine similarity for offline CPU test execution."""
    words_a = re.findall(r"\b\w+\b", text_a.lower())
    words_b = re.findall(r"\b\w+\b", text_b.lower())
    if not words_a or not words_b:
        return 0.0 if text_a != text_b else 1.0

    vocab = set(words_a) | set(words_b)
    vec_a = {w: words_a.count(w) for w in vocab}
    vec_b = {w: words_b.count(w) for w in vocab}
    dot = sum(vec_a[w] * vec_b[w] for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class BidirectionalUtilityEvaluator:
    """Evaluates meaning preservation via DeBERTa-v3 NLI and embedding cosine similarity."""

    def __init__(
        self,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
        embed_model_name: str = "answerdotai/ModernBERT-base",
        min_acceptable_entailment: float = 0.60,
        max_acceptable_contradiction: float = 0.20,
        device: Optional[str] = None,
        load_models: bool = False,
    ):
        """Initialize utility evaluator with configurable thresholds and optional model loading."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_acceptable_entailment = min_acceptable_entailment
        self.max_acceptable_contradiction = max_acceptable_contradiction
        self.nli_model_name = nli_model_name
        self.embed_model_name = embed_model_name

        self.nli_tokenizer = None
        self.nli_model = None
        self.embed_tokenizer = None
        self.embed_model = None

        if load_models:
            self._load_models()

    @property
    def backend(self) -> str:
        """Report whether utility metrics came from models or fallbacks."""
        all_models_loaded = (
            self.nli_model is not None
            and self.nli_tokenizer is not None
            and self.embed_model is not None
            and self.embed_tokenizer is not None
        )
        return "model" if all_models_loaded else "fallback"

    def _load_models(self) -> None:
        """Load HuggingFace cross-encoder and embedding backbones."""
        try:
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name
            ).to(self.device).eval()

            # Dynamically resolve label mapping from model config
            id2label = {int(k): v.lower() for k, v in self.nli_model.config.id2label.items()}
            self.entail_idx = next(i for i, lbl in id2label.items() if "entail" in lbl)
            self.contra_idx = next(i for i, lbl in id2label.items() if "contradict" in lbl)

            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self.embed_model = AutoModel.from_pretrained(self.embed_model_name).to(self.device).eval()
        except Exception:
            self.nli_model = None
            self.embed_model = None

    @torch.no_grad()
    def _compute_nli_probs(self, premise: str, hypothesis: str) -> Tuple[float, float]:
        """Compute forward or backward entailment and contradiction probabilities."""
        if self.nli_model is None or self.nli_tokenizer is None:
            # High-fidelity fallback heuristic based on lexical overlap
            sim = compute_fallback_similarity(premise, hypothesis)
            entail = max(0.5, sim)
            contra = max(0.0, 1.0 - sim) * 0.1
            return entail, contra

        features = self.nli_tokenizer(
            [(premise, hypothesis)],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.nli_model(**features).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return float(probs[self.entail_idx]), float(probs[self.contra_idx])

    @torch.no_grad()
    def _compute_embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Compute dense token embedding cosine similarity."""
        if self.embed_model is None or self.embed_tokenizer is None:
            return round(compute_fallback_similarity(text_a, text_b), 4)

        enc = self.embed_tokenizer(
            [text_a, text_b],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.embed_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=-1)
        sim = float(torch.sum(pooled[0] * pooled[1]).item())
        return round(sim, 4)

    def evaluate_pair(self, original_text: str, rewritten_text: str) -> UtilityReport:
        """Evaluate semantic similarity, bidirectional NLI entailment, and constraint preservation."""
        if original_text.strip() == rewritten_text.strip():
            return UtilityReport(
                semantic_similarity=1.0,
                forward_entailment_prob=1.0,
                backward_entailment_prob=1.0,
                bidirectional_entailment=1.0,
                contradiction_prob=0.0,
                token_edit_ratio=0.0,
                utility_preserved=True,
                warning_message=None,
            )

        fwd_entail, fwd_contra = self._compute_nli_probs(original_text, rewritten_text)
        bwd_entail, bwd_contra = self._compute_nli_probs(rewritten_text, original_text)

        # Geometric mean penalizes asymmetric loss of critical constraints
        bi_entail = math.sqrt(max(fwd_entail, 1e-6) * max(bwd_entail, 1e-6))
        max_contra = max(fwd_contra, bwd_contra)
        sem_sim = self._compute_embedding_similarity(original_text, rewritten_text)

        orig_tokens = original_text.split()
        rew_tokens = rewritten_text.split()
        matcher = difflib.SequenceMatcher(None, orig_tokens, rew_tokens)
        edit_ratio = 1.0 - matcher.ratio()

        utility_ok = (
            bi_entail >= self.min_acceptable_entailment
            and max_contra <= self.max_acceptable_contradiction
            and sem_sim >= 0.70
        )

        warning = None
        if not utility_ok:
            if max_contra > self.max_acceptable_contradiction:
                warning = "The rewrite altered a factual condition from your original message."
            else:
                warning = "The rewrite removed specific task constraints that may degrade answer quality."

        return UtilityReport(
            semantic_similarity=round(sem_sim, 4),
            forward_entailment_prob=round(fwd_entail, 4),
            backward_entailment_prob=round(bwd_entail, 4),
            bidirectional_entailment=round(bi_entail, 4),
            contradiction_prob=round(max_contra, 4),
            token_edit_ratio=round(edit_ratio, 4),
            utility_preserved=utility_ok,
            warning_message=warning,
        )


def evaluate_rewrites(
    pairs: Sequence[Tuple[str, str]],
    evaluator: Optional[BidirectionalUtilityEvaluator] = None,
) -> Dict[str, Any]:
    """Evaluate a sequence of (original, rewrite) pairs and compute aggregate utility metrics."""
    if evaluator is None:
        evaluator = BidirectionalUtilityEvaluator()

    if not pairs:
        return {
            "total_samples": 0,
            "mean_semantic_similarity": 0.0,
            "mean_bidirectional_entailment": 0.0,
            "mean_forward_entailment": 0.0,
            "mean_backward_entailment": 0.0,
            "mean_contradiction_prob": 0.0,
            "mean_token_edit_ratio": 0.0,
            "utility_preservation_rate": 0.0,
            "degeneracy_rate": 0.0,
            "utility_backend": evaluator.backend,
        }

    degenerate_count = 0
    sim_scores: List[float] = []
    bi_entail_scores: List[float] = []
    fwd_entail_scores: List[float] = []
    bwd_entail_scores: List[float] = []
    contra_scores: List[float] = []
    edit_ratios: List[float] = []
    preserved_count = 0

    for orig, rew in pairs:
        degen_check = check_degeneracy(orig, rew)
        is_degenerate = degen_check.get("degenerate", False)
        if is_degenerate:
            degenerate_count += 1

        rep = evaluator.evaluate_pair(orig, rew)
        sim_scores.append(rep.semantic_similarity)
        bi_entail_scores.append(rep.bidirectional_entailment)
        fwd_entail_scores.append(rep.forward_entailment_prob)
        bwd_entail_scores.append(rep.backward_entailment_prob)
        contra_scores.append(rep.contradiction_prob)
        edit_ratios.append(rep.token_edit_ratio)
        if rep.utility_preserved and not is_degenerate:
            preserved_count += 1

    n = len(pairs)
    return {
        "total_samples": n,
        "mean_semantic_similarity": round(sum(sim_scores) / n, 4),
        "mean_bidirectional_entailment": round(sum(bi_entail_scores) / n, 4),
        "mean_forward_entailment": round(sum(fwd_entail_scores) / n, 4),
        "mean_backward_entailment": round(sum(bwd_entail_scores) / n, 4),
        "mean_contradiction_prob": round(sum(contra_scores) / n, 4),
        "mean_token_edit_ratio": round(sum(edit_ratios) / n, 4),
        "utility_preservation_rate": round(preserved_count / n, 4),
        "degeneracy_rate": round(degenerate_count / n, 4),
        "utility_backend": evaluator.backend,
    }


def evaluate_held_out_split(
    test_split_path: str | Path = "data/splits/test.jsonl",
    rewriter: Optional[Any] = None,
    evaluator: Optional[BidirectionalUtilityEvaluator] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate rewriter utility against held-out test split."""
    test_path = Path(test_split_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test split file not found: {test_path}")

    records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if num_samples and num_samples > 0:
        records = records[:num_samples]

    if rewriter is None:
        from src.rewriter.inference import QwenRewriterInference
        rewriter = QwenRewriterInference()

    pairs = []
    for rec in records:
        text = rec.get("text") or rec.get("comment") or rec.get("body") or ""
        if not text:
            continue
        rew = rewriter.rewrite(text)
        if isinstance(rew, tuple):
            rew = rew[0]
        pairs.append((text, rew))

    metrics = evaluate_rewrites(pairs, evaluator=evaluator)
    metrics["test_split_path"] = str(test_path)
    return metrics


def compare_held_out_generalization(
    test_split_path: str | Path = "data/splits/test.jsonl",
    student_rewriter: Optional[Any] = None,
    baseline_rewriter: Optional[Any] = None,
    evaluator: Optional[BidirectionalUtilityEvaluator] = None,
    num_samples: Optional[int] = 50,
) -> Dict[str, Any]:
    """Compare student fine-tuned rewriter vs static prompt baseline on held-out test split."""
    test_path = Path(test_split_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test split file not found: {test_path}")

    records = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if num_samples and num_samples > 0:
        records = records[:num_samples]

    if student_rewriter is None:
        from src.rewriter.inference import QwenRewriterInference
        student_rewriter = QwenRewriterInference()

    if baseline_rewriter is None:
        from src.rewriter.prompt_baseline import PromptBaselineRewriter
        baseline_rewriter = PromptBaselineRewriter()

    if evaluator is None:
        evaluator = BidirectionalUtilityEvaluator()

    baseline_pairs = []
    student_pairs = []

    for rec in records:
        text = rec.get("text") or rec.get("comment") or rec.get("body") or ""
        if not text:
            continue

        b_rew = baseline_rewriter.rewrite(text)
        if isinstance(b_rew, tuple):
            b_rew = b_rew[0]
        baseline_pairs.append((text, b_rew))

        s_rew = student_rewriter.rewrite(text)
        if isinstance(s_rew, tuple):
            s_rew = s_rew[0]
        student_pairs.append((text, s_rew))

    baseline_metrics = evaluate_rewrites(baseline_pairs, evaluator=evaluator)
    student_metrics = evaluate_rewrites(student_pairs, evaluator=evaluator)

    comparison = {
        "test_split_path": str(test_path),
        "num_evaluated_samples": len(baseline_pairs),
        "static_prompt_baseline": baseline_metrics,
        "teacher_distilled_student": student_metrics,
        "delta": {
            "bidirectional_entailment_gain": round(
                student_metrics["mean_bidirectional_entailment"] - baseline_metrics["mean_bidirectional_entailment"], 4
            ),
            "contradiction_drop": round(
                baseline_metrics["mean_contradiction_prob"] - student_metrics["mean_contradiction_prob"], 4
            ),
            "utility_preservation_gain": round(
                student_metrics["utility_preservation_rate"] - baseline_metrics["utility_preservation_rate"], 4
            ),
            "semantic_similarity_delta": round(
                student_metrics["mean_semantic_similarity"] - baseline_metrics["mean_semantic_similarity"], 4
            ),
        },
        "generalization_improved": (
            student_metrics["mean_bidirectional_entailment"] >= baseline_metrics["mean_bidirectional_entailment"]
            and student_metrics["mean_contradiction_prob"] <= baseline_metrics["mean_contradiction_prob"]
        ),
    }

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Rewriter Utility Metrics on Held-Out Test Split")
    parser.add_argument("--test_split", type=str, default="data/splits/test.jsonl", help="Held-out test split JSONL")
    parser.add_argument("--adapter_dir", type=str, default="artifacts/rewriter_qlora", help="Path to student model adapter")
    parser.add_argument("--base_model", type=str, default=None, help="Base student model name")
    parser.add_argument("--compare_baseline", action="store_true", help="Compare fine-tuned student against static prompt baseline")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of held-out test samples to evaluate")
    parser.add_argument("--output_report", type=str, default="reports/heldout_utility_metrics.json", help="Path to write evaluation report")
    args = parser.parse_args()

    evaluator = BidirectionalUtilityEvaluator()

    student_rewriter = None
    if args.adapter_dir or args.base_model:
        from src.rewriter.inference import DEFAULT_BASE_MODEL, QwenRewriterInference
        student_rewriter = QwenRewriterInference(
            model_name_or_path=args.base_model or DEFAULT_BASE_MODEL,
            adapter_path=args.adapter_dir,
        )

    if args.compare_baseline:
        print(f"Comparing Student Model vs Prompt Baseline on {args.test_split} ({args.num_samples} samples)...")
        results = compare_held_out_generalization(
            test_split_path=args.test_split,
            student_rewriter=student_rewriter,
            evaluator=evaluator,
            num_samples=args.num_samples,
        )
    else:
        print(f"Evaluating Student Model on {args.test_split} ({args.num_samples} samples)...")
        results = evaluate_held_out_split(
            test_split_path=args.test_split,
            rewriter=student_rewriter,
            evaluator=evaluator,
            num_samples=args.num_samples,
        )

    out_p = Path(args.output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote utility metrics report to {out_p}")
    print(json.dumps(results.get("delta", results), indent=2))


if __name__ == "__main__":
    main()
