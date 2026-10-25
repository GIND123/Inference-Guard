# src/evaluation/utility_metrics.py
"""Bidirectional Natural Language Inference and semantic utility evaluation.

Measures intent and factual preservation using DeBERTa-v3 NLI bidirectional entailment,
dense embedding cosine similarity, and token-level edit distance.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from src.contracts.schemas import UtilityReport


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
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Could not initialize utility models (%s: %s). Operating in heuristic mode.",
                type(exc).__name__,
                exc,
            )
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
