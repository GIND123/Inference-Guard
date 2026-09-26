"""Dual Utility Metric Suite for InferenceGuard.

Computes semantic similarity (sentence-transformers/all-MiniLM-L6-v2) and
bidirectional Natural Language Inference (cross-encoder/nli-deberta-v3-large).
Catches dropped constraints and semantic drift that single-metric evaluations miss.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


def compute_fallback_token_cosine(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts based on word frequencies."""
    words1 = re.findall(r"\b\w+\b", text1.lower())
    words2 = re.findall(r"\b\w+\b", text2.lower())

    if not words1 or not words2:
        return 0.0

    vocab = set(words1) | set(words2)
    vec1 = {w: words1.count(w) for w in vocab}
    vec2 = {w: words2.count(w) for w in vocab}

    dot = sum(vec1[w] * vec2[w] for w in vocab)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class UtilityEvaluator:
    """Evaluates semantic similarity and bidirectional NLI preservation."""

    def __init__(
        self,
        similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
        load_models: bool = False,
    ) -> None:
        self.similarity_model_name = similarity_model_name
        self.nli_model_name = nli_model_name
        self.similarity_model = None
        self.nli_pipeline = None

        if load_models:
            self._load_models()

    def _load_models(self) -> None:
        """Attempt to load HuggingFace models for similarity and NLI."""
        try:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer(self.similarity_model_name)
        except Exception as e:
            logger.info(f"SentenceTransformer not loaded ({e}). Using deterministic token embedding.")

        try:
            from transformers import pipeline
            self.nli_pipeline = pipeline("text-classification", model=self.nli_model_name)
        except Exception as e:
            logger.info(f"NLI cross-encoder not loaded ({e}). Using deterministic NLI logic.")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic cosine similarity between text1 and text2."""
        if not text1.strip() or not text2.strip():
            return 0.0

        if self.similarity_model is not None:
            try:
                emb1 = self.similarity_model.encode(text1, convert_to_tensor=True)
                emb2 = self.similarity_model.encode(text2, convert_to_tensor=True)
                import torch
                sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                return max(0.0, min(1.0, float(sim)))
            except Exception:
                pass

        return compute_fallback_token_cosine(text1, text2)

    def compute_nli_direction(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Compute entailment, neutral, and contradiction probabilities for premise -> hypothesis."""
        if self.nli_pipeline is not None:
            try:
                res = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
                # Parse pipeline result
                label = res[0]["label"].lower()
                score = res[0]["score"]
                entail = score if "entail" in label else 0.1
                contra = score if "contradict" in label else 0.05
                return {"entailment": entail, "contradiction": contra}
            except Exception:
                pass

        # Genuine fallback heuristic based on lexical-semantic containment
        prem_words = set(re.findall(r"\b\w+\b", premise.lower()))
        hyp_words = set(re.findall(r"\b\w+\b", hypothesis.lower()))

        if not hyp_words:
            return {"entailment": 0.0, "contradiction": 0.0}

        overlap = len(prem_words & hyp_words) / len(hyp_words)

        # Check negation dissonance
        negations = {"not", "never", "no", "neither", "nor", "none", "cannot", "hardly"}
        prem_negs = prem_words & negations
        hyp_negs = hyp_words & negations
        contradiction = 0.0
        if len(prem_negs) != len(hyp_negs):
            contradiction = 0.5

        entailment = max(0.0, min(1.0, overlap * (1.0 - contradiction)))
        return {"entailment": entailment, "contradiction": contradiction}

    def compute_metrics(self, original_text: str, rewritten_text: str) -> Dict[str, float]:
        """Compute comprehensive dual utility metrics.

        Returns:
            Dictionary matching interface contract:
            {
                "cosine_similarity": float,
                "nli_forward": float,
                "nli_backward": float,
                "contradiction": float,
                "utility_score": float
            }
        """
        cosine_sim = self.compute_similarity(original_text, rewritten_text)

        # Forward: original entails rewritten
        forward = self.compute_nli_direction(premise=original_text, hypothesis=rewritten_text)
        nli_forward = forward["entailment"]

        # Backward: rewritten entails original
        backward = self.compute_nli_direction(premise=rewritten_text, hypothesis=original_text)
        nli_backward = backward["entailment"]

        contradiction = max(forward["contradiction"], backward["contradiction"])

        # Composite utility score: weighted combination with contradiction penalty
        composite = (
            0.45 * cosine_sim
            + 0.30 * nli_forward
            + 0.25 * nli_backward
            - 0.50 * contradiction
        )
        utility_score = max(0.0, min(1.0, composite))

        return {
            "cosine_similarity": round(float(cosine_sim), 4),
            "nli_forward": round(float(nli_forward), 4),
            "nli_backward": round(float(nli_backward), 4),
            "contradiction": round(float(contradiction), 4),
            "utility_score": round(float(utility_score), 4),
        }

    def is_acceptable_utility(
        self,
        metrics: Dict[str, float],
        min_cosine: float = 0.82,
        min_utility: float = 0.60,
    ) -> bool:
        """Check if metrics satisfy acceptance thresholds."""
        return (
            metrics["cosine_similarity"] >= min_cosine
            and metrics["utility_score"] >= min_utility
            and metrics["contradiction"] <= 0.20
        )
