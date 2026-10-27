"""Dual Utility Metric Suite for InferenceGuard.

Computes semantic similarity (sentence-transformers/all-MiniLM-L6-v2) and
bidirectional Natural Language Inference (cross-encoder/nli-deberta-v3-large).
Catches dropped constraints and semantic drift that single-metric evaluations miss.

Two guards on top of the scores:

* ``utility_backend`` says whether the numbers came from the embedding and NLI
  models (``"model"``) or from the lexical fallback (``"fallback"``). A fallback
  score is word overlap and must not be reported as an embedding/NLI score.
* ``check_degeneracy`` rejects rewrites that no similarity score should rescue.
  The motivating case is the Session 04 resume rewrite
  ``"123 an area, an area an area"``, which MiniLM scored at cosine 0.907.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BACKEND_MODEL = "model"
BACKEND_FALLBACK = "fallback"

NLI_LABELS = ("contradiction", "entailment", "neutral")

# Degeneracy rule. A rewrite is degenerate if ANY of these holds. Each check
# is relative to the original wherever it can be, so long or number-heavy
# originals are not punished for what they already were.
#
#   empty           the rewrite has no word tokens
#   repetition      share of tokens that repeat an earlier token is at least
#                   DEGENERACY_REPETITION_EXCESS higher than in the original
#                   (rewrite must have >= DEGENERACY_MIN_TOKENS tokens)
#   length          rewrite/original token ratio outside
#                   [DEGENERACY_MIN_LENGTH_RATIO, DEGENERACY_MAX_LENGTH_RATIO]
#                   (original must have >= DEGENERACY_MIN_TOKENS tokens)
#   non_alphabetic  share of tokens with no letter is above
#                   DEGENERACY_MAX_NON_ALPHA and at least
#                   DEGENERACY_NON_ALPHA_EXCESS higher than in the original
#
# The thresholds are a first cut chosen to catch the Session 04 failure while
# passing the heuristic rewriter's normal output; they are not fitted on data.
DEGENERACY_MIN_TOKENS = 4
DEGENERACY_REPETITION_EXCESS = 0.25
DEGENERACY_MIN_LENGTH_RATIO = 0.25
DEGENERACY_MAX_LENGTH_RATIO = 3.0
DEGENERACY_MAX_NON_ALPHA = 0.5
DEGENERACY_NON_ALPHA_EXCESS = 0.25


def _tokens(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def _repetition_ratio(tokens) -> float:
    return (len(tokens) - len(set(tokens))) / len(tokens) if tokens else 0.0


def _non_alpha_ratio(tokens) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if not re.search(r"[a-z]", t)) / len(tokens)


def check_degeneracy(original_text: str, rewritten_text: str) -> Dict[str, Any]:
    """Apply the documented degeneracy rule to a rewrite.

    Returns the three statistics plus ``degenerate`` (bool) and ``reason``
    (the first rule that fired, or None).
    """
    orig = _tokens(original_text)
    rew = _tokens(rewritten_text)

    repetition = _repetition_ratio(rew)
    length_ratio = len(rew) / len(orig) if orig else 0.0
    non_alpha = _non_alpha_ratio(rew)

    reason: Optional[str] = None
    if not rew:
        reason = "empty"
    elif (
        len(rew) >= DEGENERACY_MIN_TOKENS
        and repetition - _repetition_ratio(orig) >= DEGENERACY_REPETITION_EXCESS
    ):
        reason = "repetition"
    elif len(orig) >= DEGENERACY_MIN_TOKENS and not (
        DEGENERACY_MIN_LENGTH_RATIO <= length_ratio <= DEGENERACY_MAX_LENGTH_RATIO
    ):
        reason = "length"
    elif (
        non_alpha > DEGENERACY_MAX_NON_ALPHA
        and non_alpha - _non_alpha_ratio(orig) >= DEGENERACY_NON_ALPHA_EXCESS
    ):
        reason = "non_alphabetic"

    return {
        "degenerate": reason is not None,
        "reason": reason,
        "repetition_ratio": round(repetition, 4),
        "length_ratio": round(length_ratio, 4),
        "non_alpha_ratio": round(non_alpha, 4),
    }


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
            logger.info(f"SentenceTransformer not loaded ({e}). Attempting transformers AutoModel fallback.")
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                class _HFSimilarityModel:
                    def __init__(self, model_name: str):
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
                        except Exception:
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                            self.model = AutoModel.from_pretrained(model_name)
                        self.model.eval()

                    def encode(self, text: str, convert_to_tensor: bool = True):
                        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                        with torch.no_grad():
                            out = self.model(**inputs)
                            mask = inputs["attention_mask"].unsqueeze(-1)
                            emb = (out[0] * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                            return emb.squeeze(0) if convert_to_tensor else emb.squeeze(0).cpu().numpy()

                self.similarity_model = _HFSimilarityModel(self.similarity_model_name)
                logger.info(f"Loaded similarity model via transformers: {self.similarity_model_name}")
            except Exception as exc:
                logger.info(f"Similarity model not loaded ({exc}). Using deterministic token embedding.")

        try:
            import os
            from transformers import pipeline
            # Check local files first to prevent unauthenticated multi-GB download hangs
            try:
                self.nli_pipeline = pipeline("text-classification", model=self.nli_model_name, model_kwargs={"local_files_only": True})
            except Exception:
                if os.environ.get("IG_DOWNLOAD_LARGE_MODELS", "0") == "1":
                    self.nli_pipeline = pipeline("text-classification", model=self.nli_model_name)
                else:
                    logger.info("NLI cross-encoder not cached locally. Using deterministic NLI logic.")
        except Exception as e:
            logger.info(f"NLI cross-encoder not loaded ({e}). Using deterministic NLI logic.")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic cosine similarity between text1 and text2."""
        return self._similarity_with_backend(text1, text2)[0]

    def _similarity_with_backend(self, text1: str, text2: str) -> Tuple[float, str]:
        if not text1.strip() or not text2.strip():
            return 0.0, BACKEND_MODEL if self.similarity_model is not None else BACKEND_FALLBACK

        if self.similarity_model is not None:
            try:
                emb1 = self.similarity_model.encode(text1, convert_to_tensor=True)
                emb2 = self.similarity_model.encode(text2, convert_to_tensor=True)
                import torch
                sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                return max(0.0, min(1.0, float(sim))), BACKEND_MODEL
            except Exception as e:
                logger.warning(f"Similarity model failed ({e}); using lexical fallback.")

        return compute_fallback_token_cosine(text1, text2), BACKEND_FALLBACK

    def compute_nli_direction(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Compute entailment, neutral, and contradiction probabilities for premise -> hypothesis."""
        return self._nli_with_backend(premise, hypothesis)[0]

    def _nli_with_backend(self, premise: str, hypothesis: str) -> Tuple[Dict[str, float], str]:
        if self.nli_pipeline is not None:
            try:
                # A cross-encoder takes the pair as two segments; joining them
                # into one string with a literal "[SEP]" is not the same input.
                # top_k=None returns every label, so all three are real
                # probabilities rather than one score and two invented ones.
                res = self.nli_pipeline({"text": premise, "text_pair": hypothesis}, top_k=None)
                if res and isinstance(res[0], list):
                    res = res[0]
                probs = {r["label"].lower(): float(r["score"]) for r in res}
                missing = [lbl for lbl in NLI_LABELS if lbl not in probs]
                if missing:
                    raise ValueError(
                        f"NLI model labels {sorted(probs)} do not include {missing}; "
                        "check the model config id2label"
                    )
                return {lbl: probs[lbl] for lbl in NLI_LABELS}, BACKEND_MODEL
            except Exception as e:
                logger.warning(f"NLI model failed ({e}); using lexical fallback.")

        # Genuine fallback heuristic based on lexical-semantic containment
        prem_words = set(re.findall(r"\b\w+\b", premise.lower()))
        hyp_words = set(re.findall(r"\b\w+\b", hypothesis.lower()))

        if not hyp_words:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}, BACKEND_FALLBACK

        overlap = len(prem_words & hyp_words) / len(hyp_words)

        # Check negation dissonance
        negations = {"not", "never", "no", "neither", "nor", "none", "cannot", "hardly"}
        prem_negs = prem_words & negations
        hyp_negs = hyp_words & negations
        contradiction = 0.0
        if len(prem_negs) != len(hyp_negs):
            contradiction = 0.5

        entailment = max(0.0, min(1.0, overlap * (1.0 - contradiction)))
        neutral = max(0.0, 1.0 - entailment - contradiction)
        return {"entailment": entailment, "contradiction": contradiction, "neutral": neutral}, BACKEND_FALLBACK

    def compute_metrics(self, original_text: str, rewritten_text: str) -> Dict[str, Any]:
        """Compute comprehensive dual utility metrics.

        Returns:
            Dictionary matching interface contract:
            {
                "cosine_similarity": float,
                "nli_forward": float,
                "nli_backward": float,
                "contradiction": float,
                "utility_score": float,       # 0.0 when degenerate
                "degenerate": bool,
                "degenerate_reason": str | None,
                "utility_backend": "model" | "fallback"
            }
        """
        cosine_sim, sim_backend = self._similarity_with_backend(original_text, rewritten_text)

        # Forward: original entails rewritten
        forward, fwd_backend = self._nli_with_backend(premise=original_text, hypothesis=rewritten_text)
        nli_forward = forward["entailment"]

        # Backward: rewritten entails original
        backward, bwd_backend = self._nli_with_backend(premise=rewritten_text, hypothesis=original_text)
        nli_backward = backward["entailment"]

        # "model" only if every component came from a model.
        backend = (
            BACKEND_MODEL
            if sim_backend == fwd_backend == bwd_backend == BACKEND_MODEL
            else BACKEND_FALLBACK
        )

        contradiction = max(forward["contradiction"], backward["contradiction"])

        # Composite utility score: weighted combination with contradiction penalty
        composite = (
            0.45 * cosine_sim
            + 0.30 * nli_forward
            + 0.25 * nli_backward
            - 0.50 * contradiction
        )
        utility_score = max(0.0, min(1.0, composite))

        degeneracy = check_degeneracy(original_text, rewritten_text)
        if degeneracy["degenerate"]:
            utility_score = 0.0

        return {
            "cosine_similarity": round(float(cosine_sim), 4),
            "nli_forward": round(float(nli_forward), 4),
            "nli_backward": round(float(nli_backward), 4),
            "contradiction": round(float(contradiction), 4),
            "utility_score": round(float(utility_score), 4),
            "degenerate": degeneracy["degenerate"],
            "degenerate_reason": degeneracy["reason"],
            "utility_backend": backend,
        }

    def is_acceptable_utility(
        self,
        metrics: Dict[str, Any],
        min_cosine: float = 0.82,
        min_utility: float = 0.60,
    ) -> bool:
        """Check if metrics satisfy acceptance thresholds."""
        return (
            metrics["cosine_similarity"] >= min_cosine
            and metrics["utility_score"] >= min_utility
            and metrics["contradiction"] <= 0.20
            and not metrics.get("degenerate", False)
        )
