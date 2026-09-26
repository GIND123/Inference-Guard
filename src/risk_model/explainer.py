"""Integrated Gradients attribution and Cue extraction for ModernBERT risk classifier.

Axiomatic attribution method replacing O(N) leave-one-out token masking.
Calculates token-level gradient path integrals from a [PAD] reference baseline,
verifies the completeness axiom convergence delta (<= 0.05), aggregates subword
tokens into word/entity spans, and maps results to the Cue dataclass contract.
"""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from src.product.risk_bands import (
    ATTRIBUTES,
    CUE_IMPORTANCE_FLOOR,
    MAX_CUES_SHOWN,
    Cue,
)

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """Computes Integrated Gradients attribution for multi-head risk classifiers."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        baseline_strategy: str = "pad",
        default_steps: int = 30,
        max_steps: int = 50,
        convergence_threshold: float = 0.05,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize explainer.

        Args:
            model: ModernBertRiskClassifier or multi-head sequence classification model.
            tokenizer: Fast tokenizer matching model pre-training.
            baseline_strategy: "pad" (default) for [PAD] token baseline, or "zero" for zeros.
            default_steps: Number of Riemann summation steps (default 30).
            max_steps: Expanded step count if convergence delta > 0.05 (default 50).
            convergence_threshold: Maximum allowable completeness delta (default 0.05).
            device: Execution device (cuda / cpu). If None, inferred from model parameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_strategy = baseline_strategy
        self.default_steps = default_steps
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.eval()

    def _resolve_target_head(
        self,
        target_head: Optional[Union[int, str]],
        text: str,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, str]:
        """Map target_head parameter to head index (0..3) and attribute name string."""
        if target_head is None:
            # Predict scores to determine primary risk head
            if input_ids is None and self.tokenizer is not None:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

            if input_ids is not None:
                with torch.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = []
                    for logit in logits:
                        if logit.shape[-1] == 2:
                            p = torch.softmax(logit, dim=-1)[:, 1].item()
                        else:
                            p = torch.sigmoid(logit).squeeze().item()
                        probs.append(p)
                head_idx = int(torch.tensor(probs).argmax().item())
                return head_idx, ATTRIBUTES[head_idx]
            return 0, ATTRIBUTES[0]

        if isinstance(target_head, str):
            target_clean = target_head.strip().lower()
            if target_clean not in ATTRIBUTES:
                raise ValueError(
                    f"Unknown attribute {target_head!r}. Must be one of {ATTRIBUTES}"
                )
            return ATTRIBUTES.index(target_clean), target_clean

        if isinstance(target_head, int):
            if not 0 <= target_head < len(ATTRIBUTES):
                raise ValueError(
                    f"Invalid target_head index {target_head}. Must be in range 0..{len(ATTRIBUTES) - 1}"
                )
            return target_head, ATTRIBUTES[target_head]

        raise TypeError(f"target_head must be int, str, or None; got {type(target_head)}")

    def _get_embeddings_layer(self) -> nn.Embedding:
        """Extract input embedding layer from model."""
        if hasattr(self.model, "get_input_embeddings"):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return emb
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "get_input_embeddings"):
            emb = self.model.bert.get_input_embeddings()
            if emb is not None:
                return emb
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "embeddings"):
            if hasattr(self.model.bert.embeddings, "tok_embeddings"):
                return self.model.bert.embeddings.tok_embeddings
            return self.model.bert.embeddings
        if hasattr(self.model, "embeddings"):
            return self.model.embeddings
        raise AttributeError("Could not locate input embeddings layer on model.")

    def _get_target_scalar_logit(
        self, logits: Union[List[torch.Tensor], torch.Tensor], head_idx: int
    ) -> torch.Tensor:
        """Extract target logit tensor from multi-head logits output."""
        if isinstance(logits, list):
            head_logit = logits[head_idx]
        elif isinstance(logits, torch.Tensor) and logits.ndim == 2:
            head_logit = logits[:, head_idx : head_idx + 1]
        else:
            raise TypeError(f"Unexpected logits type: {type(logits)}")

        if head_logit.shape[-1] == 2:
            return head_logit[:, 1] - head_logit[:, 0]
        return head_logit.squeeze(-1)

    def _forward_with_embeddings(
        self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor], head_idx: int
    ) -> torch.Tensor:
        """Forward pass supporting direct inputs_embeds."""
        # Attempt direct model forward with inputs_embeds
        try:
            logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return self._get_target_scalar_logit(logits, head_idx)
        except (TypeError, AttributeError):
            pass

        # Fallback for models where backbone wraps bert
        if hasattr(self.model, "bert"):
            outputs = self.model.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            if hasattr(outputs, "last_hidden_state"):
                seq_output = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                seq_output = outputs[0]
            else:
                seq_output = outputs

            cls_rep = seq_output[:, 0, :]
            if hasattr(self.model, "dropout"):
                cls_rep = self.model.dropout(cls_rep)
            head_logits = self.model.classifiers[head_idx](cls_rep)
            if head_logits.shape[-1] == 2:
                return head_logits[:, 1] - head_logits[:, 0]
            return head_logits.squeeze(-1)

        raise RuntimeError("Model does not support forward pass with inputs_embeds.")

    def attribute(
        self,
        text: str,
        target_head: Optional[Union[int, str]] = None,
        steps: Optional[int] = None,
        additional_entity_spans: Optional[Sequence[Tuple[int, int]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute Integrated Gradients attribution and extract top risk cues.

        Args:
            text: Raw input text.
            target_head: Head index (0..3), attribute string, or None for auto-detection.
            steps: Riemann summation step count (defaults to self.default_steps).
            additional_entity_spans: Optional list of (char_start, char_end) entity spans.
            input_ids: Optional pre-tokenized tensor [1, seq_len].
            attention_mask: Optional attention mask tensor [1, seq_len].

        Returns:
            dict with keys:
                - cues: list[Cue] (filtered by floor >= 0.20, capped at 5)
                - convergence_delta: float
                - target_attribute: str
                - target_head: int
                - converged: bool
        """
        if not text or not text.strip():
            if target_head is None:
                head_idx = 0
                attr_name = ATTRIBUTES[0]
            else:
                head_idx, attr_name = self._resolve_target_head(target_head, text)
            return {
                "cues": [],
                "convergence_delta": 0.0,
                "target_attribute": attr_name,
                "target_head": head_idx,
                "converged": True,
            }

        head_idx, attr_name = self._resolve_target_head(
            target_head, text, input_ids=input_ids, attention_mask=attention_mask
        )
        n_steps = steps or self.default_steps

        # Tokenize if tokens not provided
        if input_ids is None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided when input_ids is not supplied")
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            offsets = encoded["offset_mapping"][0].tolist()
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids).to(self.device)
            else:
                attention_mask = attention_mask.to(self.device)
            # Create synthetic word boundaries if tokenizer is absent
            words = text.split()
            offsets = []
            curr = 0
            for w in words:
                idx_start = text.find(w, curr)
                idx_end = idx_start + len(w)
                offsets.append((idx_start, idx_end))
                curr = idx_end
            while len(offsets) < input_ids.shape[1]:
                offsets.append((0, 0))

        embeddings_layer = self._get_embeddings_layer()

        with torch.no_grad():
            x = embeddings_layer(input_ids)  # Shape: (1, seq_len, hidden_dim)

            if self.baseline_strategy == "zero":
                x_prime = torch.zeros_like(x)
            else:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
                baseline_ids = torch.full_like(input_ids, pad_id)
                x_prime = embeddings_layer(baseline_ids)

            # Baseline and target scalar forward outputs for completeness check
            f_x = self._forward_with_embeddings(x, attention_mask, head_idx).item()
            f_baseline = self._forward_with_embeddings(x_prime, attention_mask, head_idx).item()
            delta_f = f_x - f_baseline

        # Execute Riemann path integration
        token_attr, conv_delta, converged = self._compute_ig_path(
            x=x,
            x_prime=x_prime,
            attention_mask=attention_mask,
            head_idx=head_idx,
            steps=n_steps,
            delta_f=delta_f,
        )

        # Retry with max_steps if convergence delta violated
        if not converged and n_steps < self.max_steps:
            token_attr, conv_delta, converged = self._compute_ig_path(
                x=x,
                x_prime=x_prime,
                attention_mask=attention_mask,
                head_idx=head_idx,
                steps=self.max_steps,
                delta_f=delta_f,
            )

        # Aggregate tokens to spans and build Cues
        cues = self._aggregate_spans_to_cues(
            text=text,
            token_attributions=token_attr,
            offsets=offsets,
            attribute_name=attr_name,
            additional_entity_spans=additional_entity_spans,
        )

        return {
            "cues": cues,
            "convergence_delta": round(float(conv_delta), 6),
            "target_attribute": attr_name,
            "target_head": head_idx,
            "converged": converged,
        }

    def _compute_ig_path(
        self,
        x: torch.Tensor,
        x_prime: torch.Tensor,
        attention_mask: torch.Tensor,
        head_idx: int,
        steps: int,
        delta_f: float,
    ) -> Tuple[torch.Tensor, float, bool]:
        """Compute Riemann path integral for given number of steps."""
        alphas = torch.linspace(1.0 / steps, 1.0, steps, device=self.device)

        # Shape: (steps, seq_len, hidden_dim)
        x_diff = x - x_prime
        interpolated = x_prime + alphas.view(steps, 1, 1) * x_diff
        interpolated = interpolated.detach().clone().requires_grad_(True)

        batch_mask = attention_mask.repeat(steps, 1)

        # Forward pass over batched interpolations
        logits = self._forward_with_embeddings(interpolated, batch_mask, head_idx)

        # Backward pass to obtain gradients
        grad_outputs = torch.ones_like(logits)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Average gradients across interpolation steps (Riemann sum)
        avg_grads = grads.mean(dim=0, keepdim=True)

        # Token-level attribution: sum over hidden_dim of (x - x_prime) * avg_grads
        token_attr = (x_diff * avg_grads).sum(dim=-1).squeeze(0)

        total_attr = token_attr.sum().item()
        conv_delta = abs(delta_f - total_attr)
        converged = bool(conv_delta <= self.convergence_threshold)

        return token_attr.detach().cpu(), conv_delta, converged

    def _aggregate_spans_to_cues(
        self,
        text: str,
        token_attributions: torch.Tensor,
        offsets: List[Tuple[int, int]],
        attribute_name: str,
        additional_entity_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> List[Cue]:
        """Aggregate token-level attributions into word/entity spans and return Cues."""
        seq_len = min(len(offsets), token_attributions.shape[0])
        raw_spans: List[Dict[str, Any]] = []

        entity_spans = list(additional_entity_spans or [])
        punctuation_strip = string.punctuation + string.whitespace

        idx = 0
        while idx < seq_len:
            start_char, end_char = offsets[idx]
            # Exclude special tokens ([CLS], [SEP], padding)
            if start_char == end_char or (start_char == 0 and end_char == 0 and idx > 0):
                idx += 1
                continue

            # Check if token falls inside any multi-word entity span
            matched_entity = None
            for ent_start, ent_end in entity_spans:
                if ent_start <= start_char and end_char <= ent_end:
                    matched_entity = (ent_start, ent_end)
                    break

            if matched_entity:
                ent_start, ent_end = matched_entity
                span_tokens = []
                while idx < seq_len and offsets[idx][0] >= ent_start and offsets[idx][1] <= ent_end:
                    span_tokens.append(idx)
                    idx += 1
                span_text = text[ent_start:ent_end].strip(punctuation_strip)
                if span_text and any(c.isalnum() for c in span_text):
                    span_score = float(token_attributions[span_tokens].sum().item())
                    raw_spans.append({"span": span_text, "score": max(0.0, span_score)})
                continue

            # Default: Aggregate subwords into single word
            span_tokens = [idx]
            word_start, word_end = start_char, end_char
            idx += 1
            while idx < seq_len:
                next_start, next_end = offsets[idx]
                if next_start == next_end:
                    idx += 1
                    continue
                # Contiguous subword without intervening whitespace
                if next_start == word_end:
                    span_tokens.append(idx)
                    word_end = next_end
                    idx += 1
                else:
                    break

            span_text = text[word_start:word_end].strip(punctuation_strip)
            if span_text and any(c.isalnum() for c in span_text):
                span_score = float(token_attributions[span_tokens].sum().item())
                raw_spans.append({"span": span_text, "score": max(0.0, span_score)})

        if not raw_spans:
            return []

        # Step 1b: Aggregate peak attribution score per unique span across occurrences
        span_max_scores: Dict[str, float] = {}
        for s in raw_spans:
            span_clean = s["span"]
            if span_clean not in span_max_scores or s["score"] > span_max_scores[span_clean]:
                span_max_scores[span_clean] = s["score"]

        if not span_max_scores:
            return []

        # Step 2: Normalize span scores relative to maximum score across all spans
        max_score = max(span_max_scores.values())
        if max_score <= 1e-9:
            return []

        # Step 3: Filter by floor, sort descending with tie-break, cap at MAX_CUES_SHOWN
        candidates: List[Cue] = []
        for span_clean, score in span_max_scores.items():
            norm_importance = round(score / max_score, 4)
            if norm_importance >= CUE_IMPORTANCE_FLOOR and any(c.isalnum() for c in span_clean):
                candidates.append(
                    Cue(span=span_clean, attribute=attribute_name, importance=norm_importance)
                )

        # Deterministic sorting: highest importance first, ties broken alphabetically
        candidates.sort(key=lambda c: (-c.importance, c.span))

        return candidates[:MAX_CUES_SHOWN]
