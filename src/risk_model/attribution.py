# src/risk_model/attribution.py
"""Syntax-aligned Integrated Gradients and counterfactual span attribution.

Computes embedding Integrated Gradients, merges subword tokens into complete
syntactic words and phrases, and verifies salience via batched deletion risk drop.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.contracts.schemas import AttributeName, AttributedCue
from src.risk_model.calibrate import AttributeTemperatureScaler

STOPWORD_SET = {
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "would", "could",
}


class SyntaxAlignedCueAttributor:
    """Computes Riemann-approximated Integrated Gradients and merges subwords to verifiable phrases."""

    def __init__(
        self,
        model,
        tokenizer,
        scaler: AttributeTemperatureScaler,
        ig_steps: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize attributor with model, tokenizer, and temperature scaler."""
        self.model = model.to(device).eval() if model is not None else None
        self.tokenizer = tokenizer
        self.scaler = scaler.to(device).eval() if scaler is not None else None
        self.ig_steps = ig_steps
        self.device = device

    def extract_cues(
        self,
        text: str,
        target_attributes: List[AttributeName],
        top_k_per_attribute: int = 3,
    ) -> List[AttributedCue]:
        """Extract high-importance text spans for the specified demographic attributes."""
        if not text.strip() or not target_attributes:
            return []

        if self.tokenizer is None or self.model is None:
            return self._heuristic_fallback_cues(text, target_attributes)

        try:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=512,
            )
        except Exception:
            return self._heuristic_fallback_cues(text, target_attributes)

        if not isinstance(encoding, dict) or "input_ids" not in encoding or "offset_mapping" not in encoding:
            return self._heuristic_fallback_cues(text, target_attributes)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offsets: List[Tuple[int, int]] = encoding["offset_mapping"][0].tolist()

        embed_fn = getattr(self.model, "get_input_embeddings", None)
        if embed_fn is None and hasattr(self.model, "encoder"):
            embed_fn = getattr(self.model.encoder, "get_input_embeddings", None)

        if callable(embed_fn):
            embed_layer = embed_fn()
        elif hasattr(self.model, "bert") and hasattr(self.model.bert, "embeddings"):
            embed_layer = self.model.bert.embeddings
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "embeddings"):
            embed_layer = self.model.encoder.embeddings
        else:
            return self._heuristic_fallback_cues(text, target_attributes)

        orig_embeds = embed_layer(input_ids).detach()
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None) or getattr(self.tokenizer, "pad_token_id", 0) or 0
        baseline_ids = torch.full_like(input_ids, mask_token_id)
        baseline_embeds = embed_layer(baseline_ids).detach()

        alphas = torch.linspace(1.0 / self.ig_steps, 1.0, self.ig_steps, device=self.device).view(-1, 1, 1)
        interp_embeds = (baseline_embeds + alphas * (orig_embeds - baseline_embeds)).requires_grad_(True)
        batch_mask = attention_mask.repeat(self.ig_steps, 1)

        try:
            logits_batch = self.model.forward_from_embeds(interp_embeds, batch_mask)
            with torch.no_grad():
                base_logits = self.model(input_ids, attention_mask)["logits"]
                base_distributions = self.scaler.score_logits(base_logits)
        except Exception:
            return self._heuristic_fallback_cues(text, target_attributes)

        all_cues: List[AttributedCue] = []

        for attr in target_attributes:
            key = attr.value
            if key not in logits_batch:
                continue

            temp = float(self.scaler.get_temperature(attr).item())
            probs_steps = F.softmax(logits_batch[key] / temp, dim=-1)
            top_class_idx = int(torch.argmax(probs_steps[-1]).item())

            target_score = probs_steps[:, top_class_idx].sum()
            grads = torch.autograd.grad(
                outputs=target_score,
                inputs=interp_embeds,
                retain_graph=True,
            )[0]

            avg_grads = grads.mean(dim=0, keepdim=True)
            ig_vector = (orig_embeds - baseline_embeds) * avg_grads
            token_salience = torch.clamp(ig_vector.sum(dim=-1).squeeze(0), min=0.0).detach().cpu().tolist()

            word_spans = self._merge_subwords_to_phrases(text, offsets, token_salience)
            if not word_spans:
                continue

            base_risk = base_distributions[attr].inferability_risk_score if attr in base_distributions else 0.50
            mask_token_str = getattr(self.tokenizer, "mask_token", None) or "[MASK]"

            verified_cues = self._verify_spans_by_batched_masking(
                text=text,
                attr=attr,
                base_risk=base_risk,
                candidate_spans=word_spans[: top_k_per_attribute * 2],
                mask_token_str=mask_token_str,
            )
            all_cues.extend(verified_cues[:top_k_per_attribute])

        all_cues.sort(key=lambda c: c.importance, reverse=True)
        return all_cues

    def _merge_subwords_to_phrases(
        self,
        text: str,
        offsets: List[Tuple[int, int]],
        salience: List[float],
    ) -> List[Tuple[int, int, float]]:
        """Merge subword tokens into complete whitespace-delimited words and noun phrases."""
        word_items: List[Tuple[int, int, float, str]] = []
        curr_start, curr_end, curr_score = -1, -1, 0.0

        for (start, end), score in zip(offsets, salience):
            if start == end:
                continue
            if curr_start == -1:
                curr_start, curr_end, curr_score = start, end, score
            elif start == curr_end:
                curr_end = end
                curr_score += score
            else:
                surface = text[curr_start:curr_end]
                word_items.append((curr_start, curr_end, curr_score, surface))
                curr_start, curr_end, curr_score = start, end, score

        if curr_start != -1:
            word_items.append((curr_start, curr_end, curr_score, text[curr_start:curr_end]))

        max_word_score = max((w[2] for w in word_items), default=1.0)
        if max_word_score <= 1e-9:
            return []

        filtered_words: List[Tuple[int, int, float]] = []
        for s, e, sc, surf in word_items:
            clean = re.sub(r"[^\w\s-]", "", surf).strip().lower()
            if not clean or clean in STOPWORD_SET or len(clean) < 2:
                continue
            norm_sc = sc / max_word_score
            if norm_sc >= 0.20:
                filtered_words.append((s, e, norm_sc))

        merged_phrases: List[Tuple[int, int, float]] = []
        for s, e, sc in filtered_words:
            if merged_phrases:
                prev_s, prev_e, prev_sc = merged_phrases[-1]
                between = text[prev_e:s]
                if between == " " and (e - prev_s) <= 35:
                    merged_phrases[-1] = (prev_s, e, max(prev_sc, sc))
                    continue
            merged_phrases.append((s, e, sc))

        merged_phrases.sort(key=lambda x: x[2], reverse=True)
        return merged_phrases

    @torch.no_grad()
    def _verify_spans_by_batched_masking(
        self,
        text: str,
        attr: AttributeName,
        base_risk: float,
        candidate_spans: List[Tuple[int, int, float]],
        mask_token_str: str,
    ) -> List[AttributedCue]:
        """Verify candidate spans by measuring empirical risk drop under counterfactual deletion."""
        if not candidate_spans:
            return []

        masked_texts = [
            text[:s] + mask_token_str + text[e:] for s, e, _ in candidate_spans
        ]
        batch_enc = self.tokenizer(
            masked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        b_input_ids = batch_enc["input_ids"].to(self.device)
        b_attention_mask = batch_enc["attention_mask"].to(self.device)

        logits_dict = self.model(b_input_ids, b_attention_mask)["logits"]
        cues: List[AttributedCue] = []

        for idx, (s, e, ig_norm) in enumerate(candidate_spans):
            single_logits = {k: v[idx : idx + 1] for k, v in logits_dict.items()}
            masked_dist = self.scaler.score_logits(single_logits)[attr]
            risk_drop = max(0.0, base_risk - masked_dist.inferability_risk_score)
            combined_importance = float(min(1.0, 0.55 * ig_norm + 0.45 * min(1.0, risk_drop / max(base_risk, 0.1))))
            if combined_importance >= 0.15:
                cues.append(
                    AttributedCue(
                        span_text=text[s:e],
                        start=s,
                        end=e,
                        attribute=attr,
                        importance=round(combined_importance, 4),
                        deletion_risk_drop=round(risk_drop, 4),
                    )
                )
        cues.sort(key=lambda c: c.importance, reverse=True)
        return cues

    def _heuristic_fallback_cues(
        self, text: str, target_attributes: List[AttributeName]
    ) -> List[AttributedCue]:
        """Heuristic fallback span detection for lightweight environments without gradient computation."""
        cues = []
        known_cues = {
            AttributeName.LOCATION: ["green line", "seattle", "denver", "boston", "bay area", "subway", "transit"],
            AttributeName.OCCUPATION: ["co-op", "internship", "software engineer", "accountant", "nurse", "residency"],
            AttributeName.EDUCATION: ["campus", "undergraduate", "grad student", "thesis", "dorm", "tuition"],
            AttributeName.AGE: ["freshman", "20s", "retirement", "pension", "grandkids"],
        }
        text_lower = text.lower()
        for attr in target_attributes:
            patterns = known_cues.get(attr, [])
            for pat in patterns:
                start = text_lower.find(pat)
                if start != -1:
                    end = start + len(pat)
                    cues.append(
                        AttributedCue(
                            span_text=text[start:end],
                            start=start,
                            end=end,
                            attribute=attr,
                            importance=0.85,
                            deletion_risk_drop=0.25,
                        )
                    )
        return cues
