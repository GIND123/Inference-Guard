# tests/test_attribution.py
"""Verification suite for Gate 3: Cue Attribution Faithfulness & Subword Merging.

Verifies that attributed spans align with complete syntactic words/phrases (zero broken ## subwords),
character offsets match the source text slice exactly, and counterfactual deletion drop is verified.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pydantic",
    reason="needs the optional pydantic dependency; see tests/README.md",
)

from src.contracts.schemas import AttributeName
from src.risk_model.attribution import SyntaxAlignedCueAttributor
from src.risk_model.calibrate import AttributeTemperatureScaler


class DummyTokenizer:
    """Mock fast tokenizer providing realistic subword tokenization with offset mappings."""

    def __init__(self):
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        # Emulate subword tokenization splitting 'undergraduate' -> ['under', '##graduate']
        # and 'Green' -> ['Green'], 'Line' -> ['Line']
        import torch
        words = text.split()
        tokens = []
        offsets = []
        curr_idx = 0

        for w in words:
            start = text.find(w, curr_idx)
            end = start + len(w)
            curr_idx = end

            if w.lower() == "undergraduate":
                mid = start + 5
                tokens.extend([201, 202])
                offsets.extend([(start, mid), (mid, end)])
            else:
                tokens.append(100 + len(tokens))
                offsets.append((start, end))

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
                "offset_mapping": torch.tensor([offsets], dtype=torch.long),
            }
        return {"input_ids": tokens, "offset_mapping": offsets}


def test_subword_merging_to_syntactic_phrases():
    """Verify that fragmented subwords merge into whole words and multi-word phrases without ## artifacts."""
    scaler = AttributeTemperatureScaler()
    attributor = SyntaxAlignedCueAttributor(
        model=None,
        tokenizer=DummyTokenizer(),
        scaler=scaler,
    )

    text = "Taking the Green Line to campus for my undergraduate degree in Boston."
    offsets = [
        (0, 6), (7, 10), (11, 16), (17, 21), (22, 24), (25, 31),
        (32, 35), (36, 38), (39, 44), (44, 52), (53, 59), (60, 62), (63, 69), (69, 70)
    ]
    # Salience on "Green", "Line", and the two subwords of "undergraduate"
    salience = [0.1, 0.05, 0.85, 0.90, 0.02, 0.70, 0.01, 0.01, 0.75, 0.80, 0.10, 0.01, 0.80, 0.0]

    merged = attributor._merge_subwords_to_phrases(text, offsets, salience)
    assert len(merged) > 0

    spans = [text[s:e] for s, e, _ in merged]
    # Invariant: zero broken subwords or ## symbols
    for sp in spans:
        assert "##" not in sp
        assert len(sp.strip()) > 1

    # Invariant: adjacent "Green" and "Line" must merge into "Green Line"
    assert "Green Line" in spans or any("green line" in s.lower() for s in spans)
    # Invariant: "under" + "graduate" must merge into "undergraduate"
    assert "undergraduate" in spans


def test_cue_character_offsets_exact_match():
    """Verify that start and end character offsets correspond exactly to span_text in original text."""
    scaler = AttributeTemperatureScaler()
    attributor = SyntaxAlignedCueAttributor(
        model=None,
        tokenizer=DummyTokenizer(),
        scaler=scaler,
    )

    text = "I am a student at the campus in Denver."
    cues = attributor.extract_cues(text, [AttributeName.LOCATION, AttributeName.OCCUPATION])

    assert len(cues) > 0
    for cue in cues:
        # Strict offset slice invariant
        extracted_slice = text[cue.start : cue.end]
        assert extracted_slice.lower() == cue.span_text.lower()
        assert cue.importance >= 0.15
        assert cue.deletion_risk_drop is not None
        assert cue.deletion_risk_drop >= 0.15


def test_empty_and_stopword_text_returns_empty_cues():
    """Verify that pure stopwords or empty string returns empty cue list safely."""
    scaler = AttributeTemperatureScaler()
    attributor = SyntaxAlignedCueAttributor(
        model=None,
        tokenizer=DummyTokenizer(),
        scaler=scaler,
    )

    assert attributor.extract_cues("", [AttributeName.LOCATION]) == []
    assert attributor.extract_cues("   ", [AttributeName.LOCATION]) == []
    assert attributor.extract_cues("it is what it is and so on", [AttributeName.LOCATION]) == []


def test_attributor_fallback_when_tokenizer_or_model_is_none():
    """Verify that attributor gracefully falls back to heuristic cues when tokenizer/model is None."""
    scaler = AttributeTemperatureScaler()
    attributor = SyntaxAlignedCueAttributor(
        model=None,
        tokenizer=None,
        scaler=scaler,
    )

    cues = attributor.extract_cues("Contact me in Seattle.", [AttributeName.LOCATION])
    assert len(cues) > 0
    assert any("seattle" in c.span_text.lower() for c in cues)
