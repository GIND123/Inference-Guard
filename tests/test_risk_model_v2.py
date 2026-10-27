# tests/test_risk_model_v2.py
"""Unit tests for MultiTaskModernBERTRiskModel and CalibratedRiskPredictor.

Verifies:
1. Multi-task ModernBERT head architectures, attention pooling, and Focal Loss computation.
2. CalibratedRiskPredictor multi-attribute scoring, margin/entropy sensitivity, and fallback handling.
3. Resilience against dictionary-returning tokenizers (no AttributeError on .to(device)).
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "torch",
    reason="needs the optional torch dependency; see tests/README.md",
)
import torch
import torch.nn as nn

from src.contracts.schemas import AttributeName, RiskBand
from src.risk_model.calibrate import AttributeTemperatureScaler
from src.risk_model.model import FocalCrossEntropyLoss, MultiTaskModernBERTRiskModel
from src.risk_model.predict import CalibratedRiskPredictor
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY


class DummyBackbone(nn.Module):
    """Mock backbone simulating ModernBERT embedding and sequence hidden states."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 500):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        # Return inputs_embeds unchanged as sequence hidden state [B, L, H]
        return inputs_embeds


class DictTokenizer:
    """Tokenizer returning plain Python dicts rather than HuggingFace BatchEncoding."""

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 512):
        tokens = [min(abs(hash(w)) % self.vocab_size, self.vocab_size - 1) for w in text.split()]
        if not tokens:
            tokens = [0]
        tokens = tokens[:max_length]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
        }


def test_focal_cross_entropy_loss():
    """Verify focal cross entropy computes non-negative scalar loss and ignores -100."""
    loss_fn = FocalCrossEntropyLoss(gamma=1.5, ignore_index=-100)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, -100, 3])

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # All ignored targets should produce 0.0 loss
    all_ignored = torch.tensor([-100, -100, -100, -100])
    loss_zero = loss_fn(logits, all_ignored)
    assert loss_zero.item() == 0.0


def test_multitask_modernbert_forward_and_loss():
    """Verify multi-task forward pass outputs valid logits for all 4 attributes and computes focal loss."""
    backbone = DummyBackbone(hidden_size=64)
    model = MultiTaskModernBERTRiskModel(backbone=backbone, hidden_size=64)

    input_ids = torch.randint(0, 400, (2, 8))
    attention_mask = torch.ones((2, 8), dtype=torch.long)

    # Forward without labels
    out = model(input_ids, attention_mask)
    assert "logits" in out
    logits = out["logits"]

    assert set(logits.keys()) == {attr.value for attr in AttributeName}
    for attr, labels in ATTRIBUTE_TAXONOMY.items():
        assert logits[attr.value].shape == (2, len(labels))

    # Forward with labels
    labels = {
        "age": torch.tensor([0, 1]),
        "location": torch.tensor([2, -100]),
    }
    out_with_loss = model(input_ids, attention_mask, labels=labels)
    assert "loss" in out_with_loss
    assert out_with_loss["loss"].item() > 0.0
    assert "per_attribute_loss" in out_with_loss
    assert "age" in out_with_loss["per_attribute_loss"]
    assert "location" in out_with_loss["per_attribute_loss"]


def test_calibrated_risk_predictor_with_dict_tokenizer():
    """Verify CalibratedRiskPredictor handles plain dict tokenizers without AttributeError on .to()."""
    backbone = DummyBackbone(hidden_size=64)
    model = MultiTaskModernBERTRiskModel(backbone=backbone, hidden_size=64)
    scaler = AttributeTemperatureScaler()
    tokenizer = DictTokenizer()

    predictor = CalibratedRiskPredictor(model=model, tokenizer=tokenizer, scaler=scaler)

    report = predictor.predict("I am an undergraduate student in Boston.")
    assert 0.0 <= report.overall_risk_score <= 1.0
    assert report.overall_risk_band in (RiskBand.LOW, RiskBand.MEDIUM, RiskBand.HIGH)
    assert len(report.attributes) == 4
    for attr in AttributeName:
        assert attr in report.attributes
        dist = report.attributes[attr]
        assert 0.0 <= dist.inferability_risk_score <= 1.0


def test_calibrated_risk_predictor_empty_text():
    """Verify empty or whitespace-only text returns safe baseline report."""
    backbone = DummyBackbone(hidden_size=64)
    model = MultiTaskModernBERTRiskModel(backbone=backbone, hidden_size=64)
    scaler = AttributeTemperatureScaler()
    tokenizer = DictTokenizer()

    predictor = CalibratedRiskPredictor(model=model, tokenizer=tokenizer, scaler=scaler)

    report = predictor.predict("   ")
    assert report.overall_risk_score == 0.0
    assert report.overall_risk_band == RiskBand.LOW
    assert report.primary_risk_attribute is None
    for attr in AttributeName:
        assert report.attributes[attr].predicted_value == "UNKNOWN"
        assert report.attributes[attr].inferability_risk_score == 0.0
