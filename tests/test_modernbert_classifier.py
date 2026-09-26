"""Unit tests for ModernBertRiskClassifier architecture, forward pass, and persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn

from src.risk_model.model import ATTRIBUTES, ModernBertRiskClassifier


class TinyMockBackbone(nn.Module):
    """Lightweight mock backbone for fast offline testing."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embeddings(input_ids)
        h = self.linear(h)
        return h


@pytest.fixture
def mock_classifier() -> ModernBertRiskClassifier:
    """Create a 4-head classifier using the tiny mock backbone."""
    backbone = TinyMockBackbone(vocab_size=100, hidden_size=32)
    classifier = ModernBertRiskClassifier(
        model_name="mock-modernbert",
        num_heads=4,
        num_classes_per_head=1,
        backbone=backbone,
        hidden_size=32,
    )
    return classifier


def test_classifier_head_count_and_attributes(mock_classifier: ModernBertRiskClassifier):
    """Verify classifier initializes 4 heads corresponding to product attributes."""
    assert len(mock_classifier.classifiers) == 4
    assert mock_classifier.num_heads == 4
    assert len(ATTRIBUTES) == 4
    assert "age" in ATTRIBUTES
    assert "location" in ATTRIBUTES
    assert "occupation" in ATTRIBUTES
    assert "education" in ATTRIBUTES


def test_forward_with_input_ids(mock_classifier: ModernBertRiskClassifier):
    """Verify forward pass with input_ids returns 4 scalar logit tensors."""
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    logits = mock_classifier(input_ids=input_ids, attention_mask=attention_mask)
    assert isinstance(logits, list)
    assert len(logits) == 4
    for head_logit in logits:
        assert head_logit.shape == (batch_size, 1)


def test_forward_with_inputs_embeds(mock_classifier: ModernBertRiskClassifier):
    """Verify forward pass accepts continuous inputs_embeds for gradient attribution."""
    batch_size, seq_len, hidden_dim = 2, 8, 32
    embeds = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    attention_mask = torch.ones((batch_size, seq_len))

    logits = mock_classifier(inputs_embeds=embeds, attention_mask=attention_mask)
    assert len(logits) == 4
    for head_logit in logits:
        assert head_logit.shape == (batch_size, 1)

    # Test backpropagation to inputs_embeds
    loss = logits[0].sum()
    loss.backward()
    assert embeds.grad is not None
    assert embeds.grad.shape == embeds.shape


def test_forward_input_validation(mock_classifier: ModernBertRiskClassifier):
    """Verify forward raises ValueError when neither or both input representations are passed."""
    batch_size, seq_len, hidden_dim = 1, 4, 32
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    embeds = torch.randn(batch_size, seq_len, hidden_dim)

    # Both provided
    with pytest.raises(ValueError, match="Must specify exactly one"):
        mock_classifier(input_ids=input_ids, inputs_embeds=embeds)

    # Neither provided
    with pytest.raises(ValueError, match="Must specify exactly one"):
        mock_classifier()


def test_temperature_scaling_in_forward(mock_classifier: ModernBertRiskClassifier):
    """Verify temperature scaling scales logits without changing raw outputs."""
    mock_classifier.eval()
    input_ids = torch.randint(0, 50, (1, 4))
    raw_logits = mock_classifier(input_ids=input_ids, apply_temperature=False)

    # Set head 0 temperature to 2.0 and head 1 to 0.5
    mock_classifier.set_temperatures([2.0, 0.5, 1.0, 1.0])
    scaled_logits = mock_classifier(input_ids=input_ids, apply_temperature=True)

    expected_0 = raw_logits[0] / 2.0
    expected_1 = raw_logits[1] / 0.5

    assert torch.allclose(scaled_logits[0], expected_0, atol=1e-5)
    assert torch.allclose(scaled_logits[1], expected_1, atol=1e-5)


def test_temperature_getter_and_setter(mock_classifier: ModernBertRiskClassifier):
    """Verify get_temperatures and set_temperatures validation and storage."""
    assert mock_classifier.get_temperatures() == [1.0, 1.0, 1.0, 1.0]

    mock_classifier.set_temperatures([1.2, 0.8, 1.5, 0.9])
    assert mock_classifier.get_temperatures() == [1.2, 0.8, 1.5, 0.9]

    # Non-positive temperature error
    with pytest.raises(ValueError, match="strictly positive"):
        mock_classifier.set_temperatures([1.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="strictly positive"):
        mock_classifier.set_temperatures([1.0, -0.5, 1.0, 1.0])

    # Wrong head count error
    with pytest.raises(ValueError, match="Expected 4 temperatures"):
        mock_classifier.set_temperatures([1.0, 1.0])


def test_predict_proba_returns_bounded_probabilities(mock_classifier: ModernBertRiskClassifier):
    """Verify predict_proba returns a dictionary of valid probabilities in [0.0, 1.0]."""
    input_ids = torch.randint(0, 50, (3, 6))
    probs = mock_classifier.predict_proba(input_ids=input_ids)

    assert isinstance(probs, dict)
    assert set(probs.keys()) == set(ATTRIBUTES)
    for attr, prob_tensor in probs.items():
        assert prob_tensor.shape == (3, 1)
        assert torch.all(prob_tensor >= 0.0)
        assert torch.all(prob_tensor <= 1.0)


def test_get_input_embeddings(mock_classifier: ModernBertRiskClassifier):
    """Verify get_input_embeddings returns backbone embedding layer."""
    emb = mock_classifier.get_input_embeddings()
    assert isinstance(emb, nn.Embedding)
    assert emb.embedding_dim == 32


def test_model_save_and_from_pretrained(mock_classifier: ModernBertRiskClassifier):
    """Verify model weights and configuration roundtrip serialization."""
    mock_classifier.set_temperatures([1.25, 0.75, 1.10, 0.95])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "checkpoint"
        mock_classifier.save_pretrained(save_path)

        assert (save_path / "model.pt").exists()
        assert (save_path / "risk_model_config.json").exists()

        loaded_backbone = TinyMockBackbone(vocab_size=100, hidden_size=32)
        loaded_model = ModernBertRiskClassifier.from_pretrained(
            save_path,
            device="cpu",
            backbone=loaded_backbone,
        )

        assert loaded_model.num_heads == 4
        assert loaded_model.get_temperatures() == [1.25, 0.75, 1.10, 0.95]

        # Verify weights match
        mock_classifier.eval()
        loaded_model.eval()
        dummy_input = torch.randint(0, 50, (1, 4))
        with torch.no_grad():
            orig_out = mock_classifier(input_ids=dummy_input)
            loaded_out = loaded_model(input_ids=dummy_input)

        for orig_h, loaded_h in zip(orig_out, loaded_out):
            assert torch.allclose(orig_h, loaded_h, atol=1e-5)
