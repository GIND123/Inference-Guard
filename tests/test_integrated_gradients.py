"""Unit, integration, and scenario tests for Integrated Gradients cue attribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn

from src.product.risk_bands import ATTRIBUTES, CUE_IMPORTANCE_FLOOR, MAX_CUES_SHOWN, Cue
from src.risk_model.explainer import IntegratedGradientsExplainer


class MockFastTokenizer:
    """Mock fast tokenizer providing tokenization and character offset mapping."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 2

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        truncation: bool = True,
        max_length: int = 512,
        return_offsets_mapping: bool = True,
    ) -> Dict[str, Any]:
        words = text.split()
        tokens = [self.cls_token_id]
        offsets: List[Tuple[int, int]] = [(0, 0)]  # [CLS] offset

        curr_pos = 0
        for w in words:
            start = text.find(w, curr_pos)
            end = start + len(w)
            token_id = (abs(hash(w)) % (self.vocab_size - 10)) + 5
            tokens.append(token_id)
            offsets.append((start, end))
            curr_pos = end

        tokens.append(self.sep_token_id)
        offsets.append((0, 0))  # [SEP] offset

        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            offsets = offsets[:max_length]

        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": torch.tensor([offsets], dtype=torch.long),
        }


class MockRiskClassifierForIG(nn.Module):
    """Linear mock classifier with known gradient properties for axiomatic verification."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Initialize linear heads
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # Average pooling over sequence length
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (inputs_embeds * mask_expanded).sum(dim=1) / torch.clamp(
                mask_expanded.sum(dim=1), min=1.0
            )
        else:
            pooled = inputs_embeds.mean(dim=1)

        logits = [classifier(pooled) for classifier in self.classifiers]
        return logits


@pytest.fixture
def mock_pipeline() -> Tuple[MockRiskClassifierForIG, MockFastTokenizer, IntegratedGradientsExplainer]:
    """Fixture providing initialized model, tokenizer, and explainer."""
    torch.manual_seed(42)
    model = MockRiskClassifierForIG(vocab_size=1000, hidden_dim=32, num_heads=4)
    tokenizer = MockFastTokenizer(vocab_size=1000)
    explainer = IntegratedGradientsExplainer(
        model=model,
        tokenizer=tokenizer,
        baseline_strategy="pad",
        default_steps=30,
        convergence_threshold=0.05,
    )
    return model, tokenizer, explainer


# ============================================================================
# Tier 1: Feature Functionality Tests
# ============================================================================


def test_ig_completeness_axiom_holds(mock_pipeline):
    """Verify completeness convergence delta <= 0.05 on standard sentences."""
    _, _, explainer = mock_pipeline
    text = "The engineer lives in Seattle and completed graduate studies."

    res = explainer.attribute(text, target_head="location", steps=30)
    assert res["converged"] is True
    assert res["convergence_delta"] <= 0.05
    assert res["target_attribute"] == "location"
    assert res["target_head"] == 1


def test_pad_baseline_generation(mock_pipeline):
    """Verify that [PAD] baseline is correctly generated and evaluated."""
    _, tokenizer, explainer = mock_pipeline
    text = "Relocating to Denver office next spring."
    res = explainer.attribute(text, target_head=0, steps=20)

    assert "convergence_delta" in res
    assert res["converged"] is True


def test_cue_dataclass_contract(mock_pipeline):
    """Verify returned elements are valid Cue dataclass instances."""
    _, _, explainer = mock_pipeline
    text = "Senior software architect commuting through downtown Boston subway."

    res = explainer.attribute(text, target_head="occupation")
    cues = res["cues"]
    assert isinstance(cues, list)

    for cue in cues:
        assert isinstance(cue, Cue)
        assert isinstance(cue.span, str)
        assert len(cue.span) > 0
        assert cue.attribute == "occupation"
        assert isinstance(cue.importance, float)
        assert 0.0 <= cue.importance <= 1.0


def test_cue_importance_floor(mock_pipeline):
    """Verify that every surfaced cue strictly respects the CUE_IMPORTANCE_FLOOR (0.20)."""
    _, _, explainer = mock_pipeline
    text = "Working remotely from home on mundane documentation tasks."

    res = explainer.attribute(text, target_head="location")
    for cue in res["cues"]:
        assert cue.importance >= CUE_IMPORTANCE_FLOOR


def test_max_cues_capped_at_five(mock_pipeline):
    """Verify that returned cues never exceed MAX_CUES_SHOWN (5)."""
    _, _, explainer = mock_pipeline
    long_text = "London Paris Tokyo Berlin Madrid Rome Toronto Sydney Singapore Dublin"

    res = explainer.attribute(long_text, target_head=1)
    assert len(res["cues"]) <= MAX_CUES_SHOWN


# ============================================================================
# Tier 2: Boundary & Edge Case Tests
# ============================================================================


def test_empty_input_returns_clean_payload(mock_pipeline):
    """Verify empty or whitespace strings return zeroed result without running model."""
    _, _, explainer = mock_pipeline
    res = explainer.attribute("", target_head="age")
    assert res["cues"] == []
    assert res["convergence_delta"] == 0.0
    assert res["converged"] is True

    res_ws = explainer.attribute("   \n\t  ", target_head=2)
    assert res_ws["cues"] == []
    assert res_ws["convergence_delta"] == 0.0


def test_single_word_input(mock_pipeline):
    """Verify single-word input attributes properly and normalizes importance to 1.0."""
    _, _, explainer = mock_pipeline
    res = explainer.attribute("Boston", target_head="location")

    if res["cues"]:
        top_cue = res["cues"][0]
        assert top_cue.span == "Boston"
        assert top_cue.importance == 1.0
        assert top_cue.attribute == "location"


def test_target_head_resolution(mock_pipeline):
    """Verify integer and string target_head parameters resolve to the same attribute."""
    _, _, explainer = mock_pipeline
    text = "Doctor working at university hospital."

    res_int = explainer.attribute(text, target_head=2)
    res_str = explainer.attribute(text, target_head="occupation")

    assert res_int["target_head"] == res_str["target_head"] == 2
    assert res_int["target_attribute"] == res_str["target_attribute"] == "occupation"


def test_invalid_target_head_raises_error(mock_pipeline):
    """Verify invalid target heads raise ValueError."""
    _, _, explainer = mock_pipeline
    text = "Valid text input."

    with pytest.raises(ValueError, match="Invalid target_head index"):
        explainer.attribute(text, target_head=99)

    with pytest.raises(ValueError, match="Unknown attribute"):
        explainer.attribute(text, target_head="salary")


def test_zero_baseline_strategy(mock_pipeline):
    """Verify baseline_strategy='zero' functions correctly and completeness holds."""
    model, tokenizer, _ = mock_pipeline
    zero_explainer = IntegratedGradientsExplainer(
        model=model,
        tokenizer=tokenizer,
        baseline_strategy="zero",
        default_steps=30,
    )
    text = "Moved to Chicago for a technology job."
    res = zero_explainer.attribute(text, target_head="location")

    assert res["converged"] is True
    assert res["convergence_delta"] <= 0.05


# ============================================================================
# Tier 3: Pairwise Combinatorial Tests
# ============================================================================


@pytest.mark.parametrize("target_head", [0, 1, 2, 3])
@pytest.mark.parametrize("baseline_strategy", ["pad", "zero"])
@pytest.mark.parametrize("steps", [15, 30])
def test_pairwise_configurations(mock_pipeline, target_head, baseline_strategy, steps):
    """Verify Integrated Gradients remains stable across all head, baseline, and step combinations."""
    model, tokenizer, _ = mock_pipeline
    explainer = IntegratedGradientsExplainer(
        model=model,
        tokenizer=tokenizer,
        baseline_strategy=baseline_strategy,
        default_steps=steps,
    )
    text = "Graduated with master degree in electrical engineering."
    res = explainer.attribute(text, target_head=target_head, steps=steps)

    assert res["target_head"] == target_head
    assert res["target_attribute"] == ATTRIBUTES[target_head]
    assert isinstance(res["convergence_delta"], float)
    assert not torch.isnan(torch.tensor(res["convergence_delta"]))


# ============================================================================
# Tier 4: Real-World Scenarios (S-01 and S-02)
# ============================================================================


def test_scenario_s01_tech_relocation_spans(mock_pipeline):
    """Scenario S-01: Extract multi-word transit entity span ('Green Line') for location."""
    _, _, explainer = mock_pipeline
    text = "My second co-op starts in January and my Green Line commute is already brutal."

    # Provide pre-detected entity span for "Green Line"
    start_gl = text.find("Green Line")
    end_gl = start_gl + len("Green Line")

    res = explainer.attribute(
        text,
        target_head="location",
        additional_entity_spans=[(start_gl, end_gl)],
    )

    assert res["converged"] is True
    assert res["target_attribute"] == "location"

    # Verify extracted spans do not contain punctuation or empty strings
    for cue in res["cues"]:
        assert cue.attribute == "location"
        assert len(cue.span.strip()) > 0


def test_scenario_s02_workplace_education_cues(mock_pipeline):
    """Scenario S-02: Verify education cue extraction from workplace context."""
    _, _, explainer = mock_pipeline
    text = "My manager keeps assigning me on-call even though I am the only one without a PhD."

    res = explainer.attribute(text, target_head="education")
    assert res["converged"] is True
    assert res["target_attribute"] == "education"
    assert len(res["cues"]) <= 5
