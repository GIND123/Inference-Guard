"""Adversarial stress-testing suite for Milestone 1: Risk Modeling.

Tests edge cases, boundary inputs, degeneracies, and completeness axioms for:
- ModernBertRiskClassifier (src/risk_model/model.py)
- compute_ece and TemperatureScaler (src/risk_model/calibration.py)
- IntegratedGradientsExplainer (src/risk_model/explainer.py)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.product.risk_bands import (
    ATTRIBUTES,
    CUE_IMPORTANCE_FLOOR,
    MAX_CUES_SHOWN,
    Cue,
)
from src.risk_model.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    compute_ece,
)
from src.risk_model.explainer import IntegratedGradientsExplainer
from src.risk_model.model import ModernBertRiskClassifier


# ============================================================================
# Test Fixtures and Mock Infrastructure
# ============================================================================

class TinyMockBackbone(nn.Module):
    """Mock backbone producing linear transformations for fast deterministic tests."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
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
        return self.linear(h)


class MockFastTokenizer:
    """Mock fast tokenizer with deterministic token IDs and offset mapping."""

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
        offsets: List[Tuple[int, int]] = [(0, 0)]

        curr_pos = 0
        for w in words:
            start = text.find(w, curr_pos)
            if start == -1:
                start = curr_pos
            end = start + len(w)
            token_id = (abs(hash(w)) % (self.vocab_size - 10)) + 5
            tokens.append(token_id)
            offsets.append((start, end))
            curr_pos = end

        tokens.append(self.sep_token_id)
        offsets.append((0, 0))

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


class LinearRiskClassifierForIG(nn.Module):
    """Linear classifier with known analytical gradients for Integrated Gradients tests."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
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

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (inputs_embeds * mask_expanded).sum(dim=1) / torch.clamp(
                mask_expanded.sum(dim=1), min=1.0
            )
        else:
            pooled = inputs_embeds.mean(dim=1)

        return [classifier(pooled) for classifier in self.classifiers]


# ============================================================================
# Suite 1: ModernBertRiskClassifier Adversarial Tests
# ============================================================================

class TestClassifierEdgeCases:
    """Stress tests for ModernBertRiskClassifier under adversarial inputs."""

    @pytest.fixture
    def classifier(self) -> ModernBertRiskClassifier:
        backbone = TinyMockBackbone(vocab_size=1000, hidden_size=32)
        return ModernBertRiskClassifier(
            model_name="mock-modernbert",
            num_heads=4,
            num_classes_per_head=1,
            backbone=backbone,
            hidden_size=32,
        )

    def test_single_token_input(self, classifier: ModernBertRiskClassifier):
        """Single-token sequence [1, 1] must forward without shape errors."""
        input_ids = torch.tensor([[42]], dtype=torch.long)
        logits = classifier(input_ids=input_ids)
        assert len(logits) == 4
        for l in logits:
            assert l.shape == (1, 1)
            assert torch.isfinite(l).all()

        probs = classifier.predict_proba(input_ids=input_ids)
        assert len(probs) == 4
        for attr, p in probs.items():
            assert p.shape == (1, 1)
            assert 0.0 <= p.item() <= 1.0

    def test_max_length_512_tokens(self, classifier: ModernBertRiskClassifier):
        """Max-length sequence [2, 512] must process cleanly across all 4 heads."""
        input_ids = torch.randint(0, 1000, (2, 512), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = classifier(input_ids=input_ids, attention_mask=attention_mask)
        assert len(logits) == 4
        for l in logits:
            assert l.shape == (2, 1)
            assert torch.isfinite(l).all()

    def test_random_embeddings_extreme_values(self, classifier: ModernBertRiskClassifier):
        """Embeddings with extreme values (+100.0, -100.0, 0.0) must not cause NaN."""
        batch_size, seq_len, hidden_size = 2, 64, 32
        normal_embeds = torch.randn(batch_size, seq_len, hidden_size)
        logits_normal = classifier(inputs_embeds=normal_embeds)
        assert all(torch.isfinite(l).all() for l in logits_normal)

        extreme_embeds = torch.full((batch_size, seq_len, hidden_size), 100.0)
        extreme_embeds[1] = -100.0
        logits_extreme = classifier(inputs_embeds=extreme_embeds)
        assert all(torch.isfinite(l).all() for l in logits_extreme)

        zero_embeds = torch.zeros(batch_size, seq_len, hidden_size)
        logits_zero = classifier(inputs_embeds=zero_embeds)
        assert all(torch.isfinite(l).all() for l in logits_zero)

    def test_extreme_temperatures_near_zero(self, classifier: ModernBertRiskClassifier):
        """Temperature T -> 0 (e.g. 1e-4) must scale logits without zero division crash."""
        classifier.set_temperatures([1e-4, 1e-4, 1e-4, 1e-4])
        input_ids = torch.tensor([[10, 20, 30]])
        logits = classifier(input_ids=input_ids, apply_temperature=True)
        assert len(logits) == 4
        assert all(torch.isfinite(l).all() for l in logits)

        probs = classifier.predict_proba(input_ids=input_ids, apply_temperature=True)
        for attr, p in probs.items():
            assert torch.isfinite(p).all()
            assert 0.0 <= p.item() <= 1.0
            assert p.item() == 0.0 or p.item() == 1.0 or abs(p.item() - 0.5) > 0.49

    def test_extreme_temperatures_large(self, classifier: ModernBertRiskClassifier):
        """Temperature T -> 100 smoothly compresses logits to 0 and probs to ~0.50."""
        classifier.set_temperatures([100.0, 100.0, 100.0, 100.0])
        input_ids = torch.tensor([[10, 20, 30]])
        logits = classifier(input_ids=input_ids, apply_temperature=True)
        for l in logits:
            assert abs(l.item()) < 1.0

        probs = classifier.predict_proba(input_ids=input_ids, apply_temperature=True)
        for attr, p in probs.items():
            assert abs(p.item() - 0.5) < 0.15

    def test_invalid_temperature_values(self, classifier: ModernBertRiskClassifier):
        """Temperatures <= 0 or wrong length must raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            classifier.set_temperatures([0.0, 1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="strictly positive"):
            classifier.set_temperatures([-0.5, 1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="Expected 4 temperatures"):
            classifier.set_temperatures([1.0, 1.0])

    def test_empty_sequence_raises(self, classifier: ModernBertRiskClassifier):
        """Empty sequence [1, 0] must raise an error cleanly rather than silent corrupt output."""
        empty_ids = torch.empty((1, 0), dtype=torch.long)
        with pytest.raises((IndexError, ValueError)):
            classifier(input_ids=empty_ids)

    def test_exclusive_input_validation(self, classifier: ModernBertRiskClassifier):
        """Specifying both or neither of input_ids and inputs_embeds must raise ValueError."""
        with pytest.raises(ValueError, match="Must specify exactly one"):
            classifier(input_ids=None, inputs_embeds=None)

        ids = torch.tensor([[1, 2]])
        emb = torch.randn(1, 2, 32)
        with pytest.raises(ValueError, match="Must specify exactly one"):
            classifier(input_ids=ids, inputs_embeds=emb)


# ============================================================================
# Suite 2: compute_ece and Calibration Edge Cases
# ============================================================================

class TestCalibrationEdgeCases:
    """Stress tests for compute_ece and TemperatureScaler under degenerate scenarios."""

    def test_all_zeros_predictions_and_labels(self):
        """All-zero probabilities and labels must produce ECE=0.0 and is_calibrated=True."""
        probs = [0.0] * 100
        labels = [0] * 100
        metrics = compute_ece(probs, labels)
        assert metrics.ece == 0.0
        assert metrics.mce == 0.0
        assert metrics.brier_score == 0.0
        assert metrics.is_calibrated is True

    def test_all_ones_predictions_and_labels(self):
        """All-one probabilities and labels must produce ECE=0.0 and is_calibrated=True."""
        probs = [1.0] * 100
        labels = [1] * 100
        metrics = compute_ece(probs, labels)
        assert metrics.ece == 0.0
        assert metrics.mce == 0.0
        assert metrics.brier_score == 0.0
        assert metrics.is_calibrated is True

    def test_inverted_degenerate_predictions(self):
        """Probabilities=1.0 for all-0 labels must produce ECE=1.0 and is_calibrated=False."""
        probs = [1.0] * 100
        labels = [0] * 100
        metrics = compute_ece(probs, labels)
        assert abs(metrics.ece - 1.0) < 1e-6
        assert abs(metrics.mce - 1.0) < 1e-6
        assert abs(metrics.brier_score - 1.0) < 1e-6
        assert metrics.is_calibrated is False

    def test_single_bin_configuration(self):
        """n_bins=1 must compute ECE as the overall mean confidence-accuracy gap."""
        probs = [0.2, 0.8]
        labels = [0, 1]
        metrics = compute_ece(probs, labels, n_bins=1)
        assert metrics.n_bins == 1
        assert metrics.ece == 0.0
        assert metrics.bin_counts == [2]

        metrics2 = compute_ece([0.8, 0.8], [0, 0], n_bins=1)
        assert abs(metrics2.ece - 0.8) < 1e-6

    def test_boundary_probabilities_zero_and_one(self):
        """Boundary probabilities exactly 0.0 and 1.0 must be correctly partitioned."""
        probs = [0.0, 1.0, 0.0, 1.0]
        labels = [0, 1, 0, 1]
        metrics = compute_ece(probs, labels, n_bins=10)
        assert metrics.ece == 0.0
        assert metrics.is_calibrated is True
        assert metrics.bin_counts[0] == 2
        assert metrics.bin_counts[-1] == 2

    def test_uniform_random_confidences(self):
        """Uniform random confidences must compute valid bounded metrics."""
        np.random.seed(42)
        p = np.random.uniform(0.0, 1.0, 500)
        y = np.random.choice([0, 1], 500)
        metrics = compute_ece(p, y, n_bins=10)
        assert 0.0 <= metrics.ece <= 1.0
        assert 0.0 <= metrics.mce <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert sum(metrics.bin_counts) == 500

    def test_quantile_strategy_with_identical_probabilities(self):
        """Quantile strategy must not crash when all predicted probabilities are identical."""
        probs = [0.5] * 100
        labels = [0] * 50 + [1] * 50
        metrics = compute_ece(probs, labels, n_bins=5, strategy="quantile")
        assert np.isfinite(metrics.ece)
        assert abs(metrics.ece) < 1e-5

    def test_n_bins_zero_or_negative_raises_value_error(self):
        """n_bins <= 0 must raise ValueError rather than returning fake ece=0.0."""
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_ece([0.5], [1], n_bins=0)

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_ece([0.5], [1], n_bins=-1)

    def test_empty_predictions_raise_value_error(self):
        """Empty input sequences must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot compute calibration metrics on empty arrays"):
            compute_ece([], [])

    def test_non_finite_inputs_raise_value_error(self):
        """NaN or Inf in inputs must raise ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            compute_ece([0.5, float("nan")], [0, 1])

        with pytest.raises(ValueError, match="non-finite"):
            compute_ece([0.5, 0.6], [0, float("inf")])

    def test_out_of_bounds_probabilities_raise_value_error(self):
        """Probabilities < 0.0 or > 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="probabilities must lie in"):
            compute_ece([-0.01, 0.5], [0, 1])

        with pytest.raises(ValueError, match="probabilities must lie in"):
            compute_ece([1.01, 0.5], [0, 1])

    def test_non_binary_labels_raise_value_error(self):
        """Labels other than {0, 1} must raise ValueError."""
        with pytest.raises(ValueError, match="labels must be binary"):
            compute_ece([0.5, 0.5], [0, 2])

    def test_temperature_scaler_with_homogeneous_labels(self):
        """TemperatureScaler.fit must handle all-zero or all-one validation labels gracefully."""
        scaler = TemperatureScaler(max_iter=20)
        z = [torch.randn(30) for _ in range(4)]
        y_zeros = [torch.zeros(30) for _ in range(4)]

        temps = scaler.fit(z, y_zeros)
        for attr, t in temps.items():
            assert t > 0.0
            assert np.isfinite(t)

    def test_temperature_scaler_with_extreme_logits(self):
        """TemperatureScaler must remain stable when logits are large (+-1000)."""
        scaler = TemperatureScaler(max_iter=20)
        z_large = [torch.tensor([1000.0, -1000.0] * 15) for _ in range(4)]
        y_mix = [torch.tensor([1.0, 0.0] * 15) for _ in range(4)]

        temps = scaler.fit(z_large, y_mix)
        for attr, t in temps.items():
            assert t > 0.0
            assert np.isfinite(t)

        cal_probs = scaler.predict_calibrated(z_large)
        for attr, p in cal_probs.items():
            assert 0.0 <= p <= 1.0


# ============================================================================
# Suite 3: Integrated Gradients Explainer Edge Cases
# ============================================================================

class TestIntegratedGradientsEdgeCases:
    """Stress tests for IntegratedGradientsExplainer under adversarial inputs."""

    @pytest.fixture
    def explainer(self) -> IntegratedGradientsExplainer:
        model = LinearRiskClassifierForIG(vocab_size=1000, hidden_dim=32, num_heads=4)
        tokenizer = MockFastTokenizer(vocab_size=1000)
        return IntegratedGradientsExplainer(
            model=model,
            tokenizer=tokenizer,
            baseline_strategy="pad",
            default_steps=30,
            max_steps=50,
            convergence_threshold=0.05,
        )

    def test_zero_cues_neutral_input(self, explainer: IntegratedGradientsExplainer):
        """Neutral input where no token exceeds floor must return empty cues list."""
        res = explainer.attribute("the an of in", target_head=0)
        assert isinstance(res["cues"], list)
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True
        for cue in res["cues"]:
            assert cue.importance >= CUE_IMPORTANCE_FLOOR

    def test_all_identical_tokens(self, explainer: IntegratedGradientsExplainer):
        """Input with all identical tokens must maintain completeness delta <= 0.05."""
        text = "word word word word word word word word"
        res = explainer.attribute(text, target_head=1)
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True
        assert len(res["cues"]) <= MAX_CUES_SHOWN

    def test_special_punctuation_string(self, explainer: IntegratedGradientsExplainer):
        """Inputs with mixed special characters must satisfy completeness axiom delta <= 0.05."""
        text = "user@domain.com #engineer (level-4) salary: $150k+ [boston, ma]!"
        res = explainer.attribute(text, target_head="occupation")
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True
        for cue in res["cues"]:
            assert cue.importance >= CUE_IMPORTANCE_FLOOR
            assert len(cue.span) > 0

    def test_only_punctuation_input(self, explainer: IntegratedGradientsExplainer):
        """Input composed entirely of punctuation must return 0 cues without crashing."""
        text = "... ??? !!! ::: ;;; ---"
        res = explainer.attribute(text, target_head=2)
        assert res["cues"] == []
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True

    def test_single_character_input(self, explainer: IntegratedGradientsExplainer):
        """Single character input must compute attribution and converge."""
        res = explainer.attribute("X", target_head=3)
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True
        assert len(res["cues"]) <= 1

    def test_long_sequence_beyond_512_tokens(self, explainer: IntegratedGradientsExplainer):
        """Sequences longer than 512 tokens must truncate gracefully and converge."""
        long_text = " ".join([f"token{i}" for i in range(600)])
        res = explainer.attribute(long_text, target_head=0)
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True
        assert len(res["cues"]) <= MAX_CUES_SHOWN

    def test_zero_baseline_strategy_completeness(self, explainer: IntegratedGradientsExplainer):
        """Zero baseline strategy must also satisfy completeness axiom delta <= 0.05."""
        explainer_zero = IntegratedGradientsExplainer(
            model=explainer.model,
            tokenizer=explainer.tokenizer,
            baseline_strategy="zero",
            default_steps=30,
            convergence_threshold=0.05,
        )
        res = explainer_zero.attribute(
            "software engineer working in cambridge", target_head="location"
        )
        assert res["convergence_delta"] <= 0.05
        assert res["converged"] is True

    def test_target_head_resolution_edge_cases(self, explainer: IntegratedGradientsExplainer):
        """Target head must accept valid int, valid str, None, and reject invalid values."""
        res_auto = explainer.attribute("test prompt", target_head=None)
        assert res_auto["target_head"] in range(4)
        assert res_auto["target_attribute"] in ATTRIBUTES

        for i, attr in enumerate(ATTRIBUTES):
            res_idx = explainer.attribute("test prompt", target_head=i)
            assert res_idx["target_head"] == i
            assert res_idx["target_attribute"] == attr

        for attr in ATTRIBUTES:
            res_str = explainer.attribute("test prompt", target_head=attr.upper())
            assert res_str["target_attribute"] == attr

        # Empty text with string target_head must not silently default to "age"
        res_empty_loc = explainer.attribute("", target_head="location")
        assert res_empty_loc["target_attribute"] == "location"
        assert res_empty_loc["target_head"] == 1

        with pytest.raises(ValueError, match="Invalid target_head index"):
            explainer.attribute("test prompt", target_head=4)

        with pytest.raises(ValueError, match="Unknown attribute"):
            explainer.attribute("test prompt", target_head="salary")

    def test_cue_sorting_and_cap_contract(self, explainer: IntegratedGradientsExplainer):
        """Cues must be sorted by importance descending with alphabetical tie-breaks, capped at 5."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        res = explainer.attribute(text, target_head=0)
        cues = res["cues"]
        assert len(cues) <= MAX_CUES_SHOWN

        for i in range(len(cues) - 1):
            c1, c2 = cues[i], cues[i + 1]
            if c1.importance == c2.importance:
                assert c1.span <= c2.span
            else:
                assert c1.importance > c2.importance


# ============================================================================
# Suite 4: Codebase Static Compliance (No Em-Dashes, No Emojis)
# ============================================================================

def test_no_em_dashes_or_emojis_in_source_code():
    """Verify that no em-dashes (U+2014) or emojis exist in src/, scripts/, tests/, web/."""
    import re

    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F]|"
        r"[\U0001F300-\U0001F5FF]|"
        r"[\U0001F680-\U0001F6FF]|"
        r"[\U0001F1E0-\U0001F1FF]|"
        r"[\U0001F900-\U0001F9FF]|"
        r"[\U0001FA70-\U0001FAFF]|"
        r"[\u2600-\u26FF]|"
        r"[\u2700-\u27BF]|"
        r"[\u2014]"
    )

    project_root = Path(__file__).resolve().parent.parent
    code_dirs = ["src", "scripts", "tests", "web"]
    violations = []

    for cdir in code_dirs:
        dir_path = project_root / cdir
        if not dir_path.exists():
            continue
        for p in dir_path.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"}:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        matches = emoji_pattern.findall(line)
                        if matches:
                            violations.append(f"{p}:{line_no}: {matches} in '{line.strip()}'")

    assert not violations, "Found em-dash or emoji violations in code:\n" + "\n".join(violations[:10])
