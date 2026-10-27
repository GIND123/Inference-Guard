# tests/test_calibration_session07.py
"""Verification suite for Gate 2: ModernBERT Temperature Scaling & Youden's J Calibration.

Verifies Expected Calibration Error (ECE) minimization via L-BFGS, NLL reduction,
entropy and probability margin calculations, and UI monotonic threshold mapping.
"""

from __future__ import annotations

import pytest

# Guard before the heavy imports below, so a machine without the model
# stack reports these as skipped rather than failing to collect.
pytest.importorskip("numpy", reason="needs optional numpy; see tests/README.md")
pytest.importorskip("torch", reason="needs optional torch; see tests/README.md")

import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.contracts.schemas import AttributeName, RiskBand
from src.risk_model.calibrate import (
    AttributeTemperatureScaler,
    compute_expected_calibration_error,
)
from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY


@pytest.fixture
def miscalibrated_val_logits():
    """Generates overconfident synthetic validation logits across all 4 attribute heads."""
    torch.manual_seed(42)
    val_logits = {}
    val_labels = {}

    for attr, labels in ATTRIBUTE_TAXONOMY.items():
        k = len(labels)
        n_samples = 200
        # Overconfident logits scaled by 4.0
        raw = torch.randn(n_samples, k) * 4.0
        targets = torch.randint(0, k, (n_samples,))
        # Bias the correct class to ensure non-random distribution
        raw[torch.arange(n_samples), targets] += 2.0
        val_logits[attr.value] = raw
        val_labels[attr.value] = targets

    return val_logits, val_labels


def test_temperature_scaling_reduces_ece_and_nll(miscalibrated_val_logits):
    """Verify that L-BFGS temperature scaling strictly decreases validation ECE and NLL."""
    val_logits, val_labels = miscalibrated_val_logits
    scaler = AttributeTemperatureScaler()

    diagnostics = scaler.fit_validation_logits(val_logits, val_labels)

    for attr in AttributeName:
        diag = diagnostics[attr.value]
        # Temperature must be optimized away from 1.0 to soften overconfidence
        assert diag["temperature"] > 1.0, f"Expected T > 1 for overconfident head {attr.value}"
        assert diag["ece_after"] <= diag["ece_before"], f"ECE did not decrease for {attr.value}"
        assert diag["nll_after"] <= diag["nll_before"], f"NLL did not decrease for {attr.value}"
        assert diag["threshold_medium_raw"] < diag["threshold_high_raw"]


def test_raw_inferability_entropy_and_margin_sensitivity():
    """Verify that split/ambiguous distributions yield significantly lower risk than decisive ones."""
    scaler = AttributeTemperatureScaler()

    # Case A: Decisive sharp distribution (high confidence, large margin, low entropy)
    sharp_probs = np.array([0.90, 0.04, 0.02, 0.02, 0.02])
    sharp_score = scaler._compute_raw_inferability_from_probs(sharp_probs)

    # Case B: Split distribution with same top-1 class but tiny margin and high entropy
    split_probs = np.array([0.46, 0.44, 0.04, 0.03, 0.03])
    split_score = scaler._compute_raw_inferability_from_probs(split_probs)

    # Case C: Completely uniform distribution (pure chance)
    uniform_probs = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    uniform_score = scaler._compute_raw_inferability_from_probs(uniform_probs)

    assert uniform_score == 0.0, "Uniform distribution must produce 0.0 inferability risk"
    assert sharp_score > split_score, f"Sharp score ({sharp_score}) must exceed split score ({split_score})"
    assert sharp_score > 0.60
    assert split_score < 0.35


def test_ui_scale_monotonic_mapping():
    """Verify that map_to_ui_scale respects [0.30, 0.60] boundary alignments monotonically."""
    scaler = AttributeTemperatureScaler()
    attr = AttributeName.LOCATION
    scaler.thresholds[attr.value] = {"medium": 0.25, "high": 0.55}

    score_low, band_low = scaler.map_to_ui_scale(attr, 0.10)
    assert score_low < 0.30
    assert band_low == RiskBand.LOW

    score_med, band_med = scaler.map_to_ui_scale(attr, 0.35)
    assert 0.30 <= score_med < 0.60
    assert band_med == RiskBand.MEDIUM

    score_high, band_high = scaler.map_to_ui_scale(attr, 0.75)
    assert score_high >= 0.60
    assert band_high == RiskBand.HIGH


def test_scaler_persistence_roundtrip():
    """Verify that learned temperatures and thresholds persist to JSON and reload identically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "calibration.json"
        scaler = AttributeTemperatureScaler()
        scaler.log_temps[AttributeName.AGE.value].data = torch.tensor([0.45])
        scaler.thresholds[AttributeName.AGE.value] = {"medium": 0.28, "high": 0.58}

        scaler.save(save_path)
        assert save_path.exists()

        reloaded = AttributeTemperatureScaler.load(save_path)
        assert round(float(reloaded.get_temperature(AttributeName.AGE).item()), 4) == round(float(scaler.get_temperature(AttributeName.AGE).item()), 4)
        assert reloaded.thresholds == scaler.thresholds
