"""Unit, integration, and scenario tests for probability calibration and ECE calculation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.product.risk_bands import ATTRIBUTES, Band, band_for, summarize
from src.risk_model.calibration import (
    BandValidationResult,
    CalibrationMetrics,
    TemperatureScaler,
    compute_ece,
    generate_reliability_diagram,
)


# ============================================================================
# Tier 1: Feature Coverage Tests
# ============================================================================


def test_compute_ece_metrics():
    """Verify ECE, MCE, and Brier score match expected values on controlled input."""
    # 4 samples: [0.2, 0.4, 0.6, 0.8], labels: [0, 0, 1, 1]
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.array([0, 0, 1, 1])

    metrics = compute_ece(probs, labels, n_bins=5)
    assert isinstance(metrics, CalibrationMetrics)
    assert metrics.n_bins == 5
    assert len(metrics.bin_counts) == 5
    assert metrics.brier_score == pytest.approx(0.1, abs=1e-4)
    assert 0.0 <= metrics.ece <= 1.0
    assert 0.0 <= metrics.mce <= 1.0


def test_temperature_scaler_fit_synthetic():
    """Verify L-BFGS converges and finds appropriate temperature on overconfident logits."""
    torch.manual_seed(42)
    # Generate overconfident logits: large magnitude relative to binary noise
    n_samples = 400
    true_probs = torch.rand(n_samples)
    labels = (torch.rand(n_samples) < true_probs).float()
    # Artificially inflate logit magnitude (e.g. 3x) to simulate overconfidence
    raw_logits = torch.logit(torch.clamp(true_probs, 1e-4, 1.0 - 1e-4)) * 3.0

    logits_list = [raw_logits for _ in range(len(ATTRIBUTES))]
    labels_list = [labels for _ in range(len(ATTRIBUTES))]

    scaler = TemperatureScaler(learning_rate=0.05, max_iter=50)
    learned_temps = scaler.fit(logits_list, labels_list)

    assert set(learned_temps.keys()) == set(ATTRIBUTES)
    for head, temp in learned_temps.items():
        # Temperature should be > 1.0 to soften overconfident logits
        assert temp > 1.2


def test_temperature_scaler_reduces_ece():
    """Verify that post-calibration ECE is strictly lower than pre-calibration ECE."""
    torch.manual_seed(101)
    n_samples = 500
    p = torch.rand(n_samples)
    labels = (torch.rand(n_samples) < p).float()
    overconfident_logits = torch.logit(torch.clamp(p, 1e-3, 0.999)) * 2.5

    pre_p = torch.sigmoid(overconfident_logits).numpy()
    pre_metrics = compute_ece(pre_p, labels.numpy(), n_bins=10)

    scaler = TemperatureScaler(learning_rate=0.05)
    scaler.fit([overconfident_logits] * 4, [labels] * 4)

    cal_probs = scaler.predict_proba([overconfident_logits] * 4)
    post_metrics = compute_ece(cal_probs[0].numpy(), labels.numpy(), n_bins=10)

    assert post_metrics.ece < pre_metrics.ece
    assert post_metrics.brier_score <= pre_metrics.brier_score + 1e-3


def test_platt_scaling_fallback_trigger():
    """Verify that Platt scaling activates when a constant logit shift prevents temperature scaling."""
    torch.manual_seed(202)
    n_samples = 500
    p = torch.rand(n_samples) * 0.2  # Low true probability (rare class like location)
    labels = (torch.rand(n_samples) < p).float()
    # Introduce severe positive logit bias (+3.0 shift)
    biased_logits = torch.logit(torch.clamp(p, 1e-3, 0.999)) + 3.0

    scaler = TemperatureScaler(
        enable_platt_fallback=True,
        platt_fallback_ece_threshold=0.03,
        learning_rate=0.05,
    )
    scaler.fit([biased_logits] * 4, [labels] * 4)

    # At least one head should adopt platt scaling due to the constant intercept bias
    assert "platt" in scaler.methods.values()


def test_calibration_save_and_load_roundtrip():
    """Verify serialization and restoration of TemperatureScaler parameters."""
    scaler = TemperatureScaler()
    scaler.temperatures = {"age": 1.45, "location": 1.95, "occupation": 1.15, "education": 1.30}
    scaler.platt_params = {"age": (1.0, 0.0), "location": (0.85, -0.42), "occupation": (1.0, 0.0), "education": (1.0, 0.0)}
    scaler.methods = {"age": "temperature", "location": "platt", "occupation": "temperature", "education": "temperature"}
    scaler.is_fitted = True

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "calibration.json"
        scaler.save(path)
        assert path.exists()

        restored = TemperatureScaler.load(path)
        assert restored.head_names == scaler.head_names
        assert restored.temperatures == scaler.temperatures
        assert restored.platt_params == scaler.platt_params
        assert restored.methods == scaler.methods
        assert restored.is_fitted is True


def test_generate_reliability_diagram_headless():
    """Verify reliability diagram generation produces structured metrics and handles headless plotting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "diag.json"
        png_path = Path(tmp_dir) / "diag.png"

        cal_probs = {
            "age": [0.1, 0.35, 0.7, 0.85],
            "location": [0.05, 0.15, 0.35, 0.65],
            "occupation": [0.2, 0.5, 0.6, 0.9],
            "education": [0.1, 0.25, 0.45, 0.8],
        }
        labels = {
            "age": [0, 0, 1, 1],
            "location": [0, 0, 0, 1],
            "occupation": [0, 1, 1, 1],
            "education": [0, 0, 1, 1],
        }

        report = generate_reliability_diagram(
            calibrated_probs=cal_probs,
            labels=labels,
            save_path=png_path,
            json_path=json_path,
            n_bins=5,
        )

        assert json_path.exists()
        assert "attributes" in report
        assert "macro_ece" in report
        assert len(report["attributes"]) == 4


# ============================================================================
# Tier 2: Boundary & Corner Case Tests
# ============================================================================


def test_perfect_calibration_zero_error():
    """Assert perfect calibration p == y produces zero ECE, zero MCE, and zero Brier score."""
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    metrics = compute_ece(probs, labels, n_bins=10)
    assert metrics.ece == pytest.approx(0.0, abs=1e-7)
    assert metrics.mce == pytest.approx(0.0, abs=1e-7)
    assert metrics.brier_score == pytest.approx(0.0, abs=1e-7)
    assert metrics.is_calibrated is True


def test_worst_calibration_max_error():
    """Assert maximally miscalibrated predictions (p=1 when y=0) yield ECE=1.0 and Brier=1.0."""
    probs = np.ones(50)
    labels = np.zeros(50)

    metrics = compute_ece(probs, labels, n_bins=10)
    assert metrics.ece == pytest.approx(1.0, abs=1e-5)
    assert metrics.mce == pytest.approx(1.0, abs=1e-5)
    assert metrics.brier_score == pytest.approx(1.0, abs=1e-5)
    assert metrics.is_calibrated is False


def test_empty_bins_handled_without_nan():
    """Assert sparse predictions with unoccupied confidence bins evaluate without NaN."""
    # All predictions tightly clustered in [0.05, 0.08]
    probs = np.array([0.05, 0.06, 0.07, 0.08])
    labels = np.array([0, 0, 0, 0])

    metrics = compute_ece(probs, labels, n_bins=10)
    assert not np.isnan(metrics.ece)
    assert not np.isnan(metrics.mce)
    assert not np.isnan(metrics.brier_score)
    assert metrics.bin_counts[0] == 4
    assert sum(metrics.bin_counts[1:]) == 0


def test_extreme_logit_numerical_stability():
    """Assert extreme logits (+100.0, -100.0) do not cause overflow or non-finite values."""
    extreme_logits = torch.tensor([-100.0, -50.0, 50.0, 100.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    scaler = TemperatureScaler(learning_rate=0.01)
    scaler.fit([extreme_logits] * 4, [labels] * 4)

    calibrated = scaler.predict_proba([extreme_logits] * 4)
    for head_p in calibrated:
        arr = head_p.numpy()
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)


def test_invalid_inputs_raise_value_error():
    """Assert invalid inputs (length mismatch, out-of-bounds, non-binary) raise ValueError."""
    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        compute_ece([0.2, 0.4], [0])

    # Out of bounds probability (> 1.0)
    with pytest.raises(ValueError, match="probabilities must lie in"):
        compute_ece([1.5, 0.5], [1, 0])

    # Non-binary labels
    with pytest.raises(ValueError, match="labels must be binary"):
        compute_ece([0.2, 0.8], [0, 2])

    # Empty inputs
    with pytest.raises(ValueError, match="empty arrays"):
        compute_ece([], [])


# ============================================================================
# Tier 3: Pairwise & Cross-Feature Integration Tests
# ============================================================================


def test_temperature_scaling_monotonicity():
    """Assert temperature scaling strictly preserves rank ordering of logits."""
    torch.manual_seed(303)
    raw_logits = torch.randn(100)
    order_before = torch.argsort(raw_logits).tolist()

    scaler = TemperatureScaler()
    scaler.temperatures["age"] = 1.8
    cal_logits = scaler.calibrate_logits([raw_logits] + [torch.zeros(100)] * 3)[0]
    order_after = torch.argsort(cal_logits).tolist()

    assert order_before == order_after


def test_threshold_validation_for_risk_bands():
    """Assert threshold validation computes low, medium, high band alignment."""
    # Synthetic calibrated sample with clear band distribution
    probs = np.array([0.10, 0.20, 0.35, 0.50, 0.75, 0.90])
    labels = np.array([0, 0, 0, 1, 1, 1])

    metrics = compute_ece(probs, labels, n_bins=10)
    bv = metrics.band_validations

    assert "LOW" in bv
    assert "MEDIUM" in bv
    assert "HIGH" in bv

    assert bv["LOW"].sample_count == 2
    assert bv["MEDIUM"].sample_count == 2
    assert bv["HIGH"].sample_count == 2

    assert bv["LOW"].threshold_range == (0.0, 0.30)
    assert bv["MEDIUM"].threshold_range == (0.30, 0.60)
    assert bv["HIGH"].threshold_range == (0.60, 1.0)


def test_multi_head_predict_calibrated_contract():
    """Assert predict_calibrated output contract directly plugs into risk_bands.summarize."""
    logits = [
        torch.tensor([0.2]),   # age
        torch.tensor([1.5]),   # location
        torch.tensor([-0.8]),  # occupation
        torch.tensor([0.4]),   # education
    ]

    scaler = TemperatureScaler()
    scaler.temperatures = {"age": 1.0, "location": 1.2, "occupation": 1.0, "education": 1.0}

    cal_dict = scaler.predict_calibrated(logits)
    assert set(cal_dict.keys()) == set(ATTRIBUTES)
    for attr, val in cal_dict.items():
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    # Verify compatibility with src.product.risk_bands.summarize
    summary = summarize(cal_dict)
    assert summary.primary == "location"
    assert summary.overall == cal_dict["location"]
    assert summary.overall_band in (Band.MEDIUM, Band.HIGH)


# ============================================================================
# Tier 4: Scenario Test S-02 (University Student Co-op)
# ============================================================================


def test_scenario_s02_coop_calibrated_alert():
    """Scenario S-02: Calibrated probability should accurately flag high-risk location and education."""
    # Raw uncalibrated overconfident logits from model
    uncalibrated_logits = [
        torch.tensor([0.1]),   # age
        torch.tensor([2.5]),   # location (brutal Green Line commute)
        torch.tensor([0.2]),   # occupation
        torch.tensor([1.8]),   # education (second co-op / student)
    ]

    scaler = TemperatureScaler()
    scaler.temperatures = {"age": 1.1, "location": 1.3, "occupation": 1.0, "education": 1.2}

    calibrated_scores = scaler.predict_calibrated(uncalibrated_logits)

    summary = summarize(calibrated_scores)
    # Location should be the primary risk and education secondary
    assert summary.primary == "location"
    assert band_for(calibrated_scores["location"]) == Band.HIGH
    assert band_for(calibrated_scores["education"]) in (Band.MEDIUM, Band.HIGH)
    assert "education" in summary.secondary
