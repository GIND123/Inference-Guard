"""Tests for fitting the user-facing band edges from validation data."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.risk_bands import Band  # noqa: E402
from src.product.thresholds import (  # noqa: E402
    band_for,
    expected_calibration_error,
    fit_attribute,
    fit_thresholds,
    summarize,
)


def separable(n=400, sep=0.45, seed=0):
    """Scores a decent classifier would produce: positives shifted upward."""
    rng = random.Random(seed)
    y, p = [], []
    for i in range(n):
        pos = i % 4 == 0
        y.append(int(pos))
        centre = 0.5 + sep / 2 if pos else 0.5 - sep / 2
        p.append(min(1.0, max(0.0, rng.gauss(centre, 0.12))))
    return y, p


def test_edges_are_ordered_and_bands_follow_them():
    y, p = separable()
    fit = fit_attribute("occupation", y, p)
    assert 0.0 <= fit.medium_edge < fit.high_edge <= 1.0
    assert band_for(fit.high_edge, fit) is Band.HIGH
    assert band_for(fit.medium_edge, fit) is Band.MEDIUM
    assert band_for(max(0.0, fit.medium_edge - 0.01), fit) is Band.LOW


def test_high_band_meets_its_precision_target():
    y, p = separable()
    fit = fit_attribute("occupation", y, p, high_precision=0.60)
    assert fit.high_precision >= 0.60
    assert not fit.degraded


def test_raising_the_precision_target_raises_the_high_edge():
    y, p = separable()
    lenient = fit_attribute("occupation", y, p, high_precision=0.50)
    strict = fit_attribute("occupation", y, p, high_precision=0.90)
    assert strict.high_edge >= lenient.high_edge


def test_different_base_rates_produce_different_edges():
    """The reason thresholds are per-attribute and not global."""
    common_y, common_p = separable(n=400, sep=0.45, seed=1)
    rare_y, rare_p = [], []
    rng = random.Random(2)
    for i in range(400):  # ~4% positives, like location
        pos = i % 25 == 0
        rare_y.append(int(pos))
        rare_p.append(min(1.0, max(0.0, rng.gauss(0.72 if pos else 0.48, 0.12))))
    fits = fit_thresholds(
        {"occupation": common_y, "location": rare_y},
        {"occupation": common_p, "location": rare_p},
    )
    assert fits["occupation"].high_edge != fits["location"].high_edge


def test_thin_positive_support_is_flagged_unusable():
    """Location has ~48 positives in the test split; 10 must not drive a badge."""
    y = [1 if i % 40 == 0 else 0 for i in range(400)]
    p = [0.8 if t else 0.2 for t in y]
    fit = fit_attribute("location", y, p)
    assert fit.n_positive == 10
    assert not fit.usable


def test_unreachable_precision_degrades_loudly_rather_than_inventing_an_edge():
    """A model with no signal must not silently get plausible-looking bands."""
    rng = random.Random(3)
    y = [i % 4 == 0 for i in range(400)]
    p = [rng.random() for _ in y]  # scores independent of the label
    fit = fit_attribute("age", [int(t) for t in y], p, high_precision=0.95)
    assert fit.degraded
    assert "precision" in fit.note
    assert not fit.usable


def test_no_positives_is_degraded_not_a_crash():
    fit = fit_attribute("income", [0] * 50, [0.1] * 50)
    assert fit.degraded and fit.n_positive == 0
    assert "no positive examples" in fit.note


def test_edges_never_collapse_into_each_other():
    y = [1 if i % 2 else 0 for i in range(200)]
    p = [0.9 if t else 0.1 for t in y]  # perfectly separable
    fit = fit_attribute("occupation", y, p)
    assert fit.high_edge - fit.medium_edge >= 0.05


@pytest.mark.parametrize(
    "bad", [([1, 0], [0.5]), ([], []), ([1, 0], [0.5, 1.4])]
)
def test_malformed_input_raises(bad):
    with pytest.raises(ValueError):
        fit_attribute("age", bad[0], bad[1])


def test_fit_thresholds_ignores_unknown_attributes_and_needs_one_known():
    y, p = separable(n=100)
    fits = fit_thresholds({"occupation": y, "shoe_size": y}, {"occupation": p, "shoe_size": p})
    assert set(fits) == {"occupation"}
    with pytest.raises(ValueError):
        fit_thresholds({"shoe_size": y}, {"shoe_size": p})


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    # 100 bundles of 10, where a claimed probability of k/10 is right k times.
    y, p = [], []
    for k in range(11):
        for i in range(100):
            y.append(1 if i % 10 < k else 0)
            p.append(k / 10)
    assert expected_calibration_error(y, p) < 0.01


def test_ece_catches_systematic_overconfidence():
    y = [1 if i % 10 == 0 else 0 for i in range(500)]  # true rate 0.10
    p = [0.95] * 500                                    # model claims 0.95
    assert expected_calibration_error(y, p) > 0.8


def test_summary_names_the_caveat_not_just_the_number():
    fits = {"location": fit_attribute("location", [1 if i % 40 == 0 else 0 for i in range(400)],
                                      [0.8 if i % 40 == 0 else 0.2 for i in range(400)])}
    out = summarize(fits)
    assert "location" in out and "not usable" in out
