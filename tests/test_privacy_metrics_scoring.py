"""Attacker scoring must count a genuinely correct inference as correct.

Locks down the measurement bug where every reported attacker accuracy was
0.0000 for every method including ``raw``. Ground truth in the test split is
free text ("41", "Bangkok, Thailand", "Masters in Business Administration",
"business consultant"), while the held-out adversary predicts coarse taxonomy
labels ("35-44", "asia_pacific", "master", "business_finance"). The metric used
to filter ground truth by ``t in valid_labels`` against the free text, which
dropped every pair and zeroed accuracy even for the unprotected ``raw`` text.

These tests pin the two halves of the fix:

* ``normalize_to_taxonomy_label`` collapses the real free-text ground truth into
  the taxonomy label space (the narrow, dependency-free check).
* ``compute_method_metrics`` turns a mock adversary's correct predictions into a
  NONZERO ``raw`` attack-success rate (the full scoring path).
"""

from __future__ import annotations

import pytest

from src.contracts.schemas import AttributeName
from src.risk_model.taxonomy import normalize_to_taxonomy_label


# Real ground-truth shape from data/splits/test.jsonl: free-text strings keyed
# by the lowercase attribute name.
REAL_GT = {
    "age": "41",
    "location": "Bangkok, Thailand",
    "occupation": "business consultant",
    "education": "Masters in Business Administration",
}

# The coarse taxonomy labels those free-text values collapse to; this is the
# space the held-out adversary predicts in.
EXPECTED_LABELS = {
    AttributeName.AGE: "35-44",
    AttributeName.LOCATION: "asia_pacific",
    AttributeName.OCCUPATION: "business_finance",
    AttributeName.EDUCATION: "master",
}


def test_normalize_free_text_ground_truth_to_taxonomy_labels():
    """Free-text ground truth maps onto the taxonomy label space, not UNKNOWN."""
    for attr, expected in EXPECTED_LABELS.items():
        assert normalize_to_taxonomy_label(attr, REAL_GT[attr.value]) == expected


def test_normalize_blank_ground_truth_is_unknown():
    """A missing/blank ground-truth value abstains rather than inventing a class."""
    assert normalize_to_taxonomy_label(AttributeName.AGE, "") == "UNKNOWN"
    assert normalize_to_taxonomy_label(AttributeName.LOCATION, None) == "UNKNOWN"


def test_raw_attack_accuracy_is_nonzero_when_adversary_is_correct():
    """A correct inference against free-text ground truth is actually counted.

    Builds the exact predictions/ground-truth path that
    ``run_comparative_benchmark`` feeds into ``compute_method_metrics`` for the
    ``raw`` method, with a mock adversary returning the correct coarse label for
    every attribute on two records. Before the fix the taxonomy-membership
    filter dropped every free-text pair and the attack-success rate was 0.0; now
    it must be 1.0.
    """
    # numpy/sklearn are project deps used by compute_method_metrics; skip
    # cleanly in minimal environments that lack them (the sandbox), run in CI.
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from src.evaluation.privacy_metrics import compute_method_metrics

    records = [{"attributes": dict(REAL_GT)}, {"attributes": dict(REAL_GT)}]

    # Mock adversary: returns the correct coarse taxonomy label per attribute,
    # exactly as HeldOutPhi4Adversary.infer_attributes would on a correct guess.
    predictions_by_attr = {attr: [EXPECTED_LABELS[attr]] * len(records) for attr in AttributeName}
    # Ground truth as the harness collects it: raw free text via attrs_gt.get.
    ground_truth_by_attr = {
        attr: [rec["attributes"].get(attr.value, "UNKNOWN") for rec in records]
        for attr in AttributeName
    }

    metrics = compute_method_metrics(
        records=records,
        predictions_by_attr=predictions_by_attr,
        ground_truth_by_attr=ground_truth_by_attr,
        utility_scores=[{"bidirectional_entailment": 1.0}] * len(records),
        latencies_ms=[1.0] * len(records),
    )

    assert metrics["overall_attack_success_rate"] > 0.0
    assert metrics["overall_attack_success_rate"] == pytest.approx(1.0)
    # Every attribute must be scored (not skipped as an empty valid_pairs set).
    assert set(metrics["per_attribute"].keys()) == {a.value for a in AttributeName}
    for attr in AttributeName:
        assert metrics["per_attribute"][attr.value]["attack_accuracy"] == pytest.approx(1.0)


def test_wrong_prediction_is_not_counted_correct():
    """A wrong coarse inference is not silently scored as a hit."""
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from src.evaluation.privacy_metrics import compute_method_metrics

    records = [{"attributes": dict(REAL_GT)}]
    # Deliberately wrong: predict a different valid taxonomy label for each attr.
    wrong = {
        AttributeName.AGE: "18-24",
        AttributeName.LOCATION: "us_west",
        AttributeName.OCCUPATION: "healthcare",
        AttributeName.EDUCATION: "high_school",
    }
    predictions_by_attr = {attr: [wrong[attr]] for attr in AttributeName}
    ground_truth_by_attr = {
        attr: [records[0]["attributes"].get(attr.value, "UNKNOWN")] for attr in AttributeName
    }

    metrics = compute_method_metrics(
        records=records,
        predictions_by_attr=predictions_by_attr,
        ground_truth_by_attr=ground_truth_by_attr,
        utility_scores=[{"bidirectional_entailment": 1.0}],
        latencies_ms=[1.0],
    )
    assert metrics["overall_attack_success_rate"] == pytest.approx(0.0)
