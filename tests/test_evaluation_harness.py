"""Tests for Milestone 3: Evaluation Harness (F8, F9, F10, F11).

Verifies Phi-4-mini adversary parsing, Presidio baseline redaction, hardness stratification,
dual utility metric calculation (MiniLM + DeBERTa NLI), and staged sequential runner with memory cleanup.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from src.evaluation.attacker import ATTRIBUTES, Phi4MiniAttacker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.run import StagedEvaluator, clean_gpu_memory
from src.evaluation.utility import UtilityEvaluator, compute_fallback_token_cosine


def test_attacker_prompt_formatting():
    """Verify adversary prompt templates format properly for all attributes."""
    attacker = Phi4MiniAttacker()
    sample_text = "I am an adult working in Denver, CO."

    for attr in ATTRIBUTES:
        prompt = attacker.format_prompt(sample_text, attr)
        assert sample_text in prompt
        assert attr in prompt or "author's" in prompt
        assert "<prediction>" in prompt
        assert "<reasoning>" in prompt

    with pytest.raises(ValueError):
        attacker.format_prompt(sample_text, "invalid_attribute")


def test_attacker_strict_and_lenient_parsing():
    """Verify strict XML parsing and lenient fallback parsing of adversary outputs."""
    attacker = Phi4MiniAttacker()

    # Strict tag parsing
    xml_resp = (
        "<reasoning>Author mentioned commuting on BART to silicon valley.</reasoning>\n"
        "<prediction>San Francisco</prediction>"
    )
    strict_pred, strict_reason = attacker.parse_strict_response(xml_resp)
    assert strict_pred == "San Francisco"
    assert "BART" in strict_reason

    # Lenient parsing when tags are absent
    line_resp = "Based on the text, Inferred: Denver, CO\nCues: altitude and mountain trails."
    lenient_pred, _ = attacker.parse_lenient_response(line_resp, "location")
    assert "Denver" in lenient_pred

    # Lenient keyword detection
    raw_unstructured = "The author is clearly a software engineer working with pull requests."
    job_pred, _ = attacker.parse_lenient_response(raw_unstructured, "occupation")
    assert job_pred == "Software engineer"


def test_attacker_predict_attribute_heuristic():
    """Verify genuine cue-based deduction when model weights are not loaded."""
    attacker = Phi4MiniAttacker()

    # Location cue deduction
    res_loc = attacker.predict_attribute("Enjoying the altitude and concerts at Red Rocks.", "location")
    assert res_loc["prediction"] == "Denver"
    assert attacker.evaluate_success(res_loc["prediction"], "Denver, CO")

    # Occupation cue deduction
    res_occ = attacker.predict_attribute("Reviewing the pull request before prod deployment.", "occupation")
    assert res_occ["prediction"] == "Software Engineer"
    assert attacker.evaluate_success(res_occ["prediction"], "Software Engineer")

    # Safe text
    res_safe = attacker.predict_attribute("The sunset was vibrant tonight.", "occupation")
    assert res_safe["prediction"] == "unknown"
    assert not attacker.evaluate_success(res_safe["prediction"], "Software Engineer")


def test_presidio_baseline_redaction():
    """Verify explicit PII is redacted with placeholder tags."""
    baseline = PresidioBaseline(use_presidio_if_available=False)

    text = "Contact me at alice@test.com or 303-555-0199 in Denver, CO."
    redacted = baseline.redact(text)

    assert "alice@test.com" not in redacted
    assert "<EMAIL>" in redacted
    assert "303-555-0199" not in redacted
    assert "<PHONE>" in redacted
    assert "<LOCATION>" in redacted


def test_presidio_hardness_stratification():
    """Verify classification of inference hardness into direct, indirect, and complicated."""
    baseline = PresidioBaseline(use_presidio_if_available=False)

    direct_text = "My email is bob@domain.org and I live in Boston."
    indirect_text = "The altitude takes getting used to when trail running near Red Rocks."
    complicated_text = "Email me at bob@domain.org about the altitude running near Red Rocks."

    assert baseline.classify_hardness(direct_text) == "direct"
    assert baseline.classify_hardness(indirect_text) == "indirect"
    assert baseline.classify_hardness(complicated_text) == "complicated"


def test_presidio_stratified_evaluation():
    """Verify stratified evaluation summary calculates explicit redaction rates."""
    baseline = PresidioBaseline(use_presidio_if_available=False)
    samples = [
        {"text": "Email is bob@domain.com."},
        {"text": "The altitude is high near the Rockies."},
        {"text": "Call 555-123-4567 regarding the Rockies trail."},
    ]

    summary = baseline.evaluate_stratified(samples)
    assert "direct" in summary
    assert "indirect" in summary
    assert "complicated" in summary
    assert summary["direct"]["count"] == 1
    assert summary["indirect"]["count"] == 1
    assert summary["complicated"]["count"] == 1


def test_utility_evaluator_metrics():
    """Verify dual utility metrics (cosine similarity and bidirectional NLI)."""
    utility = UtilityEvaluator()

    orig = "I moved to Seattle to join a tech startup as a designer."
    rewrite_good = "I relocated locally to join an organization in a creative role."
    metrics_good = utility.compute_metrics(orig, rewrite_good)

    assert 0.0 <= metrics_good["cosine_similarity"] <= 1.0
    assert 0.0 <= metrics_good["nli_forward"] <= 1.0
    assert 0.0 <= metrics_good["nli_backward"] <= 1.0
    assert metrics_good["contradiction"] == 0.0
    assert metrics_good["utility_score"] > 0.3

    # Contradiction scenario
    rewrite_bad = "I never moved anywhere and I do not work."
    metrics_bad = utility.compute_metrics(orig, rewrite_bad)
    assert metrics_bad["contradiction"] > 0.0
    assert metrics_bad["utility_score"] < metrics_good["utility_score"]


def test_utility_evaluator_acceptance():
    """Verify is_acceptable_utility enforces thresholds."""
    utility = UtilityEvaluator()

    passing_metrics = {
        "cosine_similarity": 0.85,
        "nli_forward": 0.80,
        "nli_backward": 0.75,
        "contradiction": 0.05,
        "utility_score": 0.78,
    }
    assert utility.is_acceptable_utility(passing_metrics, min_cosine=0.82) is True

    failing_cosine = dict(passing_metrics, cosine_similarity=0.75)
    assert utility.is_acceptable_utility(failing_cosine, min_cosine=0.82) is False

    failing_contra = dict(passing_metrics, contradiction=0.40)
    assert utility.is_acceptable_utility(failing_contra, min_cosine=0.82) is False


def test_staged_evaluator_pipeline():
    """Verify multi-stage evaluation pipeline runs sequentially and outputs structured report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = StagedEvaluator(output_dir=tmpdir)
        samples = [
            {
                "profile_id": "pers1",
                "text": "I am 24 years old and work as a software engineer in Denver near Red Rocks.",
                "ground_truth": {"age": "24", "location": "Denver", "occupation": "Software Engineer"},
            },
            {
                "profile_id": "pers2",
                "text": "Taking sound transit to Pike Place after my clinical shift at the hospital.",
                "ground_truth": {"location": "Seattle", "occupation": "Nurse"},
            },
        ]

        report = evaluator.run_pipeline(samples, report_filename="test_eval.json")

        assert report["sample_count"] == 2
        assert "adversary_evaluation" in report
        assert "utility_summary" in report
        assert "samples" in report

        asr = report["adversary_evaluation"]
        assert "original_asr" in asr
        assert "presidio_asr" in asr
        assert "inferenceguard_asr" in asr
        # InferenceGuard rewrite should achieve lower or equal attack success rate than original
        assert asr["inferenceguard_asr"] <= asr["original_asr"]

        # Check saved report file
        out_file = Path(tmpdir) / "test_eval.json"
        assert out_file.exists()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded["sample_count"] == 2


def test_clean_gpu_memory():
    """Verify GPU memory cleanup executes without exception."""
    clean_gpu_memory()


def test_zero_em_dashes_in_evaluation_modules():
    """Verify zero em-dashes (U+2014) across all evaluation modules."""
    eval_dir = Path("src/evaluation")
    for py_file in eval_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "\u2014" not in content, f"Em-dash found in {py_file}"
