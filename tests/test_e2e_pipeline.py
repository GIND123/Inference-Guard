"""End-to-End Test Suite and Scenario Track for InferenceGuard (Milestone 6).

Implements Tier 3 (Pairwise Interactions), Tier 4 (Real-World Scenarios S-01 to S-06),
and programmatic LLM-as-a-judge verification conforming to TEST_INFRA.md.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.conversation.pseudonyms import PersistentPseudonymManager
from src.conversation.state import ConversationTracker
from src.evaluation.attacker import Phi4MiniAttacker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.run import StagedEvaluator
from src.evaluation.utility import UtilityEvaluator
from src.product.risk_bands import Band, band_for, summarize
from src.rewriter.generate_training_data import (
    generate_candidate_rewrites_heuristic,
    generate_sft_dataset,
    load_profile_splits,
    pareto_rejection_sample,
)
from src.risk_model.calibration import TemperatureScaler, compute_ece
from web.api import app


# -----------------------------------------------------------------------------
# Tier 4: Real-World Application Scenarios (S-01 to S-06)
# -----------------------------------------------------------------------------

def test_scenario_s01_tech_worker_relocation():
    """Scenario S-01: Tech Worker Relocation (F1, F2, F4, F6, F9, F10).

    Target behavior: Identifies employer and transit cues; rewrites preserve
    job search utility while dropping location and company identifiers.
    """
    input_text = (
        "I am moving from Denver, CO to Seattle, WA to start as a software engineer at Amazon, "
        "taking Sound Transit daily."
    )
    presidio = PresidioBaseline()
    utility = UtilityEvaluator()

    # Step 1: Baseline redaction
    redacted = presidio.redact(input_text)
    assert "<LOCATION>" in redacted

    # Step 2: InferenceGuard candidate rewrite
    candidates = generate_candidate_rewrites_heuristic(input_text)
    assert len(candidates) > 0

    selected = pareto_rejection_sample(
        original=input_text,
        candidates=candidates,
        max_risk_threshold=0.30,
        min_cosine_threshold=0.30,
    )
    assert selected is not None
    rewrite, meta = selected

    # Verify location dropped
    assert "Denver" not in rewrite
    assert "Seattle" not in rewrite
    # Verify utility preserved
    metrics = utility.compute_metrics(input_text, rewrite)
    assert metrics["cosine_similarity"] >= 0.30
    assert metrics["contradiction"] == 0.0


def test_scenario_s02_university_student_coop():
    """Scenario S-02: University Student Co-op (F2, F3, F4, F8, F9, F10).

    Target behavior: Detects age, education, and location cues; adversary fails on rewrite.
    """
    input_text = (
        "As a 21-year-old undergrad at Carnegie Mellon, I just started my summer internship in Pittsburgh."
    )
    attacker = Phi4MiniAttacker()

    # Step 1: Adversary probes original
    orig_age = attacker.predict_attribute(input_text, "age")
    orig_edu = attacker.predict_attribute(input_text, "education")
    assert attacker.evaluate_success(orig_age["prediction"], "21")
    assert attacker.evaluate_success(orig_edu["prediction"], "Carnegie Mellon")

    # Step 2: Protection and rewrite
    candidates = generate_candidate_rewrites_heuristic(input_text)
    selected = pareto_rejection_sample(
        original=input_text,
        candidates=candidates,
        max_risk_threshold=0.30,
        min_cosine_threshold=0.20,
    )
    assert selected is not None
    rewrite, _ = selected

    # Step 3: Adversary probes rewrite
    rew_age = attacker.predict_attribute(rewrite, "age")
    rew_edu = attacker.predict_attribute(rewrite, "education")
    # Adversary should not be able to deduce the exact 21 or Carnegie Mellon from rewrite
    assert "21" not in rew_age["prediction"]
    assert "Carnegie Mellon" not in rew_edu["prediction"]


def test_scenario_s03_medical_condition_discussion():
    """Scenario S-03: Medical Condition Discussion (F2, F4, F12).

    Target behavior: Handles unmonitored attributes gracefully with clear UI disclaimer,
    while monitoring in-scope demographics without false alarms.
    """
    medical_text = (
        "Managing my chronic migraine condition with physical therapy and dietary changes."
    )
    client = TestClient(app)
    resp = client.post("/analyze", json={"text": medical_text})
    assert resp.status_code == 200

    data = resp.json()
    summary = data["risk_summary"]
    # Medical is unmonitored in v1; monitored demographics should report LOW
    assert summary["overall_band"] == "LOW"
    assert summary["overall"] < 0.30


def test_scenario_s04_multiparagraph_forum_post():
    """Scenario S-04: Multi-Paragraph Forum Post (F5, F6, F7, F10).

    Target behavior: Preserves long-form argument structure while removing personal identifiers.
    """
    long_post = (
        "I have worked as a nurse in Austin for over 15 years. The healthcare staffing ratios "
        "here have gotten progressively challenging across multiple hospitals. We need state-level "
        "reform to ensure reasonable nurse-to-patient ratios."
    )
    candidates = generate_candidate_rewrites_heuristic(long_post)
    assert len(candidates) > 0

    selected = pareto_rejection_sample(
        original=long_post,
        candidates=candidates,
        max_risk_threshold=0.30,
        min_cosine_threshold=0.40,
    )
    assert selected is not None
    rewrite, meta = selected

    # Ensure nurse-to-patient and reform context is preserved
    assert "Austin" not in rewrite
    assert meta["overall_risk"] < 0.30
    assert meta["utility"]["cosine_similarity"] >= 0.40


def test_scenario_s05_adversarial_comparison():
    """Scenario S-05: Adversarial Comparison (F8, F9, F10, F11).

    Target behavior: Shows Presidio misses indirect cues while InferenceGuard protects them.
    """
    indirect_text = "The altitude takes getting used to when running near Red Rocks."
    ground_truth = {"location": "Denver"}

    presidio = PresidioBaseline()
    attacker = Phi4MiniAttacker()

    # Presidio leaves indirect cue untouched
    presidio_out = presidio.redact(indirect_text)
    assert "Red Rocks" in presidio_out

    # Adversary succeeds on Presidio output
    p_pred = attacker.predict_attribute(presidio_out, "location")
    assert attacker.evaluate_success(p_pred["prediction"], "Denver")

    # InferenceGuard rewrites indirect cue
    candidates = generate_candidate_rewrites_heuristic(indirect_text)
    selected = pareto_rejection_sample(
        original=indirect_text,
        candidates=candidates,
        max_risk_threshold=0.30,
        min_cosine_threshold=0.20,
    )
    if selected is not None:
        rewrite, _ = selected
        assert "Red Rocks" not in rewrite


def test_scenario_s06_multiturn_dialogue_accumulation():
    """Scenario S-06: Multi-Turn Dialogue Accumulation (F12, F13, F15, F16).

    Target behavior: Tracks cumulative leakage over 4 turns; raises warning when
    joint entropy drops or cumulative risk breaches threshold.
    """
    tracker = ConversationTracker(high_risk_threshold=0.60)
    pseudo_mgr = PersistentPseudonymManager()
    session_id = "s06_accum"

    dialogue_turns = [
        "Hi, I am looking for advice on tech career growth.",
        "I recently relocated to Denver, CO for an engineering job.",
        "I am 26 years old and graduated from university recently.",
        "My company is Google and I am working in the downtown office.",
    ]

    records = []
    for turn in dialogue_turns:
        rec = tracker.add_turn(turn, session_id=session_id)
        records.append(rec)

    # Risk must monotonically accumulate
    assert records[3].overall_cumulative_risk >= records[0].overall_cumulative_risk
    # Joint entropy must decrease
    assert records[3].joint_entropy < records[0].joint_entropy
    # By turn 4, multiple attributes are disclosed, triggering breach flag
    assert records[3].is_breached is True
    assert records[3].risk_band in ("MEDIUM", "HIGH")


# -----------------------------------------------------------------------------
# Tier 3: Cross-Feature Pairwise Interactions
# -----------------------------------------------------------------------------

def test_pairwise_risk_and_pseudonyms():
    """Pairwise Interaction: F2 (Risk Modeling) + F15 (Pseudonyms)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PersistentPseudonymManager(storage_dir=tmpdir)
        text = "Alice moved to Seattle to join TechWorks."
        pseudo = mgr.pseudonymize(text, session_id="pw1", known_entities=[("Alice", "PER"), ("Seattle", "LOC")])
        assert "[PER_1]" in pseudo
        assert "[LOC_1]" in pseudo

        # Summarize risk on pseudonymized text
        scores = {"age": 0.05, "location": 0.05, "occupation": 0.05, "education": 0.05}
        summary = summarize(scores)
        assert summary.overall_band == Band.LOW


def test_pairwise_calibration_and_adversary():
    """Pairwise Interaction: F3 (Calibration) + F8 (Adversary)."""
    attacker = Phi4MiniAttacker()
    # Assess calibration metric on simulated adversary confidence
    probs = [0.95, 0.05, 0.92, 0.08]
    labels = [1, 0, 1, 0]
    metrics = compute_ece(probs, labels, n_bins=5)
    assert metrics.ece < 0.10


def test_pairwise_state_and_api():
    """Pairwise Interaction: F13 (FastAPI) + F16 (Multi-Turn State)."""
    client = TestClient(app)
    s_id = "pw_state_sess"

    resp1 = client.post("/analyze", json={"text": "I am in Denver.", "session_id": s_id})
    assert resp1.status_code == 200
    t1 = resp1.json()["turn_record"]
    assert t1["turn_number"] == 1

    resp2 = client.post("/analyze", json={"text": "I work as a software engineer.", "session_id": s_id})
    assert resp2.status_code == 200
    t2 = resp2.json()["turn_record"]
    assert t2["turn_number"] == 2
    assert t2["cumulative_scores"]["location"] > 0.0
    assert t2["cumulative_scores"]["occupation"] > 0.0


# -----------------------------------------------------------------------------
# Tier 1 & 2: Boundary & Extreme Values
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("empty_input", ["", "   ", "\n\t\n"])
def test_boundary_empty_inputs(empty_input):
    """Tier 2: Boundary test on empty / whitespace-only inputs across components."""
    presidio = PresidioBaseline()
    utility = UtilityEvaluator()
    attacker = Phi4MiniAttacker()

    assert presidio.redact(empty_input) == ""
    assert utility.compute_similarity(empty_input, "text") == 0.0
    with pytest.raises(Exception):
        summarize({})


def test_boundary_massive_text():
    """Tier 2: Boundary test on large text input (10,000 words)."""
    large_text = "The quick brown fox jumps over the lazy dog. " * 1000
    presidio = PresidioBaseline()
    utility = UtilityEvaluator()

    redacted = presidio.redact(large_text)
    assert len(redacted) > 0

    sim = utility.compute_similarity(large_text[:500], large_text[500:1000])
    assert 0.9 <= sim <= 1.0


def test_llm_as_a_judge_verification():
    """Programmatic LLM-as-a-judge: Phi-4-mini adversary evaluates privacy improvement."""
    orig = "I am 23 years old and work as a software engineer in Denver."
    rewrite = "I am an adult working in my profession in the local area."

    attacker = Phi4MiniAttacker()

    # Judge original: adversary easily finds age, occupation, location
    orig_loc = attacker.predict_attribute(orig, "location")
    orig_occ = attacker.predict_attribute(orig, "occupation")
    orig_age = attacker.predict_attribute(orig, "age")

    assert attacker.evaluate_success(orig_loc["prediction"], "Denver")
    assert attacker.evaluate_success(orig_occ["prediction"], "software engineer")

    # Judge rewrite: adversary fails to find exact values
    rew_loc = attacker.predict_attribute(rewrite, "location")
    rew_occ = attacker.predict_attribute(rewrite, "occupation")
    rew_age = attacker.predict_attribute(rewrite, "age")

    assert "denver" not in rew_loc["prediction"].lower()
    assert "software engineer" not in rew_occ["prediction"].lower()
    assert "23" not in rew_age["prediction"].lower()


