"""Empirical stress tests for PII veto domain contract and API response composition.

Executed by challenger_2 to verify:
1. Explicit PII (email, SSN, phone) in candidate rewrite forces is_win=False and verdict='explicit_pii_remains'.
2. Clean inputs evaluate standard win conditions correctly (improved, no_change, improved_with_cost, worse).
3. Response structure preserves all required fields (risk_summary, rewritten_text, utility_metrics, turn_record, etc.).
4. Adversarial edge cases and boundary conditions.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.product.thresholds import PROVISIONAL_MAX_RISK_THRESHOLD
from web import api
from web.api import app


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# SECTION 1: Explicit PII Veto Stress Tests (/analyze)
# ==============================================================================


def test_analyze_retained_email_vetoes_win(client, monkeypatch):
    """Verify that candidate rewrite retaining an EMAIL vetoes win: is_win=False, verdict='explicit_pii_remains'."""
    raw_text = "I am a 28 years old accountant living in Chicago, IL. Contact me at accountant_dan@example.com."
    leaked_rewrite = "I am an adult working as a professional in the area. Contact me at accountant_dan@example.com."

    # Monkeypatch pareto_rejection_sample so the selected rewrite candidate retains the email
    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (leaked_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "pii_email_test"})
    assert resp.status_code == 200
    data = resp.json()

    # Rule 2 invariants
    assert data["is_win"] is False
    assert data["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["is_win"] is False
    assert data["comparison"]["hard_pii_remaining"] >= 1
    assert any("identifying span(s) still present" in n for n in data["comparison"]["notes"])


def test_analyze_retained_ssn_vetoes_win(client, monkeypatch):
    """Verify that candidate rewrite retaining an SSN vetoes win: is_win=False, verdict='explicit_pii_remains'."""
    raw_text = "I am 35 years old living in Boston, MA. My SSN is 012-34-5678."
    leaked_rewrite = "An adult lives in the area. SSN: 012-34-5678."

    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (leaked_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "pii_ssn_test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_win"] is False
    assert data["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["is_win"] is False
    assert data["comparison"]["hard_pii_remaining"] >= 1


def test_analyze_retained_phone_vetoes_win(client, monkeypatch):
    """Verify that candidate rewrite retaining a PHONE NUMBER vetoes win: is_win=False, verdict='explicit_pii_remains'."""
    raw_text = "I am 42 years old living in Austin, TX. Reach me at 555-867-5309."
    leaked_rewrite = "An adult lives in the area. Reach out at 555-867-5309."

    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (leaked_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "pii_phone_test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_win"] is False
    assert data["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["is_win"] is False
    assert data["comparison"]["hard_pii_remaining"] >= 1


def test_analyze_retained_multiple_pii_types_vetoes_win(client, monkeypatch):
    """Verify that candidate rewrite retaining multiple PII types (email, SSN, phone) vetoes win."""
    raw_text = "I am 30 years old. Email: user@test.com, Phone: 303-555-1234, SSN: 999-88-7777."
    leaked_rewrite = "An adult. Email: user@test.com, Phone: 303-555-1234, SSN: 999-88-7777."

    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (leaked_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "pii_multi_test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_win"] is False
    assert data["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["hard_pii_remaining"] >= 3


def test_analyze_natural_heuristic_ssn_leakage_vetoes_win(client):
    """Natural stress test: Heuristic rewriter without mock.
    
    Verifies that if the rewrite candidate retains the SSN,
    the API's explicit PII span extractor detects the SSN and vetoes the win.
    """
    raw_text = "I am 26 years old living in Denver, CO and my SSN is 123-45-6789."
    resp = client.post("/analyze", json={"text": raw_text, "session_id": "natural_ssn_test"})
    assert resp.status_code == 200
    data = resp.json()

    if data["comparison"]["hard_pii_remaining"] > 0:
        assert data["is_win"] is False
        assert data["verdict"] == "explicit_pii_remains"
        assert data["comparison"]["verdict"] == "explicit_pii_remains"
    else:
        assert data["verdict"] in ("improved", "improved_with_cost")


def test_analyze_adversarial_inferential_drop_gaming_blocked(client, monkeypatch):
    """Stress test: Inferential risk plunges from 0.95 to 0.05, but 1 email remains.
    
    Verifies Rule 2: No inferential improvement compensates for explicit PII.
    """
    raw_text = "I am a 26 years old doctor working at Denver General Hospital in Denver, CO. Contact doc@clinic.org."
    # Candidate with near-zero inferential risk, but retaining the email
    leaked_rewrite = "A generic person working somewhere. Contact doc@clinic.org."

    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (leaked_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "gaming_test"})
    assert resp.status_code == 200
    data = resp.json()

    # Even if inferential delta is large, is_win MUST be False
    assert data["is_win"] is False
    assert data["verdict"] == "explicit_pii_remains"
    assert data["comparison"]["inferential_delta"] > 0.30
    assert data["comparison"]["hard_pii_remaining"] >= 1


# ==============================================================================
# SECTION 2: Clean Inputs & Standard Win Conditions (/analyze)
# ==============================================================================


def test_analyze_clean_scrubbed_input_wins(client, monkeypatch):
    """Verify standard win condition: inferential risk reduced, PII scrubbed, utility >= floor -> is_win=True."""
    raw_text = "I am 26 years old living in Denver, CO and working as a software engineer."
    # Clean rewrite that preserves meaning (utility >= 0.70) while dropping demographic cues
    clean_rewrite = "I am living in the area and working as a software engineer."

    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (clean_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "clean_win_test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["comparison"]["hard_pii_remaining"] == 0
    assert data["utility_metrics"]["utility_score"] >= 0.70
    assert data["verdict"] == "improved"
    assert data["is_win"] is True
    assert data["comparison"]["is_win"] is True
    assert data["comparison"]["inferential_delta"] > 0.01


def test_analyze_clean_low_risk_passthrough(client):
    """Verify clean input with LOW risk passes through unchanged: verdict='no_change', is_win=False."""
    raw_text = "The rapid development of technology is transforming various aspects of daily life."
    resp = client.post("/analyze", json={"text": raw_text, "session_id": "passthrough_test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["risk_summary"]["overall_band"] == "LOW"
    assert data["rewritten_text"] == raw_text
    assert data["verdict"] == "no_change"
    assert data["is_win"] is False
    assert data["comparison"]["hard_pii_remaining"] == 0


def test_analyze_improved_with_cost_when_utility_below_floor(client, monkeypatch):
    """Verify that when inferential risk is reduced but utility < floor (0.70), verdict='improved_with_cost', is_win=False."""
    raw_text = "I am a 26 years old software engineer living in Denver, CO."
    # Candidate with low utility score
    unrelated_rewrite = "The galaxy contains billions of stars and planets."

    monkeypatch.setattr(api, "pareto_rejection_sample", lambda **kwargs: (unrelated_rewrite, {}))

    resp = client.post("/analyze", json={"text": raw_text, "session_id": "low_utility_test"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["comparison"]["hard_pii_remaining"] == 0
    # Inferential risk dropped, but utility < 0.70
    assert data["verdict"] == "improved_with_cost"
    assert data["is_win"] is False
    assert any("utility" in n and "below" in n for n in data["comparison"]["notes"])


# ==============================================================================
# SECTION 3: Response Structure Preservation & Contract Integrity
# ==============================================================================


def test_analyze_response_structure_completeness(client):
    """Verify response structure preserves all required fields according to PROJECT.md and web UI expectations."""
    raw_text = "I work as a software engineer in Denver, Colorado."
    resp = client.post("/analyze", json={"text": raw_text, "session_id": "schema_test"})
    assert resp.status_code == 200
    data = resp.json()

    # 1. risk_summary
    assert "risk_summary" in data
    risk = data["risk_summary"]
    assert "overall" in risk
    assert "overall_band" in risk
    assert "primary" in risk
    assert "scores" in risk
    assert "cues" in risk
    assert "leakage_delta" in risk
    assert isinstance(risk["scores"], dict)

    # 2. rewritten_text & presidio_text
    assert "rewritten_text" in data
    assert "presidio_text" in data
    assert isinstance(data["rewritten_text"], str)
    assert isinstance(data["presidio_text"], str)

    # 3. utility_metrics
    assert "utility_metrics" in data
    util = data["utility_metrics"]
    assert "cosine_similarity" in util
    assert "nli_forward" in util
    assert "utility_score" in util

    # 4. turn_record
    assert "turn_record" in data
    turn = data["turn_record"]
    assert "turn_number" in turn
    assert "session_id" in turn
    assert "turn_scores" in turn
    assert "cumulative_scores" in turn
    assert "joint_entropy" in turn
    assert "leakage_delta" in turn

    # 5. privacy_report
    assert "privacy_report" in data
    report = data["privacy_report"]
    assert "explicit" in report
    assert "inferential" in report
    assert "count" in report["explicit"]
    assert "hard_count" in report["explicit"]
    assert "by_type" in report["explicit"]
    assert "spans" in report["explicit"]

    # 6. comparison
    assert "comparison" in data
    comp = data["comparison"]
    assert "verdict" in comp
    assert "inferential_before" in comp
    assert "inferential_after" in comp
    assert "inferential_delta" in comp
    assert "per_attribute_delta" in comp
    assert "explicit_before" in comp
    assert "explicit_after" in comp
    assert "hard_pii_remaining" in comp
    assert "is_win" in comp
    assert "notes" in comp

    # 7. top-level verdict and is_win
    assert "verdict" in data
    assert "is_win" in data
    assert data["verdict"] == comp["verdict"]
    assert data["is_win"] == comp["is_win"]


def test_rewrite_endpoint_adversarial_pii(client):
    """Verify /rewrite endpoint behavior on adversarial PII inputs."""
    payload = {
        "text": "Call me at 303-555-0192 or email test@example.com. I am 26 years old.",
        "max_risk": PROVISIONAL_MAX_RISK_THRESHOLD,
    }
    resp = client.post("/rewrite", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "original_text" in data
    assert "rewritten_text" in data
    assert "utility_metrics" in data
    assert data["original_text"] == payload["text"]
    assert data["rewritten_text"] != ""


# ==============================================================================
# SECTION 4: Edge Cases & Robustness
# ==============================================================================


def test_analyze_empty_and_whitespace_rejection(client):
    """Verify edge cases: empty strings or whitespace-only inputs are properly rejected."""
    for bad_text in ["", "   ", "\n\t"]:
        resp = client.post("/analyze", json={"text": bad_text})
        assert resp.status_code in (400, 422)


def test_forbidden_claims_in_report_strings(client):
    """Verify Rule 4: Module output never contains forbidden claims ('safe', 'anonymous', etc.)."""
    from src.product.report import FORBIDDEN_CLAIMS

    raw_text = "I am 26 years old living in Denver, CO and my email is test@example.com."
    resp = client.post("/analyze", json={"text": raw_text})
    assert resp.status_code == 200
    data = resp.json()

    dumped = str(data).lower()
    for forbidden in FORBIDDEN_CLAIMS:
        assert forbidden.lower() not in dumped, f"Forbidden claim '{forbidden}' detected in response: {dumped}"
