"""Tests for Milestone 4: UI and Colab Integration (F12, F13, F14).

Verifies FastAPI /analyze and /rewrite endpoints, static HTML mounting,
Colab tunneling fallbacks, and code compliance / code compliance compliance.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from web.api import app, launch_tunnel


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_endpoint(client):
    """Verify health endpoint returns status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_analyze_endpoint_conformance(client):
    """Verify /analyze endpoint returns all required fields in AnalyzeResponse."""
    payload = {
        "text": "I am 26 years old living in Denver, CO and working as a software engineer.",
        "session_id": "test_session_ui",
        "user_id": "user_ui_test",
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "risk_summary" in data
    assert "rewritten_text" in data
    assert "presidio_text" in data
    assert "utility_metrics" in data
    assert "turn_record" in data

    # Verify risk_summary matches src.product.risk_bands.RiskSummary contract
    risk = data["risk_summary"]
    assert "overall" in risk
    assert "overall_band" in risk
    assert risk["overall_band"] in ("LOW", "MEDIUM", "HIGH")
    assert "scores" in risk
    assert "age" in risk["scores"]
    assert "location" in risk["scores"]

    # Verify utility metrics
    util = data["utility_metrics"]
    assert "cosine_similarity" in util
    assert "nli_forward" in util
    assert "utility_score" in util

    # Verify turn record
    turn = data["turn_record"]
    assert turn["session_id"] == "test_session_ui"
    assert "joint_entropy" in turn
    assert "leakage_delta" in turn


def test_api_analyze_validation_error(client):
    """Verify /analyze rejects empty text with 400 or 422."""
    resp = client.post("/analyze", json={"text": "   "})
    assert resp.status_code in (400, 422)


def test_api_rewrite_endpoint(client):
    """Verify /rewrite endpoint returns rewritten text and utility scores."""
    payload = {
        "text": "I am 26 years old living in Denver, CO.",
        "max_risk": 0.30,
    }
    resp = client.post("/rewrite", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "rewritten_text" in data
    assert "utility_metrics" in data
    assert data["rewritten_text"] != ""


def test_api_static_html_served(client):
    """Verify web root serves the minimalist HTML interface."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "InferenceGuard" in resp.text
    assert "text-input" in resp.text
    assert "analyze-btn" in resp.text


def test_launch_tunnel_fallback():
    """Verify launch_tunnel returns a valid URL string without crashing."""
    url = launch_tunnel(port=8000, tunnel_type="invalid_tunnel_type")
    assert url.startswith("http")


