"""The API must say which engine produced the rewrite and whether it changed.

Session 06: P02, P03 and P05 received their own message back unchanged and
believed they were protected; P04's logged rewrite differed from what the UI
showed. These pin rewrite_engine, rewrite_changed, served_text_sha256 and the
per-stage timings.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from web import api
from web.api import REWRITE_ENGINES, TIMING_STAGES, app, rewrite_changed

SAFE = "This is a completely safe and generic sentence."
HEURISTIC_OK = "I am 26 years old living in Denver, CO."
# HIGH risk, nothing the heuristic or Presidio can change: the P05 situation.
UNCHANGED_HIGH = "I work as a nurse and my boss keeps changing the rota."


@pytest.fixture(autouse=True)
def _isolate_transparency(monkeypatch):
    """Isolate transparency tests from persistent global neural rewriter adapters."""
    monkeypatch.setattr(api, "qwen_rewriter", None)
    monkeypatch.setattr(api, "get_qwen_rewriter", lambda: api.qwen_rewriter)


@pytest.fixture
def client():
    return TestClient(app)


def _analyze(client, text, session):
    resp = client.post("/analyze", json={"text": text, "session_id": session})
    assert resp.status_code == 200
    return resp.json()


def test_passthrough_low_risk(client):
    data = _analyze(client, SAFE, "transparency_low")
    assert data["rewrite_engine"] == "passthrough_low_risk"
    assert data["rewrite_changed"] is False


def test_heuristic_engine(client):
    data = _analyze(client, HEURISTIC_OK, "transparency_heur")
    assert data["rewrite_engine"] == "heuristic"
    assert data["rewrite_changed"] is True
    assert data["rewritten_text"] != HEURISTIC_OK


def test_presidio_fallback_unchanged_is_reported_as_unchanged(client):
    data = _analyze(client, UNCHANGED_HIGH, "transparency_presidio")
    assert data["risk_summary"]["overall_band"] != "LOW"
    assert data["rewrite_engine"] == "presidio_fallback"
    assert data["rewrite_changed"] is False


def test_served_text_hash_matches_returned_text(client):
    for i, text in enumerate((SAFE, HEURISTIC_OK, UNCHANGED_HIGH)):
        data = _analyze(client, text, f"transparency_hash_{i}")
        expected = hashlib.sha256(data["rewritten_text"].encode("utf-8")).hexdigest()
        assert data["served_text_sha256"] == expected


def test_rewrite_endpoint_reports_engine(client):
    data = client.post("/rewrite", json={"text": HEURISTIC_OK}).json()
    assert data["rewrite_engine"] == "heuristic"
    assert data["rewrite_changed"] is True

    data = client.post("/rewrite", json={"text": UNCHANGED_HIGH}).json()
    assert data["rewrite_engine"] == "presidio_fallback"
    assert data["rewrite_changed"] is False


class StubQwen:
    """Stands in for a loaded adapter without any weights."""

    is_adapter_loaded = True

    def __init__(self, output=None) -> None:
        self.output = output

    @staticmethod
    def _heuristic_fallback(text):
        return "HEURISTIC FALLBACK OUTPUT"

    def rewrite(self, text):
        return self.output or "I am an adult who lives in a mid-sized city."


def test_qwen_adapter_engine(client, monkeypatch):
    monkeypatch.setattr(api, "qwen_rewriter", StubQwen())
    data = _analyze(client, HEURISTIC_OK, "transparency_qwen")
    assert data["rewrite_engine"] == "qwen_adapter"
    assert data["rewritten_text"] == "I am an adult who lives in a mid-sized city."


def test_qwen_internal_fallback_is_labelled_heuristic(client, monkeypatch):
    # rewrite() returning exactly the heuristic fallback means generation failed.
    stub = StubQwen(output="I am an adult who lives somewhere.")
    stub._heuristic_fallback = staticmethod(lambda text: "I am an adult who lives somewhere.")
    monkeypatch.setattr(api, "qwen_rewriter", stub)
    data = client.post("/rewrite", json={"text": HEURISTIC_OK}).json()
    assert data["rewrite_engine"] == "heuristic"


def test_every_engine_value_is_declared():
    assert set(REWRITE_ENGINES) == {"qwen_adapter", "heuristic", "presidio_fallback", "passthrough_low_risk"}


def test_rewrite_changed_ignores_whitespace_only():
    assert rewrite_changed("a  b\nc", "a b c") is False
    assert rewrite_changed("a b c", "a b d") is True


def test_timings_per_stage(client):
    low = _analyze(client, SAFE, "transparency_t_low")["timings_ms"]
    high = _analyze(client, HEURISTIC_OK, "transparency_t_high")["timings_ms"]

    assert set(low) == set(TIMING_STAGES) == set(high)
    assert low["rewrite"] is None and low["rescore"] is None
    for stage in TIMING_STAGES:
        assert high[stage] is not None and high[stage] >= 0.0
    parts = sum(high[s] for s in TIMING_STAGES if s != "total")
    assert high["total"] >= parts * 0.99


def test_allowed_origins_env(monkeypatch):
    monkeypatch.delenv("IG_ALLOWED_ORIGINS", raising=False)
    assert api.allowed_origins() == ["*"]
    monkeypatch.setenv("IG_ALLOWED_ORIGINS", "http://localhost:8000, https://example.edu")
    assert api.allowed_origins() == ["http://localhost:8000", "https://example.edu"]
