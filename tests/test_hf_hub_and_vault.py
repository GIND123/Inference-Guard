"""Tests for Hugging Face Hub manifest pinning, zero-plaintext HMAC vault, and session auth."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi",
    reason="needs the optional fastapi dependency; see tests/README.md",
)
from fastapi.testclient import TestClient

from web import api
from web.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_zero_plaintext_hmac_vault_and_session_secret_auth(client):
    sid = "vault_security_test"
    secret = "a" * 64
    client.delete(f"/session/{sid}", headers={"X-Session-Secret": secret})

    r1 = client.post(
        "/analyze",
        headers={"X-Session-Secret": secret},
        json={
            "text": "My manager Sarah asked me to email jane.doe@northeastern.edu about the schedule.",
            "session_id": sid,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert "Sarah" in d1["pseudonyms"]
    assert "jane.doe@northeastern.edu" in d1["pseudonyms"]
    sarah_tok = d1["pseudonyms"]["Sarah"]
    email_tok = d1["pseudonyms"]["jane.doe@northeastern.edu"]

    # Server state must contain ZERO plaintext names or emails
    state = api.pseudonym_states[sid]
    serialized_state = str(state.model_dump())
    assert "Sarah" not in serialized_state
    assert "sarah" not in serialized_state
    assert "jane.doe@northeastern.edu" not in serialized_state
    assert state.reverse_entity_map == {}

    # History entries must not store plaintext pseudonyms
    for item in d1["conversation_history"]:
        assert "pseudonyms" not in item

    # Turn 2 with same secret maps Sarah -> same PERSON_k token
    r2 = client.post(
        "/analyze",
        headers={"X-Session-Secret": secret},
        json={
            "text": "Can you draft a polite follow-up email to Sarah asking for feedback?",
            "session_id": sid,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["pseudonyms"].get("Sarah") == sarah_tok
    assert sarah_tok in d2["rewritten_text"]
    assert email_tok != sarah_tok

    # Mismatched or missing X-Session-Secret for protected session must return 403
    r_bad = client.post(
        "/analyze",
        headers={"X-Session-Secret": "wrong_secret"},
        json={"text": "Hello", "session_id": sid},
    )
    assert r_bad.status_code == 403

    r_missing = client.post(
        "/analyze",
        json={"text": "Hello", "session_id": sid},
    )
    assert r_missing.status_code == 403

    # Cleanup requires matching secret
    assert client.delete(f"/session/{sid}", headers={"X-Session-Secret": "wrong"}).status_code == 403
    assert client.delete(f"/session/{sid}", headers={"X-Session-Secret": secret}).status_code == 200


def test_status_reports_pinned_hf_hub_model_ids_and_revisions(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "AT707/inferenceguard-modernbert-risk" in str(data["risk_model_id"])
    assert "AT707/inferenceguard-rewriter-v1" in str(data["rewriter_model_id"])
    if data["risk_model_source"] == "huggingface_hub":
        assert data["risk_model_revision"] == "4cecb3c90e84112ce5a5e3ed173060a1ab9609a7"
