"""Tests for coreference sub-key linking, stream alias restoration, /v1/chat/completions proxy, and Split-Brain UI."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "fastapi",
    reason="needs the optional fastapi dependency; see tests/README.md",
)
from fastapi.testclient import TestClient

from src.contracts.schemas import ConversationState
from src.conversation.entity_map import DeterministicPseudonymManager, StreamAliasRestorer
from web import api
from web.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_coreference_subkey_linking_across_turns():
    mgr = DeterministicPseudonymManager(referents_only=True)
    state = ConversationState(conversation_id="coref_unit")
    secret = "s" * 64

    p1, spans1, _ = mgr.pseudonymize_turn(
        "My manager Sarah Chen asked me to review the deck.",
        state,
        session_secret=secret,
        blind_index=True,
    )
    assert "PERSON_1" in p1
    assert any(s.original_surface == "Sarah Chen" and s.pseudonym_token == "PERSON_1" for s in spans1)

    p2, spans2, _ = mgr.pseudonymize_turn(
        "Sarah said we should emphasize Q4 metrics.",
        state,
        session_secret=secret,
        blind_index=True,
    )
    assert "PERSON_1" in p2
    assert "Sarah" not in p2

    p3, spans3, _ = mgr.pseudonymize_turn(
        "Can you draft a reply to Ms. Chen?",
        state,
        session_secret=secret,
        blind_index=True,
    )
    assert "PERSON_1" in p3
    assert "Chen" not in p3

    # Distinct second person sharing first name gets PERSON_2 without overwriting PERSON_1
    p4, spans4, _ = mgr.pseudonymize_turn(
        "Sarah Patel joined the call.",
        state,
        session_secret=secret,
        blind_index=True,
    )
    assert "PERSON_2" in p4
    assert state.reverse_entity_map == {}


def test_stream_alias_restorer_handles_split_token_boundaries():
    restorer = StreamAliasRestorer({"PERSON_1": "Sarah Chen", "EMAIL_1": "sarah@example.com"})
    chunks = ["Hello PER", "SON_1, please ", "CC EM", "AIL_1."]
    out = "".join(restorer.feed(c) for c in chunks) + restorer.flush()
    assert out == "Hello Sarah Chen, please CC sarah@example.com."


def test_openai_compatible_chat_completions_proxy_multiturn(client):
    sid = "proxy_mt_test"
    secret = "p" * 64
    client.delete(f"/session/{sid}", headers={"X-Session-Secret": secret})

    r1 = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Secret": secret},
        json={
            "model": "gpt-4o-mini",
            "session_id": sid,
            "messages": [
                {"role": "user", "content": "My manager Sarah Chen asked me to prep slides for our Q4 board review in Boston."}
            ],
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["object"] == "chat.completion"
    assert d1["choices"][0]["message"]["role"] == "assistant"
    ig1 = d1["inferenceguard"]
    assert "PERSON_1" in ig1["sanitized_user_message"]
    assert "Sarah Chen" not in ig1["sanitized_user_message"]
    assert "Boston" not in ig1["sanitized_user_message"]

    r2 = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Secret": secret},
        json={
            "model": "gpt-4o-mini",
            "session_id": sid,
            "messages": [
                {"role": "user", "content": ig1["sanitized_user_message"]},
                {"role": "assistant", "content": d1["choices"][0]["message"]["content"]},
                {"role": "user", "content": "Can you draft a polite follow-up email to Ms. Chen asking for feedback?"},
            ],
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    ig2 = d2["inferenceguard"]
    assert "PERSON_1" in ig2["sanitized_user_message"]
    assert "Chen" not in ig2["sanitized_user_message"]
    assert "PERSON_1" in d2["choices"][0]["message"]["content"]

    # Zero plaintext in server pseudonym_states
    serialized = str(api.pseudonym_states[sid].model_dump())
    assert "Sarah" not in serialized
    assert "Chen" not in serialized
