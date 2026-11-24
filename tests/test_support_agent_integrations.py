"""Integration tests for Hugging Face Hub readiness reporting & B2B support agent gateway."""

from __future__ import annotations

import pytest

# Guard above the heavy imports so this reports as skipped rather than
# failing collection without the stack. See tests/README.md.
pytest.importorskip("fastapi", reason="needs optional fastapi; see tests/README.md")

import secrets
from fastapi.testclient import TestClient

import web.api as api
from src.sdk.support_agent import InferenceGuardSupportClient


client = TestClient(api.app)


def test_ready_and_status_report_hf_hub_model_ids():
    """Verify /ready and /status report the active HF Hub repo IDs instead of local fallback paths."""
    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    st = status_resp.json()
    assert st["risk_model_loaded"] is True
    if st["risk_model_source"] == "huggingface_hub":
        assert st["risk_model_id"] == "AT707/inferenceguard-modernbert-risk"
        assert st["risk_model_path"] == "AT707/inferenceguard-modernbert-risk"

    ready_resp = client.get("/ready")
    expected_ready = bool(st["risk_model_loaded"] and st["qwen_adapter_loaded"])
    assert ready_resp.status_code == (200 if expected_ready else 503)
    rd = ready_resp.json()
    assert rd["ready"] is expected_ready
    assert rd["model_id"] == st["risk_model_id"]
    assert rd["rewriter_model_id"] == st["rewriter_model_id"]


def test_openai_sdk_header_gateway_restore_mode():
    """Verify 1-line OpenAI SDK header integration restores customer reply while keeping CRM log zero-PII."""
    sdk = InferenceGuardSupportClient(privacy_mode="business_support", restore_mode="gateway")
    sid = f"sdk-{secrets.token_hex(4)}"
    kwargs = sdk.openai_client_kwargs(api_key="", session_id=sid)
    assert kwargs["base_url"].endswith("/v1")
    assert kwargs["default_headers"]["X-Privacy-Mode"] == "business_support"
    assert kwargs["default_headers"]["X-InferenceGuard-Restore"] == "gateway"

    resp = client.post(
        "/v1/chat/completions",
        headers=kwargs["default_headers"],
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Hi, I am Sarah Chen at 42 Boylston St, Apt 4B (sarah.chen@gmail.com). "
                        "I am a 62-year-old retired teacher in Boston on a fixed income - "
                        "can you waive the $45 late fee on Order #88412?"
                    ),
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    ig = data["inferenceguard"]
    assert ig["privacy_mode"] == "business_support"
    assert ig["restore_mode"] == "gateway"

    # Customer-facing reply in choices[0] is automatically restored for SMS/Zendesk/Intercom
    customer_reply = data["choices"][0]["message"]["content"]
    assert "Sarah Chen" in customer_reply
    assert "42 Boylston St, Apt 4B" in customer_reply
    assert "sarah.chen@gmail.com" in customer_reply
    assert "Order #88412" in customer_reply

    # CRM-safe transcript has ZERO raw customer PII
    crm_dump = str(ig["crm_safe_transcript"])
    for raw_pii in ("Sarah Chen", "42 Boylston St", "sarah.chen@gmail.com"):
        assert raw_pii not in crm_dump
    assert "PERSON_1" in crm_dump
    assert "ADDRESS_1" in crm_dump
    assert "Order #88412" in crm_dump


def test_zendesk_support_webhook_with_crm_context_and_jurisdiction():
    """Verify POST /v1/support/webhook blind-indexes CRM records and returns dual outputs."""
    ticket_id = f"ZD-{secrets.token_hex(3)}"
    sec = secrets.token_hex(16)
    resp = client.post(
        "/v1/support/webhook",
        headers={"X-Session-Secret": sec},
        json={
            "ticket_id": ticket_id,
            "channel": "zendesk",
            "customer_message": (
                "Hi, I am Sarah Chen (555-867-5309). I moved from 100 Beacon St to "
                "42 Boylston St, Apt 4B. What shipping address and phone number did I use on Order #88412?"
            ),
            "crm_context": {
                "order_id": "#88412",
                "shipping_address": "42 Boylston St, Apt 4B",
                "phone": "555-867-5309",
                "email": "sarah.chen@gmail.com",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_id"] == ticket_id
    assert body["channel"] == "zendesk"

    # Customer-facing reply has exact restored values
    assert "Sarah Chen" in body["customer_facing_reply"]
    assert "42 Boylston St, Apt 4B" in body["customer_facing_reply"]
    assert "555-867-5309" in body["customer_facing_reply"]

    # CRM-safe ticket note & transcript have ZERO raw PII
    assert "Sarah Chen" not in body["crm_safe_ticket_note"]
    assert "42 Boylston St" not in body["crm_safe_ticket_note"]
    assert "555-867-5309" not in body["crm_safe_ticket_note"]
    assert "Order #88412" in body["crm_safe_ticket_note"]
    assert body["privacy_audit"]["entities_aliased"] >= 3
