"""Drop-in Python SDK for integrating InferenceGuard with customer support AI agents.

Supports two enterprise deployment patterns:
  1. 1-line OpenAI SDK / LangChain base_url swap using HTTP headers
     (X-Privacy-Mode: business_support, X-InferenceGuard-Restore: gateway).
  2. Zendesk / Intercom / Salesforce webhook middleware via POST /v1/support/webhook.
"""

from __future__ import annotations

import json
import secrets
import urllib.request
from typing import Any, Dict, List, Optional


class InferenceGuardSupportClient:
    """Lightweight zero-dependency client for InferenceGuard Business Support Gateway."""

    def __init__(
        self,
        gateway_base_url: str = "http://127.0.0.1:8000",
        session_secret: Optional[str] = None,
        privacy_mode: str = "business_support",
        restore_mode: str = "gateway",
    ) -> None:
        self.gateway_base_url = gateway_base_url.rstrip("/")
        self.session_secret = session_secret or secrets.token_hex(32)
        self.privacy_mode = privacy_mode
        self.restore_mode = restore_mode

    def openai_client_kwargs(
        self,
        api_key: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return kwargs for `openai.OpenAI(**client.openai_client_kwargs(...))`.

        Enables a 1-line drop-in swap for LangChain, LlamaIndex, CrewAI, or raw OpenAI SDK
        support agents with automatic gateway-side restoration and zero-PII CRM transcripts.
        """
        headers = {
            "X-Session-Secret": self.session_secret,
            "X-Privacy-Mode": self.privacy_mode,
            "X-InferenceGuard-Restore": self.restore_mode,
        }
        if session_id:
            headers["X-Session-ID"] = session_id
        return {
            "base_url": f"{self.gateway_base_url}/v1",
            "api_key": api_key,
            "default_headers": headers,
        }

    def handle_webhook(
        self,
        ticket_id: str,
        customer_message: str,
        channel: str = "zendesk",
        crm_context: Optional[Dict[str, str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o-mini",
        upstream_api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Dispatch an inbound Zendesk/Intercom/Salesforce ticket through /v1/support/webhook."""
        payload: Dict[str, Any] = {
            "ticket_id": ticket_id,
            "channel": channel,
            "customer_message": customer_message,
            "crm_context": crm_context or {},
            "history": history or [],
            "model": model,
        }
        if upstream_api_key:
            payload["upstream_api_key"] = upstream_api_key

        req = urllib.request.Request(
            f"{self.gateway_base_url}/v1/support/webhook",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Session-Secret": self.session_secret,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
