"""The proxy's protection and honesty contract.

Each test pins one finding from the Session 12 audit. Written as source and
config assertions rather than live requests so they run with no model stack
and no network -- the same reason the rest of the guard tests are shaped this
way (see tests/README.md).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = (ROOT / "web" / "api.py").read_text(encoding="utf-8")


def _demo_mode_enabled() -> bool:
    """Mirror of web.api._demo_mode_enabled, importable without fastapi."""
    return os.getenv("IG_DEMO_MODE", "false").strip().lower() in ("1", "true", "yes", "on")


# --- P0: the whole transcript is protected, not just the last turn --------

def test_no_role_is_passed_through_unconditionally():
    """Before the fix the branch read:

        elif msg.role in ("user", "tool") and ...:
            <pseudonymize>
        else:
            content = msg.content     # assistant and system, verbatim

    Assistant turns quote the user's details back and system prompts carry
    account context, so that sent upstream exactly what the last turn had
    just been rewritten to withhold.
    """
    assert 'elif msg.role in ("user", "tool")' not in API, (
        "the sanitising loop is role-gated again; assistant and system content "
        "would reach the upstream model unprotected"
    )
    assert "elif p_state_ctx is not None and active_sec_ctx:" in API


def test_messages_that_could_not_be_protected_are_reported():
    """Falling back to raw text is sometimes unavoidable -- doing it silently
    is not. The caller is told which roles went out unprotected."""
    assert "unprotected_roles" in API
    assert '"unprotected_roles": sorted(unprotected_roles)' in API


# --- P0: no canned replies pretending to be the upstream model ------------

def test_production_returns_an_error_rather_than_a_simulated_reply():
    """The canned replies assert completed actions -- "applied a one-time
    courtesy waiver", "your replacement shipment is scheduled for dispatch" --
    and were returned with finish_reason "stop" in an OpenAI-shaped response
    whenever the upstream call threw. A caller cannot tell that apart from a
    real answer."""
    assert "if not assistant_reply and not _demo_mode_enabled():" in API
    assert "upstream_unavailable" in API
    assert "status_code=502" in API


def test_a_simulated_reply_is_labelled_as_one():
    assert '"simulated_reply": upstream_mode != "live_upstream"' in API


def test_demo_mode_is_off_unless_explicitly_set():
    assert 'os.getenv("IG_DEMO_MODE", "false")' in API


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), ("false", False), ("0", False), ("no", False),
     ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True)],
)
def test_demo_mode_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("IG_DEMO_MODE", raising=False)
    else:
        monkeypatch.setenv("IG_DEMO_MODE", value)
    assert _demo_mode_enabled() is expected


# --- P1: the global debug trace is not readable by default ----------------

def test_debug_trace_is_gated():
    """`last_debug_trace` is global, not session-scoped, so an open endpoint
    serves the previous caller's pseudonymized text and candidate rewrites."""
    endpoint = API.split('@app.get("/debug/last_rewrite")')[1].split("@app.")[0]
    assert "_demo_mode_enabled()" in endpoint
    assert "debug_disabled" in endpoint


# --- P1: one worker, because conversation state is process-local ----------

def test_container_runs_a_single_worker():
    """Aliases and cumulative leakage live in a process-local dict, so a
    second worker splits one conversation across two states."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmd = [l for l in dockerfile.splitlines() if l.startswith("CMD [") and "uvicorn" in l]
    assert cmd, "no uvicorn CMD found in the Dockerfile"
    workers = re.search(r'"--workers",\s*"(\d+)"', cmd[-1])
    assert workers, "worker count is not pinned"
    assert workers.group(1) == "1", (
        f'Dockerfile starts {workers.group(1)} workers, but conversation state is '
        f"process-local; a shared store is needed before raising this"
    )


def test_known_limitations_are_written_down():
    """The findings not fixed this week must be documented, not forgotten."""
    doc = ROOT / "docs" / "engineering" / "known-limitations.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for topic in ("web.api", "src.api.app", "LOAD_REWRITER_WEIGHTS", "raw_text"):
        assert topic in text, f"{topic} is not covered in known-limitations.md"
