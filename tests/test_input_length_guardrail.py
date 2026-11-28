"""The product must not score a prefix and present it as the whole message.

The risk model tokenizes with truncation=True, max_length=512. Nothing told
the user, nothing told the caller, and no user-study input ever came close to
the limit -- the longest was 140 words -- so it never surfaced in testing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = (ROOT / "web" / "api.py").read_text(encoding="utf-8")
APPJS = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

WORD_LIMIT = 300
MODEL_TOKEN_LIMIT = 512


def measure(text: str) -> dict:
    """Mirror of web.api.measure_input_length, importable without fastapi."""
    words = len(text.split())
    est = int(words / 0.75) if words else 0
    return {
        "words": words,
        "estimated_tokens": est,
        "over_limit": words > WORD_LIMIT,
        "truncated_for_analysis": est > MODEL_TOKEN_LIMIT,
    }


# --- the measurement -----------------------------------------------------

@pytest.mark.parametrize(
    "words,over,truncated",
    [(0, False, False), (29, False, False), (300, False, False),
     (301, True, False), (384, True, False), (500, True, True)],
)
def test_thresholds(words, over, truncated):
    r = measure(" ".join(["w"] * words))
    assert r["over_limit"] is over
    assert r["truncated_for_analysis"] is truncated


def test_the_shown_limit_sits_below_the_model_cap():
    """300 words is ~400 tokens against a 512-token model window. The margin is
    deliberate: punctuation and rare words tokenize worse than plain prose, so
    a message at the shown limit must not already be truncated."""
    assert measure(" ".join(["w"] * WORD_LIMIT))["estimated_tokens"] < MODEL_TOKEN_LIMIT


# --- the server discloses it ---------------------------------------------

def test_analyze_reports_input_length():
    assert "ANALYSIS_MODEL_TOKEN_LIMIT = 512" in API
    assert "ANALYSIS_WORD_LIMIT = 300" in API
    assert "def measure_input_length" in API
    assert "input_length: Dict[str, Any]" in API, "not declared on AnalyzeResponse"
    assert "input_length=input_length" in API, "not passed when building the response"


def test_truncation_is_logged_not_swallowed():
    assert 'if input_length["truncated_for_analysis"]:' in API
    block = API.split('if input_length["truncated_for_analysis"]:')[1][:400]
    assert "logger.warning" in block


# --- the user sees it live -----------------------------------------------

def test_counter_is_in_the_markup_and_announced():
    assert 'id="word-count"' in HTML
    assert 'aria-live="polite"' in HTML, "the count changes as you type; it must be announced"
    assert 'aria-describedby="input-help word-count"' in HTML, (
        "the textarea should point at the counter for screen readers"
    )


def test_counter_updates_on_every_keystroke():
    assert 'addEventListener("input", updateWordCount)' in APPJS
    assert "updateWordCount();" in APPJS, "should also run once on load"


def test_counter_warns_before_the_model_truncates():
    """'near' has to fire below the limit, and 'truncated' at the real cap --
    a counter that only turns red after the damage is not a guardrail."""
    assert "const WORD_LIMIT = 300;" in APPJS
    assert "const MODEL_WORD_CAP = 384;" in APPJS
    for state in ('"near"', '"over"', '"truncated"'):
        assert state in APPJS


def test_counter_styles_exist_for_every_state():
    css = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
    for state in ("near", "over", "truncated"):
        assert f'data-state="{state}"' in css, f"no visual state for {state}"
