"""Multi-turn leakage: acting on what the conversation has revealed, session limits, measurement.

The S-06 failure this pins: a turn that is LOW on its own but adds to an
attribute earlier turns exposed was passed through, because /analyze decided
on the turn's own score only.
"""

from __future__ import annotations

import hashlib

import pytest

from src.conversation.state import ConversationTracker, contributing_attributes, exposed_attributes
from src.rewriter.generate_training_data import estimate_privacy_risk

pytest.importorskip("fastapi")


def scripted(scores_by_text):
    """Estimator returning fixed scores per message; 0.05 baseline otherwise."""

    def estimate(text):
        base = {"age": 0.05, "location": 0.05, "occupation": 0.05, "education": 0.05}
        base.update(scores_by_text.get(text, {}))
        return base

    return estimate


T1 = "I take the Green Line in most mornings."
T2 = "Thinking about which side of the river to live on."
T3 = "Any tips for keeping a budget spreadsheet?"
SCRIPT = scripted({T1: {"location": 0.75}, T2: {"location": 0.15}})


# --- tracker ------------------------------------------------------------------

def test_contribution_needs_an_exposed_attribute_and_new_evidence():
    tracker = ConversationTracker(risk_estimator_fn=SCRIPT)
    r1 = tracker.add_turn(T1, "s")
    r2 = tracker.add_turn(T2, "s")
    r3 = tracker.add_turn(T3, "s")
    assert exposed_attributes(r1) == ["location"] and contributing_attributes(r1) == ["location"]
    assert contributing_attributes(r2) == ["location"]  # LOW alone, adds to an exposed attribute
    assert exposed_attributes(r3) == ["location"] and contributing_attributes(r3) == []


def test_tracker_keeps_a_hash_not_the_text():
    tracker = ConversationTracker(risk_estimator_fn=SCRIPT)
    rec = tracker.add_turn("my co-op ends in December", "s")
    assert rec.text_sha256 == hashlib.sha256(b"my co-op ends in December").hexdigest()
    assert "my co-op ends in December" not in str(rec.to_dict())


def test_idle_sessions_expire_and_restart():
    now = [0.0]
    tracker = ConversationTracker(risk_estimator_fn=SCRIPT, session_ttl_seconds=60, clock=lambda: now[0])
    tracker.add_turn(T1, "a")
    tracker.add_turn(T1, "b")
    now[0] = 30.0
    assert tracker.add_turn(T3, "a").turn_number == 2
    now[0] = 100.0  # "b" idle 100 s: pruned when another session is active
    tracker.add_turn(T3, "a")
    assert "b" not in tracker.sessions
    now[0] = 200.0  # "a" itself idle 100 s: its next turn starts over
    assert tracker.add_turn(T3, "a").turn_number == 1


def test_oldest_sessions_evicted_beyond_the_cap():
    now = [0.0]
    tracker = ConversationTracker(risk_estimator_fn=SCRIPT, max_sessions=2, clock=lambda: now[0])
    for i, sid in enumerate(["a", "b", "c"]):
        now[0] = float(i)
        tracker.add_turn(T3, sid)
    assert set(tracker.sessions) == {"b", "c"}


def test_reset_session_reports_whether_it_existed():
    tracker = ConversationTracker(risk_estimator_fn=SCRIPT)
    tracker.add_turn(T1, "s")
    assert tracker.reset_session("s") is True
    assert tracker.reset_session("s") is False


# --- API ----------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from web import api

    monkeypatch.setattr(api, "get_risk_model", lambda: None)
    monkeypatch.setattr(api, "conversation_tracker", ConversationTracker(risk_estimator_fn=SCRIPT))
    return TestClient(api.app)


def _analyze(client, text, session="conv"):
    resp = client.post("/analyze", json={"text": text, "session_id": session})
    assert resp.status_code == 200
    return resp.json()


def test_low_turn_that_adds_to_an_exposed_attribute_is_not_passed_through(client):
    first = _analyze(client, T1)
    assert first["conversation"]["decision_basis"] == "turn"

    second = _analyze(client, T2)
    assert second["risk_summary"]["overall_band"] == "LOW"
    assert second["conversation"]["decision_basis"] == "cumulative"
    assert second["conversation"]["contributing_attributes"] == ["location"]
    assert second["rewrite_engine"] != "passthrough_low_risk"


def test_unrelated_low_turn_passes_through_but_reports_the_exposure(client):
    _analyze(client, T1)
    third = _analyze(client, T3)
    assert third["conversation"]["decision_basis"] == "none"
    assert third["rewrite_engine"] == "passthrough_low_risk"
    assert third["conversation"]["exposed_attributes"] == ["location"]


def test_same_low_turn_in_a_fresh_session_passes_through(client):
    data = _analyze(client, T2, session="fresh")
    assert data["conversation"]["decision_basis"] == "none"
    assert data["rewrite_engine"] == "passthrough_low_risk"


def test_delete_session_forgets_the_conversation(client):
    _analyze(client, T1)
    assert client.delete("/session/conv").json() == {"session_id": "conv", "reset": True}
    assert _analyze(client, T2)["conversation"]["decision_basis"] == "none"
    assert client.delete("/session/never-seen").json()["reset"] is False


# --- rule estimator -------------------------------------------------------------

@pytest.mark.parametrize(
    "text,location_is_cue",
    [
        ("My co-op ends in December.", False),  # bare "co" used to count as Colorado
        ("I am moving to Seattle, WA soon.", True),
        ("Back home in Denver, CO for the holidays.", True),
    ],
)
def test_state_codes_count_only_after_a_city(text, location_is_cue):
    assert (estimate_privacy_risk(text)["location"] > 0.05) is location_is_cue


# --- measurement ------------------------------------------------------------------

def _rec(rid, author, thread, text, *inferable):
    fields = {"age": "age", "location": "city_country", "occupation": "occupation", "education": "education"}
    human = {f: {"estimate": ""} for f in fields.values()}
    for attr in inferable:
        human[fields[attr]] = {"estimate": "guess"}
    return {"id": rid, "profile_id": author, "thread_id": thread, "text": text, "reviews": {"human": human}}


def test_evaluate_counts_exposure_only_earlier_turns_reveal():
    from scripts.eval_multiturn import build_conversations, evaluate

    records = [
        _rec("1", "p1", "t1", T1, "location"),
        _rec("2", "p1", "t1", T3),
        _rec("3", "p2", "t1", T3),  # single-turn conversation: skipped
    ]
    conversations = build_conversations(records, min_turns=2)
    assert len(conversations) == 1

    result = evaluate(conversations, SCRIPT)
    assert result["data"]["exposed_pairs"] == 1
    assert result["data"]["exposed_only_by_earlier_turns"] == 1
    assert result["detection"]["recall_last_turn"] == 0.0  # the last message alone shows nothing
    assert result["detection"]["recall_cumulative"] == 1.0
    assert result["detection"]["false_alarm_cumulative"] == 0.0
