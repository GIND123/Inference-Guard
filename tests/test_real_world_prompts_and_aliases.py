"""Regression tests for real-world prompts, unified aliasing, and OOD generalization."""

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


def test_coop_december_green_line_preserves_timing_and_generalizes(client):
    text = (
        "My second co-op ends in December and I take the Green Line past Back Bay "
        "to campus every morning—should I renew my lease now or wait until finals?"
    )
    resp = client.post("/analyze", json={"text": text, "session_id": "rw_coop_dec"})
    assert resp.status_code == 200
    data = resp.json()
    rw = data["rewritten_text"]

    assert "<DATE_TIME>" not in rw
    assert "<LOCATION>" not in rw
    assert "December" in rw
    assert "Green Line" not in rw
    assert "Back Bay" not in rw
    assert data["comparison"]["inferential_after"] <= data["comparison"]["inferential_before"]


def test_analyst_tenure_fintech_commute_grammar_and_numbers(client):
    text = (
        "I'm a 27-year-old senior data analyst with 7 years of tenure at a Boston fintech firm, "
        "comparing a 55-minute Green Line commute for a $1,200/month apartment vs. a 15-minute "
        "walk for $1,950/month. Which makes more financial sense?"
    )
    resp = client.post("/analyze", json={"text": text, "session_id": "rw_analyst"})
    assert resp.status_code == 200
    data = resp.json()
    rw = data["rewritten_text"]

    assert "a the area" not in rw.lower()
    assert "Boston" not in rw
    assert "27-year-old" not in rw
    assert "Green Line" not in rw
    assert ("1,200" in rw or "1200" in rw)
    assert ("1,950" in rw or "1950" in rw)
    assert "55" in rw and "15" in rw


def test_pediatric_resident_hospital_preserves_patient_dosage(client):
    text = (
        "As a pediatric resident at Boston Children's Hospital working 80-hour weeks, "
        "I need to calculate the amoxicillin dosage for a 4-year-old patient weighing 16 kg "
        "with acute otitis media—can you double-check my mg/kg/day math?"
    )
    resp = client.post("/analyze", json={"text": text, "session_id": "rw_pediatric"})
    assert resp.status_code == 200
    data = resp.json()
    rw = data["rewritten_text"]

    assert "Children's company" not in rw
    assert "Boston Children's" not in rw
    assert "HOSPITAL_" not in rw
    assert "4-year-old patient" in rw
    assert "16 kg" in rw
    assert "amoxicillin" in rw.lower()


def test_multiturn_unified_aliases_and_no_q4_or_next_month_false_positives(client):
    sid = "rw_multiturn_sarah"
    client.delete(f"/session/{sid}")

    turns = [
        "My manager Sarah asked me to prep slides for our Q4 board review.",
        "I'm taking the Red Line from Kendall Square to meet her early.",
        "Sarah wants the slides to highlight our post-doc lab's NIH grant.",
        "Can you draft a polite follow-up email to Sarah asking for feedback before my thesis defense next month?",
    ]
    responses = []
    for t in turns:
        r = client.post("/analyze", json={"text": t, "session_id": sid})
        assert r.status_code == 200
        responses.append(r.json())

    # Q4 must never be aliased or tagged as LOCATION; next month must not be <DATE_TIME>
    last = responses[-1]
    assert "Q4" not in last["pseudonyms"]
    assert "<LOCATION>" not in responses[0]["rewritten_text"]
    assert "<DATE_TIME>" not in last["rewritten_text"]
    assert "next month" in last["rewritten_text"].lower() or "upcoming" in last["rewritten_text"].lower()

    # If Sarah was detected by NER/fallback, she must have one consistent PERSON_k alias inside rewritten_text
    sarah_tok = last["pseudonyms"].get("Sarah")
    if sarah_tok:
        assert sarah_tok in responses[0]["rewritten_text"]
        assert sarah_tok in responses[2]["rewritten_text"]
        assert sarah_tok in responses[3]["rewritten_text"]
        assert "Sarah" not in last["rewritten_text"]

    # Inferential cues in Turn 2 and Turn 3 should be generalized
    assert "Red Line" not in responses[1]["rewritten_text"]
    assert "Kendall Square" not in responses[1]["rewritten_text"]

    # Multi-turn history tracks 4 turns and protected cumulative risk < unprotected
    hist = last["conversation_history"]
    assert len(hist) == 4
    assert hist[-1]["cumulative_risk_after"] < hist[-1]["cumulative_risk_before"]


def test_unseen_out_of_distribution_prompts_preserve_task_math_and_generalize(client):
    p1 = (
        "As a 34-year-old neonatal fellow at Zurich University Hospital riding the S-Bahn from Oerlikon, "
        "how should I structure a handoff checklist for a 2-year-old patient on 12 mg/kg maintenance fluids?"
    )
    r1 = client.post("/analyze", json={"text": p1, "session_id": "rw_ood_1"}).json()
    rw1 = r1["rewritten_text"]
    assert "Zurich University Hospital" not in rw1
    assert "2-year-old patient" in rw1
    assert "12 mg/kg" in rw1

    p2 = (
        "I've spent 9 years as a principal hydrogeologist in Perth near Kings Park—"
        "should I take a $145,000 remote contract or stay for $160,000 onsite?"
    )
    r2 = client.post("/analyze", json={"text": p2, "session_id": "rw_ood_2"}).json()
    rw2 = r2["rewritten_text"]
    assert ("145,000" in rw2 or "145000" in rw2)
    assert ("160,000" in rw2 or "160000" in rw2)
