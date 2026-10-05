"""Tests for the user-facing privacy report contract.

Each rule in report.py's docstring came from something that went wrong, so
each has a test named after the failure rather than after the method.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.report import (  # noqa: E402
    DEFAULT_UTILITY_FLOOR,
    FORBIDDEN_CLAIMS,
    PiiSpan,
    compare,
    compose,
    render,
    render_comparison,
)
from src.product.risk_bands import Cue  # noqa: E402


def r(scores, pii=(), **kw):
    return compose(pii_spans=pii, risk_scores=scores, **kw)


# --- rule 1: the two halves never merge -----------------------------------

def test_explicit_pii_does_not_enter_the_inferential_score():
    quiet = {"location": 0.1, "occupation": 0.1}
    with_pii = r(quiet, pii=[PiiSpan("PERSON", 0, 4), PiiSpan("EMAIL_ADDRESS", 9, 24)])
    without = r(quiet)
    assert with_pii.inferential_overall == without.inferential_overall
    assert with_pii.explicit.count == 2 and without.explicit.count == 0


def test_name_is_stripped_from_risk_scores_if_a_caller_passes_it():
    """Session 04's headline was a name detector because `name` was a dimension."""
    rep = r({"location": 0.2, "name": 0.99})
    assert "name" not in rep.inferential.scores
    assert rep.inferential_overall == 0.2


def test_no_score_at_all_raises_rather_than_reporting_zero():
    with pytest.raises(ValueError):
        compose(risk_scores={"name": 0.9})


# --- rule 2: remaining explicit PII overrides an inferential win ----------

def test_the_resume_case_is_not_reported_as_a_success():
    """location 0.859 -> 0.006 while name, email and phone survived."""
    before = r({"location": 0.859, "occupation": 0.997, "age": 0.982},
               pii=[PiiSpan("PERSON", 18, 26), PiiSpan("EMAIL_ADDRESS", 55, 71),
                    PiiSpan("PHONE_NUMBER", 74, 88)])
    after = r({"location": 0.006, "occupation": 0.987, "age": 0.982},
              pii=[PiiSpan("PERSON", 18, 26), PiiSpan("EMAIL_ADDRESS", 55, 71),
                   PiiSpan("PHONE_NUMBER", 74, 88)])
    c = compare(before, after)

    assert c.verdict == "explicit_pii_remains"
    assert not c.is_win
    assert c.hard_pii_remaining == 3
    assert c.per_attribute_delta["location"] == 0.853   # the tempting headline
    assert c.inferential_delta == 0.01                  # what actually happened
    assert any("still present" in n for n in c.notes)
    assert any("misdescribe" in n for n in c.notes)


def test_soft_pii_alone_does_not_veto_a_win():
    """DATE_TIME or a generic LOCATION tag is not an identifier on its own."""
    before = r({"location": 0.9, "age": 0.3}, pii=[PiiSpan("DATE_TIME", 0, 8)])
    after = r({"location": 0.2, "age": 0.3}, pii=[PiiSpan("DATE_TIME", 0, 8)])
    assert compare(before, after).verdict == "improved"


def test_clearing_the_pii_lets_the_same_rewrite_count_as_a_win():
    before = r({"location": 0.9}, pii=[PiiSpan("PERSON", 0, 4)])
    after = r({"location": 0.2}, pii=[])
    c = compare(before, after)
    assert c.verdict == "improved" and c.explicit_before == 1 and c.explicit_after == 0


# --- rule 3: utility is a guardrail ---------------------------------------

def test_privacy_gain_below_the_utility_floor_is_a_cost_not_a_win():
    before, after = r({"location": 0.9}), r({"location": 0.1})
    assert compare(before, after, utility=0.95).verdict == "improved"
    assert compare(before, after, utility=0.40).verdict == "improved_with_cost"


def test_utility_floor_is_configurable_and_reported():
    before, after = r({"location": 0.9}), r({"location": 0.1})
    c = compare(before, after, utility=0.75, utility_floor=0.80)
    assert c.verdict == "improved_with_cost" and c.utility_floor == 0.80
    assert DEFAULT_UTILITY_FLOOR == 0.70


# --- verdicts that are not wins -------------------------------------------

def test_no_change_and_worse_are_distinguished():
    base = r({"location": 0.5})
    assert compare(base, r({"location": 0.5})).verdict == "no_change"
    assert compare(base, r({"location": 0.8})).verdict == "worse"


def test_overall_is_the_worst_attribute_so_one_collapse_cannot_carry_it():
    before = r({"location": 0.9, "occupation": 0.9})
    after = r({"location": 0.0, "occupation": 0.9})
    c = compare(before, after)
    assert c.per_attribute_delta["location"] == 0.9
    assert c.inferential_delta == 0.0
    assert c.verdict == "no_change"


def test_compare_requires_shared_attributes():
    with pytest.raises(ValueError):
        compare(r({"location": 0.5}), r({"age": 0.5}))


# --- rule 4: the claims we must never make --------------------------------

@pytest.mark.parametrize("verdict_scores,pii", [
    (({"location": 0.9}, {"location": 0.05}), []),
    (({"location": 0.9}, {"location": 0.9}), []),
    (({"location": 0.9}, {"location": 0.1}), [PiiSpan("PERSON", 0, 4)]),
])
def test_no_rendered_output_ever_claims_anonymity(verdict_scores, pii):
    before = r(verdict_scores[0])
    after = r(verdict_scores[1], pii=pii)
    text = " ".join([render(before), render(after), render_comparison(compare(before, after))]).lower()
    for claim in FORBIDDEN_CLAIMS:
        assert claim not in text, f"output claimed {claim!r}"


def test_every_comparison_carries_the_uncertainty_line():
    before, after = r({"location": 0.9}), r({"location": 0.1})
    assert "does not make you unidentifiable" in render_comparison(compare(before, after))


def test_verdict_leads_the_comparison_not_the_best_number():
    before = r({"location": 0.9}, pii=[PiiSpan("PERSON", 0, 4)])
    after = r({"location": 0.01}, pii=[PiiSpan("PERSON", 0, 4)])
    assert render_comparison(compare(before, after)).splitlines()[0].startswith("Not protected")


# --- rendering and serialisation ------------------------------------------

def test_render_lists_both_halves_and_the_cues():
    rep = r({"location": 0.88, "age": 0.1},
            pii=[PiiSpan("PERSON", 0, 4)],
            cues=[Cue("Green Line", "location", 0.73)])
    out = render(rep)
    assert "Explicit identifiers found: 1 (PERSON)" in out
    assert "Inference risk: HIGH" in out
    assert '"Green Line"' in out


def test_report_holds_no_message_text():
    rep = r({"location": 0.5}, pii=[PiiSpan("PERSON", 0, 4)])
    blob = repr(rep.to_dict())
    assert "start" in blob and "end" in blob
    assert "text" not in blob


def test_spans_accept_plain_dicts_from_presidio():
    rep = compose(pii_spans=[{"entity_type": "PERSON", "start": 0, "end": 4}],
                  risk_scores={"location": 0.5})
    assert rep.explicit.spans[0].is_hard


def test_leakage_delta_is_surfaced_when_present():
    assert "+0.18" in render(r({"location": 0.5}, leakage_delta=0.18))
