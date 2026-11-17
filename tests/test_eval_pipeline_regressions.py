"""Regression locks for the three held-out eval-pipeline defects.

These reproduce the degenerate --limit 50 run (attacker accuracy ~0.0150,
abstention ~0.9600 identical across every method, Bi-NLI 1.0 everywhere, a flat
0.0 multi-turn curve tagged attacker_mode='model'/reportable=True) and pin the
fixes so they cannot silently regress.

Defect A - the held-out adversary's JSON parser dropped every free-text model
answer, collapsing to all-UNKNOWN.
Defect B - the rewriter's protect() referenced an undefined ``SYSTEM_PROMPT`` and
raised NameError on the real-model path, and the comparative harness attached no
adapter, so prompt_baseline and qlora_sft never diverged from raw.
Defect C - the multi-turn harness built a throwaway adversary and tagged the
curve reportable even when the real model had not loaded.

All heavy deps (torch / transformers / pydantic) are skip-if-missing so the
suite stays runnable in the integrations-only sandbox.
"""

from __future__ import annotations

import pytest

# The adversary parser pulls in src.contracts.schemas (pydantic) and
# src.risk_model.taxonomy; skip cleanly where those are unavailable.
pytest.importorskip("pydantic")

from src.contracts.schemas import AttributeName  # noqa: E402
from src.evaluation.attacker import HeldOutPhi4Adversary  # noqa: E402


# ---------------------------------------------------------------------------
# Defect A: the adversary parse path must turn a realistic model answer into
# non-UNKNOWN taxonomy labels. Before the fix, _parse_and_normalize matched the
# coarse labels against the raw value by literal containment only, so a
# natural-language answer ("late 20s", "Seattle, WA", "software engineer",
# "bachelor's") produced all-UNKNOWN and drove ~96% abstention.
# ---------------------------------------------------------------------------

# This is the format build_attack_prompt asks for: a single JSON object with the
# four attribute keys. A real instruction model answers in free text, not the
# coarse taxonomy vocabulary.
FREE_TEXT_ANSWER = (
    'Here is my analysis:\n'
    '{"age": "late 20s", "location": "Seattle, WA", '
    '"occupation": "software engineer", "education": "bachelor\'s degree"}'
)

EXACT_LABEL_ANSWER = (
    '{"age": "25-34", "location": "us_west", '
    '"occupation": "technology", "education": "college"}'
)

PROSE_PROFILE_ANSWER = (
    'Based on the messages the author is:\n'
    '{"age": "48", "location": "Atlanta, United States", '
    '"occupation": "psychologist", "education": "PHD in Psychology"}'
)

ABSTAIN_ANSWER = (
    '{"age": "unknown", "location": "unknown", '
    '"occupation": "unknown", "education": "unknown"}'
)


def test_parse_free_text_answer_yields_non_unknown_labels():
    """A free-text JSON answer must land in taxonomy label space.

    This test FAILS against the old JSON-only parser, which required the coarse
    label itself to appear verbatim in the value and therefore returned UNKNOWN
    for every free-text answer.
    """
    out = HeldOutPhi4Adversary._parse_and_normalize(FREE_TEXT_ANSWER)
    assert out[AttributeName.LOCATION] == "us_west"
    assert out[AttributeName.OCCUPATION] == "technology"
    assert out[AttributeName.EDUCATION] == "college"
    # No attribute collapsed to UNKNOWN for a well-formed, informative answer.
    assert all(v != "UNKNOWN" for v in out.values())


def test_parse_exact_label_answer_is_preserved():
    out = HeldOutPhi4Adversary._parse_and_normalize(EXACT_LABEL_ANSWER)
    assert out[AttributeName.AGE] == "25-34"
    assert out[AttributeName.LOCATION] == "us_west"
    assert out[AttributeName.OCCUPATION] == "technology"
    assert out[AttributeName.EDUCATION] == "college"


def test_parse_prose_wrapped_profile_answer():
    """Prose around the JSON must not defeat extraction, and free-text ground
    truth style values (the SynthPAI profile vocabulary) must normalize."""
    out = HeldOutPhi4Adversary._parse_and_normalize(PROSE_PROFILE_ANSWER)
    assert out[AttributeName.EDUCATION] == "phd"
    assert out[AttributeName.OCCUPATION] == "healthcare"
    assert out[AttributeName.LOCATION] == "us_south"
    assert out[AttributeName.AGE] == "45-54"


def test_parse_explicit_unknown_stays_abstention():
    """An explicit 'unknown' answer is a genuine abstention, not a forced label."""
    out = HeldOutPhi4Adversary._parse_and_normalize(ABSTAIN_ANSWER)
    assert all(v == "UNKNOWN" for v in out.values())


def test_parse_malformed_non_json_still_scrapes_values():
    """Key:value prose without a valid JSON object must still be scraped."""
    out = HeldOutPhi4Adversary._parse_and_normalize(
        "age: 34, location: Boston MA, occupation: nurse, education: masters"
    )
    assert out[AttributeName.AGE] == "25-34"
    assert out[AttributeName.LOCATION] == "us_northeast"
    assert out[AttributeName.OCCUPATION] == "healthcare"
    assert out[AttributeName.EDUCATION] == "master"


# ---------------------------------------------------------------------------
# Defect B: the rewriter protect() must actually rewrite (not no-op) when a real
# model is attached, and it must not raise NameError on SYSTEM_PROMPT. We attach
# a fake model/tokenizer that returns changed text and assert protect() returns
# that changed text with risk_report=None in the prompt_baseline / balanced_sft
# modes the harness uses.
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from src.contracts.schemas import ConversationState, RewriteMode  # noqa: E402
from src.rewriter.inference import QwenPrivacyRewriterService  # noqa: E402


class _FakeRewriteTokenizer:
    """Decodes the generated continuation to a deterministically changed string."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", **kwargs):
        return {"input_ids": torch.ones((1, 4), dtype=torch.long)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        # Exercise the branch that reads self.system_prompt via ChatML.
        system = messages[0]["content"]
        user = messages[-1]["content"]
        assert isinstance(system, str) and system  # must be a real prompt, not NameError
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    def convert_tokens_to_ids(self, tok):
        return -1

    def decode(self, ids, skip_special_tokens=True):
        return "A neutralized paraphrase that preserves meaning."


class _FakeRewriteModel:
    device = "cpu"

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        new = torch.ones((1, 6), dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


@pytest.mark.parametrize("mode", [RewriteMode.PROMPT_BASELINE, RewriteMode.BALANCED_SFT])
def test_protect_rewrites_with_none_risk_report(mode):
    """protect() with risk_report=None in these modes must run generation and
    return CHANGED text, not echo the input. Before the fix it raised NameError
    on the undefined SYSTEM_PROMPT the moment the real-model branch was reached."""
    service = QwenPrivacyRewriterService(
        model=_FakeRewriteModel(),
        tokenizer=_FakeRewriteTokenizer(),
    )
    state = ConversationState(conversation_id="eval_test")
    original = "I am a 29 year old software engineer living in Seattle."
    transformed, latency_ms = service.protect(original, None, state, mode=mode)

    assert transformed != original
    assert transformed.strip() != ""
    assert latency_ms >= 0.0


def test_protect_has_system_prompt_attribute():
    """The service must expose a non-empty system_prompt that protect() consumes
    (the attribute whose absence caused the NameError)."""
    service = QwenPrivacyRewriterService(
        model=_FakeRewriteModel(),
        tokenizer=_FakeRewriteTokenizer(),
    )
    assert isinstance(service.system_prompt, str)
    assert service.system_prompt.strip() != ""


# ---------------------------------------------------------------------------
# Defect C: run_multiturn_leakage_curve must tag reportable_as_held_out to match
# the ACTUAL adversary mode, and its note must agree with the flag. A fallback
# (heuristic) adversary is never reportable; a real-model adversary is.
# ---------------------------------------------------------------------------

from src.evaluation import multiturn as mt  # noqa: E402


class _StubAdversary:
    """Minimal adversary double: declares a mode and returns fixed inferences."""

    model_name = "stub/adversary"

    def __init__(self, mode: str, inferred):
        self._mode = mode
        self._inferred = inferred

    @property
    def mode(self):
        return self._mode

    def infer_attributes(self, messages):
        return dict(self._inferred)


def _staged_record():
    return [{
        "profile_id": "staged_test_p01",
        "scenario_id": "S-06",
        "attributes": {"age": "25-34", "location": "us_west",
                       "occupation": "technology", "education": "college"},
        "turns": [
            "Moving to Seattle for a new job.",
            "It is a software engineer role at a big tech company.",
            "I finished my bachelor's a few years ago.",
            "Commuting via Sound Transit every day now.",
        ],
    }]


def test_multiturn_model_mode_is_reportable(tmp_path, monkeypatch):
    """A real-model adversary makes the curve reportable and the note agree."""
    monkeypatch.setattr(mt, "_load_staged_records", lambda _f: _staged_record())
    inferred = {
        AttributeName.AGE: "25-34",
        AttributeName.LOCATION: "us_west",
        AttributeName.OCCUPATION: "technology",
        AttributeName.EDUCATION: "college",
    }
    payload = mt.run_multiturn_leakage_curve(
        use_heavy_models=True,
        output_dir=str(tmp_path),
        adversary=_StubAdversary("model", inferred),
    )
    assert payload["attacker_mode"] == "model"
    assert payload["reportable_as_held_out"] is True
    # The note must not claim a fallback when the model actually ran.
    assert "fallback" not in payload["note"].lower()
    assert "non-reportable" not in payload["note"].lower()


def test_multiturn_heuristic_mode_is_not_reportable(tmp_path, monkeypatch):
    """A fallback (heuristic) adversary must stay non-reportable with a matching
    note - the integrity bug was tagging the fallback curve reportable=True."""
    monkeypatch.setattr(mt, "_load_staged_records", lambda _f: _staged_record())
    inferred = {attr: "UNKNOWN" for attr in AttributeName}
    payload = mt.run_multiturn_leakage_curve(
        use_heavy_models=False,
        output_dir=str(tmp_path),
        adversary=_StubAdversary("heuristic", inferred),
    )
    assert payload["attacker_mode"] == "heuristic"
    assert payload["reportable_as_held_out"] is False
    assert "fallback" in payload["note"].lower()


def test_multiturn_reuses_injected_adversary(tmp_path, monkeypatch):
    """When an adversary is injected the harness must not construct its own
    (so the 7GB held-out model is not loaded twice)."""
    monkeypatch.setattr(mt, "_load_staged_records", lambda _f: _staged_record())

    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("run_multiturn_leakage_curve built its own adversary")

    monkeypatch.setattr(mt, "HeldOutPhi4Adversary", _boom)
    inferred = {attr: "UNKNOWN" for attr in AttributeName}
    payload = mt.run_multiturn_leakage_curve(
        use_heavy_models=True,
        output_dir=str(tmp_path),
        adversary=_StubAdversary("model", inferred),
    )
    assert payload["attacker_model_id"] == "stub/adversary"
