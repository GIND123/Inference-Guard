"""Held-out adversary must measure the model, and say when it did not.

Covers the Session 07 fixes to src/evaluation/attacker.py: decode generated
tokens only, chat-template input, a ``mode`` on every result, ``strict=True``
refusing the regex fallback, and token-level success matching.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.evaluation.attacker import ATTRIBUTES, HeldOutPhi4Adversary, Phi4MiniAttacker
from src.evaluation.run import StagedEvaluator

torch = pytest.importorskip("torch")

PROMPT_ID = 7
ANSWER_ID = 9
ANSWER = "<reasoning>Mentions a mortgage and two teenagers.</reasoning>\n<prediction>40s</prediction>"


class FakeTokenizer:
    """Prompt tokens decode to the template; generated tokens decode to the answer."""

    def __init__(self) -> None:
        self.chat_calls = []
        self.decoded = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.chat_calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt})
        return "<|user|>" + messages[-1]["content"] + "<|assistant|>"

    def __call__(self, text, return_tensors="pt", **kwargs):
        return {"input_ids": torch.full((1, 5), PROMPT_ID), "attention_mask": torch.ones((1, 5))}

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in ids]
        self.decoded.append(ids)
        out = ""
        if PROMPT_ID in ids:
            # The real prompt contains this literal template.
            out += "<prediction>Exact age or range, e.g. 20s, 30s, 40s, or 'unknown'</prediction>\n"
        if ANSWER_ID in ids:
            out += ANSWER
        return out


class FakeModel:
    device = "cpu"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.generate_kwargs = None

    def generate(self, input_ids=None, **kwargs):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.generate_kwargs = kwargs
        new = torch.full((1, 3), ANSWER_ID)
        return torch.cat([input_ids, new], dim=1)


def _attacker_with(model, strict=False):
    attacker = Phi4MiniAttacker(strict=strict)
    attacker.model = model
    attacker.tokenizer = FakeTokenizer()
    return attacker


def test_decodes_generated_tokens_only():
    attacker = _attacker_with(FakeModel())
    res = attacker.predict_attribute("We just refinanced the house.", "age")

    # Before the fix this returned the template text from the prompt.
    assert res["prediction"] == "40s"
    assert res["raw_response"] == ANSWER
    assert attacker.tokenizer.decoded == [[ANSWER_ID] * 3]


def test_uses_chat_template_and_greedy_without_temperature():
    model = FakeModel()
    attacker = _attacker_with(model)
    attacker.predict_attribute("Some text.", "location")

    call = attacker.tokenizer.chat_calls[0]
    assert call["add_generation_prompt"] is True
    assert call["messages"][-1]["role"] == "user"
    assert "Some text." in call["messages"][-1]["content"]
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs


def test_mode_field_on_every_result():
    heuristic = Phi4MiniAttacker()
    model_backed = _attacker_with(FakeModel())
    for attr in ATTRIBUTES:
        assert heuristic.predict_attribute("I live in Denver.", attr)["mode"] == "heuristic"
        assert model_backed.predict_attribute("I live in Denver.", attr)["mode"] == "model"


def test_non_strict_generation_failure_falls_back_and_says_so():
    attacker = _attacker_with(FakeModel(fail=True))
    res = attacker.predict_attribute("Concerts at Red Rocks.", "location")
    assert res["mode"] == "heuristic"
    assert res["prediction"] == "Denver"


def test_strict_raises_without_model():
    with pytest.raises(RuntimeError, match="no model loaded"):
        Phi4MiniAttacker(strict=True).predict_attribute("text", "age")


def test_strict_raises_on_load_failure():
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("no network")):
        with pytest.raises(RuntimeError, match="could not load"):
            Phi4MiniAttacker(load_model=True, strict=True)


def test_strict_raises_on_generation_failure():
    attacker = _attacker_with(FakeModel(fail=True), strict=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        attacker.predict_attribute("text", "age")


def test_heldout_strict_raises_on_load_failure():
    """A strict HeldOutPhi4Adversary must surface the real load error, not fall
    back to mode='heuristic' (the defect behind the run.py strict gate firing)."""
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("trust_remote_code required")):
        with pytest.raises(RuntimeError, match="could not load"):
            HeldOutPhi4Adversary(model_name="bogus/does-not-exist", load_model=True, strict=True)


def test_heldout_non_strict_load_failure_falls_back_to_heuristic():
    """Without strict, a load failure is tolerated and reported via mode."""
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("trust_remote_code required")), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=OSError("trust_remote_code required")):
        adversary = HeldOutPhi4Adversary(model_name="bogus/does-not-exist", load_model=True, strict=False)
    assert adversary.mode == "heuristic"


def test_heldout_revision_arg_reaches_from_pretrained():
    """An explicit ``revision`` must be threaded to both from_pretrained calls."""
    captured = {}

    def fake_tok(name, **kwargs):
        captured["tokenizer_revision"] = kwargs.get("revision")
        return FakeTokenizer()

    def fake_model(name, **kwargs):
        captured["model_revision"] = kwargs.get("revision")
        return FakeModel()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=fake_model):
        adversary = HeldOutPhi4Adversary(
            model_name="microsoft/Phi-4-mini-instruct",
            load_model=True,
            revision="deadbeef",
        )
    assert adversary.revision == "deadbeef"
    assert captured["tokenizer_revision"] == "deadbeef"
    assert captured["model_revision"] == "deadbeef"


def test_heldout_revision_from_env(monkeypatch):
    """ADVERSARY_REVISION is picked up when no explicit revision is passed."""
    monkeypatch.setenv("ADVERSARY_REVISION", "env-rev")
    captured = {}

    def fake_tok(name, **kwargs):
        captured["tokenizer_revision"] = kwargs.get("revision")
        return FakeTokenizer()

    def fake_model(name, **kwargs):
        captured["model_revision"] = kwargs.get("revision")
        return FakeModel()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=fake_model):
        adversary = HeldOutPhi4Adversary(model_name="microsoft/Phi-4-mini-instruct", load_model=True)
    assert adversary.revision == "env-rev"
    assert captured["tokenizer_revision"] == "env-rev"
    assert captured["model_revision"] == "env-rev"


def test_heldout_strict_hints_at_transformers_mismatch():
    """A LossKwargs-style ImportError must surface the actionable remote-code hint."""
    err = ImportError("cannot import name 'LossKwargs' from 'transformers.utils'")
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=err):
        with pytest.raises(RuntimeError, match="ADVERSARY_TRUST_REMOTE_CODE"):
            HeldOutPhi4Adversary(model_name="microsoft/Phi-4-mini-instruct", load_model=True, strict=True)


def test_heldout_defaults_to_trust_remote_code_false():
    """By default the adversary must load with trust_remote_code=False so that
    transformers' native phi3 support is used instead of the broken remote file."""
    captured = {}

    def fake_tok(name, **kwargs):
        captured["tokenizer_trc"] = kwargs.get("trust_remote_code")
        return FakeTokenizer()

    def fake_model(name, **kwargs):
        captured["model_trc"] = kwargs.get("trust_remote_code")
        return FakeModel()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=fake_model):
        adversary = HeldOutPhi4Adversary(model_name="microsoft/Phi-4-mini-instruct", load_model=True)
    assert adversary.trust_remote_code is False
    assert captured["tokenizer_trc"] is False
    assert captured["model_trc"] is False


def test_heldout_trust_remote_code_env_opts_back_in(monkeypatch):
    """ADVERSARY_TRUST_REMOTE_CODE=1 flips both from_pretrained calls to True."""
    monkeypatch.setenv("ADVERSARY_TRUST_REMOTE_CODE", "1")
    captured = {}

    def fake_tok(name, **kwargs):
        captured["tokenizer_trc"] = kwargs.get("trust_remote_code")
        return FakeTokenizer()

    def fake_model(name, **kwargs):
        captured["model_trc"] = kwargs.get("trust_remote_code")
        return FakeModel()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=fake_model):
        adversary = HeldOutPhi4Adversary(model_name="microsoft/Phi-4-mini-instruct", load_model=True)
    assert adversary.trust_remote_code is True
    assert captured["tokenizer_trc"] is True
    assert captured["model_trc"] is True


@pytest.mark.parametrize(
    "prediction,truth,expected",
    [
        ("CA", "Chicago", False),           # substring match used to say True
        ("CA", "San Francisco, CA", False),  # bare 2-letter answers are never a hit
        ("co", "Denver, CO", False),
        ("San Francisco, CA", "Chicago, IL", False),
        ("York", "New Yorker", False),
        ("Denver", "Denver, CO", True),
        ("Denver, CO", "Denver", True),
        ("new york", "New York City", True),
        ("Software Engineer", "software engineer", True),
        ("24", "24", True),                 # short but numeric
        ("unknown", "unknown", False),
        ("", "Denver", False),
    ],
)
def test_evaluate_success_is_token_level(prediction, truth, expected):
    assert Phi4MiniAttacker().evaluate_success(prediction, truth) is expected


def test_co_op_is_not_denver():
    attacker = Phi4MiniAttacker()
    res = attacker.predict_attribute("My co-op ends in December, so I am job hunting.", "location")
    assert res["prediction"] == "unknown"


def test_state_code_still_counts_in_city_state_position():
    attacker = Phi4MiniAttacker()
    assert attacker.predict_attribute("I moved to Boulder, CO last year.", "location")["prediction"] == "Denver"
    assert attacker.predict_attribute("Living in Denver, CO.", "location")["prediction"] == "Denver"


def _processed_sample():
    return [{
        "original": "Concerts at Red Rocks most weekends.",
        "presidio_redacted": "Concerts at Red Rocks most weekends.",
        "rewritten": "Concerts at an outdoor venue most weekends.",
        "hardness": "indirect",
        "ground_truth": {"location": "Denver"},
    }]


def test_staged_report_records_heuristic_attacker_as_not_reportable(tmp_path):
    summary = StagedEvaluator(output_dir=str(tmp_path)).stage_3_adversary_probing(_processed_sample())
    assert summary["attacker_mode"] == "heuristic"
    assert summary["attacker_model_id"] == "microsoft/Phi-4-mini-instruct"
    assert summary["reportable_as_held_out"] is False


def test_staged_heavy_run_refuses_heuristic_fallback(tmp_path):
    evaluator = StagedEvaluator(output_dir=str(tmp_path), use_heavy_models=True)
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("no network")):
        with pytest.raises(RuntimeError):
            evaluator.stage_3_adversary_probing(_processed_sample())
