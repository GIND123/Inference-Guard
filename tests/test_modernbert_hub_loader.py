"""The served risk classifier must load the hosted multitask ModernBERT, not fall back to untrained.

The user's hosted repo ``AT707/inferenceguard-modernbert-risk`` contains only
``['.gitattributes', 'README.md', 'calibration_params.json', 'multitask_heads.pt',
'tokenizer.json', 'tokenizer_config.json']`` - no ``config.json``/safetensors and no
``risk_model_config.json``. Before FEAT-001, ``load_risk_classifier`` had no branch for
this layout from the Hub, so it silently built an UNTRAINED RobertaRiskClassifier and the
held-out benchmark ran on random heads.

These tests exercise the multitask-detection path offline: a tiny
``MultiTaskModernBERTRiskModel`` built on a mock backbone, its ``state_dict`` saved to
``multitask_heads.pt``, and the ``huggingface_hub`` helpers monkeypatched so no network,
no real weights, and no base-model download are required. Each assertion would fail if the
loader still fell back to the untrained RoBERTa for the multitask layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch",
    reason="needs the optional torch dependency; see tests/README.md",
)
import torch.nn as nn  # noqa: E402

import src.risk_model.model as model_module  # noqa: E402
from src.contracts.schemas import AttributeName  # noqa: E402
from src.risk_model.calibrate import AttributeTemperatureScaler  # noqa: E402
from src.risk_model.model import (  # noqa: E402
    MultiTaskModernBERTRiskModel,
    load_risk_classifier,
)
from src.risk_model.predict import CalibratedRiskPredictor  # noqa: E402

HIDDEN = 64


class DummyBackbone(nn.Module):
    """Mock backbone simulating ModernBERT embedding and sequence hidden states.

    Mirrors tests/test_risk_model_v2.py so no real ModernBERT download is needed.
    """

    def __init__(self, hidden_size: int = HIDDEN, vocab_size: int = 500):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        return inputs_embeds


class DictTokenizer:
    """Tokenizer returning plain Python dicts rather than HuggingFace BatchEncoding."""

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        tokens = [min(abs(hash(w)) % self.vocab_size, self.vocab_size - 1) for w in text.split()]
        if not tokens:
            tokens = [0]
        tokens = tokens[:max_length]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
        }


# The user's EXACT uploaded listing for AT707/inferenceguard-modernbert-risk.
HUB_REPO_ID = "AT707/inferenceguard-modernbert-risk"
HUB_FILES = {
    ".gitattributes",
    "README.md",
    "calibration_params.json",
    "multitask_heads.pt",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _build_tiny_multitask_model():
    """A MultiTaskModernBERTRiskModel on a DummyBackbone: no network, no base download."""
    return MultiTaskModernBERTRiskModel(backbone=DummyBackbone(hidden_size=HIDDEN), hidden_size=HIDDEN)


@pytest.fixture
def multitask_checkpoint(tmp_path):
    """A trained-looking multitask checkpoint: multitask_heads.pt + calibration_params.json.

    Heads are perturbed away from init so "weights actually loaded" is observable.
    """
    model = _build_tiny_multitask_model()
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.classifiers.parameters():
            p.add_(torch.randn_like(p))
    saved = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save(model.state_dict(), tmp_path / "multitask_heads.pt")
    AttributeTemperatureScaler().save(tmp_path / "calibration_params.json")
    return tmp_path, saved


@pytest.fixture
def patched_base_model(monkeypatch):
    """Patch MultiTaskModernBERTRiskModel so the loader builds it on a DummyBackbone.

    ``load_risk_classifier`` would otherwise call
    ``MultiTaskModernBERTRiskModel(model_name='answerdotai/ModernBERT-base')`` which needs a
    network download. We keep the real class and substitute a factory that returns a tiny
    model, so the multitask branch runs fully offline.
    """
    real_cls = MultiTaskModernBERTRiskModel
    monkeypatch.setattr(
        model_module,
        "MultiTaskModernBERTRiskModel",
        lambda *a, **k: real_cls(backbone=DummyBackbone(hidden_size=HIDDEN), hidden_size=HIDDEN),
    )
    return real_cls


def test_local_dir_loads_multitask_weights(multitask_checkpoint, patched_base_model):
    """Test A: a local dir with multitask_heads.pt routes to the multitask branch and loads weights."""
    tmp_path, saved = multitask_checkpoint
    real_cls = patched_base_model

    loaded = load_risk_classifier(str(tmp_path), device="cpu")

    # Branch (b) ran: this is the multitask model, not the untrained RoBERTa fallback.
    assert isinstance(loaded, real_cls)

    # forward() yields a {'logits': {attr: tensor}} dict for all four attributes.
    input_ids = torch.randint(0, 400, (2, 8))
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    out = loaded(input_ids, attention_mask)
    assert "logits" in out
    assert set(out["logits"].keys()) == {attr.value for attr in AttributeName}

    # The trained classifier weights were actually loaded, not reinitialized.
    loaded_state = loaded.state_dict()
    classifier_keys = [k for k in saved if "classifiers" in k]
    assert classifier_keys, "expected classifier parameters in the saved state dict"
    assert all(torch.equal(loaded_state[k], saved[k]) for k in classifier_keys)


def test_hub_repo_id_routes_to_multitask_branch(multitask_checkpoint, patched_base_model, monkeypatch):
    """Test B: a Hub repo id whose listing has multitask_heads.pt loads the multitask model, not untrained RoBERTa."""
    tmp_path, _ = multitask_checkpoint
    real_cls = patched_base_model

    def fake_available(model_name_or_path):
        assert model_name_or_path == HUB_REPO_ID
        return set(HUB_FILES)

    def fake_resolve(model_name_or_path, filename):
        assert model_name_or_path == HUB_REPO_ID
        if filename in {"multitask_heads.pt", "calibration_params.json"}:
            return str(tmp_path / filename)
        # e.g. multitask_config.json is not in the user's repo.
        return None

    monkeypatch.setattr(model_module, "_available_model_files", fake_available)
    monkeypatch.setattr(model_module, "_resolve_repo_file", fake_resolve)

    loaded = load_risk_classifier(HUB_REPO_ID, device="cpu")

    # Proves the Hub multitask case no longer falls back to the untrained RobertaRiskClassifier.
    assert isinstance(loaded, real_cls)
    from src.risk_model.roberta_model import RobertaRiskClassifier

    assert not isinstance(loaded, RobertaRiskClassifier)


def test_sidecar_head_size_mismatch_falls_back_loudly(multitask_checkpoint, patched_base_model, caplog):
    """A multitask_config.json whose num_classes_per_head disagrees with the taxonomy must not silently load.

    Issue #1 (review v1): a taxonomy/weights drift used to degrade to the untrained
    fallback via a swallowed shape error. The loader now raises a named mismatch
    inside _load_multitask_modernbert; load_risk_classifier catches it in branch (b)
    and falls through loudly, logging the taxonomy-mismatch cause. This locks in that
    the drift is reported rather than masquerading as a loaded model.
    """
    import json as _json

    from src.risk_model.taxonomy import ATTRIBUTE_TAXONOMY

    tmp_path, _ = multitask_checkpoint

    # Declare a wrong head size for exactly one attribute (occupation has 7 labels).
    occupation = AttributeName.OCCUPATION.value
    correct = {attr.value: len(labels) for attr, labels in ATTRIBUTE_TAXONOMY.items()}
    wrong = dict(correct)
    wrong[occupation] = correct[occupation] + 1
    (tmp_path / "multitask_config.json").write_text(
        _json.dumps({"base_model": "answerdotai/ModernBERT-base", "num_classes_per_head": wrong}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="src.risk_model.model"):
        loaded = load_risk_classifier(str(tmp_path), device="cpu")

    # Branch (b) raised the named ValueError and fell through: not the multitask model.
    assert not isinstance(loaded, MultiTaskModernBERTRiskModel)

    messages = [r.getMessage() for r in caplog.records]
    assert any("head size mismatch" in m for m in messages)
    assert any(occupation in m for m in messages)


def test_detected_multitask_env_failure_raises_not_untrained(multitask_checkpoint, monkeypatch, caplog):
    """A detected multitask checkpoint that fails to load for an ENVIRONMENT reason must fail loudly.

    The user's run detected multitask_heads.pt for the Hub repo but the ModernBERT
    encoder build raised ``RuntimeError: ... No module named 'torchvision.ops' ...``
    (a broken torchvision breaking transformers' lazy vision imports). Previously the
    broad ``except Exception`` swallowed this and degraded to an UNTRAINED
    RobertaRiskClassifier, serving random risk scores after a 40-minute run. Branch (b)
    now re-raises non-ValueError failures with the real cause instead of falling through.
    """
    tmp_path, _ = multitask_checkpoint

    def boom(*a, **k):
        raise RuntimeError(
            "Failed to import transformers.models.timm_wrapper.configuration_timm_wrapper "
            "... No module named 'torchvision.ops'; 'torchvision' is not a package"
        )

    monkeypatch.setattr(model_module, "_load_multitask_modernbert", boom)

    from src.risk_model.roberta_model import RobertaRiskClassifier

    with caplog.at_level(logging.ERROR, logger="src.risk_model.model"):
        with pytest.raises(RuntimeError) as excinfo:
            load_risk_classifier(str(tmp_path), device="cpu")

    # Actionable message that names the environment cause and refuses to degrade.
    assert "could not load it due to an environment" in str(excinfo.value)
    assert "torchvision" in str(excinfo.value)
    # The original exception is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    # It did NOT silently build the untrained RoBERTa fallback.
    messages = [r.getMessage() for r in caplog.records]
    assert not any("UNTRAINED" in m for m in messages)


def test_detected_multitask_value_error_falls_through(multitask_checkpoint, monkeypatch, caplog):
    """A ValueError from _load_multitask_modernbert (arch mismatch) still falls through to the next format.

    ValueError is the "these weights are genuinely not this architecture" signal, so the
    loader must keep its legacy behavior: log and continue to RoBERTa / the loud untrained
    fallback, rather than raising.
    """
    tmp_path, _ = multitask_checkpoint

    def arch_mismatch(*a, **k):
        raise ValueError(
            "multitask_heads.pt did not match MultiTaskModernBERTRiskModel "
            "(no classifier/attention_pooler keys loaded)."
        )

    monkeypatch.setattr(model_module, "_load_multitask_modernbert", arch_mismatch)

    from src.risk_model.roberta_model import RobertaRiskClassifier

    with caplog.at_level(logging.WARNING, logger="src.risk_model.model"):
        loaded = load_risk_classifier(str(tmp_path), device="cpu")

    # Fell through to the loud untrained RoBERTa fallback instead of raising.
    assert isinstance(loaded, RobertaRiskClassifier)
    messages = [r.getMessage() for r in caplog.records]
    assert any("falling through to other formats" in m for m in messages)


def test_unknown_layout_falls_back_loudly(tmp_path, caplog):
    """Test C: an unrecognized/empty layout still hits the loud untrained fallback naming files found."""
    empty_dir = tmp_path / "unknown-layout"
    empty_dir.mkdir()
    (empty_dir / "unrelated.txt").write_text("not a model", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="src.risk_model.model"):
        load_risk_classifier(str(empty_dir), device="cpu")

    messages = [r.getMessage() for r in caplog.records]
    assert any("UNTRAINED" in m for m in messages)
    assert any("found files" in m for m in messages)


def test_predictor_wiring_from_loaded_multitask(multitask_checkpoint, patched_base_model):
    """Test D: CalibratedRiskPredictor on the loaded multitask model yields a 4-attribute RiskReport."""
    tmp_path, _ = multitask_checkpoint

    loaded = load_risk_classifier(str(tmp_path), device="cpu")

    predictor = CalibratedRiskPredictor(
        model=loaded,
        tokenizer=DictTokenizer(),
        scaler=AttributeTemperatureScaler(),
    )

    report = predictor.predict("I moved to Seattle for a software engineering job.")
    assert len(report.attributes) == 4
    for attr in AttributeName:
        assert attr in report.attributes
        assert 0.0 <= report.attributes[attr].inferability_risk_score <= 1.0
