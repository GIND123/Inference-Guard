"""Session 12: the checks that sit between "uploaded" and "reported".

scripts/verify_risk_release.py re-derives a released risk classifier's numbers
without the loading code it is checking; scripts/heldout_intervals.py attaches
confidence intervals to a held-out benchmark. All offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from heldout_intervals import intervals, wilson  # noqa: E402


# --- heldout_intervals -----------------------------------------------------------

def test_wilson_matches_known_values():
    lo, hi = wilson(0.5, 100)
    assert round(lo, 3) == 0.404 and round(hi, 3) == 0.596
    assert wilson(0.0, 50)[0] == 0.0 and wilson(1.0, 50)[1] == pytest.approx(1.0)


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson(0.5, 0)
    with pytest.raises(ValueError):
        wilson(1.5, 10)


def _bench(rates):
    attrs = ("age", "location", "occupation", "education")
    return {
        "sample_count": 50,
        "methods": {m: {"overall_attack_success_rate": r} for m, r in rates.items()},
        "per_attribute_attack_success": {m: {a: r for a in attrs} for m, r in rates.items()},
    }


def test_intervals_separate_large_gaps_and_flag_small_ones():
    out = intervals(_bench({"raw": 0.365, "prompt": 0.185, "tuned": 0.150}))
    assert out["methods"]["raw"]["trials"] == 200
    pairs = {(p["a"], p["b"]): p["intervals_overlap"] for p in out["pairs"]}
    assert pairs[("raw", "tuned")] is False
    assert pairs[("prompt", "tuned")] is True


# --- verify_risk_release ---------------------------------------------------------

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from verify_risk_release import (  # noqa: E402
    build_strict,
    compare_claim,
    compare_state_dicts,
    download_revision,
    product_loader_check,
)


def test_a_branch_name_is_not_a_pinned_revision():
    with pytest.raises(ValueError, match="40-character"):
        download_revision("org/model", "main", None)


@pytest.fixture
def tiny_release(tmp_path):
    """A ModernBertRiskClassifier in the hosted layout, backed by a tiny local ModernBERT."""
    from src.risk_model.model import ModernBertRiskClassifier

    backbone_dir = tmp_path / "backbone"
    transformers.ModernBertModel(transformers.ModernBertConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=37, max_position_embeddings=64, pad_token_id=0,
    )).save_pretrained(backbone_dir)
    model = ModernBertRiskClassifier(model_name=str(backbone_dir))
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.classifiers.parameters():
            p.add_(torch.randn_like(p))
    release = tmp_path / "release"
    model.save_pretrained(release)
    config = json.loads((release / "risk_model_config.json").read_text())
    return release, model, config


def test_strict_build_reproduces_the_release(tiny_release):
    release, model, config = tiny_release
    sd = torch.load(release / "model.pt", map_location="cpu")
    rebuilt = build_strict(release, sd, config)
    assert compare_state_dicts(rebuilt.state_dict(), model.state_dict())["identical"]


def test_strict_build_refuses_a_partial_checkpoint(tiny_release):
    release, model, config = tiny_release
    heads_only = {k: v for k, v in model.state_dict().items() if not k.startswith("bert.")}
    with pytest.raises(RuntimeError):
        build_strict(release, heads_only, config)


def test_compare_state_dicts_finds_a_changed_tensor(tiny_release):
    _, model, _ = tiny_release
    a = model.state_dict()
    b = {k: v.clone() for k, v in a.items()}
    key = next(k for k in b if k.startswith("classifiers."))
    b[key] += 1.0
    result = compare_state_dicts(a, b)
    assert not result["identical"] and result["differ"] == [key]


def test_product_loader_check_reports_different_weights(tiny_release):
    from src.risk_model.model import ModernBertRiskClassifier

    release, _, config = tiny_release
    other = ModernBertRiskClassifier(model_name=config["model_name"])  # fresh heads, not the release
    check = product_loader_check(release, other)
    assert check["loads_same_weights"] is False


def test_compare_claim_flags_a_mismatch(tmp_path):
    claimed = tmp_path / "claim.json"
    claimed.write_text(json.dumps({"revision": "x", "metrics": {
        "macro": {"f1": 0.50, "pr_auc": 0.47, "roc_auc": 0.84},
        "per_attribute": {"age": {"f1": 0.18, "pr_auc": 0.15, "roc_auc": 0.73}},
    }}))
    measured = {"macro": {"f1": 0.50, "pr_auc": 0.47, "roc_auc": 0.80},
                "per_attribute": {a: {"f1": 0.18, "pr_auc": 0.15, "roc_auc": 0.73}
                                  for a in ("age", "location", "occupation", "education")}}
    result = compare_claim(measured, claimed)
    assert result["all_match"] is False
    assert [r for r in result["rows"] if not r["match"]] == [
        {"scope": "macro", "metric": "roc_auc", "claimed": 0.84, "measured": 0.80, "match": False}]
