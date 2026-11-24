"""Score the risk classifier the product serves, on the validation split.

The container serves `RISK_MODEL_ID` (default AT707/InferenceGuard-Roberta-Risk).
Until this script, nothing had measured whether it detects anything: the
hosting checks verified that the weights load and the heads have the right
shape. Every risk-model number we had published was ModernBERT (Session 05).

Comparable to Session 05 by construction:
    labels   src.risk_model.train.extract_inferability_labels
             (a SynthPAI annotator's estimate is non-empty)
    score    sigmoid(logit / T) per attribute: what CalibratedRiskPredictor
             serves for a single-logit head (src/risk_model/calibrate.py)
    metrics  the F1 (at 0.5), PR-AUC, ROC-AUC and balanced-accuracy functions
             in src.risk_model.train that produced the Session 05 numbers

Fail-closed: the product's loader can silently fall back to random heads,
and its predictor to keyword heuristics. This script refuses both. The Hub
checkpoint must load with no missing or unexpected keys, the product loader
must produce the same parameters, and scores come from the model's forward
pass, never from a fallback.

Also reported: expected calibration error, how the served bands (0.40/0.70 in
calibrate.py) and the documented bands (0.30/0.60 in risk_bands.py) split the
scores, and per-attribute band edges fitted on these scores with
src.product.thresholds (HIGH for precision, LOW for recall). Fitting the bands
on val and reporting them on val is in-sample; the test split is not touched.

Two checkpoint layouts are supported, detected from the files in the repo:

    roberta    RobertaRiskClassifier (config.json + model.safetensors), one
               logit per attribute. Score: sigmoid(logit / T) as above.
    multitask  MultiTaskModernBERTRiskModel (multitask_heads.pt +
               calibration_params.json, Session 11). Each head predicts a
               taxonomy class, not inferability. Score: the
               inferability_risk_score CalibratedRiskPredictor serves, built
               from the hosted calibration_params.json, i.e. exactly the number
               the product and src/evaluation/run.py would use. The product
               loader reads multitask_heads.pt with strict=False, so this
               script re-reads the file and requires every model tensor to be
               present and identical.

Usage:
    python scripts/score_risk_model.py
    python scripts/score_risk_model.py --model-id AT707/InferenceGuard-Roberta-Risk --device cpu
    python scripts/score_risk_model.py --model-id AT707/inferenceguard-modernbert-risk
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATTRIBUTES = ("age", "location", "occupation", "education")
SERVED_BANDS = {"medium": 0.40, "high": 0.70}      # src/risk_model/calibrate.py, binary heads
DOCUMENTED_BANDS = {"medium": 0.30, "high": 0.60}  # src/product/risk_bands.py, model card, UI

# Session 05, ModernBERT-base, validation split: reports/session05.md.
SESSION05_MODERNBERT = {
    "macro": {"f1": 0.461, "pr_auc": 0.457, "roc_auc": 0.853, "balanced_accuracy": 0.750},
    "per_attribute": {
        "occupation": {"f1": 0.618, "pr_auc": 0.612, "roc_auc": 0.870, "pos_count": 281},
        "education": {"f1": 0.534, "pr_auc": 0.564, "roc_auc": 0.883, "pos_count": 146},
        "location": {"f1": 0.462, "pr_auc": 0.484, "roc_auc": 0.907, "pos_count": 56},
        "age": {"f1": 0.232, "pr_auc": 0.169, "roc_auc": 0.753, "pos_count": 83},
    },
}


def load_records(split: str, manifest_path: Path) -> List[Dict[str, Any]]:
    """The committed split, checked against its manifest hash. Test is refused."""
    if split == "test":
        raise ValueError("the test split is reserved for the held-out evaluation")
    path = ROOT / f"data/splits/{split}.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest[split]["sha256"]:
        raise ValueError(f"{path} does not match {manifest_path} ({digest} != {manifest[split]['sha256']})")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_served_model(model_id: str, device: str):
    """Load strictly, then prove the product loader yields the same weights."""
    import torch

    from src.risk_model.model import load_risk_classifier
    from src.risk_model.roberta_model import RobertaRiskClassifier

    strict, info = RobertaRiskClassifier.from_pretrained(model_id, output_loading_info=True)
    problems = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    if problems:
        raise RuntimeError(f"{model_id} did not load cleanly: {problems}")
    layers = getattr(getattr(strict.roberta, "config", None), "num_hidden_layers", None)

    served = load_risk_classifier(model_id, device="cpu")
    strict_sd, served_sd = strict.state_dict(), served.state_dict()
    if strict_sd.keys() != served_sd.keys() or any(
        not torch.equal(strict_sd[k], served_sd[k]) for k in strict_sd
    ):
        raise RuntimeError("the product loader (load_risk_classifier) produced different weights than the checkpoint")
    del strict

    served = served.to(device).eval()
    return served, {"backbone_layers": layers, "parameters": sum(p.numel() for p in served.parameters())}


def detect_layout(model_id: str) -> str:
    """'modernbert_bce' when risk_model_config.json is present, 'multitask' for multitask_heads.pt, else 'roberta'."""
    from src.risk_model.model import _available_model_files

    path = Path(model_id)
    files = {p.name for p in path.iterdir()} if path.is_dir() else _available_model_files(model_id)
    if "risk_model_config.json" in files:
        return "modernbert_bce"
    return "multitask" if "multitask_heads.pt" in files else "roberta"


def load_modernbert_bce_model(model_id: str, device: str):
    """Load ModernBertRiskClassifier through the product loader."""
    from src.risk_model.model import ModernBertRiskClassifier, load_risk_classifier

    served = load_risk_classifier(model_id, device=device)
    if not isinstance(served, ModernBertRiskClassifier):
        raise RuntimeError(f"Expected ModernBertRiskClassifier for {model_id}, got {type(served).__name__}")

    tokenizer_name = getattr(served, "model_name", "answerdotai/ModernBERT-base")
    facts = {
        "layout": "modernbert_bce",
        "model_name": tokenizer_name,
        "num_heads": served.num_heads,
        "parameters": sum(p.numel() for p in served.parameters()),
        "temperatures": served.get_temperatures(),
    }
    return served, facts


def score_modernbert_bce(model, tokenizer, texts: Sequence[str], device: str, batch_size: int):
    """Score texts using ModernBertRiskClassifier sigmoid probabilities with temperature."""
    import torch

    probs: Dict[str, List[float]] = {a: [] for a in ATTRIBUTES}
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), return_tensors="pt", truncation=True,
                        max_length=512, padding=True).to(device)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], apply_temperature=True)
        for k, a in enumerate(ATTRIBUTES):
            p = torch.sigmoid(logits[k].view(-1)).cpu().tolist()
            probs[a].extend(p)
    return probs


def load_multitask_model(model_id: str, device: str):
    """Load through the product loader, then prove every tensor came from the checkpoint.

    _load_multitask_modernbert uses load_state_dict(strict=False) and only
    requires one head key to match, so a checkpoint missing the encoder would
    load silently on top of the stock base model. Refuse that here.
    """
    import torch

    from src.risk_model.model import MultiTaskModernBERTRiskModel, _resolve_repo_file, load_risk_classifier

    served = load_risk_classifier(model_id, device="cpu")
    if not isinstance(served, MultiTaskModernBERTRiskModel):
        raise RuntimeError(f"the product loader returned {type(served).__name__} for {model_id}, "
                           "not MultiTaskModernBERTRiskModel (it fell back)")
    weights = _resolve_repo_file(model_id, "multitask_heads.pt")
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
        ckpt = ckpt["state_dict"]
    ckpt = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    model_sd = served.state_dict()
    missing = sorted(k for k in model_sd if k not in ckpt)
    unexpected = sorted(k for k in ckpt if k not in model_sd)
    mismatched = sorted(k for k in model_sd if k in ckpt and not torch.equal(model_sd[k], ckpt[k].to(model_sd[k].dtype)))
    if missing or unexpected or mismatched:
        raise RuntimeError(f"{model_id} did not load cleanly: missing={missing[:5]} ({len(missing)}), "
                           f"unexpected={unexpected[:5]} ({len(unexpected)}), mismatched={mismatched[:5]}")
    calibration = getattr(served, "_calibration_params", None)
    if not isinstance(calibration, dict):
        raise RuntimeError(f"{model_id} has no readable calibration_params.json; the product would serve "
                           "an identity scaler, which is not what was published")
    encoder_cfg = getattr(served.encoder, "config", None)
    facts = {
        "layout": "multitask",
        "backbone_layers": getattr(encoder_cfg, "num_hidden_layers", None),
        "base_model": getattr(encoder_cfg, "_name_or_path", None),
        "checkpoint_tensors": len(ckpt),
        "parameters": sum(p.numel() for p in served.parameters()),
        "calibration_params": calibration,
    }
    return served.to(device).eval(), facts


def score_multitask(model, tokenizer, texts: Sequence[str], device: str, batch_size: int):
    """The product's inferability_risk_score and band per attribute, from the forward pass only."""
    import torch

    from src.contracts.schemas import AttributeName
    from src.risk_model.predict import CalibratedRiskPredictor

    scaler = CalibratedRiskPredictor._scaler_from_calibration_params(model._calibration_params)
    probs: Dict[str, List[float]] = {a: [] for a in ATTRIBUTES}
    bands: Dict[str, List[str]] = {a: [] for a in ATTRIBUTES}
    top1: Dict[str, List[int]] = {a: [] for a in ATTRIBUTES}
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), return_tensors="pt", truncation=True,
                        max_length=512, padding=True).to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
        for a in ATTRIBUTES:
            top1[a].extend(logits[a].argmax(dim=-1).cpu().tolist())
        for j in range(enc["input_ids"].shape[0]):
            dists = scaler.score_logits({a: logits[a][j:j + 1].cpu() for a in ATTRIBUTES})
            for a in ATTRIBUTES:
                d = dists[AttributeName(a)]
                probs[a].append(float(d.inferability_risk_score))
                bands[a].append(d.risk_band.value if hasattr(d.risk_band, "value") else str(d.risk_band))
    return probs, bands, top1


def class_accuracy(records, top1: Dict[str, List[int]], train_records) -> Dict[str, Any]:
    """The multitask heads' own task: top-1 taxonomy class against the profile's class.

    Labels come from SynthPAIMultiTaskDataset, the dataset the heads were trained
    on. The baseline always predicts the most common class in the TRAIN split.
    """
    from collections import Counter

    from src.risk_model.train import SynthPAIMultiTaskDataset

    truth = SynthPAIMultiTaskDataset(records).targets
    train_truth = SynthPAIMultiTaskDataset(train_records).targets
    out: Dict[str, Any] = {}
    for a in ATTRIBUTES:
        y = [t[a] for t in truth]
        majority = Counter(t[a] for t in train_truth).most_common(1)[0][0]
        out[a] = {
            "accuracy": round(sum(int(p == t) for p, t in zip(top1[a], y)) / len(y), 4),
            "train_majority_baseline": round(sum(int(t == majority) for t in y) / len(y), 4),
            "distinct_predicted_classes": len(set(top1[a])),
        }
    return out


def served_band_shares(bands: Sequence[str], y_true: Sequence[int]) -> Dict[str, Any]:
    """Band shares from the bands the product actually assigned."""
    n, pos = len(bands), sum(y_true)
    low = [str(b).upper() == "LOW" for b in bands]
    high = [str(b).upper() == "HIGH" for b in bands]
    tp_high = sum(1 for h, t in zip(high, y_true) if h and t)
    return {
        "share_low": round(sum(low) / n, 4),
        "share_high": round(sum(high) / n, 4),
        "high_precision": round(tp_high / sum(high), 4) if sum(high) else None,
        "positives_missed_in_low": round(sum(1 for lo, t in zip(low, y_true) if lo and t) / pos, 4) if pos else None,
    }


def score(model, tokenizer, texts: Sequence[str], device: str, batch_size: int) -> Dict[str, List[float]]:
    """sigmoid(logit / T) per attribute, from the forward pass only."""
    import torch

    temps = {a: float(model.temperatures[a].item()) for a in ATTRIBUTES}
    probs: Dict[str, List[float]] = {a: [] for a in ATTRIBUTES}
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), return_tensors="pt", truncation=True,
                        max_length=512, padding=True).to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"])
        logits = out["logits"]
        for a in ATTRIBUTES:
            probs[a].extend(torch.sigmoid(logits[a].view(-1) / temps[a]).cpu().tolist())
    return probs


def band_shares(y_prob: Sequence[float], y_true: Sequence[int], edges: Dict[str, float]) -> Dict[str, Any]:
    """How a pair of edges splits the scores, and how many positives land in LOW."""
    n, pos = len(y_prob), sum(y_true)
    low = [p < edges["medium"] for p in y_prob]
    high = [p >= edges["high"] for p in y_prob]
    tp_high = sum(1 for h, t in zip(high, y_true) if h and t)
    return {
        "share_low": round(sum(low) / n, 4),
        "share_high": round(sum(high) / n, 4),
        "high_precision": round(tp_high / sum(high), 4) if sum(high) else None,
        "positives_missed_in_low": round(sum(1 for lo, t in zip(low, y_true) if lo and t) / pos, 4) if pos else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="AT707/InferenceGuard-Roberta-Risk")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--split-manifest", default="data/manifests/split_manifest.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default="artifacts/risk_eval")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import model_info
    from transformers import AutoTokenizer

    from src.product.thresholds import expected_calibration_error, fit_thresholds
    from src.risk_model.train import (
        binary_balanced_accuracy,
        binary_f1_score,
        binary_pr_auc,
        binary_roc_auc,
        extract_inferability_labels,
    )

    records = load_records(args.split, ROOT / args.split_manifest)
    labels = {a: [int(extract_inferability_labels(r)[k]) for r in records] for k, a in enumerate(ATTRIBUTES)}

    layout = detect_layout(args.model_id)
    texts = [r.get("text", "") for r in records]
    t0 = time.perf_counter()
    served_bands: Optional[Dict[str, List[str]]] = None
    class_task: Optional[Dict[str, Any]] = None
    if layout == "modernbert_bce":
        model, model_facts = load_modernbert_bce_model(args.model_id, args.device)
        tokenizer_id = model_facts.get("model_name", "answerdotai/ModernBERT-base")
        tok_source = args.model_id if (Path(args.model_id).exists() and (Path(args.model_id) / "tokenizer.json").exists()) else tokenizer_id
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        probs = score_modernbert_bce(model, tokenizer, texts, args.device, args.batch_size)
        temperatures = dict(zip(ATTRIBUTES, model.get_temperatures()))
    elif layout == "multitask":
        model, model_facts = load_multitask_model(args.model_id, args.device)
        # The tokenizer CalibratedRiskPredictor.from_pretrained uses for this layout: the repo's own.
        tokenizer_id = args.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        probs, served_bands, top1 = score_multitask(model, tokenizer, texts, args.device, args.batch_size)
        class_task = class_accuracy(records, top1, load_records("train", ROOT / args.split_manifest))
        temperatures = model_facts["calibration_params"].get("temperatures")
    else:
        model, model_facts = load_served_model(args.model_id, args.device)
        model_facts["layout"] = "roberta"
        # The tokenizer CalibratedRiskPredictor.from_pretrained uses: the config's base model.
        tokenizer_id = getattr(model.config, "base_model_name", args.model_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        probs = score(model, tokenizer, texts, args.device, args.batch_size)
        temperatures = {a: float(model.temperatures[a].item()) for a in ATTRIBUTES}
    seconds = time.perf_counter() - t0

    per_attr: Dict[str, Any] = {}
    for a in ATTRIBUTES:
        y_true, y_prob = np.array(labels[a], dtype=float), np.array(probs[a])
        y_pred = (y_prob >= 0.5).astype(float)
        per_attr[a] = {
            "f1": round(binary_f1_score(y_true, y_pred), 4),
            "pr_auc": round(binary_pr_auc(y_true, y_prob), 4),
            "roc_auc": round(binary_roc_auc(y_true, y_prob), 4),
            "balanced_accuracy": round(binary_balanced_accuracy(y_true, y_pred), 4),
            "pos_count": int(y_true.sum()),
            "ece": expected_calibration_error(labels[a], probs[a]),
            "mean_score": round(float(y_prob.mean()), 4),
            "bands_served_0.40_0.70": band_shares(probs[a], labels[a], SERVED_BANDS),
            "bands_documented_0.30_0.60": band_shares(probs[a], labels[a], DOCUMENTED_BANDS),
        }
        if served_bands is not None:
            # Multitask scores are already on the UI scale; these are the bands the product assigned.
            per_attr[a]["bands_as_served"] = served_band_shares(served_bands[a], labels[a])
    macro = {m: round(float(np.mean([per_attr[a][m] for a in ATTRIBUTES])), 4)
             for m in ("f1", "pr_auc", "roc_auc", "balanced_accuracy", "ece")}

    fits = fit_thresholds(labels, probs)
    fitted = {
        a: {
            "medium_edge": round(f.medium_edge, 4), "high_edge": round(f.high_edge, 4),
            "high_precision": round(f.high_precision, 4), "high_recall": round(f.high_recall, 4),
            "low_recall": round(f.low_recall, 4), "n_positive": f.n_positive,
            "usable": f.usable, "degraded": f.degraded, "note": f.note,
        }
        for a, f in fits.items()
    }

    try:
        revision = model_info(args.model_id).sha
    except Exception:
        revision = None

    result = {
        "model_id": args.model_id,
        "revision": revision,
        "model": model_facts,
        "temperatures": temperatures,
        "tokenizer": tokenizer_id,
        "split": args.split,
        "comments": len(records),
        "profiles": len({r.get("profile_id") or r.get("author") for r in records}),
        "metrics": {"macro": macro, "per_attribute": per_attr},
        "session05_modernbert_for_comparison": SESSION05_MODERNBERT,
        "fitted_bands_in_sample": fitted,
        "multitask_class_task": class_task,
        "score_definition": ("CalibratedRiskPredictor inferability_risk_score from hosted calibration_params.json"
                             if layout == "multitask" else "sigmoid(logit / T), the binary-head path in calibrate.py"),
        "notes": [
            "Fitted bands are fitted and evaluated on the same split (in-sample).",
            "Which split the hosted model was trained on is not recorded in its model card.",
        ],
        "scores": {"ids": [r.get("id") for r in records], **{a: [round(p, 5) for p in probs[a]] for a in ATTRIBUTES}},
        "command": "python scripts/score_risk_model.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": {"device": args.device, "platform": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__},
        "seconds": round(seconds, 1),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model_id.split('/')[-1].lower()}_{args.split}_{dt.date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"{args.model_id} @ {revision} on {args.split}: {len(records)} comments, "
          f"{result['profiles']} profiles, {seconds:.0f}s on {args.device}")
    print(f"{'attribute':<12}{'F1':>7}{'PR-AUC':>8}{'ROC-AUC':>9}{'ECE':>7}{'pos':>6}   ModernBERT S05 F1/PR/ROC")
    for a in ATTRIBUTES:
        m, s = per_attr[a], SESSION05_MODERNBERT["per_attribute"][a]
        print(f"{a:<12}{m['f1']:>7.3f}{m['pr_auc']:>8.3f}{m['roc_auc']:>9.3f}{m['ece']:>7.3f}{m['pos_count']:>6}"
              f"   {s['f1']:.3f}/{s['pr_auc']:.3f}/{s['roc_auc']:.3f} ({s['pos_count']})")
    s = SESSION05_MODERNBERT["macro"]
    print(f"{'macro':<12}{macro['f1']:>7.3f}{macro['pr_auc']:>8.3f}{macro['roc_auc']:>9.3f}{macro['ece']:>7.3f}"
          f"{'':>6}   {s['f1']:.3f}/{s['pr_auc']:.3f}/{s['roc_auc']:.3f}")
    print(f"wrote {out.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()
