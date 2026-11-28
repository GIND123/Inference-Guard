"""Independently verify a released risk classifier before its numbers go in a report.

Session 10 published a RoBERTa that was never trained; Session 11 found the
multitask ModernBERT near chance. Both were reported or served before anyone
measured them. This script is the check that should run between "uploaded" and
"reported", for the binary-head ModernBertRiskClassifier layout
(risk_model_config.json + model.pt / model.safetensors).

It does not reuse the loading code it is checking. For a PINNED Hub revision it:

  1. downloads that revision and loads it strictly (every tensor must match);
  2. if the repo has both model.pt and model.safetensors, requires them to be
     identical (the product loads one, training saved the other);
  3. counts backbone tensors identical to the stock base model, so an untrained
     upload (Session 10) fails here instead of in production;
  4. scores the validation split with sigmoid(logit / T), using Session 05's
     labels and metric functions, and fits bands with src.product.thresholds;
  5. compares the result with a claimed metrics artifact, if one is given;
  6. checks whether this checkout's product loader (load_risk_classifier)
     yields the same weights, i.e. whether the code on this branch can serve it.

The test split is never read.

Usage:
    python scripts/verify_risk_release.py \\
        --model-id AT707/inferenceguard-modernbert-risk \\
        --revision 4cecb3c90e84112ce5a5e3ed173060a1ab9609a7 \\
        --claimed artifacts/risk_eval/inferenceguard-modernbert-risk_val_2026-11-18.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ATTRIBUTES = ("age", "location", "occupation", "education")
SERVED_BANDS = {"medium": 0.30, "high": 0.60}  # src/product/risk_bands.py
CLAIM_TOLERANCE = 0.005


def download_revision(model_id: str, revision: str, cache_dir: Optional[str]) -> Path:
    """The pinned revision's files, locally. A branch name is refused."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"--revision must be a full 40-character commit sha, got {revision!r}")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id, revision=revision, cache_dir=cache_dir,
        allow_patterns=["risk_model_config.json", "calibration_params.json", "config.json",
                        "model.pt", "model.safetensors", "tokenizer*", "special_tokens_map.json"],
    )
    return Path(path)


def read_state_dicts(local: Path) -> Dict[str, Dict[str, Any]]:
    """Every weights file in the snapshot, keyed by filename."""
    import torch

    found: Dict[str, Dict[str, Any]] = {}
    if (local / "model.pt").exists():
        sd = torch.load(local / "model.pt", map_location="cpu")
        found["model.pt"] = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    if (local / "model.safetensors").exists():
        from safetensors.torch import load_file

        found["model.safetensors"] = load_file(str(local / "model.safetensors"))
    if not found:
        raise FileNotFoundError(f"no model.pt or model.safetensors in {local}")
    return found


def compare_state_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    shared = set(a) & set(b)
    differ = sorted(k for k in shared if a[k].shape != b[k].shape or not torch.equal(a[k].float(), b[k].float()))
    return {"only_in_first": sorted(set(a) - set(b)), "only_in_second": sorted(set(b) - set(a)),
            "differ": differ, "identical": not differ and set(a) == set(b)}


def build_strict(local: Path, state_dict: Dict[str, Any], config: Dict[str, Any]):
    """ModernBertRiskClassifier with every tensor from the checkpoint (strict=True)."""
    from src.risk_model.model import ModernBertRiskClassifier

    model = ModernBertRiskClassifier(
        model_name=config.get("model_name", "answerdotai/ModernBERT-base"),
        num_heads=config.get("num_heads", 4),
        num_classes_per_head=config.get("num_classes_per_head", 1),
    )
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def backbone_vs_base(model, base_name: str) -> Dict[str, Any]:
    """How many backbone tensors are byte-identical to the stock base model."""
    import torch
    from transformers import AutoModel

    base = AutoModel.from_pretrained(base_name).state_dict()
    mine = model.bert.state_dict()
    shared = [k for k in mine if k in base and mine[k].shape == base[k].shape]
    same = sum(1 for k in shared if torch.equal(mine[k], base[k]))
    return {"base_model": base_name, "compared": len(shared), "identical_to_base": same,
            "trained": same < len(shared)}


def score(model, tokenizer, texts: Sequence[str], temps: Sequence[float], batch_size: int, device: str):
    import torch

    probs: Dict[str, List[float]] = {a: [] for a in ATTRIBUTES}
    model.to(device)
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), return_tensors="pt", truncation=True,
                        max_length=512, padding=True).to(device)
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        for k, a in enumerate(ATTRIBUTES):
            probs[a].extend(torch.sigmoid(logits[k].view(-1) / temps[k]).float().cpu().tolist())
    return probs


def product_loader_check(local: Path, reference) -> Dict[str, Any]:
    """Does this checkout's load_risk_classifier load the same weights?"""
    import torch

    from src.risk_model.model import load_risk_classifier

    try:
        served = load_risk_classifier(str(local), device="cpu")
    except Exception as exc:  # recorded, not raised: this is a finding about the branch
        return {"loads_same_weights": False, "error": f"{type(exc).__name__}: {exc}"}
    ref, got = reference.state_dict(), served.state_dict()
    differ = [k for k in ref if k not in got or got[k].shape != ref[k].shape or not torch.equal(got[k], ref[k])]
    return {"class": type(served).__name__, "loads_same_weights": not differ,
            "tensors_differing": len(differ), "tensors_total": len(ref), "first_differing": differ[:5]}


def compare_claim(measured: Dict[str, Any], claimed_path: Path) -> Dict[str, Any]:
    claimed = json.loads(claimed_path.read_text(encoding="utf-8"))
    cm = claimed.get("metrics", claimed)
    rows = []
    for scope in ("macro",) + ATTRIBUTES:
        mine = measured["macro"] if scope == "macro" else measured["per_attribute"][scope]
        theirs = cm.get("macro", {}) if scope == "macro" else cm.get("per_attribute", {}).get(scope, {})
        for m in ("f1", "pr_auc", "roc_auc"):
            if m in theirs and m in mine:
                rows.append({"scope": scope, "metric": m, "claimed": theirs[m], "measured": mine[m],
                             "match": abs(float(theirs[m]) - float(mine[m])) <= CLAIM_TOLERANCE})
    return {"claimed_file": str(claimed_path.relative_to(ROOT) if claimed_path.is_relative_to(ROOT) else claimed_path),
            "claimed_revision": claimed.get("revision"), "tolerance": CLAIM_TOLERANCE,
            "all_match": bool(rows) and all(r["match"] for r in rows), "rows": rows}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="full 40-character Hub commit sha")
    parser.add_argument("--claimed", help="a metrics JSON whose numbers should be reproduced")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--split-manifest", default="data/manifests/split_manifest.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default="artifacts/risk_eval")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer

    from score_risk_model import band_shares, load_records
    from src.product.thresholds import expected_calibration_error, fit_thresholds
    from src.risk_model.train import (binary_balanced_accuracy, binary_f1_score, binary_pr_auc,
                                      binary_roc_auc, extract_inferability_labels)

    t0 = time.perf_counter()
    local = download_revision(args.model_id, args.revision, args.cache_dir)
    config = json.loads((local / "risk_model_config.json").read_text(encoding="utf-8"))
    weights = read_state_dicts(local)
    files_agree = (compare_state_dicts(weights["model.pt"], weights["model.safetensors"])
                   if len(weights) == 2 else None)
    primary = "model.safetensors" if "model.safetensors" in weights else "model.pt"
    model = build_strict(local, weights[primary], config)
    base = backbone_vs_base(model, config.get("model_name", "answerdotai/ModernBERT-base"))

    temps = config.get("temperatures") or [1.0] * len(ATTRIBUTES)
    calib_path = local / "calibration_params.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else None
    calib_temps = [calib["temperatures"][a] for a in ATTRIBUTES] if calib and "temperatures" in calib else None
    temps_agree = calib_temps is None or all(abs(x - y) < 1e-4 for x, y in zip(temps, calib_temps))

    records = load_records(args.split, ROOT / args.split_manifest)
    labels = {a: [int(extract_inferability_labels(r)[k]) for r in records] for k, a in enumerate(ATTRIBUTES)}
    tok_dir = local if (local / "tokenizer.json").exists() else config.get("model_name")
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    probs = score(model, tokenizer, [r.get("text", "") for r in records], temps, args.batch_size, args.device)

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
            "bands_served_0.30_0.60": band_shares(probs[a], labels[a], SERVED_BANDS),
        }
    macro = {m: round(float(np.mean([per_attr[a][m] for a in ATTRIBUTES])), 4)
             for m in ("f1", "pr_auc", "roc_auc", "balanced_accuracy", "ece")}
    fits = fit_thresholds(labels, probs)
    fitted = {a: {"medium_edge": round(f.medium_edge, 4), "high_edge": round(f.high_edge, 4),
                  "high_precision": round(f.high_precision, 4), "low_recall": round(f.low_recall, 4),
                  "usable": f.usable, "degraded": f.degraded, "note": f.note} for a, f in fits.items()}
    measured = {"macro": macro, "per_attribute": per_attr}

    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "checks": {
            "strict_load": True,
            "weights_files": sorted(weights),
            "model_pt_equals_safetensors": None if files_agree is None else files_agree["identical"],
            "backbone_vs_base": base,
            "config_temperatures_equal_calibration_params": temps_agree,
            "product_loader_on_this_branch": product_loader_check(local, model),
        },
        "temperatures": dict(zip(ATTRIBUTES, temps)),
        "split": args.split,
        "comments": len(records),
        "profiles": len({r.get("profile_id") or r.get("author") for r in records}),
        "metrics": measured,
        "fitted_bands_in_sample": fitted,
        "claim": compare_claim(measured, ROOT / args.claimed if not Path(args.claimed).is_absolute() else Path(args.claimed))
        if args.claimed else None,
        "scores": {"ids": [r.get("id") for r in records], **{a: [round(p, 5) for p in probs[a]] for a in ATTRIBUTES}},
        "command": "python scripts/verify_risk_release.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "hardware": {"device": args.device, "platform": platform.platform(),
                     "torch": torch.__version__, "transformers": transformers.__version__},
        "seconds": round(time.perf_counter() - t0, 1),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"verify_{args.model_id.split('/')[-1].lower()}_{args.revision[:8]}_{args.split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    c = result["checks"]
    print(f"{args.model_id} @ {args.revision[:8]}: strict load ok; files {c['weights_files']}; "
          f"pt==safetensors {c['model_pt_equals_safetensors']}; backbone identical to base "
          f"{base['identical_to_base']}/{base['compared']}; product loader same weights: "
          f"{c['product_loader_on_this_branch'].get('loads_same_weights')}")
    print(f"{'attribute':<12}{'F1':>7}{'PR-AUC':>8}{'ROC-AUC':>9}{'ECE':>7}{'pos':>6}")
    for a in ATTRIBUTES:
        m = per_attr[a]
        print(f"{a:<12}{m['f1']:>7.3f}{m['pr_auc']:>8.3f}{m['roc_auc']:>9.3f}{m['ece']:>7.3f}{m['pos_count']:>6}")
    print(f"{'macro':<12}{macro['f1']:>7.3f}{macro['pr_auc']:>8.3f}{macro['roc_auc']:>9.3f}{macro['ece']:>7.3f}")
    if result["claim"]:
        print(f"claim {result['claim']['claimed_file']}: all match within {CLAIM_TOLERANCE}: {result['claim']['all_match']}")
    print(f"wrote {out.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()
