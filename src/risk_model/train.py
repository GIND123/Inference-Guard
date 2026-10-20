"""Training pipeline for 4-head ModernBERT risk classifier and MultiTask taxonomy architecture.

Handles profile-disjoint data splitting, dynamic class weighting,
multi-task Focal/BCE loss, AdamW optimization, linear warmup, and metric evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.contracts.schemas import AttributeName
from src.risk_model.model import ATTRIBUTES, ModernBertRiskClassifier, MultiTaskModernBERTRiskModel
from src.risk_model.taxonomy import (
    ATTRIBUTE_TAXONOMY,
    LABEL_TO_ID,
    map_raw_to_taxonomy_label,
    map_to_taxonomy_id,
)

logger = logging.getLogger(__name__)

SYNTHPAI_ATTR_MAP: Dict[str, str] = {
    "age": "age",
    "location": "city_country",
    "occupation": "occupation",
    "education": "education",
}


def extract_inferability_labels(row: Dict[str, Any]) -> List[float]:
    """Extract binary inferability ground truth from human review estimates.

    A non-empty string in reviews.human[attribute].estimate indicates inferability (1.0).
    An empty or missing string indicates non-inferability (0.0).
    """
    human_reviews = (row.get("reviews") or {}).get("human") or {}
    labels: List[float] = []
    for attr in ATTRIBUTES:
        synth_key = SYNTHPAI_ATTR_MAP[attr]
        attr_data = human_reviews.get(synth_key) or {}
        est = str(attr_data.get("estimate", "") or "").strip()
        labels.append(1.0 if est != "" else 0.0)
    return labels


class SynthPAIRiskDataset(Dataset):
    """Profile-isolated dataset for 4-head risk inferability."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ) -> None:
        self.texts: List[str] = [str(r.get("text", "") or "") for r in rows]
        self.labels: List[List[float]] = [extract_inferability_labels(r) for r in rows]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            words = text.split()[: self.max_length]
            token_ids = [(abs(hash(w)) % 999) + 1 for w in words]
            if len(token_ids) < self.max_length:
                pad_len = self.max_length - len(token_ids)
                token_ids.extend([0] * pad_len)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = (input_ids != 0).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class SynthPAIMultiTaskDataset(Dataset):
    """Categorical multi-task dataset for ModernBERT demographic classification."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ) -> None:
        self.texts: List[str] = [str(r.get("text", "") or "") for r in rows]
        self.targets: List[Dict[str, int]] = []
        for r in rows:
            p = r.get("profile", {}) or r.get("attributes", {})
            self.targets.append({
                AttributeName.AGE.value: map_to_taxonomy_id(AttributeName.AGE, p.get("age")),
                AttributeName.EDUCATION.value: map_to_taxonomy_id(AttributeName.EDUCATION, p.get("education")),
                AttributeName.OCCUPATION.value: map_to_taxonomy_id(AttributeName.OCCUPATION, p.get("occupation")),
                AttributeName.LOCATION.value: map_to_taxonomy_id(
                    AttributeName.LOCATION, p.get("city_country") or p.get("location")
                ),
            })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            words = text.split()[: self.max_length]
            token_ids = [(abs(hash(w)) % 999) + 1 for w in words]
            if len(token_ids) < self.max_length:
                pad_len = self.max_length - len(token_ids)
                token_ids.extend([0] * pad_len)
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = (input_ids != 0).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": {k: torch.tensor(v, dtype=torch.long) for k, v in self.targets[idx].items()},
        }


def compute_pos_weights(labels_matrix: np.ndarray) -> List[float]:
    """Compute pos_weight for each head: (N - N_pos) / max(N_pos, 1)."""
    n_samples, n_heads = labels_matrix.shape
    weights: List[float] = []
    for k in range(n_heads):
        n_pos = int(np.sum(labels_matrix[:, k]))
        n_neg = n_samples - n_pos
        w = float(n_neg / max(n_pos, 1))
        weights.append(w)
    return weights


def binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary F1 score."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return float((2.0 * tp) / denominator)


def binary_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy score: (sensitivity + specificity) / 2."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    sensitivity = (np.sum(y_pred[pos_mask] == 1) / n_pos) if n_pos > 0 else 0.5
    specificity = (np.sum(y_pred[neg_mask] == 0) / n_neg) if n_neg > 0 else 0.5
    return float(0.5 * (sensitivity + specificity))


def binary_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall AUC (Average Precision)."""
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_true_sorted == 1)
    fp_cumsum = np.cumsum(y_true_sorted == 0)
    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([precisions[0] if len(precisions) > 0 else 0.0], precisions))
    ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    return max(0.0, min(1.0, ap))


def binary_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC via Wilcoxon-Mann-Whitney rank statistic with tied-rank averaging."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    try:
        from scipy.stats import rankdata
        ranks = rankdata(y_prob, method="average")
    except ImportError:
        _, inv, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        starts = np.pad(cum[:-1], (1, 0), constant_values=0)
        unique_avg_ranks = (starts + 1.0 + cum) / 2.0
        ranks = unique_avg_ranks[inv]

    rank_sum_pos = float(np.sum(ranks[pos_mask]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = float(u_stat / (n_pos * n_neg))
    return max(0.0, min(1.0, auc))


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    pos_weights: List[float],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate classifier on validation set across loss, F1, PR-AUC, and Balanced Accuracy."""
    model.eval()
    criterion = [
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], device=device))
        for w in pos_weights
    ]
    total_loss = 0.0
    all_preds: List[List[float]] = [[] for _ in range(len(ATTRIBUTES))]
    all_probs: List[List[float]] = [[] for _ in range(len(ATTRIBUTES))]
    all_targets: List[List[float]] = [[] for _ in range(len(ATTRIBUTES))]

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum(
                criterion[k](logits[k].view(-1), labels[:, k])
                for k in range(len(ATTRIBUTES))
            )
            total_loss += float(loss.item())

            for k in range(len(ATTRIBUTES)):
                head_logits = logits[k].view(-1)
                probs = torch.sigmoid(head_logits).cpu().numpy()
                preds = (probs >= 0.5).astype(float)
                targets = labels[:, k].cpu().numpy()

                all_probs[k].extend(probs.tolist())
                all_preds[k].extend(preds.tolist())
                all_targets[k].extend(targets.tolist())

    n_batches = max(len(data_loader), 1)
    val_loss = total_loss / n_batches

    metrics_per_attr: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []
    pr_auc_list: List[float] = []
    roc_auc_list: List[float] = []
    balanced_acc_list: List[float] = []

    for k, attr in enumerate(ATTRIBUTES):
        y_true = np.array(all_targets[k])
        y_pred = np.array(all_preds[k])
        y_prob = np.array(all_probs[k])

        f1 = binary_f1_score(y_true, y_pred)
        pr_auc = binary_pr_auc(y_true, y_prob)
        roc_auc = binary_roc_auc(y_true, y_prob)
        bal_acc = binary_balanced_accuracy(y_true, y_pred)

        metrics_per_attr[attr] = {
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "balanced_accuracy": bal_acc,
            "pos_count": int(np.sum(y_true)),
        }

        f1_list.append(f1)
        pr_auc_list.append(pr_auc)
        roc_auc_list.append(roc_auc)
        balanced_acc_list.append(bal_acc)

    return {
        "val_loss": val_loss,
        "per_attribute": metrics_per_attr,
        "macro_f1": float(np.mean(f1_list)),
        "macro_pr_auc": float(np.mean(pr_auc_list)),
        "macro_roc_auc": float(np.mean(roc_auc_list)),
        "macro_balanced_accuracy": float(np.mean(balanced_acc_list)),
    }


def evaluate_multitask(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate multi-task ModernBERT classifier across accuracy and multi-class macro F1."""
    model.eval()
    total_loss = 0.0
    correct_by_attr = {attr.value: 0 for attr in AttributeName}
    total_by_attr = {attr.value: 0 for attr in AttributeName}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            logits_dict = out["logits"]
            total_loss += out["loss"].item()

            for attr in AttributeName:
                key = attr.value
                preds = torch.argmax(logits_dict[key], dim=-1)
                correct_by_attr[key] += int((preds == targets[key]).sum().item())
                total_by_attr[key] += len(preds)

    n_batches = max(len(data_loader), 1)
    acc_by_attr = {k: correct_by_attr[k] / max(total_by_attr[k], 1) for k in correct_by_attr}
    mean_acc = float(np.mean(list(acc_by_attr.values())))

    return {
        "val_loss": total_loss / n_batches,
        "per_attribute_accuracy": acc_by_attr,
        "macro_accuracy": mean_acc,
    }


def load_train_val_rows(
    data_path: Optional[str] = "data/raw/synthpai.jsonl",
    splits_path: str = "artifacts/profile_splits.json",
    train_data: Optional[str] = None,
    val_data: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load train and validation records from pre-split files or raw SynthPAI dataset."""
    def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    train_file = Path(train_data) if train_data else None
    val_file = Path(val_data) if val_data else None

    if train_file and train_file.exists():
        train_rows = _load_jsonl(train_file)
        val_rows = _load_jsonl(val_file) if (val_file and val_file.exists()) else train_rows[:max(1, int(len(train_rows) * 0.2))]
        return train_rows, val_rows

    if not train_data and Path("data/splits/train.jsonl").exists():
        train_rows = _load_jsonl(Path("data/splits/train.jsonl"))
        val_rows = _load_jsonl(Path("data/splits/val.jsonl")) if Path("data/splits/val.jsonl").exists() else train_rows[:max(1, int(len(train_rows) * 0.2))]
        return train_rows, val_rows

    from src.data.splits import load_or_fetch_synthpai
    rows = load_or_fetch_synthpai(Path(data_path) if data_path and Path(data_path).exists() else None)

    splits_file = Path(splits_path)
    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        train_pids = set(splits_data.get("train", []))
        val_pids = set(splits_data.get("val", []))
    else:
        all_pids = list({str(r.get("author") or r.get("profile_id")) for r in rows})
        np.random.seed(42)
        np.random.shuffle(all_pids)
        n_train = int(len(all_pids) * 0.8)
        train_pids = set(all_pids[:n_train])
        val_pids = set(all_pids[n_train:])

    train_rows = [r for r in rows if str(r.get("author") or r.get("profile_id")) in train_pids]
    val_rows = [r for r in rows if str(r.get("author") or r.get("profile_id")) in val_pids]
    return train_rows, val_rows


def train_multitask_modernbert(
    data_path: Optional[str] = "data/raw/synthpai.jsonl",
    splits_path: str = "artifacts/profile_splits.json",
    train_data: Optional[str] = None,
    val_data: Optional[str] = None,
    output_dir: str = "artifacts/risk_model",
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    backbone: Optional[nn.Module] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train MultiTaskModernBERTRiskModel on categorical taxonomy heads with Focal Loss."""
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = load_train_val_rows(data_path, splits_path, train_data, val_data)

    tokenizer = None
    if backbone is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = SynthPAIMultiTaskDataset(train_rows, tokenizer=tokenizer)
    val_dataset = SynthPAIMultiTaskDataset(val_rows, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskModernBERTRiskModel(
        model_name=model_name,
        backbone=backbone,
        hidden_size=getattr(backbone, "hidden_size", None),
    )
    model.to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))

    best_acc = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"MultiTask Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(selected_device)
            attention_mask = batch["attention_mask"].to(selected_device)
            targets = {k: v.to(selected_device) for k, v in batch["targets"].items()}

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        val_metrics = evaluate_multitask(model, val_loader, selected_device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = epoch_loss / max(len(train_loader), 1)
        history.append(val_metrics)

        if val_metrics["macro_accuracy"] > best_acc:
            best_acc = val_metrics["macro_accuracy"]
            best_dir = output_path if output_path.name == "best_model" else (output_path / "best_model")
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_dir / "multitask_heads.pt")
            if tokenizer is not None:
                tokenizer.save_pretrained(best_dir)

    summary = {"best_macro_accuracy": best_acc, "history": history}
    with open(output_path / "multitask_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def train_risk_model(
    data_path: Optional[str] = "data/raw/synthpai.jsonl",
    splits_path: str = "artifacts/profile_splits.json",
    train_data: Optional[str] = None,
    val_data: Optional[str] = None,
    output_dir: str = "artifacts/risk_model",
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    backbone: Optional[nn.Module] = None,
    device: Optional[str] = None,
    model_type: str = "multitask_modernbert",
) -> Dict[str, Any]:
    """Train risk classifier supporting either multi-task taxonomy or legacy BCE classifier."""
    if model_type == "multitask_modernbert":
        return train_multitask_modernbert(
            data_path=data_path,
            splits_path=splits_path,
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            backbone=backbone,
            device=device,
        )

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = load_train_val_rows(data_path, splits_path, train_data, val_data)

    train_labels = np.array([extract_inferability_labels(r) for r in train_rows])
    pos_weights = compute_pos_weights(train_labels)

    tokenizer = None
    if backbone is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = SynthPAIRiskDataset(train_rows, tokenizer=tokenizer)
    val_dataset = SynthPAIRiskDataset(val_rows, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ModernBertRiskClassifier(
        model_name=model_name,
        backbone=backbone,
        hidden_size=getattr(backbone, "hidden_size", None),
    )
    model.to(selected_device)

    criterion = [
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], device=selected_device))
        for w in pos_weights
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, 1)
    )

    best_macro_f1 = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(selected_device)
            attention_mask = batch["attention_mask"].to(selected_device)
            labels = batch["labels"].to(selected_device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum(
                criterion[k](logits[k].view(-1), labels[:, k])
                for k in range(len(ATTRIBUTES))
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, pos_weights, selected_device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = epoch_loss / max(len(train_loader), 1)
        history.append(val_metrics)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            model.save_pretrained(output_path / "best_model")

    summary = {
        "best_macro_f1": best_macro_f1,
        "pos_weights": dict(zip(ATTRIBUTES, pos_weights)),
        "history": history,
    }
    with open(output_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task risk classifier")
    parser.add_argument("--data_path", type=str, default="data/raw/synthpai.jsonl")
    parser.add_argument("--splits_path", type=str, default="artifacts/profile_splits.json")
    parser.add_argument("--train_data", "--train_path", type=str, default=None, help="Path to pre-split train jsonl")
    parser.add_argument("--val_data", "--val_path", type=str, default=None, help="Path to pre-split val jsonl")
    parser.add_argument("--output_dir", type=str, default="artifacts/risk_model")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--model_type", type=str, choices=["multitask_modernbert", "binary_classifier"], default="multitask_modernbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--mock", action="store_true", help="Use lightweight mock backbone for offline testing")
    args = parser.parse_args()

    backbone = None
    if args.mock:
        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 64
                self.embeddings = nn.Embedding(1001, 64)
                self.lin = nn.Linear(64, 64)

            def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
                if inputs_embeds is None:
                    inputs_embeds = self.embeddings(input_ids)
                return self.lin(inputs_embeds)

        backbone = TinyBackbone()

    train_risk_model(
        data_path=args.data_path,
        splits_path=args.splits_path,
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=backbone,
    )
