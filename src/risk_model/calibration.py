"""Post-hoc probability calibration and Expected Calibration Error (ECE) suite.

Provides head-specific Temperature Scaling, Platt Scaling fallback, ECE/MCE/Brier
metrics, risk band threshold validation (0.30 / 0.60), and reliability diagram generation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.product.risk_bands import ATTRIBUTES, HIGH_EDGE, MEDIUM_EDGE


@dataclass(frozen=True)
class BandValidationResult:
    """Evaluation of risk band alignment against empirical ground truth."""

    band_name: str
    threshold_range: Tuple[float, float]
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float
    is_well_calibrated: bool


@dataclass(frozen=True)
class CalibrationMetrics:
    """Comprehensive calibration metrics container."""

    ece: float
    mce: float
    brier_score: float
    n_bins: int
    bin_edges: List[float]
    bin_confidences: List[float]
    bin_accuracies: List[float]
    bin_counts: List[int]
    band_validations: Dict[str, BandValidationResult]
    is_calibrated: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        d = asdict(self)
        d["band_validations"] = {k: asdict(v) for k, v in self.band_validations.items()}
        return d


def compute_ece(
    probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    labels: Union[torch.Tensor, np.ndarray, Sequence[Union[int, float]]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationMetrics:
    """Compute Expected Calibration Error, MCE, Brier score, and band validations.

    Args:
        probs: Array-like sequence of predicted probabilities in [0.0, 1.0].
        labels: Array-like sequence of binary ground truth labels in {0, 1}.
        n_bins: Number of confidence bins (default: 10).
        strategy: 'uniform' (equal width) or 'quantile' (equal frequency).

    Returns:
        CalibrationMetrics instance.

    Raises:
        ValueError: If n_bins < 1, probs and labels differ in length, contain non-finite values,
            or probs fall outside [0.0, 1.0], or labels are not in {0, 1}.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    if isinstance(probs, torch.Tensor):
        p_arr = probs.detach().cpu().numpy().flatten()
    else:
        p_arr = np.asarray(probs, dtype=np.float64).flatten()

    if isinstance(labels, torch.Tensor):
        y_arr = labels.detach().cpu().numpy().flatten()
    else:
        y_arr = np.asarray(labels, dtype=np.float64).flatten()

    if len(p_arr) != len(y_arr):
        raise ValueError(f"probs and labels must have the same length, got {len(p_arr)} and {len(y_arr)}")

    if len(p_arr) == 0:
        raise ValueError("Cannot compute calibration metrics on empty arrays")

    if not np.all(np.isfinite(p_arr)):
        raise ValueError("probs contains non-finite values (NaN or Inf)")

    if not np.all(np.isfinite(y_arr)):
        raise ValueError("labels contains non-finite values (NaN or Inf)")

    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError(
            f"probabilities must lie in [0.0, 1.0], min={np.min(p_arr)}, max={np.max(p_arr)}"
        )

    unique_labels = np.unique(y_arr)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError(f"labels must be binary in {{0, 1}}, got {unique_labels}")

    n_samples = len(p_arr)

    # Binning
    if strategy == "quantile":
        try:
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            bin_edges = np.percentile(p_arr, quantiles * 100.0).tolist()
            # Ensure strictly increasing
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
            for b in range(1, len(bin_edges)):
                if bin_edges[b] <= bin_edges[b - 1]:
                    bin_edges[b] = bin_edges[b - 1] + 1e-6
        except Exception:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1).tolist()
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1).tolist()

    bin_confidences: List[float] = []
    bin_accuracies: List[float] = []
    bin_counts: List[int] = []

    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        low = bin_edges[m]
        high = bin_edges[m + 1]

        if m == 0:
            mask = (p_arr >= low) & (p_arr <= high)
        else:
            mask = (p_arr > low) & (p_arr <= high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            conf = float(np.mean(p_arr[mask]))
            acc = float(np.mean(y_arr[mask]))
            gap = abs(acc - conf)

            bin_confidences.append(conf)
            bin_accuracies.append(acc)

            ece += (count / n_samples) * gap
            if gap > mce:
                mce = gap
        else:
            bin_confidences.append(float((low + high) / 2.0))
            bin_accuracies.append(0.0)

    # Brier score
    brier_score = float(np.mean((p_arr - y_arr) ** 2))

    # Risk band validation for 0.30 and 0.60 thresholds
    band_defs = [
        ("LOW", (0.0, MEDIUM_EDGE)),
        ("MEDIUM", (MEDIUM_EDGE, HIGH_EDGE)),
        ("HIGH", (HIGH_EDGE, 1.0)),
    ]

    band_validations: Dict[str, BandValidationResult] = {}
    for band_name, (b_low, b_high) in band_defs:
        if band_name == "HIGH":
            b_mask = (p_arr >= b_low) & (p_arr <= b_high)
        else:
            b_mask = (p_arr >= b_low) & (p_arr < b_high)

        b_count = int(np.sum(b_mask))
        if b_count > 0:
            b_conf = float(np.mean(p_arr[b_mask]))
            b_acc = float(np.mean(y_arr[b_mask]))
            b_gap = abs(b_acc - b_conf)
            is_well = b_gap < 0.08
        else:
            b_conf = float((b_low + b_high) / 2.0)
            b_acc = 0.0
            b_gap = 0.0
            is_well = True

        band_validations[band_name] = BandValidationResult(
            band_name=band_name,
            threshold_range=(b_low, b_high),
            sample_count=b_count,
            mean_confidence=b_conf,
            empirical_accuracy=b_acc,
            calibration_gap=b_gap,
            is_well_calibrated=is_well,
        )

    is_calibrated = bool(ece < 0.05)

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        brier_score=brier_score,
        n_bins=n_bins,
        bin_edges=[float(e) for e in bin_edges],
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        band_validations=band_validations,
        is_calibrated=is_calibrated,
    )


class TemperatureScaler:
    """Head-specific post-hoc temperature and Platt calibration for ModernBERT."""

    def __init__(
        self,
        head_names: Sequence[str] = ATTRIBUTES,
        optimizer_type: str = "lbfgs",
        learning_rate: float = 0.1,
        max_iter: int = 50,
        enable_platt_fallback: bool = True,
        platt_fallback_ece_threshold: float = 0.05,
    ) -> None:
        self.head_names = list(head_names)
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.enable_platt_fallback = enable_platt_fallback
        self.platt_fallback_ece_threshold = platt_fallback_ece_threshold

        self.temperatures: Dict[str, float] = {h: 1.0 for h in self.head_names}
        self.platt_params: Dict[str, Tuple[float, float]] = {h: (1.0, 0.0) for h in self.head_names}
        self.methods: Dict[str, str] = {h: "temperature" for h in self.head_names}
        self.is_fitted = False

    def _normalize_tensors(
        self,
        inputs: Union[List[torch.Tensor], torch.Tensor, np.ndarray],
    ) -> List[torch.Tensor]:
        """Convert input data into list of 1D float32 tensors, one per head."""
        if isinstance(inputs, list):
            out = []
            for t in inputs:
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                out.append(t.detach().cpu().float().view(-1))
            return out
        elif isinstance(inputs, np.ndarray):
            t = torch.from_numpy(inputs).float()
            if t.ndim == 2 and t.shape[1] == len(self.head_names):
                return [t[:, i] for i in range(len(self.head_names))]
            return [t.view(-1)]
        elif isinstance(inputs, torch.Tensor):
            t = inputs.detach().cpu().float()
            if t.ndim == 2 and t.shape[1] == len(self.head_names):
                return [t[:, i] for i in range(len(self.head_names))]
            return [t.view(-1)]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

    def fit(
        self,
        val_logits: Union[List[torch.Tensor], torch.Tensor, np.ndarray],
        val_labels: Union[List[torch.Tensor], torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """Fit head-specific temperatures Tk (and Platt parameters if needed).

        Args:
            val_logits: List of 4 tensors or 2D tensor [N, 4].
            val_labels: List of 4 tensors or 2D tensor [N, 4].

        Returns:
            Dictionary mapping head name to learned temperature Tk.
        """
        norm_logits = self._normalize_tensors(val_logits)
        norm_labels = self._normalize_tensors(val_labels)

        if len(norm_logits) != len(self.head_names) or len(norm_labels) != len(self.head_names):
            raise ValueError(
                f"Expected {len(self.head_names)} heads, got {len(norm_logits)} logits and {len(norm_labels)} labels"
            )

        for k, head in enumerate(self.head_names):
            z = norm_logits[k]
            y = norm_labels[k]

            if len(z) != len(y):
                raise ValueError(f"Length mismatch for head {head}: {len(z)} logits, {len(y)} labels")

            # 1. Optimize temperature parameter tau = log(T)
            tau = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            optimizer = torch.optim.LBFGS([tau], lr=self.learning_rate, max_iter=self.max_iter)

            def eval_temp():
                optimizer.zero_grad()
                temp = torch.exp(tau)
                scaled_logits = z / temp
                loss = F.binary_cross_entropy_with_logits(scaled_logits, y)
                loss.backward()
                return loss

            try:
                optimizer.step(eval_temp)
            except Exception:
                # Fallback to Adam if L-BFGS encounters issues
                adam_opt = torch.optim.Adam([tau], lr=self.learning_rate)
                for _ in range(100):
                    adam_opt.zero_grad()
                    temp = torch.exp(tau)
                    loss = F.binary_cross_entropy_with_logits(z / temp, y)
                    loss.backward()
                    adam_opt.step()

            learned_temp = float(torch.exp(tau).item())
            # Ensure strictly positive
            learned_temp = max(learned_temp, 1e-4)
            self.temperatures[head] = learned_temp
            self.methods[head] = "temperature"

            # Check post-temperature calibration
            cal_p = torch.sigmoid(z / learned_temp).numpy()
            metrics = compute_ece(cal_p, y.numpy())

            # 2. Platt scaling fallback if ECE remains high
            if self.enable_platt_fallback and metrics.ece >= self.platt_fallback_ece_threshold:
                alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
                beta = nn.Parameter(torch.zeros(1, dtype=torch.float32))
                platt_opt = torch.optim.LBFGS([alpha, beta], lr=self.learning_rate, max_iter=self.max_iter)

                def eval_platt():
                    platt_opt.zero_grad()
                    a = torch.exp(alpha)
                    scaled = a * z + beta
                    loss = F.binary_cross_entropy_with_logits(scaled, y)
                    reg = 0.01 * ((a - 1.0) ** 2) + 0.01 * (beta ** 2)
                    total_loss = loss + reg
                    total_loss.backward()
                    return total_loss

                try:
                    platt_opt.step(eval_platt)
                except Exception:
                    adam_platt = torch.optim.Adam([alpha, beta], lr=self.learning_rate)
                    for _ in range(100):
                        adam_platt.zero_grad()
                        a = torch.exp(alpha)
                        loss = F.binary_cross_entropy_with_logits(a * z + beta, y)
                        reg = 0.01 * ((a - 1.0) ** 2) + 0.01 * (beta ** 2)
                        (loss + reg).backward()
                        adam_platt.step()

                learned_a = float(torch.exp(alpha).item())
                learned_b = float(beta.item())
                platt_p = torch.sigmoid(learned_a * z + learned_b).numpy()
                platt_metrics = compute_ece(platt_p, y.numpy())

                if platt_metrics.ece < metrics.ece:
                    self.platt_params[head] = (learned_a, learned_b)
                    self.methods[head] = "platt"

        self.is_fitted = True
        return dict(self.temperatures)

    def calibrate_logits(
        self,
        logits: Union[List[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        """Scale logits by learned head parameters."""
        norm_logits = self._normalize_tensors(logits)
        calibrated: List[torch.Tensor] = []

        for k, head in enumerate(self.head_names):
            z = norm_logits[k]
            method = self.methods.get(head, "temperature")
            if method == "platt":
                a, b = self.platt_params[head]
                calibrated.append(a * z + b)
            else:
                t = self.temperatures.get(head, 1.0)
                calibrated.append(z / t)

        return calibrated

    def predict_proba(
        self,
        logits: Union[List[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        """Emit calibrated probabilities in [0, 1] for each head."""
        cal_logits = self.calibrate_logits(logits)
        return [torch.sigmoid(z) for z in cal_logits]

    def predict_calibrated(
        self,
        logits: Union[List[torch.Tensor], torch.Tensor],
    ) -> Dict[str, float]:
        """Predict calibrated probabilities for a single input text.

        Returns:
            Dictionary mapping attribute name to calibrated float in [0, 1],
            matching input contract of src.product.risk_bands.summarize.
        """
        probs_list = self.predict_proba(logits)
        out: Dict[str, float] = {}
        for k, head in enumerate(self.head_names):
            p = probs_list[k]
            val = float(p[0].item() if p.numel() > 0 else 0.0)
            out[head] = round(max(0.0, min(1.0, val)), 4)
        return out

    def evaluate(
        self,
        val_logits: Union[List[torch.Tensor], torch.Tensor],
        val_labels: Union[List[torch.Tensor], torch.Tensor],
    ) -> Dict[str, CalibrationMetrics]:
        """Evaluate post-calibration metrics across all heads."""
        cal_probs = self.predict_proba(val_logits)
        norm_labels = self._normalize_tensors(val_labels)
        metrics: Dict[str, CalibrationMetrics] = {}

        for k, head in enumerate(self.head_names):
            p = cal_probs[k].numpy()
            y = norm_labels[k].numpy()
            metrics[head] = compute_ece(p, y)

        return metrics

    def save(self, path: Union[str, Path]) -> None:
        """Serialize parameters (temperatures, Platt params, methods) to JSON."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "head_names": self.head_names,
            "temperatures": self.temperatures,
            "platt_params": self.platt_params,
            "methods": self.methods,
            "is_fitted": self.is_fitted,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> TemperatureScaler:
        """Instantiate fitted TemperatureScaler from saved JSON parameters."""
        load_path = Path(path)
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scaler = cls(head_names=data.get("head_names", ATTRIBUTES))
        scaler.temperatures = data.get("temperatures", {})
        scaler.platt_params = {k: tuple(v) for k, v in data.get("platt_params", {}).items()}
        scaler.methods = data.get("methods", {})
        scaler.is_fitted = data.get("is_fitted", True)
        return scaler


def generate_reliability_diagram(
    calibrated_probs: Mapping[str, Sequence[float]],
    labels: Mapping[str, Sequence[Union[int, float]]],
    temperatures: Optional[Mapping[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    json_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Generate structured calibration metrics and multi-head reliability diagrams.

    Args:
        calibrated_probs: Mapping of attribute name to predicted probabilities.
        labels: Mapping of attribute name to binary ground truth labels.
        temperatures: Optional mapping of attribute name to learned Tk.
        save_path: Optional path to save PNG figure.
        json_path: Optional path to save structured metrics JSON report.
        n_bins: Number of confidence bins.

    Returns:
        Structured dictionary containing metrics, bin data, and threshold validation.
    """
    report: Dict[str, Any] = {
        "attributes": {},
        "macro_ece": 0.0,
        "all_heads_calibrated": True,
    }

    ece_list: List[float] = []

    for head in calibrated_probs:
        if head not in labels:
            continue
        p = calibrated_probs[head]
        y = labels[head]
        metrics = compute_ece(p, y, n_bins=n_bins)
        head_dict = metrics.to_dict()
        if temperatures and head in temperatures:
            head_dict["temperature"] = temperatures[head]
        report["attributes"][head] = head_dict
        ece_list.append(metrics.ece)
        if not metrics.is_calibrated:
            report["all_heads_calibrated"] = False

    if ece_list:
        report["macro_ece"] = float(np.mean(ece_list))

    # Save JSON report if requested
    if json_path is not None:
        jp = Path(json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # Plot reliability diagram if requested and matplotlib available
    if save_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            heads = list(report["attributes"].keys())
            n_heads = len(heads)
            if n_heads > 0:
                fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4), squeeze=False)
                for idx, head in enumerate(heads):
                    ax = axes[0, idx]
                    h_data = report["attributes"][head]
                    confs = h_data["bin_confidences"]
                    accs = h_data["bin_accuracies"]
                    ece_val = h_data["ece"]

                    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
                    ax.plot(confs, accs, marker="o", color="#1f77b4", label=f"ECE = {ece_val:.3f}")
                    ax.axvline(x=MEDIUM_EDGE, color="orange", linestyle=":", label="0.30 Medium")
                    ax.axvline(x=HIGH_EDGE, color="red", linestyle=":", label="0.60 High")
                    ax.set_title(head.capitalize())
                    ax.set_xlabel("Confidence")
                    ax.set_ylabel("Empirical Accuracy")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                sp = Path(save_path)
                sp.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(sp, dpi=150)
                plt.close(fig)
        except Exception:
            pass

    return report
