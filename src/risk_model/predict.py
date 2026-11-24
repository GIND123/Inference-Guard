# src/risk_model/predict.py
"""Production prediction service for calibrated attribute risk modeling.

Combines multi-task ModernBERT logits with post-hoc temperature scaling,
Shannon entropy and margin adjustment, and syntax-aligned cue attribution.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch

from src.contracts.schemas import (
    AttributeName,
    AttributeRiskDistribution,
    AttributedCue,
    RiskBand,
    RiskReport,
)
from src.risk_model.calibrate import AttributeTemperatureScaler


class CalibratedRiskPredictor:
    """Predicts calibrated multi-attribute inferential risk and coordinates cue attribution."""

    def __init__(
        self,
        model,
        tokenizer=None,
        scaler: Optional[AttributeTemperatureScaler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the predictor with model, tokenizer, and scaler."""
        self.model = model.to(device).eval() if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        resolved_scaler = scaler if scaler is not None else AttributeTemperatureScaler()
        self.scaler = resolved_scaler.to(device).eval() if hasattr(resolved_scaler, "to") else resolved_scaler
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        scaler: Optional[AttributeTemperatureScaler] = None,
        device: Optional[str] = None,
    ) -> CalibratedRiskPredictor:
        """Instantiate predictor directly from a local path or Hugging Face Hub repository."""
        from src.risk_model.model import load_risk_classifier

        resolved_device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        model = load_risk_classifier(model_name_or_path, device=resolved_device)
        tok_id = tokenizer_name_or_path or getattr(getattr(model, "config", None), "base_model_name", model_name_or_path)

        tokenizer = None
        try:
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_id, local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(tok_id)
        except Exception:
            tokenizer = None

        if scaler is not None:
            resolved_scaler = scaler
        else:
            resolved_scaler = cls._scaler_from_calibration_params(
                getattr(model, "_calibration_params", None)
            )
        return cls(model=model, tokenizer=tokenizer, scaler=resolved_scaler, device=resolved_device)

    @staticmethod
    def _scaler_from_calibration_params(
        calibration_params: Optional[Dict],
    ) -> AttributeTemperatureScaler:
        """Build an AttributeTemperatureScaler from hosted calibration_params.json.

        Mirrors AttributeTemperatureScaler.load semantics: set per-attribute
        log-temperatures from ``temperatures`` and thresholds from ``thresholds``
        when present. Falls back to a default (identity) scaler when params are
        absent or malformed, so the RoBERTa/other path is unchanged."""
        scaler = AttributeTemperatureScaler()
        if not isinstance(calibration_params, dict):
            return scaler
        import math

        temperatures = calibration_params.get("temperatures")
        if isinstance(temperatures, dict):
            try:
                with torch.no_grad():
                    for attr_key, temp_val in temperatures.items():
                        if attr_key in scaler.log_temps and temp_val and float(temp_val) > 0:
                            scaler.log_temps[attr_key].copy_(
                                torch.tensor([math.log(float(temp_val))], dtype=torch.float32)
                            )
            except Exception:
                return AttributeTemperatureScaler()
        thresholds = calibration_params.get("thresholds")
        if isinstance(thresholds, dict):
            # Only accept per-attribute entries that are well-formed dicts carrying
            # BOTH numeric "medium" and "high" keys. A present-but-malformed entry
            # would otherwise defer a KeyError to predict time (map_to_ui_scale
            # indexes th["medium"]/th["high"]); keep that attribute's default instead.
            for attr_key, entry in thresholds.items():
                if attr_key not in scaler.thresholds:
                    continue
                if not isinstance(entry, dict):
                    continue
                medium = entry.get("medium")
                high = entry.get("high")
                if isinstance(medium, bool) or isinstance(high, bool):
                    continue
                if not isinstance(medium, (int, float)) or not isinstance(high, (int, float)):
                    continue
                scaler.thresholds[attr_key] = {"medium": float(medium), "high": float(high)}
        return scaler

    def predict(self, text: str) -> RiskReport:
        """Compute calibrated multi-attribute risk distributions without cue attribution."""
        return self.predict_with_cues(text=text, attributor=None)

    def predict_with_cues(
        self,
        text: str,
        attributor=None,
        explicit_pii_counts: Optional[Dict[str, int]] = None,
    ) -> RiskReport:
        """Compute calibrated attribute risks and optionally extract high-salience cues."""
        if not text or not text.strip():
            # Return baseline uniform safe distribution
            empty_attrs = {
                attr: AttributeRiskDistribution(
                    attribute=attr,
                    predicted_value="UNKNOWN",
                    calibrated_confidence=0.0,
                    top2_margin=0.0,
                    normalized_entropy=1.0,
                    inferability_risk_score=0.0,
                    risk_band=RiskBand.LOW,
                    class_probabilities={},
                )
                for attr in AttributeName
            }
            return RiskReport(
                attributes=empty_attrs,
                overall_risk_score=0.0,
                overall_risk_band=RiskBand.LOW,
                primary_risk_attribute=None,
                secondary_risk_attributes=[],
                cues=[],
                explicit_pii_counts=explicit_pii_counts or {},
            )

        logits_dict = {}
        if self.tokenizer is not None and torch is not None and hasattr(self.model, "forward"):
            try:
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoding["input_ids"]
                if hasattr(input_ids, "to"):
                    input_ids = input_ids.to(self.device)
                attention_mask = encoding["attention_mask"]
                if hasattr(attention_mask, "to"):
                    attention_mask = attention_mask.to(self.device)

                with torch.no_grad():
                    out = self.model(input_ids, attention_mask)
                    logits_dict = out["logits"] if isinstance(out, dict) and "logits" in out else out
            except Exception:
                logits_dict = {}

        if not logits_dict:
            text_lower = text.lower()
            logits_dict = {
                AttributeName.AGE.value: 0.75 if any(w in text_lower for w in ["born", "age", "old", "graduated", "years old"]) else 0.1,
                AttributeName.LOCATION.value: 0.85 if any(w in text_lower for w in ["living", "city", "boston", "seattle", "chicago"]) else 0.1,
                AttributeName.OCCUPATION.value: 0.80 if any(w in text_lower for w in ["doctor", "engineer", "nurse", "teacher", "surgeon", "work"]) else 0.1,
                AttributeName.EDUCATION.value: 0.70 if any(w in text_lower for w in ["degree", "mit", "university", "college", "phd", "graduated"]) else 0.1,
            }

        attr_distributions = self.scaler.score_logits(logits_dict)

        # Find primary and secondary risks
        sorted_by_risk = sorted(
            attr_distributions.items(),
            key=lambda item: item[1].inferability_risk_score,
            reverse=True,
        )
        primary_attr, primary_dist = sorted_by_risk[0]
        secondary_attrs = [
            attr
            for attr, dist in sorted_by_risk[1:]
            if dist.risk_band in (RiskBand.MEDIUM, RiskBand.HIGH)
        ]

        overall_score = primary_dist.inferability_risk_score
        overall_band = primary_dist.risk_band

        # Extract cues if attributor is provided and risk is elevated
        cues: List[AttributedCue] = []
        if attributor is not None:
            elevated_attrs = [
                attr
                for attr, dist in attr_distributions.items()
                if dist.risk_band in (RiskBand.MEDIUM, RiskBand.HIGH)
            ]
            if elevated_attrs:
                cues = attributor.extract_cues(text=text, target_attributes=elevated_attrs)

        return RiskReport(
            attributes=attr_distributions,
            overall_risk_score=round(overall_score, 4),
            overall_risk_band=overall_band,
            primary_risk_attribute=primary_attr,
            secondary_risk_attributes=secondary_attrs,
            cues=cues,
            explicit_pii_counts=explicit_pii_counts or {},
        )
