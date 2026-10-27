# src/api/app.py
"""Production FastAPI service for InferenceGuard end-to-end privacy layer.

Exposes REST endpoints for calibrated risk prediction, syntax-aligned cue highlighting,
QLoRA rewriting, session-stable pseudonym management, and zero-PII telemetry logging.
Consolidates all web analysis, rewriting, and telemetry endpoints into a single backend.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.ui_helpers import (
    PrivacySafeTelemetryLogger,
    SafeTelemetryEvent,
    render_highlighted_html_from_offsets,
)
from src.contracts.schemas import (
    AttributedCue,
    AttributeName,
    AttributeRiskDistribution,
    ConversationState,
    NormalizedInput,
    ProtectionResponse,
    RewriteMode,
    RewriteRequest,
    RiskBand,
    RiskReport,
)
from src.conversation.entity_map import DeterministicPseudonymManager
from src.conversation.leakage import BayesianConversationLeakageTracker
from src.conversation.pseudonyms import PersistentPseudonymManager
from src.conversation.state import ConversationTracker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.utility import UtilityEvaluator
from src.evaluation.utility_metrics import BidirectionalUtilityEvaluator
from src.pipeline.orchestrator import InferenceGuardPipeline
from src.product.report import (
    DEFAULT_UTILITY_FLOOR,
    PiiSpan,
    compare,
    compose,
)
from src.product.risk_bands import ATTRIBUTES, Band, Cue
from src.product.thresholds import (
    PROVISIONAL_CUE_IMPORTANCE_FLOOR,
    PROVISIONAL_MAX_RISK_THRESHOLD,
    PROVISIONAL_MIN_COSINE_THRESHOLD,
)
from src.rewriter.generate_training_data import (
    estimate_privacy_risk,
    generate_candidate_rewrites_heuristic,
    pareto_rejection_sample,
)
from src.rewriter.inference import QwenPrivacyRewriterService
from src.risk_model.attribution import SyntaxAlignedCueAttributor
from src.risk_model.calibrate import AttributeTemperatureScaler
from src.risk_model.predict import CalibratedRiskPredictor
from src.schemas.user_study import Session06RecordSchema, write_session_record

logger = logging.getLogger(__name__)

app = FastAPI(title="InferenceGuard Unified API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent services and state managers
pseudonym_mgr = DeterministicPseudonymManager()
legacy_pseudonym_mgr = PersistentPseudonymManager()
scaler = AttributeTemperatureScaler()
utility_eval = BidirectionalUtilityEvaluator()
legacy_utility_eval = UtilityEvaluator()
presidio_baseline = PresidioBaseline()
rewriter_service = QwenPrivacyRewriterService()
telemetry_logger = PrivacySafeTelemetryLogger()

# Rule-guided calibrated fallback predictor for offline / headless environments
CUE_PATTERNS: Dict[str, List[str]] = {
    "age": [
        r"\b\d{1,2}\s*(?:years old|yo|y/o)\b",
        r"\b(?:in my (?:20s|30s|40s|50s|60s|70s|twenties|thirties|forties))\b",
        r"\b(?:retired|pensioner|high school|college student|freshman|sophomore)\b",
    ],
    "location": [
        r"\b(?:living in|moved to|commute to|transit in|downtown|suburbs of)\b",
        r"\b(?:denver|seattle|austin|california|new york|boston|chicago|london|berlin|paris|tokyo|montreal|atlanta)\b",
        r"\b(?:[A-Z]{2}\b|\bco\b|\bwa\b|\btx\b|\bca\b|\bny\b)",
    ],
    "occupation": [
        r"\b(?:work as|employed at|my boss|colleagues|software engineer|nurse|doctor|teacher|barista|lawyer|psychologist)\b",
        r"\b(?:tech company|startup|hospital|clinic|law firm|school district|consultancy)\b",
    ],
    "education": [
        r"\b(?:bachelor'?s?|master'?s?|ph\.?d\.?|degree in|undergrad|grad school|alumni|major in)\b",
        r"\b(?:university of|college of|mit|stanford|harvard|berkeley|carnegie mellon)\b",
    ],
}


class RuleGuidedCalibratedPredictor:
    """Predicts inferential risk using demographic patterns and calibrated scaling."""

    def __init__(self, scaler: AttributeTemperatureScaler):
        self.scaler = scaler

    def predict(self, text: str) -> RiskReport:
        return self.predict_with_cues(text)

    def predict_with_cues(
        self,
        text: str,
        attributor=None,
        explicit_pii_counts: Optional[Dict[str, int]] = None,
    ) -> RiskReport:
        t_low = text.lower()
        attr_scores: Dict[AttributeName, float] = {}
        cues: List[AttributedCue] = []

        for attr in AttributeName:
            patterns = CUE_PATTERNS.get(attr.value, [])
            matched = any(re.search(pat, t_low) for pat in patterns)
            score = 0.72 if matched else 0.12
            attr_scores[attr] = score

        max_score = max(attr_scores.values()) if attr_scores else 0.0
        primary_attr = max(attr_scores, key=attr_scores.get) if max_score >= 0.40 else None
        overall_band = RiskBand.HIGH if max_score >= 0.65 else (RiskBand.MEDIUM if max_score >= 0.35 else RiskBand.LOW)

        attrs: Dict[AttributeName, AttributeRiskDistribution] = {}
        for attr, score in attr_scores.items():
            band = RiskBand.HIGH if score >= 0.65 else (RiskBand.MEDIUM if score >= 0.35 else RiskBand.LOW)
            attrs[attr] = AttributeRiskDistribution(
                attribute=attr,
                predicted_value="detected" if score >= 0.35 else "UNKNOWN",
                calibrated_confidence=score,
                top2_margin=0.35,
                normalized_entropy=0.30 if score >= 0.35 else 0.85,
                inferability_risk_score=score,
                risk_band=band,
                class_probabilities={},
            )

        if attributor is not None and primary_attr is not None:
            cues = attributor.extract_cues(text, [primary_attr])

        return RiskReport(
            attributes=attrs,
            overall_risk_score=max_score,
            overall_risk_band=overall_band,
            primary_risk_attribute=primary_attr,
            secondary_risk_attributes=[a for a, s in attr_scores.items() if a != primary_attr and s >= 0.35],
            cues=cues,
            explicit_pii_counts=explicit_pii_counts or {},
        )


# Instantiate production MultiTaskModernBERT model or fallback
try:
    from transformers import AutoTokenizer
    from src.risk_model.model import MultiTaskModernBERTRiskModel
    logger.info("Initializing MultiTaskModernBERTRiskModel backbone...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modernbert_model = MultiTaskModernBERTRiskModel(model_name="answerdotai/ModernBERT-base").to(device)
    risk_estimator = CalibratedRiskPredictor(
        model=modernbert_model,
        tokenizer=tokenizer,
        scaler=scaler,
        device=device,
    )
    cue_attributor = SyntaxAlignedCueAttributor(
        model=modernbert_model,
        tokenizer=tokenizer,
        scaler=scaler,
    )
    logger.info("MultiTaskModernBERTRiskModel loaded successfully.")
except Exception as exc:
    logger.warning("Could not initialize ModernBERT model (%s). Using RuleGuidedCalibratedPredictor.", exc)
    risk_estimator = RuleGuidedCalibratedPredictor(scaler=scaler)
    cue_attributor = SyntaxAlignedCueAttributor(
        model=None,
        tokenizer=None,
        scaler=scaler,
    )

leakage_tracker = BayesianConversationLeakageTracker(scaler)

pipeline = InferenceGuardPipeline(
    pseudonym_manager=pseudonym_mgr,
    risk_estimator=risk_estimator,
    cue_attributor=cue_attributor,
    rewriter_service=rewriter_service,
    utility_evaluator=utility_eval,
    leakage_tracker=leakage_tracker,
)

conversation_tracker = ConversationTracker(risk_estimator_fn=None)


# ---------------------------------------------------------------------------
# PII Extraction & Cue Matching Utilities
# ---------------------------------------------------------------------------

def extract_pii_spans(text: str) -> List[PiiSpan]:
    """Extract PiiSpan objects using Presidio analyzer or regex fallback."""
    if not text or not text.strip():
        return []
    spans: List[PiiSpan] = []

    if presidio_baseline.analyzer is not None:
        try:
            results = presidio_baseline.analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "PERSON",
                    "LOCATION",
                    "EMAIL_ADDRESS",
                    "PHONE_NUMBER",
                    "IP_ADDRESS",
                    "DATE_TIME",
                    "US_SSN",
                    "CRYPTO",
                    "CREDIT_CARD",
                    "IBAN_CODE",
                    "US_PASSPORT",
                    "US_DRIVER_LICENSE",
                    "MEDICAL_LICENSE",
                ],
            )
            for res in results:
                spans.append(
                    PiiSpan(
                        entity_type=res.entity_type,
                        start=res.start,
                        end=res.end,
                        score=float(res.score),
                    )
                )
            spans.sort(key=lambda s: (s.start, s.end))
            return spans
        except Exception as e:
            logger.warning("Presidio analyze error: %s. Falling back to regex.", e)

    entity_map = {
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "SSN": "US_SSN",
        "IP": "IP_ADDRESS",
        "PERSON": "PERSON",
        "LOCATION": "LOCATION",
        "DATE": "DATE_TIME",
        "URL": "URL",
    }
    for entity_name, pat in presidio_baseline.patterns.items():
        canonical_type = entity_map.get(entity_name, entity_name)
        for m in re.finditer(pat, text):
            start = m.start(1) if m.lastindex and m.lastindex >= 1 else m.start()
            end = m.end(1) if m.lastindex and m.lastindex >= 1 else m.end()
            spans.append(PiiSpan(entity_type=canonical_type, start=start, end=end, score=1.0))

    spans.sort(key=lambda s: (s.start, s.end))
    return spans


def to_canonical_entity_type(etype: str) -> str:
    """Map detected PII entity type string to canonical pseudonym manager category."""
    upper = etype.upper()
    if "LOC" in upper:
        return "LOC"
    if "PER" in upper:
        return "PER"
    if "ORG" in upper:
        return "ORG"
    if "DATE" in upper or "TIME" in upper:
        return "DATE"
    return "ID"


def extract_cues(text: str, scores: Dict[str, float]) -> List[Cue]:
    """Extract character-offset cues matching elevated demographic inferability."""
    found_cues: List[Cue] = []
    seen_spans = set()
    for attr, patterns in CUE_PATTERNS.items():
        score = scores.get(attr, 0.0)
        if score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR:
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    span_str = m.group(0).strip()
                    if span_str and len(span_str) >= 2 and span_str.lower() not in seen_spans:
                        seen_spans.add(span_str.lower())
                        found_cues.append(
                            Cue(
                                span=span_str,
                                attribute=attr,
                                importance=round(score, 2),
                                start=m.start(),
                                end=m.end(),
                            )
                        )
    if not found_cues:
        for attr, score in scores.items():
            if attr in ATTRIBUTES and score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR:
                found_cues.append(Cue(span=attr, attribute=attr, importance=round(score, 2)))
    return found_cues


# ---------------------------------------------------------------------------
# API Contract Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text message to analyze")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    user_id: Optional[str] = Field("default_user", description="User identifier")


class AnalyzeResponse(BaseModel):
    risk_summary: Dict[str, Any]
    original_text: Optional[str] = None
    rewritten_text: str
    presidio_text: str
    pseudonymized_text: Optional[str] = None
    utility_metrics: Dict[str, float]
    turn_record: Optional[Dict[str, Any]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None
    is_win: Optional[bool] = None


class HighlightRequest(BaseModel):
    text: str = Field(..., min_length=1)
    cues: List[AttributedCue] = Field(default_factory=list)


class WebRewriteResponse(BaseModel):
    original_text: str
    rewritten_text: str
    utility_metrics: Dict[str, float]
    pseudonymized_text: Optional[str] = None


RewriteResponse = WebRewriteResponse


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Healthcheck endpoint confirming API availability."""
    return {"status": "ok", "service": "InferenceGuard API", "version": "2.0.0"}


@app.post("/protect", response_model=ProtectionResponse)
def protect_turn(request: RewriteRequest) -> ProtectionResponse:
    """Process a single message through the full 6-step privacy protection pipeline."""
    conv_id = request.conversation_id or request.session_id or "default_session"
    return pipeline.process_turn(
        conversation_id=conv_id,
        message=request.text,
        mode=request.mode,
        commit_to_session=True,
    )


@app.post("/highlight")
def highlight_text(req: HighlightRequest) -> Dict[str, str]:
    """Render character-offset interval highlights safely without regex text replacement."""
    html_output = render_highlighted_html_from_offsets(req.text, req.cues)
    return {"highlighted_html": html_output}


@app.post("/telemetry")
def record_telemetry(event: SafeTelemetryEvent) -> Dict[str, str]:
    """Record anonymized user study event strictly validated against zero-PII whitelist."""
    telemetry_logger.log_event(event)
    return {"status": "recorded", "event_id": event.event_id}


@app.post("/session/reset")
def reset_session(session_id: str) -> Dict[str, str]:
    """Reset active conversation state and pseudonym mappings."""
    pipeline.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze input text for inferential privacy risk and return sanitized rewrite."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    session_id = request.session_id or "default"

    # Multi-turn tracking
    turn_rec = conversation_tracker.add_turn(raw_text=text, session_id=session_id)

    # Explicit PII detection on input
    input_spans = extract_pii_spans(text)

    # Inferential risk estimation & cues
    raw_scores = turn_rec.turn_scores
    cues_list: List[Cue] = extract_cues(text, raw_scores)

    before_report = compose(
        pii_spans=input_spans,
        risk_scores=raw_scores,
        cues=cues_list,
        leakage_delta=turn_rec.leakage_delta,
    )

    # Presidio baseline redaction & persistent pseudonymization
    presidio_text = presidio_baseline.redact(text)
    known_entities = [
        (text[s.start : s.end], to_canonical_entity_type(s.entity_type))
        for s in input_spans
        if s.end <= len(text)
    ]
    pseudonymized_text = legacy_pseudonym_mgr.pseudonymize(
        text, session_id=session_id, known_entities=known_entities
    )

    # Two-Stage Rewriting
    if before_report.inferential_band == Band.LOW:
        rewritten_text = text
        after_report = before_report
    else:
        source_text = pseudonymized_text if pseudonymized_text else text
        candidates = generate_candidate_rewrites_heuristic(source_text)
        import sys
        sampler = getattr(sys.modules.get("web.api"), "pareto_rejection_sample", pareto_rejection_sample)
        selected = sampler(
            original=source_text,
            candidates=candidates,
            risk_model_fn=None,
            max_risk_threshold=PROVISIONAL_MAX_RISK_THRESHOLD,
            min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
        )
        rewritten_text = selected[0] if selected else source_text

        rewrite_spans = extract_pii_spans(rewritten_text)
        after_scores = estimate_privacy_risk(rewritten_text, risk_model_fn=None)
        after_cues: List[Cue] = extract_cues(rewritten_text, after_scores)
        after_report = compose(
            pii_spans=rewrite_spans,
            risk_scores=after_scores,
            cues=after_cues,
        )

    utility_metrics = legacy_utility_eval.compute_metrics(text, rewritten_text)
    utility_score = utility_metrics.get("utility_score")
    comparison = compare(
        before=before_report,
        after=after_report,
        utility=utility_score,
        utility_floor=DEFAULT_UTILITY_FLOOR,
    )

    comparison_dict = dataclasses.asdict(comparison)
    comparison_dict["is_win"] = comparison.is_win

    return AnalyzeResponse(
        risk_summary=before_report.inferential.to_dict(),
        original_text=text,
        rewritten_text=rewritten_text,
        presidio_text=presidio_text,
        pseudonymized_text=pseudonymized_text,
        utility_metrics=utility_metrics,
        turn_record=turn_rec.to_dict(),
        privacy_report=before_report.to_dict(),
        comparison=comparison_dict,
        verdict=comparison.verdict,
        is_win=comparison.is_win,
    )


@app.post("/rewrite", response_model=WebRewriteResponse)
def rewrite_text(request: RewriteRequest) -> WebRewriteResponse:
    """Rewrite text under inferential privacy and explicit PII constraints."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    session_id = request.session_id or request.conversation_id or "default"
    input_spans = extract_pii_spans(text)
    known_entities = [
        (text[s.start : s.end], to_canonical_entity_type(s.entity_type))
        for s in input_spans
        if s.end <= len(text)
    ]
    pseudo_text = legacy_pseudonym_mgr.pseudonymize(
        text, session_id=session_id, known_entities=known_entities
    )

    candidates = generate_candidate_rewrites_heuristic(pseudo_text)
    import sys
    sampler = getattr(sys.modules.get("web.api"), "pareto_rejection_sample", pareto_rejection_sample)
    selected = sampler(
        original=pseudo_text,
        candidates=candidates,
        risk_model_fn=None,
        max_risk_threshold=request.max_risk or PROVISIONAL_MAX_RISK_THRESHOLD,
        min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
    )
    rewritten_text = selected[0] if selected else pseudo_text
    utility_metrics = legacy_utility_eval.compute_metrics(text, rewritten_text)

    return WebRewriteResponse(
        original_text=text,
        rewritten_text=rewritten_text,
        utility_metrics=utility_metrics,
        pseudonymized_text=pseudo_text,
    )


@app.post("/log_session")
def log_session_record(record: Session06RecordSchema) -> Dict[str, Any]:
    """Append validated user study record atomically to session log."""
    log_file = Path("docs/userstudy/session06_record.json")
    validated = write_session_record(record, log_file)
    return {"status": "success", "participant_id": validated.participant_id}


def launch_tunnel(port: int = 8000, tunnel_type: Optional[str] = None) -> str:
    """Launch public tunnel for Colab environments using ngrok, localtunnel, or cloudflared."""
    if tunnel_type == "cloudflared" or (tunnel_type is None and shutil.which("cloudflared")):
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(30):
                line = proc.stderr.readline() if proc.stderr else ""
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    return match.group(0)
        except Exception as e:
            logger.warning("cloudflared tunnel failed: %s", e)

    if tunnel_type == "localtunnel" or (tunnel_type is None and shutil.which("lt")):
        try:
            proc = subprocess.Popen(
                ["lt", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(20):
                line = proc.stdout.readline() if proc.stdout else ""
                match = re.search(r"https://[a-zA-Z0-9-]+\.loca\.lt", line)
                if match:
                    return match.group(0)
        except Exception as e:
            logger.warning("localtunnel failed: %s", e)

    return f"http://localhost:{port}"


# Mount static web directory
web_dir = Path("web")
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
