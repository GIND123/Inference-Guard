"""Lightweight FastAPI server for InferenceGuard with Colab tunneling support.

Exposes /analyze and /rewrite endpoints, static web interface mounting,
and pyngrok / localtunnel / cloudflared tunnel initialization.
Conforms strictly to PROJECT.md interface contracts.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.conversation.state import ConversationTracker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.utility import UtilityEvaluator
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

import torch
from src.risk_model.model import ModernBertRiskClassifier

logger = logging.getLogger(__name__)

app = FastAPI(title="InferenceGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ModernBERT if available
risk_model = None
model_path = Path("artifacts/risk_model/best_model")
if model_path.exists():
    try:
        from transformers import AutoTokenizer
        logger.info("Loading trained ModernBERT risk classifier...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = ModernBertRiskClassifier.from_pretrained(str(model_path)).to(device)
        model.eval()

        def ml_risk_estimator(text: str) -> Dict[str, float]:
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs)
                scores = {attr: torch.sigmoid(logits[i]).item() for i, attr in enumerate(["age", "location", "occupation", "education"])}
            scores["overall"] = max(scores.values()) if scores else 0.0
            return scores
            
        risk_model = ml_risk_estimator
        logger.info("ModernBERT loaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to load ModernBERT: {e}")

# Global persistent services
conversation_tracker = ConversationTracker(risk_estimator_fn=risk_model)
presidio_baseline = PresidioBaseline()
utility_evaluator = UtilityEvaluator()

logger.info("Initializing Qwen Rewriter Inference Engine...")
try:
    from src.rewriter import DEFAULT_BASE_MODEL, QwenRewriterInference
    qwen_rewriter = QwenRewriterInference(
        model_name_or_path=DEFAULT_BASE_MODEL,
        adapter_path="artifacts/rewriter_qlora"
    )
    logger.info("Qwen Rewriter successfully initialized!")
except Exception as e:
    logger.warning(f"Failed to initialize Qwen Rewriter: {e}")
    qwen_rewriter = None


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
            logger.warning(f"Presidio analyze error: {e}. Falling back to regex.")

    # Fallback to regex pattern extraction mapped to canonical types in HARD_PII_TYPES / report.py
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
            if m.lastindex and m.lastindex >= 1:
                start, end = m.start(1), m.end(1)
            else:
                start, end = m.start(), m.end()
            spans.append(
                PiiSpan(
                    entity_type=canonical_type,
                    start=start,
                    end=end,
                    score=1.0,
                )
            )

    spans.sort(key=lambda s: (s.start, s.end))
    return spans


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text message to analyze")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    user_id: Optional[str] = Field("default_user", description="User identifier")


class AnalyzeResponse(BaseModel):
    risk_summary: Dict[str, Any]
    rewritten_text: str
    presidio_text: str
    utility_metrics: Dict[str, float]
    turn_record: Optional[Dict[str, Any]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None
    is_win: Optional[bool] = None


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_risk: float = Field(default=PROVISIONAL_MAX_RISK_THRESHOLD, ge=0.0, le=1.0)


class RewriteResponse(BaseModel):
    original_text: str
    rewritten_text: str
    utility_metrics: Dict[str, float]


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze input text for inferential privacy risk and return sanitized rewrite."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    session_id = request.session_id or "default"

    # 1. Multi-turn tracking
    turn_rec = conversation_tracker.add_turn(raw_text=text, session_id=session_id)

    # 2. Explicit PII detection on input
    input_spans = extract_pii_spans(text)

    # 3. Inferential risk estimation & cues
    raw_scores = turn_rec.turn_scores
    cues_list: List[Cue] = []
    for attr, score in raw_scores.items():
        if score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR:
            cues_list.append(Cue(span=f"[{attr} cue]", attribute=attr, importance=round(score, 2)))

    # Delegate input privacy report composition to src.product.report.compose
    before_report = compose(
        pii_spans=input_spans,
        risk_scores=raw_scores,
        cues=cues_list,
        leakage_delta=turn_rec.leakage_delta,
    )

    # 4. Presidio baseline redaction
    presidio_text = presidio_baseline.redact(text)

    # 5. Rewriter candidate generation and rejection sampling
    if before_report.inferential_band == Band.LOW:
        rewritten_text = text
        after_report = before_report
    else:
        if qwen_rewriter is not None and qwen_rewriter.is_adapter_loaded:
            candidates = [qwen_rewriter.rewrite(text)]
        else:
            candidates = generate_candidate_rewrites_heuristic(text)

        selected = pareto_rejection_sample(
            original=text,
            candidates=candidates,
            max_risk_threshold=PROVISIONAL_MAX_RISK_THRESHOLD,
            min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
        )

        if selected is not None:
            rewritten_text, _ = selected
        else:
            if qwen_rewriter is not None and qwen_rewriter.is_adapter_loaded:
                heuristic_cands = generate_candidate_rewrites_heuristic(text)
                selected_heur = pareto_rejection_sample(
                    original=text,
                    candidates=heuristic_cands,
                    max_risk_threshold=PROVISIONAL_MAX_RISK_THRESHOLD,
                    min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
                )
                if selected_heur is not None:
                    rewritten_text, _ = selected_heur
                else:
                    rewritten_text = presidio_text
            else:
                rewritten_text = presidio_text

        # Extract explicit PII spans on rewrite
        rewrite_spans = extract_pii_spans(rewritten_text)
        after_scores = estimate_privacy_risk(rewritten_text, risk_model_fn=risk_model)
        after_cues: List[Cue] = [
            Cue(span=f"[{attr} cue]", attribute=attr, importance=round(score, 2))
            for attr, score in after_scores.items()
            if attr in ATTRIBUTES and score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR
        ]
        after_report = compose(
            pii_spans=rewrite_spans,
            risk_scores=after_scores,
            cues=after_cues,
        )

    # 6. Utility metrics
    utility_metrics = utility_evaluator.compute_metrics(text, rewritten_text)

    # 7. Comparison and Rule 2 veto enforcement
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
        rewritten_text=rewritten_text,
        presidio_text=presidio_text,
        utility_metrics=utility_metrics,
        turn_record=turn_rec.to_dict(),
        privacy_report=before_report.to_dict(),
        comparison=comparison_dict,
        verdict=comparison.verdict,
        is_win=comparison.is_win,
    )


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_text(request: RewriteRequest) -> RewriteResponse:
    """Rewrite text with privacy constraints."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if qwen_rewriter is not None and qwen_rewriter.is_adapter_loaded:
        candidates = [qwen_rewriter.rewrite(text)]
    else:
        candidates = generate_candidate_rewrites_heuristic(text)

    selected = pareto_rejection_sample(
        original=text,
        candidates=candidates,
        risk_model_fn=risk_model,
        max_risk_threshold=request.max_risk,
        min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
    )
    if selected is not None:
        rewritten, _ = selected
    else:
        if qwen_rewriter is not None and qwen_rewriter.is_adapter_loaded:
            heuristic_cands = generate_candidate_rewrites_heuristic(text)
            selected_heur = pareto_rejection_sample(
                original=text,
                candidates=heuristic_cands,
                risk_model_fn=risk_model,
                max_risk_threshold=request.max_risk,
                min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
            )
            if selected_heur is not None:
                rewritten, _ = selected_heur
            else:
                rewritten = presidio_baseline.redact(text)
        else:
            rewritten = presidio_baseline.redact(text)

    utility = utility_evaluator.compute_metrics(text, rewritten)
    return RewriteResponse(
        original_text=text,
        rewritten_text=rewritten,
        utility_metrics=utility,
    )


def launch_tunnel(
    port: int = 8000,
    tunnel_type: str = "pyngrok",
    authtoken: Optional[str] = None,
) -> str:
    """Launch public tunnel to expose local FastAPI server from Colab.

    Supports 'pyngrok', 'localtunnel', and 'cloudflared'.
    """
    if tunnel_type == "pyngrok":
        try:
            from pyngrok import ngrok
            if authtoken:
                ngrok.set_auth_token(authtoken)
            tunnel = ngrok.connect(port)
            url = str(tunnel.public_url)
            logger.info(f"pyngrok tunnel established: {url}")
            return url
        except Exception as e:
            logger.warning(f"pyngrok failed: {e}. Trying localtunnel.")

    if tunnel_type in ("localtunnel", "pyngrok") and shutil.which("npx"):
        try:
            proc = subprocess.Popen(
                ["npx", "localtunnel", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Read first line of output
            line = proc.stdout.readline() if proc.stdout else ""
            if "url is:" in line:
                url = line.split("url is:")[-1].strip()
                logger.info(f"localtunnel established: {url}")
                return url
        except Exception as e:
            logger.warning(f"localtunnel failed: {e}. Trying cloudflared.")

    if shutil.which("cloudflared"):
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                stderr=subprocess.PIPE,
                text=True,
            )
            return "http://localhost:8000 (cloudflared background)"
        except Exception as e:
            logger.warning(f"cloudflared failed: {e}")

    return f"http://localhost:{port}"


# Mount static files to serve the minimalist web frontend
web_dir = Path(__file__).resolve().parent
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
