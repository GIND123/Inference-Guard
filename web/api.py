"""Lightweight FastAPI server for InferenceGuard with Colab tunneling support.

Exposes /analyze and /rewrite endpoints, static web interface mounting,
and pyngrok / localtunnel / cloudflared tunnel initialization.
Conforms strictly to PROJECT.md interface contracts.
"""

from __future__ import annotations

import logging
import os
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
from src.product.risk_bands import Cue, summarize
from src.rewriter.generate_training_data import (
    estimate_privacy_risk,
    generate_candidate_rewrites_heuristic,
    pareto_rejection_sample,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="InferenceGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global persistent services
conversation_tracker = ConversationTracker()
presidio_baseline = PresidioBaseline()
utility_evaluator = UtilityEvaluator()


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


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_risk: float = Field(0.30, ge=0.0, le=1.0)


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

    # 2. Risk estimation
    raw_scores = turn_rec.turn_scores
    # Extract cues
    cues_list: List[Cue] = []
    # Identify cues based on attributes clearing MEDIUM threshold
    for attr, score in raw_scores.items():
        if score >= 0.20:
            cues_list.append(Cue(span=f"[{attr} cue]", attribute=attr, importance=round(score, 2)))

    risk_summary_obj = summarize(
        scores=raw_scores,
        cues=cues_list,
        leakage_delta=turn_rec.leakage_delta,
    )
    risk_summary = risk_summary_obj.to_dict()

    # 3. Presidio baseline redaction
    presidio_text = presidio_baseline.redact(text)

    # 4. Rewriter candidate generation and rejection sampling
    candidates = generate_candidate_rewrites_heuristic(text)
    selected = pareto_rejection_sample(
        original=text,
        candidates=candidates,
        max_risk_threshold=0.30,
        min_cosine_threshold=0.30,
    )

    if selected is not None:
        rewritten_text, _ = selected
    else:
        rewritten_text = candidates[0] if candidates else presidio_text

    # 5. Utility metrics
    utility_metrics = utility_evaluator.compute_metrics(text, rewritten_text)

    return AnalyzeResponse(
        risk_summary=risk_summary,
        rewritten_text=rewritten_text,
        presidio_text=presidio_text,
        utility_metrics=utility_metrics,
        turn_record=turn_rec.to_dict(),
    )


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_text(request: RewriteRequest) -> RewriteResponse:
    """Rewrite text with privacy constraints."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    candidates = generate_candidate_rewrites_heuristic(text)
    selected = pareto_rejection_sample(
        original=text,
        candidates=candidates,
        max_risk_threshold=request.max_risk,
        min_cosine_threshold=0.30,
    )
    if selected is not None:
        rewritten, _ = selected
    else:
        rewritten = candidates[0] if candidates else text

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
