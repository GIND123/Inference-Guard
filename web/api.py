"""Lightweight FastAPI server for InferenceGuard with Colab tunneling support.

Exposes /analyze and /rewrite endpoints, static web interface mounting,
and pyngrok / localtunnel / cloudflared tunnel initialization.
Conforms strictly to PROJECT.md interface contracts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.conversation.state import ConversationTracker
from src.evaluation.presidio_baseline import PresidioBaseline
from src.evaluation.utility import UtilityEvaluator, check_degeneracy
from src.product.report import (
    DEFAULT_UTILITY_FLOOR,
    PiiSpan,
    PrivacyReport,
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
    neutralize_education_cues,
    neutralize_occupation_cues,
    pareto_rejection_sample,
    scrub_hard_pii,
)

import torch
from src.risk_model.model import ModernBertRiskClassifier

logger = logging.getLogger(__name__)

app = FastAPI(title="InferenceGuard API", version="1.0.0")


def allowed_origins() -> List[str]:
    """CORS origins from IG_ALLOWED_ORIGINS (comma-separated).

    Defaults to "*" so the Colab + tunnel demo keeps working. That default,
    combined with allow_credentials=True, lets any web page call this API
    from a visitor's browser; set IG_ALLOWED_ORIGINS for anything beyond a
    class demo. See docs/engineering/production-readiness.md.
    """
    raw = os.environ.get("IG_ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_risk_estimator(model: Any, tokenizer: Any, device: str) -> Callable[[str], Dict[str, float]]:
    """Wrap a ModernBertRiskClassifier as the text -> scores function the API uses."""

    def ml_risk_estimator(text: str) -> Dict[str, float]:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs)
            scores = {attr: torch.sigmoid(logits[i]).item() for i, attr in enumerate(["age", "location", "occupation", "education"])}

        rule_scores = estimate_privacy_risk(text, risk_model_fn=None)
        scores = {
            attr: round(0.5 * scores[attr] + 0.5 * rule_scores.get(attr, 0.05), 4)
            for attr in ["age", "location", "occupation", "education"]
        }

        text_lower = text.lower()
        has_first_person = bool(re.search(
            r"\b(i|i'm|i am|my|me|myself|mine|we|our|ours|us|i've|i'd|i'll)\b", text_lower
        )) or bool(re.search(
            r"\b(?:living in|commute to|working as|graduated from|started my career|my job|my role|my co-op)\b", text_lower
        ))

        if not has_first_person:
            # Third-party or factual context: heavily discount demographic inferential scores
            scores["location"] = min(scores["location"], 0.15)
            scores["education"] = min(scores["education"], 0.15)
            scores["occupation"] = min(scores["occupation"], 0.15)
            scores["age"] = min(scores["age"], 0.15)
        else:
            # Boost signature indirect cues that are sparse in SynthPAI training data
            if re.search(r"\b(?:green line|red line|orange line|blue line|silver line|mbta|sound transit|bart|caltrain)\b", text_lower):
                scores["location"] = max(scores["location"], 0.55)
            if re.search(r"\b(?:co-op|intern|internship|residency|resident|charge nurse)\b", text_lower):
                scores["occupation"] = max(scores["occupation"], 0.50)
            if re.search(r"\b(?:campus|coursework|dorm|undergraduate|thesis|dissertation|doctoral committee)\b", text_lower):
                scores["education"] = max(scores["education"], 0.50)
            if re.search(r"\b(?:started my (?:it\s+)?career back in|after \d{1,2} years (?:of|in)|before retirement)\b", text_lower):
                scores["age"] = max(scores["age"], 0.50)
            if re.search(r"\b(?:diagnose|patient|child|son|daughter|infant|toddler)\b.*?\b\d{1,2}[-\s]*years?[-\s]*old\b", text_lower):
                if not re.search(r"\b(?:i am|i'm|in my)\s+\d{1,2}\b", text_lower):
                    scores["age"] = min(scores["age"], 0.15)

        scores["overall"] = max(scores.values()) if scores else 0.0
        return scores

    return ml_risk_estimator


REPO_ROOT = Path(__file__).resolve().parent.parent


def find_adapter_path() -> Optional[Path]:
    """Find the rewriter adapter directory across common execution contexts."""
    env_path = os.environ.get("REWRITER_ADAPTER_PATH")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        REPO_ROOT / "artifacts" / "rewriter_qlora",
        Path("artifacts/rewriter_qlora"),
        Path("/content/drive/MyDrive/Inference-Guard/artifacts/rewriter_qlora"),
        Path("/content/Inference-Guard/artifacts/rewriter_qlora"),
        Path("/content/artifacts/rewriter_qlora"),
    ])
    for p in candidates:
        if p.exists() and ((p / "adapter_config.json").exists() or p.is_dir()):
            return p.resolve()
    return None


def find_risk_model_path() -> Optional[Path]:
    """Find the ModernBERT risk classifier directory across common execution contexts."""
    env_path = os.environ.get("RISK_MODEL_PATH")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        REPO_ROOT / "artifacts" / "risk_model" / "best_model",
        Path("artifacts/risk_model/best_model"),
        Path("/content/drive/MyDrive/Inference-Guard/artifacts/risk_model/best_model"),
        Path("/content/Inference-Guard/artifacts/risk_model/best_model"),
    ])
    for p in candidates:
        if p.exists() and ((p / "config.json").exists() or p.is_dir()):
            return p.resolve()
    return None


# Load trained ModernBERT if available
risk_model = None
model_path = find_risk_model_path()
if model_path is not None:
    try:
        from transformers import AutoTokenizer
        logger.info("Loading trained ModernBERT risk classifier from '%s'...", model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = ModernBertRiskClassifier.from_pretrained(str(model_path)).to(device)
        model.eval()

        risk_model = make_risk_estimator(model, tokenizer, device)
        logger.info("ModernBERT loaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to load ModernBERT: {e}")


def get_risk_model() -> Optional[Callable[[str], Dict[str, float]]]:
    """Retrieve or dynamically load ModernBERT risk estimator if available on disk."""
    global risk_model
    if risk_model is not None:
        return risk_model
    m_path = find_risk_model_path()
    if m_path is not None:
        try:
            from transformers import AutoTokenizer
            logger.info("Dynamically loading trained ModernBERT from '%s'...", m_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            model = ModernBertRiskClassifier.from_pretrained(str(m_path)).to(device)
            model.eval()
            risk_model = make_risk_estimator(model, tokenizer, device)
            logger.info("ModernBERT loaded dynamically!")
        except Exception as exc:
            logger.warning(f"Failed to dynamically load ModernBERT: {exc}")
    return risk_model


# Global persistent services
conversation_tracker = ConversationTracker(risk_estimator_fn=risk_model)
presidio_baseline = PresidioBaseline()
utility_evaluator = UtilityEvaluator(load_models=True)
qwen_rewriter = None


def get_qwen_rewriter():
    """Retrieve or dynamically initialize the QwenRewriterInference engine when adapter is trained."""
    global qwen_rewriter
    if qwen_rewriter is not None and getattr(qwen_rewriter, "is_adapter_loaded", False):
        return qwen_rewriter

    adapter_p = find_adapter_path()
    if adapter_p is not None:
        try:
            from src.rewriter import DEFAULT_BASE_MODEL, QwenRewriterInference
            logger.info("Loading Qwen Rewriter from '%s'...", adapter_p)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            qwen_rewriter = QwenRewriterInference(
                model_name_or_path=DEFAULT_BASE_MODEL,
                adapter_path=str(adapter_p),
                device=device,
                load_in_4bit=torch.cuda.is_available(),
            )
            logger.info("Qwen Rewriter loaded dynamically! is_adapter_loaded: %s", qwen_rewriter.is_adapter_loaded)
        except Exception as exc:
            logger.warning("Dynamic reload of Qwen Rewriter failed: %s", exc)
    return qwen_rewriter


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

    # Filter out false positive uppercase acronyms tagged as PERSON (e.g. ED = Emergency Department)
    filtered_spans: List[PiiSpan] = []
    for s in spans:
        span_text = text[s.start:s.end].strip()
        if s.entity_type == "PERSON" and span_text.upper() in {"ED", "ER", "ICU", "OR", "IT", "HR", "PR", "AI", "ML", "RN", "MD"}:
            continue
        filtered_spans.append(s)

    filtered_spans.sort(key=lambda s: (s.start, s.end))
    return filtered_spans


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text message to analyze")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    user_id: Optional[str] = Field("default_user", description="User identifier")


# Which component produced rewritten_text. Session 06 participants trusted
# output that was identical to their input, so the response has to say.
ENGINE_QWEN = "qwen_adapter"
ENGINE_HEURISTIC = "heuristic"
ENGINE_PRESIDIO = "presidio_fallback"
ENGINE_PASSTHROUGH = "passthrough_low_risk"
REWRITE_ENGINES = (ENGINE_QWEN, ENGINE_HEURISTIC, ENGINE_PRESIDIO, ENGINE_PASSTHROUGH)

TIMING_STAGES = ("presidio", "risk_model", "rewrite", "rescore", "utility", "total")


class AnalyzeResponse(BaseModel):
    risk_summary: Dict[str, Any]
    rewritten_text: str
    presidio_text: str
    utility_metrics: Dict[str, Any]
    turn_record: Optional[Dict[str, Any]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None
    is_win: Optional[bool] = None
    rewrite_engine: str
    rewrite_changed: bool
    # SHA-256 of exactly the rewritten_text string returned, so a study log
    # can be checked against what the UI displayed without logging the text.
    served_text_sha256: str
    # Wall-clock milliseconds per stage; None for a stage that did not run.
    timings_ms: Dict[str, Optional[float]]
    debug_info: Optional[Dict[str, Any]] = None


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_risk: float = Field(default=PROVISIONAL_MAX_RISK_THRESHOLD, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(default="default", description="Session ID")


class RewriteResponse(BaseModel):
    original_text: str
    rewritten_text: str
    utility_metrics: Dict[str, Any]
    rewrite_engine: str
    rewrite_changed: bool


def _normalise_ws(text: str) -> str:
    return " ".join(text.split())


def rewrite_changed(original: str, rewritten: str) -> bool:
    """True if the rewrite differs from the input beyond whitespace."""
    return _normalise_ws(original) != _normalise_ws(rewritten)


def served_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def presidio_stage(text: str) -> Tuple[List[PiiSpan], str]:
    """Explicit PII spans on the input, and the Presidio-redacted baseline text."""
    return extract_pii_spans(text), presidio_baseline.redact(text)


def risk_stage(text: str, session_id: str, input_spans: List[PiiSpan]) -> Tuple[Any, PrivacyReport]:
    """Multi-turn tracking and inferential risk (the risk model runs inside add_turn)."""
    active_rm = get_risk_model()
    if active_rm is not None:
        conversation_tracker.risk_estimator_fn = active_rm
    turn_rec = conversation_tracker.add_turn(raw_text=text, session_id=session_id)
    cues_list: List[Cue] = []
    for attr, score in turn_rec.turn_scores.items():
        if score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR:
            cues_list.append(Cue(span=f"[{attr} cue]", attribute=attr, importance=round(score, 2)))

    # Delegate input privacy report composition to src.product.report.compose
    before_report = compose(
        pii_spans=input_spans,
        risk_scores=turn_rec.turn_scores,
        cues=cues_list,
        leakage_delta=turn_rec.leakage_delta,
    )
    return turn_rec, before_report


def _is_impersonal_query(text: str) -> bool:
    """Check if text is an impersonal factual question or task with no first-person disclosure."""
    text_lower = text.lower()
    has_self_ref = bool(re.search(
        r"\b(i|i'm|i am|my|me|myself|mine|we|our|ours|us|i've|i'd|i'll)\b",
        text_lower,
    )) or bool(re.search(
        r"\b(?:living in|commute to|working as|graduated from|started my career|my job|my role|my co-op)\b",
        text_lower,
    ))
    return not has_self_ref


last_debug_trace: Dict[str, Any] = {}


def select_rewrite(
    text: str,
    presidio_text: str,
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]],
    max_risk: float,
) -> Tuple[str, str]:
    """Pick the served rewrite and name the engine that produced it.

    Order: Qwen adapter (if loaded) -> heuristic candidates -> Presidio text.
    QwenRewriterInference.rewrite() falls back to the heuristic internally on
    a generation failure and returns a plain string, so an output identical
    to that fallback is labelled "heuristic", never "qwen_adapter".
    """
    global last_debug_trace
    trace: Dict[str, Any] = {
        "timestamp": time.time(),
        "input_text": text,
        "qwen_adapter_loaded": False,
        "qwen_output_raw": None,
        "qwen_clean_output": None,
        "qwen_error": None,
        "qwen_is_fallback": False,
        "qwen_changed": False,
        "qwen_had_pii": False,
        "candidates_evaluated": [],
        "decision_reason": "",
        "selected_engine": ENGINE_PRESIDIO,
        "selected_text": presidio_text,
    }

    rewriter = get_qwen_rewriter()
    if rewriter is not None and getattr(rewriter, "is_adapter_loaded", False):
        trace["qwen_adapter_loaded"] = True
        try:
            qwen_out = rewriter.rewrite(text)
        except Exception as exc:
            logger.warning("Qwen rewriter generation failed: %s", exc)
            trace["qwen_error"] = str(exc)
            qwen_out = ""

        if not trace["qwen_error"] and hasattr(rewriter, "last_error") and rewriter.last_error:
            trace["qwen_error"] = rewriter.last_error

        is_fallback = (
            hasattr(rewriter, "_heuristic_fallback")
            and qwen_out == rewriter._heuristic_fallback(text)
        )
        engine = ENGINE_HEURISTIC if is_fallback else ENGINE_QWEN
        trace["qwen_is_fallback"] = is_fallback
        trace["qwen_output_raw"] = qwen_out

        clean_qwen = qwen_out.strip() if qwen_out else ""
        trace["qwen_clean_output"] = clean_qwen
        is_changed = rewrite_changed(text, clean_qwen)
        trace["qwen_changed"] = is_changed

        if (
            clean_qwen
            and not check_degeneracy(text, clean_qwen)["degenerate"]
            and is_changed
        ):
            _, had_pii = scrub_hard_pii(clean_qwen)
            trace["qwen_had_pii"] = had_pii
            if not had_pii:
                selected = pareto_rejection_sample(
                    original=text,
                    candidates=[clean_qwen],
                    risk_model_fn=risk_model_fn,
                    embedder_fn=utility_evaluator.compute_similarity,
                    max_risk_threshold=max_risk,
                    min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
                )
                if selected is not None:
                    trace["selected_engine"] = engine
                    trace["selected_text"] = selected[0]
                    trace["decision_reason"] = f"Accepted via Pareto rejection sampling ({engine})."
                    last_debug_trace = trace
                    return selected[0], engine



    # Heuristic Candidate Evaluation
    heuristic_cands = generate_candidate_rewrites_heuristic(text)
    selected = pareto_rejection_sample(
        original=text,
        candidates=heuristic_cands,
        risk_model_fn=risk_model_fn,
        embedder_fn=utility_evaluator.compute_similarity,
        max_risk_threshold=max_risk,
        min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
    )
    if selected is not None:
        trace["selected_engine"] = ENGINE_HEURISTIC
        trace["selected_text"] = selected[0]
        trace["decision_reason"] = "Heuristic candidate satisfied strict Pareto thresholds."
        last_debug_trace = trace
        return selected[0], ENGINE_HEURISTIC

    cleaned_presidio = re.sub(r"<LOCATION>,\s*[A-Z]{2}\b", "<LOCATION>", presidio_text) if "<LOCATION>" in presidio_text else presidio_text
    trace["selected_engine"] = ENGINE_PRESIDIO
    trace["selected_text"] = cleaned_presidio
    trace["decision_reason"] = "No candidate satisfied Pareto risk threshold; served Presidio baseline."
    last_debug_trace = trace
    return cleaned_presidio, ENGINE_PRESIDIO


def rescore_stage(rewritten_text: str) -> PrivacyReport:
    """Explicit PII and inferential risk on the rewrite."""
    active_risk = get_risk_model()
    rewrite_spans = extract_pii_spans(rewritten_text)
    after_scores = estimate_privacy_risk(rewritten_text, risk_model_fn=active_risk)
    after_cues: List[Cue] = [
        Cue(span=f"[{attr} cue]", attribute=attr, importance=round(score, 2))
        for attr, score in after_scores.items()
        if attr in ATTRIBUTES and score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR
    ]
    return compose(
        pii_spans=rewrite_spans,
        risk_scores=after_scores,
        cues=after_cues,
    )


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/status")
def server_status() -> Dict[str, Any]:
    """Diagnostic status endpoint inspecting loaded neural engines, adapter paths, and VRAM."""
    rewriter = get_qwen_rewriter()
    rm = get_risk_model()
    adapter_p = find_adapter_path()
    risk_p = find_risk_model_path()

    cuda_avail = torch.cuda.is_available()
    vram_mb = None
    if cuda_avail:
        try:
            vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    return {
        "status": "ok",
        "qwen_adapter_loaded": bool(rewriter and getattr(rewriter, "is_adapter_loaded", False)),
        "qwen_adapter_path": str(adapter_p) if adapter_p else None,
        "risk_model_loaded": bool(rm is not None),
        "risk_model_path": str(risk_p) if risk_p else None,
        "cuda_available": cuda_avail,
        "device": "cuda" if cuda_avail else "cpu",
        "vram_allocated_mb": vram_mb,
    }


@app.get("/debug/last_rewrite")
def get_last_rewrite_debug() -> Dict[str, Any]:
    """Inspect trace and decisions of the most recent rewrite."""
    return last_debug_trace or {"status": "no rewrites processed yet"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze input text for inferential privacy risk and return sanitized rewrite."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    session_id = request.session_id or "default"
    timings: Dict[str, Optional[float]] = {stage: None for stage in TIMING_STAGES}
    t_total = time.perf_counter()

    # 1. Explicit PII detection on input + Presidio baseline redaction
    t = time.perf_counter()
    input_spans, presidio_text = presidio_stage(text)
    timings["presidio"] = _ms_since(t)

    # 2. Multi-turn tracking and inferential risk estimation & cues
    t = time.perf_counter()
    turn_rec, before_report = risk_stage(text, session_id, input_spans)
    timings["risk_model"] = _ms_since(t)

    # 3. Rewriter candidate generation and rejection sampling
    # Passthrough is strictly prohibited if any hard explicit PII is present (Rule 2)
    if (before_report.inferential_band == Band.LOW or _is_impersonal_query(text)) and before_report.explicit.hard_count == 0:
        rewritten_text = text
        engine = ENGINE_PASSTHROUGH
        after_report = before_report
    else:
        active_risk = get_risk_model()
        t = time.perf_counter()
        rewritten_text, engine = select_rewrite(
            text,
            presidio_text,
            risk_model_fn=active_risk,
            max_risk=PROVISIONAL_MAX_RISK_THRESHOLD,
        )
        timings["rewrite"] = _ms_since(t)

        t = time.perf_counter()
        after_report = rescore_stage(rewritten_text)
        timings["rescore"] = _ms_since(t)

    # 4. Utility metrics
    t = time.perf_counter()
    utility_metrics = utility_evaluator.compute_metrics(text, rewritten_text)
    timings["utility"] = _ms_since(t)

    # 5. Comparison and Rule 2 veto enforcement
    utility_score = utility_metrics.get("utility_score")
    comparison = compare(
        before=before_report,
        after=after_report,
        utility=utility_score,
        utility_floor=DEFAULT_UTILITY_FLOOR,
    )

    comparison_dict = dataclasses.asdict(comparison)
    comparison_dict["is_win"] = comparison.is_win
    timings["total"] = _ms_since(t_total)

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
        rewrite_engine=engine,
        rewrite_changed=rewrite_changed(text, rewritten_text),
        served_text_sha256=served_text_sha256(rewritten_text),
        timings_ms=timings,
        debug_info=last_debug_trace,
    )


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_text(request: RewriteRequest) -> RewriteResponse:
    """Rewrite text with privacy constraints."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    active_risk = get_risk_model()
    rewritten, engine = select_rewrite(
        text,
        presidio_baseline.redact(text),
        risk_model_fn=active_risk,
        max_risk=request.max_risk,
    )

    utility = utility_evaluator.compute_metrics(text, rewritten)
    return RewriteResponse(
        original_text=text,
        rewritten_text=rewritten,
        utility_metrics=utility,
        rewrite_engine=engine,
        rewrite_changed=rewrite_changed(text, rewritten),
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
