"""Lightweight FastAPI server for InferenceGuard with Colab tunneling support.

Exposes /analyze and /rewrite endpoints, static web interface mounting,
and pyngrok / localtunnel / cloudflared tunnel initialization.
Conforms strictly to PROJECT.md interface contracts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.contracts.schemas import ConversationState
from src.conversation.entity_map import DeterministicPseudonymManager
from src.conversation.state import ConversationTracker
from src.evaluation.presidio_baseline import (
    PresidioBaseline,
    is_duration_phrase,
    is_non_identifying_presidio_span,
)
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
    evaluate_rewrite_utility,
    generate_candidate_rewrites_heuristic,
    neutralize_education_cues,
    neutralize_occupation_cues,
    pareto_rejection_sample,
    scrub_hard_pii,
)

import torch
from src.rewriter.inference import DEFAULT_BASE_MODEL
from src.risk_model.model import ModernBertRiskClassifier

logger = logging.getLogger(__name__)

app = FastAPI(title="InferenceGuard API", version="1.0.0")


def allowed_origins() -> List[str]:
    """CORS origins from IG_ALLOWED_ORIGINS (comma-separated).

    Defaults to "*" so the Colab + tunnel demo keeps working. The wildcard
    default is NOT combined with credentialed requests (see
    cors_allow_credentials); set IG_ALLOWED_ORIGINS to explicit origins for
    anything beyond a class demo. See docs/engineering/production-readiness.md.
    """
    raw = os.environ.get("IG_ALLOWED_ORIGINS", "*")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


def cors_allow_credentials(origins: List[str]) -> bool:
    """Decide whether CORS credentials may be allowed for the given origins.

    Browsers reject ``Access-Control-Allow-Origin: *`` together with
    ``Access-Control-Allow-Credentials: true``, and allowing both would also
    let any site make credentialed calls. So credentials are permitted ONLY
    when ``origins`` is an explicit allow-list; a wildcard (``['*']``, or an
    empty/falsey value that resolves to wildcard) yields ``False``. This keeps
    the ``*`` convenience default working for the unauthenticated Colab demo,
    just without credentialed wildcard.
    """
    if not origins:
        return False
    if any(o == "*" for o in origins):
        return False
    return True


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=cors_allow_credentials(allowed_origins()),
    allow_methods=["*"],
    allow_headers=["*"],
)


def session_ttl_seconds() -> int:
    """Idle session TTL in seconds from IG_SESSION_TTL_SECONDS (default 3600)."""
    try:
        return int(os.environ.get("IG_SESSION_TTL_SECONDS", "3600"))
    except (TypeError, ValueError):
        return 3600


def max_sessions() -> int:
    """Max retained in-memory sessions from IG_MAX_SESSIONS (default 10000)."""
    try:
        return int(os.environ.get("IG_MAX_SESSIONS", "10000"))
    except (TypeError, ValueError):
        return 10000

def make_risk_estimator(model: Any, tokenizer: Any, device: str) -> Callable[[str], Dict[str, float]]:
    """Wrap a ModernBertRiskClassifier as the text -> scores function the API uses."""

    def ml_risk_estimator(text: str) -> Dict[str, float]:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            if hasattr(model, "predict_proba"):
                proba_dict = model.predict_proba(**inputs, apply_temperature=True)
                ml_scores = {
                    attr: float(proba_dict[attr].item())
                    for attr in ["age", "location", "occupation", "education"]
                    if attr in proba_dict
                }
            else:
                logits = model(**inputs)
                if isinstance(logits, dict) and "logits" in logits:
                    logits = logits["logits"]
                if isinstance(logits, dict):
                    ml_scores = {attr: torch.sigmoid(logits[attr]).item() for attr in ["age", "location", "occupation", "education"]}
                else:
                    ml_scores = {attr: torch.sigmoid(logits[i]).item() for i, attr in enumerate(["age", "location", "occupation", "education"])}

        rule_scores = estimate_privacy_risk(text, risk_model_fn=None)
        scores: Dict[str, float] = {}
        for attr in ["age", "location", "occupation", "education"]:
            r = rule_scores.get(attr, 0.05)
            m = ml_scores.get(attr, 0.05)
            if r >= 0.75:
                scores[attr] = round(max(m, r), 4)
            elif r >= 0.45:
                scores[attr] = round(max(m, 0.45), 4)
            else:
                scores[attr] = round(m, 4)

        text_lower = text.lower()
        has_first_person = bool(re.search(
            r"\b(i|i'm|i am|my|me|myself|mine|we|our|ours|us|i've|i'd|i'll)\b", text_lower
        )) or bool(re.search(
            r"\b(?:living in|commute to|working as|graduated from|started my career|my job|my role|my co-op)\b", text_lower
        )) or bool(re.search(
            r"\bas\s+(?:a|an|the)\s+(?!(?:result|rule|whole|matter|consequence|general|example|part|way|means)\b)[a-z0-9-]+",
            text_lower,
        ))

        if not has_first_person:
            # Third-party or factual context: discount demographic inferential scores unless explicit disclosure
            for attr in ["age", "location", "occupation", "education"]:
                if rule_scores.get(attr, 0.05) < 0.75:
                    scores[attr] = min(scores[attr], 0.20)
        else:
            # Floor boost for direct age/birthdate disclosures (e.g., "58 years old", "Born 1991", "born on March 14, 1998")
            if re.search(
                r"\b(?:\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)|born\s+(?:on\s+)?(?:[a-z]+\s+\d{1,2},?\s+(?:19|20)\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:19|20)\d{2}))\b",
                text_lower,
            ):
                if not re.search(r"\b(?:patient|child|son|daughter|infant|toddler)\b", text_lower):
                    scores["age"] = max(scores["age"], 0.75)
            # Boost signature indirect cues sparse in SynthPAI
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

            # k-Anonymity single-attribute calibration: when age and location are neutralized (< 0.35)
            # and no explicit education disclosure is present, an isolated broad professional role
            # carries moderate rather than high re-identification risk.
            if (
                scores.get("occupation", 0.0) >= 0.45
                and scores.get("age", 0.0) < 0.35
                and scores.get("location", 0.0) < 0.35
                and rule_scores.get("education", 0.05) < 0.45
                and scores.get("education", 0.0) < 0.55
                and rule_scores.get("occupation", 0.05) < 0.75
            ):
                scores["occupation"] = round(min(scores["occupation"], 0.42), 4)

        scores["overall"] = max(scores.values()) if scores else 0.0
        return scores

    return ml_risk_estimator


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_manifest_pins() -> Dict[str, Dict[str, Optional[str]]]:
    """Read authoritative Hugging Face Hub repo IDs and revision SHAs from artifacts/model_manifest.json."""
    pins: Dict[str, Dict[str, Optional[str]]] = {
        "risk": {
            "repo_id": os.getenv("RISK_MODEL_ID", "AT707/inferenceguard-modernbert-risk"),
            "revision": "4cecb3c90e84112ce5a5e3ed173060a1ab9609a7",
            "base_model": "answerdotai/ModernBERT-base",
        },
        "rewriter": {
            "repo_id": os.getenv("REWRITER_MODEL_ID", "AT707/inferenceguard-rewriter-v1"),
            "revision": "a84ca5623236ebdc325128e44baa25ec5494d592",
            "base_model": DEFAULT_BASE_MODEL,
        },
    }
    manifest_p = REPO_ROOT / "artifacts" / "model_manifest.json"
    if manifest_p.exists():
        try:
            data = json.loads(manifest_p.read_text(encoding="utf-8"))
            for entry in data.get("models", []):
                role = entry.get("role")
                if role in ("risk_modernbert_web", "risk_modernbert") and not os.getenv("RISK_MODEL_ID"):
                    pins["risk"]["repo_id"] = entry.get("repo_id") or pins["risk"]["repo_id"]
                    pins["risk"]["revision"] = entry.get("revision") or pins["risk"]["revision"]
                    pins["risk"]["base_model"] = entry.get("base_model") or pins["risk"]["base_model"]
                elif role == "rewriter" and not os.getenv("REWRITER_MODEL_ID"):
                    pins["rewriter"]["repo_id"] = entry.get("repo_id") or pins["rewriter"]["repo_id"]
                    pins["rewriter"]["revision"] = entry.get("revision") or pins["rewriter"]["revision"]
                    pins["rewriter"]["base_model"] = entry.get("base_model") or pins["rewriter"]["base_model"]
        except Exception as exc:
            logger.debug("Could not parse model_manifest.json: %s", exc)
    return pins


MODEL_PINS = _load_manifest_pins()
_risk_model_source: str = "fallback_rule_based"
_risk_model_id_loaded: Optional[str] = None
_risk_model_rev_loaded: Optional[str] = None
_rewriter_id_loaded: Optional[str] = None
_rewriter_rev_loaded: Optional[str] = None


def find_adapter_path() -> Optional[Path]:
    """Find the rewriter adapter directory across common execution contexts."""
    env_path = os.environ.get("REWRITER_ADAPTER_PATH")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        REPO_ROOT / "artifacts" / "rewriter_qlora",
        Path("artifacts/rewriter_qlora"),
        REPO_ROOT / "artifacts" / "rewriter_qlora_1.5b",
        Path("artifacts/rewriter_qlora_1.5b"),
        REPO_ROOT / "artifacts" / "rewriter_qlora_0.5b",
        Path("artifacts/rewriter_qlora_0.5b"),
        Path("/content/drive/MyDrive/InferenceGuard/rewriter_0.5b"),
        Path("/content/drive/MyDrive/Inference-Guard/artifacts/rewriter_qlora"),
        Path("/content/drive/MyDrive/Inference-Guard/artifacts/rewriter_qlora_0.5b"),
        Path("/content/Inference-Guard/artifacts/rewriter_qlora"),
        Path("/content/Inference-Guard/artifacts/rewriter_qlora_0.5b"),
        Path("/content/artifacts/rewriter_qlora"),
        Path("/content/artifacts/rewriter_qlora_0.5b"),
    ])
    for p in candidates:
        if p.exists() and (p / "adapter_config.json").exists():
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


# Load trained ModernBERT (Hugging Face Hub first unless IG_PREFER_LOCAL_ARTIFACTS=1)
risk_model = None
cue_attributor = None
_risk_init_lock = threading.Lock()


def _init_risk_model() -> Optional[Callable[[str], Dict[str, float]]]:
    global risk_model, cue_attributor, _risk_model_source, _risk_model_id_loaded, _risk_model_rev_loaded
    if risk_model is not None:
        return risk_model

    with _risk_init_lock:
        if risk_model is not None:
            return risk_model

        from transformers import AutoTokenizer
        from src.risk_model.attribution import SyntaxAlignedCueAttributor
        from src.risk_model.calibrate import AttributeTemperatureScaler
        from src.risk_model.model import load_risk_classifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        prefer_local = os.environ.get("IG_PREFER_LOCAL_ARTIFACTS", "").strip().lower() in {"1", "true", "yes"}
        local_path = find_risk_model_path()
        hub_id = MODEL_PINS["risk"]["repo_id"] or "AT707/inferenceguard-modernbert-risk"
        hub_rev = MODEL_PINS["risk"]["revision"]
        base_tok_id = MODEL_PINS["risk"]["base_model"] or "answerdotai/ModernBERT-base"

        attempts: List[Tuple[str, str, Optional[str]]] = []
        if prefer_local and local_path is not None:
            attempts.append(("local_artifact", str(local_path), None))
            attempts.append(("huggingface_hub", hub_id, hub_rev))
        else:
            attempts.append(("huggingface_hub", hub_id, hub_rev))
            if local_path is not None:
                attempts.append(("local_artifact", str(local_path), None))

        for src_kind, target, rev in attempts:
            try:
                logger.info("Loading ModernBERT risk classifier (%s) from '%s' (rev=%s)...", src_kind, target, rev)
                if src_kind == "local_artifact":
                    model = ModernBertRiskClassifier.from_pretrained(target, device=device).to(device)
                else:
                    model = load_risk_classifier(target, device=device)
                    if hasattr(model, "to"):
                        model = model.to(device)
                model.eval()
                try:
                    tokenizer = AutoTokenizer.from_pretrained(target)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(base_tok_id)

                risk_model = make_risk_estimator(model, tokenizer, device)
                cue_attributor = SyntaxAlignedCueAttributor(
                    model=model,
                    tokenizer=tokenizer,
                    scaler=AttributeTemperatureScaler(),
                    ig_steps=8,
                    device=device,
                )
                _risk_model_source = src_kind
                _risk_model_id_loaded = target
                _risk_model_rev_loaded = rev
                logger.info("ModernBERT loaded successfully from %s (%s)!", src_kind, target)
                return risk_model
            except Exception as exc:
                logger.warning("Failed to load ModernBERT from %s ('%s'): %s", src_kind, target, exc)

        return None


_init_risk_model()


def get_risk_model() -> Optional[Callable[[str], Dict[str, float]]]:
    """Retrieve or dynamically load ModernBERT risk estimator."""
    if risk_model is not None:
        return risk_model
    return _init_risk_model()


# Global persistent services
MAX_HISTORY_TURNS = 25
conversation_tracker = ConversationTracker(risk_estimator_fn=risk_model)
protected_tracker = ConversationTracker(risk_estimator_fn=risk_model)
pseudonym_manager = DeterministicPseudonymManager(referents_only=True)
pseudonym_states: Dict[str, ConversationState] = {}
session_secret_hashes: Dict[str, str] = {}
session_internal_secrets: Dict[str, str] = {}
presidio_baseline = PresidioBaseline()
utility_evaluator = UtilityEvaluator(load_models=True, device="cpu")
qwen_rewriter = None
_qwen_next_retry_ts: float = 0.0
_qwen_init_lock = threading.Lock()


def get_qwen_rewriter():
    """Retrieve or dynamically initialize QwenRewriterInference (HF Hub first unless IG_PREFER_LOCAL_ARTIFACTS=1)."""
    global qwen_rewriter, _qwen_next_retry_ts, _rewriter_id_loaded, _rewriter_rev_loaded
    if qwen_rewriter is not None and getattr(qwen_rewriter, "is_adapter_loaded", False):
        return qwen_rewriter

    with _qwen_init_lock:
        if qwen_rewriter is not None and getattr(qwen_rewriter, "is_adapter_loaded", False):
            return qwen_rewriter

        now = time.time()
        if now < _qwen_next_retry_ts:
            return qwen_rewriter

        from src.rewriter import QwenRewriterInference
        from src.rewriter.inference import SMALL_STUDENT_MODEL

        prefer_local = os.environ.get("IG_PREFER_LOCAL_ARTIFACTS", "").strip().lower() in {"1", "true", "yes"}
        local_p = find_adapter_path()
        hub_id = MODEL_PINS["rewriter"]["repo_id"] or "AT707/inferenceguard-rewriter-v1"
        hub_rev = MODEL_PINS["rewriter"]["revision"]
        base_model = MODEL_PINS["rewriter"]["base_model"] or SMALL_STUDENT_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"

        attempts: List[Tuple[str, Optional[str]]] = []
        if prefer_local and local_p is not None:
            attempts.append((str(local_p), None))
            attempts.append((hub_id, hub_rev))
        else:
            attempts.append((hub_id, hub_rev))
            if local_p is not None:
                attempts.append((str(local_p), None))

        for target, rev in attempts:
            try:
                logger.info("Loading Qwen Rewriter from '%s' (rev=%s, base=%s)...", target, rev, base_model)
                candidate = QwenRewriterInference(
                    model_name_or_path=base_model,
                    adapter_path=target,
                    adapter_revision=rev,
                    device=device,
                    load_in_4bit=torch.cuda.is_available(),
                )
                if getattr(candidate, "is_adapter_loaded", False):
                    qwen_rewriter = candidate
                    _rewriter_id_loaded = target
                    _rewriter_rev_loaded = rev
                    logger.info("Qwen Rewriter loaded from '%s' (rev=%s)!", target, rev)
                    return qwen_rewriter
            except Exception as exc:
                logger.warning("Loading Qwen Rewriter from '%s' failed: %s", target, exc)

        _qwen_next_retry_ts = now + 300.0
        return qwen_rewriter


_ALIAS_TOKEN_PATTERN = re.compile(
    r"^(?:PERSON|EMAIL|PHONE|ADDRESS|HOSPITAL|UNIVERSITY|ORG|LOCATION)_\d+$"
)


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
                span_txt = text[res.start:res.end]
                if is_non_identifying_presidio_span(
                    res.entity_type, span_txt, full_text=text, start=res.start
                ):
                    continue
                if _ALIAS_TOKEN_PATTERN.match(span_txt.strip()):
                    continue
                spans.append(
                    PiiSpan(
                        entity_type=res.entity_type,
                        start=res.start,
                        end=res.end,
                        score=float(res.score),
                    )
                )
            # Supplement Presidio with high-precision phone regex so synthetic 555 numbers are never missed
            for m in re.finditer(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)", text):
                if not any(s.start < m.end() and m.start() < s.end for s in spans):
                    spans.append(PiiSpan(entity_type="PHONE_NUMBER", start=m.start(), end=m.end(), score=0.95))
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
    # and duration phrases tagged as DATE_TIME (e.g. "seven years", "20-minute")
    filtered_spans: List[PiiSpan] = []
    for s in spans:
        span_text = text[s.start:s.end].strip()
        if s.entity_type == "PERSON" and span_text.upper() in {"ED", "ER", "ICU", "OR", "IT", "HR", "PR", "AI", "ML", "RN", "MD"}:
            continue
        if s.entity_type == "DATE_TIME" and is_duration_phrase(span_text):
            continue
        filtered_spans.append(s)

    filtered_spans.sort(key=lambda s: (s.start, s.end))
    return filtered_spans


PRIVACY_MODE_THRESHOLDS: Dict[str, float] = {
    "personal_balanced": PROVISIONAL_MAX_RISK_THRESHOLD,
    "personal_strict": 0.18,
    "business_support": 0.25,
}


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text message to analyze")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    privacy_mode: str = Field(
        default="personal_balanced",
        description="Privacy mode: personal_balanced, personal_strict, or business_support",
    )


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
    pseudonymized_text: Optional[str] = None
    pseudonyms: Dict[str, str] = Field(default_factory=dict)
    utility_metrics: Dict[str, Any]
    turn_record: Optional[Dict[str, Any]] = None
    protected_turn_record: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
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
    elevated_attrs = [
        attr for attr, score in turn_rec.turn_scores.items()
        if score >= PROVISIONAL_CUE_IMPORTANCE_FLOOR
    ]
    if elevated_attrs:
        # Fast-path bank cue attribution (<1 ms) before falling back to 8-step Integrated Gradients
        try:
            from src.rewriter.persona_banks import default_matcher
            bank_matches = default_matcher().scan(text)
            for m in bank_matches:
                attr_name = m.attribute if m.attribute in {"age", "location", "occupation", "education"} else None
                if attr_name and attr_name in elevated_attrs:
                    cues_list.append(
                        Cue(
                            span=m.surface,
                            attribute=attr_name,
                            importance=round(float(turn_rec.turn_scores.get(attr_name, 0.5)), 2),
                            start=m.start,
                            end=m.end,
                        )
                    )
        except Exception:
            pass

    if not cues_list and cue_attributor is not None and elevated_attrs:
        try:
            from src.contracts.schemas import AttributeName
            attr_enums = [AttributeName(a) for a in elevated_attrs if a in {"age", "location", "occupation", "education"}]
            if attr_enums:
                ig_cues = cue_attributor.extract_cues(text, attr_enums)
                for c in ig_cues:
                    cues_list.append(
                        Cue(
                            span=c.span_text,
                            attribute=c.attribute.value if hasattr(c.attribute, "value") else str(c.attribute),
                            importance=round(float(c.importance), 2),
                            start=c.start,
                            end=c.end,
                        )
                    )
        except Exception as exc:
            logger.debug("Cue attribution failed (%s); falling back to attribute-level cues.", exc)

    if not cues_list:
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
    )) or bool(re.search(
        r"\bas\s+(?:a|an|the)\s+(?!(?:result|rule|whole|matter|consequence|general|example|part|way|means)\b)[a-z0-9-]+",
        text_lower,
    ))
    return not has_self_ref


last_debug_trace: Dict[str, Any] = {}


_POLICY_QUERY_RE = re.compile(
    r"\b(?:policy|policies|law|laws|statute|tax|taxes|return|returns|refund|warranty|regulation|eligible|eligibility|consumer|shipping\s+zone|delivery\s+window)\b",
    re.IGNORECASE,
)
_JURISDICTION_RE = re.compile(
    r"\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|"
    r"Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
    r"Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|"
    r"New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
    r"Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|"
    r"Washington|West\s+Virginia|Wisconsin|Wyoming|Canada|UK|EU|European\s+Union)\b"
)


def _detect_policy_jurisdictions(text: str) -> List[str]:
    """Return macro-jurisdictions (states/countries) required to answer a policy/legal query."""
    if not _POLICY_QUERY_RE.search(text):
        return []
    return list(dict.fromkeys(_JURISDICTION_RE.findall(text)))


def _preserves_alias_tokens(candidate: str, required_tokens: Sequence[str]) -> bool:
    """Return True when every session alias token (e.g., PERSON_1) is preserved in candidate."""
    if not required_tokens:
        return True
    return all(tok in candidate for tok in required_tokens)


def _extract_task_core(text: str) -> str:
    """Extract the actionable request/question core so persona excision is not penalized."""
    s = text.strip()
    if not s:
        return s
    # Split into sentences and keep question/request sentences
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    request_sents = [
        sent for sent in sentences
        if "?" in sent
        or re.search(r"\b(?:can you|could you|should i|how can|which makes|help me|draft|calculate|compare|double-check|please)\b", sent, re.IGNORECASE)
    ]
    if request_sents:
        core = " ".join(request_sents)
        # If single long sentence has an em-dash or comma separating backstory from question, extract from the clause
        if len(sentences) == 1:
            for sep in ("—", " -- ", " - "):
                if sep in core:
                    tail = core.split(sep, 1)[1].strip()
                    if len(tail.split()) >= 5:
                        return tail
            m_comma = re.search(
                r",\s+((?:I\s+need\s+to|comparing|how|what|which|when|where|can|could|should|would|please)\b.+)$",
                core,
                re.IGNORECASE,
            )
            if m_comma and len(m_comma.group(1).split()) >= 5:
                return m_comma.group(1).strip()
        return core
    return s


def _compute_effective_similarity(original: str, candidate: str) -> float:
    """Max of full-text similarity and task-core similarity (when core is shorter than original)."""
    full_sim = float(utility_evaluator.compute_similarity(original, candidate))
    core = _extract_task_core(original)
    if core and len(core.split()) < len(original.split()) and len(candidate.split()) >= max(5, int(0.45 * len(core.split()))):
        core_sim = float(utility_evaluator.compute_similarity(core, candidate))
        return max(full_sim, 0.95 * core_sim)
    return full_sim


def _retains_critical_numbers(original: str, candidate: str) -> bool:
    """True when task-critical numbers and support/legal identifiers in original are preserved.

    Excludes alias token indices (PERSON_1, ADDRESS_1) and author-biographical numbers (author age,
    career tenure, personal workweek hours) while protecting third-party ages
    ('4-year-old patient'), task quantities ('$1,200', '16 kg', '55-minute'),
    support IDs ('#88412', 'TKT-4402', 'INV-901'), percentages ('15%'),
    ratios ('1:8'), and statutory citations ('19-9-3', '3553').
    """
    orig_clean, _ = scrub_hard_pii(original)
    orig_clean = re.sub(r"\b(?:PERSON|EMAIL|PHONE|ADDRESS|HOSPITAL|UNIVERSITY|ORG|LOCATION)_\d+\b", "", orig_clean)
    orig_clean = orig_clean.lower()
    _tp_followers = r"(?!\s+(?:patient|client|infant|child|toddler|boy|girl|man|woman|male|female|son|daughter|student|applicant)\b)"
    orig_clean = re.sub(rf"\b\d{{1,2}}[-\s]*(?:years?[-\s]*old|yrs?[-\s]*old|yo|y/o|year-old)\b{_tp_followers}", "", orig_clean)
    orig_clean = re.sub(r"\b(?:born(?:\s+on)?|class of)\s+(?:[a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b", "", orig_clean)
    orig_clean = re.sub(r"\b(?:in my|in their|early|mid|late)\s+\d{2}s\b", "", orig_clean)
    orig_clean = re.sub(r"\b(?:spent|with|for|after)?\s*\d+(?:\.\d+)?\s*(?:\+\s*)?(?:years?|yrs?|months?)\s+(?:of\s+)?(?:tenure|experience|service|practice|as\b)", "", orig_clean)
    orig_clean = re.sub(r"\b(?:working|work)\s+\d+[-\s]*hour\s+weeks?\b", "", orig_clean)

    cand_clean = re.sub(r"\b(?:PERSON|EMAIL|PHONE|ADDRESS|HOSPITAL|UNIVERSITY|ORG|LOCATION)_\d+\b", "", candidate).lower()

    # Check structured support/legal/quantitative identifiers explicitly
    _structured_pat = r"(?:#\d{3,}|\b(?:tkt|inv|ord|case)-\d+\b|\b\d+(?:\.\d+)?%|\b\d+:\d+\b|\b\d+-\d+(?:-\d+)+\b)"
    orig_struct = set(re.findall(_structured_pat, orig_clean))
    if orig_struct:
        cand_struct = set(re.findall(_structured_pat, cand_clean))
        if orig_struct - cand_struct:
            return False

    orig_nums = {n.replace(",", "") for n in re.findall(r"\b\d+(?:[.,]\d+)?\b", orig_clean)}
    critical_nums = {n for n in orig_nums if float(n) not in (0, 1)}
    cand_nums = {n.replace(",", "") for n in re.findall(r"\b\d+(?:[.,]\d+)?\b", cand_clean)}
    return not bool(critical_nums - cand_nums)


def _select_relative_risk_candidate(
    original: str,
    candidates: List[str],
    before_overall: float,
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]],
    max_risk: float,
    required_tokens: Sequence[str] = (),
) -> Optional[Tuple[str, float, float]]:
    """Evaluate candidates for relative risk reduction when strict < max_risk is unmet."""
    best: Optional[Tuple[str, float, float, float]] = None
    for cand in candidates:
        if not cand or not rewrite_changed(original, cand):
            continue
        if check_degeneracy(original, cand)["degenerate"]:
            continue
        _, had_pii = scrub_hard_pii(cand)
        if had_pii:
            continue
        if not _preserves_alias_tokens(cand, required_tokens):
            continue
        if not _retains_critical_numbers(original, cand):
            continue

        after_scores = estimate_privacy_risk(cand, risk_model_fn=risk_model_fn)
        after_overall = float(after_scores.get("overall", 1.0))
        eff_sim = _compute_effective_similarity(original, cand)
        if eff_sim < PROVISIONAL_MIN_COSINE_THRESHOLD:
            continue

        reduces_meaningfully = (
            after_overall < max_risk
            or after_overall < (before_overall - 0.05)
            or (after_overall <= before_overall + 0.03 and before_overall > max_risk)
        )
        if not reduces_meaningfully:
            continue

        utility = evaluate_rewrite_utility(
            original,
            cand,
            embedder_fn=lambda a, b: eff_sim,
        )
        u_score = float(utility.get("utility_score", eff_sim))
        if best is None or (after_overall < best[2] - 0.02) or (abs(after_overall - best[2]) <= 0.02 and u_score > best[1]):
            best = (cand, u_score, after_overall, eff_sim)

    if best is None:
        return None
    return best[0], best[2], best[3]


def _evaluate_neural_candidate(
    original: str,
    cand: str,
    before_overall: float,
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]],
    max_risk: float,
    required_tokens: Sequence[str] = (),
) -> Optional[Tuple[str, float, float]]:
    """Score a single neural candidate using task-core similarity and relative/strict risk."""
    if not cand or not rewrite_changed(original, cand):
        return None
    if check_degeneracy(original, cand)["degenerate"]:
        return None
    _, had_pii = scrub_hard_pii(cand)
    if had_pii:
        return None
    if not _preserves_alias_tokens(cand, required_tokens):
        return None
    if not _retains_critical_numbers(original, cand):
        return None

    after_scores = estimate_privacy_risk(cand, risk_model_fn=risk_model_fn)
    after_overall = float(after_scores.get("overall", 1.0))
    eff_sim = _compute_effective_similarity(original, cand)
    if eff_sim < PROVISIONAL_MIN_COSINE_THRESHOLD:
        return None
    if (
        after_overall < max_risk
        or after_overall < (before_overall - 0.05)
        or (after_overall <= before_overall and before_overall > max_risk)
    ):
        return cand, after_overall, eff_sim
    return None


def _strip_control_tokens(text: str) -> str:
    """Strip ChatML/reasoning control tokens so prompt-injected tags never leak."""
    cleaned = re.sub(r"<\|(?:im_start|im_end|endoftext|thought)\|>(?:assistant|user|system)?", " ", text)
    cleaned = re.sub(r"</?(?:think|thought)>", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def select_rewrite(
    text: str,
    presidio_text: str,
    risk_model_fn: Optional[Callable[[str], Dict[str, float]]],
    max_risk: float,
    attributed_spans: Optional[Sequence[str]] = None,
    required_tokens: Sequence[str] = (),
) -> Tuple[str, str]:
    """Pick the served rewrite and name the engine that produced it.

    Order: Qwen adapter (Pass 1 greedy -> Pass 2 gradient-cue guided if needed)
    -> heuristic candidates -> Presidio/aliased fallback.
    """
    global last_debug_trace
    before_scores = estimate_privacy_risk(text, risk_model_fn=risk_model_fn)
    before_overall = float(before_scores.get("overall", 0.0))

    trace: Dict[str, Any] = {
        "timestamp": time.time(),
        "input_text": text,
        "before_overall": before_overall,
        "attributed_spans": list(attributed_spans or []),
        "required_tokens": list(required_tokens),
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
        neural_candidates: List[Tuple[str, str]] = []
        try:
            qwen_out = rewriter.rewrite(text)
            is_fb = hasattr(rewriter, "_heuristic_fallback") and qwen_out == rewriter._heuristic_fallback(text)
            neural_candidates.append((qwen_out, ENGINE_HEURISTIC if is_fb else ENGINE_QWEN))
            trace["qwen_is_fallback"] = is_fb
            trace["qwen_output_raw"] = qwen_out
            trace["qwen_clean_output"] = qwen_out.strip() if qwen_out else ""
            trace["qwen_changed"] = rewrite_changed(text, trace["qwen_clean_output"])
        except Exception as exc:
            logger.warning("Qwen rewriter generation failed: %s", exc)
            trace["qwen_error"] = str(exc)

        if not trace["qwen_error"] and hasattr(rewriter, "last_error") and rewriter.last_error:
            trace["qwen_error"] = rewriter.last_error

        # Check Pass 1; if it failed acceptance or still has >= max_risk, run residual-cue attribution on Pass 1's output
        best_neural: Optional[Tuple[str, float, float, str]] = None
        for cand_raw, eng in neural_candidates:
            cand = cand_raw.strip() if cand_raw else ""
            ev = _evaluate_neural_candidate(text, cand, before_overall, risk_model_fn, max_risk, required_tokens)
            if ev is not None:
                best_neural = (ev[0], ev[1], ev[2], eng)
                break

        residual_spans: List[str] = []
        if best_neural is not None:
            try:
                from src.rewriter.persona_banks import default_matcher
                residual_spans.extend(m.surface for m in default_matcher().scan(best_neural[0]))
            except Exception:
                pass
            if best_neural[1] >= max_risk and cue_attributor is not None:
                try:
                    from src.contracts.schemas import AttributeName
                    cand_scores = estimate_privacy_risk(best_neural[0], risk_model_fn=risk_model_fn)
                    elevated = [
                        AttributeName(a) for a in ("age", "location", "occupation", "education")
                        if cand_scores.get(a, 0.0) >= max_risk
                    ]
                    if elevated:
                        ig_res = cue_attributor.extract_cues(best_neural[0], elevated)
                        residual_spans.extend(c.span_text for c in ig_res if c.span_text and len(c.span_text) >= 3)
                except Exception as exc:
                    logger.debug("Residual-cue attribution on Pass 1 failed: %s", exc)

        stubborn_spans = list(dict.fromkeys(
            residual_spans + [
                sp for sp in (attributed_spans or [])
                if sp and len(sp) >= 3 and not sp.startswith("[")
                and (best_neural is None or sp.lower() in best_neural[0].lower())
            ]
        ))
        if (best_neural is None or stubborn_spans) and stubborn_spans:
            try:
                pass2_input = best_neural[0] if best_neural is not None else text
                guidance = (
                    "Ensure these remaining personal or contextual details about the author are generalized to "
                    "broad, non-identifying terms while keeping all questions, role context, numbers, and requests intact: "
                    + ", ".join(stubborn_spans[:5])
                )
                qwen_p2 = rewriter.rewrite(pass2_input, extra_instruction=guidance)
                is_fb2 = hasattr(rewriter, "_heuristic_fallback") and qwen_p2 == rewriter._heuristic_fallback(pass2_input)
                if not is_fb2 and qwen_p2:
                    ev2 = _evaluate_neural_candidate(
                        text, qwen_p2.strip(), before_overall, risk_model_fn, max_risk, required_tokens
                    )
                    if ev2 is not None and (
                        best_neural is None
                        or ev2[1] < best_neural[1] - 0.02
                        or sum(1 for sp in stubborn_spans if sp.lower() in ev2[0].lower())
                        < sum(1 for sp in stubborn_spans if sp.lower() in best_neural[0].lower())
                    ):
                        best_neural = (ev2[0], ev2[1], ev2[2], ENGINE_QWEN)
                        trace["qwen_clean_output"] = ev2[0]
                        trace["qwen_changed"] = True
            except Exception as exc:
                logger.debug("Qwen Pass 2 cue-guided retry failed: %s", exc)

        # Deterministic polish if Qwen left any residual capitalized location/transit cues (e.g. Back Bay, Green Line)
        if best_neural is not None:
            try:
                from src.rewriter.persona_banks import PersonaMatcher
                loc_matcher = PersonaMatcher(banks=("cities", "local_references"))
                proper_matches = [
                    m for m in loc_matcher.scan(best_neural[0])
                    if any(w[:1].isupper() for w in m.surface.split())
                ]
                if proper_matches:
                    polished = best_neural[0]
                    for m in reversed(proper_matches):
                        rep = re.sub(r"^(?:a|an)\s+", "", m.generalizations[0], flags=re.IGNORECASE) if m.generalizations else "the area"
                        polished = polished[:m.start] + rep + polished[m.end:]
                    polished = re.sub(r"\bpublic\s+transportation\s+commute\b", "transit commute", polished, flags=re.IGNORECASE)
                    polished = re.sub(r"\s{2,}", " ", polished).strip()
                    if (
                        polished
                        and not check_degeneracy(text, polished)["degenerate"]
                        and _preserves_alias_tokens(polished, required_tokens)
                        and _retains_critical_numbers(text, polished)
                        and utility_evaluator.compute_similarity(best_neural[0], polished) >= 0.80
                    ):
                        pol_scores = estimate_privacy_risk(polished, risk_model_fn=risk_model_fn)
                        pol_risk = float(pol_scores.get("overall", 1.0))
                        if pol_risk <= max(best_neural[1], before_overall) + 0.03:
                            best_neural = (polished, pol_risk, best_neural[2], best_neural[3])
            except Exception:
                pass

        if best_neural is not None:
            trace["selected_engine"] = best_neural[3]
            trace["selected_text"] = best_neural[0]
            trace["decision_reason"] = (
                f"Accepted neural rewrite ({best_neural[3]}: {before_overall:.3f} -> {best_neural[1]:.3f}, sim={best_neural[2]:.3f})."
            )
            last_debug_trace = trace
            return best_neural[0], best_neural[3]

    # Heuristic Candidate Evaluation (prefer natural generalizations over bracketed [AGE]/[LOCATION] tags)
    from src.rewriter.inference import QwenRewriterInference
    bank_cand = QwenRewriterInference._heuristic_fallback(text)
    raw_heur = [c for c in generate_candidate_rewrites_heuristic(text) if c != bank_cand] + [bank_cand]
    natural_heur = [c for c in raw_heur if not re.search(r"\[(?:AGE|LOCATION|OCCUPATION|EDUCATION)\]", c)]
    heuristic_cands = natural_heur if natural_heur else raw_heur
    if required_tokens:
        heuristic_cands = [c for c in heuristic_cands if _preserves_alias_tokens(c, required_tokens)]

    selected = pareto_rejection_sample(
        original=text,
        candidates=heuristic_cands,
        risk_model_fn=risk_model_fn,
        embedder_fn=_compute_effective_similarity,
        max_risk_threshold=max_risk,
        min_cosine_threshold=PROVISIONAL_MIN_COSINE_THRESHOLD,
    )
    if selected is not None:
        out_txt = _strip_control_tokens(selected[0])
        trace["selected_engine"] = ENGINE_HEURISTIC
        trace["selected_text"] = out_txt
        trace["decision_reason"] = "Heuristic candidate satisfied strict Pareto thresholds."
        last_debug_trace = trace
        return out_txt, ENGINE_HEURISTIC

    rel_heur = _select_relative_risk_candidate(
        original=text,
        candidates=heuristic_cands,
        before_overall=before_overall,
        risk_model_fn=risk_model_fn,
        max_risk=max_risk,
        required_tokens=required_tokens,
    )
    if rel_heur is not None:
        out_txt = _strip_control_tokens(rel_heur[0])
        trace["selected_engine"] = ENGINE_HEURISTIC
        trace["selected_text"] = out_txt
        trace["decision_reason"] = (
            f"Heuristic candidate accepted via relative risk reduction ({before_overall:.3f} -> {rel_heur[1]:.3f})."
        )
        last_debug_trace = trace
        return out_txt, ENGINE_HEURISTIC

    cleaned_presidio = re.sub(r"<LOCATION>,\s*[A-Z]{2}\b", "<LOCATION>", presidio_text) if "<LOCATION>" in presidio_text else presidio_text
    cleaned_presidio = _strip_control_tokens(cleaned_presidio)
    trace["selected_engine"] = ENGINE_PRESIDIO
    trace["selected_text"] = cleaned_presidio
    trace["decision_reason"] = "No candidate satisfied Pareto or relative risk threshold; served Presidio baseline."
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


def _env_truthy(name: str) -> bool:
    """True when env var `name` is set to a truthy value ('1', 'true', 'yes')."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


@app.get("/health")
def health_check() -> Any:
    """Health check reporting actual loaded state, not a static 'ok'.

    Reports whether the trained/hosted risk model and rewriter adapter loaded,
    so an orchestrator can tell a real deployment from one silently serving the
    rule-based risk estimate and heuristic rewriter (see
    docs/engineering/production-readiness.md section 7).

    Fields:
      - status: 'ok' when the trained risk model loaded, else 'degraded'
        (or 'unhealthy' in strict mode, see below).
      - risk_model_loaded: get_risk_model() is not None.
      - risk_model_source: 'model' when a trained/hosted checkpoint loaded,
        'fallback_rule_based' when get_risk_model() is None and the rule-based
        estimate is used instead.
      - rewriter_adapter_loaded: the Qwen adapter is loaded.
      - rewrite_engine_default: 'qwen_adapter' when the adapter loaded, else
        'heuristic'.

    Strict mode (opt-in): set IG_REQUIRE_MODEL to a truthy value ('1'/'true'/
    'yes') to make /health return HTTP 503 with status='unhealthy' when the
    real risk model did not load, so traffic is not routed to a server serving
    the untrained/rule-based fallback. With IG_REQUIRE_MODEL unset the default
    behavior is unchanged (always HTTP 200), so the Colab demo is unaffected.
    """
    rm = get_risk_model()
    rewriter = get_qwen_rewriter()
    risk_model_loaded = rm is not None
    rewriter_adapter_loaded = bool(rewriter and getattr(rewriter, "is_adapter_loaded", False))

    payload: Dict[str, Any] = {
        "status": "ok" if risk_model_loaded else "degraded",
        "version": "1.0.0",
        "risk_model_loaded": risk_model_loaded,
        "risk_model_source": "model" if risk_model_loaded else "fallback_rule_based",
        "risk_model_backend": _risk_model_source if risk_model_loaded else "fallback_rule_based",
        "risk_model_id": _risk_model_id_loaded or MODEL_PINS["risk"]["repo_id"],
        "risk_model_revision": _risk_model_rev_loaded,
        "rewriter_adapter_loaded": rewriter_adapter_loaded,
        "rewriter_model_id": _rewriter_id_loaded or MODEL_PINS["rewriter"]["repo_id"],
        "rewriter_model_revision": _rewriter_rev_loaded,
        "rewrite_engine_default": ENGINE_QWEN if rewriter_adapter_loaded else ENGINE_HEURISTIC,
    }

    if _env_truthy("IG_REQUIRE_MODEL") and not risk_model_loaded:
        payload["status"] = "unhealthy"
        payload["reason"] = (
            "IG_REQUIRE_MODEL is set but the trained risk model did not load; "
            "the server would serve the rule-based fallback."
        )
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=payload)

    return payload


@app.get("/ready")
def readiness_check() -> Any:
    """Fail-closed readiness probe requiring trained risk model and rewriter adapter."""
    rm = get_risk_model()
    rewriter = get_qwen_rewriter()
    risk_loaded = rm is not None
    adapter_loaded = bool(rewriter and getattr(rewriter, "is_adapter_loaded", False))
    ready = risk_loaded and adapter_loaded
    payload: Dict[str, Any] = {
        "ready": ready,
        "risk_model_loaded": risk_loaded,
        "risk_model_source": _risk_model_source if risk_loaded else "fallback_rule_based",
        "rewriter_adapter_loaded": adapter_loaded,
        "model_id": _risk_model_id_loaded or MODEL_PINS["risk"]["repo_id"],
        "risk_model_revision": _risk_model_rev_loaded,
        "rewriter_model_id": _rewriter_id_loaded or MODEL_PINS["rewriter"]["repo_id"],
        "rewriter_model_revision": _rewriter_rev_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
    }
    if not ready:
        from fastapi.responses import JSONResponse
        payload["error"] = "Required ML models (risk model or Qwen rewriter adapter) are not loaded."
        return JSONResponse(status_code=503, content=payload)
    return payload


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
        "qwen_adapter_path": str(_rewriter_id_loaded or adapter_p) if (_rewriter_id_loaded or adapter_p) else None,
        "rewriter_model_id": _rewriter_id_loaded or MODEL_PINS["rewriter"]["repo_id"],
        "rewriter_model_revision": _rewriter_rev_loaded,
        "risk_model_loaded": bool(rm is not None),
        "risk_model_source": _risk_model_source if rm is not None else "fallback_rule_based",
        "risk_model_id": _risk_model_id_loaded or MODEL_PINS["risk"]["repo_id"],
        "risk_model_revision": _risk_model_rev_loaded,
        "risk_model_path": str(_risk_model_id_loaded or risk_p) if (_risk_model_id_loaded or risk_p) else None,
        "cuda_available": cuda_avail,
        "device": "cuda" if cuda_avail else "cpu",
        "vram_allocated_mb": vram_mb,
    }


@app.get("/debug/last_rewrite")
def get_last_rewrite_debug() -> Dict[str, Any]:
    """Inspect trace and decisions of the most recent rewrite.

    The trace is global, not session-scoped, so without this gate any caller
    could read the previous caller's pseudonymized text, contextual details
    and candidate rewrites. Off unless IG_DEMO_MODE is set.
    """
    if not _demo_mode_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "debug_disabled",
                "message": (
                    "The rewrite trace is global rather than session-scoped, so it is "
                    "disabled unless IG_DEMO_MODE=true."
                ),
            },
        )
    return last_debug_trace or {"status": "no rewrites processed yet"}


def _verify_or_register_session_secret(session_id: str, provided_secret: Optional[str]) -> str:
    """Verify X-Session-Secret against an existing session or register a blind-index secret."""
    if session_id in session_secret_hashes:
        expected_hash = session_secret_hashes[session_id]
        if not provided_secret:
            raise HTTPException(status_code=403, detail="Missing X-Session-Secret for protected session.")
        provided_hash = hashlib.sha256(provided_secret.encode("utf-8")).hexdigest()
        if provided_hash != expected_hash:
            raise HTTPException(status_code=403, detail="Invalid X-Session-Secret for session.")
        return provided_secret
    if provided_secret:
        session_secret_hashes[session_id] = hashlib.sha256(provided_secret.encode("utf-8")).hexdigest()
        return provided_secret
    # Backward-compatible internal secret when caller omits X-Session-Secret
    return session_internal_secrets.setdefault(session_id, secrets.token_hex(32))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(
    request: AnalyzeRequest,
    x_session_secret: Optional[str] = Header(default=None, alias="X-Session-Secret"),
) -> AnalyzeResponse:
    """Analyze input text for inferential privacy risk and return sanitized rewrite."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    session_id = request.session_id or "default"

    # Reap idle/over-cap sessions on normal traffic so the in-memory tracker
    # cannot grow unbounded (no background thread; cheap dict scan). Safe when
    # the env vars are unset thanks to the sane defaults.
    evicted = conversation_tracker.evict_idle(
        ttl_seconds=session_ttl_seconds(),
        max_sessions=max_sessions(),
    )
    for ev_id in evicted:
        protected_tracker.delete_session(ev_id)
        pseudonym_states.pop(ev_id, None)
        session_secret_hashes.pop(ev_id, None)
        session_internal_secrets.pop(ev_id, None)

    active_secret = _verify_or_register_session_secret(session_id, x_session_secret)

    timings: Dict[str, Optional[float]] = {stage: None for stage in TIMING_STAGES}
    t_total = time.perf_counter()

    # 1. Explicit PII detection on input + Presidio baseline redaction + zero-plaintext HMAC pseudonyms
    t = time.perf_counter()
    input_spans, presidio_text = presidio_stage(text)
    p_state = pseudonym_states.setdefault(session_id, ConversationState(conversation_id=session_id))
    pseudonymized_text, turn_pii_spans, _ = pseudonym_manager.pseudonymize_turn(
        text, p_state, session_secret=active_secret, blind_index=True
    )
    # Return ONLY the current turn's transient aliases (never stored on server)
    pseudonyms_map = {s.original_surface: s.pseudonym_token for s in turn_pii_spans}
    required_tokens = [
        s.pseudonym_token for s in turn_pii_spans
        if not s.pseudonym_token.startswith("LOCATION_")
    ]
    if (request.privacy_mode or "").strip().lower() == "business_support":
        for juris in _detect_policy_jurisdictions(text):
            if juris not in required_tokens:
                required_tokens.append(juris)
    timings["presidio"] = _ms_since(t)

    # 2. Multi-turn tracking and inferential risk estimation & cues (scored on raw_text)
    t = time.perf_counter()
    turn_rec, before_report = risk_stage(text, session_id, input_spans)
    attributed_spans = [
        c.span for c in before_report.inferential.cues
        if c.span and not c.span.startswith("[")
    ]
    timings["risk_model"] = _ms_since(t)

    # 3. Unified Rewriting on pseudonymized_text (preserves PERSON_1/ADDRESS_1/EMAIL_1 while generalizing cues)
    privacy_mode = (request.privacy_mode or "personal_balanced").strip().lower()
    mode_max_risk = PRIVACY_MODE_THRESHOLDS.get(privacy_mode, PROVISIONAL_MAX_RISK_THRESHOLD)
    low_risk_for_mode = (
        before_report.inferential_band == Band.LOW
        and before_report.inferential.overall < mode_max_risk
    )
    if (low_risk_for_mode or _is_impersonal_query(text)) and before_report.explicit.hard_count == 0 and not turn_pii_spans:
        rewritten_text = text
        engine = ENGINE_PASSTHROUGH
        after_report = before_report
    elif low_risk_for_mode and turn_pii_spans:
        rewritten_text = pseudonymized_text
        engine = ENGINE_HEURISTIC
        t = time.perf_counter()
        after_report = rescore_stage(rewritten_text)
        timings["rescore"] = _ms_since(t)
    else:
        active_risk = get_risk_model()
        prior_hist = conversation_tracker.get_history(session_id)
        prior_cum = prior_hist[-2].overall_cumulative_risk if len(prior_hist) >= 2 else 0.0
        mosaic_tightened = prior_cum >= 0.60
        effective_max_risk = min(0.20, mode_max_risk) if mosaic_tightened else mode_max_risk
        t = time.perf_counter()
        rewritten_text, engine = select_rewrite(
            pseudonymized_text,
            presidio_text,
            risk_model_fn=active_risk,
            max_risk=effective_max_risk,
            attributed_spans=attributed_spans,
            required_tokens=required_tokens,
        )
        if isinstance(last_debug_trace, dict):
            last_debug_trace["mosaic_shield_tightened"] = mosaic_tightened
            last_debug_trace["privacy_mode"] = privacy_mode
            last_debug_trace["effective_max_risk"] = effective_max_risk
        # If no rewrite candidate passed and aliases were applied cleanly, prefer persistent aliases over generic <PERSON>
        if engine == ENGINE_PRESIDIO and turn_pii_spans and pseudonymized_text != text:
            _, pseudo_has_hard = scrub_hard_pii(pseudonymized_text)
            if not pseudo_has_hard:
                rewritten_text = pseudonymized_text
                engine = ENGINE_HEURISTIC
        timings["rewrite"] = _ms_since(t)

        t = time.perf_counter()
        after_report = rescore_stage(rewritten_text)
        timings["rescore"] = _ms_since(t)

    # 4. Utility metrics
    t = time.perf_counter()
    utility_metrics = utility_evaluator.compute_metrics(text, rewritten_text)
    timings["utility"] = _ms_since(t)

    # 5. Track protected cumulative conversation risk & build turn history
    active_rm_for_prot = get_risk_model()
    if active_rm_for_prot is not None:
        protected_tracker.risk_estimator_fn = active_rm_for_prot
    protected_turn_rec = protected_tracker.add_turn(raw_text=rewritten_text, session_id=session_id)

    raw_history = conversation_tracker.get_history(session_id)
    prot_history = protected_tracker.get_history(session_id)
    start_idx = max(0, len(raw_history) - MAX_HISTORY_TURNS)
    conversation_history: List[Dict[str, Any]] = []
    for idx in range(start_idx, len(raw_history)):
        raw_rec = raw_history[idx]
        prot_rec = prot_history[idx] if idx < len(prot_history) else None
        preview = (raw_rec.raw_text[:80] + "…") if len(raw_rec.raw_text) > 80 else raw_rec.raw_text
        conversation_history.append(
            {
                "turn_number": raw_rec.turn_number,
                "preview": preview,
                "turn_risk_before": raw_rec.overall_turn_risk,
                "turn_risk_after": prot_rec.overall_turn_risk if prot_rec else raw_rec.overall_turn_risk,
                "cumulative_risk_before": raw_rec.overall_cumulative_risk,
                "cumulative_risk_after": prot_rec.overall_cumulative_risk if prot_rec else raw_rec.overall_cumulative_risk,
                "cumulative_scores_before": raw_rec.cumulative_scores,
                "cumulative_scores_after": prot_rec.cumulative_scores if prot_rec else raw_rec.cumulative_scores,
                "leakage_delta_before": raw_rec.leakage_delta,
                "leakage_delta_after": prot_rec.leakage_delta if prot_rec else raw_rec.leakage_delta,
                "risk_band_before": raw_rec.risk_band,
                "risk_band_after": prot_rec.risk_band if prot_rec else raw_rec.risk_band,
            }
        )

    # 6. Comparison and Rule 2 veto enforcement
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
        pseudonymized_text=pseudonymized_text,
        pseudonyms=pseudonyms_map,
        utility_metrics=utility_metrics,
        turn_record=turn_rec.to_dict(),
        protected_turn_record=protected_turn_rec.to_dict(),
        conversation_history=conversation_history,
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


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    messages: List[ChatMessage] = Field(..., min_length=1)
    session_id: Optional[str] = Field(default="chat-default")
    privacy_mode: str = Field(
        default="personal_balanced",
        description="Privacy mode: personal_balanced, personal_strict, or business_support",
    )
    upstream_base_url: Optional[str] = Field(default=None)
    upstream_api_key: Optional[str] = Field(default=None)


def _find_latest_tokens(pattern: str, primary_text: str, all_messages: List[Dict[str, str]]) -> List[str]:
    found = list(dict.fromkeys(re.findall(pattern, primary_text)))
    if not found:
        for m in reversed(all_messages):
            hits = re.findall(pattern, m.get("content", ""))
            if hits:
                found = list(dict.fromkeys(hits))
                break
    return found


_DEMO_ORDER_DB: Dict[str, Dict[str, str]] = {
    "#88412": {
        "order_label": "Order #88412",
        "shipping_address": "42 Boylston St, Apt 4B",
        "phone": "555-867-5309",
        "email": "sarah.chen@gmail.com",
    },
    "TKT-4402": {
        "order_label": "TKT-4402",
        "shipping_address": "718 Oak Ave, Suite 200 in Austin, TX",
        "phone": "555-234-8891",
        "email": "priya.nair@outlook.com",
    },
    "INV-901": {
        "order_label": "INV-901",
        "shipping_address": "1205 Pine Road, Unit 9",
        "phone": "555-310-4420",
        "email": "priya.nair@outlook.com",
    },
    "#55190": {
        "order_label": "Order #55190",
        "shipping_address": "88 Maple Street, Apt 3C",
        "phone": "555-902-1184",
        "email": "marcus.webb@example.com",
    },
    "TKT-7731": {
        "order_label": "TKT-7731",
        "shipping_address": "310 Cedar Blvd in Denver, CO",
        "phone": "555-774-2290",
        "email": "elena.gomez@yahoo.com",
    },
}


def _demo_mode_enabled() -> bool:
    """Whether canned assistant replies and global debug traces are allowed.

    Off unless explicitly enabled. The proxy is OpenAI-compatible, so a caller
    cannot tell a simulated reply from a real one by its shape -- and the
    canned replies assert completed actions ("applied a one-time courtesy
    waiver", "your replacement shipment is scheduled for dispatch"). Served to
    a real user after an upstream outage, that is a false statement about
    their account, not a degraded answer.
    """
    return os.getenv("IG_DEMO_MODE", "false").strip().lower() in ("1", "true", "yes", "on")


def _generate_local_demo_reply(
    sanitized_user_msg: str,
    all_messages: List[Dict[str, str]],
    order_lookup_tokens: Optional[Dict[str, str]] = None,
) -> str:
    """Offline context-aware assistant for live presentations when no upstream API key is configured."""
    low = sanitized_user_msg.lower()
    persons = _find_latest_tokens(r"\bPERSON_\d+\b", sanitized_user_msg, all_messages)
    emails = _find_latest_tokens(r"\bEMAIL_\d+\b", sanitized_user_msg, all_messages)
    phones = _find_latest_tokens(r"\bPHONE_\d+\b", sanitized_user_msg, all_messages)
    addresses = _find_latest_tokens(r"\bADDRESS_\d+\b", sanitized_user_msg, all_messages)
    order_ids = _find_latest_tokens(r"(?:Order\s*#\d+|#\d{3,}|(?:TKT|INV|ORD)-\d+)", sanitized_user_msg, all_messages)
    primary_person = persons[0] if persons else None
    greeting_name = primary_person or "there"
    primary_email = emails[0] if emails else None
    primary_phone = phones[0] if phones else None
    primary_address = addresses[0] if addresses else None
    primary_order = order_ids[0] if order_ids else "your order"

    # B2B Customer Support: Zero-trust order address/phone lookup
    if any(k in low for k in ("which shipping address", "what shipping address", "which address", "what address", "which phone", "phone number did i use", "address on file", "phone on file", "have on file")):
        matched_addr = (order_lookup_tokens or {}).get("shipping_address") or primary_address or "your verified address on file"
        matched_phone = (order_lookup_tokens or {}).get("phone") or primary_phone or "your verified phone on file"
        other_addrs = [a for a in addresses if a != matched_addr]
        contrast = f" (rather than {', '.join(other_addrs)})" if other_addrs else ""
        return (
            f"Hi {greeting_name}, I checked {primary_order} via our blind-indexed order record: "
            f"it was placed using shipping address {matched_addr}{contrast} and contact phone {matched_phone}. "
            "Would you like me to update either field before dispatch?"
        )

    # B2B Customer Support: State / regional policy & consumer law inquiries
    if any(k in low for k in ("consumer law", "return policy", "warranty", "entitled to a full refund")):
        jurisdictions = _JURISDICTION_RE.findall(sanitized_user_msg)
        j_str = " / ".join(dict.fromkeys(jurisdictions)) if jurisdictions else "your state"
        return (
            f"Under our {j_str} return & consumer policy for {primary_order}, eligible items qualify for a full refund "
            "within the statutory 30-day window (including requests made after 14 days) with no restocking fee. "
            "Would you like me to initiate the return authorization now?"
        )

    # B2B Customer Support: Multi-party billing / ticket transfer
    if any(k in low for k in ("transfer billing", "transfer ownership", "from my colleague")):
        secondary_person = persons[1] if len(persons) > 1 else "your colleague"
        email_note = f" and copy {primary_email}" if primary_email else ""
        phone_note = f" (contact: {primary_phone})" if primary_phone else ""
        return (
            f"Hi {greeting_name}{phone_note}, I have initiated the billing transfer on {primary_order} "
            f"from {secondary_person} to {greeting_name} while preserving your loyalty discount. "
            f"I will send confirmation once the transfer finalizes{email_note}."
        )

    # B2B Customer Support AI Agent responses (zero-PII alias-preserving)
    if any(k in low for k in ("late fee", "waive", "refund", "billing", "invoice")):
        email_note = f" and emailed confirmation to {primary_email}" if primary_email else ""
        addr_note = f" (account address on file: {primary_address})" if primary_address else ""
        fee_match = re.search(r"\$\d+(?:\.\d{2})?", sanitized_user_msg)
        fee_str = f" {fee_match.group(0)}" if fee_match else ""
        return (
            f"Hi {greeting_name}, I have reviewed {primary_order}{addr_note} and applied a one-time courtesy waiver "
            f"for the{fee_str} late fee{email_note}. Your updated balance will reflect within 1 business day. "
            "Is there anything else I can help with on this order?"
        )
    if any(k in low for k in ("shipment", "shipping", "replacement", "delivery", "tracking", "receipt")):
        dest = primary_address or "your verified address on file"
        email_str = f" to {primary_email}" if primary_email else ""
        return (
            f"Confirmed, {greeting_name}: your replacement shipment for {primary_order} is scheduled for dispatch "
            f"to {dest}. I have also queued the itemized receipt for {greeting_name}{email_str}."
        )
    if any(k in low for k in ("loyalty discount", "upgrade my plan", "upgrade plan", "subscription")):
        disc_match = re.search(r"\$\d+(?:\.\d{2})?", sanitized_user_msg)
        disc_str = f" {disc_match.group(0)}" if disc_match else ""
        return (
            f"Yes, {greeting_name}—when you upgrade your plan next month, your{disc_str} loyalty discount "
            "automatically carries over to the new tier with no action needed on your account."
        )

    colleague_ref = primary_person or "your colleague"
    if any(k in low for k in ("draft", "email", "follow-up", "reply", "write a note", "feedback")):
        cc_line = f"\nCC: {', '.join(emails)}" if emails else ""
        return (
            f"Subject: Quick Follow-Up & Feedback Request{cc_line}\n\n"
            f"Hi {colleague_ref},\n\n"
            "I wanted to share a brief update and ask if you have time to review the materials "
            "ahead of our upcoming milestone. I have incorporated the latest project highlights "
            "and would appreciate any feedback you can share before the deadline.\n\n"
            "Best regards,"
        )
    if any(k in low for k in ("commute", "apartment", "rent", "financial", "$1,200", "$1,950")):
        return (
            "Comparing the two housing options: the closer option costs $750/month more ($9,000/year), "
            "while saving roughly 80 minutes per workday (~26–28 hours/month). That values your saved "
            "time at about $27–$29/hour post-tax—worth taking if your schedule is demanding or you value "
            "flexibility, whereas the transit option builds $9,000/year in savings."
        )
    if any(k in low for k in ("amoxicillin", "dosage", "mg/kg", "otitis")):
        return (
            "For standard high-dose pediatric amoxicillin in acute otitis media (80–90 mg/kg/day divided BID): "
            "for a 16 kg patient, total daily dose is 1,280–1,440 mg/day, or ~640–720 mg per dose every 12 hours "
            "(verify against institutional formulary and patient allergy history)."
        )
    if any(k in low for k in ("slide", "deck", "review", "grant", "board", "agenda", "presentation")):
        return (
            f"Here is a concise structure to share with {primary_person}:\n"
            "1. Executive Summary & Q4 Objectives (1 slide)\n"
            "2. Key Milestones & Research/Grant Impact Metrics (2 slides)\n"
            "3. Risks, Timeline & Decision Points for Review (1 slide)\n\n"
            f"Would you like me to draft a short follow-up note to {primary_person} with these bullet points?"
        )
    return (
        f"I received your sanitized message cleanly (referencing {primary_person if persons else 'your request'}). "
        "Here are three actionable next steps based on your context:\n"
        "1. Clarify the core decision criteria and timeline.\n"
        "2. Summarize measurable impact without exposing sensitive background details.\n"
        "3. Prepare a short follow-up summary to confirm alignment."
    )


@app.post("/v1/chat/completions")
def chat_completions_proxy(
    request: ChatCompletionRequest,
    x_session_secret: Optional[str] = Header(default=None, alias="X-Session-Secret"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    x_privacy_mode: Optional[str] = Header(default=None, alias="X-Privacy-Mode"),
    x_restore_mode: Optional[str] = Header(default=None, alias="X-InferenceGuard-Restore"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    """OpenAI-compatible bidirectional privacy proxy (/v1/chat/completions).

    Supports unmodified OpenAI SDK / LangChain clients via HTTP headers:
      - X-Privacy-Mode: personal_balanced | personal_strict | business_support
      - X-InferenceGuard-Restore: client (default) | gateway (auto-restores customer reply)
      - X-Session-ID / X-Session-Secret: zero-plaintext HMAC session binding
    """
    user_indices = [i for i, m in enumerate(request.messages) if m.role == "user" and m.content.strip()]
    if not user_indices:
        raise HTTPException(status_code=400, detail="At least one non-empty user message is required.")

    last_user_idx = user_indices[-1]
    raw_user_text = request.messages[last_user_idx].content.strip()
    hdr_session_id = x_session_id if isinstance(x_session_id, str) and x_session_id.strip() else None
    hdr_privacy_mode = x_privacy_mode if isinstance(x_privacy_mode, str) and x_privacy_mode.strip() else None
    hdr_restore_mode = x_restore_mode if isinstance(x_restore_mode, str) and x_restore_mode.strip() else None
    session_id = hdr_session_id or request.session_id or "chat-default"
    effective_privacy_mode = (hdr_privacy_mode or request.privacy_mode or "personal_balanced").strip().lower()
    restore_mode = (hdr_restore_mode or "client").strip().lower()

    # Run through InferenceGuard's unified /analyze pipeline (zero-plaintext HMAC + QLoRA rewrite)
    analyze_res = analyze_text(
        AnalyzeRequest(
            text=raw_user_text,
            session_id=session_id,
            user_id="proxy_user",
            privacy_mode=effective_privacy_mode,
        ),
        x_session_secret=x_session_secret,
    )

    sanitized_messages: List[Dict[str, str]] = []
    extra_pseudonyms: Dict[str, str] = {}
    unprotected_roles: set = set()
    p_state_ctx = pseudonym_states.get(session_id)
    active_sec_ctx = x_session_secret or session_internal_secrets.get(session_id)
    for idx, msg in enumerate(request.messages):
        if idx == last_user_idx:
            content = analyze_res.rewritten_text
        elif p_state_ctx is not None and active_sec_ctx:
            # Every other outbound message -- prior user turns, tool/RAG payloads,
            # and assistant and system content -- is pseudonymized too. Assistant
            # turns quote the user's details back ("your order ships to <address>")
            # and system prompts carry account context, so passing those through
            # unchanged sent to the upstream model exactly what the last turn was
            # rewritten to withhold.
            pseudo_ctx, ctx_spans, _ = pseudonym_manager.pseudonymize_turn(
                msg.content, p_state_ctx, session_secret=active_sec_ctx, blind_index=True
            )
            content = pseudo_ctx
            for sp in ctx_spans:
                extra_pseudonyms.setdefault(sp.original_surface, sp.pseudonym_token)
            unprotected_roles.discard(msg.role)
        else:
            # No pseudonym state or session secret: we cannot protect this
            # message. Record it rather than quietly forwarding the raw text --
            # the caller is told which roles went out unprotected.
            content = msg.content
            unprotected_roles.add(msg.role)
        sanitized_messages.append({"role": msg.role, "content": content})

    # Resolve optional upstream API key & base URL
    bearer_key = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_key = authorization.split(" ", 1)[1].strip()
    api_key = request.upstream_api_key or bearer_key or os.getenv("OPENAI_API_KEY")
    base_url = (request.upstream_base_url or os.getenv("IG_UPSTREAM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    assistant_reply = ""
    upstream_mode = "local_demo_assistant"
    if api_key:
        try:
            import urllib.request
            payload = json.dumps({
                "model": request.model,
                "messages": sanitized_messages,
                "temperature": 0.4,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                assistant_reply = body["choices"][0]["message"]["content"]
                upstream_mode = "live_upstream"
        except Exception as exc:
            logger.warning("Upstream LLM dispatch failed (%s); using local demo assistant.", exc)

    if not assistant_reply and not _demo_mode_enabled():
        # Production: say the upstream call failed. Do not substitute a canned
        # reply -- see _demo_mode_enabled.
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_unavailable",
                "message": (
                    "The upstream model could not be reached, and demo replies are "
                    "disabled. No assistant response was generated. Set IG_DEMO_MODE=true "
                    "only for presentations, never for a caller acting on the reply."
                ),
                "upstream_configured": bool(api_key),
                "session_id": session_id,
            },
        )

    if not assistant_reply:
        # Zero-trust blind-indexed order lookup for B2B support address/phone verification
        order_lookup_tokens: Dict[str, str] = {}
        merged_pseudonyms = {**extra_pseudonyms, **dict(analyze_res.pseudonyms)}
        if effective_privacy_mode == "business_support":
            p_state = pseudonym_states.get(session_id)
            active_sec = x_session_secret or session_internal_secrets.get(session_id)
            if p_state is not None and active_sec:
                for oid_key, rec in _DEMO_ORDER_DB.items():
                    if oid_key.lower() in analyze_res.rewritten_text.lower():
                        for field in ("shipping_address", "phone", "email"):
                            raw_val = rec.get(field)
                            if raw_val:
                                tok_val, spans, _ = pseudonym_manager.pseudonymize_turn(
                                    raw_val, p_state, session_secret=active_sec, blind_index=True
                                )
                                order_lookup_tokens[field] = tok_val
                                for sp in spans:
                                    merged_pseudonyms.setdefault(sp.original_surface, sp.pseudonym_token)
                        break
        assistant_reply = _generate_local_demo_reply(
            analyze_res.rewritten_text,
            sanitized_messages,
            order_lookup_tokens=order_lookup_tokens,
        )
    else:
        merged_pseudonyms = {**extra_pseudonyms, **dict(analyze_res.pseudonyms)}

    # Optional VPC gateway-side restoration for channels without a client vault (Zendesk/Intercom/SMS)
    wire_assistant_reply = assistant_reply
    served_assistant_reply = wire_assistant_reply
    if restore_mode == "gateway" and merged_pseudonyms:
        from src.conversation.entity_map import StreamAliasRestorer
        token_to_orig = {tok: orig for orig, tok in merged_pseudonyms.items()}
        restorer = StreamAliasRestorer(token_to_orig)
        served_assistant_reply = restorer.feed(wire_assistant_reply) + restorer.flush()

    last_hist = analyze_res.conversation_history[-1] if analyze_res.conversation_history else {}
    crm_safe_transcript = sanitized_messages + [{"role": "assistant", "content": wire_assistant_reply}]
    return {
        "id": f"chatcmpl-ig-{secrets.token_hex(6)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": served_assistant_reply},
                "finish_reason": "stop",
            }
        ],
        "inferenceguard": {
            "session_id": session_id,
            "privacy_mode": effective_privacy_mode,
            "restore_mode": restore_mode,
            "sanitized_user_message": analyze_res.rewritten_text,
            "sanitized_assistant_reply": wire_assistant_reply,
            "sanitized_messages_sent": sanitized_messages,
            "crm_safe_transcript": crm_safe_transcript,
            "pseudonyms": merged_pseudonyms,
            "risk_before": (analyze_res.comparison or {}).get("inferential_before"),
            "risk_after": (analyze_res.comparison or {}).get("inferential_after"),
            "cumulative_risk_before": last_hist.get("cumulative_risk_before"),
            "cumulative_risk_after": last_hist.get("cumulative_risk_after"),
            "rewrite_engine": analyze_res.rewrite_engine,
            "upstream_mode": upstream_mode,
            "simulated_reply": upstream_mode != "live_upstream",
            "unprotected_roles": sorted(unprotected_roles),
        },
    }


class SupportWebhookRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1, description="Helpdesk ticket or conversation ID")
    channel: str = Field(default="zendesk", description="Helpdesk channel: zendesk, intercom, salesforce, sms")
    customer_message: str = Field(..., min_length=1, description="Raw inbound customer message")
    crm_context: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional CRM/order record fields (shipping_address, phone, email, order_id) blind-indexed on the fly",
    )
    history: List[ChatMessage] = Field(default_factory=list, description="Optional prior sanitized conversation turns")
    model: str = Field(default="gpt-4o-mini")
    upstream_base_url: Optional[str] = Field(default=None)
    upstream_api_key: Optional[str] = Field(default=None)


@app.post("/v1/support/webhook")
def support_webhook_gateway(
    request: SupportWebhookRequest,
    x_session_secret: Optional[str] = Header(default=None, alias="X-Session-Secret"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Dict[str, Any]:
    """Drop-in B2B webhook middleware for Zendesk, Intercom, Salesforce, and Freshdesk.

    Returns both:
      - customer_facing_reply: natural response with customer PII restored for the chat/SMS channel
      - crm_safe_ticket_note / crm_safe_transcript: zero-PII anonymized log safe for CRM/Snowflake/Datadog
    """
    session_id = f"ticket-{request.ticket_id}"
    active_secret = _verify_or_register_session_secret(session_id, x_session_secret)
    p_state = pseudonym_states.setdefault(session_id, ConversationState(conversation_id=session_id))

    # Blind-index any caller-supplied CRM record fields into the session state BEFORE proxy execution
    crm_pseudonyms: Dict[str, str] = {}
    crm_tool_parts: List[str] = []
    for k, v in (request.crm_context or {}).items():
        if not v or not isinstance(v, str):
            continue
        tok_v, spans, _ = pseudonym_manager.pseudonymize_turn(
            v, p_state, session_secret=active_secret, blind_index=True
        )
        crm_tool_parts.append(f"{k}={tok_v}")
        for sp in spans:
            crm_pseudonyms.setdefault(sp.original_surface, sp.pseudonym_token)

    msgs: List[ChatMessage] = list(request.history)
    if crm_tool_parts:
        msgs.append(ChatMessage(role="tool", content="CRM record: " + ", ".join(crm_tool_parts)))
    msgs.append(ChatMessage(role="user", content=request.customer_message))

    proxy_resp = chat_completions_proxy(
        ChatCompletionRequest(
            model=request.model,
            messages=msgs,
            session_id=session_id,
            privacy_mode="business_support",
            upstream_base_url=request.upstream_base_url,
            upstream_api_key=request.upstream_api_key,
        ),
        x_session_secret=active_secret,
        x_session_id=session_id,
        x_privacy_mode="business_support",
        x_restore_mode="client",
        authorization=authorization,
    )
    ig = proxy_resp["inferenceguard"]
    all_pseudonyms = {**crm_pseudonyms, **ig.get("pseudonyms", {})}
    wire_reply = ig.get("sanitized_assistant_reply") or proxy_resp["choices"][0]["message"]["content"]

    from src.conversation.entity_map import StreamAliasRestorer
    restorer = StreamAliasRestorer({tok: orig for orig, tok in all_pseudonyms.items()})
    customer_reply = restorer.feed(wire_reply) + restorer.flush()

    sanitized_user = ig["sanitized_user_message"]
    jurisdictions = _detect_policy_jurisdictions(request.customer_message)
    crm_safe_note = (
        f"[{request.channel.upper()} ZERO-PII LOG · {request.ticket_id}]\n"
        f"Customer: {sanitized_user}\n"
        f"Agent: {wire_reply}"
    )

    return {
        "ticket_id": request.ticket_id,
        "channel": request.channel,
        "session_id": session_id,
        "customer_facing_reply": customer_reply,
        "sanitized_customer_message": sanitized_user,
        "sanitized_agent_reply": wire_reply,
        "crm_safe_ticket_note": crm_safe_note,
        "crm_safe_transcript": ig["crm_safe_transcript"],
        "privacy_audit": {
            "risk_before": ig["risk_before"],
            "risk_after": ig["risk_after"],
            "cumulative_risk_before": ig["cumulative_risk_before"],
            "cumulative_risk_after": ig["cumulative_risk_after"],
            "entities_aliased": len(all_pseudonyms),
            "jurisdictions_preserved": jurisdictions,
            "rewrite_engine": ig["rewrite_engine"],
            "upstream_mode": ig["upstream_mode"],
        },
    }


@app.delete("/session/{session_id}")
def delete_session(
    session_id: str,
    x_session_secret: Optional[str] = Header(default=None, alias="X-Session-Secret"),
) -> Dict[str, Any]:
    """Clear all tracked state for a conversation session.

    Removes the session's conversation/turn/risk history (and any per-session
    entity/pseudonym state the tracker holds) via ConversationTracker. The
    operation is idempotent: it always returns HTTP 200 whether or not the
    session existed, with ``cleared`` indicating whether state was present.
    """
    if session_id in session_secret_hashes:
        _verify_or_register_session_secret(session_id, x_session_secret)
    cleared = conversation_tracker.delete_session(session_id)
    protected_tracker.delete_session(session_id)
    pseudonym_states.pop(session_id, None)
    session_secret_hashes.pop(session_id, None)
    session_internal_secrets.pop(session_id, None)
    return {"session_id": session_id, "cleared": cleared}


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
                stdout=subprocess.DEVNULL,
                text=True,
            )
            # Poll up to 12 seconds to extract the public *.trycloudflare.com URL
            t_end = time.time() + 12.0
            while time.time() < t_end and proc.poll() is None:
                line = proc.stderr.readline() if proc.stderr else ""
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    url = match.group(0)
                    logger.info(f"cloudflared tunnel established: {url}")
                    return url
                if not line:
                    time.sleep(0.2)
            return f"http://localhost:{port} (cloudflared background)"
        except Exception as e:
            logger.warning(f"cloudflared failed: {e}")

    return f"http://localhost:{port}"


# Mount static files and define explicit root handlers to serve the minimalist web frontend
web_dir = Path(__file__).resolve().parent


@app.get("/", response_class=FileResponse)
def serve_root() -> FileResponse:
    """Serve the minimalist web UI index page."""
    index_file = web_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(str(index_file))


@app.get("/index.html", response_class=FileResponse)
def serve_index_html() -> FileResponse:
    """Explicit alias for web UI index page."""
    return serve_root()


if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
