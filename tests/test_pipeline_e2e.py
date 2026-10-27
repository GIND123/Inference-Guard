# tests/test_pipeline_e2e.py
"""Verification suite for Gate 6: End-to-End Latency, Interval Highlights, and Zero-PII Telemetry.

Verifies single-turn protection orchestration, character-offset HTML highlight
torture testing (overlapping and nested intervals), and zero-PII telemetry schema gating.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

pytest.importorskip(
    "fastapi",
    reason="needs the optional fastapi dependency; see tests/README.md",
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import app
from src.api.ui_helpers import (
    PrivacySafeTelemetryLogger,
    SafeTelemetryEvent,
    render_highlighted_html_from_offsets,
)
from src.contracts.schemas import (
    AttributeName,
    AttributedCue,
    ConversationState,
    ProtectionResponse,
    RewriteMode,
    RiskBand,
)
from src.conversation.entity_map import DeterministicPseudonymManager
from src.conversation.leakage import BayesianConversationLeakageTracker
from src.evaluation.utility_metrics import BidirectionalUtilityEvaluator
from src.pipeline.orchestrator import InferenceGuardPipeline
from src.rewriter.inference import QwenPrivacyRewriterService
from src.risk_model.attribution import SyntaxAlignedCueAttributor
from src.risk_model.calibrate import AttributeTemperatureScaler
from src.risk_model.predict import CalibratedRiskPredictor


@pytest.fixture
def test_pipeline():
    """Builds a fast, deterministic InferenceGuardPipeline for end-to-end integration testing."""
    scaler = AttributeTemperatureScaler()
    pseudonym_mgr = DeterministicPseudonymManager()
    utility_eval = BidirectionalUtilityEvaluator()
    rewriter = QwenPrivacyRewriterService()
    leakage_tracker = BayesianConversationLeakageTracker(scaler)

    class FastTestPredictor:
        def predict_with_cues(self, text, attributor=None, explicit_pii_counts=None):
            from src.contracts.schemas import AttributeRiskDistribution, RiskReport
            is_risky = any(w in text.lower() for w in ["green line", "seattle", "amazon"])
            score = 0.70 if is_risky else 0.10
            band = RiskBand.HIGH if is_risky else RiskBand.LOW
            attrs = {
                attr: AttributeRiskDistribution(
                    attribute=attr,
                    predicted_value="us_west" if is_risky and attr == AttributeName.LOCATION else "UNKNOWN",
                    calibrated_confidence=score,
                    top2_margin=0.3,
                    normalized_entropy=0.4,
                    inferability_risk_score=score,
                    risk_band=band,
                    class_probabilities={},
                )
                for attr in AttributeName
            }
            cues = []
            if is_risky:
                cues.append(
                    AttributedCue(
                        span_text="Green Line",
                        start=15,
                        end=25,
                        attribute=AttributeName.LOCATION,
                        importance=0.9,
                        deletion_risk_drop=0.3,
                    )
                )
            return RiskReport(
                attributes=attrs,
                overall_risk_score=score,
                overall_risk_band=band,
                primary_risk_attribute=AttributeName.LOCATION if is_risky else None,
                cues=cues,
                explicit_pii_counts=explicit_pii_counts or {},
            )

    attributor = SyntaxAlignedCueAttributor(
        model=None,
        tokenizer=None,
        scaler=scaler,
    )

    return InferenceGuardPipeline(
        pseudonym_manager=pseudonym_mgr,
        risk_estimator=FastTestPredictor(),
        cue_attributor=attributor,
        rewriter_service=rewriter,
        utility_evaluator=utility_eval,
        leakage_tracker=leakage_tracker,
    )


def test_e2e_single_turn_protection(test_pipeline):
    """Verify full 6-step protection turn: pseudonymization, risk before/after, utility, and state."""
    raw_message = "My email is alice@test.com and I take the Green Line to campus."
    res: ProtectionResponse = test_pipeline.process_turn(
        conversation_id="e2e_conv_1",
        message=raw_message,
        mode=RewriteMode.BALANCED_SFT,
        commit_to_session=True,
    )

    assert res.conversation_id == "e2e_conv_1"
    assert res.turn_id == 1
    assert "alice@test.com" not in res.protected_text
    assert "EMAIL_1" in res.protected_text
    assert res.utility.bidirectional_entailment >= 0.60
    assert res.conversation_leakage_after >= 0.0
    assert res.latency_ms > 0.0
    assert res.latency_ms < 5000.0  # Fast execution check


def test_character_offset_highlighter_overlap_torture():
    """Verify that overlapping, nested, and touching cues render without broken or duplicated HTML."""
    text = "Taking the Green Line to campus in Boston."

    cues = [
        # Span 1: "Green Line" (chars 11-21, importance 0.90)
        AttributedCue(
            span_text="Green Line",
            start=11,
            end=21,
            attribute=AttributeName.LOCATION,
            importance=0.90,
        ),
        # Span 2: Nested inside "Green" (chars 11-16, lower importance 0.50 -> must be dropped)
        AttributedCue(
            span_text="Green",
            start=11,
            end=16,
            attribute=AttributeName.LOCATION,
            importance=0.50,
        ),
        # Span 3: Overlapping "Line to" (chars 17-24, lower importance 0.40 -> must be dropped)
        AttributedCue(
            span_text="Line to",
            start=17,
            end=24,
            attribute=AttributeName.LOCATION,
            importance=0.40,
        ),
        # Span 4: Separate cue "Boston" (chars 35-41, importance 0.85 -> accepted)
        AttributedCue(
            span_text="Boston",
            start=35,
            end=41,
            attribute=AttributeName.LOCATION,
            importance=0.85,
        ),
    ]

    html_out = render_highlighted_html_from_offsets(text, cues)

    # Invariant 1: Higher importance span wins
    assert 'data-attribute="location" data-importance="0.90">Green Line' in html_out
    assert 'data-attribute="location" data-importance="0.85">Boston' in html_out
    # Invariant 2: No nested or malformed tags
    assert "<mark" in html_out
    assert html_out.count("<mark") == 2
    assert html_out.count("</mark>") == 2


def test_zero_pii_telemetry_schema_whitelist():
    """Verify that telemetry strictly forbids raw text or personal identifiers from disk logs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "telemetry.jsonl"
        logger = PrivacySafeTelemetryLogger(log_path=log_file)

        event = SafeTelemetryEvent(
            session_id="anon_sess_1",
            scenario_id="S-01",
            event_type="rewrite_accepted",
            rewrite_mode="balanced_sft",
            overall_risk_before=0.72,
            overall_risk_after=0.25,
            bidirectional_entailment=0.88,
            utility_warning_shown=False,
            latency_ms=120.5,
        )

        logger.log_event(event)
        assert log_file.exists()

        log_content = log_file.read_text(encoding="utf-8")
        assert "anon_sess_1" in log_content
        assert "rewrite_accepted" in log_content
        # Hard safety: raw text fields must not exist on SafeTelemetryEvent
        assert not hasattr(event, "raw_text")
        assert not hasattr(event, "prompt")
        assert not hasattr(event, "user_input")

        # Invalid event type rejected by validation
        with pytest.raises(ValidationError):
            SafeTelemetryEvent(
                session_id="anon_sess_2",
                scenario_id="S-01",
                event_type="UNAUTHORIZED_TYPE",
                rewrite_mode="balanced_sft",
                overall_risk_before=0.5,
                overall_risk_after=0.2,
                bidirectional_entailment=0.8,
                utility_warning_shown=False,
                latency_ms=50.0,
            )


def test_fastapi_rest_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr('src.api.app.telemetry_logger.log_path', tmp_path / 'telemetry.jsonl')
    """Verify that FastAPI /health, /protect, /highlight, and /telemetry respond cleanly."""
    client = TestClient(app)

    # Test /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # Test /protect
    r_protect = client.post(
        "/protect",
        json={"text": "Contact me at alice@example.com in Seattle.", "mode": "balanced_sft"},
    )
    assert r_protect.status_code == 200
    resp_data = r_protect.json()
    assert "protected_text" in resp_data
    assert "utility" in resp_data

    # Test /highlight
    r_hl = client.post(
        "/highlight",
        json={
            "text": "Living in Seattle.",
            "cues": [{"span_text": "Seattle", "start": 10, "end": 17, "attribute": "location", "importance": 0.85}],
        },
    )
    assert r_hl.status_code == 200
    assert "<mark" in r_hl.json()["highlighted_html"]

    # Test /telemetry
    r_tel = client.post(
        "/telemetry",
        json={
            "session_id": "test_sess",
            "scenario_id": "S-02",
            "event_type": "rewrite_requested",
            "rewrite_mode": "balanced_sft",
            "overall_risk_before": 0.65,
            "overall_risk_after": 0.20,
            "bidirectional_entailment": 0.85,
            "utility_warning_shown": False,
            "latency_ms": 115.0,
        },
    )
    assert r_tel.status_code == 200
    assert r_tel.json()["status"] == "recorded"
