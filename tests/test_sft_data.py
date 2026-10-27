# tests/test_sft_data.py
"""Verification suite for Gate 4: SFT Dataset Purity and Pareto Filtering.

Verifies that 100% of SFT rows originate from the train split, data leakage from
val/test is blocked with fatal assertions, ~15% identity no-ops are injected,
and non-noop rewrites satisfy bidirectional NLI >= 0.65.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

pytest.importorskip(
    "pydantic",
    reason="needs the optional pydantic dependency; see tests/README.md",
)

from src.contracts.schemas import (
    AttributeName,
    AttributeRiskDistribution,
    AttributedCue,
    RiskBand,
    RiskReport,
    UtilityReport,
)
from src.rewriter.generate_training_data import (
    ParetoSFTDatasetBuilder,
    format_cue_conditioned_instruction,
)


class MockRiskPredictor:
    """Mock risk predictor returning high risk for messages containing privacy cues."""

    def predict(self, text: str) -> RiskReport:
        t_low = text.lower()
        has_cue = "green line" in t_low or "seattle" in t_low or "co-op" in t_low
        score = 0.75 if has_cue else 0.15
        band = RiskBand.HIGH if has_cue else RiskBand.LOW

        attrs = {
            AttributeName.LOCATION: AttributeRiskDistribution(
                attribute=AttributeName.LOCATION,
                predicted_value="us_west" if "seattle" in t_low else "UNKNOWN",
                calibrated_confidence=score,
                top2_margin=0.40,
                normalized_entropy=0.25,
                inferability_risk_score=score,
                risk_band=band,
                class_probabilities={},
            )
        }
        for attr in [AttributeName.AGE, AttributeName.OCCUPATION, AttributeName.EDUCATION]:
            attrs[attr] = AttributeRiskDistribution(
                attribute=attr,
                predicted_value="UNKNOWN",
                calibrated_confidence=0.1,
                top2_margin=0.0,
                normalized_entropy=0.9,
                inferability_risk_score=0.1,
                risk_band=RiskBand.LOW,
                class_probabilities={},
            )

        cues = []
        if has_cue:
            cues.append(
                AttributedCue(
                    span_text="Green Line" if "green line" in t_low else "Seattle",
                    start=0,
                    end=10,
                    attribute=AttributeName.LOCATION,
                    importance=0.88,
                    deletion_risk_drop=0.35,
                )
            )

        return RiskReport(
            attributes=attrs,
            overall_risk_score=score,
            overall_risk_band=band,
            cues=cues,
        )


class MockUtilityEvaluator:
    """Mock utility evaluator simulating bidirectional NLI entailment scores."""

    def evaluate_pair(self, original_text: str, rewritten_text: str) -> UtilityReport:
        # Simulate realistic utility preservation: 0.85 entailment, 0.05 contradiction
        return UtilityReport(
            semantic_similarity=0.88,
            forward_entailment_prob=0.86,
            backward_entailment_prob=0.84,
            bidirectional_entailment=0.85,
            contradiction_prob=0.05,
            token_edit_ratio=0.25,
            utility_preserved=True,
        )


def dummy_candidate_generator(text: str) -> list[str]:
    """Generates synthetic candidate rewrites abstracting regional and occupational markers."""
    t_clean = (
        text.replace("Green Line", "public transit")
        .replace("co-op", "work placement")
        .replace("Seattle", "the metropolitan area")
    )
    return [t_clean, f"Protected: {t_clean}", t_clean.lower(), text]


def test_sft_dataset_purity_and_invariants():
    """Verify that all rows come from train, ~15% identity no-ops are present, and NLI >= 0.65."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_jsonl = Path(tmp_dir) / "train_sft.jsonl"
        builder = ParetoSFTDatasetBuilder(
            risk_predictor=MockRiskPredictor(),
            utility_evaluator=MockUtilityEvaluator(),
            candidate_generator_fn=dummy_candidate_generator,
            min_relative_risk_drop=0.20,
            min_bidirectional_nli=0.65,
        )

        train_records = []
        # 20 high-risk comments needing rewrites
        for i in range(20):
            train_records.append({
                "profile_id": f"p_{i}",
                "split": "train",
                "pseudonymized_text": f"I take the Green Line to my campus co-op every day {i}.",
                "active_pseudonyms": [],
            })
        # 10 low-risk comments suitable for identity no-ops
        for i in range(10):
            train_records.append({
                "profile_id": f"low_{i}",
                "split": "train",
                "pseudonymized_text": f"Hello, how can I configure my git repository {i}?",
                "active_pseudonyms": [],
            })

        stats = builder.build_sft_dataset(
            train_records=train_records,
            output_jsonl_path=out_jsonl,
            identity_ratio=0.15,
        )

        assert stats["accepted_rewrites"] > 0
        assert stats["accepted_identity_no_ops"] > 0
        assert out_jsonl.exists()

        rows = []
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        total_rows = len(rows)
        noop_rows = [r for r in rows if r["metadata"]["is_identity_noop"]]
        rewrite_rows = [r for r in rows if not r["metadata"]["is_identity_noop"]]

        # Invariant 1: Identity no-op proportion close to 15%
        noop_ratio = len(noop_rows) / total_rows
        assert 0.10 <= noop_ratio <= 0.25, f"Unexpected noop ratio: {noop_ratio:.2f}"

        # Invariant 2: 100% of non-noop rows satisfy bidirectional NLI >= 0.65
        for r in rewrite_rows:
            assert r["metadata"]["bidirectional_nli"] >= 0.65
            assert r["metadata"]["risk_before"] > r["metadata"]["risk_after"]

        # Invariant 3: Structured prompt format compliance
        for r in rows:
            assert "<|im_start|>system" in r["prompt"]
            assert "[RISK_ATTRIBUTES]:" in r["prompt"]
            assert "[DETECTED_CUES]:" in r["prompt"]
            assert "[PRESERVE_TOKENS]:" in r["prompt"]
            assert "[MESSAGE]:" in r["prompt"]
            assert r["completion"].endswith("<|im_end|>")


def test_data_leakage_fatal_assertion():
    """Verify that passing val or test records to SFT builder raises immediate fatal RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_jsonl = Path(tmp_dir) / "leaked_sft.jsonl"
        builder = ParetoSFTDatasetBuilder(
            risk_predictor=MockRiskPredictor(),
            utility_evaluator=MockUtilityEvaluator(),
        )

        contaminated_records = [
            {"profile_id": "val_pers", "split": "val", "pseudonymized_text": "Validation comment."},
        ]

        with pytest.raises(RuntimeError, match="CRITICAL DATA LEAKAGE BLOCKED"):
            builder.build_sft_dataset(contaminated_records, out_jsonl)
