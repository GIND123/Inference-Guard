"""Tests for comparative evaluation harness and demographic taxonomy mapping.

Verifies raw SynthPAI attribute mapping into coarse demographic taxonomy buckets,
abstention-aware metric computation, and comparative benchmark pipeline execution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.contracts.schemas import AttributeName
from src.evaluation.privacy_metrics import compute_method_metrics
from src.evaluation.run import run_comparative_benchmark
from src.risk_model.taxonomy import (
    ATTRIBUTE_TAXONOMY,
    ID_TO_LABEL,
    LABEL_TO_ID,
    map_raw_to_taxonomy_label,
    map_to_taxonomy_id,
)


def test_taxonomy_mapping_fine_grained_to_coarse():
    """Verify raw fine-grained strings map accurately to coarse taxonomy buckets."""
    # Age mappings
    assert map_raw_to_taxonomy_label(AttributeName.AGE, "48") == "45-54"
    assert map_raw_to_taxonomy_label(AttributeName.AGE, 22) == "18-24"
    assert map_raw_to_taxonomy_label(AttributeName.AGE, "early 30s") == "25-34"
    assert map_raw_to_taxonomy_label(AttributeName.AGE, "65") == "55+"
    assert map_raw_to_taxonomy_label(AttributeName.AGE, None) == "UNKNOWN"

    # Location mappings
    assert map_raw_to_taxonomy_label(AttributeName.LOCATION, "Atlanta, United States") == "us_south"
    assert map_raw_to_taxonomy_label(AttributeName.LOCATION, "Seattle, WA") == "us_west"
    assert map_raw_to_taxonomy_label(AttributeName.LOCATION, "London, UK") == "uk_europe"
    assert map_raw_to_taxonomy_label(AttributeName.LOCATION, "Toronto, Ontario") == "canada"
    assert map_raw_to_taxonomy_label(AttributeName.LOCATION, "us_northeast") == "us_northeast"

    # Education mappings
    assert map_raw_to_taxonomy_label(AttributeName.EDUCATION, "PHD in Psychology") == "graduate_professional"
    assert map_raw_to_taxonomy_label(AttributeName.EDUCATION, "Bachelor of Science") == "undergraduate"
    assert map_raw_to_taxonomy_label(AttributeName.EDUCATION, "high_school") == "high_school"

    # Occupation mappings
    assert map_raw_to_taxonomy_label(AttributeName.OCCUPATION, "psychologist") == "healthcare"
    assert map_raw_to_taxonomy_label(AttributeName.OCCUPATION, "Senior Software Engineer") == "technology"
    assert map_raw_to_taxonomy_label(AttributeName.OCCUPATION, "Undergraduate student") == "student"

    # Integer ID mapping
    loc_id = map_to_taxonomy_id(AttributeName.LOCATION, "Atlanta, United States")
    assert ID_TO_LABEL[AttributeName.LOCATION][loc_id] == "us_south"


def test_compute_method_metrics_resilient_to_raw_strings():
    """Verify compute_method_metrics handles raw strings without dropping valid samples."""
    records = [{"profile_id": "p1"}, {"profile_id": "p2"}]

    preds = {
        AttributeName.LOCATION: ["us_south", "UNKNOWN"],
        AttributeName.AGE: ["45-54", "25-34"],
        AttributeName.OCCUPATION: ["healthcare", "UNKNOWN"],
        AttributeName.EDUCATION: ["graduate_professional", "undergraduate"],
    }

    # Ground truth passed as raw SynthPAI strings rather than pre-mapped coarse labels
    trues = {
        AttributeName.LOCATION: ["Atlanta, United States", "Seattle, WA"],
        AttributeName.AGE: ["48", "30"],
        AttributeName.OCCUPATION: ["psychologist", "software engineer"],
        AttributeName.EDUCATION: ["PHD in Psychology", "Bachelor of Science"],
    }

    metrics = compute_method_metrics(
        records=records,
        predictions_by_attr=preds,
        ground_truth_by_attr=trues,
        utility_scores=[{"bidirectional_entailment": 0.95, "semantic_similarity": 0.92, "contradiction_prob": 0.0}],
        latencies_ms=[12.5, 14.2],
    )

    assert "per_attribute" in metrics
    for attr in AttributeName:
        assert attr.value in metrics["per_attribute"]

    loc_m = metrics["per_attribute"]["location"]
    # 1 hit ("us_south" == "us_south"), 1 abstention ("UNKNOWN") -> 50% accuracy on sample 1
    assert loc_m["attack_accuracy"] == 0.50
    assert loc_m["abstention_rate"] == 0.50
    assert metrics["overall_attacker_abstention_rate"] > 0.0


def test_comparative_benchmark_runner_offline_mode():
    """Verify run_comparative_benchmark completes and exports metadata in offline mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_comparative_benchmark(
            split="val",
            methods=("raw", "presidio"),
            output_dir=tmpdir,
            limit=2,
            use_heavy_models=False,
        )

        assert results["sample_count"] == 2
        assert results["adversary_mode"] == "heuristic"
        assert results["rewriter_engine"] == "heuristic"
        assert results["utility_mode"] == "heuristic"
        assert "raw" in results["methods"]
        assert "presidio" in results["methods"]

        raw_metrics = results["methods"]["raw"]
        assert "overall_attack_success_rate" in raw_metrics
        assert "overall_attacker_abstention_rate" in raw_metrics
        assert "mean_bidirectional_entailment" in raw_metrics
        assert "per_attribute" in raw_metrics
        assert len(raw_metrics["per_attribute"]) == 4

        saved_file = Path(tmpdir) / "comparative_benchmark_val.json"
        assert saved_file.exists()
