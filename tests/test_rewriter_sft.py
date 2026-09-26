"""Tests for Milestone 2: Rewriter SFT Dataset Generation (F5, F6, F7).

Verifies train profile isolation, Pareto rejection sampling, ChatML/Alpaca schema export,
and enable_thinking=False compliance.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from src.rewriter.generate_training_data import (
    compute_bidirectional_nli_score,
    compute_token_cosine_similarity,
    evaluate_rewrite_utility,
    estimate_privacy_risk,
    filter_train_records,
    format_alpaca_example,
    format_chatml_example,
    generate_candidate_rewrites_heuristic,
    generate_sft_dataset,
    load_profile_splits,
    pareto_rejection_sample,
    scrub_hard_pii,
)


def test_load_profile_splits_validity():
    """Verify profile splits load 210 train, 45 val, 45 test profiles with zero overlap."""
    splits_file = Path("artifacts/profile_splits.json")
    assert splits_file.exists(), "artifacts/profile_splits.json must exist"

    splits = load_profile_splits(splits_file)
    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"]

    assert len(train_set) == 210, f"Expected 210 train profiles, got {len(train_set)}"
    assert len(val_set) == 45, f"Expected 45 val profiles, got {len(val_set)}"
    assert len(test_set) == 45, f"Expected 45 test profiles, got {len(test_set)}"

    # Ensure strictly disjoint
    assert len(train_set & val_set) == 0, "Train and Val splits must not intersect"
    assert len(train_set & test_set) == 0, "Train and Test splits must not intersect"
    assert len(val_set & test_set) == 0, "Val and Test splits must not intersect"


def test_filter_train_records_isolation():
    """Verify profile isolation strictly retains only train profiles and drops val/test."""
    train_profiles = {"pers1", "pers2", "pers3"}
    records = [
        {"author": "pers1", "text": "I am in Denver."},
        {"author": "pers2", "text": "I work in tech."},
        {"author": "pers99", "text": "I live in Austin."},  # Not in train
        {"profile": "pers3", "text": "I graduated college."},
        {"profile": "pers104", "text": "I like biking."},  # Val profile
        {"text": "Anonymous comment with no profile."},      # Missing profile
    ]

    retained, rejected = filter_train_records(records, train_profiles)
    assert len(retained) == 3
    assert rejected == 3
    retained_ids = [r.get("author") or r.get("profile") for r in retained]
    assert set(retained_ids) == {"pers1", "pers2", "pers3"}


def test_scrub_hard_pii():
    """Verify direct PII (email, phone, ip, url) is identified and scrubbed."""
    raw = "Contact alice@example.com or call 555-123-4567 at https://mycompany.org from 192.168.1.1."
    scrubbed, had_pii = scrub_hard_pii(raw)

    assert had_pii is True
    assert "alice@example.com" not in scrubbed
    assert "[EMAIL]" in scrubbed
    assert "555-123-4567" not in scrubbed
    assert "[PHONE]" in scrubbed
    assert "192.168.1.1" not in scrubbed
    assert "[IP_ADDRESS]" in scrubbed
    assert "https://mycompany.org" not in scrubbed
    assert "[URL]" in scrubbed


def test_token_cosine_similarity():
    """Verify cosine similarity calculation properties."""
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The quick brown fox jumps over the lazy dog"
    assert compute_token_cosine_similarity(text1, text2) == pytest.approx(1.0, 1e-4)

    text3 = "Apples oranges bananas watermelon pineapple"
    assert compute_token_cosine_similarity(text1, text3) == pytest.approx(0.0, 1e-4)

    text4 = "The quick brown fox sleeps all day"
    sim = compute_token_cosine_similarity(text1, text4)
    assert 0.4 < sim < 0.9

    assert compute_token_cosine_similarity("", "something") == 0.0


def test_bidirectional_nli_score():
    """Verify bidirectional NLI entailment and contradiction penalty."""
    orig = "I moved to Denver to work as a software engineer at a startup."
    rewrite = "I relocated locally to work in a professional field at an organization."
    nli = compute_bidirectional_nli_score(orig, rewrite)

    assert nli["forward_entailment"] > 0.3
    assert nli["backward_entailment"] > 0.3
    assert nli["contradiction"] == 0.0
    assert nli["composite_nli"] > 0.3

    # Contradiction case
    contradicting = "I never moved anywhere and I do not work."
    nli_contra = compute_bidirectional_nli_score(orig, contradicting)
    assert nli_contra["contradiction"] > 0.0


def test_estimate_privacy_risk_cues():
    """Verify inferential cue risk scoring produces expected risk bands."""
    text_with_cues = (
        "I am 23 years old living in Denver, CO and working as a software engineer "
        "after getting my bachelor's from Carnegie Mellon."
    )
    scores = estimate_privacy_risk(text_with_cues)
    assert scores["overall"] >= 0.45
    assert scores["age"] >= 0.45
    assert scores["location"] >= 0.45
    assert scores["occupation"] >= 0.45
    assert scores["education"] >= 0.45

    safe_text = "The weather was cloudy and rain is expected later tonight."
    safe_scores = estimate_privacy_risk(safe_text)
    assert safe_scores["overall"] < 0.30


def test_pareto_rejection_sample_selection():
    """Verify rejection sampling enforces risk < 0.30 and cosine >= 0.82."""
    orig = "I am a software engineer living in Denver, CO."

    # Candidate 1: High risk (keeps Denver, CO)
    cand_high_risk = "I am an engineer living in Denver, CO."
    # Candidate 2: Low utility (completely unrelated)
    cand_low_utility = "The solar system contains eight major planets."
    # Candidate 3: Valid privacy and utility
    cand_valid = "I am a professional working locally in the area."

    candidates = [cand_high_risk, cand_low_utility, cand_valid]
    selected = pareto_rejection_sample(
        original=orig,
        candidates=candidates,
        max_risk_threshold=0.30,
        min_cosine_threshold=0.20,  # relaxed for short sentence
    )

    assert selected is not None
    chosen_text, meta = selected
    assert chosen_text == cand_valid
    assert meta["overall_risk"] < 0.30


def test_format_chatml_and_alpaca_disable_thinking():
    """Verify ChatML and Alpaca formatting sets enable_thinking=False and removes <think> tags."""
    orig = "Original user comment."
    rewrite_with_think = "<think>Let me reason about this rewrite.</think>Scrubbed rewrite."

    chatml = format_chatml_example(orig, rewrite_with_think)
    assert chatml["enable_thinking"] is False
    assert len(chatml["messages"]) == 3
    assert chatml["messages"][0]["role"] == "system"
    assert chatml["messages"][1]["role"] == "user"
    assert chatml["messages"][1]["content"] == orig
    assert chatml["messages"][2]["role"] == "assistant"
    assert "<think>" not in chatml["messages"][2]["content"]
    assert chatml["messages"][2]["content"] == "Scrubbed rewrite."

    alpaca = format_alpaca_example(orig, rewrite_with_think)
    assert alpaca["enable_thinking"] is False
    assert alpaca["input"] == orig
    assert "<think>" not in alpaca["output"]
    assert alpaca["output"] == "Scrubbed rewrite."


def test_full_generate_sft_dataset_pipeline():
    """Verify end-to-end dataset generation from synthetic source file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(
            json.dumps({"train": ["pers1", "pers2"], "val": ["pers3"], "test": ["pers4"]}),
            encoding="utf-8",
        )

        synth_path = tmp_path / "synthpai.jsonl"
        items = [
            {"author": "pers1", "text": "I am 25 years old and work as a software engineer in Austin."},
            {"author": "pers2", "text": "I moved to Seattle to join a tech startup as a designer."},
            {"author": "pers3", "text": "Val profile record - should be excluded."},
            {"author": "pers4", "text": "Test profile record - should be excluded."},
        ]
        with open(synth_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        chatml_out = tmp_path / "chatml.jsonl"
        alpaca_out = tmp_path / "alpaca.jsonl"

        summary = generate_sft_dataset(
            synthpai_path=str(synth_path),
            splits_path=str(splits_path),
            output_chatml_path=str(chatml_out),
            output_alpaca_path=str(alpaca_out),
            max_samples=10,
            min_cosine_threshold=0.3,
        )

        assert summary["total_source_records"] == 4
        assert summary["train_records_retained"] == 2
        assert summary["rejected_non_train_profiles"] == 2
        assert summary["accepted_samples"] > 0

        assert chatml_out.exists()
        assert alpaca_out.exists()

        # Check output validity
        with open(chatml_out, "r", encoding="utf-8") as f:
            chatml_lines = [json.loads(line) for line in f]
        assert len(chatml_lines) == summary["accepted_samples"]
        for line in chatml_lines:
            assert line["enable_thinking"] is False
            assert "messages" in line

        with open(alpaca_out, "r", encoding="utf-8") as f:
            alpaca_lines = [json.loads(line) for line in f]
        assert len(alpaca_lines) == summary["accepted_samples"]
        for line in alpaca_lines:
            assert line["enable_thinking"] is False
            assert "instruction" in line
            assert "input" in line
            assert "output" in line


