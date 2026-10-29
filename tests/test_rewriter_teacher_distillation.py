"""Tests for Task 1: Rewriter Generalization via Teacher Distillation.

Verifies:
1. Teacher candidate generation (diverse tones, preamble pruning, complex sentence structures).
2. SFT training configuration with small student model aliases and Google Drive checkpointing.
3. Held-out utility evaluation metrics and generalization comparison against static prompt baseline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from src.rewriter.prompting import is_personal_disclosure, prune_persona_bidirectional
from src.rewriter.generate_training_data import (
    TeacherCandidateGenerator,
    generate_candidate_rewrites_teacher,
    generate_diverse_teacher_candidates,
    generate_sft_dataset,
)
from src.rewriter.train_sft import (
    RewriterQLoRAConfig,
    detect_google_drive_path,
    get_default_dataset_path,
    parse_args,
    resolve_model_name,
)
from src.evaluation.utility_metrics import (
    BidirectionalUtilityEvaluator,
    compare_held_out_generalization,
    evaluate_held_out_split,
    evaluate_rewrites,
)


def test_generate_diverse_teacher_candidates_tones_and_structures():
    """Verify that programmatic teacher generation produces varied tones and complex structures."""
    text = (
        "I am a 28-year-old software engineer living in Seattle, WA. "
        "Can you help me optimize this database query for production?"
    )
    candidates = generate_diverse_teacher_candidates(text)

    assert len(candidates) >= 5, "Expected at least 5 diverse candidates"

    # 1. Preamble pruning: Should have an objective variant starting directly with the task
    has_pruned = any("Can you help me optimize" in c for c in candidates)
    assert has_pruned, "Expected at least one preamble-pruned objective rewrite"

    # 2. Professional / Formal tone: Neutralizes specific occupation and city into broad domain terms
    has_prof = any("Seattle" not in c and "28-year-old" not in c for c in candidates)
    assert has_prof, "Expected professional generalization neutralizing demographic cues"

    # 3. Conversational tone
    has_conversational = any("around where I live" in c or "work in my job" in c or "my region" in c for c in candidates)
    assert has_conversational, "Expected conversational variant"

    # 4. Complex sentence structure (subordinate clauses)
    has_subordinate = any("While " in c or "in my professional role" in c for c in candidates)
    assert has_subordinate, "Expected complex subordinate structure candidate"


def test_teacher_candidate_generator_class():
    """Verify TeacherCandidateGenerator initializes and generates cleanly."""
    generator = TeacherCandidateGenerator(teacher_model="gpt-4")
    sample = "I am a 34yo doctor in Denver working at a clinic."
    cands = generator.generate_candidates(sample)
    assert len(cands) > 0
    for c in cands:
        assert isinstance(c, str)
        assert c.strip()


def test_student_model_alias_resolution():
    """Verify small student model aliases resolve to official model identifiers."""
    assert resolve_model_name("qwen-0.6b") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolve_model_name("qwen0.6b") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolve_model_name("0.6b") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolve_model_name("qwen-0.5b") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolve_model_name("qwen-1.5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    # Unmapped model names pass through untouched
    assert resolve_model_name("Custom/Model-7B") == "Custom/Model-7B"


def test_google_drive_checkpoint_detection(monkeypatch):
    """Verify Google Drive path detection from argument, environment variable, or Colab mount."""
    # 1. Explicit user argument
    custom_path = "D:/ColabDrive/InferenceGuard"
    assert detect_google_drive_path(custom_path) == Path(custom_path)

    # 2. Environment variable
    monkeypatch.setenv("GOOGLE_DRIVE_PATH", "/content/drive/MyDrive/Backups")
    assert detect_google_drive_path(None) == Path("/content/drive/MyDrive/Backups")

    monkeypatch.delenv("GOOGLE_DRIVE_PATH", raising=False)
    monkeypatch.setenv("GDRIVE_DIR", "/content/drive/MyDrive/AltBackup")
    assert detect_google_drive_path(None) == Path("/content/drive/MyDrive/AltBackup")

    # 3. None when not mounted
    monkeypatch.delenv("GDRIVE_DIR", raising=False)
    if not Path("/content/drive/MyDrive").exists():
        assert detect_google_drive_path(None) is None


def test_train_sft_cli_arguments():
    """Verify parse_args supports Google Drive checkpointing and student model configuration."""
    test_args = [
        "train_sft.py",
        "--model_name", "qwen-0.6b",
        "--drive_path", "/content/drive/MyDrive/InferenceGuard/checkpoints",
        "--save_strategy", "steps",
        "--save_steps", "100",
        "--save_total_limit", "5",
        "--max_steps", "10",
    ]
    with patch("sys.argv", test_args):
        args = parse_args()
        assert args.model_name == "qwen-0.6b"
        assert resolve_model_name(args.model_name) == "Qwen/Qwen2.5-0.5B-Instruct"
        assert args.drive_path == "/content/drive/MyDrive/InferenceGuard/checkpoints"
        assert args.save_strategy == "steps"
        assert args.save_steps == 100
        assert args.save_total_limit == 5
        assert args.max_steps == 10


def test_evaluate_rewrites_aggregation():
    """Verify evaluate_rewrites computes correct aggregate metrics and detects degeneracy."""
    evaluator = BidirectionalUtilityEvaluator()
    pairs = [
        ("I like reading books on weekends.", "I enjoy reading books during the weekend."),
        ("Order 123 arrived yesterday.", "Order 123 arrived yesterday."),
        ("Normal message.", "123 an area, an area an area"),  # degenerate
    ]

    metrics = evaluate_rewrites(pairs, evaluator=evaluator)
    assert metrics["total_samples"] == 3
    assert 0.0 <= metrics["mean_semantic_similarity"] <= 1.0
    assert 0.0 <= metrics["mean_bidirectional_entailment"] <= 1.0
    assert metrics["degeneracy_rate"] == pytest.approx(1.0 / 3.0, 1e-3)
    assert metrics["utility_backend"] in ("fallback", "model")


def test_compare_held_out_generalization_synthetic():
    """Verify compare_held_out_generalization accurately reports deltas and improvements."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test.jsonl"
        items = [
            {"text": "I am 30 years old living in Boston, MA working in tech."},
            {"text": "Can you recommend a good book about system architecture?"},
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        class MockStudentRewriter:
            def rewrite(self, text: str) -> str:
                # Student generalizes nicely
                return text.replace("30 years old", "an adult").replace("Boston, MA", "locally")

        class MockBaselineRewriter:
            def rewrite(self, text: str) -> tuple[str, float]:
                # Baseline drops content aggressively
                return ("I work.", 50.0)

        evaluator = BidirectionalUtilityEvaluator()
        comparison = compare_held_out_generalization(
            test_split_path=test_file,
            student_rewriter=MockStudentRewriter(),
            baseline_rewriter=MockBaselineRewriter(),
            evaluator=evaluator,
            num_samples=2,
        )

        assert comparison["num_evaluated_samples"] == 2
        assert "delta" in comparison
        assert comparison["teacher_distilled_student"]["mean_semantic_similarity"] >= comparison["static_prompt_baseline"]["mean_semantic_similarity"]
        assert comparison["generalization_improved"] is True


def test_bidirectional_persona_pruning():
    """Verify bidirectional persona pruning strips disclosures in preambles, postambles, and parentheticals."""
    # 1. Preamble (front)
    front = (
        "I am a 62-year-old retired cardiologist living in Miami Beach. "
        "What is the latest guideline for SGLT2 inhibitors in heart failure with preserved ejection fraction?"
    )
    assert prune_persona_bidirectional(front) == (
        "What is the latest guideline for SGLT2 inhibitors in heart failure with preserved ejection fraction?"
    )

    # 2. Postamble (back)
    back = (
        "What is the latest guideline for SGLT2 inhibitors in heart failure with preserved ejection fraction? "
        "I am a 62-year-old retired cardiologist living in Miami Beach."
    )
    assert prune_persona_bidirectional(back) == (
        "What is the latest guideline for SGLT2 inhibitors in heart failure with preserved ejection fraction?"
    )

    # 3. Introductory clause (front) with decimal preservation (18.01)
    clause = (
        "As an 18-year-old freshman at MIT struggling with 18.01 single variable calculus, "
        "can you explain the epsilon-delta definition of a limit with an intuitive diagram?"
    )
    assert prune_persona_bidirectional(clause) == (
        "Can you explain the epsilon-delta definition of a limit with an intuitive diagram?"
    )

    # 4. Trailing note (back)
    trailing_note = (
        "Can you explain the epsilon-delta definition of a limit with an intuitive diagram? "
        "FYI, I am an 18-year-old freshman at MIT struggling with 18.01 calculus."
    )
    assert prune_persona_bidirectional(trailing_note) == (
        "Can you explain the epsilon-delta definition of a limit with an intuitive diagram?"
    )

    # 5. Trailing parenthetical (back)
    paren = (
        "Can you help me optimize my 401(k) asset allocation between VOO and VXUS? "
        "(Software engineer in Seattle earning $180k)"
    )
    assert prune_persona_bidirectional(paren) == (
        "Can you help me optimize my 401(k) asset allocation between VOO and VXUS?"
    )

    # 6. Objective domain entity preservation (no persona)
    clean = "Calculate the eigenvalues of the matrix [[1, 2], [3, 4]] with step-by-step derivation."
    assert prune_persona_bidirectional(clean) == clean
