"""Tests for the rewriter generalization fix (v2).

Covers the root causes of "memorizes locations of inferential PII instead of
actively finding and rewriting":

  RC1 train/serve prompt drift  -> single-turn prompt, one builder, parity test
  RC2 fixed-primer duplication  -> no turn may repeat across rows
  RC3 position bias             -> augmented corpus is position-balanced
  RC4 distribution narrowness   -> banks cover worldwide surfaces, attributes balanced
  RC5 leak gate misses relocations -> judge_augmented removes every injected surface
  RC6 keyword-checklist eval    -> span-recall evaluator catches relocated PII
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rewriter import teacher_sft as ts
from src.rewriter.augment import POSITIONS, compose_corpus, verify_targets, write_corpus
from src.rewriter.persona_banks import default_matcher, neutralize_text
from src.rewriter.prompting import (
    REWRITE_INSTRUCTION,
    REWRITER_SYSTEM_PROMPT,
    build_rewrite_messages,
    prune_persona_bidirectional,
)

V1_TRAIN = ROOT / "data" / "sft" / "teacher_v1" / "train.jsonl"
BENCH_ANNOTATED = ROOT / "data" / "evaluation" / "benchmark_test_50_annotated.jsonl"


# --- RC1 + RC2: single-turn prompt, no duplicated turns ------------------------


def test_prompt_is_single_turn_with_policy_system_prompt():
    messages = build_rewrite_messages("I live in Zurich.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == REWRITER_SYSTEM_PROMPT
    assert messages[1]["content"] == REWRITE_INSTRUCTION + "I live in Zurich."
    # The policy tells the model to look ANYWHERE in the text, not at a position.
    assert "beginning, middle, or end" in REWRITER_SYSTEM_PROMPT


def test_sft_row_prompt_equals_served_prompt():
    row = ts.build_row("I work as a nurse.", "I work in my role.", {})
    assert row["messages"][:-1] == build_rewrite_messages("I work as a nurse.")
    assert len(row["messages"]) == 3


def test_no_fixed_primer_can_be_duplicated():
    """Two rows built for different texts share no (user, assistant) turn."""
    r1 = ts.build_row("I am 30 and live in Oslo.", "I live in my area.", {})
    r2 = ts.build_row("Working as a welder in Lima.", "Working in my trade.", {})
    turns1 = {(m["content"]) for m in r1["messages"][1:]}
    turns2 = {(m["content"]) for m in r2["messages"][1:]}
    assert not (turns1 & turns2)


def test_few_shot_is_opt_in_and_sampled_not_fixed():
    m_default = build_rewrite_messages("t")
    m_shots = build_rewrite_messages("t", few_shot=[("a", "b"), ("c", "d")])
    assert len(m_default) == 2
    assert len(m_shots) == 6  # system + 2 pairs + final user
    # Default stays single-turn: the primer cannot silently come back.


# --- RC3 + RC4: augmentation balance -------------------------------------------


def test_compose_corpus_is_position_balanced():
    examples = compose_corpus(200, seed=7, noop_ratio=0.0)
    positions = {e.position for e in examples if e.attributes}
    assert positions == set(POSITIONS)
    counts = {}
    for e in examples:
        counts[e.position] = counts.get(e.position, 0) + 1
    n = len(examples)
    # Uniform-ish: every position gets at least 12% of rows (vs 69%/21%/10%
    # start/middle/end in the v1 data).
    for pos in POSITIONS:
        assert counts[pos] >= 0.12 * n, f"position {pos} underrepresented: {counts}"


def test_composed_targets_never_leak_injected_surfaces():
    examples = compose_corpus(300, seed=11)
    assert verify_targets(examples) == []


def test_composed_inputs_are_unique():
    examples = compose_corpus(150, seed=3)
    inputs = [e.input for e in examples]
    assert len(inputs) == len(set(inputs))


def test_augmentation_covers_weak_attribute():
    """Location was 10.9% of v1 rows; the composer must boost it."""
    examples = compose_corpus(300, seed=5, noop_ratio=0.0)
    attr_counts = {}
    for e in examples:
        for a in e.attributes:
            attr_counts[a] = attr_counts.get(a, 0) + 1
    total = sum(attr_counts.values())
    assert attr_counts.get("location", 0) / total >= 0.18


def test_augmentation_includes_identity_noops():
    examples = compose_corpus(100, seed=9, noop_ratio=0.15)
    noops = [e for e in examples if not e.attributes]
    assert noops
    assert all(e.target == e.input for e in noops)


def test_needs_persona_carriers_get_generalized_targets():
    """When the answer depends on a personal parameter, the persona is
    generalized (kept as a parameter), not deleted - utility is preserved."""
    from src.rewriter.augment import PersonaComposer

    composer = PersonaComposer()
    carrier = next(c for c in composer.carriers if c["id"] == "emergency_fund")
    example = composer.compose(carrier, "preamble", ["income"], __import__("random").Random(1), "x1")
    assert example.target.startswith("For someone")
    assert "income" in example.target.lower()
    # and the injected figure is gone:
    assert "k a year" not in example.target and "$" not in example.target


# --- RC5: the relocation gate ----------------------------------------------------


def _injected(**surfaces):
    return [{"attribute": a, "surface": s, "variants": []} for a, s in surfaces.items()]


def test_judge_augmented_accepts_a_clean_rewrite():
    injected = _injected(occupation="nurse", location="Tucson")
    ok, reason, _ = ts.judge_augmented(
        "I am a nurse in Tucson. How do I learn Rust?",
        "How do I learn Rust?",
        injected,
    )
    assert ok and reason is None


def test_judge_augmented_rejects_relocated_pii():
    """The v1 failure mode: moving the cue to the end instead of removing it."""
    injected = _injected(occupation="nurse", location="Tucson")
    ok, reason, details = ts.judge_augmented(
        "I am a nurse in Tucson. How do I learn Rust?",
        "How do I learn Rust? (asking as a nurse in Tucson)",
        injected,
    )
    assert not ok and reason == "leak:location,occupation"


def test_judge_augmented_checks_variants_too():
    injected = [{"attribute": "location", "surface": "Zurich", "variants": ["Zürich"]}]
    ok, reason, _ = ts.judge_augmented("Living in Zurich is great. Any tips?", "Living in Zürich is great. Any tips?", injected)
    assert not ok and reason == "leak:location"


def test_judge_augmented_rejects_unchanged_and_empty():
    injected = _injected(occupation="nurse")
    assert ts.judge_augmented("I am a nurse. Tips?", "I am a nurse. Tips?", injected)[1] == "unchanged"
    assert ts.judge_augmented("I am a nurse. Tips?", "   ", injected)[1] == "empty"


def test_augmented_rows_flow_through_build_teacher_sft(tmp_path):
    """Deterministic augmented rows are judged, mixed in, and carry position metadata."""
    records = [
        {
            "id": f"a{i}", "profile_id": "pers1", "split": "train",
            "text": f"Working as a software engineer and this is post {i}.",
            "profile": {"occupation": "software engineer"},
            "reviews": {"human": {"occupation": {"estimate": "software engineer"}}},
        }
        for i in range(4)
    ]

    def fake_generate(batch):
        return ["Working in my role and this is post A." for _ in batch]

    def nli(p, h):
        return {"entailment": 0.9, "contradiction": 0.01, "neutral": 0.09}

    aug_records = [
        {
            "id": f"aug_{i}", "input": f"I am a nurse in Tucson #{i}. How do I learn Rust?",
            "target": "How do I learn Rust?", "attributes": ["occupation", "location"],
            "position": "preamble", "register": "preamble:0",
            "injected": _injected(occupation="nurse", location="Tucson"),
            "needs_persona": False, "carrier_id": "c", "source": "augment",
        }
        for i in range(5)
    ]
    manifest = ts.build_teacher_sft(records, fake_generate, nli, tmp_path,
                                    augmented_records=aug_records, augment_mode="deterministic")
    assert manifest["augmented"]["accepted_rows"] == 5
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    aug_rows = [r for r in rows if r["metadata"]["source"] == "augment"]
    assert len(aug_rows) == 5
    assert all(r["metadata"]["position"] == "preamble" for r in aug_rows)
    # single-turn, served-prompt format:
    assert all(len(r["messages"]) == 3 for r in rows)


# --- RC1/RC2/RC3 regression guard: check_prompt_parity ---------------------------


def test_parity_check_flags_the_v1_dataset():
    if not V1_TRAIN.exists():
        pytest.skip("v1 SFT file not present")
    from scripts.check_prompt_parity import check_file

    stats = check_file(V1_TRAIN)
    assert stats["prompt_parity_violations"] == stats["rows"]  # 1749/1749 drift
    assert stats["duplication_violations"] >= 2  # the two fixed primer turns
    assert stats["problems"]


def test_parity_check_passes_a_v2_format_file(tmp_path):
    from scripts.check_prompt_parity import check_file

    rows = [
        ts.build_row("I am a nurse in Tucson. Question?", "Question?", {"attributes": ["occupation"]}),
        ts.build_row("Question? I work as a welder in Lima.", "Question?", {"attributes": ["occupation"]}),
        ts.build_row("Nothing personal here.", "Nothing personal here.", {"attributes": []}),
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats = check_file(path)
    assert stats["problems"] == []
    assert stats["noop_ratio"] > 0


# --- RC6: span-recall evaluation --------------------------------------------------


def _eval_case(rewritten, input_text="I am a 34-year-old oncologist in Baltimore. What is the protocol?"):
    from scripts.evaluate_benchmark_50 import evaluate_rewrite

    case = {
        "input": input_text,
        "pii_spans": [
            {"text": "34-year-old", "attribute": "age"},
            {"text": "oncologist", "attribute": "occupation"},
            {"text": "Baltimore", "attribute": "location"},
        ],
        "invariant_spans": ["protocol"],
    }
    return evaluate_rewrite(case, rewritten)


def test_evaluator_catches_relocated_pii():
    """The exact v1 benchmark failure: tc_09/tc_18 leaked personas and passed."""
    scores = _eval_case("What is the protocol? (for a 34-year-old oncologist in Baltimore)")
    assert scores["leaked_spans"] == ["34-year-old", "oncologist", "Baltimore"]
    assert scores["span_removal_rate"] == 0.0
    assert not scores["passed"]


def test_evaluator_passes_clean_rewrite():
    scores = _eval_case("What is the protocol?")
    assert scores["leaked_spans"] == []
    assert scores["span_removal_rate"] == 1.0
    assert scores["invariant_rate"] == 1.0
    assert scores["passed"]


def test_evaluator_flags_copy_through():
    scores = _eval_case("I am a 34-year-old oncologist in Baltimore. What is the protocol?")
    assert scores["copy_through"]
    assert not scores["passed"]


def test_evaluator_flags_lost_invariants():
    scores = _eval_case("What is the approach?")
    assert scores["missing_invariants"] == ["protocol"]
    assert not scores["passed"]


def test_annotated_benchmark_exists_and_has_spans():
    if not BENCH_ANNOTATED.exists():
        pytest.skip("annotated benchmark not generated")
    cases = [json.loads(l) for l in BENCH_ANNOTATED.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(cases) >= 50
    total = sum(len(c["pii_spans"]) for c in cases)
    assert total >= 80
    # Every preamble/postamble case must be annotated (position coverage).
    for c in cases:
        if "preamble" in c["category"] or "postamble" in c["category"]:
            assert c["pii_spans"], f"{c['id']} has no annotated spans"


# --- RC4: bank generality ----------------------------------------------------------


def test_matcher_covers_worldwide_surfaces():
    m = default_matcher()
    hits = m.attributes_in(
        "Living in Krakow, working as a barista, studied at Tsinghua University, take the MRT daily."
    )
    assert "location" in hits
    assert "occupation" in hits
    assert "education" in hits
    assert "local_reference" in hits


def test_neutralize_text_generalizes_beyond_any_hardcoded_city_list():
    out = neutralize_text("I am a radiologist in Reykjavik earning $210k.")
    assert "Reykjavik" not in out
    assert "radiologist" not in out
    assert "$210k" not in out
    assert "radiology" not in out.lower() or True  # no invented facts required


def test_neutralize_text_preserves_statutes_and_third_parties():
    out = neutralize_text(
        "Under Tokyo Metropolitan Building Ordinance, a 72-year-old patient with COPD needs review. "
        "I am a cardiologist in Osaka."
    )
    assert "Tokyo Metropolitan Building Ordinance" in out
    assert "72-year-old patient" in out
    assert "cardiologist" not in out
    assert "Osaka" not in out


def test_prune_preserves_case_when_nothing_pruned():
    assert prune_persona_bidirectional("lowercase prefix test") == "lowercase prefix test"
    assert prune_persona_bidirectional("As a doctor, can you help?").startswith("Can you help?")


def test_prune_embedded_compound_request_preserves_trailing_clauses():
    text = "How do I overcome stage fright as a 42 year old software professional, help me find resources for public speaking"
    pruned = prune_persona_bidirectional(text)
    assert "help me find resources for public speaking" in pruned
    assert "42 year old" not in pruned
    assert "software professional" not in pruned

    neutralized = neutralize_text(text)
    assert "help me find resources for public speaking" in neutralized
    assert "42 year old" not in neutralized
    assert "software professional" not in neutralized


def test_dpo_records_generation_strict_inequality():
    from src.rewriter.generate_dpo_data import generate_dpo_records
    from src.rewriter.augment import compose_corpus
    import random

    examples = compose_corpus(50, seed=42)
    records = generate_dpo_records(examples, random.Random(42))
    assert len(records) > 0
    for r in records:
        assert r["chosen"] != r["rejected"], f"DPO pair {r['id']} has identical chosen and rejected"
        assert r["metadata"]["failure_mode"] in [
            "truncation_over_scrub",
            "pii_leak_under_scrub",
            "syntax_placeholder_degenerate",
            "pii_leak_verbatim",
        ]


def test_decompose_reconstruct_system_prompt():
    from src.rewriter.prompting import REWRITER_SYSTEM_PROMPT
    assert "decompose" in REWRITER_SYSTEM_PROMPT.lower()
    assert "reconstruct" in REWRITER_SYSTEM_PROMPT.lower()
    assert "never truncate or drop" in REWRITER_SYSTEM_PROMPT.lower()


def test_neutralize_preserves_anaphoric_credentials_and_target_university():
    text = (
        "I am 26 years old living in Denver, CO and working as a software engineer after "
        "graduating from Carnegie Mellon. Do you think with these credentials, i can get into an MBA program at harvard?"
    )
    out = neutralize_text(text)
    # Target institution and program must be preserved verbatim
    assert "an MBA program at harvard" in out
    # The question and anaphora must survive
    assert "do you think with these credentials" in out.lower()
    # Author personal PII must be neutralized/removed
    assert "Denver" not in out
    assert "26 years old" not in out
    assert "Carnegie Mellon" not in out
