# Rewriter Generalization Fix — Root Causes and Changes

## The symptom you reported

> "It does not generalize properly to a user input and often memorizes
> locations of inferential PII instead of actively trying to find and rewrite
> the sentence."

## Root-cause analysis (all verified against the committed artifacts)

| # | Root cause | Evidence |
|---|---|---|
| RC1 | **Train/serve prompt drift.** The committed `data/sft/teacher_v1/train.jsonl` was built with an older 2-shot primer; `src/rewriter/prompting.py` served a different 3-shot prompt. | 1749/1749 training rows (100%) do not match the prompt inference serves (`scripts/check_prompt_parity.py` confirms). The adapter was served prompts it never saw. |
| RC2 | **A fixed few-shot primer embedded in every row.** With `DataCollatorForCompletionOnlyLM`, every assistant turn gets loss. | 3,498 of 5,247 supervised turns (66.7%) are the same two completions ("Google→[COMPANY]", "mosque→personal religious obligations"), each receiving 1,749 gradient updates — the strongest memorization signal in the dataset. Only 1,487 unique rewrites carry real signal. |
| RC3 | **Position bias.** The cue sat at the start of the input in training. | 69% of first-changed tokens are in the first third of the input; 21% middle; 10% end. Inputs are median 12 words. The model learns "the PII is at the beginning", not cue detection. The served few-shot reinforced it (2 of 3 examples were preamble pruning). |
| RC4 | **Distribution narrowness + attribute imbalance.** | Casual Reddit one-liners (p25=9, p75=28 words) vs. real inputs (long, formal, multi-clause). Location in only 190/1749 rows (10.9%); occupation 57%. The legacy heuristic pipeline hardcodes ~10 US cities, ~10 jobs, 5 universities — a ceiling on surface generalization. |
| RC5 | **The acceptance gate measured verbatim-value leak, not inferability.** | `judge()` only rejected a rewrite if the author's exact ground-truth token run appeared. Relocating the value into a new clause, or keeping a different equally-identifying cue, passed. So the training signal is "substitute the memorized string", not "find and neutralize". 15% identity no-ops additionally taught copy-through (held-out eval showed token_edit_ratio 0.0 — the student copied inputs verbatim). |
| RC6 | **Evaluation was a hardcoded keyword checklist.** | `scripts/evaluate_benchmark_50.py` checked 5 fixed strings; tc_09 and tc_18 leaked full personas ("for an 18-year-old first-year pre-law undergraduate at NYU in Manhattan") and still PASSED. Broken feedback loop (documented as "Goodhart" in your own methodology review). |

## The fix (root causes, not surface patches)

### 1. Single-turn prompt, one builder (`src/rewriter/prompting.py`)
- `build_rewrite_messages(text)` is now `[system, user]` — the rewriting policy lives in `REWRITER_SYSTEM_PROMPT` (explicitly: look for cues "beginning, middle, or end, in any wording").
- No fixed primer by default ⇒ nothing can be memorized by repetition, nothing can drift (one code path used by SFT construction, inference and the tests).
- Optional `few_shot` parameter exists for inference-time experiments only; there is no default pool to go stale.
- `prune_persona_bidirectional` / `is_personal_disclosure` kept for the fallback; fixed case preservation (capitalize only when something was actually pruned) and extended third-party guards (clinical presentations like "A 72-year-old male presents with…").

### 2. Externalized persona banks (`data/rewriter/personas/*.json` + `src/rewriter/persona_banks.py`)
- 210+ worldwide cities, 156 occupations across 14 sectors, 105 global institutions, 40 indirect local references (transit lines, landmarks, customs), age and income expression patterns.
- One matcher (`PersonaMatcher`) drives the fallback neutralizer, the acceptance gate, the benchmark annotator and the composition engine — replacing the hardcoded regex lists in code. Growing the system's coverage is now a data edit, not a code change.
- `neutralize_text()` (the no-adapter fallback) is bank-driven with context guards: statute references ("Tokyo Metropolitan Building Ordinance") and third-party subjects ("a 3-year-old patient") are preserved, not neutralized.

### 3. Position-controlled augmentation (`src/rewriter/augment.py`)
- Composes examples **by construction** with known cue spans at uniformly sampled positions: `preamble / postamble / mid_sentence / parenthetical / split` (both ends), across registers (4+ frame variants per position), attribute counts 1–3 with location boosted (was the thinnest class).
- Carriers (70 PII-free domain questions: medicine, law, engineering, finance, cooking, fitness, travel…) × banks × positions × registers ⇒ thousands of unique rows, deterministic, no GPU.
- Because the composer knows what it injected, the target is known by construction: prune (persona irrelevant) or *generalize* (persona is a parameter the answer needs — e.g. "For someone with variable income: how should an emergency fund be structured?").
- Guarantees: no duplicate inputs, no target contains any injected surface (verified at build time), carrier/injection collisions are redrawn.

### 4. Honest acceptance gate (`judge_augmented` in `src/rewriter/teacher_sft.py`)
- Rejects a rewrite unless **every injected surface form (and its variants) is absent** — whole-phrase, case-insensitive. "Relocated instead of removed" is now a hard reject.
- Identity is *correct* (accepted) for no-fact rows — calibrated no-ops instead of copy-through pressure.
- `--augment-dir` mixes augmented rows into the teacher build through the same gates; `--augment-only` builds an SFT set with zero GPU; `--augment-mode teacher` lets the teacher paraphrase for naturalness while removal stays verified.
- Manifest now reports position/attribute distributions and the augmented share.

### 5. Regression guard (`scripts/check_prompt_parity.py`)
- Fails (exit 1) on: prompt drift vs. `build_rewrite_messages`; any (user, assistant) turn duplicated across >1% of rows; multi-turn rows.
- Reports cue-position distribution (from row metadata), noop ratio, length stats, attribute coverage — so distribution regressions are visible before training.
- Run it on any SFT file before `train_sft.py`.

### 6. Honest evaluation (`scripts/evaluate_benchmark_50.py` + `data/evaluation/annotate_benchmark_spans.py`)
- The 50-case benchmark is now span-annotated (`pii_spans`, `invariant_spans`) from the same banks — reviewable data, not per-case code.
- Scores per case: **span removal rate**, **invariant preservation** (jurisdictions, standards, third-party subjects), **copy-through**, **added PII**; aggregated per category (position-stratified).
- `--fallback_baseline` scores the bank-driven fallback as the floor the trained model must beat.
- The tc_09/tc_18 relocation leak is now caught (span recall 0).

### 7. Incidental pre-existing bugs fixed (found by the suite, verified failing on the pristine tree)
- `generate_candidate_rewrites_heuristic`: the pruned candidate was never hard-PII-scrubbed (email leaked); `placeholder_cues()` existed but was never wired into the candidate list.
- `web/api.py` / `train_sft.py` hardcoded model IDs instead of the shared constant (their own `test_model_consistency` flagged them; `SMALL_STUDENT_MODEL` added to `src/rewriter/inference.py`).

## How to retrain (the v2 flow)

```bash
# 1. Position-controlled corpus (deterministic, no GPU)
python -m src.rewriter.augment --n 2000 --out data/sft/augmented_v1/augmented.jsonl

# 2a. Bootstrap SFT without any teacher (optional but useful)
python -m src.rewriter.teacher_sft --augment-only --augment-dir data/sft/augmented_v1 --out-dir data/sft/augmented_v1

# 2b. Full teacher run (Qwen2.5-7B) mixing SynthPAI + augmented through the gates
python -m src.rewriter.teacher_sft --split train --device cuda \
    --out-dir data/sft/teacher_v2 --augment-dir data/sft/augmented_v1

# 3. Gate the dataset (must exit 0)
python scripts/check_prompt_parity.py data/sft/teacher_v2/train.jsonl

# 4. Train
python -m src.rewriter.train_sft --dataset_path data/sft/teacher_v2/train.jsonl

# 5. Evaluate honestly
python scripts/evaluate_benchmark_50.py --adapter_path artifacts/rewriter_qlora
```

A ready-made `data/sft/augmented_v1/train.jsonl` (2,000 rows, parity-clean,
positions: preamble 386 / postamble 384 / parenthetical 486 / split 391 /
mid 282 / noop 71) is committed as a starting point.

## Test status

31 modules, **415 tests passing**, including the new
`tests/test_rewriter_generalization_fix.py` (26 tests covering RC1–RC6).
Excluded from this environment (identical behavior on the pristine tree):
`test_slow_real_models`, `test_e2e_qwen_inference` (need real weights),
`test_stress_loader_fallback` (hangs identically on pristine).

## What is deliberately NOT claimed

- The benchmark fallback score (100% removal / 100% invariants) is the
  *mechanical floor*, not a model result. The trained student must beat it on
  naturalness; the benchmark must keep growing (10 cases/release, spans
  reviewed by hand).
- The span annotations are matcher-driven and reviewable; the annotator prints
  any case it cannot annotate. Review before trusting category-level numbers.
- The held-out Phi-4 attacker evaluation (blocked in session 08 pending an
  adapter trained on v2 data) should be unblocked and run before any privacy
  claim is repeated.
