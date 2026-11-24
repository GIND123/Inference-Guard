# Requirements coverage — design doc vs. repository

**Owner:** Anna Thomas (Data&Eval) · **Session:** 08 follow-up · **Status:** evidence-grounded gap analysis

This maps every numbered key requirement of the *InferenceGuard — Technical Design and
Team Execution Document* (items 1–19) to a status of **Satisfied**, **Partial**, or
**Missing**, with a one-line justification that points at a real file, symbol, test, or
report in this repository (or names the absence). Each claim was checked by reading the
implementing code and grepping for the symbol, not assumed.

Where the repository deliberately diverges from the design doc (Qwen3.5 → Qwen2.5,
location as classification rather than generative scoring, education buckets, closed-loop
vs. held-out adversary), the divergence is presented as a **documented decision** recorded
in [`docs/methodology-review.md`](methodology-review.md), not as a defect.

## Verification caveat

This analysis is based entirely on **static reading of code, docs, and committed reports**.
The sandbox runs in `INTEGRATIONS_ONLY` network mode: `pip`/PyPI and
`download.pytorch.org` are blocked, and `torch`, `numpy`, `transformers`, `pydantic`,
`scipy`, `datasets`, and `huggingface_hub` are not importable. The pytest suite
(`tests/run_all_tests.py`) therefore **could not be executed here**, and no test-run
result is claimed. Every metric quoted below is traced to an existing file; no numbers were
invented. Where a report itself labels a number as void, proxy, or pending, that label is
preserved.

## Status summary

| Status | Count | Requirement items |
|---|---:|---|
| Satisfied | 9 | 1, 3, 4, 5, 7, 11, 12, 13, 15 |
| Partial | 9 | 2, 6, 8, 9, 10, 14, 16, 18, 19 |
| Missing | 0 | — |

"Partial" is the honest majority verdict: the architecture, contracts, and harnesses exist
and are tested for shape, but the project's own Session 07–08 reports state that the
single end-to-end held-out leakage number the thesis rests on has not yet been produced,
because no trained rewriter adapter or risk checkpoint is committed.

---

## Requirement-by-requirement

### 1. Inferential-privacy scope + four core attributes + single/multi-turn — **Satisfied**
The four core attributes are `age, location, occupation, education`, frozen as
`AttributeName` in [`src/contracts/schemas.py`](../src/contracts/schemas.py) and as
`ATTRIBUTES` in [`src/risk_model/model.py`](../src/risk_model/model.py) and
[`src/product/risk_bands.py`](../src/product/risk_bands.py). Multi-turn accumulation is
implemented in [`src/conversation/leakage.py`](../src/conversation/leakage.py)
(`BayesianConversationLeakageTracker`) and surfaced through `process_turn` in
[`src/pipeline/orchestrator.py`](../src/pipeline/orchestrator.py). The README frames the
inferential-vs-explicit scope with the Green Line example.

### 2. Seven-stage architecture — **Partial**
All seven stages exist as code: (1) Presidio explicit-PII in
[`src/evaluation/presidio_baseline.py`](../src/evaluation/presidio_baseline.py) and
pseudonymisation in [`src/conversation/pseudonyms.py`](../src/conversation/pseudonyms.py);
(2) ModernBERT risk in [`src/risk_model/model.py`](../src/risk_model/model.py); (3) cue
attribution via Integrated Gradients in
[`src/risk_model/attribution.py`](../src/risk_model/attribution.py)
(`SyntaxAlignedCueAttributor`); (4) conversation state in
[`src/conversation/state.py`](../src/conversation/state.py) and
`ConversationState`/`TurnHistoryItem` schemas; (5) Qwen rewriter in
[`src/rewriter/inference.py`](../src/rewriter/inference.py); (6) post-rewrite rescoring
(Step 5 of `InferenceGuardPipeline.process_turn`); (7) held-out Phi-4 adversary in
[`src/evaluation/attacker.py`](../src/evaluation/attacker.py). **Partial** because the
pipeline runs end-to-end only in fallback mode: no trained risk checkpoint or rewriter
adapter is committed, so the served risk model uses untrained heads and the rewriter falls
back to `presidio_fallback`/heuristic (production-readiness.md §7; session08.md "What did
not work").

### 3. Model stack table vs. `configs/models.yaml` — **Satisfied (with documented divergence)**
[`configs/models.yaml`](../configs/models.yaml) records the verified stack: ModernBERT-base
risk, `Qwen/Qwen2.5-1.5B-Instruct` rewriter (with `Qwen2.5-0.5B-Instruct` as the on-device
candidate), `microsoft/Phi-4-mini-instruct` held-out adversary, MiniLM + `nli-deberta-v3-large`
utility, Presidio baseline. **Documented divergence:** the design doc names Qwen3.5-0.8B/2B,
which [`docs/methodology-review.md`](methodology-review.md) finding 1 shows are
vision-language models (`Qwen3_5ForConditionalGeneration`, vision_config, 248k vocab); the
team deliberately chose text-only Qwen2.5. `DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`
in [`src/rewriter/inference.py`](../src/rewriter/inference.py) is the single source of truth,
guarded by `tests/test_model_consistency.py`. (Note: a stale `Qwen/Qwen3.5-0.8B` default
literal survives in [`src/rewriter/prompt_baseline.py`](../src/rewriter/prompt_baseline.py);
it is overridden by `DEFAULT_BASE_MODEL` on tokenizer load but should be cleaned up.)

### 4. Risk model multi-task heads + calibration + risk bands — **Satisfied**
`MultiTaskModernBERTRiskModel` in [`src/risk_model/model.py`](../src/risk_model/model.py)
has four attention-pooled multi-class heads over `ATTRIBUTE_TAXONOMY`
([`src/risk_model/taxonomy.py`](../src/risk_model/taxonomy.py)). Production calibration is
`AttributeTemperatureScaler` in [`src/risk_model/calibrate.py`](../src/risk_model/calibrate.py):
L-BFGS temperature scaling, `compute_expected_calibration_error`, and Youden's-J risk
thresholds, tested by `tests/test_calibration_session07.py`. Bands LOW/MEDIUM/HIGH are
`RiskBand` in schemas and `Band` in [`src/product/risk_bands.py`](../src/product/risk_bands.py).
Reported validation numbers (macro F1 0.461, macro ROC-AUC 0.853) are from
[`reports/session05.md`](../reports/session05.md) and carried in session07/08; the test split
is still untouched by the team's own statement.

### 5. SynthPAI profile-disjoint 70/15/15 splits + PromptPAI secondary set — **Satisfied (primary); PromptPAI absent as documented scope cut**
Profile-disjoint splits are committed: `data/splits/{train,val,test}.jsonl` with SHA-256
manifest [`data/manifests/split_manifest.json`](../data/manifests/split_manifest.json),
counts **train 210 profiles / 5,232 comments, val 45 / 1,416, test 45 / 1,175** (verified in
the manifest and corroborated by session07.md/session08.md). Generated by
[`src/data/splits.py`](../src/data/splits.py), validated by
[`scripts/check_splits.py`](../scripts/check_splits.py) and `tests/test_splits.py`. A grep for
`PromptPAI` returns no matches anywhere in the repo, so the secondary PromptPAI set is not
present; it is a scope item the reports never claim to have delivered.

### 6. Rewriter SFT: train-partition only + teacher LLM + leak/NLI filtering + Phi-4 never used + DPO stretch — **Partial**
[`src/rewriter/teacher_sft.py`](../src/rewriter/teacher_sft.py) builds the SFT set strictly
from the train split (refuses val/test by manifest hash), uses a local teacher LLM
(Qwen2.5-7B per session08.md), and applies a six-gate filter: changed, non-degenerate, no
verbatim leak, bidirectional NLI contradiction bound, and forward-entailment/token-cosine.
Its docstring states the held-out Phi-4 adversary is refused as teacher — independence
enforced. session08.md PR #69 reports 1,749 distilled pairs. **Partial** because: the trained
adapter is **not committed** (session08.md "Absence of committed neural adapter"), so the
served rewriter is a fallback; and the **DPO stretch goal has no implementation** (grep for
`DPO`/`dpo` returns no source hits).

### 7. Utility: embedding + NLI (deberta-v3 cross-encoder) + task utility + human acceptance — **Satisfied**
Bidirectional NLI + cosine is implemented in
[`src/evaluation/utility.py`](../src/evaluation/utility.py) and
`BidirectionalUtilityEvaluator` (`src/evaluation/utility_metrics.py`), returning a
`UtilityReport` with forward/backward entailment, contradiction, and a degeneracy rule
(the `0.907` MiniLM failure from methodology-review.md finding 6 is the motivating case).
`cross-encoder/nli-deberta-v3-large` is the configured model (`configs/models.yaml`).
Human acceptance is captured in the user-study records
([`docs/userstudy/`](../docs/userstudy/), [`docs/experiment_protocol.md`](experiment_protocol.md)
§7.4 rewrite decision). Tested by `tests/test_utility_guardrail.py`.

### 8. Core eval metrics + multi-turn after 1/2/4/8 turns + comparison methods — **Partial**
The comparison harness exists: `run_comparative_benchmark` in
[`src/evaluation/run.py`](../src/evaluation/run.py) scores four methods
(raw / presidio / prompt_baseline / qlora_sft) with abstention-aware attacker metrics in
[`src/evaluation/privacy_metrics.py`](../src/evaluation/privacy_metrics.py), and a committed
output [`reports/comparative_benchmark_val.json`](../reports/comparative_benchmark_val.json).
Multi-turn cumulative leakage is implemented (`leakage.py`) and the study protocol defines a
staged multi-turn scenario ([`docs/experiment_protocol.md`](experiment_protocol.md) §16).
**Partial** because: the committed benchmark shows all attacker success rates at `0.0`
because it ran with the heuristic/fallback adversary (not held-out `mode: "model"`), and
there is **no fixed "after 1/2/4/8 turns" leakage curve** committed; session07.md explicitly
voids all prior attack-success numbers.

### 9. North-star: held-out leakage rate + utility guardrail — **Partial**
Defined and instrumented: north-star is "attribute leakage under a held-out Phi-4 adversary
on the test split, with a utility guardrail" ([`docs/lean-canvas.md`](lean-canvas.md) §6,
report frontmatter). The code records `attacker_mode`, `reportable_as_held_out`, and
`strict=True` gating in [`src/evaluation/run.py`](../src/evaluation/run.py) and
[`src/evaluation/attacker.py`](../src/evaluation/attacker.py). **Partial** because the number
itself is **not yet computed** — session07.md and session08.md both state the held-out run is
pending a trained adapter; the only historical figures are the closed-loop proxy deltas from
session04/05, which the reports label as not a privacy result.

### 10. Final comparison matrix — **Partial**
The matrix is produced by `run_comparative_benchmark` / `StagedEvaluator` and printed by
`main()` in [`src/evaluation/run.py`](../src/evaluation/run.py), with a committed
[`reports/comparative_benchmark_val.json`](../reports/comparative_benchmark_val.json) over 50
val samples. **Partial** because the committed matrix carries fallback-adversary zeros, not a
held-out result, and is on the val split rather than the final test split; a reportable final
matrix awaits the Session 08/09 held-out run.

### 11. Conversation-level privacy — **Satisfied**
Deterministic pseudonyms (`[PER_1]`, `[LOC_2]`, `[ORG_1]`, etc.) are generated by
`PersistentPseudonymManager.get_or_create` in
[`src/conversation/pseudonyms.py`](../src/conversation/pseudonyms.py) and
`DeterministicPseudonymManager` (entity_map); risk history and incremental leakage live in
`ConversationState.risk_history` / `cumulative_leakage_score` with monotonic Bayesian updating
([`src/conversation/leakage.py`](../src/conversation/leakage.py)). Session reset exists as
`reset_session`/`clear_session`. No-raw-logging is enforced: `process_turn` stores
`"[REDACTED_IN_MEMORY_ONLY]"` and production-readiness.md §6 documents hash-only audit
(`served_text_sha256`). Note the pseudonym token format is `[PER_1]`/`[LOC_1]` rather than the
doc's `PERSON_1`/`LOCATION_1` spelling — a cosmetic naming difference, not a capability gap.

### 12. Five-member ownership map — **Satisfied**
The five hats and owners are recorded consistently in every report frontmatter
([`reports/session04.md`](../reports/session04.md)–[`session08.md`](../reports/session08.md)):
Anna Thomas (Data&Eval), Chatur Bandaru (Engineering), Govind Arun (Product), Yiwei Jin
(Users&Research), Bingqi Lian (Operations); doc headers carry matching `Owner:` lines
(methodology-review, lean-canvas, production-readiness, scenario_cards). Ownership is tracked
in report/doc frontmatter rather than a dedicated `CODEOWNERS` file (none exists).

### 13. Integration contracts — **Satisfied**
Typed Pydantic contracts are centralised in
[`src/contracts/schemas.py`](../src/contracts/schemas.py) (`RiskReport`, `ProtectionResponse`,
`UtilityReport`, `ConversationState`, `EvaluationRecord`, `RewriteRequest/Response`). The
cross-module calls the doc specifies are realised in
[`src/pipeline/orchestrator.py`](../src/pipeline/orchestrator.py): `risk_estimator.predict_with_cues`
(risk_model.predict), `rewriter_service.protect` (rewriter.protect),
`leakage_tracker.compute_cumulative_leakage` (conversation.update), and the evaluate path in
[`src/evaluation/run.py`](../src/evaluation/run.py). Contract shape is tested by
`tests/test_challenger_stress_contracts.py` and `tests/test_schemas_ingestion.py`.

### 14. Roadmap sessions 4–12 — **Partial**
Sessions 4, 5, 7, 8 are documented with committed reports
([`reports/session04.md`](../reports/session04.md), `session05.md`, `session07.md`,
`session08.md`) plus session06 artifacts ([`artifacts/session06-doc.md`](../artifacts/session06-doc.md),
[`docs/userstudy/session06_record.json`](../docs/userstudy/session06_record.json)). **Partial**
because there is **no `reports/session06.md`** (session 06 lives only as artifacts), and
sessions 9–12 are future work with no committed deliverables yet. Each report has a concrete
"Next week's goal", so the roadmap is active but incomplete against the full 4–12 span.

### 15. Ethics — **Satisfied**
[`docs/lean-canvas.md`](lean-canvas.md) §7 covers data rights (SynthPAI CC-BY-NC-SA-4.0,
synthetic, non-commercial), no-real-PII in testing, bias as stereotype-shaped inference,
the "never say you are anonymous" claim boundary, and local retention. The "attacker could
infer" phrasing is enforced in code: `describe()` in
[`src/product/risk_bands.py`](../src/product/risk_bands.py) and tested by
`tests/test_risk_bands.py`. The study protocol's synthetic-only and zero-coaching rules are
in [`docs/experiment_protocol.md`](experiment_protocol.md) §22. Licence inventory is in
production-readiness.md §6.

### 16. Ablations A–D — **Partial**
Ablation A (direct vs. indirect+complicated hardness split) is implemented — `hardness_stats`
and the `hardness_breakdown` output in `StagedEvaluator.stage_3_adversary_probing`
([`src/evaluation/run.py`](../src/evaluation/run.py)), motivated by methodology-review.md
finding 8. Ablation B (prompt-only vs. fine-tuned rewriter) exists as
`RewriteMode.PROMPT_BASELINE` + [`src/rewriter/prompt_baseline.py`](../src/rewriter/prompt_baseline.py),
compared against `qlora_sft` in the benchmark. **Partial** because the named set A–D is not
fully enumerated or run: the A/B machinery is present but the committed benchmark used the
fallback adversary, and there is no committed artifact that labels and reports all of
ablations A, B, C, D with held-out numbers.

### 18. Definition-of-Done checklist — **Partial**
A production/DoD-style checklist exists as production-readiness.md §7 ("Production gap
checklist") and the Operations reportability gates
([`configs/experiments/session08.yaml`](../configs/experiments/session08.yaml),
[`scripts/validate_experiment.py`](../scripts/validate_experiment.py),
[`scripts/check_eval_readiness.py`](../scripts/check_eval_readiness.py)) which emit explicit
`READY`/`BLOCKED`/`PASS`/`DIAGNOSTIC_ONLY` statuses. **Partial** because, per session08.md,
the readiness gate currently reports **BLOCKED on "Student Adapter"** — the committed state
does not yet satisfy a held-out-reportable DoD; model packaging, versioning, monitoring, auth,
and session TTL are all listed as "production needs" not yet met.

### 19. Final thesis evidence chain — **Partial**
The pieces of the evidence chain are in place and honestly annotated: the premise
(PII redaction ≠ inferential privacy) is quantified from SynthPAI hardness fields
(methodology-review.md finding 8, 90.5% indirect/complicated); the risk model, calibration,
rewriter SFT, utility guardrail, and held-out-adversary harness all exist and are tested for
shape. **Partial** because the closing link — a trustworthy held-out leakage-reduction number
with a surviving utility guardrail — is explicitly **not yet produced** (session07.md "every
attack-success and NLI-utility number we have published is void"; session08.md north-star
"pending student model adapter training"). The thesis is fully scaffolded and instrumented but
not yet empirically closed by the project's own account.

---

## Documented divergences from the design doc (decisions, not defects)

All four are recorded in [`docs/methodology-review.md`](methodology-review.md) and should be
read as deliberate, justified choices:

1. **Rewriter model.** Doc names Qwen3.5-0.8B/2B; those are vision-language models. The team
   chose text-only `Qwen2.5-1.5B-Instruct` (0.5B on-device candidate). (finding 1;
   `configs/models.yaml`; `DEFAULT_BASE_MODEL`).
2. **Location handling.** Doc/earlier taxonomy froze US-region location classes; only ~7% of
   SynthPAI is US. The review recommends generative free-text city/country scoring instead of
   classification (finding 2). The code still ships a coarse location taxonomy
   (`src/risk_model/taxonomy.py`), so this divergence is partially adopted.
3. **Education buckets.** Local four-way bucket vs. the SynthPAI paper's
   high-school/college/master/PhD scale (finding 2).
4. **Closed-loop vs. held-out adversary.** The reported deltas to date come from the same
   estimator that guides the rewrite; the held-out Phi-4 run that would break the loop is
   implemented but not yet executed (finding 4; session07.md, session08.md).

## Stale / cleanup observations (for the improvements list, not graded above)

- `src/rewriter/prompt_baseline.py` still defaults `model_name="Qwen/Qwen3.5-0.8B"`, a model
  the methodology review flagged as wrong; it is overridden on load but is misleading.
- A legacy `src/risk_model/calibration.py` (`TemperatureScaler`, `compute_ece`,
  `generate_reliability_diagram`, `CalibrationMetrics`) is referenced **only** by tests
  (`tests/test_calibration.py`, `tests/test_e2e_pipeline.py`) and by no `src/` code; the
  production calibration path is `src/risk_model/calibrate.py`. Candidate for consolidation.
- `roberta-base` / `microsoft/deberta-v3-base` appear only as baseline candidates in
  `scripts/benchmark_risk_models.py`; neither is a trained production model.
- The pseudonym token format (`[PER_1]`/`[LOC_1]`) differs from the doc's `PERSON_1`/`LOCATION_1`.
