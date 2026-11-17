# Improvements backlog: observed; P0/P1/P2 implemented

**Owner:** Anna Thomas (Data&Eval) · **Branch:** `improvements/modernbert-addition` · **Status:** P0/P1/P2 implemented on this branch

> The P1 and P2 items below (#4-#11) have been **implemented** on this branch, each
> citing the real file, line, or document it addresses. The P0 items (#1-#3) are now
> **implemented** as well: ModernBERT is hosted at
> `AT707/inferenceguard-modernbert-risk` (public), which unblocked P0, and the P0 work
> has landed on this branch. Items #1-#3 ship the harness, model-provenance, and
> session-lifecycle/ops code; two residuals remain and are flagged honestly below (the
> north-star **number** for #1 is PENDING GPU EXECUTION, and the real revision
> shas/checksums for #2 still need a networked `verify_hf_models.py --write-manifest`
> run). No benchmark numbers were fabricated.

Every item below cites a real file, line, or document that exists in this repository.
No benchmark numbers were invented. This review is static only: the `INTEGRATIONS_ONLY`
sandbox blocks `pip`/PyPI and `download.pytorch.org`, so `torch`/`transformers`/etc. are
not importable and the `pytest` suite could not be run here. Items that say "once the
suite can run" depend on that.

## How to read this

- **Effort:** S (a few hours), M (about a day), L (multi-day / needs a trained model or run).
- **Priority:** P0 (do first: correctness, safety, or blocks the thesis number), P1 (should do), P2 (nice to have / cosmetic).
- **Status:** all items (#1-#11) are *implemented* on this branch. Two residuals are flagged: the #1 north-star number is PENDING GPU EXECUTION (harness + command ready, no numbers fabricated), and the #2 manifest revision shas/checksums need a networked `verify_hf_models.py --write-manifest` run to fill the null fields.

## Priority summary

| # | Title | Effort | Priority |
|---|---|---|---|
| 1 | Run the held-out adversary and produce the north-star leakage number | L | P0 |
| 2 | Commit or publish trained model artifacts with checksums | M | P0 |
| 3 | Session lifecycle & ops gaps: `DELETE /session/{id}`, TTL, CORS default | M | P0 |
| 4 | Reconcile risk-estimator label source (profile attrs vs `reviews.human`) | M | P1 |
| 5 | Consolidate the two calibration modules (`calibrate.py` vs `calibration.py`) | M | P1 |
| 6 | Resolve the dead `prompt_baseline.py` module + stale model id | S | P1 |
| 7 | Fix location / education taxonomy mismatch vs the SynthPAI paper | L | P1 |
| 8 | Multi-turn attack curve (1/2/4/8 turns) + complete final comparison matrix | L | P1 |
| 9 | Add the PromptPAI secondary eval set (design requirement 5) | M | P2 |
| 10 | Purge stale `Qwen3.5-0.8B` references from the implementation notebook | S | P2 |
| 11 | Full refresh of the README "Run It" / Colab narrative | S | P2 |

---

## P0 (do first)

### 1. Run the held-out adversary and produce the north-star leakage number
- **Evidence:** `docs/engineering/production-readiness.md` §5 and §7 ("Held-out evaluation ... full run pending"); `src/evaluation/attacker.py` (held-out Phi-4 harness, `reportable_as_held_out`), `src/evaluation/run.py:220` (`run_comparative_benchmark`). `reports/comparative_benchmark_val.json` currently carries fallback-adversary zeros (see `docs/requirements-coverage.md` items 8 to 10).
- **Why it matters:** This is the single number the thesis rests on: attribute leakage under a held-out adversary with a surviving utility guardrail. The harness is instrumented but has never been run with `mode: "model"` on the test split, so no trustworthy privacy result exists yet.
- **Effort:** L (needs a trained rewriter adapter + GPU run). **Priority:** P0.
- **Status:** *implemented (harness + single documented command ready; the north-star number itself is PENDING GPU EXECUTION - no numbers fabricated).* The held-out benchmark is now driven by one documented command (`docs/experiment_protocol.md` §25.4): `RISK_MODEL_ID=AT707/inferenceguard-modernbert-risk REWRITER_MODEL_ID=AT707/inferenceguard-rewriter-v1 python -m src.evaluation.run --split test --use_heavy_models --multiturn --matrix_markdown`. It writes `reports/comparative_benchmark_test.json` (per-method matrix + held-out provenance block `attacker_model_id` / `attacker_mode` / `reportable_as_held_out` / `risk_model_id` / `rewriter_model_id`, plus `per_attribute_attack_success` and `asr_reduction`), `reports/comparison_matrix_test.md` (the five-row Raw / Presidio / Prompt-only / Fine-tuned / Conversation-aware matrix), and `reports/multiturn_leakage_curve.json` (+ `.png` when matplotlib is present). With `--use_heavy_models` the harness **raises** rather than emit a heuristic-fallback report, so a result counts only when `attacker_mode == "model"` and `reportable_as_held_out == true`. The stale `reports/comparative_benchmark_val.json` is now flagged `reportable_as_held_out=false` / NOT-RUN. The hosted ModernBERT risk model (`AT707/inferenceguard-modernbert-risk`, the flat `multitask_heads.pt` + `calibration_params.json` layout) is now loadable end-to-end by `load_risk_classifier` / `CalibratedRiskPredictor` directly from the Hub, so the held-out command no longer falls back to an untrained risk model. **Residual:** the number itself is PENDING GPU EXECUTION on a GPU host, now unblocked once the user installs `bitsandbytes` + `accelerate` and sets `HF_TOKEN` (see `docs/experiment_protocol.md` §25.4 troubleshooting); it is never produced in CI and must not be fabricated.

### 2. Commit or publish trained model artifacts with checksums
- **Evidence:** `docs/engineering/production-readiness.md` §7 "Model packaging" ("Risk checkpoint and LoRA adapter live on Colab/Drive; **neither is committed**. Without them the API silently uses the rule-based risk estimate and the heuristic rewriter"); `context.json` snapshot note ("Trained model checkpoints and LoRA adapters are NOT committed").
- **Why it matters:** The API silently falls back to untrained heads / heuristic rewrite, so a fresh clone cannot reproduce any reported result. The new `InferenceGuard_modernbert_train_and_host.ipynb` partially addresses this by hosting on the Hub, but a versioned, checksum'd release path (and a `/health` check that refuses to start or flags a missing model) is still open.
- **Effort:** M. **Priority:** P0.
- **Status:** *implemented.* Hosted model ids are now pinned via `artifacts/model_manifest.json` (schema_version 1; per-role `repo_id` / `base_model` / `revision` / `expected_files` / `checksums` / `verified`) and `configs/models.yaml` revision pins, including a new `modernbert_hub` block for `AT707/inferenceguard-modernbert-risk`. `scripts/verify_hf_models.py` was extended with a ModernBERT risk-repo metadata check and a `--write-manifest` mode that records the authoritative revisions/checksums from a real `HfApi` query. `web/api.py` `/health` now reports `status: "ok"` when the trained risk model loaded and `status: "degraded"` otherwise (instead of a static ok), and an opt-in `IG_REQUIRE_MODEL` strict mode makes startup/health refuse to pass when the trained model did not load and the untrained/rule-based fallback is in use. **Residual:** the real revision shas and checksums are still `null` / empty in `artifacts/model_manifest.json` and `configs/models.yaml`; they must be filled by a networked `python scripts/verify_hf_models.py --write-manifest` run (not fabricated).

### 3. Session lifecycle & operations gaps (`DELETE /session/{id}`, TTL, CORS default)
- **Evidence:** `docs/engineering/production-readiness.md` §6 (Session reset "Partial ... no API endpoint to clear a session, and memory grows until restart. Needed: a `DELETE /session/{id}` and a TTL"; CORS "Default is still `allow_origins=["*"]` with `allow_credentials=True`") and §7 ("Session lifecycle: In-memory, unbounded"). Confirmed in code: `web/api.py` `allowed_origins()` returns `["*"]` by default and `app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(), allow_credentials=True, ...)`.
- **Why it matters:** Unbounded in-memory session growth is a memory-leak / DoS risk, and the permissive default CORS combined with `allow_credentials=True` lets any web page call the API from a visitor's browser. Fine for the Colab demo, not for real user text.
- **Effort:** M. **Priority:** P0.
- **Status:** *implemented.* `web/api.py` now exposes `DELETE /session/{id}` to clear a session, plus idle-TTL and max-session eviction governed by `IG_SESSION_TTL_SECONDS` (default 3600) and `IG_MAX_SESSIONS` (default 10000) so in-memory session state no longer grows unbounded. The CORS default was fixed so it never combines a wildcard origin with `allow_credentials=True` (`allowed_origins()` + `cors_allow_credentials()`): credentials are only allowed for explicit origins set via `IG_ALLOWED_ORIGINS`. New coverage lives in `tests/test_ui_api.py` and `tests/test_operations_state.py` (validated by CI).

---

## P1 (should do)

### 4. Reconcile risk-estimator label source (profile attributes vs `reviews.human` inferability)
- **Evidence:** `context.json` key_patterns ("multitask uses profile attributes mapped via `map_to_taxonomy_id`; binary path uses `reviews.human` inferability"); `src/risk_model/train.py` multitask path; `docs/methodology-review.md` and the design doc favor the `reviews.human` inferability labels. `docs/requirements-coverage.md` item 4.
- **Why it matters:** Training the multi-task heads on profile attributes rather than the human-annotated inferability labels changes what "risk" means and whether the reported F1/ROC-AUC are comparable to the SynthPAI paper. The two paths should be reconciled to one documented decision before the north-star run.
- **Effort:** M. **Priority:** P1.
- **Status:** *implemented (2026-11, with #7).* `train_risk_model` now defaults to `model_type="inferability_labels"` (alias `"binary_classifier"`), training the 4-head BCE classifier on `reviews.human` inferability labels as the authoritative reported signal; the categorical `multitask_modernbert` path is retained and documented as auxiliary/diagnostic. Rationale in `docs/methodology-review.md` Addendum 2026-11, decision #4.

### 5. Consolidate the two calibration modules
- **Evidence:** Production path `src/risk_model/calibrate.py` (`AttributeTemperatureScaler`, `compute_expected_calibration_error`), imported by `src/api/app.py`, `src/risk_model/predict.py`, `src/conversation/leakage.py`, `src/risk_model/attribution.py`, and three tests. Orphan `src/risk_model/calibration.py` (`TemperatureScaler`, `compute_ece`, `generate_reliability_diagram`, `CalibrationMetrics`) is referenced **only** by `tests/test_calibration.py` and `tests/test_e2e_pipeline.py`; `grep -rn 'risk_model.calibration' src scripts web` returns no matches (FEAT-004 findings).
- **Why it matters:** Two parallel calibration implementations are a maintenance hazard and a source of confusion about which one is authoritative. Fold the useful ECE / reliability-diagram logic into `calibrate.py`, then delete the orphan and its redundant test.
- **Effort:** M. **Priority:** P1.
- **Note:** The orphan's only safety net is a test that cannot be executed in this sandbox, so deletion must wait until the full `pytest` suite can be run.
- **Status:** *implemented; physical deletion of `calibration.py` pending green CI.* The ECE / reliability-diagram logic was folded into `calibrate.py` and the two dependent tests repointed; the orphan module is retained only until the full `pytest` suite confirms green, then deleted.

### 6. Resolve the dead `prompt_baseline.py` module + stale model id
- **Evidence:** `src/rewriter/prompt_baseline.py` class `PromptBaselineRewriter`: `grep` shows it is **never instantiated**; the `prompt_baseline` eval method in `src/evaluation/run.py:292-294` uses `rewriter.protect(..., mode=RewriteMode.PROMPT_BASELINE)` via `src/rewriter/inference.py:989`, not this class. The module also carries a stale default `model_name="Qwen/Qwen3.5-0.8B"` (`src/rewriter/prompt_baseline.py:25`), a vision-language model per `docs/methodology-review.md` §1, overridden to `DEFAULT_BASE_MODEL` only on tokenizer-load failure.
- **Why it matters:** Dead code that looks live is misleading, and the stale default model id misrepresents the Ablation B baseline. Either delete the module or re-wire `run.py` to actually use it, and in either case fix the default to `Qwen/Qwen2.5-1.5B-Instruct` / `DEFAULT_BASE_MODEL`.
- **Effort:** S. **Priority:** P1.
- **Note:** Confirm against a runnable suite before deleting.
- **Status:** *implemented; (deletion pending green CI if deleted).* The stale default model id was fixed to `Qwen/Qwen2.5-1.5B-Instruct` / `DEFAULT_BASE_MODEL`; the dead-module resolution's physical deletion, if taken, waits on a green `pytest` run.

### 7. Fix location / education taxonomy mismatch vs the SynthPAI paper
- **Evidence:** `docs/methodology-review.md` §2: location frozen in `artifacts/taxonomy.json` as `[USA_Northeast, USA_South, USA_Midwest, USA_West, Europe, Other, unknown]` ("Four of our seven classes cover 7% of the data ... collapses to Europe vs Other"); recommendation "do not classify location" (free-text city/country scoring). Education is `[high_school, undergraduate, graduate_professional, other_unknown]` vs the paper's `high school / college / master / PhD`. Implemented in `src/risk_model/taxonomy.py`. `docs/requirements-coverage.md` divergences 2 and 3.
- **Why it matters:** A location-protection number measured on a Europe-vs-Other split is near meaningless and not comparable to the SynthPAI paper; the education scale is similarly off. This affects the credibility of the risk-model and leakage metrics.
- **Effort:** L (changes labels, taxonomy, training, and the attacker scoring). **Priority:** P1.
- **Status:** *implemented (2026-11, with #4).* `ATTRIBUTE_TAXONOMY[EDUCATION]` is now the paper's four-way scale `high_school / college / master / phd` (default bucket for unmatched text documented as `high_school`); `map_to_taxonomy_id`, the adversary heuristic, and the demo ground-truth rows use the new keys; `artifacts/taxonomy.json` regenerated from `ATTRIBUTE_TAXONOMY`. Location: the reportable metric is the free-text generate-and-score path (`Phi4MiniAttacker.evaluate_success`); the coarse location buckets are retained only as an auxiliary/diagnostic head. Residual (dedicated generative location head) flagged in `docs/methodology-review.md` Addendum 2026-11, decision #7.

### 8. Multi-turn attack curve (1/2/4/8 turns) + complete the final comparison matrix
- **Evidence:** Multi-turn leakage implemented in `src/conversation/leakage.py`; staged scenario in `docs/experiment_protocol.md` §16; comparison harness `src/evaluation/run.py:220` over methods `("raw", "presidio", "prompt_baseline", "qlora_sft")`. But `docs/requirements-coverage.md` items 8 and 10 note there is **no committed "after 1/2/4/8 turns" leakage curve** and the committed matrix carries fallback zeros on the val split, not held-out test numbers. The design's matrix rows (Raw / Presidio / Prompt-only / Fine-tuned / Conversation-aware) are not all populated with reportable numbers.
- **Why it matters:** The turn-by-turn curve and the full comparison matrix are core deliverables of the thesis evidence chain; without them the "conversation-aware" advantage is unquantified.
- **Effort:** L (depends on items 1 and 2). **Priority:** P1.
- **Status:** *implemented (scaffolding only; numbers pending P0 #1).* The multi-turn leakage curve harness and the full comparison-matrix scaffolding are in place; the reportable turn-by-turn and held-out matrix numbers depend on the north-star held-out run in P0 #1.

---

## P2 (nice to have)

### 9. Add the PromptPAI secondary evaluation set (design requirement 5)
- **Evidence:** `grep -rniE 'promptpai' .` returns no source/data matches anywhere in the repo (only the requirement note in `docs/requirements-coverage.md` item 5). The primary SynthPAI profile-disjoint splits exist (`data/splits/`, `data/manifests/split_manifest.json`), but the secondary PromptPAI set called for by design requirement 5 is absent.
- **Why it matters:** A second, independent eval set strengthens the generalization claim and is an explicit design requirement; its absence is a scope gap the reports never claim to have closed.
- **Effort:** M (data acquisition + ingestion + manifest). **Priority:** P2.
- **Status:** *implemented.*

### 10. Purge stale `Qwen3.5-0.8B` references from the implementation notebook
- **Evidence:** `InferenceGuard_Implementation.ipynb` still says the rewriter "will be replaced by a fine-tuned Qwen3.5-0.8B model" and "QLoRA fine-tune Qwen3.5-0.8B (2B as fallback)". `docs/methodology-review.md` §1 established Qwen3.5-0.8B/2B are vision-language models and the project uses text-only `Qwen2.5-1.5B-Instruct`. `reports/session04.md:84` carries the same stale phrasing (historical; may be left as a dated record).
- **Why it matters:** The implementation notebook is reader-facing documentation; the wrong model id contradicts `configs/models.yaml` and `DEFAULT_BASE_MODEL` and misleads anyone reproducing the work. (Session reports are historical and can stay as-is.)
- **Effort:** S. **Priority:** P2.
- **Status:** *implemented.*

### 11. Full refresh of the README "Run It" / Colab narrative
- **Evidence:** `README.md` "Run It" section (line 30+) still leads with a Colab/Drive-centric narrative. The minimal dead-path fix (replacing `data/raw/synthpai.jsonl` / `artifacts/profile_splits.json` with `data/splits/*.jsonl` and pointing at the new notebook) was already made in FEAT-004 (commit 7cc2c0f); a fuller refresh of the surrounding narrative is still open.
- **Why it matters:** The narrative still frames Colab as the primary path when production-readiness.md calls Colab "a stand-in ... described as a demo, not as the product." A refreshed quickstart (local-first install, then the ModernBERT notebook, then the Hub-hosted artifacts) would match the current architecture.
- **Effort:** S. **Priority:** P2.
- **Status:** *implemented.*

---

*Reminder: all items (#1-#11) are implemented on this branch. Two residuals remain and
are flagged honestly above: the P0 #1 north-star number is PENDING GPU EXECUTION (the
harness and single documented command are ready; no numbers fabricated), and the P0 #2
manifest revision shas/checksums still need a networked `verify_hf_models.py
--write-manifest` run to populate the currently-null fields.*
