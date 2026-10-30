---
team: InferenceGuard
session: 08
date: 2026-10-27
members:
  - name: Anna Thomas
    github: AnnaThomas2060
    hat: Data&Eval
  - name: Chatur Bandaru
    github: chaturbandaru
    hat: Engineering
  - name: Govind Arun
    github: GIND123
    hat: Product
  - name: Yiwei Jin
    github: runyoucleverdr-beep
    hat: Users&Research
  - name: Bingqi Lian
    github: lianbingqi
    hat: Operations
north_star:
  metric: >-
    End-to-end inferential privacy protection: Attribute leakage reduction under a held-out adversary (Phi-4-mini)
    on the profile-disjoint test split, constrained by utility retention (NLI contradiction <= 0.20, cosine similarity >= 0.70)
    and zero explicit PII leakage.
  value: >-
    Pipeline contracts operational end-to-end across all stages (detection, redaction, rewriting, rescoring, and utility validation).
    Evaluator verified across 67 unit and stress contracts with 100% pass rate. Current baseline achieves 0 hard PII leaks,
    0.973 utility preservation on fallback, and unblocked held-out adversary execution pending student model adapter training.
  previous: >-
    Session 07: Instrumentation repaired; all prior attacker metrics voided and discarded due to substring and prompt decode defects.
---

## Shipped this week

- **Teacher-distilled rewriter SFT dataset pipeline (PR #69).** Distilled 1,749 verified
  training pairs from `RobinSta/SynthPAI` train split using `Qwen2.5-7B-Instruct` as an offline teacher LLM
  (`src/rewriter/teacher_sft.py`). Replaced heuristic stubs with fluent, context-aware rewrites enforced by a
  6-gate validation pipeline: DeBERTa-v3 NLI contradiction <= 0.20, verbatim profile leak prevention, repetition
  checking, and evaluator independence. Added 15% identity no-ops (262 rows) to penalize over-sanitization.
  Enforced byte-for-byte prompt parity between training data generation and serving inference via `src/rewriter/prompting.py`.

- **Multi-cue demographic risk model calibration.** Calibrated neural ModernBERT risk predictions
  against discrete risk band boundaries in `web/api.py` (`make_risk_estimator`). Single demographic cues (`rule == 0.45`)
  are bounded strictly within the medium band `[0.45, 0.50]`, while multiple corroborating cues (`rule >= 0.75`) are mapped
  to high risk (`0.95`). Resolved the regression where clean scrubbed inputs scored `worse` instead of `improved`.

- **Transparency test isolation.** Decoupled unit tests in `tests/test_api_rewrite_transparency.py` from
  disk-cached LoRA adapters (`artifacts/rewriter_qlora`) using an autouse fixture (`_isolate_transparency`).
  Guarantees that isolated engine tests assert strictly against `presidio_fallback` rather than being hijacked
  by local checkpoint weights.

- **Rewriter generation stability and EOS handling.** Resolved infinite generation loops by unbanning
  `<|im_end|>` and `<|endoftext|>` in `src/rewriter/inference.py` logits processors (`_get_banned_token_ids`), allowing
  causal models to stop naturally upon completion. Added PEFT `torchao` import guards to protect pre-installed
  Colab environments.

- **CUDA device tensor placement safeguards.** Added parameter device alignment in `MultiTaskModernBERTRiskModel.forward`
  (`src/risk_model/model.py`) and dynamic device resolution in `src/api/app.py`, preventing runtime device mismatch errors on GPU.

- **Robust utility evaluation fallbacks.** Integrated automatic `transformers.AutoModel` sentence embedding
  fallback in `src/evaluation/utility.py` when `sentence_transformers` is missing, and added local cache checks
  to prevent unauthenticated multi-GB download hangs when cross-encoder weights are not locally cached.

- **Diagnostic observability endpoints.** Implemented `/status` and `/debug/last_rewrite` endpoints in `web/api.py`
  providing runtime inspection of active neural adapters, loaded risk backbones, device placement, and VRAM memory allocation.

- **Evaluation terminology user testing (PR #70).** Evaluated interactive question-mark tooltips
  for complex evaluation terminology on participant P01 across controlled workplace conflict scenario S02.
  Recorded detailed user observations, system response metrics, and debrief feedback in `docs/userstudy/session08_record.json`
  and `artifacts/session08-doc.md`.

## User evidence

- **Session 08 Terminology Comprehension Study (Participant P01, Scenario S02, 26 October).**
  - Conducted controlled scenario S02 (Workplace Conflict) with participant P01 to test comprehension of
    evaluation terms after adding tooltip annotations (`artifacts/session08-doc.md`).
  - **Key findings from session logs (`docs/session08-log/01.json`):**
    - The participant understood that occupation was flagged as the primary concern (`primary_attr: "occupation"`),
      but reported difficulty interpreting mathematical labels such as *Residual Anonymity Capacity* (recorded value `2.70`)
      and *Leakage Delta* (`0.75`), even after reading tooltip definitions.
    - System output: input risk was 0.75 (HIGH), Presidio replaced one explicit date span ("seven years" -> `<DATE_TIME>`),
      and the served rewrite was `presidio_fallback` (`rewrite_changed: true`, `is_win: false`, `inferential_risk: 0.75`).
    - Lexical utility scored 0.9787 cosine similarity, with composite utility at 0.9733. Total backend latency was 192.5 ms.
    - The participant expressed a strong preference for keeping the numerical scores visible while adding
      concrete, plain-language summaries (e.g. clearly indicating whether higher or lower is better and whether
      the rewrite successfully reduced exposure).
    - The participant accepted the rewrite without edits, rated perceived usefulness as 7, and confirmed willingness
      to use the tool before submitting sensitive queries.

## Metrics snapshot

### Model Provenance & Latency Alignment

In Session 07, preliminary CPU generation latency was benchmarked on base weights labeled `Qwen3-0.6B`
(Apple M4 Pro CPU: 1,561 ms rewrite p50 / 3,571 ms p95 / 4,226 MB RSS, recorded in `artifacts/latency/cpu-apple-m4-pro_2026-10-17_qwen3-0.6b.json`).

Following the methodology review (`docs/methodology-review.md`) and PR #69 teacher distillation (`src/rewriter/teacher_sft.py`),
the rewriter stack is grounded in the verified `Qwen2.5` family (`configs/models.yaml`):

| Role | Model Identifier | Parameters | Target Hardware | Latency Profile | Purpose |
|---|---|---:|---|---|---|
| **Teacher** | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | Offline / GPU | N/A (Offline) | Distilled 1,749-row SFT dataset with 6-gate validation (`data/sft/teacher_v1/`) |
| **Student (Local)** | `Qwen/Qwen2.5-0.5B-Instruct` | 494M | Laptop CPU | ~1.5 s p50 CPU, 1.0 GB RAM | Honest on-device local-first target matching Session 07 SLA |
| **Student (Ceiling)** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | GPU / 4-bit | ~3.5–5.0 s CPU, 3.1 GB RAM | Quality ceiling baseline (`DEFAULT_BASE_MODEL` in `src/rewriter/inference.py`) |

### Risk Classifier

Validation macro ROC-AUC 0.853, macro F1 0.461 across coarse demographic attributes (`src/risk_model/train.py`, `tests/test_calibration_session07.py`).
Calibration bounds in `web/api.py` now ensure single-cue inputs remain bounded within `[0.45, 0.50]`, while multi-cue corroborating
inputs score `0.95`.

### Data Splits Hygiene

Cryptographic SHA-256 manifests (`data/manifests/split_manifest.json`) lock 210 train profiles (5,232 comments),
45 val profiles (1,416 comments), and 45 test profiles (1,175 comments) with zero profile contamination (`src/data/splits.py`, `tests/test_splits.py`).

## What did not work

- **Tooltip annotations alone did not solve terminology comprehension.** As demonstrated in the Session 08 user study
  (`artifacts/session08-doc.md`), placing technical definitions in hover tooltips left significant cognitive interpretation load
  on the user. Users require direct, plain-language visual indicators rather than academic terminology.

- **Absence of committed neural adapter during user study.** The live interface served during Session 08 user testing
  relied on `presidio_fallback` because fine-tuned LoRA adapter weights had not yet been committed to `artifacts/rewriter_qlora`.
  Consequently, inferential risk remained at 0.75 before and after the rewrite in the user test (`docs/session08-log/01.json`).

- **Terminology drift across early documentation.** Early documentation referred interchangeably to `Qwen3.5`, `Qwen3-0.6B`,
  and `Qwen2.5-1.5B`. Reconciled this week by establishing `Qwen2.5-7B` as the offline teacher and distinguishing
  the `0.5B` on-device student from the `1.5B` quality ceiling in `configs/models.yaml`.

## Challenges / blockers

- **Running the held-out Phi-4 adversary over the test split.** With data splits locked (`data/manifests/split_manifest.json`),
  the held-out run is blocked only on executing SFT fine-tuning on `data/sft/teacher_v1/train.jsonl` to generate
  the final evaluation checkpoints.
- **Simplifying UI evaluation presentation.** Designing plain-language risk summaries in `web/index.html` that explain whether
  risk was meaningfully reduced without discarding underlying technical transparency.

## Next week's goal

Fine-tune `Qwen2.5-0.5B` and `Qwen2.5-1.5B` student adapters on `teacher_v1` data, commit the trained weights, and execute
the held-out Phi-4 evaluation across `data/splits/test.jsonl` to report final attribute leakage reduction.

## Individual contributions

- **Chatur Bandaru (Engineering):**
  - [To be filled by Chatur]

- **Govind Arun (Product):**
  - [To be filled by Govind]

- **Anna Thomas (Data&Eval):**
  - **Multi-Cue Neural Risk Calibration (`web/api.py`, `tests/test_challenger_stress_contracts.py`, `tests/test_attribution.py`, `tests/test_pipeline_e2e.py`):** Calibrated ModernBERT
    predictions against discrete risk bands, bounding isolated single cues (`0.45`) to Medium risk and multi-cue corroborating
    evidence (`0.75+`) to High risk (`0.95`), resolving the `test_analyze_clean_scrubbed_input_wins` contract regression.
  - **Rewriter Generation Stability & Token Loop Mitigation (`src/rewriter/inference.py`, `tests/test_adversarial_token_sanitization.py`):**
    Eliminated infinite generation runaway loops by unbanning natural EOS stop tokens (`<|im_end|>`, `<|endoftext|>`) in rewriter
    logits processing, and patched PEFT `torchao` import dispatch to ensure runtime stability across Colab GPU environments.
  - **GPU Device Placement, Attributor Fallback & Utility Hardening (`src/risk_model/model.py`, `src/api/app.py`, `src/risk_model/attribution.py`, `src/evaluation/utility.py`):**
    Resolved CUDA device tensor mismatch crashes via dynamic parameter-device alignment; fixed uninitialized `NoneType` tokenizer
    crashes in `SyntaxAlignedCueAttributor.extract_cues` with automatic heuristic cue fallback; and implemented a standalone
    `transformers.AutoModel` sentence embedding fallback and local cache checks to prevent multi-GB unauthenticated network hangs.
  - **API Observability & Test Fixture Isolation (`web/api.py`, `tests/test_api_rewrite_transparency.py`, `src/rewriter/train_sft.py`):**
    Engineered `/status` and `/debug/last_rewrite` diagnostic endpoints for live inspection of neural weights and VRAM; introduced
    test fixture isolation (`_isolate_transparency`) to decouple local disk-cached LoRA adapters from unit test suites; and
    standardized SFT training CLI flags (`--lr`, `--train_data`).

- **Yiwei Jin (Users&Research):**
  - **UI improvements (PR #65):** Added question-mark tooltips to eight evaluation metrics and two comparison headings, providing explanations of technical terms such as Residual Anonymity Capacity, Leakage Delta, and Persistent Pseudonyms.
  - **UI usability testing (PR #70):** Conducted Session 08 to assess users’ understanding of the evaluation interface. The participant wanted access to numerical results but found the terminology difficult even with tooltips. Findings and recommended simpler labels and visible explanations are documented, alongside a proposal for user-adjustable protection levels for age, location, education, and occupation.

- **Bingqi Lian (Operations):**
  - **Bingqi Lian (Operations):**
  - **Evaluation readiness and artifact-provenance gate (PR #72):** Extended the Session 08 Operations framework from deciding whether completed results are reportable to deciding whether the underlying trained artifact chain is ready for formal held-out evaluation.
  - Added provenance checks for the teacher SFT dataset, train/test split hashes, active student base model, student adapter artifacts, held-out attacker configuration, and utility-model configuration.
  - Added student-adapter registration support that records the exact base model, SFT dataset SHA-256, teacher-manifest SHA-256, adapter-file hashes, and associated git commit in a reproducible training manifest.
  - Implemented explicit `READY` / `BLOCKED` preflight status so diagnostic development can continue without allowing incomplete artifacts to produce headline held-out claims. On the current repository state, all readiness stages pass except Student Adapter, correctly blocking formal evaluation until a trained adapter is registered.
  - Added readiness and registration regression coverage. Final validation passed 11 readiness tests, 32 existing Operations tests, and 479 repository tests with 2 skipped; `git diff --check` was clean.
  - (evidence: PR #72, `configs/experiments/session08.yaml`, `scripts/check_eval_readiness.py`, `tests/test_eval_readiness.py`)

## Lean canvas changes (if any)

- **Solution & Cost:** Grounded the rewriter architecture in the dual-student strategy: `Qwen2.5-0.5B` provides
  the on-device local execution path preserving our $0 marginal operating cost claim, while `Qwen2.5-1.5B` provides
  the high-capability server-assisted option (`docs/lean-canvas.md`).
- **User Risk:** The Session 08 user testing confirms that complex metric terminology poses a risk to user trust;
  simplified risk indicators are necessary to ensure users understand the protection they receive (`artifacts/session08-doc.md`).
