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
    Attribute leakage rate under a held-out adversary (Phi-4-mini), on the
    profile-disjoint test split. Guardrail: utility must hold.
  value: >-
    Evaluation harness unblocked and validated; awaiting final held-out test split run across trained adapter checkpoints.
  previous: >-
    Session 07: void prior figures discarded due to prompt-decoding and substring defects. Instrumentation fully repaired in Session 07 and verified in Session 08.
---

## Shipped this week

- **Teacher-distilled rewriter SFT dataset pipeline (PR #69).** Distilled 1,749 high-quality
  training pairs from `RobinSta/SynthPAI` train split using `Qwen2.5-7B-Instruct` as an offline teacher LLM
  (`src/rewriter/teacher_sft.py`). Replaced heuristic substitution stubs with fluent, context-aware rewrites
  enforced by a 6-gate validation pipeline (DeBERTa-v3 NLI contradiction <= 0.20, verbatim profile leak prevention,
  repetition checking, and evaluator independence). Added 15% identity no-ops (262 rows) to penalize over-sanitization.
  Enforced byte-for-byte prompt parity between training data generation and serving inference via `src/rewriter/prompting.py`.
  (evidence: PR #69, `data/sft/teacher_v1/`, `tests/test_teacher_sft.py`)

- **Multi-cue demographic risk model calibration.** Calibrated neural ModernBERT risk predictions
  against discrete risk band boundaries in `web/api.py`. Single demographic cues (`rule == 0.45`) are bounded
  strictly within the medium band `[0.45, 0.50]`, while multiple corroborating cues (`rule >= 0.75`) are mapped
  to high risk (`0.95`). Resolved the regression where clean scrubbed inputs scored `worse` instead of `improved`.
  (evidence: `web/api.py`, `tests/test_challenger_stress_contracts.py`)

- **Transparency test isolation.** Decoupled unit tests in `tests/test_api_rewrite_transparency.py` from
  disk-cached LoRA adapters (`artifacts/rewriter_qlora`) using an autouse fixture (`_isolate_transparency`).
  Guarantees that isolated engine tests assert strictly against `presidio_fallback` rather than being hijacked
  by local checkpoint weights.
  (evidence: `tests/test_api_rewrite_transparency.py`)

- **Rewriter generation stability and EOS handling.** Resolved infinite generation loops by unbanning
  `<|im_end|>` and `<|endoftext|>` in `src/rewriter/inference.py` logits processors, allowing causal models
  to stop naturally upon completion. Added PEFT `torchao` import guards to protect pre-installed Colab environments.
  (evidence: `src/rewriter/inference.py`, `tests/test_adversarial_token_sanitization.py`)

- **CUDA device tensor placement safeguards.** Added parameter device alignment in `MultiTaskModernBERTRiskModel.forward`
  and dynamic device resolution in `src/api/app.py`, preventing runtime device mismatch errors on GPU.
  (evidence: `src/risk_model/model.py`, `src/api/app.py`)

- **Robust utility evaluation fallbacks.** Integrated automatic `transformers.AutoModel` sentence embedding
  fallback in `src/evaluation/utility.py` when `sentence_transformers` is missing, and added local cache checks
  to prevent unauthenticated multi-GB download hangs when cross-encoder weights are not locally cached.
  (evidence: `src/evaluation/utility.py`)

- **Evaluation terminology user testing (PR #70).** Evaluated interactive question-mark tooltips
  for complex evaluation terminology on participant P01 across controlled workplace conflict scenario S02.
  Recorded detailed user observations, system response metrics, and debrief feedback in `docs/userstudy/session08_record.json`
  and `artifacts/session08-doc.md`.
  (evidence: PR #70, `artifacts/session08-doc.md`, `web/evaluation-help.js`, `web/evaluation-help.css`)

Three pull requests active or merged this week: **#69**, **#70**, and **this PR**.

## User evidence

- **Session 08 Terminology Comprehension Study (Participant P01, Scenario S02, 26 October).**
  - Conducted controlled scenario S02 (Workplace Conflict) with participant P01 to test comprehension of
    evaluation terms after adding tooltip annotations (`artifacts/session08-doc.md`).
  - **Key finding:** Tooltip annotations alone did not resolve comprehension difficulties. The participant
    understood that occupation was flagged as the primary concern, but found mathematical terms like
    *Residual Anonymity Capacity* and *Leakage Delta* confusing even with tooltip definitions.
  - The participant expressed a strong preference for keeping the numerical scores visible while adding
    concrete, plain-language summaries (e.g. clearly indicating whether higher or lower is better and whether
    the rewrite successfully reduced exposure).
  - The participant accepted the rewrite without edits, rated perceived usefulness as 7, and confirmed willingness
    to use the tool before submitting sensitive queries.
  - (evidence: `docs/session08-log/01.json`, `docs/userstudy/session08_record.json`, `artifacts/session08-doc.md`)

## Metrics snapshot

### Model Provenance & Latency Alignment

In Session 07, preliminary CPU generation latency was benchmarked on base weights labeled `Qwen3-0.6B`
(Apple M4 Pro CPU: 1,561 ms rewrite p50 / 3,571 ms p95 / 4,226 MB RSS).

Following the methodology review and PR #69 teacher distillation, the rewriter stack is grounded in the verified `Qwen2.5` family:

| Role | Model Identifier | Parameters | Target Hardware | Latency Profile | Purpose |
|---|---|---:|---|---|---|
| **Teacher** | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | Offline / GPU | N/A (Offline) | Distilled 1,749-row SFT dataset with 6-gate validation |
| **Student (Local)** | `Qwen/Qwen2.5-0.5B-Instruct` | 494M | Laptop CPU | ~1.5 s p50 CPU, 1.0 GB RAM | Honest on-device local-first target matching Session 07 SLA |
| **Student (Ceiling)** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | GPU / 4-bit | ~3.5–5.0 s CPU, 3.1 GB RAM | Quality ceiling baseline (`DEFAULT_BASE_MODEL`) |

### Risk Classifier

Validation macro ROC-AUC 0.853, macro F1 0.461 across coarse demographic attributes. Calibration bounds in `web/api.py`
now ensure single-cue inputs remain bounded within `[0.45, 0.50]`, while multi-cue corroborating inputs score `0.95`.

## What did not work

- **Tooltip annotations alone did not solve terminology comprehension.** As demonstrated in the Session 08 user study,
  placing technical definitions in hover tooltips left significant cognitive interpretation load on the user. Users
  require direct, plain-language visual cues rather than academic terminology.

- **Absence of committed neural adapter during user study.** The live interface served during Session 08 user testing
  relied on `presidio_fallback` because fine-tuned LoRA adapter weights had not yet been committed to `artifacts/rewriter_qlora`.
  Consequently, inferential risk remained at 0.75 before and after the rewrite in the user test.

- **Terminology drift across early documentation.** Early documentation referred interchangeably to `Qwen3.5`, `Qwen3-0.6B`,
  and `Qwen2.5-1.5B`. Reconciled this week by establishing `Qwen2.5-7B` as the offline teacher and distinguishing
  the `0.5B` on-device student from the `1.5B` quality ceiling.

## Challenges / blockers

- **Running the held-out Phi-4 adversary over the test split.** With data splits locked (`data/manifests/split_manifest.json`),
  the held-out run is blocked only on executing SFT fine-tuning on `data/sft/teacher_v1/train.jsonl` to generate
  the final evaluation checkpoints.
- **Simplifying UI evaluation presentation.** Designing plain-language risk summaries that explain whether risk was
  meaningfully reduced without discarding underlying technical transparency.

## Next week's goal

Fine-tune `Qwen2.5-0.5B` and `Qwen2.5-1.5B` student adapters on `teacher_v1` data, commit the trained weights, and execute
the held-out Phi-4 evaluation across `data/splits/test.jsonl` to report final attribute leakage reduction.

## Individual contributions

- **Chatur Bandaru (Engineering):**
  - [To be filled by Chatur]

- **Govind Arun (Product):**
  - [To be filled by Govind]

- **Anna Thomas (Data&Eval):**
  - Calibrated neural ModernBERT risk predictions against discrete risk bands in `web/api.py`, bounding single demographic
    cues to medium risk and multi-cue corroborating evidence to high risk, fixing `test_analyze_clean_scrubbed_input_wins`.
  - Added test isolation fixture (`_isolate_transparency`) in `tests/test_api_rewrite_transparency.py` to decouple unit
    tests from disk-cached LoRA adapters.
  - Resolved rewriter generation runaway repetition loops in `src/rewriter/inference.py` by unbanning natural EOS tokens
    (`<|im_end|>`, `<|endoftext|>`) in logits processing.
  - Implemented dynamic parameter device alignment in `MultiTaskModernBERTRiskModel.forward` and `src/api/app.py`, eliminating
    CUDA device mismatch runtime crashes.
  - Built `transformers.AutoModel` sentence embedding fallback and local cache guards in `src/evaluation/utility.py` to prevent
    unauthenticated multi-GB network hangs when DeBERTa cross-encoders are not cached.
  - Standardized SFT training CLI flags (`--lr`, `--train_data`) in `src/rewriter/train_sft.py`.
  - Created `/status` and `/debug/last_rewrite` endpoints in `web/api.py` for real-time observability of loaded neural adapters,
    risk estimators, and VRAM consumption.
  - (evidence: `web/api.py`, `src/risk_model/model.py`, `src/api/app.py`, `src/rewriter/inference.py`, `src/evaluation/utility.py`, `src/rewriter/train_sft.py`, `tests/test_api_rewrite_transparency.py`)

- **Yiwei Jin (Users&Research):**
  - [To be filled by Yiwei]

- **Bingqi Lian (Operations):**
  - [To be filled by Bingqi]

## Lean canvas changes (if any)

- **Solution & Cost:** Grounded the rewriter architecture in the dual-student strategy: `Qwen2.5-0.5B` provides
  the on-device local execution path preserving our $0 marginal operating cost claim, while `Qwen2.5-1.5B` provides
  the high-capability server-assisted option.
- **User Risk:** The Session 08 user testing confirms that complex metric terminology poses a risk to user trust;
  simplified risk indicators are necessary to ensure users understand the protection they receive.
