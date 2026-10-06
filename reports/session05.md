---
team: InferenceGuard
session: 05
date: 2026-09-29
members:
  - name: Anna Thomas
    github: AnnaThomas2060
    hat: Data&Eval
  - name: Chatur Bandaru
    github: chaturbandaru
    hat: Engineering
  - name: Govind
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
    Target: attribute leakage rate under a held-out adversary. Proxy:
    overall risk delta, taken as the max across attributes, with util_sim
    as the utility guardrail.
  value: >-
    ModernBERT 4-head risk classifier, measured on the held-out validation split
    (45 profile-disjoint profiles), best epoch 3 of 3: macro F1 0.461, macro PR-AUC
    0.457, macro ROC-AUC 0.853, macro balanced accuracy 0.750. Per attribute
    (F1 / PR-AUC / validation positives): occupation 0.618 / 0.612 / 281,
    education 0.534 / 0.564 / 146, location 0.462 / 0.484 / 56, age 0.232 / 0.169 / 83.
    The checkpoint was selected on this same validation split, so treat it as an
    optimistic estimate; the test split is untouched. However, the overall risk delta
    target is pending. The Qwen3-1.7B LLM needs to be fine-tuned
    (Session 06) for meaningful privacy rewrites to measure actual risk reduction at scale.
  previous: >-
    Proxy only, 3 dev-test messages in data/logs/session04_usage_log.csv:
    overall confidence delta 0.428 / 0.010 / 0.056; util_sim 0.916 / 0.907 /
    0.827. Team-generated, so not user evidence.
---

## Shipped this week
- **Codebase Modularization**: Migrated the monolithic notebook into a proper Python architecture (`src/`, `web/`, `tests/`), unblocking CI/CD and independent feature development.
- **ModernBERT Risk Classifier Fine-tuning**: Successfully trained the ModernBERT-base 4-head risk classifier on the SynthPAI dataset. Handled highly imbalanced labels with strong positive class weighting.
- **SFT Data Generation Pipeline**: Built a Pareto Rejection Sampling pipeline to generate Supervised Fine-Tuning data for the Qwen3-1.7B rewriter. It successfully processed 7823 records, strictly filtering for train split profiles to guarantee zero test leakage, and exported high-confidence candidates to ChatML and Alpaca schemas.
- **FastAPI Web UI with Localtunnel**: Developed a lightweight, interactive FastAPI web interface. Bypassed aggressive Colab ngrok bans by implementing dynamic Localtunnel routing, allowing real-time model interaction powered by the live A100 GPU.
- **Adversarial Evaluation Harness**: Designed the evaluation harness utilizing Phi-4-mini to act as a zero-shot attribute guesser, alongside DeBERTa-v3 Bidirectional NLI for semantic preservation checking.

## User evidence
- P05 - P09 user tests were carried out before web UI was merged. Session05 user test showed flaws in test design. 

## Metrics snapshot
- Risk Classifier (ModernBERT), **validation split** (45 held-out profiles, profile-disjoint, seed 42), best epoch 3 of 3. `best_macro_f1` in `train_risk_model` is the best per-epoch **validation** macro F1, not a training-split number:

  | attribute | val F1 | val PR-AUC | val ROC-AUC | val positives |
  |---|---|---|---|---|
  | occupation | 0.618 | 0.612 | 0.870 | 281 |
  | education | 0.534 | 0.564 | 0.883 | 146 |
  | location | 0.462 | 0.484 | 0.907 | 56 |
  | age | 0.232 | 0.169 | 0.753 | 83 |
  | **macro** | **0.461** | **0.457** | **0.853** | — |

  Positive class weights used in training: location 20.4×, age 14.1×, education 9.0×, occupation 3.4×. Macro balanced accuracy 0.750. Two honest caveats: the saved checkpoint is the epoch with the highest validation macro F1, so the same split was used for selection and reporting, and validation loss rose across epochs (3.391 → 3.424 → 4.009) while macro F1 improved. The test split has not been scored. Evidence: `InferenceGuard_finetune.ipynb`, output of the *Finetune ModernBERT* cell.
- SFT Data Generation: Processed 7823 total records. Retained 5232 strict training split records. Evaluated 5244 heuristic candidates, and exported 27 high-confidence pareto-optimal samples (low output is expected due to strict heuristic rules).
- Measured on: The `RobinSta/SynthPAI` dataset.
- Is this the same model that is running in the product? Yes, the fine-tuned `ModernBertRiskClassifier` is now dynamically loaded into the live `web/api.py` FastAPI backend. The rewriter is currently falling back to a heuristic rule system until Qwen is trained.

## What did not work
- Running `uvicorn` directly in Google Colab crashed due to `asyncio` event loop conflicts. We resolved this by spawning the web server in a background shell process.
- Free ngrok accounts aggressively banned Colab VM traffic due to automated anti-phishing filters. We mitigated this by migrating the UI tunneling entirely to `localtunnel`.
- The strict 0.82 cosine similarity threshold for the token-based heuristic evaluator rejected almost all generated SFT candidates. We explicitly lowered it to 0.50 to successfully generate the dataset.

## Challenges / blockers
- GPU memory pressure continues to be a bottleneck when attempting to load ModernBERT, Qwen, and the DeBERTa NLI cross-encoder into VRAM simultaneously for the SFT generation pipeline.
- We require real users to test the UI flow. The Qwen LLM needs to be fine-tuned for meaningful privacy rewrites, but users can still test out how their inputs evaluate against the live privacy risk evaluation model.

## Next week's goal
- Fine-tune the Qwen3-1.7B LLM on the generated SFT dataset using QLoRA.
- Run the Phi-4-mini adversarial evaluation harness over the validation set to quantify the empirical privacy-utility tradeoff.

## Individual contributions
- Anna Thomas (Data&Eval):
    - Migrated the codebase from a monolithic Jupyter Notebook to a fully modularized Python project architecture (`src/`, `web/`, `tests/`).
    - Fixed stringified JSON profile parsing in the SynthPAI dataset loader to prevent complete dataset rejection during training data filtering.
    - Diagnosed and fixed the Colab Jupyter kernel caching trap that was preventing updated Python modules from executing during SFT generation.
    - Resolved the `uvicorn` and `asyncio` conflict in Colab by implementing a background shell launcher, and migrated the tunneling infrastructure from Ngrok to Localtunnel.
    - Updated the `web/api.py` endpoint to dynamically load the fine-tuned `ModernBertRiskClassifier` weights from the `best_model` subdirectory into the GPU for live inference.
    - Built a dedicated Info Callout in the UI to clearly explain the underlying model architecture (ModernBERT vs Heuristic Fallback).

- Govind (Product):
    - PR #29 (merged 2026-09-27): fitted the user-facing risk band edges from validation data, per attribute, instead of hand-picked cutoffs (`src/product/thresholds.py`, `tests/test_thresholds.py`).
    - PR #30 (merged 2026-09-27): composed the privacy report contract with the honesty rules enforced in code (`src/product/report.py`, `tests/test_report.py`).
    - Reviewed PR #28 in full and requested changes (2026-09-27), raising five blockers, including a committed benchmark file that was simulation output and the split mislabelling corrected in this report's north star.

- Chatur Bandaru (Engineering):
    - Corrected the north star and Metrics snapshot to report the **validation** macro F1 with per-attribute F1, PR-AUC and positive counts, traced to the committed `InferenceGuard_finetune.ipynb` output (issue #33, blocker 3 of the PR #28 review).
    - Filled in this Individual contributions section from work visible in the repo, rather than leaving four of five members as "Pending" (issue #33, blocker 5).
    - Reviewed and approved PR #32 (external user validation logs, 2026-09-27).
    - Outstanding and not done this session: issue #21, `make_artifacts.py` split reproducibility, is assigned to Engineering and is still open.

- Yiwei Jin (Users&Research):
    - Opened issue #31 and PR #32: usability sessions with five external, non-contributor participants (P01–P05) running the synthetic scenarios in `docs/product/user_test_scenarios.md`, with the raw logs committed in `docs/session05-log`.
    - Reviewed, approved and merged PRs #29 and #30.

- Bingqi Lian (Operations):
    - Designed and implemented the Experiment Operations framework in PR #38, establishing a standardized protocol for reproducible, auditable, and comparable user and model evaluations across sessions.
    - Added `docs/experiment_protocol.md` and `configs/experiments/session06.yaml` to define experiment IDs, configuration requirements, user-study procedures, metric categories, failure taxonomy, raw-evidence handling, and held-out evaluation rules.
    - Implemented `scripts/validate_experiment.py` to validate experiment metadata, participant/scenario IDs, score ranges, session consistency, latency values, and possible PII before analysis.
    - Implemented `scripts/summarize_experiment.py` to aggregate privacy, utility, usability, reliability, per-scenario, and failure-mode metrics into report-ready outputs.
    - Added automated regression tests for the Experiment Operations pipeline; all 9 tests passed before merge.
    - PR #38 was reviewed and merged into `main` on 2026-09-29.

## Lean canvas changes (if any)
- No changes to the Lean Canvas this session. The product vision and distribution strategy established in Session 04 remain accurate.
