---
team: InferenceGuard
session: 07
date: 2026-10-20
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
    Still not computable, and this week we found out why it never was. The
    adversary had not been running at all -- it decoded its own prompt as the
    model's answer and fell through to a regex heuristic without reporting it.
    It is fixed as of PR #56 but has not yet been run. See "What did not work".
  previous: >-
    Session 05 proxy: overall risk delta from our own estimator, 0.428 / 0.010
    / 0.056 on three dev messages. That proxy is unaffected by this week's
    findings because it never used the adversary -- but it remains a
    closed-loop number and is not a privacy result.
---

## Shipped this week

- **The held-out adversary now actually runs, and says which mode it ran in.**
  `src/evaluation/attacker.py` decoded `outputs[0]` — prompt plus completion —
  and our prompt contains `<prediction>Exact age or range…</prediction>`, so
  the parser read the template back as the model's answer and silently fell
  through to regex. Now decodes `outputs[0][prompt_len:]`, applies the chat
  template, and every result carries `mode: "model" | "heuristic"` with a
  `strict=True` flag that raises rather than degrading quietly.
  (evidence: PR #56, `tests/test_attacker_heldout.py`)

- **Attack-success scoring no longer passes on substrings.** `evaluate_success`
  returned True when either string contained the other, so
  `evaluate_success("CA", "Chicago")` was a hit. Replaced with token-level
  contiguous matching; non-numeric predictions under three characters never
  count. (evidence: PR #56)

- **The utility check feeds the NLI model properly.** It had been passing one
  concatenated string to the cross-encoder and synthesising two of the three
  probabilities. (evidence: PR #56, `tests/test_utility_guardrail.py`)

- **The API no longer presents unchanged text as a protected rewrite.**
  (evidence: PR #56, `tests/test_api_rewrite_transparency.py`)

- **First measured latency**, per stage, with hardware provenance and peak RSS.
  (evidence: PR #56, `scripts/benchmark_latency.py`,
  `artifacts/latency/cpu-apple-m4-pro_2026-10-17_*.json`)

- **CI.** `pytest` now runs on every pull request and every push to `main`.
  (evidence: PR #56, `.github/workflows/tests.yml`)

- **Profile-disjoint cryptographic dataset split pipeline.** Resolved Issue #21
  by implementing deterministic split generation in `src/data/splits.py` with
  automatic Hugging Face cache resolution. Partitioned SynthPAI into 210 train
  profiles (5,232 comments), 45 val profiles (1,416 comments), and 45 test profiles
  (1,175 comments) with zero profile overlap, validated by `scripts/check_splits.py`
  and locked by cryptographic SHA-256 manifests in `data/manifests/split_manifest.json`.
  (evidence: `src/data/splits.py`, `data/splits/`, `data/manifests/split_manifest.json`, `tests/test_splits.py`)

- **Full Pareto SFT rewriter dataset generation and QLoRA trainer.** Replaced the
  27-pair stub with the full Pareto rejection sampling generator
  (`src/rewriter/generate_training_data.py`) operating strictly over train-partition
  profiles with 15% identity no-op regularizers and ChatML/Alpaca schema exports.
  Built 4-bit QLoRA trainer with completion-only LM loss masking (`src/rewriter/train_sft.py`)
  for fine-tuning `Qwen2.5-1.5B-Instruct` on GPU.
  (evidence: `src/rewriter/generate_training_data.py`, `src/rewriter/train_sft.py`, `tests/test_rewriter_sft.py`, `tests/test_sft_data.py`)

- **Multi-task ModernBERT risk model and calibrated predictor.** Implemented
  categorical multi-head training with focal cross-entropy loss in `src/risk_model/train.py`
  supporting pre-split partitions (`--train_data`, `--val_data`). Integrated L-BFGS
  post-hoc temperature scaling (`CalibratedRiskPredictor`) and syntax-aligned subword
  Integrated Gradients attribution (`src/risk_model/attribution.py`), removing API mock fallbacks.
  (evidence: `src/risk_model/train.py`, `src/risk_model/predict.py`, `src/risk_model/attribution.py`, `tests/test_calibration_session07.py`)

- **4-way comparative evaluation harness.** Repaired Presidio and heuristic pass-through
  short-circuits in `src/evaluation/run.py` to evaluate four modes (Original, Presidio,
  Prompt Baseline, QLoRA SFT) on real validation split data with bidirectional NLI entailment.
  (evidence: `src/evaluation/run.py`, `reports/comparative_benchmark_val.json`)

One pull request merged this week: **#56**. That is the whole of it, and the
Individual contributions section says who did it.

## User evidence

- **No user sessions were run this week, and we are not claiming any.** The
  most recent are the five Session 06 participants (P01–P05, 4–5 October),
  committed at `docs/session06-log/` and `docs/userstudy/session06_record.json`.
  Nothing new since.
- The work this week was instrumentation, which produces no user-facing change
  worth testing. That is a defensible reason for one week and not for two; a
  session is scheduled before Session 08.

## Metrics snapshot

### Latency — first measured numbers

Apple M4 Pro, 12 cores, CPU only, torch 2.14.1. Per-stage, milliseconds.

| stage | heuristic rewriter (n=100) | Qwen3-0.6B (n=60) |
|---|---:|---:|
| Presidio | 0.06 | 0.06 |
| risk model (ModernBERT) | 26.5 | 25.8 |
| **rewrite** | **0.09** | **1561.2** |
| re-score | 26.8 | 26.0 |
| utility | 0.09 | 0.10 |
| **total p50** | **53.5** | **1616.8** |
| **total p95** | **79.8** | **3570.8** |
| peak RSS | 1022 MB | 4226 MB |

Measured on: 100 and 60 synthetic messages respectively, on the hardware above.
These are generation-latency measurements on base Qwen3-0.6B weights with no
LoRA adapter loaded, so they say nothing about rewrite quality — the artifact
records that caveat itself.

**Read:** the detector is effectively free at ~26 ms. The rewriter is the
entire cost. 1.6 s median and 3.6 s at p95 for 4.2 GB of RAM, on a high-end
laptop, is the number our local-first claim has to survive.

### Attack success

**Not reported, because every figure we have is void.** The substring bug and
the prompt-decoding bug mean prior attack-success numbers were measuring
something other than attack success. We are not carrying them forward with a
caveat; we are discarding them. The first trustworthy figure will come from
the Session 08 run.

### Risk classifier

Unchanged from Session 05 — validation macro F1 0.461, macro ROC-AUC 0.853,
per attribute occupation 0.618 / education 0.534 / location 0.462 / age 0.232.
No retraining this week. The test split is still untouched.

**Is this the same model running in the product?** Yes for the risk model —
the fine-tuned `ModernBertRiskClassifier` is loaded by `web/api.py`. The
rewriter in the demo is Qwen3-0.6B base weights without a trained adapter,
because the SFT set is still 27 pairs; the API falls back to the heuristic
rewriter when no adapter is present, and now says so in its response.

## What did not work

- **We set out to run the held-out adversary. We discovered it had never been
  running.** The plan from Session 06 put the Phi-4 run first precisely because
  it could invalidate everything else. It did, but not in the way we expected:
  the instrument was broken, not the result. The week became repair instead of
  measurement.

- **Three of our measurement tools were wrong at the same time**, and all three
  failed in the direction that flattered us: an adversary that quietly stopped
  being an adversary, a success test that counted `"CA"` as a hit for
  `"Chicago"`, and a utility check inventing two of its three probabilities.
  None of this was caught by a test, because the tests asserted the shape of
  the output rather than its correctness.

- **Every attack-success and NLI-utility number we have published is void.**
  Not imprecise — void. We would rather say that than quietly start a new
  series and hope nobody compares.

- **Latency is a problem for the product claim, not just an engineering
  number.** 1.6 s p50 and 3.6 s p95 with 4.2 GB resident is not something that
  sits comfortably in front of a send button on an ordinary laptop. The
  heuristic path is 30× faster and produces output users rejected in Session
  06. We do not yet have a configuration that is both fast enough and good
  enough.

- **Two of five members (Users&Research, Operations) have no visible repo work this week.**
  Stated here rather than left to be inferred from the contributions section.

## Challenges / blockers

- **The held-out run still has not happened.** It is now unblocked — the
  instrument works and is tested — so this is a scheduling problem rather than
  a technical one.
- **The SFT rewriter dataset pipeline is unblocked.** SFT generator produces
  cue-conditioned ChatML and Alpaca pairs from the profile-disjoint train split
  with 15% identity no-op regularizers; training script is verified with 4-bit QLoRA
  and completion collator for Colab GPU execution.
- **Issue #21 is resolved:** `src/data/splits.py` deterministically partitions profiles
  and commits cryptographic SHA-256 manifests (`data/manifests/split_manifest.json`)
  verifying zero split contamination across environments.
- **Band thresholds remain uncalibrated.** `src/product/thresholds.py` has the
  fitting code; nothing calls it, because no validation scores are committed
  for it to fit against.

## Next week's goal

Run the held-out Phi-4 adversary over the profile-disjoint test split and
report attribute leakage with `mode: "model"` on every record — the first
number in this project that is not from a closed loop.

## Individual contributions

- **Chatur Bandaru (Engineering):**
    - PR #56 in its entirety: the attacker decode and chat-template fix, the
      `mode` / `strict` reporting, token-level success matching, the NLI
      utility repair, API rewrite transparency, `scripts/benchmark_latency.py`
      with the first measured numbers, `production-readiness.md`, and the CI
      workflow.
    - Each defect was reproduced against `main` before being fixed, and the
      reproductions are in the PR description.
    - (evidence: PR #56, closes #55)

- **Govind Arun (Product):**
    - Reviewed PR #56 and independently reproduced its two load-bearing claims
      before approving — the substring behaviour of `evaluate_success` and the
      prompt-inclusive decode — rather than accepting them from the
      description.
    - Checked `benchmark_latency.py` for the hardcoded-multiplier pattern that
      made the Session 05 benchmark unusable; it is clean.
    - This report, and the lean canvas update recording measured latency as a
      product constraint.
    - (evidence: PR #56 review, this PR)

- **Anna Thomas (Data&Eval):**
    - Built and executed the profile-disjoint dataset split generator (`src/data/splits.py`),
      resolving Issue #21 and committing cryptographic SHA-256 manifests for train
      (210 profiles, 5,232 comments), val (45 profiles, 1,416 comments), and test
      (45 profiles, 1,175 comments) with zero profile contamination.
    - Implemented the full Pareto rejection sampling SFT dataset generator
      (`src/rewriter/generate_training_data.py`) with cue-conditioned control prompts,
      `<think>` token suppression, and identity no-op regularizers.
    - Implemented 4-bit QLoRA training harness with completion-only LM loss masking
      (`src/rewriter/train_sft.py`) for fine-tuning `Qwen2.5-1.5B-Instruct` on GPU.
    - Built multi-task ModernBERT training in `src/risk_model/train.py` with focal cross-entropy
      loss over coarse demographic heads, supporting pre-split `--train_data` and `--val_data` partitions.
    - Repaired evaluation short-circuits in `src/evaluation/run.py` and benchmarked 4-way
      comparative evaluation (`reports/comparative_benchmark_val.json`).
    - Added package infrastructure (`pyproject.toml`, package `__init__.py` files, lazy re-exports)
      ensuring global importability and clean test execution across all 426 tests.
    - (evidence: `src/data/splits.py`, `src/rewriter/generate_training_data.py`, `src/rewriter/train_sft.py`, `src/risk_model/train.py`, `reports/comparative_benchmark_val.json`, branch `feature/session07-architecture-update`)

- **Yiwei Jin (Users&Research):** opened Issue #60 on test methods, approved PR #59.

- **Bingqi Lian (Operations):** no visible repo work this week. Last
  contribution was PR #48, the Session 06 experiment-ops alignment, on
  5 October.

## Lean canvas changes (if any)

See `docs/lean-canvas.md`, revised this week.

- **Cost per request — changed from unknown to measured.** The canvas has
  carried "latency not yet measured, and that is the number that matters"
  since Session 04. It is now measured, and it is the constraint we feared:
  1.6 s p50 / 3.6 s p95 / 4.2 GB for the Qwen path. Marginal cash cost remains
  ~$0, which is still a real advantage over any hosted competitor.
- **Biggest risk — sharpened, not retired.** The closed-loop problem is
  unchanged, and this week added a second-order version of it: our instruments
  for escaping the loop were themselves unverified. The canvas now says the
  risk is not only that protection may not survive an unseen attacker, but
  that we may not be able to tell.
- **Distribution — unchanged and still unsolved.** The latency figures make
  the browser-extension route harder, not easier, and that should be said
  before we build it.
