---
team: InferenceGuard
session: 11
date: 2026-11-17
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
    Attribute leakage reduction under a held-out adversary (Phi-4-mini) on the
    profile-disjoint test split, with utility retained.
  value: >-
    Still not computed, for the fifth consecutive session. This week it stopped
    being the most important number: the risk classifier the product serves was
    scored for the first time and does not detect anything. Macro ROC-AUC 0.453
    on 45 held-out profiles -- below chance -- because the published checkpoint
    was never trained. There is no point attacking a pipeline whose detector is
    noise, so the ordering for the final fortnight has changed.
  previous: >-
    Session 10: hosting and containerization shipped; the risk model silently
    changed from ModernBERT to RoBERTa and was not scored.
---

## Shipped this week

- **The served risk classifier was scored, and it fails.** `scripts/score_risk_model.py`
  evaluates whatever `RISK_MODEL_ID` points at, using Session 05's labels and metric
  functions, and refuses `--split test`. Result below. (evidence: PR #93, closes #92,
  `artifacts/risk_eval/inferenceguard-roberta-risk_val_2026-11-10.json`)

- **A loader bug that served a random model was found and fixed.** On current
  `transformers`, `src/risk_model/roberta_model.py` never called `post_init()`, so the
  product loaded a **1-layer RoBERTa with randomly initialised heads**. The fix restores
  all 12 layers and 219/219 tensors; three new tests fail on the pre-fix code and pass
  after. (evidence: PR #93, `tests/test_served_risk_model.py`)

- **The upload script can no longer publish an untrained model.** `host_roberta_on_hf.py`
  now requires `--checkpoint-dir` and refuses a missing, empty or key-mismatched
  checkpoint. (evidence: PR #93)

- **A production evidence gate.** `scripts/check_production_evidence.py` separates
  *service availability* from *evidence readiness*: the API could report `ready=true`
  while serving artifacts nobody had evaluated, which is precisely what happened. Fails
  closed. (evidence: PR #94, `configs/production/session10.yaml`)

- **`.env` is now gitignored, with guards.** Session 10 shipped documentation telling
  people to put a Hugging Face write token in `.env` in a public repository, and nothing
  ignored it. Nothing leaked. (evidence: PR #91)

## User evidence

- 4 User test carried out this week, recorded in `docs/userstudy/session011_record.json`.
- his week’s study found that people may be willing to share sensitive details when they believed those details would help the AI give a better answer.
- Using current running guide in repo, the Qwen adapter was not loaded. Running directly from the InferenceGuard_finetune.ipynb Colab notebook doesn't work either. 

## Metrics snapshot

### The served risk classifier, scored for the first time

45 held-out profiles, 1,416 comments, Session 05's labels and metric functions. Positive
counts match Session 05 exactly (83 / 56 / 281 / 146), so this is the same measurement
applied to a different model.

| | macro F1 | macro PR-AUC | macro ROC-AUC | ECE |
|---|---:|---:|---:|---:|
| ModernBERT-base (Session 05) | 0.461 | 0.457 | **0.853** | — |
| RoBERTa `8273bc7c` — **what we serve** | 0.019 | 0.101 | **0.453** | 0.390 |

Per attribute, ROC-AUC:

| attribute | ModernBERT | served RoBERTa | positives |
|---|---:|---:|---:|
| occupation | 0.870 | 0.567 | 281 |
| education | 0.883 | 0.466 | 146 |
| location | 0.907 | 0.388 | 56 |
| age | 0.753 | 0.390 | 83 |

**0.453 is below chance.** A coin flip scores 0.5. Macro F1 of 0.019 means it is
essentially never correctly flagging a risky comment.

### Why: two defects, stacked

1. **The published checkpoint was never trained.** 198 of 199 backbone tensors are
   byte-identical to stock `roberta-base`; the only difference is a randomly initialised
   pooler. The head weights have standard deviation 0.0291–0.0296, which is PyTorch's
   default initialisation for a 384→1 linear layer. Scores fall in a band roughly 0.47 to
   0.52 for every comment — what near-zero logits produce.
2. **The loader could not have used it anyway.** Without `post_init()` the product served
   a 1-layer backbone with random heads.

Engineering ruled out the alternative explanations before reporting this: tokenizer
equivalence was checked (identical token IDs and scores), the score collapse was
characterised, and the backbone was compared tensor by tensor.

**So the container's risk scores have been noise since the Session 10 hosting switch.**

### Band calibration: nothing to calibrate

This was the Session 10 plan item. `src/product/thresholds.py` run against the served
model's validation scores marks **all four attributes degraded and not usable** — no
threshold reaches 60% precision. The served 0.40/0.70 edges place every validation
comment in MEDIUM, for every attribute.

Separately, the served edges (0.40/0.70 in `calibrate.py`) disagree with the documented
ones (0.30/0.60 in `risk_bands.py`, the model card, and the UI). Recorded, not changed.

### Is this the same model running in the product?

**No, and the gap is now the headline rather than a footnote.**

- **Risk classifier:** the product serves the untrained RoBERTa scored above. ModernBERT's
  0.853 describes a model that is not deployed.
- **Rewriter:** the hosted 1.54B has not passed the readiness gate, which still reports
  `BLOCKED` on the 0.5B-versus-1.5B mismatch from Session 09.
- **Held-out adversary:** still not run.

## What did not work

- **We shipped a model without measuring it, and it took a session to notice.** The
  Session 10 hosting work was careful in every respect except the one that mattered: the
  checks verified that weights *load* and heads have the right *shape*, never that the
  model *detects* anything. "The weights are on the Hub" became a proxy for "the model
  works", and the report described the move without a single accuracy figure.

- **The loader bug hid behind the training bug.** Either alone would have been caught by
  scoring. Together they produced a system that loaded, served, responded, passed its
  health checks and returned plausible-looking numbers between 0.47 and 0.52 — and
  detected nothing. Nothing in the stack distinguishes a plausible number from a correct
  one, which is the same lesson as the simulated benchmark in Session 05 and the attacker
  that parsed its own prompt in Session 07.

- **This is the third time the pattern has recurred**, and the first time it reached
  production. The fix that keeps working is the one Engineering and Operations both
  applied this week: make the artifact state which mode produced it, and fail closed.

- **The north star has not moved in five sessions.** We have good reasons each week. The
  cumulative result is that with two weeks to demo day, the project's central claim has
  never been tested.

## Challenges / blockers

- **Three decisions from PR #93 are unanswered, and all three block the demo.** Whether
  to publish a trained checkpoint or revert to ModernBERT; whether to mark the container
  not-ready until it serves a measured model; and which band edges are canonical.
  Product's recommendation is in Lean canvas changes below, but the team has to agree it
  this week, not next.
- **The final user study cannot slip again.** It needs a product that detects something.
- **Demo day is 1 December.** Two weeks, and the risk classifier in the demo is currently
  a random-weight model.

## Next week's goal

Serve a measured risk model, then run the held-out adversary against it. Concretely:
publish or revert to a checkpoint whose validation numbers we can quote, re-run
`scripts/score_risk_model.py` to confirm, then run the Phi-4 adversary on the test split.
That is the first number in this project that would not come from a closed loop, and it
is the last session in which starting it is still possible.

## Individual contributions

- **Chatur Bandaru (Engineering):**
  - PR #93 (closes #92): wrote `scripts/score_risk_model.py`, scored the served model,
    and found both defects — the untrained checkpoint and the `post_init()` loader bug
    that served a 1-layer random model. Ruled out tokenizer mismatch, characterised the
    score collapse, and compared the backbone tensor by tensor before reporting.
  - Fixed the loader, hardened `host_roberta_on_hf.py` so an untrained checkpoint cannot
    be published again, and added 9 offline tests, three of which fail on the pre-fix code.
  - Set out the three decisions the team now owes an answer to.

- **Bingqi Lian (Operations):**
  - PR #94: production evidence gate separating service availability from evidence
    readiness, with a fail-closed contract (`scripts/check_production_evidence.py`,
    `configs/production/session10.yaml`, `artifacts/production/session10_evidence.json`).
    Written directly in response to the Session 10 finding that the API could report
    `ready=true` while serving unevaluated artifacts.

- **Govind Arun (Product):**
  - PR #91: found that `.env` was not gitignored after Session 10 shipped documentation
    telling people to put a Hugging Face write token in it, in a public repository.
    Added the rule and two guards, verified by removing the rule and confirming failure.
  - Flagged in the same PR that the risk model had silently changed from ModernBERT to
    RoBERTa with no accuracy number and no disclosure, and that the required same-model
    line was missing from the Session 10 report. That flag is what #92 and #93 acted on.
  - Reviewed and merged #89, #90, #93, #94, #95, reproducing load-bearing claims on a
    clean checkout rather than accepting them from descriptions.
  - This report, and the recommendation below.

- **Yiwei Jin (Users&Research):**
  - PR #95: Session 10 user-research summary.
  - PR #90 (merged 10 November, Session 10): revised scenario cards and protocol so
    participants state what their question might reveal *before* seeing the system's
    verdict, which removes the anchoring that made Session 06's acceptance readings
    unreliable. Closes #60.
  - Session 11 user study (session11_record.json). Tested InferenceGuard with four participants, comparing what they wanted to protect before seeing the rewrite with what they chose to send afterward.
  - Results analysis (session11_doc.txt). Compared participant feedback with system logs and found that some rewrites removed details needed for the task while leaving information participants wanted to protect.

- **Anna Thomas (Data&Eval):**
  - **ModernBERT training notebook and HF hosting (`improvements/modernbert-addition`):** Authored `InferenceGuard_modernbert_train_and_host.ipynb` covering end-to-end ModernBERT fine-tuning, multitask head training, and publishing to Hugging Face Hub; updated `load_risk_classifier` to load the hosted multitask checkpoint (`multitask_heads.pt`) from Hub or local dir, with sidecar validation and loud failure on mismatched head sizes.
  - **Eval pipeline repairs and held-out adversary wiring:** Fixed the held-out Phi-4 adversary parse, rewriter wiring, and multi-turn mode flags; scored the attacker against taxonomy-normalized ground truth; added multi-turn leakage curve scaffolding and a synthetic PromptPAI eval set; reconciled label source to `reviews.human` and aligned education/location taxonomy with the SynthPAI paper.
  - **Requirements audit, doc cleanup, and serving hardening:** Produced `docs/requirements-coverage.md` gap analysis, refreshed the README with local-first setup, pinned hosted model IDs in `/health` with degraded/strict risk-model state surfacing, added session TTL/LRU eviction and `DELETE /session`, and updated the improvements backlog marking P0 items implemented.

## Lean canvas changes (if any)

- **Biggest risk — realised, in the mildest available form.** The canvas has said since
  Session 07 that the danger is not only that protection may fail but that *we may not be
  able to tell*. This week it happened in production: a model that loaded, served and
  answered, and detected nothing. It was caught by us, before demo day, by a measurement
  we chose to run. That is the good version of this outcome.

- **Product recommendation on the three open decisions.** Mine, for the team to accept or
  reject this week:

  1. **Serve ModernBERT, not RoBERTa.** It is the only risk model we have numbers for
     (ROC-AUC 0.853), `docs/methodology-review.md` chose it deliberately, and with two
     weeks left, training and publishing a RoBERTa is the higher-variance path. Switching
     back costs a config change; the alternative costs a training run we have not started.
  2. **Mark the container not-ready until it serves a measured model.** #94 gives us the
     mechanism. A product that reports healthy while detecting nothing is the failure mode
     the canvas calls highest-cost, because a user who trusts it shares *more*.
  3. **0.30/0.60 is canonical** — it is what the UI, the model card and `risk_bands.py`
     say — and re-fit per attribute once a trained model is served.

- **What we will claim on demo day.** On current evidence: that inferential leakage is
  real and measurable, that we built an end-to-end system for it, and that our own
  measurement caught our own shipped model doing nothing. Not that the system reduces
  leakage against an independent attacker — that remains unevidenced, and the honest
  framing of it is the main thing between now and 1 December.
