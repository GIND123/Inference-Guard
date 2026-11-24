# InferenceGuard Experiment Protocol: session06

**Study:** Inferential Privacy User Testing  
**Session:** 06  
**Protocol Type:** Controlled, goal-driven usability experiment   
**Participant Data Policy:** Synthetic scenarios only

## 1. Purpose

This experiment evaluates how users interact with an inferential-privacy system when independently formulating realistic AI queries.

The study examines the complete interaction chain:

> **Context available → Context disclosed → Risk detected → Risk communicated → Rewrite performed → User interpretation → User decision**

The experiment is designed to avoid a methodological problem observed in earlier sessions: participants often produced generic, low-risk queries unless evaluators verbally encouraged greater specificity.

Such prompting makes it impossible to distinguish naturally occurring disclosure from evaluator-induced disclosure.

Session 06 therefore uses **goal-driven synthetic vignettes and a zero-coaching protocol**.

The experiment does **not** require participants to produce privacy-risky queries. Whether they disclose available contextual information is itself an experimental outcome.


---

## 2. Experiment Types

InferenceGuard currently uses two major experiment types.

### 2.1 Model Evaluation

Model evaluation measures system performance without human participants.

Typical measurements include:

- attribute inference attack success rate
- privacy-risk reduction
- semantic similarity
- bidirectional NLI preservation
- explicit PII removal
- rewrite failure rate

Model evaluation should use fixed datasets, fixed splits, fixed checkpoints, and fixed experiment configurations.

### 2.2 User Evaluation

User evaluation measures how real users interact with the running product.

Typical measurements include:

- whether users understand the privacy warning
- whether users accept, edit, or reject a rewrite
- perceived usefulness
- time to decision
- whether users would use the tool before sending sensitive text
- qualitative failure feedback

User evaluation must use synthetic scenarios only. Participants must never be asked to provide real personal or sensitive information.

# 3. Research Questions

### RQ1 — Natural Disclosure

When participants are given realistic synthetic context, which inference-relevant details do they independently include in their queries?

### RQ2 — Risk Detection

When participants produce inference-bearing queries, does the system correctly identify the relevant inferential risk?

### RQ3 — Risk Comprehension

Can participants explain the system's risk warning in their own words without evaluator assistance?

### RQ4 — Privacy–Utility Trade-off

When the system rewrites an inference-bearing query, does the rewrite reduce privacy risk while preserving the information necessary to accomplish the user's goal?

### RQ5 — Zero-Risk Calibration

When no inferential risk is detected and no rewrite occurs, do participants correctly understand that the system deliberately preserved the original query?

### RQ6 — Adoption

After observing either intervention or non-intervention, would participants use the system before sending a real AI query?

---

# 5. Scenario Construction

Generative scenarios must contain three components.

## 5.1 Situation

A concise description of the synthetic persona's problem.

The Situation establishes motivation but must not tell the participant what information to include.

## 5.2 Context Card

A set of synthetic facts available to the persona.

Context may contain:

- age-related information;
- education context;
- occupation;
- geographic anchors;
- institutional context;
- schedules;
- transportation constraints;
- clinical context;
- third-party information.

These details create an **opportunity for inference**.

They must not be presented as information that the participant is required to reproduce.

## 5.3 Goal

A concrete outcome the participant should obtain from the AI.

The goal should create a realistic functional dependency on some contextual information while leaving the participant free to determine how much context is necessary.

---

# 6. Independent Variables

The experiment manipulates or controls the following factors.

### Task formulation

- participant-generated query;
- researcher-specified query.

### Risk state

Observed after submission:

- risk-positive;
- risk-negative.

### Interaction structure

- single-turn;
- multi-turn cumulative disclosure.

### Privacy–utility dependency

Scenarios vary in whether inference-bearing information is:

- unnecessary to task completion;
- useful but removable;
- important to preserving task meaning.

These factors should be recorded rather than collapsed into a single success/failure measure.

---

# 7. Primary Outcome Measures

## 7.1 Disclosure Behavior

For generative scenarios:

**Anchor Disclosure Rate (ADR)**

ADR = (Number of inference-relevant anchors used) / (Number of inference-relevant anchors available)

This measures how much synthetic contextual information the participant independently incorporates.

A lower ADR is not automatically a failure. It may indicate natural data minimization.

---

## 7.2 Detection

Record:

- detected risk;
- risk score/confidence;
- inferred attributes identified by the system;
- system state.

Detection should be evaluated against the information actually present in the participant's query, not against every attribute contained in the Context Card.

---

## 7.3 Warning Comprehension

Immediately after the system displays its risk state, ask:

> **"What is the system telling you?"**

Record the answer as `warning_in_their_words`.

Code comprehension as:

- **correct** — participant identifies the essential inference risk or correctly understands the absence of detected risk;
- **partial** — participant identifies some but not all essential meaning;
- **incorrect** — participant forms a substantially incorrect interpretation;
- **echo** — participant merely repeats the input or interface wording without demonstrating understanding.

The verbal response is the primary evidence. A binary self-report such as "I understood" is secondary.

---

## 7.4 Rewrite Decision

Record whether the participant would:

- `accept_as_is`;
- `accept_with_edits`;
- `reject`;
- `not_applicable`.

If edits are made, record them verbatim.

---

## 7.5 Utility

After viewing the final system state, ask the participant to rate usefulness on the study's standard scale.

The rating must be interpreted together with:

- whether a rewrite actually occurred;
- whether the participant understood the intervention;
- whether the task-relevant constraints survived the rewrite.

A high usefulness score alone must not be treated as evidence of successful privacy protection.

---

## 7.6 Adoption

Ask:

> **"Would you use something like this before sending a real AI query?"**

Record:

- yes/no;
- participant reasoning.

---

## 7.7 Decision Time

Measure:

- query formulation time;
- time from system response to final send/edit/reject decision.

---

# 8. System States

Every trial must be classified into one of three primary states.

### State A — Risk-Positive / Rewrite

Inference risk is detected and the system modifies the query.

Evaluate:

- detection;
- explanation;
- privacy reduction;
- utility preservation;
- acceptance.

### State B — Risk-Positive / Utility Conflict

Risk is detected, but removing the relevant information threatens the usefulness or meaning of the query.

Evaluate whether the participant restores removed information or rejects the rewrite.

### State C — Risk-Negative / Passthrough

No meaningful inference risk is detected.

The interface should explicitly communicate:

> **No inferential markers detected. Original text preserved.**

No rewrite should be implied.

This state remains a valid experimental outcome.

---

# 9. Facilitator Protocol

## Phase 0 — Setup

Before the participant begins:

1. Start screen recording.
2. Confirm that the running system is ready.
3. Assign an anonymous participant ID.
4. Explain that all scenario information is fictional.
5. Tell the participant not to substitute real personal information.

Do not explain which privacy attributes the system is expected to detect.

---

## Phase 1 — Scenario Presentation

For a generative task, provide:

**Situation + Context Card + Goal**

Use only the following standardized instruction:

> **"Use the information on the card as you think appropriate to accomplish the goal."**

Do not provide example queries.

---

# 10. Zero-Coaching Rule

From scenario presentation until the participant submits the initial query, the facilitator must not influence query content.

### Prohibited interventions include:

> "Can you be more specific?"

> "Think about your own situation."

> "Could you give it more context?"

> "Would your location matter?"

> "What about your age?"

> "Maybe mention your university."

> "Is there anything else the AI needs to know?"

The facilitator must also avoid non-verbal prompting that clearly signals approval or disapproval of particular details.

If the participant asks:

> "Should I include my location?"

the standardized response is:

> **"Use whichever information you think is appropriate for accomplishing the goal."**

Do not answer the substantive question.

---
# 11. Finalize the Original Query

Ask the participant to compose the question they would send to an AI. Once
they indicate it is ready, save the exact text and record the formulation
time and the scenario details they included.

Keep the finalized query as the trial input. In the current interface,
wait to click **Check my message** until the pre-result questions are complete.

# 12. Pre-Result Privacy and Rewrite Assessment

Ask these questions in order, before showing the system's risk scores,
warnings, or suggested text.

1. **Would you send this question as written, change it first, or choose
   not to send it? What makes you say that?**
2. **Is there anything in your question that you would prefer to keep
   private? If so, what?**
3. **Would you want this question rewritten for privacy before sending
   it? Why or why not?**

Allow the participant to say “none” or “unsure.” Record their answers in
their own words. If they propose a change, record it as a pre-result
preference and submit the finalized original for the system comparison.

Use neutral prompts. Avoid naming specific privacy attributes, pointing
out clues, or telling the participant whether their judgment is correct.

# 13. Show the System Result and Collect Feedback

Submit the finalized original query and show the original and suggested
text together with the system's evaluation. Ask:

1. **In your own words, what is the system telling you about your question
   and what it changed, if anything?**
2. **Did the result address the details you wanted to keep private? Is
   there anything you still want to keep private?**
3. **Did it keep the details you need for your question? Was anything
   important removed or changed?**
4. **Which would you send: the original, the suggested version, a version
   you edit first, or neither? Why?**

Record the final choice and the reason. If the participant edits either
version, save the exact edited text and record which version they started
from. Measure decision time from the moment the result is visible.

# 14. Unchanged Output

If the system returns the original text, show the actual result and ask
the same questions. For the final choice, use “send it as written,”
“edit it first,” or “send neither.”

Record the reported risk separately from whether the text changed.
Distinguish a low-risk passthrough from an unchanged result that still
has flagged risk. Record whether the participant noticed that the text
was unchanged and how that affected their decision.



---

# 15. Adoption Probe

At the end of every scenario, ask:

> **"Would you use something like this before sending a real AI query?"**

Follow with:

> **"Why or why not?"**

Do not persuade the participant to adopt the product.

---

# 16. Multi-Turn Protocol — S-06

S-06 uses staged context disclosure.

Participants must not receive four pre-written queries.

### Stage 1

Provide the initial Situation, Context Card, and Goal.

Allow the participant to formulate the first query.

Record the cumulative risk state.

### Stage 2

Reveal one additional synthetic constraint.

Say only:

> **"You now also know this."**

Allow the participant to decide whether and how to continue the conversation.

Record the cumulative risk state.

### Stage 3

Repeat with the next contextual constraint.

### Stage 4

Reveal the final contextual constraint and allow the participant to continue.

At every stage record:

- new context available;
- new context disclosed;
- exact participant message;
- turn-level risk;
- cumulative risk;
- inferred attributes.

The facilitator must never instruct the participant to incorporate newly revealed information.

---

# 17. Recording Schema

Record one JSON object per participant per scenario.

```json id="ijrc21"
{
  "participant_id": "anonymous-01",
  "scenario_id": "S-03",
  "scenario_mode": "generative",
  "date": "2026-10-XX",
  "facilitator": "<github handle>",
  "artifact": "user_studies/session06/anonymous-01.mp4",

  "anchors_available": [
    "age",
    "neighborhood",
    "medication",
    "provider_context"
  ],

  "initial_query": "Are metformin and statins generally safe together?",
  "query_formulation_seconds": 18,

  "anchors_used": [
    "medication"
  ],

  "evaluator_intervention_pre_submit": false,

  "detected_risk": false,
  "risk_score": 0.001,
  "system_state": "risk_negative_passthrough",

  "warning_in_their_words":
    "It doesn't think this question reveals anything about me.",
  "warning_comprehension": "correct",

  "requested_rewrite": false,
  "rewrite_changed_text": false,
  "rewrite_decision": "not_applicable",
  "edit_made": null,

  "usefulness": 6,
  "would_use_before_sending": true,
  "adoption_reason":
    "I like knowing when nothing needs to be changed.",

  "time_to_decision_seconds": 41,
  "comments": ""

  "pre_result": {
    "send_decision": "send_as_written | edit_first | do_not_send | unsure",
    "decision_reason": "",
    "privacy_response_verbatim": "",
    "perceived_private_details": [],
    "wants_privacy_rewrite": "yes | no | unsure",
    "rewrite_reason": ""
  },
  "post_result": {
    "final_choice": "original | suggestion | edited_version | neither",
    "choice_reason": "",
    "remaining_privacy_concerns": "",
    "task_details_removed_or_changed": "",
    "edited_from": null,
    "edited_query": null
  }
}
```

---

# 18. Protocol Deviations

Any violation of the zero-coaching rule must be recorded.

Examples include:

- evaluator suggested greater specificity;
- evaluator mentioned an inference-relevant attribute;
- participant accidentally used real personal information;
- system malfunctioned;
- recording failed;
- participant saw information from another condition;
- facilitator explained the warning before the comprehension probe.

Record:

```json id="nvt37s"
{
  "protocol_deviation": true,
  "deviation_type": "pre_submit_coaching",
  "description": "Facilitator asked participant to provide more context."
}
```

Trials containing major pre-submission coaching must not be used as evidence of natural disclosure behavior.

They may still be retained as qualitative usability evidence if clearly labeled.

---

# 19. Analysis Rules

### Rule 1 — Do not treat generic queries as failed trials.

A participant may rationally omit contextual information.

Report this as disclosure behavior.

### Rule 2 — Evaluate detection against disclosed information.

Do not penalize the detector for failing to infer an attribute that exists only on the Context Card and was never communicated.

### Rule 3 — Separate system failure from scenario outcome.

For example:

> Context available → participant omits it

is different from:

> participant discloses context → detector misses it.

### Rule 4 — Separate comprehension from agreement.

A participant may correctly understand the warning and still disagree with it.

### Rule 5 — Separate rewrite quality from adoption.

A technically effective rewrite may still introduce too much friction to be useful.

### Rule 6 — Do not interpret unchanged text as a successful rewrite.

Zero-risk passthrough must be analyzed separately.

### Rule 7 — Do not interpret disclosure frequency as real-world prevalence.

These are controlled synthetic scenarios.

The experiment measures behavior under designed information dependencies, not population-level privacy behavior.

---

# 20. Scenario Allocation

| Scenario | Condition | Primary Measure |
|---|---|---|
| S-01 University advice | Generative | Natural disclosure of education/location anchors |
| S-02 Workplace conflict | Controlled | Privacy–utility preservation |
| S-03 Health question | Generative | Selective disclosure + capability understanding |
| S-04 Travel planning | Generative | Constraint preservation + third-party leakage |
| S-05 Explicit PII | Controlled | Explicit-PII baseline |
| S-06 Multi-turn | Generative / staged | Cumulative inference |

S-02 and S-05 should remain stable across sessions where possible to provide longitudinal regression evidence.

---

# 21. Evidence Requirements

A valid trial requires:

- a real participant who is not a member of the project team;
- interaction with the running product;
- contemporaneous recording;
- anonymous participant ID;
- scenario ID;
- exact initial query;
- system output;
- comprehension response;
- final decision.

Recordings and structured observations should be committed during the same reporting period in which they are cited.

Team testing remains development evidence and must not be reported as user evidence.

---

# 22. Ethics and Data Handling

All scenario information must remain synthetic.

Participants must never be asked to:

- provide their real address;
- identify their real university or employer;
- disclose their actual medical history;
- provide their real age;
- disclose another person's private information;
- substitute real personal information for fictional scenario details.

If a participant voluntarily begins entering real personal information, stop them before submission where practical and ask them to use only the synthetic scenario information.

Do not commit real participant PII to the repository.

---

# 23. Validity Controls

The experiment prioritizes four forms of validity.

### Experimental Control

Participants receive standardized synthetic contexts and facilitator instructions.

### Participant Autonomy

Participants independently determine how to formulate generative queries.

### Task Coverage

Scenarios contain information dependencies capable of producing meaningful inferential privacy risk.

### Ecological Validity

Tasks approximate goal-directed AI use rather than requiring participants to recite researcher-written privacy leaks.

These scenarios remain controlled simulations. Results should therefore not be presented as estimates of natural real-world disclosure prevalence.

---

# 24. Primary Protocol Principle

The governing rule for Session 06 is:

> **Create the opportunity for inference. Do not create the disclosure for the participant.**

The scenario controls the available context.

The participant controls disclosure.

The evaluator controls neither.

The system then demonstrates intervention or deliberate non-intervention.

Both outcomes are valid experimental evidence.

---

# 25. Automated Evaluation Harness — Run Commands

This section documents the single commands that produce the quantitative
evaluation artifacts (backlog items #8 and #9). All commands run from the repo
root with the project environment (CPU torch + `requirements-ci.txt`) installed.

> **Numbers are pending P0 #1 (the north-star GPU run).** Until a trained
> ModernBERT risk model, the LoRA rewriter adapter, and the held-out Phi-4-mini
> adversary are available, every harness below runs in `load_model=False`
> fallback mode and tags its output `reportable_as_held_out = false`. The
> harness never fabricates leakage or attack-success numbers; non-model runs are
> explicitly non-reportable.

## 25.1 Full comparison matrix + multi-turn leakage curve

One command produces all five design-matrix rows (Raw / Presidio / Prompt-only /
Fine-tuned / Conversation-aware) and the 1/2/4/8-turn cumulative leakage curve:

```bash
python -m src.evaluation.run --split test --multiturn --matrix_markdown --use_heavy_models
```

Outputs (under `reports/`):

- `comparative_benchmark_test.json` — per-method matrix plus the
  `conversation_aware` section (driven by the multi-turn leakage tracker).
- `comparison_matrix_test.md` — the matrix as a markdown table.
- `multiturn_leakage_curve.json` — cumulative leakage + per-attribute risk at
  turn counts 1/2/4/8 (clamped to the available staged turns).
- `multiturn_leakage_curve.png` — rendered curve, written only when matplotlib
  is installed; the JSON is always written.

Omit `--use_heavy_models` to run the fallback locally without GPU models (the
outputs are then non-reportable, as flagged in each file).

## 25.2 Multi-turn curve only

The multi-turn leakage curve drives the S-06 staged protocol (section 16) over
cumulative 1/2/4/8-turn prefixes of a committed staged sample
(`data/evaluation/staged_turns_sample.jsonl`). To regenerate it on its own:

```bash
python -c "from src.evaluation.multiturn import run_multiturn_leakage_curve as r; r(use_heavy_models=True)"
```

## 25.3 PromptPAI secondary evaluation set (#9)

The synthetic PromptPAI set is a secondary, AI-assistant-prompt-style evaluation
input deterministically derived from the committed synthetic SynthPAI profiles
(no downloads, no real data). A small sample is committed under
`data/promptpai/sample.jsonl` with `data/promptpai/manifest_sample.json`.

Generate the full 300-500 row set (seeded, deterministic):

```bash
python src/data/promptpai.py --source_split test --num_rows 400 \
    --out data/promptpai/promptpai.jsonl --manifest data/promptpai/manifest.json
```

Re-generate the committed small sample:

```bash
python src/data/promptpai.py --sample --source_split test
```

> Run the generator via the file path (`python src/data/promptpai.py ...`) rather
> than `-m src.data.promptpai`; the latter imports `src/data/__init__.py`, which
> pulls `numpy`. The generator itself is stdlib-only (json, hashlib, random,
> pathlib) so it runs in restricted environments.

Evaluate any method against the PromptPAI set (records already match the harness
record shape `{profile_id, text, attributes}`):

```bash
python -m src.evaluation.run --dataset promptpai --multiturn --matrix_markdown
```

## 25.4 North-star held-out leakage benchmark (P0 #1)

This is the thesis deliverable: the per-attribute attribute-leakage / attack-success
number under the **held-out** Phi-4-mini adversary, with the hosted ModernBERT risk
model and the hosted LoRA rewriter actually loaded. It is produced by a **single
command on a GPU host** and is an extension of the Evidence Requirements
(section 21): a benchmark result only counts when `attacker_mode == "model"` and
`reportable_as_held_out == true`.

> **The number itself is PENDING GPU EXECUTION.** It is not produced in the
> CPU/CI environment and must never be fabricated. A run whose adversary falls
> back to the heuristic (`attacker_mode == "heuristic"`) is NON-REPORTABLE; the
> harness raises instead of writing such a report when `--use_heavy_models` is
> set. For the number to count as held-out evidence, the written payload must
> carry `attacker_mode: "model"` and `reportable_as_held_out: true`.

### Required environment (GPU host)

- `RISK_MODEL_ID=AT707/inferenceguard-modernbert-risk`: the hosted ModernBERT
  risk model (configs/models.yaml `risk_estimator.chosen = answerdotai/ModernBERT-base`).
  `load_risk_classifier` auto-detects ModernBERT vs RoBERTa, so this does not
  disturb the RoBERTa serving default in `web/api.py` / `scripts/score_risk_model.py`.
- `REWRITER_MODEL_ID=AT707/inferenceguard-rewriter-v1`: the hosted LoRA rewriter.
- Held-out adversary: `microsoft/Phi-4-mini-instruct` (the `--adversary_model_id`
  default; different vendor / tokenizer / pretraining from the rewriter, so it is
  never used to select rewrites or tune thresholds).
- `transformers`, `torch`, and `huggingface_hub` installed with GPU access.

### The single command

```bash
RISK_MODEL_ID=AT707/inferenceguard-modernbert-risk \
REWRITER_MODEL_ID=AT707/inferenceguard-rewriter-v1 \
python -m src.evaluation.run --split test --use_heavy_models --multiturn --matrix_markdown
```

The model ids can also be passed explicitly via `--risk_model_id`,
`--rewriter_model_id`, and `--adversary_model_id` instead of the environment
variables.

### Troubleshooting / required Colab dependencies

A real GPU run (for example on Colab) needs the full heavy-model stack. A partial
install is the most common cause of a non-reportable run.

- **Required pip dependencies:** `transformers`, `torch`, `accelerate`,
  `bitsandbytes>=0.46.1`, `huggingface_hub`, `sentence-transformers`,
  `presidio-analyzer`, `spacy`. After installing `spacy`, also download the large
  English model: `python -m spacy download en_core_web_lg`.
- **`HF_TOKEN` must be set.** The hosted risk model, the LoRA rewriter, and the
  Phi-4-mini adversary are all pulled from the Hub. Unauthenticated requests fail
  or get rate-limited, which can leave a model unloaded and silently degrade the
  run. Export `HF_TOKEN` (or run `huggingface_hub` login) before launching.
- **Missing `bitsandbytes` cascades into a non-reportable run.** When
  `bitsandbytes` is absent, the quantized rewriter AND the Phi-4-mini adversary
  both silently drop to heuristic mode (`attacker_mode == "heuristic"`). The
  harness then CORRECTLY raises a `RuntimeError` and refuses to emit a
  non-reportable number rather than writing a misleading report. The fix is:
  `pip install -U "bitsandbytes>=0.46.1" accelerate`, set `HF_TOKEN`, then restart
  the runtime so the heavy models load under GPU.
- **Phi-4-mini loads via native transformers support (no remote code by default).**
  `microsoft/Phi-4-mini-instruct`'s bundled `modeling_phi3.py` imports the
  `LossKwargs` symbol (around line 38, from `transformers.utils`) that was removed
  in transformers>=5.0, so executing that remote file fails with
  `ImportError: cannot import name 'LossKwargs' from 'transformers.utils'`. That
  file was only ever committed once and is present on every revision, so **no
  Phi-4-mini-instruct revision fixes it**; pinning `ADVERSARY_REVISION` cannot
  resolve the incompatibility. Because Phi-4-mini uses the `phi3` architecture
  that modern transformers supports natively, the harness now loads the adversary
  with `trust_remote_code=False` by default: transformers uses its own built-in
  `phi3` implementation (works on transformers 5.x) and never touches the broken
  remote file. No env var is needed for this fix; just run:

  ```bash
  RISK_MODEL_ID=AT707/inferenceguard-modernbert-risk \
  REWRITER_MODEL_ID=AT707/inferenceguard-rewriter-v1 \
  python -m src.evaluation.run --split test --use_heavy_models --multiturn --matrix_markdown
  ```

  The `ADVERSARY_TRUST_REMOTE_CODE` env var exists to opt back into remote code
  (`ADVERSARY_TRUST_REMOTE_CODE=1`) only if you ever need it, but doing so on
  transformers>=5.0 reintroduces the `LossKwargs` `ImportError`, so leave it unset
  (the default is `False`). `ADVERSARY_REVISION` remains available as a secondary
  revision pin but, per the finding above, does not help with the `LossKwargs`
  incompatibility.
- **The strict guard is intended.** The harness raising a `RuntimeError` on any
  heuristic fallback (under `--use_heavy_models`) is deliberate behavior and must
  NOT be weakened or bypassed. Emitting a number from a heuristic adversary would
  violate the held-out evidence requirement; the only correct response is to fix
  the environment so the real models load.

### Attacker scoring: ground truth is normalized to the taxonomy label space

The test split stores ground truth as free text (`age: "41"`,
`location: "Bangkok, Thailand"`, `education: "Masters in Business Administration"`,
`occupation: "business consultant"`), while the held-out adversary predicts the
coarse taxonomy labels (`35-44`, `asia_pacific`, `master`, `business_finance`).
`compute_method_metrics` now collapses the free-text ground truth into the
taxonomy label space via `normalize_to_taxonomy_label` before comparing, so a
correct coarse inference is counted. A blank or missing ground-truth value maps
to `UNKNOWN` and is excluded from the per-attribute denominator. Values already
stored as taxonomy labels pass through unchanged, so the inline smoke records are
unaffected.

The adversary PREDICTION side is normalized the same way. `build_attack_prompt`
asks for one JSON object with the four attribute keys, but a real model answers
in free text (`"late 20s"`, `"Seattle, WA"`, `"software engineer"`). The parser
now extracts the value robustly (JSON object, else a key:value scrape) and routes
each value through the same `normalize_to_taxonomy_label`, so a correct
free-text guess lands on the taxonomy label instead of collapsing to `UNKNOWN`.
An explicit `"unknown"` answer stays an abstention.

The comparative harness now attaches the QLoRA adapter (`REWRITER_ADAPTER_DIR`,
default `artifacts/rewriter_qlora`) in the `--use_heavy_models` path, so
`prompt_baseline` (adapter disabled) and `qlora_sft` (adapter engaged) diverge
instead of running the same model. The rewriter's rewrite-time system prompt is
read from `self.system_prompt` (`REWRITER_SYSTEM_PROMPT`); a prior undefined name
made every real-model rewrite fail and fall back to unchanged text. The
multi-turn curve reuses the already-loaded adversary and tags
`reportable_as_held_out` to match that adversary's actual mode.

### Output files (under `reports/`)

- `comparative_benchmark_test.json`: the per-method matrix plus the top-level
  held-out provenance block (`attacker_model_id`, `attacker_mode`,
  `reportable_as_held_out`, `risk_model_id`, `rewriter_model_id`), the
  `per_attribute_attack_success` matrix, and the `asr_reduction` block (each
  method vs the Raw baseline, overall and per attribute).
- `comparison_matrix_test.md`: the five-row matrix (Raw / Presidio / Prompt-only /
  Fine-tuned / Conversation-aware) with the per-row reportable flag.
- `multiturn_leakage_curve.json` (+ `.png` when matplotlib is present): the
  1/2/4/8-turn cumulative leakage curve driving the Conversation-aware row.

The result counts as the north-star held-out number only when
`comparative_benchmark_test.json` reports `attacker_mode: "model"` and
`reportable_as_held_out: true`. Until the GPU run is executed and these files are
committed with those flags true, the north-star number remains **PENDING GPU
EXECUTION** and no value may be reported.

### Runtime progress and expected duration

- The heavy run processes samples sequentially, and each sample runs several LLM
  generations (rewriter plus the Phi-4-mini adversary), so it is not fast. Expect
  minutes for `--limit 50` and roughly 1-4 hours for the full `test` split on an
  A100. Run a small `--limit N` smoke check first to confirm the run is
  reportable before launching the full split.
- A `tqdm` progress bar (one bar per method, with throughput and ETA) and a
  device banner now print at startup. Confirm the banner reports `cuda` and the
  A100 name rather than CPU, and watch `nvidia-smi` in a second cell for
  `GPU-Util > 0` to confirm the GPU is actually working.
- Set `IG_REQUIRE_CUDA=1` to make the run fail fast with a clear error when CUDA
  is unavailable, instead of silently falling back to a very slow CPU run. It is
  unset by default, so CPU environments are unaffected.



