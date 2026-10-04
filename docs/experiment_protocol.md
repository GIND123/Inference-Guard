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

# 11. Initial Submission

Once the participant submits the query:

1. save the exact query;
2. record query formulation time;
3. record which available anchors were used;
4. record whether any pre-submission evaluator intervention occurred;
5. capture the system risk output.

If the query is generic or risk-negative, **do not ask the participant to reformulate it**.

Continue the experiment normally.

---

# 12. Comprehension Probe

Before any rewrite occurs, ask:

> **"What is the system telling you?"**

Do not:

- explain the warning;
- correct the participant;
- identify missed inference vectors;
- tell them whether their interpretation is correct.

Record the response.

Then ask:

> **"Is anything here surprising or unclear?"**

Record the response without defending the system.

---

# 13. Rewrite Phase

If a rewrite is available, allow the participant to request it.

After the rewrite appears, ask:

> **"Would you send this version as-is, edit it first, or not send it?"**

Then ask:

> **"Why?"**

If the participant edits the rewrite, capture the exact edit.

Particular attention should be paid to information that the participant restores after the system removes it. Restored information may indicate that the rewrite eliminated a task-critical constraint.

---

# 14. Zero-Risk Phase

If no rewrite is performed, do not simulate or request one merely to continue the experiment.

Confirm comprehension using the same neutral probe:

> **"What is the system telling you?"**

The participant should ideally recognize that:

1. the system did not identify meaningful inferential markers;
2. the original query was preserved;
3. no privacy-enhancing edit was performed.

A participant who believes that an unchanged query was nevertheless "made safer" should be recorded as a **trust-calibration failure**.

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
  "participant_id": "",
  "scenario_id": "",
  "scenario_mode": "",
  "session_id": "",
  "date": "",
  "facilitator": "",

  "task_context": {
    "anchors_available": [],
    "anchors_used": [],
    "anchor_disclosure_ratio": null,
    "query_formulation_seconds": null,
    "evaluator_intervention_pre_submit": false
  },

  "participant_observation": {
    "warning_in_their_words": "",
    "warning_comprehension": "",
    "surprising_or_unclear": "",

    "rewrite_decision": "",
    "edit_made": null,
    "edited_query": "",

    "usefulness": null,
    "would_use_before_sending": null,
    "adoption_reason": "",
    "time_to_decision_seconds": null
  },

  "system_log": {
    "exported_at": "",
    "client_latency_seconds": null,
    "input_text": "",

    "risk_summary": {
      "overall": null,
      "overall_band": "",
      "primary": "",
      "secondary": [],
      "bands": {
        "age": "",
        "location": "",
        "occupation": "",
        "education": ""
      },
      "scores": {
        "age": null,
        "location": null,
        "occupation": null,
        "education": null
      },
      "cues": [],
      "leakage_delta": null
    },

    "rewrite": {
      "rewritten_text": "",
      "presidio_text": "",
      "utility_metrics": {
        "cosine_similarity": null,
        "nli_forward": null,
        "nli_backward": null,
        "contradiction": null,
        "utility_score": null
      }
    },

    "turn_record": {
      "turn_number": null,
      "turn_scores": {},
      "cumulative_scores": {},
      "overall_turn_risk": null,
      "overall_cumulative_risk": null,
      "joint_entropy": null,
      "leakage_delta": null,
      "is_breached": null,
      "risk_band": ""
    }
  },

  "researcher_coding": {
    "expected_inference_attributes": [],
    "observed_disclosure_attributes": [],
    "detection_outcome": {},
    "rewrite_preserved_task_intent": null,
    "notable_model_behavior": "",
    "comments": ""
  },

  "protocol": {
    "protocol_deviation": false,
    "deviation_type": null,
    "deviation_description": null
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



