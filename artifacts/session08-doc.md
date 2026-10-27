# Session 08 — Evaluation Terminology Comprehension

## Overview

Session 08 examines whether users can understand the evaluation information presented in the InferenceGuard web interface after question-mark tooltips were added to technical terms.

The current session includes one participant, P01, completing controlled scenario S02 (workplace conflict). The test was conducted on October 26, 2026.

The evaluation focuses on:

- comprehension of evaluation terminology and numerical results;
- whether tooltip explanations make technical metrics understandable;
- users' preferences for simpler ways to communicate evaluation results; and
- the relationship between reported comprehension and the system output shown during the task.

The central finding is that **the participant wanted access to the numerical results but found the terminology difficult to understand, even with annotations**. The feedback supports simplifying the primary presentation rather than relying on explanations of technical names alone.

These findings are exploratory observations from one participant, not evidence of general usability performance or a causal comparison of interfaces with and without tooltips.

## Study Materials

- **`01.json`**: exported system response, evaluation metrics, rewrite output, and timing information for session `01`.
- **`session08_record.json`**: participant observations, task context, researcher coding, and protocol record for P01/S02.
- **Facilitator's supplementary debrief summary**: the participant found evaluation terminology difficult despite the annotations, preferred simpler explanations, and still wanted to know the numerical results.
- **recording link**: `https://drive.google.com/drive/folders/1GqVcO7nvjkx1wShVypwAad3sl-GpdacM?usp=drive_link`.

The supplementary debrief provides the specific feedback about tooltips and simpler terminology. The structured record codes warning comprehension as `partial` and the unclear-content field as `unclear`, but does not preserve a term-by-term account of the participant's difficulties. Debrief feedback is paraphrased here rather than presented as a verbatim quotation.

---

## Current Participants

| Participant | Scenario | Mode | ADR | Warning Comprehension | Rewrite Decision | Usefulness | Would Use |
|---|---|---|---:|---|---|---|---|
| P01 | Workplace Conflict (S02) | Controlled | 1.0 | Partial | Accept as is | 7; scale not specified in record | Yes |

ADR denotes anchor disclosure ratio. All supplied scenario anchors were used. No pre-submission evaluator intervention or protocol deviation was recorded. The recorded time to a rewrite decision was 10 seconds; this is not a measure of tooltip-reading time or successful metric comprehension.

---

## Preliminary Findings

### 1. Tooltip Annotations Did Not Resolve the Reported Comprehension Difficulty

Understanding the evaluation terminology remained difficult even after annotations were added. The participant preferred a simpler way of explaining what the results mean.

This extends the issue identified in Session 06, where a participant asked about Joint Entropy and Leakage Delta. Session 08 suggests that adding a definition beside a technical term may still leave substantial interpretation work for the user.

The available record does not identify which individual definitions failed, whether every tooltip was opened, or whether discovery, wording, length, or mathematical concepts were the main source of difficulty. It therefore supports a finding of **reported difficulty with the annotated interface**, rather than a claim that every tooltip was ineffective.

### 2. The Participant Wanted the Numbers, but Needed a More Accessible Meaning

The participant's feedback did not call for removing numerical evaluation. They wanted to know the values while finding the labels too technical.

The design implication is to preserve the numerical information while making the first explanation concrete: what is being measured, whether a higher or lower value is preferable, and what the result means for this message or conversation.

For example, a user should not need to understand the term Residual Anonymity Capacity before learning that a lower value indicates greater accumulated exposure across the four tracked attributes. Likewise, Leakage Delta should first communicate how much additional risk the latest message introduced.

### 3. Recognizing the Main Risk Did Not Establish Full Evaluation Comprehension

The structured record captures the participant's interpretation as “occupation is the main concern” and codes warning comprehension as `partial`.

This is consistent with the system identifying occupation as the primary risk. However, it does not establish that the participant understood the remaining metrics, the difference between current-message and cumulative risk, or whether the rewrite reduced exposure.

Future comprehension checks should distinguish recognizing a category label from explaining a numerical result in the participant's own words.

### 4. Positive Usefulness and Acceptance Coexisted with Partial Understanding

P01 accepted the rewrite without edits, reported usefulness as 7, and indicated willingness to use the tool before sending a query. 

These responses show that a participant can express interest in the product while still having difficulty interpreting its evaluation. Acceptance and willingness to use should therefore remain separate from comprehension measures.

---

## Observed System Behavior and Metrics

The scenario concerned raising a workplace concern about assignments favoring colleagues with postgraduate qualifications. Relevant details included seven years on the team, training new staff, and not having a postgraduate degree.

| Metric | Recorded Result | Interpretation for This Session |
|---|---|---|
| Overall Risk | 0.75, HIGH | Highest estimated attribute risk for the input; not a validated probability of identification. |
| Primary Risk | Occupation | Highest-scoring attribute. |
| Attribute scores | Occupation 0.75; education 0.45; age 0.05; location 0.05 | The system flagged occupation and education more strongly than age or location. |
| Residual Anonymity Capacity | 2.70 | Sum of one minus each cumulative attribute score across four attributes. A heuristic index on a 0–4 scale, not Shannon entropy despite the interface's bits label. |
| Leakage Delta | 0.75 | First-turn cumulative overall risk compared with zero; not a reduction achieved by rewriting. |
| Inferential risk before / after | 0.75 / 0.75 | No measured reduction in inferential risk. |
| Explicit detections before / after | 1 / 0 | One DATE_TIME detection was removed; no hard PII was detected in the input. |
| Cosine Similarity | 0.9787 | Fallback lexical similarity, not a MiniLM embedding result. |
| Forward / Backward NLI | 0.9783 / 0.9574 | Fallback heuristic scores, not DeBERTa model assessments. |
| Composite Utility | 0.9733 | High fallback preservation score; not proof of semantic equivalence. |
| Rewrite engine | `presidio_fallback` | Served output matched the baseline redaction. |
| Rewrite changed | `true` | Text changed, but this alone does not indicate privacy improvement. |
| Verdict / Is win | `no_change` / `false` | The system did not classify the result as a privacy-protection win. |
| Backend / client latency | 192.463 ms / approximately 0.918 s | Processing and client-observed response time, respectively; neither measures comprehension. |

The only change in the served rewrite was replacing **seven years** with **`<DATE_TIME>`**. The occupation and education information remained. The researcher coded task-intent preservation as `true` and noted that the primary risk information was not rewritten.

These metrics matter to the comprehension study because several different messages coexist on the page: high input risk, high preservation, a changed output, and no measured inferential-risk reduction. Users need accessible explanations to distinguish them. The logs establish these system outcomes, but do not establish whether P01 understood each distinction.

---

## Interface Implications

The following are proposed design changes, not changes evaluated in Session 08.

| Current Term | Suggested Primary Wording | Information to Show Beside the Value |
|---|---|---|
| Overall Risk | Personal-information exposure risk | Higher means greater estimated exposure in this message. |
| Primary Risk | Most exposed information | Name the attribute, such as occupation. |
| Residual Anonymity Capacity | Privacy remaining across this conversation | Show the 0–4 range and that higher is better; retain the formula in details and avoid the misleading bits unit. |
| Leakage Delta | Added risk from this message | Explain the increase relative to earlier messages and the first-message baseline. |
| Cosine Similarity | How similar the two texts are | State whether the value uses a model or word overlap. |
| Forward NLI | Does the rewrite add unsupported information? | Explain that a higher entailment score suggests fewer unsupported additions; retain the original score direction. |
| Backward NLI | Does the rewrite keep the original details? | Higher suggests fewer details were lost. |
| Composite Utility | Overall meaning-preservation score | Higher is better, with a clear indication of the evaluation backend. |
| Persistent Pseudonyms | Replace identifying details with consistent aliases | Explain repeated-name consistency and disclose when baseline output is shown instead. |
| Presidio Explicit Baseline | Basic personal-information masking | Explain direct-information redaction and identify any regex fallback. |

Use short, visible explanations for essential interpretation. Keep technical names, formulas, and implementation details available as secondary information. Tooltips can supplement the explanation, but should not carry the entire burden of understanding the result.

For the recorded output, an appropriate visible outcome statement would be: **The text changed, but estimated inference risk did not decrease.** This preserves the distinction between a successful edit and a successful privacy intervention.

---

## Limitations and Follow-Up

- Only one participant and one controlled scenario are represented.
- There is no within-session comparison against an interface without tooltips or one with simpler primary labels.
- No tooltip interaction log or per-term comprehension assessment is available.
- The structured unclear-content field does not identify specific terms; the principal terminology finding relies on the supplementary debrief.
- The structured detection outcome remains `not_coded`; the expected inclusion of age should not be treated as a confirmed detection failure.

In future tests, ask the participant to explain selected results before providing assistance: what 2.70 means, whether a larger Leakage Delta is preferable, and whether a high utility score means privacy improved. Record their answer, correctness, confidence, and whether they needed a tooltip. Compare the existing terminology with plain-language primary wording while keeping the same numerical results.

## Conclusion

Session 08 identifies a comprehension problem that remained after technical-term annotations were introduced. The participant wanted the evaluation values but preferred a simpler way to understand them.

The recommended next iteration is to **retain the numbers, simplify the primary wording, and make the practical meaning of each result visible**. Technical definitions should remain available for users who want more detail. Whether this improves comprehension requires a further test.

## Status

- Participants completed: **1**
- Current range: **P01**
- Scenario: **S02, controlled workplace conflict**
- Evaluation status: **Current participant record complete; per-term follow-up needed**
- Findings: **Completed**
