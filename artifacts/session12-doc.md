# Session 12 — Privacy Rewrites and Task-Relevant Details in a Moving Decision

## Overview

Two user tests examined whether InferenceGuard protected information participants considered private while preserving the details needed to decide whether to move. Both records use scenario S-01. The analysis pairs two participant records with two system exports, `01.json` and `02.json`. Video: https://drive.google.com/drive/folders/1hqvMsQJXy6SOEdHDfLBGOVFI5qlawlQj?usp=drive_link

Before seeing the result, participants explained whether they would send their original question, what they considered private, and whether they wanted a privacy rewrite. After seeing the result, they reviewed the changes and selected which version they would send. Both records describe generative testing with no facilitator intervention before submission. The second input closely follows the full scenario card, whereas the first is a shorter formulation.

## Participant Feedback

| Participant | Before seeing the result | After seeing the result |
|---|---|---|
| anonymous-01 | Would send the original and initially identified no private details. Nevertheless, wanted a privacy rewrite to protect their privacy. | Understood that the system flagged education and occupation, and was surprised that education was treated as private. Chose the original because the graduation timeline and lease-end month helped the AI calculate the moving decision. Was unsure about future use, suggesting they might use the tool for tasks containing private information. |
| anonymous-02 | Would send the original. Identified the December lease end, Green Line, and 55-minute commute as private details. Was unsure about a rewrite and would consider it if the question's meaning was preserved. | Understood the location warning but questioned why Green Line remained visible. Chose the original because removing both commute durations left the AI without enough information to compare the apartments. Would not use the tool before sending because it did not help with the task. |

Neither participant recorded a manual edit. Recorded usefulness ratings were 6 and 2, respectively. The supplied record does not specify the rating scale. Both participants selected the original after reviewing the suggestion.

## Observed System Behavior and Metrics

| Test | Original risk | Returned text | Composite utility |
|---|---:|---|---:|
| anonymous-01 · `01.json` | 0.50 · Medium, primarily education with occupation secondary | Presidio baseline. Masked the eight-month graduation timeline, December lease end, and four-day weekly on-site schedule. Retained the undergraduate degree and approximately $1,200 cost. | 0.9170 |
| anonymous-02 · `02.json` | 0.75 · High, primarily location with education and occupation secondary | Presidio baseline. Masked dates, renewal wording, work frequency, both commute durations, and monthly rent wording. Retained Green Line, undergraduate degree, second co-op, and the $1,200 moving cost. | 0.9327 |

The Colab deployment reported the risk model and Qwen adapter as loaded. Both exports confirm that the Qwen adapter generated a candidate without a recorded generation error. The pipeline's decision reason states that no candidate satisfied the Pareto risk threshold, so both requests ultimately served the Presidio baseline (`presidio_fallback`).

The internal Qwen candidate for the first request generalized the undergraduate degree to “my studies.” The second candidate also generalized this phrase but retained Green Line and ended with an incomplete question. These candidates were not the suggestions shown to participants. The exports do not provide candidate-level scores explaining each rejection, so the cause of rejection cannot be attributed to a particular text defect from these logs alone.

Overall inferential risk remained 0.50 for the first request and 0.75 for the second after the served rewrites. Both comparisons recorded `inferential_delta: 0` and `is_win: false`. The reported explicit-entity counts fell from 4 to 0 and 9 to 0, respectively. All detected entities in those counts were `DATE_TIME`, including durations and recurring schedules needed for the task.

Utility evaluation used the fallback backend for both requests. The high composite scores therefore coexisted with participant-reported losses of necessary information. In particular, the second suggestion scored 0.9327 after removing both commute durations.

## Findings

- **Recognizing private information did not establish willingness to remove it.** The second participant identified privacy cues before submission and still chose the original after seeing how the suggestion affected the commute comparison. The first participant requested privacy protection despite initially identifying no private details, then prioritized the dates needed for the decision.
- **The served suggestions removed necessary details while retaining flagged cues.** The first output kept the undergraduate degree despite education being the primary risk. The second kept Green Line despite location being the primary risk. Both suggestions masked scheduling information, and the second removed both commute durations.
- **A loaded rewriter did not ensure that users saw its output.** Qwen generated candidates for both requests, but the pipeline selected Presidio for both. The user feedback describes reactions to those served suggestions.
- **Aggregate utility scores did not capture the participants' task concerns.** Scores of 0.9170 and 0.9327 accompanied rejection of both suggestions. Future checks should verify preservation of the specific quantities required to answer the question, alongside privacy reduction.

