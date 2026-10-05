# Session 06 — User Testing

## Overview

Session 06 evaluates how users interact with InferenceGuard under a silent-observer protocol. The current pilot includes four participants (P01–P04) across generative and controlled scenarios.

The evaluation focuses on:

- contextual disclosure;
- comprehension of privacy warnings;
- responses to privacy-preserving query rewriting;
- perceived usefulness; and
- willingness to adopt the system.

These results are preliminary and are intended to identify behavioral patterns and usability issues rather than support statistical conclusions.

## Study Materials

**Google Drive:**  
[Session 06 User Testing Materials](https://drive.google.com/drive/folders/1w6L4govgNyL-mayQUSpt5CI7dO4psfWJ?usp=drive_link)

The Drive folder contains the original participant records, system logs, and other testing materials.

---

## Current Participants

| Participant | Scenario | Mode | ADR | Usefulness | Would Use |
|---|---|---|---:|---:|---|
| P01 | Time-Off Planning | Generative | 1.0 | 1/10 | No |
| P02 | Workplace Conflict | Controlled | 1.0 | 5/10 | Yes |
| P03 | University Housing | Generative | 1.0 | 8/10 | Yes |
| P04 | Medication Follow-Up | Generative | 0.3 | 5–6/10 | No |

No pre-submission evaluator intervention was recorded for P01–P04.

---

## Preliminary Findings

### 1. Privacy Awareness Did Not Necessarily Lead to Adoption

Understanding an inferential privacy warning did not necessarily result in willingness to use the system.

P01 demonstrated correct warning comprehension but rated the system only 1/10 and indicated that they would not use it before submitting an AI query. The participant preferred allowing an LLM to retain some personal context to avoid repeatedly providing the same information and also expressed concern about whether the privacy tool itself might retain private information.

P04 similarly indicated a willingness to provide some personal information when doing so could improve the quality of an AI response.

These observations suggest a potential tension between **privacy protection and personalization**.

### 2. Perceived Protection Could Occur Without an Effective Rewrite

For P02 and P03, the system returned the original query without meaningful modification. Nevertheless, both participants accepted the output and reported positive attitudes toward the system.

P02 stated that the tool showed privacy risk and protected their privacy, while P03 rated the system's usefulness as 8/10.

This suggests a possible gap between **perceived protection and observable privacy intervention**. A privacy warning may increase the user's perception of protection even when the query itself remains unchanged.

### 3. Technical Privacy Metrics Were Not Fully Interpretable

P03 explicitly asked about the meanings of **Joint Entropy** and **Leakage Delta**.

This observation suggests that these technical metrics are not necessarily self-explanatory to users. Although they may be useful for system evaluation, a participant-facing interface may require plain-language explanations or a simpler presentation.

### 4. Users Expressed Different Privacy–Utility Preferences

Participants did not demonstrate a uniform preference for stronger privacy protection.

P01 preferred retaining contextual information to support personalization, while P04 was willing to disclose some personal information when it could improve the relevance of the AI response. In contrast, P02 and P03 expressed greater willingness to use the privacy tool.

These observations suggest that inferential privacy protection may need to support **selective disclosure** rather than assume that minimizing disclosure is always the user's preferred outcome.

---

## Observed System Behavior

The system did not meaningfully modify the original queries for P01, P02, or P03 despite presenting privacy-risk information.

For P04, the recorded system log contained a modification of explicit age information, while the researcher observed that this modification was not clearly presented in the interface.

The current pilot therefore indicates a potential mismatch among:

**risk detection → risk communication → rewriting behavior → user perception**

This relationship should be examined further in subsequent Session 06 tests.

---

## Preliminary Conclusion

The first four Session 06 tests suggest that the usability of InferenceGuard depends on more than inference-risk detection alone.

Three emerging dimensions appear particularly important:

1. whether users understand what information may be inferred from their queries;
2. whether users can distinguish a privacy warning from an actual privacy intervention; and
3. whether the privacy benefit of removing contextual information outweighs the perceived loss of personalization or response quality.

At this stage, these findings should be treated as exploratory rather than generalizable. However, the P01–P04 results indicate that **trust calibration, privacy–utility preferences, and transparency of rewrite behavior** are important dimensions for the remaining Session 06 evaluation.

---

## Status

- Participants completed: **4**
- Current range: **P01–P04**
- Evaluation status: **Ongoing**
- Findings: **Preliminary**