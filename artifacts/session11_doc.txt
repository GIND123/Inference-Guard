# Session 11 — Privacy Judgments Before and After Seeing a Rewrite

## Overview

Four user tests were carried out on whether InferenceGuard's suggestions protected the details they considered private while preserving the information needed for their tasks. 

Three scenarios used participant-written prompts (moving, workplace discussion, and contacting a GP). The CV scenario used a controlled prompt. The analysis pairs four participant records with six unique system submissions. Two scenarios had a second submission, and one duplicate CV export was excluded.

Video: https://drive.google.com/drive/folders/1cPrtL-Ac9COAxPd9CWPgNTHRAb3FYgA8?usp=drive_link

## Participant Feedback

| Scenario | Before seeing the result | After seeing the result |
|---|---|---|
| S-01 · Moving decision | Would send the original and declined a privacy rewrite. Identified the $1,200 upfront cost as financial information but wanted enough detail for the AI to calculate the trade-off. | Chose the original and questioned why useful details had been removed. The record describes restoring both commute times and the monthly cost wording, and removing the Green Line reference in a trial edit. |
| S-02 · Workplace discussion | Was unsure whether to send the original or request a rewrite. Identified the job title and regional employer description as details that could help someone identify them. | Chose the original. The participant restored the detail of seven-year because it supported the case for the assignment. |
| S-03 · GP follow-up | Identified exact age, medication, and GP location as private, but would send the original and declined a rewrite because the details could help the AI respond. | Received the original unchanged. Questioned the 0.45 age score because the prompt stated their exact age, and suggested adding health information to the risk categories. |
| S-04 · CV editing | Would send the original as written and declined a privacy rewrite. The participant considered the identifying details appropriate for the CV task. | Chose the edited version because the suggestion masked the email address and phone number. The participant added a request for an editable file. |

## Observed System Behavior and Metrics

| Scenario | Original risk | Returned text | Composite utility |
|---|---:|---|---:|
| S-01 · First submission | 0.45 | Presidio baseline. Masked both commute times and monthly cost wording. Retained the $1,200 cost and Green Line reference. | 0.9539 |
| S-01 · Resubmission | 0.45 | Presidio baseline. Retained 25 minutes and $1,200 but masked 55 minutes and monthly cost wording again. | 0.9844 |
| S-02 | 0.75 | Presidio baseline. Retained the job title and masked tenure and meeting timing. | 0.9506 |
| S-03 | 0.45 | Returned the original unchanged, including exact age, medication, and GP location. | 1.0000 |
| S-04 · First submission | 0.75 | Heuristic rewrite. Masked city, email, and phone. Retained name, street address, and birth year. The system flagged remaining explicit personal information. | 0.9334 |
| S-04 · Resubmission | 0.05 | Presidio baseline on an input already containing placeholders. Masked name and years of experience, partly masked the street address, and retained birth year. | 0.8878 |

Using current running method (step 0 - 3) in the github repo, the Qwen adapter was unloaded and utility scoring used the fallback backend for all six submissions. Running directly from the InferenceGuard_finetune.ipynb Colab notebook doesn't work either. 
These results characterize the deployed fallback behavior. The 1.0000 utility score for S-03 reflects identical text. The 0.9844 score for the second S-01 submission coexisted with a missing commute time needed to compare the two apartments.



