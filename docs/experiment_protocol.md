# InferenceGuard Experiment Protocol

## 1. Purpose

This document defines the standard procedure for running, recording, validating, and reporting experiments in InferenceGuard.

The goal is to make every experiment reproducible, auditable, and comparable across sessions. Raw evidence should remain unchanged after collection, while derived metrics should be reproducible from the raw artifacts and experiment configuration.

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

---

## 3. Experiment Configuration

Every formal experiment must have a configuration file stored under:

`configs/experiments/`

Each configuration should record at least:

- experiment ID
- session number
- experiment type
- dataset name
- dataset revision
- data split
- random seed
- risk model checkpoint
- rewriter checkpoint
- attacker model
- scenario set
- metrics
- output directory

A result should not be reported unless the configuration used to produce it is available in the repository.

---

## 4. Experiment IDs

Each experiment must have a unique ID.

Recommended format:

`sessionNN_<experiment_name>_vX`

Examples:

- `session06_user_eval_v1`
- `session06_adversarial_eval_v1`
- `session07_multiturn_eval_v2`

The experiment ID should appear in:

- the config file
- the output directory
- generated summaries
- weekly report references

---

## 5. Directory Structure

Each experiment should write outputs into:

`experiments/<experiment_id>/`

Recommended structure:

```text
experiments/
└── session06_user_eval_v1/
    ├── config.yaml
    ├── manifest.json
    ├── raw/
    ├── derived/
    └── README.md
