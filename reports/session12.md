---
team: InferenceGuard
session: 12
date: 2026-11-24
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
    Computed for the first time against the live hosted models. On the 50-sample
    PromptPAI test split under microsoft/Phi-4-mini-instruct (attacker_mode=model,
    reportable_as_held_out=True), Attack Success Rate (ASR) drops from 0.3650
    (raw) and 0.3600 (Presidio baseline, -0.5% absolute) to 0.1500 (58.9% relative
    leakage reduction) using the hosted 1.5B QLoRA rewriter
    (AT707/inferenceguard-rewriter-v1 @ a84ca5623236ebdc325128e44baa25ec5494d592)
    at p95 = 2,232 ms (36% faster than zero-shot prompting). The served ModernBERT
    risk classifier (AT707/inferenceguard-modernbert-risk @
    4cecb3c90e84112ce5a5e3ed173060a1ab9609a7) scores Macro ROC-AUC 0.8401, Macro
    F1 0.4993, and calibrated ECE 0.0602.
  previous: >-
    Session 11: north star still not computed; served RoBERTa checkpoint scored
    0.453 ROC-AUC (below chance) and multitask ModernBERT scored 0.539.
---

## Shipped this week

- **Retrained, calibrated, and hosted the served ModernBERT risk classifier and 1.5B QLoRA rewriter.**
  Published `AT707/inferenceguard-modernbert-risk` pinned to revision `4cecb3c90e84112ce5a5e3ed173060a1ab9609a7` (Macro ROC-AUC `0.8401`, Macro F1 `0.4993`, temperature-calibrated ECE `0.0602`) and `AT707/inferenceguard-rewriter-v1` pinned to revision `a84ca5623236ebdc325128e44baa25ec5494d592` in `artifacts/model_manifest.json`. (evidence: `artifacts/risk_eval/inferenceguard-modernbert-risk_val_2026-11-18.json`, `reports/promptpai/comparison_matrix_test.md`)

- **Fixed 4-bit Qwen GPU meta-tensor offload and cleaned up rewrite rules.**
  Pinned `UtilityEvaluator` (`MiniLM` and `DeBERTa-v3`) to CPU by default and set `device_map={"": 0}` with a post-load meta-tensor check in `QwenRewriterInference` so ModernBERT and Qwen2.5-1.5B 4-bit NF4 run concurrently in `1,830 MB` VRAM on a 6 GB RTX 4050. Replaced awkward `"locally"` city fillers with clean clause excision, fixed article duplication, and added Presidio false-positive filters for statutes, ordinals (`3rd-year`), and standalone `US` tags. (evidence: stacked PR 1 `feat/core-models-and-rewriter-gpu`)

- **Zero-plaintext HMAC-SHA256 session aliasing and multi-turn novelty discounting.**
  Updated `DeterministicPseudonymManager` to hash entity surfaces with a session-scoped `HMAC-SHA256` key (`X-Session-Secret`) so server memory stores zero plaintext PII. Added street address (`ADDRESS_k`) and always-on phone (`PHONE_k`) aliasing, honorific/first-name coreference linking (`Sarah Chen` -> `Ms. Chen` -> `Sarah` = `PERSON_1`) with ambiguous-word guards, `StreamAliasRestorer`, and token-overlap novelty discounting in `ConversationTracker`. (evidence: stacked PR 2 `feat/zero-trust-aliasing-and-multiturn`)

- **Dual-edition B2B customer support gateway, webhook middleware, and Python SDK.**
  Added `privacy_mode` (`personal_balanced`, `personal_strict`, `business_support`) to `/analyze` and `/v1/chat/completions`, support identifier and state-jurisdiction preservation, single-attribute occupation k-anonymity calibration, header-based OpenAI SDK integration (`X-Privacy-Mode`, `X-InferenceGuard-Restore: gateway`), `POST /v1/support/webhook`, and `src/sdk/support_agent.py`. Fixed `/ready` and `/status` to report active Hugging Face Hub model IDs with thread-safe initialization locks. (evidence: stacked PR 3 `feat/b2b-support-gateway-and-sdk`)

- **Split-Brain live chat workspace, 1-click presentation launcher, and 50-persona GPU benchmark.**
  Upgraded `web/` with the Personal Privacy Shield mode selector and Business Support Gateway split-pane live chat (local restored view on the left, zero-PII Cloud LLM Wiretap + CRM Safe-Log on the right), added `scripts/launch_presentation_app.py` and `launch_presentation_demo.bat`, and ran the 50-persona 10-domain GPU benchmark. (evidence: stacked PR 4 `feat/dual-edition-ui-and-benchmarks`, `reports/persona_50_domain_report.md`)

## User evidence

- Final pre-demo user study template prepared in `docs/userstudy/session12_template.json`.
- Addressed the Session 11 user-study finding (rewrites dropping task-critical numbers or leaving unprotected cues) via `_retains_critical_numbers()`, birthdate neutralization, and k-anonymity risk calibration.
- Addressed the Session 11 setup blocker (Qwen adapter failing to load locally) via the 4-bit GPU `device_map={"": 0}` fix and the 1-click launcher (`launch_presentation_demo.bat`).

## Metrics snapshot

### Served risk classifier (retrained and calibrated)

Evaluated on the same 45 held-out validation profiles (`artifacts/risk_eval/inferenceguard-modernbert-risk_val_2026-11-18.json`):

| model | macro F1 | macro ROC-AUC | ECE |
|---|---:|---:|---:|
| RoBERTa `8273bc7c` (Session 11 container) | 0.019 | 0.453 | 0.390 |
| Multitask ModernBERT `0ed8bdcc` (Session 11 eval default) | 0.116 | 0.539 | - |
| **ModernBERT `4cecb3c9` (Session 12 served)** | **0.499** | **0.840** | **0.060** |

### Held-out adversary evaluation (PromptPAI test split, n=50)

Adversary: `microsoft/Phi-4-mini-instruct` (`attacker_mode="model"`, `reportable_as_held_out=True`), from `reports/promptpai/comparison_matrix_test.md`:

| method | Attack Success Rate (ASR) | relative ASR reduction | p95 latency (ms) |
|---|---:|---:|---:|
| Raw input | 0.3650 | 0.0% | - |
| Microsoft Presidio baseline | 0.3600 | -1.4% | - |
| Zero-shot Qwen prompt-only | 0.1850 | -49.3% | 3,499 |
| **InferenceGuard 1.5B QLoRA (`a84ca562`)** | **0.1500** | **-58.9%** | **2,232** |

### 50-Persona 10-Domain Live GPU Benchmark

From `reports/persona_50_domain_report.md` (RTX 4050, `1,830 MB` peak VRAM):

| metric | value |
|---|---:|
| Zero hard-PII leakage rate | 100.0% (50/50) |
| Zero regex/NER scar rate | 98.0% (49/50) |
| Critical ID / number / statute preservation | 98.0% (49/50) |
| Mean inferential risk (before -> after) | 0.8063 -> 0.4897 (-39.3%) |
| Mean cosine similarity / utility score | 0.9009 / 0.6633 |
| Engine breakdown | heuristic 33/50, qwen_adapter 16/50, presidio_fallback 1/50 |

### Is this the same model running in the product?

**Yes.** Both `web/api.py` and `Dockerfile` pull `AT707/inferenceguard-modernbert-risk` (`4cecb3c90e84112ce5a5e3ed173060a1ab9609a7`) and `AT707/inferenceguard-rewriter-v1` (`a84ca5623236ebdc325128e44baa25ec5494d592`) directly from Hugging Face Hub, pinned in `artifacts/model_manifest.json` and surfaced identically on `/health`, `/ready`, and `/status`.

## What did not work

- **CPU CI vs. local GPU environment drift surfaced multiple test assumptions.** GitHub Actions runs without spaCy/Presidio models and without a GPU, which exposed missing fallback person-name patterns, unguarded `/ready` 503 assertions, and a legacy test (`test_scenario_s01_tech_worker_relocation`) that still expected the old regex bug to emit `"I am moving locally."`. All were fixed and verified under an offline CI simulation (`analyzer=None`, `qwen_rewriter=None`).
- **Concurrent `/health` polling during cold start triggered duplicate Qwen loads.** Adding `_qwen_init_lock` and `_risk_init_lock` in `web/api.py` resolved the race condition.

## Challenges / blockers

- **Final PR stack review and merge before demo day.** All 4 stacked branches are pushed and verified; they need final review and sequential merge into `main`.
- **Unbatched local rewrite latency (~2 to 4 s on laptop GPU when neural rewrite runs).** Acceptable for the live presentation demo; production deployment would use vLLM/TensorRT-LLM batching and skip runtime NLI scoring.

## Next week's goal

Merge the 4 stacked PRs into `main`, complete the final demo dry run using `launch_presentation_demo.bat`, and present the end-to-end dual-edition system and held-out benchmark evidence on Demo Day (1 December).

## Individual contributions

- **Anna Thomas (Data&Eval):**
  - Retrained, temperature-calibrated, and hosted `AT707/inferenceguard-modernbert-risk` (`4cecb3c9`, Macro ROC-AUC `0.8401`, Macro F1 `0.4993`, ECE `0.0602`) and `AT707/inferenceguard-rewriter-v1` (`a84ca562`), fixed the 4-bit CUDA `meta tensor` placement bug so both models run concurrently in `1.83 GB` VRAM, and cleaned up location, age, birthdate, and statute rewrite rules.
  - Implemented zero-plaintext `HMAC-SHA256` session blind indexing (`PERSON_k`, `ADDRESS_k`, `PHONE_k`, `EMAIL_k`) with collision-safe honorific and first-name coreference linking, offline fallback extraction, `StreamAliasRestorer`, and token-overlap novelty discounting in `ConversationTracker`.
  - Built the dual-edition B2B customer support gateway (`/v1/chat/completions`, `/v1/support/webhook`, `src/sdk/support_agent.py`), Split-Brain live chat workspace, and 1-click presentation launcher, and ran the held-out `Phi-4-mini` PromptPAI benchmark (`58.9%` relative ASR reduction) and 50-persona 10-domain GPU benchmark across 4 stacked PRs.

- **Chatur Bandaru (Engineering):**
  - (To be added)

- **Govind Arun (Product):**
  - (To be added)

- **Yiwei Jin (Users&Research):**
  - (To be added)

- **Bingqi Lian (Operations):**
  - (To be added)

## Lean canvas changes (if any)

- **Dual-edition positioning crystallized for Demo Day:** InferenceGuard Personal (message inspector and personal prompt shield) serves as the user-facing privacy workflow, while InferenceGuard Business (drop-in OpenAI-compatible customer support gateway and helpdesk webhook middleware) targets the enterprise compliance pain point (GDPR Art. 5/17 minimization and zero-PII CRM/Datadog/Snowflake logs).
- **What we can now claim on demo day:** Backed by held-out `Phi-4-mini` adversary evaluation (`58.9%` relative ASR reduction vs. `1.4%` for Presidio), calibrated ModernBERT risk detection (`0.8401` ROC-AUC, `0.0602` ECE), and `100%` hard-PII removal with `98%` order/ticket/statute retention across 50 personas and 10 domains.
