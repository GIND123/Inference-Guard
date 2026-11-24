# Production readiness — engineering view

**Owner:** Chatur Bandaru (Engineering) · **Date:** 2026-10-17 (Session 07) · **Status:** first draft; measured figures are labelled with the command and hardware they came from, everything else is `TBD`.

This answers the professor's production questions from the engineering side:
what we would deploy, why latency matters, what hosting costs, whether "an
LLM could just do this", why we measure what we measure, and what is missing.
The legal side (state privacy law, ToS, consent) belongs to Product and
Users&Research. Section 6 lists the engineering facts they need for it.

Every number in this document traces to one of:

- `artifacts/latency/*.json`: written by `scripts/benchmark_latency.py`;
- `configs/models.yaml`: model sizes checked against the Hugging Face API on 2026-09-21;
- a provider's public pricing page, with the date it was checked.

---

## 1. Deployment target

| | (a) On-device (desktop app / browser extension) | (b) Self-hosted server, run by the user's organisation | (c) Our hosted SaaS |
|---|---|---|---|
| Where raw text goes | Stays on the user's machine | The organisation's own server | **Our** server |
| Who could read it | The user | The org's admins | Us, our cloud provider, anyone who breaches us |
| Matches "local-first"? | **Yes** | Partly: local to the org, not the person | **No** |
| Hardware | User's laptop CPU (or integrated GPU) | Org buys or rents a GPU | We rent GPUs |
| Model size we can afford | Small: Qwen3-0.6B, ModernBERT-base | Up to Qwen3-1.7B or larger | Anything |
| Marginal cost to us | ~0 (one-time download) | 0 (the org pays) | GPU hours (Section 3) |
| Update path | Ship new weights to every client | The org pulls a new image | We redeploy |
| Main engineering risk | Latency on weak laptops; download size | Org has to operate it | We become a store of exactly the text the product exists to protect |

The product's claim is that it runs *before text leaves user control*. Only
(a) delivers that. (b) is acceptable for an employer that already holds its
employees' text. (c) needs the most care. To rewrite a message we would
receive it raw. A sanitiser that collects every unsanitised message in one
place is a worse privacy position than not using one. We would offer (c) only
as a demo, or with a contractual no-retention promise, and that promise
cannot be verified by the user.

**Recommendation:** (a) is the target. (b) is the fallback for organisations
whose users have weak hardware. Today's Colab + tunnel demo is a stand-in
for (c) and should be described as a demo, not as the product.

## 2. Latency budget, and why latency is critical

InferenceGuard runs between the user pressing *send* and the message leaving.
It is in the critical path of typing. Whatever it costs is added to **every**
message, including the majority that are low-risk and come back unchanged.
A user who waits a few seconds per message will turn the tool off, and a
privacy tool that is turned off protects nobody. This makes latency a
precondition for the privacy metric, not a nicety beside it. Our north-star
number only counts on messages that pass through the tool.

Two paths matter:

- **Fast path:** a LOW-risk message. Presidio + risk model + utility; no rewrite, no rescore.
- **Slow path:** a message that is rewritten. Adds rewrite generation and a second risk pass.

The targets below are a design budget we set, not a measurement. They come
from a simple rule. The fast path should be unnoticeable next to the network
round trip of the chat service itself. The slow path is allowed to be
noticeable, because the user is being shown a warning anyway, but it should
not feel like a stall.

| Stage | Target p95 | Measured p50 / p95, Apple M4 Pro CPU, heuristic rewriter | Measured p50 / p95, Apple M4 Pro CPU, Qwen3-0.6B (base weights) | Colab T4 / A100, Qwen3-1.7B |
|---|---:|---:|---:|---:|
| Presidio (explicit PII) | 20 ms | 0.06 / 0.10 ms ¹ | 0.06 / 0.10 ms ¹ | TBD |
| Risk model (ModernBERT-base) | 50 ms | 26.54 / 39.90 ms ² | 25.79 / 37.89 ms ² | TBD |
| Rewrite | 1,500 ms | 0.09 / 0.47 ms | **1,561.20 / 3,494.75 ms** | TBD |
| Rescore (ModernBERT again) | 50 ms | 26.78 / 39.36 ms ² | 25.95 / 38.92 ms ² | TBD |
| Utility (serving path) | 20 ms | 0.09 / 0.19 ms ³ | 0.10 / 0.21 ms ³ | TBD |
| **Total, slow path** | **1,700 ms** | **53.51 / 79.80 ms** | **1,616.83 / 3,570.85 ms** | TBD |
| Peak process memory (RSS, fp32) | — | 1,021.5 MB | 4,226.0 MB | TBD |

Source: `artifacts/latency/cpu-apple-m4-pro_2026-10-17_heuristic.json` and
`artifacts/latency/cpu-apple-m4-pro_2026-10-17_qwen3-0.6b.json`, produced by
`python scripts/benchmark_latency.py --rewriter {heuristic,qwen3-0.6b} --device cpu --runs N`
on an Apple M4 Pro (12 cores, 8 torch threads), torch 2.14.1 CPU, transformers 5.20.1, fp32. Heuristic: 5 runs × 20 messages (n=100 per stage); Qwen3-0.6B: 3 runs × 20 (n=60).
20 synthetic messages, 12–104 words. An earlier identical run in the same
session gave 48.13 / 70.66 ms (heuristic) and 1,428.75 / 2,992.72 ms (Qwen3-0.6B)
for the total, so treat about ±10–15% as run-to-run noise on this laptop. Every message is pushed through every
stage, so these are slow-path figures. The GPU column is filled by the Colab
commands in the PR description.

**What the measurements say.** ModernBERT inference, run twice, accounts for nearly
all of the heuristic path, and that path is well inside budget. With a
generative rewriter, generation is the whole story. Qwen3-0.6B in fp32 on a
current laptop CPU has a p50 just inside the 1.7 s slow-path budget, but a p95
of 3.6 s, about twice the budget. So the on-device claim does **not** hold
at p95 today in this configuration. A quantised CPU build, a shorter
`max_new_tokens`, and dropping the few-shot examples from the prompt are the
levers to measure next. Qwen3-1.7B on CPU was not run: it is 2.7× the
download of 0.6B (Section 3) and would be slower still.

Caveats on these figures, which a reader must not skip:

1. ¹ Presidio was **not installed** in the benchmark environment, so this is the regex fallback in `PresidioBaseline`, not Presidio + spaCy `en_core_web_lg`. Real Presidio will be slower. `TBD — measure on Colab with presidio-analyzer installed`.
2. ² ModernBERT-base backbone with **untrained heads**: the trained checkpoint is not in the repo. Latency depends on architecture and input length, not on head weights, so this is representative. The scores it produced are meaningless.
3. ³ The served utility check is the lexical fallback (`utility_backend: "fallback"` in every response). MiniLM and the DeBERTa-v3-large NLI cross-encoder are evaluation-time only and are not in this budget. Putting DeBERTa-large (0.87 GB fp16, `configs/models.yaml`) into the serving path would need its own measurement.
4. Qwen3-0.6B was run with base weights through the real `QwenRewriterInference.rewrite()` path (same prompt, greedy, `max_new_tokens=128`, same output cleaning), because no LoRA adapter is committed. Unmerged LoRA adds a small per-token cost. Merging the adapter before deployment removes it.
5. The model runs twice per rewritten message (input and rescore). Batching the rescore with the next message, or skipping it when the rewrite engine is `presidio_fallback` and nothing changed, are cheap wins once we have GPU numbers.

**Qwen3 thinking mode must stay off.** Qwen3's chat template emits a
`<think>…</think>` block before the answer unless it is called with
`enable_thinking=False` (`configs/models.yaml`, `rewriter.gotcha`). That block
can be many times longer than the rewrite itself, and generation time scales
with tokens produced. Thinking mode would multiply the dominant stage of the
slow path. The output would also have to be stripped, and the stripping
regexes have known holes (seven `xfail` cases in
`tests/test_adversarial_token_sanitization.py`). `format_chat_prompt` passes
`enable_thinking=False` today. Keep it that way.

## 3. Hosting budget

Option (a) costs us nothing per message. The user downloads the weights once
and runs them on their own hardware:

| Component | Download (safetensors, Hub API 2026-10-17) | Weights in memory (`configs/models.yaml`) |
|---|---:|---:|
| ModernBERT-base risk model | 0.60 GB | 0.30 GB fp16 |
| Qwen3-0.6B rewriter (on-device candidate) | 1.50 GB | 0.41 GB NF4 |
| Qwen3-1.7B rewriter (quality ceiling) | 4.06 GB | 4.06 GB fp16 / 1.12 GB NF4 |

`configs/models.yaml` gives weights only, not a minimum laptop RAM. The
measured peak process memory in Section 2 includes Python, torch and both
models in fp32 on CPU. That is an upper bound for the fp32 path, and the
honest number to quote until a quantised CPU build is measured.
`TBD — measure peak RSS for an int8/NF4 CPU build`.

For (b) and (c), we pay per GPU hour:

```
msgs_per_gpu_hour   = 3,600,000 ms / slow_path_ms_per_msg × concurrent_streams
cost_per_1k_msgs    = gpu_hourly_price / (msgs_per_gpu_hour / 1000)
monthly_cost        = max(always_on_floor, msgs_per_month / msgs_per_gpu_hour × gpu_hourly_price)
always_on_floor     = 720 h × gpu_hourly_price     # one GPU kept warm all month
```

`slow_path_ms_per_msg` is the measured total from Section 2. Using it for every
message overstates cost, because LOW-risk messages skip the rewrite. The share
of messages that take the slow path is `TBD — from real usage`.

| GPU | Price / GPU-hour | Source (checked 2026-10-17) | Measured slow-path ms/msg | Cost / 1k msgs | Always-on floor / month |
|---|---:|---|---:|---:|---:|
| NVIDIA T4 16 GB | TBD — look up | not listed on Lambda or RunPod | TBD | TBD | TBD |
| NVIDIA L4 24 GB | $0.59 | RunPod Secure Cloud, https://www.runpod.io/pricing | TBD | TBD | $424.80 |
| NVIDIA A100 40 GB | $1.99 | Lambda on-demand, https://lambda.ai/pricing | TBD | TBD | $1,432.80 |
| NVIDIA A100 80 GB | $1.79 | RunPod Secure Cloud, https://www.runpod.io/pricing | TBD | TBD | $1,288.80 |

The floor column is plain arithmetic (720 × price). Every other cost cell
depends on GPU throughput we have not measured, so it stays `TBD` until the
Colab runs in the PR description land.

At three usage levels, with the same formulas:

| Usage | Messages / month | (a) on-device | (b)/(c) cost |
|---|---:|---:|---|
| Class pilot (10 users × 50 msgs/day) | 15,000 | $0 | the always-on floor dominates: one GPU idle most of the month |
| Small org (1,000 users × 50/day) | 1,500,000 | $0 | `TBD` = 1.5M / msgs_per_gpu_hour × price |
| Large org (50,000 users × 50/day) | 75,000,000 | $0 | `TBD`, likely several GPUs; throughput measurement first |

The shape is clear even without the GPU numbers. At small scale a hosted
deployment costs the always-on floor whatever the traffic, which argues
again for (a). Colab, which we use today, has no SLA and drops sessions.
It is not a hosting option.

## 4. "Can't an LLM just do this?"

Three honest parts.

1. **A cloud LLM sanitiser defeats the purpose.** To have GPT-class model X rewrite a message, you send X the raw message. The party we protect the text from is very often the operator of that same model. Prompting a cloud LLM to "remove anything that reveals my age or city" is a privacy request delivered to the recipient.
2. **Only a local model keeps the raw text on the device.** That forces a small model: 0.6B–1.7B parameters, which fit and run on a laptop (Section 3). A small model zero-shot is weaker than a frontier model at this task. Fine-tuning is how we try to close that gap for this one narrow job.
3. **The empirical answer is an ablation we have not run yet.** The comparison that settles it is:
   - prompt-only rewrite (same small model, no fine-tuning);
   - our fine-tuned rewriter;
   - Presidio redaction;
   - all scored by the **held-out Phi-4-mini attacker**, on the same test profiles, with the same strict success rule.

   This PR makes that attacker trustworthy (Section 5). The run itself is Data&Eval's Session 07 deliverable. **Until it runs, we do not claim that we beat a prompted LLM.** Today the rewriter is selected on our own risk model and scored on it. Any number from that loop measures how well the rewriter games our own scorer.

## 5. Why these metrics

**Held-out attacker accuracy (north star).** The product's job is to stop an
adversary from recovering attributes, so the metric is how often an adversary
gets them right, before vs. after. It must be a model we never use for
selection, thresholds or SFT filtering. Otherwise the rewriter is optimised
against it and the number measures overfitting to the judge (methodology
review, finding 4). Phi-4-mini is a different vendor, tokenizer and
pretraining from the rewriter. Two engineering fixes in this PR make the
number mean what it says. The attacker now decodes only the model's answer:
it used to parse the prompt's own `<prediction>` template. And it refuses
to fall back to regex silently. Every result carries `mode`, and the report
records `attacker_mode`, `attacker_model_id` and `reportable_as_held_out`.

**Utility guardrail.** A rewrite that deletes the message is perfectly
private and useless. We need a floor. Cosine similarity alone cannot be that
floor: MiniLM scored `"123 an area, an area an area"` at 0.907 (finding 6).
The guardrail now combines bidirectional NLI on real three-way probabilities
with a documented degeneracy rule (repetition, length ratio, non-alphabetic
share), and reports which backend produced the score. On the heuristic
rewriter the rule already caught a live failure. The location regex turns
"I am moving from Denver, CO to Seattle, WA to start as a software engineer…"
into "I am moving locally."

**Latency.** Section 2. Latency decides whether the tool stays on. It is a
precondition for the privacy metric, so it is a metric.

**Rewrite-changed rate.** The share of non-LOW messages whose served text
differs from the input (`rewrite_changed`, per response). Session 06 showed
that users treat an unchanged message as protected (P02, P03, P05). A high
risk score on an unchanged message means *risk detected, nothing done*.
This rate tells us how often that happens. The UI now says so in plain words
instead of showing the original as a "result".

## 6. Data handling controls

Engineering controls only.

| Control | State today |
|---|---|
| No raw-text logging in the API | **Have.** `web/api.py` logs model-loading events and errors, not message text. The new `timings_ms` are numbers only. Uvicorn's default access log does record client IP, method and path per request (not the body). Run with `--no-access-log` for real users. |
| Hash-only audit of served text | **Have (this PR).** `served_text_sha256` is the SHA-256 of exactly the `rewritten_text` returned, so a study log can be checked against what the UI showed (the P04 mismatch) without storing the text. |
| Engine transparency | **Have (this PR).** `rewrite_engine` ∈ {`qwen_adapter`, `heuristic`, `presidio_fallback`, `passthrough_low_risk`} and `rewrite_changed` on `/analyze` and `/rewrite`. |
| Session reset | **Have (this PR).** `DELETE /session/{id}` clears a session's conversation/turn/risk history via `ConversationTracker.delete_session` and is idempotent (200 with a `cleared` flag). Memory no longer grows until restart: `ConversationTracker.evict_idle` reaps idle sessions past a TTL and LRU-evicts over a cap, called at the start of `/analyze` using `IG_SESSION_TTL_SECONDS` (default 3600) and `IG_MAX_SESSIONS` (default 10000). |
| Conversation state at rest | **None.** The server keeps state in RAM only; nothing is written to disk. The UI's "Download Result" exports the input text to the user's own machine, by the user's action. |
| CORS | **Safer default (this PR).** `allow_origins` still defaults to `["*"]` so the unauthenticated Colab demo works, but the wildcard is no longer combined with credentials: `cors_allow_credentials(origins)` returns `False` for a wildcard and `allow_credentials=True` only when `IG_ALLOWED_ORIGINS=https://host1,https://host2` lists explicit origins. Set explicit origins for anything beyond a class demo. |
| Public exposure | **Demo only.** The README exposes the Colab server through localtunnel/ngrok on a public URL with no authentication. Acceptable for a class demo with synthetic text. Not acceptable for real user text. |
| Consent screen | **Hook needed.** The UI has no consent step. Engineering can add a blocking first-run screen whose wording comes from Users&Research and Product. |

### For Product and Users&Research: legal analysis (to be filled by them)

#### Applicable state and federal privacy law (e.g. CCPA/CPRA)
#### Terms of Service wording
#### Consent requirements and the consent screen text
#### Data retention statement

Engineering facts they will need:

- **What is processed:** the text of each message the user submits, plus a session id the user chooses. No account. Our code does not log IPs, but uvicorn's default access log does (IP, method, path), and the hosting platform or tunnel may as well.
- **Where:** in deployment (a), on the user's device. In today's demo, on a Google Colab VM, reached through a public localtunnel/ngrok URL.
- **Retention:** in-process memory for the life of the server process (conversation history per session id). Nothing is written to disk by the API. The user's own "Download Result" file is under their control.
- **Third-party components and licences** (from each model card / repository, checked 2026-10-17):

| Component | Licence |
|---|---|
| SynthPAI dataset (`RobinSta/SynthPAI`) | CC-BY-NC-SA-4.0 (**non-commercial**; also share-alike for derived data such as our SFT set) |
| ModernBERT-base (`answerdotai/ModernBERT-base`) | Apache-2.0 |
| Qwen3-0.6B / Qwen3-1.7B (`Qwen/…`) | Apache-2.0 |
| Phi-4-mini-instruct (`microsoft/Phi-4-mini-instruct`) | MIT |
| all-MiniLM-L6-v2 (`sentence-transformers/…`) | Apache-2.0 |
| nli-deberta-v3-large (`cross-encoder/…`) | Apache-2.0 |
| Presidio (`microsoft/presidio`) | MIT |

The model licences permit commercial use. The dataset licence does not. Any
model trained on SynthPAI (our risk heads and SFT adapter) inherits that
question for a commercial product. That is for Product to resolve.

## 7. Production gap checklist

| Area | Exists today | Production needs |
|---|---|---|
| Model packaging | Risk checkpoint, RoBERTa risk model and LoRA rewriter are now **hosted on the Hub** (`AT707/inferenceguard-modernbert-risk`, `AT707/InferenceGuard-Roberta-Risk`, `AT707/inferenceguard-rewriter-v1`); weights stay out of git. `artifacts/model_manifest.json` (this PR) enumerates the three authoritative repos with `expected_files` and a `verified` flag; `/health` now reports `risk_model_loaded` + `risk_model_source` (`model` vs `fallback_rule_based`) so a server silently serving the rule-based fallback is visible, and strict mode (`IG_REQUIRE_MODEL`) makes `/health` return 503 in that case. | Pin revisions + per-file checksums in the manifest via `scripts/verify_hf_models.py --write-manifest` on a networked host; gate deploy on a verified manifest. |
| Model versioning | `DEFAULT_BASE_MODEL` shared constant; adapter's own config trusted for its base. `configs/models.yaml` now carries a `revision:` field for each hosted model (`risk_estimator.roberta_hub`, the new `risk_estimator.modernbert_hub` block, and `rewriter.hub_model`); pins are null with a TODO until `verify_hf_models.py --write-manifest` populates the real shas from the Hub (never fabricated in-sandbox). | Real revision shas + adapter hash pinned in `configs/models.yaml` / `artifacts/model_manifest.json` and surfaced in every response. |
| CI | **Added in this PR:** `.github/workflows/tests.yml`, CPU-only, slow tests opt-in. | Lint, and a scheduled job that runs `--run-slow` on a GPU runner. |
| Monitoring | None. | Per-stage latency histograms (the `timings_ms` already exist), error rates, fallback rates by engine, rewrite-changed rate. No text. |
| Error budgets / SLO | None. | e.g. p95 slow path under the Section 2 budget for 99% of a week; alert on breach. |
| Rate limiting | None. | Per-client limits; generation is the expensive stage. |
| Auth | None: open endpoint behind a public tunnel. | Token auth for (b)/(c). Not needed for (a), where the API binds to localhost only. |
| Input limits | ModernBERT truncates at 512 tokens; no request size cap. | Request body cap; reject or chunk long inputs explicitly. |
| Session lifecycle | In-memory, now bounded (this PR): `DELETE /session/{id}` clears a session, and `evict_idle` enforces a TTL (`IG_SESSION_TTL_SECONDS`, default 3600) plus a max-session LRU cap (`IG_MAX_SESSIONS`, default 10000) on each `/analyze`. | Persist/offload for multi-process deployments; background reaper if traffic is too sparse to trigger inline eviction. |
| Held-out evaluation | Attacker fixed in this PR; full run pending (Data&Eval). | Run in CI on a fixed test split before each model release. |
