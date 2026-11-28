# Known limitations

Findings from the Session 12 audit that are **real and not fixed**, with what
each one costs and what fixing it would take. Written down rather than carried
in someone's head, so the final report's Ethics and Limitations section can
cite it and so nobody rediscovers them in January.

Four findings from the same audit *were* fixed: whole-transcript protection,
the simulated-reply fallback, the ungated debug trace and the worker count.
Those are in `tests/test_proxy_protection_contract.py`.

Last reviewed: 2026-11-27.

---

## 1. There are two runtime paths, and they behave differently

| | entrypoint | used by |
|---|---|---|
| web app | `uvicorn web.api:app` | README, the Colab notebook, every user study so far |
| container | `uvicorn src.api.app:app` | `Dockerfile`, `docker-compose.yml` |

They are not thin wrappers over one core. The UI/proxy endpoints, candidate
selection, session handling and leakage calculation differ between them.

**What it costs.** A result measured against one path does not necessarily
describe the other. Every user session we have run went through `web.api`;
the hosted gateway runs `src.api.app`. When the report says "the product",
it is ambiguous which one is meant.

**What fixing it takes.** Move the protection logic into one service and
reduce both entrypoints to API adapters over it. That is a refactor of the
part of the system the demo depends on, which is why it is not happening four
days before demo day.

**Until then:** say which path a number came from. The held-out benchmark and
the user studies both used `web.api`.

---

## 2. The container can serve heuristics while the architecture says Qwen

`src/api/app.py:95` reads:

```python
load_model=os.getenv("LOAD_REWRITER_WEIGHTS", "false").lower() in ("true", "1", "yes")
```

Default **false**. The utility evaluator defaults to unloaded models as well.

**What it costs.** A deployment can answer requests, report healthy and
produce plausible rewrites from the regex fallback, while every diagram and
slide describes Qwen and DeBERTa. This is the same failure as the untrained
RoBERTa in Session 10 — a system that works well enough to look like it is
working.

**Partly mitigated.** `/v1/chat/completions` now returns `simulated_reply` and
`upstream_mode`, so a caller can tell when the assistant reply is canned. The
*rewriter* backend is still not surfaced in every result.

**What fixing it takes.** Require explicit model configuration in production
(fail to start rather than default to off), and add the active rewriter and
utility backends to every response alongside `simulated_reply`.

---

## 3. Training and serving prompts disagree, across all 4,034 training rows

The dataset hash matches the adapter's recorded training manifest, so this is
not a provenance error — the adapter was trained on exactly what the manifest
says. The two policies simply differ.

The stored training policy generalizes job titles to "my work". The serving
prompt in `src/rewriter/prompting.py:12` asks for broader role context to be
preserved.

**What it costs.** The model was taught one target and is asked for another at
inference. This is a plausible contributor to the finding in #109 that
fine-tuned and prompt-only rewriters are **not distinguishable at n=50** — if
the fine-tuning pulled against the serving prompt, we would expect exactly
that.

**What fixing it takes.** Pick one versioned policy, regenerate the targets
against it, review a sample by hand, and retrain. Not a four-day task, and
not one to start the week of a demo.

**Until then:** do not claim the fine-tuning improved on prompting. The
intervals do not support it and this is a reason to think they may not.

---

## 4. "Zero plaintext" is not a system-wide property

`src/conversation/state.py:210` stores `raw_text` on every `TurnRiskRecord`,
and the web path returns raw turn records to the caller.

Alias storage *is* blind-indexed, and that part is real. But blind-indexed
aliases plus in-memory raw transcripts do not add up to a zero-plaintext
system, and the phrase invites a reader to assume they do.

**What it costs.** Overclaiming on exactly the axis the product is about. The
lean canvas names user over-trust as the highest-cost failure: someone who
believes the tool more than it deserves shares *more* than they otherwise
would.

**What fixing it takes.** Decide retention explicitly and separately for the
two trust boundaries — a local browser session, where keeping the transcript
is the user's own choice, and a hosted gateway, where it is not. Then minimise
what is stored on the hosted path and say precisely which property holds
where.

**Until then:** say "blind-indexed aliases", not "zero plaintext".

---

## How to use this before demo day

If a question lands on any of these, the honest answer is the one above: the
finding is real, we found it, it is written down, and here is what it would
take. That is a better position than discovering it live — and under the
rubric, naming a limitation scores where an unevidenced claim does not.
