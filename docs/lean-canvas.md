# InferenceGuard — Lean Canvas

**Owner:** Govind (Product) · **Last revised:** 2026-09-21 (Session 04) · **Status:** v1, first version

Revised weekly. Changes are summarised in the "Lean canvas changes" section of that
week's report rather than tracked here, so this file always reads as current.

---

## 1. User and problem

**User.** Someone who pastes personal context into a general-purpose LLM — a student
asking for advice on a co-op, someone drafting a message about a workplace conflict,
someone asking a health or immigration question. They are not privacy experts. They
are aware enough to strip their name and email, and they assume that is sufficient.

**Problem.** It is not sufficient. A capable model recovers age, location, occupation
and education from ordinary detail that contains no identifier at all. Our own Session
04 baseline shows the shape of it: Presidio finds explicit spans in *"My co-op ends in
December, and I take the Green Line to campus"* — it tags the date and the transit line
— but nothing in its output addresses that "co-op" plus "campus" implies an
undergraduate, and that a Green Line commute narrows location to one metro area. The
user removed what looked identifying and left the inference intact.

The leak also compounds. Four messages that are individually unremarkable can describe
one person, and nothing in the user's interface shows that accumulating.

**What causes the pain today.** There is no moment in the workflow where anyone
inspects the text between typing and sending. The user's only signal is their own
intuition about what "counts" as personal, and that intuition is calibrated for
explicit identifiers.

## 2. Alternatives

| Alternative | What it does | Why it leaves the problem open |
|---|---|---|
| Doing nothing / self-censoring | User guesses what to omit | Guessing is exactly the failing skill; people redact names and keep the commute |
| Presidio and similar PII redactors | Tag and replace explicit spans | Built for identifiers, not implications. Zero coverage of the cues that carry the inference |
| Provider privacy controls (opt-out of training, chat history off) | Limit retention and reuse | Governs what the provider does with the text; the text still leaves the device fully intact |
| Local LLMs | Nothing leaves the machine | Solves it by giving up the frontier model. Our user wants the good model *and* the privacy |
| Manual rewriting | Works, when done well | Slow, and the user is back to guessing which details matter |

The gap we sit in: nobody rewrites for *inference* rather than identifiers, and nobody
does it locally.

## 3. Value proposition

> **We help people who paste personal context into an LLM reduce what a model can infer
> about them — not just what they explicitly wrote — before the text leaves their device,
> and without losing the detail their question depends on.**

The honest version of the pitch is the tradeoff, not the protection. Deleting everything
is perfect privacy and a useless product. What we are actually claiming is a better
privacy–utility curve than redaction, measured rather than asserted.

## 4. Distribution

Not solved, and we should stop describing it as if it were.

Today the only surface is a Colab notebook, which reaches nobody outside the team. The
plausible progression:

1. **Now** — web UI in the repo, paste-in / paste-out. Enough to run user sessions against.
2. **Realistic for this semester** — browser extension that intercepts the text box on
   the major chat UIs and offers a rewrite before send. This is where the product
   actually belongs: the protection has to sit in the moment of sending, not in a second
   tab the user has to remember.
3. **Beyond scope** — OS-level clipboard or keyboard integration.

**Open question for Session 05:** whether anyone will accept a second step before hitting
send. That is the distribution risk and it is cheaper to test than to build — the user
sessions should measure abandonment, not just rewrite quality.

## 5. Cost per user request

Local-first, so there is no per-request API cost — the cost is the user's own compute
and the patience it consumes. Current stack per request: ModernBERT-base zero-shot,
Qwen2.5-0.5B-Instruct for the rewrite (up to 3 attempts under `rewrite_until_safe`), and
all-MiniLM-L6-v2 for the similarity check.

**Marginal cash cost: ~$0.** **Latency: not yet measured**, and that is the number that
matters. Three rewrite attempts through a 0.5B model on CPU is the obvious risk to the
"before you send" premise. Instrumenting it is a Session 05 task.

This is a real advantage over a hosted competitor and we should say so plainly: at zero
marginal cost, there is no usage cap to enforce and no reason to retain user text.

## 6. North-star metric

**Held-out attribute leakage rate** — the share of protected target attributes that an
independent attacker, never used during training or tuning, still infers correctly.
Lower is better.

**Guardrail:** utility must hold. A leakage drop that comes with a rewrite people reject
is not progress, so the metric is only reported alongside rewrite acceptance and
semantic similarity.

*Session 04 status, stated honestly:* we cannot compute this yet. The held-out adversary
(Phi-4-mini-instruct) is not implemented, so the report uses a proxy — before/after risk
delta from the same estimator that guided the rewrite, which is a weaker claim and
should be labelled as one every week until the real attacker exists.

*Proposed change for Session 05:* the proxy reported in Session 04 was a single
attribute's delta. On the resume row of `data/logs/session04_usage_log.csv` that reads
0.852 while overall risk moved 0.01 and the rewrite left a name, an email and a phone
number in place. Any headline number we quote should be the worst attribute, not the
best one. `src/product/risk_bands.py` enforces this in code.

## 7. Ethics and privacy

- **Data rights.** SynthPAI (CC-BY-NC-SA-4.0), fully synthetic, no real individuals.
  Non-commercial terms suit a course project and would need revisiting for anything else.
- **No real personal data in user testing.** Participants work from synthetic scenarios
  in `docs/product/user_test_scenarios.md`. If a participant volunteers their own text
  anyway, it is processed and discarded — never committed, never in a report.
- **Bias.** Attribute inference is stereotype-shaped by construction: a model guessing
  occupation from phrasing is reproducing a correlation from its training data. Two
  consequences we accept as obligations — per-attribute error analysis rather than one
  averaged score, and output phrased as *"an attacker could infer"* and never *"you are"*.
  The second is enforced in `describe()` and tested.
- **The claim we must not make.** "Inference risk reduced", never "you are anonymous".
  A user who over-trusts the tool and shares more than they otherwise would is worse off
  than a user with no tool at all. This is the failure mode with the highest cost and it
  is a *product* failure, not a model failure.
- **Retention.** Session state is local, raw messages are not logged by default,
  pseudonym maps are cleared on reset.

## 8. Risk

**Biggest threat to viability:** that the protection does not survive contact with an
attacker we did not optimise against. Everything measured so far uses the same estimator
that drives the rewrite — a closed loop that can only flatter us. If leakage under
Phi-4-mini-instruct is close to the unprotected baseline, the core claim fails.

**Cheapest way to find out:** run the held-out attacker against the *existing* Session 04
rewrites. No training, no new data, one evaluation script. If the number is bad we learn
it in Session 05 instead of Session 09, and a negative result found early is reportable
work rather than a late collapse.

**Second risk:** that nobody tolerates an extra step before sending. Cheapest test is
measuring abandonment in the scheduled user sessions rather than asking people whether
they would use it.

**Third risk:** Qwen2.5-0.5B is not reliably capable of the rewrite. It already leaks
entities the prompt told it to remove, which is why a regex scrub sits behind it. We are
partly measuring a regex. Mitigation is the 2B fallback, at a cost to the local-first
premise.
