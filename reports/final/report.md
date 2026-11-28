# InferenceGuard

**DATA/MSML 641, Fall 2026.** Anna Thomas (Data & Evaluation) · Chatur Bandaru
(Engineering) · Govind Arun (Product) · Yiwei Jin (Users & Research) ·
Bingqi Lian (Operations)

Repository: `github.com/GIND123/Inference-Guard`

---

## Problem and Users

People paste personal context into language models. Not search queries —
situations: a workplace grievance, a health question, a housing decision. They
protect themselves the way they were taught to, by deleting the name and the
email address, and they assume that is enough.

It is not. A model does not need an identifier. It reads implications.

> *"My co-op ends in December, and I take the Green Line to campus most mornings."*

No name, no address, no email. A model infers undergraduate, Boston, student,
18–24.

This is not a marginal case. In SynthPAI — 7,823 synthetic forum comments
across 300 profiles, with human annotations of what each comment reveals —
**90.5% of successful attribute inferences come from indirect or complicated
cues rather than an explicit mention** (n = 3,958 annotated guesses). Only
9.5% are direct.

**The user** is anyone pasting personal context into an LLM who is not a
privacy expert. **What they do today** is guess which details matter, and the
guess is calibrated for a world of forms and databases where the identifier
*was* the risk.

**What exists today.** Enterprise DLP tools, Presidio among them, are built
for compliance: they find identifiers in documents at rest. Provider privacy
settings govern retention, not content — the text still leaves the device
intact. Local models buy privacy by giving up the model you wanted. Nothing
sits in the moment between typing and sending, and nothing scores what a
message *implies*.

---

## Product

InferenceGuard inspects a message before it is sent, and offers a rewrite that
reveals less while keeping the question answerable.

```
message ──▶ Presidio ──▶ ModernBERT ──▶ Qwen rewriter ──▶ re-score ──▶ before / after
            explicit     inference        privacy            measure
            PII          risk             rewrite            the change
```

Two of those we trained: the **ModernBERT risk classifier**, which scores how
inferable age, location, occupation and education are from the text, and the
**Qwen rewriter**, fine-tuned with QLoRA on distilled teacher data. Presidio is
the baseline we have to beat, built into the product so we are measured
against it on every request.

The user sees a risk band per attribute, the cues driving it, a suggested
rewrite with the changes highlighted, and a before/after comparison. A live
word counter shows input length against the limit the classifier can actually
read, because past that the model truncates and a score would describe only
the start of the message.

There are two surfaces: a local web app, and an OpenAI-compatible gateway that
sits in front of an upstream model for business use.

**Honesty rules built into the code, not just the copy.** Explicit PII and
inferential risk are reported as two numbers and never averaged — merging them
once let a name detector masquerade as inferential protection. Remaining
identifiers veto a "protected" verdict regardless of what the attribute scores
did. Every response states which engine produced it, and a simulated reply is
flagged as simulated. The interface says *inference risk reduced*; it never
says *anonymous*.

---

## NLP Method and Evaluation

**Data.** SynthPAI (ETH Zurich, CC-BY-NC-SA-4.0), fully synthetic, pinned to
revision `b572595f`. Split by **profile, never by comment** — 210 / 45 / 45 —
so no author appears in both training and test; a comment-level split would
let the model learn writing style and score far better than it deserves.

**Risk classifier.** ModernBERT-base, four binary heads, trained on SynthPAI's
per-comment human inferability annotations rather than on author attributes.
That distinction matters: profile attributes attach to the *author*, so
labelling every comment with them would train the model on messages containing
no evidence at all.

Validation, 45 held-out profiles:

| | macro F1 | macro ROC-AUC | ECE |
|---|---:|---:|---:|
| ModernBERT, released checkpoint | 0.492 | **0.842** | 0.060 |

Independently re-measured by a script that does not use the product's loading
code; the training claim was 0.840.

**Rewriter.** Qwen, QLoRA, trained on teacher-distilled pairs with an NLI
acceptance gate.

**Held-out adversary.** Phi-4-mini-instruct — a different vendor, different
pretraining, different tokenizer, and never used to train or select anything.
Attack success rate over 50 PromptPAI test prompts, 200 attacks per method:

| method | ASR | 95% interval |
|---|---:|---|
| Raw text | 0.365 | 0.301 – 0.434 |
| Presidio | 0.360 | 0.297 – 0.429 |
| Prompt-only rewrite | 0.185 | 0.137 – 0.245 |
| Fine-tuned rewrite | 0.150 | 0.107 – 0.206 |

**Two readings, and the second is the more interesting one.**

Our rewriters roughly halve attack success. Presidio does not move it.

And per attribute, Presidio is *worse* than raw on three of four:

| attribute | raw | Presidio |
|---|---:|---:|
| age | 0.32 | **0.34** |
| education | 0.28 | **0.30** |
| occupation | 0.42 | **0.48** |
| location | 0.44 | 0.32 |

Redaction strips the surface identifiers and leaves the inferential signal
standing — on occupation it appears to sharpen it. That is the thesis of this
project, measured against an attacker we never trained on.

**Failure analysis.** Three things we found by measuring our own work:

1. **A risk model that detected nothing shipped to production.** The hosted
   checkpoint had never been trained — 198 of 199 backbone tensors were
   byte-identical to stock weights, heads at default initialisation — and
   separately the loader served a 1-layer model. Measured ROC-AUC 0.453,
   *below chance*. Found by scoring the model the product actually served,
   which nobody had done.
2. **An adversary that was never running.** It decoded its own prompt as the
   model's answer and silently fell back to a regex. Every attack number
   before that fix was void.
3. **Fine-tuned is not distinguishable from prompt-only at n = 50.** The
   intervals overlap substantially. We cannot claim the fine-tuning is what
   produced the reduction. A likely contributor: the training and serving
   prompts encode different policies across all 4,034 rows.

---

## User Evidence

Twenty-two sessions with external participants across Sessions 5–12, using
synthetic scenarios so nobody was asked for real personal information. Raw
logs and records are committed per session under `docs/`.

What they changed:

- **Session 06, 4 of 5 would not use the output.** One case returned the
  user's text unmodified after three failed rewrites while the interface still
  presented it as protected. That became a rule enforced in code: an unchanged
  message is never shown as protected.
- **Participants could not read the metrics.** Terminology help was added
  inline, and in writing it we found the UI had been printing "bits" for a
  quantity that is not entropy.
- **The protocol itself was wrong.** Participants were shown the verdict and
  then asked what they thought, which anchored them — one rated the system
  privacy-protective while it had rewritten nothing. Sessions now ask what the
  participant expects *before* revealing the result, so agreement and
  disagreement are both measurable.

---

## Ethics and Limitations

**We do not claim anonymity.** The product says *inference risk reduced*. A
user who over-trusts it and therefore shares more is worse off than one with
no tool at all, and that is the failure we design against.

**Data.** SynthPAI is synthetic and non-commercial. No real person's
attributes were used. User studies ran on invented scenarios; participants are
pseudonymous in every committed record.

**Bias.** Attribute inference is stereotype-shaped by construction — a model
guessing occupation from phrasing is reproducing a correlation. We report per
attribute rather than averaged, and every output is phrased as what an
attacker could infer, never as a fact about the user.

**What does not work yet, stated plainly:**

- The held-out result is **50 templated prompts scored on coarse categories**,
  not the SynthPAI test split our north star defines, and per-sample outputs
  are not committed, so it cannot be re-derived from the repository.
- **Fine-tuning is not shown to beat prompting** at this sample size.
- **The rewriter is the entire latency cost** — 1.6 s median, 3.6 s at p95,
  4.2 GB resident on a high-end laptop. We do not yet have a configuration
  that is both fast enough and good enough.
- Four further architectural limitations — two divergent runtime paths, a
  deployment that can serve heuristics while the architecture describes Qwen,
  the training/serving prompt disagreement, and in-memory raw transcripts that
  make "zero plaintext" an overstatement — are documented in
  `docs/engineering/known-limitations.md` with what each would take to fix.

**The honest summary.** The problem is real and we measured it. Traditional
redaction does not solve it, and on three attributes of four it makes things
worse. Our system roughly halves what an independent attacker recovers. We
cannot yet say the fine-tuning is what did it, and three times this semester
we shipped something that looked like it was working and was not — each time
caught by measuring it rather than by trusting it.
