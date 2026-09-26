# Methodology review  --  base research

**Owner:** Govind (Product) · **Date:** 2026-09-21 (Session 04) · **Status:** research findings, nothing built

I went through the model stack and the evaluation design against what SynthPAI
actually contains and what the models on Hugging Face actually are. Ten things
need deciding before we build on top of them. Two are wrong picks, four are
measurement problems that would make our numbers mean something other than
what we say they mean, and four are assets in the dataset we are not using.

Every number below is reproducible:

```bash
python scripts/dataset_report.py          # downloads the pinned revision
```

Model facts come from each model's own `config.json` and safetensors index,
checked 2026-09-21. The stack is written up in `configs/models.yaml`.

---

## The two wrong picks

### 1. Qwen3.5-0.8B and Qwen3.5-2B are vision-language models

The design doc names them as the privacy rewriter. They are not text models:

```
architectures: ["Qwen3_5ForConditionalGeneration"]
vision_config:  present
pipeline_tag:   image-text-to-text
vocab_size:     248320
```

This holds for the **entire** Qwen3.5 family  --  0.8B, 2B, 4B, 9B, 27B, the MoE
variants, all of them. There is no text-only Qwen3.5.

Why it matters beyond tidiness: the 0.8B has 873M parameters, of which roughly
254M are a 248k-token embedding table, plus a vision tower that will never see
an image in this product. For a model whose entire selling point is running
locally, we would be paying about a third of the parameter budget for capacity
we cannot use.

**Recommendation: `Qwen/Qwen3-1.7B`**  --  text-only `Qwen3ForCausalLM`, 2.03B
params, 1.12 GB in 4-bit, QLoRA-trainable on a T4. With `Qwen/Qwen3-0.6B`
(0.41 GB in 4-bit) as the genuine on-device candidate. Measure latency for
both before committing to the local-first claim.

One gotcha: Qwen3 has hybrid thinking, and the chat template will emit a
`<think>` block before the rewrite unless you pass `enable_thinking=False`.
That is a latency and parsing problem waiting to happen.

### 2. Our location taxonomy does not fit the dataset

`artifacts/taxonomy.json` freezes location as
`[USA_Northeast, USA_South, USA_Midwest, USA_West, Europe, Other, unknown]`.

SynthPAI's 300 profiles:

```
countries: 74      USA: 21/300 (7%)
top:  United States 18, Australia 15, Japan 12, China 12, Germany 11
```

Four of our seven classes cover 7% of the data between them. The task
collapses to "Europe vs Other", and a location-protection number measured on
that is close to meaningless  --  reducing an attacker from "Other" to "unknown"
is not a privacy win anyone would pay for.

**Recommendation: do not classify location.** Have the attacker guess a
free-text city and country and score the guess for correctness, which is what
the SynthPAI paper does and the only way our number is comparable to theirs.
Report city-level and country-level hits separately: a correct country is
weak, a correct city is the thing users actually fear.

The same logic applies to education. Ours is
`[high_school, undergraduate, graduate_professional, other_unknown]`; the
paper uses `high school / college / master / PhD` and has published results
for 18 models on that scale. Our own buckets throw a free baseline away, and
the paper's scale is better balanced on this data anyway  --  college 38.3%,
master 37.7%, PhD 15.7%, high school 8.3%.

---

## The four measurement problems

### 3. Our "risk score" measures topic, not risk

`modernbert_predict` runs zero-shot classification with labels like
`"location"`. Zero-shot NLI scores how well the text matches the hypothesis
*this text is about location*. That is topical relevance. Inference risk is a
different quantity: how well an adversary can recover the **true value** of
the attribute.

They come apart in both directions. "I love reading about different cities"
scores high on location and reveals nothing. A quiet mention of a regional
supermarket chain scores low and narrows the country. The score also has no
value attached, so there is nothing to be right or wrong about  --  which is why
we cannot currently compute an attack accuracy at all, only a delta in a
number whose units are undefined.

This is the finding I would act on first, because our north-star metric is
built on it.

### 4. The evaluation loop is closed

`rewrite_until_safe(threshold=0.4, max_attempts=3)` generates candidates and
keeps the one that scores best under the risk estimator. We then report the
improvement in that same estimator's score as our result.

That is not a privacy measurement. It is adversarial-example search against
our own scorer, and the reported delta is a measure of how well the rewriter
games it. Goodhart in its most literal form. The design doc already calls for
a held-out adversary, so the fix is scheduled  --  the point is that until it
exists, **every number we quote is of this kind**, and the report should keep
saying so.

### 5. `confidence = max(risk_scores)` is mostly measuring name detection

`RISK_DIMENSIONS` includes `name`, and `confidence` is the max across all five
dimensions. Names are by far the easiest thing for NER to find, so the
headline number is dominated by explicit PII  --  exactly the thing Presidio
already solves and exactly the baseline we are supposed to be beating. We are
crediting our inferential system for work the baseline does.

**Recommendation:** report explicit PII and inferential risk as two separate
numbers, and exclude `name` from the inferential aggregate.

### 6. Our utility metric cannot detect the failures that matter

We use MiniLM cosine similarity between original and rewrite. Sentence
embeddings are dominated by topic, so dropping a single decisive constraint
barely moves them. The Session 04 log demonstrates this better than any
argument: the resume rewrite produced

```
Fix this resume: John Doe | 123 an area, an area an area | jdoe@example.com | ...
```

and scored **util_sim 0.907**. A metric that rates that as 91% preserved
cannot be used to defend a privacy - utility tradeoff.

**Recommendation:** keep cosine as a cheap screen, add bidirectional NLI
(`cross-encoder/nli-deberta-v3-large`) so that omitting a constraint shows up
as a failed entailment in at least one direction.

---

## The four assets we already have and are not using

### 7. SynthPAI ships per-comment inferability labels

Every one of the 7,823 comments carries `reviews.human`, a human annotation
per attribute with an estimate, a hardness rating and a certainty rating. A
non-empty estimate means *a human could infer this attribute from this
comment*.

That is precisely the label the risk estimator needs, and the design doc
already anticipates it ("where the dataset supports explicit inferability
labels, use those directly"). We built a zero-shot proxy instead.

```
occupation  1726 / 7823   22.1%
education    781 / 7823   10.0%
income       521 / 7823    6.7%
age          495 / 7823    6.3%
location     340 / 7823    4.3%
```

This also heads off a labelling mistake we have not made yet but were on
course for. Profile attributes attach to the **author**, not the comment. If
we label all 7,823 comments with their author's attributes and train
text → value, most comments carry a label for something they contain no
evidence of, and the only route to low loss is author style  --  which the
profile-disjoint split then correctly punishes at test time. It would look
like a hard task rather than a mislabelled one.

Two consequences to plan for: the classes are heavily imbalanced (location at
4.3% means a model that always says "not inferable" is 95.7% accurate, so
class weighting is not optional), and **the test split holds only 48 location
positives**. Any location claim needs a confidence interval attached or it is
noise.

### 8. The hardness field is the strongest evidence for our core premise

Each published guess is tagged `direct`, `indirect`, or `complicated`:

```
indirect        2813    71.1%
complicated      769    19.4%
direct           376     9.5%
```

**90.5% of successful attribute inferences in SynthPAI come from something
other than an explicit mention.** That is our thesis  --  "PII redaction does not
solve this"  --  measured on 3,958 annotated cases instead of argued from one
Green Line example. It is a far better opening slide than anything we have.

It also gives us Ablation A for free: report protection separately on the
`direct` subset (what Presidio can reach) and the `indirect` + `complicated`
subset (what only we can reach). If our advantage over Presidio does not show
up in that split, it does not exist.

### 9. There are published attack baselines in the file

`guesses[].model_eval` grades three ranked candidates per attribute against
ground truth, giving a reference attacker without us running anything:

| attribute | n | top-1 | top-1 w/ partial | top-3 |
|---|---:|---:|---:|---:|
| age | 410 | 49.8% | 49.8% | 100.0% |
| location | 306 | 58.5% | 62.7% | 95.1% |
| education | 1045 | 17.0% | 46.0% | 36.4% |
| occupation | 1649 | 41.8% | 61.6% | 69.0% |
| income | 548 | 63.3% | 63.3% | 100.0% |

### 10. `model_eval` is ternary, and it will bite us

It takes three values  --  0 wrong, **0.5 partially correct**, 1 correct  --  and
2,232 of 14,244 graded candidates are 0.5.

I hit this while writing the report script: `bool(model_eval[0])` treats 0.5
as a hit and reports occupation at 61.6% instead of 41.8%. A twenty-point
swing from a scoring convention, introduced by a `bool()` call that looks
completely reasonable.

The danger for us is not the absolute number, it is comparing a strict
"before" against a lenient "after" and reporting the difference as protection.
**Every attack number we publish has to state its convention, and before/after
pairs have to use the same one.** I would default to strict  --  a partially
correct location is a real privacy outcome, but it is not the attack
succeeding.

Related: top-3 is not a useful metric for low-cardinality attributes. Income
has four possible values, so three guesses cover 75% of the space, and top-3
reads 100% for both income and age. It measures the size of the label set, not
the attacker.

---

## One product problem, not a methodology one

Our demo persona is a Boston undergraduate with a Green Line commute and a
co-op. SynthPAI's median age is **38**; 18 - 24 is the **smallest** age bucket at
10.0%; 7% of profiles are in the USA at all.

So we are training on a global, mostly middle-aged forum population and
demoing on a young American student. Neither is wrong on its own, but we
should stop pretending they are the same thing. Either the demo moves toward
the data, or we say plainly in the report that the demo is illustrative and
the evaluation population is different. The second is honest and costs us
nothing; discovering the gap during Q&A on demo day would cost a lot.

---

## What I would do, in order

1. **Swap the rewriter** to `Qwen/Qwen3-1.7B` (or 0.6B for the on-device
   claim). Cheap, and it unblocks Session 07's QLoRA work on a T4.
2. **Retrain the risk estimator on `reviews.human` inferability labels**,
   multi-label with class weights, ModernBERT-base fine-tuned rather than
   zero-shot. This replaces the metric the whole project reports.
3. **Unfreeze the location and education taxonomies.** Location becomes
   generative-and-scored, education adopts the paper's four-way scale.
4. **Split every result by hardness** (`direct` vs `indirect` + `complicated`).
   This is the ablation that shows whether we beat Presidio.
5. **Add bidirectional NLI** to the utility side.
6. **Stand up the held-out adversary early** and run it against the rewrites we
   already have. It is one script, no training. If protection does not survive
   an attacker we did not optimise against, I want to know in Session 05, not
   Session 09.

Items 1 and 3 are decisions the team can take this week. Items 2, 4, 5 and 6
are Session 05 - 07 work and belong in issues, not in this document.

## Sources

- SynthPAI  --  https://huggingface.co/datasets/RobinSta/SynthPAI ·
  paper https://arxiv.org/abs/2406.07217 · CC-BY-NC-SA-4.0 ·
  revision `b572595f543a51db789caddbb81a9fc4edc6c32f`
- Model configs and parameter counts: Hugging Face model API and each
  repository's `config.json`, checked 2026-09-21. Recorded in
  `configs/models.yaml`.
