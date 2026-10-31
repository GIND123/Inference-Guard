# Training Data Sources for the InferenceGuard Rewriter

The v1 rewriter was trained on a single narrow corpus (SynthPAI Reddit-style
comments: median 12 words, 69% of cues at the start, location in only 10.9% of
rows). Generalization needs *variety along four axes*: text register, cue
position, surface form, and attribute mix. This document lists verified,
license-noted sources for each axis, ordered by effort-to-impact.

`scripts/fetch_external_corpora.py --dry-run` prints the same table with the
exact download commands; `--download` fetches a source into `data/external/`.

---

## 1. Synthetic generation at scale (highest leverage, no licensing risk)

### 1.1 Persona-driven synthesis (recommended first step)
- **Dataset:** [proj-persona/PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) — ~200K-1M diverse personas with domain labels (`persona` split: 200K rows).
- **License:** CC-BY-NC-SA-4.0 (same family as SynthPAI; non-commercial, share-alike).
- **Method:** [Scaling Synthetic Data Creation with 1,000,000,000 Personas (arXiv:2406.20094)](https://arxiv.org/abs/2406.20094) — use each persona as a seed for your local teacher (Qwen2.5-7B) to write first-person texts in a target register, then pipe through `src/rewriter/augment.py`-style position control and the `judge_augmented` gate.
- **Why it matters:** the composer in `src/rewriter/augment.py` already provides position/register/attribute control; PersonaHub supplies unlimited *natural* persona variety (500+ distinct occupations, worldwide locations, life stages) so the model cannot memorize a fixed value inventory. Sample personas you do NOT have in the banks, annotate spans with the matcher, and let the teacher produce the natural rewrite.

### 1.2 The built-in composer (already in this repo)
- **Files:** `src/rewriter/augment.py` + `data/rewriter/personas/*.json` + `data/rewriter/carriers.json`.
- **Current yield:** 2,000 unique rows from 70 carriers x ~560 bank surfaces x 5 positions x 4 registers per position; deterministic, zero GPU.
- **How to scale:** (a) add carriers (aim for 300+ across more domains — one file, no code); (b) grow the banks (cities: 210 today; occupations: 156; institutions: 105); (c) `--augment-mode teacher` to have the local teacher paraphrase targets for naturalness while `judge_augmented` still verifies removal of every injected surface.
- **Rule of thumb:** keep synthetic rows ≤ ~50% of the final mix; more than that and the student starts speaking in template cadences.

## 2. Real PII-annotated corpora (explicit PII axis)

### 2.1 ai4privacy PII masking
- **Datasets:** [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) (407K rows; EN/FR/DE/ES/IT), plus 65K and nano-1K variants.
- **License:** custom (`license.md` — review before use; the org publishes an explicit license file, it is not a standard OSS license).
- **Columns:** `source_text`, `masked_text`, `privacy_mask` (entity spans + types), `mbert_token_classes`.
- **Use:** the source texts span real registers (business, legal, psychology notes) that SynthPAI lacks. Take `source_text`, map entity types to your attributes, and either (a) build explicit-PII scrubbing examples, or (b) use them as *carriers* — inject your own persona facts and use the real text's domain jargon as the utility content.

### 2.2 Presidio research synthetic generator
- **Tool:** [microsoft/presidio-research](https://github.com/microsoft/presidio-research) — faker-based PII data generator (`python -m presidio_anonymizer...` / the `data_generator` package).
- **License:** MIT.
- **Use:** unlimited fake-but-realistic PII entities (names, IDs, addresses, credit cards) with span labels; good for the explicit-PII axis and for stress-testing `scrub_hard_pii` beyond the 5 regexes currently in `generate_training_data.py`.

## 3. Cue-rich social text (inferential axis)

### 3.1 PAN Author Profiling corpora
- **Source:** [PAN @ CLEF author profiling](https://pan.webis.de/clef18/pan18-web/author-profiling.html) — Twitter/blog corpora labeled with **age group and gender** per author (EN/ES/AR, 2016-2018 editions; some editions add cross-genre essays).
- **License:** free for research (per PAN data agreement).
- **Use:** these texts are *dense in inferential cues by construction* (that is what the labels mean). Pair each text with its label as the "annotator hint" for the teacher — exactly the SynthPAI flow — and the teacher must remove what supports the label. Age-labeled tweets teach age-cue detection far better than SynthPAI's sparse age rows (269/1749).

### 3.2 Blog Authorship Corpus (informal)
- **Dataset:** [ blogger_authorship_corpus on Kaggle](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus) — 19K bloggers, ~680K posts with **age, gender, industry** metadata.
- **License:** research use.
- **Use:** longer-form informal text (paragraphs, not one-liners) — fixes the length tail (v1 p75 = 28 words). Sample posts, use metadata as hints.

## 4. Instruction/query corpora (carrier variety)

The composer's carriers are hand-written. To scale register and domain coverage:

- **[HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)** (10K human-written prompts, CC-BY-4.0... check card) — real user-LLM prompts across categories (generation, coding, advice). Filter to question-style prompts, strip any PII, use as carriers.
- **[WizardLM alpaca-evol / evol-instruct datasets](https://huggingface.co/datasets/WizardLM/Evol-Instruct-110K)** — more volume, more template-y; dedupe before use.
- **[fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)** (CC0) — prompt-style seeds.

**Pipeline:** strip entities from each prompt (`persona_banks.neutralize_text` + Presidio), keep only prompts with a clear question/request, then compose with persona injection at controlled positions.

## 5. Evaluation-only sources (never train on these)

- **[SALT-NLP/PrivacyLens](https://github.com/salt-nlp/privacylens)** — contextual-integrity benchmark for LLM agents in action; useful as an external probe of *what the rewriter lets through* in agentic settings, and its data-construction methodology (persona + tool + norm) transfers to your benchmark.
- **ConfAIde** (UMich) — contextual-integrity Q&A; small but adversarial.
- **Your own `benchmark_test_50_annotated.jsonl`** — keep growing it: 10 new cases per release, each with `pii_spans` reviewed by hand. The annotation script prints any case it cannot annotate; treat that as a to-do.

---

## Mixing strategy (what to actually train on)

| Slice | Source | Share | Why |
|---|---|---|---|
| Real rewrites (teacher-judged) | SynthPAI train split via `teacher_sft.py` | ~40% | natural social register, real cues |
| Position-controlled synthetic | `augment.py` (deterministic or teacher mode) | ~35% | position/register/surface control |
| High-density age/gender cues | PAN / Blogger + teacher | ~15% | fixes thin attributes |
| Long formal queries | ai4privacy/no_robots carriers + injection | ~10% | fixes the length/register tail |
| Identity no-ops | from all sources | ~10% of total | teaches when NOT to rewrite |

Checks before every train run:
```bash
python scripts/check_prompt_parity.py data/sft/<your_mix>/train.jsonl   # must exit 0
python data/evaluation/annotate_benchmark_spans.py                      # keep spans fresh
python scripts/evaluate_benchmark_50.py --fallback_baseline             # floor to beat
```

## Data hygiene rules that stay

1. Profile-disjoint splits (already enforced by `teacher_sft.load_split`).
2. The held-out attacker (Phi-4 family) never generates or filters training data (already enforced).
3. Any new synthetic source records provenance in its manifest (command, seed, git commit — `augment.py` and `teacher_sft.py` already do).
4. License check before any corpus enters `data/external/`; note it in this file.
