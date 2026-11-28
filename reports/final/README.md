# InferenceGuard — final submission

**A local-first privacy layer that protects what your text *implies*, not just
what it says.** It inspects a message before you send it to a language model,
scores how inferable your age, location, occupation and education are, shows
which words are responsible, and offers a rewrite that reveals less while
keeping your question answerable.

Against a held-out attacker it never trained on, it roughly halves attack
success where Presidio redaction does not move it at all.

| | |
|---|---|
| **Full report** | [`report.md`](report.md) |
| **Slides** | `slides.pdf` — *to be added before submission* |
| **Demo** | live; recorded backup `demo.mp4` — *to be added* |
| **Live URL** | *to be added once the container is deployed* |
| **Known limitations** | [`../../docs/engineering/known-limitations.md`](../../docs/engineering/known-limitations.md) |

Team: Anna Thomas (Data & Evaluation) · Chatur Bandaru (Engineering) ·
Govind Arun (Product) · Yiwei Jin (Users & Research) · Bingqi Lian (Operations)

---

## Running it

From a clean clone. Python 3.11 is what CI uses.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

**Fetch the dataset.** SynthPAI is CC-BY-NC-SA-4.0 and is not committed, so a
fresh clone has to pull it at the pinned revision:

```bash
python scripts/prepare_data.py
```

**Start the web app:**

```bash
uvicorn web.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Paste a message, press **Check my message**.

**Or run the container:**

```bash
docker compose up --build
```

One worker, deliberately — conversation state is process-local, so a second
worker would split a single conversation across two states.

### Try these

```
My co-op ends in December, and I take the Green Line to campus most mornings.
```

No name, no address, no email — and the detector should still flag education
and location. That is the whole point of the product in one line.

```
I am 26 years old living in Denver, CO and working as a software engineer
after graduating from Carnegie Mellon.
```

Explicit PII and inferential cues together, so you can see the two halves of
the report reported separately.

---

## Reproducing the numbers

Everything quoted in `report.md` comes from a committed artifact, and most of
it can be re-derived by running a committed script.

```bash
# Risk classifier on the held-out validation profiles.
# Does not use the product's loading code, by design.
python scripts/verify_risk_release.py \
    --model-id AT707/inferenceguard-modernbert-risk \
    --revision 4cecb3c90e84112ce5a5e3ed173060a1ab9609a7 \
    --claimed artifacts/risk_eval/inferenceguard-modernbert-risk_val_2026-11-18.json

# Held-out attacker: Wilson intervals and pairwise overlap.
python scripts/heldout_intervals.py reports/promptpai/comparative_benchmark_test.json

# Dataset statistics, including the 90.5% indirect-cue figure.
python scripts/dataset_report.py

# Per-stage latency with hardware provenance.
python scripts/benchmark_latency.py
```

Committed results live in `artifacts/risk_eval/`, `artifacts/heldout/` and
`artifacts/latency/`.

---

## Tests

```bash
pytest tests -q
```

**264 passed, 37 skipped** on a machine with no model stack — skips are the
tests needing torch, fastapi, numpy or pydantic, and they run in CI, which
installs `requirements-ci.txt` on every pull request. **Nothing should error.**
A non-zero exit means something is actually wrong.

A number of the tests are guards written after a specific defect, each
verified by reintroducing the regression and confirming the guard fails:

| guard | the defect it pins |
|---|---|
| `test_proxy_protection_contract.py` | the proxy protected only the last turn; upstream failures returned canned replies asserting completed actions |
| `test_input_length_guardrail.py` | the classifier silently truncated at 512 tokens and scored a prefix |
| `test_model_consistency.py` | train and serve drifted onto different base models; `.env` was not gitignored |
| `test_dataset_pin.py` | the profile split was not reproducible because downloads were unpinned |
| `test_served_risk_model.py` | the product served a 1-layer model with random heads |

---

## Configuration

| variable | default | meaning |
|---|---|---|
| `IG_DEMO_MODE` | `false` | Allows canned assistant replies and the global debug trace. **Leave off** anywhere a caller might act on the reply — the canned replies assert completed actions such as a fee waiver. |
| `RISK_MODEL_ID` | hosted ModernBERT | The risk classifier to serve. |
| `LOAD_REWRITER_WEIGHTS` | `false` | Loads the neural rewriter. Off means the heuristic path. See known limitations. |
| `HF_TOKEN` | — | Only needed to publish models. `.env` is gitignored; never commit it. |

---

## What this does not do

It does not make you anonymous and never claims to. It reduces what a model
can infer. `report.md` and the known-limitations document state what is
measured, what is estimated, and what has not been tested — including that
fine-tuning is not yet shown to beat prompting at the sample size we have.
