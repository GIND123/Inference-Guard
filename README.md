<p align="center"> <img src="img/inferential_privacy.png" width="120" /> </p> <h1 align="center">InferenceGuard: Local-First Privacy Layer for Inferential Privacy</h1> <p align="center"> <strong>explicit PII removal to risk estimation to privacy-preserving rewrite</strong> </p> <p align="center">
<a href="https://colab.research.google.com/drive/1l_hrmfaKqt69D_8QrEa3KCCYpPrWiG8y?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/notebook-open%20in%20colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Colab"></a> 
<img src="https://img.shields.io/badge/hardware-A100%20%7C%20T4-76B900?style=flat&logo=nvidia&logoColor=white" alt="Hardware"> <img src="https://img.shields.io/badge/dataset-SynthPAI%207.8k-blue?style=flat" alt="Dataset"> <img src="https://img.shields.io/badge/models-ModernBERT%20%7C%20Qwen2.5-blueviolet?style=flat" alt="Models"> </p> <p align="center"> <a href="#what-it-is">What</a> • <a href="#setup">Setup</a> • <a href="#run-it">Run It</a> </p> 

---

## What It Is

Most privacy tools only redact explicit PII like names, emails, and phone numbers. They miss inferential privacy, which is what a strong LLM can infer from ordinary clues.

> "My co-op ends in December, and I take the Green Line to campus most mornings."

There is no name or address here, but an attacker can still infer Boston from Green Line, undergraduate status from campus language, student occupation from co-op, and age range 18 to 24.

**InferenceGuard** is a local-first layer that runs before text leaves user control. It estimates how inferable attributes like Age, Location, Occupation, and Education are, highlights the cues that cause risk, and produces a rewritten version that reduces leakage while keeping intent. For example, Green Line becomes public transportation and co-op becomes work placement.

The core idea is to protect what text implies, not just what it explicitly says, and to handle multi-turn leakage where individually harmless messages accumulate over a conversation.

## Setup

**Dataset:** SynthPAI `RobinSta/SynthPAI` - 300 synthetic profiles, 103 threads, 7,823 comments, CC-BY-NC-SA-4.0  
**Models:** ModernBERT-base for risk, Qwen2.5-1.5B-Instruct for rewriter, Phi-4-mini-instruct for held-out attacker, Presidio for explicit PII baseline  

**Core Dependencies:**
```bash
pip install -q datasets huggingface_hub presidio-analyzer presidio-anonymizer spacy scikit-learn pandas matplotlib seaborn transformers torch peft trl fastapi uvicorn pyngrok
python -m spacy download en_core_web_lg
```

## Run It

InferenceGuard is a **local-first** product: the whole pipeline runs on your own
machine so raw text never leaves your control. The deployment target is on-device
(see `docs/engineering/production-readiness.md` §1); the Colab + tunnel flow at the
end is a **demo**, not the product. Run it locally first.

### 0. Install locally
From the repo root, install the package (and its dependencies) in editable mode:
```bash
pip install -e .
```
The profile-disjoint splits are committed under `data/splits/` and hashed in
`data/manifests/split_manifest.json`, so nothing needs downloading first. If they
are ever missing, `python -m src.data.splits` regenerates them from the pinned
SynthPAI revision, and `python scripts/check_splits.py` checks them.

If you also need the raw SynthPAI comment text (it is CC-BY-NC-SA-4.0 and is
**not** committed to this repository), fetch it before running anything that
reads the raw corpus:

```bash
python scripts/prepare_data.py
```

That downloads `RobinSta/SynthPAI` at the revision pinned in
`configs/models.yaml` to `data/raw/synthpai.jsonl`, and re-running it is a
no-op once the file is there. The pin matters: the profile-disjoint split is a
function of dataset iteration order as well as the seed, so an unpinned
download can change the split without anything failing (issue #21).

### 1. Train the ModernBERT risk classifier
Point the trainer at the committed splits under `data/splits/` (the trainer prefers
these automatically when present):
```bash
python -m src.risk_model.train \
    --train_data data/splits/train.jsonl \
    --val_data data/splits/val.jsonl \
    --output_dir artifacts/risk_model \
    --epochs 3
```

### 2. Build the rewriter SFT set and fine-tune the adapter
The profile-disjoint splits are committed under `data/splits/` and hashed in
`data/manifests/split_manifest.json`, so the splits themselves need no
download, but the comment text they index does, via step 0 above. If the
splits are ever missing, `python -m src.data.splits` regenerates them from the
pinned SynthPAI revision, and `python scripts/check_splits.py` checks them.

**v2 (single-turn prompt + position-controlled augmentation).** The v1 SFT set
embedded one fixed two-shot primer in every training row, so 66.7% of the
supervised turns were the same two completions and the served prompt no longer
matched the trained one (1749/1749 rows) - the adapter memorized instead of
generalizing. The student is now trained single-turn (policy lives in the
system prompt, from `src/rewriter/prompting.py`), and a position-randomized
augmented corpus teaches cue detection anywhere in the text:

```bash
# 1. Compose the position-controlled corpus (deterministic, no GPU):
python -m src.rewriter.augment --n 2000 --out data/sft/augmented_v1/augmented.jsonl

# 2a. Bootstrap an SFT set from the corpus alone (no teacher, no GPU):
python -m src.rewriter.teacher_sft --augment-only --augment-dir data/sft/augmented_v1 --out-dir data/sft/augmented_v1

# 2b. Full run: teacher rewrites SynthPAI train comments AND mixes the
#     augmented corpus through the same acceptance gates:
python -m src.rewriter.teacher_sft --split train --device cuda \
    --out-dir data/sft/teacher_v2 --augment-dir data/sft/augmented_v1

# 3. Gate the dataset before training (train/serve parity, no duplicated
#    turns, position statistics; exits non-zero on regression):
python scripts/check_prompt_parity.py data/sft/augmented_v1/train.jsonl
```

The SFT targets from the teacher are kept only if they pass a verbatim leak
check against the author's true attributes, the degeneracy rule, and NLI
entailment; augmented rows additionally must remove *every injected surface
form* (variants included), so "relocated instead of removed" is rejected. The
held-out Phi-4 attacker is never used. See the docstrings of
`src/rewriter/teacher_sft.py` and `src/rewriter/augment.py`.
Then QLoRA fine-tune the Qwen2.5-1.5B-Instruct adapter on it with
`python -m src.rewriter.train_sft --dataset_path data/sft/teacher_v2/train.jsonl`.

More training data: `docs/DATA_SOURCES.md` lists verified corpora (PersonaHub,
ai4privacy PII masking, PAN author profiling, Blogger) and
`scripts/fetch_external_corpora.py --dry-run` prints the download plan.

### 2b. Evaluate the rewriter honestly
```bash
python data/evaluation/annotate_benchmark_spans.py       # span annotations, review the output
python scripts/evaluate_benchmark_50.py --adapter_path artifacts/rewriter_qlora
python scripts/evaluate_benchmark_50.py --fallback_baseline   # bank-driven floor to beat
```
The v2 evaluator scores PII-span removal, invariant (jurisdiction/standard)
preservation and copy-through per category - no hardcoded keyword checklist.

### 3. Launch the Web UI locally
Serve the FastAPI app with the trained risk model + adapter and open it in your
browser:
```bash
uvicorn web.api:app --host 127.0.0.1 --port 8000
```
Browse to `http://127.0.0.1:8000` to test the Privacy Risk Evaluation powered by
your trained ModernBERT risk classifier and Qwen2.5-1.5B-Instruct rewriter.

### Train on a GPU and host on the Hugging Face Hub
If you do not have a local GPU, use the notebook
**`InferenceGuard_modernbert_train_and_host.ipynb`** to train the multi-task
ModernBERT risk classifier on a Colab GPU and push it to the Hugging Face Hub.
Once the trained risk model and LoRA adapter are on the Hub, the local pipeline
and the Web UI load them from there, so step 3 above works without retraining.
(`InferenceGuard_finetune.ipynb` covers the broader fine-tuning flow.)

### Demo: Colab + Localtunnel
As a **demo only** (not the local-first product path), you can expose the Web UI
running inside Colab through a public tunnel:
```python
import subprocess
import urllib.request
import time

# Start FastAPI server in the background
get_ipython().system_raw('uvicorn web.api:app --host 0.0.0.0 --port 8000 &')
time.sleep(5)  # Wait for ModernBERT to load into GPU

# Get Colab IP for Localtunnel password
colab_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()

# Spawn Localtunnel
proc = subprocess.Popen(['npx', 'localtunnel', '--port', '8000'], stdout=subprocess.PIPE, text=True)
url = proc.stdout.readline().strip().replace('your url is: ', '')

print(f"🚀 INFERENCEGUARD UI IS LIVE AT: {url}")
print(f"🔒 Tunnel Password: {colab_ip}")
```

Click the Localtunnel URL, enter your Colab IP as the password, and you can test
the Privacy Risk Evaluation powered by your trained ModernBERT model. For real
user text, prefer the local-first path above; the tunnel demo sends text off the
machine and is not the product.

## Run Unit Tests
To verify all pipelines and integrations (including the Phi-4-mini adversarial evaluation), run the master test suite:
```bash
python tests/run_all_tests.py
```

## Production Deployment & Containerization

### Docker Build & Run
InferenceGuard includes a multi-stage production Docker container with non-root execution and health probes:

```bash
docker build -t inference-guard-api:latest .
docker run -d -p 8000:8000 --name inference_guard inference-guard-api:latest
```

### Docker Compose
For single-node staging or local multi-container development:
```bash
docker compose up -d
```

Verify service liveness and readiness:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

See [docs/PRODUCTION_DEPLOYMENT.md](file:///c:/Users/annat/Downloads/Inference-Guard-main/docs/PRODUCTION_DEPLOYMENT.md) for full deployment documentation.

## Hugging Face Hub Hosting (RoBERTa Risk Model)
Package and publish the RoBERTa multi-task risk classifier directly to the Hugging Face Hub:

```bash
python scripts/host_roberta_on_hf.py \
    --repo-id "your-org/inferenceguard-roberta-risk" \
    --base-model "roberta-base"
```

Or deploy using the standalone `hf_roberta_risk_model/` module with remote code execution.
See [docs/HUGGINGFACE_ROBERTA_HOSTING.md](file:///c:/Users/annat/Downloads/Inference-Guard-main/docs/HUGGINGFACE_ROBERTA_HOSTING.md) for detailed instructions.


