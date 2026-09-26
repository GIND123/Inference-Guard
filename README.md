<p align="center"> <img src="img/inferential_privacy.png" width="120" /> </p> <h1 align="center">InferenceGuard: Local-First Privacy Layer for Inferential Privacy</h1> <p align="center"> <strong>explicit PII removal to risk estimation to privacy-preserving rewrite</strong> </p> <p align="center">
<a href="https://colab.research.google.com/drive/1l_hrmfaKqt69D_8QrEa3KCCYpPrWiG8y?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/notebook-open%20in%20colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Colab"></a> 
<img src="https://img.shields.io/badge/hardware-A100%20%7C%20T4-76B900?style=flat&logo=nvidia&logoColor=white" alt="Hardware"> <img src="https://img.shields.io/badge/dataset-SynthPAI%207.8k-blue?style=flat" alt="Dataset"> <img src="https://img.shields.io/badge/models-ModernBERT%20%7C%20Qwen3.5-blueviolet?style=flat" alt="Models"> </p> <p align="center"> <a href="#what-it-is">What</a> • <a href="#setup">Setup</a> • <a href="#run-it">Run It</a> </p> 

---

## What It Is

Most privacy tools only redact explicit PII like names, emails, and phone numbers. They miss inferential privacy, which is what a strong LLM can infer from ordinary clues.

> "My co-op ends in December, and I take the Green Line to campus most mornings."

There is no name or address here, but an attacker can still infer Boston from Green Line, undergraduate status from campus language, student occupation from co-op, and age range 18 to 24.

**InferenceGuard** is a local-first layer that runs before text leaves user control. It estimates how inferable attributes like Age, Location, Occupation, and Education are, highlights the cues that cause risk, and produces a rewritten version that reduces leakage while keeping intent. For example, Green Line becomes public transportation and co-op becomes work placement.

The core idea is to protect what text implies, not just what it explicitly says, and to handle multi-turn leakage where individually harmless messages accumulate over a conversation.

## Setup

**Dataset:** SynthPAI `RobinSta/SynthPAI` - 300 synthetic profiles, 103 threads, 7,823 comments, CC-BY-NC-SA-4.0  
**Models:** ModernBERT-base for risk, Qwen3.5-1.7B for rewriter, Phi-4-mini-instruct for held-out attacker, Presidio for explicit PII baseline  

**Core Dependencies:**
```bash
pip install -q datasets huggingface_hub presidio-analyzer presidio-anonymizer spacy scikit-learn pandas matplotlib seaborn transformers torch peft trl fastapi uvicorn pyngrok
python -m spacy download en_core_web_lg
```

## Run It

InferenceGuard is now fully modularized into a robust Python architecture. You can run the entire pipeline—from dataset generation to fine-tuning and launching the live UI—directly from the **`InferenceGuard_finetune.ipynb`** Colab notebook.

### 1. Fine-Tune ModernBERT
To train the multi-task risk classifier on the SynthPAI dataset, use the module directly:
```bash
python -m src.risk_model.train \
    --data_path data/raw/synthpai.jsonl \
    --splits_path artifacts/profile_splits.json \
    --output_dir artifacts/risk_model \
    --epochs 3
```

### 2. Generate SFT Dataset
To generate the Supervised Fine-Tuning dataset for the Qwen rewriter using Pareto Rejection Sampling:
```python
from src.rewriter.generate_training_data import generate_sft_dataset

sft_metrics = generate_sft_dataset(
    synthpai_path="data/raw/synthpai.jsonl",
    splits_path="artifacts/profile_splits.json",
    output_chatml_path="artifacts/rewriter_sft_chatml.jsonl",
    output_alpaca_path="artifacts/rewriter_sft_alpaca.json",
    max_risk_threshold=0.30,
    min_cosine_threshold=0.50
)
```

### 3. Launch the Web UI
To spin up the live interactive web UI inside Colab and test your trained models:
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

Click the Localtunnel URL, enter your Colab IP as the password, and you can instantly test the Privacy Risk Evaluation powered by your fine-tuned ModernBERT model!

## Run Unit Tests
To verify all pipelines and integrations (including the Phi-4-mini adversarial evaluation), run the master test suite:
```bash
python tests/run_all_tests.py
```

