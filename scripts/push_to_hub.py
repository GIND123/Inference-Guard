"""Automated Hugging Face Hub Deployment Script for InferenceGuard Models.

Uploads fine-tuned QLoRA rewriter adapters or merged base models to the Hugging Face Hub,
complete with structured model cards, inference tags, and usage instructions.

Usage:
    # 1. Login to Hugging Face (or set HF_TOKEN environment variable)
    huggingface-cli login

    # 2. Push LoRA adapter (lightweight, ~50MB)
    python scripts/push_to_hub.py \
        --adapter_dir artifacts/rewriter_qlora \
        --repo_id your-username/inferenceguard-qwen2.5-1.5b-rewriter

    # 3. Merge LoRA weights into base model and push full standalone model (~3GB)
    python scripts/push_to_hub.py \
        --adapter_dir artifacts/rewriter_qlora \
        --repo_id your-username/inferenceguard-qwen2.5-1.5b-standalone \
        --merge
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_model_card(repo_id: str, is_merged: bool = False, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    """Generate Markdown Model Card for Hugging Face repository."""
    model_type = "Merged Full Weights" if is_merged else "PEFT LoRA Adapter"
    return f"""---
language:
- en
license: apache-2.0
tags:
- privacy
- pii-redaction
- inferential-privacy
- qwen
- peft
- lora
base_model: {base_model}
pipeline_tag: text-generation
---

# InferenceGuard: Privacy-Preserving LLM Query Rewriter

**InferenceGuard** neutralizes **inferred demographic context** (Age, Location, Occupation, Education) from user prompts sent to external LLMs, ensuring that the downstream model can solve the user's problem without exposing the user's private demographic profile.

- **Base Model**: `{base_model}`
- **Artifact Type**: {model_type}
- **Target Attributes**: Location, Occupation, Education, Age

## Key Capabilities

1. **Syntactic Preamble Pruning**: Strips autobiographical framing clauses that are unneeded for the task (e.g. *"I'm a 3rd-year pediatric resident at Boston Children's Hospital working 80-hour weeks. Can you help me diagnose a 4-year-old presenting with..."* $\\to$ *"Can you help me diagnose a 4-year-old presenting with..."*).
2. **Natural Entangled Generalization**: Smoothly generalizes parameters (e.g. retirement timeline, career longevity) without artificial bracketed tags like `[LOCATION]` or awkward fillers like `"the area"`.
3. **Problem & Domain Invariance**: 100% preservation of third-party patient ages, clinical symptoms, programming code, formulas, and legal statutes.
4. **Zero-PII Pass-Through**: Leaves neutral technical queries untouched.

## Usage Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "{base_model}"
adapter_name = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
{"# Model has merged weights" if is_merged else "model = PeftModel.from_pretrained(base_model, adapter_name)"}

prompt = "I'm a 3rd-year pediatric resident at Boston Children's Hospital working 80-hour weeks. Can you help me differential-diagnose a 4-year-old presenting with high fever for 6 days, strawberry tongue, and conjunctival injection?"

messages = [
    {{"role": "system", "content": "You are InferenceGuard Privacy Rewriter. Neutralize the author's demographic inferential cues (location, occupation, education, age) while preserving 100% of problem-solving utility."}},
    {{"role": "user", "content": prompt}}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.15)
rewritten = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Sanitized Prompt:", rewritten)
```
"""


def push_model_to_hub(
    adapter_dir: str,
    repo_id: str,
    merge: bool = False,
    hf_token: Optional[str] = None,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> None:
    """Execute upload of model artifact to Hugging Face Hub."""
    token = hf_token or os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    print(f"Ensuring repository exists: https://huggingface.co/{repo_id}")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory '{adapter_dir}' does not exist.")

    if merge:
        print(f"Loading base model '{base_model_name}' and merging LoRA weights from '{adapter_dir}'...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        merged_model = peft_model.merge_and_unload()

        temp_merged_dir = Path("artifacts/merged_model_for_upload")
        temp_merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged weights locally to '{temp_merged_dir}'...")
        merged_model.save_pretrained(temp_merged_dir)
        tokenizer.save_pretrained(temp_merged_dir)

        # Write Model Card
        card_content = generate_model_card(repo_id=repo_id, is_merged=True, base_model=base_model_name)
        (temp_merged_dir / "README.md").write_text(card_content, encoding="utf-8")

        print(f"Uploading merged model to '{repo_id}' on Hugging Face...")
        api.upload_folder(
            folder_path=str(temp_merged_dir),
            repo_id=repo_id,
            token=token,
        )
        print("Upload complete!")
    else:
        print(f"Uploading PEFT adapter files from '{adapter_path}' to '{repo_id}' on Hugging Face...")
        # Write Model Card
        card_content = generate_model_card(repo_id=repo_id, is_merged=False, base_model=base_model_name)
        (adapter_path / "README.md").write_text(card_content, encoding="utf-8")

        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            token=token,
        )
        print(f"Upload complete! View your model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push InferenceGuard models to Hugging Face Hub.")
    parser.add_argument("--adapter_dir", type=str, default="artifacts/rewriter_qlora", help="Path to LoRA adapter")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g. username/inferenceguard-qwen-rewriter)")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter into base model before uploading")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API write token")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    push_model_to_hub(
        adapter_dir=args.adapter_dir,
        repo_id=args.repo_id,
        merge=args.merge,
        hf_token=args.hf_token,
        base_model_name=args.base_model,
    )


if __name__ == "__main__":
    main()
