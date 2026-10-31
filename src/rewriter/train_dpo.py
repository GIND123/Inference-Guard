"""Direct Preference Optimization (DPO) for the Qwen rewriter.

Aligns the student model to penalize over-scrubbing (truncation) and under-scrubbing (PII leaks).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rewriter.constants import DEFAULT_BASE_MODEL, SMALL_STUDENT_MODEL

# Guard against broken C++ binary extensions in Colab
for _pkg in ("torchvision", "torchaudio"):
    sys.modules[_pkg] = None
    for _mod in list(sys.modules.keys()):
        if _mod.startswith(f"{_pkg}."):
            del sys.modules[_mod]

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import DPOTrainer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

STUDENT_MODEL_ALIASES: Dict[str, str] = {
    "0.5b": SMALL_STUDENT_MODEL,
    "qwen-0.5b": SMALL_STUDENT_MODEL,
    "1.5b": DEFAULT_BASE_MODEL,
    "qwen-1.5b": DEFAULT_BASE_MODEL,
}


def resolve_model_name(name: str) -> str:
    return STUDENT_MODEL_ALIASES.get(name.lower(), name)


def filter_supported_kwargs(target_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by target_cls.__init__ across versions."""
    try:
        params = inspect.signature(target_cls.__init__).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return kwargs


def get_warmup_kwargs(config_cls: Any, ratio: float = 0.1) -> Dict[str, Any]:
    """Provide cross-version compatible warmup argument (ratio in v4 vs float steps in v5)."""
    try:
        params = inspect.signature(config_cls.__init__).parameters
        if "warmup_ratio" in params:
            return {"warmup_ratio": ratio}
    except Exception:
        pass
    return {"warmup_steps": ratio}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO alignment for InferenceGuard rewriter.")
    parser.add_argument("--dataset_path", type=str, default="data/dpo/train_dpo.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--ref_adapter_path", type=str, default="artifacts/rewriter_qlora")
    parser.add_argument("--output_dir", type=str, default="artifacts/rewriter_dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter beta.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=384)
    parser.add_argument("--dry_run", action="store_true", help="Validate setup without loading weights.")
    return parser.parse_args()


def train_dpo(args: argparse.Namespace) -> None:
    if args.dry_run or not _TORCH_AVAILABLE:
        print(f"Validation mode: DPO dataset {args.dataset_path} verified.")
        if not Path(args.dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"Found {count} preference pairs. DPO pipeline ready.")
        return

    model_id = resolve_model_name(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def format_dpo_row(example):
        if isinstance(example.get("prompt"), list):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
        return example

    dataset = dataset.map(format_dpo_row)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.ref_adapter_path and Path(args.ref_adapter_path).exists() and (Path(args.ref_adapter_path) / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            logger.info("Loading reference SFT adapter from %s", args.ref_adapter_path)
            model = PeftModel.from_pretrained(model, args.ref_adapter_path, is_trainable=True)
            peft_config = None
        except Exception as exc:
            logger.warning("Could not load adapter from %s: %s. Using fresh LoRA.", args.ref_adapter_path, exc)

    try:
        from trl import DPOConfig
        config_cls = DPOConfig
    except ImportError:
        config_cls = TrainingArguments

    dpo_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "bf16": is_cuda and torch.cuda.is_bf16_supported(),
        "fp16": is_cuda and not torch.cuda.is_bf16_supported(),
        "use_cpu": not is_cuda,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        **get_warmup_kwargs(config_cls, 0.1),
    }
    training_args = config_cls(**filter_supported_kwargs(config_cls, dpo_args_dict))

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = DPOTrainer(**filter_supported_kwargs(DPOTrainer, trainer_kwargs))

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO alignment complete. Adapter saved to {args.output_dir}")


def main() -> None:
    args = parse_args()
    train_dpo(args)


if __name__ == "__main__":
    main()
