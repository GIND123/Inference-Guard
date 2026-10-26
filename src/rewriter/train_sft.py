"""
Fine-tune the Qwen LLM on the generated SFT dataset using QLoRA.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from src.rewriter.inference import DEFAULT_BASE_MODEL

# Guard against PEFT dispatch bug when incompatible torchao (< 0.16.0) is pre-installed in Colab
try:
    import peft.import_utils
    _orig_is_torchao = getattr(peft.import_utils, "is_torchao_available", None)
    if _orig_is_torchao is not None:
        def _safe_is_torchao_available() -> bool:
            try:
                return bool(_orig_is_torchao())
            except ImportError:
                return False

        peft.import_utils.is_torchao_available = _safe_is_torchao_available
        try:
            import peft.tuners.lora.torchao as _tao
            _tao.is_torchao_available = _safe_is_torchao_available
        except Exception:
            pass
except Exception:
    pass

logger = logging.getLogger(__name__)

@dataclass
class RewriterQLoRAConfig:
    """Configuration parameters for QLoRA fine-tuning with completion-only loss masking."""
    base_model_name: str = DEFAULT_BASE_MODEL
    train_jsonl_path: str = "data/sft/train_pareto_sft.jsonl"
    val_jsonl_path: Optional[str] = "data/sft/val_pareto_sft.jsonl"
    output_adapter_dir: str = "models/qwen35_08b_qlora_sft"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.10
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 4
    warmup_ratio: float = 0.05
    neftune_noise_alpha: float = 5.0
    seed: int = 42


def get_warmup_kwargs(config_cls: Any, ratio: float = 0.05) -> Dict[str, Any]:
    """Provide cross-version compatible warmup argument (ratio in v4 vs float steps in v5)."""
    import inspect
    try:
        params = inspect.signature(config_cls.__init__).parameters
        if "warmup_ratio" in params:
            return {"warmup_ratio": ratio}
    except Exception:
        pass
    return {"warmup_steps": ratio}


def filter_supported_kwargs(config_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by config_cls.__init__ across TRL/Transformers versions."""
    import inspect
    try:
        params = inspect.signature(config_cls.__init__).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return kwargs


def train_qlora_rewriter(cfg: RewriterQLoRAConfig) -> None:
    """Execute QLoRA fine-tuning with completion-only LM loss masking."""
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            device_map="cpu",
            trust_remote_code=True,
        )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    data_files = {"train": cfg.train_jsonl_path}
    if cfg.val_jsonl_path and Path(cfg.val_jsonl_path).exists():
        data_files["validation"] = cfg.val_jsonl_path
    dataset = load_dataset("json", data_files=data_files)

    def format_row(example):
        if "prompt" in example and "completion" in example:
            return {"text": example["prompt"] + example["completion"]}
        return example

    dataset = dataset.map(format_row)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    sft_kwargs = {
        "output_dir": cfg.output_adapter_dir,
        "dataset_text_field": "text",
        "max_length": cfg.max_seq_length,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "logging_steps": 10,
        "bf16": is_cuda and torch.cuda.is_bf16_supported(),
        "fp16": is_cuda and not torch.cuda.is_bf16_supported(),
        "use_cpu": not is_cuda,
        "neftune_noise_alpha": cfg.neftune_noise_alpha,
        "seed": cfg.seed,
        "report_to": "none",
        **get_warmup_kwargs(SFTConfig, cfg.warmup_ratio),
    }
    sft_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_kwargs))

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=collator,
        args=sft_args,
    )

    trainer.train()
    Path(cfg.output_adapter_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(cfg.output_adapter_dir)
    tokenizer.save_pretrained(cfg.output_adapter_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for InferenceGuard Rewriter")
    parser.add_argument("--model_name", "--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model name. Must match what inference loads: see DEFAULT_BASE_MODEL.")
    parser.add_argument(
        "--dataset_path",
        "--train_path",
        "--train_data",
        type=str,
        default="artifacts/rewriter_sft_chatml.jsonl",
        help="Path to the JSONL dataset (ChatML or Alpaca format)",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/rewriter_qlora", help="Output directory for adapters")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank dimension (capacity)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.10, help="LoRA dropout rate (regularization)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay parameter")
    return parser.parse_args()


def get_formatting_func(tokenizer):
    def format_prompt(example):
        if "messages" in example:
            prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            return {"text": prompt}
            
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        prompt = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n{output}{tokenizer.eos_token}"
        return {"text": prompt}
    return format_prompt

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    if "text" not in dataset.column_names:
        dataset = dataset.map(get_formatting_func(tokenizer))

    device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    is_cuda = torch.cuda.is_available()
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = is_cuda and not torch.cuda.is_bf16_supported()
    
    if is_cuda:
        logger.info("Configuring 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16
        )
        logger.info("Loading base model with quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("CUDA not available, loading model without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Handle cross-version compatibility for TRL and Transformers
    try:
        from trl import SFTConfig
        sft_kwargs = {
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "optim": "paged_adamw_32bit",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "use_cpu": not is_cuda,
            "gradient_checkpointing": True,
            "max_length": args.max_length,
            "dataset_text_field": "text",
            **get_warmup_kwargs(SFTConfig, 0.05),
        }
        training_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_kwargs))
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=training_args,
        )
    except (ImportError, TypeError):
        t_kwargs = {
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "optim": "paged_adamw_32bit",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "use_cpu": not is_cuda,
            "gradient_checkpointing": True,
            **get_warmup_kwargs(TrainingArguments, 0.05),
        }
        training_args = TrainingArguments(**filter_supported_kwargs(TrainingArguments, t_kwargs))
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text",
        )

    logger.info("Starting QLoRA fine-tuning")
    trainer.train()

    logger.info(f"Saving final adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
