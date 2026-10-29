"""
Fine-tune the Qwen LLM on the generated SFT dataset using QLoRA.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
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

STUDENT_MODEL_ALIASES: Dict[str, str] = {
    "qwen-0.6b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen0.6b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "0.6b": "Qwen/Qwen2.5-0.5B-Instruct",
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def resolve_model_name(model_name: str) -> str:
    """Resolve model name or alias (e.g. 'qwen-0.6b' -> 'Qwen/Qwen2.5-0.5B-Instruct')."""
    key = model_name.lower().strip()
    return STUDENT_MODEL_ALIASES.get(key, model_name)


def get_default_dataset_path() -> str:
    """Resolve default SFT training dataset path in priority order."""
    candidates = [
        "data/sft/teacher_v1/train.jsonl",
        "data/sft/train_queries_sft.jsonl",
        "artifacts/rewriter_sft_chatml.jsonl",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[0]


def detect_google_drive_path(drive_arg: Optional[str] = None) -> Optional[Path]:
    """Auto-detect or resolve target Google Drive directory for saving checkpoints."""
    if drive_arg:
        return Path(drive_arg)
    env_var = os.environ.get("GOOGLE_DRIVE_PATH") or os.environ.get("GDRIVE_DIR")
    if env_var:
        return Path(env_var)
    colab_mount = Path("/content/drive/MyDrive")
    if colab_mount.exists():
        return colab_mount / "InferenceGuard" / "rewriter_checkpoints"
    return None


try:
    from transformers import TrainerCallback

    class GoogleDriveSyncCallback(TrainerCallback):
        """Syncs intermediate checkpoints directly to mounted Google Drive to protect against Colab disconnects."""

        def __init__(self, drive_path: Path, local_dir: Path):
            self.drive_path = drive_path
            self.local_dir = local_dir
            self.drive_path.mkdir(parents=True, exist_ok=True)

        def on_save(self, args, state, control, **kwargs):
            try:
                ckpts = sorted(self.local_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    latest = ckpts[-1]
                    target = self.drive_path / latest.name
                    shutil.copytree(latest, target, dirs_exist_ok=True)
                    logger.info(f"[Google Drive Sync] Persisted intermediate checkpoint {latest.name} -> {target}")
            except Exception as e:
                logger.warning(f"[Google Drive Sync] Failed to sync checkpoint to Drive: {e}")
except Exception:
    GoogleDriveSyncCallback = None


@dataclass
class RewriterQLoRAConfig:
    """Configuration parameters for QLoRA fine-tuning with completion-only loss masking."""
    base_model_name: str = DEFAULT_BASE_MODEL
    train_jsonl_path: str = "data/sft/teacher_v1/train.jsonl"
    val_jsonl_path: Optional[str] = None
    output_adapter_dir: str = "artifacts/rewriter_qlora"
    drive_output_dir: Optional[str] = None
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
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    max_steps: Optional[int] = None


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

    resolved_base_model = resolve_model_name(cfg.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved_base_model, trust_remote_code=True)
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
            resolved_base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
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
        if "text" in example and example["text"]:
            return example
        if "prompt" in example and "completion" in example:
            return {"text": example["prompt"] + example["completion"]}
        if "messages" in example:
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        return example

    dataset = dataset.map(format_row)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    drive_path = detect_google_drive_path(cfg.drive_output_dir)
    callbacks = []
    if drive_path and GoogleDriveSyncCallback is not None:
        callbacks.append(GoogleDriveSyncCallback(drive_path=drive_path, local_dir=Path(cfg.output_adapter_dir)))

    sft_kwargs = {
        "output_dir": cfg.output_adapter_dir,
        "dataset_text_field": "text",
        "max_length": cfg.max_seq_length,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "weight_decay": cfg.weight_decay,
        "logging_steps": 10,
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "bf16": is_cuda and torch.cuda.is_bf16_supported(),
        "fp16": is_cuda and not torch.cuda.is_bf16_supported(),
        "use_cpu": not is_cuda,
        "neftune_noise_alpha": cfg.neftune_noise_alpha,
        "seed": cfg.seed,
        "report_to": "none",
        **get_warmup_kwargs(SFTConfig, cfg.warmup_ratio),
    }
    if cfg.max_steps is not None and cfg.max_steps > 0:
        sft_kwargs["max_steps"] = cfg.max_steps

    sft_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_kwargs))

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
        args=sft_args,
    )

    trainer.train()
    Path(cfg.output_adapter_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(cfg.output_adapter_dir)
    tokenizer.save_pretrained(cfg.output_adapter_dir)

    if drive_path:
        drive_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(drive_path))
        tokenizer.save_pretrained(str(drive_path))
        logger.info(f"Persisted final adapter and tokenizer to Google Drive: {drive_path}")


def parse_args():
    default_dataset = get_default_dataset_path()
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for InferenceGuard Rewriter")
    parser.add_argument("--model_name", "--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model name or alias (e.g. qwen-0.6b, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct).")
    parser.add_argument(
        "--dataset_path",
        "--train_path",
        "--train_data",
        type=str,
        default=default_dataset,
        help="Path to the JSONL dataset (ChatML or Alpaca format)",
    )
    parser.add_argument(
        "--val_path",
        "--val_data",
        type=str,
        default=None,
        help="Path to validation JSONL dataset",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/rewriter_qlora", help="Output directory for adapters")
    parser.add_argument("--drive_path", "--gdrive_dir", "--drive_output_dir", type=str, default=None,
                        help="Mounted Google Drive path to save intermediate checkpoints and final weights directly (prevents Colab disconnection data loss)")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch", "no"], help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save step frequency")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to retain on disk/drive")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--learning_rate", "--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (useful for verification)")
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

    resolved_model_name = resolve_model_name(args.model_name)
    logger.info(f"Resolved student model name: {resolved_model_name} (from input '{args.model_name}')")

    drive_path = detect_google_drive_path(args.drive_path)
    if drive_path:
        drive_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Target Google Drive path configured: {drive_path}. Intermediate checkpoints and final weights will be stored directly to Google Drive.")

    logger.info(f"Loading tokenizer for {resolved_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    if "text" not in dataset.column_names:
        dataset = dataset.map(get_formatting_func(tokenizer))

    eval_dataset = None
    if args.val_path and Path(args.val_path).exists():
        logger.info(f"Loading validation dataset from {args.val_path}")
        eval_dataset = load_dataset("json", data_files=args.val_path, split="train")
        if "text" not in eval_dataset.column_names:
            eval_dataset = eval_dataset.map(get_formatting_func(tokenizer))

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
            resolved_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("CUDA not available, loading model without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
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

    # Completion-only loss masking so the model only learns to output clean rewrites
    collator = None
    try:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(
            response_template="<|im_start|>assistant\n",
            tokenizer=tokenizer,
        )
    except Exception as exc:
        logger.warning(f"Could not initialize DataCollatorForCompletionOnlyLM: {exc}")

    callbacks = []
    if drive_path and GoogleDriveSyncCallback is not None:
        callbacks.append(GoogleDriveSyncCallback(drive_path=drive_path, local_dir=Path(args.output_dir)))

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
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "optim": "paged_adamw_32bit",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "use_cpu": not is_cuda,
            "gradient_checkpointing": True,
            "max_length": args.max_length,
            "dataset_text_field": "text",
            **get_warmup_kwargs(SFTConfig, 0.05),
        }
        if args.max_steps is not None and args.max_steps > 0:
            sft_kwargs["max_steps"] = args.max_steps
        training_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_kwargs))
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
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
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "optim": "paged_adamw_32bit",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "use_cpu": not is_cuda,
            "gradient_checkpointing": True,
            **get_warmup_kwargs(TrainingArguments, 0.05),
        }
        if args.max_steps is not None and args.max_steps > 0:
            t_kwargs["max_steps"] = args.max_steps
        training_args = TrainingArguments(**filter_supported_kwargs(TrainingArguments, t_kwargs))
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text",
        )

    logger.info("Starting QLoRA fine-tuning")
    trainer.train()

    logger.info(f"Saving final adapter to local directory: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if drive_path:
        logger.info(f"Saving final adapter weights directly to Google Drive: {drive_path}")
        drive_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(drive_path))
        tokenizer.save_pretrained(str(drive_path))
        logger.info(f"Final adapter and tokenizer successfully persisted to Google Drive: {drive_path}")

    # Register student adapter provenance manifest for evaluation readiness
    try:
        from scripts.check_eval_readiness import register_adapter_manifest, load_yaml
        config_p = Path("configs/experiments/session08.yaml")
        if config_p.exists():
            cfg = load_yaml(config_p)
            manifest_file = register_adapter_manifest(config=cfg, root=Path("."))
            logger.info(f"Registered student adapter provenance manifest: {manifest_file}")
            if drive_path and manifest_file.exists():
                shutil.copy2(manifest_file, drive_path / "training_manifest.json")
    except Exception as exc:
        logger.warning(f"Could not automatically register adapter manifest: {exc}")

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
