"""
Fine-tune the Qwen LLM on the generated SFT dataset using QLoRA.
"""

import argparse
import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for InferenceGuard Rewriter")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Base model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset (ChatML or Alpaca format)")
    parser.add_argument("--output_dir", type=str, default="artifacts/rewriter_qlora", help="Output directory for adapters")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    return parser.parse_args()

def format_prompt(example):
    """Format Alpaca schema to text."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    prompt = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n{output}"
    return {"text": prompt}

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    if "text" not in dataset.column_names:
        dataset = dataset.map(format_prompt)

    logger.info("Configuring 4-bit quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    logger.info("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

    # Note: Using standard SFTTrainer format for ease of instruction tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text"
    )

    logger.info("Starting QLoRA fine-tuning")
    trainer.train()

    logger.info(f"Saving final adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
