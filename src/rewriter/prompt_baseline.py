# src/rewriter/prompt_baseline.py
"""Frozen Prompt-Only Baseline privacy rewriter service.

Wraps the base causal language model without adapters using deterministic structured
prompting, serving as the experimental baseline (Ablation B) against QLoRA SFT.
"""

from __future__ import annotations

import re
import time
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.contracts.schemas import ConversationState, RewriteMode, RiskReport
from src.rewriter.constants import DEFAULT_BASE_MODEL


class PromptBaselineRewriter:
    """Deterministic prompt-only baseline rewriter operating without fine-tuned adapters."""

    def __init__(
        self,
        model_name: str = DEFAULT_BASE_MODEL,
        device: Optional[str] = None,
        base_model: Optional[object] = None,
        tokenizer: Optional[object] = None,
    ):
        """Initialize the prompt baseline rewriter with base model and tokenizer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, trust_remote_code=True)

        if base_model is not None:
            self.model = base_model
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                ).eval()
            except Exception:
                self.model = None

    def rewrite(
        self,
        text: str,
        risk_report: Optional[RiskReport] = None,
        conversation_state: Optional[ConversationState] = None,
    ) -> Tuple[str, float]:
        """Rewrite input text using frozen base model and structured prompt instructions."""
        start_ts = time.perf_counter()
        if not text.strip():
            return text, 0.0

        if self.model is None:
            # Graceful fallback when running in lightweight CPU environment without base model weights
            elapsed = (time.perf_counter() - start_ts) * 1000.0
            return text, round(elapsed, 2)

        prompt = (
            f"<|im_start|>system\n"
            f"You are InferenceGuard, a privacy rewriter. "
            f"Rewrite the message to remove demographic and regional clues while preserving the core task intent. "
            f"Output ONLY the rewritten message with no meta commentary.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(256, int(input_len * 1.5) + 32),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][input_len:]
        rewritten = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        rewritten = re.sub(r"<\|im_end\|>.*$", "", rewritten, flags=re.DOTALL).strip()
        rewritten = re.sub(r"^(Here is the rewritten (text|message):\s*)", "", rewritten, flags=re.IGNORECASE).strip(' "')

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        return rewritten or text, round(elapsed_ms, 2)
