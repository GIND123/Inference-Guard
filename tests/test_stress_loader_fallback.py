"""Empirical Stress Test Suite for Model/Adapter Loading, Fallback, and Memory Management.

Stress tests edge cases and error resilience in `src/rewriter/inference.py`:
1. Nonexistent adapter directory (verifying is_adapter_loaded == False and fallback rewriting).
2. Adapter directory with missing or corrupted safetensors/weights file.
3. Adapter config with invalid/mismatched tensor shapes (verifying RuntimeError cleanly trapped).
4. Instantiation with load_in_4bit=True on CPU (verifying no ValueError and warning logged).
5. Batch rewriting with varying text lengths, empty text, whitespace, and single-item batches.
6. Invocation of unload() and clean_gpu_memory() verifying memory release and idempotence.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.rewriter.inference import (
    QwenRewriterInference,
    clean_gpu_memory,
)


# ==============================================================================
# Lightweight CPU Mocks for Fast & Deterministic Testing
# ==============================================================================

class MockTokenizer:
    """Lightweight mock of AutoTokenizer for CPU testing."""

    def __init__(self) -> None:
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"
        self.pad_token_id = 151643
        self.eos_token_id = 151645
        self.padding_side = "left"

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> str:
        prompt_parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)

    def __call__(
        self,
        text: Any,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        **kwargs: Any,
    ) -> Any:
        if isinstance(text, str):
            b_size = 1
        elif isinstance(text, (list, tuple)):
            b_size = len(text)
        else:
            b_size = 1

        input_ids = torch.ones((b_size, 16), dtype=torch.long)
        attention_mask = torch.ones((b_size, 16), dtype=torch.long)

        class MockBatchEncoding(dict):
            def to(self, *args: Any, **kwargs: Any) -> "MockBatchEncoding":
                return self

        return MockBatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> str:
        return "I relocated to a new residence in the metropolitan area.<|im_end|>"

    def batch_decode(
        self,
        sequences: Any,
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        count = len(sequences) if hasattr(sequences, "__len__") else 1
        return [
            f"Neutralized text revision item {idx}.<|im_end|>"
            for idx in range(count)
        ]


class MockModel:
    """Lightweight mock of CausalLM / PeftModel for CPU testing."""

    def __init__(self, fail_on_batch: bool = False) -> None:
        self.device = "cpu"
        self.config = MagicMock()
        self.fail_on_batch = fail_on_batch

    def to(self, *args: Any, **kwargs: Any) -> "MockModel":
        return self

    def eval(self) -> "MockModel":
        return self

    def parameters(self) -> Any:
        return iter([torch.zeros(1)])

    def generate(self, input_ids: Any = None, **kwargs: Any) -> Any:
        b_size = input_ids.shape[0] if input_ids is not None else 1
        if self.fail_on_batch and b_size > 1:
            raise RuntimeError("Simulated OOM or batch generation failure")
        return torch.ones((b_size, 32), dtype=torch.long)


@pytest.fixture
def mock_transformers_peft():
    """Patches AutoTokenizer, AutoModelForCausalLM, and PeftModel with lightweight mocks."""
    mock_tok = MockTokenizer()
    mock_mdl = MockModel()

    patches = [
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_mdl),
        patch("peft.PeftModel.from_pretrained", return_value=mock_mdl),
        patch("peft.PeftConfig.from_pretrained", return_value=MagicMock(base_model_name_or_path="Qwen/Qwen3-1.7B")),
    ]
    for p in patches:
        p.start()
    yield mock_tok, mock_mdl
    for p in patches:
        p.stop()


# ==============================================================================
# 1. Nonexistent Adapter Directory Tests
# ==============================================================================

class TestNonexistentAdapter:
    """Empirical tests for nonexistent, empty, or file-path adapter paths."""

    def test_nonexistent_adapter_dir_sets_flag_false(self):
        nonexistent_path = "artifacts/definitely_does_not_exist_987654321"
        assert not Path(nonexistent_path).exists()

        rewriter = QwenRewriterInference(adapter_path=nonexistent_path)
        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None
        assert rewriter.tokenizer is None

    def test_nonexistent_adapter_dir_fallback_rewriting_works(self):
        nonexistent_path = "artifacts/does_not_exist_fallback_test"
        rewriter = QwenRewriterInference(adapter_path=nonexistent_path)

        input_text = "My name is John Doe, I am 34 years old, and my phone is 555-0199."
        output = rewriter.rewrite(input_text)

        assert isinstance(output, str)
        assert len(output) > 0
        assert "555-0199" not in output
        assert "[PHONE]" in output or "phone" not in output

    def test_none_adapter_path_falls_back(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None

        output = rewriter.rewrite("Hello Alice in Chicago.")
        assert isinstance(output, str)
        assert len(output) > 0

    def test_empty_string_adapter_path_falls_back(self):
        rewriter = QwenRewriterInference(adapter_path="")
        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None

        output = rewriter.rewrite("Hello Bob in Seattle.")
        assert isinstance(output, str)
        assert len(output) > 0

    def test_adapter_path_points_to_file_not_dir(self, tmp_path):
        dummy_file = tmp_path / "not_a_dir.txt"
        dummy_file.write_text("just text", encoding="utf-8")

        rewriter = QwenRewriterInference(adapter_path=str(dummy_file))
        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None

        output = rewriter.rewrite("Testing file path fallback.")
        assert isinstance(output, str)
        assert len(output) > 0


# ==============================================================================
# 2. Missing or Corrupted Adapter Weights Tests
# ==============================================================================

class TestMissingOrCorruptWeights:
    """Empirical tests for missing safetensors, corrupted 0-byte weights, and invalid configs."""

    def test_adapter_dir_missing_safetensors_and_bin(self, tmp_path, caplog):
        # Create adapter dir with adapter_config.json but NO weights
        cfg_file = tmp_path / "adapter_config.json"
        cfg_file.write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "peft_type": "LORA"}),
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None
        assert any("No adapter weight files found in" in r.message for r in caplog.records)

        # Fallback rewrite must operate cleanly
        out = rewriter.rewrite("Sample text to verify fallback after missing weights.")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_adapter_dir_missing_adapter_config(self, tmp_path, caplog):
        # Weight file exists, but adapter_config.json is missing
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"\x00" * 64)

        with caplog.at_level(logging.WARNING):
            rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None
        assert any("Missing 'adapter_config.json'" in r.message for r in caplog.records)

        out = rewriter.rewrite("Fallback verification text.")
        assert len(out) > 0

    def test_corrupted_empty_safetensors_file(self, tmp_path, caplog, mock_transformers_peft):
        # adapter_config.json exists and 0-byte adapter_model.safetensors exists
        cfg_file = tmp_path / "adapter_config.json"
        cfg_file.write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "peft_type": "LORA"}),
            encoding="utf-8",
        )
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"")

        # When PeftModel tries to load the empty file, it raises an exception
        with patch("peft.PeftModel.from_pretrained", side_effect=Exception("SafetensorError: File empty or corrupt")):
            with caplog.at_level(logging.WARNING):
                rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert any("Failed to attach LoRA adapter" in r.message for r in caplog.records)

        out = rewriter.rewrite("Fallback verification text with corrupted weights.")
        assert len(out) > 0

    def test_malformed_adapter_config_json(self, tmp_path, caplog):
        # adapter_config.json with invalid JSON syntax
        cfg_file = tmp_path / "adapter_config.json"
        cfg_file.write_text("{invalid json syntax, not json!", encoding="utf-8")
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"fake_weights")

        with caplog.at_level(logging.WARNING):
            rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert any("Could not parse PeftConfig" in r.message for r in caplog.records)

        out = rewriter.rewrite("Fallback test with malformed config.")
        assert len(out) > 0


# ==============================================================================
# 3. Tensor Shape and Size Mismatch Tests
# ==============================================================================

class TestTensorShapeMismatch:
    """Empirical tests verifying RuntimeError on shape mismatch is trapped cleanly without crashing."""

    def test_peft_runtime_error_tensor_shape_mismatch_trapped(self, tmp_path, caplog, mock_transformers_peft):
        cfg_file = tmp_path / "adapter_config.json"
        cfg_file.write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "peft_type": "LORA"}),
            encoding="utf-8",
        )
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"dummy_weights")

        mismatch_msg = (
            "size mismatch for base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight: "
            "copying a param with shape torch.Size([8, 1024]) from checkpoint, "
            "the shape in current model is torch.Size([8, 512])."
        )

        with patch("peft.PeftModel.from_pretrained", side_effect=RuntimeError(mismatch_msg)):
            with caplog.at_level(logging.WARNING):
                rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert any("RuntimeError" in r.message and "Failed to attach LoRA adapter" in r.message for r in caplog.records)

        # Ensure subsequent operations do not crash
        out = rewriter.rewrite("Testing shape mismatch resilience.")
        assert isinstance(out, str)
        assert len(out) > 0

        batch_out = rewriter.batch_rewrite(["First text", "Second text"])
        assert len(batch_out) == 2

    def test_base_model_runtime_error_trapped(self, tmp_path, caplog):
        cfg_file = tmp_path / "adapter_config.json"
        cfg_file.write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "peft_type": "LORA"}),
            encoding="utf-8",
        )
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"dummy_weights")

        with patch("peft.PeftConfig.from_pretrained", return_value=MagicMock(base_model_name_or_path="Qwen/Qwen3-1.7B")):
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer()):
                with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=RuntimeError("CUDA out of memory or shape error")):
                    with caplog.at_level(logging.WARNING):
                        rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None
        assert any("Could not initialize base model" in r.message for r in caplog.records)

        out = rewriter.rewrite("Base model failure fallback.")
        assert len(out) > 0

    def test_peft_key_error_or_attribute_error_trapped(self, tmp_path, caplog, mock_transformers_peft):
        cfg_file = tmp_path / "adapter_config.json"
        cfg_file.write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B", "peft_type": "LORA"}),
            encoding="utf-8",
        )
        weight_file = tmp_path / "adapter_model.safetensors"
        weight_file.write_bytes(b"dummy_weights")

        with patch("peft.PeftModel.from_pretrained", side_effect=KeyError("target_modules 'q_proj' not found in model")):
            with caplog.at_level(logging.WARNING):
                rewriter = QwenRewriterInference(adapter_path=str(tmp_path))

        assert rewriter.is_adapter_loaded is False
        assert any("KeyError" in r.message for r in caplog.records)
        out = rewriter.rewrite("Resilient fallback test.")
        assert len(out) > 0


# ==============================================================================
# 4. CPU 4-bit Quantization Safety Guard Tests
# ==============================================================================

class TestCpuQuantizationGuard:
    """Empirical tests verifying load_in_4bit=True on CPU raises no ValueError and logs a warning."""

    def test_load_in_4bit_cpu_no_value_error_raised(self, caplog):
        # On CPU, load_in_4bit=True must NOT crash with ValueError
        with caplog.at_level(logging.WARNING):
            rewriter = QwenRewriterInference(
                adapter_path=None,
                load_in_4bit=True,
                device="cpu",
            )

        assert rewriter.load_in_4bit is False
        assert any(
            "4-bit quantization requested but CUDA is not available" in r.message
            for r in caplog.records
        )

    def test_load_in_4bit_default_auto_device_on_cpu(self, caplog):
        # Auto-resolved device when CUDA is unavailable
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.WARNING):
                rewriter = QwenRewriterInference(
                    adapter_path=None,
                    load_in_4bit=True,
                    device=None,
                )

            assert rewriter.device == "cpu"
            assert rewriter.load_in_4bit is False
            assert any(
                "Disabling 4-bit quantization and reverting to standard CPU loading" in r.message
                for r in caplog.records
            )

    def test_load_in_4bit_with_cuda_device_string_when_cuda_unavailable(self, caplog):
        # User specified device="cuda", but system has no CUDA driver
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.WARNING):
                rewriter = QwenRewriterInference(
                    adapter_path=None,
                    load_in_4bit=True,
                    device="cuda",
                )

            assert rewriter.load_in_4bit is False
            assert any("Disabling 4-bit quantization" in r.message for r in caplog.records)


# ==============================================================================
# 5. Batch Rewriting Stress Tests
# ==============================================================================

class TestBatchRewritingStress:
    """Empirical stress tests for batch rewriting under varying sequence lengths, empty text, and failures."""

    def test_batch_rewrite_empty_input_list(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        assert rewriter.batch_rewrite([]) == []

    def test_batch_rewrite_single_item_batch(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        single = ["A single test sentence with John Doe in Seattle."]
        out = rewriter.batch_rewrite(single, batch_size=4)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0] == rewriter.rewrite(single[0])

    def test_batch_rewrite_empty_and_whitespace_strings(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        inputs = ["", "   ", "\t\n\r  ", "Valid sentence."]
        out = rewriter.batch_rewrite(inputs, batch_size=2)
        assert len(out) == 4
        assert out[0] == ""
        assert out[1] == ""
        assert out[2] == ""
        assert len(out[3]) > 0

    def test_batch_rewrite_mixed_lengths_stress(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        inputs = [
            "Hi",
            "Moderate sentence containing John at 555-0199 in Boston, MA.",
            "A " * 300 + "very long repetitive text stream.",
            "",
            "Doctor in California.",
            "B" * 5000,
        ]
        out = rewriter.batch_rewrite(inputs, batch_size=2)
        assert len(out) == len(inputs)
        assert out[3] == ""
        assert "555-0199" not in out[1]
        assert len(out[2]) > 0
        assert len(out[5]) > 0

    def test_batch_rewrite_invariance_across_batch_sizes(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        inputs = [
            "Sample number one.",
            "Sample number two with Alice in Dallas.",
            "Sample number three with Bob at 555-0188.",
            "Sample number four.",
            "Sample number five.",
        ]
        out_b1 = rewriter.batch_rewrite(inputs, batch_size=1)
        out_b2 = rewriter.batch_rewrite(inputs, batch_size=2)
        out_b5 = rewriter.batch_rewrite(inputs, batch_size=5)
        out_b50 = rewriter.batch_rewrite(inputs, batch_size=50)

        assert out_b1 == out_b2 == out_b5 == out_b50

    def test_batch_rewrite_with_adapter_loaded_mixed_lengths(self, mock_transformers_peft):
        # Manually wire mock model and tokenizer to simulate loaded adapter
        rewriter = QwenRewriterInference(adapter_path=None)
        rewriter.model = mock_transformers_peft[1]
        rewriter.tokenizer = mock_transformers_peft[0]
        rewriter._is_adapter_loaded = True

        inputs = [
            "Tiny",
            "",
            "A longer piece of text testing batch padding and token alignment.",
            "   ",
            "Final sentence.",
        ]
        out = rewriter.batch_rewrite(inputs, batch_size=2)
        assert len(out) == len(inputs)
        assert out[1] == ""
        assert out[3] == ""
        assert len(out[0]) > 0
        assert len(out[2]) > 0
        assert len(out[4]) > 0

    def test_batch_rewrite_chunk_failure_falls_back_to_single_rewrite(self, caplog):
        # Model generate fails when batch_size > 1, but succeeds for single item (batch_size == 1)
        mock_tok = MockTokenizer()
        failing_model = MockModel(fail_on_batch=True)

        rewriter = QwenRewriterInference(adapter_path=None)
        rewriter.model = failing_model
        rewriter.tokenizer = mock_tok
        rewriter._is_adapter_loaded = True

        inputs = ["Item A", "Item B", "Item C"]
        with caplog.at_level(logging.WARNING):
            out = rewriter.batch_rewrite(inputs, batch_size=2)

        assert len(out) == 3
        # Should log that batch generation failed and it fell back to single-sample rewrite
        assert any("Batch generation failed for chunk" in r.message for r in caplog.records)
        assert all(isinstance(res, str) and len(res) > 0 for res in out)


# ==============================================================================
# 6. Unload and GPU Memory Release Tests
# ==============================================================================

class TestUnloadAndMemoryManagement:
    """Empirical tests for unload(), clean_gpu_memory(), context manager, and destructor."""

    def test_unload_releases_all_model_references(self, mock_transformers_peft):
        rewriter = QwenRewriterInference(adapter_path=None)
        rewriter.model = mock_transformers_peft[1]
        rewriter.tokenizer = mock_transformers_peft[0]
        rewriter._is_adapter_loaded = True

        assert rewriter.is_adapter_loaded is True
        assert rewriter.model is not None
        assert rewriter.tokenizer is not None

        rewriter.unload()

        assert rewriter.is_adapter_loaded is False
        assert rewriter.model is None
        assert rewriter.tokenizer is None

    def test_unload_is_idempotent(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        # Calling unload repeatedly must not raise exceptions
        rewriter.unload()
        rewriter.unload()
        rewriter.unload()
        assert rewriter.model is None
        assert rewriter.is_adapter_loaded is False

    def test_rewriting_after_unload_falls_back_cleanly(self, mock_transformers_peft):
        rewriter = QwenRewriterInference(adapter_path=None)
        rewriter.model = mock_transformers_peft[1]
        rewriter.tokenizer = mock_transformers_peft[0]
        rewriter._is_adapter_loaded = True

        rewriter.unload()

        # Both rewrite and batch_rewrite must fall back without AttributeError
        out1 = rewriter.rewrite("Test string after unload.")
        assert isinstance(out1, str)
        assert len(out1) > 0

        out2 = rewriter.batch_rewrite(["Test A", "Test B"])
        assert len(out2) == 2
        assert all(len(s) > 0 for s in out2)

    def test_clean_gpu_memory_direct_invocation_on_cpu(self):
        # Must execute without error on CPU
        clean_gpu_memory()

    def test_clean_gpu_memory_calls_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                with patch("torch.cuda.ipc_collect") as mock_ipc_collect:
                    clean_gpu_memory()
                    mock_empty_cache.assert_called_once()
                    mock_ipc_collect.assert_called_once()

    def test_clean_gpu_memory_suppresses_cuda_driver_errors(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache", side_effect=RuntimeError("CUDA driver shutting down")):
                # clean_gpu_memory should catch exception and not crash caller
                clean_gpu_memory()

    def test_context_manager_lifecycle(self, mock_transformers_peft):
        rewriter_instance = None
        with QwenRewriterInference(adapter_path=None) as rewriter:
            rewriter.model = mock_transformers_peft[1]
            rewriter.tokenizer = mock_transformers_peft[0]
            rewriter._is_adapter_loaded = True
            rewriter_instance = rewriter
            assert rewriter_instance.is_adapter_loaded is True

        # Upon exiting with-block, unload() must have been called
        assert rewriter_instance.is_adapter_loaded is False
        assert rewriter_instance.model is None
        assert rewriter_instance.tokenizer is None

    def test_destructor_cleanup_safety(self):
        rewriter = QwenRewriterInference(adapter_path=None)
        # Calling __del__ directly or via garbage collection must not throw
        rewriter.__del__()
        del rewriter
        gc.collect()
