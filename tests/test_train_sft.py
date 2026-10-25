import sys
from unittest.mock import patch
from src.rewriter.train_sft import (
    parse_args,
    get_formatting_func,
    get_warmup_kwargs,
    filter_supported_kwargs,
)
from src.rewriter.constants import DEFAULT_BASE_MODEL

def test_parse_args():
    test_args = ["train_sft.py", "--dataset_path", "dummy.jsonl"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.dataset_path == "dummy.jsonl"
        assert args.model_name == DEFAULT_BASE_MODEL

    # Test --train_data alias
    test_args_alias = ["train_sft.py", "--train_data", "alias.jsonl"]
    with patch.object(sys, "argv", test_args_alias):
        args = parse_args()
        assert args.dataset_path == "alias.jsonl"

def test_warmup_and_filter_helpers():
    class DummyConfig:
        def __init__(self, warmup_steps=0.0, output_dir=None):
            pass

    warmup_res = get_warmup_kwargs(DummyConfig, 0.05)
    assert warmup_res == {"warmup_steps": 0.05}

    filtered = filter_supported_kwargs(DummyConfig, {"output_dir": "out", "invalid_param": 123})
    assert "output_dir" in filtered
    assert "invalid_param" not in filtered

def test_format_prompt():
    class DummyTokenizer:
        eos_token = "<eos>"
    format_prompt = get_formatting_func(DummyTokenizer())
    dummy = {"instruction": "Test instruction", "input": "Test input", "output": "Test output"}
    res = format_prompt(dummy)
    assert "Instruction:\nTest instruction" in res["text"]
    assert "Input:\nTest input" in res["text"]
    assert "Output:\nTest output" in res["text"]
