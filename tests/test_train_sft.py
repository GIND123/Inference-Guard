import sys
from unittest.mock import patch
from src.rewriter.train_sft import parse_args, get_formatting_func
from src.rewriter.constants import DEFAULT_BASE_MODEL

def test_parse_args():
    test_args = ["train_sft.py", "--dataset_path", "dummy.jsonl"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.dataset_path == "dummy.jsonl"
        assert args.model_name == DEFAULT_BASE_MODEL

def test_format_prompt():
    class DummyTokenizer:
        eos_token = "<eos>"
    format_prompt = get_formatting_func(DummyTokenizer())
    dummy = {"instruction": "Test instruction", "input": "Test input", "output": "Test output"}
    res = format_prompt(dummy)
    assert "Instruction:\nTest instruction" in res["text"]
    assert "Input:\nTest input" in res["text"]
    assert "Output:\nTest output" in res["text"]
