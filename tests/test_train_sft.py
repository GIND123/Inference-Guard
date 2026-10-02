import sys
from unittest.mock import patch
from src.rewriter.train_sft import parse_args, format_prompt

def test_parse_args():
    test_args = ["train_sft.py", "--dataset_path", "dummy.jsonl"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.dataset_path == "dummy.jsonl"
        assert args.model_name == "Qwen/Qwen3-1.7B"

def test_format_prompt():
    dummy = {"instruction": "Test instruction", "input": "Test input", "output": "Test output"}
    res = format_prompt(dummy)
    assert "Instruction:\nTest instruction" in res["text"]
    assert "Input:\nTest input" in res["text"]
    assert "Output:\nTest output" in res["text"]
