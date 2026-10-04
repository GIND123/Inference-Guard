"""InferenceGuard Rewriter Module."""

from src.rewriter.constants import DEFAULT_BASE_MODEL
from src.rewriter.generate_training_data import (
    DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT,
    generate_candidate_rewrites_heuristic,
)

__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_INSTRUCTION",
    "QwenRewriterInference",
    "SYSTEM_PROMPT",
    "clean_gpu_memory",
    "generate_candidate_rewrites_heuristic",
]

def __getattr__(name: str):
    if name == "QwenRewriterInference":
        from src.rewriter.inference import QwenRewriterInference
        return QwenRewriterInference
    if name == "clean_gpu_memory":
        from src.rewriter.inference import clean_gpu_memory
        return clean_gpu_memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
