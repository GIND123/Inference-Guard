"""InferenceGuard Rewriter Module.

`inference` imports torch at module scope, so re-exporting it eagerly here
makes importing *anything* under `src.rewriter` require torch. That is not
hypothetical: `src/evaluation/run.py` imports `generate_training_data` from
this package, so an eager re-export made the whole evaluation harness -- and
`tests/test_evaluation_harness.py`, which passed before -- fail at collection
on any machine without torch installed.

Only Colab has the GPU stack. Teammates working on evaluation, the data layer
or the product modules should not need a multi-gigabyte dependency to run
their own tests, so the heavy names resolve on first access instead (PEP 562).
The public API is unchanged: `from src.rewriter import QwenRewriterInference`
still works and still imports torch, just not until something asks for it.
"""

from typing import Any

from src.rewriter.generate_training_data import (
    DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT,
    generate_candidate_rewrites_heuristic,
)

# Resolved from src.rewriter.inference on first attribute access.
_LAZY = frozenset({"DEFAULT_BASE_MODEL", "QwenRewriterInference", "clean_gpu_memory"})

__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_INSTRUCTION",
    "QwenRewriterInference",
    "SYSTEM_PROMPT",
    "clean_gpu_memory",
    "generate_candidate_rewrites_heuristic",
]


def __getattr__(name: str) -> Any:
    """Import the torch-dependent names only when one is actually used."""
    if name in _LAZY:
        from src.rewriter import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
