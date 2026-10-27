"""Guards against the two defects this file was written for.

Both are the kind that pass every unit test and only show up on stage:

1. Train, evaluate and serve drifting onto different base models, so a LoRA
   adapter is applied to a model it was never trained against. When that
   fails, QwenRewriterInference falls back to heuristics and logs a warning
   the UI does not surface -- the demo claims Qwen and serves regex.

2. A heavy import creeping into a package `__init__`, making torch-free parts
   of the codebase unimportable for anyone without the GPU stack.

Deliberately dependency-free: these run on a laptop with no torch, which is
the situation they exist to protect.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Every file that names a base model to load. If a new one appears, it should
# be added here and use the shared constant.
MODEL_CONSUMERS = [
    Path("web/api.py"),
    Path("src/evaluation/run.py"),
    Path("src/rewriter/train_sft.py"),
]

QWEN_LITERAL = re.compile(r'["\']Qwen/[\w.\-]+["\']')


def _constant_value() -> str:
    """Read DEFAULT_BASE_MODEL out of the source without importing torch."""
    tree = ast.parse((ROOT / "src/rewriter/inference.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "DEFAULT_BASE_MODEL" for t in node.targets
        ):
            return node.value.value
    raise AssertionError("DEFAULT_BASE_MODEL not found in src/rewriter/inference.py")


def test_the_shared_constant_exists_and_names_a_text_only_model():
    value = _constant_value()
    assert value.startswith("Qwen/")
    # Qwen3.5 is the vision-language family (Qwen3_5ForConditionalGeneration,
    # vision_config, 248k vocab). See docs/methodology-review.md finding 1.
    assert "Qwen3.5" not in value, f"{value} is a vision-language model, not a text LM"


@pytest.mark.parametrize("path", MODEL_CONSUMERS, ids=lambda p: str(p))
def test_no_file_hardcodes_a_base_model_instead_of_the_constant(path):
    """The actual Session 06 defect: three files, two different models.

    train_sft.py trained the adapter on Qwen3-1.7B while web/api.py and
    src/evaluation/run.py both loaded Qwen2.5-1.5B-Instruct. Different hidden
    size, different tokenizer; the adapter cannot transfer.
    """
    source = (ROOT / path).read_text(encoding="utf-8")
    literals = {m.group(0).strip("'\"") for m in QWEN_LITERAL.finditer(source)}
    assert not literals, (
        f"{path} hardcodes {sorted(literals)}; import DEFAULT_BASE_MODEL from "
        "src.rewriter.inference instead so train/evaluate/serve cannot drift"
    )


def test_adapter_config_is_trusted_unconditionally():
    """The guard existed but only fired on the default argument.

    `if extracted_base and self.model_name_or_path == "Qwen/Qwen3-1.7B"` meant
    the adapter's own record of its base model was discarded by exactly the
    two call sites that passed something different.
    """
    source = (ROOT / "src/rewriter/inference.py").read_text(encoding="utf-8")
    assert "if extracted_base:" in source
    assert 'self.model_name_or_path == "Qwen/' not in source, (
        "base-model resolution is conditional on the caller's argument again"
    )


def test_a_mismatch_is_logged_rather_than_swallowed():
    source = (ROOT / "src/rewriter/inference.py").read_text(encoding="utf-8")
    block = source.split("if extracted_base:")[1].split("except")[0]
    assert "logger.warning" in block, "a silent swap is how the demo misreports itself"


# --- import hygiene (pure AST static analysis) -----------------------------

def _get_toplevel_imports(path: Path) -> list[str]:
    """Parse a file's AST and return all module-scope imported module names (including if/try blocks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules: list[str] = []

    def visit(nodes: list[ast.AST]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Lazy / local imports inside functions or classes are allowed
                continue
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module)
            elif isinstance(node, (ast.If, ast.While, ast.For)):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                visit(node.body)

    visit(tree.body)
    return imported_modules


def test_rewriter_package_imports_without_torch():
    """Static AST check: rewriter __init__ must never import torch at module scope."""
    imports = _get_toplevel_imports(ROOT / "src/rewriter/__init__.py")
    assert not any(
        m == "torch" or m.startswith("torch.") or m == "src.rewriter.inference"
        for m in imports
    ), f"src/rewriter/__init__.py has eager heavy imports: {imports}"


def test_evaluation_harness_imports_without_torch():
    """Static AST check: src/evaluation/run.py must never import torch at module scope."""
    imports = _get_toplevel_imports(ROOT / "src/evaluation/run.py")
    assert not any(
        m == "torch" or m.startswith("torch.") or m == "src.rewriter.inference"
        for m in imports
    ), f"src/evaluation/run.py has eager heavy imports: {imports}"



def test_lazy_names_are_still_advertised():
    import src.rewriter as r

    for name in ("QwenRewriterInference", "clean_gpu_memory", "DEFAULT_BASE_MODEL"):
        assert name in r.__all__ and name in dir(r)


def test_unknown_attribute_still_raises_attribute_error():
    import src.rewriter as r

    with pytest.raises(AttributeError):
        r.not_a_real_name


# --- honesty of the names we ship ----------------------------------------

def test_runtime_thresholds_are_not_called_validated():
    """We have been bitten twice by a confident label on an unearned number.

    A simulated benchmark committed to reports/, and a training-split F1
    reported as the north star. The runtime cut-offs are carried over
    unchanged from the Session 04 notebook and nothing fits them, so they may
    not be named as though something did. `fit_attribute()` is what would earn
    that name; when it is wired up, this test is the thing to delete.
    """
    src = (ROOT / "src/product/thresholds.py").read_text(encoding="utf-8")
    assert "VALIDATED_" not in src, (
        "a threshold constant is named VALIDATED but nothing calls fit_attribute()"
    )
    assert "PROVISIONAL_MAX_RISK_THRESHOLD" in src


def test_committed_user_study_records_are_parseable():
    """session06_record.json shipped as four concatenated objects, so every
    reader of it — including our own aggregation harness — raised
    JSONDecodeError. Evidence nothing can open is not evidence."""
    import json

    for path in sorted((ROOT / "docs").rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")


# --- stdlib APIs newer than our declared floor ----------------------------

# (pattern, first version that has it, what to use instead). One entry today;
# the shape is here so the next one is a line rather than a new test.
VERSION_GATED_APIS = [
    (
        re.compile(r"\.write_text\([^)]*\bnewline\s*="),
        (3, 10),
        "Path.write_bytes(text.encode('utf-8')), or open(..., newline=...)",
    ),
]


def _declared_python_floor() -> tuple[int, int]:
    """Lowest Python pyproject.toml claims to support.

    Read with a regex rather than tomllib, which is itself 3.11+ — a version
    check that only runs on new Pythons would miss the thing it is checking.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*["\']>=\s*(\d+)\.(\d+)', text)
    assert m, "pyproject.toml does not declare requires-python"
    return int(m.group(1)), int(m.group(2))


def test_source_does_not_use_apis_newer_than_the_declared_python_floor():
    """CI runs one Python version, so version drift is invisible to it.

    `src/data/splits.py` called `Path.write_text(..., newline="\\n")`, which
    arrived in 3.10, while pyproject declared >=3.9. CI on 3.11 passed, and
    the failure only appeared on a developer machine running the version we
    claim to support. A static check is the cheapest way to catch the class.
    """
    floor = _declared_python_floor()
    offenders = []
    for path in sorted(ROOT.glob("src/**/*.py")) + sorted(ROOT.glob("scripts/*.py")):
        source = path.read_text(encoding="utf-8")
        for pattern, added_in, alternative in VERSION_GATED_APIS:
            if added_in > floor and pattern.search(source):
                offenders.append(
                    f"{path.relative_to(ROOT)} uses an API added in "
                    f"{added_in[0]}.{added_in[1]}, but requires-python is "
                    f">={floor[0]}.{floor[1]} — use {alternative}"
                )
    assert not offenders, "\n".join(offenders)
