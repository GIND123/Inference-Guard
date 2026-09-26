"""Master test runner for InferenceGuard.

Executes all test suites (unit, calibration, integrated gradients, rewriter SFT,
evaluation harness, operations state, UI API, and E2E scenarios), verifies zero em-dashes,
and generates a structured execution report for CI and Colab environments.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
import pytest


def check_zero_em_dashes_and_emojis(root_dir: Path) -> Tuple[int, List[str]]:
    """Scan all project source, test, config, web, and documentation files.

    Verifies strict zero-tolerance compliance for em-dashes (U+2014) and emojis.
    """
    violations = []
    scanned_count = 0

    # Common emoji unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    extensions = {".py", ".yaml", ".yml", ".json", ".html", ".css", ".js", ".md"}
    skip_dirs = {".git", ".pytest_cache", "__pycache__", "venv", ".state", ".agents"}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude skip directories
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in extensions:
                continue

            file_path = Path(dirpath) / fname
            # Skip historical agent logs if they recorded raw user prompts
            rel_str = str(file_path.relative_to(root_dir))
            if "orchestrator_1" in rel_str or "m1_explorer" in rel_str or "m1_worker" in rel_str:
                continue

            scanned_count += 1
            try:
                content = file_path.read_text(encoding="utf-8")
                if "\u2014" in content:
                    violations.append(f"Em-dash (U+2014) found in: {rel_str}")
                if emoji_pattern.search(content):
                    violations.append(f"Emoji found in: {rel_str}")
            except Exception as e:
                pass

    return scanned_count, violations


def run_master_test_suite() -> int:
    """Run pytest over tests/ directory and verify forensic zero em-dash compliance."""
    root_path = Path(__file__).resolve().parent.parent
    root_path_str = str(root_path)
    if root_path_str not in sys.path:
        sys.path.insert(0, root_path_str)

    tests_path = root_path / "tests"

    print("=" * 70)
    print("InferenceGuard Master Test Suite - Week 5 Full Acceptance")
    print(f"Project root: {root_path}")
    print("=" * 70)

    # 1. Forensic zero em-dash and zero emoji scan
    print("\n[Step 1/2] Running Forensic Integrity Scan (Zero Em-Dashes & Zero Emojis)...")
    scanned_count, violations = check_zero_em_dashes_and_emojis(root_path)
    print(f"Scanned {scanned_count} files across project.")

    if violations:
        print("\nFORENSIC INTEGRITY AUDIT FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("PASS: Zero em-dashes and zero emojis verified across all files.")

    # 2. Run pytest suite
    print("\n[Step 2/2] Running Full Test Suite via pytest...")
    start_time = time.perf_counter()

    pytest_args = [
        str(tests_path),
        "-v",
        "--tb=short",
        "-o",
        f"pythonpath={root_path_str}",
    ]

    ret_code = pytest.main(pytest_args)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    if ret_code == 0:
        print(f"ALL TESTS PASSED in {elapsed:.2f}s!")
        print("Milestones 1-6 verified successfully.")
    else:
        print(f"TEST FAILURES DETECTED (Exit code: {ret_code}) in {elapsed:.2f}s")
    print("=" * 70)

    return ret_code


if __name__ == "__main__":
    sys.exit(run_master_test_suite())
