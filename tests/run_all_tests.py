"""Master test runner for InferenceGuard.

Executes all test suites (unit, calibration, integrated gradients, rewriter SFT,
evaluation harness, operations state, UI API, and E2E scenarios),
and generates a structured execution report for CI and Colab environments.
"""

import sys
import time
from pathlib import Path
import pytest

def run_master_test_suite() -> int:
    """Run pytest over tests/ directory."""
    root_path = Path(__file__).resolve().parent.parent
    root_path_str = str(root_path)
    if root_path_str not in sys.path:
        sys.path.insert(0, root_path_str)

    tests_path = root_path / "tests"

    print("=" * 70)
    print("InferenceGuard Master Test Suite")
    print(f"Project root: {root_path}")
    print("=" * 70)

    print("\n[Step 1/1] Running Full Test Suite via pytest...")
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
        print("System verified successfully.")
    else:
        print(f"TEST FAILURES DETECTED (Exit code: {ret_code}) in {elapsed:.2f}s")
    print("=" * 70)

    return ret_code

if __name__ == "__main__":
    sys.exit(run_master_test_suite())
