"""1-Click Presentation Desktop App Launcher for InferenceGuard.

Starts (or reuses) the FastAPI GPU backend on http://127.0.0.1:8000,
pre-warms the Hugging Face Hub models (/ready), and opens InferenceGuard
in a clean standalone Chrome/Edge desktop app window (--app mode).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_URL = "http://127.0.0.1:8000"


def _check_ready(timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"{APP_URL}/ready", timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def _find_chromium_app_browser() -> str | None:
    candidates = [
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return str(path)
    return None


def _open_desktop_window() -> None:
    browser = _find_chromium_app_browser()
    if browser:
        print(f"[InferenceGuard] Opening native app window via {Path(browser).name}...")
        subprocess.Popen(
            [browser, f"--app={APP_URL}", "--window-size=1280,900"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[InferenceGuard] Opening {APP_URL} in default browser...")
        webbrowser.open(APP_URL)


def main() -> None:
    os.chdir(REPO_ROOT)
    print("============================================================")
    print("  InferenceGuard — Dual-Edition AI Privacy Presentation App")
    print("============================================================")

    ready = _check_ready(timeout=3.0)
    server_proc: subprocess.Popen | None = None

    if ready:
        print(
            f"[InferenceGuard] Backend already running and ready!\n"
            f"  Risk Model : {ready.get('model_id')} ({ready.get('risk_model_source')})\n"
            f"  Rewriter   : {ready.get('rewriter_model_id')} on {ready.get('device')}"
        )
    else:
        print("[InferenceGuard] Starting FastAPI GPU server on http://127.0.0.1:8000 ...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "web.api:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(REPO_ROOT),
        )
        print("[InferenceGuard] Pre-warming Hugging Face Hub models (ModernBERT + Qwen 1.5B QLoRA)...")
        deadline = time.time() + 180.0
        while time.time() < deadline:
            ready = _check_ready(timeout=60.0)
            if ready and ready.get("ready"):
                print(
                    f"[InferenceGuard] Models warm & ready!\n"
                    f"  Risk Model : {ready.get('model_id')} ({ready.get('risk_model_source')})\n"
                    f"  Rewriter   : {ready.get('rewriter_model_id')} on {ready.get('device')}"
                )
                break
            time.sleep(2.0)
        else:
            print("[InferenceGuard] Warning: Timed out waiting for /ready; opening UI anyway.")

    _open_desktop_window()

    if server_proc is not None:
        print("\n[InferenceGuard] Press Ctrl+C in this window to stop the presentation server.")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            print("\n[InferenceGuard] Shutting down server...")
            server_proc.terminate()
            server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
