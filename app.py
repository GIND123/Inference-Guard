"""Entrypoint for Hugging Face Spaces and local production serving.

Launches the InferenceGuard FastAPI backend with mounted static web interface (web/index.html).
Configured for automatic compatibility with Hugging Face Spaces (port 7860) and local serving.
"""

from __future__ import annotations

import os
import uvicorn

from src.api.app import app

if __name__ == "__main__":
    # Hugging Face Spaces runs by default on port 7860; local dev defaults to 8000
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting InferenceGuard service at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
