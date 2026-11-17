"""Utility script to verify Hugging Face Hub hosted models for InferenceGuard.

Validates that AT707/inferenceguard-rewriter-v1, the RoBERTa risk model, and
the ModernBERT risk model packages can be queried, inspected, and initialized
in the InferenceGuard pipeline.

With --write-manifest (networked host only) it also records the authoritative
revision sha and per-file lfs sha256 checksums for each hosted repo into
artifacts/model_manifest.json. Revisions/checksums are only ever read from a
real HfApi query; they are never fabricated.

The module must import cleanly even when huggingface_hub is NOT installed:
the import is done lazily inside each function, so `python -m py_compile` and a
plain `import` succeed in a dependency-free sandbox.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_hf_models")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST_PATH = ROOT / "artifacts" / "model_manifest.json"

# Authoritative hosted model ids. role -> (repo_id, base_model).
HOSTED_MODELS = {
    "risk_modernbert": ("AT707/inferenceguard-modernbert-risk", "answerdotai/ModernBERT-base"),
    "risk_roberta": ("AT707/InferenceGuard-Roberta-Risk", "roberta-base"),
    "rewriter": ("AT707/inferenceguard-rewriter-v1", "Qwen/Qwen2.5-1.5B-Instruct"),
}
EXPECTED_FILES = ["config.json", "model.safetensors", "tokenizer.json"]


def check_hf_rewriter_metadata(repo_id: str = "AT707/inferenceguard-rewriter-v1") -> bool:
    """Inspect the Hugging Face repository metadata for the rewriter model."""
    logger.info("Checking repository metadata for '%s'...", repo_id)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id)
        logger.info("Successfully reached repository: %s (author: %s)", info.id, info.author)
        logger.info("Total downloads: %s, Sha: %s", getattr(info, "downloads", 0), info.sha[:10] if info.sha else "N/A")

        files = [f.rfilename for f in info.siblings]
        for req in EXPECTED_FILES:
            if any(f.endswith(req) for f in files):
                logger.info("  [PASS] Found %s", req)
            else:
                logger.warning("  [WARN] Missing expected file %s in repository", req)
        return True
    except Exception as exc:
        logger.warning("Could not query Hugging Face API directly (%s: %s).", type(exc).__name__, exc)
        return False


def check_hf_risk_model_metadata(repo_id: str = "AT707/InferenceGuard-Roberta-Risk") -> bool:
    """Inspect the Hugging Face repository metadata for the RoBERTa risk model."""
    logger.info("Checking repository metadata for '%s'...", repo_id)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id)
        logger.info("Successfully reached repository: %s (author: %s)", info.id, info.author)
        files = [f.rfilename for f in info.siblings]
        for req in EXPECTED_FILES:
            if any(f.endswith(req) for f in files):
                logger.info("  [PASS] Found %s", req)
            else:
                logger.warning("  [WARN] Missing expected file %s in repository", req)
        return True
    except Exception as exc:
        logger.warning("Could not query Hugging Face API directly (%s: %s).", type(exc).__name__, exc)
        return False


def check_hf_modernbert_risk_metadata(repo_id: str = "AT707/inferenceguard-modernbert-risk") -> bool:
    """Inspect the Hugging Face repository metadata for the ModernBERT risk model.

    This is the hosted north-star risk scorer (P0 #1). Mirrors the RoBERTa check.
    """
    logger.info("Checking repository metadata for '%s'...", repo_id)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id)
        logger.info("Successfully reached repository: %s (author: %s)", info.id, info.author)
        files = [f.rfilename for f in info.siblings]
        for req in EXPECTED_FILES:
            if any(f.endswith(req) for f in files):
                logger.info("  [PASS] Found %s", req)
            else:
                logger.warning("  [WARN] Missing expected file %s in repository", req)
        return True
    except Exception as exc:
        logger.warning("Could not query Hugging Face API directly (%s: %s).", type(exc).__name__, exc)
        return False


def verify_pipeline_factory() -> bool:
    """Verify that InferenceGuardPipeline.from_pretrained configures both models."""
    logger.info("Verifying InferenceGuardPipeline.from_pretrained wiring...")
    from src.pipeline.orchestrator import InferenceGuardPipeline

    pipeline = InferenceGuardPipeline.from_pretrained(
        risk_model_id="roberta-base",
        rewriter_model_id="AT707/inferenceguard-rewriter-v1",
        device="cpu",
    )
    assert pipeline.rewriter_service is not None
    assert pipeline.rewriter_service.base_model_name == "AT707/inferenceguard-rewriter-v1"
    logger.info("Pipeline rewriter configured correctly: %s", pipeline.rewriter_service.base_model_name)
    return True


def _query_repo_manifest_entry(role: str, repo_id: str, base_model: str) -> dict:
    """Query HfApi for a repo's real revision sha + per-file lfs sha256 checksums.

    Returns a manifest entry dict. Raises on any API failure so the caller can
    decide how to degrade; never fabricates a sha or checksum.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    checksums: dict = {}
    for sibling in info.siblings:
        name = sibling.rfilename
        lfs = getattr(sibling, "lfs", None)
        sha256 = None
        if lfs is not None:
            # lfs may be a dict or an object depending on hub version.
            sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        size = getattr(sibling, "size", None)
        if name in EXPECTED_FILES or sha256 is not None:
            checksums[name] = {"size": size, "sha256": sha256}
    return {
        "role": role,
        "repo_id": repo_id,
        "base_model": base_model,
        "revision": info.sha,
        "expected_files": list(EXPECTED_FILES),
        "checksums": checksums,
        "verified": True,
    }


def write_manifest(manifest_path: Path) -> bool:
    """Query the Hub for every hosted model and write the real revision + checksums.

    Only ever runs on a networked host with huggingface_hub installed. On any
    failure it logs a warning and returns False without fabricating values.
    """
    logger.info("Writing model manifest to '%s' from live Hugging Face Hub metadata...", manifest_path)
    try:
        from huggingface_hub import HfApi  # noqa: F401  (import-time availability check)
    except Exception as exc:
        logger.warning("huggingface_hub is unavailable (%s: %s); cannot write manifest.", type(exc).__name__, exc)
        return False

    import datetime

    models = []
    all_ok = True
    for role, (repo_id, base_model) in HOSTED_MODELS.items():
        try:
            entry = _query_repo_manifest_entry(role, repo_id, base_model)
            logger.info("  [PASS] %s -> %s @ %s", role, repo_id, entry["revision"])
        except Exception as exc:
            logger.warning("  [WARN] Could not query %s (%s): %s", repo_id, type(exc).__name__, exc)
            entry = {
                "role": role,
                "repo_id": repo_id,
                "base_model": base_model,
                "revision": None,
                "expected_files": list(EXPECTED_FILES),
                "checksums": {},
                "verified": False,
            }
            all_ok = False
        models.append(entry)

    manifest = {
        "schema_version": 1,
        "note": (
            "Authoritative hosted model ids for InferenceGuard. Revisions and checksums are "
            "populated by `python scripts/verify_hf_models.py --write-manifest` on a networked "
            "host; they are NOT fabricated. A null revision or empty checksums means the field "
            "has not yet been pinned from a real HfApi query. Weights live on the Hugging Face "
            "Hub and are NOT committed to git."
        ),
        "generated_by": "scripts/verify_hf_models.py --write-manifest",
        "verified_at": datetime.datetime.utcnow().isoformat() + "Z" if all_ok else None,
        "models": models,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Manifest written: %s", manifest_path)
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Hugging Face hosted models for InferenceGuard.")
    parser.add_argument("--rewriter-repo", default=HOSTED_MODELS["rewriter"][0], help="Rewriter model repo ID.")
    parser.add_argument("--risk-repo", default=HOSTED_MODELS["risk_roberta"][0], help="RoBERTa risk model repo ID.")
    parser.add_argument(
        "--modernbert-repo",
        default=HOSTED_MODELS["risk_modernbert"][0],
        help="ModernBERT risk model repo ID (hosted north-star scorer).",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Query the Hub for real revision shas + checksums and (re)write the manifest. Networked host only.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the model manifest JSON (default: artifacts/model_manifest.json).",
    )
    args = parser.parse_args()

    risk_ok = check_hf_risk_model_metadata(args.risk_repo)
    modernbert_ok = check_hf_modernbert_risk_metadata(args.modernbert_repo)
    rewriter_ok = check_hf_rewriter_metadata(args.rewriter_repo)
    pipeline_ok = verify_pipeline_factory()

    manifest_ok = True
    if args.write_manifest:
        manifest_ok = write_manifest(Path(args.manifest))

    if risk_ok and modernbert_ok and rewriter_ok and pipeline_ok and manifest_ok:
        logger.info("All model verification checks completed successfully.")
        return 0
    else:
        logger.warning("Some checks did not pass.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
