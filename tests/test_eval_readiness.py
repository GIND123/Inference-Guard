from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.check_eval_readiness import (
    ADAPTER_MANIFEST_SCHEMA,
    check_readiness,
    register_adapter_manifest,
)


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def make_fixture(tmp_path: Path):
    train = tmp_path / "data/splits/train.jsonl"
    test = tmp_path / "data/splits/test.jsonl"
    sft = tmp_path / "data/sft/teacher_v1/train.jsonl"

    train_hash = write_bytes(train, b'{"profile_id":"train-1"}\n')
    test_hash = write_bytes(test, b'{"profile_id":"test-1"}\n')
    sft_hash = write_bytes(sft, b'{"messages":[]}\n')

    split_manifest = tmp_path / "data/manifests/split_manifest.json"
    write_json(
        split_manifest,
        {"train": {"sha256": train_hash}, "test": {"sha256": test_hash}},
    )

    teacher_path = tmp_path / "data/sft/teacher_v1/manifest_train.json"
    teacher = {
        "split": "train",
        "git_dirty": False,
        "teacher": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "revision": "1" * 40,
        },
        "nli": {
            "model_id": "cross-encoder/nli-deberta-v3-large",
            "revision": "2" * 40,
        },
        "input": {
            "path": "data/splits/train.jsonl",
            "sha256": train_hash,
        },
        "outputs": {
            "sft": {
                "path": "data/sft/teacher_v1/train.jsonl",
                "sha256": sft_hash,
            }
        },
    }
    teacher_hash = write_json(teacher_path, teacher)

    write_json(
        tmp_path / "artifacts/profile_splits.json",
        {"seed": 42},
    )

    adapter_dir = tmp_path / "artifacts/rewriter_qlora"

    config = {
        "experiment_id": "session08_heldout_eval_v1",
        "session": 8,
        "experiment_type": "heldout_evaluation",
        "schema_version": "session08_v1",
        "random_seed": 42,
        "dataset": {
            "name": "RobinSta/SynthPAI",
            "revision": "a" * 40,
            "split": "test",
        },
        "split_provenance": {
            "profile_split_file": "artifacts/profile_splits.json",
            "manifest_file": "data/manifests/split_manifest.json",
            "test_split_file": "data/splits/test.jsonl",
            "expected_seed": 42,
            "require_profile_disjoint": True,
            "require_manifest_hashes": True,
        },
        "risk_model": {
            "name": "ModernBERT-base",
            "checkpoint": "artifacts/risk_model/best_model",
        },
        "rewriter": {
            "name": "Qwen/Qwen2.5-1.5B-Instruct",
            "checkpoint": "artifacts/rewriter_qlora",
        },
        "training_provenance": {
            "teacher_manifest_file": "data/sft/teacher_v1/manifest_train.json",
            "training_dataset_file": "data/sft/teacher_v1/train.jsonl",
            "require_teacher_manifest_clean_git": True,
            "require_training_dataset_hash_match": True,
            "require_train_split_hash_match": True,
            "forbid_attacker_as_teacher": True,
            "require_teacher_revision": True,
        },
        "student_provenance": {
            "active_base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter_dir": "artifacts/rewriter_qlora",
            "adapter_manifest_file": "artifacts/rewriter_qlora/training_manifest.json",
            "require_adapter_manifest": True,
            "require_base_model_match": True,
            "require_training_dataset_hash_match": True,
            "require_teacher_manifest_hash_match": True,
            "require_adapter_file_hashes": True,
        },
        "attacker": {
            "name": "microsoft/Phi-4-mini-instruct",
            "required_mode": "model",
        },
        "utility_evaluator": {
            "semantic_similarity_model": "sentence-transformers/all-MiniLM-L6-v2",
            "nli_model": "cross-encoder/nli-deberta-v3-large",
            "required_backend_for_nli_claims": "model",
        },
        "evaluation": {
            "profile_disjoint_test_only": True,
            "heavy_models_required": True,
            "allow_diagnostic_fallbacks": True,
        },
        "reportability": {
            "privacy": {
                "require_attacker_model_match": True,
                "require_attacker_mode_model": True,
                "require_reportable_as_held_out": True,
                "require_test_split": True,
                "require_pinned_dataset_revision": True,
                "require_split_manifest": True,
            },
            "utility": {
                "require_model_backend_for_nli_claims": True,
                "fallback_metrics_diagnostic_only": True,
            },
            "rewrite": {
                "require_engine_provenance": True,
                "unchanged_high_risk_is_not_protected": True,
            },
        },
        "metrics": {"privacy": [], "utility": [], "provenance": [], "performance": []},
        "output": {"root": "experiments/session08"},
        "validation": {
            "fail_on_missing_required_fields": True,
            "fail_on_dataset_revision_mismatch": True,
            "fail_on_split_mismatch": True,
            "fail_on_missing_split_manifest": True,
            "fail_on_attacker_model_mismatch": True,
            "fail_on_non_model_attacker_for_privacy_claim": True,
            "fail_on_false_heldout_reportability": True,
            "block_nli_claims_on_fallback_backend": True,
            "require_rewrite_engine_provenance": True,
            "require_git_commit_sha": True,
        },
    }

    return {
        "config": config,
        "sft_hash": sft_hash,
        "teacher_hash": teacher_hash,
        "teacher_path": teacher_path,
        "adapter_dir": adapter_dir,
    }


def add_adapter(fx):
    d = fx["adapter_dir"]
    ac_hash = write_json(
        d / "adapter_config.json",
        {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"},
    )
    model_hash = write_bytes(
        d / "adapter_model.safetensors",
        b"fake-adapter",
    )
    manifest = {
        "schema_version": ADAPTER_MANIFEST_SCHEMA,
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "training_dataset": {
            "path": "data/sft/teacher_v1/train.jsonl",
            "sha256": fx["sft_hash"],
        },
        "teacher_manifest": {
            "path": "data/sft/teacher_v1/manifest_train.json",
            "sha256": fx["teacher_hash"],
        },
        "adapter_files": {
            "adapter_config.json": ac_hash,
            "adapter_model.safetensors": model_hash,
        },
    }
    write_json(d / "training_manifest.json", manifest)


def test_missing_adapter_blocks_but_other_core_checks_pass(tmp_path):
    fx = make_fixture(tmp_path)
    r = check_readiness(fx["config"], root=tmp_path)
    assert r["overall_status"] == "BLOCKED"
    assert r["checks"]["config_contract"]["status"] == "PASS"
    assert r["checks"]["training_dataset"]["status"] == "PASS"
    assert r["checks"]["test_split"]["status"] == "PASS"
    assert r["checks"]["student_adapter"]["status"] == "BLOCKED"


def test_good_adapter_is_ready(tmp_path):
    fx = make_fixture(tmp_path)
    add_adapter(fx)
    r = check_readiness(fx["config"], root=tmp_path)
    assert r["overall_status"] == "READY"


def test_training_dataset_hash_mismatch_blocks(tmp_path):
    fx = make_fixture(tmp_path)
    (tmp_path / "data/sft/teacher_v1/train.jsonl").write_bytes(b"tampered\n")
    r = check_readiness(fx["config"], root=tmp_path)
    assert any("Training dataset SHA-256" in x for x in r["blocking_reasons"])


def test_test_split_hash_mismatch_blocks(tmp_path):
    fx = make_fixture(tmp_path)
    add_adapter(fx)
    (tmp_path / "data/splits/test.jsonl").write_bytes(b"tampered\n")
    r = check_readiness(fx["config"], root=tmp_path)
    assert any("Test split SHA-256" in x for x in r["blocking_reasons"])


def test_attacker_cannot_be_teacher(tmp_path):
    fx = make_fixture(tmp_path)
    t = json.loads(fx["teacher_path"].read_text(encoding="utf-8"))
    t["teacher"]["model_id"] = "microsoft/Phi-4-mini-instruct"
    write_json(fx["teacher_path"], t)
    r = check_readiness(fx["config"], root=tmp_path)
    assert any("may not be used as the SFT teacher" in x for x in r["blocking_reasons"])


def test_missing_teacher_revision_blocks(tmp_path):
    fx = make_fixture(tmp_path)
    t = json.loads(fx["teacher_path"].read_text(encoding="utf-8"))
    t["teacher"]["revision"] = None
    write_json(fx["teacher_path"], t)
    r = check_readiness(fx["config"], root=tmp_path)
    assert any("Teacher model revision is not pinned" in x for x in r["blocking_reasons"])


def test_adapter_file_hash_mismatch_blocks(tmp_path):
    fx = make_fixture(tmp_path)
    add_adapter(fx)
    (fx["adapter_dir"] / "adapter_model.safetensors").write_bytes(b"tampered")
    r = check_readiness(fx["config"], root=tmp_path)
    assert any("Adapter artifact hash mismatch" in x for x in r["blocking_reasons"])



def init_git_repo(path: Path):
    subprocess.run(
        ["git", "init"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
    )

    marker = path / ".test-git-marker"
    marker.write_text("test\n", encoding="utf-8")

    subprocess.run(
        ["git", "add", "."],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def add_unregistered_adapter(fx, base_model=None):
    d = fx["adapter_dir"]

    write_json(
        d / "adapter_config.json",
        {
            "base_model_name_or_path": (
                base_model
                or "Qwen/Qwen2.5-1.5B-Instruct"
            )
        },
    )

    write_bytes(
        d / "adapter_model.safetensors",
        b"realistic-fake-adapter-bytes",
    )


def test_register_adapter_manifest_records_exact_artifacts(
    tmp_path,
):
    fx = make_fixture(tmp_path)
    add_unregistered_adapter(fx)
    init_git_repo(tmp_path)

    manifest_path = register_adapter_manifest(
        fx["config"],
        root=tmp_path,
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["schema_version"]
        == ADAPTER_MANIFEST_SCHEMA
    )

    assert (
        manifest["base_model_id"]
        == "Qwen/Qwen2.5-1.5B-Instruct"
    )

    assert (
        manifest["training_dataset"]["sha256"]
        == fx["sft_hash"]
    )

    assert (
        manifest["teacher_manifest"]["sha256"]
        == fx["teacher_hash"]
    )

    assert (
        "adapter_config.json"
        in manifest["adapter_files"]
    )

    assert (
        "adapter_model.safetensors"
        in manifest["adapter_files"]
    )

    assert len(
        manifest["git_commit_sha"]
    ) == 40

    # Once registered, the same artifact chain becomes READY.
    readiness = check_readiness(
        fx["config"],
        root=tmp_path,
    )

    assert readiness["overall_status"] == "READY"


def test_register_adapter_manifest_rejects_wrong_base_model(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    add_unregistered_adapter(
        fx,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    init_git_repo(tmp_path)

    try:
        register_adapter_manifest(
            fx["config"],
            root=tmp_path,
        )
    except ValueError as exc:
        assert (
            "does not match active evaluation student"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected wrong-base-model registration to fail."
        )



def test_register_adapter_manifest_records_provenance(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    adapter_dir = fx["adapter_dir"]

    write_json(
        adapter_dir / "adapter_config.json",
        {
            "base_model_name_or_path":
                "Qwen/Qwen2.5-1.5B-Instruct"
        },
    )

    write_bytes(
        adapter_dir / "adapter_model.safetensors",
        b"registered-adapter",
    )

    manifest_path = register_adapter_manifest(
        config=fx["config"],
        root=tmp_path,
        git_commit_sha="4" * 40,
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["schema_version"]
        == ADAPTER_MANIFEST_SCHEMA
    )

    assert (
        manifest["base_model_id"]
        == "Qwen/Qwen2.5-1.5B-Instruct"
    )

    assert (
        manifest[
            "training_dataset"
        ]["sha256"]
        == fx["sft_hash"]
    )

    assert (
        manifest[
            "teacher_manifest"
        ]["sha256"]
        == fx["teacher_hash"]
    )

    assert (
        manifest["git_commit_sha"]
        == "4" * 40
    )

    assert (
        "adapter_config.json"
        in manifest["adapter_files"]
    )

    assert (
        "adapter_model.safetensors"
        in manifest["adapter_files"]
    )


def test_registered_adapter_transitions_readiness_to_ready(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    adapter_dir = fx["adapter_dir"]

    write_json(
        adapter_dir / "adapter_config.json",
        {
            "base_model_name_or_path":
                "Qwen/Qwen2.5-1.5B-Instruct"
        },
    )

    write_bytes(
        adapter_dir / "adapter_model.safetensors",
        b"registered-adapter",
    )

    register_adapter_manifest(
        config=fx["config"],
        root=tmp_path,
        git_commit_sha="5" * 40,
    )

    result = check_readiness(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["overall_status"]
        == "READY"
    )

    assert (
        result["checks"][
            "student_adapter"
        ]["status"]
        == "PASS"
    )
