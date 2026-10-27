from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from scripts.validate_experiment import validate_config
except ModuleNotFoundError:
    from validate_experiment import validate_config

READINESS_SCHEMA = "session08_eval_readiness_v1"
ADAPTER_MANIFEST_SCHEMA = "session08_student_adapter_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def new_check() -> Dict[str, Any]:
    return {"status": "PASS", "blocking_reasons": [], "warnings": []}


def block(check: Dict[str, Any], message: str) -> None:
    check["status"] = "BLOCKED"
    check["blocking_reasons"].append(message)


def warn(check: Dict[str, Any], message: str) -> None:
    check["warnings"].append(message)


def validate_new_contract(config: Dict[str, Any], check: Dict[str, Any]) -> None:
    training = config.get("training_provenance")
    student = config.get("student_provenance")
    split_cfg = config.get("split_provenance", {})

    if not isinstance(training, dict):
        block(check, "training_provenance section is missing")
        return

    if not isinstance(student, dict):
        block(check, "student_provenance section is missing")
        return

    required_training = [
        "teacher_manifest_file",
        "training_dataset_file",
        "require_teacher_manifest_clean_git",
        "require_training_dataset_hash_match",
        "require_train_split_hash_match",
        "forbid_attacker_as_teacher",
        "require_teacher_revision",
    ]
    required_student = [
        "active_base_model",
        "adapter_dir",
        "adapter_manifest_file",
        "require_adapter_manifest",
        "require_base_model_match",
        "require_training_dataset_hash_match",
        "require_teacher_manifest_hash_match",
        "require_adapter_file_hashes",
    ]

    for field in required_training:
        if field not in training:
            block(check, f"training_provenance missing {field}")

    for field in required_student:
        if field not in student:
            block(check, f"student_provenance missing {field}")

    if not split_cfg.get("test_split_file"):
        block(check, "split_provenance.test_split_file is missing")

    expected = config.get("rewriter", {}).get("name")
    actual = student.get("active_base_model")
    if actual != expected:
        block(check, f"active_base_model mismatch: {actual!r} != {expected!r}")



def _git_snapshot(root: Path) -> Dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return {
        "git_commit_sha": head,
        "git_dirty": bool(status),
    }


def register_adapter_manifest(
    config: Dict[str, Any],
    root: Path = Path("."),
    git_commit_sha: Optional[str] = None,
) -> Path:
    """
    Register an already-trained student adapter for held-out evaluation.

    This function does NOT train or modify model weights.

    It records:
      - student base model,
      - SFT training dataset SHA-256,
      - teacher manifest SHA-256,
      - adapter artifact SHA-256 values,
      - git commit associated with registration.

    The resulting training_manifest.json allows the Student Adapter
    readiness check to move from BLOCKED to PASS.
    """
    root = root.resolve()

    student_cfg = config.get(
        "student_provenance",
        {},
    )

    training_cfg = config.get(
        "training_provenance",
        {},
    )

    adapter_dir = resolve(
        root,
        student_cfg.get(
            "adapter_dir",
            "__missing_adapter__",
        ),
    )

    manifest_path = resolve(
        root,
        student_cfg.get(
            "adapter_manifest_file",
            "__missing_adapter_manifest__",
        ),
    )

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Student adapter directory not found: {adapter_dir}"
        )

    adapter_config_path = (
        adapter_dir / "adapter_config.json"
    )

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            "PEFT adapter_config.json not found: "
            f"{adapter_config_path}"
        )

    adapter_config = load_json(
        adapter_config_path
    )

    actual_base_model = adapter_config.get(
        "base_model_name_or_path"
    )

    expected_base_model = student_cfg.get(
        "active_base_model"
    )

    if actual_base_model != expected_base_model:
        raise ValueError(
            "Adapter base model does not match active "
            "evaluation student: "
            f"{actual_base_model!r} != "
            f"{expected_base_model!r}"
        )

    training_dataset_path = resolve(
        root,
        training_cfg.get(
            "training_dataset_file",
            "__missing_training_dataset__",
        ),
    )

    teacher_manifest_path = resolve(
        root,
        training_cfg.get(
            "teacher_manifest_file",
            "__missing_teacher_manifest__",
        ),
    )

    if not training_dataset_path.exists():
        raise FileNotFoundError(
            "Training dataset not found: "
            f"{training_dataset_path}"
        )

    if not teacher_manifest_path.exists():
        raise FileNotFoundError(
            "Teacher manifest not found: "
            f"{teacher_manifest_path}"
        )

    # Hash every adapter artifact except the manifest
    # that this function is about to create.
    adapter_files: Dict[str, str] = {}

    for file_path in sorted(
        adapter_dir.rglob("*")
    ):
        if not file_path.is_file():
            continue

        if (
            file_path.resolve()
            == manifest_path.resolve()
        ):
            continue

        relative_path = (
            file_path
            .relative_to(adapter_dir)
            .as_posix()
        )

        adapter_files[
            relative_path
        ] = sha256_file(
            file_path
        )

    if not adapter_files:
        raise ValueError(
            "No student adapter artifacts were found "
            f"under {adapter_dir}"
        )

    # Production usage:
    #   no SHA supplied -> record current repository HEAD.
    #
    # Tests:
    #   inject a deterministic SHA.
    if git_commit_sha is None:
        proc = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        git_commit_sha = (
            proc.stdout.strip()
        )

    manifest = {
        "schema_version":
            ADAPTER_MANIFEST_SCHEMA,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "base_model_id":
            actual_base_model,

        "training_dataset": {
            "path":
                training_cfg.get(
                    "training_dataset_file"
                ),

            "sha256":
                sha256_file(
                    training_dataset_path
                ),
        },

        "teacher_manifest": {
            "path":
                training_cfg.get(
                    "teacher_manifest_file"
                ),

            "sha256":
                sha256_file(
                    teacher_manifest_path
                ),
        },

        "git_commit_sha":
            git_commit_sha,

        "adapter_files":
            adapter_files,
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest_path

def check_readiness(config: Dict[str, Any], root: Path = Path(".")) -> Dict[str, Any]:
    root = root.resolve()
    checks = {
        "config_contract": new_check(),
        "training_dataset": new_check(),
        "student_adapter": new_check(),
        "test_split": new_check(),
        "attacker_configuration": new_check(),
        "utility_configuration": new_check(),
    }
    provenance: Dict[str, Any] = {}

    # Existing config + new contract
    cr = validate_config(config)
    for e in cr.errors:
        block(checks["config_contract"], e)
    for w in cr.warnings:
        warn(checks["config_contract"], w)
    validate_new_contract(config, checks["config_contract"])

    training_cfg = config.get("training_provenance", {})
    student_cfg = config.get("student_provenance", {})
    split_cfg = config.get("split_provenance", {})

    # Teacher/training provenance
    tc = checks["training_dataset"]
    teacher_path = resolve(root, training_cfg.get("teacher_manifest_file", "__missing__"))
    train_data_path = resolve(root, training_cfg.get("training_dataset_file", "__missing__"))
    split_manifest_path = resolve(root, split_cfg.get("manifest_file", "__missing__"))

    teacher = None
    split_manifest = None

    if not teacher_path.exists():
        block(tc, f"Teacher manifest not found: {teacher_path}")
    else:
        teacher = load_json(teacher_path)

    if not train_data_path.exists():
        block(tc, f"Training dataset not found: {train_data_path}")

    if not split_manifest_path.exists():
        block(tc, f"Split manifest not found: {split_manifest_path}")
    else:
        split_manifest = load_json(split_manifest_path)

    if teacher is not None:
        if teacher.get("split") != "train":
            block(tc, "Teacher SFT manifest must use split='train'")

        if training_cfg.get("require_teacher_manifest_clean_git") and teacher.get("git_dirty") is not False:
            block(tc, "Teacher SFT manifest was not produced from a clean git tree")

        tinfo = teacher.get("teacher", {})
        tmodel = tinfo.get("model_id")
        trev = tinfo.get("revision")
        provenance["teacher_model_id"] = tmodel
        provenance["teacher_model_revision"] = trev

        if training_cfg.get("require_teacher_revision") and not trev:
            block(tc, "Teacher model revision is not pinned")

        attacker = str(config.get("attacker", {}).get("name", "")).lower()
        if training_cfg.get("forbid_attacker_as_teacher") and tmodel:
            tlower = str(tmodel).lower()
            if tlower == attacker or ("phi-4" in tlower and "phi-4" in attacker):
                block(tc, "Held-out attacker may not be used as the SFT teacher")

        if not teacher.get("nli", {}).get("revision"):
            warn(
                tc,
                "Teacher manifest does not pin the NLI model revision; exact teacher-selection replay is incomplete",
            )

    if teacher is not None and train_data_path.exists():
        actual = sha256_file(train_data_path)
        expected = teacher.get("outputs", {}).get("sft", {}).get("sha256")
        provenance["training_dataset_sha256"] = actual
        provenance["teacher_manifest_sha256"] = sha256_file(teacher_path)
        if training_cfg.get("require_training_dataset_hash_match") and actual != expected:
            block(tc, "Training dataset SHA-256 does not match teacher manifest")

    if teacher is not None and split_manifest is not None:
        input_info = teacher.get("input", {})
        train_split = resolve(root, input_info.get("path", "data/splits/train.jsonl"))
        if not train_split.exists():
            block(tc, f"Teacher input train split not found: {train_split}")
        else:
            actual = sha256_file(train_split)
            teacher_hash = input_info.get("sha256")
            split_hash = split_manifest.get("train", {}).get("sha256")
            provenance["train_split_sha256"] = actual
            if training_cfg.get("require_train_split_hash_match") and (
                actual != teacher_hash or actual != split_hash
            ):
                block(tc, "Train split hash does not agree across manifests")

    # Student adapter provenance
    sc = checks["student_adapter"]
    adapter_dir = resolve(root, student_cfg.get("adapter_dir", "__missing__"))
    adapter_manifest_path = resolve(
        root, student_cfg.get("adapter_manifest_file", "__missing__")
    )

    if not adapter_dir.exists():
        block(sc, f"Student adapter directory not found: {adapter_dir}")

    adapter_manifest = None
    if not adapter_manifest_path.exists():
        block(sc, f"Student adapter provenance manifest is missing: {adapter_manifest_path}")
    else:
        adapter_manifest = load_json(adapter_manifest_path)

    if adapter_manifest is not None:
        provenance["student_adapter_manifest_sha256"] = sha256_file(adapter_manifest_path)

        if adapter_manifest.get("schema_version") != ADAPTER_MANIFEST_SCHEMA:
            block(sc, "Student adapter manifest schema is not recognized")

        expected_base = student_cfg.get("active_base_model")
        actual_base = adapter_manifest.get("base_model_id")
        provenance["student_base_model_id"] = actual_base

        if student_cfg.get("require_base_model_match") and actual_base != expected_base:
            block(sc, f"Student adapter base model mismatch: {actual_base!r} != {expected_base!r}")

        if student_cfg.get("require_training_dataset_hash_match"):
            if adapter_manifest.get("training_dataset", {}).get("sha256") != provenance.get("training_dataset_sha256"):
                block(sc, "Student adapter training-dataset hash mismatch")

        if student_cfg.get("require_teacher_manifest_hash_match"):
            if adapter_manifest.get("teacher_manifest", {}).get("sha256") != provenance.get("teacher_manifest_sha256"):
                block(sc, "Student adapter teacher-manifest hash mismatch")

        files = adapter_manifest.get("adapter_files")
        if student_cfg.get("require_adapter_file_hashes") and not isinstance(files, dict):
            block(sc, "Student adapter manifest does not contain adapter file hashes")
        elif isinstance(files, dict):
            for rel, expected_hash in files.items():
                fp = adapter_dir / rel
                if not fp.exists():
                    block(sc, f"Registered adapter artifact is missing: {rel}")
                    continue
                if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
                    block(sc, f"Invalid SHA-256 in adapter manifest: {rel}")
                    continue
                if sha256_file(fp) != expected_hash:
                    block(sc, f"Adapter artifact hash mismatch: {rel}")

    # Test split
    testc = checks["test_split"]
    test_path = resolve(root, split_cfg.get("test_split_file", "__missing__"))

    if config.get("dataset", {}).get("split") != "test":
        block(testc, "Evaluation config is not pinned to the test split")

    if not test_path.exists():
        block(testc, f"Test split artifact not found: {test_path}")
    elif split_manifest is not None:
        actual = sha256_file(test_path)
        expected = split_manifest.get("test", {}).get("sha256")
        provenance["test_split_sha256"] = actual
        provenance["split_manifest_sha256"] = sha256_file(split_manifest_path)
        if actual != expected:
            block(testc, "Test split SHA-256 does not match split manifest")

    # Attacker
    ac = checks["attacker_configuration"]
    attacker_cfg = config.get("attacker", {})
    if not attacker_cfg.get("name"):
        block(ac, "Held-out attacker model is not configured")
    if attacker_cfg.get("required_mode") != "model":
        block(ac, "Held-out attacker must require mode='model'")
    provenance["attacker_model_id"] = attacker_cfg.get("name")

    # Utility
    uc = checks["utility_configuration"]
    utility_cfg = config.get("utility_evaluator", {})
    if not utility_cfg.get("nli_model"):
        block(uc, "NLI utility model is not configured")
    if utility_cfg.get("required_backend_for_nli_claims") != "model":
        block(uc, "Reportable NLI claims must require model backend")
    provenance["utility_nli_model_id"] = utility_cfg.get("nli_model")

    reasons: List[str] = []
    warnings: List[str] = []
    for name, check in checks.items():
        reasons.extend(f"{name}: {x}" for x in check["blocking_reasons"])
        warnings.extend(f"{name}: {x}" for x in check["warnings"])

    return {
        "schema_version": READINESS_SCHEMA,
        "overall_status": "READY" if not reasons else "BLOCKED",
        "checks": checks,
        "provenance": provenance,
        "blocking_reasons": reasons,
        "warnings": warnings,
    }


def print_readiness(r: Dict[str, Any]) -> None:
    print("\n=== SESSION 08 HELD-OUT EVALUATION READINESS ===")
    for name, check in r["checks"].items():
        print(f"{name.replace('_', ' ').title()}: {check['status']}")
    print("\nOVERALL:", r["overall_status"])

    if r["blocking_reasons"]:
        print("\nBlocking reasons:")
        for x in r["blocking_reasons"]:
            print("-", x)

    if r["warnings"]:
        print("\nWarnings:")
        for x in r["warnings"]:
            print("-", x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/session08.yaml")
    parser.add_argument("--output")
    parser.add_argument(
        "--register-adapter",
        action="store_true",
        help=(
            "Register an already-trained student adapter by writing "
            "student_provenance.adapter_manifest_file. "
            "Do not use this until real adapter files exist."
        ),
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    config = load_yaml(resolve(root, args.config))

    if args.register_adapter:
        manifest_path = register_adapter_manifest(
            config=config,
            root=root,
        )
        print(
            "Registered student adapter provenance:",
            manifest_path,
        )

    readiness = check_readiness(config, root)
    print_readiness(readiness)

    output = args.output or config.get("output", {}).get("readiness_file")
    if output:
        p = resolve(root, output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(readiness, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote readiness artifact: {p}")

    return 0 if readiness["overall_status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
