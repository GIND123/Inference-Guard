from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.check_release_evidence import (
    check_release_evidence,
)


WEB_RISK = "example/web-risk"
HELDOUT_RISK = "example/heldout-risk"
REWRITER = "example/rewriter"
ADVERSARY = "example/adversary"

WEB_REV = "a" * 40
HELDOUT_REV = "b" * 40
REWRITER_REV = "c" * 40


def write_json(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_yaml(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            value,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_runner(
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            [
                f'DEFAULT_RISK_MODEL_ID = "{HELDOUT_RISK}"',
                f'DEFAULT_REWRITER_MODEL_ID = "{REWRITER}"',
                f'DEFAULT_ADVERSARY_MODEL_ID = "{ADVERSARY}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_risk_eval(
    *,
    model_id: str,
    revision: str,
    usable: bool = True,
):
    return {
        "model_id":
            model_id,

        "revision":
            revision,

        "split":
            "val",

        "metrics": {
            "macro": {
                "f1": 0.5,
                "pr_auc": 0.5,
                "roc_auc": 0.8,
            }
        },

        "fitted_bands_in_sample": {
            attr: {
                "usable": usable,
                "degraded": not usable,
            }
            for attr in (
                "age",
                "location",
                "occupation",
                "education",
            )
        },
    }


def make_fixture(
    tmp_path: Path,
):
    compose = {
        "services": {
            "inference-guard-api": {
                "environment": [
                    (
                        "RISK_MODEL_ID="
                        "${RISK_MODEL_ID:-"
                        + WEB_RISK
                        + "}"
                    ),
                    (
                        "REWRITER_MODEL_ID="
                        "${REWRITER_MODEL_ID:-"
                        + REWRITER
                        + "}"
                    ),
                ]
            }
        }
    }

    write_yaml(
        tmp_path / "docker-compose.yml",
        compose,
    )

    write_runner(
        tmp_path / "src/evaluation/run.py",
    )

    manifest = {
        "schema_version": 1,

        "verified_at":
            "2026-11-17T00:00:00Z",

        "models": [
            {
                "role":
                    "risk_roberta",

                "repo_id":
                    WEB_RISK,

                "base_model":
                    "roberta-base",

                "revision":
                    WEB_REV,

                "expected_files": [
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                ],

                "checksums": {
                    "config.json":
                        "1" * 64,

                    "model.safetensors":
                        "2" * 64,

                    "tokenizer.json":
                        "3" * 64,
                },

                "verified":
                    True,
            },

            {
                "role":
                    "risk_modernbert",

                "repo_id":
                    HELDOUT_RISK,

                "base_model":
                    "answerdotai/ModernBERT-base",

                "revision":
                    HELDOUT_REV,

                "expected_files": [
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                ],

                "checksums": {
                    "config.json":
                        "4" * 64,

                    "model.safetensors":
                        "5" * 64,

                    "tokenizer.json":
                        "6" * 64,
                },

                "verified":
                    True,
            },

            {
                "role":
                    "rewriter",

                "repo_id":
                    REWRITER,

                "base_model":
                    "Qwen/Qwen2.5-1.5B-Instruct",

                "revision":
                    REWRITER_REV,

                "expected_files": [
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                ],

                "checksums": {
                    "config.json":
                        "7" * 64,

                    "model.safetensors":
                        "8" * 64,

                    "tokenizer.json":
                        "9" * 64,
                },

                "verified":
                    True,
            },
        ],
    }

    manifest_path = (
        tmp_path
        / "artifacts/model_manifest.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    web_eval_path = (
        tmp_path
        / "artifacts/risk_eval/web.json"
    )

    write_json(
        web_eval_path,
        make_risk_eval(
            model_id=WEB_RISK,
            revision=WEB_REV,
        ),
    )

    heldout_eval_path = (
        tmp_path
        / "artifacts/risk_eval/heldout.json"
    )

    write_json(
        heldout_eval_path,
        make_risk_eval(
            model_id=HELDOUT_RISK,
            revision=HELDOUT_REV,
        ),
    )

    rewriter_eval_path = (
        tmp_path
        / "artifacts/rewriter/eval.json"
    )

    write_json(
        rewriter_eval_path,
        {
            "model_id":
                REWRITER,

            "revision":
                REWRITER_REV,

            "reportable":
                True,
        },
    )

    heldout_result_path = (
        tmp_path
        / "artifacts/heldout/result.json"
    )

    write_json(
        heldout_result_path,
        {
            "attacker_model_id":
                ADVERSARY,

            "attacker_mode":
                "model",

            "reportable_as_held_out":
                True,
        },
    )

    config = {
        "schema_version":
            "session11_release_evidence_v1",

        "release_id":
            "test_release",

        "sources": {
            "model_manifest":
                "artifacts/model_manifest.json",

            "compose_file":
                "docker-compose.yml",

            "evaluation_runner":
                "src/evaluation/run.py",
        },

        "roles": {
            "web_risk": {
                "manifest_role":
                    "risk_roberta",

                "runtime": {
                    "source":
                        "compose",

                    "key":
                        "RISK_MODEL_ID",
                },

                "evaluation_artifact": {
                    "path":
                        "artifacts/risk_eval/web.json",
                },

                "require_manifest_revision":
                    True,

                "require_manifest_verified":
                    True,

                "require_manifest_checksums":
                    True,

                "require_evaluation_model_match":
                    True,

                "require_usable_fitted_bands":
                    True,
            },

            "heldout_risk": {
                "manifest_role":
                    "risk_modernbert",

                "runtime": {
                    "source":
                        "python_constant",

                    "constant":
                        "DEFAULT_RISK_MODEL_ID",
                },

                "evaluation_artifact": {
                    "path":
                        "artifacts/risk_eval/heldout.json",
                },

                "require_manifest_revision":
                    True,

                "require_manifest_verified":
                    True,

                "require_manifest_checksums":
                    True,

                "require_evaluation_model_match":
                    True,

                "require_usable_fitted_bands":
                    True,
            },

            "rewriter": {
                "manifest_role":
                    "rewriter",

                "runtime": {
                    "source":
                        "compose",

                    "key":
                        "REWRITER_MODEL_ID",
                },

                "evaluation_artifact": {
                    "path":
                        "artifacts/rewriter/eval.json",
                },

                "require_manifest_revision":
                    True,

                "require_manifest_verified":
                    True,

                "require_manifest_checksums":
                    True,

                "require_evaluation_model_match":
                    True,

                "require_reportable_evaluation":
                    True,
            },

            "heldout_adversary": {
                "expected_model_id":
                    ADVERSARY,

                "runtime": {
                    "source":
                        "python_constant",

                    "constant":
                        "DEFAULT_ADVERSARY_MODEL_ID",
                },

                "require_reportable_result":
                    True,

                "reportable_result_artifact":
                    "artifacts/heldout/result.json",
            },
        },

        "output": {
            "evidence_file":
                "artifacts/operations/evidence.json",
        },
    }

    return {
        "config":
            config,

        "manifest_path":
            manifest_path,

        "compose_path":
            tmp_path / "docker-compose.yml",

        "web_eval_path":
            web_eval_path,

        "heldout_eval_path":
            heldout_eval_path,

        "rewriter_eval_path":
            rewriter_eval_path,

        "heldout_result_path":
            heldout_result_path,
    }


def test_complete_release_evidence_is_ready(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["overall_status"]
        == "READY"
    )

    assert all(
        check["status"] == "PASS"
        for check
        in result["checks"].values()
    )


def test_runtime_model_drift_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    compose = yaml.safe_load(
        fx["compose_path"].read_text(
            encoding="utf-8"
        )
    )

    env = (
        compose["services"]
        ["inference-guard-api"]
        ["environment"]
    )

    env[0] = (
        "RISK_MODEL_ID="
        "${RISK_MODEL_ID:-example/wrong-risk}"
    )

    write_yaml(
        fx["compose_path"],
        compose,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "runtime_role_alignment"
        ]["status"]
        == "BLOCKED"
    )


def test_missing_manifest_role_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    manifest = json.loads(
        fx["manifest_path"].read_text(
            encoding="utf-8"
        )
    )

    manifest["models"] = [
        model
        for model in manifest["models"]
        if model["role"]
        != "risk_modernbert"
    ]

    write_json(
        fx["manifest_path"],
        manifest,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "manifest_contract"
        ]["status"]
        == "BLOCKED"
    )


def test_unverified_manifest_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    manifest = json.loads(
        fx["manifest_path"].read_text(
            encoding="utf-8"
        )
    )

    manifest["models"][0][
        "revision"
    ] = None

    manifest["models"][0][
        "verified"
    ] = False

    manifest["models"][0][
        "checksums"
    ] = {}

    write_json(
        fx["manifest_path"],
        manifest,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "manifest_verification"
        ]["status"]
        == "BLOCKED"
    )

    reasons = "\n".join(
        result["blocking_reasons"]
    )

    assert (
        "revision is not pinned"
        in reasons
    )

    assert (
        "manifest entry is not verified"
        in reasons
    )

    assert (
        "verified artifact checksums are missing"
        in reasons
    )


def test_wrong_evaluation_model_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    artifact = json.loads(
        fx["web_eval_path"].read_text(
            encoding="utf-8"
        )
    )

    artifact[
        "model_id"
    ] = "example/different-risk"

    write_json(
        fx["web_eval_path"],
        artifact,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "evaluation_registry"
        ]["status"]
        == "BLOCKED"
    )


def test_unusable_risk_bands_block(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    artifact = json.loads(
        fx[
            "heldout_eval_path"
        ].read_text(
            encoding="utf-8"
        )
    )

    artifact[
        "fitted_bands_in_sample"
    ][
        "occupation"
    ][
        "usable"
    ] = False

    artifact[
        "fitted_bands_in_sample"
    ][
        "occupation"
    ][
        "degraded"
    ] = True

    write_json(
        fx["heldout_eval_path"],
        artifact,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "risk_evidence_usability"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "occupation"
        in reason
        for reason
        in result["blocking_reasons"]
    )


def test_nonreportable_heldout_result_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    result_artifact = json.loads(
        fx[
            "heldout_result_path"
        ].read_text(
            encoding="utf-8"
        )
    )

    result_artifact[
        "attacker_mode"
    ] = "heuristic"

    result_artifact[
        "reportable_as_held_out"
    ] = False

    write_json(
        fx["heldout_result_path"],
        result_artifact,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "heldout_reportability"
        ]["status"]
        == "BLOCKED"
    )


def test_incomplete_manifest_checksums_block(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    manifest = json.loads(
        fx["manifest_path"].read_text(
            encoding="utf-8"
        )
    )

    manifest["models"][0][
        "checksums"
    ].pop(
        "model.safetensors"
    )

    write_json(
        fx["manifest_path"],
        manifest,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "manifest_verification"
        ]["status"]
        == "BLOCKED"
    )

    assert any(
        "model.safetensors"
        in reason
        for reason
        in result["blocking_reasons"]
    )


def test_nonreportable_rewriter_evaluation_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    artifact = json.loads(
        fx["rewriter_eval_path"].read_text(
            encoding="utf-8"
        )
    )

    artifact["reportable"] = False

    write_json(
        fx["rewriter_eval_path"],
        artifact,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "evaluation_registry"
        ]["status"]
        == "BLOCKED"
    )


def test_wrong_heldout_attacker_blocks(
    tmp_path,
):
    fx = make_fixture(tmp_path)

    artifact = json.loads(
        fx["heldout_result_path"].read_text(
            encoding="utf-8"
        )
    )

    artifact[
        "attacker_model_id"
    ] = "example/wrong-attacker"

    write_json(
        fx["heldout_result_path"],
        artifact,
    )

    result = check_release_evidence(
        fx["config"],
        root=tmp_path,
    )

    assert (
        result["checks"][
            "heldout_reportability"
        ]["status"]
        == "BLOCKED"
    )
