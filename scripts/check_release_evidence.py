from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


SCHEMA_VERSION = "session11_release_evidence_v1"

ENV_DEFAULT_RE = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-"
    r"(?P<default>.*)\}$"
)


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a JSON object"
        )

    return data


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping"
        )

    return data


def resolve(
    root: Path,
    value: str,
) -> Path:
    path = Path(value)

    return (
        path
        if path.is_absolute()
        else root / path
    )


def new_check() -> Dict[str, Any]:
    return {
        "status": "PASS",
        "blocking_reasons": [],
        "warnings": [],
    }


def block(
    check: Dict[str, Any],
    message: str,
) -> None:
    check["status"] = "BLOCKED"
    check["blocking_reasons"].append(
        message
    )


def warn(
    check: Dict[str, Any],
    message: str,
) -> None:
    check["warnings"].append(
        message
    )


def parse_compose_environment(
    compose: Dict[str, Any],
) -> Dict[str, str]:
    services = compose.get(
        "services",
        {},
    )

    if not isinstance(
        services,
        dict,
    ):
        return {}

    # Current production compose has one API service.
    service = services.get(
        "inference-guard-api",
        {},
    )

    environment = service.get(
        "environment",
        {},
    )

    result: Dict[str, str] = {}

    if isinstance(
        environment,
        list,
    ):
        for item in environment:
            if (
                not isinstance(item, str)
                or "=" not in item
            ):
                continue

            key, value = item.split(
                "=",
                1,
            )

            result[
                key.strip()
            ] = value.strip()

    elif isinstance(
        environment,
        dict,
    ):
        for key, value in (
            environment.items()
        ):
            result[str(key)] = (
                ""
                if value is None
                else str(value)
            )

    return result


def compose_default(
    environment: Dict[str, str],
    key: str,
) -> Optional[str]:
    value = environment.get(key)

    if not value:
        return None

    match = ENV_DEFAULT_RE.fullmatch(
        value
    )

    if match:
        return match.group(
            "default"
        )

    return value


def python_constant(
    path: Path,
    name: str,
) -> Optional[str]:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:
            if (
                isinstance(
                    target,
                    ast.Name,
                )
                and target.id == name
                and isinstance(
                    node.value,
                    ast.Constant,
                )
                and isinstance(
                    node.value.value,
                    str,
                )
            ):
                return node.value.value

    return None


def index_manifest(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    models = manifest.get(
        "models",
        [],
    )

    if not isinstance(
        models,
        list,
    ):
        return {}

    indexed: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for model in models:
        if not isinstance(
            model,
            dict,
        ):
            continue

        role = model.get(
            "role"
        )

        if isinstance(
            role,
            str,
        ):
            indexed[
                role
            ] = model

    return indexed


def resolve_runtime_model_id(
    role_cfg: Dict[str, Any],
    *,
    compose_environment: Dict[str, str],
    runner_path: Path,
) -> Optional[str]:
    runtime = role_cfg.get(
        "runtime",
        {},
    )

    source = runtime.get(
        "source"
    )

    if source == "compose":
        key = runtime.get(
            "key"
        )

        if not isinstance(
            key,
            str,
        ):
            return None

        return compose_default(
            compose_environment,
            key,
        )

    if source == "python_constant":
        constant = runtime.get(
            "constant"
        )

        if not isinstance(
            constant,
            str,
        ):
            return None

        return python_constant(
            runner_path,
            constant,
        )

    return None


def check_release_evidence(
    config: Dict[str, Any],
    root: Path = Path("."),
) -> Dict[str, Any]:

    root = root.resolve()

    checks = {
        "manifest_contract":
            new_check(),

        "runtime_role_alignment":
            new_check(),

        "manifest_verification":
            new_check(),

        "evaluation_registry":
            new_check(),

        "risk_evidence_usability":
            new_check(),

        "heldout_reportability":
            new_check(),
    }

    provenance: Dict[
        str,
        Any,
    ] = {}

    sources = config.get(
        "sources",
        {},
    )

    roles = config.get(
        "roles",
        {},
    )

    # -----------------------------------------------------
    # Load manifest / runtime sources
    # -----------------------------------------------------

    manifest_path = resolve(
        root,
        sources.get(
            "model_manifest",
            "__missing__",
        ),
    )

    compose_path = resolve(
        root,
        sources.get(
            "compose_file",
            "__missing__",
        ),
    )

    runner_path = resolve(
        root,
        sources.get(
            "evaluation_runner",
            "__missing__",
        ),
    )

    manifest = None

    if not manifest_path.exists():
        block(
            checks[
                "manifest_contract"
            ],
            "Model manifest is missing.",
        )

        manifest_index = {}

    else:
        manifest = load_json(
            manifest_path
        )

        manifest_index = (
            index_manifest(
                manifest
            )
        )

    if not compose_path.exists():
        block(
            checks[
                "runtime_role_alignment"
            ],
            "Docker compose file is missing.",
        )

        compose_environment = {}

    else:
        compose = load_yaml(
            compose_path
        )

        compose_environment = (
            parse_compose_environment(
                compose
            )
        )

    if not runner_path.exists():
        block(
            checks[
                "runtime_role_alignment"
            ],
            "Evaluation runner is missing.",
        )

    # -----------------------------------------------------
    # Manifest contract
    # -----------------------------------------------------

    hosted_roles = {
        name: role_cfg
        for name, role_cfg
        in roles.items()
        if isinstance(
            role_cfg,
            dict,
        )
        and role_cfg.get(
            "manifest_role"
        )
    }

    required_manifest_roles = [
        role_cfg[
            "manifest_role"
        ]
        for role_cfg
        in hosted_roles.values()
    ]

    if (
        len(
            required_manifest_roles
        )
        != len(
            set(
                required_manifest_roles
            )
        )
    ):
        block(
            checks[
                "manifest_contract"
            ],
            "Multiple runtime roles map to "
            "the same manifest role.",
        )

    for role_name, role_cfg in (
        hosted_roles.items()
    ):
        manifest_role = (
            role_cfg[
                "manifest_role"
            ]
        )

        if (
            manifest_role
            not in manifest_index
        ):
            block(
                checks[
                    "manifest_contract"
                ],
                f"{role_name}: manifest role "
                f"{manifest_role!r} is missing.",
            )

    # -----------------------------------------------------
    # Runtime role -> manifest alignment
    # -----------------------------------------------------

    for role_name, role_cfg in (
        roles.items()
    ):
        if not isinstance(
            role_cfg,
            dict,
        ):
            continue

        runtime_id = (
            resolve_runtime_model_id(
                role_cfg,
                compose_environment=
                    compose_environment,
                runner_path=
                    runner_path,
            )
        )

        provenance[
            f"{role_name}_runtime_model_id"
        ] = runtime_id

        if role_name == (
            "heldout_adversary"
        ):
            expected = (
                role_cfg.get(
                    "expected_model_id"
                )
            )

            if runtime_id != expected:
                block(
                    checks[
                        "runtime_role_alignment"
                    ],
                    f"{role_name}: runtime model "
                    f"{runtime_id!r} does not match "
                    f"{expected!r}.",
                )

            continue

        manifest_role = (
            role_cfg.get(
                "manifest_role"
            )
        )

        manifest_entry = (
            manifest_index.get(
                manifest_role,
                {},
            )
        )

        manifest_repo = (
            manifest_entry.get(
                "repo_id"
            )
        )

        provenance[
            f"{role_name}_manifest_model_id"
        ] = manifest_repo

        if runtime_id != manifest_repo:
            block(
                checks[
                    "runtime_role_alignment"
                ],
                f"{role_name}: runtime model "
                f"{runtime_id!r} does not match "
                f"manifest model {manifest_repo!r}.",
            )

    # -----------------------------------------------------
    # Manifest revision / checksum verification
    # -----------------------------------------------------

    for role_name, role_cfg in (
        hosted_roles.items()
    ):
        manifest_entry = (
            manifest_index.get(
                role_cfg[
                    "manifest_role"
                ],
                {},
            )
        )

        revision = (
            manifest_entry.get(
                "revision"
            )
        )

        verified = (
            manifest_entry.get(
                "verified"
            )
        )

        checksums = (
            manifest_entry.get(
                "checksums"
            )
        )

        provenance[
            f"{role_name}_manifest_revision"
        ] = revision

        provenance[
            f"{role_name}_manifest_verified"
        ] = verified

        if (
            role_cfg.get(
                "require_manifest_revision"
            )
            and not revision
        ):
            block(
                checks[
                    "manifest_verification"
                ],
                f"{role_name}: model revision "
                "is not pinned.",
            )

        if (
            role_cfg.get(
                "require_manifest_verified"
            )
            and verified is not True
        ):
            block(
                checks[
                    "manifest_verification"
                ],
                f"{role_name}: manifest entry "
                "is not verified.",
            )

        if role_cfg.get(
            "require_manifest_checksums"
        ):
            if (
                not isinstance(
                    checksums,
                    dict,
                )
                or not checksums
            ):
                block(
                    checks[
                        "manifest_verification"
                    ],
                    f"{role_name}: verified "
                    "artifact checksums are missing.",
                )

            else:
                expected_files = (
                    manifest_entry.get(
                        "expected_files",
                        [],
                    )
                )

                missing_checksums = sorted(
                    name
                    for name
                    in expected_files
                    if name not in checksums
                )

                if missing_checksums:
                    block(
                        checks[
                            "manifest_verification"
                        ],
                        f"{role_name}: checksums are "
                        "missing for expected files: "
                        + ", ".join(
                            missing_checksums
                        ),
                    )

                invalid_checksums = sorted(
                    name
                    for name, value
                    in checksums.items()
                    if not isinstance(
                        value,
                        str,
                    )
                    or re.fullmatch(
                        r"[0-9a-fA-F]{64}",
                        value,
                    )
                    is None
                )

                if invalid_checksums:
                    block(
                        checks[
                            "manifest_verification"
                        ],
                        f"{role_name}: invalid SHA-256 "
                        "checksums for: "
                        + ", ".join(
                            invalid_checksums
                        ),
                    )

    # -----------------------------------------------------
    # Evaluation evidence registry
    # -----------------------------------------------------

    risk_eval_artifacts: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for role_name, role_cfg in (
        hosted_roles.items()
    ):
        evaluation_cfg = (
            role_cfg.get(
                "evaluation_artifact",
                {},
            )
        )

        evaluation_path_value = (
            evaluation_cfg.get(
                "path"
            )
            if isinstance(
                evaluation_cfg,
                dict,
            )
            else None
        )

        if not evaluation_path_value:
            block(
                checks[
                    "evaluation_registry"
                ],
                f"{role_name}: no evaluation "
                "artifact is registered.",
            )

            continue

        evaluation_path = resolve(
            root,
            evaluation_path_value,
        )

        if not evaluation_path.exists():
            block(
                checks[
                    "evaluation_registry"
                ],
                f"{role_name}: evaluation "
                f"artifact is missing: "
                f"{evaluation_path_value}",
            )

            continue

        evaluation = load_json(
            evaluation_path
        )

        manifest_entry = (
            manifest_index.get(
                role_cfg[
                    "manifest_role"
                ],
                {},
            )
        )

        expected_model = (
            manifest_entry.get(
                "repo_id"
            )
        )

        evidence_model = (
            evaluation.get(
                "model_id"
            )
        )

        evidence_revision = (
            evaluation.get(
                "revision"
            )
        )

        provenance[
            f"{role_name}_evaluation_model_id"
        ] = evidence_model

        provenance[
            f"{role_name}_evaluation_revision"
        ] = evidence_revision

        if (
            role_cfg.get(
                "require_evaluation_model_match"
            )
            and evidence_model
            != expected_model
        ):
            block(
                checks[
                    "evaluation_registry"
                ],
                f"{role_name}: evaluation "
                "artifact belongs to "
                f"{evidence_model!r}, expected "
                f"{expected_model!r}.",
            )

        manifest_revision = (
            manifest_entry.get(
                "revision"
            )
        )

        if (
            manifest_revision
            and evidence_revision
            and manifest_revision
            != evidence_revision
        ):
            block(
                checks[
                    "evaluation_registry"
                ],
                f"{role_name}: evaluation "
                "revision does not match "
                "manifest revision.",
            )

        if (
            role_name == "rewriter"
            and role_cfg.get(
                "require_reportable_evaluation"
            )
            and evaluation.get(
                "reportable"
            )
            is not True
        ):
            block(
                checks[
                    "evaluation_registry"
                ],
                "rewriter: registered evaluation "
                "artifact is not marked reportable.",
            )

        if (
            role_name
            in (
                "web_risk",
                "heldout_risk",
            )
        ):
            risk_eval_artifacts[
                role_name
            ] = evaluation

    # -----------------------------------------------------
    # Risk evidence usability
    # -----------------------------------------------------

    for role_name in (
        "web_risk",
        "heldout_risk",
    ):
        role_cfg = roles.get(
            role_name,
            {},
        )

        if not role_cfg.get(
            "require_usable_fitted_bands"
        ):
            continue

        evaluation = (
            risk_eval_artifacts.get(
                role_name
            )
        )

        if evaluation is None:
            block(
                checks[
                    "risk_evidence_usability"
                ],
                f"{role_name}: risk evidence "
                "usability cannot be verified "
                "without its evaluation artifact.",
            )

            continue

        fitted = evaluation.get(
            "fitted_bands_in_sample"
        )

        if not isinstance(
            fitted,
            dict,
        ):
            block(
                checks[
                    "risk_evidence_usability"
                ],
                f"{role_name}: evaluation "
                "artifact has no fitted-band "
                "usability evidence.",
            )

            continue

        unusable = sorted(
            attribute
            for attribute, item
            in fitted.items()
            if (
                not isinstance(
                    item,
                    dict,
                )
                or item.get(
                    "usable"
                )
                is not True
            )
        )

        provenance[
            f"{role_name}_unusable_bands"
        ] = unusable

        if unusable:
            block(
                checks[
                    "risk_evidence_usability"
                ],
                f"{role_name}: risk bands are "
                "not usable for: "
                + ", ".join(
                    unusable
                ),
            )

    # -----------------------------------------------------
    # Held-out reportability
    # -----------------------------------------------------

    adversary_cfg = roles.get(
        "heldout_adversary",
        {},
    )

    result_value = (
        adversary_cfg.get(
            "reportable_result_artifact"
        )
    )

    if (
        adversary_cfg.get(
            "require_reportable_result"
        )
    ):
        if not result_value:
            block(
                checks[
                    "heldout_reportability"
                ],
                "No reportable held-out "
                "result is registered.",
            )

        else:
            result_path = resolve(
                root,
                result_value,
            )

            if not result_path.exists():
                block(
                    checks[
                        "heldout_reportability"
                    ],
                    "Registered held-out result "
                    "artifact is missing.",
                )

            else:
                result = load_json(
                    result_path
                )

                expected_attacker = (
                    adversary_cfg.get(
                        "expected_model_id"
                    )
                )

                result_attacker = (
                    result.get(
                        "attacker_model_id"
                    )
                )

                result_mode = (
                    result.get(
                        "attacker_mode"
                    )
                )

                if (
                    result_attacker
                    != expected_attacker
                ):
                    block(
                        checks[
                            "heldout_reportability"
                        ],
                        "Held-out result belongs to "
                        f"{result_attacker!r}, expected "
                        f"{expected_attacker!r}.",
                    )

                if result_mode != "model":
                    block(
                        checks[
                            "heldout_reportability"
                        ],
                        "Held-out result was not "
                        "produced by the real model "
                        f"attacker: mode={result_mode!r}.",
                    )

                if (
                    result.get(
                        "reportable_as_held_out"
                    )
                    is not True
                ):
                    block(
                        checks[
                            "heldout_reportability"
                        ],
                        "Held-out result is not "
                        "marked reportable.",
                    )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    blocking_reasons: List[
        str
    ] = []

    warnings: List[
        str
    ] = []

    for name, check in (
        checks.items()
    ):
        blocking_reasons.extend(
            f"{name}: {reason}"
            for reason
            in check[
                "blocking_reasons"
            ]
        )

        warnings.extend(
            f"{name}: {message}"
            for message
            in check[
                "warnings"
            ]
        )

    return {
        "schema_version":
            SCHEMA_VERSION,

        "release_id":
            config.get(
                "release_id"
            ),

        "overall_status":
            (
                "READY"
                if not blocking_reasons
                else "BLOCKED"
            ),

        "checks":
            checks,

        "provenance":
            provenance,

        "blocking_reasons":
            blocking_reasons,

        "warnings":
            warnings,
    }


def print_result(
    result: Dict[str, Any],
) -> None:

    print(
        "\n=== SESSION 11 RELEASE EVIDENCE ==="
    )

    for name, check in (
        result["checks"].items()
    ):
        label = (
            name.replace(
                "_",
                " ",
            ).title()
        )

        print(
            f"{label}: "
            f"{check['status']}"
        )

    print(
        "\nOVERALL:",
        result["overall_status"],
    )

    if result[
        "blocking_reasons"
    ]:
        print(
            "\nBlocking reasons:"
        )

        for reason in result[
            "blocking_reasons"
        ]:
            print(
                "-",
                reason,
            )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/operations/"
            "session11_release.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    root = Path(".").resolve()

    config = load_yaml(
        resolve(
            root,
            args.config,
        )
    )

    result = check_release_evidence(
        config,
        root=root,
    )

    print_result(
        result
    )

    output_value = (
        args.output
        or config.get(
            "output",
            {},
        ).get(
            "evidence_file"
        )
    )

    if output_value:
        output_path = resolve(
            root,
            output_value,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "\nWrote release evidence:",
            output_path,
        )

    return (
        0
        if result[
            "overall_status"
        ]
        == "READY"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
