"""Validation and promotion manifest helpers for DESC joint banana runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from banana_opt.desc_joint_result_schema import (
    DESC_JOINT_RESULT_SCHEMA_VERSION,
    validate_desc_joint_result_payload,
)

DESC_JOINT_VALIDATION_MANIFEST_SCHEMA_VERSION = "desc_joint_validation_manifest_v1"
DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION = (
    "desc_joint_final_oracle_evidence_v1"
)
DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION = (
    "desc_joint_simsopt_physics_validation_v1"
)
_FINAL_ORACLE_SOURCE = "direct_loaded_artifact_hardware_contact_oracle"
_SIMSOPT_PHYSICS_VALIDATION_SOURCE = "simsopt_boozer_poincare_sidecars"
_SHA256_HEXDIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def build_desc_joint_validation_manifest(
    *,
    result_payload: Mapping[str, object],
    exported_artifact_paths: Sequence[str],
    expected_source_artifact_checksums: Mapping[str, str] | None = None,
    physics_validation_passed: bool | None,
    artifact_hardware_passed: bool | None,
    search_hardware_passed: bool | None,
    final_oracle_passed: bool,
    final_oracle_evidence_path: str | None,
    physics_validation_evidence_paths: Sequence[str] = (),
) -> dict[str, object]:
    validate_desc_joint_result_payload(result_payload)
    fixed_polish_predecessor_status = _fixed_polish_predecessor_status(
        result_payload,
        expected_source_artifact_checksums=expected_source_artifact_checksums,
    )
    lane_b_predecessor_status = _lane_b_predecessor_status(
        result_payload,
        expected_source_artifact_checksums=expected_source_artifact_checksums,
    )
    promotion_status = resolve_desc_joint_promotion_status(
        run_mode=result_payload["run_mode"],
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=expected_source_artifact_checksums,
        desc_solve_passed=_desc_solve_passed(result_payload),
        fixed_polish_predecessor_passed=(
            fixed_polish_predecessor_status["passed"] is True
        ),
        lane_b_predecessor_passed=lane_b_predecessor_status["passed"] is True,
        physics_validation_passed=physics_validation_passed,
        artifact_hardware_passed=artifact_hardware_passed,
        final_oracle_passed=final_oracle_passed,
        final_oracle_evidence_path=final_oracle_evidence_path,
    )
    manifest = {
        "schema_version": DESC_JOINT_VALIDATION_MANIFEST_SCHEMA_VERSION,
        "result_schema_version": DESC_JOINT_RESULT_SCHEMA_VERSION,
        "run_mode": result_payload["run_mode"],
        "exported_artifact_paths": _coerce_artifact_paths(exported_artifact_paths),
        "source_artifact_checksums": _optional_source_checksum_map(
            expected_source_artifact_checksums,
            field_name="expected_source_artifact_checksums",
        ),
        "desc_solve_status": dict(result_payload["desc_solve_status"]),
        "search_hardware_status": {
            "passed": search_hardware_passed,
            "source": "search_time_steering",
        },
        "artifact_hardware_status": {
            "passed": artifact_hardware_passed,
            "source": "exported_loaded_artifact",
        },
        "fixed_polish_predecessor_status": fixed_polish_predecessor_status,
        "lane_b_predecessor_status": lane_b_predecessor_status,
        "physics_validation_status": {
            "passed": physics_validation_passed,
            "source": "simsopt_boozer_poincare",
            "evidence_paths": _coerce_artifact_paths(
                physics_validation_evidence_paths,
            ),
        },
        "final_oracle_status": {
            "passed": final_oracle_passed,
            "evidence_path": final_oracle_evidence_path,
            "source": "direct_loaded_artifact_hardware_contact_oracle",
        },
        "promotion_status": promotion_status,
    }
    validate_desc_joint_validation_manifest(manifest)
    return manifest


def resolve_desc_joint_promotion_status(
    *,
    run_mode: object,
    exported_artifact_paths: Sequence[str],
    expected_source_artifact_checksums: Mapping[str, str] | None,
    desc_solve_passed: bool,
    fixed_polish_predecessor_passed: bool,
    lane_b_predecessor_passed: bool,
    physics_validation_passed: bool | None,
    artifact_hardware_passed: bool | None,
    final_oracle_passed: bool,
    final_oracle_evidence_path: str | None,
) -> dict[str, object]:
    if not desc_solve_passed:
        return {
            "state": "blocked",
            "reason": "DESC optimization has not passed",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if physics_validation_passed is False:
        return {
            "state": "failed",
            "reason": "SIMSOPT physics validation did not pass",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if not fixed_polish_predecessor_passed:
        return {
            "state": "blocked",
            "reason": (
                "fixed-equilibrium polish predecessor validation has not passed"
            ),
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if run_mode == "finite_beta_joint" and not lane_b_predecessor_passed:
        return {
            "state": "blocked",
            "reason": "Lane B vacuum-joint predecessor validation has not passed",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if physics_validation_passed is None:
        return {
            "state": "blocked",
            "reason": "SIMSOPT physics validation has not run",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if artifact_hardware_passed is None:
        return {
            "state": "blocked",
            "reason": "exported artifact hardware validation has not run",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if not artifact_hardware_passed:
        return {
            "state": "failed",
            "reason": "exported artifact did not pass hardware constraints",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if not final_oracle_evidence_path:
        return {
            "state": "blocked",
            "reason": "direct hardware/contact oracle evidence is required",
            "final_oracle_evidence_path": None,
        }
    oracle_path = Path(final_oracle_evidence_path).expanduser()
    if not oracle_path.is_file():
        return {
            "state": "blocked",
            "reason": "direct hardware/contact oracle evidence path is missing",
            "final_oracle_evidence_path": final_oracle_evidence_path,
        }
    if not final_oracle_passed:
        return {
            "state": "failed",
            "reason": "direct hardware/contact oracle did not pass",
            "final_oracle_evidence_path": str(oracle_path.resolve()),
        }
    if expected_source_artifact_checksums is None:
        raise ValueError(
            "expected_source_artifact_checksums is required when final oracle "
            "status is passed."
        )
    validate_desc_joint_final_oracle_evidence(
        str(oracle_path),
        expected_source_artifact_checksums=expected_source_artifact_checksums,
        expected_exported_artifact_paths=exported_artifact_paths,
    )
    return {
        "state": "passed",
        "reason": "physics validation, artifact hardware checks, and final oracle passed",
        "final_oracle_evidence_path": str(oracle_path.resolve()),
    }


def validate_desc_joint_validation_manifest(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != DESC_JOINT_VALIDATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unexpected DESC joint validation manifest schema_version.")
    for section in (
        "desc_solve_status",
        "search_hardware_status",
        "artifact_hardware_status",
        "physics_validation_status",
        "final_oracle_status",
        "promotion_status",
    ):
        if section not in payload:
            raise ValueError(
                f"DESC joint validation manifest missing section {section!r}."
            )
    final_oracle_status = payload["final_oracle_status"]
    if not isinstance(final_oracle_status, Mapping):
        raise ValueError("final_oracle_status must be an object.")
    promotion_status = payload["promotion_status"]
    if not isinstance(promotion_status, Mapping):
        raise ValueError("promotion_status must be an object.")
    source_artifact_checksums = _optional_source_checksum_map(
        payload.get("source_artifact_checksums"),
        field_name="source_artifact_checksums",
    )
    _validate_validation_status_section(
        "search_hardware_status",
        payload["search_hardware_status"],
        allow_not_run=True,
    )
    _validate_validation_status_section(
        "artifact_hardware_status",
        payload["artifact_hardware_status"],
        allow_not_run=True,
    )
    _validate_validation_status_section(
        "physics_validation_status",
        payload["physics_validation_status"],
        allow_not_run=True,
    )
    _validate_validation_status_section(
        "final_oracle_status",
        final_oracle_status,
        allow_not_run=False,
    )
    if final_oracle_status.get("passed") is True:
        if source_artifact_checksums is None:
            raise ValueError(
                "source_artifact_checksums is required when "
                "final_oracle_status.passed is true."
            )
        _require_passed_final_oracle_evidence(
            final_oracle_status.get("evidence_path"),
            field_name="final_oracle_status.evidence_path",
            expected_source_artifact_checksums=source_artifact_checksums,
            exported_artifact_paths=_coerce_artifact_paths(
                payload.get("exported_artifact_paths", [])
            ),
        )
    if promotion_status.get("state") == "passed":
        if source_artifact_checksums is None:
            raise ValueError(
                "source_artifact_checksums is required when promotion_status "
                "passes."
            )
        desc_solve_status = payload["desc_solve_status"]
        if not isinstance(desc_solve_status, Mapping):
            raise ValueError("desc_solve_status must be an object.")
        if desc_solve_status.get("state") != "passed":
            raise ValueError(
                "promotion_status cannot pass unless desc_solve_status.state "
                "is 'passed'."
            )
        if payload.get("run_mode") in {"vacuum_joint", "finite_beta_joint"}:
            fixed_polish_predecessor_status = payload.get(
                "fixed_polish_predecessor_status"
            )
            if (
                not isinstance(fixed_polish_predecessor_status, Mapping)
                or fixed_polish_predecessor_status.get("passed") is not True
            ):
                raise ValueError(
                    "promotion_status cannot pass unless "
                    "fixed_polish_predecessor_status.passed is true."
                )
            _validate_fixed_polish_predecessor_status(
                fixed_polish_predecessor_status,
                expected_source_artifact_checksums=source_artifact_checksums,
            )
        if payload.get("run_mode") == "finite_beta_joint":
            lane_b_predecessor_status = payload.get("lane_b_predecessor_status")
            if (
                not isinstance(lane_b_predecessor_status, Mapping)
                or lane_b_predecessor_status.get("passed") is not True
            ):
                raise ValueError(
                    "promotion_status cannot pass unless "
                    "lane_b_predecessor_status.passed is true."
                )
            _validate_lane_b_predecessor_status(
                lane_b_predecessor_status,
                expected_source_artifact_checksums=source_artifact_checksums,
            )
        if payload["physics_validation_status"].get("passed") is not True:
            raise ValueError(
                "promotion_status cannot pass unless physics_validation_status.passed "
                "is true."
            )
        if payload["artifact_hardware_status"].get("passed") is not True:
            raise ValueError(
                "promotion_status cannot pass unless artifact_hardware_status.passed "
                "is true."
            )
        promotion_oracle_path = _require_passed_final_oracle_evidence(
            promotion_status.get("final_oracle_evidence_path"),
            field_name="promotion_status.final_oracle_evidence_path",
            expected_source_artifact_checksums=source_artifact_checksums,
            exported_artifact_paths=_coerce_artifact_paths(
                payload.get("exported_artifact_paths", [])
            ),
        )
        if final_oracle_status.get("passed") is not True:
            raise ValueError(
                "promotion_status cannot pass unless final_oracle_status.passed "
                "is true."
            )
        final_oracle_path = _require_passed_final_oracle_evidence(
            final_oracle_status.get("evidence_path"),
            field_name="final_oracle_status.evidence_path",
            expected_source_artifact_checksums=source_artifact_checksums,
            exported_artifact_paths=_coerce_artifact_paths(
                payload.get("exported_artifact_paths", [])
            ),
        )
        if promotion_oracle_path.resolve() != final_oracle_path.resolve():
            raise ValueError(
                "promotion_status final_oracle_evidence_path must match "
                "final_oracle_status evidence_path."
            )


def render_desc_joint_validation_report(payload: Mapping[str, object]) -> str:
    validate_desc_joint_validation_manifest(payload)
    promotion_status = payload["promotion_status"]
    assert isinstance(promotion_status, Mapping)
    lines = [
        "# DESC Joint Validation Report",
        "",
        f"- run_mode: {payload['run_mode']}",
        f"- desc_solve_status: {payload['desc_solve_status']}",
        f"- search_hardware_status: {payload['search_hardware_status']}",
        f"- artifact_hardware_status: {payload['artifact_hardware_status']}",
        f"- fixed_polish_predecessor_status: {payload.get('fixed_polish_predecessor_status')}",
        f"- lane_b_predecessor_status: {payload.get('lane_b_predecessor_status')}",
        f"- physics_validation_status: {payload['physics_validation_status']}",
        f"- final_oracle_status: {payload['final_oracle_status']}",
        f"- exported_artifact_paths: {payload['exported_artifact_paths']}",
        f"- promotion_status: {promotion_status['state']} - {promotion_status['reason']}",
    ]
    return "\n".join(lines) + "\n"


def _coerce_artifact_paths(paths: Sequence[str]) -> list[str]:
    if isinstance(paths, str):
        raise ValueError("exported_artifact_paths must be a list of strings.")
    coerced: list[str] = []
    for path in paths:
        if not isinstance(path, str) or path == "":
            raise ValueError("exported_artifact_paths must contain nonempty strings.")
        coerced.append(path)
    return coerced


def _resolved_path_tuple(paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(Path(path).expanduser().resolve()) for path in paths)


def _desc_solve_passed(result_payload: Mapping[str, object]) -> bool:
    desc_solve_status = result_payload["desc_solve_status"]
    if not isinstance(desc_solve_status, Mapping):
        raise ValueError("desc_solve_status must be an object.")
    return desc_solve_status.get("state") == "passed"


def _fixed_polish_predecessor_status(
    result_payload: Mapping[str, object],
    *,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> dict[str, object]:
    run_mode = result_payload["run_mode"]
    if run_mode == "fixed_equilibrium_polish":
        return {
            "passed": True,
            "reason": "fixed-equilibrium polish is the active predecessor lane",
            "artifact_paths": [],
        }
    raw_status = result_payload.get("fixed_polish_predecessor_status")
    if not isinstance(raw_status, Mapping):
        return {
            "passed": False,
            "reason": (
                "joint promotion requires fixed-polish predecessor validation "
                "status"
            ),
            "artifact_paths": [],
        }
    artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            raw_status.get("artifact_paths", []),
            field_name="fixed_polish_predecessor_status.artifact_paths",
        )
    )
    reason = raw_status.get("reason")
    if not isinstance(reason, str) or reason == "":
        reason = "fixed-polish predecessor status did not include a reason"
    if raw_status.get("state") != "passed":
        return {
            "passed": False,
            "reason": reason,
            "artifact_paths": artifact_paths,
        }
    if not artifact_paths:
        return {
            "passed": False,
            "reason": (
                "fixed-polish predecessor status passed without evidence paths"
            ),
            "artifact_paths": artifact_paths,
        }
    try:
        validated_paths = _validate_fixed_polish_predecessor_status(
            raw_status,
            expected_source_artifact_checksums=expected_source_artifact_checksums,
        )
    except ValueError as exc:
        return {
            "passed": False,
            "reason": f"invalid fixed-polish predecessor evidence: {exc}",
            "artifact_paths": artifact_paths,
        }
    return {
        "passed": True,
        "reason": reason,
        "artifact_paths": [str(path) for path in validated_paths],
    }


def _lane_b_predecessor_status(
    result_payload: Mapping[str, object],
    *,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> dict[str, object]:
    run_mode = result_payload["run_mode"]
    if run_mode != "finite_beta_joint":
        return {
            "passed": True,
            "reason": "Lane B predecessor is not required for this run mode",
            "artifact_paths": [],
        }
    raw_status = result_payload.get("lane_b_predecessor_status")
    if not isinstance(raw_status, Mapping):
        return {
            "passed": False,
            "reason": (
                "finite-beta joint promotion requires Lane B vacuum-joint "
                "predecessor validation status"
            ),
            "artifact_paths": [],
        }
    artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            raw_status.get("artifact_paths", []),
            field_name="lane_b_predecessor_status.artifact_paths",
        )
    )
    reason = raw_status.get("reason")
    if not isinstance(reason, str) or reason == "":
        reason = "Lane B predecessor status did not include a reason"
    if raw_status.get("state") != "passed":
        return {
            "passed": False,
            "reason": reason,
            "artifact_paths": artifact_paths,
        }
    if not artifact_paths:
        return {
            "passed": False,
            "reason": "Lane B predecessor status passed without evidence paths",
            "artifact_paths": artifact_paths,
        }
    try:
        validated_paths = _validate_lane_b_predecessor_status(
            raw_status,
            expected_source_artifact_checksums=expected_source_artifact_checksums,
        )
    except ValueError as exc:
        return {
            "passed": False,
            "reason": f"invalid Lane B predecessor evidence: {exc}",
            "artifact_paths": artifact_paths,
        }
    return {
        "passed": True,
        "reason": reason,
        "artifact_paths": [str(path) for path in validated_paths],
    }


def _validate_fixed_polish_predecessor_status(
    raw_status: Mapping[str, object],
    *,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> tuple[Path, ...]:
    artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            raw_status.get("artifact_paths", []),
            field_name="fixed_polish_predecessor_status.artifact_paths",
        )
    )
    if (
        raw_status.get("state") != "passed"
        and raw_status.get("passed") is not True
    ):
        raise ValueError(
            "fixed_polish_predecessor_status must record state='passed' or "
            "passed=true."
        )
    if not artifact_paths:
        raise ValueError(
            "fixed_polish_predecessor_status.artifact_paths must include a "
            "fixed-polish validation manifest."
        )
    validated_paths: list[Path] = []
    for artifact_path in artifact_paths:
        path = Path(artifact_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"fixed-polish predecessor evidence path is missing: {artifact_path}."
            )
        _validate_fixed_polish_predecessor_manifest(
            path,
            expected_source_artifact_checksums=expected_source_artifact_checksums,
        )
        validated_paths.append(path)
    return tuple(validated_paths)


def _validate_lane_b_predecessor_status(
    raw_status: Mapping[str, object],
    *,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> tuple[Path, ...]:
    artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            raw_status.get("artifact_paths", []),
            field_name="lane_b_predecessor_status.artifact_paths",
        )
    )
    if (
        raw_status.get("state") != "passed"
        and raw_status.get("passed") is not True
    ):
        raise ValueError(
            "lane_b_predecessor_status must record state='passed' or "
            "passed=true."
        )
    if not artifact_paths:
        raise ValueError(
            "lane_b_predecessor_status.artifact_paths must include a "
            "vacuum_joint validation manifest."
        )
    validated_paths: list[Path] = []
    for artifact_path in artifact_paths:
        path = Path(artifact_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"Lane B predecessor evidence path is missing: {artifact_path}."
            )
        _validate_lane_b_predecessor_manifest(
            path,
            expected_source_artifact_checksums=expected_source_artifact_checksums,
        )
        validated_paths.append(path)
    return tuple(validated_paths)


def _validate_fixed_polish_predecessor_manifest(
    path: Path,
    *,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> None:
    _validate_predecessor_manifest(
        path,
        expected_run_mode="fixed_equilibrium_polish",
        predecessor_label="fixed-polish predecessor",
        status_field_name="fixed_polish_predecessor_status",
        expected_source_artifact_checksums=expected_source_artifact_checksums,
    )


def _validate_lane_b_predecessor_manifest(
    path: Path,
    *,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> None:
    _validate_predecessor_manifest(
        path,
        expected_run_mode="vacuum_joint",
        predecessor_label="Lane B predecessor",
        status_field_name="lane_b_predecessor_status",
        expected_source_artifact_checksums=expected_source_artifact_checksums,
    )


def _validate_predecessor_manifest(
    path: Path,
    *,
    expected_run_mode: str,
    predecessor_label: str,
    status_field_name: str,
    expected_source_artifact_checksums: Mapping[str, str] | None,
) -> None:
    payload = _read_json_mapping(
        path,
        field_name=f"{status_field_name}.artifact_paths entry",
    )
    validate_desc_joint_validation_manifest(payload)
    if payload.get("run_mode") != expected_run_mode:
        raise ValueError(
            f"{predecessor_label} evidence must be a {expected_run_mode} "
            "validation manifest."
        )
    desc_solve_status = payload["desc_solve_status"]
    if not isinstance(desc_solve_status, Mapping):
        raise ValueError(f"{predecessor_label} desc_solve_status must be an object.")
    if desc_solve_status.get("state") != "passed":
        raise ValueError(f"{predecessor_label} DESC solve did not pass.")
    physics_status = payload["physics_validation_status"]
    if not isinstance(physics_status, Mapping):
        raise ValueError(
            f"{predecessor_label} physics_validation_status must be an object."
        )
    if physics_status.get("passed") is not True:
        raise ValueError(
            f"{predecessor_label} SIMSOPT round-trip validation did not pass."
        )
    evidence_paths = _coerce_artifact_paths(
        _require_sequence(
            physics_status.get("evidence_paths"),
            field_name=f"{predecessor_label} physics_validation_status.evidence_paths",
        )
    )
    if not evidence_paths:
        raise ValueError(
            f"{predecessor_label} physics validation evidence is required."
        )
    exported_artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            payload.get("exported_artifact_paths"),
            field_name=f"{predecessor_label} exported_artifact_paths",
        )
    )
    for evidence_path in evidence_paths:
        _validate_predecessor_physics_evidence(
            Path(evidence_path).expanduser(),
            predecessor_label=predecessor_label,
            expected_exported_artifact_paths=exported_artifact_paths,
        )
    if expected_source_artifact_checksums is None:
        return
    source_artifact_checksums = _optional_source_checksum_map(
        payload.get("source_artifact_checksums"),
        field_name=f"{predecessor_label} source_artifact_checksums",
    )
    if source_artifact_checksums is None:
        raise ValueError(
            f"{predecessor_label} source_artifact_checksums are required."
        )
    expected_checksums = _coerce_source_checksum_map(
        expected_source_artifact_checksums,
        field_name="expected_source_artifact_checksums",
    )
    if source_artifact_checksums != expected_checksums:
        raise ValueError(
            f"{predecessor_label} source_artifact_checksums do not match "
            "the joint candidate source artifacts."
        )


def _validate_predecessor_physics_evidence(
    path: Path,
    *,
    predecessor_label: str,
    expected_exported_artifact_paths: Sequence[str],
) -> None:
    if not path.is_file():
        raise ValueError(
            f"{predecessor_label} physics evidence path is missing: {path}."
        )
    payload = _read_json_mapping(
        path,
        field_name=f"{predecessor_label} physics evidence",
    )
    if (
        payload.get("schema_version")
        != DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{predecessor_label} physics evidence must use schema_version "
            f"{DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION!r}."
        )
    if payload.get("source") != _SIMSOPT_PHYSICS_VALIDATION_SOURCE:
        raise ValueError(
            f"{predecessor_label} physics evidence must come from "
            f"{_SIMSOPT_PHYSICS_VALIDATION_SOURCE!r}."
        )
    if payload.get("passed") is not True:
        raise ValueError(
            f"{predecessor_label} physics evidence must record passed=true."
        )
    evidence_artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            payload.get("exported_artifact_paths"),
            field_name=f"{predecessor_label} physics evidence exported_artifact_paths",
        )
    )
    if _resolved_path_tuple(evidence_artifact_paths) != _resolved_path_tuple(
        expected_exported_artifact_paths
    ):
        raise ValueError(
            f"{predecessor_label} physics evidence exported_artifact_paths "
            "do not match the predecessor validation manifest."
        )
    live_exported_checksums = _exported_artifact_checksum_map(evidence_artifact_paths)
    evidence_exported_checksums = _coerce_source_checksum_map(
        payload.get("exported_artifact_checksums"),
        field_name=f"{predecessor_label} physics evidence exported_artifact_checksums",
    )
    if evidence_exported_checksums != live_exported_checksums:
        raise ValueError(
            f"{predecessor_label} physics evidence exported_artifact_checksums "
            "do not match the live exported artifacts."
        )
    _validate_passed_physics_sidecar_summaries(
        payload.get("poincare_metrics"),
        field_name=f"{predecessor_label} physics evidence poincare_metrics",
        expected_exported_artifact_paths=evidence_artifact_paths,
        sidecar_kind="poincare",
        require_nonempty=True,
    )
    _validate_passed_physics_sidecar_summaries(
        payload.get("boozer_states", []),
        field_name=f"{predecessor_label} physics evidence boozer_states",
        expected_exported_artifact_paths=evidence_artifact_paths,
        sidecar_kind="boozer",
        require_nonempty=payload.get("require_boozer_state") is True,
    )


def _validate_passed_physics_sidecar_summaries(
    value: object,
    *,
    field_name: str,
    expected_exported_artifact_paths: Sequence[str],
    sidecar_kind: str,
    require_nonempty: bool,
) -> None:
    summaries = _require_sequence(value, field_name=field_name)
    if require_nonempty and not summaries:
        raise ValueError(f"{field_name} must include at least one sidecar summary.")
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError(f"{field_name} entries must be objects.")
        if summary.get("passed") is not True:
            raise ValueError(f"{field_name} entries must record passed=true.")
        raw_path = summary.get("path")
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(f"{field_name} entries must record a nonempty path.")
        sidecar_path = Path(raw_path).expanduser()
        if not sidecar_path.is_file():
            raise ValueError(
                f"{field_name} sidecar path is missing: {raw_path}."
            )
        raw_sha256 = summary.get("sha256")
        if (
            not isinstance(raw_sha256, str)
            or not _SHA256_HEXDIGEST_RE.fullmatch(raw_sha256)
        ):
            raise ValueError(f"{field_name} entries must record a sidecar sha256.")
        if raw_sha256 != _sha256_file(sidecar_path):
            raise ValueError(
                f"{field_name} sidecar sha256 does not match the live file."
            )
        _validate_fixed_polish_physics_sidecar(
            sidecar_path,
            expected_exported_artifact_paths=expected_exported_artifact_paths,
            sidecar_kind=sidecar_kind,
            field_name=field_name,
        )


def _validate_fixed_polish_physics_sidecar(
    path: Path,
    *,
    expected_exported_artifact_paths: Sequence[str],
    sidecar_kind: str,
    field_name: str,
) -> None:
    payload = _read_json_mapping(path, field_name=field_name)
    _validate_sidecar_export_binding(
        payload,
        expected_exported_artifact_paths=expected_exported_artifact_paths,
        field_name=field_name,
    )
    if sidecar_kind == "poincare":
        if payload.get("validation_status") != "validated":
            raise ValueError(
                f"{field_name} Poincare sidecar must record validation_status "
                "'validated'."
            )
        if payload.get("design_only_override", False) is not False:
            raise ValueError(
                f"{field_name} Poincare sidecar must not be design-only."
            )
    elif sidecar_kind == "boozer":
        if not isinstance(payload.get("iota"), int | float):
            raise ValueError(f"{field_name} Boozer sidecar must record numeric iota.")
        if not isinstance(payload.get("G"), int | float):
            raise ValueError(f"{field_name} Boozer sidecar must record numeric G.")
    else:
        raise ValueError(f"Unknown physics sidecar kind: {sidecar_kind}.")


def _validate_sidecar_export_binding(
    payload: Mapping[str, object],
    *,
    expected_exported_artifact_paths: Sequence[str],
    field_name: str,
) -> None:
    sidecar_artifact_paths = _coerce_artifact_paths(
        _require_sequence(
            payload.get("exported_artifact_paths"),
            field_name=f"{field_name}.exported_artifact_paths",
        )
    )
    if _resolved_path_tuple(sidecar_artifact_paths) != _resolved_path_tuple(
        expected_exported_artifact_paths
    ):
        raise ValueError(
            f"{field_name} exported_artifact_paths do not match the physics "
            "validation report."
        )
    live_exported_checksums = _exported_artifact_checksum_map(sidecar_artifact_paths)
    sidecar_exported_checksums = _coerce_source_checksum_map(
        payload.get("exported_artifact_checksums"),
        field_name=f"{field_name}.exported_artifact_checksums",
    )
    if sidecar_exported_checksums != live_exported_checksums:
        raise ValueError(
            f"{field_name} exported_artifact_checksums do not match the live "
            "exported artifacts."
        )


def _validate_validation_status_section(
    section: str,
    value: object,
    *,
    allow_not_run: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be an object.")
    passed = value.get("passed")
    if allow_not_run:
        if passed is not None and not isinstance(passed, bool):
            raise ValueError(f"{section}.passed must be true, false, or null.")
    elif not isinstance(passed, bool):
        raise ValueError(f"{section}.passed must be true or false.")
    source = value.get("source")
    if not isinstance(source, str) or source == "":
        raise ValueError(f"{section}.source must be a nonempty string.")


def _optional_source_checksum_map(
    value: object,
    *,
    field_name: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    return _coerce_source_checksum_map(value, field_name=field_name)


def _require_existing_final_oracle_path(raw_path: object, *, field_name: str) -> Path:
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError(f"{field_name} must be a nonempty path string.")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ValueError(f"{field_name} must be an existing file: {raw_path}.")
    return path


def _require_passed_final_oracle_evidence(
    raw_path: object,
    *,
    field_name: str,
    expected_source_artifact_checksums: Mapping[str, str],
    exported_artifact_paths: Sequence[str],
) -> Path:
    return validate_desc_joint_final_oracle_evidence(
        raw_path,
        field_name=field_name,
        expected_source_artifact_checksums=expected_source_artifact_checksums,
        expected_exported_artifact_paths=exported_artifact_paths,
    )


def validate_desc_joint_final_oracle_evidence(
    raw_path: object,
    *,
    field_name: str = "final_oracle_evidence_path",
    expected_source_artifact_checksums: Mapping[str, str] | None = None,
    expected_exported_artifact_paths: Sequence[str] | None = None,
) -> Path:
    path = _require_existing_final_oracle_path(raw_path, field_name=field_name)
    payload = _read_json_mapping(path, field_name=field_name)
    if payload.get("schema_version") != DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"{field_name} must use schema_version "
            f"{DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION!r}."
        )
    if payload.get("source") != _FINAL_ORACLE_SOURCE:
        raise ValueError(f"{field_name} must record source {_FINAL_ORACLE_SOURCE!r}.")
    if payload.get("passed") is not True:
        raise ValueError(f"{field_name} must record passed=true.")
    source_checksums = _coerce_source_checksum_map(
        payload.get("source_artifact_checksums"),
        field_name=f"{field_name}.source_artifact_checksums",
    )
    if expected_source_artifact_checksums is not None:
        expected_checksums = _coerce_source_checksum_map(
            expected_source_artifact_checksums,
            field_name="expected_source_artifact_checksums",
        )
        if source_checksums != expected_checksums:
            raise ValueError(
                f"{field_name} source_artifact_checksums do not match the "
                "expected source artifact checksums."
            )
    if expected_exported_artifact_paths is not None:
        evidence_paths = _coerce_artifact_paths(
            _require_sequence(
                payload.get("exported_artifact_paths"),
                field_name=f"{field_name}.exported_artifact_paths",
            )
        )
        expected_paths = _coerce_artifact_paths(expected_exported_artifact_paths)
        if not expected_paths:
            raise ValueError(
                f"{field_name} must be bound to at least one exported artifact."
            )
        if evidence_paths != expected_paths:
            raise ValueError(
                f"{field_name} exported_artifact_paths do not match the "
                "validation manifest."
            )
        expected_exported_checksums = _exported_artifact_checksum_map(expected_paths)
        evidence_exported_checksums = _coerce_source_checksum_map(
            payload.get("exported_artifact_checksums"),
            field_name=f"{field_name}.exported_artifact_checksums",
        )
        if evidence_exported_checksums != expected_exported_checksums:
            raise ValueError(
                f"{field_name} exported_artifact_checksums do not match the "
                "live exported artifacts."
            )
    return path


def _read_json_mapping(path: Path, *, field_name: str) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must contain a JSON object.")
    return payload


def _require_sequence(value: object, *, field_name: str) -> Sequence[object]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list.")
    return value


def _coerce_source_checksum_map(
    value: object,
    *,
    field_name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a nonempty object.")
    checksums: dict[str, str] = {}
    for name, checksum in value.items():
        if not isinstance(name, str) or name == "":
            raise ValueError(f"{field_name} keys must be nonempty strings.")
        if (
            not isinstance(checksum, str)
            or not _SHA256_HEXDIGEST_RE.fullmatch(checksum)
        ):
            raise ValueError(
                f"{field_name}.{name} must be a lowercase SHA-256 hex digest."
            )
        checksums[name] = checksum
    return checksums


def _exported_artifact_checksum_map(paths: Sequence[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path_string in paths:
        path = Path(path_string).expanduser()
        if not path.is_file():
            raise ValueError(
                "exported_artifact_paths must point to existing files before "
                f"promotion can pass: {path_string}."
            )
        checksums[path_string] = _sha256_file(path)
    return checksums


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION",
    "DESC_JOINT_VALIDATION_MANIFEST_SCHEMA_VERSION",
    "build_desc_joint_validation_manifest",
    "render_desc_joint_validation_report",
    "resolve_desc_joint_promotion_status",
    "validate_desc_joint_final_oracle_evidence",
    "validate_desc_joint_validation_manifest",
]
