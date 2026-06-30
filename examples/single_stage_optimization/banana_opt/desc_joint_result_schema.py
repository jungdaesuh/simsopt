"""Result payload schema for DESC joint banana optimization runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DESC_JOINT_RESULT_SCHEMA_VERSION = "desc_joint_result_v1"

DescJointRunMode = Literal[
    "fixed_equilibrium_polish",
    "vacuum_joint",
    "finite_beta_joint",
]
DescJointSectionState = Literal[
    "not_started",
    "not_run",
    "preflight_passed",
    "running",
    "passed",
    "failed",
    "blocked",
]
PromotionState = Literal["not_requested", "passed", "failed", "blocked"]

DESC_JOINT_RUN_MODES: tuple[str, ...] = (
    "fixed_equilibrium_polish",
    "vacuum_joint",
    "finite_beta_joint",
)
DESC_JOINT_REQUIRED_RESULT_SECTIONS: tuple[str, ...] = (
    "input_contract",
    "desc_solve_status",
    "search_hardware_status",
    "artifact_hardware_status",
    "physics_validation_status",
    "promotion_status",
)
DESC_JOINT_STATUS_SECTIONS: tuple[str, ...] = (
    "desc_solve_status",
    "search_hardware_status",
    "artifact_hardware_status",
    "physics_validation_status",
    "promotion_status",
)
_SECTION_STATES = frozenset(
    {
        "not_started",
        "not_run",
        "preflight_passed",
        "running",
        "passed",
        "failed",
        "blocked",
    }
)
_PROMOTION_STATES = frozenset({"not_requested", "passed", "failed", "blocked"})
_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION = "desc_joint_final_oracle_evidence_v1"
_FINAL_ORACLE_SOURCE = "direct_loaded_artifact_hardware_contact_oracle"
_SHA256_HEXDIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DescJointStatus:
    """Small typed status record used by every result section."""

    state: DescJointSectionState | PromotionState
    reason: str
    artifact_paths: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "artifact_paths": list(self.artifact_paths),
        }


def empty_status(
    state: DescJointSectionState | PromotionState,
    reason: str,
) -> DescJointStatus:
    return DescJointStatus(state=state, reason=reason)


def build_preflight_result_payload(
    *,
    mode: DescJointRunMode,
    input_contract: Mapping[str, object],
    objective_stack: Sequence[str],
) -> dict[str, object]:
    validate_desc_joint_mode(mode)
    payload: dict[str, object] = {
        "schema_version": DESC_JOINT_RESULT_SCHEMA_VERSION,
        "run_mode": mode,
        "objective_stack": list(objective_stack),
        "input_contract": dict(input_contract),
        "desc_solve_status": empty_status(
            "preflight_passed",
            "preflight completed; optimization has not started",
        ).to_json_dict(),
        "search_hardware_status": empty_status(
            "not_run",
            "search-time hardware steering has not evaluated this preflight",
        ).to_json_dict(),
        "artifact_hardware_status": empty_status(
            "not_run",
            "no DESC-exported artifact exists yet",
        ).to_json_dict(),
        "physics_validation_status": empty_status(
            "not_run",
            "SIMSOPT Boozer/Poincare validation has not run",
        ).to_json_dict(),
        "promotion_status": empty_status(
            "not_requested",
            "promotion requires physics validation plus direct hardware oracle evidence",
        ).to_json_dict(),
    }
    validate_desc_joint_result_payload(payload)
    return payload


def validate_desc_joint_mode(mode: object) -> None:
    if mode not in DESC_JOINT_RUN_MODES:
        choices = ", ".join(DESC_JOINT_RUN_MODES)
        raise ValueError(f"DESC joint run mode must be one of {{{choices}}}; got {mode!r}.")


def validate_desc_joint_result_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != DESC_JOINT_RESULT_SCHEMA_VERSION:
        raise ValueError("Unexpected DESC joint result schema_version.")
    validate_desc_joint_mode(payload.get("run_mode"))
    missing = [
        section
        for section in DESC_JOINT_REQUIRED_RESULT_SECTIONS
        if section not in payload
    ]
    if missing:
        raise ValueError(f"DESC joint result payload missing sections: {missing}.")
    objective_stack = payload.get("objective_stack")
    if isinstance(objective_stack, str) or not isinstance(objective_stack, Sequence):
        raise ValueError("DESC joint result payload objective_stack must be a list.")
    for section in DESC_JOINT_STATUS_SECTIONS:
        _validate_status_section(section, payload[section])
    _validate_promotion_pass_dependencies(payload)


def _validate_status_section(section: str, value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"DESC joint result section {section!r} must be an object.")
    state = value.get("state")
    allowed = _PROMOTION_STATES if section == "promotion_status" else _SECTION_STATES
    if state not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"DESC joint result section {section!r} has invalid state {state!r}; "
            f"expected one of {{{choices}}}."
        )
    reason = value.get("reason")
    if not isinstance(reason, str) or reason == "":
        raise ValueError(f"DESC joint result section {section!r} requires a reason.")
    artifact_paths = value.get("artifact_paths", [])
    if isinstance(artifact_paths, str) or not isinstance(artifact_paths, Sequence):
        raise ValueError(
            f"DESC joint result section {section!r} artifact_paths must be a list."
        )
    for artifact_path in artifact_paths:
        if not isinstance(artifact_path, str) or artifact_path == "":
            raise ValueError(
                f"DESC joint result section {section!r} artifact_paths must contain "
                "nonempty strings."
            )


def _validate_promotion_pass_dependencies(payload: Mapping[str, object]) -> None:
    promotion_status = _status_mapping(payload, "promotion_status")
    if promotion_status.get("state") != "passed":
        return
    artifact_hardware_status = _status_mapping(payload, "artifact_hardware_status")
    for section in (
        "desc_solve_status",
        "physics_validation_status",
    ):
        status = _status_mapping(payload, section)
        if status.get("state") != "passed":
            raise ValueError(
                "promotion_status cannot pass unless "
                f"{section}.state is 'passed'."
            )
    if artifact_hardware_status.get("state") != "passed":
        raise ValueError(
            "promotion_status cannot pass unless artifact_hardware_status.state "
            "is 'passed'."
        )
    exported_artifact_paths = _coerce_status_artifact_paths(
        artifact_hardware_status,
        section="artifact_hardware_status",
    )
    if not exported_artifact_paths:
        raise ValueError(
            "promotion_status cannot pass unless artifact_hardware_status records "
            "exported artifact paths."
        )
    artifact_paths = promotion_status.get("artifact_paths", [])
    if not artifact_paths:
        raise ValueError(
            "promotion_status cannot pass without direct hardware/contact oracle "
            "evidence in artifact_paths."
        )
    _validate_promotion_oracle_evidence_paths(
        artifact_paths,
        exported_artifact_paths=exported_artifact_paths,
    )


def _status_mapping(payload: Mapping[str, object], section: str) -> Mapping[str, object]:
    value = payload[section]
    if not isinstance(value, Mapping):
        raise ValueError(f"DESC joint result section {section!r} must be an object.")
    return value


def _coerce_status_artifact_paths(
    status: Mapping[str, object],
    *,
    section: str,
) -> tuple[str, ...]:
    artifact_paths = status.get("artifact_paths", [])
    if isinstance(artifact_paths, str) or not isinstance(artifact_paths, Sequence):
        raise ValueError(f"{section}.artifact_paths must be a list of strings.")
    coerced: list[str] = []
    for artifact_path in artifact_paths:
        if not isinstance(artifact_path, str) or artifact_path == "":
            raise ValueError(f"{section}.artifact_paths must contain nonempty strings.")
        coerced.append(artifact_path)
    return tuple(coerced)


def _validate_promotion_oracle_evidence_paths(
    paths: object,
    *,
    exported_artifact_paths: Sequence[str],
) -> None:
    if isinstance(paths, str) or not isinstance(paths, Sequence):
        raise ValueError("promotion_status artifact_paths must be a list of strings.")
    for raw_path in paths:
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(
                "promotion_status artifact_paths must contain direct oracle paths."
            )
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise ValueError(
                "promotion_status cannot pass without existing direct "
                f"hardware/contact oracle evidence: {raw_path}."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence must "
                f"be a JSON object: {raw_path}."
            )
        if payload.get("schema_version") != _FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence has "
                f"unexpected schema_version: {raw_path}."
            )
        if payload.get("source") != _FINAL_ORACLE_SOURCE:
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence has "
                f"unexpected source: {raw_path}."
            )
        if payload.get("passed") is not True:
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence must "
                f"record passed=true: {raw_path}."
            )
        _validate_oracle_export_binding(
            payload,
            exported_artifact_paths=exported_artifact_paths,
            evidence_path=raw_path,
        )


def _validate_oracle_export_binding(
    payload: Mapping[str, object],
    *,
    exported_artifact_paths: Sequence[str],
    evidence_path: str,
) -> None:
    oracle_paths = _coerce_oracle_artifact_paths(
        payload.get("exported_artifact_paths"),
        field_name="exported_artifact_paths",
        evidence_path=evidence_path,
    )
    expected_paths = tuple(str(Path(path).expanduser().resolve()) for path in exported_artifact_paths)
    if oracle_paths != expected_paths:
        raise ValueError(
            "promotion_status direct hardware/contact oracle evidence exported "
            f"artifact paths do not match artifact_hardware_status: {evidence_path}."
        )
    checksums = _coerce_oracle_checksum_map(
        payload.get("exported_artifact_checksums"),
        field_name="exported_artifact_checksums",
        evidence_path=evidence_path,
    )
    for artifact_path in expected_paths:
        live_path = Path(artifact_path)
        if not live_path.is_file():
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence refers "
                f"to a missing exported artifact: {artifact_path}."
            )
        expected_sha = checksums.get(artifact_path)
        if expected_sha is None:
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence missing "
                f"checksum for exported artifact: {artifact_path}."
            )
        if _sha256_file(live_path) != expected_sha:
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence checksum "
                f"does not match live exported artifact: {artifact_path}."
            )
    source_checksums = _coerce_oracle_checksum_map(
        payload.get("source_artifact_checksums"),
        field_name="source_artifact_checksums",
        evidence_path=evidence_path,
    )
    if not source_checksums:
        raise ValueError(
            "promotion_status direct hardware/contact oracle evidence must include "
            f"source_artifact_checksums: {evidence_path}."
        )


def _coerce_oracle_artifact_paths(
    value: object,
    *,
    field_name: str,
    evidence_path: str,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(
            "promotion_status direct hardware/contact oracle evidence field "
            f"{field_name} must be a list: {evidence_path}."
        )
    coerced: list[str] = []
    for raw_path in value:
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence field "
                f"{field_name} must contain nonempty paths: {evidence_path}."
            )
        coerced.append(str(Path(raw_path).expanduser().resolve()))
    if not coerced:
        raise ValueError(
            "promotion_status direct hardware/contact oracle evidence field "
            f"{field_name} must not be empty: {evidence_path}."
        )
    return tuple(coerced)


def _coerce_oracle_checksum_map(
    value: object,
    *,
    field_name: str,
    evidence_path: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "promotion_status direct hardware/contact oracle evidence field "
            f"{field_name} must be an object: {evidence_path}."
        )
    checksums: dict[str, str] = {}
    for key, checksum in value.items():
        if not isinstance(key, str) or key == "":
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence field "
                f"{field_name} has an invalid key: {evidence_path}."
            )
        if not isinstance(checksum, str) or _SHA256_HEXDIGEST_RE.fullmatch(checksum) is None:
            raise ValueError(
                "promotion_status direct hardware/contact oracle evidence field "
                f"{field_name} has an invalid SHA-256 digest: {evidence_path}."
            )
        checksums[str(Path(key).expanduser().resolve())] = checksum
    return checksums


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "DESC_JOINT_REQUIRED_RESULT_SECTIONS",
    "DESC_JOINT_RESULT_SCHEMA_VERSION",
    "DESC_JOINT_RUN_MODES",
    "DescJointRunMode",
    "DescJointStatus",
    "build_preflight_result_payload",
    "empty_status",
    "validate_desc_joint_mode",
    "validate_desc_joint_result_payload",
]
