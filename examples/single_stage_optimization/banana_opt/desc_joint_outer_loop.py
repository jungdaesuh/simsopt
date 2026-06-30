"""Fail-closed outer-loop gate for DESC joint production candidates."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from banana_opt.desc_joint_result_schema import validate_desc_joint_result_payload
from banana_opt.desc_joint_validation import validate_desc_joint_validation_manifest

DESC_JOINT_OUTER_LOOP_DECISION_SCHEMA_VERSION = "desc_joint_outer_loop_decision_v1"
DescJointOuterLoopDecision = Literal["accepted", "rejected"]
_JOINT_RUN_MODES = frozenset({"vacuum_joint", "finite_beta_joint"})


@dataclass(frozen=True, slots=True)
class DescJointOuterLoopDecisionArtifact:
    decision_path: Path
    payload: Mapping[str, object]


def materialize_desc_joint_outer_loop_decision(
    *,
    result_payload: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    output_root: Path,
    validation_manifest_path: Path | None = None,
) -> DescJointOuterLoopDecisionArtifact:
    """Write the production accept/reject decision for one joint candidate."""

    validate_desc_joint_result_payload(result_payload)
    validate_desc_joint_validation_manifest(validation_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = desc_joint_outer_loop_decision_payload(
        result_payload=result_payload,
        validation_manifest=validation_manifest,
        validation_manifest_path=validation_manifest_path,
    )
    decision_path = output_root / "desc_joint_outer_loop_decision.json"
    decision_path.write_text(_json_dumps(payload), encoding="utf-8")
    return DescJointOuterLoopDecisionArtifact(
        decision_path=decision_path,
        payload=payload,
    )


def desc_joint_outer_loop_decision_payload(
    *,
    result_payload: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    validation_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Return the fail-closed production gate decision for one joint candidate."""

    validate_desc_joint_result_payload(result_payload)
    validate_desc_joint_validation_manifest(validation_manifest)
    run_mode = result_payload["run_mode"]
    validation_run_mode = validation_manifest["run_mode"]
    if validation_run_mode != run_mode:
        raise ValueError("validation_manifest run_mode does not match desc_result.json.")
    validation_exported_artifact_paths = _coerce_path_sequence(
        validation_manifest.get("exported_artifact_paths", ()),
        field_name="validation_manifest.exported_artifact_paths",
    )
    result_exported_artifact_paths = _require_result_validation_artifact_binding(
        result_payload=result_payload,
        validation_exported_artifact_paths=validation_exported_artifact_paths,
    )
    desc_solve_status = _mapping(
        result_payload["desc_solve_status"],
        "desc_solve_status",
    )
    fixed_polish_status = _mapping(
        validation_manifest.get("fixed_polish_predecessor_status", {}),
        "fixed_polish_predecessor_status",
    )
    lane_b_status = _mapping(
        validation_manifest.get("lane_b_predecessor_status", {}),
        "lane_b_predecessor_status",
    )
    physics_status = _mapping(
        validation_manifest["physics_validation_status"],
        "physics_validation_status",
    )
    artifact_hardware_status = _mapping(
        validation_manifest["artifact_hardware_status"],
        "artifact_hardware_status",
    )
    final_oracle_status = _mapping(
        validation_manifest["final_oracle_status"],
        "final_oracle_status",
    )
    promotion_status = _mapping(
        validation_manifest["promotion_status"],
        "promotion_status",
    )
    if run_mode in _JOINT_RUN_MODES and promotion_status.get("state") == "passed":
        _require_joint_result_manifest_runtime_binding(
            result_payload=result_payload,
            validation_manifest=validation_manifest,
        )
    decision, rejection_stage, reason = _resolve_outer_loop_decision(
        run_mode=run_mode,
        desc_solve_status=desc_solve_status,
        fixed_polish_status=fixed_polish_status,
        lane_b_status=lane_b_status,
        physics_status=physics_status,
        artifact_hardware_status=artifact_hardware_status,
        final_oracle_status=final_oracle_status,
        promotion_status=promotion_status,
    )
    return {
        "schema_version": DESC_JOINT_OUTER_LOOP_DECISION_SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "rejection_stage": rejection_stage,
        "run_mode": run_mode,
        "eligible_for_next_search_stage": decision == "accepted",
        "eligible_for_promotion": decision == "accepted",
        "validation_manifest_path": (
            None
            if validation_manifest_path is None
            else os.fspath(validation_manifest_path.resolve())
        ),
        "result_exported_artifact_paths": result_exported_artifact_paths,
        "exported_artifact_paths": validation_exported_artifact_paths,
        "desc_solve_status": dict(desc_solve_status),
        "fixed_polish_predecessor_status": dict(fixed_polish_status),
        "lane_b_predecessor_status": dict(lane_b_status),
        "physics_validation_status": dict(physics_status),
        "artifact_hardware_status": dict(artifact_hardware_status),
        "final_oracle_status": dict(final_oracle_status),
        "promotion_status": dict(promotion_status),
    }


def _resolve_outer_loop_decision(
    *,
    run_mode: object,
    desc_solve_status: Mapping[str, object],
    fixed_polish_status: Mapping[str, object],
    lane_b_status: Mapping[str, object],
    physics_status: Mapping[str, object],
    artifact_hardware_status: Mapping[str, object],
    final_oracle_status: Mapping[str, object],
    promotion_status: Mapping[str, object],
) -> tuple[DescJointOuterLoopDecision, str | None, str]:
    if run_mode not in _JOINT_RUN_MODES:
        return (
            "rejected",
            "run_mode",
            "outer-loop hardware gate is only for vacuum_joint or finite_beta_joint",
        )
    if desc_solve_status.get("state") != "passed":
        return (
            "rejected",
            "desc_solve",
            _status_text(
                desc_solve_status,
                "reason",
                "DESC solve status has not passed",
            ),
        )
    if fixed_polish_status.get("passed") is not True:
        return (
            "rejected",
            "fixed_polish_predecessor",
            _status_text(
                fixed_polish_status,
                "reason",
                "fixed-polish predecessor validation has not passed",
            ),
        )
    if run_mode == "finite_beta_joint" and lane_b_status.get("passed") is not True:
        return (
            "rejected",
            "lane_b_predecessor",
            _status_text(
                lane_b_status,
                "reason",
                "Lane B vacuum-joint predecessor validation has not passed",
            ),
        )
    if physics_status.get("passed") is not True:
        return (
            "rejected",
            "physics_validation",
            _status_text(
                physics_status,
                "source",
                "SIMSOPT physics validation did not pass",
            ),
        )
    if artifact_hardware_status.get("passed") is not True:
        return (
            "rejected",
            "artifact_hardware",
            _status_text(
                artifact_hardware_status,
                "source",
                "artifact hardware validation did not pass",
            ),
        )
    if final_oracle_status.get("passed") is not True:
        return "rejected", "final_oracle", "direct hardware/contact oracle did not pass"
    if promotion_status.get("state") != "passed":
        return (
            "rejected",
            "promotion",
            _status_text(
                promotion_status,
                "reason",
                "joint candidate promotion is blocked",
            ),
        )
    return (
        "accepted",
        None,
        "joint candidate passed predecessor, physics, hardware, and oracle gates",
    )


def _status_text(
    status: Mapping[str, object],
    field_name: str,
    default: str,
) -> str:
    value = status.get(field_name)
    if isinstance(value, str) and value != "":
        return value
    return default


def _require_result_validation_artifact_binding(
    *,
    result_payload: Mapping[str, object],
    validation_exported_artifact_paths: list[str],
) -> list[str]:
    run_mode = result_payload["run_mode"]
    if run_mode not in _JOINT_RUN_MODES:
        return []
    result_paths = _infer_result_exported_artifact_paths(result_payload)
    if not result_paths:
        raise ValueError(
            "desc_result.json does not record exported artifact paths for this "
            "joint candidate."
        )
    if _resolved_path_tuple(result_paths) != _resolved_path_tuple(
        validation_exported_artifact_paths,
    ):
        raise ValueError(
            "validation_manifest exported artifact paths do not match "
            "desc_result.json."
        )
    return list(result_paths)


def _infer_result_exported_artifact_paths(
    result_payload: Mapping[str, object],
) -> tuple[str, ...]:
    artifact_hardware_status = result_payload.get("artifact_hardware_status")
    if isinstance(artifact_hardware_status, Mapping):
        artifact_paths = artifact_hardware_status.get("artifact_paths")
        if not isinstance(artifact_paths, str) and isinstance(
            artifact_paths,
            Sequence,
        ):
            paths = tuple(
                path
                for path in artifact_paths
                if isinstance(path, str) and path != ""
            )
            if paths:
                return paths
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if isinstance(runtime_artifacts, Mapping):
        exported_biot_savart = runtime_artifacts.get("exported_biot_savart")
        if isinstance(exported_biot_savart, str) and exported_biot_savart != "":
            return (exported_biot_savart,)
    conversion_artifacts = result_payload.get("conversion_artifacts")
    if isinstance(conversion_artifacts, Mapping):
        exported_biot_savart = conversion_artifacts.get("exported_biot_savart")
        if isinstance(exported_biot_savart, str) and exported_biot_savart != "":
            return (exported_biot_savart,)
    return ()


def _require_joint_result_manifest_runtime_binding(
    *,
    result_payload: Mapping[str, object],
    validation_manifest: Mapping[str, object],
) -> None:
    result_equilibrium_path = _required_runtime_artifact_path(
        result_payload,
        payload_name="desc_result.json",
        artifact_name="desc_equilibrium",
    )
    manifest_equilibrium_path = _required_runtime_artifact_path(
        validation_manifest,
        payload_name="validation_manifest",
        artifact_name="desc_equilibrium",
    )
    if result_equilibrium_path != manifest_equilibrium_path:
        raise ValueError(
            "validation_manifest desc_runtime_artifacts.desc_equilibrium does not "
            "match desc_result.json."
        )


def _required_runtime_artifact_path(
    payload: Mapping[str, object],
    *,
    payload_name: str,
    artifact_name: str,
) -> Path:
    runtime_artifacts = _mapping(
        payload.get("desc_runtime_artifacts"),
        f"{payload_name}.desc_runtime_artifacts",
    )
    raw_path = runtime_artifacts.get(artifact_name)
    if not isinstance(raw_path, str) or raw_path == "":
        raise ValueError(
            f"{payload_name}.desc_runtime_artifacts.{artifact_name} must be a "
            "nonempty path."
        )
    return Path(raw_path).expanduser().resolve()


def _coerce_path_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of strings.")
    paths: list[str] = []
    for raw_path in value:
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(f"{field_name} must contain nonempty strings.")
        paths.append(raw_path)
    return tuple(paths)


def _resolved_path_tuple(paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(os.fspath(Path(path).expanduser().resolve()) for path in paths)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "DESC_JOINT_OUTER_LOOP_DECISION_SCHEMA_VERSION",
    "DescJointOuterLoopDecisionArtifact",
    "desc_joint_outer_loop_decision_payload",
    "materialize_desc_joint_outer_loop_decision",
]
