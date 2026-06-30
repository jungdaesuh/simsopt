"""SIMSOPT physics-validation wrapper for DESC joint exported artifacts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from banana_opt.desc_joint_io import (
    read_json_mapping,
    sha256_file as _sha256_file,
)
from banana_opt.desc_joint_validation import (
    build_desc_joint_validation_manifest,
    render_desc_joint_validation_report,
)

DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION = (
    "desc_joint_simsopt_physics_validation_v1"
)
_POINCARE_PASS_STATUS = "validated"
_POINCARE_DIAGNOSTIC_STATUS = "diagnostic_only"


@dataclass(frozen=True, slots=True)
class DescJointSimsoptValidationArtifacts:
    physics_report_path: Path
    validation_manifest_path: Path
    validation_report_path: Path

    def artifact_paths(self) -> tuple[str, ...]:
        return (
            os.fspath(self.physics_report_path),
            os.fspath(self.validation_manifest_path),
            os.fspath(self.validation_report_path),
        )


def build_desc_joint_simsopt_physics_report(
    *,
    exported_artifact_paths: Sequence[str | Path],
    poincare_metrics_paths: Sequence[str | Path],
    boozer_state_paths: Sequence[str | Path] = (),
    require_boozer_state: bool = False,
    validated_surface_path: str | Path | None = None,
    joint_equilibrium_artifact_path: str | Path | None = None,
) -> dict[str, object]:
    exported_paths = _coerce_existing_paths(
        exported_artifact_paths,
        field_name="exported_artifact_paths",
    )
    if not exported_paths:
        raise ValueError(
            "DESC joint SIMSOPT validation requires at least one exported "
            "SIMSOPT artifact."
        )
    poincare_paths = _coerce_existing_paths(
        poincare_metrics_paths,
        field_name="poincare_metrics_paths",
    )
    validated_surface = _optional_existing_path(
        validated_surface_path,
        field_name="validated_surface_path",
    )
    joint_equilibrium_artifact = _optional_existing_path(
        joint_equilibrium_artifact_path,
        field_name="joint_equilibrium_artifact_path",
    )
    exported_artifact_checksums = {
        os.fspath(path): _sha256_file(path) for path in exported_paths
    }
    boozer_paths = _coerce_existing_paths(
        boozer_state_paths,
        field_name="boozer_state_paths",
    )
    poincare_summaries = tuple(
        _poincare_summary(
            path,
            expected_exported_artifact_checksums=exported_artifact_checksums,
        )
        for path in poincare_paths
    )
    boozer_summaries = tuple(
        _boozer_state_summary(
            path,
            expected_exported_artifact_checksums=exported_artifact_checksums,
        )
        for path in boozer_paths
    )
    passed, reason = _physics_status(
        poincare_summaries=poincare_summaries,
        boozer_summaries=boozer_summaries,
        require_boozer_state=require_boozer_state,
    )
    return {
        "schema_version": DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION,
        "source": "simsopt_boozer_poincare_sidecars",
        "passed": passed,
        "reason": reason,
        "exported_artifact_paths": [os.fspath(path) for path in exported_paths],
        "exported_artifact_checksums": exported_artifact_checksums,
        "validated_surface_path": (
            None if validated_surface is None else os.fspath(validated_surface)
        ),
        "validated_surface_sha256": (
            None if validated_surface is None else _sha256_file(validated_surface)
        ),
        "joint_equilibrium_artifact_path": (
            None
            if joint_equilibrium_artifact is None
            else os.fspath(joint_equilibrium_artifact)
        ),
        "joint_equilibrium_artifact_sha256": (
            None
            if joint_equilibrium_artifact is None
            else _sha256_file(joint_equilibrium_artifact)
        ),
        "poincare_metrics": [dict(summary) for summary in poincare_summaries],
        "boozer_states": [dict(summary) for summary in boozer_summaries],
        "require_boozer_state": bool(require_boozer_state),
    }


def materialize_desc_joint_simsopt_validation(
    *,
    result_payload: Mapping[str, object],
    exported_artifact_paths: Sequence[str | Path],
    poincare_metrics_paths: Sequence[str | Path],
    output_root: Path,
    boozer_state_paths: Sequence[str | Path] = (),
    require_boozer_state: bool = False,
    validated_surface_path: str | Path | None = None,
) -> DescJointSimsoptValidationArtifacts:
    output_root.mkdir(parents=True, exist_ok=True)
    joint_equilibrium_artifact_path = _joint_equilibrium_artifact_path(result_payload)
    effective_validated_surface_path = _effective_validated_surface_path(
        result_payload,
        explicit_surface_path=validated_surface_path,
        boozer_state_paths=boozer_state_paths,
    )
    physics_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=exported_artifact_paths,
        poincare_metrics_paths=poincare_metrics_paths,
        boozer_state_paths=boozer_state_paths,
        require_boozer_state=require_boozer_state,
        validated_surface_path=effective_validated_surface_path,
        joint_equilibrium_artifact_path=joint_equilibrium_artifact_path,
    )
    physics_report_path = output_root / "desc_joint_simsopt_physics_validation.json"
    _write_json(physics_report_path, physics_report)
    manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=_report_path_strings(
            physics_report,
            field_name="exported_artifact_paths",
        ),
        expected_source_artifact_checksums=None,
        physics_validation_passed=bool(physics_report["passed"]),
        artifact_hardware_passed=None,
        search_hardware_passed=None,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
        physics_validation_evidence_paths=(os.fspath(physics_report_path),),
    )
    validation_manifest_path = output_root / "desc_joint_validation_manifest.json"
    _write_json(validation_manifest_path, manifest)
    validation_report_path = output_root / "desc_joint_validation_report.md"
    validation_report_path.write_text(
        render_desc_joint_validation_report(manifest),
        encoding="utf-8",
    )
    return DescJointSimsoptValidationArtifacts(
        physics_report_path=physics_report_path,
        validation_manifest_path=validation_manifest_path,
        validation_report_path=validation_report_path,
    )


def _joint_equilibrium_artifact_path(
    result_payload: Mapping[str, object],
) -> Path | None:
    if not _is_joint_run(result_payload):
        return None
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise ValueError(
            "joint-mode physics validation requires "
            "desc_runtime_artifacts.desc_equilibrium."
        )
    raw_equilibrium_path = runtime_artifacts.get("desc_equilibrium")
    if not isinstance(raw_equilibrium_path, str) or raw_equilibrium_path == "":
        raise ValueError(
            "joint-mode physics validation requires a nonempty "
            "desc_runtime_artifacts.desc_equilibrium path."
        )
    return _coerce_existing_path(
        raw_equilibrium_path,
        field_name="desc_runtime_artifacts.desc_equilibrium",
    )


def _joint_exported_surface_path(result_payload: Mapping[str, object]) -> Path:
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise ValueError(
            "joint-mode physics validation requires "
            "desc_runtime_artifacts.exported_surface."
        )
    raw_surface_path = runtime_artifacts.get("exported_surface")
    if not isinstance(raw_surface_path, str) or raw_surface_path == "":
        raise ValueError(
            "joint-mode physics validation requires a nonempty "
            "desc_runtime_artifacts.exported_surface path."
        )
    return _coerce_existing_path(
        raw_surface_path,
        field_name="desc_runtime_artifacts.exported_surface",
    )


def _effective_validated_surface_path(
    result_payload: Mapping[str, object],
    *,
    explicit_surface_path: str | Path | None,
    boozer_state_paths: Sequence[str | Path],
) -> Path | None:
    if not _is_joint_run(result_payload):
        if explicit_surface_path is not None:
            return _coerce_existing_path(
                explicit_surface_path,
                field_name="validated_surface_path",
            )
        return None
    exported_surface = _joint_exported_surface_path(result_payload)
    if explicit_surface_path is not None:
        explicit_surface = _coerce_existing_path(
            explicit_surface_path,
            field_name="validated_surface_path",
        )
        _require_same_resolved_path(
            explicit_surface,
            exported_surface,
            field_name="validated_surface_path",
            expected_field_name="desc_runtime_artifacts.exported_surface",
        )
        return explicit_surface
    derived_surface = _single_boozer_surface_path(boozer_state_paths)
    if derived_surface is not None:
        _require_same_resolved_path(
            derived_surface,
            exported_surface,
            field_name="Boozer state surface_path",
            expected_field_name="desc_runtime_artifacts.exported_surface",
        )
        return derived_surface
    return exported_surface


def _require_same_resolved_path(
    path: Path,
    expected_path: Path,
    *,
    field_name: str,
    expected_field_name: str,
) -> None:
    if path.resolve() != expected_path.resolve():
        raise ValueError(
            f"joint-mode physics validation {field_name} must match "
            f"{expected_field_name}."
        )


def _single_boozer_surface_path(
    boozer_state_paths: Sequence[str | Path],
) -> Path | None:
    if not boozer_state_paths:
        return None
    surfaces: list[Path] = []
    for state_path in _coerce_existing_paths(
        boozer_state_paths,
        field_name="boozer_state_paths",
    ):
        payload = _read_json_mapping(state_path, field_name="boozer_state")
        raw_surface_path = payload.get("surface_path")
        if raw_surface_path is None:
            continue
        if not isinstance(raw_surface_path, str) or raw_surface_path == "":
            raise ValueError("Boozer state surface_path must be a nonempty string.")
        surfaces.append(
            _coerce_existing_path(
                raw_surface_path,
                field_name=f"{state_path}.surface_path",
            )
        )
    unique_surfaces = tuple(dict.fromkeys(surfaces))
    if len(unique_surfaces) > 1:
        raise ValueError(
            "Boozer state sidecars must not reference multiple validated surfaces."
        )
    return None if not unique_surfaces else unique_surfaces[0]


def _is_joint_run(result_payload: Mapping[str, object]) -> bool:
    return result_payload.get("run_mode") in {"vacuum_joint", "finite_beta_joint"}


def _physics_status(
    *,
    poincare_summaries: tuple[Mapping[str, object], ...],
    boozer_summaries: tuple[Mapping[str, object], ...],
    require_boozer_state: bool,
) -> tuple[bool, str]:
    validation_poincare_summaries = tuple(
        summary
        for summary in poincare_summaries
        if summary["validation_status"] != _POINCARE_DIAGNOSTIC_STATUS
    )
    if not validation_poincare_summaries:
        return False, "No strict Poincare validation sidecar was supplied."
    failed_poincare = [
        str(summary["path"])
        for summary in validation_poincare_summaries
        if summary["passed"] is not True
    ]
    if failed_poincare:
        return False, f"Poincare validation failed: {failed_poincare}."
    if require_boozer_state and not boozer_summaries:
        return False, "Boozer state validation was required but no state was supplied."
    failed_boozer = [
        str(summary["path"])
        for summary in boozer_summaries
        if summary["passed"] is not True
    ]
    if failed_boozer:
        return False, f"Boozer state validation failed: {failed_boozer}."
    return True, "SIMSOPT Poincare/Boozer validation sidecars passed."


def _poincare_summary(
    path: Path,
    *,
    expected_exported_artifact_checksums: Mapping[str, str],
) -> Mapping[str, object]:
    payload = _read_json_mapping(path, field_name="poincare_metrics")
    _validate_sidecar_artifact_binding(
        payload,
        expected_exported_artifact_checksums=expected_exported_artifact_checksums,
        field_name=f"Poincare metrics {path}",
    )
    validation_status = payload.get("validation_status")
    if not isinstance(validation_status, str) or validation_status == "":
        raise ValueError(
            f"Poincare metrics must record a nonempty validation_status: {path}."
        )
    design_only_override = payload.get("design_only_override", False)
    if not isinstance(design_only_override, bool):
        raise ValueError("Poincare metrics design_only_override must be boolean.")
    metrics = payload.get("metrics", {})
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, Mapping):
        raise ValueError("Poincare metrics field 'metrics' must be an object.")
    summary = {
        "path": os.fspath(path),
        "sha256": _sha256_file(path),
        "render_mode": payload.get("render_mode"),
        "validation_status": validation_status,
        "passed": (
            validation_status == _POINCARE_PASS_STATUS
            and not design_only_override
        ),
        "design_only_override": design_only_override,
        "nfieldlines": _optional_metric_int(payload, metrics, "nfieldlines"),
        "survived_lines": _optional_metric_int(payload, metrics, "survived_lines"),
        "plot_filename": payload.get("plot_filename"),
    }
    return summary


def _boozer_state_summary(
    path: Path,
    *,
    expected_exported_artifact_checksums: Mapping[str, str],
) -> Mapping[str, object]:
    payload = _read_json_mapping(path, field_name="boozer_state")
    _validate_sidecar_artifact_binding(
        payload,
        expected_exported_artifact_checksums=expected_exported_artifact_checksums,
        field_name=f"Boozer state {path}",
    )
    passed = payload.get("passed")
    if passed is not None and not isinstance(passed, bool):
        raise ValueError("Boozer state passed must be boolean when set.")
    if passed is False:
        reason = payload.get("reason")
        if not isinstance(reason, str) or reason == "":
            raise ValueError("Failed Boozer state sidecars must record a reason.")
        return {
            "path": os.fspath(path),
            "sha256": _sha256_file(path),
            "schema_version": payload.get("schema_version"),
            "surface_path": payload.get("surface_path"),
            "iota": None,
            "G": None,
            "passed": False,
            "reason": reason,
        }
    iota = _finite_scalar(payload.get("iota"), field_name="iota")
    G = _finite_scalar(payload.get("G"), field_name="G")
    return {
        "path": os.fspath(path),
        "sha256": _sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "surface_path": payload.get("surface_path"),
        "iota": iota,
        "G": G,
        "passed": True,
    }


def _optional_metric_int(
    payload: Mapping[str, object],
    metrics: Mapping[str, object],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Poincare metric {key!r} must be an integer when set.")
    if value < 0:
        raise ValueError(f"Poincare metric {key!r} must be nonnegative.")
    return value


def _finite_scalar(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Boozer state field {field_name!r} must be numeric.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"Boozer state field {field_name!r} must be finite.")
    return scalar


def _validate_sidecar_artifact_binding(
    payload: Mapping[str, object],
    *,
    expected_exported_artifact_checksums: Mapping[str, str],
    field_name: str,
) -> None:
    expected_paths = tuple(expected_exported_artifact_checksums)
    sidecar_paths = _sidecar_path_strings(
        payload.get("exported_artifact_paths"),
        field_name=f"{field_name}.exported_artifact_paths",
    )
    if sidecar_paths != expected_paths:
        raise ValueError(
            f"{field_name} exported_artifact_paths do not match the validated "
            "exported artifact paths."
        )
    sidecar_checksums = _sidecar_checksum_map(
        payload.get("exported_artifact_checksums"),
        field_name=f"{field_name}.exported_artifact_checksums",
    )
    if dict(sidecar_checksums) != dict(expected_exported_artifact_checksums):
        raise ValueError(
            f"{field_name} exported_artifact_checksums do not match the live "
            "exported artifacts."
        )


def _sidecar_path_strings(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of path strings.")
    paths: list[str] = []
    for raw_path in value:
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(f"{field_name} entries must be nonempty strings.")
        paths.append(raw_path)
    if not paths:
        raise ValueError(f"{field_name} must contain at least one exported artifact.")
    return tuple(paths)


def _sidecar_checksum_map(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    checksums: dict[str, str] = {}
    for raw_name, raw_checksum in value.items():
        if not isinstance(raw_name, str) or raw_name == "":
            raise ValueError(f"{field_name} keys must be nonempty strings.")
        if not isinstance(raw_checksum, str) or raw_checksum == "":
            raise ValueError(f"{field_name} values must be nonempty strings.")
        checksums[raw_name] = raw_checksum
    return checksums


def _coerce_existing_paths(
    paths: Sequence[str | Path],
    *,
    field_name: str,
) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)) or not isinstance(paths, Sequence):
        raise ValueError(f"{field_name} must be a sequence of paths.")
    resolved_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{field_name} entry must be an existing file: {path}.")
        resolved_paths.append(path)
    return tuple(resolved_paths)


def _coerce_existing_path(path: str | Path, *, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be an existing file: {resolved}.")
    return resolved


def _optional_existing_path(
    path: str | Path | None,
    *,
    field_name: str,
) -> Path | None:
    if path is None:
        return None
    return _coerce_existing_path(path, field_name=field_name)


def _report_path_strings(
    payload: Mapping[str, object],
    *,
    field_name: str,
) -> tuple[str, ...]:
    raw_paths = payload.get(field_name)
    if isinstance(raw_paths, str) or not isinstance(raw_paths, Sequence):
        raise ValueError(f"{field_name} must be a list of path strings.")
    paths: list[str] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(f"{field_name} entries must be nonempty strings.")
        paths.append(raw_path)
    return tuple(paths)


def _read_json_mapping(path: Path, *, field_name: str) -> Mapping[str, object]:
    return read_json_mapping(
        path,
        error_message=f"{field_name} must contain a JSON object: {path}.",
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION",
    "DescJointSimsoptValidationArtifacts",
    "build_desc_joint_simsopt_physics_report",
    "materialize_desc_joint_simsopt_validation",
]
