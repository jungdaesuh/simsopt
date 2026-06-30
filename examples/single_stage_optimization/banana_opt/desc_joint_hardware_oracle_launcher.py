"""Launch direct hardware/contact oracle checks for DESC joint exports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from banana_opt.desc_joint_simsopt_validation import (
    DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION,
)
from banana_opt.desc_joint_validation import (
    DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
    build_desc_joint_validation_manifest,
    render_desc_joint_validation_report,
)
from banana_opt.desc_joint_validation_launcher import (
    infer_desc_joint_exported_artifact_paths,
)

DESC_JOINT_HARDWARE_ORACLE_LAUNCH_SCHEMA_VERSION = (
    "desc_joint_hardware_oracle_launch_v1"
)
_FINAL_ORACLE_SOURCE = "direct_loaded_artifact_hardware_contact_oracle"
_AUDIT_FILENAME = "hardware_contact_audit.json"


@dataclass(frozen=True, slots=True)
class DescJointHardwareOracleLaunchArtifacts:
    launch_report_path: Path
    final_oracle_evidence_path: Path | None
    validation_manifest_path: Path | None
    validation_report_path: Path | None


def launch_desc_joint_hardware_oracle(
    *,
    result_payload: Mapping[str, object],
    exported_artifact_paths: Sequence[str | Path],
    oracle_source_artifact_path: str | Path,
    output_root: Path,
    audit_script_path: str | Path,
    physics_report_path: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> DescJointHardwareOracleLaunchArtifacts:
    """Launch the direct CAD/contact oracle and materialize promotion evidence."""

    output_root.mkdir(parents=True, exist_ok=True)
    exported_paths = _coerce_existing_paths(
        exported_artifact_paths,
        field_name="exported_artifact_paths",
    )
    if not exported_paths:
        raise ValueError(
            "DESC joint hardware oracle launch requires exported artifacts."
        )
    oracle_source_path = _coerce_existing_path(
        oracle_source_artifact_path,
        field_name="oracle_source_artifact_path",
    )
    audit_script = _coerce_existing_path(
        audit_script_path,
        field_name="audit_script_path",
    )
    physics_report = (
        None
        if physics_report_path is None
        else _coerce_existing_path(physics_report_path, field_name="physics_report")
    )
    physics_payload = (
        None if physics_report is None else _read_physics_report(physics_report)
    )
    joint_equilibrium_artifact_path = _validate_joint_oracle_binding(
        result_payload=result_payload,
        oracle_source_path=oracle_source_path,
        physics_payload=physics_payload,
    )
    source_checksums = source_artifact_checksums_from_result_payload(result_payload)
    exported_checksums = _path_checksum_map(exported_paths)
    command = [
        os.fspath(python_executable),
        os.fspath(audit_script),
        "--artifact",
        os.fspath(oracle_source_path),
    ]
    launch_report_path = output_root / "desc_joint_hardware_oracle_launch_report.json"
    _write_json(
        launch_report_path,
        _launch_report_payload(
            status="prepared" if dry_run else "running",
            result_payload=result_payload,
            command=command,
            audit_script=audit_script,
            oracle_source_path=oracle_source_path,
            joint_equilibrium_artifact_path=joint_equilibrium_artifact_path,
            exported_artifact_checksums=exported_checksums,
            source_artifact_checksums=source_checksums,
            physics_report_path=physics_report,
            audit_path=None,
            evidence_path=None,
            validation_manifest_path=None,
            validation_report_path=None,
            elapsed_seconds=None,
            subprocess_payload=None,
        ),
    )
    if dry_run:
        return DescJointHardwareOracleLaunchArtifacts(
            launch_report_path=launch_report_path,
            final_oracle_evidence_path=None,
            validation_manifest_path=None,
            validation_report_path=None,
        )

    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=os.fspath(audit_script.parent),
        check=False,
        timeout=timeout_seconds,
        text=True,
        capture_output=True,
    )
    audit_path = oracle_source_path.parent / _AUDIT_FILENAME
    audit_payload = _read_json_mapping(audit_path) if audit_path.is_file() else None
    final_oracle_passed = (
        completed.returncode == 0
        and audit_payload is not None
        and audit_payload.get("hits") == 0
    )
    evidence_path: Path | None = None
    if final_oracle_passed:
        evidence_path = output_root / "desc_joint_final_oracle_evidence.json"
        _write_json(
            evidence_path,
            _final_oracle_evidence_payload(
                exported_artifact_checksums=exported_checksums,
                source_artifact_checksums=source_checksums,
                joint_equilibrium_artifact_path=joint_equilibrium_artifact_path,
                oracle_source_path=oracle_source_path,
                audit_path=audit_path,
                audit_script=audit_script,
                command=command,
            ),
        )
    physics_passed = None
    physics_evidence_paths: tuple[str, ...] = ()
    if physics_payload is not None:
        physics_passed = physics_payload.get("passed") is True
        physics_evidence_paths = (os.fspath(physics_report),)
    validation_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=[os.fspath(path) for path in exported_paths],
        expected_source_artifact_checksums=source_checksums
        if final_oracle_passed
        else None,
        physics_validation_passed=physics_passed,
        artifact_hardware_passed=final_oracle_passed,
        search_hardware_passed=None,
        final_oracle_passed=final_oracle_passed,
        final_oracle_evidence_path=None
        if evidence_path is None
        else os.fspath(evidence_path),
        physics_validation_evidence_paths=physics_evidence_paths,
    )
    validation_manifest_path = output_root / "desc_joint_validation_manifest.json"
    _write_json(validation_manifest_path, validation_manifest)
    validation_report_path = output_root / "desc_joint_validation_report.md"
    validation_report_path.write_text(
        render_desc_joint_validation_report(validation_manifest),
        encoding="utf-8",
    )
    elapsed_seconds = time.monotonic() - start
    status = "passed" if final_oracle_passed else "failed"
    _write_json(
        launch_report_path,
        _launch_report_payload(
            status=status,
            result_payload=result_payload,
            command=command,
            audit_script=audit_script,
            oracle_source_path=oracle_source_path,
            joint_equilibrium_artifact_path=joint_equilibrium_artifact_path,
            exported_artifact_checksums=exported_checksums,
            source_artifact_checksums=source_checksums,
            physics_report_path=physics_report,
            audit_path=audit_path if audit_path.is_file() else None,
            evidence_path=evidence_path,
            validation_manifest_path=validation_manifest_path,
            validation_report_path=validation_report_path,
            elapsed_seconds=elapsed_seconds,
            subprocess_payload={
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        ),
    )
    return DescJointHardwareOracleLaunchArtifacts(
        launch_report_path=launch_report_path,
        final_oracle_evidence_path=evidence_path,
        validation_manifest_path=validation_manifest_path,
        validation_report_path=validation_report_path,
    )


def source_artifact_checksums_from_result_payload(
    result_payload: Mapping[str, object],
) -> dict[str, str]:
    for report_path in _candidate_export_report_paths(result_payload):
        if not report_path.is_file():
            continue
        payload = _read_json_mapping(report_path)
        artifact_metadata = payload.get("artifact_metadata")
        if not isinstance(artifact_metadata, Mapping):
            continue
        checksums = artifact_metadata.get("source_artifact_checksums")
        if isinstance(checksums, Mapping) and checksums:
            return _coerce_checksum_map(
                checksums,
                field_name=f"{report_path}.artifact_metadata.source_artifact_checksums",
            )
    input_contract = result_payload.get("input_contract")
    if isinstance(input_contract, Mapping):
        selected_seed = input_contract.get("selected_seed")
        if isinstance(selected_seed, Mapping):
            checksums = selected_seed.get("source_checksums")
            if isinstance(checksums, Mapping) and checksums:
                return _coerce_checksum_map(
                    checksums,
                    field_name="input_contract.selected_seed.source_checksums",
                )
    raise ValueError(
        "DESC joint hardware oracle launch requires source artifact checksums "
        "from the DESC export report or selected seed contract."
    )


def _read_physics_report(path: Path) -> Mapping[str, object]:
    physics_payload = _read_json_mapping(path)
    if (
        physics_payload.get("schema_version")
        != DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION
    ):
        raise ValueError("physics_report must be a DESC joint SIMSOPT physics report.")
    return physics_payload


def _validate_joint_oracle_binding(
    *,
    result_payload: Mapping[str, object],
    oracle_source_path: Path,
    physics_payload: Mapping[str, object] | None,
) -> Path | None:
    if not _is_joint_run(result_payload):
        return None
    if physics_payload is None:
        raise ValueError(
            "joint-mode hardware oracle launch requires a physics_report that "
            "binds the validated surface and optimized equilibrium artifact."
        )
    joint_equilibrium_path = _joint_equilibrium_artifact_path(result_payload)
    raw_physics_equilibrium_path = physics_payload.get(
        "joint_equilibrium_artifact_path",
    )
    if (
        not isinstance(raw_physics_equilibrium_path, str)
        or raw_physics_equilibrium_path == ""
    ):
        raise ValueError(
            "joint-mode physics_report must record "
            "joint_equilibrium_artifact_path."
        )
    physics_equilibrium_path = _coerce_existing_path(
        raw_physics_equilibrium_path,
        field_name="physics_report.joint_equilibrium_artifact_path",
    )
    if physics_equilibrium_path != joint_equilibrium_path:
        raise ValueError(
            "joint-mode physics_report joint_equilibrium_artifact_path does not "
            "match desc_result.json."
        )
    raw_equilibrium_sha = physics_payload.get("joint_equilibrium_artifact_sha256")
    if raw_equilibrium_sha != _sha256_file(joint_equilibrium_path):
        raise ValueError(
            "joint-mode physics_report joint_equilibrium_artifact_sha256 does "
            "not match the live optimized equilibrium artifact."
        )
    raw_surface_path = physics_payload.get("validated_surface_path")
    if not isinstance(raw_surface_path, str) or raw_surface_path == "":
        raise ValueError("joint-mode physics_report must record validated_surface_path.")
    validated_surface_path = _coerce_existing_path(
        raw_surface_path,
        field_name="physics_report.validated_surface_path",
    )
    if validated_surface_path != oracle_source_path:
        raise ValueError(
            "joint-mode hardware oracle source must match "
            "physics_report.validated_surface_path."
        )
    raw_surface_sha = physics_payload.get("validated_surface_sha256")
    if raw_surface_sha != _sha256_file(validated_surface_path):
        raise ValueError(
            "joint-mode physics_report validated_surface_sha256 does not match "
            "the live oracle source artifact."
        )
    return joint_equilibrium_path


def _joint_equilibrium_artifact_path(
    result_payload: Mapping[str, object],
) -> Path:
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise ValueError(
            "joint-mode hardware oracle launch requires "
            "desc_runtime_artifacts.desc_equilibrium."
        )
    raw_equilibrium_path = runtime_artifacts.get("desc_equilibrium")
    if not isinstance(raw_equilibrium_path, str) or raw_equilibrium_path == "":
        raise ValueError(
            "joint-mode hardware oracle launch requires a nonempty "
            "desc_runtime_artifacts.desc_equilibrium path."
        )
    return _coerce_existing_path(
        raw_equilibrium_path,
        field_name="desc_runtime_artifacts.desc_equilibrium",
    )


def _is_joint_run(result_payload: Mapping[str, object]) -> bool:
    return result_payload.get("run_mode") in {"vacuum_joint", "finite_beta_joint"}


def infer_hardware_oracle_exported_artifact_paths(
    result_payload: Mapping[str, object],
) -> tuple[str, ...]:
    return infer_desc_joint_exported_artifact_paths(result_payload)


def _candidate_export_report_paths(
    result_payload: Mapping[str, object],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if isinstance(runtime_artifacts, Mapping):
        raw_report = runtime_artifacts.get("optimized_simsopt_export_report")
        if isinstance(raw_report, str) and raw_report != "":
            paths.append(Path(raw_report).expanduser().resolve())
    conversion_artifacts = result_payload.get("conversion_artifacts")
    if isinstance(conversion_artifacts, Mapping):
        raw_report = conversion_artifacts.get("export_report")
        if isinstance(raw_report, str) and raw_report != "":
            paths.append(Path(raw_report).expanduser().resolve())
    return tuple(paths)


def _launch_report_payload(
    *,
    status: str,
    result_payload: Mapping[str, object],
    command: Sequence[str],
    audit_script: Path,
    oracle_source_path: Path,
    joint_equilibrium_artifact_path: Path | None,
    exported_artifact_checksums: Mapping[str, str],
    source_artifact_checksums: Mapping[str, str],
    physics_report_path: Path | None,
    audit_path: Path | None,
    evidence_path: Path | None,
    validation_manifest_path: Path | None,
    validation_report_path: Path | None,
    elapsed_seconds: float | None,
    subprocess_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": DESC_JOINT_HARDWARE_ORACLE_LAUNCH_SCHEMA_VERSION,
        "status": status,
        "run_mode": result_payload.get("run_mode"),
        "oracle_source_artifact_path": os.fspath(oracle_source_path),
        "joint_equilibrium_artifact_path": (
            None
            if joint_equilibrium_artifact_path is None
            else os.fspath(joint_equilibrium_artifact_path)
        ),
        "joint_equilibrium_artifact_sha256": (
            None
            if joint_equilibrium_artifact_path is None
            else _sha256_file(joint_equilibrium_artifact_path)
        ),
        "audit_script_path": os.fspath(audit_script),
        "command": list(command),
        "exported_artifact_paths": list(exported_artifact_checksums),
        "exported_artifact_checksums": dict(exported_artifact_checksums),
        "source_artifact_checksums": dict(source_artifact_checksums),
        "physics_report_path": None
        if physics_report_path is None
        else os.fspath(physics_report_path),
        "hardware_contact_audit_path": None
        if audit_path is None
        else os.fspath(audit_path),
        "final_oracle_evidence_path": None
        if evidence_path is None
        else os.fspath(evidence_path),
        "validation_manifest_path": None
        if validation_manifest_path is None
        else os.fspath(validation_manifest_path),
        "validation_report_path": None
        if validation_report_path is None
        else os.fspath(validation_report_path),
        "elapsed_seconds": elapsed_seconds,
    }
    if subprocess_payload is not None:
        payload["subprocess"] = dict(subprocess_payload)
    return payload


def _final_oracle_evidence_payload(
    *,
    exported_artifact_checksums: Mapping[str, str],
    source_artifact_checksums: Mapping[str, str],
    joint_equilibrium_artifact_path: Path | None,
    oracle_source_path: Path,
    audit_path: Path,
    audit_script: Path,
    command: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
        "source": _FINAL_ORACLE_SOURCE,
        "passed": True,
        "exported_artifact_paths": list(exported_artifact_checksums),
        "exported_artifact_checksums": dict(exported_artifact_checksums),
        "source_artifact_checksums": dict(source_artifact_checksums),
        "joint_equilibrium_artifact_path": (
            None
            if joint_equilibrium_artifact_path is None
            else os.fspath(joint_equilibrium_artifact_path)
        ),
        "joint_equilibrium_artifact_sha256": (
            None
            if joint_equilibrium_artifact_path is None
            else _sha256_file(joint_equilibrium_artifact_path)
        ),
        "oracle_source_artifact_path": os.fspath(oracle_source_path),
        "hardware_contact_audit_path": os.fspath(audit_path),
        "hardware_contact_audit_sha256": _sha256_file(audit_path),
        "audit_script_path": os.fspath(audit_script),
        "command": list(command),
    }


def _coerce_existing_paths(
    paths: Sequence[str | Path],
    *,
    field_name: str,
) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)) or not isinstance(paths, Sequence):
        raise ValueError(f"{field_name} must be a sequence of paths.")
    return tuple(
        _coerce_existing_path(path, field_name=f"{field_name} entry")
        for path in paths
    )


def _coerce_existing_path(path: str | Path, *, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be an existing file: {resolved}.")
    return resolved


def _coerce_checksum_map(
    value: Mapping[object, object],
    *,
    field_name: str,
) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for key, checksum in value.items():
        if not isinstance(key, str) or key == "":
            raise ValueError(f"{field_name} keys must be nonempty strings.")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"{field_name}.{key} must be a SHA-256 hex digest.")
        int(checksum, 16)
        checksums[key] = checksum
    if not checksums:
        raise ValueError(f"{field_name} must not be empty.")
    return checksums


def _path_checksum_map(paths: Sequence[Path]) -> dict[str, str]:
    return {os.fspath(path): _sha256_file(path) for path in paths}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "DESC_JOINT_HARDWARE_ORACLE_LAUNCH_SCHEMA_VERSION",
    "DescJointHardwareOracleLaunchArtifacts",
    "infer_hardware_oracle_exported_artifact_paths",
    "launch_desc_joint_hardware_oracle",
    "source_artifact_checksums_from_result_payload",
]
