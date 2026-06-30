"""Launch high-cost SIMSOPT validation for DESC-exported banana artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from banana_opt.desc_joint_simsopt_validation import (
    DescJointSimsoptValidationArtifacts,
    materialize_desc_joint_simsopt_validation,
)
from banana_opt.desc_joint_validation import build_desc_joint_validation_manifest
from banana_opt.json_compat import load_boozer_finite_i as load

DESC_JOINT_SIMSOPT_VALIDATION_LAUNCH_SCHEMA_VERSION = (
    "desc_joint_simsopt_validation_launch_v1"
)
_EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
_POINCARE_SCRIPT_PATH = (
    _EXAMPLES_ROOT / "POINCARE_PLOTTING" / "poincare_surfaces.py"
)
_POINCARE_SCRIPT_DIR = _POINCARE_SCRIPT_PATH.parent
_POINCARE_METRIC_SUFFIX_BY_MODE = {
    "validation": "_validation",
    "diagnostic": "_diagnostic",
    "default": "_default",
}


@dataclass(frozen=True, slots=True)
class DescJointSimsoptValidationLaunchArtifacts:
    launch_report_path: Path
    physics_artifacts: DescJointSimsoptValidationArtifacts | None

    def validation_manifest_path(self) -> Path | None:
        if self.physics_artifacts is None:
            return None
        return self.physics_artifacts.validation_manifest_path


def launch_desc_joint_simsopt_validation(
    *,
    result_payload: Mapping[str, object],
    exported_artifact_paths: Sequence[str | Path],
    output_root: Path,
    surface_path: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    poincare_render_modes: Sequence[str] = ("validation",),
    poincare_timeout_seconds: float | None = None,
    run_poincare: bool = True,
    run_boozer: bool = True,
    require_boozer_state: bool | None = None,
    dry_run: bool = False,
    iota_guess: float | None = None,
    G_guess: float | None = None,
) -> DescJointSimsoptValidationLaunchArtifacts:
    """Prepare and optionally execute SIMSOPT Poincare/Boozer validation.

    The exported artifact paths remain the evidence identity. Files copied into
    the validation working directory are execution inputs only; generated
    sidecars are patched with the live exported-artifact paths and SHA-256s.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    exported_paths = _coerce_existing_paths(
        exported_artifact_paths,
        field_name="exported_artifact_paths",
    )
    if not exported_paths:
        raise ValueError(
            "DESC joint validation launch requires at least one exported artifact."
        )
    resolved_surface_path = _resolve_surface_path(
        result_payload,
        surface_path=surface_path,
    )
    joint_equilibrium_artifact_path = _joint_equilibrium_artifact_path(result_payload)
    validation_run_dir = output_root / "simsopt_validation_run"
    validation_run_dir.mkdir(parents=True, exist_ok=True)
    copied_field_path = validation_run_dir / "biot_savart_opt.json"
    copied_surface_path = validation_run_dir / "surf_opt.json"
    _copy_artifact(exported_paths[0], copied_field_path)
    _copy_artifact(resolved_surface_path, copied_surface_path)

    exported_checksums = _path_checksum_map(exported_paths)
    render_modes = _coerce_poincare_render_modes(poincare_render_modes)
    resolved_python_executable = _resolve_python_executable(python_executable)
    command = [
        resolved_python_executable,
        os.fspath(_POINCARE_SCRIPT_PATH),
    ]
    env_overrides = {
        "POINCARE_OUT_DIR": os.fspath(validation_run_dir),
        "POINCARE_RENDER_MODES": ",".join(render_modes),
    }
    _write_launch_report(
        output_root / "desc_joint_simsopt_validation_launch_report.json",
        _launch_report_payload(
            status="prepared" if dry_run else "running",
            result_payload=result_payload,
            exported_artifact_checksums=exported_checksums,
            joint_equilibrium_artifact_path=joint_equilibrium_artifact_path,
            surface_path=resolved_surface_path,
            validation_run_dir=validation_run_dir,
            copied_field_path=copied_field_path,
            copied_surface_path=copied_surface_path,
            command=command,
            env_overrides=env_overrides,
            poincare_metrics_paths=(),
            boozer_state_path=None,
            physics_artifacts=None,
            elapsed_seconds=None,
        ),
    )
    if dry_run:
        return DescJointSimsoptValidationLaunchArtifacts(
            launch_report_path=output_root
            / "desc_joint_simsopt_validation_launch_report.json",
            physics_artifacts=None,
        )

    start = time.monotonic()
    poincare_metrics_paths: tuple[Path, ...] = ()
    if run_poincare:
        completed = subprocess.run(
            command,
            cwd=os.fspath(_POINCARE_SCRIPT_DIR),
            check=True,
            timeout=poincare_timeout_seconds,
            env=_subprocess_env(env_overrides),
            text=True,
            capture_output=True,
        )
        poincare_metrics_paths = _expected_poincare_metrics_paths(
            validation_run_dir,
            render_modes=render_modes,
        )
        for metrics_path in poincare_metrics_paths:
            _bind_sidecar_to_exported_artifacts(
                metrics_path,
                exported_artifact_checksums=exported_checksums,
            )
        command_output = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    else:
        command_output = None

    boozer_state_path: Path | None = None
    if run_boozer:
        warm_start = _resolve_boozer_warm_start(
            result_payload,
            iota_guess=iota_guess,
            G_guess=G_guess,
        )
        try:
            boozer_state_path = materialize_desc_joint_boozer_validation_state(
                biot_savart_path=copied_field_path,
                surface_path=copied_surface_path,
                output_root=validation_run_dir,
                exported_artifact_checksums=exported_checksums,
                iota_guess=warm_start[0],
                G_guess=warm_start[1],
            )
        except Exception as exc:
            boozer_state_path = materialize_failed_desc_joint_boozer_validation_state(
                surface_path=resolved_surface_path,
                output_root=validation_run_dir,
                exported_artifact_checksums=exported_checksums,
                reason=_exception_reason(exc),
            )

    effective_require_boozer = (
        run_boozer if require_boozer_state is None else require_boozer_state
    )
    if not poincare_metrics_paths:
        raise ValueError("No Poincare metrics were produced for physics validation.")
    physics_artifacts = materialize_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=exported_paths,
        poincare_metrics_paths=poincare_metrics_paths,
        boozer_state_paths=()
        if boozer_state_path is None
        else (boozer_state_path,),
        require_boozer_state=effective_require_boozer,
        validated_surface_path=resolved_surface_path,
        output_root=output_root,
    )
    elapsed_seconds = time.monotonic() - start
    launch_report_path = output_root / "desc_joint_simsopt_validation_launch_report.json"
    report_payload = _launch_report_payload(
        status="completed",
        result_payload=result_payload,
        exported_artifact_checksums=exported_checksums,
        joint_equilibrium_artifact_path=joint_equilibrium_artifact_path,
        surface_path=resolved_surface_path,
        validation_run_dir=validation_run_dir,
        copied_field_path=copied_field_path,
        copied_surface_path=copied_surface_path,
        command=command,
        env_overrides=env_overrides,
        poincare_metrics_paths=poincare_metrics_paths,
        boozer_state_path=boozer_state_path,
        physics_artifacts=physics_artifacts,
        elapsed_seconds=elapsed_seconds,
    )
    if command_output is not None:
        report_payload["poincare_subprocess"] = command_output
    _write_launch_report(launch_report_path, report_payload)
    return DescJointSimsoptValidationLaunchArtifacts(
        launch_report_path=launch_report_path,
        physics_artifacts=physics_artifacts,
    )


def materialize_desc_joint_boozer_validation_state(
    *,
    biot_savart_path: Path,
    surface_path: Path,
    output_root: Path,
    exported_artifact_checksums: Mapping[str, str],
    iota_guess: float,
    G_guess: float,
) -> Path:
    """Re-solve Boozer state for the exported SIMSOPT field and fixed surface."""

    from simsopt.field import BiotSavart
    from simsopt.geo import BoozerSurface
    from simsopt.geo.surfaceobjectives import Volume

    biot_savart = load(os.fspath(biot_savart_path))
    if not isinstance(biot_savart, BiotSavart):
        raise TypeError(f"Expected BiotSavart in {biot_savart_path}.")
    loaded_surface = load(os.fspath(surface_path))
    surface = getattr(loaded_surface, "surface", loaded_surface)
    label = Volume(surface)
    targetlabel = _boozer_target_label(loaded_surface, label)
    constraint_weight = getattr(loaded_surface, "constraint_weight", None)
    options = getattr(loaded_surface, "options", None)
    boozer_surface = BoozerSurface(
        biot_savart,
        surface,
        label,
        targetlabel,
        constraint_weight=constraint_weight,
        options={} if options is None else dict(options),
    )
    result = boozer_surface.run_code(iota_guess, G=G_guess)
    if not bool(result.get("success")):
        raise RuntimeError(_boozer_failure_reason(result))
    output_root.mkdir(parents=True, exist_ok=True)
    solved_surface_path = output_root / "surf_desc_export_boozer_surface.json"
    boozer_surface.save(os.fspath(solved_surface_path))
    state_path = output_root / "surf_desc_export_boozer_state.json"
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "surface_path": os.fspath(solved_surface_path),
            "iota": float(boozer_surface.res["iota"]),
            "G": float(boozer_surface.res["G"]),
            "exported_artifact_paths": list(exported_artifact_checksums),
            "exported_artifact_checksums": dict(exported_artifact_checksums),
            "boozer_validation": {
                "source": "desc_joint_export_boozer_resolve",
                "input_biot_savart": os.fspath(biot_savart_path),
                "input_surface": os.fspath(surface_path),
                "targetlabel": float(targetlabel),
                "constraint_weight": constraint_weight,
            },
        },
    )
    return state_path


def materialize_failed_desc_joint_boozer_validation_state(
    *,
    surface_path: Path,
    output_root: Path,
    exported_artifact_checksums: Mapping[str, str],
    reason: str,
) -> Path:
    """Write checksum-bound failed Boozer evidence instead of aborting validation."""

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "surf_desc_export_boozer_state_failed.json"
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "surface_path": os.fspath(surface_path),
            "passed": False,
            "reason": reason,
            "exported_artifact_paths": list(exported_artifact_checksums),
            "exported_artifact_checksums": dict(exported_artifact_checksums),
            "boozer_validation": {
                "source": "desc_joint_export_boozer_resolve",
                "status": "failed",
            },
        },
    )
    return state_path


def infer_desc_joint_exported_artifact_paths(
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
    return ()


def _resolve_surface_path(
    result_payload: Mapping[str, object],
    *,
    surface_path: str | Path | None,
) -> Path:
    if _is_joint_run(result_payload):
        expected_surface = _joint_exported_surface_path(result_payload)
        if surface_path is None:
            return expected_surface
        explicit_surface = _coerce_existing_path(
            surface_path,
            field_name="surface_path",
        )
        if explicit_surface.resolve() != expected_surface.resolve():
            raise ValueError(
                "joint-mode SIMSOPT validation surface must match "
                "desc_runtime_artifacts.exported_surface."
            )
        return explicit_surface
    if surface_path is not None:
        return _coerce_existing_path(surface_path, field_name="surface_path")
    input_contract = result_payload.get("input_contract")
    if not isinstance(input_contract, Mapping):
        raise ValueError("result_payload.input_contract must be an object.")
    selected_seed = input_contract.get("selected_seed")
    if not isinstance(selected_seed, Mapping):
        raise ValueError(
            "result_payload.input_contract.selected_seed must be an object or "
            "surface_path must be supplied."
        )
    raw_surface = selected_seed.get("surface")
    if not isinstance(raw_surface, str) or raw_surface == "":
        raise ValueError(
            "result_payload.input_contract.selected_seed.surface must be a "
            "nonempty path string or surface_path must be supplied."
        )
    return _coerce_existing_path(raw_surface, field_name="selected_seed.surface")


def _joint_exported_surface_path(result_payload: Mapping[str, object]) -> Path:
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise ValueError(
            "joint-mode validation launch requires "
            "desc_runtime_artifacts.exported_surface."
        )
    raw_surface_path = runtime_artifacts.get("exported_surface")
    if not isinstance(raw_surface_path, str) or raw_surface_path == "":
        raise ValueError(
            "joint-mode validation launch requires a nonempty "
            "desc_runtime_artifacts.exported_surface path."
        )
    return _coerce_existing_path(
        raw_surface_path,
        field_name="desc_runtime_artifacts.exported_surface",
    )


def _joint_equilibrium_artifact_path(
    result_payload: Mapping[str, object],
) -> Path | None:
    if not _is_joint_run(result_payload):
        return None
    runtime_artifacts = result_payload.get("desc_runtime_artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise ValueError(
            "joint-mode validation launch requires "
            "desc_runtime_artifacts.desc_equilibrium."
        )
    raw_equilibrium_path = runtime_artifacts.get("desc_equilibrium")
    if not isinstance(raw_equilibrium_path, str) or raw_equilibrium_path == "":
        raise ValueError(
            "joint-mode validation launch requires a nonempty "
            "desc_runtime_artifacts.desc_equilibrium path."
        )
    return _coerce_existing_path(
        raw_equilibrium_path,
        field_name="desc_runtime_artifacts.desc_equilibrium",
    )


def _is_joint_run(result_payload: Mapping[str, object]) -> bool:
    return result_payload.get("run_mode") in {"vacuum_joint", "finite_beta_joint"}


def _resolve_boozer_warm_start(
    result_payload: Mapping[str, object],
    *,
    iota_guess: float | None,
    G_guess: float | None,
) -> tuple[float, float]:
    if _is_joint_run(result_payload):
        if iota_guess is None or G_guess is None:
            raise ValueError(
                "joint-mode Boozer validation requires explicit --iota and --G "
                "for the optimized surface; seed-state warm starts are not "
                "authoritative after moving-boundary optimization."
            )
        return (
            _finite_number(iota_guess, field_name="--iota"),
            _finite_number(G_guess, field_name="--G"),
        )
    if iota_guess is not None and G_guess is not None:
        return float(iota_guess), float(G_guess)
    state_path = _selected_seed_state_path(result_payload)
    if state_path is not None:
        payload = _read_json_mapping(state_path)
        state_iota = _finite_number(payload.get("iota"), field_name="state.iota")
        state_G = _finite_number(payload.get("G"), field_name="state.G")
        return (
            state_iota if iota_guess is None else float(iota_guess),
            state_G if G_guess is None else float(G_guess),
        )
    raise ValueError(
        "Boozer validation requires --iota/--G or a selected_seed.state sidecar "
        "with finite iota and G."
    )


def _selected_seed_state_path(result_payload: Mapping[str, object]) -> Path | None:
    input_contract = result_payload.get("input_contract")
    if not isinstance(input_contract, Mapping):
        return None
    selected_seed = input_contract.get("selected_seed")
    if not isinstance(selected_seed, Mapping):
        return None
    raw_state = selected_seed.get("state")
    if raw_state is None:
        return None
    if not isinstance(raw_state, str) or raw_state == "":
        raise ValueError("selected_seed.state must be a nonempty path string.")
    return _coerce_existing_path(raw_state, field_name="selected_seed.state")


def _launch_report_payload(
    *,
    status: str,
    result_payload: Mapping[str, object],
    exported_artifact_checksums: Mapping[str, str],
    joint_equilibrium_artifact_path: Path | None,
    surface_path: Path,
    validation_run_dir: Path,
    copied_field_path: Path,
    copied_surface_path: Path,
    command: Sequence[str],
    env_overrides: Mapping[str, str],
    poincare_metrics_paths: Sequence[Path],
    boozer_state_path: Path | None,
    physics_artifacts: DescJointSimsoptValidationArtifacts | None,
    elapsed_seconds: float | None,
) -> dict[str, object]:
    manifest_path = None
    validation_report_path = None
    physics_report_path = None
    if physics_artifacts is not None:
        manifest_path = os.fspath(physics_artifacts.validation_manifest_path)
        validation_report_path = os.fspath(physics_artifacts.validation_report_path)
        physics_report_path = os.fspath(physics_artifacts.physics_report_path)
    return {
        "schema_version": DESC_JOINT_SIMSOPT_VALIDATION_LAUNCH_SCHEMA_VERSION,
        "status": status,
        "run_mode": result_payload.get("run_mode"),
        "validation_run_dir": os.fspath(validation_run_dir),
        "source_surface_path": os.fspath(surface_path),
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
        "prepared_inputs": {
            "biot_savart_opt": os.fspath(copied_field_path),
            "surf_opt": os.fspath(copied_surface_path),
        },
        "exported_artifact_paths": list(exported_artifact_checksums),
        "exported_artifact_checksums": dict(exported_artifact_checksums),
        "poincare": {
            "script": os.fspath(_POINCARE_SCRIPT_PATH),
            "command": list(command),
            "env_overrides": dict(env_overrides),
            "metrics_paths": [os.fspath(path) for path in poincare_metrics_paths],
        },
        "boozer_state_path": None
        if boozer_state_path is None
        else os.fspath(boozer_state_path),
        "physics_report_path": physics_report_path,
        "validation_manifest_path": manifest_path,
        "validation_report_path": validation_report_path,
        "elapsed_seconds": elapsed_seconds,
    }


def _write_launch_report(path: Path, payload: Mapping[str, object]) -> None:
    _write_json(path, payload)


def _bind_sidecar_to_exported_artifacts(
    path: Path,
    *,
    exported_artifact_checksums: Mapping[str, str],
) -> None:
    payload = dict(_read_json_mapping(path))
    payload["exported_artifact_paths"] = list(exported_artifact_checksums)
    payload["exported_artifact_checksums"] = dict(exported_artifact_checksums)
    _write_json(path, payload)


def _expected_poincare_metrics_paths(
    validation_run_dir: Path,
    *,
    render_modes: Sequence[str],
) -> tuple[Path, ...]:
    paths = tuple(
        validation_run_dir
        / f"PoincareMetrics_opt{_POINCARE_METRIC_SUFFIX_BY_MODE[mode]}.json"
        for mode in render_modes
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Poincare validation did not write metrics: {missing}.")
    return paths


def _coerce_poincare_render_modes(render_modes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(render_modes, str) or not isinstance(render_modes, Sequence):
        raise ValueError("poincare_render_modes must be a sequence of strings.")
    modes: list[str] = []
    for mode in render_modes:
        if mode not in _POINCARE_METRIC_SUFFIX_BY_MODE:
            raise ValueError(
                "poincare_render_modes entries must be validation, diagnostic, "
                f"or default; got {mode!r}."
            )
        modes.append(mode)
    if not modes:
        raise ValueError("poincare_render_modes must not be empty.")
    if len(set(modes)) != len(modes):
        raise ValueError("poincare_render_modes must not repeat modes.")
    return tuple(modes)


def _copy_artifact(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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


def _subprocess_env(env_overrides: Mapping[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(dict(env_overrides))
    return env


def _resolve_python_executable(python_executable: str | Path) -> str:
    raw_executable = os.fspath(python_executable)
    expanded_executable = os.path.expanduser(raw_executable)
    if os.path.isabs(expanded_executable):
        return expanded_executable
    has_path_separator = os.sep in expanded_executable or (
        os.altsep is not None and os.altsep in expanded_executable
    )
    if has_path_separator:
        return os.fspath(Path(expanded_executable).resolve())
    return expanded_executable


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    scalar = float(value)
    if scalar != scalar or scalar in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite.")
    return scalar


def _boozer_target_label(loaded_surface: object, label: object) -> float:
    targetlabel = getattr(loaded_surface, "targetlabel", None)
    if targetlabel is None:
        return float(label.J())
    return float(targetlabel)


def _boozer_failure_reason(result: Mapping[str, object]) -> str:
    residual_norm = _array_inf_norm(result.get("residual"))
    fields: list[str] = ["Boozer validation solve failed"]
    for name in ("success", "iter", "iota", "G", "type"):
        value = result.get(name)
        if _json_scalar(value):
            fields.append(f"{name}={value}")
    if residual_norm is not None:
        fields.append(f"residual_inf={residual_norm:.3e}")
    return "; ".join(fields)


def _exception_reason(exc: Exception) -> str:
    message = str(exc)
    if len(message) > 1000:
        message = message[:997] + "..."
    return f"{type(exc).__name__}: {message}"


def _array_inf_norm(value: object) -> float | None:
    try:
        import numpy as np

        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.size == 0 or not np.isfinite(array).all():
        return None
    norm = float(np.linalg.norm(array, ord=np.inf))
    if not math.isfinite(norm):
        return None
    return norm


def _json_scalar(value: object) -> bool:
    return isinstance(value, (bool, int, float, str))


__all__ = [
    "DESC_JOINT_SIMSOPT_VALIDATION_LAUNCH_SCHEMA_VERSION",
    "DescJointSimsoptValidationLaunchArtifacts",
    "infer_desc_joint_exported_artifact_paths",
    "launch_desc_joint_simsopt_validation",
    "materialize_desc_joint_boozer_validation_state",
    "materialize_failed_desc_joint_boozer_validation_state",
]
