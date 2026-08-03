"""Publish and validate custom quasi-Newton measurement receipts.

The runner deliberately writes ignored working directories.  This module is
the small, deterministic boundary that turns one or more runner directories
into a tracked receipt and verifies both the tracked copy and its archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Literal, cast
from urllib.parse import unquote, urlparse

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simsopt_jax.parity_tolerances import PARITY_LADDER_TOLERANCES
from simsopt_jax.solve.endpoint_certificate import certify_optimization_endpoint

from benchmarks.boozer_trial_diagnostic import validate_boozer_trial_trace
from benchmarks.process_gpu_monitor import parse_process_gpu_memory_artifact

_LEGACY_SCHEMA_VERSION = 1
_LEGACY_SCHEMA_VERSIONS = frozenset((1, 3, 5))
_SCHEMA_VERSION = 2
_LEGACY_RUNNER_SCHEMA_VERSION = 7
_RUNNER_SCHEMA_VERSION = 8
_JSON_INDENT = 2

QualificationKind = Literal["diagnostic", "scientific", "performance"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=_JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _archive_path(uri: str, *, repo_root: Path) -> Path:
    if uri.startswith("repo://"):
        return repo_root / uri.removeprefix("repo://")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or not parsed.path:
        raise ValueError(
            f"archive URI must use file:// or repo:// for local validation: {uri!r}"
        )
    return Path(unquote(parsed.path))


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _runner_payload(run: Path) -> dict[str, object]:
    measurements = run / "measurements.json"
    if not measurements.is_file():
        raise ValueError(f"runner directory has no measurements.json: {run}")
    payload = _json_object(measurements)
    rows = payload.get("measurements")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"runner measurements are empty: {measurements}")
    return payload


def _runner_schema_version(payload: dict[str, object]) -> int:
    value = payload.get("schema_version", _LEGACY_SCHEMA_VERSION)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("runner schema version must be an integer")
    if value < _LEGACY_SCHEMA_VERSION or value > _RUNNER_SCHEMA_VERSION:
        raise ValueError(f"unsupported runner schema version: {value}")
    return value


def _runner_commit(payload: dict[str, object]) -> str:
    value = payload.get("git_commit")
    if not isinstance(value, str) or not value.strip():
        raise TypeError("runner git_commit must be a nonempty string")
    return value


def _runner_clean(payload: dict[str, object]) -> bool:
    value = payload.get("git_clean")
    if not isinstance(value, bool):
        raise TypeError("runner git_clean must be a boolean")
    return value


def _validate_runner_child_provenance(
    run: Path,
    payload: dict[str, object],
) -> None:
    provider_child = payload.get("provider_child")
    if not isinstance(provider_child, bool):
        raise TypeError("runner provider_child must be a boolean")
    orchestrator_clean = payload.get("orchestrator_git_clean")
    if not isinstance(orchestrator_clean, bool):
        raise TypeError("runner orchestrator_git_clean must be a boolean")
    children = payload.get("provider_children")
    if not isinstance(children, list):
        raise TypeError("runner provider_children must be a list")
    if provider_child:
        if children:
            raise ValueError("provider child cannot contain nested child provenance")
        if _runner_clean(payload) is not orchestrator_clean:
            raise ValueError("provider child clean state is inconsistent")
        requested_device = payload.get("requested_device")
        method = payload.get("method")
        identity = payload.get("device_identity")
        rows = _measurement_rows(payload)
        if any(row.get("device") != requested_device for row in rows):
            raise ValueError("provider child row device differs from invocation")
        if any(row.get("method") != method for row in rows):
            raise ValueError("provider child row method differs from invocation")
        if any(row.get("device_identity") != identity for row in rows):
            raise ValueError("provider child row identity differs from invocation")
        if len({str(row.get("provider")) for row in rows}) != 1:
            raise ValueError("provider child contains multiple providers")
        return
    if not children:
        raise ValueError("runner parent omitted provider child provenance")
    parent_rows = _measurement_rows(payload)
    child_rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    seen_providers: set[str] = set()
    child_clean: list[bool] = []
    parent_commit = _runner_commit(payload)
    for raw_child in children:
        if not isinstance(raw_child, dict):
            raise TypeError("provider child provenance entries must be objects")
        child = cast(dict[str, object], raw_child)
        provider = child.get("provider")
        relative = child.get("measurements_path")
        if not isinstance(provider, str) or not provider:
            raise TypeError("provider child provenance has no provider")
        if not isinstance(relative, str) or not relative:
            raise TypeError("provider child provenance has no measurements path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("provider child measurements path escapes runner root")
        if relative in seen_paths or provider in seen_providers:
            raise ValueError("provider child provenance is duplicated")
        seen_paths.add(relative)
        seen_providers.add(provider)
        child_path = run / relative_path
        if not child_path.is_file():
            raise FileNotFoundError(
                f"provider child measurements are missing: {child_path}"
            )
        if child.get("measurements_sha256") != _sha256(child_path):
            raise ValueError("provider child measurements checksum mismatch")
        child_payload = _json_object(child_path)
        if child_payload.get("provider_child") is not True:
            raise ValueError("nested provider payload is not a provider child")
        _validate_runner_child_provenance(child_path.parent, child_payload)
        child_measurements = _measurement_rows(child_payload)
        if child.get("measurement_count") != len(child_measurements):
            raise ValueError("provider child measurement count mismatch")
        if any(row.get("provider") != provider for row in child_measurements):
            raise ValueError("provider child rows do not match bound provider")
        for field in (
            "git_commit",
            "git_clean",
            "runtime_environment",
            "requested_device",
            "method",
            "device_identity",
        ):
            if child.get(field) != child_payload.get(field):
                raise ValueError(f"provider child {field} provenance mismatch")
        if child_payload.get("git_commit") != parent_commit:
            raise ValueError("provider child commit differs from parent runner")
        if child_payload.get("runtime_environment") != payload.get(
            "runtime_environment"
        ):
            raise ValueError("provider child environment differs from parent runner")
        if child_payload.get("requested_device") != payload.get("requested_device"):
            raise ValueError("provider child device request differs from parent runner")
        if child_payload.get("method") != payload.get("method"):
            raise ValueError("provider child method differs from parent runner")
        child_clean.append(_runner_clean(child_payload))
        child_rows.extend(child_measurements)
    if child_rows != parent_rows:
        raise ValueError("parent measurements do not match provider child rows")
    expected_clean = orchestrator_clean and all(child_clean)
    if _runner_clean(payload) is not expected_clean:
        raise ValueError("parent clean state does not match provider children")


def _v7_rows_with_artifact_roots(
    run: Path,
    payload: dict[str, object],
) -> list[tuple[dict[str, object], Path]]:
    if payload.get("provider_child") is True:
        return [(row, run) for row in _measurement_rows(payload)]
    rows: list[tuple[dict[str, object], Path]] = []
    for raw_child in cast(list[object], payload["provider_children"]):
        child = cast(dict[str, object], raw_child)
        relative = cast(str, child["measurements_path"])
        child_path = run / relative
        child_payload = _json_object(child_path)
        rows.extend(
            (row, child_path.parent) for row in _measurement_rows(child_payload)
        )
    return rows


def _measurement_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_rows = payload["measurements"]
    rows: list[dict[str, object]] = []
    for raw_row in cast(list[object], raw_rows):
        if not isinstance(raw_row, dict):
            raise TypeError("runner measurements must contain JSON objects")
        rows.append(cast(dict[str, object], raw_row))
    return rows


def _all_success(rows: Iterable[dict[str, object]]) -> bool:
    return all(row.get("success") is True for row in rows)


def _required_bool(row: dict[str, object], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _required_int(row: dict[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _optional_int(row: dict[str, object], field: str) -> int | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer or null")
    return value


def _tolerance(name: str) -> tuple[float, float]:
    contract = PARITY_LADDER_TOLERANCES[name]
    rtol = contract["rtol"]
    atol = contract["atol"]
    if not isinstance(rtol, float) or not isinstance(atol, float):
        raise TypeError(f"parity tolerance {name!r} is not numerical")
    return rtol, atol


def _numeric_array(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field} must contain finite values")
    return array


def _close(
    actual: object,
    expected: object,
    *,
    tolerance_name: str,
    field: str,
) -> bool:
    rtol, atol = _tolerance(tolerance_name)
    actual_array = _numeric_array(actual, field=field)
    expected_array = _numeric_array(expected, field=field)
    return bool(
        actual_array.shape == expected_array.shape
        and np.allclose(actual_array, expected_array, rtol=rtol, atol=atol)
    )


def _recompute_endpoint_certificate(row: dict[str, object]) -> dict[str, object]:
    raw_constraint_norm = row.get("constraint_norm")
    constraint_norm = (
        None
        if raw_constraint_norm is None
        else _nonnegative_number(raw_constraint_norm, field="constraint_norm")
    )
    certificate = certify_optimization_endpoint(
        provider_success=_required_bool(row, "success"),
        provider_status=_optional_int(row, "status"),
        iterations=_required_int(row, "iterations"),
        max_iterations=_required_int(row, "maxiter"),
        initial_gradient_inf_norm=_nonnegative_number(
            row.get("initial_gradient_inf_norm"),
            field="initial_gradient_inf_norm",
        ),
        final_gradient_inf_norm=_nonnegative_number(
            row.get("final_gradient_inf_norm"),
            field="final_gradient_inf_norm",
        ),
        parameters_finite=_required_bool(row, "parameters_finite"),
        observables_finite=_required_bool(row, "observables_finite"),
        inner_success=_required_bool(row, "inner_success"),
        constraint_norm=constraint_norm,
    )
    stored = row.get("endpoint_certificate")
    recomputed = cast(dict[str, object], asdict(certificate))
    if stored != recomputed:
        raise ValueError("stored endpoint certificate does not match raw fields")
    if row.get("stopping_reason") != certificate.stopping_reason:
        raise ValueError("stored stopping reason does not match raw fields")
    return recomputed


def _validate_device_identity(
    identity: dict[str, object],
    *,
    requested_device: object,
) -> None:
    if requested_device not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported requested device: {requested_device!r}")
    if identity.get("requested_device") != requested_device:
        raise ValueError("device identity does not match measurement device")
    for field in ("backend", "platform", "jax_device", "device_kind", "hostname"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise TypeError(f"device_identity.{field} must be a nonempty string")
    if requested_device == "cpu":
        if identity.get("backend") != "cpu" or identity.get("platform") != "cpu":
            raise ValueError("CPU measurement identity is not a CPU execution")
        for field in (
            "gpu_uuid",
            "gpu_model",
            "compute_capability",
            "total_memory_bytes",
            "driver_version",
            "cuda_version",
            "visible_devices",
        ):
            if identity.get(field) is not None:
                raise ValueError(f"CPU measurement carries GPU identity field {field}")
        return
    if identity.get("backend") not in {"gpu", "cuda"}:
        raise ValueError("GPU measurement backend is not GPU execution")
    if identity.get("platform") != "cuda":
        raise ValueError("GPU measurement platform is not CUDA")
    gpu_uuid = identity.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("device_identity.gpu_uuid must be an NVIDIA GPU UUID")
    for field in (
        "gpu_model",
        "compute_capability",
        "driver_version",
        "cuda_version",
        "visible_devices",
    ):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise TypeError(f"device_identity.{field} must be a nonempty string")
    if _required_int(identity, "total_memory_bytes") <= 0:
        raise ValueError("device total memory must be positive")


def _validate_phase_rss(row: dict[str, object]) -> None:
    raw_phases = row.get("phase_rss")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise TypeError("phase_rss must be a nonempty list")
    phases: list[dict[str, object]] = []
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, dict):
            raise TypeError("phase_rss entries must be JSON objects")
        phase = cast(dict[str, object], raw_phase)
        name = phase.get("phase")
        if not isinstance(name, str) or not name:
            raise TypeError("phase_rss.phase must be a nonempty string")
        start = _required_int(phase, "start_rss_kib")
        peak = _required_int(phase, "peak_rss_kib")
        end = _required_int(phase, "end_rss_kib")
        samples = _required_int(phase, "sample_count")
        if min(start, peak, end) < 0:
            raise ValueError("phase RSS values must be nonnegative")
        if peak < start or peak < end:
            raise ValueError("phase RSS peak is inconsistent")
        if samples < 2:
            raise ValueError("phase RSS requires at least two samples")
        if phase.get("scope") != "self_proc_status_poll_10ms":
            raise ValueError("phase RSS scope is unsupported")
        phases.append(phase)
    solver_phases = [
        phase
        for phase in phases
        if phase["phase"] in {"preparation", "cold_solver", "warm_solver"}
    ]
    solver_names = {str(phase["phase"]) for phase in solver_phases}
    if not {"cold_solver", "warm_solver"}.issubset(solver_names):
        raise ValueError("phase_rss omits required solver phases")
    expected_peak = max(_required_int(phase, "peak_rss_kib") for phase in solver_phases)
    expected_delta = max(
        _required_int(phase, "peak_rss_kib") - _required_int(phase, "start_rss_kib")
        for phase in solver_phases
    )
    if _required_int(row, "solver_peak_rss_kib") != expected_peak:
        raise ValueError("solver RSS peak does not match phase measurements")
    if _required_int(row, "solver_peak_rss_delta_kib") != expected_delta:
        raise ValueError("solver RSS delta does not match phase measurements")
    lifetime_peak = _required_int(row, "peak_rss_kib")
    if lifetime_peak < expected_peak:
        raise ValueError("process lifetime RSS peak is below solver phase peak")
    if row.get("peak_rss_scope") != "provider_child_process_lifetime":
        raise ValueError("process lifetime RSS scope is unsupported")


def _validate_v7_measurement(
    row: dict[str, object],
    *,
    source_run: Path | None,
    bind_artifacts: bool = True,
) -> None:
    case = row.get("case")
    provider = row.get("provider")
    method = row.get("method")
    device = row.get("device")
    intent = row.get("intent")
    route = row.get("solver_route")
    if not isinstance(case, str) or not case:
        raise TypeError("measurement case must be a nonempty string")
    if provider not in {"native", "custom", "optax"}:
        raise ValueError(f"unsupported measurement provider: {provider!r}")
    if method not in {"bfgs", "lbfgs"}:
        raise ValueError(f"unsupported measurement method: {method!r}")
    if device not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported measurement device: {device!r}")
    if intent not in {"fast", "parity"}:
        raise ValueError(f"unsupported measurement intent: {intent!r}")
    expected_route = {
        ("native", "bfgs"): "scipy_bfgs",
        ("native", "lbfgs"): "scipy_lbfgsb",
        ("custom", "bfgs"): "custom_bfgs_stepwise",
        ("optax", "lbfgs"): "optax_lbfgs",
    }.get((provider, method))
    if provider == "custom" and method == "lbfgs":
        expected_route = "fused_stepwise" if intent == "fast" else "stepwise"
    if route != expected_route:
        raise ValueError(
            f"solver route {route!r} does not match provider/method "
            f"{provider!r}/{method!r}"
        )
    iterations = _required_int(row, "iterations")
    maxiter = _required_int(row, "maxiter")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if iterations > maxiter:
        raise ValueError("iterations cannot exceed maxiter")
    _finite_number(row.get("initial_objective"), field="initial_objective")
    _finite_number(row.get("final_objective"), field="final_objective")
    for field in (
        "fixture_build_seconds",
        "preparation_seconds",
        "first_execution_seconds",
        "cold_seconds",
        "warm_seconds",
        "scientific_certification_seconds",
        "solver_start_rss_kib",
        "solver_peak_rss_kib",
        "solver_peak_rss_delta_kib",
    ):
        _nonnegative_number(row.get(field), field=field)
    solver_start_rss = _nonnegative_number(
        row.get("solver_start_rss_kib"), field="solver_start_rss_kib"
    )
    solver_peak_rss = _nonnegative_number(
        row.get("solver_peak_rss_kib"), field="solver_peak_rss_kib"
    )
    _nonnegative_number(
        row.get("solver_peak_rss_delta_kib"),
        field="solver_peak_rss_delta_kib",
    )
    if solver_peak_rss < solver_start_rss:
        raise ValueError("solver RSS peak predates the solver boundary")
    _validate_phase_rss(row)
    if device == "gpu" or row.get("peak_vram_mib") is not None:
        _nonnegative_number(row.get("peak_vram_mib"), field="peak_vram_mib")
    identity = row.get("device_identity")
    if not isinstance(identity, dict):
        raise TypeError("device_identity must be a JSON object")
    _validate_device_identity(identity, requested_device=device)
    work = row.get("work_counters")
    if not isinstance(work, dict):
        raise TypeError("work_counters must be a JSON object")
    for field in (
        "accepted_iterations",
        "transfer_calls",
        "transfer_leaves",
        "transfer_bytes",
    ):
        if _required_int(work, field) < 0:
            raise ValueError(f"work_counters.{field} must be nonnegative")
    for field in ("objective_evaluations", "advance_observations"):
        value = _optional_int(work, field)
        if value is not None and value < 0:
            raise ValueError(f"work_counters.{field} must be nonnegative")
    if work["accepted_iterations"] != iterations:
        raise ValueError("work counter iterations do not match raw iterations")
    evaluations = _optional_int(row, "evaluations")
    if evaluations is not None and evaluations < 0:
        raise ValueError("evaluations must be nonnegative")
    if work["objective_evaluations"] != evaluations:
        raise ValueError("work counter evaluations do not match raw evaluations")
    _numeric_array(row.get("initial_parameters"), field="initial_parameters")
    _numeric_array(row.get("final_parameters"), field="final_parameters")
    _recompute_endpoint_certificate(row)
    scientific_observables = row.get("scientific_observables")
    if not isinstance(scientific_observables, dict):
        raise TypeError("scientific_observables must be a JSON object")
    for name, value in scientific_observables.items():
        if not isinstance(name, str):
            raise TypeError("scientific observable names must be strings")
        _finite_number(value, field=f"scientific_observables.{name}")
    diagnostic_artifacts = row.get("diagnostic_artifacts")
    if not isinstance(diagnostic_artifacts, dict):
        raise TypeError("diagnostic_artifacts must be a JSON object")
    if set(diagnostic_artifacts) != {"memory_trace", "trial_trace"}:
        raise ValueError("diagnostic artifact keys do not match schema v8")
    for name, relative in diagnostic_artifacts.items():
        if relative is None:
            if name == "memory_trace" and bind_artifacts:
                raise ValueError("GPU memory artifact reference is missing")
            continue
        if not isinstance(relative, str) or not relative:
            raise TypeError(f"diagnostic artifact {name!r} must be a path or null")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"diagnostic artifact path escapes source run: {relative}")
        if source_run is not None and not (source_run / relative_path).is_file():
            raise FileNotFoundError(
                f"diagnostic artifact is missing: {source_run / relative_path}"
            )
        if name == "memory_trace" and source_run is not None and bind_artifacts:
            memory_artifact = parse_process_gpu_memory_artifact(
                source_run / relative_path
            )
            process_pid = _required_int(row, "process_pid")
            if memory_artifact.provider_pid != process_pid:
                raise ValueError(
                    "GPU memory artifact provider PID differs from measurement"
                )
            identity = cast(dict[str, object], row["device_identity"])
            if memory_artifact.gpu_uuid != identity.get("gpu_uuid"):
                raise ValueError(
                    "GPU memory artifact GPU UUID differs from measurement"
                )
            peak_vram = row.get("peak_vram_mib")
            if peak_vram != memory_artifact.peak_used_memory_mib:
                raise ValueError(
                    "GPU memory artifact peak does not match peak_vram_mib"
                )
            if device == "gpu" and memory_artifact.availability != "available":
                raise ValueError(
                    "GPU measurement memory artifact is unavailable"
                )
            if device == "cpu" and memory_artifact.availability != "unavailable":
                raise ValueError(
                    "CPU measurement memory artifact is unexpectedly available"
                )
        if name == "trial_trace" and source_run is not None and bind_artifacts:
            if case != "boozer" or provider not in {"native", "custom"}:
                raise ValueError("Boozer trial trace is attached to an invalid row")
            if evaluations is None:
                raise TypeError(
                    "Boozer trial trace measurement evaluations must be an integer"
                )
            status = _optional_int(row, "status")
            if status is None:
                raise TypeError(
                    "Boozer trial trace measurement status must be an integer"
                )
            validate_boozer_trial_trace(
                source_run / relative_path,
                expected_provider=cast(str, provider),
                expected_production_route=cast(str, route),
                expected_maxiter=maxiter,
                expected_evaluations=evaluations,
                expected_final_parameters=_numeric_array(
                    row.get("final_parameters"), field="final_parameters"
                ),
                expected_final_objective=float(
                    _finite_number(row.get("final_objective"), field="final_objective")
                ),
                expected_final_gradient_inf_norm=float(
                    _nonnegative_number(
                        row.get("final_gradient_inf_norm"),
                        field="final_gradient_inf_norm",
                    )
                ),
                expected_final_status=status,
            )


def _scientific_qualification(rows: list[dict[str, object]]) -> dict[str, object]:
    endpoint_certificates = [_recompute_endpoint_certificate(row) for row in rows]
    failures: list[str] = []
    if not all(certificate["success"] is True for certificate in endpoint_certificates):
        failures.append("endpoint-certificate")
    cases = sorted({str(row.get("case")) for row in rows})
    comparison_count = 0
    trajectory_comparisons: list[dict[str, object]] = []
    if cases != ["boozer"]:
        failures.append("scientific-case-must-be-boozer")
    expected_observables = {
        "final_boozer_residual",
        "final_non_qs",
        "final_iota",
        "final_volume",
    }
    for case in cases:
        case_rows = [row for row in rows if row.get("case") == case]
        if any(row.get("method") != "bfgs" for row in case_rows):
            failures.append(f"{case}:method-must-be-bfgs")
        if any(row.get("intent") != "parity" for row in case_rows):
            failures.append(f"{case}:intent-must-be-parity")
        if any(row.get("maxiter") != 1000 for row in case_rows):
            failures.append(f"{case}:maxiter-must-be-1000")
        native_rows = [
            row
            for row in case_rows
            if row.get("provider") == "native" and row.get("device") == "cpu"
        ]
        if len(native_rows) != 1:
            failures.append(f"{case}:native-authority-count")
            continue
        if any(row.get("provider") == "optax" for row in case_rows):
            failures.append(f"{case}:optax-not-scientific-authority")
        custom_cpu = [
            row
            for row in case_rows
            if row.get("provider") == "custom" and row.get("device") == "cpu"
        ]
        custom_gpu = [
            row
            for row in case_rows
            if row.get("provider") == "custom" and row.get("device") == "gpu"
        ]
        if len(custom_cpu) != 1:
            failures.append(f"{case}:custom-cpu-count")
        gpu_models = [
            cast(dict[str, object], row["device_identity"]).get("gpu_model")
            for row in custom_gpu
        ]
        if sum("RTX 5090" in str(model) for model in gpu_models) != 1:
            failures.append(f"{case}:rtx-5090-count")
        if sum("A100" in str(model) for model in gpu_models) != 1:
            failures.append(f"{case}:a100-count")
        if len(custom_gpu) != 2:
            failures.append(f"{case}:custom-gpu-count")
        rtx_rows = [
            row
            for row in custom_gpu
            if "RTX 5090"
            in str(cast(dict[str, object], row["device_identity"]).get("gpu_model"))
        ]
        required_diagnostic_rows = native_rows + custom_cpu + rtx_rows
        if any(
            cast(dict[str, object], row["diagnostic_artifacts"]).get("trial_trace")
            is None
            for row in required_diagnostic_rows
        ):
            failures.append(f"{case}:trial-trace-required")
        native = native_rows[0]
        for row in case_rows:
            if row is native:
                continue
            if row.get("initial_parameters") != native.get("initial_parameters"):
                failures.append(f"{case}:initial-parameters")
            comparisons = (
                (
                    "initial_objective",
                    "mirror_single_stage_initial_objective",
                ),
                (
                    "initial_gradient_inf_norm",
                    "mirror_single_stage_initial_gradient",
                ),
                ("final_objective", "mirror_single_stage_final_value"),
                ("final_parameters", "mirror_single_stage_final_parameters"),
                (
                    "final_gradient_inf_norm",
                    "mirror_single_stage_terminal_gradient",
                ),
            )
            for field, tolerance_name in comparisons:
                comparison_count += 1
                if not _close(
                    row.get(field),
                    native.get(field),
                    tolerance_name=tolerance_name,
                    field=field,
                ):
                    failures.append(f"{case}:{field}")
            comparison_count += 1
            native_constraint = native.get("constraint_norm")
            lane_constraint = row.get("constraint_norm")
            if (native_constraint is None) != (lane_constraint is None) or native_constraint is not None and not _close(
                lane_constraint,
                native_constraint,
                tolerance_name="mirror_single_stage_terminal_constraint",
                field="constraint_norm",
            ):
                failures.append(f"{case}:constraint_norm")
            native_status = _optional_int(native, "status")
            lane_status = _optional_int(row, "status")
            native_iterations = _required_int(native, "iterations")
            lane_iterations = _required_int(row, "iterations")
            native_evaluations = _required_int(native, "evaluations")
            lane_evaluations = _required_int(row, "evaluations")
            native_stopping_reason = native.get("stopping_reason")
            lane_stopping_reason = row.get("stopping_reason")
            comparison_count += 4
            if native_status != 0 or lane_status != 0:
                failures.append(f"{case}:raw-status")
            if (
                native_stopping_reason != "converged"
                or lane_stopping_reason != "converged"
            ):
                failures.append(f"{case}:stopping-reason")
            trajectory_comparisons.append(
                {
                    "provider": row.get("provider"),
                    "device": row.get("device"),
                    "gpu_uuid": cast(dict[str, object], row["device_identity"]).get(
                        "gpu_uuid"
                    ),
                    "native_iterations": native_iterations,
                    "lane_iterations": lane_iterations,
                    "iterations_equal": lane_iterations == native_iterations,
                    "native_evaluations": native_evaluations,
                    "lane_evaluations": lane_evaluations,
                    "evaluations_equal": lane_evaluations == native_evaluations,
                    "native_status": native_status,
                    "lane_status": lane_status,
                    "native_stopping_reason": native_stopping_reason,
                    "lane_stopping_reason": lane_stopping_reason,
                }
            )
            observables = row.get("scientific_observables")
            native_observables = native.get("scientific_observables")
            if not isinstance(observables, dict) or not isinstance(
                native_observables, dict
            ):
                raise TypeError("scientific observables must be JSON objects")
            if (
                set(observables) != expected_observables
                or set(native_observables) != expected_observables
            ):
                failures.append(f"{case}:scientific-observable-contract")
            if observables.keys() != native_observables.keys():
                failures.append(f"{case}:scientific-observable-keys")
                continue
            for field in sorted(observables):
                comparison_count += 1
                tolerance_name = (
                    "mirror_surface_invariant"
                    if field in {"final_iota", "final_volume"}
                    else "mirror_single_stage_final_value"
                )
                if not _close(
                    observables[field],
                    native_observables[field],
                    tolerance_name=tolerance_name,
                    field=f"scientific_observables.{field}",
                ):
                    failures.append(f"{case}:scientific-observables:{field}")
    return {
        "passed": not failures,
        "failure_reasons": sorted(set(failures)),
        "comparison_count": comparison_count,
        "trajectory_comparisons": trajectory_comparisons,
    }


def _qualification(
    rows: list[dict[str, object]],
    kind: QualificationKind,
) -> dict[str, object]:
    if kind == "diagnostic":
        return {
            "passed": False,
            "failure_reasons": ["diagnostic-not-promotion"],
            "comparison_count": 0,
        }
    scientific = _scientific_qualification(rows)
    if kind == "scientific":
        return scientific
    performance_route_failures = [
        "performance-custom-lbfgs-requires-fast-fused-stepwise"
        for row in rows
        if row.get("provider") == "custom"
        and row.get("method") == "lbfgs"
        and (row.get("intent") != "fast" or row.get("solver_route") != "fused_stepwise")
    ]
    return {
        "passed": False,
        "failure_reasons": sorted(
            set(
                cast(list[str], scientific["failure_reasons"])
                + performance_route_failures
                + ["performance-qualification-not-implemented"]
            )
        ),
        "comparison_count": scientific["comparison_count"],
    }


def _finite_number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _nonnegative_number(value: object, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _derive_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    providers = sorted({str(row.get("provider")) for row in rows})
    sample_counts: dict[str, int] = {}
    warm_seconds: dict[str, dict[str, float]] = {}
    maximum_rss_kib: dict[str, int] = {}
    maximum_rss_delta_kib: dict[str, int] = {}
    maximum_vram_mib: dict[str, float] = {}
    for provider in providers:
        provider_rows = [row for row in rows if row.get("provider") == provider]
        sample_counts[provider] = len(provider_rows)
        warm_samples = [
            _finite_number(row["warm_seconds"], field="warm_seconds")
            for row in provider_rows
            if "warm_seconds" in row
        ]
        if warm_samples:
            warm_seconds[provider] = {
                "median": float(statistics.median(warm_samples)),
                "minimum": min(warm_samples),
                "maximum": max(warm_samples),
            }
        rss_samples = [
            int(_finite_number(row["peak_rss_kib"], field="peak_rss_kib"))
            for row in provider_rows
            if "peak_rss_kib" in row
        ]
        if rss_samples:
            maximum_rss_kib[provider] = max(rss_samples)
        rss_delta_samples = [
            int(
                _finite_number(
                    row["solver_peak_rss_delta_kib"],
                    field="solver_peak_rss_delta_kib",
                )
            )
            for row in provider_rows
            if "solver_peak_rss_delta_kib" in row
        ]
        if rss_delta_samples:
            maximum_rss_delta_kib[provider] = max(rss_delta_samples)
        vram_samples = [
            _finite_number(row["peak_vram_mib"], field="peak_vram_mib")
            for row in provider_rows
            if row.get("peak_vram_mib") is not None
        ]
        if vram_samples:
            maximum_vram_mib[provider] = max(vram_samples)
    return {
        "sample_counts": sample_counts,
        "warm_seconds": warm_seconds,
        "maximum_rss_kib": maximum_rss_kib,
        "maximum_rss_delta_kib": maximum_rss_delta_kib,
        "maximum_vram_mib": maximum_vram_mib,
    }


def _artifact_inventory_sha256(artifacts: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        artifacts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_artifact_manifest(root: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        artifacts.append({"path": relative, "sha256": _sha256(path)})
    return artifacts


def _copy_runner_tree(run: Path, destination: Path) -> None:
    for source in _iter_files(run):
        relative = source.relative_to(run)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _summary(
    receipt_id: str,
    run_payloads: list[tuple[Path, dict[str, object]]],
    verdict: str,
) -> str:
    lines = [f"# {receipt_id}", "", f"Verdict: `{verdict}`", ""]
    for run, payload in run_payloads:
        rows = _measurement_rows(payload)
        lines.append(f"## {run.name}")
        for row in rows:
            provider = str(row.get("provider", "unknown"))
            status = row.get("status")
            success = row.get("success")
            iterations = row.get("iterations")
            final_objective = row.get("final_objective")
            lines.append(
                f"- `{provider}`: success={success}, status={status}, "
                f"iterations={iterations}, final objective={final_objective}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def publish(
    runs: tuple[Path, ...],
    *,
    environment_lock: Path,
    destination: Path,
    archive_uri: str,
    repo_root: Path,
    qualification_kind: QualificationKind = "diagnostic",
    archive_storage_identity: str | None = None,
) -> Path:
    """Publish runner directories atomically and return the destination."""

    if not runs:
        raise ValueError("at least one --run directory is required")
    if not environment_lock.is_file():
        raise ValueError(f"environment lock does not exist: {environment_lock}")
    if destination.exists():
        raise FileExistsError(f"receipt destination already exists: {destination}")

    run_payloads = [(run, _runner_payload(run)) for run in runs]
    run_names = [run.name for run, _payload in run_payloads]
    if len(set(run_names)) != len(run_names):
        raise ValueError("runner directory names must be unique")
    archive = _archive_path(archive_uri, repo_root=repo_root)
    if destination.resolve() == archive.resolve():
        raise ValueError(
            "receipt archive must be distinct from the tracked destination"
        )
    if archive.exists():
        raise FileExistsError(f"receipt archive destination already exists: {archive}")

    runner_schema_versions = {
        _runner_schema_version(payload) for _run, payload in run_payloads
    }
    if len(runner_schema_versions) != 1:
        raise ValueError("receipt source runs must use one runner schema version")
    runner_schema_version = next(iter(runner_schema_versions))
    receipt_schema_version = (
        _SCHEMA_VERSION
        if runner_schema_version
        in {_LEGACY_RUNNER_SCHEMA_VERSION, _RUNNER_SCHEMA_VERSION}
        else _LEGACY_SCHEMA_VERSION
    )
    if receipt_schema_version == _SCHEMA_VERSION:
        for run, payload in run_payloads:
            _validate_runner_child_provenance(run, payload)
            for row, artifact_root in _v7_rows_with_artifact_roots(run, payload):
                _validate_v7_measurement(
                    row,
                    source_run=artifact_root,
                    bind_artifacts=(
                        _runner_schema_version(payload) == _RUNNER_SCHEMA_VERSION
                    ),
                )
        if qualification_kind == "scientific" and len(run_payloads) > 1:
            for _run, payload in run_payloads:
                runtime_environment = cast(
                    dict[str, object], payload["runtime_environment"]
                )
                expected_backend_mode = (
                    f"jax_{payload['requested_device']}_parity"
                )
                if (
                    runtime_environment.get("SIMSOPT_BACKEND_MODE")
                    != expected_backend_mode
                ):
                    raise ValueError(
                        "SIMSOPT_BACKEND_MODE differs from scientific lane device"
                    )
            jax_versions = {
                payload.get("jax_version") for _run, payload in run_payloads
            }
            if len(jax_versions) != 1:
                raise ValueError("jax_version differs across lanes")
    candidate_commits = {_runner_commit(payload) for _run, payload in run_payloads}
    clean_values = [_runner_clean(payload) for _run, payload in run_payloads]
    rows = [row for _run, payload in run_payloads for row in _measurement_rows(payload)]
    eligible_source = (
        len(candidate_commits) == 1 and all(clean_values) and _all_success(rows)
    )
    qualification = (
        _qualification(rows, qualification_kind)
        if receipt_schema_version == _SCHEMA_VERSION
        else {
            "passed": False,
            "failure_reasons": ["legacy-receipt-not-promotion"],
            "comparison_count": 0,
        }
    )
    verdict = (
        "pass"
        if eligible_source
        and receipt_schema_version == _SCHEMA_VERSION
        and qualification.get("passed") is True
        else (
            "diagnostic-pass-not-promotion"
            if qualification_kind == "diagnostic"
            else "fail"
        )
    )
    lock_sha256 = _sha256(environment_lock)

    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination_parent
    ) as destination_tmp_name, tempfile.TemporaryDirectory(
        prefix=f".{archive.name}.", dir=archive.parent
    ) as archive_tmp_name:
        destination_tmp = Path(destination_tmp_name)
        archive_tmp = Path(archive_tmp_name)
        raw_root = destination_tmp / "raw"
        for run, _payload in run_payloads:
            _copy_runner_tree(run, raw_root / run.name)

        metrics = {
            "schema_version": receipt_schema_version,
            "receipt_id": destination.name,
            "source_runs": [run.name for run, _payload in run_payloads],
            "candidate_commits": sorted(candidate_commits),
            "candidate_worktree_clean": all(clean_values),
            "environment_lock": str(environment_lock),
            "environment_lock_sha256": lock_sha256,
            "measurements": rows,
            "verdict": verdict,
        }
        if receipt_schema_version == _SCHEMA_VERSION:
            metrics.update(
                {
                    "runner_schema_version": runner_schema_version,
                    "qualification_kind": qualification_kind,
                    "qualification": qualification,
                    "derivations": _derive_metrics(rows),
                }
            )
        _write_json(destination_tmp / "metrics.json", metrics)
        (destination_tmp / "summary.md").write_text(
            _summary(destination.name, run_payloads, verdict), encoding="utf-8"
        )
        artifacts = _relative_artifact_manifest(destination_tmp)
        manifest = {
            "schema_version": receipt_schema_version,
            "receipt_id": destination.name,
            "kind": "custom-quasi-newton-runner",
            "candidate_commit": (
                next(iter(candidate_commits)) if len(candidate_commits) == 1 else None
            ),
            "candidate_worktree_clean": all(clean_values),
            "environment_lock": {
                "path": str(environment_lock),
                "sha256": lock_sha256,
            },
            "source_runs": [run.name for run, _payload in run_payloads],
            "archive_uri": archive_uri,
            "artifacts": artifacts,
            "verdict": verdict,
        }
        if receipt_schema_version == _SCHEMA_VERSION:
            manifest.update(
                {
                    "runner_schema_version": runner_schema_version,
                    "qualification_kind": qualification_kind,
                    "qualification": qualification,
                    "expected_sample_counts": _derive_metrics(rows)["sample_counts"],
                    "archive_bundle": {
                        "receipt_id": destination.name,
                        "inventory_sha256": _artifact_inventory_sha256(artifacts),
                        "storage_identity": (archive_storage_identity or archive_uri),
                    },
                }
            )
        _write_json(destination_tmp / "manifest.json", manifest)
        _copy_runner_tree(destination_tmp, archive_tmp)
        destination_tmp.replace(destination)
        archive_tmp.replace(archive)
    return destination


def _validate_v2_semantics(
    receipt: Path,
    manifest: dict[str, object],
) -> None:
    metrics = _json_object(receipt / "metrics.json")
    if metrics.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"receipt metrics schema mismatch: {receipt}")
    if manifest.get("runner_schema_version") not in {
        _LEGACY_RUNNER_SCHEMA_VERSION,
        _RUNNER_SCHEMA_VERSION,
    }:
        raise ValueError(f"receipt runner schema mismatch: {receipt}")
    if metrics.get("runner_schema_version") != manifest.get("runner_schema_version"):
        raise ValueError(f"metrics runner schema mismatch: {receipt}")
    if manifest.get("receipt_id") != receipt.name:
        raise ValueError(f"manifest receipt identity mismatch: {receipt}")
    if metrics.get("receipt_id") != receipt.name:
        raise ValueError(f"metrics receipt identity mismatch: {receipt}")
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list) or not source_runs:
        raise TypeError(f"receipt source runs are invalid: {receipt}")
    if metrics.get("source_runs") != source_runs:
        raise ValueError(f"metrics source runs do not match manifest: {receipt}")
    raw_rows: list[dict[str, object]] = []
    source_payloads: list[dict[str, object]] = []
    for source_run in cast(list[object], source_runs):
        if not isinstance(source_run, str):
            raise TypeError(f"receipt source run is invalid: {receipt}")
        source_path = Path(source_run)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError(f"receipt source run escapes raw root: {source_run}")
        payload = _runner_payload(receipt / "raw" / source_run)
        payload_runner_schema = _runner_schema_version(payload)
        if payload_runner_schema not in {
            _LEGACY_RUNNER_SCHEMA_VERSION,
            _RUNNER_SCHEMA_VERSION,
        }:
            raise ValueError(f"receipt contains a legacy promotion run: {receipt}")
        _validate_runner_child_provenance(receipt / "raw" / source_run, payload)
        source_payloads.append(payload)
        source_rows = _measurement_rows(payload)
        for row, artifact_root in _v7_rows_with_artifact_roots(
            receipt / "raw" / source_run,
            payload,
        ):
            _validate_v7_measurement(
                row,
                source_run=artifact_root,
                bind_artifacts=payload_runner_schema == _RUNNER_SCHEMA_VERSION,
            )
        raw_rows.extend(source_rows)
    if metrics.get("measurements") != raw_rows:
        raise ValueError(f"receipt measurements do not match raw runs: {receipt}")
    derivations = _derive_metrics(raw_rows)
    if metrics.get("derivations") != derivations:
        raise ValueError(f"receipt derivations do not match raw runs: {receipt}")
    if manifest.get("expected_sample_counts") != derivations["sample_counts"]:
        raise ValueError(f"receipt sample counts do not match raw runs: {receipt}")
    qualification_kind = manifest.get("qualification_kind")
    if qualification_kind not in {"diagnostic", "scientific", "performance"}:
        raise ValueError(f"receipt qualification kind is invalid: {receipt}")
    if metrics.get("qualification_kind") != qualification_kind:
        raise ValueError(f"metrics qualification kind does not match: {receipt}")
    recomputed_qualification = _qualification(
        raw_rows,
        cast(QualificationKind, qualification_kind),
    )
    if metrics.get("qualification") != recomputed_qualification:
        raise ValueError(f"receipt qualification does not match raw runs: {receipt}")
    if manifest.get("qualification") != recomputed_qualification:
        raise ValueError(f"manifest qualification does not match raw runs: {receipt}")
    candidate_commits = {_runner_commit(payload) for payload in source_payloads}
    clean_values = [_runner_clean(payload) for payload in source_payloads]
    if metrics.get("candidate_commits") != sorted(candidate_commits):
        raise ValueError(f"metrics candidate commits do not match raw runs: {receipt}")
    if metrics.get("candidate_worktree_clean") is not all(clean_values):
        raise ValueError(f"metrics clean state does not match raw runs: {receipt}")
    expected_candidate = (
        next(iter(candidate_commits)) if len(candidate_commits) == 1 else None
    )
    if manifest.get("candidate_commit") != expected_candidate:
        raise ValueError(
            f"manifest candidate commit does not match raw runs: {receipt}"
        )
    if manifest.get("candidate_worktree_clean") is not all(clean_values):
        raise ValueError(f"manifest clean state does not match raw runs: {receipt}")
    environment_lock = manifest.get("environment_lock")
    if not isinstance(environment_lock, dict):
        raise TypeError(f"manifest environment lock is invalid: {receipt}")
    if metrics.get("environment_lock") != environment_lock.get("path"):
        raise ValueError(f"metrics environment lock path does not match: {receipt}")
    if metrics.get("environment_lock_sha256") != environment_lock.get("sha256"):
        raise ValueError(f"metrics environment lock hash does not match: {receipt}")
    eligible_source = (
        len(candidate_commits) == 1 and all(clean_values) and _all_success(raw_rows)
    )
    expected_verdict = (
        "pass"
        if eligible_source and recomputed_qualification["passed"] is True
        else (
            "diagnostic-pass-not-promotion"
            if qualification_kind == "diagnostic"
            else "fail"
        )
    )
    if metrics.get("verdict") != expected_verdict:
        raise ValueError(f"receipt verdict does not match raw runs: {receipt}")
    if manifest.get("verdict") != expected_verdict:
        raise ValueError(f"manifest verdict does not match raw runs: {receipt}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"manifest artifacts are invalid: {receipt}")
    typed_artifacts = cast(list[dict[str, str]], artifacts)
    if typed_artifacts != _relative_artifact_manifest(receipt):
        raise ValueError(f"manifest artifact inventory is incomplete: {receipt}")
    archive_bundle = manifest.get("archive_bundle")
    if not isinstance(archive_bundle, dict):
        raise TypeError(f"manifest archive bundle is invalid: {receipt}")
    bundle = cast(dict[str, object], archive_bundle)
    if bundle.get("receipt_id") != manifest.get("receipt_id"):
        raise ValueError(f"archive bundle receipt identity mismatch: {receipt}")
    if bundle.get("inventory_sha256") != _artifact_inventory_sha256(typed_artifacts):
        raise ValueError(f"archive bundle inventory mismatch: {receipt}")
    if not isinstance(bundle.get("storage_identity"), str):
        raise TypeError(f"archive storage identity is invalid: {receipt}")
    for row in raw_rows:
        for field in (
            "preparation_seconds",
            "first_execution_seconds",
            "cold_seconds",
            "warm_seconds",
            "initial_objective",
            "initial_gradient_inf_norm",
            "final_objective",
            "final_gradient_inf_norm",
            "solver_start_rss_kib",
            "solver_peak_rss_kib",
            "solver_peak_rss_delta_kib",
            "scientific_certification_seconds",
        ):
            _finite_number(row.get(field), field=field)
        _recompute_endpoint_certificate(row)
        scientific_observables = row.get("scientific_observables")
        if not isinstance(scientific_observables, dict):
            raise TypeError(f"scientific observables are invalid: {receipt}")
        for name, value in scientific_observables.items():
            if not isinstance(name, str):
                raise TypeError(f"scientific observable name is invalid: {receipt}")
            _finite_number(value, field=f"scientific_observables.{name}")
        preparation_seconds = _finite_number(
            row["preparation_seconds"], field="preparation_seconds"
        )
        first_execution_seconds = _finite_number(
            row["first_execution_seconds"], field="first_execution_seconds"
        )
        cold_seconds = _finite_number(row["cold_seconds"], field="cold_seconds")
        if not math.isclose(
            cold_seconds,
            preparation_seconds + first_execution_seconds,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"cold timing derivation mismatch: {receipt}")
        if not isinstance(row.get("solver_route"), str):
            raise TypeError(f"solver route is invalid: {receipt}")
        if not isinstance(row.get("device_identity"), dict):
            raise TypeError(f"device identity is invalid: {receipt}")
        if not isinstance(row.get("work_counters"), dict):
            raise TypeError(f"work counters are invalid: {receipt}")
        diagnostic_artifacts = row.get("diagnostic_artifacts")
        if not isinstance(diagnostic_artifacts, dict):
            raise TypeError(f"diagnostic artifact references are invalid: {receipt}")
        for artifact_name, relative in diagnostic_artifacts.items():
            if relative is None:
                continue
            if not isinstance(artifact_name, str) or not isinstance(relative, str):
                raise TypeError(f"diagnostic artifact reference is invalid: {receipt}")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"diagnostic artifact path escapes source run: {relative}"
                )


def _validate_receipt(
    receipt: Path,
    *,
    repo_root: Path,
    archive_root: Path | None = None,
) -> None:
    manifest = _json_object(receipt / "manifest.json")
    schema_version = manifest.get("schema_version")
    if schema_version not in _LEGACY_SCHEMA_VERSIONS | {_SCHEMA_VERSION}:
        raise ValueError(f"unsupported receipt schema version: {schema_version!r}")
    if schema_version == _SCHEMA_VERSION:
        _validate_v2_semantics(receipt, manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"manifest artifacts are invalid: {receipt}")
    archive_uri = manifest.get("archive_uri")
    if not isinstance(archive_uri, str):
        raise TypeError(f"manifest archive URI is invalid: {receipt}")
    receipt_id = manifest.get("receipt_id")
    if archive_root is not None:
        if not isinstance(receipt_id, str):
            raise TypeError(f"manifest receipt ID is invalid: {receipt}")
        receipt_id_path = Path(receipt_id)
        if (
            receipt_id_path.is_absolute()
            or ".." in receipt_id_path.parts
            or len(receipt_id_path.parts) != 1
        ):
            raise ValueError(f"manifest receipt ID escapes archive root: {receipt_id}")
        archive = archive_root / receipt_id
    else:
        archive = _archive_path(archive_uri, repo_root=repo_root)
    archived_manifest = _json_object(archive / "manifest.json")
    if archived_manifest != manifest:
        raise ValueError(
            f"archived manifest does not match tracked manifest: {archive}"
        )
    environment_lock = manifest.get("environment_lock")
    if isinstance(environment_lock, dict):
        lock_record = cast(dict[str, object], environment_lock)
        lock_path_value = lock_record.get("path")
        lock_sha256 = lock_record.get("sha256")
        if not isinstance(lock_path_value, str) or not isinstance(lock_sha256, str):
            raise TypeError(f"manifest environment lock is invalid: {receipt}")
        lock_path = Path(lock_path_value)
        if not lock_path.is_absolute():
            lock_path = repo_root / lock_path
        if not lock_path.is_file():
            raise FileNotFoundError(f"environment lock missing: {lock_path}")
        if _sha256(lock_path) != lock_sha256:
            raise ValueError(f"environment lock checksum mismatch: {lock_path}")
    for raw_artifact in cast(list[object], artifacts):
        if not isinstance(raw_artifact, dict):
            raise TypeError(f"manifest artifact entry is invalid: {receipt}")
        artifact = cast(dict[str, object], raw_artifact)
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError(f"manifest artifact entry is invalid: {receipt}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"manifest artifact path escapes receipt: {relative}")
        tracked = receipt / relative_path
        archived = archive / relative_path
        for path in (tracked, archived):
            if not path.is_file():
                raise FileNotFoundError(f"receipt artifact missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"receipt artifact checksum mismatch: {path} {actual} != {expected}"
                )


def validate_all(
    root: Path,
    *,
    repo_root: Path,
    archive_root: Path | None = None,
) -> int:
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"no receipt manifests found under {root}")
    for manifest in manifests:
        _validate_receipt(
            manifest.parent,
            repo_root=repo_root,
            archive_root=archive_root,
        )
    print(json.dumps({"validated": len(manifests), "root": str(root)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--run", type=Path, action="append", required=True)
    publish_parser.add_argument("--environment-lock", type=Path, required=True)
    publish_parser.add_argument("--destination", type=Path, required=True)
    publish_parser.add_argument("--archive-uri", required=True)
    publish_parser.add_argument(
        "--qualification-kind",
        choices=("diagnostic", "scientific", "performance"),
        default="diagnostic",
    )
    publish_parser.add_argument("--archive-storage-identity")
    publish_parser.add_argument("--repo-root", type=Path, default=Path.cwd())

    validate_parser = subparsers.add_parser("validate-all")
    validate_parser.add_argument("--root", type=Path, required=True)
    validate_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    validate_parser.add_argument("--archive-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        destination = publish(
            tuple(args.run),
            environment_lock=args.environment_lock,
            destination=args.destination,
            archive_uri=args.archive_uri,
            repo_root=args.repo_root,
            qualification_kind=args.qualification_kind,
            archive_storage_identity=args.archive_storage_identity,
        )
        print(destination)
        return 0
    return validate_all(
        args.root,
        repo_root=args.repo_root,
        archive_root=args.archive_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
