"""Run one supervised, provenance-bound CFS-GNTR1 RTX 5090 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import jax
import jaxlib
import numpy as np

jax.config.update("jax_enable_x64", True)

from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonAttemptOutcome,
    ProjectedGaussNewtonStatus,
)
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedSteihaugTermination,
)
from simsopt_jax.solve.fullspace_gauss_newton_trust_region import (
    CFS_GNTR1_OPTIONS,
    CfsGntr1Result,
    prepare_cfs_gntr1,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)

from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryMonitor,
    ProcessGpuMemoryResult,
)
from benchmarks.single_stage_fullspace_gntr_receipt import (
    EQUALITY_SIZE,
    GPU_UUID,
    MAXIMUM_MEMORY_FRACTION,
    OBJECTIVE_RESIDUAL_SIZE,
    PLAN_SHA256,
    ROUTE,
    SCHEMA_VERSION,
    STATE_SIZE,
    canonical_json_bytes,
    derive_gntr_gate,
)

WORKER_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-gntr1-worker-v1"
WORKER_MODULE: Final = "benchmarks.run_single_stage_fullspace_gauss_newton_trust_region"
SUPERVISOR_TIMEOUT_SECONDS: Final = 1200.0
SOURCE_PATHS: Final = (
    Path("benchmarks/run_single_stage_fullspace_gauss_newton_trust_region.py"),
    Path("benchmarks/single_stage_fullspace_gntr_receipt.py"),
    Path("docs/single_stage_jax_gpu_gauss_newton_trust_region_implementation_plan.md"),
    Path("src/simsopt_jax/core/__init__.py"),
    Path("src/simsopt_jax/core/quasisymmetry.py"),
    Path("src/simsopt_jax/geo/optimizers/dense_sqp.py"),
    Path("src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py"),
    Path("src/simsopt_jax/geo/optimizers/projected_hvp_trust_region.py"),
    Path("src/simsopt_jax/objectives/single_stage_fullspace.py"),
    Path("src/simsopt_jax/objectives/single_stage_fullspace_residuals.py"),
    Path("src/simsopt_jax/solve/fullspace.py"),
    Path("src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py"),
    Path("src/simsopt_jax/solve/fullspace_sqp.py"),
    Path("tests/benchmarks/test_single_stage_fullspace_gauss_newton_trust_region.py"),
    Path("tests/benchmarks/test_single_stage_fullspace_gntr_receipt.py"),
    Path("tests/geo/test_projected_gauss_newton_trust_region.py"),
    Path("tests/geo/test_fullspace_gauss_newton_trust_region.py"),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tracked_paths(repo_root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = {Path(field.decode()) for field in completed.stdout.split(b"\0") if field}
    paths.update(SOURCE_PATHS)
    return tuple(
        sorted(
            path
            for path in paths
            if (repo_root / path).is_file()
            and (
                path.name == ".env.example"
                or (path.name != ".env" and not path.name.startswith(".env."))
            )
        )
    )


def _source_manifest(repo_root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.as_posix(),
            "sha256": _sha256((repo_root / path).read_bytes()),
            "size_bytes": (repo_root / path).stat().st_size,
        }
        for path in _tracked_paths(repo_root)
    ]


def _git_output(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def _worker_argv() -> tuple[str, ...]:
    """Return the package-safe worker invocation from the repository root."""

    return (sys.executable, "-m", WORKER_MODULE, "--worker")


def _gpu_identity() -> dict[str, object]:
    output = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = [
        tuple(field.strip() for field in line.split(","))
        for line in output.splitlines()
    ]
    matching = [row for row in rows if row[1] == GPU_UUID]
    if len(matching) != 1:
        raise ValueError("the frozen RTX 5090 UUID was not uniquely available")
    index, uuid, name, total_mib, driver = matching[0]
    if "RTX 5090" not in name:
        raise ValueError("the frozen GPU UUID does not identify an RTX 5090")
    return {
        "physical_index": int(index),
        "uuid": uuid,
        "name": name,
        "total_memory_mib": int(total_mib),
        "driver_version": driver,
    }


def _float(value: object) -> float | None:
    scalar = float(np.asarray(value))
    return scalar if np.isfinite(scalar) else None


def _int(value: object) -> int:
    return int(np.asarray(value))


def _bool(value: object) -> bool:
    return bool(np.asarray(value))


def _array_identity(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _sha256(array.tobytes()),
    }


def _runtime_payload(gpu: dict[str, object]) -> dict[str, object]:
    devices = jax.devices()
    return {
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "device": str(devices[0]) if len(devices) == 1 else None,
        "gpu": gpu,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    }


def _endpoint_payload(endpoint: object) -> dict[str, object]:
    return {
        "physical_objective": _float(endpoint.physical_objective),
        "raw_feasibility_inf": _float(endpoint.raw_constraint_infinity_norm),
        "scaled_feasibility_inf": _float(endpoint.scaled_constraint_infinity_norm),
        "raw_kkt_stationarity_inf": _float(endpoint.raw_kkt_stationarity_infinity_norm),
        "physical_state": _array_identity(endpoint.physical_state),
        "scaled_multipliers": _array_identity(endpoint.scaled_multipliers),
        "raw_multipliers": _array_identity(endpoint.raw_multipliers),
        "all_finite": _bool(endpoint.all_finite),
    }


def _attempt_payloads(result: CfsGntr1Result) -> list[dict[str, object]]:
    optimizer = result.optimizer_result
    attempts = _int(optimizer.attempts)
    history = optimizer.history
    floating_fields = (
        "current_objective",
        "current_feasibility_inf",
        "current_stationarity_inf",
        "candidate_objective",
        "candidate_feasibility_inf",
        "actual_reduction",
        "predicted_reduction",
        "reduction_ratio",
        "trust_radius",
        "next_trust_radius",
        "tangent_step_norm",
        "correction_norm",
        "applied_step_norm",
        "correction_step_ratio",
        "corrected_radius_ratio",
        "terminal_normalized_curvature",
        "residual_value_defect",
        "residual_gradient_defect",
        "hvp_symmetry_defect",
        "probe_normalized_curvature",
        "direction_rotation",
        "correction_relative_residual",
        "correction_forward_error_bound",
        "trial_gram_factorization_relative_residual",
        "trial_gram_solve_relative_residual",
        "current_projection_tangency_relative_residual",
        "current_projection_solve_relative_residual",
        "current_projection_forward_error_bound",
        "steihaug_tangency_relative_residual",
        "steihaug_final_projected_residual_norm",
        "steihaug_projected_residual_target",
        "steihaug_residual_projection_tangency_relative_residual",
        "steihaug_residual_projection_solve_relative_residual",
        "steihaug_residual_projection_forward_error_bound",
    )
    payloads: list[dict[str, object]] = []
    for index in range(attempts):
        payload: dict[str, object] = {
            "attempt": index + 1,
            "outcome": ProjectedGaussNewtonAttemptOutcome(
                _int(history.outcome[index])
            ).name,
            "accepted_step_number": _int(history.accepted_step_number[index]),
            "steihaug_iterations": _int(history.steihaug_iterations[index]),
            "steihaug_hvp_evaluations": _int(history.steihaug_hvp_evaluations[index]),
            "steihaug_termination": ProjectedSteihaugTermination(
                _int(history.steihaug_termination[index])
            ).name,
            "steihaug_hit_boundary": _bool(history.steihaug_hit_boundary[index]),
        }
        payload.update(
            {field: _float(getattr(history, field)[index]) for field in floating_fields}
        )
        payloads.append(payload)
    return payloads


def _accepted_states_payload(
    result: CfsGntr1Result,
    attempts: list[dict[str, object]],
) -> list[dict[str, object]]:
    optimizer = result.optimizer_result
    if not attempts:
        return []
    states: list[dict[str, object]] = [
        {
            "accepted_step": 0,
            "physical_objective": attempts[0]["current_objective"],
            "scaled_feasibility_inf": attempts[0]["current_feasibility_inf"],
            "scaled_stationarity_inf": attempts[0]["current_stationarity_inf"],
        }
    ]
    for index, attempt in enumerate(attempts):
        if attempt["outcome"] != "ACCEPTED":
            continue
        next_stationarity = (
            attempts[index + 1]["current_stationarity_inf"]
            if index + 1 < len(attempts)
            else _float(optimizer.scaled_stationarity_inf)
        )
        states.append(
            {
                "accepted_step": attempt["accepted_step_number"],
                "physical_objective": attempt["candidate_objective"],
                "scaled_feasibility_inf": attempt["candidate_feasibility_inf"],
                "scaled_stationarity_inf": next_stationarity,
            }
        )
    return states


def _route_result_payload(result: CfsGntr1Result) -> dict[str, object]:
    optimizer = result.optimizer_result
    attempts = _attempt_payloads(result)
    final_certificate = optimizer.final_certificate
    return {
        "schema_version": result.schema_version,
        "route": result.route,
        "optimizer": {
            "status": ProjectedGaussNewtonStatus(_int(optimizer.status)).name,
            "fatal": _bool(optimizer.fatal),
            "bounded_complete": _bool(optimizer.bounded_complete),
            "mechanism_exercised": _bool(optimizer.mechanism_exercised),
            "accepted_steps": _int(optimizer.accepted_steps),
            "attempts": _int(optimizer.attempts),
            "retryable_rejections": _int(optimizer.retryable_rejections),
            "final_trust_radius": _float(optimizer.trust_radius),
            "final_scaled_stationarity_inf": _float(optimizer.scaled_stationarity_inf),
            "final_scaled_feasibility_inf": _float(optimizer.scaled_feasibility_inf),
            "all_accepted_states_finite": _bool(optimizer.all_accepted_states_finite),
            "all_finite": _bool(optimizer.all_finite),
            "usable": _bool(optimizer.usable),
        },
        "final_certificate": {
            "coordinates_finite": _bool(final_certificate.coordinates_finite),
            "residual_value_defect": _float(final_certificate.residual_value_defect),
            "residual_gradient_defect": _float(
                final_certificate.residual_gradient_defect
            ),
            "hvp_symmetry_defect": _float(final_certificate.hvp_symmetry_defect),
            "probe_normalized_curvature": _float(
                final_certificate.probe_normalized_curvature
            ),
            "gram_factorization_relative_residual": _float(
                final_certificate.gram_factorization_relative_residual
            ),
            "multiplier_relative_residual": _float(
                final_certificate.multiplier_relative_residual
            ),
            "multiplier_forward_error_bound": _float(
                final_certificate.multiplier_forward_error_bound
            ),
            "projection_tangency_relative_residual": _float(
                final_certificate.projection_tangency_relative_residual
            ),
            "projection_solve_relative_residual": _float(
                final_certificate.projection_solve_relative_residual
            ),
            "projection_forward_error_bound": _float(
                final_certificate.projection_forward_error_bound
            ),
            "all_finite": _bool(final_certificate.all_finite),
            "certified": _bool(final_certificate.certified),
        },
        "attempts": attempts,
        "accepted_states": _accepted_states_payload(result, attempts),
        "initial_endpoint": _endpoint_payload(result.initial_endpoint),
        "final_endpoint": _endpoint_payload(result.final_endpoint),
        "residual_value_defect": _float(result.residual_value_defect),
        "residual_gradient_relative_defect": _float(
            result.residual_gradient_relative_defect
        ),
        "stationarity_scaling_relative_defect": _float(
            result.stationarity_scaling_relative_defect
        ),
        "objective_residual_size": _int(result.objective_residual_size),
        "state_size": _int(result.state_size),
        "equality_size": _int(result.equality_size),
        "bootstrap_matches_initial": _bool(result.bootstrap_matches_initial),
        "dimensions_valid": _bool(result.dimensions_valid),
        "fp64_valid": _bool(result.fp64_valid),
        "residual_contract_valid": _bool(result.residual_contract_valid),
        "current_state_certificates_valid": _bool(
            result.current_state_certificates_valid
        ),
        "solver_result_consistent": _bool(result.solver_result_consistent),
        "all_finite": _bool(result.all_finite),
        "canary_usable_before_resource_gate": _bool(result.canary_usable),
    }


def _failure_payload(
    *,
    stage: str,
    error: BaseException,
    gpu: dict[str, object],
    bootstrap: dict[str, object] | None,
    compile_seconds: float | None,
    execution_seconds: float | None,
) -> dict[str, object]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "runtime": _runtime_payload(gpu),
        "bootstrap": bootstrap,
        "execution": {
            "stage": stage,
            "lower_compile_attempted": stage == "LOWER_OR_COMPILE",
            "lower_compile_succeeded": stage != "LOWER_OR_COMPILE",
            "bounded_solve_attempted": stage in {"EXECUTION", "FINALIZATION"},
            "bounded_solve_completed": stage == "FINALIZATION",
            "endpoint_finalization_completed": False,
            "error_type": type(error).__name__,
            "error_sha256": _sha256(str(error).encode()),
        },
        "timing": {
            "lower_compile_seconds": compile_seconds,
            "synchronized_solve_seconds": execution_seconds,
            "endpoint_finalize_seconds": None,
        },
        "transfer_audit": {
            "initial_h2d_calls": 1 if bootstrap is not None else 0,
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "final_d2h_calls": 0,
            "timed_execution_transfer_guard": "disallow",
        },
        "route_result": None,
    }


def _run_worker() -> dict[str, object]:
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("CFS-GNTR1 requires exactly one JAX GPU")
    if not bool(jax.config.jax_enable_x64):
        raise ValueError("CFS-GNTR1 requires JAX x64")
    gpu = _gpu_identity()
    bootstrap = build_single_stage_fullspace_bootstrap()
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))
    bootstrap_payload = {
        "state": _array_identity(host_z),
        "state_size": int(host_z.size),
        "equality_size": int(bootstrap.problem.exact_mask_indices.size + 1),
        "objective_residual_size": OBJECTIVE_RESIDUAL_SIZE,
        "dtype": str(host_z.dtype),
    }
    compile_start = time.perf_counter_ns()
    try:
        prepared = prepare_cfs_gntr1(device_problem, device_z, device_z)
        jax.block_until_ready(prepared.initial_optimizer_coordinates)
    except Exception as error:  # noqa: BLE001 - terminal compile receipt is required.
        return _failure_payload(
            stage="LOWER_OR_COMPILE",
            error=error,
            gpu=gpu,
            bootstrap=bootstrap_payload,
            compile_seconds=(time.perf_counter_ns() - compile_start) / 1.0e9,
            execution_seconds=None,
        )
    compile_seconds = (time.perf_counter_ns() - compile_start) / 1.0e9
    execution_start = time.perf_counter_ns()
    try:
        with jax.transfer_guard("disallow"):
            optimizer_result = prepared.run_solver()
            jax.block_until_ready(optimizer_result)
    except Exception as error:  # noqa: BLE001 - terminal resource receipt is required.
        return _failure_payload(
            stage="EXECUTION",
            error=error,
            gpu=gpu,
            bootstrap=bootstrap_payload,
            compile_seconds=compile_seconds,
            execution_seconds=(time.perf_counter_ns() - execution_start) / 1.0e9,
        )
    execution_seconds = (time.perf_counter_ns() - execution_start) / 1.0e9
    finalize_start = time.perf_counter_ns()
    try:
        finalized = prepared.finalize_result(optimizer_result)
        jax.block_until_ready(finalized)
    except Exception as error:  # noqa: BLE001 - terminal resource receipt is required.
        return _failure_payload(
            stage="FINALIZATION",
            error=error,
            gpu=gpu,
            bootstrap=bootstrap_payload,
            compile_seconds=compile_seconds,
            execution_seconds=execution_seconds,
        )
    finalize_seconds = (time.perf_counter_ns() - finalize_start) / 1.0e9
    host_result = jax.device_get(finalized)
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "runtime": _runtime_payload(gpu),
        "bootstrap": bootstrap_payload,
        "execution": {
            "stage": "COMPLETE",
            "lower_compile_attempted": True,
            "lower_compile_succeeded": True,
            "bounded_solve_attempted": True,
            "bounded_solve_completed": True,
            "endpoint_finalization_completed": True,
            "error_type": None,
            "error_sha256": None,
        },
        "timing": {
            "lower_compile_seconds": compile_seconds,
            "synchronized_solve_seconds": execution_seconds,
            "endpoint_finalize_seconds": finalize_seconds,
        },
        "transfer_audit": {
            "initial_h2d_calls": 1,
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "final_d2h_calls": 1,
            "timed_execution_transfer_guard": "disallow",
        },
        "route_result": _route_result_payload(host_result),
    }


def _memory_payload(
    measurement: object,
    *,
    gpu: dict[str, object],
    provider_pid: int,
    monitor_error: BaseException | None,
) -> dict[str, object]:
    if isinstance(measurement, ProcessGpuMemoryResult):
        peak_fraction = measurement.peak_used_memory_mib / int(gpu["total_memory_mib"])
        return {
            "gpu_uuid": measurement.gpu_uuid,
            "provider_pid": measurement.provider_pid,
            "target_pid_observed": True,
            "sample_count": len(measurement.samples),
            "peak_used_memory_mib": measurement.peak_used_memory_mib,
            "peak_memory_fraction": peak_fraction,
            "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
            "monitor_error_type": None,
            "monitor_error_sha256": None,
        }
    return {
        "gpu_uuid": GPU_UUID,
        "provider_pid": provider_pid,
        "target_pid_observed": False,
        "sample_count": 0,
        "peak_used_memory_mib": None,
        "peak_memory_fraction": None,
        "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
        "monitor_error_type": (
            None if monitor_error is None else type(monitor_error).__name__
        ),
        "monitor_error_sha256": (
            None if monitor_error is None else _sha256(str(monitor_error).encode())
        ),
    }


def _supervisor_failure_worker(
    *,
    gpu: dict[str, object],
    stage: str,
    error_type: str,
    error_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "runtime": {
            "python": sys.version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": "unobserved",
            "device": None,
            "gpu": gpu,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        },
        "bootstrap": {
            "state": None,
            "state_size": STATE_SIZE,
            "equality_size": EQUALITY_SIZE,
            "objective_residual_size": OBJECTIVE_RESIDUAL_SIZE,
            "dtype": "float64",
        },
        "execution": {
            "stage": stage,
            "lower_compile_attempted": None,
            "lower_compile_succeeded": None,
            "bounded_solve_attempted": None,
            "bounded_solve_completed": None,
            "endpoint_finalization_completed": None,
            "error_type": error_type,
            "error_sha256": _sha256(error_bytes),
        },
        "timing": {
            "lower_compile_seconds": None,
            "synchronized_solve_seconds": None,
            "endpoint_finalize_seconds": None,
        },
        "transfer_audit": {
            "initial_h2d_calls": 0,
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "final_d2h_calls": 0,
            "timed_execution_transfer_guard": "disallow",
        },
        "route_result": None,
    }


def run(output_root: Path) -> dict[str, object]:
    """Supervise exactly one worker and seal its terminal receipt."""

    repo_root = Path(__file__).resolve().parents[1]
    output = output_root.absolute()
    output.mkdir(parents=True, exist_ok=False)
    gpu = _gpu_identity()
    source_manifest = _source_manifest(repo_root)
    manifest_bytes = canonical_json_bytes(source_manifest)
    started = time.perf_counter_ns()
    worker = subprocess.Popen(
        _worker_argv(),
        cwd=repo_root,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    monitor = ProcessGpuMemoryMonitor(
        gpu_uuid=GPU_UUID,
        provider_pid=worker.pid,
        interval_seconds=0.1,
    )
    monitor.start()
    try:
        stdout, stderr = worker.communicate(timeout=SUPERVISOR_TIMEOUT_SECONDS)
        timed_out = False
    except subprocess.TimeoutExpired:
        worker.kill()
        stdout, stderr = worker.communicate()
        timed_out = True
    try:
        measurement = monitor.finish()
        monitor_error: BaseException | None = None
    except Exception as error:  # noqa: BLE001 - resource failure must be receipted.
        measurement = None
        monitor_error = error
    wall_seconds = (time.perf_counter_ns() - started) / 1.0e9
    if timed_out:
        worker_payload = _supervisor_failure_worker(
            gpu=gpu,
            stage="SUPERVISOR_TIMEOUT",
            error_type="TimeoutExpired",
            error_bytes=stderr,
        )
    elif worker.returncode != 0:
        worker_payload = _supervisor_failure_worker(
            gpu=gpu,
            stage="WORKER_FAILURE",
            error_type="WorkerProcessError",
            error_bytes=stderr,
        )
    else:
        try:
            decoded = json.loads(stdout)
            if not isinstance(decoded, dict):
                raise TypeError("CFS-GNTR1 worker output must be a JSON object")
            if canonical_json_bytes(decoded) != stdout:
                raise ValueError("CFS-GNTR1 worker output is not canonical strict JSON")
            worker_payload = decoded
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            worker_payload = _supervisor_failure_worker(
                gpu=gpu,
                stage="WORKER_PROTOCOL_FAILURE",
                error_type=type(error).__name__,
                error_bytes=stdout + stderr,
            )
    post_manifest = _source_manifest(repo_root)
    post_manifest_sha256 = _sha256(canonical_json_bytes(post_manifest))
    source = {
        "git_head": _git_output(repo_root, "rev-parse", "HEAD").decode().strip(),
        "git_status_sha256": _sha256(
            _git_output(repo_root, "status", "--porcelain=v1", "-z")
        ),
        "manifest_sha256": _sha256(manifest_bytes),
        "post_manifest_sha256": post_manifest_sha256,
        "manifest_entry_count": len(source_manifest),
        "pre_post_manifest_identical": source_manifest == post_manifest,
    }
    timing = dict(worker_payload["timing"])
    timing["supervised_wall_seconds"] = wall_seconds
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "route": ROUTE,
        "plan_sha256": PLAN_SHA256,
        "terminal_status": "CANARY_NOT_USABLE",
        "promotion_eligible": False,
        "source": source,
        "runtime": worker_payload["runtime"],
        "bootstrap": worker_payload["bootstrap"],
        "policy": {
            "maximum_accepted_steps": CFS_GNTR1_OPTIONS.maximum_accepted_steps,
            "maximum_attempts": CFS_GNTR1_OPTIONS.maximum_attempts,
            "initial_trust_radius": CFS_GNTR1_OPTIONS.initial_trust_radius,
            "minimum_trust_radius": CFS_GNTR1_OPTIONS.minimum_trust_radius,
            "maximum_trust_radius": CFS_GNTR1_OPTIONS.maximum_trust_radius,
            "maximum_steihaug_iterations": (
                CFS_GNTR1_OPTIONS.maximum_steihaug_iterations
            ),
            "projected_residual_tolerance": (
                CFS_GNTR1_OPTIONS.projected_residual_tolerance
            ),
            "linear_residual_tolerance": (CFS_GNTR1_OPTIONS.linear_residual_tolerance),
            "corrected_feasibility_tolerance": (
                CFS_GNTR1_OPTIONS.corrected_feasibility_tolerance
            ),
            "forward_error_tolerance": CFS_GNTR1_OPTIONS.forward_error_tolerance,
            "residual_value_defect_tolerance": (
                CFS_GNTR1_OPTIONS.residual_value_defect_tolerance
            ),
            "residual_gradient_defect_tolerance": (
                CFS_GNTR1_OPTIONS.residual_gradient_defect_tolerance
            ),
            "normalized_curvature_tolerance": (
                CFS_GNTR1_OPTIONS.normalized_curvature_tolerance
            ),
            "maximum_correction_step_ratio": (
                CFS_GNTR1_OPTIONS.maximum_correction_step_ratio
            ),
            "maximum_corrected_radius_excess": (
                CFS_GNTR1_OPTIONS.maximum_corrected_radius_excess
            ),
            "mechanism_rotation_threshold": (
                CFS_GNTR1_OPTIONS.mechanism_rotation_threshold
            ),
        },
        "execution": worker_payload["execution"],
        "timing": timing,
        "transfer_audit": worker_payload["transfer_audit"],
        "memory": _memory_payload(
            measurement,
            gpu=gpu,
            provider_pid=worker.pid,
            monitor_error=monitor_error,
        ),
        "route_result": worker_payload["route_result"],
        "gate": {},
    }
    gate = derive_gntr_gate(payload)
    payload["gate"] = gate
    payload["terminal_status"] = gate["terminal_status"]
    manifest_path = output / "source-manifest.json"
    result_path = output / "result.json"
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    with result_path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    manifest_path.chmod(0o444)
    result_path.chmod(0o444)
    output.chmod(0o555)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.worker:
        print(canonical_json_bytes(_run_worker()).decode(), end="")
        return 0
    if arguments.output is None:
        parser.error("--output is required for the supervisor")
    payload = run(arguments.output)
    print(canonical_json_bytes(payload).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
