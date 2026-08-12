"""Run the provenance-bound CFS-GN1 Gauss--Newton GPU canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Final

import jax
import jaxlib
import numpy as np

jax.config.update("jax_enable_x64", True)

from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedSteihaugTermination,
)
from simsopt_jax.solve.fullspace_gauss_newton_canary import (
    FullSpaceGaussNewtonCanaryEndpoint,
    FullSpaceGaussNewtonCanaryResult,
    run_fullspace_gauss_newton_canary,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)

from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryMonitor,
    ProcessGpuMemoryResult,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-gauss-newton-canary-v1"
ROUTE: Final = "CFS-GN1"
GPU_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
STATE_SIZE: Final = 716
EQUALITY_SIZE: Final = 255
OBJECTIVE_RESIDUAL_SIZE: Final = 2110
TRUST_RADIUS: Final = 2.0**-10
MAXIMUM_CG_ITERATIONS: Final = 32
MAXIMUM_MEMORY_FRACTION: Final = 0.8
VALUE_RECONSTRUCTION_TOLERANCE: Final = 1.0e-12
GRADIENT_RECONSTRUCTION_TOLERANCE: Final = 1.0e-10
GN_SYMMETRY_TOLERANCE: Final = 1.0e-10
GN_NORMALIZED_PSD_LOWER_BOUND: Final = -1.0e-10
SOURCE_PATHS: Final = (
    Path("benchmarks/run_single_stage_fullspace_gauss_newton_canary.py"),
    Path("docs/single_stage_jax_gpu_gauss_newton_canary_implementation_plan.md"),
    Path("src/simsopt_jax/core/__init__.py"),
    Path("src/simsopt_jax/core/quasisymmetry.py"),
    Path("src/simsopt_jax/geo/optimizers/projected_hvp_trust_region.py"),
    Path("src/simsopt_jax/objectives/single_stage_fullspace.py"),
    Path("src/simsopt_jax/objectives/single_stage_fullspace_residuals.py"),
    Path("src/simsopt_jax/solve/fullspace_gauss_newton_canary.py"),
    Path("tests/benchmarks/test_single_stage_fullspace_gauss_newton_canary.py"),
    Path("tests/geo/test_fullspace_gauss_newton_canary.py"),
    Path("tests/geo/test_projected_hvp_trust_region.py"),
    Path("tests/jax/objectives/test_single_stage_fullspace_core.py"),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


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


def _finite_at_most(value: object, upper_bound: float) -> bool:
    scalar = float(np.asarray(value))
    return bool(np.isfinite(scalar) and scalar <= upper_bound)


def _finite_at_least(value: object, lower_bound: float) -> bool:
    scalar = float(np.asarray(value))
    return bool(np.isfinite(scalar) and scalar >= lower_bound)


def _array_identity(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _sha256(array.tobytes()),
    }


def _termination_name(value: object) -> str:
    return ProjectedSteihaugTermination(_int(value)).name


def _endpoint_payload(
    endpoint: FullSpaceGaussNewtonCanaryEndpoint,
) -> dict[str, object]:
    optimizer = endpoint.optimizer
    physical = endpoint.physical
    return {
        "physical_objective": _float(physical.physical_objective),
        "raw_constraint_infinity_norm": _float(physical.raw_constraint_infinity_norm),
        "scaled_constraint_infinity_norm": _float(
            physical.scaled_constraint_infinity_norm
        ),
        "raw_kkt_stationarity_infinity_norm": _float(
            physical.raw_kkt_stationarity_infinity_norm
        ),
        "scaled_stationarity_infinity_norm": _float(optimizer.scaled_stationarity_inf),
        "scaled_feasibility_infinity_norm": _float(optimizer.scaled_feasibility_inf),
        "tangent_step_norm": _float(optimizer.tangent_step_norm),
        "model_step_norm": _float(optimizer.model_step_norm),
        "applied_step_norm": _float(optimizer.applied_step_norm),
        "predicted_reduction": _float(optimizer.predicted_reduction),
        "tangency_relative_residual": _float(optimizer.tangency_relative_residual),
        "cg_iterations": _int(optimizer.cg_iterations),
        "cg_hvp_evaluations": _int(optimizer.cg_hvp_evaluations),
        "cg_termination": _termination_name(optimizer.cg_termination),
        "cg_hit_boundary": _bool(optimizer.cg_hit_boundary),
        "cg_negative_curvature": _bool(optimizer.cg_negative_curvature),
        "cg_initial_projected_residual_norm": _float(
            optimizer.cg_initial_projected_residual_norm
        ),
        "cg_final_projected_residual_norm": _float(
            optimizer.cg_final_projected_residual_norm
        ),
        "cg_projected_residual_target": _float(optimizer.cg_projected_residual_target),
        "correction_relative_residual": _float(optimizer.correction_relative_residual),
        "correction_forward_error_bound": _float(
            optimizer.correction_forward_error_bound
        ),
        "multiplier_projection_relative_residual": _float(
            optimizer.multiplier_projection_relative_residual
        ),
        "multiplier_projection_forward_error_bound": _float(
            optimizer.multiplier_projection_forward_error_bound
        ),
        "optimizer_coordinates": _array_identity(optimizer.coordinates),
        "multipliers": _array_identity(optimizer.multipliers),
        "tangent_step": _array_identity(optimizer.tangent_step),
        "correction": _array_identity(optimizer.correction),
        "usable": _bool(optimizer.usable),
        "all_finite": _bool(optimizer.all_finite) and _bool(physical.all_finite),
    }


def _terminal_status(*, usable: bool, supported: bool) -> str:
    if not usable:
        return "CANARY_NOT_USABLE"
    if supported:
        return "SUPPORTED_BY_ONE_STEP_CANARY"
    return "NOT_SUPPORTED_BY_ONE_STEP_CANARY"


def _physical_endpoint_finite(endpoint: FullSpaceGaussNewtonCanaryEndpoint) -> bool:
    physical = endpoint.physical
    objective = float(np.asarray(physical.physical_objective))
    raw_kkt = float(np.asarray(physical.raw_kkt_stationarity_infinity_norm))
    return bool(
        _bool(physical.all_finite) and np.isfinite(objective) and np.isfinite(raw_kkt)
    )


def _numerical_gates_pass(result: FullSpaceGaussNewtonCanaryResult) -> bool:
    reconstruction = result.residual_reconstruction
    endpoints_finite = all(
        _physical_endpoint_finite(endpoint)
        for endpoint in (result.initial, result.identity, result.gauss_newton)
    )
    return bool(
        _bool(result.both_variants_usable)
        and _bool(result.all_finite)
        and _int(result.objective_residual_size) == OBJECTIVE_RESIDUAL_SIZE
        and _bool(reconstruction.residual_valid)
        and _bool(reconstruction.all_finite)
        and _finite_at_most(
            reconstruction.value_scaled_defect,
            VALUE_RECONSTRUCTION_TOLERANCE,
        )
        and _finite_at_most(
            reconstruction.gradient_scaled_defect,
            GRADIENT_RECONSTRUCTION_TOLERANCE,
        )
        and _finite_at_most(
            result.gauss_newton_hvp_bilinear_symmetry_relative_defect,
            GN_SYMMETRY_TOLERANCE,
        )
        and _finite_at_least(
            result.gauss_newton_probe_normalized_curvature,
            GN_NORMALIZED_PSD_LOWER_BOUND,
        )
        and _finite_at_least(
            result.gauss_newton_terminal_normalized_curvature,
            GN_NORMALIZED_PSD_LOWER_BOUND,
        )
        and endpoints_finite
    )


def _resource_gates_pass(
    *,
    pre_post_manifest_identical: bool,
    peak_memory_fraction: float,
) -> bool:
    return bool(
        pre_post_manifest_identical
        and np.isfinite(peak_memory_fraction)
        and peak_memory_fraction < MAXIMUM_MEMORY_FRACTION
    )


def run(output_root: Path) -> dict[str, object]:
    """Execute once and publish a sealed result and exact-byte source manifest."""

    repo_root = Path(__file__).resolve().parents[1]
    output = output_root.absolute()
    output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices()
    if jax.default_backend() != "gpu" or len(devices) != 1:
        raise ValueError("CFS-GN1 requires exactly one JAX GPU")
    if not bool(jax.config.jax_enable_x64):
        raise ValueError("CFS-GN1 requires JAX x64")
    gpu = _gpu_identity()
    if "RTX 5090" not in str(gpu["name"]) or gpu["uuid"] != GPU_UUID:
        raise ValueError("CFS-GN1 requires the frozen RTX 5090")

    source_manifest = _source_manifest(repo_root)
    source_manifest_bytes = _canonical_json_bytes(source_manifest)
    source_manifest_sha256 = _sha256(source_manifest_bytes)
    head = _git_output(repo_root, "rev-parse", "HEAD").decode().strip()
    status_bytes = _git_output(repo_root, "status", "--porcelain=v1", "-z")

    bootstrap = build_single_stage_fullspace_bootstrap()
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    device_z, device_problem = jax.device_put((host_z, host_problem), device=devices[0])
    jax.block_until_ready((device_z, device_problem))
    if (
        device_z.shape != (STATE_SIZE,)
        or device_z.dtype != np.dtype(np.float64)
        or bootstrap.problem.exact_mask_indices.shape != (EQUALITY_SIZE - 1,)
    ):
        raise ValueError("CFS-GN1 bootstrap differs from FP64 716/255")

    kernel = jax.jit(
        lambda problem, state: run_fullspace_gauss_newton_canary(
            problem,
            state,
            trust_radius=TRUST_RADIUS,
            maximum_iterations=MAXIMUM_CG_ITERATIONS,
        )
    )
    monitor = ProcessGpuMemoryMonitor(
        gpu_uuid=GPU_UUID,
        provider_pid=os.getpid(),
        interval_seconds=0.1,
    )
    monitor.start()
    compile_start = time.perf_counter_ns()
    executable = kernel.lower(device_problem, device_z).compile()
    compile_seconds = (time.perf_counter_ns() - compile_start) / 1.0e9
    execution_start = time.perf_counter_ns()
    with jax.transfer_guard("disallow"):
        device_result = executable(device_problem, device_z)
        jax.block_until_ready(device_result)
    execution_seconds = (time.perf_counter_ns() - execution_start) / 1.0e9
    host_result = jax.device_get(device_result)
    memory = monitor.finish()
    if not isinstance(memory, ProcessGpuMemoryResult):
        raise TypeError("CFS-GN1 process GPU memory was not observed")
    peak_memory_fraction = memory.peak_used_memory_mib / int(gpu["total_memory_mib"])
    post_manifest = _source_manifest(repo_root)
    pre_post_manifest_identical = source_manifest == post_manifest

    numerically_usable = _numerical_gates_pass(host_result)
    resource_usable = _resource_gates_pass(
        pre_post_manifest_identical=pre_post_manifest_identical,
        peak_memory_fraction=peak_memory_fraction,
    )
    usable = numerically_usable and resource_usable
    supported = usable and _bool(host_result.gauss_newton_supported)
    reconstruction = host_result.residual_reconstruction
    payload = {
        "schema_version": SCHEMA_VERSION,
        "route": ROUTE,
        "terminal_status": _terminal_status(usable=usable, supported=supported),
        "promotion_eligible": False,
        "three_step_authorized": supported,
        "source": {
            "git_head": head,
            "git_status_sha256": _sha256(status_bytes),
            "manifest_sha256": source_manifest_sha256,
            "manifest_entry_count": len(source_manifest),
            "pre_post_manifest_identical": pre_post_manifest_identical,
        },
        "runtime": {
            "python": os.sys.version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "device": str(devices[0]),
            "gpu": gpu,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        },
        "bootstrap": {
            "state": _array_identity(host_z),
            "state_size": STATE_SIZE,
            "equality_size": EQUALITY_SIZE,
            "objective_residual_size": _int(host_result.objective_residual_size),
        },
        "policy": {
            "trust_radius": TRUST_RADIUS,
            "maximum_cg_iterations": MAXIMUM_CG_ITERATIONS,
            "projected_residual_tolerance": 1.0e-10,
            "feasibility_tolerance": 1.0e-10,
            "linear_residual_tolerance": 1.0e-10,
            "forward_error_limit": 1.0e-7,
            "value_reconstruction_tolerance": VALUE_RECONSTRUCTION_TOLERANCE,
            "gradient_reconstruction_tolerance": (GRADIENT_RECONSTRUCTION_TOLERANCE),
            "gauss_newton_symmetry_tolerance": GN_SYMMETRY_TOLERANCE,
            "gauss_newton_normalized_psd_lower_bound": (GN_NORMALIZED_PSD_LOWER_BOUND),
        },
        "timing": {
            "compile_seconds": compile_seconds,
            "synchronized_execution_seconds": execution_seconds,
            "synchronization": "block_until_ready",
        },
        "transfer_audit": {
            "initial_h2d_calls": 1,
            "hot_h2d_calls": 0,
            "hot_d2h_calls": 0,
            "final_d2h_calls": 1,
            "timed_execution_transfer_guard": "disallow",
        },
        "memory": {
            "provider_pid": memory.provider_pid,
            "gpu_uuid": memory.gpu_uuid,
            "sample_count": len(memory.samples),
            "peak_used_memory_mib": memory.peak_used_memory_mib,
            "peak_memory_fraction": peak_memory_fraction,
            "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
            "passed": peak_memory_fraction < MAXIMUM_MEMORY_FRACTION,
        },
        "residual_reconstruction": {
            "residual_size": _int(host_result.objective_residual_size),
            "reconstructed_value": _float(reconstruction.reconstructed_value),
            "authoritative_value": _float(reconstruction.authoritative_value),
            "value_scaled_defect": _float(reconstruction.value_scaled_defect),
            "gradient_scaled_defect": _float(reconstruction.gradient_scaled_defect),
            "residual_valid": _bool(reconstruction.residual_valid),
            "all_finite": _bool(reconstruction.all_finite),
        },
        "gauss_newton_hvp": {
            "matrix_free": True,
            "bilinear_symmetry_relative_defect": _float(
                host_result.gauss_newton_hvp_bilinear_symmetry_relative_defect
            ),
            "probe_normalized_curvature": _float(
                host_result.gauss_newton_probe_normalized_curvature
            ),
            "terminal_normalized_curvature": _float(
                host_result.gauss_newton_terminal_normalized_curvature
            ),
        },
        "initial": _endpoint_payload(host_result.initial),
        "identity": _endpoint_payload(host_result.identity),
        "gauss_newton": _endpoint_payload(host_result.gauss_newton),
        "all_finite": _bool(host_result.all_finite),
        "numerical_gates_passed": numerically_usable,
        "resource_gates_passed": resource_usable,
        "both_variants_usable_before_resource_gate": _bool(
            host_result.both_variants_usable
        ),
        "gauss_newton_supported_before_resource_gate": _bool(
            host_result.gauss_newton_supported
        ),
    }
    manifest_path = output / "source-manifest.json"
    result_path = output / "result.json"
    with manifest_path.open("xb") as stream:
        stream.write(source_manifest_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    with result_path.open("xb") as stream:
        stream.write(_canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    manifest_path.chmod(0o444)
    result_path.chmod(0o444)
    output.chmod(0o555)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run(arguments.output)
    print(_canonical_json_bytes(payload).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
