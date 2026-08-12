"""Run the provenance-bound CFS-CURV1 one-step GPU diagnostic."""

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
from simsopt_jax.solve.fullspace_curvature_canary import (
    FullSpaceCurvatureCanaryEndpoint,
    run_fullspace_curvature_canary,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)

from benchmarks.process_gpu_monitor import (
    ProcessGpuMemoryMonitor,
    ProcessGpuMemoryResult,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-curvature-canary-v1"
ROUTE: Final = "CFS-CURV1"
GPU_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
STATE_SIZE: Final = 716
EQUALITY_SIZE: Final = 255
MAXIMUM_MEMORY_FRACTION: Final = 0.8
SOURCE_PATHS: Final = (
    Path("benchmarks/run_single_stage_fullspace_curvature_canary.py"),
    Path("docs/single_stage_jax_gpu_curvature_canary_implementation_plan.md"),
    Path("src/simsopt_jax/geo/optimizers/curvature_canary.py"),
    Path("src/simsopt_jax/solve/fullspace_curvature_canary.py"),
    Path("tests/geo/test_curvature_canary.py"),
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


def _bool(value: object) -> bool:
    return bool(np.asarray(value))


def _array_identity(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _sha256(array.tobytes()),
    }


def _endpoint_payload(endpoint: FullSpaceCurvatureCanaryEndpoint) -> dict[str, object]:
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
        "raw_direction_norm": _float(optimizer.raw_direction_norm),
        "applied_step_norm": _float(optimizer.applied_step_norm),
        "kkt_relative_residual": _float(optimizer.kkt_relative_residual),
        "kkt_condition_estimate": _float(optimizer.kkt_condition_estimate),
        "kkt_forward_error_bound": _float(optimizer.kkt_forward_error_bound),
        "multiplier_projection_relative_residual": _float(
            optimizer.multiplier_projection_relative_residual
        ),
        "multiplier_projection_reciprocal_condition": _float(
            optimizer.multiplier_projection_reciprocal_condition
        ),
        "multiplier_projection_forward_error_bound": _float(
            optimizer.multiplier_projection_forward_error_bound
        ),
        "correction_relative_residual": _float(optimizer.correction_relative_residual),
        "correction_forward_error_bound": _float(
            optimizer.correction_forward_error_bound
        ),
        "optimizer_coordinates": _array_identity(optimizer.coordinates),
        "multipliers": _array_identity(optimizer.multipliers),
        "raw_direction": _array_identity(optimizer.raw_direction),
        "all_finite": _bool(optimizer.all_finite) and _bool(physical.all_finite),
    }


def _terminal_status(*, usable: bool, supported: bool) -> str:
    if not usable:
        return "CANARY_NOT_USABLE"
    if supported:
        return "SUPPORTED_BY_ONE_STEP_CANARY"
    return "NOT_SUPPORTED_BY_ONE_STEP_CANARY"


def run(output_root: Path) -> dict[str, object]:
    """Execute once and publish sealed result and exact-byte source manifest."""

    repo_root = Path(__file__).resolve().parents[1]
    output = output_root.absolute()
    output.mkdir(parents=True, exist_ok=False)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("CFS-CURV1 requires exactly one JAX GPU")
    gpu = _gpu_identity()
    if "RTX 5090" not in str(gpu["name"]) or gpu["uuid"] != GPU_UUID:
        raise ValueError("CFS-CURV1 requires the frozen RTX 5090")

    source_manifest = _source_manifest(repo_root)
    source_manifest_bytes = _canonical_json_bytes(source_manifest)
    source_manifest_sha256 = _sha256(source_manifest_bytes)
    head = _git_output(repo_root, "rev-parse", "HEAD").decode().strip()
    status_bytes = _git_output(repo_root, "status", "--porcelain=v1", "-z")

    bootstrap = build_single_stage_fullspace_bootstrap()
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))
    if device_z.shape != (
        STATE_SIZE,
    ) or bootstrap.problem.exact_mask_indices.shape != (EQUALITY_SIZE - 1,):
        raise ValueError("CFS-CURV1 bootstrap dimensions differ from 716/255")

    kernel = jax.jit(
        lambda problem, state: run_fullspace_curvature_canary(
            problem,
            state,
            hessian_batch_width=1,
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
        raise TypeError("CFS-CURV1 process GPU memory was not observed")
    peak_memory_fraction = memory.peak_used_memory_mib / int(gpu["total_memory_mib"])

    usable = _bool(host_result.both_variants_usable) and (
        peak_memory_fraction < MAXIMUM_MEMORY_FRACTION
    )
    supported = usable and _bool(host_result.curvature_hypothesis_supported)
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
            "pre_post_manifest_identical": source_manifest
            == _source_manifest(repo_root),
        },
        "runtime": {
            "python": os.sys.version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "gpu": gpu,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        },
        "bootstrap": {
            "state": _array_identity(host_z),
            "state_size": STATE_SIZE,
            "equality_size": EQUALITY_SIZE,
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
        "hessian": {
            **_array_identity(host_result.exact_hessian),
            "symmetry_relative_defect": _float(
                host_result.exact_hessian_symmetry_relative_defect
            ),
            "action_relative_defect": _float(
                host_result.exact_hessian_action_relative_defect
            ),
        },
        "initial": _endpoint_payload(host_result.initial),
        "identity": _endpoint_payload(host_result.identity),
        "exact": _endpoint_payload(host_result.exact),
        "all_finite": _bool(host_result.all_finite),
        "both_variants_usable_before_memory_gate": _bool(
            host_result.both_variants_usable
        ),
        "curvature_hypothesis_supported_before_memory_gate": _bool(
            host_result.curvature_hypothesis_supported
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
