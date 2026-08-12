"""Fail-closed isolated Python module launches from an immutable snapshot."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from benchmarks.single_stage_changed_state_profiler_policy import (
    TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT,
)
from benchmarks.single_stage_compute_graph_snapshot import load_snapshot_manifest

ALLOWED_MODULES: Final = frozenset(
    {
        "benchmarks.landau_a100_qualification",
        "benchmarks.single_stage_compute_graph_c0_evaluator",
        "benchmarks.single_stage_compute_graph_c0_runner",
        "benchmarks.single_stage_compute_graph_canary_evaluator",
        "benchmarks.single_stage_compute_graph_canary_profile",
        "benchmarks.single_stage_compute_graph_command_buffer_control",
        "benchmarks.single_stage_compute_graph_newton_telemetry",
        "benchmarks.single_stage_compute_graph_native_reference",
        "benchmarks.single_stage_compute_graph_native_trajectory",
        "benchmarks.single_stage_compute_graph_native_trajectory_runner",
        "benchmarks.single_stage_compute_graph_phase0_post_gate",
        "benchmarks.single_stage_compute_graph_phase0_workflow",
        "benchmarks.single_stage_compute_graph_variant_trajectory",
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner",
        "examples.jax.parity.child",
    }
)
EXECUTION_SOURCE_MODULES: Final = frozenset({"examples.jax.parity.child"})
STATIC_TIMING_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "JAX_TRANSFER_GUARD",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PATH",
        "SIMSOPT_BACKEND_MODE",
        "SIMSOPT_BACKEND_STRICT",
        "SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE",
        "SIMSOPT_EXACT_ADJOINT_DENSE_LU",
        "SIMSOPT_JAX_BACKEND",
        "SIMSOPT_JAX_CHUNK_AUTOTUNE",
        "SIMSOPT_JAX_COIL_CHUNK_SIZE",
        "SIMSOPT_JAX_GPU_ALLOCATOR",
        "SIMSOPT_JAX_GPU_MEM_FRACTION",
        "SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB",
        "SIMSOPT_JAX_GPU_PREALLOCATE",
        "SIMSOPT_JAX_POINT_CHUNK_SIZE",
        "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
        "SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU",
        "SIMSOPT_PRECISION",
        "SIMSOPT_TARGET_LANE_STRICT",
        "SIMSOPT_TF_GPU_ALLOCATOR",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_FLAGS",
    }
)
ROUTE_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "JAX_COMPILATION_CACHE_DIR",
        "PHASE0_EXPECTED_GPU_UUID",
        "SINGLE_STAGE_COMPUTE_GRAPH_CHILD_OUTPUT",
        "SINGLE_STAGE_COMPUTE_GRAPH_LANE",
        "SINGLE_STAGE_COMPUTE_GRAPH_MODE",
        "SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX",
        "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT",
    }
)
TRANSPORT_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT",
        "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY",
        TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT,
    }
)
TIMING_ENVIRONMENT_KEYS: Final = (
    STATIC_TIMING_ENVIRONMENT_KEYS | ROUTE_ENVIRONMENT_KEYS | TRANSPORT_ENVIRONMENT_KEYS
)
ISOLATED_MODULE_BOOTSTRAP: Final = """
import importlib.machinery
import runpy
import sys

allowed_finders = (
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
)
sys.meta_path[:] = [finder for finder in sys.meta_path if finder in allowed_finders]
module = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""


class SnapshotModuleLaunchError(RuntimeError):
    """The requested child cannot be launched from the frozen snapshot."""


@dataclass(frozen=True, slots=True)
class SnapshotModuleLaunch:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]


def normalize_timing_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Return the complete allowlisted environment that may affect a timed child."""

    return {
        key: environment[key]
        for key in sorted(TIMING_ENVIRONMENT_KEYS)
        if key in environment
    }


def normalize_static_timing_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Return only behavior-changing controls shared by every timed route."""

    return {
        key: environment[key]
        for key in sorted(STATIC_TIMING_ENVIRONMENT_KEYS)
        if key in environment
    }


def normalize_route_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Return explicitly constructed per-invocation routing controls."""

    return {
        key: environment[key]
        for key in sorted(ROUTE_ENVIRONMENT_KEYS)
        if key in environment
    }


def observe_effective_numerical_policies(
    quadrature_nodes: int,
) -> dict[str, object]:
    """Read the effective policy from imported production runtime modules."""

    if quadrature_nodes < 1:
        raise SnapshotModuleLaunchError("quadrature_nodes must be positive")
    from simsopt_jax.core.biotsavart import _read_tuning_config
    from simsopt_jax.geo.optimizers.optimizer import (
        dense_operator_chunk_batch_size,
    )

    coil_size, quadrature_size, point_size = _read_tuning_config()
    block_size = quadrature_nodes if quadrature_size <= 0 else quadrature_size
    full_blocks, tail = divmod(quadrature_nodes, block_size)
    blocks = [block_size] * full_blocks
    if tail:
        blocks.append(tail)
    return {
        "dense_batch_width": dense_operator_chunk_batch_size(),
        "point_chunk_size": None if point_size <= 0 else point_size,
        "coil_chunk_size": None if coil_size <= 0 else coil_size,
        "quadrature_block_sizes": blocks,
    }


def build_snapshot_module_launch(
    interpreter: Path,
    snapshot_root: Path,
    module: str,
    module_args: Sequence[str],
    base_environment: Mapping[str, str],
) -> SnapshotModuleLaunch:
    """Construct one manifest-validated, ambient-import-isolated module launch."""

    if not interpreter.is_absolute():
        raise SnapshotModuleLaunchError("interpreter path must be absolute")
    # Preserve the virtual-environment entry-point path. Resolving its symlink
    # selects the base interpreter directly and therefore drops the venv's
    # pyvenv.cfg/site-packages runtime identity.
    interpreter = Path(os.path.abspath(interpreter))
    snapshot_root = snapshot_root.resolve()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise SnapshotModuleLaunchError(
            "interpreter must be an executable regular file"
        )
    if module not in ALLOWED_MODULES:
        raise SnapshotModuleLaunchError("module is not allowed for isolated launch")
    try:
        entries, _manifest_sha256 = load_snapshot_manifest(snapshot_root)
    except (OSError, ValueError) as error:
        raise SnapshotModuleLaunchError(
            f"immutable snapshot is invalid: {error}"
        ) from error
    if not any(entry.role == "native_extension" for entry in entries):
        raise SnapshotModuleLaunchError("immutable snapshot lacks a native extension")
    module_relative_path = module.replace(".", "/") + ".py"
    module_entries = tuple(
        entry for entry in entries if entry.relative_path == module_relative_path
    )
    expected_role = (
        "execution_source" if module in EXECUTION_SOURCE_MODULES else "benchmark"
    )
    if len(module_entries) != 1 or module_entries[0].role != expected_role:
        raise SnapshotModuleLaunchError(
            "allowed module is absent from the validated benchmark manifest"
        )
    environment = normalize_timing_environment(base_environment)
    environment.update(
        {
            "PYTHONPATH": f"{snapshot_root / 'src'}:{snapshot_root}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return SnapshotModuleLaunch(
        argv=(
            str(interpreter),
            "-P",
            "-s",
            "-c",
            ISOLATED_MODULE_BOOTSTRAP,
            module,
            *tuple(module_args),
        ),
        cwd=snapshot_root,
        environment=MappingProxyType(environment),
    )


__all__ = [
    "ALLOWED_MODULES",
    "ISOLATED_MODULE_BOOTSTRAP",
    "ROUTE_ENVIRONMENT_KEYS",
    "STATIC_TIMING_ENVIRONMENT_KEYS",
    "TIMING_ENVIRONMENT_KEYS",
    "TRANSPORT_ENVIRONMENT_KEYS",
    "SnapshotModuleLaunch",
    "SnapshotModuleLaunchError",
    "build_snapshot_module_launch",
    "normalize_route_environment",
    "normalize_static_timing_environment",
    "normalize_timing_environment",
    "observe_effective_numerical_policies",
]
