"""Single owner for JAX example and parity lane environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Mapping

JaxLane = Literal["cpu-smoke", "gpu-strict"]

LANE_ENVIRONMENT: dict[JaxLane, dict[str, str]] = {
    "cpu-smoke": {
        "SIMSOPT_BACKEND_MODE": "jax_cpu_parity",
        "SIMSOPT_BACKEND_STRICT": "1",
        "SIMSOPT_JAX_TRANSFER_GUARD": "log",
        "JAX_TRANSFER_GUARD": "allow",
        "SIMSOPT_PRECISION": "fp64",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        "CUDA_VISIBLE_DEVICES": "",
    },
    "gpu-strict": {
        "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
        "SIMSOPT_BACKEND_STRICT": "1",
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "JAX_TRANSFER_GUARD": "disallow",
        "SIMSOPT_PRECISION": "fp64",
        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    },
}


def build_lane_environment(
    lane: JaxLane,
    base_environment: Mapping[str, str],
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Overlay one typed lane before the child can import JAX."""
    environment = dict(base_environment)
    environment.update(LANE_ENVIRONMENT[lane])
    environment["MPI4PY_RC_INITIALIZE"] = "false"
    if repo_root is not None:
        source_root = str(repo_root / "src")
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not inherited_pythonpath
            else os.pathsep.join((source_root, inherited_pythonpath))
        )
    return environment
