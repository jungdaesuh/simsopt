"""Pre-import subprocess environments for native and JAX parity lanes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Mapping

from examples.jax._lane_environment import build_lane_environment

ParityLane = Literal["native-cpu", "jax-cpu", "jax-gpu"]


def build_parity_lane_environment(
    lane: ParityLane,
    base_environment: Mapping[str, str],
    *,
    repo_root: Path,
) -> dict[str, str]:
    """Return the fixed environment for one isolated parity child."""
    if lane == "jax-cpu":
        environment = build_lane_environment(
            "cpu-smoke", base_environment, repo_root=repo_root
        )
    elif lane == "jax-gpu":
        environment = build_lane_environment(
            "gpu-strict", base_environment, repo_root=repo_root
        )
    else:
        environment = dict(base_environment)
        environment.update(
            {
                "SIMSOPT_BACKEND_MODE": "native_cpu",
                "SIMSOPT_BACKEND_STRICT": "1",
                "SIMSOPT_PRECISION": "fp64",
                "JAX_ENABLE_X64": "1",
                "JAX_TRANSFER_GUARD": "allow",
                "JAX_PLATFORMS": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "MPI4PY_RC_INITIALIZE": "false",
            }
        )
        source_root = str(repo_root / "src")
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not inherited_pythonpath
            else os.pathsep.join((source_root, inherited_pythonpath))
        )
    pythonpath = environment["PYTHONPATH"].split(os.pathsep)
    if str(repo_root) not in pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(
            (environment["PYTHONPATH"], str(repo_root))
        )
    return environment
