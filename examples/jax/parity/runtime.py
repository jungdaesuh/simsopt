"""Pre-import subprocess environments for native and JAX parity lanes."""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Literal, Mapping

from examples.jax._lane_environment import HOST_THREAD_POLICY, build_lane_environment

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
        dependency_root = sysconfig.get_paths()["purelib"]
        pythonpath_entries = (
            (source_root, dependency_root)
            if not inherited_pythonpath
            else (source_root, inherited_pythonpath, dependency_root)
        )
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    # Every compared lane, not only the native one, runs under one host-threading
    # policy: the lanes are compared numerically, so leaving two of the three to
    # inherit the launching shell's thread count makes the comparison depend on
    # how the run was started.  This bounds launch-dependent host variability; it
    # is not a claim that every backend reduction order is thereby fixed.
    environment.update(HOST_THREAD_POLICY)
    pythonpath = environment["PYTHONPATH"].split(os.pathsep)
    if str(repo_root) not in pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(
            (environment["PYTHONPATH"], str(repo_root))
        )
    return environment
