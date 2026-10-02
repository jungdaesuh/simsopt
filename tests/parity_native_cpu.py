"""Run a native or JAX lane child with OpenMP pinned before the extension loads."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Literal, Mapping

from examples.jax._lane_environment import HOST_THREAD_POLICY, build_lane_environment
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

LaneChild = Literal["native-cpu", "jax-cpu", "jax-gpu"]

#: The native lane's fixed runtime: the native backend, FP64, and JAX pinned to
#: the host so that a JAX import in the child cannot select a device.
_NATIVE_CPU_ENVIRONMENT: Mapping[str, str] = {
    "SIMSOPT_BACKEND_MODE": "native_cpu",
    "SIMSOPT_BACKEND_STRICT": "1",
    "SIMSOPT_PRECISION": "fp64",
    "JAX_ENABLE_X64": "1",
    "JAX_TRANSFER_GUARD": "allow",
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "MPI4PY_RC_INITIALIZE": "false",
}


def _lane_environment(
    lane: LaneChild, base_environment: Mapping[str, str], *, repo_root: Path
) -> dict[str, str]:
    """Return one lane's child environment, every lane under one thread policy."""
    if lane == "native-cpu":
        environment = {**base_environment, **_NATIVE_CPU_ENVIRONMENT}
        inherited_pythonpath = environment.get("PYTHONPATH")
        source_root = str(repo_root / "src")
        dependency_root = sysconfig.get_paths()["purelib"]
        entries = (
            (source_root, dependency_root)
            if not inherited_pythonpath
            else (source_root, inherited_pythonpath, dependency_root)
        )
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    else:
        environment = build_lane_environment(
            "cpu-smoke" if lane == "jax-cpu" else "gpu-strict",
            base_environment,
            repo_root=repo_root,
        )
    environment.update(HOST_THREAD_POLICY)
    if str(repo_root) not in environment["PYTHONPATH"].split(os.pathsep):
        environment["PYTHONPATH"] = os.pathsep.join(
            (environment["PYTHONPATH"], str(repo_root))
        )
    return environment


def run_native_cpu_child(
    source: str,
    *args: str,
    repo_root: Path,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``source`` as ``python -S -c`` under the native-cpu lane env.

    The environment pins OpenMP before libgomp starts, and
    ``pythonpath_with_loaded_kernel`` puts the compiled extension ahead of
    ``src/simsoptpp``. An in-process env pin cannot undo the pytest process team.
    """
    return run_lane_child("native-cpu", source, *args, repo_root=repo_root, cwd=cwd)


def run_lane_child(
    lane: LaneChild,
    source: str,
    *args: str,
    repo_root: Path,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``source`` as ``python -S -c`` under ``lane``'s environment.

    Every lane, JAX lanes included, runs under one host-threading policy
    (``HOST_THREAD_POLICY``), so a lane compared against another does not
    depend on how the pytest process was launched.
    """
    environment = _lane_environment(lane, dict(os.environ), repo_root=repo_root)
    environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
        *environment["PYTHONPATH"].split(os.pathsep)
    )
    return subprocess.run(
        (sys.executable, "-S", "-c", source, *args),
        cwd=repo_root if cwd is None else cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
