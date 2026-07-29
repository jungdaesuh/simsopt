"""Single owner for JAX example and parity lane environments."""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Literal, Mapping

from simsopt_jax.config import (
    ExecutionIntent,
    JaxDevice,
    JaxExecutionProfile,
    resolve_jax_execution_profile,
)

JaxLane = Literal["cpu-smoke", "gpu-strict"]

LEGACY_JAX_LANES: tuple[JaxLane, ...] = ("cpu-smoke", "gpu-strict")

_RUNTIME_SELECTOR_ENVIRONMENT = frozenset(
    {
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_TRANSFER_GUARD",
        "SIMSOPT_BACKEND",
        "SIMSOPT_BACKEND_MODE",
        "SIMSOPT_BACKEND_STRICT",
        "SIMSOPT_JAX_BACKEND",
        "SIMSOPT_JAX_PLATFORM",
        "SIMSOPT_JAX_TRANSFER_GUARD",
        "SIMSOPT_PRECISION",
        "SIMSOPT_TARGET_LANE_STRICT",
        "STAGE2_BACKEND",
    }
)


def _profile_environment(profile: JaxExecutionProfile) -> dict[str, str]:
    platform = "cpu" if profile.device == "cpu" else "cuda"
    strict_gpu_parity = profile.device == "gpu" and profile.certification_eligible
    jax_transfer_guard = (
        "disallow"
        if strict_gpu_parity
        else "allow"
        if profile.certification_eligible
        else "log"
    )
    environment = {
        "SIMSOPT_BACKEND_MODE": profile.mode,
        "SIMSOPT_BACKEND_STRICT": "1",
        "SIMSOPT_JAX_TRANSFER_GUARD": ("disallow" if strict_gpu_parity else "log"),
        "JAX_TRANSFER_GUARD": jax_transfer_guard,
        "SIMSOPT_PRECISION": "fp64",
        "JAX_PLATFORMS": platform,
        "JAX_ENABLE_X64": "1",
    }
    if profile.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if profile.certification_eligible:
            environment["XLA_FLAGS"] = "--xla_gpu_exclude_nondeterministic_ops=true"
    return environment


def build_execution_environment(
    device: JaxDevice,
    intent: ExecutionIntent,
    base_environment: Mapping[str, str],
    *,
    repo_root: Path | None = None,
) -> tuple[JaxExecutionProfile, dict[str, str]]:
    """Pin one ordinary-runner profile after removing inherited selectors."""
    profile = resolve_jax_execution_profile(device, intent)
    environment = {
        name: value
        for name, value in base_environment.items()
        if name not in _RUNTIME_SELECTOR_ENVIRONMENT
    }
    environment.update(_profile_environment(profile))
    environment["MPI4PY_RC_INITIALIZE"] = "false"
    if repo_root is not None:
        source_root = str(repo_root / "src")
        inherited_pythonpath = environment.get("PYTHONPATH")
        dependency_root = sysconfig.get_paths()["purelib"]
        pythonpath_entries = (
            (source_root, dependency_root)
            if not inherited_pythonpath
            else (source_root, inherited_pythonpath, dependency_root)
        )
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))
    return profile, environment


def build_lane_environment(
    lane: JaxLane,
    base_environment: Mapping[str, str],
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Build a retained legacy parity lane before child JAX initialization."""
    device: JaxDevice = "cpu" if lane == "cpu-smoke" else "gpu"
    _, environment = build_execution_environment(
        device,
        "parity",
        base_environment,
        repo_root=repo_root,
    )
    return environment
