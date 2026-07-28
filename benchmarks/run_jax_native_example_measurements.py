"""Collect matched native/JAX timing and peak-memory evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchmarks.jax_native_example_measurement_contract import (
    MEASUREMENT_PROFILE_IDS,
    WARM_SAMPLE_COUNT,
    MeasurementProfileId,
)
from examples.jax._lane_environment import build_execution_environment

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_ENVIRONMENT_NAMES = frozenset(
    (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "SIMSOPT_BACKEND_MODE",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
)


@dataclass(frozen=True)
class MeasurementSchedule:
    """Cold, warmup, and seven balanced warm profile orders."""

    cold: tuple[MeasurementProfileId, ...]
    warmup: tuple[MeasurementProfileId, ...]
    warm: tuple[tuple[MeasurementProfileId, ...], ...]


def _rotate(
    values: tuple[MeasurementProfileId, ...], offset: int
) -> tuple[MeasurementProfileId, ...]:
    normalized = offset % len(values)
    return values[normalized:] + values[:normalized]


def build_measurement_schedule(mirror_index: int) -> MeasurementSchedule:
    """Return a deterministic rotating five-profile measurement schedule."""
    if mirror_index < 0:
        raise ValueError("mirror_index must be nonnegative")
    cold = _rotate(MEASUREMENT_PROFILE_IDS, mirror_index)
    return MeasurementSchedule(
        cold=cold,
        warmup=tuple(reversed(cold)),
        warm=tuple(_rotate(cold, index) for index in range(WARM_SAMPLE_COUNT)),
    )


def _jax_profile_parts(
    profile_id: MeasurementProfileId,
) -> tuple[Literal["cpu", "gpu"], Literal["fast", "parity"]]:
    if profile_id == "jax_cpu_fast":
        return "cpu", "fast"
    if profile_id == "jax_cpu_parity":
        return "cpu", "parity"
    if profile_id == "jax_gpu_fast":
        return "gpu", "fast"
    if profile_id == "jax_gpu_parity":
        return "gpu", "parity"
    raise ValueError("native_cpu is not a JAX profile")


def build_measurement_environment(
    profile_id: MeasurementProfileId,
    *,
    allocation_sensitive: bool,
    base_environment: Mapping[str, str],
    gpu_index: int = 0,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, str]:
    """Resolve one isolated profile without inheriting stale backend policy."""
    if gpu_index < 0:
        raise ValueError("gpu_index must be nonnegative")
    scrubbed = {
        name: value
        for name, value in base_environment.items()
        if name not in _PROFILE_ENVIRONMENT_NAMES
    }
    if profile_id == "native_cpu":
        if allocation_sensitive:
            raise ValueError("allocation-sensitive collection is GPU-only")
        return scrubbed
    device, intent = _jax_profile_parts(profile_id)
    if allocation_sensitive and device != "gpu":
        raise ValueError("allocation-sensitive collection is GPU-only")
    _, environment = build_execution_environment(
        device,
        intent,
        scrubbed,
        repo_root=repo_root,
    )
    if device == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "false" if allocation_sensitive else "true"
    )
    return environment


__all__ = [
    "MeasurementSchedule",
    "build_measurement_environment",
    "build_measurement_schedule",
]


if __name__ == "__main__":
    raise SystemExit(
        "The measurement CLI is not available until source-owned commands are bound."
    )
