"""Explicit sharding helpers for pure grouped-field and pairwise kernels."""

from __future__ import annotations

from functools import lru_cache

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..backend import get_sharding_tuning, maybe_initialize_distributed_jax
from ..backend.runtime import register_backend_cache_clear

__all__ = [
    "inspect_array_sharding_summary",
    "maybe_shard_grouped_field_inputs",
    "maybe_shard_pairwise_row_inputs",
    "summarize_array_sharding",
]


def _devices_for_platform(platform: str) -> tuple[object, ...]:
    backend_name = "gpu" if platform == "cuda" else platform
    maybe_initialize_distributed_jax()
    try:
        return tuple(jax.devices(backend=backend_name))
    except Exception:
        return ()


@lru_cache(maxsize=8)
def _mesh_for(platform: str, axis_name: str) -> Mesh | None:
    devices = _devices_for_platform(platform)
    if not devices:
        return None
    return Mesh(np.asarray(devices, dtype=object), (axis_name,))


def _partition_spec_for_axis(axis_name: str, ndim: int) -> P:
    if ndim <= 0:
        return P()
    return P(axis_name, *([None] * (ndim - 1)))


@lru_cache(maxsize=16)
def _point_sharding_for(platform: str, axis_name: str, ndim: int) -> NamedSharding | None:
    mesh = _mesh_for(platform, axis_name)
    if mesh is None:
        return None
    return NamedSharding(mesh, _partition_spec_for_axis(axis_name, ndim))


@lru_cache(maxsize=8)
def _replicated_sharding_for(platform: str, axis_name: str) -> NamedSharding | None:
    mesh = _mesh_for(platform, axis_name)
    if mesh is None:
        return None
    return NamedSharding(mesh, P())


def _place_array(array, sharding):
    if isinstance(array, (np.ndarray, jax.Array)):
        return jax.device_put(array, sharding)
    return lax.with_sharding_constraint(jnp.asarray(array), sharding)


def _should_shard_points(points, tuning) -> bool:
    if not tuning.active or tuning.strategy not in {"points", "hybrid"}:
        return False
    return int(points.shape[0]) >= int(tuning.min_points_to_shard)


def _should_shard_pairwise_rows(points_a, tuning) -> bool:
    if not tuning.active or tuning.strategy not in {"pairwise_rows", "hybrid"}:
        return False
    return int(points_a.shape[0]) >= int(tuning.min_pairwise_rows_to_shard)


def maybe_shard_grouped_field_inputs(points, coil_arrays, *, mode: str | None = None):
    """Shard grouped-field point clouds while replicating coil-group inputs."""
    tuning = get_sharding_tuning(mode)
    if not _should_shard_points(points, tuning):
        return points, coil_arrays

    point_sharding = _point_sharding_for(
        tuning.platform,
        tuning.mesh_axis_name,
        int(jnp.ndim(points)),
    )
    replicated_sharding = _replicated_sharding_for(
        tuning.platform,
        tuning.mesh_axis_name,
    )
    if point_sharding is None or replicated_sharding is None:
        return points, coil_arrays

    sharded_points = _place_array(points, point_sharding)
    replicated_arrays = tuple(
        (
            _place_array(gammas, replicated_sharding),
            _place_array(gammadashs, replicated_sharding),
            _place_array(currents, replicated_sharding),
        )
        for gammas, gammadashs, currents in coil_arrays
    )
    return sharded_points, replicated_arrays


def maybe_shard_pairwise_row_inputs(
    points_a,
    points_b,
    *,
    mode: str | None = None,
):
    """Shard the row-owned side of pairwise kernels while replicating RHS inputs."""
    tuning = get_sharding_tuning(mode)
    if not _should_shard_pairwise_rows(points_a, tuning):
        return points_a, points_b

    left_sharding = _point_sharding_for(
        tuning.platform,
        tuning.mesh_axis_name,
        int(jnp.ndim(points_a)),
    )
    replicated_sharding = _replicated_sharding_for(
        tuning.platform,
        tuning.mesh_axis_name,
    )
    if left_sharding is None or replicated_sharding is None:
        return points_a, points_b

    return (
        _place_array(points_a, left_sharding),
        _place_array(points_b, replicated_sharding),
    )


def _sharding_attr_bool(value) -> bool | None:
    if value is None:
        return None
    return bool(value() if callable(value) else value)


def summarize_array_sharding(value) -> dict[str, object]:
    """Return a stable, JSON-friendly summary of an array's sharding state."""
    if not isinstance(value, jax.Array):
        return {
            "kind": "non_jax_array",
            "spec": None,
            "device_count": 0,
            "fully_replicated": None,
        }
    sharding = getattr(value, "sharding", None)
    device_set = getattr(sharding, "device_set", None)
    mesh = getattr(sharding, "mesh", None)
    summary = {
        "kind": None if sharding is None else type(sharding).__name__,
        "spec": None if sharding is None else str(getattr(sharding, "spec", None)),
        "device_count": len(device_set) if device_set is not None else 1,
        "fully_replicated": None
        if sharding is None
        else _sharding_attr_bool(getattr(sharding, "is_fully_replicated", None)),
    }
    if mesh is not None:
        summary["mesh_shape"] = dict(getattr(mesh, "shape", {}))
    return summary


def inspect_array_sharding_summary(value) -> dict[str, object]:
    """Capture a sharding summary using JAX's inspection hook when available."""
    summary = summarize_array_sharding(value)
    inspect_fn = getattr(getattr(jax, "debug", None), "inspect_array_sharding", None)
    if inspect_fn is None or not isinstance(value, jax.Array):
        return summary
    observed: dict[str, object] = {}

    def _capture(observed_sharding):
        observed["kind"] = type(observed_sharding).__name__
        observed["spec"] = str(getattr(observed_sharding, "spec", None))

    try:
        inspect_fn(value, callback=_capture)
    except Exception:
        return summary
    if observed:
        summary["inspected_kind"] = observed.get("kind")
        summary["inspected_spec"] = observed.get("spec")
    return summary


def _clear_sharding_caches() -> None:
    _mesh_for.cache_clear()
    _point_sharding_for.cache_clear()
    _replicated_sharding_for.cache_clear()


register_backend_cache_clear(_clear_sharding_caches)
