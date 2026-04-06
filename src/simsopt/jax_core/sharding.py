"""Explicit sharding helpers for pure grouped-field kernels."""

from __future__ import annotations

from functools import lru_cache

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..backend import get_sharding_tuning
from ..backend.runtime import register_backend_cache_clear

__all__ = ["maybe_shard_grouped_field_inputs"]


def _local_devices_for_platform(platform: str) -> tuple[object, ...]:
    backend_name = "gpu" if platform == "cuda" else platform
    try:
        return tuple(jax.local_devices(backend=backend_name))
    except Exception:
        return ()


@lru_cache(maxsize=8)
def _mesh_for(platform: str, axis_name: str) -> Mesh | None:
    devices = _local_devices_for_platform(platform)
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
    if not tuning.active or tuning.strategy != "points":
        return False
    return int(points.shape[0]) >= int(tuning.min_points_to_shard)


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


def _clear_sharding_caches() -> None:
    _mesh_for.cache_clear()
    _point_sharding_for.cache_clear()
    _replicated_sharding_for.cache_clear()


register_backend_cache_clear(_clear_sharding_caches)
