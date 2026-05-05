"""JAX-native point-cloud candidate cullers for distance objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _min_pairwise_distance_squared(left, right):
    diff = left[:, None, :] - right[None, :, :]
    return jnp.min(jnp.sum(jnp.square(diff), axis=-1))


def _point_cloud_array(point_cloud):
    return jnp.asarray(
        np.asarray(point_cloud, dtype=np.float64),
        dtype=jnp.float64,
    )


def _too_close(left, right, threshold: float) -> bool:
    threshold_squared = float(threshold) * float(threshold)
    distance_squared = _min_pairwise_distance_squared(left, right)
    return bool(np.asarray(jax.device_get(distance_squared)) < threshold_squared)


def get_close_candidates_within_collection(
    point_clouds,
    threshold: float,
    num_base_curves: int,
) -> list[tuple[int, int]]:
    """Return C++-compatible lower-triangle close point-cloud pairs."""
    point_clouds = tuple(_point_cloud_array(point_cloud) for point_cloud in point_clouds)
    base_count = int(num_base_curves)
    candidates: list[tuple[int, int]] = []
    for i, left in enumerate(point_clouds):
        for j in range(min(i, base_count)):
            if _too_close(left, point_clouds[j], threshold):
                candidates.append((i, j))
    return candidates


def get_close_candidates_between_collections(
    left_point_clouds,
    right_point_clouds,
    threshold: float,
) -> list[tuple[int, int]]:
    """Return C++-compatible close pairs across two point-cloud collections."""
    left_point_clouds = tuple(
        _point_cloud_array(point_cloud) for point_cloud in left_point_clouds
    )
    right_point_clouds = tuple(
        _point_cloud_array(point_cloud) for point_cloud in right_point_clouds
    )
    candidates: list[tuple[int, int]] = []
    for i, left in enumerate(left_point_clouds):
        for j, right in enumerate(right_point_clouds):
            if _too_close(left, right, threshold):
                candidates.append((i, j))
    return candidates
