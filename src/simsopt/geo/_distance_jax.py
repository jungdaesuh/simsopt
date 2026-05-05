"""JAX-native point-cloud candidate cullers for distance objectives."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _stack_point_clouds(point_clouds):
    arrays = tuple(
        np.asarray(point_cloud, dtype=np.float64) for point_cloud in point_clouds
    )
    max_points = max(point_cloud.shape[0] for point_cloud in arrays)
    points = np.zeros((len(arrays), max_points, 3), dtype=np.float64)
    valid = np.zeros((len(arrays), max_points), dtype=bool)
    for index, point_cloud in enumerate(arrays):
        count = point_cloud.shape[0]
        points[index, :count, :] = point_cloud
        valid[index, :count] = True
    return jnp.asarray(points, dtype=jnp.float64), jnp.asarray(valid)


def _min_dist2_matrix(left_points, left_valid, right_points, right_valid):
    diff = left_points[:, None, :, None, :] - right_points[None, :, None, :, :]
    dist2 = jnp.sum(jnp.square(diff), axis=-1)
    valid_pairs = left_valid[:, None, :, None] & right_valid[None, :, None, :]
    return jnp.min(jnp.where(valid_pairs, dist2, jnp.inf), axis=(-1, -2))


def _candidate_pairs_from_mask(mask) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(np.asarray(mask))
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


@partial(jax.jit, static_argnames=("num_base_curves",))
def _within_collection_candidate_mask(
    points,
    valid,
    threshold: float,
    num_base_curves: int,
):
    dist2 = _min_dist2_matrix(points, valid, points, valid)
    indices = jnp.arange(points.shape[0])
    lower_triangle = indices[:, None] > indices[None, :]
    base_curve_mask = indices[None, :] < num_base_curves
    threshold_jax = jnp.asarray(threshold, dtype=points.dtype)
    return (dist2 < jnp.square(threshold_jax)) & lower_triangle & base_curve_mask


@jax.jit
def _between_collections_candidate_mask(
    left_points,
    left_valid,
    right_points,
    right_valid,
    threshold: float,
):
    dist2 = _min_dist2_matrix(left_points, left_valid, right_points, right_valid)
    threshold_jax = jnp.asarray(threshold, dtype=left_points.dtype)
    return dist2 < jnp.square(threshold_jax)


def get_close_candidates_within_collection(
    point_clouds,
    threshold: float,
    num_base_curves: int,
) -> list[tuple[int, int]]:
    """Return C++-compatible lower-triangle close point-cloud pairs."""
    points, valid = _stack_point_clouds(point_clouds)
    mask = _within_collection_candidate_mask(
        points,
        valid,
        threshold,
        num_base_curves,
    )
    return _candidate_pairs_from_mask(mask)


def get_close_candidates_between_collections(
    left_point_clouds,
    right_point_clouds,
    threshold: float,
) -> list[tuple[int, int]]:
    """Return C++-compatible close pairs across two point-cloud collections."""
    left_points, left_valid = _stack_point_clouds(left_point_clouds)
    right_points, right_valid = _stack_point_clouds(right_point_clouds)
    mask = _between_collections_candidate_mask(
        left_points,
        left_valid,
        right_points,
        right_valid,
        threshold,
    )
    return _candidate_pairs_from_mask(mask)
