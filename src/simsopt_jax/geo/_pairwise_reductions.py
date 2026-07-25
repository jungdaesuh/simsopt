"""Compatibility imports for pairwise reductions now owned by ``simsopt_jax.core``."""

from simsopt_jax.core._pairwise_reductions import (
    _chunk_rows,
    _chunk_rows_with_valid_weights,
    _masked_pairwise_distances,
    _masked_point_cloud_center_radius,
    _masked_point_cloud_lower_bound,
    _pairwise_distances,
    _point_cloud_center_radius,
    _point_cloud_lower_bound,
    _resolve_pairwise_penalty_chunk_size,
    _scalar_like,
    _use_dense_pairwise_path,
    pairwise_min_distance_batched_pure,
    pairwise_min_distance_pure,
    pairwise_selected_smoothmin_distance_batched_pure,
    pairwise_selected_smoothmin_distance_pure,
    pairwise_thresholded_mean_square_distance_pure,
)

__all__ = [
    "_chunk_rows",
    "_chunk_rows_with_valid_weights",
    "_masked_pairwise_distances",
    "_masked_point_cloud_center_radius",
    "_masked_point_cloud_lower_bound",
    "_pairwise_distances",
    "_point_cloud_center_radius",
    "_point_cloud_lower_bound",
    "_resolve_pairwise_penalty_chunk_size",
    "_scalar_like",
    "_use_dense_pairwise_path",
    "pairwise_min_distance_batched_pure",
    "pairwise_min_distance_pure",
    "pairwise_selected_smoothmin_distance_batched_pure",
    "pairwise_selected_smoothmin_distance_pure",
    "pairwise_thresholded_mean_square_distance_pure",
]
