import numpy as np


DISTANCE_SELECTION_CHUNK_ROWS = 64


def _iter_pairwise_distance_chunks(left_points, right_points):
    for row_start in range(0, left_points.shape[0], DISTANCE_SELECTION_CHUNK_ROWS):
        row_stop = min(row_start + DISTANCE_SELECTION_CHUNK_ROWS, left_points.shape[0])
        diffs = left_points[row_start:row_stop, None, :] - right_points[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        yield row_start, diffs, dists


def pairwise_block_min(left_points, right_points):
    min_distance = np.inf
    for _row_start, _diffs, dists in _iter_pairwise_distance_chunks(
        left_points,
        right_points,
    ):
        min_distance = min(min_distance, float(np.min(dists)))
    return min_distance


def select_pairwise_near_min(left_points, right_points, threshold):
    selected_rows = []
    selected_cols = []
    selected_diffs = []
    selected_distances = []
    for row_start, diffs, dists in _iter_pairwise_distance_chunks(
        left_points,
        right_points,
    ):
        mask = dists <= threshold
        if np.any(mask):
            local_rows, cols = np.nonzero(mask)
            selected_rows.append(row_start + local_rows)
            selected_cols.append(cols)
            selected_diffs.append(diffs[local_rows, cols])
            selected_distances.append(dists[local_rows, cols])

    return (
        np.concatenate(selected_rows),
        np.concatenate(selected_cols),
        np.concatenate(selected_diffs),
        np.concatenate(selected_distances),
    )


def surface_dgamma_by_dcoeff_derivative(surface, point_gradient):
    from simsopt._core.derivative import Derivative

    surface_vjp = surface.dgamma_by_dcoeff_vjp(point_gradient)
    if isinstance(surface_vjp, Derivative):
        return surface_vjp
    return Derivative({surface: np.asarray(surface_vjp, dtype=float)})
