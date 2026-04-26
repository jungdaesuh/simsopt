import numpy as np
from scipy.spatial import cKDTree
from simsopt._core.derivative import Derivative


def pairwise_block_min(left_points, right_points):
    distances, _indices = cKDTree(np.asarray(right_points, dtype=float)).query(
        np.asarray(left_points, dtype=float),
        k=1,
    )
    return float(np.min(distances))


def select_pairwise_near_min(left_points, right_points, threshold):
    left = np.asarray(left_points, dtype=float)
    right = np.asarray(right_points, dtype=float)
    right_tree = cKDTree(right)
    candidate_cols_by_row = right_tree.query_ball_point(left, r=float(threshold))
    pairs = np.asarray(
        [
            (row_index, col_index)
            for row_index, candidate_cols in enumerate(candidate_cols_by_row)
            for col_index in sorted(candidate_cols)
        ],
        dtype=int,
    ).reshape((-1, 2))
    rows = pairs[:, 0]
    cols = pairs[:, 1]
    diffs = left[rows] - right[cols]
    distances = np.linalg.norm(diffs, axis=1)

    return (
        rows,
        cols,
        diffs,
        distances,
    )


def surface_dgamma_by_dcoeff_derivative(surface, point_gradient):
    surface_vjp = surface.dgamma_by_dcoeff_vjp(point_gradient)
    if isinstance(surface_vjp, Derivative):
        return surface_vjp
    return Derivative({surface: np.asarray(surface_vjp, dtype=float)})
