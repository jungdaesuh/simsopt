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
    candidate_lists = cKDTree(right).query_ball_point(left, r=float(threshold))
    counts = np.fromiter(
        (len(candidates) for candidates in candidate_lists),
        dtype=np.intp,
        count=len(candidate_lists),
    )
    rows = np.repeat(np.arange(counts.size, dtype=np.intp), counts)
    cols = np.fromiter(
        (col for candidates in candidate_lists for col in candidates),
        dtype=np.intp,
        count=int(counts.sum()),
    )
    diffs = left[rows] - right[cols]
    distances = np.linalg.norm(diffs, axis=1)
    return rows, cols, diffs, distances


def surface_dgamma_by_dcoeff_derivative(surface, point_gradient):
    surface_vjp = surface.dgamma_by_dcoeff_vjp(point_gradient)
    if isinstance(surface_vjp, Derivative):
        return surface_vjp
    return Derivative({surface: np.asarray(surface_vjp, dtype=float)})
