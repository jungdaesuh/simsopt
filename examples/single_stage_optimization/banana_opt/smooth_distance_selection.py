import numpy as np
from scipy.spatial import cKDTree
from simsopt._core.derivative import Derivative


def point_tree(points):
    return cKDTree(np.asarray(points, dtype=float))


def pairwise_block_min(left_points, right_points, *, right_tree=None):
    tree = point_tree(right_points) if right_tree is None else right_tree
    distances, _indices = tree.query(
        np.asarray(left_points, dtype=float),
        k=1,
    )
    return float(np.min(distances))


def select_pairwise_near_min(left_points, right_points, threshold, *, right_tree=None):
    left = np.asarray(left_points, dtype=float)
    right = np.asarray(right_points, dtype=float)
    tree = point_tree(right) if right_tree is None else right_tree
    candidate_lists = tree.query_ball_point(left, r=float(threshold))
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
