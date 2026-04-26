from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np
from scipy.spatial import cKDTree
from simsopt._core.derivative import Derivative


_SURFACE_TREE_CACHE = WeakKeyDictionary()
_SURFACE_TREE_CACHE_LOCK = RLock()


def point_tree(points):
    return cKDTree(np.asarray(points, dtype=float))


def _surface_dof_fingerprint(surface) -> tuple[tuple[int, ...], str, bytes]:
    dofs = np.asarray(surface.x, dtype=float)
    return dofs.shape, dofs.dtype.str, dofs.tobytes()


def surface_points_tree_shape(surface):
    fingerprint = _surface_dof_fingerprint(surface)
    with _SURFACE_TREE_CACHE_LOCK:
        cached = _SURFACE_TREE_CACHE.get(surface)
        if cached is not None and cached[0] == fingerprint:
            return cached[1], cached[2], cached[3]

    gamma = np.asarray(surface.gamma(), dtype=float)
    points = gamma.reshape((-1, 3))
    tree = point_tree(points)
    with _SURFACE_TREE_CACHE_LOCK:
        _SURFACE_TREE_CACHE[surface] = (fingerprint, points, tree, gamma.shape)
    return points, tree, gamma.shape


def surface_points_and_tree(surface):
    points, tree, _shape = surface_points_tree_shape(surface)
    return points, tree


def pairwise_block_min(left_points, right_points, *, right_tree=None):
    tree = point_tree(right_points) if right_tree is None else right_tree
    distances, _indices = tree.query(
        np.asarray(left_points, dtype=float),
        k=1,
    )
    return float(np.min(distances))


def select_pairwise_near_min(
    left_points,
    right_points,
    threshold,
    *,
    left_tree=None,
    right_tree=None,
):
    left = np.asarray(left_points, dtype=float)
    right = np.asarray(right_points, dtype=float)
    source_tree = point_tree(left) if left_tree is None else left_tree
    tree = point_tree(right) if right_tree is None else right_tree
    sparse_distances = source_tree.sparse_distance_matrix(
        tree,
        float(threshold),
        output_type="coo_matrix",
    )
    rows = np.asarray(sparse_distances.row, dtype=np.intp)
    cols = np.asarray(sparse_distances.col, dtype=np.intp)
    diffs = left[rows] - right[cols]
    distances = np.asarray(sparse_distances.data, dtype=float)
    return rows, cols, diffs, distances


def surface_dgamma_by_dcoeff_derivative(surface, point_gradient):
    surface_vjp = surface.dgamma_by_dcoeff_vjp(point_gradient)
    if isinstance(surface_vjp, Derivative):
        return surface_vjp
    return Derivative({surface: np.asarray(surface_vjp, dtype=float)})
