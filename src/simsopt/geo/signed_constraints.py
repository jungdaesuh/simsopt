"""Smooth signed geometry constraints ``g(x) <= 0`` for constrained coil optimization.

Every kernel returns ``(signed_value, grad, hard_signed_value)``. ``signed_value``
is a log-sum-exp surrogate of ``hard_signed_value`` at ``temperature`` (never
looser than it), and ``grad`` is ``d(signed_value)/dx`` over the free dofs of
``objective_optimizable``. Unlike the stock hinge objectives, the value keeps the
slack when the constraint is inactive.
"""

from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np
from scipy.spatial import cKDTree

from .._core.derivative import Derivative

__all__ = [
    "smooth_max_curvature_signed_constraint",
    "smooth_min_curve_curve_signed_constraint",
    "smooth_min_curve_surface_signed_constraint",
]


_SMOOTHING_EPS = float(np.finfo(float).eps)
_SURFACE_TREE_CACHE = WeakKeyDictionary()
_SURFACE_TREE_CACHE_LOCK = RLock()
_SOFTMIN_SELECTION_WINDOW_TEMPERATURES = 4.0


def stable_softmax(values, smoothing_eps: float):
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    total = max(float(np.sum(weights)), float(smoothing_eps))
    return weights / total


def smoothmax_selected(values, temperature: float, smoothing_eps: float):
    bounded_temperature = max(float(temperature), float(smoothing_eps))
    values_array = np.asarray(values, dtype=float)
    maximum_value = float(np.max(values_array))
    exp_shifted = np.exp((values_array - maximum_value) / bounded_temperature)
    total = max(float(np.sum(exp_shifted)), float(smoothing_eps))
    weights = exp_shifted / total
    smooth_value = maximum_value + bounded_temperature * float(np.log(total))
    return smooth_value, weights


def smoothmin_selected(values, temperature: float, smoothing_eps: float):
    bounded_temperature = max(float(temperature), float(smoothing_eps))
    values_array = np.asarray(values, dtype=float)
    minimum_value = float(np.min(values_array))
    exp_shifted = np.exp(-(values_array - minimum_value) / bounded_temperature)
    total = max(float(np.sum(exp_shifted)), float(smoothing_eps))
    weights = exp_shifted / total
    smooth_value = minimum_value - bounded_temperature * float(np.log(total))
    return smooth_value, weights


def softmin_selection_window(temperature):
    """Return the truncated soft-min support window.

    Distances outside hard_min + 4T have Boltzmann weights below exp(-4)
    relative to the hard-min pair; hard certification remains exhaustive.
    """
    return _SOFTMIN_SELECTION_WINDOW_TEMPERATURES * float(temperature)


def point_tree(points):
    return cKDTree(np.asarray(points, dtype=float))


def surface_points_tree_shape(surface):
    """``surface.gamma()`` as ``(n, 3)`` points, their KD-tree, and gamma's shape.

    The tree is cached per surface and keyed by the sampled geometry itself
    (gamma's shape and bytes), so any change to it rebuilds the tree: free or
    fixed coefficients, quadrature points, or anything else ``gamma()`` reads.
    The points are a read-only view of that key's bytes, the snapshot the tree
    was built from; a surface recomputing gamma in place does not move them.
    """
    gamma = np.asarray(surface.gamma(), dtype=float)
    geometry_key = (gamma.shape, gamma.tobytes())
    with _SURFACE_TREE_CACHE_LOCK:
        cached = _SURFACE_TREE_CACHE.get(surface)
        if cached is not None and cached[0] == geometry_key:
            return cached[1], cached[2], gamma.shape

    points = np.frombuffer(geometry_key[1], dtype=float).reshape((-1, 3))
    tree = point_tree(points)
    with _SURFACE_TREE_CACHE_LOCK:
        _SURFACE_TREE_CACHE[surface] = (geometry_key, points, tree)
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


def _new_derivative():
    return Derivative({})


def _no_pair_result(minimum_distance, objective_optimizable):
    # No pair can violate the spacing, so the constraint holds with the full
    # clearance as slack, in both the smooth and the hard signal.
    signed_value = -float(minimum_distance)
    return (
        signed_value,
        np.zeros_like(np.asarray(objective_optimizable.x)),
        signed_value,
    )


def smooth_max_curvature_signed_constraint(
    curve,
    threshold,
    temperature,
    objective_optimizable,
    *,
    kappa=None,
):
    """Signed ``max(kappa) - threshold`` for one curve.

    ``kappa`` optionally passes ``curve.kappa()`` already evaluated at the
    current dofs, so a caller that also reports the hard maximum evaluates it once.
    """
    kappa = np.asarray(curve.kappa() if kappa is None else kappa, dtype=float)
    hard_max = float(np.max(kappa))
    active_mask = kappa >= (hard_max - 4.0 * float(temperature))
    if not np.any(active_mask):
        active_mask[np.argmax(kappa)] = True
    smooth_max, active_weights = smoothmax_selected(
        kappa[active_mask],
        temperature,
        _SMOOTHING_EPS,
    )
    full_weights = np.zeros_like(kappa)
    full_weights[active_mask] = active_weights
    grad = np.asarray(
        curve.dkappa_by_dcoeff_vjp(full_weights)(objective_optimizable),
        dtype=float,
    )
    signed_value = smooth_max - float(threshold)
    hard_signed_value = hard_max - float(threshold)
    return signed_value, grad, hard_signed_value


def smooth_min_curve_curve_signed_constraint(
    curves,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    """Signed ``minimum_distance - min_{i<j} dist(curve_i, curve_j)``.

    Fewer than two curves has no pair, so it returns ``-minimum_distance`` and a
    zero gradient.
    """
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    curve_trees = [point_tree(points) for points in curve_points]
    pair_blocks = []
    hard_min = np.inf
    for i, gamma_i in enumerate(curve_points):
        for j in range(i):
            block_min = pairwise_block_min(
                gamma_i,
                curve_points[j],
                right_tree=curve_trees[j],
            )
            hard_min = min(hard_min, block_min)
            pair_blocks.append((i, j, block_min))

    if not pair_blocks:
        return _no_pair_result(minimum_distance, objective_optimizable)

    selection_window = softmin_selection_window(temperature)
    selected_distances = []
    selected_entries = []
    selection_threshold = hard_min + selection_window
    for i, j, block_min in pair_blocks:
        if block_min > selection_threshold:
            continue
        rows, cols, diffs, distances = select_pairwise_near_min(
            curve_points[i],
            curve_points[j],
            selection_threshold,
            left_tree=curve_trees[i],
            right_tree=curve_trees[j],
        )
        selected_distances.append(distances)
        selected_entries.append((i, j, rows, cols, diffs, distances))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(gamma) for gamma in curve_points]
    offset = 0
    for i, j, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset : offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(point_gradients[i], rows, local_weights[:, None] * directions)
        np.add.at(point_gradients[j], cols, -local_weights[:, None] * directions)

    derivative = _new_derivative()
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, -grad, hard_signed_value


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    """Signed ``minimum_distance - min_i dist(curve_i, surface)``.

    The gradient includes the surface dofs when ``objective_optimizable`` owns
    them. No curves returns ``-minimum_distance`` and a zero gradient.
    """
    if not curves:
        return _no_pair_result(minimum_distance, objective_optimizable)

    surface_points, surface_tree, surface_gamma_shape = surface_points_tree_shape(
        surface
    )
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    curve_trees = [None] * len(curve_points)
    curve_blocks = []
    hard_min = np.inf
    for curve_index, gamma in enumerate(curve_points):
        block_min = pairwise_block_min(gamma, surface_points, right_tree=surface_tree)
        hard_min = min(hard_min, block_min)
        curve_blocks.append((curve_index, block_min))

    selection_window = softmin_selection_window(temperature)
    selected_distances = []
    selected_entries = []
    selection_threshold = hard_min + selection_window
    for curve_index, block_min in curve_blocks:
        if block_min > selection_threshold:
            continue
        if curve_trees[curve_index] is None:
            curve_trees[curve_index] = point_tree(curve_points[curve_index])
        rows, cols, diffs, distances = select_pairwise_near_min(
            curve_points[curve_index],
            surface_points,
            selection_threshold,
            left_tree=curve_trees[curve_index],
            right_tree=surface_tree,
        )
        selected_distances.append(distances)
        selected_entries.append((curve_index, rows, cols, diffs, distances))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    curve_gradients = [np.zeros_like(gamma) for gamma in curve_points]
    surface_gradient = np.zeros_like(surface_points)
    offset = 0
    for curve_index, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset : offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(
            curve_gradients[curve_index], rows, local_weights[:, None] * directions
        )
        np.add.at(surface_gradient, cols, -local_weights[:, None] * directions)

    derivative = _new_derivative()
    for curve, point_gradient in zip(curves, curve_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    if np.any(surface_gradient):
        derivative += surface_dgamma_by_dcoeff_derivative(
            surface,
            surface_gradient.reshape(surface_gamma_shape),
        )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - float(hard_min)
    return signed_value, -grad, hard_signed_value
