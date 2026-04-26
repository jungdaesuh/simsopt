import numpy as np

from alm_utils import zero_gradient_like
from banana_opt.smoothing import smoothmax_selected, smoothmin_selected
from banana_opt.smooth_distance_selection import (
    pairwise_block_min,
    point_tree,
    select_pairwise_near_min,
    surface_dgamma_by_dcoeff_derivative,
    surface_points_tree_shape,
)


_SMOOTHING_EPS = float(np.finfo(float).eps)


def _new_derivative():
    from simsopt._core.derivative import Derivative

    return Derivative({})


def _with_optional_hard_signal(result, hard_signed_value, include_hard_signal):
    if include_hard_signal:
        return (*result, hard_signed_value, max(0.0, hard_signed_value))
    return result


def smooth_max_curvature_signed_constraint(
    curve,
    threshold,
    temperature,
    objective_optimizable,
):
    kappa = np.asarray(curve.kappa(), dtype=float)
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
    signed_value = float(smooth_max - float(threshold))
    return signed_value, grad, max(0.0, signed_value)


def smooth_min_curve_curve_signed_constraint(
    curves,
    minimum_distance,
    temperature,
    objective_optimizable,
    *,
    include_hard_signal=False,
):
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
        result = (
            float(minimum_distance),
            zero_gradient_like(objective_optimizable.x),
            0.0,
        )
        return _with_optional_hard_signal(
            result,
            float(minimum_distance),
            include_hard_signal,
        )

    selection_window = 4.0 * float(temperature)
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
    signed_value = float(minimum_distance) - float(smooth_min)
    hard_signed_value = float(minimum_distance) - float(hard_min)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    result = (signed_value, -grad, max(0.0, signed_value))
    return _with_optional_hard_signal(result, hard_signed_value, include_hard_signal)


def smooth_min_curve_curve_signed_constraint_with_hard_signal(
    curves,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    return smooth_min_curve_curve_signed_constraint(
        curves,
        minimum_distance,
        temperature,
        objective_optimizable,
        include_hard_signal=True,
    )


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance,
    temperature,
    objective_optimizable,
    *,
    include_hard_signal=False,
):
    flat_surface, surface_tree, surface_gamma_shape = surface_points_tree_shape(surface)
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    curve_trees = [None] * len(curve_points)
    hard_min = np.inf
    curve_blocks = []
    for curve_index, curve_gamma in enumerate(curve_points):
        block_min = pairwise_block_min(curve_gamma, flat_surface, right_tree=surface_tree)
        hard_min = min(hard_min, block_min)
        curve_blocks.append((curve_index, block_min))

    selection_window = 4.0 * float(temperature)
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
            flat_surface,
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
    surface_gradient = np.zeros_like(flat_surface)
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
    signed_value = float(minimum_distance) - float(smooth_min)
    hard_signed_value = float(minimum_distance) - float(hard_min)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    result = (signed_value, -grad, max(0.0, signed_value))
    return _with_optional_hard_signal(result, hard_signed_value, include_hard_signal)


def smooth_min_curve_surface_signed_constraint_with_hard_signal(
    curves,
    surface,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    return smooth_min_curve_surface_signed_constraint(
        curves,
        surface,
        minimum_distance,
        temperature,
        objective_optimizable,
        include_hard_signal=True,
    )


def smooth_min_surface_surface_signed_constraint(
    surface_1,
    surface_2,
    minimum_distance,
    temperature,
    objective_optimizable,
    *,
    include_hard_signal=False,
):
    flat_gamma_1, flat_gamma_1_tree, gamma_1_shape = surface_points_tree_shape(surface_1)
    flat_gamma_2, flat_gamma_2_tree, gamma_2_shape = surface_points_tree_shape(
        surface_2
    )
    hard_min = pairwise_block_min(
        flat_gamma_1,
        flat_gamma_2,
        right_tree=flat_gamma_2_tree,
    )
    selection_window = 4.0 * float(temperature)
    rows, cols, diffs, selected_distances = select_pairwise_near_min(
        flat_gamma_1,
        flat_gamma_2,
        hard_min + selection_window,
        left_tree=flat_gamma_1_tree,
        right_tree=flat_gamma_2_tree,
    )
    smooth_min, weights = smoothmin_selected(
        selected_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    directions = diffs / np.maximum(selected_distances[:, None], _SMOOTHING_EPS)
    gradient_1 = np.zeros_like(flat_gamma_1)
    gradient_2 = np.zeros_like(flat_gamma_2)
    np.add.at(gradient_1, rows, weights[:, None] * directions)
    np.add.at(gradient_2, cols, -weights[:, None] * directions)

    derivative = _new_derivative()
    derivative += surface_dgamma_by_dcoeff_derivative(
        surface_1,
        gradient_1.reshape(gamma_1_shape),
    )
    derivative += surface_dgamma_by_dcoeff_derivative(
        surface_2,
        gradient_2.reshape(gamma_2_shape),
    )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    hard_signed_value = float(minimum_distance) - float(hard_min)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    result = (signed_value, -grad, max(0.0, signed_value))
    return _with_optional_hard_signal(result, hard_signed_value, include_hard_signal)


def smooth_min_surface_surface_signed_constraint_with_hard_signal(
    surface_1,
    surface_2,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    return smooth_min_surface_surface_signed_constraint(
        surface_1,
        surface_2,
        minimum_distance,
        temperature,
        objective_optimizable,
        include_hard_signal=True,
    )


def smooth_min_surface_stack_signed_constraint(
    surfaces,
    minimum_distance,
    temperature,
    objective_optimizable,
    *,
    include_hard_signal=False,
):
    surface_entries = [surface_points_tree_shape(surface) for surface in surfaces]
    flat_gammas = [points for points, _tree, _shape in surface_entries]
    flat_trees = [tree for _points, tree, _shape in surface_entries]
    surface_gamma_shapes = [shape for _points, _tree, shape in surface_entries]
    pair_blocks = []
    hard_min = np.inf
    for upper_index in range(1, len(flat_gammas)):
        lower_index = upper_index - 1
        block_min = pairwise_block_min(
            flat_gammas[lower_index],
            flat_gammas[upper_index],
            right_tree=flat_trees[upper_index],
        )
        hard_min = min(hard_min, block_min)
        pair_blocks.append((lower_index, upper_index, block_min))

    selection_window = 4.0 * float(temperature)
    selection_threshold = hard_min + selection_window
    selected_distances = []
    selected_entries = []
    for lower_index, upper_index, block_min in pair_blocks:
        if block_min > selection_threshold:
            continue
        rows, cols, diffs, distances = select_pairwise_near_min(
            flat_gammas[lower_index],
            flat_gammas[upper_index],
            selection_threshold,
            left_tree=flat_trees[lower_index],
            right_tree=flat_trees[upper_index],
        )
        selected_distances.append(distances)
        selected_entries.append((lower_index, upper_index, rows, cols, diffs, distances))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(gamma) for gamma in flat_gammas]
    offset = 0
    for lower_index, upper_index, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset : offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(
            point_gradients[lower_index],
            rows,
            local_weights[:, None] * directions,
        )
        np.add.at(
            point_gradients[upper_index],
            cols,
            -local_weights[:, None] * directions,
        )

    derivative = _new_derivative()
    for surface, surface_gamma_shape, point_gradient in zip(
        surfaces,
        surface_gamma_shapes,
        point_gradients,
    ):
        if np.any(point_gradient):
            derivative += surface_dgamma_by_dcoeff_derivative(
                surface,
                point_gradient.reshape(surface_gamma_shape),
            )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    hard_signed_value = float(minimum_distance) - float(hard_min)
    result = (signed_value, -grad, max(0.0, signed_value))
    return _with_optional_hard_signal(result, hard_signed_value, include_hard_signal)


def smooth_min_surface_stack_signed_constraint_with_hard_signal(
    surfaces,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    return smooth_min_surface_stack_signed_constraint(
        surfaces,
        minimum_distance,
        temperature,
        objective_optimizable,
        include_hard_signal=True,
    )


def single_stage_constraint_activity_tolerances(
    distance_smoothing,
    curvature_smoothing,
    *,
    include_surface_surface,
    include_surface_stack=False,
):
    tolerances = [
        4.0 * float(distance_smoothing),
        4.0 * float(distance_smoothing),
        4.0 * float(curvature_smoothing),
    ]
    if include_surface_surface:
        tolerances.append(4.0 * float(distance_smoothing))
    if include_surface_stack:
        tolerances.append(4.0 * float(distance_smoothing))
    return np.asarray([max(value, _SMOOTHING_EPS) for value in tolerances], dtype=float)
