import numpy as np

from alm_utils import zero_gradient_like
from banana_opt.smoothing import smoothmax_selected, smoothmin_selected
from banana_opt.smooth_distance_selection import (
    pairwise_block_min,
    select_pairwise_near_min,
    surface_dgamma_by_dcoeff_derivative,
)


_SMOOTHING_EPS = float(np.finfo(float).eps)


def _new_derivative():
    from simsopt._core.derivative import Derivative

    return Derivative({})


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
):
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    pair_blocks = []
    hard_min = np.inf
    for i, gamma_i in enumerate(curve_points):
        for j in range(i):
            block_min = pairwise_block_min(gamma_i, curve_points[j])
            hard_min = min(hard_min, block_min)
            pair_blocks.append((i, j, block_min))

    if not pair_blocks:
        return (
            float(minimum_distance),
            zero_gradient_like(objective_optimizable.x),
            0.0,
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
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    return signed_value, -grad, max(0.0, signed_value)


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    surface_gamma = np.asarray(surface.gamma(), dtype=float)
    flat_surface = surface_gamma.reshape((-1, 3))
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    hard_min = np.inf
    curve_blocks = []
    for curve_index, curve_gamma in enumerate(curve_points):
        block_min = pairwise_block_min(curve_gamma, flat_surface)
        hard_min = min(hard_min, block_min)
        curve_blocks.append((curve_index, block_min))

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    selection_threshold = hard_min + selection_window
    for curve_index, block_min in curve_blocks:
        if block_min > selection_threshold:
            continue
        rows, cols, diffs, distances = select_pairwise_near_min(
            curve_points[curve_index],
            flat_surface,
            selection_threshold,
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
            surface_gradient.reshape(surface_gamma.shape),
        )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    return signed_value, -grad, max(0.0, signed_value)


def smooth_min_surface_surface_signed_constraint(
    surface_1,
    surface_2,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    gamma_1 = np.asarray(surface_1.gamma(), dtype=float)
    gamma_2 = np.asarray(surface_2.gamma(), dtype=float)
    flat_gamma_1 = gamma_1.reshape((-1, 3))
    flat_gamma_2 = gamma_2.reshape((-1, 3))
    hard_min = pairwise_block_min(flat_gamma_1, flat_gamma_2)
    selection_window = 4.0 * float(temperature)
    rows, cols, diffs, selected_distances = select_pairwise_near_min(
        flat_gamma_1,
        flat_gamma_2,
        hard_min + selection_window,
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
        gradient_1.reshape(gamma_1.shape),
    )
    derivative += surface_dgamma_by_dcoeff_derivative(
        surface_2,
        gradient_2.reshape(gamma_2.shape),
    )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    return signed_value, -grad, max(0.0, signed_value)


def smooth_min_surface_stack_signed_constraint(
    surfaces,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    surface_gammas = [np.asarray(surface.gamma(), dtype=float) for surface in surfaces]
    flat_gammas = [gamma.reshape((-1, 3)) for gamma in surface_gammas]
    pair_blocks = []
    hard_min = np.inf
    for upper_index in range(1, len(flat_gammas)):
        lower_index = upper_index - 1
        block_min = pairwise_block_min(
            flat_gammas[lower_index],
            flat_gammas[upper_index],
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
    for surface, surface_gamma, point_gradient in zip(
        surfaces,
        surface_gammas,
        point_gradients,
    ):
        if np.any(point_gradient):
            derivative += surface_dgamma_by_dcoeff_derivative(
                surface,
                point_gradient.reshape(surface_gamma.shape),
            )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    return signed_value, -grad, max(0.0, signed_value)


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
