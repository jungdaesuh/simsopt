import numpy as np

from alm_utils import zero_gradient_like
from banana_opt.smoothing import smoothmax_selected, smoothmin_selected
from simsopt._core.derivative import Derivative


_SMOOTHING_EPS = float(np.finfo(float).eps)


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
    pair_blocks = []
    hard_min = np.inf
    for i, curve_i in enumerate(curves):
        gamma_i = np.asarray(curve_i.gamma(), dtype=float)
        for j in range(i):
            curve_j = curves[j]
            gamma_j = np.asarray(curve_j.gamma(), dtype=float)
            diffs = gamma_i[:, None, :] - gamma_j[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            hard_min = min(hard_min, float(np.min(dists)))
            pair_blocks.append((i, j, diffs, dists))

    if not pair_blocks:
        return (
            float(minimum_distance),
            zero_gradient_like(objective_optimizable.x),
            0.0,
        )

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    for i, j, diffs, dists in pair_blocks:
        mask = dists <= (hard_min + selection_window)
        if not np.any(mask):
            mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
        rows, cols = np.nonzero(mask)
        selected_distances.append(dists[rows, cols])
        selected_entries.append((i, j, rows, cols, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    offset = 0
    for i, j, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(point_gradients[i], rows, local_weights[:, None] * directions)
        np.add.at(point_gradients[j], cols, -local_weights[:, None] * directions)

    derivative = Derivative({})
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    return signed_value, grad, max(0.0, signed_value)


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    surface_gamma = np.asarray(surface.gamma(), dtype=float)
    flat_surface = surface_gamma.reshape((-1, 3))
    hard_min = np.inf
    curve_blocks = []
    for curve_index, curve in enumerate(curves):
        curve_gamma = np.asarray(curve.gamma(), dtype=float)
        diffs = curve_gamma[:, None, :] - flat_surface[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        hard_min = min(hard_min, float(np.min(dists)))
        curve_blocks.append((curve_index, diffs, dists))

    selection_window = 4.0 * float(temperature)
    selected_distances = []
    selected_entries = []
    for curve_index, diffs, dists in curve_blocks:
        mask = dists <= (hard_min + selection_window)
        if not np.any(mask):
            mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
        rows, cols = np.nonzero(mask)
        selected_distances.append(dists[rows, cols])
        selected_entries.append((curve_index, rows, cols, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = smoothmin_selected(
        flat_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    curve_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    surface_gradient = np.zeros_like(flat_surface)
    offset = 0
    for curve_index, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = diffs / np.maximum(distances[:, None], _SMOOTHING_EPS)
        np.add.at(curve_gradients[curve_index], rows, local_weights[:, None] * directions)
        np.add.at(surface_gradient, cols, -local_weights[:, None] * directions)

    derivative = Derivative({})
    for curve, point_gradient in zip(curves, curve_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    if np.any(surface_gradient):
        derivative += surface.dgamma_by_dcoeff_vjp(surface_gradient.reshape(surface_gamma.shape))
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    return signed_value, grad, max(0.0, signed_value)


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
    diffs = flat_gamma_1[:, None, :] - flat_gamma_2[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    hard_min = float(np.min(dists))
    selection_window = 4.0 * float(temperature)
    mask = dists <= (hard_min + selection_window)
    if not np.any(mask):
        mask[np.unravel_index(np.argmin(dists), dists.shape)] = True
    rows, cols = np.nonzero(mask)
    selected_distances = dists[rows, cols]
    smooth_min, weights = smoothmin_selected(
        selected_distances,
        temperature,
        _SMOOTHING_EPS,
    )

    directions = diffs[rows, cols] / np.maximum(selected_distances[:, None], _SMOOTHING_EPS)
    gradient_1 = np.zeros_like(flat_gamma_1)
    gradient_2 = np.zeros_like(flat_gamma_2)
    np.add.at(gradient_1, rows, weights[:, None] * directions)
    np.add.at(gradient_2, cols, -weights[:, None] * directions)

    derivative = Derivative({})
    derivative += surface_1.dgamma_by_dcoeff_vjp(gradient_1.reshape(gamma_1.shape))
    derivative += surface_2.dgamma_by_dcoeff_vjp(gradient_2.reshape(gamma_2.shape))
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    signed_value = float(minimum_distance) - float(smooth_min)
    return signed_value, grad, max(0.0, signed_value)


def single_stage_constraint_activity_tolerances(
    distance_smoothing,
    curvature_smoothing,
    *,
    include_surface_surface,
):
    tolerances = [
        4.0 * float(distance_smoothing),
        4.0 * float(distance_smoothing),
        4.0 * float(curvature_smoothing),
    ]
    if include_surface_surface:
        tolerances.append(4.0 * float(distance_smoothing))
    return np.asarray([max(value, _SMOOTHING_EPS) for value in tolerances], dtype=float)
