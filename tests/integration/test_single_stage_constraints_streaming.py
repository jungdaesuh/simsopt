from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_STAGE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(SINGLE_STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SINGLE_STAGE_ROOT))

from banana_opt import single_stage_constraints as constraints  # noqa: E402


def _curve(radius: float, x_shift: float, y_shift: float) -> CurveXYZFourier:
    curve = CurveXYZFourier(16, 1)
    curve.set("xc(0)", x_shift)
    curve.set("yc(0)", y_shift)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    return curve


def _surface(major_radius: float, minor_radius: float) -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=1,
        nphi=8,
        ntheta=8,
        mpol=1,
        ntor=0,
    )
    surface.set("rc(0,0)", major_radius)
    surface.set("rc(1,0)", minor_radius)
    surface.set("zs(1,0)", minor_radius)
    return surface


def _assert_constraint_tuple_allclose(actual, expected):
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=1.0e-12)


def _dense_curve_curve_constraint(curves, minimum_distance, temperature, objective):
    pair_blocks = []
    hard_min = np.inf
    for i, curve_i in enumerate(curves):
        gamma_i = np.asarray(curve_i.gamma(), dtype=float)
        for j in range(i):
            gamma_j = np.asarray(curves[j].gamma(), dtype=float)
            diffs = gamma_i[:, None, :] - gamma_j[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            hard_min = min(hard_min, float(np.min(dists)))
            pair_blocks.append((i, j, diffs, dists))

    selected_distances = []
    selected_entries = []
    for i, j, diffs, dists in pair_blocks:
        rows, cols = constraints._selected_distance_rows_and_cols(
            dists,
            hard_min=hard_min,
            temperature=temperature,
        )
        selected_distances.append(dists[rows, cols])
        selected_entries.append((i, j, rows, cols, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = constraints.smoothmin_selected(
        flat_distances,
        temperature,
        constraints._SMOOTHING_EPS,
    )

    point_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    offset = 0
    for i, j, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = constraints._distance_directions(diffs, distances)
        np.add.at(point_gradients[i], rows, local_weights[:, None] * directions)
        np.add.at(point_gradients[j], cols, -local_weights[:, None] * directions)

    derivative = constraints._new_derivative()
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    grad = np.asarray(derivative(objective), dtype=float)
    return constraints._smooth_min_signed_constraint(minimum_distance, smooth_min, grad)


def _dense_curve_surface_constraint(curves, surface, minimum_distance, temperature, objective):
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

    selected_distances = []
    selected_entries = []
    for curve_index, diffs, dists in curve_blocks:
        rows, cols = constraints._selected_distance_rows_and_cols(
            dists,
            hard_min=hard_min,
            temperature=temperature,
        )
        selected_distances.append(dists[rows, cols])
        selected_entries.append((curve_index, rows, cols, diffs[rows, cols], dists[rows, cols]))

    flat_distances = np.concatenate(selected_distances)
    smooth_min, flat_weights = constraints.smoothmin_selected(
        flat_distances,
        temperature,
        constraints._SMOOTHING_EPS,
    )

    curve_gradients = [np.zeros_like(np.asarray(curve.gamma(), dtype=float)) for curve in curves]
    surface_gradient = np.zeros_like(flat_surface)
    offset = 0
    for curve_index, rows, cols, diffs, distances in selected_entries:
        count = len(distances)
        local_weights = flat_weights[offset:offset + count]
        offset += count
        directions = constraints._distance_directions(diffs, distances)
        np.add.at(curve_gradients[curve_index], rows, local_weights[:, None] * directions)
        np.add.at(surface_gradient, cols, -local_weights[:, None] * directions)

    derivative = constraints._new_derivative()
    for curve, point_gradient in zip(curves, curve_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    if np.any(surface_gradient):
        derivative += constraints._surface_dgamma_by_dcoeff_derivative(
            surface,
            surface_gradient.reshape(surface_gamma.shape),
        )
    grad = np.asarray(derivative(objective), dtype=float)
    return constraints._smooth_min_signed_constraint(minimum_distance, smooth_min, grad)


def _dense_surface_surface_constraint(surface_1, surface_2, minimum_distance, temperature, objective):
    gamma_1 = np.asarray(surface_1.gamma(), dtype=float)
    gamma_2 = np.asarray(surface_2.gamma(), dtype=float)
    flat_gamma_1 = gamma_1.reshape((-1, 3))
    flat_gamma_2 = gamma_2.reshape((-1, 3))
    diffs = flat_gamma_1[:, None, :] - flat_gamma_2[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    hard_min = float(np.min(dists))
    rows, cols = constraints._selected_distance_rows_and_cols(
        dists,
        hard_min=hard_min,
        temperature=temperature,
    )
    selected_distances = dists[rows, cols]
    smooth_min, weights = constraints.smoothmin_selected(
        selected_distances,
        temperature,
        constraints._SMOOTHING_EPS,
    )

    directions = constraints._distance_directions(diffs[rows, cols], selected_distances)
    gradient_1 = np.zeros_like(flat_gamma_1)
    gradient_2 = np.zeros_like(flat_gamma_2)
    np.add.at(gradient_1, rows, weights[:, None] * directions)
    np.add.at(gradient_2, cols, -weights[:, None] * directions)

    derivative = constraints._new_derivative()
    derivative += constraints._surface_dgamma_by_dcoeff_derivative(
        surface_1,
        gradient_1.reshape(gamma_1.shape),
    )
    derivative += constraints._surface_dgamma_by_dcoeff_derivative(
        surface_2,
        gradient_2.reshape(gamma_2.shape),
    )
    grad = np.asarray(derivative(objective), dtype=float)
    return constraints._smooth_min_signed_constraint(minimum_distance, smooth_min, grad)


def test_curve_curve_streaming_matches_dense_value_and_gradient():
    curves = [
        _curve(1.0, 0.0, 0.0),
        _curve(0.8, 0.35, 0.1),
        _curve(0.7, -0.2, 0.45),
    ]
    objective = Optimizable(depends_on=curves)

    actual = constraints.smooth_min_curve_curve_signed_constraint(
        curves,
        minimum_distance=0.2,
        temperature=0.05,
        objective_optimizable=objective,
    )
    expected = _dense_curve_curve_constraint(curves, 0.2, 0.05, objective)

    _assert_constraint_tuple_allclose(actual, expected)


def test_curve_surface_streaming_matches_dense_value_and_gradient():
    curves = [_curve(1.0, 0.0, 0.0), _curve(0.8, 0.35, 0.1)]
    surface = _surface(1.7, 0.25)
    objective = Optimizable(depends_on=[*curves, surface])

    actual = constraints.smooth_min_curve_surface_signed_constraint(
        curves,
        surface,
        minimum_distance=0.2,
        temperature=0.05,
        objective_optimizable=objective,
    )
    expected = _dense_curve_surface_constraint(curves, surface, 0.2, 0.05, objective)

    _assert_constraint_tuple_allclose(actual, expected)


def test_surface_surface_streaming_matches_dense_value_and_gradient(monkeypatch):
    surface_1 = _surface(1.7, 0.25)
    surface_2 = _surface(2.1, 0.3)
    objective = Optimizable(depends_on=[surface_1, surface_2])
    monkeypatch.setattr(constraints, "_SURFACE_SURFACE_DISTANCE_BLOCK_ROWS", 5)

    actual = constraints.smooth_min_surface_surface_signed_constraint(
        surface_1,
        surface_2,
        minimum_distance=0.2,
        temperature=0.05,
        objective_optimizable=objective,
    )
    expected = _dense_surface_surface_constraint(surface_1, surface_2, 0.2, 0.05, objective)

    _assert_constraint_tuple_allclose(actual, expected)
