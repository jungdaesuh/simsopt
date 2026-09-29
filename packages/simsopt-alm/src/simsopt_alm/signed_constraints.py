"""Smooth signed geometry constraints ``g(x) <= 0`` for constrained coil optimization.

Every kernel returns ``(signed_value, grad, hard_signed_value)``.
``hard_signed_value`` is the constraint at the extremum over the sampled points
(quadrature points of curves and surface), and ``signed_value`` its log-sum-exp
surrogate at ``temperature`` T over every sample (curvature) or every point pair
(distances), with no support truncation. A pair enters through the smooth
distance ``s(d) = sqrt(d^2 + T^2) - T`` (``d - T <= s(d) <= d``, differentiable
at coincident points), so ``signed_value`` is a smooth function of the sampled
points and ``grad = d(signed_value)/dx`` over the free dofs of
``objective_optimizable`` holds everywhere. The surrogate is conservative:
``hard_signed_value <= signed_value <= hard_signed_value + T log N`` for N
curvature samples, ``+ T (log N + 1)`` for N distance pairs. Unlike the stock hinge objectives, the value keeps the slack
when the constraint is inactive. ``temperature`` must be finite and positive
(in the constrained quantity's units); every kernel raises ``ValueError``
otherwise. Zero is rejected, not read as the hard limit: that limit is
``hard_signed_value``.
"""

import numpy as np

from simsopt._core.derivative import Derivative

__all__ = [
    "smooth_max_curvature_signed_constraint",
    "smooth_min_curve_curve_signed_constraint",
    "smooth_min_curve_surface_signed_constraint",
]


# Point pairs per distance block. A block holds its pair differences, distances
# and weights (about 6 floats, 48 bytes, per pair), so the kernels' scratch
# memory stays near 3 MiB whatever the number of pairs.
_PAIR_BLOCK = 1 << 16

def require_smoothing_temperature(temperature) -> float:
    """``temperature`` as a float; ``ValueError`` unless finite and positive."""
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"smoothing temperature must be finite and positive; got {temperature!r}"
        )
    return value


def soft_min_pair_distance(point_sets, set_pairs, temperature: float):
    """Soft minimum of the distances between every point pair of the listed sets.

    ``point_sets`` are ``(n_k, 3)`` arrays; ``set_pairs`` lists ``(a, b)``
    indices of sets whose every point pair counts. Returns ``(hard_min,
    soft_min, point_gradients)``: the smallest pair distance d, ``-T log sum
    exp(-s/T)`` over all pairs of the smooth distance ``s = sqrt(d^2 + T^2) -
    T`` (at most ``hard_min``, at least ``hard_min - T (log N + 1)``), and
    ``d(soft_min)/d points`` per set. Pairs are visited in blocks of
    ``_PAIR_BLOCK`` with the running minimum of s as the exponent shift, so
    every exponent is <= 0.
    """
    gradients = [np.zeros_like(points) for points in point_sets]
    shift = np.inf
    min_squared_distance = np.inf
    weight_sum = 0.0
    for left_index, right_index in set_pairs:
        left, right = point_sets[left_index], point_sets[right_index]
        column_step = min(len(right), _PAIR_BLOCK)
        row_step = max(1, _PAIR_BLOCK // column_step)
        for row_start in range(0, len(left), row_step):
            rows = slice(row_start, row_start + row_step)
            for column_start in range(0, len(right), column_step):
                columns = slice(column_start, column_start + column_step)
                left_block, right_block = left[rows], right[columns]
                # Squared distances from the coordinate differences (no |x|^2 +
                # |y|^2 - 2 x.y cancellation), so hard_min is the sampled minimum.
                squared = np.zeros((len(left_block), len(right_block)))
                for axis in range(3):
                    difference = np.subtract.outer(left_block[:, axis], right_block[:, axis])
                    squared += np.square(difference, out=difference)
                min_squared_distance = min(min_squared_distance, float(np.min(squared)))
                # s = d^2 / (r + T), r = sqrt(d^2 + T^2): sqrt(d^2 + T^2) - T
                # without its cancellation for d << T.
                roots = np.sqrt(squared + temperature * temperature)
                distances = squared / (roots + temperature)
                block_min = float(np.min(distances))
                if block_min < shift:
                    # Re-reference the sums to the new minimum (0 on the first block).
                    rescale = float(np.exp((block_min - shift) / temperature))
                    weight_sum *= rescale
                    for gradient in gradients:
                        gradient *= rescale
                    shift = block_min
                weights = np.subtract(shift, distances)
                weights /= temperature
                np.exp(weights, out=weights)
                weight_sum += float(np.sum(weights))
                # d(s_ij)/d(left_i) = (left_i - right_j) / r_ij (r >= T > 0), so
                # the weighted sums over j (i) are left_i * sum_j c_ij -
                # (c @ right)_i and right_j * sum_i c_ij - (c.T @ left)_j with
                # c = weights / r.
                coefficients = np.divide(weights, roots, out=weights)
                gradients[left_index][rows] += (
                    left_block * np.sum(coefficients, axis=1)[:, None]
                    - coefficients @ right_block
                )
                gradients[right_index][columns] += (
                    right_block * np.sum(coefficients, axis=0)[:, None]
                    - coefficients.T @ left_block
                )
    for gradient in gradients:
        gradient /= weight_sum
    soft_min = shift - temperature * float(np.log(weight_sum))
    return float(np.sqrt(min_squared_distance)), soft_min, gradients


def surface_dgamma_by_dcoeff_derivative(surface, point_gradient):
    surface_vjp = surface.dgamma_by_dcoeff_vjp(point_gradient)
    if isinstance(surface_vjp, Derivative):
        return surface_vjp
    return Derivative({surface: np.asarray(surface_vjp, dtype=float)})


def _no_pair_result(minimum_distance, objective_optimizable):
    # No pair can violate the spacing, so the constraint holds with the full
    # clearance as slack, in both the smooth and the hard signal.
    signed_value = -float(minimum_distance)
    return (
        signed_value,
        np.zeros_like(np.asarray(objective_optimizable.x)),
        signed_value,
    )


def _curve_derivative(curves, point_gradients) -> Derivative:
    derivative = Derivative({})
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    return derivative


def smooth_max_curvature_signed_constraint(
    curve,
    threshold,
    temperature,
    objective_optimizable,
    *,
    kappa=None,
):
    """Signed ``max(kappa) - threshold`` for one curve, over its quadrature points.

    ``kappa`` optionally passes ``curve.kappa()`` already evaluated at the
    current dofs, so a caller that also reports the hard maximum evaluates it once.
    """
    temperature = require_smoothing_temperature(temperature)
    kappa = np.asarray(curve.kappa() if kappa is None else kappa, dtype=float)
    hard_max = float(np.max(kappa))
    exp_shifted = np.exp((kappa - hard_max) / temperature)
    weight_sum = float(np.sum(exp_shifted))
    smooth_max = hard_max + temperature * float(np.log(weight_sum))
    grad = np.asarray(
        curve.dkappa_by_dcoeff_vjp(exp_shifted / weight_sum)(objective_optimizable),
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
    """Signed ``minimum_distance - min_{i<j} dist(curve_i, curve_j)`` over the
    curves' quadrature points.

    Fewer than two curves has no pair, so it returns ``-minimum_distance`` and a
    zero gradient.
    """
    temperature = require_smoothing_temperature(temperature)
    if len(curves) < 2:
        return _no_pair_result(minimum_distance, objective_optimizable)
    curve_points = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    hard_min, smooth_min, point_gradients = soft_min_pair_distance(
        curve_points,
        [(i, j) for i in range(len(curve_points)) for j in range(i)],
        temperature,
    )
    grad = np.asarray(
        _curve_derivative(curves, point_gradients)(objective_optimizable), dtype=float
    )
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - hard_min
    return signed_value, -grad, hard_signed_value


def smooth_min_curve_surface_signed_constraint(
    curves,
    surface,
    minimum_distance,
    temperature,
    objective_optimizable,
):
    """Signed ``minimum_distance - min_i dist(curve_i, surface)`` over the
    curves' and the surface's quadrature points.

    The gradient includes the surface dofs when ``objective_optimizable`` owns
    them. No curves returns ``-minimum_distance`` and a zero gradient.
    """
    temperature = require_smoothing_temperature(temperature)
    if not curves:
        return _no_pair_result(minimum_distance, objective_optimizable)
    surface_gamma = np.asarray(surface.gamma(), dtype=float)
    point_sets = [np.asarray(curve.gamma(), dtype=float) for curve in curves]
    point_sets.append(surface_gamma.reshape((-1, 3)))
    surface_index = len(curves)
    hard_min, smooth_min, point_gradients = soft_min_pair_distance(
        point_sets,
        [(curve_index, surface_index) for curve_index in range(len(curves))],
        temperature,
    )
    derivative = _curve_derivative(curves, point_gradients[:surface_index])
    surface_gradient = point_gradients[surface_index]
    if np.any(surface_gradient):
        derivative += surface_dgamma_by_dcoeff_derivative(
            surface,
            surface_gradient.reshape(surface_gamma.shape),
        )
    grad = np.asarray(derivative(objective_optimizable), dtype=float)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - hard_min
    return signed_value, -grad, hard_signed_value
