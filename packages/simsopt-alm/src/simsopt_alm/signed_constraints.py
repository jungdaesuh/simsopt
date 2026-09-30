"""Smooth signed geometry constraints ``g(x) <= 0`` for constrained coil optimization.

Every kernel returns ``(signed_value, grad, hard_signed_value)``.
``hard_signed_value`` is the constraint at the extremum over the sampled points
(quadrature points of curves and surface), and ``signed_value`` its log-sum-exp
surrogate at ``temperature`` T over every sample (curvature) or every point pair
(distances), with no support truncation. A pair enters through the smooth
distance ``s(d) = sqrt(d^2 + T^2) - T`` (``d - T <= s(d) <= d``, differentiable
at coincident points), so ``signed_value`` is a smooth function of the sampled
points and ``grad = d(signed_value)/dx`` over the free dofs of
``objective_optimizable`` holds everywhere (an empty array when it has
none). The surrogate is conservative:
``hard_signed_value <= signed_value <= hard_signed_value + T log N`` for N
curvature samples, ``+ T (log N + 1)`` for N distance pairs. Unlike the stock
hinge objectives, the value keeps the slack when the constraint is inactive.

Domain: ``temperature`` lies in [1e-100, 1e100] (in the constrained quantity's
units) and every sample coordinate is finite with ``|x| <= 1e100``; every
kernel raises ``ValueError`` naming the bound and the value otherwise, before
any early return. There every distance is below 3.5e100, ``sqrt(d^2 + T^2)``
and ``T log N`` are representable, and near contact d underflows gracefully.
Zero temperature is rejected, not read as the hard limit: that limit is
``hard_signed_value``.

Accuracy: a distance row's signed value has absolute error at most a small
multiple of machine epsilon times the row's scale, ``max(bound, the largest
sampled distance, T log N)``; its gradient, whose log-sum-exp weights depend on
``s / T``, at most a small multiple of epsilon times ``scale / T``. A value much
smaller than its scale (``bound - soft`` for bound and distances near 1e100) is
not resolved: it is a difference of nearly equal numbers. The curvature row is
the soft maximum of the curvature simsopt's ``curve.kappa()`` and
``dkappa_by_dcoeff_vjp`` provide, so its accuracy is simsopt's; a nonfinite
curvature or curvature derivative from them raises ``ValueError`` naming the
curve, so no nonfinite gradient is returned.
"""

import numpy as np

from simsopt._core.derivative import Derivative

__all__ = [
    "smooth_max_curvature_signed_constraint",
    "smooth_min_curve_curve_signed_constraint",
    "smooth_min_curve_surface_signed_constraint",
]


# Point pairs per distance block. A block holds its three coordinate
# differences, squared distances, regularized norms, smooth distances and
# weights (about 8 floats, 64 bytes, per pair), so the kernels' scratch memory
# stays near 4-5 MiB whatever the number of pairs.
_PAIR_BLOCK = 1 << 16

# The kernels' domain (module docstring).
TEMPERATURE_RANGE = (1.0e-100, 1.0e100)
COORDINATE_BOUND = 1.0e100


def require_smoothing_temperature(temperature) -> float:
    """``temperature`` as a float; ``ValueError`` unless in ``TEMPERATURE_RANGE``."""
    value = float(temperature)
    lower, upper = TEMPERATURE_RANGE
    if not lower <= value <= upper:
        raise ValueError(
            f"smoothing temperature must lie in [{lower:g}, {upper:g}]; got {temperature!r}"
        )
    return value


def require_sample_coordinates(points) -> np.ndarray:
    """``points`` as a float array; ``ValueError`` unless every coordinate is
    finite with ``|x| <= COORDINATE_BOUND``."""
    array = np.asarray(points, dtype=float)
    largest = float(np.max(np.abs(array), initial=0.0))
    if not largest <= COORDINATE_BOUND:
        raise ValueError(
            f"sample coordinates must be finite with |x| <= {COORDINATE_BOUND:g}; "
            f"got max |x| = {largest!r}"
        )
    return array


def soft_min_pair_distance(point_sets, set_pairs, temperature: float):
    """Soft minimum of the distances between every point pair of the listed sets.

    ``point_sets`` are ``(n_k, 3)`` arrays; ``set_pairs`` lists ``(a, b)``
    indices of sets whose every point pair counts. Returns ``(hard_min,
    soft_min, point_gradients)``: the smallest pair distance d, ``-T log sum
    exp(-s/T)`` over all pairs of the smooth distance ``s = sqrt(d^2 + T^2) -
    T`` (at most ``hard_min``, at least ``hard_min - T (log N + 1)``), and
    ``d(soft_min)/d points`` per set. ``temperature`` and the points must lie
    in the module's domain (``ValueError`` otherwise); no square or quotient
    of d or T is formed unscaled, so d underflows gracefully near contact.
    Pairs are visited in blocks of
    ``_PAIR_BLOCK`` with the running minimum of s as the exponent shift, so
    every exponent is <= 0.
    """
    temperature = require_smoothing_temperature(temperature)
    point_sets = [require_sample_coordinates(points) for points in point_sets]
    gradients = [np.zeros_like(points) for points in point_sets]
    shift = np.inf
    hard_min = np.inf
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
                # d from the coordinate differences by hypot, which neither
                # underflows nor overflows where d is representable (a sum of
                # squares loses d = 1e-170 and d = 1e200); hard_min is the
                # sampled minimum.
                differences = [
                    np.subtract.outer(left_block[:, axis], right_block[:, axis])
                    for axis in range(3)
                ]
                pair_distances = np.hypot(np.hypot(differences[0], differences[1]), differences[2])
                hard_min = min(hard_min, float(np.min(pair_distances)))
                # r = hypot(d, T) >= T > 0, and s = sqrt(d^2 + T^2) - T as
                # d (d / r) / (1 + T / r): no cancellation for d << T, and
                # both quotients lie in [0, 1].
                roots = np.hypot(pair_distances, temperature)
                distances = pair_distances / roots
                distances /= 1.0 + temperature / roots
                distances *= pair_distances
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
                # d(s_ij)/d(left_i) = (left_i - right_j) / r_ij, a vector no
                # longer than 1, summed with the weights directly (the matmul
                # identity left_i sum_j c_ij - (c @ right)_i cancels terms of
                # size |left| / T, garbage when T is small).
                for axis, directions in enumerate(differences):
                    directions /= roots
                    gradients[left_index][rows, axis] += np.einsum("ij,ij->i", weights, directions)
                    gradients[right_index][columns, axis] -= np.einsum(
                        "ij,ij->j", weights, directions
                    )
    for gradient in gradients:
        gradient /= weight_sum
    soft_min = shift - temperature * float(np.log(weight_sum))
    return hard_min, soft_min, gradients


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


def _gradient_over_free_dofs(derivative: Derivative, objective_optimizable) -> np.ndarray:
    """``derivative`` over the free dofs of ``objective_optimizable``, empty
    when it has none (as ``_no_pair_result``): simsopt cannot evaluate a
    ``Derivative`` over no dofs."""
    if np.size(objective_optimizable.x) == 0:
        return np.zeros(0)
    return np.asarray(derivative(objective_optimizable), dtype=float)


def _curve_derivative(curves, point_gradients) -> Derivative:
    derivative = Derivative({})
    for curve, point_gradient in zip(curves, point_gradients):
        if np.any(point_gradient):
            derivative += curve.dgamma_by_dcoeff_vjp(point_gradient)
    return derivative


def _finite_curvature_output(curve, quantity: str, values: np.ndarray) -> np.ndarray:
    """``values`` (simsopt's curvature of ``curve``, or its derivative);
    ``ValueError`` naming the curve if any is nonfinite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"simsopt returned a nonfinite {quantity} for curve {curve.name}; the "
            "curvature row takes simsopt's curvature as given"
        )
    return values


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
    require_sample_coordinates(curve.gamma())
    kappa = _finite_curvature_output(
        curve, "curvature", np.asarray(curve.kappa() if kappa is None else kappa, dtype=float)
    )
    hard_max = float(np.max(kappa))
    exp_shifted = np.exp((kappa - hard_max) / temperature)
    weight_sum = float(np.sum(exp_shifted))
    smooth_max = hard_max + temperature * float(np.log(weight_sum))
    grad = _finite_curvature_output(
        curve,
        "curvature derivative",
        _gradient_over_free_dofs(
            curve.dkappa_by_dcoeff_vjp(exp_shifted / weight_sum), objective_optimizable
        ),
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
    curve_points = [require_sample_coordinates(curve.gamma()) for curve in curves]
    if len(curves) < 2:
        return _no_pair_result(minimum_distance, objective_optimizable)
    hard_min, smooth_min, point_gradients = soft_min_pair_distance(
        curve_points,
        [(i, j) for i in range(len(curve_points)) for j in range(i)],
        temperature,
    )
    grad = _gradient_over_free_dofs(
        _curve_derivative(curves, point_gradients), objective_optimizable
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
    surface_gamma = require_sample_coordinates(surface.gamma())
    point_sets = [require_sample_coordinates(curve.gamma()) for curve in curves]
    if not curves:
        return _no_pair_result(minimum_distance, objective_optimizable)
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
    grad = _gradient_over_free_dofs(derivative, objective_optimizable)
    # grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
    # so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
    signed_value = float(minimum_distance) - smooth_min
    hard_signed_value = float(minimum_distance) - hard_min
    return signed_value, -grad, hard_signed_value
