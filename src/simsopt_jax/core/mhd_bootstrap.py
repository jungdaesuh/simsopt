"""Pure JAX kernels for MHD bootstrap-current support quantities."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ._math_utils import as_jax_float64

__all__ = ["compute_trapped_fraction_jax"]


def _not_a_knot_coefficients(values: jax.Array) -> jax.Array:
    """Return piecewise cubic coefficients for unit-spaced not-a-knot data."""
    y = as_jax_float64(values)
    intervals = int(y.shape[0]) - 1
    delta = y[1:] - y[:-1]

    left_c = 0.5 * (delta[1] - delta[0])
    tail_size = intervals - 2
    if intervals == 3:
        c_tail = jnp.reshape(0.5 * (delta[-1] - delta[-2]), (1,))
    else:
        interior_rhs = 3.0 * (delta[2 : intervals - 1] - delta[1 : intervals - 2])
        tail_rhs = jnp.concatenate(
            (interior_rhs, jnp.reshape(delta[-1] - delta[-2], (1,))),
            axis=0,
        )
        tail_rhs = tail_rhs.at[0].add(-left_c)
        tail_lower = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=y.dtype),
                jnp.ones((tail_size - 2,), dtype=y.dtype),
                jnp.zeros((1,), dtype=y.dtype),
            ),
            axis=0,
        )
        tail_diag = jnp.concatenate(
            (
                jnp.full((tail_size - 1,), 4.0, dtype=y.dtype),
                jnp.full((1,), 2.0, dtype=y.dtype),
            ),
            axis=0,
        )
        tail_upper = jnp.concatenate(
            (
                jnp.ones((tail_size - 1,), dtype=y.dtype),
                jnp.zeros((1,), dtype=y.dtype),
            ),
            axis=0,
        )
        c_tail = jax.lax.linalg.tridiagonal_solve(
            tail_lower, tail_diag, tail_upper, tail_rhs[:, None]
        )[:, 0]

    c = jnp.concatenate(
        (
            jnp.reshape(2.0 * left_c - c_tail[0], (1,)),
            jnp.reshape(left_c, (1,)),
            c_tail,
        ),
        axis=0,
    )
    d_internal = (c[1:] - c[:-1]) / 3.0
    d = jnp.concatenate((d_internal, d_internal[-1:]), axis=0)
    b = delta - c - d
    return jnp.stack((y[:-1], b, c, d), axis=1)


def _eval_cubic(coeffs: jax.Array, x: jax.Array, derivative: int = 0) -> jax.Array:
    intervals = int(coeffs.shape[0])
    x_clamped = jnp.clip(x, 0.0, float(intervals))
    interval = jnp.minimum(jnp.floor(x_clamped).astype(jnp.int32), intervals - 1)
    t = x_clamped - interval.astype(x_clamped.dtype)
    a, b, c, d = jnp.moveaxis(jnp.take(coeffs, interval, axis=0), -1, 0)
    if derivative == 0:
        return ((d * t + c) * t + b) * t + a
    if derivative == 1:
        return (3.0 * d * t + 2.0 * c) * t + b
    if derivative == 2:
        return 6.0 * d * t + 2.0 * c
    raise ValueError("derivative must be 0, 1, or 2")


def _bounded_newton_cubic_extremum(
    coeffs: jax.Array, start_index: jax.Array, iterations: int
) -> jax.Array:
    max_x = float(int(coeffs.shape[0]))
    initial = start_index.astype(coeffs.dtype)
    step_factors = jnp.asarray(
        (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125), dtype=coeffs.dtype
    )

    def step(point, _):
        first = _eval_cubic(coeffs, point, 1)
        second = _eval_cubic(coeffs, point, 2)
        has_curvature = jnp.abs(second) > 1e-14
        safe_second = jnp.where(has_curvature, second, 1.0)
        newton = jnp.where(has_curvature, point - first / safe_second, point)
        gradient_step = first / jnp.maximum(jnp.abs(first), 1.0)
        candidates = jnp.concatenate(
            (
                jnp.reshape(point, (1,)),
                point + step_factors * (newton - point),
                point - step_factors * gradient_step,
            )
        )
        candidates = jnp.clip(candidates, 0.0, max_x)
        values = _eval_cubic(coeffs, candidates)
        return candidates[jnp.argmin(values)], None

    point, _ = jax.lax.scan(step, initial, None, length=int(iterations))
    return _eval_cubic(coeffs, point)


def _min_max_cubic(values: jax.Array, iterations: int) -> tuple[jax.Array, jax.Array]:
    coeffs = _not_a_knot_coefficients(values)
    min_index = jnp.argmin(values)
    max_index = jnp.argmax(values)
    minimum = _bounded_newton_cubic_extremum(coeffs, min_index, iterations)
    maximum = -_bounded_newton_cubic_extremum(-coeffs, max_index, iterations)
    return minimum, maximum


def _eval_tensor_cubic(
    x_coeffs_by_y: jax.Array,
    x: jax.Array,
    y: jax.Array,
    derivative_x: int,
    derivative_y: int,
) -> jax.Array:
    y_values = jax.vmap(lambda coeffs: _eval_cubic(coeffs, x, derivative_x))(
        x_coeffs_by_y
    )
    y_coeffs = _not_a_knot_coefficients(y_values)
    return _eval_cubic(y_coeffs, y, derivative_y)


def _tensor_value_grad_hess(
    x_coeffs_by_y: jax.Array, x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    value = _eval_tensor_cubic(x_coeffs_by_y, x, y, 0, 0)
    grad = jnp.array(
        (
            _eval_tensor_cubic(x_coeffs_by_y, x, y, 1, 0),
            _eval_tensor_cubic(x_coeffs_by_y, x, y, 0, 1),
        ),
        dtype=value.dtype,
    )
    hess = jnp.array(
        (
            (
                _eval_tensor_cubic(x_coeffs_by_y, x, y, 2, 0),
                _eval_tensor_cubic(x_coeffs_by_y, x, y, 1, 1),
            ),
            (
                _eval_tensor_cubic(x_coeffs_by_y, x, y, 1, 1),
                _eval_tensor_cubic(x_coeffs_by_y, x, y, 0, 2),
            ),
        ),
        dtype=value.dtype,
    )
    return value, grad, hess


def _bounded_newton_extremum(
    grid: jax.Array, start_index: jax.Array, iterations: int
) -> jax.Array:
    x_coeffs_by_y = jax.vmap(_not_a_knot_coefficients, in_axes=1, out_axes=0)(grid)
    lower = jnp.array((0.0, 0.0), dtype=grid.dtype)
    upper = jnp.array((float(int(grid.shape[0]) - 1), float(int(grid.shape[1]) - 1)))
    initial = jnp.array(
        (start_index[0].astype(grid.dtype), start_index[1].astype(grid.dtype)),
        dtype=grid.dtype,
    )
    step_factors = jnp.asarray(
        (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125), dtype=grid.dtype
    )

    def step(point, _):
        _, grad, hess = _tensor_value_grad_hess(x_coeffs_by_y, point[0], point[1])
        det = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]
        has_full_inverse = jnp.abs(det) > 1e-14
        safe_det = jnp.where(has_full_inverse, det, 1.0)
        inverse_step = (
            jnp.array(
                (
                    hess[1, 1] * grad[0] - hess[0, 1] * grad[1],
                    -hess[1, 0] * grad[0] + hess[0, 0] * grad[1],
                ),
                dtype=x_coeffs_by_y.dtype,
            )
            / safe_det
        )
        has_theta_curvature = jnp.abs(hess[0, 0]) > 1e-14
        has_zeta_curvature = jnp.abs(hess[1, 1]) > 1e-14
        safe_theta_curvature = jnp.where(has_theta_curvature, hess[0, 0], 1.0)
        safe_zeta_curvature = jnp.where(has_zeta_curvature, hess[1, 1], 1.0)
        diagonal_step = jnp.array(
            (
                jnp.where(has_theta_curvature, grad[0] / safe_theta_curvature, 0.0),
                jnp.where(has_zeta_curvature, grad[1] / safe_zeta_curvature, 0.0),
            ),
            dtype=x_coeffs_by_y.dtype,
        )
        candidate = jnp.where(
            has_full_inverse, point - inverse_step, point - diagonal_step
        )
        grad_norm = jnp.maximum(jnp.linalg.norm(grad), 1.0)
        gradient_step = grad / grad_norm
        candidate_points = jnp.concatenate(
            (
                point[None, :],
                point[None, :] + step_factors[:, None] * (candidate - point)[None, :],
                point[None, :] - step_factors[:, None] * gradient_step[None, :],
            ),
            axis=0,
        )
        candidate_points = jnp.clip(candidate_points, lower, upper)
        candidate_values = jax.vmap(
            lambda trial: _eval_tensor_cubic(x_coeffs_by_y, trial[0], trial[1], 0, 0)
        )(candidate_points)
        return candidate_points[jnp.argmin(candidate_values)], None

    point, _ = jax.lax.scan(step, initial, None, length=int(iterations))
    return _eval_tensor_cubic(x_coeffs_by_y, point[0], point[1], 0, 0)


def _min_max_tensor_cubic(
    grid: jax.Array, iterations: int
) -> tuple[jax.Array, jax.Array]:
    min_index = jnp.array(jnp.unravel_index(jnp.argmin(grid), grid.shape))
    max_index = jnp.array(jnp.unravel_index(jnp.argmax(grid), grid.shape))
    minimum = _bounded_newton_extremum(grid, min_index, iterations)
    maximum = -_bounded_newton_extremum(-grid, max_index, iterations)
    return minimum, maximum


def _quadrature_nodes(node_count: int, dtype) -> tuple[jax.Array, jax.Array]:
    nodes, weights = np.polynomial.legendre.leggauss(int(node_count))
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype)


def _compute_trapped_fraction_surface(
    modB,
    sqrtg,
    Bmin,
    Bmax,
    *,
    axes,
    quadrature_node_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    sqrtg_mean = jnp.mean(sqrtg, axis=axes)
    fsa_B2 = jnp.mean(modB * modB * sqrtg, axis=axes) / sqrtg_mean
    fsa_1overB = jnp.mean(sqrtg / modB, axis=axes) / sqrtg_mean
    nodes, weights = _quadrature_nodes(quadrature_node_count, modB.dtype)
    grid_shape = (1,) * (modB.ndim - 1)
    lambda_shape = (int(quadrature_node_count),) + grid_shape + (1,)
    surface_shape = (1,) + grid_shape + Bmax.shape
    lambdas = jnp.reshape(nodes, lambda_shape) / jnp.reshape(Bmax, surface_shape)
    denominator_axes = (
        tuple(axis + 1 for axis in axes) if isinstance(axes, tuple) else axes + 1
    )
    lambda_gap = 1.0 - lambdas * modB[None, ...]
    has_lambda_gap = lambda_gap > 0.0
    safe_lambda_gap = jnp.where(has_lambda_gap, lambda_gap, 1.0)
    denominator = jnp.mean(
        jnp.where(has_lambda_gap, jnp.sqrt(safe_lambda_gap), 0.0) * sqrtg[None, ...],
        axis=denominator_axes,
    )
    denominator = denominator / sqrtg_mean
    integral = (
        jnp.sum(
            jnp.reshape(weights, (-1, 1))
            * (lambdas.reshape((int(quadrature_node_count), -1)) / denominator),
            axis=0,
        )
        / Bmax
    )
    f_t = 1.0 - 0.75 * fsa_B2 * integral
    epsilon = (Bmax / Bmin - 1.0) / (Bmax / Bmin + 1.0)
    return epsilon, fsa_B2, fsa_1overB, f_t


def compute_trapped_fraction_jax(
    modB,
    sqrtg,
    *,
    quadrature_node_count: int = 96,
    extrema_iterations: int = 12,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute trapped-fraction quantities from 2-D or 3-D ``|B|`` grids."""
    modB_jax = as_jax_float64(modB)
    sqrtg_jax = as_jax_float64(sqrtg)
    if modB_jax.shape != sqrtg_jax.shape:
        raise ValueError("modB and sqrtg must have the same shape")
    if modB_jax.ndim == 2:
        modB_big = jnp.concatenate((modB_jax, modB_jax[:1, :]), axis=0)
        per_surface = jax.vmap(
            lambda values: _min_max_cubic(values, extrema_iterations),
            in_axes=1,
            out_axes=0,
        )(modB_big)
        Bmin, Bmax = per_surface
        epsilon, fsa_B2, fsa_1overB, f_t = _compute_trapped_fraction_surface(
            modB_jax,
            sqrtg_jax,
            Bmin,
            Bmax,
            axes=0,
            quadrature_node_count=quadrature_node_count,
        )
    elif modB_jax.ndim == 3:
        theta_periodic = jnp.concatenate((modB_jax, modB_jax[:1, :, :]), axis=0)
        modB_big = jnp.concatenate((theta_periodic, theta_periodic[:, :1, :]), axis=1)
        per_surface = jax.vmap(
            lambda grid: _min_max_tensor_cubic(grid, extrema_iterations),
            in_axes=2,
            out_axes=0,
        )(modB_big)
        Bmin, Bmax = per_surface
        epsilon, fsa_B2, fsa_1overB, f_t = _compute_trapped_fraction_surface(
            modB_jax,
            sqrtg_jax,
            Bmin,
            Bmax,
            axes=(0, 1),
            quadrature_node_count=quadrature_node_count,
        )
    else:
        raise ValueError("Input arrays must be 2D or 3D")
    return Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t
