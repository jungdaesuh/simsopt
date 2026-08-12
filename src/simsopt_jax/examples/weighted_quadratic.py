"""Device-resident weighted-quadratic example workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.runtime.host_boundary import block_until_ready
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)


@dataclass(frozen=True)
class WeightedQuadraticDeviceResult:
    """Completed device values plus the normalized optimizer result."""

    initial_parameters: jax.Array
    targets: jax.Array
    weights: jax.Array
    initial_residuals: jax.Array
    initial_jacobian: jax.Array
    initial_objective: jax.Array
    final_parameters: jax.Array
    final_residuals: jax.Array
    final_jacobian: jax.Array
    final_objective: jax.Array
    final_gradient: jax.Array
    optimizer: OptimizerResult


@jax.jit
def weighted_quadratic_residuals(
    parameters: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Return ``sqrt(weight) * (parameter - target)`` on the active device."""
    return jnp.sqrt(weights) * (parameters - targets)


@jax.jit
def weighted_quadratic_gradient(
    jacobian: jax.Array,
    residuals: jax.Array,
) -> jax.Array:
    """Return the gradient of the sum-of-squares objective."""
    return 2.0 * jacobian.T @ residuals


def solve_weighted_quadratic(
    *,
    initial_parameters: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
) -> WeightedQuadraticDeviceResult:
    """Solve a small diagonal weighted least-squares problem on one device."""
    initial_device = jnp.asarray(initial_parameters, dtype=jnp.float64)
    targets_device = jnp.asarray(targets, dtype=jnp.float64)
    weights_device = jnp.asarray(weights, dtype=jnp.float64)
    initial_residuals = weighted_quadratic_residuals(
        initial_device,
        targets_device,
        weights_device,
    )
    jacobian = jnp.diag(jnp.sqrt(weights_device))
    initial_objective = jnp.vdot(initial_residuals, initial_residuals).real

    def residual(parameters: jax.Array) -> jax.Array:
        return weighted_quadratic_residuals(
            parameters,
            targets_device,
            weights_device,
        )

    problem = TraceableLeastSquaresProblem(
        residual_fn=residual,
        x=initial_device,
    )
    optimizer = least_squares_serial_solve_jax(
        problem,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    final_parameters = problem.x
    final_residuals = problem.residuals()
    final_objective = problem.objective()
    final_gradient = weighted_quadratic_gradient(jacobian, final_residuals)
    completed = block_until_ready(
        (
            initial_residuals,
            jacobian,
            initial_objective,
            final_parameters,
            final_residuals,
            final_objective,
            final_gradient,
        )
    )
    return WeightedQuadraticDeviceResult(
        initial_parameters=initial_device,
        targets=targets_device,
        weights=weights_device,
        initial_residuals=completed[0],
        initial_jacobian=completed[1],
        initial_objective=completed[2],
        final_parameters=completed[3],
        final_residuals=completed[4],
        final_jacobian=completed[1],
        final_objective=completed[5],
        final_gradient=completed[6],
        optimizer=optimizer,
    )


__all__ = [
    "WeightedQuadraticDeviceResult",
    "solve_weighted_quadratic",
    "weighted_quadratic_gradient",
    "weighted_quadratic_residuals",
]
