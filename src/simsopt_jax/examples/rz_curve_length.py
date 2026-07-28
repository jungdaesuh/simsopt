"""Device-resident RZ-Fourier curve-length optimization workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.core.curve_geometry import (
    curve_incremental_arclength_from_spec,
    curve_spec_with_dofs,
)
from simsopt_jax.core.specs import make_curve_rzfourier_spec
from simsopt_jax.core.specs import CurveRZFourierSpec
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
from simsopt_jax_adapters.geo.curve_objectives import curve_length_pure


@dataclass(frozen=True)
class RZCurveLengthDeviceResult:
    """Completed curve values that remain resident on the active device."""

    initial_parameters: jax.Array
    initial_length: jax.Array
    initial_residual_jacobian: jax.Array
    final_parameters: jax.Array
    final_length: jax.Array
    final_residual_jacobian: jax.Array
    optimizer: OptimizerResult


@jax.jit
def _partition_curve_dofs(
    full_dofs: jax.Array,
    free_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    free_dofs = full_dofs[free_positions]
    fixed_dofs = full_dofs.at[free_positions].set(jnp.zeros_like(free_dofs))
    return fixed_dofs, free_dofs


@jax.jit
def _curve_length_from_free_dofs(
    free_dofs: jax.Array,
    fixed_dofs: jax.Array,
    free_positions: jax.Array,
    spec: CurveRZFourierSpec,
) -> jax.Array:
    current_full = fixed_dofs.at[free_positions].set(free_dofs)
    current_spec = curve_spec_with_dofs(spec, current_full)
    incremental_arclength = curve_incremental_arclength_from_spec(current_spec)
    return curve_length_pure(incremental_arclength)


_curve_length_value_and_grad = jax.jit(
    jax.value_and_grad(_curve_length_from_free_dofs, argnums=0)
)


def solve_rz_curve_length(
    *,
    full_dofs: jax.Array,
    quadpoints: jax.Array,
    free_positions: jax.Array,
    order: int,
    nfp: int,
    stellsym: bool,
    max_steps: int,
    rtol: float,
    atol: float,
) -> RZCurveLengthDeviceResult:
    """Minimize squared curve length without dense free-DOF expansion arrays."""
    full_dofs_device = jnp.asarray(full_dofs, dtype=jnp.float64)
    free_positions_device = jnp.asarray(free_positions, dtype=jnp.int32)
    fixed_dofs, initial_parameters = _partition_curve_dofs(
        full_dofs_device,
        free_positions_device,
    )
    spec = make_curve_rzfourier_spec(
        dofs=full_dofs_device,
        quadpoints=jnp.asarray(quadpoints, dtype=jnp.float64),
        order=order,
        nfp=nfp,
        stellsym=stellsym,
    )

    def length(parameters: jax.Array) -> jax.Array:
        return _curve_length_from_free_dofs(
            parameters,
            fixed_dofs,
            free_positions_device,
            spec,
        )

    def squared_length(parameters: jax.Array) -> jax.Array:
        length_value = length(parameters)
        return length_value * length_value

    initial_length, initial_residual_jacobian = _curve_length_value_and_grad(
        initial_parameters,
        fixed_dofs,
        free_positions_device,
        spec,
    )
    problem = TraceableScalarProblem(
        objective_fn=squared_length,
        x=initial_parameters,
    )
    optimizer = serial_solve_jax(
        problem,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        require_success=False,
    )
    final_parameters = problem.x
    final_length, final_residual_jacobian = _curve_length_value_and_grad(
        final_parameters,
        fixed_dofs,
        free_positions_device,
        spec,
    )
    completed = jax.block_until_ready(
        (
            initial_parameters,
            initial_length,
            initial_residual_jacobian,
            final_parameters,
            final_length,
            final_residual_jacobian,
        )
    )
    return RZCurveLengthDeviceResult(
        initial_parameters=completed[0],
        initial_length=completed[1],
        initial_residual_jacobian=completed[2],
        final_parameters=completed[3],
        final_length=completed[4],
        final_residual_jacobian=completed[5],
        optimizer=optimizer,
    )


__all__ = ["RZCurveLengthDeviceResult", "solve_rz_curve_length"]
