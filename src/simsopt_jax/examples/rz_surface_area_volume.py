"""Device-resident RZ-Fourier surface area-volume optimization workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.core.specs import SurfaceRZFourierSpec
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_area_from_dofs,
    surface_rz_fourier_spec_from_dofs,
    surface_rz_fourier_volume_from_dofs,
)
from simsopt_jax.runtime.host_boundary import block_until_ready
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)


@dataclass(frozen=True)
class SurfaceAreaVolumeStageDeviceResult:
    """One completed area-volume solve whose numerical values remain on device."""

    initial_parameters: jax.Array
    solver_initial_parameters: jax.Array
    initial_area: jax.Array
    initial_volume: jax.Array
    initial_residuals: jax.Array
    initial_jacobian: jax.Array
    final_parameters: jax.Array
    final_area: jax.Array
    final_volume: jax.Array
    final_residuals: jax.Array
    final_jacobian: jax.Array
    optimizer: OptimizerResult


@dataclass(frozen=True)
class SurfaceAreaVolumeSequenceDeviceResult:
    """The two sequential solves in the native ``surf_vol_area`` workflow."""

    first: SurfaceAreaVolumeStageDeviceResult
    second: SurfaceAreaVolumeStageDeviceResult


_AXIS_SYMMETRY_BREAKING_OFFSET = 1.0e-10


@jax.jit
def _partition_surface_dofs(
    full_dofs: jax.Array,
    free_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    free_dofs = full_dofs[free_positions]
    fixed_dofs = full_dofs.at[free_positions].set(jnp.zeros_like(free_dofs))
    return fixed_dofs, free_dofs


@jax.jit
def _first_stage_solver_seed(initial_parameters: jax.Array) -> jax.Array:
    """Break the circular-axis degeneracy of the exact autodiff Jacobian."""
    offset = jnp.asarray(
        (-_AXIS_SYMMETRY_BREAKING_OFFSET, _AXIS_SYMMETRY_BREAKING_OFFSET),
        dtype=initial_parameters.dtype,
    )
    return initial_parameters + offset


@jax.jit
def _surface_quantities_from_free_dofs(
    free_dofs: jax.Array,
    fixed_dofs: jax.Array,
    free_positions: jax.Array,
    spec: SurfaceRZFourierSpec,
) -> jax.Array:
    current_full = fixed_dofs.at[free_positions].set(free_dofs)
    return jnp.stack(
        (
            surface_rz_fourier_area_from_dofs(spec, current_full),
            surface_rz_fourier_volume_from_dofs(spec, current_full),
        )
    )


@jax.jit
def _quantities_and_jacobian(
    parameters: jax.Array,
    fixed_dofs: jax.Array,
    free_positions: jax.Array,
    spec: SurfaceRZFourierSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    quantities = _surface_quantities_from_free_dofs(
        parameters,
        fixed_dofs,
        free_positions,
        spec,
    )
    jacobian = jax.jacfwd(
        _surface_quantities_from_free_dofs,
        argnums=0,
    )(
        parameters,
        fixed_dofs,
        free_positions,
        spec,
    )
    return quantities[0], quantities[1], quantities, jacobian


def _solve_stage(
    *,
    initial_parameters: jax.Array,
    solver_initial_parameters: jax.Array,
    targets: jax.Array,
    fixed_dofs: jax.Array,
    free_positions: jax.Array,
    spec: SurfaceRZFourierSpec,
    max_steps: int,
    rtol: float,
    atol: float,
) -> SurfaceAreaVolumeStageDeviceResult:
    def quantities(parameters: jax.Array) -> jax.Array:
        return _surface_quantities_from_free_dofs(
            parameters,
            fixed_dofs,
            free_positions,
            spec,
        )

    def residuals(parameters: jax.Array) -> jax.Array:
        return quantities(parameters) - targets

    initial_area, initial_volume, initial_quantities, initial_jacobian = (
        _quantities_and_jacobian(
            initial_parameters,
            fixed_dofs,
            free_positions,
            spec,
        )
    )
    problem = TraceableLeastSquaresProblem(
        residual_fn=residuals,
        x=solver_initial_parameters,
    )
    optimizer = least_squares_serial_solve_jax(
        problem,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    final_parameters = problem.x
    final_area, final_volume, final_quantities, final_jacobian = (
        _quantities_and_jacobian(
            final_parameters,
            fixed_dofs,
            free_positions,
            spec,
        )
    )
    completed = block_until_ready(
        (
            initial_parameters,
            solver_initial_parameters,
            initial_area,
            initial_volume,
            initial_quantities,
            initial_quantities - targets,
            initial_jacobian,
            final_parameters,
            final_area,
            final_volume,
            final_quantities,
            final_quantities - targets,
            final_jacobian,
        )
    )
    return SurfaceAreaVolumeStageDeviceResult(
        initial_parameters=completed[0],
        solver_initial_parameters=completed[1],
        initial_area=completed[2],
        initial_volume=completed[3],
        initial_residuals=completed[5],
        initial_jacobian=completed[6],
        final_parameters=completed[7],
        final_area=completed[8],
        final_volume=completed[9],
        final_residuals=completed[11],
        final_jacobian=completed[12],
        optimizer=optimizer,
    )


def solve_rz_surface_area_volume_sequence(
    *,
    full_dofs: jax.Array,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
    free_positions: jax.Array,
    first_targets: jax.Array,
    second_targets: jax.Array,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    max_steps: int,
    rtol: float,
    atol: float,
) -> SurfaceAreaVolumeSequenceDeviceResult:
    """Run both area-volume stages without dense free-DOF expansion arrays."""
    full_dofs_device = jnp.asarray(full_dofs, dtype=jnp.float64)
    free_positions_device = jnp.asarray(free_positions, dtype=jnp.int32)
    fixed_dofs, initial_parameters = _partition_surface_dofs(
        full_dofs_device,
        free_positions_device,
    )
    spec = surface_rz_fourier_spec_from_dofs(
        full_dofs_device,
        quadpoints_phi=jnp.asarray(quadpoints_phi, dtype=jnp.float64),
        quadpoints_theta=jnp.asarray(quadpoints_theta, dtype=jnp.float64),
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )
    first = _solve_stage(
        initial_parameters=initial_parameters,
        solver_initial_parameters=_first_stage_solver_seed(initial_parameters),
        targets=jnp.asarray(first_targets, dtype=jnp.float64),
        fixed_dofs=fixed_dofs,
        free_positions=free_positions_device,
        spec=spec,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    second = _solve_stage(
        initial_parameters=first.final_parameters,
        solver_initial_parameters=first.final_parameters,
        targets=jnp.asarray(second_targets, dtype=jnp.float64),
        fixed_dofs=fixed_dofs,
        free_positions=free_positions_device,
        spec=spec,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    return SurfaceAreaVolumeSequenceDeviceResult(first=first, second=second)


__all__ = [
    "SurfaceAreaVolumeSequenceDeviceResult",
    "SurfaceAreaVolumeStageDeviceResult",
    "solve_rz_surface_area_volume_sequence",
]
