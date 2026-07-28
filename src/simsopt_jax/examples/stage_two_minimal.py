"""Device-resident minimal Stage-II coil optimization workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simsopt_jax.core.specs import CoilSetDofExtractionSpec, FixedSurfaceFluxSpec
from simsopt_jax.objectives import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    fused_stage_two_values,
    make_fused_stage_two_objective,
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax


@dataclass(frozen=True)
class MinimalStageTwoState:
    """Stage-II observables kept on the selected JAX device."""

    parameters: jax.Array
    objective: jax.Array
    objective_gradient: jax.Array
    squared_flux: jax.Array
    length_penalty: jax.Array
    maximum_normal_field: jax.Array
    total_curve_length: jax.Array


jax.tree_util.register_dataclass(
    MinimalStageTwoState,
    data_fields=[
        "parameters",
        "objective",
        "objective_gradient",
        "squared_flux",
        "length_penalty",
        "maximum_normal_field",
        "total_curve_length",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class MinimalStageTwoDeviceResult:
    """Initial/final states and Taylor-test evidence for one completed solve."""

    initial: MinimalStageTwoState
    final: MinimalStageTwoState
    taylor_errors: jax.Array
    optimizer: OptimizerResult


def _objective_from_operands(
    parameters: jax.Array,
    extraction: CoilSetDofExtractionSpec,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> jax.Array:
    return fused_stage_two_values(
        extraction,
        parameters,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
    )[0]


def _objective_with_aux_from_operands(
    parameters: jax.Array,
    extraction: CoilSetDofExtractionSpec,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    values = fused_stage_two_values(
        extraction,
        parameters,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
    )
    return values[0], values[1:]


_value_and_grad_program = jax.jit(
    jax.value_and_grad(
        _objective_with_aux_from_operands,
        argnums=0,
        has_aux=True,
    ),
    static_argnums=(5,),
)


def _taylor_errors_from_operands(
    initial_parameters: jax.Array,
    direction: jax.Array,
    extraction: CoilSetDofExtractionSpec,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> jax.Array:
    initial_gradient = jax.grad(_objective_from_operands, argnums=0)(
        initial_parameters,
        extraction,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
    )
    directional_derivative = jnp.vdot(initial_gradient, direction).real
    epsilons = jnp.asarray(
        (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7),
        dtype=initial_parameters.dtype,
    )
    central_differences = jax.vmap(
        lambda epsilon: (
            (
                _objective_from_operands(
                    initial_parameters + epsilon * direction,
                    extraction,
                    flux_spec,
                    surface_gamma,
                    surface_normal,
                    config,
                )
                - _objective_from_operands(
                    initial_parameters - epsilon * direction,
                    extraction,
                    flux_spec,
                    surface_gamma,
                    surface_normal,
                    config,
                )
            )
            / (2.0 * epsilon)
        )
    )(epsilons)
    return central_differences - directional_derivative


_taylor_errors_program = jax.jit(
    _taylor_errors_from_operands,
    static_argnums=(6,),
)


def solve_minimal_stage_two(
    *,
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    initial_parameters: jax.Array,
    taylor_direction: jax.Array,
    num_base_curves: int,
    length_weight: float,
    length_target: float,
    max_steps: int,
    rtol: float,
    atol: float,
) -> MinimalStageTwoDeviceResult:
    """Optimize the source-equivalent flux-plus-length objective on one device."""
    config = StageTwoObjectiveConfig(
        num_base_curves=num_base_curves,
        length_weight=length_weight,
        length_target=length_target,
        length_target_mode="max",
    )
    extraction = field.coil_dof_extraction_spec()
    surface_gamma_device = jnp.asarray(surface_gamma, dtype=jnp.float64)
    surface_normal_device = jnp.asarray(surface_normal, dtype=jnp.float64)
    initial_device = jnp.asarray(initial_parameters, dtype=jnp.float64)
    direction_device = jnp.asarray(taylor_direction, dtype=jnp.float64)
    objective = make_fused_stage_two_objective(
        field,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        config,
    )

    def state(parameters: jax.Array) -> MinimalStageTwoState:
        (
            (
                objective_value,
                (
                    squared_flux,
                    length_penalty,
                    maximum_normal_field,
                    total_curve_length,
                ),
            ),
            objective_gradient,
        ) = _value_and_grad_program(
            parameters,
            extraction,
            flux_spec,
            surface_gamma_device,
            surface_normal_device,
            config,
        )
        return MinimalStageTwoState(
            parameters=parameters,
            objective=objective_value,
            objective_gradient=objective_gradient,
            squared_flux=squared_flux,
            length_penalty=length_penalty,
            maximum_normal_field=maximum_normal_field,
            total_curve_length=total_curve_length,
        )

    initial = state(initial_device)
    taylor_errors = _taylor_errors_program(
        initial_device,
        direction_device,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        config,
    )

    problem = TraceableScalarProblem(
        objective_fn=objective,
        x=initial_device,
    )
    optimizer = serial_solve_jax(
        problem,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    final = state(problem.x)
    completed = jax.block_until_ready(
        (
            (
                initial.parameters,
                initial.objective,
                initial.objective_gradient,
                initial.squared_flux,
                initial.length_penalty,
                initial.maximum_normal_field,
                initial.total_curve_length,
            ),
            (
                final.parameters,
                final.objective,
                final.objective_gradient,
                final.squared_flux,
                final.length_penalty,
                final.maximum_normal_field,
                final.total_curve_length,
            ),
            taylor_errors,
        )
    )

    def completed_state(values: tuple[jax.Array, ...]) -> MinimalStageTwoState:
        return MinimalStageTwoState(
            parameters=values[0],
            objective=values[1],
            objective_gradient=values[2],
            squared_flux=values[3],
            length_penalty=values[4],
            maximum_normal_field=values[5],
            total_curve_length=values[6],
        )

    return MinimalStageTwoDeviceResult(
        initial=completed_state(completed[0]),
        final=completed_state(completed[1]),
        taylor_errors=completed[2],
        optimizer=optimizer,
    )


__all__ = [
    "MinimalStageTwoDeviceResult",
    "MinimalStageTwoState",
    "solve_minimal_stage_two",
]
