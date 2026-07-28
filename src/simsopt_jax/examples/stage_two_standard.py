"""Device-resident two-stage filamentary coil optimization workflow."""

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
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax


@dataclass(frozen=True)
class StandardStageTwoState:
    """Objective, derivative, and physical diagnostics at one accepted state."""

    parameters: jax.Array
    objective: jax.Array
    objective_gradient: jax.Array
    squared_flux: jax.Array
    geometric_penalty: jax.Array
    maximum_normal_field: jax.Array
    total_curve_length: jax.Array


jax.tree_util.register_dataclass(
    StandardStageTwoState,
    data_fields=[
        "parameters",
        "objective",
        "objective_gradient",
        "squared_flux",
        "geometric_penalty",
        "maximum_normal_field",
        "total_curve_length",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class StandardStageTwoDeviceResult:
    """Initial/two-stage states, Taylor evidence, and bounded solver metadata."""

    initial: StandardStageTwoState
    first: StandardStageTwoState
    final: StandardStageTwoState
    taylor_errors: jax.Array
    first_optimizer: OptimizerResult
    second_optimizer: OptimizerResult


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


def _state(
    parameters: jax.Array,
    extraction: CoilSetDofExtractionSpec,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
) -> StandardStageTwoState:
    (
        (
            objective,
            (
                squared_flux,
                geometric_penalty,
                maximum_normal_field,
                total_curve_length,
            ),
        ),
        gradient,
    ) = _value_and_grad_program(
        parameters,
        extraction,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
    )
    return StandardStageTwoState(
        parameters=parameters,
        objective=objective,
        objective_gradient=gradient,
        squared_flux=squared_flux,
        geometric_penalty=geometric_penalty,
        maximum_normal_field=maximum_normal_field,
        total_curve_length=total_curve_length,
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
    )(epsilons)
    return central_differences - directional_derivative


_taylor_errors_program = jax.jit(
    _taylor_errors_from_operands,
    static_argnums=(6,),
)


def solve_standard_stage_two(
    *,
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: object,
    surface_normal: object,
    initial_parameters: object,
    taylor_direction: object,
    first_config: StageTwoObjectiveConfig,
    second_config: StageTwoObjectiveConfig,
    max_steps: int,
    rtol: float,
    atol: float,
) -> StandardStageTwoDeviceResult:
    """Run both source-equivalent stages while retaining fixed-size endpoints."""
    extraction = field.coil_dof_extraction_spec()
    surface_gamma_device = jnp.asarray(surface_gamma, dtype=jnp.float64)
    surface_normal_device = jnp.asarray(surface_normal, dtype=jnp.float64)
    initial_device = jnp.asarray(initial_parameters, dtype=jnp.float64)
    direction_device = jnp.asarray(taylor_direction, dtype=jnp.float64)

    initial = _state(
        initial_device,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        first_config,
    )
    taylor_errors = _taylor_errors_program(
        initial_device,
        direction_device,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        first_config,
    )
    first_problem = TraceableScalarProblem(
        objective_fn=make_fused_stage_two_objective(
            field,
            flux_spec,
            surface_gamma_device,
            surface_normal_device,
            first_config,
        ),
        x=initial_device,
    )
    first_optimizer = serial_solve_jax(
        first_problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=int(max_steps),
        maxcor=min(int(max_steps), 300),
        rtol=float(rtol),
        atol=float(atol),
        require_success=False,
    )
    first = _state(
        first_problem.x,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        first_config,
    )

    second_problem = TraceableScalarProblem(
        objective_fn=make_fused_stage_two_objective(
            field,
            flux_spec,
            surface_gamma_device,
            surface_normal_device,
            second_config,
        ),
        x=first_problem.x,
    )
    second_optimizer = serial_solve_jax(
        second_problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=int(max_steps),
        maxcor=min(int(max_steps), 300),
        rtol=float(rtol),
        atol=float(atol),
        require_success=False,
    )
    final = _state(
        second_problem.x,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        second_config,
    )
    initial, first, final, taylor_errors = jax.block_until_ready(
        (initial, first, final, taylor_errors)
    )
    return StandardStageTwoDeviceResult(
        initial=initial,
        first=first,
        final=final,
        taylor_errors=taylor_errors,
        first_optimizer=first_optimizer,
        second_optimizer=second_optimizer,
    )


__all__ = [
    "StandardStageTwoDeviceResult",
    "StandardStageTwoState",
    "solve_standard_stage_two",
]
