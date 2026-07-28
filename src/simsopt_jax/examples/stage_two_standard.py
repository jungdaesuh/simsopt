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
)
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    serial_solve_jax,
)


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
    length_weight: jax.Array,
) -> jax.Array:
    values = fused_stage_two_values(
        extraction,
        parameters,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
    )
    return values[0] + length_weight * values[4]


def _objective_with_aux_from_operands(
    parameters: jax.Array,
    extraction: CoilSetDofExtractionSpec,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    config: StageTwoObjectiveConfig,
    length_weight: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    values = fused_stage_two_values(
        extraction,
        parameters,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
    )
    length_penalty = length_weight * values[4]
    return (
        values[0] + length_penalty,
        (
            values[1],
            values[2] + length_penalty,
            values[3],
            values[4],
        ),
    )


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
    length_weight: jax.Array,
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
        length_weight,
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
    length_weight: jax.Array,
) -> jax.Array:
    initial_gradient = jax.grad(_objective_from_operands, argnums=0)(
        initial_parameters,
        extraction,
        flux_spec,
        surface_gamma,
        surface_normal,
        config,
        length_weight,
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
                    length_weight,
                )
                - _objective_from_operands(
                    initial_parameters - epsilon * direction,
                    extraction,
                    flux_spec,
                    surface_gamma,
                    surface_normal,
                    config,
                    length_weight,
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


def solve_standard_stage_two(
    *,
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: object,
    surface_normal: object,
    initial_parameters: object,
    taylor_direction: object,
    regularization_config: StageTwoObjectiveConfig,
    first_length_weight: jax.Array,
    second_length_weight: jax.Array,
    max_steps: int,
    rtol: float,
    atol: float,
) -> StandardStageTwoDeviceResult:
    """Run both source-equivalent stages while retaining fixed-size endpoints."""
    if (
        regularization_config.length_weight != 0.0
        or regularization_config.length_target is not None
    ):
        raise ValueError(
            "Standard Stage-II length weights are explicit device parameters; "
            "config must not include a total-length penalty."
        )
    extraction = field.coil_dof_extraction_spec()
    surface_gamma_device = jnp.asarray(surface_gamma, dtype=jnp.float64)
    surface_normal_device = jnp.asarray(surface_normal, dtype=jnp.float64)
    initial_device = jnp.asarray(initial_parameters, dtype=jnp.float64)
    direction_device = jnp.asarray(taylor_direction, dtype=jnp.float64)
    first_length_weight_device = jnp.asarray(
        first_length_weight,
        dtype=initial_device.dtype,
    )
    second_length_weight_device = jnp.asarray(
        second_length_weight,
        dtype=initial_device.dtype,
    )

    initial = _state(
        initial_device,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        regularization_config,
        first_length_weight_device,
    )
    taylor_errors = _taylor_errors_program(
        initial_device,
        direction_device,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        regularization_config,
        first_length_weight_device,
    )
    problem = TraceableParametricScalarProblem(
        objective_fn=lambda parameters, length_weight: _objective_from_operands(
            parameters,
            extraction,
            flux_spec,
            surface_gamma_device,
            surface_normal_device,
            regularization_config,
            length_weight,
        ),
        objective_parameter=first_length_weight_device,
        x=initial_device,
    )
    first_optimizer = serial_solve_jax(
        problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=int(max_steps),
        maxcor=min(int(max_steps), 300),
        rtol=float(rtol),
        atol=float(atol),
        require_success=False,
    )
    first = _state(
        problem.x,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        regularization_config,
        first_length_weight_device,
    )

    problem.set_objective_parameter(second_length_weight_device)
    second_optimizer = serial_solve_jax(
        problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=int(max_steps),
        maxcor=min(int(max_steps), 300),
        rtol=float(rtol),
        atol=float(atol),
        require_success=False,
    )
    final = _state(
        problem.x,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        regularization_config,
        second_length_weight_device,
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
