"""Minimal Stage-II coil optimization workflow.

The physics is device-resident throughout. Two outer optimizers are
available and every caller names one:

``MINIMAL_STAGE_TWO_OFFICIAL_DRIVER``
    ``scipy.optimize.minimize(..., method="L-BFGS-B")`` over the JAX
    objective, which is the provider upstream's
    ``examples/1_Simple/stage_two_optimization_minimal.py`` calls. This is the
    **mirror default**: it is what the shipped example runs, through the same host SciPy route four sibling mirrors use.
``MINIMAL_STAGE_TWO_DEVICE_DRIVER``
    the in-tree device-resident L-BFGS-B, an explicit opt-in **performance
    mode** for library callers. It keeps the whole solve
    on the device and keeps its own strict-transfer and policy tests; it is
    not the mirror.

This module owns the official example's optimizer policy: the 300-iteration
budget, L-BFGS-B history 300, the single 1e-15 tolerance SciPy expands into
``ftol`` and ``gtol``, and SciPy's own evaluation and line-search limits,
which upstream does not override.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import jax
import jax.numpy as jnp

from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.specs import CoilSetDofExtractionSpec, FixedSurfaceFluxSpec
from simsopt_jax.examples.fused_lane import (
    prepare_fused_lane_objective,
    solve_fused_lane,
)
from simsopt_jax.examples.scalar_stage import solve_scalar_stage
from simsopt_jax.objectives import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    fused_stage_two_values,
    make_fused_stage_two_objective,
)
from simsopt_jax.runtime.host_boundary import block_until_ready
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions

# Official examples/1_Simple/stage_two_optimization_minimal.py policy.
MINIMAL_STAGE_TWO_LBFGS_HISTORY: Final[int] = 300
MINIMAL_STAGE_TWO_NATIVE_ITERATIONS: Final[int] = 300
MINIMAL_STAGE_TWO_TOLERANCE: Final[float] = 1.0e-15
# Upstream names maxiter, maxcor and tol and leaves the evaluation budget and
# the line search at SciPy's own defaults, so those numbers have one owner:
# SciPy's option contract.
MINIMAL_STAGE_TWO_MAX_FUNCTION_EVALUATIONS: Final[int] = ScipyLBFGSBOptions.maxfun
MINIMAL_STAGE_TWO_MAX_LINE_SEARCH_STEPS: Final[int] = ScipyLBFGSBOptions.maxls

#: The provider the official script calls, driven over the JAX objective.
MINIMAL_STAGE_TWO_OFFICIAL_DRIVER: Final[Driver] = Driver.SCIPY_LBFGSB
#: The device-resident reimplementation, selected explicitly for speed.
MINIMAL_STAGE_TWO_DEVICE_DRIVER: Final[Driver] = Driver.SIMSOPT_LBFGSB


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


@dataclass(frozen=True)
class _MinimalStageTwoOperands:
    """Everything the lane's traceable programs are parameterized by."""

    extraction: CoilSetDofExtractionSpec
    flux_spec: FixedSurfaceFluxSpec
    surface_gamma: jax.Array
    surface_normal: jax.Array
    config: StageTwoObjectiveConfig


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


def _minimal_stage_two_operands(
    *,
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    num_base_curves: int,
    length_weight: float,
    length_target: float,
) -> _MinimalStageTwoOperands:
    """Freeze the extraction spec, device geometry and objective config once."""
    return _MinimalStageTwoOperands(
        extraction=field.coil_dof_extraction_spec(),
        flux_spec=flux_spec,
        surface_gamma=jnp.asarray(surface_gamma, dtype=jnp.float64),
        surface_normal=jnp.asarray(surface_normal, dtype=jnp.float64),
        config=StageTwoObjectiveConfig(
            num_base_curves=num_base_curves,
            length_weight=length_weight,
            length_target=length_target,
            length_target_mode="max",
        ),
    )


def _state_program(
    operands: _MinimalStageTwoOperands,
) -> Callable[[jax.Array], MinimalStageTwoState]:
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
            operands.extraction,
            operands.flux_spec,
            operands.surface_gamma,
            operands.surface_normal,
            operands.config,
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

    return state


def minimal_stage_two_state(
    *,
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: jax.Array,
    surface_normal: jax.Array,
    num_base_curves: int,
    length_weight: float,
    length_target: float,
) -> Callable[[jax.Array], MinimalStageTwoState]:
    """The device state program this lane publishes, for any parameter vector.

    :func:`solve_minimal_stage_two` publishes its initial and final states
    through exactly this program, so a caller that evaluates the lane at a
    state of its own choosing -- an official replay state, for instance --
    evaluates the shipped objective rather than a second spelling of it.
    """
    return _state_program(
        _minimal_stage_two_operands(
            field=field,
            flux_spec=flux_spec,
            surface_gamma=surface_gamma,
            surface_normal=surface_normal,
            num_base_curves=num_base_curves,
            length_weight=length_weight,
            length_target=length_target,
        )
    )


def _solve_minimal_stage_two_optimizer(
    objective: Callable[[jax.Array], jax.Array],
    initial_parameters: jax.Array,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
) -> tuple[OptimizerResult, jax.Array]:
    """Run the official L-BFGS-B policy under the caller's outer optimizer."""
    # No diagnostics program: this lane publishes its diagnostics through its
    # own `state(...)` value-and-grad closure, so preparing one here would
    # closure-convert and trace the full stage-two objective for a program
    # nothing calls.
    prepared = prepare_fused_lane_objective(
        objective_fn=objective,
        initial_parameters=initial_parameters,
        objective_scale=explicit_device_array(
            1.0,
            dtype=initial_parameters.dtype,
            reference=initial_parameters,
        ),
    )
    if driver == MINIMAL_STAGE_TWO_OFFICIAL_DRIVER:
        if rtol != atol:
            raise ValueError(
                "the official provider is scipy.optimize.minimize(..., tol=...), "
                "which sets ftol and gtol together, so rtol and atol must be equal"
            )
        prepared.problem.x = prepared.initial_parameters
        optimizer = solve_scalar_stage(
            prepared.problem,
            driver=driver,
            max_steps=max_steps,
            maxcor=MINIMAL_STAGE_TWO_LBFGS_HISTORY,
            tol=rtol,
        )
    else:
        optimizer = solve_fused_lane(
            prepared,
            driver=driver,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            lbfgs_history=MINIMAL_STAGE_TWO_LBFGS_HISTORY,
            max_function_evaluations=MINIMAL_STAGE_TWO_MAX_FUNCTION_EVALUATIONS,
            lbfgs_line_search_max_steps=MINIMAL_STAGE_TWO_MAX_LINE_SEARCH_STEPS,
        )
    return optimizer, prepared.problem.x


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
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
) -> MinimalStageTwoDeviceResult:
    """Optimize the source-equivalent flux-plus-length objective on one device.

    ``driver`` has no default on purpose: which optimizer ran is the
    difference between the mirror and the performance mode, so every caller
    states it.
    """
    operands = _minimal_stage_two_operands(
        field=field,
        flux_spec=flux_spec,
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        num_base_curves=num_base_curves,
        length_weight=length_weight,
        length_target=length_target,
    )
    initial_device = jnp.asarray(initial_parameters, dtype=jnp.float64)
    direction_device = jnp.asarray(taylor_direction, dtype=jnp.float64)
    objective = make_fused_stage_two_objective(
        field,
        operands.flux_spec,
        operands.surface_gamma,
        operands.surface_normal,
        operands.config,
    )
    state = _state_program(operands)

    initial = state(initial_device)
    taylor_errors = _taylor_errors_program(
        initial_device,
        direction_device,
        operands.extraction,
        operands.flux_spec,
        operands.surface_gamma,
        operands.surface_normal,
        operands.config,
    )

    optimizer, final_parameters = _solve_minimal_stage_two_optimizer(
        objective,
        initial_device,
        driver=driver,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
    )
    final = state(final_parameters)
    completed = block_until_ready(
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
    "MINIMAL_STAGE_TWO_DEVICE_DRIVER",
    "MINIMAL_STAGE_TWO_LBFGS_HISTORY",
    "MINIMAL_STAGE_TWO_MAX_FUNCTION_EVALUATIONS",
    "MINIMAL_STAGE_TWO_MAX_LINE_SEARCH_STEPS",
    "MINIMAL_STAGE_TWO_NATIVE_ITERATIONS",
    "MINIMAL_STAGE_TWO_OFFICIAL_DRIVER",
    "MINIMAL_STAGE_TWO_TOLERANCE",
    "MinimalStageTwoDeviceResult",
    "MinimalStageTwoState",
    "minimal_stage_two_state",
    "solve_minimal_stage_two",
]
