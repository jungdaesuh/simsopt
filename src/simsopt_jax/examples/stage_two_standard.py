"""Device-resident two-stage filamentary coil optimization workflow."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp

from simsopt_jax.core.specs import CoilSetDofExtractionSpec, FixedSurfaceFluxSpec
from simsopt_jax.objectives.stage_two import (
    CoilDofExtractionProvider,
    StageTwoObjectiveConfig,
    fused_stage_two_values,
    stage_two_length_penalty,
)
from simsopt_jax.runtime.host_boundary import block_until_ready
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import TraceableParametricScalarProblem

from .scalar_stage import (
    TWO_STAGE_OPTIMIZER_OBSERVABLES,
    solve_scalar_stage,
    two_stage_optimizer_observables,
)

_STAGE_TWO_LBFGS_HISTORY_SIZE = 300


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
    """Initial/two-stage states, Taylor evidence, and bounded solver metadata.

    ``two_stage_minimize_seconds`` runs from entry of the first stage's minimize
    call to return of the second's, so it also contains the inter-stage state
    evaluation -- the same content as a native two-stage script's region between
    its last Taylor line and its last objective evaluation.  ``whole_call_seconds``
    adds device placement and the Taylor test.  Each stage's minimize call alone
    is ``first_optimizer.wallclock_s`` / ``second_optimizer.wallclock_s``, and
    ``execution_device`` is read off the solved endpoint array, not off the
    configured default backend.
    """

    initial: StandardStageTwoState
    first: StandardStageTwoState
    final: StandardStageTwoState
    taylor_errors: jax.Array
    first_optimizer: OptimizerResult
    second_optimizer: OptimizerResult
    two_stage_minimize_seconds: float
    whole_call_seconds: float
    execution_device: str


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
    return values[0] + stage_two_length_penalty(values[4], config, length_weight)


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
    length_penalty = stage_two_length_penalty(values[4], config, length_weight)
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


def _solve_stage(
    problem: TraceableParametricScalarProblem,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
) -> OptimizerResult:
    """One stage of this workflow, at the bounded history both drivers share.

    ``solve_standard_stage_two`` states its stopping rule as a ``rtol``/``atol``
    pair, while the rule a native two-stage script has is one ``tol`` that
    ``scipy.optimize.minimize`` expands into both.  A split pair is therefore a
    policy no native twin can express, on either driver, and is refused here
    rather than silently halved.  The route is :func:`solve_scalar_stage`.
    """
    if rtol != atol:
        raise ValueError(
            "this workflow reproduces a native script's single tol, which "
            "scipy.optimize.minimize expands into both ftol and gtol, so it "
            f"needs one tolerance; got rtol={rtol!r}, atol={atol!r}"
        )
    return solve_scalar_stage(
        problem,
        driver=driver,
        max_steps=max_steps,
        maxcor=_STAGE_TWO_LBFGS_HISTORY_SIZE,
        tol=rtol,
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
    driver: Driver = Driver.SIMSOPT_LBFGSB,
) -> StandardStageTwoDeviceResult:
    """Run both source-equivalent stages with one reusable compiled problem.

    ``driver`` selects the L-BFGS-B implementation. The default runs the
    device-resident port (one fused loop per stage, no host transfers,
    roughly a minute of compilation per process). ``Driver.SCIPY_LBFGSB``
    drives the same device objective program with SciPy's L-BFGS-B, the
    routine the native scripts use, so ``rtol``/``atol`` map onto its
    ``ftol``/``gtol`` exactly as native's ``tol`` does and the objective
    program is the only compiled artifact (sub-second cold compilation); it
    performs one host round trip per objective evaluation.  That single
    native ``tol`` is the whole stopping rule on either driver, so ``rtol``
    and ``atol`` must be equal; a split pair is refused, not halved.

    Length weights remain explicit fixed-shape device parameters, while the
    returned states have fixed-size storage. L-BFGS history is bounded
    independently of the iteration budget so compiled state cannot scale with
    example runtime. The returned result times the two-stage minimize region
    separately from the whole call and attests the device the endpoint arrays
    were produced on.
    """
    call_started = perf_counter()
    if regularization_config.length_weight != 0.0:
        raise ValueError(
            "Standard Stage-II length weights are explicit device parameters; "
            "config length_weight must be zero."
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
    minimize_region_started = perf_counter()
    first_optimizer = _solve_stage(problem, driver, int(max_steps), float(rtol), float(atol))
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
    second_optimizer = _solve_stage(problem, driver, int(max_steps), float(rtol), float(atol))
    two_stage_minimize_seconds = perf_counter() - minimize_region_started
    final = _state(
        problem.x,
        extraction,
        flux_spec,
        surface_gamma_device,
        surface_normal_device,
        regularization_config,
        second_length_weight_device,
    )
    # Read before the host boundary: the endpoint objective is an output of the
    # compiled program, so its device is where the optimization actually ran.
    execution_device = str(final.objective.device)
    initial, first, final, taylor_errors = block_until_ready(
        (initial, first, final, taylor_errors)
    )
    return StandardStageTwoDeviceResult(
        initial=initial,
        first=first,
        final=final,
        taylor_errors=taylor_errors,
        first_optimizer=first_optimizer,
        second_optimizer=second_optimizer,
        two_stage_minimize_seconds=two_stage_minimize_seconds,
        whole_call_seconds=perf_counter() - call_started,
        execution_device=execution_device,
    )


def standard_stage_two_state(
    *,
    field: CoilDofExtractionProvider,
    flux_spec: FixedSurfaceFluxSpec,
    surface_gamma: object,
    surface_normal: object,
    parameters: object,
    regularization_config: StageTwoObjectiveConfig,
    length_weight: object,
) -> StandardStageTwoState:
    """Evaluate one state of the same objective :func:`solve_standard_stage_two` solves.

    This is the entry point a cross-lane quality check needs: it evaluates a
    given endpoint -- this lane's or the other lane's -- through the mirror's own
    program, so the check compares evaluators instead of comparing a rewritten
    copy of one.  ``regularization_config`` and ``length_weight`` must be the
    pair the stage under comparison ran with, since the length weight is an
    explicit parameter and not part of the config.
    """
    return block_until_ready(
        _state(
            jnp.asarray(parameters, dtype=jnp.float64),
            field.coil_dof_extraction_spec(),
            flux_spec,
            jnp.asarray(surface_gamma, dtype=jnp.float64),
            jnp.asarray(surface_normal, dtype=jnp.float64),
            regularization_config,
            jnp.asarray(length_weight, dtype=jnp.float64),
        )
    )


#: This family's spelling of the shared two-stage name tuple, kept importable
#: under the name the drivers and sibling mirrors already read.
STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES = TWO_STAGE_OPTIMIZER_OBSERVABLES


def standard_stage_two_optimizer_observables(
    result: StandardStageTwoDeviceResult,
) -> dict[str, object]:
    """This workflow's result, published under the shared two-stage names.

    The device result carries the minimize-region and whole-call clocks and the
    endpoint's device attestation beside the two optimizer results, so the
    reading of every field is :func:`two_stage_optimizer_observables`'.
    """
    return two_stage_optimizer_observables(
        result.first_optimizer,
        result.second_optimizer,
        execution_device=result.execution_device,
        two_stage_minimize_seconds=result.two_stage_minimize_seconds,
        whole_call_seconds=result.whole_call_seconds,
    )


__all__ = [
    "STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES",
    "StandardStageTwoDeviceResult",
    "StandardStageTwoState",
    "solve_standard_stage_two",
    "standard_stage_two_optimizer_observables",
    "standard_stage_two_state",
]
