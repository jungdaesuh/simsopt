"""The fused on-device solve shared by the certified JAX lanes.

One traceable scaled problem, one frozen L-BFGS-B policy, and a solve whose
only host boundary crossing is its endpoint.  A caller module supplies the
physics closures and names its own frozen L-BFGS history; nothing else about
the lane differs between callers, so nothing else is restated per caller.

``lbfgs_history`` is a required keyword with no default on purpose: the
history is a measured per-workflow selection, and a default here would be a
value no caller chose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax

from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableArrayFunction,
    TraceableParametricScalarProblem,
)
from simsopt_jax.solve.simsopt.contracts import (
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
)


@dataclass(frozen=True)
class PreparedFusedLaneObjective:
    """Prepared solve programs with stable identity across repeated solves.

    The record freezes its references only; solving mutates ``problem.x``.
    Reusing one prepared record across repeated solves reuses the compiled
    fused executable — constructing a fresh record per solve retraces.

    This is what :func:`solve_fused_lane` needs. A lane that also publishes
    on-device diagnostics prepares :class:`PreparedFusedLaneSolve` instead; a
    lane that does not must not build a diagnostics program it never calls,
    because ``TraceableArrayFunction`` closure-converts and traces its function
    eagerly on construction.
    """

    problem: TraceableParametricScalarProblem
    initial_parameters: jax.Array


@dataclass(frozen=True)
class PreparedFusedLaneSolve(PreparedFusedLaneObjective):
    """A prepared lane that also publishes an on-device diagnostics program."""

    diagnostics: TraceableArrayFunction


def _prepared_problem(
    objective_fn: Callable[[jax.Array], jax.Array],
    initial_parameters: jax.Array,
    objective_scale: jax.Array,
) -> TraceableParametricScalarProblem:
    def scaled_objective(
        parameters: jax.Array,
        scale_parameter: jax.Array,
    ) -> jax.Array:
        return scale_parameter * objective_fn(parameters)

    return TraceableParametricScalarProblem(
        objective_fn=scaled_objective,
        objective_parameter=objective_scale,
        x=initial_parameters,
    )


def prepare_fused_lane_objective(
    *,
    objective_fn: Callable[[jax.Array], jax.Array],
    initial_parameters: jax.Array,
    objective_scale: jax.Array,
) -> PreparedFusedLaneObjective:
    """Build only the traceable scaled problem, for lanes with no diagnostics.

    ``objective_scale`` is the parametric solve scale; callers republish at a
    different scale through ``problem.set_objective_parameter`` without
    retracing.
    """
    return PreparedFusedLaneObjective(
        problem=_prepared_problem(objective_fn, initial_parameters, objective_scale),
        initial_parameters=initial_parameters,
    )


def prepare_fused_lane_solve(
    *,
    objective_fn: Callable[[jax.Array], jax.Array],
    diagnostics_fn: Callable[[jax.Array], jax.Array],
    initial_parameters: jax.Array,
    objective_scale: jax.Array,
) -> PreparedFusedLaneSolve:
    """Build the traceable scaled problem and diagnostics programs once.

    ``objective_scale`` is the parametric solve scale; callers republish at a
    different scale through ``problem.set_objective_parameter`` without
    retracing.
    """
    return PreparedFusedLaneSolve(
        problem=_prepared_problem(objective_fn, initial_parameters, objective_scale),
        initial_parameters=initial_parameters,
        diagnostics=TraceableArrayFunction(diagnostics_fn, initial_parameters),
    )


def solve_fused_lane(
    prepared: PreparedFusedLaneObjective,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
    lbfgs_history: int,
    max_function_evaluations: int | None = None,
    lbfgs_line_search_max_steps: int | None = None,
    line_search_max_steps: int | None = None,
) -> OptimizerResult:
    """Solve from the prepared initial state under the caller's policy.

    Every call restarts from ``initial_parameters``, so repeated solves are
    the identical computation (the warm-measurement contract).  Callbacks stay
    disabled and no host observation happens inside the solve: the fused
    L-BFGS path runs on device end to end, which is why this calls
    ``dispatch.minimize`` directly rather than ``serial_solve_jax`` (whose
    bounded-objective log materializes host arrays on every solve).

    The two line-search arguments belong to different drivers and are guarded
    against each other: ``lbfgs_line_search_max_steps`` is L-BFGS-B's
    ``maxls``, ``line_search_max_steps`` is the BFGS zoom's step cap.  Either
    left ``None`` takes its optimizer's own default, so a caller that names
    neither gets exactly the behavior it got before the knobs existed.

    ``max_function_evaluations`` supplies a workflow's external evaluation
    budget. Omitting it preserves the existing twenty-evaluations-per-step cap.
    """
    if driver == Driver.SIMSOPT_LBFGSB:
        if line_search_max_steps is not None:
            raise TypeError(
                "line_search_max_steps is a SIMSOPT_BFGS option; the L-BFGS-B "
                "line search is configured through lbfgs_line_search_max_steps"
            )
        options: SimsoptLBFGSBOptions | SimsoptBFGSOptions = SimsoptLBFGSBOptions(
            maxiter=max_steps,
            maxfun=(
                max_steps * 20
                if max_function_evaluations is None
                else max_function_evaluations
            ),
            gtol=atol,
            ftol=rtol,
            maxcor=lbfgs_history,
            maxls=(
                SimsoptLBFGSBOptions().maxls
                if lbfgs_line_search_max_steps is None
                else lbfgs_line_search_max_steps
            ),
        )
    else:
        if lbfgs_line_search_max_steps is not None:
            raise TypeError(
                "lbfgs_line_search_max_steps is a SIMSOPT_LBFGSB option; the "
                "BFGS line search is configured through line_search_max_steps"
            )
        options = SimsoptBFGSOptions(
            maxiter=max_steps,
            gtol=atol,
            xrtol=rtol,
            line_search_max_steps=(
                SimsoptBFGSOptions().line_search_max_steps
                if line_search_max_steps is None
                else line_search_max_steps
            ),
        )
    initial = prepared.initial_parameters
    prepared.problem.x = initial
    result = minimize(
        prepared.problem._solver_value_and_grad_fn,
        initial,
        driver=driver,
        options=options,
    )
    prepared.problem.x = explicit_device_array(
        result.x,
        dtype=result.x.dtype,
        reference=initial,
    )
    return result


__all__ = [
    "PreparedFusedLaneObjective",
    "PreparedFusedLaneSolve",
    "prepare_fused_lane_objective",
    "prepare_fused_lane_solve",
    "solve_fused_lane",
]
