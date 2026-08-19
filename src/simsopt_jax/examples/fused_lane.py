"""The fused on-device solve shared by the certified JAX campaign lanes.

One traceable scaled problem, one frozen L-BFGS-B policy, and a solve whose
only host boundary crossing is its endpoint.  A campaign module supplies the
physics closures and names its own frozen L-BFGS history; nothing else about
the lane differs between campaigns, so nothing else is restated per campaign.

``lbfgs_history`` is a required keyword with no default on purpose: the
history is a measured per-campaign selection, and a default here would be a
value no campaign chose.
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
class PreparedFusedLaneSolve:
    """Prepared device programs with stable identity across repeated solves.

    The record freezes its references only; solving mutates ``problem.x``.
    Reusing one prepared record across repeated solves reuses the compiled
    fused executable — constructing a fresh record per solve retraces.
    """

    problem: TraceableParametricScalarProblem
    diagnostics: TraceableArrayFunction
    initial_parameters: jax.Array


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

    def scaled_objective(
        parameters: jax.Array,
        scale_parameter: jax.Array,
    ) -> jax.Array:
        return scale_parameter * objective_fn(parameters)

    problem = TraceableParametricScalarProblem(
        objective_fn=scaled_objective,
        objective_parameter=objective_scale,
        x=initial_parameters,
    )
    return PreparedFusedLaneSolve(
        problem=problem,
        diagnostics=TraceableArrayFunction(diagnostics_fn, initial_parameters),
        initial_parameters=initial_parameters,
    )


def solve_fused_lane(
    prepared: PreparedFusedLaneSolve,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
    lbfgs_history: int,
    line_search_max_steps: int | None = None,
) -> OptimizerResult:
    """Solve from the prepared initial state under the campaign's policy.

    Every call restarts from ``initial_parameters``, so repeated solves are
    the identical computation (the warm-measurement contract).  Callbacks stay
    disabled and no host observation happens inside the solve: the fused
    L-BFGS path runs on device end to end, which is why this calls
    ``dispatch.minimize`` directly rather than ``serial_solve_jax`` (whose
    bounded-objective log materializes host arrays on every solve).
    """
    if driver == Driver.SIMSOPT_LBFGSB:
        if line_search_max_steps is not None:
            raise TypeError(
                "line_search_max_steps is a SIMSOPT_BFGS option; the L-BFGS-B "
                "line search is not configurable here"
            )
        options: SimsoptLBFGSBOptions | SimsoptBFGSOptions = SimsoptLBFGSBOptions(
            maxiter=max_steps,
            maxfun=max_steps * 20,
            gtol=atol,
            ftol=rtol,
            maxcor=lbfgs_history,
        )
    else:
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
    "PreparedFusedLaneSolve",
    "prepare_fused_lane_solve",
    "solve_fused_lane",
]
