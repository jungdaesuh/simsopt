"""Production flat-675 single-stage optimization workflow.

The internal workflow every flat-675 JAX caller routes through: the adapter
layer builds the physics (objective and diagnostics closures) from a frozen
input bundle and hands them here; this module owns the traceable program
construction, the frozen optimizer policy, and the solve.

The flat formulation has no nested Boozer solve, so the whole objective is one
device program and the solve is the fused on-device L-BFGS-B lane end to end.
That is also why the workflow drops the source campaign's host-side rejection
and anchor protocol: those existed to keep a SciPy driver consistent across
proposals whose inner solve could fail on the host, and no such boundary
survives here.

The L-BFGS history is the frozen selection of the genuine-675 campaign whose
certificate this workflow reproduces (``maxcor`` of its published L-BFGS-B
policy).  It is deliberately not a configuration knob —
``solve_single_stage_flat675`` accepts no history argument.
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

# Frozen campaign selection, single source of truth for both lanes: the
# genuine-675 native twin reads it so the matched workflows share one history.
# Not a solve parameter.
FLAT675_LBFGS_HISTORY = 300


@dataclass(frozen=True)
class PreparedSingleStageFlat675:
    """Prepared device programs with stable identity across repeated solves.

    The record freezes its references only; solving mutates ``problem.x``.
    Reusing one prepared record across repeated solves reuses the compiled
    fused executable — constructing a fresh record per solve retraces.
    """

    problem: TraceableParametricScalarProblem
    diagnostics: TraceableArrayFunction
    initial_parameters: jax.Array


def prepare_single_stage_flat675(
    *,
    objective_fn: Callable[[jax.Array], jax.Array],
    diagnostics_fn: Callable[[jax.Array], jax.Array],
    initial_parameters: jax.Array,
    objective_scale: jax.Array,
) -> PreparedSingleStageFlat675:
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
    return PreparedSingleStageFlat675(
        problem=problem,
        diagnostics=TraceableArrayFunction(diagnostics_fn, initial_parameters),
        initial_parameters=initial_parameters,
    )


def solve_single_stage_flat675(
    prepared: PreparedSingleStageFlat675,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
) -> OptimizerResult:
    """Solve from the prepared initial state under the frozen policy.

    Every call restarts from ``initial_parameters``, so repeated solves are
    the identical computation (the warm-measurement contract).  Callbacks stay
    disabled and no host observation happens inside the solve: the fused
    L-BFGS path runs on device end to end, which is why this calls
    ``dispatch.minimize`` directly rather than ``serial_solve_jax`` (whose
    bounded-objective log materializes host arrays on every solve).
    """
    if driver == Driver.SIMSOPT_LBFGSB:
        options: SimsoptLBFGSBOptions | SimsoptBFGSOptions = SimsoptLBFGSBOptions(
            maxiter=max_steps,
            maxfun=max_steps * 20,
            gtol=atol,
            ftol=rtol,
            maxcor=FLAT675_LBFGS_HISTORY,
        )
    else:
        options = SimsoptBFGSOptions(
            maxiter=max_steps,
            gtol=atol,
            xrtol=rtol,
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
