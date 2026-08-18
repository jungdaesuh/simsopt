"""Production finite-build Stage-II coil optimization workflow.

The internal workflow both finite-build JAX callers route through: the
Advanced example and the parity case build their physics (objective and
diagnostics closures) from the adapter layer and hand them here; this module
owns the traceable program construction, the frozen optimizer policy, and
the solve.

The L-BFGS history is the frozen selection of the successor speed campaign
(``docs/jax_gpu_finitebuild_native_speed_successor_plan.md``): history 10
was the only history whose crossing endpoint passed the frozen endpoint
contract, selected over histories 20 and 40 on qualifying warm solve time.
It is deliberately not a configuration knob — ``solve_finite_build_stage_two``
accepts no history argument.
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
# parity case's native twin reads it so the matched workflows share one
# history.  Not a solve parameter.
FINITE_BUILD_LBFGS_HISTORY = 10


@dataclass(frozen=True)
class PreparedFiniteBuildStageTwo:
    """Prepared device programs with stable identity across repeated solves.

    The record freezes its references only; solving mutates ``problem.x``.
    Reusing one prepared record across repeated solves reuses the compiled
    fused executable — constructing a fresh record per solve retraces.
    """

    problem: TraceableParametricScalarProblem
    diagnostics: TraceableArrayFunction
    initial_parameters: jax.Array


def prepare_finite_build_stage_two(
    *,
    objective_fn: Callable[[jax.Array], jax.Array],
    diagnostics_fn: Callable[[jax.Array], jax.Array],
    initial_parameters: jax.Array,
    objective_scale: jax.Array,
) -> PreparedFiniteBuildStageTwo:
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
    return PreparedFiniteBuildStageTwo(
        problem=problem,
        diagnostics=TraceableArrayFunction(diagnostics_fn, initial_parameters),
        initial_parameters=initial_parameters,
    )


def solve_finite_build_stage_two(
    prepared: PreparedFiniteBuildStageTwo,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
    line_search_max_steps: int | None = None,
) -> OptimizerResult:
    """Solve from the prepared initial state under the frozen policy.

    Every call restarts from ``initial_parameters``, so repeated solves are
    the identical computation (the warm-measurement contract).  Callbacks
    stay disabled and no host observation happens inside the solve: the
    fused L-BFGS path runs on device end to end, which is why this calls
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
            maxcor=FINITE_BUILD_LBFGS_HISTORY,
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
