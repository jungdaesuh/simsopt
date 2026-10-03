"""Production finite-build Stage-II coil optimization workflow.

The internal workflow the finite-build JAX callers route through: the
Advanced example builds its physics (objective and
diagnostics closures) from the adapter layer and hands them here; the fused
lane in :mod:`simsopt_jax.examples.fused_lane` owns the traceable program
construction and the device solve, and
:func:`simsopt_jax.examples.scalar_stage.solve_scalar_stage` owns the host
SciPy route both share with four sibling mirrors.

This module owns the official example's optimizer policy -- the outer
optimizer, the 400-iteration budget, L-BFGS-B history 400, function and
gradient tolerances 1e-20, and SciPy's own evaluation and line-search limits,
which upstream does not override.

Two outer optimizers are available and every caller names one:

``FINITE_BUILD_OFFICIAL_DRIVER``
    ``scipy.optimize.minimize(..., method="L-BFGS-B")`` over the JAX
    objective, which is the provider upstream's
    ``examples/3_Advanced/stage_two_optimization_finitebuild.py`` calls. This
    is the **mirror default**: it is what the shipped example and the parity
    case run.
``FINITE_BUILD_DEVICE_DRIVER``
    the in-tree device-resident L-BFGS-B, an explicit opt-in **performance
    mode** for library callers. It carries a certified
    GPU speed-up and keeps its own tests; it is not the mirror.

Historical history-10 speed results describe an earlier implementation only.
"""

from __future__ import annotations

from typing import Final

from simsopt_jax.examples.fused_lane import (
    PreparedFusedLaneSolve,
    prepare_fused_lane_solve,
    solve_fused_lane,
)
from simsopt_jax.examples.scalar_stage import solve_scalar_stage
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions

# Official examples/3_Advanced/stage_two_optimization_finitebuild.py policy.
FINITE_BUILD_LBFGS_HISTORY: Final[int] = 400
FINITE_BUILD_TOLERANCE: Final[float] = 1.0e-20
FINITE_BUILD_NATIVE_ITERATIONS: Final[int] = 400
# Upstream names maxiter, maxcor and tol and leaves the evaluation budget at
# SciPy's own default, so the number has one owner: SciPy's option contract.
FINITE_BUILD_MAX_FUNCTION_EVALUATIONS: Final[int] = ScipyLBFGSBOptions.maxfun

#: The provider the official script calls, driven over the JAX objective.
FINITE_BUILD_OFFICIAL_DRIVER: Final[Driver] = Driver.SCIPY_LBFGSB
#: The device-resident reimplementation, selected explicitly for speed.
FINITE_BUILD_DEVICE_DRIVER: Final[Driver] = Driver.SIMSOPT_LBFGSB

# The finite-build lane prepares exactly the shared fused lane; the name is
# kept because the Advanced example and the strict-transfer test spell the
# workflow this way.
PreparedFiniteBuildStageTwo = PreparedFusedLaneSolve
prepare_finite_build_stage_two = prepare_fused_lane_solve


def solve_finite_build_stage_two(
    prepared: PreparedFusedLaneSolve,
    *,
    driver: Driver,
    max_steps: int,
    rtol: float,
    atol: float,
    line_search_max_steps: int | None = None,
) -> OptimizerResult:
    """Solve with the official finite-build history and evaluation budget.

    ``driver`` has no default on purpose: which optimizer ran is the
    difference between the mirror and the performance mode, so every caller
    states it.  Solving restarts from ``prepared.initial_parameters`` on
    either route.
    """
    if driver == FINITE_BUILD_OFFICIAL_DRIVER:
        if line_search_max_steps is not None:
            raise TypeError(
                "line_search_max_steps is a SIMSOPT_BFGS option; the official "
                "provider leaves SciPy's own maxls at its default"
            )
        if rtol != atol:
            raise ValueError(
                "the official provider is scipy.optimize.minimize(..., tol=...), "
                "which sets ftol and gtol together, so rtol and atol must be equal"
            )
        prepared.problem.x = prepared.initial_parameters
        return solve_scalar_stage(
            prepared.problem,
            driver=driver,
            max_steps=max_steps,
            maxcor=FINITE_BUILD_LBFGS_HISTORY,
            tol=rtol,
        )
    return solve_fused_lane(
        prepared,
        driver=driver,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        lbfgs_history=FINITE_BUILD_LBFGS_HISTORY,
        max_function_evaluations=FINITE_BUILD_MAX_FUNCTION_EVALUATIONS,
        line_search_max_steps=line_search_max_steps,
    )


__all__ = [
    "FINITE_BUILD_DEVICE_DRIVER",
    "FINITE_BUILD_LBFGS_HISTORY",
    "FINITE_BUILD_MAX_FUNCTION_EVALUATIONS",
    "FINITE_BUILD_NATIVE_ITERATIONS",
    "FINITE_BUILD_OFFICIAL_DRIVER",
    "FINITE_BUILD_TOLERANCE",
    "PreparedFiniteBuildStageTwo",
    "prepare_finite_build_stage_two",
    "solve_finite_build_stage_two",
]
