"""One L-BFGS-B stage of an example mirror, and the names a stage publishes.

Three mirrors -- standard stage two, stochastic stage two, and coil forces --
route their stages the same way and publish their optimizer outcome under one
spelling.  Both live here once, so changing the route, the options a native
twin is matched on, or a published name is a change to this file rather than a
coordinated edit across three examples.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Final

from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.dispatch import minimize as dispatch_minimize
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    TraceableScalarProblem,
    serial_solve_jax,
)


def solve_scalar_stage(
    problem: TraceableScalarProblem | TraceableParametricScalarProblem,
    *,
    driver: Driver,
    max_steps: int,
    maxcor: int,
    tol: float,
    maxls: int = ScipyLBFGSBOptions.maxls,
) -> OptimizerResult:
    """One L-BFGS-B stage from ``problem.x``; the endpoint becomes ``problem.x``.

    ``Driver.SCIPY_LBFGSB`` is the routine the native scripts use, driven over
    the device objective and named on exactly what a native call names --
    ``maxiter``, ``maxcor``, the single ``tol`` SciPy expands into ``ftol``
    and ``gtol``, and ``maxls`` when the native call names it -- so the two
    lanes stop under one rule.  ``Driver.SIMSOPT_LBFGSB`` selects the fused
    device port instead, which carries its own ``maxfun = 20 * maxiter``
    evaluation cap.
    """
    if driver == Driver.SIMSOPT_LBFGSB:
        return serial_solve_jax(
            problem,
            driver=driver,
            max_steps=max_steps,
            maxcor=maxcor,
            rtol=tol,
            atol=tol,
            require_success=False,
        )
    if driver != Driver.SCIPY_LBFGSB:
        raise ValueError(
            "solve_scalar_stage supports Driver.SCIPY_LBFGSB or "
            f"Driver.SIMSOPT_LBFGSB, got {driver!r}"
        )
    initial = problem.x
    # The cache-marked compiled callable ``serial_solve_jax`` drives internally,
    # so the two routes differ in the optimizer and in nothing else.  Neither
    # the whole-objective evaluations that function takes around its solve nor
    # its bounded-objective log write happen on this route, which keeps host
    # round trips out of the region ``wallclock_s`` brackets.
    result = dispatch_minimize(
        problem._solver_value_and_grad_fn,
        initial,
        driver=driver,
        options=ScipyLBFGSBOptions.native_matched(
            maxiter=max_steps,
            maxcor=maxcor,
            tol=tol,
            maxls=maxls,
        ),
    )
    problem.x = explicit_device_array(result.x, dtype=initial.dtype, reference=initial)
    return result


#: Every observable :func:`stage_optimizer_observables` publishes, in
#: publication order.  A driver that republishes a single-stage mirror's
#: observables reads the names from here instead of re-typing them, so a name
#: added below reaches the artifact without a second edit somewhere else.
STAGE_OPTIMIZER_OBSERVABLES: Final[tuple[str, ...]] = (
    "solver_driver",
    "solver_success",
    "solver_status",
    "solver_message",
    "solver_iterations",
    "solver_evaluations",
    "solver_gradient_evaluations",
    "solver_options",
    "minimize_seconds",
)

#: Every observable :func:`two_stage_optimizer_observables` publishes, in
#: publication order.  ``standard_solve_seconds`` is the whole call that
#: produced both stages; the name is the spelling the standard mirror
#: established and every two-stage mirror and driver now reads.
TWO_STAGE_OPTIMIZER_OBSERVABLES: Final[tuple[str, ...]] = (
    "execution_device",
    "solver_driver",
    "solver_success",
    "solver_stage_success",
    "solver_status",
    "solver_message",
    "solver_iterations",
    "solver_evaluations",
    "solver_gradient_evaluations",
    "solver_options",
    "first_stage_minimize_seconds",
    "second_stage_minimize_seconds",
    "two_stage_minimize_seconds",
    "standard_solve_seconds",
)


def _published_options(optimizer: OptimizerResult) -> dict[str, object]:
    """The options object the stage ran under, as publishable values."""
    return {
        "type": type(optimizer.options_used).__name__,
        **asdict(optimizer.options_used),
    }


def stage_optimizer_observables(optimizer: OptimizerResult) -> dict[str, object]:
    """One stage's own verdict, policy and wall clock, as publishable values.

    Nothing is collapsed into a verdict word: a run that lowered the objective
    while returning ``success=False`` at the iteration cap is visible as
    exactly that.  ``solver_options`` is the options object the solve ran
    under, so a reader compares policy against a native twin from the artifact
    rather than from a declaration.
    """
    return {
        "solver_driver": optimizer.driver.value,
        "solver_success": bool(optimizer.success),
        "solver_status": int(optimizer.status),
        "solver_message": str(optimizer.message),
        "solver_iterations": int(optimizer.nit),
        "solver_evaluations": int(optimizer.nfev),
        "solver_gradient_evaluations": int(optimizer.njev),
        "solver_options": _published_options(optimizer),
        "minimize_seconds": float(optimizer.wallclock_s),
    }


def two_stage_optimizer_observables(
    first: OptimizerResult,
    second: OptimizerResult,
    *,
    execution_device: str,
    two_stage_minimize_seconds: float,
    whole_call_seconds: float,
) -> dict[str, object]:
    """Both stages' verdicts, policies and wall clocks, as publishable values.

    No field is collapsed into a verdict word: each is a ``(first, second)``
    pair beside ``solver_success``, the conjunction of the two.
    ``two_stage_minimize_seconds`` is the region from entry of the first
    stage's minimize call to return of the second's, whatever the caller does
    between them included; the per-stage clocks are the primary numbers.
    """
    return {
        "execution_device": execution_device,
        "solver_driver": first.driver.value,
        "solver_success": bool(first.success and second.success),
        "solver_stage_success": (bool(first.success), bool(second.success)),
        "solver_status": (int(first.status), int(second.status)),
        "solver_message": (str(first.message), str(second.message)),
        "solver_iterations": (int(first.nit), int(second.nit)),
        "solver_evaluations": (int(first.nfev), int(second.nfev)),
        "solver_gradient_evaluations": (int(first.njev), int(second.njev)),
        "solver_options": (_published_options(first), _published_options(second)),
        "first_stage_minimize_seconds": float(first.wallclock_s),
        "second_stage_minimize_seconds": float(second.wallclock_s),
        "two_stage_minimize_seconds": float(two_stage_minimize_seconds),
        "standard_solve_seconds": float(whole_call_seconds),
    }


__all__ = [
    "STAGE_OPTIMIZER_OBSERVABLES",
    "TWO_STAGE_OPTIMIZER_OBSERVABLES",
    "solve_scalar_stage",
    "stage_optimizer_observables",
    "two_stage_optimizer_observables",
]
