"""VMEC-free JAX single-stage optimization on an implicit Boozer surface.

The host builds the native NCSX coil graph and a volume-labelled Boozer surface.
Every outer evaluation is then ONE compiled JAX program on the selected CPU or
GPU: the exact Boozer Newton solve followed by the implicit-function gradient of
the objective at the state that solve returned.

This mirror runs the native example's policy, not an approximation of it: the
same problem construction, the same objective terms and weights, the same SciPy
BFGS with ``gtol`` 1.0e-15 and iteration budget, and the same failed-inner-solve
rule -- report the 1.0e3 sentinel and restore the pre-evaluation surface, iota
and G as the next warm start.  Iterate trajectories are therefore comparable
term for term, not only at the endpoint.

``status`` is the scientific finite-improvement gate -- a real Boozer state,
finite parameters and observables, and an objective below the starting one.  It
is NOT the optimizer's verdict: at ``gtol`` 1.0e-15, below this objective's
rounding floor, spending the iteration budget is native's normal termination.
That verdict is published beside it, fail-closed and unweakened, in
``outer_solver_success``, ``outer_stopping_reason``, ``solver_status`` and the
endpoint certificate's stationarity fields.

One bounded divergence remains, and only on a failed endpoint solve: native's
final block re-reads the raw objective, while this mirror publishes the sentinel
the same evaluation reported.  Such an endpoint fails the scientific gate on
both mirrors through ``inner_solver_success``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from simsopt.single_stage_boozer_vacuum import (
    NATIVE_ITERATIONS,
    OUTER_GRADIENT_TOLERANCE,
)
from simsopt_contracts.optimization_endpoint import certify_optimization_endpoint
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
)
from simsopt_jax.solve import Driver, ScipyBFGSOptions
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    BOUNDED_SCALE,
    NATIVE_SCALE,
    SingleStageVacuumProblem,
)

EXAMPLE_ID = "native-single-stage-boozer-vacuum-optimization"


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    native_scale = scale == "native_default"
    problem = SingleStageVacuumProblem(NATIVE_SCALE if native_scale else BOUNDED_SCALE)
    initial_objective, initial_gradient = problem.value_and_gradient(
        problem.initial_coil_dofs
    )

    optimizer_result = minimize(
        problem.value_and_gradient,
        problem.initial_coil_dofs,
        driver=Driver.SCIPY_BFGS,
        options=ScipyBFGSOptions(maxiter=max_steps, gtol=OUTER_GRADIENT_TOLERANCE),
    )

    solution = np.asarray(optimizer_result.x, dtype=np.float64)
    endpoint = problem.endpoint(solution)
    gradient = endpoint.gradient
    parameters_finite = bool(
        np.all(np.isfinite(solution)) and np.all(np.isfinite(gradient))
    )
    observables_finite = bool(
        np.isfinite(endpoint.value)
        and np.isfinite(endpoint.iota)
        and np.isfinite(endpoint.volume)
        and np.isfinite(endpoint.non_qs_ratio)
        and np.isfinite(endpoint.boozer_residual)
        and np.isfinite(endpoint.boozer_residual_rms)
    )
    # The seed exact solve is a construction precondition of the problem -- it
    # raises rather than returning an unsolved surface -- so the endpoint solve
    # is the whole of native's ``initial and final`` inner-success conjunction.
    inner_success = endpoint.inner_success
    endpoint_certificate = certify_optimization_endpoint(
        status_convention="scipy-bfgs",
        provider_success=bool(optimizer_result.success),
        provider_status=int(optimizer_result.status),
        iterations=int(optimizer_result.nit),
        max_iterations=max_steps,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_gradient_inf_norm=float(np.max(np.abs(gradient))),
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        inner_success=inner_success,
    )
    # ``status`` is the scientific finite-improvement gate: a real Boozer state,
    # finite parameters and observables, and an objective below the one this run
    # started from.  The optimizer's own verdict is published beside it and never
    # folded into it -- at ``gtol`` 1.0e-15, which sits below the rounding floor
    # of this objective, spending the iteration budget is native's normal
    # termination, not a defect.  ``outer_solver_success``,
    # ``outer_stopping_reason``, ``solver_status`` and the certificate's
    # stationarity fields carry that verdict, unweakened and fail-closed.
    scientific_success = bool(
        inner_success
        and parameters_finite
        and observables_finite
        and endpoint.value < initial_objective
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_objective": initial_objective,
            "initial_gradient": tuple(float(value) for value in initial_gradient),
            "final_objective": endpoint.value,
            "solution": tuple(float(value) for value in solution),
            "gradient": tuple(float(value) for value in gradient),
            "inner_solver_success": inner_success,
            "outer_solver_success": bool(optimizer_result.success),
            "outer_stopping_reason": endpoint_certificate.stopping_reason,
            "initial_stationary": endpoint_certificate.initial_stationary,
            "terminal_stationary": endpoint_certificate.terminal_stationary,
            "solver_status": int(optimizer_result.status),
            "solver_iterations": int(optimizer_result.nit),
            "solver_evaluations": int(optimizer_result.nfev),
            "iota": endpoint.iota,
            "volume": endpoint.volume,
            "non_qs_ratio": endpoint.non_qs_ratio,
            "boozer_residual": endpoint.boozer_residual,
            "boozer_residual_rms": endpoint.boozer_residual_rms,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-single-stage-boozer-",
        bounded_steps=2,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
