"""JAX port of ``examples/2_Intermediate/boozerQA.py``.

The host constructs the NCSX coils and the initial volume-labelled
``SurfaceXYZTensorFourier``. ``BoozerQAProblem`` then runs upstream's own
algorithm on the device: every objective evaluation re-solves the Boozer surface
with the dense analytic Newton route (direct solve, undamped step, tolerance
1e-13, cap 20) and returns upstream's ``J = 1e3`` sentinel with the previous
surface restored when that solve fails, and the outer gradient is the
implicit-function derivative through a transpose solve of the same Jacobian. The
outer optimization remains host-driven so the nested surface solves are compiled
as bounded kernels instead of one memory-heavy optimization graph.

Every official setting comes from ``simsopt_jax.examples.boozer_official``, and
the workflow itself is owned by
``simsopt_jax_adapters.geo.boozer_qa_problem``, so the shipped script and the
tested workflow cannot drift apart.

The outer solve is budget-bound at the official settings, on upstream as well as
here, so ``solver_success`` is false while the run is healthy: the published
``outer_stopping_reason`` names the endpoint's mode, and the reported status is
gated on it -- a budget exit is upstream's own mode and stays admissible, while
a failed line search or a non-finite endpoint cannot report ``ok``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from simsopt.configs import get_data
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.geo.curve import Curve
from simsopt_contracts.optimization_endpoint import certify_optimization_endpoint
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
)
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS,
    OFFICIAL_QA_INITIAL_IOTA,
    OFFICIAL_QA_LINE_SEARCH_MAXITER,
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NEWTON_TOLERANCE,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_OUTER_MAXITER,
    OFFICIAL_QA_SURFACE_DISTANCE,
    OFFICIAL_QA_SURFACE_RESOLUTION,
)
from simsopt_jax.examples.single_stage_boozer_vacuum import OUTER_GRADIENT_TOLERANCE
from simsopt_jax.geo.optimizer_host_lbfgs import (
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
)
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAProblem

EXAMPLE_ID = "native-boozerqa"
NATIVE_OUTER_ITERATIONS = OFFICIAL_QA_OUTER_MAXITER
#: ``minimize_bfgs_host_core`` is the host core of the official outer method
#: (upstream ``minimize(..., method='BFGS')``); its termination integers are the
#: ``host-bfgs`` convention of ``simsopt_contracts.optimization_endpoint``.
OUTER_STATUS_CONVENTION = "host-bfgs"


def _surface_resolution(scale: ExecutionScale) -> tuple[int, int]:
    resolution = OFFICIAL_QA_SURFACE_RESOLUTION
    return (resolution, resolution) if scale == "native_default" else (2, 2)


def _ncsx_configuration_options(scale: ExecutionScale) -> dict[str, int]:
    if scale == "native_default":
        return {}
    return {
        "coil_order": 3,
        "magnetic_axis_order": 3,
        "points_per_period": 8,
    }


def _boozer_options(scale: ExecutionScale) -> dict[str, object]:
    return {
        "newton_maxiter": OFFICIAL_QA_NEWTON_MAXITER,
        "newton_tol": (
            OFFICIAL_QA_NEWTON_TOLERANCE if scale == "native_default" else 1.0e-10
        ),
        "verbose": False,
    }


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        **_ncsx_configuration_options(scale),
    )
    magnetic_axis = cast(Curve, magnetic_axis)
    base_currents[0].fix_all()
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    mpol, ntor = _surface_resolution(scale)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, OFFICIAL_QA_SURFACE_DISTANCE, flip_theta=True)

    qs_resolution = OFFICIAL_QA_NON_QS_RESOLUTION if scale == "native_default" else 4
    # Construction solves the seed Boozer surface and freezes the iota,
    # major-radius and coil-length targets at that solution, as upstream does.
    problem = BoozerQAProblem(
        base_curves=base_curves,
        native_field=native_field,
        surface=surface,
        nfp=nfp,
        initial_G=G0,
        initial_iota=OFFICIAL_QA_INITIAL_IOTA,
        boozer_options=_boozer_options(scale),
        non_qs_resolution=qs_resolution,
    )
    initial_solver_success = problem.initial_inner_success
    initial_residual = problem.initial_residual

    initial_parameters = problem.initial_coil_dofs
    initial_value_and_gradient = problem.value_and_gradient(initial_parameters)

    optimizer_result = minimize_bfgs_host_core(
        problem.value_and_gradient,
        initial_parameters,
        maxiter=max_steps,
        gtol=OUTER_GRADIENT_TOLERANCE,
        maxls=OFFICIAL_QA_LINE_SEARCH_MAXITER,
        initial_value_and_grad=initial_value_and_gradient,
        line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
    )
    solution = np.asarray(optimizer_result.x_k, dtype=np.float64)
    final_objective = float(optimizer_result.f_k)
    endpoint = problem.endpoint(solution)
    final_residual = endpoint.boozer_residual_rms
    final_iota = endpoint.iota
    final_volume = endpoint.volume
    inner_solver_success = endpoint.inner_success
    outer_solver_success = bool(optimizer_result.converged)
    # Upstream's own BoozerQA run is budget-bound by design: MAXITER = 1e3 with
    # tol = 1e-15 (upstream examples/2_Intermediate/boozerQA.py:143-145), and
    # the official capture ends at status 1 after 1000 iterations with
    # "Maximum number of iterations has been exceeded."
    # (the official reference record native-boozerqa). So
    # ``solver_success`` false here is the honest report of the SAME budget
    # exit, not a failure of the example: the stopping reason below names it.
    # The reason comes from the repository's single owner of the emitter
    # vocabularies rather than from a table restated here, and no budget exit
    # is ever relabelled converged.
    outer_certificate = certify_optimization_endpoint(
        status_convention=OUTER_STATUS_CONVENTION,
        provider_success=outer_solver_success,
        provider_status=int(optimizer_result.status),
        iterations=int(optimizer_result.k),
        max_iterations=max_steps,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_value_and_gradient[1]))),
        final_gradient_inf_norm=float(np.max(np.abs(optimizer_result.g_k))),
        parameters_finite=bool(np.all(np.isfinite(solution))),
        observables_finite=bool(
            np.isfinite(final_objective)
            and np.isfinite(final_residual)
            and np.isfinite(final_iota)
            and np.isfinite(final_volume)
        ),
        inner_success=bool(initial_solver_success and inner_solver_success),
    )
    # The certificate is what decides whether this endpoint may be reported as
    # a success: its stopping reason must be one upstream's own run reaches.
    outer_endpoint_admissible = (
        outer_certificate.stopping_reason
        in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    )
    solver_success = bool(
        initial_solver_success and inner_solver_success and outer_solver_success
    )
    scientific_success = bool(
        initial_solver_success
        and inner_solver_success
        and outer_endpoint_admissible
        and np.isfinite(initial_residual)
        and np.isfinite(final_residual)
        and final_residual <= 1.0e-7
        and np.isfinite(initial_value_and_gradient[0])
        and np.isfinite(final_objective)
        and final_objective <= initial_value_and_gradient[0]
        and np.all(np.isfinite(optimizer_result.g_k))
        and np.isfinite(final_iota)
        and np.isfinite(final_volume)
        and abs(final_volume) > 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_residual": initial_residual,
            "final_residual": final_residual,
            "initial_objective": initial_value_and_gradient[0],
            "final_objective": final_objective,
            "non_qs_ratio": endpoint.non_qs_ratio,
            "iota": final_iota,
            "volume": final_volume,
            "major_radius_penalty": endpoint.major_radius_penalty,
            "total_base_coil_length_penalty": endpoint.length_penalty,
            "solver_success": solver_success,
            "solver_status": int(optimizer_result.status),
            "solver_iterations": int(optimizer_result.k),
            "solver_evaluations": int(optimizer_result.nfev),
            "outer_solver_success": outer_solver_success,
            "outer_solver_iteration_budget": int(max_steps),
            "outer_stopping_reason": outer_certificate.stopping_reason,
            "inner_solver_success": inner_solver_success,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-boozerqa-",
        bounded_steps=2,
        native_default_steps=NATIVE_OUTER_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
