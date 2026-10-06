"""A short ``2_Intermediate/boozerQA.py`` outer run: native library against the JAX mirror.

Both sides start from the reduced NCSX construction (coil and axis order 3,
eight points per period, ``mpol = ntor = 2``, Newton tolerance 1e-10) and run
five BFGS iterations: the native side is upstream's objective over the native
exact Newton solve under ``scipy.optimize.minimize(method='BFGS')``; the JAX
side is :class:`~simsopt_jax_adapters.geo.boozer_qa_problem.BoozerQAProblem`
under the host BFGS core the shipped mirror drives.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveLength,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.objectives import QuadraticPenalty
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    StoppingReason,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_INITIAL_IOTA,
    OFFICIAL_QA_LINE_SEARCH_MAXITER,
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_RESIDUAL_WEIGHT,
    OFFICIAL_QA_SURFACE_DISTANCE,
)
from simsopt_jax.examples.single_stage_boozer_vacuum import OUTER_GRADIENT_TOLERANCE
from simsopt_jax.geo.optimizer_host_lbfgs import (
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
)
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAProblem

_REDUCED_COILS = {"coil_order": 3, "magnetic_axis_order": 3, "points_per_period": 8}
_RESOLUTION = 2
_NEWTON_TOLERANCE = 1.0e-10
_OUTER_MAXITER = 5


def _construction():
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx", **_REDUCED_COILS
    )
    base_currents[0].fix_all()
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    initial_G = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    surface = SurfaceXYZTensorFourier(
        mpol=_RESOLUTION,
        ntor=_RESOLUTION,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * _RESOLUTION + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * _RESOLUTION + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, OFFICIAL_QA_SURFACE_DISTANCE, flip_theta=True)
    return base_curves, native_field, surface, int(nfp), float(initial_G)


def _boozer_options() -> dict[str, object]:
    return {
        "newton_maxiter": OFFICIAL_QA_NEWTON_MAXITER,
        "newton_tol": _NEWTON_TOLERANCE,
        "verbose": False,
    }


def _labelled(
    convention: StatusConvention,
    *,
    provider_success: bool,
    provider_status: int,
    iterations: int,
    initial_gradient: np.ndarray,
    final_gradient: np.ndarray,
    final_parameters: np.ndarray,
    observables_finite: bool,
    inner_success: bool,
) -> StoppingReason:
    return certify_optimization_endpoint(
        status_convention=convention,
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=_OUTER_MAXITER,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
        parameters_finite=bool(np.all(np.isfinite(final_parameters))),
        observables_finite=observables_finite,
        inner_success=inner_success,
    ).stopping_reason


def _native_run() -> dict[str, object]:
    """Upstream boozerQA.py's objective and failure policy under SciPy BFGS."""
    base_curves, native_field, surface, _nfp, initial_G = _construction()
    construction_surface_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    volume = Volume(surface)
    solver = BoozerSurface(
        native_field, surface, volume, float(volume.J()), options=_boozer_options()
    )
    initial_solution = solver.solve_residual_equation_exactly_newton(
        tol=_NEWTON_TOLERANCE,
        maxiter=OFFICIAL_QA_NEWTON_MAXITER,
        iota=OFFICIAL_QA_INITIAL_IOTA,
        G=initial_G,
    )
    iota_target = float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius = MajorRadius(solver)
    total_length = sum(CurveLength(curve) for curve in base_curves)
    non_qs = NonQuasiSymmetricRatio(
        solver, BiotSavart(native_field.coils), sDIM=OFFICIAL_QA_NON_QS_RESOLUTION
    )
    radius_penalty = QuadraticPenalty(major_radius, float(major_radius.J()), "identity")
    length_penalty = QuadraticPenalty(total_length, float(total_length.J()), "max")
    objective = (
        non_qs
        + OFFICIAL_QA_RESIDUAL_WEIGHT * BoozerResidual(solver, native_field)
        + QuadraticPenalty(Iotas(solver), iota_target, "identity")
        + radius_penalty
        + length_penalty
    )

    def value_and_grad(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        previous_surface = np.asarray(solver.surface.x, dtype=np.float64)
        previous_iota = float(solver.res["iota"])
        previous_G = float(solver.res["G"])
        objective.x = parameters
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=np.float64)
        if not bool(solver.res["success"]):
            value = 1.0e3
            solver.surface.x = previous_surface
            solver.res["iota"] = previous_iota
            solver.res["G"] = previous_G
        return value, gradient

    initial_parameters = np.asarray(objective.x, dtype=np.float64)
    initial_objective, initial_gradient = value_and_grad(initial_parameters)
    result = minimize(
        value_and_grad,
        initial_parameters,
        jac=True,
        method="BFGS",
        options={"maxiter": _OUTER_MAXITER, "gtol": OUTER_GRADIENT_TOLERANCE},
    )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    objective.x = final_parameters
    values: dict[str, object] = {
        "construction:surface_dofs": construction_surface_dofs,
        "initial:parameters": initial_parameters,
        "initial:objective": initial_objective,
        "initial:gradient": initial_gradient,
        "initial:iota": iota_target,
        "initial:volume": initial_volume,
        "final:parameters": final_parameters,
        "final:objective": float(objective.J()),
        "final:gradient": np.asarray(objective.dJ(), dtype=np.float64),
        "final:non_qs_ratio": float(non_qs.J()),
        "final:iota": float(solver.res["iota"]),
        "final:volume": float(volume.J()),
        "final:major_radius_penalty": float(radius_penalty.J()),
        "final:length_penalty": float(length_penalty.J()),
        "final:inner_solver_success": bool(solver.res["success"]),
    }
    values["stopping_reason"] = _labelled(
        "scipy-bfgs",
        provider_success=bool(result.success),
        provider_status=int(result.status),
        iterations=int(result.nit),
        initial_gradient=initial_gradient,
        final_gradient=values["final:gradient"],
        final_parameters=final_parameters,
        observables_finite=all(
            bool(np.all(np.isfinite(values[key])))
            for key in values
            if key.startswith("final:")
        ),
        inner_success=bool(initial_solution["success"])
        and bool(values["final:inner_solver_success"]),
    )
    return values


def _jax_run() -> dict[str, object]:
    """The shipped mirror's problem under the host BFGS core it drives."""
    base_curves, native_field, surface, nfp, initial_G = _construction()
    construction_surface_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    problem = BoozerQAProblem(
        base_curves=base_curves,
        native_field=native_field,
        surface=surface,
        nfp=nfp,
        initial_G=initial_G,
        initial_iota=OFFICIAL_QA_INITIAL_IOTA,
        boozer_options=_boozer_options(),
        non_qs_resolution=OFFICIAL_QA_NON_QS_RESOLUTION,
        residual_weight=OFFICIAL_QA_RESIDUAL_WEIGHT,
    )
    initial_parameters = problem.initial_coil_dofs
    initial_objective, initial_gradient = problem.value_and_gradient(initial_parameters)
    result = minimize_bfgs_host_core(
        problem.value_and_gradient,
        initial_parameters,
        maxiter=_OUTER_MAXITER,
        gtol=OUTER_GRADIENT_TOLERANCE,
        maxls=OFFICIAL_QA_LINE_SEARCH_MAXITER,
        initial_value_and_grad=(initial_objective, initial_gradient),
        line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
    )
    final_parameters = np.asarray(result.x_k, dtype=np.float64)
    endpoint = problem.endpoint(final_parameters)
    values: dict[str, object] = {
        "construction:surface_dofs": construction_surface_dofs,
        "initial:parameters": np.asarray(initial_parameters, dtype=np.float64),
        "initial:objective": float(initial_objective),
        "initial:gradient": np.asarray(initial_gradient, dtype=np.float64),
        "initial:iota": problem.iota_target,
        "initial:volume": problem.initial_volume,
        "final:parameters": final_parameters,
        "final:objective": endpoint.value,
        "final:gradient": np.asarray(endpoint.gradient, dtype=np.float64),
        "final:non_qs_ratio": endpoint.non_qs_ratio,
        "final:iota": endpoint.iota,
        "final:volume": endpoint.volume,
        "final:major_radius_penalty": endpoint.major_radius_penalty,
        "final:length_penalty": endpoint.length_penalty,
        "final:inner_solver_success": bool(endpoint.inner_success),
    }
    values["stopping_reason"] = _labelled(
        "host-bfgs",
        provider_success=bool(result.converged),
        provider_status=int(result.status),
        iterations=int(result.k),
        initial_gradient=values["initial:gradient"],
        final_gradient=values["final:gradient"],
        final_parameters=final_parameters,
        observables_finite=bool(
            all(
                bool(np.all(np.isfinite(values[key])))
                for key in values
                if key.startswith("final:")
            )
            and np.isfinite(endpoint.boozer_residual)
        ),
        inner_success=bool(problem.initial_inner_success and endpoint.inner_success),
    )
    return values


def _terminal(values: dict[str, object]):
    """The fold of the outer stopping reason with the objective-decrease predicate."""
    decreased = bool(
        np.all(np.isfinite(values["initial:gradient"]))
        and np.all(np.isfinite(values["final:gradient"]))
        and np.all(np.isfinite(values["final:parameters"]))
        and np.isfinite(float(values["final:objective"]))
        and float(values["final:objective"]) <= float(values["initial:objective"])
        and bool(values["final:inner_solver_success"])
    )
    return normalized_terminal_status(
        scientific_predicate=decreased,
        stage_stopping_reasons=(values["stopping_reason"],),
    )


def test_short_boozerqa_run_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_run()
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = _jax_run()

    # The five-iteration budget ends both runs on their iteration cap: neither
    # convergence nor failure.
    for values in (native, jax):
        assert values["stopping_reason"] == "iteration-limit"
        terminal = _terminal(values)
        assert terminal.normalized_status == "budget_exhausted"
        assert terminal.success is False
    assert set(native) == set(jax)

    for observable in (
        "construction:surface_dofs",
        "initial:parameters",
        "initial:objective",
        "initial:gradient",
        "initial:iota",
        "initial:volume",
    ):
        np.testing.assert_allclose(
            jax[observable],
            native[observable],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    for observable in (
        "final:objective",
        "final:non_qs_ratio",
        "final:iota",
        "final:volume",
        "final:major_radius_penalty",
        "final:length_penalty",
    ):
        np.testing.assert_allclose(
            jax[observable],
            native[observable],
            rtol=1.0e-3,
            atol=1.0e-8,
        )

    np.testing.assert_allclose(
        jax["final:parameters"],
        native["final:parameters"],
        rtol=0.0,
        atol=2.0e-3,
    )
    assert float(native["final:objective"]) <= float(native["initial:objective"])
    assert float(jax["final:objective"]) <= float(jax["initial:objective"])
