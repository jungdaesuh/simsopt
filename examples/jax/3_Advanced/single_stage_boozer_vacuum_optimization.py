"""VMEC-free JAX single-stage optimization on an implicit Boozer surface.

The host constructs the native NCSX coil graph and a volume-labelled surface.
Each objective evaluation runs two bounded JAX kernels on the selected CPU or
GPU: an exact Boozer solve followed by an implicit-adjoint value/gradient
evaluation. Fast mode uses bounded-memory SIMSOPT L-BFGS; parity mode uses
SIMSOPT BFGS to match the native reference algorithm. The outer loop remains
host-driven so nested optimization is not captured in one large JIT program.

What "mirror" claims, and what it does not: this script is faithful in
objective and physics (term-for-term, proven by endpoint cross-evaluation); the
outer optimizer implementation and failed-solve handling differ from native
(SIMSOPT BFGS reimplementation vs scipy BFGS; NaN-sentinel/accepted-incumbent
vs 1.0e3-with-state-restore), so iterate trajectories are not comparable and the
parity gate checks endpoint observables only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import jax
import numpy as np
from simsopt.configs import get_data
from simsopt.geo import CurveLength, SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import Curve
from simsopt.optimization_endpoint import certify_optimization_endpoint
from simsopt.single_stage_boozer_vacuum import (
    NATIVE_ITERATIONS,
    OUTER_GRADIENT_TOLERANCE,
)
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    scalar_example_driver,
)
from simsopt_jax.geo.optimizer_host_lbfgs import (
    lbfgs_status_is_success,
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
    minimize_lbfgs_host_core,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.surface_objectives import (
    make_traceable_objective_runtime_bundle,
    make_traceable_objective_session,
    traceable_forward_result_outer_raw_terms,
)

EXAMPLE_ID = "native-single-stage-boozer-vacuum-optimization"


def _host_array(value: object) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _host_float(value: object) -> float:
    return float(_host_array(value))


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _configuration_options(native_scale: bool) -> dict[str, int]:
    if native_scale:
        return {}
    return {
        "coil_order": 3,
        "magnetic_axis_order": 3,
        "points_per_period": 8,
    }


def _objective_config(
    *,
    nfp: int,
    native_scale: bool,
    surface: SurfaceXYZTensorFourier,
    major_radius_target: float,
    length_target: float,
) -> dict[str, object]:
    qs_resolution = 20 if native_scale else 4
    return {
        "non_qs_weight": 1.0,
        "residual_weight": 1.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0,
            1.0 / nfp,
            2 * qs_resolution,
            endpoint=False,
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0,
            1.0,
            2 * qs_resolution,
            endpoint=False,
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    native_scale = scale == "native_default"
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        **_configuration_options(native_scale),
    )
    magnetic_axis = cast(Curve, magnetic_axis)
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    resolution = 6 if native_scale else 1
    surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * resolution + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * resolution + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    volume_label = Volume(surface)
    volume_target = float(volume_label.J())
    boozer_surface = BoozerSurfaceJAX(
        field,
        surface,
        volume_label,
        volume_target,
        options={
            "newton_maxiter": 20,
            "newton_tol": 1.0e-13,
            "verbose": False,
        },
    )
    initial_solve = cast(
        Mapping[str, object],
        boozer_surface.run_code_traceable(
            field.coil_set_spec(),
            jax.device_put(np.asarray(surface.get_dofs(), dtype=np.float64)),
            jax.device_put(np.asarray(-0.406, dtype=np.float64)),
            jax.device_put(np.asarray(G0, dtype=np.float64)),
        ),
    )
    boozer_surface.install_traceable_solved_runtime_state(initial_solve)
    initial_inner_success = _host_bool(initial_solve["success"])
    iota_target = _host_float(initial_solve["iota"])
    length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    objective_config = _objective_config(
        nfp=nfp,
        native_scale=native_scale,
        surface=surface,
        major_radius_target=float(surface.major_radius()),
        length_target=length_target,
    )
    session = make_traceable_objective_session(
        boozer_surface,
        field,
        iota_target,
        outer_objective_config=objective_config,
    )
    runtime_bundle = make_traceable_objective_runtime_bundle(
        boozer_surface,
        field,
        iota_target,
        outer_objective_config=objective_config,
        session=session,
    )
    reporting_metrics_from_solution = cast(
        Callable[..., Mapping[str, object]],
        runtime_bundle["reporting_metrics_from_solution"],
    )

    initial_parameters = np.asarray(field.x, dtype=np.float64)
    incumbent_controller = session.accepted_incumbent_host_value_and_grad()
    initial_value_and_gradient = incumbent_controller.value_and_grad(initial_parameters)

    driver = scalar_example_driver()
    if driver == Driver.SIMSOPT_LBFGSB:
        optimizer_result = minimize_lbfgs_host_core(
            incumbent_controller.value_and_grad,
            initial_parameters,
            maxiter=max_steps,
            maxcor=min(max_steps, 200),
            ftol=0.0,
            gtol=OUTER_GRADIENT_TOLERANCE,
            maxls=20,
            initial_value_and_grad=initial_value_and_gradient,
            callback=incumbent_controller.accept,
        )
    else:
        optimizer_result = minimize_bfgs_host_core(
            incumbent_controller.value_and_grad,
            initial_parameters,
            maxiter=max_steps,
            gtol=OUTER_GRADIENT_TOLERANCE,
            maxls=20,
            initial_value_and_grad=initial_value_and_gradient,
            line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
            callback=incumbent_controller.accept,
        )
    solution = np.asarray(optimizer_result.x_k, dtype=np.float64)
    final_evaluation = session.evaluate_candidate_from_anchor(
        solution,
        incumbent_controller.current_inner_state,
    )
    final_forward = final_evaluation.forward_result
    final_value = _host_float(final_forward["value"])
    gradient = _host_array(final_evaluation.gradient)
    final_metrics = reporting_metrics_from_solution(
        final_evaluation.candidate_inner_state.coil_dofs,
        final_forward["x"],
        final_forward["success"],
        include_distance_metrics=False,
        outer_raw_terms=traceable_forward_result_outer_raw_terms(final_forward),
    )
    jax.block_until_ready(final_metrics)
    inner_solver_success = _host_bool(final_metrics["solver_success"])
    boozer_residual = _host_float(final_metrics["final_boozer_residual"])
    final_iota = _host_float(final_metrics["final_iota"])
    final_volume = _host_float(final_metrics["final_volume"])
    final_non_qs = _host_float(final_metrics["final_non_qs"])
    parameters_finite = bool(
        np.all(np.isfinite(solution)) and np.all(np.isfinite(gradient))
    )
    observables_finite = bool(
        np.isfinite(final_value)
        and np.isfinite(final_iota)
        and np.isfinite(final_volume)
        and np.isfinite(final_non_qs)
        and np.isfinite(boozer_residual)
    )
    provider_state_invalid = bool(
        not np.isfinite(final_value) or not np.all(np.isfinite(gradient))
    )
    outer_solver_success = bool(
        lbfgs_status_is_success(optimizer_result.status, provider_state_invalid)
        if driver == Driver.SIMSOPT_LBFGSB
        else optimizer_result.converged
    )
    endpoint_certificate = certify_optimization_endpoint(
        # The example's outer drivers are the HOST cores
        # (minimize_lbfgs_host_core / minimize_bfgs_host_core), whose
        # status vocabularies differ from the benchmark runner's private
        # on-device lanes, so the emitter convention is named directly.
        status_convention=(
            "host-lbfgsb" if driver == Driver.SIMSOPT_LBFGSB else "host-bfgs"
        ),
        provider_success=outer_solver_success,
        provider_status=int(optimizer_result.status),
        iterations=int(optimizer_result.k),
        max_iterations=max_steps,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_value_and_gradient[1]))),
        final_gradient_inf_norm=float(np.max(np.abs(gradient))),
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        inner_success=bool(initial_inner_success and inner_solver_success),
    )
    scientific_success = bool(
        endpoint_certificate.success and final_value <= initial_value_and_gradient[0]
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_objective": initial_value_and_gradient[0],
            "initial_gradient": tuple(
                float(value) for value in initial_value_and_gradient[1]
            ),
            "final_objective": final_value,
            "solution": tuple(float(value) for value in solution),
            "gradient": tuple(float(value) for value in gradient),
            "inner_solver_success": inner_solver_success,
            "outer_solver_success": outer_solver_success,
            "outer_stopping_reason": endpoint_certificate.stopping_reason,
            "initial_stationary": endpoint_certificate.initial_stationary,
            "terminal_stationary": endpoint_certificate.terminal_stationary,
            "solver_status": int(optimizer_result.status),
            "solver_iterations": int(optimizer_result.k),
            "solver_evaluations": int(optimizer_result.nfev),
            "iota": final_iota,
            "volume": final_volume,
            "non_qs_ratio": final_non_qs,
            "boozer_residual": boozer_residual,
            "boozer_residual_rms": float(np.sqrt(2.0 * boozer_residual)),
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
