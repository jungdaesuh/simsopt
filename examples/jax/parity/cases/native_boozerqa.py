"""Exact matched workflow for ``2_Intermediate/boozerQA.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.runtime import ParityLane
from simsopt.optimization_endpoint import (
    OptimizationEndpointCertificate,
    certify_optimization_endpoint,
)
from simsopt.optimization_trajectory import (
    OptimizationMeasurementWindow,
    OptimizationTrajectoryRecorder,
)
from simsopt.single_stage_boozer_vacuum import (
    JAX_FAST_DRIVER_ID,
    JAX_OPTAX_DRIVER_ID,
    JAX_PARITY_DRIVER_ID,
    OUTER_GRADIENT_TOLERANCE,
)
from simsopt_jax.examples import ExecutionScale, scalar_example_driver
from simsopt_jax.solve.driver import Driver

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_volume_labelled_surface",
    "solve_initial_boozer_surface",
    "assemble_nonqs_iota_radius_and_length_objective",
    "evaluate_initial_objective_and_gradient",
    "optimize_coils_and_currents_with_bfgs",
    "record_final_objective_gradient_and_physics_terms",
)


@dataclass(frozen=True, slots=True)
class BoozerSingleStageSpec:
    """Immutable scientific differences between Boozer single-stage examples."""

    case_id: str
    workflow_stages: tuple[str, ...]
    bounded_resolution: int
    native_resolution: int
    inner_tolerance: float
    bounded_outer_maxiter: int
    native_outer_maxiter: int
    bounded_non_qs_sdim: int
    native_non_qs_sdim: int
    residual_weight: float
    report_residual: bool
    enforce_endpoint_certificate: bool = False


BOOZER_QA_SPEC = BoozerSingleStageSpec(
    case_id="native-boozerqa",
    workflow_stages=WORKFLOW_STAGES,
    bounded_resolution=2,
    native_resolution=6,
    inner_tolerance=1.0e-10,
    bounded_outer_maxiter=5,
    native_outer_maxiter=1_000,
    bounded_non_qs_sdim=20,
    native_non_qs_sdim=20,
    residual_weight=0.0,
    report_residual=False,
)


_InitialEvaluation = TypeVar("_InitialEvaluation")


@contextmanager
def _measurement_optimization_window(
    measurement: MeasurementExecution | None,
    evaluate_initial: Callable[[], _InitialEvaluation],
) -> Iterator[tuple[_InitialEvaluation, OptimizationTrajectoryRecorder | None]]:
    """Start trajectory time before the lane's required initial evaluation."""
    with OptimizationMeasurementWindow(
        trajectory_path=(
            measurement.trajectory_path if measurement is not None else None
        ),
        timing_path=(
            measurement.optimization_timing_path if measurement is not None else None
        ),
    ) as trajectory:
        yield evaluate_initial(), trajectory


def _scale_configuration(
    scale: ExecutionScale,
    spec: BoozerSingleStageSpec,
) -> dict[str, object]:
    native_scale = scale == "native_default"
    resolution = spec.native_resolution if native_scale else spec.bounded_resolution
    return {
        "mpol": resolution,
        "ntor": resolution,
        "inner_maxiter": 20,
        "inner_tolerance": spec.inner_tolerance,
        "outer_maxiter": (
            spec.native_outer_maxiter if native_scale else spec.bounded_outer_maxiter
        ),
        "outer_rtol": 0.0,
        "outer_atol": OUTER_GRADIENT_TOLERANCE,
        "initial_iota": -0.406,
        "surface_distance": 0.10,
        "non_qs_sdim": (
            spec.native_non_qs_sdim if native_scale else spec.bounded_non_qs_sdim
        ),
        "residual_weight": spec.residual_weight,
        "report_residual": spec.report_residual,
        "reduced_coil_order": 3,
        "reduced_axis_order": 3,
        "reduced_points_per_period": 8,
    }


def _configuration_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _problem(configuration: Mapping[str, object], scale: ExecutionScale):
    from simsopt.configs import get_data
    from simsopt.geo import SurfaceXYZTensorFourier

    options = (
        {}
        if scale == "native_default"
        else {
            "coil_order": _configuration_int(configuration, "reduced_coil_order"),
            "magnetic_axis_order": _configuration_int(
                configuration, "reduced_axis_order"
            ),
            "points_per_period": _configuration_int(
                configuration, "reduced_points_per_period"
            ),
        }
    )
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        **options,
    )
    base_currents[0].fix_all()
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    mpol = _configuration_int(configuration, "mpol")
    ntor = _configuration_int(configuration, "ntor")
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * ntor + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * mpol + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(
        magnetic_axis,
        _configuration_float(configuration, "surface_distance"),
        flip_theta=True,
    )
    return (
        base_curves,
        base_currents,
        magnetic_axis,
        int(nfp),
        native_field,
        surface,
        float(G0),
    )


def create_variant_input(
    root: Path,
    scale: ExecutionScale,
    spec: BoozerSingleStageSpec,
) -> InputBundle:
    """Freeze one configured Boozer single-stage problem for every lane."""
    configuration = _scale_configuration(scale, spec)
    (
        _base_curves,
        _base_currents,
        magnetic_axis,
        nfp,
        native_field,
        surface,
        G0,
    ) = _problem(configuration, scale)
    return create_input_bundle(
        root,
        case_id=spec.case_id,
        random_seed=1,
        arrays={
            "axis_dofs": np.asarray(
                magnetic_axis.local_full_x,
                dtype=np.float64,
            ),
            "coil_dofs": np.asarray(native_field.x, dtype=np.float64),
            "surface_dofs": np.asarray(surface.get_dofs(), dtype=np.float64),
        },
        configuration={
            **configuration,
            "nfp": nfp,
            "initial_G": G0,
        },
        scale=scale,
    )


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the bounded/full Boozer-QA state for every lane."""
    return create_variant_input(root, scale, BOOZER_QA_SPEC)


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    surface_dofs: np.ndarray,
    coil_dofs: np.ndarray,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "surface_dofs": _array_digest(surface_dofs),
            "coil_dofs": _array_digest(coil_dofs),
            **bundle.configuration,
        },
    )


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    measurement: MeasurementExecution | None = None,
) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart
    from simsopt.geo import (
        BoozerResidual,
        BoozerSurface,
        CurveLength,
        Iotas,
        MajorRadius,
        NonQuasiSymmetricRatio,
        Volume,
    )
    from simsopt.objectives import QuadraticPenalty

    (
        base_curves,
        _base_currents,
        _magnetic_axis,
        _nfp,
        native_field,
        surface,
        G0,
    ) = _problem(bundle.configuration, bundle.scale)
    volume = Volume(surface)
    solver = BoozerSurface(
        native_field,
        surface,
        volume,
        float(volume.J()),
        options={
            "newton_maxiter": _configuration_int(
                bundle.configuration,
                "inner_maxiter",
            ),
            "newton_tol": _configuration_float(
                bundle.configuration,
                "inner_tolerance",
            ),
            "verbose": False,
        },
    )
    initial_solution = solver.solve_residual_equation_exactly_newton(
        tol=_configuration_float(bundle.configuration, "inner_tolerance"),
        maxiter=_configuration_int(bundle.configuration, "inner_maxiter"),
        iota=_configuration_float(bundle.configuration, "initial_iota"),
        G=G0,
    )
    iota_target = float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius = MajorRadius(solver)
    lengths = [CurveLength(curve) for curve in base_curves]
    non_qs = NonQuasiSymmetricRatio(
        solver,
        BiotSavart(native_field.coils),
        sDIM=_configuration_int(bundle.configuration, "non_qs_sdim"),
    )
    residual = BoozerResidual(solver, native_field)
    iota_penalty = QuadraticPenalty(Iotas(solver), iota_target, "identity")
    radius_penalty = QuadraticPenalty(
        major_radius,
        float(major_radius.J()),
        "identity",
    )
    length_penalty = QuadraticPenalty(
        sum(lengths),
        float(sum(lengths).J()),
        "max",
    )
    objective = (
        non_qs
        + _configuration_float(bundle.configuration, "residual_weight") * residual
        + iota_penalty
        + radius_penalty
        + length_penalty
    )
    initial_parameters = np.asarray(objective.x, dtype=np.float64)

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

    native_iteration = 0
    with _measurement_optimization_window(
        measurement,
        lambda: value_and_grad(initial_parameters),
    ) as ((initial_objective, initial_gradient), trajectory):
        if trajectory is None:
            record_iteration = None
        else:

            def record_iteration(intermediate_result) -> None:
                nonlocal native_iteration
                native_iteration += 1
                trajectory.record(native_iteration, float(intermediate_result.fun))

        optimizer_result = minimize(
            value_and_grad,
            initial_parameters,
            jac=True,
            method="BFGS",
            options={
                "maxiter": _configuration_int(bundle.configuration, "outer_maxiter"),
                "gtol": OUTER_GRADIENT_TOLERANCE,
            },
            callback=record_iteration,
        )
    final_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
    objective.x = final_parameters
    final_objective = float(objective.J())
    final_gradient = np.asarray(objective.dJ(), dtype=np.float64)
    final_non_qs_ratio = float(non_qs.J())
    final_iota = float(solver.res["iota"])
    final_volume = float(volume.J())
    final_major_radius_penalty = float(radius_penalty.J())
    final_length_penalty = float(length_penalty.J())
    final_boozer_residual = float(residual.J())
    inner_solver_success = bool(solver.res["success"])
    outer_solver_success = bool(optimizer_result.success)
    endpoint_certificate = (
        certify_optimization_endpoint(
            status_convention="scipy-bfgs",
            provider_success=outer_solver_success,
            provider_status=int(optimizer_result.status),
            iterations=int(optimizer_result.nit),
            max_iterations=_configuration_int(
                bundle.configuration,
                "outer_maxiter",
            ),
            initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
            final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
            parameters_finite=bool(np.all(np.isfinite(final_parameters))),
            observables_finite=bool(
                np.isfinite(final_objective)
                and np.all(np.isfinite(final_gradient))
                and np.isfinite(final_non_qs_ratio)
                and np.isfinite(final_iota)
                and np.isfinite(final_volume)
                and np.isfinite(final_major_radius_penalty)
                and np.isfinite(final_length_penalty)
                and np.isfinite(final_boozer_residual)
            ),
            inner_success=bool(initial_solution["success"] and inner_solver_success),
        )
        if spec.enforce_endpoint_certificate
        else None
    )
    values = _values(
        surface_dofs=arrays["surface_dofs"],
        coil_dofs=arrays["coil_dofs"],
        initial_parameters=initial_parameters,
        initial_objective=initial_objective,
        initial_gradient=initial_gradient,
        initial_iota=iota_target,
        initial_volume=initial_volume,
        final_parameters=final_parameters,
        final_objective=final_objective,
        final_gradient=final_gradient,
        final_non_qs_ratio=final_non_qs_ratio,
        final_iota=final_iota,
        final_volume=final_volume,
        final_major_radius_penalty=final_major_radius_penalty,
        final_length_penalty=final_length_penalty,
        final_boozer_residual=final_boozer_residual,
        inner_solver_success=inner_solver_success,
        outer_solver_success=outer_solver_success,
        outer_solver_status=int(optimizer_result.status),
        report_residual=spec.report_residual,
        endpoint_certificate=endpoint_certificate,
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsopt_scipy_bfgs_with_boozer_newton",
        workflow_stages=spec.workflow_stages,
        solver_counts=(
            int(optimizer_result.nit),
            int(optimizer_result.nfev),
            int(optimizer_result.njev),
        ),
        endpoint_certificate=endpoint_certificate,
    )


def _host_float(value: object) -> float:
    import jax

    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _host_array(value: object) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(value), dtype=np.float64)


def _host_bool(value: object) -> bool:
    import jax

    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    measurement: MeasurementExecution | None = None,
) -> LaneObservation:
    from simsopt.geo import CurveLength, Volume
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.optimizer_host_lbfgs import (
        lbfgs_status_is_success,
        line_search_value_and_grad_more_thuente_host,
        minimize_bfgs_host_core,
        minimize_lbfgs_host_core,
    )
    from simsopt_jax.geo.optimizers.optimizer import (
        resolve_optimizer_backend_method,
        target_minimize,
    )
    from simsopt_jax.geo.optimizers.single_stage_routing import (
        resolve_single_stage_jax_boozer_optimizer_backend,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
    from simsopt_jax_adapters.geo.surface_objectives import (
        make_traceable_objective_runtime_bundle,
        make_traceable_objective_session,
        traceable_forward_result_outer_raw_terms,
    )

    import jax

    (
        base_curves,
        _base_currents,
        _magnetic_axis,
        nfp,
        native_field,
        surface,
        G0,
    ) = _problem(bundle.configuration, bundle.scale)
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    optimizer_backend = (
        measurement.optimizer_backend if measurement is not None else None
    )
    solver_options: dict[str, object] = {
        "newton_maxiter": _configuration_int(bundle.configuration, "inner_maxiter"),
        "newton_tol": _configuration_float(bundle.configuration, "inner_tolerance"),
        "verbose": False,
    }
    if optimizer_backend is not None:
        solver_options["optimizer_backend"] = (
            resolve_single_stage_jax_boozer_optimizer_backend(
                "jax",
                optimizer_backend,
            )
        )
    solver = BoozerSurfaceJAX(
        field,
        surface,
        volume,
        float(volume.J()),
        options=solver_options,
    )
    initial_solution = cast(
        Mapping[str, object],
        solver.run_code_traceable(
            field.coil_set_spec(),
            jax.device_put(np.asarray(surface.get_dofs(), dtype=np.float64)),
            jax.device_put(
                np.asarray(
                    _configuration_float(bundle.configuration, "initial_iota"),
                    dtype=np.float64,
                )
            ),
            jax.device_put(np.asarray(G0, dtype=np.float64)),
        ),
    )
    solver.install_traceable_solved_runtime_state(initial_solution)
    initial_inner_success = _host_bool(initial_solution["success"])
    iota_target = _host_float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius_target = float(surface.major_radius())
    total_length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    objective_configuration: dict[str, object] = {
        "non_qs_weight": 1.0,
        "residual_weight": _configuration_float(
            bundle.configuration,
            "residual_weight",
        ),
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.asarray(
            np.linspace(
                0.0,
                1.0 / nfp,
                2
                * _configuration_int(
                    bundle.configuration,
                    "non_qs_sdim",
                ),
                endpoint=False,
            ),
            dtype=np.float64,
        ),
        "non_qs_quadpoints_theta": np.asarray(
            np.linspace(
                0.0,
                1.0,
                2
                * _configuration_int(
                    bundle.configuration,
                    "non_qs_sdim",
                ),
                endpoint=False,
            ),
            dtype=np.float64,
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": total_length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }
    session = make_traceable_objective_session(
        solver,
        field,
        iota_target,
        outer_objective_config=objective_configuration,
    )
    runtime = make_traceable_objective_runtime_bundle(
        solver,
        field,
        iota_target,
        outer_objective_config=objective_configuration,
        session=session,
    )
    reporting = cast(
        Callable[..., Mapping[str, object]],
        runtime["reporting_metrics_from_solution"],
    )
    initial_parameters = np.asarray(field.x, dtype=np.float64)
    incumbent_controller = session.accepted_incumbent_host_value_and_grad()
    value_and_grad = cast(Callable, runtime["value_and_grad"])

    def evaluate_initial() -> tuple[float, np.ndarray]:
        if optimizer_backend == "optax-lbfgs":
            initial_objective_device, initial_gradient_device = value_and_grad(
                jax.device_put(initial_parameters)
            )
            jax.block_until_ready((initial_objective_device, initial_gradient_device))
            return (
                _host_float(initial_objective_device),
                _host_array(initial_gradient_device),
            )
        return incumbent_controller.value_and_grad(initial_parameters)

    with _measurement_optimization_window(
        measurement,
        evaluate_initial,
    ) as ((initial_objective, initial_gradient), trajectory):
        if trajectory is None:
            progress_callback = None
        else:

            def record_iteration(
                iteration: int,
                objective: float,
                _grad_norm: float,
            ) -> None:
                trajectory.record(iteration, objective)

            progress_callback = record_iteration
        if optimizer_backend == "optax-lbfgs":
            optimizer_result = target_minimize(
                value_and_grad,
                jax.device_put(initial_parameters),
                method=resolve_optimizer_backend_method(
                    optimizer_backend,
                    limited_memory=True,
                ),
                tol=OUTER_GRADIENT_TOLERANCE,
                maxiter=_configuration_int(bundle.configuration, "outer_maxiter"),
                options={
                    "maxcor": min(
                        _configuration_int(bundle.configuration, "outer_maxiter"),
                        200,
                    ),
                    "maxls": 20,
                },
                value_and_grad=True,
                progress_callback=progress_callback,
            )
            jax.block_until_ready(optimizer_result.x)
            driver = Driver.OPTAX_LBFGS
            final_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
            optimizer_iterations = int(optimizer_result.nit)
            optimizer_evaluations = int(optimizer_result.nfev)
            optimizer_gradient_evaluations = int(optimizer_result.njev)
            optimizer_status = int(optimizer_result.status)
            outer_solver_success = bool(optimizer_result.success)
            status_convention = "optax-lbfgs"
        else:
            driver = scalar_example_driver()
            if driver == Driver.SIMSOPT_LBFGSB:
                optimizer_result = minimize_lbfgs_host_core(
                    incumbent_controller.value_and_grad,
                    initial_parameters,
                    maxiter=_configuration_int(
                        bundle.configuration,
                        "outer_maxiter",
                    ),
                    maxcor=min(
                        _configuration_int(bundle.configuration, "outer_maxiter"),
                        200,
                    ),
                    ftol=0.0,
                    gtol=OUTER_GRADIENT_TOLERANCE,
                    maxls=20,
                    initial_value_and_grad=(initial_objective, initial_gradient),
                    callback=incumbent_controller.accept,
                    progress_callback=progress_callback,
                )
            else:
                optimizer_result = minimize_bfgs_host_core(
                    incumbent_controller.value_and_grad,
                    initial_parameters,
                    maxiter=_configuration_int(
                        bundle.configuration,
                        "outer_maxiter",
                    ),
                    gtol=OUTER_GRADIENT_TOLERANCE,
                    maxls=20,
                    initial_value_and_grad=(initial_objective, initial_gradient),
                    line_search_value_and_grad=(
                        line_search_value_and_grad_more_thuente_host
                    ),
                    callback=incumbent_controller.accept,
                    progress_callback=progress_callback,
                )
            final_parameters = np.asarray(optimizer_result.x_k, dtype=np.float64)
            optimizer_iterations = int(optimizer_result.k)
            optimizer_evaluations = int(optimizer_result.nfev)
            optimizer_gradient_evaluations = int(optimizer_result.ngev)
            optimizer_status = int(optimizer_result.status)
            outer_solver_success = bool(
                lbfgs_status_is_success(optimizer_result.status, False)
                if driver == Driver.SIMSOPT_LBFGSB
                else optimizer_result.converged
            )
            status_convention = (
                "host-lbfgsb" if driver == Driver.SIMSOPT_LBFGSB else "host-bfgs"
            )
    final_evaluation = session.evaluate_candidate_from_anchor(
        final_parameters,
        incumbent_controller.current_inner_state,
    )
    final_forward = final_evaluation.forward_result
    final_objective = _host_float(final_forward["value"])
    final_gradient = _host_array(final_evaluation.gradient)
    if optimizer_backend != "optax-lbfgs":
        provider_state_invalid = bool(
            not np.isfinite(final_objective) or not np.all(np.isfinite(final_gradient))
        )
        outer_solver_success = bool(
            lbfgs_status_is_success(optimizer_status, provider_state_invalid)
            if driver == Driver.SIMSOPT_LBFGSB
            else optimizer_result.converged
        )
    final_metrics = reporting(
        final_evaluation.candidate_inner_state.coil_dofs,
        final_forward["x"],
        final_forward["success"],
        include_distance_metrics=False,
        outer_raw_terms=traceable_forward_result_outer_raw_terms(final_forward),
    )
    jax.block_until_ready((final_forward, final_metrics))
    final_non_qs_ratio = _host_float(final_metrics["final_non_qs"])
    final_iota = _host_float(final_metrics["final_iota"])
    final_volume = _host_float(final_metrics["final_volume"])
    final_major_radius_penalty = _host_float(
        final_metrics["final_major_radius_penalty"]
    )
    final_length_penalty = _host_float(final_metrics["final_length_penalty"])
    final_boozer_residual = _host_float(final_metrics["final_boozer_residual"])
    inner_solver_success = _host_bool(final_metrics["solver_success"])
    endpoint_certificate = (
        certify_optimization_endpoint(
            status_convention=status_convention,
            provider_success=outer_solver_success,
            provider_status=optimizer_status,
            iterations=optimizer_iterations,
            max_iterations=_configuration_int(
                bundle.configuration,
                "outer_maxiter",
            ),
            initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
            final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
            parameters_finite=bool(np.all(np.isfinite(final_parameters))),
            observables_finite=bool(
                np.isfinite(final_objective)
                and np.all(np.isfinite(final_gradient))
                and np.isfinite(final_non_qs_ratio)
                and np.isfinite(final_iota)
                and np.isfinite(final_volume)
                and np.isfinite(final_major_radius_penalty)
                and np.isfinite(final_length_penalty)
                and np.isfinite(final_boozer_residual)
            ),
            inner_success=bool(initial_inner_success and inner_solver_success),
        )
        if spec.enforce_endpoint_certificate
        else None
    )
    values = _values(
        surface_dofs=arrays["surface_dofs"],
        coil_dofs=arrays["coil_dofs"],
        initial_parameters=initial_parameters,
        initial_objective=initial_objective,
        initial_gradient=initial_gradient,
        initial_iota=iota_target,
        initial_volume=initial_volume,
        final_parameters=final_parameters,
        final_objective=final_objective,
        final_gradient=final_gradient,
        final_non_qs_ratio=final_non_qs_ratio,
        final_iota=final_iota,
        final_volume=final_volume,
        final_major_radius_penalty=final_major_radius_penalty,
        final_length_penalty=final_length_penalty,
        final_boozer_residual=final_boozer_residual,
        inner_solver_success=inner_solver_success,
        outer_solver_success=outer_solver_success,
        outer_solver_status=optimizer_status,
        report_residual=spec.report_residual,
        endpoint_certificate=endpoint_certificate,
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver=(
            JAX_OPTAX_DRIVER_ID
            if driver == Driver.OPTAX_LBFGS
            else JAX_FAST_DRIVER_ID
            if driver == Driver.SIMSOPT_LBFGSB
            else JAX_PARITY_DRIVER_ID
        ),
        workflow_stages=spec.workflow_stages,
        solver_counts=(
            optimizer_iterations,
            optimizer_evaluations,
            optimizer_gradient_evaluations,
        ),
        endpoint_certificate=endpoint_certificate,
    )


def _values(
    *,
    surface_dofs: np.ndarray,
    coil_dofs: np.ndarray,
    initial_parameters: np.ndarray,
    initial_objective: float,
    initial_gradient: np.ndarray,
    initial_iota: float,
    initial_volume: float,
    final_parameters: np.ndarray,
    final_objective: float,
    final_gradient: np.ndarray,
    final_non_qs_ratio: float,
    final_iota: float,
    final_volume: float,
    final_major_radius_penalty: float,
    final_length_penalty: float,
    final_boozer_residual: float,
    inner_solver_success: bool,
    outer_solver_success: bool,
    outer_solver_status: int,
    report_residual: bool,
    endpoint_certificate: OptimizationEndpointCertificate | None,
) -> dict[str, np.ndarray]:
    values = {
        "construction:surface_dofs": surface_dofs,
        "construction:coil_dofs": coil_dofs,
        "initial:parameters": initial_parameters,
        "initial:objective": np.asarray(initial_objective, dtype=np.float64),
        "initial:gradient": initial_gradient,
        "initial:iota": np.asarray(initial_iota, dtype=np.float64),
        "initial:volume": np.asarray(initial_volume, dtype=np.float64),
        "final:parameters": final_parameters,
        "final:objective": np.asarray(final_objective, dtype=np.float64),
        "final:gradient": final_gradient,
        "final:non_qs_ratio": np.asarray(final_non_qs_ratio, dtype=np.float64),
        "final:iota": np.asarray(final_iota, dtype=np.float64),
        "final:volume": np.asarray(final_volume, dtype=np.float64),
        "final:major_radius_penalty": np.asarray(
            final_major_radius_penalty,
            dtype=np.float64,
        ),
        "final:length_penalty": np.asarray(
            final_length_penalty,
            dtype=np.float64,
        ),
        "final:inner_solver_success": np.asarray(
            inner_solver_success,
            dtype=np.bool_,
        ),
        "final:outer_solver_success": np.asarray(
            outer_solver_success,
            dtype=np.bool_,
        ),
    }
    if endpoint_certificate is not None:
        values.update(
            {
                "final:endpoint_certificate_success": np.asarray(
                    endpoint_certificate.success,
                    dtype=np.bool_,
                ),
                "final:endpoint_initial_stationary": np.asarray(
                    endpoint_certificate.initial_stationary,
                    dtype=np.bool_,
                ),
                "final:endpoint_terminal_stationary": np.asarray(
                    endpoint_certificate.terminal_stationary,
                    dtype=np.bool_,
                ),
                "final:endpoint_constraints_satisfied": np.asarray(
                    endpoint_certificate.constraints_satisfied,
                    dtype=np.bool_,
                ),
                "final:outer_solver_status": np.asarray(
                    outer_solver_status,
                    dtype=np.int64,
                ),
            }
        )
    if report_residual:
        values["final:boozer_residual"] = np.asarray(
            final_boozer_residual,
            dtype=np.float64,
        )
        values["final:boozer_residual_rms"] = np.asarray(
            np.sqrt(2.0 * final_boozer_residual),
            dtype=np.float64,
        )
    return values


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
    workflow_stages: tuple[str, ...],
    solver_counts: tuple[int, int, int],
    endpoint_certificate: OptimizationEndpointCertificate | None = None,
) -> LaneObservation:
    nit, nfev, njev = solver_counts
    objective_decreased = bool(
        np.all(np.isfinite(values["initial:gradient"]))
        and np.all(np.isfinite(values["final:gradient"]))
        and np.all(np.isfinite(values["final:parameters"]))
        and np.isfinite(float(values["final:objective"]))
        and float(values["final:objective"]) <= float(values["initial:objective"])
        and bool(values["final:inner_solver_success"])
    )
    success = bool(
        objective_decreased
        and (endpoint_certificate is None or endpoint_certificate.success)
    )
    if endpoint_certificate is None:
        normalized_status = "converged" if success else "failed"
        raw_status = (
            f"inner={bool(values['final:inner_solver_success'])};"
            f"outer={bool(values['final:outer_solver_success'])}"
        )
    else:
        normalized_status = (
            "converged"
            if success
            else "budget_exhausted"
            if endpoint_certificate.stopping_reason
            in {"iteration-limit", "evaluation-limit"}
            else "failed"
        )
        raw_status = (
            f"inner={bool(values['final:inner_solver_success'])};"
            f"outer={bool(values['final:outer_solver_success'])};"
            f"certificate={endpoint_certificate.success};"
            f"stopping_reason={endpoint_certificate.stopping_reason};"
            f"initial_stationary={endpoint_certificate.initial_stationary};"
            f"terminal_stationary={endpoint_certificate.terminal_stationary};"
            f"constraints_satisfied={endpoint_certificate.constraints_satisfied}"
        )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle,
            values["construction:surface_dofs"],
            values["construction:coil_dofs"],
        ),
        driver=driver,
        normalized_status=normalized_status,
        raw_status=raw_status,
        success=success,
        nit=nit,
        nfev=nfev,
        njev=njev,
        completed_workflow_stages=workflow_stages,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched Boozer-QA coil optimization."""
    return execute_variant(lane, bundle, arrays, BOOZER_QA_SPEC)


def execute_variant(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    *,
    measurement: MeasurementExecution | None = None,
) -> LaneObservation:
    """Execute one configured native/JAX Boozer single-stage workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays, spec, measurement)
    return _jax(lane, bundle, arrays, spec, measurement)
