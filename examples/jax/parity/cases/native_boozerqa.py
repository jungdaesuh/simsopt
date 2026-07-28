"""Exact matched workflow for ``2_Intermediate/boozerQA.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_volume_labelled_surface",
    "solve_initial_boozer_surface",
    "assemble_nonqs_iota_radius_and_length_objective",
    "evaluate_initial_objective_and_gradient",
    "optimize_coils_and_currents_with_bfgs",
    "record_final_objective_gradient_and_physics_terms",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "mpol": 6 if native_scale else 2,
        "ntor": 6 if native_scale else 2,
        "inner_maxiter": 20,
        "inner_tolerance": 1.0e-10,
        "outer_maxiter": 1_000 if native_scale else 5,
        "outer_rtol": 0.0,
        "outer_atol": 1.0e-15,
        "initial_iota": -0.406,
        "surface_distance": 0.10,
        "non_qs_sdim": 20,
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


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze reduced/full NCSX and initial surface state for every lane."""
    configuration = _scale_configuration(scale)
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
        case_id="native-boozerqa",
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
) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart
    from simsopt.geo import (
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
    )
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
    objective = non_qs + iota_penalty + radius_penalty + length_penalty
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

    initial_objective, initial_gradient = value_and_grad(initial_parameters)
    optimizer_result = minimize(
        value_and_grad,
        initial_parameters,
        jac=True,
        method="BFGS",
        options={"maxiter": _configuration_int(bundle.configuration, "outer_maxiter")},
        tol=1.0e-15,
    )
    final_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
    final_objective, final_gradient = value_and_grad(final_parameters)
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
        final_non_qs_ratio=float(non_qs.J()),
        final_iota=float(solver.res["iota"]),
        final_volume=float(volume.J()),
        final_major_radius_penalty=float(radius_penalty.J()),
        final_length_penalty=float(length_penalty.J()),
        inner_solver_success=bool(solver.res["success"]),
        outer_solver_success=bool(optimizer_result.success),
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsopt_scipy_bfgs_with_boozer_newton",
    )


def _host_float(value: object) -> float:
    import jax

    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _host_bool(value: object) -> bool:
    import jax

    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import jax
    from simsopt.geo import CurveLength, Volume
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.optimizer_host_lbfgs import (
        line_search_value_and_grad_more_thuente_host,
        minimize_bfgs_host_core,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
    from simsopt_jax_adapters.geo.surface_objectives import (
        make_traceable_objective_runtime_bundle,
        traceable_forward_result_outer_raw_terms,
    )

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
    solver = BoozerSurfaceJAX(
        field,
        surface,
        volume,
        float(volume.J()),
        options={
            "newton_maxiter": _configuration_int(bundle.configuration, "inner_maxiter"),
            "newton_tol": _configuration_float(bundle.configuration, "inner_tolerance"),
            "verbose": False,
        },
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
    iota_target = _host_float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius_target = float(surface.major_radius())
    total_length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    objective_configuration: dict[str, object] = {
        "non_qs_weight": 1.0,
        "residual_weight": 0.0,
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
    runtime = make_traceable_objective_runtime_bundle(
        solver,
        field,
        iota_target,
        outer_objective_config=objective_configuration,
    )
    objective = cast(Callable[[jax.Array], jax.Array], runtime["objective"])
    forward_result = cast(
        Callable[[jax.Array], Mapping[str, object]],
        runtime["forward_result"],
    )
    reporting = cast(
        Callable[..., Mapping[str, object]],
        runtime["reporting_metrics_from_solution"],
    )
    initial_parameters_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    initial_objective_device, initial_gradient_device = jax.value_and_grad(objective)(
        initial_parameters_device
    )
    jax.block_until_ready((initial_objective_device, initial_gradient_device))
    initial_objective = _host_float(initial_objective_device)
    initial_gradient = np.asarray(
        jax.device_get(initial_gradient_device),
        dtype=np.float64,
    )

    value_and_gradient = jax.value_and_grad(objective)

    def host_value_and_gradient(
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        parameters_device = jax.device_put(np.asarray(parameters, dtype=np.float64))
        value_device, gradient_device = value_and_gradient(parameters_device)
        jax.block_until_ready((value_device, gradient_device))
        return (
            _host_float(value_device),
            np.asarray(jax.device_get(gradient_device), dtype=np.float64),
        )

    optimizer_result = minimize_bfgs_host_core(
        host_value_and_gradient,
        np.asarray(
            jax.device_get(initial_parameters_device),
            dtype=np.float64,
        ),
        maxiter=_configuration_int(bundle.configuration, "outer_maxiter"),
        gtol=_configuration_float(bundle.configuration, "outer_atol"),
        maxls=20,
        initial_value_and_grad=(initial_objective, initial_gradient),
        line_search_value_and_grad=(line_search_value_and_grad_more_thuente_host),
    )
    final_parameters = np.asarray(optimizer_result.x_k, dtype=np.float64)
    final_parameters_device = jax.device_put(final_parameters)
    final_forward = forward_result(final_parameters_device)
    final_metrics = reporting(
        final_parameters_device,
        final_forward["x"],
        final_forward["success"],
        include_distance_metrics=False,
        outer_raw_terms=traceable_forward_result_outer_raw_terms(final_forward),
    )
    jax.block_until_ready((final_forward, final_metrics))
    initial_parameters = np.asarray(
        jax.device_get(initial_parameters_device),
        dtype=np.float64,
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
        final_objective=float(optimizer_result.f_k),
        final_gradient=np.asarray(optimizer_result.g_k, dtype=np.float64),
        final_non_qs_ratio=_host_float(final_metrics["final_non_qs"]),
        final_iota=_host_float(final_metrics["final_iota"]),
        final_volume=_host_float(final_metrics["final_volume"]),
        final_major_radius_penalty=_host_float(
            final_metrics["final_major_radius_penalty"]
        ),
        final_length_penalty=_host_float(final_metrics["final_length_penalty"]),
        inner_solver_success=_host_bool(final_metrics["solver_success"]),
        outer_solver_success=bool(optimizer_result.converged),
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver="simsopt_jax_host_bfgs_with_traceable_boozer_newton",
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
    inner_solver_success: bool,
    outer_solver_success: bool,
) -> dict[str, np.ndarray]:
    return {
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


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
) -> LaneObservation:
    success = bool(
        np.all(np.isfinite(values["initial:gradient"]))
        and np.all(np.isfinite(values["final:gradient"]))
        and np.all(np.isfinite(values["final:parameters"]))
        and np.isfinite(float(values["final:objective"]))
        and float(values["final:objective"]) <= float(values["initial:objective"])
        and bool(values["final:inner_solver_success"])
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
        normalized_status="converged" if success else "failed",
        raw_status=(
            f"inner={bool(values['final:inner_solver_success'])};"
            f"outer={bool(values['final:outer_solver_success'])}"
        ),
        success=success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched Boozer-QA coil optimization."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
