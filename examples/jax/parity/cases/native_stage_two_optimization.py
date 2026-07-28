"""Exact matched workflow for ``2_Intermediate/stage_two_optimization.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.LandremanPaul2021_QA"

WORKFLOW_STAGES = (
    "construct_landreman_paul_qa_surface",
    "construct_four_base_coils_currents_and_stellarator_symmetries",
    "evaluate_flux_length_clearance_curvature_and_msc_terms",
    "perform_directional_taylor_test",
    "optimize_with_source_regularization_weights",
    "reduce_length_weight_and_optimize_second_stage",
    "evaluate_final_flux_geometry_and_gradient",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "surface_resolution": 32 if native_scale else 4,
        "curve_order": 5 if native_scale else 2,
        "curve_quadrature": 100 if native_scale else 16,
        "num_base_curves": 4,
        "major_radius": 1.0,
        "minor_radius": 0.5,
        "initial_current": 1.0e5,
        "first_length_weight": 1.0e-6,
        "second_length_weight": 1.0e-7,
        "curve_curve_threshold": 0.1,
        "curve_curve_weight": 1000.0,
        "curve_surface_threshold": 0.3,
        "curve_surface_weight": 10.0,
        "curvature_threshold": 5.0,
        "curvature_weight": 1.0e-6,
        "mean_squared_curvature_threshold": 5.0,
        "mean_squared_curvature_weight": 1.0e-6,
        "max_steps": 400 if native_scale else 50,
        "rtol": 1.0e-12,
        "atol": 1.0e-10,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
    }


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _mapping_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _mapping_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _build_geometry(configuration: Mapping[str, object]):
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves

    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="half period",
        nphi=_mapping_int(configuration, "surface_resolution"),
        ntheta=_mapping_int(configuration, "surface_resolution"),
    )
    surface.fix_all()
    base_curves = create_equally_spaced_curves(
        _mapping_int(configuration, "num_base_curves"),
        surface.nfp,
        stellsym=True,
        R0=_mapping_float(configuration, "major_radius"),
        R1=_mapping_float(configuration, "minor_radius"),
        order=_mapping_int(configuration, "curve_order"),
        numquadpoints=_mapping_int(configuration, "curve_quadrature"),
    )
    base_currents = [
        Current(_mapping_float(configuration, "initial_current")) for _ in base_curves
    ]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        True,
    )
    return surface, base_curves, coils


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize source-equivalent starting DOFs and Taylor direction."""
    from simsopt.field import BiotSavart

    configuration = _scale_configuration(scale)
    _surface, _base_curves, coils = _build_geometry(configuration)
    field = BiotSavart(coils)
    initial_parameters = np.asarray(field.x, dtype=np.float64)
    return create_input_bundle(
        root,
        case_id="native-stage-two-optimization",
        random_seed=1,
        arrays={
            "initial_parameters": initial_parameters,
            "taylor_direction": np.random.RandomState(1).uniform(
                size=initial_parameters.shape
            ),
        },
        configuration=configuration,
        scale=scale,
    )


def _effective_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    surface,
    base_curves,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "initial_parameters": arrays["initial_parameters"].tolist(),
            "taylor_direction": arrays["taylor_direction"].tolist(),
            "surface_dofs": np.asarray(surface.local_full_x).tolist(),
            "surface_gamma": np.asarray(surface.gamma()).tolist(),
            "surface_normal": np.asarray(surface.normal()).tolist(),
            "base_curve_dofs": [
                np.asarray(curve.local_full_x).tolist() for curve in base_curves
            ],
            "nfp": surface.nfp,
            **bundle.configuration,
        },
    )


def _values(
    prefix: str,
    *,
    parameters: np.ndarray,
    objective: float,
    objective_gradient: np.ndarray,
    squared_flux: float,
    geometric_penalty: float,
    maximum_normal_field: float,
    total_curve_length: float,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(
            objective_gradient,
            dtype=np.float64,
        ),
        f"{prefix}:squared_flux": np.asarray(squared_flux, dtype=np.float64),
        f"{prefix}:geometric_penalty": np.asarray(
            geometric_penalty,
            dtype=np.float64,
        ),
        f"{prefix}:maximum_normal_field": np.asarray(
            maximum_normal_field,
            dtype=np.float64,
        ),
        f"{prefix}:total_curve_length": np.asarray(
            total_curve_length,
            dtype=np.float64,
        ),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
        LpCurveCurvature,
        MeanSquaredCurvature,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    surface, base_curves, coils = _build_geometry(dict(bundle.configuration))
    construction_fingerprint = _effective_fingerprint(
        bundle,
        arrays,
        surface,
        base_curves,
    )
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    lengths = [CurveLength(curve) for curve in base_curves]
    curve_curve = CurveCurveDistance(
        [coil.curve for coil in coils],
        _configuration_float(bundle, "curve_curve_threshold"),
        num_basecurves=_configuration_int(bundle, "num_base_curves"),
    )
    curve_surface = CurveSurfaceDistance(
        [coil.curve for coil in coils],
        surface,
        _configuration_float(bundle, "curve_surface_threshold"),
    )
    curvatures = [
        LpCurveCurvature(
            curve,
            2,
            _configuration_float(bundle, "curvature_threshold"),
        )
        for curve in base_curves
    ]
    mean_squared_curvatures = [MeanSquaredCurvature(curve) for curve in base_curves]
    mean_squared_penalties = [
        QuadraticPenalty(
            value,
            _configuration_float(bundle, "mean_squared_curvature_threshold"),
            "max",
        )
        for value in mean_squared_curvatures
    ]
    unit_normal = surface.unitnormal().reshape((-1, 3))

    def objective(length_weight: float):
        return (
            flux
            + length_weight * sum(lengths)
            + _configuration_float(bundle, "curve_curve_weight") * curve_curve
            + _configuration_float(bundle, "curve_surface_weight") * curve_surface
            + _configuration_float(bundle, "curvature_weight") * sum(curvatures)
            + _configuration_float(bundle, "mean_squared_curvature_weight")
            * sum(mean_squared_penalties)
        )

    def state(
        prefix: str,
        parameters: np.ndarray,
        current_objective,
    ) -> dict[str, np.ndarray]:
        current_objective.x = parameters
        squared_flux = float(flux.J())
        objective_value = float(current_objective.J())
        magnetic_field = field.B()
        return _values(
            prefix,
            parameters=parameters,
            objective=objective_value,
            objective_gradient=np.asarray(
                current_objective.dJ(),
                dtype=np.float64,
            ),
            squared_flux=squared_flux,
            geometric_penalty=objective_value - squared_flux,
            maximum_normal_field=float(
                np.max(np.abs(np.sum(magnetic_field * unit_normal, axis=1)))
            ),
            total_curve_length=float(sum(length.J() for length in lengths)),
        )

    initial_parameters = arrays["initial_parameters"]
    direction = arrays["taylor_direction"]
    first_objective = objective(_configuration_float(bundle, "first_length_weight"))
    initial_values = state(
        "initial",
        initial_parameters,
        first_objective,
    )
    directional_derivative = float(
        np.vdot(initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7):
        first_objective.x = initial_parameters + epsilon * direction
        plus = float(first_objective.J())
        first_objective.x = initial_parameters - epsilon * direction
        minus = float(first_objective.J())
        taylor_errors.append((plus - minus) / (2.0 * epsilon) - directional_derivative)

    def minimize_objective(current_objective, initial: np.ndarray):
        def value_and_gradient(parameters: np.ndarray):
            current_objective.x = parameters
            return (
                float(current_objective.J()),
                np.asarray(current_objective.dJ(), dtype=np.float64),
            )

        return minimize(
            value_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": _configuration_int(bundle, "max_steps"),
                "maxcor": 300,
            },
            tol=1.0e-15,
        )

    first_result = minimize_objective(first_objective, initial_parameters)
    first_parameters = np.asarray(first_result.x, dtype=np.float64)
    first_values = state(
        "first",
        first_parameters,
        first_objective,
    )
    second_objective = objective(_configuration_float(bundle, "second_length_weight"))
    second_result = minimize_objective(second_objective, first_parameters)
    final_parameters = np.asarray(second_result.x, dtype=np.float64)
    final_values = state(
        "final",
        final_parameters,
        second_objective,
    )
    success = bool(
        np.isfinite(final_values["final:objective"])
        and final_values["final:objective"] < initial_values["initial:objective"]
        and np.all(np.isfinite(final_values["final:objective_gradient"]))
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=construction_fingerprint,
        driver="scipy_lbfgsb_two_stage",
        normalized_status="converged" if success else "failed",
        raw_status=f"{first_result.status},{second_result.status}",
        success=success,
        nit=int(first_result.nit + second_result.nit),
        nfev=int(first_result.nfev + second_result.nfev),
        njev=int(first_result.njev + second_result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **first_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.examples import solve_standard_stage_two
    from simsopt_jax.objectives import StageTwoObjectiveConfig
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    import jax

    surface, base_curves, coils = _build_geometry(dict(bundle.configuration))
    construction_fingerprint = _effective_fingerprint(
        bundle,
        arrays,
        surface,
        base_curves,
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    first_length_weight_device = jax.device_put(
        np.asarray(
            _configuration_float(bundle, "first_length_weight"),
            dtype=np.float64,
        ),
        device,
    )
    second_length_weight_device = jax.device_put(
        np.asarray(
            _configuration_float(bundle, "second_length_weight"),
            dtype=np.float64,
        ),
        device,
    )

    def config() -> StageTwoObjectiveConfig:
        return StageTwoObjectiveConfig(
            num_base_curves=_configuration_int(bundle, "num_base_curves"),
            curve_curve_minimum_distance=_configuration_float(
                bundle,
                "curve_curve_threshold",
            ),
            curve_curve_weight=_configuration_float(
                bundle,
                "curve_curve_weight",
            ),
            curve_surface_minimum_distance=_configuration_float(
                bundle,
                "curve_surface_threshold",
            ),
            curve_surface_weight=_configuration_float(
                bundle,
                "curve_surface_weight",
            ),
            curvature_threshold=_configuration_float(
                bundle,
                "curvature_threshold",
            ),
            curvature_weight=_configuration_float(bundle, "curvature_weight"),
            mean_squared_curvature_threshold=_configuration_float(
                bundle,
                "mean_squared_curvature_threshold",
            ),
            mean_squared_curvature_weight=_configuration_float(
                bundle,
                "mean_squared_curvature_weight",
            ),
        )

    device_result = solve_standard_stage_two(
        field=field,
        flux_spec=flux.fixed_surface_flux_spec(),
        surface_gamma=jax.device_put(
            np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
            device,
        ),
        surface_normal=jax.device_put(
            np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
            device,
        ),
        initial_parameters=jax.device_put(arrays["initial_parameters"], device),
        taylor_direction=jax.device_put(arrays["taylor_direction"], device),
        config=config(),
        first_length_weight=first_length_weight_device,
        second_length_weight=second_length_weight_device,
        max_steps=_configuration_int(bundle, "max_steps"),
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
    )
    initial, first, final, taylor_errors = jax.device_get(
        (
            device_result.initial,
            device_result.first,
            device_result.final,
            device_result.taylor_errors,
        )
    )

    def state_values(prefix: str, state) -> dict[str, np.ndarray]:
        return _values(
            prefix,
            parameters=np.asarray(state.parameters, dtype=np.float64),
            objective=float(state.objective),
            objective_gradient=np.asarray(
                state.objective_gradient,
                dtype=np.float64,
            ),
            squared_flux=float(state.squared_flux),
            geometric_penalty=float(state.geometric_penalty),
            maximum_normal_field=float(state.maximum_normal_field),
            total_curve_length=float(state.total_curve_length),
        )

    initial_values = state_values("initial", initial)
    first_values = state_values("first", first)
    final_values = state_values("final", final)
    success = bool(
        np.isfinite(final_values["final:objective"])
        and final_values["final:objective"] < initial_values["initial:objective"]
        and np.all(np.isfinite(final_values["final:objective_gradient"]))
    )
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=construction_fingerprint,
        driver=device_result.first_optimizer.driver.value,
        normalized_status="converged" if success else "failed",
        raw_status=(
            f"{device_result.first_optimizer.status},"
            f"{device_result.second_optimizer.status}"
        ),
        success=success,
        nit=(device_result.first_optimizer.nit + device_result.second_optimizer.nit),
        nfev=(device_result.first_optimizer.nfev + device_result.second_optimizer.nfev),
        njev=(device_result.first_optimizer.njev + device_result.second_optimizer.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **first_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact native or JAX standard Stage-II workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
