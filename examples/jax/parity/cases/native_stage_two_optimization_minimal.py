"""Exact matched workflow for ``1_Simple/stage_two_optimization_minimal.py``."""

from __future__ import annotations

import hashlib
import os
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
    "evaluate_initial_flux_length_penalty_and_gradient",
    "perform_directional_taylor_test",
    "optimize_flux_plus_one_sided_total_length_penalty",
    "evaluate_final_flux_length_normal_field_and_gradient",
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
        "length_weight": 1.0,
        "length_target": 18.0,
        "max_steps": 300 if native_scale else 80,
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


def _build_geometry(configuration: dict[str, object]):
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves

    surface_resolution = configuration["surface_resolution"]
    curve_order = configuration["curve_order"]
    curve_quadrature = configuration["curve_quadrature"]
    num_base_curves = configuration["num_base_curves"]
    assert isinstance(surface_resolution, int)
    assert isinstance(curve_order, int)
    assert isinstance(curve_quadrature, int)
    assert isinstance(num_base_curves, int)
    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )
    base_curves = create_equally_spaced_curves(
        num_base_curves,
        surface.nfp,
        stellsym=True,
        R0=float(configuration["major_radius"]),
        R1=float(configuration["minor_radius"]),
        order=curve_order,
        numquadpoints=curve_quadrature,
    )
    base_currents = [
        Current(float(configuration["initial_current"])) for _ in base_curves
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
    """Materialize the exact initial DOFs and Taylor direction once."""
    from simsopt.field import BiotSavart
    from simsopt.geo import CurveLength
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    configuration = _scale_configuration(scale)
    surface, base_curves, coils = _build_geometry(configuration)
    field = BiotSavart(coils)
    flux = SquaredFlux(surface, field)
    total_length = sum(CurveLength(curve) for curve in base_curves)
    objective = flux + float(configuration["length_weight"]) * QuadraticPenalty(
        total_length,
        float(configuration["length_target"]),
        "max",
    )
    initial_parameters = np.asarray(objective.x, dtype=np.float64)
    return create_input_bundle(
        root,
        case_id="native-stage-two-optimization-minimal",
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
    length_penalty: float,
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
        f"{prefix}:length_penalty": np.asarray(
            length_penalty,
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
    from simsopt.geo import CurveLength
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
    total_length = sum(CurveLength(curve) for curve in base_curves)
    length_penalty = QuadraticPenalty(
        total_length,
        _configuration_float(bundle, "length_target"),
        "max",
    )
    objective = (
        flux
        + _configuration_float(
            bundle,
            "length_weight",
        )
        * length_penalty
    )
    initial_parameters = arrays["initial_parameters"]
    taylor_direction = arrays["taylor_direction"]
    unit_normal = surface.unitnormal().reshape((-1, 3))

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        objective.x = parameters
        magnetic_field = field.B()
        return _values(
            prefix,
            parameters=parameters,
            objective=float(objective.J()),
            objective_gradient=np.asarray(objective.dJ(), dtype=np.float64),
            squared_flux=float(flux.J()),
            length_penalty=(
                _configuration_float(bundle, "length_weight")
                * float(length_penalty.J())
            ),
            maximum_normal_field=float(
                np.max(np.abs(np.sum(magnetic_field * unit_normal, axis=1)))
            ),
            total_curve_length=float(total_length.J()),
        )

    initial_values = state("initial", initial_parameters)
    initial_gradient = initial_values["initial:objective_gradient"]
    directional_derivative = float(np.vdot(initial_gradient, taylor_direction))
    taylor_errors = []
    for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7):
        plus = initial_parameters + epsilon * taylor_direction
        minus = initial_parameters - epsilon * taylor_direction
        objective.x = plus
        plus_value = float(objective.J())
        objective.x = minus
        minus_value = float(objective.J())
        taylor_errors.append(
            (plus_value - minus_value) / (2.0 * epsilon) - directional_derivative
        )

    def fun(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    result = minimize(
        fun,
        initial_parameters,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": _configuration_int(bundle, "max_steps"),
            "maxcor": 300,
        },
        tol=1.0e-15,
    )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    final_values = state("final", final_parameters)
    success = bool(
        np.isfinite(final_values["final:objective"]).all()
        and float(final_values["final:objective"])
        < float(initial_values["initial:objective"])
        and np.linalg.norm(
            final_values["final:objective_gradient"],
            ord=np.inf,
        )
        <= 1.0e-4
        and float(final_values["final:total_curve_length"])
        <= 1.1 * _configuration_float(bundle, "length_target")
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
        driver="scipy_lbfgsb",
        normalized_status="converged" if success else "failed",
        raw_status=str(result.status),
        success=success,
        nit=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.examples import solve_minimal_stage_two
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
    device_result = solve_minimal_stage_two(
        field=field,
        flux_spec=flux.fixed_surface_flux_spec(),
        surface_gamma=jax.device_put(
            np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
        ),
        surface_normal=jax.device_put(
            np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3))
        ),
        initial_parameters=jax.device_put(arrays["initial_parameters"]),
        taylor_direction=jax.device_put(arrays["taylor_direction"]),
        num_base_curves=_configuration_int(bundle, "num_base_curves"),
        length_weight=_configuration_float(bundle, "length_weight"),
        length_target=_configuration_float(bundle, "length_target"),
        max_steps=_configuration_int(bundle, "max_steps"),
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
    )
    host_values = jax.device_get(
        (
            device_result.initial.parameters,
            device_result.initial.objective,
            device_result.initial.objective_gradient,
            device_result.initial.squared_flux,
            device_result.initial.length_penalty,
            device_result.initial.maximum_normal_field,
            device_result.initial.total_curve_length,
            device_result.final.parameters,
            device_result.final.objective,
            device_result.final.objective_gradient,
            device_result.final.squared_flux,
            device_result.final.length_penalty,
            device_result.final.maximum_normal_field,
            device_result.final.total_curve_length,
            device_result.taylor_errors,
        )
    )
    initial_values = _values(
        "initial",
        parameters=np.asarray(host_values[0]),
        objective=float(host_values[1]),
        objective_gradient=np.asarray(host_values[2]),
        squared_flux=float(host_values[3]),
        length_penalty=float(host_values[4]),
        maximum_normal_field=float(host_values[5]),
        total_curve_length=float(host_values[6]),
    )
    final_values = _values(
        "final",
        parameters=np.asarray(host_values[7]),
        objective=float(host_values[8]),
        objective_gradient=np.asarray(host_values[9]),
        squared_flux=float(host_values[10]),
        length_penalty=float(host_values[11]),
        maximum_normal_field=float(host_values[12]),
        total_curve_length=float(host_values[13]),
    )
    success = bool(
        device_result.optimizer.success
        and float(final_values["final:objective"])
        < float(initial_values["initial:objective"])
        and np.linalg.norm(
            final_values["final:objective_gradient"],
            ord=np.inf,
        )
        <= 1.0e-4
        and float(final_values["final:total_curve_length"])
        <= 1.1 * _configuration_float(bundle, "length_target")
    )
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=construction_fingerprint,
        driver=device_result.optimizer.driver.value,
        normalized_status="converged" if success else "failed",
        raw_status=str(device_result.optimizer.status),
        success=success,
        nit=device_result.optimizer.nit,
        nfev=device_result.optimizer.nfev,
        njev=device_result.optimizer.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **final_values,
            "taylor:errors": np.asarray(host_values[14], dtype=np.float64),
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact native or JAX minimal Stage-II workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
