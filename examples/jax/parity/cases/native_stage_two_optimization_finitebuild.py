"""Exact matched workflow for finite-build Stage-II optimization."""

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
    "construct_landreman_paul_qa_surface_and_base_coils",
    "construct_shared_frame_multifilament_coil_packs",
    "evaluate_flux_length_and_clearance_terms",
    "perform_directional_taylor_test",
    "optimize_scaled_finite_build_objective",
    "evaluate_final_flux_geometry_frame_and_gradient",
)
_TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)


def _configuration(scale: ExecutionScale) -> dict[str, object]:
    native = scale == "native_default"
    return {
        "surface_resolution": 32 if native else 4,
        "curve_order": 5 if native else 2,
        "curve_quadrature": 75 if native else 8,
        "num_base_curves": 4,
        "major_radius": 1.0,
        "minor_radius": 0.7,
        "initial_total_current": 1.0e5,
        "num_filaments_normal": 2,
        "num_filaments_binormal": 3,
        "gap_size_normal": 0.02,
        "gap_size_binormal": 0.04,
        "rotation_order": 1,
        "length_weight": 1.0e-2,
        "curve_curve_threshold": 0.1,
        "curve_curve_weight": 10.0,
        "objective_scale": 1.0e-4,
        "max_steps": 400 if native else 3,
        "rtol": 1.0e-15,
        "atol": 1.0e-12,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
    }


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _mapping_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _mapping_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _build_geometry(configuration: Mapping[str, object]):
    from simsopt.field import (
        Coil,
        Current,
        apply_symmetries_to_currents,
        apply_symmetries_to_curves,
    )
    from simsopt.geo import (
        CurveLength,
        SurfaceRZFourier,
        create_equally_spaced_curves,
        create_multifilament_grid,
    )
    from simsopt_jax.core import compute_filament_offsets
    from simsopt_jax_adapters.objectives import FiniteBuildStageTwoConfig

    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=_mapping_int(configuration, "surface_resolution"),
        ntheta=_mapping_int(configuration, "surface_resolution"),
    )
    base_curves = create_equally_spaced_curves(
        _mapping_int(configuration, "num_base_curves"),
        surface.nfp,
        stellsym=True,
        R0=_mapping_float(configuration, "major_radius"),
        R1=_mapping_float(configuration, "minor_radius"),
        order=_mapping_int(configuration, "curve_order"),
        numquadpoints=_mapping_int(configuration, "curve_quadrature"),
        use_jax_curve=False,
    )
    filament_count = _mapping_int(
        configuration,
        "num_filaments_normal",
    ) * _mapping_int(configuration, "num_filaments_binormal")
    base_currents = []
    for index in range(_mapping_int(configuration, "num_base_curves")):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(
            current
            * (_mapping_float(configuration, "initial_total_current") / filament_count)
        )
    base_filaments = sum(
        (
            create_multifilament_grid(
                curve,
                _mapping_int(configuration, "num_filaments_normal"),
                _mapping_int(configuration, "num_filaments_binormal"),
                _mapping_float(configuration, "gap_size_normal"),
                _mapping_float(configuration, "gap_size_binormal"),
                rotation_order=_mapping_int(configuration, "rotation_order"),
            )
            for curve in base_curves
        ),
        [],
    )
    filament_currents = sum(
        ([current] * filament_count for current in base_currents),
        [],
    )
    filament_curves = apply_symmetries_to_curves(
        base_filaments,
        surface.nfp,
        True,
    )
    currents = apply_symmetries_to_currents(
        filament_currents,
        surface.nfp,
        True,
    )
    symmetric_base_curves = apply_symmetries_to_curves(
        base_curves,
        surface.nfp,
        True,
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(filament_curves, currents, strict=True)
    ]
    length_targets = tuple(float(CurveLength(curve).J()) for curve in base_curves)
    config = FiniteBuildStageTwoConfig(
        num_base_curves=_mapping_int(configuration, "num_base_curves"),
        filament_offsets=compute_filament_offsets(
            numfilaments_n=_mapping_int(
                configuration,
                "num_filaments_normal",
            ),
            numfilaments_b=_mapping_int(
                configuration,
                "num_filaments_binormal",
            ),
            gapsize_n=_mapping_float(configuration, "gap_size_normal"),
            gapsize_b=_mapping_float(configuration, "gap_size_binormal"),
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=length_targets,
        length_weight=_mapping_float(configuration, "length_weight"),
        curve_curve_minimum_distance=_mapping_float(
            configuration,
            "curve_curve_threshold",
        ),
        curve_curve_weight=_mapping_float(
            configuration,
            "curve_curve_weight",
        ),
    )
    return surface, base_curves, symmetric_base_curves, coils, config


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze source-equivalent finite-build DOFs and Taylor direction."""
    from simsopt.field import BiotSavart

    configuration = _configuration(scale)
    _, _, _, coils, config = _build_geometry(configuration)
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    return create_input_bundle(
        root,
        case_id="native-stage-two-optimization-finitebuild",
        random_seed=1,
        arrays={
            "initial_parameters": initial_parameters,
            "taylor_direction": np.random.RandomState(1).uniform(
                size=initial_parameters.shape
            ),
            "filament_offsets": np.asarray(
                config.filament_offsets,
                dtype=np.float64,
            ),
        },
        configuration={
            **configuration,
            "length_targets": list(config.length_targets),
            "symmetry_copies": config.symmetry_copies,
        },
        scale=scale,
    )


def _fingerprint(
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
            "filament_offsets": arrays["filament_offsets"].tolist(),
            "surface_dofs": np.asarray(surface.local_full_x).tolist(),
            "surface_gamma": np.asarray(surface.gamma()).tolist(),
            "surface_normal": np.asarray(surface.normal()).tolist(),
            "base_curve_dofs": [
                np.asarray(curve.local_full_x).tolist() for curve in base_curves
            ],
            **bundle.configuration,
        },
    )


def _state_values(
    prefix: str,
    *,
    parameters: np.ndarray,
    objective: float,
    gradient: np.ndarray,
    squared_flux: float,
    length_penalty: float,
    distance_penalty: float,
    coil_lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(gradient, dtype=np.float64),
        f"{prefix}:squared_flux": np.asarray(squared_flux, dtype=np.float64),
        f"{prefix}:length_penalty": np.asarray(length_penalty, dtype=np.float64),
        f"{prefix}:distance_penalty": np.asarray(
            distance_penalty,
            dtype=np.float64,
        ),
        f"{prefix}:coil_lengths": np.asarray(coil_lengths, dtype=np.float64),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart
    from simsopt.geo import CurveCurveDistance, CurveLength
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    surface, base_curves, symmetric_base_curves, coils, config = _build_geometry(
        bundle.configuration
    )
    fingerprint = _fingerprint(bundle, arrays, surface, base_curves)
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    lengths = [CurveLength(curve) for curve in base_curves]
    distance = CurveCurveDistance(
        symmetric_base_curves,
        _configuration_float(bundle, "curve_curve_threshold"),
    )
    length_term = _configuration_float(bundle, "length_weight") * sum(
        QuadraticPenalty(length, target, "max")
        for length, target in zip(lengths, config.length_targets, strict=True)
    )
    distance_term = (
        _configuration_float(
            bundle,
            "curve_curve_weight",
        )
        * distance
    )
    objective = flux + length_term + distance_term

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        objective.x = parameters
        squared_flux = float(flux.J())
        length_penalty = float(length_term.J())
        distance_penalty = float(distance_term.J())
        return _state_values(
            prefix,
            parameters=parameters,
            objective=squared_flux + length_penalty + distance_penalty,
            gradient=np.asarray(objective.dJ(), dtype=np.float64),
            squared_flux=squared_flux,
            length_penalty=length_penalty,
            distance_penalty=distance_penalty,
            coil_lengths=np.asarray(
                [length.J() for length in lengths],
                dtype=np.float64,
            ),
        )

    initial_parameters = arrays["initial_parameters"]
    initial_values = state("initial", initial_parameters)
    scale = _configuration_float(bundle, "objective_scale")
    direction = arrays["taylor_direction"]
    directional_derivative = float(
        np.vdot(scale * initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8):
        objective.x = initial_parameters + epsilon * direction
        plus = scale * float(objective.J())
        objective.x = initial_parameters - epsilon * direction
        minus = scale * float(objective.J())
        taylor_errors.append((plus - minus) / (2 * epsilon) - directional_derivative)

    def value_and_gradient(parameters: np.ndarray):
        objective.x = parameters
        return (
            scale * float(objective.J()),
            scale * np.asarray(objective.dJ(), dtype=np.float64),
        )

    result = minimize(
        value_and_gradient,
        initial_parameters,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": _configuration_int(bundle, "max_steps"),
            "maxcor": min(_configuration_int(bundle, "max_steps"), 400),
            "gtol": 1.0e-20,
            "ftol": 1.0e-20,
        },
        tol=1.0e-20,
    )
    final_values = state(
        "final",
        np.asarray(result.x, dtype=np.float64),
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
        effective_construction_fingerprint=fingerprint,
        driver="scipy_lbfgsb_finite_build",
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
    import jax
    import jax.numpy as jnp

    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax.solve.serial import (
        TraceableArrayFunction,
        TraceableScalarProblem,
        serial_solve_jax,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives import (
        finite_build_stage_two_diagnostics,
        make_finite_build_stage_two_objective,
    )
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    surface, base_curves, _, coils, config = _build_geometry(bundle.configuration)
    fingerprint = _fingerprint(bundle, arrays, surface, base_curves)
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    objective = make_finite_build_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        config,
    )
    diagnostics = finite_build_stage_two_diagnostics(
        field,
        flux.fixed_surface_flux_spec(),
        config,
    )
    device = get_runtime_jax_device()
    scale = _configuration_float(bundle, "objective_scale")

    def scaled_objective(parameters: jax.Array) -> jax.Array:
        return scale * objective(parameters)

    initial_parameters = jax.device_put(arrays["initial_parameters"], device)
    reporting_problem = TraceableScalarProblem(objective, initial_parameters)
    scaled_problem = TraceableScalarProblem(scaled_objective, initial_parameters)
    diagnostic_program = TraceableArrayFunction(diagnostics, initial_parameters)

    def state(prefix: str, parameters: jax.Array) -> dict[str, np.ndarray]:
        objective_value, gradient = reporting_problem.value_and_grad(parameters)
        diagnostic_values = diagnostic_program(parameters)
        published = jax.device_get(
            (parameters, objective_value, gradient, diagnostic_values)
        )
        values = np.asarray(published[3], dtype=np.float64)
        return _state_values(
            prefix,
            parameters=np.asarray(published[0], dtype=np.float64),
            objective=float(published[1]),
            gradient=np.asarray(published[2], dtype=np.float64),
            squared_flux=float(values[0]),
            length_penalty=float(values[1]),
            distance_penalty=float(values[2]),
            coil_lengths=values[4:],
        )

    initial_values = state("initial", initial_parameters)
    direction = jax.device_put(arrays["taylor_direction"], device)
    initial_scaled, initial_scaled_gradient = scaled_problem.value_and_grad(
        initial_parameters
    )
    directional_derivative = jnp.vdot(initial_scaled_gradient, direction)
    epsilons = jax.device_put(
        np.asarray(_TAYLOR_EPSILONS),
        device,
    )

    def taylor_error(epsilon: jax.Array) -> jax.Array:
        plus = scaled_problem.objective(initial_parameters + epsilon * direction)
        minus = scaled_problem.objective(initial_parameters - epsilon * direction)
        return (plus - minus) / (epsilon + epsilon) - directional_derivative

    taylor_errors_device = jax.vmap(taylor_error)(epsilons)
    result = serial_solve_jax(
        scaled_problem,
        driver=Driver.SIMSOPT_LBFGSB,
        max_steps=_configuration_int(bundle, "max_steps"),
        maxcor=min(_configuration_int(bundle, "max_steps"), 400),
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
        require_success=False,
    )
    final_values = state("final", scaled_problem.x)
    taylor_errors = np.asarray(
        jax.device_get(taylor_errors_device),
        dtype=np.float64,
    )
    success = bool(
        np.isfinite(final_values["final:objective"])
        and final_values["final:objective"] < initial_values["initial:objective"]
        and np.all(np.isfinite(final_values["final:objective_gradient"]))
    )
    platform = "cpu" if device is None else device.platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver=Driver.SIMSOPT_LBFGSB.value,
        normalized_status="converged" if success else "failed",
        raw_status=str(result.status),
        success=success,
        nit=result.nit,
        nfev=result.nfev,
        njev=result.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **final_values,
            "taylor:errors": taylor_errors,
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact native or JAX finite-build workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
