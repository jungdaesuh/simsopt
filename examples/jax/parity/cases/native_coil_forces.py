"""Exact matched workflow for ``3_Advanced/coil_forces.py``."""

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
    "construct_landreman_paul_qa_surface_and_regularized_coils",
    "evaluate_flux_engineering_force_and_vacuum_energy_terms",
    "perform_directional_taylor_test",
    "optimize_first_force_stage",
    "reduce_length_penalty_and_optimize_second_stage",
    "evaluate_final_force_energy_flux_geometry_and_gradient",
)
_TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)


def _configuration(scale: ExecutionScale) -> dict[str, object]:
    native = scale == "native_default"
    return {
        "surface_resolution": 32 if native else 4,
        "curve_order": 5 if native else 2,
        "curve_quadrature": 75 if native else 8,
        "num_base_curves": 3,
        "major_radius": 1.0,
        "minor_radius": 0.5,
        "initial_current": 1.0e5,
        "first_length_weight": 1.0e-3,
        "second_length_weight": 1.0e-4,
        "length_target": 17.4,
        "curve_curve_threshold": 0.1,
        "curve_curve_weight": 1000.0,
        "curve_surface_threshold": 0.3,
        "curve_surface_weight": 10.0,
        "curvature_threshold": 5.0,
        "curvature_weight": 1.0e-6,
        "mean_squared_curvature_threshold": 5.0,
        "mean_squared_curvature_weight": 1.0e-6,
        "force_weight": 1.0e-2,
        "force_power": 4.0,
        "force_threshold": 0.0,
        "vacuum_energy_weight": 1.0e-4,
        "regularization": 0.05**2 / np.sqrt(np.e),
        "max_steps": 400 if native else 3,
        "rtol": 1.0e-15,
        "atol": 1.0e-8,
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
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves

    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
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
        use_jax_curve=False,
    )
    base_currents = [
        Current(_mapping_float(configuration, "initial_current")) for _ in base_curves
    ]
    base_currents[0].fix_all()
    regularizations = [
        _mapping_float(configuration, "regularization") for _ in base_curves
    ]
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        regularizations,
    )
    return surface, base_curves, coils


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze source-equivalent coil parameters and Taylor direction."""
    from simsopt.field import BiotSavart

    configuration = _configuration(scale)
    _, _, coils = _build_geometry(configuration)
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    return create_input_bundle(
        root,
        case_id="native-coil-forces",
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
            "surface_dofs": np.asarray(surface.local_full_x).tolist(),
            "surface_gamma": np.asarray(surface.gamma()).tolist(),
            "surface_normal": np.asarray(surface.normal()).tolist(),
            "base_curve_dofs": [
                np.asarray(curve.local_full_x).tolist() for curve in base_curves
            ],
            "base_curve_quadpoints": [
                np.asarray(curve.quadpoints).tolist() for curve in base_curves
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
    force_objective: float,
    vacuum_energy: float,
    total_curve_length: float,
    force_weight: float,
    vacuum_energy_weight: float,
) -> dict[str, np.ndarray]:
    force_and_energy = (
        force_weight * force_objective + vacuum_energy_weight * vacuum_energy
    )
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(gradient, dtype=np.float64),
        f"{prefix}:squared_flux": np.asarray(squared_flux, dtype=np.float64),
        f"{prefix}:geometric_penalty": np.asarray(
            objective - squared_flux - force_and_energy,
            dtype=np.float64,
        ),
        f"{prefix}:force_objective": np.asarray(
            force_objective,
            dtype=np.float64,
        ),
        f"{prefix}:vacuum_energy": np.asarray(vacuum_energy, dtype=np.float64),
        f"{prefix}:total_curve_length": np.asarray(
            total_curve_length,
            dtype=np.float64,
        ),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart
    from simsopt.field.force import B2Energy, LpCurveForce
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
        LpCurveCurvature,
        MeanSquaredCurvature,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    surface, base_curves, coils = _build_geometry(bundle.configuration)
    fingerprint = _fingerprint(bundle, arrays, surface, base_curves)
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
    force = LpCurveForce(
        coils[: _configuration_int(bundle, "num_base_curves")],
        coils,
        p=_configuration_float(bundle, "force_power"),
        threshold=_configuration_float(bundle, "force_threshold"),
    )
    vacuum_energy = B2Energy(coils)

    def objective(length_weight: float):
        return (
            flux
            + length_weight
            * QuadraticPenalty(
                sum(lengths),
                _configuration_float(bundle, "length_target"),
                "max",
            )
            + _configuration_float(bundle, "curve_curve_weight") * curve_curve
            + _configuration_float(bundle, "curve_surface_weight") * curve_surface
            + _configuration_float(bundle, "curvature_weight") * sum(curvatures)
            + _configuration_float(bundle, "mean_squared_curvature_weight")
            * sum(mean_squared_penalties)
            + _configuration_float(bundle, "force_weight") * force
            + _configuration_float(bundle, "vacuum_energy_weight") * vacuum_energy
        )

    def state(prefix: str, parameters: np.ndarray, current_objective):
        current_objective.x = parameters
        return _state_values(
            prefix,
            parameters=parameters,
            objective=float(current_objective.J()),
            gradient=np.asarray(current_objective.dJ(), dtype=np.float64),
            squared_flux=float(flux.J()),
            force_objective=float(force.J()),
            vacuum_energy=float(vacuum_energy.J()),
            total_curve_length=float(sum(length.J() for length in lengths)),
            force_weight=_configuration_float(bundle, "force_weight"),
            vacuum_energy_weight=_configuration_float(
                bundle,
                "vacuum_energy_weight",
            ),
        )

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
                "maxcor": min(_configuration_int(bundle, "max_steps"), 300),
            },
            tol=_configuration_float(bundle, "rtol"),
        )

    initial_parameters = arrays["initial_parameters"]
    first_objective = objective(_configuration_float(bundle, "first_length_weight"))
    initial_values = state("initial", initial_parameters, first_objective)
    direction = arrays["taylor_direction"]
    directional_derivative = float(
        np.vdot(initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7):
        first_objective.x = initial_parameters + epsilon * direction
        plus = float(first_objective.J())
        first_objective.x = initial_parameters - epsilon * direction
        minus = float(first_objective.J())
        taylor_errors.append((plus - minus) / (2 * epsilon) - directional_derivative)
    first_result = minimize_objective(first_objective, initial_parameters)
    first_parameters = np.asarray(first_result.x, dtype=np.float64)
    first_values = state("first", first_parameters, first_objective)
    second_objective = objective(_configuration_float(bundle, "second_length_weight"))
    second_result = minimize_objective(second_objective, first_parameters)
    final_parameters = np.asarray(second_result.x, dtype=np.float64)
    final_values = state("final", final_parameters, second_objective)
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
        driver="scipy_lbfgsb_two_stage_force",
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
    import jax
    import jax.numpy as jnp

    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.objectives import (
        StageTwoObjectiveConfig,
        stage_two_coil_geometry,
    )
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax.solve.serial import (
        TraceableArrayFunction,
        TraceableScalarProblem,
        serial_solve_jax,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives import (
        ForceStageTwoConfig,
        force_stage_two_diagnostics,
        make_force_stage_two_objective,
    )
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    surface, base_curves, coils = _build_geometry(bundle.configuration)
    fingerprint = _fingerprint(bundle, arrays, surface, base_curves)
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    flux_objective = flux.traceable_objective()
    device = get_runtime_jax_device()

    stage_config = {
        "num_base_curves": _configuration_int(bundle, "num_base_curves"),
        "length_target": _configuration_float(bundle, "length_target"),
        "length_target_mode": "max",
        "curve_curve_minimum_distance": _configuration_float(
            bundle,
            "curve_curve_threshold",
        ),
        "curve_curve_weight": _configuration_float(
            bundle,
            "curve_curve_weight",
        ),
        "curve_surface_minimum_distance": _configuration_float(
            bundle,
            "curve_surface_threshold",
        ),
        "curve_surface_weight": _configuration_float(
            bundle,
            "curve_surface_weight",
        ),
        "curvature_threshold": _configuration_float(
            bundle,
            "curvature_threshold",
        ),
        "curvature_weight": _configuration_float(bundle, "curvature_weight"),
        "mean_squared_curvature_threshold": _configuration_float(
            bundle,
            "mean_squared_curvature_threshold",
        ),
        "mean_squared_curvature_weight": _configuration_float(
            bundle,
            "mean_squared_curvature_weight",
        ),
    }
    force_config = ForceStageTwoConfig(
        num_force_coils=_configuration_int(bundle, "num_base_curves"),
        force_weight=_configuration_float(bundle, "force_weight"),
        vacuum_energy_weight=_configuration_float(
            bundle,
            "vacuum_energy_weight",
        ),
        force_power=_configuration_float(bundle, "force_power"),
        force_threshold=_configuration_float(bundle, "force_threshold"),
    )
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    target_quadpoints = jax.device_put(
        np.stack(
            [np.asarray(curve.quadpoints, dtype=np.float64) for curve in base_curves]
        ),
        device,
    )
    regularizations = jax.device_put(
        np.full(
            len(coils),
            _configuration_float(bundle, "regularization"),
            dtype=np.float64,
        ),
        device,
    )
    diagnostics = force_stage_two_diagnostics(
        field,
        target_quadpoints,
        regularizations,
        force_config,
    )
    extraction = field.coil_dof_extraction_spec()

    def objective(length_weight: float):
        return make_force_stage_two_objective(
            field,
            flux_objective,
            surface_gamma,
            surface_normal,
            target_quadpoints,
            regularizations,
            StageTwoObjectiveConfig(
                **stage_config,
                length_weight=length_weight,
            ),
            force_config,
        )

    num_base_curves = _configuration_int(bundle, "num_base_curves")

    def state_diagnostics(parameters: jax.Array) -> jax.Array:
        force_objective, _, vacuum_energy = diagnostics(parameters)
        gamma, gammadash, _, _ = stage_two_coil_geometry(extraction, parameters)
        base_gammadash = jax.lax.slice_in_dim(
            gammadash,
            0,
            num_base_curves,
            axis=0,
        )
        total_length = jnp.sum(
            jnp.mean(
                jnp.linalg.norm(base_gammadash, axis=-1),
                axis=1,
            )
        )
        return jnp.stack(
            (
                flux_objective(parameters),
                force_objective,
                vacuum_energy,
                total_length,
            )
        )

    def state(
        prefix: str,
        parameters: jax.Array,
        problem: TraceableScalarProblem,
        state_program: TraceableArrayFunction,
    ):
        objective_value, gradient = problem.value_and_grad(parameters)
        published = jax.device_get(
            (
                parameters,
                objective_value,
                gradient,
                state_program(parameters),
            )
        )
        scalar_values = np.asarray(published[3], dtype=np.float64)
        return _state_values(
            prefix,
            parameters=np.asarray(published[0], dtype=np.float64),
            objective=float(published[1]),
            gradient=np.asarray(published[2], dtype=np.float64),
            squared_flux=float(scalar_values[0]),
            force_objective=float(scalar_values[1]),
            vacuum_energy=float(scalar_values[2]),
            total_curve_length=float(scalar_values[3]),
            force_weight=_configuration_float(bundle, "force_weight"),
            vacuum_energy_weight=_configuration_float(
                bundle,
                "vacuum_energy_weight",
            ),
        )

    def solve(problem: TraceableScalarProblem):
        result = serial_solve_jax(
            problem,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=_configuration_int(bundle, "max_steps"),
            maxcor=min(_configuration_int(bundle, "max_steps"), 300),
            rtol=_configuration_float(bundle, "rtol"),
            atol=_configuration_float(bundle, "atol"),
            require_success=False,
        )
        return result, problem.x

    initial_parameters = jax.device_put(arrays["initial_parameters"], device)
    direction = jax.device_put(arrays["taylor_direction"], device)
    first_objective = objective(_configuration_float(bundle, "first_length_weight"))
    first_problem = TraceableScalarProblem(first_objective, initial_parameters)
    state_program = TraceableArrayFunction(
        state_diagnostics,
        initial_parameters,
    )
    initial_values = state(
        "initial",
        initial_parameters,
        first_problem,
        state_program,
    )
    initial_value, initial_gradient = first_problem.value_and_grad(initial_parameters)
    directional_derivative = jnp.vdot(initial_gradient, direction)
    epsilons = jax.device_put(
        np.asarray(_TAYLOR_EPSILONS),
        device,
    )

    def taylor_error(epsilon: jax.Array) -> jax.Array:
        plus = first_problem.objective(initial_parameters + epsilon * direction)
        minus = first_problem.objective(initial_parameters - epsilon * direction)
        return (plus - minus) / (epsilon + epsilon) - directional_derivative

    taylor_errors_device = jax.vmap(taylor_error)(epsilons)
    first_result, first_parameters = solve(first_problem)
    first_values = state("first", first_parameters, first_problem, state_program)
    second_objective = objective(_configuration_float(bundle, "second_length_weight"))
    second_problem = TraceableScalarProblem(second_objective, first_parameters)
    second_result, final_parameters = solve(second_problem)
    final_values = state(
        "final",
        final_parameters,
        second_problem,
        state_program,
    )
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
        raw_status=f"{first_result.status},{second_result.status}",
        success=success,
        nit=first_result.nit + second_result.nit,
        nfev=first_result.nfev + second_result.nfev,
        njev=first_result.njev + second_result.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **first_values,
            **final_values,
            "taylor:errors": taylor_errors,
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact native or JAX force-optimization workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
