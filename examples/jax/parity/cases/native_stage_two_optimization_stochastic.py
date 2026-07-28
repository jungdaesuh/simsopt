"""Matched workflow for ``stage_two_optimization_stochastic.py``."""

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
    "construct_full_torus_landreman_paul_qa_surface",
    "construct_four_base_coils_currents_and_stellarator_symmetries",
    "materialize_systematic_and_statistical_training_samples",
    "evaluate_stochastic_flux_and_nominal_geometry_penalties",
    "perform_directional_taylor_test",
    "optimize_source_stochastic_objective",
    "evaluate_training_nominal_and_out_of_sample_flux",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "surface_nphi": 64 if native_scale else 4,
        "surface_ntheta": 16 if native_scale else 4,
        "curve_order": 24 if native_scale else 2,
        "curve_quadrature": 360 if native_scale else 16,
        "num_base_curves": 4,
        "major_radius": 1.0,
        "minor_radius": 0.5,
        "initial_current": 1.0e5,
        "length_weight": 1.0e-6,
        "curve_curve_threshold": 0.1,
        "curve_curve_weight": 10.0,
        "curvature_threshold": 5.0,
        "curvature_weight": 1.0e-6,
        "mean_squared_curvature_threshold": 5.0,
        "mean_squared_curvature_weight": 1.0e-6,
        "arclength_variation_weight": 1.0e-2,
        "perturbation_sigma": 1.0e-3,
        "perturbation_length_scale": 0.5,
        "training_sample_count": 16 if native_scale else 2,
        "out_of_sample_count": 256 if native_scale else 4,
        "training_seed": 0,
        "out_of_sample_seed": 1,
        "max_steps": 400 if native_scale else 20,
        "rtol": 1.0e-15,
        "atol": 1.0e-8,
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
        range="full torus",
        nphi=_mapping_int(configuration, "surface_nphi"),
        ntheta=_mapping_int(configuration, "surface_ntheta"),
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


def _symmetry_layout(coils, base_curves) -> tuple[tuple[int, ...], np.ndarray]:
    from simsopt.geo import RotatedCurve

    source_indices = tuple(
        final_index % len(base_curves) for final_index in range(len(coils))
    )
    rotations: list[np.ndarray] = []
    for coil, source_index in zip(coils, source_indices, strict=True):
        curve = coil.curve
        if curve is base_curves[source_index]:
            rotations.append(np.eye(3, dtype=np.float64))
        else:
            if not isinstance(curve, RotatedCurve) or (
                curve.curve is not base_curves[source_index]
            ):
                raise TypeError("symmetry-expanded coils must use RotatedCurve")
            rotations.append(np.asarray(curve.rotmat, dtype=np.float64))
    return source_indices, np.stack(rotations)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize source-ordered perturbations and shared starting DOFs."""
    from simsopt.field import BiotSavart
    from simsopt.geo import GaussianSampler
    from simsopt_jax.examples import materialize_stochastic_coil_perturbations

    configuration = _scale_configuration(scale)
    _surface, base_curves, coils = _build_geometry(configuration)
    source_indices, rotations = _symmetry_layout(coils, base_curves)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=_mapping_float(configuration, "perturbation_sigma"),
        length_scale=_mapping_float(configuration, "perturbation_length_scale"),
        n_derivs=1,
    )
    training = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=_mapping_int(configuration, "training_sample_count"),
        seed=_mapping_int(configuration, "training_seed"),
    )
    out_of_sample = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=_mapping_int(configuration, "out_of_sample_count"),
        seed=_mapping_int(configuration, "out_of_sample_seed"),
    )
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    return create_input_bundle(
        root,
        case_id="native-stage-two-optimization-stochastic",
        random_seed=0,
        arrays={
            "initial_parameters": initial_parameters,
            "taylor_direction": np.random.RandomState(1).uniform(
                size=initial_parameters.shape
            ),
            "training_gamma": training.gamma,
            "training_gammadash": training.gammadash,
            "out_of_sample_gamma": out_of_sample.gamma,
            "out_of_sample_gammadash": out_of_sample.gammadash,
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
    training_flux: float,
    nominal_flux: float,
    out_of_sample_flux: float,
    total_curve_length: float,
    shortest_curve_distance: float,
    maximum_curvature: float,
    maximum_mean_squared_curvature: float,
    maximum_arclength_variation: float,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(
            objective_gradient,
            dtype=np.float64,
        ),
        f"{prefix}:training_flux": np.asarray(training_flux, dtype=np.float64),
        f"{prefix}:nominal_flux": np.asarray(nominal_flux, dtype=np.float64),
        f"{prefix}:out_of_sample_flux": np.asarray(
            out_of_sample_flux,
            dtype=np.float64,
        ),
        f"{prefix}:total_curve_length": np.asarray(
            total_curve_length,
            dtype=np.float64,
        ),
        f"{prefix}:shortest_curve_distance": np.asarray(
            shortest_curve_distance,
            dtype=np.float64,
        ),
        f"{prefix}:maximum_curvature": np.asarray(
            maximum_curvature,
            dtype=np.float64,
        ),
        f"{prefix}:maximum_mean_squared_curvature": np.asarray(
            maximum_mean_squared_curvature,
            dtype=np.float64,
        ),
        f"{prefix}:maximum_arclength_variation": np.asarray(
            maximum_arclength_variation,
            dtype=np.float64,
        ),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart, Coil
    from simsopt.geo import (
        ArclengthVariation,
        CurveCurveDistance,
        CurveLength,
        CurvePerturbed,
        GaussianSampler,
        LpCurveCurvature,
        MeanSquaredCurvature,
        PerturbationSample,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    surface, base_curves, coils = _build_geometry(dict(bundle.configuration))
    construction_fingerprint = _effective_fingerprint(
        bundle,
        arrays,
        surface,
        base_curves,
    )
    nominal_field = BiotSavart(coils)
    nominal_flux = SquaredFlux(surface, nominal_field)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=_configuration_float(bundle, "perturbation_sigma"),
        length_scale=_configuration_float(bundle, "perturbation_length_scale"),
        n_derivs=1,
    )

    def sampled_fluxes(gamma_name: str, gammadash_name: str):
        objectives = []
        for gamma_sample, gammadash_sample in zip(
            arrays[gamma_name],
            arrays[gammadash_name],
            strict=True,
        ):
            perturbed_coils = [
                Coil(
                    CurvePerturbed(
                        coil.curve,
                        PerturbationSample(
                            sampler,
                            sample=[
                                gamma_perturbation,
                                gammadash_perturbation,
                            ],
                        ),
                    ),
                    coil.current,
                )
                for coil, gamma_perturbation, gammadash_perturbation in zip(
                    coils,
                    gamma_sample,
                    gammadash_sample,
                    strict=True,
                )
            ]
            objectives.append(SquaredFlux(surface, BiotSavart(perturbed_coils)))
        return objectives

    training_fluxes = sampled_fluxes("training_gamma", "training_gammadash")
    out_of_sample_fluxes = sampled_fluxes(
        "out_of_sample_gamma",
        "out_of_sample_gammadash",
    )
    training_flux = sum(training_fluxes) * (1.0 / len(training_fluxes))
    lengths = [CurveLength(curve) for curve in base_curves]
    curve_curve = CurveCurveDistance(
        [coil.curve for coil in coils],
        _configuration_float(bundle, "curve_curve_threshold"),
        num_basecurves=_configuration_int(bundle, "num_base_curves"),
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
    arclength_variations = [ArclengthVariation(curve) for curve in base_curves]
    objective = (
        training_flux
        + _configuration_float(bundle, "length_weight") * sum(lengths)
        + _configuration_float(bundle, "curve_curve_weight") * curve_curve
        + _configuration_float(bundle, "curvature_weight") * sum(curvatures)
        + _configuration_float(bundle, "mean_squared_curvature_weight")
        * sum(
            QuadraticPenalty(
                value,
                _configuration_float(
                    bundle,
                    "mean_squared_curvature_threshold",
                ),
                "max",
            )
            for value in mean_squared_curvatures
        )
        + _configuration_float(bundle, "arclength_variation_weight")
        * sum(arclength_variations)
    )

    def mean_flux(current_objectives) -> float:
        return float(
            sum(
                float(current_objective.J()) for current_objective in current_objectives
            )
            / len(current_objectives)
        )

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        objective.x = parameters
        arclength_ratios = [
            np.max(curve.incremental_arclength())
            / np.min(curve.incremental_arclength())
            - 1.0
            for curve in base_curves
        ]
        return _values(
            prefix,
            parameters=parameters,
            objective=float(objective.J()),
            objective_gradient=np.asarray(objective.dJ(), dtype=np.float64),
            training_flux=mean_flux(training_fluxes),
            nominal_flux=float(nominal_flux.J()),
            out_of_sample_flux=mean_flux(out_of_sample_fluxes),
            total_curve_length=float(sum(length.J() for length in lengths)),
            shortest_curve_distance=float(curve_curve.shortest_distance()),
            maximum_curvature=float(
                max(np.max(curve.kappa()) for curve in base_curves)
            ),
            maximum_mean_squared_curvature=float(
                max(value.J() for value in mean_squared_curvatures)
            ),
            maximum_arclength_variation=float(max(arclength_ratios)),
        )

    initial_parameters = arrays["initial_parameters"]
    initial_values = state("initial", initial_parameters)
    direction = arrays["taylor_direction"]
    directional_derivative = float(
        np.vdot(initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7):
        objective.x = initial_parameters + epsilon * direction
        plus = float(objective.J())
        objective.x = initial_parameters - epsilon * direction
        minus = float(objective.J())
        taylor_errors.append((plus - minus) / (2.0 * epsilon) - directional_derivative)

    def value_and_gradient(parameters: np.ndarray):
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    optimizer = minimize(
        value_and_gradient,
        initial_parameters,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": _configuration_int(bundle, "max_steps"),
            "maxcor": 400,
        },
        tol=1.0e-15,
    )
    final_parameters = np.asarray(optimizer.x, dtype=np.float64)
    final_values = state("final", final_parameters)
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
        driver="scipy_lbfgsb",
        normalized_status="converged" if success else "failed",
        raw_status=str(optimizer.status),
        success=success,
        nit=int(optimizer.nit),
        nfev=int(optimizer.nfev),
        njev=int(optimizer.njev),
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
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.core.biotsavart import biot_savart_B
    from simsopt_jax.core.curve_kernels import kappa_pure
    from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral_from_B
    from simsopt_jax.objectives import (
        StageTwoObjectiveConfig,
        StochasticCoilPerturbations,
        make_stochastic_stage_two_objective,
        stage_two_coil_geometry,
        stage_two_geometric_penalty,
        stochastic_flux_mean_from_geometry,
    )
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    import jax
    import jax.numpy as jnp

    surface, base_curves, coils = _build_geometry(dict(bundle.configuration))
    construction_fingerprint = _effective_fingerprint(
        bundle,
        arrays,
        surface,
        base_curves,
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    flux_spec = flux.fixed_surface_flux_spec()
    extraction = field.coil_dof_extraction_spec()
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    training = StochasticCoilPerturbations(
        gamma=jax.device_put(arrays["training_gamma"], device),
        gammadash=jax.device_put(arrays["training_gammadash"], device),
    )
    out_of_sample = StochasticCoilPerturbations(
        gamma=jax.device_put(arrays["out_of_sample_gamma"], device),
        gammadash=jax.device_put(arrays["out_of_sample_gammadash"], device),
    )
    config = StageTwoObjectiveConfig(
        num_base_curves=_configuration_int(bundle, "num_base_curves"),
        length_weight=_configuration_float(bundle, "length_weight"),
        curve_curve_minimum_distance=_configuration_float(
            bundle,
            "curve_curve_threshold",
        ),
        curve_curve_weight=_configuration_float(bundle, "curve_curve_weight"),
        curvature_threshold=_configuration_float(bundle, "curvature_threshold"),
        curvature_weight=_configuration_float(bundle, "curvature_weight"),
        mean_squared_curvature_threshold=_configuration_float(
            bundle,
            "mean_squared_curvature_threshold",
        ),
        mean_squared_curvature_weight=_configuration_float(
            bundle,
            "mean_squared_curvature_weight",
        ),
        arclength_variation_weight=_configuration_float(
            bundle,
            "arclength_variation_weight",
        ),
    )
    objective = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        training,
        surface_gamma,
        surface_normal,
        config,
    )
    initial_parameters = jax.device_put(arrays["initial_parameters"], device)
    direction = jax.device_put(arrays["taylor_direction"], device)
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_parameters)
    initial_objective, initial_gradient = problem.value_and_grad(initial_parameters)
    optimizer = serial_solve_jax(
        problem,
        driver=Driver.SIMSOPT_BFGS,
        max_steps=_configuration_int(bundle, "max_steps"),
        line_search_max_steps=40,
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
        require_success=False,
    )
    final_parameters = problem.x
    final_objective, final_gradient = problem.value_and_grad(final_parameters)

    def diagnostics(
        extraction_operand,
        parameters,
        flux_spec_operand,
        training_operand,
        out_of_sample_operand,
        surface_gamma_operand,
        surface_normal_operand,
    ):
        gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
            extraction_operand,
            parameters,
        )
        training_value = stochastic_flux_mean_from_geometry(
            gamma,
            gammadash,
            currents,
            flux_spec_operand,
            training_operand,
        )
        out_of_sample_value = stochastic_flux_mean_from_geometry(
            gamma,
            gammadash,
            currents,
            flux_spec_operand,
            out_of_sample_operand,
        )
        nominal_value = fixed_surface_flux_integral_from_B(
            biot_savart_B(
                flux_spec_operand.points,
                gamma,
                gammadash,
                currents,
            ),
            flux_spec_operand,
        )
        base_speed = jnp.linalg.norm(
            gammadash[: config.num_base_curves],
            axis=2,
        )
        base_kappa = jax.vmap(kappa_pure)(
            gammadash[: config.num_base_curves],
            gammadashdash[: config.num_base_curves],
        )
        mean_squared_curvature = jnp.sum(
            base_kappa * base_kappa * base_speed,
            axis=1,
        ) / jnp.sum(base_speed, axis=1)
        pairs = tuple(
            (index, base_index)
            for index in range(int(gamma.shape[0]))
            for base_index in range(min(index, config.num_base_curves))
        )
        pair_indices = jnp.asarray(pairs, dtype=jnp.int32)
        pair_distances = jax.lax.map(
            lambda pair: jnp.min(
                jnp.linalg.norm(
                    gamma[pair[0], :, None, :] - gamma[pair[1], None, :, :],
                    axis=2,
                )
            ),
            pair_indices,
        )
        geometric_value = stage_two_geometric_penalty(
            gamma,
            gammadash,
            gammadashdash,
            surface_gamma_operand,
            surface_normal_operand,
            config,
        )
        return jnp.stack(
            (
                training_value + geometric_value,
                training_value,
                nominal_value,
                out_of_sample_value,
                jnp.sum(jnp.mean(base_speed, axis=1)),
                jnp.min(pair_distances),
                jnp.max(base_kappa),
                jnp.max(mean_squared_curvature),
                jnp.max(
                    jnp.max(base_speed, axis=1) / jnp.min(base_speed, axis=1) - 1.0
                ),
            )
        )

    diagnostics_program = jax.jit(diagnostics)
    parameter_states = jnp.stack((initial_parameters, final_parameters))
    diagnostic_states = jax.vmap(
        lambda parameters: diagnostics_program(
            extraction,
            parameters,
            flux_spec,
            training,
            out_of_sample,
            surface_gamma,
            surface_normal,
        )
    )(parameter_states)
    directional_derivative = jnp.vdot(initial_gradient, direction)
    epsilons = jnp.asarray(
        (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7),
        dtype=initial_parameters.dtype,
    )
    taylor_errors = jax.vmap(
        lambda epsilon: (
            (
                problem.value_and_grad(initial_parameters + epsilon * direction)[0]
                - problem.value_and_grad(initial_parameters - epsilon * direction)[0]
            )
            / (2.0 * epsilon)
            - directional_derivative
        )
    )(epsilons)
    (
        initial_parameters_host,
        initial_objective_host,
        initial_gradient_host,
        final_parameters_host,
        final_objective_host,
        final_gradient_host,
        diagnostic_states_host,
        taylor_errors_host,
    ) = jax.device_get(
        jax.block_until_ready(
            (
                initial_parameters,
                initial_objective,
                initial_gradient,
                final_parameters,
                final_objective,
                final_gradient,
                diagnostic_states,
                taylor_errors,
            )
        )
    )

    def state_values(
        prefix: str,
        parameters: np.ndarray,
        objective_value: float,
        gradient: np.ndarray,
        values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return _values(
            prefix,
            parameters=np.asarray(parameters, dtype=np.float64),
            objective=float(objective_value),
            objective_gradient=np.asarray(gradient, dtype=np.float64),
            training_flux=float(values[1]),
            nominal_flux=float(values[2]),
            out_of_sample_flux=float(values[3]),
            total_curve_length=float(values[4]),
            shortest_curve_distance=float(values[5]),
            maximum_curvature=float(values[6]),
            maximum_mean_squared_curvature=float(values[7]),
            maximum_arclength_variation=float(values[8]),
        )

    initial_values = state_values(
        "initial",
        initial_parameters_host,
        initial_objective_host,
        initial_gradient_host,
        diagnostic_states_host[0],
    )
    final_values = state_values(
        "final",
        final_parameters_host,
        final_objective_host,
        final_gradient_host,
        diagnostic_states_host[1],
    )
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
        driver=optimizer.driver.value,
        normalized_status="converged" if success else "failed",
        raw_status=str(optimizer.status),
        success=success,
        nit=optimizer.nit,
        nfev=optimizer.nfev,
        njev=optimizer.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors_host, dtype=np.float64),
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact native or JAX stochastic Stage-II workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
