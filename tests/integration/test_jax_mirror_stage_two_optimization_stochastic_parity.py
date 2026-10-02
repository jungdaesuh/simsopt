"""Exact parity for ``stage_two_optimization_stochastic.py``.

Both lanes are built live at ``bounded`` scale from one shared input: the
geometry and the host-materialized training and out-of-sample perturbations of
the shipped mirror, the starting coil DOFs and a Taylor direction.  The native
lane is SIMSOPT's own stochastic objective of
``examples/2_Intermediate/stage_two_optimization_stochastic.py`` under SciPy
L-BFGS-B; the JAX lane is the mirror's ``make_stochastic_stage_two_objective``
under ``solve_stochastic_stage_two`` with the same SciPy L-BFGS-B policy.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import (
    ArclengthVariation,
    Curve,
    CurveCurveDistance,
    CurveLength,
    CurvePerturbed,
    GaussianSampler,
    LpCurveCurvature,
    MeanSquaredCurvature,
    PerturbationSample,
    RotatedCurve,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_contracts.optimization_endpoint import (
    TerminalStatus,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core.biotsavart import biot_savart_B
from simsopt_jax.core.curve_kernels import kappa_pure
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral_from_B
from simsopt_jax.examples import (
    StochasticStageTwoConfiguration,
    materialize_stochastic_coil_perturbations,
    stochastic_stage_two_configuration,
)
from simsopt_jax.examples.stochastic_stage_two import solve_stochastic_stage_two
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StochasticCoilPerturbations,
    make_stochastic_stage_two_objective,
    stage_two_coil_geometry,
    stage_two_geometric_penalty,
    stochastic_flux_mean_from_geometry,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import TraceableScalarProblem
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "test_files"
    / "input.LandremanPaul2021_QA"
)
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
INITIAL_OBSERVABLES = (
    "parameters",
    "objective",
    "objective_gradient",
    "training_flux",
    "nominal_flux",
    "out_of_sample_flux",
    "total_curve_length",
    "shortest_curve_distance",
    "maximum_curvature",
    "maximum_mean_squared_curvature",
    "maximum_arclength_variation",
)


@dataclass(frozen=True)
class _SharedInput:
    """The one construction both lanes consume."""

    configuration: StochasticStageTwoConfiguration
    training_gamma: np.ndarray
    training_gammadash: np.ndarray
    out_of_sample_gamma: np.ndarray
    out_of_sample_gammadash: np.ndarray
    initial_parameters: np.ndarray
    taylor_direction: np.ndarray


@dataclass(frozen=True)
class _Lane:
    """One lane's published values and the label its optimizer earns."""

    values: dict[str, np.ndarray]
    terminal: TerminalStatus


def _geometry(
    configuration: StochasticStageTwoConfiguration,
) -> tuple[SurfaceRZFourier, list[Curve], list[Coil]]:
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="full torus",
        nphi=configuration.surface_nphi,
        ntheta=configuration.surface_ntheta,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_curves(
        configuration.num_base_curves,
        surface.nfp,
        stellsym=True,
        R0=configuration.major_radius,
        R1=configuration.minor_radius,
        order=configuration.curve_order,
        numquadpoints=configuration.curve_quadrature,
    )
    base_currents = [Current(configuration.initial_current) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


def _sampler(
    configuration: StochasticStageTwoConfiguration, base_curves: list[Curve]
) -> GaussianSampler:
    return GaussianSampler(
        base_curves[0].quadpoints,
        sigma=configuration.perturbation_sigma,
        length_scale=configuration.perturbation_length_scale,
        n_derivs=1,
    )


def _shared_input(configuration: StochasticStageTwoConfiguration) -> _SharedInput:
    """Materialize the mirror's source-ordered perturbations and starting DOFs."""
    _surface, base_curves, coils = _geometry(configuration)
    source_indices = tuple(index % len(base_curves) for index in range(len(coils)))
    rotations = []
    for coil, source_index in zip(coils, source_indices, strict=True):
        curve = coil.curve
        if curve is base_curves[source_index]:
            rotations.append(np.eye(3, dtype=np.float64))
        else:
            assert isinstance(curve, RotatedCurve)
            assert curve.curve is base_curves[source_index]
            rotations.append(np.asarray(curve.rotmat, dtype=np.float64))
    sampler = _sampler(configuration, base_curves)

    def samples(count: int, seed: int):
        return materialize_stochastic_coil_perturbations(
            sampler,
            source_indices=source_indices,
            rotations=np.stack(rotations),
            base_curve_count=len(base_curves),
            sample_count=count,
            seed=seed,
        )

    training = samples(configuration.training_sample_count, configuration.training_seed)
    out_of_sample = samples(
        configuration.out_of_sample_count, configuration.out_of_sample_seed
    )
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    return _SharedInput(
        configuration=configuration,
        training_gamma=training.gamma,
        training_gammadash=training.gammadash,
        out_of_sample_gamma=out_of_sample.gamma,
        out_of_sample_gammadash=out_of_sample.gammadash,
        initial_parameters=initial_parameters,
        taylor_direction=np.random.RandomState(1).uniform(
            size=initial_parameters.shape
        ),
    )


def _values(prefix: str, **observables: object) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:{name}": np.asarray(value, dtype=np.float64)
        for name, value in observables.items()
    }


def _terminal(
    optimizer_success: bool,
    optimizer_status: int,
    optimizer_iterations: int,
    max_iterations: int,
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
) -> TerminalStatus:
    """The SciPy L-BFGS-B stage's label, from the shared endpoint contract."""
    scientific_predicate = bool(
        np.isfinite(final["final:objective"])
        and final["final:objective"] < initial["initial:objective"]
        and np.all(np.isfinite(final["final:objective_gradient"]))
    )
    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(
            certify_optimization_endpoint(
                status_convention="scipy-lbfgsb",
                provider_success=optimizer_success,
                provider_status=optimizer_status,
                iterations=optimizer_iterations,
                max_iterations=max_iterations,
                initial_gradient_inf_norm=float(
                    np.max(np.abs(initial["initial:objective_gradient"]))
                ),
                final_gradient_inf_norm=float(
                    np.max(np.abs(final["final:objective_gradient"]))
                ),
                parameters_finite=bool(np.all(np.isfinite(final["final:parameters"]))),
                observables_finite=bool(np.isfinite(final["final:objective"])),
                inner_success=True,
            ).stopping_reason,
        ),
    )


def _native(shared: _SharedInput) -> _Lane:
    configuration = shared.configuration
    surface, base_curves, coils = _geometry(configuration)
    nominal_flux = SquaredFlux(surface, BiotSavart(coils))
    sampler = _sampler(configuration, base_curves)

    def sampled_fluxes(gamma: np.ndarray, gammadash: np.ndarray):
        objectives = []
        for gamma_sample, gammadash_sample in zip(gamma, gammadash, strict=True):
            perturbed_coils = [
                Coil(
                    CurvePerturbed(
                        coil.curve,
                        PerturbationSample(
                            sampler,
                            sample=[gamma_perturbation, gammadash_perturbation],
                        ),
                    ),
                    coil.current,
                )
                for coil, gamma_perturbation, gammadash_perturbation in zip(
                    coils, gamma_sample, gammadash_sample, strict=True
                )
            ]
            objectives.append(SquaredFlux(surface, BiotSavart(perturbed_coils)))
        return objectives

    training_fluxes = sampled_fluxes(shared.training_gamma, shared.training_gammadash)
    out_of_sample_fluxes = sampled_fluxes(
        shared.out_of_sample_gamma, shared.out_of_sample_gammadash
    )
    training_flux = sum(training_fluxes) * (1.0 / len(training_fluxes))
    lengths = [CurveLength(curve) for curve in base_curves]
    curve_curve = CurveCurveDistance(
        [coil.curve for coil in coils],
        configuration.curve_curve_threshold,
        num_basecurves=configuration.num_base_curves,
    )
    curvatures = [
        LpCurveCurvature(curve, 2, configuration.curvature_threshold)
        for curve in base_curves
    ]
    mean_squared_curvatures = [MeanSquaredCurvature(curve) for curve in base_curves]
    arclength_variations = [ArclengthVariation(curve) for curve in base_curves]
    objective = (
        training_flux
        + configuration.length_weight * sum(lengths)
        + configuration.curve_curve_weight * curve_curve
        + configuration.curvature_weight * sum(curvatures)
        + configuration.mean_squared_curvature_weight
        * sum(
            QuadraticPenalty(
                value, configuration.mean_squared_curvature_threshold, "max"
            )
            for value in mean_squared_curvatures
        )
        + configuration.arclength_variation_weight * sum(arclength_variations)
    )

    def mean_flux(current_objectives) -> float:
        return float(
            sum(float(current.J()) for current in current_objectives)
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

    initial_parameters = shared.initial_parameters
    initial_values = state("initial", initial_parameters)
    direction = shared.taylor_direction
    directional_derivative = float(
        np.vdot(initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in TAYLOR_EPSILONS:
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
            "maxiter": configuration.max_steps,
            "maxcor": configuration.lbfgs_history_size,
        },
        tol=configuration.rtol,
    )
    final_values = state("final", np.asarray(optimizer.x, dtype=np.float64))
    return _Lane(
        values={
            **initial_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
        terminal=_terminal(
            bool(optimizer.success),
            int(optimizer.status),
            int(optimizer.nit),
            configuration.max_steps,
            initial_values,
            final_values,
        ),
    )


def _jax(shared: _SharedInput) -> _Lane:
    configuration = shared.configuration
    surface, _base_curves, coils = _geometry(configuration)
    field = BiotSavartJAX(coils)
    flux_spec = SquaredFluxJAX(surface, field).fixed_surface_flux_spec()
    extraction = field.coil_dof_extraction_spec()
    device = get_runtime_jax_device()
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)), device
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)), device
    )
    training = StochasticCoilPerturbations(
        gamma=jax.device_put(shared.training_gamma, device),
        gammadash=jax.device_put(shared.training_gammadash, device),
    )
    out_of_sample = StochasticCoilPerturbations(
        gamma=jax.device_put(shared.out_of_sample_gamma, device),
        gammadash=jax.device_put(shared.out_of_sample_gammadash, device),
    )
    config = StageTwoObjectiveConfig(
        num_base_curves=configuration.num_base_curves,
        length_weight=configuration.length_weight,
        curve_curve_minimum_distance=configuration.curve_curve_threshold,
        curve_curve_weight=configuration.curve_curve_weight,
        curvature_threshold=configuration.curvature_threshold,
        curvature_weight=configuration.curvature_weight,
        mean_squared_curvature_threshold=configuration.mean_squared_curvature_threshold,
        mean_squared_curvature_weight=configuration.mean_squared_curvature_weight,
        arclength_variation_weight=configuration.arclength_variation_weight,
    )
    objective = make_stochastic_stage_two_objective(
        field, flux_spec, training, surface_gamma, surface_normal, config
    )
    initial_parameters = jax.device_put(shared.initial_parameters, device)
    direction = jax.device_put(shared.taylor_direction, device)
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_parameters)
    initial_objective, initial_gradient = problem.value_and_grad(initial_parameters)
    # The route the shipped mirror runs: SciPy L-BFGS-B over the device
    # objective at the native example's policy.
    optimizer = solve_stochastic_stage_two(
        problem,
        driver=Driver.SCIPY_LBFGSB,
        max_steps=configuration.max_steps,
        maxcor=configuration.lbfgs_history_size,
        tol=configuration.rtol,
    )
    final_parameters = problem.x
    final_objective, final_gradient = problem.value_and_grad(final_parameters)

    def diagnostics(parameters):
        gamma, gammadash, gammadashdash, currents = stage_two_coil_geometry(
            extraction, parameters
        )
        training_value = stochastic_flux_mean_from_geometry(
            gamma, gammadash, currents, flux_spec, training
        )
        out_of_sample_value = stochastic_flux_mean_from_geometry(
            gamma, gammadash, currents, flux_spec, out_of_sample
        )
        nominal_value = fixed_surface_flux_integral_from_B(
            biot_savart_B(flux_spec.points, gamma, gammadash, currents), flux_spec
        )
        base_speed = jnp.linalg.norm(gammadash[: config.num_base_curves], axis=2)
        base_kappa = jax.vmap(kappa_pure)(
            gammadash[: config.num_base_curves],
            gammadashdash[: config.num_base_curves],
        )
        mean_squared_curvature = jnp.sum(
            base_kappa * base_kappa * base_speed, axis=1
        ) / jnp.sum(base_speed, axis=1)
        pairs = tuple(
            (index, base_index)
            for index in range(int(gamma.shape[0]))
            for base_index in range(min(index, config.num_base_curves))
        )
        pair_distances = jax.lax.map(
            lambda pair: jnp.min(
                jnp.linalg.norm(
                    gamma[pair[0], :, None, :] - gamma[pair[1], None, :, :], axis=2
                )
            ),
            jnp.asarray(pairs, dtype=jnp.int32),
        )
        geometric_value = stage_two_geometric_penalty(
            gamma, gammadash, gammadashdash, surface_gamma, surface_normal, config
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

    diagnostic_states = jax.jit(jax.vmap(diagnostics))(
        jnp.stack((initial_parameters, final_parameters))
    )
    directional_derivative = jnp.vdot(initial_gradient, direction)
    taylor_errors = jax.jit(
        jax.vmap(
            lambda epsilon: (
                (
                    objective(initial_parameters + epsilon * direction)
                    - objective(initial_parameters - epsilon * direction)
                )
                / (2.0 * epsilon)
                - directional_derivative
            )
        )
    )(jax.device_put(np.asarray(TAYLOR_EPSILONS, dtype=np.float64), device))
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

    def state_values(prefix, parameters, objective_value, gradient, values):
        return _values(
            prefix,
            parameters=parameters,
            objective=float(objective_value),
            objective_gradient=gradient,
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
    return _Lane(
        values={
            **initial_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors_host, dtype=np.float64),
        },
        terminal=_terminal(
            bool(optimizer.success),
            int(optimizer.status),
            int(optimizer.nit),
            configuration.max_steps,
            initial_values,
            final_values,
        ),
    )


def test_exact_stochastic_stage_two_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _shared_input(stochastic_stage_two_configuration("bounded"))

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = _native(shared)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane = _jax(shared)

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the
    # scientific predicate, because a false predicate is labelled ``failed``.
    for lane in (native, jax_lane):
        assert lane.terminal.normalized_status == "budget_exhausted"
        assert lane.terminal.success is False

    for observable in INITIAL_OBSERVABLES:
        np.testing.assert_allclose(
            jax_lane.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=3.0e-8,
            atol=3.0e-10,
        )

    for lane in (native, jax_lane):
        values = lane.values
        assert values["final:objective"] < values["initial:objective"]
        assert np.all(np.isfinite(values["final:parameters"]))
        assert np.all(np.isfinite(values["final:objective_gradient"]))
        assert np.isfinite(values["final:training_flux"])
        assert np.isfinite(values["final:nominal_flux"])
        assert np.isfinite(values["final:out_of_sample_flux"])

    assert float(jax_lane.values["final:objective"]) <= (
        1.10 * float(native.values["final:objective"]) + 1.0e-9
    )

    for lane in (native, jax_lane):
        taylor_errors = np.abs(lane.values["taylor:errors"][:3])
        assert taylor_errors[1] <= 3.0e-2 * taylor_errors[0]
        assert taylor_errors[2] <= 3.0e-2 * taylor_errors[1]
        assert taylor_errors[2] <= 1.0e-4
