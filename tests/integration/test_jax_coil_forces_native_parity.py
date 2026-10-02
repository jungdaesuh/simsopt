"""The ``3_Advanced/coil_forces.py`` mirror against native SIMSOPT.

Both lanes are built live at the reduced (bounded) scale: the native lane is
upstream's two-stage L-BFGS-B over flux, coil regularization, ``LpCurveForce``
and ``B2Energy``; the JAX lane is the adapter force objective
(:func:`simsopt_jax_adapters.objectives.make_force_stage_two_objective`) solved
through :func:`simsopt_jax.examples.solve_scalar_stage`, the route the shipped
mirror takes.  Both start from the same coil set, Taylor direction and
post-Taylor optimizer start.  No reference data is stored.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.force import B2Energy, LpCurveForce
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_contracts.optimization_endpoint import (
    StoppingReason,
    TerminalStatus,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_scalar_stage
from simsopt_jax.objectives import StageTwoObjectiveConfig, stage_two_coil_geometry
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableArrayFunction,
    TraceableParametricScalarProblem,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    ForceStageTwoConfig,
    force_stage_two_diagnostics,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: The reduced scale of the official script.
SURFACE_RESOLUTION = 4
CURVE_ORDER = 2
CURVE_QUADRATURE = 8
NUM_BASE_CURVES = 3
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.5
INITIAL_CURRENT = 1.0e5
FIRST_LENGTH_WEIGHT = 1.0e-3
SECOND_LENGTH_WEIGHT = 1.0e-4
LENGTH_TARGET = 17.4
CURVE_CURVE_THRESHOLD = 0.1
CURVE_CURVE_WEIGHT = 1000.0
CURVE_SURFACE_THRESHOLD = 0.3
CURVE_SURFACE_WEIGHT = 10.0
CURVATURE_THRESHOLD = 5.0
CURVATURE_WEIGHT = 1.0e-6
MEAN_SQUARED_CURVATURE_THRESHOLD = 5.0
MEAN_SQUARED_CURVATURE_WEIGHT = 1.0e-6
FORCE_WEIGHT = 1.0e-2
FORCE_POWER = 4.0
FORCE_THRESHOLD = 0.0
VACUUM_ENERGY_WEIGHT = 1.0e-4
REGULARIZATION = 0.05**2 / np.sqrt(np.e)
MAX_STEPS = 3
TOL = 1.0e-15
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)


def _geometry():
    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=SURFACE_RESOLUTION,
        ntheta=SURFACE_RESOLUTION,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_curves(
        NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=MAJOR_RADIUS,
        R1=MINOR_RADIUS,
        order=CURVE_ORDER,
        numquadpoints=CURVE_QUADRATURE,
        use_jax_curve=False,
    )
    base_currents = [Current(INITIAL_CURRENT) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        [REGULARIZATION for _ in base_curves],
    )
    return surface, base_curves, coils


def _optimizer_start(initial: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """The state the official script minimizes from: its last Taylor evaluation.

    ``examples/3_Advanced/coil_forces.py`` runs its Taylor test through the same
    ``JF`` it optimizes and then re-reads ``dofs = JF.x``, so stage one starts
    from ``dofs0 - 1e-7 * h``; the initial observables and the Taylor test stay
    at ``dofs0``.  Both lanes take the start from this one host expression.
    """
    return initial - TAYLOR_EPSILONS[-1] * direction


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
) -> dict[str, np.ndarray]:
    force_and_energy = FORCE_WEIGHT * force_objective + (
        VACUUM_ENERGY_WEIGHT * vacuum_energy
    )
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(gradient, dtype=np.float64),
        f"{prefix}:squared_flux": np.asarray(squared_flux, dtype=np.float64),
        f"{prefix}:geometric_penalty": np.asarray(
            objective - squared_flux - force_and_energy, dtype=np.float64
        ),
        f"{prefix}:force_objective": np.asarray(force_objective, dtype=np.float64),
        f"{prefix}:vacuum_energy": np.asarray(vacuum_energy, dtype=np.float64),
        f"{prefix}:total_curve_length": np.asarray(
            total_curve_length, dtype=np.float64
        ),
    }


@dataclass(frozen=True)
class _StageStop:
    success: bool
    status: int
    iterations: int


@dataclass(frozen=True)
class _LaneRun:
    values: dict[str, np.ndarray]
    stages: tuple[_StageStop, _StageStop]


def _stage_reason(
    stop: _StageStop, values: dict[str, np.ndarray], start: str, end: str
) -> StoppingReason:
    return certify_optimization_endpoint(
        # Both lanes ran SciPy's own L-BFGS-B, so one status table reads both.
        status_convention="scipy-lbfgsb",
        provider_success=stop.success,
        provider_status=stop.status,
        iterations=stop.iterations,
        max_iterations=MAX_STEPS,
        initial_gradient_inf_norm=float(
            np.max(np.abs(values[f"{start}:objective_gradient"]))
        ),
        final_gradient_inf_norm=float(
            np.max(np.abs(values[f"{end}:objective_gradient"]))
        ),
        parameters_finite=bool(np.all(np.isfinite(values[f"{end}:parameters"]))),
        observables_finite=bool(np.isfinite(values[f"{end}:objective"])),
        inner_success=True,
    ).stopping_reason


def _terminal_status(run: _LaneRun) -> TerminalStatus:
    """The library's fold of both stages and the case's scientific predicate."""
    values = run.values
    scientific_predicate = bool(
        np.isfinite(values["final:objective"])
        and values["final:objective"] < values["initial:objective"]
        and np.all(np.isfinite(values["final:objective_gradient"]))
    )
    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(
            _stage_reason(run.stages[0], values, "initial", "first"),
            _stage_reason(run.stages[1], values, "first", "final"),
        ),
    )


def _run_native(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    lengths = [CurveLength(curve) for curve in base_curves]
    curve_curve = CurveCurveDistance(
        [coil.curve for coil in coils],
        CURVE_CURVE_THRESHOLD,
        num_basecurves=NUM_BASE_CURVES,
    )
    curve_surface = CurveSurfaceDistance(
        [coil.curve for coil in coils], surface, CURVE_SURFACE_THRESHOLD
    )
    curvatures = [
        LpCurveCurvature(curve, 2, CURVATURE_THRESHOLD) for curve in base_curves
    ]
    mean_squared_penalties = [
        QuadraticPenalty(
            MeanSquaredCurvature(curve), MEAN_SQUARED_CURVATURE_THRESHOLD, "max"
        )
        for curve in base_curves
    ]
    force = LpCurveForce(
        coils[:NUM_BASE_CURVES], coils, p=FORCE_POWER, threshold=FORCE_THRESHOLD
    )
    vacuum_energy = B2Energy(coils)

    def objective(length_weight: float) -> Optimizable:
        return (
            flux
            + length_weight * QuadraticPenalty(sum(lengths), LENGTH_TARGET, "max")
            + CURVE_CURVE_WEIGHT * curve_curve
            + CURVE_SURFACE_WEIGHT * curve_surface
            + CURVATURE_WEIGHT * sum(curvatures)
            + MEAN_SQUARED_CURVATURE_WEIGHT * sum(mean_squared_penalties)
            + FORCE_WEIGHT * force
            + VACUUM_ENERGY_WEIGHT * vacuum_energy
        )

    def state(prefix: str, parameters: np.ndarray, current: Optimizable):
        current.x = parameters
        return _state_values(
            prefix,
            parameters=parameters,
            objective=float(current.J()),
            gradient=np.asarray(current.dJ(), dtype=np.float64),
            squared_flux=float(flux.J()),
            force_objective=float(force.J()),
            vacuum_energy=float(vacuum_energy.J()),
            total_curve_length=float(sum(length.J() for length in lengths)),
        )

    def solve(current: Optimizable, start: np.ndarray) -> OptimizeResult:
        def value_and_gradient(parameters: np.ndarray):
            current.x = parameters
            return float(current.J()), np.asarray(current.dJ(), dtype=np.float64)

        return minimize(
            value_and_gradient,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": MAX_STEPS, "maxcor": min(MAX_STEPS, 300)},
            tol=TOL,
        )

    first_objective = objective(FIRST_LENGTH_WEIGHT)
    initial_values = state("initial", initial, first_objective)
    directional_derivative = float(
        np.vdot(initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in TAYLOR_EPSILONS:
        first_objective.x = initial + epsilon * direction
        plus = float(first_objective.J())
        first_objective.x = initial - epsilon * direction
        minus = float(first_objective.J())
        taylor_errors.append((plus - minus) / (2 * epsilon) - directional_derivative)
    first = solve(first_objective, _optimizer_start(initial, direction))
    first_parameters = np.asarray(first.x, dtype=np.float64)
    first_values = state("first", first_parameters, first_objective)
    second_objective = objective(SECOND_LENGTH_WEIGHT)
    second = solve(second_objective, first_parameters)
    final_values = state(
        "final", np.asarray(second.x, dtype=np.float64), second_objective
    )
    return _LaneRun(
        values={
            **initial_values,
            **first_values,
            **final_values,
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
        stages=tuple(
            _StageStop(bool(result.success), int(result.status), int(result.nit))
            for result in (first, second)
        ),
    )


def _run_jax(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    surface, base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    flux_objective = SquaredFluxJAX(surface, field).traceable_objective()
    device = get_runtime_jax_device()

    def put(value: object) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    stage_two_config = StageTwoObjectiveConfig(
        num_base_curves=NUM_BASE_CURVES,
        length_target=LENGTH_TARGET,
        length_target_mode="max",
        curve_curve_minimum_distance=CURVE_CURVE_THRESHOLD,
        curve_curve_weight=CURVE_CURVE_WEIGHT,
        curve_surface_minimum_distance=CURVE_SURFACE_THRESHOLD,
        curve_surface_weight=CURVE_SURFACE_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        mean_squared_curvature_threshold=MEAN_SQUARED_CURVATURE_THRESHOLD,
        mean_squared_curvature_weight=MEAN_SQUARED_CURVATURE_WEIGHT,
    )
    force_config = ForceStageTwoConfig(
        num_force_coils=NUM_BASE_CURVES,
        force_weight=FORCE_WEIGHT,
        vacuum_energy_weight=VACUUM_ENERGY_WEIGHT,
        force_power=FORCE_POWER,
        force_threshold=FORCE_THRESHOLD,
    )
    target_quadpoints = put(np.stack([curve.quadpoints for curve in base_curves]))
    regularizations = put(np.full(len(coils), REGULARIZATION))
    diagnostics = force_stage_two_diagnostics(
        field, target_quadpoints, regularizations, force_config
    )
    extraction = field.coil_dof_extraction_spec()
    objective = make_force_stage_two_objective(
        field,
        flux_objective,
        put(surface.gamma().reshape((-1, 3))),
        put(surface.normal().reshape((-1, 3))),
        target_quadpoints,
        regularizations,
        stage_two_config,
        force_config,
    )
    length_penalty = make_force_stage_two_length_penalty(field, stage_two_config)

    def weighted_objective(parameters: jax.Array, length_weight: jax.Array):
        return objective(parameters) + length_penalty(parameters, length_weight)

    def state_diagnostics(parameters: jax.Array) -> jax.Array:
        force_objective, _, vacuum_energy = diagnostics(parameters)
        _gamma, gammadash, _, _ = stage_two_coil_geometry(extraction, parameters)
        base_gammadash = jax.lax.slice_in_dim(gammadash, 0, NUM_BASE_CURVES, axis=0)
        total_length = jnp.sum(
            jnp.mean(jnp.linalg.norm(base_gammadash, axis=-1), axis=1)
        )
        return jnp.stack(
            (flux_objective(parameters), force_objective, vacuum_energy, total_length)
        )

    initial_parameters = put(initial)
    direction_device = put(direction)
    problem = TraceableParametricScalarProblem(
        objective_fn=weighted_objective,
        objective_parameter=put(FIRST_LENGTH_WEIGHT),
        x=put(_optimizer_start(initial, direction)),
    )
    state_program = TraceableArrayFunction(state_diagnostics, initial_parameters)

    def state(prefix: str, parameters: jax.Array) -> dict[str, np.ndarray]:
        objective_value, gradient = problem.value_and_grad(parameters)
        published = jax.device_get(
            (parameters, objective_value, gradient, state_program(parameters))
        )
        scalars = np.asarray(published[3], dtype=np.float64)
        return _state_values(
            prefix,
            parameters=np.asarray(published[0], dtype=np.float64),
            objective=float(published[1]),
            gradient=np.asarray(published[2], dtype=np.float64),
            squared_flux=float(scalars[0]),
            force_objective=float(scalars[1]),
            vacuum_energy=float(scalars[2]),
            total_curve_length=float(scalars[3]),
        )

    def solve():
        # The delivered mirror's route (``STAGE_DRIVER``): SciPy's own L-BFGS-B
        # over the device objective, named on what the official call names.
        return solve_scalar_stage(
            problem,
            driver=Driver.SCIPY_LBFGSB,
            max_steps=MAX_STEPS,
            maxcor=min(MAX_STEPS, 300),
            tol=TOL,
        )

    initial_values = state("initial", initial_parameters)
    _initial_value, initial_gradient = problem.value_and_grad(initial_parameters)
    directional_derivative = jnp.vdot(initial_gradient, direction_device)

    def taylor_error(epsilon: jax.Array) -> jax.Array:
        plus = problem.objective(initial_parameters + epsilon * direction_device)
        minus = problem.objective(initial_parameters - epsilon * direction_device)
        return (plus - minus) / (epsilon + epsilon) - directional_derivative

    taylor_errors = jax.vmap(taylor_error)(put(TAYLOR_EPSILONS))
    first = solve()
    first_values = state("first", problem.x)
    problem.set_objective_parameter(put(SECOND_LENGTH_WEIGHT))
    second = solve()
    final_values = state("final", problem.x)
    return _LaneRun(
        values={
            **initial_values,
            **first_values,
            **final_values,
            "taylor:errors": np.asarray(
                jax.device_get(taylor_errors), dtype=np.float64
            ),
        },
        stages=tuple(
            _StageStop(bool(result.success), int(result.status), int(result.nit))
            for result in (first, second)
        ),
    )


def test_exact_coil_forces_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.asarray(BiotSavart(_geometry()[2]).x, dtype=np.float64)
    direction = np.random.RandomState(1).uniform(size=initial.shape)

    native = _run_native(initial, direction)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane = _run_jax(initial, direction)

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
    for run in (native, jax_lane):
        terminal = _terminal_status(run)
        assert terminal.normalized_status == "budget_exhausted"
        assert terminal.success is False
    assert set(native.values) == set(jax_lane.values)

    for observable in native.values:
        if observable.startswith("initial:"):
            np.testing.assert_allclose(
                jax_lane.values[observable],
                native.values[observable],
                rtol=2.0e-8,
                atol=2.0e-10,
            )

    for stage in ("first", "final"):
        for run in (native, jax_lane):
            assert float(run.values[f"{stage}:objective"]) < float(
                run.values["initial:objective"]
            )
            assert np.all(np.isfinite(run.values[f"{stage}:objective_gradient"]))

    np.testing.assert_allclose(
        jax_lane.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-2,
        atol=1.0e-9,
    )
    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax_lane.values["taylor:errors"][:3])) <= 1.0e-4
