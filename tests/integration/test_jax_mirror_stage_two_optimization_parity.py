"""Exact parity for ``2_Intermediate/stage_two_optimization.py``.

Both lanes are built live at the reduced (bounded) scale: the native lane is
upstream's two-stage L-BFGS-B over the native SIMSOPT objective, the JAX lane
is :func:`simsopt_jax.examples.solve_standard_stage_two` on the same coil set,
start vector and Taylor direction.  No reference data is stored.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Current, coils_via_symmetries
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
from simsopt_jax.examples import solve_standard_stage_two, standard_stage_two_state
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: The reduced scale of the official script: geometry shrunk, policy kept.
SURFACE_RESOLUTION = 4
CURVE_ORDER = 2
CURVE_QUADRATURE = 16
NUM_BASE_CURVES = 4
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.5
INITIAL_CURRENT = 1.0e5
FIRST_LENGTH_WEIGHT = 1.0e-6
SECOND_LENGTH_WEIGHT = 1.0e-7
CURVE_CURVE_THRESHOLD = 0.1
CURVE_CURVE_WEIGHT = 1000.0
CURVE_SURFACE_THRESHOLD = 0.3
CURVE_SURFACE_WEIGHT = 10.0
CURVATURE_THRESHOLD = 5.0
CURVATURE_WEIGHT = 1.0e-6
MEAN_SQUARED_CURVATURE_THRESHOLD = 5.0
MEAN_SQUARED_CURVATURE_WEIGHT = 1.0e-6
MAX_STEPS = 50
#: What the official ``tol=1e-15`` makes SciPy's L-BFGS-B set: ftol and gtol.
RTOL = 1.0e-15
ATOL = 1.0e-15
LBFGS_HISTORY = 300
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)

#: Largest absolute gradient difference between the two evaluators at one state,
#: measured at both lanes' endpoints over OMP_NUM_THREADS 1/2/4/8/16: 3.772e-17,
#: against a gradient whose inf-norm is ~1e-5.  The gate is ~26x that, which is
#: tight enough that a real evaluator divergence cannot hide under it and loose
#: enough to absorb summation-order noise between the C++ and XLA reductions.
CROSS_EVALUATION_GRADIENT_ATOL = 1.0e-15

#: Largest relative objective difference in the same measurement: 3.826e-15.
CROSS_EVALUATION_OBJECTIVE_RTOL = 1.0e-12

_STATE_OBSERVABLES = (
    "parameters",
    "objective",
    "objective_gradient",
    "squared_flux",
    "geometric_penalty",
    "maximum_normal_field",
    "total_curve_length",
)


def _geometry():
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
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
    )
    base_currents = [Current(INITIAL_CURRENT) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


def _start() -> tuple[np.ndarray, np.ndarray]:
    """The source-equivalent start vector and the seeded Taylor direction."""
    _surface, _curves, coils = _geometry()
    initial = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    return initial, np.random.RandomState(1).uniform(size=initial.shape)


@dataclass(frozen=True)
class _NativeProblem:
    field: BiotSavart
    flux: SquaredFlux
    lengths: tuple[CurveLength, ...]
    unit_normal: np.ndarray
    objective: Callable[[float], Optimizable]


def _native_problem() -> _NativeProblem:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    lengths = tuple(CurveLength(curve) for curve in base_curves)
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

    def objective(length_weight: float) -> Optimizable:
        return (
            flux
            + length_weight * sum(lengths)
            + CURVE_CURVE_WEIGHT * curve_curve
            + CURVE_SURFACE_WEIGHT * curve_surface
            + CURVATURE_WEIGHT * sum(curvatures)
            + MEAN_SQUARED_CURVATURE_WEIGHT * sum(mean_squared_penalties)
        )

    return _NativeProblem(
        field=field,
        flux=flux,
        lengths=lengths,
        unit_normal=surface.unitnormal().reshape((-1, 3)),
        objective=objective,
    )


@dataclass(frozen=True)
class _StageStop:
    """Raw provider state of one optimizer stage."""

    success: bool
    status: int
    iterations: int


@dataclass(frozen=True)
class _LaneRun:
    values: dict[str, np.ndarray]
    stages: tuple[_StageStop, _StageStop]


def _state_values(prefix: str, **observables: object) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:{name}": np.asarray(observables[name], dtype=np.float64)
        for name in _STATE_OBSERVABLES
    }


def _stage_reason(
    stop: _StageStop,
    values: dict[str, np.ndarray],
    start: str,
    end: str,
) -> StoppingReason:
    return certify_optimization_endpoint(
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
    problem = _native_problem()

    def state(prefix: str, parameters: np.ndarray, current: Optimizable):
        current.x = parameters
        squared_flux = float(problem.flux.J())
        objective_value = float(current.J())
        normal_field = np.sum(problem.field.B() * problem.unit_normal, axis=1)
        return _state_values(
            prefix,
            parameters=parameters,
            objective=objective_value,
            objective_gradient=np.asarray(current.dJ(), dtype=np.float64),
            squared_flux=squared_flux,
            geometric_penalty=objective_value - squared_flux,
            maximum_normal_field=float(np.max(np.abs(normal_field))),
            total_curve_length=float(sum(length.J() for length in problem.lengths)),
        )

    first_objective = problem.objective(FIRST_LENGTH_WEIGHT)
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
        taylor_errors.append((plus - minus) / (2.0 * epsilon) - directional_derivative)

    def solve(current: Optimizable, start: np.ndarray) -> OptimizeResult:
        def value_and_gradient(parameters: np.ndarray):
            current.x = parameters
            return float(current.J()), np.asarray(current.dJ(), dtype=np.float64)

        return minimize(
            value_and_gradient,
            start,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": MAX_STEPS,
                "maxcor": LBFGS_HISTORY,
                "ftol": RTOL,
                "gtol": ATOL,
            },
        )

    first = solve(first_objective, initial)
    first_parameters = np.asarray(first.x, dtype=np.float64)
    first_values = state("first", first_parameters, first_objective)
    second_objective = problem.objective(SECOND_LENGTH_WEIGHT)
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


def _regularization_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=NUM_BASE_CURVES,
        curve_curve_minimum_distance=CURVE_CURVE_THRESHOLD,
        curve_curve_weight=CURVE_CURVE_WEIGHT,
        curve_surface_minimum_distance=CURVE_SURFACE_THRESHOLD,
        curve_surface_weight=CURVE_SURFACE_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        mean_squared_curvature_threshold=MEAN_SQUARED_CURVATURE_THRESHOLD,
        mean_squared_curvature_weight=MEAN_SQUARED_CURVATURE_WEIGHT,
    )


def _run_jax(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    surface, _base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    device = get_runtime_jax_device()

    def put(value: np.ndarray) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    result = solve_standard_stage_two(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=put(surface.gamma().reshape((-1, 3))),
        surface_normal=put(surface.normal().reshape((-1, 3))),
        initial_parameters=put(initial),
        taylor_direction=put(direction),
        regularization_config=_regularization_config(),
        first_length_weight=put(FIRST_LENGTH_WEIGHT),
        second_length_weight=put(SECOND_LENGTH_WEIGHT),
        max_steps=MAX_STEPS,
        rtol=RTOL,
        atol=ATOL,
        driver=Driver.SCIPY_LBFGSB,
    )
    initial_state, first_state, final_state, taylor_errors = jax.device_get(
        (result.initial, result.first, result.final, result.taylor_errors)
    )
    values: dict[str, np.ndarray] = {
        "taylor:errors": np.asarray(taylor_errors, dtype=np.float64)
    }
    for prefix, state in (
        ("initial", initial_state),
        ("first", first_state),
        ("final", final_state),
    ):
        values.update(
            _state_values(
                prefix, **{name: getattr(state, name) for name in _STATE_OBSERVABLES}
            )
        )
    return _LaneRun(
        values=values,
        stages=tuple(
            _StageStop(bool(optimizer.success), int(optimizer.status), optimizer.nit)
            for optimizer in (result.first_optimizer, result.second_optimizer)
        ),
    )


def _run_both_lanes(monkeypatch: pytest.MonkeyPatch) -> tuple[_LaneRun, _LaneRun]:
    initial, direction = _start()
    native = _run_native(initial, direction)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    return native, _run_jax(initial, direction)


def test_exact_standard_stage_two_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native, jax_lane = _run_both_lanes(monkeypatch)

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
    for run in (native, jax_lane):
        terminal = _terminal_status(run)
        assert terminal.normalized_status == "budget_exhausted"
        assert terminal.success is False
    assert set(native.values) == set(jax_lane.values)

    for observable in _STATE_OBSERVABLES:
        np.testing.assert_allclose(
            jax_lane.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    for stage in ("first", "final"):
        for run in (native, jax_lane):
            assert run.values[f"{stage}:objective"] < run.values["initial:objective"]
            assert np.all(np.isfinite(run.values[f"{stage}:objective_gradient"]))

    # Endpoint gates, measured rather than guessed, with atol removed: at this
    # scale |final delta| is ~1e-9 and an atol of 1e-9 passed the assertion on
    # its own, whatever rtol said.  Both lanes run SciPy L-BFGS-B under one
    # stopping rule and both stop on the iteration cap, so what is left is
    # trajectory divergence from floating-point arithmetic -- and the native
    # lane's trajectory is thread-count sensitive at this scale.  Measured
    # relative deltas over OMP_NUM_THREADS 1/2/4/8/16 on this box:
    #   first:objective  1.174e-07, 1.174e-07, 1.170e-07, 1.175e-07, 1.177e-07
    #   final:objective  1.750e-03, 8.516e-04, 1.418e-04, 3.094e-03, 5.723e-04
    # The gates are the worst of each sweep with roughly an order of magnitude
    # (first) and 3x (final) of margin.  They compare two TRAJECTORIES and are
    # deliberately the loose half of this file; the tight half is
    # test_both_evaluators_agree_at_each_lanes_endpoint, which compares the two
    # evaluators at one state and gates at 1e-15 absolute.
    np.testing.assert_allclose(
        jax_lane.values["first:objective"],
        native.values["first:objective"],
        rtol=1.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax_lane.values["final:objective"],
        native.values["final:objective"],
        rtol=1.0e-2,
        atol=0.0,
    )
    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax_lane.values["taylor:errors"][:3])) <= 1.0e-4


def _native_evaluate_at(parameters: np.ndarray) -> dict[str, np.ndarray]:
    current = _native_problem().objective(SECOND_LENGTH_WEIGHT)
    current.x = np.asarray(parameters, dtype=np.float64)
    return {
        "objective": np.asarray(float(current.J()), dtype=np.float64),
        "objective_gradient": np.asarray(current.dJ(), dtype=np.float64),
    }


def _jax_evaluate_at(parameters: np.ndarray) -> dict[str, np.ndarray]:
    surface, _base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    state = jax.device_get(
        standard_stage_two_state(
            field=field,
            flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
            surface_gamma=np.asarray(surface.gamma(), dtype=np.float64).reshape(
                (-1, 3)
            ),
            surface_normal=np.asarray(surface.normal(), dtype=np.float64).reshape(
                (-1, 3)
            ),
            parameters=parameters,
            regularization_config=_regularization_config(),
            length_weight=SECOND_LENGTH_WEIGHT,
        )
    )
    return {
        "objective": np.asarray(float(state.objective), dtype=np.float64),
        "objective_gradient": np.asarray(state.objective_gradient, dtype=np.float64),
    }


def test_both_evaluators_agree_at_each_lanes_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-evaluation: one state, two evaluators, at both lanes' endpoints.

    The endpoint comparison above cannot separate "the two lanes compute
    different physics" from "the two lanes walked different trajectories to the
    same cap".  This one can: each lane's final coordinates are handed to BOTH
    evaluators, at the second stage's length weight (the one both lanes finish
    in), so the only thing that can move the numbers is the objective program
    itself.
    """
    native, jax_lane = _run_both_lanes(monkeypatch)

    for holder, run in (("jax", jax_lane), ("native", native)):
        endpoint = np.asarray(run.values["final:parameters"], dtype=np.float64)
        by_native = _native_evaluate_at(endpoint)
        by_jax = _jax_evaluate_at(endpoint)

        np.testing.assert_allclose(
            by_jax["objective"],
            by_native["objective"],
            rtol=CROSS_EVALUATION_OBJECTIVE_RTOL,
            atol=0.0,
            err_msg=(
                f"the two evaluators disagree about the objective at the {holder} "
                "lane's endpoint, which is a difference in the objective program "
                "and not in either optimizer's trajectory"
            ),
        )
        np.testing.assert_allclose(
            by_jax["objective_gradient"],
            by_native["objective_gradient"],
            rtol=0.0,
            atol=CROSS_EVALUATION_GRADIENT_ATOL,
            err_msg=(
                f"the two evaluators disagree about the gradient at the {holder} "
                "lane's endpoint; the endpoint-objective gate above cannot see "
                "this, which is why this comparison exists"
            ),
        )
        assert np.all(np.isfinite(by_jax["objective_gradient"]))
        assert np.all(np.isfinite(by_native["objective_gradient"]))
