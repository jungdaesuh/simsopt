"""The ``stage_two_optimization_planar_coils.py`` mirror against native SIMSOPT.

Both lanes are built live at the reduced (bounded) scale: the native lane is
upstream's two-stage L-BFGS-B over the native planar-coil objective, the JAX
lane is :func:`simsopt_jax.examples.solve_standard_stage_two` with the planar
regularization, plus the library's planar topology diagnostics.  Both start
from the same coil set and Taylor direction.  No reference data is stored.

The end point forks between runs at round-off at this scale, so it is
informational; the stages are judged lane against lane by the first-stage end
objective, and same-state agreement is
``test_jax_planar_coils_objective_roundoff_agreement.py``.
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
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_planar_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_contracts.optimization_endpoint import (
    StoppingReason,
    TerminalStatus,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_standard_stage_two
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    stage_two_planar_topology_values,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: The reduced scale of the official script.
SURFACE_RESOLUTION = 4
CURVE_ORDER = 2
CURVE_QUADRATURE = 32
NUM_BASE_CURVES = 4
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.5
INITIAL_CURRENT = 1.0e5
LENGTH_TARGET = 10.4
FIRST_LENGTH_WEIGHT = 10.0
SECOND_LENGTH_WEIGHT = 1.0
CURVE_CURVE_THRESHOLD = 0.08
CURVE_CURVE_WEIGHT = 1000.0
CURVE_SURFACE_THRESHOLD = 0.12
CURVE_SURFACE_WEIGHT = 10.0
CURVATURE_THRESHOLD = 10.0
CURVATURE_WEIGHT = 1.0e-6
MEAN_SQUARED_CURVATURE_THRESHOLD = 10.0
MEAN_SQUARED_CURVATURE_WEIGHT = 1.0e-6
LINKING_NUMBER_WEIGHT = 1.0
MAX_STEPS = 50
#: What the official ``tol=1e-15`` makes SciPy's L-BFGS-B set: ftol and gtol.
RTOL = 1.0e-15
ATOL = 1.0e-15
LBFGS_HISTORY = 300
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)


def _geometry():
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="half period",
        nphi=SURFACE_RESOLUTION,
        ntheta=SURFACE_RESOLUTION,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_planar_curves(
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


def _state_values(
    prefix: str,
    *,
    parameters: np.ndarray,
    objective: float,
    objective_gradient: np.ndarray,
    squared_flux: float,
    geometric_penalty: float,
    planarity_penalty: float,
    linking_number: float,
    canonical_geometry: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(
            objective_gradient, dtype=np.float64
        ),
        f"{prefix}:squared_flux": np.asarray(squared_flux, dtype=np.float64),
        f"{prefix}:geometric_penalty": np.asarray(geometric_penalty, dtype=np.float64),
        f"{prefix}:planarity_penalty": np.asarray(planarity_penalty, dtype=np.float64),
        f"{prefix}:linking_number": np.asarray(linking_number, dtype=np.float64),
        f"{prefix}:canonical_geometry": np.asarray(
            canonical_geometry, dtype=np.float64
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
        and values["final:planarity_penalty"] <= 1.0e-24
        and values["final:linking_number"] == 0.0
    )
    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(
            _stage_reason(run.stages[0], values, "initial", "first"),
            _stage_reason(run.stages[1], values, "first", "final"),
        ),
    )


def _native_topology(base_curves, linking_number) -> tuple[float, float, np.ndarray]:
    gamma = np.stack([np.asarray(curve.gamma()) for curve in base_curves])
    gammadash = np.stack([np.asarray(curve.gammadash()) for curve in base_curves])
    centered = gamma - np.mean(gamma, axis=1, keepdims=True)
    covariance = np.einsum("nqi,nqj->nij", centered, centered) / gamma.shape[1]
    minimum_variance = np.linalg.eigvalsh(covariance)[:, 0]
    lengths = np.mean(np.linalg.norm(gammadash, axis=2), axis=1)
    canonical_geometry = np.concatenate(
        (
            lengths[:, None],
            np.mean(gamma, axis=1),
            covariance.reshape((gamma.shape[0], 9)),
        ),
        axis=1,
    ).reshape((-1,))
    return (
        float(np.sum(np.square(minimum_variance))),
        float(linking_number.J()),
        canonical_geometry,
    )


def _run_native(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in coils]
    flux = SquaredFlux(surface, field)
    length_penalty = QuadraticPenalty(
        sum(CurveLength(curve) for curve in base_curves), LENGTH_TARGET
    )
    curve_curve = CurveCurveDistance(
        curves, CURVE_CURVE_THRESHOLD, num_basecurves=NUM_BASE_CURVES
    )
    curve_surface = CurveSurfaceDistance(curves, surface, CURVE_SURFACE_THRESHOLD)
    curvature = sum(
        LpCurveCurvature(curve, 2, CURVATURE_THRESHOLD) for curve in base_curves
    )
    mean_squared_curvature = sum(
        QuadraticPenalty(MeanSquaredCurvature(curve), MEAN_SQUARED_CURVATURE_THRESHOLD)
        for curve in base_curves
    )
    linking_number = LinkingNumber(curves)

    def weighted(length_weight: float) -> Optimizable:
        """The script's objective at one stage's length weight, in its order."""
        return (
            flux
            + length_weight * length_penalty
            + CURVE_CURVE_WEIGHT * curve_curve
            + CURVE_SURFACE_WEIGHT * curve_surface
            + CURVATURE_WEIGHT * curvature
            + MEAN_SQUARED_CURVATURE_WEIGHT * mean_squared_curvature
            + LINKING_NUMBER_WEIGHT * linking_number
        )

    def state(prefix: str, parameters: np.ndarray, current: Optimizable):
        current.x = parameters
        squared_flux = float(flux.J())
        objective_value = float(current.J())
        planarity, linking, canonical_geometry = _native_topology(
            base_curves, linking_number
        )
        return _state_values(
            prefix,
            parameters=parameters,
            objective=objective_value,
            objective_gradient=np.asarray(current.dJ(), dtype=np.float64),
            squared_flux=squared_flux,
            geometric_penalty=objective_value - squared_flux,
            planarity_penalty=planarity,
            linking_number=linking,
            canonical_geometry=canonical_geometry,
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
            options={
                "maxiter": MAX_STEPS,
                "maxcor": LBFGS_HISTORY,
                "ftol": RTOL,
                "gtol": ATOL,
            },
        )

    first_objective = weighted(FIRST_LENGTH_WEIGHT)
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
    first = solve(first_objective, initial)
    first_parameters = np.asarray(first.x, dtype=np.float64)
    first_values = state("first", first_parameters, first_objective)
    second_objective = weighted(SECOND_LENGTH_WEIGHT)
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
        length_target=LENGTH_TARGET,
        length_target_mode="identity",
        curve_curve_minimum_distance=CURVE_CURVE_THRESHOLD,
        curve_curve_weight=CURVE_CURVE_WEIGHT,
        curve_surface_minimum_distance=CURVE_SURFACE_THRESHOLD,
        curve_surface_weight=CURVE_SURFACE_WEIGHT,
        curvature_threshold=CURVATURE_THRESHOLD,
        curvature_weight=CURVATURE_WEIGHT,
        mean_squared_curvature_threshold=MEAN_SQUARED_CURVATURE_THRESHOLD,
        mean_squared_curvature_target_mode="identity",
        mean_squared_curvature_weight=MEAN_SQUARED_CURVATURE_WEIGHT,
        linking_number_weight=LINKING_NUMBER_WEIGHT,
    )


def _run_jax(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    surface, _base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    device = get_runtime_jax_device()

    def put(value: object) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    regularization_config = _regularization_config()
    result = solve_standard_stage_two(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=put(surface.gamma().reshape((-1, 3))),
        surface_normal=put(surface.normal().reshape((-1, 3))),
        initial_parameters=put(initial),
        taylor_direction=put(direction),
        regularization_config=regularization_config,
        first_length_weight=put(FIRST_LENGTH_WEIGHT),
        second_length_weight=put(SECOND_LENGTH_WEIGHT),
        max_steps=MAX_STEPS,
        rtol=RTOL,
        atol=ATOL,
        driver=Driver.SCIPY_LBFGSB,
    )
    parameter_states = jnp.stack(
        (result.initial.parameters, result.first.parameters, result.final.parameters)
    )

    def evaluate_topology(extraction_operand, parameter_states_operand):
        return jax.vmap(
            lambda parameters: stage_two_planar_topology_values(
                extraction_operand, parameters, regularization_config.num_base_curves
            )
        )(parameter_states_operand)

    topology_states = jax.jit(evaluate_topology)(
        field.coil_dof_extraction_spec(), parameter_states
    )
    initial_state, first_state, final_state, taylor_errors, topology_states = (
        jax.device_get(
            (
                result.initial,
                result.first,
                result.final,
                result.taylor_errors,
                jax.block_until_ready(topology_states),
            )
        )
    )
    values: dict[str, np.ndarray] = {
        "taylor:errors": np.asarray(taylor_errors, dtype=np.float64)
    }
    for index, (prefix, state) in enumerate(
        (("initial", initial_state), ("first", first_state), ("final", final_state))
    ):
        planarity, linking, canonical_geometry = (
            topology[index] for topology in topology_states
        )
        values.update(
            _state_values(
                prefix,
                parameters=state.parameters,
                objective=float(state.objective),
                objective_gradient=state.objective_gradient,
                squared_flux=float(state.squared_flux),
                geometric_penalty=float(state.geometric_penalty),
                planarity_penalty=float(planarity),
                linking_number=float(linking),
                canonical_geometry=canonical_geometry,
            )
        )
    return _LaneRun(
        values=values,
        stages=tuple(
            _StageStop(bool(optimizer.success), int(optimizer.status), optimizer.nit)
            for optimizer in (result.first_optimizer, result.second_optimizer)
        ),
    )


def test_exact_planar_stage_two_matches_native_and_jax_cpu(
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

    for observable in (
        "parameters",
        "objective",
        "objective_gradient",
        "squared_flux",
        "geometric_penalty",
        "planarity_penalty",
        "linking_number",
        "canonical_geometry",
    ):
        np.testing.assert_allclose(
            jax_lane.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    for stage in ("first", "final"):
        for run in (native, jax_lane):
            values = run.values
            assert values[f"{stage}:objective"] < values["initial:objective"]
            assert np.all(np.isfinite(values[f"{stage}:objective_gradient"]))
            assert float(values[f"{stage}:planarity_penalty"]) <= 1.0e-24
            assert float(values[f"{stage}:linking_number"]) == 0.0
            assert float(values[f"{stage}:squared_flux"]) <= 1.0e-2
            assert float(values[f"{stage}:geometric_penalty"]) <= 1.0e-3
            canonical_geometry = values[f"{stage}:canonical_geometry"].reshape((-1, 13))
            assert np.all(np.isfinite(canonical_geometry))
            assert abs(float(np.sum(canonical_geometry[:, 0])) - 10.4) <= 3.0e-2

    # The first stage's end objective still decides lane against lane; the
    # final one is informational.
    assert float(jax_lane.values["first:objective"]) <= (
        1.03 * float(native.values["first:objective"]) + 1.0e-9
    )

    for run in (native, jax_lane):
        taylor_errors = np.abs(run.values["taylor:errors"][:3])
        assert taylor_errors[1] <= 2.0e-2 * taylor_errors[0]
        assert taylor_errors[2] <= 2.0e-2 * taylor_errors[1]
        assert taylor_errors[2] <= 1.0e-4
