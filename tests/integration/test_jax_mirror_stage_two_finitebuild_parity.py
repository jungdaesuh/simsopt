"""Exact parity for the finite-build Stage-II mirror.

Both lanes are built live at the reduced (bounded) scale of
``3_Advanced/stage_two_optimization_finitebuild.py``: the native lane is
upstream's scaled L-BFGS-B over the native multifilament objective, the JAX
lane is :func:`simsopt_jax.examples.stage_two_finitebuild.solve_finite_build_stage_two`
over the adapter objective, from the same start vector.  No reference data is
stored.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import itertools
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
)
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_contracts.optimization_endpoint import (
    TerminalStatus,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.examples.stage_two_finitebuild import (
    FINITE_BUILD_LBFGS_HISTORY,
    FINITE_BUILD_MAX_FUNCTION_EVALUATIONS,
    FINITE_BUILD_OFFICIAL_DRIVER,
    FINITE_BUILD_TOLERANCE,
    prepare_finite_build_stage_two,
    solve_finite_build_stage_two,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.finite_build_stage_two import (
    FINITE_BUILD_DIAGNOSTIC_FIELDS,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: The reduced scale of the official script.
SURFACE_RESOLUTION = 4
CURVE_ORDER = 2
CURVE_QUADRATURE = 8
NUM_BASE_CURVES = 4
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.7
INITIAL_TOTAL_CURRENT = 1.0e5
NUM_FILAMENTS_NORMAL = 2
NUM_FILAMENTS_BINORMAL = 3
GAP_SIZE_NORMAL = 0.02
GAP_SIZE_BINORMAL = 0.04
ROTATION_ORDER = 1
LENGTH_WEIGHT = 1.0e-2
CURVE_CURVE_THRESHOLD = 0.1
CURVE_CURVE_WEIGHT = 10.0
#: Upstream's provider call is ``fun -> (1e-4 * J, 1e-4 * dJ)``; the published
#: states report the unscaled objective.
OBJECTIVE_SCALE = 1.0e-4
PUBLISHED_OBJECTIVE_SCALE = 1.0
MAX_STEPS = 3
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)

_DIAGNOSTIC_INDEX = {
    name: index for index, name in enumerate(FINITE_BUILD_DIAGNOSTIC_FIELDS)
}


@dataclass(frozen=True)
class _Geometry:
    surface: SurfaceRZFourier
    base_curves: list
    symmetric_base_curves: list
    coils: list[Coil]
    config: FiniteBuildStageTwoConfig


def _geometry() -> _Geometry:
    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT,
        range="half period",
        nphi=SURFACE_RESOLUTION,
        ntheta=SURFACE_RESOLUTION,
    )
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
    filament_count = NUM_FILAMENTS_NORMAL * NUM_FILAMENTS_BINORMAL
    base_currents = []
    for index in range(NUM_BASE_CURVES):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(current * (INITIAL_TOTAL_CURRENT / filament_count))
    base_filaments = list(
        itertools.chain.from_iterable(
            create_multifilament_grid(
                curve,
                NUM_FILAMENTS_NORMAL,
                NUM_FILAMENTS_BINORMAL,
                GAP_SIZE_NORMAL,
                GAP_SIZE_BINORMAL,
                rotation_order=ROTATION_ORDER,
            )
            for curve in base_curves
        )
    )
    filament_currents = list(
        itertools.chain.from_iterable(
            [current] * filament_count for current in base_currents
        )
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(
            apply_symmetries_to_curves(base_filaments, surface.nfp, True),
            apply_symmetries_to_currents(filament_currents, surface.nfp, True),
            strict=True,
        )
    ]
    config = FiniteBuildStageTwoConfig(
        num_base_curves=NUM_BASE_CURVES,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=NUM_FILAMENTS_NORMAL,
            numfilaments_b=NUM_FILAMENTS_BINORMAL,
            gapsize_n=GAP_SIZE_NORMAL,
            gapsize_b=GAP_SIZE_BINORMAL,
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=tuple(float(CurveLength(curve).J()) for curve in base_curves),
        length_weight=LENGTH_WEIGHT,
        curve_curve_minimum_distance=CURVE_CURVE_THRESHOLD,
        curve_curve_weight=CURVE_CURVE_WEIGHT,
    )
    return _Geometry(
        surface=surface,
        base_curves=base_curves,
        symmetric_base_curves=apply_symmetries_to_curves(
            base_curves, surface.nfp, True
        ),
        coils=coils,
        config=config,
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
    minimum_clearance: float,
    coil_lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": np.asarray(gradient, dtype=np.float64),
        f"{prefix}:squared_flux": np.asarray(squared_flux, dtype=np.float64),
        f"{prefix}:length_penalty": np.asarray(length_penalty, dtype=np.float64),
        f"{prefix}:distance_penalty": np.asarray(distance_penalty, dtype=np.float64),
        f"{prefix}:minimum_clearance": np.asarray(minimum_clearance, dtype=np.float64),
        f"{prefix}:coil_lengths": np.asarray(coil_lengths, dtype=np.float64),
    }


@dataclass(frozen=True)
class _LaneRun:
    values: dict[str, np.ndarray]
    success: bool
    status: int
    iterations: int


def _terminal_status(run: _LaneRun) -> TerminalStatus:
    """The library's fold of the stop and the case's scientific predicate."""
    values = run.values
    reason = certify_optimization_endpoint(
        # Both lanes ran SciPy's own L-BFGS-B, so one status table reads both.
        status_convention="scipy-lbfgsb",
        provider_success=run.success,
        provider_status=run.status,
        iterations=run.iterations,
        max_iterations=MAX_STEPS,
        initial_gradient_inf_norm=float(
            np.max(np.abs(values["initial:objective_gradient"]))
        ),
        final_gradient_inf_norm=float(
            np.max(np.abs(values["final:objective_gradient"]))
        ),
        parameters_finite=bool(np.all(np.isfinite(values["final:parameters"]))),
        observables_finite=bool(np.isfinite(values["final:objective"])),
        inner_success=True,
    ).stopping_reason
    scientific_predicate = bool(
        np.isfinite(values["final:objective"])
        and values["final:objective"] < values["initial:objective"]
        and np.all(np.isfinite(values["final:objective_gradient"]))
    )
    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(reason,),
    )


def _run_native(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    geometry = _geometry()
    field = BiotSavart(geometry.coils)
    field.set_points(geometry.surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(geometry.surface, field)
    lengths = [CurveLength(curve) for curve in geometry.base_curves]
    distance = CurveCurveDistance(geometry.symmetric_base_curves, CURVE_CURVE_THRESHOLD)
    length_term = LENGTH_WEIGHT * sum(
        QuadraticPenalty(length, target, "max")
        for length, target in zip(lengths, geometry.config.length_targets, strict=True)
    )
    distance_term = CURVE_CURVE_WEIGHT * distance
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
            minimum_clearance=float(distance.shortest_distance()),
            coil_lengths=np.asarray([length.J() for length in lengths]),
        )

    initial_values = state("initial", initial)
    directional_derivative = float(
        np.vdot(
            OBJECTIVE_SCALE * initial_values["initial:objective_gradient"], direction
        )
    )
    taylor_errors = []
    for epsilon in TAYLOR_EPSILONS:
        objective.x = initial + epsilon * direction
        plus = OBJECTIVE_SCALE * float(objective.J())
        objective.x = initial - epsilon * direction
        minus = OBJECTIVE_SCALE * float(objective.J())
        taylor_errors.append((plus - minus) / (2 * epsilon) - directional_derivative)

    def value_and_gradient(parameters: np.ndarray):
        objective.x = parameters
        return (
            OBJECTIVE_SCALE * float(objective.J()),
            OBJECTIVE_SCALE * np.asarray(objective.dJ(), dtype=np.float64),
        )

    result = minimize(
        value_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": MAX_STEPS,
            "maxcor": FINITE_BUILD_LBFGS_HISTORY,
            "maxfun": FINITE_BUILD_MAX_FUNCTION_EVALUATIONS,
            "gtol": FINITE_BUILD_TOLERANCE,
            "ftol": FINITE_BUILD_TOLERANCE,
        },
        tol=FINITE_BUILD_TOLERANCE,
    )
    return _LaneRun(
        values={
            **initial_values,
            **state("final", np.asarray(result.x, dtype=np.float64)),
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
        success=bool(result.success),
        status=int(result.status),
        iterations=int(result.nit),
    )


def _run_jax(initial: np.ndarray, direction: np.ndarray) -> tuple[_LaneRun, Driver]:
    geometry = _geometry()
    field = BiotSavartJAX(geometry.coils)
    flux_spec = SquaredFluxJAX(geometry.surface, field).fixed_surface_flux_spec()
    device = get_runtime_jax_device()

    def put(value: object) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    initial_parameters = put(initial)
    prepared = prepare_finite_build_stage_two(
        objective_fn=make_finite_build_stage_two_objective(
            field, flux_spec, geometry.config
        ),
        diagnostics_fn=finite_build_stage_two_diagnostics(
            field, flux_spec, geometry.config
        ),
        initial_parameters=initial_parameters,
        objective_scale=put(PUBLISHED_OBJECTIVE_SCALE),
    )
    problem = prepared.problem

    def state(prefix: str, parameters: jax.Array) -> dict[str, np.ndarray]:
        objective_value, gradient = problem.value_and_grad(parameters)
        published = jax.device_get(
            (parameters, objective_value, gradient, prepared.diagnostics(parameters))
        )
        packed = np.asarray(published[3], dtype=np.float64)
        return _state_values(
            prefix,
            parameters=np.asarray(published[0], dtype=np.float64),
            objective=float(published[1]),
            gradient=np.asarray(published[2], dtype=np.float64),
            squared_flux=float(packed[_DIAGNOSTIC_INDEX["squared_flux"]]),
            length_penalty=float(packed[_DIAGNOSTIC_INDEX["length_penalty"]]),
            distance_penalty=float(packed[_DIAGNOSTIC_INDEX["distance_penalty"]]),
            minimum_clearance=float(packed[_DIAGNOSTIC_INDEX["minimum_clearance"]]),
            coil_lengths=packed[len(FINITE_BUILD_DIAGNOSTIC_FIELDS) :],
        )

    initial_values = state("initial", initial_parameters)
    direction_device = put(direction)
    problem.set_objective_parameter(put(OBJECTIVE_SCALE))
    _scaled_value, scaled_gradient = problem.value_and_grad(initial_parameters)
    directional_derivative = jnp.vdot(scaled_gradient, direction_device)

    def taylor_error(epsilon: jax.Array) -> jax.Array:
        plus = problem.objective(initial_parameters + epsilon * direction_device)
        minus = problem.objective(initial_parameters - epsilon * direction_device)
        return (plus - minus) / (epsilon + epsilon) - directional_derivative

    taylor_errors = jax.vmap(taylor_error)(put(TAYLOR_EPSILONS))
    result = solve_finite_build_stage_two(
        prepared,
        driver=FINITE_BUILD_OFFICIAL_DRIVER,
        max_steps=MAX_STEPS,
        rtol=FINITE_BUILD_TOLERANCE,
        atol=FINITE_BUILD_TOLERANCE,
    )
    problem.set_objective_parameter(put(PUBLISHED_OBJECTIVE_SCALE))
    run = _LaneRun(
        values={
            **initial_values,
            **state("final", problem.x),
            "taylor:errors": np.asarray(
                jax.device_get(taylor_errors), dtype=np.float64
            ),
        },
        success=bool(result.success),
        status=int(result.status),
        iterations=int(result.nit),
    )
    return run, result.driver


def test_exact_finitebuild_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.asarray(BiotSavart(_geometry().coils).x, dtype=np.float64)
    direction = np.random.RandomState(1).uniform(size=initial.shape)

    native = _run_native(initial, direction)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane, jax_driver = _run_jax(initial, direction)

    # The mirror solves with the provider upstream calls; a lane that ran the
    # library's device L-BFGS-B would not be mirroring upstream's workflow.
    assert jax_driver.value == Driver.SCIPY_LBFGSB.value

    # The bounded budget ends every stage on its iteration cap. The label says
    # so; it is neither convergence nor failure, and it still implies the case's
    # scientific predicate, because a false predicate is labelled ``failed``.
    for run in (native, jax_lane):
        terminal = _terminal_status(run)
        assert terminal.normalized_status == "budget_exhausted"
        assert terminal.success is False
    published = set(native.values)
    assert published == set(jax_lane.values)
    assert {"initial:minimum_clearance", "final:minimum_clearance"} <= published

    for observable in native.values:
        if observable.startswith("initial:"):
            np.testing.assert_allclose(
                jax_lane.values[observable],
                native.values[observable],
                rtol=2.0e-8,
                atol=2.0e-10,
            )

    for run in (native, jax_lane):
        assert float(run.values["final:objective"]) < float(
            run.values["initial:objective"]
        )
        assert np.all(np.isfinite(run.values["final:objective_gradient"]))

    np.testing.assert_allclose(
        jax_lane.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-2,
        atol=1.0e-9,
    )

    # Minimum clearance is an exact geometry quantity that the two lanes
    # compute with independent evaluators: native reduces the simsoptpp
    # ``CurveCurveDistance.shortest_distance()`` over the symmetric base
    # curves, JAX takes the pairwise minimum packed in the diagnostics.
    # Measured once at bounded scale on CPU fp64: the initial state agrees
    # bitwise (gap exactly 0.0) and the final state agrees to a relative gap
    # of 3.17e-12 (absolute 3.00e-13), inherited from the 4.08e-12 spread
    # between the two lanes' converged parameters rather than from the
    # reductions.  The tolerance below sits ~160x above that measured gap.
    for prefix in ("initial", "final"):
        observable = f"{prefix}:minimum_clearance"
        assert float(native.values[observable]) > 0.0
        np.testing.assert_allclose(
            jax_lane.values[observable],
            native.values[observable],
            rtol=5.0e-10,
            atol=1.0e-12,
        )

    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax_lane.values["taylor:errors"][:3])) <= 1.0e-4
