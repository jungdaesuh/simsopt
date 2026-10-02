"""The ``stage_two_optimization_minimal.py`` mirror against native SIMSOPT.

Both lanes are built live at the reduced (bounded) scale: the native lane is
upstream's single L-BFGS-B call over the native flux-plus-length objective,
the JAX lane is :func:`simsopt_jax.examples.solve_minimal_stage_two` on the same
coil set, start vector and Taylor direction.  The optimizer policy the library
owns is checked against upstream's script itself.  No reference data is stored.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    TerminalStatus,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_minimal_stage_two
from simsopt_jax.examples.stage_two_minimal import (
    MINIMAL_STAGE_TWO_LBFGS_HISTORY,
    MINIMAL_STAGE_TWO_NATIVE_ITERATIONS,
    MINIMAL_STAGE_TWO_OFFICIAL_DRIVER,
    MINIMAL_STAGE_TWO_TOLERANCE,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

_REPO_ROOT = Path(__file__).resolve().parents[2]
SURFACE_INPUT = _REPO_ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"
OFFICIAL_SCRIPT = (
    _REPO_ROOT / "examples" / "1_Simple" / ("stage_two_optimization_minimal.py")
)

#: The reduced scale: geometry shrunk, the official optimizer policy kept.
SURFACE_RESOLUTION = 4
CURVE_ORDER = 2
CURVE_QUADRATURE = 16
NUM_BASE_CURVES = 4
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.5
#: Official: ``Current(1.0) * 1e5``, so every free current dof is 1.0.
INITIAL_CURRENT_DEGREE_OF_FREEDOM = 1.0
CURRENT_SCALE = 1.0e5
LENGTH_WEIGHT = 1.0
LENGTH_TARGET = 18.0
MAX_STEPS = MINIMAL_STAGE_TWO_NATIVE_ITERATIONS
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)

#: Stationarity bound of a CONVERGED endpoint; it binds a converged stop only.
CONVERGED_GRADIENT_INF_NORM_BOUND = 1.0e-4

_STATE_OBSERVABLES = (
    "parameters",
    "objective",
    "objective_gradient",
    "squared_flux",
    "length_penalty",
    "maximum_normal_field",
    "total_curve_length",
)


def _official_script_maxiter() -> int:
    """``MAXITER = 50 if in_github_actions else 300``: the non-CI branch."""
    for node in ast.parse(OFFICIAL_SCRIPT.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "MAXITER"
            and isinstance(node.value, ast.IfExp)
            and isinstance(node.value.orelse, ast.Constant)
        ):
            value = node.value.orelse.value
            assert isinstance(value, int)
            return value
    raise AssertionError(f"{OFFICIAL_SCRIPT.name} defines no MAXITER")


def test_minimal_solver_policy_keeps_official_tolerance() -> None:
    # The official call is options={'maxiter': MAXITER, 'maxcor': 300},
    # tol=1e-15 (upstream examples/1_Simple/stage_two_optimization_minimal.py);
    # the library owns those numbers for every scale its callers run.
    assert MINIMAL_STAGE_TWO_TOLERANCE == 1.0e-15
    assert MINIMAL_STAGE_TWO_LBFGS_HISTORY == 300
    assert MINIMAL_STAGE_TWO_NATIVE_ITERATIONS == _official_script_maxiter()


def test_minimal_native_default_keeps_official_iteration_budget() -> None:
    assert MINIMAL_STAGE_TWO_NATIVE_ITERATIONS == _official_script_maxiter() == 300


def _geometry():
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
    )
    base_currents = [
        Current(INITIAL_CURRENT_DEGREE_OF_FREEDOM) * CURRENT_SCALE for _ in base_curves
    ]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


@dataclass(frozen=True)
class _LaneRun:
    values: dict[str, np.ndarray]
    status_convention: StatusConvention
    success: bool
    status: int
    iterations: int


def _state_values(prefix: str, **observables: object) -> dict[str, np.ndarray]:
    return {
        f"{prefix}:{name}": np.asarray(observables[name], dtype=np.float64)
        for name in _STATE_OBSERVABLES
    }


def _terminal_status(run: _LaneRun) -> TerminalStatus:
    """The library's fold of the stop and the minimal workflow's predicate.

    A finite objective below its start and a total coil length within 10 % of
    the target; the gradient bound is a stationarity claim and binds only an
    endpoint the provider reports as converged.
    """
    values = run.values
    reason = certify_optimization_endpoint(
        status_convention=run.status_convention,
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
    final_objective = float(values["final:objective"])
    gradient_inf_norm = float(
        np.linalg.norm(values["final:objective_gradient"], ord=np.inf)
    )
    scientific_predicate = bool(
        np.isfinite(final_objective)
        and final_objective < float(values["initial:objective"])
        and (
            reason != "converged"
            or gradient_inf_norm <= CONVERGED_GRADIENT_INF_NORM_BOUND
        )
        and float(values["final:total_curve_length"]) <= 1.1 * LENGTH_TARGET
    )
    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(reason,),
    )


def _run_native(initial: np.ndarray, direction: np.ndarray) -> _LaneRun:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    total_length = sum(CurveLength(curve) for curve in base_curves)
    length_penalty = QuadraticPenalty(total_length, LENGTH_TARGET, "max")
    objective = flux + LENGTH_WEIGHT * length_penalty
    unit_normal = surface.unitnormal().reshape((-1, 3))

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        objective.x = parameters
        normal_field = np.sum(field.B() * unit_normal, axis=1)
        return _state_values(
            prefix,
            parameters=parameters,
            objective=float(objective.J()),
            objective_gradient=np.asarray(objective.dJ(), dtype=np.float64),
            squared_flux=float(flux.J()),
            length_penalty=LENGTH_WEIGHT * float(length_penalty.J()),
            maximum_normal_field=float(np.max(np.abs(normal_field))),
            total_curve_length=float(total_length.J()),
        )

    initial_values = state("initial", initial)
    directional_derivative = float(
        np.vdot(initial_values["initial:objective_gradient"], direction)
    )
    taylor_errors = []
    for epsilon in TAYLOR_EPSILONS:
        objective.x = initial + epsilon * direction
        plus = float(objective.J())
        objective.x = initial - epsilon * direction
        minus = float(objective.J())
        taylor_errors.append((plus - minus) / (2.0 * epsilon) - directional_derivative)

    def value_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    result = minimize(
        value_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAX_STEPS, "maxcor": MINIMAL_STAGE_TWO_LBFGS_HISTORY},
        tol=MINIMAL_STAGE_TWO_TOLERANCE,
    )
    return _LaneRun(
        values={
            **initial_values,
            **state("final", np.asarray(result.x, dtype=np.float64)),
            "taylor:errors": np.asarray(taylor_errors, dtype=np.float64),
        },
        status_convention="scipy-lbfgsb",
        success=bool(result.success),
        status=int(result.status),
        iterations=int(result.nit),
    )


def _run_jax(initial: np.ndarray, direction: np.ndarray) -> tuple[_LaneRun, Driver]:
    surface, _base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    device = get_runtime_jax_device()

    def put(value: np.ndarray) -> jax.Array:
        return jax.device_put(np.asarray(value, dtype=np.float64), device)

    result = solve_minimal_stage_two(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=put(surface.gamma().reshape((-1, 3))),
        surface_normal=put(surface.normal().reshape((-1, 3))),
        initial_parameters=put(initial),
        taylor_direction=put(direction),
        num_base_curves=NUM_BASE_CURVES,
        length_weight=LENGTH_WEIGHT,
        length_target=LENGTH_TARGET,
        driver=MINIMAL_STAGE_TWO_OFFICIAL_DRIVER,
        max_steps=MAX_STEPS,
        rtol=MINIMAL_STAGE_TWO_TOLERANCE,
        atol=MINIMAL_STAGE_TWO_TOLERANCE,
    )
    initial_state, final_state, taylor_errors = jax.device_get(
        (result.initial, result.final, result.taylor_errors)
    )
    values: dict[str, np.ndarray] = {
        "taylor:errors": np.asarray(taylor_errors, dtype=np.float64)
    }
    for prefix, state in (("initial", initial_state), ("final", final_state)):
        values.update(
            _state_values(
                prefix, **{name: getattr(state, name) for name in _STATE_OBSERVABLES}
            )
        )
    optimizer = result.optimizer
    run = _LaneRun(
        values=values,
        # The official driver is SciPy's own L-BFGS-B over the device
        # objective, so its stop is read through SciPy's status table; the
        # test asserts the driver that ran.
        status_convention="scipy-lbfgsb",
        success=bool(optimizer.success),
        status=int(optimizer.status),
        iterations=int(optimizer.nit),
    )
    return run, optimizer.driver


def test_exact_stage_two_minimal_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = np.asarray(BiotSavart(_geometry()[2]).x, dtype=np.float64)
    direction = np.random.RandomState(1).uniform(size=initial.shape)

    native = _run_native(initial, direction)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_lane, jax_driver = _run_jax(initial, direction)

    # The mirror solves with the provider upstream calls; a lane that ran the
    # library's device L-BFGS-B would not be mirroring upstream's workflow.
    assert jax_driver.value == Driver.SCIPY_LBFGSB.value

    # Both lanes satisfy the shared official 1e-15 stopping tolerance inside the
    # official 300-iteration cap, so both report their own convergence rather
    # than a stop imposed by a branch-chosen budget.
    native_status = _terminal_status(native)
    jax_status = _terminal_status(jax_lane)
    assert native_status.success is True
    assert jax_status.success is True
    assert native_status.normalized_status == jax_status.normalized_status
    assert native_status.normalized_status == "converged"
    assert native.iterations < MAX_STEPS
    assert jax_lane.iterations < MAX_STEPS
    assert set(native.values) == set(jax_lane.values)

    for observable in _STATE_OBSERVABLES:
        np.testing.assert_allclose(
            jax_lane.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=1.0e-8,
            atol=1.0e-10,
        )

    for run in (native, jax_lane):
        assert run.values["final:objective"] < run.values["initial:objective"]
        assert (
            np.linalg.norm(run.values["final:objective_gradient"], ord=np.inf) <= 1.0e-4
        )
        assert run.values["final:total_curve_length"] <= 1.1 * LENGTH_TARGET

    np.testing.assert_allclose(
        jax_lane.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-3,
        atol=1.0e-10,
    )
