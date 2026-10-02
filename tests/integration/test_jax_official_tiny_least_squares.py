"""The official tiny least-squares trio: JAX mirrors against live native SIMSOPT.

``just_a_quadratic.py``, ``minimize_curve_length.py`` and ``surf_vol_area.py``
solve their problems through ``simsopt_jax.examples.official_tiny_least_squares``.
Each workflow is run here twice at the same small scale: natively, through the
official ``least_squares_serial_solve`` wrapper, and through the JAX residual
with the same SciPy TRF policy. The native run is the live reference; nothing
is read from a stored capture.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
import os
import subprocess
import sys
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment
from scipy.optimize import OptimizeResult, least_squares
from simsopt import load
from simsopt.geo import CurveLength, CurveRZFourier, SurfaceRZFourier
from simsopt.objectives import LeastSquaresProblem
from simsopt.objectives.functions import Identity
from simsopt.solve import least_squares_serial_solve
from simsopt.solve import serial as official_serial
from simsopt_contracts.optimization_endpoint import (
    SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD,
)
from simsopt_jax.examples.official_tiny_least_squares import (
    CONTROLLED_CURVE_DRAWS_BEFORE_START,
    CONTROLLED_CURVE_INITIAL_FULL,
    CONTROLLED_CURVE_REPLAY_SEED,
    TRF_STATUS_CONVENTION,
    TrfOutcome,
    combine_trf_outcomes,
    curve_length_residual,
    guard_finite_endpoint,
    official_default_max_nfev,
    quadratic_residual,
    solve_jax_residual,
    surface_area_volume_residual,
    trf_outcome,
    value_and_jacobian,
)
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]

QUADRATIC_TARGETS = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
QUADRATIC_WEIGHTS = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
#: Small-scale evaluation budgets; each only caps, SciPy's defaults decide the stop.
QUADRATIC_MAX_NFEV = 32
CURVE_MAX_NFEV = 512
SURFACE_MAX_NFEV = 64
#: The coarse grid the surface workflow is compared on (phi and theta alike).
SURFACE_QUADRATURE = np.linspace(0.0, 1.0, 32, endpoint=False)
SURFACE_INITIAL = (0.1, 0.1)
SURFACE_STAGE_TARGETS = ((8.0, 0.6), (9.0, 0.8))

GTOL_STOP = "1 `gtol` termination condition is satisfied."
FTOL_STOP = "2 `ftol` termination condition is satisfied."

Values = dict[str, np.ndarray]


def _jax_parity_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")


def _official_solve(
    problem: LeastSquaresProblem, **keywords: float | int | str
) -> OptimizeResult:
    """Run the official wrapper unchanged and keep the SciPy result it discards."""
    recorded: list[OptimizeResult] = []

    def recording_least_squares(
        objective: Callable[[np.ndarray], np.ndarray],
        initial: np.ndarray,
        **inner_keywords: object,
    ) -> OptimizeResult:
        result = least_squares(objective, initial, **inner_keywords)
        recorded.append(result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(official_serial, "least_squares", recording_least_squares)
        least_squares_serial_solve(problem, **keywords)
    (result,) = recorded
    return result


def _budget(max_nfev: int | None) -> dict[str, int]:
    return {} if max_nfev is None else {"max_nfev": max_nfev}


# --- just_a_quadratic -------------------------------------------------------------


def _quadratic_values(
    initial_parameters: np.ndarray, final_parameters: np.ndarray
) -> Values:
    residual_jacobian = np.diag(np.sqrt(QUADRATIC_WEIGHTS))

    def state(prefix: str, parameters: np.ndarray) -> Values:
        residual = np.sqrt(QUADRATIC_WEIGHTS) * (parameters - QUADRATIC_TARGETS)
        return {
            f"{prefix}:parameters": parameters,
            f"{prefix}:residual": residual,
            f"{prefix}:residual_jacobian": residual_jacobian,
            f"{prefix}:objective_sum_squares": np.asarray(
                np.dot(residual, residual), dtype=np.float64
            ),
            f"{prefix}:objective_gradient": 2.0 * residual_jacobian.T @ residual,
        }

    return {**state("initial", initial_parameters), **state("final", final_parameters)}


def _native_quadratic() -> tuple[Values, TrfOutcome]:
    initial = np.zeros(3, dtype=np.float64)
    identities = tuple(Identity() for _ in initial)
    problem = LeastSquaresProblem.from_tuples(
        [
            (identity.f, cast(Real, float(target)), cast(Real, float(weight)))
            for identity, target, weight in zip(
                identities, QUADRATIC_TARGETS, QUADRATIC_WEIGHTS, strict=True
            )
        ]
    )
    result = _official_solve(problem, **_budget(QUADRATIC_MAX_NFEV))
    values = _quadratic_values(initial, np.asarray(problem.x, dtype=np.float64))
    return values, guard_finite_endpoint(trf_outcome(result), values.values())


def _jax_quadratic() -> tuple[Values, TrfOutcome]:
    initial = np.zeros(3, dtype=np.float64)
    optimizer = solve_jax_residual(
        quadratic_residual(QUADRATIC_TARGETS, QUADRATIC_WEIGHTS),
        initial,
        max_nfev=QUADRATIC_MAX_NFEV,
    )
    values = _quadratic_values(initial, np.asarray(optimizer.x, dtype=np.float64))
    return values, guard_finite_endpoint(trf_outcome(optimizer), values.values())


def test_quadratic_mirror_solve_matches_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``least_squares_serial_solve`` writes its objective log into the cwd.
    monkeypatch.chdir(tmp_path)
    native_values, native = _native_quadratic()
    _jax_parity_environment(monkeypatch)
    jax_values, jax = _jax_quadratic()

    assert native.success is True
    assert jax.success is True
    # One stopping rule for both lanes: the official SciPy defaults.
    assert native.normalized_status == jax.normalized_status == "converged"
    assert native.raw_status == jax.raw_status == GTOL_STOP
    # Both lanes are pinned to 4 evaluations because this problem's three
    # weighted residuals are affine in its three parameters: TRF's Gauss-Newton
    # step is exact, so the trajectory cannot depend on rounding and 4 is an
    # invariant of the problem rather than of a build.
    assert (native.nfev, native.njev) == (4, 4)
    assert (jax.nfev, jax.njev) == (4, 4)
    for name in native_values:
        np.testing.assert_allclose(
            jax_values[name], native_values[name], rtol=1.0e-10, atol=1.0e-12
        )
    # SciPy stops on `gtol` ~8e-10 from the exact minimizer, so the mirror is
    # held to the native end point rather than to (1, 2, 3).
    np.testing.assert_allclose(
        jax_values["final:parameters"],
        native_values["final:parameters"],
        rtol=1.0e-12,
        atol=0.0,
    )


# --- minimize_curve_length --------------------------------------------------------


def _controlled_curve() -> CurveRZFourier:
    curve = CurveRZFourier(100, 4, 5, True)
    curve.x = np.asarray(CONTROLLED_CURVE_INITIAL_FULL, dtype=np.float64)
    curve.fix(0)
    return curve


def _curve_state(
    prefix: str, parameters: np.ndarray, length: float, jacobian: np.ndarray
) -> Values:
    return {
        f"{prefix}:parameters": parameters,
        f"{prefix}:length": np.asarray(length, dtype=np.float64),
        f"{prefix}:residual": np.asarray((length,), dtype=np.float64),
        f"{prefix}:residual_jacobian": jacobian[np.newaxis, :],
        f"{prefix}:objective_sum_squares": np.asarray(
            length * length, dtype=np.float64
        ),
        f"{prefix}:objective_gradient": 2.0 * length * jacobian,
    }


def _native_curve(max_nfev: int | None) -> tuple[Values, TrfOutcome]:
    curve = _controlled_curve()
    objective = CurveLength(curve)
    problem = LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
    initial = _curve_state(
        "initial",
        np.asarray(problem.x, dtype=np.float64),
        float(objective.J()),
        np.asarray(objective.dJ(), dtype=np.float64),
    )
    result = _official_solve(problem, **_budget(max_nfev))
    final = _curve_state(
        "final",
        np.asarray(problem.x, dtype=np.float64),
        float(objective.J()),
        np.asarray(objective.dJ(), dtype=np.float64),
    )
    values = {**initial, **final}
    return values, guard_finite_endpoint(trf_outcome(result), values.values())


def _jax_curve() -> tuple[Values, TrfOutcome]:
    curve = _controlled_curve()
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    residual = curve_length_residual(
        full_dofs,
        np.asarray(curve.quadpoints, dtype=np.float64),
        free_positions,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )
    initial_parameters = full_dofs[free_positions]
    initial_value, initial_jacobian = value_and_jacobian(residual, initial_parameters)
    optimizer = solve_jax_residual(
        residual, initial_parameters, max_nfev=CURVE_MAX_NFEV
    )
    final_parameters = np.asarray(optimizer.x, dtype=np.float64)
    final_value, final_jacobian = value_and_jacobian(residual, final_parameters)
    values = {
        **_curve_state(
            "initial", initial_parameters, float(initial_value[0]), initial_jacobian[0]
        ),
        **_curve_state(
            "final", final_parameters, float(final_value[0]), final_jacobian[0]
        ),
    }
    return values, guard_finite_endpoint(trf_outcome(optimizer), values.values())


def test_curve_length_mirror_solve_matches_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    native_values, native = _native_curve(CURVE_MAX_NFEV)
    _jax_parity_environment(monkeypatch)
    jax_values, jax = _jax_curve()

    assert native.success is True
    assert jax.success is True
    # One stopping rule for both lanes: the official SciPy defaults.
    assert native.normalized_status == jax.normalized_status == "converged"
    assert native.raw_status == jax.raw_status == FTOL_STOP
    for name in (
        "initial:parameters",
        "initial:length",
        "initial:residual",
        "initial:residual_jacobian",
        "initial:objective_sum_squares",
        "initial:objective_gradient",
        "final:length",
        "final:residual",
        "final:objective_sum_squares",
    ):
        np.testing.assert_allclose(
            jax_values[name], native_values[name], rtol=1.0e-9, atol=1.0e-11
        )
    # SciPy stops on `ftol` ~3e-07 short of the circle oracle 6*pi, so the
    # circle is a diagnostic and the native end point is the assertion.
    np.testing.assert_allclose(
        jax_values["final:length"],
        native_values["final:length"],
        rtol=1.0e-9,
        atol=0.0,
    )


# --- surf_vol_area ----------------------------------------------------------------


def _parameter_invariants(parameters: np.ndarray) -> np.ndarray:
    """Quotient coordinates of two interchangeable ellipse semi-axes."""
    values = np.asarray(parameters, dtype=np.float64)
    return np.asarray((np.sum(values), np.prod(values)), dtype=np.float64)


def _column_swap_jacobian_invariants(jacobian: np.ndarray) -> np.ndarray:
    """Identify the two Jacobian columns up to one global column exchange.

    Area and volume are invariant under exchanging rc(1,0) and zs(1,0), so a
    solve from the symmetric start may land on either mirrored solution. The
    normalized difference association couples every residual row, and
    differences at roundoff scale count as exact coincidence so the
    normalization does not amplify an arbitrary sign.
    """
    left, right = jacobian[:, 0], jacobian[:, 1]
    difference = left - right
    difference_scale = np.max(np.abs(difference))
    value_scale = np.max(np.abs(left)) + np.max(np.abs(right))
    roundoff_threshold = (
        128.0 * np.finfo(left.dtype).eps * (value_scale + (value_scale == 0))
    )
    normalized = (difference / (difference_scale + (difference_scale == 0))) * (
        difference_scale > roundoff_threshold
    )
    return np.concatenate(
        (
            left + right,
            left * right,
            (normalized[:, None] * normalized[None, :]).reshape(-1),
        )
    )


def _surface_state(
    prefix: str,
    parameters: np.ndarray,
    area: float,
    volume: float,
    residual: np.ndarray,
    jacobian: np.ndarray,
) -> Values:
    objective_gradient = 2.0 * jacobian.T @ residual
    return {
        f"{prefix}:parameters": parameters,
        f"{prefix}:parameter_invariants": _parameter_invariants(parameters),
        f"{prefix}:area": np.asarray(area, dtype=np.float64),
        f"{prefix}:volume": np.asarray(volume, dtype=np.float64),
        f"{prefix}:residual": residual,
        f"{prefix}:residual_jacobian_invariants": _column_swap_jacobian_invariants(
            jacobian
        ),
        f"{prefix}:objective_sum_squares": np.asarray(
            np.vdot(residual, residual), dtype=np.float64
        ),
        f"{prefix}:objective_gradient_invariants": _parameter_invariants(
            objective_gradient
        ),
    }


def _coarse_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=0,
        nfp=1,
        stellsym=True,
        quadpoints_phi=SURFACE_QUADRATURE,
        quadpoints_theta=SURFACE_QUADRATURE,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, SURFACE_INITIAL[0])
    surface.set_zs(1, 0, SURFACE_INITIAL[1])
    surface.fix("rc(0,0)")
    return surface


def _official_surface() -> SurfaceRZFourier:
    """The official example's surface: default grid and start, rc(0,0) fixed."""
    surface = SurfaceRZFourier()
    surface.fix("rc(0,0)")
    return surface


def _roundtrip(surface: SurfaceRZFourier, path: Path) -> SurfaceRZFourier:
    surface.save(str(path), indent=2)
    return load(str(path))


def _native_surface_stage(
    surface: SurfaceRZFourier,
    targets: np.ndarray,
    prefix: str,
    max_nfev: int | None,
    **method: str,
) -> tuple[Values, TrfOutcome]:
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    problem = LeastSquaresProblem.from_tuples(
        (
            (surface.area, float(targets[0]), 1.0),
            (surface.volume, float(targets[1]), 1.0),
        )
    )

    def state(phase: str) -> Values:
        area = float(surface.area())
        volume = float(surface.volume())
        residual = np.asarray((area, volume), dtype=np.float64) - targets
        jacobian = np.stack(
            (
                np.asarray(surface.darea(), dtype=np.float64)[free_positions],
                np.asarray(surface.dvolume(), dtype=np.float64)[free_positions],
            )
        )
        return _surface_state(
            f"{prefix}:{phase}",
            np.asarray(problem.x, dtype=np.float64),
            area,
            volume,
            residual,
            jacobian,
        )

    initial = state("initial")
    result = _official_solve(problem, **method, **_budget(max_nfev))
    values = {**initial, **state("final")}
    return values, guard_finite_endpoint(trf_outcome(result), values.values())


def _native_surface_workflow(
    surface: SurfaceRZFourier, directory: Path, max_nfev: int | None
) -> tuple[Values, TrfOutcome]:
    """Both official solves; the second, like upstream's, passes ``diff_method``."""
    first_targets, second_targets = np.asarray(SURFACE_STAGE_TARGETS)
    first_values, first = _native_surface_stage(
        surface, first_targets, "first", max_nfev
    )
    second_surface = _roundtrip(surface, directory / "surf_fw.json")
    second_values, second = _native_surface_stage(
        second_surface, second_targets, "second", max_nfev, diff_method="centered"
    )
    return {**first_values, **second_values}, combine_trf_outcomes(first, second)


def _jax_surface_stage(
    surface: SurfaceRZFourier, targets: np.ndarray, prefix: str
) -> tuple[Values, TrfOutcome]:
    residual = surface_area_volume_residual(
        np.asarray(surface.local_full_x, dtype=np.float64),
        np.asarray(surface.quadpoints_phi, dtype=np.float64),
        np.asarray(surface.quadpoints_theta, dtype=np.float64),
        np.flatnonzero(surface.local_dofs_free_status),
        targets,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    initial_parameters = np.asarray(surface.x, dtype=np.float64)
    initial_residual, initial_jacobian = value_and_jacobian(
        residual, initial_parameters
    )
    optimizer = solve_jax_residual(
        residual, initial_parameters, max_nfev=SURFACE_MAX_NFEV
    )
    final_parameters = np.asarray(optimizer.x, dtype=np.float64)
    final_residual, final_jacobian = value_and_jacobian(residual, final_parameters)
    surface.x = final_parameters
    values = {
        **_surface_state(
            f"{prefix}:initial",
            initial_parameters,
            float(initial_residual[0] + targets[0]),
            float(initial_residual[1] + targets[1]),
            initial_residual,
            initial_jacobian,
        ),
        **_surface_state(
            f"{prefix}:final",
            final_parameters,
            float(final_residual[0] + targets[0]),
            float(final_residual[1] + targets[1]),
            final_residual,
            final_jacobian,
        ),
    }
    return values, guard_finite_endpoint(trf_outcome(optimizer), values.values())


def _jax_surface_workflow(directory: Path) -> tuple[Values, TrfOutcome]:
    first_targets, second_targets = np.asarray(SURFACE_STAGE_TARGETS)
    surface = _coarse_surface()
    first_values, first = _jax_surface_stage(surface, first_targets, "first")
    second_surface = _roundtrip(surface, directory / "surf_fw.json")
    second_values, second = _jax_surface_stage(second_surface, second_targets, "second")
    return {**first_values, **second_values}, combine_trf_outcomes(first, second)


def test_surf_vol_area_mirror_solves_match_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "native").mkdir()
    (tmp_path / "jax").mkdir()
    native_values, native = _native_surface_workflow(
        _coarse_surface(), tmp_path / "native", SURFACE_MAX_NFEV
    )
    _jax_parity_environment(monkeypatch)
    jax_values, jax = _jax_surface_workflow(tmp_path / "jax")

    assert native.success is True
    assert jax.success is True
    # One stopping rule for both lanes: the official SciPy defaults.
    assert native.normalized_status == jax.normalized_status == "converged"
    assert native.raw_status == jax.raw_status == f"{GTOL_STOP} | {GTOL_STOP}"
    # Stop reason (above) and end point (below), not counters: both lanes run
    # ``jac="2-point"`` over independent residual implementations, so the
    # trust-region trial sequence is decided by rounding-level differences and
    # equal counts are a coincidence, not an invariant. Published instead.
    print(
        "native-surf-vol-area counters (nfev, njev): "
        f"native={(native.nfev, native.njev)} jax={(jax.nfev, jax.njev)}"
    )
    for stage in ("first", "second"):
        for phase in ("initial", "final"):
            for observable in (
                "parameter_invariants",
                "area",
                "volume",
                "residual",
                "residual_jacobian_invariants",
                "objective_sum_squares",
                "objective_gradient_invariants",
            ):
                name = f"{stage}:{phase}:{observable}"
                np.testing.assert_allclose(
                    jax_values[name], native_values[name], rtol=1.0e-8, atol=1.0e-10
                )
    for stage in ("first", "second"):
        np.testing.assert_allclose(
            native_values[f"{stage}:final:residual"],
            np.zeros(2),
            rtol=0.0,
            atol=1.0e-8,
        )


# --- the TRF outcome contract -----------------------------------------------------


def _least_squares_result(status: int, *, success: bool) -> OptimizeResult:
    """A ``least_squares`` result carrying one status, everything else finite."""
    return OptimizeResult(
        status=status,
        success=success,
        nfev=9,
        njev=4,
        x=np.zeros(3),
        fun=np.zeros(2),
        message=f"status {status}",
    )


def test_a_status_this_emitter_does_not_define_is_a_failure_not_a_budget_stop() -> None:
    """An unknown ``least_squares`` status must never become ``budget_exhausted``.

    ``least_squares`` reports NO iteration count, so ``trf_outcome`` used to
    hand the certificate ``nfev`` against itself as a validity guard. That made
    ``iterations >= max_iterations`` always true, and the contract's terminal
    arm turned EVERY status outside the ``scipy-trf`` table into
    ``iteration-limit`` -> ``budget_exhausted``: a fabricated budget stop for an
    emitter whose vocabulary has no iteration limit. Measured before the fix
    (installed SciPy 1.17.1): statuses 5, 7 and -3 all reported
    ``budget_exhausted``.

    The statuses SciPy 1.17.1 does define are covered by
    ``tests/test_scipy_status_convention_coverage.py``; this pins what happens
    to one it does not.
    """
    for status in (5, 7, -3):
        outcome = trf_outcome(_least_squares_result(status, success=False))
        assert outcome.normalized_status == "failed", status
        assert outcome.success is False

    # The real budget signal is untouched: status 0 is this emitter's own
    # evaluation limit (``trf.py:405-406``), and a convergence status still
    # converges.
    assert (
        trf_outcome(_least_squares_result(0, success=False)).normalized_status
        == "budget_exhausted"
    )
    assert (
        trf_outcome(_least_squares_result(2, success=True)).normalized_status
        == "converged"
    )
    # A success flag that contradicts the reported status fails closed.
    assert (
        trf_outcome(_least_squares_result(0, success=True)).normalized_status
        == "failed"
    )


def test_this_module_names_an_emitter_the_contract_knows() -> None:
    """The convention this module declares is one the contract carries.

    ``SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD`` is the contract's map for
    ``scipy.optimize.minimize`` methods; ``least_squares`` is a different entry
    point with its own vocabulary, which is why this module names
    ``scipy-trf`` directly instead of looking a method up there.
    """
    assert TRF_STATUS_CONVENTION == "scipy-trf"
    assert TRF_STATUS_CONVENTION not in set(
        SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD.values()
    )


@pytest.mark.parametrize("parameter_count", (1, 2, 3))
def test_official_default_budget_is_scipys_own(parameter_count: int) -> None:
    """SciPy's real default budget, observed instead of restated.

    ``scipy/optimize/_lsq/trf.py:250`` and ``:453`` compute
    ``max_nfev = x0.size * 100`` when the caller passes none. The observation
    is a run that cannot converge -- ``exp(-x/100)`` has no minimizer, and with
    ``gtol``/``xtol`` disabled and ``ftol`` at machine epsilon no convergence
    criterion can fire -- so the only stop left is the budget SciPy computed
    for itself. ``max_nfev`` is never passed. If SciPy changed the factor, this
    fails while a comparison against ``100 * n`` could not.
    """
    calls: list[np.ndarray] = []

    def residual(parameters: np.ndarray) -> np.ndarray:
        calls.append(np.array(parameters, copy=True))
        return np.exp(-np.asarray(parameters, dtype=np.float64) / 100.0)

    observed = least_squares(
        residual,
        np.zeros(parameter_count),
        ftol=np.finfo(np.float64).eps * 4.0,
        xtol=None,
        gtol=None,
    )

    # Status, not message text: SciPy's wording is a version detail
    # (``pyproject.toml`` allows ``scipy>=1.13``), while ``status == 0`` is
    # ``least_squares``' own budget signal and is what this observes.
    assert observed.status == 0
    assert observed.nfev == official_default_max_nfev(parameter_count)
    # SciPy counts one ``nfev`` per trial point and caches repeated points, so
    # the residual is called at least once per counted evaluation here.
    assert len(calls) >= observed.nfev


def test_controlled_curve_start_regenerates_from_its_seed() -> None:
    """The nine curve start values are regenerated from the recipe, numpy only.

    ``CONTROLLED_CURVE_INITIAL_FULL`` is one realization of the unseeded
    official body: the legacy generator is seeded, the official body's own
    ``import`` statements consume ``CONTROLLED_CURVE_DRAWS_BEFORE_START``
    draws, then ``x0 = np.random.rand(curve.dof_size) - 0.5`` with
    ``x0[0] = 3.0``. This reproduces those nine numbers bit for bit without
    importing simsopt.

    A LOCAL ``RandomState`` rather than ``np.random.seed``: the same MT19937
    stream (the legacy global functions are methods of a hidden ``RandomState``)
    without leaving every later test in this process with a seeded global
    generator.
    """
    generator = np.random.RandomState(CONTROLLED_CURVE_REPLAY_SEED)
    generator.rand(CONTROLLED_CURVE_DRAWS_BEFORE_START)
    regenerated = generator.rand(len(CONTROLLED_CURVE_INITIAL_FULL)) - 0.5
    regenerated[0] = 3.0

    assert tuple(regenerated) == CONTROLLED_CURVE_INITIAL_FULL


# --- the shipped scripts, against the live native workflow --------------------------


def _source_checkout_environment() -> dict[str, str]:
    _, environment = build_execution_environment(
        "cpu",
        "fast",
        os.environ,
        repo_root=REPO_ROOT,
    )
    environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
        *environment["PYTHONPATH"].split(os.pathsep)
    )
    return environment


def _smoke_observables(script: str) -> dict[str, object]:
    example = REPO_ROOT / "examples" / "jax" / "1_Simple" / script
    completed = subprocess.run(
        (sys.executable, "-S", str(example), "--smoke", "--json"),
        cwd=REPO_ROOT,
        env=_source_checkout_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    return payload["observables"]


def test_smoke_curve_completes_the_official_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official curve solve must run to its own `ftol` stop, uncapped.

    A smoke cap shows up as status ``0`` (``max_nfev`` exceeded), so the stop
    reason is what this asserts, and the end point is the native official
    workflow's, run here with SciPy's own budget. The counters come from two
    residual implementations (simsopt's compiled curve length against this
    mirror's JAX quadrature) and are reported, not gated.
    """
    observables = _smoke_observables("minimize_curve_length.py")
    monkeypatch.chdir(tmp_path)
    native_values, native = _native_curve(None)

    assert observables["optimizer_status"] == 2
    assert observables["solver_success"] is True
    assert observables["final_length"] == pytest.approx(
        float(native_values["final:length"]), rel=1.0e-9, abs=0.0
    )
    print(
        "native-minimize-curve-length smoke counters (nfev, njev): "
        f"{(observables['function_evaluations'], observables['gradient_evaluations'])}"
        f" against the native run's {(native.nfev, native.njev)}"
    )


def test_smoke_surface_completes_the_official_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observables = _smoke_observables("surf_vol_area.py")
    monkeypatch.chdir(tmp_path)
    native_values, native = _native_surface_workflow(
        _official_surface(), tmp_path, None
    )

    assert observables["first_solver_status"] == 1
    assert observables["second_solver_status"] == 1
    # Reported, not gated, for the same reason as the curve: each count is one
    # implementation's trust-region trajectory, while the end points below are
    # the workflow's real contract.
    print(
        "native-surf-vol-area smoke counters (nfev, njev): "
        f"first={(observables['first_function_evaluations'], observables['first_jacobian_evaluations'])} "
        f"second={(observables['second_function_evaluations'], observables['second_jacobian_evaluations'])}"
        f" against the native run's {(native.nfev, native.njev)}"
    )
    # Area and volume are invariant under exchanging the two ellipse semi-axes
    # rc(1,0) and zs(1,0), so the problem has a mirrored pair of solutions and a
    # solve started from the symmetric (0.1, 0.1) may land on either; the end
    # point is compared up to that exchange.
    np.testing.assert_allclose(
        np.sort(observables["first_solution"]),
        np.sort(native_values["first:final:parameters"]),
        rtol=1.0e-9,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.sort(observables["second_solution"]),
        np.sort(native_values["second:final:parameters"]),
        rtol=1.0e-9,
        atol=0.0,
    )
