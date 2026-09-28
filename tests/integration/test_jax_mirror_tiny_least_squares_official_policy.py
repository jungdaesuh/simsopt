"""One official stopping rule and one provider outcome for the tiny LS trio.

Every official fact asserted here is read from the tracked official-reference
fixture ``examples/jax/parity/official_reference`` (upstream
``9e027eac38028d57aa23777be52a781aa860e347``): never pasted as a literal and
never read from the git-ignored campaign capture tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment
from examples.jax.parity.cases import (
    get_case,
    native_just_a_quadratic,
    native_minimize_curve_length,
    native_surf_vol_area,
)
from examples.jax.parity.cases._official_least_squares import (
    solve_official_least_squares,
)
from examples.jax.parity.input_bundle import load_input_bundle
from examples.jax.parity.official_reference import load_official_reference
from scipy.optimize import OptimizeResult, least_squares
from simsopt.objectives import LeastSquaresProblem
from simsopt.objectives.functions import Identity
from simsopt.solve import serial as official_serial
from simsopt_contracts.optimization_endpoint import (
    SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD,
)
from simsopt_jax.examples.official_tiny_least_squares import (
    CONTROLLED_CURVE_DRAWS_BEFORE_START,
    CONTROLLED_CURVE_INITIAL_FULL,
    CONTROLLED_CURVE_REPLAY_SEED,
    TRF_STATUS_CONVENTION,
    official_default_max_nfev,
    trf_outcome,
)
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OfficialNativeOutcome:
    """The provider outcome one official capture records for one case."""

    case_id: str
    module: ModuleType
    raw_status: str
    nfev: int
    njev: int
    bounded_keywords: tuple[dict[str, float | int | str], ...]


def _official_provider_outcome(case_id: str) -> tuple[str, int, int]:
    """The official capture's own joined stop reason and summed counters.

    The native lane joins one ``"<status> <message>"`` per ``least_squares``
    call with ``" | "``; this rebuilds the same string from the tracked
    fixture's ``provider_calls`` so the expectation is the official run rather
    than a literal transcribed next to it.
    """
    calls = load_official_reference(case_id).provider_calls
    return (
        " | ".join(
            f"{int(call.result['status'])} {call.result['message']}" for call in calls
        ),
        sum(int(call.result["nfev"]) for call in calls),
        sum(int(call.result["njev"]) for call in calls),
    )


OFFICIAL_NATIVE_OUTCOMES = (
    OfficialNativeOutcome(
        "native-just-a-quadratic",
        native_just_a_quadratic,
        *_official_provider_outcome("native-just-a-quadratic"),
        ({"max_nfev": 32},),
    ),
    OfficialNativeOutcome(
        "native-minimize-curve-length",
        native_minimize_curve_length,
        *_official_provider_outcome("native-minimize-curve-length"),
        ({"max_nfev": 512},),
    ),
    OfficialNativeOutcome(
        "native-surf-vol-area",
        native_surf_vol_area,
        *_official_provider_outcome("native-surf-vol-area"),
        ({"max_nfev": 64}, {"diff_method": "centered", "max_nfev": 64}),
    ),
)


def _quadratic_problem() -> LeastSquaresProblem:
    identities = tuple(Identity() for _ in range(3))
    return LeastSquaresProblem.from_tuples(
        [
            (identity.f, target, weight)
            for identity, target, weight in zip(
                identities, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), strict=True
            )
        ]
    )


def test_official_wrapper_outcome_is_recorded_and_the_binding_restored(
    tmp_path: Path,
) -> None:
    """The wrapper discards SciPy's result; the recorder publishes it."""
    original_least_squares = official_serial.least_squares
    with chdir(tmp_path):
        result = solve_official_least_squares(_quadratic_problem())

    assert isinstance(result, OptimizeResult)
    assert result.status == 1
    assert result.message == "`gtol` termination condition is satisfied."
    assert (result.nfev, result.njev) == (4, 4)
    np.testing.assert_allclose(
        result.x,
        (1.0, 1.9999999992096138, 2.9999999998773283),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert official_serial.least_squares is original_least_squares


def test_official_wrapper_binding_is_restored_when_the_solve_raises(
    tmp_path: Path,
) -> None:
    original_least_squares = official_serial.least_squares
    with chdir(tmp_path), pytest.raises(ValueError):
        solve_official_least_squares(_quadratic_problem(), max_nfev=0)

    assert official_serial.least_squares is original_least_squares


@pytest.mark.parametrize(
    "official",
    OFFICIAL_NATIVE_OUTCOMES,
    ids=lambda official: official.case_id,
)
def test_native_lane_publishes_the_official_provider_outcome(
    official: OfficialNativeOutcome,
    tmp_path: Path,
) -> None:
    """At the official scale the native lane reports what upstream reported."""
    case = get_case(official.case_id)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "native_default")
    _, arrays = load_input_bundle(input_root, bundle)

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = case.execute("native-cpu", bundle, arrays)

    assert native.normalized_status == "converged"
    assert native.success is True
    assert native.raw_status == official.raw_status
    assert (native.nfev, native.njev) == (official.nfev, official.njev)


@pytest.mark.parametrize(
    "official",
    OFFICIAL_NATIVE_OUTCOMES,
    ids=lambda official: official.case_id,
)
def test_bounded_native_lane_passes_only_its_declared_work_budget(
    official: OfficialNativeOutcome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded scale adds a budget and never a second stopping rule."""
    case = get_case(official.case_id)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    calls: list[dict[str, float | int | str]] = []
    original_solve = official.module.solve_official_least_squares

    def recording_solve(
        problem: LeastSquaresProblem, **keywords: float | int | str
    ) -> OptimizeResult:
        calls.append(keywords)
        return original_solve(problem, **keywords)

    monkeypatch.setattr(
        official.module, "solve_official_least_squares", recording_solve
    )
    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        case.execute("native-cpu", bundle, arrays)

    assert tuple(calls) == official.bounded_keywords
    assert not {"ftol", "xtol", "gtol"} & {name for call in calls for name in call}


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


@pytest.mark.parametrize(
    "case_id", [official.case_id for official in OFFICIAL_NATIVE_OUTCOMES]
)
def test_case_configuration_declares_no_dead_tolerance(
    case_id: str, tmp_path: Path
) -> None:
    """A tolerance nobody reads must not enter the configuration fingerprint."""
    bundle = get_case(case_id).create_input(tmp_path / case_id, "bounded")

    assert not {"rtol", "atol"} & set(bundle.configuration)


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


def test_smoke_quadratic_completes_the_official_workflow() -> None:
    observables = _smoke_observables("just_a_quadratic.py")

    assert observables["solver_status"] == 1
    # The one counter pin that is deterministic by construction: the three
    # weighted residuals are affine in the three parameters, so TRF's
    # Gauss-Newton step is exact, the trajectory cannot depend on rounding and
    # the official capture's 4 evaluations are an invariant of the problem, not
    # of the build. The curve and surface counts below are not, and are
    # published rather than pinned.
    assert observables["function_evaluations"] == 4
    assert observables["jacobian_evaluations"] == 4


def test_smoke_curve_completes_the_official_workflow() -> None:
    """The official curve solve must run to its own `ftol` stop, uncapped.

    A smoke cap shows up as status ``0`` (``max_nfev`` exceeded), so the stop
    reason is what this asserts; the official capture's 62/55 counts came from
    a different residual implementation (simsopt's compiled curve length
    against this mirror's JAX quadrature) and are reported, not gated.
    """
    observables = _smoke_observables("minimize_curve_length.py")

    assert observables["optimizer_status"] == 2
    assert observables["solver_success"] is True
    assert observables["final_length"] == pytest.approx(
        18.849556246039942, rel=1.0e-9, abs=0.0
    )
    print(
        "native-minimize-curve-length smoke counters (nfev, njev): "
        f"{(observables['function_evaluations'], observables['gradient_evaluations'])}"
        " against the official capture's (62, 55)"
    )


def test_smoke_surface_completes_the_official_workflow() -> None:
    observables = _smoke_observables("surf_vol_area.py")

    assert observables["first_solver_status"] == 1
    assert observables["second_solver_status"] == 1
    # Reported, not gated, for the same reason as the curve: the official
    # capture's (9, 8) then (5, 5) are one implementation's trust-region
    # trajectory, while the end points below are the workflow's real contract.
    print(
        "native-surf-vol-area smoke counters (nfev, njev): "
        f"first={(observables['first_function_evaluations'], observables['first_jacobian_evaluations'])} "
        f"second={(observables['second_function_evaluations'], observables['second_jacobian_evaluations'])}"
        " against the official capture's (9, 8) and (5, 5)"
    )
    # Area and volume are invariant under exchanging the two ellipse semi-axes
    # rc(1,0) and zs(1,0), so the problem has a mirrored pair of solutions and a
    # solve started from the symmetric (0.1, 0.1) may land on either. The parity
    # harness compares these parameters through
    # ``examples/jax/parity/symmetry.parameter_invariants`` for the same reason;
    # here the end point is compared up to that exchange.
    np.testing.assert_allclose(
        np.sort(observables["first_solution"]),
        np.sort((0.27727411213693803, 0.10962565115956309)),
        rtol=1.0e-9,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.sort(observables["second_solution"]),
        np.sort((0.30617894188519335, 0.13236858553169742)),
        rtol=1.0e-9,
        atol=0.0,
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


def test_controlled_curve_start_is_the_official_captures_own_start() -> None:
    """The nine curve start values are the ones upstream actually started from.

    Without this the regeneration check below is self-consistent only: a recipe
    fitted to a constant reproduces that constant whatever the constant is.
    The anchor is the official capture's recorded ``initial:full_parameters``,
    read from the tracked fixture, and it is asserted bit for bit.
    """
    official = load_official_reference("native-minimize-curve-length").array(
        "initial:full_parameters"
    )

    assert official.shape == (len(CONTROLLED_CURVE_INITIAL_FULL),)
    assert (
        official.tobytes()
        == np.asarray(CONTROLLED_CURVE_INITIAL_FULL, dtype=np.float64).tobytes()
    )


def test_controlled_curve_start_regenerates_from_its_seed() -> None:
    """The nine curve start values are regenerated from the recipe, numpy only.

    ``CONTROLLED_CURVE_INITIAL_FULL`` is the realization the official capture
    ran: the capture seeds the legacy generator immediately before the verbatim
    official body, whose own ``import`` statements consume
    ``CONTROLLED_CURVE_DRAWS_BEFORE_START`` draws before
    ``x0 = np.random.rand(curve.dof_size) - 0.5`` with ``x0[0] = 3.0``. This
    reproduces those nine numbers bit for bit without importing simsopt, and
    the test above ties them to the official capture rather than to the recipe.

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
