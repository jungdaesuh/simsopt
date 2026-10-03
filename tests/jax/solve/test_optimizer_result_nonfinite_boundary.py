"""The shared result boundary: a non-finite returned state is never a success.

Every ``minimize`` / ``least_squares`` driver returns through one conversion
(``dispatch._public_result``).  When the returned ``x``, ``fun`` or the returned
derivative/residual data (``jac``, ``residual``) is non-finite, the result is
unsuccessful with ``NONFINITE_RESULT_STATUS``, names the non-finite fields, keeps
the backend's own status, success flag and message in the raw fields, and keeps
the backend's returned state itself (no substitute point).  A finite result
passes through unchanged, bit for bit.  Only the returned state is judged: an
invalid trial the solver evaluated and then recovered from does not poison it.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize
from simsopt_jax.solve import (
    NONFINITE_RESULT_STATUS,
    STATUS_CODES,
    Driver,
    OptimizerResult,
    ScipyBFGSOptions,
    ScipyBounds,
    ScipyLBFGSBOptions,
)
from simsopt_jax.solve.dispatch import (
    _LEAST_SQUARES_OPTIONS,
    _MINIMIZE_OPTIONS,
    _public_result,
    minimize,
)
from simsopt_jax.solve.termination import (
    Emitter,
    EmitterStop,
    TerminationLabel,
    termination_record_defects,
    termination_report,
    verify_termination_evidence,
)

_CONVERGED = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
_LBFGSB_OPTIONS = ScipyLBFGSBOptions(maxiter=50)


def _backend(**fields: object) -> OptimizeResult:
    base: dict[str, object] = {
        "x": np.array([1.0, 2.0]),
        "fun": 3.0,
        "jac": np.array([0.0, 0.0]),
        "nit": 4,
        "nfev": 5,
        "njev": 5,
        "status": 0,
        "success": True,
        "message": _CONVERGED,
    }
    base.update(fields)
    return OptimizeResult(**base)


_OPTIONS_BY_DRIVER = {**_MINIMIZE_OPTIONS, **_LEAST_SQUARES_OPTIONS}


def _convert(
    backend: OptimizeResult, driver: Driver = Driver.SCIPY_LBFGSB
) -> OptimizerResult:
    """The shared conversion, under ``driver``'s own default options."""
    result = _public_result(
        backend,
        driver=driver,
        options_used=_OPTIONS_BY_DRIVER[driver](),
        wallclock_s=0.0,
    )
    assert result.driver is driver
    return result


def test_the_conversion_fixtures_cover_every_driver() -> None:
    assert set(_OPTIONS_BY_DRIVER) == set(Driver)


def _same_array(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.dtype == right.dtype and left.tobytes() == right.tobytes()


def _same_float(left: float, right: float) -> bool:
    return math.copysign(1.0, left) == math.copysign(1.0, right) and (
        left == right or (math.isnan(left) and math.isnan(right))
    )


def _assert_nonfinite(
    result: OptimizerResult,
    fields: tuple[str, ...],
    *,
    raw_status: int,
    raw_success: bool,
    raw_message: str,
) -> None:
    assert result.success is False
    assert result.status == NONFINITE_RESULT_STATUS
    assert result.nonfinite_fields == fields
    assert (result.raw_status, result.raw_success, result.raw_message) == (
        raw_status,
        raw_success,
        raw_message,
    )
    assert result.message == f"NONFINITE RESULT ({', '.join(fields)}): {raw_message}"


def _assert_unchanged(result: OptimizerResult, backend: OptimizeResult) -> None:
    assert (result.status, result.success, result.message) == (
        backend.status,
        backend.success,
        backend.message,
    )
    assert result.nonfinite_fields == ()
    assert (result.raw_status, result.raw_success, result.raw_message) == (
        None,
        None,
        None,
    )


def _label(result: OptimizerResult) -> TerminationLabel:
    assert result.jac is not None
    return termination_report(
        EmitterStop(
            Emitter.SCIPY_LBFGSB, result.status, result.message, result.success
        ),
        x=result.x,
        fun=result.fun,
        gradient=result.jac,
        bounds=None,
        accepted_fun=(),
    ).label


# ---------------------------------------------------------------------------
# The status contract.


def test_the_nonfinite_status_is_in_every_driver_vocabulary_exactly_once() -> None:
    for driver, codes in STATUS_CODES.items():
        assert codes.count(NONFINITE_RESULT_STATUS) == 1, driver


# ---------------------------------------------------------------------------
# The conversion itself, on backend results of every shape.

_NAN = math.nan
_INF = math.inf


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"fun": _NAN}, ("fun",)),
        ({"fun": _INF}, ("fun",)),
        ({"fun": -_INF}, ("fun",)),
        ({"jac": np.array([0.0, _NAN])}, ("jac",)),
        ({"jac": np.array([_INF, 0.0])}, ("jac",)),
        ({"x": np.array([_NAN, 2.0])}, ("x",)),
        (
            {"x": np.array([_INF, 2.0]), "fun": _NAN, "jac": np.array([_NAN, 0.0])},
            ("x", "fun", "jac"),
        ),
        ({"residual": np.array([1.0, _NAN])}, ("residual",)),
        ({"jac": None, "fun": _NAN}, ("fun",)),
    ],
)
@pytest.mark.parametrize(
    ("status", "success", "message"),
    [(0, True, _CONVERGED), (2, False, "ABNORMAL: ")],
)
@pytest.mark.parametrize("driver", list(Driver), ids=lambda driver: driver.value)
def test_a_nonfinite_returned_state_is_never_a_success(
    fields: dict[str, object],
    expected: tuple[str, ...],
    status: int,
    success: bool,
    message: str,
    driver: Driver,
) -> None:
    backend = _backend(status=status, success=success, message=message, **fields)

    result = _convert(backend, driver)

    _assert_nonfinite(
        result, expected, raw_status=status, raw_success=success, raw_message=message
    )
    # The backend's returned state and counts, never a substitute.
    assert _same_array(result.x, np.asarray(backend.x, dtype=float))
    assert _same_float(result.fun, float(backend.fun))
    assert _same_array(
        result.jac, None if backend.jac is None else np.asarray(backend.jac)
    )
    assert (result.nit, result.nfev, result.njev) == (4, 5, 5)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"jac": None},
        {"residual": np.array([0.5, -0.5])},
        # Optional diagnostics are not the returned state.
        {"hessian": np.full((2, 2), _NAN)},
        {"status": 1, "success": False, "message": "STOP: TOTAL NO. OF ITERATIONS"},
    ],
)
@pytest.mark.parametrize("driver", list(Driver), ids=lambda driver: driver.value)
def test_a_finite_returned_state_passes_through_unchanged(
    fields: dict[str, object], driver: Driver
) -> None:
    backend = _backend(**fields)

    result = _convert(backend, driver)

    _assert_unchanged(result, backend)
    assert _same_array(result.x, np.asarray(backend.x, dtype=float))
    assert _same_float(result.fun, float(backend.fun))


# ---------------------------------------------------------------------------
# Real drivers through ``minimize`` / ``least_squares``.


def _direct_lbfgsb(fun, x0: np.ndarray, options: ScipyLBFGSBOptions) -> OptimizeResult:
    return scipy_minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": options.maxiter,
            "maxfun": options.maxfun,
            "gtol": options.gtol,
            "ftol": options.ftol,
            "maxcor": options.maxcor,
            "maxls": options.maxls,
        },
    )


def _assert_backend_state(result: OptimizerResult, direct: OptimizeResult) -> None:
    """The routed result holds SciPy's own returned state and counts, bit for bit."""
    assert _same_array(result.x, np.asarray(direct.x, dtype=float))
    assert _same_float(result.fun, float(direct.fun))
    assert _same_array(result.jac, np.asarray(direct.jac, dtype=float))
    assert (result.nit, result.nfev, result.njev) == (
        direct.nit,
        direct.nfev,
        direct.njev,
    )


def _infinite_value(x: np.ndarray) -> tuple[float, np.ndarray]:
    return math.inf, 2.0 * np.asarray(x, dtype=float)


def _nan_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
    return float(np.sum(np.asarray(x) ** 2)), np.full(np.shape(x), _NAN)


def test_lbfgsb_success_on_an_infinite_value_is_reported_nonfinite() -> None:
    """SciPy L-BFGS-B reports CONVERGENCE (status 0, success) with ``fun = inf``."""
    x0 = np.ones(2)

    result = minimize(
        _infinite_value, x0, driver=Driver.SCIPY_LBFGSB, options=_LBFGSB_OPTIONS
    )
    direct = _direct_lbfgsb(_infinite_value, x0, _LBFGSB_OPTIONS)

    assert (direct.status, direct.success, direct.message) == (0, True, _CONVERGED)
    _assert_nonfinite(
        result, ("fun",), raw_status=0, raw_success=True, raw_message=_CONVERGED
    )
    _assert_backend_state(result, direct)
    assert _label(result) is TerminationLabel.NONFINITE


def test_lbfgsb_abnormal_stop_on_a_nan_gradient_keeps_its_raw_termination() -> None:
    x0 = np.ones(2)

    result = minimize(
        _nan_gradient, x0, driver=Driver.SCIPY_LBFGSB, options=_LBFGSB_OPTIONS
    )
    direct = _direct_lbfgsb(_nan_gradient, x0, _LBFGSB_OPTIONS)

    assert (direct.status, direct.success) == (2, False)
    _assert_nonfinite(
        result,
        ("fun", "jac"),
        raw_status=2,
        raw_success=False,
        raw_message=str(direct.message),
    )
    _assert_backend_state(result, direct)


def test_scipy_bfgs_nan_result_is_reported_nonfinite() -> None:
    result = minimize(
        _nan_gradient,
        np.ones(2),
        driver=Driver.SCIPY_BFGS,
        options=ScipyBFGSOptions(maxiter=20),
    )

    _assert_nonfinite(
        result,
        ("jac",),
        raw_status=3,
        raw_success=False,
        raw_message="NaN result encountered.",
    )
    assert _same_array(result.x, np.ones(2))
    assert result.fun == 2.0


def test_a_recovered_finite_endpoint_is_judged_on_its_own_state() -> None:
    """A NaN trial the line search backed off from does not poison a finite end."""
    trials: list[float] = []

    def invalid_below_half(x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        trials.append(float(x[0]))
        gradient = 2.0 * (x - 1.0)
        if x[0] < 0.5:
            return _NAN, gradient
        return float((x[0] - 1.0) ** 2), gradient

    x0 = np.array([1.2])
    result = minimize(
        invalid_below_half, x0, driver=Driver.SCIPY_LBFGSB, options=_LBFGSB_OPTIONS
    )
    routed_trials = list(trials)
    trials.clear()
    direct = _direct_lbfgsb(invalid_below_half, x0, _LBFGSB_OPTIONS)

    assert any(trial < 0.5 for trial in routed_trials), routed_trials
    assert routed_trials == trials
    _assert_unchanged(result, direct)
    _assert_backend_state(result, direct)
    assert result.success is True and np.isfinite(result.fun)


def _rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    value = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
    gradient = np.zeros_like(x)
    gradient[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    gradient[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return value, gradient


@pytest.mark.parametrize(
    "options",
    [
        ScipyLBFGSBOptions(),
        ScipyLBFGSBOptions.native_matched(maxiter=300, maxcor=300, tol=1e-15),
    ],
    ids=["default", "native_matched"],
)
def test_a_finite_lbfgsb_solve_is_bitwise_scipy_with_no_raw_fields(
    options: ScipyLBFGSBOptions,
) -> None:
    x0 = np.array([-1.2, 1.0, 0.5, -0.3])

    result = minimize(_rosenbrock, x0, driver=Driver.SCIPY_LBFGSB, options=options)
    direct = _direct_lbfgsb(_rosenbrock, x0, options)

    _assert_unchanged(result, direct)
    _assert_backend_state(result, direct)


# ---------------------------------------------------------------------------
# SciPy's all-fixed early return (``scipy/optimize/_minimize.py:721,1160-1198``):
# no method runs, and the result carries x, fun, success, message, nfev, njev, nhev
# only -- no jac, status or nit.

_ALL_FIXED_MESSAGE = "All independent variables were fixed by bounds."
_ALL_FIXED_X0 = np.array([1.0, -2.0])
_ALL_FIXED_BOX = ScipyBounds(lower=(1.0, -2.0), upper=(1.0, -2.0))


def _all_fixed(
    value: float, restart: bool
) -> tuple[OptimizerResult, OptimizeResult, int]:
    """(routed result, direct SciPy result, routed objective calls)."""
    calls: list[np.ndarray] = []

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        calls.append(np.array(x, dtype=float))
        return value, 2.0 * np.asarray(x, dtype=float)

    options = ScipyLBFGSBOptions(maxiter=50, restart_after_nonwolfe_stop=restart)
    result = minimize(
        objective,
        _ALL_FIXED_X0,
        driver=Driver.SCIPY_LBFGSB,
        options=options,
        bounds=_ALL_FIXED_BOX,
    )
    routed_calls = len(calls)
    direct = scipy_minimize(
        objective,
        _ALL_FIXED_X0,
        jac=True,
        method="L-BFGS-B",
        bounds=_ALL_FIXED_BOX.scipy_bounds(),
        options={"maxiter": options.maxiter, "maxfun": options.maxfun},
    )
    return result, direct, routed_calls


@pytest.mark.parametrize("restart", [False, True], ids=["default", "restart"])
def test_an_all_fixed_problem_returns_scipys_early_result(restart: bool) -> None:
    result, direct, routed_calls = _all_fixed(5.0, restart)

    assert not {"jac", "status", "nit"} & set(direct), sorted(direct)
    assert (bool(direct.success), str(direct.message)) == (True, _ALL_FIXED_MESSAGE)
    # jac absent (no re-evaluation), no iteration ran, and the route's defined status
    # for SciPy's success-without-status is 0.
    assert result.jac is None
    assert (result.status, result.success, result.message) == (
        0,
        True,
        _ALL_FIXED_MESSAGE,
    )
    assert (result.nit, result.nfev, result.njev) == (0, direct.nfev, direct.njev)
    assert routed_calls == 1
    assert _same_array(result.x, np.asarray(direct.x, dtype=float))
    assert result.fun == float(direct.fun) == 5.0
    assert result.nonfinite_fields == () and result.raw_status is None
    assert result.restart_log == ()
    assert result.optimizer_state_trace == ([] if restart else None)


@pytest.mark.parametrize("restart", [False, True], ids=["default", "restart"])
def test_an_all_fixed_nonfinite_value_still_reaches_the_boundary(restart: bool) -> None:
    result, direct, routed_calls = _all_fixed(_NAN, restart)

    assert (bool(direct.success), str(direct.message)) == (True, _ALL_FIXED_MESSAGE)
    _assert_nonfinite(
        result, ("fun",), raw_status=0, raw_success=True, raw_message=_ALL_FIXED_MESSAGE
    )
    assert result.jac is None and routed_calls == 1
    assert _same_array(result.x, np.asarray(direct.x, dtype=float))


# ---------------------------------------------------------------------------
# Code A termination reads the status the same way.


def test_the_termination_table_reads_the_nonfinite_status_as_nonfinite() -> None:
    finite = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, NONFINITE_RESULT_STATUS, "", False),
        x=np.zeros(2),
        fun=0.0,
        gradient=np.zeros(2),
        bounds=None,
        accepted_fun=(),
    )

    assert finite.label is TerminationLabel.NONFINITE
    # The status names a non-finite returned state; a record whose evidence is all
    # finite contradicts it and is never well formed.
    assert any(
        "NONFINITE_RESULT_STATUS" in defect
        for defect in termination_record_defects(finite.to_record())
    )


def test_a_nonfinite_boundary_record_is_well_formed_and_labelled_nonfinite() -> None:
    result = minimize(
        _infinite_value, np.ones(2), driver=Driver.SCIPY_LBFGSB, options=_LBFGSB_OPTIONS
    )
    assert result.jac is not None
    report = termination_report(
        EmitterStop(
            Emitter.SCIPY_LBFGSB, result.status, result.message, result.success
        ),
        x=result.x,
        fun=result.fun,
        gradient=result.jac,
        bounds=None,
        accepted_fun=(),
    )
    # The backend's own status gives the same label: NONFINITE takes precedence.
    assert result.raw_status is not None and result.raw_message is not None
    raw = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, result.raw_status, result.raw_message, True),
        x=result.x,
        fun=result.fun,
        gradient=result.jac,
        bounds=None,
        accepted_fun=(),
    )

    assert report.label is raw.label is TerminationLabel.NONFINITE
    assert termination_record_defects(report.to_record()) == ()


# Only the dispatch boundary emits status 8, and SciPy L-BFGS-B is the only emitter
# routed through it; the other emitters are not dispatch drivers.
_NON_BOUNDARY_EMITTERS = [
    emitter for emitter in Emitter if emitter is not Emitter.SCIPY_LBFGSB
]


def _finite_record(emitter: Emitter, status: int) -> dict[str, object]:
    return termination_report(
        EmitterStop(emitter, status, "", False),
        x=np.zeros(2),
        fun=0.0,
        gradient=np.zeros(2),
        bounds=None,
        accepted_fun=(),
    ).to_record()


@pytest.mark.parametrize("emitter", _NON_BOUNDARY_EMITTERS, ids=str)
def test_the_boundary_status_is_refused_on_an_emitter_that_cannot_emit_it(
    emitter: Emitter,
) -> None:
    with pytest.raises(ValueError, match="NONFINITE_RESULT_STATUS"):
        EmitterStop(emitter, NONFINITE_RESULT_STATUS, "", False)
    record = _finite_record(emitter, 0)
    for nonfinite in ([], ["hessian"], ["fun"]):
        record.update(status=NONFINITE_RESULT_STATUS, nonfinite_fields=nonfinite)
        assert any(
            "NONFINITE_RESULT_STATUS" in defect
            for defect in termination_record_defects(record)
        ), (emitter, nonfinite)


def test_a_boundary_status_record_needs_a_nonfinite_returned_field() -> None:
    """A non-finite post-solve Hessian alone cannot be the boundary's cause."""
    record = termination_report(
        EmitterStop(Emitter.SCIPY_LBFGSB, NONFINITE_RESULT_STATUS, "", False),
        x=np.zeros(2),
        fun=0.0,
        gradient=np.zeros(2),
        bounds=None,
        accepted_fun=(),
        hessian=np.full((2, 2), _NAN),
    ).to_record()

    assert record["nonfinite_fields"] == ["hessian"]
    assert any(
        "NONFINITE_RESULT_STATUS" in defect
        for defect in termination_record_defects(record)
    )


def test_a_boundary_status_record_is_checked_against_the_endpoint_evidence() -> None:
    """A record claiming a non-finite value is refused on finite evidence."""
    claimed = _finite_record(Emitter.SCIPY_LBFGSB, 0)
    claimed.update(
        status=NONFINITE_RESULT_STATUS,
        nonfinite_fields=["fun"],
        label=TerminationLabel.NONFINITE.value,
    )
    assert termination_record_defects(claimed) == ()

    verified = verify_termination_evidence(
        claimed,
        x=np.zeros(2),
        fun=0.0,
        gradient=np.zeros(2),
        bounds=None,
        accepted_fun=(),
    )
    genuine = verify_termination_evidence(
        claimed,
        x=np.zeros(2),
        fun=_NAN,
        gradient=np.zeros(2),
        bounds=None,
        accepted_fun=(),
    )

    assert any("nonfinite_fields" in defect for defect in verified.defects)
    assert genuine.defects == ()
    assert genuine.report is not None
    assert genuine.report.label is TerminationLabel.NONFINITE
