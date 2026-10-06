"""``ScipyLBFGSBOptions.restart_after_nonwolfe_stop`` on the SciPy L-BFGS-B route.

The stall objective is a gentle quadratic (curvature 1e-4) whose minimizer
lies behind a smoothstep wall of height 1e14 with an exactly flat top.  Once
L-BFGS-B's memory holds the quadratic's curvature, its unit step lands on the
flat top.  From ``(0, f_b, g_b.d)`` and ``(1, f_b + 1e14, ~0)`` dcstep's cubic
step is exactly 0 (``|g_b.d|`` vanishes in the rounding of ``|theta| ~ 3e14``),
the next trial is ``x_b`` bitwise, dcsrch ends on the WARNING ``ROUNDING ERRORS
PREVENT PROGRESS``, ``lnsrlb`` accepts it like convergence and the
relative-reduction test stops the solve with a projected gradient near 7e-4:
a cliff stall at a steep penalty wall, in two dimensions.
The smoothstep is exactly zero before the wall and exactly flat after it, so
the arithmetic, and with it the stall, is deterministic.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from simsopt_jax.solve import (
    Driver,
    SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS,
    STATUS_CODES,
    LbfgsbRestartEvent,
    LbfgsbRestartReason,
    ScipyBounds,
    ScipyLBFGSBOptions,
)
from simsopt_jax.solve.dispatch import (
    _HostEvaluation,
    _LbfgsbCallRecorder,
    _LineSearch,
    minimize,
)

RELATIVE_REDUCTION_STOP = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
ITERATION_LIMIT_STOP = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
EVALUATION_LIMIT_STOP = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"

_CURVATURE = 1e-4
_WALL_HEIGHT = 1e14
_WALL_WIDTH = 0.05
_TARGET = np.array([8.0, 1.0])
_X0 = np.array([0.0, 0.3])
_LOOSE_BOX = ScipyBounds(lower=(-100.0, -100.0), upper=(100.0, 100.0))


def _options(*, restart: bool, maxiter: int = 60, maxfun: int = 15000):
    return ScipyLBFGSBOptions(
        maxiter=maxiter,
        maxfun=maxfun,
        maxcor=300,
        ftol=1e-15,
        gtol=1e-15,
        restart_after_nonwolfe_stop=restart,
    )


def _walled_quadratic(wall_start: float):
    """``0.5 c |x - target|^2`` plus a flat-topped wall rising over ``[wall_start, wall_start + w]`` in ``x[0]``."""

    def value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        point = np.asarray(x, dtype=float)
        residual = point - _TARGET
        t = min(max((point[0] - wall_start) / _WALL_WIDTH, 0.0), 1.0)
        value = 0.5 * _CURVATURE * float(residual @ residual)
        value += _WALL_HEIGHT * t * t * (3.0 - 2.0 * t)
        gradient = _CURVATURE * residual
        gradient[0] += _WALL_HEIGHT * 6.0 * t * (1.0 - t) / _WALL_WIDTH
        return value, gradient

    return value_and_gradient


class _CountedObjective:
    """The objective with a count of the calls actually made to it."""

    def __init__(self, value_and_gradient) -> None:
        self._value_and_gradient = value_and_gradient
        self.calls = 0

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        self.calls += 1
        return self._value_and_gradient(x)


def _direct_scipy(value_and_gradient, options, bounds: ScipyBounds | None):
    return scipy_minimize(
        value_and_gradient,
        _X0,
        jac=True,
        method="L-BFGS-B",
        bounds=None if bounds is None else bounds.scipy_bounds(),
        options={
            "maxiter": options.maxiter,
            "maxfun": options.maxfun,
            "gtol": options.gtol,
            "ftol": options.ftol,
            "maxcor": options.maxcor,
            "maxls": options.maxls,
        },
    )


def _solve(value_and_gradient, options, bounds: ScipyBounds | None = None, **kwargs):
    return minimize(
        value_and_gradient,
        _X0,
        driver=Driver.SCIPY_LBFGSB,
        options=options,
        bounds=bounds,
        **kwargs,
    )


def _grad_norm_inf(result) -> float:
    return float(np.max(np.abs(result.jac)))


# ---------------------------------------------------------------------------
# T1: faithful default.


@pytest.mark.parametrize("bounds", [None, _LOOSE_BOX], ids=["unbounded", "boxed"])
def test_option_off_is_scipy_bit_for_bit_through_a_cliff_stall(bounds) -> None:
    """Off, the route is SciPy: the stall happens exactly as SciPy has it, and nothing follows."""
    objective = _walled_quadratic(wall_start=3.0)
    options = _options(restart=False)

    routed = _solve(objective, options, bounds)
    direct = _direct_scipy(objective, options, bounds)

    assert direct.message == RELATIVE_REDUCTION_STOP and direct.nit == 2, (
        "the synthetic cliff stall no longer reproduces in SciPy "
        f"({direct.message!r} after {direct.nit} iterations)"
    )
    assert np.array_equal(routed.x, direct.x)
    assert routed.fun == direct.fun
    assert (routed.nit, routed.nfev, routed.njev, routed.status, routed.message) == (
        direct.nit,
        direct.nfev,
        direct.njev,
        direct.status,
        direct.message,
    )
    assert routed.restart_log == ()


@pytest.mark.parametrize("bounds", [None, _LOOSE_BOX], ids=["unbounded", "boxed"])
def test_option_on_without_a_stall_is_the_option_off_run(bounds) -> None:
    """Without a non-Wolfe stop the option changes nothing, bit for bit."""
    objective = _walled_quadratic(wall_start=50.0)

    off = _solve(objective, _options(restart=False), bounds)
    on = _solve(objective, _options(restart=True), bounds)

    assert np.array_equal(on.x, off.x)
    assert on.fun == off.fun
    assert np.array_equal(on.jac, off.jac)
    assert (on.nit, on.nfev, on.njev, on.status, on.message) == (
        off.nit,
        off.nfev,
        off.njev,
        off.status,
        off.message,
    )
    assert on.restart_log == ()


# ---------------------------------------------------------------------------
# T2: the detector.


def _evaluation(x, fun, gradient) -> _HostEvaluation:
    return _HostEvaluation(
        x=np.asarray(x, dtype=float),
        fun=fun,
        gradient=np.asarray(gradient, dtype=float),
    )


_BASE = _evaluation([0.0, 0.0], 1.0, [-1.0, 0.0])
_WALL_TRIAL = _evaluation([1.0, 0.0], 1e11, [3.0, 0.0])


def _search(*, accepted_x, accepted_gradient) -> _LineSearch:
    return _LineSearch(
        base=_BASE,
        trials=(_WALL_TRIAL, _evaluation(accepted_x, 0.5, accepted_gradient)),
    )


def test_detector_flags_an_accepted_step_whose_slope_did_not_flatten() -> None:
    search = _search(accepted_x=[1e-12, 0.0], accepted_gradient=[-1.0, 0.25])

    assert search.on_one_ray
    assert search.curvature_ratio == -1.0
    assert search.fails_curvature_condition


def test_detector_passes_a_wolfe_step() -> None:
    search = _search(accepted_x=[0.4, 0.0], accepted_gradient=[-0.15, 7.0])

    assert search.curvature_ratio == -0.15
    assert not search.fails_curvature_condition


def test_detector_passes_a_slope_exactly_at_the_curvature_bound() -> None:
    """``|g_new.d| <= 0.9 |g_old.d|`` is dcsrch's acceptance, so equality passes."""
    search = _search(accepted_x=[0.2, 0.0], accepted_gradient=[-0.9, 0.0])

    assert search.curvature_ratio == -0.9
    assert not search.fails_curvature_condition


def test_detector_flags_a_zero_step() -> None:
    """A step of exactly 0 accepts the base itself: the slope is ``g_b.d``."""
    search = _search(accepted_x=[0.0, 0.0], accepted_gradient=[-1.0, 0.0])

    assert search.on_one_ray
    assert search.accepted_step_fraction == 0.0
    assert search.curvature_ratio == -1.0
    assert search.fails_curvature_condition


def test_first_trial_value_ratio_is_defined_at_a_zero_base() -> None:
    def ratio(base_fun: float, trial_fun: float) -> float:
        return _LineSearch(
            base=_evaluation([0.0], base_fun, [-1.0]),
            trials=(_evaluation([1.0], trial_fun, [1.0]),),
        ).first_trial_fun_ratio

    assert ratio(2.0, 5.0) == 2.5
    assert ratio(0.0, 5.0) == math.inf
    assert ratio(0.0, -5.0) == -math.inf
    assert ratio(0.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# Internal retries: an iteration that holds two searches is not judged.


def test_trials_at_rounding_distance_from_the_base_stay_on_the_ray() -> None:
    """Collapsed trials (``stp ~ 1e-16``) differ from the base by rounding only."""
    base = _evaluation([1.0, -3.0, 0.25], 1.0, [-1.0, 2.0, 0.5])
    trials = (
        _evaluation([2.0, -5.0, 0.0], 1e11, [1.0, 1.0, 1.0]),
        _evaluation([np.nextafter(1.0, 2.0), -3.0, 0.25], 1.0, [-1.0, 2.0, 0.5]),
        _evaluation([1.5, -4.0, 0.125], 0.9, [-0.5, 1.0, 0.25]),
        _evaluation([1.0, -3.0, 0.25], 1.0, [-1.0, 2.0, 0.5]),
    )

    assert _LineSearch(base=base, trials=trials).on_one_ray


def test_a_retried_search_is_not_on_one_ray() -> None:
    """L-BFGS-B's internal retry: a failed search on ray A, then an accepted one on ray B.

    Judged along A (the first trial's ray), the accepted step would look like a
    non-Wolfe stall (ratio -1); along its own ray B it is a Wolfe step.  The
    iteration must not be judged at all.
    """
    base = _evaluation([0.0, 0.0], 1.0, [-1.0, -1.0])
    failed_on_a = (
        _evaluation([1.0, 0.0], 1e11, [5.0, 0.0]),
        _evaluation([0.5, 0.0], 1e9, [4.0, 0.0]),
    )
    accepted_on_b = _evaluation([0.0, 0.5], 0.6, [-1.0, 0.05])
    search = _LineSearch(base=base, trials=(*failed_on_a, accepted_on_b))

    assert search.curvature_ratio == -1.0, "the hazard: ray A says non-Wolfe"
    assert not search.on_one_ray


def test_recorder_marks_an_iteration_with_two_rays() -> None:
    """The recorder as SciPy drives it: evaluations, then one iteration end."""
    points = iter(
        [
            ([0.0, 0.0], 1.0, [-1.0, -1.0]),
            ([1.0, 0.0], 1e11, [5.0, 0.0]),
            ([0.0, 0.5], 0.6, [-1.0, 0.05]),
        ]
    )

    def value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        _x, value, gradient = next(points)
        return value, np.asarray(gradient, dtype=float)

    recorder = _LbfgsbCallRecorder(value_and_gradient, None, None)
    for x in ([0.0, 0.0], [1.0, 0.0], [0.0, 0.5]):
        recorder.fun(np.asarray(x, dtype=float))
    recorder.end_iteration(np.array([0.0, 0.5]))

    assert recorder.last_search is not None
    assert not recorder.last_search.on_one_ray


# ---------------------------------------------------------------------------
# Owned snapshots.

_ANISOTROPIC_CURVATURE = np.array([1.0, 1e-3, 30.0, 0.2])
_ANISOTROPIC_CENTER = np.array([1.0, -2.0, 0.5, 3.0])


class _BufferReusingQuadratic:
    """A callable that writes every gradient into one buffer and returns it."""

    def __init__(self) -> None:
        self.buffer = np.zeros(4)

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = np.asarray(x, dtype=float) - _ANISOTROPIC_CENTER
        self.buffer[:] = _ANISOTROPIC_CURVATURE * residual
        return 0.5 * float(np.sum(_ANISOTROPIC_CURVATURE * residual**2)), self.buffer


def test_recorder_snapshots_do_not_alias_a_reused_gradient_buffer() -> None:
    objective = _BufferReusingQuadratic()
    recorder = _LbfgsbCallRecorder(objective, None, None)

    recorder.fun(np.zeros(4))
    recorder.fun(np.ones(4))
    recorder.end_iteration(np.ones(4))
    objective(np.full(4, 7.0))  # e.g. a callback re-evaluation

    search = recorder.last_search
    assert search is not None
    np.testing.assert_array_equal(
        search.base.gradient, -_ANISOTROPIC_CURVATURE * _ANISOTROPIC_CENTER
    )
    np.testing.assert_array_equal(
        search.accepted.gradient,
        _ANISOTROPIC_CURVATURE * (1.0 - _ANISOTROPIC_CENTER),
    )
    assert not search.base.gradient.flags.writeable
    assert not search.accepted.x.flags.writeable


def test_a_reused_gradient_buffer_does_not_turn_a_wolfe_stop_into_a_restart() -> None:
    """A Wolfe-step relative-reduction stop (ratio ~ 0) stays SciPy's result."""
    options = dict(maxiter=100, maxcor=10, ftol=1e-6, gtol=1e-15)
    off = minimize(
        _BufferReusingQuadratic(),
        np.zeros(4),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(**options),
    )
    on = minimize(
        _BufferReusingQuadratic(),
        np.zeros(4),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(**options, restart_after_nonwolfe_stop=True),
    )

    assert off.message == RELATIVE_REDUCTION_STOP
    assert on.restart_log == ()
    assert np.array_equal(on.x, off.x)
    assert (on.nit, on.nfev, on.status, on.success) == (
        off.nit,
        off.nfev,
        off.status,
        off.success,
    )


# ---------------------------------------------------------------------------
# T3: the synthetic cliff stall is resumed.


def test_restart_resumes_a_cliff_stall_and_lowers_value_and_gradient() -> None:
    objective = _walled_quadratic(wall_start=3.0)
    off = _solve(objective, _options(restart=False))
    counted = _CountedObjective(objective)

    on = _solve(counted, _options(restart=True))

    first = on.restart_log[0]
    assert first == LbfgsbRestartEvent(
        iteration=off.nit,
        reason=LbfgsbRestartReason.RESTARTED,
        curvature_ratio=-1.0,
        fun=off.fun,
        projected_grad_norm_inf=_grad_norm_inf(off),
        first_trial_fun=first.first_trial_fun,
        first_trial_fun_ratio=first.first_trial_fun / off.fun,
        accepted_step_fraction=0.0,
        scipy_status=0,
        scipy_message=RELATIVE_REDUCTION_STOP,
    ), "the first stop is the option-off stop, and it is a non-Wolfe zero step"
    assert first.first_trial_fun > off.fun + 0.99 * _WALL_HEIGHT
    assert on.fun < off.fun
    assert _grad_norm_inf(on) < _grad_norm_inf(off)
    assert on.nit <= 60
    assert on.nfev == counted.calls, "a restart's start is served, not re-evaluated"
    iterations = [event.iteration for event in on.restart_log]
    assert iterations == sorted(set(iterations))
    assert all(event.curvature_ratio < -0.9 for event in on.restart_log)
    # The chain ends when a fresh call's first unit step lands on the wall's
    # top: progress was made, but the solve ends on an unresolved stall.
    assert on.restart_log[-1].reason is LbfgsbRestartReason.FRESH_MEMORY_STALL
    assert (on.success, on.status) == (False, SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS)


def test_restarts_share_one_iteration_budget() -> None:
    """Boxed, each fresh call moves by ``-g`` and stalls again; the total stops at ``maxiter``."""
    objective = _walled_quadratic(wall_start=3.0)
    off = _solve(objective, _options(restart=False, maxiter=20), _LOOSE_BOX)

    on = _solve(objective, _options(restart=True, maxiter=20), _LOOSE_BOX)

    assert on.nit == 20
    assert (on.status, on.success, on.message) == (1, False, ITERATION_LIMIT_STOP)
    assert len(on.restart_log) >= 2
    assert all(
        event.reason is LbfgsbRestartReason.RESTARTED for event in on.restart_log
    )
    assert on.fun < off.fun
    assert _grad_norm_inf(on) < _grad_norm_inf(off)


def test_a_restarted_call_gets_exactly_the_evaluations_left() -> None:
    """``maxfun`` 5: the first call spends 4 (x0, the unit step, the wall trial,
    x_b again) and stalls; the restarted call gets ``maxfun = 5 - 4 = 1``.  SciPy
    counts its served start as its first evaluation, spends the unit step (the
    5th true evaluation), and stops at that iteration end: 5 in all, never
    more than the original ``maxfun``.
    """
    counted = _CountedObjective(_walled_quadratic(wall_start=3.0))

    on = _solve(counted, _options(restart=True, maxfun=5))

    assert (on.status, on.message) == (1, EVALUATION_LIMIT_STOP)
    assert on.nfev == counted.calls == 5
    assert on.nit == 3
    assert [event.reason for event in on.restart_log] == [LbfgsbRestartReason.RESTARTED]


def test_a_stall_with_no_evaluation_left_is_unresolved() -> None:
    """``maxfun`` 4 is spent by the first call exactly; no call can follow."""
    counted = _CountedObjective(_walled_quadratic(wall_start=3.0))

    on = _solve(counted, _options(restart=True, maxfun=4))

    assert on.nfev == counted.calls == 4
    assert [event.reason for event in on.restart_log] == [
        LbfgsbRestartReason.BUDGET_EXHAUSTED
    ]
    assert (on.success, on.status) == (False, SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS)
    assert on.message == (
        "UNRESOLVED NON-WOLFE STALL (budget_exhausted_not_restarted): "
        + RELATIVE_REDUCTION_STOP
    )


def test_callback_history_is_continuous_across_restarts() -> None:
    events = []

    on = _solve(
        _walled_quadratic(wall_start=3.0),
        _options(restart=True),
        callback=events.append,
    )

    assert len(on.restart_log) >= 2
    assert [event.iteration for event in events] == list(range(1, on.nit + 1))
    assert all(
        later.wallclock_s >= earlier.wallclock_s
        for earlier, later in zip(events, events[1:])
    )
    assert np.array_equal(events[-1].x, on.x)


# ---------------------------------------------------------------------------
# T4: a fresh call that stalls on its first iteration is not restarted.


def test_fresh_memory_stall_is_not_restarted_and_is_unsuccessful() -> None:
    """After one restart the fresh unit step lands on the wall's top and stalls the same way."""
    off = _solve(_walled_quadratic(wall_start=1.5), _options(restart=False))

    on = _solve(_walled_quadratic(wall_start=1.5), _options(restart=True))

    assert [event.reason for event in on.restart_log] == [
        LbfgsbRestartReason.RESTARTED,
        LbfgsbRestartReason.FRESH_MEMORY_STALL,
    ]
    assert [event.iteration for event in on.restart_log] == [off.nit, off.nit + 1]
    stalled = on.restart_log[-1]
    assert stalled.curvature_ratio < -0.9
    assert (stalled.scipy_status, stalled.scipy_message) == (
        0,
        RELATIVE_REDUCTION_STOP,
    ), "the stalled call's own SciPy termination is kept"
    assert (on.success, on.status) == (False, SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS)
    assert on.message == (
        "UNRESOLVED NON-WOLFE STALL (fresh_memory_stall_not_restarted): "
        + RELATIVE_REDUCTION_STOP
    )
    assert SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS in STATUS_CODES[Driver.SCIPY_LBFGSB]
    assert on.nit == off.nit + 1


# ---------------------------------------------------------------------------
# T5: contract.


def test_restart_option_is_off_by_default_and_in_native_matched_options() -> None:
    assert ScipyLBFGSBOptions().restart_after_nonwolfe_stop is False
    matched = ScipyLBFGSBOptions.native_matched(maxiter=300, maxcor=300, tol=1e-15)
    assert matched.restart_after_nonwolfe_stop is False
    assert dataclasses.asdict(_options(restart=True))["restart_after_nonwolfe_stop"]


def test_restart_event_is_immutable() -> None:
    event = LbfgsbRestartEvent(
        iteration=2,
        reason=LbfgsbRestartReason.RESTARTED,
        curvature_ratio=-1.0,
        fun=1.0,
        projected_grad_norm_inf=0.5,
        first_trial_fun=2.0,
        first_trial_fun_ratio=2.0,
        accepted_step_fraction=0.0,
        scipy_status=0,
        scipy_message=RELATIVE_REDUCTION_STOP,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.iteration = 3  # type: ignore[misc]
