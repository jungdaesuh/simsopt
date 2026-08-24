"""Pin scipy 1.17.1's ``StopIteration``-from-callback control flow.

The nested-LS outer children drive ``scipy.optimize.minimize(...,
method="L-BFGS-B", callback=...)`` and the upgrade plan uses
``StopIteration`` raised from that callback as a halt signal. The
status/message/success triple that halt produces is version-specific, and the
published accounts disagree between status ``2`` and status ``99``. Both are
real: they are two different *boundaries* of the same run, and this file pins
each one at the boundary it belongs to (see
``test_lbfgsb_solver_boundary_reports_status_2_before_minimize_overrides``).

This test is deliberately version-scoped. Every expectation here was READ OFF
a real run on the installed scipy, not derived from documentation. A scipy
version bump must therefore re-observe the triple and the adjacent facts and
rewrite the pins from what the new version actually does — editing an
expectation to make this file pass again is the one repair that is forbidden,
because it would silently relabel a changed control flow as unchanged.

Where scipy implements this (paths relative to the venv's ``site-packages``):

* the catch itself is ``scipy/_lib/_util.py:1006-1011``
  (``_call_callback_maybe_halt``): it calls the wrapped callback inside
  ``try``, and on ``StopIteration`` sets ``callback.stop_iteration = True``
  and returns ``True`` to halt. Nothing re-raises.
* ``scipy/optimize/_lbfgsb_py.py:475-477`` turns that ``True`` into the
  solver's own stop code (``task = (5, 505)``), whose message is composed at
  ``scipy/optimize/_lbfgsb_py.py:505`` and returned, together with the
  ``warnflag`` derived at ``:487-492``, from the ``OptimizeResult`` built at
  ``:507-510`` — ``status=2`` / ``"STOP: CALLBACK REQUESTED HALT"``.
* ``scipy/optimize/_minimize.py:823-826`` then OVERRIDES that triple on the
  ``minimize`` result: ``status=99``, ``message="`callback` raised
  `StopIteration`."``, ``success=False``.
* ``scipy/optimize/_optimize.py:88-109`` (``_wrap_callback``) dispatches on
  the callback signature; both dispatch arms funnel through the single catch
  above, which is why the triple does not depend on the callback style.

The solver-boundary half is CONDITIONAL and the condition matters here. The
halt store at ``:475-477`` is followed at ``:478`` by a separate un-chained
``if n_iterations >= maxiter:`` — an ``if``, not an ``elif`` — which
overwrites ``task`` with ``(5, 504)``, and ``:489-490`` then derives
``warnflag = 1``. So a halt that lands ON the iteration budget leaves
``_minimize_lbfgsb`` reporting the budget stop, not the callback stop: the
callback's fingerprint is erased at that boundary. The ``minimize`` boundary
is immune, because ``_minimize.py:823`` keys on the wrapped callback's
``stop_iteration`` flag rather than on the task code, so the status-99 triple
survives the coincidence. Both cases are pinned below
(``test_solver_boundary_halt_is_erased_when_it_coincides_with_the_budget``,
``test_minimize_boundary_halt_survives_coinciding_with_the_budget``) because
a halt-signal migration that also runs under a ``maxiter`` is exactly where
the difference bites.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.optimize._lbfgsb_py import _minimize_lbfgsb

PINNED_SCIPY_VERSION = "1.17.1"

# Rosenbrock from the classic start point: smooth, unconstrained, and it needs
# far more than three iterations, so halting at the third accepted iterate is a
# genuine interruption rather than a convergence that happens to coincide.
_X0: NDArray[np.float64] = np.array([-1.2, 1.0])
_STOP_AT_CALLBACK = 3

# `nit` is the number of accepted iterations, so halting at the third callback
# invocation pins it structurally rather than arithmetically.
_OBSERVED_NIT = 3

# For the record, this host observed nfev=5 and exactly 1 rejected line-search
# trial for the halted fixture, and nit=36 for the unhalted control. Those
# counts are NOT asserted: how many trials L-BFGS-B's line search rejects (and
# therefore how many iterations Rosenbrock takes) depends on the BLAS/libm
# build, and this suite also runs on other hosts. The assertions below pin the
# relations those counts illustrate, which are the actual claims.


def _rosenbrock(x: NDArray[np.float64]) -> float:
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def _rosenbrock_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )


class _HaltingRun:
    """One L-BFGS-B run that halts by raising ``StopIteration``.

    Records every point the objective was evaluated at (``trials``) and every
    iterate scipy handed to the callback (``accepted``), so the adjacent facts
    can be asserted against observed state rather than against scipy's code.
    """

    def __init__(self, *, style: str, stop_at: int = _STOP_AT_CALLBACK) -> None:
        self.trials: list[NDArray[np.float64]] = []
        self.accepted: list[NDArray[np.float64]] = []
        self._stop_at = stop_at
        self.result = minimize(
            self._objective,
            _X0,
            jac=_rosenbrock_grad,
            method="L-BFGS-B",
            callback=self._new_style if style == "new" else self._old_style,
        )

    def _objective(self, x: NDArray[np.float64]) -> float:
        self.trials.append(np.array(x, copy=True))
        return _rosenbrock(x)

    def _record(self, x: NDArray[np.float64]) -> None:
        self.accepted.append(np.array(x, copy=True))
        if len(self.accepted) >= self._stop_at:
            raise StopIteration

    def _new_style(self, intermediate_result: Any) -> None:
        self._record(intermediate_result.x)

    def _old_style(self, xk: NDArray[np.float64]) -> None:
        self._record(xk)


@pytest.fixture(scope="module")
def halted() -> _HaltingRun:
    """A completed new-style-callback run halted at the third iterate."""

    return _HaltingRun(style="new")


def test_installed_scipy_is_the_pinned_version() -> None:
    assert scipy.__version__ == PINNED_SCIPY_VERSION, (
        "scipy version drifted: every pin in this file was observed on "
        f"{PINNED_SCIPY_VERSION} and must be RE-OBSERVED on "
        f"{scipy.__version__}, not edited to match."
    )


def test_stop_iteration_halts_the_run_early_instead_of_propagating(
    halted: _HaltingRun,
) -> None:
    """Reaching the assertions at all is the "does not propagate" half."""

    unhalted = minimize(_rosenbrock, _X0, jac=_rosenbrock_grad, method="L-BFGS-B")
    assert (unhalted.status, unhalted.success) == (0, True), (
        "the control run no longer CONVERGES, so 'halted early' is no longer "
        "measured against a known convergence. Observed: "
        f"status={unhalted.status} success={unhalted.success} "
        f"nit={unhalted.nit}. (The iteration count itself is deliberately not "
        "pinned — it is BLAS/libm-build dependent and this suite runs on "
        "several hosts.)"
    )
    assert halted.result.nit < unhalted.nit, (
        "fact drifted: StopIteration from the callback no longer stopped "
        f"L-BFGS-B early ({halted.result.nit} iterations vs {unhalted.nit} "
        "for the same problem without a callback)."
    )


def test_a_non_stop_iteration_exception_still_reaches_the_caller() -> None:
    def raise_runtime_error(intermediate_result: Any) -> None:
        raise RuntimeError("callback failure must not be swallowed")

    with pytest.raises(RuntimeError, match="must not be swallowed"):
        minimize(
            _rosenbrock,
            _X0,
            jac=_rosenbrock_grad,
            method="L-BFGS-B",
            callback=raise_runtime_error,
        )


def test_a_stop_iteration_subclass_is_also_caught_as_a_halt() -> None:
    class HaltSignal(StopIteration):
        pass

    def raise_subclass(intermediate_result: Any) -> None:
        raise HaltSignal

    result = minimize(
        _rosenbrock,
        _X0,
        jac=_rosenbrock_grad,
        method="L-BFGS-B",
        callback=raise_subclass,
    )
    assert result.status == 99, (
        "fact drifted: a StopIteration SUBCLASS no longer registers as a halt "
        f"(status {result.status}); scipy catches the base class, so a typed "
        "halt signal derived from StopIteration was usable."
    )


def test_minimize_reports_the_pinned_stop_iteration_triple(
    halted: _HaltingRun,
) -> None:
    observed = (
        halted.result.status,
        halted.result.message,
        halted.result.success,
    )
    assert observed == (99, "`callback` raised `StopIteration`.", False), (
        "the minimize()-boundary StopIteration triple drifted from the "
        "scipy 1.17.1 observation; re-observe and repin, do not edit the "
        f"expectation. Observed: {observed!r}"
    )


def _halt_after(stop_at: int) -> Any:
    """A new-style callback that raises ``StopIteration`` on its Nth call."""

    seen: list[int] = []

    def halt(intermediate_result: Any) -> None:
        seen.append(1)
        if len(seen) >= stop_at:
            raise StopIteration

    return halt


def test_lbfgsb_solver_boundary_reports_status_2_before_minimize_overrides() -> None:
    """The other half of the 2-vs-99 disagreement, at its own boundary.

    ``_minimize_lbfgsb`` is scipy-private on purpose: this test exists to show
    that status ``2`` is what the solver itself returns and status ``99`` is
    what ``minimize`` overwrites it with, so a caller that reads a status
    knows which boundary produced it.

    Scoped to a halt that does NOT coincide with the iteration budget — the
    default ``maxiter`` is far above ``_STOP_AT_CALLBACK``. The coincident
    case reports a different triple and is pinned separately by
    ``test_solver_boundary_halt_is_erased_when_it_coincides_with_the_budget``.
    """

    result = _minimize_lbfgsb(
        _rosenbrock, _X0, jac=_rosenbrock_grad, callback=_halt_after(_STOP_AT_CALLBACK)
    )
    observed = (result.status, result.message, result.success)
    assert observed == (2, "STOP: CALLBACK REQUESTED HALT", False), (
        "the solver-boundary triple drifted: status 2 / 'STOP: CALLBACK "
        "REQUESTED HALT' is what _minimize_lbfgsb returns before minimize() "
        f"overrides it to 99. Observed: {observed!r}"
    )


def test_solver_boundary_halt_is_erased_when_it_coincides_with_the_budget() -> None:
    """A halt landing ON ``maxiter`` reports the budget stop, not the halt.

    ``_lbfgsb_py.py:478`` is an ``if``, not an ``elif``, so the iteration-budget
    branch overwrites the callback branch's ``task`` unconditionally. At the
    solver boundary the callback's fingerprint is therefore GONE whenever the
    halt and the budget land on the same iteration: a consumer reading only
    ``_minimize_lbfgsb``'s status cannot tell a requested halt from a run that
    merely ran out of iterations. That is precisely the case a halt-signal
    migration creates, since the children run under an outer iteration budget.
    """

    result = _minimize_lbfgsb(
        _rosenbrock,
        _X0,
        jac=_rosenbrock_grad,
        callback=_halt_after(_STOP_AT_CALLBACK),
        maxiter=_STOP_AT_CALLBACK,
    )
    observed = (result.status, result.message, result.success)
    assert observed == (1, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT", False), (
        "the coincident solver-boundary triple drifted. On scipy 1.17.1 a "
        "callback halt on the maxiter-th iteration is overwritten by the "
        "budget stop (status 1), which is why the status-2 pin is scoped to "
        f"non-coincident halts. Observed: {observed!r}"
    )


def test_minimize_boundary_halt_survives_coinciding_with_the_budget() -> None:
    """``minimize`` still reports the halt when it lands ON ``maxiter``.

    ``_minimize.py:823`` keys on the wrapped callback's ``stop_iteration``
    flag, not on the solver's task code, so the rewrite that the solver
    boundary loses to the budget branch is applied here regardless. This is
    the boundary the children actually read, and this test is what says the
    halt signal stays legible there even under an outer iteration budget.
    """

    result = minimize(
        _rosenbrock,
        _X0,
        jac=_rosenbrock_grad,
        method="L-BFGS-B",
        callback=_halt_after(_STOP_AT_CALLBACK),
        options={"maxiter": _STOP_AT_CALLBACK},
    )
    observed = (result.status, result.message, result.success)
    assert observed == (99, "`callback` raised `StopIteration`.", False), (
        "the minimize()-boundary halt no longer survives coinciding with the "
        "iteration budget. If minimize starts keying on the solver's task "
        "code instead of the callback's stop_iteration flag, a halt on the "
        "last permitted iteration becomes indistinguishable from budget "
        f"exhaustion. Observed: {observed!r}"
    )


def test_nit_counts_the_callback_invocation_that_raised(
    halted: _HaltingRun,
) -> None:
    assert len(halted.accepted) == _STOP_AT_CALLBACK
    assert halted.result.nit == _OBSERVED_NIT, (
        "fact drifted: result.nit no longer equals the number of callback "
        f"invocations ({len(halted.accepted)}); the halting iteration was "
        f"counted differently. Observed nit={halted.result.nit}."
    )


def test_the_callback_never_fires_for_the_start_point(
    halted: _HaltingRun,
) -> None:
    assert not np.array_equal(halted.accepted[0], _X0), (
        "fact drifted: scipy handed x0 to the callback. The children rely on "
        "the callback naming accepted steps only, so x0 arriving as an "
        "'accepted' iterate would promote a candidate that was never taken."
    )


def test_nfev_counts_line_search_trials_beyond_the_accepted_iterates(
    halted: _HaltingRun,
) -> None:
    accepted_keys = {a.tobytes() for a in halted.accepted}
    rejected = [t for t in halted.trials[1:] if t.tobytes() not in accepted_keys]
    assert halted.result.nfev == len(halted.trials), (
        "fact drifted: result.nfev no longer matches the objective calls this "
        f"run actually made. The objective was called {len(halted.trials)} "
        f"times; result.nfev reports {halted.result.nfev}. The children size "
        "their evaluation budgets off nfev, so a divergence would misprice "
        "every rejected trial."
    )
    assert len(rejected) >= 1, (
        "fact drifted: the line search rejected no trial at all before the "
        "halt, so nfev > nit no longer demonstrates that nfev counts trial "
        f"evaluations. Objective calls {len(halted.trials)}, accepted "
        f"iterates {len(halted.accepted)}, nfev={halted.result.nfev}, "
        f"nit={halted.result.nit}."
    )


def test_result_x_is_the_last_accepted_iterate_not_a_later_trial(
    halted: _HaltingRun,
) -> None:
    assert np.array_equal(halted.result.x, halted.accepted[-1]), (
        "fact drifted: result.x is no longer the iterate the halting callback "
        "was handed. The children promote the candidate named by the "
        "callback, so a result.x that is some other point would silently "
        "return an unevaluated state."
    )
    assert np.array_equal(halted.trials[-1], halted.accepted[-1]), (
        "fact drifted: scipy evaluated the objective again after the halting "
        "callback, so result.x is no longer the last point evaluated."
    )
    assert halted.result.fun == _rosenbrock(halted.result.x), (
        "fact drifted: result.fun no longer belongs to result.x."
    )


def test_old_style_callback_produces_the_same_triple_as_new_style(
    halted: _HaltingRun,
) -> None:
    old = _HaltingRun(style="old")
    new_triple = (halted.result.status, halted.result.message, halted.result.success)
    old_triple = (old.result.status, old.result.message, old.result.success)
    assert old_triple == new_triple, (
        "fact drifted: the halt triple now depends on the callback signature. "
        "scipy dispatches on the signature in _wrap_callback but both arms "
        f"share one catch. old={old_triple!r} new={new_triple!r}"
    )
    assert (old.result.nit, old.result.nfev) == (
        halted.result.nit,
        halted.result.nfev,
    )
    assert np.array_equal(old.result.x, halted.result.x)


def test_new_style_callback_receives_the_live_iterate_buffer() -> None:
    """The one place the two callback styles genuinely differ.

    ``_wrap_callback`` hands an old-style callback ``np.copy(res.x)`` but hands
    a new-style callback the ``OptimizeResult`` whose ``.x`` IS the array
    L-BFGS-B mutates in place. A new-style callback that stores
    ``intermediate_result.x`` without copying therefore ends up holding the
    same buffer for every iterate.
    """

    new_refs: list[NDArray[np.float64]] = []

    def new_style(intermediate_result: Any) -> None:
        new_refs.append(intermediate_result.x)
        if len(new_refs) >= _STOP_AT_CALLBACK:
            raise StopIteration

    old_refs: list[NDArray[np.float64]] = []

    def old_style(xk: NDArray[np.float64]) -> None:
        old_refs.append(xk)
        if len(old_refs) >= _STOP_AT_CALLBACK:
            raise StopIteration

    new_result = minimize(
        _rosenbrock, _X0, jac=_rosenbrock_grad, method="L-BFGS-B", callback=new_style
    )
    minimize(
        _rosenbrock, _X0, jac=_rosenbrock_grad, method="L-BFGS-B", callback=old_style
    )

    assert all(np.array_equal(ref, new_result.x) for ref in new_refs), (
        "fact drifted: uncopied new-style iterates no longer all read as the "
        "final x. They used to alias one live buffer, which is why a "
        "new-style callback must copy before recording a candidate."
    )
    assert not np.array_equal(old_refs[0], old_refs[-1]), (
        "fact drifted: the old-style callback no longer receives a distinct "
        "copy per iteration; the children record xk directly and would start "
        "seeing every recorded candidate collapse onto the last one."
    )
