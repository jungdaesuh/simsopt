"""``ScipyLBFGSBOptions.evaluation_budgeted``, the accepted-iterate trace of the restart route and its
evaluation-accounting validator.

Test groups: L7 the policy, L3 the real wrapper's budget semantics, L3' the in-iteration memory reset and the
validator's ceilings, then the validator's inputs (every entry has a populated ``nfev``; the terminal ABNORMAL
tail without NEW_X, including zero accepted iterations; served-start offsets; the complete validator) and the
restart route's callback. Every objective is a tiny analytic function driven through the real SciPy wrapper.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import dataclasses

import numpy as np
import pytest
from simsopt_jax.solve import (
    Driver,
    LbfgsbRestartReason,
    OptimizerResult,
    ScipyBounds,
    ScipyLBFGSBOptions,
)
from simsopt_jax.solve.contracts import OptimizerStateTraceEntry
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax.solve.lbfgsb_accounting import (
    lbfgsb_evaluation_accounting_defects,
    lbfgsb_result_accounting_defects,
)
from simsopt_jax.solve.termination import (
    Emitter,
    EmitterStop,
    TerminationLabel,
    termination_report,
)

ITERATION_LIMIT_STOP = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
EVALUATION_LIMIT_STOP = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
ABNORMAL_STOP = "ABNORMAL: "


class _Recorded:
    """The objective with every true evaluation kept in call order: the lane-level count below SciPy's memo."""

    def __init__(self, value_and_gradient) -> None:
        self._value_and_gradient = value_and_gradient
        self.values: list[float] = []

    def __call__(self, x):
        value, gradient = self._value_and_gradient(np.asarray(x, dtype=float))
        self.values.append(float(value))
        return value, gradient

    @property
    def calls(self) -> int:
        return len(self.values)


def _spread_quadratic(n: int):
    """Curvatures spread over [0.2, 1]: in a box every L-BFGS-B iteration accepts its first trial."""
    curvature = np.linspace(0.2, 1.0, n)
    center = np.random.default_rng(0).uniform(-1.0, 1.0, n)

    def value_and_gradient(x):
        residual = x - center
        return 0.5 * float(np.sum(curvature * residual**2)), curvature * residual

    return value_and_gradient


def _rosenbrock(x):
    value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    gradient = np.array(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )
    return float(value), gradient


class _PoisonedSearch:
    """A quadratic whose true evaluations number ``first + 1 .. first + count`` (1-based) are poisoned.

    A poisoned evaluation returns the value at evaluation ``first`` plus ``|g_base| |x - x_base|`` (always above the
    base, rising with the distance, so the line search keeps shrinking its step without collapsing onto the base) and
    the true gradient. Evaluation ``first`` is the accepted point of the iteration before the poisoned search, so
    L-BFGS-B's sufficient-decrease test fails on every poisoned trial and the search ends only when ``maxls`` trials
    are spent (``__lbfgsb.c:917-918``).
    """

    def __init__(self, *, first: int, count: int) -> None:
        self._curvature = np.array([1.0, 0.3])
        self._center = np.array([3.0, -2.0])
        self._first = first
        self._count = count
        self._base: tuple[np.ndarray, float, np.ndarray] | None = None
        self.values: list[float] = []

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        residual = x - self._center
        value, gradient = (
            0.5 * float(np.sum(self._curvature * residual**2)),
            self._curvature * residual,
        )
        index = len(self.values) + 1
        if index == self._first:
            self._base = (x.copy(), value, gradient)
        elif self._first < index <= self._first + self._count:
            assert self._base is not None
            base_x, base_value, base_gradient = self._base
            value = base_value + float(np.linalg.norm(base_gradient)) * float(
                np.linalg.norm(x - base_x)
            )
        self.values.append(value)
        return value, gradient

    @property
    def calls(self) -> int:
        return len(self.values)


def _budgeted(
    objective, x0, *, maxfun: int, maxls: int = 20, bounds=None
) -> OptimizerResult:
    return minimize(
        objective,
        x0,
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions.evaluation_budgeted(
            maxfun=maxfun, maxcor=300, tol=1e-15, maxls=maxls
        ),
        bounds=bounds,
    )


def _trace_nfev(result: OptimizerResult) -> list[int | None]:
    assert result.optimizer_state_trace is not None
    return [entry.nfev for entry in result.optimizer_state_trace]


def _per_iteration(result: OptimizerResult) -> list[int]:
    boundaries = [1, *_trace_nfev(result)]
    return [later - earlier for earlier, later in zip(boundaries, boundaries[1:])]


def _label(result: OptimizerResult) -> TerminationLabel:
    assert result.jac is not None
    return termination_report(
        EmitterStop(
            Emitter.SCIPY_LBFGSB,
            result.status,
            result.message,
            result.success,
            result.restart_log[-1].reason if result.restart_log else None,
        ),
        x=result.x,
        fun=result.fun,
        gradient=result.jac,
        bounds=None,
        accepted_fun=(),
    ).label


# ---------------------------------------------------------------------------
# L7: policy.


def test_l7_native_matched_is_frozen() -> None:
    matched = ScipyLBFGSBOptions.native_matched(maxiter=300, maxcor=300, tol=1e-15)

    assert dataclasses.asdict(matched) == {
        "maxiter": 300,
        "maxfun": 15000,
        "gtol": 1e-15,
        "ftol": 1e-15,
        "maxcor": 300,
        "maxls": 20,
        "restart_after_nonwolfe_stop": False,
    }


def test_l7_evaluation_budgeted_fields() -> None:
    budgeted = ScipyLBFGSBOptions.evaluation_budgeted(
        maxfun=6000, maxcor=300, tol=1e-15
    )

    assert dataclasses.asdict(budgeted) == {
        "maxiter": 6000,
        "maxfun": 6000,
        "gtol": 1e-15,
        "ftol": 1e-15,
        "maxcor": 300,
        "maxls": 20,
        "restart_after_nonwolfe_stop": True,
    }
    assert (
        ScipyLBFGSBOptions.evaluation_budgeted(
            maxfun=10, maxcor=5, tol=1e-9, maxls=7
        ).maxls
        == 7
    )


# ---------------------------------------------------------------------------
# L3: the real wrapper's budget semantics.


def test_l3_one_evaluation_per_iteration_ends_on_the_iteration_message_at_nit_equal_maxfun() -> (
    None
):
    objective = _Recorded(_spread_quadratic(20))
    box = ScipyBounds(lower=(-10.0,) * 20, upper=(10.0,) * 20)

    result = _budgeted(objective, np.zeros(20), maxfun=8, bounds=box)

    assert (result.status, result.message, result.nit) == (1, ITERATION_LIMIT_STOP, 8)
    assert _per_iteration(result) == [1] * 8
    assert result.nfev == objective.calls == 9, (
        "one past maxfun, inside the final iteration"
    )
    assert _label(result) is TerminationLabel.BUDGET_EXHAUSTED
    assert (
        lbfgsb_result_accounting_defects(result, true_evaluations=objective.calls) == ()
    )


def test_l3_backtracking_ends_on_the_evaluation_message() -> None:
    objective = _Recorded(_rosenbrock)

    result = _budgeted(objective, np.array([-1.2, 1.0]), maxfun=10)

    assert (result.status, result.message) == (1, EVALUATION_LIMIT_STOP)
    assert result.nit < 10
    assert result.nfev == objective.calls > 10
    trace = _trace_nfev(result)
    assert trace[-2] is not None and trace[-2] <= 10, (
        "the overshoot lies within the final iteration"
    )
    assert _label(result) is TerminationLabel.BUDGET_EXHAUSTED
    assert (
        lbfgsb_result_accounting_defects(result, true_evaluations=objective.calls) == ()
    )


# ---------------------------------------------------------------------------
# L3': the in-iteration memory reset and its validator.


def test_l3_prime_memory_reset_iteration_holds_more_than_maxls_evaluations() -> None:
    """Iteration 2's first search spends all 20 trials, L-BFGS-B clears its memory and retries the SAME iteration
    (``__lbfgsb.c:938-950``); the retry needs 4 more: 24 evaluations in one iteration, above r2's maxls + 1."""
    objective = _PoisonedSearch(first=2, count=20)

    result = _budgeted(objective, np.zeros(2), maxfun=500)

    assert _per_iteration(result)[:2] == [1, 24]
    assert result.nfev == objective.calls
    assert (
        lbfgsb_result_accounting_defects(result, true_evaluations=objective.calls) == ()
    )


def _accounting(
    trace,
    *,
    nit=None,
    nfev,
    true_evaluations=None,
    maxiter=100,
    maxfun=100,
    maxls=20,
    call_starts=(1,),
):
    return lbfgsb_evaluation_accounting_defects(
        trace_nfev=tuple(trace),
        call_start_iterations=tuple(call_starts),
        nit=len(trace) if nit is None else nit,
        nfev=nfev,
        true_evaluations=nfev if true_evaluations is None else true_evaluations,
        maxiter=maxiter,
        maxfun=maxfun,
        maxls=maxls,
    )


def test_l3_prime_validator_ceilings() -> None:
    one_per_iteration_to_maxfun = list(range(2, 101))
    assert _accounting([2, 26], nfev=26) == ()
    assert _accounting([2, 42], nfev=42) == (), (
        "2 * maxls in the final iteration is allowed"
    )
    assert _accounting([2, 43], nfev=43) != (), "2 * maxls + 1 in one iteration is not"
    assert _accounting([2, 43, 44], nfev=44) != (), (
        "the ceiling holds for every iteration"
    )
    assert (
        _accounting([*one_per_iteration_to_maxfun, 140], nfev=140, maxiter=200) == ()
    ), "maxfun + 2 * maxls is the call ceiling"
    assert (
        _accounting([*one_per_iteration_to_maxfun, 101, 102], nfev=102, maxiter=200)
        != ()
    ), "a pre-final count above maxfun: the wrapper missed a budget stop"
    assert (
        _accounting([*one_per_iteration_to_maxfun, 140], nfev=141, maxiter=200) != ()
    ), "evaluations after a NEW_X that was already past maxfun"
    assert _accounting([2, 3], nfev=3, maxiter=1) != (), "nit above maxiter"


# ---------------------------------------------------------------------------
# Every accounting entry has a populated nfev; None never skips a check.


def test_a_missing_nfev_fails_validation() -> None:
    assert _accounting([2, None, 4], nfev=4) != ()
    assert _accounting([None], nfev=1) != ()


def test_a_result_without_a_trace_fails_validation() -> None:
    objective = _Recorded(_rosenbrock)
    unbudgeted = minimize(
        objective,
        np.array([-1.2, 1.0]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(maxiter=5, maxfun=5),
    )

    assert unbudgeted.optimizer_state_trace is None, (
        "the route without the restart policy is not touched"
    )
    assert (
        lbfgsb_result_accounting_defects(unbudgeted, true_evaluations=objective.calls)
        != ()
    )


def test_entries_with_nfev_none_in_a_real_result_fail() -> None:
    objective = _Recorded(_rosenbrock)
    result = _budgeted(objective, np.array([-1.2, 1.0]), maxfun=10)
    assert result.optimizer_state_trace is not None
    stripped = dataclasses.replace(
        result,
        optimizer_state_trace=[
            dataclasses.replace(entry, nfev=None)
            for entry in result.optimizer_state_trace
        ],
    )

    assert (
        lbfgsb_result_accounting_defects(stripped, true_evaluations=objective.calls)
        != ()
    )


def test_the_optional_nfev_field_defaults_to_none_for_other_routes() -> None:
    assert (
        OptimizerStateTraceEntry(iteration=1, fun=0.0, grad_norm_inf=0.0).nfev is None
    )


# ---------------------------------------------------------------------------
# The terminal ABNORMAL tail without NEW_X.


def test_abnormal_tail_after_an_accepted_iteration() -> None:
    """maxls 5: iteration 2's first search fails (memory non-empty), memory is cleared and the retry fails too
    (``col == 0``, ``__lbfgsb.c:924-936``): 2 * maxls = 10 trial evaluations after the last NEW_X, no callback."""
    objective = _PoisonedSearch(first=2, count=10)

    result = _budgeted(objective, np.zeros(2), maxfun=500, maxls=5)

    assert (result.status, result.message, result.nit) == (2, ABNORMAL_STOP, 1)
    assert _trace_nfev(result) == [2]
    assert result.nfev == objective.calls == 12
    assert _label(result) is TerminationLabel.LINE_SEARCH_FAILED
    assert lbfgsb_result_accounting_defects(result, true_evaluations=12) == ()
    assert lbfgsb_result_accounting_defects(result, true_evaluations=13) != ()
    corrupted = dataclasses.replace(result, nfev=13)
    assert lbfgsb_result_accounting_defects(corrupted, true_evaluations=13) != (), (
        "tail of 2 * maxls + 1"
    )


def test_abnormal_tail_with_zero_accepted_iterations() -> None:
    """From x0 the first search fails with an empty memory: ABNORMAL at nit 0 after x0 plus maxls trials."""
    objective = _PoisonedSearch(first=1, count=20)

    result = _budgeted(objective, np.zeros(2), maxfun=500)

    assert (result.status, result.message, result.nit) == (2, ABNORMAL_STOP, 0)
    assert _trace_nfev(result) == []
    assert result.nfev == objective.calls == 21
    assert lbfgsb_result_accounting_defects(result, true_evaluations=21) == ()
    assert _accounting([], nfev=21) == ()
    assert _accounting([], nfev=22) != (), (
        "a call's first iteration has an empty memory, so no retry: at most maxls"
    )
    assert _accounting([], nfev=0) != (), "x0 is always evaluated"


def test_a_fresh_call_first_iteration_is_bounded_by_maxls() -> None:
    """The empty-memory iteration of every call has one search only."""
    assert _accounting([21], nfev=21) == ()
    assert _accounting([22], nfev=22) != ()
    assert _accounting([2, 3, 4, 26], nfev=26) == (), "a retry iteration: 2 * maxls"
    assert _accounting([2, 3, 4, 26], nfev=26, call_starts=(1, 4)) != (), (
        "the same count in a restarted call's first iteration"
    )
    assert _accounting([2, 3, 4, 24], nfev=24, call_starts=(1, 4)) == ()


def test_first_iteration_counts_x0_once() -> None:
    objective = _Recorded(_rosenbrock)

    result = _budgeted(objective, np.array([-1.2, 1.0]), maxfun=10)

    trace = result.optimizer_state_trace
    assert trace is not None
    for entry in trace:
        assert entry.nfev is not None
        assert objective.values[entry.nfev - 1] == entry.fun, (
            "the entry's count ends at its accepted point"
        )
    assert [entry.iteration for entry in trace] == list(range(1, result.nit + 1))


# ---------------------------------------------------------------------------
# Served starts: a restarted call's start adds no true evaluation.

_CURVATURE = 1e-4
_WALL_HEIGHT = 1e14
_WALL_WIDTH = 0.05
_TARGET = np.array([8.0, 1.0])


def _walled_quadratic(x):
    residual = x - _TARGET
    t = min(max((x[0] - 3.0) / _WALL_WIDTH, 0.0), 1.0)
    value = 0.5 * _CURVATURE * float(residual @ residual) + _WALL_HEIGHT * t * t * (
        3.0 - 2.0 * t
    )
    gradient = _CURVATURE * residual
    gradient[0] += _WALL_HEIGHT * 6.0 * t * (1.0 - t) / _WALL_WIDTH
    return value, gradient


def test_served_start_offsets_across_restarts() -> None:
    objective = _Recorded(_walled_quadratic)

    result = minimize(
        objective,
        np.array([0.0, 0.3]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(
            maxiter=60,
            maxcor=300,
            ftol=1e-15,
            gtol=1e-15,
            restart_after_nonwolfe_stop=True,
        ),
    )

    assert [event.reason for event in result.restart_log][:1] == [
        LbfgsbRestartReason.RESTARTED
    ]
    trace = result.optimizer_state_trace
    assert trace is not None and len(trace) == result.nit
    for entry in trace:
        assert entry.nfev is not None
        assert objective.values[entry.nfev - 1] == entry.fun
    assert result.nfev == objective.calls
    assert (
        lbfgsb_result_accounting_defects(result, true_evaluations=objective.calls) == ()
    )
    restart_iterations = {event.iteration for event in result.restart_log}
    counted_served = [
        (entry.nfev or 0)
        + sum(1 for iteration in restart_iterations if iteration < entry.iteration)
        for entry in trace
    ]
    assert (
        _accounting(counted_served, nfev=result.nfev, maxiter=60, maxfun=15000) != ()
    ), (
        "a trace that counted each served start as an evaluation ends above the true total"
    )


def test_trace_is_absent_on_the_route_without_restart() -> None:
    result = minimize(
        _walled_quadratic,
        np.array([0.0, 0.3]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(maxiter=60, maxcor=300, ftol=1e-15, gtol=1e-15),
    )

    assert result.optimizer_state_trace is None


@pytest.mark.parametrize("field", ["maxiter", "maxfun"])
def test_result_validator_reads_the_budgets_from_options_used(field) -> None:
    objective = _Recorded(_rosenbrock)
    result = _budgeted(objective, np.array([-1.2, 1.0]), maxfun=10)
    options = dataclasses.replace(result.options_used, **{field: 1})

    assert (
        lbfgsb_result_accounting_defects(
            dataclasses.replace(result, options_used=options),
            true_evaluations=objective.calls,
        )
        != ()
    )


# ---------------------------------------------------------------------------
# A callback on the restart route re-uses the accepted evaluation.


def test_restart_callback_makes_no_evaluation_outside_the_count() -> None:
    objective = _Recorded(_walled_quadratic)
    events = []

    result = minimize(
        objective,
        np.array([0.0, 0.3]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(
            maxiter=60,
            maxcor=300,
            ftol=1e-15,
            gtol=1e-15,
            restart_after_nonwolfe_stop=True,
        ),
        callback=events.append,
    )

    assert len(result.restart_log) >= 2
    assert result.nfev == objective.calls, (
        "every objective call is a counted evaluation"
    )
    assert (
        lbfgsb_result_accounting_defects(result, true_evaluations=objective.calls) == ()
    )
    trace = result.optimizer_state_trace
    assert trace is not None
    assert [event.iteration for event in events] == [entry.iteration for entry in trace]
    assert [event.fun for event in events] == [entry.fun for entry in trace]
    assert [event.grad_norm_inf for event in events] == [
        entry.grad_norm_inf for entry in trace
    ]
    assert np.array_equal(events[-1].x, result.x)


def test_restart_callback_results_equal_the_no_callback_run() -> None:
    options = ScipyLBFGSBOptions(
        maxiter=60, maxcor=300, ftol=1e-15, gtol=1e-15, restart_after_nonwolfe_stop=True
    )
    quiet = minimize(
        _walled_quadratic,
        np.array([0.0, 0.3]),
        driver=Driver.SCIPY_LBFGSB,
        options=options,
    )
    observed = minimize(
        _walled_quadratic,
        np.array([0.0, 0.3]),
        driver=Driver.SCIPY_LBFGSB,
        options=options,
        callback=lambda _event: None,
    )

    assert np.array_equal(quiet.x, observed.x)
    assert (quiet.fun, quiet.nit, quiet.nfev, quiet.status) == (
        observed.fun,
        observed.nit,
        observed.nfev,
        observed.status,
    )
    assert quiet.optimizer_state_trace == observed.optimizer_state_trace


def test_the_route_without_restart_keeps_its_callback_evaluation() -> None:
    """Option off, the route is unchanged: SciPy's callback re-evaluates the iterate (outside nfev)."""
    objective = _Recorded(_rosenbrock)

    result = minimize(
        objective,
        np.array([-1.2, 1.0]),
        driver=Driver.SCIPY_LBFGSB,
        options=ScipyLBFGSBOptions(maxiter=5, maxfun=100),
        callback=lambda _event: None,
    )

    assert objective.calls == result.nfev + result.nit
