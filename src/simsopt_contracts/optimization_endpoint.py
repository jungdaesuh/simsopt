"""Fail-closed scientific certification for host-driven optimization endpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

TERMINAL_STATIONARITY_ATOL: Final[float] = 1.0e-7
TERMINAL_CONSTRAINT_NORM_ATOL: Final[float] = 1.0e-10

__all__ = (
    "BUDGET_STOPPING_REASONS",
    "SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD",
    "TERMINAL_CONSTRAINT_NORM_ATOL",
    "TERMINAL_STATIONARITY_ATOL",
    "NormalizedTerminalStatus",
    "OptimizationEndpointCertificate",
    "StatusConvention",
    "StoppingReason",
    "TerminalStatus",
    "certify_optimization_endpoint",
    "normalized_terminal_status",
    "scipy_minimize_stopping_reason",
    "stopping_reason_for_status",
)

StoppingReason = Literal[
    "converged",
    "iteration-limit",
    "evaluation-limit",
    "line-search-failed",
    "callback-stopped",
    "nonfinite",
    "failed",
]
StatusConvention = Literal[
    "scipy-bfgs",
    "private-bfgs",
    "host-bfgs",
    "scipy-lbfgsb",
    "private-lbfgsb",
    "host-lbfgsb",
    "scipy-slsqp",
    "scipy-trf",
]
#: Terminal label of one optimizer-backed workflow: the vocabulary
#: :func:`normalized_terminal_status` folds stopping reasons into.
NormalizedTerminalStatus = Literal["converged", "budget_exhausted", "failed"]

BUDGET_STOPPING_REASONS: Final[frozenset[StoppingReason]] = frozenset(
    {"iteration-limit", "evaluation-limit"}
)

_SUCCESS_STATUSES: Final[Mapping[StatusConvention, frozenset[int]]] = MappingProxyType(
    {
        "scipy-bfgs": frozenset({0}),
        "private-bfgs": frozenset({0}),
        "host-bfgs": frozenset({0}),
        "scipy-lbfgsb": frozenset({0}),
        "private-lbfgsb": frozenset({0}),
        "host-lbfgsb": frozenset({0, 4}),
        # scipy.optimize.minimize(method="SLSQP"): the result carries the
        # solver's exit mode, and success is exactly mode 0
        # (scipy/optimize/_slsqp_py.py:562-565,
        # ``success=(state_dict['mode'] == 0)``).
        "scipy-slsqp": frozenset({0}),
        # scipy.optimize.least_squares(method="trf"): a POSITIVE status names
        # the convergence criterion that fired -- 1 gtol, 2 ftol, 3 xtol,
        # 4 both ftol and xtol (scipy/optimize/_lsq/least_squares.py:23-31,
        # ``TERMINATION_MESSAGES``).
        "scipy-trf": frozenset({1, 2, 3, 4}),
    }
)
# Each table transcribes one emitter's actual vocabulary; none is shared
# across implementations because the same integer means different things
# in different solvers (private BFGS: 2=nonfinite/3=line-search, SciPy
# BFGS: 2=line-search/3=nonfinite).
_FAILURE_REASON_BY_STATUS: Final[
    Mapping[StatusConvention, Mapping[int, StoppingReason]]
] = MappingProxyType(
    {
        # scipy.optimize.minimize(method="BFGS"): 2 is precision-loss in
        # the line search, 3 is a NaN result.
        "scipy-bfgs": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
                3: "nonfinite",
            }
        ),
        # private _bfgs.py: outer failure status is 2 for a nonfinite
        # trial (inner line-search status -1) and 2 + ls_status
        # otherwise (ls 1 = failed, ls 3 = line-search budget).
        "private-bfgs": MappingProxyType(
            {
                1: "iteration-limit",
                2: "nonfinite",
                3: "line-search-failed",
                5: "line-search-failed",
                99: "callback-stopped",
            }
        ),
        # minimize_bfgs_host_core: 2 covers both a failed line search and
        # a nonfinite trial; the certificate's finite evidence separates
        # the nonfinite case before this table is consulted.
        "host-bfgs": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
            }
        ),
        # scipy.optimize.minimize(method="L-BFGS-B"): 1 merges the
        # iteration and evaluation budgets (discriminated below).
        "scipy-lbfgsb": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
            }
        ),
        # private _lbfgsb_scipy.py lbfgsb_public_status_from_state:
        # 1 merges the budgets, 2 is a finite ABNORMAL line search,
        # 6 is ABNORMAL with nonfinite state, 99 is a callback stop.
        "private-lbfgsb": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
                6: "nonfinite",
                99: "callback-stopped",
            }
        ),
        # minimize_lbfgs_host_core: distinct budget statuses
        # (1 iterations, 2 nfev, 3 ngev), 5 line-search, 6 nonfinite.
        "host-lbfgsb": MappingProxyType(
            {
                1: "iteration-limit",
                2: "evaluation-limit",
                3: "evaluation-limit",
                5: "line-search-failed",
                6: "nonfinite",
            }
        ),
        # scipy.optimize.minimize(method="SLSQP") exit modes
        # (scipy/optimize/_slsqp_py.py:373-383): 9 is the iteration limit,
        # 8 a positive directional derivative in the line search, and 2..7
        # are LSQ subproblem failures. Modes -1 and 1 are the transient
        # "evaluation required" requests the solver loop consumes
        # (``if abs(state_dict['mode']) != 1: break``, :546-547), so they are
        # never returned terminally; they are transcribed rather than omitted
        # because this table is the emitter's whole vocabulary.
        "scipy-slsqp": MappingProxyType(
            {
                -1: "failed",
                1: "failed",
                2: "failed",
                3: "failed",
                4: "failed",
                5: "failed",
                6: "failed",
                7: "failed",
                8: "line-search-failed",
                9: "iteration-limit",
            }
        ),
        # scipy.optimize.least_squares(method="trf")
        # (scipy/optimize/_lsq/least_squares.py:23-31): 0 is the evaluation
        # budget (``trf.py:405-406``, reached when nothing else fired), -1 is
        # improper input from ``leastsq``, and -2 is a callback that raised
        # ``StopIteration`` or returned True (``trf.py:402``). There is no
        # iteration limit in this emitter's vocabulary.
        "scipy-trf": MappingProxyType(
            {
                -2: "callback-stopped",
                -1: "failed",
                0: "evaluation-limit",
            }
        ),
    }
)
# Conventions whose single budget status merges the iteration and
# evaluation limits; the iteration evidence disambiguates after lookup.
_MERGED_BUDGET_STATUSES: Final[frozenset[tuple[StatusConvention, int]]] = frozenset(
    {
        ("scipy-lbfgsb", 1),
        ("private-lbfgsb", 1),
    }
)
#: Emitter convention of each ``scipy.optimize.minimize`` method whose status
#: vocabulary is transcribed above. A method that is not listed raises
#: ``KeyError``: a status is never read through a guessed table, and SLSQP's
#: status 1 does not mean what L-BFGS-B's does.
SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD: Final[Mapping[str, StatusConvention]] = (
    MappingProxyType(
        {
            "BFGS": "scipy-bfgs",
            "L-BFGS-B": "scipy-lbfgsb",
            "SLSQP": "scipy-slsqp",
        }
    )
)


@dataclass(frozen=True)
class OptimizationEndpointCertificate:
    """Report whether raw solver and scientific endpoint fields permit promotion."""

    success: bool
    stopping_reason: StoppingReason
    initial_stationary: bool
    terminal_stationary: bool
    constraints_satisfied: bool


def _failure_reason(
    status_convention: StatusConvention,
    provider_status: int | None,
) -> StoppingReason | None:
    """The emitter's own reason for this failure status, or ``None`` if its
    table does not carry it. The distinction matters: a table that maps a
    status to ``"failed"`` has classified it, a table that lacks it has not.
    """
    if provider_status is None:
        return None
    return _FAILURE_REASON_BY_STATUS[status_convention].get(provider_status)


def stopping_reason_for_status(
    *,
    status_convention: StatusConvention,
    provider_success: bool,
    provider_status: int | None,
    finite: bool,
) -> StoppingReason:
    """Classify one endpoint from the emitter's own status ALONE.

    This is the whole classification for an emitter that reports no iteration
    count: ``scipy.optimize.least_squares`` reports ``nfev`` only and its
    vocabulary has no iteration limit, so a caller that fabricated an iteration
    pair to satisfy :func:`certify_optimization_endpoint` would reach that
    function's budget arm with an invented number and turn every status its
    table does not carry into a budget stop. Here a status the table does not
    carry is ``"failed"``: a status is never guessed.

    A success flag that contradicts the convention's success set fails closed,
    and a non-finite endpoint is ``"nonfinite"`` whatever the status says --
    the same order :func:`certify_optimization_endpoint` applies.
    """
    if provider_success and provider_status not in _SUCCESS_STATUSES[status_convention]:
        return "failed"
    if not finite:
        return "nonfinite"
    if provider_success:
        return "converged"
    failure_reason = _failure_reason(status_convention, provider_status)
    return "failed" if failure_reason is None else failure_reason


def _stopping_reason(
    *,
    provider_success: bool,
    provider_status: int | None,
    status_convention: StatusConvention,
    iterations: int,
    max_iterations: int,
    finite: bool,
) -> StoppingReason:
    """The status classification, plus what ITERATION evidence adds to it."""
    if iterations < 0 or max_iterations <= 0 or iterations > max_iterations:
        return "failed"
    reason = stopping_reason_for_status(
        status_convention=status_convention,
        provider_success=provider_success,
        provider_status=provider_status,
        finite=finite,
    )
    if (
        reason == "iteration-limit"
        and (status_convention, provider_status) in _MERGED_BUDGET_STATUSES
        and iterations < max_iterations
    ):
        # One status for both budgets; the counter separates them.
        return "evaluation-limit"
    if (
        reason == "failed"
        and not provider_success
        and _failure_reason(status_convention, provider_status) is None
        and iterations >= max_iterations
    ):
        # The emitter's table does not carry this status, but the emitter DOES
        # report iterations and they reached the budget.
        return "iteration-limit"
    return reason


def accepted_step_contract(*, iterations: int, initial_stationary: bool) -> bool:
    """A zero-step endpoint is acceptable only from an already-stationary start.

    Single owner of the rule ``certify_optimization_endpoint`` folds into
    ``success``; scale-aware gates that name the conjunct explicitly must call
    this instead of re-deriving it.
    """

    return iterations > 0 or initial_stationary


def certify_optimization_endpoint(
    *,
    provider_success: bool,
    provider_status: int | None,
    status_convention: StatusConvention,
    iterations: int,
    max_iterations: int,
    initial_gradient_inf_norm: float,
    final_gradient_inf_norm: float,
    parameters_finite: bool,
    observables_finite: bool,
    inner_success: bool,
    constraint_norm: float | None = None,
) -> OptimizationEndpointCertificate:
    """Certify one endpoint from raw provider state and canonical tolerances.

    A zero-step endpoint is eligible only when the initial point is already
    stationary. Finite or decreasing values never override provider failure.
    """

    finite = bool(
        parameters_finite
        and observables_finite
        and math.isfinite(initial_gradient_inf_norm)
        and math.isfinite(final_gradient_inf_norm)
        and initial_gradient_inf_norm >= 0.0
        and final_gradient_inf_norm >= 0.0
    )
    valid_budget = 0 <= iterations <= max_iterations and max_iterations > 0
    stopping_reason = _stopping_reason(
        provider_success=provider_success,
        provider_status=provider_status,
        status_convention=status_convention,
        iterations=iterations,
        max_iterations=max_iterations,
        finite=finite,
    )
    initial_stationary = bool(
        finite and initial_gradient_inf_norm <= TERMINAL_STATIONARITY_ATOL
    )
    terminal_stationary = bool(
        finite and final_gradient_inf_norm <= TERMINAL_STATIONARITY_ATOL
    )
    constraints_satisfied = bool(
        constraint_norm is None
        or (
            math.isfinite(constraint_norm)
            and constraint_norm >= 0.0
            and constraint_norm <= TERMINAL_CONSTRAINT_NORM_ATOL
        )
    )
    accepted_step = accepted_step_contract(
        iterations=iterations,
        initial_stationary=initial_stationary,
    )
    success = bool(
        valid_budget
        and inner_success
        and provider_success
        and stopping_reason == "converged"
        and terminal_stationary
        and constraints_satisfied
        and accepted_step
    )
    return OptimizationEndpointCertificate(
        success=success,
        stopping_reason=stopping_reason,
        initial_stationary=initial_stationary,
        terminal_stationary=terminal_stationary,
        constraints_satisfied=constraints_satisfied,
    )


def scipy_minimize_stopping_reason(
    *,
    method: str,
    provider_success: bool,
    provider_status: int,
    iterations: int,
    max_iterations: int,
    endpoint_finite: bool,
) -> StoppingReason:
    """Stopping reason of one ``scipy.optimize.minimize`` call, by method.

    One owner for the routing every host driver over ``minimize`` needs: the
    method names the emitter convention, the provider's own counters carry the
    budget evidence, and the endpoint's finiteness is passed explicitly. The
    gradient norms :func:`certify_optimization_endpoint` takes feed only its
    stationarity fields, which a stopping reason does not read
    (:func:`_stopping_reason`), so they are zero here; a caller that needs
    ``initial_stationary`` / ``terminal_stationary`` or ``success`` calls that
    function itself.
    """
    return certify_optimization_endpoint(
        status_convention=SCIPY_MINIMIZE_STATUS_CONVENTION_BY_METHOD[method],
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=max_iterations,
        initial_gradient_inf_norm=0.0,
        final_gradient_inf_norm=0.0,
        parameters_finite=endpoint_finite,
        observables_finite=endpoint_finite,
        inner_success=True,
    ).stopping_reason


@dataclass(frozen=True)
class TerminalStatus:
    normalized_status: NormalizedTerminalStatus
    success: bool
    stage_stopping_reasons: tuple[StoppingReason, ...]


def normalized_terminal_status(
    *,
    scientific_predicate: bool,
    stage_stopping_reasons: Sequence[StoppingReason],
) -> TerminalStatus:
    """Fold per-stage stopping reasons and the scientific predicate into one label.

    A label states what the optimizer reported, never what the objective did:

    * the predicate is false, or any stage stopped for a reason other than its
      own convergence or a declared budget: ``failed``;
    * every stage reported convergence: ``converged``;
    * otherwise (every stage converged or stopped on its budget, at least one on
      its budget): ``budget_exhausted``.

    ``success`` is true only for ``converged``. A finite, decreasing objective is
    a scientific predicate; it never promotes a budget stop to convergence and a
    budget stop never demotes it to failure.
    """

    reasons = tuple(stage_stopping_reasons)
    if not reasons:
        raise ValueError("an optimizer-backed lane reports at least one stage")
    if not scientific_predicate or any(
        reason != "converged" and reason not in BUDGET_STOPPING_REASONS
        for reason in reasons
    ):
        normalized_status: NormalizedTerminalStatus = "failed"
    elif all(reason == "converged" for reason in reasons):
        normalized_status = "converged"
    else:
        normalized_status = "budget_exhausted"
    return TerminalStatus(
        normalized_status=normalized_status,
        success=normalized_status == "converged",
        stage_stopping_reasons=reasons,
    )
