"""Fail-closed scientific certification for host-driven optimization endpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from simsopt_jax.parity_tolerances import PARITY_LADDER_TOLERANCES

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
    "optax-lbfgs",
]

_SUCCESS_STATUSES: Final[Mapping[StatusConvention, frozenset[int]]] = MappingProxyType(
    {
        "scipy-bfgs": frozenset({0}),
        "private-bfgs": frozenset({0}),
        "host-bfgs": frozenset({0}),
        "scipy-lbfgsb": frozenset({0}),
        "private-lbfgsb": frozenset({0}),
        "host-lbfgsb": frozenset({0, 4}),
        "optax-lbfgs": frozenset({0}),
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
        # The Optax lane emits its own vocabulary (benchmarks runtime
        # _run_optax): 2 is a Wolfe line-search failure, never an
        # evaluation budget.
        "optax-lbfgs": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
                6: "nonfinite",
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

_TERMINAL_STATIONARITY_ATOL = cast(
    float,
    PARITY_LADDER_TOLERANCES["native_workflow"]["terminal_stationarity_atol"],
)
_TERMINAL_CONSTRAINT_NORM_ATOL = cast(
    float,
    PARITY_LADDER_TOLERANCES["native_workflow"]["terminal_constraint_norm_atol"],
)


@dataclass(frozen=True)
class OptimizationEndpointCertificate:
    """Report whether raw solver and scientific endpoint fields permit promotion."""

    success: bool
    stopping_reason: StoppingReason
    initial_stationary: bool
    terminal_stationary: bool
    constraints_satisfied: bool


def status_convention_for(
    provider: str,
    method: str,
    *,
    accepted_incumbent: bool,
) -> StatusConvention:
    """Return the emitter convention for a benchmark-runner solver lane.

    The benchmark lanes route provider+method to one concrete emitter,
    except custom BFGS, which runs the host core under accepted-incumbent
    continuation and the private on-device solver otherwise. Callers
    outside the benchmark runner (the example's host drivers) must name
    their emitter convention directly instead of using this mapping.
    """

    if provider == "custom" and method == "bfgs":
        return "host-bfgs" if accepted_incumbent else "private-bfgs"
    if accepted_incumbent:
        raise ValueError(
            "accepted-incumbent continuation exists only on the custom BFGS "
            f"lane, not provider={provider!r}, method={method!r}"
        )
    if provider == "custom" and method == "lbfgs":
        return "private-lbfgsb"
    if provider == "native" and method == "bfgs":
        return "scipy-bfgs"
    if provider == "native" and method == "lbfgs":
        return "scipy-lbfgsb"
    if provider == "optax" and method == "lbfgs":
        return "optax-lbfgs"
    raise ValueError(
        "unsupported optimizer status convention for "
        f"provider={provider!r}, method={method!r}"
    )


def _stopping_reason(
    *,
    provider_success: bool,
    provider_status: int | None,
    status_convention: StatusConvention,
    iterations: int,
    max_iterations: int,
    finite: bool,
) -> StoppingReason:
    if iterations < 0 or max_iterations <= 0 or iterations > max_iterations:
        return "failed"
    if provider_success and provider_status not in _SUCCESS_STATUSES[status_convention]:
        return "failed"
    if not finite:
        return "nonfinite"
    if provider_success:
        return "converged"
    if provider_status is not None:
        failure_reason = _FAILURE_REASON_BY_STATUS[status_convention].get(
            provider_status
        )
        if failure_reason is not None:
            if (
                status_convention,
                provider_status,
            ) in _MERGED_BUDGET_STATUSES and iterations < max_iterations:
                return "evaluation-limit"
            return failure_reason
    if iterations >= max_iterations:
        return "iteration-limit"
    return "failed"


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
        finite and initial_gradient_inf_norm <= _TERMINAL_STATIONARITY_ATOL
    )
    terminal_stationary = bool(
        finite and final_gradient_inf_norm <= _TERMINAL_STATIONARITY_ATOL
    )
    constraints_satisfied = bool(
        constraint_norm is None
        or (
            math.isfinite(constraint_norm)
            and constraint_norm >= 0.0
            and constraint_norm <= _TERMINAL_CONSTRAINT_NORM_ATOL
        )
    )
    accepted_step_contract = iterations > 0 or initial_stationary
    success = bool(
        valid_budget
        and inner_success
        and provider_success
        and stopping_reason == "converged"
        and terminal_stationary
        and constraints_satisfied
        and accepted_step_contract
    )
    return OptimizationEndpointCertificate(
        success=success,
        stopping_reason=stopping_reason,
        initial_stationary=initial_stationary,
        terminal_stationary=terminal_stationary,
        constraints_satisfied=constraints_satisfied,
    )
