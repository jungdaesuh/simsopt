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
StatusConvention = Literal["bfgs", "host-lbfgsb", "scipy-lbfgsb", "optax-lbfgs"]

_SUCCESS_STATUSES: Final[Mapping[StatusConvention, frozenset[int]]] = MappingProxyType(
    {
        "bfgs": frozenset({0}),
        "host-lbfgsb": frozenset({0, 4}),
        "scipy-lbfgsb": frozenset({0}),
        "optax-lbfgs": frozenset({0}),
    }
)
_FAILURE_REASON_BY_STATUS: Final[
    Mapping[StatusConvention, Mapping[int, StoppingReason]]
] = MappingProxyType(
    {
        "bfgs": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
                3: "nonfinite",
                6: "nonfinite",
                99: "callback-stopped",
            }
        ),
        "host-lbfgsb": MappingProxyType(
            {
                1: "iteration-limit",
                2: "evaluation-limit",
                3: "evaluation-limit",
                5: "line-search-failed",
                6: "nonfinite",
                99: "callback-stopped",
            }
        ),
        "scipy-lbfgsb": MappingProxyType(
            {
                1: "iteration-limit",
                2: "line-search-failed",
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


def status_convention_for(provider: str, method: str) -> StatusConvention:
    """Return the canonical status namespace for a supported solver lane."""

    if method == "bfgs":
        return "bfgs"
    if method == "lbfgs" and provider == "native":
        return "scipy-lbfgsb"
    if method == "lbfgs" and provider == "custom":
        return "host-lbfgsb"
    if method == "lbfgs" and provider == "optax":
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
