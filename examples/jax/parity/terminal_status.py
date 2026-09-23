"""Normalized terminal status of an optimizer-backed parity lane.

A lane's label states what its optimizer reported, never what its objective did.
The per-emitter status tables and the stopping-reason vocabulary are owned by
:mod:`simsopt_contracts.optimization_endpoint`; this module only folds one
stopping reason per workflow stage, plus the case-owned scientific predicate,
into the arbiter's existing vocabulary:

* the predicate is false, or any stage stopped for a reason other than its own
  convergence or a declared budget: ``failed``;
* every stage reported convergence: ``converged``;
* otherwise (every stage converged or stopped on its budget, at least one on
  its budget): ``budget_exhausted``.

``success`` is true only for ``converged``. A finite, decreasing objective is a
scientific predicate; it never promotes a budget stop to convergence and a
budget stop never demotes it to failure. Pure: no I/O, no JAX, no globals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    StoppingReason,
    certify_optimization_endpoint,
)

NormalizedTerminalStatus = Literal["converged", "budget_exhausted", "failed"]

BUDGET_STOPPING_REASONS: Final[frozenset[StoppingReason]] = frozenset(
    {"iteration-limit", "evaluation-limit"}
)

# Solver driver id -> the emitter whose status vocabulary that driver publishes.
# ``scipy_lbfgsb`` is scipy.optimize.minimize(method="L-BFGS-B") on the host, on
# either lane; ``simsopt_lbfgsb`` publishes ``lbfgsb_public_status_from_state``
# of the private on-device L-BFGS-B. A driver that is not listed is a KeyError:
# a status is never read through a guessed table.
_STATUS_CONVENTION_BY_DRIVER: Final[Mapping[str, StatusConvention]] = MappingProxyType(
    {
        "scipy_lbfgsb": "scipy-lbfgsb",
        "simsopt_lbfgsb": "private-lbfgsb",
        # The three official tiny least-squares mirrors, both lanes: host
        # scipy.optimize.least_squares(method="trf", jac="2-point"), which is
        # what ``least_squares_serial_solve(grad=None)`` runs.
        "scipy_least_squares_trf_jax_quadratic": "scipy-trf",
        "scipy_least_squares_trf_jax_curve_length": "scipy-trf",
        "scipy_least_squares_trf_jax_surf_vol_area": "scipy-trf",
    }
)


@dataclass(frozen=True)
class StageTermination:
    """Raw provider state of one optimizer stage, named by its emitter convention.

    The gradients and the endpoint only feed the contract's finiteness guard; a
    stage whose endpoint is not finite reports ``nonfinite`` whatever the
    provider claimed.
    """

    status_convention: StatusConvention
    provider_success: bool
    provider_status: int
    iterations: int
    max_iterations: int
    initial_gradient: NDArray[np.float64]
    final_gradient: NDArray[np.float64]
    final_parameters: NDArray[np.float64]
    final_objective: float


@dataclass(frozen=True)
class TerminalStatus:
    normalized_status: NormalizedTerminalStatus
    success: bool
    stage_stopping_reasons: tuple[StoppingReason, ...]


def status_convention_for_driver(driver: str) -> StatusConvention:
    """Emitter convention of the solver driver a lane actually ran."""

    return _STATUS_CONVENTION_BY_DRIVER[driver]


def stage_termination_from_values(
    *,
    status_convention: StatusConvention,
    provider_success: bool,
    provider_status: int,
    iterations: int,
    max_iterations: int,
    start: tuple[str, Mapping[str, NDArray[np.float64]]],
    end: tuple[str, Mapping[str, NDArray[np.float64]]],
    gradient_observable: str,
) -> StageTermination:
    """Build one stage from the lane's published ``phase:observable`` values."""

    start_phase, start_values = start
    end_phase, end_values = end
    return StageTermination(
        status_convention=status_convention,
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=max_iterations,
        initial_gradient=start_values[f"{start_phase}:{gradient_observable}"],
        final_gradient=end_values[f"{end_phase}:{gradient_observable}"],
        final_parameters=end_values[f"{end_phase}:parameters"],
        final_objective=float(end_values[f"{end_phase}:objective"]),
    )


def stage_stopping_reason(stage: StageTermination) -> StoppingReason:
    """Classify one stage with the contract's emitter tables."""

    return certify_optimization_endpoint(
        status_convention=stage.status_convention,
        provider_success=stage.provider_success,
        provider_status=stage.provider_status,
        iterations=stage.iterations,
        max_iterations=stage.max_iterations,
        initial_gradient_inf_norm=float(np.max(np.abs(stage.initial_gradient))),
        final_gradient_inf_norm=float(np.max(np.abs(stage.final_gradient))),
        parameters_finite=bool(np.all(np.isfinite(stage.final_parameters))),
        observables_finite=bool(np.isfinite(stage.final_objective)),
        inner_success=True,
    ).stopping_reason


def normalized_terminal_status(
    *,
    scientific_predicate: bool,
    stage_stopping_reasons: Sequence[StoppingReason],
) -> TerminalStatus:
    """Fold per-stage stopping reasons and the scientific predicate into one label."""

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


def lane_terminal_status(
    *,
    scientific_predicate: bool,
    stages: Sequence[StageTermination],
) -> TerminalStatus:
    """Terminal status of a lane from the raw provider state of each stage."""

    return normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=tuple(stage_stopping_reason(stage) for stage in stages),
    )
