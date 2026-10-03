"""Typed public contracts for the JAX optimizer API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Literal, Protocol, TypeAlias

import jax
import numpy as np

from .driver import Driver
from .shared import (
    InvalidStepEvent,
    InvalidStepReason,
    LineSearchStatus,
    OptimizerStateTraceEntry,
)


@dataclass(frozen=True)
class OptionsBase:
    """Marker base for driver-specific optimizer options."""


ScalarResult: TypeAlias = float | np.floating | jax.Array
ArrayResult: TypeAlias = np.ndarray | jax.Array
OptimizerInput: TypeAlias = np.ndarray | jax.Array
ValueAndGradFn: TypeAlias = Callable[[OptimizerInput], tuple[ScalarResult, ArrayResult]]
ResidualFn: TypeAlias = Callable[[OptimizerInput], ArrayResult]


class InverseHessianOperator(Protocol):
    """SciPy-compatible limited-memory inverse-Hessian operator."""

    shape: tuple[int, int]
    n_corrs: int

    def __call__(self, vector: np.ndarray) -> np.ndarray: ...

    def todense(self) -> np.ndarray: ...


HessianInverse: TypeAlias = np.ndarray | InverseHessianOperator


class LbfgsbRestartReason(StrEnum):
    """What ``restart_after_nonwolfe_stop`` did at one relative-reduction stop."""

    # Non-Wolfe final step; a new call continues from the accepted point.
    RESTARTED = "restarted"
    # Non-Wolfe final step in a call's first iteration, whose memory was
    # already empty: not restarted, the solve ends unresolved.
    FRESH_MEMORY_STALL = "fresh_memory_stall_not_restarted"
    # Non-Wolfe final step with no iteration or evaluation left: not
    # restarted, the solve ends unresolved.
    BUDGET_EXHAUSTED = "budget_exhausted_not_restarted"
    # The final iteration's trials do not lie on one ray (L-BFGS-B retried the
    # search internally), so the step cannot be judged: not restarted, and
    # SciPy's result stands as returned.
    UNCLASSIFIED_SEARCH = "unclassified_search_not_restarted"


# ``OptimizerResult.status`` of a SciPy L-BFGS-B solve that ended on a
# recognized non-Wolfe stall it could not resume (``FRESH_MEMORY_STALL`` or
# ``BUDGET_EXHAUSTED``); ``success`` is False.  SciPy itself returns 0..2.
SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS = 7
# ``OptimizerResult.status`` of any driver whose returned state is non-finite:
# ``x``, ``fun``, or the derivative/residual data the driver returned (``jac``,
# ``residual``) holds a NaN or an infinity.  ``success`` is then False whatever
# the backend reported, and the backend's own termination is kept in
# ``raw_status`` / ``raw_success`` / ``raw_message``.  No driver's own
# vocabulary uses 8 (``STATUS_CODES``).
NONFINITE_RESULT_STATUS = 8


@dataclass(frozen=True, slots=True)
class LbfgsbRestartEvent:
    """One SciPy L-BFGS-B relative-reduction stop judged for a restart.

    See ``ScipyLBFGSBOptions.restart_after_nonwolfe_stop``.  With ``d`` the
    search direction, ``curvature_ratio`` is ``g_accepted.d / |g_base.d|``
    (the step fails the curvature condition when its magnitude exceeds 0.9);
    ``accepted_step_fraction`` is the accepted step over the first trial step
    (0 when the search accepted its base).  Both are NaN for
    ``UNCLASSIFIED_SEARCH``.  ``iteration`` counts every call's iterations so
    far.  ``fun`` and ``projected_grad_norm_inf`` (SciPy's ``projgr``) are at
    the accepted point, where a restart starts; ``first_trial_fun_ratio`` is
    ``first_trial_fun / base_fun``, with a zero base giving +-inf by the sign
    of ``first_trial_fun``, or 1.0 when that is zero too.  ``scipy_status``
    and ``scipy_message`` are that SciPy call's own termination.
    """

    iteration: int
    reason: LbfgsbRestartReason
    curvature_ratio: float
    fun: float
    projected_grad_norm_inf: float
    first_trial_fun: float
    first_trial_fun_ratio: float
    accepted_step_fraction: float
    scipy_status: int
    scipy_message: str


@dataclass(frozen=True)
class OptimizerResult:
    x: np.ndarray
    fun: float
    jac: np.ndarray | None
    nit: int
    nfev: int
    njev: int
    status: int
    success: bool
    message: str
    driver: Driver
    options_used: OptionsBase
    wallclock_s: float
    residual: np.ndarray | None = None
    residual_jacobian: np.ndarray | None = None
    hessian: np.ndarray | None = None
    hess_inv: HessianInverse | None = None
    invalid_step_log: list[InvalidStepEvent] | None = None
    optimizer_state_trace: list[OptimizerStateTraceEntry] | None = None
    # The SciPy L-BFGS-B route's non-Wolfe stops, in order; empty when there
    # was none or the driver has no such policy.
    restart_log: tuple[LbfgsbRestartEvent, ...] = ()
    # The returned fields ("x", "fun", "jac", "residual") that are non-finite;
    # empty for a finite result.  Non-empty exactly when ``status`` is
    # ``NONFINITE_RESULT_STATUS``: ``success`` is then False and the raw
    # fields hold the backend's own ``status``, ``success`` and ``message``
    # (for SciPy's all-fixed early return, which states no status, the SciPy
    # route's defined status 0; ``dispatch._scipy_termination``).
    # For a finite result the raw fields are None and ``status``, ``success``
    # and ``message`` are the route's unchanged.  ``x``, ``fun`` and ``jac``
    # are always the backend's returned state, never a substitute.
    nonfinite_fields: tuple[str, ...] = ()
    # Every Jacobian the route evaluated, where that differs from ``njev``'s
    # solver meaning: lm-minpack reports MINPACK's njev (one per outer
    # iteration, as SciPy does) and also evaluates J at the accepted end point
    # for the result.  None for routes that report only ``njev``.
    jacobian_evaluations: int | None = None
    raw_status: int | None = None
    raw_success: bool | None = None
    raw_message: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class _OptimizerCallbackEventBase:
    iteration: int
    x: np.ndarray
    fun: float
    grad_norm_inf: float
    wallclock_s: float


@dataclass(frozen=True, kw_only=True, slots=True)
class ScipyLBFGSBCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SCIPY_LBFGSB] = Driver.SCIPY_LBFGSB


@dataclass(frozen=True, kw_only=True, slots=True)
class ScipyBFGSCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SCIPY_BFGS] = Driver.SCIPY_BFGS


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLBFGSBCallbackEvent(_OptimizerCallbackEventBase):
    accepted_alpha: float
    num_linesearch_steps: int
    driver: Literal[Driver.SIMSOPT_LBFGSB] = Driver.SIMSOPT_LBFGSB


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptBFGSCallbackEvent(_OptimizerCallbackEventBase):
    accepted_alpha: float
    num_linesearch_steps: int
    driver: Literal[Driver.SIMSOPT_BFGS] = Driver.SIMSOPT_BFGS


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMQRCallbackEvent(_OptimizerCallbackEventBase):
    residual_norm: float
    damping: float
    minpack_info: int | None
    driver: Literal[Driver.SIMSOPT_LM_QR] = Driver.SIMSOPT_LM_QR


OptimizerCallbackEvent: TypeAlias = (
    ScipyLBFGSBCallbackEvent
    | ScipyBFGSCallbackEvent
    | SimsoptLBFGSBCallbackEvent
    | SimsoptBFGSCallbackEvent
    | SimsoptLMQRCallbackEvent
)
Callback: TypeAlias = Callable[[OptimizerCallbackEvent], None]


# Each driver's own vocabulary, then the shared result boundary's
# ``NONFINITE_RESULT_STATUS``, which any driver's result can carry.
STATUS_CODES: dict[Driver, tuple[int, ...]] = {
    driver: (*codes, NONFINITE_RESULT_STATUS)
    for driver, codes in {
        Driver.SCIPY_LBFGSB: (0, 1, 2, 6, SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS),
        Driver.SCIPY_LM: (-1, 0, 1, 2, 3, 4),
        Driver.SCIPY_BFGS: (0, 1, 2, 3, 6),
        Driver.SIMSOPT_LBFGSB: (0, 1, 2, 3, 4, 5, 6),
        Driver.SIMSOPT_BFGS: (-1, 0, 1, 2, 3, 5, 99),
        Driver.SIMSOPT_LM_QR: (0, 1, 2),
    }.items()
}


__all__ = [
    "NONFINITE_RESULT_STATUS",
    "SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS",
    "STATUS_CODES",
    "ArrayResult",
    "Callback",
    "Driver",
    "HessianInverse",
    "InvalidStepEvent",
    "InvalidStepReason",
    "InverseHessianOperator",
    "LbfgsbRestartEvent",
    "LbfgsbRestartReason",
    "LineSearchStatus",
    "OptimizerCallbackEvent",
    "OptimizerInput",
    "OptimizerResult",
    "OptimizerStateTraceEntry",
    "OptionsBase",
    "ResidualFn",
    "ScalarResult",
    "ValueAndGradFn",
]
