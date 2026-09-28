"""Typed public contracts for the JAX optimizer API."""

from __future__ import annotations

import hashlib
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
    optimistix_result: str | None = None
    optimistix_result_message: str | None = None
    # The SciPy L-BFGS-B route's non-Wolfe stops, in order; empty when there
    # was none or the driver has no such policy.
    restart_log: tuple[LbfgsbRestartEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class OptimizerResultFingerprint:
    driver: Driver
    status: int
    success: bool
    x_shape: tuple[int, ...]
    x_dtype: str
    x_digest_blake2b: str


def fingerprint_optimizer_result(result: OptimizerResult) -> OptimizerResultFingerprint:
    x = np.ascontiguousarray(result.x)
    digest = hashlib.blake2b(x.view(np.uint8), digest_size=16).hexdigest()
    return OptimizerResultFingerprint(
        driver=result.driver,
        status=result.status,
        success=result.success,
        x_shape=tuple(int(dim) for dim in x.shape),
        x_dtype=x.dtype.str,
        x_digest_blake2b=digest,
    )


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
class OptaxLBFGSCallbackEvent(_OptimizerCallbackEventBase):
    learning_rate: float
    num_linesearch_steps: int
    decrease_error: float
    curvature_error: float
    driver: Literal[Driver.OPTAX_LBFGS] = Driver.OPTAX_LBFGS


@dataclass(frozen=True, kw_only=True, slots=True)
class OptaxAdamCallbackEvent(_OptimizerCallbackEventBase):
    learning_rate: float
    driver: Literal[Driver.OPTAX_ADAM] = Driver.OPTAX_ADAM


@dataclass(frozen=True, kw_only=True, slots=True)
class OptimistixLBFGSCallbackEvent(_OptimizerCallbackEventBase):
    history_length: int
    optimistix_result: str
    driver: Literal[Driver.OPTIMISTIX_LBFGS] = Driver.OPTIMISTIX_LBFGS


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
class SimsoptTraceLBFGSCallbackEvent(_OptimizerCallbackEventBase):
    accepted_alpha: float
    rejected_alphas: tuple[float, ...]
    line_search_status: LineSearchStatus
    invalid_step_reason: InvalidStepReason | None
    driver: Literal[Driver.SIMSOPT_TRACE_LBFGS] = Driver.SIMSOPT_TRACE_LBFGS


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptAdamHostCallbackEvent(_OptimizerCallbackEventBase):
    learning_rate: float
    driver: Literal[Driver.SIMSOPT_ADAM_HOST] = Driver.SIMSOPT_ADAM_HOST


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptAdamCallbackEvent(_OptimizerCallbackEventBase):
    learning_rate: float
    driver: Literal[Driver.SIMSOPT_ADAM] = Driver.SIMSOPT_ADAM


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMGMRESHostCallbackEvent(_OptimizerCallbackEventBase):
    residual_norm: float
    damping: float
    gmres_iterations: int
    driver: Literal[Driver.SIMSOPT_LM_GMRES_HOST] = Driver.SIMSOPT_LM_GMRES_HOST


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMGMRESCallbackEvent(_OptimizerCallbackEventBase):
    residual_norm: float
    damping: float
    gmres_iterations: int
    driver: Literal[Driver.SIMSOPT_LM_GMRES] = Driver.SIMSOPT_LM_GMRES


@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMQRCallbackEvent(_OptimizerCallbackEventBase):
    residual_norm: float
    damping: float
    minpack_info: int | None
    driver: Literal[Driver.SIMSOPT_LM_QR] = Driver.SIMSOPT_LM_QR


OptimizerCallbackEvent: TypeAlias = (
    ScipyLBFGSBCallbackEvent
    | ScipyBFGSCallbackEvent
    | OptaxLBFGSCallbackEvent
    | OptaxAdamCallbackEvent
    | OptimistixLBFGSCallbackEvent
    | SimsoptLBFGSBCallbackEvent
    | SimsoptBFGSCallbackEvent
    | SimsoptTraceLBFGSCallbackEvent
    | SimsoptAdamHostCallbackEvent
    | SimsoptAdamCallbackEvent
    | SimsoptLMGMRESHostCallbackEvent
    | SimsoptLMGMRESCallbackEvent
    | SimsoptLMQRCallbackEvent
)
Callback: TypeAlias = Callable[[OptimizerCallbackEvent], None]


STATUS_CODES: dict[Driver, tuple[int, ...]] = {
    Driver.SCIPY_LBFGSB: (0, 1, 2, 6, SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS),
    Driver.SCIPY_LM: (-1, 0, 1, 2, 3, 4),
    Driver.SCIPY_BFGS: (0, 1, 2, 3, 6),
    Driver.OPTAX_LBFGS: (0, 1, 2),
    Driver.OPTAX_ADAM: (0, 1, 2),
    Driver.OPTIMISTIX_LBFGS: (0, 1, 2),
    Driver.OPTIMISTIX_LM: (0, 1, 2),
    Driver.SIMSOPT_LBFGSB: (0, 1, 2, 3, 4, 5, 6),
    Driver.SIMSOPT_BFGS: (-1, 0, 1, 2, 3, 5, 99),
    Driver.SIMSOPT_TRACE_LBFGS: (0, 1, 2, 3, 4, 5, 6),
    Driver.SIMSOPT_ADAM_HOST: (0, 1, 2),
    Driver.SIMSOPT_ADAM: (0, 1, 2),
    Driver.SIMSOPT_LM_GMRES_HOST: (0, 1, 2),
    Driver.SIMSOPT_LM_GMRES: (0, 1, 2),
    Driver.SIMSOPT_LM_QR: (0, 1, 2),
}


__all__ = [
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
    "OptimizerResultFingerprint",
    "OptimizerStateTraceEntry",
    "OptionsBase",
    "ResidualFn",
    "ScalarResult",
    "ValueAndGradFn",
    "fingerprint_optimizer_result",
]
