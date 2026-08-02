"""Typed public contracts for the JAX optimizer API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
    Driver.SCIPY_LBFGSB: (0, 1, 2, 6),
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
    "STATUS_CODES",
    "ArrayResult",
    "Callback",
    "Driver",
    "HessianInverse",
    "InvalidStepEvent",
    "InvalidStepReason",
    "InverseHessianOperator",
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
