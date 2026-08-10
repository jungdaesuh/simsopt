"""Public prepared interface for the callback-free fused L-BFGS solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from .private import PreparedLBFGS as _PrivatePreparedLBFGS
from .private import PreparedParametricLBFGS as _PrivatePreparedParametricLBFGS
from .private import prepare_lbfgs_private as _prepare_lbfgs_private
from .private import (
    prepare_parametric_lbfgs_private as _prepare_parametric_lbfgs_private,
)
from .private._types import _LBFGSInvalidStepLog, _LBFGSResults


@dataclass(frozen=True, slots=True)
class FusedLBFGSOptions:
    """Immutable numerical policy for a prepared fused L-BFGS program."""

    history_size: int = 10
    function_tolerance: float = 0.0
    gradient_tolerance: float = 1.0e-5
    maximum_line_search_steps: int = 20


_DEFAULT_FUSED_LBFGS_OPTIONS = FusedLBFGSOptions()


class FusedLBFGSState(NamedTuple):
    """Device-array optimizer state retained at the terminal point."""

    parameters: jax.Array
    objective_value: jax.Array
    gradient: jax.Array
    inverse_hessian_s: jax.Array
    inverse_hessian_y: jax.Array
    inverse_hessian_rho: jax.Array
    inverse_hessian_scale: jax.Array
    inverse_hessian_corrections: jax.Array


class FusedLBFGSInvalidStepLog(NamedTuple):
    """Fixed-shape device-array diagnostics for an invalid terminal step."""

    event_count: jax.Array
    write_index: jax.Array
    iteration: jax.Array
    step_scale: jax.Array
    line_search_failed: jax.Array
    nonfinite_step: jax.Array
    stalled_step: jax.Array
    valid_curvature: jax.Array
    trial_converged: jax.Array
    line_search_status: jax.Array
    requested_initial_step: jax.Array
    first_tested_alpha: jax.Array
    best_finite_alpha: jax.Array
    returned_alpha: jax.Array
    failure_reason: jax.Array
    armijo_margin: jax.Array
    curvature_margin: jax.Array


class FusedLBFGSResult(NamedTuple):
    """Public device-array result of one fused L-BFGS solve.

    ``evaluated_nonfinite_count`` includes every nonfinite value/gradient
    evaluation. ``all_accepted_states_finite`` latches false if any accepted
    ``NEW_X`` state has nonfinite parameters, value, or gradient.
    """

    state: FusedLBFGSState
    converged: jax.Array
    failed: jax.Array
    iterations: jax.Array
    function_evaluations: jax.Array
    gradient_evaluations: jax.Array
    status: jax.Array
    line_search_status: jax.Array
    evaluated_nonfinite_count: jax.Array
    all_accepted_states_finite: jax.Array
    invalid_step_log: FusedLBFGSInvalidStepLog
    task: jax.Array


_PreparedRun = Callable[[jax.Array, int, int | None], FusedLBFGSResult]
_PreparedParametricRun = Callable[[jax.Array, jax.Array], FusedLBFGSResult]


@dataclass(frozen=True, slots=True)
class PreparedFusedLBFGS:
    """One fixed-shape compiled solver whose run budgets remain dynamic."""

    initial_value: jax.Array
    initial_gradient: jax.Array
    history_size: int
    _run_prepared: _PreparedRun = field(repr=False, compare=False)

    def run(
        self,
        x0: jax.Array,
        *,
        maxiter: int,
        maxfun: int | None = None,
    ) -> FusedLBFGSResult:
        """Run the compiled callback-free solve for the supplied budgets."""

        return self._run_prepared(x0, maxiter, maxfun)


@dataclass(frozen=True, slots=True)
class PreparedParametricFusedLBFGS:
    """Composable fused solve with one explicit fixed-shape device parameter."""

    initial_value: jax.Array
    initial_gradient: jax.Array
    history_size: int
    parameter_shape: tuple[int, ...]
    parameter_dtype: str
    maximum_iterations: int
    maximum_function_evaluations: int
    _run_staged: _PreparedParametricRun = field(repr=False, compare=False)

    def run_staged(
        self,
        x0: jax.Array,
        objective_parameter: jax.Array,
    ) -> FusedLBFGSResult:
        """Stage or execute the solve without introducing a host boundary."""

        parameter = jnp.asarray(objective_parameter)
        if parameter.shape != self.parameter_shape:
            raise ValueError(
                "parametric fused L-BFGS objective parameter shape must remain "
                f"{self.parameter_shape}, got {parameter.shape}"
            )
        if str(parameter.dtype) != self.parameter_dtype:
            raise TypeError(
                "parametric fused L-BFGS objective parameter dtype must remain "
                f"{self.parameter_dtype}, got {parameter.dtype}"
            )
        return self._run_staged(x0, parameter)


def _public_invalid_step_log(
    log: _LBFGSInvalidStepLog,
) -> FusedLBFGSInvalidStepLog:
    return FusedLBFGSInvalidStepLog(
        event_count=cast(jax.Array, log.count),
        write_index=cast(jax.Array, log.write_index),
        iteration=log.iteration,
        step_scale=log.step_scale,
        line_search_failed=log.line_search_failed,
        nonfinite_step=log.nonfinite_step,
        stalled_step=log.stalled_step,
        valid_curvature=log.valid_curvature,
        trial_converged=log.trial_converged,
        line_search_status=log.ls_status,
        requested_initial_step=log.requested_initial_step,
        first_tested_alpha=log.first_tested_alpha,
        best_finite_alpha=log.best_finite_alpha,
        returned_alpha=log.returned_alpha,
        failure_reason=log.failure_reason,
        armijo_margin=log.armijo_margin,
        curvature_margin=log.curvature_margin,
    )


def _public_result(result: _LBFGSResults) -> FusedLBFGSResult:
    return FusedLBFGSResult(
        state=FusedLBFGSState(
            parameters=result.x_k,
            objective_value=result.f_k,
            gradient=result.g_k,
            inverse_hessian_s=cast(jax.Array, result.hess_inv_s),
            inverse_hessian_y=cast(jax.Array, result.hess_inv_y),
            inverse_hessian_rho=result.rho_history,
            inverse_hessian_scale=cast(jax.Array, result.gamma),
            inverse_hessian_corrections=cast(jax.Array, result.hess_inv_n_corrs),
        ),
        converged=result.converged,
        failed=result.failed,
        iterations=cast(jax.Array, result.k),
        function_evaluations=cast(jax.Array, result.nfev),
        gradient_evaluations=cast(jax.Array, result.ngev),
        status=cast(jax.Array, result.status),
        line_search_status=cast(jax.Array, result.ls_status),
        evaluated_nonfinite_count=result.evaluated_nonfinite_count,
        all_accepted_states_finite=result.all_accepted_states_finite,
        invalid_step_log=_public_invalid_step_log(result.invalid_step_log),
        task=cast(jax.Array, result.task),
    )


def _bind_public_run(prepared: _PrivatePreparedLBFGS) -> _PreparedRun:
    def run(
        x0: jax.Array,
        maxiter: int,
        maxfun: int | None,
    ) -> FusedLBFGSResult:
        return _public_result(prepared.run(x0, maxiter=maxiter, maxfun=maxfun))

    return run


def _bind_public_parametric_run(
    prepared: _PrivatePreparedParametricLBFGS,
) -> _PreparedParametricRun:
    def run(
        x0: jax.Array,
        objective_parameter: jax.Array,
    ) -> FusedLBFGSResult:
        return _public_result(prepared.run_staged(x0, objective_parameter))

    return run


def prepare_fused_lbfgs(
    objective: Callable[[jax.Array], jax.Array],
    x0: jax.typing.ArrayLike,
    *,
    options: FusedLBFGSOptions = _DEFAULT_FUSED_LBFGS_OPTIONS,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> PreparedFusedLBFGS:
    """Compile one fixed-shape, callback-free L-BFGS solve.

    Preparation compiles the single on-device ``lax.while_loop`` executable.
    Repeated runs reuse it while accepting dynamic ``maxiter`` and ``maxfun``
    budgets; each run must preserve the prepared parameter shape and dtype.
    """

    prepared = _prepare_lbfgs_private(
        objective,
        x0,
        maxcor=options.history_size,
        ftol=options.function_tolerance,
        gtol=options.gradient_tolerance,
        maxls=options.maximum_line_search_steps,
        x_dtype=x_dtype,
        run_mode="fused_stepwise",
    )
    return PreparedFusedLBFGS(
        initial_value=prepared.initial_value,
        initial_gradient=prepared.initial_gradient,
        history_size=prepared.history_size,
        _run_prepared=_bind_public_run(prepared),
    )


def prepare_parametric_fused_lbfgs(
    objective: Callable[[jax.Array, jax.Array], jax.Array],
    x0: jax.typing.ArrayLike,
    objective_parameter: jax.typing.ArrayLike,
    *,
    maximum_iterations: int,
    maximum_function_evaluations: int,
    options: FusedLBFGSOptions = _DEFAULT_FUSED_LBFGS_OPTIONS,
    x_dtype: jax.typing.DTypeLike | None = None,
) -> PreparedParametricFusedLBFGS:
    """Stage a fixed-budget fused solve with one dynamic device parameter."""

    prepared = _prepare_parametric_lbfgs_private(
        objective,
        x0,
        objective_parameter,
        maxiter=maximum_iterations,
        maxfun=maximum_function_evaluations,
        maxcor=options.history_size,
        ftol=options.function_tolerance,
        gtol=options.gradient_tolerance,
        maxls=options.maximum_line_search_steps,
        x_dtype=x_dtype,
    )
    return PreparedParametricFusedLBFGS(
        initial_value=prepared.initial_value,
        initial_gradient=prepared.initial_gradient,
        history_size=prepared.history_size,
        parameter_shape=prepared.parameter_shape,
        parameter_dtype=prepared.parameter_dtype,
        maximum_iterations=maximum_iterations,
        maximum_function_evaluations=maximum_function_evaluations,
        _run_staged=_bind_public_parametric_run(prepared),
    )


__all__ = (
    "FusedLBFGSInvalidStepLog",
    "FusedLBFGSOptions",
    "FusedLBFGSResult",
    "FusedLBFGSState",
    "PreparedFusedLBFGS",
    "PreparedParametricFusedLBFGS",
    "prepare_fused_lbfgs",
    "prepare_parametric_fused_lbfgs",
)
