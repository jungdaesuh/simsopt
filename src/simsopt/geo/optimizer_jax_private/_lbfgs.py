"""Host-dispatched L-BFGS over cached JAX value-and-gradient kernels.

The two-loop recursion follows Nocedal & Wright, *Numerical Optimization*,
Algorithm 7.4; the step selection is the shared host strong-Wolfe line search.
"""

from __future__ import annotations

import numpy as np

import jax

from ..optimizer_host_lbfgs import (
    line_search_failure_reason_to_code,
    line_search_value_and_grad_host as _line_search_value_and_grad_host,
    minimize_lbfgs_host_core,
)
from ..optimizer_jax import (
    _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
    _prepare_optimizer_callable_inputs,
)
from ._common import (
    _as_jax_dtype,
    _bool_scalar,
    _cached_private_solver,
    _int_scalar,
    _require_private_optimizer_runtime,
    _resolve_lbfgs_limits,
    _scalar_value_and_grad,
)
from ._types import (
    _LBFGSInvalidStepLog,
    _LBFGSResults,
)


# Cap the on-device rejected-step ring buffer so capacity does not scale with
# maxiter. The retry policy only reads the most recent event, so a small bound
# is sufficient and keeps the buffer size independent of problem dimension.
_INVALID_STEP_LOG_MAX_CAPACITY = 256


def _as_host_array(value, *, dtype):
    return np.asarray(jax.device_get(value), dtype=np.dtype(dtype))


def _as_host_scalar(value, *, dtype):
    return _as_host_array(value, dtype=dtype).reshape(()).item()


def _as_device_array(value, *, dtype):
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _as_device_scalar(value, *, dtype):
    return _as_device_array(
        np.asarray(value, dtype=np.dtype(dtype)).reshape(()),
        dtype=dtype,
    )


def _coerce_value_and_grad_result(fun, x):
    value, grad = fun(x)
    value = _as_jax_dtype(value, x.dtype)
    grad = _as_jax_dtype(grad, x.dtype)
    if grad.shape != x.shape:
        raise ValueError(
            "On-device explicit value-and-gradient objectives must return a "
            f"gradient matching x.shape={x.shape}, got {grad.shape}."
        )
    return value, grad


def _invalid_step_log_from_events(events, *, capacity, dtype):
    capacity = max(int(capacity), 1)
    recent = tuple(events)[-capacity:]
    iteration = np.zeros((capacity,), dtype=np.int32)
    step_scale = np.zeros((capacity,), dtype=np.dtype(dtype))
    line_search_failed = np.zeros((capacity,), dtype=np.bool_)
    nonfinite_step = np.zeros((capacity,), dtype=np.bool_)
    stalled_step = np.zeros((capacity,), dtype=np.bool_)
    valid_curvature = np.zeros((capacity,), dtype=np.bool_)
    trial_converged = np.zeros((capacity,), dtype=np.bool_)
    ls_status = np.zeros((capacity,), dtype=np.int32)
    requested_initial_step = np.zeros((capacity,), dtype=np.dtype(dtype))
    first_tested_alpha = np.zeros((capacity,), dtype=np.dtype(dtype))
    best_finite_alpha = np.zeros((capacity,), dtype=np.dtype(dtype))
    returned_alpha = np.zeros((capacity,), dtype=np.dtype(dtype))
    failure_reason = np.zeros((capacity,), dtype=np.int32)
    armijo_margin = np.full((capacity,), np.nan, dtype=np.dtype(dtype))
    curvature_margin = np.full((capacity,), np.nan, dtype=np.dtype(dtype))

    for index, event in enumerate(recent):
        iteration[index] = event.iteration
        step_scale[index] = event.step_scale
        line_search_failed[index] = event.line_search_failed
        nonfinite_step[index] = event.nonfinite_step
        stalled_step[index] = event.stalled_step
        valid_curvature[index] = event.valid_curvature
        trial_converged[index] = event.trial_converged
        ls_status[index] = event.ls_status
        requested_initial_step[index] = event.requested_initial_step
        first_tested_alpha[index] = event.first_tested_alpha
        best_finite_alpha[index] = event.best_finite_alpha
        returned_alpha[index] = event.returned_alpha
        failure_reason[index] = line_search_failure_reason_to_code(event.failure_reason)
        armijo_margin[index] = event.armijo_margin
        curvature_margin[index] = event.curvature_margin

    return _LBFGSInvalidStepLog(
        count=_int_scalar(len(recent)),
        write_index=_int_scalar(len(recent) % capacity),
        iteration=_as_device_array(iteration, dtype=np.int32),
        step_scale=_as_device_array(step_scale, dtype=dtype),
        line_search_failed=_as_device_array(line_search_failed, dtype=np.bool_),
        nonfinite_step=_as_device_array(nonfinite_step, dtype=np.bool_),
        stalled_step=_as_device_array(stalled_step, dtype=np.bool_),
        valid_curvature=_as_device_array(valid_curvature, dtype=np.bool_),
        trial_converged=_as_device_array(trial_converged, dtype=np.bool_),
        ls_status=_as_device_array(ls_status, dtype=np.int32),
        requested_initial_step=_as_device_array(requested_initial_step, dtype=dtype),
        first_tested_alpha=_as_device_array(first_tested_alpha, dtype=dtype),
        best_finite_alpha=_as_device_array(best_finite_alpha, dtype=dtype),
        returned_alpha=_as_device_array(returned_alpha, dtype=dtype),
        failure_reason=_as_device_array(failure_reason, dtype=np.int32),
        armijo_margin=_as_device_array(armijo_margin, dtype=dtype),
        curvature_margin=_as_device_array(curvature_margin, dtype=dtype),
    )


def _host_state_to_lbfgs_results(
    state,
    *,
    invalid_log_capacity,
    dtype,
    optimizer_state_trace=(),
):
    return _LBFGSResults(
        converged=_bool_scalar(state.converged),
        failed=_bool_scalar(state.failed),
        k=_int_scalar(state.k),
        nfev=_int_scalar(state.nfev),
        ngev=_int_scalar(state.ngev),
        x_k=_as_device_array(state.x_k, dtype=dtype),
        f_k=_as_device_scalar(state.f_k, dtype=dtype),
        g_k=_as_device_array(state.g_k, dtype=dtype),
        s_history=_as_device_array(state.s_history, dtype=dtype),
        y_history=_as_device_array(state.y_history, dtype=dtype),
        rho_history=_as_device_array(state.rho_history, dtype=dtype),
        gamma=_as_device_scalar(state.gamma, dtype=dtype),
        status=_int_scalar(state.status),
        ls_status=_int_scalar(state.ls_status),
        invalid_step_log=_invalid_step_log_from_events(
            state.invalid_step_events,
            capacity=invalid_log_capacity,
            dtype=dtype,
        ),
        optimizer_state_trace=tuple(optimizer_state_trace),
    )


def _cached_lbfgs_value_and_grad_kernel(
    value_and_grad_fun,
    *,
    cache_owner,
    adapter,
    objective_mode,
    dtype,
    shape,
):
    structured_solver_cache_token = None
    if adapter is not None and cache_owner is not None:
        structured_solver_cache_token = getattr(
            cache_owner,
            _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
            None,
        )
    adapter_cache_key = None if adapter is None else adapter.solver_cache_key()
    can_cache_kernel = cache_owner is not None and (
        adapter is None or structured_solver_cache_token is not None
    )

    def build_kernel():
        def lbfgs_private_value_and_grad(x):
            return _coerce_value_and_grad_result(value_and_grad_fun, x)

        lbfgs_private_value_and_grad.__name__ = "lbfgs_private_value_and_grad"
        return jax.jit(lbfgs_private_value_and_grad)

    return _cached_private_solver(
        cache_owner if can_cache_kernel else None,
        cache_key=(
            "lbfgs-value-and-grad",
            str(objective_mode),
            structured_solver_cache_token,
            adapter_cache_key,
            np.dtype(dtype).str,
            tuple(int(dim) for dim in shape),
        ),
        builder=build_kernel,
    )


def _eval_value_and_grad_host(kernel, x_host, *, dtype):
    x_device = _as_device_array(x_host, dtype=dtype)
    value_device, grad_device = kernel(x_device)
    return (
        float(_as_host_scalar(value_device, dtype=dtype)),
        _as_host_array(grad_device, dtype=dtype),
    )


def _coerce_initial_value_and_grad_result_host(
    initial_value_and_grad, x_shape, *, dtype
):
    value, grad = initial_value_and_grad
    value = float(_as_host_scalar(value, dtype=dtype))
    grad = _as_host_array(grad, dtype=dtype)
    if grad.shape != tuple(x_shape):
        raise ValueError(
            "initial_value_and_grad must provide a gradient matching "
            f"x.shape={tuple(x_shape)}, got {grad.shape}."
        )
    return value, grad


def _minimize_lbfgs_private_impl(
    value_and_grad_fun,
    x0,
    *,
    cache_owner=None,
    objective_mode,
    maxiter=None,
    norm=np.inf,
    maxcor=200,
    ftol=0.0,
    gtol=1e-5,
    maxfun=None,
    maxgrad=None,
    maxls=20,
    initial_step_size=None,
    callback=None,
    progress_callback=None,
    failure_callback=None,
    initial_value_and_grad=None,
):
    value_and_grad_fun, x0, callback, adapter = _prepare_optimizer_callable_inputs(
        value_and_grad_fun,
        x0,
        value_and_grad=True,
        callback=callback,
    )
    x0 = _require_private_optimizer_runtime(x0)
    d = len(x0)
    dtype = x0.dtype
    x0_host = _as_host_array(x0, dtype=dtype)
    maxiter_limit_value, _, _ = _resolve_lbfgs_limits(
        d,
        maxiter,
        maxfun,
        maxgrad,
    )
    invalid_log_capacity = max(
        1,
        min(int(maxiter_limit_value), _INVALID_STEP_LOG_MAX_CAPACITY),
    )

    value_and_grad_kernel = _cached_lbfgs_value_and_grad_kernel(
        value_and_grad_fun,
        cache_owner=cache_owner,
        adapter=adapter,
        objective_mode=objective_mode,
        dtype=dtype,
        shape=x0.shape,
    )

    def eval_value_and_grad_host(x_host):
        return _eval_value_and_grad_host(value_and_grad_kernel, x_host, dtype=dtype)

    initial_value_and_grad_host = (
        None
        if initial_value_and_grad is None
        else _coerce_initial_value_and_grad_result_host(
            initial_value_and_grad,
            x0.shape,
            dtype=dtype,
        )
    )
    host_result = minimize_lbfgs_host_core(
        eval_value_and_grad_host,
        x0_host,
        maxiter=maxiter,
        norm=norm,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxgrad=maxgrad,
        maxls=maxls,
        initial_step_size=initial_step_size,
        callback=callback,
        progress_callback=progress_callback,
        failure_callback=failure_callback,
        initial_value_and_grad=initial_value_and_grad_host,
        line_search_value_and_grad=_line_search_value_and_grad_host,
    )
    return _host_state_to_lbfgs_results(
        host_result,
        invalid_log_capacity=invalid_log_capacity,
        dtype=dtype,
        optimizer_state_trace=host_result.optimizer_state_trace,
    )


def _minimize_lbfgs_private(
    fun,
    x0,
    *,
    maxiter=None,
    norm=np.inf,
    maxcor=200,
    ftol=0.0,
    gtol=1e-5,
    maxfun=None,
    maxgrad=None,
    maxls=20,
    initial_step_size=None,
    callback=None,
    progress_callback=None,
    failure_callback=None,
):
    return _minimize_lbfgs_private_impl(
        _scalar_value_and_grad(fun),
        x0,
        cache_owner=fun,
        objective_mode="scalar",
        maxiter=maxiter,
        norm=norm,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxgrad=maxgrad,
        maxls=maxls,
        initial_step_size=initial_step_size,
        callback=callback,
        progress_callback=progress_callback,
        failure_callback=failure_callback,
    )


def _minimize_lbfgs_private_value_and_grad(
    fun,
    x0,
    *,
    maxiter=None,
    norm=np.inf,
    maxcor=200,
    ftol=0.0,
    gtol=1e-5,
    maxfun=None,
    maxgrad=None,
    maxls=20,
    initial_step_size=None,
    callback=None,
    progress_callback=None,
    failure_callback=None,
    initial_value_and_grad=None,
):
    return _minimize_lbfgs_private_impl(
        fun,
        x0,
        cache_owner=fun,
        objective_mode="value_and_grad",
        maxiter=maxiter,
        norm=norm,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxgrad=maxgrad,
        maxls=maxls,
        initial_step_size=initial_step_size,
        callback=callback,
        progress_callback=progress_callback,
        failure_callback=failure_callback,
        initial_value_and_grad=initial_value_and_grad,
    )
