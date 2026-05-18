"""SciPy-compatible L-BFGS-B over cached JAX value-and-gradient kernels."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ..optimizer_jax import (
    _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
    _prepare_optimizer_callable_inputs,
)
from ..optimizer_host_lbfgs import (
    LINE_SEARCH_FAILURE_REASON_FAILED,
    line_search_failure_reason_to_code,
)
from ._common import (
    _as_jax_dtype,
    _cached_private_solver,
    _normalize_lbfgs_counter_limit,
    _require_private_optimizer_runtime,
    _scalar_value_and_grad,
)
from . import _lbfgsb_scipy as lbfgsb
from ._types import (
    _LBFGSInvalidStepLog,
    _LBFGSResults,
)


_SCIPY_LBFGSB_DEFAULT_MAXITER = 15000
_SCIPY_LBFGSB_DEFAULT_MAXFUN = 15000
_DEFAULT_OPTIMIZER_STATE_TRACE_MAX_BYTES = 64 * 1024 * 1024
_TRACE_ARRAYS_PER_ENTRY = 2
_TRACE_SCALARS_PER_ENTRY = 5


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


def _cached_lbfgs_value_and_grad_kernel(
    value_and_grad_fun,
    *,
    cache_owner,
    adapter,
    objective_mode,
    dtype,
    shape,
):
    cache_owner, cache_key_prefix = _lbfgsb_cache_context(
        cache_owner,
        adapter,
        objective_mode,
        dtype,
        shape,
    )

    def build_kernel():
        def lbfgs_private_value_and_grad(x):
            return _coerce_value_and_grad_result(value_and_grad_fun, x)

        lbfgs_private_value_and_grad.__name__ = "lbfgs_private_value_and_grad"
        return jax.jit(lbfgs_private_value_and_grad)

    return _cached_private_solver(
        cache_owner,
        cache_key=("lbfgs-value-and-grad", *cache_key_prefix),
        builder=build_kernel,
    )


def _lbfgsb_cache_context(cache_owner, adapter, objective_mode, dtype, shape):
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
    return (
        cache_owner if can_cache_kernel else None,
        (
            str(objective_mode),
            structured_solver_cache_token,
            adapter_cache_key,
            np.dtype(dtype).str,
            tuple(int(dim) for dim in shape),
        ),
    )


def _lbfgsb_initial_state_kernel(
    *,
    cache_owner=None,
    cache_key_prefix=(),
    m: int,
    ftol: float,
    gtol: float,
    maxls: int,
):
    def build(x0):
        return lbfgsb.lbfgsb_initial_state(
            x0,
            m=m,
            bounds=None,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
        )

    return _cached_private_solver(
        cache_owner,
        cache_key=(
            "lbfgsb-initial-state",
            *cache_key_prefix,
            int(m),
            float(ftol),
            float(gtol),
            int(maxls),
        ),
        builder=lambda: jax.jit(build),
    )


def _lbfgsb_mainlb_kernel(
    value_and_grad,
    *,
    cache_owner=None,
    cache_key_prefix=(),
    maxiter: int,
    maxfun: int,
    accepted_step_callback=None,
):
    def run(state: lbfgsb.LbfgsbState):
        final_state = lbfgsb.lbfgsb_mainlb(
            value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=accepted_step_callback,
        )
        history = lbfgsb.lbfgsb_inverse_hessian_history(final_state)
        return _lbfgsb_state_to_lbfgs_results(
            final_state,
            history=history,
            maxiter_limit=jnp.asarray(maxiter, dtype=jnp.int32),
            maxfun_limit=jnp.asarray(maxfun, dtype=jnp.int32),
        )

    return _cached_private_solver(
        cache_owner if accepted_step_callback is None else None,
        cache_key=(
            "lbfgsb-mainlb",
            *cache_key_prefix,
            int(maxiter),
            int(maxfun),
        ),
        builder=lambda: jax.jit(run, donate_argnums=(0,)),
    )


def _resolve_scipy_lbfgsb_limits(maxiter, maxfun):
    maxiter_limit = _normalize_lbfgs_counter_limit(
        _SCIPY_LBFGSB_DEFAULT_MAXITER if maxiter is None else maxiter
    )
    maxfun_limit = _normalize_lbfgs_counter_limit(
        _SCIPY_LBFGSB_DEFAULT_MAXFUN if maxfun is None else maxfun
    )
    return maxiter_limit, maxfun_limit


def _check_lbfgsb_trace_budget(
    d: int,
    maxiter_limit,
    *,
    record_optimizer_state_trace: bool,
    max_optimizer_state_trace_bytes,
) -> None:
    if not record_optimizer_state_trace:
        return
    iterations = max(1, int(maxiter_limit))
    trace_bytes = iterations * (
        (_TRACE_ARRAYS_PER_ENTRY * int(d) + _TRACE_SCALARS_PER_ENTRY)
        * np.dtype(np.float64).itemsize
    )
    limit = (
        _DEFAULT_OPTIMIZER_STATE_TRACE_MAX_BYTES
        if max_optimizer_state_trace_bytes is None
        else int(max_optimizer_state_trace_bytes)
    )
    if trace_bytes > limit:
        raise ValueError(
            "optimizer_state_trace would allocate "
            f"{trace_bytes} bytes for d={int(d)} and iterations={iterations}, "
            f"exceeding max_optimizer_state_trace_bytes={limit}."
        )


def _resolve_lbfgs_history_size(maxcor, *, maxiter_limit) -> int:
    return max(1, min(int(maxcor), int(maxiter_limit)))


def _lbfgsb_public_status(state, *, maxiter_limit, maxfun_limit):
    task0 = state.workspace.task[0]
    limited = (state.nfev > maxfun_limit) | (state.n_iterations >= maxiter_limit)
    zero = jnp.zeros_like(task0)
    one = jnp.ones_like(task0)
    return jnp.where(
        task0 == lbfgsb.CONVERGENCE,
        zero,
        jnp.where(limited, one, one + one),
    )


def _lbfgsb_rho_history(history: lbfgsb.LbfgsbInverseHessianHistory) -> jax.Array:
    return jnp.zeros((history.s.shape[0],), dtype=history.s.dtype)


def _lbfgsb_invalid_step_log(state: lbfgsb.LbfgsbState) -> _LBFGSInvalidStepLog:
    int_zero = state.nfev - state.nfev
    int_slot = jnp.zeros_like(state.workspace.isave[:1])
    float_slot = jnp.zeros_like(state.workspace.dsave[:1])
    abnormal = state.workspace.task[0] == lbfgsb.ABNORMAL
    abnormal_slot = jnp.reshape(abnormal, (1,))
    false_slot = jnp.zeros_like(int_slot, dtype=bool)
    true_slot = jnp.ones_like(int_slot, dtype=bool)
    failed_reason = jnp.asarray(
        line_search_failure_reason_to_code(LINE_SEARCH_FAILURE_REASON_FAILED),
        dtype=state.workspace.isave.dtype,
    )
    step = jnp.where(abnormal, state.workspace.dsave[13], state.workspace.dsave[13] * 0)
    step_slot = jnp.reshape(step, (1,))
    nonfinite = (~jnp.isfinite(state.f)) | jnp.any(~jnp.isfinite(state.g))
    nonfinite_slot = jnp.reshape(abnormal & nonfinite, (1,))
    failure_reason_slot = jnp.where(abnormal_slot, failed_reason, int_slot)
    return _LBFGSInvalidStepLog(
        count=jnp.where(abnormal, int_zero + 1, int_zero),
        write_index=int_zero,
        iteration=jnp.where(
            abnormal_slot, jnp.reshape(state.n_iterations, (1,)), int_slot
        ),
        step_scale=step_slot,
        line_search_failed=abnormal_slot,
        nonfinite_step=nonfinite_slot,
        stalled_step=false_slot,
        valid_curvature=true_slot,
        trial_converged=false_slot,
        ls_status=jnp.where(abnormal_slot, state.workspace.isave[34:35], int_slot),
        requested_initial_step=step_slot,
        first_tested_alpha=step_slot,
        best_finite_alpha=float_slot,
        returned_alpha=float_slot,
        failure_reason=failure_reason_slot,
        armijo_margin=float_slot,
        curvature_margin=float_slot,
    )


def _lbfgsb_state_with_initial_value_and_grad(
    state: lbfgsb.LbfgsbState,
    initial_value_and_grad,
    *,
    dtype,
) -> lbfgsb.LbfgsbState:
    started = lbfgsb.lbfgsb_setulb(state)
    value, grad = initial_value_and_grad
    grad = _as_jax_dtype(grad, dtype)
    if grad.shape != started.x.shape:
        raise ValueError(
            "initial_value_and_grad must provide a gradient matching "
            f"x.shape={started.x.shape}, got {grad.shape}."
        )
    return started._replace(
        f=_as_jax_dtype(value, dtype),
        g=grad,
        nfev=started.nfev + 1,
        njev=started.njev + 1,
    )


def _lbfgsb_accepted_step_observer(
    *,
    callback,
    progress_callback,
    optimizer_state_trace,
    record_optimizer_state_trace: bool,
):
    if (
        callback is None
        and progress_callback is None
        and not record_optimizer_state_trace
    ):
        return None

    def observe(iteration, x, f, g, nfev, njev):
        x_host = np.asarray(x, dtype=float)
        g_host = np.asarray(g, dtype=float)
        f_host = float(np.asarray(f).reshape(()).item())
        iteration_host = int(np.asarray(iteration).reshape(()).item())
        if callback is not None:
            callback(x_host)
        if progress_callback is not None:
            progress_callback(
                iteration_host,
                f_host,
                float(np.linalg.norm(g_host, ord=np.inf)),
            )
        if record_optimizer_state_trace:
            optimizer_state_trace.append(
                {
                    "iteration": iteration_host,
                    "x": x_host,
                    "fun": f_host,
                    "jac": g_host,
                    "jac_inf_norm": float(np.linalg.norm(g_host, ord=np.inf)),
                    "nfev": int(np.asarray(nfev).reshape(()).item()),
                    "njev": int(np.asarray(njev).reshape(()).item()),
                }
            )

    return observe


def _lbfgsb_state_to_lbfgs_results(
    state: lbfgsb.LbfgsbState,
    *,
    history: lbfgsb.LbfgsbInverseHessianHistory,
    maxiter_limit,
    maxfun_limit,
    optimizer_state_trace=(),
) -> _LBFGSResults:
    status = _lbfgsb_public_status(
        state,
        maxiter_limit=maxiter_limit,
        maxfun_limit=maxfun_limit,
    )
    return _LBFGSResults(
        converged=status == jnp.zeros_like(status),
        failed=status != jnp.zeros_like(status),
        k=state.n_iterations,
        nfev=state.nfev,
        ngev=state.njev,
        x_k=state.x,
        f_k=state.f,
        g_k=state.g,
        s_history=history.s,
        y_history=history.y,
        rho_history=_lbfgsb_rho_history(history),
        gamma=state.workspace.dsave[0],
        status=status,
        ls_status=state.workspace.isave[34],
        invalid_step_log=_lbfgsb_invalid_step_log(state),
        optimizer_state_trace=tuple(optimizer_state_trace),
        hess_inv_s=history.s,
        hess_inv_y=history.y,
        hess_inv_n_corrs=history.n_corrs,
        task=state.workspace.task,
    )


def _minimize_lbfgs_private_impl(
    value_and_grad_fun,
    x0,
    *,
    cache_owner=None,
    objective_mode,
    maxiter=None,
    maxcor=10,
    ftol=0.0,
    gtol=1e-5,
    maxfun=None,
    maxls=20,
    callback=None,
    progress_callback=None,
    initial_value_and_grad=None,
    record_optimizer_state_trace=False,
    max_optimizer_state_trace_bytes=None,
):
    value_and_grad_fun, x0, callback, adapter = _prepare_optimizer_callable_inputs(
        value_and_grad_fun,
        x0,
        value_and_grad=True,
        callback=callback,
    )
    x0 = _require_private_optimizer_runtime(x0)
    dtype = x0.dtype
    maxiter_limit_value, maxfun_limit_value = _resolve_scipy_lbfgsb_limits(
        maxiter,
        maxfun,
    )
    history_size = _resolve_lbfgs_history_size(
        maxcor,
        maxiter_limit=maxiter_limit_value,
    )
    _check_lbfgsb_trace_budget(
        int(x0.size),
        maxiter_limit_value,
        record_optimizer_state_trace=record_optimizer_state_trace,
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
    )

    value_and_grad_kernel = _cached_lbfgs_value_and_grad_kernel(
        value_and_grad_fun,
        cache_owner=cache_owner,
        adapter=adapter,
        objective_mode=objective_mode,
        dtype=dtype,
        shape=x0.shape,
    )
    solver_cache_owner, solver_cache_key_prefix = _lbfgsb_cache_context(
        cache_owner,
        adapter,
        objective_mode,
        dtype,
        x0.shape,
    )

    state = _lbfgsb_initial_state_kernel(
        cache_owner=solver_cache_owner,
        cache_key_prefix=solver_cache_key_prefix,
        m=history_size,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
    )(x0)
    if initial_value_and_grad is not None:
        state = _lbfgsb_state_with_initial_value_and_grad(
            state,
            initial_value_and_grad,
            dtype=dtype,
        )
    optimizer_state_trace = []
    accepted_step_callback = _lbfgsb_accepted_step_observer(
        callback=callback,
        progress_callback=progress_callback,
        optimizer_state_trace=optimizer_state_trace,
        record_optimizer_state_trace=record_optimizer_state_trace,
    )
    result = _lbfgsb_mainlb_kernel(
        value_and_grad_kernel,
        cache_owner=solver_cache_owner,
        cache_key_prefix=solver_cache_key_prefix,
        maxiter=int(maxiter_limit_value),
        maxfun=int(maxfun_limit_value),
        accepted_step_callback=accepted_step_callback,
    )(state)
    if accepted_step_callback is not None:
        jax.effects_barrier()
    return result._replace(optimizer_state_trace=tuple(optimizer_state_trace))


def _minimize_lbfgs_private(
    fun,
    x0,
    *,
    maxiter=None,
    maxcor=10,
    ftol=0.0,
    gtol=1e-5,
    maxfun=None,
    maxls=20,
    callback=None,
    progress_callback=None,
    record_optimizer_state_trace=False,
    max_optimizer_state_trace_bytes=None,
):
    return _minimize_lbfgs_private_impl(
        _scalar_value_and_grad(fun),
        x0,
        cache_owner=fun,
        objective_mode="scalar",
        maxiter=maxiter,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxls=maxls,
        callback=callback,
        progress_callback=progress_callback,
        record_optimizer_state_trace=record_optimizer_state_trace,
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
    )


def _minimize_lbfgs_private_value_and_grad(
    fun,
    x0,
    *,
    maxiter=None,
    maxcor=10,
    ftol=0.0,
    gtol=1e-5,
    maxfun=None,
    maxls=20,
    callback=None,
    progress_callback=None,
    initial_value_and_grad=None,
    record_optimizer_state_trace=False,
    max_optimizer_state_trace_bytes=None,
):
    return _minimize_lbfgs_private_impl(
        fun,
        x0,
        cache_owner=fun,
        objective_mode="value_and_grad",
        maxiter=maxiter,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxls=maxls,
        callback=callback,
        progress_callback=progress_callback,
        initial_value_and_grad=initial_value_and_grad,
        record_optimizer_state_trace=record_optimizer_state_trace,
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
    )
