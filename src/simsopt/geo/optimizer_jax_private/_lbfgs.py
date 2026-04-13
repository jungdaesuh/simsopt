"""On-device L-BFGS solver (lax.while_loop)."""

from __future__ import annotations

import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from ..optimizer_jax import _prepare_optimizer_callable_inputs
from ._common import (
    _as_jax_dtype,
    _bool_scalar,
    _cached_private_solver,
    _dot,
    _emit_iteration_callbacks,
    _int_scalar,
    _norm,
    _require_private_optimizer_runtime,
    _resolve_lbfgs_limits,
    _scalar_value_and_grad,
    _zeros,
)
from ._line_search import _line_search_value_and_grad
from ._types import _LBFGSResults


_LBFGS_DEBUG_ENABLED = os.environ.get("SIMSOPT_LBFGS_DEBUG", "").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}


def _emit_lbfgs_runtime_debug(
    stage,
    *,
    iteration,
    fun_value,
    grad,
    step_scale=None,
    ls_failed=None,
    ls_status=None,
    nonfinite_step=None,
    stalled_step=None,
    valid_curvature=None,
    converged=None,
):
    """Emit ordered runtime diagnostics when SIMSOPT_LBFGS_DEBUG is enabled."""
    if not _LBFGS_DEBUG_ENABLED:
        return

    grad_inf = jnp.max(jnp.abs(grad))
    if step_scale is None:
        step_scale = _as_jax_dtype(0.0, fun_value.dtype)
    if ls_failed is None:
        ls_failed = _bool_scalar(False)
    if ls_status is None:
        ls_status = _int_scalar(0)
    if nonfinite_step is None:
        nonfinite_step = _bool_scalar(False)
    if stalled_step is None:
        stalled_step = _bool_scalar(False)
    if valid_curvature is None:
        valid_curvature = _bool_scalar(True)
    if converged is None:
        converged = _bool_scalar(False)

    jax.debug.callback(
        lambda k, f, ginf, alpha, failed, status, nonfinite, stalled, curvature, done: (
            print(
                "[lbfgs-debug] "
                f"stage={stage} "
                f"iter={int(k)} "
                f"f={float(f):.16e} "
                f"grad_inf={float(ginf):.16e} "
                f"alpha={float(alpha):.16e} "
                f"ls_failed={bool(failed)} "
                f"ls_status={int(status)} "
                f"nonfinite_step={bool(nonfinite)} "
                f"stalled_step={bool(stalled)} "
                f"valid_curvature={bool(curvature)} "
                f"converged={bool(done)}",
                flush=True,
            )
        ),
        iteration,
        fun_value,
        grad_inf,
        step_scale,
        ls_failed,
        ls_status,
        nonfinite_step,
        stalled_step,
        valid_curvature,
        converged,
        ordered=True,
    )


def _shift_history(history, new):
    length = history.shape[0]
    if length == 1:
        return jnp.reshape(new, history.shape)
    shifted = lax.slice_in_dim(history, 1, length, axis=0)
    new_row = jnp.expand_dims(new, axis=0)
    return lax.concatenate((shifted, new_row), dimension=0)


def _update_history_vectors(history, new):
    return _shift_history(history, new)


def _update_history_scalars(history, new):
    length = history.shape[0]
    if length == 1:
        return jnp.reshape(new, history.shape)
    shifted = lax.slice_in_dim(history, 1, length, axis=0)
    return lax.concatenate((shifted, jnp.reshape(new, (1,))), dimension=0)


def _take_axis0(array, index):
    index = _as_jax_dtype(index, jnp.int32)
    return jnp.squeeze(lax.dynamic_slice_in_dim(array, index, 1, axis=0), axis=0)


def _two_loop_recursion(state):
    dtype = state.rho_history.dtype
    his_size = len(state.rho_history)
    his_size_jax = _int_scalar(his_size)
    curr_size = jnp.where(state.k < his_size_jax, state.k, his_size_jax)
    q = -jnp.conj(state.g_k)
    a_his = jnp.zeros_like(state.rho_history)

    def body_fun1(j, carry):
        i = _int_scalar(his_size - 1) - j
        _q, _a_his = carry
        rho_i = _take_axis0(state.rho_history, i)
        s_i = _take_axis0(state.s_history, i)
        y_i = _take_axis0(state.y_history, i)
        a_i = rho_i * _dot(jnp.conj(s_i), _q).real.astype(dtype)
        _a_his = _a_his.at[i].set(a_i)
        _q = _q - a_i * jnp.conj(y_i)
        return _q, _a_his

    q, a_his = lax.fori_loop(_int_scalar(0), curr_size, body_fun1, (q, a_his))
    q = state.gamma * q

    def body_fun2(j, _q):
        i = (his_size_jax - curr_size) + j
        rho_i = _take_axis0(state.rho_history, i)
        y_i = _take_axis0(state.y_history, i)
        s_i = _take_axis0(state.s_history, i)
        a_i = _take_axis0(a_his, i)
        b_i = rho_i * _dot(y_i, _q).real.astype(dtype)
        return _q + (a_i - b_i) * s_i

    return lax.fori_loop(_int_scalar(0), curr_size, body_fun2, q)


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


def _lbfgs_step_tolerances(x):
    eps = _as_jax_dtype(jnp.finfo(x.dtype).eps, x.dtype)
    step_eps = jnp.sqrt(eps)
    gamma_max = jnp.reciprocal(step_eps)
    return step_eps, gamma_max


def _resolve_lbfgs_history_size(maxcor, *, d, maxiter_limit_value):
    """Cap history to the reachable correction budget for this solve.

    More than ``d`` correction pairs are redundant in an ``d``-dimensional
    problem, and more than ``maxiter`` pairs are unreachable during this solve.
    Clipping here shrinks the traced solver state without changing the set of
    curvature updates that the run can actually observe.
    """
    return max(1, min(int(maxcor), int(d), int(maxiter_limit_value)))


def _minimize_lbfgs_private_impl(
    value_and_grad_fun,
    x0,
    *,
    cache_owner=None,
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
    value_and_grad_fun, x0, callback, adapter = _prepare_optimizer_callable_inputs(
        value_and_grad_fun,
        x0,
        value_and_grad=True,
        callback=callback,
    )
    x0 = _require_private_optimizer_runtime(x0)
    d = len(x0)
    dtype = x0.dtype
    maxiter_limit_value, maxfun_limit_value, maxgrad_limit_value = (
        _resolve_lbfgs_limits(d, maxiter, maxfun, maxgrad)
    )
    history_size = _resolve_lbfgs_history_size(
        maxcor,
        d=d,
        maxiter_limit_value=maxiter_limit_value,
    )
    initial_step_size_value = (
        None
        if initial_step_size is None
        else np.asarray(initial_step_size, dtype=np.dtype(dtype)).item()
    )
    if initial_step_size_value is not None and initial_step_size_value <= 0.0:
        raise ValueError("initial_step_size must be positive when provided.")
    ftol_value = np.asarray(ftol, dtype=np.dtype(dtype)).item()
    gtol_value = np.asarray(gtol, dtype=np.dtype(dtype)).item()

    f_0, g_0 = _coerce_value_and_grad_result(value_and_grad_fun, x0)
    state_initial = _LBFGSResults(
        converged=_norm(g_0, ord=norm) < _as_jax_dtype(gtol_value, dtype),
        failed=_bool_scalar(False),
        k=_int_scalar(0),
        nfev=_int_scalar(1),
        ngev=_int_scalar(1),
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        s_history=_zeros((history_size, d), dtype),
        y_history=_zeros((history_size, d), dtype),
        rho_history=_zeros((history_size,), dtype),
        gamma=_as_jax_dtype(1.0, dtype),
        status=_int_scalar(0),
        ls_status=_int_scalar(0),
    )
    initial_status = _int_scalar(0)
    initial_status = jnp.where(
        state_initial.ngev >= _int_scalar(maxgrad_limit_value),
        _int_scalar(3),
        initial_status,
    )
    initial_status = jnp.where(
        state_initial.nfev >= _int_scalar(maxfun_limit_value),
        _int_scalar(2),
        initial_status,
    )
    initial_status = jnp.where(
        state_initial.k >= _int_scalar(maxiter_limit_value),
        _int_scalar(1),
        initial_status,
    )
    state_initial = state_initial._replace(
        failed=(initial_status > _int_scalar(0)) & (~state_initial.converged),
        status=jnp.where(state_initial.converged, _int_scalar(0), initial_status),
    )

    def cond_fun(state):
        return (
            (~state.converged)
            & (~state.failed)
            & jnp.isfinite(state.f_k)
            & jnp.all(jnp.isfinite(state.g_k))
        )

    def body_fun(state):
        ftol_jax = _as_jax_dtype(ftol_value, state.f_k.dtype)
        gtol_jax = _as_jax_dtype(gtol_value, state.g_k.dtype)
        step_eps, gamma_max = _lbfgs_step_tolerances(state.x_k)
        maxiter_limit = _as_jax_dtype(maxiter_limit_value, state.k.dtype)
        maxfun_limit = _as_jax_dtype(maxfun_limit_value, state.nfev.dtype)
        maxgrad_limit = _as_jax_dtype(maxgrad_limit_value, state.ngev.dtype)
        # 0.0 → line search sees use_initial_step_override=False (0.0 > 0.0 is
        # False), falling back to the default Wolfe start value on later iters.
        line_search_initial_step_size = (
            None
            if initial_step_size_value is None
            else jnp.where(
                state.k == _int_scalar(0),
                _as_jax_dtype(initial_step_size_value, state.x_k.dtype),
                _as_jax_dtype(0.0, state.x_k.dtype),
            )
        )
        p_k = _two_loop_recursion(state)
        ls_results = _line_search_value_and_grad(
            fun=value_and_grad_fun,
            xk=state.x_k,
            pk=p_k,
            old_fval=state.f_k,
            gfk=state.g_k,
            initial_step_size=line_search_initial_step_size,
            maxiter=maxls,
        )
        ls_status = _as_jax_dtype(ls_results.status, state.ls_status.dtype)

        next_nfev = state.nfev + ls_results.nfev
        next_ngev = state.ngev + ls_results.ngev
        s_k = jnp.asarray(ls_results.a_k).astype(p_k.dtype) * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        y_norm_sq = jnp.real(_dot(jnp.conj(y_k), y_k))
        rho_k = jnp.reciprocal(rho_k_inv).astype(y_k.dtype)
        gamma_raw = rho_k_inv / y_norm_sq
        gamma = jnp.clip(
            _as_jax_dtype(gamma_raw, state.gamma.dtype),
            step_eps,
            gamma_max,
        )
        next_k = state.k + _int_scalar(1)
        s_k_norm = _norm(s_k)
        y_k_norm = _norm(y_k)
        converged = _norm(g_kp1, ord=norm) < gtol_jax
        step_tol = step_eps * jnp.maximum(
            _as_jax_dtype(1.0, state.x_k.dtype),
            _norm(state.x_k),
        )
        function_change = jnp.abs(state.f_k - f_kp1)
        objective_tol = step_eps * jnp.maximum(
            jnp.abs(state.f_k),
            jnp.abs(f_kp1),
        )
        gradient_change = y_k_norm
        gradient_tol = step_eps * jnp.maximum(
            _norm(state.g_k),
            _norm(g_kp1),
        )
        stalled_step = (
            (~converged)
            & (s_k_norm <= step_tol)
            & (function_change <= objective_tol)
            & (gradient_change <= gradient_tol)
        )
        curvature_scale = s_k_norm * y_k_norm
        curvature_tol = step_eps * _as_jax_dtype(curvature_scale, rho_k_inv.dtype)
        valid_curvature = (
            jnp.isfinite(rho_k_inv)
            & jnp.isfinite(y_norm_sq)
            & jnp.isfinite(curvature_scale)
            & jnp.isfinite(gamma_raw)
            & jnp.isfinite(rho_k)
            & (rho_k_inv > curvature_tol)
            & (y_norm_sq > _as_jax_dtype(0.0, y_norm_sq.dtype))
        )
        update_curvature = valid_curvature & (~stalled_step)
        nonfinite_step = (
            (~jnp.isfinite(f_kp1))
            | (~jnp.all(jnp.isfinite(g_kp1)))
            | (~jnp.all(jnp.isfinite(s_k)))
            | (~jnp.all(jnp.isfinite(x_kp1)))
            | (~jnp.all(jnp.isfinite(y_k)))
        )
        rejected_step = (
            nonfinite_step | stalled_step | ((~converged) & (~valid_curvature))
        )
        _emit_lbfgs_runtime_debug(
            "post_line_search",
            iteration=next_k,
            fun_value=f_kp1,
            grad=g_kp1,
            step_scale=ls_results.a_k,
            ls_failed=ls_results.failed,
            ls_status=ls_status,
            nonfinite_step=nonfinite_step,
            stalled_step=stalled_step,
            valid_curvature=valid_curvature,
            converged=converged,
        )

        def failed_step(_):
            if failure_callback is not None:
                jax.debug.callback(
                    lambda iteration, trial_x, trial_f, trial_g, search_direction, step_vector, step_scale, line_search_failed, nonfinite_step_detected, stalled_step_detected, valid_curvature_detected, trial_converged, line_search_status: failure_callback(
                        int(iteration),
                        np.asarray(trial_x, dtype=np.float64),
                        float(trial_f),
                        np.asarray(trial_g, dtype=np.float64),
                        np.asarray(search_direction, dtype=np.float64),
                        np.asarray(step_vector, dtype=np.float64),
                        float(step_scale),
                        bool(line_search_failed),
                        bool(nonfinite_step_detected),
                        bool(stalled_step_detected),
                        bool(valid_curvature_detected),
                        bool(trial_converged),
                        int(line_search_status),
                    ),
                    next_k,
                    x_kp1,
                    f_kp1,
                    g_kp1,
                    p_k,
                    s_k,
                    ls_results.a_k,
                    ls_results.failed,
                    nonfinite_step,
                    stalled_step,
                    valid_curvature,
                    converged,
                    ls_status,
                    ordered=True,
                )
            return state._replace(
                converged=_bool_scalar(False),
                failed=_bool_scalar(True),
                nfev=next_nfev,
                ngev=next_ngev,
                status=_int_scalar(5),
                ls_status=jnp.where(
                    ls_results.failed,
                    ls_status,
                    _int_scalar(0),
                ),
            )

        def accepted_step(_):
            status = _int_scalar(0)
            status = jnp.where(state.f_k - f_kp1 < ftol_jax, _int_scalar(4), status)
            status = jnp.where(next_ngev >= maxgrad_limit, _int_scalar(3), status)
            status = jnp.where(next_nfev >= maxfun_limit, _int_scalar(2), status)
            status = jnp.where(next_k >= maxiter_limit, _int_scalar(1), status)
            _emit_iteration_callbacks(
                callback, progress_callback, x_kp1, next_k, f_kp1, g_kp1
            )
            return state._replace(
                converged=converged,
                failed=(status > _int_scalar(0)) & (~converged),
                k=next_k,
                nfev=next_nfev,
                ngev=next_ngev,
                x_k=_as_jax_dtype(x_kp1, state.x_k.dtype),
                f_k=_as_jax_dtype(f_kp1, state.f_k.dtype),
                g_k=_as_jax_dtype(g_kp1, state.g_k.dtype),
                s_history=jnp.where(
                    update_curvature,
                    _update_history_vectors(state.s_history, s_k),
                    state.s_history,
                ),
                y_history=jnp.where(
                    update_curvature,
                    _update_history_vectors(state.y_history, y_k),
                    state.y_history,
                ),
                rho_history=jnp.where(
                    update_curvature,
                    _update_history_scalars(state.rho_history, rho_k),
                    state.rho_history,
                ),
                gamma=jnp.where(update_curvature, gamma, state.gamma),
                status=jnp.where(converged, _int_scalar(0), status),
                ls_status=ls_status,
            )

        return lax.cond(
            ls_results.failed | rejected_step,
            failed_step,
            accepted_step,
            operand=None,
        )

    def run_solver(initial_state):
        gtol_jax = _as_jax_dtype(gtol_value, initial_state.g_k.dtype)
        maxiter_limit = _as_jax_dtype(maxiter_limit_value, initial_state.k.dtype)
        maxfun_limit = _as_jax_dtype(maxfun_limit_value, initial_state.nfev.dtype)
        maxgrad_limit = _as_jax_dtype(maxgrad_limit_value, initial_state.ngev.dtype)
        state = lax.while_loop(cond_fun, body_fun, initial_state)
        _emit_lbfgs_runtime_debug(
            "pre_final_eval",
            iteration=state.k,
            fun_value=state.f_k,
            grad=state.g_k,
            ls_status=state.ls_status,
            converged=state.converged,
        )
        f_final, g_final = _coerce_value_and_grad_result(value_and_grad_fun, state.x_k)
        _emit_lbfgs_runtime_debug(
            "post_final_eval",
            iteration=state.k,
            fun_value=f_final,
            grad=g_final,
            ls_status=state.ls_status,
            converged=state.converged,
        )
        converged_final = _norm(g_final, ord=norm) < gtol_jax
        state = state._replace(
            converged=converged_final,
            nfev=state.nfev + _int_scalar(1),
            ngev=state.ngev + _int_scalar(1),
            f_k=_as_jax_dtype(f_final, state.f_k.dtype),
            g_k=_as_jax_dtype(g_final, state.g_k.dtype),
        )
        status = jnp.where(
            state.converged,
            _int_scalar(0),
            jnp.where(
                state.k >= maxiter_limit,
                _int_scalar(1),
                jnp.where(
                    state.nfev >= maxfun_limit,
                    _int_scalar(2),
                    jnp.where(
                        state.ngev >= maxgrad_limit,
                        _int_scalar(3),
                        jnp.where(state.failed, _int_scalar(5), state.status),
                    ),
                ),
            ),
        )
        return state._replace(status=status)

    run_solver.__name__ = "lbfgs_private_run_solver"
    adapter_cache_key = None if adapter is None else adapter.solver_cache_key()
    can_cache_solver = (
        cache_owner is not None
        and callback is None
        and progress_callback is None
        and failure_callback is None
    )
    solver = _cached_private_solver(
        cache_owner if can_cache_solver else None,
        cache_key=(
            "lbfgs",
            adapter_cache_key,
            norm,
            int(history_size),
            int(maxls),
            None if initial_step_size_value is None else float(initial_step_size_value),
            float(ftol_value),
            float(gtol_value),
            int(maxiter_limit_value),
            int(maxfun_limit_value),
            int(maxgrad_limit_value),
        ),
        builder=lambda: jax.jit(run_solver),
    )
    return solver(state_initial)


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
):
    return _minimize_lbfgs_private_impl(
        fun,
        x0,
        cache_owner=fun,
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
