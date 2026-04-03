"""On-device L-BFGS solver (lax.while_loop)."""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from jax import lax, value_and_grad

from ._common import (
    _dot,
    _emit_iteration_callbacks,
    _norm,
    _require_private_optimizer_runtime,
    _resolve_lbfgs_limits,
)
from ._line_search import _line_search_value_and_grad
from ._types import _LBFGSResults


def _update_history_vectors(history, new):
    return jnp.roll(history, -1, axis=0).at[-1, :].set(new)


def _update_history_scalars(history, new):
    return jnp.roll(history, -1, axis=0).at[-1].set(new)


def _two_loop_recursion(state):
    dtype = state.rho_history.dtype
    his_size = len(state.rho_history)
    curr_size = jnp.where(state.k < his_size, state.k, his_size)
    q = -jnp.conj(state.g_k)
    a_his = jnp.zeros_like(state.rho_history)

    def body_fun1(j, carry):
        i = his_size - 1 - j
        _q, _a_his = carry
        a_i = state.rho_history[i] * _dot(jnp.conj(state.s_history[i]), _q).real.astype(
            dtype
        )
        _a_his = _a_his.at[i].set(a_i)
        _q = _q - a_i * jnp.conj(state.y_history[i])
        return _q, _a_his

    q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))
    q = state.gamma * q

    def body_fun2(j, _q):
        i = his_size - curr_size + j
        b_i = state.rho_history[i] * _dot(state.y_history[i], _q).real.astype(dtype)
        return _q + (a_his[i] - b_i) * state.s_history[i]

    return lax.fori_loop(0, curr_size, body_fun2, q)


def _coerce_value_and_grad_result(fun, x):
    value, grad = fun(x)
    value = jnp.asarray(value, dtype=x.dtype)
    grad = jnp.asarray(grad, dtype=x.dtype)
    if grad.shape != x.shape:
        raise ValueError(
            "On-device explicit value-and-gradient objectives must return a "
            f"gradient matching x.shape={x.shape}, got {grad.shape}."
        )
    return value, grad


def _minimize_lbfgs_private_impl(
    value_and_grad_fun,
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
    callback=None,
    progress_callback=None,
):
    x0 = _require_private_optimizer_runtime(x0)
    d = len(x0)
    dtype = x0.dtype
    maxiter, maxfun, maxgrad = _resolve_lbfgs_limits(d, maxiter, maxfun, maxgrad)

    f_0, g_0 = _coerce_value_and_grad_result(value_and_grad_fun, x0)
    state_initial = _LBFGSResults(
        converged=_norm(g_0, ord=norm) < gtol,
        failed=jnp.array(False, dtype=bool),
        k=0,
        nfev=1,
        ngev=1,
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        s_history=jnp.zeros((maxcor, d), dtype=dtype),
        y_history=jnp.zeros((maxcor, d), dtype=dtype),
        rho_history=jnp.zeros((maxcor,), dtype=dtype),
        gamma=1.0,
        status=0,
        ls_status=0,
    )
    initial_status = jnp.array(0)
    initial_status = jnp.where(state_initial.ngev >= maxgrad, 3, initial_status)
    initial_status = jnp.where(state_initial.nfev >= maxfun, 2, initial_status)
    initial_status = jnp.where(state_initial.k >= maxiter, 1, initial_status)
    state_initial = state_initial._replace(
        failed=(initial_status > 0) & (~state_initial.converged),
        status=jnp.where(state_initial.converged, 0, initial_status),
    )

    def cond_fun(state):
        return (
            (~state.converged)
            & (~state.failed)
            & jnp.isfinite(state.f_k)
            & jnp.all(jnp.isfinite(state.g_k))
        )

    def body_fun(state):
        p_k = _two_loop_recursion(state)
        ls_results = _line_search_value_and_grad(
            fun=value_and_grad_fun,
            xk=state.x_k,
            pk=p_k,
            old_fval=state.f_k,
            gfk=state.g_k,
            maxiter=maxls,
        )

        next_nfev = state.nfev + ls_results.nfev
        next_ngev = state.ngev + ls_results.ngev
        s_k = jnp.asarray(ls_results.a_k).astype(p_k.dtype) * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        rho_k = jnp.reciprocal(rho_k_inv).astype(y_k.dtype)
        gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))
        next_k = state.k + 1

        def failed_step(_):
            return state._replace(
                converged=False,
                failed=True,
                nfev=next_nfev,
                ngev=next_ngev,
                status=jnp.array(5),
                ls_status=ls_results.status,
            )

        def accepted_step(_):
            status = jnp.array(0)
            status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
            status = jnp.where(next_ngev >= maxgrad, 3, status)
            status = jnp.where(next_nfev >= maxfun, 2, status)
            status = jnp.where(next_k >= maxiter, 1, status)
            converged = _norm(g_kp1, ord=norm) < gtol
            _emit_iteration_callbacks(
                callback, progress_callback, x_kp1, next_k, f_kp1, g_kp1
            )
            return state._replace(
                converged=converged,
                failed=(status > 0) & (~converged),
                k=next_k,
                nfev=next_nfev,
                ngev=next_ngev,
                x_k=x_kp1.astype(state.x_k.dtype),
                f_k=f_kp1.astype(state.f_k.dtype),
                g_k=g_kp1.astype(state.g_k.dtype),
                s_history=_update_history_vectors(state.s_history, s_k),
                y_history=_update_history_vectors(state.y_history, y_k),
                rho_history=_update_history_scalars(state.rho_history, rho_k),
                gamma=gamma.astype(state.g_k.dtype),
                status=jnp.where(converged, 0, status),
                ls_status=ls_results.status,
            )

        return lax.cond(ls_results.failed, failed_step, accepted_step, operand=None)

    state = lax.while_loop(cond_fun, body_fun, state_initial)
    f_final, g_final = _coerce_value_and_grad_result(value_and_grad_fun, state.x_k)
    converged_final = _norm(g_final, ord=norm) < gtol
    state = state._replace(
        converged=converged_final,
        nfev=state.nfev + 1,
        ngev=state.ngev + 1,
        f_k=jnp.asarray(f_final, dtype=state.f_k.dtype),
        g_k=jnp.asarray(g_final, dtype=state.g_k.dtype),
    )
    status = jnp.where(
        state.converged,
        0,
        jnp.where(
            state.k >= maxiter,
            1,
            jnp.where(
                state.nfev >= maxfun,
                2,
                jnp.where(
                    state.ngev >= maxgrad, 3, jnp.where(state.failed, 5, state.status)
                ),
            ),
        ),
    )
    return state._replace(status=status)


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
    callback=None,
    progress_callback=None,
):
    return _minimize_lbfgs_private_impl(
        value_and_grad(fun),
        x0,
        maxiter=maxiter,
        norm=norm,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxgrad=maxgrad,
        maxls=maxls,
        callback=callback,
        progress_callback=progress_callback,
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
    callback=None,
    progress_callback=None,
):
    return _minimize_lbfgs_private_impl(
        fun,
        x0,
        maxiter=maxiter,
        norm=norm,
        maxcor=maxcor,
        ftol=ftol,
        gtol=gtol,
        maxfun=maxfun,
        maxgrad=maxgrad,
        maxls=maxls,
        callback=callback,
        progress_callback=progress_callback,
    )
