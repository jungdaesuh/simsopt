"""On-device BFGS solver (lax.while_loop) and hybrid continuation state."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from ._common import (
    _as_jax_dtype,
    _dot,
    _eye,
    _einsum,
    _emit_iteration_callbacks,
    _norm,
    _require_private_optimizer_runtime,
    _scalar_value_and_grad,
)
from ._line_search import _line_search
from ._result_converters import _coerce_dense_hess_inv
from ._types import _BFGSResults


def _int_scalar(value):
    return _as_jax_dtype(value, jnp.int32)


def _bool_scalar(value):
    return _as_jax_dtype(value, jnp.bool_)


def _minimize_bfgs_private(
    fun,
    x0,
    *,
    maxiter=None,
    norm=np.inf,
    gtol=1e-5,
    line_search_maxiter=10,
    initial_state=None,
    callback=None,
    progress_callback=None,
):
    x0 = _require_private_optimizer_runtime(x0)
    if maxiter is None:
        maxiter = np.size(x0) * 200
    maxiter = int(maxiter)

    d = x0.shape[0]
    scalar_value_and_grad = _scalar_value_and_grad(fun)
    gtol_jax = _as_jax_dtype(gtol, x0.dtype)
    half = _as_jax_dtype(0.5, x0.dtype)
    wolfe_c1 = _as_jax_dtype(1e-4, x0.dtype)
    wolfe_c2 = _as_jax_dtype(0.9, x0.dtype)
    maxiter_jax = _as_jax_dtype(maxiter, jnp.int32)
    base_identity = _eye(d, x0.dtype)
    if initial_state is None:
        initial_H = base_identity
        f_0, g_0 = scalar_value_and_grad(x0)
        state = _BFGSResults(
            converged=_norm(g_0, ord=norm) < gtol_jax,
            failed=_bool_scalar(False),
            k=_int_scalar(0),
            nfev=_int_scalar(1),
            ngev=_int_scalar(1),
            nhev=_int_scalar(0),
            x_k=x0,
            f_k=f_0,
            g_k=g_0,
            H_k=initial_H,
            old_old_fval=f_0 + _norm(g_0) * half,
            status=_int_scalar(0),
            line_search_status=_int_scalar(0),
        )
    else:
        state = initial_state._replace(
            x_k=_as_jax_dtype(initial_state.x_k, x0.dtype),
            f_k=_as_jax_dtype(initial_state.f_k, x0.dtype),
            g_k=_as_jax_dtype(initial_state.g_k, x0.dtype),
            H_k=_as_jax_dtype(initial_state.H_k, x0.dtype),
            old_old_fval=_as_jax_dtype(initial_state.old_old_fval, x0.dtype),
        )

    def cond_fun(state):
        return (
            jnp.logical_not(state.converged)
            & jnp.logical_not(state.failed)
            & (state.k < maxiter_jax)
            & jnp.isfinite(state.f_k)
            & jnp.all(jnp.isfinite(state.g_k))
        )

    def body_fun(state):
        p_k = -_dot(state.H_k, state.g_k)
        line_search_results = _line_search(
            fun,
            state.x_k,
            p_k,
            old_fval=state.f_k,
            old_old_fval=state.old_old_fval,
            gfk=state.g_k,
            maxiter=line_search_maxiter,
        )
        next_nfev = state.nfev + line_search_results.nfev
        next_ngev = state.ngev + line_search_results.ngev
        s_k = line_search_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k = jnp.reciprocal(_dot(y_k, s_k))

        sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
        identity = _as_jax_dtype(base_identity, rho_k.dtype)
        w = identity - rho_k * sy_k
        H_kp1 = (
            _einsum("ij,jk,lk", w, state.H_k, w)
            + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :]
        )
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
        converged = _norm(g_kp1, ord=norm) < gtol_jax
        next_k = state.k + _int_scalar(1)
        dphi_0 = jnp.real(_dot(state.g_k, p_k))
        dphi_kp1 = jnp.real(_dot(g_kp1, p_k))
        strong_wolfe = (
            jnp.isfinite(f_kp1)
            & jnp.all(jnp.isfinite(g_kp1))
            & (f_kp1 <= state.f_k + wolfe_c1 * line_search_results.a_k * dphi_0)
            & (jnp.abs(dphi_kp1) <= -wolfe_c2 * dphi_0)
        )
        step_eps = jnp.sqrt(
            _as_jax_dtype(jnp.finfo(state.x_k.dtype).eps, state.x_k.dtype)
        )
        step_tol = step_eps * jnp.maximum(
            _as_jax_dtype(1.0, state.x_k.dtype),
            _norm(state.x_k),
        )
        stalled_step = (~converged) & (_norm(s_k) <= step_tol)
        nonfinite_step = (~jnp.isfinite(f_kp1)) | (~jnp.all(jnp.isfinite(g_kp1)))
        failure_line_search_status = jnp.where(
            line_search_results.failed,
            line_search_results.status,
            jnp.where(
                stalled_step | (~strong_wolfe),
                _as_jax_dtype(0, line_search_results.status.dtype),
                line_search_results.status,
            ),
        )

        def nonfinite_step_result(_):
            return state._replace(
                converged=_bool_scalar(False),
                failed=_bool_scalar(True),
                k=next_k,
                nfev=next_nfev,
                ngev=next_ngev,
                line_search_status=failure_line_search_status,
            )

        def failed_step(_):
            return state._replace(
                converged=_bool_scalar(False),
                failed=_bool_scalar(True),
                k=next_k,
                nfev=next_nfev,
                ngev=next_ngev,
                line_search_status=failure_line_search_status,
            )

        def accepted_step(_):
            _emit_iteration_callbacks(
                callback, progress_callback, x_kp1, next_k, f_kp1, g_kp1
            )
            return state._replace(
                converged=converged,
                nfev=next_nfev,
                ngev=next_ngev,
                k=next_k,
                x_k=x_kp1,
                f_k=f_kp1,
                g_k=g_kp1,
                H_k=H_kp1,
                old_old_fval=state.f_k,
                line_search_status=line_search_results.status,
            )

        return lax.cond(
            nonfinite_step,
            nonfinite_step_result,
            lambda _: lax.cond(
                line_search_results.failed | stalled_step | (~strong_wolfe),
                failed_step,
                accepted_step,
                operand=None,
            ),
            operand=None,
        )

    def run_solver(initial_state):
        state = lax.while_loop(cond_fun, body_fun, initial_state)
        f_final, g_final = scalar_value_and_grad(state.x_k)
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
                state.k == maxiter_jax,
                _int_scalar(1),
                jnp.where(
                    state.failed,
                    _int_scalar(2) + state.line_search_status,
                    _int_scalar(-1),
                ),
            ),
        )
        return state._replace(status=status)

    return jax.jit(run_solver)(state)


def _make_bfgs_continuation_state(result, *, gtol, norm):
    x_k = _as_jax_dtype(result.x, jnp.float64)
    f_k = _as_jax_dtype(result.fun, x_k.dtype)
    g_k = _as_jax_dtype(result.jac, x_k.dtype)
    H_k = _coerce_dense_hess_inv(
        getattr(result, "hess_inv", None), x_k.shape[0], x_k.dtype
    )

    dphi_0 = _dot(g_k, -_dot(H_k, g_k))
    H_k = jnp.where(dphi_0 < 0, H_k, _eye(H_k.shape[0], x_k.dtype))

    return _BFGSResults(
        converged=_bool_scalar(False),
        failed=_bool_scalar(False),
        k=_int_scalar(0),
        nfev=_int_scalar(int(getattr(result, "nfev", 0))),
        ngev=_int_scalar(int(getattr(result, "njev", getattr(result, "nfev", 0)))),
        nhev=_int_scalar(0),
        x_k=x_k,
        f_k=f_k,
        g_k=g_k,
        H_k=H_k,
        old_old_fval=f_k + _norm(g_k) * _as_jax_dtype(0.5, x_k.dtype),
        status=_int_scalar(0),
        line_search_status=_int_scalar(0),
    )
