"""On-device BFGS solver (lax.while_loop) and hybrid continuation state."""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from jax import lax
from jax._src import api
from jax._src.numpy import linalg as jnp_linalg

from ._common import (
    _dot,
    _einsum,
    _emit_iteration_callbacks,
    _require_private_optimizer_runtime,
)
from ._line_search import _line_search
from ._result_converters import _coerce_dense_hess_inv
from ._types import _BFGSResults


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
    if initial_state is None:
        initial_H = jnp.eye(d, dtype=x0.dtype)
        f_0, g_0 = api.value_and_grad(fun)(x0)
        state = _BFGSResults(
            converged=jnp_linalg.norm(g_0, ord=norm) < gtol,
            failed=False,
            k=0,
            nfev=1,
            ngev=1,
            nhev=0,
            x_k=x0,
            f_k=f_0,
            g_k=g_0,
            H_k=initial_H,
            old_old_fval=f_0 + jnp_linalg.norm(g_0) / 2.0,
            status=0,
            line_search_status=0,
        )
    else:
        state = initial_state._replace(
            x_k=jnp.asarray(initial_state.x_k, dtype=x0.dtype),
            f_k=jnp.asarray(initial_state.f_k, dtype=x0.dtype),
            g_k=jnp.asarray(initial_state.g_k, dtype=x0.dtype),
            H_k=jnp.asarray(initial_state.H_k, dtype=x0.dtype),
            old_old_fval=jnp.asarray(initial_state.old_old_fval, dtype=x0.dtype),
        )

    def cond_fun(state):
        return (
            jnp.logical_not(state.converged)
            & jnp.logical_not(state.failed)
            & (state.k < maxiter)
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
        w = jnp.eye(d, dtype=rho_k.dtype) - rho_k * sy_k
        H_kp1 = (
            _einsum("ij,jk,lk", w, state.H_k, w)
            + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :]
        )
        H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
        converged = jnp_linalg.norm(g_kp1, ord=norm) < gtol
        next_k = state.k + 1

        def failed_step(_):
            return state._replace(
                converged=False,
                failed=True,
                nfev=next_nfev,
                ngev=next_ngev,
                line_search_status=line_search_results.status,
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
            line_search_results.failed,
            failed_step,
            accepted_step,
            operand=None,
        )

    state = lax.while_loop(cond_fun, body_fun, state)
    f_final, g_final = api.value_and_grad(fun)(state.x_k)
    converged_final = jnp_linalg.norm(g_final, ord=norm) < gtol
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
            state.k == maxiter,
            1,
            jnp.where(state.failed, 2 + state.line_search_status, -1),
        ),
    )
    return state._replace(status=status)


def _make_bfgs_continuation_state(result, *, gtol, norm):
    x_k = jnp.asarray(result.x, dtype=jnp.float64)
    f_k = jnp.asarray(result.fun, dtype=x_k.dtype)
    g_k = jnp.asarray(result.jac, dtype=x_k.dtype)
    H_k = _coerce_dense_hess_inv(
        getattr(result, "hess_inv", None), x_k.shape[0], x_k.dtype
    )

    dphi_0 = _dot(g_k, -_dot(H_k, g_k))
    H_k = jnp.where(dphi_0 < 0, H_k, jnp.eye(H_k.shape[0], dtype=x_k.dtype))

    return _BFGSResults(
        converged=False,
        failed=False,
        k=0,
        nfev=int(getattr(result, "nfev", 0)),
        ngev=int(getattr(result, "njev", getattr(result, "nfev", 0))),
        nhev=0,
        x_k=x_k,
        f_k=f_k,
        g_k=g_k,
        H_k=H_k,
        old_old_fval=f_k + jnp_linalg.norm(g_k) / 2.0,
        status=0,
        line_search_status=0,
    )
