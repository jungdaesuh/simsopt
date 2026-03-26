"""Explicit (host-loop) L-BFGS solver with value-and-gradient interface."""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from scipy.optimize import OptimizeResult

from ._common import _require_private_optimizer_runtime, _resolve_lbfgs_limits
from ._result_converters import _is_invalid_state, _status_message_lbfgs


def _evaluate_explicit_value_and_grad(fun, x):
    x_array = np.array(x, dtype=float, copy=True)
    value, grad = fun(x_array)
    value = float(np.asarray(value, dtype=float))
    grad = np.asarray(grad, dtype=float)
    if grad.shape != x_array.shape:
        raise ValueError(
            "Explicit value-and-gradient objectives must return a gradient "
            f"matching x.shape={x_array.shape}, got {grad.shape}."
        )
    return value, grad


def _explicit_line_search_result(
    *,
    failed,
    status,
    alpha,
    fun,
    grad,
    nfev,
    ngev,
):
    return {
        "failed": bool(failed),
        "status": int(status),
        "alpha": float(alpha),
        "fun": float(fun),
        "grad": np.asarray(grad, dtype=float),
        "nfev": int(nfev),
        "ngev": int(ngev),
    }


def _two_loop_recursion_explicit(grad, s_history, y_history, rho_history, gamma):
    q = -np.asarray(grad, dtype=float)
    alphas = []
    for s_k, y_k, rho_k in zip(
        reversed(s_history),
        reversed(y_history),
        reversed(rho_history),
    ):
        alpha_k = rho_k * float(np.dot(s_k, q))
        alphas.append(alpha_k)
        q = q - alpha_k * y_k

    r = float(gamma) * q
    for s_k, y_k, rho_k, alpha_k in zip(
        s_history,
        y_history,
        rho_history,
        reversed(alphas),
    ):
        beta_k = rho_k * float(np.dot(y_k, r))
        r = r + s_k * (alpha_k - beta_k)
    return r


def _line_search_explicit_value_and_grad(
    fun,
    xk,
    pk,
    *,
    old_fval,
    gfk,
    maxiter,
    c1=1e-4,
):
    slope0 = float(np.dot(gfk, pk))
    if not np.isfinite(slope0) or slope0 >= 0.0:
        return _explicit_line_search_result(
            failed=True,
            status=5,
            alpha=0.0,
            fun=old_fval,
            grad=gfk,
            nfev=0,
            ngev=0,
        )

    alpha = 1.0
    nfev = 0
    ngev = 0
    min_alpha = 1e-12

    for _ in range(int(maxiter)):
        trial_x = xk + alpha * pk
        trial_fun, trial_grad = _evaluate_explicit_value_and_grad(fun, trial_x)
        nfev += 1
        ngev += 1
        if (
            np.isfinite(trial_fun)
            and np.all(np.isfinite(trial_grad))
            and trial_fun <= old_fval + c1 * alpha * slope0
            and trial_fun < old_fval
        ):
            return _explicit_line_search_result(
                failed=False,
                status=0,
                alpha=alpha,
                fun=trial_fun,
                grad=trial_grad,
                nfev=nfev,
                ngev=ngev,
            )
        alpha *= 0.5
        if alpha < min_alpha:
            break

    return _explicit_line_search_result(
        failed=True,
        status=5,
        alpha=0.0,
        fun=old_fval,
        grad=gfk,
        nfev=nfev,
        ngev=ngev,
    )


def _minimize_lbfgs_explicit_value_and_grad(
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
):
    x0 = np.asarray(_require_private_optimizer_runtime(x0), dtype=float)
    d = len(x0)
    maxiter, maxfun, maxgrad = _resolve_lbfgs_limits(d, maxiter, maxfun, maxgrad)

    f_k, g_k = _evaluate_explicit_value_and_grad(fun, x0)
    x_k = x0.copy()
    nit = 0
    nfev = 1
    ngev = 1
    s_history = []
    y_history = []
    rho_history = []
    gamma = 1.0
    converged = np.linalg.norm(g_k, ord=norm) < gtol
    status = 0 if converged else None

    while (
        not converged
        and np.isfinite(f_k)
        and np.all(np.isfinite(g_k))
        and nit < maxiter
        and nfev < maxfun
        and ngev < maxgrad
    ):
        p_k = _two_loop_recursion_explicit(
            g_k, s_history, y_history, rho_history, gamma
        )
        if not np.isfinite(np.linalg.norm(p_k)) or float(np.dot(g_k, p_k)) >= 0.0:
            p_k = -g_k

        ls = _line_search_explicit_value_and_grad(
            fun,
            x_k,
            p_k,
            old_fval=f_k,
            gfk=g_k,
            maxiter=maxls,
        )
        nfev += ls["nfev"]
        ngev += ls["ngev"]
        if ls["failed"]:
            status = ls["status"]
            break

        x_next = x_k + ls["alpha"] * p_k
        f_next = float(ls["fun"])
        g_next = np.asarray(ls["grad"], dtype=float)
        s_k = x_next - x_k
        y_k = g_next - g_k
        ys = float(np.dot(y_k, s_k))
        yy = float(np.dot(y_k, y_k))

        x_k = x_next
        f_prev = f_k
        f_k = f_next
        g_k = g_next
        nit += 1

        if ys > 0.0 and yy > 0.0 and np.isfinite(ys) and np.isfinite(yy):
            if len(s_history) == maxcor:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_k.copy())
            y_history.append(y_k.copy())
            rho_history.append(1.0 / ys)
            gamma = ys / yy

        if callback is not None:
            callback(np.asarray(x_k, dtype=float))

        converged = np.linalg.norm(g_k, ord=norm) < gtol
        if converged:
            status = 0
            break
        if f_prev - f_k < ftol:
            status = 4
            break
        if ngev >= maxgrad:
            status = 3
            break
        if nfev >= maxfun:
            status = 2
            break

    if status is None:
        if converged:
            status = 0
        elif nit >= maxiter:
            status = 1
        elif not np.isfinite(f_k) or not np.all(np.isfinite(g_k)):
            status = 5
        elif ngev >= maxgrad:
            status = 3
        elif nfev >= maxfun:
            status = 2
        else:
            status = 5

    invalid_state = _is_invalid_state(f_k, g_k)
    return OptimizeResult(
        x=jnp.asarray(x_k),
        fun=float(f_k),
        jac=jnp.asarray(g_k),
        nit=int(nit),
        nfev=int(nfev),
        njev=int(ngev),
        success=bool(converged) and not invalid_state,
        status=int(status),
        message=_status_message_lbfgs(int(status), invalid_state),
        ls_status=int(status if int(status) == 5 else 0),
    )
