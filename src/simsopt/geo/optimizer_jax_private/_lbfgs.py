"""Host-dispatched L-BFGS over cached JAX value-and-gradient kernels."""

from __future__ import annotations

import json
import os
from typing import NamedTuple

import numpy as np

import jax

from ..optimizer_jax import _prepare_optimizer_callable_inputs
from ._common import (
    _as_jax_dtype,
    _bool_scalar,
    _cached_private_solver,
    _int_scalar,
    _require_private_optimizer_runtime,
    _resolve_lbfgs_limits,
    _scalar_value_and_grad,
    _STRUCTURED_SOLVER_CACHE_TOKEN_ATTR,
)
from ._types import (
    LBFGS_STATUS_NONFINITE,
    _LBFGSInvalidStepLog,
    _LBFGSResults,
    _LineSearchResults,
)


_LBFGS_DEBUG_ENABLED = os.environ.get("SIMSOPT_LBFGS_DEBUG", "").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_LBFGS_STATE_DUMP_ENABLED = os.environ.get(
    "SIMSOPT_LBFGS_STATE_DUMP", ""
).lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}

# Cap the on-device rejected-step ring buffer so capacity does not scale with
# maxiter. The retry policy only reads the most recent event, so a small bound
# is sufficient and keeps the buffer size independent of problem dimension.
_INVALID_STEP_LOG_MAX_CAPACITY = 256


class _HostZoomState(NamedTuple):
    done: bool
    failed: bool
    j: int
    a_lo: float
    phi_lo: float
    dphi_lo: float
    g_lo: np.ndarray
    a_hi: float
    phi_hi: float
    dphi_hi: float
    g_hi: np.ndarray
    has_rec: bool
    a_rec: float
    phi_rec: float
    dphi_rec: float
    g_rec: np.ndarray
    a_star: float
    phi_star: float
    dphi_star: float
    g_star: np.ndarray
    best_a: float
    best_phi: float
    best_dphi: float
    best_g: np.ndarray
    nfev: int
    ngev: int


class _HostLineSearchState(NamedTuple):
    done: bool
    failed: bool
    i: int
    a_i2: float
    phi_i2: float
    dphi_i2: float
    g_i2: np.ndarray
    a_i1: float
    phi_i1: float
    dphi_i1: float
    g_i1: np.ndarray
    best_a: float
    best_phi: float
    best_dphi: float
    best_g: np.ndarray
    nfev: int
    ngev: int
    a_star: float
    phi_star: float
    dphi_star: float
    g_star: np.ndarray


class _HostInvalidStepEvent(NamedTuple):
    iteration: int
    step_scale: float
    line_search_failed: bool
    nonfinite_step: bool
    stalled_step: bool
    valid_curvature: bool
    trial_converged: bool
    ls_status: int


class _HostLBFGSState(NamedTuple):
    converged: bool
    failed: bool
    k: int
    nfev: int
    ngev: int
    x_k: np.ndarray
    f_k: float
    g_k: np.ndarray
    old_old_fval: float
    s_history: np.ndarray
    y_history: np.ndarray
    rho_history: np.ndarray
    gamma: float
    history_count: int
    status: int
    ls_status: int
    invalid_step_events: tuple[_HostInvalidStepEvent, ...]


def _host_norm(x, *, ord=None):
    abs_x = np.abs(np.asarray(x))
    if ord in (None, 2):
        return np.sqrt(np.sum(abs_x * abs_x))
    if ord == np.inf:
        return np.max(abs_x)
    return np.linalg.norm(x, ord=ord)


def _as_host_array(value, *, dtype):
    return np.asarray(jax.device_get(value), dtype=np.dtype(dtype))


def _as_host_scalar(value, *, dtype):
    return _as_host_array(value, dtype=dtype).reshape(()).item()


def _as_host_bool(value):
    return bool(_as_host_scalar(value, dtype=np.bool_))


def _as_host_int(value):
    return int(_as_host_scalar(value, dtype=np.int32))


def _as_host_float(value, *, dtype):
    return float(_as_host_scalar(value, dtype=dtype))


def _as_device_array(value, *, dtype):
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _as_device_scalar(value, *, dtype):
    return _as_device_array(
        np.asarray(value, dtype=np.dtype(dtype)).reshape(()),
        dtype=dtype,
    )


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
    """Emit runtime diagnostics when SIMSOPT_LBFGS_DEBUG is enabled.

    L-BFGS control flow is host-dispatched, so debug output stays outside
    ``jax.debug.callback`` and does not add runtime callback effects.
    """
    if not _LBFGS_DEBUG_ENABLED:
        return

    grad_inf = _host_norm(grad, ord=np.inf)
    if step_scale is None:
        step_scale = 0.0
    if ls_failed is None:
        ls_failed = False
    if ls_status is None:
        ls_status = 0
    if nonfinite_step is None:
        nonfinite_step = False
    if stalled_step is None:
        stalled_step = False
    if valid_curvature is None:
        valid_curvature = True
    if converged is None:
        converged = False

    print(
        "[lbfgs-debug] "
        f"stage={stage} "
        f"iter={int(iteration)} "
        f"f={float(fun_value):.16e} "
        f"grad_inf={float(grad_inf):.16e} "
        f"alpha={float(step_scale):.16e} "
        f"ls_failed={bool(ls_failed)} "
        f"ls_status={int(ls_status)} "
        f"nonfinite_step={bool(nonfinite_step)} "
        f"stalled_step={bool(stalled_step)} "
        f"valid_curvature={bool(valid_curvature)} "
        f"converged={bool(converged)}",
        flush=True,
    )


def _emit_lbfgs_state_dump(state_initial):
    """Dump the structured solver input leaves before entering the host solve."""
    if not _LBFGS_STATE_DUMP_ENABLED:
        return

    for field_name, leaf in state_initial._asdict().items():
        shape = getattr(leaf, "shape", ())
        sharding = getattr(leaf, "sharding", None)
        print(
            "[lbfgs-state-dump] "
            + json.dumps(
                {
                    "field": field_name,
                    "python_type": type(leaf).__name__,
                    "dtype": None
                    if getattr(leaf, "dtype", None) is None
                    else str(leaf.dtype),
                    "shape": [int(dim) for dim in shape],
                    "sharding": None if sharding is None else str(sharding),
                },
                sort_keys=True,
            ),
            flush=True,
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


def _host_iteration_budget(d, maxiter_limit, maxfun_limit, maxgrad_limit):
    """Bound host-dispatched step calls from the resolved optimizer limits."""
    int32_max = np.iinfo(np.int32).max
    finite_limits = [
        int(limit)
        for limit in (maxiter_limit, maxfun_limit, maxgrad_limit)
        if int(limit) < int32_max
    ]
    if finite_limits:
        return max(0, min(finite_limits))
    return max(0, int(d) * 200)


def _host_limit_status(k, nfev, ngev, *, maxiter_limit, maxfun_limit, maxgrad_limit):
    status = 0
    if int(ngev) >= int(maxgrad_limit):
        status = 3
    if int(nfev) >= int(maxfun_limit):
        status = 2
    if int(k) >= int(maxiter_limit):
        status = 1
    return status


def _resolve_lbfgs_history_size(maxcor, *, d, maxiter_limit_value):
    """Cap history to the reachable correction budget for this solve.

    More than ``d`` correction pairs are redundant in an ``d``-dimensional
    problem, and more than ``maxiter`` pairs are unreachable during this solve.
    Clipping here shrinks the traced solver state without changing the set of
    curvature updates that the run can actually observe.
    """
    return max(1, min(int(maxcor), int(d), int(maxiter_limit_value)))


def _relative_objective_reduction_host(f_k, f_kp1, *, dtype):
    dtype = np.dtype(dtype)
    denominator = max(abs(float(f_k)), abs(float(f_kp1)), float(dtype.type(1.0)))
    return (float(f_k) - float(f_kp1)) / denominator


def _host_cubicmin(a, fa, fpa, b, fb, c, fc):
    dtype = np.result_type(a, fa, fpa, b, fb, c, fc)
    a = dtype.type(a)
    fa = dtype.type(fa)
    fpa = dtype.type(fpa)
    b = dtype.type(b)
    fb = dtype.type(fb)
    c = dtype.type(c)
    fc = dtype.type(fc)
    db = b - a
    dc = c - a
    denom = (db * dc) * (db * dc) * (db - dc)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = np.asarray(
            (
                (dc * dc, -(db * db)),
                (-(dc * dc * dc), db * db * db),
            ),
            dtype=dtype,
        )
        d2 = np.asarray((fb - fa - fpa * db, fc - fa - fpa * dc), dtype=dtype)
        a_coeff, b_coeff = np.dot(d1, d2) / denom
        radical = b_coeff * b_coeff - dtype.type(3.0) * a_coeff * fpa
        xmin = a + (-b_coeff + np.sqrt(radical)) / (dtype.type(3.0) * a_coeff)
    return xmin


def _host_quadmin(a, fa, fpa, b, fb):
    dtype = np.result_type(a, fa, fpa, b, fb)
    a = dtype.type(a)
    fa = dtype.type(fa)
    fpa = dtype.type(fpa)
    b = dtype.type(b)
    fb = dtype.type(fb)
    db = b - a
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b_coeff = (fb - fa - fpa * db) / (db * db)
        xmin = a - fpa / (dtype.type(2.0) * b_coeff)
    return xmin


def _line_search_sample_valid_host(phi, dphi, grad):
    return (
        np.isfinite(phi)
        and np.isfinite(dphi)
        and np.all(np.isfinite(np.asarray(grad)))
    )


def _cache_zoom_sample_host(state, *, alpha, phi, dphi, grad):
    return state._replace(
        has_rec=True,
        a_rec=float(alpha),
        phi_rec=float(phi),
        dphi_rec=float(dphi),
        g_rec=np.asarray(grad, dtype=state.g_rec.dtype),
    )


def _zoom_host(
    restricted_func_and_grad,
    wolfe_one,
    wolfe_two,
    phi_0,
    a_lo,
    phi_lo,
    dphi_lo,
    g_lo,
    a_hi,
    phi_hi,
    dphi_hi,
    g_hi,
    has_rec,
    a_rec,
    phi_rec,
    dphi_rec,
    g_rec,
    maxiter,
    pass_through,
):
    dtype = np.asarray(a_lo).dtype
    half = dtype.type(0.5)
    delta1 = dtype.type(0.2)
    delta2 = dtype.type(0.1)
    max_zoom_iter = int(maxiter)
    lo_valid = _line_search_sample_valid_host(phi_lo, dphi_lo, g_lo)
    hi_valid = _line_search_sample_valid_host(phi_hi, dphi_hi, g_hi)
    lo_is_better = lo_valid and ((not hi_valid) or phi_lo <= phi_hi)
    state = _HostZoomState(
        done=False,
        failed=bool(pass_through),
        j=0,
        a_lo=float(a_lo),
        phi_lo=float(phi_lo),
        dphi_lo=float(dphi_lo),
        g_lo=np.asarray(g_lo, dtype=dtype),
        a_hi=float(a_hi),
        phi_hi=float(phi_hi),
        dphi_hi=float(dphi_hi),
        g_hi=np.asarray(g_hi, dtype=dtype),
        has_rec=bool(has_rec),
        a_rec=float(a_rec),
        phi_rec=float(phi_rec),
        dphi_rec=float(dphi_rec),
        g_rec=np.asarray(g_rec, dtype=dtype),
        a_star=1.0,
        phi_star=float(phi_lo),
        dphi_star=float(dphi_lo),
        g_star=np.asarray(g_lo, dtype=dtype),
        best_a=float(a_lo if lo_is_better else a_hi),
        best_phi=float(phi_lo if lo_is_better else phi_hi),
        best_dphi=float(dphi_lo if lo_is_better else dphi_hi),
        best_g=np.asarray(g_lo if lo_is_better else g_hi, dtype=dtype),
        nfev=0,
        ngev=0,
    )

    while (not state.done) and (not state.failed):
        if state.j >= max_zoom_iter:
            state = state._replace(failed=True)
            break

        dalpha = abs(state.a_hi - state.a_lo)
        threshold = dtype.type(1e-5 if np.finfo(dtype).bits < 64 else 1e-10)
        if dalpha <= threshold:
            state = state._replace(failed=True)
            break

        a = min(state.a_hi, state.a_lo)
        b = max(state.a_hi, state.a_lo)
        cchk = float(delta1 * dalpha)
        qchk = float(delta2 * dalpha)
        a_j_cubic = _host_cubicmin(
            state.a_lo,
            state.phi_lo,
            state.dphi_lo,
            state.a_hi,
            state.phi_hi,
            state.a_rec,
            state.phi_rec,
        )
        use_cubic = (
            state.has_rec
            and np.isfinite(a_j_cubic)
            and (a_j_cubic > a + cchk)
            and (a_j_cubic < b - cchk)
        )
        a_j_quad = _host_quadmin(
            state.a_lo,
            state.phi_lo,
            state.dphi_lo,
            state.a_hi,
            state.phi_hi,
        )
        use_quad = (
            (not use_cubic)
            and np.isfinite(a_j_quad)
            and (a_j_quad > a + qchk)
            and (a_j_quad < b - qchk)
        )
        if use_cubic:
            a_j = float(a_j_cubic)
        elif use_quad:
            a_j = float(a_j_quad)
        else:
            a_j = float((state.a_lo + state.a_hi) * half)

        if state.has_rec and a_j == state.a_rec:
            phi_j = state.phi_rec
            dphi_j = state.dphi_rec
            g_j = state.g_rec
            sample_eval_count = 0
        else:
            phi_j, dphi_j, g_j = restricted_func_and_grad(a_j)
            phi_j = float(phi_j)
            dphi_j = float(dphi_j)
            g_j = np.asarray(g_j, dtype=dtype)
            sample_eval_count = 1

        state = state._replace(
            nfev=state.nfev + sample_eval_count,
            ngev=state.ngev + sample_eval_count,
        )
        sample_valid = _line_search_sample_valid_host(phi_j, dphi_j, g_j)
        if sample_valid and phi_j < state.best_phi:
            state = state._replace(
                best_a=float(a_j),
                best_phi=float(phi_j),
                best_dphi=float(dphi_j),
                best_g=g_j,
            )

        hi_to_j = (not sample_valid) or wolfe_one(a_j, phi_j) or (
            phi_j >= state.phi_lo
        )
        star_to_j = sample_valid and wolfe_two(dphi_j) and not hi_to_j
        hi_to_lo = (
            sample_valid
            and (dphi_j * (state.a_hi - state.a_lo) >= 0.0)
            and not hi_to_j
            and not star_to_j
        )
        lo_to_j = sample_valid and (not hi_to_j) and (not star_to_j)
        previous_a_lo = state.a_lo
        previous_phi_lo = state.phi_lo
        previous_dphi_lo = state.dphi_lo
        previous_g_lo = state.g_lo
        previous_a_hi = state.a_hi
        previous_phi_hi = state.phi_hi
        previous_dphi_hi = state.dphi_hi
        previous_g_hi = state.g_hi

        if hi_to_j:
            state = state._replace(
                a_hi=float(a_j),
                phi_hi=float(phi_j),
                dphi_hi=float(dphi_j),
                g_hi=g_j,
            )
            state = _cache_zoom_sample_host(
                state,
                alpha=previous_a_hi,
                phi=previous_phi_hi,
                dphi=previous_dphi_hi,
                grad=previous_g_hi,
            )
        if star_to_j:
            state = state._replace(
                done=True,
                a_star=float(a_j),
                phi_star=float(phi_j),
                dphi_star=float(dphi_j),
                g_star=g_j,
            )
        if hi_to_lo:
            state = state._replace(
                a_hi=float(state.a_lo),
                phi_hi=float(state.phi_lo),
                dphi_hi=float(state.dphi_lo),
                g_hi=state.g_lo,
            )
            state = _cache_zoom_sample_host(
                state,
                alpha=previous_a_hi,
                phi=previous_phi_hi,
                dphi=previous_dphi_hi,
                grad=previous_g_hi,
            )
        if lo_to_j:
            state = state._replace(
                a_lo=float(a_j),
                phi_lo=float(phi_j),
                dphi_lo=float(dphi_j),
                g_lo=g_j,
            )
            if not hi_to_lo:
                state = _cache_zoom_sample_host(
                    state,
                    alpha=previous_a_lo,
                    phi=previous_phi_lo,
                    dphi=previous_dphi_lo,
                    grad=previous_g_lo,
                )

        state = state._replace(j=state.j + 1)
        if state.j >= max_zoom_iter and not state.done:
            state = state._replace(failed=True)

    best_is_acceptable = (
        _line_search_sample_valid_host(state.best_phi, state.best_dphi, state.best_g)
        and (not wolfe_one(state.best_a, state.best_phi))
        and (state.best_phi < phi_0)
    )
    if state.failed and best_is_acceptable:
        state = state._replace(
            failed=False,
            done=True,
            a_star=state.best_a,
            phi_star=state.best_phi,
            dphi_star=state.best_dphi,
            g_star=state.best_g,
        )
    return state


def _apply_zoom_branch_result_host(state, zoom, *, wolfe_one):
    state = state._replace(
        nfev=state.nfev + zoom.nfev,
        ngev=state.ngev + zoom.ngev,
    )
    improves_best = (
        _line_search_sample_valid_host(zoom.best_phi, zoom.best_dphi, zoom.best_g)
        and (not wolfe_one(zoom.best_a, zoom.best_phi))
        and (zoom.best_phi < state.best_phi)
    )
    if improves_best:
        state = state._replace(
            best_a=zoom.best_a,
            best_phi=zoom.best_phi,
            best_dphi=zoom.best_dphi,
            best_g=zoom.best_g,
        )
    return state._replace(
        done=True,
        failed=zoom.failed or state.failed,
        a_star=zoom.a_star,
        phi_star=zoom.phi_star,
        dphi_star=zoom.dphi_star,
        g_star=zoom.g_star,
    )


def _line_search_from_restricted_func_and_grad_host(
    restricted_func_and_grad,
    *,
    pk,
    old_fval,
    gfk,
    old_old_fval=None,
    initial_step_size=None,
    c1=1e-4,
    c2=0.9,
    maxiter=20,
):
    dtype = np.asarray(pk).dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)
    two = dtype.type(2.0)
    one_point_01 = dtype.type(1.01)
    c1 = dtype.type(c1)
    c2 = dtype.type(c2)
    maxiter_value = int(maxiter)
    pk = np.asarray(pk, dtype=dtype)
    gfk = np.asarray(gfk, dtype=dtype)
    phi_0 = float(dtype.type(old_fval))
    dphi_0 = float(np.dot(gfk, pk))

    if not np.isfinite(dphi_0) or dphi_0 >= 0.0:
        return _LineSearchResults(
            failed=True,
            nit=0,
            nfev=0,
            ngev=0,
            k=1,
            a_k=float(zero),
            f_k=float(phi_0),
            g_k=gfk,
            status=1,
        )

    if initial_step_size is None:
        use_initial_step_override = False
        initial_step_value = zero
    else:
        initial_step_value = dtype.type(initial_step_size)
        use_initial_step_override = np.isfinite(initial_step_value) and (
            initial_step_value > zero
        )

    if old_old_fval is not None:
        candidate_start_value = (
            one_point_01 * two * (phi_0 - dtype.type(old_old_fval)) / dphi_0
        )
        if candidate_start_value <= zero:
            candidate_start_value = one
        start_value = min(float(one), float(candidate_start_value))
    else:
        start_value = float(one)
    if use_initial_step_override:
        start_value = float(initial_step_value)

    def wolfe_one(a_i, phi_i):
        return phi_i > phi_0 + float(c1) * a_i * dphi_0

    def wolfe_two(dphi_i):
        return abs(dphi_i) <= -float(c2) * dphi_0

    state = _HostLineSearchState(
        done=False,
        failed=False,
        i=1,
        a_i2=float(zero),
        phi_i2=float(phi_0),
        dphi_i2=float(dphi_0),
        g_i2=gfk,
        a_i1=float(zero),
        phi_i1=float(phi_0),
        dphi_i1=float(dphi_0),
        g_i1=gfk,
        best_a=float(zero),
        best_phi=float(phi_0),
        best_dphi=float(dphi_0),
        best_g=gfk,
        nfev=0,
        ngev=0,
        a_star=float(zero),
        phi_star=float(phi_0),
        dphi_star=float(dphi_0),
        g_star=gfk,
    )

    while (not state.done) and (state.i <= maxiter_value) and (not state.failed):
        a_i = start_value if state.i == 1 else state.a_i1 * float(two)
        phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
        phi_i = float(phi_i)
        dphi_i = float(dphi_i)
        g_i = np.asarray(g_i, dtype=dtype)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)

        sample_valid = _line_search_sample_valid_host(phi_i, dphi_i, g_i)
        improves_best_i = (
            sample_valid
            and (not wolfe_one(a_i, phi_i))
            and (phi_i < state.best_phi)
        )
        if improves_best_i:
            state = state._replace(
                best_a=float(a_i),
                best_phi=float(phi_i),
                best_dphi=float(dphi_i),
                best_g=g_i,
            )

        star_to_zoom1 = (not sample_valid) or wolfe_one(a_i, phi_i) or (
            (phi_i >= state.phi_i1) and (state.i > 1)
        )
        star_to_i = sample_valid and wolfe_two(dphi_i) and not star_to_zoom1
        star_to_zoom2 = (
            sample_valid and (dphi_i >= 0.0) and not star_to_zoom1 and not star_to_i
        )
        remaining_zoom_budget = max(0, maxiter_value - state.nfev)
        zoom_budget_exhausted = remaining_zoom_budget <= 0

        if star_to_zoom1:
            zoom = _zoom_host(
                restricted_func_and_grad,
                wolfe_one,
                wolfe_two,
                phi_0,
                state.a_i1,
                state.phi_i1,
                state.dphi_i1,
                state.g_i1,
                a_i,
                phi_i,
                dphi_i,
                g_i,
                state.i > 1,
                state.a_i2,
                state.phi_i2,
                state.dphi_i2,
                state.g_i2,
                remaining_zoom_budget,
                zoom_budget_exhausted,
            )
            state = _apply_zoom_branch_result_host(state, zoom, wolfe_one=wolfe_one)
        elif star_to_i:
            state = state._replace(
                done=True,
                a_star=float(a_i),
                phi_star=float(phi_i),
                dphi_star=float(dphi_i),
                g_star=g_i,
            )
        elif star_to_zoom2:
            zoom = _zoom_host(
                restricted_func_and_grad,
                wolfe_one,
                wolfe_two,
                phi_0,
                a_i,
                phi_i,
                dphi_i,
                g_i,
                state.a_i1,
                state.phi_i1,
                state.dphi_i1,
                state.g_i1,
                state.i > 1,
                state.a_i2,
                state.phi_i2,
                state.dphi_i2,
                state.g_i2,
                remaining_zoom_budget,
                zoom_budget_exhausted,
            )
            state = _apply_zoom_branch_result_host(state, zoom, wolfe_one=wolfe_one)

        state = state._replace(
            i=state.i + 1,
            a_i2=state.a_i1,
            phi_i2=state.phi_i1,
            dphi_i2=state.dphi_i1,
            g_i2=state.g_i1,
            a_i1=float(a_i),
            phi_i1=float(phi_i),
            dphi_i1=float(dphi_i),
            g_i1=g_i,
        )

    best_is_acceptable = _line_search_sample_valid_host(
        state.best_phi,
        state.best_dphi,
        state.best_g,
    ) and (state.best_phi < phi_0)
    if (state.failed or (not state.done)) and best_is_acceptable:
        state = state._replace(
            failed=False,
            done=True,
            a_star=state.best_a,
            phi_star=state.best_phi,
            dphi_star=state.best_dphi,
            g_star=state.best_g,
        )

    status = 1 if state.failed else (3 if state.i > maxiter_value else 0)
    alpha_k = dtype.type(state.a_star)
    if np.finfo(dtype).bits != 64 and abs(alpha_k) < dtype.type(1e-8):
        alpha_k = np.sign(alpha_k) * dtype.type(1e-8)
    return _LineSearchResults(
        failed=state.failed or (not state.done),
        nit=state.i - 1,
        nfev=state.nfev,
        ngev=state.ngev,
        k=state.i,
        a_k=float(alpha_k),
        f_k=float(state.phi_star),
        g_k=np.asarray(state.g_star, dtype=dtype),
        status=status,
    )


def _line_search_value_and_grad_host(
    fun,
    xk,
    pk,
    old_fval,
    gfk,
    old_old_fval=None,
    initial_step_size=None,
    c1=1e-4,
    c2=0.9,
    maxiter=20,
):
    xk = np.asarray(xk)
    pk = np.asarray(pk, dtype=xk.dtype)
    old_fval = np.asarray(old_fval, dtype=xk.dtype).reshape(()).item()
    gfk = np.asarray(gfk, dtype=xk.dtype)

    def restricted_func_and_grad(t):
        phi, grad = fun(xk + np.asarray(t, dtype=xk.dtype) * pk)
        grad = np.asarray(grad, dtype=xk.dtype)
        dphi = float(np.dot(grad, pk))
        return float(phi), dphi, grad

    return _line_search_from_restricted_func_and_grad_host(
        restricted_func_and_grad,
        pk=pk,
        old_fval=old_fval,
        old_old_fval=old_old_fval,
        gfk=gfk,
        initial_step_size=initial_step_size,
        c1=c1,
        c2=c2,
        maxiter=maxiter,
    )


def _coerce_line_search_results_host(results, *, dtype):
    return _LineSearchResults(
        failed=_as_host_bool(results.failed),
        nit=_as_host_int(results.nit),
        nfev=_as_host_int(results.nfev),
        ngev=_as_host_int(results.ngev),
        k=_as_host_int(results.k),
        a_k=_as_host_float(results.a_k, dtype=dtype),
        f_k=_as_host_float(results.f_k, dtype=dtype),
        g_k=_as_host_array(results.g_k, dtype=dtype),
        status=_as_host_int(results.status),
    )


def _two_loop_recursion_host(g_k, gamma, s_history, y_history, rho_history, history_count):
    history_count = int(history_count)
    history_size = int(rho_history.shape[0])
    curr_size = min(history_count, history_size)
    q = -np.asarray(g_k)
    a_his = np.zeros_like(rho_history)

    for offset in range(curr_size - 1, -1, -1):
        i = (history_count - curr_size + offset) % history_size
        rho_i = rho_history[i]
        s_i = s_history[i]
        y_i = y_history[i]
        a_i = rho_i * np.dot(s_i, q)
        a_his[i] = a_i
        q = q - a_i * y_i
    q = gamma * q

    for offset in range(curr_size):
        i = (history_count - curr_size + offset) % history_size
        rho_i = rho_history[i]
        y_i = y_history[i]
        s_i = s_history[i]
        a_i = a_his[i]
        b_i = rho_i * np.dot(y_i, q)
        q = q + (a_i - b_i) * s_i
    return q


def _record_invalid_step_event(
    events,
    *,
    capacity,
    iteration,
    step_scale,
    line_search_failed,
    nonfinite_step,
    stalled_step,
    valid_curvature,
    trial_converged,
    ls_status,
):
    event = _HostInvalidStepEvent(
        iteration=int(iteration),
        step_scale=float(step_scale),
        line_search_failed=bool(line_search_failed),
        nonfinite_step=bool(nonfinite_step),
        stalled_step=bool(stalled_step),
        valid_curvature=bool(valid_curvature),
        trial_converged=bool(trial_converged),
        ls_status=int(ls_status),
    )
    keep = max(int(capacity) - 1, 0)
    return (*tuple(events)[-keep:], event)


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

    for index, event in enumerate(recent):
        iteration[index] = event.iteration
        step_scale[index] = event.step_scale
        line_search_failed[index] = event.line_search_failed
        nonfinite_step[index] = event.nonfinite_step
        stalled_step[index] = event.stalled_step
        valid_curvature[index] = event.valid_curvature
        trial_converged[index] = event.trial_converged
        ls_status[index] = event.ls_status

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
    )


def _host_state_to_lbfgs_results(state, *, invalid_log_capacity, dtype):
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
    can_cache_kernel = (
        cache_owner is not None
        and (adapter is None or structured_solver_cache_token is not None)
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


def _coerce_initial_value_and_grad_result_host(initial_value_and_grad, x_shape, *, dtype):
    value, grad = initial_value_and_grad
    value = float(_as_host_scalar(value, dtype=dtype))
    grad = _as_host_array(grad, dtype=dtype)
    if grad.shape != tuple(x_shape):
        raise ValueError(
            "initial_value_and_grad must provide a gradient matching "
            f"x.shape={tuple(x_shape)}, got {grad.shape}."
        )
    return value, grad


def _emit_iteration_callbacks_host(
    callback,
    progress_callback,
    x_kp1,
    next_k,
    f_kp1,
    g_kp1,
):
    if callback is not None:
        callback(np.asarray(x_kp1, dtype=float))
    if progress_callback is not None:
        progress_callback(
            int(next_k),
            float(f_kp1),
            float(_host_norm(g_kp1, ord=np.inf)),
        )


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
    np_dtype = np.dtype(dtype)
    x0_host = _as_host_array(x0, dtype=dtype)
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
        else np.asarray(initial_step_size, dtype=np_dtype).item()
    )
    if initial_step_size_value is not None and initial_step_size_value <= 0.0:
        raise ValueError("initial_step_size must be positive when provided.")
    ftol_value = np.asarray(ftol, dtype=np_dtype).item()
    gtol_value = np.asarray(gtol, dtype=np_dtype).item()
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

    if initial_value_and_grad is None:
        f_0, g_0 = eval_value_and_grad_host(x0_host)
    else:
        f_0, g_0 = _coerce_initial_value_and_grad_result_host(
            initial_value_and_grad,
            x0.shape,
            dtype=dtype,
        )
    initial_nonfinite = (not np.isfinite(f_0)) or (not np.all(np.isfinite(g_0)))
    initial_converged = (
        (not initial_nonfinite) and (_host_norm(g_0, ord=norm) < gtol_value)
    )

    initial_status = _host_limit_status(
        0,
        1,
        1,
        maxiter_limit=maxiter_limit_value,
        maxfun_limit=maxfun_limit_value,
        maxgrad_limit=maxgrad_limit_value,
    )
    if initial_nonfinite:
        initial_status = LBFGS_STATUS_NONFINITE

    state = _HostLBFGSState(
        converged=bool(initial_converged),
        failed=((initial_status > 0) and (not initial_converged)) or initial_nonfinite,
        k=0,
        nfev=1,
        ngev=1,
        x_k=x0_host,
        f_k=float(f_0),
        g_k=np.asarray(g_0, dtype=np_dtype),
        old_old_fval=float(f_0 + _host_norm(g_0) * 0.5),
        s_history=np.zeros((history_size, d), dtype=np_dtype),
        y_history=np.zeros((history_size, d), dtype=np_dtype),
        rho_history=np.zeros((history_size,), dtype=np_dtype),
        gamma=float(np_dtype.type(1.0)),
        history_count=0,
        status=0 if initial_converged else int(initial_status),
        ls_status=0,
        invalid_step_events=(),
    )
    _emit_lbfgs_state_dump(state)

    for _ in range(
        _host_iteration_budget(
            d,
            maxiter_limit_value,
            maxfun_limit_value,
            maxgrad_limit_value,
        )
    ):
        if (
            state.converged
            or state.failed
            or state.status != 0
            or (not np.isfinite(state.f_k))
            or (not np.all(np.isfinite(state.g_k)))
        ):
            break

        step_eps = np.sqrt(np.finfo(np_dtype).eps)
        gamma_max = 1.0 / step_eps
        line_search_initial_step_size = (
            None
            if initial_step_size_value is None
            else (
                np_dtype.type(initial_step_size_value)
                if state.k == 0
                else np_dtype.type(0.0)
            )
        )
        _emit_lbfgs_runtime_debug(
            "body_entry",
            iteration=state.k,
            fun_value=state.f_k,
            grad=state.g_k,
            step_scale=line_search_initial_step_size,
            ls_status=state.ls_status,
            converged=state.converged,
        )
        p_k = _two_loop_recursion_host(
            state.g_k,
            state.gamma,
            state.s_history,
            state.y_history,
            state.rho_history,
            state.history_count,
        )
        ls_results = _coerce_line_search_results_host(
            _line_search_value_and_grad_host(
                fun=eval_value_and_grad_host,
                xk=state.x_k,
                pk=p_k,
                old_fval=state.f_k,
                gfk=state.g_k,
                old_old_fval=state.old_old_fval,
                initial_step_size=line_search_initial_step_size,
                maxiter=maxls,
            ),
            dtype=dtype,
        )

        next_nfev = state.nfev + int(ls_results.nfev)
        next_ngev = state.ngev + int(ls_results.ngev)
        s_k = np_dtype.type(ls_results.a_k) * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = float(ls_results.f_k)
        g_kp1 = np.asarray(ls_results.g_k, dtype=np_dtype)
        y_k = g_kp1 - state.g_k
        rho_k_inv = float(np.dot(y_k, s_k))
        y_norm_sq = float(np.dot(y_k, y_k))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rho_k = float(np.divide(np_dtype.type(1.0), np_dtype.type(rho_k_inv)))
            gamma_raw = float(
                np.divide(np_dtype.type(rho_k_inv), np_dtype.type(y_norm_sq))
            )
        gamma = float(np.clip(np_dtype.type(gamma_raw), step_eps, gamma_max))
        next_k = state.k + 1
        s_k_norm = float(_host_norm(s_k))
        y_k_norm = float(_host_norm(y_k))
        converged = bool(_host_norm(g_kp1, ord=norm) < gtol_value)
        step_tol = step_eps * max(1.0, float(_host_norm(state.x_k)))
        function_change = abs(state.f_k - f_kp1)
        objective_tol = step_eps * max(abs(state.f_k), abs(f_kp1))
        gradient_change = y_k_norm
        gradient_tol = step_eps * max(
            float(_host_norm(state.g_k)),
            float(_host_norm(g_kp1)),
        )
        stalled_step = (
            (not converged)
            and (s_k_norm <= step_tol)
            and (function_change <= objective_tol)
            and (gradient_change <= gradient_tol)
        )
        curvature_scale = s_k_norm * y_k_norm
        curvature_tol = step_eps * curvature_scale
        valid_curvature = (
            np.isfinite(rho_k_inv)
            and np.isfinite(y_norm_sq)
            and np.isfinite(curvature_scale)
            and np.isfinite(gamma_raw)
            and np.isfinite(rho_k)
            and (rho_k_inv > curvature_tol)
            and (y_norm_sq > 0.0)
        )
        update_curvature = valid_curvature and (not stalled_step)
        nonfinite_step = (
            (not np.isfinite(f_kp1))
            or (not np.all(np.isfinite(g_kp1)))
            or (not np.all(np.isfinite(s_k)))
            or (not np.all(np.isfinite(x_kp1)))
            or (not np.all(np.isfinite(y_k)))
        )
        rejected_step = bool(ls_results.failed) or nonfinite_step or stalled_step
        ls_status = int(ls_results.status)
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

        if rejected_step:
            invalid_step_events = _record_invalid_step_event(
                state.invalid_step_events,
                capacity=invalid_log_capacity,
                iteration=next_k,
                step_scale=ls_results.a_k,
                line_search_failed=ls_results.failed,
                nonfinite_step=nonfinite_step,
                stalled_step=stalled_step,
                valid_curvature=valid_curvature,
                trial_converged=converged,
                ls_status=ls_status,
            )
            if failure_callback is not None:
                failure_callback(
                    int(next_k),
                    np.asarray(x_kp1, dtype=np.float64),
                    float(f_kp1),
                    np.asarray(g_kp1, dtype=np.float64),
                    np.asarray(p_k, dtype=np.float64),
                    np.asarray(s_k, dtype=np.float64),
                    float(ls_results.a_k),
                    bool(ls_results.failed),
                    bool(nonfinite_step),
                    bool(stalled_step),
                    bool(valid_curvature),
                    bool(converged),
                    int(ls_status),
                )
            state = state._replace(
                converged=False,
                failed=True,
                nfev=next_nfev,
                ngev=next_ngev,
                status=5,
                ls_status=ls_status if bool(ls_results.failed) else 0,
                invalid_step_events=invalid_step_events,
            )
            break

        status = 0
        if (
            _relative_objective_reduction_host(state.f_k, f_kp1, dtype=dtype)
            <= ftol_value
        ):
            status = 4
        limit_status = _host_limit_status(
            next_k,
            next_nfev,
            next_ngev,
            maxiter_limit=maxiter_limit_value,
            maxfun_limit=maxfun_limit_value,
            maxgrad_limit=maxgrad_limit_value,
        )
        if limit_status != 0:
            status = limit_status

        _emit_iteration_callbacks_host(
            callback,
            progress_callback,
            x_kp1=x_kp1,
            next_k=next_k,
            f_kp1=f_kp1,
            g_kp1=g_kp1,
        )

        if update_curvature:
            history_index = state.history_count % history_size
            s_history = np.array(state.s_history, copy=True)
            y_history = np.array(state.y_history, copy=True)
            rho_history = np.array(state.rho_history, copy=True)
            s_history[history_index] = s_k
            y_history[history_index] = y_k
            rho_history[history_index] = np_dtype.type(rho_k)
            next_history_count = state.history_count + 1
            next_gamma = gamma
        else:
            s_history = state.s_history
            y_history = state.y_history
            rho_history = state.rho_history
            next_history_count = state.history_count
            next_gamma = state.gamma

        state = state._replace(
            converged=converged,
            failed=(status > 0) and (status != 4) and (not converged),
            k=next_k,
            nfev=next_nfev,
            ngev=next_ngev,
            x_k=np.asarray(x_kp1, dtype=np_dtype),
            f_k=float(np_dtype.type(f_kp1)),
            g_k=np.asarray(g_kp1, dtype=np_dtype),
            old_old_fval=state.f_k,
            s_history=s_history,
            y_history=y_history,
            rho_history=rho_history,
            gamma=float(np_dtype.type(next_gamma)),
            history_count=next_history_count,
            status=0 if converged else status,
            ls_status=ls_status,
        )

    _emit_lbfgs_runtime_debug(
        "pre_final_eval",
        iteration=state.k,
        fun_value=state.f_k,
        grad=state.g_k,
        ls_status=state.ls_status,
        converged=state.converged,
    )
    if initial_value_and_grad is None or state.k > 0:
        f_final, g_final = eval_value_and_grad_host(state.x_k)
        final_eval_increment = 1
    else:
        f_final, g_final = f_0, g_0
        final_eval_increment = 0
    _emit_lbfgs_runtime_debug(
        "post_final_eval",
        iteration=state.k,
        fun_value=f_final,
        grad=g_final,
        ls_status=state.ls_status,
        converged=state.converged,
    )
    state_nonfinite = (not np.isfinite(f_final)) or (
        not np.all(np.isfinite(g_final))
    )
    converged_final = (
        (not state_nonfinite) and (_host_norm(g_final, ord=norm) < gtol_value)
    )
    state = state._replace(
        converged=bool(converged_final),
        failed=bool(state.failed or state_nonfinite),
        nfev=state.nfev + final_eval_increment,
        ngev=state.ngev + final_eval_increment,
        f_k=float(f_final),
        g_k=np.asarray(g_final, dtype=np_dtype),
    )
    limit_status = _host_limit_status(
        state.k,
        state.nfev,
        state.ngev,
        maxiter_limit=maxiter_limit_value,
        maxfun_limit=maxfun_limit_value,
        maxgrad_limit=maxgrad_limit_value,
    )
    if state.converged:
        status = 0
    elif state_nonfinite:
        status = LBFGS_STATUS_NONFINITE
    elif limit_status != 0:
        status = limit_status
    elif state.failed:
        status = 5
    else:
        status = state.status
    state = state._replace(status=int(status))
    return _host_state_to_lbfgs_results(
        state,
        invalid_log_capacity=invalid_log_capacity,
        dtype=dtype,
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
