"""Pure host L-BFGS core shared by JAX target and CPU reference lanes."""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np


LBFGS_STATUS_NONFINITE = 6
_INT32_COUNTER_MAX = np.iinfo(np.int32).max
_INVALID_STEP_LOG_MAX_CAPACITY = 256
_DEFAULT_OPTIMIZER_STATE_TRACE_MAX_BYTES = 64 * 1024 * 1024
_OPTIMIZER_STATE_TRACE_ARRAYS_PER_ENTRY = 6
_OPTIMIZER_STATE_TRACE_SCALARS_PER_ENTRY = 24
LINE_SEARCH_FAILURE_REASON_ACCEPTED = "accepted"
LINE_SEARCH_FAILURE_REASON_NOT_DESCENT = "not_descent"
LINE_SEARCH_FAILURE_REASON_FAILED = "line_search_failed"
LINE_SEARCH_FAILURE_REASON_MAXITER = "maxiter_exhausted"
_LINE_SEARCH_FAILURE_REASON_CODES = {
    LINE_SEARCH_FAILURE_REASON_ACCEPTED: 0,
    LINE_SEARCH_FAILURE_REASON_NOT_DESCENT: 1,
    LINE_SEARCH_FAILURE_REASON_FAILED: 2,
    LINE_SEARCH_FAILURE_REASON_MAXITER: 3,
}
_LINE_SEARCH_FAILURE_REASONS_BY_CODE = {
    code: reason for reason, code in _LINE_SEARCH_FAILURE_REASON_CODES.items()
}


def line_search_failure_reason_to_code(reason):
    return _LINE_SEARCH_FAILURE_REASON_CODES[str(reason)]


def line_search_failure_reason_from_code(code):
    return _LINE_SEARCH_FAILURE_REASONS_BY_CODE[int(code)]


class HostLineSearchResults(NamedTuple):
    failed: bool
    nit: int
    nfev: int
    ngev: int
    k: int
    a_k: float
    f_k: float
    g_k: np.ndarray
    status: int
    requested_initial_step: float
    first_tested_alpha: float
    best_finite_alpha: float
    returned_alpha: float
    failure_reason: str
    armijo_margin: float
    curvature_margin: float


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
    best_finite_a: float
    best_finite_phi: float
    best_finite_dphi: float
    best_finite_g: np.ndarray
    first_tested_alpha: float
    nfev: int
    ngev: int
    a_star: float
    phi_star: float
    dphi_star: float
    g_star: np.ndarray


class HostInvalidStepEvent(NamedTuple):
    iteration: int
    step_scale: float
    line_search_failed: bool
    nonfinite_step: bool
    stalled_step: bool
    valid_curvature: bool
    trial_converged: bool
    ls_status: int
    requested_initial_step: float
    first_tested_alpha: float
    best_finite_alpha: float
    returned_alpha: float
    failure_reason: str
    armijo_margin: float
    curvature_margin: float


class HostLBFGSResult(NamedTuple):
    converged: bool
    failed: bool
    k: int
    nfev: int
    ngev: int
    x_k: np.ndarray
    f_k: float
    g_k: np.ndarray
    s_history: np.ndarray
    y_history: np.ndarray
    rho_history: np.ndarray
    gamma: float
    status: int
    ls_status: int
    invalid_step_events: tuple[HostInvalidStepEvent, ...]
    optimizer_state_trace: tuple[dict[str, object], ...] = ()


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
    invalid_step_events: tuple[HostInvalidStepEvent, ...]


def host_norm(x, *, ord=None):
    abs_x = np.abs(np.asarray(x))
    if ord in (None, 2):
        return np.sqrt(np.sum(abs_x * abs_x))
    if ord == np.inf:
        return np.max(abs_x)
    return np.linalg.norm(x, ord=ord)


def _normalize_counter_limit(limit) -> np.int32:
    scalar_limit = np.asarray(limit).item()
    if np.isinf(scalar_limit):
        return np.int32(_INT32_COUNTER_MAX)
    return np.int32(min(int(np.ceil(scalar_limit)), _INT32_COUNTER_MAX))


def resolve_lbfgs_limits(d, maxiter, maxfun, maxgrad):
    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
    return tuple(
        _normalize_counter_limit(limit)
        for limit in (
            np.inf if maxiter is None else maxiter,
            np.inf if maxfun is None else maxfun,
            np.inf if maxgrad is None else maxgrad,
        )
    )


def resolve_lbfgs_history_size(maxcor, *, d, maxiter_limit_value):
    return max(1, min(int(maxcor), int(d), int(maxiter_limit_value)))


def relative_objective_reduction_host(f_k, f_kp1, *, dtype):
    dtype = np.dtype(dtype)
    denominator = max(abs(float(f_k)), abs(float(f_kp1)), float(dtype.type(1.0)))
    return (float(f_k) - float(f_kp1)) / denominator


def _iteration_budget(d, maxiter_limit, maxfun_limit, maxgrad_limit):
    finite_limits = [
        int(limit)
        for limit in (maxiter_limit, maxfun_limit, maxgrad_limit)
        if int(limit) < _INT32_COUNTER_MAX
    ]
    if finite_limits:
        return max(0, min(finite_limits))
    return max(0, int(d) * 200)


def _limit_status(k, nfev, ngev, *, maxiter_limit, maxfun_limit, maxgrad_limit):
    status = 0
    if int(ngev) >= int(maxgrad_limit):
        status = 3
    if int(nfev) >= int(maxfun_limit):
        status = 2
    if int(k) >= int(maxiter_limit):
        status = 1
    return status


def _cubicmin(a, fa, fpa, b, fb, c, fc):
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


def _quadmin(a, fa, fpa, b, fb):
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


def _line_search_sample_valid(phi, dphi, grad):
    return (
        np.isfinite(phi) and np.isfinite(dphi) and np.all(np.isfinite(np.asarray(grad)))
    )


def _line_search_margins(*, phi_0, dphi_0, c1, c2, alpha, phi, dphi):
    if (
        float(alpha) <= 0.0
        or (not np.isfinite(phi))
        or (not np.isfinite(dphi))
        or (not np.isfinite(dphi_0))
    ):
        return float("nan"), float("nan")
    armijo_margin = float(phi - (phi_0 + float(c1) * float(alpha) * dphi_0))
    curvature_margin = float(abs(dphi) - (-float(c2) * dphi_0))
    return armijo_margin, curvature_margin


def _cache_zoom_sample(state, *, alpha, phi, dphi, grad):
    return state._replace(
        has_rec=True,
        a_rec=float(alpha),
        phi_rec=float(phi),
        dphi_rec=float(dphi),
        g_rec=np.asarray(grad, dtype=state.g_rec.dtype),
    )


def _zoom(
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
    lo_valid = _line_search_sample_valid(phi_lo, dphi_lo, g_lo)
    hi_valid = _line_search_sample_valid(phi_hi, dphi_hi, g_hi)
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
        a_star=0.0,
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
        a_j_cubic = _cubicmin(
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
        a_j_quad = _quadmin(
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

        lo_valid = _line_search_sample_valid(state.phi_lo, state.dphi_lo, state.g_lo)
        hi_valid = _line_search_sample_valid(state.phi_hi, state.dphi_hi, state.g_hi)
        nonfinite_shrink = float(dtype.type(1.0e-2))
        if lo_valid and not hi_valid:
            a_j = float(state.a_lo + nonfinite_shrink * (state.a_hi - state.a_lo))
        elif hi_valid and not lo_valid:
            a_j = float(state.a_hi + nonfinite_shrink * (state.a_lo - state.a_hi))

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
        sample_valid = _line_search_sample_valid(phi_j, dphi_j, g_j)
        if sample_valid and phi_j < state.best_phi:
            state = state._replace(
                best_a=float(a_j),
                best_phi=float(phi_j),
                best_dphi=float(dphi_j),
                best_g=g_j,
            )

        hi_to_j = (not sample_valid) or wolfe_one(a_j, phi_j) or (phi_j >= state.phi_lo)
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
            state = _cache_zoom_sample(
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
            state = _cache_zoom_sample(
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
                state = _cache_zoom_sample(
                    state,
                    alpha=previous_a_lo,
                    phi=previous_phi_lo,
                    dphi=previous_dphi_lo,
                    grad=previous_g_lo,
                )

        state = state._replace(j=state.j + 1)
        if state.j >= max_zoom_iter and not state.done:
            state = state._replace(failed=True)

    best_is_acceptable = _line_search_sample_valid(
        state.best_phi, state.best_dphi, state.best_g
    ) and (state.best_phi < phi_0)
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


def _apply_zoom_branch_result(state, zoom, *, wolfe_one):
    state = state._replace(
        nfev=state.nfev + zoom.nfev,
        ngev=state.ngev + zoom.ngev,
    )
    improves_best = _line_search_sample_valid(
        zoom.best_phi, zoom.best_dphi, zoom.best_g
    ) and (zoom.best_phi < state.best_phi)
    if improves_best:
        state = state._replace(
            best_a=zoom.best_a,
            best_phi=zoom.best_phi,
            best_dphi=zoom.best_dphi,
            best_g=zoom.best_g,
        )
    improves_best_finite = _line_search_sample_valid(
        zoom.best_phi, zoom.best_dphi, zoom.best_g
    ) and (zoom.best_phi < state.best_finite_phi)
    if improves_best_finite:
        state = state._replace(
            best_finite_a=zoom.best_a,
            best_finite_phi=zoom.best_phi,
            best_finite_dphi=zoom.best_dphi,
            best_finite_g=zoom.best_g,
        )
    return state._replace(
        done=True,
        failed=zoom.failed or state.failed,
        a_star=zoom.a_star,
        phi_star=zoom.phi_star,
        dphi_star=zoom.dphi_star,
        g_star=zoom.g_star,
    )


def _line_search_from_restricted_func_and_grad(
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
        return HostLineSearchResults(
            failed=True,
            nit=0,
            nfev=0,
            ngev=0,
            k=1,
            a_k=float(zero),
            f_k=float(phi_0),
            g_k=gfk,
            status=1,
            requested_initial_step=0.0,
            first_tested_alpha=0.0,
            best_finite_alpha=0.0,
            returned_alpha=0.0,
            failure_reason=LINE_SEARCH_FAILURE_REASON_NOT_DESCENT,
            armijo_margin=float("nan"),
            curvature_margin=float("nan"),
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
        best_finite_a=float(zero),
        best_finite_phi=float("inf"),
        best_finite_dphi=float("nan"),
        best_finite_g=gfk,
        first_tested_alpha=float(zero),
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
        first_tested_alpha = float(a_i) if state.nfev == 0 else state.first_tested_alpha
        state = state._replace(
            first_tested_alpha=first_tested_alpha,
            nfev=state.nfev + 1,
            ngev=state.ngev + 1,
        )

        sample_valid = _line_search_sample_valid(phi_i, dphi_i, g_i)
        improves_best_finite_i = sample_valid and phi_i < state.best_finite_phi
        if improves_best_finite_i:
            state = state._replace(
                best_finite_a=float(a_i),
                best_finite_phi=float(phi_i),
                best_finite_dphi=float(dphi_i),
                best_finite_g=g_i,
            )
        improves_best_i = (
            sample_valid and (not wolfe_one(a_i, phi_i)) and (phi_i < state.best_phi)
        )
        if improves_best_i:
            state = state._replace(
                best_a=float(a_i),
                best_phi=float(phi_i),
                best_dphi=float(dphi_i),
                best_g=g_i,
            )

        star_to_zoom1 = (
            (not sample_valid)
            or wolfe_one(a_i, phi_i)
            or ((phi_i >= state.phi_i1) and (state.i > 1))
        )
        star_to_i = sample_valid and wolfe_two(dphi_i) and not star_to_zoom1
        star_to_zoom2 = (
            sample_valid and (dphi_i >= 0.0) and not star_to_zoom1 and not star_to_i
        )
        remaining_zoom_budget = max(0, maxiter_value - state.nfev)
        zoom_budget_exhausted = remaining_zoom_budget <= 0

        if star_to_zoom1:
            zoom = _zoom(
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
            state = _apply_zoom_branch_result(state, zoom, wolfe_one=wolfe_one)
        elif star_to_i:
            state = state._replace(
                done=True,
                a_star=float(a_i),
                phi_star=float(phi_i),
                dphi_star=float(dphi_i),
                g_star=g_i,
            )
        elif star_to_zoom2:
            zoom = _zoom(
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
            state = _apply_zoom_branch_result(state, zoom, wolfe_one=wolfe_one)

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

    best_is_acceptable = _line_search_sample_valid(
        state.best_finite_phi,
        state.best_finite_dphi,
        state.best_finite_g,
    ) and (state.best_finite_phi < phi_0)
    if (state.failed or (not state.done)) and best_is_acceptable:
        state = state._replace(
            failed=False,
            done=True,
            a_star=state.best_finite_a,
            phi_star=state.best_finite_phi,
            dphi_star=state.best_finite_dphi,
            g_star=state.best_finite_g,
        )

    status = (
        0
        if state.done and not state.failed
        else (1 if state.failed else (3 if state.i > maxiter_value else 0))
    )
    line_search_failed = bool(state.failed or (not state.done))
    failure_reason = LINE_SEARCH_FAILURE_REASON_ACCEPTED
    if line_search_failed:
        failure_reason = (
            LINE_SEARCH_FAILURE_REASON_FAILED
            if state.failed
            else LINE_SEARCH_FAILURE_REASON_MAXITER
        )
    diagnostic_alpha = state.a_star if not line_search_failed else state.best_finite_a
    diagnostic_phi = state.phi_star if not line_search_failed else state.best_finite_phi
    diagnostic_dphi = (
        state.dphi_star if not line_search_failed else state.best_finite_dphi
    )
    armijo_margin, curvature_margin = _line_search_margins(
        phi_0=phi_0,
        dphi_0=dphi_0,
        c1=c1,
        c2=c2,
        alpha=diagnostic_alpha,
        phi=diagnostic_phi,
        dphi=diagnostic_dphi,
    )
    alpha_k = dtype.type(0.0 if line_search_failed else state.a_star)
    if np.finfo(dtype).bits != 64 and abs(alpha_k) < dtype.type(1e-8):
        alpha_k = np.sign(alpha_k) * dtype.type(1e-8)
    return HostLineSearchResults(
        failed=line_search_failed,
        nit=state.i - 1,
        nfev=state.nfev,
        ngev=state.ngev,
        k=state.i,
        a_k=float(alpha_k),
        f_k=float(phi_0 if line_search_failed else state.phi_star),
        g_k=np.asarray(gfk if line_search_failed else state.g_star, dtype=dtype),
        status=status,
        requested_initial_step=float(start_value),
        first_tested_alpha=float(state.first_tested_alpha),
        best_finite_alpha=float(state.best_finite_a),
        returned_alpha=float(state.a_star),
        failure_reason=failure_reason,
        armijo_margin=float(armijo_margin),
        curvature_margin=float(curvature_margin),
    )


def line_search_value_and_grad_host(
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

    return _line_search_from_restricted_func_and_grad(
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


def coerce_line_search_results(results, *, dtype):
    failed = bool(np.asarray(results.failed).item())
    raw_a_k = float(np.asarray(results.a_k, dtype=np.dtype(dtype)).item())
    a_k = 0.0 if failed else raw_a_k
    f_k = float(np.asarray(results.f_k, dtype=np.dtype(dtype)).item())
    g_k = np.asarray(results.g_k, dtype=np.dtype(dtype))
    requested_initial_step = float(
        np.asarray(
            getattr(results, "requested_initial_step", raw_a_k),
            dtype=np.dtype(dtype),
        ).item()
    )
    first_tested_alpha = float(
        np.asarray(
            getattr(results, "first_tested_alpha", requested_initial_step),
            dtype=np.dtype(dtype),
        ).item()
    )
    best_finite_alpha = float(
        np.asarray(
            getattr(results, "best_finite_alpha", raw_a_k),
            dtype=np.dtype(dtype),
        ).item()
    )
    returned_alpha = float(
        np.asarray(
            getattr(results, "returned_alpha", raw_a_k),
            dtype=np.dtype(dtype),
        ).item()
    )
    failure_reason = getattr(
        results,
        "failure_reason",
        (
            LINE_SEARCH_FAILURE_REASON_FAILED
            if failed
            else LINE_SEARCH_FAILURE_REASON_ACCEPTED
        ),
    )
    return HostLineSearchResults(
        failed=failed,
        nit=int(np.asarray(results.nit).item()),
        nfev=int(np.asarray(results.nfev).item()),
        ngev=int(np.asarray(results.ngev).item()),
        k=int(np.asarray(results.k).item()),
        a_k=a_k,
        f_k=f_k,
        g_k=g_k,
        status=int(np.asarray(results.status).item()),
        requested_initial_step=requested_initial_step,
        first_tested_alpha=first_tested_alpha,
        best_finite_alpha=best_finite_alpha,
        returned_alpha=returned_alpha,
        failure_reason=str(failure_reason),
        armijo_margin=float(
            np.asarray(
                getattr(results, "armijo_margin", np.nan),
                dtype=np.dtype(dtype),
            ).item()
        ),
        curvature_margin=float(
            np.asarray(
                getattr(results, "curvature_margin", np.nan),
                dtype=np.dtype(dtype),
            ).item()
        ),
    )


def two_loop_recursion_host(
    g_k,
    gamma,
    s_history,
    y_history,
    rho_history,
    history_count,
):
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
    requested_initial_step,
    first_tested_alpha,
    best_finite_alpha,
    returned_alpha,
    failure_reason,
    armijo_margin,
    curvature_margin,
):
    event = HostInvalidStepEvent(
        iteration=int(iteration),
        step_scale=float(step_scale),
        line_search_failed=bool(line_search_failed),
        nonfinite_step=bool(nonfinite_step),
        stalled_step=bool(stalled_step),
        valid_curvature=bool(valid_curvature),
        trial_converged=bool(trial_converged),
        ls_status=int(ls_status),
        requested_initial_step=float(requested_initial_step),
        first_tested_alpha=float(first_tested_alpha),
        best_finite_alpha=float(best_finite_alpha),
        returned_alpha=float(returned_alpha),
        failure_reason=str(failure_reason),
        armijo_margin=float(armijo_margin),
        curvature_margin=float(curvature_margin),
    )
    keep = max(int(capacity) - 1, 0)
    return (*tuple(events)[-keep:], event)


def optimizer_state_trace_entry(
    *,
    iteration,
    x,
    f,
    g,
    search_direction,
    step_scale,
    step,
    trial_x,
    trial_f,
    trial_g,
    nfev,
    njev,
    line_search_status,
    line_search_failed,
    valid_curvature,
    curvature_update_applied,
    rho_inv,
    gamma,
    history_count_before,
    history_count_after,
    converged,
):
    dphi0 = float(np.dot(g, search_direction))
    dphi_trial = float(np.dot(trial_g, search_direction))
    armijo_margin = float(trial_f - (f + 1e-4 * step_scale * dphi0))
    curvature_margin = float(abs(dphi_trial) - (-0.9 * dphi0))
    wolfe_satisfied = (
        np.isfinite(armijo_margin)
        and np.isfinite(curvature_margin)
        and armijo_margin <= 0.0
        and curvature_margin <= 0.0
    )
    return {
        "iteration": int(iteration),
        "x": np.asarray(x, dtype=np.float64),
        "fun": float(f),
        "jac": np.asarray(g, dtype=np.float64),
        "jac_inf_norm": float(host_norm(g, ord=np.inf)),
        "search_direction": np.asarray(search_direction, dtype=np.float64),
        "search_direction_dot_grad": dphi0,
        "step_scale": float(step_scale),
        "step": np.asarray(step, dtype=np.float64),
        "trial_x": np.asarray(trial_x, dtype=np.float64),
        "trial_fun": float(trial_f),
        "trial_jac": np.asarray(trial_g, dtype=np.float64),
        "trial_jac_inf_norm": float(host_norm(trial_g, ord=np.inf)),
        "nfev": int(nfev),
        "njev": int(njev),
        "line_search_status": int(line_search_status),
        "line_search_failed": bool(line_search_failed),
        "valid_curvature": bool(valid_curvature),
        "curvature_update_applied": bool(curvature_update_applied),
        "rho_inv": float(rho_inv),
        "gamma": float(gamma),
        "history_count_before": int(history_count_before),
        "history_count_after": int(history_count_after),
        "dphi_trial": dphi_trial,
        "armijo_margin": armijo_margin,
        "curvature_margin": curvature_margin,
        "wolfe_satisfied": bool(wolfe_satisfied),
        "line_search_acceptance": (
            "strong_wolfe" if wolfe_satisfied else "best_improving_or_limit"
        ),
        "accepted": True,
        "converged": bool(converged),
    }


def optimizer_state_trace_memory_bytes(d, iterations):
    return int(iterations) * (
        (
            _OPTIMIZER_STATE_TRACE_ARRAYS_PER_ENTRY * int(d)
            + _OPTIMIZER_STATE_TRACE_SCALARS_PER_ENTRY
        )
        * np.dtype(np.float64).itemsize
    )


def _check_optimizer_state_trace_budget(
    d,
    maxiter_limit,
    *,
    record_optimizer_state_trace,
    max_optimizer_state_trace_bytes,
):
    if not record_optimizer_state_trace:
        return
    iterations = max(1, int(maxiter_limit))
    trace_bytes = optimizer_state_trace_memory_bytes(d, iterations)
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


def _coerce_initial_value_and_grad(initial_value_and_grad, x_shape, *, dtype):
    value, grad = initial_value_and_grad
    value = float(np.asarray(value, dtype=np.dtype(dtype)).reshape(()).item())
    grad = np.asarray(grad, dtype=np.dtype(dtype))
    if grad.shape != tuple(x_shape):
        raise ValueError(
            "initial_value_and_grad must provide a gradient matching "
            f"x.shape={tuple(x_shape)}, got {grad.shape}."
        )
    return value, grad


def _emit_iteration_callbacks(
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
            float(host_norm(g_kp1, ord=np.inf)),
        )


def minimize_lbfgs_host_core(
    eval_value_and_grad_host: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0_host,
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
    line_search_value_and_grad=line_search_value_and_grad_host,
    record_optimizer_state_trace=False,
    max_optimizer_state_trace_bytes=None,
):
    x0_host = np.asarray(x0_host)
    if x0_host.ndim != 1:
        raise ValueError(
            f"L-BFGS expects a flat 1-D decision vector, got {x0_host.shape}."
        )
    d = len(x0_host)
    dtype = x0_host.dtype
    np_dtype = np.dtype(dtype)
    maxiter_limit_value, maxfun_limit_value, maxgrad_limit_value = resolve_lbfgs_limits(
        d, maxiter, maxfun, maxgrad
    )
    _check_optimizer_state_trace_budget(
        d,
        maxiter_limit_value,
        record_optimizer_state_trace=bool(record_optimizer_state_trace),
        max_optimizer_state_trace_bytes=max_optimizer_state_trace_bytes,
    )
    history_size = resolve_lbfgs_history_size(
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

    if initial_value_and_grad is None:
        f_0, g_0 = eval_value_and_grad_host(x0_host)
    else:
        f_0, g_0 = _coerce_initial_value_and_grad(
            initial_value_and_grad,
            x0_host.shape,
            dtype=dtype,
        )
    initial_nonfinite = (not np.isfinite(f_0)) or (not np.all(np.isfinite(g_0)))
    initial_converged = (not initial_nonfinite) and (
        host_norm(g_0, ord=norm) < gtol_value
    )

    initial_status = _limit_status(
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
        old_old_fval=float(f_0 + host_norm(g_0) * 0.5),
        s_history=np.zeros((history_size, d), dtype=np_dtype),
        y_history=np.zeros((history_size, d), dtype=np_dtype),
        rho_history=np.zeros((history_size,), dtype=np_dtype),
        gamma=float(np_dtype.type(1.0)),
        history_count=0,
        status=0 if initial_converged else int(initial_status),
        ls_status=0,
        invalid_step_events=(),
    )
    optimizer_state_trace = ()

    for _ in range(
        _iteration_budget(
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
        p_k = two_loop_recursion_host(
            state.g_k,
            state.gamma,
            state.s_history,
            state.y_history,
            state.rho_history,
            state.history_count,
        )
        ls_results = coerce_line_search_results(
            line_search_value_and_grad(
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
        s_k_norm = float(host_norm(s_k))
        y_k_norm = float(host_norm(y_k))
        converged = bool(host_norm(g_kp1, ord=norm) < gtol_value)
        step_tol = step_eps * max(1.0, float(host_norm(state.x_k)))
        function_change = abs(state.f_k - f_kp1)
        objective_tol = step_eps * max(abs(state.f_k), abs(f_kp1))
        gradient_change = y_k_norm
        gradient_tol = step_eps * max(
            float(host_norm(state.g_k)),
            float(host_norm(g_kp1)),
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
                requested_initial_step=ls_results.requested_initial_step,
                first_tested_alpha=ls_results.first_tested_alpha,
                best_finite_alpha=ls_results.best_finite_alpha,
                returned_alpha=ls_results.returned_alpha,
                failure_reason=ls_results.failure_reason,
                armijo_margin=ls_results.armijo_margin,
                curvature_margin=ls_results.curvature_margin,
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
            relative_objective_reduction_host(state.f_k, f_kp1, dtype=dtype)
            <= ftol_value
        ):
            status = 4
        limit_status = _limit_status(
            next_k,
            next_nfev,
            next_ngev,
            maxiter_limit=maxiter_limit_value,
            maxfun_limit=maxfun_limit_value,
            maxgrad_limit=maxgrad_limit_value,
        )
        if limit_status != 0:
            status = limit_status

        history_count_before = state.history_count
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

        if record_optimizer_state_trace:
            optimizer_state_trace = (
                *optimizer_state_trace,
                optimizer_state_trace_entry(
                    iteration=next_k,
                    x=state.x_k,
                    f=state.f_k,
                    g=state.g_k,
                    search_direction=p_k,
                    step_scale=ls_results.a_k,
                    step=s_k,
                    trial_x=x_kp1,
                    trial_f=f_kp1,
                    trial_g=g_kp1,
                    nfev=next_nfev,
                    njev=next_ngev,
                    line_search_status=ls_status,
                    line_search_failed=ls_results.failed,
                    valid_curvature=valid_curvature,
                    curvature_update_applied=update_curvature,
                    rho_inv=rho_k_inv,
                    gamma=next_gamma,
                    history_count_before=history_count_before,
                    history_count_after=next_history_count,
                    converged=converged,
                ),
            )

        _emit_iteration_callbacks(
            callback,
            progress_callback,
            x_kp1=x_kp1,
            next_k=next_k,
            f_kp1=f_kp1,
            g_kp1=g_kp1,
        )

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

    if initial_value_and_grad is None or state.k > 0:
        f_final, g_final = eval_value_and_grad_host(state.x_k)
        final_eval_increment = 1
    else:
        f_final, g_final = f_0, g_0
        final_eval_increment = 0
    state_nonfinite = (not np.isfinite(f_final)) or (not np.all(np.isfinite(g_final)))
    converged_final = (not state_nonfinite) and (
        host_norm(g_final, ord=norm) < gtol_value
    )
    state = state._replace(
        converged=bool(converged_final),
        failed=bool(state.failed or state_nonfinite),
        nfev=state.nfev + final_eval_increment,
        ngev=state.ngev + final_eval_increment,
        f_k=float(f_final),
        g_k=np.asarray(g_final, dtype=np_dtype),
    )
    limit_status = _limit_status(
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
    return HostLBFGSResult(
        converged=state.converged,
        failed=state.failed,
        k=state.k,
        nfev=state.nfev,
        ngev=state.ngev,
        x_k=state.x_k,
        f_k=state.f_k,
        g_k=state.g_k,
        s_history=state.s_history,
        y_history=state.y_history,
        rho_history=state.rho_history,
        gamma=state.gamma,
        status=state.status,
        ls_status=state.ls_status,
        invalid_step_events=state.invalid_step_events,
        optimizer_state_trace=optimizer_state_trace,
    )


def host_invalid_step_log_to_list(events):
    return [
        {
            "iteration": int(event.iteration),
            "step_scale": float(event.step_scale),
            "line_search_failed": bool(event.line_search_failed),
            "nonfinite_step": bool(event.nonfinite_step),
            "stalled_step": bool(event.stalled_step),
            "valid_curvature": bool(event.valid_curvature),
            "trial_converged": bool(event.trial_converged),
            "ls_status": int(event.ls_status),
            "requested_initial_step": float(event.requested_initial_step),
            "first_tested_alpha": float(event.first_tested_alpha),
            "best_finite_alpha": float(event.best_finite_alpha),
            "returned_alpha": float(event.returned_alpha),
            "failure_reason": str(event.failure_reason),
            "armijo_margin": float(event.armijo_margin),
            "curvature_margin": float(event.curvature_margin),
        }
        for event in events
    ]


_LBFGS_STATUS_MESSAGES = {
    0: "Optimization terminated successfully.",
    1: "Maximum number of iterations reached.",
    2: "Maximum number of function evaluations reached.",
    3: "Maximum number of gradient evaluations reached.",
    4: "Optimization terminated successfully (ftol).",
    5: "Line search failed or produced an invalid step.",
    LBFGS_STATUS_NONFINITE: (
        "Non-finite objective or gradient encountered during iteration."
    ),
}
_LBFGS_SUCCESS_STATUSES = frozenset({0, 4})


def lbfgs_status_message(status, invalid_state):
    if invalid_state:
        return _LBFGS_STATUS_MESSAGES[LBFGS_STATUS_NONFINITE]
    return _LBFGS_STATUS_MESSAGES.get(
        int(status),
        f"Unknown L-BFGS termination status {int(status)}.",
    )


def lbfgs_status_is_success(status, invalid_state):
    return (int(status) in _LBFGS_SUCCESS_STATUSES) and not invalid_state
