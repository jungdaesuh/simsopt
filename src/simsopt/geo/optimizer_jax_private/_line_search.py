"""Strong Wolfe line search for on-device BFGS / L-BFGS.

Derived from the JAX 0.9.2 strong-Wolfe flow, with repo-specific changes for
the explicit old_fval/gfk contract and cached zoom-sample reuse.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
from jax import lax

from ._common import (
    _as_jax_dtype,
    _bool_scalar,
    _dot,
    _emit_host_callback,
    _int_scalar,
    _promote_dtypes_inexact,
    _scalar_value_and_grad,
)
from ._types import _LineSearchResults, _LineSearchState, _ZoomState


_LINE_SEARCH_DEBUG_ENABLED = os.environ.get("SIMSOPT_LBFGS_DEBUG", "").lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}


def _emit_line_search_runtime_debug(
    stage,
    *,
    iteration,
    alpha,
    phi,
    dphi,
):
    """Emit runtime diagnostics when SIMSOPT_LBFGS_DEBUG is enabled.

    The callback routes through ``_emit_host_callback`` (``ordered=False``) so
    strict ``transfer_guard='disallow'`` lanes do not trip on the JAX 0.9.2
    ``bool[0]`` host token associated with ``ordered=True``. One consequence is
    that debug prints from the line search may interleave with other unordered
    callbacks (e.g. the L-BFGS body debug). Use SIMSOPT_LBFGS_DEBUG only for
    ad-hoc tracing; do not rely on print ordering across stages.
    """
    if not _LINE_SEARCH_DEBUG_ENABLED:
        return
    _emit_host_callback(
        lambda i, a, f, df: print(
            "[line-search-debug] "
            f"stage={stage} "
            f"iter={int(i)} "
            f"alpha={float(a):.16e} "
            f"phi={float(f):.16e} "
            f"dphi={float(df):.16e}",
            flush=True,
        ),
        iteration,
        alpha,
        phi,
        dphi,
    )


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    dtype = jnp.result_type(a, fa, fpa, b, fb, c, fc)
    three = _as_jax_dtype(3.0, dtype)
    C = fpa
    db = b - a
    dc = c - a
    db2 = db * db
    dc2 = dc * dc
    denom = (db * dc) * (db * dc) * (db - dc)
    d1 = jnp.stack(
        (
            jnp.stack((dc2, -db2)),
            jnp.stack((-(dc2 * dc), db2 * db)),
        )
    ).astype(dtype)
    d2 = jnp.stack((fb - fa - C * db, fc - fa - C * dc)).astype(dtype)
    A, B = _dot(d1, d2) / denom

    radical = B * B - three * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (three * A)
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    dtype = jnp.result_type(a, fa, fpa, b, fb)
    two = _as_jax_dtype(2.0, dtype)
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db * db)
    xmin = a - C / (two * B)
    return xmin


def _line_search_sample_valid(phi, dphi, grad):
    return jnp.isfinite(phi) & jnp.isfinite(dphi) & jnp.all(jnp.isfinite(grad))


def _binary_replace(replace_bit, original_dict, new_dict, keys=None):
    if keys is None:
        keys = new_dict.keys()
    return {
        key: jnp.where(replace_bit, new_dict[key], original_dict[key]) for key in keys
    }


def _cache_zoom_sample(state, *, alpha, phi, dphi, grad):
    return state._replace(
        has_rec=_bool_scalar(True),
        a_rec=alpha,
        phi_rec=phi,
        dphi_rec=dphi,
        g_rec=grad,
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
    dtype = a_lo.dtype
    zero = _as_jax_dtype(0.0, dtype)
    one = _as_jax_dtype(1.0, dtype)
    half = _as_jax_dtype(0.5, dtype)
    delta1 = _as_jax_dtype(0.2, dtype)
    delta2 = _as_jax_dtype(0.1, dtype)
    max_zoom_iter = maxiter
    lo_valid = _line_search_sample_valid(phi_lo, dphi_lo, g_lo)
    hi_valid = _line_search_sample_valid(phi_hi, dphi_hi, g_hi)
    lo_is_better = lo_valid & ((~hi_valid) | (phi_lo <= phi_hi))
    state = _ZoomState(
        done=_bool_scalar(False),
        failed=pass_through,
        j=_int_scalar(0),
        a_lo=a_lo,
        phi_lo=phi_lo,
        dphi_lo=dphi_lo,
        g_lo=g_lo,
        a_hi=a_hi,
        phi_hi=phi_hi,
        dphi_hi=dphi_hi,
        g_hi=g_hi,
        has_rec=has_rec,
        a_rec=a_rec,
        phi_rec=phi_rec,
        dphi_rec=dphi_rec,
        g_rec=g_rec,
        a_star=one,
        phi_star=phi_lo,
        dphi_star=dphi_lo,
        g_star=g_lo,
        best_a=jnp.where(lo_is_better, a_lo, a_hi),
        best_phi=jnp.where(lo_is_better, phi_lo, phi_hi),
        best_dphi=jnp.where(lo_is_better, dphi_lo, dphi_hi),
        best_g=jnp.where(lo_is_better, g_lo, g_hi),
        nfev=_int_scalar(0),
        ngev=_int_scalar(0),
    )

    def body(state):
        dalpha = jnp.abs(state.a_hi - state.a_lo)
        a = jnp.minimum(state.a_hi, state.a_lo)
        b = jnp.maximum(state.a_hi, state.a_lo)
        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        if jnp.finfo(dalpha.dtype).bits < 64:
            threshold = _as_jax_dtype(1e-5, dalpha.dtype)
        else:
            threshold = _as_jax_dtype(1e-10, dalpha.dtype)
        state = state._replace(failed=state.failed | (dalpha <= threshold))

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
            state.has_rec & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
        )
        a_j_quad = _quadmin(
            state.a_lo,
            state.phi_lo,
            state.dphi_lo,
            state.a_hi,
            state.phi_hi,
        )
        use_quad = (~use_cubic) & (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
        a_j_bisection = (state.a_lo + state.a_hi) * half

        a_j = jnp.where(use_cubic, a_j_cubic, state.a_rec)
        a_j = jnp.where(use_quad, a_j_quad, a_j)
        a_j = jnp.where((~use_cubic) & (~use_quad), a_j_bisection, a_j)

        reuse_rec_sample = state.has_rec & (a_j == state.a_rec)
        phi_j, dphi_j, g_j, sample_eval_count = lax.cond(
            reuse_rec_sample,
            lambda _: (
                state.phi_rec,
                state.dphi_rec,
                state.g_rec,
                _int_scalar(0),
            ),
            lambda _: (*restricted_func_and_grad(a_j), _int_scalar(1)),
            operand=None,
        )
        phi_j = phi_j.astype(state.phi_lo.dtype)
        dphi_j = dphi_j.astype(state.dphi_lo.dtype)
        g_j = g_j.astype(state.g_star.dtype)
        _emit_line_search_runtime_debug(
            "zoom_trial",
            iteration=state.j + _int_scalar(1),
            alpha=a_j,
            phi=phi_j,
            dphi=dphi_j,
        )
        state = state._replace(
            nfev=state.nfev + sample_eval_count,
            ngev=state.ngev + sample_eval_count,
        )
        sample_valid = _line_search_sample_valid(phi_j, dphi_j, g_j)
        improves_best = sample_valid & (phi_j < state.best_phi)
        state = state._replace(
            best_a=jnp.where(improves_best, a_j, state.best_a),
            best_phi=jnp.where(improves_best, phi_j, state.best_phi),
            best_dphi=jnp.where(improves_best, dphi_j, state.best_dphi),
            best_g=jnp.where(improves_best, g_j, state.best_g),
        )

        hi_to_j = (~sample_valid) | wolfe_one(a_j, phi_j) | (phi_j >= state.phi_lo)
        star_to_j = sample_valid & wolfe_two(dphi_j) & (~hi_to_j)
        hi_to_lo = (
            sample_valid
            & (dphi_j * (state.a_hi - state.a_lo) >= zero)
            & (~hi_to_j)
            & (~star_to_j)
        )
        lo_to_j = sample_valid & (~hi_to_j) & (~star_to_j)
        previous_a_lo = state.a_lo
        previous_phi_lo = state.phi_lo
        previous_dphi_lo = state.dphi_lo
        previous_g_lo = state.g_lo
        previous_a_hi = state.a_hi
        previous_phi_hi = state.phi_hi
        previous_dphi_hi = state.dphi_hi
        previous_g_hi = state.g_hi

        state = state._replace(
            **_binary_replace(
                hi_to_j,
                state._asdict(),
                dict(
                    a_hi=a_j,
                    phi_hi=phi_j,
                    dphi_hi=dphi_j,
                    g_hi=g_j,
                ),
            )
        )
        state = lax.cond(
            hi_to_j,
            lambda current: _cache_zoom_sample(
                current,
                alpha=previous_a_hi,
                phi=previous_phi_hi,
                dphi=previous_dphi_hi,
                grad=previous_g_hi,
            ),
            lambda current: current,
            operand=state,
        )
        state = state._replace(
            done=star_to_j | state.done,
            **_binary_replace(
                star_to_j,
                state._asdict(),
                dict(a_star=a_j, phi_star=phi_j, dphi_star=dphi_j, g_star=g_j),
            ),
        )
        state = state._replace(
            **_binary_replace(
                hi_to_lo,
                state._asdict(),
                dict(
                    a_hi=state.a_lo,
                    phi_hi=state.phi_lo,
                    dphi_hi=state.dphi_lo,
                    g_hi=state.g_lo,
                ),
            )
        )
        state = lax.cond(
            hi_to_lo,
            lambda current: _cache_zoom_sample(
                current,
                alpha=previous_a_hi,
                phi=previous_phi_hi,
                dphi=previous_dphi_hi,
                grad=previous_g_hi,
            ),
            lambda current: current,
            operand=state,
        )
        state = state._replace(
            **_binary_replace(
                lo_to_j,
                state._asdict(),
                dict(a_lo=a_j, phi_lo=phi_j, dphi_lo=dphi_j, g_lo=g_j),
            )
        )
        state = lax.cond(
            lo_to_j & ~hi_to_lo,
            lambda current: _cache_zoom_sample(
                current,
                alpha=previous_a_lo,
                phi=previous_phi_lo,
                dphi=previous_dphi_lo,
                grad=previous_g_lo,
            ),
            lambda current: current,
            operand=state,
        )
        state = state._replace(j=state.j + _int_scalar(1))
        state = state._replace(failed=state.failed | (state.j >= max_zoom_iter))
        return state

    state = lax.while_loop(
        lambda state: (~state.done) & (~pass_through) & (~state.failed),
        body,
        state,
    )
    best_is_acceptable = (
        _line_search_sample_valid(state.best_phi, state.best_dphi, state.best_g)
        & (~wolfe_one(state.best_a, state.best_phi))
        & (state.best_phi < phi_0)
    )
    return state._replace(
        failed=jnp.where(
            state.failed & best_is_acceptable, _bool_scalar(False), state.failed
        ),
        done=jnp.where(
            state.failed & best_is_acceptable, _bool_scalar(True), state.done
        ),
        a_star=jnp.where(state.failed & best_is_acceptable, state.best_a, state.a_star),
        phi_star=jnp.where(
            state.failed & best_is_acceptable,
            state.best_phi,
            state.phi_star,
        ),
        dphi_star=jnp.where(
            state.failed & best_is_acceptable,
            state.best_dphi,
            state.dphi_star,
        ),
        g_star=jnp.where(state.failed & best_is_acceptable, state.best_g, state.g_star),
    )


def _apply_zoom_branch_result(
    state,
    zoom,
    *,
    wolfe_one,
):
    state = state._replace(
        nfev=state.nfev + zoom.nfev,
        ngev=state.ngev + zoom.ngev,
    )
    improves_best = (
        _line_search_sample_valid(zoom.best_phi, zoom.best_dphi, zoom.best_g)
        & (~wolfe_one(zoom.best_a, zoom.best_phi))
        & (zoom.best_phi < state.best_phi)
    )
    return state._replace(
        best_a=jnp.where(improves_best, zoom.best_a, state.best_a),
        best_phi=jnp.where(improves_best, zoom.best_phi, state.best_phi),
        best_dphi=jnp.where(improves_best, zoom.best_dphi, state.best_dphi),
        best_g=jnp.where(improves_best, zoom.best_g, state.best_g),
        done=_bool_scalar(True),
        failed=zoom.failed | state.failed,
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
    dtype = pk.dtype
    zero = _as_jax_dtype(0.0, dtype)
    one = _as_jax_dtype(1.0, dtype)
    two = _as_jax_dtype(2.0, dtype)
    one_point_01 = _as_jax_dtype(1.01, dtype)
    c1 = _as_jax_dtype(c1, dtype)
    c2 = _as_jax_dtype(c2, dtype)
    maxiter_jax = _int_scalar(maxiter)
    if initial_step_size is None:
        initial_step_value = zero
        use_initial_step_override = _bool_scalar(False)
    else:
        initial_step_value = _as_jax_dtype(initial_step_size, dtype)
        use_initial_step_override = jnp.isfinite(initial_step_value) & (
            initial_step_value > zero
        )

    phi_0 = old_fval
    dphi_0 = jnp.real(_dot(gfk, pk))

    if old_old_fval is not None:
        candidate_start_value = one_point_01 * two * (phi_0 - old_old_fval) / dphi_0
        candidate_start_value = jnp.where(
            candidate_start_value > zero, candidate_start_value, one
        )
        start_value = jnp.where(candidate_start_value > one, one, candidate_start_value)
    else:
        start_value = one
    start_value = jnp.where(
        use_initial_step_override,
        initial_step_value,
        start_value,
    )

    def wolfe_one(a_i, phi_i):
        return phi_i > phi_0 + c1 * a_i * dphi_0

    def wolfe_two(dphi_i):
        return jnp.abs(dphi_i) <= -c2 * dphi_0

    state = _LineSearchState(
        done=_bool_scalar(False),
        failed=_bool_scalar(False),
        i=_int_scalar(1),
        a_i2=zero,
        phi_i2=phi_0,
        dphi_i2=dphi_0,
        g_i2=gfk,
        a_i1=zero,
        phi_i1=phi_0,
        dphi_i1=dphi_0,
        g_i1=gfk,
        best_a=zero,
        best_phi=phi_0,
        best_dphi=dphi_0,
        best_g=gfk,
        nfev=_int_scalar(0),
        ngev=_int_scalar(0),
        a_star=zero,
        phi_star=phi_0,
        dphi_star=dphi_0,
        g_star=gfk,
    )
    _emit_line_search_runtime_debug(
        "search_entry",
        iteration=state.i,
        alpha=start_value,
        phi=phi_0,
        dphi=dphi_0,
    )

    def body(state):
        a_i = jnp.where(state.i == _int_scalar(1), start_value, state.a_i1 * two)

        phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
        _emit_line_search_runtime_debug(
            "trial",
            iteration=state.i,
            alpha=a_i,
            phi=phi_i,
            dphi=dphi_i,
        )
        state = state._replace(
            nfev=state.nfev + _int_scalar(1),
            ngev=state.ngev + _int_scalar(1),
        )
        sample_valid = _line_search_sample_valid(phi_i, dphi_i, g_i)
        improves_best_i = sample_valid & (~wolfe_one(a_i, phi_i)) & (
            phi_i < state.best_phi
        )
        state = state._replace(
            best_a=jnp.where(improves_best_i, a_i, state.best_a),
            best_phi=jnp.where(improves_best_i, phi_i, state.best_phi),
            best_dphi=jnp.where(improves_best_i, dphi_i, state.best_dphi),
            best_g=jnp.where(improves_best_i, g_i, state.best_g),
        )

        star_to_zoom1 = (~sample_valid) | wolfe_one(a_i, phi_i) | (
            (phi_i >= state.phi_i1) & (state.i > _int_scalar(1))
        )
        star_to_i = sample_valid & wolfe_two(dphi_i) & (~star_to_zoom1)
        star_to_zoom2 = (
            sample_valid & (dphi_i >= zero) & (~star_to_zoom1) & (~star_to_i)
        )
        remaining_zoom_budget = jnp.maximum(
            _int_scalar(0),
            maxiter_jax - state.nfev,
        )
        zoom_budget_exhausted = remaining_zoom_budget <= _int_scalar(0)
        state = lax.cond(
            star_to_zoom1,
            lambda current: _apply_zoom_branch_result(
                current,
                _zoom(
                    restricted_func_and_grad,
                    wolfe_one,
                    wolfe_two,
                    phi_0,
                    current.a_i1,
                    current.phi_i1,
                    current.dphi_i1,
                    current.g_i1,
                    a_i,
                    phi_i,
                    dphi_i,
                    g_i,
                    current.i > _int_scalar(1),
                    current.a_i2,
                    current.phi_i2,
                    current.dphi_i2,
                    current.g_i2,
                    remaining_zoom_budget,
                    zoom_budget_exhausted,
                ),
                wolfe_one=wolfe_one,
            ),
            lambda current: lax.cond(
                star_to_i,
                lambda accepted: accepted._replace(
                    done=_bool_scalar(True),
                    a_star=a_i,
                    phi_star=phi_i,
                    dphi_star=dphi_i,
                    g_star=g_i,
                ),
                lambda continued: lax.cond(
                    star_to_zoom2,
                    lambda zoom_state: _apply_zoom_branch_result(
                        zoom_state,
                        _zoom(
                            restricted_func_and_grad,
                            wolfe_one,
                            wolfe_two,
                            phi_0,
                            a_i,
                            phi_i,
                            dphi_i,
                            g_i,
                            zoom_state.a_i1,
                            zoom_state.phi_i1,
                            zoom_state.dphi_i1,
                            zoom_state.g_i1,
                            zoom_state.i > _int_scalar(1),
                            zoom_state.a_i2,
                            zoom_state.phi_i2,
                            zoom_state.dphi_i2,
                            zoom_state.g_i2,
                            remaining_zoom_budget,
                            zoom_budget_exhausted,
                        ),
                        wolfe_one=wolfe_one,
                    ),
                    lambda unchanged: unchanged,
                    operand=continued,
                ),
                operand=current,
            ),
            operand=state,
        )
        return state._replace(
            i=state.i + _int_scalar(1),
            a_i2=state.a_i1,
            phi_i2=state.phi_i1,
            dphi_i2=state.dphi_i1,
            g_i2=state.g_i1,
            a_i1=a_i,
            phi_i1=phi_i,
            dphi_i1=dphi_i,
            g_i1=g_i,
        )

    state = lax.while_loop(
        lambda state: (~state.done) & (state.i <= maxiter_jax) & (~state.failed),
        body,
        state,
    )
    best_is_acceptable = (
        _line_search_sample_valid(state.best_phi, state.best_dphi, state.best_g)
        & (state.best_phi < phi_0)
    )
    state = state._replace(
        failed=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable,
            _bool_scalar(False),
            state.failed,
        ),
        done=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable,
            _bool_scalar(True),
            state.done,
        ),
        a_star=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable,
            state.best_a,
            state.a_star,
        ),
        phi_star=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable,
            state.best_phi,
            state.phi_star,
        ),
        dphi_star=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable,
            state.best_dphi,
            state.dphi_star,
        ),
        g_star=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable,
            state.best_g,
            state.g_star,
        ),
    )
    _emit_line_search_runtime_debug(
        "search_exit",
        iteration=state.i - _int_scalar(1),
        alpha=state.a_star,
        phi=state.phi_star,
        dphi=state.dphi_star,
    )

    status = jnp.where(
        state.failed,
        _int_scalar(1),
        jnp.where(state.i > maxiter_jax, _int_scalar(3), _int_scalar(0)),
    )
    alpha_k = jnp.asarray(state.a_star)
    if jnp.finfo(alpha_k.dtype).bits != 64:
        alpha_k = jnp.where(
            jnp.abs(alpha_k) < _as_jax_dtype(1e-8, alpha_k.dtype),
            jnp.sign(alpha_k) * _as_jax_dtype(1e-8, alpha_k.dtype),
            alpha_k,
        )
    return _LineSearchResults(
        failed=state.failed | (~state.done),
        nit=state.i - _int_scalar(1),
        nfev=state.nfev,
        ngev=state.ngev,
        k=state.i,
        a_k=alpha_k,
        f_k=state.phi_star,
        g_k=state.g_star,
        status=status,
    )


def _line_search(
    f,
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
    xk, pk = _promote_dtypes_inexact(xk, pk)
    old_fval = _as_jax_dtype(old_fval, pk.dtype)
    gfk = _as_jax_dtype(gfk, pk.dtype)
    scalar_value_and_grad = _scalar_value_and_grad(f)

    def restricted_func_and_grad(t):
        t = _as_jax_dtype(t, pk.dtype)
        phi, g = scalar_value_and_grad(xk + t * pk)
        dphi = jnp.real(_dot(g, pk))
        return phi, dphi, g

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


def _line_search_value_and_grad(
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
    xk, pk = _promote_dtypes_inexact(xk, pk)
    old_fval = _as_jax_dtype(old_fval, pk.dtype)
    gfk = _as_jax_dtype(gfk, pk.dtype)

    def restricted_func_and_grad(t):
        t = _as_jax_dtype(t, pk.dtype)
        phi, g = fun(xk + t * pk)
        phi = _as_jax_dtype(phi, pk.dtype)
        g = _as_jax_dtype(g, pk.dtype)
        dphi = jnp.real(_dot(g, pk))
        return phi, dphi, g

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
