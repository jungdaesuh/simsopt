"""Strong Wolfe line search for on-device BFGS / L-BFGS.

Mirrors the JAX 0.9.2 line-search semantics so
the line-search semantics stay stable across this project.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from ._common import (
    _as_jax_dtype,
    _bool_scalar,
    _dot,
    _int_scalar,
    _promote_dtypes_inexact,
    _scalar_value_and_grad,
)
from ._types import _LineSearchResults, _LineSearchState, _ZoomState


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


def _binary_replace(replace_bit, original_dict, new_dict, keys=None):
    if keys is None:
        keys = new_dict.keys()
    return {
        key: jnp.where(replace_bit, new_dict[key], original_dict[key]) for key in keys
    }


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
    lo_is_better = phi_lo <= phi_hi
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
        a_rec=(a_lo + a_hi) * half,
        phi_rec=(phi_lo + phi_hi) * half,
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
            (state.j > _int_scalar(0)) & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
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

        phi_j, dphi_j, g_j = restricted_func_and_grad(a_j)
        phi_j = phi_j.astype(state.phi_lo.dtype)
        dphi_j = dphi_j.astype(state.dphi_lo.dtype)
        g_j = g_j.astype(state.g_star.dtype)
        state = state._replace(
            nfev=state.nfev + _int_scalar(1),
            ngev=state.ngev + _int_scalar(1),
        )
        improves_best = jnp.isfinite(phi_j) & (phi_j < state.best_phi)
        state = state._replace(
            best_a=jnp.where(improves_best, a_j, state.best_a),
            best_phi=jnp.where(improves_best, phi_j, state.best_phi),
            best_dphi=jnp.where(improves_best, dphi_j, state.best_dphi),
            best_g=jnp.where(improves_best, g_j, state.best_g),
        )

        hi_to_j = wolfe_one(a_j, phi_j) | (phi_j >= state.phi_lo)
        star_to_j = wolfe_two(dphi_j) & (~hi_to_j)
        hi_to_lo = (
            (dphi_j * (state.a_hi - state.a_lo) >= zero) & (~hi_to_j) & (~star_to_j)
        )
        lo_to_j = (~hi_to_j) & (~star_to_j)

        state = state._replace(
            **_binary_replace(
                hi_to_j,
                state._asdict(),
                dict(
                    a_hi=a_j,
                    phi_hi=phi_j,
                    dphi_hi=dphi_j,
                    g_hi=g_j,
                    a_rec=state.a_hi,
                    phi_rec=state.phi_hi,
                ),
            )
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
                    a_rec=state.a_hi,
                    phi_rec=state.phi_hi,
                ),
            )
        )
        state = state._replace(
            **_binary_replace(
                lo_to_j & ~hi_to_lo,
                state._asdict(),
                dict(a_rec=state.a_lo, phi_rec=state.phi_lo),
            )
        )
        state = state._replace(
            **_binary_replace(
                lo_to_j,
                state._asdict(),
                dict(a_lo=a_j, phi_lo=phi_j, dphi_lo=dphi_j, g_lo=g_j),
            )
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
        jnp.isfinite(state.best_phi)
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
        jnp.isfinite(zoom.best_phi)
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
    old_fval=None,
    old_old_fval=None,
    gfk=None,
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

    if old_fval is None or gfk is None:
        phi_0, dphi_0, gfk = restricted_func_and_grad(zero)
    else:
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

    def wolfe_one(a_i, phi_i):
        return phi_i > phi_0 + c1 * a_i * dphi_0

    def wolfe_two(dphi_i):
        return jnp.abs(dphi_i) <= -c2 * dphi_0

    state = _LineSearchState(
        done=_bool_scalar(False),
        failed=_bool_scalar(False),
        i=_int_scalar(1),
        a_i1=zero,
        phi_i1=phi_0,
        dphi_i1=dphi_0,
        g_i1=gfk,
        best_a=zero,
        best_phi=phi_0,
        best_dphi=dphi_0,
        best_g=gfk,
        nfev=_int_scalar(1 if (old_fval is None or gfk is None) else 0),
        ngev=_int_scalar(1 if (old_fval is None or gfk is None) else 0),
        a_star=zero,
        phi_star=phi_0,
        dphi_star=dphi_0,
        g_star=gfk,
    )

    def body(state):
        a_i = jnp.where(state.i == _int_scalar(1), start_value, state.a_i1 * two)

        phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
        state = state._replace(
            nfev=state.nfev + _int_scalar(1),
            ngev=state.ngev + _int_scalar(1),
        )
        improves_best_i = (
            jnp.isfinite(phi_i) & (~wolfe_one(a_i, phi_i)) & (phi_i < state.best_phi)
        )
        state = state._replace(
            best_a=jnp.where(improves_best_i, a_i, state.best_a),
            best_phi=jnp.where(improves_best_i, phi_i, state.best_phi),
            best_dphi=jnp.where(improves_best_i, dphi_i, state.best_dphi),
            best_g=jnp.where(improves_best_i, g_i, state.best_g),
        )

        star_to_zoom1 = wolfe_one(a_i, phi_i) | (
            (phi_i >= state.phi_i1) & (state.i > _int_scalar(1))
        )
        star_to_i = wolfe_two(dphi_i) & (~star_to_zoom1)
        star_to_zoom2 = (dphi_i >= zero) & (~star_to_zoom1) & (~star_to_i)
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
    best_is_acceptable = jnp.isfinite(state.best_phi) & (state.best_phi < phi_0)
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
    f, xk, pk, old_fval=None, old_old_fval=None, gfk=None, c1=1e-4, c2=0.9, maxiter=20
):
    xk, pk = _promote_dtypes_inexact(xk, pk)
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
        c1=c1,
        c2=c2,
        maxiter=maxiter,
    )


def _line_search_value_and_grad(
    fun,
    xk,
    pk,
    old_fval=None,
    old_old_fval=None,
    gfk=None,
    c1=1e-4,
    c2=0.9,
    maxiter=20,
):
    xk, pk = _promote_dtypes_inexact(xk, pk)

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
        c1=c1,
        c2=c2,
        maxiter=maxiter,
    )
