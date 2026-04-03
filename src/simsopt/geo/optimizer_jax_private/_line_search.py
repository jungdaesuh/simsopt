"""Strong Wolfe line search for on-device BFGS / L-BFGS.

Mirrors the JAX 0.9.2 line-search semantics so
the line-search semantics stay stable across this project.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax, value_and_grad

from ._common import _dot, _promote_dtypes_inexact
from ._types import _LineSearchResults, _LineSearchState, _ZoomState


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    dtype = jnp.result_type(a, fa, fpa, b, fb, c, fc)
    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]], dtype=dtype)
    d2 = jnp.array([fb - fa - C * db, fc - fa - C * dc], dtype=dtype)
    A, B = _dot(d1, d2) / denom

    radical = B * B - 3.0 * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db**2)
    xmin = a - C / (2.0 * B)
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
    pass_through,
):
    lo_is_better = phi_lo <= phi_hi
    state = _ZoomState(
        done=False,
        failed=False,
        j=0,
        a_lo=a_lo,
        phi_lo=phi_lo,
        dphi_lo=dphi_lo,
        g_lo=g_lo,
        a_hi=a_hi,
        phi_hi=phi_hi,
        dphi_hi=dphi_hi,
        g_hi=g_hi,
        a_rec=(a_lo + a_hi) / 2.0,
        phi_rec=(phi_lo + phi_hi) / 2.0,
        a_star=1.0,
        phi_star=phi_lo,
        dphi_star=dphi_lo,
        g_star=g_lo,
        best_a=jnp.where(lo_is_better, a_lo, a_hi),
        best_phi=jnp.where(lo_is_better, phi_lo, phi_hi),
        best_dphi=jnp.where(lo_is_better, dphi_lo, dphi_hi),
        best_g=jnp.where(lo_is_better, g_lo, g_hi),
        nfev=0,
        ngev=0,
    )
    delta1 = 0.2
    delta2 = 0.1

    def body(state):
        dalpha = jnp.abs(state.a_hi - state.a_lo)
        a = jnp.minimum(state.a_hi, state.a_lo)
        b = jnp.maximum(state.a_hi, state.a_lo)
        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        threshold = jnp.where((jnp.finfo(dalpha.dtype).bits < 64), 1e-5, 1e-10)
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
        use_cubic = (state.j > 0) & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
        a_j_quad = _quadmin(
            state.a_lo,
            state.phi_lo,
            state.dphi_lo,
            state.a_hi,
            state.phi_hi,
        )
        use_quad = (~use_cubic) & (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
        a_j_bisection = (state.a_lo + state.a_hi) / 2.0

        a_j = jnp.where(use_cubic, a_j_cubic, state.a_rec)
        a_j = jnp.where(use_quad, a_j_quad, a_j)
        a_j = jnp.where((~use_cubic) & (~use_quad), a_j_bisection, a_j)

        phi_j, dphi_j, g_j = restricted_func_and_grad(a_j)
        phi_j = phi_j.astype(state.phi_lo.dtype)
        dphi_j = dphi_j.astype(state.dphi_lo.dtype)
        g_j = g_j.astype(state.g_star.dtype)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
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
            (dphi_j * (state.a_hi - state.a_lo) >= 0.0) & (~hi_to_j) & (~star_to_j)
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
        state = state._replace(j=state.j + 1)
        state = state._replace(failed=state.failed | (state.j >= 30))
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
        failed=jnp.where(state.failed & best_is_acceptable, False, state.failed),
        done=jnp.where(state.failed & best_is_acceptable, True, state.done),
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
    if old_fval is None or gfk is None:
        phi_0, dphi_0, gfk = restricted_func_and_grad(0)
    else:
        phi_0 = old_fval
        dphi_0 = jnp.real(_dot(gfk, pk))

    if old_old_fval is not None:
        candidate_start_value = 1.01 * 2 * (phi_0 - old_old_fval) / dphi_0
        candidate_start_value = jnp.where(
            candidate_start_value > 0, candidate_start_value, 1.0
        )
        start_value = jnp.where(candidate_start_value > 1, 1.0, candidate_start_value)
    else:
        start_value = 1

    def wolfe_one(a_i, phi_i):
        return phi_i > phi_0 + c1 * a_i * dphi_0

    def wolfe_two(dphi_i):
        return jnp.abs(dphi_i) <= -c2 * dphi_0

    state = _LineSearchState(
        done=jnp.array(False, dtype=bool),
        failed=jnp.array(False, dtype=bool),
        i=1,
        a_i1=0.0,
        phi_i1=phi_0,
        dphi_i1=dphi_0,
        g_i1=gfk,
        best_a=0.0,
        best_phi=phi_0,
        best_dphi=dphi_0,
        best_g=gfk,
        nfev=1 if (old_fval is None or gfk is None) else 0,
        ngev=1 if (old_fval is None or gfk is None) else 0,
        a_star=0.0,
        phi_star=phi_0,
        dphi_star=dphi_0,
        g_star=gfk,
    )

    def body(state):
        a_i = jnp.where(state.i == 1, start_value, state.a_i1 * 2.0)

        phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
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
            (phi_i >= state.phi_i1) & (state.i > 1)
        )
        star_to_i = wolfe_two(dphi_i) & (~star_to_zoom1)
        star_to_zoom2 = (dphi_i >= 0.0) & (~star_to_zoom1) & (~star_to_i)

        zoom1 = _zoom(
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
            ~star_to_zoom1,
        )
        state = state._replace(
            nfev=state.nfev + zoom1.nfev, ngev=state.ngev + zoom1.ngev
        )
        improves_best_zoom1 = (
            jnp.isfinite(zoom1.best_phi)
            & (~wolfe_one(zoom1.best_a, zoom1.best_phi))
            & (zoom1.best_phi < state.best_phi)
        )
        state = state._replace(
            best_a=jnp.where(improves_best_zoom1, zoom1.best_a, state.best_a),
            best_phi=jnp.where(improves_best_zoom1, zoom1.best_phi, state.best_phi),
            best_dphi=jnp.where(improves_best_zoom1, zoom1.best_dphi, state.best_dphi),
            best_g=jnp.where(improves_best_zoom1, zoom1.best_g, state.best_g),
        )

        zoom2 = _zoom(
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
            ~star_to_zoom2,
        )
        state = state._replace(
            nfev=state.nfev + zoom2.nfev, ngev=state.ngev + zoom2.ngev
        )
        improves_best_zoom2 = (
            jnp.isfinite(zoom2.best_phi)
            & (~wolfe_one(zoom2.best_a, zoom2.best_phi))
            & (zoom2.best_phi < state.best_phi)
        )
        state = state._replace(
            best_a=jnp.where(improves_best_zoom2, zoom2.best_a, state.best_a),
            best_phi=jnp.where(improves_best_zoom2, zoom2.best_phi, state.best_phi),
            best_dphi=jnp.where(improves_best_zoom2, zoom2.best_dphi, state.best_dphi),
            best_g=jnp.where(improves_best_zoom2, zoom2.best_g, state.best_g),
        )

        state = state._replace(
            done=star_to_zoom1 | state.done,
            failed=(star_to_zoom1 & zoom1.failed) | state.failed,
            **_binary_replace(
                star_to_zoom1,
                state._asdict(),
                zoom1._asdict(),
                keys=["a_star", "phi_star", "dphi_star", "g_star"],
            ),
        )
        state = state._replace(
            done=star_to_i | state.done,
            **_binary_replace(
                star_to_i,
                state._asdict(),
                dict(a_star=a_i, phi_star=phi_i, dphi_star=dphi_i, g_star=g_i),
            ),
        )
        state = state._replace(
            done=star_to_zoom2 | state.done,
            failed=(star_to_zoom2 & zoom2.failed) | state.failed,
            **_binary_replace(
                star_to_zoom2,
                state._asdict(),
                zoom2._asdict(),
                keys=["a_star", "phi_star", "dphi_star", "g_star"],
            ),
        )
        return state._replace(
            i=state.i + 1,
            a_i1=a_i,
            phi_i1=phi_i,
            dphi_i1=dphi_i,
            g_i1=g_i,
        )

    state = lax.while_loop(
        lambda state: (~state.done) & (state.i <= maxiter) & (~state.failed),
        body,
        state,
    )
    best_is_acceptable = jnp.isfinite(state.best_phi) & (state.best_phi < phi_0)
    state = state._replace(
        failed=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable, False, state.failed
        ),
        done=jnp.where(
            (state.failed | (~state.done)) & best_is_acceptable, True, state.done
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
        jnp.array(1),
        jnp.where(state.i > maxiter, jnp.array(3), jnp.array(0)),
    )
    alpha_k = jnp.asarray(state.a_star)
    alpha_k = jnp.where(
        (jnp.finfo(alpha_k.dtype).bits != 64) & (jnp.abs(alpha_k) < 1e-8),
        jnp.sign(alpha_k) * 1e-8,
        alpha_k,
    )
    return _LineSearchResults(
        failed=state.failed | (~state.done),
        nit=state.i - 1,
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

    def restricted_func_and_grad(t):
        t = jnp.array(t, dtype=pk.dtype)
        phi, g = value_and_grad(f)(xk + t * pk)
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
        t = jnp.array(t, dtype=pk.dtype)
        phi, g = fun(xk + t * pk)
        phi = jnp.asarray(phi, dtype=pk.dtype)
        g = jnp.asarray(g, dtype=pk.dtype)
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
