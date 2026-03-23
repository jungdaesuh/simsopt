"""
JAX optimizer adapter for the Boozer inner solve.

Reference/oracle methods:
  - ``method="bfgs"``: host-driven SciPy BFGS loop with JAX value/grad.
  - ``method="lbfgs"``: host-driven SciPy L-BFGS-B loop with JAX value/grad.

Transitional private method (validated on JAX 0.9.2):
  - ``method="bfgs-hybrid"``: SciPy BFGS prefix, then JAX on-device BFGS.

Target private methods (validated on JAX 0.9.2):
  - ``method="bfgs-ondevice"``: JAX on-device BFGS.
  - ``method="lbfgs-ondevice"``: JAX on-device L-BFGS.

The private methods intentionally mirror the JAX 0.9.2 optimizer internals so
the line-search and iteration semantics stay stable across this project. The
reference source is the upstream ``jax-v0.9.2`` tag
(``a659757d768587a81d095a9fab5f0c36f8beb218``).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple
import warnings

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import api
from jax._src import dtypes
from jax._src.numpy import linalg as jnp_linalg
from jax._src.numpy.util import promote_dtypes_inexact
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

__all__ = [
    "PRIVATE_OPTIMIZER_JAX_VERSION",
    "VALID_OPTIMIZER_BACKENDS",
    "REFERENCE_OPTIMIZER_BACKENDS",
    "TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS",
    "jax_minimize",
    "newton_polish",
    "newton_exact",
    "require_target_backend_x64",
    "resolve_optimizer_backend_method",
]


PRIVATE_OPTIMIZER_JAX_VERSION = "0.9.2"
VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "hybrid", "ondevice"})
OPTIMIZER_BACKEND_ROLE = {
    "scipy": "reference",
    "hybrid": "transitional",
    "ondevice": "target",
}
TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS = frozenset({"hybrid", "ondevice"})
_SUPPORTED_METHODS = {
    "bfgs",
    "lbfgs",
    "bfgs-hybrid",
    "bfgs-ondevice",
    "lbfgs-ondevice",
}
REFERENCE_OPTIMIZER_BACKENDS = frozenset({"scipy"})
_REFERENCE_METHODS = frozenset({"bfgs", "lbfgs"})

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


class _BFGSResults(NamedTuple):
    converged: bool | jax.Array
    failed: bool | jax.Array
    k: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    nhev: int | jax.Array
    x_k: jax.Array
    f_k: jax.Array
    g_k: jax.Array
    H_k: jax.Array
    old_old_fval: jax.Array
    status: int | jax.Array
    line_search_status: int | jax.Array


class _ZoomState(NamedTuple):
    done: bool | jax.Array
    failed: bool | jax.Array
    j: int | jax.Array
    a_lo: float | jax.Array
    phi_lo: float | jax.Array
    dphi_lo: float | jax.Array
    a_hi: float | jax.Array
    phi_hi: float | jax.Array
    dphi_hi: float | jax.Array
    a_rec: float | jax.Array
    phi_rec: float | jax.Array
    a_star: float | jax.Array
    phi_star: float | jax.Array
    dphi_star: float | jax.Array
    g_star: jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array


class _LineSearchState(NamedTuple):
    done: jax.Array
    failed: jax.Array
    i: int | jax.Array
    a_i1: float | jax.Array
    phi_i1: float | jax.Array
    dphi_i1: float | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    a_star: float | jax.Array
    phi_star: jax.Array
    dphi_star: jax.Array
    g_star: jax.Array


class _LineSearchResults(NamedTuple):
    failed: bool | jax.Array
    nit: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    k: int | jax.Array
    a_k: float | jax.Array
    f_k: jax.Array
    g_k: jax.Array
    status: bool | jax.Array


class _LBFGSResults(NamedTuple):
    converged: jax.Array
    failed: jax.Array
    k: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    x_k: jax.Array
    f_k: jax.Array
    g_k: jax.Array
    s_history: jax.Array
    y_history: jax.Array
    rho_history: jax.Array
    gamma: float | jax.Array
    status: int | jax.Array
    ls_status: int | jax.Array


def _require_private_optimizer_runtime(x0):
    if jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION:
        raise RuntimeError(
            f"On-device optimizer is validated on JAX "
            f"{PRIVATE_OPTIMIZER_JAX_VERSION}; found {jax.__version__}. "
            "Use envs/columbia-jax-0.9.2.yml for the supported runtime or "
            "fall back to optimizer_backend='scipy'."
        )
    if not _x64_enabled():
        raise RuntimeError(
            "On-device optimizer requires jax_enable_x64=True before import/use."
        )

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    if x0.ndim != 1:
        raise ValueError(
            f"On-device optimizer expects a flat 1-D decision vector, got {x0.shape}."
        )
    return x0


def _x64_enabled():
    return bool(jnp.zeros(1).dtype == jnp.float64)


def resolve_optimizer_backend_method(optimizer_backend, *, limited_memory):
    """Map the public backend contract to the concrete optimizer method."""
    if optimizer_backend not in VALID_OPTIMIZER_BACKENDS:
        raise ValueError(
            "optimizer_backend must be one of: scipy, hybrid, ondevice."
        )
    if optimizer_backend == "scipy":
        return "lbfgs" if limited_memory else "bfgs"
    if optimizer_backend == "hybrid":
        if limited_memory:
            raise ValueError(
                "optimizer_backend='hybrid' is transitional and does not support "
                "limited_memory=True."
            )
        return "bfgs-hybrid"
    return "lbfgs-ondevice" if limited_memory else "bfgs-ondevice"


def require_target_backend_x64(optimizer_backend):
    """Fail fast when a target-lane backend is requested without float64."""
    if optimizer_backend not in TARGET_X64_REQUIRED_OPTIMIZER_BACKENDS:
        return
    if _x64_enabled():
        return
    role = OPTIMIZER_BACKEND_ROLE[optimizer_backend]
    raise RuntimeError(
        f"optimizer_backend='{optimizer_backend}' ({role}) requires "
        "jax_enable_x64=True before import/use."
    )


def _strip_internal_options(options, method):
    if not options:
        return {}
    internal = {"hybrid_scipy_maxiter", "line_search_maxiter"}
    if method == "bfgs":
        internal |= {"maxcor", "ftol", "maxfun", "maxgrad", "maxls"}
    elif method == "lbfgs":
        internal |= {"maxgrad"}
    return {key: value for key, value in options.items() if key not in internal}


def _normalize_scipy_result(result):
    result.x = jnp.asarray(result.x)
    result.jac = jnp.asarray(result.jac)
    result.nit = int(getattr(result, "nit", 0))
    result.nfev = int(getattr(result, "nfev", 0))
    if hasattr(result, "njev"):
        result.njev = int(result.njev)
    result.success = bool(result.success)
    if hasattr(result, "status"):
        result.status = int(result.status)
    return result


def _status_message_bfgs(status, invalid_state):
    if invalid_state:
        return "Optimization failed with non-finite objective or gradient."
    messages = {
        0: "Optimization terminated successfully.",
        1: "Maximum number of iterations reached.",
        3: "Line search zoom failed.",
        5: "Line search reached its iteration limit.",
        -1: "Optimization failed.",
    }
    return messages.get(status, f"Optimization failed with status {status}.")


def _status_message_lbfgs(status, invalid_state):
    if invalid_state:
        return "Optimization failed with non-finite objective or gradient."
    messages = {
        0: "Optimization terminated successfully.",
        1: "Maximum number of iterations reached.",
        2: "Maximum number of function evaluations reached.",
        3: "Maximum number of gradient evaluations reached.",
        4: "Insufficient progress (ftol).",
        5: "Line search failed.",
    }
    return messages.get(status, f"Optimization failed with status {status}.")


def _private_bfgs_result_to_optimize_result(state, *, total_nit=None):
    invalid_state = not bool(
        np.isfinite(np.asarray(state.f_k))
        and np.all(np.isfinite(np.asarray(state.g_k)))
    )
    status = int(state.status)
    nit = int(state.k if total_nit is None else total_nit)
    return OptimizeResult(
        x=jnp.asarray(state.x_k),
        fun=float(np.asarray(state.f_k)),
        jac=jnp.asarray(state.g_k),
        nit=nit,
        nfev=int(state.nfev),
        njev=int(state.ngev),
        nhev=int(state.nhev),
        success=bool(state.converged) and not invalid_state,
        status=status,
        message=_status_message_bfgs(status, invalid_state),
        hess_inv=jnp.asarray(state.H_k),
        line_search_status=int(state.line_search_status),
    )


def _private_lbfgs_result_to_optimize_result(state):
    invalid_state = not bool(
        np.isfinite(np.asarray(state.f_k))
        and np.all(np.isfinite(np.asarray(state.g_k)))
    )
    status = int(state.status)
    return OptimizeResult(
        x=jnp.asarray(state.x_k),
        fun=float(np.asarray(state.f_k)),
        jac=jnp.asarray(state.g_k),
        nit=int(state.k),
        nfev=int(state.nfev),
        njev=int(state.ngev),
        success=bool(state.converged) and not invalid_state,
        status=status,
        message=_status_message_lbfgs(status, invalid_state),
        ls_status=int(state.ls_status),
    )


def _coerce_dense_hess_inv(hess_inv, n, dtype):
    if hess_inv is None:
        warnings.warn(
            "Hybrid BFGS continuation received no dense hess_inv; falling back to "
            "identity warm start.",
            RuntimeWarning,
            stacklevel=3,
        )
        return jnp.eye(n, dtype=dtype)
    try:
        dense = np.asarray(hess_inv)
    except Exception:
        warnings.warn(
            "Hybrid BFGS continuation could not densify hess_inv; falling back to "
            "identity warm start.",
            RuntimeWarning,
            stacklevel=3,
        )
        return jnp.eye(n, dtype=dtype)
    if dense.ndim != 2 or dense.shape != (n, n):
        warnings.warn(
            "Hybrid BFGS continuation received mismatched hess_inv shape; "
            "falling back to identity warm start.",
            RuntimeWarning,
            stacklevel=3,
        )
        return jnp.eye(n, dtype=dtype)
    return jnp.asarray(dense, dtype=dtype)


def _scipy_result_is_continuable(result):
    return (
        np.isfinite(getattr(result, "fun", np.nan))
        and np.all(np.isfinite(np.asarray(result.x)))
        and np.all(np.isfinite(np.asarray(result.jac)))
    )


def _hessian_vector_product_fn(objective_fn):
    grad_fn = jax.grad(objective_fn)
    return jax.jit(lambda x, v: jax.jvp(grad_fn, (x,), (v,))[1])


def _materialize_dense_hessian(hvp_fn, x):
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    cols = lax.map(lambda basis: hvp_fn(x, basis), eye)
    dense = jnp.swapaxes(cols, 0, 1)
    return 0.5 * (dense + dense.T)


def _gmres_solve_newton_system(hvp_fn, x, rhs, *, stab, tol):
    n = int(rhs.shape[0])
    restart = max(5, min(n, 50))
    maxiter = max(10, min(4 * n, 200))

    def matvec(v):
        Hv = hvp_fn(x, v)
        if stab != 0.0:
            Hv = Hv + stab * v
        return Hv

    dx, _ = gmres(
        matvec,
        rhs,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
    )
    residual = rhs - matvec(dx)
    return dx, residual, matvec


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    dtype = dtypes.result_type(a, fa, fpa, b, fb, c, fc)
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
    a_lo,
    phi_lo,
    dphi_lo,
    a_hi,
    phi_hi,
    dphi_hi,
    g_0,
    pass_through,
):
    state = _ZoomState(
        done=False,
        failed=False,
        j=0,
        a_lo=a_lo,
        phi_lo=phi_lo,
        dphi_lo=dphi_lo,
        a_hi=a_hi,
        phi_hi=phi_hi,
        dphi_hi=dphi_hi,
        a_rec=(a_lo + a_hi) / 2.0,
        phi_rec=(phi_lo + phi_hi) / 2.0,
        a_star=1.0,
        phi_star=phi_lo,
        dphi_star=dphi_lo,
        g_star=g_0,
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

        threshold = jnp.where((dtypes.finfo(dalpha.dtype).bits < 64), 1e-5, 1e-10)
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
                dict(a_lo=a_j, phi_lo=phi_j, dphi_lo=dphi_j),
            )
        )
        state = state._replace(j=state.j + 1)
        state = state._replace(failed=state.failed | (state.j >= 30))
        return state

    return lax.while_loop(
        lambda state: (~state.done) & (~pass_through) & (~state.failed),
        body,
        state,
    )


def _line_search(
    f, xk, pk, old_fval=None, old_old_fval=None, gfk=None, c1=1e-4, c2=0.9, maxiter=20
):
    xk, pk = promote_dtypes_inexact(xk, pk)

    def restricted_func_and_grad(t):
        t = jnp.array(t, dtype=pk.dtype)
        phi, g = api.value_and_grad(f)(xk + t * pk)
        dphi = jnp.real(_dot(g, pk))
        return phi, dphi, g

    if old_fval is None or gfk is None:
        phi_0, dphi_0, gfk = restricted_func_and_grad(0)
    else:
        phi_0 = old_fval
        dphi_0 = jnp.real(_dot(gfk, pk))

    if old_old_fval is not None:
        # Upstream line_search.py says old_old_fval is "unused", but the
        # actual start-value heuristic still consumes it.
        # The hybrid handoff must seed this value across the SciPy->JAX
        # handoff seam to preserve the upstream starting-step heuristic; the
        # positive clamp below is a later-iteration backstop for mid-loop
        # objective increase on the previous accepted step.
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

        star_to_zoom1 = wolfe_one(a_i, phi_i) | (
            (phi_i >= state.phi_i1) & (state.i > 1)
        )
        star_to_i = wolfe_two(dphi_i) & (~star_to_zoom1)
        star_to_zoom2 = (dphi_i >= 0.0) & (~star_to_zoom1) & (~star_to_i)

        zoom1 = _zoom(
            restricted_func_and_grad,
            wolfe_one,
            wolfe_two,
            state.a_i1,
            state.phi_i1,
            state.dphi_i1,
            a_i,
            phi_i,
            dphi_i,
            gfk,
            ~star_to_zoom1,
        )
        state = state._replace(
            nfev=state.nfev + zoom1.nfev, ngev=state.ngev + zoom1.ngev
        )

        zoom2 = _zoom(
            restricted_func_and_grad,
            wolfe_one,
            wolfe_two,
            a_i,
            phi_i,
            dphi_i,
            state.a_i1,
            state.phi_i1,
            state.dphi_i1,
            gfk,
            ~star_to_zoom2,
        )
        state = state._replace(
            nfev=state.nfev + zoom2.nfev, ngev=state.ngev + zoom2.ngev
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
        return state._replace(i=state.i + 1, a_i1=a_i, phi_i1=phi_i, dphi_i1=dphi_i)

    state = lax.while_loop(
        lambda state: (~state.done) & (state.i <= maxiter) & (~state.failed),
        body,
        state,
    )

    status = jnp.where(
        state.failed,
        jnp.array(1),
        jnp.where(state.i > maxiter, jnp.array(3), jnp.array(0)),
    )
    alpha_k = jnp.asarray(state.a_star)
    alpha_k = jnp.where(
        (dtypes.finfo(alpha_k.dtype).bits != 64) & (jnp.abs(alpha_k) < 1e-8),
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


def _minimize_bfgs_private(
    fun,
    x0,
    *,
    maxiter=None,
    norm=np.inf,
    gtol=1e-5,
    line_search_maxiter=10,
    initial_state=None,
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
        state = state._replace(
            nfev=state.nfev + line_search_results.nfev,
            ngev=state.ngev + line_search_results.ngev,
            failed=line_search_results.failed,
            line_search_status=line_search_results.status,
        )
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

        return state._replace(
            converged=converged,
            k=state.k + 1,
            x_k=x_kp1,
            f_k=f_kp1,
            g_k=g_kp1,
            H_k=H_kp1,
            old_old_fval=state.f_k,
        )

    state = lax.while_loop(cond_fun, body_fun, state)
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
):
    x0 = _require_private_optimizer_runtime(x0)
    d = len(x0)
    dtype = dtypes.dtype(x0)

    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
    if maxiter is None:
        maxiter = np.inf
    if maxfun is None:
        maxfun = np.inf
    if maxgrad is None:
        maxgrad = np.inf

    f_0, g_0 = api.value_and_grad(fun)(x0)
    state_initial = _LBFGSResults(
        converged=jnp_linalg.norm(g_0, ord=norm) < gtol,
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
        ls_results = _line_search(
            f=fun,
            xk=state.x_k,
            pk=p_k,
            old_fval=state.f_k,
            gfk=state.g_k,
            maxiter=maxls,
        )

        s_k = jnp.asarray(ls_results.a_k).astype(p_k.dtype) * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        rho_k = jnp.reciprocal(rho_k_inv).astype(y_k.dtype)
        gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))

        next_k = state.k + 1
        next_nfev = state.nfev + ls_results.nfev
        next_ngev = state.ngev + ls_results.ngev

        status = jnp.array(0)
        status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
        status = jnp.where(next_ngev >= maxgrad, 3, status)
        status = jnp.where(next_nfev >= maxfun, 2, status)
        status = jnp.where(next_k >= maxiter, 1, status)
        status = jnp.where(ls_results.failed, 5, status)

        converged = jnp_linalg.norm(g_kp1, ord=norm) < gtol
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

    return lax.while_loop(cond_fun, body_fun, state_initial)


def _make_bfgs_continuation_state(result, *, gtol, norm):
    x_k = jnp.asarray(result.x, dtype=jnp.float64)
    f_k = jnp.asarray(result.fun, dtype=x_k.dtype)
    g_k = jnp.asarray(result.jac, dtype=x_k.dtype)
    H_k = _coerce_dense_hess_inv(
        getattr(result, "hess_inv", None), x_k.shape[0], x_k.dtype
    )

    # Warm-started H_k must still produce a descent direction.
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


def _scipy_minimize(fun, x0, *, method, tol, maxiter, options):
    val_and_grad_fn = jax.jit(jax.value_and_grad(fun))

    def scipy_fun(x_np):
        x_jax = jnp.asarray(x_np)
        val, grad = val_and_grad_fn(x_jax)
        return float(val), np.asarray(grad)

    stripped_options = _strip_internal_options(options, method)
    if method == "bfgs":
        scipy_method = "BFGS"
        scipy_opts = {"maxiter": maxiter, "gtol": tol, **stripped_options}
    else:
        scipy_method = "L-BFGS-B"
        scipy_opts = {
            "maxiter": maxiter,
            "gtol": tol,
            "maxcor": 200,
            **stripped_options,
        }

    return _normalize_scipy_result(
        scipy_minimize(
            scipy_fun,
            np.asarray(x0),
            jac=True,
            method=scipy_method,
            options=scipy_opts,
        )
    )


def _scipy_minimize_value_and_grad(fun, x0, *, method, tol, maxiter, options):
    def scipy_fun(x_np):
        val, grad = fun(np.asarray(x_np))
        return float(val), np.asarray(grad, dtype=float)

    stripped_options = _strip_internal_options(options, method)
    if method == "bfgs":
        scipy_method = "BFGS"
        scipy_opts = {"maxiter": maxiter, "gtol": tol, **stripped_options}
    else:
        scipy_method = "L-BFGS-B"
        scipy_opts = {
            "maxiter": maxiter,
            "gtol": tol,
            "maxcor": 200,
            **stripped_options,
        }

    return _normalize_scipy_result(
        scipy_minimize(
            scipy_fun,
            np.asarray(x0),
            jac=True,
            method=scipy_method,
            options=scipy_opts,
        )
    )


def jax_minimize(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
):
    """Optimizer adapter for Boozer LS minimization.

    Contract by method family:

    - ``bfgs`` / ``lbfgs``:
      trusted reference/oracle path using host-side SciPy loops.
    - ``bfgs-hybrid``:
      transitional private path for staged migration away from SciPy.
    - ``bfgs-ondevice`` / ``lbfgs-ondevice``:
      private target path for the full-GPU optimizer lane.

    If ``value_and_grad=True``, ``fun`` must return ``(value, grad)`` directly.
    That explicit value/gradient contract is currently supported only on the
    trusted SciPy reference methods.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Supported: {sorted(_SUPPORTED_METHODS)}."
        )

    options = dict(options or {})
    if method in _REFERENCE_METHODS:
        scipy_adapter = (
            _scipy_minimize_value_and_grad if value_and_grad else _scipy_minimize
        )
        return scipy_adapter(
            fun,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
        )
    if value_and_grad:
        raise RuntimeError(
            "Explicit value-and-gradient objectives are only supported on the "
            "trusted SciPy reference methods today."
        )

    if method == "bfgs-ondevice":
        state = _minimize_bfgs_private(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            line_search_maxiter=int(options.get("line_search_maxiter", 10)),
        )
        return _private_bfgs_result_to_optimize_result(state)

    if method == "lbfgs-ondevice":
        # ftol=0 keeps the private L-BFGS loop in the "stop on actual
        # objective increase" regime instead of treating ftol as disabled.
        state = _minimize_lbfgs_private(
            fun,
            x0,
            maxiter=maxiter,
            gtol=tol,
            maxcor=int(options.get("maxcor", 200)),
            ftol=float(options.get("ftol", 0.0)),
            maxfun=options.get("maxfun"),
            maxgrad=options.get("maxgrad"),
            maxls=int(options.get("maxls", 20)),
        )
        return _private_lbfgs_result_to_optimize_result(state)

    total_maxiter = int(maxiter)
    prefix_cap = int(options.get("hybrid_scipy_maxiter", min(total_maxiter // 2, 100)))
    prefix_cap = max(0, min(prefix_cap, total_maxiter - 1))
    prefix_result = _scipy_minimize(
        fun,
        x0,
        method="bfgs",
        tol=tol,
        maxiter=prefix_cap,
        options=options,
    )
    if prefix_result.success:
        return prefix_result
    if not _scipy_result_is_continuable(prefix_result):
        prefix_result.success = False
        prefix_result.message = (
            "SciPy prefix produced a non-finite state; on-device continuation skipped."
        )
        return prefix_result

    remaining_maxiter = max(0, total_maxiter - int(prefix_result.nit))
    if remaining_maxiter == 0:
        prefix_result.success = False
        return prefix_result

    continuation_state = _make_bfgs_continuation_state(
        prefix_result,
        gtol=tol,
        norm=np.inf,
    )
    final_state = _minimize_bfgs_private(
        fun,
        prefix_result.x,
        maxiter=remaining_maxiter,
        gtol=tol,
        line_search_maxiter=int(options.get("line_search_maxiter", 10)),
        initial_state=continuation_state,
    )
    result = _private_bfgs_result_to_optimize_result(
        final_state,
        total_nit=int(prefix_result.nit) + int(final_state.k),
    )
    result.hess_inv = jnp.asarray(final_state.H_k)
    return result


def newton_polish(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
):
    """Newton polish using exact Hessian-vector products.

    Iterations solve the Newton system with GMRES against the exact
    Hessian linear operator, avoiding the peak memory cost of
    ``jax.hessian(objective_fn)`` on large Boozer LS problems.

    The dense Hessian is still materialized once at the final iterate so
    callers retain the existing adjoint/PLU contract.
    """
    val_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))
    hvp_fn = _hessian_vector_product_fn(objective_fn)

    x = x0
    val, grad = val_and_grad_fn(x)
    norm = jnp.linalg.norm(grad)

    nit = 0
    while nit < maxiter and float(norm) > tol:
        linear_tol = min(1e-10, max(float(tol) * 0.1, 1e-14))
        dx, linear_residual, _ = _gmres_solve_newton_system(
            hvp_fn,
            x,
            grad,
            stab=stab,
            tol=linear_tol,
        )
        if not np.all(np.isfinite(np.asarray(dx))) or (
            np.linalg.norm(np.asarray(linear_residual))
            > max(1e-10, 1e-3 * float(norm))
        ):
            H_solve = _materialize_dense_hessian(hvp_fn, x)
            if stab != 0.0:
                H_solve = H_solve + stab * jnp.eye(H_solve.shape[0], dtype=H_solve.dtype)
            dx = jnp.linalg.solve(H_solve, grad)
            linear_residual = grad - H_solve @ dx
        if float(norm) < 1e-9:
            correction, _, _ = _gmres_solve_newton_system(
                hvp_fn,
                x,
                linear_residual,
                stab=stab,
                tol=linear_tol,
            )
            if np.all(np.isfinite(np.asarray(correction))):
                dx = dx + correction
        x = x - dx
        val, grad = val_and_grad_fn(x)
        norm = jnp.linalg.norm(grad)
        nit += 1

    H = _materialize_dense_hessian(hvp_fn, x)

    return {
        "x": x,
        "fun": val,
        "grad": grad,
        "hessian": H,
        "nit": nit,
        "success": bool(float(norm) <= tol),
    }


def newton_exact(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
):
    """Newton solver for the exact Boozer residual system ``r(x) = 0``."""
    jac_fn = jax.jit(jax.jacfwd(residual_fn))
    res_fn = jax.jit(residual_fn)

    x = x0
    r = res_fn(x)
    J = jac_fn(x)
    norm = jnp.linalg.norm(r)

    nit = 0
    while nit < maxiter and float(norm) > tol:
        dx = jnp.linalg.solve(J, r)
        dx = dx + jnp.linalg.solve(J, r - J @ dx)
        x = x - dx
        r = res_fn(x)
        J = jac_fn(x)
        norm = jnp.linalg.norm(r)
        nit += 1

    return {
        "x": x,
        "residual": r,
        "jacobian": J,
        "nit": nit,
        "success": bool(float(norm) <= tol),
    }
