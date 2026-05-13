"""Shared utilities for private optimizer submodules.

PRIVATE_OPTIMIZER_JAX_VERSION and _x64_enabled are imported from the public
module (``optimizer_jax``).  This is safe because both are defined *before*
the public module's ``try: from .optimizer_jax_private import ...`` block,
so they exist on the partially-initialized module object when this file loads.
"""

from __future__ import annotations

from functools import partial
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..._core.jax_host_boundary import host_array as _callback_host_array
from ..optimizer_jax import (
    PRIVATE_OPTIMIZER_JAX_VERSION,
    _CACHEABLE_VALUE_AND_GRAD_ATTR,
    _x64_enabled,
    private_optimizer_runtime_is_supported,
)

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)
_INT32_COUNTER_MAX = np.iinfo(np.int32).max
_PRIVATE_SOLVER_CACHE_LOCK = Lock()
_PRIVATE_SOLVER_CACHE_ATTR = "_simsopt_cached_private_solver"


def _as_jax_dtype(value, dtype):
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))
    return jnp.asarray(value, dtype=dtype)


def _as_numpy_dtype(value, dtype):
    return np.asarray(value, dtype=np.dtype(dtype))


def _eye(n, dtype):
    return jax.device_put(np.eye(int(n), dtype=np.dtype(dtype)))


def _zeros(shape, dtype):
    if np.isscalar(shape):
        shape = (int(shape),)
    else:
        shape = tuple(int(dim) for dim in shape)
    return jax.device_put(np.zeros(shape, dtype=np.dtype(dtype)))


def _int_scalar(value):
    return _as_jax_dtype(value, jnp.int32)


def _bool_scalar(value):
    return _as_jax_dtype(value, jnp.bool_)


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
    return a + (-B + jnp.sqrt(radical)) / (three * A)


def _quadmin(a, fa, fpa, b, fb):
    dtype = jnp.result_type(a, fa, fpa, b, fb)
    two = _as_jax_dtype(2.0, dtype)
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db * db)
    return a - C / (two * B)


def _line_search_sample_valid(phi, dphi, grad):
    return jnp.isfinite(phi) & jnp.isfinite(dphi) & jnp.all(jnp.isfinite(grad))


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
        np.isfinite(phi) and np.isfinite(dphi) and np.all(np.isfinite(np.asarray(grad)))
    )


def _emit_debug_callback(callback, *args):
    """Dispatch private-optimizer callbacks without ordered-effect tokens.

    JAX 0.9.2 lowers ``ordered=True`` debug callbacks through a host token that
    strict ``transfer_guard='disallow'`` rejects as a ``bool[0]`` host-to-device
    transfer. Keep the private optimizer callback path unordered so strict
    on-device optimizer lanes remain executable.
    """
    jax.debug.callback(callback, *args, ordered=False)


def _reduce_sum_all(x):
    flat = jnp.reshape(jnp.asarray(x), (-1,))
    return lax.reduce(flat, _as_numpy_dtype(0.0, flat.dtype), lax.add, (0,))


def _reduce_max_all(x):
    flat = jnp.reshape(jnp.asarray(x), (-1,))
    return lax.reduce(flat, _as_numpy_dtype(-np.inf, flat.dtype), lax.max, (0,))


def _norm(x, *, ord=None):
    abs_x = jnp.abs(x)
    if ord in (None, 2):
        return jnp.sqrt(_reduce_sum_all(abs_x * abs_x))
    if ord == np.inf:
        return _reduce_max_all(abs_x)
    return jnp.linalg.norm(x, ord=ord)


def _pytree_inexact_dtype(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.float32
    dtype = jnp.result_type(*[jnp.asarray(leaf) for leaf in leaves])
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.promote_types(dtype, jnp.float32)
    return dtype


def _scalar_value_and_grad(fun):
    def wrapped(x):
        dtype = _pytree_inexact_dtype(x)
        value, pullback = jax.vjp(fun, x)
        value = jnp.asarray(value, dtype=dtype)
        cotangent = _as_jax_dtype(1.0, value.dtype)
        (grad,) = pullback(cotangent)
        grad = jax.tree_util.tree_map(
            lambda leaf: jnp.asarray(leaf, dtype=dtype),
            grad,
        )
        return value, grad

    return wrapped


def _cached_private_solver(cache_owner, *, cache_key, builder):
    if cache_owner is None or not getattr(
        cache_owner, _CACHEABLE_VALUE_AND_GRAD_ATTR, False
    ):
        return builder()
    cached = getattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR, None)
    if cached is not None:
        compiled = cached.get(cache_key)
        if compiled is not None:
            return compiled
    compiled = builder()
    # Double-checked install under the cache lock. ``cache_owner`` has been
    # marked via ``_mark_cacheable_jit_value_and_grad`` (the marker check
    # above gated this branch), so ``setattr`` cannot raise.
    with _PRIVATE_SOLVER_CACHE_LOCK:
        cached = getattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR, None)
        if cached is None:
            cached = {}
            setattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR, cached)
        existing = cached.get(cache_key)
        if existing is None:
            cached[cache_key] = compiled
            return compiled
        return existing


def _promote_dtypes_inexact(*args):
    """Promote arguments to a shared inexact dtype using public JAX APIs."""
    dtype = jnp.result_type(*args)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.promote_types(dtype, jnp.float32)
    return tuple(_as_jax_dtype(arg, dtype) for arg in args)


def _require_private_optimizer_runtime(x0):
    import jax

    if not private_optimizer_runtime_is_supported(jax.__version__):
        raise RuntimeError(
            f"On-device optimizer requires JAX >= "
            f"{PRIVATE_OPTIMIZER_JAX_VERSION}; found {jax.__version__}. "
            "Use a supported JAX runtime or "
            "fall back to optimizer_backend='scipy'."
        )
    if not _x64_enabled():
        raise RuntimeError(
            "On-device optimizer requires jax_enable_x64=True before import/use."
        )

    x0 = _as_jax_dtype(x0, jnp.float64)
    if x0.ndim != 1:
        raise ValueError(
            f"On-device optimizer expects a flat 1-D decision vector, got {x0.shape}."
        )
    return x0


def _resolve_lbfgs_limits(d, maxiter, maxfun, maxgrad):
    """Resolve None defaults onto the int32 counter domain used by L-BFGS."""
    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
    return tuple(
        _normalize_lbfgs_counter_limit(limit)
        for limit in (
            np.inf if maxiter is None else maxiter,
            np.inf if maxfun is None else maxfun,
            np.inf if maxgrad is None else maxgrad,
        )
    )


def _normalize_lbfgs_counter_limit(limit) -> np.int32:
    """Map user limits onto the integer counter space used inside the solver.

    The solver state tracks ``k``/``nfev``/``ngev`` as int32 counters, so keep
    comparison limits in the same domain. Finite non-integer inputs are rounded
    up to preserve the old ``counter >= float_limit`` semantics.
    """

    scalar_limit = np.asarray(limit).item()
    if np.isinf(scalar_limit):
        return np.int32(_INT32_COUNTER_MAX)
    return np.int32(min(int(np.ceil(scalar_limit)), _INT32_COUNTER_MAX))


def _emit_iteration_callbacks(callback, progress_callback, x_kp1, next_k, f_kp1, g_kp1):
    """Dispatch callback/progress_callback via jax.debug.callback.

    Shared by the BFGS and L-BFGS on-device body functions.
    Host materialization lives in ``jax_host_boundary`` so the solver core
    only exposes an observability/compatibility seam here.
    """
    if callback is not None:
        _emit_debug_callback(
            lambda x: callback(_callback_host_array(x, dtype=float)),
            x_kp1,
        )
    if progress_callback is not None:
        grad_inf = _norm(g_kp1, ord=jnp.inf)
        _emit_debug_callback(
            lambda iteration, fun_value, grad_inf_value: progress_callback(
                int(iteration),
                float(fun_value),
                float(grad_inf_value),
            ),
            next_k,
            f_kp1,
            grad_inf,
        )
