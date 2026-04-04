"""Shared utilities for private optimizer submodules.

PRIVATE_OPTIMIZER_JAX_VERSION and _x64_enabled are imported from the public
module (``optimizer_jax``).  This is safe because both are defined *before*
the public module's ``try: from .optimizer_jax_private import ...`` block,
so they exist on the partially-initialized module object when this file loads.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..optimizer_jax import PRIVATE_OPTIMIZER_JAX_VERSION, _x64_enabled

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)
_INT32_COUNTER_MAX = np.iinfo(np.int32).max


def _as_jax_dtype(value, dtype):
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))
    return jnp.asarray(value, dtype=dtype)


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


def _reduce_sum_all(x):
    flat = jnp.reshape(jnp.asarray(x), (-1,))
    return lax.reduce(flat, _as_jax_dtype(0.0, flat.dtype), lax.add, (0,))


def _reduce_max_all(x):
    flat = jnp.reshape(jnp.asarray(x), (-1,))
    return lax.reduce(flat, _as_jax_dtype(-np.inf, flat.dtype), lax.max, (0,))


def _norm(x, *, ord=None):
    abs_x = jnp.abs(x)
    if ord in (None, 2):
        return jnp.sqrt(_reduce_sum_all(abs_x * abs_x))
    if ord == np.inf:
        return _reduce_max_all(abs_x)
    return jnp.linalg.norm(x, ord=ord)


def _scalar_value_and_grad(fun):
    def wrapped(x):
        value, pullback = jax.vjp(fun, x)
        value = jnp.asarray(value, dtype=x.dtype)
        cotangent = _as_jax_dtype(1.0, value.dtype)
        (grad,) = pullback(cotangent)
        return value, jnp.asarray(grad, dtype=x.dtype)

    return wrapped


def _promote_dtypes_inexact(*args):
    """Promote arguments to a shared inexact dtype using public JAX APIs."""
    dtype = jnp.result_type(*args)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.promote_types(dtype, jnp.float32)
    return tuple(_as_jax_dtype(arg, dtype) for arg in args)


def _require_private_optimizer_runtime(x0):
    import jax

    if jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION:
        raise RuntimeError(
            f"On-device optimizer is validated on JAX "
            f"{PRIVATE_OPTIMIZER_JAX_VERSION}; found {jax.__version__}. "
            "Use envs/jax-0.9.2.yml for the supported runtime or "
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
    """
    import jax
    import numpy as np

    if callback is not None:
        jax.debug.callback(
            lambda x: callback(np.asarray(x, dtype=float)),
            x_kp1,
            ordered=True,
        )
    if progress_callback is not None:
        grad_inf = _norm(g_kp1, ord=jnp.inf)
        jax.debug.callback(
            lambda iteration, fun_value, grad_inf_value: progress_callback(
                int(iteration),
                float(fun_value),
                float(grad_inf_value),
            ),
            next_k,
            f_kp1,
            grad_inf,
            ordered=True,
        )
