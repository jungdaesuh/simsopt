"""Shared utilities for private optimizer submodules.

PRIVATE_OPTIMIZER_JAX_VERSION and _x64_enabled are imported from the shared
optimizer helper module so private modules do not depend on the public
optimizer module during import.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import partial
from threading import Event, Lock, get_ident
from typing import Generic, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from simsopt_jax.backend import (
    get_backend_policy,
    is_float32_smoke_policy,
)
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.sharding import active_replicated_sharding
from simsopt_jax.runtime.host_boundary import host_array as _callback_host_array

from .._shared import (
    _CACHEABLE_VALUE_AND_GRAD_ATTR,
    PRIVATE_OPTIMIZER_JAX_VERSION,
    _x64_enabled,
    private_optimizer_runtime_is_supported,
)

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)
_INT32_COUNTER_MAX = np.iinfo(np.int32).max
_PRIVATE_SOLVER_CACHE_LOCK = Lock()
_PRIVATE_SOLVER_CACHE_ATTR = "_simsopt_cached_private_solver"
_PRIVATE_SOLVER_CACHE_CAPACITY = 8
_SolverT = TypeVar("_SolverT")


@dataclass
class _PendingPrivateSolver(Generic[_SolverT]):
    """Single-flight marker for one objective-owned compilation."""

    owner_thread_id: int
    ready: Event = field(default_factory=Event)
    result: _SolverT | None = None
    error: BaseException | None = None


def _private_optimizer_device_put(
    value: jax.typing.ArrayLike,
    *,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    sharding = active_replicated_sharding()
    if not isinstance(value, jax.Array) and hasattr(value, "aval"):
        array = jnp.asarray(value, dtype=dtype)
        if sharding is None:
            return array
        return lax.with_sharding_constraint(array, sharding)
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        array = jnp.asarray(value, dtype=dtype)
    elif isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        array = np.asarray(value, dtype=np.dtype(dtype))
    else:
        array = jnp.asarray(value, dtype=dtype)
    if sharding is not None:
        return explicit_device_array(array, dtype=dtype, target=sharding)
    return explicit_device_array(array, dtype=dtype)


def _as_jax_dtype(
    value: jax.typing.ArrayLike,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    if isinstance(value, jax.Array):
        return _private_optimizer_device_put(value, dtype=dtype)
    if hasattr(value, "aval"):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return _private_optimizer_device_put(value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype)


def _as_numpy_dtype(
    value: jax.typing.ArrayLike, dtype: jax.typing.DTypeLike
) -> np.ndarray:
    return np.asarray(value, dtype=np.dtype(dtype))


def _eye(n: int, dtype: jax.typing.DTypeLike) -> jax.Array:
    return _private_optimizer_device_put(
        np.eye(int(n), dtype=np.dtype(dtype)),
        dtype=dtype,
    )


def _zeros(
    shape: int | tuple[int, ...],
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    if np.isscalar(shape):
        shape = (int(shape),)
    else:
        shape = tuple(int(dim) for dim in shape)
    return _private_optimizer_device_put(
        np.zeros(shape, dtype=np.dtype(dtype)),
        dtype=dtype,
    )


def _int_scalar(value: jax.typing.ArrayLike) -> jax.Array:
    return _as_jax_dtype(value, jnp.int32)


def _bool_scalar(value: jax.typing.ArrayLike) -> jax.Array:
    return _as_jax_dtype(value, jnp.bool_)


def _cubicmin(
    a: jax.Array,
    fa: jax.Array,
    fpa: jax.Array,
    b: jax.Array,
    fb: jax.Array,
    c: jax.Array,
    fc: jax.Array,
) -> jax.Array:
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


def _quadmin(
    a: jax.Array,
    fa: jax.Array,
    fpa: jax.Array,
    b: jax.Array,
    fb: jax.Array,
) -> jax.Array:
    dtype = jnp.result_type(a, fa, fpa, b, fb)
    two = _as_jax_dtype(2.0, dtype)
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db * db)
    return a - C / (two * B)


def _line_search_sample_valid(
    phi: jax.Array,
    dphi: jax.Array,
    grad: jax.Array,
) -> jax.Array:
    return jnp.isfinite(phi) & jnp.isfinite(dphi) & jnp.all(jnp.isfinite(grad))


def _emit_debug_callback(callback: Callable[..., object], *args: jax.Array) -> None:
    """Dispatch private-optimizer callbacks without ordered-effect tokens.

    ``ordered=True`` debug callbacks lower through a host token that strict
    ``transfer_guard='disallow'`` rejects as a host-to-device transfer. Keep the
    private optimizer callback path unordered so strict on-device optimizer
    lanes remain executable.
    """
    jax.debug.callback(callback, *args, ordered=False)


def _reduce_sum_all(x: jax.Array) -> jax.Array:
    flat = jnp.reshape(jnp.asarray(x), (-1,))
    return lax.reduce(flat, _as_numpy_dtype(0.0, flat.dtype), lax.add, (0,))


def _reduce_max_all(x: jax.Array) -> jax.Array:
    flat = jnp.reshape(jnp.asarray(x), (-1,))
    return lax.reduce(flat, _as_numpy_dtype(-np.inf, flat.dtype), lax.max, (0,))


def _norm(x: jax.Array, *, ord: float | None = None) -> jax.Array:
    abs_x = jnp.abs(x)
    if ord in (None, 2):
        return jnp.sqrt(_reduce_sum_all(abs_x * abs_x))
    if ord == np.inf:
        return _reduce_max_all(abs_x)
    return jnp.linalg.norm(x, ord=ord)


def _pytree_inexact_dtype(tree: object) -> jax.typing.DTypeLike:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    dtype = jnp.result_type(*[jnp.asarray(leaf) for leaf in leaves])
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.promote_types(dtype, jnp.float32)
    return dtype


def _scalar_value_and_grad(
    fun: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    def wrapped(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = _pytree_inexact_dtype(x)
        value, pullback = jax.vjp(fun, x)
        cotangent = jnp.ones_like(value)
        (grad,) = pullback(cotangent)
        value = jnp.asarray(value, dtype=dtype)
        grad = jax.tree.map(
            lambda leaf: jnp.asarray(leaf, dtype=dtype),
            grad,
        )
        return value, grad

    return jax.jit(wrapped)


def _cached_private_solver(
    cache_owner: object | None,
    *,
    cache_key: Hashable,
    builder: Callable[[], _SolverT],
) -> _SolverT:
    if cache_owner is None or not getattr(
        cache_owner, _CACHEABLE_VALUE_AND_GRAD_ATTR, False
    ):
        return builder()
    pending: _PendingPrivateSolver | None = None
    with _PRIVATE_SOLVER_CACHE_LOCK:
        cached = getattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR, None)
        if cached is None:
            cached = OrderedDict()
            setattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR, cached)
        if cache_key in cached:
            existing = cached[cache_key]
            cached.move_to_end(cache_key)
            if isinstance(existing, _PendingPrivateSolver):
                if existing.owner_thread_id == get_ident():
                    raise RuntimeError("recursive private solver compilation")
                pending = existing
            else:
                return cast(_SolverT, existing)
        else:
            pending = _PendingPrivateSolver(owner_thread_id=get_ident())
            cached[cache_key] = pending
            cached.move_to_end(cache_key)

    if pending is not None and pending.owner_thread_id != get_ident():
        pending.ready.wait()
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise RuntimeError("private solver compilation completed without a result")
        return pending.result

    assert pending is not None
    try:
        compiled = builder()
    except BaseException as error:
        with _PRIVATE_SOLVER_CACHE_LOCK:
            cached = getattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR)
            if cached.get(cache_key) is pending:
                del cached[cache_key]
            pending.error = error
            pending.ready.set()
        raise

    with _PRIVATE_SOLVER_CACHE_LOCK:
        cached = getattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR)
        cached[cache_key] = compiled
        cached.move_to_end(cache_key)
        evictable_keys = [
            key
            for key, value in cached.items()
            if not isinstance(value, _PendingPrivateSolver)
        ]
        while len(cached) > _PRIVATE_SOLVER_CACHE_CAPACITY and evictable_keys:
            evicted_key = evictable_keys.pop(0)
            if evicted_key in cached and not isinstance(
                cached[evicted_key], _PendingPrivateSolver
            ):
                del cached[evicted_key]
        pending.result = compiled
        pending.ready.set()
    return compiled


def _private_solver_cache_size(cache_owner: object) -> int:
    """Return the number of retained compiled wrappers for one owner."""

    with _PRIVATE_SOLVER_CACHE_LOCK:
        cached = getattr(cache_owner, _PRIVATE_SOLVER_CACHE_ATTR, None)
        return 0 if cached is None else len(cached)


def _promote_dtypes_inexact(*args: jax.typing.ArrayLike) -> tuple[jax.Array, ...]:
    """Promote arguments to a shared inexact dtype using public JAX APIs."""
    dtype = jnp.result_type(*args)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.promote_types(dtype, jnp.float32)
    return tuple(_as_jax_dtype(arg, dtype) for arg in args)


def _require_private_optimizer_runtime(
    x0: jax.typing.ArrayLike,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> jax.Array:
    if not private_optimizer_runtime_is_supported(jax.__version__):
        raise RuntimeError(
            f"On-device optimizer requires JAX >= "
            f"{PRIVATE_OPTIMIZER_JAX_VERSION}; found {jax.__version__}. "
            "Use a supported JAX runtime or select optimizer_backend='scipy'."
        )
    policy = get_backend_policy()
    if not _x64_enabled() and not is_float32_smoke_policy(policy):
        raise RuntimeError(
            "On-device optimizer requires jax_enable_x64=True before import/use."
        )

    target_dtype = jnp.dtype(policy.runtime_dtype if dtype is None else dtype)
    x0 = _as_jax_dtype(x0, target_dtype)
    if x0.ndim != 1:
        raise ValueError(
            f"On-device optimizer expects a flat 1-D decision vector, got {x0.shape}."
        )
    return x0


def _resolve_lbfgs_limits(
    d: int,
    maxiter: jax.typing.ArrayLike | None,
    maxfun: jax.typing.ArrayLike | None,
    maxgrad: jax.typing.ArrayLike | None,
) -> tuple[np.int32, np.int32, np.int32]:
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


def _normalize_lbfgs_counter_limit(limit: jax.typing.ArrayLike) -> np.int32:
    """Map user limits onto the integer counter space used inside the solver.

    The solver state tracks ``k``/``nfev``/``ngev`` as int32 counters, so keep
    comparison limits in the same domain. Finite non-integer inputs are rounded
    up to preserve the old ``counter >= float_limit`` semantics.
    """

    scalar_limit = np.asarray(limit).item()
    if np.isinf(scalar_limit):
        return np.int32(_INT32_COUNTER_MAX)
    return np.int32(min(int(np.ceil(scalar_limit)), _INT32_COUNTER_MAX))


def _emit_iteration_callbacks(
    callback: Callable[[np.ndarray], object] | None,
    progress_callback: Callable[[int, float, float], object] | None,
    x_kp1: jax.Array,
    next_k: jax.Array,
    f_kp1: jax.Array,
    g_kp1: jax.Array,
) -> None:
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
