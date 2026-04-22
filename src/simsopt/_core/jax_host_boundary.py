"""Shared runtime/host boundary helpers for JAX-backed compatibility lanes."""

from __future__ import annotations

import numpy as np


def _require_jax():
    import jax

    return jax


def host_array(value, *, dtype=None):
    jax = _require_jax()
    array = np.asarray(jax.device_get(value))
    if dtype is None:
        return array
    return np.asarray(array, dtype=dtype)


def host_scalar(value, *, dtype=None):
    return host_array(value, dtype=dtype).item()


def host_float(value, *, dtype=np.float64) -> float:
    return float(host_scalar(value, dtype=dtype))


def host_int(value, *, dtype=np.int64) -> int:
    return int(host_scalar(value, dtype=dtype))


def host_bool(value) -> bool:
    return bool(host_scalar(value, dtype=np.bool_))


def host_all_finite(value, *, dtype=None) -> bool:
    return bool(np.all(np.isfinite(host_array(value, dtype=dtype))))


def host_inf_norm(value, *, dtype=None) -> float:
    return float(np.max(np.abs(host_array(value, dtype=dtype))))


def host_tree(value, *, dtype=None):
    jax = _require_jax()

    def _hostify_leaf(leaf):
        if isinstance(leaf, jax.core.Tracer):
            return leaf
        if isinstance(leaf, jax.Array):
            leaf_dtype = np.dtype(leaf.dtype) if dtype is None else dtype
            return host_array(leaf, dtype=leaf_dtype)
        if isinstance(leaf, np.ndarray):
            leaf_dtype = leaf.dtype if dtype is None else dtype
            return np.asarray(leaf, dtype=leaf_dtype)
        if dtype is not None and (
            isinstance(leaf, np.generic) or np.isscalar(leaf)
        ):
            return np.asarray(leaf, dtype=dtype)
        return leaf

    return jax.tree_util.tree_map(_hostify_leaf, value)


def scalar_pullback_seed(value):
    jax = _require_jax()
    # Build the pullback seed from ``value`` itself so the scalar cotangent stays
    # on-device under ``jax.transfer_guard("disallow")``.
    always_true = jax.numpy.logical_or(
        jax.numpy.equal(value, value),
        jax.numpy.not_equal(value, value),
    )
    return always_true.astype(value.dtype)


def strict_scalar_grad(fun, arg):
    jax = _require_jax()
    value, pullback = jax.vjp(fun, arg)
    (gradient,) = pullback(scalar_pullback_seed(value))
    return gradient


def strict_scalar_value_and_grad(fun, arg, *args):
    jax = _require_jax()

    def _objective(first_arg):
        return fun(first_arg, *args)

    value, pullback = jax.vjp(_objective, arg)
    (gradient,) = pullback(scalar_pullback_seed(value))
    return host_scalar(value), gradient


def explicit_cotangent_basis(length: int, index: int, *, dtype):
    jax = _require_jax()
    basis = np.zeros(int(length), dtype=np.dtype(dtype))
    basis[int(index)] = 1.0
    return jax.device_put(basis)
