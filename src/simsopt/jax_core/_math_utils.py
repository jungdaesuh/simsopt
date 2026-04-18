"""Shared strict-safe JAX math and array helpers for kernel code."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _explicit_device_array(value, *, dtype) -> jax.Array:
    from simsopt.backend import maybe_initialize_distributed_jax

    maybe_initialize_distributed_jax()
    return jax.device_put(np.asarray(value, dtype=np.dtype(dtype)))


def _shape_tuple(shape) -> tuple[int, ...]:
    if np.isscalar(shape):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _contains_jax_leaves(value) -> bool:
    return any(
        isinstance(leaf, jax.Array) or hasattr(leaf, "aval")
        for leaf in jax.tree_util.tree_leaves(value)
    )


def as_jax_array(value, *, dtype) -> jax.Array:
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (list, tuple)) and _contains_jax_leaves(value):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return _explicit_device_array(value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype)


def as_jax_float64(value) -> jax.Array:
    return as_jax_array(value, dtype=jnp.float64)


def as_jax_int32(value) -> jax.Array:
    return as_jax_array(value, dtype=jnp.int32)


def as_runtime_array(value, *, dtype, reference):
    # ``reference`` is accepted for call-site symmetry with tracer-aware APIs;
    # conversion is device-uniform and does not branch on it.
    del reference
    return as_jax_array(value, dtype=dtype)


def as_runtime_float64(value, *, reference):
    return as_runtime_array(value, dtype=jnp.float64, reference=reference)


def concat_jax_float64(*parts) -> jax.Array:
    return jnp.concatenate(tuple(as_jax_float64(part) for part in parts))


def scalar_at_axis0(array, index: int) -> jax.Array:
    selector = np.zeros(int(array.shape[0]), dtype=np.float64)
    selector[int(index)] = 1.0
    return jnp.dot(array, _explicit_device_array(selector, dtype=np.float64))


def scalar_like(reference, value) -> jax.Array:
    return _explicit_device_array(value, dtype=reference.dtype)


def zeros(shape, dtype=jnp.float64) -> jax.Array:
    return _explicit_device_array(
        np.zeros(_shape_tuple(shape), dtype=np.dtype(dtype)),
        dtype=dtype,
    )


def eye(size: int, dtype=jnp.float64) -> jax.Array:
    return _explicit_device_array(
        np.eye(int(size), dtype=np.dtype(dtype)),
        dtype=dtype,
    )


def _explicit_inv_impl(x):
    return jnp.divide(scalar_like(x, 1.0), x)


@jax.custom_jvp
def explicit_inv(x):
    return _explicit_inv_impl(x)


@explicit_inv.defjvp
def _explicit_inv_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = _explicit_inv_impl(x)
    tangent_out = jnp.negative(x_dot * primal_out * primal_out)
    return primal_out, tangent_out


def _explicit_rsqrt_impl(x):
    return jnp.divide(scalar_like(x, 1.0), jnp.sqrt(x))


@jax.custom_jvp
def explicit_rsqrt(x):
    return _explicit_rsqrt_impl(x)


@explicit_rsqrt.defjvp
def _explicit_rsqrt_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = _explicit_rsqrt_impl(x)
    tangent_out = x_dot * scalar_like(x, -0.5) * primal_out * _explicit_inv_impl(x)
    return primal_out, tangent_out
