"""Compatibility facade for backend-owned JAX dtype helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.backend.dtypes import (
    _shape_tuple,
    as_jax_array,
    as_jax_float64,
    as_jax_int32,
    as_runtime_array,
    as_runtime_float64,
    as_runtime_value,
    explicit_device_array as _explicit_device_array,
    host_dtype,
    require_float64_dtype,
    require_runtime_dtype,
    runtime_device_put,
    runtime_dtype,
    runtime_eye,
    runtime_host_dtype,
    runtime_jnp_dtype,
    runtime_np_dtype,
    runtime_zeros,
)

__all__ = (
    "_explicit_device_array",
    "as_jax_array",
    "as_jax_float64",
    "as_jax_int32",
    "as_runtime_array",
    "as_runtime_float64",
    "as_runtime_value",
    "axis0_entries",
    "concat_jax_float64",
    "explicit_inv",
    "explicit_rsqrt",
    "eye",
    "host_dtype",
    "iter_axis0_entries",
    "pad_axis",
    "require_float64_dtype",
    "require_runtime_dtype",
    "runtime_device_put",
    "runtime_dtype",
    "runtime_eye",
    "runtime_host_dtype",
    "runtime_jnp_dtype",
    "runtime_np_dtype",
    "runtime_zeros",
    "scalar_like",
    "zero_padding_like",
    "zeros",
)


def iter_axis0_entries(array):
    """Yield axis-0 slices from a shaped JAX array."""
    axis0_size = int(array.shape[0])
    if axis0_size == 0:
        return
    for entry in jnp.split(array, axis0_size, axis=0):
        yield jnp.squeeze(entry, axis=0)


def axis0_entries(array: object) -> tuple[jax.Array, ...]:
    array_jax = as_jax_float64(array)
    if array_jax.ndim == 0:
        return (array_jax,)
    return tuple(iter_axis0_entries(array_jax))


def concat_jax_float64(*parts) -> jax.Array:
    return jnp.concatenate(tuple(as_jax_float64(part) for part in parts))


def scalar_like(reference, value) -> jax.Array:
    return as_jax_array(value, dtype=reference.dtype)


def zero_padding_like(array, *, axis: int, pad_width: int):
    axis_index = int(axis) if axis >= 0 else array.ndim + int(axis)
    zero_slice = jnp.sum(array, axis=axis_index, keepdims=True, dtype=array.dtype)
    zero_slice = zero_slice - zero_slice
    target_shape = (
        array.shape[:axis_index] + (int(pad_width),) + array.shape[axis_index + 1 :]
    )
    return jnp.broadcast_to(zero_slice, target_shape)


def pad_axis(array, *, axis: int, padded_size: int):
    axis_index = int(axis) if axis >= 0 else array.ndim + int(axis)
    pad_width = int(padded_size) - int(array.shape[axis_index])
    if pad_width <= 0:
        return array
    return jnp.concatenate(
        (
            array,
            zero_padding_like(array, axis=axis_index, pad_width=pad_width),
        ),
        axis=axis_index,
    )


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
    tangent_out = x_dot * scalar_like(x, -0.5) / (x * jnp.sqrt(x))
    return primal_out, tangent_out
