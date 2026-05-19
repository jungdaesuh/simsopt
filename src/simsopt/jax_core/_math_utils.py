"""Shared strict-safe JAX math and array helpers for kernel code."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_DTYPE_BY_NAME = {
    "float64": jnp.float64,
    "float32": jnp.float32,
}
_HOST_DTYPE_BY_NAME = {
    "float64": np.dtype(np.float64),
    "float32": np.dtype(np.float32),
}


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
        for leaf in jax.tree.leaves(value)
    )


def _array_like_dtype(value) -> np.dtype | None:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    if isinstance(value, (np.ndarray, np.generic)):
        return np.asarray(value).dtype
    if isinstance(value, (list, tuple)):
        if _contains_jax_leaves(value):
            return np.dtype(jnp.asarray(value).dtype)
        return np.asarray(value).dtype
    return None


def _dtype_name(dtype, *, source: str) -> str:
    if isinstance(dtype, str):
        name = dtype
    else:
        name = np.dtype(dtype).name
    if name not in _DTYPE_BY_NAME:
        accepted = tuple(_DTYPE_BY_NAME)
        raise TypeError(f"{source} must be one of {accepted}; got {name!r}.")
    return name


def _jnp_dtype_from_name(name: str, *, source: str):
    if name not in _DTYPE_BY_NAME:
        accepted = tuple(_DTYPE_BY_NAME)
        raise TypeError(f"{source} must be one of {accepted}; got {name!r}.")
    return _DTYPE_BY_NAME[name]


def _np_dtype_from_name(name: str, *, source: str) -> np.dtype:
    if name not in _HOST_DTYPE_BY_NAME:
        accepted = tuple(_HOST_DTYPE_BY_NAME)
        raise TypeError(f"{source} must be one of {accepted}; got {name!r}.")
    return _HOST_DTYPE_BY_NAME[name]


def _resolve_jnp_dtype(dtype, *, source: str):
    if dtype is None:
        return runtime_jnp_dtype()
    return _jnp_dtype_from_name(_dtype_name(dtype, source=source), source=source)


def _resolve_np_dtype(dtype, *, source: str) -> np.dtype:
    if dtype is None:
        return runtime_np_dtype()
    return _np_dtype_from_name(_dtype_name(dtype, source=source), source=source)


def runtime_jnp_dtype():
    from simsopt.backend import get_backend_policy

    dtype_name = get_backend_policy().runtime_dtype
    return _jnp_dtype_from_name(dtype_name, source="BackendPolicy.runtime_dtype")


def runtime_np_dtype() -> np.dtype:
    from simsopt.backend import get_backend_policy

    dtype_name = get_backend_policy().runtime_dtype
    return _np_dtype_from_name(dtype_name, source="BackendPolicy.runtime_dtype")


def runtime_host_dtype() -> np.dtype:
    from simsopt.backend import get_backend_policy

    dtype_name = get_backend_policy().host_dtype
    return _np_dtype_from_name(dtype_name, source="BackendPolicy.host_dtype")


def require_runtime_dtype(name: str, value, *, dtype=None) -> None:
    expected_dtype = _resolve_np_dtype(dtype, source="dtype")
    dtype = _array_like_dtype(value)
    if dtype is not None and dtype != expected_dtype:
        raise TypeError(
            f"{name} must have runtime dtype {expected_dtype.name}; got {dtype}."
        )


def require_float64_dtype(name: str, value) -> None:
    require_runtime_dtype(name, value, dtype="float64")


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


def iter_axis0_entries(array):
    """Yield axis-0 slices from a shaped JAX array."""
    for index in range(int(array.shape[0])):
        yield array[index]


def axis0_entries(array: object) -> tuple[jax.Array, ...]:
    array_jax = as_jax_float64(array)
    if array_jax.ndim == 0:
        return (array_jax,)
    return tuple(iter_axis0_entries(array_jax))


def as_jax_int32(value) -> jax.Array:
    return as_jax_array(value, dtype=jnp.int32)


def as_runtime_array(value, *, dtype, reference):
    # ``reference`` is accepted for call-site symmetry with tracer-aware APIs;
    # conversion is device-uniform and does not branch on it.
    del reference
    return as_jax_array(value, dtype=dtype)


def as_runtime_value(value, *, reference, dtype=None):
    require_runtime_dtype("reference", reference, dtype=dtype)
    return as_runtime_array(
        value,
        dtype=_resolve_jnp_dtype(dtype, source="dtype"),
        reference=reference,
    )


def as_runtime_float64(value, *, reference):
    return as_runtime_array(value, dtype=jnp.float64, reference=reference)


def concat_jax_float64(*parts) -> jax.Array:
    return jnp.concatenate(tuple(as_jax_float64(part) for part in parts))


def scalar_like(reference, value) -> jax.Array:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return jnp.asarray(value, dtype=reference.dtype)
    return _explicit_device_array(value, dtype=reference.dtype)


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
