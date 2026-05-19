"""Policy-owned JAX dtype and device-placement helpers."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

_DTYPE_BY_NAME = {
    "float64": jnp.float64,
    "float32": jnp.float32,
}
_HOST_DTYPE_BY_NAME = {
    "float64": np.dtype(np.float64),
    "float32": np.dtype(np.float32),
}


def _shape_tuple(shape) -> tuple[int, ...]:
    if np.isscalar(shape):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _contains_jax_leaves(value) -> bool:
    return any(
        isinstance(leaf, jax.Array) or hasattr(leaf, "aval")
        for leaf in jax.tree.leaves(value)
    )


def _has_jax_array_value(value) -> bool:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return True
    return isinstance(value, (list, tuple)) and _contains_jax_leaves(value)


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
    if np.isscalar(value):
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


def _resolve_jnp_dtype(dtype, *, source: str):
    if dtype is None:
        return runtime_jnp_dtype()
    return _jnp_dtype_from_name(_dtype_name(dtype, source=source), source=source)


def _resolve_np_dtype(dtype, *, source: str) -> np.dtype:
    if dtype is None:
        return runtime_np_dtype()
    return _np_dtype_from_name(_dtype_name(dtype, source=source), source=source)


def _device_put_target(target, device):
    if target is not None and device is not None:
        raise TypeError("runtime_device_put accepts either target or device, not both.")
    return device if device is not None else target


def _runtime_device_put_dtype(value, dtype) -> np.dtype | None:
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            return runtime_np_dtype()
        return dtype
    value_dtype = _array_like_dtype(value)
    if value_dtype is not None and value_dtype.kind == "f":
        return runtime_np_dtype()
    return None


def runtime_device_put(value, *, dtype=None, target=None, device=None) -> jax.Array:
    """Place host values on a JAX device using runtime policy for float dtypes."""
    placement = _device_put_target(target, device)
    resolved_dtype = _runtime_device_put_dtype(value, dtype)
    if _has_jax_array_value(value):
        array = jnp.asarray(value, dtype=resolved_dtype)
    elif resolved_dtype is None:
        array = np.asarray(value)
    else:
        array = np.asarray(value, dtype=resolved_dtype)
    from simsopt.backend import maybe_initialize_distributed_jax

    maybe_initialize_distributed_jax()
    if placement is None:
        return jax.device_put(array)
    return jax.device_put(array, placement)


def as_jax_array(value, *, dtype) -> jax.Array:
    if _has_jax_array_value(value):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return runtime_device_put(value, dtype=dtype)
    return jnp.asarray(value, dtype=dtype)


def as_jax_float64(value) -> jax.Array:
    return as_runtime_array(value)


def as_jax_int32(value) -> jax.Array:
    return as_jax_array(value, dtype=jnp.int32)


def as_runtime_array(value, *, dtype=None, reference=None):
    del reference
    return as_jax_array(value, dtype=_resolve_jnp_dtype(dtype, source="dtype"))


def as_runtime_value(value, *, reference, dtype=None):
    require_runtime_dtype("reference", reference, dtype=dtype)
    return as_runtime_array(value, dtype=dtype, reference=reference)


def as_runtime_float64(value, *, reference):
    del reference
    return as_runtime_array(value)


def require_runtime_dtype(name: str, value, *, dtype=None) -> None:
    expected_dtype = _resolve_np_dtype(dtype, source="dtype")
    actual_dtype = _array_like_dtype(value)
    if actual_dtype is not None and actual_dtype != expected_dtype:
        raise TypeError(
            f"{name} must have runtime dtype {expected_dtype.name}; got {actual_dtype}."
        )


def require_float64_dtype(name: str, value) -> None:
    require_runtime_dtype(name, value)


def runtime_dtype():
    return runtime_jnp_dtype()


def host_dtype() -> np.dtype:
    return runtime_host_dtype()


def runtime_zeros(shape) -> jax.Array:
    return runtime_device_put(np.zeros(_shape_tuple(shape), dtype=runtime_np_dtype()))


def runtime_eye(n: int) -> jax.Array:
    return runtime_device_put(np.eye(int(n), dtype=runtime_np_dtype()))


def explicit_device_array(value, *, dtype) -> jax.Array:
    return runtime_device_put(value, dtype=dtype)
