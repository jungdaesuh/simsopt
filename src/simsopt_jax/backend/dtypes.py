"""Policy-owned JAX dtype and device-placement helpers."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from simsopt_jax.backend.runtime import (
    get_backend_policy,
    maybe_initialize_distributed_jax,
)

__all__ = [
    "as_jax_array",
    "as_jax_float64",
    "as_jax_int32",
    "as_runtime_array",
    "as_runtime_float64",
    "as_runtime_value",
    "explicit_device_array",
    "host_dtype",
    "require_float64_dtype",
    "require_runtime_dtype",
    "runtime_device_put",
    "runtime_dtype",
    "runtime_eye",
    "runtime_host_dtype",
    "runtime_jnp_dtype",
    "runtime_np_dtype",
    "runtime_zeros",
]

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


def _is_jax_tracer(value) -> bool:
    return isinstance(value, jax.core.Tracer)


def _contains_traced_jax_leaves(value) -> bool:
    return any(_is_jax_tracer(leaf) for leaf in jax.tree.leaves(value))


def _has_jax_array_value(value) -> bool:
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return True
    return isinstance(value, (list, tuple)) and _contains_jax_leaves(value)


def _contains_concrete_jax_leaves(value) -> bool:
    return any(
        isinstance(leaf, jax.Array) and not _is_jax_tracer(leaf)
        for leaf in jax.tree.leaves(value)
    )


def _has_only_traced_jax_leaves(value) -> bool:
    return _has_jax_array_value(value) and not _contains_concrete_jax_leaves(value)


def _reference_sharding(reference, *, ndim: int | None = None):
    # A tracer is a ``jax.Array`` but carries no concrete sharding; probing
    # ``tracer.sharding`` raises ``AttributeError`` whose message eagerly walks
    # the entire jaxpr (jax's ``_origin_msg``/``find_progenitors``) only to be
    # discarded by ``getattr(..., None)``. Paid once per ``as_runtime_array``
    # call across an O(jaxpr) trace, that is an O(jaxpr^2) construction cost that
    # scales with resolution. Skip tracers: their reference sharding is always
    # ``None`` and ``as_runtime_array`` bypasses reference placement for traced
    # values regardless (see ``_has_only_traced_jax_leaves`` guard there).
    if _is_jax_tracer(reference):
        return None
    if isinstance(reference, jax.Array):
        sharding = getattr(reference, "sharding", None)
        return _compatible_reference_sharding(sharding, ndim=ndim)
    if isinstance(reference, (list, tuple)):
        for leaf in jax.tree.leaves(reference):
            if isinstance(leaf, jax.Array) and not _is_jax_tracer(leaf):
                sharding = getattr(leaf, "sharding", None)
                if sharding is not None:
                    return _compatible_reference_sharding(sharding, ndim=ndim)
    return None


def _compatible_reference_sharding(sharding, *, ndim: int | None):
    if not isinstance(sharding, NamedSharding):
        return None
    if ndim is None or len(sharding.spec) <= ndim:
        return sharding
    return NamedSharding(sharding.mesh, P())


def _value_ndim(value) -> int | None:
    if isinstance(value, (list, tuple)) and _contains_jax_leaves(value):
        return None
    if _contains_traced_jax_leaves(value):
        return None
    if isinstance(value, jax.Array):
        return int(value.ndim)
    if hasattr(value, "aval"):
        return None
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return int(np.ndim(value))
    return None


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
    dtype_name = get_backend_policy().runtime_dtype
    return _jnp_dtype_from_name(dtype_name, source="BackendPolicy.runtime_dtype")


def runtime_np_dtype() -> np.dtype:
    dtype_name = get_backend_policy().runtime_dtype
    return _np_dtype_from_name(dtype_name, source="BackendPolicy.runtime_dtype")


def runtime_host_dtype() -> np.dtype:
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
    resolved_dtype = _resolve_jnp_dtype(dtype, source="dtype")
    reference_sharding = _reference_sharding(reference, ndim=_value_ndim(value))
    if reference_sharding is not None and not _has_only_traced_jax_leaves(value):
        return runtime_device_put(
            value, dtype=resolved_dtype, target=reference_sharding
        )
    return as_jax_array(value, dtype=resolved_dtype)


def as_runtime_value(value, *, reference, dtype=None):
    require_runtime_dtype("reference", reference, dtype=dtype)
    return as_runtime_array(value, dtype=dtype, reference=reference)


def as_runtime_float64(value, *, reference):
    return as_runtime_array(value, reference=reference)


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
