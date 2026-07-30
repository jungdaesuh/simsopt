"""Helpers for constructing scalar values on the same device as a reference array."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import explicit_device_array


@lru_cache(maxsize=None)
def _staged_scalar_builder(host_value: object, dtype_string: str):
    resolved_dtype = np.dtype(dtype_string)

    @jax.jit
    def build(reference):
        zero = jnp.sum(reference - reference).astype(resolved_dtype)
        literal = jnp.asarray(host_value, dtype=resolved_dtype)
        return zero + literal

    return build


def device_one(reference: jax.Array) -> jax.Array:
    return jnp.exp(jnp.sum(reference - reference))


def two_pi(reference: jax.Array) -> jax.Array:
    pi = jax.lax.stop_gradient(jnp.arccos(-device_one(reference)))
    return pi + pi


def float_scalar(value: int, reference: jax.Array) -> jax.Array:
    return jnp.sum(jnp.broadcast_to(device_one(reference), (value,)))


def staged_like(reference: jax.Array, host_value, *, dtype=None) -> jax.Array:
    """Explicitly stage a host literal with reference-compatible placement."""
    reference = jnp.asarray(reference)
    resolved_dtype = reference.dtype if dtype is None else np.dtype(dtype)
    if isinstance(host_value, jax.Array):
        return jnp.asarray(host_value, dtype=resolved_dtype)
    if isinstance(reference, jax.core.Tracer) and np.ndim(host_value) == 0:
        typed_host_value = np.asarray(host_value, dtype=resolved_dtype)[()]
        return _staged_scalar_builder(
            typed_host_value,
            resolved_dtype.str,
        )(reference)
    if isinstance(reference, jax.core.Tracer):
        return jnp.asarray(host_value, dtype=resolved_dtype)
    return explicit_device_array(
        host_value,
        dtype=resolved_dtype,
        reference=reference,
    )
