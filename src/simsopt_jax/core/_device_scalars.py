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
    """A 1.0 placed and typed like ``reference``, with no derivative path.

    The value is built from ``reference`` only for placement.  Its derivative
    is zero, and it must also be computed as zero: left differentiable, reverse
    mode sends the full cotangent of every product ``device_one(r) * y`` into
    ``r`` twice, as ``+c`` and ``-c``, and accumulates them beside ``r``'s own
    cotangent.  When ``c`` dwarfs that cotangent -- ``mu0/4pi`` scaling a field
    whose currents are its reference -- the sum ``(g + c) - c`` keeps only
    ``u |c|`` of ``g``'s accuracy.
    """
    return jax.lax.stop_gradient(jnp.exp(jnp.sum(reference - reference)))


def two_pi(reference: jax.Array) -> jax.Array:
    pi = jnp.arccos(-device_one(reference))
    return pi + pi


def float_scalar(value: int, reference: jax.Array) -> jax.Array:
    return jnp.sum(jnp.broadcast_to(device_one(reference), (value,)))


def staged_like(reference: jax.Array, host_value, *, dtype=None) -> jax.Array:
    """Explicitly stage a value with reference-compatible placement.

    A host value is placed with the reference; so is a concrete device array
    held elsewhere (an explicit transfer), so the result always joins the
    reference in one program. Under a trace the value is converted in place.
    """
    reference = jnp.asarray(reference)
    resolved_dtype = reference.dtype if dtype is None else np.dtype(dtype)
    if isinstance(host_value, jax.Array):
        if isinstance(host_value, jax.core.Tracer) or isinstance(
            reference, jax.core.Tracer
        ):
            return jnp.asarray(host_value, dtype=resolved_dtype)
        return explicit_device_array(
            host_value,
            dtype=resolved_dtype,
            reference=reference,
        )
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
