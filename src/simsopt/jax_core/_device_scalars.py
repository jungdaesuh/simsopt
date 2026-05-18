"""Helpers for constructing scalar values on the same device as a reference array."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def device_one(reference: jax.Array) -> jax.Array:
    return jnp.exp(jnp.sum(reference - reference))


def two_pi(reference: jax.Array) -> jax.Array:
    pi = jax.lax.stop_gradient(jnp.arccos(-device_one(reference)))
    return pi + pi


def float_scalar(value: int, reference: jax.Array) -> jax.Array:
    return jnp.sum(jnp.broadcast_to(device_one(reference), (value,)))
