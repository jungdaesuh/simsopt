"""Stable vector normalization primitives for tiny nonzero vectors."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _norm3_primal(vector: jax.Array) -> jax.Array:
    abs_vector = jnp.abs(vector)
    return jnp.hypot(
        jnp.hypot(abs_vector[..., 0:1], abs_vector[..., 1:2]),
        abs_vector[..., 2:3],
    )


@jax.custom_jvp
def norm3(vector: jax.Array) -> jax.Array:
    return _norm3_primal(vector)


@norm3.defjvp
def _norm3_jvp(
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    (vector,) = primals
    (vector_dot,) = tangents
    norm = _norm3_primal(vector)
    unit_vector = vector / norm
    norm_dot = jnp.sum(unit_vector * vector_dot, axis=-1, keepdims=True)
    return norm, norm_dot


@jax.custom_jvp
def unit_vector3(vector: jax.Array) -> jax.Array:
    return vector / _norm3_primal(vector)


@unit_vector3.defjvp
def _unit_vector3_jvp(
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    (vector,) = primals
    (vector_dot,) = tangents
    norm = _norm3_primal(vector)
    unit_vector = vector / norm
    tangent_projection = jnp.sum(unit_vector * vector_dot, axis=-1, keepdims=True)
    unit_vector_dot = (vector_dot - unit_vector * tangent_projection) / norm
    return unit_vector, unit_vector_dot
