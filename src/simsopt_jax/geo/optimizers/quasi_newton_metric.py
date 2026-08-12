"""Fixed-shape limited-memory secant metric for device-resident optimizers.

The module owns one piece of knowledge: how a bounded ring of accepted
curvature pairs turns into an inverse-curvature operator.  Every array has a
compile-time shape and every admission decision is masked, so the metric can be
carried through ``jax.lax`` loop state and applied inside a traced solve.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# A pair is admitted only when its curvature is positive relative to the pair's
# own magnitude; the floor also rejects nonfinite pairs, since every comparison
# against NaN is False.
_CURVATURE_ADMISSION_FLOOR = 1.0e-10


class QuasiNewtonMetric(NamedTuple):
    """Ring of admitted secant pairs with the newest pair at ``insertion_index - 1``."""

    steps: jax.Array
    gradient_changes: jax.Array
    reciprocal_curvatures: jax.Array
    pair_valid: jax.Array
    insertion_index: jax.Array


def empty_quasi_newton_metric(
    memory: int,
    dimension: int,
    dtype: jnp.dtype,
) -> QuasiNewtonMetric:
    """Return a metric holding no pair, which applies as the exact identity."""

    return QuasiNewtonMetric(
        steps=jnp.zeros((memory, dimension), dtype=dtype),
        gradient_changes=jnp.zeros((memory, dimension), dtype=dtype),
        reciprocal_curvatures=jnp.zeros((memory,), dtype=dtype),
        pair_valid=jnp.zeros((memory,), dtype=jnp.bool_),
        insertion_index=jnp.asarray(0, dtype=jnp.int32),
    )


def curvature_pair_admissible(step: jax.Array, gradient_change: jax.Array) -> jax.Array:
    """Report whether one pair has positive curvature relative to its magnitude."""

    curvature = step @ gradient_change
    return curvature > _CURVATURE_ADMISSION_FLOOR * (
        jnp.linalg.norm(step) * jnp.linalg.norm(gradient_change)
    )


def insert_curvature_pair(
    metric: QuasiNewtonMetric,
    step: jax.Array,
    gradient_change: jax.Array,
) -> QuasiNewtonMetric:
    """Overwrite the oldest ring slot, but only with an admissible pair."""

    admissible = curvature_pair_admissible(step, gradient_change)
    curvature = step @ gradient_change
    memory = metric.steps.shape[0]
    index = jnp.mod(metric.insertion_index, memory)
    safe_curvature = jnp.where(admissible, curvature, jnp.ones_like(curvature))
    return QuasiNewtonMetric(
        steps=metric.steps.at[index].set(
            jnp.where(admissible, step, metric.steps[index])
        ),
        gradient_changes=metric.gradient_changes.at[index].set(
            jnp.where(admissible, gradient_change, metric.gradient_changes[index])
        ),
        reciprocal_curvatures=metric.reciprocal_curvatures.at[index].set(
            jnp.where(
                admissible,
                1.0 / safe_curvature,
                metric.reciprocal_curvatures[index],
            )
        ),
        pair_valid=metric.pair_valid.at[index].set(
            admissible | metric.pair_valid[index]
        ),
        insertion_index=metric.insertion_index + admissible.astype(jnp.int32),
    )


def quasi_newton_metric_scaling(metric: QuasiNewtonMetric) -> jax.Array:
    """Return ``(s.y)/(y.y)`` of the newest admitted pair, or one when empty."""

    memory = metric.steps.shape[0]
    newest = jnp.mod(metric.insertion_index - 1, memory)
    gradient_change = metric.gradient_changes[newest]
    curvature = metric.steps[newest] @ gradient_change
    gradient_change_squared = gradient_change @ gradient_change
    return jnp.where(
        metric.pair_valid[newest],
        curvature / gradient_change_squared,
        jnp.ones_like(curvature),
    )


def apply_quasi_newton_metric(
    metric: QuasiNewtonMetric,
    vector: jax.Array,
) -> jax.Array:
    """Apply the two-loop recursion, which is the identity for an empty metric."""

    memory = metric.steps.shape[0]
    scaling = quasi_newton_metric_scaling(metric)

    def newest_to_oldest(
        offset: int, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        residual, coefficients = carry
        index = jnp.mod(metric.insertion_index - 1 - offset, memory)
        coefficient = jnp.where(
            metric.pair_valid[index],
            metric.reciprocal_curvatures[index] * (metric.steps[index] @ residual),
            jnp.zeros((), dtype=vector.dtype),
        )
        return (
            residual - coefficient * metric.gradient_changes[index],
            coefficients.at[index].set(coefficient),
        )

    residual, coefficients = jax.lax.fori_loop(
        0,
        memory,
        newest_to_oldest,
        (vector, jnp.zeros((memory,), dtype=vector.dtype)),
    )

    def oldest_to_newest(offset: int, applied: jax.Array) -> jax.Array:
        index = jnp.mod(metric.insertion_index + offset, memory)
        correction = coefficients[index] - jnp.where(
            metric.pair_valid[index],
            metric.reciprocal_curvatures[index]
            * (metric.gradient_changes[index] @ applied),
            jnp.zeros((), dtype=vector.dtype),
        )
        return applied + correction * metric.steps[index]

    return jax.lax.fori_loop(0, memory, oldest_to_newest, scaling * residual)


def valid_pair_count(metric: QuasiNewtonMetric) -> jax.Array:
    """Count admitted pairs currently held by the ring."""

    return jnp.sum(metric.pair_valid.astype(jnp.int32))


__all__ = (
    "QuasiNewtonMetric",
    "apply_quasi_newton_metric",
    "curvature_pair_admissible",
    "empty_quasi_newton_metric",
    "insert_curvature_pair",
    "quasi_newton_metric_scaling",
    "valid_pair_count",
)
