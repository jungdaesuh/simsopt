"""Trust-region adapter for the shared L-BFGS-B inverse-curvature store.

The correction-pair layout, the rollover bookkeeping and the two-loop recursion
all live in ``private/_lbfgsb_scipy.py``, the scipy L-BFGS-B parity port.  This
module owns only what a trust-region caller adds on top: which accepted steps
become correction pairs, and how the stored inverse Hessian is handed to a
preconditioned solve as an operator on an arbitrary vector.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .private._lbfgsb_scipy import (
    LbfgsbInverseHessianHistory,
    lbfgsb_apply_inverse_hessian,
    lbfgsb_matupd,
)

# A pair is admitted only when its curvature is positive relative to the pair's
# own magnitude; the floor also rejects nonfinite pairs, since every comparison
# against NaN is False.  L-BFGS-B decides admission in its caller too
# (``mainlb``'s ``skip_update``), but with the line-search directional
# derivatives a trust-region step does not have.
_CURVATURE_ADMISSION_FLOOR = 1.0e-10


class QuasiNewtonMetric(NamedTuple):
    """Shared correction store with the bookkeeping ``lbfgsb_matupd`` maintains.

    ``inverse_hessian_theta`` is L-BFGS-B's ``theta``: the initial inverse
    Hessian is ``I / theta``, so an empty store applies as the identity.
    """

    history: LbfgsbInverseHessianHistory
    correction_gram: jax.Array
    step_gram: jax.Array
    tail_index: jax.Array
    head_index: jax.Array
    admitted_updates: jax.Array
    inverse_hessian_theta: jax.Array


def empty_quasi_newton_metric(
    memory: int,
    dimension: int,
    dtype: jnp.dtype,
) -> QuasiNewtonMetric:
    """Return a store holding no pair, which applies as the exact identity."""

    return QuasiNewtonMetric(
        history=LbfgsbInverseHessianHistory(
            s=jnp.zeros((memory, dimension), dtype=dtype),
            y=jnp.zeros((memory, dimension), dtype=dtype),
            n_corrs=jnp.asarray(0, dtype=jnp.int32),
        ),
        correction_gram=jnp.zeros((memory, memory), dtype=dtype),
        step_gram=jnp.zeros((memory, memory), dtype=dtype),
        tail_index=jnp.asarray(0, dtype=jnp.int32),
        head_index=jnp.asarray(0, dtype=jnp.int32),
        admitted_updates=jnp.asarray(0, dtype=jnp.int32),
        inverse_hessian_theta=jnp.asarray(1.0, dtype=dtype),
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
    """Store one admissible pair through the shared L-BFGS-B update.

    The accepted step is already the whole step, so the update runs at unit
    line-search scale.
    """

    admissible = curvature_pair_admissible(step, gradient_change)
    update = lbfgsb_matupd(
        metric.history.s,
        metric.history.y,
        metric.correction_gram,
        metric.step_gram,
        step,
        gradient_change,
        metric.tail_index,
        metric.admitted_updates + 1,
        metric.history.n_corrs,
        metric.head_index,
        gradient_change @ gradient_change,
        step @ gradient_change,
        jnp.ones((), dtype=step.dtype),
        step @ step,
    )
    updated = QuasiNewtonMetric(
        history=LbfgsbInverseHessianHistory(
            s=update.ws, y=update.wy, n_corrs=update.col
        ),
        correction_gram=update.sy,
        step_gram=update.ss,
        tail_index=update.itail,
        head_index=update.head,
        admitted_updates=metric.admitted_updates + 1,
        inverse_hessian_theta=update.theta,
    )
    return jax.tree.map(
        lambda admitted, retained: jnp.where(admissible, admitted, retained),
        updated,
        metric,
    )


def apply_quasi_newton_metric(
    metric: QuasiNewtonMetric,
    vector: jax.Array,
) -> jax.Array:
    """Apply the stored inverse Hessian, which is the identity when empty."""

    return lbfgsb_apply_inverse_hessian(
        metric.history.s,
        metric.history.y,
        metric.head_index,
        metric.history.n_corrs,
        metric.inverse_hessian_theta,
        vector,
    )


def valid_pair_count(metric: QuasiNewtonMetric) -> jax.Array:
    """Count live correction pairs currently held by the store."""

    return metric.history.n_corrs


__all__ = (
    "QuasiNewtonMetric",
    "apply_quasi_newton_metric",
    "curvature_pair_admissible",
    "empty_quasi_newton_metric",
    "insert_curvature_pair",
    "valid_pair_count",
)
