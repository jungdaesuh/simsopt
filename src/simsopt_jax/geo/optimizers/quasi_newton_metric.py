"""Projected-optimizer adapter for the shared L-BFGS-B inverse-curvature store.

The correction-pair layout, the rollover bookkeeping and the two-loop recursion
all live in ``private/_lbfgsb_scipy.py``, the scipy L-BFGS-B parity port.  This
module owns only what a projected caller adds on top: which accepted steps
become correction pairs, how the stored inverse Hessian is handed to a solve --
a search direction, or a preconditioner -- as an operator on an arbitrary
vector, and how a store built in one tangent space is carried into another.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.backend.dtypes import explicit_device_array

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
    """Return a store holding no pair, which applies as the exact identity.

    The store is minted at an EXPLICIT host-to-device boundary rather than by
    ``jnp.zeros``: an empty store is built on the host, once, outside any
    traced kernel, and a strict transfer guard refuses the implicit placement
    ``jnp.zeros`` performs there.  Same values, same dtypes, same shapes.
    """

    return QuasiNewtonMetric(
        history=LbfgsbInverseHessianHistory(
            s=explicit_device_array(np.zeros((memory, dimension), dtype), dtype=dtype),
            y=explicit_device_array(np.zeros((memory, dimension), dtype), dtype=dtype),
            n_corrs=explicit_device_array(0, dtype=jnp.int32),
        ),
        correction_gram=explicit_device_array(
            np.zeros((memory, memory), dtype), dtype=dtype
        ),
        step_gram=explicit_device_array(np.zeros((memory, memory), dtype), dtype=dtype),
        tail_index=explicit_device_array(0, dtype=jnp.int32),
        head_index=explicit_device_array(0, dtype=jnp.int32),
        admitted_updates=explicit_device_array(0, dtype=jnp.int32),
        inverse_hessian_theta=explicit_device_array(1.0, dtype=dtype),
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


class TransportedMetric(NamedTuple):
    """A store carried into another tangent space, with what that cost."""

    metric: QuasiNewtonMetric
    masked_pairs: jax.Array
    live_pairs: jax.Array


def transport_quasi_newton_metric(
    metric: QuasiNewtonMetric,
    transport: Callable[[jax.Array], jax.Array],
) -> TransportedMetric:
    """Carry every stored pair through ``transport`` and re-admit the survivors.

    On a manifold each pair was formed in the tangent space of its own iterate,
    so applying the store at a later point compares vectors that live in
    different spaces.  ``transport`` maps a ``(memory, dimension)`` batch into
    the current tangent space; a pair whose curvature no longer clears the
    admission floor there was carrying curvature that the manifold's rotation
    invented, and is zeroed.

    A zeroed pair is an exact no-op in the two-loop recursion -- its ``s.y`` is
    zero, which the recursion already guards, making ``rho`` multiply a zero
    inner product -- so masking needs no change to the shared correction store.
    """

    transported_steps = transport(metric.history.s)
    transported_changes = transport(metric.history.y)
    curvatures = jnp.sum(transported_steps * transported_changes, axis=1)
    magnitudes = jnp.linalg.norm(transported_steps, axis=1) * jnp.linalg.norm(
        transported_changes, axis=1
    )
    admissible = curvatures > _CURVATURE_ADMISSION_FLOOR * magnitudes
    live = jnp.arange(metric.history.s.shape[0]) < metric.history.n_corrs
    keep = (admissible & live)[:, None]
    return TransportedMetric(
        metric=metric._replace(
            history=metric.history._replace(
                s=jnp.where(keep, transported_steps, 0.0),
                y=jnp.where(keep, transported_changes, 0.0),
            )
        ),
        masked_pairs=jnp.sum(live & ~admissible, dtype=jnp.int32),
        live_pairs=jnp.sum(live, dtype=jnp.int32),
    )


def valid_pair_count(metric: QuasiNewtonMetric) -> jax.Array:
    """Count live correction pairs currently held by the store."""

    return metric.history.n_corrs


__all__ = (
    "QuasiNewtonMetric",
    "TransportedMetric",
    "apply_quasi_newton_metric",
    "curvature_pair_admissible",
    "empty_quasi_newton_metric",
    "insert_curvature_pair",
    "transport_quasi_newton_metric",
    "valid_pair_count",
)
