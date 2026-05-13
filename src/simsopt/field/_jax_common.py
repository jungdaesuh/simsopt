"""Shared host/device helpers for the per-class JAX MagneticField wrappers."""

from __future__ import annotations

import jax
import numpy as np

from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64


def points_device(points: np.ndarray) -> jax.Array:
    """Stage host points to a JAX float64 device array via ``jax.device_put``.

    The CPU ``MagneticField`` cache hands wrappers a contiguous NumPy array.
    Routing the staging through :func:`simsopt.jax_core._math_utils.as_jax_float64`
    forces :func:`jax.device_put`, which is explicit and allowed under
    ``transfer_guard("disallow")``. The result is reused for every kernel call
    until ``set_points`` invalidates the cache.
    """

    return _as_jax_float64(points)
