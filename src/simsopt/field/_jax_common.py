"""Shared host/device helpers for the per-class JAX MagneticField wrappers."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .._core.jax_host_boundary import host_array
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64

if TYPE_CHECKING:
    import jax


def points_device(points: np.ndarray) -> jax.Array:
    """Stage host ``(npoints, 3)`` coordinates to a JAX float64 device array.

    The explicit ``device_put`` boundary stays legal under
    ``transfer_guard("disallow")`` and is reused until ``set_points`` refreshes
    the CPU ``MagneticField`` cache.
    """

    return _as_jax_float64(points)


# Field wrappers keep this documented host-cache boundary alias.
def host_cache_array(value, *, dtype) -> np.ndarray:
    """Materialize a JAX result into an upstream NumPy cache buffer."""
    return host_array(value, dtype=dtype)
