"""Private MPS custom calls for SIMSOPT Biot-Savart kernels."""

from __future__ import annotations

import jax
import numpy as np

SIMSOPT_MPS_BIOT_SAVART_B_GROUP_TARGET = "mps.simsopt_biot_savart_b_group"


def _validate_b_group_shapes(
    points: jax.Array,
    gammas: jax.Array,
    gammadashs: jax.Array,
    currents: jax.Array,
) -> None:
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (npoints, 3)")
    if len(gammas.shape) != 3 or gammas.shape[2] != 3:
        raise ValueError("gammas must have shape (ncoils, nquad, 3)")
    if gammas.shape[1] == 0:
        raise ValueError("gammas must include at least one quadrature point")
    if gammadashs.shape != gammas.shape:
        raise ValueError("gammadashs must have the same shape as gammas")
    if len(currents.shape) != 1 or currents.shape[0] != gammas.shape[0]:
        raise ValueError("currents must have shape (ncoils,)")
    if points.dtype != gammas.dtype or points.dtype != gammadashs.dtype:
        raise ValueError("points, gammas, and gammadashs must share a dtype")
    if points.dtype != currents.dtype:
        raise ValueError("currents must have the same dtype as points")
    if points.dtype != np.dtype(np.float32):
        raise ValueError("points, gammas, gammadashs, and currents must be float32")


def simsopt_mps_biot_savart_b_group(
    points: jax.Array,
    gammas: jax.Array,
    gammadashs: jax.Array,
    currents: jax.Array,
) -> jax.Array:
    """Return one grouped Biot-Savart B-field block through jax-mps."""
    _validate_b_group_shapes(points, gammas, gammadashs, currents)
    return jax.ffi.ffi_call(
        SIMSOPT_MPS_BIOT_SAVART_B_GROUP_TARGET,
        jax.ShapeDtypeStruct(points.shape, points.dtype),
        vmap_method="broadcast_all",
        custom_call_api_version=4,
    )(points, gammas, gammadashs, currents)
