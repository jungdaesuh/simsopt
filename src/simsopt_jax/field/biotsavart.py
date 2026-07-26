"""Compatibility shim for the pure JAX Biot-Savart kernels.

Elemental kernels live in ``simsopt_jax.core.biotsavart`` and grouped dispatch
lives in ``simsopt_jax.core.field``. This file preserves the historical import
contract.
"""

from simsopt_jax.core.biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_B_vjp,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
    biot_savart_d2A_by_dXdX,
    biot_savart_d2B_by_dXdX,
    group_coil_data,
    invalidate_kernel_cache,
)
from simsopt_jax.core.field import grouped_biot_savart_A, grouped_biot_savart_B

__all__ = (
    "biot_savart_A",
    "biot_savart_B",
    "biot_savart_B_and_dB",
    "biot_savart_B_vjp",
    "biot_savart_dA_by_dX",
    "biot_savart_dB_by_dX",
    "biot_savart_d2A_by_dXdX",
    "biot_savart_d2B_by_dXdX",
    "group_coil_data",
    "grouped_biot_savart_A",
    "grouped_biot_savart_B",
    "invalidate_kernel_cache",
)
