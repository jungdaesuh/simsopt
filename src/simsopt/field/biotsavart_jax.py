"""Compatibility shim for the pure JAX Biot-Savart kernels.

The implementation lives in ``simsopt.jax_core.biotsavart``.
This file preserves the historical import contract.
"""

from simsopt.jax_core.biotsavart import (
    _biot_savart_A_integrand,
    _biot_savart_B_integrand,
    _one_point_dense,
    _read_tuning_config,
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_B_vjp,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
    biot_savart_d2A_by_dXdX,
    biot_savart_d2B_by_dXdX,
    group_coil_data,
    grouped_biot_savart_A,
    grouped_biot_savart_B,
    invalidate_kernel_cache,
)

__all__ = (
    "_biot_savart_A_integrand",
    "_biot_savart_B_integrand",
    "_one_point_dense",
    "_read_tuning_config",
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
