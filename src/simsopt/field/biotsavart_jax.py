"""Compatibility shim for the pure JAX Biot-Savart kernels.

The implementation lives in ``simsopt.jax_core.biotsavart``.
This file preserves the historical import and direct-path loader contract.
"""

from pathlib import Path
import sys


def _ensure_src_root_on_path() -> None:
    src_root = str(Path(__file__).resolve().parents[2])
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_ensure_src_root_on_path()

from simsopt.jax_core.biotsavart import (  # noqa: E402
    _coil_chunk_size,
    _biot_savart_A_one_point_dense,
    _biot_savart_one_point_dense,
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_B_vjp,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
    group_coil_data,
    grouped_biot_savart_A,
    grouped_biot_savart_B,
)

__all__ = (
    "_biot_savart_A_one_point_dense",
    "_biot_savart_one_point_dense",
    "_coil_chunk_size",
    "biot_savart_A",
    "biot_savart_B",
    "biot_savart_B_and_dB",
    "biot_savart_B_vjp",
    "biot_savart_dA_by_dX",
    "biot_savart_dB_by_dX",
    "group_coil_data",
    "grouped_biot_savart_A",
    "grouped_biot_savart_B",
)
