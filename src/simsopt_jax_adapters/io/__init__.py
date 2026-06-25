"""Adapter-owned JSON helpers for explicit legacy/JAX bridge artifacts."""

from .specs import (
    load_specs,
    save_biot_savart_spec,
    save_surface_rz_fourier_spec,
    save_surface_xyz_fourier_spec,
    save_surface_xyz_tensor_fourier_spec,
)

__all__ = (
    "load_specs",
    "save_biot_savart_spec",
    "save_surface_rz_fourier_spec",
    "save_surface_xyz_fourier_spec",
    "save_surface_xyz_tensor_fourier_spec",
)
