"""Shared surface-DOF dispatch helpers.

Spec-kind dispatch from a DOF vector to surface geometry (gamma + tangents,
volume). Used by both the QFM solver (:mod:`simsopt_jax.core.qfm_solver`) and
the surface objectives (:mod:`simsopt_jax_adapters.geo.surface_objectives`), which
previously each carried a byte-identical copy.
"""

from __future__ import annotations

from dataclasses import replace

from ._math_utils import as_jax_float64
from .specs import surface_spec_kind
from .surface_fourier import (
    surface_xyz_fourier_gamma_from_spec,
    surface_xyz_fourier_gammadash1_from_spec,
    surface_xyz_fourier_gammadash2_from_spec,
    surface_xyz_fourier_volume_from_spec,
    surface_xyz_tensor_fourier_gamma_from_spec,
    surface_xyz_tensor_fourier_gammadash1_from_spec,
    surface_xyz_tensor_fourier_gammadash2_from_spec,
    surface_xyz_tensor_fourier_volume_from_spec,
)
from .surface_rzfourier import (
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_gammadash1_from_dofs,
    surface_rz_fourier_gammadash2_from_dofs,
    surface_rz_fourier_volume_from_dofs,
)

__all__ = [
    "surface_spec_with_dofs",
    "surface_gamma_tangents_from_dofs",
    "surface_volume_from_dofs",
]


def surface_spec_with_dofs(spec, dofs):
    return replace(spec, dofs=as_jax_float64(dofs))


def surface_gamma_tangents_from_dofs(spec, dofs):
    dofs = as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return (
            surface_rz_fourier_gamma_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash1_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash2_from_dofs(spec, dofs),
        )
    spec_with_dofs = surface_spec_with_dofs(spec, dofs)
    if kind == "xyz_fourier":
        return (
            surface_xyz_fourier_gamma_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash1_from_spec(spec_with_dofs),
            surface_xyz_fourier_gammadash2_from_spec(spec_with_dofs),
        )
    if kind == "xyz_tensor_fourier":
        return (
            surface_xyz_tensor_fourier_gamma_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash1_from_spec(spec_with_dofs),
            surface_xyz_tensor_fourier_gammadash2_from_spec(spec_with_dofs),
        )
    raise TypeError(f"Unsupported surface spec kind {kind!r}.")


def surface_volume_from_dofs(spec, dofs):
    dofs = as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return surface_rz_fourier_volume_from_dofs(spec, dofs)
    spec_with_dofs = surface_spec_with_dofs(spec, dofs)
    if kind == "xyz_fourier":
        return surface_xyz_fourier_volume_from_spec(spec_with_dofs)
    if kind == "xyz_tensor_fourier":
        return surface_xyz_tensor_fourier_volume_from_spec(spec_with_dofs)
    raise TypeError(f"Unsupported surface spec kind {kind!r}.")
