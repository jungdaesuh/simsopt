"""Public surface-DOF dispatch over immutable JAX surface specs."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeAlias, cast

from ._math_utils import as_jax_float64
from .specs import (
    SurfaceSpec,
    SurfaceXYZFourierSpec,
    SurfaceXYZTensorFourierSpec,
    surface_spec_kind,
)
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
    "surface_gamma_tangents_from_dofs",
    "surface_spec_with_dofs",
    "surface_volume_from_dofs",
]

SurfaceDofSpec: TypeAlias = SurfaceXYZFourierSpec | SurfaceXYZTensorFourierSpec


def surface_spec_with_dofs(spec: SurfaceDofSpec, dofs: object) -> SurfaceDofSpec:
    return replace(spec, dofs=as_jax_float64(dofs))


def surface_gamma_tangents_from_dofs(spec: SurfaceSpec, dofs: object):
    dofs = as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return (
            surface_rz_fourier_gamma_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash1_from_dofs(spec, dofs),
            surface_rz_fourier_gammadash2_from_dofs(spec, dofs),
        )
    spec_with_dofs = surface_spec_with_dofs(cast(SurfaceDofSpec, spec), dofs)
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


def surface_volume_from_dofs(spec: SurfaceSpec, dofs: object):
    dofs = as_jax_float64(dofs)
    kind = surface_spec_kind(spec)
    if kind == "rz_fourier":
        return surface_rz_fourier_volume_from_dofs(spec, dofs)
    spec_with_dofs = surface_spec_with_dofs(cast(SurfaceDofSpec, spec), dofs)
    if kind == "xyz_fourier":
        return surface_xyz_fourier_volume_from_spec(spec_with_dofs)
    if kind == "xyz_tensor_fourier":
        return surface_xyz_tensor_fourier_volume_from_spec(spec_with_dofs)
    raise TypeError(f"Unsupported surface spec kind {kind!r}.")
