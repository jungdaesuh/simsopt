"""Immutable JAX surface specs from simsopt surface objects."""

from __future__ import annotations

from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.core.specs import (
    make_surface_xyz_fourier_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs

__all__ = ["surface_spec_from_surface"]


def surface_spec_from_surface(surface):
    """The immutable spec of a surface: its own ``surface_spec()`` when it has one.

    Plain simsopt ``SurfaceRZFourier``, ``SurfaceXYZFourier`` and
    ``SurfaceXYZTensorFourier`` objects are converted from their dofs and
    quadrature points; any other surface type is refused.
    """
    surface_spec_fn = getattr(surface, "surface_spec", None)
    if callable(surface_spec_fn):
        return surface_spec_fn()

    surface_type = type(surface).__name__
    if surface_type == "SurfaceRZFourier":
        return surface_rz_fourier_spec_from_dofs(
            _as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        )
    if surface_type == "SurfaceXYZFourier":
        return make_surface_xyz_fourier_spec(
            dofs=_as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
        )
    if surface_type == "SurfaceXYZTensorFourier":
        return make_surface_xyz_tensor_fourier_spec(
            dofs=_as_jax_float64(surface.get_dofs()),
            quadpoints_phi=_as_jax_float64(surface.quadpoints_phi),
            quadpoints_theta=_as_jax_float64(surface.quadpoints_theta),
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            clamped_dims=tuple(getattr(surface, "clamped_dims", (False, False, False))),
        )
    raise NotImplementedError(
        "JAX surface adapters require an explicit spec builder for "
        f"{surface_type}; supported: SurfaceRZFourier, SurfaceXYZFourier, "
        "SurfaceXYZTensorFourier, or any surface exposing surface_spec()."
    )
