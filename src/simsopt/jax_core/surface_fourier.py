"""Pure JAX non-RZ surface geometry built on immutable specs."""

from __future__ import annotations

import jax.numpy as jnp

from ..geo.surface_fourier_jax import (
    surface_area,
    surface_gamma_from_dofs as _surface_xyz_tensor_gamma_from_dofs,
    surface_gammadash1_from_dofs as _surface_xyz_tensor_gammadash1_from_dofs,
    surface_gammadash1dash1_from_dofs as _surface_xyz_tensor_gammadash1dash1_from_dofs,
    surface_gammadash1dash2_from_dofs as _surface_xyz_tensor_gammadash1dash2_from_dofs,
    surface_gammadash2_from_dofs as _surface_xyz_tensor_gammadash2_from_dofs,
    surface_gammadash2dash2_from_dofs as _surface_xyz_tensor_gammadash2dash2_from_dofs,
    surface_unitnormal_from_dofs as _surface_xyz_tensor_unitnormal_from_dofs,
    surface_volume,
    surface_xyzfourier_gamma_from_dofs as _surface_xyz_fourier_gamma_from_dofs,
    surface_xyzfourier_gammadash1_from_dofs as _surface_xyz_fourier_gammadash1_from_dofs,
    surface_xyzfourier_gammadash1dash1_from_dofs as _surface_xyz_fourier_gammadash1dash1_from_dofs,
    surface_xyzfourier_gammadash1dash2_from_dofs as _surface_xyz_fourier_gammadash1dash2_from_dofs,
    surface_xyzfourier_gammadash2_from_dofs as _surface_xyz_fourier_gammadash2_from_dofs,
    surface_xyzfourier_gammadash2dash2_from_dofs as _surface_xyz_fourier_gammadash2dash2_from_dofs,
    surface_xyzfourier_unitnormal_from_dofs as _surface_xyz_fourier_unitnormal_from_dofs,
    surface_xyzfourier_volume_from_dofs as _surface_xyz_fourier_volume_from_dofs,
)
from .specs import SurfaceXYZFourierSpec, SurfaceXYZTensorFourierSpec


def surface_xyz_fourier_gamma_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_gamma_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_gammadash1_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_gammadash1_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_gammadash2_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_gammadash2_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_gammadash1dash1_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_gammadash1dash1_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_gammadash1dash2_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_gammadash1dash2_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_gammadash2dash2_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_gammadash2dash2_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_normal_from_spec(spec: SurfaceXYZFourierSpec):
    return jnp.cross(
        surface_xyz_fourier_gammadash1_from_spec(spec),
        surface_xyz_fourier_gammadash2_from_spec(spec),
    )


def surface_xyz_fourier_unitnormal_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_unitnormal_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_area_from_spec(spec: SurfaceXYZFourierSpec):
    return surface_area(surface_xyz_fourier_normal_from_spec(spec))


def surface_xyz_fourier_volume_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_xyz_fourier_volume_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def _scatter_indices_or_none(spec: SurfaceXYZTensorFourierSpec):
    if spec.stellsym:
        return spec.scatter_indices
    return None


def _clamped_dims_or_default(spec: SurfaceXYZTensorFourierSpec):
    # ``SurfaceXYZTensorFourierSpec.clamped_dims`` is a declared dataclass
    # field with a ``(False, False, False)`` default (specs.py), so the
    # attribute is always present on a correctly-typed spec. Direct access
    # surfaces a clear ``AttributeError`` if a wrong type ever leaks in,
    # rather than silently returning the unclamped default.
    return spec.clamped_dims


def surface_xyz_tensor_fourier_gamma_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return _surface_xyz_tensor_gamma_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_gammadash1_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return _surface_xyz_tensor_gammadash1_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_gammadash2_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return _surface_xyz_tensor_gammadash2_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_gammadash1dash1_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
):
    return _surface_xyz_tensor_gammadash1dash1_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_gammadash1dash2_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
):
    return _surface_xyz_tensor_gammadash1dash2_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_gammadash2dash2_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
):
    return _surface_xyz_tensor_gammadash2dash2_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_normal_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return jnp.cross(
        surface_xyz_tensor_fourier_gammadash1_from_spec(spec),
        surface_xyz_tensor_fourier_gammadash2_from_spec(spec),
    )


def surface_xyz_tensor_fourier_unitnormal_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return _surface_xyz_tensor_unitnormal_from_dofs(
        spec.dofs,
        spec.quadpoints_phi,
        spec.quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_area_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return surface_area(surface_xyz_tensor_fourier_normal_from_spec(spec))


def surface_xyz_tensor_fourier_volume_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return surface_volume(
        surface_xyz_tensor_fourier_gamma_from_spec(spec),
        surface_xyz_tensor_fourier_normal_from_spec(spec),
    )
