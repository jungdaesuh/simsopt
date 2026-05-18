"""Pure JAX non-RZ surface geometry built on immutable specs."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from .surface_fourier_kernels import (
    surface_gamma_from_dofs as _surface_xyz_tensor_gamma_from_dofs,
    surface_gammadash1_from_dofs as _surface_xyz_tensor_gammadash1_from_dofs,
    surface_gammadash1dash1_from_dofs as _surface_xyz_tensor_gammadash1dash1_from_dofs,
    surface_gammadash1dash1_lin_from_dofs as _surface_xyz_tensor_gammadash1dash1_lin_from_dofs,
    surface_gammadash1dash1dash1_lin_from_dofs as _surface_xyz_tensor_gammadash1dash1dash1_lin_from_dofs,
    surface_gammadash1dash1dash2_lin_from_dofs as _surface_xyz_tensor_gammadash1dash1dash2_lin_from_dofs,
    surface_gammadash1dash2_from_dofs as _surface_xyz_tensor_gammadash1dash2_from_dofs,
    surface_gammadash1dash2_lin_from_dofs as _surface_xyz_tensor_gammadash1dash2_lin_from_dofs,
    surface_gammadash1dash2dash2_lin_from_dofs as _surface_xyz_tensor_gammadash1dash2dash2_lin_from_dofs,
    surface_gammadash2_from_dofs as _surface_xyz_tensor_gammadash2_from_dofs,
    surface_gammadash2dash2_from_dofs as _surface_xyz_tensor_gammadash2dash2_from_dofs,
    surface_gammadash2dash2_lin_from_dofs as _surface_xyz_tensor_gammadash2dash2_lin_from_dofs,
    surface_gammadash2dash2dash2_lin_from_dofs as _surface_xyz_tensor_gammadash2dash2dash2_lin_from_dofs,
    surface_unitnormal_from_dofs as _surface_xyz_tensor_unitnormal_from_dofs,
    surface_xyzfourier_gamma_from_dofs as _surface_xyz_fourier_gamma_from_dofs,
    surface_xyzfourier_gammadash1_from_dofs as _surface_xyz_fourier_gammadash1_from_dofs,
    surface_xyzfourier_gammadash1dash1_from_dofs as _surface_xyz_fourier_gammadash1dash1_from_dofs,
    surface_xyzfourier_gammadash1dash1_lin_from_dofs as _surface_xyz_fourier_gammadash1dash1_lin_from_dofs,
    surface_xyzfourier_gammadash1dash1dash1_lin_from_dofs as _surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs,
    surface_xyzfourier_gammadash1dash1dash2_lin_from_dofs as _surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs,
    surface_xyzfourier_gammadash1dash2_from_dofs as _surface_xyz_fourier_gammadash1dash2_from_dofs,
    surface_xyzfourier_gammadash1dash2_lin_from_dofs as _surface_xyz_fourier_gammadash1dash2_lin_from_dofs,
    surface_xyzfourier_gammadash1dash2dash2_lin_from_dofs as _surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs,
    surface_xyzfourier_gammadash2_from_dofs as _surface_xyz_fourier_gammadash2_from_dofs,
    surface_xyzfourier_gammadash2dash2_from_dofs as _surface_xyz_fourier_gammadash2dash2_from_dofs,
    surface_xyzfourier_gammadash2dash2_lin_from_dofs as _surface_xyz_fourier_gammadash2dash2_lin_from_dofs,
    surface_xyzfourier_gammadash2dash2dash2_lin_from_dofs as _surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs,
    surface_xyzfourier_unitnormal_from_dofs as _surface_xyz_fourier_unitnormal_from_dofs,
    surface_xyzfourier_volume_from_dofs as _surface_xyz_fourier_volume_from_dofs,
)
from .specs import SurfaceXYZFourierSpec, SurfaceXYZTensorFourierSpec
from .surface_integrals import surface_area, surface_volume


def _spec_with_dofs(spec, dofs):
    return replace(spec, dofs=jnp.asarray(dofs, dtype=spec.dofs.dtype))


def _first_fund_form_from_tangents(gammadash1, gammadash2):
    return jnp.stack(
        [
            jnp.sum(gammadash1 * gammadash1, axis=-1),
            jnp.sum(gammadash1 * gammadash2, axis=-1),
            jnp.sum(gammadash2 * gammadash2, axis=-1),
        ],
        axis=-1,
    )


def _second_fund_form_from_derivatives(
    unitnormal,
    gammadash1dash1,
    gammadash1dash2,
    gammadash2dash2,
):
    return jnp.stack(
        [
            jnp.sum(unitnormal * gammadash1dash1, axis=-1),
            jnp.sum(unitnormal * gammadash1dash2, axis=-1),
            jnp.sum(unitnormal * gammadash2dash2, axis=-1),
        ],
        axis=-1,
    )


def _surface_curvatures_from_forms(first, second):
    e = first[:, :, 0]
    f = first[:, :, 1]
    g = first[:, :, 2]
    ell = second[:, :, 0]
    m = second[:, :, 1]
    n = second[:, :, 2]
    denom = e * g - f * f
    mean_curvature = (ell * g - 2.0 * f * m + n * e) / (2.0 * denom)
    gaussian_curvature = (ell * n - m * m) / denom
    principal_offset = jnp.sqrt(mean_curvature * mean_curvature - gaussian_curvature)
    return jnp.stack(
        [
            mean_curvature,
            gaussian_curvature,
            mean_curvature + principal_offset,
            mean_curvature - principal_offset,
        ],
        axis=-1,
    )


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


def _surface_xyz_fourier_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
    kernel,
):
    return kernel(
        spec.dofs,
        quadpoints_phi,
        quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        spec.scatter_indices,
        spec.coeff_template,
    )


def surface_xyz_fourier_gammadash1dash1_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash1dash1_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash1dash2_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash1dash2_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash2dash2_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash2dash2_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash1dash1dash1_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash1dash1dash2_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash1dash2dash2_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash2dash2dash2_lin_from_spec(
    spec: SurfaceXYZFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs,
    )


def surface_xyz_fourier_gammadash1dash1_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash1dash1_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_fourier_gammadash1dash2_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash1dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_fourier_gammadash2dash2_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash2dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash1dash1dash1_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash1dash1dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash1dash2dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_fourier_gammadash2dash2dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
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


def surface_xyz_fourier_first_fund_form_from_spec(spec: SurfaceXYZFourierSpec):
    return _first_fund_form_from_tangents(
        surface_xyz_fourier_gammadash1_from_spec(spec),
        surface_xyz_fourier_gammadash2_from_spec(spec),
    )


def surface_xyz_fourier_second_fund_form_from_spec(spec: SurfaceXYZFourierSpec):
    return _second_fund_form_from_derivatives(
        surface_xyz_fourier_unitnormal_from_spec(spec),
        surface_xyz_fourier_gammadash1dash1_from_spec(spec),
        surface_xyz_fourier_gammadash1dash2_from_spec(spec),
        surface_xyz_fourier_gammadash2dash2_from_spec(spec),
    )


def surface_xyz_fourier_surface_curvatures_from_spec(spec: SurfaceXYZFourierSpec):
    return _surface_curvatures_from_forms(
        surface_xyz_fourier_first_fund_form_from_spec(spec),
        surface_xyz_fourier_second_fund_form_from_spec(spec),
    )


def surface_xyz_fourier_first_fund_form_from_dofs(spec: SurfaceXYZFourierSpec, dofs):
    return surface_xyz_fourier_first_fund_form_from_spec(_spec_with_dofs(spec, dofs))


def surface_xyz_fourier_second_fund_form_from_dofs(spec: SurfaceXYZFourierSpec, dofs):
    return surface_xyz_fourier_second_fund_form_from_spec(_spec_with_dofs(spec, dofs))


def surface_xyz_fourier_surface_curvatures_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
):
    return surface_xyz_fourier_surface_curvatures_from_spec(_spec_with_dofs(spec, dofs))


def surface_xyz_fourier_dfirst_fund_form_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
):
    return jax.jacobian(lambda x: surface_xyz_fourier_first_fund_form_from_dofs(spec, x))(
        jnp.asarray(dofs, dtype=spec.dofs.dtype)
    )


def surface_xyz_fourier_dsecond_fund_form_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
):
    return jax.jacobian(
        lambda x: surface_xyz_fourier_second_fund_form_from_dofs(spec, x)
    )(jnp.asarray(dofs, dtype=spec.dofs.dtype))


def surface_xyz_fourier_dsurface_curvatures_from_dofs(
    spec: SurfaceXYZFourierSpec,
    dofs,
):
    return jax.jacobian(
        lambda x: surface_xyz_fourier_surface_curvatures_from_dofs(spec, x)
    )(jnp.asarray(dofs, dtype=spec.dofs.dtype))


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


def _surface_xyz_tensor_fourier_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
    kernel,
):
    return kernel(
        spec.dofs,
        quadpoints_phi,
        quadpoints_theta,
        spec.mpol,
        spec.ntor,
        spec.nfp,
        spec.stellsym,
        _scatter_indices_or_none(spec),
        clamped_dims=_clamped_dims_or_default(spec),
    )


def surface_xyz_tensor_fourier_gammadash1dash1_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash1dash1_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash1dash2_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash1dash2_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash2dash2_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash2dash2_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash1dash1dash1_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash1dash1dash2_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash1dash2dash2_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
    quadpoints_phi,
    quadpoints_theta,
):
    return _surface_xyz_tensor_fourier_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        _surface_xyz_tensor_gammadash2dash2dash2_lin_from_dofs,
    )


def surface_xyz_tensor_fourier_gammadash1dash1_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash1dash1_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_tensor_fourier_gammadash1dash2_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash1dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_tensor_fourier_gammadash2dash2_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash2dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
    quadpoints_phi,
    quadpoints_theta,
):
    return surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_spec(
        _spec_with_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
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


def surface_xyz_tensor_fourier_first_fund_form_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
):
    return _first_fund_form_from_tangents(
        surface_xyz_tensor_fourier_gammadash1_from_spec(spec),
        surface_xyz_tensor_fourier_gammadash2_from_spec(spec),
    )


def surface_xyz_tensor_fourier_second_fund_form_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
):
    return _second_fund_form_from_derivatives(
        surface_xyz_tensor_fourier_unitnormal_from_spec(spec),
        surface_xyz_tensor_fourier_gammadash1dash1_from_spec(spec),
        surface_xyz_tensor_fourier_gammadash1dash2_from_spec(spec),
        surface_xyz_tensor_fourier_gammadash2dash2_from_spec(spec),
    )


def surface_xyz_tensor_fourier_surface_curvatures_from_spec(
    spec: SurfaceXYZTensorFourierSpec,
):
    return _surface_curvatures_from_forms(
        surface_xyz_tensor_fourier_first_fund_form_from_spec(spec),
        surface_xyz_tensor_fourier_second_fund_form_from_spec(spec),
    )


def surface_xyz_tensor_fourier_first_fund_form_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
):
    return surface_xyz_tensor_fourier_first_fund_form_from_spec(
        _spec_with_dofs(spec, dofs)
    )


def surface_xyz_tensor_fourier_second_fund_form_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
):
    return surface_xyz_tensor_fourier_second_fund_form_from_spec(
        _spec_with_dofs(spec, dofs)
    )


def surface_xyz_tensor_fourier_surface_curvatures_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
):
    return surface_xyz_tensor_fourier_surface_curvatures_from_spec(
        _spec_with_dofs(spec, dofs)
    )


def surface_xyz_tensor_fourier_dfirst_fund_form_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
):
    return jax.jacobian(
        lambda x: surface_xyz_tensor_fourier_first_fund_form_from_dofs(spec, x)
    )(jnp.asarray(dofs, dtype=spec.dofs.dtype))


def surface_xyz_tensor_fourier_dsecond_fund_form_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
):
    return jax.jacobian(
        lambda x: surface_xyz_tensor_fourier_second_fund_form_from_dofs(spec, x)
    )(jnp.asarray(dofs, dtype=spec.dofs.dtype))


def surface_xyz_tensor_fourier_dsurface_curvatures_from_dofs(
    spec: SurfaceXYZTensorFourierSpec,
    dofs,
):
    return jax.jacobian(
        lambda x: surface_xyz_tensor_fourier_surface_curvatures_from_dofs(spec, x)
    )(jnp.asarray(dofs, dtype=spec.dofs.dtype))


def surface_xyz_tensor_fourier_area_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return surface_area(surface_xyz_tensor_fourier_normal_from_spec(spec))


def surface_xyz_tensor_fourier_volume_from_spec(spec: SurfaceXYZTensorFourierSpec):
    return surface_volume(
        surface_xyz_tensor_fourier_gamma_from_spec(spec),
        surface_xyz_tensor_fourier_normal_from_spec(spec),
    )
