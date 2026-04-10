"""Pure fixed-surface flux helpers built on immutable specs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._device_scalars import float_scalar, two_pi
from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_runtime_float64 as _as_runtime_float64,
)
from .field import grouped_biot_savart_B_from_spec
from .specs import (
    FieldEvalSpec,
    FixedSurfaceFluxSpec,
    make_field_eval_spec,
    make_fixed_surface_flux_spec,
)
from .surface_rzfourier import (
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_normal_from_spec,
)
from ..objectives.integral_bdotn_jax import (
    integral_BdotN as integral_BdotN_jax,
    residual_BdotN as residual_BdotN_jax,
)


def _fixed_surface_target_array(normal, target):
    if target is None:
        zero_target = jnp.sum(normal, axis=-1)
        return zero_target - zero_target
    return _as_jax_float64(target)


def build_fourier_basis(quadpoints_jax, order):
    """Precompute the CurveXYZFourier basis matrix and its derivative."""
    zeros = quadpoints_jax - quadpoints_jax
    basis_columns = [jnp.exp(zeros)]
    dbasis_columns = [zeros]
    angle_scale = two_pi(quadpoints_jax)
    for j in range(1, order + 1):
        mode = float_scalar(j, quadpoints_jax)
        arg = angle_scale * mode * quadpoints_jax
        s = jnp.sin(arg)
        c = jnp.cos(arg)
        basis_columns.extend((s, c))
        dbasis_columns.extend((angle_scale * mode * c, -(angle_scale * mode) * s))

    return jnp.stack(basis_columns, axis=1), jnp.stack(dbasis_columns, axis=1)


def fixed_surface_flux_integral_from_B(B, flux_spec: FixedSurfaceFluxSpec):
    Bcoil = B.reshape((flux_spec.nphi, flux_spec.ntheta, 3))
    return integral_BdotN_jax(
        Bcoil,
        _as_runtime_float64(flux_spec.target, reference=B),
        _as_runtime_float64(flux_spec.normal, reference=B),
        flux_spec.definition,
    )


def fixed_surface_flux_residual_from_B(B, flux_spec: FixedSurfaceFluxSpec):
    Bcoil = B.reshape((flux_spec.nphi, flux_spec.ntheta, 3))
    return residual_BdotN_jax(
        Bcoil,
        _as_runtime_float64(flux_spec.target, reference=B),
        _as_runtime_float64(flux_spec.normal, reference=B),
        flux_spec.definition,
    )


def fixed_surface_flux_integral(
    coil_spec,
    flux_spec: FixedSurfaceFluxSpec,
):
    B = grouped_biot_savart_B_from_spec(flux_spec.points, coil_spec)
    return fixed_surface_flux_integral_from_B(B, flux_spec)


def fixed_surface_geometry_from_surface(surface):
    surface_spec_fn = getattr(surface, "surface_spec", None)
    if callable(surface_spec_fn):
        surface_spec = surface_spec_fn()
        gamma = surface_rz_fourier_gamma_from_spec(surface_spec)
        normal = surface_rz_fourier_normal_from_spec(surface_spec)
        return gamma, normal
    raise NotImplementedError(
        "SquaredFluxJAX fixed-surface setup requires a surface exposing "
        "surface_spec()."
    )


def fixed_surface_flux_specs_from_surface(
    surface,
    *,
    target=None,
    definition: str,
) -> tuple[FieldEvalSpec, FixedSurfaceFluxSpec]:
    gamma, normal = fixed_surface_geometry_from_surface(surface)
    target_jax = _fixed_surface_target_array(normal, target)
    field_eval_spec = make_field_eval_spec(gamma.reshape((-1, 3)))
    flux_spec = make_fixed_surface_flux_spec(
        points=field_eval_spec.points,
        normal=normal,
        target=target_jax,
        definition=definition,
    )
    return field_eval_spec, flux_spec
