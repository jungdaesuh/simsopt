"""Pure fixed-surface flux helpers built on immutable specs."""

from __future__ import annotations

import jax.numpy as jnp

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
from ..objectives.integral_bdotn_jax import integral_BdotN as integral_BdotN_jax


def _fixed_surface_target_array(normal, target):
    nphi, ntheta = normal.shape[:2]
    if target is None:
        return jnp.zeros((nphi, ntheta), dtype=jnp.float64)
    return jnp.asarray(target, dtype=jnp.float64)


def build_fourier_basis(quadpoints_jax, order):
    """Precompute the CurveXYZFourier basis matrix and its derivative."""
    k = 2 * order + 1
    npts = quadpoints_jax.shape[0]
    basis = jnp.zeros((npts, k), dtype=jnp.float64)
    dbasis = jnp.zeros((npts, k), dtype=jnp.float64)

    basis = basis.at[:, 0].set(1.0)
    for j in range(1, order + 1):
        arg = 2.0 * jnp.pi * j * quadpoints_jax
        s = jnp.sin(arg)
        c = jnp.cos(arg)
        basis = basis.at[:, 2 * j - 1].set(s)
        basis = basis.at[:, 2 * j].set(c)
        dbasis = dbasis.at[:, 2 * j - 1].set(2.0 * jnp.pi * j * c)
        dbasis = dbasis.at[:, 2 * j].set(-2.0 * jnp.pi * j * s)

    return basis, dbasis


def fixed_surface_flux_integral_from_B(B, flux_spec: FixedSurfaceFluxSpec):
    Bcoil = B.reshape((flux_spec.nphi, flux_spec.ntheta, 3))
    return integral_BdotN_jax(
        Bcoil,
        flux_spec.target,
        flux_spec.normal,
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
    return jnp.asarray(surface.gamma()), jnp.asarray(surface.normal())


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
