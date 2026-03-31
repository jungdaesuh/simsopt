"""Pure JAX SurfaceRZFourier geometry built on immutable specs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .specs import SurfaceRZFourierSpec


def _mode_angles(spec: SurfaceRZFourierSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    theta = 2.0 * jnp.pi * spec.quadpoints_theta
    phi = 2.0 * jnp.pi * spec.quadpoints_phi
    m = jnp.arange(spec.mpol + 1, dtype=jnp.float64)
    n = jnp.arange(-spec.ntor, spec.ntor + 1, dtype=jnp.float64)
    angles = (
        m[None, None, :, None] * theta[None, :, None, None]
        - spec.nfp * n[None, None, None, :] * phi[:, None, None, None]
    )
    return phi, jnp.cos(angles), jnp.sin(angles)


def _sum_fourier_modes(
    cos_coeffs: jax.Array,
    sin_coeffs: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
) -> jax.Array:
    return jnp.sum(
        cos_coeffs[None, None, :, :] * cos_terms
        + sin_coeffs[None, None, :, :] * sin_terms,
        axis=(2, 3),
    )


def _radius_height_from_modes(
    spec: SurfaceRZFourierSpec,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return (
        _sum_fourier_modes(spec.rc, spec.rs, cos_terms, sin_terms),
        _sum_fourier_modes(spec.zc, spec.zs, cos_terms, sin_terms),
    )


def _phi_frame(phi: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.cos(phi)[:, None], jnp.sin(phi)[:, None]


def surface_rz_fourier_gamma_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    r, z = _radius_height_from_modes(spec, cos_terms, sin_terms)
    cos_phi, sin_phi = _phi_frame(phi)
    return jnp.stack([r * cos_phi, r * sin_phi, z], axis=-1)


def surface_rz_fourier_gammadash1_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    n = jnp.arange(-spec.ntor, spec.ntor + 1, dtype=jnp.float64)
    scale = 2.0 * jnp.pi * spec.nfp * n[None, None, None, :]
    d_r = jnp.sum(
        spec.rc[None, None, :, :] * sin_terms * scale
        - spec.rs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    d_z = jnp.sum(
        spec.zc[None, None, :, :] * sin_terms * scale
        - spec.zs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    r, _ = _radius_height_from_modes(spec, cos_terms, sin_terms)
    cos_phi, sin_phi = _phi_frame(phi)
    two_pi = 2.0 * jnp.pi
    return jnp.stack(
        [
            d_r * cos_phi - r * (two_pi * sin_phi),
            d_r * sin_phi + r * (two_pi * cos_phi),
            d_z,
        ],
        axis=-1,
    )


def surface_rz_fourier_gammadash2_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    m = jnp.arange(spec.mpol + 1, dtype=jnp.float64)
    scale = 2.0 * jnp.pi * m[None, None, :, None]
    d_r = jnp.sum(
        -spec.rc[None, None, :, :] * sin_terms * scale
        + spec.rs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    d_z = jnp.sum(
        -spec.zc[None, None, :, :] * sin_terms * scale
        + spec.zs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    cos_phi, sin_phi = _phi_frame(phi)
    return jnp.stack([d_r * cos_phi, d_r * sin_phi, d_z], axis=-1)


def surface_rz_fourier_normal_from_spec(spec: SurfaceRZFourierSpec):
    return jnp.cross(
        surface_rz_fourier_gammadash1_from_spec(spec),
        surface_rz_fourier_gammadash2_from_spec(spec),
    )


def surface_rz_fourier_unitnormal_from_spec(spec: SurfaceRZFourierSpec):
    normal = surface_rz_fourier_normal_from_spec(spec)
    norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / norm


def surface_rz_fourier_area_from_spec(spec: SurfaceRZFourierSpec):
    normal = surface_rz_fourier_normal_from_spec(spec)
    nphi, ntheta = normal.shape[:2]
    return jnp.sum(jnp.linalg.norm(normal, axis=-1)) / (nphi * ntheta)


def surface_rz_fourier_volume_from_spec(spec: SurfaceRZFourierSpec):
    gamma = surface_rz_fourier_gamma_from_spec(spec)
    normal = surface_rz_fourier_normal_from_spec(spec)
    nphi, ntheta = gamma.shape[:2]
    return jnp.sum(jnp.sum(gamma * normal, axis=-1)) / (3.0 * nphi * ntheta)
