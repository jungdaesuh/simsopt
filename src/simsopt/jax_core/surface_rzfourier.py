"""Pure JAX SurfaceRZFourier geometry built on immutable specs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .specs import make_surface_rzfourier_spec
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


def _block_mode_indices(
    *,
    mpol: int,
    ntor: int,
    include_zero_mode: bool,
) -> tuple[jax.Array, jax.Array]:
    m_idx: list[int] = []
    n_idx: list[int] = []

    start_n = 0 if include_zero_mode else 1
    for n in range(start_n, ntor + 1):
        m_idx.append(0)
        n_idx.append(n + ntor)

    for m in range(1, mpol + 1):
        for n in range(-ntor, ntor + 1):
            m_idx.append(m)
            n_idx.append(n + ntor)

    return (
        jnp.asarray(m_idx, dtype=jnp.int32),
        jnp.asarray(n_idx, dtype=jnp.int32),
    )


def _coefficients_from_dofs(
    dofs: jax.Array,
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    coeff_shape = (mpol + 1, 2 * ntor + 1)
    include_m, include_n = _block_mode_indices(
        mpol=mpol,
        ntor=ntor,
        include_zero_mode=True,
    )
    exclude_m, exclude_n = _block_mode_indices(
        mpol=mpol,
        ntor=ntor,
        include_zero_mode=False,
    )

    rc_count = int(include_m.shape[0])
    tail_count = int(exclude_m.shape[0])
    dofs = jnp.asarray(dofs, dtype=jnp.float64)

    rc = (
        jnp.zeros(coeff_shape, dtype=jnp.float64)
        .at[include_m, include_n]
        .set(dofs[:rc_count])
    )
    if stellsym:
        zs = (
            jnp.zeros(coeff_shape, dtype=jnp.float64)
            .at[exclude_m, exclude_n]
            .set(dofs[rc_count : rc_count + tail_count])
        )
        zero = jnp.zeros(coeff_shape, dtype=jnp.float64)
        return rc, zero, zero, zs

    rs_start = rc_count
    zc_start = rs_start + tail_count
    zs_start = zc_start + rc_count

    rs = (
        jnp.zeros(coeff_shape, dtype=jnp.float64)
        .at[exclude_m, exclude_n]
        .set(dofs[rs_start : rs_start + tail_count])
    )
    zc = (
        jnp.zeros(coeff_shape, dtype=jnp.float64)
        .at[include_m, include_n]
        .set(dofs[zc_start : zc_start + rc_count])
    )
    zs = (
        jnp.zeros(coeff_shape, dtype=jnp.float64)
        .at[exclude_m, exclude_n]
        .set(dofs[zs_start : zs_start + tail_count])
    )
    return rc, rs, zc, zs


def surface_rz_fourier_dofs_from_spec(spec: SurfaceRZFourierSpec) -> jax.Array:
    include_m, include_n = _block_mode_indices(
        mpol=spec.mpol,
        ntor=spec.ntor,
        include_zero_mode=True,
    )
    exclude_m, exclude_n = _block_mode_indices(
        mpol=spec.mpol,
        ntor=spec.ntor,
        include_zero_mode=False,
    )
    rc = spec.rc[include_m, include_n]
    if spec.stellsym:
        zs = spec.zs[exclude_m, exclude_n]
        return jnp.concatenate((rc, zs))
    rs = spec.rs[exclude_m, exclude_n]
    zc = spec.zc[include_m, include_n]
    zs = spec.zs[exclude_m, exclude_n]
    return jnp.concatenate((rc, rs, zc, zs))


def surface_rz_fourier_spec_from_dofs(
    dofs: jax.Array,
    *,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
) -> SurfaceRZFourierSpec:
    rc, rs, zc, zs = _coefficients_from_dofs(
        dofs,
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
    )
    return make_surface_rzfourier_spec(
        rc=rc,
        rs=rs,
        zc=zc,
        zs=zs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        stellsym=stellsym,
    )


def _spec_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
) -> SurfaceRZFourierSpec:
    return surface_rz_fourier_spec_from_dofs(
        dofs,
        quadpoints_phi=spec.quadpoints_phi,
        quadpoints_theta=spec.quadpoints_theta,
        mpol=spec.mpol,
        ntor=spec.ntor,
        nfp=spec.nfp,
        stellsym=spec.stellsym,
    )


def _evaluate_from_dofs(
    evaluator,
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
):
    return evaluator(_spec_from_dofs(spec, dofs))


def _evaluate_jacobian_from_dofs(
    evaluator,
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
):
    dofs_jax = jnp.asarray(dofs, dtype=jnp.float64)
    return jax.jacfwd(lambda x: evaluator(_spec_from_dofs(spec, x)))(dofs_jax)


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


def surface_rz_fourier_unitnormal_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_unitnormal_from_spec, spec, dofs)


def surface_rz_fourier_area_from_spec(spec: SurfaceRZFourierSpec):
    normal = surface_rz_fourier_normal_from_spec(spec)
    nphi, ntheta = normal.shape[:2]
    return jnp.sum(jnp.linalg.norm(normal, axis=-1)) / (nphi * ntheta)


def surface_rz_fourier_volume_from_spec(spec: SurfaceRZFourierSpec):
    gamma = surface_rz_fourier_gamma_from_spec(spec)
    normal = surface_rz_fourier_normal_from_spec(spec)
    nphi, ntheta = gamma.shape[:2]
    return jnp.sum(jnp.sum(gamma * normal, axis=-1)) / (3.0 * nphi * ntheta)


def surface_rz_fourier_dnormal_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_jacobian_from_dofs(surface_rz_fourier_normal_from_spec, spec, dofs)


def surface_rz_fourier_dunitnormal_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_unitnormal_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_gamma_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_gamma_from_spec, spec, dofs)


def surface_rz_fourier_gammadash1_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_gammadash1_from_spec, spec, dofs)


def surface_rz_fourier_gammadash2_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_gammadash2_from_spec, spec, dofs)


def surface_rz_fourier_normal_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_normal_from_spec, spec, dofs)


def surface_rz_fourier_area_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_area_from_spec, spec, dofs)


def surface_rz_fourier_volume_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_volume_from_spec, spec, dofs)
