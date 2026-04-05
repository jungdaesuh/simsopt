"""Pure JAX SurfaceRZFourier geometry built on immutable specs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._device_scalars import device_one, float_scalar, two_pi
from .specs import make_surface_rzfourier_spec
from .specs import SurfaceRZFourierSpec


def _as_jax_float64(value) -> jax.Array:
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.float64)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return jax.device_put(np.asarray(value, dtype=np.float64))
    return jnp.asarray(value, dtype=jnp.float64)


def _zero_based_mode_range(count: int, reference: jax.Array) -> jax.Array:
    ones = jnp.broadcast_to(device_one(reference), (count,))
    return jnp.cumsum(ones) - ones


def _poloidal_modes(spec: SurfaceRZFourierSpec) -> jax.Array:
    return _zero_based_mode_range(spec.mpol + 1, spec.quadpoints_theta)


def _toroidal_modes(spec: SurfaceRZFourierSpec) -> jax.Array:
    zero_based = _zero_based_mode_range(2 * spec.ntor + 1, spec.quadpoints_phi)
    return zero_based - float_scalar(spec.ntor, zero_based)


def _mode_angles(spec: SurfaceRZFourierSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    angle_scale = two_pi(spec.quadpoints_theta)
    theta = angle_scale * spec.quadpoints_theta
    phi = angle_scale * spec.quadpoints_phi
    m = _poloidal_modes(spec)
    n = _toroidal_modes(spec)
    nfp = float_scalar(spec.nfp, n)
    angles = (
        m[None, None, :, None] * theta[None, :, None, None]
        - nfp * n[None, None, None, :] * phi[:, None, None, None]
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
        jax.device_put(np.asarray(m_idx, dtype=np.int32)),
        jax.device_put(np.asarray(n_idx, dtype=np.int32)),
    )


def _block_mode_positions(
    *,
    mpol: int,
    ntor: int,
    include_zero_mode: bool,
) -> np.ndarray:
    width = 2 * ntor + 1
    positions: list[int] = []

    start_n = 0 if include_zero_mode else 1
    for n in range(start_n, ntor + 1):
        positions.append(n + ntor)

    for m in range(1, mpol + 1):
        for n in range(-ntor, ntor + 1):
            positions.append(m * width + n + ntor)

    return np.asarray(positions, dtype=np.int64)


def _gather_matrix(positions: np.ndarray, source_size: int) -> jax.Array:
    matrix = np.zeros((positions.size, source_size), dtype=np.float64)
    matrix[np.arange(positions.size), positions] = 1.0
    return _as_jax_float64(matrix)


def _scatter_matrix(
    positions: np.ndarray,
    *,
    target_size: int,
    source_size: int,
    source_offset: int,
) -> jax.Array:
    matrix = np.zeros((target_size, source_size), dtype=np.float64)
    source_columns = np.arange(positions.size) + source_offset
    matrix[positions, source_columns] = 1.0
    return _as_jax_float64(matrix)


def _coefficients_from_dofs(
    dofs: jax.Array,
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    coeff_shape = (mpol + 1, 2 * ntor + 1)
    flat_size = coeff_shape[0] * coeff_shape[1]
    include_positions = _block_mode_positions(
        mpol=mpol,
        ntor=ntor,
        include_zero_mode=True,
    )
    exclude_positions = _block_mode_positions(
        mpol=mpol,
        ntor=ntor,
        include_zero_mode=False,
    )

    rc_count = int(include_positions.size)
    tail_count = int(exclude_positions.size)
    dofs = _as_jax_float64(dofs)
    total_dofs = rc_count + tail_count if stellsym else 2 * rc_count + 2 * tail_count
    zero_flat = _as_jax_float64(np.zeros(flat_size, dtype=np.float64))

    rc = jnp.reshape(
        _scatter_matrix(
            include_positions,
            target_size=flat_size,
            source_size=total_dofs,
            source_offset=0,
        )
        @ dofs,
        coeff_shape,
    )
    if stellsym:
        zs = jnp.reshape(
            _scatter_matrix(
                exclude_positions,
                target_size=flat_size,
                source_size=total_dofs,
                source_offset=rc_count,
            )
            @ dofs,
            coeff_shape,
        )
        zero = jnp.reshape(zero_flat, coeff_shape)
        return rc, zero, zero, zs

    rs_start = rc_count
    zc_start = rs_start + tail_count
    zs_start = zc_start + rc_count

    rs = jnp.reshape(
        _scatter_matrix(
            exclude_positions,
            target_size=flat_size,
            source_size=total_dofs,
            source_offset=rs_start,
        )
        @ dofs,
        coeff_shape,
    )
    zc = jnp.reshape(
        _scatter_matrix(
            include_positions,
            target_size=flat_size,
            source_size=total_dofs,
            source_offset=zc_start,
        )
        @ dofs,
        coeff_shape,
    )
    zs = jnp.reshape(
        _scatter_matrix(
            exclude_positions,
            target_size=flat_size,
            source_size=total_dofs,
            source_offset=zs_start,
        )
        @ dofs,
        coeff_shape,
    )
    return rc, rs, zc, zs


def surface_rz_fourier_dofs_from_spec(spec: SurfaceRZFourierSpec) -> jax.Array:
    include_positions = _block_mode_positions(
        mpol=spec.mpol,
        ntor=spec.ntor,
        include_zero_mode=True,
    )
    exclude_positions = _block_mode_positions(
        mpol=spec.mpol,
        ntor=spec.ntor,
        include_zero_mode=False,
    )
    flat_size = int((spec.mpol + 1) * (2 * spec.ntor + 1))
    include_selector = _gather_matrix(include_positions, flat_size)
    exclude_selector = _gather_matrix(exclude_positions, flat_size)
    rc = include_selector @ jnp.reshape(_as_jax_float64(spec.rc), (flat_size,))
    if spec.stellsym:
        zs = exclude_selector @ jnp.reshape(_as_jax_float64(spec.zs), (flat_size,))
        return jnp.concatenate((rc, zs))
    rs = exclude_selector @ jnp.reshape(_as_jax_float64(spec.rs), (flat_size,))
    zc = include_selector @ jnp.reshape(_as_jax_float64(spec.zc), (flat_size,))
    zs = exclude_selector @ jnp.reshape(_as_jax_float64(spec.zs), (flat_size,))
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
    dofs_jax = _as_jax_float64(dofs)
    return jax.jacfwd(lambda x: evaluator(_spec_from_dofs(spec, x)))(dofs_jax)


def surface_rz_fourier_gamma_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    r, z = _radius_height_from_modes(spec, cos_terms, sin_terms)
    cos_phi, sin_phi = _phi_frame(phi)
    return jnp.stack([r * cos_phi, r * sin_phi, z], axis=-1)


def surface_rz_fourier_gammadash1_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    n = _toroidal_modes(spec)
    angle_scale = two_pi(n)
    nfp = float_scalar(spec.nfp, n)
    scale = angle_scale * nfp * n[None, None, None, :]
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
    return jnp.stack(
        [
            d_r * cos_phi - r * (angle_scale * sin_phi),
            d_r * sin_phi + r * (angle_scale * cos_phi),
            d_z,
        ],
        axis=-1,
    )


def surface_rz_fourier_gammadash2_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    m = _poloidal_modes(spec)
    scale = two_pi(m) * m[None, None, :, None]
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
    safe_norm = jnp.where(norm > 0, norm, jnp.ones_like(norm))
    return normal / safe_norm


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
