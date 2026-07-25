"""Shared pure JAX curve kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._pairwise_reductions import (
    _chunk_rows,
    _chunk_rows_with_valid_weights,
    _masked_pairwise_distances,
    _pairwise_distances,
    _resolve_pairwise_penalty_chunk_size,
    _use_dense_pairwise_path,
)

from ._device_scalars import float_scalar, staged_like, two_pi
from ._math_utils import (
    as_compute_array as _as_compute_array,
    as_runtime_float64 as _as_runtime_float64,
)
from .surface_rzfourier import (
    surface_rz_fourier_spec_from_dofs,
)


def _dtype_scalar_like(reference, value):
    return staged_like(reference, value)


@jax.jit
def incremental_arclength_pure(d1gamma):
    """Return pointwise curve arclength increments."""
    return jnp.linalg.norm(d1gamma, axis=1)


@jax.jit
def curve_length_from_incremental_arclength_pure(incremental_arclength):
    """Return the CurveLengthJAX mean incremental-arclength normalization."""
    return jnp.mean(incremental_arclength)


@jax.jit
def kappa_pure(d1gamma, d2gamma):
    """Return pointwise curvature for first and second curve derivatives."""
    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )


@jax.jit
def curvature_p_norm_from_kappa_pure(kappa, gammadash, p, desired_kappa):
    """Return the excess-curvature p-norm used by LpCurveCurvatureJAX."""
    p_jax = jnp.asarray(p, dtype=kappa.dtype)
    desired_kappa_jax = jnp.asarray(desired_kappa, dtype=kappa.dtype)
    zero = jnp.asarray(0.0, dtype=kappa.dtype)
    one = jnp.asarray(1.0, dtype=kappa.dtype)
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    excess = jnp.maximum(kappa - desired_kappa_jax, zero)
    return (one / p_jax) * jnp.mean((excess**p_jax) * arc_length)


def curve_curve_distance_penalty_pure(
    gamma1,
    gammadash1,
    gamma2,
    gammadash2,
    minimum_distance,
):
    """Return the CurveCurveDistanceJAX squared lower-bound penalty."""
    gamma1 = jnp.asarray(gamma1)
    gammadash1 = jnp.asarray(gammadash1)
    gamma2 = jnp.asarray(gamma2, dtype=gamma1.dtype)
    gammadash2 = jnp.asarray(gammadash2, dtype=gamma1.dtype)
    minimum_distance_jax = _dtype_scalar_like(gamma1, minimum_distance)
    zero = _dtype_scalar_like(gamma1, 0.0)
    row_count = int(gamma1.shape[0])
    col_count = int(gamma2.shape[0])
    if row_count == 0 or col_count == 0:
        return zero
    normalization = _dtype_scalar_like(gamma1, row_count * col_count)

    arc_length_1 = jnp.linalg.norm(gammadash1, axis=1)
    arc_length_2 = jnp.linalg.norm(gammadash2, axis=1)
    chunk_size = _resolve_pairwise_penalty_chunk_size()
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        distances = _pairwise_distances(gamma1, gamma2)
        arc_length = arc_length_1[:, None] * arc_length_2[None, :]
        excess = jnp.maximum(minimum_distance_jax - distances, zero)
        return jnp.sum(arc_length * jnp.square(excess)) / normalization

    gamma1_chunks, gamma1_masks = _chunk_rows(gamma1, chunk_size)
    gamma2_chunks, gamma2_masks = _chunk_rows(gamma2, chunk_size)
    arc_length_1_chunks, _ = _chunk_rows(arc_length_1, chunk_size)
    arc_length_2_chunks, _ = _chunk_rows(arc_length_2, chunk_size)

    def _scan_gamma1_chunks(total, gamma1_inputs):
        gamma1_chunk, arc_length_1_chunk, gamma1_mask = gamma1_inputs

        def _scan_gamma2_chunks(row_total, gamma2_inputs):
            gamma2_chunk, arc_length_2_chunk, gamma2_mask = gamma2_inputs
            valid = gamma1_mask[:, None] & gamma2_mask[None, :]
            distances = _masked_pairwise_distances(
                gamma1_chunk,
                gamma2_chunk,
                valid,
                minimum_distance_jax,
            )
            arc_length = arc_length_1_chunk[:, None] * arc_length_2_chunk[None, :]
            safe_distances = jnp.where(valid, distances, minimum_distance_jax)
            diff = minimum_distance_jax - safe_distances
            excess = jnp.where(diff > zero, diff, zero)
            block_total = jnp.sum(
                jnp.where(valid, arc_length * jnp.square(excess), zero)
            )
            return row_total + block_total, None

        total, _ = jax.lax.scan(
            jax.checkpoint(_scan_gamma2_chunks),
            total,
            (gamma2_chunks, arc_length_2_chunks, gamma2_masks),
        )
        return total, None

    total, _ = jax.lax.scan(
        _scan_gamma1_chunks,
        zero,
        (gamma1_chunks, arc_length_1_chunks, gamma1_masks),
    )
    return total / normalization


def curve_surface_distance_penalty_pure(
    curve_gamma,
    curve_gammadash,
    surface_gamma,
    surface_normal,
    minimum_distance,
):
    """Return the CurveSurfaceDistanceJAX squared lower-bound penalty."""
    curve_gamma = jnp.asarray(curve_gamma)
    curve_gammadash = jnp.asarray(curve_gammadash)
    surface_gamma = jnp.asarray(surface_gamma, dtype=curve_gamma.dtype)
    surface_normal = jnp.asarray(surface_normal, dtype=curve_gamma.dtype)
    minimum_distance_jax = _dtype_scalar_like(curve_gamma, minimum_distance)
    zero = _dtype_scalar_like(curve_gamma, 0.0)
    one = _dtype_scalar_like(curve_gamma, 1.0)
    row_count = int(curve_gamma.shape[0])
    col_count = int(surface_gamma.shape[0])
    if row_count == 0 or col_count == 0:
        return zero

    curve_weights = jnp.linalg.norm(curve_gammadash, axis=1)
    surface_weights = jnp.linalg.norm(surface_normal, axis=1)
    chunk_size = _resolve_pairwise_penalty_chunk_size()
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        distances = _pairwise_distances(curve_gamma, surface_gamma)
        integral_weight = curve_weights[:, None] * surface_weights[None, :]
        diff = minimum_distance_jax - distances
        excess = jnp.where(diff > zero, diff, zero)
        normalization = jnp.sum(jnp.broadcast_to(one, distances.shape))
        return jnp.sum(integral_weight * jnp.square(excess)) / normalization

    def _chunk_with_weights(array):
        return _chunk_rows_with_valid_weights(array, chunk_size, one, zero)

    curve_gamma_chunks, curve_gamma_masks = _chunk_with_weights(curve_gamma)
    surface_gamma_chunks, surface_gamma_masks = _chunk_with_weights(surface_gamma)
    curve_weight_chunks, _ = _chunk_with_weights(curve_weights)
    surface_weight_chunks, _ = _chunk_with_weights(surface_weights)

    def _scan_curve_chunks(carry, curve_inputs):
        total, normalization = carry
        curve_gamma_chunk, curve_weight_chunk, curve_gamma_mask = curve_inputs

        def _scan_surface_chunks(inner_carry, surface_inputs):
            row_total, row_normalization = inner_carry
            surface_gamma_chunk, surface_weight_chunk, surface_gamma_mask = (
                surface_inputs
            )
            valid_weight = curve_gamma_mask[:, None] * surface_gamma_mask[None, :]
            valid = valid_weight > zero
            distances = _masked_pairwise_distances(
                curve_gamma_chunk,
                surface_gamma_chunk,
                valid,
                minimum_distance_jax,
            )
            integral_weight = (
                curve_weight_chunk[:, None] * surface_weight_chunk[None, :]
            )
            safe_distances = jnp.where(valid, distances, minimum_distance_jax)
            diff = minimum_distance_jax - safe_distances
            excess = jnp.where(diff > zero, diff, zero)
            block_total = jnp.sum(
                jnp.where(valid, integral_weight * jnp.square(excess), zero)
            )
            block_normalization = jnp.sum(valid_weight)
            return (
                row_total + block_total,
                row_normalization + block_normalization,
            ), None

        (total, normalization), _ = jax.lax.scan(
            jax.checkpoint(_scan_surface_chunks),
            (total, normalization),
            (surface_gamma_chunks, surface_weight_chunks, surface_gamma_masks),
        )
        return (total, normalization), None

    (total, normalization), _ = jax.lax.scan(
        _scan_curve_chunks,
        (zero, zero),
        (curve_gamma_chunks, curve_weight_chunks, curve_gamma_masks),
    )
    return total / normalization


@jax.jit
def torsion_pure(d1gamma, d2gamma, d3gamma):
    """Return pointwise torsion for first three curve derivatives."""
    cross12 = jnp.cross(d1gamma, d2gamma, axis=1)
    return jnp.sum(cross12 * d3gamma, axis=1) / jnp.sum(cross12 * cross12, axis=1)


def _array_for_work_dtype(value, *, reference, use_compute_dtype: bool):
    if use_compute_dtype:
        reference_array = _as_compute_array(reference)
        return _as_compute_array(
            value, dtype=reference_array.dtype, reference=reference_array
        )
    return _as_runtime_float64(value, reference=reference)


def _selector_matrix(size, positions, *, reference, use_compute_dtype: bool = False):
    matrix = np.zeros((len(positions), size), dtype=np.float64)
    if positions:
        matrix[np.arange(len(positions)), positions] = 1.0
    return _array_for_work_dtype(
        matrix,
        reference=reference,
        use_compute_dtype=use_compute_dtype,
    )


def _curve_mode_selectors(order, *, reference, use_compute_dtype: bool = False):
    size = 4 * order + 2
    return (
        _selector_matrix(
            size,
            list(range(0, order + 1)),
            reference=reference,
            use_compute_dtype=use_compute_dtype,
        ),
        _selector_matrix(
            size,
            list(range(order + 1, 2 * order + 1)),
            reference=reference,
            use_compute_dtype=use_compute_dtype,
        ),
        _selector_matrix(
            size,
            list(range(2 * order + 1, 3 * order + 2)),
            reference=reference,
            use_compute_dtype=use_compute_dtype,
        ),
        _selector_matrix(
            size,
            list(range(3 * order + 2, 4 * order + 2)),
            reference=reference,
            use_compute_dtype=use_compute_dtype,
        ),
    )


def _harmonic_terms(
    qpts, start_mode, count, trig_fn, *, use_compute_dtype: bool = False
):
    modes = _array_for_work_dtype(
        np.arange(start_mode, start_mode + count, dtype=np.float64),
        reference=qpts,
        use_compute_dtype=use_compute_dtype,
    )
    angles = qpts[:, None] * two_pi(qpts) * modes[None, :]
    return trig_fn(angles)


def gamma_2d(
    modes, qpts, order, G: int = 0, H: int = 0, *, use_compute_dtype: bool = False
):
    """Return the 2D curve-on-surface coordinates ``(phi, theta)``."""
    modes_jax = _array_for_work_dtype(
        modes,
        reference=modes,
        use_compute_dtype=use_compute_dtype,
    )
    qpts_jax = _array_for_work_dtype(
        qpts,
        reference=modes_jax,
        use_compute_dtype=use_compute_dtype,
    )
    phic_sel, phis_sel, thetac_sel, thetas_sel = _curve_mode_selectors(
        order,
        reference=modes_jax,
        use_compute_dtype=use_compute_dtype,
    )
    phic = phic_sel @ modes_jax
    phis = phis_sel @ modes_jax
    thetac = thetac_sel @ modes_jax
    thetas = thetas_sel @ modes_jax

    cos_terms = _harmonic_terms(
        qpts_jax,
        0,
        order + 1,
        jnp.cos,
        use_compute_dtype=use_compute_dtype,
    )
    sin_terms = _harmonic_terms(
        qpts_jax,
        1,
        order,
        jnp.sin,
        use_compute_dtype=use_compute_dtype,
    )
    theta = (
        cos_terms @ thetac
        + sin_terms @ thetas
        + jax.lax.stop_gradient(float_scalar(int(G), qpts_jax)) * qpts_jax
    )
    phi = (
        cos_terms @ phic
        + sin_terms @ phis
        + jax.lax.stop_gradient(float_scalar(int(H), qpts_jax)) * qpts_jax
    )
    return phi, theta


def _surface_rz_fourier_gamma_pointwise(
    surface_spec,
    phi_qpts,
    theta_qpts,
    *,
    use_compute_dtype: bool = False,
):
    """Evaluate an RZ Fourier surface at paired ``(phi, theta)`` samples."""
    angle_scale = two_pi(theta_qpts)
    phi = angle_scale * phi_qpts
    theta = angle_scale * theta_qpts
    m = _array_for_work_dtype(
        np.arange(surface_spec.mpol + 1, dtype=np.float64),
        reference=theta_qpts,
        use_compute_dtype=use_compute_dtype,
    )
    n = _array_for_work_dtype(
        np.arange(2 * surface_spec.ntor + 1, dtype=np.float64) - surface_spec.ntor,
        reference=phi_qpts,
        use_compute_dtype=use_compute_dtype,
    )
    nfp = float_scalar(surface_spec.nfp, n)
    angles = (
        m[None, :, None] * theta[:, None, None]
        - nfp * n[None, None, :] * phi[:, None, None]
    )
    cos_terms = jnp.cos(angles)
    sin_terms = jnp.sin(angles)
    radius = jnp.sum(
        surface_spec.rc[None, :, :] * cos_terms
        + surface_spec.rs[None, :, :] * sin_terms,
        axis=(1, 2),
    )
    z = jnp.sum(
        surface_spec.zc[None, :, :] * cos_terms
        + surface_spec.zs[None, :, :] * sin_terms,
        axis=(1, 2),
    )
    return jnp.stack([radius * jnp.cos(phi), radius * jnp.sin(phi), z], axis=-1)


def curve_cws_rz_gamma_from_dofs(
    curve_dofs,
    qpts,
    order,
    G,
    H,
    surf_dofs,
    mpol,
    ntor,
    nfp,
    stellsym=True,
    use_custom_vjp=False,
    use_compute_dtype=False,
):
    """Return 3D coordinates for a CWS Fourier curve on an RZ Fourier surface."""
    phi, theta = gamma_2d(
        curve_dofs,
        qpts,
        order,
        G,
        H,
        use_compute_dtype=use_compute_dtype,
    )
    surf_dofs_jax = (
        _as_compute_array(surf_dofs, dtype=phi.dtype, reference=phi)
        if use_compute_dtype
        else _as_runtime_float64(surf_dofs, reference=curve_dofs)
    )
    surface_spec = surface_rz_fourier_spec_from_dofs(
        surf_dofs_jax,
        quadpoints_phi=phi,
        quadpoints_theta=theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        use_custom_vjp=use_custom_vjp,
        use_compute_dtype=use_compute_dtype,
    )
    return _surface_rz_fourier_gamma_pointwise(
        surface_spec,
        phi,
        theta,
        use_compute_dtype=use_compute_dtype,
    )
