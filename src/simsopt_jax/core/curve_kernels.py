"""Shared pure JAX curve kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._pairwise_reductions import (
    _chunk_rows,
    _chunk_rows_with_valid_weights,
    _masked_pairwise_distances,
    _pairwise_distances,
    _resolve_pairwise_penalty_chunk_size,
    _use_dense_pairwise_path,
)

from ._device_scalars import staged_like


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
