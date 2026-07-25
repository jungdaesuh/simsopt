"""Online mixed-precision Biot-Savart ``B`` reduction.

The grouped-field dispatch selects this implementation only for the production
mixed-compute lane with float32 points and source arrays. It evaluates a
flattened source representation in bounded source tiles, then combines tile
partials with compensated float64 accumulation. Float64 and non-mixed callers
retain the established grouped Biot-Savart implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from .biotsavart import (
    _MU0_OVER_4PI,
    _compute_inv,
    _compute_rsqrt,
    _cross_product,
    _radius_squared,
)
from .reductions import pairwise_sum_axis as _pairwise_sum_axis

__all__ = [
    "flatten_biot_savart_sources",
    "flatten_grouped_biot_savart_sources",
    "mixed_biot_savart_B_online",
]


def _require_float32(name: str, value) -> None:
    dtype = jnp.asarray(value).dtype
    if dtype != jnp.float32:
        raise TypeError(f"{name} must have dtype float32; got {dtype}.")


def _validate_flat_sources(points, source_positions, source_vectors) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P, 3); got {points.shape}.")
    if source_positions.ndim != 2 or source_positions.shape[1] != 3:
        raise ValueError(
            f"source_positions must have shape (S, 3); got {source_positions.shape}."
        )
    if source_vectors.shape != source_positions.shape:
        raise ValueError(
            "source_vectors must match source_positions; "
            f"got {source_vectors.shape} and {source_positions.shape}."
        )
    if source_positions.shape[0] == 0:
        raise ValueError("at least one flattened source is required.")
    _require_float32("points", points)
    _require_float32("source_positions", source_positions)
    _require_float32("source_vectors", source_vectors)


def _validate_group_sources(gammas, gammadashs, currents) -> None:
    if gammas.ndim != 3 or gammas.shape[2] != 3:
        raise ValueError(f"gammas must have shape (C, Q, 3); got {gammas.shape}.")
    if gammadashs.shape != gammas.shape:
        raise ValueError(
            f"gammadashs must match gammas; got {gammadashs.shape} and {gammas.shape}."
        )
    if currents.shape != (gammas.shape[0],):
        raise ValueError(
            "currents must have shape (C,); "
            f"got {currents.shape} for C={gammas.shape[0]}."
        )
    if gammas.shape[0] == 0 or gammas.shape[1] == 0:
        raise ValueError("each source group must contain coils and quadrature points.")
    _require_float32("gammas", gammas)
    _require_float32("gammadashs", gammadashs)
    _require_float32("currents", currents)


def flatten_biot_savart_sources(
    gammas,
    gammadashs,
    currents,
) -> tuple[jax.Array, jax.Array]:
    """Flatten one equal-quadrature coil group into weighted line sources.

    The returned source vector is ``current * gammadash / Q``.  Keeping this
    weight inside the vector removes the separate current array from the hot
    point-by-source kernel while preserving derivatives with respect to curve
    geometry and current through ordinary JAX transformations of this helper.
    """

    gammas = jnp.asarray(gammas)
    gammadashs = jnp.asarray(gammadashs)
    currents = jnp.asarray(currents)
    _validate_group_sources(gammas, gammadashs, currents)
    quadrature_scale = jnp.asarray(1.0 / gammas.shape[1], dtype=jnp.float32)
    weighted_vectors = gammadashs * currents[:, None, None] * quadrature_scale
    return (
        jnp.reshape(gammas, (-1, 3)),
        jnp.reshape(weighted_vectors, (-1, 3)),
    )


def flatten_grouped_biot_savart_sources(
    groups: Sequence[tuple[jax.Array, jax.Array, jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """Flatten and concatenate statically grouped coil arrays.

    Groups may have different quadrature counts; each group receives its own
    exact ``1 / Q`` weight before concatenation.  The group sequence must be
    non-empty and is intentionally static at trace time.
    """

    if not groups:
        raise ValueError("at least one coil group is required.")
    flattened = tuple(
        flatten_biot_savart_sources(gammas, gammadashs, currents)
        for gammas, gammadashs, currents in groups
    )
    return (
        jnp.concatenate(tuple(group[0] for group in flattened), axis=0),
        jnp.concatenate(tuple(group[1] for group in flattened), axis=0),
    )


def _tile_value_sum(points, source_positions, source_vectors):
    diff = source_positions[None, :, :] - points[:, None, :]
    radius_squared = _radius_squared(diff)
    radius_inverse = _compute_rsqrt(radius_squared)
    radius_inverse_cubed = radius_inverse * _compute_inv(radius_squared)
    contributions = (
        _cross_product(diff, source_vectors[None, :, :])
        * radius_inverse_cubed[..., None]
    )
    return _pairwise_sum_axis(contributions, axis=1)


def _tile_tangent_sum(
    points,
    source_positions,
    source_vectors,
    points_dot,
    source_positions_dot,
    source_vectors_dot,
):
    diff = source_positions[None, :, :] - points[:, None, :]
    diff_dot = source_positions_dot[None, :, :] - points_dot[:, None, :]
    radius_squared = _radius_squared(diff)
    radius_inverse = _compute_rsqrt(radius_squared)
    radius_inverse_cubed = radius_inverse * _compute_inv(radius_squared)
    diff_inner_dot = jnp.sum(diff * diff_dot, axis=-1)
    radius_inverse_cubed_dot = (
        jnp.asarray(-3.0, dtype=jnp.float32)
        * diff_inner_dot
        * radius_inverse_cubed
        * _compute_inv(radius_squared)
    )
    cross = _cross_product(diff, source_vectors[None, :, :])
    cross_dot = _cross_product(diff_dot, source_vectors[None, :, :]) + _cross_product(
        diff,
        source_vectors_dot[None, :, :],
    )
    contributions_dot = (
        cross_dot * radius_inverse_cubed[..., None]
        + cross * radius_inverse_cubed_dot[..., None]
    )
    return _pairwise_sum_axis(contributions_dot, axis=1)


def _zero_accumulator(points):
    zero_float32 = points - points
    zero_float64 = jnp.asarray(zero_float32, dtype=jnp.float64)
    return zero_float64, zero_float64


def _neumaier_add(state, value):
    total, compensation = state
    updated_total = total + value
    correction = jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - updated_total) + value,
        (value - updated_total) + total,
    )
    return updated_total, compensation + correction


def _neumaier_add_jvp(primal_state, tangent_state, value, value_dot):
    total, compensation = primal_state
    total_dot, compensation_dot = tangent_state
    updated_total = total + value
    updated_total_dot = total_dot + value_dot

    # The selection depends only on primal values.  Applying that same selection
    # to the differentiated arithmetic keeps the tangent map linear while
    # preserving the primal compensated-summation semantics.
    total_dominates = jnp.abs(total) >= jnp.abs(value)
    correction = jnp.where(
        total_dominates,
        (total - updated_total) + value,
        (value - updated_total) + total,
    )
    correction_dot = jnp.where(
        total_dominates,
        (total_dot - updated_total_dot) + value_dot,
        (value_dot - updated_total_dot) + total_dot,
    )
    return (
        (updated_total, compensation + correction),
        (updated_total_dot, compensation_dot + correction_dot),
    )


def _accumulator_value(state):
    total, compensation = state
    return total + compensation


def _source_tiles(array, *, tile_size: int, full_tile_count: int):
    full_source_count = full_tile_count * tile_size
    return jnp.reshape(array[:full_source_count], (full_tile_count, tile_size, 3))


def _accumulate_primal_tiles(
    points,
    source_positions,
    source_vectors,
    *,
    source_tile_size: int,
):
    source_count = source_positions.shape[0]
    full_tile_count = source_count // source_tile_size
    tail_start = full_tile_count * source_tile_size
    state = _zero_accumulator(points)

    if full_tile_count:
        position_tiles = _source_tiles(
            source_positions,
            tile_size=source_tile_size,
            full_tile_count=full_tile_count,
        )
        vector_tiles = _source_tiles(
            source_vectors,
            tile_size=source_tile_size,
            full_tile_count=full_tile_count,
        )

        def body(carry, tiles):
            position_tile, vector_tile = tiles
            tile_sum = jax.checkpoint(_tile_value_sum)(
                points,
                position_tile,
                vector_tile,
            )
            return _neumaier_add(
                carry,
                jnp.asarray(tile_sum, dtype=jnp.float64),
            ), None

        state, _ = lax.scan(body, state, (position_tiles, vector_tiles))

    if tail_start < source_count:
        tail_sum = jax.checkpoint(_tile_value_sum)(
            points,
            source_positions[tail_start:],
            source_vectors[tail_start:],
        )
        state = _neumaier_add(state, jnp.asarray(tail_sum, dtype=jnp.float64))

    return _accumulator_value(state)


def _accumulate_primal_and_tangent_tiles(
    points,
    source_positions,
    source_vectors,
    points_dot,
    source_positions_dot,
    source_vectors_dot,
    *,
    source_tile_size: int,
):
    """Return primal and tangent values from one ordered compensated pass."""
    source_count = source_positions.shape[0]
    full_tile_count = source_count // source_tile_size
    tail_start = full_tile_count * source_tile_size
    primal_state = _zero_accumulator(points)
    tangent_state = _zero_accumulator(points)

    if full_tile_count:
        tiled_arrays = tuple(
            _source_tiles(
                array,
                tile_size=source_tile_size,
                full_tile_count=full_tile_count,
            )
            for array in (
                source_positions,
                source_vectors,
                source_positions_dot,
                source_vectors_dot,
            )
        )

        def body(carry, tiles):
            primal_carry, tangent_carry = carry
            (
                position_tile,
                vector_tile,
                position_dot_tile,
                vector_dot_tile,
            ) = tiles
            tile_sum = jax.checkpoint(_tile_value_sum)(
                points,
                position_tile,
                vector_tile,
            )
            tile_sum_dot = jax.checkpoint(_tile_tangent_sum)(
                points,
                position_tile,
                vector_tile,
                points_dot,
                position_dot_tile,
                vector_dot_tile,
            )
            return _neumaier_add_jvp(
                primal_carry,
                tangent_carry,
                jnp.asarray(tile_sum, dtype=jnp.float64),
                jnp.asarray(tile_sum_dot, dtype=jnp.float64),
            ), None

        (primal_state, tangent_state), _ = lax.scan(
            body,
            (primal_state, tangent_state),
            tiled_arrays,
        )

    if tail_start < source_count:
        tail_sum = jax.checkpoint(_tile_value_sum)(
            points,
            source_positions[tail_start:],
            source_vectors[tail_start:],
        )
        tail_sum_dot = jax.checkpoint(_tile_tangent_sum)(
            points,
            source_positions[tail_start:],
            source_vectors[tail_start:],
            points_dot,
            source_positions_dot[tail_start:],
            source_vectors_dot[tail_start:],
        )
        primal_state, tangent_state = _neumaier_add_jvp(
            primal_state,
            tangent_state,
            jnp.asarray(tail_sum, dtype=jnp.float64),
            jnp.asarray(tail_sum_dot, dtype=jnp.float64),
        )

    return _accumulator_value(primal_state), _accumulator_value(tangent_state)


def _scale_float64(value):
    return value * jnp.asarray(_MU0_OVER_4PI, dtype=jnp.float64)


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _mixed_biot_savart_B_online(
    points,
    source_positions,
    source_vectors,
    source_tile_size: int,
):
    return _scale_float64(
        _accumulate_primal_tiles(
            points,
            source_positions,
            source_vectors,
            source_tile_size=source_tile_size,
        )
    )


@_mixed_biot_savart_B_online.defjvp
def _mixed_biot_savart_B_online_jvp(
    source_tile_size: int,
    primals,
    tangents,
):
    points, source_positions, source_vectors = primals
    points_dot, source_positions_dot, source_vectors_dot = tangents
    primal_sum, tangent_sum = _accumulate_primal_and_tangent_tiles(
        points,
        source_positions,
        source_vectors,
        points_dot,
        source_positions_dot,
        source_vectors_dot,
        source_tile_size=source_tile_size,
    )
    return _scale_float64(primal_sum), _scale_float64(tangent_sum)


def mixed_biot_savart_B_online(
    points,
    source_positions,
    source_vectors,
    *,
    source_tile_size: int,
):
    """Evaluate mixed-precision ``B`` from flattened weighted line sources.

    All inputs must be float32.  Each source tile is evaluated and reduced in
    float32; only the ``(P, 3)`` tile partial crosses into the compensated
    float64 accumulator.  The return value is float64.  This strict mixed-only
    contract prevents accidental replacement or numerical drift of the
    established float64 reference implementation.
    """

    if source_tile_size <= 0:
        raise ValueError(f"source_tile_size must be positive; got {source_tile_size}.")
    points = jnp.asarray(points)
    source_positions = jnp.asarray(source_positions)
    source_vectors = jnp.asarray(source_vectors)
    _validate_flat_sources(points, source_positions, source_vectors)
    return _mixed_biot_savart_B_online(
        points,
        source_positions,
        source_vectors,
        int(source_tile_size),
    )
