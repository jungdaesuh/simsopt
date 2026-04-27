from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ..backend import get_pairwise_penalty_chunk_size
from ..jax_core._math_utils import as_jax_float64 as _runtime_as_jax_float64
from ..jax_core._math_utils import as_runtime_float64 as _runtime_as_runtime_float64
from ..jax_core.sharding import maybe_shard_pairwise_row_inputs


def _as_jax_float64(value):
    return _runtime_as_jax_float64(value)


def _scalar_like(reference, value):
    return _runtime_as_runtime_float64(value, reference=reference)


def _pairwise_distances(gamma1, gamma2):
    delta = gamma1[:, None, :] - gamma2[None, :, :]
    return jnp.sqrt(jnp.sum(jnp.square(delta), axis=2))


def _masked_pairwise_distances(gamma1, gamma2, valid, fill_distance):
    delta = gamma1[:, None, :] - gamma2[None, :, :]
    squared = jnp.sum(jnp.square(delta), axis=2)
    fill_squared = jnp.square(fill_distance)
    return jnp.sqrt(jnp.where(valid, squared, fill_squared))


def _resolve_pairwise_penalty_chunk_size(chunk_size=None) -> int:
    if chunk_size is None:
        return int(get_pairwise_penalty_chunk_size())
    return int(chunk_size)


def _use_dense_pairwise_path(row_count: int, col_count: int, chunk_size: int) -> bool:
    return chunk_size <= 0 or (row_count <= chunk_size and col_count <= chunk_size)


def _padded_row_chunks(array, chunk_size: int):
    row_count = int(array.shape[0])
    chunk_count = 0 if row_count == 0 else (row_count + chunk_size - 1) // chunk_size
    padded_row_count = chunk_count * chunk_size
    pad_rows = padded_row_count - row_count
    if pad_rows > 0:
        zero_rows = jnp.sum(array, axis=0, keepdims=True, dtype=array.dtype)
        zero_rows = zero_rows - zero_rows
        zero_rows = jnp.broadcast_to(zero_rows, (pad_rows, *array.shape[1:]))
        padded = jnp.concatenate((array, zero_rows), axis=0)
    else:
        zero_rows = None
        padded = array
    chunk_shape = (chunk_count, chunk_size, *array.shape[1:])
    return padded.reshape(chunk_shape), zero_rows


def _row_zero_values(array):
    row_zero = array - array
    if array.ndim == 1:
        return row_zero
    axes = tuple(range(1, array.ndim))
    return jnp.sum(row_zero, axis=axes)


def _row_true_values(array):
    row_zero = _row_zero_values(array)
    return (row_zero == row_zero) | (row_zero != row_zero)


def _row_false_values(array):
    row_true = _row_true_values(array)
    return row_true & ~row_true


def _row_weight_values(array, value, zero):
    if array.ndim == 1:
        return array * zero + value
    axes = tuple(range(1, array.ndim))
    return jnp.sum(array * zero, axis=axes) + value


def _chunk_rows(array, chunk_size: int):
    chunks, zero_rows = _padded_row_chunks(array, chunk_size)
    chunk_count = int(chunks.shape[0])
    valid = _row_true_values(array)
    if zero_rows is not None:
        invalid = _row_false_values(zero_rows)
        valid = jnp.concatenate((valid, invalid), axis=0)
    valid = valid.reshape((chunk_count, chunk_size))
    return chunks, valid


def _chunk_rows_with_valid_weights(array, chunk_size: int, one, zero):
    chunks, zero_rows = _padded_row_chunks(array, chunk_size)
    chunk_count = int(chunks.shape[0])
    valid = _row_weight_values(array, one, zero)
    if zero_rows is not None:
        invalid = _row_weight_values(zero_rows, zero, zero)
        valid = jnp.concatenate((valid, invalid), axis=0)
    return chunks, valid.reshape((chunk_count, chunk_size))


def _point_cloud_center_radius(points, zero):
    center = jnp.mean(points, axis=0)
    distances = jnp.sqrt(jnp.sum(jnp.square(points - center[None, :]), axis=1))
    return center, jnp.max(distances)


def _masked_point_cloud_center_radius(points, valid, zero):
    valid_weight = valid.astype(points.dtype)
    count = jnp.sum(valid_weight)
    center = jnp.sum(points * valid_weight[:, None], axis=0) / count
    distances = jnp.sqrt(jnp.sum(jnp.square(points - center[None, :]), axis=1))
    radius = jnp.max(jnp.where(valid, distances, zero))
    return center, radius


def _point_cloud_lower_bound(points_a, points_b, zero):
    center_a, radius_a = _point_cloud_center_radius(points_a, zero)
    center_b, radius_b = _point_cloud_center_radius(points_b, zero)
    center_distance = jnp.sqrt(jnp.sum(jnp.square(center_a - center_b)))
    return jnp.maximum(center_distance - radius_a - radius_b, zero)


def _masked_point_cloud_lower_bound(points_a, valid_a, points_b, valid_b, zero):
    center_a, radius_a = _masked_point_cloud_center_radius(points_a, valid_a, zero)
    center_b, radius_b = _masked_point_cloud_center_radius(points_b, valid_b, zero)
    center_distance = jnp.sqrt(jnp.sum(jnp.square(center_a - center_b)))
    return jnp.maximum(center_distance - radius_a - radius_b, zero)


def _pairwise_rowwise_min_distance(points_a, points_b, *, chunk_size=None):
    points_a = _as_jax_float64(points_a)
    points_b = _as_jax_float64(points_b)
    points_a, points_b = maybe_shard_pairwise_row_inputs(points_a, points_b)
    row_count = int(points_a.shape[0])
    col_count = int(points_b.shape[0])
    inf = _scalar_like(points_a, np.inf)
    if row_count == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if col_count == 0:
        return jnp.full((row_count,), inf, dtype=jnp.float64)
    chunk_size = _resolve_pairwise_penalty_chunk_size(chunk_size)
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        return jnp.min(_pairwise_distances(points_a, points_b), axis=1)

    point_chunks, point_masks = _chunk_rows(points_a, chunk_size)
    other_chunks, other_masks = _chunk_rows(points_b, chunk_size)
    initial_row_min = jnp.full((chunk_size,), inf, dtype=jnp.float64)

    def _scan_point_chunks(carry, point_inputs):
        point_chunk, point_mask = point_inputs

        def _scan_other_chunks(row_min, other_inputs):
            other_chunk, other_mask = other_inputs
            valid = point_mask[:, None] & other_mask[None, :]
            dists = _masked_pairwise_distances(point_chunk, other_chunk, valid, inf)
            block_row_min = jnp.min(jnp.where(valid, dists, inf), axis=1)
            return jnp.minimum(row_min, block_row_min), None

        row_min, _ = lax.scan(
            jax.checkpoint(_scan_other_chunks),
            initial_row_min,
            (other_chunks, other_masks),
        )
        row_min = jnp.where(point_mask, row_min, inf)
        return carry, row_min

    _, rowwise_chunks = lax.scan(
        _scan_point_chunks,
        _as_jax_float64(0.0),
        (point_chunks, point_masks),
    )
    return rowwise_chunks.reshape((-1,))[:row_count]


def _pairwise_rowwise_pnorm_distance(points_a, points_b, p, *, chunk_size=None):
    points_a = _as_jax_float64(points_a)
    points_b = _as_jax_float64(points_b)
    points_a, points_b = maybe_shard_pairwise_row_inputs(points_a, points_b)
    p_jax = _scalar_like(points_a, p)
    one = _scalar_like(points_a, 1.0)
    zero = _scalar_like(points_a, 0.0)
    row_count = int(points_a.shape[0])
    col_count = int(points_b.shape[0])
    if row_count == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if col_count == 0:
        return jnp.full((row_count,), jnp.inf, dtype=jnp.float64)
    chunk_size = _resolve_pairwise_penalty_chunk_size(chunk_size)
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        dists = _pairwise_distances(points_a, points_b)
        return jnp.sum(dists**p_jax, axis=1) ** (one / p_jax)

    point_chunks, point_masks = _chunk_rows(points_a, chunk_size)
    other_chunks, other_masks = _chunk_rows(points_b, chunk_size)
    initial_row_sum = jnp.zeros((chunk_size,), dtype=jnp.float64)

    def _scan_point_chunks(carry, point_inputs):
        point_chunk, point_mask = point_inputs

        def _scan_other_chunks(row_sum, other_inputs):
            other_chunk, other_mask = other_inputs
            valid = point_mask[:, None] & other_mask[None, :]
            dists = _masked_pairwise_distances(point_chunk, other_chunk, valid, one)
            safe_dists = jnp.where(valid, dists, one)
            block_power_sum = jnp.sum(
                jnp.where(valid, safe_dists**p_jax, zero),
                axis=1,
            )
            return row_sum + block_power_sum, None

        row_sum, _ = lax.scan(
            jax.checkpoint(_scan_other_chunks),
            initial_row_sum,
            (other_chunks, other_masks),
        )
        row_sum = jnp.where(point_mask, row_sum, zero)
        return carry, row_sum

    _, rowwise_chunks = lax.scan(
        _scan_point_chunks,
        zero,
        (point_chunks, point_masks),
    )
    rowwise_power_sum = rowwise_chunks.reshape((-1,))[:row_count]
    return rowwise_power_sum ** (one / p_jax)


def pairwise_min_distance_pure(points_a, points_b, *, chunk_size=None):
    """Return the minimum sampled pairwise distance using fixed-size JAX blocks."""
    points_a = _as_jax_float64(points_a)
    points_b = _as_jax_float64(points_b)
    points_a, points_b = maybe_shard_pairwise_row_inputs(points_a, points_b)
    row_count = int(points_a.shape[0])
    col_count = int(points_b.shape[0])
    inf = _scalar_like(points_a, np.inf)
    if row_count == 0 or col_count == 0:
        return inf
    chunk_size = _resolve_pairwise_penalty_chunk_size(chunk_size)
    if _use_dense_pairwise_path(row_count, col_count, chunk_size):
        return jnp.min(_pairwise_distances(points_a, points_b))

    point_chunks, point_masks = _chunk_rows(points_a, chunk_size)
    other_chunks, other_masks = _chunk_rows(points_b, chunk_size)

    def _scan_point_chunks(current_min, point_inputs):
        point_chunk, point_mask = point_inputs

        def _scan_other_chunks(block_min, other_inputs):
            other_chunk, other_mask = other_inputs
            valid = point_mask[:, None] & other_mask[None, :]
            dists = _masked_pairwise_distances(point_chunk, other_chunk, valid, inf)
            next_min = jnp.min(jnp.where(valid, dists, inf))
            return jnp.minimum(block_min, next_min), None

        current_min, _ = lax.scan(
            jax.checkpoint(_scan_other_chunks),
            current_min,
            (other_chunks, other_masks),
        )
        return current_min, None

    pair_min, _ = lax.scan(
        jax.checkpoint(_scan_point_chunks),
        inf,
        (point_chunks, point_masks),
    )
    return pair_min


def pairwise_thresholded_mean_square_distance_pure(
    points_a,
    points_b,
    minimum_distance,
    *,
    chunk_size=None,
):
    points_a = _as_jax_float64(points_a)
    points_b = _as_jax_float64(points_b)
    points_a, points_b = maybe_shard_pairwise_row_inputs(points_a, points_b)
    minimum_distance_jax = _scalar_like(points_a, minimum_distance)
    zero = _scalar_like(points_a, 0.0)
    row_count = int(points_a.shape[0])
    col_count = int(points_b.shape[0])
    if row_count == 0 or col_count == 0:
        return zero
    normalization = _scalar_like(points_a, row_count * col_count)
    chunk_size = _resolve_pairwise_penalty_chunk_size(chunk_size)

    def _dense_total():
        dists = _pairwise_distances(points_a, points_b)
        excess = jnp.maximum(minimum_distance_jax - dists, zero)
        return jnp.sum(jnp.square(excess)) / normalization

    def _chunked_total():
        point_chunks, point_masks = _chunk_rows(points_a, chunk_size)
        other_chunks, other_masks = _chunk_rows(points_b, chunk_size)

        def _scan_point_chunks(total, point_inputs):
            point_chunk, point_mask = point_inputs

            def _scan_other_chunks(row_total, other_inputs):
                other_chunk, other_mask = other_inputs
                block_lower_bound = _masked_point_cloud_lower_bound(
                    point_chunk,
                    point_mask,
                    other_chunk,
                    other_mask,
                    zero,
                )

                def _skip_block(_):
                    return row_total, None

                def _compute_block(_):
                    valid = point_mask[:, None] & other_mask[None, :]
                    dists = _masked_pairwise_distances(
                        point_chunk,
                        other_chunk,
                        valid,
                        minimum_distance_jax,
                    )
                    safe_dists = jnp.where(valid, dists, minimum_distance_jax)
                    excess = jnp.maximum(minimum_distance_jax - safe_dists, zero)
                    block_total = jnp.sum(jnp.where(valid, jnp.square(excess), zero))
                    return row_total + block_total, None

                return lax.cond(
                    block_lower_bound >= minimum_distance_jax,
                    _skip_block,
                    _compute_block,
                    operand=None,
                )

            total, _ = lax.scan(
                jax.checkpoint(_scan_other_chunks),
                total,
                (other_chunks, other_masks),
                _split_transpose=True,
            )
            return total, None

        total, _ = lax.scan(
            jax.checkpoint(_scan_point_chunks),
            zero,
            (point_chunks, point_masks),
            _split_transpose=True,
        )
        return total / normalization

    def _sweep(_):
        if _use_dense_pairwise_path(row_count, col_count, chunk_size):
            return _dense_total()
        return _chunked_total()

    return lax.cond(
        _point_cloud_lower_bound(points_a, points_b, zero) >= minimum_distance_jax,
        lambda _: zero,
        _sweep,
        operand=None,
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def _chunked_selected_exp_sum(
    points_a,
    points_b,
    hard_min,
    temperature_jax,
    cutoff,
    *,
    chunk_size,
):
    zero = hard_min - hard_min
    point_chunks, point_masks = _chunk_rows(points_a, chunk_size)
    other_chunks, other_masks = _chunk_rows(points_b, chunk_size)

    def _scan_point_chunks(total, point_inputs):
        point_chunk, point_mask = point_inputs
        local_zero = total - total

        def _scan_other_chunks(row_total, other_inputs):
            other_chunk, other_mask = other_inputs
            block_lower_bound = _masked_point_cloud_lower_bound(
                point_chunk,
                point_mask,
                other_chunk,
                other_mask,
                local_zero,
            )

            def _skip_block(_):
                return row_total, None

            def _compute_block(_):
                valid = point_mask[:, None] & other_mask[None, :]
                dists = _masked_pairwise_distances(
                    point_chunk,
                    other_chunk,
                    valid,
                    cutoff,
                )
                selected = valid & (dists <= cutoff)
                shifted = -(dists - hard_min) / temperature_jax
                block_sum = jnp.sum(jnp.where(selected, jnp.exp(shifted), local_zero))
                return row_total + block_sum, None

            return lax.cond(
                block_lower_bound > cutoff,
                _skip_block,
                _compute_block,
                operand=None,
            )

        total, _ = lax.scan(
            jax.checkpoint(_scan_other_chunks),
            total,
            (other_chunks, other_masks),
            _split_transpose=True,
        )
        return total, None

    total, _ = lax.scan(
        jax.checkpoint(_scan_point_chunks),
        zero,
        (point_chunks, point_masks),
        _split_transpose=True,
    )
    return total


def pairwise_selected_smoothmin_distance_pure(
    point_pairs,
    temperature,
    *,
    chunk_size=None,
):
    normalized_pairs = tuple(
        (
            _as_jax_float64(points_a).reshape((-1, 3)),
            _as_jax_float64(points_b).reshape((-1, 3)),
        )
        for points_a, points_b in point_pairs
    )
    normalized_pairs = tuple(
        (points_a, points_b)
        for points_a, points_b in normalized_pairs
        if int(points_a.shape[0]) > 0 and int(points_b.shape[0]) > 0
    )
    if len(normalized_pairs) == 0:
        return _as_jax_float64(np.inf)

    reference = normalized_pairs[0][0]
    zero = _scalar_like(reference, 0.0)
    temperature_jax = _scalar_like(reference, temperature)
    min_temperature = _scalar_like(reference, np.finfo(np.float64).eps)
    temperature_jax = jnp.maximum(temperature_jax, min_temperature)
    chunk_size = _resolve_pairwise_penalty_chunk_size(chunk_size)

    hard_min = _scalar_like(reference, np.inf)
    for points_a, points_b in normalized_pairs:
        pair_min = pairwise_min_distance_pure(
            points_a,
            points_b,
            chunk_size=chunk_size,
        )
        hard_min = jnp.minimum(hard_min, pair_min)

    hard_min = lax.stop_gradient(hard_min)
    cutoff = hard_min + _scalar_like(reference, 4.0) * temperature_jax
    cutoff = lax.stop_gradient(cutoff)

    def _dense_pair_exp_sum(points_a, points_b):
        dists = _pairwise_distances(points_a, points_b)
        selected = dists <= cutoff
        shifted = -(dists - hard_min) / temperature_jax
        return jnp.sum(jnp.where(selected, jnp.exp(shifted), zero))

    sum_exp = zero
    for points_a, points_b in normalized_pairs:
        row_count = int(points_a.shape[0])
        col_count = int(points_b.shape[0])
        if _use_dense_pairwise_path(row_count, col_count, chunk_size):
            pair_sum = _dense_pair_exp_sum(points_a, points_b)
        else:
            pair_sum = _chunked_selected_exp_sum(
                points_a,
                points_b,
                hard_min,
                temperature_jax,
                cutoff,
                chunk_size=chunk_size,
            )
        sum_exp = sum_exp + pair_sum
    return hard_min - temperature_jax * jnp.log(sum_exp)
