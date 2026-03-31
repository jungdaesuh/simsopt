"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from ..backend import get_chunk_policy
from .specs import GroupedCoilSetSpec, make_grouped_coil_set_spec
from ..field.biotsavart_jax import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_dB_by_dX,
    group_coil_data,
)

_POINT_CHUNK_SIZE_BY_POLICY = {
    "host_reference": 0,
    "stable_default": 256,
    "performance_tuned": 1024,
}


def _point_chunk_size() -> int:
    return int(_POINT_CHUNK_SIZE_BY_POLICY.get(get_chunk_policy(), 0))


def _empty_grouped_field_result(points: object, kernel):
    point_count = int(points.shape[0])
    zeros = jnp.zeros
    if kernel in {biot_savart_B, biot_savart_A}:
        return zeros((point_count, 3), dtype=jnp.float64)
    if kernel is biot_savart_dB_by_dX:
        return zeros((point_count, 3, 3), dtype=jnp.float64)
    if kernel is biot_savart_B_and_dB:
        return (
            zeros((point_count, 3), dtype=jnp.float64),
            zeros((point_count, 3, 3), dtype=jnp.float64),
        )
    raise ValueError(f"Unsupported grouped-field kernel: {kernel!r}")


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_dynamic_update(prefix_tree, chunk_tree, start_index: int):
    return jax.tree_util.tree_map(
        lambda acc, update: lax.dynamic_update_slice(
            acc,
            update,
            (start_index,) + (0,) * (acc.ndim - 1),
        ),
        prefix_tree,
        chunk_tree,
    )


def _tree_trim(prefix_tree, size: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[:size], prefix_tree)


def _tree_zeros_like_prefix(reference_tree, prefix_size: int):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(
            (prefix_size,) + tuple(leaf.shape[1:]),
            dtype=leaf.dtype,
        ),
        reference_tree,
    )


def _slice_point_chunk(points: object, start: int, chunk_size: int):
    return lax.dynamic_slice(
        points,
        (start, 0),
        (chunk_size, points.shape[1]),
    )


def _kernel_on_point_chunks(points: object, kernel, gammas, gammadashs, currents):
    point_count = int(points.shape[0])
    chunk_size = _point_chunk_size()
    if point_count == 0 or chunk_size <= 0 or point_count <= chunk_size:
        return kernel(points, gammas, gammadashs, currents)

    chunk_count = (point_count + chunk_size - 1) // chunk_size
    padded_point_count = chunk_count * chunk_size
    padded_points = jnp.pad(
        points,
        ((0, padded_point_count - point_count), (0, 0)),
    )
    first_chunk_points = _slice_point_chunk(padded_points, 0, chunk_size)
    first_result = kernel(first_chunk_points, gammas, gammadashs, currents)
    padded_result = _tree_dynamic_update(
        _tree_zeros_like_prefix(first_result, padded_point_count),
        first_result,
        0,
    )

    def body(chunk_index: int, acc):
        start = chunk_index * chunk_size
        chunk_points = _slice_point_chunk(padded_points, start, chunk_size)
        chunk_result = kernel(chunk_points, gammas, gammadashs, currents)
        return _tree_dynamic_update(acc, chunk_result, start)

    padded_result = lax.fori_loop(1, chunk_count, body, padded_result)
    return _tree_trim(padded_result, point_count)


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    if not coil_arrays:
        return _empty_grouped_field_result(points, kernel)

    gammas, gammadashs, currents = coil_arrays[0]
    result = _kernel_on_point_chunks(points, kernel, gammas, gammadashs, currents)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = _tree_add(
            result,
            _kernel_on_point_chunks(points, kernel, gammas, gammadashs, currents),
        )
    return result


def grouped_coil_set_spec_from_lists(
    gammas_list: object,
    gammadashs_list: object,
    currents_list: object,
) -> GroupedCoilSetSpec:
    return make_grouped_coil_set_spec(
        group_coil_data(gammas_list, gammadashs_list, currents_list)
    )


def grouped_coil_set_spec_from_grouped_data(groups: object) -> GroupedCoilSetSpec:
    return make_grouped_coil_set_spec(groups)


def grouped_coil_set_spec_from_inputs(coil_arrays: object) -> GroupedCoilSetSpec:
    groups = []
    coil_offset = 0
    for gammas, gammadashs, currents in coil_arrays:
        group_size = int(currents.shape[0])
        groups.append(
            (
                gammas,
                gammadashs,
                currents,
                tuple(range(coil_offset, coil_offset + group_size)),
            )
        )
        coil_offset += group_size
    return make_grouped_coil_set_spec(groups)


def _coil_set_spec_from_inputs(coil_arrays: object) -> GroupedCoilSetSpec:
    return grouped_coil_set_spec_from_inputs(coil_arrays)


def grouped_field_inputs_from_spec(
    coil_spec: GroupedCoilSetSpec,
) -> tuple[tuple[object, object, object], ...]:
    return coil_spec.field_inputs()


def grouped_field_data_from_spec(
    coil_spec: GroupedCoilSetSpec,
) -> tuple[tuple[object, object, object, list[int]], ...]:
    return coil_spec.as_grouped_data()


def grouped_coil_index_lists_from_spec(
    coil_spec: GroupedCoilSetSpec,
) -> tuple[list[int], ...]:
    return tuple(list(indices) for indices in coil_spec.coil_index_lists())


def grouped_coil_currents_from_spec(
    coil_spec: GroupedCoilSetSpec,
    *,
    coil_count: int | None = None,
):
    if coil_count is None:
        coil_count = (
            max(
                (max(indices) for indices in coil_spec.coil_index_lists()),
                default=-1,
            )
            + 1
        )
    currents = jnp.zeros((coil_count,), dtype=jnp.float64)
    for group in coil_spec.groups:
        index_array = jnp.asarray(group.coil_indices, dtype=jnp.int32)
        currents = currents.at[index_array].set(group.currents)
    return currents


def grouped_coil_currents_from_inputs(coil_arrays: object):
    return grouped_coil_currents_from_spec(_coil_set_spec_from_inputs(coil_arrays))


def grouped_biot_savart_B_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_B)


def grouped_biot_savart_B_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_B_from_spec(
        points,
        _coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_A_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_A)


def grouped_biot_savart_A_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_A_from_spec(
        points,
        _coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_dB_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dB_by_dX)


def grouped_biot_savart_B_and_dB_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    B, dB = _accumulate_grouped_field(points, coil_spec, biot_savart_B_and_dB)
    return B, dB
