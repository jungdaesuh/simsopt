"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_dB_by_dX,
    group_coil_data,
)
from .curve_geometry import curve_gamma_and_dash_from_spec
from .specs import (
    CoilSpec,
    GroupedCoilSetSpec,
    apply_coil_symmetry,
    make_grouped_coil_set_spec,
)


def _empty_grouped_field_result(points: object, kernel):
    point_count = points.shape[0]
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


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    if not coil_arrays:
        return _empty_grouped_field_result(points, kernel)

    gammas, gammadashs, currents = coil_arrays[0]
    result = kernel(points, gammas, gammadashs, currents)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = _tree_add(result, kernel(points, gammas, gammadashs, currents))
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


def grouped_coil_set_spec_from_coil_specs(
    coil_specs: tuple[CoilSpec, ...] | list[CoilSpec],
) -> GroupedCoilSetSpec:
    gammas = []
    gammadashs = []
    currents = []
    for coil_spec in coil_specs:
        gamma, gammadash = curve_gamma_and_dash_from_spec(coil_spec.curve)
        gamma, gammadash, current = apply_coil_symmetry(
            gamma,
            gammadash,
            coil_spec.current.value[0],
            coil_spec.symmetry,
        )
        gammas.append(gamma)
        gammadashs.append(gammadash)
        currents.append(current)
    return grouped_coil_set_spec_from_lists(gammas, gammadashs, currents)


def grouped_coil_set_spec_from_inputs(coil_arrays: object) -> GroupedCoilSetSpec:
    groups = []
    coil_offset = 0
    for gammas, gammadashs, currents in coil_arrays:
        group_size = currents.shape[0]
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


def grouped_coil_set_spec_from_source(coil_source: object) -> GroupedCoilSetSpec:
    if isinstance(coil_source, GroupedCoilSetSpec):
        return coil_source
    return grouped_coil_set_spec_from_inputs(coil_source)


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
    return grouped_coil_currents_from_spec(
        grouped_coil_set_spec_from_inputs(coil_arrays)
    )


def grouped_biot_savart_B_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_B)


def grouped_biot_savart_B_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_B_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_A_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_A)


def grouped_biot_savart_A_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_A_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
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
