"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

import jax.numpy as jnp

from .specs import GroupedCoilSetSpec, make_grouped_coil_set_spec
from ..field.biotsavart_jax import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_dB_by_dX,
    group_coil_data,
)


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    gammas, gammadashs, currents = coil_arrays[0]
    result = kernel(points, gammas, gammadashs, currents)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + kernel(points, gammas, gammadashs, currents)
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


def grouped_biot_savart_B_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_B)


def grouped_biot_savart_A_from_spec(points: object, coil_spec: GroupedCoilSetSpec):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_A)


def grouped_biot_savart_dB_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dB_by_dX)


def grouped_biot_savart_B_and_dB_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    gammas, gammadashs, currents = coil_arrays[0]
    B, dB = biot_savart_B_and_dB(points, gammas, gammadashs, currents)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        Bi, dBi = biot_savart_B_and_dB(points, gammas, gammadashs, currents)
        B = B + Bi
        dB = dB + dBi
    return B, dB
