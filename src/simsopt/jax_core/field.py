"""Pure grouped-field helpers that operate on immutable specs."""

from __future__ import annotations

from functools import partial

import jax
from jax import lax
import numpy as np
from jax.sharding import PartitionSpec as P

from .biotsavart import (
    biot_savart_A,
    biot_savart_B,
    biot_savart_B_and_dB,
    biot_savart_B_and_dB_with_point_axis,
    biot_savart_B_vjp,
    biot_savart_d2A_by_dXdX,
    biot_savart_dA_by_dX,
    biot_savart_dB_by_dX,
    group_coil_data,
)
from .sharding import (
    coil_group_collective_config,
    collective_field_sharding_summary,
    maybe_shard_grouped_field_inputs,
)
from ._math_utils import (
    as_runtime_float64 as _as_runtime_float64,
    pad_axis as _pad_axis,
)
from .curve_geometry import (
    curve_gamma_and_dash_from_spec,
    curve_spec_with_dofs,
    optimizable_input_dofs_from_map_spec,
)
from .specs import (
    CoilDofExtractionSpec,
    CoilSetDofExtractionSpec,
    CoilSpec,
    CurrentValueSpec,
    GroupedCoilSetSpec,
    apply_coil_symmetry,
    make_grouped_coil_set_spec,
)


def _zeros_float64(shape):
    return jax.device_put(np.zeros(shape, dtype=np.float64))


def _empty_grouped_field_result(points: object, kernel):
    point_count = points.shape[0]
    if kernel in {biot_savart_B, biot_savart_A}:
        return _zeros_float64((point_count, 3))
    if kernel is biot_savart_dA_by_dX:
        return _zeros_float64((point_count, 3, 3))
    if kernel is biot_savart_d2A_by_dXdX:
        return _zeros_float64((point_count, 3, 3, 3))
    if kernel is biot_savart_dB_by_dX:
        return _zeros_float64((point_count, 3, 3))
    if kernel is biot_savart_B_and_dB:
        return (
            _zeros_float64((point_count, 3)),
            _zeros_float64((point_count, 3, 3)),
        )
    raise ValueError(f"Unsupported grouped-field kernel: {kernel!r}")


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_trim_axis0(tree, size: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[:size], tree)


def _runtime_group_inputs(reference, gammas, gammadashs, currents):
    return (
        _as_runtime_float64(gammas, reference=reference),
        _as_runtime_float64(gammadashs, reference=reference),
        _as_runtime_float64(currents, reference=reference),
    )


def _pad_coil_axis_to_device_count(gammas, gammadashs, currents, device_count: int):
    coil_count = int(currents.shape[0])
    pad_count = (-coil_count) % device_count
    if pad_count == 0:
        return gammas, gammadashs, currents
    padded_count = coil_count + pad_count
    # Sharding requires axis sizes divisible by the device count. The padding
    # cost is bounded by device_count - 1 entries; keep this simple unless a
    # JAX device-memory profile shows material peak-memory pressure.
    return (
        _pad_axis(gammas, axis=0, padded_size=padded_count),
        _pad_axis(gammadashs, axis=0, padded_size=padded_count),
        _pad_axis(currents, axis=0, padded_size=padded_count),
    )


def _pad_point_axis_to_device_count(points, device_count: int):
    point_count = int(points.shape[0])
    pad_count = (-point_count) % device_count
    if pad_count == 0:
        return points
    return _pad_axis(points, axis=0, padded_size=point_count + pad_count)


def _field_out_specs(kernel, config):
    point_axis_name = config.point_axis_name
    if point_axis_name is None:
        if kernel is biot_savart_B_and_dB:
            return P(), P()
        return P()
    if kernel is biot_savart_B_and_dB:
        return P(point_axis_name, None), P(point_axis_name, None, None)
    if kernel in {biot_savart_B, biot_savart_A}:
        return P(point_axis_name, None)
    if kernel in {biot_savart_dA_by_dX, biot_savart_dB_by_dX}:
        return P(point_axis_name, None, None)
    return P(point_axis_name, None, None, None)


def _collective_kernel(kernel, config):
    if config.point_axis_name is not None and kernel is biot_savart_B_and_dB:
        return partial(
            biot_savart_B_and_dB_with_point_axis,
            point_axis_name=config.point_axis_name,
        )
    return kernel


def _collective_group_field(points, gammas, gammadashs, currents, kernel, config):
    point_count = int(points.shape[0])
    if config.point_axis_name is not None:
        points = _pad_point_axis_to_device_count(points, config.point_device_count)
    group_kernel = _collective_kernel(kernel, config)
    gammas, gammadashs, currents = _pad_coil_axis_to_device_count(
        gammas,
        gammadashs,
        currents,
        config.coil_device_count,
    )
    point_spec = (
        P()
        if config.point_axis_name is None
        else P(config.point_axis_name, None)
    )

    @partial(
        jax.shard_map,
        mesh=config.mesh,
        in_specs=(
            point_spec,
            P(config.coil_axis_name, None, None),
            P(config.coil_axis_name, None, None),
            P(config.coil_axis_name),
        ),
        out_specs=_field_out_specs(kernel, config),
        check_vma=True,
    )
    def _group_kernel(points_block, gammas_block, gammadashs_block, currents_block):
        return jax.tree_util.tree_map(
            lambda value: lax.psum(value, config.reduced_axis_name),
            group_kernel(
                points_block,
                gammas_block,
                gammadashs_block,
                currents_block,
            ),
        )

    result = _group_kernel(points, gammas, gammadashs, currents)
    if config.point_axis_name is None:
        return result
    return _tree_trim_axis0(result, point_count)


def _evaluate_grouped_field_group(points, gammas, gammadashs, currents, kernel):
    gammas, gammadashs, currents = _runtime_group_inputs(
        points,
        gammas,
        gammadashs,
        currents,
    )
    config = coil_group_collective_config(currents)
    if config is None:
        return kernel(points, gammas, gammadashs, currents), config
    return (
        _collective_group_field(
            points,
            gammas,
            gammadashs,
            currents,
            kernel,
            config,
        ),
        config,
    )


def _accumulate_grouped_field_with_config(
    points: object,
    coil_spec: GroupedCoilSetSpec,
    kernel,
):
    coil_arrays = grouped_field_inputs_from_spec(coil_spec)
    if not coil_arrays:
        return _empty_grouped_field_result(points, kernel), None
    points, coil_arrays = maybe_shard_grouped_field_inputs(points, coil_arrays)

    result, collective_config = _evaluate_grouped_field_group(
        points,
        *coil_arrays[0],
        kernel,
    )
    for gammas, gammadashs, currents in coil_arrays[1:]:
        group_result, group_config = _evaluate_grouped_field_group(
            points,
            gammas,
            gammadashs,
            currents,
            kernel,
        )
        result = _tree_add(result, group_result)
        if collective_config is None:
            collective_config = group_config
    return result, collective_config


def _accumulate_grouped_field(points: object, coil_spec: GroupedCoilSetSpec, kernel):
    result, _config = _accumulate_grouped_field_with_config(points, coil_spec, kernel)
    return result


def grouped_field_sharding_summary(points: object, coil_spec: GroupedCoilSetSpec):
    """Return grouped-field output sharding plus collective-route metadata."""
    result, config = _accumulate_grouped_field_with_config(
        points,
        coil_spec,
        biot_savart_B,
    )
    return collective_field_sharding_summary(result, config=config)


def biot_savart_B_vjp_maybe_collective(points, v, gammas, gammadashs, currents):
    """Return B pullback, using the coil-axis collective path when active."""
    gammas, gammadashs, currents = _runtime_group_inputs(
        points,
        gammas,
        gammadashs,
        currents,
    )
    config = coil_group_collective_config(currents)
    if config is None:
        return biot_savart_B_vjp(points, v, gammas, gammadashs, currents)

    coil_count = int(currents.shape[0])
    padded_gammas, padded_gammadashs, padded_currents = (
        _pad_coil_axis_to_device_count(
            gammas,
            gammadashs,
            currents,
            config.coil_device_count,
        )
    )

    def _collective_forward(group_gammas, group_gammadashs, group_currents):
        return _collective_group_field(
            points,
            group_gammas,
            group_gammadashs,
            group_currents,
            biot_savart_B,
            config,
        )

    _, pullback = jax.vjp(
        _collective_forward,
        padded_gammas,
        padded_gammadashs,
        padded_currents,
    )
    return _tree_trim_axis0(
        pullback(_as_runtime_float64(v, reference=points)),
        coil_count,
    )


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


def _coil_current_value_from_dofs(
    extraction_spec: CoilDofExtractionSpec,
    owner_dofs: object,
) -> CurrentValueSpec:
    current_dofs = optimizable_input_dofs_from_map_spec(
        extraction_spec.current_map,
        owner_dofs,
    )
    if current_dofs.shape[0] != 1:
        raise RuntimeError(
            "coil_specs_from_dof_extraction_spec() only supports scalar Current "
            "degrees of freedom."
        )
    return CurrentValueSpec(value=current_dofs[:1])


def coil_specs_from_dof_extraction_spec(
    extraction_spec: CoilSetDofExtractionSpec,
    owner_dofs: object,
) -> tuple[CoilSpec, ...]:
    owner_dofs = _as_runtime_float64(owner_dofs, reference=owner_dofs)
    return tuple(
        CoilSpec(
            curve=curve_spec_with_dofs(
                coil_spec.curve,
                optimizable_input_dofs_from_map_spec(
                    coil_spec.curve_map,
                    owner_dofs,
                ),
            ),
            current=_coil_current_value_from_dofs(coil_spec, owner_dofs),
            symmetry=coil_spec.symmetry,
        )
        for coil_spec in extraction_spec.coils
    )


def coil_set_spec_from_dof_extraction_spec(
    extraction_spec: CoilSetDofExtractionSpec,
    owner_dofs: object,
) -> GroupedCoilSetSpec:
    return grouped_coil_set_spec_from_coil_specs(
        coil_specs_from_dof_extraction_spec(extraction_spec, owner_dofs)
    )


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
    currents = _zeros_float64((coil_count,))
    for group in coil_spec.groups:
        positions = np.asarray(group.coil_indices, dtype=np.int64)
        insert = np.zeros((coil_count, positions.size), dtype=np.float64)
        insert[positions, np.arange(positions.size)] = 1.0
        keep_mask = np.ones(coil_count, dtype=np.float64)
        keep_mask[positions] = 0.0
        currents = (
            currents * jax.device_put(keep_mask)
            + jax.device_put(insert) @ group.currents
        )
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


def grouped_biot_savart_dA_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dA_by_dX)


def grouped_biot_savart_dA_by_dX_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_dA_by_dX_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_d2A_by_dXdX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_d2A_by_dXdX)


def grouped_biot_savart_d2A_by_dXdX_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_d2A_by_dXdX_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_dB_by_dX_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    return _accumulate_grouped_field(points, coil_spec, biot_savart_dB_by_dX)


def grouped_biot_savart_dB_by_dX_from_inputs(points: object, coil_arrays: object):
    return grouped_biot_savart_dB_by_dX_from_spec(
        points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )


def grouped_biot_savart_B_and_dB_from_spec(
    points: object,
    coil_spec: GroupedCoilSetSpec,
):
    B, dB = _accumulate_grouped_field(points, coil_spec, biot_savart_B_and_dB)
    return B, dB
