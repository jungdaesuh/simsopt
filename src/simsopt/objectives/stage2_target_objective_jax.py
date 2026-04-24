"""JAX objective bundle used by the Stage 2 ondevice target lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .._core.jax_host_boundary import (
    host_array as _shared_host_array,
    host_tree as _shared_host_tree,
)
from ..field.biotsavart_jax_backend import _unwrap_coil_curve_and_current
from ..geo.curve import incremental_arclength_pure, kappa_pure
from ..jax_core._math_utils import (
    as_jax_float64 as _math_as_jax_float64,
    as_runtime_float64 as _as_runtime_float64,
)
from ..jax_core.field import (
    grouped_field_sharding_summary,
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_coil_set_spec_from_lists,
)
from ..jax_core import (
    curve_geometry_from_dofs,
    curve_spec_from_curve,
)
from ..jax_core.objectives_flux import (
    fixed_surface_flux_integral_from_B,
    fixed_surface_flux_residual_from_B,
    fixed_surface_flux_specs_from_surface,
)
from ..jax_core.sharding import (
    maybe_shard_pairwise_row_trees,
    summarize_array_sharding,
)
from ..geo.curveobjectives import (
    Lp_curvature_pure,
    cc_distance_pure,
    curve_length_pure,
)
from ..geo.optimizer_jax import (
    _mark_cacheable_jit_value_and_grad,
    _mark_structured_private_solver_cacheable,
)

__all__ = [
    "Stage2PenaltyConfig",
    "Stage2TargetObjectiveBundle",
    "Stage2TargetOptimizerState",
    "Stage2TargetObjectiveTerm",
    "build_stage2_target_objective",
    "stage2_target_optimizer_state_from_dofs",
    "stage2_target_optimizer_state_to_dofs",
]

Stage2ObjectiveFn = Callable[[jnp.ndarray], jnp.ndarray]
Stage2ValueAndGradFn = Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
Stage2ResidualFn = Callable[[jnp.ndarray], jnp.ndarray]
Stage2ALMValueAndGradFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]
Stage2ALMValueAndGradBuilder = Callable[..., Stage2ALMValueAndGradFn]


class Stage2TargetObjectiveTerm(NamedTuple):
    name: str
    weight: float


class Stage2PenaltyConfig(NamedTuple):
    squared_flux_weight: float
    length_weight: float
    length_target: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    curvature_threshold: float
    curvature_p_norm: float
    squared_flux_definition: str = "quadratic flux"


@dataclass(frozen=True)
class Stage2TargetOptimizerState:
    """Structured optimizer state for the Stage 2 target lane."""

    current_dof: jax.Array
    curve_dofs: jax.Array


jax.tree_util.register_dataclass(
    Stage2TargetOptimizerState,
    data_fields=["current_dof", "curve_dofs"],
    meta_fields=[],
)


class Stage2TargetObjectiveBundle(NamedTuple):
    objective: Stage2ObjectiveFn
    expected_dof_count: int
    value_and_grad: Stage2ValueAndGradFn | None = None
    terms: tuple[Stage2TargetObjectiveTerm, ...] = ()
    raw_terms: Stage2ObjectiveFn | None = None
    least_squares_residual: Stage2ResidualFn | None = None
    alm_value_and_grad_builder: Stage2ALMValueAndGradBuilder | None = None
    field_sharding_summary: Callable[[jnp.ndarray], dict[str, object]] | None = None
    pairwise_penalty_sharding_summary: (
        Callable[[jnp.ndarray], dict[str, object]] | None
    ) = None


def _as_jax_float64(value) -> jax.Array:
    return _math_as_jax_float64(value)


def _device_float64_array(value) -> jax.Array:
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _runtime_float64_array(value, *, reference) -> jax.Array:
    return _as_runtime_float64(value, reference=reference)


def _runtime_float64_scalar(value, *, reference) -> jax.Array:
    return _as_runtime_float64(value, reference=reference)


def _selected_smoothmin(values, temperature):
    values = _as_jax_float64(values).reshape((-1,))
    if int(values.shape[0]) == 0:
        return _runtime_float64_scalar(np.inf, reference=values)
    bounded_temperature = jnp.maximum(
        _runtime_float64_scalar(temperature, reference=values),
        _runtime_float64_scalar(np.finfo(np.float64).eps, reference=values),
    )
    hard_min = jnp.min(values)
    logits = -(values - hard_min) / bounded_temperature
    selection_mask = values <= (
        hard_min + _runtime_float64_scalar(4.0, reference=values) * bounded_temperature
    )
    masked_logits = jnp.where(selection_mask, logits, -jnp.inf)
    return hard_min - bounded_temperature * jax.nn.logsumexp(masked_logits)


def _selected_smoothmax(values, temperature):
    values = _as_jax_float64(values).reshape((-1,))
    if int(values.shape[0]) == 0:
        return _runtime_float64_scalar(-np.inf, reference=values)
    bounded_temperature = jnp.maximum(
        _runtime_float64_scalar(temperature, reference=values),
        _runtime_float64_scalar(np.finfo(np.float64).eps, reference=values),
    )
    hard_max = jnp.max(values)
    logits = (values - hard_max) / bounded_temperature
    selection_mask = values >= (
        hard_max - _runtime_float64_scalar(4.0, reference=values) * bounded_temperature
    )
    masked_logits = jnp.where(selection_mask, logits, -jnp.inf)
    return hard_max + bounded_temperature * jax.nn.logsumexp(masked_logits)


def _as_objective_dofs(value) -> jax.Array:
    if isinstance(value, np.ndarray):
        return _device_float64_array(value)
    return _as_jax_float64(value)


def _runtimeify_tree(value):
    return _shared_host_tree(value, dtype=np.float64)


def _split_stage2_dofs(dofs, curve_dof_count=None):
    state = stage2_target_optimizer_state_from_dofs(
        dofs,
        curve_dof_count=curve_dof_count,
    )
    return state.current_dof, state.curve_dofs


def stage2_target_optimizer_state_from_dofs(dofs, *, curve_dof_count=None):
    if isinstance(dofs, Stage2TargetOptimizerState):
        return Stage2TargetOptimizerState(
            current_dof=_as_objective_dofs(dofs.current_dof),
            curve_dofs=_as_objective_dofs(dofs.curve_dofs),
        )
    dofs = _as_objective_dofs(dofs)
    current_dof = jnp.sum(jax.lax.slice_in_dim(dofs, 0, 1, axis=0))
    if curve_dof_count is None:
        curve_dofs = jax.lax.slice_in_dim(dofs, 1, dofs.shape[0], axis=0)
    else:
        curve_dofs = jax.lax.slice_in_dim(
            dofs,
            start_index=1,
            limit_index=1 + int(curve_dof_count),
            axis=0,
        )
    return Stage2TargetOptimizerState(
        current_dof=current_dof,
        curve_dofs=curve_dofs,
    )


def stage2_target_optimizer_state_to_dofs(state) -> jax.Array:
    state = stage2_target_optimizer_state_from_dofs(state)
    return jnp.concatenate(
        (
            jnp.reshape(_as_objective_dofs(state.current_dof), (1,)),
            _as_objective_dofs(state.curve_dofs),
        )
    )


def _fixed_curve_penalty(curves, minimum_distance):
    total = _as_jax_float64(0.0)
    minimum_distance_jax = _as_jax_float64(minimum_distance)
    for i, (gamma_i, gammadash_i) in enumerate(curves):
        gamma_i_jax = _as_jax_float64(gamma_i)
        gammadash_i_jax = _as_jax_float64(gammadash_i)
        for gamma_j, gammadash_j in curves[:i]:
            total = total + cc_distance_pure(
                gamma_i_jax,
                gammadash_i_jax,
                _as_jax_float64(gamma_j),
                _as_jax_float64(gammadash_j),
                minimum_distance_jax,
            )
    return total


def _curve_pairs_from_grouped_coil_set_spec(coil_set_spec):
    curve_pairs = []
    for group in coil_set_spec.groups:
        gammas, gammadashs = _curve_group_host_arrays(group)
        group_size = int(gammas.shape[0])
        for coil_index in range(group_size):
            curve_pairs.append(
                (
                    gammas[coil_index],
                    gammadashs[coil_index],
                )
            )
    return tuple(curve_pairs)


def _curve_group_arrays(group):
    return _as_jax_float64(group.gammas), _as_jax_float64(group.gammadashs)


def _curve_group_host_arrays(group):
    return (
        _shared_host_array(group.gammas, dtype=np.float64),
        _shared_host_array(group.gammadashs, dtype=np.float64),
    )


def _host_float64_array(value):
    return _shared_host_array(value, dtype=np.float64)


def _host_float64_scalar(value) -> float:
    return float(_host_float64_array(value).reshape(()))


def _curve_groups_from_grouped_coil_set_spec(coil_set_spec):
    curve_groups = []
    for group in coil_set_spec.groups:
        curve_groups.append(_curve_group_host_arrays(group))
    return tuple(curve_groups)


def _banana_symmetry_runtime_inputs_from_coils(banana_coils):
    banana_rotmats = []
    banana_current_scales = []
    for _, rotmat, _, scale in (
        _unwrap_coil_curve_and_current(coil) for coil in banana_coils
    ):
        if rotmat is None:
            rotmat = np.eye(3, dtype=np.float64)
        banana_rotmats.append(_host_float64_array(rotmat))
        banana_current_scales.append(_host_float64_scalar(scale))
    return (
        np.asarray(banana_rotmats, dtype=np.float64),
        np.asarray(banana_current_scales, dtype=np.float64),
    )


def _build_dynamic_curve_data(
    base_gamma,
    base_gammadash,
    banana_rotmats,
    banana_current_scales,
    current_dof,
):
    def _apply_one(rotmat, current_scale):
        return (
            base_gamma @ rotmat,
            base_gammadash @ rotmat,
            current_dof * current_scale,
        )

    return jax.vmap(_apply_one, in_axes=(0, 0))(
        banana_rotmats,
        banana_current_scales,
    )


def _pairwise_curve_distance_penalty_scan(
    left_gammas,
    left_gammadashs,
    right_gammas,
    right_gammadashs,
    minimum_distance,
    *,
    strict_lower_triangle=False,
):
    zero = _runtime_float64_scalar(0.0, reference=minimum_distance)
    if int(left_gammas.shape[0]) == 0 or int(right_gammas.shape[0]) == 0:
        return zero

    left_indices = jnp.arange(left_gammas.shape[0], dtype=jnp.int32)
    right_indices = jnp.arange(right_gammas.shape[0], dtype=jnp.int32)
    (
        (
            left_indices,
            left_gammas,
            left_gammadashs,
        ),
        (
            right_indices,
            right_gammas,
            right_gammadashs,
        ),
    ) = maybe_shard_pairwise_row_trees(
        (left_indices, left_gammas, left_gammadashs),
        (right_indices, right_gammas, right_gammadashs),
    )

    def _scan_left_chunks(total, left_inputs):
        left_index, left_gamma, left_gammadash = left_inputs

        def _scan_right_chunks(row_total, right_inputs):
            right_index, right_gamma, right_gammadash = right_inputs
            if strict_lower_triangle:
                pair_penalty = jax.lax.cond(
                    right_index < left_index,
                    lambda _: cc_distance_pure(
                        left_gamma,
                        left_gammadash,
                        right_gamma,
                        right_gammadash,
                        minimum_distance,
                    ),
                    lambda _: zero,
                    operand=None,
                )
            else:
                pair_penalty = cc_distance_pure(
                    left_gamma,
                    left_gammadash,
                    right_gamma,
                    right_gammadash,
                    minimum_distance,
                )
            return row_total + pair_penalty, None

        row_total, _ = jax.lax.scan(
            _scan_right_chunks,
            zero,
            (right_indices, right_gammas, right_gammadashs),
        )
        return total + row_total, None

    total, _ = jax.lax.scan(
        _scan_left_chunks,
        zero,
        (left_indices, left_gammas, left_gammadashs),
    )
    return total


def _dynamic_curve_distance_penalty(
    dynamic_gammas,
    dynamic_gammadashs,
    tf_curve_groups,
    minimum_distance,
    initial_penalty,
):
    total = _runtime_float64_scalar(initial_penalty, reference=minimum_distance)
    for tf_gammas, tf_gammadashs in tf_curve_groups:
        total = total + _pairwise_curve_distance_penalty_scan(
            dynamic_gammas,
            dynamic_gammadashs,
            _runtime_float64_array(tf_gammas, reference=dynamic_gammas),
            _runtime_float64_array(tf_gammadashs, reference=dynamic_gammadashs),
            minimum_distance,
        )
    return total + _pairwise_curve_distance_penalty_scan(
        dynamic_gammas,
        dynamic_gammadashs,
        dynamic_gammas,
        dynamic_gammadashs,
        minimum_distance,
        strict_lower_triangle=True,
    )


def _summarize_pairwise_row_triplet_sharding(
    left_triplet,
    right_triplet,
) -> dict[str, object]:
    sharded_left, sharded_right = maybe_shard_pairwise_row_trees(
        left_triplet,
        right_triplet,
    )
    left_indices, left_gammas, left_gammadashs = sharded_left
    right_indices, right_gammas, right_gammadashs = sharded_right
    return {
        "left": {
            "indices": summarize_array_sharding(left_indices),
            "gammas": summarize_array_sharding(left_gammas),
            "gammadashs": summarize_array_sharding(left_gammadashs),
        },
        "right": {
            "indices": summarize_array_sharding(right_indices),
            "gammas": summarize_array_sharding(right_gammas),
            "gammadashs": summarize_array_sharding(right_gammadashs),
        },
    }


def build_stage2_target_objective(
    *,
    surface,
    tf_coils,
    banana_coils,
    banana_curve,
    penalty_config: Stage2PenaltyConfig,
):
    """Build the JAX objective bundle for the target Stage 2 lane.

    The returned bundle accepts either the historical Stage 2 free-vector
    ``[banana_current, curve_dofs...]`` or ``Stage2TargetOptimizerState`` with
    the same logical payload split into named fields.
    """
    squared_flux_weight = penalty_config.squared_flux_weight
    length_weight = penalty_config.length_weight
    length_target = penalty_config.length_target
    cc_weight = penalty_config.cc_weight
    cc_threshold = penalty_config.cc_threshold
    curvature_weight = penalty_config.curvature_weight
    curvature_threshold = penalty_config.curvature_threshold
    curvature_p_norm = penalty_config.curvature_p_norm

    field_eval_spec, flux_spec = fixed_surface_flux_specs_from_surface(
        surface,
        definition=penalty_config.squared_flux_definition,
    )
    del field_eval_spec
    points = _host_float64_array(flux_spec.points)
    banana_curve_spec = curve_spec_from_curve(banana_curve)
    curve_dof_count = int(banana_curve_spec.dofs.shape[0])
    length_target = float(length_target)
    cc_threshold = float(cc_threshold)
    curvature_p_norm = float(curvature_p_norm)
    curvature_threshold = float(curvature_threshold)
    half = 0.5
    zero = 0.0
    surface_gamma = _host_float64_array(surface.gamma()).reshape((-1, 3))

    if tf_coils:
        tf_coil_spec = grouped_coil_set_spec_from_coil_specs(
            tuple(coil.to_spec() for coil in tf_coils)
        )
        fixed_field = _host_float64_array(
            grouped_biot_savart_B_from_spec(_as_jax_float64(points), tf_coil_spec)
        )
        tf_curve_data = _curve_pairs_from_grouped_coil_set_spec(tf_coil_spec)
        tf_curve_groups = _curve_groups_from_grouped_coil_set_spec(tf_coil_spec)
        fixed_curve_penalty = _host_float64_scalar(
            _fixed_curve_penalty(tf_curve_data, cc_threshold)
        )
    else:
        fixed_field = np.zeros((points.shape[0], 3), dtype=np.float64)
        tf_curve_groups = ()
        fixed_curve_penalty = 0.0

    fixed_curve_curve_distance_blocks = []
    fixed_curve_surface_distance_blocks = []
    for tf_gammas, _tf_gammadashs in tf_curve_groups:
        fixed_curve_surface_distance_blocks.append(
            np.linalg.norm(
                tf_gammas[:, :, None, :] - surface_gamma[None, None, :, :],
                axis=3,
            ).reshape((-1,))
        )
    for left_group_index, (left_group_gammas, _left_gammadashs) in enumerate(
        tf_curve_groups
    ):
        left_group_size = int(left_group_gammas.shape[0])
        for left_curve_index in range(left_group_size):
            left_gamma = left_group_gammas[left_curve_index]
            for right_group_index in range(left_group_index + 1):
                right_group_gammas, _right_gammadashs = tf_curve_groups[
                    right_group_index
                ]
                right_limit = (
                    left_curve_index
                    if right_group_index == left_group_index
                    else int(right_group_gammas.shape[0])
                )
                for right_curve_index in range(right_limit):
                    right_gamma = right_group_gammas[right_curve_index]
                    fixed_curve_curve_distance_blocks.append(
                        np.linalg.norm(
                            left_gamma[:, None, :] - right_gamma[None, :, :],
                            axis=2,
                        ).reshape((-1,))
                    )

    banana_rotmats, banana_current_scales = _banana_symmetry_runtime_inputs_from_coils(
        banana_coils
    )
    banana_curve_spec_runtime = _runtimeify_tree(banana_curve_spec)
    tf_curve_groups_runtime = _runtimeify_tree(tf_curve_groups)
    flux_spec_runtime = _runtimeify_tree(flux_spec)
    surface_gamma_runtime = _runtimeify_tree(surface_gamma)
    fixed_curve_curve_distance_blocks_runtime = _runtimeify_tree(
        tuple(fixed_curve_curve_distance_blocks)
    )
    fixed_curve_surface_distance_blocks_runtime = _runtimeify_tree(
        tuple(fixed_curve_surface_distance_blocks)
    )

    def _dynamic_curve_runtime_state(dofs):
        state = stage2_target_optimizer_state_from_dofs(
            dofs,
            curve_dof_count=curve_dof_count,
        )
        current_dof = state.current_dof
        curve_dofs = state.curve_dofs
        flat_dofs = stage2_target_optimizer_state_to_dofs(state)
        banana_rotmats_jax = _runtime_float64_array(
            banana_rotmats,
            reference=flat_dofs,
        )
        banana_current_scales_jax = _runtime_float64_array(
            banana_current_scales,
            reference=flat_dofs,
        )
        base_gamma, base_gammadash, base_gammadashdash = curve_geometry_from_dofs(
            banana_curve_spec_runtime,
            curve_dofs,
        )

        dynamic_gammas, dynamic_gammadashs, dynamic_current_array = (
            _build_dynamic_curve_data(
                base_gamma,
                base_gammadash,
                banana_rotmats_jax,
                banana_current_scales_jax,
                current_dof,
            )
        )
        return (
            flat_dofs,
            base_gamma,
            base_gammadash,
            base_gammadashdash,
            dynamic_gammas,
            dynamic_gammadashs,
            dynamic_current_array,
        )

    def _evaluate_dynamic_stage2_state(dofs):
        (
            flat_dofs,
            _base_gamma,
            base_gammadash,
            base_gammadashdash,
            dynamic_gammas,
            dynamic_gammadashs,
            dynamic_current_array,
        ) = _dynamic_curve_runtime_state(dofs)
        fixed_field_jax = _runtime_float64_array(fixed_field, reference=flat_dofs)
        points_jax = _runtime_float64_array(points, reference=flat_dofs)
        length_target_jax = _runtime_float64_scalar(length_target, reference=flat_dofs)
        cc_threshold_jax = _runtime_float64_scalar(cc_threshold, reference=flat_dofs)
        curvature_p_norm_jax = _runtime_float64_scalar(
            curvature_p_norm,
            reference=flat_dofs,
        )
        curvature_threshold_jax = _runtime_float64_scalar(
            curvature_threshold,
            reference=flat_dofs,
        )
        half_jax = _runtime_float64_scalar(half, reference=flat_dofs)
        zero_jax = _runtime_float64_scalar(zero, reference=flat_dofs)

        dynamic_coil_spec = grouped_coil_set_spec_from_lists(
            dynamic_gammas,
            dynamic_gammadashs,
            dynamic_current_array,
        )
        total_field = fixed_field_jax + grouped_biot_savart_B_from_spec(
            points_jax,
            dynamic_coil_spec,
        )

        incremental_arclength = incremental_arclength_pure(base_gammadash)
        curve_length = curve_length_pure(incremental_arclength)
        length_excess = jnp.maximum(curve_length - length_target_jax, zero_jax)
        curvature_penalty = Lp_curvature_pure(
            kappa_pure(base_gammadash, base_gammadashdash),
            base_gammadash,
            curvature_p_norm_jax,
            curvature_threshold_jax,
        )

        coil_distance_penalty = _dynamic_curve_distance_penalty(
            dynamic_gammas,
            dynamic_gammadashs,
            tf_curve_groups_runtime,
            cc_threshold_jax,
            fixed_curve_penalty,
        )

        return (
            flat_dofs,
            total_field,
            length_excess,
            curvature_penalty,
            coil_distance_penalty,
            half_jax,
            zero_jax,
        )

    def _raw_terms(dofs):
        (
            _flat_dofs,
            total_field,
            length_excess,
            curvature_penalty,
            coil_distance_penalty,
            half_jax,
            _zero_jax,
        ) = _evaluate_dynamic_stage2_state(dofs)
        flux = fixed_surface_flux_integral_from_B(total_field, flux_spec_runtime)
        length_penalty = half_jax * (length_excess * length_excess)

        return jnp.stack(
            (
                flux,
                length_penalty,
                coil_distance_penalty,
                curvature_penalty,
            )
        )

    def _least_squares_residual(dofs):
        (
            flat_dofs,
            total_field,
            length_excess,
            curvature_penalty,
            coil_distance_penalty,
            _half_jax,
            zero_jax,
        ) = _evaluate_dynamic_stage2_state(dofs)
        two_jax = _runtime_float64_scalar(2.0, reference=flat_dofs)
        squared_flux_weight_jax = _runtime_float64_scalar(
            squared_flux_weight,
            reference=flat_dofs,
        )
        length_weight_jax = _runtime_float64_scalar(
            length_weight,
            reference=flat_dofs,
        )
        cc_weight_jax = _runtime_float64_scalar(
            cc_weight,
            reference=flat_dofs,
        )
        curvature_weight_jax = _runtime_float64_scalar(
            curvature_weight,
            reference=flat_dofs,
        )
        flux_residual = fixed_surface_flux_residual_from_B(
            total_field,
            flux_spec_runtime,
        )

        penalty_terms = jnp.asarray(
            (
                length_excess * jnp.sqrt(length_weight_jax),
                jnp.sqrt(jnp.maximum(two_jax * cc_weight_jax * coil_distance_penalty, zero_jax)),
                jnp.sqrt(jnp.maximum(two_jax * curvature_weight_jax * curvature_penalty, zero_jax)),
            ),
            dtype=jnp.float64,
        )
        return jnp.concatenate(
            (
                flux_residual * jnp.sqrt(squared_flux_weight_jax),
                penalty_terms,
            )
        )

    terms = (
        Stage2TargetObjectiveTerm("squared_flux", float(squared_flux_weight)),
        Stage2TargetObjectiveTerm("length_penalty", float(length_weight)),
        Stage2TargetObjectiveTerm("coil_distance_penalty", float(cc_weight)),
        Stage2TargetObjectiveTerm("curvature_penalty", float(curvature_weight)),
    )

    raw_terms_fun = jax.jit(_raw_terms)
    least_squares_residual = jax.jit(_least_squares_residual)

    def build_alm_value_and_grad(
        *,
        distance_smoothing,
        curvature_smoothing,
        curve_surface_threshold=None,
        banana_current_threshold,
    ):
        include_coil_surface = curve_surface_threshold is not None

        def _curve_curve_signed_constraint(dynamic_gammas, *, reference):
            pairwise_blocks = []
            for fixed_block in fixed_curve_curve_distance_blocks_runtime:
                pairwise_blocks.append(
                    _runtime_float64_array(fixed_block, reference=reference)
                )
            for dynamic_index in range(int(dynamic_gammas.shape[0])):
                gamma_i = dynamic_gammas[dynamic_index]
                for tf_gammas, _tf_gammadashs in tf_curve_groups_runtime:
                    dists = jnp.linalg.norm(
                        gamma_i[:, None, None, :] - tf_gammas[None, :, :, :],
                        axis=3,
                    )
                    pairwise_blocks.append(dists.reshape((-1,)))
                for previous_index in range(dynamic_index):
                    gamma_j = dynamic_gammas[previous_index]
                    dists = jnp.linalg.norm(
                        gamma_i[:, None, :] - gamma_j[None, :, :],
                        axis=2,
                    )
                    pairwise_blocks.append(dists.reshape((-1,)))
            if not pairwise_blocks:
                return _runtime_float64_scalar(cc_threshold, reference=reference)
            smooth_min = _selected_smoothmin(
                jnp.concatenate(pairwise_blocks),
                temperature=distance_smoothing,
            )
            return _runtime_float64_scalar(cc_threshold, reference=smooth_min) - smooth_min

        def _curve_surface_signed_constraint(dynamic_gammas, *, reference):
            pairwise_blocks = []
            flat_surface = _runtime_float64_array(
                surface_gamma_runtime,
                reference=reference,
            ).reshape((-1, 3))
            for fixed_block in fixed_curve_surface_distance_blocks_runtime:
                pairwise_blocks.append(
                    _runtime_float64_array(fixed_block, reference=reference)
                )
            for dynamic_index in range(int(dynamic_gammas.shape[0])):
                dists = jnp.linalg.norm(
                    dynamic_gammas[dynamic_index][:, None, :] - flat_surface[None, :, :],
                    axis=2,
                )
                pairwise_blocks.append(dists.reshape((-1,)))
            if not pairwise_blocks:
                return _runtime_float64_scalar(
                    curve_surface_threshold,
                    reference=reference,
                )
            smooth_min = _selected_smoothmin(
                jnp.concatenate(pairwise_blocks),
                temperature=distance_smoothing,
            )
            return (
                _runtime_float64_scalar(curve_surface_threshold, reference=smooth_min)
                - smooth_min
            )

        def _constraint_values(dofs):
            (
                flat_dofs,
                _base_gamma,
                base_gammadash,
                base_gammadashdash,
                dynamic_gammas,
                _dynamic_gammadashs,
                dynamic_current_array,
            ) = _dynamic_curve_runtime_state(dofs)
            coil_length = curve_length_pure(incremental_arclength_pure(base_gammadash))
            max_curvature = _selected_smoothmax(
                kappa_pure(base_gammadash, base_gammadashdash),
                temperature=curvature_smoothing,
            )
            banana_current_abs = jnp.max(jnp.abs(dynamic_current_array))
            constraint_values = [
                _curve_curve_signed_constraint(dynamic_gammas, reference=flat_dofs),
            ]
            if include_coil_surface:
                constraint_values.append(
                    _curve_surface_signed_constraint(
                        dynamic_gammas,
                        reference=flat_dofs,
                    )
                )
            constraint_values.extend(
                (
                    max_curvature
                    - _runtime_float64_scalar(
                        curvature_threshold,
                        reference=max_curvature,
                    ),
                    coil_length
                    - _runtime_float64_scalar(length_target, reference=coil_length),
                    banana_current_abs
                    - _runtime_float64_scalar(
                        banana_current_threshold,
                        reference=banana_current_abs,
                    ),
                )
            )
            return jnp.stack(constraint_values)

        def _alm_objective_impl(dofs, multipliers, penalty):
            raw_terms_value = raw_terms_fun(dofs)
            squared_flux_weight_jax = _runtime_float64_scalar(
                squared_flux_weight,
                reference=raw_terms_value,
            )
            base_value = squared_flux_weight_jax * raw_terms_value[0]
            constraint_values = _constraint_values(dofs)
            multipliers_arr = _as_jax_float64(multipliers).reshape((-1,))
            penalty_value = jnp.asarray(penalty, dtype=jnp.float64)
            positive_shift = jnp.maximum(
                _runtime_float64_scalar(0.0, reference=constraint_values),
                multipliers_arr + penalty_value * constraint_values,
            )
            return base_value + _runtime_float64_scalar(
                0.5, reference=constraint_values
            ) / penalty_value * (
                jnp.vdot(positive_shift, positive_shift)
                - jnp.vdot(multipliers_arr, multipliers_arr)
            )

        alm_value_and_grad = _mark_cacheable_jit_value_and_grad(
            jax.jit(jax.value_and_grad(_alm_objective_impl))
        )
        return _mark_structured_private_solver_cacheable(
            alm_value_and_grad,
            cache_token=(
                "stage2-target-alm",
                bool(include_coil_surface),
                float(distance_smoothing),
                float(curvature_smoothing),
                (
                    None
                    if curve_surface_threshold is None
                    else float(curve_surface_threshold)
                ),
                float(banana_current_threshold),
            ),
        )

    def objective_impl(dofs):
        raw_terms_value = raw_terms_fun(dofs)
        squared_flux_weight_jax = _runtime_float64_scalar(
            squared_flux_weight,
            reference=raw_terms_value,
        )
        length_weight_jax = _runtime_float64_scalar(
            length_weight,
            reference=raw_terms_value,
        )
        cc_weight_jax = _runtime_float64_scalar(
            cc_weight,
            reference=raw_terms_value,
        )
        curvature_weight_jax = _runtime_float64_scalar(
            curvature_weight,
            reference=raw_terms_value,
        )
        return (
            squared_flux_weight_jax * raw_terms_value[0]
            + length_weight_jax * raw_terms_value[1]
            + cc_weight_jax * raw_terms_value[2]
            + curvature_weight_jax * raw_terms_value[3]
        )

    objective = jax.jit(objective_impl)
    value_and_grad = _mark_cacheable_jit_value_and_grad(
        jax.jit(jax.value_and_grad(objective_impl))
    )
    value_and_grad = _mark_structured_private_solver_cacheable(
        value_and_grad,
        cache_token=("stage2-target-objective",),
    )

    def _dynamic_field_collective_summary(dofs):
        (
            flat_dofs,
            base_gamma,
            base_gammadash,
            base_gammadashdash,
            dynamic_gammas,
            dynamic_gammadashs,
            dynamic_current_array,
        ) = _dynamic_curve_runtime_state(dofs)
        del base_gamma, base_gammadash, base_gammadashdash
        dynamic_coil_spec = grouped_coil_set_spec_from_lists(
            dynamic_gammas,
            dynamic_gammadashs,
            dynamic_current_array,
        )
        return grouped_field_sharding_summary(
            _runtime_float64_array(points, reference=flat_dofs),
            dynamic_coil_spec,
        )

    def field_sharding_summary(dofs):
        _, total_field, *_ = _evaluate_dynamic_stage2_state(dofs)
        summary = summarize_array_sharding(total_field)
        dynamic_summary = _dynamic_field_collective_summary(dofs)
        summary["field_collective"] = dynamic_summary["field_collective"]
        if dynamic_summary["field_collective"]:
            summary["strategy"] = dynamic_summary["strategy"]
            summary["collective_axis"] = dynamic_summary["collective_axis"]
            summary["collective_mesh_shape"] = dynamic_summary["mesh_shape"]
            summary["collective_device_count"] = dynamic_summary[
                "collective_device_count"
            ]
        return summary

    def pairwise_penalty_sharding_summary(dofs):
        (
            _flat_dofs,
            _base_gamma,
            _base_gammadash,
            _base_gammadashdash,
            dynamic_gammas,
            dynamic_gammadashs,
            _dynamic_current_array,
        ) = _dynamic_curve_runtime_state(dofs)
        dynamic_triplet = (
            jnp.arange(dynamic_gammas.shape[0], dtype=jnp.int32),
            dynamic_gammas,
            dynamic_gammadashs,
        )
        tf_group_summaries = []
        for group_index, (tf_gammas, tf_gammadashs) in enumerate(
            tf_curve_groups_runtime
        ):
            tf_group_triplet = (
                jnp.arange(tf_gammas.shape[0], dtype=jnp.int32),
                _runtime_float64_array(tf_gammas, reference=dynamic_gammas),
                _runtime_float64_array(tf_gammadashs, reference=dynamic_gammadashs),
            )
            tf_group_summaries.append(
                {
                    "group_index": group_index,
                    "right_row_count": int(tf_gammas.shape[0]),
                    **_summarize_pairwise_row_triplet_sharding(
                        dynamic_triplet,
                        tf_group_triplet,
                    ),
                }
            )
        return {
            "dynamic_row_count": int(dynamic_gammas.shape[0]),
            "dynamic_vs_tf_groups": tf_group_summaries,
            "dynamic_self": {
                "right_row_count": int(dynamic_gammas.shape[0]),
                "strict_lower_triangle": True,
                **_summarize_pairwise_row_triplet_sharding(
                    dynamic_triplet,
                    dynamic_triplet,
                ),
            },
        }

    return Stage2TargetObjectiveBundle(
        objective=objective,
        expected_dof_count=curve_dof_count + 1,
        value_and_grad=value_and_grad,
        terms=terms,
        raw_terms=raw_terms_fun,
        least_squares_residual=least_squares_residual,
        alm_value_and_grad_builder=build_alm_value_and_grad,
        field_sharding_summary=field_sharding_summary,
        pairwise_penalty_sharding_summary=pairwise_penalty_sharding_summary,
    )
