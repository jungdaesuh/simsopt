"""JAX objective bundle used by the Stage 2 ondevice target lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..field.biotsavart_jax_backend import _unwrap_coil_curve_and_current
from ..geo.curve import incremental_arclength_pure, kappa_pure
from ..jax_core._math_utils import as_runtime_float64 as _as_runtime_float64
from ..jax_core.field import (
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_coil_set_spec_from_lists,
)
from ..jax_core import (
    apply_coil_symmetry,
    curve_geometry_from_dofs,
    curve_spec_from_curve,
    make_coil_symmetry_spec,
)
from ..jax_core.objectives_flux import (
    fixed_surface_flux_integral_from_B,
    fixed_surface_flux_residual_from_B,
    fixed_surface_flux_specs_from_surface,
)
from ..geo.curveobjectives import (
    Lp_curvature_pure,
    cc_distance_pure,
    curve_length_pure,
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


def _as_jax_float64(value) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float64)


def _device_float64_array(value) -> jax.Array:
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _runtime_float64_array(value, *, reference) -> jax.Array:
    return _as_runtime_float64(value, reference=reference)


def _runtime_float64_scalar(value, *, reference) -> jax.Array:
    return _as_runtime_float64(value, reference=reference)


def _as_objective_dofs(value) -> jax.Array:
    if isinstance(value, np.ndarray):
        return _device_float64_array(value)
    return _as_jax_float64(value)


def _as_host_float64(value) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _hostify_tree(value):
    def _hostify_leaf(leaf):
        if isinstance(leaf, jax.Array):
            return _as_host_float64(leaf)
        if isinstance(leaf, np.ndarray):
            return np.asarray(leaf, dtype=np.float64)
        return leaf

    return jax.tree_util.tree_map(_hostify_leaf, value)


def _runtimeify_tree(value):
    return _hostify_tree(value)


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
    current_dof = dofs[0]
    if curve_dof_count is None:
        curve_dofs = dofs[1:]
    else:
        curve_dofs = jax.lax.dynamic_slice_in_dim(
            dofs,
            start_index=1,
            slice_size=int(curve_dof_count),
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
    total = _device_float64_array(0.0)
    minimum_distance_jax = _device_float64_array(minimum_distance)
    for i, (gamma_i, gammadash_i) in enumerate(curves):
        gamma_i_jax = _device_float64_array(gamma_i)
        gammadash_i_jax = _device_float64_array(gammadash_i)
        for gamma_j, gammadash_j in curves[:i]:
            total = total + cc_distance_pure(
                gamma_i_jax,
                gammadash_i_jax,
                _device_float64_array(gamma_j),
                _device_float64_array(gammadash_j),
                minimum_distance_jax,
            )
    return total


def _curve_pairs_from_grouped_coil_set_spec(coil_set_spec):
    curve_pairs = []
    for group in coil_set_spec.groups:
        gammas = np.asarray(jax.device_get(group.gammas), dtype=np.float64)
        gammadashs = np.asarray(jax.device_get(group.gammadashs), dtype=np.float64)
        for gamma, gammadash in zip(gammas, gammadashs):
            curve_pairs.append((gamma, gammadash))
    return tuple(curve_pairs)


def _build_dynamic_curve_data(
    base_gamma,
    base_gammadash,
    banana_symmetry_specs,
    current_dof,
):
    dynamic_gammas = []
    dynamic_gammadashs = []
    dynamic_currents = []
    for symmetry_spec in banana_symmetry_specs:
        gamma, gammadash, current = apply_coil_symmetry(
            base_gamma,
            base_gammadash,
            current_dof,
            symmetry_spec,
        )
        dynamic_gammas.append(gamma)
        dynamic_gammadashs.append(gammadash)
        dynamic_currents.append(current)
    return (
        tuple(dynamic_gammas),
        tuple(dynamic_gammadashs),
        jnp.stack(dynamic_currents),
    )


def _dynamic_curve_distance_penalty(
    dynamic_pairs,
    tf_curve_data,
    minimum_distance,
    initial_penalty,
):
    total = _runtime_float64_scalar(initial_penalty, reference=minimum_distance)
    for gamma, gammadash in dynamic_pairs:
        for tf_gamma, tf_gammadash in tf_curve_data:
            total = total + cc_distance_pure(
                gamma,
                gammadash,
                _runtime_float64_array(tf_gamma, reference=gamma),
                _runtime_float64_array(tf_gammadash, reference=gammadash),
                minimum_distance,
            )
    for i, (gamma_i, gammadash_i) in enumerate(dynamic_pairs):
        for gamma_j, gammadash_j in dynamic_pairs[:i]:
            total = total + cc_distance_pure(
                gamma_i,
                gammadash_i,
                gamma_j,
                gammadash_j,
                minimum_distance,
            )
    return total


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
    flux_spec = _hostify_tree(flux_spec)
    points = np.asarray(flux_spec.points, dtype=np.float64)
    banana_curve_spec = _hostify_tree(curve_spec_from_curve(banana_curve))
    curve_dof_count = int(np.asarray(banana_curve_spec.dofs, dtype=np.float64).shape[0])
    length_target = float(length_target)
    cc_threshold = float(cc_threshold)
    curvature_p_norm = float(curvature_p_norm)
    curvature_threshold = float(curvature_threshold)
    half = 0.5
    zero = 0.0

    if tf_coils:
        tf_coil_spec = grouped_coil_set_spec_from_coil_specs(
            tuple(coil.to_spec() for coil in tf_coils)
        )
        points_jax = _device_float64_array(points)
        fixed_field = _as_host_float64(
            grouped_biot_savart_B_from_spec(points_jax, tf_coil_spec)
        )
        tf_curve_data = _curve_pairs_from_grouped_coil_set_spec(tf_coil_spec)
        fixed_curve_penalty = float(
            _as_host_float64(_fixed_curve_penalty(tf_curve_data, cc_threshold))
        )
    else:
        fixed_field = np.zeros((points.shape[0], 3), dtype=np.float64)
        tf_curve_data = ()
        fixed_curve_penalty = 0.0

    banana_symmetry_specs = _hostify_tree(
        tuple(
            make_coil_symmetry_spec(
                rotmat=rotmat,
                scale=scale,
            )
            for _, rotmat, _, scale in (
                _unwrap_coil_curve_and_current(coil) for coil in banana_coils
            )
        )
    )
    banana_curve_spec_runtime = _runtimeify_tree(banana_curve_spec)
    banana_symmetry_specs_runtime = _runtimeify_tree(banana_symmetry_specs)
    tf_curve_data_runtime = _runtimeify_tree(tf_curve_data)
    flux_spec_runtime = _runtimeify_tree(flux_spec)
    objective_weights = np.array(
        (
            squared_flux_weight,
            length_weight,
            cc_weight,
            curvature_weight,
        ),
        dtype=np.float64,
    )
    least_squares_weights = np.asarray(objective_weights, dtype=np.float64)

    def _evaluate_dynamic_stage2_state(dofs):
        state = stage2_target_optimizer_state_from_dofs(
            dofs,
            curve_dof_count=curve_dof_count,
        )
        current_dof = state.current_dof
        curve_dofs = state.curve_dofs
        flat_dofs = stage2_target_optimizer_state_to_dofs(state)
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

        base_gamma, base_gammadash, base_gammadashdash = curve_geometry_from_dofs(
            banana_curve_spec_runtime,
            curve_dofs,
        )

        dynamic_gammas, dynamic_gammadashs, dynamic_current_array = (
            _build_dynamic_curve_data(
                base_gamma,
                base_gammadash,
                banana_symmetry_specs_runtime,
                current_dof,
            )
        )
        dynamic_pairs = tuple(zip(dynamic_gammas, dynamic_gammadashs))
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
            dynamic_pairs,
            tf_curve_data_runtime,
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
            _zero_jax,
        ) = _evaluate_dynamic_stage2_state(dofs)
        two_jax = _runtime_float64_scalar(2.0, reference=flat_dofs)
        least_squares_weights_jax = _runtime_float64_array(
            least_squares_weights,
            reference=flat_dofs,
        )
        flux_residual = fixed_surface_flux_residual_from_B(
            total_field,
            flux_spec_runtime,
        )

        penalty_terms = jnp.asarray(
            (
                length_excess * jnp.sqrt(least_squares_weights_jax[1]),
                jnp.sqrt(two_jax * least_squares_weights_jax[2] * coil_distance_penalty),
                jnp.sqrt(two_jax * least_squares_weights_jax[3] * curvature_penalty),
            ),
            dtype=jnp.float64,
        )
        return jnp.concatenate(
            (
                flux_residual * jnp.sqrt(least_squares_weights_jax[0]),
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

    def objective_impl(dofs):
        raw_terms_value = raw_terms_fun(dofs)
        weights = _runtime_float64_array(objective_weights, reference=raw_terms_value)
        return jnp.dot(weights, raw_terms_value)

    objective = jax.jit(objective_impl)
    value_and_grad = jax.jit(jax.value_and_grad(objective_impl))

    return Stage2TargetObjectiveBundle(
        objective=objective,
        expected_dof_count=curve_dof_count + 1,
        value_and_grad=value_and_grad,
        terms=terms,
        raw_terms=raw_terms_fun,
        least_squares_residual=least_squares_residual,
    )
