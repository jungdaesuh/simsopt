"""Scalar JAX objective used by the Stage 2 ondevice target lane."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..field.biotsavart_jax_backend import _unwrap_coil_curve_and_current
from ..geo.curve import incremental_arclength_pure, kappa_pure
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
    "Stage2TargetObjectiveTerm",
    "build_stage2_target_objective",
]

Stage2ObjectiveFn = Callable[[jnp.ndarray], jnp.ndarray]


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


class Stage2TargetObjectiveBundle(NamedTuple):
    objective: Stage2ObjectiveFn
    expected_dof_count: int
    terms: tuple[Stage2TargetObjectiveTerm, ...] = ()
    raw_terms: Stage2ObjectiveFn | None = None


def _as_jax_float64(value) -> jax.Array:
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.float64)
    if isinstance(value, (np.ndarray, np.generic, list, tuple)) or np.isscalar(value):
        return jax.device_put(np.asarray(value, dtype=np.float64))
    return jnp.asarray(value, dtype=jnp.float64)


def _zero_scalar():
    return _as_jax_float64(0.0)


def _split_stage2_dofs(dofs, curve_dof_count):
    dofs = _as_jax_float64(dofs)
    total_dofs = curve_dof_count + 1
    current_selector = _as_jax_float64(
        np.concatenate(
            (
                np.array([1.0], dtype=np.float64),
                np.zeros(curve_dof_count, dtype=np.float64),
            )
        )
    )
    curve_selector = _as_jax_float64(
        np.eye(curve_dof_count, total_dofs, k=1, dtype=np.float64)
    )
    return jnp.vdot(current_selector, dofs), curve_selector @ dofs


def _fixed_curve_penalty(curves, minimum_distance):
    total = _zero_scalar()
    for i, (gamma_i, gammadash_i) in enumerate(curves):
        for gamma_j, gammadash_j in curves[:i]:
            total = total + cc_distance_pure(
                gamma_i,
                gammadash_i,
                gamma_j,
                gammadash_j,
                minimum_distance,
            )
    return total


def _curve_pairs_from_grouped_coil_set_spec(coil_set_spec):
    curve_pairs = []
    for group in coil_set_spec.groups:
        gammas = np.asarray(jax.device_get(group.gammas), dtype=np.float64)
        gammadashs = np.asarray(jax.device_get(group.gammadashs), dtype=np.float64)
        for gamma, gammadash in zip(gammas, gammadashs):
            curve_pairs.append((_as_jax_float64(gamma), _as_jax_float64(gammadash)))
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
    total = initial_penalty
    for gamma, gammadash in dynamic_pairs:
        for tf_gamma, tf_gammadash in tf_curve_data:
            total = total + cc_distance_pure(
                gamma,
                gammadash,
                tf_gamma,
                tf_gammadash,
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
    """Build a scalar JAX objective for the target Stage 2 lane.

    The returned callable consumes the Stage 2 free-vector in the same order as
    the existing composite objective contract: ``[banana_current, curve_dofs...]``.
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
    points = flux_spec.points
    banana_curve_spec = curve_spec_from_curve(banana_curve)
    curve_dof_count = int(banana_curve_spec.dofs.shape[0])
    length_target_jax = _as_jax_float64(length_target)
    cc_threshold_jax = _as_jax_float64(cc_threshold)
    curvature_p_norm_jax = _as_jax_float64(curvature_p_norm)
    curvature_threshold_jax = _as_jax_float64(curvature_threshold)
    half = _as_jax_float64(0.5)
    zero = _zero_scalar()

    if tf_coils:
        tf_coil_spec = grouped_coil_set_spec_from_coil_specs(
            tuple(coil.to_spec() for coil in tf_coils)
        )
        fixed_field = grouped_biot_savart_B_from_spec(points, tf_coil_spec)
        tf_curve_data = _curve_pairs_from_grouped_coil_set_spec(tf_coil_spec)
        fixed_curve_penalty = _fixed_curve_penalty(tf_curve_data, cc_threshold_jax)
    else:
        fixed_field = _as_jax_float64(np.zeros((points.shape[0], 3), dtype=np.float64))
        tf_curve_data = ()
        fixed_curve_penalty = _zero_scalar()

    banana_symmetry_specs = tuple(
        make_coil_symmetry_spec(
            rotmat=rotmat,
            scale=scale,
        )
        for _, rotmat, _, scale in (
            _unwrap_coil_curve_and_current(coil) for coil in banana_coils
        )
    )

    def _raw_terms(dofs):
        current_dof, curve_dofs = _split_stage2_dofs(dofs, curve_dof_count)

        base_gamma, base_gammadash, base_gammadashdash = curve_geometry_from_dofs(
            banana_curve_spec,
            curve_dofs,
        )

        dynamic_gammas, dynamic_gammadashs, dynamic_current_array = (
            _build_dynamic_curve_data(
                base_gamma,
                base_gammadash,
                banana_symmetry_specs,
                current_dof,
            )
        )
        dynamic_pairs = tuple(zip(dynamic_gammas, dynamic_gammadashs))
        dynamic_coil_spec = grouped_coil_set_spec_from_lists(
            dynamic_gammas,
            dynamic_gammadashs,
            dynamic_current_array,
        )
        dynamic_field = grouped_biot_savart_B_from_spec(points, dynamic_coil_spec)
        flux = fixed_surface_flux_integral_from_B(
            fixed_field + dynamic_field,
            flux_spec,
        )

        incremental_arclength = incremental_arclength_pure(base_gammadash)
        curve_length = curve_length_pure(incremental_arclength)
        length_excess = jnp.maximum(curve_length - length_target_jax, zero)
        length_penalty = half * (length_excess * length_excess)

        curvature_penalty = Lp_curvature_pure(
            kappa_pure(base_gammadash, base_gammadashdash),
            base_gammadash,
            curvature_p_norm_jax,
            curvature_threshold_jax,
        )

        coil_distance_penalty = _dynamic_curve_distance_penalty(
            dynamic_pairs,
            tf_curve_data,
            cc_threshold_jax,
            fixed_curve_penalty,
        )

        return jnp.stack(
            (
                flux,
                length_penalty,
                coil_distance_penalty,
                curvature_penalty,
            )
        )

    terms = (
        Stage2TargetObjectiveTerm("squared_flux", float(squared_flux_weight)),
        Stage2TargetObjectiveTerm("length_penalty", float(length_weight)),
        Stage2TargetObjectiveTerm("coil_distance_penalty", float(cc_weight)),
        Stage2TargetObjectiveTerm("curvature_penalty", float(curvature_weight)),
    )
    term_weights = _as_jax_float64(
        np.array(
            [
                squared_flux_weight,
                length_weight,
                cc_weight,
                curvature_weight,
            ],
            dtype=np.float64,
        )
    )

    def objective(dofs):
        return jnp.vdot(term_weights, _raw_terms(dofs))

    return Stage2TargetObjectiveBundle(
        objective=objective,
        expected_dof_count=curve_dof_count + 1,
        terms=terms,
        raw_terms=_raw_terms,
    )
