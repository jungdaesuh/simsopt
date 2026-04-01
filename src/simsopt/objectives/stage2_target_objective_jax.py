"""Scalar JAX objective used by the Stage 2 ondevice target lane."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp

from ..field.biotsavart_jax_backend import _unwrap_coil_curve_and_current
from ..geo.curve import incremental_arclength_pure, kappa_pure
from ..jax_core.field import (
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_coil_specs,
    grouped_coil_set_spec_from_lists,
)
from ..jax_core import (
    apply_coil_symmetry,
    curve_gamma_from_dofs,
    curve_gammadash_from_dofs,
    curve_gammadashdash_from_dofs,
    curve_spec_from_curve,
    make_coil_symmetry_spec,
)
from ..jax_core.objectives_flux import (
    fixed_surface_flux_integral_from_B,
    fixed_surface_flux_specs_from_surface,
)
from ..geo.curveobjectives import (
    curvature_barrier_pure,
    cc_distance_barrier_pure,
    curve_length_pure,
)

__all__ = [
    "Stage2TargetObjectiveBundle",
    "Stage2TargetObjectiveTerm",
    "build_stage2_target_objective",
]

Stage2ObjectiveFn = Callable[[jnp.ndarray], jnp.ndarray]


class Stage2TargetObjectiveTerm(NamedTuple):
    name: str
    weight: float


class Stage2TargetObjectiveBundle(NamedTuple):
    objective: Stage2ObjectiveFn
    expected_dof_count: int
    terms: tuple[Stage2TargetObjectiveTerm, ...] = ()
    raw_terms: Stage2ObjectiveFn | None = None


def _fixed_curve_penalty(curves, minimum_distance):
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for i, (gamma_i, gammadash_i) in enumerate(curves):
        for gamma_j, gammadash_j in curves[:i]:
            total = total + cc_distance_barrier_pure(
                gamma_i,
                gammadash_i,
                gamma_j,
                gammadash_j,
                minimum_distance,
            )
    return total


def _curve_pairs_from_grouped_coil_set_spec(coil_set_spec):
    return tuple(
        (group.gammas[index], group.gammadashs[index])
        for group in coil_set_spec.groups
        for index in range(group.gammas.shape[0])
    )


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
            base_gamma, base_gammadash, current_dof, symmetry_spec,
        )
        dynamic_gammas.append(gamma)
        dynamic_gammadashs.append(gammadash)
        dynamic_currents.append(current)
    return (
        tuple(dynamic_gammas),
        tuple(dynamic_gammadashs),
        jnp.asarray(dynamic_currents, dtype=jnp.float64),
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
            total = total + cc_distance_barrier_pure(
                gamma,
                gammadash,
                tf_gamma,
                tf_gammadash,
                minimum_distance,
            )
    for i, (gamma_i, gammadash_i) in enumerate(dynamic_pairs):
        for gamma_j, gammadash_j in dynamic_pairs[:i]:
            total = total + cc_distance_barrier_pure(
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
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    curvature_threshold,
    curvature_p_norm,
    squared_flux_definition="quadratic flux",
):
    """Build a scalar JAX objective for the target Stage 2 lane.

    The returned callable consumes the Stage 2 free-vector in the same order as
    the existing composite objective contract: ``[banana_current, curve_dofs...]``.
    """
    field_eval_spec, flux_spec = fixed_surface_flux_specs_from_surface(
        surface,
        definition=squared_flux_definition,
    )
    del field_eval_spec
    points = flux_spec.points
    banana_curve_spec = curve_spec_from_curve(banana_curve)
    curve_dof_count = int(banana_curve_spec.dofs.shape[0])

    if tf_coils:
        tf_coil_spec = grouped_coil_set_spec_from_coil_specs(
            tuple(coil.to_spec() for coil in tf_coils)
        )
        fixed_field = grouped_biot_savart_B_from_spec(points, tf_coil_spec)
        tf_curve_data = _curve_pairs_from_grouped_coil_set_spec(tf_coil_spec)
        fixed_curve_penalty = _fixed_curve_penalty(tf_curve_data, cc_threshold)
    else:
        fixed_field = jnp.zeros((points.shape[0], 3), dtype=jnp.float64)
        tf_curve_data = ()
        fixed_curve_penalty = jnp.asarray(0.0, dtype=jnp.float64)

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
        dofs = jnp.asarray(dofs, dtype=jnp.float64)
        current_dof = dofs[0]
        curve_dofs = dofs[1 : 1 + curve_dof_count]

        base_gamma = curve_gamma_from_dofs(banana_curve_spec, curve_dofs)
        base_gammadash = curve_gammadash_from_dofs(banana_curve_spec, curve_dofs)
        base_gammadashdash = curve_gammadashdash_from_dofs(
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
        length_penalty = 0.5 * jnp.maximum(curve_length - length_target, 0.0) ** 2

        curvature_penalty = curvature_barrier_pure(
            kappa_pure(base_gammadash, base_gammadashdash),
            base_gammadash,
            curvature_threshold,
        )

        coil_distance_penalty = _dynamic_curve_distance_penalty(
            dynamic_pairs,
            tf_curve_data,
            cc_threshold,
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
        Stage2TargetObjectiveTerm("coil_distance_barrier", float(cc_weight)),
        Stage2TargetObjectiveTerm("curvature_barrier", float(curvature_weight)),
    )

    def objective(dofs):
        raw_terms = _raw_terms(dofs)
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for index, term in enumerate(terms):
            total = total + term.weight * raw_terms[index]
        return total

    return Stage2TargetObjectiveBundle(
        objective=objective,
        expected_dof_count=curve_dof_count + 1,
        terms=terms,
        raw_terms=_raw_terms,
    )
