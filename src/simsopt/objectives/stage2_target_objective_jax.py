from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import jax.numpy as jnp

from ..field.biotsavart_jax import biot_savart_B, group_coil_data, grouped_biot_savart_B
from ..field.coil import ScaledCurrent
from ..geo.curve import RotatedCurve, incremental_arclength_pure, kappa_pure
from ..geo.curveobjectives import Lp_curvature_pure, cc_distance_pure, curve_length_pure
from .integral_bdotn_jax import integral_BdotN

__all__ = ["Stage2TargetObjectiveBundle", "build_stage2_target_objective"]


class Stage2TargetObjectiveBundle(NamedTuple):
    objective: Callable[[jnp.ndarray], jnp.ndarray]
    expected_dof_count: int


def _as_jax_float64_array(values, *, contiguous=False):
    if contiguous:
        values = np.ascontiguousarray(values)
    return jnp.asarray(values, dtype=jnp.float64)


def _rotation_and_scale_from_coil(coil):
    curve = coil.curve
    rotmat = None
    while isinstance(curve, RotatedCurve):
        next_rotmat = _as_jax_float64_array(curve.rotmat)
        rotmat = next_rotmat if rotmat is None else next_rotmat @ rotmat
        curve = curve.curve

    current = coil.current
    scale = 1.0
    while isinstance(current, ScaledCurrent):
        scale *= float(current.scale)
        current = current.current_to_scale

    return curve, rotmat, scale


def _fixed_curve_penalty(curves, minimum_distance):
    total = jnp.asarray(0.0, dtype=jnp.float64)
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


def _build_dynamic_curve_data(
    base_gamma,
    base_gammadash,
    banana_descriptors,
    current_dof,
):
    dynamic_gammas = []
    dynamic_gammadashs = []
    dynamic_currents = []
    for rotmat, scale in banana_descriptors:
        gamma = base_gamma if rotmat is None else base_gamma @ rotmat
        gammadash = base_gammadash if rotmat is None else base_gammadash @ rotmat
        dynamic_gammas.append(gamma)
        dynamic_gammadashs.append(gammadash)
        dynamic_currents.append(scale * current_dof)
    return (
        tuple(dynamic_gammas),
        tuple(dynamic_gammadashs),
        _as_jax_float64_array(dynamic_currents),
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
    squared_flux_weight,
    length_weight,
    length_target,
    cc_weight,
    cc_threshold,
    curvature_weight,
    curvature_threshold,
    curvature_p_norm,
):
    """Build a scalar JAX objective for the target Stage 2 lane.

    The returned callable consumes the Stage 2 free-vector in the same order as
    the existing composite objective contract: ``[banana_current, curve_dofs...]``.
    """
    points = _as_jax_float64_array(surface.gamma().reshape((-1, 3)), contiguous=True)
    normal = _as_jax_float64_array(surface.normal(), contiguous=True)
    target = jnp.zeros(normal.shape[:2], dtype=jnp.float64)
    surf_dofs = _as_jax_float64_array(np.asarray(banana_curve.surf.get_dofs()))
    curve_dof_count = int(banana_curve.num_dofs())

    tf_groups = tuple(
        (gammas, gammadashs, currents)
        for gammas, gammadashs, currents, _ in group_coil_data(
            [coil.curve.gamma() for coil in tf_coils],
            [coil.curve.gammadash() for coil in tf_coils],
            [coil.current.get_value() for coil in tf_coils],
        )
    )
    if tf_groups:
        fixed_field = grouped_biot_savart_B(points, tf_groups)
    else:
        fixed_field = jnp.zeros((points.shape[0], 3), dtype=jnp.float64)

    tf_curve_data = tuple(
        (
            _as_jax_float64_array(coil.curve.gamma(), contiguous=True),
            _as_jax_float64_array(coil.curve.gammadash(), contiguous=True),
        )
        for coil in tf_coils
    )
    fixed_curve_penalty = _fixed_curve_penalty(tf_curve_data, cc_threshold)

    banana_descriptors = []
    for coil in banana_coils:
        _, rotmat, scale = _rotation_and_scale_from_coil(coil)
        banana_descriptors.append(
            (
                rotmat,
                _as_jax_float64_array(scale),
            )
        )
    banana_descriptors = tuple(banana_descriptors)

    def objective(dofs):
        dofs = jnp.asarray(dofs, dtype=jnp.float64)
        current_dof = dofs[0]
        curve_dofs = dofs[1 : 1 + curve_dof_count]

        base_gamma = banana_curve.gamma_jax(curve_dofs, surf_dofs)
        base_gammadash = banana_curve.gammadash_jax(curve_dofs, surf_dofs)
        base_gammadashdash = banana_curve.gammadashdash_jax(curve_dofs, surf_dofs)

        dynamic_gammas, dynamic_gammadashs, dynamic_current_array = _build_dynamic_curve_data(
            base_gamma,
            base_gammadash,
            banana_descriptors,
            current_dof,
        )
        dynamic_pairs = tuple(zip(dynamic_gammas, dynamic_gammadashs))
        dynamic_field = biot_savart_B(
            points,
            jnp.stack(dynamic_gammas),
            jnp.stack(dynamic_gammadashs),
            dynamic_current_array,
        )
        flux = integral_BdotN(
            (fixed_field + dynamic_field).reshape(normal.shape),
            target,
            normal,
            definition="quadratic flux",
        )

        incremental_arclength = incremental_arclength_pure(base_gammadash)
        curve_length = curve_length_pure(incremental_arclength)
        length_penalty = 0.5 * jnp.maximum(curve_length - length_target, 0.0) ** 2

        curvature_penalty = Lp_curvature_pure(
            kappa_pure(base_gammadash, base_gammadashdash),
            base_gammadash,
            curvature_p_norm,
            curvature_threshold,
        )

        coil_distance_penalty = _dynamic_curve_distance_penalty(
            dynamic_pairs,
            tf_curve_data,
            cc_threshold,
            fixed_curve_penalty,
        )

        return (
            squared_flux_weight * flux
            + length_weight * length_penalty
            + cc_weight * coil_distance_penalty
            + curvature_weight * curvature_penalty
        )

    return Stage2TargetObjectiveBundle(
        objective=objective,
        expected_dof_count=curve_dof_count + 1,
    )
