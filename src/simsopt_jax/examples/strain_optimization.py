"""Device-resident strain objective and memory-bounded L-BFGS workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from simsopt_jax.geo.optimizers.optimizer import target_minimize

from simsopt_jax.core._math_utils import as_runtime_array as _as_runtime_array
from simsopt_jax.core.framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotation_alpha,
    rotation_alphadash,
)

__all__ = (
    "StrainOptimizationDeviceResult",
    "StrainState",
    "solve_strain_rotation",
)


@dataclass(frozen=True)
class StrainState:
    """Objective, derivative, and physical strains at one rotation state."""

    parameters: jax.Array
    objective: jax.Array
    gradient: jax.Array
    torsional_strain: jax.Array
    binormal_curvature_strain: jax.Array
    maximum_torsional_strain: jax.Array
    maximum_binormal_curvature_strain: jax.Array


jax.tree_util.register_dataclass(
    StrainState,
    data_fields=[
        "parameters",
        "objective",
        "gradient",
        "torsional_strain",
        "binormal_curvature_strain",
        "maximum_torsional_strain",
        "maximum_binormal_curvature_strain",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class StrainOptimizationDeviceResult:
    """Initial/final strain states and fixed-size device solver diagnostics."""

    initial: StrainState
    final: StrainState
    success: jax.Array
    status: jax.Array
    iterations: jax.Array
    function_evaluations: jax.Array
    gradient_evaluations: jax.Array


jax.tree_util.register_dataclass(
    StrainOptimizationDeviceResult,
    data_fields=[
        "initial",
        "final",
        "success",
        "status",
        "iterations",
        "function_evaluations",
        "gradient_evaluations",
    ],
    meta_fields=[],
)


def _strain_values(
    quadpoints: jax.Array,
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    rotation_dofs: jax.Array,
    *,
    rotation_order: int,
    width: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    alpha = rotation_alpha(rotation_dofs, quadpoints, rotation_order)
    alphadash = rotation_alphadash(
        rotation_dofs,
        quadpoints,
        rotation_order,
    )
    _tangent, _normal, binormal = rotated_centroid_frame(
        gamma,
        gammadash,
        alpha,
    )
    tangent_dash, normal_dash, _binormal_dash = rotated_centroid_frame_dash(
        gamma,
        gammadash,
        gammadashdash,
        alpha,
        alphadash,
    )
    torsion = jnp.sum(
        (normal_dash / arc_length[:, None]) * binormal,
        axis=1,
    )
    binormal_curvature = jnp.sum(
        (tangent_dash / arc_length[:, None]) * binormal,
        axis=1,
    )
    return (
        torsion**2 * width**2 / 12.0,
        width * jnp.abs(binormal_curvature) / 2.0,
        arc_length,
    )


def _strain_objective(
    quadpoints: jax.Array,
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    rotation_dofs: jax.Array,
    *,
    rotation_order: int,
    objective_width: float,
    torsional_threshold: float,
    curvature_threshold: float,
) -> jax.Array:
    torsional_strain, binormal_strain, arc_length = _strain_values(
        quadpoints,
        gamma,
        gammadash,
        gammadashdash,
        rotation_dofs,
        rotation_order=rotation_order,
        width=objective_width,
    )
    torsional_excess = jnp.maximum(
        torsional_strain - torsional_threshold,
        0.0,
    )
    binormal_excess = jnp.maximum(
        binormal_strain - curvature_threshold,
        0.0,
    )
    return 0.5 * jnp.mean(
        (torsional_excess**2 + binormal_excess**2) * arc_length
    )


def solve_strain_rotation(
    *,
    quadpoints: object,
    gamma: object,
    gammadash: object,
    gammadashdash: object,
    initial_parameters: object,
    rotation_order: int,
    objective_width: float,
    reporting_width: float,
    torsional_threshold: float,
    curvature_threshold: float,
    maxiter: int,
    maxfun: int,
    gtol: float,
    ftol: float,
    maxcor: int,
    maxls: int,
) -> StrainOptimizationDeviceResult:
    """Optimize rotation DOFs while retaining only fixed-size final state."""
    quadpoints_array = _as_runtime_array(quadpoints)
    gamma_array = _as_runtime_array(gamma)
    gammadash_array = _as_runtime_array(gammadash)
    gammadashdash_array = _as_runtime_array(gammadashdash)
    initial_array = _as_runtime_array(initial_parameters)
    objective = partial(
        _strain_objective,
        quadpoints_array,
        gamma_array,
        gammadash_array,
        gammadashdash_array,
        rotation_order=int(rotation_order),
        objective_width=float(objective_width),
        torsional_threshold=float(torsional_threshold),
        curvature_threshold=float(curvature_threshold),
    )
    value_and_gradient = jax.jit(jax.value_and_grad(objective))

    @jax.jit
    def state(parameters: jax.Array) -> StrainState:
        objective_value, gradient = value_and_gradient(parameters)
        torsional_strain, binormal_strain, _arc_length = _strain_values(
            quadpoints_array,
            gamma_array,
            gammadash_array,
            gammadashdash_array,
            parameters,
            rotation_order=int(rotation_order),
            width=float(reporting_width),
        )
        return StrainState(
            parameters=parameters,
            objective=objective_value,
            gradient=gradient,
            torsional_strain=torsional_strain,
            binormal_curvature_strain=binormal_strain,
            maximum_torsional_strain=jnp.max(torsional_strain),
            maximum_binormal_curvature_strain=jnp.max(binormal_strain),
        )

    initial_state = state(initial_array)
    optimizer_result = target_minimize(
        value_and_gradient,
        initial_array,
        method="lbfgs-ondevice",
        tol=float(gtol),
        maxiter=int(maxiter),
        options={
            "ftol": float(ftol),
            "maxcor": int(maxcor),
            "maxfun": int(maxfun),
            "maxls": int(maxls),
            "record_optimizer_state_trace": False,
            "lbfgs_run_mode": "stepwise",
        },
        value_and_grad=True,
        initial_value_and_grad=(
            initial_state.objective,
            initial_state.gradient,
        ),
    )
    final_parameters = jax.device_put(optimizer_result.x, initial_array.sharding)
    return StrainOptimizationDeviceResult(
        initial=initial_state,
        final=state(final_parameters),
        success=jax.device_put(optimizer_result.success, initial_array.sharding),
        status=jax.device_put(optimizer_result.status, initial_array.sharding),
        iterations=jax.device_put(optimizer_result.nit, initial_array.sharding),
        function_evaluations=jax.device_put(
            optimizer_result.nfev,
            initial_array.sharding,
        ),
        gradient_evaluations=jax.device_put(
            optimizer_result.njev,
            initial_array.sharding,
        ),
    )
