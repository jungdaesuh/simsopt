"""Device-resident RCLS and field diagnostics for wireframe examples."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax_adapters.solve.wireframe import (
    regularized_constrained_least_squares_jax,
)

from simsopt_jax.core._math_utils import (
    as_jax_int32 as _as_jax_int32,
)
from simsopt_jax.core._math_utils import (
    as_runtime_array as _as_runtime_array,
)
from simsopt_jax.core._math_utils import (
    as_runtime_value as _as_runtime_value,
)
from simsopt_jax.core.wireframe import wireframe_B

__all__ = (
    "WireframeRCLSDeviceResult",
    "WireframeRCLSState",
    "solve_wireframe_rcls",
)


@dataclass(frozen=True)
class WireframeRCLSState:
    """One complete device-resident RCLS diagnostic state."""

    currents: jax.Array
    normal_field_residual: jax.Array
    normal_objective: jax.Array
    regularization_objective: jax.Array
    total_objective: jax.Array
    constraint_residual: jax.Array
    constraint_max_abs: jax.Array


jax.tree_util.register_dataclass(
    WireframeRCLSState,
    data_fields=[
        "currents",
        "normal_field_residual",
        "normal_objective",
        "regularization_objective",
        "total_objective",
        "constraint_residual",
        "constraint_max_abs",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class WireframeRCLSDeviceResult:
    """Final-state-only RCLS solve and full source-level field diagnostics."""

    initial: WireframeRCLSState
    final: WireframeRCLSState
    magnetic_field: jax.Array
    normal_field: jax.Array
    mean_relative_normal_field: jax.Array
    maximum_current: jax.Array
    finite_currents: jax.Array


jax.tree_util.register_dataclass(
    WireframeRCLSDeviceResult,
    data_fields=[
        "initial",
        "final",
        "magnetic_field",
        "normal_field",
        "mean_relative_normal_field",
        "maximum_current",
        "finite_currents",
    ],
    meta_fields=[],
)


@jax.jit
def _expand_free_currents(
    initial_currents: jax.Array,
    free_segments: jax.Array,
    free_currents: jax.Array,
) -> jax.Array:
    return jnp.zeros_like(initial_currents).at[free_segments, :].set(free_currents)


@jax.jit
def _wireframe_rcls_diagnostics(
    response: jax.Array,
    target: jax.Array,
    regularization: jax.Array,
    constraint_matrix: jax.Array,
    constraint_target: jax.Array,
    free_segments: jax.Array,
    initial_currents: jax.Array,
    final_currents: jax.Array,
    plasma_points: jax.Array,
    plasma_unit_normal: jax.Array,
    plasma_area_weights: jax.Array,
    wireframe_nodes: jax.Array,
    wireframe_segments: jax.Array,
    wireframe_segment_signs: jax.Array,
) -> WireframeRCLSDeviceResult:
    def state(currents: jax.Array) -> WireframeRCLSState:
        residual = response @ currents - target
        constraint_residual = (
            constraint_matrix @ currents[free_segments] - constraint_target
        )
        normal_objective = 0.5 * jnp.vdot(residual, residual)
        regularization_objective = (
            0.5 * regularization**2 * jnp.vdot(currents, currents)
        )
        return WireframeRCLSState(
            currents=currents,
            normal_field_residual=residual,
            normal_objective=normal_objective,
            regularization_objective=regularization_objective,
            total_objective=normal_objective + regularization_objective,
            constraint_residual=constraint_residual,
            constraint_max_abs=jnp.max(jnp.abs(constraint_residual)),
        )

    magnetic_field = wireframe_B(
        plasma_points,
        wireframe_nodes,
        wireframe_segments,
        wireframe_segment_signs,
        final_currents.reshape(-1),
    )
    normal_field = jnp.sum(magnetic_field * plasma_unit_normal, axis=1)
    field_magnitude = jnp.linalg.norm(magnetic_field, axis=1)
    mean_relative_normal_field = jnp.sum(
        jnp.abs(normal_field / field_magnitude) * plasma_area_weights
    ) / jnp.sum(plasma_area_weights)
    return WireframeRCLSDeviceResult(
        initial=state(initial_currents),
        final=state(final_currents),
        magnetic_field=magnetic_field,
        normal_field=normal_field,
        mean_relative_normal_field=mean_relative_normal_field,
        maximum_current=jnp.max(jnp.abs(final_currents)),
        finite_currents=jnp.all(jnp.isfinite(final_currents)),
    )


@jax.jit
def _solve_wireframe_rcls_device(
    response: jax.Array,
    target: jax.Array,
    regularization: jax.Array,
    constraint_matrix: jax.Array,
    constraint_target: jax.Array,
    free_segments: jax.Array,
    initial_currents: jax.Array,
    plasma_points: jax.Array,
    plasma_unit_normal: jax.Array,
    plasma_area_weights: jax.Array,
    wireframe_nodes: jax.Array,
    wireframe_segments: jax.Array,
    wireframe_segment_signs: jax.Array,
) -> WireframeRCLSDeviceResult:
    free_currents = regularized_constrained_least_squares_jax(
        jnp.take(response, free_segments, axis=1),
        target,
        regularization,
        constraint_matrix,
        constraint_target,
    )
    final_currents = _expand_free_currents(
        initial_currents,
        free_segments,
        free_currents,
    )
    return _wireframe_rcls_diagnostics(
        response,
        target,
        regularization,
        constraint_matrix,
        constraint_target,
        free_segments,
        initial_currents,
        final_currents,
        plasma_points,
        plasma_unit_normal,
        plasma_area_weights,
        wireframe_nodes,
        wireframe_segments,
        wireframe_segment_signs,
    )


def solve_wireframe_rcls(
    *,
    wireframe,
    response: object,
    target: object,
    regularization: float,
    initial_currents: object,
    plasma_points: object,
    plasma_unit_normal: object,
    plasma_area_weights: object,
    wireframe_nodes: object,
    wireframe_segments: object,
    wireframe_segment_signs: object,
    assume_no_crossings: bool,
) -> WireframeRCLSDeviceResult:
    """Solve RCLS and evaluate all numerical source diagnostics on-device."""
    response_array = _as_runtime_array(response)
    target_array = jnp.reshape(_as_runtime_array(target), (-1, 1))
    initial_array = jnp.reshape(
        _as_runtime_array(initial_currents),
        (int(wireframe.n_segments), 1),
    )
    constraint, constraint_target = wireframe.constraint_matrices(
        assume_no_crossings=assume_no_crossings,
        remove_constrained_segments=True,
    )
    constraint_array = _as_runtime_array(constraint)
    constraint_target_array = jnp.reshape(
        _as_runtime_array(constraint_target),
        (-1, 1),
    )
    free_segments = _as_jax_int32(
        np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    )
    return _solve_wireframe_rcls_device(
        response_array,
        target_array,
        _as_runtime_value(
            regularization,
            reference=response_array,
            dtype=response_array.dtype,
        ),
        constraint_array,
        constraint_target_array,
        free_segments,
        initial_array,
        _as_runtime_array(plasma_points),
        _as_runtime_array(plasma_unit_normal),
        _as_runtime_array(plasma_area_weights),
        _as_runtime_array(wireframe_nodes),
        _as_jax_int32(wireframe_segments),
        _as_runtime_array(wireframe_segment_signs),
    )
