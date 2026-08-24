"""Restartable pure-JAX live loops for permanent-magnet workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp

from simsopt_jax.runtime.host_boundary import host_int

from ._bounded_scan import bounded_scan_until_done as _bounded_scan_until_done
from ._math_utils import as_runtime_array as _as_runtime_array
from ._math_utils import has_tracer_leaf as _has_tracer_leaf
from ._math_utils import runtime_init_array as _runtime_init_array
from ._math_utils import runtime_init_scalar as _runtime_init_scalar
from .pm_optimization import (
    GPMOArbVecBacktrackingSpec,
    GPMOArbVecSpec,
    GPMOBacktrackingSpec,
    GPMOBaselineSpec,
    GPMOMultiSpec,
    _gpmo_arbvec_contributions,
    _validate_gpmo_arbvec_backtracking_static_args,
    _validate_gpmo_arbvec_static_args,
    _validate_gpmo_backtracking_static_args,
    _validate_gpmo_multi_static_args,
    gpmo_arbvec_backtracking_step,
    gpmo_arbvec_step,
    gpmo_backtracking_step,
    gpmo_baseline_step,
    gpmo_connectivity_matrix,
    initialize_gpmo_arbvec,
    gpmo_multi_step,
)

__all__ = [
    "PMGPMOArbVecBacktrackingLiveState",
    "PMGPMOArbVecLiveState",
    "PMGPMOBacktrackingLiveState",
    "PMGPMOLiveState",
    "PMGPMOMultiLiveState",
    "pm_gpmo_arbvec_backtracking_initial_state",
    "pm_gpmo_arbvec_backtracking_live_loop_jax",
    "pm_gpmo_arbvec_initial_state",
    "pm_gpmo_arbvec_live_loop_jax",
    "pm_gpmo_arbvec_never_stop",
    "pm_gpmo_baseline_initial_state",
    "pm_gpmo_backtracking_initial_state",
    "pm_gpmo_backtracking_live_loop_jax",
    "pm_gpmo_live_loop_jax",
    "pm_gpmo_multi_initial_state",
    "pm_gpmo_multi_live_loop_jax",
    "pm_gpmo_multi_never_stop",
    "pm_gpmo_never_stop",
    "pm_gpmo_no_prune",
]


@dataclass(frozen=True)
class PMGPMOLiveState:
    """Fixed-shape baseline GPMO state for restartable device-side scans."""

    x: jax.Array
    residual: jax.Array
    available: jax.Array
    steps_taken: jax.Array
    done: jax.Array
    selected_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    residual_history: jax.Array


jax.tree_util.register_dataclass(
    PMGPMOLiveState,
    data_fields=[
        "x",
        "residual",
        "available",
        "steps_taken",
        "done",
        "selected_dipoles",
        "selected_components",
        "selected_signs",
        "residual_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class PMGPMOMultiLiveState:
    """Fixed-shape multi-neighbour GPMO state for restartable scans."""

    x: jax.Array
    residual: jax.Array
    available: jax.Array
    steps_taken: jax.Array
    done: jax.Array
    selected_seed_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    residual_history: jax.Array
    selected_groups: jax.Array


jax.tree_util.register_dataclass(
    PMGPMOMultiLiveState,
    data_fields=[
        "x",
        "residual",
        "available",
        "steps_taken",
        "done",
        "selected_seed_dipoles",
        "selected_components",
        "selected_signs",
        "residual_history",
        "selected_groups",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class PMGPMOArbVecLiveState:
    """Fixed-shape ArbVec GPMO state for restartable scans."""

    x: jax.Array
    residual: jax.Array
    available: jax.Array
    steps_taken: jax.Array
    done: jax.Array
    selected_dipoles: jax.Array
    selected_vector_indices: jax.Array
    selected_signs: jax.Array
    residual_history: jax.Array


jax.tree_util.register_dataclass(
    PMGPMOArbVecLiveState,
    data_fields=[
        "x",
        "residual",
        "available",
        "steps_taken",
        "done",
        "selected_dipoles",
        "selected_vector_indices",
        "selected_signs",
        "residual_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class PMGPMOBacktrackingLiveState:
    """Fixed-shape backtracking GPMO state for restartable scans."""

    x: jax.Array
    residual: jax.Array
    available: jax.Array
    current_signs: jax.Array
    current_components: jax.Array
    steps_taken: jax.Array
    done: jax.Array
    selected_dipoles: jax.Array
    selected_components: jax.Array
    selected_signs: jax.Array
    residual_history: jax.Array
    x_history: jax.Array
    num_nonzeros_history: jax.Array
    removed_pair_count_history: jax.Array
    done_history: jax.Array


jax.tree_util.register_dataclass(
    PMGPMOBacktrackingLiveState,
    data_fields=[
        "x",
        "residual",
        "available",
        "current_signs",
        "current_components",
        "steps_taken",
        "done",
        "selected_dipoles",
        "selected_components",
        "selected_signs",
        "residual_history",
        "x_history",
        "num_nonzeros_history",
        "removed_pair_count_history",
        "done_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class PMGPMOArbVecBacktrackingLiveState:
    """Fixed-shape ArbVec-backtracking GPMO state for restartable scans."""

    x: jax.Array
    residual: jax.Array
    available: jax.Array
    current_vector_indices: jax.Array
    current_signs: jax.Array
    steps_taken: jax.Array
    done: jax.Array
    selected_dipoles: jax.Array
    selected_vector_indices: jax.Array
    selected_signs: jax.Array
    residual_history: jax.Array
    x_history: jax.Array
    num_nonzeros_history: jax.Array
    removed_pair_count_history: jax.Array
    done_history: jax.Array
    initial_x: jax.Array
    initial_residual: jax.Array
    initial_num_nonzero: jax.Array


jax.tree_util.register_dataclass(
    PMGPMOArbVecBacktrackingLiveState,
    data_fields=[
        "x",
        "residual",
        "available",
        "current_vector_indices",
        "current_signs",
        "steps_taken",
        "done",
        "selected_dipoles",
        "selected_vector_indices",
        "selected_signs",
        "residual_history",
        "x_history",
        "num_nonzeros_history",
        "removed_pair_count_history",
        "done_history",
        "initial_x",
        "initial_residual",
        "initial_num_nonzero",
    ],
    meta_fields=[],
)

PMPruneRule = Callable[[PMGPMOLiveState], tuple[PMGPMOLiveState, jax.Array]]
PMStopRule = Callable[[PMGPMOLiveState], jax.Array]
PMMultiStopRule = Callable[[PMGPMOMultiLiveState], jax.Array]
PMArbVecStopRule = Callable[[PMGPMOArbVecLiveState], jax.Array]


def pm_gpmo_no_prune(state: PMGPMOLiveState) -> tuple[PMGPMOLiveState, jax.Array]:
    """Return the state unchanged with no unavailable-candidate additions."""

    return state, jnp.zeros_like(state.available)


def pm_gpmo_never_stop(state: PMGPMOLiveState) -> jax.Array:
    """Keep scanning until the static ``max_steps`` budget is consumed."""

    return _runtime_init_scalar(False, jnp.bool_)


def pm_gpmo_multi_never_stop(state: PMGPMOMultiLiveState) -> jax.Array:
    """Keep scanning until the static ``max_steps`` budget is consumed."""

    return _runtime_init_scalar(False, jnp.bool_)


def pm_gpmo_arbvec_never_stop(state: PMGPMOArbVecLiveState) -> jax.Array:
    """Keep scanning until the static ``max_steps`` budget is consumed."""

    return _runtime_init_scalar(False, jnp.bool_)


def pm_gpmo_baseline_initial_state(
    A_scaled: jax.Array,
    b: jax.Array,
    *,
    ndipoles: int,
    history_capacity: int,
) -> PMGPMOLiveState:
    """Create an empty baseline GPMO live state with fixed history capacity."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    return PMGPMOLiveState(
        x=_runtime_init_array((ndipoles, 3), 0, A_arr.dtype),
        residual=-b_arr,
        available=_runtime_init_array((ndipoles, 3), True, jnp.bool_),
        steps_taken=_runtime_init_scalar(0, jnp.int32),
        done=_runtime_init_scalar(False, jnp.bool_),
        selected_dipoles=_runtime_init_array((history_capacity,), 0, jnp.int64),
        selected_components=_runtime_init_array((history_capacity,), 0, jnp.int64),
        selected_signs=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        residual_history=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
    )


def pm_gpmo_multi_initial_state(
    A_scaled: jax.Array,
    b: jax.Array,
    spec: GPMOMultiSpec,
    *,
    history_capacity: int,
) -> PMGPMOMultiLiveState:
    """Create an empty multi-neighbour GPMO live state."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    ndipoles = int(spec.m_maxima.shape[0])
    return PMGPMOMultiLiveState(
        x=_runtime_init_array((ndipoles, 3), 0, A_arr.dtype),
        residual=-b_arr,
        available=_runtime_init_array((ndipoles, 3), True, jnp.bool_),
        steps_taken=_runtime_init_scalar(0, jnp.int32),
        done=_runtime_init_scalar(False, jnp.bool_),
        selected_seed_dipoles=_runtime_init_array((history_capacity,), 0, jnp.int64),
        selected_components=_runtime_init_array((history_capacity,), 0, jnp.int64),
        selected_signs=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        residual_history=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        selected_groups=_runtime_init_array(
            (history_capacity, spec.Nadjacent), 0, jnp.int64
        ),
    )


def pm_gpmo_arbvec_initial_state(
    A_scaled: jax.Array,
    b: jax.Array,
    spec: GPMOArbVecSpec,
    *,
    history_capacity: int,
) -> PMGPMOArbVecLiveState:
    """Create an empty ArbVec GPMO live state."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    ndipoles = int(spec.m_maxima.shape[0])
    return PMGPMOArbVecLiveState(
        x=_runtime_init_array((ndipoles, 3), 0, A_arr.dtype),
        residual=-b_arr,
        available=_runtime_init_array((ndipoles,), True, jnp.bool_),
        steps_taken=_runtime_init_scalar(0, jnp.int32),
        done=_runtime_init_scalar(False, jnp.bool_),
        selected_dipoles=_runtime_init_array((history_capacity,), 0, jnp.int64),
        selected_vector_indices=_runtime_init_array((history_capacity,), 0, jnp.int64),
        selected_signs=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        residual_history=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
    )


def pm_gpmo_backtracking_initial_state(
    A_scaled: jax.Array,
    b: jax.Array,
    spec: GPMOBacktrackingSpec,
    *,
    history_capacity: int,
) -> PMGPMOBacktrackingLiveState:
    """Create an empty backtracking GPMO live state."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    ndipoles = int(spec.m_maxima.shape[0])
    return PMGPMOBacktrackingLiveState(
        x=_runtime_init_array((ndipoles, 3), 0, A_arr.dtype),
        residual=-b_arr,
        available=_runtime_init_array((ndipoles, 3), True, jnp.bool_),
        current_signs=_runtime_init_array((ndipoles,), 0, A_arr.dtype),
        current_components=_runtime_init_array((ndipoles,), 0, jnp.int64),
        steps_taken=_runtime_init_scalar(0, jnp.int32),
        done=_runtime_init_scalar(False, jnp.bool_),
        selected_dipoles=_runtime_init_array((history_capacity,), -1, jnp.int64),
        selected_components=_runtime_init_array((history_capacity,), -1, jnp.int64),
        selected_signs=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        residual_history=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        x_history=_runtime_init_array((history_capacity, ndipoles, 3), 0, A_arr.dtype),
        num_nonzeros_history=_runtime_init_array((history_capacity,), 0, jnp.int64),
        removed_pair_count_history=_runtime_init_array(
            (history_capacity,), 0, jnp.int64
        ),
        done_history=_runtime_init_array((history_capacity,), False, jnp.bool_),
    )


def pm_gpmo_arbvec_backtracking_initial_state(
    A_scaled: jax.Array,
    b: jax.Array,
    spec: GPMOArbVecBacktrackingSpec,
    *,
    history_capacity: int,
    x_init: jax.Array | None = None,
) -> PMGPMOArbVecBacktrackingLiveState:
    """Create an ArbVec-backtracking GPMO live state."""

    A_arr = _as_runtime_array(A_scaled)
    b_arr = _as_runtime_array(b)
    pol_vectors = _as_runtime_array(spec.pol_vectors)
    ndipoles = int(spec.m_maxima.shape[0])
    _validate_gpmo_arbvec_backtracking_static_args(
        history_capacity,
        ndipoles,
        pol_vectors,
        spec.Nadjacent,
        spec.backtracking,
        spec.max_nMagnets,
        spec.thresh_angle,
    )
    if x_init is None:
        x_init_arr = _runtime_init_array((ndipoles, 3), 0, A_arr.dtype)
    else:
        x_init_arr = _as_runtime_array(x_init)
    (
        x,
        residual,
        available,
        current_vector_indices,
        current_signs,
        initial_num_nonzero,
    ) = initialize_gpmo_arbvec(x_init_arr, pol_vectors, A_arr, b_arr)
    initial_stop = (initial_num_nonzero >= ndipoles) | (
        initial_num_nonzero >= spec.max_nMagnets
    )
    return PMGPMOArbVecBacktrackingLiveState(
        x=x,
        residual=residual,
        available=available,
        current_vector_indices=current_vector_indices,
        current_signs=current_signs,
        steps_taken=_runtime_init_scalar(0, jnp.int32),
        done=initial_stop,
        selected_dipoles=_runtime_init_array((history_capacity,), -1, jnp.int64),
        selected_vector_indices=_runtime_init_array((history_capacity,), -1, jnp.int64),
        selected_signs=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        residual_history=_runtime_init_array((history_capacity,), 0, A_arr.dtype),
        x_history=_runtime_init_array((history_capacity, ndipoles, 3), 0, A_arr.dtype),
        num_nonzeros_history=_runtime_init_array((history_capacity,), 0, jnp.int64),
        removed_pair_count_history=_runtime_init_array(
            (history_capacity,), 0, jnp.int64
        ),
        done_history=_runtime_init_array((history_capacity,), False, jnp.bool_),
        initial_x=x,
        initial_residual=residual,
        initial_num_nonzero=initial_num_nonzero,
    )


def _concrete_steps_taken(value: jax.Array, function_name: str) -> int:
    if _has_tracer_leaf(value):
        raise ValueError(
            f"state.steps_taken must be concrete when tracing {function_name} "
            "so restart capacity is checked before scan."
        )
    return host_int(value)


def _validate_steps_taken_bounds(
    steps_taken: int,
    max_steps: int,
    history_capacity: int,
    *,
    history_name: str,
) -> int:
    if steps_taken < 0:
        raise ValueError(f"state.steps_taken must be nonnegative; got {steps_taken}.")
    final_step = steps_taken + max_steps
    if final_step > history_capacity:
        raise ValueError(
            f"state.steps_taken + max_steps must fit the {history_name} history "
            f"capacity; got {steps_taken} + {max_steps} > {history_capacity}."
        )
    return final_step


def _validate_baseline_live_loop_capacity(
    state: PMGPMOLiveState,
    spec: GPMOBaselineSpec,
    max_steps: int,
) -> None:
    history_capacity = int(state.selected_dipoles.shape[0])
    history_shapes = (
        state.selected_components.shape,
        state.selected_signs.shape,
        state.residual_history.shape,
    )
    expected_shape = (history_capacity,)
    if any(shape != expected_shape for shape in history_shapes):
        raise ValueError(
            "baseline GPMO history arrays must share one capacity; got "
            f"selected_dipoles={state.selected_dipoles.shape}, "
            f"selected_components={state.selected_components.shape}, "
            f"selected_signs={state.selected_signs.shape}, "
            f"residual_history={state.residual_history.shape}."
        )
    ndipoles = int(spec.m_maxima.shape[0])
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative; got {max_steps}.")
    if max_steps > history_capacity:
        raise ValueError(
            "max_steps must fit the baseline GPMO history capacity; "
            f"got max_steps={max_steps}, history_capacity={history_capacity}."
        )
    if max_steps > ndipoles:
        raise ValueError(
            "max_steps must not exceed the baseline GPMO dipole capacity; "
            f"got max_steps={max_steps}, ndipoles={ndipoles}."
        )
    steps_taken = _concrete_steps_taken(state.steps_taken, "pm_gpmo_live_loop_jax")
    final_step = _validate_steps_taken_bounds(
        steps_taken,
        max_steps,
        history_capacity,
        history_name="baseline GPMO",
    )
    if final_step > ndipoles:
        raise ValueError(
            "state.steps_taken + max_steps must not exceed the baseline GPMO "
            f"dipole capacity; got {steps_taken} + {max_steps} > {ndipoles}."
        )


def _validate_multi_live_loop_capacity(
    state: PMGPMOMultiLiveState,
    spec: GPMOMultiSpec,
    max_steps: int,
) -> None:
    history_capacity = int(state.selected_seed_dipoles.shape[0])
    expected_shape = (history_capacity,)
    history_shapes = (
        state.selected_components.shape,
        state.selected_signs.shape,
        state.residual_history.shape,
    )
    if any(shape != expected_shape for shape in history_shapes):
        raise ValueError(
            "multi-neighbour GPMO history arrays must share one capacity; got "
            f"selected_seed_dipoles={state.selected_seed_dipoles.shape}, "
            f"selected_components={state.selected_components.shape}, "
            f"selected_signs={state.selected_signs.shape}, "
            f"residual_history={state.residual_history.shape}."
        )
    if state.selected_groups.shape != (history_capacity, spec.Nadjacent):
        raise ValueError(
            "selected_groups must have shape (history_capacity, Nadjacent); got "
            f"{state.selected_groups.shape}, expected "
            f"({history_capacity}, {spec.Nadjacent})."
        )
    ndipoles = int(spec.m_maxima.shape[0])
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative; got {max_steps}.")
    if max_steps > history_capacity:
        raise ValueError(
            "max_steps must fit the multi-neighbour GPMO history capacity; "
            f"got max_steps={max_steps}, history_capacity={history_capacity}."
        )
    steps_taken = _concrete_steps_taken(
        state.steps_taken,
        "pm_gpmo_multi_live_loop_jax",
    )
    final_step = _validate_steps_taken_bounds(
        steps_taken,
        max_steps,
        history_capacity,
        history_name="multi-neighbour GPMO",
    )
    _validate_gpmo_multi_static_args(
        final_step,
        spec.single_direction,
        ndipoles,
        spec.Nadjacent,
    )


def _validate_arbvec_live_loop_capacity(
    state: PMGPMOArbVecLiveState,
    spec: GPMOArbVecSpec,
    max_steps: int,
) -> None:
    history_capacity = int(state.selected_dipoles.shape[0])
    expected_shape = (history_capacity,)
    history_shapes = (
        state.selected_vector_indices.shape,
        state.selected_signs.shape,
        state.residual_history.shape,
    )
    if any(shape != expected_shape for shape in history_shapes):
        raise ValueError(
            "ArbVec GPMO history arrays must share one capacity; got "
            f"selected_dipoles={state.selected_dipoles.shape}, "
            f"selected_vector_indices={state.selected_vector_indices.shape}, "
            f"selected_signs={state.selected_signs.shape}, "
            f"residual_history={state.residual_history.shape}."
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative; got {max_steps}.")
    if max_steps > history_capacity:
        raise ValueError(
            "max_steps must fit the ArbVec GPMO history capacity; "
            f"got max_steps={max_steps}, history_capacity={history_capacity}."
        )
    ndipoles = int(spec.m_maxima.shape[0])
    steps_taken = _concrete_steps_taken(
        state.steps_taken,
        "pm_gpmo_arbvec_live_loop_jax",
    )
    final_step = _validate_steps_taken_bounds(
        steps_taken,
        max_steps,
        history_capacity,
        history_name="ArbVec GPMO",
    )
    _validate_gpmo_arbvec_static_args(
        final_step,
        ndipoles,
        _as_runtime_array(spec.pol_vectors),
    )


def _validate_backtracking_live_loop_capacity(
    state: PMGPMOBacktrackingLiveState,
    spec: GPMOBacktrackingSpec,
    max_steps: int,
) -> None:
    history_capacity = int(state.selected_dipoles.shape[0])
    expected_shape = (history_capacity,)
    history_shapes = (
        state.selected_components.shape,
        state.selected_signs.shape,
        state.residual_history.shape,
        state.num_nonzeros_history.shape,
        state.removed_pair_count_history.shape,
        state.done_history.shape,
    )
    if any(shape != expected_shape for shape in history_shapes):
        raise ValueError(
            "backtracking GPMO history arrays must share one capacity; got "
            f"selected_dipoles={state.selected_dipoles.shape}, "
            f"selected_components={state.selected_components.shape}, "
            f"selected_signs={state.selected_signs.shape}, "
            f"residual_history={state.residual_history.shape}, "
            f"num_nonzeros_history={state.num_nonzeros_history.shape}, "
            f"removed_pair_count_history={state.removed_pair_count_history.shape}, "
            f"done_history={state.done_history.shape}."
        )
    expected_x_history_shape = (history_capacity, int(spec.m_maxima.shape[0]), 3)
    if state.x_history.shape != expected_x_history_shape:
        raise ValueError(
            "x_history must have shape (history_capacity, ndipoles, 3); got "
            f"{state.x_history.shape}, expected {expected_x_history_shape}."
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative; got {max_steps}.")
    if max_steps > history_capacity:
        raise ValueError(
            "max_steps must fit the backtracking GPMO history capacity; "
            f"got max_steps={max_steps}, history_capacity={history_capacity}."
        )
    ndipoles = int(spec.m_maxima.shape[0])
    steps_taken = _concrete_steps_taken(
        state.steps_taken,
        "pm_gpmo_backtracking_live_loop_jax",
    )
    final_step = _validate_steps_taken_bounds(
        steps_taken,
        max_steps,
        history_capacity,
        history_name="backtracking GPMO",
    )
    _validate_gpmo_backtracking_static_args(
        final_step,
        spec.single_direction,
        ndipoles,
        spec.Nadjacent,
        spec.backtracking,
        spec.max_nMagnets,
    )


def _validate_arbvec_backtracking_live_loop_capacity(
    state: PMGPMOArbVecBacktrackingLiveState,
    spec: GPMOArbVecBacktrackingSpec,
    max_steps: int,
) -> None:
    history_capacity = int(state.selected_dipoles.shape[0])
    expected_shape = (history_capacity,)
    history_shapes = (
        state.selected_vector_indices.shape,
        state.selected_signs.shape,
        state.residual_history.shape,
        state.num_nonzeros_history.shape,
        state.removed_pair_count_history.shape,
        state.done_history.shape,
    )
    if any(shape != expected_shape for shape in history_shapes):
        raise ValueError(
            "ArbVec-backtracking GPMO history arrays must share one capacity; got "
            f"selected_dipoles={state.selected_dipoles.shape}, "
            f"selected_vector_indices={state.selected_vector_indices.shape}, "
            f"selected_signs={state.selected_signs.shape}, "
            f"residual_history={state.residual_history.shape}, "
            f"num_nonzeros_history={state.num_nonzeros_history.shape}, "
            f"removed_pair_count_history={state.removed_pair_count_history.shape}, "
            f"done_history={state.done_history.shape}."
        )
    ndipoles = int(spec.m_maxima.shape[0])
    expected_x_history_shape = (history_capacity, ndipoles, 3)
    if state.x_history.shape != expected_x_history_shape:
        raise ValueError(
            "x_history must have shape (history_capacity, ndipoles, 3); got "
            f"{state.x_history.shape}, expected {expected_x_history_shape}."
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative; got {max_steps}.")
    if max_steps > history_capacity:
        raise ValueError(
            "max_steps must fit the ArbVec-backtracking GPMO history capacity; "
            f"got max_steps={max_steps}, history_capacity={history_capacity}."
        )
    steps_taken = _concrete_steps_taken(
        state.steps_taken,
        "pm_gpmo_arbvec_backtracking_live_loop_jax",
    )
    final_step = _validate_steps_taken_bounds(
        steps_taken,
        max_steps,
        history_capacity,
        history_name="ArbVec-backtracking GPMO",
    )
    _validate_gpmo_arbvec_backtracking_static_args(
        final_step,
        ndipoles,
        _as_runtime_array(spec.pol_vectors),
        spec.Nadjacent,
        spec.backtracking,
        spec.max_nMagnets,
        spec.thresh_angle,
    )


def pm_gpmo_live_loop_jax(
    state: PMGPMOLiveState,
    spec: GPMOBaselineSpec,
    A_scaled: jax.Array,
    *,
    max_steps: int,
    prune_rule: PMPruneRule = pm_gpmo_no_prune,
    stop_rule: PMStopRule = pm_gpmo_never_stop,
) -> PMGPMOLiveState:
    """Advance baseline GPMO state through a fixed-length device-side scan."""

    _validate_baseline_live_loop_capacity(state, spec, int(max_steps))
    A_arr = _as_runtime_array(A_scaled)
    scan_spec = replace(
        spec,
        m_maxima=_as_runtime_array(spec.m_maxima),
        reg_l2=_as_runtime_array(spec.reg_l2),
    )

    def _active_step(current: PMGPMOLiveState) -> PMGPMOLiveState:
        pruned, prune_mask = prune_rule(current)
        available = pruned.available & ~prune_mask
        (x, residual, available), trace = gpmo_baseline_step(
            scan_spec,
            (pruned.x, pruned.residual, available),
            A_arr,
        )
        dipole, component, sign, residual_sq = trace
        history_index = pruned.steps_taken
        next_state = PMGPMOLiveState(
            x=x,
            residual=residual,
            available=available,
            steps_taken=history_index + _runtime_init_scalar(1, history_index.dtype),
            done=_runtime_init_scalar(False, jnp.bool_),
            selected_dipoles=pruned.selected_dipoles.at[history_index].set(dipole),
            selected_components=pruned.selected_components.at[history_index].set(
                component
            ),
            selected_signs=pruned.selected_signs.at[history_index].set(sign),
            residual_history=pruned.residual_history.at[history_index].set(residual_sq),
        )
        return replace(next_state, done=stop_rule(next_state))

    def _step(current: PMGPMOLiveState, _iteration: jax.Array) -> PMGPMOLiveState:
        return _active_step(current)

    final_state = _bounded_scan_until_done(
        state,
        max_steps=int(max_steps),
        is_done=lambda current: current.done,
        step=_step,
    )
    return final_state


def pm_gpmo_arbvec_live_loop_jax(
    state: PMGPMOArbVecLiveState,
    spec: GPMOArbVecSpec,
    A_scaled: jax.Array,
    *,
    max_steps: int,
    stop_rule: PMArbVecStopRule = pm_gpmo_arbvec_never_stop,
) -> PMGPMOArbVecLiveState:
    """Advance ArbVec GPMO state through a fixed-length scan."""

    _validate_arbvec_live_loop_capacity(state, spec, int(max_steps))
    A_arr = _as_runtime_array(A_scaled)
    scan_spec = replace(
        spec,
        m_maxima=_as_runtime_array(spec.m_maxima),
        reg_l2=_as_runtime_array(spec.reg_l2),
        pol_vectors=_as_runtime_array(spec.pol_vectors),
    )
    contributions = _gpmo_arbvec_contributions(A_arr, scan_spec.pol_vectors)

    def _active_step(current: PMGPMOArbVecLiveState) -> PMGPMOArbVecLiveState:
        next_core, trace = gpmo_arbvec_step(
            scan_spec,
            (current.x, current.residual, current.available),
            A_arr,
            contributions,
        )
        dipole, vector_index, sign, residual_sq = trace
        x, residual, available = next_core
        history_index = current.steps_taken
        next_state = PMGPMOArbVecLiveState(
            x=x,
            residual=residual,
            available=available,
            steps_taken=history_index + _runtime_init_scalar(1, history_index.dtype),
            done=_runtime_init_scalar(False, jnp.bool_),
            selected_dipoles=current.selected_dipoles.at[history_index].set(dipole),
            selected_vector_indices=current.selected_vector_indices.at[
                history_index
            ].set(vector_index),
            selected_signs=current.selected_signs.at[history_index].set(sign),
            residual_history=current.residual_history.at[history_index].set(
                residual_sq
            ),
        )
        return replace(next_state, done=stop_rule(next_state))

    def _scan_body(current: PMGPMOArbVecLiveState, _iteration: jax.Array):
        next_state = jax.lax.cond(
            current.done,
            lambda done_state: done_state,
            _active_step,
            current,
        )
        return next_state, None

    final_state, _ = jax.lax.scan(
        _scan_body,
        state,
        jnp.arange(int(max_steps), dtype=jnp.int32),
    )
    return final_state


def pm_gpmo_backtracking_live_loop_jax(
    state: PMGPMOBacktrackingLiveState,
    spec: GPMOBacktrackingSpec,
    A_scaled: jax.Array,
    *,
    max_steps: int,
) -> PMGPMOBacktrackingLiveState:
    """Advance backtracking GPMO state through a fixed-length scan."""

    _validate_backtracking_live_loop_capacity(state, spec, int(max_steps))
    A_arr = _as_runtime_array(A_scaled)
    scan_spec = replace(
        spec,
        m_maxima=_as_runtime_array(spec.m_maxima),
        reg_l2=_as_runtime_array(spec.reg_l2),
        dipole_grid_xyz=_as_runtime_array(spec.dipole_grid_xyz),
    )
    connectivity = gpmo_connectivity_matrix(scan_spec.dipole_grid_xyz)
    history_capacity = int(state.selected_dipoles.shape[0])

    def _active_step(
        current: PMGPMOBacktrackingLiveState,
    ) -> PMGPMOBacktrackingLiveState:
        next_core, trace = gpmo_backtracking_step(
            scan_spec,
            (
                current.x,
                current.residual,
                current.available,
                current.current_signs,
                current.current_components,
                current.selected_dipoles,
                current.selected_components,
                current.selected_signs,
                current.done,
            ),
            A_arr,
            connectivity,
            current.steps_taken,
            K=history_capacity,
        )
        (
            dipole,
            component,
            sign,
            residual_sq,
            x_snapshot,
            num_nonzeros,
            removed_pair_count,
            done_snapshot,
        ) = trace
        (
            x,
            residual,
            available,
            current_signs,
            current_components,
            selected_dipoles,
            selected_components,
            selected_signs,
            done,
        ) = next_core
        history_index = current.steps_taken
        return PMGPMOBacktrackingLiveState(
            x=x,
            residual=residual,
            available=available,
            current_signs=current_signs,
            current_components=current_components,
            steps_taken=history_index + _runtime_init_scalar(1, history_index.dtype),
            done=done,
            selected_dipoles=selected_dipoles,
            selected_components=selected_components,
            selected_signs=selected_signs,
            residual_history=current.residual_history.at[history_index].set(
                residual_sq
            ),
            x_history=current.x_history.at[history_index].set(x_snapshot),
            num_nonzeros_history=current.num_nonzeros_history.at[history_index].set(
                num_nonzeros
            ),
            removed_pair_count_history=current.removed_pair_count_history.at[
                history_index
            ].set(removed_pair_count),
            done_history=current.done_history.at[history_index].set(done_snapshot),
        )

    def _scan_body(current: PMGPMOBacktrackingLiveState, _iteration: jax.Array):
        return _active_step(current), None

    final_state, _ = jax.lax.scan(
        _scan_body,
        state,
        jnp.arange(int(max_steps), dtype=jnp.int32),
    )
    return final_state


def pm_gpmo_arbvec_backtracking_live_loop_jax(
    state: PMGPMOArbVecBacktrackingLiveState,
    spec: GPMOArbVecBacktrackingSpec,
    A_scaled: jax.Array,
    *,
    max_steps: int,
) -> PMGPMOArbVecBacktrackingLiveState:
    """Advance ArbVec-backtracking GPMO state through a fixed-length scan."""

    _validate_arbvec_backtracking_live_loop_capacity(state, spec, int(max_steps))
    A_arr = _as_runtime_array(A_scaled)
    scan_spec = replace(
        spec,
        m_maxima=_as_runtime_array(spec.m_maxima),
        reg_l2=_as_runtime_array(spec.reg_l2),
        dipole_grid_xyz=_as_runtime_array(spec.dipole_grid_xyz),
        pol_vectors=_as_runtime_array(spec.pol_vectors),
    )
    connectivity = gpmo_connectivity_matrix(scan_spec.dipole_grid_xyz)
    # Host libm, matching gpmo_arbvec_backtracking_solve: one cosine
    # implementation owns both the exact-pi gate (math.cos inside the step)
    # and this FP-path threshold, in every caller.
    cos_thresh_angle = _runtime_init_scalar(math.cos(spec.thresh_angle), A_arr.dtype)
    contributions = _gpmo_arbvec_contributions(A_arr, scan_spec.pol_vectors)

    def _active_step(
        current: PMGPMOArbVecBacktrackingLiveState,
    ) -> PMGPMOArbVecBacktrackingLiveState:
        next_core, trace = gpmo_arbvec_backtracking_step(
            scan_spec,
            (
                current.x,
                current.residual,
                current.available,
                current.current_vector_indices,
                current.current_signs,
                current.selected_dipoles,
                current.selected_vector_indices,
                current.selected_signs,
                current.done,
            ),
            A_arr,
            connectivity,
            cos_thresh_angle,
            current.steps_taken,
            contributions,
        )
        (
            dipole,
            vector_index,
            sign,
            residual_sq,
            x_snapshot,
            num_nonzeros,
            removed_pair_count,
            done_snapshot,
        ) = trace
        (
            x,
            residual,
            available,
            current_vector_indices,
            current_signs,
            selected_dipoles,
            selected_vector_indices,
            selected_signs,
            done,
        ) = next_core
        history_index = current.steps_taken
        return PMGPMOArbVecBacktrackingLiveState(
            x=x,
            residual=residual,
            available=available,
            current_vector_indices=current_vector_indices,
            current_signs=current_signs,
            steps_taken=history_index + _runtime_init_scalar(1, history_index.dtype),
            done=done,
            selected_dipoles=selected_dipoles,
            selected_vector_indices=selected_vector_indices,
            selected_signs=selected_signs,
            residual_history=current.residual_history.at[history_index].set(
                residual_sq
            ),
            x_history=current.x_history.at[history_index].set(x_snapshot),
            num_nonzeros_history=current.num_nonzeros_history.at[history_index].set(
                num_nonzeros
            ),
            removed_pair_count_history=current.removed_pair_count_history.at[
                history_index
            ].set(removed_pair_count),
            done_history=current.done_history.at[history_index].set(done_snapshot),
            initial_x=current.initial_x,
            initial_residual=current.initial_residual,
            initial_num_nonzero=current.initial_num_nonzero,
        )

    def _scan_body(
        current: PMGPMOArbVecBacktrackingLiveState,
        _iteration: jax.Array,
    ):
        return _active_step(current), None

    final_state, _ = jax.lax.scan(
        _scan_body,
        state,
        jnp.arange(int(max_steps), dtype=jnp.int32),
    )
    return final_state


def pm_gpmo_multi_live_loop_jax(
    state: PMGPMOMultiLiveState,
    spec: GPMOMultiSpec,
    A_scaled: jax.Array,
    *,
    max_steps: int,
    stop_rule: PMMultiStopRule = pm_gpmo_multi_never_stop,
) -> PMGPMOMultiLiveState:
    """Advance multi-neighbour GPMO state through a fixed-length scan."""

    _validate_multi_live_loop_capacity(state, spec, int(max_steps))
    A_arr = _as_runtime_array(A_scaled)
    scan_spec = replace(
        spec,
        m_maxima=_as_runtime_array(spec.m_maxima),
        reg_l2=_as_runtime_array(spec.reg_l2),
        dipole_grid_xyz=_as_runtime_array(spec.dipole_grid_xyz),
    )
    connectivity = gpmo_connectivity_matrix(scan_spec.dipole_grid_xyz)

    def _active_step(current: PMGPMOMultiLiveState) -> PMGPMOMultiLiveState:
        next_core, trace = gpmo_multi_step(
            scan_spec,
            (current.x, current.residual, current.available),
            A_arr,
            connectivity,
        )
        seed_dipole, component, sign, residual_sq, selected_group = trace
        x, residual, available = next_core
        history_index = current.steps_taken
        next_state = PMGPMOMultiLiveState(
            x=x,
            residual=residual,
            available=available,
            steps_taken=history_index + _runtime_init_scalar(1, history_index.dtype),
            done=_runtime_init_scalar(False, jnp.bool_),
            selected_seed_dipoles=current.selected_seed_dipoles.at[history_index].set(
                seed_dipole
            ),
            selected_components=current.selected_components.at[history_index].set(
                component
            ),
            selected_signs=current.selected_signs.at[history_index].set(sign),
            residual_history=current.residual_history.at[history_index].set(
                residual_sq
            ),
            selected_groups=current.selected_groups.at[history_index].set(
                selected_group
            ),
        )
        return replace(next_state, done=stop_rule(next_state))

    def _scan_body(current: PMGPMOMultiLiveState, _iteration: jax.Array):
        next_state = jax.lax.cond(
            current.done,
            lambda done_state: done_state,
            _active_step,
            current,
        )
        return next_state, None

    final_state, _ = jax.lax.scan(
        _scan_body,
        state,
        jnp.arange(int(max_steps), dtype=jnp.int32),
    )
    return final_state
