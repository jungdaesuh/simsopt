"""Restartable pure-JAX live loops for wireframe GSCO workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import io_callback

from ._bounded_scan import bounded_scan_until_done as _bounded_scan_until_done
from ._math_utils import as_runtime_array as _as_runtime_array
from ._math_utils import as_jax_int32 as _as_jax_int32
from ._math_utils import has_tracer_leaf as _has_tracer_leaf
from ._math_utils import runtime_init_array as _runtime_init_array
from ._math_utils import runtime_init_scalar as _runtime_init_scalar

__all__ = [
    "WireframeGSCOResult",
    "WireframeGSCOLiveParams",
    "WireframeGSCOLiveState",
    "WireframeGSCOMultistepResult",
    "WireframeGSCOMultistepState",
    "find_wireframe_coil_sizes_jax",
    "greedy_stellarator_coil_optimization_jax",
    "gsco_live_loop_jax",
    "wireframe_gsco_initial_state",
    "wireframe_gsco_multistep_loop_jax",
    "wireframe_gsco_never_stop",
]


@dataclass(frozen=True)
class WireframeGSCOResult:
    """Immutable fixed-shape result from the JAX GSCO kernel."""

    x: jax.Array
    loop_count: jax.Array
    history_length: jax.Array
    iter_history: jax.Array
    curr_history: jax.Array
    loop_history: jax.Array
    f_B_history: jax.Array
    f_S_history: jax.Array
    f_history: jax.Array


jax.tree_util.register_dataclass(
    WireframeGSCOResult,
    data_fields=[
        "x",
        "loop_count",
        "history_length",
        "iter_history",
        "curr_history",
        "loop_history",
        "f_B_history",
        "f_S_history",
        "f_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class WireframeGSCOLiveParams:
    """Fixed-shape GSCO arrays and static loop rules."""

    A: jax.Array
    loops: jax.Array
    free_loops: jax.Array
    segments: jax.Array
    connections: jax.Array
    default_current: jax.Array
    max_current: jax.Array
    lambda_s: jax.Array
    tol: jax.Array
    max_loop_count: int
    no_crossing: bool
    no_new_coils: bool
    match_current: bool
    loop_columns: jax.Array | None = None


jax.tree_util.register_dataclass(
    WireframeGSCOLiveParams,
    data_fields=[
        "A",
        "loops",
        "free_loops",
        "segments",
        "connections",
        "default_current",
        "max_current",
        "lambda_s",
        "tol",
        "loop_columns",
    ],
    meta_fields=["max_loop_count", "no_crossing", "no_new_coils", "match_current"],
)


@dataclass(frozen=True)
class WireframeGSCOLiveState:
    """Fixed-shape restart state for one GSCO live loop."""

    x: jax.Array
    loop_count: jax.Array
    residual: jax.Array
    two_f_b: jax.Array
    two_f_s: jax.Array
    two_f: jax.Array
    opt_ind_prev: jax.Array
    history_length: jax.Array
    done: jax.Array
    iter_history: jax.Array
    curr_history: jax.Array
    loop_history: jax.Array
    f_B_history: jax.Array
    f_S_history: jax.Array
    f_history: jax.Array


jax.tree_util.register_dataclass(
    WireframeGSCOLiveState,
    data_fields=[
        "x",
        "loop_count",
        "residual",
        "two_f_b",
        "two_f_s",
        "two_f",
        "opt_ind_prev",
        "history_length",
        "done",
        "iter_history",
        "curr_history",
        "loop_history",
        "f_B_history",
        "f_S_history",
        "f_history",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class WireframeGSCOMultistepState:
    """Fixed-shape state for the GSCO multistep orchestration loop."""

    x: jax.Array
    previous_x: jax.Array
    loop_count: jax.Array
    enclosed_segment_mask: jax.Array
    current_fraction: jax.Array
    has_previous: jax.Array
    done: jax.Array
    nonfinal_steps: jax.Array
    final_adjustment_run: jax.Array


jax.tree_util.register_dataclass(
    WireframeGSCOMultistepState,
    data_fields=[
        "x",
        "previous_x",
        "loop_count",
        "enclosed_segment_mask",
        "current_fraction",
        "has_previous",
        "done",
        "nonfinal_steps",
        "final_adjustment_run",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class WireframeGSCOMultistepResult:
    """Final fixed-shape result from GSCO multistep orchestration."""

    x: jax.Array
    loop_count: jax.Array
    enclosed_segment_mask: jax.Array
    nonfinal_steps: jax.Array
    final_adjustment_run: jax.Array
    stage_objectives: jax.Array
    stage_count: jax.Array


jax.tree_util.register_dataclass(
    WireframeGSCOMultistepResult,
    data_fields=[
        "x",
        "loop_count",
        "enclosed_segment_mask",
        "nonfinal_steps",
        "final_adjustment_run",
        "stage_objectives",
        "stage_count",
    ],
    meta_fields=[],
)


GSCOStopRule = Callable[[WireframeGSCOLiveState], jax.Array]


def wireframe_gsco_never_stop(state: WireframeGSCOLiveState) -> jax.Array:
    """Keep scanning until the static ``max_steps`` budget is consumed."""

    return _runtime_init_scalar(False, jnp.bool_)


def _gsco_active_entries(x: jax.Array, tol: jax.Array) -> jax.Array:
    return (jnp.abs(x) > tol).astype(x.dtype)


def _gsco_signed_loop_delta(currents: jax.Array) -> jax.Array:
    return jnp.stack((currents, currents, -currents, -currents), axis=-1)


@jax.jit
def _update_vector_entry(
    vector: jax.Array, index: jax.Array, value: jax.Array
) -> jax.Array:
    # jit so the scatter's negative-index normalization (which materializes the
    # axis size as an int64 device scalar) is traced to a compile-time constant
    # rather than executed eagerly -- the latter trips transfer_guard("disallow")
    # when this helper is called outside jit (e.g. the final-flush history writes).
    return vector.at[index].set(value, mode="drop")


@jax.jit
def _gsco_loop_columns(A: jax.Array, loops: jax.Array) -> jax.Array:
    # jit for the same reason as _update_vector_entry: the eager gather below
    # would normalize indices via a device-resident size scalar (host->device
    # transfer) when hoisted out of the jitted candidate-objective body.
    n_loops = int(loops.shape[0])
    candidate_ids = jnp.arange(2 * n_loops, dtype=jnp.int32)
    loop_ids = candidate_ids % _runtime_init_scalar(n_loops, candidate_ids.dtype)
    return A[:, loops[loop_ids, :]]


def _gsco_two_f_s(x: jax.Array, tol: jax.Array) -> jax.Array:
    return jnp.sum(_gsco_active_entries(x, tol))


def _gsco_opposite_candidate_index(opt_ind: jax.Array, n_loops: int) -> jax.Array:
    n_loops_arr = _runtime_init_scalar(n_loops, opt_ind.dtype)
    return (opt_ind + n_loops_arr) % _runtime_init_scalar(2 * n_loops, opt_ind.dtype)


def _gsco_candidate_currents(
    x: jax.Array,
    loop_count: jax.Array,
    params: WireframeGSCOLiveParams,
) -> jax.Array:
    n_loops = int(params.loops.shape[0])
    candidate_ids = jnp.arange(2 * n_loops, dtype=jnp.int32)
    n_loops_arr = _runtime_init_scalar(n_loops, candidate_ids.dtype)
    loop_ids = candidate_ids % n_loops_arr
    positive_float = _runtime_init_scalar(1, x.dtype)
    positive_int = _runtime_init_scalar(1, loop_count.dtype)
    first_direction = candidate_ids < n_loops_arr
    directions = jnp.where(first_direction, positive_float, -positive_float).astype(
        x.dtype
    )
    direction_counts = jnp.where(first_direction, positive_int, -positive_int).astype(
        loop_count.dtype
    )
    loop_inds = params.loops[loop_ids, :]
    loop_x = x[loop_inds]

    eligible = params.free_loops[loop_ids] == _runtime_init_scalar(
        1, params.free_loops.dtype
    )
    if params.max_loop_count > 0:
        next_counts = loop_count[loop_ids] + direction_counts
        eligible = eligible & (
            jnp.abs(next_counts)
            <= _runtime_init_scalar(params.max_loop_count, next_counts.dtype)
        )

    if params.no_new_coils:
        eligible = eligible & jnp.any(
            _gsco_active_entries(loop_x, params.tol) > _runtime_init_scalar(0, x.dtype),
            axis=1,
        )

    zero = _runtime_init_scalar(0, x.dtype)
    if params.match_current:
        abs_loop_x = jnp.abs(loop_x)
        nonzero_currents = abs_loop_x > zero
        matched_abs_current = jnp.max(
            jnp.where(nonzero_currents, abs_loop_x, zero),
            axis=1,
        )
        mismatch = jnp.any(
            jnp.where(
                nonzero_currents,
                abs_loop_x != matched_abs_current[:, None],
                nonzero_currents != nonzero_currents,
            ),
            axis=1,
        )
        loop_current = jnp.where(
            matched_abs_current != zero,
            directions * matched_abs_current,
            directions * params.default_current,
        )
        eligible = eligible & ~mismatch
    else:
        loop_current = directions * params.default_current

    loop_deltas = _gsco_signed_loop_delta(loop_current)
    candidate_x = loop_x + loop_deltas
    eligible = eligible & jnp.all(jnp.abs(candidate_x) <= params.max_current, axis=1)

    if params.no_crossing:
        toroidal_segment_inds = jnp.stack((loop_inds[:, 0], loop_inds[:, 2]), axis=1)
        nodes = jnp.reshape(params.segments[toroidal_segment_inds, :], (2 * n_loops, 4))
        connected = params.connections[nodes, :]
        matches = connected[:, :, :, None] == loop_inds[:, None, None, :]
        first_match = jnp.argmax(matches, axis=3)
        matched_delta = jnp.take_along_axis(
            jnp.broadcast_to(loop_deltas[:, None, None, :], matches.shape),
            first_match[:, :, :, None],
            axis=3,
        )[:, :, :, 0]
        current_to_add = jnp.where(jnp.any(matches, axis=3), matched_delta, zero)
        active_connections = jnp.abs(x[connected] + current_to_add) > params.tol
        crossing_found = jnp.any(
            jnp.sum(active_connections, axis=2) > _runtime_init_scalar(2, jnp.int32),
            axis=1,
        )
        eligible = eligible & ~crossing_found

    return jnp.where(
        eligible, loop_current, _runtime_init_scalar(0, loop_current.dtype)
    )


def _gsco_candidate_objectives(
    x: jax.Array,
    residual: jax.Array,
    candidate_currents: jax.Array,
    two_f_s_latest: jax.Array,
    params: WireframeGSCOLiveParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_loops = int(params.loops.shape[0])
    candidate_ids = jnp.arange(2 * n_loops, dtype=jnp.int32)
    loop_inds = params.loops[
        candidate_ids % _runtime_init_scalar(n_loops, candidate_ids.dtype), :
    ]
    loop_delta = _gsco_signed_loop_delta(candidate_currents)

    loop_columns = (
        _gsco_loop_columns(params.A, params.loops)
        if params.loop_columns is None
        else params.loop_columns
    )
    field_delta = jnp.sum(loop_columns * loop_delta[None, :, :], axis=2)
    two_f_b = jnp.sum((residual[:, None] + field_delta) ** 2, axis=0)

    loop_x = x[loop_inds]
    two_df_orig = jnp.sum(_gsco_active_entries(loop_x, params.tol), axis=1)
    two_df = jnp.sum(_gsco_active_entries(loop_x + loop_delta, params.tol), axis=1)
    two_f_s = two_f_s_latest + two_df - two_df_orig
    two_f = two_f_b + params.lambda_s * two_f_s
    inf = jnp.finfo(params.A.dtype).max
    eligible = candidate_currents != _runtime_init_scalar(0, candidate_currents.dtype)
    return (
        jnp.where(eligible, two_f_b, inf),
        jnp.where(eligible, two_f_s, inf),
        jnp.where(eligible, two_f, inf),
    )


def _validate_gsco_history_capacity(
    state: WireframeGSCOLiveState,
    max_steps: int,
) -> WireframeGSCOLiveState:
    history_capacity = int(state.iter_history.shape[0])
    expected_shape = (history_capacity,)
    history_shapes = (
        state.curr_history.shape,
        state.loop_history.shape,
        state.f_B_history.shape,
        state.f_S_history.shape,
        state.f_history.shape,
    )
    if any(shape != expected_shape for shape in history_shapes):
        raise ValueError(
            "GSCO history arrays must share one capacity; got "
            f"iter_history={state.iter_history.shape}, "
            f"curr_history={state.curr_history.shape}, "
            f"loop_history={state.loop_history.shape}, "
            f"f_B_history={state.f_B_history.shape}, "
            f"f_S_history={state.f_S_history.shape}, "
            f"f_history={state.f_history.shape}."
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be nonnegative; got {max_steps}.")
    if max_steps + 1 > history_capacity:
        raise ValueError(
            "max_steps must fit the GSCO history capacity; "
            f"got max_steps={max_steps}, history_capacity={history_capacity}."
        )
    history_dtype = state.history_length.dtype
    if _has_tracer_leaf(state.history_length):
        max_steps_value = jnp.asarray(max_steps, dtype=history_dtype)
        history_capacity_value = jnp.asarray(
            history_capacity,
            dtype=history_dtype,
        )
        history_is_valid = (state.history_length >= 1) & (
            state.history_length + max_steps_value <= history_capacity_value
        )
        return jax.lax.cond(
            history_is_valid,
            lambda valid_state: valid_state,
            lambda invalid_state: _raise_traced_gsco_history_capacity(
                invalid_state,
                max_steps_value,
                history_capacity_value,
            ),
            state,
        )
    history_length = _validate_gsco_history_length_runtime(
        state.history_length,
        jnp.asarray(max_steps, dtype=history_dtype),
        jnp.asarray(history_capacity, dtype=history_dtype),
    )
    return replace(state, history_length=history_length)


def _raise_traced_gsco_history_capacity(
    state: WireframeGSCOLiveState,
    max_steps: jax.Array,
    history_capacity: jax.Array,
) -> WireframeGSCOLiveState:
    checked_length = io_callback(
        _raise_invalid_gsco_history_length,
        jax.ShapeDtypeStruct((), state.history_length.dtype),
        state.history_length,
        max_steps,
        history_capacity,
        ordered=True,
    )
    return replace(state, history_length=checked_length)


def _raise_invalid_gsco_history_length(
    history_length,
    max_steps,
    history_capacity,
):
    history_length_int = int(history_length)
    max_steps_int = int(max_steps)
    history_capacity_int = int(history_capacity)
    if history_length_int < 1:
        raise ValueError(
            "state.history_length must include the initial row; "
            f"got {history_length_int}."
        )
    raise ValueError(
        "state.history_length + max_steps must fit GSCO history capacity; "
        f"got {history_length_int} + {max_steps_int} > {history_capacity_int}."
    )


def _validate_gsco_history_length_runtime(
    history_length: jax.Array,
    max_steps: jax.Array,
    history_capacity: jax.Array,
) -> jax.Array:
    history_length_int = int(history_length)
    max_steps_int = int(max_steps)
    history_capacity_int = int(history_capacity)
    if history_length_int < 1:
        raise ValueError(
            "state.history_length must include the initial row; "
            f"got {history_length_int}."
        )
    if history_length_int + max_steps_int > history_capacity_int:
        raise ValueError(
            "state.history_length + max_steps must fit GSCO history capacity; "
            f"got {history_length_int} + {max_steps_int} > {history_capacity_int}."
        )
    return history_length


def _validate_record_every(record_every: int | None) -> None:
    if record_every is None:
        return
    if not isinstance(record_every, int):
        raise TypeError(
            f"record_every must be a Python int; got {type(record_every).__name__}"
        )
    if record_every < 1:
        raise ValueError(f"record_every must be positive; got {record_every}.")


def _gsco_record_capacity(max_steps: int, record_every: int) -> int:
    return 1 + max_steps // record_every + 1


def _wireframe_free_loops(
    cell_key: jax.Array,
    constrained_segment_mask: jax.Array,
) -> jax.Array:
    return ~jnp.any(constrained_segment_mask[cell_key], axis=1)


def find_wireframe_coil_sizes_jax(
    loop_count: jax.Array,
    neighbors: jax.Array,
) -> jax.Array:
    """Return connected active-cell sizes for a wireframe GSCO loop count."""

    loop_count_arr = jnp.reshape(_as_jax_int32(loop_count), (-1,))
    neighbors_arr = _as_jax_int32(neighbors)
    n_cells = int(loop_count_arr.shape[0])
    zero_count = _runtime_init_scalar(0, loop_count_arr.dtype)
    active = loop_count_arr != zero_count
    sentinel = _runtime_init_scalar(n_cells, jnp.int32)
    initial_labels = jnp.where(active, jnp.arange(n_cells, dtype=jnp.int32), sentinel)

    def _propagate(labels, _iteration):
        neighbor_labels = labels[neighbors_arr]
        candidate_labels = jnp.where(
            active[neighbors_arr],
            neighbor_labels,
            sentinel,
        )
        return jnp.minimum(labels, jnp.min(candidate_labels, axis=1)), None

    labels, _ = jax.lax.scan(
        _propagate,
        initial_labels,
        jnp.arange(n_cells, dtype=jnp.int32),
    )
    label_for_count = jnp.where(active, labels, _runtime_init_scalar(0, labels.dtype))
    component_sizes = jnp.bincount(
        label_for_count,
        weights=active.astype(jnp.int32),
        length=n_cells,
    )
    safe_labels = jnp.minimum(labels, _runtime_init_scalar(n_cells - 1, labels.dtype))
    return jnp.where(
        active,
        component_sizes[safe_labels],
        _runtime_init_scalar(0, component_sizes.dtype),
    )


def _wireframe_enclosed_segment_mask(
    x: jax.Array,
    loop_count: jax.Array,
    cell_key: jax.Array,
) -> jax.Array:
    active_cells = jnp.reshape(loop_count, (-1,)) != _runtime_init_scalar(
        0, loop_count.dtype
    )
    n_segments = int(x.shape[0])
    enclosed_count = jnp.bincount(
        jnp.reshape(cell_key, (-1,)),
        weights=jnp.repeat(active_cells.astype(jnp.int32), cell_key.shape[1]),
        length=n_segments,
    )
    enclosed = enclosed_count > _runtime_init_scalar(0, enclosed_count.dtype)
    return enclosed & (jnp.reshape(x, (-1,)) == _runtime_init_scalar(0, x.dtype))


def _wireframe_prune_small_coils(
    x: jax.Array,
    loop_count: jax.Array,
    cell_key: jax.Array,
    neighbors: jax.Array,
    *,
    min_coil_size: int,
) -> tuple[jax.Array, jax.Array]:
    coil_sizes = find_wireframe_coil_sizes_jax(loop_count, neighbors)
    small_cells = (coil_sizes > _runtime_init_scalar(0, coil_sizes.dtype)) & (
        coil_sizes < _runtime_init_scalar(int(min_coil_size), coil_sizes.dtype)
    )
    n_segments = int(x.shape[0])
    segment_prune_count = jnp.bincount(
        jnp.reshape(cell_key, (-1,)),
        weights=jnp.repeat(small_cells.astype(jnp.int32), cell_key.shape[1]),
        length=n_segments,
    )
    segment_prune_mask = segment_prune_count > _runtime_init_scalar(
        0, segment_prune_count.dtype
    )
    return (
        jnp.where(segment_prune_mask, x - x, x),
        jnp.where(small_cells, loop_count - loop_count, loop_count),
    )


def wireframe_gsco_initial_state(
    params: WireframeGSCOLiveParams,
    b: jax.Array,
    x_init: jax.Array,
    loop_count_init: jax.Array,
    *,
    history_capacity: int,
) -> WireframeGSCOLiveState:
    """Create a fixed-capacity GSCO live state with the initial history row."""

    b_arr = jnp.reshape(_as_runtime_array(b), (-1,))
    x0 = jnp.reshape(_as_runtime_array(x_init), (-1,))
    loop_count0 = jnp.reshape(_as_jax_int32(loop_count_init), (-1,))
    residual0 = params.A @ x0 - b_arr
    two_f_b0 = jnp.sum(residual0 * residual0)
    two_f_s0 = _gsco_two_f_s(x0, params.tol)
    two_f0 = two_f_b0 + params.lambda_s * two_f_s0
    half = _runtime_init_scalar(0.5, params.A.dtype)

    iter_history = _runtime_init_array((history_capacity,), 0, jnp.int32)
    curr_history = _runtime_init_array((history_capacity,), 0, params.A.dtype)
    loop_history = _runtime_init_array((history_capacity,), 0, jnp.int32)
    history_tail = _runtime_init_array((history_capacity - 1,), 0, params.A.dtype)
    f_b_history = jnp.concatenate((jnp.reshape(half * two_f_b0, (1,)), history_tail))
    f_s_history = jnp.concatenate((jnp.reshape(half * two_f_s0, (1,)), history_tail))
    f_history = jnp.concatenate((jnp.reshape(half * two_f0, (1,)), history_tail))
    return WireframeGSCOLiveState(
        x=x0,
        loop_count=loop_count0,
        residual=residual0,
        two_f_b=two_f_b0,
        two_f_s=two_f_s0,
        two_f=two_f0,
        opt_ind_prev=_runtime_init_scalar(-1, jnp.int32),
        history_length=_runtime_init_scalar(1, jnp.int32),
        done=_runtime_init_scalar(False, jnp.bool_),
        iter_history=iter_history,
        curr_history=curr_history,
        loop_history=loop_history,
        f_B_history=f_b_history,
        f_S_history=f_s_history,
        f_history=f_history,
    )


def _gsco_live_step(
    state: WireframeGSCOLiveState,
    iteration: jax.Array,
    params: WireframeGSCOLiveParams,
    stop_rule: GSCOStopRule,
) -> WireframeGSCOLiveState:
    candidate_currents = _gsco_candidate_currents(state.x, state.loop_count, params)
    two_f_bs, two_f_ss, two_fs = _gsco_candidate_objectives(
        state.x,
        state.residual,
        candidate_currents,
        state.two_f_s,
        params,
    )

    n_loops = int(params.loops.shape[0])
    opt_ind = jnp.argmin(two_fs).astype(jnp.int32)
    current = candidate_currents[opt_ind]
    n_loops_arr = _runtime_init_scalar(n_loops, opt_ind.dtype)
    loop_ind = (opt_ind % n_loops_arr).astype(jnp.int32)
    one_loop_count = _runtime_init_scalar(1, state.loop_count.dtype)
    direction_count = jnp.where(
        opt_ind < n_loops_arr,
        one_loop_count,
        -one_loop_count,
    ).astype(state.loop_count.dtype)
    loop_inds = params.loops[loop_ind, :]
    delta_x = _gsco_signed_loop_delta(current)
    residual_delta = jnp.sum(params.A[:, loop_inds] * delta_x[None, :], axis=1)

    n_eligible = jnp.sum(
        candidate_currents != _runtime_init_scalar(0, candidate_currents.dtype)
    )
    stop_none_eligible = n_eligible < _runtime_init_scalar(1, n_eligible.dtype)
    opposite_opt_ind = _gsco_opposite_candidate_index(opt_ind, n_loops)
    stop_undone_loop = (iteration > _runtime_init_scalar(0, iteration.dtype)) & (
        opposite_opt_ind == state.opt_ind_prev
    )
    reject_undo = stop_undone_loop & (two_fs[opt_ind] > state.two_f)
    accept_loop = (~stop_none_eligible) & (~reject_undo)
    stop_now = stop_none_eligible | stop_undone_loop

    x_candidate = state.x.at[loop_inds].add(delta_x)
    loop_count_candidate = state.loop_count.at[loop_ind].add(direction_count)
    residual_candidate = state.residual + residual_delta
    two_f_b_candidate = two_f_bs[opt_ind]
    two_f_s_candidate = two_f_ss[opt_ind]
    two_f_candidate = two_fs[opt_ind]

    x_next = jnp.where(accept_loop, x_candidate, state.x)
    loop_count_next = jnp.where(accept_loop, loop_count_candidate, state.loop_count)
    residual_next = jnp.where(accept_loop, residual_candidate, state.residual)
    two_f_b_next = jnp.where(accept_loop, two_f_b_candidate, state.two_f_b)
    two_f_s_next = jnp.where(accept_loop, two_f_s_candidate, state.two_f_s)
    two_f_next = jnp.where(accept_loop, two_f_candidate, state.two_f)

    write_index = state.history_length
    dropped_index = _runtime_init_scalar(state.iter_history.shape[0], write_index.dtype)
    write_index_or_drop = jnp.where(accept_loop, write_index, dropped_index)
    history_length_next = write_index + accept_loop.astype(write_index.dtype)
    iter_history_candidate = _update_vector_entry(
        state.iter_history, write_index_or_drop, write_index
    )
    curr_history_candidate = _update_vector_entry(
        state.curr_history, write_index_or_drop, current
    )
    loop_history_candidate = _update_vector_entry(
        state.loop_history, write_index_or_drop, loop_ind
    )
    half = _runtime_init_scalar(0.5, params.A.dtype)
    f_b_history_candidate = _update_vector_entry(
        state.f_B_history, write_index_or_drop, half * two_f_b_next
    )
    f_s_history_candidate = _update_vector_entry(
        state.f_S_history, write_index_or_drop, half * two_f_s_next
    )
    f_history_candidate = _update_vector_entry(
        state.f_history, write_index_or_drop, half * two_f_next
    )

    next_state = WireframeGSCOLiveState(
        x=x_next,
        loop_count=loop_count_next,
        residual=residual_next,
        two_f_b=two_f_b_next,
        two_f_s=two_f_s_next,
        two_f=two_f_next,
        opt_ind_prev=jnp.where(
            (~stop_none_eligible) & (~stop_undone_loop),
            opt_ind,
            state.opt_ind_prev,
        ),
        history_length=history_length_next,
        done=state.done | stop_now,
        iter_history=iter_history_candidate,
        curr_history=curr_history_candidate,
        loop_history=loop_history_candidate,
        f_B_history=f_b_history_candidate,
        f_S_history=f_s_history_candidate,
        f_history=f_history_candidate,
    )
    return replace(next_state, done=next_state.done | stop_rule(next_state))


def gsco_live_loop_jax(
    state: WireframeGSCOLiveState,
    *,
    max_steps: int,
    params: WireframeGSCOLiveParams,
    stop_rule: GSCOStopRule = wireframe_gsco_never_stop,
) -> WireframeGSCOLiveState:
    """Advance GSCO through a fixed-length device-side scan."""

    state = _validate_gsco_history_capacity(state, int(max_steps))
    return _gsco_live_loop_unchecked(
        state,
        max_steps=int(max_steps),
        params=params,
        stop_rule=stop_rule,
    )


def _gsco_live_loop_unchecked(
    state: WireframeGSCOLiveState,
    *,
    max_steps: int,
    params: WireframeGSCOLiveParams,
    stop_rule: GSCOStopRule = wireframe_gsco_never_stop,
) -> WireframeGSCOLiveState:
    """Run the GSCO scan after the caller has established capacity."""

    def _step(
        current: WireframeGSCOLiveState,
        iteration: jax.Array,
    ) -> WireframeGSCOLiveState:
        return _gsco_live_step(
            current,
            iteration,
            params,
            stop_rule,
        )

    final_state = _bounded_scan_until_done(
        state,
        max_steps=int(max_steps),
        is_done=lambda current: current.done,
        step=_step,
    )
    return final_state


def _to_result(state: WireframeGSCOLiveState) -> WireframeGSCOResult:
    return WireframeGSCOResult(
        x=jnp.reshape(state.x, (-1, 1)),
        loop_count=state.loop_count,
        history_length=state.history_length,
        iter_history=state.iter_history,
        curr_history=state.curr_history,
        loop_history=state.loop_history,
        f_B_history=state.f_B_history,
        f_S_history=state.f_S_history,
        f_history=state.f_history,
    )


def _greedy_stellarator_coil_optimization_sampled_jax(
    params: WireframeGSCOLiveParams,
    b_obj: object,
    x_init: object,
    loop_count_init: object,
    *,
    max_iter: int,
    record_every: int,
) -> WireframeGSCOResult:
    b_arr = jnp.reshape(_as_runtime_array(b_obj), (-1,))
    x0 = jnp.reshape(_as_runtime_array(x_init), (-1,))
    loop_count0 = jnp.reshape(_as_jax_int32(loop_count_init), (-1,))
    residual0 = params.A @ x0 - b_arr
    two_f_b0 = jnp.sum(residual0 * residual0)
    two_f_s0 = _gsco_two_f_s(x0, params.tol)
    two_f0 = two_f_b0 + params.lambda_s * two_f_s0
    record_capacity = _gsco_record_capacity(max_iter, record_every)
    half = _runtime_init_scalar(0.5, params.A.dtype)
    iter_history0 = _runtime_init_array((record_capacity,), 0, jnp.int32)
    curr_history0 = _runtime_init_array((record_capacity,), 0, params.A.dtype)
    loop_history0 = _runtime_init_array((record_capacity,), 0, jnp.int32)
    history_tail = _runtime_init_array((record_capacity - 1,), 0, params.A.dtype)
    f_b_history0 = jnp.concatenate((jnp.reshape(half * two_f_b0, (1,)), history_tail))
    f_s_history0 = jnp.concatenate((jnp.reshape(half * two_f_s0, (1,)), history_tail))
    f_history0 = jnp.concatenate((jnp.reshape(half * two_f0, (1,)), history_tail))

    def _active_step(carry, iteration):
        (
            x,
            loop_count,
            residual,
            two_f_b,
            two_f_s,
            two_f,
            opt_ind_prev,
            done,
            accepted_count,
            recorded_count,
            last_iter,
            last_curr,
            last_loop,
            last_f_b,
            last_f_s,
            last_f,
            iter_history,
            curr_history,
            loop_history,
            f_b_history,
            f_s_history,
            f_history,
        ) = carry
        candidate_currents = _gsco_candidate_currents(x, loop_count, params)
        two_f_bs, two_f_ss, two_fs = _gsco_candidate_objectives(
            x,
            residual,
            candidate_currents,
            two_f_s,
            params,
        )

        n_loops = int(params.loops.shape[0])
        opt_ind = jnp.argmin(two_fs).astype(jnp.int32)
        current = candidate_currents[opt_ind]
        n_loops_arr = _runtime_init_scalar(n_loops, opt_ind.dtype)
        loop_ind = (opt_ind % n_loops_arr).astype(jnp.int32)
        one_loop_count = _runtime_init_scalar(1, loop_count.dtype)
        direction_count = jnp.where(
            opt_ind < n_loops_arr,
            one_loop_count,
            -one_loop_count,
        ).astype(loop_count.dtype)
        loop_inds = params.loops[loop_ind, :]
        delta_x = _gsco_signed_loop_delta(current)
        residual_delta = jnp.sum(params.A[:, loop_inds] * delta_x[None, :], axis=1)

        zero_current = _runtime_init_scalar(0, candidate_currents.dtype)
        n_eligible = jnp.sum(candidate_currents != zero_current)
        one_count = _runtime_init_scalar(1, n_eligible.dtype)
        zero_count = _runtime_init_scalar(0, n_eligible.dtype)
        stop_none_eligible = n_eligible < one_count
        opposite_opt_ind = _gsco_opposite_candidate_index(opt_ind, n_loops)
        stop_undone_loop = (iteration > _runtime_init_scalar(0, iteration.dtype)) & (
            opposite_opt_ind == opt_ind_prev
        )
        reject_undo = stop_undone_loop & (two_fs[opt_ind] > two_f)
        accept_loop = (~stop_none_eligible) & (~reject_undo)
        stop_now = stop_none_eligible | stop_undone_loop

        x_candidate = x.at[loop_inds].add(delta_x)
        loop_count_candidate = loop_count.at[loop_ind].add(direction_count)
        residual_candidate = residual + residual_delta
        two_f_b_candidate = two_f_bs[opt_ind]
        two_f_s_candidate = two_f_ss[opt_ind]
        two_f_candidate = two_fs[opt_ind]
        x_next = jnp.where(accept_loop, x_candidate, x)
        loop_count_next = jnp.where(accept_loop, loop_count_candidate, loop_count)
        residual_next = jnp.where(accept_loop, residual_candidate, residual)
        two_f_b_next = jnp.where(accept_loop, two_f_b_candidate, two_f_b)
        two_f_s_next = jnp.where(accept_loop, two_f_s_candidate, two_f_s)
        two_f_next = jnp.where(accept_loop, two_f_candidate, two_f)

        accepted_count_next = accepted_count + accept_loop.astype(accepted_count.dtype)
        should_record = accept_loop & (
            (
                accepted_count_next
                % _runtime_init_scalar(record_every, accepted_count_next.dtype)
            )
            == zero_count
        )
        slot = recorded_count
        recorded_count_next = recorded_count + should_record.astype(
            recorded_count.dtype
        )
        dropped_slot = _runtime_init_scalar(iter_history.shape[0], slot.dtype)
        slot_or_drop = jnp.where(should_record, slot, dropped_slot)
        f_b_row = half * two_f_b_next
        f_s_row = half * two_f_s_next
        f_row = half * two_f_next

        return (
            x_next,
            loop_count_next,
            residual_next,
            two_f_b_next,
            two_f_s_next,
            two_f_next,
            jnp.where(
                (~stop_none_eligible) & (~stop_undone_loop),
                opt_ind,
                opt_ind_prev,
            ),
            done | stop_now,
            accepted_count_next,
            recorded_count_next,
            jnp.where(accept_loop, accepted_count_next, last_iter),
            jnp.where(accept_loop, current, last_curr),
            jnp.where(accept_loop, loop_ind, last_loop),
            jnp.where(accept_loop, f_b_row, last_f_b),
            jnp.where(accept_loop, f_s_row, last_f_s),
            jnp.where(accept_loop, f_row, last_f),
            _update_vector_entry(
                iter_history,
                slot_or_drop,
                accepted_count_next,
            ),
            _update_vector_entry(
                curr_history,
                slot_or_drop,
                current,
            ),
            _update_vector_entry(
                loop_history,
                slot_or_drop,
                loop_ind,
            ),
            _update_vector_entry(
                f_b_history,
                slot_or_drop,
                f_b_row,
            ),
            _update_vector_entry(
                f_s_history,
                slot_or_drop,
                f_s_row,
            ),
            _update_vector_entry(
                f_history,
                slot_or_drop,
                f_row,
            ),
        ), None

    def _scan_body(carry, iteration):
        return jax.lax.cond(
            carry[7],
            lambda done_carry: (done_carry, None),
            lambda active_carry: _active_step(active_carry, iteration),
            carry,
        )

    final_carry, _ = jax.lax.scan(
        _scan_body,
        (
            x0,
            loop_count0,
            residual0,
            two_f_b0,
            two_f_s0,
            two_f0,
            _runtime_init_scalar(-1, jnp.int32),
            _runtime_init_scalar(False, jnp.bool_),
            _runtime_init_scalar(0, jnp.int32),
            _runtime_init_scalar(1, jnp.int32),
            _runtime_init_scalar(0, jnp.int32),
            _runtime_init_scalar(0, params.A.dtype),
            _runtime_init_scalar(0, jnp.int32),
            half * two_f_b0,
            half * two_f_s0,
            half * two_f0,
            iter_history0,
            curr_history0,
            loop_history0,
            f_b_history0,
            f_s_history0,
            f_history0,
        ),
        jnp.arange(max_iter, dtype=jnp.int32),
    )
    (
        x,
        loop_count,
        _residual,
        _two_f_b,
        _two_f_s,
        _two_f,
        _opt_ind_prev,
        _done,
        accepted_count,
        recorded_count,
        last_iter,
        last_curr,
        last_loop,
        last_f_b,
        last_f_s,
        last_f,
        iter_history,
        curr_history,
        loop_history,
        f_b_history,
        f_s_history,
        f_history,
    ) = final_carry
    zero_count = _runtime_init_scalar(0, accepted_count.dtype)
    record_every_arr = _runtime_init_scalar(record_every, accepted_count.dtype)
    final_unrecorded = (accepted_count > zero_count) & (
        (accepted_count % record_every_arr) != zero_count
    )
    final_slot = recorded_count
    dropped_final_slot = _runtime_init_scalar(iter_history.shape[0], final_slot.dtype)
    final_slot_or_drop = jnp.where(final_unrecorded, final_slot, dropped_final_slot)
    history_length = recorded_count + final_unrecorded.astype(recorded_count.dtype)
    return WireframeGSCOResult(
        x=jnp.reshape(x, (-1, 1)),
        loop_count=loop_count,
        history_length=history_length,
        iter_history=_update_vector_entry(
            iter_history,
            final_slot_or_drop,
            last_iter,
        ),
        curr_history=_update_vector_entry(
            curr_history,
            final_slot_or_drop,
            last_curr,
        ),
        loop_history=_update_vector_entry(
            loop_history,
            final_slot_or_drop,
            last_loop,
        ),
        f_B_history=_update_vector_entry(
            f_b_history,
            final_slot_or_drop,
            last_f_b,
        ),
        f_S_history=_update_vector_entry(
            f_s_history,
            final_slot_or_drop,
            last_f_s,
        ),
        f_history=_update_vector_entry(
            f_history,
            final_slot_or_drop,
            last_f,
        ),
    )


def greedy_stellarator_coil_optimization_jax(
    no_crossing: bool,
    no_new_coils: bool,
    match_current: bool,
    A_obj: object,
    b_obj: object,
    default_current: float,
    max_current: float,
    max_loop_count: int,
    loops: object,
    free_loops: object,
    segments: object,
    connections: object,
    lambda_S: float,
    max_iter: int,
    x_init: object,
    loop_count_init: object,
    record_every: int | None = None,
) -> WireframeGSCOResult:
    """Run the fixed-state GSCO live loop and return fixed-shape histories."""

    A = _as_runtime_array(A_obj)
    loops_arr = _as_jax_int32(loops)
    n_iter = int(max_iter)
    _validate_record_every(record_every)
    default_current_abs = jnp.abs(_runtime_init_scalar(default_current, A.dtype))
    params = WireframeGSCOLiveParams(
        A=A,
        loops=loops_arr,
        free_loops=_as_jax_int32(free_loops),
        segments=_as_jax_int32(segments),
        connections=_as_jax_int32(connections),
        default_current=default_current_abs,
        max_current=jnp.abs(_runtime_init_scalar(max_current, A.dtype)),
        lambda_s=_runtime_init_scalar(lambda_S, A.dtype),
        tol=_runtime_init_scalar(0.001, A.dtype) * default_current_abs,
        max_loop_count=abs(int(max_loop_count)),
        no_crossing=no_crossing,
        no_new_coils=no_new_coils,
        match_current=match_current,
        loop_columns=_gsco_loop_columns(A, loops_arr),
    )
    if record_every is not None:
        return _greedy_stellarator_coil_optimization_sampled_jax(
            params,
            b_obj,
            x_init,
            loop_count_init,
            max_iter=n_iter,
            record_every=record_every,
        )
    state = wireframe_gsco_initial_state(
        params,
        b_obj,
        x_init,
        loop_count_init,
        history_capacity=n_iter + 1,
    )
    return _to_result(_gsco_live_loop_unchecked(state, max_steps=n_iter, params=params))


def _wireframe_gsco_multistep_initial_state(
    x_init: object,
    loop_count_init: object,
    *,
    current_fraction: float,
) -> WireframeGSCOMultistepState:
    x0 = jnp.reshape(_as_runtime_array(x_init), (-1,))
    loop_count0 = jnp.reshape(_as_jax_int32(loop_count_init), (-1,))
    return WireframeGSCOMultistepState(
        x=x0,
        previous_x=x0 - x0,
        loop_count=loop_count0,
        enclosed_segment_mask=x0 != x0,
        current_fraction=_runtime_init_scalar(current_fraction, x0.dtype),
        has_previous=_runtime_init_scalar(False, jnp.bool_),
        done=_runtime_init_scalar(False, jnp.bool_),
        nonfinal_steps=_runtime_init_scalar(0, jnp.int32),
        final_adjustment_run=_runtime_init_scalar(False, jnp.bool_),
    )


def wireframe_gsco_multistep_loop_jax(
    base_params: WireframeGSCOLiveParams,
    b_obj: object,
    x_init: object,
    loop_count_init: object,
    cell_key: object,
    cell_neighbors: object,
    base_constrained_segment_mask: object,
    *,
    max_iter_per_step: int,
    max_outer_steps: int,
    initial_current_fraction: float,
    current_scale: float,
    min_coil_size: int,
    final_max_current: float,
) -> WireframeGSCOMultistepResult:
    """Run the wireframe GSCO multistep numerical state machine in JAX."""

    b_arr = jnp.reshape(_as_runtime_array(b_obj), (-1,))
    cell_key_arr = _as_jax_int32(cell_key)
    cell_neighbors_arr = _as_jax_int32(cell_neighbors)
    base_constrained = jnp.reshape(
        jnp.asarray(base_constrained_segment_mask, dtype=bool),
        (-1,),
    )
    n_iter = int(max_iter_per_step)
    outer_steps = int(max_outer_steps)
    if n_iter < 0:
        raise ValueError(f"max_iter_per_step must be nonnegative; got {n_iter}.")
    if outer_steps < 1:
        raise ValueError(f"max_outer_steps must be positive; got {outer_steps}.")
    if min_coil_size < 1:
        raise ValueError(f"min_coil_size must be positive; got {min_coil_size}.")

    state0 = _wireframe_gsco_multistep_initial_state(
        x_init,
        loop_count_init,
        current_fraction=initial_current_fraction,
    )
    current_scale_arr = _runtime_init_scalar(current_scale, base_params.A.dtype)
    final_max_current_arr = _runtime_init_scalar(final_max_current, base_params.A.dtype)

    def _run_gsco_step(
        state: WireframeGSCOMultistepState,
        constrained: jax.Array,
        *,
        default_current: jax.Array,
        max_current: jax.Array,
        tol: jax.Array,
        no_new_coils: bool,
        match_current: bool,
    ) -> WireframeGSCOLiveState:
        params = replace(
            base_params,
            free_loops=_wireframe_free_loops(cell_key_arr, constrained).astype(
                base_params.free_loops.dtype
            ),
            default_current=default_current,
            max_current=max_current,
            tol=tol,
            no_new_coils=no_new_coils,
            match_current=match_current,
        )
        initial = wireframe_gsco_initial_state(
            params,
            b_arr,
            state.x,
            state.loop_count,
            history_capacity=n_iter + 1,
        )
        return _gsco_live_loop_unchecked(initial, max_steps=n_iter, params=params)

    def _nonfinal_update(
        state: WireframeGSCOMultistepState,
    ) -> WireframeGSCOMultistepState:
        gsco_state = _run_gsco_step(
            state,
            base_constrained | state.enclosed_segment_mask,
            default_current=jnp.abs(state.current_fraction * current_scale_arr),
            max_current=_runtime_init_scalar(1.1, base_params.A.dtype)
            * jnp.abs(state.current_fraction * current_scale_arr),
            tol=_runtime_init_scalar(0.001, base_params.A.dtype)
            * jnp.abs(state.current_fraction * current_scale_arr),
            no_new_coils=False,
            match_current=False,
        )
        pruned_x, pruned_loop_count = _wireframe_prune_small_coils(
            gsco_state.x,
            gsco_state.loop_count,
            cell_key_arr,
            cell_neighbors_arr,
            min_coil_size=min_coil_size,
        )
        enclosed = _wireframe_enclosed_segment_mask(
            pruned_x,
            pruned_loop_count,
            cell_key_arr,
        )
        return WireframeGSCOMultistepState(
            x=pruned_x,
            previous_x=state.x,
            loop_count=pruned_loop_count,
            enclosed_segment_mask=enclosed,
            current_fraction=_runtime_init_scalar(0.5, base_params.A.dtype)
            * state.current_fraction,
            has_previous=_runtime_init_scalar(True, jnp.bool_),
            done=_runtime_init_scalar(False, jnp.bool_),
            nonfinal_steps=state.nonfinal_steps + _runtime_init_scalar(1, jnp.int32),
            final_adjustment_run=_runtime_init_scalar(False, jnp.bool_),
        )

    def _final_update(
        state: WireframeGSCOMultistepState,
    ) -> WireframeGSCOMultistepState:
        gsco_state = _run_gsco_step(
            state,
            base_constrained,
            default_current=_runtime_init_scalar(0, base_params.A.dtype),
            max_current=jnp.abs(final_max_current_arr),
            tol=_runtime_init_scalar(0, base_params.A.dtype),
            no_new_coils=True,
            match_current=True,
        )
        return replace(
            state,
            x=gsco_state.x,
            loop_count=gsco_state.loop_count,
            enclosed_segment_mask=jnp.zeros_like(state.enclosed_segment_mask),
            done=_runtime_init_scalar(True, jnp.bool_),
            final_adjustment_run=_runtime_init_scalar(True, jnp.bool_),
        )

    def _active_outer_step(
        state: WireframeGSCOMultistepState,
    ) -> WireframeGSCOMultistepState:
        final_step = state.has_previous & jnp.all(state.previous_x == state.x)
        return jax.lax.cond(final_step, _final_update, _nonfinal_update, state)

    def _outer_scan_body(state: WireframeGSCOMultistepState, _iteration: jax.Array):
        stage_is_active = ~state.done
        next_state = jax.lax.cond(
            state.done,
            lambda done_state: done_state,
            _active_outer_step,
            state,
        )
        residual = base_params.A @ next_state.x - b_arr
        objective = _runtime_init_scalar(0.5, residual.dtype) * jnp.sum(
            residual * residual
        )
        stage_objective = jnp.where(
            stage_is_active,
            objective,
            _runtime_init_scalar(jnp.nan, objective.dtype),
        )
        return next_state, (stage_objective, stage_is_active)

    final_state, (stage_objectives, stage_is_active) = jax.lax.scan(
        _outer_scan_body,
        state0,
        jnp.arange(outer_steps, dtype=jnp.int32),
    )
    return WireframeGSCOMultistepResult(
        x=jnp.reshape(final_state.x, (-1, 1)),
        loop_count=final_state.loop_count,
        enclosed_segment_mask=final_state.enclosed_segment_mask,
        nonfinal_steps=final_state.nonfinal_steps,
        final_adjustment_run=final_state.final_adjustment_run,
        stage_objectives=stage_objectives,
        stage_count=jnp.sum(stage_is_active.astype(jnp.int32)),
    )
