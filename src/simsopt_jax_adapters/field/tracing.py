"""JAX tracing wrappers for fieldlines and particle trajectories."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from itertools import repeat
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
import simsoptpp as sopp
from jax.sharding import PartitionSpec as P

from simsopt._core.types import RealArray
from simsopt._core.util import parallel_loop_bounds
from simsopt._core.tracing_metadata import (
    levelset_classifier_from_interpolant,
    levelset_classifier_for,
)
from simsopt.field.magneticfield import MagneticField
from simsopt.field.tracing import gc_to_fullorbit_initial_guesses
from simsopt_jax.backend.runtime import get_backend_mode
from simsopt_jax.core.sharding import (
    maybe_shard_trajectory_batch_inputs,
    replicate_tree_on_mesh,
    trajectory_batch_sharding_config,
)
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.runtime.host_boundary import host_array as _jax_trace_host_array
from simsopt_jax.core.interpolated_field import (
    InterpolatedFieldCylCache,
    interpolated_field_cyl_cache_zeros,
    interpolated_field_state_B_GradAbsB_cached,
)
from simsopt_jax.core.tracing import (
    TRACING_STATUS_BOOZER_AXIS,
    TRACING_STATUS_INCOMPLETE,
    AdaptiveLoop,
    CartesianTracingContinuationState,
    FieldlineTracingResult,
    FieldlineTracingSpec,
    FullorbitTracingSpec,
    GuidingCenterTracingResult,
    GuidingCenterTracingSpec,
    IterStoppingCriterion as JaxIterStoppingCriterion,
    LevelsetStoppingCriterion as JaxLevelsetStoppingCriterion,
    MaxRStoppingCriterion as JaxMaxRStoppingCriterion,
    MaxToroidalFluxStoppingCriterion as JaxMaxToroidalFluxStoppingCriterion,
    MaxZStoppingCriterion as JaxMaxZStoppingCriterion,
    MinRStoppingCriterion as JaxMinRStoppingCriterion,
    MinToroidalFluxStoppingCriterion as JaxMinToroidalFluxStoppingCriterion,
    MinZStoppingCriterion as JaxMinZStoppingCriterion,
    ToroidalTransitStoppingCriterion as JaxToroidalTransitStoppingCriterion,
    _boozer_particle_dtmaxs,
    _cartesian_particle_dtmaxs,
    _cartesian_radii,
    _fieldline_dtmaxs,
    _magnetic_moments,
    _quarter_turn_dtmaxs,
    _trace_fieldline_chunk,
    _trace_guiding_center_chunk,
    trace_fullorbits_batched,
    trace_guiding_centers_boozer_batched,
)
from simsopt_jax_adapters.field.boozer_field import (
    BoozerRadialInterpolantJAX,
    InterpolatedBoozerFieldJAX,
)
from simsopt_jax_adapters.geo.surface_classifier import (
    levelset_classifier_fn_from_surface_classifier,
    supports_levelset_classifier_conversion,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)


logger = logging.getLogger(__name__)

# Chunk size of the continued Cartesian tracers. It sizes the fixed trajectory
# buffer of one traced call, never the horizon: chunks are repeated until the
# lane reaches ``tmax``, fires a stopping criterion, or exhausts a caller-set
# ``max_steps``. Upstream has no horizon of its own -- ``simsoptpp/tracing.cpp``
# loops ``do { ... } while(t < tmax && !stop);`` and only an opt-in
# ``IterationStoppingCriterion`` can bound the trial count.
_TRACING_CHUNK_TRIALS = 4_000

# The Boozer guiding-centre and full-orbit routes are NOT continued: their core
# entry points allocate the whole trajectory inside one traced call and expose no
# resume state, so for them the step limit is both the buffer shape and a hard
# cap. Upstream bounds neither (both call the same unlimited ``solve()``), so a
# run that exhausts one of these limits is published with status 1, never 0.
_BOOZER_GUIDING_CENTRE_STEP_LIMIT = 4_000
_FULLORBIT_STEP_LIMIT = 20_000

__all__ = [
    "compute_fieldlines",
    "compute_fieldlines_with_status",
    "trace_particles",
    "trace_particles_boozer",
    "trace_particles_with_status",
]


def _adaptive_loop_for_backend_mode(mode: str) -> AdaptiveLoop:
    """Loop form for the routes that return device arrays from one traced call.

    ``scan`` keeps ``trace_guiding_centers_boozer_batched`` and
    ``trace_fullorbits_batched`` reverse-mode differentiable (``lax.while_loop``
    is not) WHEN NO phi/zeta targets are requested; ``_fast`` modes trade that
    for the early exit. With targets the event localizer's own
    ``jax.lax.while_loop`` (``simsopt_jax.core.tracing.bracket_root_jax``) is in
    the graph for every driver, so reverse mode is blocked there whatever this
    loop form is -- ``_scan_angle_plane_events`` returns early only at
    ``num_targets == 0``.
    """
    return "while" if mode.endswith("_fast") else "scan"


# The two chunked Cartesian routes always take the early-exit form. Their chunks
# are drained to NumPy and assembled in a host loop, so nothing can differentiate
# through them and ``scan`` buys only cost: it runs every masked body iteration of
# the chunk after the last lane has finished. The two forms are bitwise equal on
# the trajectory, the event rows, ``t_final`` and ``h``
# (``tests/field/test_tracing_chunked_adapter.py::``
# ``test_chunked_routes_use_the_early_exit_loop_and_match_the_scan_form``).
#
# What this does NOT do: ``while`` shortens a chunk once ALL lanes are done,
# but under ``jax.vmap``
# ``lax.while_loop``'s batching rule keeps executing the body for lanes whose
# own predicate is already False (the result is select-masked), so a finished
# lane is still re-entered, re-traced and re-drained in every later chunk. The
# batch is deliberately not compacted: compaction would cost more than the
# masked iterations it saves.
_CHUNKED_ADAPTIVE_LOOP: AdaptiveLoop = "while"


def _stage_stopping_criterion_leaf(leaf):
    """Place host threshold leaves before they enter a strict JIT argument."""
    if isinstance(leaf, jax.Array):
        return leaf
    return _as_jax_float64(np.asarray(leaf, dtype=np.float64))


def _normalize_parallel_speeds(
    parallel_speeds: RealArray, nparticles: int
) -> np.ndarray:
    speed_par = np.asarray(parallel_speeds).reshape(-1)
    if speed_par.shape != (nparticles,):
        raise ValueError(
            f"Expected {nparticles} parallel speeds, got shape {speed_par.shape}"
        )
    return speed_par


def _allgather_flat(comm, values):
    if comm is None:
        return values
    return [item for rank_values in comm.allgather(values) for item in rank_values]


def _event_hits_prefix(phi_hits, phi_hits_count, *, context: str) -> np.ndarray:
    hits = np.asarray(phi_hits, dtype=np.float64)
    count = int(phi_hits_count)
    max_phi_hits = hits.shape[0]
    if count > max_phi_hits:
        raise RuntimeError(
            f"{context} recorded {count} event rows but max_phi_hits={max_phi_hits}; "
            "increase the JAX tracing event buffer before using these results."
        )
    return hits[:count]


def _batched_jax_event_rows(result, *, event_context: str) -> list[np.ndarray]:
    """Host copies of the recorded event rows, one ragged array per lane."""
    phi_hits = _jax_trace_host_array(result.phi_hits, dtype=np.float64)
    phi_hit_counts = _jax_trace_host_array(
        result.phi_hits_count, dtype=np.int64
    ).reshape(-1)
    return [
        _event_hits_prefix(hits, hit_count, context=event_context)
        for hits, hit_count in zip(phi_hits, phi_hit_counts, strict=True)
    ]


def _batched_jax_live_rows(result) -> list[np.ndarray]:
    """Host copies of the recorded trajectory rows, one ragged array per lane."""
    trajectories = _jax_trace_host_array(result.trajectory, dtype=np.float64)
    masks = _jax_trace_host_array(result.mask, dtype=bool)
    return [traj[mask] for traj, mask in zip(trajectories, masks, strict=True)]


@jax.jit
def _chunk_endpoint_rows(
    trajectory: jax.Array, steps_taken: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """First and last recorded row of every lane, gathered before the transfer.

    Rows are written at indices ``0 .. steps_taken``, so ``steps_taken`` indexes
    the last live row. The padded tail cannot be used instead: padding repeats
    the post-step state, which on a criterion stop is exactly the row upstream
    does not keep.
    """
    lanes = jnp.arange(trajectory.shape[0])
    return trajectory[:, 0, :], trajectory[lanes, steps_taken, :]


def _batched_jax_trace_payloads(
    result,
    *,
    forget_exact_path: bool,
    event_context: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    res_phi_hits = _batched_jax_event_rows(result, event_context=event_context)
    res_tys = [
        np.stack([live[0], live[-1]], axis=0)
        if forget_exact_path and live.shape[0] >= 2
        else live
        for live in _batched_jax_live_rows(result)
    ]
    return res_tys, res_phi_hits


def _chunk_output_specs(
    result_type,
    axis_name: str,
    field_cache_spec: InterpolatedFieldCylCache | None = None,
):
    result_specs = result_type(
        trajectory=P(axis_name, None, None),
        mask=P(axis_name, None),
        steps_taken=P(axis_name),
        status=P(axis_name),
        t_final=P(axis_name),
        phi_hits=P(axis_name, None, None),
        phi_hits_count=P(axis_name),
    )
    state_specs = CartesianTracingContinuationState(
        trial_count=P(axis_name),
        accepted_count=P(axis_name),
        t=P(axis_name),
        y=P(axis_name, None),
        h=P(axis_name),
        k_first=P(axis_name, None),
        phi_last=P(axis_name),
        phi_initial=P(axis_name),
        status_event=P(axis_name),
        stopped=P(axis_name),
        no_progress=P(axis_name),
        field_cache=None
        if field_cache_spec is None
        else jax.tree.map(lambda _leaf: P(axis_name, None, None), field_cache_spec),
    )
    return result_specs, state_specs


def _trace_cartesian_chunks(
    spec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    mus: jax.Array,
    phis: jax.Array | None,
    stopping_criteria: tuple,
    field_state,
    trace_one: Callable,
    result_type,
    *,
    total_trials: int | None,
    forget_exact_path: bool,
    event_context: str,
    field_cache_spec: InterpolatedFieldCylCache | None = None,
):
    """Run fixed-shape core chunks and assemble exact public paths and events.

    ``total_trials`` is the caller's trial budget across all chunks. ``None``
    means unbounded, which is upstream's behaviour: a lane then ends only at
    ``tmax`` or on a stopping criterion. A finite budget ends an unfinished lane
    with status 1.
    """
    if total_trials is not None and total_trials <= 0:
        raise ValueError(f"max_steps must be positive, got {total_trials}")
    lane_count = int(y0s.shape[0])
    sharding = trajectory_batch_sharding_config(y0s)
    if sharding is not None:
        y0s, dtmaxs, mus = maybe_shard_trajectory_batch_inputs(
            y0s, dtmaxs, mus, config=sharding
        )
        phis = replicate_tree_on_mesh(phis, mesh=sharding.mesh)
        stopping_criteria = replicate_tree_on_mesh(
            stopping_criteria, mesh=sharding.mesh
        )
        field_state = replicate_tree_on_mesh(field_state, mesh=sharding.mesh)
    executors = {}

    def executor(chunk_trials: int, first_chunk: bool):
        key = (chunk_trials, first_chunk)
        if key in executors:
            return executors[key]
        chunk_spec = replace(spec, max_steps=chunk_trials)

        if first_chunk:

            def run_lanes(y0_block, dtmax_block, mu_block, phis_values,
                          criteria_values, field_values):
                return jax.vmap(
                    lambda y0, dtmax, mu: trace_one(
                        chunk_spec, y0, dtmax, mu, phis_values,
                        criteria_values, field_values, None
                    )
                )(y0_block, dtmax_block, mu_block)

            def run_shard(y0_block, dtmax_block, mu_block, phis_values,
                          criteria_values, field_values):
                return jax.lax.map(
                    lambda lane: trace_one(
                        chunk_spec, lane[0], lane[1], lane[2], phis_values,
                        criteria_values, field_values, None
                    ),
                    (y0_block, dtmax_block, mu_block),
                )

        else:

            def run_lanes(y0_block, dtmax_block, mu_block, phis_values,
                          criteria_values, field_values, state):
                return jax.vmap(
                    lambda y0, dtmax, mu, prior: trace_one(
                        chunk_spec, y0, dtmax, mu, phis_values,
                        criteria_values, field_values, prior
                    )
                )(y0_block, dtmax_block, mu_block, state)

            def run_shard(y0_block, dtmax_block, mu_block, phis_values,
                          criteria_values, field_values, state):
                return jax.lax.map(
                    lambda lane: trace_one(
                        chunk_spec, lane[0], lane[1], lane[2], phis_values,
                        criteria_values, field_values, lane[3],
                    ),
                    (y0_block, dtmax_block, mu_block, state),
                )

        if sharding is None:
            compiled = jax.jit(run_lanes)
        else:
            axis = sharding.axis_name
            field_spec = jax.tree.map(lambda _leaf: P(), field_state)
            criteria_spec = jax.tree.map(lambda _leaf: P(), stopping_criteria)
            inputs = (
                P(axis, None), P(axis), P(axis),
                P() if phis is not None else None,
                criteria_spec, field_spec,
            )
            outputs = _chunk_output_specs(result_type, axis, field_cache_spec)
            if not first_chunk:
                inputs += (outputs[1],)
            compiled = jax.shard_map(
                run_shard,
                mesh=sharding.mesh,
                in_specs=inputs,
                out_specs=outputs,
                check_vma=True,
            )
        executors[key] = compiled
        return compiled

    path_parts: list[list[np.ndarray]] = [[] for _ in range(lane_count)]
    hit_parts: list[list[np.ndarray]] = [[] for _ in range(lane_count)]
    first_rows: list[np.ndarray] = []
    last_rows: list[np.ndarray] = []
    active = np.ones(lane_count, dtype=bool)
    continuation = None
    result = None
    spent = 0
    while total_trials is None or spent < total_trials:
        chunk_trials = (
            _TRACING_CHUNK_TRIALS
            if total_trials is None
            else min(_TRACING_CHUNK_TRIALS, total_trials - spent)
        )
        spent += chunk_trials
        is_first = continuation is None
        run = executor(chunk_trials, is_first)
        if is_first:
            result, continuation = run(
                y0s, dtmaxs, mus, phis, stopping_criteria, field_state
            )
        else:
            result, continuation = run(
                y0s, dtmaxs, mus, phis, stopping_criteria,
                field_state, continuation
            )
        local_hits = _batched_jax_event_rows(result, event_context=event_context)
        if forget_exact_path:
            # Only the two rows the caller keeps cross the device boundary.
            device_first, device_last = _chunk_endpoint_rows(
                result.trajectory, result.steps_taken
            )
            chunk_first = _jax_trace_host_array(device_first, dtype=np.float64)
            chunk_last = _jax_trace_host_array(device_last, dtype=np.float64)
            local_paths = None
        else:
            chunk_first = chunk_last = None
            local_paths = _batched_jax_live_rows(result)
        statuses = _batched_trace_status_arrays(result)[0]
        for lane in np.flatnonzero(active):
            if local_paths is None:
                if is_first:
                    first_rows.append(chunk_first[lane])
                    last_rows.append(chunk_last[lane])
                else:
                    last_rows[lane] = chunk_last[lane]
            else:
                rows = local_paths[lane]
                if is_first:
                    first_rows.append(rows[0])
                    last_rows.append(rows[-1])
                    path_parts[lane].append(rows)
                else:
                    last_rows[lane] = rows[-1]
                    path_parts[lane].append(rows[1:])
            hit_parts[lane].append(local_hits[lane])
        # Only status 1 ("this call's max_steps ran out") continues. A lane that
        # reached ``tmax`` (0), fired a criterion (< 0) or whose step controller
        # stopped making progress (``TRACING_STATUS_STEP_CONTROL_FAILED``) is
        # terminal, so the loop drops it and exits when none is left.
        active = statuses == TRACING_STATUS_INCOMPLETE
        if not np.any(active):
            break

    accepted = _jax_trace_host_array(
        continuation.accepted_count, dtype=np.int64
    ).reshape(-1)
    if forget_exact_path:
        paths = [
            np.stack((first_rows[i], last_rows[i]))
            if accepted[i] > 0
            else first_rows[i][None, :]
            for i in range(lane_count)
        ]
    else:
        paths = [np.concatenate(parts, axis=0) for parts in path_parts]
    hits = [np.concatenate(parts, axis=0) for parts in hit_parts]
    summary = replace(
        result,
        steps_taken=continuation.accepted_count,
        t_final=continuation.t,
    )
    return paths, hits, summary


def _jax_trace_lost(status: int, t_final: float, tmax: float) -> bool:
    return status < 0 or t_final < float(tmax) - 1e-15


def _batched_trace_status_arrays(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _jax_trace_host_array(result.status, dtype=np.int32).reshape(-1),
        _jax_trace_host_array(result.t_final, dtype=np.float64).reshape(-1),
        _jax_trace_host_array(result.steps_taken, dtype=np.int32).reshape(-1),
    )


def _log_batched_jax_trace_statuses(
    result,
    *,
    first: int,
    total: int,
    tmax: float,
    label: str,
    live_trajectories: list[np.ndarray] | None = None,
    boozer_axis_status: bool = False,
) -> int:
    statuses, t_finals, steps_taken = _batched_trace_status_arrays(result)
    loss_ctr = 0
    live_items = (
        repeat(None, statuses.shape[0])
        if live_trajectories is None
        else live_trajectories
    )
    status_items = zip(live_items, statuses, t_finals, steps_taken, strict=True)
    for offset, (live, status, t_final, step_count) in enumerate(status_items):
        i = first + offset
        status_int = int(status)
        t_final_float = float(t_final)
        if status_int > 0:
            logger.warning(
                f"{i + 1:3d}/{total}, {label} status={status_int} "
                f"t_final={t_final_float}, steps_taken={int(step_count)}"
            )
        elif boozer_axis_status and status_int == TRACING_STATUS_BOOZER_AXIS:
            # ``-2`` on the two Boozer routes means the lane left the axis
            # (``s <= 0``), which collides with the ``-1 - i`` rule at ``i = 1``;
            # the axis test wins inside the driver, so both readings are
            # reported rather than the wrong one being asserted.
            logger.debug(
                f"{i + 1:3d}/{total}, {label} left the Boozer axis (s <= 0), "
                f"or criterion index 1 fired, t_final={t_final_float}"
            )
        elif status_int < 0:
            logger.debug(
                f"{i + 1:3d}/{total}, {label} stopped by criterion "
                f"index {-1 - status_int}, t_final={t_final_float}"
            )
        elif live is not None and live.shape[0] > 0:
            dtavg = live[-1, 0] / max(live.shape[0], 1)
            logger.debug(
                f"{i + 1:3d}/{total}, t_final={live[-1, 0]}, "
                f"average timestep {dtavg:.10f}s (JAX backend)"
            )
        if _jax_trace_lost(status_int, t_final_float, tmax):
            loss_ctr += 1
    return loss_ctr


def _require_jax_field_B(field: MagneticField) -> Callable[[object], object]:
    field_fn = getattr(field, "jax_B_at", None)
    if not callable(field_fn):
        raise TypeError(
            "JAX tracing requires a JAX-native MagneticField wrapper exposing "
            "`jax_B_at(point)`. Use a *JAX field class such as ToroidalFieldJAX, "
            "or call simsopt.field.tracing for the CPU path."
        )
    return field_fn


def _resolve_jax_field_B(
    field: MagneticField,
) -> tuple[Callable[..., object], object | None]:
    state_fn = getattr(field, "jax_B_at_state", None)
    state_getter = getattr(field, "jax_tracing_state", None)
    if callable(state_fn) and callable(state_getter):
        return state_fn, state_getter()
    return _require_jax_field_B(field), None


def _require_jax_field_B_dB(
    field: MagneticField,
) -> Callable[[object], tuple[object, object]]:
    field_fn = getattr(field, "jax_B_dB_at", None)
    if callable(field_fn):
        return field_fn
    grad_abs_fn = getattr(field, "jax_B_GradAbsB_at", None)
    if callable(grad_abs_fn):

        def _field_fn(point):
            B_raw, grad_abs_raw = grad_abs_fn(point)
            B = jnp.asarray(B_raw, dtype=jnp.float64).reshape((3,))
            grad_abs_B = jnp.asarray(grad_abs_raw, dtype=jnp.float64).reshape((3,))
            abs_B = jnp.linalg.norm(B)
            dB_by_dX = grad_abs_B[:, None] * B[None, :] / abs_B
            return B, dB_by_dX

        return _field_fn
    raise TypeError(
        "JAX particle tracing requires a JAX-native MagneticField wrapper "
        "exposing `jax_B_dB_at(point)` or `jax_B_GradAbsB_at(point)`. Use a "
        "*JAX field class such as ToroidalFieldJAX or InterpolatedFieldJAX, "
        "or call simsopt.field.tracing for the CPU path."
    )


def _resolve_jax_field_B_dB(
    field: MagneticField,
) -> tuple[Callable[..., tuple[object, object]], object | None]:
    state_fn = getattr(field, "jax_B_dB_at_state", None)
    state_getter = getattr(field, "jax_tracing_state", None)
    if callable(state_fn) and callable(state_getter):
        return state_fn, state_getter()
    state_fn = getattr(field, "jax_B_GradAbsB_at_state", None)
    if callable(state_fn) and callable(state_getter):

        def _state_field_fn(state, point):
            B_raw, grad_abs_raw = state_fn(state, point)
            B = jnp.asarray(B_raw, dtype=jnp.float64).reshape((3,))
            grad_abs_B = jnp.asarray(grad_abs_raw, dtype=jnp.float64).reshape((3,))
            abs_B = jnp.linalg.norm(B)
            dB_by_dX = grad_abs_B[:, None] * B[None, :] / abs_B
            return B, dB_by_dX

        return _state_field_fn, state_getter()
    return _require_jax_field_B_dB(field), None


def _resolve_jax_field_B_GradAbsB_cached(
    field: MagneticField,
) -> tuple[
    Callable[..., tuple[jax.Array, jax.Array, InterpolatedFieldCylCache]] | None,
    object | None,
    InterpolatedFieldCylCache | None,
]:
    """Resolve the field-with-an-output-buffer tracing contract, if it applies.

    An interpolated field does NOT return zero outside its interpolation
    domain: ``RegularGridInterpolant3D::evaluate_local``
    (``legacy native C++ source regular_grid_interpolant_3d_impl.h``) returns
    without writing its output slot, and the ``CachedTensor`` holding that slot
    (``legacy native C++ source cachedtensor.h``) is reused between queries, so
    the caller reads the PREVIOUS query's cylindrical value, re-flipped and
    re-rotated for the current point. Reproducing upstream's right-hand side
    there means carrying that buffer, which is what this route returns: a field
    function that threads it, plus the buffer's initial content.

    Returns ``(field_fn, device_state, initial_buffer)``, or three ``None`` for
    a field without such a buffer -- an analytic field writes every output row,
    so its right-hand side is stateless and the zero-fill question never
    arises. The initial buffer is zeros, the C++ ``CachedTensor``'s own initial
    content; upstream additionally primes the ``B`` half at the initial point
    (``particle_guiding_center_tracing`` reads ``AbsB_ref()`` there), which is
    unobservable because the first right-hand-side evaluation is at that same
    point and that point must be inside the domain for ``mu`` to be defined.
    """

    state_fn = getattr(field, "jax_B_GradAbsB_at_state", None)
    state_getter = getattr(field, "jax_tracing_state", None)
    if not (callable(state_fn) and callable(state_getter)):
        return None, None, None
    return (
        interpolated_field_state_B_GradAbsB_cached,
        state_getter(),
        interpolated_field_cyl_cache_zeros(),
    )


def trace_particles_boozer(
    field: BoozerRadialInterpolantJAX | InterpolatedBoozerFieldJAX,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    zetas=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    max_steps: int | None = None,
):
    """Trace Boozer-coordinate particles with the JAX tracing backend.

    ``max_steps`` is the trial limit of this route, defaulting to
    ``_BOOZER_GUIDING_CENTRE_STEP_LIMIT``. Unlike the Cartesian routes it cannot
    be unbounded: this route is not continued, so the limit is also the fixed
    trajectory buffer. Upstream (``simsoptpp/tracing.cpp`` ``solve``) has no
    limit at all, so a particle that exhausts this one is reported with
    status 1 and is never counted as reaching ``tmax``.
    """
    nparticles = stz_inits.shape[0]
    speed_par = _normalize_parallel_speeds(parallel_speeds, nparticles)
    m = mass
    speed_total = sqrt(2 * Ekin / m)
    mode = mode.lower()
    assert mode in ["gc", "gc_vac", "gc_nok"]
    return _trace_particles_boozer_jax(
        field,
        stz_inits,
        speed_par,
        speed_total,
        tmax=tmax,
        mass=m,
        charge=charge,
        tol=tol,
        comm=comm,
        zetas=zetas,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
        max_steps=max_steps,
    )


def _trace_particles_boozer_jax(
    field,
    stz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    zetas,
    stopping_criteria,
    mode,
    forget_exact_path,
    max_steps,
):
    """JAX backend for Boozer-coordinate guiding-center tracing."""
    if not isinstance(field, (BoozerRadialInterpolantJAX, InterpolatedBoozerFieldJAX)):
        raise NotImplementedError(
            "trace_particles_boozer JAX backend requires a "
            "BoozerRadialInterpolantJAX or InterpolatedBoozerFieldJAX field "
            f"instance; got {type(field).__name__}. Wrap the upstream "
            "BoozerRadialInterpolant via BoozerRadialInterpolantJAX(upstream), "
            "wrap the upstream InterpolatedBoozerField via "
            "InterpolatedBoozerFieldJAX(...), or call simsopt.field.tracing "
            "for the CPU path."
        )
    rhs_mode_map = {
        "gc_vac": "vacuum",
        "gc_nok": "no_k",
        "gc": "full",
    }
    if mode not in rhs_mode_map:
        raise NotImplementedError(
            "trace_particles_boozer JAX backend supports mode in "
            f"{set(rhs_mode_map)}; got mode={mode!r}. Full-orbit "
            "Boozer tracing is not implemented."
        )
    rhs_mode = rhs_mode_map[mode]

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(zetas) > 0:
        zetas_arr = _as_jax_float64(np.asarray(list(zetas), dtype=np.float64))
    else:
        zetas_arr = None

    nparticles = stz_inits.shape[0]
    step_limit = (
        _BOOZER_GUIDING_CENTRE_STEP_LIMIT if max_steps is None else int(max_steps)
    )
    if step_limit <= 0:
        raise ValueError(f"max_steps must be positive, got {step_limit}")
    max_phi_hits = 4096
    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    local_stz = np.asarray(stz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_stz.shape[0] > 0:
        field.set_points(local_stz)
        abs_B_initial = _jax_trace_host_array(field.modB(), dtype=np.float64).reshape(
            -1
        )
        G0 = _jax_trace_host_array(field.G(), dtype=np.float64).reshape(-1)
        mus = _magnetic_moments(float(speed_total), local_speed_par, abs_B_initial)
        dtmaxs = _boozer_particle_dtmaxs(G0, abs_B_initial, float(speed_total))
        spec = GuidingCenterTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=step_limit,
            max_phi_hits=max_phi_hits,
            adaptive_loop=_adaptive_loop_for_backend_mode(get_backend_mode()),
        )
        result = trace_guiding_centers_boozer_batched(
            spec,
            _as_jax_float64(np.column_stack([local_stz, local_speed_par])),
            _as_jax_float64(dtmaxs),
            _as_jax_float64(mus),
            field,
            m=float(mass),
            q=float(charge),
            mode=rhs_mode,
            zetas=zetas_arr,
            stopping_criteria=jax_stopping_criteria,
        )
        local_res_tys, local_res_zeta_hits = _batched_jax_trace_payloads(
            result,
            forget_exact_path=forget_exact_path,
            event_context="JAX Boozer guiding-centre tracing",
        )
        res_tys.extend(local_res_tys)
        res_zeta_hits.extend(local_res_zeta_hits)
        loss_ctr += _log_batched_jax_trace_statuses(
            result,
            first=first,
            total=nparticles,
            tmax=tmax,
            label="JAX Boozer guiding-centre",
            boozer_axis_status=True,
        )
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_zeta_hits = _allgather_flat(comm, res_zeta_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX Boozer backend)"
    )
    return res_tys, res_zeta_hits


def trace_particles(
    field: MagneticField,
    xyz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    phis=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    phase_angle=0,
    max_steps: int | None = None,
):
    """Trace particles with the JAX tracing backend.

    Mirrors ``simsopt.field.tracing.trace_particles``: returns
    ``(trajectories, phi_hits)``. Use :func:`trace_particles_with_status` when
    the caller also needs the per-particle terminal status.
    """
    trajectories, phi_hits, _statuses = _trace_particles_jax(
        field,
        xyz_inits,
        parallel_speeds,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
        phase_angle=phase_angle,
        max_steps=max_steps,
        collect_status=False,
    )
    return trajectories, phi_hits


def trace_particles_with_status(
    field: MagneticField,
    xyz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    phis=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    phase_angle=0,
    max_steps: int | None = None,
):
    """Trace particles and also return each particle's terminal core status.

    Same contract as :func:`trace_particles` plus a third return value: the
    integer status per particle, ``0`` for reaching ``tmax``, ``-1 - i`` for a
    stop by ``stopping_criteria[i]``, ``1`` for a run that ended on
    ``max_steps`` without either, and ``2`` for a run this port ended itself:
    the step controller stopped making progress, a trial upstream would have
    accepted landed on a non-finite state, or an accepted step crossed a plane
    while its own dense output was non-finite (the full vocabulary is in the
    module docstring of :mod:`simsopt_jax.core.tracing`). For the
    guiding-centre modes ``max_steps=None``
    is unbounded, matching upstream; the un-continued ``'full'`` route falls back
    to ``_FULLORBIT_STEP_LIMIT``.
    """
    return _trace_particles_jax(
        field,
        xyz_inits,
        parallel_speeds,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
        phase_angle=phase_angle,
        max_steps=max_steps,
        collect_status=True,
    )


def _trace_particles_jax(
    field,
    xyz_inits,
    parallel_speeds,
    *,
    tmax,
    mass,
    charge,
    Ekin,
    tol,
    comm,
    phis,
    stopping_criteria,
    mode,
    forget_exact_path,
    phase_angle,
    max_steps,
    collect_status,
):
    """Shared particle implementation of the two public entry points.

    ``collect_status`` exists only so that :func:`trace_particles`, which mirrors
    upstream's two-value contract, also mirrors its MPI traffic: it gathers the
    trajectories and the event rows and nothing else.
    """
    mode = mode.lower()
    assert mode in ["gc", "gc_vac", "full"]
    speed_par = _normalize_parallel_speeds(parallel_speeds, xyz_inits.shape[0])
    speed_total = sqrt(2 * Ekin / mass)
    if mode == "full":
        return _trace_particles_jax_fullorbit_vacuum(
            field,
            xyz_inits,
            speed_par,
            speed_total,
            tmax=tmax,
            mass=mass,
            charge=charge,
            tol=tol,
            comm=comm,
            phis=phis,
            stopping_criteria=stopping_criteria,
            forget_exact_path=forget_exact_path,
            phase_angle=phase_angle,
            max_steps=max_steps,
            collect_status=collect_status,
        )
    return _trace_particles_jax_guiding_center_vacuum(
        field,
        xyz_inits,
        speed_par,
        speed_total,
        tmax=tmax,
        mass=mass,
        charge=charge,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
        max_steps=max_steps,
        collect_status=collect_status,
    )


def _trace_particles_jax_guiding_center_vacuum(
    field,
    xyz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    phis,
    stopping_criteria,
    mode,
    forget_exact_path,
    max_steps,
    collect_status,
):
    """JAX backend for Cartesian vacuum guiding-center tracing."""
    if mode != "gc_vac":
        raise NotImplementedError(
            "trace_particles JAX guiding-centre helper currently only "
            f"supports mode='gc_vac' (got mode={mode!r}). The non-vacuum "
            "guiding-centre mode ('gc') is a deferred follow-up. The "
            "'full' mode is routed separately to "
            "_trace_particles_jax_fullorbit_vacuum from the public "
            "trace_particles wrapper."
        )
    field_fn, field_state = _resolve_jax_field_B_dB(field)
    # Upstream's guiding-centre right-hand side reads ``GradAbsB`` from the
    # field and keeps the field's output buffer between calls; when the field
    # has such a buffer this route mirrors both (see
    # ``_resolve_jax_field_B_GradAbsB_cached``).
    (
        cached_field_fn,
        cached_field_state,
        field_cache_init,
    ) = _resolve_jax_field_B_GradAbsB_cached(field)

    jax_stopping_criteria = jax.tree.map(
        _stage_stopping_criterion_leaf,
        _translate_stopping_criteria_to_jax(stopping_criteria),
    )
    if len(phis) > 0:
        phis_arr = _as_jax_float64(np.asarray(list(phis), dtype=np.float64))
    else:
        phis_arr = None

    nparticles = xyz_inits.shape[0]
    max_phi_hits = 4096
    res_tys = []
    res_phi_hits = []
    statuses = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    local_xyz = np.asarray(xyz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_xyz.shape[0] > 0:
        local_xyz_device = _as_jax_float64(local_xyz)
        if field_state is None:
            B_initial, _ = jax.vmap(field_fn)(local_xyz_device)
        else:
            B_initial, _ = jax.vmap(lambda point: field_fn(field_state, point))(
                local_xyz_device
            )
        abs_B_initial = np.linalg.norm(
            _jax_trace_host_array(B_initial, dtype=np.float64), axis=1
        )
        mus = _magnetic_moments(float(speed_total), local_speed_par, abs_B_initial)
        dtmaxs = _cartesian_particle_dtmaxs(local_xyz, float(speed_total))
        spec = GuidingCenterTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=_TRACING_CHUNK_TRIALS,
            max_phi_hits=max_phi_hits,
            adaptive_loop=_CHUNKED_ADAPTIVE_LOOP,
        )
        y0s = np.column_stack([local_xyz, local_speed_par])
        def trace_one(chunk_spec, y0, dtmax, mu, phis_values,
                      criteria_values, current_field_state, prior):
            if field_cache_init is not None:
                field_at = lambda point, cache: cached_field_fn(
                    current_field_state, point, cache
                )
            elif current_field_state is None:
                field_at = field_fn
            else:
                field_at = lambda point: field_fn(current_field_state, point)
            return _trace_guiding_center_chunk(
                replace(chunk_spec, dtmax=dtmax), y0, field_at,
                m=float(mass), q=float(charge), mu=mu, phis=phis_values,
                stopping_criteria=criteria_values, continuation=prior,
                field_cache_init=field_cache_init,
            )

        local_res_tys, local_res_phi_hits, summary = _trace_cartesian_chunks(
            spec,
            _as_jax_float64(y0s),
            _as_jax_float64(dtmaxs),
            _as_jax_float64(mus),
            phis_arr,
            jax_stopping_criteria,
            field_state if field_cache_init is None else cached_field_state,
            trace_one,
            GuidingCenterTracingResult,
            total_trials=None if max_steps is None else int(max_steps),
            forget_exact_path=forget_exact_path,
            event_context="JAX guiding-centre tracing",
            field_cache_spec=field_cache_init,
        )
        res_tys.extend(local_res_tys)
        res_phi_hits.extend(local_res_phi_hits)
        statuses.extend(
            int(status) for status in _batched_trace_status_arrays(summary)[0]
        )
        loss_ctr += _log_batched_jax_trace_statuses(
            summary,
            first=first,
            total=nparticles,
            tmax=tmax,
            label="JAX guiding-centre",
        )
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    if collect_status:
        statuses = _allgather_flat(comm, statuses)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX backend)"
    )
    return res_tys, res_phi_hits, np.asarray(statuses, dtype=np.int64)


def _trace_particles_jax_fullorbit_vacuum(
    field,
    xyz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    phis,
    stopping_criteria,
    forget_exact_path,
    phase_angle,
    max_steps,
    collect_status,
):
    """JAX backend for Cartesian full-orbit tracing."""
    field_fn, field_state = _resolve_jax_field_B(field)
    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = _as_jax_float64(np.asarray(list(phis), dtype=np.float64))
    else:
        phis_arr = None

    step_limit = _FULLORBIT_STEP_LIMIT if max_steps is None else int(max_steps)
    if step_limit <= 0:
        raise ValueError(f"max_steps must be positive, got {step_limit}")
    max_phi_hits = 4096
    nparticles = xyz_inits.shape[0]
    first, last = parallel_loop_bounds(comm, nparticles)
    local_xyz = np.asarray(xyz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_xyz.shape[0] > 0:
        xyz_inits_full, v_inits, _ = gc_to_fullorbit_initial_guesses(
            field,
            local_xyz,
            local_speed_par,
            float(speed_total),
            float(mass),
            float(charge),
            eta=float(phase_angle),
        )
    else:
        xyz_inits_full = np.zeros((0, 3), dtype=np.float64)
        v_inits = np.zeros((0, 3), dtype=np.float64)
    res_tys = []
    res_phi_hits = []
    statuses = []
    loss_ctr = 0
    if xyz_inits_full.shape[0] > 0:
        speeds = np.linalg.norm(v_inits, axis=1)
        dtmaxs = _quarter_turn_dtmaxs(_cartesian_radii(xyz_inits_full), speeds)
        spec = FullorbitTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=step_limit,
            max_phi_hits=max_phi_hits,
            adaptive_loop=_adaptive_loop_for_backend_mode(get_backend_mode()),
        )
        result = trace_fullorbits_batched(
            spec,
            _as_jax_float64(np.column_stack([xyz_inits_full, v_inits])),
            _as_jax_float64(dtmaxs),
            field_fn,
            m=float(mass),
            q=float(charge),
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
            magnetic_field_state=field_state,
        )
        local_res_tys, local_res_phi_hits = _batched_jax_trace_payloads(
            result,
            forget_exact_path=forget_exact_path,
            event_context="JAX full-orbit tracing",
        )
        res_tys.extend(local_res_tys)
        res_phi_hits.extend(local_res_phi_hits)
        statuses.extend(
            int(status) for status in _batched_trace_status_arrays(result)[0]
        )
        loss_ctr += _log_batched_jax_trace_statuses(
            result,
            first=first,
            total=nparticles,
            tmax=tmax,
            label="JAX full-orbit",
        )
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    if collect_status:
        statuses = _allgather_flat(comm, statuses)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX full-orbit backend)"
    )
    return res_tys, res_phi_hits, np.asarray(statuses, dtype=np.int64)


def compute_fieldlines(
    field,
    R0,
    Z0,
    tmax=200,
    tol=1e-7,
    phis=[],
    stopping_criteria=[],
    comm=None,
    max_steps: int | None = None,
):
    """Compute fieldlines with the JAX tracing backend.

    Mirrors ``simsopt.field.tracing.compute_fieldlines``: returns
    ``(trajectories, phi_hits)``. Use :func:`compute_fieldlines_with_status`
    when the caller also needs the per-line terminal status.
    """
    trajectories, phi_hits, _statuses = _compute_fieldlines_jax(
        field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
        comm=comm,
        max_steps=max_steps,
        collect_status=False,
    )
    return trajectories, phi_hits


def _translate_stopping_criteria_to_jax(stopping_criteria: list) -> tuple:
    """Translate CPU stopping criteria into JAX-side dataclasses."""

    def _attr(obj, name):
        if not hasattr(obj, name):
            raise NotImplementedError(
                f"JAX tracing path cannot read attribute '{name}' on "
                f"{type(obj).__name__}; use the simsopt.field.tracing "
                "Python wrapper class rather than the bound C++ "
                "``simsoptpp.*`` class directly."
            )
        return getattr(obj, name)

    translated = []
    for crit in stopping_criteria:
        if isinstance(crit, sopp.LevelsetStoppingCriterion):
            classifier_obj = levelset_classifier_for(crit) or getattr(
                crit, "classifier", None
            )
            if (
                classifier_obj is not None
                and not supports_levelset_classifier_conversion(classifier_obj)
            ):
                classifier_obj = levelset_classifier_from_interpolant(classifier_obj)
            if classifier_obj is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.LevelsetStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.LevelsetStoppingCriterion("
                    "SurfaceClassifier(surface)) so the JAX-side interpolant "
                    "spec can be rebuilt from the classifier grid metadata."
                )
            if not supports_levelset_classifier_conversion(classifier_obj):
                raise NotImplementedError(
                    "JAX tracing path requires a SurfaceClassifier-derived "
                    "interpolant as the LevelsetStoppingCriterion argument; "
                    f"received {type(classifier_obj).__name__}."
                )
            jax_classifier_fn = levelset_classifier_fn_from_surface_classifier(
                classifier_obj
            )
            translated.append(
                JaxLevelsetStoppingCriterion(classifier_fn=jax_classifier_fn)
            )
            continue
        if isinstance(crit, sopp.MinRStoppingCriterion):
            translated.append(
                JaxMinRStoppingCriterion(crit_r=float(_attr(crit, "crit_r")))
            )
            continue
        if isinstance(crit, sopp.MaxRStoppingCriterion):
            translated.append(
                JaxMaxRStoppingCriterion(crit_r=float(_attr(crit, "crit_r")))
            )
            continue
        if isinstance(crit, sopp.MinZStoppingCriterion):
            translated.append(
                JaxMinZStoppingCriterion(crit_z=float(_attr(crit, "crit_z")))
            )
            continue
        if isinstance(crit, sopp.MaxZStoppingCriterion):
            translated.append(
                JaxMaxZStoppingCriterion(crit_z=float(_attr(crit, "crit_z")))
            )
            continue
        if isinstance(crit, sopp.ToroidalTransitStoppingCriterion):
            if bool(_attr(crit, "flux")):
                raise NotImplementedError(
                    "JAX tracing path supports "
                    "ToroidalTransitStoppingCriterion only with flux=False; "
                    "flux=True is a flux-coordinate transit criterion and is "
                    "not implemented by the Cartesian JAX tracing route."
                )
            translated.append(
                JaxToroidalTransitStoppingCriterion(
                    max_transits=float(_attr(crit, "max_transits"))
                )
            )
            continue
        if isinstance(crit, sopp.IterationStoppingCriterion):
            translated.append(
                JaxIterStoppingCriterion(max_iter=int(_attr(crit, "max_iter")))
            )
            continue
        if isinstance(crit, sopp.MinToroidalFluxStoppingCriterion):
            min_s = getattr(crit, "min_s", None)
            if min_s is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.MinToroidalFluxStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.MinToroidalFluxStoppingCriterion."
                )
            translated.append(JaxMinToroidalFluxStoppingCriterion(min_s=float(min_s)))
            continue
        if isinstance(crit, sopp.MaxToroidalFluxStoppingCriterion):
            max_s = getattr(crit, "max_s", None)
            if max_s is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.MaxToroidalFluxStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.MaxToroidalFluxStoppingCriterion."
                )
            translated.append(JaxMaxToroidalFluxStoppingCriterion(max_s=float(max_s)))
            continue
        raise NotImplementedError(
            "JAX tracing path cannot translate stopping criterion of type "
            f"{type(crit).__name__}; supported classes are LevelsetStoppingCriterion, "
            "MinR/MaxR/MinZ/MaxZ/ToroidalTransit/Iteration/Min/MaxToroidalFlux."
        )
    return tuple(translated)


def compute_fieldlines_with_status(
    field,
    R0,
    Z0,
    tmax=200,
    tol=1e-7,
    phis=[],
    stopping_criteria=[],
    comm=None,
    max_steps: int | None = None,
):
    """Compute fieldlines and also return each line's terminal core status.

    Same contract as :func:`compute_fieldlines` plus a third return value: the
    integer status per line, ``0`` for reaching ``tmax``, ``-1 - i`` for a stop
    by ``stopping_criteria[i]``, ``1`` for a line that ended on ``max_steps``
    without either, and ``2`` for a line this port ended itself: the step
    controller stopped making progress, a trial upstream would have accepted
    landed on a non-finite state, or an accepted step crossed a plane while its
    own dense output was non-finite (the full vocabulary is in the module
    docstring of :mod:`simsopt_jax.core.tracing`). ``max_steps=None`` is
    unbounded, which is what upstream's
    ``compute_fieldlines`` does -- it has no trial limit and no such parameter.
    """
    return _compute_fieldlines_jax(
        field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
        comm=comm,
        max_steps=max_steps,
        collect_status=True,
    )


def _compute_fieldlines_jax(
    field,
    R0,
    Z0,
    *,
    tmax,
    tol,
    phis,
    stopping_criteria,
    comm,
    max_steps,
    collect_status,
):
    """Shared field-line implementation of the two public entry points.

    ``collect_status`` exists only so that :func:`compute_fieldlines`, which
    mirrors upstream's two-value contract, also mirrors its MPI traffic: it
    gathers the trajectories and the event rows and nothing else.
    """
    assert len(R0) == len(Z0)
    field_fn, field_state = _resolve_jax_field_B(field)

    jax_stopping_criteria = jax.tree.map(
        _stage_stopping_criterion_leaf,
        _translate_stopping_criteria_to_jax(stopping_criteria),
    )
    if len(phis) > 0:
        phis_arr = _as_jax_float64(np.asarray(list(phis), dtype=np.float64))
    else:
        phis_arr = None

    max_phi_hits = 4096
    nlines = len(R0)
    res_tys = []
    res_phi_hits = []
    statuses = []
    R0_arr = np.asarray(R0, dtype=np.float64)
    Z0_arr = np.asarray(Z0, dtype=np.float64)
    first, last = parallel_loop_bounds(comm, nlines)
    local_y0 = np.stack(
        [
            R0_arr[first:last],
            np.zeros(last - first, dtype=np.float64),
            Z0_arr[first:last],
        ],
        axis=1,
    )
    if local_y0.shape[0] > 0:
        local_y0_device = _as_jax_float64(local_y0)
        if field_state is None:
            B_initial = jax.vmap(field_fn)(local_y0_device)
        else:
            B_initial = jax.vmap(lambda point: field_fn(field_state, point))(
                local_y0_device
            )
        abs_B_initial = np.linalg.norm(
            _jax_trace_host_array(B_initial, dtype=np.float64), axis=1
        )
        dtmaxs = _fieldline_dtmaxs(local_y0, abs_B_initial)
        spec = FieldlineTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=_TRACING_CHUNK_TRIALS,
            max_phi_hits=max_phi_hits,
            adaptive_loop=_CHUNKED_ADAPTIVE_LOOP,
        )
        def trace_one(chunk_spec, y0, dtmax, _mu, phis_values,
                      criteria_values, current_field_state, prior):
            if current_field_state is None:
                field_at = field_fn
            else:
                field_at = lambda point: field_fn(current_field_state, point)
            return _trace_fieldline_chunk(
                replace(chunk_spec, dtmax=dtmax), y0, field_at,
                phis=phis_values, stopping_criteria=criteria_values,
                continuation=prior,
            )

        local_res_tys, local_res_phi_hits, summary = _trace_cartesian_chunks(
            spec,
            local_y0_device,
            _as_jax_float64(dtmaxs),
            _as_jax_float64(np.zeros(local_y0.shape[0], dtype=np.float64)),
            phis_arr,
            jax_stopping_criteria,
            field_state,
            trace_one,
            FieldlineTracingResult,
            total_trials=None if max_steps is None else int(max_steps),
            forget_exact_path=False,
            event_context="JAX fieldline tracing",
        )
        res_tys.extend(local_res_tys)
        res_phi_hits.extend(local_res_phi_hits)
        statuses.extend(
            int(status) for status in _batched_trace_status_arrays(summary)[0]
        )
        _log_batched_jax_trace_statuses(
            summary,
            first=first,
            total=nlines,
            tmax=tmax,
            label="JAX fieldline",
            live_trajectories=local_res_tys,
        )
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    if collect_status:
        statuses = _allgather_flat(comm, statuses)
    return res_tys, res_phi_hits, np.asarray(statuses, dtype=np.int64)
