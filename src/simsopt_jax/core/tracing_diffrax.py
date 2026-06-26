"""Diffrax-backed tracing drivers (``integrator="dopri5_diffrax"``).

This is an *alternative backend* for :func:`simsopt_jax.core.tracing.trace_fieldline`,
not a replacement. It delegates the adaptive Dormand-Prince RK4(5) stepping to `diffrax`
(Patrick Kidger's vetted JAX ODE library), while reusing the in-repo crossing/stopping/
output machinery so the returned :class:`~simsopt_jax.core.tracing.FieldlineTracingResult`
conforms to the same contract as the hand-rolled ``dopri5_native`` path.

Division of labour
==================
- **diffrax** owns: the adaptive ``Dopri5`` step sequence (``SaveAt(steps=True)`` gives
  the per-step ``(t, y)`` grid, padded to ``max_steps + 1`` with non-finite rows).
- **native helpers** (imported from :mod:`simsopt_jax.core.tracing`) own: φ-plane
  crossing localization (``_scan_angle_plane_events`` + the same 5th-order ``dopri5_step``
  sub-step localizer used by ``trace_fieldline``), the event-row buffer layout
  (``_append_event_row``), the ``status`` taxonomy, and trajectory padding.

Parity with ``dopri5_native``
============================
- **Integration path (no criterion fires / reaches ``tmax``):** the backends agree to
  integrator tolerance (~1e-8). Same Dopri5 tableau; φ-plane punctures match because
  the crossing localizer is the shared native ``dopri5_step`` sub-step.
- **Stopping-criterion exit:** criteria are evaluated in a post-solve pass over accepted
  diffrax steps. The firing step is recorded as an ``idx < 0`` event row and used as
  ``t_final``, but is not a live trajectory row, matching the native result contract.

The post-solve event pass is the single source of truth for the result contract: diffrax
produces accepted DOPRI5 states, then the native JAX crossing/stopping helpers classify
φ/zeta hits, transit/flux criteria, status, and padding. ``IterStoppingCriterion`` stays
on the native fallback because its contract counts rejected adaptive attempts, which are
not present in diffrax's accepted-step save grid. This keeps the native ``dopri5_native``
path selectable as the parity reference while letting diffrax own the adaptive step
sequence.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import diffrax

from ._device_scalars import two_pi as _device_two_pi
from .tracing import (
    FullorbitTracingResult,
    FullorbitTracingSpec,
    FieldlineTracingResult,
    FieldlineTracingSpec,
    GuidingCenterTracingResult,
    GuidingCenterTracingSpec,
    LevelsetStoppingCriterion,
    MaxRStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    _BOOZER_AXIS_STATUS,
    _FIELDLINE_INITIAL_STEP_FRACTION,
    _PARTICLE_INITIAL_STEP_FRACTION,
    _append_event_row,
    _as_device_array,
    _boozer_axis_invalid,
    _continuous_angle,
    _continuous_phi,
    _continuous_phi_from_state,
    _device_array,
    _device_false,
    _device_index,
    _device_zeros,
    _event_row_from_state,
    _initial_step_size,
    _scan_angle_plane_events,
    _stopping_criterion_should_stop,
    dopri5_step,
    fieldline_rhs,
    fullorbit_vacuum_rhs,
    guiding_center_boozer_rhs,
    guiding_center_no_k_boozer_rhs,
    guiding_center_vacuum_boozer_rhs,
    guiding_center_vacuum_rhs,
)

__all__ = [
    "trace_fieldline_diffrax",
    "trace_fullorbit_diffrax",
    "trace_guiding_center_boozer_diffrax",
    "trace_guiding_center_diffrax",
]

# Stopping criteria accepted by the diffrax backend. They are evaluated in the
# post-solve pass with the shared native predicate, so Cartesian routes preserve
# native no-op behavior for Boozer/flux-coordinate criteria.
_SUPPORTED_DIFFRAX_CRITERIA = (
    MinRStoppingCriterion,
    MaxRStoppingCriterion,
    MinZStoppingCriterion,
    MaxZStoppingCriterion,
    LevelsetStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
)


def _validate_criteria(stopping_criteria: tuple) -> None:
    for criterion in stopping_criteria:
        if not isinstance(criterion, _SUPPORTED_DIFFRAX_CRITERIA):
            raise NotImplementedError(
                "The diffrax tracing backend (integrator='dopri5_diffrax') supports "
                "only the JAX tracing stopping criteria "
                f"{tuple(c.__name__ for c in _SUPPORTED_DIFFRAX_CRITERIA)}; got "
                f"{type(criterion).__name__}."
            )


def _raw_dtmax_arg(raw_dtmax, dtmax):
    if isinstance(raw_dtmax, (int, float, np.floating)) and not np.isfinite(raw_dtmax):
        return None
    return dtmax


def _solve_diffrax_grid(
    *,
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    y0: jax.Array,
    tmax: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
    dtmax: jax.Array,
    raw_dtmax,
    h0: jax.Array,
    max_steps: int,
):
    term = diffrax.ODETerm(lambda _t, y, _args: rhs(_t, y))
    controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmax=_raw_dtmax_arg(raw_dtmax, dtmax),
    )
    return diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=_device_array(0.0, y0.dtype),
        t1=tmax,
        dt0=h0,
        y0=y0,
        saveat=diffrax.SaveAt(t0=True, steps=True),
        stepsize_controller=controller,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
        throw=False,
    )


def _postprocess_diffrax_grid(
    *,
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    ts: jax.Array,
    ys: jax.Array,
    finite: jax.Array,
    y0: jax.Array,
    tmax: jax.Array,
    max_steps: int,
    max_root_iters: int,
    max_hits: int,
    angle_targets: jax.Array | None,
    stopping_criteria: tuple,
    angle_initial: jax.Array,
    angle_at_state: Callable[[jax.Array, jax.Array], jax.Array],
    dtype,
    is_boozer_state: bool = False,
    stop_on_boozer_axis: bool = False,
):
    """Convert a diffrax accepted-step grid into the native tracing result contract."""

    state_dim = int(y0.shape[0])
    event_width = state_dim + 2
    max_hits_i32 = _device_index(max_hits)
    if angle_targets is None:
        targets = _device_zeros((0,), dtype)
    else:
        targets = _as_device_array(angle_targets, dtype).reshape((-1,))
    num_targets = int(targets.shape[0])

    hits0 = _device_zeros((max_hits, event_width), dtype)
    count0 = _device_index(0)
    stop_found0 = _device_false()
    stop_steps0 = _device_index(0)
    stop_status0 = _device_index(0)
    t_stop0 = _device_array(0.0, dtype)
    y_stop0 = y0
    two_pi = _device_two_pi(_device_array(1.0, dtype))

    def scan_segment(carry, i):
        (
            hits,
            count,
            angle_last,
            angle_init,
            stop_found,
            stop_steps,
            stop_status,
            t_stop,
            y_stop,
        ) = carry
        next_i = i + _device_index(1)
        seg_finite = jnp.logical_and(finite[i], finite[next_i])
        active = jnp.logical_and(seg_finite, jnp.logical_not(stop_found))
        t_i = ts[i]
        h = ts[next_i] - t_i
        y_i = ys[i]
        y_next = ys[next_i]
        t_next = ts[next_i]

        axis_invalid = (
            _boozer_axis_invalid(y_next) if stop_on_boozer_axis else _device_false()
        )
        event_active = jnp.logical_and(active, jnp.logical_not(axis_invalid))
        angle_current = angle_at_state(y_next, angle_last)

        def scan_angles(args):
            hits_in, count_in = args
            k_first = rhs(t_i, y_i)

            def state_at_fraction(s):
                y_sub, _err, _k7 = dopri5_step(rhs, t_i, y_i, s * h, k_first)
                return y_sub

            return _scan_angle_plane_events(
                hits=hits_in,
                count=count_in,
                angle_last=angle_last,
                angle_current=angle_current,
                targets=targets,
                num_targets=num_targets,
                two_pi=two_pi,
                dtype=dtype,
                t=t_i,
                h_clamped=h,
                max_root_iters=max_root_iters,
                max_hits_i32=max_hits_i32,
                state_at_fraction=state_at_fraction,
                angle_at_state=angle_at_state,
            )

        hits_after_angles, count_after_angles = jax.lax.cond(
            event_active,
            scan_angles,
            lambda args: (args[0], args[1]),
            operand=(hits, count),
        )

        first_accepted_step = i == _device_index(0)
        angle_init_for_criteria = jnp.where(first_accepted_step, angle_current, angle_init)

        def apply_criteria(args):
            hits_in, count_in = args
            return _apply_stopping_criteria_events_diffrax(
                stopping_criteria=stopping_criteria,
                hits=hits_in,
                count=count_in,
                iter_count=next_i,
                angle_current=angle_current,
                angle_initial=angle_init_for_criteria,
                t_event=t_next,
                state=y_next,
                dtype=dtype,
                max_hits_i32=max_hits_i32,
                is_boozer_state=is_boozer_state,
            )

        (
            hits_after_criteria,
            count_after_criteria,
            criteria_status,
            criteria_stop,
        ) = jax.lax.cond(
            event_active,
            apply_criteria,
            lambda args: (args[0], args[1], _device_index(0), _device_false()),
            operand=(hits_after_angles, count_after_angles),
        )

        stop_this_segment = jnp.logical_or(
            jnp.logical_and(active, axis_invalid),
            jnp.logical_and(event_active, criteria_stop),
        )
        status_this_segment = jnp.where(
            axis_invalid,
            _device_index(_BOOZER_AXIS_STATUS),
            criteria_status,
        )
        take_stop = jnp.logical_and(jnp.logical_not(stop_found), stop_this_segment)
        angle_last_next = jnp.where(active, angle_current, angle_last)
        angle_init_next = jnp.where(
            jnp.logical_and(event_active, first_accepted_step),
            angle_current,
            angle_init,
        )
        return (
            hits_after_criteria,
            count_after_criteria,
            angle_last_next,
            angle_init_next,
            jnp.logical_or(stop_found, stop_this_segment),
            jnp.where(take_stop, i, stop_steps),
            jnp.where(take_stop, status_this_segment, stop_status),
            jnp.where(take_stop, t_next, t_stop),
            jnp.where(take_stop, y_next, y_stop),
        ), None

    (
        hits_final,
        count_final,
        _angle_last_final,
        _angle_init_final,
        stop_found,
        stop_steps,
        stop_status,
        t_stop,
        y_stop,
    ), _ = jax.lax.scan(
        scan_segment,
        (
            hits0,
            count0,
            angle_initial,
            angle_initial,
            stop_found0,
            stop_steps0,
            stop_status0,
            t_stop0,
            y_stop0,
        ),
        jnp.arange(max_steps, dtype=jnp.int32),
    )

    finite_count = jnp.sum(finite.astype(jnp.int32))
    last_idx = (finite_count - _device_index(1)).astype(jnp.int32)
    t_end = ts[last_idx]
    y_end = ys[last_idx]
    eps_t = _device_array(1.0e-12, dtype) * jnp.maximum(
        jnp.abs(tmax), _device_array(1.0, dtype)
    )
    reached = (tmax - t_end) <= eps_t
    status_normal = jnp.where(reached, _device_index(0), _device_index(1))
    steps_taken = jnp.where(stop_found, stop_steps, last_idx)
    status = jnp.where(stop_found, stop_status, status_normal)
    t_final = jnp.where(stop_found, t_stop, t_end)
    y_final = jnp.where(stop_found, y_stop, y_end)

    mask_indices = jax.lax.iota(jnp.int32, max_steps + 1)
    mask_stop = mask_indices <= steps_taken
    mask = jnp.where(stop_found, mask_stop, finite)

    ts_filled = jnp.where(finite, ts, t_end)
    ys_filled = jnp.where(finite[:, None], ys, jnp.broadcast_to(y_end, ys.shape))
    trajectory_raw = jnp.concatenate([ts_filled[:, None], ys_filled], axis=1)
    last_row = jnp.concatenate([jnp.reshape(t_final, (1,)), y_final.reshape((state_dim,))])
    trajectory = jnp.where(
        mask[:, None],
        trajectory_raw,
        jnp.broadcast_to(last_row, trajectory_raw.shape),
    )

    if stop_on_boozer_axis:
        initial_axis_invalid = _boozer_axis_invalid(y0)
        seed_row = jnp.concatenate(
            [_device_zeros((1,), dtype), y0.reshape((state_dim,))]
        )
        seed_trajectory = jnp.broadcast_to(seed_row, trajectory.shape)
        seed_mask = mask_indices == _device_index(0)
        trajectory = jnp.where(initial_axis_invalid, seed_trajectory, trajectory)
        mask = jnp.where(initial_axis_invalid, seed_mask, mask)
        steps_taken = jnp.where(initial_axis_invalid, _device_index(0), steps_taken)
        status = jnp.where(
            initial_axis_invalid, _device_index(_BOOZER_AXIS_STATUS), status
        )
        t_final = jnp.where(initial_axis_invalid, _device_array(0.0, dtype), t_final)
        hits_final = jnp.where(initial_axis_invalid, hits0, hits_final)
        count_final = jnp.where(initial_axis_invalid, count0, count_final)

    return trajectory, mask, steps_taken, status, t_final, hits_final, count_final


def _apply_stopping_criteria_events_diffrax(
    *,
    stopping_criteria: tuple,
    hits: jax.Array,
    count: jax.Array,
    iter_count: jax.Array,
    angle_current: jax.Array,
    angle_initial: jax.Array,
    t_event: jax.Array,
    state: jax.Array,
    dtype,
    max_hits_i32: jax.Array,
    is_boozer_state: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    status = _device_index(0)
    stop = _device_false()
    for i, criterion in enumerate(stopping_criteria):
        pred = _stopping_criterion_should_stop(
            criterion,
            state[0],
            state[1],
            state[2],
            iter_count,
            angle_current,
            angle_initial,
            dtype,
            is_boozer_state=is_boozer_state,
        )
        fires = jnp.logical_and(jnp.logical_not(stop), pred)
        hit_row = _event_row_from_state(
            t_event,
            _device_array(float(-1 - i), dtype),
            state,
        )
        hits, count = _append_event_row(hits, count, fires, hit_row, max_hits_i32)
        status = jnp.where(fires, _device_index(-1 - i), status)
        stop = jnp.logical_or(stop, fires)
    return hits, count, status, stop


def trace_fieldline_diffrax(
    spec: FieldlineTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FieldlineTracingResult:
    """Trace one fieldline with diffrax's adaptive ``Dopri5``; same contract as
    :func:`simsopt_jax.core.tracing.trace_fieldline`.

    See the module docstring for the diffrax/native division of labour. ``spec``,
    ``y0``, ``magnetic_field_fn``, ``phis``, ``stopping_criteria`` and the returned
    :class:`~simsopt_jax.core.tracing.FieldlineTracingResult` are identical in meaning
    to the native driver; only the integration engine differs.
    """

    stopping_criteria = tuple(stopping_criteria)
    _validate_criteria(stopping_criteria)

    dtype = jnp.float64
    y0_arr = _as_device_array(y0, dtype).reshape((3,))
    tmax = _as_device_array(spec.tmax, dtype)
    rtol = _as_device_array(spec.rtol, dtype)
    atol = _as_device_array(spec.atol, dtype)
    dtmax = _as_device_array(spec.dtmax, dtype)
    t0 = _device_array(0.0, dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    rhs = fieldline_rhs(magnetic_field_fn)
    h0 = _initial_step_size(t0, tmax, dtmax, _FIELDLINE_INITIAL_STEP_FRACTION)
    sol = _solve_diffrax_grid(
        rhs=rhs,
        y0=y0_arr,
        tmax=tmax,
        rtol=rtol,
        atol=atol,
        dtmax=dtmax,
        raw_dtmax=spec.dtmax,
        h0=h0,
        max_steps=max_steps,
    )
    ts = sol.ts  # (max_steps + 1,)
    ys = sol.ys  # (max_steps + 1, 3)
    finite = jnp.isfinite(ts)
    angle_initial = _continuous_phi(
        y0_arr[0], y0_arr[1], _device_array(np.pi, dtype), dtype
    )
    (
        trajectory,
        mask,
        steps_taken,
        status,
        t_final,
        phi_hits,
        phi_hits_count,
    ) = _postprocess_diffrax_grid(
        rhs=rhs,
        ts=ts,
        ys=ys,
        finite=finite,
        y0=y0_arr,
        tmax=tmax,
        max_steps=max_steps,
        max_root_iters=max_root_iters,
        max_hits=max_phi_hits,
        angle_targets=phis,
        stopping_criteria=stopping_criteria,
        angle_initial=angle_initial,
        angle_at_state=lambda state, near: _continuous_phi_from_state(
            state, near, dtype
        ),
        dtype=dtype,
    )

    return FieldlineTracingResult(
        trajectory=trajectory,
        mask=mask,
        steps_taken=steps_taken,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits,
        phi_hits_count=phi_hits_count,
    )


def trace_guiding_center_diffrax(
    spec: GuidingCenterTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    m: float,
    q: float,
    mu: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> GuidingCenterTracingResult:
    """Trace one Cartesian guiding-centre orbit with diffrax Dopri5."""

    stopping_criteria = tuple(stopping_criteria)
    _validate_criteria(stopping_criteria)

    dtype = jnp.float64
    y0_arr = _as_device_array(y0, dtype).reshape((4,))
    tmax = _as_device_array(spec.tmax, dtype)
    rtol = _as_device_array(spec.rtol, dtype)
    atol = _as_device_array(spec.atol, dtype)
    dtmax = _as_device_array(spec.dtmax, dtype)
    t0 = _device_array(0.0, dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    rhs = guiding_center_vacuum_rhs(magnetic_field_fn, m, q, mu)
    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    sol = _solve_diffrax_grid(
        rhs=rhs,
        y0=y0_arr,
        tmax=tmax,
        rtol=rtol,
        atol=atol,
        dtmax=dtmax,
        raw_dtmax=spec.dtmax,
        h0=h0,
        max_steps=max_steps,
    )
    angle_initial = _continuous_phi(
        y0_arr[0], y0_arr[1], _device_array(np.pi, dtype), dtype
    )
    (
        trajectory,
        mask,
        steps_taken,
        status,
        t_final,
        phi_hits,
        phi_hits_count,
    ) = _postprocess_diffrax_grid(
        rhs=rhs,
        ts=sol.ts,
        ys=sol.ys,
        finite=jnp.isfinite(sol.ts),
        y0=y0_arr,
        tmax=tmax,
        max_steps=max_steps,
        max_root_iters=max_root_iters,
        max_hits=max_phi_hits,
        angle_targets=phis,
        stopping_criteria=stopping_criteria,
        angle_initial=angle_initial,
        angle_at_state=lambda state, near: _continuous_phi_from_state(
            state, near, dtype
        ),
        dtype=dtype,
    )
    return GuidingCenterTracingResult(
        trajectory=trajectory,
        mask=mask,
        steps_taken=steps_taken,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits,
        phi_hits_count=phi_hits_count,
    )


def trace_fullorbit_diffrax(
    spec: FullorbitTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    m: float,
    q: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FullorbitTracingResult:
    """Trace one Cartesian full-orbit Lorentz trajectory with diffrax Dopri5."""

    stopping_criteria = tuple(stopping_criteria)
    _validate_criteria(stopping_criteria)

    dtype = jnp.float64
    y0_arr = _as_device_array(y0, dtype).reshape((6,))
    tmax = _as_device_array(spec.tmax, dtype)
    rtol = _as_device_array(spec.rtol, dtype)
    atol = _as_device_array(spec.atol, dtype)
    dtmax = _as_device_array(spec.dtmax, dtype)
    t0 = _device_array(0.0, dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    rhs = fullorbit_vacuum_rhs(magnetic_field_fn, m, q)
    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    sol = _solve_diffrax_grid(
        rhs=rhs,
        y0=y0_arr,
        tmax=tmax,
        rtol=rtol,
        atol=atol,
        dtmax=dtmax,
        raw_dtmax=spec.dtmax,
        h0=h0,
        max_steps=max_steps,
    )
    angle_initial = _continuous_phi(
        y0_arr[0], y0_arr[1], _device_array(np.pi, dtype), dtype
    )
    (
        trajectory,
        mask,
        steps_taken,
        status,
        t_final,
        phi_hits,
        phi_hits_count,
    ) = _postprocess_diffrax_grid(
        rhs=rhs,
        ts=sol.ts,
        ys=sol.ys,
        finite=jnp.isfinite(sol.ts),
        y0=y0_arr,
        tmax=tmax,
        max_steps=max_steps,
        max_root_iters=max_root_iters,
        max_hits=max_phi_hits,
        angle_targets=phis,
        stopping_criteria=stopping_criteria,
        angle_initial=angle_initial,
        angle_at_state=lambda state, near: _continuous_phi_from_state(
            state, near, dtype
        ),
        dtype=dtype,
    )
    return FullorbitTracingResult(
        trajectory=trajectory,
        mask=mask,
        steps_taken=steps_taken,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits,
        phi_hits_count=phi_hits_count,
    )


def trace_guiding_center_boozer_diffrax(
    spec: GuidingCenterTracingSpec,
    y0: jax.Array,
    boozer_field,
    m: float,
    q: float,
    mu: float,
    mode: str = "vacuum",
    zetas: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> GuidingCenterTracingResult:
    """Trace one Boozer-coordinate guiding-centre orbit with diffrax Dopri5."""

    stopping_criteria = tuple(stopping_criteria)
    _validate_criteria(stopping_criteria)

    dtype = jnp.float64
    y0_arr = _as_device_array(y0, dtype).reshape((4,))
    tmax = _as_device_array(spec.tmax, dtype)
    rtol = _as_device_array(spec.rtol, dtype)
    atol = _as_device_array(spec.atol, dtype)
    dtmax = _as_device_array(spec.dtmax, dtype)
    t0 = _device_array(0.0, dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    if mode == "vacuum":
        rhs = guiding_center_vacuum_boozer_rhs(boozer_field, m, q, mu)
    elif mode == "no_k":
        rhs = guiding_center_no_k_boozer_rhs(boozer_field, m, q, mu)
    elif mode == "full":
        rhs = guiding_center_boozer_rhs(boozer_field, m, q, mu)
    else:
        raise ValueError(
            "trace_guiding_center_boozer_diffrax mode must be one of "
            f"{{'vacuum', 'no_k', 'full'}}; got mode={mode!r}."
        )

    def safe_rhs(t, y):
        return jax.lax.cond(
            _boozer_axis_invalid(y),
            lambda _: jnp.zeros((4,), dtype=dtype),
            lambda _: rhs(t, y),
            operand=None,
        )

    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    sol = _solve_diffrax_grid(
        rhs=safe_rhs,
        y0=y0_arr,
        tmax=tmax,
        rtol=rtol,
        atol=atol,
        dtmax=dtmax,
        raw_dtmax=spec.dtmax,
        h0=h0,
        max_steps=max_steps,
    )
    angle_initial = _continuous_angle(y0_arr[2], _device_array(np.pi, dtype), dtype)
    (
        trajectory,
        mask,
        steps_taken,
        status,
        t_final,
        zeta_hits,
        zeta_hits_count,
    ) = _postprocess_diffrax_grid(
        rhs=safe_rhs,
        ts=sol.ts,
        ys=sol.ys,
        finite=jnp.isfinite(sol.ts),
        y0=y0_arr,
        tmax=tmax,
        max_steps=max_steps,
        max_root_iters=max_root_iters,
        max_hits=max_phi_hits,
        angle_targets=zetas,
        stopping_criteria=stopping_criteria,
        angle_initial=angle_initial,
        angle_at_state=lambda state, near: _continuous_angle(state[2], near, dtype),
        dtype=dtype,
        is_boozer_state=True,
        stop_on_boozer_axis=True,
    )
    return GuidingCenterTracingResult(
        trajectory=trajectory,
        mask=mask,
        steps_taken=steps_taken,
        status=status,
        t_final=t_final,
        phi_hits=zeta_hits,
        phi_hits_count=zeta_hits_count,
    )
