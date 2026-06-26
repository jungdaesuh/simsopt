"""Diffrax-backed field-line tracer (optional ``integrator="dopri5_diffrax"``).

This is an *alternative backend* for :func:`simsopt_jax.core.tracing.trace_fieldline`,
not a replacement. It delegates the adaptive Dormand-Prince RK4(5) stepping and the
geometric stopping-criterion termination to `diffrax` (Patrick Kidger's vetted JAX ODE
library), while reusing the in-repo crossing/stopping/output machinery so the returned
:class:`~simsopt_jax.core.tracing.FieldlineTracingResult` conforms to the same contract
as the hand-rolled ``dopri5_native`` path.

Division of labour
==================
- **diffrax** owns: the adaptive ``Dopri5`` step sequence (``SaveAt(steps=True)`` gives
  the per-step ``(t, y)`` grid, padded to ``max_steps + 1`` with non-finite rows) and
  per-lane early termination on the geometric stopping criteria via ``diffrax.Event``.
- **native helpers** (imported from :mod:`simsopt_jax.core.tracing`) own: φ-plane
  crossing localization (``_scan_angle_plane_events`` + the same 5th-order ``dopri5_step``
  sub-step localizer used by ``trace_fieldline``), the event-row buffer layout
  (``_append_event_row``), the ``status`` taxonomy, and trajectory padding.

Parity with ``dopri5_native`` — and the deliberate divergences
=============================================================
- **Integration path (no criterion fires / reaches ``tmax``):** the backends agree to
  integrator tolerance (~1e-8). Same Dopri5 tableau; φ-plane punctures match because
  the crossing localizer is the shared native ``dopri5_step`` sub-step.
- **Stopping-criterion exit — terminal state differs by ~one step:** both backends stop
  at step granularity (no sub-step localization). diffrax stops at and RECORDS the first
  accepted step where the criterion is met; native stops at the same crossing but does
  NOT record that firing step (its recorded terminal row is the step *before* the
  boundary, while its ``t_final`` is the firing time). So the recorded terminal trajectory
  row, ``t_final``, and the ``idx<0`` stop row differ by up to one adaptive step between
  the backends (the grids differ regardless). Both report ``status = -1 - i`` and record
  the stop event into ``phi_hits`` (``idx = -1 - i``).
- **Criterion already satisfied at the seed:** honoured by stopping at the seed
  (``status = -1 - i``, seed-only trajectory). diffrax floating events cannot fire on
  the first step, so this case is handled by an explicit pre-check, not the solver.

Scope (first cut): the **fieldline** RHS ``dx/dt = B(x)`` with φ-plane recording and the
geometric stopping criteria ``MinR/MaxR/MinZ/MaxZ/Levelset`` — the Poincaré-complete set.
The guiding-centre / Boozer / full-orbit RHS variants and the integrator-state criteria
(``IterStoppingCriterion``, ``ToroidalTransitStoppingCriterion``, flux criteria) are not
handled here and raise ``NotImplementedError`` loudly (they need step/transit state that
diffrax's stateless event cond-fn does not see).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import diffrax

from ._device_scalars import two_pi as _device_two_pi
from .tracing import (
    FieldlineTracingResult,
    FieldlineTracingSpec,
    LevelsetStoppingCriterion,
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
    _FIELDLINE_INITIAL_STEP_FRACTION,
    _append_event_row,
    _as_device_array,
    _continuous_phi,
    _continuous_phi_from_state,
    _device_array,
    _device_zeros,
    _event_row_from_state,
    _initial_step_size,
    _scan_angle_plane_events,
    dopri5_step,
    fieldline_rhs,
)

__all__ = ["trace_fieldline_diffrax"]

# Stopping criteria whose predicate is a pure function of the instantaneous
# Cartesian state ``(x, y, z)`` — these map to a diffrax ``Event`` cond-fn.
_SUPPORTED_DIFFRAX_CRITERIA = (
    MinRStoppingCriterion,
    MaxRStoppingCriterion,
    MinZStoppingCriterion,
    MaxZStoppingCriterion,
    LevelsetStoppingCriterion,
)

# Event localization uses NO sub-step root-finder: diffrax stops at (and records) the
# first accepted step where the margin crosses <= 0 — matching native's step-granularity
# stop, and robust under vmap. Alternatives rejected: a bracketing root-finder
# (``optimistix.Bisection``) localizes exactly on the boundary but RAISES under vmap when
# lanes disagree on whether they fired ("root not in [lower, upper]"); the quasi-Newton
# ``VeryChord`` DIVERGES on these events (``sol.result`` = "nonlinear solve diverged",
# never ``event_occurred``). ``None`` can do neither.
_DIFFRAX_EVENT_ROOT_FINDER = None


def _criterion_event_margin(criterion, x, y, z, dtype) -> jax.Array:
    """Signed firing margin: ``> 0`` while the criterion has not fired, ``<= 0`` once it
    has. This is the single firing definition for the diffrax backend (the diffrax
    ``Event`` localizes the sign change; ``_fired_index`` classifies by the same margin).

    It mirrors the boolean predicates in
    :func:`simsopt_jax.core.tracing._stopping_criterion_should_stop`; that the two agree
    (``margin <= 0  ⟺  predicate``) is pinned by a property test, since the continuous
    margin and the bool predicate are two encodings of the same criterion.
    """

    if isinstance(criterion, MinRStoppingCriterion):
        # fires when r <= crit_r  ->  margin r - crit_r
        return jnp.sqrt(x * x + y * y) - _device_array(criterion.crit_r, dtype)
    if isinstance(criterion, MaxRStoppingCriterion):
        # fires when r >= crit_r  ->  margin crit_r - r
        return _device_array(criterion.crit_r, dtype) - jnp.sqrt(x * x + y * y)
    if isinstance(criterion, MinZStoppingCriterion):
        return z - _device_array(criterion.crit_z, dtype)
    if isinstance(criterion, MaxZStoppingCriterion):
        return _device_array(criterion.crit_z, dtype) - z
    if isinstance(criterion, LevelsetStoppingCriterion):
        # classifier_fn >= 0 inside, < 0 outside; fires on < 0 -> margin is the value.
        position = jnp.stack([x, y, z]).reshape(1, 3).astype(dtype)
        return criterion.classifier_fn(position)[0]
    raise NotImplementedError(
        f"Criterion {type(criterion).__name__} has no diffrax event margin"
    )


def _validate_criteria(stopping_criteria: tuple) -> None:
    for criterion in stopping_criteria:
        if not isinstance(criterion, _SUPPORTED_DIFFRAX_CRITERIA):
            raise NotImplementedError(
                "The diffrax fieldline backend (integrator='dopri5_diffrax') supports "
                "only the geometric stopping criteria "
                f"{tuple(c.__name__ for c in _SUPPORTED_DIFFRAX_CRITERIA)}; got "
                f"{type(criterion).__name__}. Use integrator='dopri5_native' for "
                "iteration/transit/flux criteria."
            )


def _build_event(stopping_criteria: tuple, dtype):
    """``diffrax.Event`` firing when any geometric criterion *becomes* satisfied, or None.

    ``cond_fn`` returns ``min`` over the per-criterion margins; ``direction=False``
    restricts firing to the down-crossing (margin ``+ -> -``, the criterion becoming
    satisfied) so a line *leaving* an already-satisfied region is NOT a spurious event.
    """

    if not stopping_criteria:
        return None

    def cond_fn(t, y, args, **kwargs):
        del t, args, kwargs
        x, yy, z = y[0], y[1], y[2]
        margins = jnp.stack(
            [
                _criterion_event_margin(criterion, x, yy, z, dtype)
                for criterion in stopping_criteria
            ]
        )
        return jnp.min(margins)

    return diffrax.Event(
        cond_fn, root_finder=_DIFFRAX_EVENT_ROOT_FINDER, direction=False
    )


def _fired_index(stopping_criteria: tuple, x, y, z, dtype):
    """First criterion (definition order) whose event margin is ``<= 0`` at ``(x, y, z)``.

    Returns ``(any_fired, index)``; ``index`` is the 0-based criterion position
    (``status = -1 - index``), 0 when none fired (guard with ``any_fired``). Used both
    at the seed ("is a criterion already satisfied here") and at the terminal state to
    name which criterion fired. With no sub-step root-finder the solve stops at the
    first step where some margin crosses ``<= 0`` (an overshoot), so the firing
    criterion's margin is strictly ``<= 0`` there — this strict test matches native's
    first-satisfied-in-order rule and will not credit a near-coincident criterion whose
    margin is still positive.
    """

    zero = _device_array(0.0, dtype)
    any_fired = _device_array(False, np.bool_)
    index = _device_array(0, jnp.int32)
    for i, criterion in enumerate(stopping_criteria):
        fired = _criterion_event_margin(criterion, x, y, z, dtype) <= zero
        take = jnp.logical_and(fired, jnp.logical_not(any_fired))
        index = jnp.where(take, _device_array(i, jnp.int32), index)
        any_fired = jnp.logical_or(any_fired, fired)
    return any_fired, index


def trace_fieldline_diffrax(
    spec: FieldlineTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FieldlineTracingResult:
    """Trace one fieldline with diffrax's adaptive ``Dopri5``; same contract as
    :func:`simsopt_jax.core.tracing.trace_fieldline`.

    See the module docstring for the diffrax/native division of labour and the
    deliberate stopping-exit divergences. ``spec``, ``y0``, ``magnetic_field_fn``,
    ``phis``, ``stopping_criteria`` and the returned
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
    max_phi_hits_i32 = _device_array(max_phi_hits, jnp.int32)

    rhs = fieldline_rhs(magnetic_field_fn)
    h0 = _initial_step_size(t0, tmax, dtmax, _FIELDLINE_INITIAL_STEP_FRACTION)

    # ── diffrax adaptive solve: per-step (t, y) grid + per-lane event termination ──
    term = diffrax.ODETerm(lambda _t, y, _args: rhs(_t, y))
    # diffrax caps the adaptive step at ``dtmax``; ``inf`` (the unconstrained
    # default) maps to None. A finite cap may be concrete or, on the batched
    # path, a per-lane traced scalar — both pass straight through.
    raw_dtmax = spec.dtmax
    if isinstance(raw_dtmax, (int, float, np.floating)) and not np.isfinite(raw_dtmax):
        dtmax_arg = None
    else:
        dtmax_arg = dtmax
    controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax_arg)

    event = _build_event(stopping_criteria, dtype)
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=t0,
        t1=tmax,
        dt0=h0,
        y0=y0_arr,
        saveat=diffrax.SaveAt(t0=True, steps=True),
        stepsize_controller=controller,
        event=event,
        max_steps=max_steps,
        throw=False,
    )

    ts = sol.ts  # (max_steps + 1,)
    ys = sol.ys  # (max_steps + 1, 3)
    finite = jnp.isfinite(ts)

    # Constant-extend padded rows with the last accepted state (mask-honouring
    # downstream code is unaffected; matches the native padding contract).
    # int32 throughout so ``steps_taken`` matches the native int32 contract.
    last_idx = (jnp.sum(finite.astype(jnp.int32)) - 1).astype(jnp.int32)
    t_final = ts[last_idx]
    y_final = ys[last_idx]
    ts_filled = jnp.where(finite, ts, t_final)
    ys_filled = jnp.where(finite[:, None], ys, jnp.broadcast_to(y_final, ys.shape))
    trajectory = jnp.concatenate([ts_filled[:, None], ys_filled], axis=1)
    mask = finite
    steps_taken = last_idx  # accepted steps, excluding the initial row (index 0)

    # ── status taxonomy (mirrors trace_fieldline's status assembly) ──
    # ``event_occurred`` is diffrax's own ground truth that an Event terminated the
    # solve (NOT a post-hoc predicate re-check, which mis-reads at the root tolerance
    # and would mislabel a line that merely reaches ``tmax`` with a criterion
    # coincidentally true at the endpoint).
    eps_t = _device_array(1.0e-12, dtype) * jnp.maximum(
        jnp.abs(tmax), _device_array(1.0, dtype)
    )
    reached = (tmax - t_final) <= eps_t
    event_occurred = sol.result == diffrax.RESULTS.event_occurred
    _fired, fired_index = _fired_index(
        stopping_criteria, y_final[0], y_final[1], y_final[2], dtype
    )
    status_normal = jnp.where(
        reached, _device_array(0, jnp.int32), _device_array(1, jnp.int32)
    )
    status = jnp.where(
        event_occurred, -_device_array(1, jnp.int32) - fired_index, status_normal
    )

    # ── φ-plane crossing recording over the diffrax step grid ──
    phi_hits, phi_hits_count = _record_phi_hits(
        rhs=rhs,
        ts=ts,
        ys=ys,
        finite=finite,
        phis=phis,
        max_steps=max_steps,
        max_phi_hits=max_phi_hits,
        max_root_iters=max_root_iters,
        dtype=dtype,
    )

    # Record the stopping-criterion firing row (idx = -1 - i) into the same phi_hits
    # buffer the native driver uses, so the contract's idx<0 rows are present.
    if stopping_criteria:
        stop_row = _event_row_from_state(
            t_final,
            -_device_array(1.0, dtype) - fired_index.astype(dtype),
            y_final,
        )
        phi_hits, phi_hits_count = _append_event_row(
            phi_hits, phi_hits_count, event_occurred, stop_row, max_phi_hits_i32
        )

        # Honour a criterion already satisfied AT THE SEED: diffrax floating events
        # cannot fire on step 1, so without this the line would integrate a full
        # excursion before re-crossing. Stop at the seed (status -1-i, seed-only path).
        fired_at_t0, t0_index = _fired_index(
            stopping_criteria, y0_arr[0], y0_arr[1], y0_arr[2], dtype
        )
        seed_row = jnp.concatenate([jnp.reshape(t0, (1,)), y0_arr])
        traj_seed = jnp.broadcast_to(seed_row, trajectory.shape)
        mask_seed = jax.lax.iota(jnp.int32, max_steps + 1) == _device_array(0, jnp.int32)
        status_seed = -_device_array(1, jnp.int32) - t0_index
        phi_seed = _device_zeros((max_phi_hits, 5), dtype).at[0].set(
            _event_row_from_state(
                t0, -_device_array(1.0, dtype) - t0_index.astype(dtype), y0_arr
            )
        )
        trajectory = jnp.where(fired_at_t0, traj_seed, trajectory)
        mask = jnp.where(fired_at_t0, mask_seed, mask)
        steps_taken = jnp.where(fired_at_t0, _device_array(0, jnp.int32), steps_taken)
        status = jnp.where(fired_at_t0, status_seed, status)
        t_final = jnp.where(fired_at_t0, t0, t_final)
        phi_hits = jnp.where(fired_at_t0, phi_seed, phi_hits)
        phi_hits_count = jnp.where(
            fired_at_t0, _device_array(1, jnp.int32), phi_hits_count
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


def _record_phi_hits(
    *,
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    ts: jax.Array,
    ys: jax.Array,
    finite: jax.Array,
    phis: jax.Array | None,
    max_steps: int,
    max_phi_hits: int,
    max_root_iters: int,
    dtype,
) -> tuple[jax.Array, jax.Array]:
    """Record φ-plane crossings over consecutive diffrax steps.

    For each finite segment ``[ts[i], ts[i+1]]`` the crossing is localized with the
    same 5th-order ``dopri5_step`` sub-step that ``trace_fieldline`` uses at the
    accepted-step level, so the recorded ``[t, idx, x, y, z]`` rows match the native
    backend's localization to integrator tolerance.
    """

    phi_hits = _device_zeros((max_phi_hits, 5), dtype)
    count0 = _device_array(0, jnp.int32)
    max_phi_hits_i32 = _device_array(max_phi_hits, jnp.int32)
    two_pi = _device_two_pi(_device_array(1.0, dtype))

    if phis is None:
        phis_arr = _device_zeros((0,), dtype)
    else:
        phis_arr = _as_device_array(phis, dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    if num_phis == 0:
        return phi_hits, count0

    phi_init = _continuous_phi(ys[0, 0], ys[0, 1], _device_array(np.pi, dtype), dtype)

    def seg_step(carry, i):
        hits, count, phi_last = carry
        t_i = ts[i]
        h = ts[i + 1] - t_i
        y_i = ys[i]
        y_ip = ys[i + 1]
        seg_finite = jnp.logical_and(finite[i], finite[i + 1])

        phi_current = _continuous_phi(y_ip[0], y_ip[1], phi_last, dtype)
        k_first = rhs(t_i, y_i)

        def state_at_fraction(s):
            y_sub, _err, _k7 = dopri5_step(rhs, t_i, y_i, s * h, k_first)
            return y_sub

        def do_record(operand):
            hits_in, count_in = operand
            return _scan_angle_plane_events(
                hits=hits_in,
                count=count_in,
                angle_last=phi_last,
                angle_current=phi_current,
                targets=phis_arr,
                num_targets=num_phis,
                two_pi=two_pi,
                dtype=dtype,
                t=t_i,
                h_clamped=h,
                max_root_iters=max_root_iters,
                max_hits_i32=max_phi_hits_i32,
                state_at_fraction=state_at_fraction,
                angle_at_state=lambda state, near: _continuous_phi_from_state(
                    state, near, dtype
                ),
            )

        hits_next, count_next = jax.lax.cond(
            seg_finite,
            do_record,
            lambda operand: operand,
            operand=(hits, count),
        )
        phi_last_next = jnp.where(seg_finite, phi_current, phi_last)
        return (hits_next, count_next, phi_last_next), None

    (phi_hits, phi_hits_count, _phi_last), _ = jax.lax.scan(
        seg_step,
        (phi_hits, count0, phi_init),
        jnp.arange(max_steps),
    )
    return phi_hits, phi_hits_count
