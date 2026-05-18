"""JAX port of ``simsoptpp/tracing.cpp`` (Tier P1 item 14).

This module implements an in-repo JAX Dormand-Prince RK4(5) integrator
with a PI step controller and a bracketed Illinois false-position event
localizer.
The implemented scope covers:

- the fieldline RHS ``dx/dt = B(x)`` used in the upstream C++
  ``FieldlineRHS``,
- the 4-state Cartesian vacuum guiding-centre RHS shipped under the
  item-14 follow-up (state ``[x, y, z, v_par]``, drift terms following
  the upstream ``GuidingCenterVacuumRHS::operator()`` definition in
  ``simsoptpp/tracing.cpp``). The driver :func:`trace_guiding_center`
  shares the same DOPRI5 + PI controller pattern as
  :func:`trace_fieldline`, and
- the three 4-state Boozer-coordinate guiding-centre RHS variants
  ``GuidingCenterVacuumBoozerRHS``, ``GuidingCenterNoKBoozerRHS`` and
  ``GuidingCenterBoozerRHS`` (state ``[s, theta, zeta, v_par]``). The
  driver :func:`trace_guiding_center_boozer` switches between the
  three variants via ``mode in {'vacuum', 'no_k', 'full'}`` and reuses
  the DOPRI5 + PI controller machinery from the Cartesian path, and
- the 6-state Cartesian full-orbit Lorentz RHS ``y = (x, y, z, vx, vy,
  vz)`` with ``dx/dt = v`` and ``dv/dt = (q/m) v x B`` following the
  upstream ``FullorbitRHS::operator()`` (vacuum branch; no E field).
  The driver :func:`trace_fullorbit` reuses the same DOPRI5 + PI
  controller machinery.

Carve-outs (NOT implemented here):

- Non-vacuum guiding-centre Cartesian RHS (``GuidingCenterRHS``) — not
  exposed by the upstream public surface today and not required by
  the active JAX-native consumers.

The bracketed event localizer uses an Illinois false-position update
with a fixed iteration ceiling (``max_root_iters``). This keeps the
same static ``jax.lax.while_loop`` carry shape required by JAX while
using one RHS-style event residual evaluation per active iteration and
avoiding the linear convergence of bisection. The accepted accuracy
contract is the ``event_time_tracing`` lane in
``benchmarks.validation_ladder_contract.PARITY_LADDER_TOLERANCES``.

Architecture
============

- ``FieldlineTracingSpec`` — frozen dataclass (registered as a JAX
  pytree) carrying ``tmax``, ``rtol``, ``atol``, per-lane ``dtmax``,
  ``max_steps`` (static), and ``max_root_iters`` (static).
- ``dopri5_step`` — single Dormand-Prince step returning the 5th-order
  state, the embedded error vector, and the trailing-stage derivative
  for FSAL reuse.
- ``trace_fieldline`` — adaptive driver inside a ``jax.lax.while_loop``
  with a fixed-shape trajectory carry of shape
  ``(max_steps + 1, 4)``. Padded rows are populated with the final
  accepted state; the companion mask of shape ``(max_steps + 1,)``
  identifies the live prefix.
- ``bracket_root_jax`` — Illinois false-position event localizer. Returns the
  bracketed root and a bool indicating whether the bracket actually
  contained a sign change.

The PI step controller, error norm, and step-size update follow the
Hairer reference driver used in
``simsopt.jax_core.magnetic_axis_helpers`` so the two integrators are
consistent and easy to validate against the same SciPy oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .boozer_radial_field import (
    BoozerRadialInterpolantFrozenState,
    _eval_dGds as _radial_dGds,
    _eval_dIds as _radial_dIds,
    _eval_dKdtheta as _radial_dKdtheta,
    _eval_dKdzeta as _radial_dKdzeta,
    _eval_dmodBds as _radial_dmodBds,
    _eval_dmodBdtheta as _radial_dmodBdtheta,
    _eval_dmodBdzeta as _radial_dmodBdzeta,
    _eval_G as _radial_G,
    _eval_I as _radial_I,
    _eval_iota as _radial_iota,
    _eval_K as _radial_K,
    _eval_modB as _radial_modB,
)
from .boozer_analytic import (
    BoozerAnalyticFrozenState,
    _eval_dGds as _analytic_dGds,
    _eval_dIds as _analytic_dIds,
    _eval_dKdtheta as _analytic_dKdtheta,
    _eval_dKdzeta as _analytic_dKdzeta,
    _eval_dmodBds as _analytic_dmodBds,
    _eval_dmodBdtheta as _analytic_dmodBdtheta,
    _eval_dmodBdzeta as _analytic_dmodBdzeta,
    _eval_G as _analytic_G,
    _eval_I as _analytic_I,
    _eval_iota as _analytic_iota,
    _eval_K as _analytic_K,
    _eval_modB as _analytic_modB,
)
from .interpolated_boozer_field import (
    InterpolatedBoozerFieldFrozenState,
    _INTERP_EVALUATORS,
)

__all__ = [
    "FieldlineTracingSpec",
    "FieldlineTracingResult",
    "FullorbitTracingResult",
    "FullorbitTracingSpec",
    "GuidingCenterTracingSpec",
    "GuidingCenterTracingResult",
    "IterStoppingCriterion",
    "LevelsetStoppingCriterion",
    "MaxRStoppingCriterion",
    "MaxToroidalFluxStoppingCriterion",
    "MaxZStoppingCriterion",
    "MinRStoppingCriterion",
    "MinToroidalFluxStoppingCriterion",
    "MinZStoppingCriterion",
    "ToroidalTransitStoppingCriterion",
    "bracket_root_jax",
    "dopri5_step",
    "fieldline_rhs",
    "fullorbit_vacuum_rhs",
    "get_phi",
    "guiding_center_boozer_rhs",
    "guiding_center_no_k_boozer_rhs",
    "guiding_center_vacuum_boozer_rhs",
    "guiding_center_vacuum_rhs",
    "trace_fieldline",
    "trace_fieldlines_batched",
    "trace_fullorbit",
    "trace_fullorbits_batched",
    "trace_guiding_center",
    "trace_guiding_center_boozer",
    "trace_guiding_centers_batched",
    "trace_guiding_centers_boozer_batched",
]


# ── Dormand-Prince RK4(5) Butcher tableau (Hairer et al. Table 5.2) ──

_DOPRI5_C = np.array(
    [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0],
    dtype=np.float64,
)

_DOPRI5_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ],
    dtype=np.float64,
)

_DOPRI5_B = np.array(
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
    dtype=np.float64,
)

_DOPRI5_E = np.array(
    [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ],
    dtype=np.float64,
)


# Standard PI(0.2) controller constants. ``_DOPRI5_EXP = 1/order`` for the
# adaptive RK4(5) embedded pair; the safety / clip factors match Hairer's
# canonical driver and the values reused in ``magnetic_axis_helpers``.
_DOPRI5_EXP = 0.2
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0
_FIELDLINE_INITIAL_STEP_FRACTION = 1.0e-5
_PARTICLE_INITIAL_STEP_FRACTION = 1.0e-3
_BOOZER_AXIS_STATUS = -2


def _append_event_row(
    phi_hits: jax.Array,
    phi_hits_count: jax.Array,
    event_detected: jax.Array,
    hit_row: jax.Array,
    max_phi_hits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Append ``hit_row`` when capacity permits while counting all events."""

    one = jnp.asarray(1, dtype=jnp.int32)
    has_room = phi_hits_count < max_phi_hits
    should_write = jnp.logical_and(event_detected, has_room)
    record_index = jnp.minimum(phi_hits_count, max_phi_hits - one)

    phi_hits = jax.lax.cond(
        should_write,
        lambda hits: hits.at[record_index].set(hit_row),
        lambda hits: hits,
        phi_hits,
    )
    event_count = jnp.where(event_detected, one, jnp.asarray(0, dtype=jnp.int32))
    phi_hits_count = phi_hits_count + event_count
    return phi_hits, phi_hits_count


@dataclass(frozen=True)
class FieldlineTracingSpec:
    """Immutable contract for a single fieldline integration call.

    Parameters
    ----------
    tmax
        Final integration parameter. The integrator runs from ``t=0`` to
        ``t=tmax`` using the upstream ``dx/dt = B(x)`` fieldline
        parameterisation.
    rtol
        Relative tolerance fed to the embedded error norm.
    atol
        Absolute tolerance fed to the embedded error norm.
    max_steps
        Static upper bound on the number of accepted/rejected step
        iterations the JIT-compiled while-loop will execute. The
        trajectory carry has shape ``(max_steps + 1, 4)`` (the ``+1``
        captures the initial state).
    dtmax
        Maximum absolute step size. ``inf`` leaves the adaptive controller
        unconstrained except for the final ``tmax - t`` clamp.
    max_root_iters
        Static iteration ceiling for the Illinois false-position event
        localizer.
    max_phi_hits
        Static upper bound on the number of phi-plane and stopping-
        criterion event rows recorded. The ``phi_hits`` buffer has
        shape ``(max_phi_hits, 5)`` (or ``(max_phi_hits, 6)`` for the
        guiding-centre driver). ``phi_hits_count`` counts every
        detected event, so ``phi_hits_count > max_phi_hits`` is an
        overflow signal; the buffer stores the recorded prefix.
    """

    tmax: float
    rtol: float
    atol: float
    max_steps: int
    dtmax: float = np.inf
    max_root_iters: int = 60
    max_phi_hits: int = 128


jax.tree_util.register_dataclass(
    FieldlineTracingSpec,
    data_fields=["tmax", "rtol", "atol", "dtmax"],
    meta_fields=["max_steps", "max_root_iters", "max_phi_hits"],
)


@dataclass(frozen=True)
class FieldlineTracingResult:
    """Return payload for :func:`trace_fieldline`.

    Fields are JAX arrays so the structure can be returned from inside
    a JIT-compiled wrapper without dropping device residency.

    - ``trajectory`` — ``(max_steps + 1, 4)`` float64 array. Columns are
      ``(t, x, y, z)``. Rows ``[0 : steps_taken + 1]`` are populated
      with accepted states; subsequent rows are padded with the final
      accepted state.
    - ``mask`` — ``(max_steps + 1,)`` bool array. ``True`` for rows that
      correspond to genuine accepted steps; ``False`` for padding.
    - ``steps_taken`` — int32 scalar; count of *accepted* steps the loop
      executed. Note that this excludes the initial-state row.
    - ``status`` — int32 scalar. ``0`` for normal exit (``t >= tmax``),
      ``1`` for max-step-cap exhaustion before reaching ``tmax``,
      ``-1 - i`` when stopping criterion ``i`` fired.
    - ``t_final`` — float64 scalar; ``trajectory[steps_taken, 0]``.
    - ``phi_hits`` — ``(max_phi_hits, 5)`` float64 array. Columns are
      ``[t_hit, idx, x, y, z]``. ``idx >= 0`` denotes a phi-plane
      crossing for ``phis[int(idx)]``; ``idx < 0`` denotes stopping
      criterion ``-1 - int(idx)`` firing.
    - ``phi_hits_count`` — int32 scalar; total detected event count.
      Values greater than ``max_phi_hits`` mean the fixed buffer holds
      a truncated prefix.
    """

    trajectory: jax.Array
    mask: jax.Array
    steps_taken: jax.Array
    status: jax.Array
    t_final: jax.Array
    phi_hits: jax.Array
    phi_hits_count: jax.Array


jax.tree_util.register_dataclass(
    FieldlineTracingResult,
    data_fields=[
        "trajectory",
        "mask",
        "steps_taken",
        "status",
        "t_final",
        "phi_hits",
        "phi_hits_count",
    ],
    meta_fields=[],
)


# ── Stopping-criterion dataclasses ────────────────────────────────────


@dataclass(frozen=True)
class MinRStoppingCriterion:
    """Stop when ``sqrt(x^2 + y^2) <= crit_r``.

    Mirrors :class:`simsoptpp.MinRStoppingCriterion`. Pure JAX: the
    predicate is evaluated on the post-step Cartesian state.
    """

    crit_r: float


@dataclass(frozen=True)
class MaxRStoppingCriterion:
    """Stop when ``sqrt(x^2 + y^2) >= crit_r``."""

    crit_r: float


@dataclass(frozen=True)
class MinZStoppingCriterion:
    """Stop when ``z <= crit_z``."""

    crit_z: float


@dataclass(frozen=True)
class MaxZStoppingCriterion:
    """Stop when ``z >= crit_z``."""

    crit_z: float


@dataclass(frozen=True)
class ToroidalTransitStoppingCriterion:
    """Stop after the trajectory has completed ``max_transits`` full toroidal turns.

    The transit count is unwrapped from the continuous-branch ``phi``
    accumulator the driver maintains for the phi-plane crossing scan.
    Matches :class:`simsoptpp.ToroidalTransitStoppingCriterion` with
    ``flux=False``.
    """

    max_transits: float


@dataclass(frozen=True)
class IterStoppingCriterion:
    """Stop after the integrator has run ``max_iter`` steps.

    Matches :class:`simsoptpp.IterationStoppingCriterion`. The driver
    counts every loop iteration (including rejected steps) to match
    the upstream semantics.
    """

    max_iter: int


@dataclass(frozen=True)
class MinToroidalFluxStoppingCriterion:
    """Stop when toroidal flux ``s <= min_s`` (Boozer/flux traces only).

    Carve-out keeper: this criterion only applies to flux-coordinate
    (Boozer) traces. The Cartesian fieldline / GC drivers shipped here
    never evaluate the user-supplied ``field_fn`` and the criterion is
    effectively inactive on the JAX path. Kept in the public roster so
    the field/tracing isinstance dispatch can recognise it and route it
    to the appropriate Boozer driver once that path lands.
    """

    min_s: float
    field_fn: object = None


@dataclass(frozen=True)
class MaxToroidalFluxStoppingCriterion:
    """Stop when toroidal flux ``s >= max_s``. Deferred carve-out keeper."""

    max_s: float
    field_fn: object = None


@dataclass(frozen=True)
class LevelsetStoppingCriterion:
    """Stop when the JAX surface classifier reports the trajectory is outside.

    Mirrors :class:`simsoptpp.LevelsetStoppingCriterion`. The
    ``classifier_fn`` is a JAX-traceable callable ``classifier_fn(x, y, z) ->
    sign`` produced by
    :func:`simsopt.jax_core.surface_classifier.make_levelset_classifier`
    (returns ``+1`` inside, ``-1`` outside). The criterion fires when the
    post-step Cartesian position has ``classifier_fn(x, y, z) < 0``, matching
    the upstream C++ ``f < 0`` predicate.
    """

    classifier_fn: object


def _stopping_criterion_should_stop(
    criterion: object,
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    iter_count: jax.Array,
    phi_unwrapped: jax.Array,
    phi_init: jax.Array,
    dtype: jnp.dtype,
    is_boozer_state: bool = False,
) -> jax.Array:
    """Evaluate a single stopping criterion on the post-step state.

    The predicate must return a boolean scalar that can be folded into
    the driver's ``stop`` mask via ``jnp.logical_or``. All criteria
    consume the same fixed-shape state ``(x, y, z, iter, phi_unwrap,
    phi_init)``; criteria that do not care about a given component
    ignore it. When ``is_boozer_state`` is True the first state slot
    ``x`` represents the Boozer flux coordinate ``s`` and the
    flux-coordinate criteria ``MinToroidalFluxStoppingCriterion`` /
    ``MaxToroidalFluxStoppingCriterion`` fire on ``s``; on the
    Cartesian path they remain inactive (matching the upstream
    ``simsoptpp/tracing.cpp`` flux-only contract).
    """

    if isinstance(criterion, MinRStoppingCriterion):
        r = jnp.sqrt(x * x + y * y)
        return r <= jnp.asarray(criterion.crit_r, dtype=dtype)
    if isinstance(criterion, MaxRStoppingCriterion):
        r = jnp.sqrt(x * x + y * y)
        return r >= jnp.asarray(criterion.crit_r, dtype=dtype)
    if isinstance(criterion, MinZStoppingCriterion):
        return z <= jnp.asarray(criterion.crit_z, dtype=dtype)
    if isinstance(criterion, MaxZStoppingCriterion):
        return z >= jnp.asarray(criterion.crit_z, dtype=dtype)
    if isinstance(criterion, ToroidalTransitStoppingCriterion):
        transits = jnp.abs(phi_unwrapped - phi_init) / jnp.asarray(
            2.0 * np.pi, dtype=dtype
        )
        return transits >= jnp.asarray(criterion.max_transits, dtype=dtype)
    if isinstance(criterion, IterStoppingCriterion):
        return iter_count > jnp.asarray(int(criterion.max_iter), dtype=jnp.int32)
    if isinstance(criterion, MinToroidalFluxStoppingCriterion):
        if is_boozer_state:
            return x <= jnp.asarray(criterion.min_s, dtype=dtype)
        return jnp.asarray(False)
    if isinstance(criterion, MaxToroidalFluxStoppingCriterion):
        if is_boozer_state:
            return x >= jnp.asarray(criterion.max_s, dtype=dtype)
        return jnp.asarray(False)
    if isinstance(criterion, LevelsetStoppingCriterion):
        # Surface classifier returns +1 inside, -1 outside; stop on the
        # accepted step that crosses to < 0 (matches upstream
        # ``simsoptpp/tracing.cpp::LevelsetStoppingCriterion``).
        position = jnp.stack([x, y, z]).reshape(1, 3).astype(dtype)
        sign = criterion.classifier_fn(position)[0]
        return sign < jnp.asarray(0.0, dtype=dtype)
    raise NotImplementedError(
        f"Unsupported JAX stopping criterion: {type(criterion).__name__}"
    )


def _continuous_phi(
    x: jax.Array, y: jax.Array, phi_near: jax.Array, dtype: jnp.dtype
) -> jax.Array:
    """Continuous-branch ``atan2(y, x)`` near ``phi_near``.

    Mirrors the C++ ``get_phi`` helper in ``simsoptpp/tracing.cpp``:
    pick the integer multiple of ``2*pi`` so the unwrapped ``phi`` is
    within ``pi`` of ``phi_near``. Used so the per-step ``phi_last`` /
    ``phi_current`` accumulator continuously tracks the trajectory and
    the floor-division crossing test does not miss a 2pi wrap.
    """

    two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)
    phi_raw = jnp.arctan2(y, x)
    phi = jnp.where(phi_raw < jnp.asarray(0.0, dtype=dtype), phi_raw + two_pi, phi_raw)
    nearest_multiple = (
        jnp.sign(phi_near)
        * jnp.floor(jnp.abs(phi_near / two_pi) + jnp.asarray(0.5, dtype=dtype))
        * two_pi
    )
    opt1 = nearest_multiple - two_pi + phi
    opt2 = nearest_multiple + phi
    opt3 = nearest_multiple + two_pi + phi
    dist1 = jnp.abs(opt1 - phi_near)
    dist2 = jnp.abs(opt2 - phi_near)
    dist3 = jnp.abs(opt3 - phi_near)
    return jnp.where(
        dist1 <= jnp.minimum(dist2, dist3),
        opt1,
        jnp.where(dist2 <= jnp.minimum(dist1, dist3), opt2, opt3),
    )


def get_phi(x, y, phi_near) -> jax.Array:
    """Public JAX wrapper for the C++ ``get_phi`` continuous branch helper."""

    dtype = jnp.result_type(x, y, phi_near)
    return _continuous_phi(
        jnp.asarray(x, dtype=dtype),
        jnp.asarray(y, dtype=dtype),
        jnp.asarray(phi_near, dtype=dtype),
        dtype,
    )


def _continuous_angle(
    angle_raw: jax.Array, angle_near: jax.Array, dtype: jnp.dtype
) -> jax.Array:
    """Continuous-branch unwrap of a scalar angle near ``angle_near``.

    Companion to :func:`_continuous_phi` for the Boozer-coordinate
    state where ``zeta`` is already a scalar angle (not a Cartesian
    ``atan2(y, x)``). The C++ ``get_phi`` helper has a single
    ``get_angle`` equivalent in ``simsoptpp/tracing.cpp`` driving the
    ``zeta - zeta_target`` modulo-``2*pi`` detection on the Boozer
    route. We replicate the same logic: pick the integer multiple of
    ``2*pi`` so the unwrapped angle lies within ``pi`` of ``angle_near``.
    Used so the per-step ``zeta_last`` / ``zeta_current`` accumulator
    continuously tracks the Boozer trajectory and the floor-division
    crossing test does not miss a ``2*pi`` wrap.
    """

    two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)
    k = jnp.round((angle_near - angle_raw) / two_pi)
    return angle_raw + k * two_pi


def _record_trajectory_row(
    traj: jax.Array,
    mask: jax.Array,
    accepted_count: jax.Array,
    t_next: jax.Array,
    y_next: jax.Array,
    should_record: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    write_row = accepted_count + jnp.asarray(1, dtype=jnp.int32)

    def write(args):
        traj_in, mask_in, row, time, state = args
        return (
            traj_in.at[row, 0].set(time).at[row, 1:].set(state),
            mask_in.at[row].set(True),
        )

    traj_next, mask_next = jax.lax.cond(
        should_record,
        write,
        lambda args: (args[0], args[1]),
        operand=(traj, mask, write_row, t_next, y_next),
    )
    accepted_next = accepted_count + jnp.where(
        should_record,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return traj_next, mask_next, accepted_next


def _should_record_accepted_step(
    accepted: jax.Array, stop_after: jax.Array
) -> jax.Array:
    return jnp.logical_and(accepted, jnp.logical_not(stop_after))


def _boozer_axis_invalid(y: jax.Array) -> jax.Array:
    s = y[0]
    zero = jnp.asarray(0.0, dtype=s.dtype)
    return jnp.logical_or(s <= zero, jnp.logical_not(jnp.isfinite(s)))


# ── Fieldline RHS ─────────────────────────────────────────────────────


def fieldline_rhs(
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Return ``rhs(t, y) -> dy/dt`` for the upstream fieldline equation.

    Parameters
    ----------
    magnetic_field_fn
        JAX-traceable callable mapping a Cartesian point ``[3]`` to the
        magnetic field ``B(x)`` of shape ``[3]``.

    Returns
    -------
    rhs
        Closure that evaluates ``B(y)``, matching the upstream C++
        ``FieldlineRHS`` parameterisation.
    """

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t  # Field is autonomous; signature kept for ODE-driver shape.
        B = magnetic_field_fn(y)
        return jnp.asarray(B, dtype=y.dtype).reshape((3,))

    return rhs


# ── Dormand-Prince single-step ────────────────────────────────────────


def dopri5_step(
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    t: jax.Array,
    y: jax.Array,
    h: jax.Array,
    k_first: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single Dormand-Prince RK4(5) step.

    Parameters
    ----------
    rhs
        ``rhs(t, y) -> dy/dt`` callable.
    t, y
        Current independent variable scalar and state vector.
    h
        Step size scalar.
    k_first
        Pre-computed ``rhs(t, y)`` (FSAL reuse from prior accepted step
        or freshly computed at integration start).

    Returns
    -------
    y_new
        5th-order RK estimate at ``t + h``.
    y_err
        Embedded error estimate ``b - b_hat`` weighted by ``h``.
    k7
        ``rhs(t + h, y_new)``; FSAL reuse for the next step.
    """

    dtype = y.dtype
    A = jnp.asarray(_DOPRI5_A, dtype=dtype)
    C = jnp.asarray(_DOPRI5_C, dtype=dtype)
    B = jnp.asarray(_DOPRI5_B, dtype=dtype)
    E = jnp.asarray(_DOPRI5_E, dtype=dtype)

    k1 = k_first
    k2 = rhs(t + C[1] * h, y + h * (A[1, 0] * k1))
    k3 = rhs(t + C[2] * h, y + h * (A[2, 0] * k1 + A[2, 1] * k2))
    k4 = rhs(
        t + C[3] * h,
        y + h * (A[3, 0] * k1 + A[3, 1] * k2 + A[3, 2] * k3),
    )
    k5 = rhs(
        t + C[4] * h,
        y + h * (A[4, 0] * k1 + A[4, 1] * k2 + A[4, 2] * k3 + A[4, 3] * k4),
    )
    k6 = rhs(
        t + C[5] * h,
        y
        + h
        * (A[5, 0] * k1 + A[5, 1] * k2 + A[5, 2] * k3 + A[5, 3] * k4 + A[5, 4] * k5),
    )
    y_new = y + h * (B[0] * k1 + B[2] * k3 + B[3] * k4 + B[4] * k5 + B[5] * k6)
    k7 = rhs(t + h, y_new)
    y_err = h * (E[0] * k1 + E[2] * k3 + E[3] * k4 + E[4] * k5 + E[5] * k6 + E[6] * k7)
    return y_new, y_err, k7


def _error_norm(
    y_err: jax.Array,
    y: jax.Array,
    y_new: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
) -> jax.Array:
    sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    return jnp.sqrt(jnp.mean(jnp.square(y_err / sc)))


def _initial_step_size(
    t0: jax.Array, t_end: jax.Array, dtmax: jax.Array, fraction: float
) -> jax.Array:
    span = jnp.abs(t_end - t0)
    h0 = jnp.asarray(fraction, dtype=span.dtype) * dtmax
    return jnp.minimum(h0, span)


def _clamp_step_to_domain(
    h: jax.Array, t: jax.Array, tmax: jax.Array, dtmax: jax.Array
) -> jax.Array:
    return jnp.minimum(jnp.minimum(h, tmax - t), dtmax)


def _accepted_step_time(
    t: jax.Array, h_clamped: jax.Array, tmax: jax.Array
) -> jax.Array:
    return jnp.where(h_clamped >= tmax - t, tmax, t + h_clamped)


# ── Bracketed Illinois event localizer ────────────────────────────────


def bracket_root_jax(
    f: Callable[[jax.Array], jax.Array],
    t_left: jax.Array,
    t_right: jax.Array,
    f_left: jax.Array,
    f_right: jax.Array,
    max_iters: int,
    atol: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Find ``t*`` with ``f(t*) = 0`` inside ``[t_left, t_right]``.

    Parameters
    ----------
    f
        Scalar-valued JAX function.
    t_left, t_right
        Initial bracket endpoints. They are normalized internally so
        descending brackets follow the same update path as ascending
        brackets.
    f_left, f_right
        ``f(t_left)`` and ``f(t_right)``. Supplied explicitly so the
        caller can reuse function values already computed during the
        sign-crossing detection scan.
    max_iters
        Static iteration ceiling for the Illinois false-position loop.
    atol
        Absolute width tolerance for early exit. When
        ``t_right - t_left <= atol`` the controller stops shrinking the
        bracket; the loop still executes ``max_iters`` iterations for
        fixed-shape compile but the bracket remains stationary on noop
        iterations.

    Returns
    -------
    t_star
        Best finite false-position candidate or initial endpoint by absolute residual.
    f_at_t_star
        ``f(t_star)`` evaluated at the returned ``t_star``.
    bracketed
        Bool scalar. ``True`` if ``sign(f_left) != sign(f_right)`` on
        entry, i.e. the input bracket genuinely contained a sign
        change. ``False`` leaves the loop state stationary so the caller can
        treat the result as "no event" rather than a numerical root.
    """

    dtype = f_left.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    atol_arr = jnp.asarray(atol, dtype=dtype)
    left_first = t_left <= t_right
    t_left_ordered = jnp.where(left_first, t_left, t_right)
    t_right_ordered = jnp.where(left_first, t_right, t_left)
    f_left_ordered = jnp.where(left_first, f_left, f_right)
    f_right_ordered = jnp.where(left_first, f_right, f_left)
    bracketed_in = jnp.sign(f_left_ordered) * jnp.sign(f_right_ordered) < zero
    left_better = jnp.abs(f_left_ordered) <= jnp.abs(f_right_ordered)
    init = (
        jnp.asarray(0, dtype=jnp.int32),
        t_left_ordered,
        t_right_ordered,
        f_left_ordered,
        f_right_ordered,
        jnp.where(left_better, t_left_ordered, t_right_ordered),
        jnp.where(left_better, f_left_ordered, f_right_ordered),
    )

    def cond(carry):
        i, _a, _b, _fa, _fb, _best_t, _best_f = carry
        return i < jnp.asarray(int(max_iters), dtype=jnp.int32)

    def body(carry):
        i, a, b, fa, fb, best_t, best_f = carry
        width = b - a
        converged = width <= atol_arr
        active = jnp.logical_and(bracketed_in, jnp.logical_not(converged))
        midpoint = a + half * width
        denominator = fb - fa
        false_position = jax.lax.cond(
            denominator == zero,
            lambda _: midpoint,
            lambda _: b - fb * width / denominator,
            operand=None,
        )
        candidate = jnp.where(jnp.isfinite(false_position), false_position, midpoint)
        fc = jax.lax.cond(
            active,
            lambda _: f(candidate),
            lambda _: best_f,
            operand=None,
        )
        improves_best = jnp.logical_and(
            active,
            jnp.abs(fc) < jnp.abs(best_f),
        )
        best_t_next = jnp.where(improves_best, candidate, best_t)
        best_f_next = jnp.where(improves_best, fc, best_f)

        keep_left = jnp.sign(fa) * jnp.sign(fc) <= zero
        new_a = jnp.where(active, jnp.where(keep_left, a, candidate), a)
        new_b = jnp.where(active, jnp.where(keep_left, candidate, b), b)
        new_fa = jnp.where(active, jnp.where(keep_left, half * fa, fc), fa)
        new_fb = jnp.where(active, jnp.where(keep_left, fc, half * fb), fb)
        return (
            i + jnp.asarray(1, dtype=jnp.int32),
            new_a,
            new_b,
            new_fa,
            new_fb,
            best_t_next,
            best_f_next,
        )

    _, _a_final, _b_final, _fa_final, _fb_final, t_best, f_best = jax.lax.while_loop(
        cond, body, init
    )
    return t_best, f_best, bracketed_in


# ── Adaptive driver ───────────────────────────────────────────────────


def trace_fieldline(
    spec: FieldlineTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FieldlineTracingResult:
    """Trace a single fieldline from ``y0`` for ``spec.tmax`` upstream-time units.

    Parameters
    ----------
    spec
        Tracing contract; see :class:`FieldlineTracingSpec`.
    y0
        Initial Cartesian position ``[3]``. Treated as float64.
    magnetic_field_fn
        JAX-traceable callable mapping a Cartesian point ``[3]`` to the
        magnetic field ``B(x)`` of shape ``[3]``.
    phis
        Optional 1-D array of target ``phi`` values in ``[0, 2*pi)``.
        Each detected crossing is appended to the result's ``phi_hits``
        buffer with ``idx == i``. Pass ``None`` (default) to disable
        phi-plane recording.
    stopping_criteria
        Tuple of JAX-side stopping criterion dataclasses (see
        :class:`MinRStoppingCriterion`, :class:`MaxRStoppingCriterion`,
        :class:`MinZStoppingCriterion`, :class:`MaxZStoppingCriterion`,
        :class:`ToroidalTransitStoppingCriterion`,
        :class:`IterStoppingCriterion`,
        :class:`MinToroidalFluxStoppingCriterion`,
        :class:`MaxToroidalFluxStoppingCriterion`). When multiple
        criteria fire on the same accepted step, the first matching
        criterion in iteration order wins; ``status`` then equals
        ``-1 - i`` reflecting that index.

    Returns
    -------
    result
        :class:`FieldlineTracingResult` with a padded
        ``(max_steps + 1, 4)`` trajectory, a mask, an accepted-step
        count, an exit status, ``t_final``, and the phi-crossing
        buffer.
    """

    dtype = jnp.float64
    y0_arr = jnp.asarray(y0, dtype=dtype).reshape((3,))
    tmax = jnp.asarray(spec.tmax, dtype=dtype)
    rtol = jnp.asarray(spec.rtol, dtype=dtype)
    atol = jnp.asarray(spec.atol, dtype=dtype)
    dtmax = jnp.asarray(spec.dtmax, dtype=dtype)
    t0 = jnp.asarray(0.0, dtype=dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    rhs = fieldline_rhs(magnetic_field_fn)
    h0 = _initial_step_size(t0, tmax, dtmax, _FIELDLINE_INITIAL_STEP_FRACTION)
    k0 = rhs(t0, y0_arr)
    one = jnp.asarray(1.0, dtype=dtype)

    # Pre-allocate the trajectory carry. Row 0 holds the initial state;
    # rows 1..max_steps fill in as accepted steps occur. Padding rows
    # at the end of the run get the final accepted state.
    traj = jnp.zeros((max_steps + 1, 4), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0_arr)
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)

    # Phi-plane crossing buffer. Each row is ``[t_hit, idx, x, y, z]``.
    phi_hits_buf = jnp.zeros((max_phi_hits, 5), dtype=dtype)
    phi_hits_count_init = jnp.asarray(0, dtype=jnp.int32)

    if phis is None:
        phis_arr = jnp.zeros((0,), dtype=dtype)
    else:
        phis_arr = jnp.asarray(phis, dtype=dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    # Initial unwrapped phi seed (C++ tracing.cpp uses pi).
    phi_init = _continuous_phi(
        y0_arr[0], y0_arr[1], jnp.asarray(np.pi, dtype=dtype), dtype
    )

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        jnp.asarray(0, dtype=jnp.int32),  # accepted_count
        t0,
        y0_arr,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        phi_init,  # running phi_last
        phi_init,  # transit criterion baseline, set on first accepted step
        jnp.asarray(0, dtype=jnp.int32),  # status_event (criterion idx)
        jnp.asarray(False),  # stop flag
    )

    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)
    max_phi_hits_i32 = jnp.asarray(max_phi_hits, dtype=jnp.int32)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)

    def cond(carry):
        (
            step_count,
            accepted_count,
            t,
            _y,
            _h,
            _k,
            _traj,
            _mask,
            _phi_hits,
            _phi_count,
            _phi_last,
            _phi_init,
            _status_event,
            stop,
        ) = carry
        not_done = t < tmax
        budget_ok = step_count < max_steps_i32
        accepted_ok = accepted_count < max_steps_i32
        not_stopped = jnp.logical_not(stop)
        return jnp.logical_and(
            not_done,
            jnp.logical_and(jnp.logical_and(budget_ok, accepted_ok), not_stopped),
        )

    def body(carry):
        (
            step_count,
            accepted_count,
            t,
            y,
            h,
            k_first,
            traj,
            mask,
            phi_hits_in,
            phi_hits_count_in,
            phi_last,
            phi_init,
            status_event,
            _stop,
        ) = carry
        # Clamp step to not overshoot tmax or the upstream quarter-turn ceiling.
        h_clamped = _clamp_step_to_domain(h, t, tmax, dtmax)
        y_new, y_err, k7 = dopri5_step(rhs, t, y, h_clamped, k_first)
        err = _error_norm(y_err, y, y_new, rtol, atol)
        err_safe = jnp.where(jnp.isfinite(err), err, jnp.asarray(jnp.inf, dtype=dtype))
        accepted = err_safe <= one
        factor = jnp.where(
            err_safe > jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(_SAFETY, dtype=dtype)
            * jnp.power(err_safe, jnp.asarray(-_DOPRI5_EXP, dtype=dtype)),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        factor = jnp.clip(
            factor,
            jnp.asarray(_MIN_FACTOR, dtype=dtype),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        h_next = h_clamped * factor
        t_accepted = _accepted_step_time(t, h_clamped, tmax)
        t_next = jnp.where(accepted, t_accepted, t)
        y_next = jnp.where(accepted, y_new, y)
        k_next = jnp.where(accepted, k7, k_first)

        # ── Phi-plane crossing detection on accepted steps ──
        phi_current = _continuous_phi(y_new[0], y_new[1], phi_last, dtype)

        def state_at_fraction(s):
            """Sub-step DOPRI5 from ``(t, y)`` with step ``s * h_clamped``.

            Re-runs a fresh DOPRI5 step from the prior accepted state
            so the returned state has 5th-order RK accuracy rather than
            the O(h) error of a linear interpolant. The FSAL value
            ``k_first`` is reused as the leading-stage derivative.
            """
            h_sub = s * h_clamped
            y_sub, _err, _k7 = dopri5_step(rhs, t, y, h_sub, k_first)
            return y_sub

        def scan_phis(args):
            hits_in, count_in, phi_last_in, phi_curr_in = args
            for i in range(num_phis):
                phi_target = phis_arr[i]
                # Detect a crossing of ``phi_target + k*2*pi`` for some k.
                fl_last = jnp.floor((phi_last_in - phi_target) / two_pi)
                fl_curr = jnp.floor((phi_curr_in - phi_target) / two_pi)
                crossed = fl_last != fl_curr
                # Pick integer offset so phi_shift lies inside the
                # ``[phi_last, phi_current]`` interval.
                fak = jnp.round(
                    (
                        (phi_last_in + phi_curr_in) / jnp.asarray(2.0, dtype=dtype)
                        - phi_target
                    )
                    / two_pi
                )
                phi_shift = fak * two_pi + phi_target

                def diff_at(s, phi_last_in=phi_last_in, phi_shift=phi_shift):
                    pos = state_at_fraction(s)
                    return (
                        _continuous_phi(pos[0], pos[1], phi_last_in, dtype) - phi_shift
                    )

                f_left = diff_at(jnp.asarray(0.0, dtype=dtype))
                f_right = diff_at(jnp.asarray(1.0, dtype=dtype))
                bracket_atol = jnp.asarray(1.0e-15, dtype=dtype)
                s_root, _f_root, _bracketed = bracket_root_jax(
                    diff_at,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(1.0, dtype=dtype),
                    f_left,
                    f_right,
                    max_root_iters,
                    bracket_atol,
                )
                t_root = t + s_root * h_clamped
                pos_root = state_at_fraction(s_root)
                hit_row = jnp.stack(
                    [
                        t_root,
                        jnp.asarray(float(i), dtype=dtype),
                        pos_root[0],
                        pos_root[1],
                        pos_root[2],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    crossed,
                    hit_row,
                    max_phi_hits_i32,
                )
            return hits_in, count_in

        phi_hits_after, phi_count_after = jax.lax.cond(
            accepted,
            scan_phis,
            lambda args: (args[0], args[1]),
            operand=(phi_hits_in, phi_hits_count_in, phi_last, phi_current),
        )

        # ── Stopping criteria check on accepted state ──
        first_accepted_step = accepted_count == jnp.asarray(0, dtype=jnp.int32)
        phi_init_for_criteria = jnp.where(
            first_accepted_step,
            phi_current,
            phi_init,
        )

        def apply_criteria(args):
            (
                hits_in,
                count_in,
                status_in,
                stop_in,
                iter_count_in,
                phi_curr_in,
                phi_init_in,
            ) = args
            for i, criterion in enumerate(stopping_criteria):
                pred = _stopping_criterion_should_stop(
                    criterion,
                    y_next[0],
                    y_next[1],
                    y_next[2],
                    iter_count_in,
                    phi_curr_in,
                    phi_init_in,
                    dtype,
                )
                fires = jnp.logical_and(jnp.logical_not(stop_in), pred)
                idx_val = jnp.asarray(-1 - i, dtype=jnp.int32)
                hit_row = jnp.stack(
                    [
                        t_next,
                        jnp.asarray(float(-1 - i), dtype=dtype),
                        y_next[0],
                        y_next[1],
                        y_next[2],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    fires,
                    hit_row,
                    max_phi_hits_i32,
                )
                status_in = jnp.where(fires, idx_val, status_in)
                stop_in = jnp.logical_or(stop_in, fires)
            return hits_in, count_in, status_in, stop_in

        iter_count_post = step_count + jnp.asarray(1, dtype=jnp.int32)

        (
            phi_hits_after,
            phi_count_after,
            status_after,
            stop_after,
        ) = jax.lax.cond(
            accepted,
            apply_criteria,
            lambda args: (args[0], args[1], args[2], args[3]),
            operand=(
                phi_hits_after,
                phi_count_after,
                status_event,
                jnp.asarray(False),
                iter_count_post,
                phi_current,
                phi_init_for_criteria,
            ),
        )

        # Update running phi_last only on accepted steps (matches C++).
        phi_last_next = jnp.where(accepted, phi_current, phi_last)
        phi_init_next = jnp.where(
            jnp.logical_and(accepted, first_accepted_step),
            phi_current,
            phi_init,
        )
        traj_next, mask_next, accepted_next = _record_trajectory_row(
            traj,
            mask,
            accepted_count,
            t_next,
            y_next,
            _should_record_accepted_step(accepted, stop_after),
        )

        return (
            step_count + jnp.asarray(1, dtype=jnp.int32),
            accepted_next,
            t_next,
            y_next,
            h_next,
            k_next,
            traj_next,
            mask_next,
            phi_hits_after,
            phi_count_after,
            phi_last_next,
            phi_init_next,
            status_after,
            stop_after,
        )

    (
        _step_count,
        accepted_count,
        t_final,
        y_final,
        _h_final,
        _k_final,
        traj_final,
        mask_final,
        phi_hits_final,
        phi_hits_count_final,
        _phi_last_final,
        _phi_init_final,
        status_event_final,
        stop_at_exit,
    ) = jax.lax.while_loop(cond, body, init_carry)

    # Pad unused rows with the final accepted state so downstream code
    # that ignores the mask still sees a valid (constant-extension)
    # trajectory.
    last_row = jnp.concatenate(
        [jnp.asarray([t_final], dtype=dtype), y_final.reshape((3,))]
    )

    def fill_padding(idx, traj_carry):
        row_active = mask_final[idx]
        return jax.lax.cond(
            row_active,
            lambda c: c,
            lambda c: c.at[idx].set(last_row),
            operand=traj_carry,
        )

    traj_padded = jax.lax.fori_loop(0, max_steps + 1, fill_padding, traj_final)

    eps_t = jnp.asarray(1.0e-12, dtype=dtype) * jnp.maximum(
        jnp.abs(tmax), jnp.asarray(1.0, dtype=dtype)
    )
    reached = (tmax - t_final) <= eps_t
    # Status priority: a stopping criterion (status_event_final < 0)
    # wins over budget / reached state.
    status_normal = jnp.where(
        reached,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    return FieldlineTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_final,
        phi_hits_count=phi_hits_count_final,
    )


def trace_fieldlines_batched(
    spec: FieldlineTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FieldlineTracingResult:
    """Trace a batch of fieldlines with one vmapped JAX integration graph.

    ``max_steps``, event-buffer sizes, tolerances, and stopping criteria
    are shared across the batch. ``dtmaxs`` is per lane because the
    upstream quarter-turn step cap depends on the initial radius and
    field strength.
    """

    y0s_arr = jnp.asarray(y0s, dtype=jnp.float64).reshape((-1, 3))
    dtmaxs_arr = jnp.asarray(dtmaxs, dtype=jnp.float64).reshape((-1,))

    def trace_one(y0: jax.Array, dtmax: jax.Array) -> FieldlineTracingResult:
        return trace_fieldline(
            replace(spec, dtmax=dtmax),
            y0,
            magnetic_field_fn,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    return jax.vmap(trace_one)(y0s_arr, dtmaxs_arr)


# ── Guiding-centre vacuum RHS (4-state Cartesian) ─────────────────────


@dataclass(frozen=True)
class GuidingCenterTracingSpec:
    """Immutable contract for a single guiding-centre integration call.

    Parameters mirror :class:`FieldlineTracingSpec`. The state is 4-D
    ``(x, y, z, v_par)`` instead of the fieldline's 3-D position so the
    trajectory carry has shape ``(max_steps + 1, 5)`` (columns
    ``(t, x, y, z, v_par)``). The ``phi_hits`` buffer has shape
    ``(max_phi_hits, 6)`` — extra trailing column carries ``v_par``.
    """

    tmax: float
    rtol: float
    atol: float
    max_steps: int
    dtmax: float = np.inf
    max_root_iters: int = 60
    max_phi_hits: int = 128


jax.tree_util.register_dataclass(
    GuidingCenterTracingSpec,
    data_fields=["tmax", "rtol", "atol", "dtmax"],
    meta_fields=["max_steps", "max_root_iters", "max_phi_hits"],
)


@dataclass(frozen=True)
class GuidingCenterTracingResult:
    """Return payload for :func:`trace_guiding_center`.

    - ``trajectory`` — ``(max_steps + 1, 5)`` float64 array. Columns are
      ``(t, x, y, z, v_par)``. Rows ``[0 : steps_taken + 1]`` are
      populated with accepted states; subsequent rows are padded with
      the final accepted state.
    - ``mask`` — ``(max_steps + 1,)`` bool array. ``True`` for rows that
      correspond to genuine accepted steps; ``False`` for padding.
    - ``steps_taken`` — int32 scalar; count of *accepted* steps the loop
      executed. Excludes the initial-state row.
    - ``status`` — int32 scalar. ``0`` for normal exit (``t >= tmax``),
      ``1`` for max-step-cap exhaustion before reaching ``tmax``,
      ``-1 - i`` when stopping criterion ``i`` fired.
    - ``t_final`` — float64 scalar; ``trajectory[steps_taken, 0]``.
    - ``phi_hits`` — ``(max_phi_hits, 6)`` float64 array. Columns are
      ``[t_hit, idx, x, y, z, v_par]``. ``idx >= 0`` denotes a phi-plane
      crossing for ``phis[int(idx)]``; ``idx < 0`` denotes stopping
      criterion ``-1 - int(idx)`` firing.
    - ``phi_hits_count`` — int32 scalar; total detected event count.
      Values greater than ``max_phi_hits`` mean the fixed buffer holds
      a truncated prefix.
    """

    trajectory: jax.Array
    mask: jax.Array
    steps_taken: jax.Array
    status: jax.Array
    t_final: jax.Array
    phi_hits: jax.Array
    phi_hits_count: jax.Array


jax.tree_util.register_dataclass(
    GuidingCenterTracingResult,
    data_fields=[
        "trajectory",
        "mask",
        "steps_taken",
        "status",
        "t_final",
        "phi_hits",
        "phi_hits_count",
    ],
    meta_fields=[],
)


def guiding_center_vacuum_rhs(
    magnetic_field_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    m: float,
    q: float,
    mu: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return ``rhs(t, y) -> dy/dt`` for the 4-state vacuum guiding-centre ODE.

    State is ``y = (x, y, z, v_par)``. The drift-kinetic equations
    (matching the upstream ``GuidingCenterVacuumRHS::operator()`` in
    ``simsoptpp/tracing.cpp``) are

    .. math::

       \dot{\mathbf{x}} &= \frac{v_\parallel}{|B|}\, \mathbf{B}
            + \frac{m}{q\, |B|^3}\left(\tfrac{1}{2}v_\perp^2
                + v_\parallel^2\right) \mathbf{B} \times \nabla |B|, \\
       \dot{v}_\parallel &= -\frac{\mu}{|B|}\, \mathbf{B} \cdot \nabla |B|,

    where :math:`v_\perp^2 = 2 \mu |B|`. The gradient :math:`\nabla |B|`
    is derived from the supplied ``dB_by_dX`` tensor with the
    SIMSOPT-wide convention ``dB_by_dX[j, l] = \partial_j B_l`` (axis 0
    is the derivative direction, axis 1 is the field component); the
    chain rule gives :math:`\partial_j |B| = B_l \, \partial_j B_l /
    |B|`. Upstream's ``MagneticField::GradAbsB_ref`` follows the same
    convention.

    Parameters
    ----------
    magnetic_field_fn
        JAX-traceable callable mapping a Cartesian point ``[3]`` to the
        pair ``(B, dB_by_dX)`` where ``B`` has shape ``[3]`` and
        ``dB_by_dX`` has shape ``[3, 3]``.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
        Captured at closure construction; not mutated thereafter.
    """

    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t  # Field is autonomous; signature kept for ODE-driver shape.
        position = y[:3]
        v_par = y[3]
        B_raw, dB_by_dX_raw = magnetic_field_fn(position)
        B = jnp.asarray(B_raw, dtype=y.dtype).reshape((3,))
        dB_by_dX = jnp.asarray(dB_by_dX_raw, dtype=y.dtype).reshape((3, 3))
        abs_B = jnp.linalg.norm(B)
        # GradAbsB_j = B_l * dB_l/dx_j / |B|   (upstream GradAbsB_ref).
        grad_abs_B = jnp.einsum("l,jl->j", B, dB_by_dX) / abs_B
        # B x grad|B|
        B_cross_grad_abs_B = jnp.cross(B, grad_abs_B)
        v_perp2 = jnp.asarray(2.0, dtype=y.dtype) * mu_arr * abs_B
        fak1 = v_par / abs_B
        fak2 = (
            m_arr
            / (q_arr * abs_B**3)
            * (jnp.asarray(0.5, dtype=y.dtype) * v_perp2 + v_par * v_par)
        )
        dposition = fak1 * B + fak2 * B_cross_grad_abs_B
        dv_par = -mu_arr * jnp.dot(B, grad_abs_B) / abs_B
        return jnp.stack(
            [dposition[0], dposition[1], dposition[2], dv_par],
        )

    return rhs


def trace_guiding_center(
    spec: GuidingCenterTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    m: float,
    q: float,
    mu: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> GuidingCenterTracingResult:
    """Trace a guiding-centre orbit from ``y0`` for ``spec.tmax`` seconds.

    Parameters
    ----------
    spec
        Tracing contract; see :class:`GuidingCenterTracingSpec`.
    y0
        Initial state ``[x, y, z, v_par]`` (length 4). Treated as
        float64.
    magnetic_field_fn
        JAX-traceable callable mapping a Cartesian point ``[3]`` to the
        pair ``(B, dB_by_dX)`` where ``B`` has shape ``[3]`` and
        ``dB_by_dX`` has shape ``[3, 3]``. See
        :func:`guiding_center_vacuum_rhs` for the convention.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    phis
        Optional 1-D array of target ``phi`` values. See
        :func:`trace_fieldline` for the contract.
    stopping_criteria
        Tuple of JAX-side stopping criterion dataclasses. See
        :func:`trace_fieldline` for the contract.

    Returns
    -------
    result
        :class:`GuidingCenterTracingResult` with a padded
        ``(max_steps + 1, 5)`` trajectory, a mask, an accepted-step
        count, an exit status, ``t_final``, and the phi-crossing
        buffer.
    """

    dtype = jnp.float64
    y0_arr = jnp.asarray(y0, dtype=dtype).reshape((4,))
    tmax = jnp.asarray(spec.tmax, dtype=dtype)
    rtol = jnp.asarray(spec.rtol, dtype=dtype)
    atol = jnp.asarray(spec.atol, dtype=dtype)
    dtmax = jnp.asarray(spec.dtmax, dtype=dtype)
    t0 = jnp.asarray(0.0, dtype=dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    rhs = guiding_center_vacuum_rhs(magnetic_field_fn, m, q, mu)
    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    k0 = rhs(t0, y0_arr)
    one = jnp.asarray(1.0, dtype=dtype)

    # Pre-allocate the trajectory carry with columns (t, x, y, z, v_par).
    # Row 0 holds the initial state; rows 1..max_steps fill in as
    # accepted steps occur. Padding rows at the end of the run get the
    # final accepted state.
    traj = jnp.zeros((max_steps + 1, 5), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0_arr)
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)

    phi_hits_buf = jnp.zeros((max_phi_hits, 6), dtype=dtype)
    phi_hits_count_init = jnp.asarray(0, dtype=jnp.int32)

    if phis is None:
        phis_arr = jnp.zeros((0,), dtype=dtype)
    else:
        phis_arr = jnp.asarray(phis, dtype=dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    phi_init = _continuous_phi(
        y0_arr[0], y0_arr[1], jnp.asarray(np.pi, dtype=dtype), dtype
    )

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        jnp.asarray(0, dtype=jnp.int32),  # accepted_count
        t0,
        y0_arr,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        phi_init,
        phi_init,
        jnp.asarray(0, dtype=jnp.int32),  # status_event
        jnp.asarray(False),
    )

    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)
    max_phi_hits_i32 = jnp.asarray(max_phi_hits, dtype=jnp.int32)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)

    def cond(carry):
        (
            step_count,
            accepted_count,
            t,
            _y,
            _h,
            _k,
            _traj,
            _mask,
            _phi_hits,
            _phi_count,
            _phi_last,
            _phi_init,
            _status_event,
            stop,
        ) = carry
        not_done = t < tmax
        budget_ok = step_count < max_steps_i32
        accepted_ok = accepted_count < max_steps_i32
        not_stopped = jnp.logical_not(stop)
        return jnp.logical_and(
            not_done,
            jnp.logical_and(jnp.logical_and(budget_ok, accepted_ok), not_stopped),
        )

    def body(carry):
        (
            step_count,
            accepted_count,
            t,
            y,
            h,
            k_first,
            traj,
            mask,
            phi_hits_in,
            phi_hits_count_in,
            phi_last,
            phi_init,
            status_event,
            _stop,
        ) = carry
        h_clamped = _clamp_step_to_domain(h, t, tmax, dtmax)
        y_new, y_err, k7 = dopri5_step(rhs, t, y, h_clamped, k_first)
        err = _error_norm(y_err, y, y_new, rtol, atol)
        err_safe = jnp.where(jnp.isfinite(err), err, jnp.asarray(jnp.inf, dtype=dtype))
        accepted = err_safe <= one
        factor = jnp.where(
            err_safe > jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(_SAFETY, dtype=dtype)
            * jnp.power(err_safe, jnp.asarray(-_DOPRI5_EXP, dtype=dtype)),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        factor = jnp.clip(
            factor,
            jnp.asarray(_MIN_FACTOR, dtype=dtype),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        h_next = h_clamped * factor
        t_accepted = _accepted_step_time(t, h_clamped, tmax)
        t_next = jnp.where(accepted, t_accepted, t)
        y_next = jnp.where(accepted, y_new, y)
        k_next = jnp.where(accepted, k7, k_first)

        # ── Phi-plane crossing detection on accepted steps ──
        phi_current = _continuous_phi(y_new[0], y_new[1], phi_last, dtype)

        def state_at_fraction(s):
            """Sub-step DOPRI5 from ``(t, y)`` with step ``s * h_clamped``.

            Re-runs a fresh DOPRI5 step from the prior accepted state
            so the returned state has 5th-order RK accuracy rather than
            the O(h) error of a linear interpolant. The FSAL value
            ``k_first`` is reused as the leading-stage derivative.
            """
            h_sub = s * h_clamped
            y_sub, _err, _k7 = dopri5_step(rhs, t, y, h_sub, k_first)
            return y_sub

        def scan_phis(args):
            hits_in, count_in, phi_last_in, phi_curr_in = args
            for i in range(num_phis):
                phi_target = phis_arr[i]
                fl_last = jnp.floor((phi_last_in - phi_target) / two_pi)
                fl_curr = jnp.floor((phi_curr_in - phi_target) / two_pi)
                crossed = fl_last != fl_curr
                fak = jnp.round(
                    (
                        (phi_last_in + phi_curr_in) / jnp.asarray(2.0, dtype=dtype)
                        - phi_target
                    )
                    / two_pi
                )
                phi_shift = fak * two_pi + phi_target

                def diff_at(s, phi_last_in=phi_last_in, phi_shift=phi_shift):
                    pos = state_at_fraction(s)
                    return (
                        _continuous_phi(pos[0], pos[1], phi_last_in, dtype) - phi_shift
                    )

                f_left = diff_at(jnp.asarray(0.0, dtype=dtype))
                f_right = diff_at(jnp.asarray(1.0, dtype=dtype))
                bracket_atol = jnp.asarray(1.0e-15, dtype=dtype)
                s_root, _f_root, _bracketed = bracket_root_jax(
                    diff_at,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(1.0, dtype=dtype),
                    f_left,
                    f_right,
                    max_root_iters,
                    bracket_atol,
                )
                t_root = t + s_root * h_clamped
                state_root = state_at_fraction(s_root)
                hit_row = jnp.stack(
                    [
                        t_root,
                        jnp.asarray(float(i), dtype=dtype),
                        state_root[0],
                        state_root[1],
                        state_root[2],
                        state_root[3],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    crossed,
                    hit_row,
                    max_phi_hits_i32,
                )
            return hits_in, count_in

        phi_hits_after, phi_count_after = jax.lax.cond(
            accepted,
            scan_phis,
            lambda args: (args[0], args[1]),
            operand=(phi_hits_in, phi_hits_count_in, phi_last, phi_current),
        )

        first_accepted_step = accepted_count == jnp.asarray(0, dtype=jnp.int32)
        phi_init_for_criteria = jnp.where(
            first_accepted_step,
            phi_current,
            phi_init,
        )

        def apply_criteria(args):
            (
                hits_in,
                count_in,
                status_in,
                stop_in,
                iter_count_in,
                phi_curr_in,
                phi_init_in,
            ) = args
            for i, criterion in enumerate(stopping_criteria):
                pred = _stopping_criterion_should_stop(
                    criterion,
                    y_next[0],
                    y_next[1],
                    y_next[2],
                    iter_count_in,
                    phi_curr_in,
                    phi_init_in,
                    dtype,
                )
                fires = jnp.logical_and(jnp.logical_not(stop_in), pred)
                idx_val = jnp.asarray(-1 - i, dtype=jnp.int32)
                hit_row = jnp.stack(
                    [
                        t_next,
                        jnp.asarray(float(-1 - i), dtype=dtype),
                        y_next[0],
                        y_next[1],
                        y_next[2],
                        y_next[3],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    fires,
                    hit_row,
                    max_phi_hits_i32,
                )
                status_in = jnp.where(fires, idx_val, status_in)
                stop_in = jnp.logical_or(stop_in, fires)
            return hits_in, count_in, status_in, stop_in

        iter_count_post = step_count + jnp.asarray(1, dtype=jnp.int32)

        (
            phi_hits_after,
            phi_count_after,
            status_after,
            stop_after,
        ) = jax.lax.cond(
            accepted,
            apply_criteria,
            lambda args: (args[0], args[1], args[2], args[3]),
            operand=(
                phi_hits_after,
                phi_count_after,
                status_event,
                jnp.asarray(False),
                iter_count_post,
                phi_current,
                phi_init_for_criteria,
            ),
        )

        phi_last_next = jnp.where(accepted, phi_current, phi_last)
        phi_init_next = jnp.where(
            jnp.logical_and(accepted, first_accepted_step),
            phi_current,
            phi_init,
        )
        traj_next, mask_next, accepted_next = _record_trajectory_row(
            traj,
            mask,
            accepted_count,
            t_next,
            y_next,
            _should_record_accepted_step(accepted, stop_after),
        )

        return (
            step_count + jnp.asarray(1, dtype=jnp.int32),
            accepted_next,
            t_next,
            y_next,
            h_next,
            k_next,
            traj_next,
            mask_next,
            phi_hits_after,
            phi_count_after,
            phi_last_next,
            phi_init_next,
            status_after,
            stop_after,
        )

    (
        _step_count,
        accepted_count,
        t_final,
        y_final,
        _h_final,
        _k_final,
        traj_final,
        mask_final,
        phi_hits_final,
        phi_hits_count_final,
        _phi_last_final,
        _phi_init_final,
        status_event_final,
        stop_at_exit,
    ) = jax.lax.while_loop(cond, body, init_carry)

    last_row = jnp.concatenate(
        [jnp.asarray([t_final], dtype=dtype), y_final.reshape((4,))]
    )

    def fill_padding(idx, traj_carry):
        row_active = mask_final[idx]
        return jax.lax.cond(
            row_active,
            lambda c: c,
            lambda c: c.at[idx].set(last_row),
            operand=traj_carry,
        )

    traj_padded = jax.lax.fori_loop(0, max_steps + 1, fill_padding, traj_final)

    eps_t = jnp.asarray(1.0e-12, dtype=dtype) * jnp.maximum(
        jnp.abs(tmax), jnp.asarray(1.0, dtype=dtype)
    )
    reached = (tmax - t_final) <= eps_t
    status_normal = jnp.where(
        reached,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    return GuidingCenterTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_final,
        phi_hits_count=phi_hits_count_final,
    )


def trace_guiding_centers_batched(
    spec: GuidingCenterTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    mus: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    m: float,
    q: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> GuidingCenterTracingResult:
    """Trace Cartesian guiding-centre orbits with one vmapped JAX graph."""

    y0s_arr = jnp.asarray(y0s, dtype=jnp.float64).reshape((-1, 4))
    dtmaxs_arr = jnp.asarray(dtmaxs, dtype=jnp.float64).reshape((-1,))
    mus_arr = jnp.asarray(mus, dtype=jnp.float64).reshape((-1,))

    def trace_one(
        y0: jax.Array, dtmax: jax.Array, mu: jax.Array
    ) -> GuidingCenterTracingResult:
        return trace_guiding_center(
            replace(spec, dtmax=dtmax),
            y0,
            magnetic_field_fn,
            m=m,
            q=q,
            mu=mu,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    return jax.vmap(trace_one)(y0s_arr, dtmaxs_arr, mus_arr)


# ── Shared 4-state DOPRI5 adaptive driver ─────────────────────────────


def _run_dopri5_4state(
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    y0: jax.Array,
    tmax: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
    dtmax: jax.Array,
    max_steps: int,
    max_phi_hits: int = 1,
) -> GuidingCenterTracingResult:
    """Run the DOPRI5 + PI controller driver on a generic 4-state RHS.

    Factored out so the Boozer guiding-centre variants share the same
    adaptive-step machinery as :func:`trace_guiding_center`. The
    trajectory carry has columns ``(t, y0, y1, y2, y3)`` — i.e. the
    5-wide layout used by both the Cartesian guiding centre and the
    Boozer ``[s, theta, zeta, v_par]`` state. The Boozer path does not
    consume the phi-plane crossing buffer (the upstream
    ``trace_particles_boozer`` route uses ``zetas`` rather than
    Cartesian ``phis``); the buffer is emitted as an empty
    ``(max_phi_hits, 6)`` array so the result remains pytree-compatible
    with the Cartesian guiding-centre driver.
    """

    dtype = jnp.float64
    t0 = jnp.asarray(0.0, dtype=dtype)
    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    initial_axis_invalid = _boozer_axis_invalid(y0)
    k0 = jax.lax.cond(
        initial_axis_invalid,
        lambda _: jnp.zeros((4,), dtype=dtype),
        lambda _: rhs(t0, y0),
        operand=None,
    )
    one = jnp.asarray(1.0, dtype=dtype)

    traj = jnp.zeros((max_steps + 1, 5), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0)
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        jnp.asarray(0, dtype=jnp.int32),  # accepted_count
        t0,
        y0,
        h0,
        k0,
        traj,
        mask,
        jnp.where(
            initial_axis_invalid,
            jnp.asarray(_BOOZER_AXIS_STATUS, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        initial_axis_invalid,
    )

    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)

    def cond(carry):
        (
            step_count,
            accepted_count,
            t,
            _y,
            _h,
            _k,
            _traj,
            _mask,
            _status_event,
            stop,
        ) = carry
        not_done = t < tmax
        budget_ok = step_count < max_steps_i32
        accepted_ok = accepted_count < max_steps_i32
        not_stopped = jnp.logical_not(stop)
        return jnp.logical_and(
            not_done,
            jnp.logical_and(jnp.logical_and(budget_ok, accepted_ok), not_stopped),
        )

    def body(carry):
        (
            step_count,
            accepted_count,
            t,
            y,
            h,
            k_first,
            traj,
            mask,
            status_event,
            _stop,
        ) = carry
        h_clamped = _clamp_step_to_domain(h, t, tmax, dtmax)
        y_new, y_err, k7 = dopri5_step(rhs, t, y, h_clamped, k_first)
        err = _error_norm(y_err, y, y_new, rtol, atol)
        err_safe = jnp.where(jnp.isfinite(err), err, jnp.asarray(jnp.inf, dtype=dtype))
        accepted = err_safe <= one
        axis_invalid = jnp.logical_and(accepted, _boozer_axis_invalid(y_new))
        factor = jnp.where(
            err_safe > jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(_SAFETY, dtype=dtype)
            * jnp.power(err_safe, jnp.asarray(-_DOPRI5_EXP, dtype=dtype)),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        factor = jnp.clip(
            factor,
            jnp.asarray(_MIN_FACTOR, dtype=dtype),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        h_next = h_clamped * factor
        t_accepted = _accepted_step_time(t, h_clamped, tmax)
        t_next = jnp.where(accepted, t_accepted, t)
        y_next = jnp.where(accepted, y_new, y)
        k_next = jnp.where(accepted, k7, k_first)
        traj_next, mask_next, accepted_next = _record_trajectory_row(
            traj,
            mask,
            accepted_count,
            t_next,
            y_next,
            _should_record_accepted_step(accepted, axis_invalid),
        )
        status_next = jnp.where(
            axis_invalid,
            jnp.asarray(_BOOZER_AXIS_STATUS, dtype=jnp.int32),
            status_event,
        )
        return (
            step_count + jnp.asarray(1, dtype=jnp.int32),
            accepted_next,
            t_next,
            y_next,
            h_next,
            k_next,
            traj_next,
            mask_next,
            status_next,
            axis_invalid,
        )

    (
        _step_count,
        accepted_count,
        t_final,
        y_final,
        _h_final,
        _k_final,
        traj_final,
        mask_final,
        status_event_final,
        stop_at_exit,
    ) = jax.lax.while_loop(cond, body, init_carry)

    last_row = jnp.concatenate(
        [jnp.asarray([t_final], dtype=dtype), y_final.reshape((4,))]
    )

    def fill_padding(idx, traj_carry):
        row_active = mask_final[idx]
        return jax.lax.cond(
            row_active,
            lambda c: c,
            lambda c: c.at[idx].set(last_row),
            operand=traj_carry,
        )

    traj_padded = jax.lax.fori_loop(0, max_steps + 1, fill_padding, traj_final)

    eps_t = jnp.asarray(1.0e-12, dtype=dtype) * jnp.maximum(
        jnp.abs(tmax), jnp.asarray(1.0, dtype=dtype)
    )
    reached = (tmax - t_final) <= eps_t
    status_normal = jnp.where(
        reached,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    phi_hits_empty = jnp.zeros((max_phi_hits, 6), dtype=dtype)
    return GuidingCenterTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_empty,
        phi_hits_count=jnp.asarray(0, dtype=jnp.int32),
    )


# ── Boozer-coordinate guiding-centre RHS (4-state) ────────────────────


def _resolve_boozer_field_state(boozer_field):
    """Resolve a Boozer field-like object to its frozen pytree state + psi0.

    The Boozer RHS variants are pure functions of a single Boozer point
    ``(s, theta, zeta)`` and the immutable frozen-state pytree exposed
    by :class:`simsopt.field.boozermagneticfield_jax.BoozerRadialInterpolantJAX`.
    Routing through the mutable ``set_points`` API would break the
    JIT/while-loop contract (Python side-effects on cached arrays are
    not re-executed per iteration).

    Two input shapes are accepted:

    1. A ``BoozerRadialInterpolantJAX`` instance — the frozen state is
       pulled via ``boozer_field.frozen_state`` and ``psi0`` via
       ``boozer_field.psi0``. This is the shape used by the public
       :func:`trace_particles_boozer` JAX router (item 16 follow-up).
    2. A tuple ``(frozen_state, psi0)`` — used directly by unit tests
       and downstream consumers that want to assemble the RHS without
       owning a wrapper instance.

    Anything else raises :class:`TypeError`.
    """

    if isinstance(boozer_field, tuple) and len(boozer_field) == 2:
        return boozer_field[0], float(boozer_field[1])
    frozen = getattr(boozer_field, "frozen_state", None)
    psi0 = getattr(boozer_field, "psi0", None)
    if frozen is None or psi0 is None:
        raise TypeError(
            "guiding-centre Boozer RHS requires a "
            "BoozerRadialInterpolantJAX-shaped field exposing "
            "`frozen_state` and `psi0`, or a (frozen_state, psi0) "
            f"tuple; got {type(boozer_field).__name__}."
        )
    return frozen, float(psi0)


def _boozer_point_2d(y: jax.Array) -> jax.Array:
    """Reshape an ``(s, theta, zeta)`` 1-D state to the ``(1, 3)`` eval shape."""
    return jnp.asarray(y[:3], dtype=jnp.float64).reshape((1, 3))


def _boozer_scalar(value: jax.Array) -> jax.Array:
    """Squeeze a ``(1,)`` or ``(1, 1)`` Boozer-eval scalar to a JAX scalar."""
    return jnp.asarray(value, dtype=jnp.float64).reshape(-1)[0]


# The frozen-state -> evaluator family dispatch keys. The Boozer
# guiding-centre RHS factories consume the union of these twelve scalar
# evaluators (modB + its three first derivatives, K + its two angular
# derivatives, and the four scalar radial profiles G/I/iota with the
# two radial-derivative profiles dGds/dIds). Holding the key set in one
# tuple keeps the call-site contract uniform across the three RHS
# factories and is the SSOT for what each frozen-state branch must
# provide.
_BOOZER_RHS_EVAL_KEYS: tuple[str, ...] = (
    "modB",
    "dmodBds",
    "dmodBdtheta",
    "dmodBdzeta",
    "K",
    "dKdtheta",
    "dKdzeta",
    "G",
    "I",
    "iota",
    "dGds",
    "dIds",
)


def _interpolated_boozer_evaluator(name: str) -> Callable:
    eval_fn = _INTERP_EVALUATORS[name]

    def _eval(state: InterpolatedBoozerFieldFrozenState, point: jax.Array) -> jax.Array:
        return eval_fn(state, state.specs, point)

    return _eval


def _boozer_field_evaluators(state) -> dict[str, Callable]:
    """Return the set of evaluator callables matching the frozen-state type.

    This is a Python-static dispatch evaluated once at RHS-factory time,
    outside the JIT trace. The returned callables are bound to a
    particular state shape; the inner ``rhs(t, y)`` function captures
    them by closure and JAX traces a single homogeneous evaluator graph
    per call.

    Supported state types:

    - :class:`simsopt.jax_core.boozer_analytic.BoozerAnalyticFrozenState`
      — closed-form analytic evaluators from
      :mod:`simsopt.jax_core.boozer_analytic`.
    - :class:`simsopt.field.boozermagneticfield_jax.BoozerRadialInterpolantFrozenState`
      — spline + Fourier evaluators from
      :mod:`simsopt.field.boozermagneticfield_jax`.
    - :class:`simsopt.field.boozermagneticfield_jax.InterpolatedBoozerFieldFrozenState`
      — regular-grid Boozer scalar evaluators from the same module.

    Raises:
        TypeError: when the state type has no JAX evaluators registered.
    """
    if isinstance(state, BoozerAnalyticFrozenState):
        return {
            "modB": _analytic_modB,
            "dmodBds": _analytic_dmodBds,
            "dmodBdtheta": _analytic_dmodBdtheta,
            "dmodBdzeta": _analytic_dmodBdzeta,
            "K": _analytic_K,
            "dKdtheta": _analytic_dKdtheta,
            "dKdzeta": _analytic_dKdzeta,
            "G": _analytic_G,
            "I": _analytic_I,
            "iota": _analytic_iota,
            "dGds": _analytic_dGds,
            "dIds": _analytic_dIds,
        }
    if isinstance(state, BoozerRadialInterpolantFrozenState):
        return {
            "modB": _radial_modB,
            "dmodBds": _radial_dmodBds,
            "dmodBdtheta": _radial_dmodBdtheta,
            "dmodBdzeta": _radial_dmodBdzeta,
            "K": _radial_K,
            "dKdtheta": _radial_dKdtheta,
            "dKdzeta": _radial_dKdzeta,
            "G": _radial_G,
            "I": _radial_I,
            "iota": _radial_iota,
            "dGds": _radial_dGds,
            "dIds": _radial_dIds,
        }
    if isinstance(state, InterpolatedBoozerFieldFrozenState):
        return {
            key: _interpolated_boozer_evaluator(key) for key in _BOOZER_RHS_EVAL_KEYS
        }
    raise TypeError(
        f"No JAX RHS evaluators registered for frozen-state type "
        f"{type(state).__name__}. Supported types: "
        f"BoozerAnalyticFrozenState, BoozerRadialInterpolantFrozenState, "
        f"InterpolatedBoozerFieldFrozenState."
    )


def guiding_center_vacuum_boozer_rhs(
    boozer_field,
    m: float,
    q: float,
    mu: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return ``rhs(t, y) -> dy/dt`` for the 4-state vacuum-Boozer guiding centre.

    State is ``y = (s, theta, zeta, v_par)``. The equations of motion
    follow the upstream ``GuidingCenterVacuumBoozerRHS::operator()`` in
    ``simsoptpp/tracing.cpp``:

    .. math::

       \dot s &= -|B|_{,\theta}\, \mathrm{fak1} / (q\, \psi_0), \\
       \dot \theta &= |B|_{,s}\, \mathrm{fak1} / (q\, \psi_0)
            + \iota\, v_\parallel |B| / G, \\
       \dot \zeta &= v_\parallel\, |B| / G, \\
       \dot v_\parallel &= -(\iota\, |B|_{,\theta} + |B|_{,\zeta})\,
            \mu\, |B| / G,

    where ``fak1 = m v_par^2 / |B| + m * mu`` and
    ``v_perp^2 = 2 mu |B|``. This is the ``G =`` const., ``I = 0``,
    ``K = 0`` simplification.

    Parameters
    ----------
    boozer_field
        Either a :class:`simsopt.field.boozermagneticfield_jax.BoozerRadialInterpolantJAX`
        instance, a :class:`simsopt.field.boozermagneticfield_jax.BoozerAnalyticJAX`
        instance, or a ``(frozen_state, psi0)`` tuple. The RHS reads
        ``modB``, ``modB_derivs``, ``iota``, ``G`` and ``psi0`` from
        the frozen state; the evaluator family is selected at
        factory-call time by :func:`_boozer_field_evaluators`.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    """

    state, psi0_host = _resolve_boozer_field_state(boozer_field)
    evals = _boozer_field_evaluators(state)
    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    psi0 = jnp.asarray(psi0_host, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t
        v_par = y[3]
        point = _boozer_point_2d(y)
        modB = _boozer_scalar(evals["modB"](state, point))
        dmodBds = _boozer_scalar(evals["dmodBds"](state, point))
        dmodBdtheta = _boozer_scalar(evals["dmodBdtheta"](state, point))
        dmodBdzeta = _boozer_scalar(evals["dmodBdzeta"](state, point))
        G = _boozer_scalar(evals["G"](state, point))
        iota = _boozer_scalar(evals["iota"](state, point))

        fak1 = m_arr * v_par * v_par / modB + m_arr * mu_arr

        ds = -dmodBdtheta * fak1 / (q_arr * psi0)
        dtheta = dmodBds * fak1 / (q_arr * psi0) + iota * v_par * modB / G
        dzeta = v_par * modB / G
        dv_par = -(iota * dmodBdtheta + dmodBdzeta) * mu_arr * modB / G
        return jnp.stack([ds, dtheta, dzeta, dv_par])

    return rhs


def guiding_center_no_k_boozer_rhs(
    boozer_field,
    m: float,
    q: float,
    mu: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return ``rhs(t, y) -> dy/dt`` for the 4-state ``no_K=True`` Boozer GC.

    State is ``y = (s, theta, zeta, v_par)``. The equations of motion
    follow ``GuidingCenterNoKBoozerRHS::operator()`` in
    ``simsoptpp/tracing.cpp``. The non-vacuum case uses ``G(s)`` and
    ``I(s)`` profiles but assumes ``K(s, theta, zeta) = 0``.

    Parameters
    ----------
    boozer_field
        Either a :class:`simsopt.field.boozermagneticfield_jax.BoozerRadialInterpolantJAX`
        instance, a :class:`simsopt.field.boozermagneticfield_jax.BoozerAnalyticJAX`
        instance, or a ``(frozen_state, psi0)`` tuple. The RHS reads
        ``modB``, ``modB_derivs``, ``iota``, ``G``, ``I``, ``dGds``,
        ``dIds`` and ``psi0`` from the frozen state; the evaluator
        family is selected at factory-call time by
        :func:`_boozer_field_evaluators`.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    """

    state, psi0_host = _resolve_boozer_field_state(boozer_field)
    evals = _boozer_field_evaluators(state)
    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    psi0 = jnp.asarray(psi0_host, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t
        v_par = y[3]
        point = _boozer_point_2d(y)
        modB = _boozer_scalar(evals["modB"](state, point))
        dmodBds = _boozer_scalar(evals["dmodBds"](state, point))
        dmodBdtheta = _boozer_scalar(evals["dmodBdtheta"](state, point))
        dmodBdzeta = _boozer_scalar(evals["dmodBdzeta"](state, point))
        G = _boozer_scalar(evals["G"](state, point))
        I_val = _boozer_scalar(evals["I"](state, point))
        iota = _boozer_scalar(evals["iota"](state, point))
        dGds = _boozer_scalar(evals["dGds"](state, point))
        dIds = _boozer_scalar(evals["dIds"](state, point))
        dGdpsi = dGds / psi0
        dIdpsi = dIds / psi0
        dmodBdpsi = dmodBds / psi0

        fak1 = m_arr * v_par * v_par / modB + m_arr * mu_arr
        D = (
            (q_arr + m_arr * v_par * dIdpsi / modB) * G
            - (-q_arr * iota + m_arr * v_par * dGdpsi / modB) * I_val
        ) / iota

        ds = (I_val * dmodBdzeta - G * dmodBdtheta) * fak1 / (D * iota * psi0)
        dtheta = (
            G * dmodBdpsi * fak1
            - (-q_arr * iota + m_arr * v_par * dGdpsi / modB) * v_par * modB
        ) / (D * iota)
        dzeta = (
            (q_arr + m_arr * v_par * dIdpsi / modB) * v_par * modB
            - dmodBdpsi * fak1 * I_val
        ) / (D * iota)
        # Upstream uses dv_par = -(mu/v_par) * (dmodBdpsi*ds*psi0
        #   + dmodBdtheta*dtheta + dmodBdzeta*dzeta), expressing energy
        # conservation. We reproduce that line exactly.
        dv_par = -(mu_arr / v_par) * (
            dmodBdpsi * ds * psi0 + dmodBdtheta * dtheta + dmodBdzeta * dzeta
        )
        return jnp.stack([ds, dtheta, dzeta, dv_par])

    return rhs


def guiding_center_boozer_rhs(
    boozer_field,
    m: float,
    q: float,
    mu: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return ``rhs(t, y) -> dy/dt`` for the full 4-state Boozer GC.

    State is ``y = (s, theta, zeta, v_par)``. The equations of motion
    follow ``GuidingCenterBoozerRHS::operator()`` in
    ``simsoptpp/tracing.cpp`` — the non-vacuum, ``K != 0`` case with
    ``C``, ``F``, ``D`` algebraic coefficients folded in.

    Parameters
    ----------
    boozer_field
        Either a :class:`simsopt.field.boozermagneticfield_jax.BoozerRadialInterpolantJAX`
        instance, a :class:`simsopt.field.boozermagneticfield_jax.BoozerAnalyticJAX`
        instance, or a ``(frozen_state, psi0)`` tuple. The RHS reads
        ``modB``, ``modB_derivs``, ``K``, ``K_derivs``, ``iota``,
        ``G``, ``I``, ``dGds``, ``dIds`` and ``psi0`` from the frozen
        state; the evaluator family is selected at factory-call time
        by :func:`_boozer_field_evaluators`.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    """

    state, psi0_host = _resolve_boozer_field_state(boozer_field)
    evals = _boozer_field_evaluators(state)
    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    psi0 = jnp.asarray(psi0_host, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t
        v_par = y[3]
        point = _boozer_point_2d(y)
        modB = _boozer_scalar(evals["modB"](state, point))
        dmodBds = _boozer_scalar(evals["dmodBds"](state, point))
        dmodBdtheta = _boozer_scalar(evals["dmodBdtheta"](state, point))
        dmodBdzeta = _boozer_scalar(evals["dmodBdzeta"](state, point))
        K_val = _boozer_scalar(evals["K"](state, point))
        dKdtheta = _boozer_scalar(evals["dKdtheta"](state, point))
        dKdzeta = _boozer_scalar(evals["dKdzeta"](state, point))
        G = _boozer_scalar(evals["G"](state, point))
        I_val = _boozer_scalar(evals["I"](state, point))
        iota = _boozer_scalar(evals["iota"](state, point))
        dGds = _boozer_scalar(evals["dGds"](state, point))
        dIds = _boozer_scalar(evals["dIds"](state, point))
        dGdpsi = dGds / psi0
        dIdpsi = dIds / psi0
        dmodBdpsi = dmodBds / psi0

        fak1 = m_arr * v_par * v_par / modB + m_arr * mu_arr
        # Upstream `tracing.cpp` C and F definitions:
        #   C = - m v_par (dK/dzeta - G')/|B| - q iota
        #   F = - m v_par (dK/dtheta - I')/|B| + q
        C = -m_arr * v_par * (dKdzeta - dGdpsi) / modB - q_arr * iota
        F = -m_arr * v_par * (dKdtheta - dIdpsi) / modB + q_arr
        D = (F * G - C * I_val) / iota

        ds = (I_val * dmodBdzeta - G * dmodBdtheta) * fak1 / (D * iota * psi0)
        dtheta = (
            G * dmodBdpsi * fak1 - C * v_par * modB - K_val * fak1 * dmodBdzeta
        ) / (D * iota)
        dzeta = (
            F * v_par * modB - dmodBdpsi * fak1 * I_val + K_val * fak1 * dmodBdtheta
        ) / (D * iota)
        dv_par = -(mu_arr / v_par) * (
            dmodBdpsi * ds * psi0 + dmodBdtheta * dtheta + dmodBdzeta * dzeta
        )
        return jnp.stack([ds, dtheta, dzeta, dv_par])

    return rhs


def trace_guiding_center_boozer(
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
    """Trace a Boozer-coordinate guiding-centre orbit.

    Parameters
    ----------
    spec
        Tracing contract; see :class:`GuidingCenterTracingSpec`.
    y0
        Initial state ``[s, theta, zeta, v_par]`` (length 4). Treated
        as float64.
    boozer_field
        Either a ``BoozerRadialInterpolantJAX`` instance or a
        ``(frozen_state, psi0)`` tuple. See
        :func:`guiding_center_vacuum_boozer_rhs` /
        :func:`guiding_center_no_k_boozer_rhs` /
        :func:`guiding_center_boozer_rhs` for the field-side contract.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    mode
        One of ``'vacuum'``, ``'no_k'``, ``'full'``. ``'vacuum'`` runs
        :func:`guiding_center_vacuum_boozer_rhs`; ``'no_k'`` runs
        :func:`guiding_center_no_k_boozer_rhs`; ``'full'`` runs
        :func:`guiding_center_boozer_rhs`. Any other value raises
        :class:`ValueError`.
    zetas
        Optional 1-D array of target ``zeta`` values in ``[0, 2*pi)``.
        The Boozer state ``(s, theta, zeta, v_par)`` makes zeta-plane
        detection a scalar-angle wrap of ``zeta - zeta_target`` modulo
        ``2*pi`` (no ``atan2(y, x)`` needed). Each detected crossing
        is appended to ``phi_hits`` with ``idx == i``. Pass ``None``
        (default) to disable zeta-plane recording. The recorded buffer
        is exposed as ``phi_hits`` for layout-compatibility with the
        Cartesian-route result dataclass; the columns of each row are
        ``[t_hit, idx, s, theta, zeta, v_par]``.
    stopping_criteria
        Tuple of JAX-side stopping criterion dataclasses. The Boozer
        state has no Cartesian ``(x, y, z)`` so only the
        :class:`IterStoppingCriterion`,
        :class:`MinToroidalFluxStoppingCriterion`, and
        :class:`MaxToroidalFluxStoppingCriterion` predicates are
        meaningful on the public surface today; the Cartesian-axis
        predicates (Min/Max R/Z, ToroidalTransit) are evaluated on the
        Boozer ``(s, theta, zeta)`` state mapped through
        ``_stopping_criterion_should_stop`` for layout-compatibility
        but pass identically as on the Cartesian path (they read the
        first three components of the state vector). The flux-coord
        criteria fire on ``s = y[0]``.

    Returns
    -------
    result
        :class:`GuidingCenterTracingResult` with a padded
        ``(max_steps + 1, 5)`` trajectory whose state columns are
        ``(t, s, theta, zeta, v_par)``. The ``phi_hits`` field
        records zeta-plane crossings + stopping-criterion fires; rows
        are ``[t_hit, idx, s, theta, zeta, v_par]``.
    """

    dtype = jnp.float64
    y0_arr = jnp.asarray(y0, dtype=dtype).reshape((4,))
    tmax = jnp.asarray(spec.tmax, dtype=dtype)
    rtol = jnp.asarray(spec.rtol, dtype=dtype)
    atol = jnp.asarray(spec.atol, dtype=dtype)
    dtmax = jnp.asarray(spec.dtmax, dtype=dtype)
    t0 = jnp.asarray(0.0, dtype=dtype)
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
            "trace_guiding_center_boozer mode must be one of "
            f"{{'vacuum', 'no_k', 'full'}}; got mode={mode!r}."
        )
    initial_axis_invalid = _boozer_axis_invalid(y0_arr)

    # Fast path: no events requested → reuse the lean shared driver to
    # preserve the prior compile profile on the no-events parity tests.
    if (zetas is None or len(np.asarray(zetas).reshape((-1,))) == 0) and len(
        stopping_criteria
    ) == 0:
        return _run_dopri5_4state(
            rhs,
            y0_arr,
            tmax,
            rtol,
            atol,
            dtmax,
            max_steps,
            max_phi_hits=max_phi_hits,
        )

    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    k0 = jax.lax.cond(
        initial_axis_invalid,
        lambda _: jnp.zeros((4,), dtype=dtype),
        lambda _: rhs(t0, y0_arr),
        operand=None,
    )
    one = jnp.asarray(1.0, dtype=dtype)

    traj = jnp.zeros((max_steps + 1, 5), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0_arr)
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)

    # zeta-plane crossing buffer; columns are ``[t_hit, idx, s, theta,
    # zeta, v_par]`` (6 wide, matching the upstream Boozer
    # ``res_zeta_hits`` row layout).
    phi_hits_buf = jnp.zeros((max_phi_hits, 6), dtype=dtype)
    phi_hits_count_init = jnp.asarray(0, dtype=jnp.int32)

    if zetas is None:
        zetas_arr = jnp.zeros((0,), dtype=dtype)
    else:
        zetas_arr = jnp.asarray(zetas, dtype=dtype).reshape((-1,))
    num_zetas = int(zetas_arr.shape[0])

    # Initial unwrapped zeta seed: the Boozer state stores zeta
    # directly, so we anchor near the literal initial value.
    zeta_init = _continuous_angle(y0_arr[2], jnp.asarray(np.pi, dtype=dtype), dtype)

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        jnp.asarray(0, dtype=jnp.int32),  # accepted_count
        t0,
        y0_arr,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        zeta_init,  # running zeta_last
        zeta_init,  # transit criterion baseline, set on first accepted step
        jnp.where(
            initial_axis_invalid,
            jnp.asarray(_BOOZER_AXIS_STATUS, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),  # status_event
        initial_axis_invalid,  # stop flag
    )

    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)
    max_phi_hits_i32 = jnp.asarray(max_phi_hits, dtype=jnp.int32)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)

    def cond(carry):
        (
            step_count,
            accepted_count,
            t,
            _y,
            _h,
            _k,
            _traj,
            _mask,
            _phi_hits,
            _phi_count,
            _zeta_last,
            _zeta_init,
            _status_event,
            stop,
        ) = carry
        not_done = t < tmax
        budget_ok = step_count < max_steps_i32
        accepted_ok = accepted_count < max_steps_i32
        not_stopped = jnp.logical_not(stop)
        return jnp.logical_and(
            not_done,
            jnp.logical_and(jnp.logical_and(budget_ok, accepted_ok), not_stopped),
        )

    def body(carry):
        (
            step_count,
            accepted_count,
            t,
            y,
            h,
            k_first,
            traj,
            mask,
            phi_hits_in,
            phi_hits_count_in,
            zeta_last,
            zeta_init,
            status_event,
            _stop,
        ) = carry
        h_clamped = _clamp_step_to_domain(h, t, tmax, dtmax)
        y_new, y_err, k7 = dopri5_step(rhs, t, y, h_clamped, k_first)
        err = _error_norm(y_err, y, y_new, rtol, atol)
        err_safe = jnp.where(jnp.isfinite(err), err, jnp.asarray(jnp.inf, dtype=dtype))
        accepted = err_safe <= one
        axis_invalid = jnp.logical_and(accepted, _boozer_axis_invalid(y_new))
        accepted_valid = jnp.logical_and(accepted, jnp.logical_not(axis_invalid))
        factor = jnp.where(
            err_safe > jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(_SAFETY, dtype=dtype)
            * jnp.power(err_safe, jnp.asarray(-_DOPRI5_EXP, dtype=dtype)),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        factor = jnp.clip(
            factor,
            jnp.asarray(_MIN_FACTOR, dtype=dtype),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        h_next = h_clamped * factor
        t_accepted = _accepted_step_time(t, h_clamped, tmax)
        t_next = jnp.where(accepted, t_accepted, t)
        y_next = jnp.where(accepted, y_new, y)
        k_next = jnp.where(accepted, k7, k_first)

        # ── Zeta-plane crossing detection on accepted steps ──
        # The Boozer state stores zeta directly (no atan2 needed);
        # ``_continuous_angle`` anchors the running unwrap branch
        # near ``zeta_last``.
        zeta_current = _continuous_angle(y_new[2], zeta_last, dtype)

        def state_at_fraction(s):
            h_sub = s * h_clamped
            y_sub, _err, _k7 = dopri5_step(rhs, t, y, h_sub, k_first)
            return y_sub

        def scan_zetas(args):
            hits_in, count_in, zeta_last_in, zeta_curr_in = args
            for i in range(num_zetas):
                zeta_target = zetas_arr[i]
                fl_last = jnp.floor((zeta_last_in - zeta_target) / two_pi)
                fl_curr = jnp.floor((zeta_curr_in - zeta_target) / two_pi)
                crossed = fl_last != fl_curr
                fak = jnp.round(
                    (
                        (zeta_last_in + zeta_curr_in) / jnp.asarray(2.0, dtype=dtype)
                        - zeta_target
                    )
                    / two_pi
                )
                zeta_shift = fak * two_pi + zeta_target

                def diff_at(s, zeta_last_in=zeta_last_in, zeta_shift=zeta_shift):
                    state_sub = state_at_fraction(s)
                    return (
                        _continuous_angle(state_sub[2], zeta_last_in, dtype)
                        - zeta_shift
                    )

                f_left = diff_at(jnp.asarray(0.0, dtype=dtype))
                f_right = diff_at(jnp.asarray(1.0, dtype=dtype))
                bracket_atol = jnp.asarray(1.0e-15, dtype=dtype)
                s_root, _f_root, _bracketed = bracket_root_jax(
                    diff_at,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(1.0, dtype=dtype),
                    f_left,
                    f_right,
                    max_root_iters,
                    bracket_atol,
                )
                t_root = t + s_root * h_clamped
                state_root = state_at_fraction(s_root)
                hit_row = jnp.stack(
                    [
                        t_root,
                        jnp.asarray(float(i), dtype=dtype),
                        state_root[0],
                        state_root[1],
                        state_root[2],
                        state_root[3],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    crossed,
                    hit_row,
                    max_phi_hits_i32,
                )
            return hits_in, count_in

        phi_hits_after, phi_count_after = jax.lax.cond(
            accepted_valid,
            scan_zetas,
            lambda args: (args[0], args[1]),
            operand=(phi_hits_in, phi_hits_count_in, zeta_last, zeta_current),
        )

        # ── Stopping criteria check on accepted state ──
        first_valid_accepted_step = accepted_count == jnp.asarray(0, dtype=jnp.int32)
        zeta_init_for_criteria = jnp.where(
            first_valid_accepted_step,
            zeta_current,
            zeta_init,
        )

        def apply_criteria(args):
            (
                hits_in,
                count_in,
                status_in,
                stop_in,
                iter_count_in,
                zeta_curr_in,
                zeta_init_in,
            ) = args
            for i, criterion in enumerate(stopping_criteria):
                pred = _stopping_criterion_should_stop(
                    criterion,
                    y_next[0],
                    y_next[1],
                    y_next[2],
                    iter_count_in,
                    zeta_curr_in,
                    zeta_init_in,
                    dtype,
                    is_boozer_state=True,
                )
                fires = jnp.logical_and(jnp.logical_not(stop_in), pred)
                idx_val = jnp.asarray(-1 - i, dtype=jnp.int32)
                hit_row = jnp.stack(
                    [
                        t_next,
                        jnp.asarray(float(-1 - i), dtype=dtype),
                        y_next[0],
                        y_next[1],
                        y_next[2],
                        y_next[3],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    fires,
                    hit_row,
                    max_phi_hits_i32,
                )
                status_in = jnp.where(fires, idx_val, status_in)
                stop_in = jnp.logical_or(stop_in, fires)
            return hits_in, count_in, status_in, stop_in

        iter_count_post = step_count + jnp.asarray(1, dtype=jnp.int32)

        (
            phi_hits_after,
            phi_count_after,
            status_after,
            stop_after,
        ) = jax.lax.cond(
            accepted_valid,
            apply_criteria,
            lambda args: (args[0], args[1], args[2], args[3]),
            operand=(
                phi_hits_after,
                phi_count_after,
                status_event,
                jnp.asarray(False),
                iter_count_post,
                zeta_current,
                zeta_init_for_criteria,
            ),
        )
        status_after = jnp.where(
            axis_invalid,
            jnp.asarray(_BOOZER_AXIS_STATUS, dtype=jnp.int32),
            status_after,
        )
        stop_after = jnp.logical_or(stop_after, axis_invalid)

        zeta_last_next = jnp.where(accepted, zeta_current, zeta_last)
        zeta_init_next = jnp.where(
            jnp.logical_and(accepted_valid, first_valid_accepted_step),
            zeta_current,
            zeta_init,
        )
        traj_next, mask_next, accepted_next = _record_trajectory_row(
            traj,
            mask,
            accepted_count,
            t_next,
            y_next,
            _should_record_accepted_step(accepted, stop_after),
        )

        return (
            step_count + jnp.asarray(1, dtype=jnp.int32),
            accepted_next,
            t_next,
            y_next,
            h_next,
            k_next,
            traj_next,
            mask_next,
            phi_hits_after,
            phi_count_after,
            zeta_last_next,
            zeta_init_next,
            status_after,
            stop_after,
        )

    (
        _step_count,
        accepted_count,
        t_final,
        y_final,
        _h_final,
        _k_final,
        traj_final,
        mask_final,
        phi_hits_final,
        phi_hits_count_final,
        _zeta_last_final,
        _zeta_init_final,
        status_event_final,
        stop_at_exit,
    ) = jax.lax.while_loop(cond, body, init_carry)

    last_row = jnp.concatenate(
        [jnp.asarray([t_final], dtype=dtype), y_final.reshape((4,))]
    )

    def fill_padding(idx, traj_carry):
        row_active = mask_final[idx]
        return jax.lax.cond(
            row_active,
            lambda c: c,
            lambda c: c.at[idx].set(last_row),
            operand=traj_carry,
        )

    traj_padded = jax.lax.fori_loop(0, max_steps + 1, fill_padding, traj_final)

    eps_t = jnp.asarray(1.0e-12, dtype=dtype) * jnp.maximum(
        jnp.abs(tmax), jnp.asarray(1.0, dtype=dtype)
    )
    reached = (tmax - t_final) <= eps_t
    status_normal = jnp.where(
        reached,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    return GuidingCenterTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_final,
        phi_hits_count=phi_hits_count_final,
    )


def trace_guiding_centers_boozer_batched(
    spec: GuidingCenterTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    mus: jax.Array,
    boozer_field,
    m: float,
    q: float,
    mode: str = "vacuum",
    zetas: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> GuidingCenterTracingResult:
    """Trace Boozer guiding-centre orbits with one device-side batch graph."""

    y0s_arr = jnp.asarray(y0s, dtype=jnp.float64).reshape((-1, 4))
    dtmaxs_arr = jnp.asarray(dtmaxs, dtype=jnp.float64).reshape((-1,))
    mus_arr = jnp.asarray(mus, dtype=jnp.float64).reshape((-1,))

    def trace_one(args) -> GuidingCenterTracingResult:
        y0, dtmax, mu = args
        return trace_guiding_center_boozer(
            replace(spec, dtmax=dtmax),
            y0,
            boozer_field,
            m=m,
            q=q,
            mu=mu,
            mode=mode,
            zetas=zetas,
            stopping_criteria=stopping_criteria,
        )

    return jax.lax.map(trace_one, (y0s_arr, dtmaxs_arr, mus_arr))


# ── Full-orbit Lorentz RHS (6-state Cartesian) ────────────────────────


@dataclass(frozen=True)
class FullorbitTracingSpec:
    """Immutable contract for a single full-orbit Lorentz integration call.

    Parameters mirror :class:`FieldlineTracingSpec`. The state is 6-D
    ``(x, y, z, vx, vy, vz)`` (position plus Cartesian velocity), so
    the trajectory carry has shape ``(max_steps + 1, 7)`` (columns
    ``(t, x, y, z, vx, vy, vz)``). The integrator follows the upstream
    ``FullorbitRHS::operator()`` vacuum branch in
    ``simsoptpp/tracing.cpp``. The ``phi_hits`` buffer has shape
    ``(max_phi_hits, 8)`` and records phi-plane Poincaré crossings and
    stopping-criterion fires (columns ``[t_hit, idx, x, y, z, vx, vy,
    vz]``); see :class:`FullorbitTracingResult` for the row layout.
    """

    tmax: float
    rtol: float
    atol: float
    max_steps: int
    dtmax: float = np.inf
    max_root_iters: int = 60
    max_phi_hits: int = 128


jax.tree_util.register_dataclass(
    FullorbitTracingSpec,
    data_fields=["tmax", "rtol", "atol", "dtmax"],
    meta_fields=["max_steps", "max_root_iters", "max_phi_hits"],
)


@dataclass(frozen=True)
class FullorbitTracingResult:
    """Return payload for :func:`trace_fullorbit`.

    - ``trajectory`` — ``(max_steps + 1, 7)`` float64 array. Columns are
      ``(t, x, y, z, vx, vy, vz)``. Rows ``[0 : steps_taken + 1]`` are
      populated with accepted states; subsequent rows are padded with
      the final accepted state.
    - ``mask`` — ``(max_steps + 1,)`` bool array. ``True`` for rows that
      correspond to genuine accepted steps; ``False`` for padding.
    - ``steps_taken`` — int32 scalar; count of *accepted* steps the loop
      executed. Excludes the initial-state row.
    - ``status`` — int32 scalar. ``0`` for normal exit (``t >= tmax``),
      ``1`` for max-step-cap exhaustion before reaching ``tmax``,
      ``-1 - i`` when stopping criterion ``i`` fired.
    - ``t_final`` — float64 scalar; ``trajectory[steps_taken, 0]``.
    - ``phi_hits`` — ``(max_phi_hits, 8)`` float64 array. Columns are
      ``[t_hit, idx, x, y, z, vx, vy, vz]``. ``idx >= 0`` denotes a
      phi-plane crossing for ``phis[int(idx)]``; ``idx < 0`` denotes
      stopping criterion ``-1 - int(idx)`` firing.
    - ``phi_hits_count`` — int32 scalar; total detected event count.
      Values greater than ``max_phi_hits`` mean the fixed buffer holds
      a truncated prefix.
    """

    trajectory: jax.Array
    mask: jax.Array
    steps_taken: jax.Array
    status: jax.Array
    t_final: jax.Array
    phi_hits: jax.Array
    phi_hits_count: jax.Array


jax.tree_util.register_dataclass(
    FullorbitTracingResult,
    data_fields=[
        "trajectory",
        "mask",
        "steps_taken",
        "status",
        "t_final",
        "phi_hits",
        "phi_hits_count",
    ],
    meta_fields=[],
)


def fullorbit_vacuum_rhs(
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    m: float,
    q: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    r"""Return ``rhs(t, y) -> dy/dt`` for the 6-state vacuum full-orbit ODE.

    State is ``y = (x, y, z, vx, vy, vz)``. The Lorentz equation of
    motion in vacuum (no E field; matching the upstream
    ``FullorbitRHS::operator()`` in ``simsoptpp/tracing.cpp``) is

    .. math::

       \dot{\mathbf{x}} &= \mathbf{v}, \\
       \dot{\mathbf{v}} &= \frac{q}{m}\, \mathbf{v} \times \mathbf{B}(\mathbf{x}).

    Parameters
    ----------
    magnetic_field_fn
        JAX-traceable callable mapping a Cartesian point ``[3]`` to the
        magnetic field ``B(x)`` of shape ``[3]``. Only the field value
        is required (no Jacobian) because the Lorentz force depends
        on ``B`` directly rather than its spatial gradient.
    m, q
        Particle mass and charge (Python floats). Captured at closure
        construction; not mutated thereafter.

    Returns
    -------
    rhs
        ``rhs(t, y)`` callable returning a length-6 vector
        ``[vx, vy, vz, ax, ay, az]`` with the acceleration computed as
        ``(q/m) v x B(x)``.
    """

    qoverm = jnp.asarray(q / m, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t  # Field is autonomous; signature kept for ODE-driver shape.
        position = y[:3]
        velocity = y[3:6]
        B_raw = magnetic_field_fn(position)
        B = jnp.asarray(B_raw, dtype=y.dtype).reshape((3,))
        acceleration = qoverm * jnp.cross(velocity, B)
        return jnp.concatenate([velocity, acceleration])

    return rhs


def trace_fullorbit(
    spec: FullorbitTracingSpec,
    y0: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    m: float,
    q: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FullorbitTracingResult:
    """Trace a full-orbit Lorentz trajectory from ``y0`` for ``spec.tmax`` seconds.

    Parameters
    ----------
    spec
        Tracing contract; see :class:`FullorbitTracingSpec`.
    y0
        Initial state ``[x, y, z, vx, vy, vz]`` (length 6). Treated as
        float64.
    magnetic_field_fn
        JAX-traceable callable mapping a Cartesian point ``[3]`` to the
        magnetic field ``B(x)`` of shape ``[3]``. See
        :func:`fullorbit_vacuum_rhs` for the convention.
    m, q
        Particle mass and charge (Python floats).
    phis
        Optional 1-D array of target ``phi`` values in ``[0, 2*pi)``.
        Each detected crossing is appended to the result's ``phi_hits``
        buffer with ``idx == i``. Pass ``None`` (default) to disable
        phi-plane recording. The crossing test uses the same
        ``atan2(y, x)`` continuous-branch logic as the Cartesian
        fieldline / GC drivers (see :func:`_continuous_phi`).
    stopping_criteria
        Tuple of JAX-side stopping criterion dataclasses (see
        :class:`MinRStoppingCriterion`, :class:`MaxRStoppingCriterion`,
        :class:`MinZStoppingCriterion`, :class:`MaxZStoppingCriterion`,
        :class:`ToroidalTransitStoppingCriterion`,
        :class:`IterStoppingCriterion`, and
        :class:`LevelsetStoppingCriterion`). The non-Levelset criteria
        are evaluated on the post-step Cartesian position ``(x, y, z)``
        and toroidal-transit counter; the Levelset criterion fires
        when ``classifier_fn(x, y, z) < 0`` (outside the levelset
        surface). When multiple criteria fire on the same
        accepted step, the first matching criterion in iteration order
        wins; ``status`` then equals ``-1 - i`` reflecting that index.

    Returns
    -------
    result
        :class:`FullorbitTracingResult` with a padded
        ``(max_steps + 1, 7)`` trajectory, a mask, an accepted-step
        count, an exit status, ``t_final``, and the phi-crossing
        buffer.
    """

    dtype = jnp.float64
    y0_arr = jnp.asarray(y0, dtype=dtype).reshape((6,))
    tmax = jnp.asarray(spec.tmax, dtype=dtype)
    rtol = jnp.asarray(spec.rtol, dtype=dtype)
    atol = jnp.asarray(spec.atol, dtype=dtype)
    dtmax = jnp.asarray(spec.dtmax, dtype=dtype)
    t0 = jnp.asarray(0.0, dtype=dtype)
    max_steps = int(spec.max_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    max_phi_hits = int(spec.max_phi_hits)
    if max_phi_hits <= 0:
        raise ValueError(f"max_phi_hits must be positive, got {max_phi_hits}")
    max_root_iters = int(spec.max_root_iters)

    rhs = fullorbit_vacuum_rhs(magnetic_field_fn, m, q)
    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    k0 = rhs(t0, y0_arr)
    one = jnp.asarray(1.0, dtype=dtype)

    # Pre-allocate the trajectory carry with columns
    # ``(t, x, y, z, vx, vy, vz)``. Row 0 holds the initial state;
    # rows 1..max_steps fill in as accepted steps occur. Padding rows
    # at the end of the run get the final accepted state.
    traj = jnp.zeros((max_steps + 1, 7), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0_arr)
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)

    # Phi-plane crossing buffer. Each row is ``[t_hit, idx, x, y, z, vx,
    # vy, vz]`` (8 columns to match the upstream
    # ``sopp.particle_fullorbit_tracing`` row shape).
    phi_hits_buf = jnp.zeros((max_phi_hits, 8), dtype=dtype)
    phi_hits_count_init = jnp.asarray(0, dtype=jnp.int32)

    if phis is None:
        phis_arr = jnp.zeros((0,), dtype=dtype)
    else:
        phis_arr = jnp.asarray(phis, dtype=dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    # Initial unwrapped phi seed (C++ tracing.cpp uses pi).
    phi_init = _continuous_phi(
        y0_arr[0], y0_arr[1], jnp.asarray(np.pi, dtype=dtype), dtype
    )

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        jnp.asarray(0, dtype=jnp.int32),  # accepted_count
        t0,
        y0_arr,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        phi_init,  # running phi_last
        phi_init,  # transit criterion baseline, set on first accepted step
        jnp.asarray(0, dtype=jnp.int32),  # status_event (criterion idx)
        jnp.asarray(False),  # stop flag
    )

    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)
    max_phi_hits_i32 = jnp.asarray(max_phi_hits, dtype=jnp.int32)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)

    def cond(carry):
        (
            step_count,
            accepted_count,
            t,
            _y,
            _h,
            _k,
            _traj,
            _mask,
            _phi_hits,
            _phi_count,
            _phi_last,
            _phi_init,
            _status_event,
            stop,
        ) = carry
        not_done = t < tmax
        budget_ok = step_count < max_steps_i32
        accepted_ok = accepted_count < max_steps_i32
        not_stopped = jnp.logical_not(stop)
        return jnp.logical_and(
            not_done,
            jnp.logical_and(jnp.logical_and(budget_ok, accepted_ok), not_stopped),
        )

    def body(carry):
        (
            step_count,
            accepted_count,
            t,
            y,
            h,
            k_first,
            traj,
            mask,
            phi_hits_in,
            phi_hits_count_in,
            phi_last,
            phi_init,
            status_event,
            _stop,
        ) = carry
        h_clamped = _clamp_step_to_domain(h, t, tmax, dtmax)
        y_new, y_err, k7 = dopri5_step(rhs, t, y, h_clamped, k_first)
        err = _error_norm(y_err, y, y_new, rtol, atol)
        err_safe = jnp.where(jnp.isfinite(err), err, jnp.asarray(jnp.inf, dtype=dtype))
        accepted = err_safe <= one
        factor = jnp.where(
            err_safe > jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(_SAFETY, dtype=dtype)
            * jnp.power(err_safe, jnp.asarray(-_DOPRI5_EXP, dtype=dtype)),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        factor = jnp.clip(
            factor,
            jnp.asarray(_MIN_FACTOR, dtype=dtype),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        h_next = h_clamped * factor
        t_accepted = _accepted_step_time(t, h_clamped, tmax)
        t_next = jnp.where(accepted, t_accepted, t)
        y_next = jnp.where(accepted, y_new, y)
        k_next = jnp.where(accepted, k7, k_first)

        # ── Phi-plane crossing detection on accepted steps ──
        phi_current = _continuous_phi(y_new[0], y_new[1], phi_last, dtype)

        def state_at_fraction(s):
            """Sub-step DOPRI5 from ``(t, y)`` with step ``s * h_clamped``.

            Re-runs a fresh DOPRI5 step from the prior accepted state
            so the returned 6-state has 5th-order RK accuracy. FSAL
            value ``k_first`` is reused as the leading-stage derivative.
            """
            h_sub = s * h_clamped
            y_sub, _err, _k7 = dopri5_step(rhs, t, y, h_sub, k_first)
            return y_sub

        def scan_phis(args):
            hits_in, count_in, phi_last_in, phi_curr_in = args
            for i in range(num_phis):
                phi_target = phis_arr[i]
                fl_last = jnp.floor((phi_last_in - phi_target) / two_pi)
                fl_curr = jnp.floor((phi_curr_in - phi_target) / two_pi)
                crossed = fl_last != fl_curr
                fak = jnp.round(
                    (
                        (phi_last_in + phi_curr_in) / jnp.asarray(2.0, dtype=dtype)
                        - phi_target
                    )
                    / two_pi
                )
                phi_shift = fak * two_pi + phi_target

                def diff_at(s, phi_last_in=phi_last_in, phi_shift=phi_shift):
                    pos = state_at_fraction(s)
                    return (
                        _continuous_phi(pos[0], pos[1], phi_last_in, dtype) - phi_shift
                    )

                f_left = diff_at(jnp.asarray(0.0, dtype=dtype))
                f_right = diff_at(jnp.asarray(1.0, dtype=dtype))
                bracket_atol = jnp.asarray(1.0e-15, dtype=dtype)
                s_root, _f_root, _bracketed = bracket_root_jax(
                    diff_at,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(1.0, dtype=dtype),
                    f_left,
                    f_right,
                    max_root_iters,
                    bracket_atol,
                )
                t_root = t + s_root * h_clamped
                state_root = state_at_fraction(s_root)
                hit_row = jnp.stack(
                    [
                        t_root,
                        jnp.asarray(float(i), dtype=dtype),
                        state_root[0],
                        state_root[1],
                        state_root[2],
                        state_root[3],
                        state_root[4],
                        state_root[5],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    crossed,
                    hit_row,
                    max_phi_hits_i32,
                )
            return hits_in, count_in

        phi_hits_after, phi_count_after = jax.lax.cond(
            accepted,
            scan_phis,
            lambda args: (args[0], args[1]),
            operand=(phi_hits_in, phi_hits_count_in, phi_last, phi_current),
        )

        # ── Stopping criteria check on accepted state ──
        first_accepted_step = accepted_count == jnp.asarray(0, dtype=jnp.int32)
        phi_init_for_criteria = jnp.where(
            first_accepted_step,
            phi_current,
            phi_init,
        )

        def apply_criteria(args):
            (
                hits_in,
                count_in,
                status_in,
                stop_in,
                iter_count_in,
                phi_curr_in,
                phi_init_in,
            ) = args
            for i, criterion in enumerate(stopping_criteria):
                pred = _stopping_criterion_should_stop(
                    criterion,
                    y_next[0],
                    y_next[1],
                    y_next[2],
                    iter_count_in,
                    phi_curr_in,
                    phi_init_in,
                    dtype,
                )
                fires = jnp.logical_and(jnp.logical_not(stop_in), pred)
                idx_val = jnp.asarray(-1 - i, dtype=jnp.int32)
                hit_row = jnp.stack(
                    [
                        t_next,
                        jnp.asarray(float(-1 - i), dtype=dtype),
                        y_next[0],
                        y_next[1],
                        y_next[2],
                        y_next[3],
                        y_next[4],
                        y_next[5],
                    ]
                )
                hits_in, count_in = _append_event_row(
                    hits_in,
                    count_in,
                    fires,
                    hit_row,
                    max_phi_hits_i32,
                )
                status_in = jnp.where(fires, idx_val, status_in)
                stop_in = jnp.logical_or(stop_in, fires)
            return hits_in, count_in, status_in, stop_in

        iter_count_post = step_count + jnp.asarray(1, dtype=jnp.int32)

        (
            phi_hits_after,
            phi_count_after,
            status_after,
            stop_after,
        ) = jax.lax.cond(
            accepted,
            apply_criteria,
            lambda args: (args[0], args[1], args[2], args[3]),
            operand=(
                phi_hits_after,
                phi_count_after,
                status_event,
                jnp.asarray(False),
                iter_count_post,
                phi_current,
                phi_init_for_criteria,
            ),
        )

        phi_last_next = jnp.where(accepted, phi_current, phi_last)
        phi_init_next = jnp.where(
            jnp.logical_and(accepted, first_accepted_step),
            phi_current,
            phi_init,
        )
        traj_next, mask_next, accepted_next = _record_trajectory_row(
            traj,
            mask,
            accepted_count,
            t_next,
            y_next,
            _should_record_accepted_step(accepted, stop_after),
        )

        return (
            step_count + jnp.asarray(1, dtype=jnp.int32),
            accepted_next,
            t_next,
            y_next,
            h_next,
            k_next,
            traj_next,
            mask_next,
            phi_hits_after,
            phi_count_after,
            phi_last_next,
            phi_init_next,
            status_after,
            stop_after,
        )

    (
        _step_count,
        accepted_count,
        t_final,
        y_final,
        _h_final,
        _k_final,
        traj_final,
        mask_final,
        phi_hits_final,
        phi_hits_count_final,
        _phi_last_final,
        _phi_init_final,
        status_event_final,
        stop_at_exit,
    ) = jax.lax.while_loop(cond, body, init_carry)

    last_row = jnp.concatenate(
        [jnp.asarray([t_final], dtype=dtype), y_final.reshape((6,))]
    )

    def fill_padding(idx, traj_carry):
        row_active = mask_final[idx]
        return jax.lax.cond(
            row_active,
            lambda c: c,
            lambda c: c.at[idx].set(last_row),
            operand=traj_carry,
        )

    traj_padded = jax.lax.fori_loop(0, max_steps + 1, fill_padding, traj_final)

    eps_t = jnp.asarray(1.0e-12, dtype=dtype) * jnp.maximum(
        jnp.abs(tmax), jnp.asarray(1.0, dtype=dtype)
    )
    reached = (tmax - t_final) <= eps_t
    status_normal = jnp.where(
        reached,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    return FullorbitTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_final,
        phi_hits_count=phi_hits_count_final,
    )


def trace_fullorbits_batched(
    spec: FullorbitTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    m: float,
    q: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FullorbitTracingResult:
    """Trace full-orbit Lorentz trajectories with one vmapped JAX graph."""

    y0s_arr = jnp.asarray(y0s, dtype=jnp.float64).reshape((-1, 6))
    dtmaxs_arr = jnp.asarray(dtmaxs, dtype=jnp.float64).reshape((-1,))

    def trace_one(y0: jax.Array, dtmax: jax.Array) -> FullorbitTracingResult:
        return trace_fullorbit(
            replace(spec, dtmax=dtmax),
            y0,
            magnetic_field_fn,
            m=m,
            q=q,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    return jax.vmap(trace_one)(y0s_arr, dtmaxs_arr)
