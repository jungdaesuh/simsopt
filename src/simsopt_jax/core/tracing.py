"""JAX port of ``legacy native extension/tracing.cpp`` (Tier P1 item 14).

This module implements an in-repo JAX Dormand-Prince RK4(5) integrator
with boost.odeint's step-size controller and a bracketed Illinois
false-position event localizer over the DOPRI5 dense-output polynomial.
The implemented scope covers:

- the fieldline RHS ``dx/dt = B(x)`` used in the upstream C++
  ``FieldlineRHS``,
- the 4-state Cartesian vacuum guiding-centre RHS shipped under the
  item-14 follow-up (state ``[x, y, z, v_par]``, drift terms following
  the upstream ``GuidingCenterVacuumRHS::operator()`` definition in
  ``legacy native extension/tracing.cpp``). The driver :func:`trace_guiding_center`
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

The bracketed event localizer mirrors what upstream localizes and how.
``legacy native extension/tracing.cpp`` root-finds an angle-plane crossing only
when the plane was actually crossed, brackets it in absolute time over
the accepted step, and evaluates the residual on the DOPRI5 *dense
output* polynomial (``dense.calc_state``), which costs no extra RHS
evaluation. :func:`_dense_output_state` is that polynomial, ported term
for term from ``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp``
(``calc_state``). The bracketing iteration itself is an Illinois
false-position update rather than upstream's ``toms748_solve``, whose
branchy interpolation ladder has no fixed-shape JAX form; the settings
that decide *where* the root lands are upstream's, namely the residual,
the bracket, ``rootmaxit = 200`` (``max_root_iters``) and the
``boost::math::tools::eps_tolerance`` termination
``|a - b| <= eps * min(|a|, |b|)`` (see :data:`_ROOT_BRACKET_EPS` for which
``eps`` and why). The accepted accuracy contract is the
``event_time_tracing`` lane in
``simsopt_jax.parity_tolerances.PARITY_LADDER_TOLERANCES``.

Architecture
============

- ``FieldlineTracingSpec`` — frozen dataclass (registered as a JAX
  pytree) carrying ``tmax``, ``rtol``, ``atol``, per-lane ``dtmax``,
  ``max_steps`` (static), and ``max_root_iters`` (static).
- ``dopri5_step`` — single Dormand-Prince step returning the 5th-order
  state, the embedded error vector, and the trailing-stage derivative
  for FSAL reuse.
- ``trace_fieldline`` — adaptive driver with a fixed-shape trajectory carry
  of shape ``(max_steps + 1, 4)``. Parity mode uses a differentiable
  fixed-length ``jax.lax.scan``; fast mode uses an early-exit
  ``jax.lax.while_loop``. Padded rows are populated with the final accepted
  state; the companion mask of shape ``(max_steps + 1,)`` identifies the live
  prefix.
- ``bracket_root_jax`` — Illinois false-position event localizer. Returns the
  bracketed root and a bool indicating whether the bracket actually
  contained a sign change.

The step controller, error norm and step-size update are boost.odeint's
``controlled_runge_kutta`` with ``default_error_checker`` and
``default_step_adjuster``, which is what
``make_dense_output(tol, tol, dtmax, runge_kutta_dopri5<State>())``
instantiates in ``legacy native extension/tracing.cpp``.

Terminal statuses reported by every driver here:

- ``0`` — the lane reached ``tmax``;
- ``-1 - i`` — ``stopping_criteria[i]`` fired;
- ``1`` — the call's ``max_steps`` ran out with neither of the above
  (for the chunked Cartesian routes this is "continue in the next
  chunk", not a failure);
- ``2`` (:data:`TRACING_STATUS_STEP_CONTROL_FAILED`) — the step
  controller stopped making progress (see
  :data:`_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS`), or a trial upstream would
  have accepted landed on a NON-FINITE state, which this port refuses to
  take (:func:`_dopri5_adaptive_step`), or an accepted step crossed an angle
  plane while its own dense output — the continuous extension upstream
  localizes the crossing on — was non-finite, so the row upstream would push
  cannot be published (:func:`_scan_angle_plane_events`). So ``0`` and
  ``-1 - i`` imply a finite final state, and no trajectory row, event row or
  final state is ever taken from a non-finite one;
- ``-2`` (:data:`TRACING_STATUS_BOOZER_AXIS`) — BRANCH-ADDED and emitted by
  the two Boozer drivers only: the lane left the magnetic axis (``s <= 0``),
  where the Boozer right-hand side is undefined. Upstream has no such status.
  It COLLIDES with the ``-1 - i`` rule at ``i = 1``, and the axis test wins
  (``status = where(axis_invalid, -2, status_event)``), so on a Boozer route
  ``-2`` means "left the axis, or ``stopping_criteria[1]`` fired". Consumers
  that decode negative statuses must special-case it for those two routes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

from simsopt_jax.backend.dtypes import explicit_device_array

from ._device_scalars import two_pi as _device_two_pi
from .boozer_analytic import (
    BoozerAnalyticFrozenState,
)
from .boozer_analytic import (
    _eval_dGds as _analytic_dGds,
)
from .boozer_analytic import (
    _eval_dIds as _analytic_dIds,
)
from .boozer_analytic import (
    _eval_dKdtheta as _analytic_dKdtheta,
)
from .boozer_analytic import (
    _eval_dKdzeta as _analytic_dKdzeta,
)
from .boozer_analytic import (
    _eval_dmodBds as _analytic_dmodBds,
)
from .boozer_analytic import (
    _eval_dmodBdtheta as _analytic_dmodBdtheta,
)
from .boozer_analytic import (
    _eval_dmodBdzeta as _analytic_dmodBdzeta,
)
from .boozer_analytic import (
    _eval_G as _analytic_G,
)
from .boozer_analytic import (
    _eval_I as _analytic_I,
)
from .boozer_analytic import (
    _eval_iota as _analytic_iota,
)
from .boozer_analytic import (
    _eval_K as _analytic_K,
)
from .boozer_analytic import (
    _eval_modB as _analytic_modB,
)
from .boozer_radial_field import (
    BoozerRadialInterpolantFrozenState,
)
from .boozer_radial_field import (
    _eval_dGds as _radial_dGds,
)
from .boozer_radial_field import (
    _eval_dIds as _radial_dIds,
)
from .boozer_radial_field import (
    _eval_dKdtheta as _radial_dKdtheta,
)
from .boozer_radial_field import (
    _eval_dKdtheta_from_columns as _radial_dKdtheta_from_columns,
)
from .boozer_radial_field import (
    _eval_dKdzeta as _radial_dKdzeta,
)
from .boozer_radial_field import (
    _eval_dKdzeta_from_columns as _radial_dKdzeta_from_columns,
)
from .boozer_radial_field import (
    _eval_dmodBds as _radial_dmodBds,
)
from .boozer_radial_field import (
    _eval_dmodBds_from_columns as _radial_dmodBds_from_columns,
)
from .boozer_radial_field import (
    _eval_dmodBdtheta as _radial_dmodBdtheta,
)
from .boozer_radial_field import (
    _eval_dmodBdtheta_from_columns as _radial_dmodBdtheta_from_columns,
)
from .boozer_radial_field import (
    _eval_dmodBdzeta as _radial_dmodBdzeta,
)
from .boozer_radial_field import (
    _eval_dmodBdzeta_from_columns as _radial_dmodBdzeta_from_columns,
)
from .boozer_radial_field import (
    _eval_G as _radial_G,
)
from .boozer_radial_field import (
    _eval_I as _radial_I,
)
from .boozer_radial_field import (
    _eval_iota as _radial_iota,
)
from .boozer_radial_field import (
    _eval_K as _radial_K,
)
from .boozer_radial_field import (
    _eval_K_from_columns as _radial_K_from_columns,
)
from .boozer_radial_field import (
    _eval_modB as _radial_modB,
)
from .boozer_radial_field import (
    _eval_modB_from_columns as _radial_modB_from_columns,
)
from .boozer_radial_field import (
    _eval_radial_rhs_columns as _radial_eval_rhs_columns,
)
from .interpolated_boozer_field import (
    _INTERP_EVALUATORS,
    InterpolatedBoozerFieldFrozenState,
)
from .interpolated_field import (
    InterpolatedFieldCylCache,
)
from .sharding import (
    maybe_shard_trajectory_batch_inputs,
    replicate_tree_on_mesh,
    trajectory_batch_sharding_config,
)

TracingStateInput = jax.Array | list[ArrayLike] | tuple[ArrayLike, ...]

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
    "TRACING_STATUS_BOOZER_AXIS",
    "TRACING_STATUS_INCOMPLETE",
    "TRACING_STATUS_STEP_CONTROL_FAILED",
    "ToroidalTransitStoppingCriterion",
    "TracingStateInput",
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


# Step-size control constants of boost.odeint's ``controlled_runge_kutta`` with
# ``default_error_checker`` / ``default_step_adjuster``, which is what
# ``make_dense_output(tol, tol, dtmax, runge_kutta_dopri5)`` instantiates in the
# upstream ``legacy native extension/tracing.cpp``. ``runge_kutta_dopri5``
# declares ``stepper_order() == 5`` and ``error_order() == 4``, so the rejected
# branch uses ``1/(error_order - 1)`` and the accepted branch ``1/stepper_order``.
# The accepted branch only grows the step when the error is below
# ``_INCREASE_ERROR_THRESHOLD``, and ``_MIN_INCREASE_ERROR = 5**-stepper_order``
# caps the growth factor at ``_SAFETY * 5``.
_SAFETY = 0.9
_DECREASE_EXP = 1.0 / 3.0
_INCREASE_EXP = 0.2
_MIN_FACTOR = 0.2
_INCREASE_ERROR_THRESHOLD = 0.5
_MIN_INCREASE_ERROR = 5.0**-5
# Each branch's error is clipped to the interval on which that branch is the one
# boost selects, so neither power ever sees 0 or inf. Both bounds are exact, not
# tolerances: the decrease branch runs only for ``error > 1`` and saturates at
# ``_MIN_FACTOR`` once ``0.9 * error**(-1/3) <= 0.2``, i.e. at
# ``(0.9 / 0.2)**3``; the increase branch runs only for
# ``error < _INCREASE_ERROR_THRESHOLD``.
_MAX_DECREASE_ERROR = (_SAFETY / _MIN_FACTOR) ** 3
# Upstream's capped growth factor ``0.9 * pow(5**-5, -1/5)`` is a constant of
# boost's arithmetic, evaluated by the C library's ``pow``; Python's float power
# is that same ``pow``. It is formed here, on the host, because a device power
# is not correctly rounded everywhere: CUDA gives ``pow(5**-5, -0.2) =
# 5.000000000000001`` where the C library gives ``5.0``, which alone moved every
# capped step of the GPU lane off upstream's by one ulp.
_MAX_INCREASE_FACTOR = _SAFETY * _MIN_INCREASE_ERROR**-_INCREASE_EXP

# Terminal statuses. ``0`` = reached ``tmax``; ``-1 - i`` = criterion ``i``
# fired; ``1`` = this call's ``max_steps`` ran out; ``2`` = the step controller
# stopped making progress; ``-2`` = a Boozer lane left the axis (branch-added,
# and colliding with ``-1 - i`` at ``i = 1``). The module docstring states the
# whole vocabulary, including that collision.
TRACING_STATUS_INCOMPLETE = 1
TRACING_STATUS_STEP_CONTROL_FAILED = 2

# boost.odeint bounds one accepted step to 500 trials:
# ``dense_output_runge_kutta::do_step`` builds a fresh ``failed_step_checker``
# per accepted step and calls it after EVERY ``try_step``, so the 501st trial
# throws whatever its outcome
# (``boost/numeric/odeint/stepper/dense_output_runge_kutta.hpp`` ``do_step``;
# ``boost/numeric/odeint/integrate/max_step_checker.hpp`` ``failed_step_checker``,
# default ``max_steps = 500``). The JAX loop counts trials that do not ADVANCE
# ``t`` rather than only rejected ones: a rejected trial never advances ``t``, so
# the port's rule is upstream's, STRICTER BY ONE TRIAL -- boost runs a 501st
# ``try_step`` and throws in the ``fail_checker()`` after it, while the lane here
# stops at the 500th; the outcome of that extra trial is discarded either way, so
# no result moves. The superset also catches the one
# way a finite right-hand side can spin forever -- once ``h`` underflows to
# ``0.0`` the embedded error is ``0`` and the trial is *accepted* while ``t``
# stands still. boost names that failure too: ``max_step_checker`` raises
# ``no_progress_error`` (same header). Upstream throws; a JAX lane cannot, so it
# stops with ``TRACING_STATUS_STEP_CONTROL_FAILED``.
_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS = 500

# Bracket tolerance of the event localizer, in
# ``boost::math::tools::eps_tolerance``'s sense: stop once
# ``|a - b| <= eps * min(|a|, |b|)``. This is ``eps_tolerance()``'s
# DEFAULT-CONSTRUCTED value, ``4 * tools::epsilon<double>()``
# (``boost/math/tools/toms748_solve.hpp``), i.e. the FLOOR of the value upstream
# builds from the integrator tolerance as
# ``eps_tolerance<double> roottol(-int(std::log2(tol)))``
# (``legacy native extension/tracing.cpp``).
#
# Why the floor and not the literal bits-derived value: upstream's number is a
# speed knob for TOMS-748, whose secant/quadratic/cubic ladder lands on a smooth
# root long before its bracket test fires, so what upstream ACHIEVES is machine
# precision whatever ``tol`` says. The port's Illinois iteration converges more
# slowly, so adopting the literal value would stop it while the residual is still
# large and put the hit somewhere upstream never puts it. Measured on the bounded
# NCSX case (``tol = 1e-7``, where upstream's construction gives
# ``eps = 2.38e-7``): median ``|atan2(y, x) - phi_plane|`` is 2.33e-15 for the
# native lane, 1.16e-6 for the port with the literal value and 4.55e-15 with this
# floor (maximum 1.60e-14, against the native lane's 9.91e-08). The iteration ceiling IS upstream's (``rootmaxit = 200``, the
# ``max_root_iters`` default on every spec), and with the dense-output polynomial
# the extra iterations cost no right-hand-side evaluation.
_ROOT_BRACKET_EPS = 4.0 * float(np.finfo(np.float64).eps)

_FIELDLINE_INITIAL_STEP_FRACTION = 1.0e-5
_PARTICLE_INITIAL_STEP_FRACTION = 1.0e-3
TRACING_STATUS_BOOZER_AXIS = -2
_QUARTER_TURN = 0.5 * np.pi


def _cartesian_radius(xyz_init) -> float:
    point = np.asarray(xyz_init, dtype=np.float64)
    return float(np.sqrt(point[0] * point[0] + point[1] * point[1]))


def _cartesian_radii(xyz_inits) -> np.ndarray:
    points = np.asarray(xyz_inits, dtype=np.float64)
    return np.sqrt(points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1])


def _quarter_turn_dtmax(radius: float, denominator: float) -> float:
    return float(_quarter_turn_dtmaxs(np.asarray([radius]), denominator)[0])


def _quarter_turn_dtmaxs(radii, denominators) -> np.ndarray:
    return (
        np.asarray(radii, dtype=np.float64)
        * _QUARTER_TURN
        / np.asarray(denominators, dtype=np.float64)
    )


def _cartesian_particle_dtmax(xyz_init, speed_total: float) -> float:
    return _quarter_turn_dtmax(_cartesian_radius(xyz_init), speed_total)


def _cartesian_particle_dtmaxs(xyz_inits, speed_total) -> np.ndarray:
    return _quarter_turn_dtmaxs(_cartesian_radii(xyz_inits), speed_total)


def _boozer_particle_dtmax(G0: float, modB: float, speed_total: float) -> float:
    return _quarter_turn_dtmax(abs(float(G0)) / float(modB), speed_total)


def _boozer_particle_dtmaxs(G0, modB, speed_total: float) -> np.ndarray:
    return _quarter_turn_dtmaxs(
        np.abs(G0) / np.asarray(modB, dtype=np.float64), speed_total
    )


def _fieldline_dtmax(xyz_init, abs_B: float) -> float:
    return _quarter_turn_dtmax(_cartesian_radius(xyz_init), abs_B)


def _fieldline_dtmaxs(xyz_inits, abs_B) -> np.ndarray:
    return _quarter_turn_dtmaxs(_cartesian_radii(xyz_inits), abs_B)


def _magnetic_moments(speed_total: float, speed_par, abs_B) -> np.ndarray:
    parallel_speeds = np.asarray(speed_par, dtype=np.float64)
    vperp2 = float(speed_total) * float(speed_total) - parallel_speeds * parallel_speeds
    return vperp2 / (2.0 * np.asarray(abs_B, dtype=np.float64))


def _append_event_row(
    phi_hits: jax.Array,
    phi_hits_count: jax.Array,
    event_detected: jax.Array,
    hit_row: jax.Array,
    max_phi_hits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Append ``hit_row`` when capacity permits while counting all events."""

    one = _device_index(1)
    phi_hits, phi_hits_count = _event_carry_with_lane_axis(
        phi_hits,
        phi_hits_count,
        hit_row,
    )
    has_room = phi_hits_count < max_phi_hits
    should_write = jnp.logical_and(event_detected, has_room)
    record_index = jnp.minimum(phi_hits_count, max_phi_hits - one)

    phi_hits = jax.lax.cond(
        should_write,
        lambda hits: hits.at[record_index].set(hit_row),
        lambda hits: hits,
        phi_hits,
    )
    event_count = jnp.where(event_detected, one, _device_index(0))
    phi_hits_count = phi_hits_count + event_count
    return phi_hits, phi_hits_count


def _event_carry_with_lane_axis(
    phi_hits: jax.Array,
    phi_hits_count: jax.Array,
    lane_value: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    varying_zero = _lane_axis_zero(lane_value)
    return (
        phi_hits + varying_zero,
        phi_hits_count + varying_zero.astype(phi_hits_count.dtype),
    )


def _lane_axis_zero(lane_value: jax.Array) -> jax.Array:
    """A zero that still carries ``lane_value``'s ``vmap`` lane axis.

    The carry has to depend on the per-lane value so that ``vmap`` keeps the
    lane axis on the event buffer. It must NOT be arithmetic on the value:
    ``lane_value - lane_value`` is ``nan`` for a non-finite row, and adding that
    to the carried ``phi_hits`` rewrites every row already recorded as ``nan``.
    That is how a shipped-scale particle lane published an all-``nan``
    ``poincare:positions`` while its first rows had been recorded finite.
    ``nan_to_num`` keeps the data
    dependency, and the device or host placement of ``lane_value`` (a fresh
    ``zeros_like`` constant would be a host-to-device transfer under the strict
    transfer guard), while being ``0`` for every input: ``x - x`` is ``0`` for a
    finite ``x`` and ``nan`` otherwise, and ``nan_to_num`` maps that to ``0``.
    """

    return jnp.sum(jnp.nan_to_num(lane_value - lane_value))


def _lane_axis_carry_zeroes(
    lane_value: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    varying_zero = _lane_axis_zero(lane_value)
    return (
        varying_zero,
        varying_zero.astype(jnp.int32),
        varying_zero != varying_zero,
    )


def _device_array(value, dtype):
    return explicit_device_array(value, dtype=dtype)


def _device_zeros(shape: tuple[int, ...], dtype):
    return explicit_device_array(np.zeros(shape, dtype=np.dtype(dtype)), dtype=dtype)


def _as_device_array(value, dtype) -> jax.Array:
    """Return ``value`` as a JAX array without implicit host staging.

    Sequences containing tracers stay local to the enclosing transform. Eager
    sequences containing concrete JAX arrays explicitly normalize every leaf
    to the default placement used by the tracing kernels.
    """

    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return jnp.asarray(value, dtype=dtype)
    if isinstance(value, (list, tuple)):
        leaves = jax.tree.leaves(value)
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            return jnp.asarray(value, dtype=dtype)
        concrete_arrays = tuple(leaf for leaf in leaves if isinstance(leaf, jax.Array))
        if concrete_arrays:
            target_sharding = _device_array(0, dtype).sharding

            def stage_leaf(leaf):
                if isinstance(leaf, jax.Array):
                    return explicit_device_array(
                        leaf, dtype=dtype, target=target_sharding
                    )
                return explicit_device_array(
                    leaf,
                    dtype=dtype,
                    target=target_sharding,
                )

            return jnp.asarray(jax.tree.map(stage_leaf, value), dtype=dtype)
    return _device_array(value, dtype)


def _device_index(value: int) -> jax.Array:
    return _device_array(value, jnp.int32)


def _device_false() -> jax.Array:
    return _device_array(False, np.bool_)


def _prefix3(vector: jax.Array) -> jax.Array:
    return jnp.split(vector, [3])[0]


def _split_xyz(vector: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, y, z = jnp.split(_prefix3(vector), [1, 2])
    return jnp.reshape(x, ()), jnp.reshape(y, ()), jnp.reshape(z, ())


def _continuous_phi_from_state(
    state: jax.Array, angle_near: jax.Array, dtype
) -> jax.Array:
    x, y, _z = _split_xyz(state)
    return _continuous_phi(x, y, angle_near, dtype)


def _take_entry(vector: jax.Array, index: int) -> jax.Array:
    indices = explicit_device_array((index,), dtype=np.int32)
    return jax.lax.squeeze(jnp.take(vector, indices, axis=0), (0,))


def _stage_fieldline_spec(spec: FieldlineTracingSpec) -> FieldlineTracingSpec:
    return replace(
        spec,
        tmax=_device_array(spec.tmax, np.float64),
        rtol=_device_array(spec.rtol, np.float64),
        atol=_device_array(spec.atol, np.float64),
        dtmax=_device_array(spec.dtmax, np.float64),
    )


def _stage_guiding_center_spec(
    spec: GuidingCenterTracingSpec,
) -> GuidingCenterTracingSpec:
    return replace(
        spec,
        tmax=_device_array(spec.tmax, np.float64),
        rtol=_device_array(spec.rtol, np.float64),
        atol=_device_array(spec.atol, np.float64),
        dtmax=_device_array(spec.dtmax, np.float64),
    )


def _stage_fullorbit_spec(spec: FullorbitTracingSpec) -> FullorbitTracingSpec:
    return replace(
        spec,
        tmax=_device_array(spec.tmax, np.float64),
        rtol=_device_array(spec.rtol, np.float64),
        atol=_device_array(spec.atol, np.float64),
        dtmax=_device_array(spec.dtmax, np.float64),
    )


AdaptiveLoop = Literal["scan", "while"]


def _run_adaptive_steps(
    cond,
    body,
    init_carry,
    max_steps: int,
    adaptive_loop: AdaptiveLoop,
):
    if adaptive_loop == "while":
        return jax.lax.while_loop(cond, body, init_carry)

    checked_body = jax.checkpoint(body)

    def scan_step(carry, _):
        active = cond(carry)
        carry_next = jax.lax.cond(active, checked_body, lambda value: value, carry)
        return carry_next, None

    final_carry, _ = jax.lax.scan(scan_step, init_carry, xs=None, length=max_steps)
    return final_carry


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
        iterations the JIT-compiled adaptive driver may execute. The
        trajectory carry has shape ``(max_steps + 1, 4)`` (the ``+1``
        captures the initial state).
    dtmax
        Maximum absolute step size. ``inf`` leaves the adaptive controller
        unconstrained except for the final ``tmax - t`` clamp.
    max_root_iters
        Iteration ceiling for the event localizer. The default is upstream's
        ``rootmaxit = 200`` (``legacy native extension/tracing.cpp``); the loop
        exits earlier, per lane, once the bracket meets
        ``boost::math::tools::eps_tolerance``. That tolerance does NOT follow
        ``rtol``: it is the fixed ``4 * eps`` floor of upstream's bits-derived
        value, the same number at every tolerance -- see
        :data:`_ROOT_BRACKET_EPS` for which ``eps`` and why.
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
    max_root_iters: int = 200
    max_phi_hits: int = 128
    adaptive_loop: AdaptiveLoop = "scan"


jax.tree_util.register_dataclass(
    FieldlineTracingSpec,
    data_fields=["tmax", "rtol", "atol", "dtmax"],
    meta_fields=["max_steps", "max_root_iters", "max_phi_hits", "adaptive_loop"],
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

    Mirrors :class:`legacy native extension.MinRStoppingCriterion`. Pure JAX: the
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
    Matches :class:`legacy native extension.ToroidalTransitStoppingCriterion` with
    ``flux=False``.
    """

    max_transits: float


@dataclass(frozen=True)
class IterStoppingCriterion:
    """Stop after the integrator has run ``max_iter`` steps.

    Matches :class:`legacy native extension.IterationStoppingCriterion`. The driver
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

    Mirrors :class:`legacy native extension.LevelsetStoppingCriterion`. The
    ``classifier_fn`` is a JAX-traceable callable ``classifier_fn(x, y, z) ->
    sign`` produced by
    :func:`simsopt_jax.core.surface_classifier.make_levelset_classifier`
    (returns ``+1`` inside, ``-1`` outside). The criterion fires when the
    post-step Cartesian position has ``classifier_fn(x, y, z) < 0``, matching
    the upstream C++ ``f < 0`` predicate.
    """

    classifier_fn: object


for _criterion_class, _data_fields, _meta_fields in (
    (MinRStoppingCriterion, ["crit_r"], []),
    (MaxRStoppingCriterion, ["crit_r"], []),
    (MinZStoppingCriterion, ["crit_z"], []),
    (MaxZStoppingCriterion, ["crit_z"], []),
    (ToroidalTransitStoppingCriterion, ["max_transits"], []),
    (IterStoppingCriterion, [], ["max_iter"]),
    (MinToroidalFluxStoppingCriterion, ["min_s"], ["field_fn"]),
    (MaxToroidalFluxStoppingCriterion, ["max_s"], ["field_fn"]),
    (LevelsetStoppingCriterion, ["classifier_fn"], []),
):
    jax.tree_util.register_dataclass(
        _criterion_class,
        data_fields=_data_fields,
        meta_fields=_meta_fields,
    )


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
    ``legacy native extension/tracing.cpp`` flux-only contract).
    """

    if isinstance(criterion, MinRStoppingCriterion):
        r = jnp.sqrt(x * x + y * y)
        return r <= _device_array(criterion.crit_r, dtype)
    if isinstance(criterion, MaxRStoppingCriterion):
        r = jnp.sqrt(x * x + y * y)
        return r >= _device_array(criterion.crit_r, dtype)
    if isinstance(criterion, MinZStoppingCriterion):
        return z <= _device_array(criterion.crit_z, dtype)
    if isinstance(criterion, MaxZStoppingCriterion):
        return z >= _device_array(criterion.crit_z, dtype)
    if isinstance(criterion, ToroidalTransitStoppingCriterion):
        transits = jnp.abs(phi_unwrapped - phi_init) / _device_two_pi(phi_unwrapped)
        return transits >= _device_array(criterion.max_transits, dtype)
    if isinstance(criterion, IterStoppingCriterion):
        return iter_count > _device_index(int(criterion.max_iter))
    if isinstance(criterion, MinToroidalFluxStoppingCriterion):
        if is_boozer_state:
            return x <= _device_array(criterion.min_s, dtype)
        return _device_false()
    if isinstance(criterion, MaxToroidalFluxStoppingCriterion):
        if is_boozer_state:
            return x >= _device_array(criterion.max_s, dtype)
        return _device_false()
    if isinstance(criterion, LevelsetStoppingCriterion):
        # Surface classifier returns +1 inside, -1 outside; stop on the
        # accepted step that crosses to < 0 (matches upstream
        # ``legacy native extension/tracing.cpp::LevelsetStoppingCriterion``).
        position = jnp.stack([x, y, z]).reshape(1, 3).astype(dtype)
        sign = criterion.classifier_fn(position)[0]
        return sign < _device_array(0.0, dtype)
    raise NotImplementedError(
        f"Unsupported JAX stopping criterion: {type(criterion).__name__}"
    )


def _continuous_phi(
    x: jax.Array, y: jax.Array, phi_near: jax.Array, dtype: jnp.dtype
) -> jax.Array:
    """Continuous-branch ``atan2(y, x)`` near ``phi_near``.

    Mirrors the C++ ``get_phi`` helper in ``legacy native extension/tracing.cpp``:
    pick the integer multiple of ``2*pi`` so the unwrapped ``phi`` is
    within ``pi`` of ``phi_near``. Used so the per-step ``phi_last`` /
    ``phi_current`` accumulator continuously tracks the trajectory and
    the floor-division crossing test does not miss a 2pi wrap.
    """

    phi_raw = jnp.arctan2(y, x)
    two_pi = _device_two_pi(phi_raw)
    zero = _device_array(0.0, dtype)
    half = _device_array(0.5, dtype)
    phi = jnp.where(phi_raw < zero, phi_raw + two_pi, phi_raw)
    nearest_multiple = (
        jnp.sign(phi_near) * jnp.floor(jnp.abs(phi_near / two_pi) + half) * two_pi
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
        _as_device_array(x, dtype),
        _as_device_array(y, dtype),
        _as_device_array(phi_near, dtype),
        dtype,
    )


def _continuous_angle(
    angle_raw: jax.Array, angle_near: jax.Array, dtype: jnp.dtype
) -> jax.Array:
    """Continuous-branch unwrap of a scalar angle near ``angle_near``.

    Companion to :func:`_continuous_phi` for the Boozer-coordinate
    state where ``zeta`` is already a scalar angle (not a Cartesian
    ``atan2(y, x)``). The C++ ``get_phi`` helper has a single
    ``get_angle`` equivalent in ``legacy native extension/tracing.cpp`` driving the
    ``zeta - zeta_target`` modulo-``2*pi`` detection on the Boozer
    route. We replicate the same logic: pick the integer multiple of
    ``2*pi`` so the unwrapped angle lies within ``pi`` of ``angle_near``.
    Used so the per-step ``zeta_last`` / ``zeta_current`` accumulator
    continuously tracks the Boozer trajectory and the floor-division
    crossing test does not miss a ``2*pi`` wrap.
    """

    two_pi = _device_two_pi(angle_raw)
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
    write_row = accepted_count + _device_index(1)

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
        _device_index(1),
        _device_index(0),
    )
    return traj_next, mask_next, accepted_next


def _should_record_accepted_step(
    accepted: jax.Array, stop_after: jax.Array
) -> jax.Array:
    return jnp.logical_and(accepted, jnp.logical_not(stop_after))


def _boozer_axis_invalid(y: jax.Array) -> jax.Array:
    s = y[0]
    zero = _device_array(0.0, s.dtype)
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


class _Dopri5Stages(NamedTuple):
    """One Dormand-Prince step with every stage the dense output needs.

    ``k2`` is not carried: boost's ``runge_kutta_dopri5::calc_state`` combines
    ``x_old``, ``k1``, ``k3``, ``k4``, ``k5``, ``k6`` and ``k7`` only.
    """

    y_new: jax.Array
    y_err: jax.Array
    k1: jax.Array
    k3: jax.Array
    k4: jax.Array
    k5: jax.Array
    k6: jax.Array
    k7: jax.Array
    field_cache: InterpolatedFieldCylCache | None = None


def _dopri5_stage_step(
    rhs: Callable[..., jax.Array],
    t: jax.Array,
    y: jax.Array,
    h: jax.Array,
    k_first: jax.Array,
    field_cache: InterpolatedFieldCylCache | None = None,
) -> _Dopri5Stages:
    """Single Dormand-Prince RK4(5) step, keeping the dense-output stages.

    Body of :func:`dopri5_step`; that function is the published 3-value view of
    this one so the Butcher tableau exists exactly once.

    ``field_cache`` selects the right-hand-side protocol, and both protocols
    are faithful mirrors of a C++ right-hand side; which one applies is a
    property of the FIELD, not of this stepper.

    * ``None`` (the default): ``rhs(t, y) -> dy/dt``. This is upstream's
      behaviour for every field whose ``_B_impl`` writes every output row --
      an analytic field has no stale output buffer to reproduce.
    * an :class:`InterpolatedFieldCylCache`: ``rhs(t, y, cache) ->
      (dy/dt, cache)``. An interpolated field leaves its output buffer
      untouched outside its domain, so the buffer travels from one
      right-hand-side evaluation to the next (see
      :class:`~simsopt_jax.core.interpolated_field.InterpolatedFieldCylCache`).
      The six evaluations below thread it in upstream's own evaluation order --
      ``k2, k3, k4, k5, k6, k7`` -- because that order is what decides which
      value an out-of-domain stage reads. ``k1`` is the FSAL stage and costs no
      evaluation, so it does not touch the buffer.
    """

    dtype = y.dtype
    c = lambda value: _device_array(value, dtype)
    # One Butcher tableau for both protocols: the stage calls below are written
    # once and ``stage`` decides how the right-hand side is invoked. The cache
    # travels in a trace-time cell because the stage calls happen in Python
    # source order (each stage's argument list reads the previous stages), so
    # the cell sees upstream's evaluation order, and each traced value is used
    # exactly once.
    cache_cell = [field_cache]

    if field_cache is None:

        def stage(t_stage: jax.Array, y_stage: jax.Array) -> jax.Array:
            return rhs(t_stage, y_stage)

    else:

        def stage(t_stage: jax.Array, y_stage: jax.Array) -> jax.Array:
            dydt, cache_next = rhs(t_stage, y_stage, cache_cell[0])
            cache_cell[0] = cache_next
            return dydt

    k1 = k_first
    k2 = stage(t + c(1.0 / 5.0) * h, y + h * (c(1.0 / 5.0) * k1))
    k3 = stage(
        t + c(3.0 / 10.0) * h,
        y + h * (c(3.0 / 40.0) * k1 + c(9.0 / 40.0) * k2),
    )
    k4 = stage(
        t + c(4.0 / 5.0) * h,
        y + h * (c(44.0 / 45.0) * k1 - c(56.0 / 15.0) * k2 + c(32.0 / 9.0) * k3),
    )
    k5 = stage(
        t + c(8.0 / 9.0) * h,
        y
        + h
        * (
            c(19372.0 / 6561.0) * k1
            - c(25360.0 / 2187.0) * k2
            + c(64448.0 / 6561.0) * k3
            - c(212.0 / 729.0) * k4
        ),
    )
    k6 = stage(
        t + h,
        y
        + h
        * (
            c(9017.0 / 3168.0) * k1
            - c(355.0 / 33.0) * k2
            + c(46732.0 / 5247.0) * k3
            + c(49.0 / 176.0) * k4
            - c(5103.0 / 18656.0) * k5
        ),
    )
    y_new = y + h * (
        c(35.0 / 384.0) * k1
        + c(500.0 / 1113.0) * k3
        + c(125.0 / 192.0) * k4
        - c(2187.0 / 6784.0) * k5
        + c(11.0 / 84.0) * k6
    )
    k7 = stage(t + h, y_new)
    y_err = h * (
        c(71.0 / 57600.0) * k1
        - c(71.0 / 16695.0) * k3
        + c(71.0 / 1920.0) * k4
        - c(17253.0 / 339200.0) * k5
        + c(22.0 / 525.0) * k6
        - c(1.0 / 40.0) * k7
    )
    return _Dopri5Stages(
        y_new=y_new,
        y_err=y_err,
        k1=k1,
        k3=k3,
        k4=k4,
        k5=k5,
        k6=k6,
        k7=k7,
        field_cache=cache_cell[0],
    )


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

    stages = _dopri5_stage_step(rhs, t, y, h, k_first)
    return stages.y_new, stages.y_err, stages.k7


def _dense_output_state(
    stages: _Dopri5Stages,
    y: jax.Array,
    h: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    """DOPRI5 dense output of the step ``(y, h)`` at fraction ``theta``.

    Term-for-term port of ``runge_kutta_dopri5::calc_state`` in
    ``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp``, which is the root
    function upstream's ``legacy native extension/tracing.cpp`` hands to
    ``toms748_solve``. Costs no right-hand-side evaluation: every stage it needs
    was produced by the step itself.

    Exact at both ends WHILE EVERY STAGE IS FINITE -- ``theta = 0`` returns ``y``
    and ``theta = 1`` returns ``stages.y_new``. That exactness comes from the
    weights (every ``b_i_theta`` is ``0`` at ``theta = 0``; at ``theta = 1`` they
    are the step's own ``b_i``), not from short-circuiting the stages, and
    ``0 * nan`` is ``nan``. So once ANY stage is non-finite the result is
    non-finite at EVERY ``theta``, the two ends included -- in particular the
    FSAL stage ``k7``, which ``b7_theta`` multiplies and which is ``nan`` exactly
    on the step boost accepts with a finite ``y_new`` and a ``nan`` error.
    Upstream's ``calc_state`` has the same property (it weights ``deriv_new``
    with ``dt * b7_theta`` inside one ``scale_sum7``), so this is a mirror, not a
    port defect: a caller localizing an event on such a step is handed
    non-finite residuals and gets a bracket END back rather than a root.
    :func:`_scan_angle_plane_events` therefore records no row from such a step
    and ends the lane instead.
    """

    dtype = y.dtype
    c = lambda value: _device_array(value, dtype)
    b1 = c(35.0 / 384.0)
    b3 = c(500.0 / 1113.0)
    b4 = c(125.0 / 192.0)
    b5 = c(-2187.0 / 6784.0)
    b6 = c(11.0 / 84.0)
    X1 = c(5.0) * (c(2558722523.0) - c(31403016.0) * theta) / c(11282082432.0)
    X3 = c(100.0) * (c(882725551.0) - c(15701508.0) * theta) / c(32700410799.0)
    X4 = c(25.0) * (c(443332067.0) - c(31403016.0) * theta) / c(1880347072.0)
    X5 = c(32805.0) * (c(23143187.0) - c(3489224.0) * theta) / c(199316789632.0)
    X6 = c(55.0) * (c(29972135.0) - c(7076736.0) * theta) / c(822651844.0)
    X7 = c(10.0) * (c(7414447.0) - c(829305.0) * theta) / c(29380423.0)
    theta_m_1 = theta - c(1.0)
    theta_sq = theta * theta
    A = theta_sq * (c(3.0) - c(2.0) * theta)
    B = theta_sq * theta_m_1
    C = theta_sq * theta_m_1 * theta_m_1
    D = theta * theta_m_1 * theta_m_1
    b1_theta = A * b1 - C * X1 + D
    b3_theta = A * b3 + C * X3
    b4_theta = A * b4 - C * X4
    b5_theta = A * b5 + C * X5
    b6_theta = A * b6 - C * X6
    b7_theta = B + C * X7
    return y + h * (
        b1_theta * stages.k1
        + b3_theta * stages.k3
        + b4_theta * stages.k4
        + b5_theta * stages.k5
        + b6_theta * stages.k6
        + b7_theta * stages.k7
    )


def _dense_output_query(
    stages: _Dopri5Stages,
    y: jax.Array,
    t: jax.Array,
    h_clamped: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """This step's DOPRI5 dense output as a function of ABSOLUTE time.

    ``dense.calc_state`` in ``legacy native extension/tracing.cpp``: the
    continuous extension of the step just taken, costing no extra
    right-hand-side evaluation, exact at ``t`` and at ``t + h_clamped``.

    The event scan runs on every trial, so this closure is also built for the
    h-underflow stall of :data:`_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS`, where
    ``h_clamped`` is exactly ``0.0``. A zero-length step has a single query
    time, ``t``, and its dense output there is ``y``, so the fraction is taken
    against ``1.0``: ``theta`` is then ``0`` and ``_dense_output_state`` returns
    ``y`` because it scales every stage by ``h``. Dividing by the zero step
    instead makes ``theta`` ``0/0``, turns the whole scan ``nan``, and aborts the
    run under ``jax_debug_nans`` before the stall guard can report
    :data:`TRACING_STATUS_STEP_CONTROL_FAILED`.
    """

    zero = _device_array(0.0, h_clamped.dtype)
    one = _device_array(1.0, h_clamped.dtype)
    h_fraction = jnp.where(h_clamped == zero, one, h_clamped)

    def state_at_time(t_query: jax.Array) -> jax.Array:
        return _dense_output_state(stages, y, h_clamped, (t_query - t) / h_fraction)

    return state_at_time


def _error_norm(
    y_err: jax.Array,
    y: jax.Array,
    dydt: jax.Array,
    h: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
) -> jax.Array:
    """Relative error of a trial step, exactly as upstream measures it.

    Mirrors boost.odeint ``default_error_checker::error``: componentwise
    ``|y_err| / (atol + rtol * (|y_old| + |h| * |dydt_old|))`` reduced with the
    infinity norm. ``y`` and ``dydt`` are the state and derivative at the start
    of the trial (the FSAL leading stage), and ``h`` is the trial step.

    The reduction is boost's ``norm_inf``, and its floating-point behaviour is
    load-bearing, not an implementation detail. ``array_algebra::norm_inf``
    (``boost/numeric/odeint/algebra/array_algebra.hpp``, the algebra
    ``algebra_dispatcher`` selects for the ``std::array`` states upstream uses)
    is ``init = 0; for each i: init = max(init, abs(s[i]))``, and ``std::max(a,
    b)`` is ``(a < b) ? b : a``. A ``nan`` component therefore NEVER replaces
    the running maximum and contributes nothing, while an ``inf`` component
    does replace it. So an all-``nan`` error vector reduces to ``0`` and the
    trial is ACCEPTED, whereas any ``inf`` rejects. ``jnp.max`` propagates
    ``nan`` instead, which would reject a step upstream accepts; the ``where``
    below restores upstream's rule. This is reachable: the DOPRI5 error
    estimate contains the FSAL stage ``k7 = f(t + h, y_new)``
    (``runge_kutta_dopri5.hpp`` ``do_step_impl`` with ``xerr``), so a right-hand
    side that is ``nan`` at the step END poisons every error component while
    ``y_new`` itself, built from ``k1, k3..k6`` only, stays finite.
    """
    sc = atol + rtol * (jnp.abs(y) + jnp.abs(h) * jnp.abs(dydt))
    componentwise = jnp.abs(y_err) / sc
    zero = _device_array(0.0, componentwise.dtype)
    return jnp.max(jnp.where(jnp.isnan(componentwise), zero, componentwise))


def _initial_step_size(
    t0: jax.Array, t_end: jax.Array, dtmax: jax.Array, fraction: float
) -> jax.Array:
    span = jnp.abs(t_end - t0)
    h0 = _device_array(fraction, span.dtype) * dtmax
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
    eps: jax.Array,
    *,
    active: jax.Array | None = None,
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
        Iteration ceiling for the Illinois false-position loop. Upstream's
        ``rootmaxit`` (``legacy native extension/tracing.cpp``) is the value to
        pass.
    eps
        Relative bracket tolerance with
        ``boost::math::tools::eps_tolerance``'s meaning: the loop stops for a
        lane once ``|b - a| <= eps * min(|a|, |b|)``. The tracing drivers pass
        :data:`_ROOT_BRACKET_EPS`; see the note at that constant for why it is
        ``eps_tolerance()``'s default rather than upstream's bits-derived value.
        ``eps = 0`` never terminates early, so the lane runs the full
        ``max_iters``.
    active
        Optional per-lane gate. A lane whose ``active`` is ``False`` performs no
        iteration and returns the better of its two input endpoints; under
        ``vmap`` the batched loop then runs only as long as some lane is still
        iterating, so a step that crossed no plane costs nothing. ``None`` means
        "every lane iterates".

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
    zero = _device_array(0.0, dtype)
    half = _device_array(0.5, dtype)
    eps_arr = _as_device_array(eps, dtype)
    left_first = t_left <= t_right
    t_left_ordered = jnp.where(left_first, t_left, t_right)
    t_right_ordered = jnp.where(left_first, t_right, t_left)
    f_left_ordered = jnp.where(left_first, f_left, f_right)
    f_right_ordered = jnp.where(left_first, f_right, f_left)
    t_left_ordered = t_left_ordered + f_left_ordered * zero
    t_right_ordered = t_right_ordered + f_right_ordered * zero
    bracketed_in = jnp.sign(f_left_ordered) * jnp.sign(f_right_ordered) < zero
    if active is None:
        iterating_in = bracketed_in
    else:
        iterating_in = jnp.logical_and(bracketed_in, active)
    left_better = jnp.abs(f_left_ordered) <= jnp.abs(f_right_ordered)
    init = (
        _device_index(0),
        t_left_ordered,
        t_right_ordered,
        f_left_ordered,
        f_right_ordered,
        jnp.where(left_better, t_left_ordered, t_right_ordered),
        jnp.where(left_better, f_left_ordered, f_right_ordered),
    )

    def _converged(a, b):
        # ``boost::math::tools::eps_tolerance::operator()``.
        return jnp.abs(b - a) <= eps_arr * jnp.minimum(jnp.abs(a), jnp.abs(b))

    def cond(carry):
        i, a, b, _fa, _fb, _best_t, _best_f = carry
        budget_left = i < _device_index(int(max_iters))
        lane_active = jnp.logical_and(
            iterating_in, jnp.logical_not(_converged(a, b))
        )
        return jnp.logical_and(budget_left, lane_active)

    def body(carry):
        i, a, b, fa, fb, best_t, best_f = carry
        width = b - a
        active = jnp.logical_and(iterating_in, jnp.logical_not(_converged(a, b)))
        midpoint = a + half * width
        denominator = fb - fa
        midpoint = midpoint + denominator * zero
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
            i + _device_index(1),
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


class _Dopri5AdaptiveStep(NamedTuple):
    """One adaptive DOPRI5 trial and the carry update it licenses.

    ``accepted`` is the step the lane may TAKE: boost accepted it AND its new
    state is finite. ``nonfinite_state`` is the one case where the two differ --
    boost accepted the trial but ``y_new`` is not finite -- and it is terminal
    here; :func:`_dopri5_adaptive_step` documents why the port does not follow
    upstream's integrator on it.
    """

    h_clamped: jax.Array
    y_new: jax.Array
    accepted: jax.Array
    nonfinite_state: jax.Array
    h_next: jax.Array
    t_next: jax.Array
    y_next: jax.Array
    k_next: jax.Array
    stages: _Dopri5Stages


def _dopri5_adaptive_step(
    rhs: Callable[..., jax.Array],
    t: jax.Array,
    y: jax.Array,
    h: jax.Array,
    k_first: jax.Array,
    tmax: jax.Array,
    dtmax: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
    dtype,
    field_cache: InterpolatedFieldCylCache | None = None,
) -> _Dopri5AdaptiveStep:
    """Run one adaptive DOPRI5 trial and return the accepted-state update.

    ``field_cache`` is the right-hand side's field output buffer and selects
    the right-hand-side protocol; :func:`_dopri5_stage_step` documents both.
    Its update is returned in ``stages.field_cache`` and is NOT conditioned on
    acceptance: the C++ buffer is written by every right-hand-side evaluation,
    a rejected trial's evaluations included.
    """

    h_clamped = _clamp_step_to_domain(h, t, tmax, dtmax)
    stages = _dopri5_stage_step(rhs, t, y, h_clamped, k_first, field_cache)
    y_new = stages.y_new
    k7 = stages.k7
    err = _error_norm(stages.y_err, y, k_first, h_clamped, rtol, atol)
    # ``controlled_runge_kutta::try_step`` (FSAL overload) rejects on
    # ``if (max_rel_err > 1.0)`` and otherwise accepts, so the accept test is
    # the NEGATION of a greater-than, not a less-or-equal: that is what decides
    # a ``nan`` error the same way upstream does. ``_error_norm`` already
    # reproduces boost's ``norm_inf``, so the value reaching here is finite or
    # ``+inf``; writing the predicate upstream's way keeps the two agreeing even
    # if that ever stops being true.
    upstream_accepted = jnp.logical_not(err > _device_array(1.0, dtype))
    # Upstream's integrator decides acceptance from the ERROR ALONE, so it also
    # accepts a trial whose new STATE is non-finite, and carries that state on:
    # with an all-``nan`` error ``array_algebra::norm_inf``
    # (``boost/numeric/odeint/algebra/array_algebra.hpp``) is ``0``,
    # ``controlled_runge_kutta::try_step``
    # (``boost/numeric/odeint/stepper/controlled_runge_kutta.hpp``) takes the
    # accept branch, and the ``nan`` becomes ``x_old`` of the next step, so the
    # run reaches ``tmax`` or fires a stopping criterion on a ``nan`` state.
    # This port does NOT follow it there: a lane here never reports success on,
    # nor publishes, non-finite values, so such a trial is not taken and
    # ``nonfinite_state`` ends the lane at the last finite state with
    # ``TRACING_STATUS_STEP_CONTROL_FAILED`` (``_step_control_progress``).
    # Measured on the shipped particle case before this gate existed: one
    # criterion row ``[1.7869731e-04, -1, nan, nan, nan, nan]`` published into
    # ``poincare:positions`` with the lane reporting the criterion status ``-1``.
    # Upstream's acceptance is untouched where it decides the mirror: a ``nan``
    # error with a FINITE new state -- the reachable case, since the error
    # carries the FSAL stage ``k7 = f(t + h, y_new)`` while ``y_new``, built
    # from ``k1, k3..k6``, does not -- is accepted exactly as boost accepts it,
    # so the lane lands where upstream lands and upstream's stopping criterion
    # fires there.
    state_finite = jnp.all(jnp.isfinite(y_new))
    accepted = jnp.logical_and(upstream_accepted, state_finite)
    nonfinite_state = jnp.logical_and(upstream_accepted, jnp.logical_not(state_finite))
    # ``default_step_adjuster::decrease_step`` / ``increase_step``.
    decrease = jnp.maximum(
        _device_array(_SAFETY, dtype)
        * jnp.power(
            jnp.clip(
                err,
                _device_array(1.0, dtype),
                _device_array(_MAX_DECREASE_ERROR, dtype),
            ),
            _device_array(-_DECREASE_EXP, dtype),
        ),
        _device_array(_MIN_FACTOR, dtype),
    )
    increase = jnp.where(
        err < _device_array(_INCREASE_ERROR_THRESHOLD, dtype),
        jnp.where(
            err > _device_array(_MIN_INCREASE_ERROR, dtype),
            _device_array(_SAFETY, dtype)
            * jnp.power(
                jnp.clip(
                    err,
                    _device_array(_MIN_INCREASE_ERROR, dtype),
                    _device_array(_INCREASE_ERROR_THRESHOLD, dtype),
                ),
                _device_array(-_INCREASE_EXP, dtype),
            ),
            _device_array(_MAX_INCREASE_FACTOR, dtype),
        ),
        _device_array(1.0, dtype),
    )
    # The step-size adjuster stays boost's, on boost's own accept flag; the
    # state, the time and the FSAL derivative advance only on a step the lane
    # may take.
    h_next = h_clamped * jnp.where(upstream_accepted, increase, decrease)
    t_next = jnp.where(accepted, _accepted_step_time(t, h_clamped, tmax), t)
    y_next = jnp.where(accepted, y_new, y)
    k_next = jnp.where(accepted, k7, k_first)
    return _Dopri5AdaptiveStep(
        h_clamped=h_clamped,
        y_new=y_new,
        accepted=accepted,
        nonfinite_state=nonfinite_state,
        h_next=h_next,
        t_next=t_next,
        y_next=y_next,
        k_next=k_next,
        stages=stages,
    )


def _event_row_from_state(
    t_event: jax.Array,
    event_index: jax.Array,
    state: jax.Array,
) -> jax.Array:
    return jnp.concatenate(
        [
            jnp.reshape(t_event, (1,)),
            jnp.reshape(event_index, (1,)),
            state,
        ],
        axis=0,
    )


def _scan_angle_plane_events(
    *,
    hits: jax.Array,
    count: jax.Array,
    status: jax.Array,
    stop: jax.Array,
    angle_last: jax.Array,
    angle_current: jax.Array,
    targets: jax.Array,
    num_targets: int,
    two_pi: jax.Array,
    dtype,
    t: jax.Array,
    h_clamped: jax.Array,
    max_root_iters: int,
    enabled: jax.Array,
    max_hits_i32: jax.Array,
    state_at_time: Callable[[jax.Array], jax.Array],
    angle_at_state: Callable[[jax.Array, jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Record angle-plane crossings for fieldline, GC, Boozer, and full-orbit drivers.

    Mirrors ``legacy native extension/tracing.cpp``: the crossing test is the
    ``floor((angle - target) / 2*pi)`` comparison on the accepted step, the root
    is bracketed in ABSOLUTE time over ``[t, t + h_clamped]`` with upstream's own
    two end residuals ``angle_last - shifted_target`` and
    ``angle_current - shifted_target``, and the residual is evaluated on the
    step's dense output, never by re-integrating. ``enabled``
    carries "this trial was accepted"; a target that was not crossed, or a trial
    that was not accepted, performs no root iteration at all, which is upstream's
    ``if(crossed)`` gate expressed as the loop predicate so that ``vmap`` -- which
    turns ``lax.cond`` into a ``select`` -- really skips the work.

    A crossing is recorded only when the ROW it would publish is finite, and a crossing whose row is NOT finite
    stops the lane with :data:`TRACING_STATUS_STEP_CONTROL_FAILED` instead. Upstream's dense output carries the
    FSAL stage ``k7`` with weight ``dt * b7_theta``
    (``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp`` ``calc_state``:
    ``for_each8(x, x_old, deriv_old, m_k3, m_k4, m_k5, m_k6, deriv_new, scale_sum7(...))``), and ``0 * nan`` is
    ``nan`` in C++ too. So on a step whose ``k7`` is non-finite -- the step boost ACCEPTS, because it judges the
    error alone, which this port mirrors exactly (:func:`_dopri5_adaptive_step`) -- upstream's ``dense.calc_state``
    is non-finite at EVERY query time, its ``toms748_solve`` is handed non-finite residuals, and it pushes that row
    (``legacy native extension/tracing.cpp``,
    ``res_phi_hits.push_back(join<2, RHS::Size>({troot, double(i)}, temp))``) and carries the ``nan`` on. This port
    may not publish it. Dropping the row alone would LOSE a crossing upstream records without saying so -- the same
    step can be the one that reaches ``tmax`` (status ``0``), that exhausts ``max_steps`` (status ``1``) or that a
    criterion stops (status ``-1 - i``) -- so the lane's status reports it, with the status that already means "this
    port refused a step upstream would have taken". The scan runs BEFORE the stopping criteria in every driver,
    which is upstream's own order (``tracing.cpp``: the phi loop, then the criterion loop), so a criterion cannot
    fire on a step whose continuous extension is unusable either.
    """

    if num_targets == 0:
        return hits, count, status, stop

    t_end = t + h_clamped

    def scan_one_target(i, carry):
        hits_carry, count_carry, status_carry, stop_carry = carry
        target = targets[i]
        fl_last = jnp.floor((angle_last - target) / two_pi)
        fl_curr = jnp.floor((angle_current - target) / two_pi)
        # ``nan != nan`` is TRUE, so a lane that has gone non-finite would pass
        # the floor test on every plane on every remaining trial and record an
        # all-``nan`` row. Upstream never meets that case: boost throws through
        # ``max_step_checker`` before a non-finite state is integrated, which
        # this port reports as ``TRACING_STATUS_STEP_CONTROL_FAILED``.
        angles_finite = jnp.logical_and(
            jnp.isfinite(angle_last), jnp.isfinite(angle_current)
        )
        crossed = jnp.logical_and(
            jnp.logical_and(fl_last != fl_curr, enabled), angles_finite
        )
        offset = jnp.round(
            ((angle_last + angle_current) / _device_array(2.0, dtype) - target) / two_pi
        )
        shifted_target = offset * two_pi + target

        def diff_at(t_query):
            state = state_at_time(t_query)
            return angle_at_state(state, angle_last) - shifted_target

        # Upstream hands the root finder the two residuals the crossing test
        # just compared and never evaluates the root function at the bracket
        # ends: ``toms748_solve(rootfun, tlast, tcurrent, phi_last - phi_shift,
        # phi_current - phi_shift, roottol, rootmaxit)``
        # (``legacy native extension/tracing.cpp``), whose preceding assertion is
        # that the shifted target lies between exactly these two values.
        # Re-evaluating the ends through the dense output does not reproduce
        # them: ``((t + h) - t) / h`` is not exactly ``1`` once ``t >> h``, so the
        # right end would differ from ``angle_current`` by up to an ulp of ``t``
        # in angle (1.2e-12 at the official QA ``tmax = 20000``, ~270x the
        # localizer's own residual) and can take the OPPOSITE sign, leaving the
        # bracket inconsistent with the ``crossed`` gate that records the row.
        f_left = angle_last - shifted_target
        f_right = angle_current - shifted_target
        t_root, _f_root, _bracketed = bracket_root_jax(
            diff_at,
            t,
            t_end,
            f_left,
            f_right,
            max_root_iters,
            _device_array(_ROOT_BRACKET_EPS, dtype),
            active=crossed,
        )
        state_root = state_at_time(t_root)
        hit_row = _event_row_from_state(
            t_root,
            _as_device_array(i, dtype),
            state_root,
        )
        row_finite = jnp.all(jnp.isfinite(hit_row))
        unusable = jnp.logical_and(crossed, jnp.logical_not(row_finite))
        hits_next, count_next = _append_event_row(
            hits_carry,
            count_carry,
            jnp.logical_and(crossed, row_finite),
            hit_row,
            max_hits_i32,
        )
        return (
            hits_next,
            count_next,
            jnp.where(
                unusable,
                _device_index(TRACING_STATUS_STEP_CONTROL_FAILED),
                status_carry,
            ),
            jnp.logical_or(stop_carry, unusable),
        )

    return jax.lax.fori_loop(
        0,
        num_targets,
        scan_one_target,
        (hits, count, status, stop),
    )


def _apply_stopping_criteria_events(
    *,
    stopping_criteria: tuple,
    hits: jax.Array,
    count: jax.Array,
    status: jax.Array,
    stop: jax.Array,
    iter_count: jax.Array,
    angle_current: jax.Array,
    angle_initial: jax.Array,
    t_event: jax.Array,
    state: jax.Array,
    dtype,
    max_hits_i32: jax.Array,
    is_boozer_state: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Append first-firing stopping-criterion events while preserving row layout.

    Upstream evaluates each criterion on the accepted POST-step state and pushes
    ``{t, -1 - i, y}`` built from it (``legacy native extension/tracing.cpp``,
    the criterion loop that follows the step). Its state there is finite,
    because boost's acceptance looks only at the error norm and would have
    carried a non-finite state on; this port ends such a lane at the last finite
    state instead (:func:`_dopri5_adaptive_step`), so a criterion must not fire
    on a non-finite state, must not set a status from one and must not push a
    row from one -- the same rule ``_scan_angle_plane_events`` applies to its
    own angles. ``IterStoppingCriterion`` makes the case reachable inside this
    helper even when the caller does gate the step: its predicate does not read
    the state at all.
    """

    state_finite = jnp.all(jnp.isfinite(state))
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
        fires = jnp.logical_and(
            jnp.logical_and(jnp.logical_not(stop), pred), state_finite
        )
        idx_val = _device_index(-1 - i)
        hit_row = _event_row_from_state(
            t_event,
            _device_array(float(-1 - i), dtype),
            state,
        )
        hits, count = _append_event_row(
            hits,
            count,
            fires,
            hit_row,
            max_hits_i32,
        )
        status = jnp.where(fires, idx_val, status)
        stop = jnp.logical_or(stop, fires)
    return hits, count, status, stop


# ── Adaptive driver ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CartesianTracingContinuationState:
    """Accepted state and controller history needed to resume a Cartesian trace."""

    trial_count: jax.Array
    accepted_count: jax.Array
    t: jax.Array
    y: jax.Array
    h: jax.Array
    k_first: jax.Array
    phi_last: jax.Array
    phi_initial: jax.Array
    status_event: jax.Array
    stopped: jax.Array
    no_progress: jax.Array
    # The field's output buffer, for a field that has one (see
    # :class:`~simsopt_jax.core.interpolated_field.InterpolatedFieldCylCache`).
    # Upstream runs one uninterrupted ``solve``, so the buffer must survive a
    # chunk boundary exactly as it survives a step. ``None`` for a field that
    # writes every output row.
    field_cache: InterpolatedFieldCylCache | None = None


jax.tree_util.register_dataclass(
    CartesianTracingContinuationState,
    data_fields=[
        "trial_count", "accepted_count", "t", "y", "h", "k_first",
        "phi_last", "phi_initial", "status_event", "stopped", "no_progress",
        "field_cache",
    ],
    meta_fields=[],
)


def _step_control_progress(
    *,
    no_progress: jax.Array,
    accepted: jax.Array,
    nonfinite_state: jax.Array,
    t: jax.Array,
    t_next: jax.Array,
    y: jax.Array,
    k_first: jax.Array,
    status: jax.Array,
    stop: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """boost's step-progress checks, ported into the JAX loop carry.

    Returns the updated consecutive-no-progress counter, ``status`` and ``stop``.
    A lane stops with :data:`TRACING_STATUS_STEP_CONTROL_FAILED` when the counter
    reaches :data:`_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS`, when the state it
    carries -- not a speculative trial -- is already non-finite, or when
    ``nonfinite_state`` says the trial boost would have accepted lands on a
    non-finite state (:func:`_dopri5_adaptive_step`: upstream carries that state
    on, this port ends the lane at the last finite state instead). A non-finite
    *rejected* trial state or a non-finite error is NOT terminal: upstream
    rejects exactly that and recovers by shrinking the step
    (``boost/numeric/odeint/stepper/controlled_runge_kutta.hpp`` ``try_step``),
    and a trial error that stays non-finite still ends the lane through the
    counter.
    """

    progressed = jnp.logical_and(accepted, t_next > t)
    no_progress_next = jnp.where(
        progressed, _device_index(0), no_progress + _device_index(1)
    )
    carried_finite = jnp.logical_and(
        jnp.all(jnp.isfinite(y)), jnp.all(jnp.isfinite(k_first))
    )
    failed = jnp.logical_or(
        jnp.logical_or(
            no_progress_next >= _device_index(_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS),
            jnp.logical_not(carried_finite),
        ),
        nonfinite_state,
    )
    fires = jnp.logical_and(jnp.logical_not(stop), failed)
    return (
        no_progress_next,
        jnp.where(fires, _device_index(TRACING_STATUS_STEP_CONTROL_FAILED), status),
        jnp.logical_or(stop, fires),
    )


def _trace_fieldline_chunk(
    spec: FieldlineTracingSpec,
    y0: TracingStateInput,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
    continuation: CartesianTracingContinuationState | None = None,
) -> tuple[FieldlineTracingResult, CartesianTracingContinuationState]:
    """Trace a single fieldline from ``y0`` for ``spec.tmax`` upstream-time units.

    Parameters
    ----------
    spec
        Tracing contract; see :class:`FieldlineTracingSpec`.
    y0
        Initial Cartesian position ``[3]`` as an array or flat list/tuple of
        array-like scalar components. Treated as float64.
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

    Returns a chunk-local result and immutable state for an exact next chunk.
    ``spec.max_steps`` bounds this call's trials; it is not the total horizon.
    """

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
    if continuation is None:
        h0 = _initial_step_size(t0, tmax, dtmax, _FIELDLINE_INITIAL_STEP_FRACTION)
        k0 = rhs(t0, y0_arr)
        start_t = t0
        start_y = y0_arr
        global_trials = _device_index(0)
        global_accepted = _device_index(0)
        status_initial = _device_index(0)
        stopped_initial = _device_false()
        no_progress_initial = _device_index(0)
    else:
        h0 = continuation.h
        k0 = continuation.k_first
        start_t = continuation.t
        start_y = continuation.y
        global_trials = continuation.trial_count
        global_accepted = continuation.accepted_count
        status_initial = continuation.status_event
        stopped_initial = continuation.stopped
        no_progress_initial = continuation.no_progress
    one = _device_array(1.0, dtype)
    lane_zero, lane_zero_i32, lane_false = _lane_axis_carry_zeroes(y0_arr)
    accepted_count_init = _device_array(0, jnp.int32) + lane_zero_i32

    # Pre-allocate the trajectory carry. Row 0 holds the initial state;
    # rows 1..max_steps fill in as accepted steps occur. Padding rows
    # at the end of the run get the final accepted state.
    traj_init_row = jnp.concatenate((jnp.reshape(start_t, (1,)), start_y), axis=0)
    traj = jnp.concatenate(
        (
            jnp.reshape(traj_init_row, (1, 4)),
            _device_zeros((max_steps, 4), dtype),
        ),
        axis=0,
    )
    traj = traj + lane_zero
    mask_indices = jax.lax.iota(jnp.int32, max_steps + 1)
    mask = mask_indices == _device_array(0, jnp.int32)
    mask = mask | lane_false

    # Phi-plane crossing buffer. Each row is ``[t_hit, idx, x, y, z]``.
    phi_hits_buf = _device_zeros((max_phi_hits, 5), dtype)
    phi_hits_count_init = _device_array(0, jnp.int32)
    phi_hits_buf, phi_hits_count_init = _event_carry_with_lane_axis(
        phi_hits_buf,
        phi_hits_count_init,
        y0_arr,
    )

    if phis is None:
        phis_arr = _device_zeros((0,), dtype)
    else:
        phis_arr = _as_device_array(phis, dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    # Initial unwrapped phi seed (C++ tracing.cpp uses pi).
    if continuation is None:
        phi_init = _continuous_phi(
            _take_entry(y0_arr, 0),
            _take_entry(y0_arr, 1),
            _device_array(np.pi, dtype),
            dtype,
        )
        phi_last = phi_init
    else:
        phi_init = continuation.phi_initial
        phi_last = continuation.phi_last

    init_carry = (
        _device_array(0, jnp.int32),  # step_count
        accepted_count_init,
        start_t + lane_zero,
        start_y,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        phi_last,  # running phi_last
        phi_init,  # transit criterion baseline, set on first accepted step
        status_initial + lane_zero_i32,  # status_event
        stopped_initial | lane_false,  # stop flag
        no_progress_initial + lane_zero_i32,  # consecutive non-advancing trials
    )

    max_steps_i32 = _device_array(max_steps, jnp.int32)
    max_phi_hits_i32 = _device_array(max_phi_hits, jnp.int32)
    two_pi = _device_two_pi(one)

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
            _no_progress,
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
            no_progress,
        ) = carry
        step = _dopri5_adaptive_step(
            rhs,
            t,
            y,
            h,
            k_first,
            tmax,
            dtmax,
            rtol,
            atol,
            dtype,
        )
        h_clamped = step.h_clamped
        y_new = step.y_new
        accepted = step.accepted
        h_next = step.h_next
        t_next = step.t_next
        y_next = step.y_next
        k_next = step.k_next

        # ── Phi-plane crossing detection on accepted steps ──
        phi_current = _continuous_phi(y_new[0], y_new[1], phi_last, dtype)

        state_at_time = _dense_output_query(step.stages, y, t, h_clamped)

        (
            phi_hits_after,
            phi_count_after,
            status_scan,
            stop_scan,
        ) = _scan_angle_plane_events(
            hits=phi_hits_in,
            count=phi_hits_count_in,
            status=status_event,
            stop=lane_false,
            angle_last=phi_last,
            angle_current=phi_current,
            targets=phis_arr,
            num_targets=num_phis,
            two_pi=two_pi,
            dtype=dtype,
            t=t,
            h_clamped=h_clamped,
            max_root_iters=max_root_iters,
            enabled=accepted,
            max_hits_i32=max_phi_hits_i32,
            state_at_time=state_at_time,
            angle_at_state=lambda state, angle_near: _continuous_phi_from_state(
                state, angle_near, dtype
            ),
        )

        # ── Stopping criteria check on accepted state ──
        first_accepted_step = (global_accepted + accepted_count) == _device_index(0)
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
            return _apply_stopping_criteria_events(
                stopping_criteria=stopping_criteria,
                hits=hits_in,
                count=count_in,
                status=status_in,
                stop=stop_in,
                iter_count=iter_count_in,
                angle_current=phi_curr_in,
                angle_initial=phi_init_in,
                t_event=t_next,
                state=y_next,
                dtype=dtype,
                max_hits_i32=max_phi_hits_i32,
            )

        iter_count_post = global_trials + step_count + _device_index(1)

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
                status_scan,
                stop_scan,
                iter_count_post,
                phi_current,
                phi_init_for_criteria,
            ),
        )
        _, status_zero_i32, status_false = _lane_axis_carry_zeroes(y_next)
        status_after = status_after + status_zero_i32
        stop_after = stop_after | status_false

        # Update running phi_last only on accepted steps (matches C++).
        phi_last_next = jnp.where(accepted, phi_current, phi_last)
        phi_init_next = jnp.where(
            jnp.logical_and(accepted, first_accepted_step),
            phi_current,
            phi_init,
        )
        # boost's failed-step / no-progress checks; see
        # ``_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS``.
        no_progress_next, status_after, stop_after = _step_control_progress(
            no_progress=no_progress,
            accepted=accepted,
            nonfinite_state=step.nonfinite_state,
            t=t,
            t_next=t_next,
            y=y,
            k_first=k_first,
            status=status_after,
            stop=stop_after,
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
            step_count + _device_index(1),
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
            no_progress_next,
        )

    (
        step_count_final,
        accepted_count,
        t_final,
        y_final,
        h_final,
        k_final,
        traj_final,
        mask_final,
        phi_hits_final,
        phi_hits_count_final,
        phi_last_final,
        phi_init_final,
        status_event_final,
        stop_at_exit,
        no_progress_final,
    ) = _run_adaptive_steps(
        cond,
        body,
        init_carry,
        max_steps,
        spec.adaptive_loop,
    )

    # Pad unused rows with the final accepted state so downstream code
    # that ignores the mask still sees a valid (constant-extension)
    # trajectory.
    last_row = jnp.concatenate([jnp.reshape(t_final, (1,)), y_final.reshape((3,))])

    traj_padded = jnp.where(
        mask_final[:, None],
        traj_final,
        jnp.broadcast_to(last_row, traj_final.shape),
    )

    eps_t = _device_array(1.0e-12, dtype) * jnp.maximum(
        jnp.abs(tmax), _device_array(1.0, dtype)
    )
    reached = (tmax - t_final) <= eps_t
    # Status priority: a stopping criterion (status_event_final < 0)
    # wins over budget / reached state.
    status_normal = jnp.where(
        reached,
        _device_index(0),
        _device_index(1),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    result = FieldlineTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_final,
        phi_hits_count=phi_hits_count_final,
    )
    next_state = CartesianTracingContinuationState(
        trial_count=global_trials + step_count_final,
        accepted_count=global_accepted + accepted_count,
        t=t_final,
        y=y_final,
        h=h_final,
        k_first=k_final,
        phi_last=phi_last_final,
        phi_initial=phi_init_final,
        status_event=status_event_final,
        stopped=stop_at_exit,
        no_progress=no_progress_final,
    )
    return result, next_state


def trace_fieldline(
    spec: FieldlineTracingSpec,
    y0: TracingStateInput,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> FieldlineTracingResult:
    """Trace one fieldline and return the original fixed-shape result contract."""
    result, _state = _trace_fieldline_chunk(
        spec, y0, magnetic_field_fn, phis, stopping_criteria
    )
    return result


def _make_fieldline_trace_one(spec, magnetic_field_fn, phis, stopping_criteria):
    """Build the per-lane fieldline integrator shared by the batched paths."""

    def trace_one(
        y0: jax.Array, dtmax: jax.Array, field_state: object | None = None
    ) -> FieldlineTracingResult:
        if field_state is None:
            field_fn = magnetic_field_fn
        else:

            def field_fn(point):
                return magnetic_field_fn(field_state, point)

        return trace_fieldline(
            replace(spec, dtmax=dtmax),
            y0,
            field_fn,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    return trace_one


@partial(jax.jit, static_argnames=("magnetic_field_fn",))
def _trace_fieldlines_batched_unsharded(
    spec: FieldlineTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
    magnetic_field_state: object | None = None,
) -> FieldlineTracingResult:
    trace_one = _make_fieldline_trace_one(
        spec, magnetic_field_fn, phis, stopping_criteria
    )

    if magnetic_field_state is None:
        return jax.vmap(trace_one)(y0s, dtmaxs)
    return jax.vmap(trace_one, in_axes=(0, 0, None))(
        y0s,
        dtmaxs,
        magnetic_field_state,
    )


def trace_fieldlines_batched(
    spec: FieldlineTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
    magnetic_field_state: object | None = None,
) -> FieldlineTracingResult:
    """Trace a batch of fieldlines with one vmapped JAX integration graph.

    ``max_steps``, event-buffer sizes, tolerances, and stopping criteria
    are shared across the batch. ``dtmaxs`` is per lane because the
    upstream quarter-turn step cap depends on the initial radius and
    field strength.
    """

    spec = _stage_fieldline_spec(spec)
    y0s_arr = _as_device_array(y0s, jnp.float64).reshape((-1, 3))
    dtmaxs_arr = _as_device_array(dtmaxs, jnp.float64).reshape((-1,))
    stopping_criteria = tuple(stopping_criteria)

    trace_one = _make_fieldline_trace_one(
        spec, magnetic_field_fn, phis, stopping_criteria
    )

    config = trajectory_batch_sharding_config(y0s_arr)
    if config is not None:
        y0s_arr, dtmaxs_arr = maybe_shard_trajectory_batch_inputs(
            y0s_arr,
            dtmaxs_arr,
            config=config,
        )

        out_specs = FieldlineTracingResult(
            trajectory=P(config.axis_name, None, None),
            mask=P(config.axis_name, None),
            steps_taken=P(config.axis_name),
            status=P(config.axis_name),
            t_final=P(config.axis_name),
            phi_hits=P(config.axis_name, None, None),
            phi_hits_count=P(config.axis_name),
        )
        if magnetic_field_state is None:

            @partial(
                jax.shard_map,
                mesh=config.mesh,
                in_specs=(P(config.axis_name, None), P(config.axis_name)),
                out_specs=out_specs,
                check_vma=True,
            )
            def trace_shard(y0s_block, dtmaxs_block):
                return jax.lax.map(
                    lambda inputs: trace_one(*inputs),
                    (y0s_block, dtmaxs_block),
                )

            return trace_shard(y0s_arr, dtmaxs_arr)

        field_state_specs = jax.tree.map(lambda _leaf: P(), magnetic_field_state)

        @partial(
            jax.shard_map,
            mesh=config.mesh,
            in_specs=(
                P(config.axis_name, None),
                P(config.axis_name),
                field_state_specs,
            ),
            out_specs=out_specs,
            check_vma=True,
        )
        def trace_shard(y0s_block, dtmaxs_block, field_state_block):
            return jax.lax.map(
                lambda inputs: trace_one(inputs[0], inputs[1], field_state_block),
                (y0s_block, dtmaxs_block),
            )

        return trace_shard(y0s_arr, dtmaxs_arr, magnetic_field_state)

    return _trace_fieldlines_batched_unsharded(
        spec,
        y0s_arr,
        dtmaxs_arr,
        magnetic_field_fn,
        phis,
        stopping_criteria,
        magnetic_field_state,
    )


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
    max_root_iters: int = 200
    max_phi_hits: int = 128
    adaptive_loop: AdaptiveLoop = "scan"


jax.tree_util.register_dataclass(
    GuidingCenterTracingSpec,
    data_fields=["tmax", "rtol", "atol", "dtmax"],
    meta_fields=["max_steps", "max_root_iters", "max_phi_hits", "adaptive_loop"],
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
    ``legacy native extension/tracing.cpp``) are

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

    m_arr = _as_device_array(m, jnp.float64)
    q_arr = _as_device_array(q, jnp.float64)
    mu_arr = _as_device_array(mu, jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t  # Field is autonomous; signature kept for ODE-driver shape.
        position, v_par_tail = jnp.split(y, [3])
        B_raw, dB_by_dX_raw = magnetic_field_fn(position)
        B = jnp.asarray(B_raw, dtype=y.dtype).reshape((3,))
        dB_by_dX = jnp.asarray(dB_by_dX_raw, dtype=y.dtype).reshape((3, 3))
        # GradAbsB_j = B_l * dB_l/dx_j / |B|   (upstream GradAbsB_ref).
        grad_abs_B = jnp.einsum("l,jl->j", B, dB_by_dX) / jnp.linalg.norm(B)
        return _guiding_center_vacuum_dydt(
            y, jnp.reshape(v_par_tail, ()), B, grad_abs_B, m_arr, q_arr, mu_arr
        )

    return rhs


def _guiding_center_vacuum_dydt(
    y: jax.Array,
    v_par: jax.Array,
    B: jax.Array,
    grad_abs_B: jax.Array,
    m_arr: jax.Array,
    q_arr: jax.Array,
    mu_arr: jax.Array,
) -> jax.Array:
    """``GuidingCenterVacuumRHS::operator()`` once ``B`` and ``grad|B|`` are known.

    The one copy of the drift-kinetic arithmetic; the two right-hand-side
    builders above and below differ only in where ``grad|B|`` comes from, and
    the operation order here is upstream's (``simsoptpp/tracing.cpp``:
    ``BcrossGradAbsB``, ``v_perp2``, ``fak1``, ``fak2``), because that order is
    what decides the ``inf``/``nan`` pattern where ``|B|`` vanishes.
    """

    abs_B = jnp.linalg.norm(B)
    # B x grad|B|
    B_cross_grad_abs_B = jnp.cross(B, grad_abs_B)
    v_perp2 = _device_array(2.0, y.dtype) * mu_arr * abs_B
    fak1 = v_par / abs_B
    fak2 = (
        m_arr
        / (q_arr * abs_B**3)
        * (_device_array(0.5, y.dtype) * v_perp2 + v_par * v_par)
    )
    dposition = fak1 * B + fak2 * B_cross_grad_abs_B
    dv_par = -mu_arr * jnp.dot(B, grad_abs_B) / abs_B
    return jnp.concatenate([dposition, jnp.reshape(dv_par, (1,))])


def guiding_center_vacuum_rhs_cached(
    magnetic_field_fn: Callable[
        [jax.Array, InterpolatedFieldCylCache],
        tuple[jax.Array, jax.Array, InterpolatedFieldCylCache],
    ],
    m: float,
    q: float,
    mu: float,
) -> Callable[
    [jax.Array, jax.Array, InterpolatedFieldCylCache],
    tuple[jax.Array, InterpolatedFieldCylCache],
]:
    r"""``rhs(t, y, cache) -> (dy/dt, cache)`` for a field with an output buffer.

    Same equations and the same arithmetic as
    :func:`guiding_center_vacuum_rhs` (both call
    :func:`_guiding_center_vacuum_dydt`), with two differences, both of which
    bring the port CLOSER to ``GuidingCenterVacuumRHS::operator()``
    (``legacy native extension/tracing.cpp``):

    1. ``magnetic_field_fn`` returns ``grad|B|`` itself, as
       ``field->GradAbsB_ref()`` does, so ``|B|`` divides the gradient once
       here instead of twice through a ``dB_by_dX`` round trip.
    2. The field's output buffer is threaded through the call, so a query
       outside the interpolation domain returns what upstream returns there --
       the previous query's value -- instead of zero. See
       :class:`~simsopt_jax.core.interpolated_field.InterpolatedFieldCylCache`
       for the C++ contract and
       :func:`~simsopt_jax.core.interpolated_field.interpolated_field_state_B_GradAbsB_cached`
       for the evaluation.
    """

    m_arr = _as_device_array(m, jnp.float64)
    q_arr = _as_device_array(q, jnp.float64)
    mu_arr = _as_device_array(mu, jnp.float64)

    def rhs(
        _t: jax.Array, y: jax.Array, cache: InterpolatedFieldCylCache
    ) -> tuple[jax.Array, InterpolatedFieldCylCache]:
        del _t  # Field is autonomous; signature kept for ODE-driver shape.
        position, v_par_tail = jnp.split(y, [3])
        B_raw, grad_abs_B_raw, cache_next = magnetic_field_fn(position, cache)
        B = jnp.asarray(B_raw, dtype=y.dtype).reshape((3,))
        grad_abs_B = jnp.asarray(grad_abs_B_raw, dtype=y.dtype).reshape((3,))
        dydt = _guiding_center_vacuum_dydt(
            y, jnp.reshape(v_par_tail, ()), B, grad_abs_B, m_arr, q_arr, mu_arr
        )
        return dydt, cache_next

    return rhs


def _trace_guiding_center_chunk(
    spec: GuidingCenterTracingSpec,
    y0: TracingStateInput,
    magnetic_field_fn: Callable[..., tuple[jax.Array, ...]],
    m: float,
    q: float,
    mu: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
    continuation: CartesianTracingContinuationState | None = None,
    field_cache_init: InterpolatedFieldCylCache | None = None,
) -> tuple[GuidingCenterTracingResult, CartesianTracingContinuationState]:
    """Trace a guiding-centre orbit from ``y0`` for ``spec.tmax`` seconds.

    Parameters
    ----------
    spec
        Tracing contract; see :class:`GuidingCenterTracingSpec`.
    y0
        Initial state ``[x, y, z, v_par]`` (length 4) as an array or flat
        list/tuple of array-like scalar components. Treated as float64.
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

    Returns a chunk-local result and immutable state for an exact next chunk.
    ``spec.max_steps`` bounds this call's trials; it is not the total horizon.
    """

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

    # ``field_cache_init`` decides the field contract this chunk mirrors; see
    # :func:`_dopri5_stage_step`. ``None`` is a field that writes every output
    # row (analytic fields); an :class:`InterpolatedFieldCylCache` is a field
    # that leaves its output buffer untouched outside its domain.
    if field_cache_init is None:
        rhs = guiding_center_vacuum_rhs(magnetic_field_fn, m, q, mu)
    else:
        rhs = guiding_center_vacuum_rhs_cached(magnetic_field_fn, m, q, mu)
    if continuation is None:
        h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
        if field_cache_init is None:
            k0 = rhs(t0, y0_arr)
            field_cache_start = None
        else:
            # boost's ``dense_output_runge_kutta`` evaluates the leading FSAL
            # derivative once at the initial state before the first trial, and
            # that evaluation writes the field's output buffer.
            k0, field_cache_start = rhs(t0, y0_arr, field_cache_init)
        start_t = t0
        start_y = y0_arr
        global_trials = _device_index(0)
        global_accepted = _device_index(0)
        status_initial = _device_index(0)
        stopped_initial = _device_false()
        no_progress_initial = _device_index(0)
    else:
        h0 = continuation.h
        k0 = continuation.k_first
        field_cache_start = continuation.field_cache
        start_t = continuation.t
        start_y = continuation.y
        global_trials = continuation.trial_count
        global_accepted = continuation.accepted_count
        status_initial = continuation.status_event
        stopped_initial = continuation.stopped
        no_progress_initial = continuation.no_progress
    one = _device_array(1.0, dtype)
    lane_zero, lane_zero_i32, lane_false = _lane_axis_carry_zeroes(y0_arr)
    accepted_count_init = _device_index(0) + lane_zero_i32

    # Pre-allocate the trajectory carry with columns (t, x, y, z, v_par).
    # Row 0 holds the initial state; rows 1..max_steps fill in as
    # accepted steps occur. Padding rows at the end of the run get the
    # final accepted state.
    traj_init_row = jnp.concatenate((jnp.reshape(start_t, (1,)), start_y), axis=0)
    traj = jnp.concatenate(
        (
            jnp.reshape(traj_init_row, (1, 5)),
            _device_zeros((max_steps, 5), dtype),
        ),
        axis=0,
    )
    traj = traj + lane_zero
    mask_indices = jax.lax.iota(jnp.int32, max_steps + 1)
    mask = mask_indices == _device_index(0)
    mask = mask | lane_false

    phi_hits_buf = _device_zeros((max_phi_hits, 6), dtype)
    phi_hits_count_init = _device_index(0)
    phi_hits_buf, phi_hits_count_init = _event_carry_with_lane_axis(
        phi_hits_buf,
        phi_hits_count_init,
        y0_arr,
    )

    if phis is None:
        phis_arr = _device_zeros((0,), dtype)
    else:
        phis_arr = _as_device_array(phis, dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    y0_x, y0_y, _y0_z = _split_xyz(y0_arr)
    if continuation is None:
        phi_init = _continuous_phi(y0_x, y0_y, _device_array(np.pi, dtype), dtype)
        phi_last = phi_init
    else:
        phi_init = continuation.phi_initial
        phi_last = continuation.phi_last

    init_carry = (
        _device_index(0),  # step_count
        accepted_count_init,
        start_t + lane_zero,
        start_y,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        phi_last,
        phi_init,
        status_initial + lane_zero_i32,  # status_event
        stopped_initial | lane_false,
        no_progress_initial + lane_zero_i32,  # consecutive non-advancing trials
        field_cache_start,  # the field's output buffer; None for analytic fields
    )

    max_steps_i32 = _device_index(max_steps)
    max_phi_hits_i32 = _device_index(max_phi_hits)
    two_pi = _device_two_pi(one)

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
            _no_progress,
            _field_cache,
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
            no_progress,
            field_cache,
        ) = carry
        step = _dopri5_adaptive_step(
            rhs,
            t,
            y,
            h,
            k_first,
            tmax,
            dtmax,
            rtol,
            atol,
            dtype,
            field_cache,
        )
        h_clamped = step.h_clamped
        y_new = step.y_new
        accepted = step.accepted
        h_next = step.h_next
        t_next = step.t_next
        y_next = step.y_next
        k_next = step.k_next

        # ── Phi-plane crossing detection on accepted steps ──
        y_new_x, y_new_y, _y_new_z = _split_xyz(y_new)
        phi_current = _continuous_phi(y_new_x, y_new_y, phi_last, dtype)

        state_at_time = _dense_output_query(step.stages, y, t, h_clamped)

        (
            phi_hits_after,
            phi_count_after,
            status_scan,
            stop_scan,
        ) = _scan_angle_plane_events(
            hits=phi_hits_in,
            count=phi_hits_count_in,
            status=status_event,
            stop=lane_false,
            angle_last=phi_last,
            angle_current=phi_current,
            targets=phis_arr,
            num_targets=num_phis,
            two_pi=two_pi,
            dtype=dtype,
            t=t,
            h_clamped=h_clamped,
            max_root_iters=max_root_iters,
            enabled=accepted,
            max_hits_i32=max_phi_hits_i32,
            state_at_time=state_at_time,
            angle_at_state=lambda state, angle_near: _continuous_phi_from_state(
                state, angle_near, dtype
            ),
        )

        first_accepted_step = (global_accepted + accepted_count) == _device_index(0)
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
            return _apply_stopping_criteria_events(
                stopping_criteria=stopping_criteria,
                hits=hits_in,
                count=count_in,
                status=status_in,
                stop=stop_in,
                iter_count=iter_count_in,
                angle_current=phi_curr_in,
                angle_initial=phi_init_in,
                t_event=t_next,
                state=y_next,
                dtype=dtype,
                max_hits_i32=max_phi_hits_i32,
            )

        iter_count_post = global_trials + step_count + _device_index(1)

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
                status_scan,
                stop_scan,
                iter_count_post,
                phi_current,
                phi_init_for_criteria,
            ),
        )
        _, status_zero_i32, status_false = _lane_axis_carry_zeroes(y_next)
        status_after = status_after + status_zero_i32
        stop_after = stop_after | status_false

        phi_last_next = jnp.where(accepted, phi_current, phi_last)
        phi_init_next = jnp.where(
            jnp.logical_and(accepted, first_accepted_step),
            phi_current,
            phi_init,
        )
        # boost's failed-step / no-progress checks; see
        # ``_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS``.
        no_progress_next, status_after, stop_after = _step_control_progress(
            no_progress=no_progress,
            accepted=accepted,
            nonfinite_state=step.nonfinite_state,
            t=t,
            t_next=t_next,
            y=y,
            k_first=k_first,
            status=status_after,
            stop=stop_after,
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
            step_count + _device_index(1),
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
            no_progress_next,
            # The C++ buffer is written by every right-hand-side evaluation,
            # so it advances on a rejected trial too.
            step.stages.field_cache,
        )

    (
        step_count_final,
        accepted_count,
        t_final,
        y_final,
        h_final,
        k_final,
        traj_final,
        mask_final,
        phi_hits_final,
        phi_hits_count_final,
        phi_last_final,
        phi_init_final,
        status_event_final,
        stop_at_exit,
        no_progress_final,
        field_cache_final,
    ) = _run_adaptive_steps(
        cond,
        body,
        init_carry,
        max_steps,
        spec.adaptive_loop,
    )

    last_row = jnp.concatenate([jnp.reshape(t_final, (1,)), y_final.reshape((4,))])
    traj_padded = jnp.where(
        mask_final[:, None],
        traj_final,
        jnp.broadcast_to(last_row, traj_final.shape),
    )

    eps_t = _device_array(1.0e-12, dtype) * jnp.maximum(
        jnp.abs(tmax), _device_array(1.0, dtype)
    )
    reached = (tmax - t_final) <= eps_t
    status_normal = jnp.where(
        reached,
        _device_index(0),
        _device_index(1),
    )
    status = jnp.where(stop_at_exit, status_event_final, status_normal)

    result = GuidingCenterTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_final,
        phi_hits_count=phi_hits_count_final,
    )
    next_state = CartesianTracingContinuationState(
        trial_count=global_trials + step_count_final,
        accepted_count=global_accepted + accepted_count,
        t=t_final,
        y=y_final,
        h=h_final,
        k_first=k_final,
        phi_last=phi_last_final,
        phi_initial=phi_init_final,
        status_event=status_event_final,
        stopped=stop_at_exit,
        no_progress=no_progress_final,
        field_cache=field_cache_final,
    )
    return result, next_state


def trace_guiding_center(
    spec: GuidingCenterTracingSpec,
    y0: TracingStateInput,
    magnetic_field_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    m: float,
    q: float,
    mu: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
) -> GuidingCenterTracingResult:
    """Trace one guiding center and return the original fixed-shape result."""
    result, _state = _trace_guiding_center_chunk(
        spec, y0, magnetic_field_fn, m, q, mu, phis, stopping_criteria
    )
    return result


def _make_guiding_center_trace_one(
    spec, magnetic_field_fn, m, q, phis, stopping_criteria
):
    """Build the per-lane guiding-centre integrator shared by the batched paths."""

    def trace_one(
        y0: jax.Array,
        dtmax: jax.Array,
        mu: jax.Array,
        field_state: object | None = None,
    ) -> GuidingCenterTracingResult:
        if field_state is None:
            field_fn = magnetic_field_fn
        else:

            def field_fn(point):
                return magnetic_field_fn(field_state, point)

        return trace_guiding_center(
            replace(spec, dtmax=dtmax),
            y0,
            field_fn,
            m=m,
            q=q,
            mu=mu,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    return trace_one


@partial(jax.jit, static_argnames=("magnetic_field_fn",))
def _trace_guiding_centers_batched_unsharded(
    spec: GuidingCenterTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    mus: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    m: float,
    q: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
    magnetic_field_state: object | None = None,
) -> GuidingCenterTracingResult:
    trace_one = _make_guiding_center_trace_one(
        spec, magnetic_field_fn, m, q, phis, stopping_criteria
    )

    if magnetic_field_state is None:
        return jax.vmap(trace_one)(y0s, dtmaxs, mus)
    return jax.vmap(trace_one, in_axes=(0, 0, 0, None))(
        y0s,
        dtmaxs,
        mus,
        magnetic_field_state,
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
    magnetic_field_state: object | None = None,
) -> GuidingCenterTracingResult:
    """Trace Cartesian guiding-centre orbits with one vmapped JAX graph."""

    spec = _stage_guiding_center_spec(spec)
    y0s_arr = _as_device_array(y0s, jnp.float64).reshape((-1, 4))
    dtmaxs_arr = _as_device_array(dtmaxs, jnp.float64).reshape((-1,))
    mus_arr = _as_device_array(mus, jnp.float64).reshape((-1,))
    stopping_criteria = tuple(stopping_criteria)

    trace_one = _make_guiding_center_trace_one(
        spec, magnetic_field_fn, m, q, phis, stopping_criteria
    )

    config = trajectory_batch_sharding_config(y0s_arr)
    if config is not None:
        y0s_arr, dtmaxs_arr, mus_arr = maybe_shard_trajectory_batch_inputs(
            y0s_arr,
            dtmaxs_arr,
            mus_arr,
            config=config,
        )

        out_specs = GuidingCenterTracingResult(
            trajectory=P(config.axis_name, None, None),
            mask=P(config.axis_name, None),
            steps_taken=P(config.axis_name),
            status=P(config.axis_name),
            t_final=P(config.axis_name),
            phi_hits=P(config.axis_name, None, None),
            phi_hits_count=P(config.axis_name),
        )
        if magnetic_field_state is None:

            @partial(
                jax.shard_map,
                mesh=config.mesh,
                in_specs=(
                    P(config.axis_name, None),
                    P(config.axis_name),
                    P(config.axis_name),
                ),
                out_specs=out_specs,
                check_vma=True,
            )
            def trace_shard(y0s_block, dtmaxs_block, mus_block):
                return jax.lax.map(
                    lambda inputs: trace_one(*inputs),
                    (y0s_block, dtmaxs_block, mus_block),
                )

            return trace_shard(y0s_arr, dtmaxs_arr, mus_arr)

        field_state_specs = jax.tree.map(lambda _leaf: P(), magnetic_field_state)

        @partial(
            jax.shard_map,
            mesh=config.mesh,
            in_specs=(
                P(config.axis_name, None),
                P(config.axis_name),
                P(config.axis_name),
                field_state_specs,
            ),
            out_specs=out_specs,
            check_vma=True,
        )
        def trace_shard(y0s_block, dtmaxs_block, mus_block, field_state_block):
            return jax.lax.map(
                lambda inputs: trace_one(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    field_state_block,
                ),
                (y0s_block, dtmaxs_block, mus_block),
            )

        return trace_shard(y0s_arr, dtmaxs_arr, mus_arr, magnetic_field_state)

    return _trace_guiding_centers_batched_unsharded(
        spec,
        y0s_arr,
        dtmaxs_arr,
        mus_arr,
        magnetic_field_fn,
        m,
        q,
        phis,
        stopping_criteria,
        magnetic_field_state,
    )


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
    adaptive_loop: AdaptiveLoop = "scan",
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
    lane_zero, lane_zero_i32, lane_false = _lane_axis_carry_zeroes(y0)
    k0 = jax.lax.cond(
        initial_axis_invalid,
        lambda _: jnp.zeros_like(y0),
        lambda _: rhs(t0, y0),
        operand=None,
    )
    accepted_count_init = jnp.asarray(0, dtype=jnp.int32) + lane_zero_i32
    t0_init = t0 + lane_zero

    traj = jnp.zeros((max_steps + 1, 5), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0)
    traj = traj + lane_zero
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)
    mask = mask | lane_false

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        accepted_count_init,
        t0_init,
        y0,
        h0,
        k0,
        traj,
        mask,
        jnp.where(
            initial_axis_invalid,
            jnp.asarray(TRACING_STATUS_BOOZER_AXIS, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        + lane_zero_i32,
        initial_axis_invalid | lane_false,
        _device_index(0) + lane_zero_i32,  # consecutive non-advancing trials
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
            _no_progress,
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
            no_progress,
        ) = carry
        step = _dopri5_adaptive_step(
            rhs,
            t,
            y,
            h,
            k_first,
            tmax,
            dtmax,
            rtol,
            atol,
            dtype,
        )
        y_new = step.y_new
        accepted = step.accepted
        axis_invalid = jnp.logical_and(accepted, _boozer_axis_invalid(y_new))
        h_next = step.h_next
        t_next = step.t_next
        y_next = step.y_next
        k_next = step.k_next
        status_axis = jnp.where(
            axis_invalid,
            jnp.asarray(TRACING_STATUS_BOOZER_AXIS, dtype=jnp.int32),
            status_event,
        )
        # boost's failed-step / no-progress checks; see
        # ``_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS``.
        no_progress_next, status_next, stop_next = _step_control_progress(
            no_progress=no_progress,
            accepted=accepted,
            nonfinite_state=step.nonfinite_state,
            t=t,
            t_next=t_next,
            y=y,
            k_first=k_first,
            status=status_axis,
            stop=axis_invalid,
        )
        traj_next, mask_next, accepted_next = _record_trajectory_row(
            traj,
            mask,
            accepted_count,
            t_next,
            y_next,
            _should_record_accepted_step(accepted, stop_next),
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
            stop_next,
            no_progress_next,
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
        no_progress_final,
    ) = _run_adaptive_steps(
        cond,
        body,
        init_carry,
        max_steps,
        adaptive_loop,
    )

    last_row = jnp.concatenate(
        [jnp.asarray([t_final], dtype=dtype), y_final.reshape((4,))]
    )

    traj_padded = jnp.where(
        mask_final[:, None],
        traj_final,
        jnp.broadcast_to(last_row, traj_final.shape),
    )

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
    phi_hits_empty, phi_hits_count = _event_carry_with_lane_axis(
        phi_hits_empty,
        jnp.asarray(0, dtype=jnp.int32),
        y_final,
    )
    return GuidingCenterTracingResult(
        trajectory=traj_padded,
        mask=mask_final,
        steps_taken=accepted_count,
        status=status,
        t_final=t_final,
        phi_hits=phi_hits_empty,
        phi_hits_count=phi_hits_count,
    )


# ── Boozer-coordinate guiding-centre RHS (4-state) ────────────────────


def _resolve_boozer_field_state(boozer_field):
    """Resolve a Boozer field-like object to its frozen pytree state + psi0.

    The Boozer RHS variants are pure functions of a single Boozer point
    ``(s, theta, zeta)`` and the immutable frozen-state pytree exposed
    by :class:`simsopt_jax_adapters.field.boozer_field.BoozerRadialInterpolantJAX`.
    Routing through the mutable ``set_points`` API would break the
    JIT/device-loop contract (Python side-effects on cached arrays are
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
        return boozer_field[0], boozer_field[1]
    frozen = getattr(boozer_field, "frozen_state", None)
    psi0 = getattr(boozer_field, "psi0", None)
    if frozen is None or psi0 is None:
        raise TypeError(
            "guiding-centre Boozer RHS requires a "
            "BoozerRadialInterpolantJAX-shaped field exposing "
            "`frozen_state` and `psi0`, or a (frozen_state, psi0) "
            f"tuple; got {type(boozer_field).__name__}."
        )
    return frozen, psi0


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


def _radial_column_scalar(name: str) -> Callable:
    def _eval(
        _state: BoozerRadialInterpolantFrozenState,
        columns,
        _point: jax.Array,
    ) -> jax.Array:
        return getattr(columns, name)

    return _eval


_RADIAL_RHS_COLUMN_EVALUATORS: dict[str, Callable] = {
    "modB": _radial_modB_from_columns,
    "dmodBds": _radial_dmodBds_from_columns,
    "dmodBdtheta": _radial_dmodBdtheta_from_columns,
    "dmodBdzeta": _radial_dmodBdzeta_from_columns,
    "K": _radial_K_from_columns,
    "dKdtheta": _radial_dKdtheta_from_columns,
    "dKdzeta": _radial_dKdzeta_from_columns,
    "G": _radial_column_scalar("G"),
    "I": _radial_column_scalar("I"),
    "iota": _radial_column_scalar("iota"),
    "dGds": _radial_column_scalar("dGds"),
    "dIds": _radial_column_scalar("dIds"),
}


def _boozer_rhs_evaluators(state) -> tuple[dict[str, Callable], bool]:
    if isinstance(state, BoozerRadialInterpolantFrozenState):
        return _RADIAL_RHS_COLUMN_EVALUATORS, True
    return _boozer_field_evaluators(state), False


def _boozer_radial_columns_for_point(
    state,
    point: jax.Array,
    use_radial_columns: bool,
):
    if use_radial_columns:
        return _radial_eval_rhs_columns(state, point[:, 0])
    return None


def _boozer_eval_scalar(
    evals: dict[str, Callable],
    key: str,
    state,
    point: jax.Array,
    radial_columns,
) -> jax.Array:
    if radial_columns is None:
        return _boozer_scalar(evals[key](state, point))
    return _boozer_scalar(evals[key](state, radial_columns, point))


def _boozer_field_evaluators(state) -> dict[str, Callable]:
    """Return the set of evaluator callables matching the frozen-state type.

    This is a Python-static dispatch evaluated once at RHS-factory time,
    outside the JIT trace. The returned callables are bound to a
    particular state shape; the inner ``rhs(t, y)`` function captures
    them by closure and JAX traces a single homogeneous evaluator graph
    per call.

    Supported state types:

    - :class:`simsopt_jax.core.boozer_analytic.BoozerAnalyticFrozenState`
      — closed-form analytic evaluators from
      :mod:`simsopt_jax.core.boozer_analytic`.
    - :class:`simsopt_jax_adapters.field.boozer_field.BoozerRadialInterpolantFrozenState`
      — spline + Fourier evaluators from
      :mod:`simsopt_jax_adapters.field.boozer_field`.
    - :class:`simsopt_jax_adapters.field.boozer_field.InterpolatedBoozerFieldFrozenState`
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
    ``legacy native extension/tracing.cpp``:

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
        Either a :class:`simsopt_jax_adapters.field.boozer_field.BoozerRadialInterpolantJAX`
        instance, a :class:`simsopt_jax_adapters.field.boozer_field.BoozerAnalyticJAX`
        instance, or a ``(frozen_state, psi0)`` tuple. The RHS reads
        ``modB``, ``modB_derivs``, ``iota``, ``G`` and ``psi0`` from
        the frozen state; the evaluator family is selected at
        factory-call time by :func:`_boozer_field_evaluators`.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    """

    state, psi0_host = _resolve_boozer_field_state(boozer_field)
    evals, use_radial_columns = _boozer_rhs_evaluators(state)
    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    psi0 = jnp.asarray(psi0_host, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t
        v_par = y[3]
        point = _boozer_point_2d(y)
        radial_columns = _boozer_radial_columns_for_point(
            state, point, use_radial_columns
        )
        modB = _boozer_eval_scalar(evals, "modB", state, point, radial_columns)
        dmodBds = _boozer_eval_scalar(evals, "dmodBds", state, point, radial_columns)
        dmodBdtheta = _boozer_eval_scalar(
            evals, "dmodBdtheta", state, point, radial_columns
        )
        dmodBdzeta = _boozer_eval_scalar(
            evals, "dmodBdzeta", state, point, radial_columns
        )
        G = _boozer_eval_scalar(evals, "G", state, point, radial_columns)
        iota = _boozer_eval_scalar(evals, "iota", state, point, radial_columns)

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
    ``legacy native extension/tracing.cpp``. The non-vacuum case uses ``G(s)`` and
    ``I(s)`` profiles but assumes ``K(s, theta, zeta) = 0``. The upstream
    equation contains the physical banana-tip singularity ``mu / v_par``;
    this faithful port does not regularize AD through ``v_par = 0``.

    Parameters
    ----------
    boozer_field
        Either a :class:`simsopt_jax_adapters.field.boozer_field.BoozerRadialInterpolantJAX`
        instance, a :class:`simsopt_jax_adapters.field.boozer_field.BoozerAnalyticJAX`
        instance, or a ``(frozen_state, psi0)`` tuple. The RHS reads
        ``modB``, ``modB_derivs``, ``iota``, ``G``, ``I``, ``dGds``,
        ``dIds`` and ``psi0`` from the frozen state; the evaluator
        family is selected at factory-call time by
        :func:`_boozer_field_evaluators`.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    """

    state, psi0_host = _resolve_boozer_field_state(boozer_field)
    evals, use_radial_columns = _boozer_rhs_evaluators(state)
    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    psi0 = jnp.asarray(psi0_host, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t
        v_par = y[3]
        point = _boozer_point_2d(y)
        radial_columns = _boozer_radial_columns_for_point(
            state, point, use_radial_columns
        )
        modB = _boozer_eval_scalar(evals, "modB", state, point, radial_columns)
        dmodBds = _boozer_eval_scalar(evals, "dmodBds", state, point, radial_columns)
        dmodBdtheta = _boozer_eval_scalar(
            evals, "dmodBdtheta", state, point, radial_columns
        )
        dmodBdzeta = _boozer_eval_scalar(
            evals, "dmodBdzeta", state, point, radial_columns
        )
        G = _boozer_eval_scalar(evals, "G", state, point, radial_columns)
        I_val = _boozer_eval_scalar(evals, "I", state, point, radial_columns)
        iota = _boozer_eval_scalar(evals, "iota", state, point, radial_columns)
        dGds = _boozer_eval_scalar(evals, "dGds", state, point, radial_columns)
        dIds = _boozer_eval_scalar(evals, "dIds", state, point, radial_columns)
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
    ``legacy native extension/tracing.cpp`` — the non-vacuum, ``K != 0`` case with
    ``C``, ``F``, ``D`` algebraic coefficients folded in. The upstream
    equation contains the physical banana-tip singularity ``mu / v_par``;
    this faithful port does not regularize AD through ``v_par = 0``.

    Parameters
    ----------
    boozer_field
        Either a :class:`simsopt_jax_adapters.field.boozer_field.BoozerRadialInterpolantJAX`
        instance, a :class:`simsopt_jax_adapters.field.boozer_field.BoozerAnalyticJAX`
        instance, or a ``(frozen_state, psi0)`` tuple. The RHS reads
        ``modB``, ``modB_derivs``, ``K``, ``K_derivs``, ``iota``,
        ``G``, ``I``, ``dGds``, ``dIds`` and ``psi0`` from the frozen
        state; the evaluator family is selected at factory-call time
        by :func:`_boozer_field_evaluators`.
    m, q, mu
        Particle mass, charge, and magnetic moment (Python floats).
    """

    state, psi0_host = _resolve_boozer_field_state(boozer_field)
    evals, use_radial_columns = _boozer_rhs_evaluators(state)
    m_arr = jnp.asarray(m, dtype=jnp.float64)
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    psi0 = jnp.asarray(psi0_host, dtype=jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t
        v_par = y[3]
        point = _boozer_point_2d(y)
        radial_columns = _boozer_radial_columns_for_point(
            state, point, use_radial_columns
        )
        modB = _boozer_eval_scalar(evals, "modB", state, point, radial_columns)
        dmodBds = _boozer_eval_scalar(evals, "dmodBds", state, point, radial_columns)
        dmodBdtheta = _boozer_eval_scalar(
            evals, "dmodBdtheta", state, point, radial_columns
        )
        dmodBdzeta = _boozer_eval_scalar(
            evals, "dmodBdzeta", state, point, radial_columns
        )
        K_val = _boozer_eval_scalar(evals, "K", state, point, radial_columns)
        dKdtheta = _boozer_eval_scalar(evals, "dKdtheta", state, point, radial_columns)
        dKdzeta = _boozer_eval_scalar(evals, "dKdzeta", state, point, radial_columns)
        G = _boozer_eval_scalar(evals, "G", state, point, radial_columns)
        I_val = _boozer_eval_scalar(evals, "I", state, point, radial_columns)
        iota = _boozer_eval_scalar(evals, "iota", state, point, radial_columns)
        dGds = _boozer_eval_scalar(evals, "dGds", state, point, radial_columns)
        dIds = _boozer_eval_scalar(evals, "dIds", state, point, radial_columns)
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
    if zetas is None:
        zetas_arr = jnp.zeros((0,), dtype=dtype)
    else:
        zetas_arr = jnp.asarray(zetas, dtype=dtype).reshape((-1,))
    num_zetas = int(zetas_arr.shape[0])
    initial_axis_invalid = _boozer_axis_invalid(y0_arr)

    # Fast path: no events requested → reuse the lean shared driver to
    # preserve the prior compile profile on the no-events parity tests.
    if num_zetas == 0 and len(stopping_criteria) == 0:
        return _run_dopri5_4state(
            rhs,
            y0_arr,
            tmax,
            rtol,
            atol,
            dtmax,
            max_steps,
            max_phi_hits=max_phi_hits,
            adaptive_loop=spec.adaptive_loop,
        )

    h0 = _initial_step_size(t0, tmax, dtmax, _PARTICLE_INITIAL_STEP_FRACTION)
    k0 = jax.lax.cond(
        initial_axis_invalid,
        lambda _: jnp.zeros_like(y0_arr),
        lambda _: rhs(t0, y0_arr),
        operand=None,
    )
    one = jnp.asarray(1.0, dtype=dtype)
    lane_zero, lane_zero_i32, lane_false = _lane_axis_carry_zeroes(y0_arr)
    accepted_count_init = jnp.asarray(0, dtype=jnp.int32) + lane_zero_i32
    t0_init = t0 + lane_zero

    traj = jnp.zeros((max_steps + 1, 5), dtype=dtype)
    traj = traj.at[0, 0].set(t0)
    traj = traj.at[0, 1:].set(y0_arr)
    traj = traj + lane_zero
    mask = jnp.zeros((max_steps + 1,), dtype=jnp.bool_)
    mask = mask.at[0].set(True)
    mask = mask | lane_false

    # zeta-plane crossing buffer; columns are ``[t_hit, idx, s, theta,
    # zeta, v_par]`` (6 wide, matching the upstream Boozer
    # ``res_zeta_hits`` row layout).
    phi_hits_buf = jnp.zeros((max_phi_hits, 6), dtype=dtype)
    phi_hits_count_init = jnp.asarray(0, dtype=jnp.int32)
    phi_hits_buf, phi_hits_count_init = _event_carry_with_lane_axis(
        phi_hits_buf,
        phi_hits_count_init,
        y0_arr,
    )

    # Initial unwrapped zeta seed: the Boozer state stores zeta
    # directly, so we anchor near the literal initial value.
    zeta_init = _continuous_angle(y0_arr[2], jnp.asarray(np.pi, dtype=dtype), dtype)

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        accepted_count_init,
        t0_init,
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
            jnp.asarray(TRACING_STATUS_BOOZER_AXIS, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        + lane_zero_i32,  # status_event
        initial_axis_invalid | lane_false,  # stop flag
        _device_index(0) + lane_zero_i32,  # consecutive non-advancing trials
    )

    max_steps_i32 = jnp.asarray(max_steps, dtype=jnp.int32)
    max_phi_hits_i32 = jnp.asarray(max_phi_hits, dtype=jnp.int32)
    two_pi = _device_two_pi(one)

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
            _no_progress,
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
            no_progress,
        ) = carry
        step = _dopri5_adaptive_step(
            rhs,
            t,
            y,
            h,
            k_first,
            tmax,
            dtmax,
            rtol,
            atol,
            dtype,
        )
        y_new = step.y_new
        accepted = step.accepted
        axis_invalid = jnp.logical_and(accepted, _boozer_axis_invalid(y_new))
        accepted_valid = jnp.logical_and(accepted, jnp.logical_not(axis_invalid))
        h_clamped = step.h_clamped
        h_next = step.h_next
        t_next = step.t_next
        y_next = step.y_next
        k_next = step.k_next

        # ── Zeta-plane crossing detection on accepted steps ──
        # The Boozer state stores zeta directly (no atan2 needed);
        # ``_continuous_angle`` anchors the running unwrap branch
        # near ``zeta_last``.
        zeta_current = _continuous_angle(y_new[2], zeta_last, dtype)

        state_at_time = _dense_output_query(step.stages, y, t, h_clamped)

        (
            phi_hits_after,
            phi_count_after,
            status_scan,
            stop_scan,
        ) = _scan_angle_plane_events(
            hits=phi_hits_in,
            count=phi_hits_count_in,
            status=status_event,
            stop=jnp.asarray(False),
            angle_last=zeta_last,
            angle_current=zeta_current,
            targets=zetas_arr,
            num_targets=num_zetas,
            two_pi=two_pi,
            dtype=dtype,
            t=t,
            h_clamped=h_clamped,
            max_root_iters=max_root_iters,
            enabled=accepted_valid,
            max_hits_i32=max_phi_hits_i32,
            state_at_time=state_at_time,
            angle_at_state=lambda state, angle_near: _continuous_angle(
                state[2], angle_near, dtype
            ),
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
            return _apply_stopping_criteria_events(
                stopping_criteria=stopping_criteria,
                hits=hits_in,
                count=count_in,
                status=status_in,
                stop=stop_in,
                iter_count=iter_count_in,
                angle_current=zeta_curr_in,
                angle_initial=zeta_init_in,
                t_event=t_next,
                state=y_next,
                dtype=dtype,
                max_hits_i32=max_phi_hits_i32,
                is_boozer_state=True,
            )

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
                status_scan,
                stop_scan,
                iter_count_post,
                zeta_current,
                zeta_init_for_criteria,
            ),
        )
        status_after = jnp.where(
            axis_invalid,
            jnp.asarray(TRACING_STATUS_BOOZER_AXIS, dtype=jnp.int32),
            status_after,
        )
        stop_after = jnp.logical_or(stop_after, axis_invalid)
        _, status_zero_i32, status_false = _lane_axis_carry_zeroes(y_next)
        status_after = status_after + status_zero_i32
        stop_after = stop_after | status_false

        zeta_last_next = jnp.where(accepted, zeta_current, zeta_last)
        zeta_init_next = jnp.where(
            jnp.logical_and(accepted_valid, first_valid_accepted_step),
            zeta_current,
            zeta_init,
        )
        # boost's failed-step / no-progress checks; see
        # ``_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS``.
        no_progress_next, status_after, stop_after = _step_control_progress(
            no_progress=no_progress,
            accepted=accepted,
            nonfinite_state=step.nonfinite_state,
            t=t,
            t_next=t_next,
            y=y,
            k_first=k_first,
            status=status_after,
            stop=stop_after,
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
            no_progress_next,
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
        no_progress_final,
    ) = _run_adaptive_steps(
        cond,
        body,
        init_carry,
        max_steps,
        spec.adaptive_loop,
    )

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


def _trace_guiding_centers_boozer_batched_lax_map(
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
    def trace_one(inputs) -> GuidingCenterTracingResult:
        y0, dtmax, mu = inputs
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

    return jax.lax.map(trace_one, (y0s, dtmaxs, mus))


@partial(jax.jit, static_argnames=("mode",))
def _trace_guiding_centers_boozer_batched_unsharded(
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
    return _trace_guiding_centers_boozer_batched_lax_map(
        spec,
        y0s,
        dtmaxs,
        mus,
        boozer_field,
        m,
        q,
        mode,
        zetas,
        stopping_criteria,
    )


@partial(jax.jit, static_argnames=("boozer_field", "mode"))
def _trace_guiding_centers_boozer_batched_static_field_unsharded(
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
    return _trace_guiding_centers_boozer_batched_lax_map(
        spec,
        y0s,
        dtmaxs,
        mus,
        boozer_field,
        m,
        q,
        mode,
        zetas,
        stopping_criteria,
    )


def _batched_boozer_field_jit_arg(boozer_field):
    if isinstance(boozer_field, tuple) and len(boozer_field) == 2:
        return boozer_field[0], _as_device_array(boozer_field[1], jnp.float64)
    frozen = getattr(boozer_field, "frozen_state", None)
    psi0 = getattr(boozer_field, "psi0", None)
    if isinstance(
        frozen, (BoozerAnalyticFrozenState, BoozerRadialInterpolantFrozenState)
    ):
        return frozen, _as_device_array(psi0, jnp.float64)
    return boozer_field


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

    spec = _stage_guiding_center_spec(spec)
    y0s_arr = _as_device_array(y0s, jnp.float64).reshape((-1, 4))
    dtmaxs_arr = _as_device_array(dtmaxs, jnp.float64).reshape((-1,))
    mus_arr = _as_device_array(mus, jnp.float64).reshape((-1,))
    m_arr = _as_device_array(m, jnp.float64)
    q_arr = _as_device_array(q, jnp.float64)
    zetas_arr = (
        _device_zeros((0,), jnp.float64)
        if zetas is None
        else _as_device_array(zetas, jnp.float64).reshape((-1,))
    )
    boozer_field_arg = _batched_boozer_field_jit_arg(boozer_field)
    stopping_criteria = tuple(stopping_criteria)

    frozen_state, _psi0 = _resolve_boozer_field_state(boozer_field_arg)
    shardable_field = isinstance(
        frozen_state, (BoozerAnalyticFrozenState, BoozerRadialInterpolantFrozenState)
    )
    shardable_criteria = not any(
        isinstance(criterion, LevelsetStoppingCriterion)
        for criterion in stopping_criteria
    )

    config = trajectory_batch_sharding_config(y0s_arr)
    if config is not None and shardable_field and shardable_criteria:
        y0s_arr, dtmaxs_arr, mus_arr = maybe_shard_trajectory_batch_inputs(
            y0s_arr,
            dtmaxs_arr,
            mus_arr,
            config=config,
        )
        shared_state = replicate_tree_on_mesh(
            (spec, boozer_field_arg, m_arr, q_arr, zetas_arr),
            mesh=config.mesh,
        )
        shared_state_specs = jax.tree.map(lambda _leaf: P(), shared_state)

        @partial(
            jax.shard_map,
            mesh=config.mesh,
            in_specs=(
                P(config.axis_name, None),
                P(config.axis_name),
                P(config.axis_name),
                shared_state_specs,
            ),
            out_specs=GuidingCenterTracingResult(
                trajectory=P(config.axis_name, None, None),
                mask=P(config.axis_name, None),
                steps_taken=P(config.axis_name),
                status=P(config.axis_name),
                t_final=P(config.axis_name),
                phi_hits=P(config.axis_name, None, None),
                phi_hits_count=P(config.axis_name),
            ),
            check_vma=True,
        )
        def trace_shard(
            y0s_block,
            dtmaxs_block,
            mus_block,
            shared_state_block,
        ):
            spec_block, field_block, m_block, q_block, zetas_block = shared_state_block

            def trace_one(inputs):
                y0, dtmax, mu = inputs
                return trace_guiding_center_boozer(
                    replace(spec_block, dtmax=dtmax),
                    y0,
                    field_block,
                    m=m_block,
                    q=q_block,
                    mu=mu,
                    mode=mode,
                    zetas=zetas_block,
                    stopping_criteria=stopping_criteria,
                )

            return jax.lax.map(
                trace_one,
                (y0s_block, dtmaxs_block, mus_block),
            )

        return trace_shard(y0s_arr, dtmaxs_arr, mus_arr, shared_state)

    if isinstance(
        frozen_state, (BoozerAnalyticFrozenState, BoozerRadialInterpolantFrozenState)
    ):
        return _trace_guiding_centers_boozer_batched_unsharded(
            spec,
            y0s_arr,
            dtmaxs_arr,
            mus_arr,
            boozer_field_arg,
            m_arr,
            q_arr,
            mode,
            zetas_arr,
            stopping_criteria,
        )

    if isinstance(boozer_field_arg, tuple):
        return _trace_guiding_centers_boozer_batched_lax_map(
            spec,
            y0s_arr,
            dtmaxs_arr,
            mus_arr,
            boozer_field_arg,
            m_arr,
            q_arr,
            mode,
            zetas_arr,
            stopping_criteria,
        )

    return _trace_guiding_centers_boozer_batched_static_field_unsharded(
        spec,
        y0s_arr,
        dtmaxs_arr,
        mus_arr,
        boozer_field_arg,
        m_arr,
        q_arr,
        mode,
        zetas_arr,
        stopping_criteria,
    )


# ── Full-orbit Lorentz RHS (6-state Cartesian) ────────────────────────


@dataclass(frozen=True)
class FullorbitTracingSpec:
    """Immutable contract for a single full-orbit Lorentz integration call.

    Parameters mirror :class:`FieldlineTracingSpec`. The state is 6-D
    ``(x, y, z, vx, vy, vz)`` (position plus Cartesian velocity), so
    the trajectory carry has shape ``(max_steps + 1, 7)`` (columns
    ``(t, x, y, z, vx, vy, vz)``). The integrator follows the upstream
    ``FullorbitRHS::operator()`` vacuum branch in
    ``legacy native extension/tracing.cpp``. The ``phi_hits`` buffer has shape
    ``(max_phi_hits, 8)`` and records phi-plane Poincaré crossings and
    stopping-criterion fires (columns ``[t_hit, idx, x, y, z, vx, vy,
    vz]``); see :class:`FullorbitTracingResult` for the row layout.
    """

    tmax: float
    rtol: float
    atol: float
    max_steps: int
    dtmax: float = np.inf
    max_root_iters: int = 200
    max_phi_hits: int = 128
    adaptive_loop: AdaptiveLoop = "scan"


jax.tree_util.register_dataclass(
    FullorbitTracingSpec,
    data_fields=["tmax", "rtol", "atol", "dtmax"],
    meta_fields=["max_steps", "max_root_iters", "max_phi_hits", "adaptive_loop"],
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
    ``FullorbitRHS::operator()`` in ``legacy native extension/tracing.cpp``) is

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

    qoverm = _device_array(q / m, jnp.float64)

    def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
        del _t  # Field is autonomous; signature kept for ODE-driver shape.
        position, velocity = jnp.split(y, (3,))
        B_raw = magnetic_field_fn(position)
        B = jnp.asarray(B_raw, dtype=y.dtype).reshape((3,))
        acceleration = qoverm * jnp.cross(velocity, B)
        return jnp.concatenate((velocity, acceleration))

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
    k0 = rhs(t0, y0_arr)
    one = _device_array(1.0, dtype)
    lane_zero, lane_zero_i32, lane_false = _lane_axis_carry_zeroes(y0_arr)
    accepted_count_init = _device_array(0, jnp.int32) + lane_zero_i32
    t0_init = t0 + lane_zero

    # Pre-allocate the trajectory carry with columns
    # ``(t, x, y, z, vx, vy, vz)``. Row 0 holds the initial state;
    # rows 1..max_steps fill in as accepted steps occur. Padding rows
    # at the end of the run get the final accepted state.
    traj_init_row = jnp.concatenate((jnp.reshape(t0, (1,)), y0_arr), axis=0)
    traj = jnp.concatenate(
        (
            jnp.reshape(traj_init_row, (1, 7)),
            _device_zeros((max_steps, 7), dtype),
        ),
        axis=0,
    )
    traj = traj + lane_zero
    mask_indices = jax.lax.iota(jnp.int32, max_steps + 1)
    mask = mask_indices == _device_array(0, jnp.int32)
    mask = mask | lane_false

    # Phi-plane crossing buffer. Each row is ``[t_hit, idx, x, y, z, vx,
    # vy, vz]`` (8 columns to match the upstream
    # ``sopp.particle_fullorbit_tracing`` row shape).
    phi_hits_buf = _device_zeros((max_phi_hits, 8), dtype)
    phi_hits_count_init = _device_array(0, jnp.int32)
    phi_hits_buf, phi_hits_count_init = _event_carry_with_lane_axis(
        phi_hits_buf,
        phi_hits_count_init,
        y0_arr,
    )

    if phis is None:
        phis_arr = _device_zeros((0,), dtype)
    else:
        phis_arr = _as_device_array(phis, dtype).reshape((-1,))
    num_phis = int(phis_arr.shape[0])

    # Initial unwrapped phi seed (C++ tracing.cpp uses pi).
    phi_init = _continuous_phi(
        _take_entry(y0_arr, 0),
        _take_entry(y0_arr, 1),
        _device_array(np.pi, dtype),
        dtype,
    )

    init_carry = (
        _device_array(0, jnp.int32),  # step_count
        accepted_count_init,
        t0_init,
        y0_arr,
        h0,
        k0,
        traj,
        mask,
        phi_hits_buf,
        phi_hits_count_init,
        phi_init,  # running phi_last
        phi_init,  # transit criterion baseline, set on first accepted step
        _device_index(0) + lane_zero_i32,  # status_event
        lane_false,  # stop flag
        _device_index(0) + lane_zero_i32,  # consecutive non-advancing trials
    )

    max_steps_i32 = _device_index(max_steps)
    max_phi_hits_i32 = _device_index(max_phi_hits)
    two_pi = _device_two_pi(one)

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
            _no_progress,
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
            no_progress,
        ) = carry
        step = _dopri5_adaptive_step(
            rhs,
            t,
            y,
            h,
            k_first,
            tmax,
            dtmax,
            rtol,
            atol,
            dtype,
        )
        h_clamped = step.h_clamped
        y_new = step.y_new
        accepted = step.accepted
        h_next = step.h_next
        t_next = step.t_next
        y_next = step.y_next
        k_next = step.k_next

        # ── Phi-plane crossing detection on accepted steps ──
        phi_current = _continuous_phi(y_new[0], y_new[1], phi_last, dtype)

        state_at_time = _dense_output_query(step.stages, y, t, h_clamped)

        (
            phi_hits_after,
            phi_count_after,
            status_scan,
            stop_scan,
        ) = _scan_angle_plane_events(
            hits=phi_hits_in,
            count=phi_hits_count_in,
            status=status_event,
            stop=_device_false(),
            angle_last=phi_last,
            angle_current=phi_current,
            targets=phis_arr,
            num_targets=num_phis,
            two_pi=two_pi,
            dtype=dtype,
            t=t,
            h_clamped=h_clamped,
            max_root_iters=max_root_iters,
            enabled=accepted,
            max_hits_i32=max_phi_hits_i32,
            state_at_time=state_at_time,
            angle_at_state=lambda state, angle_near: _continuous_phi_from_state(
                state, angle_near, dtype
            ),
        )

        # ── Stopping criteria check on accepted state ──
        first_accepted_step = accepted_count == _device_index(0)
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
            return _apply_stopping_criteria_events(
                stopping_criteria=stopping_criteria,
                hits=hits_in,
                count=count_in,
                status=status_in,
                stop=stop_in,
                iter_count=iter_count_in,
                angle_current=phi_curr_in,
                angle_initial=phi_init_in,
                t_event=t_next,
                state=y_next,
                dtype=dtype,
                max_hits_i32=max_phi_hits_i32,
            )

        iter_count_post = step_count + _device_index(1)

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
                status_scan,
                stop_scan,
                iter_count_post,
                phi_current,
                phi_init_for_criteria,
            ),
        )
        _, status_zero_i32, status_false = _lane_axis_carry_zeroes(y_next)
        status_after = status_after + status_zero_i32
        stop_after = stop_after | status_false

        phi_last_next = jnp.where(accepted, phi_current, phi_last)
        phi_init_next = jnp.where(
            jnp.logical_and(accepted, first_accepted_step),
            phi_current,
            phi_init,
        )
        # boost's failed-step / no-progress checks; see
        # ``_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS``.
        no_progress_next, status_after, stop_after = _step_control_progress(
            no_progress=no_progress,
            accepted=accepted,
            nonfinite_state=step.nonfinite_state,
            t=t,
            t_next=t_next,
            y=y,
            k_first=k_first,
            status=status_after,
            stop=stop_after,
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
            step_count + _device_index(1),
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
            no_progress_next,
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
        no_progress_final,
    ) = _run_adaptive_steps(
        cond,
        body,
        init_carry,
        max_steps,
        spec.adaptive_loop,
    )

    last_row = jnp.concatenate((jnp.reshape(t_final, (1,)), y_final.reshape((6,))))
    traj_padded = jnp.where(
        mask_final[:, None],
        traj_final,
        jnp.broadcast_to(last_row, traj_final.shape),
    )

    eps_t = _device_array(1.0e-12, dtype) * jnp.maximum(
        jnp.abs(tmax), _device_array(1.0, dtype)
    )
    reached = (tmax - t_final) <= eps_t
    status_normal = jnp.where(
        reached,
        _device_index(0),
        _device_index(1),
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


@partial(jax.jit, static_argnames=("magnetic_field_fn",))
def _trace_fullorbits_batched_unsharded(
    spec: FullorbitTracingSpec,
    y0s: jax.Array,
    dtmaxs: jax.Array,
    magnetic_field_fn: Callable[[jax.Array], jax.Array],
    m: float,
    q: float,
    phis: jax.Array | None = None,
    stopping_criteria: tuple = (),
    magnetic_field_state: object | None = None,
) -> FullorbitTracingResult:
    def trace_one(
        y0: jax.Array,
        dtmax: jax.Array,
        field_state: object | None = None,
    ) -> FullorbitTracingResult:
        if field_state is None:
            field_fn = magnetic_field_fn
        else:

            def field_fn(point):
                return magnetic_field_fn(field_state, point)

        return trace_fullorbit(
            replace(spec, dtmax=dtmax),
            y0,
            field_fn,
            m=m,
            q=q,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    if magnetic_field_state is None:
        return jax.vmap(trace_one)(y0s, dtmaxs)
    return jax.vmap(trace_one, in_axes=(0, 0, None))(
        y0s,
        dtmaxs,
        magnetic_field_state,
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
    magnetic_field_state: object | None = None,
) -> FullorbitTracingResult:
    """Trace full-orbit Lorentz trajectories with one vmapped JAX graph."""

    spec = _stage_fullorbit_spec(spec)
    y0s_arr = _as_device_array(y0s, jnp.float64).reshape((-1, 6))
    dtmaxs_arr = _as_device_array(dtmaxs, jnp.float64).reshape((-1,))
    stopping_criteria = tuple(stopping_criteria)

    def trace_one(
        y0: jax.Array,
        dtmax: jax.Array,
        field_state: object | None = None,
    ) -> FullorbitTracingResult:
        if field_state is None:
            field_fn = magnetic_field_fn
        else:

            def field_fn(point):
                return magnetic_field_fn(field_state, point)

        return trace_fullorbit(
            replace(spec, dtmax=dtmax),
            y0,
            field_fn,
            m=m,
            q=q,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )

    config = trajectory_batch_sharding_config(y0s_arr)
    if config is not None:
        y0s_arr, dtmaxs_arr = maybe_shard_trajectory_batch_inputs(
            y0s_arr,
            dtmaxs_arr,
            config=config,
        )

        out_specs = FullorbitTracingResult(
            trajectory=P(config.axis_name, None, None),
            mask=P(config.axis_name, None),
            steps_taken=P(config.axis_name),
            status=P(config.axis_name),
            t_final=P(config.axis_name),
            phi_hits=P(config.axis_name, None, None),
            phi_hits_count=P(config.axis_name),
        )
        if magnetic_field_state is None:

            @partial(
                jax.shard_map,
                mesh=config.mesh,
                in_specs=(P(config.axis_name, None), P(config.axis_name)),
                out_specs=out_specs,
                check_vma=True,
            )
            def trace_shard(y0s_block, dtmaxs_block):
                return jax.lax.map(
                    lambda inputs: trace_one(*inputs),
                    (y0s_block, dtmaxs_block),
                )

            return trace_shard(y0s_arr, dtmaxs_arr)

        field_state_specs = jax.tree.map(lambda _leaf: P(), magnetic_field_state)

        @partial(
            jax.shard_map,
            mesh=config.mesh,
            in_specs=(
                P(config.axis_name, None),
                P(config.axis_name),
                field_state_specs,
            ),
            out_specs=out_specs,
            check_vma=True,
        )
        def trace_shard(y0s_block, dtmaxs_block, field_state_block):
            return jax.lax.map(
                lambda inputs: trace_one(inputs[0], inputs[1], field_state_block),
                (y0s_block, dtmaxs_block),
            )

        return trace_shard(y0s_arr, dtmaxs_arr, magnetic_field_state)

    return _trace_fullorbits_batched_unsharded(
        spec,
        y0s_arr,
        dtmaxs_arr,
        magnetic_field_fn,
        m,
        q,
        phis,
        stopping_criteria,
        magnetic_field_state,
    )
