"""The event localizer's bracket is upstream's, and the stall path is finite.

``legacy native extension/tracing.cpp`` hands ``toms748_solve`` the two residuals
the crossing test itself compared::

    tracing.cpp:416  toms748_solve(rootfun, tlast, tcurrent,
                                   phi_last - phi_shift, phi_current - phi_shift,
                                   roottol, rootmaxit)

so the root function is never evaluated at the bracket ends, and the assertion
one line above the lambda (``tracing.cpp:405``) -- the shifted target lies
between ``phi_last`` and ``phi_current`` -- is a property of the values the
localizer actually receives. Re-evaluating an end through the dense-output
polynomial does not reproduce it: ``((t + h) - t) / h`` is not exactly ``1`` in
double precision once ``t >> h``. The first test states that dependency as a
fault injection -- the polynomial is poisoned at the ends -- so a localizer that
reads them cannot produce a usable hit.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.core.tracing import (
    _ROOT_BRACKET_EPS,
    TRACING_STATUS_STEP_CONTROL_FAILED,
    FieldlineTracingSpec,
    IterStoppingCriterion,
    _append_event_row,
    _apply_stopping_criteria_events,
    _continuous_phi,
    _continuous_phi_from_state,
    _dense_output_query,
    _dense_output_state,
    _dopri5_stage_step,
    _scan_angle_plane_events,
    trace_fieldline,
)

DTYPE = jnp.float64
# Rigid rotation about the z axis at unit angular speed: ``d(phi)/dt == 1``, so
# a time uncertainty converts to an angle residual one for one.
ANGULAR_SPEED = 1.0
# A step far enough from the origin that ``((t + h) - t) / h != 1``.
_STEP_START = 40.0
_STEP_SIZE = 0.05


def _rigid_rotation(_t, y):
    return jnp.stack([-y[1], y[0], jnp.zeros((), DTYPE)])


def _angle_at_state(state, angle_near):
    return _continuous_phi_from_state(state, angle_near, DTYPE)


def _one_step():
    """One accepted DOPRI5 step of the rigid rotation, with its two angles."""
    t = jnp.asarray(_STEP_START, DTYPE)
    h = jnp.asarray(_STEP_SIZE, DTYPE)
    y = jnp.asarray([1.4, 0.0, 0.0], DTYPE)
    stages = _dopri5_stage_step(_rigid_rotation, t, y, h, _rigid_rotation(t, y))
    angle_last = _continuous_phi(y[0], y[1], jnp.asarray(0.0, DTYPE), DTYPE)
    angle_current = _continuous_phi(stages.y_new[0], stages.y_new[1], angle_last, DTYPE)
    return t, h, y, stages, angle_last, angle_current


@pytest.mark.parametrize("poisoned_end", ["left", "right", "both"])
def test_localizer_does_not_read_the_dense_output_at_the_bracket_ends(
    poisoned_end: str,
) -> None:
    """A crossing is localized without ever evaluating ``t`` or ``t + h``.

    The target sits half way across the step, so the root is interior and the
    localizer has no honest reason to query an end. A localizer that takes
    ``angle_last`` and ``angle_current`` -- upstream's ``phi_last - phi_shift``
    and ``phi_current - phi_shift`` -- records the crossing normally whichever
    end is poisoned.
    """
    t, h, y, stages, angle_last, angle_current = _one_step()
    t_end = t + h
    target = jnp.asarray(0.5, DTYPE) * (angle_last + angle_current)
    poison_left = poisoned_end in ("left", "both")
    poison_right = poisoned_end in ("right", "both")

    def poisoned_state_at_time(t_query):
        state = _dense_output_state(stages, y, h, (t_query - t) / h)
        at_end = jnp.logical_or(
            jnp.logical_and(jnp.asarray(poison_left), t_query == t),
            jnp.logical_and(jnp.asarray(poison_right), t_query == t_end),
        )
        return jnp.where(at_end, jnp.full_like(state, jnp.nan), state)

    hits, count, status, stop = _scan_angle_plane_events(
        hits=jnp.zeros((8, 5), DTYPE),
        count=jnp.asarray(0, jnp.int32),
        status=jnp.asarray(0, jnp.int32),
        stop=jnp.asarray(False),
        angle_last=angle_last,
        angle_current=angle_current,
        targets=jnp.reshape(target, (1,)),
        num_targets=1,
        two_pi=jnp.asarray(2.0 * np.pi, DTYPE),
        dtype=DTYPE,
        t=t,
        h_clamped=h,
        max_root_iters=200,
        enabled=jnp.asarray(True),
        max_hits_i32=jnp.asarray(8, jnp.int32),
        state_at_time=poisoned_state_at_time,
        angle_at_state=_angle_at_state,
    )

    assert int(count) == 1
    assert int(status) == 0
    assert not bool(stop)
    row = np.asarray(hits[0], dtype=np.float64)
    assert np.all(np.isfinite(row)), f"localizer read a poisoned bracket end: {row}"
    t_root = row[0]
    assert float(t) < t_root < float(t_end)
    angle_root = float(
        _continuous_phi(
            jnp.asarray(row[2], DTYPE), jnp.asarray(row[3], DTYPE), angle_last, DTYPE
        )
    )
    # ``boost::math::tools::eps_tolerance`` stops the bracket at
    # ``|a - b| <= eps * min(|a|, |b|)``; at unit angular speed that time
    # uncertainty IS the angle residual, so the bound is computed, not written.
    residual_bound = _ROOT_BRACKET_EPS * abs(t_root) * ANGULAR_SPEED
    assert abs(angle_root - float(target)) <= residual_bound


def _toroidal_b(point):
    x, y, _z = point[0], point[1], point[2]
    radius_sq = x * x + y * y
    return jnp.stack([-y / radius_sq, x / radius_sq, jnp.zeros((), DTYPE)])


def test_stalled_lane_reports_step_control_failure_under_debug_nans():
    """The stall guard must be reachable in the repo's ``jax_debug_nans`` mode.

    ``dtmax = 0`` makes ``_clamp_step_to_domain`` return exactly ``0.0`` on every
    trial -- the h-underflow stall ``_MAX_CONSECUTIVE_NO_PROGRESS_TRIALS``
    documents: the embedded error is ``0``, so the trial is ACCEPTED while ``t``
    stands still. The event scan runs on every trial, so its dense-output query
    must not divide by that zero step; a zero-length step has one query time,
    ``t``, where the dense output is ``y``.
    """
    spec = FieldlineTracingSpec(
        tmax=1.0,
        rtol=1.0e-9,
        atol=1.0e-9,
        max_steps=520,
        dtmax=0.0,
        adaptive_loop="while",
    )
    with jax.debug_nans(True):
        result = trace_fieldline(
            spec,
            jnp.asarray([1.4, 0.0, 0.0], DTYPE),
            _toroidal_b,
            phis=jnp.asarray([0.3], DTYPE),
        )

    assert int(result.status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert float(result.t_final) == 0.0
    assert np.all(np.isfinite(np.asarray(result.trajectory, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(result.phi_hits, dtype=np.float64)))
    assert int(result.phi_hits_count) == 0


def test_a_rejected_non_finite_row_does_not_poison_the_event_buffer():
    """``particle-nan``: one bad row must not rewrite the rows already kept.

    The event carry is given the ``vmap`` lane axis by adding a zero derived
    from the candidate row. Derived by SUBTRACTING the row from itself, that
    zero is ``nan`` for a non-finite candidate and turns every recorded row into
    ``nan`` -- which is how a shipped-scale particle run published an all-``nan``
    ``poincare:positions`` (packet
    ``d1-diagnostic/runs/native-tracing-particle/20260920T054912Z-b0209248.partial``:
    10 recorded rows, all ten non-finite).
    """
    recorded = (
        jnp.zeros((4, 5), DTYPE)
        .at[0]
        .set(jnp.asarray([0.25, 0.0, 1.4, 0.0, 0.0], DTYPE))
    )
    hits, count = _append_event_row(
        recorded,
        jnp.asarray(1, jnp.int32),
        jnp.asarray(False),
        jnp.full((5,), jnp.nan, DTYPE),
        jnp.asarray(4, jnp.int32),
    )

    assert int(count) == 1
    kept = np.asarray(hits[0], dtype=np.float64)
    assert np.all(np.isfinite(kept)), (
        f"a rejected non-finite row poisoned the buffer: {kept}"
    )
    np.testing.assert_array_equal(kept, np.asarray([0.25, 0.0, 1.4, 0.0, 0.0]))


def test_non_finite_angles_are_not_a_plane_crossing():
    """``particle-nan``: ``floor(nan) != floor(nan)`` is not a crossing.

    Upstream's test (``legacy native extension/tracing.cpp``
    ``std::floor((phi_last-phi)/(2*M_PI)) != std::floor((phi_current-phi)/(2*M_PI))``)
    runs on doubles boost keeps finite -- it throws through ``max_step_checker``
    before a non-finite state can be integrated, which this port reports as
    ``TRACING_STATUS_STEP_CONTROL_FAILED``. In IEEE arithmetic ``nan != nan`` is
    TRUE, so a lane that has gone non-finite would otherwise record an event row
    for every plane on every remaining trial.
    """
    t, h, y, stages, _angle_last, _angle_current = _one_step()
    nan = jnp.asarray(np.nan, DTYPE)

    hits, count, status, stop = _scan_angle_plane_events(
        hits=jnp.zeros((8, 5), DTYPE),
        count=jnp.asarray(0, jnp.int32),
        status=jnp.asarray(0, jnp.int32),
        stop=jnp.asarray(False),
        angle_last=nan,
        angle_current=nan,
        targets=jnp.asarray([0.3], DTYPE),
        num_targets=1,
        two_pi=jnp.asarray(2.0 * np.pi, DTYPE),
        dtype=DTYPE,
        t=t,
        h_clamped=h,
        max_root_iters=200,
        enabled=jnp.asarray(True),
        max_hits_i32=jnp.asarray(8, jnp.int32),
        state_at_time=_dense_output_query(stages, y, t, h),
        angle_at_state=_angle_at_state,
    )

    assert int(count) == 0
    assert np.all(np.isfinite(np.asarray(hits, dtype=np.float64)))
    # A trial that records nothing because it never crossed is not a failure:
    # the scan leaves the lane's status and stop flag exactly as it found them.
    assert int(status) == 0
    assert not bool(stop)


def test_a_stopping_criterion_does_not_fire_on_a_non_finite_state():
    """The second producer of published rows refuses a non-finite state.

    Upstream evaluates every stopping criterion on the accepted post-step state
    and pushes ``{t, -1 - i, y}`` from it (``legacy native extension/tracing.cpp``,
    the criterion loop that follows the step). Its state there is finite,
    because boost's step acceptance looks only at the error norm and would carry
    a non-finite state on; this port ends such a lane at the last finite state
    instead (a lane never reports success on, nor publishes, non-finite values),
    so the criterion must not fire, must not set a status and must not push a
    row. ``IterStoppingCriterion`` is used because its predicate does not read
    the state, so the only thing that can stop it firing is the finiteness gate:
    the same call with a finite state DOES fire.
    """
    kwargs = dict(
        stopping_criteria=(IterStoppingCriterion(0),),
        hits=jnp.zeros((4, 5), DTYPE),
        count=jnp.asarray(0, jnp.int32),
        status=jnp.asarray(0, jnp.int32),
        stop=jnp.asarray(False),
        iter_count=jnp.asarray(1, jnp.int32),
        angle_current=jnp.asarray(0.0, DTYPE),
        angle_initial=jnp.asarray(0.0, DTYPE),
        t_event=jnp.asarray(0.25, DTYPE),
        dtype=DTYPE,
        max_hits_i32=jnp.asarray(4, jnp.int32),
    )

    hits, count, status, stop = _apply_stopping_criteria_events(
        state=jnp.full((3,), jnp.nan, DTYPE), **kwargs
    )
    assert int(count) == 0
    assert int(status) == 0
    assert not bool(stop)
    assert np.all(np.isfinite(np.asarray(hits, dtype=np.float64)))

    fired_hits, fired_count, fired_status, fired_stop = _apply_stopping_criteria_events(
        state=jnp.asarray([1.4, 0.0, 0.0], DTYPE), **kwargs
    )
    assert int(fired_count) == 1
    assert int(fired_status) == -1
    assert bool(fired_stop)
    np.testing.assert_array_equal(
        np.asarray(fired_hits[0], dtype=np.float64),
        np.asarray([0.25, -1.0, 1.4, 0.0, 0.0]),
    )


def _one_step_with_a_non_finite_fsal_stage():
    """One accepted step whose ``k7`` is ``nan`` while its new state stays finite.

    Exactly the step boost accepts and this port mirrors (decision 16): the right-hand side is ``nan`` only AT the
    step's own ``y_new``, which is where the FSAL stage ``k7 = f(t + h, y_new)`` is evaluated and no other stage is.
    """
    t = jnp.asarray(_STEP_START, DTYPE)
    h = jnp.asarray(_STEP_SIZE, DTYPE)
    y = jnp.asarray([1.4, 0.0, 0.0], DTYPE)
    reference = _dopri5_stage_step(_rigid_rotation, t, y, h, _rigid_rotation(t, y))

    def end_point_nan_rhs(t_stage, state):
        value = _rigid_rotation(t_stage, state)
        at_end = jnp.all(state == reference.y_new)
        return jnp.where(at_end, jnp.full_like(value, jnp.nan), value)

    stages = _dopri5_stage_step(end_point_nan_rhs, t, y, h, end_point_nan_rhs(t, y))
    angle_last = _continuous_phi(y[0], y[1], jnp.asarray(0.0, DTYPE), DTYPE)
    angle_current = _continuous_phi(stages.y_new[0], stages.y_new[1], angle_last, DTYPE)
    return t, h, y, stages, angle_last, angle_current


def test_a_non_finite_stage_poisons_the_dense_output_at_both_ends():
    """``_dense_output_state`` is exact at the ends only while every stage is finite.

    ``b7_theta`` is exactly ``0`` at ``theta = 0`` and at ``theta = 1``, and ``0 * nan`` is ``nan``, so a step whose
    FSAL stage is non-finite has a non-finite dense output at EVERY query time -- the ends included. Upstream's
    ``calc_state`` weights ``deriv_new`` the same way inside one ``scale_sum7``
    (``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp``), so this is upstream's arithmetic and not a port
    defect; what the port must not do is publish a row taken from it.
    """
    t, h, y, stages, _angle_last, _angle_current = (
        _one_step_with_a_non_finite_fsal_stage()
    )

    assert np.all(np.isnan(np.asarray(stages.k7, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(stages.y_new, dtype=np.float64)))
    for theta in (0.0, 0.5, 1.0):
        state = _dense_output_state(stages, y, h, jnp.asarray(theta, DTYPE))
        assert np.all(np.isnan(np.asarray(state, dtype=np.float64))), theta
    # The same query with the finite step IS exact at both ends, so the assertion above is about the poisoned
    # stage and not about the polynomial being wrong.
    finite_stages = _dopri5_stage_step(_rigid_rotation, t, y, h, _rigid_rotation(t, y))
    np.testing.assert_array_equal(
        np.asarray(_dense_output_state(finite_stages, y, h, jnp.asarray(0.0, DTYPE))),
        np.asarray(y),
    )
    np.testing.assert_array_equal(
        np.asarray(_dense_output_state(finite_stages, y, h, jnp.asarray(1.0, DTYPE))),
        np.asarray(finite_stages.y_new),
    )


def test_a_crossing_whose_row_is_non_finite_is_not_recorded_and_stops_the_lane():
    """The last unguarded row producer: the angle-plane scan.

    The crossing test itself runs on the step's two ANGLES, which are finite here (they come from ``y`` and from a
    finite ``y_new``), so ``angles_finite`` cannot see this case; the row comes from the dense output, which carries
    the ``nan`` FSAL stage. Upstream pushes that row and carries the ``nan`` on
    (``legacy native extension/tracing.cpp``); this port may not publish it, and dropping it silently would lose a
    crossing upstream records, so the lane stops with ``TRACING_STATUS_STEP_CONTROL_FAILED``.

    The control below is the SAME call on the finite step: it records the row, leaves the status at ``0`` and does
    not stop, so the test cannot pass by the scan never recording anything.
    """
    t, h, y, stages, angle_last, angle_current = (
        _one_step_with_a_non_finite_fsal_stage()
    )
    target = jnp.asarray(0.5, DTYPE) * (angle_last + angle_current)

    def scan(step_stages):
        return _scan_angle_plane_events(
            hits=jnp.zeros((8, 5), DTYPE),
            count=jnp.asarray(0, jnp.int32),
            status=jnp.asarray(0, jnp.int32),
            stop=jnp.asarray(False),
            angle_last=angle_last,
            angle_current=angle_current,
            targets=jnp.reshape(target, (1,)),
            num_targets=1,
            two_pi=jnp.asarray(2.0 * np.pi, DTYPE),
            dtype=DTYPE,
            t=t,
            h_clamped=h,
            max_root_iters=200,
            enabled=jnp.asarray(True),
            max_hits_i32=jnp.asarray(8, jnp.int32),
            state_at_time=_dense_output_query(step_stages, y, t, h),
            angle_at_state=_angle_at_state,
        )

    hits, count, status, stop = scan(stages)
    assert int(count) == 0
    assert np.all(np.isfinite(np.asarray(hits, dtype=np.float64)))
    assert int(status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert bool(stop)

    finite_stages = _dopri5_stage_step(_rigid_rotation, t, y, h, _rigid_rotation(t, y))
    control_hits, control_count, control_status, control_stop = scan(finite_stages)
    assert int(control_count) == 1
    assert np.all(np.isfinite(np.asarray(control_hits[0], dtype=np.float64)))
    assert int(control_status) == 0
    assert not bool(control_stop)
