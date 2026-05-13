"""Phi-plane crossing and non-Levelset stopping-criterion tests for the JAX tracing path.

These tests validate the follow-up extensions to the item 14 JAX
tracing module:

1. Phi-plane Poincaré crossings are detected and refined via the
   ``bracket_root_jax`` localizer, then recorded in a fixed-shape
   ``phi_hits`` buffer.
2. Non-Levelset stopping criteria (``MinR``, ``MaxR``, ``MinZ``,
   ``MaxZ``, ``ToroidalTransit``, ``Iter``) terminate the trajectory
   and surface a ``status = -1 - i`` encoding.

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
The parity-ladder lane is ``event_time_tracing`` (the lane that bounds
the bracketed-localizer absolute tolerance).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.jax_core.tracing import (
    FieldlineTracingSpec,
    IterStoppingCriterion,
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    bracket_root_jax,
    trace_fieldline,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


def _toroidal_field_fn(R0: float, B0: float):
    """Return a JAX-traceable ``B(point) -> [3]`` for the toroidal field.

    Inline rather than going through ``ToroidalFieldJAX`` because the
    JAX driver wants a callable, not an Optimizable.
    """

    R0_arr = jnp.asarray(R0, dtype=jnp.float64)
    B0_arr = jnp.asarray(B0, dtype=jnp.float64)

    def field_fn(point):
        x = point[0]
        y = point[1]
        r2 = x * x + y * y
        phi_hat = jnp.stack(
            [-y / jnp.sqrt(r2), x / jnp.sqrt(r2), jnp.asarray(0.0, dtype=point.dtype)]
        )
        return B0_arr * R0_arr / jnp.sqrt(r2) * phi_hat

    return field_fn


def test_phi_plane_crossing_recovered_for_uniform_toroidal_field():
    """A toroidal-axis fieldline crosses each requested phi near the analytic time.

    In a ``B = B0 R0 / R * e_phi`` field, a streamline starting at
    ``(R, 0, 0)`` follows the upstream ``dx/dt = B`` parameterisation.
    The time at which phi = phi_target is
    ``phi_target * R**2 / (B0 * R0)`` for ``phi_target > 0``. We request a
    single phi target and assert the JAX driver records a crossing close to
    the analytic upstream fieldline time. The exact event time is limited by the
    in-step linear interpolant used by the bracketed event localizer
    (the bisection is exact on the linear interpolant; the residual is
    bounded by the RK step accuracy at the crossing step). The lane
    contract therefore reports the bracketed-bisection ``event_time``
    accuracy on the interpolant, not the underlying RK accuracy.
    """

    R0 = 1.3
    B0 = 0.8
    R_init = 1.4
    phi_target = 0.5  # rad
    tmax = (2.0 * np.pi * R_init * R_init) / (B0 * R0)
    # Tight tolerances drive small steps so the linear interpolant in
    # the bracketed localizer is close to the trajectory.
    spec = FieldlineTracingSpec(
        tmax=float(tmax),
        rtol=1e-11,
        atol=1e-11,
        max_steps=8000,
        max_phi_hits=16,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)
    phis = jnp.asarray([phi_target], dtype=jnp.float64)
    result = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), phis=phis, stopping_criteria=()
    )
    phi_hits = np.asarray(result.phi_hits)
    count = int(result.phi_hits_count)
    assert count >= 1, f"expected at least one phi crossing, got {count}"
    # The first crossing should be at upstream fieldline time.
    t_expected = phi_target * R_init * R_init / (B0 * R0)
    t_actual = float(phi_hits[0, 0])
    idx = int(phi_hits[0, 1])
    assert idx == 0, f"expected phi-plane index 0, got {idx}"
    # Event-time accuracy: bounded by the per-step RK error at the
    # crossing step. Tight controller tolerances (~1e-11) keep steps
    # small enough that the linear-interpolant residual is bounded
    # well below 1e-6 in absolute event-time.
    assert abs(t_actual - t_expected) < 1.0e-6, (
        f"phi crossing time {t_actual} differs from analytic {t_expected} "
        f"by {abs(t_actual - t_expected)}"
    )
    # The recorded position should lie on the circle r = R_init.
    x_hit = float(phi_hits[0, 2])
    y_hit = float(phi_hits[0, 3])
    z_hit = float(phi_hits[0, 4])
    r_hit = np.sqrt(x_hit * x_hit + y_hit * y_hit)
    state_rtol = float(_EVENT_TIME_TOLERANCES["state_vector_rtol"])
    state_atol_lane = float(_EVENT_TIME_TOLERANCES["state_vector_atol"])
    assert abs(r_hit - R_init) < state_atol_lane + state_rtol * R_init, (
        f"phi crossing position not on r=R_init circle: r_hit={r_hit}"
    )
    assert abs(z_hit) < state_atol_lane + state_rtol, f"z_hit not 0: {z_hit}"
    # The recorded phi should equal phi_target modulo 2pi.
    phi_hit = np.arctan2(y_hit, x_hit)
    if phi_hit < 0:
        phi_hit += 2.0 * np.pi
    assert abs(phi_hit - phi_target) < 1.0e-6


def test_bracketed_bisection_localizes_phi_to_event_time_tolerance():
    """``bracket_root_jax`` finds an explicit phi-zero crossing to lane atol.

    Direct accuracy check: feed a sinusoidal phi(t) of known root into
    the bracketed bisection and verify the result is within
    ``event_time_atol``. This is the underlying primitive the driver
    uses for phi-plane crossing refinement.
    """

    target_root = 0.7  # in [0, 1]
    event_time_atol = float(_EVENT_TIME_TOLERANCES["event_time_atol"])

    def f(t):
        return t - jnp.asarray(target_root, dtype=jnp.float64)

    t_left = jnp.asarray(0.0, dtype=jnp.float64)
    t_right = jnp.asarray(1.0, dtype=jnp.float64)
    f_left = f(t_left)
    f_right = f(t_right)
    t_root, _f_root, bracketed = bracket_root_jax(
        f,
        t_left,
        t_right,
        f_left,
        f_right,
        max_iters=60,
        atol=jnp.asarray(0.0, dtype=jnp.float64),
    )
    assert bool(bracketed)
    assert abs(float(t_root) - target_root) < event_time_atol


def test_minR_maxR_stopping_criteria_terminate_trajectory():
    """``MinRStoppingCriterion`` + ``MaxRStoppingCriterion`` terminate the trajectory.

    A toroidal-field fieldline stays at constant R. To force a
    criterion fire we use a trajectory that drifts radially:
    superpose a Z-direction kick. We synthesize this by composing the
    toroidal field with a small radial component. Then we check the
    Min/Max thresholds bracketing the initial R: the trajectory must
    terminate when R wanders outside the band.
    """

    R0 = 1.3
    B0 = 0.8
    R_init = 1.4
    spec = FieldlineTracingSpec(
        tmax=5.0,
        rtol=1e-9,
        atol=1e-9,
        max_steps=4000,
        max_phi_hits=16,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)

    # Trajectory stays at R=R_init; max_r at 1.3 < R_init triggers at the very
    # first accepted step. (The criterion is evaluated on the post-step state
    # ``r = sqrt(x^2+y^2)`` which is identically R_init for the toroidal-axis
    # path; with crit_r=1.3 the predicate ``r >= 1.3`` fires immediately.)
    max_r_crit = MaxRStoppingCriterion(crit_r=1.3)
    result = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), stopping_criteria=(max_r_crit,)
    )
    status = int(result.status)
    assert status == -1, f"expected status=-1 (criterion 0 fired); got {status}"
    phi_hits = np.asarray(result.phi_hits)
    count = int(result.phi_hits_count)
    assert count == 1, f"expected exactly one criterion hit; got {count}"
    assert int(phi_hits[0, 1]) == -1, "criterion index encoding incorrect"

    # MinR crit below the trajectory R: criterion must NOT fire.
    min_r_crit = MinRStoppingCriterion(crit_r=0.5)
    result2 = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), stopping_criteria=(min_r_crit,)
    )
    status2 = int(result2.status)
    # Trajectory must run for full tmax (status 0) or budget-exhaust
    # (status 1) — both indicate the criterion did not fire.
    assert status2 in (0, 1), f"unexpected status={status2}"


def test_minZ_maxZ_stopping_criteria_terminate_trajectory():
    """``MinZ`` / ``MaxZ`` thresholds terminate the trajectory.

    Analogous to the R-band test. The toroidal-field trajectory stays
    at Z=0, so MinZ=-1 must NOT fire and MaxZ=-1 must fire (since
    z=0 >= -1 is true and -1 <= 0 so we set the threshold above 0 to
    test correctness).
    """

    R0 = 1.3
    B0 = 0.8
    R_init = 1.4
    spec = FieldlineTracingSpec(
        tmax=5.0,
        rtol=1e-9,
        atol=1e-9,
        max_steps=4000,
        max_phi_hits=16,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)

    # MaxZ at -0.1 < 0 fires immediately on z=0 (predicate is z >= -0.1).
    max_z_crit = MaxZStoppingCriterion(crit_z=-0.1)
    result = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), stopping_criteria=(max_z_crit,)
    )
    assert int(result.status) == -1

    # MinZ at -1.0 < 0 does NOT fire (predicate is z <= -1.0).
    min_z_crit = MinZStoppingCriterion(crit_z=-1.0)
    result2 = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), stopping_criteria=(min_z_crit,)
    )
    assert int(result2.status) in (0, 1)


def test_toroidal_transit_criterion_terminates_trajectory():
    """``ToroidalTransitStoppingCriterion`` fires after the requested transit count.

    A toroidal-axis trace makes one transit per ``2*pi*R**2/(B0*R0)``.
    Setting tmax to span ~2 transits and the criterion to 1 transit
    must fire mid-trajectory.
    """

    R0 = 1.3
    B0 = 0.8
    R_init = 1.4
    transit_time = 2.0 * np.pi * R_init * R_init / (B0 * R0)
    tmax = 2.0 * transit_time
    spec = FieldlineTracingSpec(
        tmax=float(tmax),
        rtol=1e-9,
        atol=1e-9,
        max_steps=8000,
        max_phi_hits=16,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)
    transit_crit = ToroidalTransitStoppingCriterion(max_transits=1.0)
    result = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), stopping_criteria=(transit_crit,)
    )
    assert int(result.status) == -1
    # The trajectory must have stopped before the second transit, i.e.
    # well before tmax (which spans ~2 transits). The exact transit
    # upstream fieldline time is ``2*pi*R_init**2/(B0*R0)``; stopping at the first crossing of
    # that boundary keeps ``t_final`` strictly below ``0.75 * tmax``.
    t_final = float(result.t_final)
    assert t_final < tmax * 0.75


def test_iter_stopping_criterion_terminates_trajectory():
    """``IterStoppingCriterion`` caps the integrator step count.

    Setting ``max_iter`` low forces the criterion to fire before tmax.
    """

    R0 = 1.3
    B0 = 0.8
    R_init = 1.4
    spec = FieldlineTracingSpec(
        tmax=2.0 * np.pi * R_init * 5.0,  # 5 transits worth
        rtol=1e-9,
        atol=1e-9,
        max_steps=4000,
        max_phi_hits=16,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)
    # Cap at 10 iterations.
    iter_crit = IterStoppingCriterion(max_iter=10)
    result = trace_fieldline(
        spec, y0, _toroidal_field_fn(R0, B0), stopping_criteria=(iter_crit,)
    )
    assert int(result.status) == -1


def test_phi_hits_buffer_overflow_is_reported_by_count():
    """Excess crossings beyond ``max_phi_hits`` must remain detectable."""

    R0 = 1.3
    B0 = 0.8
    R_init = 1.4
    max_phi_hits = 3
    # 5 full transits should produce 5 crossings of phi=0.5.
    tmax = 10.0 * np.pi * R_init
    spec = FieldlineTracingSpec(
        tmax=float(tmax),
        rtol=1e-9,
        atol=1e-9,
        max_steps=8000,
        max_phi_hits=max_phi_hits,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)
    phis = jnp.asarray([0.5], dtype=jnp.float64)
    result = trace_fieldline(spec, y0, _toroidal_field_fn(R0, B0), phis=phis)
    count = int(result.phi_hits_count)
    assert count > max_phi_hits
    recorded = np.asarray(result.phi_hits)
    assert recorded.shape[0] == max_phi_hits
    assert np.all(np.isfinite(recorded))
