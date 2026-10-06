"""The JAX DOPRI5 step-size controller must be upstream's, step for step.

Upstream integrates with ``make_dense_output(tol, tol, dtmax, runge_kutta_dopri5)``
(``simsoptpp/tracing.cpp``), i.e. boost.odeint's ``controlled_runge_kutta`` with
``default_error_checker`` and ``default_step_adjuster``:

* error  = ``max_i |y_err_i| / (atol + rtol * (|y_old_i| + |h| * |dydt_old_i|))``
  (infinity norm, and the scale carries the derivative term),
* accept iff NOT ``error > 1`` (the ``nan`` semantics of that test matter: see
  ``test_error_norm_is_boosts_norm_inf`` below),
* on reject ``h *= max(0.9 * error**(-1/(error_order - 1)), 0.2)``,
* on accept ``h *= 0.9 * max(error, 5**-stepper_order)**(-1/stepper_order)``
  **only when** ``error < 0.5``, otherwise ``h`` is unchanged,

with ``runge_kutta_dopri5`` declaring ``stepper_order() == 5`` and
``error_order() == 4``.

Before this was mirrored the port used Hairer's RMS controller, which accepts
systematically larger steps. On the analytic toroidal field below that showed up
as 291 JAX rows against upstream's 313, a maximum relative accepted-step
difference of 0.9995 and accepted times drifting apart by 2.25 of 30 time units.
That matters beyond step counts: upstream stops a stopping-criterion run at the
first accepted step past the criterion without root-finding, so the recorded stop
state is quantised to the accepted-step grid.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.field import ToroidalField
from simsopt.field.tracing import compute_fieldlines as native_compute_fieldlines
from simsopt_jax.core.tracing import (
    _ROOT_BRACKET_EPS,
    TRACING_STATUS_STEP_CONTROL_FAILED,
    FieldlineTracingSpec,
    IterStoppingCriterion,
    MaxRStoppingCriterion,
    _dense_output_state,
    _dopri5_adaptive_step,
    _dopri5_stage_step,
    _error_norm,
    trace_fieldline,
)
from simsopt_jax_adapters.field.tracing import compute_fieldlines
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX

# Upstream's ``rootmaxit`` (``legacy native extension/tracing.cpp``), which is
# the default this port carries on every tracing spec.
_MAX_ROOT_ITERATIONS_DOC = FieldlineTracingSpec(
    tmax=1.0, rtol=1.0e-9, atol=1.0e-9, max_steps=1
).max_root_iters

MAJOR_RADIUS = 1.3
FIELD_STRENGTH = 0.8
START_RADIUS = 1.4
HORIZON = 30.0
TOLERANCE = 1.0e-9


def _accepted_times(trajectory) -> np.ndarray:
    return np.asarray(trajectory, dtype=np.float64)[:, 0]


def test_accepted_step_sequence_matches_upstream_on_an_analytic_field() -> None:
    """Same field in both lanes, so only the controller can differ."""
    native_paths, _native_hits = native_compute_fieldlines(
        ToroidalField(MAJOR_RADIUS, FIELD_STRENGTH),
        [START_RADIUS],
        [0.0],
        tmax=HORIZON,
        tol=TOLERANCE,
        phis=[],
        stopping_criteria=[],
    )
    jax_paths, _jax_hits = compute_fieldlines(
        ToroidalFieldJAX(MAJOR_RADIUS, FIELD_STRENGTH),
        [START_RADIUS],
        [0.0],
        tmax=HORIZON,
        tol=TOLERANCE,
        phis=[],
        stopping_criteria=[],
    )
    native_times = _accepted_times(native_paths[0])
    jax_times = _accepted_times(jax_paths[0])

    # Identical controller => identical number of accepted steps.
    assert jax_times.shape == native_times.shape
    assert native_times.shape[0] > 300

    native_steps = np.diff(native_times)
    jax_steps = np.diff(jax_times)
    relative = np.abs(jax_steps - native_steps) / np.abs(native_steps)
    # Only floating-point noise in the two field evaluations survives, amplified
    # by the controller's fifth-root. The old RMS controller reached 0.9995 here.
    assert relative.max() < 1.0e-6
    assert np.max(np.abs(jax_times - native_times)) < 1.0e-7
    # The first steps are decided by the initial 1e-5 * dtmax guess and the
    # capped growth factor alone, so they must agree exactly.
    np.testing.assert_array_equal(jax_steps[:5], native_steps[:5])


def test_growth_factor_is_capped_at_the_upstream_value() -> None:
    """``increase_step`` caps growth at ``0.9 * 5``, not at 5."""
    native_paths, _hits = native_compute_fieldlines(
        ToroidalField(MAJOR_RADIUS, FIELD_STRENGTH),
        [START_RADIUS],
        [0.0],
        tmax=HORIZON,
        tol=TOLERANCE,
        phis=[],
        stopping_criteria=[],
    )
    jax_paths, _jax_hits = compute_fieldlines(
        ToroidalFieldJAX(MAJOR_RADIUS, FIELD_STRENGTH),
        [START_RADIUS],
        [0.0],
        tmax=HORIZON,
        tol=TOLERANCE,
        phis=[],
        stopping_criteria=[],
    )
    jax_steps = np.diff(_accepted_times(jax_paths[0]))
    native_steps = np.diff(_accepted_times(native_paths[0]))
    growth = jax_steps[1:] / jax_steps[:-1]
    assert growth.max() <= 4.5 + 1.0e-12
    np.testing.assert_allclose(growth[0], 4.5, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(
        native_steps[1] / native_steps[0], 4.5, rtol=1.0e-12, atol=0.0
    )


def test_dense_output_is_boosts_dopri5_continuous_extension():
    """``_dense_output_state`` IS ``runge_kutta_dopri5::calc_state``.

    Upstream's event localizer root-finds on the dense-output polynomial of the
    step it just accepted (``legacy native extension/tracing.cpp``:
    ``dense.calc_state(t, temp)`` inside the ``toms748_solve`` root function), so
    the crossing costs no extra right-hand-side evaluation. Three properties pin
    the port of ``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp``:

    * it is exact at both ends (``theta = 0`` and ``theta = 1``), which the
      bracket relies on: ``f(t)`` and ``f(t + h)`` must be the values the
      crossing test already compared;
    * inside the step it is a genuine high-order interpolant of the solution,
      not the O(h) linear one;
    * it needs zero calls to the right-hand side.
    """
    dtype = jnp.float64
    omega = 1.7

    calls = []

    def rhs(_t, y):
        calls.append(1)
        return jnp.asarray([-omega * y[1], omega * y[0], 0.0], dtype=dtype)

    t0 = jnp.asarray(0.0, dtype=dtype)
    y0 = jnp.asarray([1.0, 0.0, 0.0], dtype=dtype)
    h = jnp.asarray(0.05, dtype=dtype)
    k0 = rhs(t0, y0)
    stages = _dopri5_stage_step(rhs, t0, y0, h, k0)
    calls_after_step = len(calls)

    start = _dense_output_state(stages, y0, h, jnp.asarray(0.0, dtype=dtype))
    end = _dense_output_state(stages, y0, h, jnp.asarray(1.0, dtype=dtype))
    np.testing.assert_array_equal(np.asarray(start), np.asarray(y0))
    np.testing.assert_array_equal(np.asarray(end), np.asarray(stages.y_new))
    assert len(calls) == calls_after_step, "dense output must cost no rhs evaluation"

    # Interior accuracy against the closed-form rotation, and against the linear
    # interpolant the polynomial replaces.
    thetas = (0.125, 0.25, 0.5, 0.75, 0.875)

    def interior_error(step: jax.Array) -> float:
        first = rhs(t0, y0)
        step_stages = _dopri5_stage_step(rhs, t0, y0, step, first)
        worst = 0.0
        for theta in thetas:
            state = np.asarray(
                _dense_output_state(
                    step_stages, y0, step, jnp.asarray(theta, dtype=dtype)
                )
            )
            angle = omega * float(step) * theta
            exact = np.asarray([np.cos(angle), np.sin(angle), 0.0])
            worst = max(worst, float(np.max(np.abs(state - exact))))
        return worst

    for theta in thetas:
        theta_arr = jnp.asarray(theta, dtype=dtype)
        interpolated = np.asarray(_dense_output_state(stages, y0, h, theta_arr))
        angle = omega * float(h) * theta
        exact = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        linear = np.asarray(y0) + theta * (np.asarray(stages.y_new) - np.asarray(y0))
        # At least four orders of magnitude better than the linear interpolant a
        # bracket would otherwise be root-finding on.
        assert np.max(np.abs(interpolated - exact)) * 1.0e4 < np.max(
            np.abs(linear - exact)
        )
    assert len(calls) == calls_after_step

    # Halving the step must reduce the interior error by at least 8, i.e. the
    # interpolant is at least third order in the step -- a linear or quadratic
    # interpolant could not.
    coarse = interior_error(h)
    fine = interior_error(h * jnp.asarray(0.5, dtype=dtype))
    assert fine * 8.0 < coarse, (coarse, fine)


def test_root_finder_ceiling_is_upstreams_and_bracket_tolerance_is_the_boost_floor():
    """The ceiling is ``rootmaxit``; the bracket tolerance is the boost floor.

    ``legacy native extension/tracing.cpp`` uses ``uintmax_t rootmaxit = 200``
    and ``eps_tolerance<double> roottol(-int(std::log2(tol)))``, which stores
    ``eps = max(ldexp(1, 1 - bits), 4 * epsilon<double>())``. The port takes the
    ceiling literally and the tolerance at that expression's floor -- see
    ``_ROOT_BRACKET_EPS`` for the measurement behind that. The floor IS the value
    upstream itself gets at the QA case's ``tol = 1e-16``, so for the tightest
    official tracing tolerance the two are the same number.
    """
    assert _MAX_ROOT_ITERATIONS_DOC == 200
    assert _ROOT_BRACKET_EPS == 4.0 * float(np.finfo(np.float64).eps)
    qa_bits = -int(np.log2(1.0e-16))
    assert max(np.ldexp(1.0, 1 - qa_bits), _ROOT_BRACKET_EPS) == _ROOT_BRACKET_EPS
    # ... and it is strictly tighter than what upstream's construction gives at
    # the looser NCSX and particle tolerances, never looser.
    for tolerance in (1.0e-7, 1.0e-9):
        bits = -int(np.log2(tolerance))
        assert _ROOT_BRACKET_EPS < np.ldexp(1.0, 1 - bits)


# --- boost's ``norm_inf`` and what it does with a non-finite error ------------
# ``default_error_checker::error`` reduces the componentwise relative error with
# ``algebra.norm_inf`` (``boost/numeric/odeint/stepper/controlled_runge_kutta.hpp``),
# and for the ``std::array`` states upstream uses
# (``tracing.cpp``: ``using State = std::array<double, Size>``) the
# ``algebra_dispatcher`` selects ``array_algebra``
# (``boost/numeric/odeint/algebra/algebra_dispatcher.hpp``), whose ``norm_inf``
# is ``init = 0; for i: init = max(init, abs(s[i]))``
# (``boost/numeric/odeint/algebra/array_algebra.hpp``). ``std::max(a, b)`` is
# ``(a < b) ? b : a``, so a ``nan`` entry NEVER replaces the running maximum and
# an ``inf`` entry always does.
_NAN = float("nan")
_INF = float("inf")


def _relative_error(components):
    """``_error_norm`` fed an error vector directly: unit scale, no derivative."""
    y_err = jnp.asarray(components, dtype=jnp.float64)
    zeros = jnp.zeros_like(y_err)
    return float(
        _error_norm(
            y_err,
            zeros,
            zeros,
            jnp.asarray(0.0, jnp.float64),
            jnp.asarray(0.0, jnp.float64),
            jnp.asarray(1.0, jnp.float64),
        )
    )


def test_error_norm_is_boosts_norm_inf_including_its_nan_rule():
    """A ``nan`` component contributes nothing; an ``inf`` component dominates.

    This is not a corner case of the port's choosing: the DOPRI5 error estimate
    contains the FSAL stage ``k7 = f(t + h, y_new)``
    (``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp``, the ``do_step_impl``
    overload that fills ``xerr``), so a right-hand side that is ``nan`` at the
    END of a step makes every error component ``nan`` while ``y_new`` itself,
    built from ``k1, k3..k6`` only, stays finite. Upstream then measures error
    ``0`` and ACCEPTS that step. ``jnp.max`` would report ``nan`` instead.
    """
    assert _relative_error([0.25, 0.5, 0.125]) == 0.5
    assert _relative_error([_NAN, 0.5, 0.125]) == 0.5
    assert _relative_error([_NAN, _NAN, _NAN]) == 0.0
    assert _relative_error([_NAN, _INF, 0.5]) == _INF
    assert _relative_error([_INF, 0.5, 0.125]) == _INF


# --- the non-finite-state gate: upstream's acceptance, an honest failure -------
# Synthetic mirror of the shipped particle case (packet
# ``d1-diagnostic/runs/native-tracing-particle/20260920T054912Z-b0209248.partial``,
# particle 54): past ``x = _NAN_BOUNDARY_X`` the field is ``nan``, which is what
# an interpolated field does outside its domain, where ``|B| = 0`` and the
# guiding-centre right-hand side divides by ``|B|``. ``_NAN_BOUNDARY_DTMAX`` caps
# the step so the lane walks up to the boundary instead of clearing it in one
# step clamped to ``tmax``.
_NAN_BOUNDARY_START_X = 1.0
_NAN_BOUNDARY_X = 1.5
_NAN_BOUNDARY_DTMAX = 0.01


def _nan_boundary_field(point):
    """``dx/dt = B``; past ``x = _NAN_BOUNDARY_X`` the field is ``nan``."""
    inside = point[0] <= jnp.asarray(_NAN_BOUNDARY_X, jnp.float64)
    unit = jnp.asarray([1.0, 0.0, 0.0], jnp.float64)
    return jnp.where(inside, unit, jnp.full((3,), _NAN, jnp.float64))


def _trace_nan_boundary(stopping_criteria=()):
    spec = FieldlineTracingSpec(
        tmax=2.0,
        rtol=1.0e-9,
        atol=1.0e-9,
        max_steps=4096,
        dtmax=_NAN_BOUNDARY_DTMAX,
        adaptive_loop="while",
    )
    return trace_fieldline(
        spec,
        jnp.asarray([_NAN_BOUNDARY_START_X, 0.0, 0.0], jnp.float64),
        _nan_boundary_field,
        stopping_criteria=stopping_criteria,
    )


def _published(result):
    """The arrays a lane publishes: trajectory (padding included) and event rows."""
    trajectory = np.asarray(result.trajectory, dtype=np.float64)
    rows = trajectory[np.asarray(result.mask)]
    hits = np.asarray(result.phi_hits, dtype=np.float64)[: int(result.phi_hits_count)]
    return trajectory, rows, hits


def test_a_nan_error_with_a_finite_new_state_is_accepted_like_boost():
    """Upstream's acceptance, kept exactly where it decides the mirror.

    The FSAL stage ``k7 = f(t + h, y_new)`` enters the error estimate and NOT
    ``y_new`` (``boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp``
    ``do_step_impl`` with ``xerr``), so a right-hand side that is ``nan`` at the
    END of a step makes every error component ``nan`` while the new state stays
    finite -- which is what upstream's particle 54 does when its step leaves the
    interpolation domain. ``array_algebra::norm_inf`` folds those ``nan``
    components to ``0`` and ``try_step`` accepts. Constructed exactly, without a
    tuned boundary: the reference step fixes ``y_new``, and the second
    right-hand side is ``nan`` only AT that state, which is the one point the
    FSAL stage evaluates and no other stage does.
    """
    t = jnp.asarray(0.0, jnp.float64)
    h = jnp.asarray(0.25, jnp.float64)
    y = jnp.asarray([1.4, 0.3, 0.0], jnp.float64)
    tolerance = jnp.asarray(TOLERANCE, jnp.float64)

    def finite_rhs(_t, state):
        return jnp.stack([state[0], -state[1], jnp.zeros((), jnp.float64)])

    y_new_reference = _dopri5_stage_step(finite_rhs, t, y, h, finite_rhs(t, y)).y_new

    def end_point_nan_rhs(_t, state):
        value = finite_rhs(_t, state)
        at_end = jnp.all(state == y_new_reference)
        return jnp.where(at_end, jnp.full_like(value, jnp.nan), value)

    k_first = end_point_nan_rhs(t, y)
    step = _dopri5_adaptive_step(
        end_point_nan_rhs,
        t,
        y,
        h,
        k_first,
        jnp.asarray(1.0, jnp.float64),
        jnp.asarray(np.inf, jnp.float64),
        tolerance,
        tolerance,
        jnp.float64,
    )

    assert np.all(np.isnan(np.asarray(step.stages.y_err, dtype=np.float64)))
    assert (
        float(
            _error_norm(
                step.stages.y_err, y, k_first, step.h_clamped, tolerance, tolerance
            )
        )
        == 0.0
    )
    assert bool(step.accepted)
    assert not bool(step.nonfinite_state)
    np.testing.assert_array_equal(
        np.asarray(step.y_next, dtype=np.float64),
        np.asarray(y_new_reference, dtype=np.float64),
    )
    assert float(step.t_next) == float(t + h)


def test_an_accepted_step_onto_a_non_finite_state_ends_the_lane_at_the_last_finite_state():
    """CORRECTED from ``test_a_step_whose_error_is_all_nan_is_accepted_not_stalled``.

    Boost decides acceptance from the error alone, so it also accepts the step
    that crosses the boundary -- whose own state is ``nan`` -- and carries that
    state to ``tmax``. The wave-3 version of this test asserted exactly that
    (``status == 0``, ``t_final == 2.0``) and asserted nothing about finiteness,
    which locked in a lane reporting success with the single recorded row
    ``[2, nan, nan, nan]`` (review finding ``r3-lens2-tracing-1``, reproduced at
    shipped scale on packet particle 54: ``status -1`` with the published hit row
    ``[1.78740353e-04, -1, nan, nan, nan, nan]``). A lane here never reports
    success on, nor publishes, non-finite values, so it ends at the last finite
    state with ``TRACING_STATUS_STEP_CONTROL_FAILED``. Upstream's acceptance is
    unchanged where the new state is finite:
    ``test_a_nan_error_with_a_finite_new_state_is_accepted_like_boost``.
    """
    result = _trace_nan_boundary()
    trajectory, rows, hits = _published(result)

    assert int(result.status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert np.all(np.isfinite(trajectory))
    assert np.all(np.isfinite(hits))
    assert int(result.phi_hits_count) == 0
    # Non-degenerate: the lane really integrated, and it walked all the way up
    # to the boundary (it is not the pre-wave-3 rule stalling short of it).
    assert int(result.steps_taken) > 0
    assert rows.shape[0] == int(result.steps_taken) + 1
    assert _NAN_BOUNDARY_X - rows[-1][1] <= _NAN_BOUNDARY_DTMAX
    # The lane ends AT the last finite accepted state, not one step past it.
    assert rows[-1][1] < _NAN_BOUNDARY_X
    assert float(result.t_final) == rows[-1][0]


def test_a_criterion_just_outside_the_boundary_still_stops_the_lane_upstreams_way():
    """The gate must not steal a stop upstream makes.

    The criterion radius is the last finite accepted state of the criterion-free
    lane, i.e. the criterion sits in the final accepted-step band before the
    ``nan`` boundary -- the closest a criterion can come to it and still be
    evaluated on a finite state, which is where upstream's levelset criterion
    fires on the shipped particle case. The lane must report that criterion's
    own status with a finite hit row.
    """
    baseline_rows = _published(_trace_nan_boundary())[1]
    # Half way between the last two accepted states, so that the ulp of
    # ``sqrt(x * x + y * y)`` cannot move which step satisfies the criterion.
    crit_r = 0.5 * float(baseline_rows[-2][1] + baseline_rows[-1][1])

    result = _trace_nan_boundary(stopping_criteria=(MaxRStoppingCriterion(crit_r),))
    trajectory, _rows, hits = _published(result)

    assert int(result.status) == -1
    assert int(result.phi_hits_count) == 1
    assert np.all(np.isfinite(hits))
    assert np.all(np.isfinite(trajectory))
    assert hits[0][1] == -1.0
    assert crit_r <= hits[0][2] < _NAN_BOUNDARY_X
    assert float(result.t_final) == float(baseline_rows[-1][0])


def test_a_criterion_may_not_fire_on_the_step_that_lands_on_a_non_finite_state():
    """The published row and the status must both refuse the ``nan`` state.

    ``IterStoppingCriterion`` is the one criterion whose predicate does not read
    the state (``legacy native extension/tracing.cpp``
    ``IterationStoppingCriterion``), so it fires on whichever accepted step the
    trial counter selects -- including the one that crosses the boundary. It
    fires on the first trial whose index exceeds ``max_iter``; no trial is
    rejected on this field, so trial ``k`` is the ``k``-th accepted step and the
    two values below are DERIVED from the criterion-free lane rather than
    written down: ``accepted_steps - 1`` selects the last finite step and
    ``accepted_steps`` selects the boundary-crossing one. At HEAD the second
    reported ``status -1`` with the published row ``[0.50480429, -1, nan, nan, nan]``
    (``logs/probe_criterion_on_nan_state.before.log``).
    """
    baseline = _trace_nan_boundary()
    baseline_rows = _published(baseline)[1]
    accepted_steps = int(baseline.steps_taken)

    last_finite = _trace_nan_boundary(
        stopping_criteria=(IterStoppingCriterion(accepted_steps - 1),)
    )
    finite_hits = _published(last_finite)[2]
    assert int(last_finite.status) == -1
    assert int(last_finite.phi_hits_count) == 1
    assert np.all(np.isfinite(finite_hits))
    assert finite_hits[0][2] == baseline_rows[-1][1]

    crossing = _trace_nan_boundary(
        stopping_criteria=(IterStoppingCriterion(accepted_steps),)
    )
    crossing_trajectory, crossing_rows, crossing_hits = _published(crossing)
    assert int(crossing.status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert int(crossing.phi_hits_count) == 0
    assert np.all(np.isfinite(crossing_hits))
    assert np.all(np.isfinite(crossing_trajectory))
    np.testing.assert_array_equal(crossing_rows[-1], baseline_rows[-1])


# --- a plane crossing whose dense output is poisoned ---------------------------------------------------------------
# The other half of the same boost rule: the step with a ``nan`` FSAL stage and a finite new state is ACCEPTED (the
# test above pins that), and the event scan then builds its row from that step's dense output, which carries ``k7``
# with weight ``dt * b7_theta`` and is therefore ``nan`` at every query time.
_POISON_START = jnp.asarray([1.4, 0.0, 0.0], jnp.float64)
_POISON_PHIS = jnp.asarray([0.6], jnp.float64)
_POISON_HORIZON = 4.0
# The poisoned ball has to contain the poisoned run's OWN ``y_new`` -- which differs from the reference run's by XLA
# fusion noise, the two traces being different compiled programs (1.3e-11 here) -- and to exclude every other point
# the integrator evaluates, all of which are at least one accepted-step displacement away. One part per million of
# the smallest accepted-state displacement of the reference trace does both with orders of magnitude to spare, and
# the tests below assert the containment they rely on instead of trusting it.
_POISON_BALL_FRACTION = 1.0e-6


def _poison_spec(tmax: float) -> FieldlineTracingSpec:
    return FieldlineTracingSpec(
        tmax=tmax,
        rtol=TOLERANCE,
        atol=TOLERANCE,
        max_steps=4096,
        adaptive_loop="while",
    )


def _poison_field(point):
    x, y, z = point[0], point[1], point[2]
    scale = MAJOR_RADIUS * FIELD_STRENGTH / (x * x + y * y)
    return jnp.stack([-scale * y, scale * x, jnp.zeros_like(z)])


def _accepted_rows(result) -> np.ndarray:
    trajectory = np.asarray(result.trajectory, dtype=np.float64)
    return trajectory[np.asarray(result.mask)]


def _recorded_hits(result) -> np.ndarray:
    return np.asarray(result.phi_hits, dtype=np.float64)[: int(result.phi_hits_count)]


def _ball_radius(rows: np.ndarray) -> float:
    return _POISON_BALL_FRACTION * float(
        np.min(np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1))
    )


def _trace_with_a_poisoned_end_point(tmax: float, centre: np.ndarray, radius: float):
    """The same trace, with the right-hand side ``nan`` inside a ball around one accepted state."""
    centre_device = jnp.asarray(centre, jnp.float64)
    radius_squared = jnp.asarray(radius * radius, jnp.float64)

    def poisoned_field(point):
        value = _poison_field(point)
        inside = jnp.sum((point - centre_device) ** 2) <= radius_squared
        return jnp.where(inside, jnp.full_like(value, jnp.nan), value)

    return trace_fieldline(
        _poison_spec(tmax), _POISON_START, poisoned_field, phis=_POISON_PHIS
    )


def test_a_plane_crossing_taken_from_a_poisoned_dense_output_is_not_published():
    """No row is recorded from a step whose continuous extension is non-finite, and the status says so.

    At HEAD this lane recorded the row ``[1.1731494369951792, 0.0, nan, nan, nan]``: the crossing test passed on the
    step's two finite angles, the localizer was handed ``nan`` residuals and returned the step's END time, and the
    row came from the dense output. The lane's own status was already ``2`` here (the next trial's FSAL ``k1`` is
    the ``nan`` ``k7``), which is exactly why the row mattered: a failed lane was publishing a hit position.
    """
    reference = trace_fieldline(
        _poison_spec(_POISON_HORIZON), _POISON_START, _poison_field, phis=_POISON_PHIS
    )
    rows = _accepted_rows(reference)
    hits = _recorded_hits(reference)
    assert int(reference.status) == 0
    assert hits.shape[0] == 1
    assert np.all(np.isfinite(hits))

    crossing_step = int(np.searchsorted(rows[:, 0], hits[0][0]))
    # The crossing is inside the trace, so the poisoned step is an ordinary one.
    assert 0 < crossing_step < rows.shape[0] - 1
    radius = _ball_radius(rows)

    result = _trace_with_a_poisoned_end_point(
        _POISON_HORIZON, rows[crossing_step][1:4], radius
    )
    result_rows = _accepted_rows(result)

    assert int(result.status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert int(result.phi_hits_count) == 0
    assert np.all(np.isfinite(np.asarray(result.phi_hits, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(result.trajectory, dtype=np.float64)))
    # It really is the SAME trace up to the poisoned step, and it ends at the last finite accepted state before it.
    assert result_rows.shape[0] == crossing_step
    assert (
        float(np.linalg.norm(result_rows[-1][1:4] - rows[crossing_step - 1][1:4]))
        <= radius
    )


def test_a_poisoned_crossing_on_the_step_that_reaches_tmax_fails_the_lane():
    """The case where dropping the row alone would have left a ``converged`` lane.

    The horizon is placed inside the crossing step, so the poisoned step is the one that reaches ``tmax``: nothing
    after it can report the problem. At HEAD this lane reported status ``0`` -- reached ``tmax``, the healthiest
    status a tracing lane has -- while publishing the all-``nan`` row ``[1.151959333888081, 0.0, nan, nan, nan]``.
    """
    reference = trace_fieldline(
        _poison_spec(_POISON_HORIZON), _POISON_START, _poison_field, phis=_POISON_PHIS
    )
    rows = _accepted_rows(reference)
    crossing_time = float(_recorded_hits(reference)[0][0])
    crossing_step = int(np.searchsorted(rows[:, 0], crossing_time))
    horizon = 0.5 * (crossing_time + float(rows[crossing_step][0]))

    short = trace_fieldline(
        _poison_spec(horizon), _POISON_START, _poison_field, phis=_POISON_PHIS
    )
    short_rows = _accepted_rows(short)
    short_hits = _recorded_hits(short)
    assert int(short.status) == 0
    assert short_hits.shape[0] == 1
    # The crossing is in the LAST step, the one clamped onto ``tmax``.
    assert short_rows[-2][0] < short_hits[0][0] <= short_rows[-1][0]

    radius = _ball_radius(short_rows)
    result = _trace_with_a_poisoned_end_point(horizon, short_rows[-1][1:4], radius)
    result_rows = _accepted_rows(result)

    assert int(result.status) == TRACING_STATUS_STEP_CONTROL_FAILED
    assert int(result.phi_hits_count) == 0
    assert np.all(np.isfinite(np.asarray(result.phi_hits, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(result.trajectory, dtype=np.float64)))
    # The step itself IS taken -- upstream accepts it and its state is finite, and this port mirrors that -- so the
    # lane's time does reach the horizon; what it refuses is the ROW built from that step's dense output, and its
    # status is no longer "reached tmax". The published trajectory therefore ends at the previous accepted state,
    # exactly as upstream's ``res`` does on any stopped run (``tracing.cpp``: the final ``calc_state(tmax)`` push
    # happens only ``if(!stop)``).
    assert float(result.t_final) == horizon
    assert result_rows.shape[0] == short_rows.shape[0] - 1
    assert float(result_rows[-1][0]) < horizon
