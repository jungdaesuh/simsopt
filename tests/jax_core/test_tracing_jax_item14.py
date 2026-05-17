"""Parity tests for the JAX tracing RK path (item 14 fieldline MVP).

This module validates three slices of the new
``simsopt.jax_core.tracing`` and ``simsopt.jax_core.surface_classifier``
modules under the ``event_time_tracing`` parity-ladder lane:

1. ``test_dopri5_step_recovers_analytic_solution`` — single
   DOPRI5 step on the scalar ODE ``dy/dt = y`` matches the closed-form
   ``y(h) = exp(h)`` to within the a-priori 5th-order truncation
   ceiling.
2. ``test_trace_fieldline_matches_upstream_compute_fieldlines_toroidal_axis`` —
   a single fieldline traced through a uniform toroidal field
   ``ToroidalField(R0, B0)`` agrees with the upstream
   ``simsopt.field.tracing.compute_fieldlines`` endpoint at the lane
   state-vector tolerance under the same ``dx/dt = B`` parameterisation.
3. ``test_bracket_root_finds_zero_crossing_within_tolerance`` — the
   ``bracket_root_jax`` Illinois localizer locates the analytic root of
   ``f(t) = t - 0.7`` over ``[0, 1]`` to within the lane
   ``event_time_atol``.

The classifier module receives an end-to-end smoke test against a
hand-built signed-distance interpolant to exercise the JAX evaluation
path without depending on the full ``simsopt.geo.surface.SurfaceClassifier``
construction pipeline.

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.field import tracing as field_tracing_module
from simsopt.jax_core.regular_grid_interp import (
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
)
from simsopt.jax_core.surface_classifier import (
    make_levelset_classifier,
    signed_distance_to_cartesian_classifier,
)
from simsopt.jax_core.tracing import (
    FieldlineTracingSpec,
    FullorbitTracingSpec,
    GuidingCenterTracingSpec,
    IterStoppingCriterion,
    LevelsetStoppingCriterion,
    MaxRStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    _continuous_phi,
    _run_dopri5_4state,
    bracket_root_jax,
    dopri5_step,
    trace_fieldline,
    trace_fieldlines_batched,
    trace_fullorbit,
    trace_fullorbits_batched,
    trace_guiding_center,
    trace_guiding_centers_batched,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


@pytest.fixture(scope="module")
def event_time_lane():
    return dict(_EVENT_TIME_TOLERANCES)


# ---------------------------------------------------------------------------
# 1. Single DOPRI5 step vs analytic exp(h)
# ---------------------------------------------------------------------------


def test_dopri5_step_recovers_analytic_solution():
    """``dy/dt = y`` integrated for one DOPRI5 step matches ``exp(h)`` analytically.

    The Dormand-Prince 5(4) pair has local truncation error
    ``O(h^6)`` on the 5th-order solution; over a single step from
    ``t=0`` to ``t=h=0.1`` the error is bounded by ``C * h^6`` with
    ``C`` an ``O(1)`` constant determined by the Butcher tableau and
    the RHS Lipschitz behaviour. For ``dy/dt = y`` the Jacobian is
    bounded and ``h^6 = 1e-6`` sets the natural ceiling; we use
    ``atol=1e-5`` to leave one order of headroom for the leading
    coefficient. SciPy is intentionally absent — it carries its own
    truncation error and is not a stronger oracle than ``np.exp``.
    """

    # JAX trial step: dy/dt = y, y(0) = 1, h = 0.1.
    def rhs(t, y):
        del t
        return y

    y0 = jnp.asarray([1.0], dtype=jnp.float64)
    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    h = jnp.asarray(0.1, dtype=jnp.float64)
    k0 = rhs(t0, y0)

    y_new_jax, _y_err, _k7 = dopri5_step(rhs, t0, y0, h, k0)
    y_new_jax_np = np.asarray(y_new_jax)

    # Analytic gate: single-step truncation ceiling for DOPRI5 with h=0.1
    # is dominated by ``h^6 = 1e-6``; ``atol=1e-5`` leaves an order of
    # headroom for the leading-coefficient constant.
    y_analytic = np.exp(0.1)
    truncation_ceiling = 1.0e-5
    abs_err = float(np.max(np.abs(y_new_jax_np - y_analytic)))
    assert abs_err <= truncation_ceiling, (
        "DOPRI5 single-step analytic parity failed: "
        f"jax={y_new_jax_np}, analytic={y_analytic}, "
        f"abs_err={abs_err}, truncation_ceiling={truncation_ceiling} "
        "(h^6 ceiling for h=0.1)"
    )


# ---------------------------------------------------------------------------
# 2. Fieldline endpoint vs upstream compute_fieldlines on a toroidal field
# ---------------------------------------------------------------------------


def _toroidal_field_jax(R0: float, B0: float):
    """Return a JAX callable for the upstream ``ToroidalField`` analytic B.

    Upstream definition (``simsopt.field.magneticfieldclasses.ToroidalField``):
    ``B = B0 * R0 / R`` along the toroidal direction, i.e. in Cartesian
    coordinates,

        B_x = -B0 * R0 * sin(phi) / R
        B_y =  B0 * R0 * cos(phi) / R
        B_z =  0,

    with ``R = sqrt(x**2 + y**2)`` and ``phi = atan2(y, x)``. Substituting
    ``sin(phi) = y / R`` and ``cos(phi) = x / R`` gives

        B_x = -B0 * R0 * y / R**2
        B_y =  B0 * R0 * x / R**2
        B_z =  0.

    """

    R0_arr = jnp.asarray(R0, dtype=jnp.float64)
    B0_arr = jnp.asarray(B0, dtype=jnp.float64)

    def field_fn(point: jax.Array) -> jax.Array:
        x = point[0]
        y = point[1]
        r2 = x * x + y * y
        coeff = B0_arr * R0_arr / r2
        return jnp.stack([-coeff * y, coeff * x, jnp.asarray(0.0, dtype=jnp.float64)])

    return field_fn


def test_trace_fieldline_matches_upstream_compute_fieldlines_toroidal_axis(
    event_time_lane,
):
    """Fieldline endpoint in ``ToroidalField`` matches upstream within the lane.

    A purely toroidal field has the analytic streamline ``(R, Z) =
    (R_init, 0)``. The JAX route uses the same ``dx/dt = B`` time
    parameterisation as upstream ``compute_fieldlines``.
    """

    from simsopt.field.magneticfieldclasses import ToroidalField
    from simsopt.field.tracing import compute_fieldlines

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    R0 = 1.3
    B0 = 0.8
    R_init = 1.1
    # Upstream parametrisation: ``dx/dt = B``, so traversal time per
    # toroidal revolution is ``2 pi R / |B| = 2 pi R^2 / (B0 R0)``. We
    # pick a tmax_cpp that yields a partial revolution (~ 0.6 turn) to
    # stay well away from full-circle aliasing at the comparison plane.
    tmax_cpp = 0.6 * (2.0 * np.pi * R_init * R_init) / (B0 * R0)

    # Upstream oracle: tight tol so the run is dominated by the RK4(5)
    # method and not by event localisation noise.
    res_tys, _res_phi_hits = compute_fieldlines(
        ToroidalField(R0, B0),
        [R_init],
        [0.0],
        tmax=tmax_cpp,
        tol=1e-10,
        phis=[],
        stopping_criteria=[],
    )
    upstream_endpoint = np.asarray(res_tys[0][-1, 1:4])

    spec = FieldlineTracingSpec(
        tmax=float(tmax_cpp),
        rtol=1e-10,
        atol=1e-12,
        max_steps=2000,
    )
    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)
    field_fn = _toroidal_field_jax(R0, B0)

    result = trace_fieldline(spec, y0, field_fn)
    jax_endpoint = np.asarray(result.trajectory[int(result.steps_taken), 1:4])

    assert int(result.status) == 0, (
        "JAX integrator failed to reach tmax: "
        f"status={int(result.status)}, t_final={float(result.t_final)}, "
        f"tmax={float(tmax_cpp)}, steps_taken={int(result.steps_taken)}"
    )

    assert np.allclose(
        jax_endpoint, upstream_endpoint, rtol=state_rtol, atol=state_atol
    ), (
        "JAX vs upstream fieldline endpoint parity failed: "
        f"jax={jax_endpoint}, upstream={upstream_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


def test_trace_fieldline_uses_upstream_raw_B_parameterization():
    """Non-unit field magnitude advances at raw ``B`` speed, not unit speed."""

    def field_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([0.0, 2.0, 0.0], dtype=jnp.float64)

    spec = FieldlineTracingSpec(tmax=1.0, rtol=1e-12, atol=1e-12, max_steps=64)
    y0 = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
    result = trace_fieldline(spec, y0, field_fn)
    endpoint = np.asarray(result.trajectory[int(result.steps_taken), 1:4])

    assert int(result.status) == 0
    np.testing.assert_allclose(endpoint, np.array([0.0, 2.0, 0.0]), rtol=0, atol=1e-12)


def _assert_accepted_steps_respect_dtmax(result, dtmax, initial_fraction):
    live = np.asarray(result.trajectory)[np.asarray(result.mask)]
    steps = np.diff(live[:, 0])

    assert int(result.status) == 0
    assert steps.size > 3
    np.testing.assert_allclose(
        steps[0],
        initial_fraction * dtmax,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.max(steps) <= dtmax + 1.0e-14
    assert np.any(np.isclose(steps, dtmax, rtol=0.0, atol=1.0e-14))


def test_toroidal_transit_baseline_is_first_accepted_state():
    """Transit criteria anchor at the first post-step angle, matching C++."""

    def rotating_field(point: jax.Array) -> jax.Array:
        return jnp.asarray([-point[1], point[0], 0.0], dtype=jnp.float64)

    spec = FieldlineTracingSpec(
        tmax=1.0e-3,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=32,
        max_phi_hits=4,
        dtmax=1.0e-2,
    )
    result = trace_fieldline(
        spec,
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        rotating_field,
        stopping_criteria=(ToroidalTransitStoppingCriterion(max_transits=1.0e-8),),
    )

    assert int(result.status) == -1
    assert int(result.steps_taken) >= 1
    assert int(result.phi_hits_count) == 1
    first_recorded_step_t = float(result.trajectory[1, 0])
    stop_event_t = float(result.phi_hits[0, 0])
    assert stop_event_t > first_recorded_step_t


def test_trace_fieldline_clean_exit_reports_exact_tmax():
    """Terminal-clamped non-stopped trajectories report the exact requested tmax."""

    def field_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)

    spec = FieldlineTracingSpec(
        tmax=float(np.nextafter(0.3, 1.0)),
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=128,
        dtmax=0.07,
    )
    result = trace_fieldline(
        spec,
        jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        field_fn,
    )

    assert int(result.status) == 0
    assert float(result.t_final) == spec.tmax
    live = np.asarray(result.trajectory)[np.asarray(result.mask)]
    assert live[-1, 0] == spec.tmax


@pytest.mark.parametrize(
    ("x", "y", "phi_near"),
    [
        (-1.0, 0.0, np.pi),
        (-1.0, -0.0, np.pi),
        (0.0, 1.0, -np.pi),
        (0.0, -1.0, np.pi),
        (1.0, 0.0, np.pi),
        (1.0, -0.0, -np.pi),
        (-1.0, np.nextafter(0.0, 1.0), np.pi),
        (-1.0, np.nextafter(0.0, -1.0), -np.pi),
    ],
)
def test_continuous_phi_matches_cpp_get_phi_edges(x, y, phi_near):
    """JAX unwrap matches the C++ ``get_phi`` edge/tie convention."""
    sopp = pytest.importorskip("simsoptpp")

    actual = float(
        _continuous_phi(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(phi_near, dtype=jnp.float64),
            jnp.float64,
        )
    )
    expected = sopp.get_phi(float(x), float(y), float(phi_near))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_public_tracing_wrappers_use_cpp_quarter_turn_dtmax_formulas():
    """JAX wrapper ``dtmax`` formulas match ``simsoptpp/tracing.cpp``."""

    xyz = np.asarray([3.0, 4.0, 2.0], dtype=np.float64)
    speed_total = 2.5
    abs_B = 0.8
    G0 = -1.7
    modB = 0.4

    np.testing.assert_allclose(
        field_tracing_module._cartesian_particle_dtmax(xyz, speed_total),
        5.0 * 0.5 * np.pi / speed_total,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        field_tracing_module._fieldline_dtmax(xyz, abs_B),
        5.0 * 0.5 * np.pi / abs_B,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        field_tracing_module._boozer_particle_dtmax(G0, modB, speed_total),
        (abs(G0) / modB) * 0.5 * np.pi / speed_total,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        field_tracing_module._magnetic_moments(1.0, [2.0], [0.5]),
        [-3.0],
        rtol=0.0,
        atol=0.0,
    )


def test_trace_fieldline_respects_dtmax_step_ceiling_and_initial_heuristic():
    """Fieldline DOPRI5 uses ``1e-5 * dtmax`` and caps accepted steps."""

    def field_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

    dtmax = 0.125
    spec = FieldlineTracingSpec(
        tmax=0.45,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=64,
        dtmax=dtmax,
    )
    y0 = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
    result = trace_fieldline(spec, y0, field_fn)

    _assert_accepted_steps_respect_dtmax(result, dtmax, initial_fraction=1.0e-5)


def test_trace_guiding_center_respects_dtmax_step_ceiling_and_initial_heuristic():
    """Cartesian guiding-centre DOPRI5 uses ``1e-3 * dtmax`` and caps steps."""

    def magnetic_field_fn(_point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            jnp.zeros((3, 3), dtype=jnp.float64),
        )

    dtmax = 0.125
    spec = GuidingCenterTracingSpec(
        tmax=0.45,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=64,
        dtmax=dtmax,
    )
    y0 = jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    result = trace_guiding_center(spec, y0, magnetic_field_fn, m=1.0, q=1.0, mu=0.0)

    _assert_accepted_steps_respect_dtmax(result, dtmax, initial_fraction=1.0e-3)


def test_trace_fullorbit_respects_dtmax_step_ceiling_and_initial_heuristic():
    """Full-orbit DOPRI5 uses ``1e-3 * dtmax`` and caps accepted steps."""

    def magnetic_field_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)

    dtmax = 0.125
    spec = FullorbitTracingSpec(
        tmax=0.45,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=64,
        dtmax=dtmax,
    )
    y0 = jnp.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    result = trace_fullorbit(spec, y0, magnetic_field_fn, m=1.0, q=1.0)

    _assert_accepted_steps_respect_dtmax(result, dtmax, initial_fraction=1.0e-3)


def test_trace_guiding_center_cartesian_accepts_nonpositive_x_initial_state():
    """Cartesian GC must not apply the Boozer ``s > 0`` axis contract."""

    def magnetic_field_fn(_point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            jnp.zeros((3, 3), dtype=jnp.float64),
        )

    dtmax = 0.01
    spec = GuidingCenterTracingSpec(
        tmax=0.1,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=64,
        dtmax=dtmax,
    )
    y0 = jnp.asarray([-1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    result = trace_guiding_center(spec, y0, magnetic_field_fn, m=1.0, q=1.0, mu=0.0)
    live = np.asarray(result.trajectory)[np.asarray(result.mask)]

    assert int(result.status) == 0
    np.testing.assert_allclose(live[1, 0], 1.0e-3 * dtmax, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(live[-1], [0.1, -1.0, 0.0, 0.1, 1.0], atol=1.0e-12)


def test_tracing_batch_helpers_match_single_trajectory_contracts():
    """Vmapped tracing helpers preserve single-lane physics and result layout."""

    def fieldline_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)

    def particle_field_fn(_point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
            jnp.zeros((3, 3), dtype=jnp.float64),
        )

    def zero_B_fn(_point: jax.Array) -> jax.Array:
        return jnp.zeros((3,), dtype=jnp.float64)

    fieldline_spec = FieldlineTracingSpec(
        tmax=0.1, rtol=1.0e-12, atol=1.0e-12, max_steps=64
    )
    fieldline_batch = trace_fieldlines_batched(
        fieldline_spec,
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]], dtype=jnp.float64),
        jnp.asarray([0.01, 0.02], dtype=jnp.float64),
        fieldline_fn,
    )
    fieldline_traj = np.asarray(fieldline_batch.trajectory)
    fieldline_mask = np.asarray(fieldline_batch.mask)
    np.testing.assert_allclose(
        fieldline_traj[0][fieldline_mask[0]][-1],
        [0.1, 0.0, 0.1, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fieldline_traj[1][fieldline_mask[1]][-1],
        [0.1, 1.0, 0.1, 0.5],
        atol=1.0e-12,
    )

    gc_spec = GuidingCenterTracingSpec(
        tmax=0.1, rtol=1.0e-12, atol=1.0e-12, max_steps=64
    )
    gc_batch = trace_guiding_centers_batched(
        gc_spec,
        jnp.asarray([[1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0]], dtype=jnp.float64),
        jnp.asarray([0.01, 0.01], dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        particle_field_fn,
        m=1.0,
        q=1.0,
    )
    gc_traj = np.asarray(gc_batch.trajectory)
    gc_mask = np.asarray(gc_batch.mask)
    np.testing.assert_allclose(
        gc_traj[0][gc_mask[0]][-1],
        [0.1, 1.0, 0.0, 0.1, 1.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        gc_traj[1][gc_mask[1]][-1],
        [0.1, -1.0, 0.0, 0.1, 1.0],
        atol=1.0e-12,
    )

    fullorbit_spec = FullorbitTracingSpec(
        tmax=0.1, rtol=1.0e-12, atol=1.0e-12, max_steps=64
    )
    fullorbit_batch = trace_fullorbits_batched(
        fullorbit_spec,
        jnp.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.5, 0.0, 2.0, 0.0],
            ],
            dtype=jnp.float64,
        ),
        jnp.asarray([0.01, 0.01], dtype=jnp.float64),
        zero_B_fn,
        m=1.0,
        q=1.0,
    )
    fullorbit_traj = np.asarray(fullorbit_batch.trajectory)
    fullorbit_mask = np.asarray(fullorbit_batch.mask)
    np.testing.assert_allclose(
        fullorbit_traj[0][fullorbit_mask[0]][-1],
        [0.1, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fullorbit_traj[1][fullorbit_mask[1]][-1],
        [0.1, 1.0, 0.2, 0.5, 0.0, 2.0, 0.0],
        atol=1.0e-12,
    )


# ---------------------------------------------------------------------------
# 3. Bracketed root vs analytic zero
# ---------------------------------------------------------------------------


def test_bracket_root_finds_zero_crossing_within_tolerance(event_time_lane):
    """``bracket_root_jax`` on ``f(t) = t - 0.7`` over ``[0, 1]`` returns 0.7."""

    event_rtol = float(event_time_lane["event_time_rtol"])
    event_atol = float(event_time_lane["event_time_atol"])

    def f(t):
        return t - jnp.asarray(0.7, dtype=jnp.float64)

    t_left = jnp.asarray(0.0, dtype=jnp.float64)
    t_right = jnp.asarray(1.0, dtype=jnp.float64)
    fl = f(t_left)
    fr = f(t_right)

    t_star, f_star, bracketed = bracket_root_jax(
        f,
        t_left,
        t_right,
        fl,
        fr,
        max_iters=60,
        atol=jnp.asarray(0.0, dtype=jnp.float64),
    )

    assert bool(bracketed), "bracket_root_jax failed to recognise the sign change"
    # Lane gate: event-time accuracy.
    assert np.isclose(float(t_star), 0.7, rtol=event_rtol, atol=event_atol), (
        f"event-time parity failed: t_star={float(t_star)}, lane_atol={event_atol}"
    )
    # Residual sanity: the localizer should drive the returned residual
    # inside the event-time lane.
    assert abs(float(f_star)) <= event_atol, (
        f"event-time residual exceeds lane: |f(t_star)|={abs(float(f_star))}, "
        f"lane_atol={event_atol}"
    )


def test_bracket_root_uses_false_position_candidate_for_linear_residual():
    """A single Illinois false-position iteration solves a linear residual."""

    def f(t):
        return t - jnp.asarray(0.7, dtype=jnp.float64)

    t_left = jnp.asarray(0.0, dtype=jnp.float64)
    t_right = jnp.asarray(1.0, dtype=jnp.float64)
    t_star, f_star, bracketed = bracket_root_jax(
        f,
        t_left,
        t_right,
        f(t_left),
        f(t_right),
        max_iters=1,
        atol=jnp.asarray(0.0, dtype=jnp.float64),
    )

    assert bool(bracketed)
    np.testing.assert_allclose(float(t_star), 0.7, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(float(f_star), 0.0, rtol=0.0, atol=1e-15)


def test_bracket_root_uses_false_position_for_tiny_endpoint_residuals():
    """Tiny endpoint residuals still follow the Illinois false-position update."""

    def f(t):
        del t
        return jnp.asarray(0.0, dtype=jnp.float64)

    t_left = jnp.asarray(0.0, dtype=jnp.float64)
    t_right = jnp.asarray(1.0, dtype=jnp.float64)
    t_star, f_star, bracketed = bracket_root_jax(
        f,
        t_left,
        t_right,
        jnp.asarray(-1.0e-301, dtype=jnp.float64),
        jnp.asarray(8.0e-301, dtype=jnp.float64),
        max_iters=1,
        atol=jnp.asarray(0.0, dtype=jnp.float64),
    )

    assert bool(bracketed)
    np.testing.assert_allclose(float(t_star), 1.0 / 9.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(float(f_star), 0.0, rtol=0.0, atol=0.0)


def test_boozer_axis_status_ignores_rejected_trial_steps():
    """Rejected trial states must shrink; only accepted Boozer states can stop."""

    def rhs(_t, y):
        dsdt = jnp.where(y[0] > 5.0e-7, -1000.0, 1000.0)
        return jnp.asarray([dsdt, 0.0, 0.0, 0.0], dtype=jnp.float64)

    result = _run_dopri5_4state(
        rhs,
        jnp.asarray([1.0e-6, 0.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.asarray(1.0e-3, dtype=jnp.float64),
        jnp.asarray(1.0e-12, dtype=jnp.float64),
        jnp.asarray(1.0e-12, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        1000,
    )

    assert float(result.t_final) > 0.0
    assert int(result.steps_taken) > 0


def test_bracket_root_sorts_descending_input_bracket():
    """Descending endpoints are normalized before the Illinois loop."""

    def f(t):
        return t - jnp.asarray(0.25, dtype=jnp.float64)

    t_left = jnp.asarray(1.0, dtype=jnp.float64)
    t_right = jnp.asarray(0.0, dtype=jnp.float64)
    t_star, f_star, bracketed = bracket_root_jax(
        f,
        t_left,
        t_right,
        f(t_left),
        f(t_right),
        max_iters=60,
        atol=jnp.asarray(0.0, dtype=jnp.float64),
    )

    assert bool(bracketed)
    np.testing.assert_allclose(float(t_star), 0.25, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(float(f_star), 0.0, rtol=0.0, atol=1e-15)


def test_bracket_root_returns_false_when_no_sign_change(event_time_lane):
    """When the bracket does not contain a sign change, ``bracketed`` is ``False``.

    The lane does not gate the returned ``t_star`` in this case — the
    contract is purely on the boolean carrier.
    """

    del event_time_lane

    def f(t):
        return t * t + jnp.asarray(1.0, dtype=jnp.float64)

    t_left = jnp.asarray(0.0, dtype=jnp.float64)
    t_right = jnp.asarray(1.0, dtype=jnp.float64)
    fl = f(t_left)
    fr = f(t_right)

    _t_star, _f_star, bracketed = bracket_root_jax(
        f,
        t_left,
        t_right,
        fl,
        fr,
        max_iters=10,
        atol=jnp.asarray(0.0, dtype=jnp.float64),
    )

    assert not bool(bracketed), (
        "bracket_root_jax must report bracketed=False when no sign change is present"
    )


def test_bracket_root_keeps_equal_residual_no_bracket_result_finite():
    """Equal endpoint residuals are not allowed to manufacture a nonfinite candidate."""

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)

    def f(t):
        return jnp.where(jnp.isfinite(t), one, zero)

    t_star, f_star, bracketed = bracket_root_jax(
        f,
        zero,
        one,
        one,
        one,
        max_iters=10,
        atol=zero,
    )

    assert not bool(bracketed)
    assert np.isfinite(float(t_star))
    np.testing.assert_allclose(float(f_star), 1.0, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 4. Surface classifier smoke test
# ---------------------------------------------------------------------------


def _build_signed_distance_torus_interpolant(R0=1.3, a=0.2):
    """Build a 1-channel signed-distance interpolant for a perfect torus.

    Signed distance to the circular cross-section at major radius ``R0``
    and minor radius ``a`` is ``a - sqrt((R - R0)**2 + z**2)``. Positive
    inside the torus, negative outside.
    """

    def fbatch(rs, _phis, zs):
        return (np.asarray(a) - np.sqrt(np.square(rs - R0) + np.square(zs))).astype(
            np.float64
        )

    rule = UniformInterpolationRule(2)
    spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=(R0 - 0.4, R0 + 0.4, 12),
        yrange=(0.0, 2.0 * np.pi, 12),
        zrange=(-0.4, 0.4, 12),
        value_size=1,
        f=fbatch,
        out_of_bounds_ok=True,
    )
    return spec


def test_levelset_classifier_marks_axis_inside_and_far_point_outside():
    """The classifier returns ``+1`` near the magnetic axis and ``-1`` far outside.

    This is a structural smoke test on the JAX surface-classifier path;
    it asserts the basic inside/outside contract that the public
    ``LevelsetStoppingCriterion`` bridge enforces in the tracing loop.
    """

    R0 = 1.3
    a = 0.2
    interp_spec = _build_signed_distance_torus_interpolant(R0=R0, a=a)
    classify = make_levelset_classifier(interp_spec)

    inside_point = jnp.asarray([R0, 0.0, 0.0], dtype=jnp.float64)
    outside_point = jnp.asarray([R0 + a + 0.1, 0.0, 0.0], dtype=jnp.float64)
    outside_cuboid_point = jnp.asarray([R0 + 1.0, 0.0, 0.0], dtype=jnp.float64)

    inside_val = float(classify(inside_point))
    outside_val = float(classify(outside_point))
    outside_cuboid_val = float(classify(outside_cuboid_point))

    assert inside_val == 1.0, f"axis point classified as {inside_val}, expected +1"
    assert outside_val == -1.0, (
        f"point outside torus classified as {outside_val}, expected -1"
    )
    assert outside_cuboid_val == -1.0, (
        f"point outside classifier cuboid classified as {outside_cuboid_val}, "
        "expected -1"
    )

    # Batched evaluation: vector input must broadcast correctly.
    batch = jnp.stack([inside_point, outside_point], axis=0)
    batched = classify(batch)
    assert tuple(batched.shape) == (2,)
    assert float(batched[0]) == 1.0
    assert float(batched[1]) == -1.0

    # Same callable via the lower-level entry point — should agree.
    classify_direct = signed_distance_to_cartesian_classifier(interp_spec)
    assert float(classify_direct(inside_point)) == 1.0
    assert float(classify_direct(outside_point)) == -1.0


def test_levelset_classifier_grid_faces_remain_classified():
    """Points exactly on interpolation cuboid faces do not route to OOB."""

    interp_spec = _build_signed_distance_torus_interpolant(R0=1.3, a=0.2)
    classify = make_levelset_classifier(interp_spec)

    rmax_face = jnp.asarray([interp_spec.xmax, 0.0, 0.0], dtype=jnp.float64)
    rmin_face = jnp.asarray([interp_spec.xmin, 0.0, 0.0], dtype=jnp.float64)
    zmax_face = jnp.asarray([1.3, 0.0, interp_spec.zmax], dtype=jnp.float64)

    assert float(classify(rmax_face)) == -1.0
    assert float(classify(rmin_face)) == -1.0
    assert float(classify(zmax_face)) == -1.0


def test_trace_guiding_center_stops_after_exiting_classifier_cuboid_face():
    """An active particle trace stops after crossing the interpolation cuboid face."""

    def fbatch(rs, _phis, _zs):
        return np.ones_like(np.asarray(rs, dtype=np.float64))

    interp_spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(2),
        xrange=(1.0, 1.2, 8),
        yrange=(0.0, 2.0 * np.pi, 8),
        zrange=(-0.1, 0.1, 8),
        value_size=1,
        f=fbatch,
        out_of_bounds_ok=True,
    )
    classify = make_levelset_classifier(interp_spec)

    face_point = jnp.asarray([interp_spec.xmax, 0.0, 0.0], dtype=jnp.float64)
    assert float(classify(face_point)) == 1.0

    def field_fn(_point: jax.Array):
        return (
            jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
            jnp.zeros((3, 3), dtype=jnp.float64),
        )

    spec = GuidingCenterTracingSpec(
        tmax=0.3,
        dtmax=0.02,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=64,
        max_phi_hits=4,
    )
    initial_state = jnp.asarray([1.1, 0.0, 0.0, 1.0], dtype=jnp.float64)
    result = trace_guiding_center(
        spec,
        initial_state,
        field_fn,
        m=1.0,
        q=1.0,
        mu=0.0,
        stopping_criteria=(LevelsetStoppingCriterion(classifier_fn=classify),),
    )

    trajectory = np.asarray(result.trajectory)
    live = trajectory[np.asarray(result.mask)]
    hit_rows = np.asarray(result.phi_hits)[: int(result.phi_hits_count)]

    assert int(result.status) == -1
    assert float(result.t_final) < spec.tmax
    assert hit_rows.shape == (1, 6)
    assert int(hit_rows[0, 1]) == -1
    assert live.shape[0] > 1
    assert np.all(live[:, 1] <= interp_spec.xmax)
    assert hit_rows[0, 2] > interp_spec.xmax


def test_levelset_classifier_cpu_jax_phi_wraparound_boundary_parity():
    """CPU and JAX classifiers agree near the ``0 == 2*pi`` phi boundary."""

    from simsopt.geo.surface import SurfaceClassifier
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier

    surf = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=0,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.2)
    surf.set_zs(1, 0, 0.2)
    classifier = SurfaceClassifier(surf, h=0.1, p=2)
    classify_jax = classifier.to_jax_classifier_fn()

    radius = 1.0
    phis = np.asarray(
        [
            0.0,
            np.nextafter(0.0, 1.0),
            -np.nextafter(0.0, 1.0),
            np.nextafter(2.0 * np.pi, 0.0),
        ],
        dtype=np.float64,
    )
    xyz = np.column_stack(
        [
            radius * np.cos(phis),
            radius * np.sin(phis),
            np.zeros_like(phis),
        ]
    )

    cpu_sign = np.sign(classifier.evaluate_xyz(xyz).reshape(-1))
    jax_sign = np.asarray(classify_jax(jnp.asarray(xyz, dtype=jnp.float64)))
    np.testing.assert_array_equal(jax_sign, cpu_sign)


def test_trace_fieldline_first_stopping_criterion_wins_same_step():
    """If multiple criteria fire together, the first criterion index wins."""

    def field_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

    spec = FieldlineTracingSpec(
        tmax=0.1,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=32,
        max_phi_hits=4,
        dtmax=1.0,
    )
    result = trace_fieldline(
        spec,
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        field_fn,
        stopping_criteria=(
            MaxRStoppingCriterion(crit_r=1.0),
            IterStoppingCriterion(max_iter=0),
        ),
    )

    assert int(result.status) == -1
    assert int(result.phi_hits_count) == 1
    np.testing.assert_allclose(float(result.phi_hits[0, 1]), -1.0, rtol=0.0, atol=0.0)


def test_trace_fieldline_levelset_zero_does_not_stop():
    """A levelset value of exactly zero matches the upstream ``f < 0`` predicate."""

    def field_fn(_point: jax.Array) -> jax.Array:
        return jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

    def zero_classifier(points: jax.Array) -> jax.Array:
        return jnp.zeros((points.shape[0],), dtype=points.dtype)

    spec = FieldlineTracingSpec(
        tmax=0.1,
        rtol=1.0e-12,
        atol=1.0e-12,
        max_steps=32,
        max_phi_hits=4,
        dtmax=1.0,
    )
    result = trace_fieldline(
        spec,
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        field_fn,
        stopping_criteria=(LevelsetStoppingCriterion(classifier_fn=zero_classifier),),
    )

    assert int(result.status) == 0
    assert int(result.phi_hits_count) == 0
    assert float(result.t_final) == spec.tmax


def test_levelset_classifier_rejects_out_of_bounds_unsafe_spec():
    """``make_levelset_classifier`` rejects an OOB-strict interpolant.

    When ``out_of_bounds_ok=False`` the underlying ``evaluate_batch``
    routes outside-domain queries to ``NaN``; that breaks the
    ``+1 / -1`` contract assumed by the classifier. The factory must
    surface this as ``ValueError`` at construction time rather than
    silently propagating NaNs into the tracing loop.
    """

    def fbatch(rs, _phis, zs):
        return (0.1 - np.sqrt(np.square(rs - 1.3) + np.square(zs))).astype(np.float64)

    rule = UniformInterpolationRule(2)
    strict_spec = build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=(0.9, 1.7, 8),
        yrange=(0.0, 2.0 * np.pi, 8),
        zrange=(-0.4, 0.4, 8),
        value_size=1,
        f=fbatch,
        out_of_bounds_ok=False,
    )
    with pytest.raises(ValueError):
        make_levelset_classifier(strict_spec)


# ---------------------------------------------------------------------------
# 5. Step-count budget vs upstream (lane-gated)
# ---------------------------------------------------------------------------


def test_trace_fieldline_step_count_within_lane_max_ratio(event_time_lane):
    """The JAX accepted-step count must stay below the lane's ratio bound.

    The lane reports ``step_count_max_ratio=1.25``: the JAX integrator
    is allowed up to 25% more accepted steps than the upstream solver
    on the same fixture. This test runs a fieldline that stays at
    constant R in a toroidal field, so both integrators should stay
    comfortably inside the lane budget.
    """

    from simsopt.field.magneticfieldclasses import ToroidalField
    from simsopt.field.tracing import compute_fieldlines

    max_ratio = float(event_time_lane["step_count_max_ratio"])

    R0 = 1.0
    B0 = 1.0
    R_init = 1.2
    tmax_cpp = 0.4 * (2.0 * np.pi * R_init * R_init) / (B0 * R0)

    res_tys, _ = compute_fieldlines(
        ToroidalField(R0, B0),
        [R_init],
        [0.0],
        tmax=tmax_cpp,
        tol=1e-9,
        phis=[],
        stopping_criteria=[],
    )
    upstream_steps = int(res_tys[0].shape[0])

    spec = FieldlineTracingSpec(
        tmax=float(tmax_cpp),
        rtol=1e-9,
        atol=1e-11,
        max_steps=2000,
    )
    field_fn = _toroidal_field_jax(R0, B0)
    result = trace_fieldline(
        spec, jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64), field_fn
    )

    assert int(result.status) == 0
    jax_steps = int(result.steps_taken)
    ratio = (jax_steps + 1) / max(upstream_steps, 1)
    assert ratio <= max_ratio, (
        f"JAX step-count ratio {ratio:.3f} exceeds lane bound {max_ratio:.3f} "
        f"(jax={jax_steps}, upstream={upstream_steps})"
    )


# ---------------------------------------------------------------------------
# 6. Full trace under JIT vs analytic toroidal-field arc
# ---------------------------------------------------------------------------


def test_trace_fieldline_jit_matches_toroidal_field_closed_form():
    """JIT-compiled ``trace_fieldline`` matches the analytic toroidal-field arc.

    A pure toroidal field ``B = B0 * R0 / R`` advects a fieldline along
    a circle at constant ``R`` and ``z=0``. With ``dx/dt = B`` the
    angular rate is ``omega = |B| / R = B0 * R0 / R^2`` (constant on
    the streamline), so the closed-form endpoint after time ``t`` from
    ``(R, 0, 0)`` is

        (R * cos(omega t), R * sin(omega t), 0).

    This is the value oracle for the JIT path: parity is enforced
    against the closed-form solution, not against a peer integrator.
    """

    R0 = 1.0
    B0 = 1.0
    R_init = 1.05
    field_fn = _toroidal_field_jax(R0, B0)

    spec = FieldlineTracingSpec(tmax=0.5, rtol=1e-10, atol=1e-12, max_steps=400)

    @jax.jit
    def go(y0):
        return trace_fieldline(spec, y0, field_fn)

    y0 = jnp.asarray([R_init, 0.0, 0.0], dtype=jnp.float64)
    result = go(y0)

    assert int(result.status) == 0
    assert int(result.steps_taken) > 0

    endpoint = np.asarray(result.trajectory[int(result.steps_taken), 1:4])
    omega = B0 * R0 / (R_init * R_init)
    theta = omega * spec.tmax
    analytic_endpoint = np.array([R_init * np.cos(theta), R_init * np.sin(theta), 0.0])
    # The integrator targets rtol=1e-10 over a finite arc; leave two
    # orders of headroom for accumulated controller noise.
    np.testing.assert_allclose(endpoint, analytic_endpoint, rtol=1e-8, atol=1e-10)
