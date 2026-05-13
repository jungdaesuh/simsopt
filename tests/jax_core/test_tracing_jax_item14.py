"""Parity tests for the JAX tracing RK path (item 14 fieldline MVP).

This module validates three slices of the new
``simsopt.jax_core.tracing`` and ``simsopt.jax_core.surface_classifier``
modules under the ``event_time_tracing`` parity-ladder lane:

1. ``test_dopri5_step_recovers_analytic_solution_against_scipy`` — single
   DOPRI5 step on the scalar ODE ``dy/dt = y`` matches the
   SciPy ``solve_ivp(method='RK45')`` reference at the same step size
   to within the lane state-vector tolerance.
2. ``test_trace_fieldline_matches_upstream_compute_fieldlines_toroidal_axis`` —
   a single fieldline traced through a uniform toroidal field
   ``ToroidalField(R0, B0)`` agrees with the upstream
   ``simsopt.field.tracing.compute_fieldlines`` endpoint at the lane
   state-vector tolerance under the same ``dx/dt = B`` parameterisation.
3. ``test_bracket_root_finds_zero_crossing_within_tolerance`` — the
   ``bracket_root_jax`` bisection locates the analytic root of
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
    bracket_root_jax,
    dopri5_step,
    trace_fieldline,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


@pytest.fixture(scope="module")
def event_time_lane():
    return dict(_EVENT_TIME_TOLERANCES)


# ---------------------------------------------------------------------------
# 1. Single DOPRI5 step vs SciPy
# ---------------------------------------------------------------------------


def test_dopri5_step_recovers_analytic_solution_against_scipy(event_time_lane):
    """``dy/dt = y`` integrated for one DOPRI5 step matches SciPy at the same step.

    SciPy's ``solve_ivp(method='RK45')`` uses the same Dormand-Prince
    Butcher tableau as this port. With matched initial step size and
    tolerances set so the controller does not subdivide, the two
    integrators must produce identical 5th-order outputs to within the
    event-time state-vector lane.
    """

    from scipy.integrate import solve_ivp

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

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

    # SciPy reference run: force a single step by setting max_step = h and
    # disabling adaptive subdivision with tight tolerances. SciPy will
    # still adaptively select a step but with such loose tolerances and a
    # max_step ceiling we recover one step of the embedded RK4(5) pair.
    sol = solve_ivp(
        rhs,
        (0.0, 0.1),
        np.asarray(y0),
        method="RK45",
        rtol=1e-3,
        atol=1e-6,
        first_step=0.1,
        max_step=0.1,
        dense_output=False,
    )
    assert sol.success
    y_scipy = sol.y[:, -1]

    # Analytic check for severity context (not the gate).
    y_analytic = np.exp(0.1)

    # Lane gate: state-vector parity.
    assert np.allclose(y_new_jax_np, y_scipy, rtol=state_rtol, atol=state_atol), (
        "DOPRI5 single-step state-vector parity failed: "
        f"jax={y_new_jax_np}, scipy={y_scipy}, "
        f"analytic={y_analytic}, lane_rtol={state_rtol}, lane_atol={state_atol}"
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
    # Residual sanity: bisection should drive the residual at least as
    # tight as the bracket width.
    assert abs(float(f_star)) <= event_atol, (
        f"event-time residual exceeds lane: |f(t_star)|={abs(float(f_star))}, "
        f"lane_atol={event_atol}"
    )


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
    constant R in a toroidal field; both integrators converge to the
    same step controller, so the ratio should be well below 1.25.
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
# 6. Smoke: full trace under JIT
# ---------------------------------------------------------------------------


def test_trace_fieldline_jit_compiles_and_runs():
    """``trace_fieldline`` is JIT-compatible (no host-side appends or branches)."""

    R0 = 1.0
    B0 = 1.0
    field_fn = _toroidal_field_jax(R0, B0)

    spec = FieldlineTracingSpec(tmax=0.5, rtol=1e-7, atol=1e-9, max_steps=400)

    @jax.jit
    def go(y0):
        return trace_fieldline(spec, y0, field_fn)

    y0 = jnp.asarray([1.05, 0.0, 0.0], dtype=jnp.float64)
    result = go(y0)
    # The JIT path must produce a finite endpoint and a positive accepted
    # step count.
    endpoint = np.asarray(result.trajectory[int(result.steps_taken), 1:4])
    assert np.all(np.isfinite(endpoint))
    assert int(result.steps_taken) > 0
    assert int(result.status) == 0
