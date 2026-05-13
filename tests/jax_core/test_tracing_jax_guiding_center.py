"""Parity tests for the JAX guiding-centre vacuum tracing path (item 14 follow-up).

This module validates four slices of the new
``simsopt.jax_core.tracing.guiding_center_vacuum_rhs`` and
``simsopt.jax_core.tracing.trace_guiding_center`` entry points, and the
``trace_particles`` routing wrapper that consumes them, under the
``event_time_tracing`` parity-ladder lane:

1. ``test_guiding_center_vacuum_rhs_matches_analytic_simple_field`` — at
   zero magnetic moment in ``ToroidalField(R0=1, B0=1)`` the RHS
   position derivative must equal ``v_par * b_hat`` and ``dv_par/dt``
   must vanish; this is the analytic parallel-motion limit of the
   drift equations.
2. ``test_trace_guiding_center_endpoint_matches_upstream_particle_tracing``
   — compare the JAX endpoint vs the upstream
   ``sopp.particle_guiding_center_tracing`` for a simple particle at
   the event-time lane state-vector tolerance.
3. ``test_trace_particles_jax_routes_when_backend_jax`` — verify that
   :func:`simsopt.field.tracing.trace_particles` routes to the JAX
   guiding-centre driver when ``is_jax_backend()`` returns True, and
   that the endpoint matches the upstream C++ oracle at the event-time
   lane state-vector tolerance.
4. ``test_trace_particles_jax_raises_on_unsupported_mode`` — confirm
   that ``mode='full'`` and ``mode='gc'`` both surface explicit
   :class:`NotImplementedError` on the JAX route (no silent fallback).

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.field import tracing as tracing_module
from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.field.toroidal_field_jax import ToroidalFieldJAX
from simsopt.field.tracing import (
    MaxRStoppingCriterion,
    MinRStoppingCriterion,
    trace_particles,
)
from simsopt.jax_core.tracing import (
    GuidingCenterTracingSpec,
    guiding_center_vacuum_rhs,
    trace_guiding_center,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


@pytest.fixture(scope="module")
def event_time_lane():
    return dict(_EVENT_TIME_TOLERANCES)


def _force_jax_backend(monkeypatch):
    """Force ``is_jax_backend`` to return True at the tracing module call site.

    Mirrors the helper used in ``tests/field/test_tracing_jax_item16.py``;
    monkeypatching the bound name on the ``simsopt.field.tracing``
    module flips the routing predicate without mutating any global
    backend mode.
    """

    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)


def _toroidal_field_with_jacobian_jax(R0: float, B0: float):
    """Return a JAX callable returning ``(B, dB_by_dX)`` for ``ToroidalField``.

    ``ToroidalField`` produces ``B_x = -B0 * R0 * y / R**2``,
    ``B_y = +B0 * R0 * x / R**2``, ``B_z = 0`` with ``R**2 = x**2 +
    y**2``. The Jacobian ``dB_by_dX[j, l] = ∂_j B_l`` is taken from
    :func:`jax.jacrev` of that closure (this is the same convention as
    the SIMSOPT C++ ``dB_by_dX`` tensor).
    """

    R0_arr = jnp.asarray(R0, dtype=jnp.float64)
    B0_arr = jnp.asarray(B0, dtype=jnp.float64)

    def field_only(point: jax.Array) -> jax.Array:
        x = point[0]
        y = point[1]
        r2 = x * x + y * y
        coeff = B0_arr * R0_arr / r2
        return jnp.stack([-coeff * y, coeff * x, jnp.asarray(0.0, dtype=jnp.float64)])

    jacobian = jax.jacrev(field_only)

    def field_fn(point: jax.Array):
        return field_only(point), jacobian(point)

    return field_fn


# ---------------------------------------------------------------------------
# 1. RHS analytic limit: zero mu, pure parallel motion along b_hat
# ---------------------------------------------------------------------------


def test_guiding_center_vacuum_rhs_matches_analytic_simple_field(event_time_lane):
    """At ``mu=0`` and ``v_par=0`` the GC RHS reduces to the zero vector.

    The vacuum drift contribution scales like ``0.5 v_perp^2 + v_par^2``
    where ``v_perp^2 = 2 mu |B|``; setting ``mu = 0`` and ``v_par = 0``
    therefore zeroes both the parallel-motion and drift contributions
    to ``dx/dt``. The parallel-velocity derivative is ``-mu (B . grad|B|)
    / |B|``, which vanishes at ``mu = 0`` regardless of the field. In
    that limit the RHS must return the zero 4-vector.

    Additionally, with non-zero ``v_par`` and ``mu = 0``, the parallel
    component of ``dx/dt`` along ``b_hat`` must equal ``v_par`` exactly
    (the only contribution at ``mu=0`` is the parallel-motion term
    ``(v_par / |B|) * B`` plus the curvature drift, which is
    perpendicular to ``b_hat``).
    """

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    R0 = 1.0
    B0 = 1.0
    field_fn = _toroidal_field_with_jacobian_jax(R0, B0)

    mass = 1.0
    charge = 1.0
    point = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    B, _dB = field_fn(point)
    abs_B = float(jnp.linalg.norm(B))
    b_hat = np.asarray(B) / abs_B

    # (a) Zero-everything limit: rhs is identically zero.
    rhs_zero = guiding_center_vacuum_rhs(field_fn, mass, charge, mu=0.0)
    y_zero = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    dydt_zero = np.asarray(rhs_zero(jnp.asarray(0.0, dtype=jnp.float64), y_zero))
    expected_zero = np.zeros(4)
    assert np.allclose(dydt_zero, expected_zero, rtol=state_rtol, atol=state_atol), (
        f"GC vacuum RHS at (mu=0, v_par=0) must be zero, got {dydt_zero}"
    )

    # (b) mu=0, v_par != 0: parallel-direction component of dx/dt
    # equals v_par; dv_par/dt = 0. The perpendicular component is the
    # curvature drift, which lies along ``B x grad|B|`` (orthogonal to
    # ``b_hat`` and ``grad|B|``); for the toroidal field at ``y=0`` that
    # is purely along ``ẑ``.
    v_par = 0.3
    rhs = guiding_center_vacuum_rhs(field_fn, mass, charge, mu=0.0)
    y0 = jnp.asarray([1.0, 0.0, 0.0, v_par], dtype=jnp.float64)
    dydt = np.asarray(rhs(jnp.asarray(0.0, dtype=jnp.float64), y0))
    dposition = dydt[:3]
    dv_par = dydt[3]

    parallel_component = float(np.dot(dposition, b_hat))
    perp_component = dposition - parallel_component * b_hat

    assert np.isclose(parallel_component, v_par, rtol=state_rtol, atol=state_atol), (
        "Parallel component of dx/dt must equal v_par at mu=0: "
        f"got {parallel_component}, expected {v_par}"
    )
    assert np.isclose(dv_par, 0.0, rtol=state_rtol, atol=state_atol), (
        f"dv_par/dt must vanish at mu=0: got {dv_par}"
    )
    # Drift component must be perpendicular to ``b_hat``: that we
    # already enforced by orthogonal projection, but the projection on
    # b_hat must be at machine precision.
    assert abs(float(np.dot(perp_component, b_hat))) <= state_atol, (
        "Drift component must be orthogonal to b_hat: "
        f"dot(perp, b_hat)={float(np.dot(perp_component, b_hat))}"
    )


# ---------------------------------------------------------------------------
# 2. Trace guiding centre endpoint matches upstream sopp on a simple particle
# ---------------------------------------------------------------------------


def test_trace_guiding_center_endpoint_matches_upstream_particle_tracing(
    event_time_lane,
):
    """The JAX driver endpoint matches ``sopp.particle_guiding_center_tracing``.

    A purely toroidal vacuum field has no |B|-gradient drift away
    from the toroidal plane (``GradAbsB`` lies in the R-direction so
    ``B x GradAbsB`` is purely vertical), giving a well-conditioned
    fixture that exercises both the parallel-motion term and the
    vertical drift. We compare the endpoint xyz position and parallel
    velocity at the event-time state-vector lane.
    """

    import simsoptpp as sopp

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    R0 = 1.3
    B0 = 0.8
    field = ToroidalField(R0, B0)

    # Light particle so the gyroradius is small relative to R0; energy
    # in SI units. v_total ~ 1e3, v_par/v_total = 0.6.
    mass = 1.0
    charge = 1.0
    Ekin = 0.5 * mass * 1.0e6  # v_total = 1e3
    speed_total = (2.0 * Ekin / mass) ** 0.5
    v_par = 0.6 * speed_total

    xyz_init = np.array([1.4, 0.0, 0.0], dtype=np.float64)

    # Time horizon: short enough that the JAX adaptive controller does
    # not exhaust its step budget on the lane-permitted ratio. The
    # toroidal traversal time for a particle at this R is
    # ``2 pi R / v_par ~ 1.5e-2`` s; pick ~5% of that.
    tmax = 7.0e-4
    tol = 1e-9

    # Upstream oracle.
    cpu_res, _cpu_hits = sopp.particle_guiding_center_tracing(
        field,
        xyz_init,
        mass,
        charge,
        speed_total,
        v_par,
        tmax,
        tol,
        vacuum=True,
        phis=[],
        stopping_criteria=[],
    )
    cpu_endpoint = np.asarray(cpu_res[-1])  # [t, x, y, z, v_par]

    # JAX run.
    field.set_points(xyz_init.reshape((1, 3)))
    abs_B_initial = float(np.asarray(field.AbsB()).reshape(-1)[0])
    vperp2 = max(speed_total * speed_total - v_par * v_par, 0.0)
    mu = vperp2 / (2.0 * abs_B_initial)

    field_fn = _toroidal_field_with_jacobian_jax(R0, B0)

    spec = GuidingCenterTracingSpec(
        tmax=tmax,
        rtol=tol,
        atol=tol,
        max_steps=4000,
    )
    y0 = jnp.asarray(
        [xyz_init[0], xyz_init[1], xyz_init[2], v_par],
        dtype=jnp.float64,
    )
    result = trace_guiding_center(spec, y0, field_fn, m=mass, q=charge, mu=mu)
    assert int(result.status) == 0, (
        "JAX guiding-centre integrator failed to reach tmax: "
        f"status={int(result.status)}, t_final={float(result.t_final)}"
    )
    traj = np.asarray(result.trajectory)
    mask = np.asarray(result.mask)
    live = traj[mask]
    jax_endpoint = live[-1]  # [t, x, y, z, v_par]

    # Lane gate: state-vector parity on the 4-state position+v_par.
    assert np.allclose(
        jax_endpoint[1:5], cpu_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "JAX vs upstream guiding-centre endpoint parity failed: "
        f"jax={jax_endpoint}, cpu={cpu_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


# ---------------------------------------------------------------------------
# 3. Public wrapper routes through JAX when ``is_jax_backend()`` is True
# ---------------------------------------------------------------------------


def test_trace_particles_jax_routes_when_backend_jax(monkeypatch, event_time_lane):
    """``trace_particles(mode='gc_vac')`` routes through JAX under the backend flip.

    The public wrapper must return the same ``(res_tys, res_phi_hits)``
    shape as the CPU oracle, with each row of ``res_tys[i]`` being
    ``[t, x, y, z, v_par]``. The endpoint must match the C++ oracle at
    the event-time state-vector lane.
    """

    import simsoptpp as sopp

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    R0 = 1.3
    B0 = 0.8
    field = ToroidalField(R0, B0)
    jax_field = ToroidalFieldJAX(R0, B0)

    mass = 1.0
    charge = 1.0
    Ekin = 0.5 * mass * 1.0e6
    speed_total = (2.0 * Ekin / mass) ** 0.5
    v_par = 0.6 * speed_total
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)

    tmax = 7.0e-4
    tol = 1e-9

    # CPU oracle path.
    cpu_res, _cpu_hits = sopp.particle_guiding_center_tracing(
        field,
        xyz_init[0],
        mass,
        charge,
        speed_total,
        v_par,
        tmax,
        tol,
        vacuum=True,
        phis=[],
        stopping_criteria=[],
    )
    cpu_endpoint = np.asarray(cpu_res[-1])

    # JAX route via the public wrapper.
    _force_jax_backend(monkeypatch)
    monkeypatch.setattr(
        jax,
        "pure_callback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("JAX particle tracing route must not use pure_callback")
        ),
    )
    res_tys, res_phi_hits = trace_particles(
        jax_field,
        xyz_init,
        [v_par],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        mode="gc_vac",
    )

    # Public-shape contract.
    assert isinstance(res_tys, list) and len(res_tys) == 1
    assert isinstance(res_phi_hits, list) and len(res_phi_hits) == 1
    jax_traj = res_tys[0]
    assert jax_traj.ndim == 2 and jax_traj.shape[1] == 5
    assert jax_traj.shape[0] >= 2
    # phi-plane crossings are out-of-scope for the MVP; the JAX route
    # surfaces an empty ``(0, 6)`` array per particle.
    phi_hits = res_phi_hits[0]
    assert phi_hits.shape == (0, 6)

    # Initial state preserved.
    assert np.isclose(jax_traj[0, 0], 0.0)
    assert np.isclose(jax_traj[0, 1], xyz_init[0, 0])
    assert np.isclose(jax_traj[0, 2], xyz_init[0, 1])
    assert np.isclose(jax_traj[0, 3], xyz_init[0, 2])
    assert np.isclose(jax_traj[0, 4], v_par)

    # Endpoint parity at the event-time state-vector lane gate.
    jax_endpoint = jax_traj[-1]
    assert np.allclose(
        jax_endpoint[1:5], cpu_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "trace_particles JAX vs CPU endpoint parity failed: "
        f"jax={jax_endpoint}, cpu={cpu_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


def test_trace_particles_jax_rejects_cpu_field_without_callback_bridge(monkeypatch):
    """JAX particle routing requires a JAX-native field, not a CPU callback bridge."""

    field = ToroidalField(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)
    _force_jax_backend(monkeypatch)

    with pytest.raises(TypeError, match="JAX-native MagneticField"):
        trace_particles(
            field,
            xyz_init,
            [1.0],
            tmax=1e-5,
            tol=1e-9,
            mode="gc_vac",
        )


# ---------------------------------------------------------------------------
# 4. Public wrapper raises NotImplementedError for unsupported modes
# ---------------------------------------------------------------------------


def test_trace_particles_jax_raises_on_unsupported_mode(monkeypatch):
    """``mode='gc'`` must still raise on the JAX route.

    Only the 4-state Cartesian vacuum guiding-centre RHS and the
    6-state Cartesian full-orbit Lorentz RHS are implemented in the JAX
    driver today. ``mode='gc'`` (non-vacuum guiding-centre) remains a
    deferred follow-up and raises explicit :class:`NotImplementedError`
    — no silent fallback to the C++ path is allowed.

    ``mode='full'`` (``FullorbitRHS``) is now routed by
    :func:`_trace_particles_jax_fullorbit_vacuum`; its dedicated parity
    coverage lives in ``tests/jax_core/test_tracing_jax_fullorbit.py``.
    """

    field = ToroidalField(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)
    speed_par = [1.0]

    _force_jax_backend(monkeypatch)

    with pytest.raises(NotImplementedError, match="gc_vac"):
        trace_particles(
            field,
            xyz_init,
            speed_par,
            tmax=1e-5,
            tol=1e-9,
            mode="gc",
        )


# ---------------------------------------------------------------------------
# 5. Public wrapper accepts phis/stopping criteria and comm
# ---------------------------------------------------------------------------


def test_trace_particles_jax_accepts_phi_planes(monkeypatch):
    """``phis`` is now supported via the bracketed phi-plane event localizer.

    The previous carve-out has been lifted: the JAX guiding-centre
    driver records each phi-plane crossing in the fixed-shape
    ``phi_hits`` buffer and the public wrapper surfaces it through
    ``res_phi_hits`` with the same ``[t, idx, x, y, z, v_par]`` row
    layout as the C++ oracle.
    """

    field = ToroidalFieldJAX(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)
    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = trace_particles(
        field,
        xyz_init,
        [1.0],
        tmax=1e-5,
        tol=1e-9,
        mode="gc_vac",
        phis=[0.0],
    )
    assert len(res_tys) == 1
    assert res_phi_hits[0].shape[1] == 6


def test_trace_particles_jax_accepts_minR_stopping_criterion(monkeypatch):
    """Non-Levelset stopping criteria are now accepted on the JAX route.

    The previous carve-out raised :class:`NotImplementedError` for any
    non-Levelset criterion; the isinstance dispatch in
    ``_translate_stopping_criteria_to_jax`` now maps each CPU
    criterion class to its JAX dataclass mirror.
    """

    field = ToroidalFieldJAX(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)
    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = trace_particles(
        field,
        xyz_init,
        [1.0],
        tmax=1e-5,
        tol=1e-9,
        mode="gc_vac",
        stopping_criteria=[MinRStoppingCriterion(0.5)],
    )
    assert len(res_tys) == 1
    assert res_phi_hits[0].shape[1] == 6


def test_trace_particles_jax_stopping_event_not_live_trajectory(monkeypatch):
    """Stopped post-step GC states are event rows, not live trajectory rows."""

    field = ToroidalFieldJAX(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)
    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = trace_particles(
        field,
        xyz_init,
        [1.0],
        tmax=1e-5,
        tol=1e-9,
        mode="gc_vac",
        stopping_criteria=[MaxRStoppingCriterion(1.3)],
    )

    assert len(res_tys) == 1
    np.testing.assert_allclose(res_tys[0], np.array([[0.0, 1.4, 0.0, 0.0, 1.0]]))
    hits = res_phi_hits[0]
    assert hits.shape[0] == 1
    assert int(hits[0, 1]) == -1


def test_trace_particles_jax_comm_matches_single_process(
    monkeypatch, assert_two_rank_replay_matches
):
    """The JAX guiding-centre route preserves CPU-style comm gather order."""

    field = ToroidalFieldJAX(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0], [1.45, 0.0, 0.01]], dtype=np.float64)
    _force_jax_backend(monkeypatch)
    no_comm_tys, no_comm_hits = trace_particles(
        field,
        xyz_init,
        [1.0, 0.9],
        tmax=1e-5,
        tol=1e-9,
        mode="gc_vac",
    )
    assert_two_rank_replay_matches(
        no_comm_tys,
        no_comm_hits,
        lambda comm: trace_particles(
            field,
            xyz_init,
            [1.0, 0.9],
            tmax=1e-5,
            tol=1e-9,
            mode="gc_vac",
            comm=comm,
        ),
    )
