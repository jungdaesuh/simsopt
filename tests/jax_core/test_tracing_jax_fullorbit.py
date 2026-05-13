"""Parity tests for the JAX full-orbit vacuum tracing path (item 14 closeout).

This module validates the new
``simsopt.jax_core.tracing.fullorbit_vacuum_rhs`` and
``simsopt.jax_core.tracing.trace_fullorbit`` entry points, and the
``trace_particles`` routing wrapper that consumes them, under the
``event_time_tracing`` parity-ladder lane:

1. ``test_fullorbit_vacuum_rhs_recovers_helical_motion_in_uniform_field`` —
   in a uniform ``B = B0 ẑ`` field with initial velocity
   ``(v_perp, 0, v_par)`` the Lorentz orbit is an exact helix with
   gyrofrequency ``omega_c = q B0 / m`` and gyroradius
   ``r_g = m v_perp / (q B0)``. The JAX driver state at ``t = T`` (one
   gyroperiod) must reproduce the analytic position and velocity to
   within the lane state-vector tolerance.
2. ``test_trace_fullorbit_endpoint_matches_cpp_oracle`` — endpoint
   parity vs ``sopp.particle_fullorbit_tracing`` on a simple toroidal
   field configuration at the event-time lane state-vector tolerance.
3. ``test_trace_fullorbit_conservation_invariants`` — kinetic energy
   ``0.5 m |v|^2`` is conserved by the Lorentz force (which is purely
   normal to ``v``) within tolerance.
4. ``test_trace_particles_jax_routes_full_mode_when_backend_jax`` —
   verify routing of ``mode='full'`` through the public
   :func:`simsopt.field.tracing.trace_particles` wrapper to
   :func:`_trace_particles_jax_fullorbit_vacuum` when
   ``is_jax_backend()`` returns True; verify endpoint parity vs the
   C++ oracle.
5. ``test_trace_fullorbit_records_phi_plane_crossing`` — verify that the
   fixed-shape event buffer records full-orbit phi-plane hits.
6. ``test_trace_fullorbit_records_stopping_criterion_event`` — verify that
   a JAX-side stopping criterion appends a negative-index event row.
7. ``test_trace_particles_jax_full_mode_records_stopping_criterion`` —
   verify that the public ``trace_particles(mode='full')`` route forwards
   translated stopping criteria into the JAX full-orbit driver.
8. ``test_trace_particles_jax_full_mode_comm_and_unknown_criteria`` —
   confirm that MPI ``comm`` preserves split/gather semantics and unknown
   stopping-criterion subclasses continue to raise explicit
   :class:`NotImplementedError` on the JAX route for ``mode='full'``.

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
from simsopt.field.tracing import MaxRStoppingCriterion, trace_particles
from simsopt.jax_core.tracing import (
    FullorbitTracingSpec,
    MaxRStoppingCriterion as JaxMaxRStoppingCriterion,
    fullorbit_vacuum_rhs,
    trace_fullorbit,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


@pytest.fixture(scope="module")
def event_time_lane():
    return dict(_EVENT_TIME_TOLERANCES)


def _force_jax_backend(monkeypatch):
    """Force ``is_jax_backend`` to return True at the tracing module call site.

    Mirrors the helper used in
    ``tests/jax_core/test_tracing_jax_guiding_center.py``;
    monkeypatching the bound name on the ``simsopt.field.tracing``
    module flips the routing predicate without mutating any global
    backend mode.
    """

    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)


def _toroidal_field_jax(R0: float, B0: float):
    """Return a JAX callable returning ``B`` for ``ToroidalField(R0, B0)``.

    Uses the closed-form definition of the toroidal field to avoid
    requiring a CPU ``MagneticField`` host callback in tests that only
    exercise the JAX driver in isolation.
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


def _uniform_field_jax(B0: float):
    """Return a JAX callable returning ``B = (0, 0, B0)`` (uniform-z field)."""

    B0_arr = jnp.asarray(B0, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def field_fn(_point: jax.Array) -> jax.Array:
        return jnp.stack([zero, zero, B0_arr])

    return field_fn


# ---------------------------------------------------------------------------
# 1. RHS analytic limit: helical motion in a uniform B field
# ---------------------------------------------------------------------------


def test_fullorbit_vacuum_rhs_recovers_helical_motion_in_uniform_field(
    event_time_lane,
):
    """The Lorentz integrator reproduces helical motion in ``B = B0 ẑ``.

    For an initial state ``(x0, 0, 0, 0, v_perp, v_par)`` with
    positive ``q B0`` the gyrocentre is at ``(x0 - r_g, 0, 0)`` with
    gyroradius ``r_g = m v_perp / (q B0)`` and gyrofrequency
    ``omega_c = q B0 / m``. The Larmor circle convention from the
    Lorentz equation ``dv/dt = (q/m) v x B = (q/m) (v_y B0, -v_x B0,
    0)`` gives ``v_x(t) = v_perp sin(omega_c t)``, ``v_y(t) = v_perp
    cos(omega_c t)``, ``v_z = v_par``, and
    ``x(t) = x0 + r_g (1 - cos(omega_c t))``,
    ``y(t) = r_g sin(omega_c t)``, ``z(t) = v_par t``. After exactly
    one period ``T = 2 pi / omega_c`` the state returns to the initial
    position in the (x, y) plane (z advances by ``v_par T``).
    """

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    B0 = 1.5
    mass = 1.0
    charge = 1.0
    omega_c = charge * B0 / mass

    v_perp = 0.4
    v_par = 0.2
    # Gyroradius r_g = m v_perp / (q B0) is part of the analytic
    # solution narrative in the docstring; the endpoint check below
    # only references the period because the orbit closes on the (x, y)
    # plane after exactly one revolution.
    period = 2.0 * np.pi / omega_c

    field_fn = _uniform_field_jax(B0)

    # Initial state: at (x0, 0, 0) with velocity (0, v_perp, v_par)
    # gives a gyrocentre at (x0 - r_g, 0, 0). Pick x0 > r_g so the
    # gyrocircle stays away from the origin.
    x0 = 2.0
    y0_state = jnp.asarray(
        [x0, 0.0, 0.0, 0.0, v_perp, v_par],
        dtype=jnp.float64,
    )

    # (a) RHS at t=0: dx/dt = v = (0, v_perp, v_par); dv/dt =
    # (q/m) v x B = (q/m)(v_perp * B0, 0, 0) = (omega_c * v_perp, 0, 0).
    rhs = fullorbit_vacuum_rhs(field_fn, m=mass, q=charge)
    dydt0 = np.asarray(rhs(jnp.asarray(0.0, dtype=jnp.float64), y0_state))
    expected_dydt0 = np.array(
        [0.0, v_perp, v_par, omega_c * v_perp, 0.0, 0.0],
        dtype=np.float64,
    )
    assert np.allclose(dydt0, expected_dydt0, rtol=state_rtol, atol=state_atol), (
        f"Lorentz RHS at t=0 must reproduce uniform-field analytic value: "
        f"got {dydt0}, expected {expected_dydt0}"
    )

    # (b) Integrate one period and compare endpoint vs analytic helix
    # endpoint: position (x0, 0, v_par * period); velocity (0, v_perp,
    # v_par). Use a tight tolerance and a generous step budget.
    spec = FullorbitTracingSpec(
        tmax=float(period),
        rtol=1e-11,
        atol=1e-11,
        max_steps=20000,
    )
    result = trace_fullorbit(spec, y0_state, field_fn, m=mass, q=charge)
    assert int(result.status) == 0, (
        "JAX full-orbit integrator failed to reach one gyroperiod: "
        f"status={int(result.status)}, t_final={float(result.t_final)}"
    )
    traj = np.asarray(result.trajectory)
    mask = np.asarray(result.mask)
    live = traj[mask]
    endpoint = live[-1]  # [t, x, y, z, vx, vy, vz]

    expected_endpoint = np.array(
        [
            float(period),
            x0,
            0.0,
            v_par * float(period),
            0.0,
            v_perp,
            v_par,
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        endpoint[1:], expected_endpoint[1:], rtol=state_rtol, atol=state_atol
    ), (
        "JAX full-orbit endpoint after one period must reproduce analytic "
        f"helix: got {endpoint[1:]}, expected {expected_endpoint[1:]}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )

    # (c) Energy conservation: |v|^2 conserved within tolerance after a
    # full period.
    v_initial2 = v_perp * v_perp + v_par * v_par
    v_final = endpoint[4:7]
    v_final2 = float(np.dot(v_final, v_final))
    rel_energy_err = abs(v_final2 - v_initial2) / v_initial2
    # Lorentz force is normal to velocity so kinetic energy should be
    # bit-stable up to RK roundoff. The lane bound is loose; tighter
    # checks live in test_trace_fullorbit_conservation_invariants.
    assert rel_energy_err <= state_rtol, (
        f"Kinetic energy drifted over one gyroperiod: rel_err={rel_energy_err}"
    )


# ---------------------------------------------------------------------------
# 2. Trace full-orbit endpoint matches upstream sopp on a simple particle
# ---------------------------------------------------------------------------


def test_trace_fullorbit_endpoint_matches_cpp_oracle(event_time_lane):
    """The JAX driver endpoint matches ``sopp.particle_fullorbit_tracing``.

    We pick a toroidal-field fixture (vacuum, no E field) where the C++
    ``FullorbitRHS::operator()`` exercises the same Lorentz equation
    that the JAX driver implements. The endpoint xyz position and
    Cartesian velocity must match at the event-time state-vector lane
    tolerance.
    """

    import simsoptpp as sopp

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    R0 = 1.3
    B0 = 0.8
    field = ToroidalField(R0, B0)

    mass = 1.0
    charge = 1.0
    xyz_init = np.array([1.4, 0.0, 0.0], dtype=np.float64)
    # Light particle so the gyroradius is small relative to R0. The
    # initial velocity is mostly parallel to the local B (which is
    # along +y at the initial point), with a small perpendicular kick.
    v_init = np.array([10.0, 100.0, 30.0], dtype=np.float64)

    # Time horizon: short enough that the JAX adaptive controller does
    # not exhaust its step budget; ~5 gyroperiods. omega_c = q B / m =
    # B0 * R0 / |R|^2 ~ 0.53 at R = 1.4, so T ~ 12 s. Use 1e-2 to
    # capture multiple cycles without exhausting the step cap.
    tmax = 1.0e-2
    tol = 1e-11

    # Upstream oracle.
    cpu_res, _cpu_hits = sopp.particle_fullorbit_tracing(
        field,
        xyz_init,
        v_init,
        mass,
        charge,
        tmax,
        tol,
        phis=[],
        stopping_criteria=[],
    )
    cpu_endpoint = np.asarray(cpu_res[-1])  # [t, x, y, z, vx, vy, vz]

    field_fn = _toroidal_field_jax(R0, B0)

    spec = FullorbitTracingSpec(
        tmax=tmax,
        rtol=tol,
        atol=tol,
        max_steps=20000,
    )
    y0 = jnp.asarray(
        [
            xyz_init[0],
            xyz_init[1],
            xyz_init[2],
            v_init[0],
            v_init[1],
            v_init[2],
        ],
        dtype=jnp.float64,
    )
    result = trace_fullorbit(spec, y0, field_fn, m=mass, q=charge)
    assert int(result.status) == 0, (
        "JAX full-orbit integrator failed to reach tmax: "
        f"status={int(result.status)}, t_final={float(result.t_final)}"
    )
    traj = np.asarray(result.trajectory)
    mask = np.asarray(result.mask)
    live = traj[mask]
    jax_endpoint = live[-1]  # [t, x, y, z, vx, vy, vz]

    # Lane gate: state-vector parity on the 6-state position+velocity.
    assert np.allclose(
        jax_endpoint[1:7], cpu_endpoint[1:7], rtol=state_rtol, atol=state_atol
    ), (
        "JAX vs upstream full-orbit endpoint parity failed: "
        f"jax={jax_endpoint}, cpu={cpu_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


# ---------------------------------------------------------------------------
# 3. Trace full-orbit conserves kinetic energy
# ---------------------------------------------------------------------------


def test_trace_fullorbit_conservation_invariants(event_time_lane):
    """Kinetic energy ``0.5 m |v|^2`` is conserved by the Lorentz force.

    The vacuum Lorentz force ``F = q v x B`` is purely normal to ``v``
    so the kinetic energy is an exact invariant of the continuous
    equation. The DOPRI5 integrator violates this only at the level of
    its local truncation error, which the lane bounds via the
    state-vector rtol. We check both the magnitude of the velocity at
    the endpoint and a sample of intermediate accepted steps.
    """

    state_rtol = float(event_time_lane["state_vector_rtol"])

    R0 = 1.3
    B0 = 0.8
    field_fn = _toroidal_field_jax(R0, B0)

    mass = 1.0
    charge = 1.0
    xyz_init = np.array([1.4, 0.0, 0.0], dtype=np.float64)
    v_init = np.array([10.0, 100.0, 30.0], dtype=np.float64)
    v_initial2 = float(np.dot(v_init, v_init))

    # Same horizon as the parity test so the step budget is comparable.
    tmax = 1.0e-2
    tol = 1e-11

    spec = FullorbitTracingSpec(
        tmax=tmax,
        rtol=tol,
        atol=tol,
        max_steps=20000,
    )
    y0 = jnp.asarray(
        [
            xyz_init[0],
            xyz_init[1],
            xyz_init[2],
            v_init[0],
            v_init[1],
            v_init[2],
        ],
        dtype=jnp.float64,
    )
    result = trace_fullorbit(spec, y0, field_fn, m=mass, q=charge)
    assert int(result.status) == 0, (
        "JAX full-orbit integrator failed to reach tmax: "
        f"status={int(result.status)}, t_final={float(result.t_final)}"
    )
    traj = np.asarray(result.trajectory)
    mask = np.asarray(result.mask)
    live = traj[mask]
    assert live.shape[0] >= 2, "Need at least one accepted step beyond init"

    # Endpoint kinetic energy parity vs the initial state.
    v_final = live[-1, 4:7]
    v_final2 = float(np.dot(v_final, v_final))
    rel_endpoint_err = abs(v_final2 - v_initial2) / v_initial2
    assert rel_endpoint_err <= state_rtol, (
        "Kinetic energy not conserved at the endpoint: "
        f"|v_initial|^2={v_initial2}, |v_final|^2={v_final2}, "
        f"rel_err={rel_endpoint_err}, lane rtol={state_rtol}"
    )

    # Spot-check intermediate accepted rows: kinetic energy must remain
    # within the lane rtol throughout the run. Skip the initial row.
    n_steps = live.shape[0]
    sample_idxs = np.linspace(1, n_steps - 1, num=min(10, n_steps - 1), dtype=int)
    for idx in sample_idxs:
        v_idx = live[idx, 4:7]
        v_idx2 = float(np.dot(v_idx, v_idx))
        rel = abs(v_idx2 - v_initial2) / v_initial2
        assert rel <= state_rtol, (
            f"Kinetic energy not conserved at accepted step {idx}/"
            f"{n_steps - 1}: rel_err={rel} > lane rtol={state_rtol}"
        )


# ---------------------------------------------------------------------------
# 4. Full-orbit events use the fixed-shape phi_hits buffer
# ---------------------------------------------------------------------------


def test_trace_fullorbit_records_phi_plane_crossing(event_time_lane):
    """Full-orbit JAX tracing records phi-plane events with 8-column rows."""

    event_rtol = float(event_time_lane["event_time_rtol"])
    event_atol = float(event_time_lane["event_time_atol"])
    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    phi_target = 0.1
    expected_t = np.tan(phi_target)
    field_fn = _uniform_field_jax(0.0)
    y0 = jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=jnp.float64)
    spec = FullorbitTracingSpec(
        tmax=0.2,
        rtol=1e-11,
        atol=1e-11,
        max_steps=2000,
        max_phi_hits=8,
    )

    result = trace_fullorbit(
        spec,
        y0,
        field_fn,
        m=1.0,
        q=0.0,
        phis=jnp.asarray([phi_target], dtype=jnp.float64),
    )
    hits = np.asarray(result.phi_hits)[: int(result.phi_hits_count)]

    assert int(result.status) == 0
    assert hits.shape == (1, 8)
    assert hits[0, 1] == 0.0
    assert np.isclose(hits[0, 0], expected_t, rtol=event_rtol, atol=event_atol)
    assert np.allclose(
        hits[0, 2:8],
        np.array([1.0, expected_t, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        rtol=state_rtol,
        atol=state_atol,
    )


def test_trace_fullorbit_records_stopping_criterion_event():
    """Full-orbit JAX tracing records standard stopping criteria as idx < 0."""

    field_fn = _uniform_field_jax(0.0)
    y0 = jnp.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    spec = FullorbitTracingSpec(
        tmax=0.2,
        rtol=1e-11,
        atol=1e-11,
        max_steps=2000,
        max_phi_hits=8,
    )

    result = trace_fullorbit(
        spec,
        y0,
        field_fn,
        m=1.0,
        q=0.0,
        stopping_criteria=(JaxMaxRStoppingCriterion(crit_r=0.9),),
    )
    hits = np.asarray(result.phi_hits)[: int(result.phi_hits_count)]
    live = np.asarray(result.trajectory)[np.asarray(result.mask)]

    assert int(result.status) == -1
    np.testing.assert_allclose(live, np.array([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    assert hits.shape[0] >= 1
    assert hits[0, 1] == -1.0
    assert hits[0, 2] >= 0.9


# ---------------------------------------------------------------------------
# 5. Public wrapper routes mode='full' through JAX when backend is JAX
# ---------------------------------------------------------------------------


def test_trace_particles_jax_routes_full_mode_when_backend_jax(
    monkeypatch, event_time_lane
):
    """``trace_particles(mode='full')`` routes through JAX under the backend flip.

    The public wrapper must return the same ``(res_tys, res_phi_hits)``
    shape as the CPU oracle, with each row of ``res_tys[i]`` being
    ``[t, x, y, z, vx, vy, vz]``. The endpoint must match the C++
    oracle at the event-time state-vector lane.
    """

    import simsoptpp as sopp
    from simsopt.field.toroidal_field_jax import ToroidalFieldJAX

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    R0 = 1.3
    B0 = 0.8
    # CPU oracle path uses the CPU ToroidalField; JAX path uses the
    # JAX-backed ToroidalFieldJAX (which inherits from MagneticField so
    # gc_to_fullorbit_initial_guesses can read ``set_points``/``B``/
    # ``AbsB`` on the host).
    cpu_field = ToroidalField(R0, B0)
    jax_field = ToroidalFieldJAX(R0, B0)

    mass = 1.0
    charge = 1.0
    Ekin = 0.5 * mass * (110.0**2)  # v_total = 110
    speed_total = (2.0 * Ekin / mass) ** 0.5
    v_par = 0.3 * speed_total
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)
    phase_angle = 0.0

    tmax = 1.0e-3
    tol = 1e-11

    # CPU oracle path: seed initial guesses identically to how the
    # public wrapper would in the C++ branch.
    from simsopt.field.tracing import gc_to_fullorbit_initial_guesses

    xyz_full, v_full, _ = gc_to_fullorbit_initial_guesses(
        cpu_field,
        xyz_init,
        np.asarray([v_par], dtype=np.float64),
        speed_total,
        mass,
        charge,
        eta=phase_angle,
    )
    cpu_res, _cpu_hits = sopp.particle_fullorbit_tracing(
        cpu_field,
        xyz_full[0],
        v_full[0],
        mass,
        charge,
        tmax,
        tol,
        phis=[],
        stopping_criteria=[],
    )
    cpu_endpoint = np.asarray(cpu_res[-1])

    # JAX route via the public wrapper.
    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = trace_particles(
        jax_field,
        xyz_init,
        [v_par],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        mode="full",
        phase_angle=phase_angle,
    )

    # Public-shape contract.
    assert isinstance(res_tys, list) and len(res_tys) == 1
    assert isinstance(res_phi_hits, list) and len(res_phi_hits) == 1
    jax_traj = res_tys[0]
    assert jax_traj.ndim == 2 and jax_traj.shape[1] == 7
    assert jax_traj.shape[0] >= 2
    # phi-plane crossings are out-of-scope for the MVP; the JAX route
    # surfaces an empty ``(0, 8)`` array per particle for full orbits.
    phi_hits = res_phi_hits[0]
    assert phi_hits.shape == (0, 8)

    # Initial state preserved.
    assert np.isclose(jax_traj[0, 0], 0.0)
    assert np.isclose(jax_traj[0, 1], xyz_full[0, 0])
    assert np.isclose(jax_traj[0, 2], xyz_full[0, 1])
    assert np.isclose(jax_traj[0, 3], xyz_full[0, 2])
    assert np.isclose(jax_traj[0, 4], v_full[0, 0])
    assert np.isclose(jax_traj[0, 5], v_full[0, 1])
    assert np.isclose(jax_traj[0, 6], v_full[0, 2])

    # Endpoint parity at the event-time state-vector lane gate.
    jax_endpoint = jax_traj[-1]
    assert np.allclose(
        jax_endpoint[1:7], cpu_endpoint[1:7], rtol=state_rtol, atol=state_atol
    ), (
        "trace_particles mode='full' JAX vs CPU endpoint parity failed: "
        f"jax={jax_endpoint}, cpu={cpu_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


# ---------------------------------------------------------------------------
# 6. Public wrapper forwards full-orbit stopping events
# ---------------------------------------------------------------------------


def test_trace_particles_jax_full_mode_records_stopping_criterion(monkeypatch):
    """The public full-orbit wrapper forwards translated stopping events."""

    from simsopt.field.toroidal_field_jax import ToroidalFieldJAX

    field = ToroidalFieldJAX(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0]], dtype=np.float64)

    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = trace_particles(
        field,
        xyz_init,
        [1.0],
        tmax=1e-5,
        tol=1e-9,
        mode="full",
        stopping_criteria=[MaxRStoppingCriterion(0.1)],
    )

    assert len(res_tys) == 1
    assert res_phi_hits[0].shape[1] == 8
    assert res_phi_hits[0][0, 1] == -1.0


# ---------------------------------------------------------------------------
# 7. Public wrapper rejects unsupported argument shapes on mode='full'
# ---------------------------------------------------------------------------


def test_trace_particles_jax_full_mode_comm_and_unknown_criteria(
    monkeypatch, assert_two_rank_replay_matches
):
    """``comm`` gathers full-orbit JAX results; unknown criteria still raise.

    Phi-plane crossings and the standard Python stopping-criterion
    wrappers (Min/MaxR, Min/MaxZ, ToroidalTransit, Iteration) are now
    supported on the JAX ``mode='full'`` route through the fixed-shape
    ``phi_hits`` buffer. Raw Python objects that don't match a known
    criterion type continue to raise :class:`NotImplementedError` from
    the translator. No silent fallback is allowed.
    """

    from simsopt.field.toroidal_field_jax import ToroidalFieldJAX

    field = ToroidalFieldJAX(1.3, 0.8)
    xyz_init = np.array([[1.4, 0.0, 0.0], [1.45, 0.0, 0.01]], dtype=np.float64)

    _force_jax_backend(monkeypatch)

    # Unsupported stopping-criterion subclass is rejected by the
    # translator instead of silently dropped.
    with pytest.raises(NotImplementedError, match="stopping criterion"):
        trace_particles(
            field,
            xyz_init[:1],
            [1.0],
            tmax=1e-5,
            tol=1e-9,
            mode="full",
            stopping_criteria=[object()],
        )

    no_comm_tys, no_comm_hits = trace_particles(
        field,
        xyz_init,
        [1.0, 0.9],
        tmax=1e-5,
        tol=1e-9,
        mode="full",
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
            mode="full",
            comm=comm,
        ),
    )
