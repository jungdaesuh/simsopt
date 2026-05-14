"""Parity tests for the JAX Boozer-coordinate guiding-centre tracing path.

This module validates six slices of the JAX Boozer-coordinate
guiding-centre RHS variants and the
:func:`simsopt.field.tracing.trace_particles_boozer` routing wrapper
under the ``event_time_tracing`` parity-ladder lane:

1. ``test_gc_vacuum_boozer_rhs_recovers_axis_drift_limit_when_mu_zero``
   — analytic ``mu=0`` limit of the vacuum-Boozer RHS reduces to pure
   parallel motion plus the ``v_par^2``-driven curvature drift terms;
   the magnetic-mirror force vanishes (``dv_par/dt = 0``).
2. ``test_trace_gc_vacuum_boozer_endpoint_matches_cpp_oracle`` — JAX
   endpoint vs the upstream
   ``sopp.particle_guiding_center_boozer_tracing(vacuum=True)`` at the
   event-time state-vector lane gate.
3. ``test_trace_gc_no_k_boozer_endpoint_matches_cpp_oracle`` — same
   parity gate for the ``noK=True`` lane.
4. ``test_trace_gc_full_boozer_endpoint_matches_cpp_oracle`` — same
   parity gate for the full ``GuidingCenterBoozerRHS`` (non-vacuum,
   non-zero K) lane.
5. ``test_trace_particles_boozer_jax_routes_when_field_is_jax_wrapper``
   — :func:`trace_particles_boozer` dispatches to the JAX driver when
   ``is_jax_backend()`` is True AND the field is a
   ``BoozerRadialInterpolantJAX`` instance.
6. ``test_trace_particles_boozer_jax_raises_on_unsupported_mode`` —
   non-{gc, gc_vac, gc_nok} mode strings surface explicit failure (the
   ``mode='full'`` full-orbit-Lorentz case raises through the public
   ``assert`` on the supported set).

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.field import tracing as tracing_module
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from simsopt.field.boozermagneticfield import InterpolatedBoozerField
from simsopt.field.boozermagneticfield_jax import (
    BoozerRadialInterpolantJAX,
    InterpolatedBoozerFieldJAX,
)
from simsopt.field.tracing import trace_particles_boozer
from simsopt.jax_core.tracing import (
    _BOOZER_RHS_EVAL_KEYS,
    GuidingCenterTracingSpec,
    guiding_center_vacuum_boozer_rhs,
    trace_guiding_center_boozer,
)
from simsopt.mhd.vmec import Vmec


_TEST_FILES = (Path(__file__).parent / ".." / "test_files").resolve()
_WOUT_STELLSYM = str((_TEST_FILES / "wout_n3are_R7.75B5.7_lowres.nc").resolve())


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


@pytest.fixture(scope="module")
def event_time_lane():
    return dict(_EVENT_TIME_TOLERANCES)


@pytest.fixture(scope="module")
def vacuum_bri_and_jax():
    """``no_K=True`` Boozer field — pulls C++ vacuum/no_K RHS path."""
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(
        vmec, order=3, mpol=4, ntor=4, rescale=True, no_K=True
    )
    return bri, BoozerRadialInterpolantJAX(bri)


@pytest.fixture(scope="module")
def full_bri_and_jax():
    """``no_K=False`` Boozer field — pulls C++ full RHS path."""
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(
        vmec, order=3, mpol=4, ntor=4, rescale=True, no_K=False
    )
    return bri, BoozerRadialInterpolantJAX(bri)


def _force_jax_backend(monkeypatch):
    """Force ``is_jax_backend`` to return True at the tracing module call site.

    Mirrors the helper in ``tests/jax_core/test_tracing_jax_guiding_center.py``
    so the routing predicate flips without mutating the global backend.
    """
    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)


# ---------------------------------------------------------------------------
# 1. RHS analytic limit: mu=0 zeroes the mirror force and v_perp^2 drift
# ---------------------------------------------------------------------------


def test_gc_vacuum_boozer_rhs_recovers_axis_drift_limit_when_mu_zero(
    vacuum_bri_and_jax, event_time_lane
):
    """At ``mu = 0`` the vacuum-Boozer RHS recovers an analytic limit.

    The upstream definition of the four derivatives at ``mu = 0`` is

        ds       = -|B|_{,theta} (m v_par^2 / |B|) / (q psi0)
        dtheta   = +|B|_{,s}     (m v_par^2 / |B|) / (q psi0) + iota v_par |B|/G
        dzeta    = v_par |B|/G
        dv_par   = 0                                          # mu factor vanishes

    The mirror-force term and the v_perp^2 contribution both scale
    with ``mu``, so this is a closed-form check on the RHS branch:
    only the curvature drift and parallel motion survive, and
    ``dv_par/dt`` is identically zero.
    """

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, jax_field = vacuum_bri_and_jax

    m = 1.0
    q = 1.0
    v_par = 1.0e3
    point_stz = np.array([0.30, 1.0, 0.40], dtype=np.float64)

    bri.set_points(point_stz.reshape((1, 3)))
    modB = float(bri.modB().reshape(-1)[0])
    derivs = np.asarray(bri.modB_derivs()).reshape(-1)
    dmodBds = float(derivs[0])
    dmodBdtheta = float(derivs[1])
    # ``derivs[2]`` is dmodBdzeta but it only enters the mirror-force term
    # ``-(iota |B|_theta + |B|_zeta) mu |B|/G``, which vanishes at mu=0.
    G = float(bri.G().reshape(-1)[0])
    iota = float(bri.iota().reshape(-1)[0])
    psi0 = float(bri.psi0)

    fak1 = m * v_par * v_par / modB  # mu = 0 zeroes the m*mu term
    expected_ds = -dmodBdtheta * fak1 / (q * psi0)
    expected_dtheta = dmodBds * fak1 / (q * psi0) + iota * v_par * modB / G
    expected_dzeta = v_par * modB / G
    expected_dv_par = 0.0  # iota |B|_theta + |B|_zeta multiplied by mu = 0

    rhs = guiding_center_vacuum_boozer_rhs(jax_field, m=m, q=q, mu=0.0)
    y = jnp.asarray([*point_stz, v_par], dtype=jnp.float64)
    dydt = np.asarray(rhs(jnp.asarray(0.0, dtype=jnp.float64), y))

    assert np.allclose(
        dydt,
        np.array([expected_ds, expected_dtheta, expected_dzeta, expected_dv_par]),
        rtol=state_rtol,
        atol=state_atol,
    ), (
        "Vacuum-Boozer RHS at mu=0 must reproduce the closed-form limit: "
        f"got {dydt}, expected "
        f"({expected_ds}, {expected_dtheta}, {expected_dzeta}, "
        f"{expected_dv_par})"
    )


# ---------------------------------------------------------------------------
# 2. trace_guiding_center_boozer endpoint parity (vacuum)
# ---------------------------------------------------------------------------


def _short_orbit_inputs(modB_init: float, *, mass: float = 1.0):
    """Build a short-orbit fixture: speeds, parallel velocity, tmax.

    The orbit is set up so the integrator does not exhaust its step
    budget under the lane step-count contract. Both speed and tmax are
    deliberately modest so the CPU oracle and the JAX driver complete
    in O(100) accepted steps per call. With ``v_total=1e3 m/s`` and
    ``tmax=1e-4 s`` the C++ oracle traces ~78 accepted steps on the
    n3are_R7.75B5.7_lowres fixture; this leaves comfortable headroom
    under the lane ``step_count_max_ratio=1.25`` budget.
    """

    speed_total = 1.0e3
    Ekin = 0.5 * mass * speed_total**2
    v_par = 0.6 * speed_total
    return Ekin, speed_total, v_par


def _make_initial_point() -> np.ndarray:
    return np.array([0.30, 0.0, 0.0], dtype=np.float64)


def _run_cpp_oracle(
    bri,
    stz_init: np.ndarray,
    *,
    mass: float,
    charge: float,
    speed_total: float,
    v_par: float,
    tmax: float,
    tol: float,
    vacuum: bool,
    noK: bool,
):
    """Direct ``sopp.particle_guiding_center_boozer_tracing`` invocation."""

    res, _hits = sopp.particle_guiding_center_boozer_tracing(
        bri,
        stz_init,
        mass,
        charge,
        speed_total,
        v_par,
        tmax,
        tol,
        vacuum=vacuum,
        noK=noK,
        zetas=[],
        stopping_criteria=[],
    )
    return np.asarray(res[-1])  # [t, s, theta, zeta, v_par]


def _run_jax_path(
    jax_field,
    stz_init: np.ndarray,
    *,
    mass: float,
    charge: float,
    speed_total: float,
    v_par: float,
    tmax: float,
    tol: float,
    mode: str,
):
    """Run :func:`trace_guiding_center_boozer` and return the endpoint state."""

    jax_field.set_points(stz_init.reshape((1, 3)))
    abs_B_initial = float(np.asarray(jax_field.modB()).reshape(-1)[0])
    vperp2 = max(speed_total * speed_total - v_par * v_par, 0.0)
    mu = vperp2 / (2.0 * abs_B_initial)

    spec = GuidingCenterTracingSpec(tmax=tmax, rtol=tol, atol=tol, max_steps=4000)
    y0 = jnp.asarray([*stz_init, v_par], dtype=jnp.float64)
    result = trace_guiding_center_boozer(
        spec, y0, jax_field, m=mass, q=charge, mu=mu, mode=mode
    )
    status = int(result.status)
    assert status == 0, (
        f"JAX Boozer guiding-centre integrator failed to reach tmax: "
        f"status={status}, t_final={float(result.t_final)}"
    )
    traj = np.asarray(result.trajectory, dtype=np.float64)
    mask = np.asarray(result.mask, dtype=bool)
    live = traj[mask]
    return live[-1]


def test_trace_gc_vacuum_boozer_endpoint_matches_cpp_oracle(
    vacuum_bri_and_jax, event_time_lane
):
    """JAX vacuum-Boozer endpoint matches ``sopp`` at the lane gate."""

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, jax_field = vacuum_bri_and_jax
    mass = 1.0
    charge = 1.0
    _Ekin, speed_total, v_par = _short_orbit_inputs(modB_init=1.0, mass=mass)
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10

    cpp_endpoint = _run_cpp_oracle(
        bri,
        stz_init,
        mass=mass,
        charge=charge,
        speed_total=speed_total,
        v_par=v_par,
        tmax=tmax,
        tol=tol,
        vacuum=True,
        noK=False,
    )
    jax_endpoint = _run_jax_path(
        jax_field,
        stz_init,
        mass=mass,
        charge=charge,
        speed_total=speed_total,
        v_par=v_par,
        tmax=tmax,
        tol=tol,
        mode="vacuum",
    )

    assert np.allclose(
        jax_endpoint[1:5], cpp_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "JAX vs upstream vacuum-Boozer endpoint parity failed: "
        f"jax={jax_endpoint}, cpp={cpp_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


# ---------------------------------------------------------------------------
# 3. trace_guiding_center_boozer endpoint parity (noK)
# ---------------------------------------------------------------------------


def test_trace_gc_no_k_boozer_endpoint_matches_cpp_oracle(
    vacuum_bri_and_jax, event_time_lane
):
    """JAX no_K Boozer endpoint matches ``sopp`` at the lane gate."""

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, jax_field = vacuum_bri_and_jax
    mass = 1.0
    charge = 1.0
    _Ekin, speed_total, v_par = _short_orbit_inputs(modB_init=1.0, mass=mass)
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10

    cpp_endpoint = _run_cpp_oracle(
        bri,
        stz_init,
        mass=mass,
        charge=charge,
        speed_total=speed_total,
        v_par=v_par,
        tmax=tmax,
        tol=tol,
        vacuum=False,
        noK=True,
    )
    jax_endpoint = _run_jax_path(
        jax_field,
        stz_init,
        mass=mass,
        charge=charge,
        speed_total=speed_total,
        v_par=v_par,
        tmax=tmax,
        tol=tol,
        mode="no_k",
    )

    assert np.allclose(
        jax_endpoint[1:5], cpp_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "JAX vs upstream no_K-Boozer endpoint parity failed: "
        f"jax={jax_endpoint}, cpp={cpp_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


# ---------------------------------------------------------------------------
# 4. trace_guiding_center_boozer endpoint parity (full)
# ---------------------------------------------------------------------------


def test_trace_gc_full_boozer_endpoint_matches_cpp_oracle(
    full_bri_and_jax, event_time_lane
):
    """JAX full Boozer endpoint matches ``sopp`` at the lane gate."""

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, jax_field = full_bri_and_jax
    mass = 1.0
    charge = 1.0
    _Ekin, speed_total, v_par = _short_orbit_inputs(modB_init=1.0, mass=mass)
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10

    cpp_endpoint = _run_cpp_oracle(
        bri,
        stz_init,
        mass=mass,
        charge=charge,
        speed_total=speed_total,
        v_par=v_par,
        tmax=tmax,
        tol=tol,
        vacuum=False,
        noK=False,
    )
    jax_endpoint = _run_jax_path(
        jax_field,
        stz_init,
        mass=mass,
        charge=charge,
        speed_total=speed_total,
        v_par=v_par,
        tmax=tmax,
        tol=tol,
        mode="full",
    )

    assert np.allclose(
        jax_endpoint[1:5], cpp_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "JAX vs upstream full-Boozer endpoint parity failed: "
        f"jax={jax_endpoint}, cpp={cpp_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


# ---------------------------------------------------------------------------
# 5. Public wrapper routes through JAX when field is the JAX wrapper
# ---------------------------------------------------------------------------


def test_trace_particles_boozer_jax_routes_when_field_is_jax_wrapper(
    monkeypatch, vacuum_bri_and_jax, event_time_lane
):
    """``trace_particles_boozer`` routes through JAX with the JAX field wrapper.

    The public wrapper must return the same ``(res_tys, res_zeta_hits)``
    shape as the CPU oracle (with each ``res_tys[i]`` row being
    ``[t, s, theta, zeta, v_par]``). The endpoint must match the C++
    oracle at the event-time state-vector lane gate.
    """

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, jax_field = vacuum_bri_and_jax
    mass = 1.0
    charge = 1.0
    Ekin, speed_total, v_par = _short_orbit_inputs(modB_init=1.0, mass=mass)
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10

    # CPU oracle path through the wrapper. The CPU lane consumes the
    # CPU BoozerRadialInterpolant directly.
    cpu_res, cpu_zeta_hits = trace_particles_boozer(
        bri,
        stz_init.reshape((1, 3)),
        [v_par],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        mode="gc_vac",
    )
    cpu_endpoint = cpu_res[0][-1]
    assert len(cpu_zeta_hits) == 1

    # JAX route: flip the backend predicate and pass the JAX field.
    _force_jax_backend(monkeypatch)
    res_tys, res_zeta_hits = trace_particles_boozer(
        jax_field,
        stz_init.reshape((1, 3)),
        [v_par],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        mode="gc_vac",
    )

    # Public shape contract.
    assert isinstance(res_tys, list) and len(res_tys) == 1
    assert isinstance(res_zeta_hits, list) and len(res_zeta_hits) == 1
    jax_traj = res_tys[0]
    assert jax_traj.ndim == 2 and jax_traj.shape[1] == 5
    assert jax_traj.shape[0] >= 2
    # zeta-plane crossings: empty (0, 6) array per particle.
    assert res_zeta_hits[0].shape == (0, 6)

    # Initial state preserved.
    assert np.isclose(jax_traj[0, 0], 0.0)
    assert np.isclose(jax_traj[0, 1], stz_init[0])
    assert np.isclose(jax_traj[0, 2], stz_init[1])
    assert np.isclose(jax_traj[0, 3], stz_init[2])
    assert np.isclose(jax_traj[0, 4], v_par)

    # Endpoint parity at the event-time state-vector lane gate.
    jax_endpoint = jax_traj[-1]
    assert np.allclose(
        jax_endpoint[1:5], cpu_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "trace_particles_boozer JAX vs CPU endpoint parity failed: "
        f"jax={jax_endpoint}, cpu={cpu_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )


def test_trace_particles_boozer_jax_routes_when_field_is_interpolated_wrapper(
    monkeypatch, vacuum_bri_and_jax, event_time_lane
):
    """``trace_particles_boozer`` routes through JAX for interpolated fields."""

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, _radial_jax_field = vacuum_bri_and_jax
    nfp = int(getattr(bri.booz.bx, "nfp", 1))
    degree = 2
    srange = (0.0, 1.0, 5)
    thetarange = (0.0, np.pi, 5)
    zetarange = (0.0, 2.0 * np.pi / nfp, 5)
    cpu_field = InterpolatedBoozerField(
        bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True
    )
    jax_field = InterpolatedBoozerFieldJAX(
        bri,
        degree,
        srange,
        thetarange,
        zetarange,
        True,
        nfp=nfp,
        stellsym=True,
        scalars=_BOOZER_RHS_EVAL_KEYS,
    )

    mass = 1.0
    charge = 1.0
    Ekin, _speed_total, v_par = _short_orbit_inputs(modB_init=1.0, mass=mass)
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10

    cpu_res, cpu_zeta_hits = trace_particles_boozer(
        cpu_field,
        stz_init.reshape((1, 3)),
        [v_par],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        mode="gc_vac",
    )
    assert len(cpu_zeta_hits) == 1

    _force_jax_backend(monkeypatch)
    jax_res, jax_zeta_hits = trace_particles_boozer(
        jax_field,
        stz_init.reshape((1, 3)),
        [v_par],
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        mode="gc_vac",
    )

    assert len(jax_res) == 1
    assert len(jax_zeta_hits) == 1
    assert jax_zeta_hits[0].shape == (0, 6)
    assert np.allclose(
        jax_res[0][-1, 1:5],
        cpu_res[0][-1, 1:5],
        rtol=state_rtol,
        atol=state_atol,
    )


# ---------------------------------------------------------------------------
# 6. Public wrapper rejects unsupported shapes and preserves comm order
# ---------------------------------------------------------------------------


def test_trace_particles_boozer_jax_rejects_unsupported_shapes_and_replays_comm(
    monkeypatch, vacuum_bri_and_jax, assert_two_rank_replay_matches
):
    """The wrapper rejects unsupported shapes and preserves comm gather order.

    ``trace_particles_boozer`` accepts ``mode in {'gc', 'gc_vac', 'gc_nok'}``
    on both the CPU and JAX paths. Any other mode is rejected by the
    public-surface assertion regardless of backend. The JAX route is
    additionally guarded against unsupported CPU ``BoozerMagneticField``
    instances and unsupported stopping-criterion subclasses. ``comm=`` now
    preserves the CPU wrapper's host-level split/gather semantics.
    """

    bri, jax_field = vacuum_bri_and_jax
    stz_init = _make_initial_point().reshape((1, 3))

    _force_jax_backend(monkeypatch)

    # Unsupported mode: rejected at the public ``assert mode in [...]``
    # before the JAX router runs.
    with pytest.raises(AssertionError):
        trace_particles_boozer(
            jax_field,
            stz_init,
            [1.0e3],
            tmax=1e-6,
            tol=1e-9,
            mode="full",
        )

    # Wrong field type with the JAX backend active: the JAX router
    # rejects the CPU BoozerRadialInterpolant explicitly instead of
    # silently falling through to the C++ oracle.
    with pytest.raises(NotImplementedError, match="BoozerRadialInterpolantJAX"):
        trace_particles_boozer(
            bri,
            stz_init,
            [1.0e3],
            tmax=1e-6,
            tol=1e-9,
            mode="gc_vac",
        )

    # Unsupported stopping-criterion subclass (raw Python object that
    # isn't an isinstance match for any known sopp.* / Python wrapper)
    # is still rejected on the JAX Boozer route. ``zetas`` is now
    # supported on the JAX route; non-empty values flow through to the
    # ``trace_guiding_center_boozer`` driver and populate the
    # ``res_zeta_hits`` output.
    with pytest.raises(NotImplementedError, match="stopping criterion"):
        trace_particles_boozer(
            jax_field,
            stz_init,
            [1.0e3],
            tmax=1e-6,
            tol=1e-9,
            mode="gc_vac",
            stopping_criteria=[object()],
        )

    stz_pair = np.vstack(
        (
            stz_init[0],
            stz_init[0] + np.array([0.02, 0.01, 0.0], dtype=np.float64),
        )
    )

    no_comm_tys, no_comm_hits = trace_particles_boozer(
        jax_field,
        stz_pair,
        [1.0e3, 0.9e3],
        tmax=1e-6,
        tol=1e-9,
        mode="gc_vac",
    )
    assert_two_rank_replay_matches(
        no_comm_tys,
        no_comm_hits,
        lambda comm: trace_particles_boozer(
            jax_field,
            stz_pair,
            [1.0e3, 0.9e3],
            tmax=1e-6,
            tol=1e-9,
            mode="gc_vac",
            comm=comm,
        ),
    )
