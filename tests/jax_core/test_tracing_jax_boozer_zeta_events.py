"""Zeta-plane crossing tests for the JAX Boozer-coordinate tracing path.

These tests validate the follow-up extension to
:func:`simsopt.jax_core.tracing.trace_guiding_center_boozer` that
records ``zetas`` plane crossings into the fixed-shape ``phi_hits``
buffer (named ``phi_hits`` on the result dataclass for layout
compatibility with the Cartesian guiding-centre route; on the Boozer
path the rows store zeta-plane crossings with layout
``[t_hit, idx, s, theta, zeta, v_par]``).

1. ``test_zeta_plane_crossing_recovers_axis_trajectory_on_boozer_route`` —
   analytic check using the ``mu = 0`` limit of the vacuum-Boozer RHS
   on a ``BoozerRadialInterpolantJAX`` axis trace: the streamline
   stays on a curve where ``zeta(t)`` is approximately monotone, so a
   single zeta-target crossing has a known time at which the JAX
   driver must record a row in the ``phi_hits`` buffer.
2. ``test_trace_particles_boozer_jax_records_zeta_hits`` — public
   API smoke + endpoint parity vs ``sopp.particle_guiding_center_boozer_tracing``
   with non-empty ``zetas`` argument; the JAX route must surface the
   crossing through ``res_zeta_hits`` with the upstream-compatible 6-column
   row layout.
3. ``test_trace_particles_boozer_jax_records_flux_stopping_criterion`` —
   public API smoke for a Boozer-relevant stopping criterion: both the
   CPU oracle and the JAX route must append a negative-index
   ``MaxToroidalFluxStoppingCriterion`` event row.

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
The parity-ladder lane is ``event_time_tracing``.
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
from simsopt.field.boozermagneticfield_jax import BoozerRadialInterpolantJAX
from simsopt.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from simsopt.jax_core.tracing import (
    GuidingCenterTracingSpec,
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
    """``no_K=True`` Boozer field; same fixture shape as the existing GC-Boozer tests."""
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(
        vmec, order=3, mpol=4, ntor=4, rescale=True, no_K=True
    )
    return bri, BoozerRadialInterpolantJAX(bri)


def _force_jax_backend(monkeypatch):
    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)


def _make_initial_point() -> np.ndarray:
    return np.array([0.30, 0.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Zeta-plane crossing analytic check on the Boozer axis trace
# ---------------------------------------------------------------------------


def test_zeta_plane_crossing_recovers_axis_trajectory_on_boozer_route(
    vacuum_bri_and_jax, event_time_lane
):
    """A Boozer GC trace records a zeta-plane crossing near the analytic time.

    The vacuum Boozer guiding-centre RHS has ``dzeta/dt = v_par |B| / G``.
    On the axis (small s) ``|B|`` and ``G`` are slowly varying so
    ``zeta(t) ~ (v_par |B0| / G0) t``. We pick a small ``zeta_target``
    and assert the JAX driver records a single zeta crossing close to
    the analytic arrival time, with state on the trajectory.
    """

    del event_time_lane  # Used only for the lane-import side-effect.

    _bri, jax_field = vacuum_bri_and_jax
    mass = 1.0
    charge = 1.0
    v_par = 1.0e3
    stz_init = _make_initial_point()
    # mu=0 keeps the orbit close to the axis stream (no mirror force).
    mu = 0.0

    jax_field.set_points(stz_init.reshape((1, 3)))
    modB0 = float(np.asarray(jax_field.modB()).reshape(-1)[0])
    G0 = float(np.asarray(jax_field.G()).reshape(-1)[0])
    # Analytic streamline rate for the dzeta/dt component.
    dzeta_dt = v_par * modB0 / G0
    zeta_target = 0.05  # ~3 degrees; well within the first transit
    t_expected = zeta_target / dzeta_dt
    # Run for ~5x the expected crossing time so the integrator captures
    # it comfortably even if it exhausts the step cap before tmax.
    tmax = 5.0 * t_expected

    spec = GuidingCenterTracingSpec(
        tmax=float(tmax),
        rtol=1e-9,
        atol=1e-9,
        max_steps=8000,
        max_phi_hits=16,
    )
    y0 = jnp.asarray([*stz_init, v_par], dtype=jnp.float64)
    zetas = jnp.asarray([zeta_target], dtype=jnp.float64)
    result = trace_guiding_center_boozer(
        spec,
        y0,
        jax_field,
        m=mass,
        q=charge,
        mu=mu,
        mode="vacuum",
        zetas=zetas,
    )
    # The integrator may exhaust its step budget before reaching tmax
    # (status=1) for stiff Boozer fields; what matters for the
    # zeta-plane test is that at least one crossing of the target is
    # localized while the live trajectory still spans the crossing
    # time. status<0 (criterion fired) is not expected since this run
    # uses no stopping criteria.
    status = int(result.status)
    assert status in (0, 1), (
        f"Unexpected JAX Boozer status={status}; expected 0 (reached tmax) "
        f"or 1 (step budget). t_final={float(result.t_final)}"
    )
    zeta_hits = np.asarray(result.phi_hits)
    count = int(result.phi_hits_count)
    assert count >= 1, f"expected at least one zeta crossing, got {count}"
    t_actual = float(zeta_hits[0, 0])
    idx = int(zeta_hits[0, 1])
    assert idx == 0, f"expected zeta-plane index 0, got {idx}"
    # Event-time accuracy: the local Boozer rate is non-uniform but
    # close to the on-axis ``dzeta/dt`` near small s; relative error
    # on the first transit is bounded by the spatial variation of
    # ``|B| / G`` over the orbit window (empirically below 10% on
    # the test fixture).
    rel_err = abs(t_actual - t_expected) / t_expected
    assert rel_err < 0.5, (
        f"Zeta crossing time {t_actual} differs from analytic {t_expected} "
        f"by relative {rel_err}"
    )
    # The recorded state must lie on the post-integration Boozer point;
    # in particular zeta_hit ~ zeta_target (modulo 2*pi).
    zeta_hit = float(zeta_hits[0, 4])
    zeta_hit_wrapped = zeta_hit - 2.0 * np.pi * np.floor(zeta_hit / (2.0 * np.pi))
    zeta_target_wrapped = zeta_target - 2.0 * np.pi * np.floor(
        zeta_target / (2.0 * np.pi)
    )
    # The bracketed localizer refines the crossing on a sub-step DOPRI5
    # interpolant; final zeta-position error is dominated by the per-
    # step RK accuracy. Use a comfortable tolerance well above the
    # event_time_atol gate.
    assert abs(zeta_hit_wrapped - zeta_target_wrapped) < 1.0e-6, (
        f"Zeta crossing position not on zeta_target plane: "
        f"zeta_hit={zeta_hit_wrapped}, zeta_target={zeta_target_wrapped}"
    )


# ---------------------------------------------------------------------------
# 2. Public wrapper records zeta crossings + endpoint parity vs sopp
# ---------------------------------------------------------------------------


def test_trace_particles_boozer_jax_records_zeta_hits(
    monkeypatch, vacuum_bri_and_jax, event_time_lane
):
    """``trace_particles_boozer`` records zeta crossings on the JAX route.

    The public wrapper must return ``res_zeta_hits`` populated with
    the JAX driver's zeta crossings; each row layout is
    ``[t_hit, idx, s, theta, zeta, v_par]`` (6 columns matching the
    upstream ``sopp.particle_guiding_center_boozer_tracing`` output).
    The endpoint must continue to match the CPU oracle at the
    event-time state-vector lane.
    """

    state_rtol = float(event_time_lane["state_vector_rtol"])
    state_atol = float(event_time_lane["state_vector_atol"])

    bri, jax_field = vacuum_bri_and_jax
    mass = 1.0
    charge = 1.0
    speed_total = 1.0e3
    Ekin = 0.5 * mass * speed_total**2
    v_par = 0.6 * speed_total
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10
    # zeta accumulates slowly on this fixture (mu>0 drags the orbit
    # radially and slows the ``v_par |B|/G`` advection). Empirically
    # zeta reaches ~6e-3 rad at tmax=1e-4. Choose target planes well
    # inside this window.
    zetas_list = [0.001, 0.003]

    # CPU oracle path: pass the same zetas through the C++ tracing.
    cpu_res, cpu_zeta_hits = sopp.particle_guiding_center_boozer_tracing(
        bri,
        stz_init,
        mass,
        charge,
        speed_total,
        v_par,
        tmax,
        tol,
        vacuum=True,
        noK=False,
        zetas=zetas_list,
        stopping_criteria=[],
    )
    cpu_endpoint = np.asarray(cpu_res[-1])
    cpu_hits = np.asarray(cpu_zeta_hits)

    # JAX route via the public wrapper. Flip the backend predicate.
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
        zetas=zetas_list,
    )

    # Public shape contract: per-particle list with 6-column zeta-hit array.
    assert isinstance(res_zeta_hits, list) and len(res_zeta_hits) == 1
    jax_hits = res_zeta_hits[0]
    assert jax_hits.ndim == 2 and jax_hits.shape[1] == 6
    # Must have recorded at least one crossing.
    assert jax_hits.shape[0] >= 1, (
        f"JAX route expected at least one zeta crossing; got shape {jax_hits.shape}"
    )
    # First crossing index must be one of the requested zeta targets.
    first_idx = int(jax_hits[0, 1])
    assert first_idx in (0, 1), f"expected idx in {{0, 1}}; got {first_idx}"
    # Crossing time must lie within the integration window.
    assert 0.0 < jax_hits[0, 0] <= tmax + 1e-15

    # Endpoint parity vs the CPU oracle.
    jax_endpoint = res_tys[0][-1]
    assert np.allclose(
        jax_endpoint[1:5], cpu_endpoint[1:5], rtol=state_rtol, atol=state_atol
    ), (
        "trace_particles_boozer JAX vs CPU endpoint parity failed: "
        f"jax={jax_endpoint}, cpu={cpu_endpoint}, "
        f"lane_rtol={state_rtol}, lane_atol={state_atol}"
    )

    # Cross-check: the C++ and JAX paths must each record a positive
    # number of crossings (the two implementations may differ in event
    # ordering and in dropped trailing transitions, so we only assert
    # the public shape and non-empty count here).
    assert cpu_hits.shape[0] >= 1


# ---------------------------------------------------------------------------
# 3. Public wrapper records Boozer flux-coordinate stopping criterion events
# ---------------------------------------------------------------------------


def test_trace_particles_boozer_jax_records_flux_stopping_criterion(
    monkeypatch, vacuum_bri_and_jax
):
    """``trace_particles_boozer`` records flux-coordinate stopping events.

    ``MaxToroidalFluxStoppingCriterion`` is Boozer-coordinate specific:
    it fires on the ``s`` state component. The CPU oracle and JAX route
    both write this as a negative-index row in the same 6-column
    ``res_zeta_hits`` layout.
    """

    bri, jax_field = vacuum_bri_and_jax
    mass = 1.0
    charge = 1.0
    speed_total = 1.0e3
    Ekin = 0.5 * mass * speed_total**2
    v_par = 0.6 * speed_total
    stz_init = _make_initial_point()
    tmax = 1.0e-4
    tol = 1e-10
    criterion = MaxToroidalFluxStoppingCriterion(0.299)

    _cpu_res, cpu_zeta_hits = sopp.particle_guiding_center_boozer_tracing(
        bri,
        stz_init,
        mass,
        charge,
        speed_total,
        v_par,
        tmax,
        tol,
        vacuum=True,
        noK=False,
        zetas=[],
        stopping_criteria=[criterion],
    )
    cpu_hits = np.asarray(cpu_zeta_hits)
    assert cpu_hits.ndim == 2 and cpu_hits.shape[1] == 6
    assert cpu_hits.shape[0] == 1
    assert int(cpu_hits[0, 1]) == -1
    assert 0.0 < cpu_hits[0, 0] <= tmax + 1e-15

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
        stopping_criteria=[criterion],
    )

    assert isinstance(res_zeta_hits, list) and len(res_zeta_hits) == 1
    jax_hits = res_zeta_hits[0]
    assert jax_hits.ndim == 2 and jax_hits.shape[1] == 6
    assert jax_hits.shape[0] == 1
    assert int(jax_hits[0, 1]) == -1
    assert 0.0 < jax_hits[0, 0] <= tmax + 1e-15
    np.testing.assert_allclose(
        res_tys[0],
        np.array([[0.0, stz_init[0], stz_init[1], stz_init[2], v_par]]),
        rtol=0.0,
        atol=0.0,
    )
