"""Tests for the optional diffrax fieldline backend (``integrator="dopri5_diffrax"``).

The backend delegates adaptive Dopri5 stepping to diffrax while reusing the native
crossing/stopping/output machinery, so it must satisfy the same
``FieldlineTracingResult`` contract as ``trace_fieldline`` (``dopri5_native``).

Test design notes:
- Tight cross-backend equality is asserted on the no-stopping-criterion path
  (both engines integrate the same ODE; they agree to integrator tolerance).
- On a stopping exit the firing step is an ``idx < 0`` event row, not a live
  trajectory row. Stopping tests assert the status taxonomy and event-row geometry.
- The "did it fire" verdict is asserted from independent geometry (e.g. ``hypot(x,y)``),
  never by re-running the same predicate the backend uses (which would be tautological).
"""

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import replace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_REPO_ROOT / "src")

from simsopt_jax.core.tracing import (
    FieldlineTracingSpec,
    IterStoppingCriterion,
    LevelsetStoppingCriterion,
    MaxRStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    trace_fieldline,
    trace_fieldlines_batched,
)
from simsopt_jax.core.tracing_diffrax import trace_fieldline_diffrax
from simsopt_jax_adapters.field.tracing import (
    compute_fieldlines as adapter_compute_fieldlines,
    trace_particles as adapter_trace_particles,
    trace_particles_boozer as adapter_trace_particles_boozer,
)

_PHIS = jnp.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])


def _helix_field(point):
    """Smooth analytic field: winds in (x,y), drifts in z, radius grows slowly."""
    x, y, _z = point[0], point[1], point[2]
    return jnp.array([-y + 0.05 * x, x + 0.05 * y, 0.30])


def _pure_toroidal_field(point):
    """B = (-y, x, 0): a fieldline at z=0 is a circle of constant radius."""
    x, y, _z = point[0], point[1], point[2]
    return jnp.array([-y, x, 0.0])


def _field_inward(point):
    """Helix whose radius SHRINKS (drives a fieldline toward a MinR boundary)."""
    x, y, _z = point[0], point[1], point[2]
    return jnp.array([-y - 0.05 * x, x - 0.05 * y, 0.30])


def _field_downward(point):
    """Helix whose z DECREASES (drives a fieldline toward a MinZ boundary)."""
    x, y, _z = point[0], point[1], point[2]
    return jnp.array([-y + 0.05 * x, x + 0.05 * y, -0.30])


def _discontinuous_x_field(point):
    """Rejected-step-heavy field for the native iteration-count contract."""
    speed = jnp.where(point[0] < 0.2, 1.0, -1.0)
    return jnp.array([speed, 0.0, 0.0], dtype=jnp.float64)


def _spec(integrator, **kw):
    base = dict(tmax=15.0, rtol=1e-9, atol=1e-9, max_steps=4000, max_phi_hits=128)
    base.update(kw)
    return FieldlineTracingSpec(integrator=integrator, **base)


def _endpoint(result):
    return np.array(result.trajectory[int(result.steps_taken)])


# ── Cross-backend parity (no stopping criterion): tight equality is legitimate ──


def test_no_stop_parity_matches_native():
    """diffrax and native agree to integrator tolerance when nothing terminates early."""
    y0 = jnp.array([1.0, 0.0, 0.0])
    rn = trace_fieldline(_spec("dopri5_native"), y0, _helix_field, phis=_PHIS)
    rd = trace_fieldline(_spec("dopri5_diffrax"), y0, _helix_field, phis=_PHIS)

    assert int(rn.status) == 0 and int(rd.status) == 0
    np.testing.assert_allclose(_endpoint(rd), _endpoint(rn), rtol=0, atol=1e-6)
    assert int(rd.phi_hits_count) == int(rn.phi_hits_count)
    nd, nn = int(rd.phi_hits_count), int(rn.phi_hits_count)
    hd = np.array(rd.phi_hits[:nd])[np.argsort(np.array(rd.phi_hits[:nd, 0]))]
    hn = np.array(rn.phi_hits[:nn])[np.argsort(np.array(rn.phi_hits[:nn, 0]))]
    np.testing.assert_allclose(hd, hn, rtol=0, atol=1e-6)


def test_pure_toroidal_preserves_radius():
    """Physics invariant: a z=0 fieldline of B=(-y,x,0) keeps |r| constant (non-tautological)."""
    y0 = jnp.array([1.3, 0.0, 0.0])
    rd = trace_fieldline_diffrax(
        _spec("dopri5_diffrax", tmax=8.0), y0, _pure_toroidal_field, phis=_PHIS
    )
    live = np.array(rd.trajectory)[np.array(rd.mask)]
    radii = np.hypot(live[:, 1], live[:, 2])
    np.testing.assert_allclose(radii, 1.3, rtol=0, atol=1e-7)
    np.testing.assert_allclose(live[:, 3], 0.0, atol=1e-9)  # z stays 0


# ── Geometric stopping criteria: status + boundary geometry (not bit-parity) ──


@pytest.mark.parametrize(
    ("criterion", "field", "boundary_fn", "target"),
    [
        (MaxRStoppingCriterion(crit_r=1.20), _helix_field, lambda e: np.hypot(e[1], e[2]), 1.20),
        (MinRStoppingCriterion(crit_r=0.80), _field_inward, lambda e: np.hypot(e[1], e[2]), 0.80),
        (MaxZStoppingCriterion(crit_z=0.90), _helix_field, lambda e: e[3], 0.90),
        (MinZStoppingCriterion(crit_z=-0.90), _field_downward, lambda e: e[3], -0.90),
    ],
)
def test_geometric_criterion_fires_on_boundary(criterion, field, boundary_fn, target):
    """Each criterion fires with status -1 and the terminal state sits on its boundary.

    Seed r0=1.0, z0=0.0 is inside the box, so no criterion is satisfied at t0; the
    chosen ``field`` drives the line toward the criterion's boundary.
    """
    y0 = jnp.array([1.0, 0.0, 0.0])
    spec = _spec("dopri5_diffrax", tmax=200.0, max_steps=8000)
    rd = trace_fieldline(spec, y0, field, phis=_PHIS, stopping_criteria=(criterion,))

    assert int(rd.status) == -1, f"expected status -1, got {int(rd.status)}"
    # the idx<0 stopping-criterion row is recorded in phi_hits (contract requires it)
    nd = int(rd.phi_hits_count)
    hit_rows = np.array(rd.phi_hits[:nd])
    idx_col = hit_rows[:, 1]
    assert (idx_col < 0).any(), f"missing idx<0 stop row; idx col = {idx_col}"
    assert int(idx_col[idx_col < 0][0]) == -1
    stop_row = hit_rows[idx_col < 0][0]
    np.testing.assert_allclose(float(rd.t_final), stop_row[0], rtol=0, atol=1e-12)
    stop_state = np.concatenate([[stop_row[0]], stop_row[2:5]])
    # terminal event geometry on the boundary to step-granularity
    assert abs(boundary_fn(stop_state) - target) < 2e-2


def test_levelset_criterion_fires():
    """Levelset (stop when classifier < 0) fires and records the stop row."""
    # smooth signed classifier: > 0 inside the disk r < 1.15, < 0 outside
    def classifier(positions):
        r = jnp.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        return 1.15 - r

    crit = LevelsetStoppingCriterion(classifier_fn=classifier)
    y0 = jnp.array([1.0, 0.0, 0.0])  # inside
    spec = _spec("dopri5_diffrax", tmax=200.0, max_steps=8000)
    rd = trace_fieldline(spec, y0, _helix_field, phis=_PHIS, stopping_criteria=(crit,))
    assert int(rd.status) == -1
    nd = int(rd.phi_hits_count)
    assert (np.array(rd.phi_hits[:nd, 1]) < 0).any()


def test_phi_hits_overflow_signalled_by_count():
    """phi_hits_count tallies ALL crossings even when the fixed buffer is smaller."""
    y0 = jnp.array([1.0, 0.0, 0.0])
    spec = _spec("dopri5_diffrax", tmax=40.0, max_steps=8000, max_phi_hits=4)
    rd = trace_fieldline_diffrax(spec, y0, _helix_field, phis=_PHIS)
    assert int(rd.phi_hits_count) > 4  # many windings in tmax=40 -> overflow
    assert tuple(rd.phi_hits.shape) == (4, 5)  # buffer stays bounded


def test_seed_already_outside_records_first_accepted_stop_event():
    """A criterion satisfied at the seed follows the native post-step contract."""
    crit = MaxZStoppingCriterion(crit_z=0.5)
    y0 = jnp.array([1.0, 0.0, 1.0])  # z0=1.0 already >= 0.5
    rd = trace_fieldline(
        _spec("dopri5_diffrax", tmax=5.0), y0, _helix_field, stopping_criteria=(crit,)
    )
    assert int(rd.status) == -1
    assert int(rd.steps_taken) == 0
    assert float(rd.t_final) > 0.0
    np.testing.assert_allclose(_endpoint(rd), [0.0, 1.0, 0.0, 1.0], atol=1e-12)
    hit_rows = np.asarray(rd.phi_hits)[: int(rd.phi_hits_count)]
    assert hit_rows.shape == (1, 5)
    assert int(hit_rows[0, 1]) == -1
    np.testing.assert_allclose(float(rd.t_final), hit_rows[0, 0], rtol=0, atol=1e-12)


# ── Batched / vmap correctness (the production path) ──


def test_vmap_batch_equals_per_lane():
    """trace_fieldlines_batched == a per-lane loop over trace_fieldline_diffrax."""
    y0s = jnp.stack([jnp.array([r, 0.0, 0.0]) for r in [0.7, 0.9, 1.0, 1.1, 1.3]])
    dtmaxs = jnp.full((5,), 0.5)
    crit = (MaxRStoppingCriterion(crit_r=1.6),)
    spec = _spec("dopri5_diffrax", tmax=200.0, max_steps=8000)
    batched = trace_fieldlines_batched(spec, y0s, dtmaxs, _helix_field, phis=_PHIS,
                                       stopping_criteria=crit)
    for i in range(5):
        one = trace_fieldline_diffrax(replace(spec, dtmax=0.5), y0s[i], _helix_field,
                                      phis=_PHIS, stopping_criteria=crit)
        # discrete contract fields are bit-for-bit identical between the batched and
        # per-lane paths:
        assert int(batched.status[i]) == int(one.status)
        assert int(batched.steps_taken[i]) == int(one.steps_taken)
        assert int(batched.phi_hits_count[i]) == int(one.phi_hits_count)
        # the continuous endpoint differs only by XLA vmap-vs-scalar FP reassociation
        # (inherent, grows with field stiffness, ~1e-7 worst-case) -- assert to 1e-7,
        # still far tighter than any documented native/diffrax divergence (~1e-2).
        np.testing.assert_allclose(
            np.array(batched.trajectory[i, int(batched.steps_taken[i])]),
            _endpoint(one), rtol=0, atol=1e-7,
        )


# ── Contract: dtype, error handling, optional dependency ──


def test_steps_taken_and_status_are_int32():
    """Contract dtypes match native (jnp.sum promotes to int64 under x64 if uncast)."""
    rd = trace_fieldline_diffrax(_spec("dopri5_diffrax"), jnp.array([1.0, 0.0, 0.0]),
                                 _helix_field, phis=_PHIS)
    assert rd.steps_taken.dtype == jnp.int32
    assert rd.status.dtype == jnp.int32


def test_unknown_integrator_raises():
    with pytest.raises(ValueError, match="Unknown integrator"):
        trace_fieldline(replace(_spec("dopri5_native"), integrator="rk45_bogus"),
                        jnp.array([1.0, 0.0, 0.0]), _helix_field)


@pytest.mark.parametrize(
    ("criterion", "field", "tmax"),
    [
        (ToroidalTransitStoppingCriterion(max_transits=0.1), _pure_toroidal_field, 2.0),
    ],
)
def test_transit_criterion_is_supported_by_diffrax(criterion, field, tmax):
    rd = trace_fieldline_diffrax(
        _spec("dopri5_diffrax", tmax=tmax, max_steps=4000),
        jnp.array([1.0, 0.0, 0.0]),
        field,
        stopping_criteria=(criterion,),
    )
    assert int(rd.status) == -1
    hit_rows = np.asarray(rd.phi_hits)[: int(rd.phi_hits_count)]
    assert hit_rows.shape[0] == 1
    assert int(hit_rows[0, 1]) == -1


def test_direct_diffrax_rejects_iteration_criterion():
    with pytest.raises(NotImplementedError, match="IterStoppingCriterion"):
        trace_fieldline_diffrax(
            _spec("dopri5_diffrax"),
            jnp.array([1.0, 0.0, 0.0]),
            _helix_field,
            stopping_criteria=(IterStoppingCriterion(max_iter=0),),
        )


def test_default_diffrax_falls_back_to_native_for_iteration_criterion():
    crit = (IterStoppingCriterion(max_iter=50),)
    y0 = jnp.array([0.0, 0.0, 0.0])
    common = dict(tmax=1.0, rtol=1e-12, atol=1e-12, max_steps=400)
    native = trace_fieldline(
        FieldlineTracingSpec(integrator="dopri5_native", **common),
        y0,
        _discontinuous_x_field,
        stopping_criteria=crit,
    )
    default = trace_fieldline(
        FieldlineTracingSpec(**common),
        y0,
        _discontinuous_x_field,
        stopping_criteria=crit,
    )

    assert int(default.status) == int(native.status) == -1
    assert int(default.steps_taken) == int(native.steps_taken)
    np.testing.assert_allclose(float(default.t_final), float(native.t_final), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(default.phi_hits), np.asarray(native.phi_hits))


@pytest.mark.parametrize(
    "criterion",
    [
        MinToroidalFluxStoppingCriterion(min_s=0.1),
        MaxToroidalFluxStoppingCriterion(max_s=0.9),
    ],
)
def test_cartesian_flux_criteria_remain_inactive(criterion):
    rd = trace_fieldline_diffrax(
        _spec("dopri5_diffrax", tmax=1.0),
        jnp.array([1.0, 0.0, 0.0]),
        _helix_field,
        stopping_criteria=(criterion,),
    )
    assert int(rd.status) == 0
    hit_rows = np.asarray(rd.phi_hits)[: int(rd.phi_hits_count)]
    assert not (hit_rows[:, 1] < 0).any()


def test_phis_none_records_no_crossings():
    rd = trace_fieldline_diffrax(_spec("dopri5_diffrax"), jnp.array([1.0, 0.0, 0.0]),
                                 _helix_field, phis=None)
    assert int(rd.phi_hits_count) == 0
    assert int(rd.status) == 0


@pytest.mark.parametrize("bad", ["max_steps", "max_phi_hits"])
def test_nonpositive_size_guards(bad):
    spec = _spec("dopri5_diffrax", **{bad: 0})
    with pytest.raises(ValueError, match=bad):
        trace_fieldline_diffrax(spec, jnp.array([1.0, 0.0, 0.0]), _helix_field)


def test_native_tracing_import_does_not_require_diffrax():
    """The native path must not import diffrax (it is an optional extra)."""
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r});"
        "from repo_bootstrap import bootstrap_local_simsopt;"
        f"bootstrap_local_simsopt({str(_REPO_ROOT / 'src')!r});"
        "import jax; jax.config.update('jax_enable_x64', True);"
        "import simsopt_jax.core.tracing;"
        "assert 'diffrax' not in sys.modules, 'native path pulled in diffrax';"
        "print('OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr


# ── Public adapter routing ──


class _ToyJaxField:
    """Minimal JAX field exposing ``jax_B_at`` for the public adapter path."""

    def jax_B_at(self, point):
        return _helix_field(point)


def test_adapter_compute_fieldlines_selects_diffrax_backend():
    """Public ``compute_fieldlines(integrator=...)`` routes to the diffrax backend and
    agrees with the native backend on the no-stop endpoint."""
    field = _ToyJaxField()
    R0, Z0 = [1.0, 1.05], [0.0, 0.0]
    kw = dict(tmax=10.0, tol=1e-9, phis=[0.0, np.pi])
    tys_n, hits_n = adapter_compute_fieldlines(field, R0, Z0, integrator="dopri5_native", **kw)
    tys_d, hits_d = adapter_compute_fieldlines(field, R0, Z0, integrator="dopri5_diffrax", **kw)
    assert len(tys_n) == len(tys_d) == 2
    assert len(hits_n) == len(hits_d) == 2
    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(tys_d[i])[-1, 1:], np.asarray(tys_n[i])[-1, 1:], rtol=0, atol=1e-5
        )


def test_adapter_max_steps_param_controls_trace_length():
    """`max_steps` is a real knob: a larger cap traces more transits → more crossings.

    Pins the fix for the truncated/sparse-Poincare bug (the adapter used to hard-code
    max_steps=4000, cutting every line off mid-flight).
    """
    field = _ToyJaxField()
    R0, Z0 = [1.0], [0.0]
    kw = dict(tmax=60.0, tol=1e-7, phis=[0.0, np.pi],
              integrator="dopri5_diffrax", max_phi_hits=2048)
    _, hits_small = adapter_compute_fieldlines(field, R0, Z0, max_steps=200, **kw)
    _, hits_big = adapter_compute_fieldlines(field, R0, Z0, max_steps=3000, **kw)
    # max_steps=200 truncates well before tmax; 3000 reaches it -> strictly more crossings
    assert len(hits_big[0]) > len(hits_small[0]), (len(hits_small[0]), len(hits_big[0]))


def test_particle_wrapper_integrator_preserves_existing_positional_order():
    """Adding ``integrator`` must not shift existing positional particle args."""
    particle_params = list(inspect.signature(adapter_trace_particles).parameters)
    assert particle_params.index("integrator") > particle_params.index("phase_angle")

    boozer_params = list(inspect.signature(adapter_trace_particles_boozer).parameters)
    assert boozer_params.index("integrator") > boozer_params.index("forget_exact_path")
