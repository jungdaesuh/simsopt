"""Tests for the optional diffrax fieldline backend (``integrator="dopri5_diffrax"``).

The backend delegates adaptive Dopri5 stepping + event termination to diffrax while
reusing the native crossing/stopping/output machinery, so it must satisfy the same
``FieldlineTracingResult`` contract as ``trace_fieldline`` (``dopri5_native``).

Test design notes (these encode the *intended* divergences, not bugs):
- Tight cross-backend equality is asserted ONLY on the no-stopping-criterion path
  (both engines integrate the same ODE; they agree to integrator tolerance).
- On a stopping exit the backends diverge by ~one adaptive step (native does not record
  the firing step; diffrax records the step where the criterion is met). So stopping
  tests assert the *status taxonomy* and the *geometry* (terminal state on the boundary
  to loose tolerance), NOT bit-equality of the terminal state.
- The "did it fire" verdict is asserted from independent geometry (e.g. ``hypot(x,y)``),
  never by re-running the same predicate the backend uses (which would be tautological).
"""

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
    _stopping_criterion_should_stop,
    trace_fieldline,
    trace_fieldlines_batched,
)
from simsopt_jax.core.tracing_diffrax import (
    _criterion_event_margin,
    _fired_index,
    trace_fieldline_diffrax,
)
from simsopt_jax_adapters.field.tracing import (
    compute_fieldlines as adapter_compute_fieldlines,
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
    # terminal geometry on the boundary to step-granularity (NOT re-using the predicate)
    assert abs(boundary_fn(_endpoint(rd)) - target) < 2e-2
    # the idx<0 stopping-criterion row is recorded in phi_hits (contract requires it)
    nd = int(rd.phi_hits_count)
    idx_col = np.array(rd.phi_hits[:nd, 1])
    assert (idx_col < 0).any(), f"missing idx<0 stop row; idx col = {idx_col}"
    assert int(idx_col[idx_col < 0][0]) == -1


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


def test_fired_index_credits_the_criterion_that_actually_crossed():
    """With two near-coincident boundaries, credit the one with margin <= 0, not the
    near one whose margin is still positive (regression for a firing-index slack bug).

    Unit-level (not end-to-end) on purpose: the boundaries are ~5e-7 apart, far below the
    grid jitter of a full trace, so an end-to-end test here would be flaky.
    """
    # MaxZ(idx0) boundary 5e-7 ABOVE z=0.5 -> margin +5e-7 (NOT fired);
    # MaxR(idx1) boundary below r=1.01    -> margin -0.01   (fired).
    crit = (MaxZStoppingCriterion(crit_z=0.5 + 5e-7), MaxRStoppingCriterion(crit_r=1.0))
    x, y, z = jnp.array(1.01), jnp.array(0.0), jnp.array(0.5)
    any_fired, index = _fired_index(crit, x, y, z, jnp.float64)
    assert bool(any_fired)
    assert int(index) == 1, f"expected idx 1 (MaxR crossed), got {int(index)} (MaxZ did not)"


def test_phi_hits_overflow_signalled_by_count():
    """phi_hits_count tallies ALL crossings even when the fixed buffer is smaller."""
    y0 = jnp.array([1.0, 0.0, 0.0])
    spec = _spec("dopri5_diffrax", tmax=40.0, max_steps=8000, max_phi_hits=4)
    rd = trace_fieldline_diffrax(spec, y0, _helix_field, phis=_PHIS)
    assert int(rd.phi_hits_count) > 4  # many windings in tmax=40 -> overflow
    assert tuple(rd.phi_hits.shape) == (4, 5)  # buffer stays bounded


def test_seed_already_outside_stops_at_seed():
    """A criterion satisfied AT the seed stops at the seed (diffrax can't fire on step 1)."""
    crit = MaxZStoppingCriterion(crit_z=0.5)
    y0 = jnp.array([1.0, 0.0, 1.0])  # z0=1.0 already >= 0.5
    rd = trace_fieldline(
        _spec("dopri5_diffrax", tmax=5.0), y0, _helix_field, stopping_criteria=(crit,)
    )
    assert int(rd.status) == -1
    assert int(rd.steps_taken) == 0
    assert float(rd.t_final) == 0.0
    np.testing.assert_allclose(_endpoint(rd), [0.0, 1.0, 0.0, 1.0], atol=1e-12)


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
    "criterion",
    [
        IterStoppingCriterion(max_iter=5),
        ToroidalTransitStoppingCriterion(max_transits=3.0),
        MinToroidalFluxStoppingCriterion(min_s=0.1),
        MaxToroidalFluxStoppingCriterion(max_s=0.9),
    ],
)
def test_unsupported_criterion_raises(criterion):
    with pytest.raises(NotImplementedError, match="dopri5_native"):
        trace_fieldline_diffrax(_spec("dopri5_diffrax"), jnp.array([1.0, 0.0, 0.0]),
                                _helix_field, stopping_criteria=(criterion,))


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


# ── SSOT guard: the diffrax margin and the native predicate must agree (B6) ──


@pytest.mark.parametrize(
    "criterion",
    [
        MinRStoppingCriterion(crit_r=0.9),
        MaxRStoppingCriterion(crit_r=1.1),
        MinZStoppingCriterion(crit_z=-0.2),
        MaxZStoppingCriterion(crit_z=0.2),
    ],
)
def test_event_margin_matches_native_predicate(criterion):
    """sign(margin) <= 0  ⟺  native ``_stopping_criterion_should_stop``.

    The continuous diffrax margin and the boolean native predicate are two encodings
    of the same firing condition; this pins them so a future edit to one cannot drift
    from the other silently.
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.5, 1.5, size=(200, 3))
    dummy_iter = jnp.array(0, dtype=jnp.int32)
    dummy_phi = jnp.array(0.0)
    for p in pts:
        x, y, z = (jnp.array(v) for v in p)
        margin = _criterion_event_margin(criterion, x, y, z, jnp.float64)
        predicate = _stopping_criterion_should_stop(
            criterion, x, y, z, dummy_iter, dummy_phi, dummy_phi, jnp.float64
        )
        assert bool(margin <= 0.0) == bool(predicate), (
            f"margin/predicate disagree at {p} for {type(criterion).__name__}"
        )


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
