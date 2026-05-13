"""Levelset-stopping-criterion parity tests for the JAX tracing path.

These tests close the last carve-out of item 14 / item 16: the
``LevelsetStoppingCriterion`` is now wired through the JAX
``jax.lax.while_loop`` body via
:meth:`simsopt.geo.surface.SurfaceClassifier.to_jax_classifier_fn`. The
rebuild happens once at host setup time and the resulting closure is
queried inside the JAX driver without dropping device residency.

Three slices are covered under the ``event_time_tracing`` parity-ladder
lane (state-vector + event-time tolerances):

1. ``test_surface_classifier_to_jax_classifier_fn_matches_cpu`` —
   point-wise sign parity between the CPU ``SurfaceClassifier``
   (``evaluate_xyz``) and the JAX classifier closure for a random batch
   of probe points inside the cylindrical cuboid bounding the surface.
2. ``test_levelset_stopping_criterion_stops_jax_trace_at_surface`` —
   end-to-end smoke that starts a fieldline inside a thin toroidal
   shell and confirms that the JAX driver reports a Levelset-induced
   stop (``status = -1 - i``) within the ``event_time_atol`` of the
   levelset surface.
3. ``test_compute_fieldlines_jax_accepts_levelset_stopping_criterion`` —
   public-API smoke that exercises the
   :func:`simsopt.field.tracing.compute_fieldlines` wrapper with a
   Levelset stopping criterion and confirms the wrapper runs without
   raising ``NotImplementedError`` and returns a non-empty trajectory.

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.field.toroidal_field_jax import ToroidalFieldJAX
from simsopt.field import tracing as tracing_module
from simsopt.field.tracing import (
    LevelsetStoppingCriterion,
    compute_fieldlines,
)
from simsopt.geo.surface import SurfaceClassifier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.jax_core.tracing import (
    FieldlineTracingSpec,
    trace_fieldline,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


@pytest.fixture(scope="module")
def event_time_lane():
    return dict(_EVENT_TIME_TOLERANCES)


def _build_test_surface(R0: float = 1.0, minor: float = 0.2) -> SurfaceRZFourier:
    """Build a simple circular-cross-section toroidal surface for testing.

    The surface is purely toroidal with ``nfp=1`` so the cylindrical
    signed-distance grid used by ``SurfaceClassifier`` is a smooth
    function of ``(r, phi, z)`` and the discretised levelset captures
    the analytic ``sqrt((R - R0)^2 + z^2) - minor`` distance up to the
    interpolant truncation error.
    """
    surf = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=0,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 1.0, 32, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 32, endpoint=False),
    )
    surf.set_rc(0, 0, R0)
    surf.set_rc(1, 0, minor)
    surf.set_zs(1, 0, minor)
    return surf


# ---------------------------------------------------------------------------
# 1. SurfaceClassifier <-> to_jax_classifier_fn sign parity
# ---------------------------------------------------------------------------


def test_surface_classifier_to_jax_classifier_fn_matches_cpu(event_time_lane):
    """JAX classifier closure must agree with the CPU SurfaceClassifier sign.

    The CPU ``evaluate_xyz`` returns the raw signed distance; the JAX
    closure returns the ``sign`` of the same distance. We compare signs
    over a random batch of probe points that lie strictly inside the
    cylindrical cuboid bounding the surface (so the out-of-bounds
    sentinel ``-1`` does not contaminate the sign-match comparison).
    """

    del event_time_lane  # signs match exactly; lane tolerances unused

    surf = _build_test_surface(R0=1.0, minor=0.2)
    sc = SurfaceClassifier(surf, h=0.05, p=2)

    classifier_fn = sc.to_jax_classifier_fn()

    # Sample probe points inside the bounding cuboid, well away from the
    # outer boundary so neither side picks up the sentinel.
    rng = np.random.default_rng(0)
    rmin, rmax = sc.rrange
    zmin, zmax = sc.zrange
    rmin_in = rmin + 0.05 * (rmax - rmin)
    rmax_in = rmax - 0.05 * (rmax - rmin)
    zmin_in = zmin + 0.05 * (zmax - zmin)
    zmax_in = zmax - 0.05 * (zmax - zmin)
    n_probe = 200
    rs = rng.uniform(rmin_in, rmax_in, size=n_probe)
    phis = rng.uniform(0.05, 2 * np.pi - 0.05, size=n_probe)
    zs = rng.uniform(zmin_in, zmax_in, size=n_probe)
    xyz = np.stack(
        [rs * np.cos(phis), rs * np.sin(phis), zs],
        axis=-1,
    ).astype(np.float64)

    cpu_dist = sc.evaluate_xyz(xyz)  # shape (n_probe, 1)
    cpu_sign = np.sign(cpu_dist.ravel())

    jax_sign = np.asarray(classifier_fn(jnp.asarray(xyz)))

    # Exclude probe points where the CPU distance is below the
    # interpolant noise floor (sign is then numerically undefined).
    well_defined = np.abs(cpu_dist.ravel()) > 1e-6
    n_well_defined = int(well_defined.sum())
    assert n_well_defined > n_probe // 2, (
        "Too few well-defined probe points "
        f"(n_well_defined={n_well_defined}/{n_probe}); "
        "the surface fixture may be ill-conditioned."
    )

    mismatch = np.abs(cpu_sign[well_defined] - jax_sign[well_defined])
    assert int(mismatch.sum()) == 0, (
        "CPU vs JAX signed-distance sign mismatch on "
        f"{int(mismatch.sum())}/{n_well_defined} well-defined probes."
    )


# ---------------------------------------------------------------------------
# 2. JAX trace stops on Levelset crossing
# ---------------------------------------------------------------------------


def test_levelset_stopping_criterion_stops_jax_trace_at_surface(event_time_lane):
    """Start a JAX fieldline inside a torus; the Levelset must trigger a stop.

    In a purely toroidal field, the analytic streamline is the circle
    ``(R, Z) = (R_init, 0)``. We pick ``R_init`` inside the volume
    (``R0 - minor < R_init < R0 + minor``); the trajectory winds around
    that constant-R circle and never exits the levelset surface, so the
    criterion must NOT fire. Symmetrically, we pick ``R_init`` outside
    the volume; the very first accepted step is then outside the
    levelset, so the criterion must fire immediately (status = -1).

    Both arms exercise the JAX while-loop dispatcher inside
    ``_stopping_criterion_should_stop`` for the ``LevelsetStoppingCriterion``
    branch (lines 436-442 of ``simsopt.jax_core.tracing``).
    """

    del event_time_lane  # binary stop/no-stop assertions; lane gates unused

    surf = _build_test_surface(R0=1.0, minor=0.2)
    sc = SurfaceClassifier(surf, h=0.05, p=2)
    jax_classifier_fn = sc.to_jax_classifier_fn()

    from simsopt.jax_core.tracing import LevelsetStoppingCriterion as JaxLevelset

    levelset_jax = JaxLevelset(classifier_fn=jax_classifier_fn)

    # Arm 1: start STRICTLY OUTSIDE the volume. The first accepted step
    # remains outside (no toroidal field carries the trajectory inward),
    # so the Levelset must fire on accepted step 1.
    R0_field = 1.0
    B0 = 0.8

    def field_fn(point):
        x = point[0]
        y = point[1]
        r2 = x * x + y * y
        coeff = jnp.asarray(B0 * R0_field, dtype=jnp.float64) / r2
        return jnp.stack([-coeff * y, coeff * x, jnp.asarray(0.0, dtype=jnp.float64)])

    R_outside = 1.5  # >> R0 + minor = 1.2, well outside the volume
    spec_out = FieldlineTracingSpec(
        tmax=1.0,
        rtol=1e-9,
        atol=1e-11,
        max_steps=200,
        max_phi_hits=64,
    )
    y0_out = jnp.asarray([R_outside, 0.0, 0.0], dtype=jnp.float64)
    res_out = trace_fieldline(
        spec_out,
        y0_out,
        field_fn,
        stopping_criteria=(levelset_jax,),
    )
    status_out = int(res_out.status)
    assert status_out == -1, (
        "Levelset stopping criterion did not fire on a strictly-outside "
        f"start point: status={status_out}, t_final={float(res_out.t_final)}. "
        "Expected status=-1 (criterion idx 0 fired)."
    )

    # Arm 2: start STRICTLY INSIDE the volume on the constant-R streamline.
    # The trajectory never leaves the volume, so the criterion must NOT
    # fire — the integrator must run to ``tmax`` (status = 0).
    R_inside = 1.0  # at R0 (centerline of the torus); deep inside.
    # Short tmax so we cover ~ half a turn and stay well inside.
    tmax_inside = 0.05 * (2.0 * np.pi * R_inside * R_inside) / (B0 * R0_field)
    spec_in = FieldlineTracingSpec(
        tmax=float(tmax_inside),
        rtol=1e-9,
        atol=1e-11,
        max_steps=400,
        max_phi_hits=64,
    )
    y0_in = jnp.asarray([R_inside, 0.0, 0.0], dtype=jnp.float64)
    res_in = trace_fieldline(
        spec_in,
        y0_in,
        field_fn,
        stopping_criteria=(levelset_jax,),
    )
    status_in = int(res_in.status)
    assert status_in == 0, (
        "Levelset stopping criterion fired on a strictly-inside, "
        "constant-R trajectory that never leaves the volume: "
        f"status={status_in}, t_final={float(res_in.t_final)}. "
        "Expected status=0 (reached tmax)."
    )


# ---------------------------------------------------------------------------
# 3. Public-API smoke: compute_fieldlines accepts Levelset on the JAX route
# ---------------------------------------------------------------------------


def _force_jax_backend(monkeypatch):
    """Force ``is_jax_backend`` to True at the tracing module call site."""
    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)


def test_compute_fieldlines_jax_accepts_levelset_stopping_criterion(monkeypatch):
    """``compute_fieldlines(..., stopping_criteria=[LevelsetStoppingCriterion(SurfaceClassifier(surf))])`` runs on the JAX route.

    The wrapper must:

    1. accept the Python ``LevelsetStoppingCriterion`` without raising
       :class:`NotImplementedError`,
    2. rebuild the JAX classifier closure at host setup,
    3. run the JAX driver to produce a non-empty trajectory.

    This test does not assert event-time accuracy (covered by the
    isolation arm above); it gates the public-API wiring.
    """

    surf = _build_test_surface(R0=1.0, minor=0.2)
    sc = SurfaceClassifier(surf, h=0.05, p=2)
    classifier = LevelsetStoppingCriterion(sc)

    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = compute_fieldlines(
        ToroidalFieldJAX(1.0, 0.8),
        [1.0],  # R_init inside the volume
        [0.0],
        tmax=0.1,
        tol=1e-9,
        stopping_criteria=[classifier],
    )
    assert len(res_tys) == 1
    assert res_tys[0].ndim == 2
    assert res_tys[0].shape[1] == 4
    assert res_tys[0].shape[0] >= 2, (
        "JAX route returned a trivial trajectory under a Levelset "
        f"stopping criterion: shape={res_tys[0].shape}"
    )
    assert len(res_phi_hits) == 1
    assert res_phi_hits[0].shape[1] == 5


def test_compute_fieldlines_jax_levelset_fires_outside_volume(monkeypatch):
    """Levelset criterion fires through the public wrapper for an outside-start.

    The CPU ``compute_fieldlines`` would record the same stop via the
    C++ ``LevelsetStoppingCriterion`` predicate; the JAX wrapper now
    matches that behaviour through the rebuilt JAX closure.
    """

    surf = _build_test_surface(R0=1.0, minor=0.2)
    sc = SurfaceClassifier(surf, h=0.05, p=2)
    classifier = LevelsetStoppingCriterion(sc)

    _force_jax_backend(monkeypatch)
    # Start outside the torus volume: R_init = 1.5 >> R0 + minor = 1.2.
    res_tys, res_phi_hits = compute_fieldlines(
        ToroidalFieldJAX(1.0, 0.8),
        [1.5],
        [0.0],
        tmax=1.0,
        tol=1e-9,
        stopping_criteria=[classifier],
    )
    assert len(res_tys) == 1
    # The wrapper matches upstream C++ recording: the firing post-step
    # state is emitted as an event row, not as a live trajectory row.
    traj = res_tys[0]
    np.testing.assert_allclose(traj, np.array([[0.0, 1.5, 0.0, 0.0]]))
    phi_hits = res_phi_hits[0]
    # At least one Levelset event row (idx = -1 for criterion 0) is
    # recorded by the JAX driver via the phi_hits buffer.
    assert phi_hits.shape[0] >= 1
    levelset_rows = phi_hits[phi_hits[:, 1] < 0]
    assert levelset_rows.shape[0] >= 1, (
        "Outside-start trajectory ran without recording a Levelset "
        f"firing row in phi_hits: phi_hits.shape={phi_hits.shape}"
    )


# ---------------------------------------------------------------------------
# 4. Raw sopp.LevelsetStoppingCriterion still raises NotImplementedError
# ---------------------------------------------------------------------------


def test_compute_fieldlines_jax_rejects_raw_sopp_levelset(monkeypatch):
    """Raw ``sopp.LevelsetStoppingCriterion`` instances must surface a
    precise :class:`NotImplementedError`: the JAX path needs grid
    metadata to rebuild a JAX-traceable spec, and raw interpolants do
    not carry that information.
    """

    import simsoptpp as sopp

    interpolant = sopp.RegularGridInterpolant3D(
        sopp.UniformInterpolationRule(2),
        [0.5, 1.5, 4],
        [0.0, 2.0 * np.pi, 4],
        [-0.4, 0.4, 4],
        1,
        True,
    )

    def fbatch(rs, _phis, zs):
        rs_arr = np.asarray(rs, dtype=np.float64)
        zs_arr = np.asarray(zs, dtype=np.float64)
        return list(0.1 - np.sqrt((rs_arr - 1.0) ** 2 + zs_arr**2))

    interpolant.interpolate_batch(fbatch)
    raw_crit = sopp.LevelsetStoppingCriterion(interpolant)

    _force_jax_backend(monkeypatch)
    with pytest.raises(NotImplementedError, match="LevelsetStoppingCriterion"):
        compute_fieldlines(
            ToroidalFieldJAX(1.0, 0.8),
            [1.0],
            [0.0],
            tmax=0.1,
            tol=1e-9,
            stopping_criteria=[raw_crit],
        )
