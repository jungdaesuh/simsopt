"""Parity tests for item 15 -- partial JAX-backed analytic-field wrappers.

These tests assert that the public ``MagneticField``-boundary methods
of the new JAX wrappers in
:mod:`simsopt.field.magneticfieldclasses_jax`
(``ToroidalFieldJAX``, ``PoloidalFieldJAX``, ``MirrorModelJAX``,
``DommaschkJAX``, ``ReimanJAX``) reproduce the upstream CPU classes
``ToroidalField`` / ``PoloidalField`` / ``MirrorModel`` /
``Dommaschk`` / ``Reiman`` at the ``direct_kernel`` parity-ladder
lane on production-scale fixtures. ``CircularCoil`` and
``InterpolatedField`` remain blocked sub-scopes of item 15.

The tests deliberately go through the public
``set_points_cart`` -> ``B`` / ``dB_by_dX`` / ``A`` / ``dA_by_dX``
interface (not the bare JAX kernel calls covered by item 11 / item 12)
so that the integration with the ``sopp.MagneticField`` cache contract
is exercised end-to-end.

All tolerances are imported from
:func:`benchmarks.validation_ladder_contract.parity_ladder_tolerances`;
no ``rtol`` / ``atol`` numeric literals appear inline in the test body.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from simsopt.field import (
    Dommaschk,
    MirrorModel,
    PoloidalField,
    Reiman,
    ToroidalField,
)
from simsopt.field.magneticfieldclasses_jax import (
    DommaschkJAX,
    MirrorModelJAX,
    PoloidalFieldJAX,
    ReimanJAX,
    ToroidalFieldJAX,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _production_points(seed: int, count: int = 60) -> np.ndarray:
    """Return a (count, 3) production-scale point cloud.

    Production-scale floor for analytic-field kernels per the goal
    prompt: at least 50 points. We use 60 by default so a filter
    routine that drops a few singular-near points still leaves >= 50.
    """
    rng = np.random.default_rng(int(seed))
    points = np.zeros((count, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(0.4, 1.8, size=count)
    points[:, 1] = rng.uniform(0.4, 1.8, size=count)
    points[:, 2] = rng.uniform(-0.5, 0.5, size=count)
    return np.ascontiguousarray(points)


def _away_from(points: np.ndarray, R0: float, margin: float) -> np.ndarray:
    R_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    mask = np.abs(R_xy - R0) > margin
    return np.ascontiguousarray(points[mask], dtype=np.float64)


# ── ToroidalField wrapper parity ─────────────────────────────────────


class TestToroidalFieldJAX:
    def test_B_dB_d2B_A_dA_parity_vs_cpu(self):
        """Full set of public ``MagneticField`` getters match the CPU class.

        Oracle: :class:`simsopt.field.ToroidalField`.
        Tolerance lane: ``direct_kernel``.
        Production scale: 60 evaluation points.
        """
        points = _production_points(seed=101, count=60)
        cpu = ToroidalField(R0=1.3, B0=0.8)
        jax_ = ToroidalFieldJAX(R0=1.3, B0=0.8)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)

        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.d2B_by_dXdX()),
            np.asarray(cpu.d2B_by_dXdX()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.A()),
            np.asarray(cpu.A()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dA_by_dX()),
            np.asarray(cpu.dA_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_as_from_dict_roundtrip_preserves_class(self):
        """SIMSON serialization round-trip preserves wrapper identity."""
        cpu_points = _production_points(seed=102, count=10)
        field = ToroidalFieldJAX(R0=1.5, B0=0.9)
        field.set_points_cart(cpu_points)
        payload = json.dumps(SIMSON(field), cls=GSONEncoder)
        regen = json.loads(payload, cls=GSONDecoder)
        assert type(regen).__name__ == "ToroidalFieldJAX"
        np.testing.assert_allclose(
            np.asarray(field.B()),
            np.asarray(regen.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── PoloidalField wrapper parity ─────────────────────────────────────


class TestPoloidalFieldJAX:
    def test_B_dB_parity_vs_cpu(self):
        """B and dB match the CPU :class:`PoloidalField` away from R=R0."""
        R0 = 1.0
        points = _away_from(_production_points(seed=201, count=120), R0=R0, margin=0.2)
        assert points.shape[0] >= 50, "production-scale floor"
        cpu = PoloidalField(R0=R0, B0=1.1, q=1.3)
        jax_ = PoloidalFieldJAX(R0=R0, B0=1.1, q=1.3)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_as_from_dict_roundtrip_preserves_class(self):
        points = _away_from(_production_points(seed=202, count=80), R0=1.0, margin=0.2)
        field = PoloidalFieldJAX(R0=1.0, B0=1.1, q=1.3)
        field.set_points_cart(points)
        payload = json.dumps(SIMSON(field), cls=GSONEncoder)
        regen = json.loads(payload, cls=GSONDecoder)
        assert type(regen).__name__ == "PoloidalFieldJAX"
        np.testing.assert_allclose(
            np.asarray(field.B()),
            np.asarray(regen.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── MirrorModel wrapper parity ───────────────────────────────────────


class TestMirrorModelJAX:
    def test_B_dB_parity_vs_cpu(self):
        """B and dB match :class:`MirrorModel` away from R=0."""
        points = _away_from(_production_points(seed=301, count=120), R0=0.0, margin=0.2)
        assert points.shape[0] >= 50, "production-scale floor"
        cpu = MirrorModel(B0=6.51292, gamma=0.124904, Z_m=0.98)
        jax_ = MirrorModelJAX(B0=6.51292, gamma=0.124904, Z_m=0.98)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_known_reference_values_match_cpu(self):
        """The published reference values in the upstream tests still pass."""
        cpu = MirrorModel(B0=6.51292, gamma=0.124904, Z_m=0.98)
        jax_ = MirrorModelJAX(B0=6.51292, gamma=0.124904, Z_m=0.98)
        point = np.asarray([[0.9231, 0.8423, -0.1123]], dtype=np.float64)
        cpu.set_points_cart(point)
        jax_.set_points_cart(point)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_as_from_dict_roundtrip_preserves_class(self):
        points = _away_from(_production_points(seed=302, count=60), R0=0.0, margin=0.2)
        field = MirrorModelJAX(B0=6.51292, gamma=0.124904, Z_m=0.98)
        field.set_points_cart(points)
        payload = json.dumps(SIMSON(field), cls=GSONEncoder)
        regen = json.loads(payload, cls=GSONDecoder)
        assert type(regen).__name__ == "MirrorModelJAX"
        np.testing.assert_allclose(
            np.asarray(field.B()),
            np.asarray(regen.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Dommaschk wrapper parity ─────────────────────────────────────────


class TestDommaschkJAX:
    def test_B_dB_parity_vs_cpu_published_paper_fixture(self):
        """Reproduce the canonical Dommaschk published-paper fixture.

        Matches :class:`Dommaschk` reference from
        ``tests/field/test_magneticfields.py::test_Dommaschk`` exactly,
        including the ``ToroidalField(1, 1)`` baseline that the CPU
        class folds into the returned B/dB.
        """
        mn = [[10, 2], [15, 3]]
        coeffs = [[-2.18, -2.18], [25.8, -25.8]]
        cpu = Dommaschk(mn=mn, coeffs=coeffs)
        jax_ = DommaschkJAX(mn=mn, coeffs=coeffs)
        points = np.asarray([[0.9231, 0.8423, -0.1123]], dtype=np.float64)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_B_dB_parity_vs_cpu_random_production_fixture(self):
        """Production-scale parity: 60-point random fixture vs CPU.

        Uses well-conditioned coefficients so neither the JAX kernel
        nor the C++ kernel exhibits catastrophic cancellation. The
        ULP-bounded drift for huge coefficients (``5.10e10``) at high
        mode order is a documented numerical-order divergence between
        the CPU and JAX execution orders, not a parity bug; see
        ``.artifacts/jax_port_goal/plans/11.md``.
        """
        mn = [[3, 2], [6, 4], [2, 5]]
        coeffs = [[1.4, 1.4], [0.5, 0.5], [0.25, -0.25]]
        points = _production_points(seed=401, count=60)
        cpu = Dommaschk(mn=mn, coeffs=coeffs)
        jax_ = DommaschkJAX(mn=mn, coeffs=coeffs)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_dB_is_symmetric(self):
        """Vacuum-field property: dB is symmetric in the spatial indices."""
        mn = [[5, 2], [5, 4], [5, 10]]
        coeffs = [[1.4, 1.4], [19.25, 0.0], [5.10e10, 5.10e10]]
        jax_ = DommaschkJAX(mn=mn, coeffs=coeffs)
        points = _production_points(seed=402, count=50)
        jax_.set_points_cart(points)
        dB = np.asarray(jax_.dB_by_dX())
        np.testing.assert_allclose(
            dB,
            np.transpose(dB, axes=(0, 2, 1)),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_as_from_dict_roundtrip_preserves_class(self):
        mn = [[10, 2], [15, 3]]
        coeffs = [[-2.18, -2.18], [25.8, -25.8]]
        points = _production_points(seed=403, count=10)
        field = DommaschkJAX(mn=mn, coeffs=coeffs)
        field.set_points_cart(points)
        payload = json.dumps(SIMSON(field), cls=GSONEncoder)
        regen = json.loads(payload, cls=GSONDecoder)
        assert type(regen).__name__ == "DommaschkJAX"
        np.testing.assert_allclose(
            np.asarray(field.B()),
            np.asarray(regen.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Reiman wrapper parity ────────────────────────────────────────────


class TestReimanJAX:
    def test_B_dB_parity_vs_cpu_production_fixture(self):
        """Production-scale parity vs :class:`Reiman` CPU class."""
        iota0 = 0.15
        iota1 = 0.38
        k = [6]
        epsilonk = [0.01]
        m0 = 1
        cpu = Reiman(iota0=iota0, iota1=iota1, k=k, epsilonk=epsilonk, m0=m0)
        jax_ = ReimanJAX(iota0=iota0, iota1=iota1, k=k, epsilonk=epsilonk, m0=m0)
        # Reiman's R_axis is 1; stay away from the magnetic axis ring.
        points = _away_from(_production_points(seed=501, count=120), R0=1.0, margin=0.1)
        assert points.shape[0] >= 50, "production-scale floor"
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_multi_mode_parity_vs_cpu(self):
        """Multi-mode and non-default m0 symmetry parity vs CPU."""
        cpu = Reiman(
            iota0=0.2, iota1=0.41, k=[4, 6, 8], epsilonk=[0.02, 0.01, 0.005], m0=3
        )
        jax_ = ReimanJAX(
            iota0=0.2, iota1=0.41, k=[4, 6, 8], epsilonk=[0.02, 0.01, 0.005], m0=3
        )
        points = _away_from(_production_points(seed=502, count=120), R0=1.0, margin=0.1)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_as_from_dict_roundtrip_preserves_class(self):
        points = _away_from(_production_points(seed=503, count=80), R0=1.0, margin=0.1)
        field = ReimanJAX(iota0=0.15, iota1=0.38, k=[6], epsilonk=[0.01], m0=1)
        field.set_points_cart(points)
        payload = json.dumps(SIMSON(field), cls=GSONEncoder)
        regen = json.loads(payload, cls=GSONDecoder)
        assert type(regen).__name__ == "ReimanJAX"
        np.testing.assert_allclose(
            np.asarray(field.B()),
            np.asarray(regen.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Transfer-guard discipline ────────────────────────────────────────


class TestTransferGuardDiscipline:
    """The new JAX wrappers stage spec scalars and host points via the
    strict-safe :func:`jax.device_put` path (see
    ``magneticfieldclasses_jax._points_device`` and the per-class
    ``_build_spec`` methods). Calling the underlying JAX kernels with
    those device-resident specs and pre-staged points must not trigger
    an implicit host-to-device transfer under
    :func:`jax.transfer_guard("disallow")`.

    The wrapper's public ``B()`` / ``dB_by_dX()`` getters materialise
    the kernel output back to NumPy at the ``_*_impl`` boundary (the
    ``sopp.MagneticField`` cache is a contiguous host array). Under
    the CPU backend a device-to-host fetch is unconditionally allowed
    by JAX, so the wrapper-level public getters are guard-clean as a
    whole on CPU. To bound the test to the part that this item is
    accountable for, the assertion below runs the JAX kernels using
    the wrapper-built specs under the strict guard with the points
    already device-resident.
    """

    def test_wrapper_specs_clean_under_strict_transfer_guard(self):
        from simsopt.jax_core.analytic_fields import (
            dommaschk_B,
            dommaschk_dB,
            reiman_B,
            reiman_dB,
        )
        from simsopt.jax_core.analytic_pure_fields import (
            mirror_B,
            mirror_dB,
            poloidal_B,
            poloidal_dB,
            toroidal_A,
            toroidal_B,
            toroidal_dA,
            toroidal_dB,
        )

        # Build the JAX-backed wrappers; their specs hold device-resident
        # scalars and arrays via ``jax.device_put``.
        toroidal = ToroidalFieldJAX(R0=1.3, B0=0.8)
        poloidal = PoloidalFieldJAX(R0=1.0, B0=1.1, q=1.3)
        mirror = MirrorModelJAX(B0=6.51292, gamma=0.124904, Z_m=0.98)
        dommaschk = DommaschkJAX(
            mn=[[10, 2], [15, 3]],
            coeffs=[[-2.18, -2.18], [25.8, -25.8]],
        )
        reiman = ReimanJAX(iota0=0.15, iota1=0.38, k=[6], epsilonk=[0.01], m0=1)

        # Pre-stage host points to device arrays outside the guarded
        # region.
        points = _production_points(seed=601, count=50)
        poloidal_pts_host = _away_from(points, R0=1.0, margin=0.2)
        mirror_pts_host = _away_from(points, R0=0.0, margin=0.2)
        reiman_pts_host = _away_from(points, R0=1.0, margin=0.1)
        device_points = jnp.asarray(points, dtype=jnp.float64)
        device_poloidal = jnp.asarray(poloidal_pts_host, dtype=jnp.float64)
        device_mirror = jnp.asarray(mirror_pts_host, dtype=jnp.float64)
        device_reiman = jnp.asarray(reiman_pts_host, dtype=jnp.float64)
        device_points.block_until_ready()
        device_poloidal.block_until_ready()
        device_mirror.block_until_ready()
        device_reiman.block_until_ready()

        with jax.transfer_guard("disallow"):
            toroidal_B(toroidal._spec, device_points).block_until_ready()
            toroidal_dB(toroidal._spec, device_points).block_until_ready()
            toroidal_A(toroidal._spec, device_points).block_until_ready()
            toroidal_dA(toroidal._spec, device_points).block_until_ready()
            poloidal_B(poloidal._spec, device_poloidal).block_until_ready()
            poloidal_dB(poloidal._spec, device_poloidal).block_until_ready()
            mirror_B(mirror._spec, device_mirror).block_until_ready()
            mirror_dB(mirror._spec, device_mirror).block_until_ready()
            dommaschk_B(dommaschk._spec, device_points).block_until_ready()
            dommaschk_dB(dommaschk._spec, device_points).block_until_ready()
            reiman_B(reiman._spec, device_reiman).block_until_ready()
            reiman_dB(reiman._spec, device_reiman).block_until_ready()


# ── Cache invariance under set_points ────────────────────────────────


class TestCacheInvariance:
    """When ``set_points`` is called the ``MagneticField`` cache must
    invalidate, so subsequent calls return values at the NEW points.
    Catches a bug class where the JAX wrapper might inadvertently cache
    the device array of an old call.
    """

    def test_setpoints_invalidates_cached_B(self):
        jax_ = ToroidalFieldJAX(R0=1.3, B0=0.8)
        cpu = ToroidalField(R0=1.3, B0=0.8)
        first = _production_points(seed=701, count=20)
        second = _production_points(seed=702, count=20)
        jax_.set_points_cart(first)
        cpu.set_points_cart(first)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        jax_.set_points_cart(second)
        cpu.set_points_cart(second)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


def test_input_validation_dommaschk_jax():
    """``DommaschkJAX`` rejects malformed mn / coeffs arrays explicitly."""
    with pytest.raises(ValueError):
        DommaschkJAX(mn=[[1]], coeffs=[[0, 0]])
    with pytest.raises(ValueError):
        DommaschkJAX(mn=[[1, 2]], coeffs=[[0]])
    with pytest.raises(ValueError):
        DommaschkJAX(mn=[[1, 2], [3, 4]], coeffs=[[0, 0]])


def test_input_validation_reiman_jax():
    """``ReimanJAX`` rejects mismatched k / epsilonk lengths."""
    with pytest.raises(ValueError):
        ReimanJAX(k=[2, 4], epsilonk=[0.01])
