"""Parity tests for item 26 -- :class:`DipoleFieldJAX` public wrapper.

These tests assert that the public ``MagneticField``-boundary methods
of :class:`simsopt.field.dipole_field_jax.DipoleFieldJAX` reproduce the
upstream CPU class :class:`simsopt.field.DipoleField` at the
``direct_kernel`` parity-ladder lane on production-scale fixtures.

The tests deliberately go through the public
``set_points_cart`` -> ``B`` / ``dB_by_dX`` / ``A`` / ``dA_by_dX``
interface (not the bare JAX kernel calls covered by item 24) so that
the integration with the ``sopp.MagneticField`` cache contract is
exercised end-to-end. The symmetry-expansion path is exercised by
parametrising over ``(stellsym, nfp, coordinate_flag)``.

All tolerances are imported from
:func:`benchmarks.validation_ladder_contract.parity_ladder_tolerances`;
no ``rtol`` / ``atol`` numeric literals appear inline in the test body.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import DipoleField, DipoleFieldJAX as ExportedDipoleFieldJAX
from simsopt.field.dipole_field_jax import DipoleFieldJAX


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _seeded_dipoles(seed: int, num_dipoles: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Generate dipole positions inside a small cube and physical moments.

    Positions live in a ``[-0.3, 0.3]`` cube so the field-evaluation
    points (sampled on a ``[0.6, 2.0]`` shell) never coincide with a
    dipole site. Moments are drawn from a Gaussian and used unscaled.
    """

    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.3, 0.3, size=(num_dipoles, 3)).astype(
        np.float64, copy=False
    )
    moments = rng.normal(loc=0.0, scale=1.0, size=(num_dipoles, 3)).astype(
        np.float64, copy=False
    )
    return positions, moments


def _seeded_points(seed: int, count: int = 60) -> np.ndarray:
    """Generate evaluation points on a far-field shell.

    All components are bounded away from ``0`` so the points cannot land
    on the dipole grid in ``_seeded_dipoles``. Points are returned as a
    contiguous float64 array suitable for ``set_points_cart``.
    """

    rng = np.random.default_rng(seed)
    base = rng.uniform(0.6, 2.0, size=(count, 3))
    flips = rng.choice([-1.0, 1.0], size=base.shape)
    pts = (base * flips).astype(np.float64, copy=False)
    return np.ascontiguousarray(pts)


_PARITY_CASES = [
    pytest.param(False, 1, "cartesian", id="nosym-nfp1-cartesian"),
    pytest.param(True, 2, "cartesian", id="stellsym-nfp2-cartesian"),
    pytest.param(True, 3, "cylindrical", id="stellsym-nfp3-cylindrical"),
    pytest.param(False, 4, "toroidal", id="nosym-nfp4-toroidal"),
]


class TestDipoleFieldJAXParity:
    def test_package_export(self):
        assert ExportedDipoleFieldJAX is DipoleFieldJAX

    @pytest.mark.parametrize("stellsym, nfp, coordinate_flag", _PARITY_CASES)
    def test_B_dB_A_dA_parity_vs_cpu(self, stellsym, nfp, coordinate_flag):
        """Full set of public ``MagneticField`` getters match the CPU class.

        Oracle: :class:`simsopt.field.DipoleField`. Tolerance lane:
        ``direct_kernel``. Production scale: 14 dipoles times symmetry
        expansion, 60 evaluation points.
        """

        positions, moments = _seeded_dipoles(seed=10 + nfp, num_dipoles=14)
        points = _seeded_points(seed=100 + nfp, count=60)

        cpu = DipoleField(
            positions,
            moments,
            stellsym=stellsym,
            nfp=nfp,
            coordinate_flag=coordinate_flag,
            R0=1.0,
        )
        jax_field = DipoleFieldJAX(
            positions,
            moments,
            stellsym=stellsym,
            nfp=nfp,
            coordinate_flag=coordinate_flag,
            R0=1.0,
        )

        # Sanity check: the half-period expansion should match exactly.
        np.testing.assert_allclose(
            jax_field.dipole_grid, cpu.dipole_grid, rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(jax_field.m_vec, cpu.m_vec, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(
            jax_field.m_maxima, cpu.m_maxima, rtol=_RTOL, atol=_ATOL
        )

        cpu.set_points_cart(points)
        jax_field.set_points_cart(points)

        np.testing.assert_allclose(
            np.asarray(jax_field.B()), np.asarray(cpu.B()), rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            np.asarray(jax_field.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_field.A()), np.asarray(cpu.A()), rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            np.asarray(jax_field.dA_by_dX()),
            np.asarray(cpu.dA_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_default_m_maxima_matches_cpu(self):
        """When ``m_maxima=None`` the constructor mirrors the CPU broadcast.

        The CPU class fills ``m_maxima`` with the largest moment magnitude
        across all dipoles and repeats it for the full symmetric manifold.
        """

        positions, moments = _seeded_dipoles(seed=33, num_dipoles=12)
        cpu = DipoleField(positions, moments, stellsym=True, nfp=2)
        jax_field = DipoleFieldJAX(positions, moments, stellsym=True, nfp=2)
        np.testing.assert_allclose(
            jax_field.m_maxima, cpu.m_maxima, rtol=_RTOL, atol=_ATOL
        )

    def test_explicit_m_maxima_matches_cpu(self):
        """Caller-supplied ``m_maxima`` is broadcast through the manifold."""

        positions, moments = _seeded_dipoles(seed=21, num_dipoles=11)
        m_max_seed = np.arange(1, 12, dtype=np.float64) / 10.0
        cpu = DipoleField(positions, moments, stellsym=True, nfp=2, m_maxima=m_max_seed)
        jax_field = DipoleFieldJAX(
            positions, moments, stellsym=True, nfp=2, m_maxima=m_max_seed
        )
        np.testing.assert_allclose(
            jax_field.m_maxima, cpu.m_maxima, rtol=_RTOL, atol=_ATOL
        )

    def test_set_points_cyl_updates_device_points(self):
        """Cylindrical point updates refresh the staged JAX point buffer."""

        positions, moments = _seeded_dipoles(seed=41, num_dipoles=9)
        rphiz = np.ascontiguousarray(
            np.array(
                [
                    [1.1, 0.2, -0.3],
                    [1.4, -0.5, 0.1],
                    [1.9, 1.2, 0.4],
                    [2.2, -1.1, -0.2],
                ],
                dtype=np.float64,
            )
        )
        cpu = DipoleField(positions, moments, stellsym=True, nfp=2)
        jax_field = DipoleFieldJAX(positions, moments, stellsym=True, nfp=2)

        cpu.set_points_cyl(rphiz)
        jax_field.set_points_cyl(rphiz)

        np.testing.assert_allclose(
            np.asarray(jax_field.B()), np.asarray(cpu.B()), rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            np.asarray(jax_field.dB_by_dX()),
            np.asarray(cpu.dB_by_dX()),
            rtol=_RTOL,
            atol=_ATOL,
        )


class TestDipoleFieldJAXInputValidation:
    """Constructor input-validation behaviour."""

    def test_rejects_malformed_dipole_grid_shape(self):
        with pytest.raises(ValueError):
            DipoleFieldJAX(np.zeros((4,)), np.zeros((4, 3)))
        with pytest.raises(ValueError):
            DipoleFieldJAX(np.zeros((4, 2)), np.zeros((4, 3)))

    def test_rejects_mismatched_dipole_vectors_shape(self):
        with pytest.raises(ValueError):
            DipoleFieldJAX(np.zeros((4, 3)), np.zeros((5, 3)))
        with pytest.raises(ValueError):
            DipoleFieldJAX(np.zeros((4, 3)), np.zeros((4, 2)))

    def test_rejects_invalid_coordinate_flag(self):
        with pytest.raises(ValueError):
            DipoleFieldJAX(np.zeros((4, 3)), np.zeros((4, 3)), coordinate_flag="bogus")


class TestDipoleFieldJAXTransferGuard:
    """The compiled kernels run cleanly under ``transfer_guard('disallow')``."""

    def test_B_dB_A_dA_under_disallow_guard(self):
        """All four getters work when device buffers are pre-staged.

        The JAX kernels stage dipole arrays at construction time. The
        ``set_points_cart`` boundary performs an explicit
        :func:`jax.device_put` on the points buffer; once that buffer is
        on-device, subsequent JIT calls do not require implicit transfers.
        """

        positions, moments = _seeded_dipoles(seed=24, num_dipoles=14)
        points = _seeded_points(seed=2424, count=60)
        jax_field = DipoleFieldJAX(positions, moments, stellsym=True, nfp=2)
        jax_field.set_points_cart(points)

        # Drain pending compilation under the default (``allow``) guard so
        # the strict-guard region only measures steady-state execution.
        jax_field._dipole_points_device.block_until_ready()
        jax_field._dipole_moments_device.block_until_ready()
        B_warm = jnp.asarray(jax_field.B())
        B_warm.block_until_ready()
        dB_warm = jnp.asarray(jax_field.dB_by_dX())
        dB_warm.block_until_ready()
        A_warm = jnp.asarray(jax_field.A())
        A_warm.block_until_ready()
        dA_warm = jnp.asarray(jax_field.dA_by_dX())
        dA_warm.block_until_ready()

        with jax.transfer_guard("disallow"):
            jax_field.clear_cached_properties()
            jnp.asarray(jax_field.B()).block_until_ready()
            jnp.asarray(jax_field.dB_by_dX()).block_until_ready()
            jnp.asarray(jax_field.A()).block_until_ready()
            jnp.asarray(jax_field.dA_by_dX()).block_until_ready()


class TestDipoleFieldJAXSerialization:
    """``as_dict`` / ``from_dict`` round-trip preserves field outputs."""

    def test_round_trip_preserves_B_values(self):
        positions, moments = _seeded_dipoles(seed=51, num_dipoles=10)
        points = _seeded_points(seed=5151, count=50)
        original = DipoleFieldJAX(positions, moments, stellsym=True, nfp=2)
        original.set_points_cart(points)
        B_original = np.asarray(original.B())

        d = original.as_dict(serial_objs_dict={})
        d["points"] = points  # ``from_dict`` decoder reads this back.
        rebuilt = DipoleFieldJAX.from_dict(d, serial_objs_dict={}, recon_objs={})
        assert rebuilt.coordinate_flag == original.coordinate_flag
        assert rebuilt.stellsym == original.stellsym
        assert rebuilt.nfp == original.nfp
        B_rebuilt = np.asarray(rebuilt.B())
        np.testing.assert_allclose(B_rebuilt, B_original, rtol=_RTOL, atol=_ATOL)
