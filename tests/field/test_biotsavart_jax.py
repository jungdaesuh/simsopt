"""
Parity tests for the JAX Biot-Savart implementation.

Validates against:
1. Analytical on-axis field of a circular current loop.
2. Maxwell's equation ∇·B = 0 (trace of dB/dX).
3. C++ reference (when simsoptpp is available).
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Load JAX module directly (avoids simsopt/__init__.py → simsoptpp dep)
_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
biot_savart_B = _bs.biot_savart_B
biot_savart_dB_by_dX = _bs.biot_savart_dB_by_dX
biot_savart_B_and_dB = _bs.biot_savart_B_and_dB

MU0 = 4.0 * np.pi * 1e-7


def _make_circular_coil(R=1.0, nquad=128):
    """Create a single circular coil of radius R centred at the origin in the xy-plane."""
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    gamma = np.stack([R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], axis=-1)
    # dγ/dφ where φ ∈ [0,1) → chain rule factor 2π already present
    # but simsopt parameterises φ ∈ [0,1), so gammadash = dγ/d(φ_01) = 2π·dγ/dφ_rad
    # Actually the quadrature spacing is 1/nquad, so:
    # gammadash = dγ/dφ_01 = dγ/d(φ_rad) * d(φ_rad)/d(φ_01)
    #           = dγ/d(φ_rad) * 2π
    gammadash = np.stack(
        [
            -R * np.sin(phi) * 2 * np.pi,
            R * np.cos(phi) * 2 * np.pi,
            np.zeros_like(phi),
        ],
        axis=-1,
    )
    return (
        jnp.array(gamma[None, :, :]),  # (1, nquad, 3)
        jnp.array(gammadash[None, :, :]),  # (1, nquad, 3)
    )


class TestBiotSavartJaxAnalytical:
    """Test against the known on-axis field of a circular current loop."""

    def test_on_axis_field(self):
        """B_z at the centre of a circular loop = μ₀ I / (2R)."""
        R = 1.0
        I = 1e6  # 1 MA
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        points = jnp.array([[0.0, 0.0, 0.0]])
        B = biot_savart_B(points, gammas, gammadashs, currents)

        B_analytical = MU0 * I / (2.0 * R)
        np.testing.assert_allclose(float(B[0, 2]), B_analytical, rtol=1e-6)
        # Bx and By should be zero by symmetry
        np.testing.assert_allclose(float(B[0, 0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(float(B[0, 1]), 0.0, atol=1e-10)

    def test_on_axis_field_offset_z(self):
        """B_z at z=h on axis: B_z = μ₀IR²/(2(R²+h²)^{3/2})."""
        R = 1.0
        I = 1e6
        h = 0.5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        points = jnp.array([[0.0, 0.0, h]])
        B = biot_savart_B(points, gammas, gammadashs, currents)

        B_analytical = MU0 * I * R**2 / (2.0 * (R**2 + h**2) ** 1.5)
        np.testing.assert_allclose(float(B[0, 2]), B_analytical, rtol=1e-5)

    def test_div_B_zero(self):
        """∇·B = Tr(dB/dX) should be zero (Maxwell)."""
        R = 1.0
        I = 1e5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        # Off-axis points
        points = jnp.array(
            [
                [0.3, 0.0, 0.0],
                [0.0, 0.3, 0.1],
                [0.5, 0.5, 0.2],
            ]
        )
        dB = biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
        div_B = jnp.trace(dB, axis1=1, axis2=2)  # (npoints,)
        np.testing.assert_allclose(np.array(div_B), 0.0, atol=1e-10)

    def test_B_and_dB_consistency(self):
        """biot_savart_B_and_dB returns same values as separate calls."""
        R = 1.0
        I = 1e5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=128)
        currents = jnp.array([I])

        points = jnp.array(
            [
                [0.3, 0.1, 0.0],
                [0.0, 0.5, 0.2],
            ]
        )

        B_ref = biot_savart_B(points, gammas, gammadashs, currents)
        dB_ref = biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
        B_combo, dB_combo = biot_savart_B_and_dB(points, gammas, gammadashs, currents)

        np.testing.assert_allclose(np.array(B_combo), np.array(B_ref), atol=1e-14)
        np.testing.assert_allclose(np.array(dB_combo), np.array(dB_ref), atol=1e-14)

    def test_dB_dX_finite_difference(self):
        """dB/dX matches centred finite differences (SIMSOPT convention)."""
        R = 1.0
        I = 1e5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        x0 = jnp.array([[0.4, 0.1, 0.05]])
        # SIMSOPT convention: dB_dX[j, l] = ∂_j B_l
        dB_jax = biot_savart_dB_by_dX(x0, gammas, gammadashs, currents)[0]

        eps = 1e-5
        dB_fd = np.zeros((3, 3))
        for j in range(3):
            xp = x0.at[0, j].add(eps)
            xm = x0.at[0, j].add(-eps)
            Bp = biot_savart_B(xp, gammas, gammadashs, currents)[0]
            Bm = biot_savart_B(xm, gammas, gammadashs, currents)[0]
            # Row j = all B components differentiated w.r.t. x_j
            dB_fd[j, :] = (np.array(Bp) - np.array(Bm)) / (2 * eps)

        np.testing.assert_allclose(np.array(dB_jax), dB_fd, rtol=1e-5)

    def test_multiple_coils(self):
        """Superposition: field of two coils equals sum of individual fields."""
        R = 1.0
        gammas1, gammadashs1 = _make_circular_coil(R=R, nquad=128)
        gammas2 = gammas1 + jnp.array([[[0.0, 0.0, 0.5]]])
        gammadashs2 = gammadashs1.copy()

        currents = jnp.array([1e5, -5e4])
        gammas = jnp.concatenate([gammas1, gammas2], axis=0)
        gammadashs = jnp.concatenate([gammadashs1, gammadashs2], axis=0)

        points = jnp.array([[0.0, 0.0, 0.25]])

        B_total = biot_savart_B(points, gammas, gammadashs, currents)
        B1 = biot_savart_B(points, gammas1, gammadashs1, jnp.array([currents[0]]))
        B2 = biot_savart_B(points, gammas2, gammadashs2, jnp.array([currents[1]]))

        np.testing.assert_allclose(np.array(B_total), np.array(B1 + B2), atol=1e-14)


class TestBiotSavartJaxCppParity:
    """Compare against the C++ simsoptpp kernel (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        sopp = pytest.importorskip("simsoptpp")
        if not hasattr(sopp, "BiotSavart"):
            pytest.skip("simsoptpp compiled extensions not available")
        pytest.importorskip("simsopt")

    def test_B_parity_ncsx(self):
        from simsopt.configs import get_ncsx_data
        from simsopt.field import coils_via_symmetries, BiotSavart

        curves, currents_objs, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents_objs, 3)
        bs = BiotSavart(coils)

        npoints = 50
        np.random.seed(42)
        points_np = np.random.randn(npoints, 3) * 0.3
        points_np[:, 0] += 1.0  # shift near torus

        bs.set_points(points_np)
        B_ref = bs.B()

        gammas_np = np.array([c.curve.gamma() for c in coils])
        gds_np = np.array([c.curve.gammadash() for c in coils])
        currents_np = np.array([c.current.get_value() for c in coils])

        B_jax = biot_savart_B(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(np.array(B_jax), B_ref, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
