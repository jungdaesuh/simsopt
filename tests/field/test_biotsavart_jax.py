"""
Parity tests for the JAX Biot-Savart implementation.

Validates against:
1. Analytical on-axis field of a circular current loop.
2. Maxwell's equation ∇·B = 0 (trace of dB/dX).
3. C++ reference (when simsoptpp is available).
"""

import importlib.util
from contextlib import contextmanager
import os
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


@contextmanager
def _backend_mode(mode: str):
    previous_mode = os.environ.get("SIMSOPT_BACKEND_MODE")
    os.environ["SIMSOPT_BACKEND_MODE"] = mode
    try:
        yield
    finally:
        if previous_mode is None:
            del os.environ["SIMSOPT_BACKEND_MODE"]
        else:
            os.environ["SIMSOPT_BACKEND_MODE"] = previous_mode


def _load_with_backend_mode(mode: str):
    return _load(f"biotsavart_jax_{mode}", "field/biotsavart_jax.py")


def _load_chunked_biotsavart():
    return _load_with_backend_mode("jax_cpu_parity")


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


def _make_shifted_circular_coils(ncoils: int, *, R: float = 1.0, nquad: int = 128):
    gamma, gammadash = _make_circular_coil(R=R, nquad=nquad)
    z_offsets = jnp.linspace(-0.4, 0.4, ncoils, dtype=jnp.float64)
    gamma_stack = jnp.concatenate(
        [gamma + jnp.array([[[0.0, 0.0, offset]]]) for offset in z_offsets],
        axis=0,
    )
    gammadash_stack = jnp.concatenate([gammadash] * ncoils, axis=0)
    currents = jnp.linspace(5e4, 5e4 + 1e3 * (ncoils - 1), ncoils, dtype=jnp.float64)
    return gamma_stack, gammadash_stack, currents


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
        analytical_rel_tol = 1e-12
        symmetry_abs_tol = 1e-14

        np.testing.assert_allclose(
            float(B[0, 2]),
            B_analytical,
            rtol=analytical_rel_tol,
        )
        # Bx and By should be zero by symmetry
        np.testing.assert_allclose(float(B[0, 0]), 0.0, atol=symmetry_abs_tol)
        np.testing.assert_allclose(float(B[0, 1]), 0.0, atol=symmetry_abs_tol)

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
        analytical_rel_tol = 1e-12
        np.testing.assert_allclose(
            float(B[0, 2]),
            B_analytical,
            rtol=analytical_rel_tol,
        )

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
        divergence_abs_tol = 1e-14
        np.testing.assert_allclose(np.array(div_B), 0.0, atol=divergence_abs_tol)

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

        fd_rel_tol = 1e-8
        fd_abs_tol = 5e-11
        np.testing.assert_allclose(
            np.array(dB_jax),
            dB_fd,
            rtol=fd_rel_tol,
            atol=fd_abs_tol,
        )

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
        from simsopt.configs import get_data
        from simsopt.field import coils_via_symmetries, BiotSavart

        curves, currents_objs, _, nfp, _ = get_data("ncsx")
        coils = coils_via_symmetries(curves, currents_objs, nfp, stellsym=True)
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


class TestBiotSavartJaxChunkedParity:
    """Directly compare chunked low-level kernels against dense references."""

    def test_chunked_B_and_dB_match_dense_reference(self):
        with _backend_mode("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            assert chunked_bs._coil_chunk_size() > 0

            gammas, gammadashs, currents = _make_shifted_circular_coils(20, nquad=96)
            points = jnp.array(
                [
                    [0.2, 0.1, -0.3],
                    [0.1, -0.4, 0.0],
                    [-0.3, 0.2, 0.35],
                ],
                dtype=jnp.float64,
            )

            dense_B = jax.vmap(
                lambda x: chunked_bs._biot_savart_one_point_dense(
                    x,
                    gammas,
                    gammadashs,
                    currents,
                )
            )(points)
            dense_dB = jax.vmap(
                lambda x: jnp.swapaxes(
                    jax.jacfwd(chunked_bs._biot_savart_one_point_dense, argnums=0)(
                        x,
                        gammas,
                        gammadashs,
                        currents,
                    ),
                    -1,
                    -2,
                )
            )(points)

            B = chunked_bs.biot_savart_B(points, gammas, gammadashs, currents)
            dB = chunked_bs.biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
            B_combo, dB_combo = chunked_bs.biot_savart_B_and_dB(
                points,
                gammas,
                gammadashs,
                currents,
            )

            np.testing.assert_allclose(np.asarray(B), np.asarray(dense_B), atol=1e-14)
            np.testing.assert_allclose(np.asarray(dB), np.asarray(dense_dB), atol=1e-14)
            np.testing.assert_allclose(
                np.asarray(B_combo),
                np.asarray(dense_B),
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.asarray(dB_combo),
                np.asarray(dense_dB),
                atol=1e-14,
            )

    def test_chunked_A_matches_dense_reference(self):
        with _backend_mode("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            assert chunked_bs._coil_chunk_size() > 0

            gammas, gammadashs, currents = _make_shifted_circular_coils(20, nquad=96)
            points = jnp.array(
                [
                    [0.15, 0.05, -0.25],
                    [-0.05, -0.25, 0.1],
                ],
                dtype=jnp.float64,
            )

            dense_A = jax.vmap(
                lambda x: chunked_bs._biot_savart_A_one_point_dense(
                    x,
                    gammas,
                    gammadashs,
                    currents,
                )
            )(points)
            A = chunked_bs.biot_savart_A(points, gammas, gammadashs, currents)

            np.testing.assert_allclose(np.asarray(A), np.asarray(dense_A), atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
