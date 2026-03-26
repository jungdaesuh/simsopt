"""
Pure-JAX Biot-Savart parity tests matching upstream C++ test suite.

Section 8a of jax_gpu_remaining_todos.md.  These tests exercise functions
in ``biotsavart_jax.py`` that have upstream C++ equivalents in
``tests/field/test_biotsavart.py`` but lacked JAX coverage.

Tests:
1. Quadrature convergence (exponential decay of B and dB/dX error)
2. B = curl(A)
3. dA/dX finite difference (forward Taylor test)
4. dB/dX symmetric (vacuum) + divergence-free
5. B VJP Taylor test (reverse-mode derivative correctness)

No simsoptpp dependency — all tests use pure JAX functions.
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Load JAX module directly (avoids simsopt/__init__.py → simsoptpp dep)
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
biot_savart_B = _bs.biot_savart_B
biot_savart_dB_by_dX = _bs.biot_savart_dB_by_dX
biot_savart_A = _bs.biot_savart_A
biot_savart_dA_by_dX = _bs.biot_savart_dA_by_dX
biot_savart_B_vjp = _bs.biot_savart_B_vjp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fourier_coil(nquad=200):
    """Replicate upstream ``get_curve(nquad)`` as pure arrays.

    ``CurveXYZFourier(nquad, order=3)`` with coefficients
    ``y0=1, yc1=0.5, zc2=0.5`` (all others zero).

    Curve (t ∈ [0, 1)):
        x(t) = 0
        y(t) = 1.0 + 0.5·cos(2πt)
        z(t) = 0.5·cos(4πt)
    """
    t = np.linspace(0, 1, nquad, endpoint=False)
    twopi = 2 * np.pi

    x = np.zeros(nquad)
    y = 1.0 + 0.5 * np.cos(twopi * t)
    z = 0.5 * np.cos(2 * twopi * t)
    gamma = np.stack([x, y, z], axis=-1)

    dx = np.zeros(nquad)
    dy = -0.5 * twopi * np.sin(twopi * t)
    dz = -0.5 * 2 * twopi * np.sin(2 * twopi * t)
    gammadash = np.stack([dx, dy, dz], axis=-1)

    return (
        jnp.array(gamma[None, :, :]),  # (1, nquad, 3)
        jnp.array(gammadash[None, :, :]),  # (1, nquad, 3)
    )


# Points matching upstream (near coil at y≈0.9, outside the wire).
_BASE_POINTS = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
_CURRENT = 1e4


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestBiotSavartParitySuite:
    """Parity tests matching upstream tests/field/test_biotsavart.py."""

    def test_quadrature_convergence(self):
        """Multi-level quadrature refinement shows monotone error decay.

        Matches ``test_biotsavart_exponential_convergence``.
        Tests both B and dB/dX convergence across 4→8→16→32 refinement,
        with a machine-precision floor (the smooth Fourier coil converges
        super-exponentially, so coarser levels may already be at ε_mach).
        """
        points = jnp.array(_BASE_POINTS)
        currents = jnp.array([_CURRENT])

        g_ref, gd_ref = _make_fourier_coil(1000)
        B_ref = biot_savart_B(points, g_ref, gd_ref, currents)
        dB_ref = biot_savart_dB_by_dX(points, g_ref, gd_ref, currents)

        prev_B_err = float("inf")
        prev_dB_err = float("inf")
        for nq in [4, 8, 16, 32]:
            g, gd = _make_fourier_coil(nq)
            B = biot_savart_B(points, g, gd, currents)
            dB = biot_savart_dB_by_dX(points, g, gd, currents)

            B_err = float(jnp.linalg.norm(B - B_ref))
            dB_err = float(jnp.linalg.norm(dB - dB_ref))

            if prev_B_err > 1e-14:
                assert B_err < 0.5 * prev_B_err, (
                    f"B not converging at nq={nq}: {B_err:.2e} vs {prev_B_err:.2e}"
                )
            if prev_dB_err > 1e-14:
                assert dB_err < 0.5 * prev_dB_err, (
                    f"dB not converging at nq={nq}: {dB_err:.2e} vs {prev_dB_err:.2e}"
                )

            prev_B_err = B_err
            prev_dB_err = dB_err

    def test_B_is_curl_A(self):
        """B equals curl(A) from Biot-Savart vector potential.

        Matches ``test_biotsavart_B_is_curlA``.
        """
        gammas, gammadashs = _make_fourier_coil(200)
        currents = jnp.array([_CURRENT])
        points = jnp.array(_BASE_POINTS)

        B = biot_savart_B(points, gammas, gammadashs, currents)
        dA_dX = biot_savart_dA_by_dX(points, gammas, gammadashs, currents)

        # curl(A)_i = ε_{ijk} ∂_j A_k
        curl_A = jnp.stack(
            [
                dA_dX[:, 1, 2] - dA_dX[:, 2, 1],
                dA_dX[:, 2, 0] - dA_dX[:, 0, 2],
                dA_dX[:, 0, 1] - dA_dX[:, 1, 0],
            ],
            axis=1,
        )

        np.testing.assert_allclose(np.array(curl_A), np.array(B), atol=1e-14)

    @pytest.mark.parametrize("idx", [0, 16])
    def test_dA_dX_finite_difference(self, idx):
        """dA/dX matches forward finite differences.

        Matches ``test_biotsavart_dAdX_taylortest``.
        """
        np.random.seed(42)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = jnp.array([_CURRENT])

        points = jnp.array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )
        A0 = biot_savart_A(points, gammas, gammadashs, currents)[idx]
        dA = biot_savart_dA_by_dX(points, gammas, gammadashs, currents)[idx]

        for direction in [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        ]:
            deriv = dA.T @ direction
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                A_eps = biot_savart_A(
                    points + eps * direction, gammas, gammadashs, currents
                )[idx]
                deriv_est = (A_eps - A0) / eps
                new_err = float(jnp.linalg.norm(deriv - deriv_est))
                if new_err < 1e-14:
                    break  # machine precision reached
                assert new_err < 0.55 * err
                err = new_err

    @pytest.mark.parametrize("idx", [0, 16])
    def test_dB_dX_symmetric_and_divergence_free(self, idx):
        """dB/dX is symmetric (vacuum, curl B = 0) and trace-free (div B = 0).

        Matches ``test_biotsavart_gradient_symmetric_and_divergence_free``.
        Existing JAX test only checked div=0; this adds the symmetry check.
        """
        np.random.seed(42)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = jnp.array([_CURRENT])

        points = jnp.array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )
        dB = biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
        dB_idx = np.array(dB[idx])

        # Divergence-free: Tr(dB/dX) = 0
        assert abs(dB_idx[0, 0] + dB_idx[1, 1] + dB_idx[2, 2]) < 1e-14
        # Symmetric in vacuum: ∂_j B_l = ∂_l B_j
        np.testing.assert_allclose(dB_idx, dB_idx.T, atol=1e-12)

    def test_B_vjp_taylor_test(self):
        """VJP Taylor test: |J(γ+εh) − J(γ) − ε⟨dJ,h⟩| decays as O(ε).

        Matches ``test_dB_by_dcoilcoeff_reverse_taylortest``.
        Uses J = Σ B² and perturbs coil positions (gammas).
        """
        np.random.seed(1)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = jnp.array([_CURRENT])

        points = jnp.array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )

        B = biot_savart_B(points, gammas, gammadashs, currents)
        J0 = float(jnp.sum(B**2))

        # VJP: B_vjp(v) = dB^T v w.r.t. coil inputs
        # For J = sum(B²): dJ = 2·B_vjp(B)
        grad_gammas, _, _ = biot_savart_B_vjp(points, B, gammas, gammadashs, currents)

        h = 1e-2 * jnp.array(np.random.rand(*gammas.shape))
        dJ_dh = float(2 * jnp.sum(grad_gammas * h))

        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            B_eps = biot_savart_B(points, gammas + eps * h, gammadashs, currents)
            J_eps = float(jnp.sum(B_eps**2))
            deriv_est = (J_eps - J0) / eps
            new_err = abs(deriv_est - dJ_dh)
            assert new_err < 0.55 * err
            err = new_err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
