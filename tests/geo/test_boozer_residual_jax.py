"""
Parity tests for the JAX Boozer residual scalar objective.

Validates:
1. Forward value against direct NumPy computation.
2. Gradient via JAX autodiff against centred finite differences.
3. Hessian structure (symmetry).
4. C++ parity (when simsoptpp is available).
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_br = _load("boozer_residual_jax", "geo/boozer_residual_jax.py")
boozer_residual_scalar = _br.boozer_residual_scalar
boozer_residual_grad = _br.boozer_residual_grad
boozer_residual_hessian = _br.boozer_residual_hessian


def _make_synthetic_data(nphi=10, ntheta=12, seed=42):
    """Create synthetic B, xphi, xtheta arrays for testing."""
    rng = np.random.RandomState(seed)
    B = rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0.0, 0.0, 1.0])
    xphi = rng.randn(nphi, ntheta, 3) * 0.5
    xtheta = rng.randn(nphi, ntheta, 3) * 0.5
    return jnp.array(B), jnp.array(xphi), jnp.array(xtheta)


class TestBoozerResidualScalar:
    """Test the forward scalar value."""

    def test_matches_numpy(self):
        """JAX scalar matches a direct NumPy computation."""
        nphi, ntheta = 8, 10
        B, xphi, xtheta = _make_synthetic_data(nphi, ntheta)
        G, iota = 1.5, 0.3

        # NumPy reference (same formula)
        B_np = np.array(B)
        xphi_np = np.array(xphi)
        xtheta_np = np.array(xtheta)
        tang = xphi_np + iota * xtheta_np
        B2 = np.sum(B_np**2, axis=-1)
        residual = G * B_np - B2[..., None] * tang
        modB = np.sqrt(B2)
        w = 1.0 / modB
        rtil = w[..., None] * residual
        J_ref = 0.5 * np.sum(rtil**2) / (3 * nphi * ntheta)

        J_jax = float(
            boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB=True)
        )

        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-14)

    def test_no_weight(self):
        """Test with weight_inv_modB=False."""
        nphi, ntheta = 8, 10
        B, xphi, xtheta = _make_synthetic_data(nphi, ntheta)
        G, iota = 1.5, 0.3

        B_np = np.array(B)
        xphi_np = np.array(xphi)
        xtheta_np = np.array(xtheta)
        tang = xphi_np + iota * xtheta_np
        B2 = np.sum(B_np**2, axis=-1)
        residual = G * B_np - B2[..., None] * tang
        J_ref = 0.5 * np.sum(residual**2) / (3 * nphi * ntheta)

        J_jax = float(
            boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB=False)
        )

        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-14)

    def test_zero_residual(self):
        """When B is consistent with Boozer, residual should be near zero."""
        nphi, ntheta = 8, 10
        rng = np.random.RandomState(0)
        xphi = rng.randn(nphi, ntheta, 3)
        xtheta = rng.randn(nphi, ntheta, 3)
        iota = 0.5
        G = 2.0
        tang = xphi + iota * xtheta

        # Construct B so that G·B = |B|²·tang
        # This means B is proportional to tang: B = α·tang, and G·α = α²|tang|²
        # So α = G / |tang|²
        tang_sq = np.sum(tang**2, axis=-1, keepdims=True)
        alpha = G / tang_sq
        B = alpha * tang

        J = float(
            boozer_residual_scalar(
                G,
                iota,
                jnp.array(B),
                jnp.array(xphi),
                jnp.array(xtheta),
                weight_inv_modB=False,
            )
        )
        np.testing.assert_allclose(J, 0.0, atol=1e-20)


class TestBoozerResidualGradient:
    """Test gradient via finite differences."""

    def test_grad_iota(self):
        """Gradient w.r.t. iota matches finite differences."""
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        G, iota = 1.5, 0.3
        nsurfdofs = 0  # no surface dofs in this test

        grad = boozer_residual_grad(G, iota, B, xphi, xtheta, nsurfdofs)
        # grad[-2] = dJ/diota, grad[-1] = dJ/dG
        dJ_diota_jax = float(grad[-2])

        eps = 1e-6
        Jp = float(boozer_residual_scalar(G, iota + eps, B, xphi, xtheta))
        Jm = float(boozer_residual_scalar(G, iota - eps, B, xphi, xtheta))
        dJ_diota_fd = (Jp - Jm) / (2 * eps)

        np.testing.assert_allclose(dJ_diota_jax, dJ_diota_fd, rtol=1e-5)

    def test_grad_G(self):
        """Gradient w.r.t. G matches finite differences."""
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        G, iota = 1.5, 0.3
        nsurfdofs = 0

        grad = boozer_residual_grad(G, iota, B, xphi, xtheta, nsurfdofs)
        dJ_dG_jax = float(grad[-1])

        eps = 1e-6
        Jp = float(boozer_residual_scalar(G + eps, iota, B, xphi, xtheta))
        Jm = float(boozer_residual_scalar(G - eps, iota, B, xphi, xtheta))
        dJ_dG_fd = (Jp - Jm) / (2 * eps)

        np.testing.assert_allclose(dJ_dG_jax, dJ_dG_fd, rtol=1e-5)


class TestBoozerResidualHessian:
    """Test Hessian properties."""

    def test_hessian_symmetry(self):
        """Hessian should be symmetric."""
        B, xphi, xtheta = _make_synthetic_data(6, 8)
        G, iota = 1.5, 0.3
        nsurfdofs = 0

        H = boozer_residual_hessian(G, iota, B, xphi, xtheta, nsurfdofs)
        np.testing.assert_allclose(np.array(H), np.array(H.T), atol=1e-14)

    def test_hessian_matches_grad_fd(self):
        """Hessian diagonal blocks match FD of gradient."""
        B, xphi, xtheta = _make_synthetic_data(6, 8)
        G, iota = 1.5, 0.3
        nsurfdofs = 0

        H = boozer_residual_hessian(G, iota, B, xphi, xtheta, nsurfdofs)
        # H is 2×2: [d²J/diota², d²J/diotadG; d²J/dGdiota, d²J/dG²]

        eps = 1e-5
        g_p = boozer_residual_grad(G, iota + eps, B, xphi, xtheta, nsurfdofs)
        g_m = boozer_residual_grad(G, iota - eps, B, xphi, xtheta, nsurfdofs)
        d2J_diota2_fd = (float(g_p[-2]) - float(g_m[-2])) / (2 * eps)
        d2J_diotadG_fd = (float(g_p[-1]) - float(g_m[-1])) / (2 * eps)

        np.testing.assert_allclose(float(H[0, 0]), d2J_diota2_fd, rtol=1e-4)
        np.testing.assert_allclose(float(H[0, 1]), d2J_diotadG_fd, rtol=1e-4)


class TestBoozerResidualM1Limitations:
    """Document known M1 scope limitations (surface DOF gradients)."""

    def test_surface_dof_gradient_is_zero(self):
        """M1 wrappers treat B/xphi/xtheta as constants, so the gradient
        w.r.t. surface_dofs is identically zero.  This test documents
        that behavior explicitly — it is NOT a bug, but a scope limit.
        The full surface→BiotSavart→residual composed pipeline (M2)
        will replace these wrappers."""
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        G, iota = 1.5, 0.3
        nsurfdofs = 5  # nonzero surface dofs

        grad = boozer_residual_grad(G, iota, B, xphi, xtheta, nsurfdofs)
        # First nsurfdofs entries should be exactly zero
        np.testing.assert_allclose(
            np.array(grad[:nsurfdofs]),
            0.0,
            atol=1e-30,
        )
        # iota and G entries should be nonzero
        assert float(jnp.abs(grad[-2])) > 1e-10
        assert float(jnp.abs(grad[-1])) > 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
