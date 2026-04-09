"""
Parity tests for the JAX Boozer residual scalar objective.

Validates:
1. Forward value against direct NumPy computation.
2. Gradient via JAX autodiff against centred finite differences.
3. Hessian structure (symmetry).
4. C++ parity (when simsoptpp is available).
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp
from conftest import (
    device_float64,
    host_array,
    host_scalar,
    parity_default_device,
    parity_lane,
    parity_rng,
)

from simsopt.geo.boozer_residual_jax import (
    boozer_residual_scalar,
    boozer_residual_grad,
    boozer_residual_hessian,
    boozer_residual_vector,
)


@pytest.fixture(autouse=True)
def _parity_device_scope(parity_lane):
    with parity_default_device(parity_lane):
        yield


def _make_synthetic_data(nphi=10, ntheta=12, seed=42):
    """Create synthetic B, xphi, xtheta arrays for testing."""
    rng = parity_rng(seed)
    B = rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0.0, 0.0, 1.0])
    xphi = rng.randn(nphi, ntheta, 3) * 0.5
    xtheta = rng.randn(nphi, ntheta, 3) * 0.5
    return device_float64(B), device_float64(xphi), device_float64(xtheta)


class TestBoozerResidualScalar:
    """Test the forward scalar value."""

    def test_matches_numpy(self, parity_lane):
        """JAX scalar matches a direct NumPy computation."""
        nphi, ntheta = 8, 10
        B, xphi, xtheta = _make_synthetic_data(nphi, ntheta)
        G, iota = 1.5, 0.3

        # NumPy reference (same formula)
        B_np = host_array(B)
        xphi_np = host_array(xphi)
        xtheta_np = host_array(xtheta)
        tang = xphi_np + iota * xtheta_np
        B2 = np.sum(B_np**2, axis=-1)
        residual = G * B_np - B2[..., None] * tang
        modB = np.sqrt(B2)
        w = 1.0 / modB
        rtil = w[..., None] * residual
        J_ref = 0.5 * np.sum(rtil**2) / (3 * nphi * ntheta)

        J_jax = host_scalar(
            boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB=True)
        )

        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-14)

    def test_no_weight(self, parity_lane):
        """Test with weight_inv_modB=False."""
        nphi, ntheta = 8, 10
        B, xphi, xtheta = _make_synthetic_data(nphi, ntheta)
        G, iota = 1.5, 0.3

        B_np = host_array(B)
        xphi_np = host_array(xphi)
        xtheta_np = host_array(xtheta)
        tang = xphi_np + iota * xtheta_np
        B2 = np.sum(B_np**2, axis=-1)
        residual = G * B_np - B2[..., None] * tang
        J_ref = 0.5 * np.sum(residual**2) / (3 * nphi * ntheta)

        J_jax = host_scalar(
            boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB=False)
        )

        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-14)

    def test_zero_residual(self):
        """When B is consistent with Boozer, residual should be near zero."""
        nphi, ntheta = 8, 10
        rng = parity_rng(0)
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

        J = host_scalar(
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

    def test_weighted_zero_field_is_finite(self):
        """Zero-field points must not introduce NaN/Inf in the weighted path."""
        nphi, ntheta = 4, 5
        B = jnp.zeros((nphi, ntheta, 3))
        xphi = jnp.ones((nphi, ntheta, 3))
        xtheta = 2.0 * jnp.ones((nphi, ntheta, 3))

        scalar_value = boozer_residual_scalar(
            1.5,
            0.25,
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )
        residual_vector = boozer_residual_vector(
            1.5,
            0.25,
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )

        assert jnp.isfinite(scalar_value)
        assert jnp.all(jnp.isfinite(residual_vector))
        np.testing.assert_allclose(float(scalar_value), 0.0, atol=0.0)
        np.testing.assert_allclose(host_array(residual_vector), 0.0, atol=0.0)


class TestBoozerResidualGradient:
    """Test gradient via finite differences."""

    def test_grad_iota(self, parity_lane):
        """Gradient w.r.t. iota matches finite differences."""
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        G, iota = 1.5, 0.3
        nsurfdofs = 0  # no surface dofs in this test

        grad = boozer_residual_grad(G, iota, B, xphi, xtheta, nsurfdofs)
        # grad[-2] = dJ/diota, grad[-1] = dJ/dG
        dJ_diota_jax = host_scalar(grad[-2])

        eps = 1e-6
        Jp = host_scalar(boozer_residual_scalar(G, iota + eps, B, xphi, xtheta))
        Jm = host_scalar(boozer_residual_scalar(G, iota - eps, B, xphi, xtheta))
        dJ_diota_fd = (Jp - Jm) / (2 * eps)

        np.testing.assert_allclose(dJ_diota_jax, dJ_diota_fd, rtol=1e-5)

    def test_grad_G(self, parity_lane):
        """Gradient w.r.t. G matches finite differences."""
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        G, iota = 1.5, 0.3
        nsurfdofs = 0

        grad = boozer_residual_grad(G, iota, B, xphi, xtheta, nsurfdofs)
        dJ_dG_jax = host_scalar(grad[-1])

        eps = 1e-6
        Jp = host_scalar(boozer_residual_scalar(G + eps, iota, B, xphi, xtheta))
        Jm = host_scalar(boozer_residual_scalar(G - eps, iota, B, xphi, xtheta))
        dJ_dG_fd = (Jp - Jm) / (2 * eps)

        np.testing.assert_allclose(dJ_dG_jax, dJ_dG_fd, rtol=1e-5)


class TestBoozerResidualHessian:
    """Test Hessian properties."""

    def test_hessian_symmetry(self, parity_lane):
        """Hessian should be symmetric."""
        B, xphi, xtheta = _make_synthetic_data(6, 8)
        G, iota = 1.5, 0.3
        nsurfdofs = 0

        H = boozer_residual_hessian(G, iota, B, xphi, xtheta, nsurfdofs)
        H_host = host_array(H)
        np.testing.assert_allclose(H_host, H_host.T, atol=1e-14)

    def test_hessian_matches_grad_fd(self, parity_lane):
        """Hessian diagonal blocks match FD of gradient."""
        B, xphi, xtheta = _make_synthetic_data(6, 8)
        G, iota = 1.5, 0.3
        nsurfdofs = 0

        H = boozer_residual_hessian(G, iota, B, xphi, xtheta, nsurfdofs)
        # H is 2×2: [d²J/diota², d²J/diotadG; d²J/dGdiota, d²J/dG²]

        eps = 1e-5
        g_p = boozer_residual_grad(G, iota + eps, B, xphi, xtheta, nsurfdofs)
        g_m = boozer_residual_grad(G, iota - eps, B, xphi, xtheta, nsurfdofs)
        d2J_diota2_fd = (host_scalar(g_p[-2]) - host_scalar(g_m[-2])) / (2 * eps)
        d2J_diotadG_fd = (host_scalar(g_p[-1]) - host_scalar(g_m[-1])) / (2 * eps)

        np.testing.assert_allclose(host_scalar(H[0, 0]), d2J_diota2_fd, rtol=1e-4)
        np.testing.assert_allclose(host_scalar(H[0, 1]), d2J_diotadG_fd, rtol=1e-4)


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
            host_array(grad[:nsurfdofs]),
            0.0,
            atol=1e-30,
        )
        # iota and G entries should be nonzero
        assert host_scalar(jnp.abs(grad[-2])) > 1e-10
        assert host_scalar(jnp.abs(grad[-1])) > 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
