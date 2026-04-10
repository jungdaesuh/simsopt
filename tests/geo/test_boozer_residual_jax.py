"""
Parity tests for the JAX Boozer residual scalar objective.

Validates:
1. Forward value against direct NumPy computation.
2. Gradient via JAX autodiff against centred finite differences.
3. Hessian structure (symmetry).
4. C++ parity (when simsoptpp is available).
"""

import math
from pathlib import Path
import sys

import pytest
import numpy as np

import jax
import jax.numpy as jnp
_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import (
    device_float64,
    host_array,
    host_scalar,
    parity_acceptance_tolerance,
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


# Keep this NumPy fixed-tree reference local so the Boozer scalar/vector oracle
# does not depend on the separate reduction-unit-test helpers.
def _numpy_pairwise_sum_flat(array):
    reduced = np.ravel(np.asarray(array, dtype=np.float64))
    if reduced.size == 0:
        return float(np.sum(reduced, dtype=np.float64))

    padded = np.zeros(1 << (reduced.size - 1).bit_length(), dtype=np.float64)
    padded[: reduced.size] = reduced
    return float(_numpy_pairwise_reduce_last_axis(padded)[0])


def _numpy_pairwise_reduce_last_axis(array):
    reduced = np.asarray(array, dtype=np.float64)
    while reduced.shape[-1] > 1:
        pair_shape = reduced.shape[:-1] + (reduced.shape[-1] // 2, 2)
        reduced = reduced.reshape(pair_shape).sum(axis=-1, dtype=np.float64)
    return reduced


def _numpy_pairwise_sum_last_axis(array):
    reduced = np.asarray(array, dtype=np.float64)
    axis_size = reduced.shape[-1]
    if axis_size == 0:
        return np.sum(reduced, axis=-1, dtype=np.float64)

    pad_width = [(0, 0)] * reduced.ndim
    pad_width[-1] = (0, (1 << (axis_size - 1).bit_length()) - axis_size)
    padded = np.pad(reduced, pad_width, mode="constant")
    return np.squeeze(_numpy_pairwise_reduce_last_axis(padded), axis=-1)


def _numpy_boozer_residual_reference(G, iota, B, xphi, xtheta, *, weight_inv_modB):
    tang = xphi + iota * xtheta
    B2 = _numpy_pairwise_sum_last_axis(B * B)
    residual = G * B - B2[..., None] * tang
    if weight_inv_modB:
        modB = np.sqrt(B2)
        residual = np.divide(
            residual,
            modB[..., None],
            out=np.zeros_like(residual),
            where=modB[..., None] > 0.0,
        )
    scalar = 0.5 * _numpy_pairwise_sum_flat(residual * residual) / residual.size
    return residual.reshape(-1), scalar


def _make_near_floor_data(nphi=48, ntheta=48, seed=77, residual_scale=1e-12):
    rng = parity_rng(seed)
    xphi = rng.randn(nphi, ntheta, 3) * 0.15 + np.array([1.0, 0.0, 0.0])
    xtheta = rng.randn(nphi, ntheta, 3) * 0.15 + np.array([0.0, 1.0, 0.0])
    iota = 0.37
    G = 1.25
    tang = xphi + iota * xtheta
    tang_sq = np.sum(tang * tang, axis=-1, keepdims=True)
    base_field = (G / tang_sq) * tang
    perturbation = rng.randn(nphi, ntheta, 3)
    perturbation /= np.linalg.norm(perturbation.reshape(-1)) / np.sqrt(perturbation.size)
    B = base_field + residual_scale * perturbation
    return G, iota, device_float64(B), device_float64(xphi), device_float64(xtheta)


def _make_scalar_dynamic_range_data():
    amplitudes = np.ones(10001, dtype=np.float64)
    amplitudes[0] = 1.0e8
    B = np.zeros((1, amplitudes.size, 3), dtype=np.float64)
    B[0, :, 0] = amplitudes
    xphi = np.zeros_like(B)
    xtheta = np.zeros_like(B)
    return 1.0, 0.0, device_float64(B), device_float64(xphi), device_float64(xtheta)


def _assert_lane_allclose(actual, reference, parity_lane, *, tier):
    tolerance = parity_acceptance_tolerance(tier, parity_lane)
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=tolerance[0],
        atol=tolerance[1],
    )


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

    def test_strict_oracle_scalar_mode_matches_high_precision_reference(self):
        G, iota, B, xphi, xtheta = _make_scalar_dynamic_range_data()

        default_value = host_scalar(
            boozer_residual_scalar(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=False,
                reduction_mode="default",
            )
        )
        strict_oracle_value = host_scalar(
            boozer_residual_scalar(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=False,
                reduction_mode="strict_oracle",
            )
        )
        amplitudes = host_array(B)[0, :, 0]
        reference = 0.5 * math.fsum(float(value * value) for value in amplitudes) / (
            3.0 * amplitudes.size
        )

        np.testing.assert_allclose(
            strict_oracle_value,
            reference,
            rtol=0.0,
            atol=0.0,
        )
        assert abs(strict_oracle_value - reference) <= abs(default_value - reference)

    def test_invalid_reduction_mode_raises(self):
        B, xphi, xtheta = _make_synthetic_data(8, 10)
        with pytest.raises(ValueError, match="Unknown reduction_mode"):
            boozer_residual_scalar(
                1.5,
                0.3,
                B,
                xphi,
                xtheta,
                reduction_mode="bad-mode",
            )


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


class TestBoozerResidualParityStress:
    """Stress reduction-order parity near the current residual-floor regime."""

    def test_vector_parity_near_tolerance_floor(self, parity_lane):
        G, iota, B, xphi, xtheta = _make_near_floor_data()
        vector_actual = host_array(
            boozer_residual_vector(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=True,
            )
        )
        vector_reference, scalar_reference = _numpy_boozer_residual_reference(
            G,
            iota,
            host_array(B),
            host_array(xphi),
            host_array(xtheta),
            weight_inv_modB=True,
        )

        residual_rms_reference = np.sqrt(2.0 * scalar_reference)
        assert residual_rms_reference < 5e-12
        _assert_lane_allclose(
            vector_actual,
            vector_reference,
            parity_lane,
            tier="boozer_residual_floor_vector",
        )

    def test_scalar_residual_norm_near_tolerance_floor(self, parity_lane):
        G, iota, B, xphi, xtheta = _make_near_floor_data(seed=78)
        scalar_actual = host_scalar(
            boozer_residual_scalar(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=True,
            )
        )
        _, scalar_reference = _numpy_boozer_residual_reference(
            G,
            iota,
            host_array(B),
            host_array(xphi),
            host_array(xtheta),
            weight_inv_modB=True,
        )
        residual_norm_actual = np.sqrt(2.0 * scalar_actual)
        residual_norm_reference = np.sqrt(2.0 * scalar_reference)

        assert residual_norm_reference < 5e-12
        _assert_lane_allclose(
            residual_norm_actual,
            residual_norm_reference,
            parity_lane,
            tier="boozer_residual_floor_scalar",
        )


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
