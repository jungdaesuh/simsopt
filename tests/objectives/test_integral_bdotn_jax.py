"""
Parity tests for the JAX integral_BdotN implementation.

Validates:
1. All three definitions against direct NumPy computation.
2. Zero-target case (pure quadratic flux).
3. C++ parity (when simsoptpp is available).
"""

import math
from pathlib import Path
import sys

import pytest
import numpy as np

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

from simsopt.objectives.integral_bdotn_jax import integral_BdotN


@pytest.fixture(autouse=True)
def _parity_device_scope(parity_lane):
    with parity_default_device(parity_lane):
        yield


def _make_test_data(nphi=10, ntheta=12, seed=7):
    """Create synthetic B, target, and normal arrays."""
    rng = parity_rng(seed)
    B = rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0, 0, 1.0])
    target = rng.randn(nphi, ntheta) * 0.01
    normal = rng.randn(nphi, ntheta, 3) * 0.5
    # Make normals point outward-ish (positive z component)
    normal[..., 2] = np.abs(normal[..., 2]) + 0.1
    return device_float64(B), device_float64(target), device_float64(normal)


def _numpy_integral_BdotN(B, target, normal, definition):
    """Reference NumPy implementation for comparison."""
    nphi, ntheta, _ = B.shape
    norm_n = np.sqrt(np.sum(normal**2, axis=-1))
    unit_n = normal / norm_n[..., None]
    BdotN = np.sum(B * unit_n, axis=-1) - target

    if definition == "quadratic flux":
        return 0.5 * np.sum(BdotN**2 * norm_n) / (nphi * ntheta)
    elif definition == "normalized":
        B2 = np.sum(B**2, axis=-1)
        return 0.5 * np.sum(BdotN**2 * norm_n) / np.sum(B2 * norm_n)
    elif definition == "local":
        B2 = np.sum(B**2, axis=-1)
        return 0.5 * np.sum(BdotN**2 / B2 * norm_n) / (nphi * ntheta)


def _normalized_reduction_stress_data():
    """Return an odd-length, wide-dynamic-range case for denominator parity."""
    magnitudes = np.geomspace(1e-120, 1e120, num=257, dtype=np.float64)
    B = np.zeros((1, magnitudes.size, 3), dtype=np.float64)
    B[0, :, 0] = magnitudes
    target = np.zeros((1, magnitudes.size), dtype=np.float64)
    normal = np.zeros_like(B)
    normal[0, :, 0] = 1.0
    return device_float64(B), device_float64(target), device_float64(normal)


def _quadratic_flux_scalar_stress_data():
    """Return a dynamic-range case that exposes scalar contraction drift."""
    amplitudes = np.ones(10001, dtype=np.float64)
    amplitudes[0] = 1.0e8
    B = np.zeros((1, amplitudes.size, 3), dtype=np.float64)
    B[0, :, 0] = amplitudes
    target = np.zeros((1, amplitudes.size), dtype=np.float64)
    normal = np.zeros_like(B)
    normal[0, :, 0] = 1.0
    return device_float64(B), device_float64(target), device_float64(normal)


class TestIntegralBdotN:
    """Test all three definitions against NumPy reference."""

    @pytest.mark.parametrize(
        "definition",
        [
            "quadratic flux",
            "normalized",
            "local",
        ],
    )
    def test_parity_with_target(self, definition):
        B, target, normal = _make_test_data()
        J_jax = host_scalar(integral_BdotN(B, target, normal, definition))
        J_ref = _numpy_integral_BdotN(
            host_array(B), host_array(target), host_array(normal), definition
        )
        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-13)

    @pytest.mark.parametrize(
        "definition",
        [
            "quadratic flux",
            "normalized",
            "local",
        ],
    )
    def test_parity_zero_target(self, definition):
        B, _, normal = _make_test_data()
        target = jnp.zeros(B.shape[:2])
        J_jax = host_scalar(integral_BdotN(B, target, normal, definition))
        J_ref = _numpy_integral_BdotN(
            host_array(B), host_array(target), host_array(normal), definition
        )
        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-13)

    def test_positive_definite(self):
        """Quadratic flux should be non-negative."""
        B, target, normal = _make_test_data()
        J = float(integral_BdotN(B, target, normal, "quadratic flux"))
        assert J >= 0

    def test_zero_when_B_tangential(self):
        """If B is tangential to the surface (B·n = 0), flux should be zero."""
        nphi, ntheta = 8, 10
        rng = parity_rng(99)
        normal = rng.randn(nphi, ntheta, 3)
        norm_n = np.sqrt(np.sum(normal**2, axis=-1, keepdims=True))
        unit_n = normal / norm_n

        # Construct B perpendicular to n: B = e1 - (e1·n̂)n̂
        e1 = np.zeros((nphi, ntheta, 3))
        e1[..., 0] = 1.0
        B = e1 - np.sum(e1 * unit_n, axis=-1, keepdims=True) * unit_n
        # Add a small component so B isn't zero
        e2 = np.zeros((nphi, ntheta, 3))
        e2[..., 1] = 1.0
        B2 = e2 - np.sum(e2 * unit_n, axis=-1, keepdims=True) * unit_n
        B = B + 0.5 * B2

        target = jnp.zeros((nphi, ntheta))
        J = host_scalar(
            integral_BdotN(jnp.array(B), target, jnp.array(normal), "quadratic flux")
        )
        np.testing.assert_allclose(J, 0.0, atol=1e-25)

    def test_invalid_definition_raises(self):
        B, target, normal = _make_test_data()
        with pytest.raises(ValueError, match="Unknown definition"):
            integral_BdotN(B, target, normal, "invalid")

    def test_zero_normal_quadratic_flux_returns_zero(self):
        B = jnp.zeros((2, 3, 3))
        target = jnp.zeros((2, 3))
        normal = jnp.zeros((2, 3, 3))

        J = float(integral_BdotN(B, target, normal, "quadratic flux"))

        np.testing.assert_allclose(J, 0.0, atol=0.0)

    def test_zero_field_normalized_returns_inf(self):
        B = jnp.zeros((2, 3, 3))
        target = jnp.zeros((2, 3))
        normal = jnp.ones((2, 3, 3))

        J = float(integral_BdotN(B, target, normal, "normalized"))

        assert np.isinf(J)

    def test_zero_field_local_returns_inf(self):
        B = jnp.zeros((2, 3, 3))
        target = jnp.zeros((2, 3))
        normal = jnp.ones((2, 3, 3))

        J = float(integral_BdotN(B, target, normal, "local"))

        assert np.isinf(J)

    def test_zero_field_local_with_target_returns_inf(self):
        B = jnp.zeros((2, 3, 3))
        target = jnp.ones((2, 3))
        normal = jnp.ones((2, 3, 3))

        J = float(integral_BdotN(B, target, normal, "local"))

        assert np.isinf(J)

    def test_zero_normal_local_returns_zero(self):
        B = jnp.zeros((2, 3, 3))
        target = jnp.ones((2, 3))
        normal = jnp.zeros((2, 3, 3))

        J = float(integral_BdotN(B, target, normal, "local"))

        np.testing.assert_allclose(J, 0.0, atol=0.0)

    def test_normalized_reduction_stress_stays_on_contract(self, parity_lane):
        B, target, normal = _normalized_reduction_stress_data()
        rtol, atol = parity_acceptance_tolerance(
            "integral_bdotn_normalized_stress",
            parity_lane,
        )

        J_jax = host_scalar(integral_BdotN(B, target, normal, "normalized"))

        np.testing.assert_allclose(J_jax, 0.5, rtol=rtol, atol=atol)

    def test_strict_oracle_scalar_reduction_matches_high_precision_reference(self):
        B, target, normal = _quadratic_flux_scalar_stress_data()

        default_value = host_scalar(
            integral_BdotN(
                B,
                target,
                normal,
                "quadratic flux",
                reduction_mode="default",
            )
        )
        strict_oracle_value = host_scalar(
            integral_BdotN(
                B,
                target,
                normal,
                "quadratic flux",
                reduction_mode="strict_oracle",
            )
        )
        amplitudes = host_array(B)[0, :, 0]
        reference = (
            0.5
            * math.fsum(float(value * value) for value in amplitudes)
            / amplitudes.size
        )

        np.testing.assert_allclose(
            strict_oracle_value,
            reference,
            rtol=1e-15,
            atol=1e-4,
        )
        assert abs(strict_oracle_value - reference) < abs(default_value - reference)

    def test_invalid_reduction_mode_raises(self):
        B, target, normal = _make_test_data()
        with pytest.raises(ValueError, match="Unknown reduction_mode"):
            integral_BdotN(
                B,
                target,
                normal,
                "quadratic flux",
                reduction_mode="bad-mode",
            )


class TestIntegralBdotNCppParity:
    """Compare against simsoptpp.integral_BdotN (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")

    @pytest.mark.parametrize(
        "definition",
        [
            "quadratic flux",
            "normalized",
            "local",
        ],
    )
    def test_cpp_parity(self, definition):
        import simsoptpp as sopp

        B, target, normal = _make_test_data(nphi=15, ntheta=15)
        B_np = np.ascontiguousarray(host_array(B))
        target_np = np.ascontiguousarray(host_array(target))
        normal_np = np.ascontiguousarray(host_array(normal))

        J_cpp = sopp.integral_BdotN(B_np, target_np, normal_np, definition)
        J_jax = host_scalar(integral_BdotN(B, target, normal, definition))

        np.testing.assert_allclose(J_jax, J_cpp, rtol=1e-13)

    def test_cpp_zero_normal_quadratic_flux_returns_zero(self):
        import simsoptpp as sopp

        B = np.zeros((2, 3, 3))
        target = np.zeros((2, 3))
        normal = np.zeros((2, 3, 3))

        J_cpp = sopp.integral_BdotN(B, target, normal, "quadratic flux")
        J_jax = host_scalar(
            integral_BdotN(
                jnp.array(B), jnp.array(target), jnp.array(normal), "quadratic flux"
            )
        )

        np.testing.assert_allclose(J_cpp, 0.0, atol=0.0)
        np.testing.assert_allclose(J_jax, 0.0, atol=0.0)

    def test_cpp_zero_field_normalized_returns_inf(self):
        import simsoptpp as sopp

        B = np.zeros((2, 3, 3))
        target = np.zeros((2, 3))
        normal = np.ones((2, 3, 3))

        J_cpp = sopp.integral_BdotN(B, target, normal, "normalized")
        J_jax = host_scalar(
            integral_BdotN(
                jnp.array(B), jnp.array(target), jnp.array(normal), "normalized"
            )
        )

        assert np.isinf(J_cpp)
        assert np.isinf(J_jax)

    def test_cpp_zero_field_local_returns_inf(self):
        import simsoptpp as sopp

        B = np.zeros((2, 3, 3))
        target = np.zeros((2, 3))
        normal = np.ones((2, 3, 3))

        J_cpp = sopp.integral_BdotN(B, target, normal, "local")
        J_jax = host_scalar(
            integral_BdotN(jnp.array(B), jnp.array(target), jnp.array(normal), "local")
        )

        assert np.isinf(J_cpp)
        assert np.isinf(J_jax)

    def test_cpp_zero_field_local_with_target_returns_inf(self):
        import simsoptpp as sopp

        B = np.zeros((2, 3, 3))
        target = np.ones((2, 3))
        normal = np.ones((2, 3, 3))

        J_cpp = sopp.integral_BdotN(B, target, normal, "local")
        J_jax = host_scalar(
            integral_BdotN(jnp.array(B), jnp.array(target), jnp.array(normal), "local")
        )

        assert np.isinf(J_cpp)
        assert np.isinf(J_jax)

    def test_cpp_zero_normal_local_returns_zero(self):
        import simsoptpp as sopp

        B = np.zeros((2, 3, 3))
        target = np.ones((2, 3))
        normal = np.zeros((2, 3, 3))

        J_cpp = sopp.integral_BdotN(B, target, normal, "local")
        J_jax = host_scalar(
            integral_BdotN(jnp.array(B), jnp.array(target), jnp.array(normal), "local")
        )

        np.testing.assert_allclose(J_cpp, 0.0, atol=0.0)
        np.testing.assert_allclose(J_jax, 0.0, atol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
