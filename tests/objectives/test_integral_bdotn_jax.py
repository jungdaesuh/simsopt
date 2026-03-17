"""
Parity tests for the JAX integral_BdotN implementation.

Validates:
1. All three definitions against direct NumPy computation.
2. Zero-target case (pure quadratic flux).
3. C++ parity (when simsoptpp is available).
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

_ib = _load("integral_bdotn_jax", "objectives/integral_bdotn_jax.py")
integral_BdotN = _ib.integral_BdotN


def _make_test_data(nphi=10, ntheta=12, seed=7):
    """Create synthetic B, target, and normal arrays."""
    rng = np.random.RandomState(seed)
    B = rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0, 0, 1.0])
    target = rng.randn(nphi, ntheta) * 0.01
    normal = rng.randn(nphi, ntheta, 3) * 0.5
    # Make normals point outward-ish (positive z component)
    normal[..., 2] = np.abs(normal[..., 2]) + 0.1
    return jnp.array(B), jnp.array(target), jnp.array(normal)


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


class TestIntegralBdotN:
    """Test all three definitions against NumPy reference."""

    @pytest.mark.parametrize("definition", [
        "quadratic flux",
        "normalized",
        "local",
    ])
    def test_parity_with_target(self, definition):
        B, target, normal = _make_test_data()
        J_jax = float(integral_BdotN(B, target, normal, definition))
        J_ref = _numpy_integral_BdotN(
            np.array(B), np.array(target), np.array(normal), definition
        )
        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-13)

    @pytest.mark.parametrize("definition", [
        "quadratic flux",
        "normalized",
        "local",
    ])
    def test_parity_zero_target(self, definition):
        B, _, normal = _make_test_data()
        target = jnp.zeros(B.shape[:2])
        J_jax = float(integral_BdotN(B, target, normal, definition))
        J_ref = _numpy_integral_BdotN(
            np.array(B), np.array(target), np.array(normal), definition
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
        rng = np.random.RandomState(99)
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
        J = float(integral_BdotN(jnp.array(B), target, jnp.array(normal), "quadratic flux"))
        np.testing.assert_allclose(J, 0.0, atol=1e-25)

    def test_invalid_definition_raises(self):
        B, target, normal = _make_test_data()
        with pytest.raises(ValueError, match="Unknown definition"):
            integral_BdotN(B, target, normal, "invalid")


class TestIntegralBdotNCppParity:
    """Compare against simsoptpp.integral_BdotN (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")

    @pytest.mark.parametrize("definition", [
        "quadratic flux",
        "normalized",
        "local",
    ])
    def test_cpp_parity(self, definition):
        import simsoptpp as sopp

        B, target, normal = _make_test_data(nphi=15, ntheta=15)
        B_np = np.ascontiguousarray(np.array(B))
        target_np = np.ascontiguousarray(np.array(target))
        normal_np = np.ascontiguousarray(np.array(normal))

        J_cpp = sopp.integral_BdotN(B_np, target_np, normal_np, definition)
        J_jax = float(integral_BdotN(B, target, normal, definition))

        np.testing.assert_allclose(J_jax, J_cpp, rtol=1e-13)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
