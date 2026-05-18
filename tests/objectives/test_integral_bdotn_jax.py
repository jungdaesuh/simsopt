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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_REPO_ROOT / "src")

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import (
    device_float64,
    host_array,
    host_scalar,
    parity_acceptance_tolerance,
    parity_default_device,
    parity_rng,
)

from simsopt.objectives.integral_bdotn_jax import (
    integral_BdotN,
    residual_BdotN,
    signed_BdotN_flux,
)

_FD_GRADIENT_TOLS = parity_ladder_tolerances("fd-gradient")
_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct_kernel")
_DEFINITIONS = ("quadratic flux", "normalized", "local")


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


def _numpy_residual_BdotN(B, target, normal, definition):
    """Closed-form NumPy per-point residual oracle."""
    nphi, ntheta, _ = B.shape
    norm_n = np.sqrt(np.sum(normal * normal, axis=-1))
    has_normal = norm_n > 0.0
    safe_norm_n = np.where(has_normal, norm_n, 1.0)
    unit_n = np.where(has_normal[..., None], normal / safe_norm_n[..., None], 0.0)
    BdotN = np.sum(B * unit_n, axis=-1) - target

    if definition == "quadratic flux":
        weight = np.where(has_normal, norm_n / (nphi * ntheta), 0.0)
        residual = np.where(has_normal, BdotN * np.sqrt(weight), 0.0)
    elif definition == "normalized":
        B2 = np.sum(B * B, axis=-1)
        denominator = np.sum(B2 * norm_n)
        safe_denominator = denominator if denominator > 0.0 else 1.0
        point_weight = np.where(has_normal, norm_n / safe_denominator, 0.0)
        residual = np.where(
            denominator > 0.0,
            np.where(has_normal, BdotN * np.sqrt(point_weight), 0.0),
            np.full_like(BdotN, np.inf),
        )
    elif definition == "local":
        B2 = np.sum(B * B, axis=-1)
        singular = has_normal & (B2 <= 0.0)
        safe_B2 = np.where(B2 > 0.0, B2, 1.0)
        weight = np.where(has_normal, norm_n / (safe_B2 * (nphi * ntheta)), 0.0)
        invalid_residual = np.reciprocal(np.where(singular, B2, np.ones_like(B2)))
        residual = np.where(
            singular,
            invalid_residual,
            np.where(has_normal, BdotN * np.sqrt(weight), 0.0),
        )
    else:
        raise ValueError(f"Unknown definition: {definition!r}")
    return np.ravel(residual)


def _numpy_signed_BdotN_flux(B, normal):
    nphi, ntheta, _ = B.shape
    return np.sum(np.sum(B * normal, axis=-1)) / (nphi * ntheta)


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


def _closed_torus_normals(nphi=24, ntheta=32):
    """Return unnormalized parametric torus normals with exact signed closure."""
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    major_radius = 1.7
    minor_radius = 0.35
    radius = major_radius + minor_radius * np.cos(theta_grid)
    normal = np.empty((nphi, ntheta, 3), dtype=np.float64)
    normal[..., 0] = minor_radius * radius * np.cos(theta_grid) * np.cos(phi_grid)
    normal[..., 1] = minor_radius * radius * np.cos(theta_grid) * np.sin(phi_grid)
    normal[..., 2] = minor_radius * radius * np.sin(theta_grid)
    return normal


class TestIntegralBdotN:
    """Test all three definitions against NumPy reference."""

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_parity_with_target(self, definition):
        B, target, normal = _make_test_data()
        J_jax = host_scalar(integral_BdotN(B, target, normal, definition))
        J_ref = _numpy_integral_BdotN(
            host_array(B), host_array(target), host_array(normal), definition
        )
        np.testing.assert_allclose(J_jax, J_ref, rtol=1e-13)

    @pytest.mark.parametrize("definition", _DEFINITIONS)
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

    def test_signed_flux_keeps_raw_orientation(self):
        """A uniform field has zero signed flux through the closed torus grid."""
        normal = _closed_torus_normals()
        B = np.zeros_like(normal)
        B[..., 2] = 1.0

        signed_flux = host_scalar(
            signed_BdotN_flux(device_float64(B), device_float64(normal))
        )
        magnitude_flux = np.mean(np.abs(np.sum(B * normal, axis=-1)))

        np.testing.assert_allclose(
            signed_flux,
            0.0,
            atol=float(_FD_GRADIENT_TOLS["directional_derivative_floor"]),
        )
        assert magnitude_flux > 0.0

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_residual_matches_closed_form_numpy_per_point_oracle(self, definition):
        B, target, normal = _make_test_data(nphi=5, ntheta=7, seed=221)
        actual = host_array(residual_BdotN(B, target, normal, definition))
        expected = _numpy_residual_BdotN(
            host_array(B), host_array(target), host_array(normal), definition
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_signed_flux_matches_closed_form_numpy_oracle(self):
        B, _, normal = _make_test_data(nphi=5, ntheta=7, seed=223)
        actual = host_scalar(signed_BdotN_flux(B, normal))
        expected = _numpy_signed_BdotN_flux(host_array(B), host_array(normal))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

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

    def test_target_shape_mismatch_raises(self):
        B = jnp.ones((1, 1, 3), dtype=jnp.float64)
        target = jnp.ones((5, 7), dtype=jnp.float64)
        normal = jnp.ones((1, 1, 3), dtype=jnp.float64)

        with pytest.raises(ValueError, match="target.shape must match"):
            integral_BdotN(B, target, normal, "quadratic flux")

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_empty_target_matches_zero_target_contract(self, definition):
        B, _, normal = _make_test_data(nphi=5, ntheta=7, seed=281)
        empty_target = jnp.asarray([], dtype=jnp.float64)
        zero_target = jnp.zeros(B.shape[:2], dtype=jnp.float64)

        with jax.transfer_guard("disallow"):
            actual_device = integral_BdotN(B, empty_target, normal, definition)
        actual = host_scalar(actual_device)
        expected = host_scalar(integral_BdotN(B, zero_target, normal, definition))

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_bcoil_shape_mismatch_raises(self):
        B = jnp.ones((2, 3, 1), dtype=jnp.float64)
        target = jnp.zeros((2, 3), dtype=jnp.float64)
        normal = jnp.ones((2, 3, 3), dtype=jnp.float64)

        with pytest.raises(ValueError, match="Bcoil must have shape"):
            residual_BdotN(B, target, normal, "quadratic flux")
        with pytest.raises(ValueError, match="Bcoil must have shape"):
            integral_BdotN(B, target, normal, "quadratic flux")
        with pytest.raises(ValueError, match="Bcoil must have shape"):
            signed_BdotN_flux(B, normal)

    def test_normal_shape_mismatch_raises(self):
        B = jnp.ones((2, 3, 3), dtype=jnp.float64)
        target = jnp.zeros((2, 3), dtype=jnp.float64)
        normal = jnp.ones((1, 1, 3), dtype=jnp.float64)

        with pytest.raises(ValueError, match="normal.shape must match"):
            residual_BdotN(B, target, normal, "quadratic flux")
        with pytest.raises(ValueError, match="normal.shape must match"):
            integral_BdotN(B, target, normal, "quadratic flux")
        with pytest.raises(ValueError, match="normal.shape must match"):
            signed_BdotN_flux(B, normal)

    def test_float32_public_kernel_contract_preserves_dtype(self):
        B = jnp.ones((2, 3, 3), dtype=jnp.float32)
        target = jnp.zeros((2, 3), dtype=jnp.float32)
        normal = jnp.ones((2, 3, 3), dtype=jnp.float32)

        residual = residual_BdotN(B, target, normal, "quadratic flux")
        objective = integral_BdotN(B, target, normal, "quadratic flux")
        signed_flux = signed_BdotN_flux(B, normal)

        assert residual.dtype == jnp.float32
        assert objective.dtype == jnp.float32
        assert signed_flux.dtype == jnp.float32

    def test_complex_public_kernel_contract_raises(self):
        B = jnp.ones((2, 3, 3), dtype=jnp.complex64)
        target = jnp.zeros((2, 3), dtype=jnp.float32)
        normal = jnp.ones((2, 3, 3), dtype=jnp.float32)

        with pytest.raises(ValueError, match="Bcoil must be real-valued"):
            residual_BdotN(B, target, normal, "quadratic flux")
        with pytest.raises(ValueError, match="Bcoil must be real-valued"):
            integral_BdotN(B, target, normal, "quadratic flux")
        with pytest.raises(ValueError, match="Bcoil must be real-valued"):
            signed_BdotN_flux(B, normal)

    @pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_empty_mesh_scalar_objective_matches_cpp_boundary_contract(
        self, shape, definition
    ):
        B = jnp.zeros((*shape, 3), dtype=jnp.float64)
        target = jnp.zeros(shape, dtype=jnp.float64)
        normal = jnp.zeros((*shape, 3), dtype=jnp.float64)

        value = host_scalar(integral_BdotN(B, target, normal, definition))

        if definition == "normalized":
            assert np.isposinf(value)
        else:
            assert math.isnan(value)

    def test_empty_mesh_invalid_definition_still_raises(self):
        B = jnp.zeros((0, 3, 3), dtype=jnp.float64)
        target = jnp.zeros((0, 3), dtype=jnp.float64)
        normal = jnp.zeros((0, 3, 3), dtype=jnp.float64)

        with pytest.raises(ValueError, match="Unknown definition"):
            integral_BdotN(B, target, normal, "invalid")

    def test_empty_mesh_invalid_reduction_mode_still_raises(self):
        B = jnp.zeros((0, 3, 3), dtype=jnp.float64)
        target = jnp.zeros((0, 3), dtype=jnp.float64)
        normal = jnp.zeros((0, 3, 3), dtype=jnp.float64)

        with pytest.raises(ValueError, match="Unknown reduction_mode"):
            integral_BdotN(
                B,
                target,
                normal,
                "quadratic flux",
                reduction_mode="bad-mode",
            )

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_empty_integer_mesh_returns_float_boundary_value(self, definition):
        B = jnp.zeros((0, 3, 3), dtype=jnp.int32)
        target = jnp.zeros((0, 3), dtype=jnp.int32)
        normal = jnp.zeros((0, 3, 3), dtype=jnp.int32)

        with jax.transfer_guard("disallow"):
            value = integral_BdotN(B, target, normal, definition)

        assert jnp.issubdtype(value.dtype, jnp.floating)
        value_host = host_scalar(value)
        if definition == "normalized":
            assert np.isposinf(value_host)
        else:
            assert math.isnan(value_host)

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

    def test_zero_field_local_gradient_is_nonfinite(self):
        B = jnp.zeros((2, 3, 3), dtype=jnp.float64)
        target = jnp.zeros((2, 3), dtype=jnp.float64)
        normal = jnp.ones((2, 3, 3), dtype=jnp.float64)

        def objective(B_arg):
            return integral_BdotN(B_arg, target, normal, "local")

        J = float(objective(B))
        grad = host_array(jax.grad(objective)(B))

        assert np.isinf(J)
        assert not np.all(np.isfinite(grad))

    def test_zero_normal_local_returns_zero(self):
        B = jnp.zeros((2, 3, 3))
        target = jnp.ones((2, 3))
        normal = jnp.zeros((2, 3, 3))

        J = float(integral_BdotN(B, target, normal, "local"))

        np.testing.assert_allclose(J, 0.0, atol=0.0)

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_zero_normal_gradient_stays_finite(self, definition):
        B = jnp.ones((1, 2, 3), dtype=jnp.float64)
        target = jnp.zeros((1, 2), dtype=jnp.float64)
        normal = jnp.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=jnp.float64,
        )

        def objective(normal_arg):
            return integral_BdotN(B, target, normal_arg, definition)

        grad = host_array(jax.grad(objective)(normal))

        assert np.all(np.isfinite(grad))

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_zero_normal_masks_nonfinite_inactive_inputs(self, definition):
        B = jnp.array(
            [[[jnp.nan, jnp.nan, jnp.nan], [0.0, 0.0, 2.0]]],
            dtype=jnp.float64,
        )
        target = jnp.array([[jnp.nan, 1.0]], dtype=jnp.float64)
        normal = jnp.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=jnp.float64,
        )

        def objective(B_arg, target_arg, normal_arg):
            return integral_BdotN(B_arg, target_arg, normal_arg, definition)

        value, grads = jax.value_and_grad(objective, argnums=(0, 1, 2))(
            B,
            target,
            normal,
        )

        assert np.isfinite(host_scalar(value))
        for grad in grads:
            assert np.all(np.isfinite(host_array(grad)))

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

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_cpp_parity(self, definition):
        import simsoptpp as sopp

        B, target, normal = _make_test_data(nphi=15, ntheta=15)
        B_np = np.ascontiguousarray(host_array(B))
        target_np = np.ascontiguousarray(host_array(target))
        normal_np = np.ascontiguousarray(host_array(normal))

        J_cpp = sopp.integral_BdotN(B_np, target_np, normal_np, definition)
        J_jax = host_scalar(integral_BdotN(B, target, normal, definition))

        np.testing.assert_allclose(J_jax, J_cpp, rtol=1e-13)

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_cpp_parity_production_scale_64x64_direct_kernel_gate(self, definition):
        """64x64 C++ parity gate for the production-scale flux reducer."""
        import simsoptpp as sopp

        B, target, normal = _make_test_data(nphi=64, ntheta=64, seed=11)
        B_np = np.ascontiguousarray(host_array(B))
        target_np = np.ascontiguousarray(host_array(target))
        normal_np = np.ascontiguousarray(host_array(normal))

        J_cpp = sopp.integral_BdotN(B_np, target_np, normal_np, definition)
        J_jax = host_scalar(integral_BdotN(B, target, normal, definition))

        np.testing.assert_allclose(
            J_jax,
            J_cpp,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_cpp_empty_target_contract_matches_jax(self, definition):
        import simsoptpp as sopp

        B, _, normal = _make_test_data(nphi=5, ntheta=7, seed=283)
        B_np = np.ascontiguousarray(host_array(B))
        target_np = np.ascontiguousarray(np.asarray([], dtype=np.float64))
        normal_np = np.ascontiguousarray(host_array(normal))

        J_cpp = sopp.integral_BdotN(B_np, target_np, normal_np, definition)
        J_jax = host_scalar(
            integral_BdotN(
                B,
                jnp.asarray(target_np),
                normal,
                definition,
            )
        )

        np.testing.assert_allclose(J_jax, J_cpp, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_cpp_empty_mesh_boundary_contract_matches_jax(self, shape, definition):
        import simsoptpp as sopp

        B = np.ascontiguousarray(np.zeros((*shape, 3), dtype=np.float64))
        target = np.ascontiguousarray(np.zeros(shape, dtype=np.float64))
        normal = np.ascontiguousarray(np.zeros((*shape, 3), dtype=np.float64))

        J_cpp = sopp.integral_BdotN(B, target, normal, definition)
        J_jax = host_scalar(
            integral_BdotN(
                jnp.asarray(B),
                jnp.asarray(target),
                jnp.asarray(normal),
                definition,
            )
        )

        if np.isposinf(J_cpp):
            assert np.isposinf(J_jax)
        else:
            assert math.isnan(J_cpp)
            assert math.isnan(J_jax)

    @pytest.mark.parametrize(
        ("definition", "B", "target", "normal", "expected"),
        [
            (
                "quadratic flux",
                np.zeros((2, 3, 3), dtype=np.float64),
                np.ones((2, 3), dtype=np.float64),
                np.zeros((2, 3, 3), dtype=np.float64),
                0.0,
            ),
            (
                "normalized",
                np.zeros((2, 3, 3), dtype=np.float64),
                np.zeros((2, 3), dtype=np.float64),
                np.ones((2, 3, 3), dtype=np.float64),
                np.inf,
            ),
            (
                "local",
                np.zeros((2, 3, 3), dtype=np.float64),
                np.ones((2, 3), dtype=np.float64),
                np.ones((2, 3, 3), dtype=np.float64),
                np.inf,
            ),
            (
                "local",
                np.zeros((2, 3, 3), dtype=np.float64),
                np.ones((2, 3), dtype=np.float64),
                np.zeros((2, 3, 3), dtype=np.float64),
                0.0,
            ),
        ],
    )
    def test_cpp_boundary_contract_matches_jax(
        self, definition, B, target, normal, expected
    ):
        import simsoptpp as sopp

        B = np.ascontiguousarray(B)
        target = np.ascontiguousarray(target)
        normal = np.ascontiguousarray(normal)

        J_cpp = sopp.integral_BdotN(B, target, normal, definition)
        J_jax = host_scalar(
            integral_BdotN(
                jnp.asarray(B),
                jnp.asarray(target),
                jnp.asarray(normal),
                definition,
            )
        )

        if np.isinf(expected):
            assert np.isinf(J_cpp)
            assert np.isinf(J_jax)
        else:
            np.testing.assert_allclose(J_cpp, expected, atol=0.0)
            np.testing.assert_allclose(J_jax, expected, atol=0.0)


class TestIntegralBdotNBoundaryContracts:
    """Boundary behavior that is defined by the JAX implementation."""

    def test_zero_normal_quadratic_flux_returns_zero(self):
        B = np.zeros((2, 3, 3))
        target = np.zeros((2, 3))
        normal = np.zeros((2, 3, 3))

        J_jax = host_scalar(
            integral_BdotN(
                jnp.array(B), jnp.array(target), jnp.array(normal), "quadratic flux"
            )
        )

        np.testing.assert_allclose(J_jax, 0.0, atol=0.0)

    def test_zero_field_normalized_returns_inf(self):
        B = np.zeros((2, 3, 3))
        target = np.zeros((2, 3))
        normal = np.ones((2, 3, 3))

        J_jax = host_scalar(
            integral_BdotN(
                jnp.array(B), jnp.array(target), jnp.array(normal), "normalized"
            )
        )

        assert np.isinf(J_jax)

    def test_zero_field_local_returns_inf(self):
        B = np.zeros((2, 3, 3))
        target = np.zeros((2, 3))
        normal = np.ones((2, 3, 3))

        J_jax = host_scalar(
            integral_BdotN(jnp.array(B), jnp.array(target), jnp.array(normal), "local")
        )

        assert np.isinf(J_jax)

    def test_zero_field_local_with_target_returns_inf(self):
        B = np.zeros((2, 3, 3))
        target = np.ones((2, 3))
        normal = np.ones((2, 3, 3))

        J_jax = host_scalar(
            integral_BdotN(jnp.array(B), jnp.array(target), jnp.array(normal), "local")
        )

        assert np.isinf(J_jax)

    def test_zero_normal_local_returns_zero(self):
        B = np.zeros((2, 3, 3))
        target = np.ones((2, 3))
        normal = np.zeros((2, 3, 3))

        J_jax = host_scalar(
            integral_BdotN(jnp.array(B), jnp.array(target), jnp.array(normal), "local")
        )

        np.testing.assert_allclose(J_jax, 0.0, atol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
