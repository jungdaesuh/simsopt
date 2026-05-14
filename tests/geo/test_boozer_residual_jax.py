"""
Parity tests for the JAX Boozer residual scalar objective.

Validates:
1. Forward scalar value against the C++ ``sopp.boozer_residual`` oracle
   (acceptable oracle type 1 per ``tests/REVIEWER_ORACLE_LINT.md``).
2. Forward vector reduction against the same C++ scalar oracle via the
   ``0.5 * sum(r**2) / r.size`` scalarisation boundary; the public C++
   API exposes only the scalar residual, so per-component vector oracles
   are not invented (cf. ``src/simsoptpp/boozerresidual_py.cpp``).
3. Gradient via JAX autodiff against centred finite differences (oracle
   type 4 — FD of the JAX scalar). FD-only checks are kept FD-labelled
   and are not promoted to C++ parity claims.
4. Hessian symmetry plus FD-of-gradient consistency.

C++ oracle tests use ``pytest.importorskip("simsoptpp")`` so this file
remains runnable in pure-JAX environments without ``simsoptpp``.
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
    assert_arrays_on_device,
    device_float64,
    host_array,
    host_scalar,
    parity_acceptance_tolerance,
    parity_device,
    parity_default_device,
    parity_rng,
)
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.geo.boozer_residual_jax import (
    boozer_residual_scalar,
    boozer_residual_grad,
    boozer_residual_hessian,
    boozer_residual_vector,
)

_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct_kernel")


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


def _cpp_boozer_residual_scalar(G, iota, B, xphi, xtheta, *, weight_inv_modB):
    """C++ scalar oracle for the Boozer residual at a fixed (G, iota, surface).

    Wraps the top-level ``simsoptpp.boozer_residual`` symbol via the
    ABI-tolerant dispatcher
    ``simsopt.geo.boozersurface._call_boozer_residual``. The C++ kernel
    accumulates ``0.5 * sum_ij(r_ij**2)`` without normalisation
    (see ``src/simsoptpp/boozerresidual_impl.h``); dividing by
    ``num_res = 3 * nphi * ntheta`` recovers the JAX
    ``boozer_residual_scalar`` convention. The same vector→scalar
    boundary is used by the integration parity fixture in
    ``tests/integration/test_single_stage_jax_cpu_reference.py``.

    Oracle: C++ reference symbol (acceptable oracle type 1; see
    ``tests/REVIEWER_ORACLE_LINT.md``). Caller must call
    ``pytest.importorskip("simsoptpp")`` first to keep this module
    runnable in pure-JAX environments.
    """
    from simsopt.geo.boozersurface import _call_boozer_residual

    B_host = host_array(B, dtype=np.float64)
    xphi_host = host_array(xphi, dtype=np.float64)
    xtheta_host = host_array(xtheta, dtype=np.float64)
    val_raw = _call_boozer_residual(
        float(G),
        float(iota),
        xphi_host,
        xtheta_host,
        B_host,
        bool(weight_inv_modB),
    )
    num_res = 3 * B_host.shape[0] * B_host.shape[1]
    return float(val_raw) / num_res


def _numpy_cpu_ordered_boozer_scalar_reference(
    G,
    iota,
    B,
    xphi,
    xtheta,
    *,
    weight_inv_modB,
):
    tang = xphi + iota * xtheta
    B2 = B[..., 0] * B[..., 0] + B[..., 1] * B[..., 1] + B[..., 2] * B[..., 2]
    residual = G * B - B2[..., None] * tang
    if weight_inv_modB:
        residual = residual / np.sqrt(B2)[..., None]

    total = np.float64(0.0)
    for i in range(residual.shape[0]):
        for j in range(residual.shape[1]):
            r0 = residual[i, j, 0]
            r1 = residual[i, j, 1]
            r2 = residual[i, j, 2]
            total += r0 * r0 + r1 * r1 + r2 * r2
    return 0.5 * total / residual.size


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
    perturbation /= np.linalg.norm(perturbation.reshape(-1)) / np.sqrt(
        perturbation.size
    )
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

    @pytest.mark.parametrize("weight_inv_modB", [True, False])
    def test_scalar_matches_cpp_oracle(self, parity_lane, weight_inv_modB):
        """JAX scalar matches the C++ ``sopp.boozer_residual`` oracle.

        Oracle: C++ reference symbol ``simsoptpp.boozer_residual``
        invoked through ``simsopt.geo.boozersurface._call_boozer_residual``
        (acceptable oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``).
        Lane: ``direct_kernel`` (rtol=1e-10, atol=1e-12). Both the
        ``weight_inv_modB=True`` and ``False`` branches of the JAX kernel
        are anchored against the same C++ oracle.
        """
        pytest.importorskip("simsoptpp")

        B, xphi, xtheta = _make_synthetic_data(nphi=8, ntheta=10)
        G, iota = 1.5, 0.3

        J_jax = host_scalar(
            boozer_residual_scalar(
                G, iota, B, xphi, xtheta, weight_inv_modB=weight_inv_modB
            )
        )
        J_cpp = _cpp_boozer_residual_scalar(
            G, iota, B, xphi, xtheta, weight_inv_modB=weight_inv_modB
        )

        np.testing.assert_allclose(
            J_jax,
            J_cpp,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_cpu_ordered_reduction_matches_ordered_numpy_reference(self):
        """CPU-ordered mode mirrors the sopp point/component accumulation order."""
        nphi, ntheta = 7, 9
        B, xphi, xtheta = _make_synthetic_data(nphi, ntheta)
        G, iota = 1.5, 0.3

        B_np = host_array(B)
        xphi_np = host_array(xphi)
        xtheta_np = host_array(xtheta)
        reference = _numpy_cpu_ordered_boozer_scalar_reference(
            G,
            iota,
            B_np,
            xphi_np,
            xtheta_np,
            weight_inv_modB=True,
        )

        actual = host_scalar(
            boozer_residual_scalar(
                G,
                iota,
                B,
                xphi,
                xtheta,
                weight_inv_modB=True,
                reduction_mode="cpu_ordered",
            )
        )

        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=0.0)

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

    def test_weighted_zero_field_is_nonfinite(self):
        """Zero-field points are invalid in the 1/|B| weighted path."""
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

        assert not bool(jnp.isfinite(scalar_value))
        assert not bool(jnp.all(jnp.isfinite(residual_vector)))

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
        reference = (
            0.5
            * math.fsum(float(value * value) for value in amplitudes)
            / (3.0 * amplitudes.size)
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

    def test_vector_reduction_near_tolerance_floor_matches_cpp_scalar(
        self,
        parity_lane,
    ):
        """JAX vector reduces to the C++ scalar residual near the tolerance floor.

        The public C++ API exposes only the scalar
        ``simsoptpp.boozer_residual`` (see
        ``src/simsoptpp/boozerresidual_py.cpp``); there is no
        per-component vector oracle for ``boozer_residual_vector``. We
        therefore compare the JAX vector via the documented
        vector→scalar boundary ``0.5 * sum(r**2) / r.size`` (the JAX
        scalar definition; mirrored in
        ``src/simsoptpp/boozerresidual_impl.h`` modulo the JAX
        ``/ num_res`` normalisation) against the C++ scalar oracle.

        Oracle: C++ reference symbol ``simsoptpp.boozer_residual``
        (acceptable oracle type 1). Lanes: ``direct_kernel``
        (rtol=1e-10, atol=1e-12) for the scalar reduction and the
        parity-lane ``boozer_residual_floor_scalar`` tier — the latter
        keeps near-floor accumulation-order drift observable on CPU/GPU.
        """
        pytest.importorskip("simsoptpp")

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
        # vector→scalar boundary: 0.5 * sum(r**2) / r.size, matching the
        # JAX boozer_residual_scalar reduction.
        jax_scalar_from_vector = (
            0.5
            * float(np.sum(vector_actual.astype(np.float64) ** 2))
            / vector_actual.size
        )

        cpp_scalar = _cpp_boozer_residual_scalar(
            G, iota, B, xphi, xtheta, weight_inv_modB=True
        )

        # Guard the near-floor regime: the C++ scalar must stay below the
        # 5e-12 RMS floor so a future fixture drift cannot silently turn
        # this into a coarse-residual test.
        residual_rms_cpp = np.sqrt(2.0 * cpp_scalar)
        assert residual_rms_cpp < 5e-12

        np.testing.assert_allclose(
            jax_scalar_from_vector,
            cpp_scalar,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
        # Additional parity-lane scalar tolerance — same C++ oracle, kept
        # so the near-floor CPU/GPU contract remains observable.
        _assert_lane_allclose(
            jax_scalar_from_vector,
            cpp_scalar,
            parity_lane,
            tier="boozer_residual_floor_scalar",
        )

    def test_scalar_residual_norm_near_tolerance_floor_matches_cpp_oracle(
        self,
        parity_lane,
    ):
        """JAX scalar matches the C++ ``sopp.boozer_residual`` near the floor.

        Oracle: C++ reference symbol ``simsoptpp.boozer_residual``
        (acceptable oracle type 1). Lanes: ``direct_kernel`` (rtol=1e-10,
        atol=1e-12) plus the parity-lane ``boozer_residual_floor_scalar``
        tier — the latter keeps near-floor accumulation-order
        regressions observable.
        """
        pytest.importorskip("simsoptpp")

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
        scalar_cpp = _cpp_boozer_residual_scalar(
            G, iota, B, xphi, xtheta, weight_inv_modB=True
        )

        residual_norm_actual = np.sqrt(2.0 * scalar_actual)
        residual_norm_cpp = np.sqrt(2.0 * scalar_cpp)

        assert residual_norm_cpp < 5e-12
        np.testing.assert_allclose(
            scalar_actual,
            scalar_cpp,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
        _assert_lane_allclose(
            residual_norm_actual,
            residual_norm_cpp,
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


class TestBoozerResidualDevicePlacement:
    """Ensure scalar helpers stay on the active parity lane device."""

    def test_outputs_follow_lane_device(self, parity_lane):
        B, xphi, xtheta = _make_synthetic_data(6, 8)
        expected_device = parity_device(parity_lane)

        scalar_value = boozer_residual_scalar(1.5, 0.3, B, xphi, xtheta)
        gradient = boozer_residual_grad(1.5, 0.3, B, xphi, xtheta, nsurfdofs=0)
        hessian = boozer_residual_hessian(1.5, 0.3, B, xphi, xtheta, nsurfdofs=0)
        residual_vector = boozer_residual_vector(1.5, 0.3, B, xphi, xtheta)

        assert_arrays_on_device(expected_device, scalar_value)
        assert_arrays_on_device(expected_device, gradient)
        assert_arrays_on_device(expected_device, hessian)
        assert_arrays_on_device(expected_device, residual_vector)


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
