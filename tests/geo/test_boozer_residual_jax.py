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

import simsopt.geo.boozer_residual_jax as _brj
from simsopt.geo.boozer_residual_jax import (
    _split_decision_vector,
    _unpack_decision_vector,
    boozer_penalty_hvp_composed,
    boozer_residual_jacobian_composed,
    boozer_residual_jvp_composed,
    boozer_residual_scalar,
    boozer_residual_grad,
    boozer_residual_hessian,
    boozer_residual_vjp_composed,
    boozer_residual_vector,
)
from simsopt.geo.label_constraints_jax import compute_G_from_currents

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


def test_split_decision_vector_rejects_missing_tail_entries():
    """Decision vectors must contain iota and, when optimized, G."""

    with pytest.raises(ValueError, match="decision vector length 1 is too short"):
        _split_decision_vector(jnp.asarray([0.3], dtype=jnp.float64), optimize_G=True)

    with pytest.raises(ValueError, match="decision vector length 0 is too short"):
        _split_decision_vector(jnp.asarray([], dtype=jnp.float64), optimize_G=False)


def test_unpack_decision_vector_uses_current_ssot_when_G_not_optimized():
    """The no-``G`` path uses ``compute_G_from_currents`` for all coils."""

    sdofs = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)
    iota = jnp.asarray(0.4, dtype=jnp.float64)
    x = jnp.concatenate([sdofs, jnp.asarray([iota], dtype=jnp.float64)])
    currents_a = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    currents_b = jnp.asarray([-3.0], dtype=jnp.float64)
    empty = jnp.zeros((0, 3), dtype=jnp.float64)

    actual_sdofs, actual_iota, actual_G = _unpack_decision_vector(
        x,
        [(empty, empty, currents_a), (empty, empty, currents_b)],
        optimize_G=False,
    )
    expected_G = compute_G_from_currents(jnp.concatenate([currents_a, currents_b]))

    np.testing.assert_allclose(host_array(actual_sdofs), host_array(sdofs))
    np.testing.assert_allclose(host_scalar(actual_iota), host_scalar(iota))
    np.testing.assert_allclose(host_scalar(actual_G), host_scalar(expected_G))


def test_composed_jacobian_uses_reverse_mode_when_residual_is_smaller(monkeypatch):
    calls = {"linearize": 0, "vjp": 0}
    original_linearize = _brj.jax.linearize
    original_vjp = _brj.jax.vjp

    def fake_residual(x, **kwargs):
        del kwargs
        return jnp.stack([x[0] + 2.0 * x[1], x[1] - x[2], jnp.sum(x)])

    def recording_linearize(*args, **kwargs):
        calls["linearize"] += 1
        return original_linearize(*args, **kwargs)

    def recording_vjp(*args, **kwargs):
        calls["vjp"] += 1
        return original_vjp(*args, **kwargs)

    monkeypatch.setattr(_brj, "_boozer_residual_vector_composed", fake_residual)
    monkeypatch.setattr(_brj.jax, "linearize", recording_linearize)
    monkeypatch.setattr(_brj.jax, "vjp", recording_vjp)

    x = jnp.arange(5.0, dtype=jnp.float64)
    residual, jacobian = boozer_residual_jacobian_composed(
        x,
        quadpoints_phi=np.zeros(1, dtype=np.float64),
        quadpoints_theta=np.zeros(1, dtype=np.float64),
    )

    expected_jacobian = np.asarray(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(host_array(residual), host_array(fake_residual(x)))
    np.testing.assert_allclose(host_array(jacobian), expected_jacobian)
    assert calls == {"linearize": 0, "vjp": 1}


def test_composed_jacobian_uses_forward_mode_when_decision_vector_is_smaller(
    monkeypatch,
):
    calls = {"linearize": 0, "vjp": 0}
    original_linearize = _brj.jax.linearize
    original_vjp = _brj.jax.vjp

    def fake_residual(x, **kwargs):
        del kwargs
        return jnp.stack(
            [
                x[0],
                x[1],
                x[0] + x[1],
                x[0] - x[1],
                2.0 * x[0],
                3.0 * x[1],
            ]
        )

    def recording_linearize(*args, **kwargs):
        calls["linearize"] += 1
        return original_linearize(*args, **kwargs)

    def recording_vjp(*args, **kwargs):
        calls["vjp"] += 1
        return original_vjp(*args, **kwargs)

    monkeypatch.setattr(_brj, "_boozer_residual_vector_composed", fake_residual)
    monkeypatch.setattr(_brj.jax, "linearize", recording_linearize)
    monkeypatch.setattr(_brj.jax, "vjp", recording_vjp)

    x = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    residual, jacobian = boozer_residual_jacobian_composed(
        x,
        quadpoints_phi=np.zeros(1, dtype=np.float64),
        quadpoints_theta=np.zeros(2, dtype=np.float64),
    )

    expected_jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(host_array(residual), host_array(fake_residual(x)))
    np.testing.assert_allclose(host_array(jacobian), expected_jacobian)
    assert calls == {"linearize": 1, "vjp": 0}


def test_composed_penalty_hvp_matches_dense_hessian_product(monkeypatch):
    matrix = jnp.asarray(
        [
            [2.0, -0.5, 0.25],
            [-0.5, 3.0, 0.75],
            [0.25, 0.75, 1.5],
        ],
        dtype=jnp.float64,
    )

    def fake_penalty(x, *, scale, offset):
        quadratic = 0.5 * x @ (matrix @ x)
        return scale * (quadratic + jnp.sum(jnp.sin(x))) + offset

    monkeypatch.setattr(_brj, "boozer_penalty_composed", fake_penalty)

    x = jnp.asarray([0.2, -0.3, 0.5], dtype=jnp.float64)
    v = jnp.asarray([1.25, -0.75, 0.4], dtype=jnp.float64)
    scale = 1.7
    offset = -0.2

    actual = boozer_penalty_hvp_composed(x, v, scale=scale, offset=offset)
    dense_hessian = _brj.jax.hessian(
        lambda y: fake_penalty(y, scale=scale, offset=offset)
    )(x)
    expected = dense_hessian @ v

    np.testing.assert_allclose(host_array(actual), host_array(expected))


def test_composed_residual_jvp_vjp_match_jax_products(monkeypatch):
    def fail_dense_jacobian(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dense Jacobian should not be materialized")

    def fake_residual(x, *, scale):
        return scale * jnp.asarray(
            [
                x[0] + x[1] ** 2,
                x[2] * jnp.sin(x[0]),
                x[0] * x[1] * x[2],
                jnp.sum(x**3),
            ],
            dtype=x.dtype,
        )

    monkeypatch.setattr(_brj, "boozer_residual_jacobian_composed", fail_dense_jacobian)
    monkeypatch.setattr(_brj, "_boozer_residual_vector_composed", fake_residual)

    x = jnp.asarray([0.3, -0.4, 0.8], dtype=jnp.float64)
    tangent = jnp.asarray([1.1, -0.2, 0.5], dtype=jnp.float64)
    cotangent = jnp.asarray([0.7, -1.2, 0.25, 0.9], dtype=jnp.float64)
    scale = 2.3
    residual_fn = lambda y: fake_residual(y, scale=scale)

    actual_residual, actual_jvp = boozer_residual_jvp_composed(
        x,
        tangent,
        scale=scale,
    )
    expected_residual, expected_jvp = _brj.jax.jvp(
        residual_fn,
        (x,),
        (tangent,),
    )

    vjp_residual, actual_vjp = boozer_residual_vjp_composed(
        x,
        cotangent,
        scale=scale,
    )
    expected_vjp_residual, expected_vjp_fn = _brj.jax.vjp(residual_fn, x)
    (expected_vjp,) = expected_vjp_fn(cotangent)

    np.testing.assert_allclose(
        host_array(actual_residual),
        host_array(expected_residual),
    )
    np.testing.assert_allclose(host_array(actual_jvp), host_array(expected_jvp))
    np.testing.assert_allclose(
        host_array(vjp_residual),
        host_array(expected_vjp_residual),
    )
    np.testing.assert_allclose(host_array(actual_vjp), host_array(expected_vjp))


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

    def test_weighted_zero_field_surfaces_nonfinite_contract(self):
        nphi, ntheta = 2, 3
        B = device_float64(np.zeros((nphi, ntheta, 3), dtype=np.float64))
        xphi = device_float64(np.ones((nphi, ntheta, 3), dtype=np.float64))
        xtheta = device_float64(np.ones((nphi, ntheta, 3), dtype=np.float64))

        weighted = host_scalar(
            boozer_residual_scalar(
                1.0,
                0.5,
                B,
                xphi,
                xtheta,
                weight_inv_modB=True,
            )
        )
        unweighted = host_scalar(
            boozer_residual_scalar(
                1.0,
                0.5,
                B,
                xphi,
                xtheta,
                weight_inv_modB=False,
            )
        )

        assert math.isnan(weighted)
        assert unweighted == pytest.approx(0.0, abs=0.0)

    def test_float32_field_inputs_are_rejected_under_float64_policy(self):
        B, xphi, xtheta = _make_synthetic_data(nphi=2, ntheta=3)
        G, iota = 1.5, 0.3

        with pytest.raises(TypeError, match="B must have runtime dtype float64"):
            boozer_residual_scalar(
                G,
                iota,
                jnp.asarray(B, dtype=jnp.float32),
                xphi,
                xtheta,
            )

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
