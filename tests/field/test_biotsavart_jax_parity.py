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
5. B VJP Taylor test — gammas channel (reverse-mode derivative)
6. B VJP Taylor test — gammadash channel
7. B VJP Taylor test — currents channel
8. Grouped Biot-Savart gradient FD (mixed quadrature self-consistency)

No simsoptpp dependency — all tests use pure JAX functions.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(__file__).resolve().parents[2] / "src")

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
grouped_biot_savart_B = _bs.grouped_biot_savart_B


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


def _assert_point_perturbation_taylor_convergence(
    field_fn,
    derivative_fn,
    points,
    gammas,
    gammadashs,
    currents,
    idx,
):
    field_0 = field_fn(points, gammas, gammadashs, currents)[idx]
    derivative = derivative_fn(points, gammas, gammadashs, currents)[idx]

    for direction in [
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
    ]:
        directional_derivative = derivative.T @ direction
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            field_eps = field_fn(
                points + eps * direction, gammas, gammadashs, currents
            )[idx]
            derivative_est = (field_eps - field_0) / eps
            new_err = float(jnp.linalg.norm(directional_derivative - derivative_est))
            if new_err < 1e-14:
                break  # machine precision reached
            assert new_err < 0.55 * err
            err = new_err


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
        _assert_point_perturbation_taylor_convergence(
            biot_savart_A,
            biot_savart_dA_by_dX,
            points,
            gammas,
            gammadashs,
            currents,
            idx,
        )

    @pytest.mark.parametrize("idx", [0, 16])
    def test_dB_dX_taylor_test(self, idx):
        """dB/dX matches forward finite differences under point perturbations.

        Matches ``test_biotsavart_dBdX_taylortest`` from the upstream suite.
        This is the multi-epsilon convergence gate requested by review item P1.
        """
        np.random.seed(42)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = jnp.array([_CURRENT])

        points = jnp.array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )
        _assert_point_perturbation_taylor_convergence(
            biot_savart_B,
            biot_savart_dB_by_dX,
            points,
            gammas,
            gammadashs,
            currents,
            idx,
        )

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

    @pytest.mark.parametrize(
        "channel_idx,seed",
        [(0, 1), (1, 2), (2, 3)],
        ids=["gammas", "gammadashs", "currents"],
    )
    def test_B_vjp_taylor_test(self, channel_idx, seed):
        """VJP Taylor test: |J(x+εh) − J(x) − ε⟨dJ,h⟩| decays as O(ε).

        Validates each channel of biot_savart_B_vjp (gammas, gammadashs,
        currents) via forward FD convergence.
        """
        np.random.seed(seed)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = jnp.array([_CURRENT])

        points = jnp.array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )

        inputs = [gammas, gammadashs, currents]
        B = biot_savart_B(points, *inputs)
        J0 = float(jnp.sum(B**2))

        vjp_out = biot_savart_B_vjp(points, B, *inputs)
        grad = vjp_out[channel_idx]

        h = 1e-2 * jnp.array(np.random.rand(*inputs[channel_idx].shape))
        dJ_dh = float(2 * jnp.sum(grad * h))

        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            perturbed = list(inputs)
            perturbed[channel_idx] = inputs[channel_idx] + eps * h
            B_eps = biot_savart_B(points, *perturbed)
            J_eps = float(jnp.sum(B_eps**2))
            deriv_est = (J_eps - J0) / eps
            new_err = abs(deriv_est - dJ_dh)
            if new_err < 1e-14:
                break
            assert new_err < 0.55 * err
            err = new_err

    def test_A_quadrature_convergence(self):
        """A converges monotonically under quadrature refinement.

        Mirrors ``test_quadrature_convergence`` for the vector potential.
        """
        points = jnp.array(_BASE_POINTS)
        currents = jnp.array([_CURRENT])

        g_ref, gd_ref = _make_fourier_coil(1000)
        A_ref = biot_savart_A(points, g_ref, gd_ref, currents)
        dA_ref = biot_savart_dA_by_dX(points, g_ref, gd_ref, currents)

        prev_A_err = float("inf")
        prev_dA_err = float("inf")
        for nq in [4, 8, 16, 32]:
            g, gd = _make_fourier_coil(nq)
            A = biot_savart_A(points, g, gd, currents)
            dA = biot_savart_dA_by_dX(points, g, gd, currents)

            A_err = float(jnp.linalg.norm(A - A_ref))
            dA_err = float(jnp.linalg.norm(dA - dA_ref))

            if prev_A_err > 1e-14:
                assert A_err < 0.5 * prev_A_err, (
                    f"A not converging at nq={nq}: {A_err:.2e} vs {prev_A_err:.2e}"
                )
            if prev_dA_err > 1e-14:
                assert dA_err < 0.5 * prev_dA_err, (
                    f"dA not converging at nq={nq}: {dA_err:.2e} vs {prev_dA_err:.2e}"
                )

            prev_A_err = A_err
            prev_dA_err = dA_err

    def test_B_and_dB_linearity_in_current(self):
        """B and dB/dX are exactly linear in coil current.

        Matches ``test_biotsavart_coil_current_taylortest``.  Because the
        Biot-Savart field is strictly linear in I, a single large FD step
        recovers the derivative to machine precision — no convergence
        series needed.
        """
        gammas, gammadashs = _make_fourier_coil(200)
        points = jnp.array(_BASE_POINTS)
        I = _CURRENT

        currents_full = jnp.array([I])
        currents_zero = jnp.array([0.0])
        currents_unit = jnp.array([1.0])

        # --- B linearity ---
        B_full = biot_savart_B(points, gammas, gammadashs, currents_full)
        B_zero = biot_savart_B(points, gammas, gammadashs, currents_zero)
        B_unit = biot_savart_B(points, gammas, gammadashs, currents_unit)

        # B(I) = I * B(1)  (exact identity)
        assert float(jnp.linalg.norm(B_full - I * B_unit)) < 1e-15

        # (B(I) - B(0)) / I = B(1)
        dB_approx = (B_full - B_zero) / I
        assert float(jnp.linalg.norm(dB_approx - B_unit)) < 1e-15

        # --- dB/dX linearity ---
        dB_full = biot_savart_dB_by_dX(points, gammas, gammadashs, currents_full)
        dB_zero = biot_savart_dB_by_dX(points, gammas, gammadashs, currents_zero)
        dB_unit = biot_savart_dB_by_dX(points, gammas, gammadashs, currents_unit)

        # dB/dX(I) = I * dB/dX(1)  (exact identity)
        assert float(jnp.linalg.norm(dB_full - I * dB_unit)) < 1e-15

        # (dB/dX(I) - dB/dX(0)) / I = dB/dX(1)
        ddB_approx = (dB_full - dB_zero) / I
        assert float(jnp.linalg.norm(ddB_approx - dB_unit)) < 1e-15

        # --- A linearity ---
        A_full = biot_savart_A(points, gammas, gammadashs, currents_full)
        A_zero = biot_savart_A(points, gammas, gammadashs, currents_zero)
        A_unit = biot_savart_A(points, gammas, gammadashs, currents_unit)

        assert float(jnp.linalg.norm(A_full - I * A_unit)) < 1e-15
        dA_approx = (A_full - A_zero) / I
        assert float(jnp.linalg.norm(dA_approx - A_unit)) < 1e-15

        # --- dA/dX linearity ---
        dA_full = biot_savart_dA_by_dX(points, gammas, gammadashs, currents_full)
        dA_zero = biot_savart_dA_by_dX(points, gammas, gammadashs, currents_zero)
        dA_unit = biot_savart_dA_by_dX(points, gammas, gammadashs, currents_unit)

        assert float(jnp.linalg.norm(dA_full - I * dA_unit)) < 1e-15
        ddA_approx = (dA_full - dA_zero) / I
        assert float(jnp.linalg.norm(ddA_approx - dA_unit)) < 1e-15


# ---------------------------------------------------------------------------
# Test: Grouped Biot-Savart gradient (mixed quadrature)
# ---------------------------------------------------------------------------


class TestGroupedBiotSavartGradient:
    """Validate gradient through grouped_biot_savart_B for mixed quadrature.

    Section 8d: verifies that mixed-quadrature grouping preserves
    gradient accuracy on the JAX side (self-consistency test).
    """

    def test_mixed_quad_gradient_fd(self):
        """Gradient of J=ΣB² through mixed-quad grouped B matches central FD.

        Creates two coil groups (32-point and 64-point quadrature) and
        verifies that jax.grad through grouped_biot_savart_B matches
        central finite differences to O(ε²) convergence.
        """
        R_coil = 1.5
        twopi = 2 * np.pi

        # Group 1: 2 coils with 32 quadrature points
        nq1 = 32
        g1_np = np.zeros((2, nq1, 3))
        gd1_np = np.zeros((2, nq1, 3))
        for i in range(2):
            phi_off = twopi * i / 3
            t = np.linspace(0, twopi, nq1, endpoint=False)
            g1_np[i, :, 0] = R_coil * np.cos(t + phi_off)
            g1_np[i, :, 1] = R_coil * np.sin(t + phi_off)
            gd1_np[i, :, 0] = -R_coil * twopi * np.sin(t + phi_off)
            gd1_np[i, :, 1] = R_coil * twopi * np.cos(t + phi_off)
        g1 = jnp.array(g1_np)
        gd1 = jnp.array(gd1_np)
        c1 = jnp.array([1e5, 1e5])

        # Group 2: 1 coil with 64 quadrature points
        nq2 = 64
        phi_off = twopi * 2 / 3
        t = np.linspace(0, twopi, nq2, endpoint=False)
        g2_np = np.zeros((1, nq2, 3))
        gd2_np = np.zeros((1, nq2, 3))
        g2_np[0, :, 0] = R_coil * np.cos(t + phi_off)
        g2_np[0, :, 1] = R_coil * np.sin(t + phi_off)
        gd2_np[0, :, 0] = -R_coil * twopi * np.sin(t + phi_off)
        gd2_np[0, :, 1] = R_coil * twopi * np.cos(t + phi_off)
        g2 = jnp.array(g2_np)
        gd2 = jnp.array(gd2_np)
        c2 = jnp.array([1e5])

        coil_arrays = [(g1, gd1, c1), (g2, gd2, c2)]

        # Evaluation points inside the coil set
        rng = np.random.RandomState(10)
        pts_R = 0.8 + 0.2 * rng.rand(15)
        pts_phi = twopi * rng.rand(15)
        pts_z = 0.1 * (rng.rand(15) - 0.5)
        points = jnp.array(
            np.stack(
                [pts_R * np.cos(pts_phi), pts_R * np.sin(pts_phi), pts_z],
                axis=-1,
            )
        )

        def J(ca):
            B = grouped_biot_savart_B(points, ca)
            return jnp.sum(B**2)

        grad_ca = jax.grad(J)(coil_arrays)

        def _check_fd(grad, h, perturb, label):
            """Central FD convergence check for one gradient component."""
            dJ_dh = float(jnp.sum(grad * h))
            err = 1e9
            for i in range(8, 18):
                eps = 0.5**i
                fd = (float(J(perturb(eps * h))) - float(J(perturb(-eps * h)))) / (
                    2 * eps
                )
                new_err = abs(fd - dJ_dh)
                if new_err < 1e-12:
                    break
                assert new_err < 0.35 * err, (
                    f"{label}: err={new_err:.2e}, "
                    f"prev={err:.2e}, ratio={new_err / err:.3f}"
                )
                err = new_err

        _check_fd(
            grad_ca[0][0],
            jnp.array(1e-2 * rng.randn(*g1.shape)),
            lambda d: [(g1 + d, gd1, c1), (g2, gd2, c2)],
            "Group 1 gammas",
        )
        _check_fd(
            grad_ca[1][0],
            jnp.array(1e-2 * rng.randn(*g2.shape)),
            lambda d: [(g1, gd1, c1), (g2 + d, gd2, c2)],
            "Group 2 gammas",
        )
        _check_fd(
            grad_ca[1][2],
            jnp.array(rng.randn(*c2.shape)),
            lambda d: [(g1, gd1, c1), (g2, gd2, c2 + d)],
            "Group 2 currents",
        )


# ---------------------------------------------------------------------------
# Test: Curve type parametrization (P20)
# ---------------------------------------------------------------------------

from simsopt.jax_core import (
    make_curve_xyzfourier_spec,
    make_curve_rzfourier_spec,
    make_curve_helical_spec,
    make_curve_planarfourier_spec,
    curve_gamma_and_dash_from_spec,
)


def _make_spec_xyzfourier(quadpoints, order, rand_scale, rng):
    """Create a CurveXYZFourierSpec matching upstream get_curve DOF layout."""
    ndofs = 3 * (2 * order + 1)
    dofs = np.zeros(ndofs)
    # Upstream convention: dofs[1]=1 (xs1), dofs[2*order+3]=1 (yc1),
    # dofs[4*order+3]=1 (zc1)
    dofs[1] = 1.0
    dofs[2 * order + 3] = 1.0
    dofs[4 * order + 3] = 1.0
    dofs = dofs + rand_scale * rng.rand(ndofs)
    return make_curve_xyzfourier_spec(
        dofs=dofs,
        quadpoints=quadpoints,
        order=order,
    )


def _make_spec_rzfourier(quadpoints, order, rand_scale, rng):
    """Create a CurveRZFourierSpec (stellsym=True, nfp=2) matching upstream."""
    nfp = 2
    stellsym = True
    # stellsym=True: ndofs = (order+1) + order
    ndofs = (order + 1) + order
    dofs = np.zeros(ndofs)
    # Upstream: dofs[0]=1 (rc0), dofs[1]=0.1 (rc1), dofs[order+1]=0.1 (zs1)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    dofs = dofs + rand_scale * rng.rand(ndofs)
    # RZFourier quadpoints are in [0, 1/nfp) by convention
    qp_rz = quadpoints / nfp
    return make_curve_rzfourier_spec(
        dofs=dofs,
        quadpoints=qp_rz,
        order=order,
        nfp=nfp,
        stellsym=stellsym,
    )


def _make_spec_helical(quadpoints, order, rand_scale, rng):
    """Create a CurveHelicalSpec matching upstream get_curve parameters."""
    m, ell, R0, r = 5, 2, 1.0, 0.3
    ndofs = 1 + 2 * order
    dofs = np.zeros(ndofs)
    # Upstream: dofs[0] = pi/2
    dofs[0] = np.pi / 2
    dofs = dofs + rand_scale * rng.rand(ndofs)
    return make_curve_helical_spec(
        dofs=dofs,
        quadpoints=quadpoints,
        order=order,
        m=m,
        ell=ell,
        R0=R0,
        r=r,
    )


def _make_spec_planarfourier(quadpoints, order, rand_scale, rng):
    """Create a CurvePlanarFourierSpec matching upstream get_curve DOF layout."""
    # DOF layout: [rc0..rc_order, rs1..rs_order, q0, qi, qj, qk, X, Y, Z]
    ndofs = (order + 1) + order + 4 + 3
    dofs = np.zeros(ndofs)
    # Upstream: dofs[0]=1 (rc0), dofs[1]=0.1 (rc1), dofs[order+1]=0.1 (rs1)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    # Set quaternion q0=1 for a valid identity rotation before perturbation
    q_start = (order + 1) + order
    dofs[q_start] = 1.0
    dofs = dofs + rand_scale * rng.rand(ndofs)
    return make_curve_planarfourier_spec(
        dofs=dofs,
        quadpoints=quadpoints,
        order=order,
    )


_CURVE_SPEC_FACTORIES = {
    "CurveXYZFourier": _make_spec_xyzfourier,
    "CurveRZFourier": _make_spec_rzfourier,
    "CurveHelical": _make_spec_helical,
    "CurvePlanarFourier": _make_spec_planarfourier,
}


def _make_curve_type_fixture(curvetype, nquad=100):
    """Build a single-coil field fixture for the given curve type."""
    np.random.seed(2)
    quadpoints = np.linspace(0, 1, nquad, endpoint=False)
    spec = _CURVE_SPEC_FACTORIES[curvetype](
        quadpoints, order=4, rand_scale=0.01, rng=np.random
    )
    gamma, gammadash = curve_gamma_and_dash_from_spec(spec)
    gammas = gamma[None, :, :]
    gammadashs = gammadash[None, :, :]
    centroid = jnp.mean(gamma, axis=0)
    points = (centroid + jnp.array([0.0, 0.0, 0.05]))[None, :]
    return spec, gamma, gammadash, gammas, gammadashs, points


class TestCurveTypeParametrization:
    """Validate that biot_savart_B produces consistent, non-trivial results
    across all four core curve type parametrizations via the JAX spec system.

    Review item P20: parametrized curve-type coverage for JAX Biot-Savart.
    """

    @pytest.mark.parametrize("curvetype", list(_CURVE_SPEC_FACTORIES))
    def test_gamma_nontrivial(self, curvetype):
        """Gamma and gammadash from each spec type are non-degenerate."""
        _, gamma, gammadash, _, _, _ = _make_curve_type_fixture(curvetype)
        gamma_np = np.asarray(gamma)
        gammadash_np = np.asarray(gammadash)

        assert gamma_np.shape == (100, 3)
        gamma_extent = gamma_np.max(axis=0) - gamma_np.min(axis=0)
        assert np.max(gamma_extent) > 0.01, (
            f"{curvetype}: curve gamma has negligible spatial extent"
        )
        gammadash_norms = np.linalg.norm(gammadash_np, axis=1)
        assert np.min(gammadash_norms) > 1e-10, (
            f"{curvetype}: gammadash has near-zero entries"
        )

    @pytest.mark.parametrize("curvetype", list(_CURVE_SPEC_FACTORIES))
    def test_B_field_nontrivial(self, curvetype):
        """B field from each curve type is non-zero and has physical magnitude."""
        _, _, _, gammas, gammadashs, points = _make_curve_type_fixture(curvetype)
        B = biot_savart_B(points, gammas, gammadashs, jnp.array([1e4]))
        B_norm = float(jnp.linalg.norm(B))
        assert B_norm > 1e-10, (
            f"{curvetype}: B field norm {B_norm:.2e} is negligibly small"
        )

    @pytest.mark.parametrize("curvetype", list(_CURVE_SPEC_FACTORIES))
    def test_dB_dX_divergence_free(self, curvetype):
        """Divergence of B is zero for each curve type (Maxwell constraint)."""
        _, gamma, _, gammas, gammadashs, _ = _make_curve_type_fixture(curvetype)
        centroid = jnp.mean(gamma, axis=0)
        rng = np.random.RandomState(42)
        offsets = 0.05 * (rng.rand(5, 3) - 0.5)
        points = jnp.array(np.asarray(centroid)[None, :] + offsets)

        dB = biot_savart_dB_by_dX(points, gammas, gammadashs, jnp.array([1e4]))
        dB_np = np.asarray(dB)
        for i in range(dB_np.shape[0]):
            div_B = dB_np[i, 0, 0] + dB_np[i, 1, 1] + dB_np[i, 2, 2]
            assert abs(div_B) < 1e-12, (
                f"{curvetype} point {i}: div(B) = {div_B:.2e} is not zero"
            )

    @pytest.mark.parametrize("curvetype", list(_CURVE_SPEC_FACTORIES))
    def test_B_linearity_in_current(self, curvetype):
        """B scales linearly with current for each curve type."""
        _, _, _, gammas, gammadashs, points = _make_curve_type_fixture(curvetype)
        I = 1e4
        B_full = biot_savart_B(points, gammas, gammadashs, jnp.array([I]))
        B_unit = biot_savart_B(points, gammas, gammadashs, jnp.array([1.0]))
        err = float(jnp.linalg.norm(B_full - I * B_unit))
        assert err < 1e-15, f"{curvetype}: B linearity error {err:.2e}"

    @pytest.mark.parametrize("curvetype", list(_CURVE_SPEC_FACTORIES))
    def test_B_cross_type_consistency(self, curvetype):
        """B field changes non-trivially when curve DOFs are perturbed."""
        spec, _, _, gammas, gammadashs, points = _make_curve_type_fixture(curvetype)
        currents = jnp.array([1e4])
        B_orig = biot_savart_B(points, gammas, gammadashs, currents)

        np.random.seed(99)
        perturbed_dofs = jnp.asarray(spec.dofs) + 0.05 * jnp.array(
            np.random.rand(len(spec.dofs))
        )
        from simsopt.jax_core import curve_spec_with_dofs

        perturbed_spec = curve_spec_with_dofs(spec, perturbed_dofs)
        gamma_p, gammadash_p = curve_gamma_and_dash_from_spec(perturbed_spec)
        centroid_p = jnp.mean(gamma_p, axis=0)
        points_p = (centroid_p + jnp.array([0.0, 0.0, 0.05]))[None, :]
        B_pert = biot_savart_B(
            points_p, gamma_p[None, :, :], gammadash_p[None, :, :], currents
        )
        diff = float(jnp.linalg.norm(B_pert - B_orig))
        assert diff > 1e-10, (
            f"{curvetype}: B unchanged after DOF perturbation (diff={diff:.2e})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
