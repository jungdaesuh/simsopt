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
biot_savart_d2B_by_dXdX = _bs.biot_savart_d2B_by_dXdX
biot_savart_d2A_by_dXdX = _bs.biot_savart_d2A_by_dXdX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_float64_array(value):
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _device_float64_scalar(value):
    return _device_float64_array(value)


def _host_array(value):
    return np.asarray(jax.device_get(value))


def _host_float(value):
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


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
        _device_float64_array(gamma[None, :, :]),  # (1, nquad, 3)
        _device_float64_array(gammadash[None, :, :]),  # (1, nquad, 3)
    )


# Points matching upstream (near coil at y≈0.9, outside the wire).
_BASE_POINTS = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
_CURRENT = 1e4
_CURRENT_LINEARITY_TOL = 1e-15


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
    field_0 = _host_array(field_fn(points, gammas, gammadashs, currents))[idx]
    derivative = _host_array(derivative_fn(points, gammas, gammadashs, currents))[idx]

    for direction_host in [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    ]:
        direction = _device_float64_array(direction_host)
        directional_derivative = derivative.T @ direction_host
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            eps_device = _device_float64_scalar(eps)
            field_eps = _host_array(
                field_fn(points + eps_device * direction, gammas, gammadashs, currents)
            )[idx]
            derivative_est = (field_eps - field_0) / eps
            new_err = float(np.linalg.norm(directional_derivative - derivative_est))
            if new_err < 1e-14:
                break  # machine precision reached
            assert new_err < 0.55 * err
            err = new_err


def _assert_second_derivative_taylor_convergence(
    derivative_fn,
    second_derivative_fn,
    points,
    gammas,
    gammadashs,
    currents,
    idx,
):
    second_derivative = _host_array(
        second_derivative_fn(points, gammas, gammadashs, currents)
    )[idx]

    for d1 in range(3):
        for d2 in range(3):
            target = second_derivative[d1, d2]
            direction = _device_float64_array(np.eye(1, 3, k=d2, dtype=np.float64))
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                eps_device = _device_float64_scalar(eps)
                derivative_plus = _host_array(
                    derivative_fn(
                        points + eps_device * direction,
                        gammas,
                        gammadashs,
                        currents,
                    )
                )[idx, d1]
                derivative_minus = _host_array(
                    derivative_fn(
                        points - eps_device * direction,
                        gammas,
                        gammadashs,
                        currents,
                    )
                )[idx, d1]
                second_derivative_est = (derivative_plus - derivative_minus) / (2 * eps)
                new_err = float(np.linalg.norm(target - second_derivative_est))
                if new_err < 1e-13:
                    break
                assert new_err < 0.30 * err
                err = new_err


def _assert_current_linearity(quantity_fn, points, gammas, gammadashs, current):
    currents_full = _device_float64_array([current])
    currents_zero = _device_float64_array([0.0])
    currents_unit = _device_float64_array([1.0])

    quantity_full = quantity_fn(points, gammas, gammadashs, currents_full)
    quantity_zero = quantity_fn(points, gammas, gammadashs, currents_zero)
    quantity_unit = quantity_fn(points, gammas, gammadashs, currents_unit)
    quantity_full_host = _host_array(quantity_full)
    quantity_zero_host = _host_array(quantity_zero)
    quantity_unit_host = _host_array(quantity_unit)

    assert (
        float(np.linalg.norm(quantity_full_host - current * quantity_unit_host))
        < _CURRENT_LINEARITY_TOL
    )

    quantity_approx = (quantity_full_host - quantity_zero_host) / current
    assert (
        float(np.linalg.norm(quantity_approx - quantity_unit_host))
        < _CURRENT_LINEARITY_TOL
    )


class TestBiotSavartParitySuite:
    """Parity tests matching upstream tests/field/test_biotsavart.py."""

    def test_quadrature_convergence(self):
        """Multi-level quadrature refinement shows monotone error decay.

        Matches ``test_biotsavart_exponential_convergence``.
        Tests both B and dB/dX convergence across 4→8→16→32 refinement,
        with a machine-precision floor (the smooth Fourier coil converges
        super-exponentially, so coarser levels may already be at ε_mach).
        """
        points = _device_float64_array(_BASE_POINTS)
        currents = _device_float64_array([_CURRENT])

        g_ref, gd_ref = _make_fourier_coil(1000)
        B_ref = biot_savart_B(points, g_ref, gd_ref, currents)
        dB_ref = biot_savart_dB_by_dX(points, g_ref, gd_ref, currents)

        prev_B_err = float("inf")
        prev_dB_err = float("inf")
        for nq in [4, 8, 16, 32]:
            g, gd = _make_fourier_coil(nq)
            B = biot_savart_B(points, g, gd, currents)
            dB = biot_savart_dB_by_dX(points, g, gd, currents)

            B_err = _host_float(jnp.linalg.norm(B - B_ref))
            dB_err = _host_float(jnp.linalg.norm(dB - dB_ref))

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
        currents = _device_float64_array([_CURRENT])
        points = _device_float64_array(_BASE_POINTS)

        B = _host_array(biot_savart_B(points, gammas, gammadashs, currents))
        dA_dX = _host_array(biot_savart_dA_by_dX(points, gammas, gammadashs, currents))

        # curl(A)_i = ε_{ijk} ∂_j A_k
        curl_A = np.stack(
            [
                dA_dX[:, 1, 2] - dA_dX[:, 2, 1],
                dA_dX[:, 2, 0] - dA_dX[:, 0, 2],
                dA_dX[:, 0, 1] - dA_dX[:, 1, 0],
            ],
            axis=1,
        )

        np.testing.assert_allclose(curl_A, B, atol=1e-14)

    @pytest.mark.parametrize("idx", [0, 16])
    def test_dA_dX_finite_difference(self, idx):
        """dA/dX matches forward finite differences.

        Matches ``test_biotsavart_dAdX_taylortest``.
        """
        np.random.seed(42)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = _device_float64_array([_CURRENT])

        points = _device_float64_array(
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
        currents = _device_float64_array([_CURRENT])

        points = _device_float64_array(
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
        currents = _device_float64_array([_CURRENT])

        points = _device_float64_array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )
        dB_idx = _host_array(
            biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
        )[idx]

        # Divergence-free: Tr(dB/dX) = 0
        assert abs(dB_idx[0, 0] + dB_idx[1, 1] + dB_idx[2, 2]) < 1e-14
        # Symmetric in vacuum: ∂_j B_l = ∂_l B_j
        np.testing.assert_allclose(dB_idx, dB_idx.T, atol=1e-12)

    @pytest.mark.parametrize("idx", [0, 16])
    def test_d2B_dXdX_symmetric(self, idx):
        """d2B/dXdX is symmetric in its derivative axes.

        Matches ``test_d2B_by_dXdX_is_symmetric`` from the upstream suite.
        """
        np.random.seed(42)
        gammas, gammadashs = _make_fourier_coil(200)
        currents = _device_float64_array([_CURRENT])
        points = _device_float64_array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )

        d2B = _host_array(
            biot_savart_d2B_by_dXdX(points, gammas, gammadashs, currents)
        )[idx]
        for component in range(3):
            np.testing.assert_allclose(
                d2B[:, :, component],
                d2B[:, :, component].T,
                atol=1e-14,
            )

    @pytest.mark.parametrize("idx", [0, 16])
    @pytest.mark.parametrize(
        "derivative_fn,second_derivative_fn",
        [
            (biot_savart_dB_by_dX, biot_savart_d2B_by_dXdX),
            (biot_savart_dA_by_dX, biot_savart_d2A_by_dXdX),
        ],
        ids=["d2B_dXdX", "d2A_dXdX"],
    )
    def test_d2_dXdX_taylor_test(self, derivative_fn, second_derivative_fn, idx):
        """Second spatial derivative matches central FD of first derivative."""
        gammas, gammadashs = _make_fourier_coil(200)
        currents = _device_float64_array([_CURRENT])
        points = _device_float64_array(_BASE_POINTS)
        _assert_second_derivative_taylor_convergence(
            derivative_fn,
            second_derivative_fn,
            points,
            gammas,
            gammadashs,
            currents,
            idx,
        )

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
        currents = _device_float64_array([_CURRENT])

        points = _device_float64_array(
            _BASE_POINTS + 0.001 * (np.random.rand(*_BASE_POINTS.shape) - 0.5)
        )

        inputs = [gammas, gammadashs, currents]
        B = biot_savart_B(points, *inputs)
        J0 = _host_float(jnp.sum(jnp.square(B)))

        vjp_out = biot_savart_B_vjp(points, B, *inputs)
        grad = vjp_out[channel_idx]

        h = _device_float64_scalar(1e-2) * _device_float64_array(
            np.random.rand(*inputs[channel_idx].shape)
        )
        dJ_dh = 2.0 * _host_float(jnp.sum(grad * h))

        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            eps_device = _device_float64_scalar(eps)
            perturbed = list(inputs)
            perturbed[channel_idx] = inputs[channel_idx] + eps_device * h
            B_eps = biot_savart_B(points, *perturbed)
            J_eps = _host_float(jnp.sum(jnp.square(B_eps)))
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
        points = _device_float64_array(_BASE_POINTS)
        currents = _device_float64_array([_CURRENT])

        g_ref, gd_ref = _make_fourier_coil(1000)
        A_ref = biot_savart_A(points, g_ref, gd_ref, currents)
        dA_ref = biot_savart_dA_by_dX(points, g_ref, gd_ref, currents)

        prev_A_err = float("inf")
        prev_dA_err = float("inf")
        for nq in [4, 8, 16, 32]:
            g, gd = _make_fourier_coil(nq)
            A = biot_savart_A(points, g, gd, currents)
            dA = biot_savart_dA_by_dX(points, g, gd, currents)

            A_err = _host_float(jnp.linalg.norm(A - A_ref))
            dA_err = _host_float(jnp.linalg.norm(dA - dA_ref))

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
        points = _device_float64_array(_BASE_POINTS)
        I = _CURRENT

        for quantity_fn in (
            biot_savart_B,
            biot_savart_dB_by_dX,
            biot_savart_A,
            biot_savart_dA_by_dX,
            biot_savart_d2B_by_dXdX,
            biot_savart_d2A_by_dXdX,
        ):
            _assert_current_linearity(quantity_fn, points, gammas, gammadashs, I)


# ---------------------------------------------------------------------------
# Test: Per-coil current-linearity (W2-B6)
# ---------------------------------------------------------------------------


def _make_shifted_fourier_coils(nquad, shifts):
    """Build a multi-coil JAX-only fixture from translated Fourier coils.

    Each coil is the ``_make_fourier_coil`` curve translated by the
    corresponding 3-vector in ``shifts``. Returns stacked
    ``(ncoils, nquad, 3)`` arrays.
    """
    gammas_list = []
    gammadashs_list = []
    for shift in shifts:
        gamma, gammadash = _make_fourier_coil(nquad)
        gammas_list.append(gamma[0] + _device_float64_array(shift))
        # Translation does not change gammadash.
        gammadashs_list.append(gammadash[0])
    return (
        _device_float64_array(jnp.stack(gammas_list, axis=0)),
        _device_float64_array(jnp.stack(gammadashs_list, axis=0)),
    )


class TestBiotSavartCoilCurrentLinearity:
    """Type-3 (FD-on-the-JAX-stack) per-coil current-linearity coverage.

    These are NOT C++ parity oracles — they verify the JAX kernel's
    exact linearity in coil current for the per-coil decomposition
    that ``BiotSavartJAX.dB_by_dcoilcurrents()`` /
    ``dA_by_dcoilcurrents()`` return. The aggregate
    ``test_B_and_dB_linearity_in_current`` at ``:490`` covers the
    Σ_k linearity but NOT the per-coil decomposition.

    Upstream reference: ``test_biotsavart_coil_current_taylortest``
    in ``simsopt/tests/field/test_biotsavart.py:276`` and its
    vector-potential analog at ``:402``.
    """

    _SHIFTS = (
        (0.0, 0.0, 0.0),
        (0.05, -0.03, 0.02),
        (-0.04, 0.06, -0.01),
    )
    _BASELINE_CURRENTS = (1e4, 7.5e3, 1.2e4)
    _EPS = 1e-3

    def _per_coil_unit_field(self, kernel_fn, gammas, gammadashs, points):
        """Per-coil unit-current field — matches BiotSavartJAX.{dB,dA}_by_dcoilcurrents.

        For coil ``k``, returns ``kernel_fn(points, gammas[k:k+1],
        gammadashs[k:k+1], jnp.array([1.0]))``. The identity
        ``dB/dI_k = b_k(x)`` follows from ``B(I) = Σ_k I_k · b_k``.
        """
        unit_current = _device_float64_array([1.0])
        return [
            kernel_fn(points, gammas[k : k + 1], gammadashs[k : k + 1], unit_current)
            for k in range(gammas.shape[0])
        ]

    def _assert_per_coil_linearity(self, kernel_fn):
        gammas, gammadashs = _make_shifted_fourier_coils(nquad=200, shifts=self._SHIFTS)
        points = _device_float64_array(_BASE_POINTS)
        ncoils = gammas.shape[0]
        baseline = _device_float64_array(self._BASELINE_CURRENTS)
        eps = self._EPS

        per_coil_unit = self._per_coil_unit_field(kernel_fn, gammas, gammadashs, points)

        field_baseline = _host_array(kernel_fn(points, gammas, gammadashs, baseline))

        for k in range(ncoils):
            unit_k = _host_array(per_coil_unit[k])

            plus = baseline.at[k].add(eps)
            minus = baseline.at[k].add(-eps)
            field_plus = _host_array(kernel_fn(points, gammas, gammadashs, plus))
            field_minus = _host_array(kernel_fn(points, gammas, gammadashs, minus))

            central_fd = (field_plus - field_minus) / (2.0 * eps)
            central_err = float(np.linalg.norm(central_fd - unit_k))
            assert central_err < _CURRENT_LINEARITY_TOL, (
                f"coil {k}: central FD mismatch {central_err:.2e}"
            )

            forward_residual = field_plus - field_baseline - eps * unit_k
            forward_err = float(np.linalg.norm(forward_residual))
            assert forward_err < _CURRENT_LINEARITY_TOL, (
                f"coil {k}: forward FD mismatch {forward_err:.2e}"
            )

    def test_dB_by_dcoilcurrents_per_coil_linearity(self):
        """Per-coil ``dB/dI_k`` equals the per-coil unit-current B field.

        Verifies the exact-linear identity that
        ``BiotSavartJAX.dB_by_dcoilcurrents()`` returns, using FD on the
        pure-JAX ``biot_savart_B`` kernel.
        """
        self._assert_per_coil_linearity(biot_savart_B)

    def test_dA_by_dcoilcurrents_per_coil_linearity(self):
        """Per-coil ``dA/dI_k`` equals the per-coil unit-current A field.

        Vector-potential analogue of
        ``test_dB_by_dcoilcurrents_per_coil_linearity``.
        """
        self._assert_per_coil_linearity(biot_savart_A)


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
        g1 = _device_float64_array(g1_np)
        gd1 = _device_float64_array(gd1_np)
        c1 = _device_float64_array([1e5, 1e5])

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
        g2 = _device_float64_array(g2_np)
        gd2 = _device_float64_array(gd2_np)
        c2 = _device_float64_array([1e5])

        coil_arrays = [(g1, gd1, c1), (g2, gd2, c2)]

        # Evaluation points inside the coil set
        rng = np.random.RandomState(10)
        pts_R = 0.8 + 0.2 * rng.rand(15)
        pts_phi = twopi * rng.rand(15)
        pts_z = 0.1 * (rng.rand(15) - 0.5)
        points = _device_float64_array(
            np.stack(
                [pts_R * np.cos(pts_phi), pts_R * np.sin(pts_phi), pts_z],
                axis=-1,
            )
        )

        def J(ca):
            B = grouped_biot_savart_B(points, ca)
            return jnp.sum(jnp.square(B))

        _, pullback = jax.vjp(J, coil_arrays)
        grad_ca = pullback(_device_float64_scalar(1.0))[0]

        def _check_fd(grad, h, perturb, label):
            """Central FD convergence check for one gradient component."""
            dJ_dh = _host_float(jnp.sum(grad * h))
            err = 1e9
            for i in range(8, 18):
                eps = 0.5**i
                eps_device = _device_float64_scalar(eps)
                fd = (
                    _host_float(J(perturb(eps_device * h)))
                    - _host_float(J(perturb(-eps_device * h)))
                ) / (2 * eps)
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
            _device_float64_array(1e-2 * rng.randn(*g1.shape)),
            lambda d: [(g1 + d, gd1, c1), (g2, gd2, c2)],
            "Group 1 gammas",
        )
        _check_fd(
            grad_ca[1][0],
            _device_float64_array(1e-2 * rng.randn(*g2.shape)),
            lambda d: [(g1, gd1, c1), (g2 + d, gd2, c2)],
            "Group 2 gammas",
        )
        _check_fd(
            grad_ca[1][2],
            _device_float64_array(rng.randn(*c2.shape)),
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
    points = (centroid + _device_float64_array([0.0, 0.0, 0.05]))[None, :]
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
        gamma_np = _host_array(gamma)
        gammadash_np = _host_array(gammadash)

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
        B = biot_savart_B(points, gammas, gammadashs, _device_float64_array([1e4]))
        B_norm = _host_float(jnp.linalg.norm(B))
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
        points = _device_float64_array(_host_array(centroid)[None, :] + offsets)

        dB = biot_savart_dB_by_dX(
            points, gammas, gammadashs, _device_float64_array([1e4])
        )
        dB_np = _host_array(dB)
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
        B_full = biot_savart_B(points, gammas, gammadashs, _device_float64_array([I]))
        B_unit = biot_savart_B(points, gammas, gammadashs, _device_float64_array([1.0]))
        err = float(np.linalg.norm(_host_array(B_full) - I * _host_array(B_unit)))
        assert err < 1e-15, f"{curvetype}: B linearity error {err:.2e}"

    @pytest.mark.parametrize("curvetype", list(_CURVE_SPEC_FACTORIES))
    def test_B_cross_type_consistency(self, curvetype):
        """B field changes non-trivially when curve DOFs are perturbed."""
        spec, _, _, gammas, gammadashs, points = _make_curve_type_fixture(curvetype)
        currents = _device_float64_array([1e4])
        B_orig = biot_savart_B(points, gammas, gammadashs, currents)

        np.random.seed(99)
        perturbed_dofs = _device_float64_array(
            _host_array(spec.dofs) + 0.05 * np.random.rand(len(spec.dofs))
        )
        from simsopt.jax_core import curve_spec_with_dofs

        perturbed_spec = curve_spec_with_dofs(spec, perturbed_dofs)
        gamma_p, gammadash_p = curve_gamma_and_dash_from_spec(perturbed_spec)
        centroid_p = jnp.mean(gamma_p, axis=0)
        points_p = (centroid_p + _device_float64_array([0.0, 0.0, 0.05]))[None, :]
        B_pert = biot_savart_B(
            points_p, gamma_p[None, :, :], gammadash_p[None, :, :], currents
        )
        diff = _host_float(jnp.linalg.norm(B_pert - B_orig))
        assert diff > 1e-10, (
            f"{curvetype}: B unchanged after DOF perturbation (diff={diff:.2e})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
