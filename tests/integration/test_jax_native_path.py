"""
End-to-end test for the JAX-native Fourier/Biot-Savart flux kernel path.

Validates that the Fourier-basis forward and value_and_grad path produces
correct results. Basis-level tests use the public CPU CurveXYZFourier API as
the oracle so the JAX path is not compared only against another JAX formula.

Tests:
1. Fourier basis gamma matches CurveXYZFourier.gamma().
2. Fourier basis gammadash matches CurveXYZFourier.gammadash().
3. The fixed-surface flux kernel matches an independent NumPy Biot-Savart oracle.
4. The fixed-surface flux kernel gradient matches centred finite differences.
5. Gradient accumulation works for shared-DOF (symmetry) coils.
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.objectives.integral_bdotn_jax import integral_BdotN

_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
biot_savart_B = _bs.biot_savart_B


def _build_fourier_basis(quadpoints_jax, order):
    """Inlined from fluxobjective_jax.py (avoids relative import chain)."""
    k = 2 * order + 1
    npts = quadpoints_jax.shape[0]
    basis = jnp.zeros((npts, k))
    dbasis = jnp.zeros((npts, k))
    basis = basis.at[:, 0].set(1.0)
    for j in range(1, order + 1):
        arg = 2.0 * jnp.pi * j * quadpoints_jax
        s = jnp.sin(arg)
        c = jnp.cos(arg)
        basis = basis.at[:, 2 * j - 1].set(s)
        basis = basis.at[:, 2 * j].set(c)
        dbasis = dbasis.at[:, 2 * j - 1].set(2.0 * jnp.pi * j * c)
        dbasis = dbasis.at[:, 2 * j].set(-2.0 * jnp.pi * j * s)
    return basis, dbasis


def _central_difference_gradient(objective, flat_dofs, eps):
    grad_fd = np.zeros(len(flat_dofs))
    for i in range(len(flat_dofs)):
        fd_p = flat_dofs.at[i].add(eps)
        fd_m = flat_dofs.at[i].add(-eps)
        grad_fd[i] = (float(objective(fd_p)) - float(objective(fd_m))) / (2 * eps)
    return grad_fd


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _curvexyzfourier_cpu_geometry(dofs, quadpoints, order):
    """CPU CurveXYZFourier geometry oracle from the public SIMSOPT curve API."""
    curve = CurveXYZFourier(np.asarray(quadpoints, dtype=np.float64), order)
    curve.set_dofs(np.asarray(dofs, dtype=np.float64))
    return (
        curve.gamma(),
        curve.gammadash(),
        curve.gammadashdash(),
        curve.gammadashdashdash(),
    )


def _biot_savart_B_numpy(points, gammas, gammadashs, currents):
    points_np = np.asarray(points, dtype=np.float64)
    gammas_np = np.asarray(gammas, dtype=np.float64)
    gammadashs_np = np.asarray(gammadashs, dtype=np.float64)
    currents_np = np.asarray(currents, dtype=np.float64)
    diff = gammas_np[None, :, :, :] - points_np[:, None, None, :]
    radius_cubed = np.sum(diff * diff, axis=-1) ** 1.5
    integrand = np.cross(diff, gammadashs_np[None, :, :, :]) / radius_cubed[..., None]
    coil_integrals = np.mean(integrand, axis=2)
    return 1.0e-7 * np.einsum("c,pcj->pj", currents_np, coil_integrals)


# -----------------------------------------------------------------------
# Test 1: Fourier basis matches reference
# -----------------------------------------------------------------------


class TestFourierBasis:
    """Validate _build_fourier_basis against the CPU CurveXYZFourier API."""

    @pytest.mark.parametrize("order", [1, 3, 6])
    def test_gamma_parity(self, order):
        npts = 64
        quadpoints = jnp.linspace(0, 1, npts, endpoint=False)
        basis, _ = _build_fourier_basis(quadpoints, order)

        rng = np.random.RandomState(7)
        dofs = jnp.array(rng.randn(3 * (2 * order + 1)))
        k = 2 * order + 1

        gamma_basis = basis @ dofs.reshape(3, k).T
        gamma_ref = _curvexyzfourier_cpu_geometry(dofs, quadpoints, order)[0]

        np.testing.assert_allclose(
            np.array(gamma_basis), np.array(gamma_ref), atol=1e-13
        )

    @pytest.mark.parametrize("order", [1, 3, 6])
    def test_gammadash_parity(self, order):
        """dbasis @ coeffs.T matches CurveXYZFourier.gammadash()."""
        npts = 64
        quadpoints = jnp.linspace(0, 1, npts, endpoint=False)
        basis, dbasis = _build_fourier_basis(quadpoints, order)

        rng = np.random.RandomState(8)
        dofs = jnp.array(rng.randn(3 * (2 * order + 1)))
        k = 2 * order + 1
        coeffs = dofs.reshape(3, k)

        gd_basis = dbasis @ coeffs.T
        gd_ref = _curvexyzfourier_cpu_geometry(dofs, quadpoints, order)[1]

        np.testing.assert_allclose(np.array(gd_basis), np.array(gd_ref), atol=1e-12)

    def test_gammadash_finite_difference(self):
        """dbasis @ coeffs.T matches centred finite differences."""
        order = 3
        npts = 64
        quadpoints = jnp.linspace(0, 1, npts, endpoint=False)
        _, dbasis = _build_fourier_basis(quadpoints, order)

        rng = np.random.RandomState(9)
        dofs = jnp.array(rng.randn(3 * (2 * order + 1)))
        k = 2 * order + 1
        coeffs = dofs.reshape(3, k)

        gd_basis = np.array(dbasis @ coeffs.T)

        eps = 1e-7
        basis_p, _ = _build_fourier_basis(quadpoints + eps, order)
        basis_m, _ = _build_fourier_basis(quadpoints - eps, order)
        gd_fd = np.array((basis_p @ coeffs.T - basis_m @ coeffs.T) / (2 * eps))

        np.testing.assert_allclose(gd_basis, gd_fd, rtol=1e-5, atol=1e-10)

    @pytest.mark.parametrize("order", [1, 3, 6])
    def test_geometry_parity(self, order):
        from simsopt.geo.curvexyzfourier import jaxfouriercurve_geometry_pure

        quadpoints = jnp.array([0.0, 0.13, 0.37, 0.61, 0.92], dtype=jnp.float64)
        rng = np.random.RandomState(12 + order)
        dofs = jnp.array(rng.randn(3 * (2 * order + 1)))

        actual = tuple(
            np.asarray(part)
            for part in jaxfouriercurve_geometry_pure(dofs, quadpoints, order)
        )
        expected = tuple(
            np.asarray(part)
            for part in _curvexyzfourier_cpu_geometry(dofs, quadpoints, order)
        )

        for actual_part, expected_part in zip(actual, expected):
            np.testing.assert_allclose(actual_part, expected_part, atol=1e-13)


# -----------------------------------------------------------------------
# Test 2: End-to-end forward value
# -----------------------------------------------------------------------


class TestEndToEndForward:
    """Validate the composed DOFs → gamma → B → integral pipeline."""

    def test_single_coil_matches_numpy_biot_savart_oracle(self):
        """Single coil: basis-based forward matches an independent NumPy oracle."""
        order = 3
        nquad = 128
        quadpoints = jnp.linspace(0, 1, nquad, endpoint=False)
        basis, dbasis = _build_fourier_basis(quadpoints, order)
        k = 2 * order + 1

        rng = np.random.RandomState(10)
        dofs = jnp.array(rng.randn(3 * k) * 0.1)
        dofs = dofs.at[0].set(1.0)  # major radius
        current = 1e5

        coeffs = dofs.reshape(3, k)
        gamma = basis @ coeffs.T  # (nquad, 3)
        gammadash = dbasis @ coeffs.T  # (nquad, 3)

        # Simple surface: a few points near the coil
        nphi, ntheta = 4, 4
        nsurf = nphi * ntheta
        surf_points = jnp.array(rng.randn(nsurf, 3) * 0.1)
        surf_points = surf_points.at[:, 0].add(1.0)

        B = biot_savart_B(
            surf_points,
            gamma[None, :, :],
            gammadash[None, :, :],
            jnp.array([current]),
        )
        B_expected = _biot_savart_B_numpy(
            surf_points,
            gamma[None, :, :],
            gammadash[None, :, :],
            np.asarray([current], dtype=np.float64),
        )

        normal = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1)
        normal = normal.at[..., 2].add(1.0)
        target = jnp.zeros((nphi, ntheta))
        Bcoil = B.reshape((nphi, ntheta, 3))
        expected_Bcoil = jnp.asarray(B_expected).reshape((nphi, ntheta, 3))
        J_actual = float(integral_BdotN(Bcoil, target, normal, "quadratic flux"))
        J_expected = float(
            integral_BdotN(expected_Bcoil, target, normal, "quadratic flux")
        )

        np.testing.assert_allclose(
            np.asarray(B), B_expected, rtol=1.0e-12, atol=1.0e-16
        )
        np.testing.assert_allclose(J_actual, J_expected, rtol=1.0e-12, atol=1.0e-16)


# -----------------------------------------------------------------------
# Test 3: Gradient via value_and_grad matches finite differences
# -----------------------------------------------------------------------


class TestGradientFiniteDifference:
    """Validate that value_and_grad through the full pipeline is correct."""

    def test_gradient_single_coil(self):
        """Gradient w.r.t. curve DOFs matches FD for a single coil."""
        order = 2
        nquad = 64
        quadpoints = jnp.linspace(0, 1, nquad, endpoint=False)
        basis, dbasis = _build_fourier_basis(quadpoints, order)
        k = 2 * order + 1

        rng = np.random.RandomState(11)
        dofs = jnp.array(rng.randn(3 * k) * 0.1)
        dofs = dofs.at[0].set(1.0)
        current_val = 1e5

        # Surface data
        nphi, ntheta = 4, 4
        nsurf = nphi * ntheta
        surf_points = jnp.array(rng.randn(nsurf, 3) * 0.1)
        surf_points = surf_points.at[:, 0].add(1.0)
        normal = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1)
        normal = normal.at[..., 2].add(1.0)
        target = jnp.zeros((nphi, ntheta))

        # flat_dofs = [curve_dofs, current]
        flat_dofs = jnp.concatenate([dofs, jnp.array([current_val])])

        def objective(fd):
            cd = fd[: 3 * k]
            curr = fd[3 * k]
            coeffs = cd.reshape(3, k)
            g = basis @ coeffs.T
            gd = dbasis @ coeffs.T
            B = biot_savart_B(surf_points, g[None], gd[None], jnp.array([curr]))
            Bcoil = B.reshape((nphi, ntheta, 3))
            return integral_BdotN(Bcoil, target, normal, "quadratic flux")

        _, grad = jax.value_and_grad(objective)(flat_dofs)

        # Finite differences
        eps = 3e-7
        grad_fd = _central_difference_gradient(objective, flat_dofs, eps)

        np.testing.assert_allclose(np.array(grad), grad_fd, rtol=1e-5, atol=1e-10)

    def test_gradient_with_rotation(self):
        """Gradient through rotation matrix is correct."""
        order = 2
        nquad = 64
        quadpoints = jnp.linspace(0, 1, nquad, endpoint=False)
        basis, dbasis = _build_fourier_basis(quadpoints, order)
        k = 2 * order + 1

        rng = np.random.RandomState(12)
        dofs = jnp.array(rng.randn(3 * k) * 0.1)
        dofs = dofs.at[0].set(1.0)
        current_val = 1e5

        # Rotation matrix (60 degrees about z)
        phi = np.pi / 3
        rotmat = jnp.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        ).T

        nphi, ntheta = 4, 4
        nsurf = nphi * ntheta
        surf_points = jnp.array(rng.randn(nsurf, 3) * 0.1)
        surf_points = surf_points.at[:, 0].add(1.0)
        normal = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1)
        normal = normal.at[..., 2].add(1.0)
        target = jnp.zeros((nphi, ntheta))

        flat_dofs = jnp.concatenate([dofs, jnp.array([current_val])])

        def objective(fd):
            cd = fd[: 3 * k]
            curr = fd[3 * k]
            coeffs = cd.reshape(3, k)
            g = (basis @ coeffs.T) @ rotmat
            gd = (dbasis @ coeffs.T) @ rotmat
            B = biot_savart_B(surf_points, g[None], gd[None], jnp.array([curr]))
            Bcoil = B.reshape((nphi, ntheta, 3))
            return integral_BdotN(Bcoil, target, normal, "quadratic flux")

        _, grad = jax.value_and_grad(objective)(flat_dofs)

        eps = 1e-5
        grad_fd = _central_difference_gradient(objective, flat_dofs, eps)

        np.testing.assert_allclose(np.array(grad), grad_fd, rtol=1e-6, atol=1e-10)

    def test_shared_dofs_accumulate(self):
        """Two coils sharing DOFs: gradient correctly sums contributions."""
        order = 1
        nquad = 32
        quadpoints = jnp.linspace(0, 1, nquad, endpoint=False)
        basis, dbasis = _build_fourier_basis(quadpoints, order)
        k = 2 * order + 1

        rng = np.random.RandomState(13)
        dofs = jnp.array(rng.randn(3 * k) * 0.1)
        dofs = dofs.at[0].set(1.0)
        current_val = 1e5

        # Rotation matrix (180 degrees — stellarator half-period symmetry)
        phi2 = jnp.pi
        rotmat2 = jnp.array(
            [
                [jnp.cos(phi2), -jnp.sin(phi2), 0],
                [jnp.sin(phi2), jnp.cos(phi2), 0],
                [0, 0, 1],
            ]
        ).T

        nphi, ntheta = 4, 4
        nsurf = nphi * ntheta
        surf_points = jnp.array(rng.randn(nsurf, 3) * 0.05)
        surf_points = surf_points.at[:, 0].add(1.0)
        normal = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1)
        normal = normal.at[..., 2].add(1.0)
        target = jnp.zeros((nphi, ntheta))

        # flat_dofs has ONE set of curve DOFs + ONE current (shared by both coils)
        flat_dofs = jnp.concatenate([dofs, jnp.array([current_val])])

        def objective(fd):
            cd = fd[: 3 * k]
            curr = fd[3 * k]
            coeffs = cd.reshape(3, k)

            g1 = basis @ coeffs.T
            gd1 = dbasis @ coeffs.T
            g2 = g1 @ rotmat2
            gd2 = gd1 @ rotmat2

            gammas = jnp.stack([g1, g2])
            gammadashs = jnp.stack([gd1, gd2])
            currents = jnp.array([curr, -curr])  # flipped current for symmetry

            B = biot_savart_B(surf_points, gammas, gammadashs, currents)
            Bcoil = B.reshape((nphi, ntheta, 3))
            return integral_BdotN(Bcoil, target, normal, "quadratic flux")

        _, grad = jax.value_and_grad(objective)(flat_dofs)

        eps = 3e-7
        grad_fd = _central_difference_gradient(objective, flat_dofs, eps)

        np.testing.assert_allclose(np.array(grad), grad_fd, rtol=2e-6, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
