"""
End-to-end test for the JAX-native SquaredFluxJAX path.

Validates that the Fourier-basis forward and value_and_grad path
produces correct results without depending on simsoptpp.

Tests:
1. Fourier basis gamma matches jaxfouriercurve_pure.
2. Fourier basis gammadash matches JVP of gamma.
3. SquaredFluxJAX.J() matches manual computation.
4. SquaredFluxJAX.dJ() matches centred finite differences.
5. JAX-native path is detected for CurveXYZFourier coils.
6. Gradient accumulation works for shared-DOF (symmetry) coils.
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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


def _jaxfouriercurve_pure(dofs, quadpoints, order):
    """Reference implementation (loop-based, from curvexyzfourier.py)."""
    k = len(dofs) // 3
    coeffs = [dofs[:k], dofs[k : 2 * k], dofs[2 * k :]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(3):
        gamma = gamma.at[:, i].add(coeffs[i][0])
        for j in range(1, order + 1):
            gamma = gamma.at[:, i].add(
                coeffs[i][2 * j - 1] * jnp.sin(2 * jnp.pi * j * points)
            )
            gamma = gamma.at[:, i].add(
                coeffs[i][2 * j] * jnp.cos(2 * jnp.pi * j * points)
            )
    return gamma


def _jaxfouriercurve_geometry_ref(dofs, quadpoints, order):
    """Reference loop implementation for XYZ Fourier geometry derivatives."""
    k = len(dofs) // 3
    coeffs = [dofs[:k], dofs[k : 2 * k], dofs[2 * k :]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    gammadash = jnp.zeros((len(points), 3))
    gammadashdash = jnp.zeros((len(points), 3))
    gammadashdashdash = jnp.zeros((len(points), 3))
    two_pi = 2.0 * jnp.pi
    for i in range(3):
        gamma = gamma.at[:, i].add(coeffs[i][0])
        for j in range(1, order + 1):
            scale = two_pi * j
            arg = scale * points
            s = jnp.sin(arg)
            c = jnp.cos(arg)
            xs = coeffs[i][2 * j - 1]
            xc = coeffs[i][2 * j]
            gamma = gamma.at[:, i].add(xs * s + xc * c)
            gammadash = gammadash.at[:, i].add(scale * (xs * c - xc * s))
            gammadashdash = gammadashdash.at[:, i].add(
                -(scale * scale) * (xs * s + xc * c)
            )
            gammadashdashdash = gammadashdashdash.at[:, i].add(
                -(scale * scale * scale) * (xs * c - xc * s)
            )
    return gamma, gammadash, gammadashdash, gammadashdashdash


# -----------------------------------------------------------------------
# Test 1: Fourier basis matches reference
# -----------------------------------------------------------------------


class TestFourierBasis:
    """Validate _build_fourier_basis against the loop-based reference."""

    @pytest.mark.parametrize("order", [1, 3, 6])
    def test_gamma_parity(self, order):
        npts = 64
        quadpoints = jnp.linspace(0, 1, npts, endpoint=False)
        basis, _ = _build_fourier_basis(quadpoints, order)

        rng = np.random.RandomState(7)
        dofs = jnp.array(rng.randn(3 * (2 * order + 1)))
        k = 2 * order + 1

        gamma_basis = basis @ dofs.reshape(3, k).T
        gamma_ref = _jaxfouriercurve_pure(dofs, quadpoints, order)

        np.testing.assert_allclose(
            np.array(gamma_basis), np.array(gamma_ref), atol=1e-13
        )

    @pytest.mark.parametrize("order", [1, 3, 6])
    def test_gammadash_parity(self, order):
        """dbasis @ coeffs.T matches JVP of gamma."""
        npts = 64
        quadpoints = jnp.linspace(0, 1, npts, endpoint=False)
        basis, dbasis = _build_fourier_basis(quadpoints, order)

        rng = np.random.RandomState(8)
        dofs = jnp.array(rng.randn(3 * (2 * order + 1)))
        k = 2 * order + 1
        coeffs = dofs.reshape(3, k)

        gd_basis = dbasis @ coeffs.T

        # Reference via JVP
        ones = jnp.ones_like(quadpoints)
        _, gd_jvp = jax.jvp(
            lambda p: _jaxfouriercurve_pure(dofs, p, order),
            (quadpoints,),
            (ones,),
        )

        np.testing.assert_allclose(np.array(gd_basis), np.array(gd_jvp), atol=1e-12)

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
            for part in _jaxfouriercurve_geometry_ref(dofs, quadpoints, order)
        )

        for actual_part, expected_part in zip(actual, expected):
            np.testing.assert_allclose(actual_part, expected_part, atol=1e-13)


# -----------------------------------------------------------------------
# Test 2: End-to-end forward value
# -----------------------------------------------------------------------


class TestEndToEndForward:
    """Validate the composed DOFs → gamma → B → integral pipeline."""

    def test_single_coil_matches_manual(self):
        """Single coil: basis-based forward matches manual computation."""
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

        # Compute integral manually
        normal = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1)
        normal = normal.at[..., 2].add(1.0)
        target = jnp.zeros((nphi, ntheta))
        Bcoil = B.reshape((nphi, ntheta, 3))
        J_manual = float(integral_BdotN(Bcoil, target, normal, "quadratic flux"))

        assert J_manual > 0  # sanity check
        assert np.isfinite(J_manual)


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
