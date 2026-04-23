"""
Parity tests for the JAX SurfaceXYZTensorFourier evaluation.

Validates against:
1. Known simple torus geometry (R=1, r=0.1).
2. Finite-difference derivatives.
3. Normal vector consistency.
4. C++ reference (when simsoptpp is available).
"""

import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"

def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_sf = _load("surface_fourier_jax", "geo/surface_fourier_jax.py")
build_theta_basis = _sf.build_theta_basis
build_phi_basis = _sf.build_phi_basis
surface_gamma = _sf.surface_gamma
surface_gammadash1 = _sf.surface_gammadash1
surface_gammadash2 = _sf.surface_gammadash2
surface_normal = _sf.surface_normal
dgamma_by_dcoeff = _sf.dgamma_by_dcoeff
dgammadash1_by_dcoeff = _sf.dgammadash1_by_dcoeff
dgammadash2_by_dcoeff = _sf.dgammadash2_by_dcoeff
surface_xyzfourier_gamma_from_dofs = _sf.surface_xyzfourier_gamma_from_dofs
surface_xyzfourier_gammadash1_from_dofs = _sf.surface_xyzfourier_gammadash1_from_dofs
surface_xyzfourier_gammadash2_from_dofs = _sf.surface_xyzfourier_gammadash2_from_dofs
_DERIVATIVE_HEAVY_TOLS = parity_ladder_tolerances("derivative-heavy")


def _make_simple_torus_coeffs(R=1.0, r=0.1, mpol=1, ntor=0, nfp=1):
    """Create coefficient matrices for a circular-cross-section torus.

    The torus is: x = (R + r cos θ) cos φ,  y = (R + r cos θ) sin φ,  z = r sin θ.

    In XYZTensorFourier form:
      x̂(θ,φ) = R + r·cos(2πθ)   →  xc[0,0] = R, xc[1,0] = r
      ŷ(θ,φ) = 0
      z(θ,φ) = r·sin(2πθ)       →  zc[mpol+1, 0] = r  (sin(θ) basis index)
    """
    xc = np.zeros((2 * mpol + 1, 2 * ntor + 1))
    yc = np.zeros((2 * mpol + 1, 2 * ntor + 1))
    zc = np.zeros((2 * mpol + 1, 2 * ntor + 1))

    xc[0, 0] = R        # constant = major radius
    xc[1, 0] = r        # cos(θ) = minor radius modulation
    zc[mpol + 1, 0] = r  # sin(θ) = z variation

    return jnp.array(xc), jnp.array(yc), jnp.array(zc)


class TestSurfaceFourierJaxSimpleTorus:
    """Test against known circular torus geometry."""

    R, r = 1.0, 0.1
    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 20, 20

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.xc, self.yc, self.zc = _make_simple_torus_coeffs(
            R=self.R, r=self.r, mpol=self.mpol, ntor=self.ntor, nfp=self.nfp
        )

    def test_gamma_torus(self):
        """Gamma should match the analytical torus parametrization."""
        gamma = surface_gamma(
            self.phis, self.thetas, self.xc, self.yc, self.zc,
            self.mpol, self.ntor, self.nfp,
        )

        phi_rad = 2 * np.pi * np.array(self.phis)
        theta_rad = 2 * np.pi * np.array(self.thetas)
        phi_2d, theta_2d = np.meshgrid(phi_rad, theta_rad, indexing="ij")

        x_ref = (self.R + self.r * np.cos(theta_2d)) * np.cos(phi_2d)
        y_ref = (self.R + self.r * np.cos(theta_2d)) * np.sin(phi_2d)
        z_ref = self.r * np.sin(theta_2d)

        np.testing.assert_allclose(np.array(gamma[..., 0]), x_ref, atol=1e-14)
        np.testing.assert_allclose(np.array(gamma[..., 1]), y_ref, atol=1e-14)
        np.testing.assert_allclose(np.array(gamma[..., 2]), z_ref, atol=1e-14)

    def test_gammadash1_finite_difference(self):
        """gammadash1 should match finite differences in φ."""
        eps = 1e-7

        args = (self.thetas, self.xc, self.yc, self.zc, self.mpol, self.ntor, self.nfp)
        gd1 = np.array(surface_gammadash1(self.phis, *args))

        gd1_fd = np.zeros_like(gd1)
        for i in range(self.nphi):
            phi_p = self.phis.at[i].add(eps)
            phi_m = self.phis.at[i].add(-eps)
            g_p = np.array(surface_gamma(phi_p, *args))
            g_m = np.array(surface_gamma(phi_m, *args))
            gd1_fd[i] = (g_p[i] - g_m[i]) / (2 * eps)

        np.testing.assert_allclose(gd1, gd1_fd, rtol=1e-5, atol=1e-10)

    def test_gammadash2_finite_difference(self):
        """gammadash2 should match finite differences in θ."""
        eps = 1e-7

        args = (self.xc, self.yc, self.zc, self.mpol, self.ntor, self.nfp)
        gd2 = np.array(surface_gammadash2(self.phis, self.thetas, *args))

        gd2_fd = np.zeros_like(gd2)
        for j in range(self.ntheta):
            theta_p = self.thetas.at[j].add(eps)
            theta_m = self.thetas.at[j].add(-eps)
            g_p = np.array(surface_gamma(self.phis, theta_p, *args))
            g_m = np.array(surface_gamma(self.phis, theta_m, *args))
            gd2_fd[:, j] = (g_p[:, j] - g_m[:, j]) / (2 * eps)

        np.testing.assert_allclose(gd2, gd2_fd, rtol=1e-5, atol=1e-10)

    def test_normal_orthogonality(self):
        """Normal should be orthogonal to both tangent vectors."""
        args = (self.phis, self.thetas, self.xc, self.yc, self.zc,
                self.mpol, self.ntor, self.nfp)
        gd1 = surface_gammadash1(*args)
        gd2 = surface_gammadash2(*args)
        n = surface_normal(*args)

        dot1 = jnp.sum(n * gd1, axis=-1)
        dot2 = jnp.sum(n * gd2, axis=-1)

        np.testing.assert_allclose(np.array(dot1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.array(dot2), 0.0, atol=1e-12)


class TestSurfaceFourierJaxHigherOrder:
    """Test with higher mpol/ntor to catch basis indexing bugs."""

    def test_nontrivial_modes(self):
        """Verify round-trip: set a single mode, recover it from gamma."""
        mpol, ntor, nfp = 3, 2, 2
        nphi, ntheta = 32, 32
        phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        # Set only the cos(2θ)·cos(nfp·φ) mode for x̂
        xc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
        yc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
        zc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))

        # Major radius
        xc = xc.at[0, 0].set(1.0)
        # cos(2·2πθ) · cos(1·nfp·2πφ) mode
        xc = xc.at[2, 1].set(0.05)

        gamma = surface_gamma(phis, thetas, xc, yc, zc, mpol, ntor, nfp)

        # Verify the shape perturbation
        phi_rad = 2 * np.pi * np.array(phis)
        theta_rad = 2 * np.pi * np.array(thetas)
        phi_2d, theta_2d = np.meshgrid(phi_rad, theta_rad, indexing="ij")

        # x̂ = 1.0 + 0.05 · cos(2·2πθ) · cos(nfp·2πφ)
        xhat_expected = 1.0 + 0.05 * np.cos(2 * theta_2d) * np.cos(nfp * phi_2d)
        x_expected = xhat_expected * np.cos(phi_2d)

        np.testing.assert_allclose(np.array(gamma[..., 0]), x_expected, atol=1e-13)

    def test_basis_shape(self):
        """Basis matrices should have correct shapes."""
        mpol, ntor, nfp = 4, 3, 2
        ntheta, nphi = 15, 12

        thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)
        phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)

        W, dW = build_theta_basis(thetas, mpol)
        V, dV = build_phi_basis(phis, ntor, nfp)

        assert W.shape == (ntheta, 2 * mpol + 1)
        assert dW.shape == (ntheta, 2 * mpol + 1)
        assert V.shape == (nphi, 2 * ntor + 1)
        assert dV.shape == (nphi, 2 * ntor + 1)


class TestSurfaceFourierJaxCppParity:
    """Compare against C++ SurfaceXYZTensorFourier (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")
        pytest.importorskip("simsopt")

    def test_gamma_parity(self):
        from simsopt.geo import SurfaceXYZTensorFourier

        mpol, ntor, nfp = 2, 2, 2
        nphi, ntheta = 15, 15
        phis_np = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        thetas_np = np.linspace(0, 1.0, ntheta, endpoint=False)

        s = SurfaceXYZTensorFourier(
            mpol=mpol, ntor=ntor, nfp=nfp, stellsym=True,
            quadpoints_phi=phis_np, quadpoints_theta=thetas_np,
        )
        gamma_cpp = s.gamma()

        xc = jnp.array(np.array(s.xcs))
        yc = jnp.array(np.array(s.ycs))
        zc = jnp.array(np.array(s.zcs))

        gamma_jax = surface_gamma(
            jnp.array(phis_np), jnp.array(thetas_np),
            xc, yc, zc, mpol, ntor, nfp,
        )

        np.testing.assert_allclose(np.array(gamma_jax), gamma_cpp, atol=1e-13)

    @pytest.mark.parametrize(
        ("jax_fn", "cpp_method"),
        [
            (dgamma_by_dcoeff, "dgamma_by_dcoeff"),
            (dgammadash1_by_dcoeff, "dgammadash1_by_dcoeff"),
            (dgammadash2_by_dcoeff, "dgammadash2_by_dcoeff"),
        ],
    )
    def test_coefficient_derivatives_match_cpp(self, jax_fn, cpp_method):
        from simsopt.geo import SurfaceXYZTensorFourier

        mpol, ntor, nfp = 2, 2, 2
        surface = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=False,
            quadpoints_phi=np.linspace(0, 1.0 / nfp, 7, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, 6, endpoint=False),
        )
        rng = np.random.default_rng(7)
        dofs = surface.get_dofs().copy()
        dofs[:] = rng.normal(scale=0.1, size=dofs.shape)
        surface.set_dofs(dofs)

        derivative_jax = jax_fn(
            jnp.asarray(dofs),
            jnp.asarray(surface.quadpoints_phi),
            jnp.asarray(surface.quadpoints_theta),
            mpol,
            ntor,
            nfp,
            False,
        )
        derivative_cpp = getattr(surface, cpp_method)()

        np.testing.assert_allclose(
            np.asarray(derivative_jax),
            derivative_cpp,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )


class TestSurfaceXYZFourierJaxCppParity:
    """Compare the pure-JAX SurfaceXYZFourier path against the CPU object."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")
        pytest.importorskip("simsopt")

    @pytest.mark.parametrize("stellsym", [True, False])
    def test_geometry_and_tangents_match_cpp(self, stellsym):
        from simsopt.geo import SurfaceXYZFourier

        rng = np.random.default_rng(0 if stellsym else 1)
        surface = SurfaceXYZFourier(
            mpol=2,
            ntor=2,
            nfp=3,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0, 1.0 / 3.0, 7, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, 6, endpoint=False),
        )
        dofs = surface.get_dofs().copy()
        dofs[:] = rng.normal(size=dofs.shape)
        surface.set_dofs(dofs)

        args = (
            jnp.asarray(dofs),
            jnp.asarray(surface.quadpoints_phi),
            jnp.asarray(surface.quadpoints_theta),
            surface.mpol,
            surface.ntor,
            surface.nfp,
            surface.stellsym,
        )

        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_gamma_from_dofs(*args)),
            surface.gamma(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_gammadash1_from_dofs(*args)),
            surface.gammadash1(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_gammadash2_from_dofs(*args)),
            surface.gammadash2(),
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
