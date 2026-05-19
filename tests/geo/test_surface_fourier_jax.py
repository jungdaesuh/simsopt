"""
Parity tests for the JAX SurfaceXYZTensorFourier evaluation.

Validates against:
1. Known simple torus geometry (R=1, r=0.1).
2. Finite-difference derivatives.
3. Normal vector consistency.
4. C++ reference (when simsoptpp is available).
"""

import copy
import importlib.util
from pathlib import Path

import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core import (
    surface_xyz_fourier_dfirst_fund_form_from_dofs,
    surface_xyz_fourier_dsecond_fund_form_from_dofs,
    surface_xyz_fourier_dsurface_curvatures_from_dofs,
    surface_xyz_fourier_first_fund_form_from_spec,
    surface_xyz_fourier_second_fund_form_from_spec,
    surface_xyz_fourier_surface_curvatures_from_spec,
    surface_xyz_tensor_fourier_dfirst_fund_form_from_dofs,
    surface_xyz_tensor_fourier_dsecond_fund_form_from_dofs,
    surface_xyz_tensor_fourier_dsurface_curvatures_from_dofs,
    surface_xyz_tensor_fourier_first_fund_form_from_spec,
    surface_xyz_tensor_fourier_second_fund_form_from_spec,
    surface_xyz_tensor_fourier_surface_curvatures_from_spec,
)

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
surface_gamma_lin_from_dofs = _sf.surface_gamma_lin_from_dofs
surface_gammadash1_lin_from_dofs = _sf.surface_gammadash1_lin_from_dofs
surface_gammadash2_lin_from_dofs = _sf.surface_gammadash2_lin_from_dofs
surface_gammadash1dash1_from_dofs = _sf.surface_gammadash1dash1_from_dofs
surface_gammadash1dash2_from_dofs = _sf.surface_gammadash1dash2_from_dofs
surface_gammadash2dash2_from_dofs = _sf.surface_gammadash2dash2_from_dofs
surface_normal = _sf.surface_normal
surface_area = _sf.surface_area
surface_volume = _sf.surface_volume
surface_area_from_dofs = _sf.surface_area_from_dofs
surface_volume_from_dofs = _sf.surface_volume_from_dofs
dgamma_by_dcoeff = _sf.dgamma_by_dcoeff
dgammadash1_by_dcoeff = _sf.dgammadash1_by_dcoeff
dgammadash2_by_dcoeff = _sf.dgammadash2_by_dcoeff
dgammadash1dash1_by_dcoeff = _sf.dgammadash1dash1_by_dcoeff
dgammadash1dash2_by_dcoeff = _sf.dgammadash1dash2_by_dcoeff
dgammadash2dash2_by_dcoeff = _sf.dgammadash2dash2_by_dcoeff
dnormal_by_dcoeff = _sf.dnormal_by_dcoeff
d2normal_by_dcoeffdcoeff = _sf.d2normal_by_dcoeffdcoeff
dunitnormal_by_dcoeff = _sf.dunitnormal_by_dcoeff
darea_by_dcoeff = _sf.darea_by_dcoeff
d2area_by_dcoeffdcoeff = _sf.d2area_by_dcoeffdcoeff
dvolume_by_dcoeff = _sf.dvolume_by_dcoeff
d2volume_by_dcoeffdcoeff = _sf.d2volume_by_dcoeffdcoeff
surface_xyzfourier_gamma_from_dofs = _sf.surface_xyzfourier_gamma_from_dofs
surface_xyzfourier_gamma_lin_from_dofs = _sf.surface_xyzfourier_gamma_lin_from_dofs
surface_xyzfourier_gammadash1_from_dofs = _sf.surface_xyzfourier_gammadash1_from_dofs
surface_xyzfourier_gammadash1_lin_from_dofs = (
    _sf.surface_xyzfourier_gammadash1_lin_from_dofs
)
surface_xyzfourier_gammadash2_from_dofs = _sf.surface_xyzfourier_gammadash2_from_dofs
surface_xyzfourier_gammadash2_lin_from_dofs = (
    _sf.surface_xyzfourier_gammadash2_lin_from_dofs
)
surface_xyzfourier_gammadash1dash1_from_dofs = (
    _sf.surface_xyzfourier_gammadash1dash1_from_dofs
)
surface_xyzfourier_gammadash1dash2_from_dofs = (
    _sf.surface_xyzfourier_gammadash1dash2_from_dofs
)
surface_xyzfourier_gammadash2dash2_from_dofs = (
    _sf.surface_xyzfourier_gammadash2dash2_from_dofs
)
surface_xyzfourier_normal_from_dofs = _sf.surface_xyzfourier_normal_from_dofs
surface_xyzfourier_unitnormal_from_dofs = _sf.surface_xyzfourier_unitnormal_from_dofs
surface_xyzfourier_area_from_dofs = _sf.surface_xyzfourier_area_from_dofs
surface_xyzfourier_volume_from_dofs = _sf.surface_xyzfourier_volume_from_dofs
surface_xyzfourier_dgamma_by_dcoeff = _sf.surface_xyzfourier_dgamma_by_dcoeff
surface_xyzfourier_dgammadash1_by_dcoeff = (
    _sf.surface_xyzfourier_dgammadash1_by_dcoeff
)
surface_xyzfourier_dgammadash2_by_dcoeff = (
    _sf.surface_xyzfourier_dgammadash2_by_dcoeff
)
surface_xyzfourier_dnormal_by_dcoeff = _sf.surface_xyzfourier_dnormal_by_dcoeff
surface_xyzfourier_d2normal_by_dcoeffdcoeff = (
    _sf.surface_xyzfourier_d2normal_by_dcoeffdcoeff
)
surface_xyzfourier_dunitnormal_by_dcoeff = (
    _sf.surface_xyzfourier_dunitnormal_by_dcoeff
)
surface_xyzfourier_darea_by_dcoeff = _sf.surface_xyzfourier_darea_by_dcoeff
surface_xyzfourier_d2area_by_dcoeffdcoeff = (
    _sf.surface_xyzfourier_d2area_by_dcoeffdcoeff
)
surface_xyzfourier_dvolume_by_dcoeff = _sf.surface_xyzfourier_dvolume_by_dcoeff
surface_xyzfourier_d2volume_by_dcoeffdcoeff = (
    _sf.surface_xyzfourier_d2volume_by_dcoeffdcoeff
)
surface_xyzfourier_dgammadash1dash1_by_dcoeff = (
    _sf.surface_xyzfourier_dgammadash1dash1_by_dcoeff
)
surface_xyzfourier_dgammadash1dash2_by_dcoeff = (
    _sf.surface_xyzfourier_dgammadash1dash2_by_dcoeff
)
surface_xyzfourier_dgammadash2dash2_by_dcoeff = (
    _sf.surface_xyzfourier_dgammadash2dash2_by_dcoeff
)
stellsym_scatter_indices = _sf.stellsym_scatter_indices
_DERIVATIVE_HEAVY_TOLS = parity_ladder_tolerances("derivative-heavy")
_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct_kernel")


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

    def test_normal_matches_analytic_torus_geometry(self):
        """Normal should match the independent circular-torus oracle."""
        args = (self.phis, self.thetas, self.xc, self.yc, self.zc,
                self.mpol, self.ntor, self.nfp)
        gd1 = surface_gammadash1(*args)
        gd2 = surface_gammadash2(*args)
        n = surface_normal(*args)

        dot1 = jnp.sum(n * gd1, axis=-1)
        dot2 = jnp.sum(n * gd2, axis=-1)

        np.testing.assert_allclose(np.array(dot1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.array(dot2), 0.0, atol=1e-12)

        phi_rad = 2 * np.pi * np.array(self.phis)
        theta_rad = 2 * np.pi * np.array(self.thetas)
        phi_2d, theta_2d = np.meshgrid(phi_rad, theta_rad, indexing="ij")
        unit_normal = np.stack(
            (
                np.cos(theta_2d) * np.cos(phi_2d),
                np.cos(theta_2d) * np.sin(phi_2d),
                np.sin(theta_2d),
            ),
            axis=-1,
        )
        expected_magnitude = (
            (2.0 * np.pi) ** 2
            * self.r
            * (self.R + self.r * np.cos(theta_2d))
        )
        expected_normal = expected_magnitude[..., None] * unit_normal

        np.testing.assert_allclose(
            np.array(n),
            expected_normal,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(np.array(n), axis=-1),
            expected_magnitude,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_area_and_volume_match_analytic_torus(self):
        """Area and volume should match the closed-form circular torus oracle."""
        args = (self.phis, self.thetas, self.xc, self.yc, self.zc,
                self.mpol, self.ntor, self.nfp)
        gamma = surface_gamma(*args)
        normal = surface_normal(*args)

        expected_area = 4.0 * np.pi**2 * self.R * self.r
        expected_volume = 2.0 * np.pi**2 * self.R * self.r**2

        np.testing.assert_allclose(
            float(surface_area(normal)),
            expected_area,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        np.testing.assert_allclose(
            float(surface_volume(gamma, normal)),
            expected_volume,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_tiny_unitnormal_and_area_autodiff_are_finite(self):
        """Tiny nonzero normals should not underflow into NaN autodiff."""

        normal = jnp.asarray([[[1.0e-300, 0.0, 0.0]]], dtype=jnp.float64)
        unitnormal = _sf._unitnormal(normal)
        unitnormal_jac = jax.jacfwd(lambda n: _sf._unitnormal(n).reshape(-1))(normal)
        area_grad = jax.grad(surface_area)(normal)

        np.testing.assert_allclose(
            np.asarray(unitnormal),
            np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        assert np.isfinite(np.asarray(unitnormal_jac)).all()
        assert np.isfinite(np.asarray(area_grad)).all()


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
            (dgammadash1dash1_by_dcoeff, "dgammadash1dash1_by_dcoeff"),
            (dgammadash1dash2_by_dcoeff, "dgammadash1dash2_by_dcoeff"),
            (dgammadash2dash2_by_dcoeff, "dgammadash2dash2_by_dcoeff"),
            (dnormal_by_dcoeff, "dnormal_by_dcoeff"),
            (dunitnormal_by_dcoeff, "dunitnormal_by_dcoeff"),
        ],
    )
    @pytest.mark.parametrize("stellsym", [False, True])
    def test_coefficient_derivatives_match_cpp(self, jax_fn, cpp_method, stellsym):
        from simsopt.geo import SurfaceXYZTensorFourier

        mpol, ntor, nfp = 2, 2, 2
        surface = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
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
            stellsym,
            jnp.asarray(stellsym_scatter_indices(mpol, ntor)) if stellsym else None,
        )
        derivative_cpp = getattr(surface, cpp_method)()

        np.testing.assert_allclose(
            np.asarray(derivative_jax),
            derivative_cpp,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    @pytest.mark.parametrize(
        ("jax_fn", "cpp_method"),
        [
            (surface_gammadash1dash1_from_dofs, "gammadash1dash1"),
            (surface_gammadash1dash2_from_dofs, "gammadash1dash2"),
            (surface_gammadash2dash2_from_dofs, "gammadash2dash2"),
        ],
    )
    @pytest.mark.parametrize("stellsym", [False, True])
    def test_second_coordinate_derivatives_match_cpp(
        self,
        jax_fn,
        cpp_method,
        stellsym,
    ):
        from simsopt.geo import SurfaceXYZTensorFourier

        mpol, ntor, nfp = 3, 2, 4
        surface = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.013, (1.0 / nfp) - 0.005, 8),
            quadpoints_theta=np.linspace(0.019, 0.971, 7),
        )
        rng = np.random.default_rng(17 + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.05, size=dofs.shape)
        surface.set_dofs(dofs)

        derivative_jax = jax_fn(
            jnp.asarray(dofs),
            jnp.asarray(surface.quadpoints_phi),
            jnp.asarray(surface.quadpoints_theta),
            mpol,
            ntor,
            nfp,
            stellsym,
            jnp.asarray(stellsym_scatter_indices(mpol, ntor)) if stellsym else None,
        )

        np.testing.assert_allclose(
            np.asarray(derivative_jax),
            getattr(surface, cpp_method)(),
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
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_gammadash1dash1_from_dofs(*args)),
            surface.gammadash1dash1(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_gammadash1dash2_from_dofs(*args)),
            surface.gammadash1dash2(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_gammadash2dash2_from_dofs(*args)),
            surface.gammadash2dash2(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_normal_from_dofs(*args)),
            surface.normal(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(surface_xyzfourier_unitnormal_from_dofs(*args)),
            surface.unitnormal(),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        ("jax_fn", "cpp_method"),
        [
            (surface_xyzfourier_dgamma_by_dcoeff, "dgamma_by_dcoeff"),
            (surface_xyzfourier_dgammadash1_by_dcoeff, "dgammadash1_by_dcoeff"),
            (surface_xyzfourier_dgammadash2_by_dcoeff, "dgammadash2_by_dcoeff"),
            (surface_xyzfourier_dnormal_by_dcoeff, "dnormal_by_dcoeff"),
            (surface_xyzfourier_dunitnormal_by_dcoeff, "dunitnormal_by_dcoeff"),
            (
                surface_xyzfourier_dgammadash1dash1_by_dcoeff,
                "dgammadash1dash1_by_dcoeff",
            ),
            (
                surface_xyzfourier_dgammadash1dash2_by_dcoeff,
                "dgammadash1dash2_by_dcoeff",
            ),
            (
                surface_xyzfourier_dgammadash2dash2_by_dcoeff,
                "dgammadash2dash2_by_dcoeff",
            ),
        ],
    )
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_second_coordinate_derivative_dcoeff_match_cpp(
        self,
        jax_fn,
        cpp_method,
        stellsym,
    ):
        from simsopt.geo import SurfaceXYZFourier

        surface = SurfaceXYZFourier(
            mpol=2,
            ntor=2,
            nfp=5,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.007, (1.0 / 5.0) - 0.004, 6),
            quadpoints_theta=np.linspace(0.023, 0.977, 5),
        )
        rng = np.random.default_rng(41 + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.03, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        derivative_jax = jax_fn(
            jnp.asarray(dofs),
            jnp.asarray(surface.quadpoints_phi),
            jnp.asarray(surface.quadpoints_theta),
            surface.mpol,
            surface.ntor,
            surface.nfp,
            surface.stellsym,
            spec.scatter_indices,
            spec.coeff_template,
        )

        np.testing.assert_allclose(
            np.asarray(derivative_jax),
            getattr(surface, cpp_method)(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    @pytest.mark.parametrize(
        ("jax_fn", "cpp_method"),
        [
            (surface_xyzfourier_dgammadash1_by_dcoeff, "dgammadash1_by_dcoeff"),
            (surface_xyzfourier_dgammadash2_by_dcoeff, "dgammadash2_by_dcoeff"),
        ],
    )
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_tangent_derivative_columns_match_cpp(self, jax_fn, cpp_method, stellsym):
        """Exercise each SurfaceXYZFourier tangent-derivative DOF column."""
        from simsopt.geo import SurfaceXYZFourier

        nfp = 4
        surface = SurfaceXYZFourier(
            mpol=3,
            ntor=2,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.011, (1.0 / nfp) - 0.007, 7),
            quadpoints_theta=np.linspace(0.019, 0.981, 6),
        )
        rng = np.random.default_rng(151 + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.025, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        derivative_jax = np.asarray(
            jax_fn(
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices,
                spec.coeff_template,
            )
        )
        derivative_cpp = getattr(surface, cpp_method)()

        for column in range(dofs.size):
            np.testing.assert_allclose(
                derivative_jax[..., column],
                derivative_cpp[..., column],
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
            )

    @pytest.mark.parametrize("stellsym", [True, False])
    def test_normal_derivative_columns_match_cpp(self, stellsym):
        """Exercise SurfaceXYZFourier normal and normal-Hessian DOF columns."""
        from simsopt.geo import SurfaceXYZFourier

        nfp = 3
        surface = SurfaceXYZFourier(
            mpol=2,
            ntor=1,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.013, (1.0 / nfp) - 0.011, 5),
            quadpoints_theta=np.linspace(0.017, 0.983, 4),
        )
        rng = np.random.default_rng(171 + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.03, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()
        args = (
            jnp.asarray(dofs),
            jnp.asarray(surface.quadpoints_phi),
            jnp.asarray(surface.quadpoints_theta),
            surface.mpol,
            surface.ntor,
            surface.nfp,
            surface.stellsym,
            spec.scatter_indices,
            spec.coeff_template,
        )

        dnormal_jax = np.asarray(surface_xyzfourier_dnormal_by_dcoeff(*args))
        dnormal_cpp = surface.dnormal_by_dcoeff()
        d2normal_jax = np.asarray(surface_xyzfourier_d2normal_by_dcoeffdcoeff(*args))
        d2normal_cpp = surface.d2normal_by_dcoeffdcoeff()

        for column in range(dofs.size):
            np.testing.assert_allclose(
                dnormal_jax[..., column],
                dnormal_cpp[..., column],
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
            )
            for other_column in range(dofs.size):
                np.testing.assert_allclose(
                    d2normal_jax[..., column, other_column],
                    d2normal_cpp[..., column, other_column],
                    rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
                    atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
                )

    @pytest.mark.parametrize("stellsym", [True, False])
    def test_dnormal_by_dcoeff_vjp_matches_cpp(self, stellsym):
        """SurfaceXYZFourier normal VJP matches the CPU Surface VJP helper."""
        from simsopt.geo import SurfaceXYZFourier

        nfp = 3
        surface = SurfaceXYZFourier(
            mpol=2,
            ntor=1,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.021, (1.0 / nfp) - 0.013, 5),
            quadpoints_theta=np.linspace(0.029, 0.971, 4),
        )
        rng = np.random.default_rng(191 + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.02, size=dofs.shape)
        surface.set_dofs(dofs)
        cotangent = rng.normal(size=surface.normal().shape)
        spec = surface.surface_spec()
        args_tail = (
            jnp.asarray(surface.quadpoints_phi),
            jnp.asarray(surface.quadpoints_theta),
            surface.mpol,
            surface.ntor,
            surface.nfp,
            surface.stellsym,
            spec.scatter_indices,
            spec.coeff_template,
        )

        _, vjp_fn = jax.vjp(
            lambda dofs_arg: surface_xyzfourier_normal_from_dofs(dofs_arg, *args_tail),
            jnp.asarray(dofs),
        )
        (vjp_jax,) = vjp_fn(jnp.asarray(cotangent))

        np.testing.assert_allclose(
            np.asarray(vjp_jax),
            surface.dnormal_by_dcoeff_vjp(cotangent),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )


_NON_RZ_FORM_FUNCTIONS = {
    "SurfaceXYZFourier": (
        surface_xyz_fourier_first_fund_form_from_spec,
        surface_xyz_fourier_second_fund_form_from_spec,
        surface_xyz_fourier_surface_curvatures_from_spec,
        surface_xyz_fourier_dfirst_fund_form_from_dofs,
        surface_xyz_fourier_dsecond_fund_form_from_dofs,
        surface_xyz_fourier_dsurface_curvatures_from_dofs,
    ),
    "SurfaceXYZTensorFourier": (
        surface_xyz_tensor_fourier_first_fund_form_from_spec,
        surface_xyz_tensor_fourier_second_fund_form_from_spec,
        surface_xyz_tensor_fourier_surface_curvatures_from_spec,
        surface_xyz_tensor_fourier_dfirst_fund_form_from_dofs,
        surface_xyz_tensor_fourier_dsecond_fund_form_from_dofs,
        surface_xyz_tensor_fourier_dsurface_curvatures_from_dofs,
    ),
}


@pytest.mark.parametrize("surface_cls_name", [
    "SurfaceXYZFourier",
    "SurfaceXYZTensorFourier",
])
@pytest.mark.parametrize("stellsym", [True, False])
def test_non_rz_fundamental_forms_and_curvatures_match_cpp(
    surface_cls_name,
    stellsym,
):
    """Spec-level non-RZ form helpers match CPU ``Surface`` methods."""

    from simsopt import geo

    surface_cls = getattr(geo, surface_cls_name)
    nfp = 3
    surface = surface_cls(
        mpol=2,
        ntor=1,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=np.linspace(0.013, (1.0 / nfp) - 0.011, 5),
        quadpoints_theta=np.linspace(0.017, 0.983, 4),
    )
    rng = np.random.default_rng(211 + 10 * int(stellsym) + len(surface_cls_name))
    dofs = surface.get_dofs().copy()
    dofs += rng.normal(scale=0.02, size=dofs.shape)
    surface.set_dofs(dofs)
    spec = surface.surface_spec()
    first_fn, second_fn, curvature_fn, _dfirst_fn, _dsecond_fn, _dcurvature_fn = (
        _NON_RZ_FORM_FUNCTIONS[surface_cls_name]
    )

    np.testing.assert_allclose(
        np.asarray(first_fn(spec)),
        surface.first_fund_form(),
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )
    np.testing.assert_allclose(
        np.asarray(second_fn(spec)),
        surface.second_fund_form(),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )
    np.testing.assert_allclose(
        np.asarray(curvature_fn(spec)),
        surface.surface_curvatures(),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )


@pytest.mark.parametrize("surface_cls_name", [
    "SurfaceXYZFourier",
    "SurfaceXYZTensorFourier",
])
@pytest.mark.parametrize("stellsym", [True, False])
def test_non_rz_fundamental_form_derivatives_match_cpp(
    surface_cls_name,
    stellsym,
):
    """Dof-Jacobian non-RZ form helpers match CPU ``*_by_dcoeff`` methods."""

    from simsopt import geo

    surface_cls = getattr(geo, surface_cls_name)
    nfp = 2
    surface = surface_cls(
        mpol=1,
        ntor=1,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=np.linspace(0.019, (1.0 / nfp) - 0.023, 4),
        quadpoints_theta=np.linspace(0.031, 0.969, 3),
    )
    rng = np.random.default_rng(251 + 10 * int(stellsym) + len(surface_cls_name))
    dofs = surface.get_dofs().copy()
    dofs += rng.normal(scale=0.015, size=dofs.shape)
    surface.set_dofs(dofs)
    spec = surface.surface_spec()
    _first_fn, _second_fn, _curvature_fn, dfirst_fn, dsecond_fn, dcurvature_fn = (
        _NON_RZ_FORM_FUNCTIONS[surface_cls_name]
    )

    np.testing.assert_allclose(
        np.asarray(dfirst_fn(spec, spec.dofs)),
        surface.dfirst_fund_form_by_dcoeff(),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )
    np.testing.assert_allclose(
        np.asarray(dsecond_fn(spec, spec.dofs)),
        surface.dsecond_fund_form_by_dcoeff(),
        rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
    )
    np.testing.assert_allclose(
        np.asarray(dcurvature_fn(spec, spec.dofs)),
        surface.dsurface_curvatures_by_dcoeff(),
        rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
    )


class TestSurfaceFourierSecondNormalDerivativeParity:
    """Compare explicit heavy normal Hessian APIs against the CPU oracle."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")
        pytest.importorskip("simsopt")

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_d2normal_by_dcoeffdcoeff_matches_cpp(self, surface_cls_name, stellsym):
        from simsopt import geo

        surface_cls = getattr(geo, surface_cls_name)
        nfp = 2
        surface = surface_cls(
            mpol=1,
            ntor=1,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.011, (1.0 / nfp) - 0.017, 3),
            quadpoints_theta=np.linspace(0.019, 0.971, 4),
        )
        rng = np.random.default_rng(131 + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.04, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        if surface_cls_name == "SurfaceXYZFourier":
            derivative_jax = surface_xyzfourier_d2normal_by_dcoeffdcoeff(
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices,
                spec.coeff_template,
            )
        else:
            derivative_jax = d2normal_by_dcoeffdcoeff(
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices if stellsym else None,
            )

        np.testing.assert_allclose(
            np.asarray(derivative_jax),
            surface.d2normal_by_dcoeffdcoeff(),
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )


class TestSurfaceFourierPairedPointParity:
    """Compare paired-point ``*_lin`` kernels against CPU pybind methods."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")
        pytest.importorskip("simsopt")

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_gamma_and_tangent_lin_match_cpp(self, surface_cls_name, stellsym):
        from simsopt import geo

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                (
                    surface_xyzfourier_gamma_lin_from_dofs,
                    surface_xyzfourier_gammadash1_lin_from_dofs,
                    surface_xyzfourier_gammadash2_lin_from_dofs,
                ),
                111,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                (
                    surface_gamma_lin_from_dofs,
                    surface_gammadash1_lin_from_dofs,
                    surface_gammadash2_lin_from_dofs,
                ),
                121,
            ),
        }
        surface_cls, jax_fns, seed_base = case_by_class[surface_cls_name]
        nfp = 5
        surface = surface_cls(
            mpol=3,
            ntor=2,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 8, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 7, endpoint=False),
        )
        rng = np.random.default_rng(seed_base + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.02, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()
        phis = np.linspace(0.006, (1.0 / nfp) - 0.004, 9)
        thetas = np.linspace(0.021, 0.979, 9)

        if surface_cls_name == "SurfaceXYZFourier":
            args = (
                jnp.asarray(dofs),
                jnp.asarray(phis),
                jnp.asarray(thetas),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices,
                spec.coeff_template,
            )
        else:
            args = (
                jnp.asarray(dofs),
                jnp.asarray(phis),
                jnp.asarray(thetas),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices if stellsym else None,
            )

        for jax_fn, cpp_method in zip(
            jax_fns,
            ("gamma_lin", "gammadash1_lin", "gammadash2_lin"),
        ):
            expected = np.zeros((phis.size, 3))
            getattr(surface, cpp_method)(expected, phis, thetas)
            np.testing.assert_allclose(
                np.asarray(jax_fn(*args)),
                expected,
                rtol=1e-12,
                atol=1e-12,
            )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_higher_paired_lin_wrappers_match_cpp(self, surface_cls_name, stellsym):
        from simsopt import geo
        from simsopt.jax_core import (
            surface_xyz_fourier_gammadash1dash1_lin_from_dofs,
            surface_xyz_fourier_gammadash1dash1_lin_from_spec,
            surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs,
            surface_xyz_fourier_gammadash1dash1dash1_lin_from_spec,
            surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs,
            surface_xyz_fourier_gammadash1dash1dash2_lin_from_spec,
            surface_xyz_fourier_gammadash1dash2_lin_from_dofs,
            surface_xyz_fourier_gammadash1dash2_lin_from_spec,
            surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs,
            surface_xyz_fourier_gammadash1dash2dash2_lin_from_spec,
            surface_xyz_fourier_gammadash2dash2_lin_from_dofs,
            surface_xyz_fourier_gammadash2dash2_lin_from_spec,
            surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs,
            surface_xyz_fourier_gammadash2dash2dash2_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash1_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash1dash1_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash2_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash1dash2_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash2dash2_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash2dash2_lin_from_spec,
            surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_dofs,
            surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_spec,
        )

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                (
                    (
                        "gammadash1dash1_lin",
                        surface_xyz_fourier_gammadash1dash1_lin_from_spec,
                        surface_xyz_fourier_gammadash1dash1_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash2_lin",
                        surface_xyz_fourier_gammadash1dash2_lin_from_spec,
                        surface_xyz_fourier_gammadash1dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash2dash2_lin",
                        surface_xyz_fourier_gammadash2dash2_lin_from_spec,
                        surface_xyz_fourier_gammadash2dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash1dash1_lin",
                        surface_xyz_fourier_gammadash1dash1dash1_lin_from_spec,
                        surface_xyz_fourier_gammadash1dash1dash1_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash1dash2_lin",
                        surface_xyz_fourier_gammadash1dash1dash2_lin_from_spec,
                        surface_xyz_fourier_gammadash1dash1dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash2dash2_lin",
                        surface_xyz_fourier_gammadash1dash2dash2_lin_from_spec,
                        surface_xyz_fourier_gammadash1dash2dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash2dash2dash2_lin",
                        surface_xyz_fourier_gammadash2dash2dash2_lin_from_spec,
                        surface_xyz_fourier_gammadash2dash2dash2_lin_from_dofs,
                    ),
                ),
                131,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                (
                    (
                        "gammadash1dash1_lin",
                        surface_xyz_tensor_fourier_gammadash1dash1_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash1dash1_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash2_lin",
                        surface_xyz_tensor_fourier_gammadash1dash2_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash1dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash2dash2_lin",
                        surface_xyz_tensor_fourier_gammadash2dash2_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash2dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash1dash1_lin",
                        surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash1dash1dash1_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash1dash2_lin",
                        surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash1dash1dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash1dash2dash2_lin",
                        surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash1dash2dash2_lin_from_dofs,
                    ),
                    (
                        "gammadash2dash2dash2_lin",
                        surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_spec,
                        surface_xyz_tensor_fourier_gammadash2dash2dash2_lin_from_dofs,
                    ),
                ),
                141,
            ),
        }
        surface_cls, cases, seed_base = case_by_class[surface_cls_name]
        nfp = 5
        surface = surface_cls(
            mpol=3,
            ntor=2,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 8, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 7, endpoint=False),
        )
        rng = np.random.default_rng(seed_base + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.02, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()
        phis = np.linspace(0.006, (1.0 / nfp) - 0.004, 9)
        thetas = np.linspace(0.021, 0.979, 9)

        for cpp_method, spec_fn, dofs_fn in cases:
            expected = np.zeros((phis.size, 3))
            getattr(surface, cpp_method)(expected, phis, thetas)
            np.testing.assert_allclose(
                np.asarray(spec_fn(spec, phis, thetas)),
                expected,
                rtol=1e-11,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                np.asarray(dofs_fn(spec, dofs, phis, thetas)),
                expected,
                rtol=1e-11,
                atol=1e-10,
            )


class TestSurfaceFourierSpecCppParity:
    """Compare immutable non-RZ surface specs against CPU surface geometry."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")
        pytest.importorskip("simsopt")

    def test_tensor_surface_spec_supports_clamped_dims(self):
        """Regression: ``surface_spec()`` now constructs a spec for any
        ``clamped_dims`` combination. Full CPU/JAX parity for each of the
        8 combinations lives in
        ``tests/geo/test_surface_xyz_tensor_clamped_jax.py``.
        """
        from simsopt.geo import SurfaceXYZTensorFourier

        surface = SurfaceXYZTensorFourier(
            mpol=2,
            ntor=1,
            nfp=2,
            stellsym=True,
            clamped_dims=[True, False, False],
            quadpoints_phi=np.linspace(0, 0.5, 7, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, 6, endpoint=False),
        )

        spec = surface.surface_spec()
        assert spec.clamped_dims == (True, False, False)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_spec_geometry_and_normals_match_cpp(self, surface_cls_name, stellsym):
        from simsopt import geo
        from simsopt.jax_core import (
            surface_xyz_fourier_area_from_spec,
            surface_xyz_fourier_gamma_from_spec,
            surface_xyz_fourier_normal_from_spec,
            surface_xyz_fourier_unitnormal_from_spec,
            surface_xyz_fourier_volume_from_spec,
            surface_xyz_tensor_fourier_area_from_spec,
            surface_xyz_tensor_fourier_gamma_from_spec,
            surface_xyz_tensor_fourier_normal_from_spec,
            surface_xyz_tensor_fourier_unitnormal_from_spec,
            surface_xyz_tensor_fourier_volume_from_spec,
        )

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                surface_xyz_fourier_gamma_from_spec,
                surface_xyz_fourier_normal_from_spec,
                surface_xyz_fourier_unitnormal_from_spec,
                surface_xyz_fourier_area_from_spec,
                surface_xyz_fourier_volume_from_spec,
                11,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                surface_xyz_tensor_fourier_gamma_from_spec,
                surface_xyz_tensor_fourier_normal_from_spec,
                surface_xyz_tensor_fourier_unitnormal_from_spec,
                surface_xyz_tensor_fourier_area_from_spec,
                surface_xyz_tensor_fourier_volume_from_spec,
                21,
            ),
        }
        (
            surface_cls,
            gamma_from_spec,
            normal_from_spec,
            unitnormal_from_spec,
            area_from_spec,
            volume_from_spec,
            seed_base,
        ) = case_by_class[surface_cls_name]
        rng = np.random.default_rng(seed_base + int(not stellsym))
        surface = surface_cls(
            mpol=2,
            ntor=1,
            nfp=2,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0, 0.5, 7, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, 6, endpoint=False),
        )
        dofs = surface.get_dofs().copy()
        dofs[:] = rng.normal(size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        np.testing.assert_allclose(
            np.asarray(jax.jit(gamma_from_spec)(spec)),
            surface.gamma(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(normal_from_spec)(spec)),
            surface.normal(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(unitnormal_from_spec)(spec)),
            surface.unitnormal(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            float(jax.jit(area_from_spec)(spec)),
            surface.area(),
            rtol=_DERIVATIVE_HEAVY_TOLS["scalar_value_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["scalar_value_atol"],
        )
        np.testing.assert_allclose(
            float(jax.jit(volume_from_spec)(spec)),
            surface.volume(),
            rtol=_DERIVATIVE_HEAVY_TOLS["scalar_value_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["scalar_value_atol"],
        )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    def test_high_order_spec_geometry_is_finite_and_matches_cpp(self, surface_cls_name):
        from simsopt import geo
        from simsopt.jax_core import (
            surface_xyz_fourier_area_from_spec,
            surface_xyz_fourier_gamma_from_spec,
            surface_xyz_fourier_normal_from_spec,
            surface_xyz_fourier_unitnormal_from_spec,
            surface_xyz_fourier_volume_from_spec,
            surface_xyz_tensor_fourier_area_from_spec,
            surface_xyz_tensor_fourier_gamma_from_spec,
            surface_xyz_tensor_fourier_normal_from_spec,
            surface_xyz_tensor_fourier_unitnormal_from_spec,
            surface_xyz_tensor_fourier_volume_from_spec,
        )

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                surface_xyz_fourier_gamma_from_spec,
                surface_xyz_fourier_normal_from_spec,
                surface_xyz_fourier_unitnormal_from_spec,
                surface_xyz_fourier_area_from_spec,
                surface_xyz_fourier_volume_from_spec,
                111,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                surface_xyz_tensor_fourier_gamma_from_spec,
                surface_xyz_tensor_fourier_normal_from_spec,
                surface_xyz_tensor_fourier_unitnormal_from_spec,
                surface_xyz_tensor_fourier_area_from_spec,
                surface_xyz_tensor_fourier_volume_from_spec,
                121,
            ),
        }
        (
            surface_cls,
            gamma_from_spec,
            normal_from_spec,
            unitnormal_from_spec,
            area_from_spec,
            volume_from_spec,
            seed,
        ) = case_by_class[surface_cls_name]
        rng = np.random.default_rng(seed)
        surface = surface_cls(
            mpol=10,
            ntor=10,
            nfp=3,
            stellsym=False,
            quadpoints_phi=np.linspace(0.003, (1.0 / 3.0) - 0.005, 7),
            quadpoints_theta=np.linspace(0.007, 0.991, 6),
        )
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=1e-3, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        for jax_fn, expected in (
            (gamma_from_spec, surface.gamma()),
            (normal_from_spec, surface.normal()),
            (unitnormal_from_spec, surface.unitnormal()),
        ):
            actual = np.asarray(jax.jit(jax_fn)(spec))
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

        for jax_fn, expected in (
            (area_from_spec, surface.area()),
            (volume_from_spec, surface.volume()),
        ):
            actual = float(jax.jit(jax_fn)(spec))
            assert np.isfinite(actual)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=_DERIVATIVE_HEAVY_TOLS["scalar_value_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["scalar_value_atol"],
            )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_spec_second_coordinate_derivatives_match_cpp(
        self,
        surface_cls_name,
        stellsym,
    ):
        from simsopt import geo
        from simsopt.jax_core import (
            surface_xyz_fourier_gammadash1dash1_from_spec,
            surface_xyz_fourier_gammadash1dash2_from_spec,
            surface_xyz_fourier_gammadash2dash2_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash1_from_spec,
            surface_xyz_tensor_fourier_gammadash1dash2_from_spec,
            surface_xyz_tensor_fourier_gammadash2dash2_from_spec,
        )

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                (
                    surface_xyz_fourier_gammadash1dash1_from_spec,
                    surface_xyz_fourier_gammadash1dash2_from_spec,
                    surface_xyz_fourier_gammadash2dash2_from_spec,
                ),
                51,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                (
                    surface_xyz_tensor_fourier_gammadash1dash1_from_spec,
                    surface_xyz_tensor_fourier_gammadash1dash2_from_spec,
                    surface_xyz_tensor_fourier_gammadash2dash2_from_spec,
                ),
                61,
            ),
        }
        surface_cls, jax_fns, seed_base = case_by_class[surface_cls_name]
        nfp = 3
        rng = np.random.default_rng(seed_base + int(stellsym))
        surface = surface_cls(
            mpol=3,
            ntor=2,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.009, (1.0 / nfp) - 0.006, 8),
            quadpoints_theta=np.linspace(0.021, 0.981, 7),
        )
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.025, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        for jax_fn, cpp_method in zip(
            jax_fns,
            ("gammadash1dash1", "gammadash1dash2", "gammadash2dash2"),
        ):
            np.testing.assert_allclose(
                np.asarray(jax.jit(jax_fn)(spec)),
                getattr(surface, cpp_method)(),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
            )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    @pytest.mark.parametrize("scalar_name", ["area", "volume"])
    def test_area_volume_derivatives_match_cpp(
        self,
        surface_cls_name,
        stellsym,
        scalar_name,
    ):
        from simsopt import geo

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                {
                    "area": (
                        surface_xyzfourier_area_from_dofs,
                        surface_xyzfourier_darea_by_dcoeff,
                        surface_xyzfourier_d2area_by_dcoeffdcoeff,
                    ),
                    "volume": (
                        surface_xyzfourier_volume_from_dofs,
                        surface_xyzfourier_dvolume_by_dcoeff,
                        surface_xyzfourier_d2volume_by_dcoeffdcoeff,
                    ),
                },
                71,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                {
                    "area": (
                        surface_area_from_dofs,
                        darea_by_dcoeff,
                        d2area_by_dcoeffdcoeff,
                    ),
                    "volume": (
                        surface_volume_from_dofs,
                        dvolume_by_dcoeff,
                        d2volume_by_dcoeffdcoeff,
                    ),
                },
                81,
            ),
        }
        surface_cls, fn_by_scalar, seed_base = case_by_class[surface_cls_name]
        value_fn, grad_fn, hessian_fn = fn_by_scalar[scalar_name]
        nfp = 4
        surface = surface_cls(
            mpol=2,
            ntor=1,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.011, (1.0 / nfp) - 0.009, 7),
            quadpoints_theta=np.linspace(0.017, 0.983, 6),
        )
        rng = np.random.default_rng(seed_base + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.02, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        if surface_cls_name == "SurfaceXYZFourier":
            args = (
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices,
                spec.coeff_template,
            )
        else:
            args = (
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices if stellsym else None,
            )

        np.testing.assert_allclose(
            float(value_fn(*args)),
            getattr(surface, scalar_name)(),
            rtol=_DERIVATIVE_HEAVY_TOLS["scalar_value_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["scalar_value_atol"],
        )
        np.testing.assert_allclose(
            np.asarray(grad_fn(*args)),
            getattr(surface, f"d{scalar_name}_by_dcoeff")(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        np.testing.assert_allclose(
            np.asarray(hessian_fn(*args)),
            getattr(surface, f"d2{scalar_name}_by_dcoeffdcoeff")(),
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("scalar_name", ["area", "volume"])
    def test_nonstellsym_area_volume_hessian_mpol_gt_2_matches_cpp(
        self,
        surface_cls_name,
        scalar_name,
    ):
        from simsopt import geo

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                {
                    "area": surface_xyzfourier_d2area_by_dcoeffdcoeff,
                    "volume": surface_xyzfourier_d2volume_by_dcoeffdcoeff,
                },
                131,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                {
                    "area": d2area_by_dcoeffdcoeff,
                    "volume": d2volume_by_dcoeffdcoeff,
                },
                141,
            ),
        }
        surface_cls, hessian_by_scalar, seed_base = case_by_class[surface_cls_name]
        surface = surface_cls(
            mpol=3,
            ntor=2,
            nfp=3,
            stellsym=False,
            quadpoints_phi=np.linspace(0.011, (1.0 / 3.0) - 0.013, 6),
            quadpoints_theta=np.linspace(0.017, 0.983, 5),
        )
        rng = np.random.default_rng(seed_base)
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.015, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        if surface_cls_name == "SurfaceXYZFourier":
            args = (
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices,
                spec.coeff_template,
            )
        else:
            args = (
                jnp.asarray(dofs),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                None,
            )

        np.testing.assert_allclose(
            np.asarray(hessian_by_scalar[scalar_name](*args)),
            getattr(surface, f"d2{scalar_name}_by_dcoeffdcoeff")(),
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    @pytest.mark.parametrize("scalar_name", ["area", "volume"])
    def test_area_volume_derivative_taylor_residuals(
        self,
        surface_cls_name,
        stellsym,
        scalar_name,
    ):
        from simsopt import geo

        case_by_class = {
            "SurfaceXYZFourier": (
                geo.SurfaceXYZFourier,
                {
                    "area": (
                        surface_xyzfourier_area_from_dofs,
                        surface_xyzfourier_darea_by_dcoeff,
                        surface_xyzfourier_d2area_by_dcoeffdcoeff,
                    ),
                    "volume": (
                        surface_xyzfourier_volume_from_dofs,
                        surface_xyzfourier_dvolume_by_dcoeff,
                        surface_xyzfourier_d2volume_by_dcoeffdcoeff,
                    ),
                },
                91,
            ),
            "SurfaceXYZTensorFourier": (
                geo.SurfaceXYZTensorFourier,
                {
                    "area": (
                        surface_area_from_dofs,
                        darea_by_dcoeff,
                        d2area_by_dcoeffdcoeff,
                    ),
                    "volume": (
                        surface_volume_from_dofs,
                        dvolume_by_dcoeff,
                        d2volume_by_dcoeffdcoeff,
                    ),
                },
                101,
            ),
        }
        surface_cls, fn_by_scalar, seed_base = case_by_class[surface_cls_name]
        value_fn, grad_fn, hessian_fn = fn_by_scalar[scalar_name]
        nfp = 3
        surface = surface_cls(
            mpol=2,
            ntor=1,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.013, (1.0 / nfp) - 0.011, 7),
            quadpoints_theta=np.linspace(0.019, 0.981, 6),
        )
        rng = np.random.default_rng(seed_base + int(stellsym))
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.015, size=dofs.shape)
        direction = rng.normal(size=dofs.shape)
        direction /= np.linalg.norm(direction)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        def args_for(dofs_value):
            if surface_cls_name == "SurfaceXYZFourier":
                return (
                    jnp.asarray(dofs_value),
                    jnp.asarray(surface.quadpoints_phi),
                    jnp.asarray(surface.quadpoints_theta),
                    surface.mpol,
                    surface.ntor,
                    surface.nfp,
                    surface.stellsym,
                    spec.scatter_indices,
                    spec.coeff_template,
                )
            return (
                jnp.asarray(dofs_value),
                jnp.asarray(surface.quadpoints_phi),
                jnp.asarray(surface.quadpoints_theta),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                spec.scatter_indices if stellsym else None,
            )

        base_args = args_for(dofs)
        f0 = float(value_fn(*base_args))
        grad0 = np.asarray(grad_fn(*base_args))
        hessian0 = np.asarray(hessian_fn(*base_args))
        slope = float(np.dot(grad0, direction))
        curvature = float(direction @ hessian0 @ direction)

        def residuals(eps):
            f_eps = float(value_fn(*args_for(dofs + eps * direction)))
            delta = f_eps - f0
            first = abs(delta - eps * slope)
            second = abs(delta - eps * slope - 0.5 * eps * eps * curvature)
            return first, second

        first_big, second_big = residuals(1.0e-2)
        first_small, second_small = residuals(5.0e-3)

        assert first_small < 0.35 * first_big
        assert second_small < max(0.25 * second_big, 1e-11)
        assert second_small < first_small

    @pytest.mark.parametrize("stellsym", [True, False])
    def test_tensor_spec_tangents_area_and_volume_match_cpp(self, stellsym):
        from simsopt.geo import SurfaceXYZTensorFourier
        from simsopt.jax_core import (
            surface_xyz_tensor_fourier_area_from_spec,
            surface_xyz_tensor_fourier_gammadash1_from_spec,
            surface_xyz_tensor_fourier_gammadash2_from_spec,
            surface_xyz_tensor_fourier_volume_from_spec,
        )

        rng = np.random.default_rng(31 + int(not stellsym))
        surface = SurfaceXYZTensorFourier(
            mpol=3,
            ntor=2,
            nfp=5,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(
                0.011,
                (1.0 / 5.0) - 0.007,
                9,
                endpoint=False,
            ),
            quadpoints_theta=np.linspace(0.017, 0.983, 8, endpoint=False),
        )
        dofs = surface.get_dofs().copy()
        dofs += rng.normal(scale=0.02, size=dofs.shape)
        surface.set_dofs(dofs)
        spec = surface.surface_spec()

        np.testing.assert_allclose(
            np.asarray(jax.jit(surface_xyz_tensor_fourier_gammadash1_from_spec)(spec)),
            surface.gammadash1(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(surface_xyz_tensor_fourier_gammadash2_from_spec)(spec)),
            surface.gammadash2(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            float(jax.jit(surface_xyz_tensor_fourier_area_from_spec)(spec)),
            surface.area(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        np.testing.assert_allclose(
            float(jax.jit(surface_xyz_tensor_fourier_volume_from_spec)(spec)),
            surface.volume(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )


class TestSurfaceFourierObjectApiParity:
    """Exercise non-RZ host APIs that must still produce valid immutable specs."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        pytest.importorskip("simsoptpp")
        pytest.importorskip("simsopt")

    @staticmethod
    def _make_torus_surface(surface_cls_name, stellsym, *, mpol=2, ntor=1):
        from simsopt import geo

        surface_cls = getattr(geo, surface_cls_name)
        nfp = 3
        surface = surface_cls(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 5, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 5, endpoint=False),
        )
        if surface_cls_name == "SurfaceXYZFourier":
            for coefficients in (
                surface.xc,
                surface.xs,
                surface.yc,
                surface.ys,
                surface.zc,
                surface.zs,
            ):
                coefficients[:, :] = 0.0
            surface.xc[0, ntor] = 1.4
            surface.xc[1, ntor] = 0.22
            surface.zs[1, ntor] = 0.22
            if mpol >= 2:
                surface.xc[2, ntor] = 0.015
        else:
            for coefficients in (surface.xcs, surface.ycs, surface.zcs):
                coefficients[:, :] = 0.0
            surface.xcs[0, 0] = 1.4
            surface.xcs[1, 0] = 0.22
            surface.zcs[mpol + 1, 0] = 0.22
            if mpol >= 2 and ntor >= 1:
                surface.xcs[2, 1] = 0.015
        surface.local_full_x = surface.get_dofs()
        return surface

    @staticmethod
    def _assert_non_rz_spec_parity(surface_cls_name, surface):
        from simsopt.jax_core import (
            surface_xyz_fourier_area_from_spec,
            surface_xyz_fourier_gamma_from_spec,
            surface_xyz_fourier_gammadash1_from_spec,
            surface_xyz_fourier_gammadash2_from_spec,
            surface_xyz_fourier_normal_from_spec,
            surface_xyz_fourier_unitnormal_from_spec,
            surface_xyz_fourier_volume_from_spec,
            surface_xyz_tensor_fourier_area_from_spec,
            surface_xyz_tensor_fourier_gamma_from_spec,
            surface_xyz_tensor_fourier_gammadash1_from_spec,
            surface_xyz_tensor_fourier_gammadash2_from_spec,
            surface_xyz_tensor_fourier_normal_from_spec,
            surface_xyz_tensor_fourier_unitnormal_from_spec,
            surface_xyz_tensor_fourier_volume_from_spec,
        )

        function_by_class = {
            "SurfaceXYZFourier": (
                surface_xyz_fourier_gamma_from_spec,
                surface_xyz_fourier_gammadash1_from_spec,
                surface_xyz_fourier_gammadash2_from_spec,
                surface_xyz_fourier_normal_from_spec,
                surface_xyz_fourier_unitnormal_from_spec,
                surface_xyz_fourier_area_from_spec,
                surface_xyz_fourier_volume_from_spec,
            ),
            "SurfaceXYZTensorFourier": (
                surface_xyz_tensor_fourier_gamma_from_spec,
                surface_xyz_tensor_fourier_gammadash1_from_spec,
                surface_xyz_tensor_fourier_gammadash2_from_spec,
                surface_xyz_tensor_fourier_normal_from_spec,
                surface_xyz_tensor_fourier_unitnormal_from_spec,
                surface_xyz_tensor_fourier_area_from_spec,
                surface_xyz_tensor_fourier_volume_from_spec,
            ),
        }
        (
            gamma_from_spec,
            gammadash1_from_spec,
            gammadash2_from_spec,
            normal_from_spec,
            unitnormal_from_spec,
            area_from_spec,
            volume_from_spec,
        ) = function_by_class[surface_cls_name]
        spec = surface.surface_spec()

        np.testing.assert_allclose(
            np.asarray(jax.jit(gamma_from_spec)(spec)),
            surface.gamma(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(gammadash1_from_spec)(spec)),
            surface.gammadash1(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(gammadash2_from_spec)(spec)),
            surface.gammadash2(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(normal_from_spec)(spec)),
            surface.normal(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(jax.jit(unitnormal_from_spec)(spec)),
            surface.unitnormal(),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            float(jax.jit(area_from_spec)(spec)),
            surface.area(),
            rtol=_DERIVATIVE_HEAVY_TOLS["scalar_value_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["scalar_value_atol"],
        )
        np.testing.assert_allclose(
            float(jax.jit(volume_from_spec)(spec)),
            surface.volume(),
            rtol=_DERIVATIVE_HEAVY_TOLS["scalar_value_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["scalar_value_atol"],
        )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_copy_object_api_independent_dofs(self, surface_cls_name, stellsym):
        surface = self._make_torus_surface(surface_cls_name, stellsym)
        copied_surface = surface.copy()
        copied_dofs = copied_surface.get_dofs().copy()

        updated_dofs = surface.get_dofs().copy()
        updated_dofs[0] += 0.03
        surface.set_dofs(updated_dofs)

        np.testing.assert_allclose(copied_surface.get_dofs(), copied_dofs)
        assert not np.shares_memory(surface.get_dofs(), copied_surface.get_dofs())
        self._assert_non_rz_spec_parity(surface_cls_name, copied_surface)

    @pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy])
    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_copy_module_protocol_independent_dofs(
        self,
        copy_fn,
        surface_cls_name,
        stellsym,
    ):
        surface = self._make_torus_surface(surface_cls_name, stellsym)
        copied_surface = copy_fn(surface)
        copied_dofs = copied_surface.get_dofs().copy()

        updated_dofs = surface.get_dofs().copy()
        updated_dofs[0] += 0.03
        surface.set_dofs(updated_dofs)

        np.testing.assert_allclose(copied_surface.get_dofs(), copied_dofs)
        assert not np.shares_memory(surface.get_dofs(), copied_surface.get_dofs())
        self._assert_non_rz_spec_parity(surface_cls_name, copied_surface)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_copy_object_api_variants_preserve_spec_parity(
        self,
        surface_cls_name,
        stellsym,
    ):
        surface = self._make_torus_surface(surface_cls_name, stellsym)
        copied_surfaces = (
            surface.copy(nphi=9),
            surface.copy(ntheta=10),
            surface.copy(range="field period"),
            surface.copy(nfp=5),
            surface.copy(mpol=surface.mpol + 1, ntor=surface.ntor + 1),
            surface.copy(stellsym=not stellsym),
        )

        for copied_surface in copied_surfaces:
            self._assert_non_rz_spec_parity(surface_cls_name, copied_surface)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_to_rzfourier_and_cross_section_object_api_parity(
        self,
        surface_cls_name,
        stellsym,
    ):
        from simsopt.jax_core import surface_rz_fourier_gamma_from_spec

        surface = self._make_torus_surface(surface_cls_name, stellsym)
        thetas = np.linspace(0.0, 1.0, 9, endpoint=False)
        cross_section = surface.cross_section(0.125, thetas=thetas)
        angles = np.arctan2(cross_section[:, 1], cross_section[:, 0])

        np.testing.assert_allclose(angles, 0.25 * np.pi, atol=1e-12)
        self._assert_non_rz_spec_parity(surface_cls_name, surface)

        rz_surface = surface.to_RZFourier()
        np.testing.assert_allclose(rz_surface.gamma(), surface.gamma(), atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(jax.jit(surface_rz_fourier_gamma_from_spec)(
                rz_surface.surface_spec()
            )),
            rz_surface.gamma(),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_least_squares_fit_object_api_parity(self, surface_cls_name, stellsym):
        from simsopt import geo

        source_surface = self._make_torus_surface(surface_cls_name, stellsym)
        surface_cls = getattr(geo, surface_cls_name)
        fitted_surface = surface_cls(
            mpol=source_surface.mpol,
            ntor=source_surface.ntor,
            nfp=source_surface.nfp,
            stellsym=source_surface.stellsym,
            quadpoints_phi=source_surface.quadpoints_phi,
            quadpoints_theta=source_surface.quadpoints_theta,
        )

        fitted_surface.least_squares_fit(source_surface.gamma())

        np.testing.assert_allclose(
            fitted_surface.gamma(),
            source_surface.gamma(),
            rtol=1e-12,
            atol=1e-12,
        )
        self._assert_non_rz_spec_parity(surface_cls_name, fitted_surface)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("flip_theta", [True, False])
    def test_fit_to_curve_object_api_parity(self, surface_cls_name, flip_theta):
        from simsopt import geo

        surface_cls = getattr(geo, surface_cls_name)
        surface = surface_cls(
            mpol=1,
            ntor=0,
            nfp=1,
            stellsym=False,
            quadpoints_phi=np.linspace(0.0, 1.0, 9, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 11, endpoint=False),
        )
        curve = geo.CurveRZFourier(32, 1, 1, False)
        curve.set(0, 1.0)

        surface.fit_to_curve(curve, 0.2, flip_theta=flip_theta)

        np.testing.assert_allclose(surface.major_radius(), 1.0, atol=1e-12)
        np.testing.assert_allclose(surface.minor_radius(), 0.2, atol=1e-12)
        self._assert_non_rz_spec_parity(surface_cls_name, surface)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_scale_object_api_parity(self, surface_cls_name, stellsym):
        surface = self._make_torus_surface(
            surface_cls_name,
            stellsym,
            mpol=1,
            ntor=0,
        )
        gamma_before = surface.gamma().copy()
        minor_radius_before = surface.minor_radius()

        surface.scale(2.0)

        assert not np.allclose(surface.gamma(), gamma_before)
        np.testing.assert_allclose(
            surface.minor_radius(),
            2.0 * minor_radius_before,
            atol=1e-12,
        )
        self._assert_non_rz_spec_parity(surface_cls_name, surface)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("extension_method", [
        "extend_via_normal",
        "extend_via_projected_normal",
    ])
    def test_extend_object_api_parity(self, surface_cls_name, extension_method):
        surface = self._make_torus_surface(
            surface_cls_name,
            stellsym=True,
            mpol=1,
            ntor=0,
        )
        minor_radius = surface.minor_radius()

        getattr(surface, extension_method)(0.05)

        np.testing.assert_allclose(surface.minor_radius(), minor_radius + 0.05)
        self._assert_non_rz_spec_parity(surface_cls_name, surface)

    @pytest.mark.parametrize("surface_cls_name", [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ])
    @pytest.mark.parametrize("stellsym", [True, False])
    def test_serialization_object_api_parity(
        self,
        surface_cls_name,
        stellsym,
        tmp_path,
    ):
        from simsopt import load, save

        surface = self._make_torus_surface(surface_cls_name, stellsym)
        surface_file = tmp_path / "surface.json"

        save(surface, surface_file)
        loaded_surface = load(surface_file)

        np.testing.assert_allclose(loaded_surface.gamma(), surface.gamma())
        self._assert_non_rz_spec_parity(surface_cls_name, loaded_surface)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
