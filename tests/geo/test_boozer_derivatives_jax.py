"""
FD validation tests for M3 JAX Boozer derivative path.

Tests:
1. Surface coefficient Jacobians (dgamma_by_dcoeff, etc.) via FD.
2. Composed penalty gradient via FD.
3. Composed residual Jacobian via FD.
4. Outer coil VJP consistency.
5. Hessian symmetry and FD validation.
"""

import importlib.util
import sys
import types
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


# boozer_residual_jax.py uses relative imports (from .surface_fourier_jax ...)
# that need stub parent packages in sys.modules during importlib loading.
# We create temporary stubs, load the modules, then clean up so the real
# simsopt package can load normally in subsequent tests.
_stubs_added = []
for _pkg in ["simsopt", "simsopt.geo", "simsopt.field"]:
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = []
        sys.modules[_pkg] = _stub
        _stubs_added.append(_pkg)

_sf = _load("surface_fourier_jax", "geo/surface_fourier_jax.py")
sys.modules["simsopt.geo.surface_fourier_jax"] = _sf

_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
sys.modules["simsopt.field.biotsavart_jax"] = _bs

_br = _load("boozer_residual_jax", "geo/boozer_residual_jax.py")

# Clean up: remove stubs and synthetic entries so the real simsopt
# package (if installed) isn't shadowed for other test files.
for _entry in ["simsopt.geo.surface_fourier_jax", "simsopt.field.biotsavart_jax"]:
    sys.modules.pop(_entry, None)
for _pkg in reversed(_stubs_added):
    sys.modules.pop(_pkg, None)

surface_gamma_from_dofs = _sf.surface_gamma_from_dofs
surface_gammadash1_from_dofs = _sf.surface_gammadash1_from_dofs
surface_gammadash2_from_dofs = _sf.surface_gammadash2_from_dofs
dgamma_by_dcoeff = _sf.dgamma_by_dcoeff
dgammadash1_by_dcoeff = _sf.dgammadash1_by_dcoeff
dgammadash2_by_dcoeff = _sf.dgammadash2_by_dcoeff
stellsym_scatter_indices = _sf.stellsym_scatter_indices
biot_savart_B = _bs.biot_savart_B

boozer_penalty_composed = _br.boozer_penalty_composed
boozer_penalty_grad_composed = _br.boozer_penalty_grad_composed
boozer_residual_jacobian_composed = _br.boozer_residual_jacobian_composed
boozer_residual_coil_vjp = _br.boozer_residual_coil_vjp
boozer_residual_vector = _br.boozer_residual_vector


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_torus_dofs(mpol=1, ntor=0, nfp=1, R=1.0, r=0.1, stellsym=False):
    """Create DOF vector for a simple circular torus (non-stellsym)."""
    n_per = (2 * mpol + 1) * (2 * ntor + 1)
    dofs = np.zeros(3 * n_per)
    ncol = 2 * ntor + 1
    # xc[0,0] = R (constant term)
    dofs[0 * ncol + 0] = R
    # xc[1,0] = r (cos theta term)
    dofs[1 * ncol + 0] = r
    # zc[mpol+1, 0] = r (sin theta term)
    dofs[2 * n_per + (mpol + 1) * ncol + 0] = r
    return jnp.array(dofs)


def _make_torus_dofs_stellsym(mpol=1, ntor=0, nfp=1, R=1.0, r=0.1):
    """Create DOF vector for a simple circular torus (stellsym)."""
    scatter_idx = stellsym_scatter_indices(mpol, ntor)
    n_per = (2 * mpol + 1) * (2 * ntor + 1)

    # Build full coefficient arrays
    full = np.zeros(3 * n_per)
    ncol = 2 * ntor + 1
    full[0 * ncol + 0] = R  # xc[0,0]
    full[1 * ncol + 0] = r  # xc[1,0]
    full[2 * n_per + (mpol + 1) * ncol + 0] = r  # zc[mpol+1,0]

    # Extract only the free DOFs
    sdofs = full[scatter_idx]
    return jnp.array(sdofs), jnp.array(scatter_idx)


def _make_coil_data(ncoils=3, nquad=32):
    """Create synthetic coil data for a simple coilset."""
    R_coil = 1.5
    gammas = np.zeros((ncoils, nquad, 3))
    gammadashs = np.zeros((ncoils, nquad, 3))

    for i in range(ncoils):
        phi_offset = 2 * np.pi * i / ncoils
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        gammas[i, :, 0] = R_coil * np.cos(t + phi_offset)
        gammas[i, :, 1] = R_coil * np.sin(t + phi_offset)
        gammas[i, :, 2] = 0.0
        gammadashs[i, :, 0] = -R_coil * np.sin(t + phi_offset) * 2 * np.pi
        gammadashs[i, :, 1] = R_coil * np.cos(t + phi_offset) * 2 * np.pi
        gammadashs[i, :, 2] = 0.0

    currents = np.array([1e5, 1e5, 1e5])
    return (
        jnp.array(gammas),
        jnp.array(gammadashs),
        jnp.array(currents),
    )


# ---------------------------------------------------------------------------
# Test: Surface coefficient Jacobians
# ---------------------------------------------------------------------------


class TestDgammaByDcoeff:
    """Validate dgamma_by_dcoeff via finite differences."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 8, 8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)

    def test_dgamma_shape(self):
        J = dgamma_by_dcoeff(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )
        ndofs = len(self.dofs)
        assert J.shape == (self.nphi, self.ntheta, 3, ndofs)

    def test_dgamma_fd(self):
        """dgamma_by_dcoeff matches centred finite differences."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, False)
        J = np.array(dgamma_by_dcoeff(self.dofs, *args))

        eps = 1e-6
        ndofs = len(self.dofs)
        J_fd = np.zeros_like(J)
        for k in range(ndofs):
            dofs_p = self.dofs.at[k].add(eps)
            dofs_m = self.dofs.at[k].add(-eps)
            g_p = np.array(surface_gamma_from_dofs(dofs_p, *args))
            g_m = np.array(surface_gamma_from_dofs(dofs_m, *args))
            J_fd[:, :, :, k] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-10)

    def test_dgammadash1_fd(self):
        """dgammadash1_by_dcoeff matches centred finite differences."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, False)
        J = np.array(dgammadash1_by_dcoeff(self.dofs, *args))

        eps = 1e-6
        ndofs = len(self.dofs)
        J_fd = np.zeros_like(J)
        for k in range(ndofs):
            dofs_p = self.dofs.at[k].add(eps)
            dofs_m = self.dofs.at[k].add(-eps)
            g_p = np.array(surface_gammadash1_from_dofs(dofs_p, *args))
            g_m = np.array(surface_gammadash1_from_dofs(dofs_m, *args))
            J_fd[:, :, :, k] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-10)

    def test_dgammadash2_fd(self):
        """dgammadash2_by_dcoeff matches centred finite differences."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, False)
        J = np.array(dgammadash2_by_dcoeff(self.dofs, *args))

        eps = 1e-6
        ndofs = len(self.dofs)
        J_fd = np.zeros_like(J)
        for k in range(ndofs):
            dofs_p = self.dofs.at[k].add(eps)
            dofs_m = self.dofs.at[k].add(-eps)
            g_p = np.array(surface_gammadash2_from_dofs(dofs_p, *args))
            g_m = np.array(surface_gammadash2_from_dofs(dofs_m, *args))
            J_fd[:, :, :, k] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-10)


class TestDgammaByDcoeffStellsym:
    """Validate surface coefficient Jacobians with stellarator symmetry."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 8, 8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs, self.scatter_idx = _make_torus_dofs_stellsym(
            self.mpol,
            self.ntor,
            self.nfp,
        )

    def test_dgamma_stellsym_fd(self):
        """dgamma_by_dcoeff with stellsym matches FD."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp, True)
        J = np.array(
            dgamma_by_dcoeff(
                self.dofs,
                *args,
                scatter_indices=self.scatter_idx,
            )
        )

        eps = 1e-6
        ndofs = len(self.dofs)
        J_fd = np.zeros_like(J)
        for k in range(ndofs):
            dofs_p = self.dofs.at[k].add(eps)
            dofs_m = self.dofs.at[k].add(-eps)
            g_p = np.array(
                surface_gamma_from_dofs(
                    dofs_p,
                    *args,
                    scatter_indices=self.scatter_idx,
                )
            )
            g_m = np.array(
                surface_gamma_from_dofs(
                    dofs_m,
                    *args,
                    scatter_indices=self.scatter_idx,
                )
            )
            J_fd[:, :, :, k] = (g_p - g_m) / (2 * eps)

        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-10)

    def test_dgamma_stellsym_fewer_dofs(self):
        """Stellsym Jacobian has fewer DOF columns than non-stellsym."""
        args = (self.phis, self.thetas, self.mpol, self.ntor, self.nfp)
        J_stellsym = dgamma_by_dcoeff(
            self.dofs,
            *args,
            stellsym=True,
            scatter_indices=self.scatter_idx,
        )
        ndofs_stellsym = J_stellsym.shape[-1]

        n_per = (2 * self.mpol + 1) * (2 * self.ntor + 1)
        ndofs_full = 3 * n_per
        assert ndofs_stellsym < ndofs_full


# ---------------------------------------------------------------------------
# Test: Composed penalty gradient
# ---------------------------------------------------------------------------


class TestBoozerPenaltyGradComposed:
    """Validate the full-pipeline VJP gradient via finite differences."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 6, 6

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        # Decision vector: [sdofs, iota, G]
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            optimize_G=True,
            weight_inv_modB=False,
        )

    def test_gradient_fd(self):
        """Composed gradient matches centred finite differences."""
        val, grad = boozer_penalty_grad_composed(self.x, **self.kwargs)
        grad = np.array(grad)

        eps = 1e-6
        n = len(self.x)
        grad_fd = np.zeros(n)
        for k in range(n):
            x_p = self.x.at[k].add(eps)
            x_m = self.x.at[k].add(-eps)
            f_p = float(boozer_penalty_composed(x_p, **self.kwargs))
            f_m = float(boozer_penalty_composed(x_m, **self.kwargs))
            grad_fd[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-10)

    def test_surface_dof_gradient_nonzero(self):
        """Unlike M1, composed gradient has nonzero surface DOF entries."""
        val, grad = boozer_penalty_grad_composed(self.x, **self.kwargs)
        grad = np.array(grad)
        # Surface DOFs are x[:-2]; at least some should be nonzero
        sdof_grad = grad[:-2]
        assert np.max(np.abs(sdof_grad)) > 1e-12

    def test_gradient_optimize_G_false(self):
        """Gradient works with optimize_G=False (G from currents)."""
        x_no_G = jnp.concatenate([self.dofs, jnp.array([self.iota])])
        kwargs = {**self.kwargs, "optimize_G": False}

        val, grad = boozer_penalty_grad_composed(x_no_G, **kwargs)
        grad = np.array(grad)

        eps = 1e-6
        n = len(x_no_G)
        grad_fd = np.zeros(n)
        for k in range(n):
            x_p = x_no_G.at[k].add(eps)
            x_m = x_no_G.at[k].add(-eps)
            f_p = float(boozer_penalty_composed(x_p, **kwargs))
            f_m = float(boozer_penalty_composed(x_m, **kwargs))
            grad_fd[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: Composed residual Jacobian
# ---------------------------------------------------------------------------


class TestBoozerResidualJacobianComposed:
    """Validate the BoozerExact Jacobian via finite differences."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            weight_inv_modB=False,
        )

    def test_jacobian_shape(self):
        """Jacobian has correct shape (n_res, n_dofs)."""
        r, J = boozer_residual_jacobian_composed(self.x, **self.kwargs)
        n_res = 3 * self.nphi * self.ntheta
        n_dofs = len(self.x)
        assert J.shape == (n_res, n_dofs)
        assert r.shape == (n_res,)

    def test_jacobian_fd(self):
        """Jacobian matches centred finite differences."""
        from functools import partial

        res_fn = partial(_br._boozer_residual_vector_composed, **self.kwargs)

        r, J = boozer_residual_jacobian_composed(self.x, **self.kwargs)
        J = np.array(J)

        eps = 1e-5
        n = len(self.x)
        J_fd = np.zeros_like(J)
        for k in range(n):
            x_p = self.x.at[k].add(eps)
            x_m = self.x.at[k].add(-eps)
            r_p = np.array(res_fn(x_p))
            r_m = np.array(res_fn(x_m))
            J_fd[:, k] = (r_p - r_m) / (2 * eps)

        np.testing.assert_allclose(J, J_fd, rtol=1e-4, atol=1e-9)


# ---------------------------------------------------------------------------
# Test: Composed Hessian
# ---------------------------------------------------------------------------


class TestBoozerHessianComposed:
    """Validate Hessian of the composed penalty objective."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            optimize_G=True,
            weight_inv_modB=False,
        )

    def test_hessian_symmetry(self):
        """Hessian of composed objective is symmetric."""
        H = jax.hessian(boozer_penalty_composed)(self.x, **self.kwargs)
        np.testing.assert_allclose(np.array(H), np.array(H.T), atol=1e-12)

    def test_hessian_fd(self):
        """Hessian matches FD of gradient."""
        _, grad0 = boozer_penalty_grad_composed(self.x, **self.kwargs)
        H = np.array(
            jax.hessian(boozer_penalty_composed)(
                self.x,
                **self.kwargs,
            )
        )

        eps = 1e-5
        n = len(self.x)
        H_fd = np.zeros((n, n))
        for k in range(n):
            x_p = self.x.at[k].add(eps)
            x_m = self.x.at[k].add(-eps)
            _, g_p = boozer_penalty_grad_composed(x_p, **self.kwargs)
            _, g_m = boozer_penalty_grad_composed(x_m, **self.kwargs)
            H_fd[:, k] = (np.array(g_p) - np.array(g_m)) / (2 * eps)

        np.testing.assert_allclose(H, H_fd, rtol=1e-3, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: Outer coil VJP
# ---------------------------------------------------------------------------


class TestBoozerResidualCoilVJP:
    """Validate the outer residual VJP w.r.t. coil parameters."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0

        # Evaluate fixed surface geometry
        self.gamma = surface_gamma_from_dofs(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )
        self.xphi = surface_gammadash1_from_dofs(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )
        self.xtheta = surface_gammadash2_from_dofs(
            self.dofs,
            self.phis,
            self.thetas,
            self.mpol,
            self.ntor,
            self.nfp,
            stellsym=False,
        )

    def test_coil_vjp_currents_fd(self):
        """VJP w.r.t. coil currents matches FD of adjoint @ residual."""
        nphi, ntheta = self.nphi, self.ntheta
        n_res = 3 * nphi * ntheta
        rng = np.random.RandomState(99)
        adjoint = jnp.array(rng.randn(n_res))

        (d_coil_arrays,) = boozer_residual_coil_vjp(
            adjoint,
            gamma=self.gamma,
            xphi=self.xphi,
            xtheta=self.xtheta,
            coil_arrays=self.coil_arrays,
            iota=self.iota,
            G=self.G,
            weight_inv_modB=False,
        )
        # Single group → unpack the first (only) group cotangent
        dcg, dcgd, dci = d_coil_arrays[0]

        # FD on adjoint @ residual w.r.t. currents
        def f_currents(ci):
            points = self.gamma.reshape(-1, 3)
            B = biot_savart_B(points, self.coil_gammas, self.coil_gammadashs, ci)
            B = B.reshape(nphi, ntheta, 3)
            r = boozer_residual_vector(
                self.G,
                self.iota,
                B,
                self.xphi,
                self.xtheta,
                weight_inv_modB=False,
            )
            return jnp.dot(adjoint, r)

        dci_np = np.array(dci)
        eps = 1e-5
        ncoils = len(self.coil_currents)
        dci_fd = np.zeros(ncoils)
        for k in range(ncoils):
            ci_p = self.coil_currents.at[k].add(eps)
            ci_m = self.coil_currents.at[k].add(-eps)
            dci_fd[k] = (float(f_currents(ci_p)) - float(f_currents(ci_m))) / (2 * eps)

        np.testing.assert_allclose(dci_np, dci_fd, rtol=1e-4, atol=1e-10)

    def test_coil_vjp_shapes(self):
        """VJP outputs have correct shapes."""
        n_res = 3 * self.nphi * self.ntheta
        adjoint = jnp.ones(n_res)

        (d_coil_arrays,) = boozer_residual_coil_vjp(
            adjoint,
            gamma=self.gamma,
            xphi=self.xphi,
            xtheta=self.xtheta,
            coil_arrays=self.coil_arrays,
            iota=self.iota,
            G=self.G,
            weight_inv_modB=False,
        )
        # Single group → shapes match the input group
        dcg, dcgd, dci = d_coil_arrays[0]
        assert dcg.shape == self.coil_gammas.shape
        assert dcgd.shape == self.coil_gammadashs.shape
        assert dci.shape == self.coil_currents.shape


# ---------------------------------------------------------------------------
# Test: weight_inv_modB=True (reviewer finding #1)
# ---------------------------------------------------------------------------


class TestComposedWeightInvModB:
    """Validate composed derivatives with weight_inv_modB=True."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs = _make_torus_dofs(self.mpol, self.ntor, self.nfp)
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=False,
            scatter_indices=None,
            optimize_G=True,
            weight_inv_modB=True,
        )

    def test_gradient_weighted_fd(self):
        """Composed gradient with 1/|B| weighting matches FD."""
        val, grad = boozer_penalty_grad_composed(self.x, **self.kwargs)
        grad = np.array(grad)

        eps = 1e-6
        n = len(self.x)
        grad_fd = np.zeros(n)
        for k in range(n):
            x_p = self.x.at[k].add(eps)
            x_m = self.x.at[k].add(-eps)
            f_p = float(boozer_penalty_composed(x_p, **self.kwargs))
            f_m = float(boozer_penalty_composed(x_m, **self.kwargs))
            grad_fd[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-10)

    def test_jacobian_weighted_fd(self):
        """Composed residual Jacobian with 1/|B| weighting matches FD."""
        from functools import partial

        kwargs_res = {k: v for k, v in self.kwargs.items() if k not in ("optimize_G",)}
        res_fn = partial(_br._boozer_residual_vector_composed, **kwargs_res)

        r, J = boozer_residual_jacobian_composed(self.x, **kwargs_res)
        J = np.array(J)

        eps = 1e-5
        n = len(self.x)
        J_fd = np.zeros_like(J)
        for k in range(n):
            x_p = self.x.at[k].add(eps)
            x_m = self.x.at[k].add(-eps)
            r_p = np.array(res_fn(x_p))
            r_m = np.array(res_fn(x_m))
            J_fd[:, k] = (r_p - r_m) / (2 * eps)

        np.testing.assert_allclose(J, J_fd, rtol=1e-4, atol=1e-9)


# ---------------------------------------------------------------------------
# Test: stellsym=True in composed path (reviewer finding #2)
# ---------------------------------------------------------------------------


class TestComposedStellsym:
    """Validate composed gradient with stellarator symmetry."""

    mpol, ntor, nfp = 1, 0, 1
    nphi, ntheta = 4, 4

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phis = jnp.linspace(0, 1.0 / self.nfp, self.nphi, endpoint=False)
        self.thetas = jnp.linspace(0, 1.0, self.ntheta, endpoint=False)
        self.dofs, self.scatter_idx = _make_torus_dofs_stellsym(
            self.mpol,
            self.ntor,
            self.nfp,
        )
        self.coil_gammas, self.coil_gammadashs, self.coil_currents = _make_coil_data()
        self.coil_arrays = [
            (self.coil_gammas, self.coil_gammadashs, self.coil_currents)
        ]
        self.iota = 0.5
        self.G = 2.0
        self.x = jnp.concatenate([self.dofs, jnp.array([self.iota, self.G])])
        self.kwargs = dict(
            coil_arrays=self.coil_arrays,
            quadpoints_phi=self.phis,
            quadpoints_theta=self.thetas,
            mpol=self.mpol,
            ntor=self.ntor,
            nfp=self.nfp,
            stellsym=True,
            scatter_indices=self.scatter_idx,
            optimize_G=True,
            weight_inv_modB=False,
        )

    def test_gradient_stellsym_fd(self):
        """Composed gradient with stellsym matches FD."""
        val, grad = boozer_penalty_grad_composed(self.x, **self.kwargs)
        grad = np.array(grad)

        eps = 1e-6
        n = len(self.x)
        grad_fd = np.zeros(n)
        for k in range(n):
            x_p = self.x.at[k].add(eps)
            x_m = self.x.at[k].add(-eps)
            f_p = float(boozer_penalty_composed(x_p, **self.kwargs))
            f_m = float(boozer_penalty_composed(x_m, **self.kwargs))
            grad_fd[k] = (f_p - f_m) / (2 * eps)

        np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-10)

    def test_decision_vector_shorter(self):
        """Stellsym decision vector is shorter than non-stellsym."""
        n_per = (2 * self.mpol + 1) * (2 * self.ntor + 1)
        ndofs_full = 3 * n_per + 2  # + iota, G
        assert len(self.x) < ndofs_full


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
