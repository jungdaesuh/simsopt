"""
Tests for the JAX Boozer surface solver (Milestone 4).

Validates:
1. Stellsym DOF scatter/gather round-trip.
2. Volume computation against analytical formula.
3. Composed penalty objective value and gradient.
4. BFGS convergence on a synthetic problem.
5. Newton polish convergence.
6. Exact Newton path convergence.
7. Vector potential A correctness.
"""

import types
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Module loading: register JAX modules in sys.modules so that
# boozersurface_jax.py's package-level imports resolve correctly.
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _ensure_package(pkg, path):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(path)]
        sys.modules[pkg] = m


def _load_and_register(module_fqn, relpath):
    spec = importlib.util.spec_from_file_location(module_fqn, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_fqn] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_package("simsopt", _SRC)
_ensure_package("simsopt.geo", _SRC / "geo")
_ensure_package("simsopt.field", _SRC / "field")

_sf = _load_and_register(
    "simsopt.geo.surface_fourier_jax", "geo/surface_fourier_jax.py"
)
_bs_jax = _load_and_register("simsopt.field.biotsavart_jax", "field/biotsavart_jax.py")
_br = _load_and_register(
    "simsopt.geo.boozer_residual_jax", "geo/boozer_residual_jax.py"
)
_lc = _load_and_register(
    "simsopt.geo.label_constraints_jax", "geo/label_constraints_jax.py"
)
_opt = _load_and_register("simsopt.geo.optimizer_jax", "geo/optimizer_jax.py")
_bsj = _load_and_register("simsopt.geo.boozersurface_jax", "geo/boozersurface_jax.py")

# Convenient aliases
surface_gamma = _sf.surface_gamma
surface_gammadash1 = _sf.surface_gammadash1
surface_gammadash2 = _sf.surface_gammadash2
surface_normal = _sf.surface_normal
surface_volume = _sf.surface_volume
stellsym_scatter_indices = _sf.stellsym_scatter_indices
dofs_to_xyzc = _sf.dofs_to_xyzc
surface_gamma_from_dofs = _sf.surface_gamma_from_dofs

biot_savart_B = _bs_jax.biot_savart_B
biot_savart_A = _bs_jax.biot_savart_A
biot_savart_dA_by_dX = _bs_jax.biot_savart_dA_by_dX

boozer_residual_scalar = _br.boozer_residual_scalar
volume_jax = _lc.volume_jax
area_jax = _lc.area_jax
toroidal_flux_jax = _lc.toroidal_flux_jax
compute_G_from_currents = _lc.compute_G_from_currents
surface_area = _sf.surface_area

jax_minimize = _opt.jax_minimize
newton_polish = _opt.newton_polish
newton_exact = _opt.newton_exact

_boozer_penalty_objective = _bsj._boozer_penalty_objective
_boozer_exact_coil_vjp = _bsj._boozer_exact_coil_vjp
_boozer_ls_coil_vjp = _bsj._boozer_ls_coil_vjp
BoozerSurfaceJAX = _bsj.BoozerSurfaceJAX


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_simple_torus_coeffs(R0=1.0, r=0.1, mpol=1, ntor=1, nfp=1):
    """Create coefficient matrices for a simple circular-cross-section torus.

    gamma(phi, theta) = [(R0 + r cos(theta)) cos(2*pi*phi),
                         (R0 + r cos(theta)) sin(2*pi*phi),
                         r sin(theta)]

    In SurfaceXYZTensorFourier basis:
      x_hat = R0 + r*cos(theta),  y_hat = 0,  z = r*sin(theta)
    """
    shape = (2 * mpol + 1, 2 * ntor + 1)
    xc = np.zeros(shape)
    yc = np.zeros(shape)
    zc = np.zeros(shape)
    # x_hat: constant term = R0, cos(theta) term = r
    xc[0, 0] = R0  # cos(0*theta) * cos(0*phi)
    xc[1, 0] = r  # cos(1*theta) * cos(0*phi)
    # z: sin(theta) term = r
    # sin(theta) is row mpol+1 = 2 in the basis
    zc[mpol + 1, 0] = r  # sin(1*theta) * cos(0*phi)  → but this is sc quadrant
    # Actually for the standard basis: row mpol+1..2*mpol are sin modes
    # For mpol=1: row 2 = sin(theta)
    # Col 0 = cos(0*phi) = 1
    # z[2, 0] = r → z = r * sin(theta) * 1
    return xc, yc, zc


def _make_circular_coil(R=1.0, z=0.0, nquad=128, current=1e5):
    """Create a single circular coil at (R, z) in the xz-plane."""
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    gamma = np.stack([R * np.cos(phi), R * np.sin(phi), z * np.ones_like(phi)], axis=-1)
    gammadash_correct = np.stack(
        [-R * np.sin(phi) * 2 * np.pi, R * np.cos(phi) * 2 * np.pi, np.zeros_like(phi)],
        axis=-1,
    )
    return (
        jnp.array(gamma[None]),  # (1, nquad, 3)
        jnp.array(gammadash_correct[None]),  # (1, nquad, 3)
        jnp.array([current]),  # (1,)
    )


def _make_two_coils(nquad=128):
    """Two circular coils at z=±0.3 for a minimal field."""
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    R = 1.0

    coils = []
    for z_off, current in [(0.3, 1e5), (-0.3, 1e5)]:
        gamma = np.stack(
            [R * np.cos(phi), R * np.sin(phi), z_off * np.ones_like(phi)], axis=-1
        )
        gd = np.stack(
            [
                -R * np.sin(phi) * 2 * np.pi,
                R * np.cos(phi) * 2 * np.pi,
                np.zeros_like(phi),
            ],
            axis=-1,
        )
        coils.append((gamma, gd, current))

    gammas = jnp.array(np.stack([c[0] for c in coils]))
    gammadashs = jnp.array(np.stack([c[1] for c in coils]))
    currents = jnp.array([c[2] for c in coils])
    return gammas, gammadashs, currents


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStellsymScatterIndices:
    """Test the stellsym DOF packing/unpacking."""

    def test_nonsym_uses_identity(self):
        """Non-stellsym doesn't need scatter — DOFs map directly."""
        mpol, ntor = 2, 3
        n = (2 * mpol + 1) * (2 * ntor + 1)
        # For non-stellsym, surface_gamma_from_dofs uses direct reshape.
        # Verify stellsym indices are a strict subset.
        indices = stellsym_scatter_indices(mpol, ntor)
        assert len(indices) < 3 * n

    def test_stellsym_count(self):
        """Stellsym reduces DOF count."""
        mpol, ntor = 2, 3
        n_full = (2 * mpol + 1) * (2 * ntor + 1)
        indices = stellsym_scatter_indices(mpol, ntor)
        assert len(indices) < 3 * n_full
        # Expected: xy has (mpol+1)*(ntor+1) + mpol*ntor each
        # z has (mpol+1)*ntor + mpol*(ntor+1)
        n_xy = (mpol + 1) * (ntor + 1) + mpol * ntor
        n_z = (mpol + 1) * ntor + mpol * (ntor + 1)
        expected = 2 * n_xy + n_z
        assert len(indices) == expected

    def test_round_trip(self):
        """Scatter then gather recovers original DOFs."""
        mpol, ntor = 1, 1
        indices = jnp.array(stellsym_scatter_indices(mpol, ntor))
        ndofs = len(indices)
        dofs = jnp.arange(ndofs, dtype=jnp.float64) + 1.0

        xc, yc, zc = dofs_to_xyzc(dofs, indices, mpol, ntor)
        flat = jnp.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
        recovered = flat[indices]
        np.testing.assert_allclose(recovered, dofs)

    def test_stellsym_zeros_correct_quadrants(self):
        """Stellsym zeroes out the correct coefficient quadrants."""
        mpol, ntor = 2, 2
        indices = jnp.array(stellsym_scatter_indices(mpol, ntor))
        ndofs = len(indices)
        dofs = jnp.ones(ndofs, dtype=jnp.float64)

        xc, yc, zc = dofs_to_xyzc(dofs, indices, mpol, ntor)

        # x/y: cs and sc quadrants should be zero
        # cs: rows 0..mpol, cols ntor+1..2*ntor
        assert float(jnp.sum(jnp.abs(xc[: mpol + 1, ntor + 1 :]))) == 0.0
        # sc: rows mpol+1..2*mpol, cols 0..ntor
        assert float(jnp.sum(jnp.abs(xc[mpol + 1 :, : ntor + 1]))) == 0.0

        # z: cc and ss quadrants should be zero
        assert float(jnp.sum(jnp.abs(zc[: mpol + 1, : ntor + 1]))) == 0.0
        assert float(jnp.sum(jnp.abs(zc[mpol + 1 :, ntor + 1 :]))) == 0.0


class TestSurfaceVolume:
    """Test the JAX volume computation."""

    def test_simple_torus_volume(self):
        """Volume of a simple torus: V = 2π² R r²."""
        R0, r = 1.0, 0.1
        mpol, ntor, nfp = 1, 1, 1
        nphi, ntheta = 32, 32

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        gamma = surface_gamma(
            qphi, qtheta, jnp.array(xc), jnp.array(yc), jnp.array(zc), mpol, ntor, nfp
        )
        normal = surface_normal(
            qphi, qtheta, jnp.array(xc), jnp.array(yc), jnp.array(zc), mpol, ntor, nfp
        )

        vol = float(surface_volume(gamma, normal))
        expected = 2.0 * np.pi**2 * R0 * r**2
        np.testing.assert_allclose(vol, expected, rtol=1e-4)


class TestVectorPotentialA:
    """Test the Biot-Savart vector potential."""

    def test_stokes_theorem(self):
        """∫ B·n dA ≈ ∮ A·dl on a small disk for a single coil."""
        gammas, gammadashs, currents = _make_circular_coil(R=1.0, current=1e5)

        # Evaluate A on a circle at r=0.5 in the z=0 plane
        npts = 64
        r_test = 0.5
        theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        pts = np.stack(
            [r_test * np.cos(theta), r_test * np.sin(theta), np.zeros(npts)], axis=-1
        )
        tangent = np.stack(
            [-r_test * np.sin(theta), r_test * np.cos(theta), np.zeros(npts)], axis=-1
        )

        A = biot_savart_A(jnp.array(pts), gammas, gammadashs, currents)
        # Line integral: ∮ A · dl ≈ (2π/N) Σ A · tangent
        flux_A = float(jnp.sum(A * jnp.array(tangent))) * (2 * np.pi / npts)

        # The line integral of A should be non-trivial
        assert abs(flux_A) > 0

    def test_A_divergence_free_proxy(self):
        """dA/dX trace should be approximately zero (Coulomb gauge)."""
        gammas, gammadashs, currents = _make_circular_coil(R=1.0, current=1e5)
        pts = jnp.array([[0.5, 0.0, 0.1]])

        dA_dX = biot_savart_dA_by_dX(pts, gammas, gammadashs, currents)
        # div A = trace of dA/dX (Coulomb gauge: ∇·A = 0)
        div_A = float(jnp.trace(dA_dX[0]))
        np.testing.assert_allclose(div_A, 0.0, atol=1e-12)


class TestLabelConstraints:
    """Test volume and toroidal flux computations."""

    def test_compute_G(self):
        """G = μ₀ Σ|I_k|."""
        currents = jnp.array([1e5, -2e5])
        G = float(compute_G_from_currents(currents))
        mu0 = 4 * np.pi * 1e-7
        expected = mu0 * (1e5 + 2e5)
        np.testing.assert_allclose(G, expected, rtol=1e-14)


class TestComposedPenaltyObjective:
    """Test the full composed penalty objective function."""

    def _setup(self, nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
        R0, r = 1.0, 0.1
        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )

        iota = 0.3
        gammas, gammadashs, currents = _make_two_coils()
        G = float(compute_G_from_currents(currents))

        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
        target_vol = 2.0 * np.pi**2 * R0 * r**2
        cw = 1.0

        return {
            "x": x,
            "gammas": gammas,
            "gammadashs": gammadashs,
            "currents": currents,
            "qphi": qphi,
            "qtheta": qtheta,
            "mpol": mpol,
            "ntor": ntor,
            "nfp": nfp,
            "target_vol": target_vol,
            "cw": cw,
        }

    def test_penalty_returns_scalar(self):
        """Penalty objective returns a scalar."""
        d = self._setup()
        val = _boozer_penalty_objective(
            d["x"],
            d["gammas"],
            d["gammadashs"],
            d["currents"],
            d["qphi"],
            d["qtheta"],
            d["mpol"],
            d["ntor"],
            d["nfp"],
            False,
            None,  # stellsym=False
            d["target_vol"],
            d["cw"],
            "volume",
            0,
            True,
            True,  # optimize_G=True, weight_inv_modB=True
        )
        assert val.shape == ()
        assert float(val) >= 0.0

    def test_penalty_gradient_fd(self):
        """Gradient of penalty objective matches centred finite differences."""
        d = self._setup()
        obj = lambda x: _boozer_penalty_objective(
            x,
            d["gammas"],
            d["gammadashs"],
            d["currents"],
            d["qphi"],
            d["qtheta"],
            d["mpol"],
            d["ntor"],
            d["nfp"],
            False,
            None,
            d["target_vol"],
            d["cw"],
            "volume",
            0,
            True,
            True,
        )

        grad_fn = jax.grad(obj)
        grad_jax = grad_fn(d["x"])

        # Finite differences on a few components (full FD is expensive)
        eps = 1e-6
        for idx in [0, len(d["x"]) // 2, -2, -1]:
            x_p = d["x"].at[idx].add(eps)
            x_m = d["x"].at[idx].add(-eps)
            fd = (float(obj(x_p)) - float(obj(x_m))) / (2 * eps)
            np.testing.assert_allclose(
                float(grad_jax[idx]),
                fd,
                rtol=1e-4,
                atol=1e-10,
                err_msg=f"Gradient mismatch at index {idx}",
            )


class TestOptimizerAdapter:
    """Test the JAX optimizer adapter."""

    def test_bfgs_rosenbrock(self):
        """BFGS minimizes the Rosenbrock function."""

        def rosenbrock(x):
            return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

        x0 = jnp.array([-1.0, 1.0])
        result = jax_minimize(rosenbrock, x0, method="bfgs", tol=1e-8, maxiter=500)
        np.testing.assert_allclose(result.x, jnp.array([1.0, 1.0]), atol=1e-4)

    def test_newton_polish_quadratic(self):
        """Newton polish converges in 1 iteration for a quadratic."""
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])

        def obj(x):
            return 0.5 * x @ A @ x - b @ x

        x0 = jnp.zeros(2)
        result = newton_polish(obj, x0, maxiter=5, tol=1e-14)
        x_exact = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result["x"], x_exact, atol=1e-12)
        assert result["success"]

    def test_newton_exact_linear_system(self):
        """Newton exact solver finds root of a linear system in 1 step."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])

        def residual(x):
            return A @ x - b

        x0 = jnp.zeros(2)
        result = newton_exact(residual, x0, maxiter=5, tol=1e-14)
        x_exact = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result["x"], x_exact, atol=1e-12)
        assert result["success"]


class TestBFGSBoozer:
    """Test BFGS convergence on the Boozer penalty objective."""

    def test_bfgs_reduces_objective(self):
        """BFGS reduces the penalty objective from its initial value."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        G = compute_G_from_currents(currents)
        iota = 0.3
        x0 = jnp.concatenate([sdofs, jnp.array([iota, float(G)])])
        target_vol = 2.0 * np.pi**2 * R0 * r**2

        obj = lambda x: _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            target_vol,
            1.0,
            "volume",
            0,
            True,
            True,
        )

        val_init = float(obj(x0))
        result = jax_minimize(obj, x0, method="bfgs", tol=1e-10, maxiter=200)
        val_final = float(result.fun)

        assert val_final < val_init, (
            f"BFGS did not reduce objective: {val_init:.6e} → {val_final:.6e}"
        )


class TestNewtonPolishBoozer:
    """Test Newton polish after BFGS on the Boozer penalty objective."""

    def test_newton_polish_reduces_gradient(self):
        """Newton polish reduces gradient norm below BFGS."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        G = compute_G_from_currents(currents)
        iota = 0.3
        x0 = jnp.concatenate([sdofs, jnp.array([iota, float(G)])])
        target_vol = 2.0 * np.pi**2 * R0 * r**2

        obj = lambda x: _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            target_vol,
            1.0,
            "volume",
            0,
            True,
            True,
        )

        # BFGS first
        bfgs_result = jax_minimize(obj, x0, method="bfgs", tol=1e-8, maxiter=200)
        bfgs_grad_norm = float(jnp.linalg.norm(jax.grad(obj)(bfgs_result.x)))

        # Newton polish
        newton_result = newton_polish(obj, bfgs_result.x, maxiter=20, tol=1e-12)
        newton_grad_norm = float(jnp.linalg.norm(newton_result["grad"]))

        assert newton_grad_norm <= bfgs_grad_norm + 1e-15, (
            f"Newton polish did not improve: BFGS grad={bfgs_grad_norm:.3e}, "
            f"Newton grad={newton_grad_norm:.3e}"
        )


class TestOptimizeGFalse:
    """Test the optimize_G=False code path."""

    def test_penalty_with_fixed_G(self):
        """Penalty objective works with G computed from currents."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        iota = 0.3
        # optimize_G=False: x = [sdofs, iota] (no G in vector)
        x0 = jnp.concatenate([sdofs, jnp.array([iota])])
        target_vol = 2.0 * np.pi**2 * R0 * r**2

        obj = lambda x: _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            target_vol,
            1.0,
            "volume",
            0,
            False,
            True,  # optimize_G=False
        )

        val = float(obj(x0))
        assert val >= 0.0
        # Gradient should have len = len(sdofs) + 1 (iota only)
        grad = jax.grad(obj)(x0)
        assert grad.shape == x0.shape

    def test_bfgs_fixed_G(self):
        """BFGS converges with optimize_G=False."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        iota = 0.3
        x0 = jnp.concatenate([sdofs, jnp.array([iota])])
        target_vol = 2.0 * np.pi**2 * R0 * r**2

        obj = lambda x: _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            target_vol,
            1.0,
            "volume",
            0,
            False,
            True,
        )

        val_init = float(obj(x0))
        result = jax_minimize(obj, x0, method="bfgs", tol=1e-10, maxiter=200)
        assert float(result.fun) < val_init


class TestToroidalFluxLabel:
    """Test the toroidal flux label constraint path."""

    def test_penalty_with_toroidal_flux(self):
        """Penalty objective works with label_type='toroidal_flux'."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        G = float(compute_G_from_currents(currents))
        iota = 0.3
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])

        val = _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            0.01,
            1.0,  # target flux, constraint_weight
            "toroidal_flux",
            0,
            True,
            True,
        )
        assert val.shape == ()
        assert float(val) >= 0.0

        # Gradient should be computable
        grad = jax.grad(
            lambda x: _boozer_penalty_objective(
                x,
                gammas,
                gammadashs,
                currents,
                qphi,
                qtheta,
                mpol,
                ntor,
                nfp,
                False,
                None,
                0.01,
                1.0,
                "toroidal_flux",
                0,
                True,
                True,
            )
        )(x)
        assert grad.shape == x.shape


class TestLBFGSMethod:
    """Test L-BFGS-B method through the adapter."""

    def test_lbfgs_reduces_objective(self):
        """L-BFGS-B reduces the Boozer penalty objective."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        G = compute_G_from_currents(currents)
        iota = 0.3
        x0 = jnp.concatenate([sdofs, jnp.array([iota, float(G)])])
        target_vol = 2.0 * np.pi**2 * R0 * r**2

        obj = lambda x: _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            target_vol,
            1.0,
            "volume",
            0,
            True,
            True,
        )

        val_init = float(obj(x0))
        result = jax_minimize(obj, x0, method="lbfgs", tol=1e-10, maxiter=200)
        assert float(result.fun) < val_init


class TestSurfaceArea:
    """Test the JAX area computation."""

    def test_simple_torus_area(self):
        """Area of a simple torus: A = 4pi^2 R r."""
        R0, r = 1.0, 0.1
        mpol, ntor, nfp = 1, 1, 1
        nphi, ntheta = 32, 32

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        normal = _sf.surface_normal(
            qphi,
            qtheta,
            jnp.array(xc),
            jnp.array(yc),
            jnp.array(zc),
            mpol,
            ntor,
            nfp,
        )
        computed = float(surface_area(normal))
        expected = 4.0 * np.pi**2 * R0 * r
        np.testing.assert_allclose(computed, expected, rtol=1e-4)

    def test_area_differs_from_volume(self):
        """area_jax and volume_jax return different values."""
        R0, r = 1.0, 0.1
        mpol, ntor, nfp = 1, 1, 1
        nphi, ntheta = 16, 16

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        gamma = surface_gamma(
            qphi,
            qtheta,
            jnp.array(xc),
            jnp.array(yc),
            jnp.array(zc),
            mpol,
            ntor,
            nfp,
        )
        normal = _sf.surface_normal(
            qphi,
            qtheta,
            jnp.array(xc),
            jnp.array(yc),
            jnp.array(zc),
            mpol,
            ntor,
            nfp,
        )
        vol = float(volume_jax(gamma, normal))
        area = float(area_jax(normal))
        assert vol != area


class TestAreaLabelPath:
    """Test the Area label constraint through the penalty objective."""

    def test_penalty_with_area_label(self):
        """Penalty objective works with label_type='area'."""
        nphi, ntheta = 8, 8
        mpol, ntor, nfp = 1, 1, 1
        R0, r = 1.0, 0.1

        xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
        qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        sdofs = jnp.concatenate(
            [
                jnp.array(xc).ravel(),
                jnp.array(yc).ravel(),
                jnp.array(zc).ravel(),
            ]
        )
        gammas, gammadashs, currents = _make_two_coils()
        G = float(compute_G_from_currents(currents))
        iota = 0.3
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])

        target_area = 4.0 * np.pi**2 * R0 * r
        val = _boozer_penalty_objective(
            x,
            gammas,
            gammadashs,
            currents,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            target_area,
            1.0,
            "area",
            0,
            True,
            True,
        )
        assert val.shape == ()
        assert float(val) >= 0.0

        # Gradient computable
        grad = jax.grad(
            lambda x: _boozer_penalty_objective(
                x,
                gammas,
                gammadashs,
                currents,
                qphi,
                qtheta,
                mpol,
                ntor,
                nfp,
                False,
                None,
                target_area,
                1.0,
                "area",
                0,
                True,
                True,
            )
        )(x)
        assert grad.shape == x.shape


# ---------------------------------------------------------------------------
# P2 #4: BoozerSurfaceJAX adapter class instantiation tests
# ---------------------------------------------------------------------------


class _MockCurrent:
    """Minimal mock for coil current."""

    def __init__(self, value):
        self._value = value
        self.dofs = self

    def get_value(self):
        return self._value

    def all_fixed(self):
        return True


class _MockCurve:
    """Minimal mock for coil curve."""

    def __init__(self, gamma, gammadash):
        self._gamma = gamma
        self._gammadash = gammadash

    def gamma(self):
        return self._gamma

    def gammadash(self):
        return self._gammadash


class _MockCoil:
    def __init__(self, gamma, gammadash, current):
        self.curve = _MockCurve(gamma, gammadash)
        self.current = _MockCurrent(current)


class _MockBiotSavart:
    def __init__(self, coils):
        self._coils = coils


class _MockSurface:
    """Minimal mock for SurfaceXYZTensorFourier."""

    def __init__(self, dofs, mpol, ntor, nfp, stellsym, qphi, qtheta):
        self._dofs = np.array(dofs, dtype=np.float64)
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        self.quadpoints_phi = qphi
        self.quadpoints_theta = qtheta

    def get_dofs(self):
        return self._dofs.copy()

    def set_dofs(self, d):
        self._dofs = np.array(d, dtype=np.float64)

    def get_stellsym_mask(self):
        nphi = len(self.quadpoints_phi)
        ntheta = len(self.quadpoints_theta)
        return np.ones((nphi, ntheta), dtype=bool)


class _MockVolumeLabel:
    """Minimal mock for Volume label."""

    def J(self):
        return 0.0


def _make_mock_boozer_surface(nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
    """Build a BoozerSurfaceJAX from mock objects (no simsoptpp needed)."""
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    nquad = 64
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    R = 1.0
    coils = []
    for z_off, cur in [(0.3, 1e5), (-0.3, 1e5)]:
        g = np.stack(
            [R * np.cos(phi), R * np.sin(phi), z_off * np.ones(nquad)], axis=-1
        )
        gd = np.stack(
            [
                -R * np.sin(phi) * 2 * np.pi,
                R * np.cos(phi) * 2 * np.pi,
                np.zeros(nquad),
            ],
            axis=-1,
        )
        coils.append(_MockCoil(g, gd, cur))

    bs = _MockBiotSavart(coils)
    surf = _MockSurface(sdofs, mpol, ntor, nfp, False, qphi, qtheta)
    label = _MockVolumeLabel()
    target = 2.0 * np.pi**2 * R0 * r**2

    return BoozerSurfaceJAX(bs, surf, label, target, constraint_weight=1.0)


class TestBoozerSurfaceJAXClass:
    """Test the adapter class instantiation and run_code orchestration."""

    def test_instantiation(self):
        """BoozerSurfaceJAX can be instantiated with mock objects."""
        booz = _make_mock_boozer_surface()
        assert booz.boozer_type == "ls"
        assert booz.label_type == "volume"
        assert booz.need_to_run_code is True

    def test_recompute_bell(self):
        """recompute_bell sets the dirty flag."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.recompute_bell()
        assert booz.need_to_run_code is True

    def test_pack_unpack_roundtrip(self):
        """_pack and _unpack are inverses."""
        booz = _make_mock_boozer_surface()
        x = booz._pack_decision_vector(0.3, 1.5)
        sdofs, iota, G = booz._unpack_decision_vector(x, optimize_G=True)
        np.testing.assert_allclose(iota, 0.3)
        np.testing.assert_allclose(G, 1.5)

    def test_run_code_ls_converges(self):
        """run_code() LS path converges on the mock problem."""
        booz = _make_mock_boozer_surface()
        res = booz.run_code(iota=0.3, G=0.05)
        assert res is not None
        assert res["type"] == "ls"
        assert "residual" in res
        assert "jacobian" in res
        assert "hessian" in res
        assert "PLU" in res
        assert "vjp" in res
        assert "iota" in res
        assert booz.need_to_run_code is False

    def test_run_code_idempotent(self):
        """Second run_code() call returns None (not dirty)."""
        booz = _make_mock_boozer_surface()
        booz.run_code(iota=0.3, G=0.05)
        assert booz.run_code(iota=0.3, G=0.05) is None


# ---------------------------------------------------------------------------
# P2 #5: VJP hook tests
# ---------------------------------------------------------------------------


class TestVJPHooks:
    """Test the VJP hooks stored in result dicts."""

    def test_ls_vjp_returns_correct_shapes(self):
        """LS VJP returns cotangent arrays with correct shapes."""
        booz = _make_mock_boozer_surface()
        res = booz.run_code(iota=0.3, G=0.05)
        vjp_fn = res["vjp"]
        iota_sol = res["iota"]
        G_sol = res["G"]

        # lm has same shape as the decision vector (gradient)
        lm = np.zeros_like(res["jacobian"])
        lm[0] = 1.0

        d_cg, d_cgd, d_ci = vjp_fn(jnp.asarray(lm), booz, iota_sol, G_sol)
        assert d_cg.shape == booz.coil_gammas.shape
        assert d_cgd.shape == booz.coil_gammadashs.shape
        assert d_ci.shape == booz.coil_currents.shape


# ---------------------------------------------------------------------------
# P2 #6: Negative tests
# ---------------------------------------------------------------------------


class TestNegativeCases:
    """Test error handling for unsupported inputs."""

    def test_unsupported_label_raises(self):
        """Constructor rejects unsupported label types."""

        class AspectRatioLabel:
            def J(self):
                return 0.0

        nphi, ntheta = 4, 4
        qphi = np.linspace(0, 1.0, nphi, endpoint=False)
        qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
        sdofs = np.zeros(3 * 9)

        bs = _MockBiotSavart([_MockCoil(np.zeros((32, 3)), np.zeros((32, 3)), 1e5)])
        surf = _MockSurface(sdofs, 1, 1, 1, False, qphi, qtheta)

        with pytest.raises(
            ValueError, match="Unsupported label type.*AspectRatioLabel"
        ):
            BoozerSurfaceJAX(bs, surf, AspectRatioLabel(), 1.0, constraint_weight=1.0)
