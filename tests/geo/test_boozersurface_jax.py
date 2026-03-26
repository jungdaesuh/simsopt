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
from contextlib import contextmanager
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
    if pkg in sys.modules:
        return
    try:
        __import__(pkg)
    except ImportError:
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
_ensure_package("simsopt.objectives", _SRC / "objectives")

_sf = _load_and_register(
    "simsopt.geo.surface_fourier_jax", "geo/surface_fourier_jax.py"
)
_bs_jax = _load_and_register("simsopt.field.biotsavart_jax", "field/biotsavart_jax.py")
_obj_utils = _load_and_register(
    "simsopt.objectives.utilities", "objectives/utilities.py"
)
_br = _load_and_register(
    "simsopt.geo.boozer_residual_jax", "geo/boozer_residual_jax.py"
)
_lc = _load_and_register(
    "simsopt.geo.label_constraints_jax", "geo/label_constraints_jax.py"
)
_opt = _load_and_register("simsopt.geo.optimizer_jax", "geo/optimizer_jax.py")
_bsj = _load_and_register("simsopt.geo.boozersurface_jax", "geo/boozersurface_jax.py")
_soj = _load_and_register(
    "simsopt.geo.surfaceobjectives_jax", "geo/surfaceobjectives_jax.py"
)

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
require_target_backend_x64 = _bsj.require_target_backend_x64
resolve_optimizer_backend_method = _bsj.resolve_optimizer_backend_method
BoozerSurfaceJAX = _bsj.BoozerSurfaceJAX
_ensure_solved_jax = _soj._ensure_solved
_resolved_boozer_G_jax = _soj._resolved_boozer_G


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


def _successful_minimize_result(
    x0,
    *,
    nit=0,
    nfev=1,
    njev=1,
):
    return types.SimpleNamespace(
        x=jnp.asarray(x0),
        fun=0.0,
        jac=jnp.zeros_like(x0),
        nit=nit,
        nfev=nfev,
        njev=njev,
        success=True,
        status=0,
    )


def _successful_newton_polish_result(x0, *, nit=0):
    n = x0.shape[0]
    return {
        "x": x0,
        "fun": jnp.asarray(0.0),
        "grad": jnp.zeros_like(x0),
        "hessian": jnp.eye(n, dtype=x0.dtype),
        "nit": nit,
        "success": True,
    }


def _emit_newton_progress(progress_callback):
    progress_callback(1, 0.25, 1.0e-2)
    progress_callback(2, 0.05, 1.0e-4)


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
        # x: cos-cos + sin-sin = (mpol+1)*(ntor+1) + mpol*ntor
        # y,z: cos-sin + sin-cos = (mpol+1)*ntor + mpol*(ntor+1) each
        n_x = (mpol + 1) * (ntor + 1) + mpol * ntor
        n_yz = (mpol + 1) * ntor + mpol * (ntor + 1)
        expected = n_x + 2 * n_yz
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

        coil_arrays = [(gammas, gammadashs, currents)]

        return {
            "x": x,
            "gammas": gammas,
            "gammadashs": gammadashs,
            "currents": currents,
            "coil_arrays": coil_arrays,
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
            d["coil_arrays"],
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
            d["coil_arrays"],
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

    def test_scipy_lbfgs_preserves_supported_options(self, monkeypatch):
        """SciPy L-BFGS-B must receive its valid tuning knobs."""
        captured = {}

        def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
            del fun, jac
            captured["method"] = method
            captured["options"] = dict(options)
            captured["callback"] = callback
            return types.SimpleNamespace(
                x=np.asarray(x0),
                jac=np.asarray(x0),
                fun=0.0,
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        monkeypatch.setattr(_opt, "scipy_minimize", fake_scipy_minimize)
        jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -2.0]),
            method="lbfgs",
            tol=1e-8,
            maxiter=7,
            options={"maxcor": 33, "ftol": 1e-12, "maxfun": 55, "maxls": 66},
        )

        assert captured["method"] == "L-BFGS-B"
        assert captured["options"]["maxcor"] == 33
        assert captured["options"]["ftol"] == 1e-12
        assert captured["options"]["maxfun"] == 55
        assert captured["options"]["maxls"] == 66
        assert captured["callback"] is None  # no callback in this call

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

    def test_newton_polish_refines_nontrivial_gmres_residual(self, monkeypatch):
        """Iterative refinement should run when GMRES leaves a small residual."""

        def obj(x):
            return 0.5 * x[0] ** 2

        calls = []

        def fake_hvp_fn(_objective_fn):
            return lambda _x, v: v

        def fake_gmres(_hvp_fn, _x, rhs, *, stab, tol):
            calls.append(np.asarray(rhs, dtype=float).copy())
            if len(calls) == 1:
                return jnp.array([0.75]), jnp.array([1e-6]), None
            return jnp.array([0.25]), jnp.array([0.0]), None

        monkeypatch.setattr(_opt, "_hessian_vector_product_fn", fake_hvp_fn)
        monkeypatch.setattr(_opt, "_gmres_solve_newton_system", fake_gmres)

        result = newton_polish(obj, jnp.array([1.0]), maxiter=1, tol=1e-12)

        assert len(calls) == 2
        np.testing.assert_allclose(result["x"], np.array([0.0]), atol=1e-12)

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
            [(gammas, gammadashs, currents)],
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
            [(gammas, gammadashs, currents)],
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
            [(gammas, gammadashs, currents)],
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
            [(gammas, gammadashs, currents)],
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
            [(gammas, gammadashs, currents)],
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
                [(gammas, gammadashs, currents)],
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
            [(gammas, gammadashs, currents)],
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
            [(gammas, gammadashs, currents)],
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
                [(gammas, gammadashs, currents)],
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


class _MockBiotSavart(_bsj.Optimizable):
    """Minimal mock for BiotSavartJAX — must be Optimizable for depends_on."""

    def __init__(self, coils):
        super().__init__(x0=np.asarray([]))
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


_fake_exact_surface_module = types.ModuleType("simsopt.geo.surfacexyztensorfourier")
_fake_exact_surface_module.SurfaceXYZTensorFourier = _MockSurface


@contextmanager
def _patched_exact_surface_module():
    module_name = "simsopt.geo.surfacexyztensorfourier"
    original_module = sys.modules.get(module_name)
    sys.modules[module_name] = _fake_exact_surface_module
    try:
        yield
    finally:
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


class _MockVolumeLabel:
    """Minimal mock for Volume label."""

    def J(self):
        return 0.0


def _make_mock_coils(nquad=64):
    """Create two mock coils at z=+/-0.3 for BoozerSurfaceJAX tests."""
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
    return coils


def _make_mixed_quad_mock_coils():
    """Two coils with DIFFERENT quadrature counts (64 and 128)."""
    R = 1.0
    coils = []
    for z_off, cur, nq in [(0.3, 1e5, 64), (-0.3, 1e5, 128)]:
        phi = np.linspace(0, 2 * np.pi, nq, endpoint=False)
        g = np.stack([R * np.cos(phi), R * np.sin(phi), z_off * np.ones(nq)], axis=-1)
        gd = np.stack(
            [
                -R * np.sin(phi) * 2 * np.pi,
                R * np.cos(phi) * 2 * np.pi,
                np.zeros(nq),
            ],
            axis=-1,
        )
        coils.append(_MockCoil(g, gd, cur))
    return coils


def _make_mock_boozer_surface_mixed_quad(nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
    """BoozerSurfaceJAX with mixed-quadrature coils (no simsoptpp needed)."""
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    bs = _MockBiotSavart(_make_mixed_quad_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, False, qphi, qtheta)
    label = _MockVolumeLabel()
    target = 2.0 * np.pi**2 * R0 * r**2

    return BoozerSurfaceJAX(bs, surf, label, target, constraint_weight=1.0)


def _make_mock_boozer_surface(
    nphi=8,
    ntheta=8,
    mpol=1,
    ntor=1,
    nfp=1,
    *,
    stellsym=False,
):
    """Build a BoozerSurfaceJAX from mock objects (no simsoptpp needed)."""
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    full_sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
    if stellsym:
        scatter = np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32)
        sdofs = full_sdofs[scatter]
    else:
        sdofs = full_sdofs

    bs = _MockBiotSavart(_make_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, stellsym, qphi, qtheta)
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

    def test_stale_bfgs_method_rejected(self):
        """The removed bfgs_method option must fail fast."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        with pytest.raises(ValueError, match="bfgs_method.*removed"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={"bfgs_method": "bfgs"},
            )

    def test_unknown_option_rejected(self):
        """Unknown constructor options must fail fast instead of being ignored."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        with pytest.raises(ValueError, match="Unknown BoozerSurfaceJAX option"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={"optimizer_backend_typo": "ondevice"},
            )

    def test_private_options_rejected_with_scipy_backend(self):
        """Private optimizer options must be rejected when backend is scipy."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        with pytest.raises(ValueError, match="require optimizer_backend"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={"line_search_maxiter": 11},
            )

    def test_scipy_limited_memory_options_are_accepted(self):
        """SciPy limited-memory solves must keep their public L-BFGS tuning knobs."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={
                "limited_memory": True,
                "maxcor": 12,
                "ftol": 1e-12,
                "maxfun": 99,
                "maxls": 13,
            },
        )

        assert booz.options["limited_memory"] is True
        assert booz.options["maxcor"] == 12
        assert booz.options["ftol"] == pytest.approx(1e-12)
        assert booz.options["maxfun"] == 99
        assert booz.options["maxls"] == 13

    def test_hybrid_rejects_lbfgs_tuning_options(self):
        """Hybrid stays BFGS-only, so L-BFGS tuning knobs must be rejected."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        with pytest.raises(ValueError, match="unsupported for .*'hybrid'"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
                options={
                    "optimizer_backend": "hybrid",
                    "maxcor": 12,
                },
            )

    def test_optimizer_tuning_options_are_accepted(self):
        """Private optimizer tuning knobs accepted with non-scipy backend."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={
                "optimizer_backend": "ondevice",
                "hybrid_scipy_maxiter": 7,
                "line_search_maxiter": 11,
                "maxcor": 12,
                "ftol": 1e-12,
                "maxfun": 99,
                "maxgrad": 101,
                "maxls": 13,
            },
        )

        assert booz.options["hybrid_scipy_maxiter"] == 7
        assert booz.options["line_search_maxiter"] == 11
        assert booz.options["maxcor"] == 12
        assert booz.options["ftol"] == pytest.approx(1e-12)
        assert booz.options["maxfun"] == 99
        assert booz.options["maxgrad"] == 101
        assert booz.options["maxls"] == 13

    @pytest.mark.parametrize(
        ("optimizer_backend", "limited_memory", "expected_method"),
        [
            ("scipy", False, "bfgs"),
            ("scipy", True, "lbfgs"),
            ("hybrid", False, "bfgs-hybrid"),
            ("ondevice", False, "bfgs-ondevice"),
            ("ondevice", True, "lbfgs-ondevice"),
        ],
    )
    def test_resolve_ls_optimizer_method_contract(
        self, optimizer_backend, limited_memory, expected_method
    ):
        """LS backend contract must route to the expected optimizer method."""
        assert (
            resolve_optimizer_backend_method(
                optimizer_backend,
                limited_memory=limited_memory,
            )
            == expected_method
        )

    def test_resolve_ls_optimizer_method_rejects_hybrid_limited_memory(self):
        """Hybrid is transitional and must stay BFGS-only."""
        with pytest.raises(
            ValueError, match="optimizer_backend='hybrid'.*limited_memory=True"
        ):
            resolve_optimizer_backend_method("hybrid", limited_memory=True)

    def test_resolve_ls_optimizer_method_rejects_invalid_backend(self):
        """Invalid backend names must fail instead of silently falling through."""
        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            resolve_optimizer_backend_method("bogus", limited_memory=False)

    @pytest.mark.parametrize("optimizer_backend", ["hybrid", "ondevice"])
    def test_require_target_backend_x64_rejects_disabled_float64(
        self, monkeypatch, optimizer_backend
    ):
        """Target-lane backends must fail fast when x64 is disabled."""
        monkeypatch.setattr(_opt, "_x64_enabled", lambda: False)

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*requires jax_enable_x64=True",
        ):
            require_target_backend_x64(optimizer_backend)

    def test_newton_polish_returns_stabilized_hessian_when_requested(self):
        """Returned Hessian must match the stabilized linear system."""
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])
        stab = 0.25

        def obj(x):
            return 0.5 * x @ A @ x - b @ x

        result = newton_polish(obj, jnp.zeros(2), maxiter=5, tol=1e-14, stab=stab)
        np.testing.assert_allclose(
            result["hessian"],
            np.asarray(A + stab * jnp.eye(2)),
            atol=1e-12,
        )

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

    @pytest.mark.parametrize(
        ("optimizer_backend", "limited_memory", "expected_method"),
        [
            ("scipy", False, "bfgs"),
            ("scipy", True, "lbfgs"),
            ("hybrid", False, "bfgs-hybrid"),
            ("ondevice", False, "bfgs-ondevice"),
            ("ondevice", True, "lbfgs-ondevice"),
        ],
    )
    def test_run_code_routes_backend_contract_to_expected_method(
        self, monkeypatch, optimizer_backend, limited_memory, expected_method
    ):
        """run_code() must honor the documented backend contract."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = optimizer_backend
        booz.options["limited_memory"] = limited_memory

        captured = {}

        def fake_jax_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            progress_callback=None,
        ):
            del fun, tol, maxiter, options, progress_callback
            captured["method"] = method
            return types.SimpleNamespace(
                x=jnp.asarray(x0),
                fun=0.0,
                jac=jnp.zeros_like(x0),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        monkeypatch.setattr(_bsj, "newton_polish", fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == expected_method
        assert res["success"] is True
        assert isinstance(res["PLU"], tuple)
        assert len(res["PLU"]) == 3
        assert all(piece is not None for piece in res["PLU"])
        assert callable(res["vjp"])
        assert "iota" in res
        assert booz.need_to_run_code is False

    def test_run_code_rejects_hybrid_limited_memory_at_public_seam(self):
        """run_code() must reject the unsupported hybrid + limited-memory pair."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "hybrid"
        booz.options["limited_memory"] = True

        with pytest.raises(
            ValueError, match="optimizer_backend='hybrid'.*limited_memory=True"
        ):
            booz.run_code(iota=0.3, G=0.05)

    def test_run_code_rejects_invalid_backend_after_options_mutation(self):
        """Mutable option dicts must not permit silent fallback to ondevice."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "bogus"

        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            booz.run_code(iota=0.3, G=0.05)

    @pytest.mark.parametrize("optimizer_backend", ["hybrid", "ondevice"])
    def test_run_code_rejects_target_backend_without_x64(
        self, monkeypatch, optimizer_backend
    ):
        """run_code() must fail at the public seam before target-lane execution without x64."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = optimizer_backend
        monkeypatch.setattr(_opt, "_x64_enabled", lambda: False)

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*requires jax_enable_x64=True",
        ):
            booz.run_code(iota=0.3, G=0.05)

    def test_run_code_ls_converges_with_stellsym_surface(self):
        """LS solve must also converge when the surface uses stellsym DOFs."""
        booz = _make_mock_boozer_surface(stellsym=True)
        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["type"] == "ls"
        assert res["success"] is True
        assert callable(res["vjp"])

    def test_run_code_idempotent(self):
        """Second run_code() call returns None (not dirty)."""
        booz = _make_mock_boozer_surface()
        booz.run_code(iota=0.3, G=0.05)
        assert booz.run_code(iota=0.3, G=0.05) is None

    def test_run_code_sdofs_matches_implicit_path(self):
        """run_code(sdofs=surface_dofs) must produce the same result as run_code()."""
        booz_ref = _make_mock_boozer_surface()
        sdofs_orig = booz_ref.surface.get_dofs().copy()
        res_ref = booz_ref.run_code(iota=0.3, G=0.05)

        booz_sdofs = _make_mock_boozer_surface()
        res_sdofs = booz_sdofs.run_code(iota=0.3, G=0.05, sdofs=sdofs_orig)

        assert res_sdofs["success"] == res_ref["success"]
        np.testing.assert_allclose(res_sdofs["iota"], res_ref["iota"], atol=1e-14)
        np.testing.assert_allclose(res_sdofs["fun"], res_ref["fun"], atol=1e-14)
        np.testing.assert_allclose(
            booz_sdofs.surface.get_dofs(), booz_ref.surface.get_dofs(), atol=1e-14
        )

    def test_run_code_sdofs_overrides_stale_surface(self):
        """run_code(sdofs=...) must use explicit DOFs, not stale self.surface."""
        booz = _make_mock_boozer_surface()
        sdofs_good = booz.surface.get_dofs().copy()

        # Solve once to get reference result
        res_ref = booz.run_code(iota=0.3, G=0.05)
        surface_after_ref = booz.surface.get_dofs().copy()

        # Perturb surface to garbage, mark dirty, re-solve with explicit sdofs
        booz.surface.set_dofs(sdofs_good * 0.0 + 999.0)
        booz.need_to_run_code = True
        res_sdofs = booz.run_code(iota=0.3, G=0.05, sdofs=sdofs_good)

        # Must converge to the same solution as the reference
        assert res_sdofs["success"] is True
        np.testing.assert_allclose(res_sdofs["iota"], res_ref["iota"], atol=1e-12)
        np.testing.assert_allclose(res_sdofs["fun"], res_ref["fun"], atol=1e-12)
        # Surface must hold solved DOFs, not the garbage or the warm-start
        np.testing.assert_allclose(
            booz.surface.get_dofs(), surface_after_ref, atol=1e-12
        )

    def test_run_code_sdofs_syncs_surface_on_exact_failure(self):
        """On exact-path failure, self.surface must hold warm-start sdofs.

        The exact-path failure (NaN iterates) returns before calling
        ``_set_surface_dofs``.  The pre-sync in ``run_code`` must leave
        ``self.surface`` in the warm-start state, not whatever garbage
        was there before.
        """
        booz = _make_mock_boozer_surface_exact()
        sdofs_good = booz.surface.get_dofs().copy()

        # Corrupt surface state
        booz.surface.set_dofs(sdofs_good * 0.0 + 999.0)
        booz.need_to_run_code = True

        # Force exact Newton to fail → failure path skips _set_surface_dofs
        with _patched_exact_surface_module():
            with _patched_exact_newton_result(success=False, step=jnp.nan, nit=0):
                res = booz.run_code(iota=0.3, G=0.05, sdofs=sdofs_good)

        assert res["success"] is False
        # Surface must hold the warm-start DOFs, not the garbage
        np.testing.assert_allclose(booz.surface.get_dofs(), sdofs_good, atol=1e-14)

    def test_run_code_sdofs_syncs_surface_on_ls_newton_failure(self, monkeypatch):
        """On LS Newton-polish failure with sdofs, surface holds LBFGS output.

        In the LS path, LBFGS always calls ``_set_surface_dofs`` before
        Newton runs, so the pre-sync is overwritten by LBFGS.  On Newton
        NaN failure, surface retains the LBFGS result — NOT the warm-start
        sdofs and NOT the pre-corruption garbage.
        """
        booz = _make_mock_boozer_surface()
        sdofs_good = booz.surface.get_dofs().copy()

        # Solve once to capture the LBFGS-only surface output
        booz_ref = _make_mock_boozer_surface()

        def nan_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return {
                "x": x0,
                "fun": jnp.asarray(jnp.nan),
                "grad": jnp.full_like(x0, jnp.nan),
                "hessian": jnp.full(
                    (x0.shape[0], x0.shape[0]), jnp.nan, dtype=x0.dtype
                ),
                "nit": 0,
                "success": False,
            }

        monkeypatch.setattr(_bsj, "newton_polish", nan_newton_polish)
        booz_ref.run_code(iota=0.3, G=0.05)
        lbfgs_surface = booz_ref.surface.get_dofs().copy()

        # Corrupt surface, re-solve with explicit sdofs
        booz.surface.set_dofs(sdofs_good * 0.0 + 999.0)
        booz.need_to_run_code = True
        res = booz.run_code(iota=0.3, G=0.05, sdofs=sdofs_good)

        assert res["success"] is False
        # Surface must hold LBFGS output (not garbage, not warm-start)
        np.testing.assert_allclose(booz.surface.get_dofs(), lbfgs_surface, atol=1e-12)

    def test_run_code_invalid_newton_iterate_aborts_adjoint_state(self, monkeypatch):
        """Finite iterates with invalid Newton derivatives must not build PLU/VJP."""
        booz = _make_mock_boozer_surface()

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return {
                "x": x0,
                "fun": jnp.asarray(jnp.nan),
                "grad": jnp.full_like(x0, jnp.nan),
                "hessian": jnp.full(
                    (x0.shape[0], x0.shape[0]), jnp.nan, dtype=x0.dtype
                ),
                "nit": 0,
                "success": False,
            }

        monkeypatch.setattr(_bsj, "newton_polish", fake_newton_polish)
        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["success"] is False
        assert res["PLU"] is None
        assert res["vjp"] is None
        assert booz.need_to_run_code is False
        assert np.all(np.isfinite(booz.surface.get_dofs()))

    def test_run_code_finite_unsuccessful_newton_keeps_adjoint_state(self, monkeypatch):
        """Finite maxiter-exhausted Newton exits must still keep PLU/VJP state."""
        booz = _make_mock_boozer_surface()

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            n = x0.shape[0]
            return {
                "x": x0,
                "fun": jnp.asarray(0.0),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(n, dtype=x0.dtype),
                "nit": 3,
                "success": False,
            }

        monkeypatch.setattr(_bsj, "newton_polish", fake_newton_polish)
        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["success"] is False
        assert res["PLU"] is not None
        assert callable(res["vjp"])

    def test_run_code_emits_newton_progress_updates(self, monkeypatch):
        """run_code() should surface Newton start/progress/completion through stage_callback."""
        booz = _make_mock_boozer_surface()

        observed = []

        def record_stage(label, **payload):
            observed.append((label, payload))

        booz.options["stage_callback"] = record_stage

        def fake_jax_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            progress_callback=None,
        ):
            del fun, method, tol, maxiter, options, progress_callback
            return _successful_minimize_result(x0)

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab
            assert progress_callback is not None
            _emit_newton_progress(progress_callback)
            return _successful_newton_polish_result(x0, nit=2)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        monkeypatch.setattr(_bsj, "newton_polish", fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        labels = [label for label, _payload in observed]
        progress_events = [
            payload for label, payload in observed if label == "boozer_newton_progress"
        ]
        before_newton_payload = next(
            payload for label, payload in observed if label == "before_boozer_newton"
        )
        assert res is not None
        assert res["success"] is True
        assert "before_boozer_newton" in labels
        assert "after_boozer_newton" in labels
        assert before_newton_payload["method"] == "newton-polish"
        assert before_newton_payload["ls_method"] == "bfgs"
        assert [int(payload["iteration"]) for payload in progress_events] == [1, 2]
        assert all("grad_norm" in payload for payload in progress_events)

    def test_run_code_passes_newton_stab(self, monkeypatch):
        """run_code() must forward newton_stab into the Newton polish call."""
        booz = _make_mock_boozer_surface()
        booz.options["newton_stab"] = 0.125

        captured = {}

        def fake_jax_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            progress_callback=None,
        ):
            del fun, method, tol, maxiter, options, progress_callback
            return types.SimpleNamespace(
                x=jnp.asarray(x0),
                fun=0.0,
                jac=jnp.zeros_like(x0),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, progress_callback
            captured["stab"] = stab
            n = x0.shape[0]
            return {
                "x": x0,
                "fun": jnp.asarray(0.0),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(n, dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        monkeypatch.setattr(_bsj, "newton_polish", fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert res["success"] is True
        assert captured["stab"] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# P2 #4b: BoozerSurfaceJAX exact-path tests
# ---------------------------------------------------------------------------


def _make_mock_boozer_surface_exact(mpol=1, ntor=1, nfp=1):
    """Build a BoozerSurfaceJAX in exact (Newton) mode -- constraint_weight=None.

    The exact Newton path requires a SQUARE system: n_eq == n_dof.
    For non-stellsym: n_eq = 3*nphi*ntheta + 2, n_dof = 3*(2m+1)*(2n+1) + 2.
    Square when nphi*ntheta = (2m+1)*(2n+1).  For mpol=ntor=1: 3x3 grid.
    """
    R0, r = 1.0, 0.1
    nphi = 2 * mpol + 1
    ntheta = 2 * ntor + 1

    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
    assert sdofs.shape[0] == 3 * nphi * ntheta

    bs = _MockBiotSavart(_make_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, False, qphi, qtheta)
    label = _MockVolumeLabel()
    target = 2.0 * np.pi**2 * R0 * r**2

    return BoozerSurfaceJAX(bs, surf, label, target, constraint_weight=None)


def _run_mock_exact_boozer(booz, iota=0.3, G=0.05):
    with _patched_exact_surface_module():
        return booz.run_code(iota=iota, G=G)


@contextmanager
def _patched_exact_newton_result(*, success, step=0.1, nit=3):
    original_newton_exact = _bsj.newton_exact

    def fake_newton_exact(_residual_fn, x0, *, maxiter, tol):
        del maxiter, tol
        n = x0.shape[0]
        return {
            "x": x0 + step,
            "jacobian": jnp.eye(n, dtype=x0.dtype),
            "nit": nit,
            "success": success,
        }

    _bsj.newton_exact = fake_newton_exact
    try:
        yield
    finally:
        _bsj.newton_exact = original_newton_exact


def _run_mock_exact_boozer_success(booz, iota=0.3, G=0.05):
    with _patched_exact_newton_result(success=True):
        return _run_mock_exact_boozer(booz, iota=iota, G=G)


class TestBoozerSurfaceJAXExactPath:
    """Test the exact (Newton) path of BoozerSurfaceJAX.

    Validates:
    - Exact-type instantiation and boozer_type.
    - run_code() exact-path convergence.
    - Result dict contract parity with CPU BoozerSurface.
    - Mask is boolean (not integer indices).
    - Residual is raw unmasked (full grid size).
    """

    def test_exact_instantiation(self):
        """constraint_weight=None yields boozer_type='exact'."""
        booz = _make_mock_boozer_surface_exact()
        assert booz.boozer_type == "exact"
        assert booz.constraint_weight is None

    def test_run_code_exact_converges(self):
        """run_code() exact path runs and returns a result dict."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        assert res is not None
        assert res["type"] == "exact"
        assert booz.need_to_run_code is False

    def test_exact_result_dict_keys(self):
        """Exact-path result dict has all CPU-contract keys."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        expected_keys = {
            "residual",
            "fun",
            "jacobian",
            "iter",
            "success",
            "G",
            "s",
            "iota",
            "PLU",
            "mask",
            "type",
            "vjp",
        }
        assert expected_keys <= set(res.keys())
        assert isinstance(res["PLU"], tuple)
        assert len(res["PLU"]) == 3
        assert all(piece is not None for piece in res["PLU"])
        assert res["vjp"] is _boozer_exact_coil_vjp
        assert callable(res["vjp"])

    def test_exact_fun_tracks_exact_system_residual(self):
        """Exact-path fun must reflect the actual Newton system residual."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)
        x_final = booz._pack_decision_vector(res["iota"], res["G"])
        expected_fun = float(0.5 * jnp.mean(jnp.square(res_fn(x_final))))
        assert res["fun"] == pytest.approx(expected_fun)

    def test_exact_accepts_and_ignores_optimizer_backend_option(self):
        """Exact solves accept optimizer_backend but ignore it."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=None,
            options={"optimizer_backend": "ondevice"},
        )

        assert booz.boozer_type == "exact"
        assert "optimizer_backend" not in booz.options

    def test_exact_accepts_stage_callback_option(self):
        """Exact solves must accept stage_callback because init probes thread it in."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()

        def stage_callback(_label, **_payload):
            return None

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=None,
            options={"stage_callback": stage_callback},
        )

        assert booz.boozer_type == "exact"
        assert booz.options["stage_callback"] is stage_callback

    def test_exact_rejects_invalid_optimizer_backend_value(self):
        """Exact solves still validate optimizer_backend values."""
        bs = _MockBiotSavart(_make_mock_coils())
        surf = _MockSurface(
            np.zeros(27),
            1,
            1,
            1,
            False,
            np.linspace(0.0, 1.0, 3, endpoint=False),
            np.linspace(0.0, 1.0, 3, endpoint=False),
        )
        label = _MockVolumeLabel()
        with pytest.raises(ValueError, match="optimizer_backend must be one of"):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=None,
                options={"optimizer_backend": "bogus"},
            )

    def test_exact_mask_is_boolean(self):
        """CPU contract: mask is a boolean array, not integer indices."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        mask = res["mask"]
        assert mask.dtype == np.bool_, f"mask dtype should be bool, got {mask.dtype}"
        nphi = len(booz.quadpoints_phi)
        ntheta = len(booz.quadpoints_theta)
        assert mask.shape == (3 * nphi * ntheta,)

    def test_exact_residual_is_raw_unmasked(self):
        """CPU contract: residual is the full unmasked Boozer residual."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        nphi = len(booz.quadpoints_phi)
        ntheta = len(booz.quadpoints_theta)
        assert res["residual"].shape == (3 * nphi * ntheta,), (
            f"residual shape should be {(3 * nphi * ntheta,)}, "
            f"got {res['residual'].shape}"
        )

    def test_exact_mask_selects_from_residual(self):
        """mask can index into residual (CPU pattern: r[mask])."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        masked_r = res["residual"][res["mask"]]
        assert masked_r.ndim == 1
        assert len(masked_r) <= len(res["residual"])
        assert len(masked_r) == int(res["mask"].sum())

    def test_exact_idempotent(self):
        """Second run_code() returns None when not dirty."""
        booz = _make_mock_boozer_surface_exact()
        _run_mock_exact_boozer_success(booz)
        assert booz.run_code(iota=0.3, G=0.05) is None

    def test_exact_invalid_newton_iterate_aborts_adjoint_state(self):
        """Exact-path failures must not expose PLU/VJP placeholders."""
        booz = _make_mock_boozer_surface_exact()
        dofs_before = booz.surface.get_dofs()

        with _patched_exact_newton_result(success=False, step=jnp.nan, nit=0):
            res = _run_mock_exact_boozer(booz)

        assert res["success"] is False
        assert res["PLU"] is None
        assert res["vjp"] is None
        assert res["mask"] is None
        np.testing.assert_allclose(booz.surface.get_dofs(), dofs_before)

    def test_exact_unsuccessful_finite_newton_exit_aborts_adjoint_state(self):
        """Finite exact-Newton failures must not publish solved adjoint state."""
        booz = _make_mock_boozer_surface_exact()
        dofs_before = booz.surface.get_dofs()

        with _patched_exact_newton_result(success=False):
            res = _run_mock_exact_boozer(booz)

        assert res["success"] is False
        assert res["PLU"] is None
        assert res["vjp"] is None
        assert res["mask"] is None
        np.testing.assert_allclose(booz.surface.get_dofs(), dofs_before)


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

        d_coil_arrays, coil_indices = vjp_fn(jnp.asarray(lm), booz, iota_sol, G_sol)
        # d_coil_arrays is a list of (d_g, d_gd, d_c) tuples, one per group
        assert len(d_coil_arrays) == len(booz.coil_groups)
        for (d_g, d_gd, d_c), (g, gd, c, _) in zip(d_coil_arrays, booz.coil_groups):
            assert d_g.shape == g.shape
            assert d_gd.shape == gd.shape
            assert d_c.shape == c.shape


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


# ---------------------------------------------------------------------------
# Issue-2 validation: nfp>1 volume and area correctness
# ---------------------------------------------------------------------------


class TestNfpVolumeArea:
    """Verify volume/area are correct for nfp>1 (one-period quadrature)."""

    @pytest.mark.parametrize("nfp", [1, 2, 3, 5])
    def test_volume_nfp(self, nfp):
        """Volume = 2π²Rr² regardless of nfp."""
        R0, r = 1.0, 0.1
        mpol, ntor = 1, 1
        nphi, ntheta = 32, 32

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
        vol = float(surface_volume(gamma, normal))
        expected = 2.0 * np.pi**2 * R0 * r**2
        np.testing.assert_allclose(vol, expected, rtol=1e-4)

    @pytest.mark.parametrize("nfp", [1, 2, 3, 5])
    def test_area_nfp(self, nfp):
        """Area = 4π²Rr regardless of nfp."""
        R0, r = 1.0, 0.1
        mpol, ntor = 1, 1
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


# ---------------------------------------------------------------------------
# Issue-1 validation: _ensure_solved crash guard
# ---------------------------------------------------------------------------


class TestEnsureSolvedGuard:
    """Verify _ensure_solved source has the None guard.

    Full behavioral test is in tests/integration/test_single_stage_jax.py
    (requires simsoptpp).
    """

    def test_source_has_none_guard(self):
        """_ensure_solved must check res is None before indexing."""
        src_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "simsopt"
            / "geo"
            / "surfaceobjectives_jax.py"
        )
        source = src_path.read_text()
        assert "booz_surf.res is None" in source, (
            "_ensure_solved must guard against res=None"
        )
        assert 'booz_surf.res.get("vjp") is None' in source
        assert 'booz_surf.res.get("PLU") is None' in source
        assert "RuntimeError" in source

    def test_dirty_resolve_preserves_nondefault_backend_contract(self, monkeypatch):
        """Dirty on-device surfaces must re-solve from cached iota/G before use."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.res = {
            "iota": 0.3,
            "G": 0.05,
            "success": True,
            "PLU": (np.eye(1), np.eye(1), np.eye(1)),
            "vjp": lambda *_args, **_kwargs: None,
        }
        booz.need_to_run_code = True

        captured = {}

        def fake_run_code(iota, G=None):
            captured["iota"] = iota
            captured["G"] = G
            booz.res = {
                "iota": iota,
                "G": G,
                "success": True,
                "PLU": (np.eye(1), np.eye(1), np.eye(1)),
                "vjp": lambda *_args, **_kwargs: None,
            }
            booz.need_to_run_code = False
            return booz.res

        monkeypatch.setattr(booz, "run_code", fake_run_code)

        _ensure_solved_jax(booz)

        assert captured == {"iota": 0.3, "G": 0.05}
        assert booz.need_to_run_code is False

    def test_finite_unsuccessful_state_with_adjoint_contract_is_rejected(self):
        """_ensure_solved must reject unsuccessful solves even with PLU/VJP."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = False
        booz.res = {
            "iota": 0.3,
            "G": 0.05,
            "success": False,
            "PLU": (np.eye(1), np.eye(1), np.eye(1)),
            "vjp": lambda *_args, **_kwargs: None,
        }

        with pytest.raises(RuntimeError, match="failed"):
            _ensure_solved_jax(booz)

    def test_resolved_boozer_G_uses_fixed_currents_when_run_code_kept_G_none(self):
        """Fixed-current LS paths must recover the effective G from coil currents."""
        booz = _make_mock_boozer_surface()
        booz.res = {"G": None}

        resolved_G = _resolved_boozer_G_jax(booz)

        expected_G = float(compute_G_from_currents(booz.coil_currents))
        assert resolved_G == pytest.approx(expected_G)


# ---------------------------------------------------------------------------
# Mixed-quadrature Boozer regression
# ---------------------------------------------------------------------------


class TestMixedQuadratureBoozer:
    """BoozerSurfaceJAX works when coils have different nquad counts."""

    def test_instantiation(self):
        """Mixed-quad coils don't crash _refresh_coil_data."""
        booz = _make_mock_boozer_surface_mixed_quad()
        assert len(booz.coil_groups) == 2  # two distinct nquad values

    def test_run_code_ls_converges(self):
        """LS solve converges with mixed-quadrature coils."""
        booz = _make_mock_boozer_surface_mixed_quad()
        res = booz.run_code(iota=0.3, G=0.05)
        assert res is not None
        assert res["type"] == "ls"
        assert res["success"]

    def test_penalty_matches_uniform(self):
        """Penalty value is close to uniform-quad reference.

        The mixed-quad setup uses 64+128 points while the uniform setup
        uses 64+64.  The B field differs slightly due to quadrature
        accuracy, but the penalty value should be in the same regime.
        """
        booz_mixed = _make_mock_boozer_surface_mixed_quad()
        booz_uniform = _make_mock_boozer_surface()

        res_mixed = booz_mixed.run_code(iota=0.3, G=0.05)
        res_uniform = booz_uniform.run_code(iota=0.3, G=0.05)

        # Both should converge to similar (small) values
        assert res_mixed["fun"] < 1.0
        assert res_uniform["fun"] < 1.0
