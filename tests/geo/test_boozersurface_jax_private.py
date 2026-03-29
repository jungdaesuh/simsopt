"""Private optimizer runtime tests for BoozerSurfaceJAX (split from test_boozersurface_jax.py)."""

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
_bs_backend = _load_and_register(
    "simsopt.field.biotsavart_jax_backend", "field/biotsavart_jax_backend.py"
)
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

# Convenient aliases (only what private tests need)
stellsym_scatter_indices = _sf.stellsym_scatter_indices
jax_minimize = _opt.jax_minimize
PRIVATE_OPTIMIZER_JAX_VERSION = _opt.PRIVATE_OPTIMIZER_JAX_VERSION
BoozerSurfaceJAX = _bsj.BoozerSurfaceJAX


def test_solve_boozer_adjoint_enables_iterative_refinement(monkeypatch):
    recorded = {}

    def fake_forward_backward_jax(P, L, U, rhs, *, iterative_refinement=False):
        recorded["args"] = (P, L, U, rhs)
        recorded["iterative_refinement"] = iterative_refinement
        return "adjoint"

    monkeypatch.setattr(_soj, "forward_backward_jax", fake_forward_backward_jax)
    booz_surf = types.SimpleNamespace(res={"PLU": ("P", "L", "U")})

    result = _soj._solve_boozer_adjoint(booz_surf, "rhs")

    assert result == "adjoint"
    assert recorded["args"] == ("P", "L", "U", "rhs")
    assert recorded["iterative_refinement"] is True


# ---------------------------------------------------------------------------
# Marker constants
# ---------------------------------------------------------------------------
PRIVATE_OPTIMIZER_RUNTIME = pytest.mark.private_optimizer_runtime
PRIVATE_RUNTIME_REASON = (
    f"Private on-device optimizer behavior is validated on the JAX "
    f"{PRIVATE_OPTIMIZER_JAX_VERSION} runtime."
)
PRIVATE_LBFGS_RUNTIME_REASON = (
    f"lbfgs-ondevice behavior is validated on the JAX "
    f"{PRIVATE_OPTIMIZER_JAX_VERSION} runtime."
)
PRIVATE_LBFGS_BUDGET_REASON = (
    f"lbfgs-ondevice budget behavior is validated on the JAX "
    f"{PRIVATE_OPTIMIZER_JAX_VERSION} runtime."
)
PRIVATE_LIMITED_MEMORY_REASON = (
    f"On-device limited-memory solve is validated on the JAX "
    f"{PRIVATE_OPTIMIZER_JAX_VERSION} runtime."
)
REQUIRES_PRIVATE_OPTIMIZER_RUNTIME = pytest.mark.skipif(
    jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION,
    reason=PRIVATE_RUNTIME_REASON,
)
REQUIRES_PRIVATE_LBFGS_RUNTIME = pytest.mark.skipif(
    jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION,
    reason=PRIVATE_LBFGS_RUNTIME_REASON,
)
REQUIRES_PRIVATE_LBFGS_BUDGET_RUNTIME = pytest.mark.skipif(
    jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION,
    reason=PRIVATE_LBFGS_BUDGET_REASON,
)
REQUIRES_PRIVATE_LIMITED_MEMORY_RUNTIME = pytest.mark.skipif(
    jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION,
    reason=PRIVATE_LIMITED_MEMORY_REASON,
)


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


def _patch_newton_polish_runner(monkeypatch, fake_newton_polish):
    """Patch the centralized Newton-polish dispatch seam used by run_code()."""

    def fake_runner(
        self,
        method,
        obj_fn,
        x0,
        *,
        maxiter,
        tol,
        stab,
        progress_callback=None,
    ):
        del self, method
        return fake_newton_polish(
            obj_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            stab=stab,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(
        _bsj.BoozerSurfaceJAX,
        "_run_newton_polish_for_method",
        fake_runner,
    )


def _emit_sparse_progress(progress_callback):
    progress_callback(1, 3.0, 2.0)
    progress_callback(7, 2.0, 1.0)
    progress_callback(25, 1.0, 0.5)


# ---------------------------------------------------------------------------
# Mock classes
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


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestOptimizerAdapterPrivate:
    """Private optimizer runtime tests split from TestOptimizerAdapter."""

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_line_search_zoom2_reversed_bracket_does_not_fail(self):
        """The zoom2 branch must tolerate reversed brackets without spurious failure."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        result = _opt._line_search(
            quad,
            jnp.array([1.0], dtype=jnp.float64),
            jnp.array([-1.95], dtype=jnp.float64),
            old_fval=jnp.array(0.5, dtype=jnp.float64),
            gfk=jnp.array([1.0], dtype=jnp.float64),
            maxiter=20,
        )

        assert bool(result.failed) is False
        assert int(result.status) == 0
        assert float(result.f_k) < 1e-20

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_line_search_promotes_integer_inputs_to_inexact_dtype(self):
        """The private line search must preserve the old inexact-promotion contract."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        result = _opt._line_search(
            quad,
            jnp.array([1], dtype=jnp.int32),
            jnp.array([-1], dtype=jnp.int32),
            maxiter=5,
        )

        assert jnp.issubdtype(result.a_k.dtype, jnp.inexact)
        assert jnp.issubdtype(result.f_k.dtype, jnp.inexact)
        assert jnp.issubdtype(result.g_k.dtype, jnp.inexact)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_bfgs_private_solves_simple_quadratic(self):
        """Direct private BFGS should keep its simple quadratic contract."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        state = _opt._minimize_bfgs_private(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            maxiter=10,
            gtol=1e-8,
        )

        assert bool(state.converged) is True
        assert bool(state.failed) is False
        assert int(state.status) == 0
        np.testing.assert_allclose(np.asarray(state.x_k), np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(np.asarray(state.g_k), np.zeros(2), atol=1e-12)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_minimize_lbfgs_private_solves_simple_quadratic(self):
        """Direct private L-BFGS should keep its simple quadratic contract."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        state = _opt._minimize_lbfgs_private(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            maxiter=10,
            gtol=1e-8,
            maxcor=5,
        )

        assert bool(state.converged) is True
        assert bool(state.failed) is False
        assert int(state.status) == 0
        np.testing.assert_allclose(np.asarray(state.x_k), np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(np.asarray(state.g_k), np.zeros(2), atol=1e-12)

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_skips_continuation_after_scipy_success(self, monkeypatch):
        """Hybrid mode must return the SciPy prefix directly on convergence."""
        prefix = types.SimpleNamespace(
            x=jnp.array([1.0, -1.0]),
            fun=0.0,
            jac=jnp.array([0.0, 0.0]),
            nit=2,
            nfev=3,
            njev=3,
            nhev=0,
            success=True,
            status=0,
            hess_inv=np.eye(2),
        )

        monkeypatch.setattr(_opt, "_scipy_minimize", lambda *args, **kwargs: prefix)
        monkeypatch.setattr(
            _opt,
            "_minimize_bfgs_private",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "continuation must not run after successful SciPy prefix"
                )
            ),
        )

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0]),
            method="bfgs-hybrid",
            maxiter=8,
        )

        assert result is prefix

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_skips_nonfinite_prefix_state(self, monkeypatch):
        """Hybrid mode must not continue from a non-finite SciPy prefix."""
        prefix = types.SimpleNamespace(
            x=jnp.array([1.0, -1.0]),
            fun=np.nan,
            jac=jnp.array([0.0, 0.0]),
            nit=2,
            nfev=3,
            njev=3,
            nhev=0,
            success=False,
            status=1,
            hess_inv=np.eye(2),
            message="prefix failed",
        )

        monkeypatch.setattr(_opt, "_scipy_minimize", lambda *args, **kwargs: prefix)
        monkeypatch.setattr(
            _opt,
            "_minimize_bfgs_private",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("continuation must not run after non-finite prefix")
            ),
        )

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0]),
            method="bfgs-hybrid",
            maxiter=8,
        )

        assert result is prefix
        assert result.success is False
        assert "non-finite state" in result.message

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_prefix_cap_and_total_iteration_count(self, monkeypatch):
        """Hybrid mode must cap the SciPy prefix and report total nit."""
        captured = {}
        prefix = types.SimpleNamespace(
            x=jnp.array([0.25, -0.5], dtype=jnp.float64),
            fun=0.3125,
            jac=jnp.array([0.5, -1.0], dtype=jnp.float64),
            nit=3,
            nfev=4,
            njev=4,
            nhev=0,
            success=False,
            status=1,
            hess_inv=np.eye(2),
        )

        def fake_scipy_minimize(fun, x0, *, method, tol, maxiter, options):
            del fun, x0, method, tol, options
            captured["prefix_maxiter"] = maxiter
            return prefix

        def fake_minimize_bfgs_private(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            line_search_maxiter,
            initial_state,
            callback=None,
            progress_callback=None,
        ):
            del fun, x0, gtol, line_search_maxiter, callback, progress_callback
            captured["remaining_maxiter"] = maxiter
            captured["initial_k"] = int(initial_state.k)
            return _opt._BFGSResults(
                converged=jnp.array(True),
                failed=jnp.array(False),
                k=jnp.array(2),
                nfev=jnp.array(7),
                ngev=jnp.array(7),
                nhev=jnp.array(0),
                x_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                f_k=jnp.array(0.0, dtype=jnp.float64),
                g_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                H_k=jnp.eye(2, dtype=jnp.float64),
                old_old_fval=jnp.array(0.1, dtype=jnp.float64),
                status=jnp.array(0),
                line_search_status=jnp.array(0),
            )

        monkeypatch.setattr(_opt, "_scipy_minimize", fake_scipy_minimize)
        monkeypatch.setattr(_opt, "_minimize_bfgs_private", fake_minimize_bfgs_private)

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0], dtype=jnp.float64),
            method="bfgs-hybrid",
            maxiter=7,
        )

        assert captured["prefix_maxiter"] == 3
        assert captured["remaining_maxiter"] == 4
        assert captured["initial_k"] == 0
        assert result.nit == 5

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_missing_hess_inv_falls_back_to_identity(self, monkeypatch):
        """Hybrid continuation must recover when SciPy exposes no dense hess_inv."""
        prefix = types.SimpleNamespace(
            x=jnp.array([0.25, -0.5], dtype=jnp.float64),
            fun=0.3125,
            jac=jnp.array([0.5, -1.0], dtype=jnp.float64),
            nit=1,
            nfev=2,
            njev=2,
            nhev=0,
            success=False,
            status=1,
            hess_inv=None,
        )

        def fake_minimize_bfgs_private(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            line_search_maxiter,
            initial_state,
            callback=None,
            progress_callback=None,
        ):
            del fun, x0, maxiter, gtol, line_search_maxiter, callback, progress_callback
            np.testing.assert_allclose(
                np.asarray(initial_state.H_k),
                np.eye(2),
            )
            return _opt._BFGSResults(
                converged=jnp.array(True),
                failed=jnp.array(False),
                k=jnp.array(1),
                nfev=jnp.array(3),
                ngev=jnp.array(3),
                nhev=jnp.array(0),
                x_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                f_k=jnp.array(0.0, dtype=jnp.float64),
                g_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                H_k=jnp.eye(2, dtype=jnp.float64),
                old_old_fval=jnp.array(0.1, dtype=jnp.float64),
                status=jnp.array(0),
                line_search_status=jnp.array(0),
            )

        monkeypatch.setattr(_opt, "_scipy_minimize", lambda *args, **kwargs: prefix)
        monkeypatch.setattr(_opt, "_minimize_bfgs_private", fake_minimize_bfgs_private)

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0], dtype=jnp.float64),
            method="bfgs-hybrid",
            maxiter=5,
        )

        assert result.success is True

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_degenerate_hess_inv_resets_to_identity(self, monkeypatch):
        """Hybrid continuation must reject non-descent warm-start Hessians."""
        prefix = types.SimpleNamespace(
            x=jnp.array([0.25, -0.5], dtype=jnp.float64),
            fun=0.3125,
            jac=jnp.array([0.5, -1.0], dtype=jnp.float64),
            nit=1,
            nfev=2,
            njev=2,
            nhev=0,
            success=False,
            status=1,
            hess_inv=-np.eye(2),
        )

        def fake_minimize_bfgs_private(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            line_search_maxiter,
            initial_state,
            callback=None,
            progress_callback=None,
        ):
            del fun, x0, maxiter, gtol, line_search_maxiter, callback, progress_callback
            np.testing.assert_allclose(np.asarray(initial_state.H_k), np.eye(2))
            return _opt._BFGSResults(
                converged=jnp.array(True),
                failed=jnp.array(False),
                k=jnp.array(1),
                nfev=jnp.array(3),
                ngev=jnp.array(3),
                nhev=jnp.array(0),
                x_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                f_k=jnp.array(0.0, dtype=jnp.float64),
                g_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                H_k=jnp.eye(2, dtype=jnp.float64),
                old_old_fval=jnp.array(0.1, dtype=jnp.float64),
                status=jnp.array(0),
                line_search_status=jnp.array(0),
            )

        monkeypatch.setattr(_opt, "_scipy_minimize", lambda *args, **kwargs: prefix)
        monkeypatch.setattr(_opt, "_minimize_bfgs_private", fake_minimize_bfgs_private)

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0], dtype=jnp.float64),
            method="bfgs-hybrid",
            maxiter=5,
        )

        assert result.success is True

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_zero_budget_uses_scipy_prefix_only(self, monkeypatch):
        """Hybrid maxiter=0 must still take the SciPy-prefix path."""
        captured = {}
        prefix = types.SimpleNamespace(
            x=jnp.array([1.0, -1.0], dtype=jnp.float64),
            fun=1.5,
            jac=jnp.array([1.0, -2.0], dtype=jnp.float64),
            nit=0,
            nfev=1,
            njev=1,
            nhev=0,
            success=False,
            status=1,
            hess_inv=np.eye(2),
        )

        def fake_scipy_minimize(fun, x0, *, method, tol, maxiter, options):
            del fun, x0, method, tol, options
            captured["prefix_maxiter"] = maxiter
            return prefix

        monkeypatch.setattr(_opt, "_scipy_minimize", fake_scipy_minimize)
        monkeypatch.setattr(
            _opt,
            "_minimize_bfgs_private",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("continuation must not run when total budget is zero")
            ),
        )

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0], dtype=jnp.float64),
            method="bfgs-hybrid",
            maxiter=0,
        )

        assert captured["prefix_maxiter"] == 0
        assert result is prefix

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_hybrid_maxiter_one_still_uses_prefix_path(self, monkeypatch):
        """Hybrid maxiter=1 must still enter via the SciPy-prefix seam."""
        captured = {}
        prefix = types.SimpleNamespace(
            x=jnp.array([0.25, -0.5], dtype=jnp.float64),
            fun=0.3125,
            jac=jnp.array([0.5, -1.0], dtype=jnp.float64),
            nit=0,
            nfev=1,
            njev=1,
            nhev=0,
            success=False,
            status=1,
            hess_inv=np.eye(2),
        )

        def fake_scipy_minimize(fun, x0, *, method, tol, maxiter, options):
            del fun, x0, method, tol, options
            captured["prefix_maxiter"] = maxiter
            return prefix

        def fake_minimize_bfgs_private(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            line_search_maxiter,
            initial_state,
            callback=None,
            progress_callback=None,
        ):
            del fun, x0, gtol, line_search_maxiter, callback, progress_callback
            captured["continuation_maxiter"] = maxiter
            captured["initial_k"] = int(initial_state.k)
            return _opt._BFGSResults(
                converged=jnp.array(False),
                failed=jnp.array(False),
                k=jnp.array(1),
                nfev=jnp.array(2),
                ngev=jnp.array(2),
                nhev=jnp.array(0),
                x_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                f_k=jnp.array(0.0, dtype=jnp.float64),
                g_k=jnp.array([0.0, 0.0], dtype=jnp.float64),
                H_k=jnp.eye(2, dtype=jnp.float64),
                old_old_fval=jnp.array(0.1, dtype=jnp.float64),
                status=jnp.array(1),
                line_search_status=jnp.array(0),
            )

        monkeypatch.setattr(_opt, "_scipy_minimize", fake_scipy_minimize)
        monkeypatch.setattr(_opt, "_minimize_bfgs_private", fake_minimize_bfgs_private)

        result = jax_minimize(
            lambda x: jnp.sum(x**2),
            jnp.array([1.0, -1.0], dtype=jnp.float64),
            method="bfgs-hybrid",
            maxiter=1,
        )

        assert captured["prefix_maxiter"] == 0
        assert captured["continuation_maxiter"] == 1
        assert captured["initial_k"] == 0
        assert result.nit == 1

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_rejects_nonvector_x0(self):
        """bfgs-ondevice must reject non-flat decision vectors."""
        with pytest.raises(ValueError, match="flat 1-D decision vector"):
            jax_minimize(
                lambda x: jnp.sum(x**2),
                jnp.zeros((2, 2), dtype=jnp.float64),
                method="bfgs-ondevice",
                maxiter=3,
            )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_respects_zero_iteration_budget(self):
        """bfgs-ondevice must not take a step when maxiter=0."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = jax_minimize(quad, x0, method="bfgs-ondevice", maxiter=0)

        np.testing.assert_allclose(np.asarray(result.x), np.asarray(x0))
        assert result.nit == 0
        assert result.status == 1
        assert result.success is False

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_zero_gradient_converges_immediately(self):
        """bfgs-ondevice must report success at a stationary initial point."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.zeros(2, dtype=jnp.float64)
        result = jax_minimize(quad, x0, method="bfgs-ondevice", maxiter=5)

        np.testing.assert_allclose(np.asarray(result.x), np.asarray(x0))
        assert result.nit == 0
        assert result.status == 0
        assert result.success is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_maxiter_one_edge_case(self):
        """bfgs-ondevice maxiter=1 must permit exactly one capped step."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = jax_minimize(quad, x0, method="bfgs-ondevice", maxiter=1)

        assert float(result.fun) < float(quad(x0))
        assert result.nit == 1
        assert result.status == 1
        assert result.success is False

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_nan_objective_terminates(self):
        """A NaN objective encountered mid-loop must fail without extra iterations."""

        def nan_after_first_step(x):
            return jax.lax.cond(
                x[0] < 0.95,
                lambda y: jnp.asarray(jnp.nan, dtype=y.dtype),
                lambda y: 0.5 * jnp.dot(y, y),
                x,
            )

        x0 = jnp.array([1.0], dtype=jnp.float64)
        result = jax_minimize(
            nan_after_first_step,
            x0,
            method="bfgs-ondevice",
            maxiter=5,
        )

        assert result.success is False
        assert result.nit == 1
        assert np.isnan(float(result.fun))
        assert "non-finite objective or gradient" in result.message

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_inf_objective_preserves_last_finite_iterate(self):
        """An infinite objective must abort from the last finite iterate."""

        def inf_after_first_step(x):
            return jax.lax.cond(
                x[0] < 0.95,
                lambda y: jnp.asarray(jnp.inf, dtype=y.dtype),
                lambda y: 0.5 * jnp.dot(y, y),
                x,
            )

        x0 = jnp.array([1.0], dtype=jnp.float64)
        result = jax_minimize(
            inf_after_first_step,
            x0,
            method="bfgs-ondevice",
            maxiter=5,
        )

        assert result.success is False
        assert result.nit == 1
        assert float(result.fun) == pytest.approx(float(inf_after_first_step(x0)))
        assert np.all(np.isfinite(np.asarray(result.x)))
        assert np.all(np.isfinite(np.asarray(result.jac)))

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_is_deterministic(self):
        """Repeated on-device BFGS runs must return identical results."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        first = jax_minimize(quad, x0, method="bfgs-ondevice", maxiter=5)
        second = jax_minimize(quad, x0, method="bfgs-ondevice", maxiter=5)

        np.testing.assert_allclose(np.asarray(first.x), np.asarray(second.x))
        np.testing.assert_allclose(np.asarray(first.jac), np.asarray(second.jac))
        assert float(first.fun) == pytest.approx(float(second.fun))
        assert first.nit == second.nit
        assert first.status == second.status
        assert first.success == second.success


class TestLBFGSMethodPrivate:
    """Private L-BFGS runtime tests split from TestLBFGSMethod."""

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_BUDGET_RUNTIME
    def test_lbfgs_ondevice_respects_zero_iteration_budget(self):
        """lbfgs-ondevice must not take a step when maxiter=0."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = jax_minimize(quad, x0, method="lbfgs-ondevice", maxiter=0)

        np.testing.assert_allclose(np.asarray(result.x), np.asarray(x0))
        assert result.nit == 0
        assert result.status == 1
        assert result.success is False

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_reduces_objective_without_monkeypatch(self):
        """lbfgs-ondevice must reduce the objective through the real adapter."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = jax_minimize(quad, x0, method="lbfgs-ondevice", maxiter=5)

        assert result.success is True
        assert result.nit > 0
        assert float(result.fun) < float(quad(x0))
        assert np.linalg.norm(np.asarray(result.x)) < np.linalg.norm(np.asarray(x0))

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_accepts_explicit_value_and_grad(self):
        """lbfgs-ondevice must support explicit value/grad objectives."""

        def quad_value_and_grad(x):
            x = np.asarray(x, dtype=float)
            return 0.5 * float(np.dot(x, x)), x.copy()

        callback_calls = []
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = jax_minimize(
            quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            maxiter=5,
            value_and_grad=True,
            callback=lambda x: callback_calls.append(np.asarray(x, dtype=float)),
        )

        assert result.success is True
        assert result.nit > 0
        assert float(result.fun) < quad_value_and_grad(np.asarray(x0))[0]
        assert len(callback_calls) == result.nit

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_repeated_calls_are_stable(self):
        """Repeated lbfgs-ondevice runs must not accumulate divergent state."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        baseline = jax_minimize(quad, x0, method="lbfgs-ondevice", maxiter=5)

        for _ in range(4):
            current = jax_minimize(quad, x0, method="lbfgs-ondevice", maxiter=5)
            np.testing.assert_allclose(np.asarray(current.x), np.asarray(baseline.x))
            np.testing.assert_allclose(
                np.asarray(current.jac),
                np.asarray(baseline.jac),
            )
            assert float(current.fun) == pytest.approx(float(baseline.fun))
            assert current.nit == baseline.nit
            assert current.status == baseline.status
            assert current.success == baseline.success

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_ftol_zero_allows_tiny_objective_progress(self):
        """ftol=0 must still allow progress when the objective is ~1e-15."""

        def tiny_wave(x):
            return 1e-15 * (jnp.sin(1e6 * x[0]) + 2.0)

        x0 = jnp.array([1e-6], dtype=jnp.float64)
        result = jax_minimize(
            tiny_wave,
            x0,
            method="lbfgs-ondevice",
            tol=1e-12,
            maxiter=5,
        )

        assert result.success is True
        assert result.nit > 0
        assert float(result.fun) < float(tiny_wave(x0))


class TestBoozerSurfaceJAXClassPrivate:
    """Private BoozerSurfaceJAX class tests split from TestBoozerSurfaceJAXClass."""

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_limited_memory_routes_to_lbfgs(self, monkeypatch):
        """limited_memory=True must route LS solves through lbfgs-ondevice."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

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
            return _successful_minimize_result(x0)

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == "lbfgs-ondevice"
        assert res["success"] is True
        assert res["optimizer_method"] == "lbfgs-ondevice"

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_force_limited_memory_routes_to_lbfgs(self, monkeypatch):
        """The explicit Boozer LS override must route ondevice solves through lbfgs."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = False
        booz.options["force_ondevice_limited_memory"] = True

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
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == "lbfgs-ondevice"
        assert res["success"] is True
        assert res["optimizer_method"] == "lbfgs-ondevice"

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_does_not_enter_scipy_minimize(self, monkeypatch):
        """The target LS path must not fall back through _scipy_minimize()."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = False

        def forbidden_scipy_minimize(*_args, **_kwargs):
            raise AssertionError(
                "_scipy_minimize must not be called on the ondevice path"
            )

        monkeypatch.setattr(_opt, "_scipy_minimize", forbidden_scipy_minimize)

        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["success"] is True
        assert np.isfinite(res["fun"])

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_emits_sparse_progress_updates(self, monkeypatch):
        """On-device BFGS progress should surface iteration/fun/grad snapshots sparsely."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = False

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
            del fun, tol, maxiter, options
            assert method == "bfgs-ondevice"
            assert progress_callback is not None
            _emit_sparse_progress(progress_callback)
            return _successful_minimize_result(x0, nit=25, nfev=30, njev=30)

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        progress_events = [
            payload for label, payload in observed if label == "boozer_ls_progress"
        ]
        assert res is not None
        assert res["success"] is True
        assert res["optimizer_method"] == "bfgs-ondevice"
        assert [int(payload["iteration"]) for payload in progress_events] == [1, 25]
        assert all(payload["method"] == "bfgs-ondevice" for payload in progress_events)

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_limited_memory_emits_sparse_progress_updates(
        self, monkeypatch
    ):
        """On-device L-BFGS progress should surface the same sparse stage updates."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

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
            del fun, tol, maxiter, options
            assert method == "lbfgs-ondevice"
            assert progress_callback is not None
            _emit_sparse_progress(progress_callback)
            return _successful_minimize_result(x0, nit=25, nfev=30, njev=30)

        def fake_newton_polish(
            _objective_fn, x0, *, maxiter, tol, stab, progress_callback=None
        ):
            del maxiter, tol, stab, progress_callback
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        progress_events = [
            payload for label, payload in observed if label == "boozer_ls_progress"
        ]
        assert res is not None
        assert res["success"] is True
        assert res["optimizer_method"] == "lbfgs-ondevice"
        assert [int(payload["iteration"]) for payload in progress_events] == [1, 25]
        assert all(payload["method"] == "lbfgs-ondevice" for payload in progress_events)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LIMITED_MEMORY_RUNTIME
    def test_run_code_ondevice_limited_memory_converges_without_monkeypatch(self):
        """limited_memory=True must run a full on-device L-BFGS solve."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["type"] == "ls"
        assert res["success"] is True
        assert np.isfinite(res["fun"])
        assert res["PLU"] is not None
        assert callable(res["vjp"])
        assert res["optimizer_method"] == "lbfgs-ondevice"
