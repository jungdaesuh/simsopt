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

import inspect
import sys
import types
from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from conftest import enable_non_strict_jax_backend, enable_strict_jax_backend
from simsopt.field.coil import Coil, Current
from simsopt.geo._boozersurface_current_guard import (
    guard_none_G_coil_gradient_callback,
)
from simsopt.jax_core import (
    GroupedCoilSetSpec,
    grouped_coil_set_spec_from_lists,
    grouped_field_data_from_spec,
    make_current_value_spec,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier

from .boozersurface_jax_test_helpers import (
    BoozerSurfaceJAX,
    _MockBiotSavart,
    _MockCoil,
    _MockSurface,
    _MockVolumeLabel,
    _boozer_exact_coil_vjp,
    _bsj,
    _build_penalty_problem,
    _ensure_solved_jax,
    _build_upstream_boozer_penalty_case,
    _evaluate_upstream_boozer_penalty_case,
    _build_upstream_exact_surface_case,
    _make_circular_coil,
    _make_mixed_quad_mock_coils,
    _make_mock_boozer_surface,
    _make_mock_coils,
    _make_simple_torus_coeffs,
    _opt,
    _patch_newton_polish_runner,
    _simple_torus_geometry_values,
    _successful_minimize_result,
    _successful_newton_polish_result,
    biot_savart_A,
    biot_savart_B,
    biot_savart_dA_by_dX,
    compute_G_from_currents,
    dofs_to_xyzc,
    jax_minimize,
    jax_least_squares,
    newton_exact,
    newton_polish,
    require_target_backend_x64,
    resolve_least_squares_optimizer_method,
    resolve_optimizer_backend_method,
    stellsym_scatter_indices,
    UPSTREAM_BOOZER_OPTIMIZE_G,
    UPSTREAM_BOOZER_STELLSYM,
    UPSTREAM_BOOZER_SURFACE_TYPES,
)


_TORUS_GEOMETRY_RTOL = 1e-13
_ROSENBROCK_SOLUTION_ATOL = 1e-8
_STOKES_FLUX_RTOL = 1e-5
_STOKES_FLUX_ATOL = 5e-7
_STOKES_DISK_NR = 96
_STOKES_DISK_NTHETA = 192


def _disk_flux_through_circle_z0(*, radius, nr, ntheta, gammas, gammadashs, currents):
    rs = (np.arange(nr) + 0.5) * (radius / nr)
    thetas = (np.arange(ntheta) + 0.5) * (2.0 * np.pi / ntheta)
    rr, tt = np.meshgrid(rs, thetas, indexing="ij")
    points = np.stack(
        [rr * np.cos(tt), rr * np.sin(tt), np.zeros_like(rr)],
        axis=-1,
    ).reshape(-1, 3)
    B = np.asarray(
        biot_savart_B(jnp.array(points), gammas, gammadashs, currents)
    ).reshape(nr, ntheta, 3)
    area_element = (radius / nr) * (2.0 * np.pi / ntheta) * rr
    return float(np.sum(B[..., 2] * area_element))


def _emit_newton_progress(progress_callback):
    progress_callback(1, 0.25, 1.0e-2)
    progress_callback(2, 0.05, 1.0e-4)


def _patch_matrix_free_exact_linear_solver(monkeypatch, *, A):
    dense_calls = []

    def fake_jvp_fn(_residual_fn):
        return lambda _x, v: A @ v

    def fake_gmres(_jvp_fn, _x, rhs, *, tol):
        del _jvp_fn, _x, tol
        dx = jnp.linalg.solve(A, rhs)
        return dx, rhs - A @ dx, None

    def fake_materialize(_jvp_fn, _x):
        del _jvp_fn, _x
        dense_calls.append(True)
        return A

    monkeypatch.setattr(_opt, "_jacobian_vector_product_fn", fake_jvp_fn)
    monkeypatch.setattr(_opt, "_gmres_solve_exact_newton_system", fake_gmres)
    monkeypatch.setattr(_opt, "_materialize_dense_jacobian", fake_materialize)
    return dense_calls


def _patch_matrix_free_lm_solver(monkeypatch, *, A):
    dense_calls = []
    gmres_calls = []

    def fake_gmres(_flat_residual_fn, _x, grad, _pullback, *, damping, tol):
        del _flat_residual_fn, _x, _pullback, tol
        gmres_calls.append(True)
        hessian = A.T @ A + damping * jnp.eye(A.shape[1], dtype=A.dtype)
        step = jnp.linalg.solve(hessian, grad)
        return step, grad - hessian @ step, None

    def fake_materialize(flat_residual_fn, x):
        dense_calls.append(True)
        residual = flat_residual_fn(x)
        jacobian = A
        grad, hessian = _opt._least_squares_linearization_from_jacobian(
            residual,
            jacobian,
        )
        return residual, jacobian, grad, hessian

    monkeypatch.setattr(_opt, "_gmres_solve_least_squares_system", fake_gmres)
    monkeypatch.setattr(
        _opt,
        "_materialize_dense_least_squares_linearization",
        fake_materialize,
    )
    return dense_calls, gmres_calls


def _assert_linear_lm_result(result, *, A, b):
    np.testing.assert_allclose(
        result["x"],
        np.linalg.solve(np.asarray(A), np.asarray(b)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result["residual_jacobian"],
        np.asarray(A),
        atol=1e-12,
    )


def _make_structured_quadratic_problem():
    target_surface = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    target_iota = jnp.asarray(0.25, dtype=jnp.float64)

    def objective_fn(state):
        surface = jnp.asarray(state["surface"], dtype=jnp.float64)
        iota = jnp.asarray(state["iota"], dtype=jnp.float64)
        return 0.5 * (
            jnp.sum((surface - target_surface) ** 2) + (iota - target_iota) ** 2
        )

    x0 = {
        "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
        "iota": jnp.asarray(0.0, dtype=jnp.float64),
    }
    return objective_fn, x0, np.asarray(target_surface), float(target_iota)


def _assert_lu_is_not_called(message):
    raise AssertionError(message)


def _patch_counting_scipy_minimize(monkeypatch):
    state = {"jit_call_count": 0}
    original_jit = _opt.jax.jit

    def counting_jit(*args, **kwargs):
        state["jit_call_count"] += 1
        return original_jit(*args, **kwargs)

    def fake_scipy_minimize(fun, x0, jac, method, options, callback=None):
        del jac, method, options, callback
        value, grad = fun(np.asarray(x0))
        return types.SimpleNamespace(
            x=np.asarray(x0),
            jac=np.asarray(grad),
            fun=float(value),
            nit=0,
            nfev=1,
            njev=1,
            success=True,
            status=0,
        )

    monkeypatch.setattr(_opt.jax, "jit", counting_jit)
    monkeypatch.setattr(_opt, "scipy_minimize", fake_scipy_minimize)
    return state


_enable_strict_jax_backend = partial(enable_strict_jax_backend, mode="jax_gpu_parity")
_enable_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_gpu_parity",
)
_enable_fast_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_gpu_fast",
)
_NON_ONDEVICE_LS_BACKENDS = ("scipy", "hybrid")
_NON_TARGET_MINIMIZE_METHODS = ("adam", "bfgs", "lbfgs", "bfgs-hybrid")


_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN = (
    r"BoozerSurfaceJAX requires a biotsavart object that provides "
    r"coil_set_spec\(\) for explicit immutable grouped-coil state"
)
_LEGACY_CURVE_X = np.array([1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _assert_strict_jax_minimize_rejection(
    monkeypatch,
    request,
    *,
    method,
    match,
    fun,
    value_and_grad=False,
):
    _enable_strict_jax_backend(monkeypatch, request)
    with pytest.raises(RuntimeError, match=match):
        jax_minimize(
            fun,
            jnp.array([1.0]),
            method=method,
            value_and_grad=value_and_grad,
        )


def _make_legacy_coils_list_biotsavart(coils):
    class _LegacyCoilsListBiotSavart(_bsj.Optimizable):
        def __init__(self, legacy_coils):
            super().__init__(x0=np.asarray([]))
            self._coils = legacy_coils

    return _LegacyCoilsListBiotSavart(coils)


def _make_legacy_spec_capable_coils():
    curve = CurveXYZFourier(16, 1)
    curve.x = _LEGACY_CURVE_X.copy()
    return [Coil(curve, Current(1.23))]


def _make_curve_current_spec_only_legacy_coils():
    live_curve = CurveXYZFourier(16, 1)
    live_curve.x = _LEGACY_CURVE_X.copy()
    curve_spec = live_curve.to_spec()
    current_spec = make_current_value_spec(1.23)

    class _CurveSpecOnly:
        def to_spec(self):
            return curve_spec

        def gamma(self):
            raise AssertionError("gamma() should not be read")

        def gammadash(self):
            raise AssertionError("gammadash() should not be read")

    class _CurrentSpecOnly:
        def to_spec(self):
            return current_spec

        def get_value(self):
            raise AssertionError("get_value() should not be read")

    class _CurveCurrentSpecOnlyCoil:
        def __init__(self, legacy_curve, legacy_current):
            self.curve = legacy_curve
            self.current = legacy_current

    return [_CurveCurrentSpecOnlyCoil(_CurveSpecOnly(), _CurrentSpecOnly())]


def _make_mock_boozer_surface_with_free_currents():
    class _MutableCurrent:
        def __init__(self, value):
            self._value = value
            self.dofs = self

        def get_value(self):
            return self._value

        def all_fixed(self):
            return False

    booz = _make_mock_boozer_surface()
    booz.biotsavart._coils[0].current = _MutableCurrent(
        booz.biotsavart._coils[0].current.get_value()
    )
    return booz


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
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=1,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["volume"],
            geometry["expected_volume"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )


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
        flux_B = _disk_flux_through_circle_z0(
            radius=r_test,
            nr=_STOKES_DISK_NR,
            ntheta=_STOKES_DISK_NTHETA,
            gammas=gammas,
            gammadashs=gammadashs,
            currents=currents,
        )
        np.testing.assert_allclose(
            flux_A,
            flux_B,
            rtol=_STOKES_FLUX_RTOL,
            atol=_STOKES_FLUX_ATOL,
        )

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
        return _build_penalty_problem(
            nphi=nphi,
            ntheta=ntheta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
        )

    def test_penalty_returns_scalar(self):
        """Penalty objective returns a scalar."""
        d = self._setup()
        val = d["objective"](d["x"])
        assert val.shape == ()
        assert float(val) >= 0.0

    def test_penalty_gradient_fd(self):
        """Gradient of penalty objective matches centred finite differences."""
        d = self._setup()
        obj = d["objective"]

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
                rtol=1e-6,
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
        assert result.success
        np.testing.assert_allclose(
            result.x,
            jnp.array([1.0, 1.0]),
            atol=_ROSENBROCK_SOLUTION_ATOL,
        )

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

    def test_scipy_minimize_does_not_cache_unmarked_objective(self, monkeypatch):
        """Generic optimizer callables should keep the historical fresh-jit semantics."""
        state = _patch_counting_scipy_minimize(monkeypatch)

        def quad(x):
            return jnp.sum((x - 1.0) ** 2)

        x0 = jnp.array([0.0, 2.0], dtype=jnp.float64)
        _opt._scipy_minimize(
            quad,
            x0,
            method="lbfgs",
            tol=1e-8,
            maxiter=1,
            options={},
        )
        _opt._scipy_minimize(
            quad,
            x0,
            method="lbfgs",
            tol=1e-8,
            maxiter=1,
            options={},
        )

        assert state["jit_call_count"] == 2

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

    def test_newton_polish_rejects_worsening_step(self, monkeypatch):
        """A Newton step that increases the gradient norm should be rejected."""

        def obj(x):
            return 0.5 * x[0] ** 2

        def fake_hvp_fn(_objective_fn):
            return lambda _x, v: v

        def fake_gmres(_hvp_fn, _x, _rhs, *, stab, tol):
            del stab, tol
            return jnp.array([10.0]), jnp.array([0.0]), None

        monkeypatch.setattr(_opt, "_hessian_vector_product_fn", fake_hvp_fn)
        monkeypatch.setattr(_opt, "_gmres_solve_newton_system", fake_gmres)

        result = newton_polish(obj, jnp.array([1.0]), maxiter=5, tol=1e-12)

        np.testing.assert_allclose(result["x"], np.array([1.0]), atol=1e-12)
        np.testing.assert_allclose(result["grad"], np.array([1.0]), atol=1e-12)
        assert result["nit"] == 0
        assert not result["success"]

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
        np.testing.assert_allclose(result["jacobian"], np.asarray(A), atol=1e-12)
        assert result["success"]

    def test_newton_exact_materializes_dense_jacobian_once_at_final_iterate(
        self, monkeypatch
    ):
        """Exact Newton keeps the loop matrix-free and rebuilds ``J`` once at the end."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])
        dense_calls = _patch_matrix_free_exact_linear_solver(monkeypatch, A=A)
        x_exact = np.linalg.solve(np.asarray(A), np.asarray(b))

        def residual(x):
            return A @ x - b

        result = newton_exact(residual, jnp.zeros(2), maxiter=5, tol=1e-14)

        assert len(dense_calls) == 1
        np.testing.assert_allclose(result["x"], x_exact)
        np.testing.assert_allclose(result["jacobian"], np.asarray(A), atol=1e-12)
        assert result["success"]

    def test_newton_exact_traceable_materializes_dense_jacobian_once_at_final_iterate(
        self, monkeypatch
    ):
        """Traceable exact Newton rebuilds the dense Jacobian only at the final boundary."""
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]])
        b = jnp.array([5.0, 7.0])
        dense_calls = _patch_matrix_free_exact_linear_solver(monkeypatch, A=A)
        x_exact = np.linalg.solve(np.asarray(A), np.asarray(b))

        def residual(x):
            return A @ x - b

        result = _opt.newton_exact_traceable(
            residual,
            jnp.zeros(2),
            maxiter=5,
            tol=1e-14,
        )

        assert len(dense_calls) == 1
        np.testing.assert_allclose(result["x"], x_exact)
        np.testing.assert_allclose(result["jacobian"], np.asarray(A), atol=1e-12)
        assert bool(result["success"])


class TestNewtonPolishBoozer:
    """Test Newton polish after BFGS on the Boozer penalty objective."""

    def test_newton_polish_reduces_gradient(self):
        """Newton polish reduces gradient norm below BFGS."""
        case = _build_penalty_problem()
        obj = case["objective"]

        # BFGS first
        bfgs_result = jax_minimize(obj, case["x"], method="bfgs", tol=1e-8, maxiter=200)
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
        case = _build_penalty_problem(optimize_G=False)
        obj = case["objective"]

        val = float(obj(case["x"]))
        assert val >= 0.0
        # Gradient should have len = len(sdofs) + 1 (iota only)
        grad = jax.grad(obj)(case["x"])
        assert grad.shape == case["x"].shape


class TestToroidalFluxLabel:
    """Test the toroidal flux label constraint path."""

    def test_penalty_with_toroidal_flux(self):
        """Penalty objective works with label_type='toroidal_flux'."""
        case = _build_penalty_problem(label_type="toroidal_flux", targetlabel=0.01)
        val = case["objective"](case["x"])
        assert val.shape == ()
        assert float(val) >= 0.0

        # Gradient should be computable
        grad = jax.grad(case["objective"])(case["x"])
        assert grad.shape == case["x"].shape


class TestLBFGSMethod:
    """Test L-BFGS-B method through the adapter."""

    def test_lbfgs_reduces_objective(self):
        """L-BFGS-B reduces the Boozer penalty objective."""
        case = _build_penalty_problem()
        val_init = float(case["objective"](case["x"]))
        result = jax_minimize(
            case["objective"], case["x"], method="lbfgs", tol=1e-10, maxiter=200
        )
        assert float(result.fun) < val_init


class TestSurfaceArea:
    """Test the JAX area computation."""

    def test_simple_torus_area(self):
        """Area of a simple torus: A = 4pi^2 R r."""
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=1,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["area"],
            geometry["expected_area"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )


class TestAreaLabelPath:
    """Test the Area label constraint through the penalty objective."""

    def test_penalty_with_area_label(self):
        """Penalty objective works with label_type='area'."""
        case = _build_penalty_problem(label_type="area")
        val = case["objective"](case["x"])
        assert val.shape == ()
        assert float(val) >= 0.0

        # Gradient computable
        grad = jax.grad(case["objective"])(case["x"])
        assert grad.shape == case["x"].shape


# ---------------------------------------------------------------------------
# P2 #4: BoozerSurfaceJAX adapter class instantiation tests
# ---------------------------------------------------------------------------

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


def _make_spec_only_biotsavart(coils):
    class _SpecOnlyBiotSavart(_MockBiotSavart):
        def __init__(self, grouped_coils):
            super().__init__(grouped_coils)
            self._coil_spec = grouped_coil_set_spec_from_lists(
                [coil.curve.gamma() for coil in grouped_coils],
                [coil.curve.gammadash() for coil in grouped_coils],
                [coil.current.get_value() for coil in grouped_coils],
            )
            del self._coils

        def coil_set_spec(self):
            return self._coil_spec

    return _SpecOnlyBiotSavart(coils)


def _make_grouped_extractor_only_biotsavart(coils):
    class _GroupedExtractorOnlyBiotSavart(_MockBiotSavart):
        coil_set_spec = None

        def __init__(self, grouped_coils):
            super().__init__(grouped_coils)
            self._coil_spec = grouped_coil_set_spec_from_lists(
                [coil.curve.gamma() for coil in grouped_coils],
                [coil.curve.gammadash() for coil in grouped_coils],
                [coil.current.get_value() for coil in grouped_coils],
            )
            del self._coils

        def _extract_coil_data_grouped(self):
            return grouped_field_data_from_spec(self._coil_spec)

    return _GroupedExtractorOnlyBiotSavart(coils)


def _make_basic_mock_surface_and_label():
    surface = _MockSurface(
        np.zeros(27),
        1,
        1,
        1,
        False,
        np.linspace(0.0, 1.0, 3, endpoint=False),
        np.linspace(0.0, 1.0, 3, endpoint=False),
    )
    return surface, _MockVolumeLabel()


class TestBoozerSurfaceJAXClass:
    """Test the adapter class instantiation and run_code orchestration."""

    def test_instantiation(self):
        """BoozerSurfaceJAX can be instantiated with mock objects."""
        booz = _make_mock_boozer_surface()
        assert booz.boozer_type == "ls"
        assert booz.label_type == "volume"
        assert booz.need_to_run_code is True

    def test_instantiation_accepts_spec_only_biotsavart(self):
        """The grouped-coil spec path must not require a legacy ``_coils`` list."""

        coils = _make_mixed_quad_mock_coils()
        bs = _make_spec_only_biotsavart(coils)
        surf, label = _make_basic_mock_surface_and_label()
        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
        )

        assert isinstance(booz.coil_set_spec, GroupedCoilSetSpec)
        np.testing.assert_allclose(
            np.asarray(booz.coil_currents),
            np.asarray([coil.current.get_value() for coil in coils]),
        )

    @pytest.mark.parametrize(
        ("optimizer_backend", "expected_algorithm"),
        [
            ("scipy", "quasi-newton"),
            ("ondevice", "lm"),
        ],
    )
    def test_instantiation_defaults_least_squares_algorithm_from_backend(
        self,
        optimizer_backend,
        expected_algorithm,
    ):
        bs = _MockBiotSavart(_make_mock_coils())
        surf, label = _make_basic_mock_surface_and_label()

        booz = BoozerSurfaceJAX(
            bs,
            surf,
            label,
            1.0,
            constraint_weight=1.0,
            options={"optimizer_backend": optimizer_backend},
        )

        assert booz.options["least_squares_algorithm"] == expected_algorithm

    def test_instantiation_rejects_grouped_extractor_only_adapter(self):
        """BoozerSurfaceJAX now requires explicit ``coil_set_spec()`` state."""
        coils = _make_mixed_quad_mock_coils()
        bs = _make_grouped_extractor_only_biotsavart(coils)
        surf, label = _make_basic_mock_surface_and_label()

        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
            )

    def test_instantiation_rejects_hidden_coils_list_adapter(self):
        """Raw ``_coils`` compatibility extraction is no longer supported."""
        bs = _make_legacy_coils_list_biotsavart(_make_legacy_spec_capable_coils())
        surf, label = _make_basic_mock_surface_and_label()

        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            BoozerSurfaceJAX(
                bs,
                surf,
                label,
                1.0,
                constraint_weight=1.0,
            )

    def test_spec_only_biotsavart_supports_G_none_ls_path(self):
        """Spec-driven grouped-field state should work without a legacy ``_coils`` list."""
        coils = _make_mixed_quad_mock_coils()
        bs = _make_spec_only_biotsavart(coils)
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
        )

        result = booz.run_code(iota=0.2, G=None)

        assert result is not None
        assert result["type"] == "ls"

    def test_run_code_rejects_G_none_with_free_legacy_currents(self):
        booz = _make_mock_boozer_surface_with_free_currents()

        with pytest.raises(ValueError, match="fixed coil currents when G=None"):
            booz.run_code(iota=0.2, G=None)

    def test_none_G_coil_gradient_callback_rejects_free_legacy_currents(self):
        callback = lambda *_args, **_kwargs: None
        booz = _make_mock_boozer_surface_with_free_currents()
        guarded = guard_none_G_coil_gradient_callback(
            callback,
            biotsavart=booz.biotsavart,
            component="BoozerSurfaceJAX",
            coil_attrs=("_coils",),
            G_provided=False,
        )

        with pytest.raises(ValueError, match="fixed coil currents when G=None"):
            guarded(None)

    def test_none_G_coil_gradient_callback_allows_explicit_G(self):
        callback = lambda *_args, **_kwargs: None
        booz = _make_mock_boozer_surface_with_free_currents()
        guarded = guard_none_G_coil_gradient_callback(
            callback,
            biotsavart=booz.biotsavart,
            component="BoozerSurfaceJAX",
            coil_attrs=("_coils",),
            G_provided=True,
        )

        assert guarded is callback

    def test_reference_ls_reuses_cached_scipy_value_and_grad_transform(
        self,
        monkeypatch,
    ):
        """Repeated SciPy reference solves should not rebuild the JIT transform."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "scipy"
        booz.options["least_squares_algorithm"] = "quasi-newton"
        booz.options["limited_memory"] = True
        booz.options["verbose"] = False

        state = _patch_counting_scipy_minimize(monkeypatch)

        booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=-0.3,
            G=None,
            maxiter=1,
            verbose=False,
        )
        first_call_jit_count = state["jit_call_count"]
        assert first_call_jit_count > 0

        booz.recompute_bell()
        booz.minimize_boozer_penalty_constraints_LBFGS(
            iota=-0.3,
            G=None,
            maxiter=1,
            verbose=False,
        )

        assert state["jit_call_count"] == first_call_jit_count

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

    @pytest.mark.parametrize(
        (
            "optimizer_backend",
            "limited_memory",
            "least_squares_algorithm",
            "expected_method",
        ),
        [
            ("scipy", False, "quasi-newton", "bfgs"),
            ("scipy", True, "quasi-newton", "lbfgs"),
            ("ondevice", False, "quasi-newton", "bfgs-ondevice"),
            ("ondevice", False, "lm", "lm-ondevice"),
            ("scipy", False, "lm", "lm"),
        ],
    )
    def test_resolve_least_squares_optimizer_method_contract(
        self,
        optimizer_backend,
        limited_memory,
        least_squares_algorithm,
        expected_method,
    ):
        assert (
            resolve_least_squares_optimizer_method(
                optimizer_backend,
                limited_memory=limited_memory,
                least_squares_algorithm=least_squares_algorithm,
            )
            == expected_method
        )

    def test_resolve_least_squares_optimizer_method_rejects_hybrid_lm(self):
        with pytest.raises(
            ValueError,
            match="least_squares_algorithm='lm'.*optimizer_backend='hybrid'",
        ):
            resolve_least_squares_optimizer_method(
                "hybrid",
                limited_memory=False,
                least_squares_algorithm="lm",
            )

    def test_resolve_least_squares_optimizer_method_rejects_limited_memory_lm(self):
        with pytest.raises(
            ValueError,
            match="least_squares_algorithm='lm'.*limited_memory=True",
        ):
            resolve_least_squares_optimizer_method(
                "ondevice",
                limited_memory=True,
                least_squares_algorithm="lm",
            )

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
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

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

    def test_run_code_routes_lm_least_squares_contract(self, monkeypatch):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"

        captured = {}

        def fake_jax_least_squares(
            residual_fn,
            x0,
            *,
            method,
            tol,
            maxiter,
            options=None,
            callback=None,
            progress_callback=None,
        ):
            del residual_fn, tol, maxiter, options, callback, progress_callback
            captured["method"] = method
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                residual=jnp.zeros_like(flat_x0),
                residual_jacobian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                hessian=jnp.eye(flat_x0.size, dtype=flat_x0.dtype),
                damping=jnp.asarray(1.0e-3, dtype=flat_x0.dtype),
                nit=0,
                nfev=1,
                njev=1,
                status=0,
                success=True,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_least_squares", fake_jax_least_squares)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == "lm-ondevice"
        assert res["optimizer_method"] == "lm-ondevice"
        assert res["success"] is True

    def test_penalty_residual_closure_hostifies_surface_metadata(self):
        booz = _make_mock_boozer_surface(stellsym=True, mpol=2, ntor=2)
        residual_fn = booz._make_penalty_residual_with(
            True,
            booz.options["weight_inv_modB"],
            1.0,
        )
        closure_nonlocals = inspect.getclosurevars(inspect.unwrap(residual_fn)).nonlocals

        assert "self" not in closure_nonlocals
        assert not any(
            isinstance(leaf, jax.Array)
            for value in closure_nonlocals.values()
            for leaf in jax.tree_util.tree_leaves(value)
        )

    def test_run_code_uses_quasi_newton_for_fixed_G_ondevice_lm_option(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"

        captured = {}

        def forbidden_jax_least_squares(*args, **kwargs):
            raise AssertionError("fixed-G ondevice path should not enter lm-ondevice")

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
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_bsj, "jax_least_squares", forbidden_jax_least_squares)
        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=None)

        assert captured["method"] == "bfgs-ondevice"
        assert res["optimizer_method"] == "bfgs-ondevice"
        assert res["success"] is True

    @pytest.mark.parametrize("optimizer_backend", ["scipy", "hybrid"])
    def test_run_code_rejects_non_ondevice_ls_lane_in_strict_mode(
        self,
        monkeypatch,
        request,
        optimizer_backend,
    ):
        """Strict JAX mode must reject the first non-native LS seam it reaches."""
        booz = _make_mock_boozer_surface()
        _enable_strict_jax_backend(monkeypatch, request)
        booz.options["optimizer_backend"] = optimizer_backend

        with pytest.raises(
            RuntimeError,
            match="strict=True",
        ):
            booz.run_code(iota=0.3, G=0.05)

    @pytest.mark.parametrize("optimizer_backend", _NON_ONDEVICE_LS_BACKENDS)
    def test_resolve_optimizer_method_warns_on_non_ondevice_ls_lane_in_non_strict_mode(
        self,
        monkeypatch,
        request,
        optimizer_backend,
    ):
        booz = _make_mock_boozer_surface()
        _enable_non_strict_jax_backend(monkeypatch, request)
        booz.options["optimizer_backend"] = optimizer_backend

        with pytest.warns(
            RuntimeWarning,
            match=rf"optimizer_backend='{optimizer_backend}'.*legacy adapter seam",
        ):
            booz._resolve_optimizer_method()

    @pytest.mark.parametrize("optimizer_backend", _NON_ONDEVICE_LS_BACKENDS)
    def test_resolve_optimizer_method_rejects_non_target_ls_lane_in_fast_backend_mode(
        self,
        monkeypatch,
        request,
        optimizer_backend,
    ):
        booz = _make_mock_boozer_surface()
        _enable_fast_non_strict_jax_backend(monkeypatch, request)
        booz.options["optimizer_backend"] = optimizer_backend

        with pytest.raises(
            RuntimeError,
            match=rf"optimizer_backend='{optimizer_backend}'.*jax_gpu_fast.*fast/ondevice lane",
        ):
            booz._resolve_optimizer_method()

    @pytest.mark.parametrize("method", _NON_TARGET_MINIMIZE_METHODS)
    def test_jax_minimize_rejects_fallback_methods_in_strict_mode(
        self,
        monkeypatch,
        request,
        method,
    ):
        """Strict JAX mode must reject direct optimizer fallback lanes too."""
        _assert_strict_jax_minimize_rejection(
            monkeypatch,
            request,
            method=method,
            match=rf"optimizer_jax\.jax_minimize.*method='{method}'.*strict=True",
            fun=lambda x: jnp.sum(x**2),
        )

    def test_jax_minimize_rejects_hybrid_before_private_package_load_in_fast_backend_mode(
        self,
        monkeypatch,
        request,
    ):
        _enable_fast_non_strict_jax_backend(monkeypatch, request)

        def _fail_private_package_load():
            raise AssertionError(
                "fast/ondevice lane contract rejection must happen before "
                "private optimizer package loading."
            )

        monkeypatch.setattr(_opt, "_load_private_pkg", _fail_private_package_load)

        with pytest.raises(
            RuntimeError,
            match="optimizer_jax\\.jax_minimize.*method='bfgs-hybrid'.*jax_gpu_fast.*fast/ondevice lane",
        ):
            jax_minimize(
                lambda x: jnp.sum(x**2),
                jnp.array([1.0], dtype=jnp.float64),
                method="bfgs-hybrid",
            )

    @pytest.mark.parametrize(
        ("method", "maxiter", "options", "atol"),
        [
            ("adam", 800, {"step_size": 0.1}, 1e-4),
            ("bfgs", 100, None, 1e-8),
            ("lbfgs", 100, None, 1e-8),
        ],
    )
    def test_jax_minimize_reference_methods_still_work_in_fast_backend_mode(
        self,
        monkeypatch,
        request,
        method,
        maxiter,
        options,
        atol,
    ):
        _enable_fast_non_strict_jax_backend(monkeypatch, request)
        target = np.asarray([2.0, -1.0], dtype=float)

        def objective(x):
            diff = jnp.asarray(x, dtype=jnp.float64) - jnp.asarray(
                target, dtype=jnp.float64
            )
            return 0.5 * jnp.dot(diff, diff)

        result = jax_minimize(
            objective,
            jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            method=method,
            maxiter=maxiter,
            tol=1e-8,
            options=options,
        )

        assert result.success is True
        np.testing.assert_allclose(result.x, target, atol=atol)

    def test_jax_least_squares_reference_lm_still_works_in_fast_backend_mode(
        self,
        monkeypatch,
        request,
    ):
        _enable_fast_non_strict_jax_backend(monkeypatch, request)
        target = np.asarray([2.0, -1.0], dtype=float)

        def residual_fn(x):
            return jnp.asarray(x, dtype=jnp.float64) - jnp.asarray(
                target, dtype=jnp.float64
            )

        result = jax_least_squares(
            residual_fn,
            jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            method="lm",
            maxiter=25,
            tol=1e-12,
        )

        assert result.success is True
        np.testing.assert_allclose(result.x, target, atol=1e-8)

    def test_jax_least_squares_solves_simple_structured_problem(self):
        def residual_fn(state):
            return jnp.asarray(
                [
                    state["surface"][0] - 2.0,
                    state["surface"][1] + 1.0,
                    state["iota"] - 0.25,
                ],
                dtype=jnp.float64,
            )

        x0 = {
            "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.0, dtype=jnp.float64),
        }

        result = jax_least_squares(residual_fn, x0, method="lm", maxiter=25, tol=1e-12)

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], np.asarray([2.0, -1.0]))
        np.testing.assert_allclose(result.x["iota"], 0.25)
        np.testing.assert_allclose(result.jac["surface"], np.zeros(2), atol=1e-10)
        np.testing.assert_allclose(result.jac["iota"], 0.0, atol=1e-10)

    def test_jax_minimize_adam_solves_simple_structured_problem(self):
        objective_fn, x0, target_surface, target_iota = (
            _make_structured_quadratic_problem()
        )

        result = jax_minimize(
            objective_fn,
            x0,
            method="adam",
            maxiter=800,
            tol=1e-8,
            options={"step_size": 0.1},
        )

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], target_surface, atol=1e-4)
        np.testing.assert_allclose(result.x["iota"], target_iota, atol=1e-4)

    def test_jax_minimize_adam_supports_explicit_value_and_grad(self):
        target = np.asarray([2.0, -1.0], dtype=float)

        def objective_value_and_grad(x):
            x = np.asarray(x, dtype=float)
            diff = x - target
            return 0.5 * float(np.dot(diff, diff)), diff

        result = jax_minimize(
            objective_value_and_grad,
            np.asarray([5.0, 3.0], dtype=float),
            method="adam",
            maxiter=800,
            tol=1e-8,
            value_and_grad=True,
            options={"step_size": 0.1},
        )

        assert result.success is True
        np.testing.assert_allclose(result.x, target, atol=1e-4)

    def test_jax_minimize_adam_ondevice_solves_simple_structured_problem(self):
        objective_fn, x0, target_surface, target_iota = (
            _make_structured_quadratic_problem()
        )

        result = jax_minimize(
            objective_fn,
            x0,
            method="adam-ondevice",
            maxiter=800,
            tol=1e-8,
            options={"step_size": 0.1},
        )

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], target_surface, atol=1e-4)
        np.testing.assert_allclose(result.x["iota"], target_iota, atol=1e-4)

    def test_jax_least_squares_pytree_hot_path_skips_flattening_adapter(
        self,
        monkeypatch,
    ):
        def residual_fn(state):
            return jnp.asarray(
                [
                    state["surface"][0] - 2.0,
                    state["surface"][1] + 1.0,
                    state["iota"] - 0.25,
                ],
                dtype=jnp.float64,
            )

        x0 = {
            "surface": jnp.asarray([5.0, 3.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.0, dtype=jnp.float64),
        }

        monkeypatch.setattr(
            _opt,
            "_prepare_optimizer_callable_inputs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "LM least-squares hot path should not route through the "
                    "flat-vector pytree adapter."
                )
            ),
        )

        result = jax_least_squares(residual_fn, x0, method="lm", maxiter=25, tol=1e-12)

        assert result.success is True
        np.testing.assert_allclose(result.x["surface"], np.asarray([2.0, -1.0]))
        np.testing.assert_allclose(result.x["iota"], 0.25)

    def test_levenberg_marquardt_materializes_dense_linearization_once_at_final_iterate(
        self,
        monkeypatch,
    ):
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
        b = jnp.array([5.0, 7.0], dtype=jnp.float64)
        dense_calls, gmres_calls = _patch_matrix_free_lm_solver(monkeypatch, A=A)

        def residual(x):
            return A @ x - b

        result = _opt.levenberg_marquardt(residual, jnp.zeros(2), maxiter=25, tol=1e-14)

        assert gmres_calls
        assert len(dense_calls) == 1
        _assert_linear_lm_result(result, A=A, b=b)
        assert result["success"]

    def test_levenberg_marquardt_traceable_materializes_dense_linearization_once_at_final_iterate(
        self,
        monkeypatch,
    ):
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
        b = jnp.array([5.0, 7.0], dtype=jnp.float64)
        dense_calls, gmres_calls = _patch_matrix_free_lm_solver(monkeypatch, A=A)

        def residual(x):
            return A @ x - b

        result = _opt.levenberg_marquardt_traceable(
            residual,
            jnp.zeros(2),
            maxiter=25,
            tol=1e-14,
        )

        assert gmres_calls
        assert len(dense_calls) == 1
        _assert_linear_lm_result(result, A=A, b=b)
        assert bool(result["success"])

    def test_jax_minimize_allows_explicit_value_grad_ondevice_in_strict_mode(
        self,
        monkeypatch,
        request,
    ):
        """Strict JAX mode must allow the JAX-native explicit value/grad path."""

        target = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
        x0 = jnp.asarray([5.0, 3.0], dtype=jnp.float64)

        def objective_value_and_grad(x):
            diff = jnp.asarray(x, dtype=jnp.float64) - target
            return 0.5 * jnp.dot(diff, diff), diff

        _enable_strict_jax_backend(monkeypatch, request)
        result = jax_minimize(
            objective_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
        )

        assert result.success is True
        assert result.nit > 0
        np.testing.assert_allclose(result.x, np.asarray(target), atol=1e-10)

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

        _patch_newton_polish_runner(monkeypatch, nan_newton_polish)
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
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
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

        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)
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
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            n = x0.shape[0]
            return {
                "x": x0,
                "fun": jnp.asarray(0.0),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(n, dtype=x0.dtype),
                "nit": 3,
                "success": False,
            }

        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)
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
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, objective_args
            assert progress_callback is not None
            _emit_newton_progress(progress_callback)
            return _successful_newton_polish_result(x0, nit=2)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

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

    def test_run_code_ondevice_skips_newton_progress_callback(self, monkeypatch):
        """The ondevice Newton polish lane must stay callback-free inside the traced loop."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"

        observed = []
        captured = {}

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
            return _successful_minimize_result(x0)

        def fake_newton_polish_traceable(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            args=(),
        ):
            del maxiter, tol, stab, args
            captured["progress_callback"] = progress_callback
            return _successful_newton_polish_result(x0, nit=2)

        monkeypatch.setattr(_bsj, "jax_minimize", fake_jax_minimize)
        monkeypatch.setattr(
            _bsj,
            "newton_polish_traceable",
            fake_newton_polish_traceable,
        )

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
        assert captured["progress_callback"] is None
        assert "before_boozer_newton" in labels
        assert "after_boozer_newton" in labels
        assert before_newton_payload["method"] == "newton-polish"
        assert before_newton_payload["ls_method"] == "bfgs-ondevice"
        assert progress_events == []

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
            flat_x0, _ = ravel_pytree(x0)
            return types.SimpleNamespace(
                x=x0,
                fun=0.0,
                jac=jnp.zeros_like(flat_x0),
                nit=0,
                nfev=1,
                njev=1,
                success=True,
                status=0,
            )

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, progress_callback, objective_args
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
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert res["success"] is True
        assert captured["stab"] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# P2 #4b: BoozerSurfaceJAX exact-path tests
# ---------------------------------------------------------------------------


def _make_mock_boozer_surface_exact(mpol=1, ntor=1, nfp=1, stellsym=False):
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
    full_sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
    assert full_sdofs.shape[0] == 3 * nphi * ntheta

    if stellsym:
        scatter = np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32)
        sdofs = full_sdofs[scatter]
    else:
        sdofs = full_sdofs

    bs = _MockBiotSavart(_make_mock_coils())
    surf = _MockSurface(sdofs, mpol, ntor, nfp, stellsym, qphi, qtheta)
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

    def test_resolved_coil_set_spec_falls_back_to_default_for_coil_arrays(self):
        """coil_arrays-only overrides must not erase the default grouped spec."""
        booz = _make_mock_boozer_surface_exact()
        resolved = _bsj._resolved_coil_set_spec(
            booz.coil_set_spec,
            coil_arrays=booz._coil_arrays,
        )
        assert resolved is booz.coil_set_spec

    def test_exact_fun_tracks_exact_system_residual(self):
        """Exact-path fun must reflect the actual Newton system residual."""
        booz = _make_mock_boozer_surface_exact()
        res = _run_mock_exact_boozer_success(booz)
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)
        x_final = booz._pack_decision_vector(res["iota"], res["G"])
        expected_fun = float(0.5 * jnp.mean(jnp.square(res_fn(x_final))))
        assert res["fun"] == pytest.approx(expected_fun)

    def test_exact_residual_jits_with_integer_mask_indices(self):
        """The exact residual closure must trace with integer mask indices."""
        booz = _make_mock_boozer_surface_exact()
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = jax.jit(booz._make_exact_residual(mask_indices))
        x = booz._pack_decision_vector(0.3, 0.05)
        residual = res_fn(x)
        assert residual.shape == (mask_indices.shape[0] + 2,)
        assert jnp.all(jnp.isfinite(residual))

    def test_run_code_traceable_accepts_grouped_coil_spec_source(self, monkeypatch):
        """Traceable exact solves must accept ``GroupedCoilSetSpec`` directly."""
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured = {}

        def fake_newton_exact_traceable(residual_fn, x0, *, maxiter, tol, args=()):
            del maxiter, tol
            residual = residual_fn(x0, *args)
            captured["residual"] = residual
            n = x0.shape[0]
            return {
                "x": x0,
                "residual": residual,
                "jacobian": jnp.eye(n, dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "exact"
        assert bool(result["success"])
        assert "residual" in captured
        assert jnp.all(jnp.isfinite(captured["residual"]))

    def test_run_code_traceable_exact_reuses_stable_residual_callable(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured_ids = []

        def fake_newton_exact_traceable(residual_fn, x0, *, maxiter, tol, args=()):
            del maxiter, tol
            captured_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            n = x0.shape[0]
            return {
                "x": x0,
                "residual": residual,
                "jacobian": jnp.eye(n, dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert len(captured_ids) == 2
        assert captured_ids[0] == captured_ids[1]

    def test_run_code_traceable_exact_rebuilds_residual_callable_after_target_change(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []
        residuals = []

        def fake_newton_exact_traceable(residual_fn, x0, *, maxiter, tol, args=()):
            del maxiter, tol
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            residuals.append(np.asarray(residual))
            n = x0.shape[0]
            return {
                "x": x0,
                "residual": residual,
                "jacobian": jnp.eye(n, dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        booz.targetlabel = float(booz.targetlabel + 0.5)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert residual_ids[0] != residual_ids[1]
        assert not np.allclose(residuals[0], residuals[1])

    def test_run_code_traceable_ls_reuses_newton_fun_and_grad(self, monkeypatch):
        """LS traceable path must reuse Newton outputs instead of re-differentiating."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        expected_fun = jnp.asarray(3.5, dtype=jnp.float64)
        expected_grad = jnp.arange(
            sdofs.size + 2,
            dtype=jnp.float64,
        )

        def fake_minimize(_fun, x0, **_kwargs):
            return types.SimpleNamespace(x_k=x0)

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del obj_fn, maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": expected_fun,
                "grad": expected_grad,
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "nit": 2,
                "success": True,
            }

        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", fake_minimize)
        monkeypatch.setattr(
            _bsj.jax,
            "value_and_grad",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("run_code_traceable() should reuse Newton fun/grad")
            ),
        )
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "ls"
        assert bool(result["success"])
        np.testing.assert_allclose(np.asarray(result["fun"]), np.asarray(expected_fun))
        np.testing.assert_allclose(
            np.asarray(result["grad"]),
            np.asarray(expected_grad),
        )

    def test_run_code_traceable_ls_routes_lm_ondevice(self, monkeypatch):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured = {}

        def forbidden_private_minimize(*_args, **_kwargs):
            raise AssertionError("LM route should not enter private BFGS/L-BFGS")

        def fake_lm(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            callback=None,
            progress_callback=None,
            args=(),
        ):
            del maxiter, tol, callback, progress_callback
            captured["residual"] = residual_fn(x0, *args)
            return {
                "x": x0,
                "residual": captured["residual"],
                "residual_jacobian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "fun": jnp.asarray(1.25, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "damping": jnp.asarray(1.0e-3, dtype=x0.dtype),
                "nit": jnp.asarray(1, dtype=jnp.int32),
                "status": jnp.asarray(0, dtype=jnp.int32),
                "success": jnp.asarray(True),
            }

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del obj_fn, maxiter, tol, stab, progress_callback, objective_args
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_opt, "_minimize_bfgs_private", forbidden_private_minimize)
        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", forbidden_private_minimize)
        monkeypatch.setattr(_bsj, "levenberg_marquardt_traceable", fake_lm)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["optimizer_method"] == "lm-ondevice"
        assert bool(result["success"])
        assert jnp.all(jnp.isfinite(captured["residual"]))

    def test_run_code_traceable_lm_reuses_stable_residual_and_objective_callables(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []
        objective_ids = []

        def fake_lm(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            callback=None,
            progress_callback=None,
            args=(),
        ):
            del maxiter, tol, callback, progress_callback
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            return {
                "x": x0,
                "residual": residual,
                "residual_jacobian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "fun": jnp.asarray(1.25, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "damping": jnp.asarray(1.0e-3, dtype=x0.dtype),
                "nit": jnp.asarray(1, dtype=jnp.int32),
                "status": jnp.asarray(0, dtype=jnp.int32),
                "success": jnp.asarray(True),
            }

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback
            objective_ids.append(id(obj_fn))
            obj_fn(x0, *objective_args)
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_opt, "_minimize_bfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_bsj, "levenberg_marquardt_traceable", fake_lm)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert residual_ids == [residual_ids[0], residual_ids[0]]
        assert objective_ids == [objective_ids[0], objective_ids[0]]

    def test_run_code_traceable_lm_rebuilds_callables_after_target_change(
        self, monkeypatch
    ):
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["least_squares_algorithm"] = "lm"
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        residual_ids = []
        objective_ids = []
        residuals = []
        objective_values = []

        def fake_lm(
            residual_fn,
            x0,
            *,
            maxiter,
            tol,
            callback=None,
            progress_callback=None,
            args=(),
        ):
            del maxiter, tol, callback, progress_callback
            residual_ids.append(id(residual_fn))
            residual = residual_fn(x0, *args)
            residuals.append(np.asarray(residual))
            return {
                "x": x0,
                "residual": residual,
                "residual_jacobian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "fun": jnp.asarray(1.25, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "damping": jnp.asarray(1.0e-3, dtype=x0.dtype),
                "nit": jnp.asarray(1, dtype=jnp.int32),
                "status": jnp.asarray(0, dtype=jnp.int32),
                "success": jnp.asarray(True),
            }

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback
            objective_ids.append(id(obj_fn))
            objective_values.append(np.asarray(obj_fn(x0, *objective_args)))
            return _successful_newton_polish_result(x0)

        monkeypatch.setattr(_opt, "_minimize_bfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", lambda *_a, **_k: None)
        monkeypatch.setattr(_bsj, "levenberg_marquardt_traceable", fake_lm)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        first = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)
        booz.targetlabel = float(booz.targetlabel + 0.5)
        second = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert bool(first["success"])
        assert bool(second["success"])
        assert residual_ids[0] != residual_ids[1]
        assert objective_ids[0] != objective_ids[1]
        assert not np.allclose(residuals[0], residuals[1])
        assert not np.allclose(objective_values[0], objective_values[1])

    def test_run_code_traceable_ls_skips_lu_for_nonfinite_newton_result(
        self, monkeypatch
    ):
        """LS traceable failures must return the dummy PLU payload."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)

        def fake_minimize(_fun, x0, **_kwargs):
            return types.SimpleNamespace(x_k=x0)

        def fake_newton_polish(
            obj_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del obj_fn, maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": jnp.asarray(jnp.nan, dtype=x0.dtype),
                "grad": jnp.full_like(x0, jnp.nan),
                "hessian": jnp.full(
                    (x0.shape[0], x0.shape[0]), jnp.nan, dtype=x0.dtype
                ),
                "nit": 0,
                "success": False,
            }

        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", fake_minimize)
        monkeypatch.setattr(
            _bsj.jax.scipy.linalg,
            "lu",
            lambda *_args, **_kwargs: _assert_lu_is_not_called(
                "non-finite traceable LS solves must skip LU"
            ),
        )
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "ls"
        assert bool(result["success"]) is False
        assert result["plu"] is not None
        for factor in result["plu"]:
            np.testing.assert_allclose(
                np.asarray(factor),
                np.zeros((sdofs.size + 2, sdofs.size + 2)),
                atol=0.0,
            )

    def test_run_code_functional_reuses_traceable_inner_solve(self, monkeypatch):
        """run_code_functional() should just repackage run_code_traceable()."""
        booz = _make_mock_boozer_surface()
        coil_arrays = booz._coil_arrays
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)
        captured = {}

        def fake_run_code_traceable(coil_source, sdofs_arg, iota_arg, G_arg):
            captured["coil_source"] = coil_source
            captured["sdofs"] = sdofs_arg
            captured["iota"] = iota_arg
            captured["G"] = G_arg
            return {
                "x": booz._pack_decision_vector(iota_arg, G_arg, sdofs=sdofs_arg),
                "sdofs": sdofs_arg,
                "iota": iota_arg,
                "G": G_arg,
                "fun": jnp.asarray(1.25, dtype=jnp.float64),
                "grad": jnp.arange(sdofs_arg.size + 2, dtype=jnp.float64),
                "hessian": jnp.eye(sdofs_arg.size + 2, dtype=jnp.float64),
                "plu": tuple(
                    jnp.eye(sdofs_arg.size + 2, dtype=jnp.float64) for _ in range(3)
                ),
                "nit": jnp.asarray(3, dtype=jnp.int32),
                "success": jnp.asarray(True),
                "optimizer_method": "lbfgs-ondevice",
                "type": "ls",
                "weight_inv_modB": booz.options["weight_inv_modB"],
            }

        monkeypatch.setattr(booz, "run_code_traceable", fake_run_code_traceable)
        monkeypatch.setattr(
            _bsj,
            "jax_minimize",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("run_code_functional() should not rerun jax_minimize")
            ),
        )

        result = booz.run_code_functional(coil_arrays, sdofs, iota, G)

        assert captured["coil_source"] is coil_arrays
        np.testing.assert_allclose(np.asarray(captured["sdofs"]), np.asarray(sdofs))
        np.testing.assert_allclose(np.asarray(result["sdofs"]), np.asarray(sdofs))
        assert result["optimizer_method"] == "lbfgs-ondevice"
        assert result["success"] is True
        assert result["PLU"] is not None

    def test_run_code_traceable_exact_skips_lu_for_nonfinite_newton_result(
        self, monkeypatch
    ):
        """Exact traceable failures must return the dummy PLU payload."""
        booz = _make_mock_boozer_surface_exact()
        coil_set_spec = booz.coil_set_spec
        sdofs = jnp.asarray(booz.surface.get_dofs(), dtype=jnp.float64)
        iota = jnp.asarray(0.3, dtype=jnp.float64)
        G = jnp.asarray(0.05, dtype=jnp.float64)

        def fake_newton_exact_traceable(_residual_fn, x0, *, maxiter, tol, args=()):
            del maxiter, tol, args
            n = x0.shape[0]
            return {
                "x": x0,
                "residual": jnp.full((n - 1,), jnp.nan, dtype=x0.dtype),
                "jacobian": jnp.full((n, n), jnp.nan, dtype=x0.dtype),
                "nit": 0,
                "success": False,
            }

        monkeypatch.setattr(_bsj, "newton_exact_traceable", fake_newton_exact_traceable)
        monkeypatch.setattr(
            _bsj.jax.scipy.linalg,
            "lu",
            lambda *_args, **_kwargs: _assert_lu_is_not_called(
                "non-finite traceable exact solves must skip LU"
            ),
        )

        result = booz.run_code_traceable(coil_set_spec, sdofs, iota, G)

        assert result["type"] == "exact"
        assert bool(result["success"]) is False
        assert result["plu"] is not None
        for factor in result["plu"]:
            np.testing.assert_allclose(
                np.asarray(factor),
                np.zeros((sdofs.size + 2, sdofs.size + 2)),
                atol=0.0,
            )

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

    def test_newton_residual_uses_call_weight_override(self, monkeypatch):
        """Newton residual reconstruction must respect the call-time weighting flag."""
        booz = _make_mock_boozer_surface()
        captured = {}

        def fake_newton_polish(
            _objective_fn,
            x0,
            *,
            maxiter,
            tol,
            stab,
            progress_callback=None,
            objective_args=(),
        ):
            del maxiter, tol, stab, progress_callback, objective_args
            return {
                "x": x0,
                "fun": jnp.asarray(0.0, dtype=x0.dtype),
                "grad": jnp.zeros_like(x0),
                "hessian": jnp.eye(x0.shape[0], dtype=x0.dtype),
                "nit": 0,
                "success": True,
            }

        def fake_boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB):
            del G, iota, B
            captured["weight_inv_modB"] = bool(weight_inv_modB)
            nphi, ntheta = xphi.shape[:2]
            return jnp.zeros((3 * nphi * ntheta,), dtype=xtheta.dtype)

        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)
        monkeypatch.setattr(
            _bsj,
            "boozer_residual_vector",
            fake_boozer_residual_vector,
        )

        res = booz.minimize_boozer_penalty_constraints_newton(
            constraint_weight=booz.constraint_weight,
            iota=0.3,
            G=0.05,
            weight_inv_modB=False,
            verbose=False,
        )

        assert captured["weight_inv_modB"] is False
        assert res["weight_inv_modB"] is False


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

    def test_extract_grouped_coil_set_spec_rejects_legacy_coils_fallback(
        self,
    ):
        bs = _make_legacy_coils_list_biotsavart(_make_legacy_spec_capable_coils())
        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            _bsj._extract_grouped_coil_set_spec(bs)

    def test_extract_grouped_coil_set_spec_rejects_hidden_coils_even_if_spec_capable(
        self,
    ):
        coils = _make_curve_current_spec_only_legacy_coils()
        bs = _make_legacy_coils_list_biotsavart(coils)
        with pytest.raises(AttributeError, match=_EXPLICIT_COIL_SPEC_REQUIRED_PATTERN):
            _bsj._extract_grouped_coil_set_spec(bs)

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
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=nfp,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["volume"],
            geometry["expected_volume"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )

    @pytest.mark.parametrize("nfp", [1, 2, 3, 5])
    def test_area_nfp(self, nfp):
        """Area = 4π²Rr regardless of nfp."""
        geometry = _simple_torus_geometry_values(
            R0=1.0,
            r=0.1,
            mpol=1,
            ntor=1,
            nfp=nfp,
            nphi=32,
            ntheta=32,
        )
        np.testing.assert_allclose(
            geometry["area"],
            geometry["expected_area"],
            rtol=_TORUS_GEOMETRY_RTOL,
        )


# ---------------------------------------------------------------------------
# Issue-1 validation: _ensure_solved crash guard
# ---------------------------------------------------------------------------


class TestEnsureSolvedGuard:
    """Verify runtime guard behavior around cached Boozer solve state."""

    def test_dirty_unsolved_surface_without_cached_result_is_rejected(
        self, monkeypatch
    ):
        """Dirty surfaces without cached iota/G must fail before attempting re-solve."""
        booz = _make_mock_boozer_surface()
        booz.need_to_run_code = True
        booz.res = None

        called = False

        def fake_run_code(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("run_code should not be called without cached res")

        monkeypatch.setattr(booz, "run_code", fake_run_code)

        with pytest.raises(RuntimeError, match="has not been solved yet"):
            _ensure_solved_jax(booz)

        assert called is False

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


class TestMixedQuadratureBoozer:
    """BoozerSurfaceJAX works when coils have different nquad counts."""

    def test_instantiation(self):
        """Mixed-quad coils don't crash _refresh_coil_data."""
        booz = _make_mock_boozer_surface_mixed_quad()
        assert len(booz.coil_groups) == 2  # two distinct nquad values
        assert isinstance(booz.coil_set_spec, GroupedCoilSetSpec)
        assert len(booz.coil_set_spec.groups) == 2

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
        accuracy, but the solved penalty should stay close. The mock
        problem is not unique in ``iota``/``G``, so compare the scalar
        objective directly instead of the recovered parameters.
        """
        booz_mixed = _make_mock_boozer_surface_mixed_quad()
        booz_uniform = _make_mock_boozer_surface()

        res_mixed = booz_mixed.run_code(iota=0.3, G=0.05)
        res_uniform = booz_uniform.run_code(iota=0.3, G=0.05)
        penalty_fun_rel_tol = 2e-3
        penalty_fun_abs_tol = 2e-6

        assert res_mixed["success"]
        assert res_uniform["success"]

        # Observed mixed-vs-uniform gap on this mock torus is ~1.4e-3 relative.
        np.testing.assert_allclose(
            res_mixed["fun"],
            res_uniform["fun"],
            rtol=penalty_fun_rel_tol,
            atol=penalty_fun_abs_tol,
        )


# ---------------------------------------------------------------------------
# P3: Boozer exact-constraints Jacobian Taylor test
# ---------------------------------------------------------------------------


class TestBoozerExactConstraintsJacobianTaylor:
    """Taylor convergence test for the exact Boozer constraints Jacobian.

    Verifies that the Jacobian of ``_boozer_exact_residual`` (computed
    via ``jax.jacfwd``) is consistent with finite-difference
    directional derivatives.  For each epsilon the FD error must
    shrink by at least a factor of 0.55, confirming first-order
    convergence.

    Mirrors ``TestBoozerSurface.subtest_boozer_constrained_jacobian``
    from the upstream CPU test suite.
    """

    @staticmethod
    def _run_taylor_series(res_fn, x, epsilons, ratio_bound=0.55):
        """Run the multi-epsilon FD convergence loop.

        Returns the final error so callers can assert it shrank
        monotonically.
        """
        np.random.seed(1)
        r0 = res_fn(x)
        J = jax.jacfwd(res_fn)(x)
        h = jnp.array(np.random.uniform(size=x.shape) - 0.5)
        dr_exact = J @ h

        err_old = 1e9
        for eps in epsilons:
            r1 = res_fn(x + eps * h)
            dr_fd = (r1 - r0) / eps
            err = float(jnp.linalg.norm(dr_fd - dr_exact))
            assert err < err_old * ratio_bound, (
                f"FD error did not shrink: err={err:.3e}, "
                f"prev={err_old:.3e}, ratio={err / err_old:.3f}"
            )
            err_old = err
        return err_old

    def test_exact_jacobian_taylor_nonstellsym(self):
        """Taylor test for non-stellsym exact residual Jacobian."""
        booz = _make_mock_boozer_surface_exact()
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)

        iota, G = 0.3, 0.05
        x = booz._pack_decision_vector(iota, G)

        epsilons = jnp.pow(2.0, -jnp.arange(7, 20, dtype=jnp.float64))
        self._run_taylor_series(res_fn, x, epsilons)

    def test_exact_jacobian_taylor_stellsym(self):
        """Taylor test for stellsym exact residual Jacobian."""
        booz = _make_mock_boozer_surface_exact(stellsym=True)
        mask_indices = booz._compute_stellsym_mask_indices()
        res_fn = booz._make_exact_residual(mask_indices)

        iota, G = 0.3, 0.05
        x = booz._pack_decision_vector(iota, G)

        epsilons = jnp.pow(2.0, -jnp.arange(7, 20, dtype=jnp.float64))
        self._run_taylor_series(res_fn, x, epsilons)


# ---------------------------------------------------------------------------
# P18: Parametrized stellsym x optimize_G sweep for penalty gradient/BFGS
# ---------------------------------------------------------------------------

_STELLSYM_LIST = [True, False]
_OPTIMIZE_G_LIST = [True, False]
_UPSTREAM_PENALTY_VALUE_PARITY_RTOL = 1e-5
_UPSTREAM_PENALTY_VALUE_PARITY_ATOL = 5e-6
_UPSTREAM_PENALTY_GRADIENT_MAX_ABS_TOL = 1e-2
_UPSTREAM_PENALTY_GRADIENT_REL_NORM_TOL = 2e-3


def _assert_gradient_taylor_convergence(obj, x, *, label="", ratio_bound=0.55):
    """Multi-epsilon forward-FD Taylor test for ``jax.grad(obj)``."""
    np.random.seed(1)
    f0 = float(obj(x))
    grad = jax.grad(obj)(x)
    h = jnp.array(np.random.uniform(size=x.shape) - 0.5)
    Jex = float(jnp.dot(grad, h))

    err_old = 1e9
    epsilons = jnp.pow(2.0, -jnp.arange(7, 20, dtype=jnp.float64))
    for eps in epsilons:
        f1 = float(obj(x + eps * h))
        Jfd = (f1 - f0) / float(eps)
        err = abs(Jfd - Jex) / max(abs(Jex), 1e-30)
        assert err < err_old * ratio_bound, (
            f"{label}FD ratio {err / err_old:.3f} >= {ratio_bound} "
            f"at eps={float(eps):.2e}"
        )
        err_old = err


def _assert_upstream_penalty_parity(parity):
    """Assert the calibrated CPU/JAX tensor penalty parity contract.

    We intentionally keep the gradient gate as:
    1. a hard per-component max-abs cap, and
    2. a global relative-norm cap.

    A plain ``assert_allclose(rtol, atol)`` would weaken the hard max-abs
    bound on larger-magnitude components via ``atol + rtol * |reference|``.
    """

    np.testing.assert_allclose(
        parity["jax_value"],
        parity["cpu_value"],
        rtol=_UPSTREAM_PENALTY_VALUE_PARITY_RTOL,
        atol=_UPSTREAM_PENALTY_VALUE_PARITY_ATOL,
        err_msg="CPU/JAX penalty value mismatch",
    )
    gradient_diff = np.abs(parity["jax_gradient"] - parity["cpu_gradient"])
    gradient_max_abs = float(np.max(gradient_diff))
    assert gradient_max_abs < _UPSTREAM_PENALTY_GRADIENT_MAX_ABS_TOL, (
        "CPU/JAX gradient max-abs "
        f"{gradient_max_abs:.3e} >= {_UPSTREAM_PENALTY_GRADIENT_MAX_ABS_TOL:.0e}"
    )
    cpu_gradient_norm = float(np.linalg.norm(parity["cpu_gradient"]))
    gradient_rel_norm = float(
        np.linalg.norm(gradient_diff) / max(cpu_gradient_norm, 1e-30)
    )
    assert gradient_rel_norm < _UPSTREAM_PENALTY_GRADIENT_REL_NORM_TOL, (
        "CPU/JAX gradient rel-norm "
        f"{gradient_rel_norm:.3e} >= {_UPSTREAM_PENALTY_GRADIENT_REL_NORM_TOL:.0e}"
    )


class TestParametrizedPenaltyGradientTaylor:
    """Multi-epsilon Taylor test for penalty gradient across the sweep."""

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_gradient_taylor(self, stellsym, optimize_G):
        case = _build_penalty_problem(stellsym=stellsym, optimize_G=optimize_G)
        _assert_gradient_taylor_convergence(
            case["objective"],
            case["x"],
            label=f"stellsym={stellsym}, optimize_G={optimize_G}: ",
        )


class TestParametrizedBFGSConvergence:
    """BFGS convergence across the stellsym x optimize_G sweep."""

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_bfgs_reduces_objective(self, stellsym, optimize_G):
        case = _build_penalty_problem(stellsym=stellsym, optimize_G=optimize_G)
        val_init = float(case["objective"](case["x"]))
        result = jax_minimize(
            case["objective"],
            case["x"],
            method="bfgs",
            tol=1e-10,
            maxiter=200,
        )
        val_final = float(result.fun)
        assert val_final < val_init, (
            f"stellsym={stellsym}, optimize_G={optimize_G}: "
            f"BFGS did not reduce objective {val_init:.6e} → {val_final:.6e}"
        )


# ---------------------------------------------------------------------------
# P19 + P21: Upstream surface factory parity (coil_arrays dispatch path)
# ---------------------------------------------------------------------------


def _extract_jax_penalty_inputs(surface, bs, *, optimize_G):
    """Extract raw JAX arrays from CPU objects for the coil_arrays path.

    This bypasses the BoozerSurfaceJAX adapter and GroupedCoilSetSpec,
    exercising the ``coil_arrays``-based dispatch inside
    ``_boozer_penalty_objective`` (grouped_*_from_inputs functions).
    """
    from .boozersurface_jax_test_helpers import (
        _UPSTREAM_BOOZER_IOTA0,
        _upstream_initial_G,
    )

    mpol, ntor, nfp = surface.mpol, surface.ntor, surface.nfp
    qphi = jnp.asarray(surface.quadpoints_phi, dtype=jnp.float64)
    qtheta = jnp.asarray(surface.quadpoints_theta, dtype=jnp.float64)
    sdofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

    if surface.stellsym:
        scatter = jnp.asarray(stellsym_scatter_indices(mpol, ntor), dtype=jnp.int32)
    else:
        scatter = None

    gammas_list, dashs_list, currs_list = [], [], []
    for coil in bs.coils:
        gammas_list.append(coil.curve.gamma())
        dashs_list.append(coil.curve.gammadash())
        currs_list.append(coil.current.get_value())
    gammas = jnp.asarray(np.stack(gammas_list))
    gammadashs = jnp.asarray(np.stack(dashs_list))
    currents = jnp.asarray(np.array(currs_list))
    coil_arrays = [(gammas, gammadashs, currents)]

    G = _upstream_initial_G(currs_list, nfp)
    iota = _UPSTREAM_BOOZER_IOTA0

    if optimize_G:
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    else:
        x = jnp.concatenate([sdofs, jnp.array([iota])])

    return {
        "x": x,
        "coil_arrays": coil_arrays,
        "qphi": qphi,
        "qtheta": qtheta,
        "mpol": mpol,
        "ntor": ntor,
        "nfp": nfp,
        "stellsym": surface.stellsym,
        "scatter_indices": scatter,
        "optimize_G": optimize_G,
    }


class TestUpstreamSurfaceFactoryParity:
    """Validate the coil_arrays dispatch path of ``_boozer_penalty_objective``.

    Unlike ``TestUpstreamFactoryBoozerMatrix`` (which tests via the adapter's
    ``coil_set_spec`` path), this class passes raw ``coil_arrays`` tuples
    directly, exercising the ``grouped_*_from_inputs`` dispatch branch.
    """

    @staticmethod
    def _make_problem(stellsym, optimize_G):
        from simsopt.configs import get_data

        from .boozersurface_jax_test_helpers import (
            _UPSTREAM_BOOZER_CONSTRAINT_WEIGHT,
            _UPSTREAM_BOOZER_TF_TARGET,
            _boozer_penalty_objective,
        )
        from .surface_test_helpers import get_surface

        base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
        s = get_surface("SurfaceXYZTensorFourier", stellsym, nfp=nfp)
        s.fit_to_curve(ma, 0.1)

        inputs = _extract_jax_penalty_inputs(s, bs, optimize_G=optimize_G)

        def objective(xx):
            return _boozer_penalty_objective(
                xx,
                coil_arrays=inputs["coil_arrays"],
                quadpoints_phi=inputs["qphi"],
                quadpoints_theta=inputs["qtheta"],
                mpol=inputs["mpol"],
                ntor=inputs["ntor"],
                nfp=inputs["nfp"],
                stellsym=inputs["stellsym"],
                scatter_indices=inputs["scatter_indices"],
                surface_kind="generic",
                targetlabel=_UPSTREAM_BOOZER_TF_TARGET,
                constraint_weight=_UPSTREAM_BOOZER_CONSTRAINT_WEIGHT,
                label_type="volume",
                phi_idx=0,
                optimize_G=optimize_G,
                weight_inv_modB=True,
            )

        return objective, inputs["x"]

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_penalty_value_is_finite(self, stellsym, optimize_G):
        """Penalty objective returns a finite non-negative scalar."""
        obj, x = self._make_problem(stellsym, optimize_G)
        val = float(obj(x))
        assert np.isfinite(val), f"Non-finite penalty: {val}"
        assert val >= 0.0, f"Negative penalty: {val}"

    @pytest.mark.parametrize("stellsym", _STELLSYM_LIST)
    @pytest.mark.parametrize("optimize_G", _OPTIMIZE_G_LIST)
    def test_gradient_taylor_convergence(self, stellsym, optimize_G):
        """Multi-epsilon Taylor test via the coil_arrays dispatch path."""
        obj, x = self._make_problem(stellsym, optimize_G)
        _assert_gradient_taylor_convergence(
            obj,
            x,
            label=f"stellsym={stellsym}, optimize_G={optimize_G}: ",
        )


# ---------------------------------------------------------------------------
# P18 / P19 / P21: upstream factory matrix coverage (coil_set_spec path)
# ---------------------------------------------------------------------------


class TestUpstreamFactoryBoozerMatrix:
    """Factory-driven JAX sweep over the upstream Boozer surface matrix."""

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_gradient_taylor_matrix(
        self,
        surfacetype,
        stellsym,
        optimize_G,
    ):
        """Mirror upstream's surfacetype x stellsym x optimize_G sweep."""
        case = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            optimize_G,
        )
        objective = case.jax_boozer._make_penalty_objective_with(
            case.optimize_G,
            case.jax_boozer.options["weight_inv_modB"],
            case.constraint_weight,
        )
        x = jnp.asarray(case.x, dtype=jnp.float64)
        _assert_gradient_taylor_convergence(objective, x)

    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    @pytest.mark.parametrize("optimize_G", UPSTREAM_BOOZER_OPTIMIZE_G)
    def test_penalty_value_and_gradient_cpu_parity_tensor_matrix(
        self,
        stellsym,
        optimize_G,
    ):
        """Tensor-surface factory cases must preserve the direct CPU/JAX parity contract."""
        case = _build_upstream_boozer_penalty_case(
            "SurfaceXYZTensorFourier",
            stellsym,
            optimize_G,
        )
        parity = _evaluate_upstream_boozer_penalty_case(case)
        _assert_upstream_penalty_parity(parity)

    @pytest.mark.parametrize("surfacetype", UPSTREAM_BOOZER_SURFACE_TYPES)
    @pytest.mark.parametrize("stellsym", UPSTREAM_BOOZER_STELLSYM)
    def test_penalty_case_packs_G_only_when_requested(self, surfacetype, stellsym):
        """The upstream factory should append only the optional G decision variable."""
        case_without_G = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            False,
        )
        case_with_G = _build_upstream_boozer_penalty_case(
            surfacetype,
            stellsym,
            True,
        )

        assert case_with_G.x.shape == (case_without_G.x.size + 1,)
        np.testing.assert_allclose(case_with_G.x[:-1], case_without_G.x)
        assert np.isfinite(case_with_G.x[-1])

    def test_exact_surface_factory_rejects_surface_xyzfourier(self):
        """The exact path is hard-gated to SurfaceXYZTensorFourier."""
        case = _build_upstream_exact_surface_case("SurfaceXYZFourier")

        with pytest.raises(
            RuntimeError,
            match="Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier",
        ):
            case.jax_boozer.solve_residual_equation_exactly_newton(
                iota=case.initial_iota,
                G=case.initial_G,
                maxiter=1,
            )

    def test_exact_surface_factory_tensor_residual_is_finite(self):
        """The exact factory data remains valid for SurfaceXYZTensorFourier."""
        case = _build_upstream_exact_surface_case("SurfaceXYZTensorFourier")
        mask_indices = case.jax_boozer._compute_stellsym_mask_indices()
        residual_fn = case.jax_boozer._make_exact_residual(mask_indices)
        x = case.jax_boozer._pack_decision_vector(case.initial_iota, case.initial_G)
        residual = residual_fn(x)

        assert residual.ndim == 1
        assert residual.size > 0
        assert np.all(np.isfinite(np.asarray(residual)))
