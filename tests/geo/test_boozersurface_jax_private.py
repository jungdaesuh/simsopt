"""Private optimizer runtime tests for BoozerSurfaceJAX."""

import logging
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt.geo.optimizer_jax_private._bfgs as _private_bfgs
import simsopt.geo.optimizer_jax_private._common as _opt_common
from jax.flatten_util import ravel_pytree

from .boozersurface_jax_test_helpers import (
    PRIVATE_OPTIMIZER_JAX_VERSION,
    _bsj,
    _make_mock_boozer_surface,
    _opt,
    _patch_newton_polish_runner,
    _soj,
    _successful_minimize_result,
    _successful_newton_polish_result,
    jax_minimize,
)


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


def _assert_plu_tuple_matches(actual, expected) -> None:
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, atol=1e-14)


def _assert_plu_tuple_is_zero(parts) -> None:
    for part in parts:
        np.testing.assert_array_equal(part, np.zeros_like(part))


def test_traceable_plu_or_dummy_accepts_python_and_traced_predicates():
    matrix = jnp.asarray([[3.0, 1.0], [2.0, 4.0]], dtype=jnp.float64)
    expected = tuple(np.asarray(part) for part in jax.scipy.linalg.lu(matrix))

    eager_true = tuple(
        np.asarray(part) for part in _bsj._traceable_plu_or_dummy(matrix, finite=True)
    )
    eager_false = tuple(
        np.asarray(part) for part in _bsj._traceable_plu_or_dummy(matrix, finite=False)
    )

    _assert_plu_tuple_matches(eager_true, expected)
    _assert_plu_tuple_is_zero(eager_false)

    @jax.jit
    def traceable(finite):
        return _bsj._traceable_plu_or_dummy(matrix, finite=finite)

    traced_true = tuple(np.asarray(part) for part in traceable(jnp.asarray(True)))
    traced_false = tuple(np.asarray(part) for part in traceable(jnp.asarray(False)))

    _assert_plu_tuple_matches(traced_true, expected)
    _assert_plu_tuple_is_zero(traced_false)


def test_resolve_lbfgs_limits_normalizes_to_int32_counter_domain():
    maxiter, maxfun, maxgrad = _opt_common._resolve_lbfgs_limits(
        4,
        1.2,
        None,
        np.inf,
    )

    assert isinstance(maxiter, np.int32)
    assert isinstance(maxfun, np.int32)
    assert isinstance(maxgrad, np.int32)
    assert int(maxiter) == 2
    assert int(maxfun) == np.iinfo(np.int32).max
    assert int(maxgrad) == np.iinfo(np.int32).max


def test_reduction_helpers_pass_host_init_values_to_lax_reduce(monkeypatch):
    recorded_inits = []

    def fake_reduce(flat, init_value, reducer, dims):
        del reducer, dims
        recorded_inits.append(init_value)
        return flat[0]

    monkeypatch.setattr(_opt_common.lax, "reduce", fake_reduce)
    sample = jax.device_put(np.asarray([3.0, -2.0], dtype=np.float64))

    _opt_common._reduce_sum_all(sample)
    _opt_common._reduce_max_all(sample)

    assert len(recorded_inits) == 2
    for init_value in recorded_inits:
        assert not isinstance(init_value, jax.Array)
        assert isinstance(init_value, np.ndarray)


def test_bfgs_curvature_terms_reject_bad_curvature_updates():
    s_k = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    y_negative = jnp.asarray([-1.0e-3, 0.0], dtype=jnp.float64)
    y_near_orthogonal = jnp.asarray([1.0e-20, 1.0], dtype=jnp.float64)

    _, _, negative_valid, _ = _private_bfgs._bfgs_curvature_terms(
        s_k,
        y_negative,
        x_dtype=s_k.dtype,
    )
    _, _, near_orthogonal_valid, _ = _private_bfgs._bfgs_curvature_terms(
        s_k,
        y_near_orthogonal,
        x_dtype=s_k.dtype,
    )

    assert bool(negative_valid) is False
    assert bool(near_orthogonal_valid) is False


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
    not _opt.private_optimizer_runtime_is_supported(jax.__version__),
    reason=PRIVATE_RUNTIME_REASON,
)
REQUIRES_PRIVATE_LBFGS_RUNTIME = pytest.mark.skipif(
    not _opt.private_optimizer_runtime_is_supported(jax.__version__),
    reason=PRIVATE_LBFGS_RUNTIME_REASON,
)
REQUIRES_PRIVATE_LBFGS_BUDGET_RUNTIME = pytest.mark.skipif(
    not _opt.private_optimizer_runtime_is_supported(jax.__version__),
    reason=PRIVATE_LBFGS_BUDGET_REASON,
)
REQUIRES_PRIVATE_LIMITED_MEMORY_RUNTIME = pytest.mark.skipif(
    not _opt.private_optimizer_runtime_is_supported(jax.__version__),
    reason=PRIVATE_LIMITED_MEMORY_REASON,
)


def _emit_sparse_progress(progress_callback):
    progress_callback(1, 3.0, 2.0)
    progress_callback(7, 2.0, 1.0)
    progress_callback(25, 1.0, 0.5)


def _private_lbfgs_quadratic_state(
    monkeypatch,
    *,
    x0,
    line_search_kwargs,
    maxiter=5,
    gtol=1e-8,
    maxcor=4,
):
    from simsopt.geo.optimizer_jax_private import _LineSearchResults
    from simsopt.geo.optimizer_jax_private import _lbfgs as _lbfgs_module

    def quad(x):
        return 0.5 * jnp.dot(x, x)

    def fake_line_search(*_args, **_kwargs):
        return _LineSearchResults(
            failed=jnp.array(False),
            nit=jnp.array(1),
            nfev=jnp.array(1),
            ngev=jnp.array(1),
            k=jnp.array(1),
            **line_search_kwargs,
        )

    monkeypatch.setattr(
        _lbfgs_module,
        "_line_search_value_and_grad",
        fake_line_search,
    )
    state = _lbfgs_module._minimize_lbfgs_private(
        quad,
        x0,
        maxiter=maxiter,
        gtol=gtol,
        maxcor=maxcor,
    )
    return state, quad


def _assert_lbfgs_state_preserved(state, x0, quad, *, maxcor=4, ls_status=None):
    assert bool(state.converged) is False
    assert bool(state.failed) is True
    assert int(state.status) == 5
    if ls_status is not None:
        assert int(state.ls_status) == ls_status
    np.testing.assert_allclose(np.asarray(state.x_k), np.asarray(x0))
    np.testing.assert_allclose(np.asarray(state.f_k), np.asarray(quad(x0)))
    np.testing.assert_allclose(np.asarray(state.g_k), np.asarray(x0))
    np.testing.assert_array_equal(np.asarray(state.rho_history), np.zeros(maxcor))
    assert float(state.gamma) == pytest.approx(1.0)


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
    def test_minimize_bfgs_private_preserves_last_finite_iterate_on_nonfinite_step(
        self,
        monkeypatch,
    ):
        """A non-finite line-search proposal must keep the last finite iterate."""
        from simsopt.geo.optimizer_jax_private import _LineSearchResults
        from simsopt.geo.optimizer_jax_private import _bfgs as _bfgs_module

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def fake_line_search(*_args, **_kwargs):
            return _LineSearchResults(
                failed=jnp.array(False),
                nit=jnp.array(1),
                nfev=jnp.array(1),
                ngev=jnp.array(1),
                k=jnp.array(1),
                a_k=jnp.array(1.0, dtype=jnp.float64),
                f_k=jnp.array(np.nan, dtype=jnp.float64),
                g_k=jnp.array([np.nan, np.nan], dtype=jnp.float64),
                status=jnp.array(7),
            )

        monkeypatch.setattr(_bfgs_module, "_line_search", fake_line_search)

        state = _bfgs_module._minimize_bfgs_private(
            quad,
            x0,
            maxiter=5,
            gtol=1e-8,
        )

        assert bool(state.converged) is False
        assert bool(state.failed) is True
        assert int(state.status) == 2
        np.testing.assert_allclose(np.asarray(state.x_k), np.asarray(x0))
        np.testing.assert_allclose(np.asarray(state.f_k), np.asarray(quad(x0)))
        np.testing.assert_allclose(np.asarray(state.g_k), np.asarray(x0))

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_bfgs_private_failed_state_does_not_flip_to_converged(
        self,
        monkeypatch,
    ):
        """Post-loop gradient refresh must not turn a failed iterate into success."""
        from simsopt.geo.optimizer_jax_private import _LineSearchResults
        from simsopt.geo.optimizer_jax_private import _bfgs as _bfgs_module

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def fake_line_search(*_args, **_kwargs):
            return _LineSearchResults(
                failed=jnp.array(True),
                nit=jnp.array(1),
                nfev=jnp.array(1),
                ngev=jnp.array(1),
                k=jnp.array(1),
                a_k=jnp.array(0.0, dtype=jnp.float64),
                f_k=jnp.array(0.0, dtype=jnp.float64),
                g_k=jnp.array([0.0], dtype=jnp.float64),
                status=jnp.array(7),
            )

        monkeypatch.setattr(_bfgs_module, "_line_search", fake_line_search)

        initial_state = _opt._BFGSResults(
            converged=jnp.array(False),
            failed=jnp.array(False),
            k=jnp.array(0, dtype=jnp.int32),
            nfev=jnp.array(1, dtype=jnp.int32),
            ngev=jnp.array(1, dtype=jnp.int32),
            nhev=jnp.array(0, dtype=jnp.int32),
            x_k=jnp.array([0.0], dtype=jnp.float64),
            f_k=jnp.array(0.0, dtype=jnp.float64),
            g_k=jnp.array([1.0], dtype=jnp.float64),
            H_k=jnp.eye(1, dtype=jnp.float64),
            old_old_fval=jnp.array(0.5, dtype=jnp.float64),
            status=jnp.array(0, dtype=jnp.int32),
            line_search_status=jnp.array(0, dtype=jnp.int32),
        )

        state = _bfgs_module._minimize_bfgs_private(
            quad,
            jnp.array([0.0], dtype=jnp.float64),
            maxiter=5,
            gtol=1e-8,
            initial_state=initial_state,
        )

        assert bool(state.failed) is True
        assert bool(state.converged) is False
        assert int(state.status) == 9
        np.testing.assert_allclose(np.asarray(state.x_k), np.zeros(1), atol=1e-12)
        np.testing.assert_allclose(np.asarray(state.g_k), np.zeros(1), atol=1e-12)

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
    def test_minimize_lbfgs_private_preserves_last_finite_iterate_on_nonfinite_step(
        self,
        monkeypatch,
    ):
        """A non-finite accepted proposal must not poison the private L-BFGS state."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state, quad = _private_lbfgs_quadratic_state(
            monkeypatch,
            x0=x0,
            line_search_kwargs=dict(
                a_k=jnp.array(1.0, dtype=jnp.float64),
                f_k=jnp.array(np.nan, dtype=jnp.float64),
                g_k=jnp.array([np.nan, np.nan], dtype=jnp.float64),
                status=jnp.array(0),
            ),
        )
        _assert_lbfgs_state_preserved(state, x0, quad, ls_status=0)

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_rejects_degenerate_curvature_update(
        self,
        monkeypatch,
    ):
        """A non-converged step with unusable y^T s must fail before history updates."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state, quad = _private_lbfgs_quadratic_state(
            monkeypatch,
            x0=x0,
            line_search_kwargs=dict(
                a_k=jnp.array(1.0, dtype=jnp.float64),
                f_k=jnp.array(1.0, dtype=jnp.float64),
                g_k=jnp.array([3.0, -1.0], dtype=jnp.float64),
                status=jnp.array(0),
            ),
        )
        _assert_lbfgs_state_preserved(state, x0, quad)

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_rejects_stalled_nonconverged_step(
        self,
        monkeypatch,
    ):
        """A zero-progress accepted step must fail unless the iterate actually converged."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state, quad = _private_lbfgs_quadratic_state(
            monkeypatch,
            x0=x0,
            line_search_kwargs=dict(
                a_k=jnp.array(0.0, dtype=jnp.float64),
                f_k=jnp.array(2.5, dtype=jnp.float64),
                g_k=jnp.array([1.0, -2.0], dtype=jnp.float64),
                status=jnp.array(0),
            ),
        )
        _assert_lbfgs_state_preserved(state, x0, quad)

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_clamps_gamma_on_large_curvature_ratio(
        self,
        monkeypatch,
    ):
        """Accepted curvature updates must bound gamma to a finite, dtype-scaled range."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        gamma_max = np.reciprocal(np.sqrt(np.finfo(np.float64).eps))
        state, _ = _private_lbfgs_quadratic_state(
            monkeypatch,
            x0=x0,
            line_search_kwargs=dict(
                a_k=jnp.array(1000.0, dtype=jnp.float64),
                f_k=jnp.array(1.0, dtype=jnp.float64),
                g_k=jnp.array([0.9999999, -1.9999998], dtype=jnp.float64),
                status=jnp.array(0),
            ),
            maxiter=1,
            gtol=1e-12,
        )

        assert bool(state.failed) is True
        assert int(state.status) == 1
        assert np.isfinite(float(state.gamma))
        assert float(state.gamma) == pytest.approx(gamma_max)
        np.testing.assert_allclose(np.asarray(state.rho_history[-1]), 2000.0, rtol=1e-6)

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
    def test_hybrid_nonfinite_prefix_emits_warning_and_progress(
        self,
        monkeypatch,
        caplog,
    ):
        """Hybrid non-finite prefixes must surface observability before aborting."""
        prefix = types.SimpleNamespace(
            x=jnp.array([1.0, -1.0]),
            fun=np.nan,
            jac=jnp.array([np.nan, 0.0]),
            nit=2,
            nfev=3,
            njev=3,
            nhev=0,
            success=False,
            status=1,
            hess_inv=np.eye(2),
            message="prefix failed",
        )
        progress_events = []

        monkeypatch.setattr(_opt, "_scipy_minimize", lambda *args, **kwargs: prefix)
        monkeypatch.setattr(
            _opt,
            "_minimize_bfgs_private",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("continuation must not run after non-finite prefix")
            ),
        )

        with caplog.at_level(logging.WARNING, logger="simsopt.geo.optimizer_jax"):
            result = jax_minimize(
                lambda x: jnp.sum(x**2),
                jnp.array([1.0, -1.0]),
                method="bfgs-hybrid",
                maxiter=8,
                progress_callback=lambda nit, fun_value, grad_inf: progress_events.append(
                    (nit, fun_value, grad_inf)
                ),
            )

        assert result is prefix
        assert "non-finite continuation state" in caplog.text
        assert len(progress_events) == 1
        nit, fun_value, grad_inf = progress_events[0]
        assert nit == 2
        assert np.isnan(fun_value)
        assert np.isnan(grad_inf)

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
    def test_bfgs_ondevice_accepts_pytree_x0_and_restores_result_structure(self):
        """bfgs-ondevice should flatten pytrees internally and restore them on return."""

        def quad(state):
            return 0.5 * (
                jnp.dot(state["surface"], state["surface"])
                + jnp.dot(state["current"], state["current"])
            )

        x0 = {
            "surface": jnp.array([1.0, -2.0], dtype=jnp.float64),
            "current": jnp.array([0.5], dtype=jnp.float64),
        }
        callback_calls = []

        result = jax_minimize(
            quad,
            x0,
            method="bfgs-ondevice",
            maxiter=5,
            callback=lambda state: callback_calls.append(state),
        )

        assert isinstance(result.x, dict)
        assert isinstance(result.jac, dict)
        np.testing.assert_allclose(result.x["surface"], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(result.x["current"], np.zeros(1), atol=1e-12)
        np.testing.assert_allclose(result.jac["surface"], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(result.jac["current"], np.zeros(1), atol=1e-12)
        assert callback_calls
        assert isinstance(callback_calls[0], dict)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_accepts_pytree_x0(self):
        """Direct private L-BFGS entry should flatten pytrees before runtime checks."""

        def quad(state):
            return 0.5 * (
                jnp.dot(state["surface"], state["surface"])
                + jnp.dot(state["current"], state["current"])
            )

        x0 = {
            "surface": jnp.array([1.0, -2.0], dtype=jnp.float64),
            "current": jnp.array([0.5], dtype=jnp.float64),
        }
        flat_x0, _ = ravel_pytree(x0)

        state = _opt._minimize_lbfgs_private(
            quad,
            x0,
            maxiter=10,
            gtol=1e-8,
        )

        np.testing.assert_allclose(
            np.asarray(state.x_k), np.zeros_like(flat_x0), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(state.g_k), np.zeros_like(flat_x0), atol=1e-12
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
        """A NaN objective encountered mid-loop must fail from the last finite iterate."""

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
        assert float(result.fun) == pytest.approx(float(0.5 * jnp.dot(x0, x0)))
        assert np.all(np.isfinite(np.asarray(result.x)))
        assert np.all(np.isfinite(np.asarray(result.jac)))
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
            x = jnp.asarray(x, dtype=jnp.float64)
            return 0.5 * jnp.dot(x, x), x

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
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_skips_debug_callback_without_progress(
        self, monkeypatch
    ):
        """Traceable Newton polish must not materialize host callbacks when unused."""
        observed = {"called": False}

        def forbidden_debug_callback(*_args, **_kwargs):
            observed["called"] = True
            raise AssertionError(
                "jax.debug.callback must not run without progress_callback"
            )

        monkeypatch.setattr(_opt.jax.debug, "callback", forbidden_debug_callback)

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=1,
            tol=1e-12,
            stab=0.0,
            progress_callback=None,
        )

        assert observed["called"] is False
        np.testing.assert_allclose(
            np.asarray(result["x"]),
            np.zeros_like(np.asarray(x0)),
            atol=1e-12,
        )
        assert bool(result["success"]) is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_rejected_step_keeps_zero_iterations(
        self, monkeypatch
    ):
        """Rejected traceable Newton steps must not increment nit or emit progress."""
        observed = {"progress_calls": 0}

        def fake_gmres_solve(_hvp_fn, _x, rhs, *, stab, tol):
            del _hvp_fn, _x, stab, tol
            return -rhs, jnp.zeros_like(rhs), None

        monkeypatch.setattr(
            _opt,
            "_gmres_solve_newton_system",
            fake_gmres_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=3,
            tol=1e-12,
            stab=0.0,
            progress_callback=lambda *_args: observed.__setitem__(
                "progress_calls",
                observed["progress_calls"] + 1,
            ),
        )

        np.testing.assert_allclose(np.asarray(result["x"]), np.asarray(x0))
        assert int(result["nit"]) == 0
        assert bool(result["success"]) is False
        assert observed["progress_calls"] == 0

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
            return _successful_minimize_result(x0)

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
        assert res["optimizer_method"] == "bfgs-ondevice"
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
