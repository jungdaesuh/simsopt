"""Private optimizer runtime tests for BoozerSurfaceJAX."""

import inspect
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.geo.optimizers.reference as _opt_ref
import simsopt_jax.geo.optimizers.private._bfgs as _private_bfgs
import simsopt_jax.geo.optimizers.private._common as _opt_common
import simsopt_jax.geo.optimizers.private._lbfgs as _private_lbfgs
import simsopt_jax.geo.optimizer_host_lbfgs as _host_lbfgs
from simsopt_jax.geo.optimizers.private import (
    _BFGSResults,
    _LineSearchResults,
    _line_search,
    _line_search_module,
    _line_search_value_and_grad,
)
from conftest import enable_non_strict_jax_backend
from jax.flatten_util import ravel_pytree
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy import optimize

from .boozersurface_jax_test_helpers import (
    PRIVATE_OPTIMIZER_JAX_VERSION,
    _bsj,
    _make_mock_boozer_surface,
    _mock_linear_solve_status,
    _opt,
    _patch_newton_polish_runner,
    _soj,
    _successful_minimize_result,
    _successful_newton_polish_result,
    jax_minimize,
)


def test_pytree_inexact_dtype_empty_tree_respects_x64_policy():
    expected_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert _opt_common._pytree_inexact_dtype({}) == expected_dtype


def test_solve_boozer_adjoint_rejects_factor_only_runtime_state():
    adjoint_state = types.SimpleNamespace(
        linearization_kind="exact_jacobian",
        linear_solve_factors=("P", "L", "U"),
        plu=("P", "L", "U"),
    )

    with pytest.raises(RuntimeError, match="solve_transpose"):
        _soj._solve_boozer_adjoint(adjoint_state, "rhs")


def test_solve_boozer_adjoint_raises_on_failed_operator_runtime():
    adjoint_state = types.SimpleNamespace(
        linearization_kind="hessian",
        solve_transpose_with_status=lambda rhs: (
            rhs,
            _mock_linear_solve_status(False),
        ),
    )

    with pytest.raises(RuntimeError, match="Boozer adjoint linear solve failed"):
        _soj._solve_boozer_adjoint(adjoint_state, jnp.ones((2,), dtype=jnp.float64))


def test_traceable_forward_result_packs_newton_linear_solver_backend_code():
    """K1 progress metadata must preserve the traceable Newton backend code."""
    code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
        _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU
    ]

    result = _soj._pack_traceable_forward_result(
        value=jnp.asarray(1.25, dtype=jnp.float64),
        x=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        sdofs=jnp.asarray([1.0], dtype=jnp.float64),
        iota=jnp.asarray(0.1, dtype=jnp.float64),
        G=jnp.asarray(-2.0, dtype=jnp.float64),
        linear_solve_factors=None,
        success=jnp.asarray(True),
        primal_success=jnp.asarray(True),
        adjoint_linear_solve_available=jnp.asarray(False),
        newton_linear_solve_backend_code=jnp.asarray(code, dtype=jnp.int32),
    )

    assert bool(result["newton_linear_solve_backend_code_present"]) is True
    assert int(result["newton_linear_solve_backend_code"]) == code


def _assert_plu_tuple_matches(actual, expected) -> None:
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, atol=1e-14)


def _assert_plu_tuple_is_nan(parts) -> None:
    for part in parts:
        assert np.isnan(part).all()


def _device_half() -> jax.Array:
    return jax.device_put(np.asarray(0.5, dtype=np.float64))


def _record_host_arrays(points, *, dtype=None):
    def callback(x):
        points.append(np.asarray(x, dtype=dtype))

    return callback


def _record_progress(points):
    def callback(nit, fun, grad_norm):
        points.append((int(nit), float(fun), float(grad_norm)))

    return callback


def test_private_lbfgs_history_size_preserves_maxcor_above_dimension():
    assert _private_lbfgs._resolve_lbfgs_history_size(200, maxiter_limit=1500) == 200
    assert _private_lbfgs._resolve_lbfgs_history_size(8, maxiter_limit=1500) == 8
    assert _private_lbfgs._resolve_lbfgs_history_size(8, maxiter_limit=3) == 3


def test_private_lbfgs_workspace_bytes_reports_bounded_state_size():
    n = 7
    maxcor = 4
    dtype = np.dtype(np.float64)
    expected = (
        _private_lbfgs.lbfgsb.lbfgsb_workspace_size(n, maxcor) + 29
    ) * dtype.itemsize + (
        _private_lbfgs.lbfgsb.lbfgsb_iwa_size(n) + 2 + 2 + 4 + 44
    ) * np.dtype(np.int32).itemsize

    assert _private_lbfgs._lbfgsb_workspace_bytes(n, maxcor, dtype) == expected


def test_lbfgs_stepwise_driver_host_reads_use_host_boundary_helpers():
    status_source = inspect.getsource(_private_lbfgs._lbfgsb_stepwise_host_status)
    driver_source = inspect.getsource(_private_lbfgs._lbfgsb_stepwise_driver)
    observer_source = inspect.getsource(_private_lbfgs._lbfgsb_accepted_step_observer)
    source = "\n".join(
        [
            status_source,
            driver_source,
            observer_source,
        ]
    )

    assert "jax.device_get" not in source
    assert "np.asarray(" not in source
    assert "workspace.task" not in status_source
    assert "workspace.task" not in driver_source
    assert "host_bool(step_result.terminal" in status_source
    assert "host_int(step_result.entry_kind" in status_source
    assert "host_int" in source


def test_lbfgs_ondevice_fast_path_selection_is_bounds_derived():
    kernel_source = inspect.getsource(
        _private_lbfgs._lbfgsb_advance_to_next_observable_kernel
    )
    impl_source = inspect.getsource(_private_lbfgs._minimize_lbfgs_private_impl)

    assert _private_lbfgs._LBFGSB_PRIVATE_BOUNDS is None
    assert _private_lbfgs._lbfgsb_unconstrained_fast_path_enabled(None) is True
    assert (
        _private_lbfgs._lbfgsb_unconstrained_fast_path_enabled([(None, None)]) is False
    )
    assert "unconstrained_fast_path=True" not in kernel_source
    assert "unconstrained_fast_path=unconstrained_fast_path" in kernel_source
    assert "_lbfgsb_unconstrained_fast_path_enabled" in impl_source


def test_optax_lbfgs_ondevice_default_maxcor_uses_optax_memory_default(monkeypatch):
    captured = {}

    def fake_require_target_backend_x64(optimizer_backend):
        captured["x64_backend"] = optimizer_backend

    def fake_optax_minimize(value_and_grad, x0, *, driver, options, callback):
        value, grad = value_and_grad(x0)
        captured["driver"] = driver
        captured["options"] = options
        captured["callback"] = callback
        return types.SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            fun=float(np.asarray(value)),
            jac=np.asarray(grad, dtype=float),
            nit=0,
            nfev=1,
            njev=1,
            status=0,
            success=True,
            message="ok",
            driver=driver,
            options_used=options,
            optimistix_result=None,
            optimistix_result_message=None,
        )

    monkeypatch.setattr(
        _opt,
        "require_target_backend_x64",
        fake_require_target_backend_x64,
    )
    monkeypatch.setattr(_opt, "run_optax_minimize", fake_optax_minimize)

    def value_and_grad(x):
        return jnp.sum((x - 1.0) ** 2), 2.0 * (x - 1.0)

    result = _opt.target_minimize(
        value_and_grad,
        jnp.array([0.0, 2.0], dtype=jnp.float64),
        method="optax-lbfgs-ondevice",
        tol=1e-8,
        maxiter=3,
        options={},
        value_and_grad=True,
    )

    assert captured["x64_backend"] == "optax-lbfgs"
    assert captured["driver"] is _opt.Driver.OPTAX_LBFGS
    assert captured["options"].memory_size == 10
    assert result.message == "ok"


def test_matrix_rhs_linear_operators_apply_columns():
    x = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float64)
    rhs = jnp.asarray(
        [[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]],
        dtype=jnp.float64,
    )
    A = jnp.asarray(
        [[2.0, 0.1, -0.2], [0.3, 1.5, 0.4], [-0.1, 0.2, 1.8]],
        dtype=jnp.float64,
    )
    H = A.T @ A + jnp.diag(jnp.asarray([0.4, 0.5, 0.6], dtype=jnp.float64))

    jacobian_operator = _opt._jacobian_linear_operator(lambda y: A @ y, x)
    np.testing.assert_allclose(
        np.asarray(jacobian_operator["matvec"](rhs)),
        np.asarray(A @ rhs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jacobian_operator["transpose_matvec"](rhs)),
        np.asarray(A.T @ rhs),
        rtol=1e-12,
        atol=1e-12,
    )

    hessian_operator = _opt._hessian_linear_operator(
        lambda y: 0.5 * jnp.dot(y, H @ y),
        x,
        stab=0.25,
    )
    np.testing.assert_allclose(
        np.asarray(hessian_operator["matvec"](rhs)),
        np.asarray((H + 0.25 * jnp.eye(x.size, dtype=x.dtype)) @ rhs),
        rtol=1e-12,
        atol=1e-12,
    )


def _successful_optimize_result_for_x(x):
    from scipy.optimize import OptimizeResult

    return OptimizeResult(
        x=x,
        fun=0.0,
        jac=jnp.zeros_like(x),
        nit=0,
        nfev=0,
        njev=0,
        status=0,
        success=True,
    )


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
    _assert_plu_tuple_is_nan(eager_false)

    @jax.jit
    def traceable(finite):
        return _bsj._traceable_plu_or_dummy(matrix, finite=finite)

    traced_true = tuple(np.asarray(part) for part in traceable(jnp.asarray(True)))
    traced_false = tuple(np.asarray(part) for part in traceable(jnp.asarray(False)))

    _assert_plu_tuple_matches(traced_true, expected)
    _assert_plu_tuple_is_nan(traced_false)


def test_optimizer_dtype_uses_dtype_attr_without_eager_hostification(monkeypatch):
    class HasDtypeOnly:
        dtype = np.dtype(np.float64)

    original_asarray = _opt_common.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, HasDtypeOnly):
            raise AssertionError("np.asarray should not run when dtype attr exists")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(_opt_common.np, "asarray", guarded_asarray)

    assert _opt_common._optimizer_dtype(HasDtypeOnly()) == np.dtype(np.float64)


def test_prepare_optimizer_pytree_adapter_uses_leaf_metadata_without_hostification(
    monkeypatch,
):
    original_asarray = _opt.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            raise AssertionError(
                "np.asarray should not run on JAX pytree leaves during adapter prep"
            )
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(_opt.np, "asarray", guarded_asarray)

    adapter = _opt._prepare_optimizer_pytree_adapter(
        {
            "surface": jax.device_put(np.asarray([1.0, -2.0], dtype=np.float64)),
            "current": jax.device_put(np.asarray([0.5], dtype=np.float64)),
        }
    )

    assert adapter is not None
    assert len(adapter.leaf_signature) == 2
    assert {
        ((2,), np.dtype(np.float64).str),
        ((1,), np.dtype(np.float64).str),
    } == set(adapter.leaf_signature)


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


@pytest.mark.parametrize("method", ["bfgs-ondevice", "lbfgs-ondevice", "adam-ondevice"])
def test_target_minimize_rejects_failure_callback(method):
    def quad(x):
        return 0.5 * jnp.dot(x, x)

    with pytest.raises(
        ValueError,
        match="target_minimize\\(\\) does not support failure_callback",
    ):
        _opt.target_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method=method,
            failure_callback=lambda *args: None,
        )


@pytest.mark.parametrize(
    "method",
    ["lbfgs-ondevice", "lbfgs-scipy-jax", "lbfgs-scipy-jax-fullgraph"],
)
@pytest.mark.parametrize("option_name", ["initial_step_size", "maxgrad"])
def test_target_minimize_rejects_unsupported_scipy_lbfgsb_options(method, option_name):
    def quad(x):
        value = 0.5 * jnp.dot(x, x)
        return value, x

    with pytest.raises(
        ValueError,
        match="target L-BFGS-B methods follow SciPy L-BFGS-B options",
    ):
        _opt.target_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method=method,
            value_and_grad=True,
            options={option_name: 1.0},
        )


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


def test_line_search_value_and_grad_uses_explicit_initial_step_size():
    def quad(x):
        return 0.5 * jnp.dot(x, x), x

    xk = jnp.asarray([1.0], dtype=jnp.float64)
    pk = jnp.asarray([-1.0], dtype=jnp.float64)
    result = _line_search_value_and_grad(
        quad,
        xk,
        pk,
        old_fval=jnp.asarray(0.5, dtype=jnp.float64),
        gfk=jnp.asarray([1.0], dtype=jnp.float64),
        initial_step_size=0.125,
        maxiter=1,
    )

    assert float(result.a_k) == pytest.approx(0.125)


def test_line_search_value_and_grad_skips_zero_step_reevaluation_with_explicit_state():
    def quad(x):
        return 0.5 * jnp.dot(x, x), x

    xk = jnp.asarray([1.0], dtype=jnp.float64)
    pk = jnp.asarray([-1.0], dtype=jnp.float64)
    result = _line_search_value_and_grad(
        quad,
        xk,
        pk,
        old_fval=jnp.asarray(0.5, dtype=jnp.float64),
        gfk=jnp.asarray([1.0], dtype=jnp.float64),
        initial_step_size=0.125,
        maxiter=1,
    )

    assert int(result.nfev) == 1
    assert int(result.ngev) == 1
    assert float(result.f_k) == pytest.approx(0.5 * 0.875**2)


def test_line_search_value_and_grad_accepts_finite_decrease_when_armijo_misses():
    def armijo_miss_objective(x):
        return jnp.asarray(0.99999, dtype=x.dtype), jnp.asarray([-0.5], dtype=x.dtype)

    result = _line_search_value_and_grad(
        armijo_miss_objective,
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        old_fval=jnp.asarray(1.0, dtype=jnp.float64),
        gfk=jnp.asarray([-1.0], dtype=jnp.float64),
        initial_step_size=0.25,
        maxiter=1,
    )

    assert bool(result.failed) is False
    assert int(result.status) == 0
    assert float(result.a_k) == pytest.approx(0.25)
    assert float(result.f_k) == pytest.approx(0.99999)
    np.testing.assert_allclose(np.asarray(result.g_k), np.asarray([-0.5]))


def test_line_search_value_and_grad_shrinks_past_nonfinite_trial_gradient():
    def objective_with_invalid_gradient_region(x):
        a = x[0]
        value = -a + a * a
        grad = jnp.where(a >= 1.0e-6, jnp.nan, -1.0 + 2.0 * a)
        return value, jnp.asarray([grad], dtype=x.dtype)

    result = _line_search_value_and_grad(
        objective_with_invalid_gradient_region,
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        old_fval=jnp.asarray(0.0, dtype=jnp.float64),
        gfk=jnp.asarray([-1.0], dtype=jnp.float64),
        initial_step_size=0.2,
        maxiter=6,
    )

    assert bool(result.failed) is False
    assert float(result.a_k) < 1.0e-6
    assert np.isfinite(float(result.f_k))
    assert np.all(np.isfinite(np.asarray(result.g_k)))


def test_host_line_search_value_and_grad_shrinks_past_nonfinite_trial_gradient():
    def objective_with_invalid_gradient_region(x):
        a = np.asarray(x, dtype=np.float64)[0]
        value = -a + a * a
        grad = np.nan if a >= 1.0e-6 else -1.0 + 2.0 * a
        return value, np.asarray([grad], dtype=np.float64)

    result = _host_lbfgs.line_search_value_and_grad_host(
        objective_with_invalid_gradient_region,
        np.asarray([0.0], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        old_fval=np.float64(0.0),
        gfk=np.asarray([-1.0], dtype=np.float64),
        initial_step_size=0.2,
        maxiter=6,
    )

    assert result.failed is False
    assert result.a_k < 1.0e-6
    assert np.isfinite(result.f_k)
    assert np.all(np.isfinite(result.g_k))


def test_host_line_search_accepts_finite_decrease_when_armijo_misses():
    def restricted_func_and_grad(alpha):
        assert alpha == pytest.approx(0.25)
        return 0.99999, -0.5, np.asarray([0.5], dtype=np.float64)

    result = _host_lbfgs._line_search_from_restricted_func_and_grad(
        restricted_func_and_grad,
        pk=np.asarray([-1.0], dtype=np.float64),
        old_fval=1.0,
        gfk=np.asarray([1.0], dtype=np.float64),
        initial_step_size=0.25,
        maxiter=1,
    )

    assert result.failed is False
    assert result.a_k == pytest.approx(0.25)
    assert result.f_k == pytest.approx(0.99999)
    np.testing.assert_allclose(result.g_k, np.asarray([0.5], dtype=np.float64))
    assert result.status == 0
    assert result.requested_initial_step == pytest.approx(0.25)
    assert result.first_tested_alpha == pytest.approx(0.25)
    assert result.best_finite_alpha == pytest.approx(0.25)
    assert result.returned_alpha == pytest.approx(0.25)
    assert result.failure_reason == "accepted"
    assert result.armijo_margin == pytest.approx(1.5e-5)
    assert result.curvature_margin == pytest.approx(-0.4)


def test_host_line_search_failure_reports_trial_alpha_without_accepting_step():
    def restricted_func_and_grad(alpha):
        assert alpha == pytest.approx(0.25)
        return 1.00001, -0.5, np.asarray([0.5], dtype=np.float64)

    result = _host_lbfgs._line_search_from_restricted_func_and_grad(
        restricted_func_and_grad,
        pk=np.asarray([-1.0], dtype=np.float64),
        old_fval=1.0,
        gfk=np.asarray([1.0], dtype=np.float64),
        initial_step_size=0.25,
        maxiter=1,
    )

    assert result.failed is True
    assert result.a_k == pytest.approx(0.0)
    assert result.f_k == pytest.approx(1.0)
    np.testing.assert_allclose(result.g_k, np.asarray([1.0], dtype=np.float64))
    assert result.requested_initial_step == pytest.approx(0.25)
    assert result.first_tested_alpha == pytest.approx(0.25)
    assert result.best_finite_alpha == pytest.approx(0.0)
    assert result.returned_alpha == pytest.approx(0.0)
    assert result.failure_reason == "line_search_failed"
    assert np.isnan(result.armijo_margin)
    assert np.isnan(result.curvature_margin)


def test_minimize_lbfgs_host_core_rejected_step_log_separates_requested_alpha():
    def quad(x):
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * np.dot(x, x)), x

    def failing_line_search(**_kwargs):
        return _host_lbfgs.HostLineSearchResults(
            failed=True,
            nit=1,
            nfev=1,
            ngev=1,
            k=2,
            a_k=1.0,
            f_k=0.5,
            g_k=np.asarray([1.0], dtype=np.float64),
            status=1,
            requested_initial_step=0.5,
            first_tested_alpha=0.5,
            best_finite_alpha=0.25,
            returned_alpha=1.0,
            failure_reason="line_search_failed",
            armijo_margin=1.0e-6,
            curvature_margin=-1.0e-3,
        )

    result = _host_lbfgs.minimize_lbfgs_host_core(
        quad,
        np.asarray([1.0], dtype=np.float64),
        maxiter=2,
        initial_step_size=0.5,
        line_search_value_and_grad=failing_line_search,
    )

    assert result.failed is True
    np.testing.assert_allclose(result.x_k, np.asarray([1.0], dtype=np.float64))
    assert len(result.invalid_step_events) == 1
    event = result.invalid_step_events[0]
    assert event.step_scale == pytest.approx(0.0)
    assert event.requested_initial_step == pytest.approx(0.5)
    assert event.first_tested_alpha == pytest.approx(0.5)
    assert event.best_finite_alpha == pytest.approx(0.25)
    assert event.returned_alpha == pytest.approx(1.0)
    assert event.failure_reason == "line_search_failed"
    assert event.armijo_margin == pytest.approx(1.0e-6)
    assert event.curvature_margin == pytest.approx(-1.0e-3)


def test_minimize_lbfgs_host_core_nonfinite_step_terminates_status_nonfinite():
    def quad(x):
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * np.dot(x, x)), x

    def nonfinite_step_line_search(**_kwargs):
        return _host_lbfgs.HostLineSearchResults(
            failed=False,
            nit=1,
            nfev=1,
            ngev=1,
            k=2,
            a_k=0.5,
            f_k=0.25,
            g_k=np.asarray([np.inf], dtype=np.float64),
            status=0,
            requested_initial_step=0.5,
            first_tested_alpha=0.5,
            best_finite_alpha=0.5,
            returned_alpha=0.5,
            failure_reason="accepted",
            armijo_margin=0.0,
            curvature_margin=0.0,
        )

    result = _host_lbfgs.minimize_lbfgs_host_core(
        quad,
        np.asarray([1.0], dtype=np.float64),
        maxiter=2,
        initial_step_size=0.5,
        line_search_value_and_grad=nonfinite_step_line_search,
    )

    assert result.failed is True
    assert result.status == _host_lbfgs.LBFGS_STATUS_NONFINITE
    assert len(result.invalid_step_events) == 1
    assert result.invalid_step_events[0].nonfinite_step is True


def test_minimize_lbfgs_host_core_does_not_record_trace_by_default():
    def quad(x):
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * np.dot(x, x)), x

    result = _host_lbfgs.minimize_lbfgs_host_core(
        quad,
        np.asarray([1.0, -2.0], dtype=np.float64),
        maxiter=1,
    )

    assert result.optimizer_state_trace == ()


def test_optimizer_state_trace_memory_uses_stored_float64_arrays():
    expected_entry_bytes = (6 * 2 + 24) * np.dtype(np.float64).itemsize

    assert _host_lbfgs.optimizer_state_trace_memory_bytes(2, 3) == (
        3 * expected_entry_bytes
    )


def test_minimize_lbfgs_host_core_rejects_oversized_trace_budget():
    def quad(x):
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * np.dot(x, x)), x

    with pytest.raises(ValueError, match="optimizer_state_trace would allocate"):
        _host_lbfgs.minimize_lbfgs_host_core(
            quad,
            np.asarray([1.0, -2.0], dtype=np.float64),
            maxiter=2,
            record_optimizer_state_trace=True,
            max_optimizer_state_trace_bytes=1,
        )


def test_zoom_reuses_cached_bracketing_sample_without_extra_eval(monkeypatch):
    def _reuse_cached_cubic(_a, _fa, _fpa, _b, _fb, c, _fc):
        return c

    def _fresh_eval(_alpha):
        return (
            jnp.asarray(9.0, dtype=jnp.float64),
            jnp.asarray(4.0, dtype=jnp.float64),
            jnp.asarray([4.0], dtype=jnp.float64),
        )

    monkeypatch.setattr(_line_search_module, "_cubicmin", _reuse_cached_cubic)
    zoom = _line_search_module._zoom(
        _fresh_eval,
        lambda _alpha, _phi: jnp.asarray(False),
        lambda _dphi: jnp.asarray(True),
        jnp.asarray(1.5, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(-1.0, dtype=jnp.float64),
        jnp.asarray([-1.0], dtype=jnp.float64),
        jnp.asarray(2.0, dtype=jnp.float64),
        jnp.asarray(3.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray(True),
        jnp.asarray(1.25, dtype=jnp.float64),
        jnp.asarray(0.8, dtype=jnp.float64),
        jnp.asarray(-0.1, dtype=jnp.float64),
        jnp.asarray([-0.1], dtype=jnp.float64),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    )

    assert int(zoom.nfev) == 0
    assert int(zoom.ngev) == 0
    assert float(zoom.a_star) == pytest.approx(1.25)
    assert float(zoom.phi_star) == pytest.approx(0.8)


def test_bfgs_curvature_terms_reject_bad_curvature_updates():
    s_k = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    y_negative = jnp.asarray([-1.0e-3, 0.0], dtype=jnp.float64)
    y_near_orthogonal = jnp.asarray([1.0e-20, 1.0], dtype=jnp.float64)
    s_k_float32 = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    y_float32_boundary = jnp.asarray([1.0e-5, 1.0], dtype=jnp.float32)

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
    _, _, float32_boundary_valid, _ = _private_bfgs._bfgs_curvature_terms(
        s_k_float32,
        y_float32_boundary,
        x_dtype=s_k_float32.dtype,
    )

    assert bool(negative_valid) is False
    assert bool(near_orthogonal_valid) is False
    assert bool(float32_boundary_valid) is False


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

_ALL_JAX_BACKEND_MODES = (
    "jax_cpu_parity",
    "jax_gpu_parity",
    "jax_gpu_fast",
)


def _structured_optimizer_x0():
    return {
        "surface": jnp.array([1.0, -2.0], dtype=jnp.float64),
        "current": jnp.array([0.5], dtype=jnp.float64),
    }


def _assert_structured_zero_optimizer_result(result):
    assert isinstance(result.x, dict)
    assert isinstance(result.jac, dict)
    np.testing.assert_allclose(result.x["surface"], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(result.x["current"], np.zeros(1), atol=1e-12)
    np.testing.assert_allclose(result.jac["surface"], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(result.jac["current"], np.zeros(1), atol=1e-12)


def _emit_sparse_progress(progress_callback):
    progress_callback(1, 3.0, 2.0)
    progress_callback(7, 2.0, 1.0)
    progress_callback(25, 1.0, 0.5)


def test_two_loop_recursion_uses_history_count_not_iteration_count():
    g_k = np.asarray([3.0, -1.0], dtype=np.float64)
    gamma = 0.75
    s_history = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    y_history = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 4.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    rho_history = np.asarray([0.5, 0.25, 1.0e6], dtype=np.float64)

    direction = _host_lbfgs.two_loop_recursion_host(
        g_k,
        gamma,
        s_history,
        y_history,
        rho_history,
        history_count=2,
    )
    expected = _host_lbfgs.two_loop_recursion_host(
        g_k,
        gamma,
        s_history[:2],
        y_history[:2],
        rho_history[:2],
        history_count=2,
    )

    np.testing.assert_allclose(direction, expected, rtol=1e-12, atol=1e-12)


def test_two_loop_recursion_matches_materialized_history_after_wrap():
    g_k = np.asarray([3.0, -1.0], dtype=np.float64)
    gamma = 0.75
    s_oldest_to_newest = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.5, 0.5],
            [0.5, 1.5],
        ],
        dtype=np.float64,
    )
    y_oldest_to_newest = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 4.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )
    rho_oldest_to_newest = np.asarray([0.5, 0.25, 0.2, 0.2], dtype=np.float64)
    s_ring = np.asarray(
        [
            [1.5, 0.5],
            [0.5, 1.5],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    y_ring = np.asarray(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )
    rho_ring = np.asarray([0.2, 0.2, 0.5, 0.25], dtype=np.float64)

    ring_direction = _host_lbfgs.two_loop_recursion_host(
        g_k,
        gamma,
        s_ring,
        y_ring,
        rho_ring,
        history_count=6,
    )
    materialized_direction = _host_lbfgs.two_loop_recursion_host(
        g_k,
        gamma,
        s_oldest_to_newest,
        y_oldest_to_newest,
        rho_oldest_to_newest,
        history_count=4,
    )

    np.testing.assert_allclose(
        ring_direction,
        materialized_direction,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("backend_mode", _ALL_JAX_BACKEND_MODES)
@pytest.mark.parametrize(
    ("adapter_name", "objective_fn"),
    [
        ("_scipy_minimize", lambda x: jnp.sum(x**2)),
        (
            "_scipy_minimize_value_and_grad",
            lambda x: (jnp.sum(x**2), 2.0 * x),
        ),
    ],
)
def test_private_scipy_adapters_reject_all_jax_backend_modes(
    monkeypatch,
    request,
    backend_mode,
    adapter_name,
    objective_fn,
):
    enable_non_strict_jax_backend(monkeypatch, request, mode=backend_mode)

    def forbidden_scipy_minimize(*_args, **_kwargs):
        raise AssertionError("JAX backend modes must not enter scipy_minimize().")

    monkeypatch.setattr(_opt_ref, "scipy_minimize", forbidden_scipy_minimize)
    adapter = getattr(_opt_ref, adapter_name)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"{adapter_name}.*method='lbfgs'.*{backend_mode}.*requires an "
            r"ondevice optimizer method"
        ),
    ):
        adapter(
            objective_fn,
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs",
            tol=1e-8,
            maxiter=5,
            options={},
        )


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

        result = _line_search(
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
    def test_line_search_zoom_respects_total_eval_budget(self):
        """Zoom fallback must stay within the caller's total maxiter budget."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        result = _line_search(
            quad,
            jnp.array([1.0], dtype=jnp.float64),
            jnp.array([-1.95], dtype=jnp.float64),
            old_fval=jnp.array(0.5, dtype=jnp.float64),
            gfk=jnp.array([1.0], dtype=jnp.float64),
            maxiter=1,
        )

        assert bool(result.failed) is False
        assert int(result.nfev) == 1
        assert int(result.ngev) == 1
        assert float(result.f_k) < 0.5

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_line_search_promotes_integer_inputs_to_inexact_dtype(self):
        """The private line search must preserve the old inexact-promotion contract."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        result = _line_search(
            quad,
            jnp.array([2], dtype=jnp.int32),
            jnp.array([-1], dtype=jnp.int32),
            old_fval=jnp.array(2, dtype=jnp.int32),
            gfk=jnp.array([2], dtype=jnp.int32),
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

        monkeypatch.setattr(_private_bfgs, "_line_search", fake_line_search)

        state = _private_bfgs._minimize_bfgs_private(
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

        monkeypatch.setattr(_private_bfgs, "_line_search", fake_line_search)

        initial_state = _BFGSResults(
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

        state = _private_bfgs._minimize_bfgs_private(
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
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_minimize_lbfgs_private_preserves_float32_runtime_dtype(
        self,
        monkeypatch,
        request,
    ):
        """Float32 smoke lanes must not depend on implicit x64 truncation."""
        enable_non_strict_jax_backend(
            monkeypatch,
            request,
            mode="jax_cpu_float32_smoke",
        )

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        state = _opt._minimize_lbfgs_private(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float32),
            maxiter=10,
            gtol=1e-5,
            maxcor=5,
        )

        assert np.asarray(state.x_k).dtype == np.float32
        assert np.asarray(state.g_k).dtype == np.float32
        assert bool(state.failed) is False

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_callbacks_stay_transfer_clean_under_disallow(self):
        """Accepted-step callbacks must not trip strict transfer guard."""
        half = _device_half()

        def quad(x):
            return half * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        callback_points = []
        progress_points = []

        with jax.transfer_guard("disallow"):
            state = _opt._minimize_lbfgs_private(
                quad,
                x0,
                maxiter=10,
                gtol=1e-8,
                maxcor=5,
                callback=_record_host_arrays(callback_points),
                progress_callback=_record_progress(progress_points),
            )

        assert bool(state.converged) is True
        assert bool(state.failed) is False
        assert int(state.status) == 0
        assert callback_points
        assert progress_points
        np.testing.assert_allclose(callback_points[-1], np.zeros(2), atol=1e-12)
        assert progress_points[-1][0] >= 1

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_minimize_lbfgs_private_value_and_grad_emits_diagnostic_events(self):
        """Direct private L-BFGS must expose the diagnostic event stream."""
        half = _device_half()

        def quad_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x), x

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        progress_points = []
        diagnostic_events = []
        initial_value_and_grad = quad_value_and_grad(x0)
        state = _private_lbfgs._minimize_lbfgs_private_value_and_grad(
            quad_value_and_grad,
            x0,
            maxiter=1,
            gtol=1e-8,
            maxcor=5,
            progress_callback=_record_progress(progress_points),
            initial_value_and_grad=initial_value_and_grad,
            diagnostic_event_callback=(
                lambda label, **fields: diagnostic_events.append((label, fields))
            ),
        )

        assert int(state.k) <= 1
        assert progress_points
        assert [label for label, _ in diagnostic_events] == [
            "lbfgs_initial_state_started",
            "lbfgs_initial_state_returned",
            "lbfgs_initial_value_and_grad_seed_started",
            "lbfgs_initial_value_and_grad_seed_returned",
            "lbfgs_main_kernel_started",
            "lbfgs_main_kernel_returned",
            "lbfgs_effects_barrier_started",
            "lbfgs_effects_barrier_returned",
        ]
        assert diagnostic_events[0][1]["maxiter"] == 1
        assert diagnostic_events[0][1]["maxfun"] == 15000
        assert diagnostic_events[0][1]["maxcor"] == 1
        assert diagnostic_events[0][1]["maxls"] == 20
        assert diagnostic_events[0][1]["n"] == 2
        assert diagnostic_events[0][1]["workspace_bytes"] == (
            _private_lbfgs._lbfgsb_workspace_bytes(
                2,
                1,
                x0.dtype,
                record_optimizer_state_trace=False,
                maxiter_limit=1,
            )
        )
        assert diagnostic_events[0][1]["callback_enabled"] is True
        assert diagnostic_events[0][1]["record_optimizer_state_trace"] is False
        assert diagnostic_events[0][1]["run_mode"] == "stepwise"
        assert diagnostic_events[4][1]["accepted_step_callback"] is True
        assert diagnostic_events[4][1]["run_mode"] == "stepwise"

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_seed_transition_does_not_compile_full_setulb(self, monkeypatch):
        """The seeded START transition must avoid the full L-BFGS control loop."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state = _private_lbfgs.lbfgsb.lbfgsb_initial_state(
            x0,
            m=3,
            ftol=0.0,
            gtol=1e-8,
            maxls=20,
        )
        initial_value_and_grad = (
            jnp.asarray(2.5, dtype=jnp.float64),
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        )

        def fail_full_setulb(_state):
            raise AssertionError("seeded START path must not call lbfgsb_setulb")

        monkeypatch.setattr(_private_lbfgs.lbfgsb, "lbfgsb_setulb", fail_full_setulb)

        seeded = _private_lbfgs._lbfgsb_state_with_initial_value_and_grad(
            state,
            initial_value_and_grad,
            dtype=x0.dtype,
        )

        assert int(np.asarray(seeded.workspace.task[0])) == _private_lbfgs.lbfgsb.FG
        assert (
            int(np.asarray(seeded.workspace.task[1])) == _private_lbfgs.lbfgsb.FG_START
        )
        assert int(np.asarray(seeded.nfev)) == 1
        assert int(np.asarray(seeded.njev)) == 1
        assert float(np.asarray(seeded.f)) == pytest.approx(2.5)
        np.testing.assert_allclose(np.asarray(seeded.g), np.asarray([1.0, -2.0]))

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_macro_step_result_exposes_explicit_observable(self):
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state = _private_lbfgs.lbfgsb.lbfgsb_initial_state(
            x0,
            m=3,
            ftol=0.0,
            gtol=1e-8,
            maxls=20,
        )

        def quad_value_and_grad(x):
            return 0.5 * jnp.dot(x, x), x

        maxiter = 5
        maxfun = 10
        step_kernel = _private_lbfgs._lbfgsb_advance_to_next_observable_kernel(
            quad_value_and_grad,
            cache_owner=None,
            cache_key_prefix=(),
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
            entry_kind=_private_lbfgs._LBFGS_STEP_ENTRY_START,
        )
        result = step_kernel(state)

        task0 = int(np.asarray(result.state.workspace.task[0]))
        host_status = _private_lbfgs._lbfgsb_stepwise_host_status(result)
        expected_terminal = task0 >= _private_lbfgs.lbfgsb.CONVERGENCE

        assert task0 == _private_lbfgs.lbfgsb.NEW_X
        assert bool(np.asarray(result.terminal)) == expected_terminal
        assert host_status.terminal == expected_terminal
        assert host_status.entry_kind == _private_lbfgs._LBFGS_STEP_ENTRY_NEW_X_REENTRY

        reenter_kernel = _private_lbfgs._lbfgsb_advance_to_next_observable_kernel(
            quad_value_and_grad,
            cache_owner=None,
            cache_key_prefix=(),
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
            entry_kind=host_status.entry_kind,
        )
        next_result = reenter_kernel(result.state)
        next_host_status = _private_lbfgs._lbfgsb_stepwise_host_status(next_result)

        assert not bool(np.asarray(next_result.terminal))
        assert next_host_status.entry_kind == _private_lbfgs._LBFGS_STEP_ENTRY_SEARCH

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_skips_debug_callback_without_observability(
        self,
        monkeypatch,
    ):
        """Default solve must not invoke jax.debug.callback when no callbacks wired.

        This is the CUDA-only correctness contract: the solver's hot loop must be
        safe on ``JAX_PLATFORMS=cuda`` (no CPU backend), which requires zero host
        callback traffic when the caller omits observability hooks.
        """
        observed = {"called": False}

        def forbidden_debug_callback(*_args, **_kwargs):
            observed["called"] = True
            raise AssertionError(
                "jax.debug.callback must not run when no observability callbacks "
                "are wired to the private L-BFGS solver."
            )

        monkeypatch.setattr(jax.debug, "callback", forbidden_debug_callback)

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state = _opt._minimize_lbfgs_private(
            quad,
            x0,
            maxiter=5,
            gtol=1e-8,
            maxcor=5,
        )

        assert observed["called"] is False
        assert bool(state.converged) is True
        assert int(state.status) == 0

    def test_lbfgsb_private_solver_wrappers_are_cached_without_observers(self):
        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def quad_value_and_grad(x):
            return quad(x), jnp.asarray(x, dtype=x.dtype)

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        cacheable_quad = _opt._mark_cacheable_jit_value_and_grad(quad_value_and_grad)
        cache_owner, cache_key_prefix = _private_lbfgs._lbfgsb_cache_context(
            cacheable_quad,
            None,
            "value_and_grad",
            x0.dtype,
            x0.shape,
        )
        assert cache_owner is cacheable_quad

        initial_a = _private_lbfgs._lbfgsb_initial_state_kernel(
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            m=10,
            ftol=0.0,
            gtol=1e-5,
            maxls=20,
        )
        initial_b = _private_lbfgs._lbfgsb_initial_state_kernel(
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            m=10,
            ftol=0.0,
            gtol=1e-5,
            maxls=20,
        )
        assert initial_a is initial_b

        value_and_grad, _value_and_grad_consts = (
            _private_lbfgs._cached_lbfgs_value_and_grad_kernel(
                cacheable_quad,
                cache_owner=cacheable_quad,
                adapter=None,
                objective_mode="value_and_grad",
                dtype=x0.dtype,
                example_x=x0,
            )
        )
        step_a = _private_lbfgs._lbfgsb_advance_to_next_observable_kernel(
            value_and_grad,
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            maxiter=5,
            maxfun=5,
            accepted_step_callback=None,
        )
        step_b = _private_lbfgs._lbfgsb_advance_to_next_observable_kernel(
            value_and_grad,
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            maxiter=5,
            maxfun=5,
            accepted_step_callback=None,
        )
        result_payload_a = _private_lbfgs._lbfgsb_result_payload_kernel(
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            maxiter=5,
            maxfun=5,
        )
        result_payload_b = _private_lbfgs._lbfgsb_result_payload_kernel(
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            maxiter=5,
            maxfun=5,
        )
        observed = _private_lbfgs._lbfgsb_advance_to_next_observable_kernel(
            value_and_grad,
            cache_owner=cache_owner,
            cache_key_prefix=cache_key_prefix,
            maxiter=5,
            maxfun=5,
            accepted_step_callback=lambda *_args: None,
        )
        assert step_a is step_b
        assert result_payload_a is result_payload_b
        assert observed is not step_a

    def test_newton_linear_product_jit_helpers_cache_marked_callables(self):
        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def residual(x):
            return jnp.asarray([x[0] + 2.0 * x[1], x[0] - x[1]], dtype=x.dtype)

        cacheable_quad = _opt._mark_cacheable_jit_value_and_grad(quad)
        cacheable_residual = _opt._mark_cacheable_jit_linear_operator(residual)

        assert _opt._hessian_vector_product_fn(cacheable_quad) is (
            _opt._hessian_vector_product_fn(cacheable_quad)
        )
        assert _opt._jacobian_vector_product_fn(cacheable_residual) is (
            _opt._jacobian_vector_product_fn(cacheable_residual)
        )

    def test_hvp_objective_remat_default_off_returns_original_callable(
        self,
        monkeypatch,
    ):
        def objective_fn(x):
            return jnp.sum(x * x)

        monkeypatch.setattr(_opt, "_HVP_OBJECTIVE_REMAT", False)

        assert _opt._checkpoint_hvp_objective(objective_fn) is objective_fn

    @pytest.mark.parametrize(
        "policy",
        ["default", "dots", "dots_with_no_batch_dims_saveable"],
    )
    def test_hvp_objective_remat_policies_preserve_hvp_values(
        self,
        monkeypatch,
        policy,
    ):
        x = jnp.asarray([0.2, -0.5, 0.75], dtype=jnp.float64)
        tangent = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
        scale = jnp.asarray(1.75, dtype=jnp.float64)
        shift = jnp.asarray(-0.125, dtype=jnp.float64)

        def reference_objective(x_inner, objective_scale, objective_shift):
            shifted = x_inner + objective_shift
            return objective_scale * jnp.sum(jnp.sin(shifted) * x_inner * x_inner)

        def remat_objective(x_inner, objective_scale, objective_shift):
            shifted = x_inner + objective_shift
            return objective_scale * jnp.sum(jnp.sin(shifted) * x_inner * x_inner)

        monkeypatch.setattr(_opt, "_HVP_OBJECTIVE_REMAT", False)
        reference_hvp = _opt._hessian_vector_product_fn(reference_objective)

        monkeypatch.setattr(_opt, "_HVP_OBJECTIVE_REMAT", True)
        monkeypatch.setattr(_opt, "_HVP_OBJECTIVE_REMAT_POLICY", policy)
        remat_hvp = _opt._hessian_vector_product_fn(remat_objective)

        np.testing.assert_allclose(
            np.asarray(remat_hvp(x, tangent, scale, shift)),
            np.asarray(reference_hvp(x, tangent, scale, shift)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_hvp_objective_remat_rejects_unknown_policy(self, monkeypatch):
        monkeypatch.setattr(_opt, "_HVP_OBJECTIVE_REMAT", True)
        monkeypatch.setattr(_opt, "_HVP_OBJECTIVE_REMAT_POLICY", "unknown")

        with pytest.raises(ValueError, match="SIMSOPT_HVP_OBJECTIVE_REMAT_POLICY"):
            _opt._checkpoint_hvp_objective(lambda x: jnp.sum(x))

    def test_lbfgsb_step_kernel_omits_partial_state_donation(self, monkeypatch):
        observed = {}

        def fake_jit(fn, *jit_args, **jit_kwargs):
            del jit_args
            observed["donate_argnums"] = jit_kwargs.get("donate_argnums")
            return fn

        monkeypatch.setattr(_private_lbfgs.jax, "jit", fake_jit)

        kernel = _private_lbfgs._lbfgsb_advance_to_next_observable_kernel(
            lambda x: (jnp.asarray(0.0), jnp.zeros_like(x.x)),
            cache_owner=None,
            cache_key_prefix=(),
            maxiter=1,
            maxfun=1,
            accepted_step_callback=None,
        )

        assert callable(kernel)
        assert observed["donate_argnums"] is None

    def test_hybrid_method_is_removed_from_public_optimizer_surface(self):
        with pytest.raises(ValueError, match="Unknown method 'bfgs-hybrid'"):
            jax_minimize(
                lambda x: jnp.sum(x**2),
                jnp.array([1.0, -1.0], dtype=jnp.float64),
                method="bfgs-hybrid",
                maxiter=8,
            )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_bfgs_ondevice_accepts_pytree_x0_and_restores_result_structure(self):
        """bfgs-ondevice should flatten pytrees internally and restore them on return."""

        def quad(state):
            return 0.5 * (
                jnp.dot(state["surface"], state["surface"])
                + jnp.dot(state["current"], state["current"])
            )

        x0 = _structured_optimizer_x0()
        callback_calls = []

        result = jax_minimize(
            quad,
            x0,
            method="bfgs-ondevice",
            maxiter=5,
            callback=lambda state: callback_calls.append(state),
        )

        _assert_structured_zero_optimizer_result(result)
        assert callback_calls
        assert isinstance(callback_calls[0], dict)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_target_minimize_pytree_path_skips_public_flattening_adapter(
        self, monkeypatch
    ):
        """target_minimize() should leave pytree flattening to the private solver."""

        def quad(state):
            return 0.5 * (
                jnp.dot(state["surface"], state["surface"])
                + jnp.dot(state["current"], state["current"])
            )

        x0 = _structured_optimizer_x0()
        callback_calls = []

        def forbid_public_flattening(*_args, **_kwargs):
            raise AssertionError(
                "target_minimize() should not pre-flatten pytrees in the "
                "public target entrypoint."
            )

        monkeypatch.setattr(
            _opt,
            "_prepare_optimizer_callable_inputs",
            forbid_public_flattening,
        )

        result = _opt.target_minimize(
            quad,
            x0,
            method="lbfgs-ondevice",
            maxiter=10,
            callback=lambda state: callback_calls.append(state),
        )

        _assert_structured_zero_optimizer_result(result)
        assert callback_calls
        assert isinstance(callback_calls[0], dict)

    def test_target_minimize_lbfgs_defaults_ftol_to_tol(self, monkeypatch):
        """The target L-BFGS route should mirror SciPy's tol -> ftol contract."""
        captured = {}
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)

        def fake_minimize(_fun, _x0, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", fake_minimize)
        monkeypatch.setattr(
            _opt,
            "_private_lbfgs_result_to_optimize_result",
            lambda _state: _successful_optimize_result_for_x(x0),
        )

        _opt.target_minimize(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            method="lbfgs-ondevice",
            tol=1.0e-4,
            maxiter=3,
        )

        assert captured["gtol"] == pytest.approx(1.0e-4)
        assert captured["ftol"] == pytest.approx(1.0e-4)
        assert captured["run_mode"] == "stepwise"

    def test_target_minimize_lbfgs_explicit_ftol_overrides_tol(self, monkeypatch):
        """An explicit L-BFGS ftol option should remain independent from gtol."""
        captured = {}
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)

        def fake_minimize(_fun, _x0, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            _opt, "_minimize_lbfgs_private_value_and_grad", fake_minimize
        )
        monkeypatch.setattr(
            _opt,
            "_private_lbfgs_result_to_optimize_result",
            lambda _state: _successful_optimize_result_for_x(x0),
        )

        _opt.target_minimize(
            lambda x: (0.5 * jnp.dot(x, x), x),
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
            tol=1.0e-4,
            maxiter=3,
            options={"ftol": 1.0e-7, "lbfgs_run_mode": "monolithic_debug"},
        )

        assert captured["gtol"] == pytest.approx(1.0e-4)
        assert captured["ftol"] == pytest.approx(1.0e-7)
        assert captured["run_mode"] == "monolithic_debug"

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_accepts_pytree_x0(self):
        """Direct private L-BFGS entry should flatten pytrees before runtime checks."""

        def quad(state):
            return 0.5 * (
                jnp.dot(state["surface"], state["surface"])
                + jnp.dot(state["current"], state["current"])
            )

        x0 = _structured_optimizer_x0()
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
    def test_minimize_lbfgs_private_pytree_callback_stays_on_host(self):
        """Structured L-BFGS callbacks should not device-stage accepted host state."""

        def quad(state):
            return 0.5 * (
                jnp.dot(state["surface"], state["surface"])
                + jnp.dot(state["current"], state["current"])
            )

        callback_states = []

        def callback(state):
            callback_states.append(
                {
                    "surface": np.asarray(state["surface"], dtype=float),
                    "current": np.asarray(state["current"], dtype=float),
                }
            )

        state = _opt._minimize_lbfgs_private(
            quad,
            _structured_optimizer_x0(),
            maxiter=10,
            gtol=1e-8,
            callback=callback,
        )

        assert callback_states
        np.testing.assert_allclose(
            callback_states[-1]["surface"],
            np.zeros(2),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            callback_states[-1]["current"],
            np.zeros(1),
            atol=1e-12,
        )
        np.testing.assert_allclose(np.asarray(state.g_k), np.zeros(3), atol=1e-12)

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
        """A NaN trial must shrink/fail without poisoning the last finite iterate."""

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
        assert result.status == 2
        assert result.message == "Insufficient progress."

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
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_does_not_call_custom_host_core(self, monkeypatch):
        """lbfgs-ondevice must run the SciPy-compatible JAX state machine."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def reject_host_core(*_args, **_kwargs):
            raise AssertionError("lbfgs-ondevice called minimize_lbfgs_host_core")

        monkeypatch.setattr(
            _private_lbfgs,
            "minimize_lbfgs_host_core",
            reject_host_core,
            raising=False,
        )

        result = jax_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=5,
        )

        assert result.success is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_does_not_call_scipy_minimize(self, monkeypatch):
        """lbfgs-ondevice must not silently route target execution to SciPy."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def reject_scipy_minimize(*_args, **_kwargs):
            raise AssertionError("lbfgs-ondevice called scipy_minimize")

        monkeypatch.setattr(_opt_ref, "scipy_minimize", reject_scipy_minimize)

        result = jax_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=5,
        )

        assert result.success is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_defaults_to_stepwise_driver(self, monkeypatch):
        """Public lbfgs-ondevice must not use the legacy full-solve compile path."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        def reject_legacy_mainlb(*_args, **_kwargs):
            raise AssertionError("default lbfgs-ondevice called legacy mainlb kernel")

        monkeypatch.setattr(
            _private_lbfgs,
            "_lbfgsb_mainlb_kernel",
            reject_legacy_mainlb,
        )

        result = jax_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=5,
        )

        assert result.success is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_rejects_nonpositive_maxls_like_scipy(self):
        """Public lbfgs-ondevice must preserve SciPy's maxls contract."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        with pytest.raises(ValueError, match="maxls must be positive"):
            jax_minimize(
                quad,
                jnp.array([1.0, -2.0], dtype=jnp.float64),
                method="lbfgs-ondevice",
                maxiter=5,
                options={"maxls": 0},
            )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_BUDGET_RUNTIME
    def test_lbfgs_ondevice_zero_iteration_budget_matches_scipy_deferred_stop(
        self,
    ):
        """SciPy checks maxiter after the first accepted NEW_X iteration."""

        def jax_quad(x):
            return 0.5 * jnp.dot(x, x)

        def scipy_quad(x):
            return np.float64(0.5 * np.dot(x, x)), np.asarray(x, dtype=np.float64)

        x0 = np.array([1.0, -2.0], dtype=np.float64)
        scipy_result = optimize.minimize(
            scipy_quad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 0},
        )

        result = jax_minimize(
            jax_quad,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=0,
        )

        np.testing.assert_allclose(np.asarray(result.x), scipy_result.x, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.jac), scipy_result.jac, atol=1e-12)
        assert result.nit == scipy_result.nit
        assert result.nfev == scipy_result.nfev
        assert result.njev == scipy_result.njev
        assert result.status == scipy_result.status
        assert result.success is scipy_result.success

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_BUDGET_RUNTIME
    def test_lbfgs_ondevice_maxfun_budget_matches_scipy_after_line_search(self):
        """SciPy checks maxfun only after an accepted NEW_X step."""

        def jax_rosenbrock(x):
            return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

        def scipy_rosenbrock(x):
            value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
            grad = np.asarray(
                [
                    -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                    200.0 * (x[1] - x[0] ** 2),
                ],
                dtype=np.float64,
            )
            return np.float64(value), grad

        x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
        options = {"maxiter": 20, "maxfun": 4, "maxcor": 4, "ftol": 0.0, "gtol": 1e-8}
        scipy_result = optimize.minimize(
            scipy_rosenbrock,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        result = jax_minimize(
            jax_rosenbrock,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            tol=options["gtol"],
            maxiter=options["maxiter"],
            options={
                "maxfun": options["maxfun"],
                "maxcor": options["maxcor"],
                "ftol": options["ftol"],
            },
        )

        np.testing.assert_allclose(np.asarray(result.x), scipy_result.x, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.jac), scipy_result.jac, atol=1e-12)
        assert result.fun == pytest.approx(float(scipy_result.fun), abs=1e-12)
        assert result.nit == scipy_result.nit
        assert result.nfev == scipy_result.nfev
        assert result.njev == scipy_result.njev
        assert result.status == scipy_result.status
        assert result.success is scipy_result.success

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_BUDGET_RUNTIME
    def test_lbfgs_ondevice_maxls_exhaustion_matches_scipy_public_result(self):
        """A maxls-exhausted line search must preserve SciPy's public status."""

        def jax_rosenbrock(x):
            return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

        def scipy_rosenbrock(x):
            value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
            grad = np.asarray(
                [
                    -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                    200.0 * (x[1] - x[0] ** 2),
                ],
                dtype=np.float64,
            )
            return np.float64(value), grad

        x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
        options = {"maxiter": 20, "maxcor": 4, "ftol": 0.0, "gtol": 1e-8, "maxls": 1}
        scipy_result = optimize.minimize(
            scipy_rosenbrock,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        result = jax_minimize(
            jax_rosenbrock,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            tol=options["gtol"],
            maxiter=options["maxiter"],
            options={
                "maxcor": options["maxcor"],
                "ftol": options["ftol"],
                "maxls": options["maxls"],
            },
        )

        assert scipy_result.status == 2
        assert result.status == scipy_result.status
        assert result.message == scipy_result.message
        assert result.nit == scipy_result.nit
        assert result.nfev == scipy_result.nfev
        assert result.njev == scipy_result.njev
        assert result.success is scipy_result.success
        np.testing.assert_allclose(np.asarray(result.x), scipy_result.x, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.jac), scipy_result.jac, atol=1e-12)
        assert result.fun == pytest.approx(float(scipy_result.fun), abs=1e-12)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_unconstrained_fast_path_skips_bounded_geometry(
        self,
        monkeypatch,
    ):
        """The public no-bounds lane must not trace bounded L-BFGS-B geometry."""

        def fail_bounded_geometry(*_args, **_kwargs):
            raise AssertionError("unconstrained lbfgs-ondevice traced bounded geometry")

        monkeypatch.setattr(
            _private_lbfgs.lbfgsb, "lbfgsb_cauchy", fail_bounded_geometry
        )
        monkeypatch.setattr(
            _private_lbfgs.lbfgsb, "lbfgsb_formk", fail_bounded_geometry
        )
        monkeypatch.setattr(
            _private_lbfgs.lbfgsb, "lbfgsb_subsm", fail_bounded_geometry
        )

        matrix = jnp.asarray(
            [[3.0, 0.25], [0.25, 1.75]],
            dtype=jnp.float64,
        )
        linear = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

        def shifted_quad(x):
            return 0.5 * x @ (matrix @ x) + linear @ x

        result = jax_minimize(
            shifted_quad,
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=5,
            options={"maxcor": 3, "ftol": 0.0, "gtol": 1e-10},
        )

        assert result.nit > 0
        assert result.nfev == result.njev
        assert result.status in {0, 1}

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_live_quadratic_matches_scipy_public_result(self):
        """CPU JAX live objective must match SciPy on a pinned quadratic."""

        def jax_quad(x):
            return 0.5 * jnp.dot(x, x)

        def scipy_quad(x):
            return np.float64(0.5 * np.dot(x, x)), np.asarray(x, dtype=np.float64)

        x0 = np.array([1.0, -2.0], dtype=np.float64)
        options = {"maxiter": 5, "maxcor": 4, "ftol": 0.0, "gtol": 1e-8, "maxls": 20}
        scipy_result = optimize.minimize(
            scipy_quad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        result = jax_minimize(
            jax_quad,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            tol=options["gtol"],
            maxiter=options["maxiter"],
            options={
                "maxcor": options["maxcor"],
                "ftol": options["ftol"],
                "maxls": options["maxls"],
            },
        )

        np.testing.assert_allclose(np.asarray(result.x), scipy_result.x, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.jac), scipy_result.jac, atol=1e-12)
        assert result.fun == pytest.approx(float(scipy_result.fun), abs=1e-12)
        assert result.nit == scipy_result.nit
        assert result.nfev == scipy_result.nfev
        assert result.njev == scipy_result.njev
        assert result.status == scipy_result.status
        assert result.success is scipy_result.success

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_returns_scipy_style_hess_inv(self):
        """Public lbfgs-ondevice results expose SciPy-compatible hess_inv."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        result = jax_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=5,
            options={"maxcor": 4},
        )

        assert hasattr(result, "hess_inv")
        assert result.hess_inv.shape == (2, 2)
        assert result.hess_inv(np.array([1.0, 0.0], dtype=np.float64)).shape == (2,)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_hess_inv_matches_scipy_after_maxcor_rollover(self):
        """Limited-memory inverse-Hessian history must match SciPy after rollover."""

        matrix = np.diag(np.asarray([1.0, 2.0, 3.0], dtype=np.float64)) + 0.2
        linear = np.asarray([0.5, 1.5, 2.5], dtype=np.float64)

        def scipy_quad(x):
            x = np.asarray(x, dtype=np.float64)
            return np.float64(0.5 * x @ matrix @ x + linear @ x), matrix @ x + linear

        matrix_jax = jnp.asarray(matrix, dtype=jnp.float64)
        linear_jax = jnp.asarray(linear, dtype=jnp.float64)

        def jax_quad(x):
            return 0.5 * x @ (matrix_jax @ x) + linear_jax @ x

        x0 = np.asarray([2.0, 0.5, -1.0], dtype=np.float64)
        options = {"maxiter": 8, "maxcor": 2, "ftol": 0.0, "gtol": 1e-12, "maxls": 20}
        scipy_result = optimize.minimize(
            scipy_quad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        result = jax_minimize(
            jax_quad,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            tol=options["gtol"],
            maxiter=options["maxiter"],
            options={
                "maxcor": options["maxcor"],
                "ftol": options["ftol"],
                "maxls": options["maxls"],
            },
        )

        vector = np.asarray([-0.3, 0.2, 0.7], dtype=np.float64)
        assert scipy_result.hess_inv.n_corrs == options["maxcor"]
        assert result.hess_inv.n_corrs == options["maxcor"]
        np.testing.assert_allclose(np.asarray(result.x), scipy_result.x, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.jac), scipy_result.jac, atol=1e-12)
        np.testing.assert_allclose(
            result.hess_inv(vector),
            scipy_result.hess_inv(vector),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result.hess_inv.todense(),
            scipy_result.hess_inv.todense(),
            rtol=1e-10,
            atol=1e-10,
        )

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
        assert result.optimizer_state_trace == ()

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_records_trace_only_when_requested(self):
        """lbfgs-ondevice optimizer_state_trace is diagnostic-only."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        result = jax_minimize(
            quad,
            x0,
            method="lbfgs-ondevice",
            maxiter=5,
            options={"record_optimizer_state_trace": True},
        )

        assert result.success is True
        assert len(result.optimizer_state_trace) == result.nit
        assert result.optimizer_state_trace[0]["iteration"] == 1

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_rejects_oversized_trace_budget(self):
        """Requested lbfgs-ondevice traces must be bounded before compilation."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        with pytest.raises(ValueError, match="optimizer_state_trace would allocate"):
            jax_minimize(
                quad,
                jnp.array([1.0, -2.0], dtype=jnp.float64),
                method="lbfgs-ondevice",
                maxiter=5,
                options={
                    "record_optimizer_state_trace": True,
                    "max_optimizer_state_trace_bytes": 1,
                },
            )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_accepts_explicit_value_and_grad(self):
        """lbfgs-ondevice must support explicit value/grad objectives."""
        half = _device_half()

        def quad_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x), x

        callback_calls = []
        progress_calls = []
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        with jax.transfer_guard("disallow"):
            result = jax_minimize(
                quad_value_and_grad,
                x0,
                method="lbfgs-ondevice",
                maxiter=5,
                value_and_grad=True,
                callback=_record_host_arrays(callback_calls, dtype=float),
                progress_callback=_record_progress(progress_calls),
            )

        assert result.success is True
        assert result.nit > 0
        assert float(result.fun) < quad_value_and_grad(np.asarray(x0))[0]
        assert len(callback_calls) == result.nit
        assert len(progress_calls) == result.nit

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_gpu_closure_constants_run_under_strict_transfer_guard(
        self,
    ):
        try:
            device = jax.devices("gpu")[0]
        except RuntimeError:
            pytest.skip("CUDA device required for device-to-host transfer-guard proof.")
        dtype = np.float64
        x0 = jax.device_put(np.asarray([0.0, 0.0], dtype=dtype), device)
        target = jax.device_put(np.asarray([1.0, -2.0], dtype=dtype), device)
        active = jax.device_put(np.asarray(True), device)
        two = jax.device_put(np.asarray(2.0, dtype=dtype), device)

        def value_and_grad(x):
            residual = jnp.where(active, x - target, x)
            return jnp.vdot(residual, residual), two * residual

        with jax.transfer_guard("disallow"):
            result = jax_minimize(
                value_and_grad,
                x0,
                method="lbfgs-ondevice",
                maxiter=1,
                value_and_grad=True,
            )

        assert result.x.shape == (2,)
        assert result.nit <= 1

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_value_and_grad_kernel_accepts_traced_example_sharding(self):
        target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        scale = jnp.asarray(0.5, dtype=jnp.float64)

        def value_and_grad(x):
            residual = x - target
            return scale * jnp.vdot(residual, residual), 2.0 * scale * residual

        def build_and_call_with_traced_example(x):
            value_and_grad_kernel, value_and_grad_consts = (
                _private_lbfgs._cached_lbfgs_value_and_grad_kernel(
                    value_and_grad,
                    cache_owner=None,
                    adapter=None,
                    objective_mode="value_and_grad",
                    dtype=x.dtype,
                    example_x=x,
                )
            )
            return value_and_grad_kernel(x, value_and_grad_consts)

        x0 = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        value, grad = jax.jit(build_and_call_with_traced_example)(x0)

        np.testing.assert_allclose(value, 2.5, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(grad),
            np.asarray([-1.0, 2.0]),
            rtol=1e-14,
            atol=1e-14,
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_value_and_grad_kernel_places_scalar_consts_on_named_sharding(self):
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("x",))
        vector_sharding = NamedSharding(mesh, P("x"))
        target = jax.device_put(
            np.asarray([1.0, -2.0], dtype=np.float64),
            vector_sharding,
        )
        scale = jnp.asarray(0.5, dtype=jnp.float64)

        def value_and_grad(x):
            residual = x - target
            return scale * jnp.vdot(residual, residual), 2.0 * scale * residual

        def build_and_call_with_traced_example(x):
            value_and_grad_kernel, value_and_grad_consts = (
                _private_lbfgs._cached_lbfgs_value_and_grad_kernel(
                    value_and_grad,
                    cache_owner=None,
                    adapter=None,
                    objective_mode="value_and_grad",
                    dtype=x.dtype,
                    example_x=x,
                )
            )
            return value_and_grad_kernel(x, value_and_grad_consts)

        x0 = jax.device_put(
            np.asarray([0.0, 0.0], dtype=np.float64),
            vector_sharding,
        )
        value, grad = jax.jit(build_and_call_with_traced_example)(x0)

        np.testing.assert_allclose(value, 2.5, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(grad),
            np.asarray([-1.0, 2.0]),
            rtol=1e-14,
            atol=1e-14,
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_minimize_lbfgs_private_rejects_stepwise_inside_jit(self):
        def quad(x):
            return 0.5 * jnp.vdot(x, x)

        def run_traced_lbfgs(x):
            result = _private_lbfgs._minimize_lbfgs_private(
                quad,
                x,
                maxiter=1,
            )
            return result.x_k

        with pytest.raises(ValueError, match="stepwise mode uses host-observed"):
            jax.jit(run_traced_lbfgs)(jnp.asarray([1.0, -2.0], dtype=jnp.float64))

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_minimize_lbfgs_private_monolithic_debug_runs_inside_jit(self):
        target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

        def quad(x):
            residual = x - target
            return 0.5 * jnp.vdot(residual, residual)

        def run_traced_lbfgs(x):
            result = _private_lbfgs._minimize_lbfgs_private(
                quad,
                x,
                maxiter=3,
                gtol=1e-12,
                run_mode="monolithic_debug",
            )
            return result.x_k, result.f_k, result.k

        x, value, iterations = jax.jit(run_traced_lbfgs)(
            jnp.asarray([0.0, 0.0], dtype=jnp.float64)
        )

        assert int(iterations) > 0
        assert float(value) < 2.5
        np.testing.assert_allclose(
            np.asarray(x),
            np.asarray([1.0, -2.0]),
            rtol=1e-6,
            atol=1e-6,
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_callbacks_track_accepted_new_x_count_after_step_split(
        self,
    ):
        """Callback/progress streams are accepted-step streams, not FG probes."""
        half = _device_half()

        def quad_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x), x

        callback_calls = []
        progress_calls = []
        result = jax_minimize(
            quad_value_and_grad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method="lbfgs-ondevice",
            maxiter=1,
            value_and_grad=True,
            callback=_record_host_arrays(callback_calls, dtype=float),
            progress_callback=_record_progress(progress_calls),
        )

        assert result.nit == 1
        assert len(callback_calls) == result.nit
        assert len(progress_calls) == result.nit

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_callback_stop_iteration_matches_scipy_status(
        self,
    ):
        """StopIteration from an accepted-step callback must halt like SciPy."""
        half = _device_half()
        x0 = np.asarray([1.0, -2.0], dtype=np.float64)

        def jax_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return half * jnp.dot(x, x), x

        def scipy_value_and_grad(x):
            return np.float64(0.5 * np.dot(x, x)), np.asarray(x, dtype=np.float64)

        scipy_calls = []

        def scipy_callback(intermediate_result=None):
            scipy_calls.append(np.asarray(intermediate_result.x, dtype=float))
            raise StopIteration

        scipy_result = optimize.minimize(
            scipy_value_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            callback=scipy_callback,
            options={"maxiter": 5},
        )

        callback_calls = []
        progress_calls = []

        def callback(x):
            callback_calls.append(np.asarray(x, dtype=float))
            raise StopIteration

        with jax.transfer_guard("disallow"):
            result = jax_minimize(
                jax_value_and_grad,
                jnp.asarray(x0, dtype=jnp.float64),
                method="lbfgs-ondevice",
                maxiter=5,
                value_and_grad=True,
                callback=callback,
                progress_callback=_record_progress(progress_calls),
            )

        assert result.success is scipy_result.success
        assert result.status == scipy_result.status == 99
        assert result.message == scipy_result.message
        assert result.nit == scipy_result.nit == len(callback_calls)
        assert result.nfev == scipy_result.nfev
        assert result.njev == scipy_result.njev
        assert len(scipy_calls) == len(callback_calls)
        assert progress_calls == []
        np.testing.assert_allclose(result.x, scipy_result.x, rtol=1e-12, atol=1e-12)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_and_optax_lbfgs_are_distinct_contracts(self):
        """Optax L-BFGS is not a SciPy L-BFGS-B parity oracle."""

        def rosenbrock(x):
            return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

        def record_callback(events):
            def callback(x):
                events.append(np.asarray(x, dtype=float))

            return callback

        def record_progress(events):
            def progress(iteration, fun, grad_norm):
                events.append((int(iteration), float(fun), float(grad_norm)))

            return progress

        x0 = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
        options = {"maxcor": 3, "maxls": 5, "ftol": 0.0, "gtol": 1e-8}
        scipy_callback_events = []
        scipy_progress_events = []
        scipy_style = jax_minimize(
            rosenbrock,
            x0,
            method="lbfgs-ondevice",
            maxiter=2,
            options=options,
            callback=record_callback(scipy_callback_events),
            progress_callback=record_progress(scipy_progress_events),
        )
        optax_callback_events = []
        optax_progress_events = []
        optax_style = jax_minimize(
            rosenbrock,
            x0,
            method="optax-lbfgs-ondevice",
            maxiter=2,
            options=options,
            callback=record_callback(optax_callback_events),
            progress_callback=record_progress(optax_progress_events),
        )

        assert hasattr(scipy_style, "hess_inv")
        assert not hasattr(optax_style, "hess_inv")
        assert scipy_style.status == optax_style.status == 1
        assert scipy_style.nit == optax_style.nit == 2
        assert scipy_style.nfev == scipy_style.njev
        assert optax_style.nfev == optax_style.njev
        assert scipy_style.nfev != optax_style.nfev
        assert float(scipy_style.fun) != pytest.approx(float(optax_style.fun))
        assert not np.allclose(np.asarray(scipy_style.x), np.asarray(optax_style.x))
        assert len(scipy_callback_events) == scipy_style.nit
        assert len(optax_callback_events) == optax_style.nit
        assert len(scipy_progress_events) == scipy_style.nit
        assert len(optax_progress_events) == optax_style.nit
        assert not np.allclose(
            np.asarray(scipy_callback_events),
            np.asarray(optax_callback_events),
        )
        assert scipy_progress_events != pytest.approx(optax_progress_events)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_seeded_converged_entry_reuses_seed_without_extra_evals(
        self,
    ):
        """A converged seed must keep eval counters exact on the zero-iteration path."""
        x0 = jnp.zeros((2,), dtype=jnp.float64)
        optimizer_seed = (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.zeros_like(x0),
        )

        def quad_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return 0.5 * jnp.dot(x, x), x

        result = _opt.target_minimize(
            quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            maxiter=5,
            value_and_grad=True,
            initial_value_and_grad=optimizer_seed,
        )

        assert result.success is True
        assert result.nit == 0
        assert result.status == 0
        assert result.nfev == 1
        assert result.njev == 1
        np.testing.assert_allclose(np.asarray(result.x), np.asarray(x0))
        np.testing.assert_allclose(
            np.asarray(result.jac), np.asarray(optimizer_seed[1])
        )
        assert float(result.fun) == pytest.approx(0.0)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_seeded_first_fg_matches_unseeded_result(self):
        """An explicit seed must stand in for SciPy's first FG request only."""
        scale = jnp.asarray([1.0, 3.0], dtype=jnp.float64)

        def quad_value_and_grad(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return 0.5 * jnp.dot(scale * x, x), scale * x

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        seed = quad_value_and_grad(x0)
        options = {"maxcor": 4, "ftol": 0.0, "maxls": 20}
        seeded = _opt.target_minimize(
            quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
            initial_value_and_grad=seed,
            tol=1e-8,
            maxiter=5,
            options=options,
        )
        unseeded = _opt.target_minimize(
            quad_value_and_grad,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
            tol=1e-8,
            maxiter=5,
            options=options,
        )

        np.testing.assert_allclose(np.asarray(seeded.x), np.asarray(unseeded.x))
        np.testing.assert_allclose(np.asarray(seeded.jac), np.asarray(unseeded.jac))
        assert float(seeded.fun) == pytest.approx(float(unseeded.fun))
        assert seeded.nit == unseeded.nit
        assert seeded.nfev == unseeded.nfev
        assert seeded.njev == unseeded.njev
        assert seeded.status == unseeded.status
        assert seeded.success == unseeded.success

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

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_reports_scipy_status_for_nonfinite_initial_objective(
        self,
    ):
        """Entry with a NaN initial objective must match SciPy's task status."""

        def jax_nan_at_origin(x):
            return jnp.where(
                jnp.all(jnp.equal(x, jnp.zeros_like(x))),
                jnp.asarray(jnp.nan, dtype=x.dtype),
                0.5 * jnp.dot(x, x),
            )

        def scipy_nan_at_origin(x):
            if np.all(x == 0.0):
                return np.float64(np.nan), np.zeros_like(x)
            return np.float64(0.5 * np.dot(x, x)), np.asarray(x, dtype=np.float64)

        x0 = np.zeros((2,), dtype=np.float64)
        scipy_result = optimize.minimize(
            scipy_nan_at_origin,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5, "gtol": 1e-8},
        )
        result = jax_minimize(
            jax_nan_at_origin,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            tol=1e-8,
            maxiter=5,
        )

        assert result.success is scipy_result.success
        assert result.status == scipy_result.status
        assert result.message == scipy_result.message
        assert result.invalid_step_log == []

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_reports_repo_status_for_nonfinite_initial_gradient(
        self,
    ):
        """Entry with a non-finite gradient must terminate with repo status 6."""

        def jax_inf_grad_at_origin(x):
            value = jnp.asarray(0.0, dtype=x.dtype)
            grad = jnp.asarray(
                [jnp.inf, jnp.inf],
                dtype=x.dtype,
            )
            return value, grad

        def scipy_inf_grad_at_origin(x):
            return np.float64(0.0), np.asarray([np.inf, np.inf], dtype=np.float64)

        x0 = np.zeros((2,), dtype=np.float64)
        scipy_result = optimize.minimize(
            scipy_inf_grad_at_origin,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5, "gtol": 1e-8},
        )
        result = jax_minimize(
            jax_inf_grad_at_origin,
            jnp.asarray(x0, dtype=jnp.float64),
            method="lbfgs-ondevice",
            tol=1e-8,
            maxiter=5,
            value_and_grad=True,
        )

        assert result.success is scipy_result.success
        assert result.status == _host_lbfgs.LBFGS_STATUS_NONFINITE
        assert "non-finite" in result.message.lower()
        assert len(result.invalid_step_log) == 1
        assert result.invalid_step_log[0]["line_search_failed"] is True
        assert result.invalid_step_log[0]["nonfinite_step"] is True


class TestBoozerSurfaceJAXClassPrivate:
    """Private BoozerSurfaceJAX class tests split from TestBoozerSurfaceJAXClass."""

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_hessian_system_status_jaxpr_stays_operator_only(self):
        x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        rhs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

        def objective(z):
            return 0.5 * jnp.dot(z, z)

        jaxpr = jax.make_jaxpr(
            lambda vec: _opt._solve_hessian_system_with_status(
                objective,
                x,
                vec,
                stab=0.0,
                tol=1e-10,
            )
        )(rhs)
        jaxpr_text = str(jaxpr)

        assert "_lu_solve" not in jaxpr_text
        assert "lu_pivots_to_permutation" not in jaxpr_text

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_hessian_system_status_reports_original_residual_shape(self):
        x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        rhs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

        def objective(z):
            return 0.5 * jnp.dot(z, z)

        solution, status = _opt._solve_hessian_system_with_status(
            objective,
            x,
            rhs,
            stab=0.0,
            tol=1e-10,
        )
        residual = rhs - solution

        assert bool(np.asarray(status.success))
        assert status._fields == (
            "success",
            "residual",
            "residual_relative",
            "iterations",
        )
        assert np.asarray(status.residual).shape == ()
        assert np.asarray(status.residual_relative).shape == ()
        assert isinstance(status.iterations, jax.Array)
        assert np.asarray(status.iterations).shape == ()
        np.testing.assert_allclose(
            np.asarray(status.residual), np.linalg.norm(residual)
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_linear_solve_unknown_iterations_remain_scalar_jax_array(self):
        rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        status = _opt._linear_solve_status(
            rhs,
            jnp.zeros_like(rhs),
            rhs,
            tol=1e-10,
            iterations=_opt._linear_solve_iteration_count(None),
        )

        assert isinstance(status.iterations, jax.Array)
        assert np.asarray(status.iterations).shape == ()
        assert int(np.asarray(status.iterations)) == -1
        assert _opt._linear_solve_iterations_host_value(status.iterations) is None

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_square_operator_zero_rhs_returns_successful_zero_solution(self):
        rhs = jnp.zeros(2, dtype=jnp.float64)
        matrix = jnp.asarray(
            [[2.0, 0.25], [-0.5, 3.0]],
            dtype=jnp.float64,
        )

        solution, status = _opt._solve_square_vector_system_operator_only(
            lambda vector: matrix @ vector,
            rhs,
            tol=1e-12,
        )

        np.testing.assert_allclose(np.asarray(solution), np.zeros(2), atol=0.0)
        assert bool(np.asarray(status.success))
        assert float(np.asarray(status.residual)) == pytest.approx(0.0)
        assert float(np.asarray(status.residual_relative)) == pytest.approx(0.0)
        assert int(np.asarray(status.iterations)) == 0

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_square_operator_explicit_budget_reaches_second_correction(
        self, monkeypatch
    ):
        rhs = jnp.asarray([1.0], dtype=jnp.float64)
        residual_fraction = jnp.asarray(1.0e-4, dtype=jnp.float64)

        def inexact_identity_solve(_matvec, solve_rhs, *, tol):
            del _matvec, tol
            solve_rhs = jnp.asarray(solve_rhs, dtype=jnp.float64)
            solution = (1.0 - residual_fraction) * solve_rhs
            residual = residual_fraction * solve_rhs
            return solution, residual, jnp.asarray(1, dtype=jnp.int32)

        monkeypatch.setattr(_opt, "_gmres_solve_array_system", inexact_identity_solve)

        solution, status = _opt._solve_square_vector_system_operator_only(
            lambda vector: vector,
            rhs,
            tol=1e-11,
            max_refinement_steps=(_opt._EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS),
        )

        np.testing.assert_allclose(np.asarray(solution), np.asarray(rhs), atol=1e-11)
        assert bool(np.asarray(status.success))
        assert float(np.asarray(status.residual_relative)) <= 1e-11
        assert int(np.asarray(status.iterations)) == 3

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_square_operator_default_one_correction_reproduces_head_failure(
        self, monkeypatch
    ):
        rhs = jnp.asarray([1.0], dtype=jnp.float64)
        residual_fraction = jnp.asarray(1.0e-4, dtype=jnp.float64)

        def inexact_identity_solve(_matvec, solve_rhs, *, tol):
            del _matvec, tol
            solve_rhs = jnp.asarray(solve_rhs, dtype=jnp.float64)
            solution = (1.0 - residual_fraction) * solve_rhs
            residual = residual_fraction * solve_rhs
            return solution, residual, jnp.asarray(1, dtype=jnp.int32)

        monkeypatch.setattr(_opt, "_gmres_solve_array_system", inexact_identity_solve)

        solution, status = _opt._solve_square_vector_system_operator_only(
            lambda vector: vector,
            rhs,
            tol=1e-11,
        )

        np.testing.assert_allclose(np.asarray(solution), np.asarray([0.99999999]))
        assert not bool(np.asarray(status.success))
        assert float(np.asarray(status.residual_relative)) > 1e-11
        assert int(np.asarray(status.iterations)) == 2

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_square_operator_extra_correction_rejects_nonmonotonic_residual(
        self, monkeypatch
    ):
        rhs = jnp.asarray([1.0], dtype=jnp.float64)

        def staged_identity_solve(_matvec, solve_rhs, *, tol):
            del _matvec, tol
            solve_rhs = jnp.asarray(solve_rhs, dtype=jnp.float64)
            rhs_norm = jnp.linalg.norm(solve_rhs)
            solve_fraction = jnp.where(
                rhs_norm < jnp.asarray(5.0e-4, dtype=jnp.float64),
                jnp.asarray(20.0, dtype=jnp.float64),
                jnp.where(
                    rhs_norm < jnp.asarray(5.0e-1, dtype=jnp.float64),
                    jnp.asarray(0.9, dtype=jnp.float64),
                    jnp.asarray(0.999, dtype=jnp.float64),
                ),
            )
            solution = solve_fraction * solve_rhs
            residual = solve_rhs - solution
            return solution, residual, jnp.asarray(1, dtype=jnp.int32)

        monkeypatch.setattr(_opt, "_gmres_solve_array_system", staged_identity_solve)

        solution, status = _opt._solve_square_vector_system_operator_only(
            lambda vector: vector,
            rhs,
            tol=1e-11,
            max_refinement_steps=(_opt._EXACT_JACOBIAN_OPERATOR_GMRES_REFINEMENT_STEPS),
        )

        np.testing.assert_allclose(np.asarray(solution), np.asarray([0.9999]))
        assert not bool(np.asarray(status.success))
        assert float(np.asarray(status.residual_relative)) == pytest.approx(1.0e-4)
        assert int(np.asarray(status.iterations)) == 3

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_hessian_least_squares_dense_status_runs_under_strict_transfer_guard(self):
        x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        rhs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

        def objective(z):
            return 0.5 * jnp.dot(z, z)

        with jax.transfer_guard("disallow"):
            solution, status = _opt._solve_hessian_least_squares_system_with_status(
                objective,
                x,
                rhs,
                stab=0.0,
                tol=1e-10,
            )

        assert bool(np.asarray(status.success))
        assert isinstance(status.iterations, jax.Array)
        assert np.asarray(status.iterations).shape == ()
        np.testing.assert_allclose(np.asarray(solution), np.asarray(rhs), atol=1e-12)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_hessian_least_squares_system_rejects_original_residual_miss(self):
        x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        rhs = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

        def objective(z):
            return 0.5 * z[0] * z[0]

        solution, status = _opt._solve_hessian_least_squares_system_with_status(
            objective,
            x,
            rhs,
            stab=0.0,
            tol=1e-10,
        )
        hessian_residual = rhs - jnp.asarray([solution[0], 0.0], dtype=jnp.float64)

        assert not bool(np.asarray(status))
        assert np.all(np.isfinite(np.asarray(solution)))
        assert np.linalg.norm(np.asarray(hessian_residual)) > 1e-10
        assert float(np.asarray(status.residual_relative)) > 1e-10

    def test_gmres_iteration_limits_bound_hvp_work(self):
        assert _opt._gmres_iteration_limits(39) == (39, 10)
        assert _opt._gmres_iteration_limits(663) == (64, 10)
        assert (
            _opt._operator_gmres_matvec_budget(663, max_refinement_steps=1)
            == 1302
        )

    def test_dense_operator_chunk_batch_size_tracks_byte_budget(self):
        mib = 1024 * 1024
        assert _opt._dense_operator_chunk_batch_size_from_budget(None) == 8
        assert _opt._dense_operator_chunk_batch_size_from_budget(16 * mib) == 1
        assert _opt._dense_operator_chunk_batch_size_from_budget(255 * mib) == 7
        assert _opt._dense_operator_chunk_batch_size_from_budget(256 * mib) == 8
        assert _opt._dense_operator_chunk_batch_size_from_budget(512 * mib) == 8
        assert _opt._dense_operator_chunk_batch_size_from_budget(4096 * mib) == 8
        assert _opt._dense_operator_chunk_batch_size_from_budget(24 * 1024 * mib) == 8
        assert _opt._dense_operator_chunk_batch_size_from_budget(48 * 1024 * mib) == 16
        assert _opt._dense_operator_chunk_batch_size_from_budget(96 * 1024 * mib) == 32
        assert _opt._dense_operator_chunk_batch_size_from_budget(192 * 1024 * mib) == 64
        assert _opt._dense_operator_chunk_batch_size_from_budget(256 * 1024 * mib) == 64

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
    def test_newton_polish_traceable_materialized_policy_keeps_operator_step(
        self, monkeypatch
    ):
        """Dense compatibility metadata must not force dense Newton steps."""

        observed = {"calls": 0}

        def exact_operator_solve(_matvec, rhs, *, tol):
            del tol
            observed["calls"] += 1
            return rhs, _mock_linear_solve_status(True)

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            exact_operator_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=1,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=True,
        )

        np.testing.assert_allclose(
            np.asarray(result["x"]),
            np.zeros_like(np.asarray(x0)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(result["hessian"]),
            np.eye(x0.size),
            atol=1e-12,
        )
        assert bool(result["success"]) is True
        assert bool(result["hessian_materialized"]) is True
        assert observed["calls"] == 1

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_reports_iteration_diagnostics(
        self, monkeypatch
    ):
        """Traceable Newton must expose enough loop state to diagnose slow K1 solves."""

        def exact_operator_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(1e-14, dtype=jnp.float64),
                residual_relative=jnp.asarray(2.5e-13, dtype=jnp.float64),
                iterations=jnp.asarray(7, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            exact_operator_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        expected_matvec_budget = _opt._operator_gmres_matvec_budget(
            x0.size,
            max_refinement_steps=0,
        )
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=3,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
        )

        initial_norm = np.linalg.norm(np.asarray(x0))
        np.testing.assert_array_equal(
            np.asarray(result["newton_trace_active"]),
            np.asarray([True, False, False]),
        )
        np.testing.assert_array_equal(
            np.asarray(result["newton_trace_step_accepted"]),
            np.asarray([True, False, False]),
        )
        np.testing.assert_allclose(
            np.asarray(result["newton_trace_gradient_norm_before"])[0],
            initial_norm,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(result["newton_trace_step_norm"])[0],
            initial_norm,
            rtol=0.0,
            atol=1e-12,
        )
        assert bool(result["newton_trace_step_finite"][0]) is True
        assert bool(result["newton_trace_linear_solve_success"][0]) is True
        assert int(result["newton_trace_linear_solve_iterations"][0]) == 7
        assert (
            int(result["newton_trace_linear_solve_matvec_budget"][0])
            == expected_matvec_budget
        )
        np.testing.assert_allclose(
            np.asarray(result["newton_trace_linear_residual_relative"])[0],
            2.5e-13,
            rtol=0.0,
            atol=1e-18,
        )
        assert int(result["newton_trace_backtracking_iterations"][0]) == 1
        np.testing.assert_allclose(
            np.asarray(result["newton_trace_accepted_alpha"])[0],
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
        assert int(result["newton_attempted_iterations"]) == 1
        assert int(result["newton_iter"]) == 1
        assert bool(result["newton_stalled"]) is False
        assert int(result["newton_stop_reason_code"]) == _opt._NEWTON_STOP_SUCCESS
        assert bool(result["newton_last_step_accepted"]) is True
        assert bool(result["newton_last_linear_solve_success"]) is True
        assert int(result["newton_last_linear_solve_iterations"]) == 7
        assert (
            int(result["newton_last_linear_solve_matvec_budget"])
            == expected_matvec_budget
        )
        assert (
            int(result["newton_linear_solve_backend_code"])
            == _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
            ]
        )
        np.testing.assert_allclose(
            np.asarray(result["initial_gradient_norm"]),
            initial_norm,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(result["final_gradient_norm"]),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_dense_lu_comparator_dispatches(
        self, monkeypatch
    ):
        """Opt-in dense-LU traceable Newton uses the dense direct solve path."""

        observed = {"dense": 0}

        def dense_lu_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            observed["dense"] += 1
            return rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(0.0, dtype=jnp.float64),
                residual_relative=jnp.asarray(0.0, dtype=jnp.float64),
                iterations=jnp.asarray(0, dtype=jnp.int32),
            )

        def operator_solve(_matvec, _rhs, *, tol):
            del _matvec, _rhs, tol
            raise AssertionError("operator-GMRES fallback should not run")

        monkeypatch.setattr(
            _opt,
            "_TRACEABLE_NEWTON_LINEAR_SOLVER",
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_dense_square_operator_lu_system_with_status",
            dense_lu_solve,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            operator_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=1,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
        )

        assert observed["dense"] == 1
        assert bool(result["success"]) is True
        assert int(result["newton_last_linear_solve_iterations"]) == 0
        assert int(result["newton_last_linear_solve_matvec_budget"]) == x0.size
        assert (
            int(result["newton_linear_solve_backend_code"])
            == _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU
            ]
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_dense_lu_comparator_falls_back_when_disallowed(
        self, monkeypatch
    ):
        """Dense-LU request preserves the operator path when byte policy blocks it."""

        observed = {"operator": 0}

        def dense_lu_solve(_matvec, _rhs, *, tol):
            del _matvec, _rhs, tol
            raise AssertionError("dense-LU solve should not run")

        def operator_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            observed["operator"] += 1
            return rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(0.0, dtype=jnp.float64),
                residual_relative=jnp.asarray(0.0, dtype=jnp.float64),
                iterations=jnp.asarray(9, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_TRACEABLE_NEWTON_LINEAR_SOLVER",
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU,
        )
        monkeypatch.setattr(
            _opt,
            "_dense_square_operator_lu_materialization_allowed",
            lambda _rhs: False,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_dense_square_operator_lu_system_with_status",
            dense_lu_solve,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            operator_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=1,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
        )

        assert observed["operator"] == 1
        assert bool(result["success"]) is True
        assert int(result["newton_last_linear_solve_iterations"]) == 9
        assert (
            int(result["newton_linear_solve_backend_code"])
            == _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
                _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
            ]
        )

    def test_traceable_newton_linear_solver_resolver_accepts_hybrid_alias(
        self, monkeypatch
    ):
        """The hybrid comparator is an explicit opt-in env value."""
        monkeypatch.setenv(_opt._TRACEABLE_NEWTON_LINEAR_SOLVER_ENV, "hybrid")

        assert (
            _opt._resolve_traceable_newton_linear_solver()
            == _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_hybrid_uses_dense_lu_only_near_target(
        self, monkeypatch
    ):
        """Hybrid Newton uses GMRES while loose and dense-LU in the strict-cap region."""
        operator_code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
        ]
        dense_code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU
        ]

        def dense_lu_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(0.0, dtype=jnp.float64),
                residual_relative=jnp.asarray(0.0, dtype=jnp.float64),
                iterations=jnp.asarray(0, dtype=jnp.int32),
            )

        def operator_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return rhs * 0.5, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(1e-8, dtype=jnp.float64),
                residual_relative=jnp.asarray(1e-6, dtype=jnp.float64),
                iterations=jnp.asarray(11, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_TRACEABLE_NEWTON_LINEAR_SOLVER",
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU,
        )
        monkeypatch.setattr(
            _opt,
            "_dense_square_operator_lu_materialization_allowed",
            lambda _rhs: True,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_dense_square_operator_lu_system_with_status",
            dense_lu_solve,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            operator_solve,
        )

        x0 = jnp.asarray([1.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=6,
            tol=1e-3,
            stab=0.0,
            materialize_hessian=False,
        )

        active = np.asarray(result["newton_trace_active"], dtype=bool)
        backend_codes = np.asarray(
            result["newton_trace_linear_solve_backend_code"]
        )[active]
        iterations = np.asarray(result["newton_trace_linear_solve_iterations"])[active]
        budgets = np.asarray(result["newton_trace_linear_solve_matvec_budget"])[active]

        assert int(result["newton_attempted_iterations"]) == 5
        np.testing.assert_array_equal(
            backend_codes,
            np.asarray(
                [
                    operator_code,
                    operator_code,
                    operator_code,
                    operator_code,
                    dense_code,
                ],
                dtype=np.int32,
            ),
        )
        np.testing.assert_array_equal(
            iterations,
            np.asarray([11, 11, 11, 11, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            budgets,
            np.asarray([61, 61, 61, 61, 1], dtype=np.int32),
        )
        assert int(result["newton_linear_solve_backend_code"]) == dense_code
        assert int(result["newton_last_linear_solve_matvec_budget"]) == 1
        assert bool(result["success"]) is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_hybrid_falls_back_when_dense_disallowed(
        self, monkeypatch
    ):
        """Hybrid request preserves the operator path when byte policy blocks LU."""
        operator_code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
        ]

        def dense_lu_solve(_matvec, _rhs, *, tol):
            del _matvec, _rhs, tol
            raise AssertionError("dense-LU solve should not run")

        def operator_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(0.0, dtype=jnp.float64),
                residual_relative=jnp.asarray(0.0, dtype=jnp.float64),
                iterations=jnp.asarray(13, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_TRACEABLE_NEWTON_LINEAR_SOLVER",
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU,
        )
        monkeypatch.setattr(
            _opt,
            "_dense_square_operator_lu_materialization_allowed",
            lambda _rhs: False,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_dense_square_operator_lu_system_with_status",
            dense_lu_solve,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            operator_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=1,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
        )

        assert bool(result["success"]) is True
        assert int(result["newton_last_linear_solve_iterations"]) == 13
        assert int(result["newton_linear_solve_backend_code"]) == operator_code
        assert (
            int(result["newton_trace_linear_solve_backend_code"][0])
            == operator_code
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_traceable_newton_gmres_refinement_recovers_tight_tolerance(self):
        """One refinement pass breaks the restarted-GMRES round-off plateau.

        Restarted operator-GMRES stalls above the Eisenstat-Walker strict cap
        on ill-conditioned systems (n > restart); the near-target refinement
        pass must recover the tight tolerance the single pass cannot reach.
        """
        rng = np.random.default_rng(20260704)
        n = 300
        orthogonal, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eigenvalues = np.logspace(0.0, -np.log10(5.0e3), n)
        matrix = jnp.asarray(
            orthogonal @ np.diag(eigenvalues) @ orthogonal.T,
            dtype=jnp.float64,
        )
        rhs = jnp.asarray(rng.standard_normal(n), dtype=jnp.float64)

        def matvec(v):
            return matrix @ v

        tol = 1e-12
        single, single_status = (
            _opt._solve_traceable_newton_operator_gmres_with_status(
                matvec, rhs, tol=tol
            )
        )
        refined, refined_status = (
            _opt._refine_traceable_newton_operator_gmres_solution(
                matvec, rhs, single, single_status, tol=tol
            )
        )
        rhs_norm = float(jnp.linalg.norm(rhs))
        single_rel = float(jnp.linalg.norm(rhs - matvec(single))) / rhs_norm
        refined_rel = float(jnp.linalg.norm(rhs - matvec(refined))) / rhs_norm

        assert single_rel > 1e-10, (
            f"single-pass GMRES no longer plateaus (rel {single_rel:.3e}); "
            "test premise broken, re-derive the fixture conditioning"
        )
        assert refined_rel <= 1e-10, (
            f"refinement failed to recover tight tolerance: {refined_rel:.3e}"
        )
        assert refined_rel * 50.0 <= single_rel, (
            f"refinement recovery too weak: {single_rel:.3e} -> {refined_rel:.3e}"
        )
        # The plateau is visible in the status contract: the single pass
        # reports failure, the refined solve reports success.
        assert not bool(single_status.success)
        assert bool(refined_status.success)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_refines_unconverged_gmres_near_target(
        self, monkeypatch
    ):
        """Default polish refines an unconverged GMRES step only near target.

        Far from the target the loose E-W solve is taken as-is (plain matvec
        budget); in the strict-cap region an unconverged primary solve must
        trigger the refinement pass (refined budget) and converge the polish.
        """
        operator_code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
        ]

        def unconverged_primary_solve(_matvec, rhs, *, tol):
            del tol
            return rhs * 0.5, _opt._LinearSolveStatus(
                success=jnp.asarray(False),
                residual=jnp.asarray(1e-4, dtype=jnp.float64),
                residual_relative=jnp.asarray(1e-2, dtype=jnp.float64),
                iterations=jnp.asarray(11, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            unconverged_primary_solve,
        )

        x0 = jnp.asarray([1.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=6,
            tol=1e-3,
            stab=0.0,
            materialize_hessian=False,
        )

        plain_budget = int(
            _opt._operator_gmres_matvec_budget(1, max_refinement_steps=0)
        )
        refined_budget = (
            int(
                _opt._operator_gmres_matvec_budget(
                    1,
                    max_refinement_steps=(
                        _opt._SQUARE_OPERATOR_GMRES_REFINEMENT_STEPS
                    ),
                )
            )
            + 1
        )
        active = np.asarray(result["newton_trace_active"], dtype=bool)
        backend_codes = np.asarray(
            result["newton_trace_linear_solve_backend_code"]
        )[active]
        budgets = np.asarray(result["newton_trace_linear_solve_matvec_budget"])[
            active
        ]

        assert bool(result["success"]) is True
        np.testing.assert_array_equal(
            backend_codes,
            np.full(backend_codes.shape, operator_code, dtype=np.int32),
        )
        # Far-from-target iterations: plain single-pass budget, no refinement.
        np.testing.assert_array_equal(
            budgets[:-1],
            np.full(budgets.size - 1, plain_budget, dtype=np.int32),
        )
        # The near-target iteration refines the unconverged primary solve.
        assert int(budgets[-1]) == refined_budget
        assert int(result["newton_last_linear_solve_matvec_budget"]) == (
            refined_budget
        )
        # Refinement solved the (identity-Hessian) system exactly: the fake
        # primary reported failure, so a passing final status proves the
        # refinement pass ran and transformed the outcome. (Correction
        # iteration counts may be unknown for the library GMRES, so the
        # combined count is only guaranteed not to drop below the primary's.)
        assert bool(result["newton_last_linear_solve_success"]) is True
        assert int(result["newton_last_linear_solve_iterations"]) >= 11

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_retries_loose_gmres_rejection_at_strict_cap(
        self, monkeypatch
    ):
        """A rejected loose E-W direction retries at the strict cap, not stall.

        Regression for the A100 init stall found validating the E-W uncap:
        the first Newton iteration solves at the loose Eisenstat-Walker
        tolerance (eta ~0.5); a crude direction that fails backtracking used
        to set ``stalled`` and exit with zero iterations (pre-fix this test
        fails with success=False, attempted=1). The safeguard must rerun the
        rejected iteration at the strict cap and only stall on a rejected
        tight direction.
        """
        tol = 1e-3
        strict_cap = _opt._eisenstat_walker_strict_cap(
            jnp.asarray(tol, dtype=jnp.float64), dtype=jnp.float64
        )

        def tol_sensitive_solve(_matvec, rhs, *, tol):
            # Loose solves return an ascent direction (backtracking rejects);
            # strict-cap solves return the exact Newton step for the
            # identity-Hessian objective below.
            loose = tol > strict_cap
            dx = jnp.where(loose, -rhs, rhs)
            return dx, _opt._LinearSolveStatus(
                success=~loose,
                residual=jnp.where(loose, 1.0, 0.0).astype(jnp.float64),
                residual_relative=jnp.where(loose, 0.5, 0.0).astype(
                    jnp.float64
                ),
                iterations=jnp.asarray(7, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            tol_sensitive_solve,
        )

        x0 = jnp.asarray([1.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=6,
            tol=tol,
            stab=0.0,
            materialize_hessian=False,
        )

        assert bool(result["success"]) is True, (
            "loose-direction rejection must not terminate the polish"
        )
        assert int(result["newton_attempted_iterations"]) == 2
        active = np.asarray(result["newton_trace_active"], dtype=bool)
        accepted = np.asarray(result["newton_trace_step_accepted"])[active]
        np.testing.assert_array_equal(
            accepted, np.asarray([False, True], dtype=bool)
        )
        linear_tols = np.asarray(result["newton_trace_linear_tol"])[active]
        assert linear_tols[0] > float(strict_cap), (
            "first iteration must use the loose E-W tolerance"
        )
        assert linear_tols[1] <= float(strict_cap), (
            "retry iteration must solve at (or below) the strict cap"
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_hybrid_routes_strict_cap_retry_to_dense_lu(
        self, monkeypatch
    ):
        """Hybrid mode serves the strict-cap retry with the dense-LU solver.

        The retry exists to hand the line search one quality direction
        before the loop may stall.  Strict-cap operator GMRES runs to
        essentially the full Krylov dimension on the squared-conditioned
        Hessian (the measured ~1300-matvec grind that priced the jax-CPU
        reference lane out at 255x64), while the dense direction is
        tolerance-exact at roughly half that matvec cost.  In hybrid mode a
        far-from-target rejection must therefore route its retry iteration
        through dense-LU, leaving no strict-tolerance GMRES entry point.
        """
        tol = 1e-3
        strict_cap = _opt._eisenstat_walker_strict_cap(
            jnp.asarray(tol, dtype=jnp.float64), dtype=jnp.float64
        )
        operator_code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_OPERATOR_GMRES
        ]
        dense_code = _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_CODES[
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_DENSE_LU
        ]

        def rejecting_operator_solve(_matvec, rhs, *, tol):
            # Always an ascent direction: any operator iteration is rejected,
            # so a retry served by the operator branch could never succeed.
            del tol
            return -rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(0.0, dtype=jnp.float64),
                residual_relative=jnp.asarray(0.0, dtype=jnp.float64),
                iterations=jnp.asarray(7, dtype=jnp.int32),
            )

        def exact_dense_solve(_matvec, rhs, *, tol):
            # Exact Newton step for the identity-Hessian objective below.
            del tol
            return rhs, _opt._LinearSolveStatus(
                success=jnp.asarray(True),
                residual=jnp.asarray(0.0, dtype=jnp.float64),
                residual_relative=jnp.asarray(0.0, dtype=jnp.float64),
                iterations=jnp.asarray(0, dtype=jnp.int32),
            )

        monkeypatch.setattr(
            _opt,
            "_TRACEABLE_NEWTON_LINEAR_SOLVER",
            _opt._TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_LU,
        )
        monkeypatch.setattr(
            _opt,
            "_dense_square_operator_lu_materialization_allowed",
            lambda _rhs: True,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            rejecting_operator_solve,
        )
        monkeypatch.setattr(
            _opt,
            "_solve_dense_square_operator_lu_system_with_status",
            exact_dense_solve,
        )

        x0 = jnp.asarray([1.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=5,
            tol=tol,
            stab=0.0,
            materialize_hessian=False,
        )

        assert bool(result["success"]) is True, (
            "the dense-served retry direction must rescue the rejected "
            "loose operator iteration"
        )
        assert int(result["newton_attempted_iterations"]) == 2
        active = np.asarray(result["newton_trace_active"], dtype=bool)
        accepted = np.asarray(result["newton_trace_step_accepted"])[active]
        np.testing.assert_array_equal(
            accepted, np.asarray([False, True], dtype=bool)
        )
        backend_codes = np.asarray(
            result["newton_trace_linear_solve_backend_code"]
        )[active]
        np.testing.assert_array_equal(
            backend_codes,
            np.asarray([operator_code, dense_code], dtype=np.int32),
        )
        linear_tols = np.asarray(result["newton_trace_linear_tol"])[active]
        assert linear_tols[0] > float(strict_cap)
        assert linear_tols[1] <= float(strict_cap)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_exact_traceable_retries_loose_gmres_rejection_at_strict_cap(
        self, monkeypatch
    ):
        """The exact-Newton runner shares the strict-cap retry safeguard.

        Mirror of the LS-polish regression: a rejected loose E-W direction
        must schedule one strict-cap retry instead of stalling (pre-fix the
        first loose rejection ended the solve with success=False, nit=0).
        """
        tol = 1e-3
        strict_cap = _opt._eisenstat_walker_strict_cap(
            jnp.asarray(tol, dtype=jnp.float64), dtype=jnp.float64
        )

        def tol_sensitive_exact_solve(_jvp_fn, _x, rhs, *, tol):
            loose = tol > strict_cap
            dx = jnp.where(loose, -rhs, rhs)
            # Zero linear residual keeps the in-loop refinement correction
            # out of the picture so only the retry safeguard is exercised.
            return dx, jnp.zeros_like(rhs), jnp.asarray(0, dtype=jnp.int32)

        monkeypatch.setattr(
            _opt,
            "_gmres_solve_exact_newton_system",
            tol_sensitive_exact_solve,
        )

        result = _opt.newton_exact_traceable(
            lambda x: x,
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
            maxiter=6,
            tol=tol,
        )

        assert bool(result["success"]) is True, (
            "loose-direction rejection must not terminate the exact solve"
        )
        assert int(result["nit"]) == 1
        np.testing.assert_allclose(
            np.asarray(result["x"]), np.zeros(2), atol=1e-15
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_traceable_matvec_counts_rearm_across_kernel_executions(self):
        """Draining counts must rearm the token for later kernel executions.

        Traced decomposed K1 kernels bake the counter token in at trace
        time and re-execute against the same registry entry. The pre-fix
        destructive drain popped the entry after the first evaluation, so
        the callback's missing-entry guard silently dropped every later
        execution's counts (observed on A100: only the first K1 event of a
        run carried matvec actuals). Pre-fix this test's second drain
        returns None.
        """
        token = _opt._register_traceable_matvec_counter(3)
        try:
            _opt._invoke_traceable_matvec_counter(token, 0)
            _opt._invoke_traceable_matvec_counter(token, 0)
            _opt._invoke_traceable_matvec_counter(token, 1)
            first = _opt.traceable_newton_matvec_counts_from_token(token)
            assert first == (2, 1, 0)
            # Second execution window of the same compiled kernel.
            _opt._invoke_traceable_matvec_counter(token, 0)
            second = _opt.traceable_newton_matvec_counts_from_token(token)
            assert second == (1, 0, 0)
        finally:
            _opt._unregister_traceable_matvec_counter(token)

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_eager_matvec_counter_drain_keeps_one_shot_pop_semantics(self):
        """The eager per-call drain still removes its token (no registry
        growth for eager solves that allocate one token per call)."""
        token = _opt._register_traceable_matvec_counter(2)
        _opt._invoke_traceable_matvec_counter(token, 0)
        assert _opt._drain_traceable_matvec_counter(token) == (1, 0)
        assert _opt._drain_traceable_matvec_counter(token) is None

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_records_opt_in_matvec_counts(
        self, monkeypatch
    ):
        """Opt-in diagnostics should report actual GMRES operator matvec calls."""

        monkeypatch.setenv(_opt._TRACEABLE_NEWTON_MATVEC_COUNT_ENV, "1")

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=3,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
        )

        active = np.asarray(result["newton_trace_active"], dtype=bool)
        actual = np.asarray(result["newton_trace_linear_solve_matvec_actual"])
        active_indices = np.nonzero(active)[0]

        assert active_indices.size >= 1
        assert np.all(actual[active] > 0)
        assert np.all(
            actual[~active] == _opt._LINEAR_SOLVE_ITERATIONS_UNKNOWN
        )
        assert int(result["newton_last_linear_solve_matvec_actual"]) == int(
            actual[active_indices[-1]]
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_records_matvec_counts_inside_jit(
        self, monkeypatch
    ):
        """Matvec diagnostics must not host-convert traced Newton metadata."""

        monkeypatch.setenv(_opt._TRACEABLE_NEWTON_MATVEC_COUNT_ENV, "1")

        @jax.jit
        def run_nested_newton(x0):
            result = _opt.newton_polish_traceable(
                lambda x: 0.5 * jnp.dot(x, x),
                x0,
                maxiter=3,
                tol=1e-12,
                stab=0.0,
                materialize_hessian=False,
            )
            return (
                result["newton_trace_active"],
                result["newton_matvec_counter_token"],
            )

        active, token = run_nested_newton(
            jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        )
        active = np.asarray(active, dtype=bool)
        active_indices = np.nonzero(active)[0]
        counts = _opt.traceable_newton_matvec_counts_from_token(int(token))
        assert counts is not None
        # Rearm contract: the drain zeroes the window instead of removing it,
        # so a second drain with no interleaved kernel execution reports an
        # all-zero window (later executions of the compiled kernel keep
        # counting into the rearmed entry).
        second = _opt.traceable_newton_matvec_counts_from_token(int(token))
        assert second is not None and not any(second)
        actual = np.full(
            active.shape,
            _opt._LINEAR_SOLVE_ITERATIONS_UNKNOWN,
            dtype=np.int32,
        )
        actual[active] = np.asarray(counts, dtype=np.int32)[active]

        assert active_indices.size >= 1
        assert np.all(actual[active] > 0)
        assert np.all(
            actual[~active] == _opt._LINEAR_SOLVE_ITERATIONS_UNKNOWN
        )
        assert int(actual[active_indices[-1]]) > 0

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_accepts_finite_descent_step_with_failed_status(
        self, monkeypatch
    ):
        """Traceable Newton should match the nontraceable inexact-step contract."""

        def finite_descent_step_failed_status(_matvec, rhs, *, tol):
            del _matvec, tol
            return rhs, _mock_linear_solve_status(False)

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            finite_descent_step_failed_status,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=1,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=True,
        )

        np.testing.assert_allclose(
            np.asarray(result["x"]),
            np.zeros_like(np.asarray(x0)),
            atol=1e-12,
        )
        assert bool(result["success"]) is True

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_nonfinite_linear_step_stalls_without_dense_fallback(
        self, monkeypatch
    ):
        """Traceable Newton must fail closed instead of materializing a dense step."""

        def fake_operator_only_linear_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return jnp.full_like(rhs, jnp.nan), _mock_linear_solve_status(False)

        def forbid_dense_hessian(*_args, **_kwargs):
            raise AssertionError(
                "traceable Newton should not materialize a dense Hessian fallback"
            )

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            fake_operator_only_linear_solve,
        )
        monkeypatch.setattr(_opt, "_materialize_dense_hessian", forbid_dense_hessian)

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=3,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
        )

        np.testing.assert_allclose(np.asarray(result["x"]), np.asarray(x0))
        assert int(result["nit"]) == 0
        assert bool(result["success"]) is False
        assert int(result["newton_attempted_iterations"]) == 1
        assert bool(result["newton_stalled"]) is True
        assert bool(result["newton_last_step_finite"]) is False
        assert result["hessian"] is None

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_backtracks_norm_increasing_operator_steps(
        self, monkeypatch
    ):
        """Phase 5 production Newton rejects finite but non-descent steps."""
        observed = {"progress_calls": 0}

        def fake_operator_only_linear_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return -rhs, _mock_linear_solve_status(True)

        monkeypatch.setattr(
            _opt,
            "_solve_traceable_newton_operator_gmres_with_status",
            fake_operator_only_linear_solve,
        )

        x0 = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
        result = _opt.newton_polish_traceable(
            lambda x: 0.5 * jnp.dot(x, x),
            x0,
            maxiter=3,
            tol=1e-12,
            stab=0.0,
            materialize_hessian=False,
            progress_callback=lambda *_args: observed.__setitem__(
                "progress_calls",
                observed["progress_calls"] + 1,
            ),
        )

        np.testing.assert_allclose(np.asarray(result["x"]), np.asarray(x0))
        assert int(result["nit"]) == 0
        assert bool(result["success"]) is False
        # The loose E-W rejection schedules one strict-cap retry iteration
        # (inexact-Newton safeguard) before the tight rejection stalls.
        assert int(result["newton_attempted_iterations"]) == 2
        assert bool(result["newton_stalled"]) is True
        assert bool(result["newton_last_step_accepted"]) is False
        assert int(result["newton_last_backtracking_iterations"]) == 8
        assert observed["progress_calls"] == 0

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_limited_memory_routes_to_lbfgs(self, monkeypatch):
        """limited_memory=True must route LS solves through lbfgs-ondevice."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

        captured = {}

        def fake_target_minimize(
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

        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
        _patch_newton_polish_runner(monkeypatch, fake_newton_polish)

        res = booz.run_code(iota=0.3, G=0.05)

        assert captured["method"] == "lbfgs-ondevice"
        assert res["success"] is True
        assert res["optimizer_method"] == "lbfgs-ondevice"

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_traceable_limited_memory_pre_newton_uses_monolithic_lbfgs(
        self, monkeypatch
    ):
        """Traceable limited-memory LS must use the jit-compatible L-BFGS mode."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

        captured = {}

        def fake_minimize_lbfgs(
            fun,
            x0,
            *,
            maxiter,
            gtol,
            maxcor,
            ftol,
            maxfun,
            maxls,
            run_mode,
        ):
            del fun, maxiter, gtol, maxcor, ftol, maxfun, maxls
            captured["run_mode"] = run_mode
            return types.SimpleNamespace(
                converged=jnp.asarray(True),
                failed=jnp.asarray(False),
                k=jnp.asarray(2, dtype=jnp.int32),
                nfev=jnp.asarray(3, dtype=jnp.int32),
                ngev=jnp.asarray(4, dtype=jnp.int32),
                x_k=x0,
                f_k=jnp.asarray(0.0, dtype=x0.dtype),
                g_k=jnp.zeros_like(x0),
                ls_status=jnp.asarray(0, dtype=jnp.int32),
            )

        monkeypatch.setattr(_opt, "_minimize_lbfgs_private", fake_minimize_lbfgs)

        x0 = booz._pack_decision_vector(0.3, 0.05)
        result = booz._run_traceable_pre_newton_stage(
            booz.coil_set_spec,
            x0,
            "lbfgs-ondevice",
            optimize_G=True,
            weight_inv_modB=True,
            materialize_dense_linearization=False,
        )

        assert captured["run_mode"] == "monolithic_debug"
        np.testing.assert_allclose(np.asarray(result["x"]), np.asarray(x0))
        assert bool(np.asarray(result["success"])) is True
        assert int(np.asarray(result["nit"])) == 2

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_run_code_ondevice_force_limited_memory_routes_to_lbfgs(self, monkeypatch):
        """The explicit Boozer LS override must route ondevice solves through lbfgs."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = False
        booz.options["force_ondevice_limited_memory"] = True

        captured = {}

        def fake_target_minimize(
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

        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
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

        monkeypatch.setattr(_opt_ref, "_scipy_minimize", forbidden_scipy_minimize)

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

        def fake_target_minimize(
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

        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
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
        """On-device L-BFGS-B progress should surface sparse stage updates."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

        observed = []

        def record_stage(label, **payload):
            observed.append((label, payload))

        booz.options["stage_callback"] = record_stage

        def fake_target_minimize(
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

        monkeypatch.setattr(_bsj, "target_minimize", fake_target_minimize)
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
    def test_run_code_ondevice_limited_memory_runs_without_monkeypatch(self):
        """limited_memory=True must run lbfgs-ondevice and keep Newton polish authoritative."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True
        booz.options["ftol"] = 0.0
        booz.options["bfgs_maxiter"] = 1
        booz.options["maxcor"] = 4

        res = booz.run_code(iota=0.3, G=0.05)
        pre_newton = res["pre_newton"]

        assert res is not None
        assert res["type"] == "ls"
        assert np.isfinite(res["fun"])
        assert pre_newton["optimizer_method"] == "lbfgs-ondevice"
        assert pre_newton["iter"] == 1
        assert pre_newton["success"] is False
        assert np.all(np.isfinite(np.asarray(pre_newton["gradient"])))
        assert np.all(np.isfinite(np.asarray(res["jacobian"])))
        assert res["success"] is True
        assert res["primal_success"] is True
        assert res["adjoint_linear_solve_available"] is True
        assert res["PLU"] is not None
        assert bool(np.asarray(res["hessian_materialized"])) is True
        assert res["dense_linear_solve_factors_available"] is True
        assert res["linear_solve_backend"] == "dense-plu-shared"
        assert callable(res["vjp"])
        assert res["optimizer_method"] == "lbfgs-ondevice"
        solved_state = booz.get_solved_runtime_state()
        np.testing.assert_allclose(
            np.asarray(solved_state.sdofs), np.asarray(res["sdofs"])
        )
