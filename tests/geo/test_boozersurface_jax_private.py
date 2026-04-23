"""Private optimizer runtime tests for BoozerSurfaceJAX."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt.geo.optimizer_jax_reference as _opt_ref
import simsopt.geo.optimizer_jax_private._bfgs as _private_bfgs
import simsopt.geo.optimizer_jax_private._common as _opt_common
from conftest import enable_non_strict_jax_backend
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
        solve_transpose_with_status=lambda rhs: (rhs, False),
    )

    with pytest.raises(RuntimeError, match="operator-backed runtime path"):
        _soj._solve_boozer_adjoint(adjoint_state, jnp.ones((2,), dtype=jnp.float64))


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

    original_asarray = _opt.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, HasDtypeOnly):
            raise AssertionError("np.asarray should not run when dtype attr exists")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(_opt.np, "asarray", guarded_asarray)

    assert _opt._optimizer_dtype(HasDtypeOnly()) == np.dtype(np.float64)


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


@pytest.mark.parametrize("method", ["bfgs-ondevice", "adam-ondevice"])
def test_target_minimize_rejects_failure_callback_for_unsupported_methods(method):
    def quad(x):
        return 0.5 * jnp.dot(x, x)

    with pytest.raises(
        ValueError,
        match="only supports failure_callback for method='lbfgs-ondevice'",
    ):
        _opt.target_minimize(
            quad,
            jnp.array([1.0, -2.0], dtype=jnp.float64),
            method=method,
            failure_callback=lambda *args: None,
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
    result = _opt._line_search_value_and_grad(
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
    result = _opt._line_search_value_and_grad(
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


def test_zoom_reuses_cached_bracketing_sample_without_extra_eval(monkeypatch):
    # Access the submodule for monkeypatching, then restore the function
    # reference that __init__.py exports so _opt._line_search stays callable
    # for subsequent tests.
    import importlib
    import simsopt.geo.optimizer_jax_private as _private_pkg

    _line_search_fn = _private_pkg._line_search  # save function reference
    line_search_module = importlib.import_module(
        "simsopt.geo.optimizer_jax_private._line_search"
    )
    _private_pkg._line_search = _line_search_fn  # restore clobbered binding

    def _reuse_cached_cubic(_a, _fa, _fpa, _b, _fb, c, _fc):
        return c

    def _fresh_eval(_alpha):
        return (
            jnp.asarray(9.0, dtype=jnp.float64),
            jnp.asarray(4.0, dtype=jnp.float64),
            jnp.asarray([4.0], dtype=jnp.float64),
        )

    monkeypatch.setattr(line_search_module, "_cubicmin", _reuse_cached_cubic)
    zoom = line_search_module._zoom(
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

_ALL_JAX_BACKEND_MODES = (
    "jax_cpu_parity",
    "jax_gpu_parity",
    "jax_gpu_fast",
    "jax_metal_smoke",
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


def _assert_lbfgs_state_preserved(state, x0, quad, *, ls_status=None):
    assert bool(state.converged) is False
    assert bool(state.failed) is True
    assert int(state.status) == 5
    if ls_status is not None:
        assert int(state.ls_status) == ls_status
    np.testing.assert_allclose(np.asarray(state.x_k), np.asarray(x0))
    np.testing.assert_allclose(np.asarray(state.f_k), np.asarray(quad(x0)))
    np.testing.assert_allclose(np.asarray(state.g_k), np.asarray(x0))
    np.testing.assert_array_equal(
        np.asarray(state.rho_history),
        np.zeros_like(np.asarray(state.rho_history)),
    )
    assert float(state.gamma) == pytest.approx(1.0)


def test_lbfgs_history_updates_wrap_without_shifting():
    from simsopt.geo.optimizer_jax_private import _lbfgs as _lbfgs_module

    history = jnp.zeros((3, 2), dtype=jnp.float64)
    updates = (
        (0, [1.0, 10.0]),
        (1, [2.0, 20.0]),
        (2, [3.0, 30.0]),
        (3, [4.0, 40.0]),
    )
    for step_count, update in updates:
        history = _lbfgs_module._update_history_vectors(
            history,
            jnp.asarray(update, dtype=jnp.float64),
            step_count=jnp.asarray(step_count, dtype=jnp.int32),
        )

    np.testing.assert_allclose(
        np.asarray(history),
        np.asarray(
            [
                [4.0, 40.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=np.float64,
        ),
    )


def test_two_loop_recursion_matches_materialized_history_after_wrap():
    from simsopt.geo.optimizer_jax_private import _lbfgs as _lbfgs_module

    s_oldest_to_newest = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.5, 0.5],
            [0.5, 1.5],
        ],
        dtype=jnp.float64,
    )
    y_oldest_to_newest = jnp.asarray(
        [
            [2.0, 0.0],
            [0.0, 4.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    rho_oldest_to_newest = jnp.asarray([0.5, 0.25, 0.2, 0.2], dtype=jnp.float64)

    ring_state = types.SimpleNamespace(
        k=jnp.asarray(6, dtype=jnp.int32),
        g_k=jnp.asarray([3.0, -1.0], dtype=jnp.float64),
        gamma=jnp.asarray(0.75, dtype=jnp.float64),
        s_history=jnp.asarray(
            [
                [1.5, 0.5],
                [0.5, 1.5],
                [1.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=jnp.float64,
        ),
        y_history=jnp.asarray(
            [
                [3.0, 1.0],
                [1.0, 3.0],
                [2.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=jnp.float64,
        ),
        rho_history=jnp.asarray([0.2, 0.2, 0.5, 0.25], dtype=jnp.float64),
    )
    materialized_state = types.SimpleNamespace(
        k=jnp.asarray(4, dtype=jnp.int32),
        g_k=jnp.asarray([3.0, -1.0], dtype=jnp.float64),
        gamma=jnp.asarray(0.75, dtype=jnp.float64),
        s_history=s_oldest_to_newest,
        y_history=y_oldest_to_newest,
        rho_history=rho_oldest_to_newest,
    )

    ring_direction = _lbfgs_module._two_loop_recursion(ring_state)
    materialized_direction = _lbfgs_module._two_loop_recursion(materialized_state)

    np.testing.assert_allclose(
        np.asarray(ring_direction),
        np.asarray(materialized_direction),
        rtol=1e-12,
        atol=1e-12,
    )


def test_minimize_lbfgs_private_threads_initial_step_size_to_first_line_search(
    monkeypatch,
):
    from simsopt.geo.optimizer_jax_private import _LineSearchResults
    from simsopt.geo.optimizer_jax_private import _lbfgs as _lbfgs_module

    captured = []

    def quad(x):
        return 0.5 * jnp.dot(x, x)

    def fake_line_search(*_args, **kwargs):
        captured.append(kwargs.get("initial_step_size"))
        return _LineSearchResults(
            failed=jnp.asarray(True),
            nit=jnp.asarray(1),
            nfev=jnp.asarray(1),
            ngev=jnp.asarray(1),
            k=jnp.asarray(1),
            a_k=jnp.asarray(0.0, dtype=jnp.float64),
            f_k=jnp.asarray(0.5, dtype=jnp.float64),
            g_k=jnp.asarray([1.0], dtype=jnp.float64),
            status=jnp.asarray(3, dtype=jnp.int32),
        )

    monkeypatch.setattr(
        _lbfgs_module,
        "_line_search_value_and_grad",
        fake_line_search,
    )

    _lbfgs_module._minimize_lbfgs_private(
        quad,
        jnp.asarray([1.0], dtype=jnp.float64),
        maxiter=1,
        initial_step_size=1.0e-4,
    )

    assert len(captured) == 1
    assert captured[0] is not None
    assert getattr(captured[0], "dtype", None) == jnp.dtype(jnp.float64)


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
    def test_line_search_zoom_respects_total_eval_budget(self):
        """Zoom fallback must stay within the caller's total maxiter budget."""

        def quad(x):
            return 0.5 * jnp.dot(x, x)

        result = _opt._line_search(
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

        result = _opt._line_search(
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
    def test_minimize_lbfgs_private_emits_failure_callback_on_nonfinite_step(
        self,
        monkeypatch,
    ):
        """Rejected non-finite L-BFGS proposals must emit a host failure payload."""
        from simsopt.geo.optimizer_jax_private import _LineSearchResults
        from simsopt.geo.optimizer_jax_private import _lbfgs as _lbfgs_module

        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        observed = []
        half = _device_half()

        def quad(x):
            return half * jnp.dot(x, x)

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
                status=jnp.array(0),
            )

        monkeypatch.setattr(
            _lbfgs_module,
            "_line_search_value_and_grad",
            fake_line_search,
        )

        with jax.transfer_guard("disallow"):
            state = _lbfgs_module._minimize_lbfgs_private(
                quad,
                x0,
                maxiter=5,
                gtol=1e-8,
                failure_callback=lambda *payload: observed.append(payload),
            )

        _assert_lbfgs_state_preserved(state, x0, quad, ls_status=0)
        assert len(observed) == 1
        (
            iteration,
            trial_x,
            trial_f,
            trial_g,
            search_direction,
            step_vector,
            step_scale,
            line_search_failed,
            nonfinite_step,
            stalled_step,
            valid_curvature,
            trial_converged,
            ls_status,
        ) = observed[0]
        assert iteration == 1
        np.testing.assert_allclose(trial_x, np.zeros(2), atol=1e-12)
        assert np.isnan(trial_f)
        assert np.isnan(trial_g).all()
        np.testing.assert_allclose(search_direction, np.array([-1.0, 2.0]))
        np.testing.assert_allclose(step_vector, np.array([-1.0, 2.0]))
        assert step_scale == pytest.approx(1.0)
        assert line_search_failed is False
        assert nonfinite_step is True
        assert stalled_step is False
        assert valid_curvature is False
        assert trial_converged is False
        assert ls_status == 0
        assert int(state.invalid_step_log.count) == 1
        used = int(state.invalid_step_log.count)
        np.testing.assert_array_equal(
            np.asarray(state.invalid_step_log.iteration)[:used],
            np.asarray([1], dtype=np.int32),
        )
        np.testing.assert_allclose(
            np.asarray(state.invalid_step_log.step_scale)[:used],
            np.asarray([1.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            np.asarray(state.invalid_step_log.nonfinite_step)[:used],
            np.asarray([True], dtype=bool),
        )

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
        np.testing.assert_allclose(
            np.asarray(state.rho_history),
            np.asarray([2000.0], dtype=np.float64),
            rtol=1e-6,
        )

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_caps_history_to_iteration_budget(
        self,
        monkeypatch,
    ):
        """The private L-BFGS state should not allocate unreachable history slots."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state, _ = _private_lbfgs_quadratic_state(
            monkeypatch,
            x0=x0,
            line_search_kwargs=dict(
                a_k=jnp.array(1.0, dtype=jnp.float64),
                f_k=jnp.array(1.0, dtype=jnp.float64),
                g_k=jnp.array([3.0, -1.0], dtype=jnp.float64),
                status=jnp.array(0),
            ),
            maxiter=1,
            maxcor=7,
        )

        assert np.asarray(state.s_history).shape == (1, x0.size)
        assert np.asarray(state.y_history).shape == (1, x0.size)
        assert np.asarray(state.rho_history).shape == (1,)

    @PRIVATE_OPTIMIZER_RUNTIME
    def test_minimize_lbfgs_private_caps_history_to_problem_dimension(
        self,
        monkeypatch,
    ):
        """L-BFGS should not allocate more correction pairs than the problem rank."""
        x0 = jnp.array([1.0, -2.0], dtype=jnp.float64)
        state, _ = _private_lbfgs_quadratic_state(
            monkeypatch,
            x0=x0,
            line_search_kwargs=dict(
                a_k=jnp.array(1.0, dtype=jnp.float64),
                f_k=jnp.array(1.0, dtype=jnp.float64),
                g_k=jnp.array([3.0, -1.0], dtype=jnp.float64),
                status=jnp.array(0),
            ),
            maxiter=10,
            maxcor=7,
        )

        assert np.asarray(state.s_history).shape == (x0.size, x0.size)
        assert np.asarray(state.y_history).shape == (x0.size, x0.size)
        assert np.asarray(state.rho_history).shape == (x0.size,)

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
        np.testing.assert_allclose(np.asarray(result.jac), np.asarray(optimizer_seed[1]))
        assert float(result.fun) == pytest.approx(0.0)

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
    def test_lbfgs_ondevice_reports_status_six_for_nonfinite_initial_objective(
        self,
    ):
        """Entry with a NaN initial objective must be classified as non-finite."""

        def nan_at_origin(x):
            return jnp.where(
                jnp.all(jnp.equal(x, jnp.zeros_like(x))),
                jnp.asarray(jnp.nan, dtype=x.dtype),
                0.5 * jnp.dot(x, x),
            )

        x0 = jnp.zeros((2,), dtype=jnp.float64)
        result = jax_minimize(
            nan_at_origin,
            x0,
            method="lbfgs-ondevice",
            tol=1e-8,
            maxiter=5,
        )

        assert result.success is False
        assert result.status == 6
        assert "non-finite" in result.message.lower()

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_LBFGS_RUNTIME
    def test_lbfgs_ondevice_reports_status_six_for_nonfinite_initial_gradient(
        self,
    ):
        """Entry with a non-finite initial gradient must be classified as non-finite."""

        def inf_grad_at_origin(x):
            value = jnp.asarray(0.0, dtype=x.dtype)
            grad = jnp.asarray(
                [jnp.inf, jnp.inf],
                dtype=x.dtype,
            )
            return value, grad

        x0 = jnp.zeros((2,), dtype=jnp.float64)
        result = jax_minimize(
            inf_grad_at_origin,
            x0,
            method="lbfgs-ondevice",
            tol=1e-8,
            maxiter=5,
            value_and_grad=True,
        )

        assert result.success is False
        assert result.status == 6
        assert "non-finite" in result.message.lower()


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
    def test_newton_polish_traceable_nonfinite_linear_step_stalls_without_dense_fallback(
        self, monkeypatch
    ):
        """Traceable Newton must fail closed instead of materializing a dense step."""

        def fake_operator_only_linear_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return jnp.full_like(rhs, jnp.nan), jnp.array(False, dtype=bool)

        def forbid_dense_hessian(*_args, **_kwargs):
            raise AssertionError(
                "traceable Newton should not materialize a dense Hessian fallback"
            )

        monkeypatch.setattr(
            _opt,
            "_solve_square_array_system_operator_only",
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
        assert result["hessian"] is None

    @PRIVATE_OPTIMIZER_RUNTIME
    @REQUIRES_PRIVATE_OPTIMIZER_RUNTIME
    def test_newton_polish_traceable_rejected_step_keeps_zero_iterations(
        self, monkeypatch
    ):
        """Rejected traceable Newton steps must not increment nit or emit progress."""
        observed = {"progress_calls": 0}

        def fake_operator_only_linear_solve(_matvec, rhs, *, tol):
            del _matvec, tol
            return -rhs, jnp.array(True, dtype=bool)

        monkeypatch.setattr(
            _opt,
            "_solve_square_array_system_operator_only",
            fake_operator_only_linear_solve,
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
        """On-device L-BFGS progress should surface the same sparse stage updates."""
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
        """limited_memory=True must run the real on-device L-BFGS lane."""
        booz = _make_mock_boozer_surface()
        booz.options["optimizer_backend"] = "ondevice"
        booz.options["limited_memory"] = True

        res = booz.run_code(iota=0.3, G=0.05)

        assert res is not None
        assert res["type"] == "ls"
        assert np.isfinite(res["fun"])
        assert np.max(np.abs(np.asarray(res["jacobian"]))) < 1.0e-6
        assert res["PLU"] is not None
        assert callable(res["vjp"])
        assert res["optimizer_method"] == "lbfgs-ondevice"
