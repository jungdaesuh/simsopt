from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy
import scipy.linalg
from scipy import optimize
from scipy.optimize import _lbfgsb, _lbfgsb_py, minimize
import jax
import jax.numpy as jnp

from simsopt.geo.optimizer_jax_private import _lbfgsb_scipy as lbfgsb


_SETULB_FTOL = 1.0e-12
_SETULB_GTOL = 1.0e-8
_SETULB_KERNEL_MAX_ULP = 16
_SETULB_REPLAY_PREFIX_EVENTS = 9
_SETULB_REPLAY_MAX_ULP = 512
_SETULB_TRACE_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024
_SETULB_DEFAULT_LIMIT = 15000

_BOX_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
_LOWER_ONLY_BOUNDS = ((0.0, None), (None, None), (-1.0, None))
_UPPER_ONLY_BOUNDS = ((None, 1.0), (None, 0.5), (None, None))
_FIXED_VARIABLE_BOUNDS = ((0.25, 0.25), (-1.0, 1.0), (-1.0, 1.0))

_SETULB_BASIC_BOUND_CASES = (
    pytest.param(None, 5, id="unconstrained"),
    pytest.param(_BOX_BOUNDS, 5, id="boxed"),
    pytest.param(_FIXED_VARIABLE_BOUNDS, 5, id="fixed-variable"),
)
_SETULB_ALL_BOUND_CASES = (
    pytest.param(None, 5, id="unconstrained"),
    pytest.param(_BOX_BOUNDS, 5, id="boxed"),
    pytest.param(_LOWER_ONLY_BOUNDS, 5, id="lower-only"),
    pytest.param(_UPPER_ONLY_BOUNDS, 5, id="upper-only"),
    pytest.param(_FIXED_VARIABLE_BOUNDS, 5, id="fixed-variable"),
)
_SETULB_HESS_INV_BOUND_CASES = (
    pytest.param(None, id="unconstrained"),
    pytest.param(_BOX_BOUNDS, id="boxed"),
)


def _quadratic_x0():
    return np.array([0.9, 0.2, -0.4], dtype=np.float64)


def _floatround_objective(x):
    x0 = np.array(
        [
            0.8750000000000278,
            0.7500000000000153,
            0.9499999999999722,
            0.8214285714285992,
            0.6363636363636085,
        ],
        dtype=np.float64,
    )
    x1 = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    x2 = np.array(
        [1.0, 0.0, 0.9889733043149325, 0.0, 0.026353554421041155],
        dtype=np.float64,
    )
    x3 = np.array(
        [1.0, 0.0, 0.9889917442915558, 0.0, 0.020341986743231205],
        dtype=np.float64,
    )

    f0 = 5163.647901211178
    f1 = 5149.8181642072905
    f2 = 5149.379332309634
    f3 = 5149.374490771297

    g0 = np.array(
        [
            -0.5934820547965749,
            1.6251549718258351,
            -71.99168459202559,
            5.346636965797545,
            37.10732723092604,
        ],
        dtype=np.float64,
    )
    g1 = np.array(
        [
            -0.43295349282641515,
            1.008607936794592,
            18.223666726602975,
            31.927010036981997,
            -19.667512518739386,
        ],
        dtype=np.float64,
    )
    g2 = np.array(
        [
            -0.4699874455100256,
            0.9466285353668347,
            -0.016874360242016825,
            48.44999161133457,
            5.819631620590712,
        ],
        dtype=np.float64,
    )
    g3 = np.array(
        [
            -0.46970678696829116,
            0.9612719312174818,
            0.006129809488833699,
            48.43557729419473,
            6.005481418498221,
        ],
        dtype=np.float64,
    )

    for point, value, grad in (
        (x0, f0, g0),
        (x1, f1, g1),
        (x2, f2, g2),
        (x3, f3, g3),
    ):
        if np.allclose(x, point):
            return np.array(value, dtype=np.float64), grad.copy()
    raise ValueError("float-rounding fixture objective not defined at requested point")


def _empty_setulb_workspace(n: int, m: int):
    return {
        "wa": np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, dtype=np.float64),
        "iwa": np.zeros(3 * n, dtype=np.int32),
        "task": np.zeros(2, dtype=np.int32),
        "ln_task": np.zeros(2, dtype=np.int32),
        "lsave": np.zeros(4, dtype=np.int32),
        "isave": np.zeros(44, dtype=np.int32),
        "dsave": np.zeros(29, dtype=np.float64),
    }


def _setulb_trace_event_nbytes(n: int, m: int) -> int:
    float_items = lbfgsb.lbfgsb_workspace_size(n, m) + 3 * n + 31
    int_items = lbfgsb.lbfgsb_iwa_size(n) + 52
    return (
        float_items * np.dtype(np.float64).itemsize
        + int_items * np.dtype(np.int32).itemsize
    )


def _assert_setulb_trace_memory_budget(n: int, m: int, event_count: int) -> None:
    estimated_bytes = _setulb_trace_event_nbytes(n, m) * event_count
    if estimated_bytes > _SETULB_TRACE_MEMORY_BUDGET_BYTES:
        raise MemoryError(
            "SciPy setulb trace budget exceeded: "
            f"n={n}, m={m}, events={event_count}, bytes={estimated_bytes}"
        )


def _quadratic_value_and_grad(x, *, scale=None, target=None):
    if scale is None:
        scale = np.array([1.0, 10.0, 100.0], dtype=np.float64)
    if target is None:
        target = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    shifted = np.asarray(x, dtype=np.float64) - target
    return np.float64(0.5 * np.dot(scale * shifted, shifted)), scale * shifted


def _jax_quadratic_value_and_grad(x):
    scale = jnp.asarray([1.0, 10.0, 100.0], dtype=jnp.float64)
    target = jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float64)
    shifted = x - target
    return jnp.asarray(0.5, dtype=jnp.float64) * jnp.dot(
        scale * shifted,
        shifted,
    ), scale * shifted


def _jax_rosenbrock_value_and_grad(x):
    x0 = x[0]
    x1 = x[1]
    residual = x1 - x0 * x0
    value = 100.0 * residual * residual + (1.0 - x0) * (1.0 - x0)
    grad = jnp.asarray(
        (
            -400.0 * x0 * residual - 2.0 * (1.0 - x0),
            200.0 * residual,
        ),
        dtype=jnp.float64,
    )
    return jnp.asarray(value, dtype=jnp.float64), grad


def _ill_conditioned_diagonal_value_and_grad(x):
    return _quadratic_value_and_grad(
        x,
        scale=np.array([1.0e-3, 1.0, 1.0e3, 1.0e6], dtype=np.float64),
        target=np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64),
    )


def _rosenbrock_value_and_grad(x):
    return np.float64(optimize.rosen(x)), optimize.rosen_der(x).astype(np.float64)


@pytest.mark.parametrize(
    ("scipy_fun", "jax_fun", "x0"),
    (
        pytest.param(
            _quadratic_value_and_grad,
            _jax_quadratic_value_and_grad,
            _quadratic_x0(),
            id="quadratic",
        ),
        pytest.param(
            _rosenbrock_value_and_grad,
            _jax_rosenbrock_value_and_grad,
            np.array([-1.2, 1.0], dtype=np.float64),
            id="rosenbrock",
        ),
    ),
)
def test_jax_live_objective_matches_scipy_fixed_state(scipy_fun, jax_fun, x0):
    expected_value, expected_gradient = scipy_fun(x0)
    actual_value, actual_gradient = jax_fun(jnp.asarray(x0, dtype=jnp.float64))

    np.testing.assert_allclose(
        np.asarray(actual_value),
        np.asarray(expected_value),
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gradient),
        np.asarray(expected_gradient),
        rtol=1e-15,
        atol=1e-15,
    )


def _analytic_trigonometric_value_and_grad(x):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.array([0.15, -0.35, 0.45], dtype=np.float64)
    value = np.sum(0.25 * shifted**2 + 0.05 * np.sin(3.0 * x))
    grad = 0.5 * shifted + 0.15 * np.cos(3.0 * x)
    return np.float64(value), grad.astype(np.float64)


def _encode_bounds_for_setulb(bounds, n):
    low_bnd = np.zeros(n, dtype=np.float64)
    upper_bnd = np.zeros(n, dtype=np.float64)
    nbd = np.zeros(n, dtype=np.int32)
    if bounds is None:
        return low_bnd, upper_bnd, nbd

    for index, (lower, upper) in enumerate(bounds):
        has_lower = lower is not None and not np.isneginf(lower)
        has_upper = upper is not None and not np.isposinf(upper)
        if has_lower:
            low_bnd[index] = np.float64(lower)
        if has_upper:
            upper_bnd[index] = np.float64(upper)
        if has_lower and has_upper:
            nbd[index] = 2
        elif has_lower:
            nbd[index] = 1
        elif has_upper:
            nbd[index] = 3
    return low_bnd, upper_bnd, nbd


def _run_scipy_setulb_trace(
    fun,
    x0,
    bounds=None,
    *,
    m=5,
    ftol=1e-12,
    gtol=1e-8,
    maxls=20,
    maxfun=_SETULB_DEFAULT_LIMIT,
    maxiter=_SETULB_DEFAULT_LIMIT,
):
    x = np.asarray(x0, dtype=np.float64).copy()
    n = len(x)
    low_bnd, upper_bnd, nbd = _encode_bounds_for_setulb(bounds, n)
    f = np.array(0.0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    workspace = _empty_setulb_workspace(n, m)
    trace = []
    factr = ftol / np.finfo(float).eps
    nfev = 0
    njev = 0
    nit = 0

    while True:
        g = g.astype(np.float64)
        _lbfgsb.setulb(
            m,
            x,
            low_bnd,
            upper_bnd,
            nbd,
            f,
            g,
            factr,
            gtol,
            workspace["wa"],
            workspace["iwa"],
            workspace["task"],
            workspace["lsave"],
            workspace["isave"],
            workspace["dsave"],
            maxls,
            workspace["ln_task"],
        )
        _assert_setulb_trace_memory_budget(n, m, len(trace) + 1)
        event = {
            "task": workspace["task"].copy(),
            "ln_task": workspace["ln_task"].copy(),
            "x": x.copy(),
            "f": np.float64(f),
            "g": g.copy(),
            "wa": workspace["wa"].copy(),
            "iwa": workspace["iwa"].copy(),
            "lsave": workspace["lsave"].copy(),
            "isave": workspace["isave"].copy(),
            "dsave": workspace["dsave"].copy(),
        }
        trace.append(event)
        if workspace["task"][0] == 3:
            f, g = fun(x)
            nfev += 1
            njev += 1
            event["requested_f"] = np.float64(f)
            event["requested_g"] = np.asarray(g, dtype=np.float64).copy()
        elif workspace["task"][0] == 1:
            nit += 1
            if nit >= maxiter:
                workspace["task"][0] = 5
                workspace["task"][1] = 504
            elif nfev > maxfun:
                workspace["task"][0] = 5
                workspace["task"][1] = 502
            event["task"] = workspace["task"].copy()
            if workspace["task"][0] != 1:
                event["public_status"] = int(
                    0
                    if workspace["task"][0] == 4
                    else 1
                    if nfev > maxfun or nit >= maxiter
                    else 2
                )
                event["public_message"] = (
                    _lbfgsb_py.status_messages[int(workspace["task"][0])]
                    + ": "
                    + _lbfgsb_py.task_messages[int(workspace["task"][1])]
                )
                event["nfev"] = nfev
                event["njev"] = njev
                event["nit"] = nit
                break
        else:
            event["public_status"] = int(
                0
                if workspace["task"][0] == 4
                else 1
                if nfev > maxfun or nit >= maxiter
                else 2
            )
            event["public_message"] = (
                _lbfgsb_py.status_messages[int(workspace["task"][0])]
                + ": "
                + _lbfgsb_py.task_messages[int(workspace["task"][1])]
            )
            event["nfev"] = nfev
            event["njev"] = njev
            event["nit"] = nit
            break
        event["nfev"] = nfev
        event["njev"] = njev
        event["nit"] = nit

    return trace, workspace


@pytest.fixture(
    params=[
        pytest.param(
            (
                _quadratic_value_and_grad,
                np.array([0.9, 0.2, -0.4], dtype=np.float64),
                None,
                5,
            ),
            id="unconstrained-quadratic",
        ),
        pytest.param(
            (
                _ill_conditioned_diagonal_value_and_grad,
                np.array([0.8, -1.0, 2.0, -3.0], dtype=np.float64),
                None,
                7,
            ),
            id="ill-conditioned-diagonal",
        ),
        pytest.param(
            (
                _rosenbrock_value_and_grad,
                np.array([-1.2, 1.0], dtype=np.float64),
                None,
                5,
            ),
            id="rosenbrock",
        ),
        pytest.param(
            (
                _quadratic_value_and_grad,
                np.array([-0.7, 0.2, -1.5], dtype=np.float64),
                [(0.0, None), (None, None), (-1.0, None)],
                5,
            ),
            id="lower-bounds",
        ),
        pytest.param(
            (
                _quadratic_value_and_grad,
                np.array([1.7, 1.4, -0.4], dtype=np.float64),
                [(None, 1.0), (None, 0.5), (None, None)],
                5,
            ),
            id="upper-bounds",
        ),
        pytest.param(
            (
                _quadratic_value_and_grad,
                np.array([0.9, 0.2, -0.4], dtype=np.float64),
                [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
                5,
            ),
            id="both-bounds",
        ),
        pytest.param(
            (
                _quadratic_value_and_grad,
                np.array([0.9, 0.2, -0.4], dtype=np.float64),
                [(0.25, 0.25), (-1.0, 1.0), (-1.0, 1.0)],
                5,
            ),
            id="fixed-variable",
        ),
        pytest.param(
            (
                _analytic_trigonometric_value_and_grad,
                np.array([0.7, -0.8, 0.2], dtype=np.float64),
                [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
                5,
            ),
            id="analytic-gradient",
        ),
    ]
)
def scipy_replay_case(request):
    return request.param


def test_installed_scipy_lbfgsb_oracle_is_pinned_to_1171():
    assert scipy.__version__ == "1.17.1"
    assert Path(_lbfgsb_py.__file__).name == "_lbfgsb_py.py"
    assert Path(_lbfgsb.__file__).name.startswith("_lbfgsb.")
    assert Path(_lbfgsb_py.__file__).parent == Path(_lbfgsb.__file__).parent
    assert _lbfgsb.setulb.__doc__ == (
        "setulb(m,x,l,u,nbd,f,g,factr,pgtol,wa,iwa,task,lsave,isave,dsave,maxls,ln_task)"
    )


def test_scipy_setulb_floatround_does_not_step_outside_box():
    n = 5
    m = 10
    factr = 1e7
    pgtol = 1e-5
    maxls = 20
    nbd = np.full(shape=(n,), fill_value=2, dtype=np.int32)
    low_bnd = np.zeros(n, dtype=np.float64)
    upper_bnd = np.ones(n, dtype=np.float64)
    x = np.array(
        [
            0.8750000000000278,
            0.7500000000000153,
            0.9499999999999722,
            0.8214285714285992,
            0.6363636363636085,
        ],
        dtype=np.float64,
    )
    f = np.array(0.0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    workspace = _empty_setulb_workspace(n, m)

    for _ in range(7):
        f, g = _floatround_objective(x)
        _lbfgsb.setulb(
            m,
            x,
            low_bnd,
            upper_bnd,
            nbd,
            f,
            g,
            factr,
            pgtol,
            workspace["wa"],
            workspace["iwa"],
            workspace["task"],
            workspace["lsave"],
            workspace["isave"],
            workspace["dsave"],
            maxls,
            workspace["ln_task"],
        )
        assert (x <= upper_bnd).all()
        assert (x >= low_bnd).all()


def test_scipy_lbfgsb_accepts_float32_gradient_regression():
    def fun_single_precision(x):
        x = x.astype(np.float32)
        return np.sum(x**2), 2 * x

    res = minimize(
        fun_single_precision,
        x0=np.array([1.0, 1.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
    )

    assert_allclose(res.fun, 0.0, atol=1e-15)


def test_scipy_lbfgsb_rejects_nonpositive_maxls():
    def f(x):
        return np.dot(x, x), 2.0 * x

    with np.testing.assert_raises_regex(ValueError, "maxls must be positive."):
        minimize(
            fun=f,
            x0=np.array([1.0, 1.0], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={"maxls": 0},
        )


def test_scipy_lbfgsb_maxls_one_abnormal_line_search_status():
    result = optimize.minimize(
        optimize.rosen,
        np.array([-1.2, 1.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=optimize.rosen_der,
        options={"maxls": 1},
    )

    assert result.success is False


def test_scipy_lbfgsb_invalid_bounds_error_message():
    def f(x):
        return np.dot(x, x), 2.0 * x

    with np.testing.assert_raises_regex(
        ValueError,
        "upper bound is less",
    ):
        minimize(
            fun=f,
            x0=np.array([1.0], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[(2.0, 1.0)],
        )


def test_scipy_hess_inv_matvec_matches_dense_on_scalar_quartic():
    def f(x):
        return x**4, 4 * x**3

    for gtol in [1e-8, 1e-12, 1e-20]:
        for maxcor in range(20, 35):
            result = minimize(
                fun=f,
                jac=True,
                method="L-BFGS-B",
                x0=20,
                options={"gtol": gtol, "maxcor": maxcor},
            )

            h1 = result.hess_inv(np.array([1])).reshape(1, 1)
            h2 = result.hess_inv.todense()

            assert_allclose(h1, h2)


def test_scipy_hess_inv_matches_bfgs_on_two_dimensional_quadratic():
    h0 = [[3, 0], [1, 2]]

    def f(x):
        return np.dot(x, np.dot(scipy.linalg.inv(h0), x))

    result1 = minimize(fun=f, method="L-BFGS-B", x0=[10, 20])
    result2 = minimize(fun=f, method="BFGS", x0=[10, 20])

    h1 = result1.hess_inv.todense()
    h2 = np.vstack(
        (
            result1.hess_inv(np.array([1, 0])),
            result1.hess_inv(np.array([0, 1])),
        )
    )

    assert_allclose(
        result1.hess_inv(np.array([1, 0]).reshape(2, 1)).reshape(-1),
        result1.hess_inv(np.array([1, 0])),
    )
    assert_allclose(h1, h2)
    assert_allclose(h1, result2.hess_inv, rtol=1e-2, atol=0.03)


def test_scipy_hess_inv_todense_matches_old_dense_implementation():
    def todense_old_impl(hess_inv):
        s, y, n_corrs, rho = (
            hess_inv.sk,
            hess_inv.yk,
            hess_inv.n_corrs,
            hess_inv.rho,
        )
        identity = np.eye(*hess_inv.shape, dtype=hess_inv.dtype)
        hk = identity

        for i in range(n_corrs):
            a1 = identity - s[i][:, np.newaxis] * y[i][np.newaxis, :] * rho[i]
            a2 = identity - y[i][:, np.newaxis] * s[i][np.newaxis, :] * rho[i]

            hk = np.dot(a1, np.dot(hk, a2)) + (
                rho[i] * s[i][:, np.newaxis] * s[i][np.newaxis, :]
            )
        return hk

    h0 = [[3, 0], [1, 2]]

    def f(x):
        return np.dot(x, np.dot(scipy.linalg.inv(h0), x))

    result = minimize(fun=f, method="L-BFGS-B", x0=[10, 20])
    assert_allclose(result.hess_inv.todense(), todense_old_impl(result.hess_inv))


def test_scipy_setulb_reverse_communication_trace_records_replay_state():
    trace, workspace = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        _quadratic_x0(),
        _BOX_BOUNDS,
    )

    task_pairs = [tuple(event["task"]) for event in trace]
    assert task_pairs[0] == (3, 301)
    assert (1, 0) in task_pairs
    assert task_pairs[-1] == (4, 402)
    assert all(event["x"].dtype == np.float64 for event in trace)
    assert all(event["g"].dtype == np.float64 for event in trace)
    assert workspace["wa"].dtype == np.float64
    assert workspace["iwa"].dtype == np.int32
    assert workspace["isave"][30] > 0


def test_scipy_setulb_trace_memory_budget_rejects_expensive_trace_settings():
    n = 128
    m = 10
    event_count = (
        _SETULB_TRACE_MEMORY_BUDGET_BYTES // _setulb_trace_event_nbytes(n, m) + 1
    )

    with np.testing.assert_raises_regex(
        MemoryError,
        "SciPy setulb trace budget exceeded",
    ):
        _assert_setulb_trace_memory_budget(n, m, event_count)


def _assert_float_array_matches(actual, expected, max_numeric_ulp):
    actual = np.asarray(actual)
    if max_numeric_ulp == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_array_max_ulp(actual, expected, maxulp=max_numeric_ulp)


def _assert_jax_setulb_state_matches_scipy_event(
    actual,
    expected,
    *,
    max_numeric_ulp=0,
    max_x_ulp=None,
    max_workspace_ulp=None,
):
    if max_x_ulp is None:
        max_x_ulp = max_numeric_ulp
    if max_workspace_ulp is None:
        max_workspace_ulp = max_numeric_ulp

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), expected["task"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.ln_task),
        expected["ln_task"],
    )
    _assert_float_array_matches(actual.x, expected["x"], max_x_ulp)
    _assert_float_array_matches(actual.f, expected["f"], max_numeric_ulp)
    _assert_float_array_matches(actual.g, expected["g"], max_numeric_ulp)
    _assert_float_array_matches(actual.workspace.wa, expected["wa"], max_workspace_ulp)
    np.testing.assert_array_equal(np.asarray(actual.workspace.iwa), expected["iwa"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.lsave),
        expected["lsave"],
    )
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        expected["isave"],
    )
    _assert_float_array_matches(
        actual.workspace.dsave, expected["dsave"], max_workspace_ulp
    )


def _assert_jax_setulb_control_matches_scipy_event(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual.workspace.task), expected["task"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.ln_task),
        expected["ln_task"],
    )
    np.testing.assert_array_equal(np.asarray(actual.workspace.iwa), expected["iwa"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.lsave),
        expected["lsave"],
    )
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        expected["isave"],
    )


def _assert_jax_mainlb_terminal_matches_scipy_event(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual.workspace.task), expected["task"])
    assert int(actual.nfev) == expected["nfev"]
    assert int(actual.njev) == expected["njev"]
    assert int(actual.n_iterations) == expected["nit"]


def _jax_setulb_initial_state(
    x0,
    bounds,
    m,
    *,
    ftol=_SETULB_FTOL,
    gtol=_SETULB_GTOL,
    maxls=20,
):
    return lbfgsb.lbfgsb_initial_state(
        x0,
        m=m,
        bounds=bounds,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
    )


def _jax_setulb_started_state(
    x0,
    bounds,
    m,
    *,
    ftol=_SETULB_FTOL,
    gtol=_SETULB_GTOL,
):
    return lbfgsb.lbfgsb_setulb(
        _jax_setulb_initial_state(x0, bounds, m, ftol=ftol, gtol=gtol)
    )


def _jax_setulb_evaluated_state(state, fun, *, nfev=1, njev=1):
    value, gradient = fun(np.asarray(state.x))
    return state._replace(
        f=jnp.asarray(value, dtype=jnp.float64),
        g=jnp.asarray(gradient, dtype=jnp.float64),
        nfev=jnp.asarray(nfev, dtype=jnp.int32),
        njev=jnp.asarray(njev, dtype=jnp.int32),
    )


def _jax_setulb_replayed_evaluation_state(state, scipy_event, *, nfev, njev):
    return state._replace(
        f=jnp.asarray(scipy_event["requested_f"], dtype=jnp.float64),
        g=jnp.asarray(scipy_event["requested_g"], dtype=jnp.float64),
        nfev=jnp.asarray(nfev, dtype=jnp.int32),
        njev=jnp.asarray(njev, dtype=jnp.int32),
    )


def _jax_state_with_scipy_workspace(state, workspace):
    return state._replace(
        workspace=lbfgsb.LbfgsbWorkspace(
            wa=jnp.asarray(workspace["wa"], dtype=jnp.float64),
            iwa=jnp.asarray(workspace["iwa"], dtype=jnp.int32),
            task=jnp.asarray(workspace["task"], dtype=jnp.int32),
            ln_task=jnp.asarray(workspace["ln_task"], dtype=jnp.int32),
            lsave=jnp.asarray(workspace["lsave"], dtype=jnp.int32),
            isave=jnp.asarray(workspace["isave"], dtype=jnp.int32),
            dsave=jnp.asarray(workspace["dsave"], dtype=jnp.float64),
        )
    )


def _jax_setulb_first_line_search_request(
    x0,
    bounds,
    m,
    fun=_quadratic_value_and_grad,
    *,
    ftol=_SETULB_FTOL,
    gtol=_SETULB_GTOL,
):
    state = _jax_setulb_started_state(x0, bounds, m, ftol=ftol, gtol=gtol)
    evaluated_state = _jax_setulb_evaluated_state(state, fun)
    return lbfgsb.lbfgsb_setulb(evaluated_state)


def _jax_setulb_new_x_state(
    x0,
    bounds,
    m,
    fun=_quadratic_value_and_grad,
    *,
    ftol=_SETULB_FTOL,
    gtol=_SETULB_GTOL,
):
    state = _jax_setulb_first_line_search_request(
        x0,
        bounds,
        m,
        fun,
        ftol=ftol,
        gtol=gtol,
    )
    evaluated_state = _jax_setulb_evaluated_state(state, fun, nfev=2, njev=2)
    return lbfgsb.lbfgsb_setulb(evaluated_state)


def _jax_setulb_second_line_search_request(
    x0,
    bounds,
    m,
    fun=_quadratic_value_and_grad,
    *,
    ftol=_SETULB_FTOL,
    gtol=_SETULB_GTOL,
):
    state = _jax_setulb_new_x_state(
        x0,
        bounds,
        m,
        fun,
        ftol=ftol,
        gtol=gtol,
    )
    return lbfgsb.lbfgsb_setulb(state)


def _assert_jax_setulb_replays_scipy_prefix(fun, x0, bounds, m):
    scipy_trace, _ = _run_scipy_setulb_trace(
        fun,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=5,
    )
    state = _jax_setulb_initial_state(x0, bounds, m)
    nfev = 0
    njev = 0
    assert len(scipy_trace) >= _SETULB_REPLAY_PREFIX_EVENTS

    for expected in scipy_trace[:_SETULB_REPLAY_PREFIX_EVENTS]:
        actual = lbfgsb.lbfgsb_setulb(state)
        _assert_jax_setulb_state_matches_scipy_event(
            actual,
            expected,
            max_x_ulp=_SETULB_REPLAY_MAX_ULP,
            max_workspace_ulp=_SETULB_REPLAY_MAX_ULP,
        )

        if int(actual.workspace.task[0]) == lbfgsb.FG:
            nfev += 1
            njev += 1
            state = _jax_setulb_replayed_evaluation_state(
                actual,
                expected,
                nfev=nfev,
                njev=njev,
            )
            assert int(state.nfev) == expected["nfev"]
            assert int(state.njev) == expected["njev"]
            assert int(state.n_iterations) == expected["nit"]
        else:
            state = actual
            assert int(actual.nfev) == expected["nfev"]
            assert int(actual.njev) == expected["njev"]
            assert int(actual.n_iterations) == expected["nit"]


def _assert_jax_setulb_replays_scipy_control_trace(fun, x0, bounds, m):
    scipy_trace, _ = _run_scipy_setulb_trace(
        fun,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )
    state = _jax_setulb_initial_state(x0, bounds, m)
    nfev = 0
    njev = 0

    for expected in scipy_trace:
        actual = lbfgsb.lbfgsb_setulb(state)
        _assert_jax_setulb_control_matches_scipy_event(actual, expected)

        if int(actual.workspace.task[0]) == lbfgsb.FG:
            nfev += 1
            njev += 1
            state = _jax_setulb_replayed_evaluation_state(
                actual,
                expected,
                nfev=nfev,
                njev=njev,
            )
            assert int(state.nfev) == expected["nfev"]
            assert int(state.njev) == expected["njev"]
            assert int(state.n_iterations) == expected["nit"]
        else:
            state = actual
            assert int(actual.nfev) == expected["nfev"]
            assert int(actual.njev) == expected["njev"]
            assert int(actual.n_iterations) == expected["nit"]

    np.testing.assert_array_equal(
        np.asarray(state.workspace.task), scipy_trace[-1]["task"]
    )


@pytest.mark.parametrize(
    ("x0", "bounds", "m"),
    [
        pytest.param(
            _quadratic_x0(),
            None,
            5,
            id="unconstrained",
        ),
        pytest.param(
            np.array([-0.7, 0.2, -1.5], dtype=np.float64),
            _LOWER_ONLY_BOUNDS,
            5,
            id="project-lower",
        ),
        pytest.param(
            np.array([1.7, 1.4, -0.4], dtype=np.float64),
            _UPPER_ONLY_BOUNDS,
            5,
            id="project-upper",
        ),
        pytest.param(
            _quadratic_x0(),
            _FIXED_VARIABLE_BOUNDS,
            7,
            id="fixed-variable",
        ),
    ],
)
def test_jax_setulb_initial_start_transition_matches_scipy(x0, bounds, m):
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=1,
    )
    expected = scipy_trace[0]

    state = _jax_setulb_initial_state(x0, bounds, m)
    actual = lbfgsb.lbfgsb_setulb(state)

    _assert_jax_setulb_state_matches_scipy_event(actual, expected)
    assert int(actual.n_iterations) == 0
    assert int(actual.nfev) == 0
    assert int(actual.njev) == 0


def test_jax_setulb_initial_start_transition_is_jittable_for_static_workspace():
    x0 = _quadratic_x0()
    bounds = _BOX_BOUNDS
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=1,
    )

    actual = jax.jit(lbfgsb.lbfgsb_setulb)(_jax_setulb_initial_state(x0, bounds, 5))

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), [3, 301])
    np.testing.assert_array_equal(np.asarray(actual.x), scipy_trace[0]["x"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        scipy_trace[0]["isave"],
    )


def _lower_bound_projected_converged_value_and_grad(x):
    x = np.asarray(x, dtype=np.float64)
    shifted = x + np.array([1.0], dtype=np.float64)
    return np.float64(0.5 * np.dot(shifted, shifted)), shifted


@pytest.mark.parametrize(
    ("fun", "x0", "bounds", "m"),
    [
        pytest.param(
            _quadratic_value_and_grad,
            np.array([0.25, -0.5, 0.75], dtype=np.float64),
            None,
            5,
            id="unconstrained-zero-gradient",
        ),
        pytest.param(
            _lower_bound_projected_converged_value_and_grad,
            np.array([-2.0], dtype=np.float64),
            [(0.0, None)],
            3,
            id="lower-bound-projected-gradient",
        ),
    ],
)
def test_jax_setulb_fg_start_reentry_convergence_matches_scipy(fun, x0, bounds, m):
    scipy_trace, _ = _run_scipy_setulb_trace(
        fun,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )
    expected = scipy_trace[1]

    state = _jax_setulb_started_state(x0, bounds, m)
    evaluated_state = _jax_setulb_evaluated_state(state, fun)
    actual = lbfgsb.lbfgsb_setulb(evaluated_state)

    _assert_jax_setulb_state_matches_scipy_event(actual, expected)
    assert int(actual.n_iterations) == 0
    assert int(actual.nfev) == 1
    assert int(actual.njev) == 1


def test_jax_setulb_fg_start_reentry_convergence_is_jittable():
    x0 = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    state = _jax_setulb_started_state(x0, None, 5)
    evaluated_state = state._replace(
        f=jnp.asarray(0.0, dtype=jnp.float64),
        g=jnp.zeros_like(state.x, dtype=jnp.float64),
        nfev=jnp.asarray(1, dtype=jnp.int32),
        njev=jnp.asarray(1, dtype=jnp.int32),
    )

    actual = jax.jit(lbfgsb.lbfgsb_setulb)(evaluated_state)

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), [4, 401])
    assert int(actual.workspace.isave[33]) == 1
    assert float(actual.workspace.dsave[12]) == 0.0


@pytest.mark.parametrize(
    ("x0", "bounds", "m"),
    [
        pytest.param(
            _quadratic_x0(),
            None,
            5,
            id="unconstrained",
        ),
        pytest.param(
            _quadratic_x0(),
            _BOX_BOUNDS,
            5,
            id="boxed",
        ),
        pytest.param(
            _quadratic_x0(),
            _FIXED_VARIABLE_BOUNDS,
            7,
            id="fixed-variable",
        ),
    ],
)
def test_jax_setulb_fg_start_reentry_first_line_search_request_matches_scipy(
    x0,
    bounds,
    m,
):
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=1,
    )
    expected = scipy_trace[1]

    state = _jax_setulb_started_state(x0, bounds, m)
    evaluated_state = _jax_setulb_evaluated_state(state, _quadratic_value_and_grad)
    actual = lbfgsb.lbfgsb_setulb(evaluated_state)

    _assert_jax_setulb_state_matches_scipy_event(actual, expected)
    assert int(actual.n_iterations) == 0
    assert int(actual.nfev) == 1
    assert int(actual.njev) == 1


def test_jax_setulb_fg_start_reentry_first_line_search_request_is_jittable():
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        None,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=1,
    )
    state = _jax_setulb_started_state(x0, None, 5)
    evaluated_state = _jax_setulb_evaluated_state(state, _quadratic_value_and_grad)

    actual = jax.jit(lbfgsb.lbfgsb_setulb)(evaluated_state)

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), [3, 302])
    np.testing.assert_array_equal(np.asarray(actual.x), scipy_trace[1]["x"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        scipy_trace[1]["isave"],
    )


@pytest.mark.parametrize(
    ("x0", "bounds", "m"),
    [
        pytest.param(
            _quadratic_x0(),
            None,
            5,
            id="unconstrained",
        ),
        pytest.param(
            _quadratic_x0(),
            _BOX_BOUNDS,
            5,
            id="boxed",
        ),
        pytest.param(
            _quadratic_x0(),
            _FIXED_VARIABLE_BOUNDS,
            7,
            id="fixed-variable",
        ),
    ],
)
def test_jax_setulb_fg_lnsrch_reentry_new_x_matches_scipy(x0, bounds, m):
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=2,
    )
    expected = scipy_trace[2]
    np.testing.assert_array_equal(expected["task"], [1, 0])

    state = _jax_setulb_first_line_search_request(x0, bounds, m)
    evaluated_state = _jax_setulb_evaluated_state(
        state,
        _quadratic_value_and_grad,
        nfev=2,
        njev=2,
    )
    actual = lbfgsb.lbfgsb_setulb(evaluated_state)

    _assert_jax_setulb_state_matches_scipy_event(actual, expected)
    assert int(actual.n_iterations) == 1
    assert int(actual.nfev) == 2
    assert int(actual.njev) == 2


def test_jax_setulb_fg_lnsrch_reentry_new_x_is_jittable():
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        None,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=2,
    )
    state = _jax_setulb_first_line_search_request(x0, None, 5)
    evaluated_state = _jax_setulb_evaluated_state(
        state,
        _quadratic_value_and_grad,
        nfev=2,
        njev=2,
    )

    actual = jax.jit(lbfgsb.lbfgsb_setulb)(evaluated_state)

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), [1, 0])
    np.testing.assert_array_equal(np.asarray(actual.x), scipy_trace[2]["x"])
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        scipy_trace[2]["isave"],
    )
    assert int(actual.n_iterations) == 1


def _one_dimensional_quadratic_value_and_grad(x):
    x = np.asarray(x, dtype=np.float64)
    return np.float64(0.5 * np.dot(x, x)), x.copy()


def test_jax_setulb_new_x_reentry_projected_gradient_convergence_matches_scipy():
    x0 = np.array([1.0], dtype=np.float64)
    scipy_trace, _ = _run_scipy_setulb_trace(
        _one_dimensional_quadratic_value_and_grad,
        x0,
        None,
        m=3,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )
    expected = scipy_trace[3]
    np.testing.assert_array_equal(expected["task"], [4, 401])

    state = _jax_setulb_new_x_state(
        x0,
        None,
        3,
        _one_dimensional_quadratic_value_and_grad,
    )
    actual = lbfgsb.lbfgsb_setulb(state)

    _assert_jax_setulb_state_matches_scipy_event(actual, expected)
    assert int(actual.n_iterations) == 1
    assert int(actual.nfev) == 2
    assert int(actual.njev) == 2


def test_jax_setulb_new_x_reentry_relative_reduction_convergence_matches_scipy():
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        None,
        m=5,
        ftol=1.0,
        gtol=1.0e-12,
    )
    expected = scipy_trace[3]
    np.testing.assert_array_equal(expected["task"], [4, 402])

    state = _jax_setulb_new_x_state(
        x0,
        None,
        5,
        ftol=1.0,
        gtol=1.0e-12,
    )
    actual = lbfgsb.lbfgsb_setulb(state)

    _assert_jax_setulb_state_matches_scipy_event(actual, expected)
    assert int(actual.n_iterations) == 1
    assert int(actual.nfev) == 2
    assert int(actual.njev) == 2


@pytest.mark.parametrize(("bounds", "m"), _SETULB_ALL_BOUND_CASES)
def test_jax_setulb_new_x_reentry_next_line_search_matches_scipy(bounds, m):
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=2,
    )
    expected = scipy_trace[3]
    np.testing.assert_array_equal(expected["task"], [3, 302])

    state = _jax_setulb_new_x_state(x0, bounds, m)
    actual = lbfgsb.lbfgsb_setulb(state)

    _assert_jax_setulb_state_matches_scipy_event(
        actual,
        expected,
        max_numeric_ulp=_SETULB_KERNEL_MAX_ULP,
    )
    assert int(actual.n_iterations) == 1
    assert int(actual.nfev) == 2
    assert int(actual.njev) == 2


def test_jax_setulb_new_x_reentry_next_line_search_is_jittable():
    x0 = _quadratic_x0()
    bounds = _BOX_BOUNDS
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=2,
    )
    state = _jax_setulb_new_x_state(x0, bounds, 5)

    actual = jax.jit(lbfgsb.lbfgsb_setulb)(state)

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), [3, 302])
    np.testing.assert_array_max_ulp(
        np.asarray(actual.x),
        scipy_trace[3]["x"],
        maxulp=_SETULB_KERNEL_MAX_ULP,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        scipy_trace[3]["isave"],
    )
    assert int(actual.n_iterations) == 1


@pytest.mark.parametrize(("bounds", "m"), _SETULB_BASIC_BOUND_CASES)
def test_jax_setulb_second_line_search_accepts_new_x_matches_scipy(bounds, m):
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=3,
    )
    expected = scipy_trace[4]
    np.testing.assert_array_equal(expected["task"], [1, 0])

    state = _jax_setulb_second_line_search_request(x0, bounds, m)
    evaluated_state = _jax_setulb_replayed_evaluation_state(
        state,
        scipy_trace[3],
        nfev=3,
        njev=3,
    )
    actual = lbfgsb.lbfgsb_setulb(evaluated_state)

    _assert_jax_setulb_state_matches_scipy_event(
        actual,
        expected,
        max_numeric_ulp=_SETULB_KERNEL_MAX_ULP,
    )
    assert int(actual.n_iterations) == 2
    assert int(actual.nfev) == 3
    assert int(actual.njev) == 3


def test_jax_setulb_second_line_search_accepts_new_x_is_jittable():
    x0 = _quadratic_x0()
    bounds = _BOX_BOUNDS
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=3,
    )
    state = _jax_setulb_second_line_search_request(x0, bounds, 5)
    evaluated_state = _jax_setulb_replayed_evaluation_state(
        state,
        scipy_trace[3],
        nfev=3,
        njev=3,
    )

    actual = jax.jit(lbfgsb.lbfgsb_setulb)(evaluated_state)

    np.testing.assert_array_equal(np.asarray(actual.workspace.task), [1, 0])
    np.testing.assert_array_max_ulp(
        np.asarray(actual.x),
        scipy_trace[4]["x"],
        maxulp=_SETULB_KERNEL_MAX_ULP,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.isave),
        scipy_trace[4]["isave"],
    )


@pytest.mark.parametrize(("bounds", "m"), _SETULB_ALL_BOUND_CASES)
def test_jax_setulb_frozen_replay_prefix_matches_scipy(bounds, m):
    _assert_jax_setulb_replays_scipy_prefix(
        _quadratic_value_and_grad,
        _quadratic_x0(),
        bounds,
        m,
    )


@pytest.mark.parametrize(("bounds", "m"), _SETULB_ALL_BOUND_CASES)
def test_jax_setulb_frozen_control_trace_matches_scipy(bounds, m):
    _assert_jax_setulb_replays_scipy_control_trace(
        _quadratic_value_and_grad,
        _quadratic_x0(),
        bounds,
        m,
    )


@pytest.mark.parametrize(("bounds", "m"), _SETULB_ALL_BOUND_CASES)
def test_jax_mainlb_live_quadratic_terminal_control_matches_scipy(bounds, m):
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )

    actual = lbfgsb.lbfgsb_mainlb(
        _jax_quadratic_value_and_grad,
        _jax_setulb_initial_state(x0, bounds, m),
        maxiter=_SETULB_DEFAULT_LIMIT,
        maxfun=_SETULB_DEFAULT_LIMIT,
    )

    _assert_jax_mainlb_terminal_matches_scipy_event(actual, scipy_trace[-1])


@pytest.mark.parametrize(
    ("maxiter", "maxfun"),
    [
        pytest.param(1, _SETULB_DEFAULT_LIMIT, id="iteration-limit"),
        pytest.param(_SETULB_DEFAULT_LIMIT, 1, id="function-limit"),
        pytest.param(1, 1, id="iteration-limit-precedes-function-limit"),
    ],
)
def test_jax_mainlb_deferred_limits_match_scipy(maxiter, maxfun):
    x0 = _quadratic_x0()
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        None,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
        maxiter=maxiter,
        maxfun=maxfun,
    )

    actual = lbfgsb.lbfgsb_mainlb(
        _jax_quadratic_value_and_grad,
        _jax_setulb_initial_state(x0, None, 5),
        maxiter=maxiter,
        maxfun=maxfun,
    )

    _assert_jax_mainlb_terminal_matches_scipy_event(actual, scipy_trace[-1])


def test_jax_mainlb_maxls_one_abnormal_matches_scipy():
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    scipy_trace, _ = _run_scipy_setulb_trace(
        _rosenbrock_value_and_grad,
        x0,
        None,
        m=10,
        maxls=1,
        maxiter=_SETULB_DEFAULT_LIMIT,
        maxfun=_SETULB_DEFAULT_LIMIT,
    )

    actual = lbfgsb.lbfgsb_mainlb(
        _jax_rosenbrock_value_and_grad,
        _jax_setulb_initial_state(x0, None, 10, maxls=1),
        maxiter=_SETULB_DEFAULT_LIMIT,
        maxfun=_SETULB_DEFAULT_LIMIT,
    )

    expected = scipy_trace[-1]
    _assert_jax_setulb_control_matches_scipy_event(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.workspace.task), [lbfgsb.ABNORMAL, 0]
    )
    _assert_float_array_matches(actual.x, expected["x"], _SETULB_REPLAY_MAX_ULP)
    _assert_float_array_matches(actual.f, expected["f"], _SETULB_REPLAY_MAX_ULP)
    _assert_float_array_matches(actual.g, expected["g"], _SETULB_REPLAY_MAX_ULP)
    _assert_jax_mainlb_terminal_matches_scipy_event(actual, expected)


def test_jax_setulb_line_search_restart_with_history_matches_scipy():
    x0 = _quadratic_x0()
    m = 5
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        None,
        m=m,
        maxiter=_SETULB_DEFAULT_LIMIT,
        maxfun=_SETULB_DEFAULT_LIMIT,
    )
    event = next(
        event
        for event in scipy_trace
        if tuple(event["task"]) == (lbfgsb.FG, lbfgsb.FG_LNSRCH)
        and event["isave"][27] > 0
    )
    n = len(event["x"])
    _, _, _, _, _, _, _, _, lr, ld, lt, lxp, _ = lbfgsb._lbfgsb_workspace_offsets(n, m)

    scipy_workspace = {
        "wa": event["wa"].copy(),
        "iwa": event["iwa"].copy(),
        "task": event["task"].copy(),
        "ln_task": event["ln_task"].copy(),
        "lsave": event["lsave"].copy(),
        "isave": event["isave"].copy(),
        "dsave": event["dsave"].copy(),
    }
    scipy_workspace["wa"][ld:lt] = event["requested_g"]
    scipy_workspace["isave"][35] = 0
    scipy_x = event["x"].copy()
    scipy_f = np.array(event["requested_f"], dtype=np.float64)
    scipy_g = event["requested_g"].copy()
    _lbfgsb.setulb(
        m,
        scipy_x,
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.int32),
        scipy_f,
        scipy_g,
        _SETULB_FTOL / np.finfo(float).eps,
        _SETULB_GTOL,
        scipy_workspace["wa"],
        scipy_workspace["iwa"],
        scipy_workspace["task"],
        scipy_workspace["lsave"],
        scipy_workspace["isave"],
        scipy_workspace["dsave"],
        20,
        scipy_workspace["ln_task"],
    )
    expected = {
        "task": scipy_workspace["task"],
        "ln_task": scipy_workspace["ln_task"],
        "x": scipy_x,
        "f": scipy_f,
        "g": scipy_g,
        "wa": scipy_workspace["wa"],
        "iwa": scipy_workspace["iwa"],
        "lsave": scipy_workspace["lsave"],
        "isave": scipy_workspace["isave"],
        "dsave": scipy_workspace["dsave"],
    }

    jax_workspace = {
        "wa": event["wa"].copy(),
        "iwa": event["iwa"].copy(),
        "task": event["task"].copy(),
        "ln_task": event["ln_task"].copy(),
        "lsave": event["lsave"].copy(),
        "isave": event["isave"].copy(),
        "dsave": event["dsave"].copy(),
    }
    jax_workspace["wa"][ld:lt] = event["requested_g"]
    jax_workspace["isave"][35] = 0
    state = _jax_state_with_scipy_workspace(
        _jax_setulb_initial_state(x0, None, m),
        jax_workspace,
    )._replace(
        x=jnp.asarray(event["x"], dtype=jnp.float64),
        f=jnp.asarray(event["requested_f"], dtype=jnp.float64),
        g=jnp.asarray(event["requested_g"], dtype=jnp.float64),
        nfev=jnp.asarray(event["nfev"], dtype=jnp.int32),
        njev=jnp.asarray(event["njev"], dtype=jnp.int32),
        n_iterations=jnp.asarray(event["nit"], dtype=jnp.int32),
    )

    actual = lbfgsb.lbfgsb_setulb(state)

    np.testing.assert_array_equal(expected["task"], [lbfgsb.FG, lbfgsb.FG_LNSRCH])
    assert expected["isave"][27] == 0
    assert expected["isave"][30] == 0
    assert expected["dsave"][0] == 1.0
    _assert_jax_setulb_state_matches_scipy_event(
        actual,
        expected,
        max_numeric_ulp=_SETULB_REPLAY_MAX_ULP,
        max_x_ulp=_SETULB_REPLAY_MAX_ULP,
        max_workspace_ulp=_SETULB_REPLAY_MAX_ULP,
    )


def test_jax_mainlb_zero_iteration_limit_stops_like_scipy():
    x0 = _quadratic_x0()
    actual = jax.jit(
        lambda state: lbfgsb.lbfgsb_mainlb(
            _jax_quadratic_value_and_grad,
            state,
            maxiter=0,
            maxfun=_SETULB_DEFAULT_LIMIT,
        )
    )(_jax_setulb_initial_state(x0, None, 5))

    np.testing.assert_array_equal(
        np.asarray(actual.workspace.task),
        [lbfgsb.STOP, lbfgsb.STOP_ITERC],
    )


def test_jax_mainlb_live_quadratic_is_jittable():
    x0 = _quadratic_x0()
    bounds = _BOX_BOUNDS
    scipy_trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=5,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )

    def run_mainlb(state):
        return lbfgsb.lbfgsb_mainlb(
            _jax_quadratic_value_and_grad,
            state,
            maxiter=_SETULB_DEFAULT_LIMIT,
            maxfun=_SETULB_DEFAULT_LIMIT,
        )

    actual = jax.jit(run_mainlb)(_jax_setulb_initial_state(x0, bounds, 5))

    _assert_jax_mainlb_terminal_matches_scipy_event(actual, scipy_trace[-1])


@pytest.mark.parametrize("bounds", _SETULB_HESS_INV_BOUND_CASES)
def test_jax_lbfgsb_inverse_hessian_history_matches_scipy_hess_inv(bounds):
    x0 = _quadratic_x0()
    m = 5
    _, workspace = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        bounds,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )
    scipy_result = minimize(
        _quadratic_value_and_grad,
        x0,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "maxcor": m,
            "ftol": _SETULB_FTOL,
            "gtol": _SETULB_GTOL,
            "maxls": 20,
            "maxiter": _SETULB_DEFAULT_LIMIT,
            "maxfun": _SETULB_DEFAULT_LIMIT,
        },
    )
    state = _jax_state_with_scipy_workspace(
        _jax_setulb_initial_state(x0, bounds, m),
        workspace,
    )

    history = lbfgsb.lbfgsb_inverse_hessian_history(state)
    n_corrs = int(history.n_corrs)

    assert n_corrs == scipy_result.hess_inv.n_corrs
    np.testing.assert_array_equal(
        np.asarray(history.s)[:n_corrs],
        scipy_result.hess_inv.sk,
    )
    np.testing.assert_array_equal(
        np.asarray(history.y)[:n_corrs],
        scipy_result.hess_inv.yk,
    )


def test_jax_lbfgsb_inverse_hessian_history_is_jittable():
    x0 = _quadratic_x0()
    m = 5
    _, workspace = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        x0,
        None,
        m=m,
        ftol=_SETULB_FTOL,
        gtol=_SETULB_GTOL,
    )
    state = _jax_state_with_scipy_workspace(
        _jax_setulb_initial_state(x0, None, m),
        workspace,
    )

    history = jax.jit(lbfgsb.lbfgsb_inverse_hessian_history)(state)

    assert int(history.n_corrs) == min(int(workspace["isave"][30]), m)
    np.testing.assert_array_equal(
        np.asarray(history.s),
        workspace["wa"][: m * x0.size].reshape(m, x0.size),
    )
    np.testing.assert_array_equal(
        np.asarray(history.y),
        workspace["wa"][m * x0.size : 2 * m * x0.size].reshape(m, x0.size),
    )


def test_scipy_setulb_replay_fixture_matrix_records_internal_state(
    scipy_replay_case,
):
    fun, x0, bounds, m = scipy_replay_case
    trace, workspace = _run_scipy_setulb_trace(fun, x0, bounds, m=m)
    n = len(x0)
    wa_size = 2 * m * n + 5 * n + 11 * m * m + 8 * m

    assert tuple(trace[0]["task"]) == (3, 301)
    assert trace[-1]["task"][0] == 4
    assert all("wa" in event for event in trace)
    assert all("iwa" in event for event in trace)
    assert all("lsave" in event for event in trace)
    assert all(event["wa"].shape == (wa_size,) for event in trace)
    assert all(event["iwa"].shape == (3 * n,) for event in trace)
    assert all(event["lsave"].shape == (4,) for event in trace)
    assert all(event["isave"].shape == (44,) for event in trace)
    assert all(event["dsave"].shape == (29,) for event in trace)
    assert all(event["x"].dtype == np.float64 for event in trace)
    assert all(event["g"].dtype == np.float64 for event in trace)
    assert all(event["wa"].dtype == np.float64 for event in trace)
    assert all(event["iwa"].dtype == np.int32 for event in trace)
    assert all(event["lsave"].dtype == np.int32 for event in trace)
    assert all(event["isave"].dtype == np.int32 for event in trace)
    assert all(event["dsave"].dtype == np.float64 for event in trace)
    assert all(event["nfev"] == event["njev"] for event in trace)
    assert trace[-1]["nfev"] > 0
    assert trace[-1]["nit"] > 0
    assert trace[-1]["public_status"] == 0
    assert trace[-1]["public_message"] == (
        _lbfgsb_py.status_messages[int(trace[-1]["task"][0])]
        + ": "
        + _lbfgsb_py.task_messages[int(trace[-1]["task"][1])]
    )

    np.testing.assert_array_equal(trace[-1]["wa"], workspace["wa"])
    np.testing.assert_array_equal(trace[-1]["iwa"], workspace["iwa"])
    np.testing.assert_array_equal(trace[-1]["lsave"], workspace["lsave"])
    np.testing.assert_array_equal(trace[-1]["isave"], workspace["isave"])
    np.testing.assert_array_equal(trace[-1]["dsave"], workspace["dsave"])


def test_scipy_setulb_replay_fixture_matrix_is_bitwise_repeatable(
    scipy_replay_case,
):
    fun, x0, bounds, m = scipy_replay_case
    first_trace, _ = _run_scipy_setulb_trace(fun, x0, bounds, m=m)
    second_trace, _ = _run_scipy_setulb_trace(fun, x0, bounds, m=m)

    assert len(first_trace) == len(second_trace)
    for first_event, second_event in zip(first_trace, second_trace, strict=True):
        for field in (
            "task",
            "ln_task",
            "x",
            "f",
            "g",
            "wa",
            "iwa",
            "lsave",
            "isave",
            "dsave",
            "nfev",
            "njev",
            "nit",
        ):
            np.testing.assert_array_equal(first_event[field], second_event[field])
        if "public_status" in first_event:
            assert first_event["public_status"] == second_event["public_status"]
            assert first_event["public_message"] == second_event["public_message"]
        if "requested_f" in first_event:
            np.testing.assert_array_equal(
                first_event["requested_f"],
                second_event["requested_f"],
            )
            np.testing.assert_array_equal(
                first_event["requested_g"],
                second_event["requested_g"],
            )


def test_scipy_setulb_replay_helper_mutates_task_for_iteration_limit():
    trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        _quadratic_x0(),
        None,
        maxiter=1,
    )

    assert tuple(trace[-1]["task"]) == (5, 504)
    assert trace[-1]["nit"] == 1
    assert trace[-1]["public_status"] == 1
    assert trace[-1]["public_message"] == (
        "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
    )


def test_scipy_setulb_replay_helper_mutates_task_for_function_limit():
    trace, _ = _run_scipy_setulb_trace(
        _quadratic_value_and_grad,
        _quadratic_x0(),
        None,
        maxfun=0,
    )

    assert tuple(trace[-1]["task"]) == (5, 502)
    assert trace[-1]["nfev"] > 0
    assert trace[-1]["public_status"] == 1
    assert trace[-1]["public_message"] == (
        "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
    )
