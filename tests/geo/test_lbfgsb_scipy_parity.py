from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy
import scipy.linalg
from scipy import optimize
from scipy.optimize import _lbfgsb, _lbfgsb_py, minimize


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


def _quadratic_value_and_grad(x, *, scale=None, target=None):
    if scale is None:
        scale = np.array([1.0, 10.0, 100.0], dtype=np.float64)
    if target is None:
        target = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    shifted = np.asarray(x, dtype=np.float64) - target
    return np.float64(0.5 * np.dot(scale * shifted, shifted)), scale * shifted


def _ill_conditioned_diagonal_value_and_grad(x):
    return _quadratic_value_and_grad(
        x,
        scale=np.array([1.0e-3, 1.0, 1.0e3, 1.0e6], dtype=np.float64),
        target=np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64),
    )


def _rosenbrock_value_and_grad(x):
    return np.float64(optimize.rosen(x)), optimize.rosen_der(x).astype(np.float64)


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
    maxfun=15000,
    maxiter=15000,
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
        np.array([0.9, 0.2, -0.4], dtype=np.float64),
        [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
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
        np.array([0.9, 0.2, -0.4], dtype=np.float64),
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
        np.array([0.9, 0.2, -0.4], dtype=np.float64),
        None,
        maxfun=0,
    )

    assert tuple(trace[-1]["task"]) == (5, 502)
    assert trace[-1]["nfev"] > 0
    assert trace[-1]["public_status"] == 1
    assert trace[-1]["public_message"] == (
        "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
    )
