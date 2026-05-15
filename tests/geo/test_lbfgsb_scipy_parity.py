from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
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


def _quadratic_value_and_grad(x):
    scale = np.array([1.0, 10.0, 100.0], dtype=np.float64)
    shifted = np.asarray(x, dtype=np.float64) - np.array([0.25, -0.5, 0.75])
    return np.float64(0.5 * np.dot(scale * shifted, shifted)), scale * shifted


def _run_scipy_setulb_trace(fun, x0, bounds, *, m=5, ftol=1e-12, gtol=1e-8, maxls=20):
    x = np.asarray(x0, dtype=np.float64).copy()
    n = len(x)
    low_bnd = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper_bnd = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    nbd = np.full(n, 2, dtype=np.int32)
    f = np.array(0.0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    workspace = _empty_setulb_workspace(n, m)
    trace = []
    factr = ftol / np.finfo(float).eps

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
            "isave": workspace["isave"].copy(),
            "dsave": workspace["dsave"].copy(),
        }
        trace.append(event)
        if workspace["task"][0] == 3:
            f, g = fun(x)
            event["requested_f"] = np.float64(f)
            event["requested_g"] = np.asarray(g, dtype=np.float64).copy()
        elif workspace["task"][0] != 1:
            break

    return trace, workspace


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
