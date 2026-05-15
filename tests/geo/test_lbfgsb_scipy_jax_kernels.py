from __future__ import annotations

import numpy as np

import jax
from scipy.optimize import _lbfgsb_py

from simsopt.geo.optimizer_jax_private import _lbfgsb_scipy as lbfgsb


def _dcstep_reference(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    sgnd = dp * (dx / abs(dx))
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = max(abs(theta), abs(dx), abs(dp))
        gamma = scale * np.sqrt((theta / scale) ** 2 - (dx / scale) * (dp / scale))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (
            stp - stx
        )
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = max(abs(theta), abs(dx), abs(dp))
        gamma = scale * np.sqrt((theta / scale) ** 2 - (dx / scale) * (dp / scale))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = max(abs(theta), abs(dx), abs(dp))
        gamma = scale * np.sqrt(max(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if (r < 0.0) and (gamma != 0.0):
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            scale = max(abs(theta), abs(dy), abs(dp))
            gamma = scale * np.sqrt((theta / scale) ** 2 - (dy / scale) * (dp / scale))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpf = stp + r * (sty - stp)
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    return stx, fx, dx, sty, fy, dy, stpf, brackt


def _matupd_reference(ws, wy, sy, ss, d, r, itail, iupdat, col, head, rr, dr, stp, dtd):
    ws = np.asarray(ws, dtype=np.float64).copy()
    wy = np.asarray(wy, dtype=np.float64).copy()
    sy = np.asarray(sy, dtype=np.float64).copy()
    ss = np.asarray(ss, dtype=np.float64).copy()
    m = ws.shape[0]

    if iupdat <= m:
        col = iupdat
        itail = (head + iupdat - 1) % m
    else:
        itail = (itail + 1) % m
        head = (head + 1) % m

    ws[itail, :] = d
    wy[itail, :] = r
    theta = rr / dr

    if iupdat > m:
        for j in range(1, col):
            ss[:j, j - 1] = ss[1 : j + 1, j]
            sy[j - 1 : col - 1, j - 1] = sy[j:col, j]

    pointr = head
    for j in range(col - 1):
        sy[col - 1, j] = np.dot(d, wy[pointr])
        ss[j, col - 1] = np.dot(ws[pointr], d)
        pointr = (pointr + 1) % m

    ss[col - 1, col - 1] = dtd if stp == 1.0 else stp * stp * dtd
    sy[col - 1, col - 1] = dr
    return ws, wy, sy, ss, itail, col, head, theta


def _bmv_reference(sy, wt, col, v):
    sy = np.asarray(sy, dtype=np.float64)
    wt = np.asarray(wt, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.zeros_like(v, dtype=np.float64)
    if col == 0:
        return p, 0

    p[col] = v[col]
    for i in range(1, col):
        ssum = 0.0
        for k in range(i):
            ssum += sy[i, k] * v[k] / sy[k, k]
        p[col + i] = v[col + i] + ssum

    p[col : 2 * col] = np.linalg.solve(
        np.triu(wt[:col, :col]).T,
        p[col : 2 * col],
    )

    for i in range(col):
        p[i] = v[i] / np.sqrt(sy[i, i])

    p[col : 2 * col] = np.linalg.solve(
        np.triu(wt[:col, :col]),
        p[col : 2 * col],
    )

    for i in range(col):
        p[i] = -p[i] / np.sqrt(sy[i, i])

    for i in range(col):
        ssum = 0.0
        for k in range(i + 1, col):
            ssum += sy[k, i] * p[col + k] / sy[i, i]
        p[i] += ssum

    return p, 0


def _formt_reference(wt, sy, ss, col, theta):
    wt = np.asarray(wt, dtype=np.float64).copy()
    sy = np.asarray(sy, dtype=np.float64)
    ss = np.asarray(ss, dtype=np.float64)

    for j in range(col):
        wt[0, j] = theta * ss[0, j]

    for i in range(1, col):
        for j in range(i, col):
            ddum = 0.0
            for k in range(i):
                ddum += sy[i, k] * sy[j, k] / sy[k, k]
            wt[i, j] = ddum + theta * ss[i, j]

    t = np.triu(wt[:col, :col])
    t = t + np.triu(t, k=1).T
    wt[:col, :col] = np.linalg.cholesky(t).T
    return wt, 0


def _freev_reference(nfree, idx, idx2, iwhere, updatd, cnstnd, iteration):
    idx = np.asarray(idx, dtype=np.int32).copy()
    idx2 = np.asarray(idx2, dtype=np.int32).copy()
    iwhere = np.asarray(iwhere, dtype=np.int32)
    n = len(idx)
    nenter = 0
    ileave = n

    if iteration > 0 and cnstnd:
        for i in range(nfree):
            k = idx[i]
            if iwhere[k] > 0:
                ileave -= 1
                idx2[ileave] = k
        for i in range(nfree, n):
            k = idx[i]
            if iwhere[k] <= 0:
                idx2[nenter] = k
                nenter += 1

    next_nfree = 0
    iact = n
    next_idx = np.zeros_like(idx)
    for i in range(n):
        if iwhere[i] <= 0:
            next_idx[next_nfree] = i
            next_nfree += 1
        else:
            iact -= 1
            next_idx[iact] = i

    return next_nfree, next_idx, nenter, ileave, idx2, ileave < n or nenter > 0 or updatd


def _projected_gradient_norm_reference(l, u, nbd, x, g):
    sbgnrm = np.float64(0.0)
    for index, gi_value in enumerate(np.asarray(g, dtype=np.float64)):
        gi = np.float64(gi_value)
        if gi != gi:
            return gi
        if nbd[index] != lbfgsb.NBD_UNBOUNDED:
            if gi < 0.0:
                if nbd[index] >= lbfgsb.NBD_BOTH:
                    gi = np.maximum(x[index] - u[index], gi)
            elif nbd[index] <= lbfgsb.NBD_BOTH:
                gi = np.minimum(x[index] - l[index], gi)
        sbgnrm = np.maximum(sbgnrm, np.abs(gi))
    return sbgnrm


def _active_reference(l, u, nbd, x):
    x = np.asarray(x, dtype=np.float64).copy()
    iwhere = np.zeros_like(nbd, dtype=np.int32)
    prjctd = False
    cnstnd = False
    boxed = True

    for index in range(len(x)):
        if nbd[index] > lbfgsb.NBD_UNBOUNDED:
            if nbd[index] <= lbfgsb.NBD_BOTH and x[index] <= l[index]:
                if x[index] < l[index]:
                    prjctd = True
                    x[index] = l[index]
            elif nbd[index] >= lbfgsb.NBD_BOTH and x[index] >= u[index]:
                if x[index] > u[index]:
                    prjctd = True
                    x[index] = u[index]

    for index in range(len(x)):
        if nbd[index] != lbfgsb.NBD_BOTH:
            boxed = False
        if nbd[index] == lbfgsb.NBD_UNBOUNDED:
            iwhere[index] = -1
        else:
            cnstnd = True
            if nbd[index] == lbfgsb.NBD_BOTH and u[index] - l[index] <= 0.0:
                iwhere[index] = 3
            else:
                iwhere[index] = 0

    return x, iwhere, prjctd, cnstnd, boxed


def test_lbfgsb_status_tables_match_installed_scipy_wrapper():
    assert lbfgsb.STATUS_MESSAGES == _lbfgsb_py.status_messages
    assert lbfgsb.TASK_MESSAGES == _lbfgsb_py.task_messages
    task = np.array([lbfgsb.CONVERGENCE, lbfgsb.CONV_GRAD], dtype=np.int32)
    assert (
        lbfgsb.lbfgsb_task_message(task)
        == "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
    )


def test_lbfgsb_workspace_sizes_match_scipy_wrapper_formula():
    n = 7
    m = 13
    assert lbfgsb.lbfgsb_workspace_size(n, m) == 2 * m * n + 5 * n + 11 * m * m + 8 * m
    assert lbfgsb.lbfgsb_iwa_size(n) == 3 * n


def test_lbfgsb_initial_state_uses_fixed_scipy_workspace_shapes_and_dtypes():
    state = lbfgsb.lbfgsb_initial_state(
        np.array([0.25, -0.5, 0.75], dtype=np.float64),
        m=4,
        bounds=[(None, None), (0.0, None), (0.0, 1.0)],
        ftol=1e-12,
        gtol=1e-8,
        maxls=11,
    )

    assert state.m == 4
    assert state.maxls == 11
    np.testing.assert_array_equal(np.asarray(state.x), np.array([0.25, -0.5, 0.75]))
    np.testing.assert_array_equal(np.asarray(state.l), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(np.asarray(state.u), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(
        np.asarray(state.nbd),
        np.array(
            [lbfgsb.NBD_UNBOUNDED, lbfgsb.NBD_LOWER, lbfgsb.NBD_BOTH],
            dtype=np.int32,
        ),
    )
    assert state.workspace.wa.shape == (lbfgsb.lbfgsb_workspace_size(3, 4),)
    assert state.workspace.iwa.shape == (lbfgsb.lbfgsb_iwa_size(3),)
    assert state.workspace.task.shape == (2,)
    assert state.workspace.ln_task.shape == (2,)
    assert state.workspace.lsave.shape == (4,)
    assert state.workspace.isave.shape == (44,)
    assert state.workspace.dsave.shape == (29,)
    assert np.asarray(state.workspace.wa).dtype == np.float64
    assert np.asarray(state.workspace.iwa).dtype == np.int32
    assert np.asarray(state.factr).dtype == np.float64
    np.testing.assert_array_equal(np.asarray(state.factr), 1e-12 / np.finfo(float).eps)


def test_lbfgsb_public_status_matches_scipy_wrapper_mapping():
    assert lbfgsb.lbfgsb_public_status(lbfgsb.CONVERGENCE, 10, 2, 10, 2) == 0
    assert lbfgsb.lbfgsb_public_status(lbfgsb.STOP, 11, 1, 10, 20) == 1
    assert lbfgsb.lbfgsb_public_status(lbfgsb.STOP, 10, 20, 20, 20) == 1
    assert lbfgsb.lbfgsb_public_status(lbfgsb.ABNORMAL, 2, 1, 20, 20) == 2


def test_lbfgsb_projected_gradient_norm_matches_scipy_c_reference():
    l = np.array([-1.0, 0.0, -0.5, 0.0, -2.0], dtype=np.float64)
    u = np.array([1.0, 0.0, 0.5, 2.0, 4.0], dtype=np.float64)
    nbd = np.array(
        [
            lbfgsb.NBD_UNBOUNDED,
            lbfgsb.NBD_LOWER,
            lbfgsb.NBD_BOTH,
            lbfgsb.NBD_UPPER,
            lbfgsb.NBD_BOTH,
        ],
        dtype=np.int32,
    )
    x = np.array([3.0, -0.25, 0.7, 2.5, -3.0], dtype=np.float64)
    g = np.array([-4.0, 2.0, -8.0, -1.0, 9.0], dtype=np.float64)

    actual = np.asarray(lbfgsb.lbfgsb_projected_gradient_norm(l, u, nbd, x, g))
    expected = _projected_gradient_norm_reference(l, u, nbd, x, g)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


def test_lbfgsb_projected_gradient_norm_propagates_nan_gradient():
    l = np.zeros(3, dtype=np.float64)
    u = np.ones(3, dtype=np.float64)
    nbd = np.full(3, lbfgsb.NBD_BOTH, dtype=np.int32)
    x = np.full(3, 0.5, dtype=np.float64)
    g = np.array([0.0, np.nan, 1.0], dtype=np.float64)

    actual = np.asarray(lbfgsb.lbfgsb_projected_gradient_norm(l, u, nbd, x, g))

    assert np.isnan(actual)


def test_lbfgsb_active_matches_scipy_c_reference_for_all_nbd_classes():
    l = np.array([-1.0, 0.0, -0.5, 0.0, 2.0], dtype=np.float64)
    u = np.array([1.0, 0.0, 0.5, 2.0, 2.0], dtype=np.float64)
    nbd = np.array(
        [
            lbfgsb.NBD_UNBOUNDED,
            lbfgsb.NBD_LOWER,
            lbfgsb.NBD_BOTH,
            lbfgsb.NBD_UPPER,
            lbfgsb.NBD_BOTH,
        ],
        dtype=np.int32,
    )
    x = np.array([3.0, -0.25, 0.7, 2.5, 1.0], dtype=np.float64)

    expected_x, expected_iwhere, expected_prjctd, expected_cnstnd, expected_boxed = (
        _active_reference(l, u, nbd, x)
    )
    actual = lbfgsb.lbfgsb_active(l, u, nbd, x)

    np.testing.assert_array_equal(np.asarray(actual.x), expected_x)
    np.testing.assert_array_equal(np.asarray(actual.iwhere), expected_iwhere)
    assert bool(actual.prjctd) is expected_prjctd
    assert bool(actual.cnstnd) is expected_cnstnd
    assert bool(actual.boxed) is expected_boxed


def test_lbfgsb_bound_encoding_matches_scipy_nbd_semantics():
    low, upper, nbd = lbfgsb.lbfgsb_encode_bounds(
        [(None, None), (0.0, None), (0.0, 1.0), (None, 2.0)],
        4,
    )

    np.testing.assert_array_equal(low, np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(upper, np.array([0.0, 0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(
        nbd,
        np.array(
            [
                lbfgsb.NBD_UNBOUNDED,
                lbfgsb.NBD_LOWER,
                lbfgsb.NBD_BOTH,
                lbfgsb.NBD_UPPER,
            ],
            dtype=np.int32,
        ),
    )


def test_lbfgsb_bound_encoding_rejects_invalid_box_like_scipy():
    with np.testing.assert_raises_regex(
        ValueError,
        "LBFGSB - one of the lower bounds is greater than an upper bound.",
    ):
        lbfgsb.lbfgsb_encode_bounds([(2.0, 1.0)], 1)


def test_lbfgsb_dcstep_matches_c_reference_for_more_thuente_cases():
    cases = (
        (0.0, 1.0, -2.0, 0.0, 1.0, -2.0, 1.0, 2.0, -1.0, False, 0.0, 10.0),
        (0.0, 1.0, -2.0, 0.0, 1.0, -2.0, 1.0, 0.5, 0.5, False, 0.0, 10.0),
        (0.0, 1.0, -2.0, 2.0, 1.5, 1.0, 1.0, 0.5, -0.5, True, 0.0, 10.0),
        (0.0, 1.0, -2.0, 2.0, 1.5, 1.0, 1.0, 0.5, -3.0, True, 0.0, 10.0),
    )

    for case in cases:
        expected = _dcstep_reference(*case)
        actual = lbfgsb.lbfgsb_dcstep(*case)
        np.testing.assert_allclose(np.asarray(actual[:-1]), np.asarray(expected[:-1]))
        assert bool(actual.brackt) is expected[-1]


def test_lbfgsb_dcstep_is_jittable_with_fixed_scalar_carry():
    dcstep_jit = jax.jit(lbfgsb.lbfgsb_dcstep)
    actual = dcstep_jit(
        0.0,
        1.0,
        -2.0,
        0.0,
        1.0,
        -2.0,
        1.0,
        2.0,
        -1.0,
        False,
        0.0,
        10.0,
    )
    expected = _dcstep_reference(
        0.0,
        1.0,
        -2.0,
        0.0,
        1.0,
        -2.0,
        1.0,
        2.0,
        -1.0,
        False,
        0.0,
        10.0,
    )

    np.testing.assert_allclose(np.asarray(actual[:-1]), np.asarray(expected[:-1]))
    assert bool(actual.brackt) is expected[-1]


def test_lbfgsb_dcsrch_runs_reverse_communication_on_quadratic():
    def phi(alpha):
        return (alpha - 2.0) ** 2, 2.0 * (alpha - 2.0)

    isave = np.zeros(2, dtype=np.int32)
    dsave = np.zeros(13, dtype=np.float64)
    f0, g0 = phi(0.0)
    first = lbfgsb.lbfgsb_dcsrch(
        f0,
        g0,
        1.0,
        1e-4,
        0.1,
        1e-16,
        0.0,
        10.0,
        lbfgsb.START,
        lbfgsb.NO_MSG,
        isave,
        dsave,
    )

    assert int(first.task) == lbfgsb.FG
    assert int(first.task_msg) == lbfgsb.NO_MSG
    np.testing.assert_array_equal(np.asarray(first.stp), np.array(1.0))

    f1, g1 = phi(float(first.stp))
    second = lbfgsb.lbfgsb_dcsrch(
        f1,
        g1,
        first.stp,
        1e-4,
        0.1,
        1e-16,
        0.0,
        10.0,
        first.task,
        first.task_msg,
        first.isave,
        first.dsave,
    )

    assert int(second.task) == lbfgsb.FG
    assert int(second.task_msg) == lbfgsb.NO_MSG
    np.testing.assert_allclose(np.asarray(second.stp), np.array(2.0))

    f2, g2 = phi(float(second.stp))
    final = lbfgsb.lbfgsb_dcsrch(
        f2,
        g2,
        second.stp,
        1e-4,
        0.1,
        1e-16,
        0.0,
        10.0,
        second.task,
        second.task_msg,
        second.isave,
        second.dsave,
    )

    assert int(final.task) == lbfgsb.CONVERGENCE
    np.testing.assert_allclose(np.asarray(final.stp), np.array(2.0))


def test_lbfgsb_dcsrch_is_jittable_for_initial_reverse_communication_request():
    dcsrch_jit = jax.jit(lbfgsb.lbfgsb_dcsrch)
    result = dcsrch_jit(
        4.0,
        -4.0,
        1.0,
        1e-4,
        0.1,
        1e-16,
        0.0,
        10.0,
        lbfgsb.START,
        lbfgsb.NO_MSG,
        np.zeros(2, dtype=np.int32),
        np.zeros(13, dtype=np.float64),
    )

    assert int(result.task) == lbfgsb.FG
    assert int(result.task_msg) == lbfgsb.NO_MSG
    assert result.isave.shape == (2,)
    assert result.dsave.shape == (13,)


def test_lbfgsb_matupd_matches_c_reference_before_and_after_ring_wrap():
    m = 3
    n = 2
    ws = np.arange(m * n, dtype=np.float64).reshape(m, n) / 10.0
    wy = np.arange(m * n, 2 * m * n, dtype=np.float64).reshape(m, n) / 10.0
    sy = np.arange(m * m, dtype=np.float64).reshape(m, m)
    ss = np.arange(m * m, 2 * m * m, dtype=np.float64).reshape(m, m)
    d = np.array([1.5, -2.0], dtype=np.float64)
    r = np.array([3.0, 4.0], dtype=np.float64)

    for iupdat, itail, col, head in ((2, 0, 1, 0), (4, 2, 3, 0)):
        expected = _matupd_reference(
            ws,
            wy,
            sy,
            ss,
            d,
            r,
            itail,
            iupdat,
            col,
            head,
            rr=18.0,
            dr=6.0,
            stp=0.5,
            dtd=2.25,
        )
        actual = lbfgsb.lbfgsb_matupd(
            ws,
            wy,
            sy,
            ss,
            d,
            r,
            itail,
            iupdat,
            col,
            head,
            rr=18.0,
            dr=6.0,
            stp=0.5,
            dtd=2.25,
        )
        for actual_item, expected_item in zip(actual, expected, strict=True):
            np.testing.assert_allclose(np.asarray(actual_item), expected_item)


def test_lbfgsb_matupd_is_jittable_for_fixed_workspace_shapes():
    matupd_jit = jax.jit(lbfgsb.lbfgsb_matupd)
    m = 3
    n = 2
    actual = matupd_jit(
        np.zeros((m, n), dtype=np.float64),
        np.zeros((m, n), dtype=np.float64),
        np.zeros((m, m), dtype=np.float64),
        np.zeros((m, m), dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0, 4.0], dtype=np.float64),
        0,
        1,
        0,
        0,
        25.0,
        5.0,
        1.0,
        5.0,
    )

    np.testing.assert_array_equal(np.asarray(actual.ws[0]), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(actual.wy[0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(np.asarray(actual.theta), np.array(5.0))


def test_lbfgsb_bmv_matches_c_reference_for_full_and_partial_col():
    sy = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 9.0, 0.0],
            [2.0, 3.0, 16.0],
        ],
        dtype=np.float64,
    )
    wt = np.array(
        [
            [2.0, 0.25, -0.5],
            [0.0, 3.0, 0.75],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    v = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)

    for col in (1, 2, 3):
        expected_p, expected_info = _bmv_reference(sy, wt, col, v)
        actual = lbfgsb.lbfgsb_bmv(sy, wt, col, v)

        assert int(actual.info) == expected_info
        np.testing.assert_allclose(np.asarray(actual.p[: 2 * col]), expected_p[: 2 * col])


def test_lbfgsb_bmv_is_jittable_for_fixed_workspace_shapes():
    bmv_jit = jax.jit(lbfgsb.lbfgsb_bmv)
    sy = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 9.0, 0.0],
            [2.0, 3.0, 16.0],
        ],
        dtype=np.float64,
    )
    wt = np.array(
        [
            [2.0, 0.25, -0.5],
            [0.0, 3.0, 0.75],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    v = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)

    actual = bmv_jit(sy, wt, 2, v)
    expected_p, _ = _bmv_reference(sy, wt, 2, v)

    np.testing.assert_allclose(np.asarray(actual.p[:4]), expected_p[:4])
    assert int(actual.info) == 0


def test_lbfgsb_formt_matches_c_reference_for_partial_col():
    wt = np.zeros((3, 3), dtype=np.float64)
    sy = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 9.0, 0.0],
            [2.0, 3.0, 16.0],
        ],
        dtype=np.float64,
    )
    ss = np.array(
        [
            [2.0, -0.5, 1.25],
            [0.0, 3.0, -0.75],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )

    for col in (1, 2, 3):
        expected_wt, expected_info = _formt_reference(wt, sy, ss, col, theta=2.5)
        actual = lbfgsb.lbfgsb_formt(wt, sy, ss, col, theta=2.5)

        assert int(actual.info) == expected_info
        np.testing.assert_allclose(
            np.asarray(actual.wt[:col, :col]),
            expected_wt[:col, :col],
        )


def test_lbfgsb_formt_is_jittable_for_fixed_workspace_shapes():
    formt_jit = jax.jit(lbfgsb.lbfgsb_formt)
    wt = np.zeros((3, 3), dtype=np.float64)
    sy = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 9.0, 0.0],
            [2.0, 3.0, 16.0],
        ],
        dtype=np.float64,
    )
    ss = np.array(
        [
            [2.0, -0.5, 1.25],
            [0.0, 3.0, -0.75],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )

    actual = formt_jit(wt, sy, ss, 2, 2.5)
    expected_wt, _ = _formt_reference(wt, sy, ss, 2, theta=2.5)

    np.testing.assert_allclose(np.asarray(actual.wt[:2, :2]), expected_wt[:2, :2])
    assert int(actual.info) == 0


def test_lbfgsb_freev_matches_c_reference_for_entering_and_leaving_sets():
    idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    idx2 = np.full(5, -1, dtype=np.int32)
    iwhere = np.array([-1, 2, 0, 0, 3], dtype=np.int32)

    expected = _freev_reference(
        nfree=3,
        idx=idx,
        idx2=idx2,
        iwhere=iwhere,
        updatd=False,
        cnstnd=True,
        iteration=2,
    )
    actual = lbfgsb.lbfgsb_freev(
        3,
        idx,
        idx2,
        iwhere,
        False,
        True,
        2,
    )

    np.testing.assert_array_equal(np.asarray(actual.nfree), expected[0])
    np.testing.assert_array_equal(np.asarray(actual.idx), expected[1])
    np.testing.assert_array_equal(np.asarray(actual.nenter), expected[2])
    np.testing.assert_array_equal(np.asarray(actual.ileave), expected[3])
    np.testing.assert_array_equal(np.asarray(actual.idx2), expected[4])
    assert bool(actual.wrk) is expected[5]


def test_lbfgsb_freev_skips_enter_leave_count_on_initial_unconstrained_iteration():
    idx = np.array([0, 1, 2, 3], dtype=np.int32)
    idx2 = np.full(4, 7, dtype=np.int32)
    iwhere = np.array([-1, 0, 2, 3], dtype=np.int32)

    expected = _freev_reference(
        nfree=2,
        idx=idx,
        idx2=idx2,
        iwhere=iwhere,
        updatd=True,
        cnstnd=False,
        iteration=0,
    )
    actual = lbfgsb.lbfgsb_freev(
        2,
        idx,
        idx2,
        iwhere,
        True,
        False,
        0,
    )

    np.testing.assert_array_equal(np.asarray(actual.nfree), expected[0])
    np.testing.assert_array_equal(np.asarray(actual.idx), expected[1])
    np.testing.assert_array_equal(np.asarray(actual.nenter), expected[2])
    np.testing.assert_array_equal(np.asarray(actual.ileave), expected[3])
    np.testing.assert_array_equal(np.asarray(actual.idx2), expected[4])
    assert bool(actual.wrk) is expected[5]


def test_lbfgsb_freev_is_jittable_for_fixed_index_shapes():
    freev_jit = jax.jit(lbfgsb.lbfgsb_freev)
    idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    idx2 = np.full(5, -1, dtype=np.int32)
    iwhere = np.array([-1, 2, 0, 0, 3], dtype=np.int32)

    actual = freev_jit(3, idx, idx2, iwhere, False, True, 2)
    expected = _freev_reference(3, idx, idx2, iwhere, False, True, 2)

    np.testing.assert_array_equal(np.asarray(actual.nfree), expected[0])
    np.testing.assert_array_equal(np.asarray(actual.idx), expected[1])
    np.testing.assert_array_equal(np.asarray(actual.nenter), expected[2])
    np.testing.assert_array_equal(np.asarray(actual.ileave), expected[3])
    np.testing.assert_array_equal(np.asarray(actual.idx2), expected[4])
    assert bool(actual.wrk) is expected[5]
