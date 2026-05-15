"""SciPy L-BFGS-B 1.17.1-compatible optimizer-control helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np


START = 0
NO_MSG = 0
NEW_X = 1
RESTART = 2
FG = 3
CONVERGENCE = 4
STOP = 5
WARNING = 6
ERROR = 7
ABNORMAL = 8

FG_START = 301
FG_LNSRCH = 302

CONV_GRAD = 401
CONV_F = 402

STOP_CPU = 501
STOP_ITER = 502
STOP_GRAD = 503
STOP_ITERC = 504
STOP_CALLB = 505

WARN_ROUND = 601
WARN_STPMAX = 602
WARN_STPMIN = 603
WARN_XTOL = 604

ERROR_NOFEAS = 701
ERROR_FACTR = 702
ERROR_FTOL = 703
ERROR_GTOL = 704
ERROR_XTOL = 705
ERROR_STP1 = 706
ERROR_STP2 = 707
ERROR_STPMIN = 708
ERROR_STPMAX = 709
ERROR_INITG = 710
ERROR_M = 711
ERROR_N = 712
ERROR_NBD = 713

NBD_UNBOUNDED = 0
NBD_LOWER = 1
NBD_BOTH = 2
NBD_UPPER = 3

STATUS_MESSAGES = {
    START: "START",
    NEW_X: "NEW_X",
    RESTART: "RESTART",
    FG: "FG",
    CONVERGENCE: "CONVERGENCE",
    STOP: "STOP",
    WARNING: "WARNING",
    ERROR: "ERROR",
    ABNORMAL: "ABNORMAL",
}

TASK_MESSAGES = {
    0: "",
    FG_START: "",
    FG_LNSRCH: "",
    CONV_GRAD: "NORM OF PROJECTED GRADIENT <= PGTOL",
    CONV_F: "RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    STOP_CPU: "CPU EXCEEDING THE TIME LIMIT",
    STOP_ITER: "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    STOP_GRAD: "PROJECTED GRADIENT IS SUFFICIENTLY SMALL",
    STOP_ITERC: "TOTAL NO. OF ITERATIONS REACHED LIMIT",
    STOP_CALLB: "CALLBACK REQUESTED HALT",
    WARN_ROUND: "ROUNDING ERRORS PREVENT PROGRESS",
    WARN_STPMAX: "STP = STPMAX",
    WARN_STPMIN: "STP = STPMIN",
    WARN_XTOL: "XTOL TEST SATISFIED",
    ERROR_NOFEAS: "NO FEASIBLE SOLUTION",
    ERROR_FACTR: "FACTR < 0",
    ERROR_FTOL: "FTOL < 0",
    ERROR_GTOL: "GTOL < 0",
    ERROR_XTOL: "XTOL < 0",
    ERROR_STP1: "STP < STPMIN",
    ERROR_STP2: "STP > STPMAX",
    ERROR_STPMIN: "STPMIN < 0",
    ERROR_STPMAX: "STPMAX < STPMIN",
    ERROR_INITG: "INITIAL G >= 0",
    ERROR_M: "M <= 0",
    ERROR_N: "N <= 0",
    ERROR_NBD: "INVALID NBD",
}


class LbfgsbActiveResult(NamedTuple):
    x: jax.Array
    iwhere: jax.Array
    prjctd: jax.Array
    cnstnd: jax.Array
    boxed: jax.Array


class LbfgsbWorkspace(NamedTuple):
    wa: jax.Array
    iwa: jax.Array
    task: jax.Array
    ln_task: jax.Array
    lsave: jax.Array
    isave: jax.Array
    dsave: jax.Array


class LbfgsbState(NamedTuple):
    m: int
    x: jax.Array
    l: jax.Array
    u: jax.Array
    nbd: jax.Array
    f: jax.Array
    g: jax.Array
    factr: jax.Array
    pgtol: jax.Array
    maxls: int
    workspace: LbfgsbWorkspace
    n_iterations: jax.Array
    nfev: jax.Array
    njev: jax.Array


class LbfgsbDcstepResult(NamedTuple):
    stx: jax.Array
    fx: jax.Array
    dx: jax.Array
    sty: jax.Array
    fy: jax.Array
    dy: jax.Array
    stp: jax.Array
    brackt: jax.Array


class LbfgsbDcsrchResult(NamedTuple):
    stp: jax.Array
    task: jax.Array
    task_msg: jax.Array
    isave: jax.Array
    dsave: jax.Array


class LbfgsbMatupdResult(NamedTuple):
    ws: jax.Array
    wy: jax.Array
    sy: jax.Array
    ss: jax.Array
    itail: jax.Array
    col: jax.Array
    head: jax.Array
    theta: jax.Array


class LbfgsbBmvResult(NamedTuple):
    p: jax.Array
    info: jax.Array


class LbfgsbFormtResult(NamedTuple):
    wt: jax.Array
    info: jax.Array


class LbfgsbFormkResult(NamedTuple):
    wn: jax.Array
    wn1: jax.Array
    info: jax.Array


class LbfgsbHpsolbResult(NamedTuple):
    t: jax.Array
    iorder: jax.Array


class LbfgsbCmprlbResult(NamedTuple):
    r: jax.Array
    wa: jax.Array
    info: jax.Array


class LbfgsbSubsmResult(NamedTuple):
    x: jax.Array
    d: jax.Array
    xp: jax.Array
    iword: jax.Array
    wv: jax.Array
    info: jax.Array


class LbfgsbLnsrlbResult(NamedTuple):
    x: jax.Array
    fold: jax.Array
    gd: jax.Array
    gdold: jax.Array
    g: jax.Array
    r: jax.Array
    t: jax.Array
    stp: jax.Array
    dnorm: jax.Array
    dtd: jax.Array
    xstep: jax.Array
    stpmx: jax.Array
    ifun: jax.Array
    iback: jax.Array
    nfgv: jax.Array
    info: jax.Array
    task: jax.Array
    task_msg: jax.Array
    isave: jax.Array
    dsave: jax.Array
    temp_task: jax.Array
    temp_task_msg: jax.Array


class LbfgsbFreevResult(NamedTuple):
    nfree: jax.Array
    idx: jax.Array
    nenter: jax.Array
    ileave: jax.Array
    idx2: jax.Array
    wrk: jax.Array


def lbfgsb_workspace_size(n: int, m: int) -> int:
    return 2 * m * n + 5 * n + 11 * m * m + 8 * m


def lbfgsb_iwa_size(n: int) -> int:
    return 3 * n


def lbfgsb_empty_workspace(n: int, m: int) -> LbfgsbWorkspace:
    return LbfgsbWorkspace(
        wa=jnp.zeros((lbfgsb_workspace_size(n, m),), dtype=jnp.float64),
        iwa=jnp.zeros((lbfgsb_iwa_size(n),), dtype=jnp.int32),
        task=jnp.zeros((2,), dtype=jnp.int32),
        ln_task=jnp.zeros((2,), dtype=jnp.int32),
        lsave=jnp.zeros((4,), dtype=jnp.int32),
        isave=jnp.zeros((44,), dtype=jnp.int32),
        dsave=jnp.zeros((29,), dtype=jnp.float64),
    )


def lbfgsb_initial_state(
    x0,
    *,
    m: int,
    bounds=None,
    ftol: float = 2.2204460492503131e-09,
    gtol: float = 1e-5,
    maxls: int = 20,
) -> LbfgsbState:
    x = jnp.asarray(x0, dtype=jnp.float64).reshape((-1,))
    n = int(x.shape[0])
    low_bnd, upper_bnd, nbd = lbfgsb_encode_bounds(bounds, n)
    return LbfgsbState(
        m=int(m),
        x=x,
        l=jnp.asarray(low_bnd, dtype=jnp.float64),
        u=jnp.asarray(upper_bnd, dtype=jnp.float64),
        nbd=jnp.asarray(nbd, dtype=jnp.int32),
        f=jnp.asarray(0.0, dtype=jnp.float64),
        g=jnp.zeros_like(x, dtype=jnp.float64),
        factr=jnp.asarray(lbfgsb_factr_from_ftol(ftol), dtype=jnp.float64),
        pgtol=jnp.asarray(gtol, dtype=jnp.float64),
        maxls=int(maxls),
        workspace=lbfgsb_empty_workspace(n, int(m)),
        n_iterations=jnp.asarray(0, dtype=jnp.int32),
        nfev=jnp.asarray(0, dtype=jnp.int32),
        njev=jnp.asarray(0, dtype=jnp.int32),
    )


def lbfgsb_factr_from_ftol(ftol: float) -> np.float64:
    return np.float64(ftol) / np.finfo(float).eps


def lbfgsb_task_message(task: np.ndarray | jax.Array) -> str:
    task_host = np.asarray(jax.device_get(task), dtype=np.int32)
    return STATUS_MESSAGES[int(task_host[0])] + ": " + TASK_MESSAGES[int(task_host[1])]


def lbfgsb_public_status(
    task0: int, nfev: int, nit: int, maxfun: int, maxiter: int
) -> int:
    if int(task0) == CONVERGENCE:
        return 0
    if int(nfev) > int(maxfun) or int(nit) >= int(maxiter):
        return 1
    return 2


def lbfgsb_encode_bounds(bounds, n: int):
    low_bnd = np.zeros(n, dtype=np.float64)
    upper_bnd = np.zeros(n, dtype=np.float64)
    nbd = np.zeros(n, dtype=np.int32)
    if bounds is None:
        return low_bnd, upper_bnd, nbd
    if len(bounds) != n:
        raise ValueError("length of x0 != length of bounds")
    for i, bound in enumerate(bounds):
        lower, upper = bound
        has_lower = lower is not None and not np.isneginf(lower)
        has_upper = upper is not None and not np.isposinf(upper)
        if has_lower:
            low_bnd[i] = np.float64(lower)
        if has_upper:
            upper_bnd[i] = np.float64(upper)
        if has_lower and has_upper:
            nbd[i] = NBD_BOTH
        elif has_lower:
            nbd[i] = NBD_LOWER
        elif has_upper:
            nbd[i] = NBD_UPPER
    if np.any(low_bnd[nbd == NBD_BOTH] > upper_bnd[nbd == NBD_BOTH]):
        raise ValueError(
            "LBFGSB - one of the lower bounds is greater than an upper bound."
        )
    return low_bnd, upper_bnd, nbd


def lbfgsb_projected_gradient_norm(l, u, nbd, x, g):
    l = jnp.asarray(l, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    nbd = jnp.asarray(nbd, dtype=jnp.int32)
    x = jnp.asarray(x, dtype=jnp.float64)
    gi = jnp.asarray(g, dtype=jnp.float64)
    lower_limited = nbd <= NBD_BOTH
    upper_limited = nbd >= NBD_BOTH
    bounded = nbd != NBD_UNBOUNDED
    projected_negative = jnp.where(upper_limited, jnp.maximum(x - u, gi), gi)
    projected_positive = jnp.where(lower_limited, jnp.minimum(x - l, gi), gi)
    projected = jnp.where(gi < 0.0, projected_negative, projected_positive)
    projected = jnp.where(bounded, projected, gi)
    return jnp.max(jnp.abs(projected))


def lbfgsb_active(l, u, nbd, x):
    l = jnp.asarray(l, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    nbd = jnp.asarray(nbd, dtype=jnp.int32)
    x = jnp.asarray(x, dtype=jnp.float64)

    has_bound = nbd > NBD_UNBOUNDED
    has_lower = nbd <= NBD_BOTH
    has_upper = nbd >= NBD_BOTH
    project_lower = has_bound & has_lower & (x < l)
    project_upper = has_bound & has_upper & (x > u)
    projected_x = jnp.where(project_lower, l, x)
    projected_x = jnp.where(project_upper, u, projected_x)

    fixed = (nbd == NBD_BOTH) & ((u - l) <= 0.0)
    iwhere = jnp.where(nbd == NBD_UNBOUNDED, -1, jnp.where(fixed, 3, 0))
    return LbfgsbActiveResult(
        x=projected_x,
        iwhere=iwhere.astype(jnp.int32),
        prjctd=jnp.any(project_lower | project_upper),
        cnstnd=jnp.any(nbd != NBD_UNBOUNDED),
        boxed=jnp.all(nbd == NBD_BOTH),
    )


def lbfgsb_dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    stx = jnp.asarray(stx, dtype=jnp.float64)
    fx = jnp.asarray(fx, dtype=jnp.float64)
    dx = jnp.asarray(dx, dtype=jnp.float64)
    sty = jnp.asarray(sty, dtype=jnp.float64)
    fy = jnp.asarray(fy, dtype=jnp.float64)
    dy = jnp.asarray(dy, dtype=jnp.float64)
    stp = jnp.asarray(stp, dtype=jnp.float64)
    fp = jnp.asarray(fp, dtype=jnp.float64)
    dp = jnp.asarray(dp, dtype=jnp.float64)
    brackt = jnp.asarray(brackt, dtype=jnp.bool_)
    stpmin = jnp.asarray(stpmin, dtype=jnp.float64)
    stpmax = jnp.asarray(stpmax, dtype=jnp.float64)
    sgnd = dp * (dx / jnp.abs(dx))

    def higher_value(_):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = jnp.maximum(jnp.abs(theta), jnp.maximum(jnp.abs(dx), jnp.abs(dp)))
        gamma = scale * jnp.sqrt((theta / scale) ** 2 - (dx / scale) * (dp / scale))
        gamma = jnp.where(stp < stx, -gamma, gamma)
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        stpf = jnp.where(
            jnp.abs(stpc - stx) < jnp.abs(stpq - stx),
            stpc,
            stpc + (stpq - stpc) / 2.0,
        )
        return stpf, jnp.asarray(True, dtype=jnp.bool_)

    def lower_value_opposite_derivative(_):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = jnp.maximum(jnp.abs(theta), jnp.maximum(jnp.abs(dx), jnp.abs(dp)))
        gamma = scale * jnp.sqrt((theta / scale) ** 2 - (dx / scale) * (dp / scale))
        gamma = jnp.where(stp > stx, -gamma, gamma)
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        stpf = jnp.where(
            jnp.abs(stpc - stp) > jnp.abs(stpq - stp),
            stpc,
            stpq,
        )
        return stpf, jnp.asarray(True, dtype=jnp.bool_)

    def lower_value_decreasing_derivative(_):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = jnp.maximum(jnp.abs(theta), jnp.maximum(jnp.abs(dx), jnp.abs(dp)))
        gamma = scale * jnp.sqrt(
            jnp.maximum(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale))
        )
        gamma = jnp.where(stp > stx, -gamma, gamma)
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        stpc = jnp.where(
            (r < 0.0) & (gamma != 0.0),
            stp + r * (stx - stp),
            jnp.where(stp > stx, stpmax, stpmin),
        )
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        def bracketed(_):
            candidate = jnp.where(
                jnp.abs(stpc - stp) < jnp.abs(stpq - stp),
                stpc,
                stpq,
            )
            return jnp.where(
                stp > stx,
                jnp.minimum(stp + 0.66 * (sty - stp), candidate),
                jnp.maximum(stp + 0.66 * (sty - stp), candidate),
            )

        def unbracketed(_):
            candidate = jnp.where(
                jnp.abs(stpc - stp) > jnp.abs(stpq - stp),
                stpc,
                stpq,
            )
            return jnp.maximum(stpmin, jnp.minimum(stpmax, candidate))

        stpf = jax.lax.cond(brackt, bracketed, unbracketed, operand=None)
        return stpf, brackt

    def lower_value_not_decreasing_derivative(_):
        def bracketed(_):
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            scale = jnp.maximum(jnp.abs(theta), jnp.maximum(jnp.abs(dy), jnp.abs(dp)))
            gamma = scale * jnp.sqrt((theta / scale) ** 2 - (dy / scale) * (dp / scale))
            gamma = jnp.where(stp > sty, -gamma, gamma)
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            return stp + r * (sty - stp)

        def unbracketed(_):
            return jnp.where(stp > stx, stpmax, stpmin)

        return jax.lax.cond(brackt, bracketed, unbracketed, operand=None), brackt

    stpf, next_brackt = jax.lax.cond(
        fp > fx,
        higher_value,
        lambda _: jax.lax.cond(
            sgnd < 0.0,
            lower_value_opposite_derivative,
            lambda __: jax.lax.cond(
                jnp.abs(dp) < jnp.abs(dx),
                lower_value_decreasing_derivative,
                lower_value_not_decreasing_derivative,
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )

    sty_update = fp > fx
    opposite_derivative_update = (~sty_update) & (sgnd < 0.0)
    next_sty = jnp.where(
        sty_update, stp, jnp.where(opposite_derivative_update, stx, sty)
    )
    next_fy = jnp.where(sty_update, fp, jnp.where(opposite_derivative_update, fx, fy))
    next_dy = jnp.where(sty_update, dp, jnp.where(opposite_derivative_update, dx, dy))
    update_best = ~sty_update
    next_stx = jnp.where(update_best, stp, stx)
    next_fx = jnp.where(update_best, fp, fx)
    next_dx = jnp.where(update_best, dp, dx)

    return LbfgsbDcstepResult(
        stx=next_stx,
        fx=next_fx,
        dx=next_dx,
        sty=next_sty,
        fy=next_fy,
        dy=next_dy,
        stp=stpf,
        brackt=next_brackt,
    )


def _dcsrch_save_variables(
    *,
    stp,
    task,
    task_msg,
    brackt,
    stage,
    ginit,
    gtest,
    gx,
    gy,
    finit,
    fx,
    fy,
    stx,
    sty,
    stmin,
    stmax,
    width,
    width1,
):
    isave = jnp.asarray(
        (jnp.where(brackt, 1, 0), stage),
        dtype=jnp.int32,
    )
    dsave = jnp.asarray(
        (
            ginit,
            gtest,
            gx,
            gy,
            finit,
            fx,
            fy,
            stx,
            sty,
            stmin,
            stmax,
            width,
            width1,
        ),
        dtype=jnp.float64,
    )
    return LbfgsbDcsrchResult(
        stp=stp,
        task=jnp.asarray(task, dtype=jnp.int32),
        task_msg=jnp.asarray(task_msg, dtype=jnp.int32),
        isave=isave,
        dsave=dsave,
    )


def lbfgsb_dcsrch(
    f,
    g,
    stp,
    ftol,
    gtol,
    xtol,
    stpmin,
    stpmax,
    task,
    task_msg,
    isave,
    dsave,
):
    f = jnp.asarray(f, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64)
    stp = jnp.asarray(stp, dtype=jnp.float64)
    ftol = jnp.asarray(ftol, dtype=jnp.float64)
    gtol = jnp.asarray(gtol, dtype=jnp.float64)
    xtol = jnp.asarray(xtol, dtype=jnp.float64)
    stpmin = jnp.asarray(stpmin, dtype=jnp.float64)
    stpmax = jnp.asarray(stpmax, dtype=jnp.float64)
    task = jnp.asarray(task, dtype=jnp.int32)
    task_msg = jnp.asarray(task_msg, dtype=jnp.int32)
    isave = jnp.asarray(isave, dtype=jnp.int32)
    dsave = jnp.asarray(dsave, dtype=jnp.float64)

    def start_branch(_):
        error_msg = jnp.asarray(NO_MSG, dtype=jnp.int32)
        error_msg = jnp.where(stp < stpmin, ERROR_STP1, error_msg)
        error_msg = jnp.where(stp > stpmax, ERROR_STP2, error_msg)
        error_msg = jnp.where(g >= 0.0, ERROR_INITG, error_msg)
        error_msg = jnp.where(ftol < 0.0, ERROR_FTOL, error_msg)
        error_msg = jnp.where(gtol < 0.0, ERROR_GTOL, error_msg)
        error_msg = jnp.where(xtol < 0.0, ERROR_XTOL, error_msg)
        error_msg = jnp.where(stpmin < 0.0, ERROR_STPMIN, error_msg)
        error_msg = jnp.where(stpmax < stpmin, ERROR_STPMAX, error_msg)
        has_error = error_msg != NO_MSG

        finit = f
        ginit = g
        gtest = ftol * ginit
        width = stpmax - stpmin
        width1 = width / 0.5
        result = _dcsrch_save_variables(
            stp=stp,
            task=jnp.where(has_error, ERROR, FG),
            task_msg=error_msg,
            brackt=jnp.asarray(False, dtype=jnp.bool_),
            stage=jnp.asarray(1, dtype=jnp.int32),
            ginit=ginit,
            gtest=gtest,
            gx=ginit,
            gy=ginit,
            finit=finit,
            fx=finit,
            fy=finit,
            stx=jnp.asarray(0.0, dtype=jnp.float64),
            sty=jnp.asarray(0.0, dtype=jnp.float64),
            stmin=jnp.asarray(0.0, dtype=jnp.float64),
            stmax=stp + 4.0 * stp,
            width=width,
            width1=width1,
        )
        return result

    def continue_branch(_):
        brackt = isave[0] == 1
        stage = isave[1]
        ginit = dsave[0]
        gtest = dsave[1]
        gx = dsave[2]
        gy = dsave[3]
        finit = dsave[4]
        fx = dsave[5]
        fy = dsave[6]
        stx = dsave[7]
        sty = dsave[8]
        stmin = dsave[9]
        stmax = dsave[10]
        width = dsave[11]
        width1 = dsave[12]

        ftest = finit + stp * gtest
        stage = jnp.where((stage == 1) & (f <= ftest) & (g >= 0.0), 2, stage)

        next_task = task
        next_msg = task_msg
        round_warning = brackt & ((stp <= stmin) | (stp >= stmax))
        next_task = jnp.where(round_warning, WARNING, next_task)
        next_msg = jnp.where(round_warning, WARN_ROUND, next_msg)
        xtol_warning = brackt & ((stmax - stmin) <= xtol * stmax)
        next_task = jnp.where(xtol_warning, WARNING, next_task)
        next_msg = jnp.where(xtol_warning, WARN_XTOL, next_msg)
        stpmax_warning = (stp == stpmax) & (f <= ftest) & (g <= gtest)
        next_task = jnp.where(stpmax_warning, WARNING, next_task)
        next_msg = jnp.where(stpmax_warning, WARN_STPMAX, next_msg)
        stpmin_warning = (stp == stpmin) & ((f > ftest) | (g >= gtest))
        next_task = jnp.where(stpmin_warning, WARNING, next_task)
        next_msg = jnp.where(stpmin_warning, WARN_STPMIN, next_msg)
        converged = (f <= ftest) & (jnp.abs(g) <= gtol * (-ginit))
        next_task = jnp.where(converged, CONVERGENCE, next_task)
        terminate = (next_task == WARNING) | (next_task == CONVERGENCE)

        def terminate_branch(_):
            return _dcsrch_save_variables(
                stp=stp,
                task=next_task,
                task_msg=next_msg,
                brackt=brackt,
                stage=stage,
                ginit=ginit,
                gtest=gtest,
                gx=gx,
                gy=gy,
                finit=finit,
                fx=fx,
                fy=fy,
                stx=stx,
                sty=sty,
                stmin=stmin,
                stmax=stmax,
                width=width,
                width1=width1,
            )

        def step_branch(_):
            use_modified_function = (stage == 1) & (f <= fx) & (f > ftest)

            def modified_step(_):
                fm = f - stp * gtest
                fxm = fx - stx * gtest
                fym = fy - sty * gtest
                gm = g - gtest
                gxm = gx - gtest
                gym = gy - gtest
                step = lbfgsb_dcstep(
                    stx,
                    fxm,
                    gxm,
                    sty,
                    fym,
                    gym,
                    stp,
                    fm,
                    gm,
                    brackt,
                    stmin,
                    stmax,
                )
                return (
                    step.stx,
                    step.fx + step.stx * gtest,
                    step.dx + gtest,
                    step.sty,
                    step.fy + step.sty * gtest,
                    step.dy + gtest,
                    step.stp,
                    step.brackt,
                )

            def regular_step(_):
                step = lbfgsb_dcstep(
                    stx,
                    fx,
                    gx,
                    sty,
                    fy,
                    gy,
                    stp,
                    f,
                    g,
                    brackt,
                    stmin,
                    stmax,
                )
                return (
                    step.stx,
                    step.fx,
                    step.dx,
                    step.sty,
                    step.fy,
                    step.dy,
                    step.stp,
                    step.brackt,
                )

            stx2, fx2, gx2, sty2, fy2, gy2, stp2, brackt2 = jax.lax.cond(
                use_modified_function,
                modified_step,
                regular_step,
                operand=None,
            )
            bisect = brackt2 & (jnp.abs(sty2 - stx2) >= 0.66 * width1)
            stp2 = jnp.where(bisect, stx2 + 0.5 * (sty2 - stx2), stp2)
            width1_2 = jnp.where(brackt2, width, width1)
            width2 = jnp.where(brackt2, jnp.abs(sty2 - stx2), width)
            stmin2 = jnp.where(
                brackt2, jnp.minimum(stx2, sty2), stp2 + 1.1 * (stp2 - stx2)
            )
            stmax2 = jnp.where(
                brackt2, jnp.maximum(stx2, sty2), stp2 + 4.0 * (stp2 - stx2)
            )
            stp2 = jnp.maximum(stpmin, jnp.minimum(stpmax, stp2))
            no_progress = brackt2 & (
                (stp2 <= stmin2)
                | (stp2 >= stmax2)
                | ((stmax2 - stmin2) <= xtol * stmax2)
            )
            stp2 = jnp.where(no_progress, stx2, stp2)
            return _dcsrch_save_variables(
                stp=stp2,
                task=FG,
                task_msg=NO_MSG,
                brackt=brackt2,
                stage=stage,
                ginit=ginit,
                gtest=gtest,
                gx=gx2,
                gy=gy2,
                finit=finit,
                fx=fx2,
                fy=fy2,
                stx=stx2,
                sty=sty2,
                stmin=stmin2,
                stmax=stmax2,
                width=width2,
                width1=width1_2,
            )

        return jax.lax.cond(terminate, terminate_branch, step_branch, operand=None)

    return jax.lax.cond(task == START, start_branch, continue_branch, operand=None)


def lbfgsb_matupd(
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
    rr,
    dr,
    stp,
    dtd,
):
    ws = jnp.asarray(ws, dtype=jnp.float64)
    wy = jnp.asarray(wy, dtype=jnp.float64)
    sy = jnp.asarray(sy, dtype=jnp.float64)
    ss = jnp.asarray(ss, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    itail = jnp.asarray(itail, dtype=jnp.int32)
    iupdat = jnp.asarray(iupdat, dtype=jnp.int32)
    col = jnp.asarray(col, dtype=jnp.int32)
    head = jnp.asarray(head, dtype=jnp.int32)
    rr = jnp.asarray(rr, dtype=jnp.float64)
    dr = jnp.asarray(dr, dtype=jnp.float64)
    stp = jnp.asarray(stp, dtype=jnp.float64)
    dtd = jnp.asarray(dtd, dtype=jnp.float64)
    m = int(ws.shape[0])
    next_col = jnp.where(iupdat <= m, iupdat, col)
    next_itail = jnp.where(iupdat <= m, (head + iupdat - 1) % m, (itail + 1) % m)
    next_head = jnp.where(iupdat <= m, head, (head + 1) % m)
    ws = ws.at[next_itail, :].set(d)
    wy = wy.at[next_itail, :].set(r)
    theta = rr / dr

    rollover = iupdat > m

    for j in range(1, m):
        for offset in range(j):
            value = ss[offset + 1, j]
            ss = ss.at[offset, j - 1].set(
                jnp.where(rollover & (j < next_col), value, ss[offset, j - 1])
            )
        for offset in range(m - j):
            active = rollover & (j < next_col) & (offset < (next_col - j))
            value = sy[j + offset, j]
            sy = sy.at[j - 1 + offset, j - 1].set(
                jnp.where(active, value, sy[j - 1 + offset, j - 1])
            )

    pointr = next_head
    row = next_col - 1
    for j in range(m - 1):
        active = j < (next_col - 1)
        sy_value = jnp.dot(d, wy[pointr], precision=jax.lax.Precision.HIGHEST)
        ss_value = jnp.dot(ws[pointr], d, precision=jax.lax.Precision.HIGHEST)
        sy = sy.at[row, j].set(jnp.where(active, sy_value, sy[row, j]))
        ss = ss.at[j, row].set(jnp.where(active, ss_value, ss[j, row]))
        pointr = (pointr + 1) % m

    diagonal = jnp.where(stp == 1.0, dtd, stp * stp * dtd)
    ss = ss.at[row, row].set(diagonal)
    sy = sy.at[row, row].set(dr)

    return LbfgsbMatupdResult(
        ws=ws,
        wy=wy,
        sy=sy,
        ss=ss,
        itail=next_itail,
        col=next_col,
        head=next_head,
        theta=theta,
    )


def lbfgsb_bmv(sy, wt, col, v):
    sy = jnp.asarray(sy, dtype=jnp.float64)
    wt = jnp.asarray(wt, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    col = jnp.asarray(col, dtype=jnp.int32)
    m = int(sy.shape[0])
    p = jnp.zeros_like(v, dtype=jnp.float64)

    first_rhs = jnp.zeros((m,), dtype=jnp.float64)
    for i in range(m):
        first_rhs = first_rhs.at[i].set(jnp.where(i < col, v[col + i], 0.0))
    for i in range(1, m):
        active_i = i < col
        ssum = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(i):
            active_k = k < col
            ssum = ssum + jnp.where(
                active_i & active_k,
                sy[i, k] * v[k] / sy[k, k],
                0.0,
            )
        first_rhs = first_rhs.at[i].set(
            jnp.where(active_i, first_rhs[i] + ssum, first_rhs[i])
        )

    active = jnp.arange(m, dtype=jnp.int32) < col
    upper = jnp.triu(wt)
    identity = jnp.eye(m, dtype=jnp.float64)
    active_matrix = active[:, None] & active[None, :]
    solve_matrix = jnp.where(active_matrix, upper, identity)
    rhs = jnp.where(active, first_rhs, 0.0)
    p2 = jnp.linalg.solve(solve_matrix.T, rhs)

    diag_sy = jnp.diag(sy)
    p1 = jnp.where(active, v[:m] / jnp.sqrt(diag_sy), 0.0)
    for i in range(m):
        p = p.at[i].set(jnp.where(i < col, p1[i], p[i]))
        p = p.at[col + i].set(jnp.where(i < col, p2[i], p[col + i]))

    second_rhs = jnp.zeros((m,), dtype=jnp.float64)
    for i in range(m):
        second_rhs = second_rhs.at[i].set(jnp.where(i < col, p[col + i], 0.0))
    p2 = jnp.linalg.solve(solve_matrix, second_rhs)
    for i in range(m):
        p = p.at[col + i].set(jnp.where(i < col, p2[i], p[col + i]))
    p1 = jnp.where(active, -p[:m] / jnp.sqrt(diag_sy), 0.0)
    for i in range(m):
        p = p.at[i].set(jnp.where(i < col, p1[i], p[i]))

    for i in range(m):
        active_i = i < col
        ssum = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(i + 1, m):
            active_k = k < col
            ssum = ssum + jnp.where(
                active_i & active_k,
                sy[k, i] * p[col + k] / sy[i, i],
                0.0,
            )
        p = p.at[i].set(jnp.where(active_i, p[i] + ssum, p[i]))

    return LbfgsbBmvResult(p=p, info=jnp.asarray(0, dtype=jnp.int32))


def lbfgsb_formt(wt, sy, ss, col, theta):
    wt = jnp.asarray(wt, dtype=jnp.float64)
    sy = jnp.asarray(sy, dtype=jnp.float64)
    ss = jnp.asarray(ss, dtype=jnp.float64)
    col = jnp.asarray(col, dtype=jnp.int32)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    m = int(wt.shape[0])

    t_upper = jnp.zeros_like(wt, dtype=jnp.float64)
    for j in range(m):
        active_j = j < col
        t_upper = t_upper.at[0, j].set(
            jnp.where(active_j, theta * ss[0, j], t_upper[0, j])
        )
    for i in range(1, m):
        active_i = i < col
        for j in range(i, m):
            active_j = j < col
            ddum = jnp.asarray(0.0, dtype=jnp.float64)
            for k in range(i):
                active_k = k < col
                ddum = ddum + jnp.where(
                    active_i & active_j & active_k,
                    sy[i, k] * sy[j, k] / sy[k, k],
                    0.0,
                )
            t_upper = t_upper.at[i, j].set(
                jnp.where(active_i & active_j, ddum + theta * ss[i, j], t_upper[i, j])
            )

    active = jnp.arange(m, dtype=jnp.int32) < col
    active_matrix = active[:, None] & active[None, :]
    t_matrix = t_upper + jnp.triu(t_upper, k=1).T
    t_matrix = jnp.where(active_matrix, t_matrix, jnp.eye(m, dtype=jnp.float64))
    chol_upper = jnp.linalg.cholesky(t_matrix).T
    wt_next = jnp.where(jnp.triu(active_matrix), chol_upper, wt)
    finite = jnp.all(jnp.isfinite(chol_upper))
    return LbfgsbFormtResult(
        wt=wt_next,
        info=jnp.where(
            finite, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(-3, dtype=jnp.int32)
        ),
    )


def lbfgsb_formk(
    nsub,
    ind,
    nenter,
    ileave,
    indx2,
    iupdat,
    updatd,
    wn,
    wn1,
    ws,
    wy,
    sy,
    theta,
    col,
    head,
):
    nsub = jnp.asarray(nsub, dtype=jnp.int32)
    ind = jnp.asarray(ind, dtype=jnp.int32)
    nenter = jnp.asarray(nenter, dtype=jnp.int32)
    ileave = jnp.asarray(ileave, dtype=jnp.int32)
    indx2 = jnp.asarray(indx2, dtype=jnp.int32)
    iupdat = jnp.asarray(iupdat, dtype=jnp.int32)
    updatd = jnp.asarray(updatd, dtype=jnp.bool_)
    wn = jnp.asarray(wn, dtype=jnp.float64)
    wn1 = jnp.asarray(wn1, dtype=jnp.float64)
    ws = jnp.asarray(ws, dtype=jnp.float64)
    wy = jnp.asarray(wy, dtype=jnp.float64)
    sy = jnp.asarray(sy, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    col = jnp.asarray(col, dtype=jnp.int32)
    head = jnp.asarray(head, dtype=jnp.int32)
    m = int(ws.shape[0])
    n = int(ws.shape[1])

    shift = updatd & (iupdat > m)
    for jy in range(m - 1):
        js = m + jy
        for offset in range(m - (jy + 1)):
            wn1 = wn1.at[jy + offset, jy].set(
                jnp.where(shift, wn1[jy + 1 + offset, jy + 1], wn1[jy + offset, jy])
            )
            wn1 = wn1.at[js + offset, js].set(
                jnp.where(shift, wn1[js + 1 + offset, js + 1], wn1[js + offset, js])
            )
        for offset in range(m - 1):
            wn1 = wn1.at[m + offset, jy].set(
                jnp.where(shift, wn1[m + 1 + offset, jy + 1], wn1[m + offset, jy])
            )

    iy_new = col - 1
    is_new = m + col - 1
    ipntr = (head + col - 1) % m
    jpntr = head
    row_new = jnp.maximum(iy_new, 0)
    block_row_new = jnp.maximum(is_new, 0)
    for jy in range(m):
        active_j = updatd & (jy < col)
        temp1 = jnp.asarray(0.0, dtype=jnp.float64)
        temp2 = jnp.asarray(0.0, dtype=jnp.float64)
        temp3 = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(n):
            k1 = ind[k]
            free = k < nsub
            active = k >= nsub
            temp1 = temp1 + jnp.where(
                active_j & free,
                wy[ipntr, k1] * wy[jpntr, k1],
                0.0,
            )
            temp2 = temp2 + jnp.where(
                active_j & active,
                ws[ipntr, k1] * ws[jpntr, k1],
                0.0,
            )
            temp3 = temp3 + jnp.where(
                active_j & active,
                ws[ipntr, k1] * wy[jpntr, k1],
                0.0,
            )
        wn1 = wn1.at[row_new, jy].set(jnp.where(active_j, temp1, wn1[row_new, jy]))
        wn1 = wn1.at[block_row_new, m + jy].set(
            jnp.where(active_j, temp2, wn1[block_row_new, m + jy])
        )
        wn1 = wn1.at[block_row_new, jy].set(
            jnp.where(active_j, temp3, wn1[block_row_new, jy])
        )
        jpntr = (jpntr + 1) % m

    jy_new = jnp.maximum(col - 1, 0)
    jpntr = (head + col - 1) % m
    ipntr = head
    for i in range(m):
        active_i = updatd & (i < col)
        is_i = m + i
        temp3 = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(n):
            k1 = ind[k]
            temp3 = temp3 + jnp.where(
                active_i & (k < nsub),
                ws[ipntr, k1] * wy[jpntr, k1],
                0.0,
            )
        ipntr = (ipntr + 1) % m
        wn1 = wn1.at[is_i, jy_new].set(jnp.where(active_i, temp3, wn1[is_i, jy_new]))

    upcl = jnp.where(updatd, col - 1, col)
    ipntr = head
    for iy in range(m):
        active_iy = iy < upcl
        is_i = m + iy
        jpntr = head
        for jy in range(m):
            pair_active = active_iy & (jy <= iy)
            js = m + jy
            temp1 = jnp.asarray(0.0, dtype=jnp.float64)
            temp2 = jnp.asarray(0.0, dtype=jnp.float64)
            temp3 = jnp.asarray(0.0, dtype=jnp.float64)
            temp4 = jnp.asarray(0.0, dtype=jnp.float64)
            for k in range(n):
                k1 = indx2[k]
                entering = pair_active & (k < nenter)
                leaving = pair_active & (k >= ileave)
                temp1 = temp1 + jnp.where(
                    entering,
                    wy[ipntr, k1] * wy[jpntr, k1],
                    0.0,
                )
                temp2 = temp2 + jnp.where(
                    entering,
                    ws[ipntr, k1] * ws[jpntr, k1],
                    0.0,
                )
                temp3 = temp3 + jnp.where(
                    leaving,
                    wy[ipntr, k1] * wy[jpntr, k1],
                    0.0,
                )
                temp4 = temp4 + jnp.where(
                    leaving,
                    ws[ipntr, k1] * ws[jpntr, k1],
                    0.0,
                )
            wn1 = wn1.at[iy, jy].set(
                jnp.where(pair_active, wn1[iy, jy] + temp1 - temp3, wn1[iy, jy])
            )
            wn1 = wn1.at[is_i, js].set(
                jnp.where(pair_active, wn1[is_i, js] - temp2 + temp4, wn1[is_i, js])
            )
            jpntr = (jpntr + 1) % m
        ipntr = (ipntr + 1) % m

    ipntr = head
    for i in range(m):
        is_i = m + i
        active_is = i < upcl
        jpntr = head
        for jy in range(m):
            pair_active = active_is & (jy < upcl)
            temp1 = jnp.asarray(0.0, dtype=jnp.float64)
            temp3 = jnp.asarray(0.0, dtype=jnp.float64)
            for k in range(n):
                k1 = indx2[k]
                entering = pair_active & (k < nenter)
                leaving = pair_active & (k >= ileave)
                temp1 = temp1 + jnp.where(
                    entering,
                    ws[ipntr, k1] * wy[jpntr, k1],
                    0.0,
                )
                temp3 = temp3 + jnp.where(
                    leaving,
                    ws[ipntr, k1] * wy[jpntr, k1],
                    0.0,
                )
            delta = jnp.where(i <= jy, temp1 - temp3, -temp1 + temp3)
            wn1 = wn1.at[is_i, jy].set(
                jnp.where(pair_active, wn1[is_i, jy] + delta, wn1[is_i, jy])
            )
            jpntr = (jpntr + 1) % m
        ipntr = (ipntr + 1) % m

    for iy in range(m):
        active_iy = iy < col
        is_col = col + iy
        is1 = m + iy
        for jy in range(m):
            active_j = active_iy & (jy <= iy) & (jy < col)
            js_col = col + jy
            js1 = m + jy
            wn = wn.at[jy, iy].set(jnp.where(active_j, wn1[iy, jy] / theta, wn[jy, iy]))
            wn = wn.at[js_col, is_col].set(
                jnp.where(active_j, wn1[is1, js1] * theta, wn[js_col, is_col])
            )
        for jy in range(m):
            active_j = active_iy & (jy < col)
            value = jnp.where(jy < iy, -wn1[is1, jy], wn1[is1, jy])
            wn = wn.at[jy, is_col].set(jnp.where(active_j, value, wn[jy, is_col]))
        wn = wn.at[iy, iy].set(
            jnp.where(active_iy, wn[iy, iy] + sy[iy, iy], wn[iy, iy])
        )

    active = jnp.arange(m, dtype=jnp.int32) < col
    active_matrix = active[:, None] & active[None, :]
    first_upper = jnp.triu(wn[:m, :m])
    first_matrix = first_upper + jnp.triu(first_upper, k=1).T
    first_matrix = jnp.where(active_matrix, first_matrix, jnp.eye(m, dtype=jnp.float64))
    first_chol = jnp.linalg.cholesky(first_matrix).T
    first_finite = jnp.all(jnp.isfinite(jnp.where(active_matrix, first_chol, 0.0)))
    wn_first = jnp.where(jnp.triu(active_matrix) & first_finite, first_chol, wn[:m, :m])
    wn = wn.at[:m, :m].set(wn_first)

    rhs = jnp.zeros((m, m), dtype=jnp.float64)
    for j in range(m):
        source_col = col + j
        active_j = j < col
        for i in range(m):
            active_i = i < col
            rhs = rhs.at[i, j].set(
                jnp.where(active_i & active_j, wn[i, source_col], rhs[i, j])
            )
    solved = jsp_linalg.solve_triangular(
        first_chol,
        rhs,
        trans=1,
        lower=False,
        unit_diagonal=False,
    )
    for j in range(m):
        target_col = col + j
        active_j = j < col
        for i in range(m):
            active_i = i < col
            wn = wn.at[i, target_col].set(
                jnp.where(
                    active_i & active_j & first_finite,
                    solved[i, j],
                    wn[i, target_col],
                )
            )

    for i in range(m):
        is_col = col + i
        active_i = i < col
        for j in range(m):
            js_col = col + j
            active_j = active_i & (j >= i) & (j < col)
            dot = jnp.asarray(0.0, dtype=jnp.float64)
            for k in range(m):
                dot = dot + jnp.where(
                    (k < col) & first_finite,
                    wn[k, is_col] * wn[k, js_col],
                    0.0,
                )
            wn = wn.at[is_col, js_col].set(
                jnp.where(
                    active_j & first_finite,
                    wn[is_col, js_col] + dot,
                    wn[is_col, js_col],
                )
            )

    second_upper = jnp.zeros((m, m), dtype=jnp.float64)
    for i in range(m):
        source_row = col + i
        active_i = i < col
        for j in range(m):
            source_col = col + j
            active_j = active_i & (j >= i) & (j < col)
            second_upper = second_upper.at[i, j].set(
                jnp.where(active_j, wn[source_row, source_col], second_upper[i, j])
            )
    second_matrix = second_upper + jnp.triu(second_upper, k=1).T
    second_matrix = jnp.where(
        active_matrix, second_matrix, jnp.eye(m, dtype=jnp.float64)
    )
    second_chol = jnp.linalg.cholesky(second_matrix).T
    second_finite = jnp.all(jnp.isfinite(jnp.where(active_matrix, second_chol, 0.0)))
    for i in range(m):
        target_row = col + i
        active_i = i < col
        for j in range(m):
            target_col = col + j
            active_j = active_i & (j >= i) & (j < col)
            wn = wn.at[target_row, target_col].set(
                jnp.where(
                    active_j & first_finite & second_finite,
                    second_chol[i, j],
                    wn[target_row, target_col],
                )
            )

    return LbfgsbFormkResult(
        wn=wn,
        wn1=wn1,
        info=jnp.where(
            first_finite,
            jnp.where(
                second_finite,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(-2, dtype=jnp.int32),
            ),
            jnp.asarray(-1, dtype=jnp.int32),
        ),
    )


def lbfgsb_hpsolb(last, t, iorder, iheap):
    last = jnp.asarray(last, dtype=jnp.int32)
    t = jnp.asarray(t, dtype=jnp.float64)
    iorder = jnp.asarray(iorder, dtype=jnp.int32)
    iheap = jnp.asarray(iheap, dtype=jnp.int32)
    n_slots = int(t.shape[0])

    build_heap = iheap == 0

    def parent_slot_of(slot):
        return jnp.maximum((slot + 1) // 2 - 1, 0)

    def build_body(k, carry):
        t, iorder = carry
        active = build_heap & (k <= last + 1)
        ddum = t[k - 1]
        indxin = iorder[k - 1]
        slot = jnp.asarray(k - 1, dtype=jnp.int32)

        def bubble_cond(bubble_carry):
            slot, t, _ = bubble_carry
            parent_slot = parent_slot_of(slot)
            return active & (slot > 0) & (ddum < t[parent_slot])

        def bubble_body(bubble_carry):
            slot, t, iorder = bubble_carry
            parent_slot = parent_slot_of(slot)
            t = t.at[slot].set(t[parent_slot])
            iorder = iorder.at[slot].set(iorder[parent_slot])
            return parent_slot, t, iorder

        slot, t, iorder = jax.lax.while_loop(
            bubble_cond, bubble_body, (slot, t, iorder)
        )
        t = t.at[slot].set(jnp.where(active, ddum, t[slot]))
        iorder = iorder.at[slot].set(jnp.where(active, indxin, iorder[slot]))
        return t, iorder

    t, iorder = jax.lax.fori_loop(2, n_slots + 1, build_body, (t, iorder))

    extract = last > 0
    out = t[0]
    indxout = iorder[0]
    last_slot = jnp.minimum(last, n_slots - 1)
    ddum = t[last_slot]
    indxin = iorder[last_slot]
    slot = jnp.asarray(0, dtype=jnp.int32)

    def child_slot_of(slot, t):
        left_child = 2 * (slot + 1) - 1
        right_child = left_child + 1
        has_child = extract & (left_child < last)
        right_child = jnp.minimum(right_child, n_slots - 1)
        left_child = jnp.minimum(left_child, n_slots - 1)
        choose_right = has_child & (t[right_child] < t[left_child])
        return has_child, jnp.where(choose_right, right_child, left_child)

    def extract_cond(extract_carry):
        slot, t, _ = extract_carry
        has_child, child_slot = child_slot_of(slot, t)
        return has_child & (t[child_slot] < ddum)

    def extract_body(extract_carry):
        slot, t, iorder = extract_carry
        _, child_slot = child_slot_of(slot, t)
        t = t.at[slot].set(t[child_slot])
        iorder = iorder.at[slot].set(iorder[child_slot])
        return child_slot, t, iorder

    slot, t, iorder = jax.lax.while_loop(extract_cond, extract_body, (slot, t, iorder))

    t = t.at[slot].set(jnp.where(extract, ddum, t[slot]))
    iorder = iorder.at[slot].set(jnp.where(extract, indxin, iorder[slot]))
    t = t.at[last_slot].set(jnp.where(extract, out, t[last_slot]))
    iorder = iorder.at[last_slot].set(jnp.where(extract, indxout, iorder[last_slot]))
    return LbfgsbHpsolbResult(t=t, iorder=iorder)


def lbfgsb_cmprlb(
    x,
    g,
    ws,
    wy,
    sy,
    wt,
    z,
    r,
    wa,
    index,
    theta,
    col,
    head,
    nfree,
    cnstnd,
):
    x = jnp.asarray(x, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64)
    ws = jnp.asarray(ws, dtype=jnp.float64)
    wy = jnp.asarray(wy, dtype=jnp.float64)
    sy = jnp.asarray(sy, dtype=jnp.float64)
    wt = jnp.asarray(wt, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    wa = jnp.asarray(wa, dtype=jnp.float64)
    index = jnp.asarray(index, dtype=jnp.int32)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    col = jnp.asarray(col, dtype=jnp.int32)
    head = jnp.asarray(head, dtype=jnp.int32)
    nfree = jnp.asarray(nfree, dtype=jnp.int32)
    cnstnd = jnp.asarray(cnstnd, dtype=jnp.bool_)
    m = int(ws.shape[0])
    n = int(x.shape[0])
    col2 = 2 * m

    def unconstrained_branch(_):
        return LbfgsbCmprlbResult(
            r=-g,
            wa=wa,
            info=jnp.asarray(0, dtype=jnp.int32),
        )

    def constrained_branch(_):
        next_r = r
        for i in range(n):
            active_i = i < nfree
            k = index[i]
            value = -theta * (z[k] - x[k]) - g[k]
            next_r = next_r.at[i].set(jnp.where(active_i, value, next_r[i]))

        bmv_result = lbfgsb_bmv(sy, wt, col, wa[2 * m : 4 * m])
        next_wa = wa.at[:col2].set(bmv_result.p)
        pointr = head
        for j in range(m):
            active_j = j < col
            a1 = next_wa[j]
            a2 = theta * next_wa[col + j]
            for i in range(n):
                active_i = active_j & (i < nfree)
                k = index[i]
                value = next_r[i] + wy[pointr, k] * a1 + ws[pointr, k] * a2
                next_r = next_r.at[i].set(jnp.where(active_i, value, next_r[i]))
            pointr = (pointr + 1) % m

        return LbfgsbCmprlbResult(
            r=next_r,
            wa=next_wa,
            info=jnp.where(
                bmv_result.info != 0,
                jnp.asarray(-8, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    return jax.lax.cond(
        (~cnstnd) & (col > 0), unconstrained_branch, constrained_branch, None
    )


def lbfgsb_subsm(
    nsub,
    ind,
    l,
    u,
    nbd,
    x,
    d,
    xp,
    ws,
    wy,
    theta,
    xx,
    gg,
    col,
    head,
    wv,
    wn,
):
    nsub = jnp.asarray(nsub, dtype=jnp.int32)
    ind = jnp.asarray(ind, dtype=jnp.int32)
    l = jnp.asarray(l, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    nbd = jnp.asarray(nbd, dtype=jnp.int32)
    x = jnp.asarray(x, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    xp = jnp.asarray(xp, dtype=jnp.float64)
    ws = jnp.asarray(ws, dtype=jnp.float64)
    wy = jnp.asarray(wy, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    xx = jnp.asarray(xx, dtype=jnp.float64)
    gg = jnp.asarray(gg, dtype=jnp.float64)
    col = jnp.asarray(col, dtype=jnp.int32)
    head = jnp.asarray(head, dtype=jnp.int32)
    wv = jnp.asarray(wv, dtype=jnp.float64)
    wn = jnp.asarray(wn, dtype=jnp.float64)
    m = int(ws.shape[0])
    n = int(ws.shape[1])
    m2 = 2 * m
    original_x = x
    original_d = d
    original_xp = xp

    pointr = head
    for i in range(m):
        active_col = i < col
        temp1 = jnp.asarray(0.0, dtype=jnp.float64)
        temp2 = jnp.asarray(0.0, dtype=jnp.float64)
        for j in range(n):
            active_sub = j < nsub
            k = ind[j]
            temp1 = temp1 + jnp.where(
                active_col & active_sub,
                wy[pointr, k] * d[j],
                0.0,
            )
            temp2 = temp2 + jnp.where(
                active_col & active_sub,
                ws[pointr, k] * d[j],
                0.0,
            )
        wv = wv.at[i].set(jnp.where(active_col, temp1, wv[i]))
        wv = wv.at[col + i].set(jnp.where(active_col, theta * temp2, wv[col + i]))
        pointr = (pointr + 1) % m

    active = jnp.arange(m2, dtype=jnp.int32) < (2 * col)
    active_matrix = active[:, None] & active[None, :]
    factor = jnp.where(active_matrix, jnp.triu(wn), jnp.eye(m2, dtype=jnp.float64))
    rhs = jnp.where(active, wv, 0.0)
    solved = jsp_linalg.solve_triangular(
        factor,
        rhs,
        trans=1,
        lower=False,
        unit_diagonal=False,
    )
    first_finite = jnp.all(jnp.isfinite(jnp.where(active, solved, 0.0)))
    wv_after_first = jnp.where(active & first_finite, solved, wv)

    for i in range(m):
        wv_after_first = wv_after_first.at[i].set(
            jnp.where(
                (i < col) & first_finite,
                -wv_after_first[i],
                wv_after_first[i],
            )
        )

    rhs = jnp.where(active, wv_after_first, 0.0)
    solved = jsp_linalg.solve_triangular(
        factor,
        rhs,
        trans=0,
        lower=False,
        unit_diagonal=False,
    )
    second_finite = jnp.all(jnp.isfinite(jnp.where(active, solved, 0.0)))
    valid = first_finite & second_finite
    wv = jnp.where(active & valid, solved, wv_after_first)

    pointr = head
    for jy in range(m):
        active_col = valid & (jy < col)
        js = col + jy
        for i in range(n):
            active_sub = active_col & (i < nsub)
            k = ind[i]
            value = d[i] + wy[pointr, k] * wv[jy] / theta + ws[pointr, k] * wv[js]
            d = d.at[i].set(jnp.where(active_sub, value, d[i]))
        pointr = (pointr + 1) % m

    for i in range(n):
        d = d.at[i].set(jnp.where(valid & (i < nsub), d[i] / theta, d[i]))

    iword = jnp.asarray(0, dtype=jnp.int32)
    xp = jnp.where(valid, x, xp)
    projected_x = x
    for i in range(n):
        active_sub = valid & (i < nsub)
        k = ind[i]
        dk = d[i]
        xk = x[k]
        lower_only = nbd[k] == NBD_LOWER
        both = nbd[k] == NBD_BOTH
        upper_only = nbd[k] == NBD_UPPER
        lower_candidate = jnp.maximum(l[k], xk + dk)
        both_candidate = jnp.minimum(u[k], jnp.maximum(l[k], xk + dk))
        upper_candidate = jnp.minimum(u[k], xk + dk)
        candidate = jnp.where(
            lower_only,
            lower_candidate,
            jnp.where(
                both, both_candidate, jnp.where(upper_only, upper_candidate, xk + dk)
            ),
        )
        hit = (lower_only & (candidate == l[k])) | (
            both & ((candidate == l[k]) | (candidate == u[k]))
        )
        hit = hit | (upper_only & (candidate == u[k]))
        projected_x = projected_x.at[k].set(
            jnp.where(active_sub, candidate, projected_x[k])
        )
        iword = jnp.where(active_sub & hit, jnp.asarray(1, dtype=jnp.int32), iword)

    dd_p = jnp.sum((projected_x - xx) * gg)
    safeguard = valid & (iword == 1) & (dd_p > 0.0)
    safeguarded_x = xp
    safeguarded_d = d
    alpha = jnp.asarray(1.0, dtype=jnp.float64)
    temp1 = jnp.asarray(1.0, dtype=jnp.float64)
    ibd = jnp.asarray(0, dtype=jnp.int32)
    for i in range(n):
        active_sub = safeguard & (i < nsub)
        k = ind[i]
        dk = safeguarded_d[i]
        next_temp1 = temp1
        lower_step = l[k] - safeguarded_x[k]
        upper_step = u[k] - safeguarded_x[k]
        lower_limited = (dk < 0.0) & (nbd[k] <= NBD_BOTH) & (nbd[k] != NBD_UNBOUNDED)
        upper_limited = (dk > 0.0) & (nbd[k] >= NBD_BOTH)
        next_temp1 = jnp.where(
            active_sub & lower_limited & (lower_step >= 0.0),
            0.0,
            jnp.where(
                active_sub & lower_limited & (dk * alpha < lower_step),
                lower_step / dk,
                next_temp1,
            ),
        )
        next_temp1 = jnp.where(
            active_sub & upper_limited & (upper_step <= 0.0),
            0.0,
            jnp.where(
                active_sub & upper_limited & (dk * alpha > upper_step),
                upper_step / dk,
                next_temp1,
            ),
        )
        update_alpha = active_sub & (next_temp1 < alpha)
        alpha = jnp.where(update_alpha, next_temp1, alpha)
        ibd = jnp.where(update_alpha, jnp.asarray(i, dtype=jnp.int32), ibd)
        temp1 = next_temp1

    ibd_k = ind[ibd]
    ibd_d = safeguarded_d[ibd]
    alpha_hits_bound = safeguard & (alpha < 1.0)
    safeguarded_x = safeguarded_x.at[ibd_k].set(
        jnp.where(
            alpha_hits_bound & (ibd_d > 0.0),
            u[ibd_k],
            jnp.where(
                alpha_hits_bound & (ibd_d < 0.0),
                l[ibd_k],
                safeguarded_x[ibd_k],
            ),
        )
    )
    safeguarded_d = safeguarded_d.at[ibd].set(
        jnp.where(alpha_hits_bound & (ibd_d != 0.0), 0.0, safeguarded_d[ibd])
    )

    for i in range(n):
        active_sub = safeguard & (i < nsub)
        k = ind[i]
        safeguarded_x = safeguarded_x.at[k].set(
            jnp.where(
                active_sub,
                safeguarded_x[k] + alpha * safeguarded_d[i],
                safeguarded_x[k],
            )
        )

    final_x = jnp.where(safeguard, safeguarded_x, projected_x)
    final_d = jnp.where(safeguard, safeguarded_d, d)
    final_x = jnp.where(valid, final_x, original_x)
    final_d = jnp.where(valid, final_d, original_d)
    final_xp = jnp.where(valid, xp, original_xp)
    return LbfgsbSubsmResult(
        x=final_x,
        d=final_d,
        xp=final_xp,
        iword=jnp.where(valid, iword, jnp.asarray(0, dtype=jnp.int32)),
        wv=wv,
        info=jnp.where(
            valid, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32)
        ),
    )


def lbfgsb_lnsrlb(
    l,
    u,
    nbd,
    x,
    f,
    fold,
    gd,
    gdold,
    g,
    d,
    r,
    t,
    z,
    stp,
    dnorm,
    dtd,
    xstep,
    stpmx,
    iteration,
    ifun,
    iback,
    nfgv,
    info,
    task,
    task_msg,
    boxed,
    cnstnd,
    isave,
    dsave,
    temp_task,
    temp_task_msg,
):
    l = jnp.asarray(l, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    nbd = jnp.asarray(nbd, dtype=jnp.int32)
    x = jnp.asarray(x, dtype=jnp.float64)
    f = jnp.asarray(f, dtype=jnp.float64)
    fold = jnp.asarray(fold, dtype=jnp.float64)
    gd = jnp.asarray(gd, dtype=jnp.float64)
    gdold = jnp.asarray(gdold, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    stp = jnp.asarray(stp, dtype=jnp.float64)
    dnorm = jnp.asarray(dnorm, dtype=jnp.float64)
    dtd = jnp.asarray(dtd, dtype=jnp.float64)
    xstep = jnp.asarray(xstep, dtype=jnp.float64)
    stpmx = jnp.asarray(stpmx, dtype=jnp.float64)
    iteration = jnp.asarray(iteration, dtype=jnp.int32)
    ifun = jnp.asarray(ifun, dtype=jnp.int32)
    iback = jnp.asarray(iback, dtype=jnp.int32)
    nfgv = jnp.asarray(nfgv, dtype=jnp.int32)
    info = jnp.asarray(info, dtype=jnp.int32)
    task = jnp.asarray(task, dtype=jnp.int32)
    task_msg = jnp.asarray(task_msg, dtype=jnp.int32)
    boxed = jnp.asarray(boxed, dtype=jnp.bool_)
    cnstnd = jnp.asarray(cnstnd, dtype=jnp.bool_)
    isave = jnp.asarray(isave, dtype=jnp.int32)
    dsave = jnp.asarray(dsave, dtype=jnp.float64)
    temp_task = jnp.asarray(temp_task, dtype=jnp.int32)
    temp_task_msg = jnp.asarray(temp_task_msg, dtype=jnp.int32)
    n = int(x.shape[0])

    def setup_branch(_):
        next_dnorm = jnp.sqrt(jnp.sum(d * d))
        next_dtd = next_dnorm * next_dnorm
        next_stpmx = jnp.asarray(1.0e10, dtype=jnp.float64)

        for i in range(n):
            direction_i = d[i]
            lower_step = l[i] - x[i]
            upper_step = u[i] - x[i]
            lower_active = (
                cnstnd
                & (iteration != 0)
                & (nbd[i] != 0)
                & (direction_i < 0.0)
                & (nbd[i] <= NBD_BOTH)
            )
            upper_active = (
                cnstnd
                & (iteration != 0)
                & (nbd[i] != 0)
                & (direction_i > 0.0)
                & (nbd[i] >= NBD_BOTH)
            )
            next_stpmx = jnp.where(
                lower_active & (lower_step >= 0.0),
                0.0,
                jnp.where(
                    lower_active & (direction_i * next_stpmx < lower_step),
                    lower_step / direction_i,
                    next_stpmx,
                ),
            )
            next_stpmx = jnp.where(
                upper_active & (upper_step <= 0.0),
                0.0,
                jnp.where(
                    upper_active & (direction_i * next_stpmx > upper_step),
                    upper_step / direction_i,
                    next_stpmx,
                ),
            )

        next_stpmx = jnp.where(cnstnd & (iteration == 0), 1.0, next_stpmx)
        next_stp = jnp.where(
            (iteration == 0) & (~boxed),
            jnp.minimum(1.0 / next_dnorm, next_stpmx),
            1.0,
        )
        return (
            x,
            f,
            g,
            x,
            next_stp,
            next_dnorm,
            next_dtd,
            jnp.asarray(0.0, dtype=jnp.float64),
            next_stpmx,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(START, dtype=jnp.int32),
            jnp.asarray(NO_MSG, dtype=jnp.int32),
        )

    def continue_branch(_):
        return (
            x,
            fold,
            r,
            t,
            stp,
            dnorm,
            dtd,
            xstep,
            stpmx,
            ifun,
            iback,
            temp_task,
            temp_task_msg,
        )

    (
        x,
        fold,
        r,
        t,
        stp,
        dnorm,
        dtd,
        xstep,
        stpmx,
        ifun,
        iback,
        temp_task,
        temp_task_msg,
    ) = jax.lax.cond(task_msg == FG_LNSRCH, continue_branch, setup_branch, None)

    next_gd = jnp.dot(g, d, precision=jax.lax.Precision.HIGHEST)
    first_function_value = ifun == 0
    next_gdold = jnp.where(first_function_value, next_gd, gdold)
    non_descent = first_function_value & (next_gd >= 0.0)

    def non_descent_branch(_):
        return LbfgsbLnsrlbResult(
            x=x,
            fold=fold,
            gd=next_gd,
            gdold=next_gdold,
            g=g,
            r=r,
            t=t,
            stp=stp,
            dnorm=dnorm,
            dtd=dtd,
            xstep=xstep,
            stpmx=stpmx,
            ifun=ifun,
            iback=iback,
            nfgv=nfgv,
            info=jnp.asarray(-4, dtype=jnp.int32),
            task=task,
            task_msg=task_msg,
            isave=isave,
            dsave=dsave,
            temp_task=temp_task,
            temp_task_msg=temp_task_msg,
        )

    def line_search_branch(_):
        search = lbfgsb_dcsrch(
            f,
            next_gd,
            stp,
            1.0e-3,
            0.9,
            0.1,
            0.0,
            stpmx,
            temp_task,
            temp_task_msg,
            isave,
            dsave,
        )
        next_xstep = search.stp * dnorm
        line_search_done = (search.task == CONVERGENCE) | (search.task == WARNING)
        next_ifun = jnp.where(line_search_done, ifun, ifun + 1)
        next_nfgv = jnp.where(line_search_done, nfgv, nfgv + 1)
        next_iback = jnp.where(line_search_done, iback, next_ifun - 1)
        next_task = jnp.where(
            line_search_done,
            jnp.asarray(NEW_X, dtype=jnp.int32),
            jnp.asarray(FG, dtype=jnp.int32),
        )
        next_task_msg = jnp.where(
            line_search_done,
            jnp.asarray(NO_MSG, dtype=jnp.int32),
            jnp.asarray(FG_LNSRCH, dtype=jnp.int32),
        )
        trial_x = search.stp * d + t
        trial_x = jnp.where(
            (nbd == NBD_LOWER) | (nbd == NBD_BOTH), jnp.maximum(trial_x, l), trial_x
        )
        trial_x = jnp.where(
            (nbd == NBD_BOTH) | (nbd == NBD_UPPER), jnp.minimum(trial_x, u), trial_x
        )
        next_x = jnp.where(
            line_search_done,
            x,
            jnp.where(search.stp == 1.0, z, trial_x),
        )
        return LbfgsbLnsrlbResult(
            x=next_x,
            fold=fold,
            gd=next_gd,
            gdold=next_gdold,
            g=g,
            r=r,
            t=t,
            stp=search.stp,
            dnorm=dnorm,
            dtd=dtd,
            xstep=next_xstep,
            stpmx=stpmx,
            ifun=next_ifun,
            iback=next_iback,
            nfgv=next_nfgv,
            info=info,
            task=next_task,
            task_msg=next_task_msg,
            isave=search.isave,
            dsave=search.dsave,
            temp_task=search.task,
            temp_task_msg=search.task_msg,
        )

    return jax.lax.cond(non_descent, non_descent_branch, line_search_branch, None)


def lbfgsb_freev(nfree, idx, idx2, iwhere, updatd, cnstnd, iteration):
    nfree = jnp.asarray(nfree, dtype=jnp.int32)
    idx = jnp.asarray(idx, dtype=jnp.int32)
    idx2 = jnp.asarray(idx2, dtype=jnp.int32)
    iwhere = jnp.asarray(iwhere, dtype=jnp.int32)
    updatd = jnp.asarray(updatd, dtype=jnp.bool_)
    cnstnd = jnp.asarray(cnstnd, dtype=jnp.bool_)
    iteration = jnp.asarray(iteration, dtype=jnp.int32)
    n = int(idx.shape[0])

    nenter = jnp.asarray(0, dtype=jnp.int32)
    ileave = jnp.asarray(n, dtype=jnp.int32)
    count_changes = (iteration > 0) & cnstnd
    for i in range(n):
        active = count_changes & (i < nfree)
        k = idx[i]
        leaves = active & (iwhere[k] > 0)
        ileave_next = ileave - jnp.where(leaves, 1, 0)
        leave_slot = jnp.minimum(ileave_next, n - 1)
        idx2 = idx2.at[leave_slot].set(jnp.where(leaves, k, idx2[leave_slot]))
        ileave = ileave_next
    for i in range(n):
        active = count_changes & (i >= nfree)
        k = idx[i]
        enters = active & (iwhere[k] <= 0)
        enter_slot = jnp.minimum(nenter, n - 1)
        idx2 = idx2.at[enter_slot].set(jnp.where(enters, k, idx2[enter_slot]))
        nenter = nenter + jnp.where(enters, 1, 0)

    wrk = (ileave < n) | (nenter > 0) | updatd

    next_nfree = jnp.asarray(0, dtype=jnp.int32)
    iact = jnp.asarray(n, dtype=jnp.int32)
    next_idx = jnp.zeros_like(idx)
    for i in range(n):
        is_free = iwhere[i] <= 0
        free_slot = jnp.minimum(next_nfree, n - 1)
        active_slot = iact - 1
        next_idx = next_idx.at[free_slot].set(
            jnp.where(is_free, i, next_idx[free_slot])
        )
        next_idx = next_idx.at[active_slot].set(
            jnp.where(is_free, next_idx[active_slot], i)
        )
        next_nfree = next_nfree + jnp.where(is_free, 1, 0)
        iact = iact - jnp.where(is_free, 0, 1)

    return LbfgsbFreevResult(
        nfree=next_nfree,
        idx=next_idx,
        nenter=nenter,
        ileave=ileave,
        idx2=idx2,
        wrk=wrk,
    )
