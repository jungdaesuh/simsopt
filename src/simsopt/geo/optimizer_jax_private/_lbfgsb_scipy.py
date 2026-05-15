"""SciPy L-BFGS-B 1.17.1-compatible optimizer-control helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
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


def lbfgsb_public_status(task0: int, nfev: int, nit: int, maxfun: int, maxiter: int) -> int:
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
        raise ValueError("LBFGSB - one of the lower bounds is greater than an upper bound.")
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
    next_sty = jnp.where(sty_update, stp, jnp.where(opposite_derivative_update, stx, sty))
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
            stmin2 = jnp.where(brackt2, jnp.minimum(stx2, sty2), stp2 + 1.1 * (stp2 - stx2))
            stmax2 = jnp.where(brackt2, jnp.maximum(stx2, sty2), stp2 + 4.0 * (stp2 - stx2))
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
        first_rhs = first_rhs.at[i].set(jnp.where(active_i, first_rhs[i] + ssum, first_rhs[i]))

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
        info=jnp.where(finite, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(-3, dtype=jnp.int32)),
    )


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
        next_idx = next_idx.at[free_slot].set(jnp.where(is_free, i, next_idx[free_slot]))
        next_idx = next_idx.at[active_slot].set(jnp.where(is_free, next_idx[active_slot], i))
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
