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


class LbfgsbCauchyResult(NamedTuple):
    iorder: jax.Array
    iwhere: jax.Array
    t: jax.Array
    d: jax.Array
    xcp: jax.Array
    p: jax.Array
    c: jax.Array
    wbp: jax.Array
    v: jax.Array
    nseg: jax.Array
    info: jax.Array


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


class LbfgsbInverseHessianHistory(NamedTuple):
    s: jax.Array
    y: jax.Array
    n_corrs: jax.Array


def lbfgsb_workspace_size(n: int, m: int) -> int:
    return 2 * m * n + 5 * n + 11 * m * m + 8 * m


def lbfgsb_iwa_size(n: int) -> int:
    return 3 * n


def _lbfgsb_history_size_from_workspace(n: int, workspace_size: int) -> int:
    n = int(n)
    linear = 2 * n + 8
    constant = 5 * n - int(workspace_size)
    return int(round((-linear + np.sqrt(linear * linear - 44 * constant)) / 22))


def _lbfgsb_workspace_offsets(n: int, m: int) -> tuple[int, ...]:
    n = int(n)
    m = int(m)
    mn = m * n
    mm = m * m
    four_mm = 4 * mm
    lws = 0
    lwy = lws + mn
    lsy = lwy + mn
    lss = lsy + mm
    lwt = lss + mm
    lwn = lwt + mm
    lsnd = lwn + four_mm
    lz = lsnd + four_mm
    lr = lz + n
    ld = lr + n
    lt = ld + n
    lxp = lt + n
    lwa = lxp + n
    return lws, lwy, lsy, lss, lwt, lwn, lsnd, lz, lr, ld, lt, lxp, lwa


def _lbfgsb_workspace_partition_isave(n: int, m: int) -> jax.Array:
    n = int(n)
    m = int(m)
    mn = m * n
    mm = m * m
    return jnp.asarray(
        (
            mn,
            mm,
            4 * mm,
            *_lbfgsb_workspace_offsets(n, m),
        ),
        dtype=jnp.int32,
    )


def _lbfgsb_fortran_square(values, size: int) -> jax.Array:
    return values.reshape((size, size)).T


def _lbfgsb_flatten_fortran_square(values) -> jax.Array:
    return values.T.reshape((-1,))


def _lbfgsb_task(task, task_msg) -> jax.Array:
    return jnp.asarray((task, task_msg), dtype=jnp.int32)


def _lbfgsb_lsave(prjctd, cnstnd, boxed, updatd) -> jax.Array:
    return jnp.asarray((prjctd, cnstnd, boxed, updatd), dtype=jnp.int32)


def _lbfgsb_lsave_flags(state: LbfgsbState):
    lsave = state.workspace.lsave
    return lsave[0] != 0, lsave[1] != 0, lsave[2] != 0, lsave[3] != 0


def _lbfgsb_state_dimensions(state: LbfgsbState) -> tuple[int, int]:
    n = int(state.x.shape[0])
    return n, _lbfgsb_history_size_from_workspace(
        n,
        int(state.workspace.wa.shape[0]),
    )


def _lbfgsb_ddot(x, y) -> jax.Array:
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(int(x.shape[0])):
        product = x[i] * y[i]
        # Adding exact zero is a no-op in SciPy's BLAS accumulation.
        total = jax.lax.cond(
            product != 0.0,
            lambda value: value + product,
            lambda value: value,
            total,
        )
    return total


def _lbfgsb_dnrm2(x) -> jax.Array:
    return jnp.sqrt(_lbfgsb_ddot(x, x))


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
    if int(maxls) <= 0:
        raise ValueError("maxls must be positive.")
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


def _lbfgsb_setulb_start(state: LbfgsbState) -> LbfgsbState:
    n, m = _lbfgsb_state_dimensions(state)
    active = lbfgsb_active(state.l, state.u, state.nbd, state.x)

    iwa = state.workspace.iwa.at[n : 2 * n].set(active.iwhere)
    lsave = _lbfgsb_lsave(active.prjctd, active.cnstnd, active.boxed, False)
    isave = state.workspace.isave.at[:16].set(_lbfgsb_workspace_partition_isave(n, m))
    isave = isave.at[37].set(jnp.asarray(n, dtype=jnp.int32))
    dsave = state.workspace.dsave.at[0].set(jnp.asarray(1.0, dtype=jnp.float64))
    dsave = dsave.at[2].set(
        state.factr * jnp.asarray(np.finfo(float).eps, dtype=jnp.float64)
    )

    workspace = state.workspace._replace(
        iwa=iwa,
        task=_lbfgsb_task(FG, FG_START),
        lsave=lsave,
        isave=isave,
        dsave=dsave,
    )
    return state._replace(x=active.x, workspace=workspace)


def _lbfgsb_setulb_fg_start_line_search(state: LbfgsbState, sbgnrm) -> LbfgsbState:
    n, m = _lbfgsb_state_dimensions(state)
    lws, lwy, lsy, lss, lwt, lwn, lsnd, lz, lr, ld, lt, lxp, lwa = (
        _lbfgsb_workspace_offsets(n, m)
    )
    wa = state.workspace.wa
    iwa = state.workspace.iwa
    isave = state.workspace.isave
    dsave = state.workspace.dsave

    ws = wa[lws:lwy].reshape((m, n))
    wy = wa[lwy:lsy].reshape((m, n))
    sy = _lbfgsb_fortran_square(wa[lsy:lss], m)
    wt = _lbfgsb_fortran_square(wa[lwt:lwn], m)
    wn = _lbfgsb_fortran_square(wa[lwn:lsnd], 2 * m)
    snd = _lbfgsb_fortran_square(wa[lsnd:lz], 2 * m)

    index = iwa[:n]
    iwhere = iwa[n : 2 * n]
    indx2 = iwa[2 * n : 3 * n]
    t = wa[lt:lxp]
    d = wa[ld:lt]
    z = wa[lz:lr]
    scratch = wa[lwa:]
    p = scratch[: 2 * m]
    c = scratch[2 * m : 4 * m]
    wbp = scratch[4 * m : 6 * m]
    v = scratch[6 * m : 8 * m]

    prjctd, cnstnd, boxed, updatd = _lbfgsb_lsave_flags(state)

    nintol = isave[21]
    iback = isave[24]
    head = isave[26]
    col = isave[27]
    iteration = isave[29]
    iupdat = isave[30]
    nfree = isave[37]
    theta = dsave[0]

    cauchy = lbfgsb_cauchy(
        state.x,
        state.l,
        state.u,
        state.nbd,
        state.g,
        indx2,
        iwhere,
        t,
        d,
        z,
        wy,
        ws,
        sy,
        wt,
        theta,
        col,
        head,
        p,
        c,
        wbp,
        v,
        sbgnrm,
    )
    cauchy_scratch = scratch
    cauchy_scratch = cauchy_scratch.at[: 2 * m].set(cauchy.p)
    cauchy_scratch = cauchy_scratch.at[2 * m : 4 * m].set(cauchy.c)
    cauchy_scratch = cauchy_scratch.at[4 * m : 6 * m].set(cauchy.wbp)
    cauchy_scratch = cauchy_scratch.at[6 * m : 8 * m].set(cauchy.v)
    nintol = nintol + cauchy.nseg
    free = lbfgsb_freev(
        nfree,
        index,
        cauchy.iorder,
        cauchy.iwhere,
        updatd,
        cnstnd,
        iteration,
    )
    nact = jnp.asarray(n, dtype=jnp.int32) - free.nfree
    subspace_active = (free.nfree != 0) & (col != 0)

    def subspace_branch(_):
        formed = jax.lax.cond(
            free.wrk,
            lambda _: lbfgsb_formk(
                free.nfree,
                free.idx,
                free.nenter,
                free.ileave,
                free.idx2,
                iupdat,
                updatd,
                wn,
                snd,
                ws,
                wy,
                sy,
                theta,
                col,
                head,
            ),
            lambda _: LbfgsbFormkResult(
                wn=wn,
                wn1=snd,
                info=jnp.asarray(0, dtype=jnp.int32),
            ),
            None,
        )
        compressed = lbfgsb_cmprlb(
            state.x,
            state.g,
            ws,
            wy,
            sy,
            wt,
            cauchy.xcp,
            wa[lr:ld],
            cauchy_scratch,
            free.idx,
            theta,
            col,
            head,
            free.nfree,
            cnstnd,
        )
        subspace = lbfgsb_subsm(
            free.nfree,
            free.idx,
            state.l,
            state.u,
            state.nbd,
            cauchy.xcp,
            compressed.r,
            wa[lxp:lwa],
            ws,
            wy,
            theta,
            state.x,
            state.g,
            col,
            head,
            cauchy_scratch[: 2 * m],
            formed.wn,
        )
        return (
            formed.wn,
            formed.wn1,
            subspace.x,
            compressed.r,
            subspace.xp,
            subspace.wv,
            subspace.iword,
            formed.info,
            compressed.info,
            subspace.info,
        )

    def cauchy_branch(_):
        no_info = jnp.asarray(0, dtype=jnp.int32)
        return (
            wn,
            snd,
            cauchy.xcp,
            wa[lr:ld],
            wa[lxp:lwa],
            cauchy_scratch[: 2 * m],
            jnp.asarray(-1, dtype=jnp.int32),
            no_info,
            no_info,
            no_info,
        )

    (
        wn_next,
        snd_next,
        z_next,
        r_next,
        xp_next,
        wv_next,
        iword,
        formk_info,
        cmprlb_info,
        subsm_info,
    ) = jax.lax.cond(subspace_active, subspace_branch, cauchy_branch, None)
    line_direction = z_next - state.x
    line_search_nfgv = jnp.where(
        (state.workspace.task[0] == FG) & (state.workspace.task[1] == FG_START),
        jnp.asarray(1, dtype=jnp.int32),
        isave[33],
    )
    restart_from_geometry = (cauchy.info != 0) | (
        subspace_active & ((formk_info != 0) | (cmprlb_info != 0) | (subsm_info != 0))
    )
    geometry_wa = wa
    geometry_iwa = iwa
    geometry_isave = isave
    geometry_dsave = dsave
    search = jax.lax.cond(
        restart_from_geometry,
        lambda _: _lbfgsb_skipped_lnsrlb_result(
            state,
            r_next,
            wa[lt:lxp],
            dsave,
            isave,
            line_search_nfgv,
        ),
        lambda _: lbfgsb_lnsrlb(
            state.l,
            state.u,
            state.nbd,
            state.x,
            state.f,
            dsave[1],
            dsave[10],
            dsave[14],
            state.g,
            line_direction,
            r_next,
            wa[lt:lxp],
            z_next,
            dsave[13],
            dsave[3],
            dsave[15],
            jnp.asarray(0.0, dtype=jnp.float64),
            dsave[11],
            iteration,
            isave[35],
            iback,
            line_search_nfgv,
            isave[34],
            state.workspace.task[0],
            state.workspace.task[1],
            boxed,
            cnstnd,
            isave[42:44],
            dsave[16:29],
            state.workspace.ln_task[0],
            state.workspace.ln_task[1],
        ),
        None,
    )

    wa = wa.at[lwn:lsnd].set(_lbfgsb_flatten_fortran_square(wn_next))
    wa = wa.at[lsnd:lz].set(_lbfgsb_flatten_fortran_square(snd_next))
    wa = wa.at[lz:lr].set(z_next)
    wa = wa.at[lr:ld].set(search.r)
    wa = wa.at[ld:lt].set(line_direction)
    wa = wa.at[lt:lxp].set(search.t)
    wa = wa.at[lxp:lwa].set(xp_next)
    wa = wa.at[lwa : lwa + 2 * m].set(wv_next)
    wa = wa.at[lwa + 2 * m : lwa + 4 * m].set(cauchy_scratch[2 * m : 4 * m])
    wa = wa.at[lwa + 4 * m : lwa + 6 * m].set(cauchy_scratch[4 * m : 6 * m])
    wa = wa.at[lwa + 6 * m : lwa + 8 * m].set(cauchy_scratch[6 * m : 8 * m])

    iwa = iwa.at[:n].set(free.idx)
    iwa = iwa.at[n : 2 * n].set(cauchy.iwhere)
    iwa = iwa.at[2 * n : 3 * n].set(free.idx2)

    isave = isave.at[21].set(nintol)
    isave = isave.at[24].set(search.iback)
    isave = isave.at[32].set(cauchy.nseg)
    isave = isave.at[33].set(search.nfgv)
    isave = isave.at[34].set(search.info)
    isave = isave.at[35].set(search.ifun)
    isave = isave.at[36].set(iword)
    isave = isave.at[37].set(free.nfree)
    isave = isave.at[38].set(nact)
    isave = isave.at[39].set(free.ileave)
    isave = isave.at[40].set(free.nenter)
    isave = isave.at[42:44].set(search.isave)

    dsave = dsave.at[1].set(search.fold)
    dsave = dsave.at[3].set(search.dnorm)
    dsave = dsave.at[10].set(search.gd)
    dsave = dsave.at[11].set(search.stpmx)
    dsave = dsave.at[12].set(sbgnrm)
    dsave = dsave.at[13].set(search.stp)
    dsave = dsave.at[14].set(search.gdold)
    dsave = dsave.at[15].set(search.dtd)
    dsave = dsave.at[16:29].set(search.dsave)

    workspace = state.workspace._replace(
        wa=wa,
        iwa=iwa,
        task=_lbfgsb_task(search.task, search.task_msg),
        ln_task=_lbfgsb_task(search.temp_task, search.temp_task_msg),
        lsave=_lbfgsb_lsave(prjctd, cnstnd, boxed, updatd),
        isave=isave,
        dsave=dsave,
    )
    normal_state = state._replace(x=search.x, workspace=workspace)
    line_search_stopped = _lbfgsb_line_search_stops_iteration(search, state.maxls)
    restart_from_line_search = line_search_stopped & (col != 0)
    restart = restart_from_geometry | restart_from_line_search
    refreshed_geometry_state = _lbfgsb_setulb_refreshed_memory_state(
        state,
        geometry_wa,
        geometry_iwa,
        geometry_isave,
        geometry_dsave,
        task=jnp.asarray(RESTART, dtype=jnp.int32),
        task_msg=jnp.asarray(NO_MSG, dtype=jnp.int32),
        x=state.x,
        f=state.f,
        g=state.g,
    )
    refreshed_line_search_state = _lbfgsb_setulb_restart_after_line_search(
        state,
        wa,
        iwa,
        isave,
        dsave,
        search,
        lr=lr,
        ld=ld,
        lt=lt,
        lxp=lxp,
        sbgnrm=sbgnrm,
    )
    refreshed_state = jax.lax.cond(
        restart_from_line_search,
        lambda _: refreshed_line_search_state,
        lambda _: refreshed_geometry_state,
        None,
    )
    next_state = jax.lax.cond(
        line_search_stopped & (col == 0) & (~restart_from_geometry),
        lambda _: _lbfgsb_setulb_line_search_abnormal(
            state,
            wa,
            isave,
            dsave,
            search,
            lr=lr,
            ld=ld,
            lt=lt,
            lxp=lxp,
            sbgnrm=sbgnrm,
        ),
        lambda _: jax.lax.cond(
            restart,
            lambda _: refreshed_state,
            lambda _: normal_state,
            None,
        ),
        None,
    )
    return next_state


def _lbfgsb_setulb_subspace_line_search(state: LbfgsbState, sbgnrm) -> LbfgsbState:
    n, m = _lbfgsb_state_dimensions(state)
    lws, lwy, lsy, lss, lwt, lwn, lsnd, lz, lr, ld, lt, lxp, lwa = (
        _lbfgsb_workspace_offsets(n, m)
    )
    wa = state.workspace.wa
    iwa = state.workspace.iwa
    isave = state.workspace.isave
    dsave = state.workspace.dsave

    ws = wa[lws:lwy].reshape((m, n))
    wy = wa[lwy:lsy].reshape((m, n))
    sy = _lbfgsb_fortran_square(wa[lsy:lss], m)
    wt = _lbfgsb_fortran_square(wa[lwt:lwn], m)
    wn = _lbfgsb_fortran_square(wa[lwn:lsnd], 2 * m)
    snd = _lbfgsb_fortran_square(wa[lsnd:lz], 2 * m)
    z = state.x
    r = wa[lr:ld]
    xp = wa[lxp:lwa]
    scratch = wa[lwa:]
    wv = scratch[: 2 * m]

    prjctd, cnstnd, boxed, updatd = _lbfgsb_lsave_flags(state)

    index = iwa[:n]
    indx2 = iwa[2 * n : 3 * n]
    iback = isave[24]
    head = isave[26]
    col = isave[27]
    iteration = isave[29]
    iupdat = isave[30]
    nseg = jnp.asarray(0, dtype=jnp.int32)
    nfree = isave[37]
    ileave = isave[39]
    nenter = isave[40]
    theta = dsave[0]

    formed = lbfgsb_formk(
        nfree,
        index,
        nenter,
        ileave,
        indx2,
        iupdat,
        updatd,
        wn,
        snd,
        ws,
        wy,
        sy,
        theta,
        col,
        head,
    )
    compressed = lbfgsb_cmprlb(
        state.x,
        state.g,
        ws,
        wy,
        sy,
        wt,
        z,
        r,
        scratch,
        index,
        theta,
        col,
        head,
        nfree,
        cnstnd,
    )
    subspace = lbfgsb_subsm(
        nfree,
        index,
        state.l,
        state.u,
        state.nbd,
        z,
        compressed.r,
        xp,
        ws,
        wy,
        theta,
        state.x,
        state.g,
        col,
        head,
        wv,
        formed.wn,
    )
    line_direction = subspace.x - state.x
    restart_from_geometry = (
        (formed.info != 0) | (compressed.info != 0) | (subspace.info != 0)
    )
    geometry_wa = wa
    geometry_isave = isave
    geometry_dsave = dsave
    search = jax.lax.cond(
        restart_from_geometry,
        lambda _: _lbfgsb_skipped_lnsrlb_result(
            state,
            compressed.r,
            wa[lt:lxp],
            dsave,
            isave,
            isave[33],
        ),
        lambda _: lbfgsb_lnsrlb(
            state.l,
            state.u,
            state.nbd,
            state.x,
            state.f,
            dsave[1],
            dsave[10],
            dsave[14],
            state.g,
            line_direction,
            compressed.r,
            wa[lt:lxp],
            subspace.x,
            dsave[13],
            dsave[3],
            dsave[15],
            jnp.asarray(0.0, dtype=jnp.float64),
            dsave[11],
            iteration,
            isave[35],
            iback,
            isave[33],
            isave[34],
            state.workspace.task[0],
            state.workspace.task[1],
            boxed,
            cnstnd,
            isave[42:44],
            dsave[16:29],
            state.workspace.ln_task[0],
            state.workspace.ln_task[1],
        ),
        None,
    )

    wa = wa.at[lwn:lsnd].set(_lbfgsb_flatten_fortran_square(formed.wn))
    wa = wa.at[lsnd:lz].set(_lbfgsb_flatten_fortran_square(formed.wn1))
    wa = wa.at[lz:lr].set(subspace.x)
    wa = wa.at[lr:ld].set(search.r)
    wa = wa.at[ld:lt].set(line_direction)
    wa = wa.at[lt:lxp].set(search.t)
    wa = wa.at[lxp:lwa].set(subspace.xp)
    wa = wa.at[lwa : lwa + 2 * m].set(subspace.wv)

    isave = isave.at[24].set(search.iback)
    isave = isave.at[32].set(nseg)
    isave = isave.at[33].set(search.nfgv)
    isave = isave.at[34].set(search.info)
    isave = isave.at[35].set(search.ifun)
    isave = isave.at[36].set(subspace.iword)
    isave = isave.at[42:44].set(search.isave)

    dsave = dsave.at[1].set(search.fold)
    dsave = dsave.at[3].set(search.dnorm)
    dsave = dsave.at[10].set(search.gd)
    dsave = dsave.at[11].set(search.stpmx)
    dsave = dsave.at[12].set(sbgnrm)
    dsave = dsave.at[13].set(search.stp)
    dsave = dsave.at[14].set(search.gdold)
    dsave = dsave.at[15].set(search.dtd)
    dsave = dsave.at[16:29].set(search.dsave)

    workspace = state.workspace._replace(
        wa=wa,
        task=_lbfgsb_task(search.task, search.task_msg),
        ln_task=_lbfgsb_task(search.temp_task, search.temp_task_msg),
        lsave=_lbfgsb_lsave(prjctd, cnstnd, boxed, updatd),
        isave=isave,
        dsave=dsave,
    )
    normal_state = state._replace(x=search.x, workspace=workspace)
    line_search_stopped = _lbfgsb_line_search_stops_iteration(search, state.maxls)
    restart_from_line_search = line_search_stopped & (col != 0)
    restart = restart_from_geometry | restart_from_line_search
    refreshed_geometry_state = _lbfgsb_setulb_refreshed_memory_state(
        state,
        geometry_wa,
        iwa,
        geometry_isave,
        geometry_dsave,
        task=jnp.asarray(RESTART, dtype=jnp.int32),
        task_msg=jnp.asarray(NO_MSG, dtype=jnp.int32),
        x=state.x,
        f=state.f,
        g=state.g,
    )
    refreshed_line_search_state = _lbfgsb_setulb_restart_after_line_search(
        state,
        wa,
        iwa,
        isave,
        dsave,
        search,
        lr=lr,
        ld=ld,
        lt=lt,
        lxp=lxp,
        sbgnrm=sbgnrm,
    )
    refreshed_state = jax.lax.cond(
        restart_from_line_search,
        lambda _: refreshed_line_search_state,
        lambda _: refreshed_geometry_state,
        None,
    )
    next_state = jax.lax.cond(
        line_search_stopped & (col == 0) & (~restart_from_geometry),
        lambda _: _lbfgsb_setulb_line_search_abnormal(
            state,
            wa,
            isave,
            dsave,
            search,
            lr=lr,
            ld=ld,
            lt=lt,
            lxp=lxp,
            sbgnrm=sbgnrm,
        ),
        lambda _: jax.lax.cond(
            restart,
            lambda _: refreshed_state,
            lambda _: normal_state,
            None,
        ),
        None,
    )

    return next_state


def _lbfgsb_setulb_line_search_continue(state: LbfgsbState) -> LbfgsbState:
    n, m = _lbfgsb_state_dimensions(state)
    _, _, _, _, _, _, _, lz, lr, ld, lt, lxp, _ = _lbfgsb_workspace_offsets(n, m)
    wa = state.workspace.wa
    iwa = state.workspace.iwa
    isave = state.workspace.isave
    dsave = state.workspace.dsave

    prjctd, cnstnd, boxed, updatd = _lbfgsb_lsave_flags(state)

    iback = isave[24]

    search = lbfgsb_lnsrlb(
        state.l,
        state.u,
        state.nbd,
        state.x,
        state.f,
        dsave[1],
        dsave[10],
        dsave[14],
        state.g,
        wa[ld:lt],
        wa[lr:ld],
        wa[lt:lxp],
        wa[lz:lr],
        dsave[13],
        dsave[3],
        dsave[15],
        jnp.asarray(0.0, dtype=jnp.float64),
        dsave[11],
        isave[29],
        isave[35],
        iback,
        isave[33],
        isave[34],
        state.workspace.task[0],
        state.workspace.task[1],
        boxed,
        cnstnd,
        isave[42:44],
        dsave[16:29],
        state.workspace.ln_task[0],
        state.workspace.ln_task[1],
    )

    accepted = search.task == NEW_X
    iteration = jnp.where(accepted, isave[29] + 1, isave[29])
    sbgnrm = jnp.where(
        accepted,
        lbfgsb_projected_gradient_norm(state.l, state.u, state.nbd, search.x, state.g),
        dsave[12],
    )

    wa = wa.at[lr:ld].set(search.r)
    wa = wa.at[lt:lxp].set(search.t)

    isave = isave.at[24].set(search.iback)
    isave = isave.at[29].set(iteration)
    isave = isave.at[33].set(search.nfgv)
    isave = isave.at[34].set(search.info)
    isave = isave.at[35].set(search.ifun)
    isave = isave.at[42:44].set(search.isave)

    dsave = dsave.at[1].set(search.fold)
    dsave = dsave.at[3].set(search.dnorm)
    dsave = dsave.at[10].set(search.gd)
    dsave = dsave.at[11].set(search.stpmx)
    dsave = dsave.at[12].set(sbgnrm)
    dsave = dsave.at[13].set(search.stp)
    dsave = dsave.at[14].set(search.gdold)
    dsave = dsave.at[15].set(search.dtd)
    dsave = dsave.at[16:29].set(search.dsave)

    workspace = state.workspace._replace(
        wa=wa,
        task=_lbfgsb_task(search.task, search.task_msg),
        ln_task=_lbfgsb_task(search.temp_task, search.temp_task_msg),
        lsave=_lbfgsb_lsave(prjctd, cnstnd, boxed, updatd),
        isave=isave,
        dsave=dsave,
    )
    normal_state = state._replace(
        x=search.x,
        workspace=workspace,
        n_iterations=iteration,
    )
    line_search_stopped = _lbfgsb_line_search_stops_iteration(search, state.maxls)
    restart = line_search_stopped & (isave[27] != 0)
    refreshed_state = _lbfgsb_setulb_restart_after_line_search(
        state,
        wa,
        iwa,
        isave,
        dsave,
        search,
        lr=lr,
        ld=ld,
        lt=lt,
        lxp=lxp,
        sbgnrm=sbgnrm,
    )
    next_state = jax.lax.cond(
        line_search_stopped & (isave[27] == 0),
        lambda _: _lbfgsb_setulb_line_search_abnormal(
            state,
            wa,
            isave,
            dsave,
            search,
            lr=lr,
            ld=ld,
            lt=lt,
            lxp=lxp,
            sbgnrm=sbgnrm,
        ),
        lambda _: jax.lax.cond(
            restart,
            lambda _: refreshed_state,
            lambda _: normal_state,
            None,
        ),
        None,
    )
    return next_state


def _lbfgsb_line_search_stops_iteration(search: LbfgsbLnsrlbResult, maxls):
    return (search.info != 0) | (search.iback >= maxls)


def _lbfgsb_skipped_lnsrlb_result(
    state: LbfgsbState,
    r,
    t,
    dsave,
    isave,
    nfgv,
) -> LbfgsbLnsrlbResult:
    return LbfgsbLnsrlbResult(
        x=state.x,
        fold=dsave[1],
        gd=dsave[10],
        gdold=dsave[14],
        g=state.g,
        r=r,
        t=t,
        stp=dsave[13],
        dnorm=dsave[3],
        dtd=dsave[15],
        xstep=jnp.asarray(0.0, dtype=jnp.float64),
        stpmx=dsave[11],
        ifun=isave[35],
        iback=isave[24],
        nfgv=nfgv,
        info=jnp.asarray(0, dtype=jnp.int32),
        task=state.workspace.task[0],
        task_msg=state.workspace.task[1],
        isave=isave[42:44],
        dsave=dsave[16:29],
        temp_task=state.workspace.ln_task[0],
        temp_task_msg=state.workspace.ln_task[1],
    )


def _lbfgsb_setulb_refreshed_memory_state(
    state: LbfgsbState,
    wa,
    iwa,
    isave,
    dsave,
    *,
    task,
    task_msg,
    x,
    f,
    g,
) -> LbfgsbState:
    prjctd, cnstnd, boxed, _ = _lbfgsb_lsave_flags(state)
    int_zero = jnp.asarray(0, dtype=jnp.int32)
    isave = isave.at[26].set(int_zero)
    isave = isave.at[27].set(int_zero)
    isave = isave.at[30].set(int_zero)
    isave = isave.at[34].set(int_zero)
    dsave = dsave.at[0].set(jnp.asarray(1.0, dtype=jnp.float64))
    workspace = state.workspace._replace(
        wa=wa,
        iwa=iwa,
        task=_lbfgsb_task(task, task_msg),
        lsave=_lbfgsb_lsave(prjctd, cnstnd, boxed, False),
        isave=isave,
        dsave=dsave,
    )
    return state._replace(x=x, f=f, g=g, workspace=workspace)


def _lbfgsb_setulb_restart_after_line_search(
    state: LbfgsbState,
    wa,
    iwa,
    isave,
    dsave,
    search: LbfgsbLnsrlbResult,
    *,
    lr: int,
    ld: int,
    lt: int,
    lxp: int,
    sbgnrm,
) -> LbfgsbState:
    maxls_exhausted = search.info == 0
    maxls_correction = jnp.where(
        maxls_exhausted,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    nfgv = search.nfgv - maxls_correction
    ifun = search.ifun - maxls_correction
    iback = search.iback - maxls_correction

    wa = wa.at[lr:ld].set(search.r)
    wa = wa.at[lt:lxp].set(search.t)

    isave = isave.at[24].set(iback)
    isave = isave.at[33].set(nfgv)
    isave = isave.at[35].set(ifun)
    isave = isave.at[42:44].set(search.isave)

    dsave = dsave.at[1].set(search.fold)
    dsave = dsave.at[3].set(search.dnorm)
    dsave = dsave.at[10].set(search.gd)
    dsave = dsave.at[11].set(search.stpmx)
    dsave = dsave.at[12].set(sbgnrm)
    dsave = dsave.at[13].set(search.stp)
    dsave = dsave.at[14].set(search.gdold)
    dsave = dsave.at[15].set(search.dtd)
    dsave = dsave.at[16:29].set(search.dsave)

    return _lbfgsb_setulb_refreshed_memory_state(
        state,
        wa,
        iwa,
        isave,
        dsave,
        task=jnp.asarray(RESTART, dtype=jnp.int32),
        task_msg=jnp.asarray(NO_MSG, dtype=jnp.int32),
        x=search.t,
        f=search.fold,
        g=search.r,
    )


def _lbfgsb_setulb_line_search_abnormal(
    state: LbfgsbState,
    wa,
    isave,
    dsave,
    search: LbfgsbLnsrlbResult,
    *,
    lr: int,
    ld: int,
    lt: int,
    lxp: int,
    sbgnrm,
) -> LbfgsbState:
    prjctd, cnstnd, boxed, updatd = _lbfgsb_lsave_flags(state)
    maxls_exhausted = search.info == 0
    maxls_correction = jnp.where(
        maxls_exhausted,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    info = jnp.where(maxls_exhausted, jnp.asarray(-9, dtype=jnp.int32), search.info)
    nfgv = search.nfgv - maxls_correction
    ifun = search.ifun - maxls_correction
    iback = search.iback - maxls_correction

    wa = wa.at[lr:ld].set(search.r)
    wa = wa.at[lt:lxp].set(search.t)

    isave = isave.at[24].set(iback)
    isave = isave.at[29].set(isave[29] + jnp.asarray(1, dtype=jnp.int32))
    isave = isave.at[33].set(nfgv)
    isave = isave.at[34].set(info)
    isave = isave.at[35].set(ifun)
    isave = isave.at[42:44].set(search.isave)

    dsave = dsave.at[1].set(search.fold)
    dsave = dsave.at[3].set(search.dnorm)
    dsave = dsave.at[10].set(search.gd)
    dsave = dsave.at[11].set(search.stpmx)
    dsave = dsave.at[12].set(sbgnrm)
    dsave = dsave.at[13].set(search.stp)
    dsave = dsave.at[14].set(search.gdold)
    dsave = dsave.at[15].set(search.dtd)
    dsave = dsave.at[16:29].set(search.dsave)

    workspace = state.workspace._replace(
        wa=wa,
        task=_lbfgsb_task(ABNORMAL, NO_MSG),
        ln_task=_lbfgsb_task(search.temp_task, search.temp_task_msg),
        lsave=_lbfgsb_lsave(prjctd, cnstnd, boxed, updatd),
        isave=isave,
        dsave=dsave,
    )
    return state._replace(
        x=search.t,
        f=state.f,
        g=search.r,
        workspace=workspace,
    )


def _lbfgsb_setulb_new_x_convergence(
    state: LbfgsbState,
    task_msg: jax.Array,
    info: jax.Array,
) -> LbfgsbState:
    workspace = state.workspace._replace(
        task=_lbfgsb_task(CONVERGENCE, task_msg),
        isave=state.workspace.isave.at[34].set(info),
    )
    return state._replace(workspace=workspace)


def _lbfgsb_setulb_new_x_next_iteration(
    state: LbfgsbState,
    sbgnrm: jax.Array,
) -> LbfgsbState:
    n, m = _lbfgsb_state_dimensions(state)
    lws, lwy, lsy, lss, lwt, lwn, _, _, lr, ld, lt, _, _ = _lbfgsb_workspace_offsets(
        n, m
    )
    wa = state.workspace.wa
    isave = state.workspace.isave
    dsave = state.workspace.dsave

    ws = wa[lws:lwy].reshape((m, n))
    wy = wa[lwy:lsy].reshape((m, n))
    sy = _lbfgsb_fortran_square(wa[lsy:lss], m)
    ss = _lbfgsb_fortran_square(wa[lss:lwt], m)
    wt = _lbfgsb_fortran_square(wa[lwt:lwn], m)
    old_gradient = wa[lr:ld]
    line_direction = wa[ld:lt]

    gradient_delta = state.g - old_gradient
    rr_norm = _lbfgsb_dnrm2(gradient_delta)
    rr = rr_norm * rr_norm
    stp = dsave[13]
    gd = dsave[10]
    gdold = dsave[14]
    dr = jnp.where(stp == 1.0, gd - gdold, (gd - gdold) * stp)
    ddum = jnp.where(stp == 1.0, -gdold, -gdold * stp)
    next_direction = jnp.where(stp == 1.0, line_direction, stp * line_direction)
    skip_update = dr <= jnp.asarray(np.finfo(float).eps, dtype=jnp.float64) * ddum

    def skip_update_branch(_):
        workspace = state.workspace._replace(
            wa=wa.at[lr:ld].set(gradient_delta).at[ld:lt].set(next_direction),
            lsave=state.workspace.lsave.at[3].set(jnp.asarray(False, dtype=jnp.int32)),
            isave=state.workspace.isave.at[25].set(isave[25] + 1),
            dsave=state.workspace.dsave.at[12].set(sbgnrm),
        )
        return state._replace(workspace=workspace)

    def update_branch(_):
        iupdat = isave[30] + 1
        update = lbfgsb_matupd(
            ws,
            wy,
            sy,
            ss,
            next_direction,
            gradient_delta,
            isave[28],
            iupdat,
            isave[27],
            isave[26],
            rr,
            dr,
            stp,
            dsave[15],
        )
        form = lbfgsb_formt(wt, update.sy, update.ss, update.col, update.theta)
        refresh = form.info != 0
        next_col = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), update.col)
        next_head = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), update.head)
        next_theta = jnp.where(refresh, 1.0, update.theta)
        next_iupdat = jnp.where(refresh, jnp.asarray(0, dtype=jnp.int32), iupdat)
        next_updatd = ~refresh

        next_wa = wa
        next_wa = next_wa.at[lws:lwy].set(update.ws.reshape((-1,)))
        next_wa = next_wa.at[lwy:lsy].set(update.wy.reshape((-1,)))
        next_wa = next_wa.at[lsy:lss].set(_lbfgsb_flatten_fortran_square(update.sy))
        next_wa = next_wa.at[lss:lwt].set(_lbfgsb_flatten_fortran_square(update.ss))
        next_wa = next_wa.at[lwt:lwn].set(_lbfgsb_flatten_fortran_square(form.wt))
        next_wa = next_wa.at[lr:ld].set(gradient_delta)
        next_wa = next_wa.at[ld:lt].set(next_direction)

        next_isave = isave
        next_isave = next_isave.at[26].set(next_head)
        next_isave = next_isave.at[27].set(next_col)
        next_isave = next_isave.at[28].set(update.itail)
        next_isave = next_isave.at[30].set(next_iupdat)
        next_isave = next_isave.at[34].set(jnp.asarray(0, dtype=jnp.int32))

        next_dsave = dsave.at[0].set(next_theta)
        next_dsave = next_dsave.at[12].set(sbgnrm)

        workspace = state.workspace._replace(
            wa=next_wa,
            lsave=state.workspace.lsave.at[3].set(next_updatd.astype(jnp.int32)),
            isave=next_isave,
            dsave=next_dsave,
        )
        return state._replace(workspace=workspace)

    updated_state = jax.lax.cond(skip_update, skip_update_branch, update_branch, None)
    use_subspace_path = (updated_state.workspace.lsave[1] == 0) & (
        updated_state.workspace.isave[27] > 0
    )
    return jax.lax.cond(
        use_subspace_path,
        lambda state: _lbfgsb_setulb_subspace_line_search(state, sbgnrm),
        lambda state: _lbfgsb_setulb_fg_start_line_search(state, sbgnrm),
        updated_state,
    )


def _lbfgsb_setulb_new_x_reentry(state: LbfgsbState) -> LbfgsbState:
    dsave = state.workspace.dsave
    isave = state.workspace.isave
    sbgnrm = dsave[12]
    gradient_converged = sbgnrm <= state.pgtol
    reduction_scale = jnp.maximum(jnp.maximum(jnp.abs(dsave[1]), jnp.abs(state.f)), 1.0)
    reduction_converged = (dsave[1] - state.f) <= dsave[2] * reduction_scale
    reduction_info = jnp.where(
        isave[24] >= 10,
        jnp.asarray(-5, dtype=jnp.int32),
        isave[34],
    )

    return jax.lax.cond(
        gradient_converged,
        lambda _: _lbfgsb_setulb_new_x_convergence(
            state,
            jnp.asarray(CONV_GRAD, dtype=jnp.int32),
            isave[34],
        ),
        lambda _: jax.lax.cond(
            reduction_converged,
            lambda _: _lbfgsb_setulb_new_x_convergence(
                state,
                jnp.asarray(CONV_F, dtype=jnp.int32),
                reduction_info,
            ),
            lambda _: _lbfgsb_setulb_new_x_next_iteration(state, sbgnrm),
            None,
        ),
        None,
    )


def _lbfgsb_setulb_fg_start_converged(
    state: LbfgsbState, sbgnrm: jax.Array
) -> LbfgsbState:
    workspace = state.workspace._replace(
        task=_lbfgsb_task(CONVERGENCE, CONV_GRAD),
        isave=state.workspace.isave.at[33].set(jnp.asarray(1, dtype=jnp.int32)),
        dsave=state.workspace.dsave.at[12].set(sbgnrm),
    )
    return state._replace(workspace=workspace)


def _lbfgsb_setulb_fg_start_reentry(
    state: LbfgsbState, sbgnrm: jax.Array
) -> LbfgsbState:
    task_is_fg_start = (state.workspace.task[0] == FG) & (
        state.workspace.task[1] == FG_START
    )
    converged = task_is_fg_start & (sbgnrm <= state.pgtol)

    return jax.lax.cond(
        converged,
        lambda _: _lbfgsb_setulb_fg_start_converged(state, sbgnrm),
        lambda _: _lbfgsb_setulb_fg_start_line_search(state, sbgnrm),
        None,
    )


def _lbfgsb_setulb_reentry(state: LbfgsbState) -> LbfgsbState:
    sbgnrm = lbfgsb_projected_gradient_norm(
        state.l, state.u, state.nbd, state.x, state.g
    )
    task_is_restart = state.workspace.task[0] == RESTART
    task_is_new_x = state.workspace.task[0] == NEW_X
    task_is_fg_lnsrch = (state.workspace.task[0] == FG) & (
        state.workspace.task[1] == FG_LNSRCH
    )
    return jax.lax.cond(
        task_is_restart,
        lambda state: _lbfgsb_setulb_fg_start_line_search(state, sbgnrm),
        lambda state: jax.lax.cond(
            task_is_new_x,
            _lbfgsb_setulb_new_x_reentry,
            lambda state: jax.lax.cond(
                task_is_fg_lnsrch,
                _lbfgsb_setulb_line_search_continue,
                lambda state: _lbfgsb_setulb_fg_start_reentry(state, sbgnrm),
                state,
            ),
            state,
        ),
        state,
    )


def lbfgsb_setulb(state: LbfgsbState) -> LbfgsbState:
    original_f = state.f

    def continue_condition(carry):
        _, restart, count = carry
        return restart & (count < jnp.asarray(2, dtype=jnp.int32))

    def body(carry):
        current_state, _, count = carry
        next_state = jax.lax.cond(
            current_state.workspace.task[0] == START,
            _lbfgsb_setulb_start,
            _lbfgsb_setulb_reentry,
            current_state,
        )
        next_state = jax.lax.cond(
            current_state.workspace.task[0] == RESTART,
            lambda retry_state: retry_state._replace(f=original_f),
            lambda retry_state: retry_state,
            next_state,
        )
        return (
            next_state,
            next_state.workspace.task[0] == RESTART,
            count + jnp.asarray(1, dtype=jnp.int32),
        )

    final_state, _, _ = jax.lax.while_loop(
        continue_condition,
        body,
        (
            state,
            jnp.asarray(True, dtype=jnp.bool_),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return final_state


def _lbfgsb_evaluate_value_and_grad(value_and_grad, state: LbfgsbState) -> LbfgsbState:
    value, gradient = value_and_grad(state.x)
    return state._replace(
        f=jnp.asarray(value, dtype=jnp.float64),
        g=jnp.asarray(gradient, dtype=jnp.float64),
        nfev=state.nfev + jnp.asarray(1, dtype=jnp.int32),
        njev=state.njev + jnp.asarray(1, dtype=jnp.int32),
    )


def _lbfgsb_stop_after_new_x_limits(
    state: LbfgsbState,
    maxiter: jax.Array,
    maxfun: jax.Array,
) -> LbfgsbState:
    task = state.workspace.task
    is_new_x = task[0] == NEW_X
    iteration_limit = is_new_x & (state.n_iterations >= maxiter)
    function_limit = is_new_x & (~iteration_limit) & (state.nfev > maxfun)
    stopped = iteration_limit | function_limit
    stop_msg = jnp.where(
        iteration_limit,
        jnp.asarray(STOP_ITERC, dtype=jnp.int32),
        jnp.asarray(STOP_ITER, dtype=jnp.int32),
    )
    workspace = state.workspace._replace(
        task=_lbfgsb_task(
            jnp.where(stopped, jnp.asarray(STOP, dtype=jnp.int32), task[0]),
            jnp.where(stopped, stop_msg, task[1]),
        )
    )
    return state._replace(workspace=workspace)


def lbfgsb_mainlb(
    value_and_grad,
    state: LbfgsbState,
    *,
    maxiter: int,
    maxfun: int,
    accepted_step_callback=None,
) -> LbfgsbState:
    maxiter_array = jnp.asarray(maxiter, dtype=jnp.int32)
    maxfun_array = jnp.asarray(maxfun, dtype=jnp.int32)

    def continue_condition(state: LbfgsbState) -> jax.Array:
        return state.workspace.task[0] < CONVERGENCE

    def body(state: LbfgsbState) -> LbfgsbState:
        next_state = lbfgsb_setulb(state)
        next_state = jax.lax.cond(
            next_state.workspace.task[0] == FG,
            lambda state: _lbfgsb_evaluate_value_and_grad(value_and_grad, state),
            lambda state: state,
            next_state,
        )
        if accepted_step_callback is not None:
            next_state = jax.lax.cond(
                next_state.workspace.task[0] == NEW_X,
                lambda state: _lbfgsb_emit_accepted_step(
                    state,
                    accepted_step_callback,
                ),
                lambda state: state,
                next_state,
            )
        return _lbfgsb_stop_after_new_x_limits(
            next_state,
            maxiter_array,
            maxfun_array,
        )

    return jax.lax.while_loop(continue_condition, body, state)


def _lbfgsb_emit_accepted_step(
    state: LbfgsbState,
    accepted_step_callback,
) -> LbfgsbState:
    jax.debug.callback(
        accepted_step_callback,
        state.n_iterations,
        state.x,
        state.f,
        state.g,
        state.nfev,
        state.njev,
    )
    return state


def lbfgsb_inverse_hessian_history(
    state: LbfgsbState,
) -> LbfgsbInverseHessianHistory:
    n, m = _lbfgsb_state_dimensions(state)
    lws, lwy, lsy, *_ = _lbfgsb_workspace_offsets(n, m)
    s = state.workspace.wa[lws:lwy].reshape((m, n))
    y = state.workspace.wa[lwy:lsy].reshape((m, n))
    return LbfgsbInverseHessianHistory(
        s=s,
        y=y,
        n_corrs=jnp.minimum(
            state.workspace.isave[30],
            jnp.asarray(m, dtype=jnp.int32),
        ),
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


def lbfgsb_encode_bounds(bounds, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_bnd: np.ndarray = np.zeros(n, dtype=np.float64)
    upper_bnd: np.ndarray = np.zeros(n, dtype=np.float64)
    nbd: np.ndarray = np.zeros(n, dtype=np.int32)
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
        sy_value = _lbfgsb_ddot(d, wy[pointr])
        ss_value = _lbfgsb_ddot(ws[pointr], d)
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

    memory_index = jnp.arange(m, dtype=jnp.int32)
    active = memory_index < col
    upper = jnp.triu(wt)
    active_matrix = active[:, None] & active[None, :]
    solve_matrix = jnp.where(active_matrix, upper, jnp.eye(m, dtype=jnp.float64))
    diagonal = jnp.diag(upper)
    singular_index = jnp.min(
        jnp.where(
            active & (diagonal == 0.0),
            memory_index + 1,
            m + 1,
        )
    )
    factor_info = jnp.where(singular_index <= m, singular_index, 0)
    factor_ok = factor_info == 0
    rhs = jnp.where(active, first_rhs, 0.0)
    first_solve = jsp_linalg.solve_triangular(
        solve_matrix,
        rhs,
        trans=1,
        lower=False,
        unit_diagonal=False,
    )
    for i in range(m):
        p = p.at[col + i].set(jnp.where(i < col, first_rhs[i], p[col + i]))

    diag_sy = jnp.diag(sy)
    scaled_v = jnp.where(active, v[:m] / jnp.sqrt(diag_sy), 0.0)
    for i in range(m):
        update = factor_ok & (i < col)
        p = p.at[i].set(jnp.where(update, scaled_v[i], p[i]))
        p = p.at[col + i].set(jnp.where(update, first_solve[i], p[col + i]))

    second_rhs = jnp.zeros((m,), dtype=jnp.float64)
    for i in range(m):
        second_rhs = second_rhs.at[i].set(jnp.where(i < col, p[col + i], 0.0))
    second_solve = jsp_linalg.solve_triangular(
        solve_matrix,
        second_rhs,
        trans=0,
        lower=False,
        unit_diagonal=False,
    )
    for i in range(m):
        p = p.at[col + i].set(
            jnp.where(factor_ok & (i < col), second_solve[i], p[col + i])
        )
    negative_scaled_p = jnp.where(active, -p[:m] / jnp.sqrt(diag_sy), 0.0)
    for i in range(m):
        p = p.at[i].set(jnp.where(factor_ok & (i < col), negative_scaled_p[i], p[i]))

    for i in range(m):
        active_i = factor_ok & (i < col)
        ssum = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(i + 1, m):
            active_k = k < col
            ssum = ssum + jnp.where(
                active_i & active_k,
                sy[k, i] * p[col + k] / sy[i, i],
                0.0,
            )
        p = p.at[i].set(jnp.where(active_i, p[i] + ssum, p[i]))

    return LbfgsbBmvResult(p=p, info=factor_info)


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


def lbfgsb_cauchy(
    x,
    l,
    u,
    nbd,
    g,
    iorder,
    iwhere,
    t,
    d,
    xcp,
    wy,
    ws,
    sy,
    wt,
    theta,
    col,
    head,
    p,
    c,
    wbp,
    v,
    sbgnrm,
):
    x = jnp.asarray(x, dtype=jnp.float64)
    l = jnp.asarray(l, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    nbd = jnp.asarray(nbd, dtype=jnp.int32)
    g = jnp.asarray(g, dtype=jnp.float64)
    iorder = jnp.asarray(iorder, dtype=jnp.int32)
    iwhere = jnp.asarray(iwhere, dtype=jnp.int32)
    t = jnp.asarray(t, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    xcp = jnp.asarray(xcp, dtype=jnp.float64)
    wy = jnp.asarray(wy, dtype=jnp.float64)
    ws = jnp.asarray(ws, dtype=jnp.float64)
    sy = jnp.asarray(sy, dtype=jnp.float64)
    wt = jnp.asarray(wt, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    col = jnp.asarray(col, dtype=jnp.int32)
    head = jnp.asarray(head, dtype=jnp.int32)
    p = jnp.asarray(p, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    wbp = jnp.asarray(wbp, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)
    sbgnrm = jnp.asarray(sbgnrm, dtype=jnp.float64)
    n = int(x.shape[0])
    m = int(ws.shape[0])
    col2 = 2 * col
    active_col2 = jnp.arange(2 * m, dtype=jnp.int32) < col2
    has_gradient = sbgnrm > 0.0

    f1 = jnp.asarray(0.0, dtype=jnp.float64)
    nfree = jnp.asarray(n, dtype=jnp.int32)
    nbreak = jnp.asarray(-1, dtype=jnp.int32)
    ibkmin = jnp.asarray(0, dtype=jnp.int32)
    bkmin = jnp.asarray(0.0, dtype=jnp.float64)
    bnded = jnp.asarray(True, dtype=jnp.bool_)
    for i in range(2 * m):
        p = p.at[i].set(jnp.where((i < col2) & has_gradient, 0.0, p[i]))

    for i in range(n):
        neggi = -g[i]
        can_reset = has_gradient & (iwhere[i] != 3) & (iwhere[i] != -1)
        tl = jnp.where(nbd[i] <= NBD_BOTH, x[i] - l[i], 0.0)
        tu = jnp.where(nbd[i] >= NBD_BOTH, u[i] - x[i], 0.0)
        xlower = (nbd[i] <= NBD_BOTH) & (tl <= 0.0)
        xupper = (nbd[i] >= NBD_BOTH) & (tu <= 0.0)
        reset_iwhere = jnp.asarray(0, dtype=jnp.int32)
        reset_iwhere = jnp.where(xlower & (neggi <= 0.0), 1, reset_iwhere)
        reset_iwhere = jnp.where((~xlower) & xupper & (neggi >= 0.0), 2, reset_iwhere)
        reset_iwhere = jnp.where(
            (~xlower) & (~xupper) & (jnp.abs(neggi) <= 0.0),
            -3,
            reset_iwhere,
        )
        iwhere = iwhere.at[i].set(jnp.where(can_reset, reset_iwhere, iwhere[i]))

        movable = has_gradient & ((iwhere[i] == 0) | (iwhere[i] == -1))
        d = d.at[i].set(jnp.where(movable, neggi, 0.0))
        f1 = f1 - jnp.where(movable, neggi * neggi, 0.0)
        pointr = head
        for j in range(m):
            active_col = movable & (j < col)
            p = p.at[j].set(jnp.where(active_col, p[j] + wy[pointr, i] * neggi, p[j]))
            p = p.at[col + j].set(
                jnp.where(
                    active_col,
                    p[col + j] + ws[pointr, i] * neggi,
                    p[col + j],
                )
            )
            pointr = (pointr + 1) % m

        lower_break = (
            movable & (nbd[i] <= NBD_BOTH) & (nbd[i] != NBD_UNBOUNDED) & (neggi < 0.0)
        )
        upper_break = movable & (nbd[i] >= NBD_BOTH) & (neggi > 0.0)
        has_break = lower_break | upper_break
        next_nbreak = nbreak + jnp.where(has_break, 1, 0)
        breakpoint_value = jnp.where(lower_break, tl / (-neggi), tu / neggi)
        breakpoint_slot = jnp.maximum(next_nbreak, 0)
        t = t.at[breakpoint_slot].set(
            jnp.where(has_break, breakpoint_value, t[breakpoint_slot])
        )
        iorder = iorder.at[breakpoint_slot].set(
            jnp.where(has_break, i, iorder[breakpoint_slot])
        )
        new_minimum = has_break & ((nbreak == -1) | (breakpoint_value < bkmin))
        bkmin = jnp.where(new_minimum, breakpoint_value, bkmin)
        ibkmin = jnp.where(new_minimum, next_nbreak, ibkmin)
        nbreak = next_nbreak

        unbounded_move = movable & (~has_break)
        next_nfree = nfree - jnp.where(unbounded_move, 1, 0)
        free_slot = jnp.minimum(jnp.maximum(next_nfree, 0), n - 1)
        iorder = iorder.at[free_slot].set(
            jnp.where(unbounded_move, i, iorder[free_slot])
        )
        nfree = next_nfree
        bnded = jnp.where(unbounded_move & (jnp.abs(neggi) > 0.0), False, bnded)

    for i in range(m):
        p = p.at[col + i].set(
            jnp.where(
                has_gradient & (theta != 1.0) & (i < col),
                theta * p[col + i],
                p[col + i],
            )
        )

    xcp = x
    no_nonzero_direction = (nbreak == -1) & (nfree == n)
    for j in range(2 * m):
        c = c.at[j].set(jnp.where(has_gradient & active_col2[j], 0.0, c[j]))

    f2 = -theta * f1
    f2_org = f2
    bmv_initial = lbfgsb_bmv(sy, wt, col, p)
    initial_bmv_active = has_gradient & (col > 0) & (~no_nonzero_direction)
    v = jnp.where(initial_bmv_active, bmv_initial.p, v)
    initial_info = jnp.where(initial_bmv_active, bmv_initial.info, 0)
    f2 = jnp.where(
        initial_bmv_active,
        f2 - jnp.sum(jnp.where(active_col2, v * p, 0.0)),
        f2,
    )
    dtm = -f1 / f2
    tsum = jnp.asarray(0.0, dtype=jnp.float64)
    nseg = jnp.asarray(1, dtype=jnp.int32)
    no_breakpoints = nbreak == -1
    skip_loop = (
        (~has_gradient) | no_nonzero_direction | (initial_info != 0) | no_breakpoints
    )
    nleft = nbreak
    iteration = jnp.asarray(0, dtype=jnp.int32)
    tj = jnp.asarray(0.0, dtype=jnp.float64)
    done = skip_loop
    info = initial_info

    for _ in range(n):
        active_loop = has_gradient & (~done) & (info == 0)
        tj0 = tj
        use_initial = iteration == 0
        heap_t = t
        heap_iorder = iorder
        replace_minimum = active_loop & (iteration == 1) & (ibkmin != nbreak)
        heap_t = heap_t.at[ibkmin].set(
            jnp.where(replace_minimum, heap_t[nbreak], heap_t[ibkmin])
        )
        heap_iorder = heap_iorder.at[ibkmin].set(
            jnp.where(replace_minimum, heap_iorder[nbreak], heap_iorder[ibkmin])
        )
        heap = lbfgsb_hpsolb(jnp.maximum(nleft, 0), heap_t, heap_iorder, iteration - 1)
        use_heap = active_loop & (~use_initial)
        t = jnp.where(use_heap, heap.t, t)
        iorder = jnp.where(use_heap, heap.iorder, iorder)
        safe_nleft = jnp.minimum(jnp.maximum(nleft, 0), n - 1)
        tj_candidate = jnp.where(use_initial, bkmin, t[safe_nleft])
        ibp = jnp.where(use_initial, iorder[ibkmin], iorder[safe_nleft])
        tj = jnp.where(active_loop, tj_candidate, tj)
        dt = tj_candidate - tj0
        interval_minimum = active_loop & (dtm < dt)
        fix_breakpoint = active_loop & (~interval_minimum)

        next_tsum = tsum + jnp.where(fix_breakpoint, dt, 0.0)
        next_nleft = nleft - jnp.where(fix_breakpoint, 1, 0)
        next_iteration = iteration + jnp.where(fix_breakpoint, 1, 0)
        dibp = d[ibp]
        d = d.at[ibp].set(jnp.where(fix_breakpoint, 0.0, d[ibp]))
        upper_hit = dibp > 0.0
        zibp = jnp.where(upper_hit, u[ibp] - x[ibp], l[ibp] - x[ibp])
        xcp = xcp.at[ibp].set(
            jnp.where(
                fix_breakpoint,
                jnp.where(upper_hit, u[ibp], l[ibp]),
                xcp[ibp],
            )
        )
        iwhere = iwhere.at[ibp].set(
            jnp.where(fix_breakpoint, jnp.where(upper_hit, 2, 1), iwhere[ibp])
        )
        all_fixed = fix_breakpoint & (next_nleft == -1) & (nbreak == n)

        next_nseg = nseg
        dibp2 = dibp * dibp
        next_f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp
        next_f2 = f2 - theta * dibp2
        next_c = c
        next_p = p
        next_v = v
        next_wbp = wbp
        bmv_info = jnp.asarray(0, dtype=jnp.int32)

        col_update = fix_breakpoint & (~all_fixed) & (col > 0)
        for j in range(2 * m):
            next_c = next_c.at[j].set(
                jnp.where(col_update & (j < col2), next_c[j] + dt * p[j], next_c[j])
            )
        pointr = head
        for j in range(m):
            active_col = col_update & (j < col)
            next_wbp = next_wbp.at[j].set(
                jnp.where(active_col, wy[pointr, ibp], next_wbp[j])
            )
            next_wbp = next_wbp.at[col + j].set(
                jnp.where(active_col, theta * ws[pointr, ibp], next_wbp[col + j])
            )
            pointr = (pointr + 1) % m
        bmv_break = lbfgsb_bmv(sy, wt, col, next_wbp)
        next_v = jnp.where(col_update, bmv_break.p, next_v)
        bmv_info = jnp.where(col_update, bmv_break.info, 0)
        wmc = jnp.sum(jnp.where(active_col2, next_c * next_v, 0.0))
        wmp = jnp.sum(jnp.where(active_col2, p * next_v, 0.0))
        wmw = jnp.sum(jnp.where(active_col2, next_wbp * next_v, 0.0))
        for j in range(2 * m):
            next_p = next_p.at[j].set(
                jnp.where(
                    col_update & (j < col2), next_p[j] - dibp * next_wbp[j], next_p[j]
                )
            )
        next_f1 = jnp.where(col_update, next_f1 + dibp * wmc, next_f1)
        next_f2 = jnp.where(
            col_update,
            next_f2 + dibp * 2.0 * wmp - dibp2 * wmw,
            next_f2,
        )
        next_f2 = jnp.maximum(jnp.finfo(jnp.float64).eps * f2_org, next_f2)
        # jax-where-division-ok: next_f2 is clamped positive immediately above.
        next_dtm = jnp.where(
            next_nleft >= 0,
            -next_f1 / next_f2,
            jnp.where(bnded, 0.0, -next_f1 / next_f2),
        )
        terminal_after_update = fix_breakpoint & (all_fixed | (next_nleft < 0))
        dtm = jnp.where(
            interval_minimum,
            dtm,
            jnp.where(all_fixed, dt, jnp.where(fix_breakpoint, next_dtm, dtm)),
        )
        f1 = jnp.where(fix_breakpoint & (~all_fixed), next_f1, f1)
        f2 = jnp.where(fix_breakpoint & (~all_fixed), next_f2, f2)
        c = jnp.where(col_update, next_c, c)
        p = jnp.where(col_update, next_p, p)
        v = jnp.where(col_update, next_v, v)
        wbp = jnp.where(col_update, next_wbp, wbp)
        nseg = jnp.where(fix_breakpoint, next_nseg, nseg)
        tsum = jnp.where(fix_breakpoint, next_tsum, tsum)
        nleft = jnp.where(fix_breakpoint, next_nleft, nleft)
        iteration = jnp.where(fix_breakpoint, next_iteration, iteration)
        info = jnp.where(col_update & (bmv_info != 0), bmv_info, info)
        done = done | interval_minimum | terminal_after_update | (info != 0)

    line_search_active = has_gradient & (~no_nonzero_direction) & (info == 0)
    dtm = jnp.maximum(dtm, 0.0)
    tsum = jnp.where(line_search_active, tsum + dtm, tsum)
    xcp = jnp.where(line_search_active, xcp + tsum * d, xcp)
    for j in range(2 * m):
        c = c.at[j].set(
            jnp.where(
                line_search_active & (col > 0) & (j < col2), c[j] + dtm * p[j], c[j]
            )
        )

    return LbfgsbCauchyResult(
        iorder=iorder,
        iwhere=iwhere,
        t=t,
        d=d,
        xcp=xcp,
        p=p,
        c=c,
        wbp=wbp,
        v=v,
        nseg=jnp.where(
            (~has_gradient) | no_nonzero_direction | (initial_info != 0), 0, nseg
        ),
        info=info,
    )


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
        # jax-where-division-ok: lower_limited requires dk < 0 in this branch.
        next_temp1 = jnp.where(
            active_sub & lower_limited & (lower_step >= 0.0),
            0.0,
            jnp.where(
                active_sub & lower_limited & (dk * alpha < lower_step),
                lower_step / dk,
                next_temp1,
            ),
        )
        # jax-where-division-ok: upper_limited requires dk > 0 in this branch.
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
        next_dnorm = _lbfgsb_dnrm2(d)
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
            # jax-where-division-ok: lower_active requires direction_i < 0.
            next_stpmx = jnp.where(
                lower_active & (lower_step >= 0.0),
                0.0,
                jnp.where(
                    lower_active & (direction_i * next_stpmx < lower_step),
                    lower_step / direction_i,
                    next_stpmx,
                ),
            )
            # jax-where-division-ok: upper_active requires direction_i > 0.
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
        # jax-where-division-ok: initial direction norm is validated upstream.
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

    next_gd = _lbfgsb_ddot(g, d)
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
