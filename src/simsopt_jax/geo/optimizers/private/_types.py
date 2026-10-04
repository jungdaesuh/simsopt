"""Frozen pytree result types for private BFGS / L-BFGS solvers."""

from __future__ import annotations

import jax

from simsopt_jax.pytree import pytree_dataclass

# Private L-BFGS terminal status code emitted when f(x), x, or ∇f(x) is
# non-finite at solver entry, during an attempted step, or at final
# re-evaluation. This is repo-local; SciPy's low-level L-BFGS-B warnflag
# contract remains 0/1/2.
LBFGS_STATUS_NONFINITE = 6
LBFGS_STATUS_CALLBACK_STOP = 99
BFGS_STATUS_CALLBACK_STOP = 99


@pytree_dataclass(
    data=(
        "converged",
        "failed",
        "k",
        "nfev",
        "ngev",
        "nhev",
        "x_k",
        "f_k",
        "g_k",
        "H_k",
        "old_old_fval",
        "status",
        "line_search_status",
    ),
)
class _BFGSResults:
    converged: bool | jax.Array
    failed: bool | jax.Array
    k: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    nhev: int | jax.Array
    x_k: jax.Array
    f_k: jax.Array
    g_k: jax.Array
    H_k: jax.Array
    old_old_fval: jax.Array
    status: int | jax.Array
    line_search_status: int | jax.Array


@pytree_dataclass(
    data=(
        "done",
        "failed",
        "j",
        "a_lo",
        "phi_lo",
        "dphi_lo",
        "g_lo",
        "a_hi",
        "phi_hi",
        "dphi_hi",
        "g_hi",
        "has_rec",
        "a_rec",
        "phi_rec",
        "dphi_rec",
        "g_rec",
        "a_star",
        "phi_star",
        "dphi_star",
        "g_star",
        "best_a",
        "best_phi",
        "best_dphi",
        "best_g",
        "nfev",
        "ngev",
    ),
)
class _ZoomState:
    done: bool | jax.Array
    failed: bool | jax.Array
    j: int | jax.Array
    a_lo: float | jax.Array
    phi_lo: float | jax.Array
    dphi_lo: float | jax.Array
    g_lo: jax.Array
    a_hi: float | jax.Array
    phi_hi: float | jax.Array
    dphi_hi: float | jax.Array
    g_hi: jax.Array
    has_rec: bool | jax.Array
    a_rec: float | jax.Array
    phi_rec: float | jax.Array
    dphi_rec: float | jax.Array
    g_rec: jax.Array
    a_star: float | jax.Array
    phi_star: float | jax.Array
    dphi_star: float | jax.Array
    g_star: jax.Array
    best_a: float | jax.Array
    best_phi: float | jax.Array
    best_dphi: float | jax.Array
    best_g: jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array


@pytree_dataclass(
    data=(
        "done",
        "failed",
        "i",
        "a_i2",
        "phi_i2",
        "dphi_i2",
        "g_i2",
        "a_i1",
        "phi_i1",
        "dphi_i1",
        "g_i1",
        "best_a",
        "best_phi",
        "best_dphi",
        "best_g",
        "nfev",
        "ngev",
        "a_star",
        "phi_star",
        "dphi_star",
        "g_star",
    ),
)
class _LineSearchState:
    done: jax.Array
    failed: jax.Array
    i: int | jax.Array
    a_i2: float | jax.Array
    phi_i2: float | jax.Array
    dphi_i2: float | jax.Array
    g_i2: jax.Array
    a_i1: float | jax.Array
    phi_i1: float | jax.Array
    dphi_i1: float | jax.Array
    g_i1: jax.Array
    best_a: float | jax.Array
    best_phi: float | jax.Array
    best_dphi: float | jax.Array
    best_g: jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    a_star: float | jax.Array
    phi_star: jax.Array
    dphi_star: jax.Array
    g_star: jax.Array


@pytree_dataclass(
    data=(
        "failed",
        "nit",
        "nfev",
        "ngev",
        "k",
        "a_k",
        "f_k",
        "g_k",
        "status",
    ),
)
class _LineSearchResults:
    failed: bool | jax.Array
    nit: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    k: int | jax.Array
    a_k: float | jax.Array
    f_k: jax.Array
    g_k: jax.Array
    status: bool | jax.Array


@pytree_dataclass(
    data=(
        "converged",
        "failed",
        "k",
        "nfev",
        "ngev",
        "x_k",
        "f_k",
        "g_k",
        "s_history",
        "y_history",
        "rho_history",
        "gamma",
        "status",
        "ls_status",
        "evaluated_nonfinite_count",
        "all_accepted_states_finite",
        "invalid_step_record",
        "optimizer_state_trace",
        "hess_inv_s",
        "hess_inv_y",
        "hess_inv_n_corrs",
        "task",
    ),
)
class _LBFGSResults:
    converged: jax.Array
    failed: jax.Array
    k: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    x_k: jax.Array
    f_k: jax.Array
    g_k: jax.Array
    s_history: jax.Array
    y_history: jax.Array
    rho_history: jax.Array
    gamma: float | jax.Array
    status: int | jax.Array
    ls_status: int | jax.Array
    evaluated_nonfinite_count: jax.Array
    all_accepted_states_finite: jax.Array
    invalid_step_record: _LBFGSInvalidStepRecord
    optimizer_state_trace: tuple[dict[str, object], ...] = ()
    hess_inv_s: jax.Array | None = None
    hess_inv_y: jax.Array | None = None
    hess_inv_n_corrs: int | jax.Array | None = None
    task: jax.Array | None = None


@pytree_dataclass(
    data=(
        "recorded",
        "iteration",
        "step_scale",
        "line_search_failed",
        "nonfinite_step",
        "ls_status",
        "failure_reason",
        "curvature_margin_measured",
        "curvature_margin",
    ),
)
class _LBFGSInvalidStepRecord:
    """The one rejected-step record an L-BFGS-B solve can publish.

    ``setulb`` reports ABNORMAL only when a line search fails with no correction
    pairs left to refresh from, which ends the solve.  There is therefore at most
    one record per solve -- ``recorded`` says whether it exists -- and no history
    to ring, so every field is a scalar rather than a slot in a buffer.

    Every field is read from the terminal ``dsave``/``isave`` workspace.  Fields
    the workspace does not survive to answer are absent rather than defaulted:
    ``lnsrlb`` keeps a single ``stp`` slot that ``dcsrch`` overwrites on each
    entry, so the requested initial step and the first tested alpha are gone by
    the time the search is abandoned, and the alpha the last trial was evaluated
    at is gone too -- which is why an Armijo residual cannot be published while a
    curvature residual, which needs no alpha, can.

    ``step_scale`` and ``curvature_margin`` belong to two *different* alphas and
    must not be read as one observation.  ``step_scale`` is ``dsave[13]``, the
    step ``dcsrch`` proposed next and that no trial was ever evaluated at;
    ``curvature_margin`` is built from ``dsave[10]``, the directional derivative
    at the trial the driver last actually evaluated.

    ``curvature_margin`` is published only when ``curvature_margin_measured``:
    ``lnsrlb`` rejects a non-descent direction before calling ``dcsrch`` at all
    and returns the save area untouched, so on that path ``dsave[16]`` holds a
    previous line search's ``ginit`` -- zero on the first -- and the difference
    is not the residual of any curvature test.

    That path is reachable, not theoretical.  This lane is always unbounded, so
    a solve that terminates ABNORMAL has no live correction pair and searches
    along ``-g/theta``, making ``g . d = -|g|^2/theta`` non-positive in exact
    arithmetic.  XLA nonetheless flushes subnormal float64 products to ``-0.0``,
    so a gradient small enough that ``|g|^2`` is subnormal (``|g|`` below about
    1.5e-154) yields ``gd == -0.0``, which satisfies ``gd >= 0`` and reports
    NOT_DESCENT.
    """

    recorded: jax.Array
    iteration: jax.Array
    step_scale: jax.Array
    line_search_failed: jax.Array
    nonfinite_step: jax.Array
    ls_status: jax.Array
    failure_reason: jax.Array
    curvature_margin_measured: jax.Array
    curvature_margin: jax.Array
