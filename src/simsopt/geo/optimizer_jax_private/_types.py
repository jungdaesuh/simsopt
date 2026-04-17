"""NamedTuple result types for private BFGS / L-BFGS solvers."""

from __future__ import annotations

from typing import NamedTuple

import jax


# Private L-BFGS terminal status code emitted when f(x) or ∇f(x) is non-finite
# at solver entry or at the final re-evaluation. The solver encodes this as a
# JAX literal; the host converter maps it to the "Non-finite objective or
# gradient encountered during iteration." message. Kept here as the single
# contract between solver-site emission and host-side decoding.
LBFGS_STATUS_NONFINITE = 6


class _BFGSResults(NamedTuple):
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


class _ZoomState(NamedTuple):
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


class _LineSearchState(NamedTuple):
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


class _LineSearchResults(NamedTuple):
    failed: bool | jax.Array
    nit: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    k: int | jax.Array
    a_k: float | jax.Array
    f_k: jax.Array
    g_k: jax.Array
    status: bool | jax.Array


class _LBFGSResults(NamedTuple):
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
    invalid_step_log: "_LBFGSInvalidStepLog"


class _LBFGSInvalidStepLog(NamedTuple):
    count: int | jax.Array
    write_index: int | jax.Array
    iteration: jax.Array
    step_scale: jax.Array
    line_search_failed: jax.Array
    nonfinite_step: jax.Array
    stalled_step: jax.Array
    valid_curvature: jax.Array
    trial_converged: jax.Array
    ls_status: jax.Array
