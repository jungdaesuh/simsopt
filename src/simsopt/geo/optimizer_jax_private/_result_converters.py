"""Convert private solver state to SciPy OptimizeResult."""

from __future__ import annotations

import numpy as np

from scipy.optimize import OptimizeResult
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

from ..._core.jax_host_boundary import (
    host_array as _as_host_numpy,
    host_bool as _host_bool,
    host_float as _host_float,
    host_int as _host_int,
)
from ..optimizer_host_lbfgs import line_search_failure_reason_from_code
from ._types import LBFGS_STATUS_NONFINITE


def _is_invalid_state(f, g):
    """Check whether objective or gradient contains non-finite values."""
    return not bool(
        np.isfinite(_as_host_numpy(f)) and np.all(np.isfinite(_as_host_numpy(g)))
    )


_BFGS_STATUS_MESSAGES = {
    0: "Optimization terminated successfully.",
    1: "Maximum number of iterations reached.",
    2: "Insufficient progress.",
    3: "Line search zoom failed.",
    5: "Line search reached its iteration limit.",
    -1: "Optimization failed.",
}

_LBFGS_STATUS_MESSAGES = {
    0: "Optimization terminated successfully.",
    1: "Maximum number of iterations reached.",
    2: "Maximum number of function evaluations reached.",
    3: "Maximum number of gradient evaluations reached.",
    4: "Optimization terminated successfully (ftol).",
    5: "Line search failed.",
    LBFGS_STATUS_NONFINITE: (
        "Non-finite objective or gradient encountered during iteration."
    ),
}

_LBFGS_SUCCESS_STATUSES = frozenset({0, 4})

_INVALID_STATE_MESSAGE = "Optimization failed with non-finite objective or gradient."


def _status_message(status, invalid_state, messages):
    if invalid_state:
        return _INVALID_STATE_MESSAGE
    return messages.get(status, f"Optimization failed with status {status}.")


def _status_message_bfgs(status, invalid_state):
    return _status_message(status, invalid_state, _BFGS_STATUS_MESSAGES)


def _status_message_lbfgs(status, invalid_state):
    return _status_message(status, invalid_state, _LBFGS_STATUS_MESSAGES)


def _lbfgsb_hess_inv_from_state(state):
    hess_inv_s = getattr(state, "hess_inv_s", None)
    hess_inv_y = getattr(state, "hess_inv_y", None)
    hess_inv_n_corrs = getattr(state, "hess_inv_n_corrs", None)
    if hess_inv_s is None or hess_inv_y is None or hess_inv_n_corrs is None:
        return None
    n_corrs = _host_int(hess_inv_n_corrs)
    return LbfgsInvHessProduct(
        _as_host_numpy(hess_inv_s)[:n_corrs],
        _as_host_numpy(hess_inv_y)[:n_corrs],
    )


def _lbfgs_success(status, invalid_state, state):
    if getattr(state, "task", None) is not None:
        return status == 0
    return (status in _LBFGS_SUCCESS_STATUSES) and not invalid_state


def _lbfgs_message(status, invalid_state, state):
    task = getattr(state, "task", None)
    if task is not None:
        from . import _lbfgsb_scipy as lbfgsb

        return lbfgsb.lbfgsb_task_message(task)
    return _status_message_lbfgs(status, invalid_state)


def _private_lbfgs_invalid_step_log_to_host(invalid_step_log):
    count = _host_int(invalid_step_log.count)
    if count <= 0:
        return []
    iterations = _as_host_numpy(invalid_step_log.iteration)
    capacity = int(iterations.shape[0])
    if capacity <= 0:
        return []
    write_index = _host_int(invalid_step_log.write_index)
    start_index = (write_index - count) % capacity
    step_scales = _as_host_numpy(invalid_step_log.step_scale)
    line_search_failed = _as_host_numpy(invalid_step_log.line_search_failed)
    nonfinite_step = _as_host_numpy(invalid_step_log.nonfinite_step)
    stalled_step = _as_host_numpy(invalid_step_log.stalled_step)
    valid_curvature = _as_host_numpy(invalid_step_log.valid_curvature)
    trial_converged = _as_host_numpy(invalid_step_log.trial_converged)
    line_search_statuses = _as_host_numpy(invalid_step_log.ls_status)
    requested_initial_steps = _as_host_numpy(invalid_step_log.requested_initial_step)
    first_tested_alphas = _as_host_numpy(invalid_step_log.first_tested_alpha)
    best_finite_alphas = _as_host_numpy(invalid_step_log.best_finite_alpha)
    returned_alphas = _as_host_numpy(invalid_step_log.returned_alpha)
    failure_reasons = _as_host_numpy(invalid_step_log.failure_reason)
    armijo_margins = _as_host_numpy(invalid_step_log.armijo_margin)
    curvature_margins = _as_host_numpy(invalid_step_log.curvature_margin)
    events = []
    for offset in range(count):
        index = (start_index + offset) % capacity
        events.append(
            {
                "iteration": int(iterations[index]),
                "step_scale": float(step_scales[index]),
                "line_search_failed": bool(line_search_failed[index]),
                "nonfinite_step": bool(nonfinite_step[index]),
                "stalled_step": bool(stalled_step[index]),
                "valid_curvature": bool(valid_curvature[index]),
                "trial_converged": bool(trial_converged[index]),
                "ls_status": int(line_search_statuses[index]),
                "requested_initial_step": float(requested_initial_steps[index]),
                "first_tested_alpha": float(first_tested_alphas[index]),
                "best_finite_alpha": float(best_finite_alphas[index]),
                "returned_alpha": float(returned_alphas[index]),
                "failure_reason": line_search_failure_reason_from_code(
                    failure_reasons[index]
                ),
                "armijo_margin": float(armijo_margins[index]),
                "curvature_margin": float(curvature_margins[index]),
            }
        )
    return events


def _private_bfgs_result_to_optimize_result(state, *, total_nit=None):
    line_search_status = _host_int(state.line_search_status)
    invalid_state = _is_invalid_state(state.f_k, state.g_k) or line_search_status < 0
    status = _host_int(state.status)
    nit = _host_int(state.k if total_nit is None else total_nit)
    return OptimizeResult(
        x=_as_host_numpy(state.x_k),
        fun=_host_float(state.f_k),
        jac=_as_host_numpy(state.g_k),
        nit=nit,
        nfev=_host_int(state.nfev),
        njev=_host_int(state.ngev),
        nhev=_host_int(state.nhev),
        success=_host_bool(state.converged) and not invalid_state,
        status=status,
        message=_status_message_bfgs(status, invalid_state),
        hess_inv=_as_host_numpy(state.H_k),
        line_search_status=line_search_status,
    )


def _private_lbfgs_result_to_optimize_result(state):
    invalid_state = _is_invalid_state(state.f_k, state.g_k)
    status = _host_int(state.status)
    ls_status = _host_int(state.ls_status)
    invalid_step_log = _private_lbfgs_invalid_step_log_to_host(state.invalid_step_log)
    optimizer_state_trace = tuple(state.optimizer_state_trace)
    result_fields = {
        "x": _as_host_numpy(state.x_k),
        "fun": _host_float(state.f_k),
        "jac": _as_host_numpy(state.g_k),
        "nit": _host_int(state.k),
        "nfev": _host_int(state.nfev),
        "njev": _host_int(state.ngev),
        "success": _lbfgs_success(status, invalid_state, state),
        "status": status,
        "message": _lbfgs_message(status, invalid_state, state),
        "ls_status": ls_status,
        "line_search_final_status": ls_status,
        "maxiter_hit": status == 1,
        "rejected_step_count": len(invalid_step_log),
        "invalid_step_log": invalid_step_log,
        "optimizer_state_trace": optimizer_state_trace,
    }
    hess_inv = _lbfgsb_hess_inv_from_state(state)
    if hess_inv is not None:
        result_fields["hess_inv"] = hess_inv
    return OptimizeResult(**result_fields)


def _scipy_result_is_continuable(result):
    return (
        np.isfinite(getattr(result, "fun", np.nan))
        and np.all(np.isfinite(_as_host_numpy(result.x)))
        and np.all(np.isfinite(_as_host_numpy(result.jac)))
    )
