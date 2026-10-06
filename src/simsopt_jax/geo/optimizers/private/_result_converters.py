"""Convert private solver state to SciPy OptimizeResult."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

from simsopt_jax.geo.optimizer_host_lbfgs import line_search_failure_reason_from_code
from simsopt_jax.runtime.host_boundary import (
    host_array as _as_host_numpy,
)
from simsopt_jax.runtime.host_boundary import (
    host_bool as _host_bool,
)
from simsopt_jax.runtime.host_boundary import (
    host_float as _host_float,
)
from simsopt_jax.runtime.host_boundary import (
    host_int as _host_int,
)
from simsopt_jax.runtime.host_boundary import host_transfer_phase

from . import _lbfgsb_scipy as lbfgsb
from ._types import (
    BFGS_STATUS_CALLBACK_STOP,
    LBFGS_STATUS_CALLBACK_STOP,
    LBFGS_STATUS_NONFINITE,
)


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
    BFGS_STATUS_CALLBACK_STOP: "`callback` raised `StopIteration`.",
}

_LBFGS_STATUS_MESSAGES = {
    0: "Optimization terminated successfully.",
    1: "Maximum number of iterations reached.",
    2: "Maximum number of function evaluations reached.",
    3: "Maximum number of gradient evaluations reached.",
    4: "Optimization terminated successfully (ftol).",
    5: "Line search failed.",
    LBFGS_STATUS_NONFINITE: (
        "Non-finite objective, iterate, or gradient encountered during iteration."
    ),
    LBFGS_STATUS_CALLBACK_STOP: "`callback` raised `StopIteration`.",
}

_LBFGS_SUCCESS_STATUSES = frozenset({0, 4})

_INVALID_STATE_MESSAGE = (
    "Optimization failed with non-finite objective, iterate, or gradient."
)


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
    if status in {LBFGS_STATUS_NONFINITE, LBFGS_STATUS_CALLBACK_STOP}:
        return _status_message_lbfgs(status, False)
    task = getattr(state, "task", None)
    if task is not None:
        return lbfgsb.lbfgsb_task_message(task)
    return _status_message_lbfgs(status, invalid_state)


def _private_lbfgs_invalid_step_record_to_host(invalid_step_record):
    """Publish the L-BFGS-B rejected-step record as a zero- or one-event list.

    An L-BFGS-B solve stops at its first abnormal line-search termination, so the
    record is a single terminal observation rather than a buffer of trials.
    """
    if not _host_bool(invalid_step_record.recorded):
        return []
    event = {
        "iteration": _host_int(invalid_step_record.iteration),
        "step_scale": _host_float(invalid_step_record.step_scale),
        "line_search_failed": _host_bool(invalid_step_record.line_search_failed),
        "nonfinite_step": _host_bool(invalid_step_record.nonfinite_step),
        "ls_status": _host_int(invalid_step_record.ls_status),
        "failure_reason": line_search_failure_reason_from_code(
            _host_int(invalid_step_record.failure_reason)
        ),
    }
    # A non-descent direction is rejected before dcsrch runs, so no curvature
    # test was ever evaluated and the key is absent rather than carrying a
    # number derived from a stale save area.
    if _host_bool(invalid_step_record.curvature_margin_measured):
        event["curvature_margin"] = _host_float(invalid_step_record.curvature_margin)
    return [event]


def _private_bfgs_result_to_optimize_result(state, *, total_nit=None):
    with host_transfer_phase("final_result"):
        line_search_status = _host_int(state.line_search_status)
        invalid_state = (
            _is_invalid_state(state.f_k, state.g_k) or line_search_status < 0
        )
        status = _host_int(state.status)
        nit = _host_int(state.k if total_nit is None else total_nit)
        return OptimizeResult(
            x=_as_host_numpy(state.x_k),
            x_device=state.x_k,
            fun=_host_float(state.f_k),
            jac=_as_host_numpy(state.g_k),
            jac_device=state.g_k,
            nit=nit,
            nfev=_host_int(state.nfev),
            njev=_host_int(state.ngev),
            nhev=_host_int(state.nhev),
            success=(
                _host_bool(state.converged)
                and not invalid_state
                and status != BFGS_STATUS_CALLBACK_STOP
            ),
            failed=_host_bool(state.failed),
            status=status,
            message=_status_message_bfgs(status, invalid_state),
            hess_inv=_as_host_numpy(state.H_k),
            line_search_status=line_search_status,
        )


def _private_lbfgs_result_to_optimize_result(state):
    with host_transfer_phase("final_result"):
        invalid_state = _is_invalid_state(state.f_k, state.g_k)
        status = _host_int(state.status)
        ls_status = _host_int(state.ls_status)
        invalid_step_log = _private_lbfgs_invalid_step_record_to_host(
            state.invalid_step_record
        )
        optimizer_state_trace = tuple(state.optimizer_state_trace)
        result_fields = {
            "x": _as_host_numpy(state.x_k),
            "x_device": state.x_k,
            "fun": _host_float(state.f_k),
            "jac": _as_host_numpy(state.g_k),
            "jac_device": state.g_k,
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
