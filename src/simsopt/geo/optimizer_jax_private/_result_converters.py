"""Convert private solver state to SciPy OptimizeResult."""

from __future__ import annotations

import warnings

import jax
import numpy as np

import jax.numpy as jnp
from scipy.optimize import OptimizeResult


def _as_host_numpy(value, *, dtype=None):
    array = np.asarray(jax.device_get(value))
    if dtype is None:
        return array
    return np.asarray(array, dtype=dtype)


def _as_host_scalar(value, *, dtype=None):
    return _as_host_numpy(value, dtype=dtype).item()


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
    4: "Insufficient progress (ftol).",
    5: "Line search failed.",
}

_INVALID_STATE_MESSAGE = "Optimization failed with non-finite objective or gradient."


def _status_message(status, invalid_state, messages):
    if invalid_state:
        return _INVALID_STATE_MESSAGE
    return messages.get(status, f"Optimization failed with status {status}.")


def _status_message_bfgs(status, invalid_state):
    return _status_message(status, invalid_state, _BFGS_STATUS_MESSAGES)


def _status_message_lbfgs(status, invalid_state):
    return _status_message(status, invalid_state, _LBFGS_STATUS_MESSAGES)


def _private_bfgs_result_to_optimize_result(state, *, total_nit=None):
    invalid_state = _is_invalid_state(state.f_k, state.g_k)
    status = int(_as_host_scalar(state.status))
    nit = int(_as_host_scalar(state.k if total_nit is None else total_nit))
    return OptimizeResult(
        x=_as_host_numpy(state.x_k),
        fun=float(_as_host_scalar(state.f_k)),
        jac=_as_host_numpy(state.g_k),
        nit=nit,
        nfev=int(_as_host_scalar(state.nfev)),
        njev=int(_as_host_scalar(state.ngev)),
        nhev=int(_as_host_scalar(state.nhev)),
        success=bool(_as_host_scalar(state.converged)) and not invalid_state,
        status=status,
        message=_status_message_bfgs(status, invalid_state),
        hess_inv=_as_host_numpy(state.H_k),
        line_search_status=int(_as_host_scalar(state.line_search_status)),
    )


def _private_lbfgs_result_to_optimize_result(state):
    invalid_state = _is_invalid_state(state.f_k, state.g_k)
    status = int(_as_host_scalar(state.status))
    return OptimizeResult(
        x=_as_host_numpy(state.x_k),
        fun=float(_as_host_scalar(state.f_k)),
        jac=_as_host_numpy(state.g_k),
        nit=int(_as_host_scalar(state.k)),
        nfev=int(_as_host_scalar(state.nfev)),
        njev=int(_as_host_scalar(state.ngev)),
        success=bool(_as_host_scalar(state.converged)) and not invalid_state,
        status=status,
        message=_status_message_lbfgs(status, invalid_state),
        ls_status=int(_as_host_scalar(state.ls_status)),
    )


def _coerce_dense_hess_inv(hess_inv, n, dtype):
    if hess_inv is None:
        warnings.warn(
            "Hybrid BFGS continuation received no dense hess_inv; falling back to "
            "identity warm start.",
            RuntimeWarning,
            stacklevel=3,
        )
        return jnp.eye(n, dtype=dtype)
    try:
        dense = _as_host_numpy(hess_inv)
    except Exception:
        warnings.warn(
            "Hybrid BFGS continuation could not densify hess_inv; falling back to "
            "identity warm start.",
            RuntimeWarning,
            stacklevel=3,
        )
        return jnp.eye(n, dtype=dtype)
    if dense.ndim != 2 or dense.shape != (n, n):
        warnings.warn(
            "Hybrid BFGS continuation received mismatched hess_inv shape; "
            "falling back to identity warm start.",
            RuntimeWarning,
            stacklevel=3,
        )
        return jnp.eye(n, dtype=dtype)
    return jnp.asarray(dense, dtype=dtype)


def _scipy_result_is_continuable(result):
    return (
        np.isfinite(getattr(result, "fun", np.nan))
        and np.all(np.isfinite(_as_host_numpy(result.x)))
        and np.all(np.isfinite(_as_host_numpy(result.jac)))
    )
