"""Reference-lane optimizer wrappers and SciPy host adapters.

SciPy's public ``minimize()`` contract is host-array based: ``x0`` is an
``ndarray`` and, when ``jac=True``, the objective callable returns
``(float, array_like)``. This module is therefore the intentional host NumPy
boundary for SciPy-controlled lanes. CPU/reference execution enters through
``reference_*`` helpers; the explicit target SciPy-control lane enters through
``target_scipy_minimize_value_and_grad`` from ``optimizer_jax.target_minimize``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import jax
import numpy as np

from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from simsopt_jax.geo.optimizer_host_lbfgs import (
    LBFGS_STATUS_NONFINITE,
    host_invalid_step_log_to_list,
    lbfgs_status_is_success,
    lbfgs_status_message,
    minimize_lbfgs_host_core,
)
from ._shared import (
    _optimizer_dtype,
    _optimizer_flat_vector,
    _prepare_optimizer_callable_inputs,
)
from ._evaluation_provider import (
    TargetScipyDeviceEvaluation as _TargetScipyDeviceEvaluation,
    TargetScipyDeviceIdentity as _TargetScipyDeviceIdentity,
    TargetScipyEvaluationClass as _TargetScipyEvaluationClass,
    TargetScipyEvaluationLifecycle as _TargetScipyEvaluationLifecycle,
    TargetScipyEvaluationOutcome as _TargetScipyEvaluationOutcome,
    TargetScipyEvaluationProvider as _TargetScipyEvaluationProvider,
    TargetScipyHostEvaluation as _TargetScipyHostEvaluation,
    _require_host_resident_target_scipy_evaluation,
    inspect_target_scipy_device_packet_layout as _inspect_target_scipy_device_packet_layout,
)
from . import optimizer as _optimizer

__all__ = [
    "reference_least_squares",
    "reference_minimize",
    "_scipy_dispatch",
    "_scipy_minimize",
    "_scipy_minimize_value_and_grad",
    "target_scipy_minimize_value_and_grad",
]


def _strip_internal_options(options, method):
    if not options:
        return {}
    internal = {
        "line_search_maxiter",
        "callback",
        "progress_callback",
        "failure_callback",
        "record_scipy_callback_trace",
    }
    if method == "bfgs":
        internal |= {"maxcor", "ftol", "maxfun", "maxgrad", "maxls"}
    elif method == "lbfgs":
        internal |= {"maxgrad"}
    return {key: value for key, value in options.items() if key not in internal}


def _scipy_callback_contract(callback):
    return None if callback is None else "callable"


@dataclass(frozen=True)
class _ScipyObjectiveEvaluation:
    """Host snapshot of one objective evaluation already returned to SciPy."""

    objective_evaluation_index: int
    decision_vector: np.ndarray
    fun: np.floating
    gradient: np.ndarray


def _latest_exact_decision_vector_index(
    accepted_decision_vector: np.ndarray,
    candidate_decision_vectors: Iterable[np.ndarray],
) -> int | None:
    """Return the last candidate index exactly matching SciPy's accepted x."""
    exact_index = None
    for index, candidate_decision_vector in enumerate(candidate_decision_vectors):
        if _decision_vectors_have_identical_bytes(
            accepted_decision_vector,
            candidate_decision_vector,
        ):
            exact_index = index
    return exact_index


def _decision_vectors_have_identical_bytes(
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.dtype.str == right_array.dtype.str
        and left_array.shape == right_array.shape
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


@contextmanager
def _target_scipy_host_extension_scope() -> Iterator[None]:
    """Forbid implicit transfers while provider host extensions execute."""
    with jax.transfer_guard_host_to_device("disallow"):
        with jax.transfer_guard_device_to_host("disallow"):
            yield


@dataclass
class _ScipyObjectiveEvaluationMemo:
    """Retain one callback window of exact objective evaluations."""

    pending_evaluations: list[_ScipyObjectiveEvaluation] = field(default_factory=list)

    def record(self, evaluation: _ScipyObjectiveEvaluation) -> None:
        self.pending_evaluations.append(evaluation)

    def resolve_accepted(
        self,
        accepted_decision_vector: np.ndarray,
    ) -> _ScipyObjectiveEvaluation:
        accepted_evaluation_index = _latest_exact_decision_vector_index(
            accepted_decision_vector,
            (evaluation.decision_vector for evaluation in self.pending_evaluations),
        )
        if accepted_evaluation_index is None:
            raise RuntimeError(
                "SciPy accepted iterate has no exact objective evaluation; "
                "progress metrics cannot be attributed without reevaluation."
            )
        accepted_evaluation = self.pending_evaluations[accepted_evaluation_index]
        self.pending_evaluations.clear()
        return accepted_evaluation


@dataclass(frozen=True)
class _PendingTargetScipyTrial:
    """One unresolved trial and its exact SciPy decision vector."""

    decision_vector: np.ndarray
    lifecycle: _TargetScipyEvaluationLifecycle | None


@dataclass
class _TargetScipyEvaluationLifecycleController:
    """Resolve provider trials at SciPy's exact accepted callback boundary."""

    optimizer_evaluation_count: int = 0
    pending_trials: list[_PendingTargetScipyTrial] = field(default_factory=list)

    def evaluation_returned(
        self,
        decision_vector: np.ndarray,
        lifecycle: _TargetScipyEvaluationLifecycle | None,
    ) -> None:
        evaluation_class = (
            _TargetScipyEvaluationClass.INITIAL_OPTIMIZER_EVALUATION
            if self.optimizer_evaluation_count == 0
            else _TargetScipyEvaluationClass.OPTIMIZER_TRIAL
        )
        self.optimizer_evaluation_count += 1
        if lifecycle is not None:
            with _target_scipy_host_extension_scope():
                lifecycle.classify_target_scipy_evaluation(evaluation_class)
        if evaluation_class is _TargetScipyEvaluationClass.OPTIMIZER_TRIAL:
            self.pending_trials.append(
                _PendingTargetScipyTrial(
                    decision_vector=decision_vector.copy(),
                    lifecycle=lifecycle,
                )
            )

    def accepted_trial(self, decision_vector: np.ndarray) -> None:
        if not self.pending_trials:
            return
        accepted_trial_index = _latest_exact_decision_vector_index(
            decision_vector,
            (pending_trial.decision_vector for pending_trial in self.pending_trials),
        )
        if accepted_trial_index is None:
            raise RuntimeError(
                "SciPy accepted iterate has no exact typed provider trial evaluation."
            )
        with _target_scipy_host_extension_scope():
            for index, pending_trial in enumerate(self.pending_trials):
                if pending_trial.lifecycle is None:
                    continue
                outcome = (
                    _TargetScipyEvaluationOutcome.ACCEPTED
                    if index == accepted_trial_index
                    else _TargetScipyEvaluationOutcome.REJECTED
                )
                pending_trial.lifecycle.resolve_target_scipy_evaluation(outcome)
        self.pending_trials.clear()

    def optimizer_completed(self) -> None:
        with _target_scipy_host_extension_scope():
            for pending_trial in self.pending_trials:
                if pending_trial.lifecycle is not None:
                    pending_trial.lifecycle.resolve_target_scipy_evaluation(
                        _TargetScipyEvaluationOutcome.REJECTED
                    )
        self.pending_trials.clear()


def _scipy_scalar_value(value, *, dtype):
    scalar = _scipy_host_array(value, dtype=dtype)
    if scalar.shape != ():
        raise ValueError("SciPy objective callable must return scalar shape ().")
    return scalar[()]


def _scipy_host_array(value, *, dtype):
    with jax.transfer_guard_device_to_host("allow"):
        return np.asarray(value, dtype=np.dtype(dtype))


def _immutable_scipy_decision_snapshot(value, *, dtype) -> np.ndarray:
    snapshot = _scipy_host_array(value, dtype=dtype).copy()
    snapshot.flags.writeable = False
    return snapshot


def _isolated_scipy_decision_snapshot(value: np.ndarray) -> np.ndarray:
    snapshot = value.copy()
    snapshot.flags.writeable = False
    return snapshot


def _target_array_from_scipy_host(value, *, dtype):
    with jax.transfer_guard_host_to_device("allow"):
        return _optimizer_flat_vector(value, dtype=dtype)


def _scipy_initial_call_contract(x_np, value, gradient, *, dtype):
    return {
        "decision_vector": _scipy_host_array(x_np, dtype=dtype).copy(),
        "fun": _scipy_scalar_value(value, dtype=dtype),
        "gradient": _scipy_host_array(gradient, dtype=dtype).copy(),
    }


def _scipy_objective_trace_entry(
    evaluation: _ScipyObjectiveEvaluation,
) -> dict[str, object]:
    return {
        "objective_evaluation_index": evaluation.objective_evaluation_index,
        "decision_vector": evaluation.decision_vector.copy(),
        "fun": evaluation.fun,
        "gradient": evaluation.gradient.copy(),
    }


def _scipy_result_contract(
    result, *, semantic_method, scipy_method, scipy_opts, callback
):
    return {
        "semantic_method": str(semantic_method),
        "scipy_method": str(scipy_method),
        "scipy_options": dict(scipy_opts),
        "callback": _scipy_callback_contract(callback),
        "success": bool(result.success),
        "status": int(getattr(result, "status", 0)),
        "message": str(getattr(result, "message", "")),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "njev": int(getattr(result, "njev", 0)),
    }


def _scipy_result_has_invalid_state(result) -> bool:
    return (
        not np.all(np.isfinite(np.asarray(result.fun)))
        or not np.all(np.isfinite(np.asarray(result.jac)))
        or not np.all(np.isfinite(np.asarray(result.x)))
    )


def _mark_scipy_result_invalid_state(result):
    result.success = False
    result.status = LBFGS_STATUS_NONFINITE
    result.message = (
        "Non-finite objective, iterate, or gradient encountered during iteration."
    )
    return result


def _normalize_scipy_result(result, *, x_dtype):
    invalid_state = _scipy_result_has_invalid_state(result)
    result.x = _target_array_from_scipy_host(result.x, dtype=x_dtype)
    result.jac = _target_array_from_scipy_host(result.jac, dtype=x_dtype)
    result.nit = int(getattr(result, "nit", 0))
    result.nfev = int(getattr(result, "nfev", 0))
    if hasattr(result, "njev"):
        result.njev = int(result.njev)
    result.success = bool(result.success)
    if hasattr(result, "status"):
        result.status = int(result.status)
    if invalid_state:
        _mark_scipy_result_invalid_state(result)
    return result


def _scipy_dispatch_core(
    scipy_fun,
    x0,
    *,
    method,
    tol,
    maxiter,
    options,
    lifecycle_controller: _TargetScipyEvaluationLifecycleController | None = None,
):
    stripped_options = _strip_internal_options(options, method)
    x_dtype = _optimizer_dtype(x0)
    if method == "bfgs":
        scipy_method = "BFGS"
        scipy_opts = {"maxiter": maxiter, "gtol": tol, **stripped_options}
    else:
        scipy_method = "L-BFGS-B"
        scipy_opts = {
            "maxiter": maxiter,
            "gtol": tol,
            "maxcor": 200,
            **stripped_options,
        }
    callback = options.get("callback")
    progress_callback = options.get("progress_callback")
    objective_evaluation_index = 0
    objective_evaluation_memo = _ScipyObjectiveEvaluationMemo()
    scipy_objective_trace = (
        [] if options.get("record_scipy_callback_trace", False) else None
    )
    callback_uses_objective_evaluations = (
        progress_callback is not None
        or callback is not None
        or lifecycle_controller is not None
    )
    record_objective_evaluations = (
        callback_uses_objective_evaluations or scipy_objective_trace is not None
    )

    def scipy_fun_with_evidence(
        x_np: np.ndarray,
    ) -> tuple[float | np.floating, np.ndarray]:
        nonlocal objective_evaluation_index
        value, gradient = scipy_fun(x_np)
        objective_evaluation_index += 1
        evaluation = _ScipyObjectiveEvaluation(
            objective_evaluation_index=objective_evaluation_index,
            decision_vector=_scipy_host_array(x_np, dtype=x_dtype).copy(),
            fun=_scipy_scalar_value(value, dtype=x_dtype),
            gradient=_scipy_host_array(gradient, dtype=x_dtype).copy(),
        )
        if callback_uses_objective_evaluations:
            objective_evaluation_memo.record(evaluation)
        if scipy_objective_trace is not None:
            scipy_objective_trace.append(_scipy_objective_trace_entry(evaluation))
        return value, gradient

    scipy_objective = (
        scipy_fun_with_evidence if record_objective_evaluations else scipy_fun
    )
    scipy_callback = None
    if progress_callback is not None:
        accepted_iterations = 0

        def scipy_progress_callback(x_np: np.ndarray) -> None:
            nonlocal accepted_iterations
            accepted_evaluation = objective_evaluation_memo.resolve_accepted(
                _scipy_host_array(x_np, dtype=x_dtype)
            )
            if lifecycle_controller is not None:
                lifecycle_controller.accepted_trial(accepted_evaluation.decision_vector)
            accepted_iterations += 1
            if callback is not None:
                callback(_target_array_from_scipy_host(x_np, dtype=x_dtype))
            progress_callback(
                accepted_iterations,
                float(accepted_evaluation.fun),
                float(np.linalg.norm(accepted_evaluation.gradient, ord=np.inf)),
            )

        scipy_callback = scipy_progress_callback
    elif callback is not None or lifecycle_controller is not None:

        def scipy_state_callback(x_np: np.ndarray) -> None:
            accepted_evaluation = objective_evaluation_memo.resolve_accepted(
                _scipy_host_array(x_np, dtype=x_dtype)
            )
            if lifecycle_controller is not None:
                lifecycle_controller.accepted_trial(accepted_evaluation.decision_vector)
            if callback is not None:
                callback(_target_array_from_scipy_host(x_np, dtype=x_dtype))

        scipy_callback = scipy_state_callback

    raw_result = scipy_minimize(
        scipy_objective,
        _scipy_host_array(x0, dtype=x_dtype),
        jac=True,
        method=scipy_method,
        options=scipy_opts,
        callback=scipy_callback,
    )
    if lifecycle_controller is not None:
        lifecycle_controller.optimizer_completed()
    result = _normalize_scipy_result(raw_result, x_dtype=x_dtype)
    result.scipy_call_contract = _scipy_result_contract(
        result,
        semantic_method=method,
        scipy_method=scipy_method,
        scipy_opts=scipy_opts,
        callback=(callback if callback is not None else progress_callback),
    )
    result.scipy_objective_evaluation_trace = scipy_objective_trace
    result.scipy_callback_trace = scipy_objective_trace
    return result


def _scipy_dispatch(scipy_fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_dispatch",
        method=method,
    )
    return _scipy_dispatch_core(
        scipy_fun, x0, method=method, tol=tol, maxiter=maxiter, options=options
    )


def _make_scipy_host_value_and_grad_objective(
    value_and_grad_fn,
    *,
    x_dtype,
    initial_call,
    lifecycle_controller: _TargetScipyEvaluationLifecycleController | None = None,
):
    target_scipy_provider = (
        value_and_grad_fn
        if isinstance(value_and_grad_fn, _TargetScipyEvaluationProvider)
        else None
    )

    def scipy_fun(x_np):
        authoritative_decision_snapshot = _immutable_scipy_decision_snapshot(
            x_np,
            dtype=x_dtype,
        )
        x_jax = _target_array_from_scipy_host(
            authoritative_decision_snapshot,
            dtype=x_dtype,
        )
        expected_device = _inspect_target_scipy_device_packet_layout(x_jax).device
        if target_scipy_provider is not None:
            evaluation = target_scipy_provider.evaluate_target_scipy(
                x_jax,
                _isolated_scipy_decision_snapshot(authoritative_decision_snapshot),
            )
        else:
            evaluation = value_and_grad_fn(x_jax)
        host_evaluation = _materialize_target_scipy_evaluation(
            evaluation,
            host_decision_vector=_isolated_scipy_decision_snapshot(
                authoritative_decision_snapshot
            ),
            expected_device=expected_device,
        )
        # ``minimize(jac=True)`` consumes the same host scalar/array shape
        # returned by the CPU Boozer objective callable.
        host_value = _scipy_scalar_value(host_evaluation.value, dtype=x_dtype)
        host_gradient = _scipy_host_array(
            host_evaluation.gradient,
            dtype=x_dtype,
        )
        if lifecycle_controller is not None:
            lifecycle_controller.evaluation_returned(
                authoritative_decision_snapshot,
                host_evaluation.lifecycle,
            )
        if "payload" not in initial_call:
            initial_call["payload"] = _scipy_initial_call_contract(
                authoritative_decision_snapshot,
                host_value,
                host_gradient,
                dtype=x_dtype,
            )
        return host_value, host_gradient

    return scipy_fun


def _materialize_target_scipy_evaluation(
    evaluation,
    *,
    host_decision_vector: np.ndarray,
    expected_device: _TargetScipyDeviceIdentity,
) -> _TargetScipyHostEvaluation:
    """Materialize one complete provider packet before typed host finalization."""
    if isinstance(evaluation, _TargetScipyDeviceEvaluation):
        device_layout = _inspect_target_scipy_device_packet_layout(
            evaluation.device_packet
        )
        if device_layout.device != expected_device:
            raise ValueError(
                "Target SciPy device packet must share the decision vector device."
            )
        with jax.transfer_guard_device_to_host("allow"):
            host_packet = jax.device_get(evaluation.device_packet)
        with _target_scipy_host_extension_scope():
            host_evaluation = evaluation.finalize_host(
                host_decision_vector,
                host_packet,
                device_layout,
            )
        return _require_host_resident_target_scipy_evaluation(host_evaluation)
    with jax.transfer_guard_device_to_host("allow"):
        host_value, host_gradient = jax.device_get(evaluation)
    return _require_host_resident_target_scipy_evaluation(
        _TargetScipyHostEvaluation(
            value=host_value,
            gradient=host_gradient,
        )
    )


def _scipy_minimize_value_and_grad_core(
    value_and_grad_fn,
    x0,
    *,
    method,
    tol,
    maxiter,
    options,
):
    x_dtype = _optimizer_dtype(x0)
    initial_call = {}
    lifecycle_controller = (
        _TargetScipyEvaluationLifecycleController()
        if isinstance(value_and_grad_fn, _TargetScipyEvaluationProvider)
        else None
    )
    scipy_fun = _make_scipy_host_value_and_grad_objective(
        value_and_grad_fn,
        x_dtype=x_dtype,
        initial_call=initial_call,
        lifecycle_controller=lifecycle_controller,
    )

    result = _scipy_dispatch_core(
        scipy_fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
        lifecycle_controller=lifecycle_controller,
    )
    result.scipy_initial_call = initial_call["payload"]
    return result


def _scipy_minimize(fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_minimize",
        method=method,
    )
    return _scipy_minimize_value_and_grad_core(
        _optimizer._cached_jit_value_and_grad(fun),
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
    )


def target_scipy_minimize_value_and_grad(
    fun,
    x0,
    *,
    method,
    tol,
    maxiter,
    options,
):
    """Run SciPy host control against a JAX target value/grad evaluator."""
    if method not in {"bfgs", "lbfgs"}:
        raise ValueError(
            "target_scipy_minimize_value_and_grad() only supports "
            "method='bfgs' or method='lbfgs'."
        )
    return _scipy_minimize_value_and_grad_core(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
    )


def _scipy_minimize_value_and_grad(fun, x0, *, method, tol, maxiter, options):
    _optimizer._require_native_cpu_reference_backend_for_scipy_adapter(
        component="optimizer_jax_reference._scipy_minimize_value_and_grad",
        method=method,
    )
    return _scipy_minimize_value_and_grad_core(
        fun,
        x0,
        method=method,
        tol=tol,
        maxiter=maxiter,
        options=options,
    )


def _host_trace_result_to_optimize_result(result):
    invalid_state = (not np.isfinite(result.f_k)) or (
        not np.all(np.isfinite(result.g_k))
    )
    invalid_step_log = host_invalid_step_log_to_list(result.invalid_step_events)
    return OptimizeResult(
        x=np.asarray(result.x_k),
        fun=float(result.f_k),
        jac=np.asarray(result.g_k),
        nit=int(result.k),
        nfev=int(result.nfev),
        njev=int(result.ngev),
        success=lbfgs_status_is_success(result.status, invalid_state),
        status=int(result.status),
        message=lbfgs_status_message(result.status, invalid_state),
        ls_status=int(result.ls_status),
        line_search_final_status=int(result.ls_status),
        maxiter_hit=int(result.status) == 1,
        rejected_step_count=len(invalid_step_log),
        invalid_step_log=invalid_step_log,
        optimizer_state_trace=tuple(result.optimizer_state_trace),
    )


def _trace_minimize_value_and_grad(
    fun,
    x0,
    *,
    method,
    tol,
    maxiter,
    options,
    initial_value_and_grad=None,
):
    _optimizer._require_native_cpu_reference_backend_for_trace_adapter(
        component="optimizer_jax_reference._trace_minimize_value_and_grad",
        method=method,
    )
    if method != "lbfgs-trace":
        raise ValueError(f"Unknown CPU/C++ trace optimizer method {method!r}.")
    x_dtype = _optimizer_dtype(x0)

    def eval_value_and_grad_host(x_np):
        x_host = np.asarray(x_np, dtype=np.dtype(x_dtype))
        val, grad = fun(x_host)
        return float(val), np.asarray(grad, dtype=np.dtype(x_dtype))

    result = minimize_lbfgs_host_core(
        eval_value_and_grad_host,
        _scipy_host_array(x0, dtype=x_dtype),
        maxiter=maxiter,
        gtol=tol,
        maxcor=int(options.get("maxcor", 200)),
        ftol=float(options.get("ftol", tol)),
        maxfun=options.get("maxfun"),
        maxgrad=options.get("maxgrad"),
        maxls=int(options.get("maxls", 20)),
        initial_step_size=options.get("initial_step_size"),
        callback=options.get("callback"),
        progress_callback=options.get("progress_callback"),
        failure_callback=options.get("failure_callback"),
        initial_value_and_grad=initial_value_and_grad,
        record_optimizer_state_trace=bool(
            options.get("record_optimizer_state_trace", True)
        ),
        max_optimizer_state_trace_bytes=options.get("max_optimizer_state_trace_bytes"),
        invalid_step_log_capacity=options.get("invalid_step_log_capacity"),
    )
    return _host_trace_result_to_optimize_result(result)


def reference_least_squares(
    residual_fn,
    x0,
    *,
    method="lm",
    tol=1e-10,
    maxiter=1500,
    options=None,
    callback=None,
    progress_callback=None,
):
    """Run the CPU/reference least-squares lane."""
    if method != "lm":
        raise ValueError(
            f"reference_least_squares() only supports method='lm'. Got {method!r}."
        )

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback

    _optimizer._raise_if_target_lane_required(
        component="optimizer_jax_reference.reference_least_squares",
        method=method,
        detail=_optimizer._STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
    )
    _optimizer._raise_if_strict_optimizer_fallback(
        component="optimizer_jax_reference.reference_least_squares",
        method=method,
        detail=_optimizer._STRICT_REFERENCE_LEAST_SQUARES_DETAIL,
    )
    result = _optimizer.levenberg_marquardt(
        residual_fn,
        x0,
        maxiter=maxiter,
        tol=tol,
        ftol=options.get("ftol", 1e-8),
        xtol=options.get("xtol", 1e-8),
        gtol=options.get("gtol"),
        callback=options.get("callback"),
        progress_callback=options.get("progress_callback"),
    )

    nit = int(_optimizer._host_scalar(result["nit"], dtype=np.int64))
    status = int(_optimizer._host_scalar(result["status"], dtype=np.int64))
    info = int(_optimizer._host_scalar(result["info"], dtype=np.int64))
    success = _optimizer._host_bool(result["success"])
    return OptimizeResult(
        x=result["x"],
        fun=result["fun"],
        jac=result["grad"],
        residual=result["residual"],
        residual_jacobian=result["residual_jacobian"],
        hessian=result["hessian"],
        damping=result["damping"],
        nit=nit,
        nfev=nit + 1,
        njev=nit + 1,
        status=status,
        info=info,
        success=success,
        message=_optimizer._least_squares_result_message(
            status,
            success,
            info=info,
        ),
    )


def reference_minimize(
    fun,
    x0,
    *,
    method="bfgs",
    tol=1e-10,
    maxiter=1500,
    options=None,
    value_and_grad=False,
    callback=None,
    progress_callback=None,
    failure_callback=None,
    initial_value_and_grad=None,
    allow_jax_host_control=False,
):
    """Run the CPU/reference optimizer lane."""
    if method in _optimizer._REFERENCE_TRACE_METHODS and not value_and_grad:
        raise ValueError(
            "reference_minimize() requires value_and_grad=True for "
            "method='lbfgs-trace'."
        )
    if (
        method
        not in _optimizer._REFERENCE_METHODS | _optimizer._REFERENCE_TRACE_METHODS
    ):
        raise ValueError(
            "reference_minimize() only supports reference methods "
            f"{sorted(_optimizer._REFERENCE_METHODS | _optimizer._REFERENCE_TRACE_METHODS)}. "
            f"Got {method!r}."
        )

    fun, x0, callback, pytree_adapter = _prepare_optimizer_callable_inputs(
        fun,
        x0,
        value_and_grad=value_and_grad,
        callback=callback,
    )

    def finalize(result):
        return _optimizer._finalize_optimizer_result(result, pytree_adapter)

    options = dict(options or {})
    if callback is not None:
        options["callback"] = callback
    if progress_callback is not None:
        options["progress_callback"] = progress_callback
    if failure_callback is not None:
        options["failure_callback"] = failure_callback

    if not allow_jax_host_control:
        _optimizer._raise_if_target_lane_required(
            component="optimizer_jax_reference.reference_minimize",
            method=method,
            detail=_optimizer._STRICT_REFERENCE_OPTIMIZER_DETAIL,
        )
        _optimizer._raise_if_strict_optimizer_fallback(
            component="optimizer_jax_reference.reference_minimize",
            method=method,
            detail=_optimizer._STRICT_REFERENCE_OPTIMIZER_DETAIL,
        )

    if method in _optimizer._REFERENCE_TRACE_METHODS:
        return finalize(
            _trace_minimize_value_and_grad(
                fun,
                x0,
                method=method,
                tol=tol,
                maxiter=maxiter,
                options=options,
                initial_value_and_grad=initial_value_and_grad,
            )
        )

    scipy_adapter = (
        _scipy_minimize_value_and_grad_core
        if allow_jax_host_control and value_and_grad
        else (_scipy_minimize_value_and_grad if value_and_grad else _scipy_minimize)
    )
    return finalize(
        scipy_adapter(
            fun,
            x0,
            method=method,
            tol=tol,
            maxiter=maxiter,
            options=options,
        )
    )
