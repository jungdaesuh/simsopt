"""Driver dispatch for the public ``simsopt_jax.solve`` API."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Callable, TypeAlias, TypeVar, cast

import jax
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares as scipy_least_squares
from scipy.optimize import minimize as scipy_minimize

from simsopt_jax.geo.optimizers import optimizer as legacy
from simsopt_jax.runtime.host_boundary import (
    host_array_after_ready,
    host_float_after_ready,
)

from .contracts import (
    Callback,
    Driver,
    HessianInverse,
    InvalidStepEvent,
    LbfgsbRestartEvent,
    LbfgsbRestartReason,
    NONFINITE_RESULT_STATUS,
    OptimizerCallbackEvent,
    OptimizerResult,
    OptimizerStateTraceEntry,
    OptionsBase,
    ResidualFn,
    SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS,
    ScipyBFGSCallbackEvent,
    ScipyLBFGSBCallbackEvent,
    SimsoptBFGSCallbackEvent,
    SimsoptLBFGSBCallbackEvent,
    SimsoptLMQRCallbackEvent,
    ValueAndGradFn,
)
from .driver import (
    legacy_target_least_squares_method,
    legacy_target_minimize_method,
)
from .scipy.contracts import (
    ScipyBFGSOptions,
    ScipyBounds,
    ScipyLBFGSBOptions,
    ScipyLMOptions,
)
from .simsopt.contracts import (
    SimsoptBFGSOptions,
    SimsoptLBFGSBOptions,
    SimsoptLMQROptions,
)
from .termination import projected_gradient_inf_norm

_OptionsT = TypeVar("_OptionsT", bound=OptionsBase)
_LegacyProgress: TypeAlias = tuple[int, float, float]
_LegacyEventFactory: TypeAlias = Callable[
    [np.ndarray, int, float, float, float],
    OptimizerCallbackEvent,
]


_MINIMIZE_OPTIONS: dict[Driver, type[OptionsBase]] = {
    Driver.SCIPY_LBFGSB: ScipyLBFGSBOptions,
    Driver.SCIPY_BFGS: ScipyBFGSOptions,
    Driver.SIMSOPT_LBFGSB: SimsoptLBFGSBOptions,
    Driver.SIMSOPT_BFGS: SimsoptBFGSOptions,
}

_LEAST_SQUARES_OPTIONS: dict[Driver, type[OptionsBase]] = {
    Driver.SCIPY_LM: ScipyLMOptions,
    Driver.SIMSOPT_LM_QR: SimsoptLMQROptions,
}


def _resolve_options(
    driver: Driver,
    options: OptionsBase | None,
    options_by_driver: dict[Driver, type[_OptionsT]],
) -> _OptionsT:
    options_type = options_by_driver.get(driver)
    if options_type is None:
        allowed = ", ".join(sorted(item.value for item in options_by_driver))
        raise ValueError(
            f"Driver {driver.value!r} is not valid here. Choose: {allowed}."
        )
    if options is None:
        return options_type()
    if type(options) is not options_type:
        raise TypeError(
            f"Driver {driver.value!r} requires options of type "
            f"{options_type.__name__}, got {type(options).__name__}."
        )
    return options


def _host_optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return host_array_after_ready(value)


def _legacy_lbfgsb_options(
    options: SimsoptLBFGSBOptions,
    *,
    observes_accepted_steps: bool,
) -> dict[str, object]:
    # The fused route runs the whole solve inside one device loop, so it has no
    # host boundary left to deliver per-iteration events on. Keep the
    # callback-capable stepwise driver exactly when an observer is attached.
    return {
        "ftol": options.ftol,
        "maxcor": options.maxcor,
        "maxfun": options.maxfun,
        "maxls": options.maxls,
        "lbfgs_run_mode": "stepwise" if observes_accepted_steps else "fused_stepwise",
    }


def _legacy_bfgs_options(options: ScipyBFGSOptions | SimsoptBFGSOptions):
    payload = {"xrtol": options.xrtol}
    if isinstance(options, ScipyBFGSOptions):
        payload["norm"] = options.norm
    else:
        payload["line_search_maxiter"] = options.line_search_max_steps
    return payload


def _legacy_lm_options(options: SimsoptLMQROptions):
    return {
        "ftol": options.ftol,
        "xtol": options.xtol,
        "gtol": options.gtol,
        "max_dense_linearization_bytes": options.max_dense_linearization_bytes,
    }


def _invalid_step_events(result: OptimizeResult) -> list[InvalidStepEvent] | None:
    raw_log = getattr(result, "invalid_step_log", None)
    if raw_log is None:
        return None
    return [
        InvalidStepEvent(
            iteration=int(entry["iteration"]),
            step_scale=float(entry["step_scale"]),
            reason=str(entry["failure_reason"]),
        )
        for entry in raw_log
    ]


def _optimizer_state_trace(
    result: OptimizeResult,
) -> list[OptimizerStateTraceEntry] | None:
    raw_trace = getattr(result, "optimizer_state_trace", None)
    if raw_trace is None:
        return None
    return [
        OptimizerStateTraceEntry(
            iteration=int(entry["iteration"]),
            fun=float(entry["fun"]),
            grad_norm_inf=float(entry["jac_inf_norm"]),
            # Only the SciPy L-BFGS-B restart route records it.
            nfev=entry.get("nfev"),
        )
        for entry in raw_trace
    ]


def _nonfinite_result_fields(
    *,
    x: np.ndarray,
    fun: float,
    jac: np.ndarray | None,
    residual: np.ndarray | None,
) -> tuple[str, ...]:
    """The returned-state fields holding a NaN or an infinity, in field order.

    ``jac`` and ``residual`` are judged when the driver returned them; an absent
    one is not an error.  Only the returned state is judged, never the trials a
    solver evaluated on its way there.
    """
    finite = (
        ("x", bool(np.all(np.isfinite(x)))),
        ("fun", math.isfinite(fun)),
        ("jac", jac is None or bool(np.all(np.isfinite(jac)))),
        ("residual", residual is None or bool(np.all(np.isfinite(residual)))),
    )
    return tuple(name for name, ok in finite if not ok)


def _public_result(
    result: OptimizeResult,
    *,
    driver: Driver,
    options_used: OptionsBase,
    wallclock_s: float,
) -> OptimizerResult:
    """The one conversion every ``minimize`` / ``least_squares`` driver returns through.

    A non-finite returned state is never a success, whatever the backend
    reported: ``status`` is ``NONFINITE_RESULT_STATUS``, ``success`` False, and
    the backend's own termination moves to the raw fields.  The returned state
    itself is kept as diagnostic evidence, never replaced.  A finite result is
    the backend's unchanged.
    """
    hess_inv = cast(HessianInverse | None, getattr(result, "hess_inv", None))
    x = host_array_after_ready(result.x)
    fun = host_float_after_ready(result.fun)
    jac = _host_optional_array(getattr(result, "jac", None))
    residual = _host_optional_array(getattr(result, "residual", None))
    status = int(getattr(result, "status", 0))
    success = bool(getattr(result, "success", False))
    message = str(getattr(result, "message", ""))
    nonfinite_fields = _nonfinite_result_fields(
        x=x, fun=fun, jac=jac, residual=residual
    )
    raw_status: int | None = None
    raw_success: bool | None = None
    raw_message: str | None = None
    jacobian_evaluations = getattr(result, "jacobian_evaluations", None)
    if nonfinite_fields:
        raw_status, raw_success, raw_message = status, success, message
        status, success = NONFINITE_RESULT_STATUS, False
        message = f"NONFINITE RESULT ({', '.join(nonfinite_fields)}): {raw_message}"
    return OptimizerResult(
        x=x,
        fun=fun,
        jac=jac,
        nit=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        status=status,
        success=success,
        message=message,
        driver=driver,
        options_used=options_used,
        wallclock_s=wallclock_s,
        residual=residual,
        residual_jacobian=_host_optional_array(
            getattr(result, "residual_jacobian", None)
        ),
        hessian=_host_optional_array(getattr(result, "hessian", None)),
        hess_inv=hess_inv,
        invalid_step_log=_invalid_step_events(result),
        optimizer_state_trace=_optimizer_state_trace(result),
        restart_log=tuple(getattr(result, "restart_log", ())),
        nonfinite_fields=nonfinite_fields,
        jacobian_evaluations=(
            None if jacobian_evaluations is None else int(jacobian_evaluations)
        ),
        raw_status=raw_status,
        raw_success=raw_success,
        raw_message=raw_message,
    )


def _scipy_lm_result(
    residual_fn: ResidualFn,
    x0,
    *,
    options: ScipyLMOptions,
    residual_args: tuple[object, ...] = (),
) -> OptimizeResult:
    def residual_host(x_host):
        return np.asarray(
            residual_fn(np.asarray(x_host), *residual_args),
            dtype=float,
        ).reshape(-1)

    result = scipy_least_squares(
        residual_host,
        np.asarray(x0, dtype=float),
        method="lm",
        max_nfev=options.max_nfev,
        ftol=options.ftol,
        xtol=options.xtol,
        gtol=options.gtol,
    )
    residual = np.asarray(result.fun, dtype=float).reshape(-1)
    jacobian = np.asarray(result.jac, dtype=float)
    gradient = jacobian.T @ residual
    hessian = jacobian.T @ jacobian
    njev = 0 if result.njev is None else int(result.njev)
    return OptimizeResult(
        x=np.asarray(result.x, dtype=float),
        fun=float(0.5 * np.dot(residual, residual)),
        jac=gradient,
        nit=njev,
        nfev=int(result.nfev),
        njev=njev,
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
        residual=residual,
        residual_jacobian=jacobian,
        hessian=hessian,
    )


# The two SciPy-internal facts ``restart_after_nonwolfe_stop`` relies on (scipy
# 1.17.1, git 527eb7fd).  L-BFGS-B's line search ``lnsrlb`` hands dcsrch
# ``gtol = 0.9`` (``scipy/optimize/__lbfgsb.c:2578``), so a step dcsrch accepts
# on convergence satisfies ``|g_new.d| <= 0.9 |g_old.d|``; ``lnsrlb`` accepts a
# dcsrch WARNING exactly like convergence (``:2655-2679``), and such a step may
# fail it.
_LBFGSB_CURVATURE_GTOL = 0.9
# ``mainlb``'s relative-reduction test ``(fold - f) <= factr*epsmch*max(|fold|,
# |f|, 1)`` (``__lbfgsb.c:975-984``, CONVERGENCE / CONV_F), as
# ``_lbfgsb_py.py:487-505`` reports it: ``status`` 0 and this ``message``.
_LBFGSB_RELATIVE_REDUCTION_STOP = (
    0,
    "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
)
# A trial of one search is ``x = stp * d + t`` rounded (at most two roundings,
# then a clamp that only undoes rounding past a bound, ``__lbfgsb.c:2662-2675``),
# so its offset from ``t`` is off the ray through the first trial by at most a
# few ``eps`` times the magnitudes involved; this factor is that "few", with room.
_RAY_ROUNDING_FACTOR = 8.0
# ``scipy.optimize.minimize`` returns early, before any method runs, when the
# bounds fix every variable (``_minimize.py:721`` ->
# ``_optimize_result_for_equal_bounds``, ``:1160-1198``, scipy 1.17.1): that
# result holds x, fun (one evaluation), success, message, nfev, njev and nhev
# only -- no status, nit or jac.  Without constraints (this route passes none)
# it is always this message, with success True.
_SCIPY_ALL_FIXED_MESSAGE = "All independent variables were fixed by bounds."
# The status this route defines for that early return, which carries none:
# SciPy's success status for L-BFGS-B, the success its result states.
_SCIPY_ALL_FIXED_STATUS = 0


@dataclass(frozen=True)
class _ScipyTermination:
    """A SciPy result's status, iteration count and gradient, as this route reads them."""

    status: int
    nit: int
    jac: np.ndarray | None


def _scipy_termination(result: OptimizeResult) -> _ScipyTermination:
    """The fields SciPy returned; for its all-fixed early return, the defined ones.

    That early return ran no iteration (``nit`` 0) and computed no gradient
    (``jac`` None: nothing is evaluated to supply one); its status is
    ``_SCIPY_ALL_FIXED_STATUS``.  Any other result without a status is not a
    shape this route knows, and is refused.
    """
    if "status" in result:
        return _ScipyTermination(
            status=int(result.status), nit=int(result.nit), jac=result.jac
        )
    if str(result.message) != _SCIPY_ALL_FIXED_MESSAGE:
        raise ValueError(
            f"SciPy returned no status, with message {result.message!r}; only its "
            "all-fixed early return may"
        )
    return _ScipyTermination(status=_SCIPY_ALL_FIXED_STATUS, nit=0, jac=None)


def _lbfgsb_scipy_options(
    options: ScipyLBFGSBOptions, *, maxiter: int, maxfun: int
) -> dict[str, object]:
    return {
        "maxiter": maxiter,
        "maxfun": maxfun,
        "gtol": options.gtol,
        "ftol": options.ftol,
        "maxcor": options.maxcor,
        "maxls": options.maxls,
    }


def _owned_snapshot(array) -> np.ndarray:
    """A read-only float64 copy no caller buffer can alias."""
    snapshot = np.array(array, dtype=float, copy=True)
    snapshot.setflags(write=False)
    return snapshot


@dataclass(frozen=True)
class _HostEvaluation:
    """One objective evaluation handed to SciPy: owned, read-only host arrays."""

    x: np.ndarray
    fun: float
    gradient: np.ndarray


@dataclass(frozen=True)
class _LineSearch:
    """One L-BFGS-B iteration: its start and every trial it evaluated, in order.

    The last trial is the accepted point.  A single search's trials all lie on
    ``base.x + stp * d``, so ``first_trial.x - base.x`` is ``d`` up to the
    positive first ``stp``, a factor the curvature test cancels.  An iteration
    can hold two searches -- L-BFGS-B retries a failed one from the same base
    along a new direction before the iteration ends (``__lbfgsb.c:918-951``)
    -- and then the first trial's ray is not the accepted search's:
    ``on_one_ray`` is False unless every trial lies on the first trial's ray
    to within rounding, and nothing else here may be read when it is False.
    """

    base: _HostEvaluation
    trials: tuple[_HostEvaluation, ...]

    @property
    def first_trial(self) -> _HostEvaluation:
        return self.trials[0]

    @property
    def accepted(self) -> _HostEvaluation:
        return self.trials[-1]

    @property
    def on_one_ray(self) -> bool:
        eps = float(np.finfo(float).eps)
        base = self.base.x
        direction = self.first_trial.x - base
        direction_rounding = np.abs(self.first_trial.x) + np.abs(base)
        for trial in self.trials[1:]:
            offset = trial.x - base
            # Trials sit at ``stp >= 0``; a negative projection is off the ray.
            scale = max(float(offset @ direction) / float(direction @ direction), 0.0)
            rounding = (
                _RAY_ROUNDING_FACTOR
                * eps
                * np.linalg.norm(
                    np.abs(trial.x)
                    + np.abs(base)
                    + scale * (np.abs(direction) + direction_rounding)
                )
            )
            if np.linalg.norm(offset - scale * direction) > rounding:
                return False
        return True

    def _slopes(self) -> tuple[float, float]:
        """``(g_base.d, g_accepted.d)``."""
        direction = self.first_trial.x - self.base.x
        return (
            float(self.base.gradient @ direction),
            float(self.accepted.gradient @ direction),
        )

    @property
    def curvature_ratio(self) -> float:
        base_slope, accepted_slope = self._slopes()
        return accepted_slope / abs(base_slope)

    @property
    def fails_curvature_condition(self) -> bool:
        base_slope, accepted_slope = self._slopes()
        return abs(accepted_slope) > _LBFGSB_CURVATURE_GTOL * abs(base_slope)

    @property
    def accepted_step_fraction(self) -> float:
        return float(
            np.linalg.norm(self.accepted.x - self.base.x)
            / np.linalg.norm(self.first_trial.x - self.base.x)
        )

    @property
    def first_trial_fun_ratio(self) -> float:
        base_fun, trial_fun = self.base.fun, self.first_trial.fun
        if base_fun != 0.0:
            return trial_fun / base_fun
        return 1.0 if trial_fun == 0.0 else math.copysign(math.inf, trial_fun)


class _LbfgsbCallRecorder:
    """The evaluations one SciPy L-BFGS-B call needs kept to judge its last search.

    ``fun`` is the objective handed to SciPy; it keeps owned read-only
    snapshots, so a callable that reuses one output buffer cannot rewrite the
    record.  It answers the first request at ``served_start.x`` (bitwise) from
    ``served_start`` instead of evaluating, so a restarted call starts from the
    logged value and gradient; ``served_evaluations`` is what SciPy's
    ``nfev``/``njev`` count but nothing evaluated.  ``end_iteration`` is
    SciPy's per-iteration callback and then hands ``callback`` the accepted
    evaluation it already holds, so observing the solve evaluates nothing.  Only the
    running iteration's evaluations are kept.  ``state_trace`` holds one raw
    accepted-iterate entry per iteration end, numbered and counted from the
    chain's totals before this call: ``nfev`` is the chain's true evaluations
    up to that NEW_X, so an ABNORMAL tail after the last one is ``nfev`` of
    the result minus the last entry's.
    """

    def __init__(
        self,
        scipy_fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
        served_start: _HostEvaluation | None,
        callback: Callable[[_HostEvaluation], None] | None,
        *,
        iteration_offset: int = 0,
        evaluation_offset: int = 0,
    ) -> None:
        self._scipy_fun = scipy_fun
        self._served_start = served_start
        self._callback = callback
        self._iteration_offset = iteration_offset
        self._evaluation_offset = evaluation_offset
        self._iteration_evaluations: list[_HostEvaluation] = []
        self._true_evaluations = 0
        self.served_evaluations = 0
        self.last_search: _LineSearch | None = None
        self.state_trace: list[dict[str, float | int]] = []

    def fun(self, x_host: np.ndarray) -> tuple[float, np.ndarray]:
        served = self._served_start
        if served is not None and np.array_equal(x_host, served.x):
            self._served_start = None
            self.served_evaluations += 1
            evaluation = served
        else:
            value, gradient = self._scipy_fun(x_host)
            self._true_evaluations += 1
            evaluation = _HostEvaluation(
                x=_owned_snapshot(x_host),
                fun=value,
                gradient=_owned_snapshot(gradient),
            )
        self._iteration_evaluations.append(evaluation)
        return evaluation.fun, evaluation.gradient

    def end_iteration(self, x_host: np.ndarray) -> None:
        evaluations = self._iteration_evaluations
        # A search whose every trial rounded to its base bitwise never reached
        # the objective (SciPy's memo served them all): there is no direction
        # to judge it by, and ``last_search`` is None.
        self.last_search = (
            _LineSearch(base=evaluations[0], trials=tuple(evaluations[1:]))
            if len(evaluations) > 1
            else None
        )
        self._iteration_evaluations = [evaluations[-1]]
        # SciPy's iterate at NEW_X is the last point it had evaluated.
        accepted = evaluations[-1]
        self.state_trace.append(
            {
                "iteration": self._iteration_offset + len(self.state_trace) + 1,
                "fun": accepted.fun,
                "jac_inf_norm": float(np.max(np.abs(accepted.gradient))),
                "nfev": self._evaluation_offset + self._true_evaluations,
            }
        )
        if self._callback is not None:
            self._callback(accepted)


def _minimize_lbfgsb_with_restarts(
    scipy_fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x_start: np.ndarray,
    *,
    options: ScipyLBFGSBOptions,
    bounds: ScipyBounds | None,
    callback: Callable[[_HostEvaluation], None] | None,
) -> tuple[OptimizeResult, tuple[LbfgsbRestartEvent, ...]]:
    """SciPy L-BFGS-B under ``restart_after_nonwolfe_stop``, as one or more SciPy calls.

    ``nit``/``nfev``/``njev`` are summed over the calls and count true
    evaluations.  Each call gets exactly what is left of ``maxiter`` and
    ``maxfun``; SciPy tests both only at iteration ends, so like one call the
    chain can end past ``maxfun`` by at most its last iteration's evaluations.
    ``x``, ``fun`` and ``jac`` are the last call's; so are ``status``,
    ``success`` and ``message`` unless the chain ends on a recognized stall it
    could not resume, which is ``SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS``,
    unsuccessful, with that call's own termination kept in the message and in
    the last ``restart_log`` event.  ``optimizer_state_trace`` has one raw
    entry per accepted iteration of the chain, with its cumulative ``nfev``.
    """
    restart_log: list[LbfgsbRestartEvent] = []
    state_trace: list[dict[str, float | int]] = []
    nit = nfev = njev = 0
    x = x_start
    served_start: _HostEvaluation | None = None
    scipy_bounds = None if bounds is None else bounds.scipy_bounds()
    while True:
        recorder = _LbfgsbCallRecorder(
            scipy_fun,
            served_start,
            callback,
            iteration_offset=nit,
            evaluation_offset=nfev,
        )
        call = scipy_minimize(
            recorder.fun,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options=_lbfgsb_scipy_options(
                options, maxiter=options.maxiter - nit, maxfun=options.maxfun - nfev
            ),
            callback=recorder.end_iteration,
        )
        termination = _scipy_termination(call)
        nit += termination.nit
        nfev += int(call.nfev) - recorder.served_evaluations
        njev += int(call.njev) - recorder.served_evaluations
        state_trace.extend(recorder.state_trace)
        search = recorder.last_search
        if (
            search is None
            or (termination.status, str(call.message))
            != _LBFGSB_RELATIVE_REDUCTION_STOP
        ):
            break
        classified = search.on_one_ray
        if not classified:
            reason = LbfgsbRestartReason.UNCLASSIFIED_SEARCH
        elif not search.fails_curvature_condition:
            break
        # A call's first iteration already ran with an empty memory, which is
        # where L-BFGS-B itself aborts instead of restarting (``col == 0``,
        # ``__lbfgsb.c:924-938``).
        elif termination.nit == 1:
            reason = LbfgsbRestartReason.FRESH_MEMORY_STALL
        # SciPy completes at least one iteration whatever its budget says.
        elif nit >= options.maxiter or nfev >= options.maxfun:
            reason = LbfgsbRestartReason.BUDGET_EXHAUSTED
        else:
            reason = LbfgsbRestartReason.RESTARTED
        restart_log.append(
            LbfgsbRestartEvent(
                iteration=nit,
                reason=reason,
                curvature_ratio=search.curvature_ratio if classified else math.nan,
                fun=search.accepted.fun,
                projected_grad_norm_inf=projected_gradient_inf_norm(
                    search.accepted.x, search.accepted.gradient, bounds
                ),
                first_trial_fun=search.first_trial.fun,
                first_trial_fun_ratio=search.first_trial_fun_ratio,
                accepted_step_fraction=search.accepted_step_fraction
                if classified
                else math.nan,
                scipy_status=termination.status,
                scipy_message=str(call.message),
            )
        )
        if reason is not LbfgsbRestartReason.RESTARTED:
            break
        served_start = search.accepted
        x = np.array(search.accepted.x)
    unresolved = bool(restart_log) and restart_log[-1].reason in (
        LbfgsbRestartReason.FRESH_MEMORY_STALL,
        LbfgsbRestartReason.BUDGET_EXHAUSTED,
    )
    return (
        OptimizeResult(
            x=call.x,
            fun=call.fun,
            jac=termination.jac,
            nit=nit,
            nfev=nfev,
            njev=njev,
            status=SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS
            if unresolved
            else termination.status,
            success=False if unresolved else call.success,
            message=f"UNRESOLVED NON-WOLFE STALL ({restart_log[-1].reason}): "
            f"{call.message}"
            if unresolved
            else call.message,
            optimizer_state_trace=state_trace,
        ),
        tuple(restart_log),
    )


def _run_scipy_minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    driver: Driver,
    options: ScipyLBFGSBOptions | ScipyBFGSOptions,
    callback: Callback | None,
    bounds: ScipyBounds | None,
) -> OptimizeResult:
    iteration = 0
    start = time.perf_counter()
    # SciPy runs on the host, so this route hands the objective the parameters
    # where ``x0`` already lives: a device ``x0`` is a device objective and each
    # evaluation crosses the boundary once in each direction, while a host
    # ``x0`` never leaves the host.  Both crossings are ``jax.device_put`` /
    # ``jax.device_get``, they are the only ones on this route, and they are all
    # in ``value_and_gradient_at`` -- so a caller may hold JAX's strict transfer
    # guard around the whole solve.
    placement = x0.sharding if isinstance(x0, jax.Array) else None

    def value_and_gradient_at(x_host):
        """Host parameters in, host value and gradient out."""
        parameters = x_host if placement is None else jax.device_put(x_host, placement)
        value, gradient = value_and_grad_fn(parameters)
        return jax.device_get(value), jax.device_get(gradient)

    # Every ``np.asarray`` below acts on a host value ``value_and_gradient_at``
    # has already brought across, and normalizes what SciPy's float64 drivers
    # were already handed: a callable may return a Python float for the value.

    def scipy_fun(x_host):
        value, gradient = value_and_gradient_at(x_host)
        return float(np.asarray(value).reshape(())), np.asarray(gradient, dtype=float)

    def emit_event(x_host, value, gradient):
        nonlocal iteration
        if callback is None:
            return

        iteration += 1
        grad_host = np.asarray(gradient, dtype=float)
        event_fields = {
            "iteration": iteration,
            "x": np.asarray(x_host, dtype=float).copy(),
            "fun": float(np.asarray(value).reshape(())),
            "grad_norm_inf": float(np.linalg.norm(grad_host, ord=np.inf)),
            "wallclock_s": time.perf_counter() - start,
        }
        if driver == Driver.SCIPY_LBFGSB:
            callback(ScipyLBFGSBCallbackEvent(**event_fields))
        else:
            callback(ScipyBFGSCallbackEvent(**event_fields))

    def scipy_callback(x_host):
        # SciPy's callback carries x only, so this route evaluates the iterate again
        # (outside ``nfev``); the restart route below re-uses its accepted evaluation.
        value, gradient = value_and_gradient_at(x_host)
        emit_event(x_host, value, gradient)

    def accepted_callback(accepted: _HostEvaluation):
        emit_event(accepted.x, accepted.fun, accepted.gradient)

    if isinstance(options, ScipyLBFGSBOptions):
        scipy_method = "L-BFGS-B"
        scipy_options = _lbfgsb_scipy_options(
            options, maxiter=options.maxiter, maxfun=options.maxfun
        )
    else:
        scipy_method = "BFGS"
        scipy_options = {
            "maxiter": options.maxiter,
            "gtol": options.gtol,
            "xrtol": options.xrtol,
            "norm": options.norm,
        }
    x_start = np.asarray(jax.device_get(x0), dtype=float)
    scipy_bounds = None if bounds is None else bounds.scipy_bounds()
    observer = scipy_callback if callback is not None else None
    restart_log: tuple[LbfgsbRestartEvent, ...] = ()
    state_trace: list[dict[str, float | int]] | None = None
    if isinstance(options, ScipyLBFGSBOptions) and options.restart_after_nonwolfe_stop:
        result, restart_log = _minimize_lbfgsb_with_restarts(
            scipy_fun,
            x_start,
            options=options,
            bounds=bounds,
            callback=accepted_callback if callback is not None else None,
        )
        state_trace = result.optimizer_state_trace
    else:
        result = scipy_minimize(
            scipy_fun,
            x_start,
            jac=True,
            method=scipy_method,
            bounds=scipy_bounds,
            options=scipy_options,
            callback=observer,
        )
    termination = _scipy_termination(result)
    return OptimizeResult(
        x=np.asarray(result.x, dtype=float),
        fun=float(np.asarray(result.fun).reshape(())),
        jac=None
        if termination.jac is None
        else np.asarray(termination.jac, dtype=float),
        nit=termination.nit,
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        status=termination.status,
        success=bool(result.success),
        message=str(result.message),
        hessian=getattr(result, "hess_inv", None)
        if driver == Driver.SCIPY_BFGS
        else None,
        restart_log=restart_log,
        optimizer_state_trace=state_trace,
    )


def _legacy_callback_pair(
    callback: Callback,
    make_event: _LegacyEventFactory,
):
    pending_x: deque[np.ndarray] = deque()
    pending_progress: deque[_LegacyProgress] = deque()
    pending_lock = Lock()
    start = time.perf_counter()

    def ready_event():
        if not pending_x or not pending_progress:
            return None
        x_host = pending_x.popleft()
        iteration, fun, grad_norm_inf = pending_progress.popleft()
        return make_event(
            x_host.copy(),
            iteration,
            fun,
            grad_norm_inf,
            time.perf_counter() - start,
        )

    def legacy_callback(x):
        x_host = host_array_after_ready(x, dtype=float)
        with pending_lock:
            pending_x.append(x_host)
            event = ready_event()
        if event is not None:
            callback(event)

    def legacy_progress(iteration_value, fun_value, grad_norm_inf):
        progress = (
            int(np.asarray(iteration_value).reshape(())),
            float(np.asarray(fun_value).reshape(())),
            float(np.asarray(grad_norm_inf).reshape(())),
        )
        with pending_lock:
            pending_progress.append(progress)
            event = ready_event()
        if event is not None:
            callback(event)

    return legacy_callback, legacy_progress


def _legacy_minimize_callbacks(
    callback: Callback | None,
    *,
    driver: Driver,
):
    if callback is None:
        return None, None

    def make_event(
        x_host: np.ndarray,
        iteration: int,
        fun: float,
        grad_norm_inf: float,
        wallclock_s: float,
    ) -> OptimizerCallbackEvent:
        base_fields = {
            "iteration": iteration,
            "x": x_host,
            "fun": fun,
            "grad_norm_inf": grad_norm_inf,
            "wallclock_s": wallclock_s,
        }
        if driver == Driver.SIMSOPT_LBFGSB:
            return SimsoptLBFGSBCallbackEvent(
                **base_fields,
                accepted_alpha=float("nan"),
                num_linesearch_steps=0,
            )
        if driver == Driver.SIMSOPT_BFGS:
            return SimsoptBFGSCallbackEvent(
                **base_fields,
                accepted_alpha=float("nan"),
                num_linesearch_steps=0,
            )
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")

    return _legacy_callback_pair(callback, make_event)


def _legacy_least_squares_callbacks(
    callback: Callback | None,
    *,
    driver: Driver,
    options: OptionsBase,
):
    if callback is None:
        return None, None

    def make_event(
        x_host: np.ndarray,
        iteration: int,
        fun: float,
        grad_norm_inf: float,
        wallclock_s: float,
    ) -> OptimizerCallbackEvent:
        residual_norm = float(np.sqrt(max(0.0, 2.0 * fun)))
        base_fields = {
            "iteration": iteration,
            "x": x_host,
            "fun": fun,
            "grad_norm_inf": grad_norm_inf,
            "wallclock_s": wallclock_s,
        }
        if isinstance(options, SimsoptLMQROptions):
            return SimsoptLMQRCallbackEvent(
                **base_fields,
                residual_norm=residual_norm,
                damping=float("nan"),
                minpack_info=None,
            )
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")

    return _legacy_callback_pair(callback, make_event)


def minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0,
    *,
    driver: Driver,
    options: OptionsBase | None = None,
    callback: Callback | None = None,
    bounds: ScipyBounds | None = None,
) -> OptimizerResult:
    """Minimize a scalar objective with a typed JAX-lane driver.

    ``bounds`` boxes the parameters and is accepted only by
    ``Driver.SCIPY_LBFGSB``, which hands it to SciPy unchanged; no other driver
    enforces bounds, so passing them to one is an error, not a silent no-op.
    """
    if bounds is not None and driver != Driver.SCIPY_LBFGSB:
        raise ValueError(
            f"bounds are enforced only by {Driver.SCIPY_LBFGSB.value!r}; "
            f"driver {driver.value!r} would ignore them."
        )
    options_used = _resolve_options(driver, options, _MINIMIZE_OPTIONS)
    start = time.perf_counter()
    if isinstance(options_used, ScipyLBFGSBOptions | ScipyBFGSOptions):
        result = _run_scipy_minimize(
            value_and_grad_fn,
            x0,
            driver=driver,
            options=options_used,
            callback=callback,
            bounds=bounds,
        )
    elif isinstance(options_used, SimsoptLBFGSBOptions):
        legacy_callback, legacy_progress_callback = _legacy_minimize_callbacks(
            callback,
            driver=driver,
        )
        result = legacy.target_minimize(
            value_and_grad_fn,
            x0,
            method=legacy_target_minimize_method(driver),
            tol=options_used.gtol,
            maxiter=options_used.maxiter,
            options=_legacy_lbfgsb_options(
                options_used,
                observes_accepted_steps=legacy_callback is not None
                or legacy_progress_callback is not None,
            ),
            value_and_grad=True,
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    elif isinstance(options_used, SimsoptBFGSOptions):
        legacy_callback, legacy_progress_callback = _legacy_minimize_callbacks(
            callback,
            driver=driver,
        )
        result = legacy.target_minimize(
            value_and_grad_fn,
            x0,
            method=legacy_target_minimize_method(driver),
            tol=options_used.gtol,
            maxiter=options_used.maxiter,
            options=_legacy_bfgs_options(options_used),
            value_and_grad=True,
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
        )
    else:
        raise TypeError(f"Unsupported minimize options {type(options_used).__name__}.")
    wallclock_s = time.perf_counter() - start
    return _public_result(
        result,
        driver=driver,
        options_used=options_used,
        wallclock_s=wallclock_s,
    )


def least_squares(
    residual_fn: ResidualFn,
    x0,
    *,
    driver: Driver,
    options: OptionsBase | None = None,
    callback: Callback | None = None,
    residual_args: tuple[object, ...] = (),
) -> OptimizerResult:
    """Solve a residual least-squares problem with a typed JAX-lane driver."""
    options_used = _resolve_options(driver, options, _LEAST_SQUARES_OPTIONS)
    if callback is not None and driver == Driver.SCIPY_LM:
        raise ValueError(f"driver={driver.value!r} does not support callbacks.")
    start = time.perf_counter()

    if isinstance(options_used, ScipyLMOptions):
        if residual_args:
            result = _scipy_lm_result(
                residual_fn,
                x0,
                options=options_used,
                residual_args=residual_args,
            )
        else:
            result = _scipy_lm_result(residual_fn, x0, options=options_used)
    elif isinstance(options_used, SimsoptLMQROptions):
        legacy_callback, legacy_progress_callback = _legacy_least_squares_callbacks(
            callback,
            driver=driver,
            options=options_used,
        )
        result = legacy.target_least_squares(
            residual_fn,
            x0,
            method=legacy_target_least_squares_method(driver),
            maxiter=options_used.maxiter,
            options=_legacy_lm_options(options_used),
            callback=legacy_callback,
            progress_callback=legacy_progress_callback,
            args=residual_args,
        )
    else:
        raise TypeError(
            f"Unsupported least-squares options {type(options_used).__name__}."
        )
    wallclock_s = time.perf_counter() - start
    return _public_result(
        result,
        driver=driver,
        options_used=options_used,
        wallclock_s=wallclock_s,
    )


__all__ = ["least_squares", "minimize"]
