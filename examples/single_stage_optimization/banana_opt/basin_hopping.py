from __future__ import annotations

from collections.abc import Callable
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import basinhopping


DEFAULT_NORMALIZED_STEP_RMS_LIMIT = 5.0
DEFAULT_BASIN_TEMPERATURE = 1.0
BASIN_TELEMETRY_FIELDS = (
    "basin_accepted_hops",
    "basin_rejected_hops",
    "basin_best_objective",
    "basin_accept_test_rejections",
    "basin_accept_test_triggered",
)


def _normalized_step_rms(x_old, x_new) -> float:
    old = np.asarray(x_old, dtype=float)
    new = np.asarray(x_new, dtype=float)
    if old.shape != new.shape:
        raise ValueError("Basin-hopping vectors must have matching shapes")
    if old.size == 0:
        return 0.0
    scale = np.maximum(np.abs(old), 1.0)
    delta = (new - old) / scale
    return float(np.linalg.norm(delta) / math.sqrt(delta.size))


@dataclass
class BasinHoppingMonitor:
    normalized_step_rms_limit: float = DEFAULT_NORMALIZED_STEP_RMS_LIMIT
    accepted_hops: int = 0
    rejected_hops: int = 0
    completed_hops: int = 0
    best_objective: float | None = None
    initial_objective: float | None = None
    best_hop_objective: float | None = None
    best_hop_index: int | None = None
    best_result_source: str | None = None
    accept_test_rejections: int = 0
    accept_test_triggered: bool = False
    nonfinite_rejections: int = 0
    normalized_step_rejections: int = 0
    _seen_initial_callback: bool = False

    def _record_rejection(self, *, nonfinite: bool = False, normalized_step: bool = False) -> bool:
        self.accept_test_rejections += 1
        self.accept_test_triggered = True
        if nonfinite:
            self.nonfinite_rejections += 1
        if normalized_step:
            self.normalized_step_rejections += 1
        return False

    def accept_test(self, f_new, x_new, f_old, x_old):
        if not np.isfinite(float(f_new)) or not np.all(np.isfinite(np.asarray(x_new, dtype=float))):
            return self._record_rejection(nonfinite=True)

        if _normalized_step_rms(x_old, x_new) > self.normalized_step_rms_limit:
            return self._record_rejection(normalized_step=True)
        return True

    def callback(self, x, f, accept) -> bool:
        objective = float(f)
        if not self._seen_initial_callback:
            self._seen_initial_callback = True
            self.initial_objective = objective
            self.best_objective = objective
            self.best_result_source = "initial_local"
            return False
        self.completed_hops += 1
        if self.best_hop_objective is None or objective < self.best_hop_objective:
            self.best_hop_objective = objective
            self.best_hop_index = self.completed_hops
        if self.best_objective is None or objective < self.best_objective:
            self.best_objective = objective
            self.best_result_source = "hop"
            self.best_hop_objective = objective
            self.best_hop_index = self.completed_hops
        if accept:
            self.accepted_hops += 1
        else:
            self.rejected_hops += 1
        return False

    def as_dict(self) -> dict[str, float | int | bool | None]:
        objective_improvement = None
        if self.initial_objective is not None and self.best_objective is not None:
            objective_improvement = self.initial_objective - self.best_objective
        return {
            "basin_accepted_hops": self.accepted_hops,
            "basin_rejected_hops": self.rejected_hops,
            "basin_completed_hops": self.completed_hops,
            "basin_best_objective": self.best_objective,
            "basin_initial_objective": self.initial_objective,
            "basin_best_hop_objective": self.best_hop_objective,
            "basin_best_hop_index": self.best_hop_index,
            "basin_best_result_source": self.best_result_source,
            "basin_objective_improvement": objective_improvement,
            "basin_accept_test_rejections": self.accept_test_rejections,
            "basin_accept_test_triggered": self.accept_test_triggered,
            "basin_nonfinite_rejections": self.nonfinite_rejections,
            "basin_normalized_step_rejections": self.normalized_step_rejections,
        }


def telemetry_values(telemetry: dict[str, float | int | bool | None]) -> tuple[float | int | bool | None, ...]:
    return tuple(telemetry[field] for field in BASIN_TELEMETRY_FIELDS)


def run_basin_hopping(
    fun,
    dofs,
    *,
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = DEFAULT_BASIN_TEMPERATURE,
    basin_niter_success: int | None = None,
    rng_seed: int,
    minimizer_kwargs: dict,
    normalized_step_rms_limit: float = DEFAULT_NORMALIZED_STEP_RMS_LIMIT,
    disp: bool = True,
    local_minimum_callback: Callable[[np.ndarray, float, bool], bool | None] | None = None,
):
    monitor = BasinHoppingMonitor(
        normalized_step_rms_limit=normalized_step_rms_limit,
    )
    basin_callback = monitor.callback
    if local_minimum_callback is not None:
        def basin_callback(x, f, accept) -> bool:
            should_stop = bool(monitor.callback(x, f, accept))
            callback_result = local_minimum_callback(
                np.asarray(x, dtype=float).copy(),
                float(f),
                bool(accept),
            )
            return should_stop or bool(callback_result)

    result = basinhopping(
        fun,
        dofs,
        minimizer_kwargs=minimizer_kwargs,
        niter=basin_hops,
        stepsize=basin_stepsize,
        T=basin_temperature,
        niter_success=basin_niter_success,
        rng=np.random.default_rng(rng_seed),
        accept_test=monitor.accept_test,
        callback=basin_callback,
        disp=disp,
    )
    return result, monitor.as_dict()
