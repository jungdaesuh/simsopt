"""Opt-in ALM history: one entry per outer-step decision of :func:`minimize_alm`.

:class:`ALMHistoryRecorder` consumes the :class:`~.events.ALMOuterStepEvent`
stream and rebuilds, from the events alone, the per-step history that
drivers and artifact writers read: 53 shared keys per entry
(`_build_alm_history_entry`), the conditioning metrics, the decision
annotations, and per-constraint diagnostics that are materialized only for
surviving entries, once, when a callback or ``history()`` first needs them.
Importing :mod:`simsopt_alm` does not load this module.
"""

from __future__ import annotations

import pickle
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from .continuation import ALMIterateMeasurement
from .core import (
    _POSITIVE_WHEN_PROVIDED,
    ALMConstraintRoutingState,
    ALMSettings,
    _as_float_list,
    _augmented_stationarity_norm,
    _conditioning_metrics,
    _finite_alm_integer_or_none,
    _max_value,
    _penalty_values,
    _positive_shift_and_augmented_terms,
    _surrogate_kkt_stationarity_norm,
    alm_raw_dual_estimates,
)
from .events import ALMOuterStepEvent
from .inner import _move_tolerance

__all__ = ["ALMHistoryRecorder"]

_HISTORY_DIAGNOSTICS_SOURCE_KEY = "_constraint_history_diagnostics_source"


def _optional_array_to_float_list(values) -> Optional[List[float]]:
    return None if values is None else _as_float_list(values)


def _optional_float_array(evaluation: dict, key: str, fallback) -> Optional[np.ndarray]:
    values = evaluation.get(key)
    if values is None:
        values = fallback
    if values is None:
        return None
    return np.asarray(values, dtype=float).reshape(-1).copy()


def _optional_string_list(evaluation: dict, key: str) -> Optional[List[str]]:
    values = evaluation.get(key)
    if values is None:
        return None
    return [str(value) for value in values]


def _multiplier_interpretation(evaluation: dict) -> str:
    gradient_kinds = evaluation.get("gradient_value_kinds")
    dual_update_kinds = evaluation.get("dual_update_value_kinds")
    if gradient_kinds is None or dual_update_kinds is None:
        return "differentiable_alm_multipliers"
    for gradient_kind, dual_update_kind in zip(gradient_kinds, dual_update_kinds):
        if str(gradient_kind) != str(dual_update_kind):
            return "search_multipliers"
    return "differentiable_alm_multipliers"


class ALMHistoryRecorder:
    """Builds the ALM history from ``minimize_alm``'s outer-step events.

    Pass ``recorder.record`` as ``on_outer_step``; ``history()`` then returns
    the entries. ``history_callback(history, latest_entry, multipliers,
    penalty)`` runs after each entry. At most ``max_entries`` (None: all) of
    the latest entries are kept, and ``truncated_count`` counts the dropped
    ones. Every entry handed out is the caller's own copy, nested lists and
    dicts included; the recorder keeps an immutable snapshot of each entry.
    An entry's ``constraint_values`` is the signed g, as in ``ALMResult``,
    and ``violation_values`` the evaluator's per-row ``feasibility_values``.
    """

    def __init__(
        self,
        max_entries: Optional[int],
        history_callback: Optional[
            Callable[[List[dict], dict, np.ndarray, float], None]
        ] = None,
    ) -> None:
        max_entries = _finite_alm_integer_or_none(
            "ALMHistoryRecorder.max_entries", max_entries
        )
        out_of_range, requirement = _POSITIVE_WHEN_PROVIDED
        if max_entries is not None and out_of_range(max_entries):
            raise ValueError(f"ALMHistoryRecorder.max_entries {requirement}")
        self._max_entries = max_entries
        self._history_callback = history_callback
        # A pending dict (diagnostics source not yet materialized) or, once a
        # callback or history() needed it, its snapshot (_history_entry_snapshot).
        self._entries: List[Union[dict, bytes]] = []
        self._truncated_count = 0

    @classmethod
    def from_settings(
        cls,
        settings: ALMSettings,
        history_callback: Optional[
            Callable[[List[dict], dict, np.ndarray, float], None]
        ] = None,
    ) -> ALMHistoryRecorder:
        """A recorder keeping ``settings.history_max_entries`` entries: that
        setting's one consumer (``minimize_alm`` itself keeps no history)."""
        return cls(settings.history_max_entries, history_callback)

    def record(self, event: ALMOuterStepEvent) -> None:
        """Append the entry of one outer-step decision; run ``history_callback``."""
        self._entries.append(_history_entry_from_event(event))
        if self._max_entries is not None:
            excess_entries = len(self._entries) - self._max_entries
            if excess_entries > 0:
                self._truncated_count += excess_entries
                del self._entries[:excess_entries]
        if self._history_callback is None:
            return
        snapshots = self._snapshots()
        self._history_callback(
            [pickle.loads(snapshot) for snapshot in snapshots],
            pickle.loads(snapshots[-1]),
            event.after.multipliers.copy(),
            float(event.after.penalty),
        )

    def history(self) -> List[dict]:
        """This run's surviving entries so far, per-constraint diagnostics
        materialized, as the caller's own copies (nested lists and dicts too):
        later records and writes to a returned entry change nothing else."""
        return [pickle.loads(snapshot) for snapshot in self._snapshots()]

    @property
    def truncated_count(self) -> int:
        """How many of the oldest entries ``max_entries`` has dropped."""
        return self._truncated_count

    def _snapshots(self) -> List[bytes]:
        """The surviving entries' snapshots, each entry materialized once, in place."""
        self._entries[:] = [_history_entry_snapshot(entry) for entry in self._entries]
        return self._entries


def _history_entry_snapshot(entry: Union[dict, bytes]) -> bytes:
    """``entry`` as the recorder keeps it once needed: diagnostics materialized,
    then pickled. An entry holds plain data (numbers, strings, None, lists,
    dicts), so ``pickle.loads`` of the snapshot is an exact copy the caller owns,
    and the bytes cannot change. A snapshot is returned as is."""
    if isinstance(entry, bytes):
        return entry
    return pickle.dumps(
        _materialize_history_entry_diagnostics(entry), protocol=pickle.HIGHEST_PROTOCOL
    )


def _materialize_history_entry_diagnostics(entry: dict) -> dict:
    if _HISTORY_DIAGNOSTICS_SOURCE_KEY not in entry:
        return dict(entry)
    materialized = dict(entry)
    source = materialized.pop(_HISTORY_DIAGNOSTICS_SOURCE_KEY)
    materialized.update(_constraint_history_diagnostics_from_source(source))
    return materialized


def _history_entry_from_event(event: ALMOuterStepEvent) -> dict:
    """The history entry of one outer-step decision, keys in recorded order.

    Base fields describe the measured iterate; a penalty raise refreshes them
    to the re-evaluation at the new penalty. Decision annotations follow, then
    ``action``, ``trust_radius`` and ``outer_termination``.
    """
    measured = event.measured
    if event.inner is None:
        entry = _build_skipped_inner_history_entry(event)
    else:
        entry = _build_post_inner_history_entry(event)
    entry.update(_conditioning_metrics(measured.evaluation))
    if event.inner is not None:
        entry["sufficient_decrease_measure"] = event.inner.sufficient_decrease_measure
    entry[_HISTORY_DIAGNOSTICS_SOURCE_KEY] = _constraint_history_diagnostics_source(
        measured.evaluation,
        measured.multipliers,
        measured.penalty,
        event.constraint_names,
        measured.solver_constraint_values,
        measured.feasibility_values,
        measured.routing_state,
        measured.effective_feasibility_tol,
    )
    if event.dual_update is not None:
        entry["post_update_multipliers"] = _as_float_list(event.dual_update.multipliers)
        entry["multiplier_cap_binding"] = event.dual_update.multiplier_cap_binding
        entry["multiplier_cap_binding_indices"] = list(
            event.dual_update.multiplier_cap_binding_indices
        )
    if event.penalty_update is not None:
        _refresh_alm_history_for_penalty_update(
            entry, event.penalty_update, event.constraint_names
        )
    if event.signal_mismatch_repair:
        entry["signal_mismatch_continuation_repair"] = True
    if event.subproblem_limit_reason is not None or (
        event.action == "subproblem_continue"
    ):
        entry["subproblem_limit_reason"] = event.subproblem_limit_reason
    # A dual-update penalty raise that hits the cap never recorded its reason
    # (baseline behavior, pinned by the golden trajectories).
    if event.dual_update_penalty_reason is not None and (
        event.action != "penalty_cap_reached"
    ):
        entry["dual_update_penalty_increase"] = True
        entry["dual_update_penalty_increase_reason"] = event.dual_update_penalty_reason
    if event.action == "sufficient_decrease_hold":
        entry["penalty_hold_sufficient_decrease"] = True
        entry["penalty_hold_measure"] = event.inner.sufficient_decrease_measure
        entry["penalty_hold_reference_measure"] = (
            event.inner.sufficient_decrease_reference
        )
    entry["action"] = event.action
    entry["trust_radius"] = event.after.trust_radius
    if event.outer_termination is not None:
        entry["outer_termination"] = event.outer_termination
    return entry


def _build_skipped_inner_history_entry(event: ALMOuterStepEvent) -> dict:
    """The ``converged`` entry of the pre-inner shortcut, where the start
    iterate already satisfies the KKT gate: every inner-solve field takes its
    zero/None value."""
    start = event.start
    payload = _build_alm_history_entry(
        outer_iteration=event.outer_iteration,
        continuation_iteration=event.continuation_iteration,
        constraint_names=event.constraint_names,
        penalty=start.penalty,
        penalty_argument=start.penalty,
        multipliers=start.multipliers,
        post_update_multipliers=start.multipliers,
        routing_state=start.routing_state,
        feasibility_values=start.feasibility_values,
        solver_constraint_values=start.solver_constraint_values,
        max_feasibility_violation=start.max_feasibility_violation,
        stationarity_norm=start.stationarity_norm,
        raw_stationarity_norm=_augmented_stationarity_norm(start.evaluation),
        kkt_stationarity_norm=start.kkt_stationarity_norm,
        signal_mismatch_active=start.signal_mismatch_active,
        update_feasibility_tol=start.update_feasibility_tol,
        effective_feasibility_tol=start.effective_feasibility_tol,
        update_stationarity_tol=start.update_stationarity_tol,
        trust_radius=event.after.trust_radius,
        start_x=event.start_x,
        inner_iterations=0,
        inner_success=True,
        inner_message=(
            "ALM skipped inner solve; current iterate already satisfies the "
            "KKT stationarity gate."
        ),
        inner_maxiter=None,
        inner_maxls=None,
        inner_maxfun=None,
        inner_profile=None,
        inner_lbfgsb_projected_gradient_norm=None,
        inner_attempts=0,
        accepted_move_norm=0.0,
        objective_delta=0.0,
        feasibility_delta=0.0,
        feasibility_delta_tolerance=0.0,
        stationarity_delta=0.0,
        meaningful_progress=False,
        feasible_stall_count=0,
        infeasible_stall=False,
        inner_false_success=False,
        inner_stall_reason=None,
        active_violation_index=None,
        active_constraint_name=None,
        nonfinite_candidate_evaluation=False,
        nonfinite_candidate_fields=None,
    )
    payload["action"] = "converged"
    return payload


def _build_post_inner_history_entry(event: ALMOuterStepEvent) -> dict:
    """The entry of a step that ran the inner solve, before its decision's
    annotations. Inner options are the ones the last attempt ran with."""
    inner = event.inner
    measured = inner.measured
    last_used = inner.inner_options
    hard_violation_values = measured.routing_state.signal_state.hard_violation_values
    active_violation_index = (
        None
        if hard_violation_values.size == 0
        or measured.routing_state.hard_max_violation <= 0.0
        else int(np.argmax(hard_violation_values))
    )
    return _build_alm_history_entry(
        outer_iteration=event.outer_iteration,
        continuation_iteration=event.continuation_iteration,
        constraint_names=event.constraint_names,
        penalty=measured.penalty,
        penalty_argument=measured.penalty,
        multipliers=measured.multipliers,
        post_update_multipliers=measured.multipliers,
        routing_state=measured.routing_state,
        feasibility_values=measured.feasibility_values,
        solver_constraint_values=measured.solver_constraint_values,
        max_feasibility_violation=measured.max_feasibility_violation,
        stationarity_norm=measured.stationarity_norm,
        raw_stationarity_norm=_augmented_stationarity_norm(measured.evaluation),
        kkt_stationarity_norm=measured.kkt_stationarity_norm,
        signal_mismatch_active=measured.signal_mismatch_active,
        update_feasibility_tol=measured.update_feasibility_tol,
        effective_feasibility_tol=measured.effective_feasibility_tol,
        update_stationarity_tol=measured.update_stationarity_tol,
        trust_radius=event.after.trust_radius,
        start_x=event.start_x,
        inner_iterations=inner.iterations,
        inner_success=inner.optimizer_success,
        inner_message=inner.optimizer_message,
        inner_maxiter=None
        if last_used is None or "maxiter" not in last_used
        else int(last_used["maxiter"]),
        inner_maxls=None
        if last_used is None or "maxls" not in last_used
        else int(last_used["maxls"]),
        inner_maxfun=None
        if last_used is None or "maxfun" not in last_used
        else int(last_used["maxfun"]),
        inner_profile=inner.inner_profile,
        inner_lbfgsb_projected_gradient_norm=_lbfgsb_projected_gradient_max_norm(
            measured.evaluation["grad"],
            inner.x,
            inner.bounds,
        ),
        inner_attempts=inner.attempts,
        accepted_move_norm=float(np.linalg.norm(inner.x - event.start_x)),
        objective_delta=(
            float(event.start.evaluation["total"])
            - float(measured.evaluation["total"])
        ),
        feasibility_delta=inner.feasibility_delta,
        feasibility_delta_tolerance=inner.feasibility_delta_tolerance,
        stationarity_delta=(
            float(event.start.stationarity_norm) - float(measured.stationarity_norm)
        ),
        meaningful_progress=inner.meaningful_progress,
        feasible_stall_count=event.feasible_stall_count,
        infeasible_stall=inner.infeasible_stall,
        inner_false_success=inner.inner_false_success,
        inner_stall_reason=inner.infeasible_stall_reason,
        active_violation_index=active_violation_index,
        active_constraint_name=(
            None
            if active_violation_index is None
            else str(event.constraint_names[active_violation_index])
        ),
        nonfinite_candidate_evaluation=inner.nonfinite_candidate_evaluation,
        nonfinite_candidate_fields=(
            None
            if inner.nonfinite_candidate_fields is None
            else list(inner.nonfinite_candidate_fields)
        ),
    )


def _build_alm_history_entry(
    *,
    outer_iteration: int,
    continuation_iteration: int,
    constraint_names: Sequence[str],
    penalty: float,
    penalty_argument,
    multipliers: np.ndarray,
    post_update_multipliers: np.ndarray,
    routing_state,
    feasibility_values: np.ndarray,
    solver_constraint_values: np.ndarray,
    max_feasibility_violation: float,
    stationarity_norm: float,
    raw_stationarity_norm: float,
    kkt_stationarity_norm: Optional[float],
    signal_mismatch_active: bool,
    update_feasibility_tol: float,
    effective_feasibility_tol: float,
    update_stationarity_tol: float,
    trust_radius: Optional[float],
    start_x: np.ndarray,
    inner_iterations: int,
    inner_success: bool,
    inner_message: str,
    inner_maxiter: Optional[int],
    inner_maxls: Optional[int],
    inner_maxfun: Optional[int],
    inner_profile: Optional[str],
    inner_lbfgsb_projected_gradient_norm: Optional[float],
    inner_attempts: int,
    accepted_move_norm: float,
    objective_delta: float,
    feasibility_delta: float,
    feasibility_delta_tolerance: float,
    stationarity_delta: float,
    meaningful_progress: bool,
    feasible_stall_count: int,
    infeasible_stall: bool,
    inner_false_success: bool,
    inner_stall_reason: Optional[str],
    active_violation_index: Optional[int],
    active_constraint_name: Optional[str],
    nonfinite_candidate_evaluation: bool,
    nonfinite_candidate_fields: Optional[List[str]],
) -> dict:
    signal_state = routing_state.signal_state
    return {
        "outer_iteration": int(outer_iteration),
        "continuation_iteration": int(continuation_iteration),
        "constraint_names": [str(name) for name in constraint_names],
        "inner_iterations": int(inner_iterations),
        "inner_success": bool(inner_success),
        "inner_message": str(inner_message),
        "penalty": float(penalty),
        "penalty_values": _as_float_list(
            _penalty_values(penalty_argument, len(constraint_names))
        ),
        "block_penalties": None,
        "max_violation": float(max_feasibility_violation),
        "stationarity_norm": float(stationarity_norm),
        "raw_stationarity_norm": float(raw_stationarity_norm),
        "kkt_stationarity_norm": kkt_stationarity_norm,
        # Signed g, as ``ALMResult.constraint_values``; the clipped per-row
        # violation (the evaluator's feasibility_values) is violation_values.
        "constraint_values": _as_float_list(solver_constraint_values),
        "violation_values": _as_float_list(feasibility_values),
        "solver_constraint_values": _as_float_list(solver_constraint_values),
        "hard_signed_constraint_values": _as_float_list(
            signal_state.hard_signed_constraint_values
        ),
        "hard_violation_values": _as_float_list(signal_state.hard_violation_values),
        "surrogate_signed_constraint_values": _as_float_list(
            signal_state.surrogate_signed_constraint_values
        ),
        "hard_max_violation": float(routing_state.hard_max_violation),
        "surrogate_max_value": float(routing_state.surrogate_max_value),
        "hard_positive_shift_zero": bool(routing_state.hard_positive_shift_zero),
        "signal_mismatch_active": bool(signal_mismatch_active),
        "multipliers": _as_float_list(multipliers),
        "post_update_multipliers": _as_float_list(post_update_multipliers),
        "feasibility_tolerance": float(update_feasibility_tol),
        "effective_feasibility_tolerance": float(effective_feasibility_tol),
        "stationarity_tolerance": float(update_stationarity_tol),
        "trust_radius": trust_radius,
        "inner_maxiter": inner_maxiter,
        "inner_maxls": inner_maxls,
        "inner_maxfun": inner_maxfun,
        "inner_profile": inner_profile,
        "inner_lbfgsb_projected_gradient_norm": inner_lbfgsb_projected_gradient_norm,
        "inner_attempts": int(inner_attempts),
        "accepted_move_norm": float(accepted_move_norm),
        "accepted_move_norm_scaled": float(accepted_move_norm)
        / max(1.0, float(np.linalg.norm(start_x))),
        "infeasible_stall_move_tolerance": float(_move_tolerance(start_x)),
        "objective_delta": float(objective_delta),
        "feasibility_delta": float(feasibility_delta),
        "feasibility_delta_tolerance": float(feasibility_delta_tolerance),
        "stationarity_delta": float(stationarity_delta),
        "meaningful_progress": bool(meaningful_progress),
        "feasible_stall_count": int(feasible_stall_count),
        "infeasible_stall": bool(infeasible_stall),
        "inner_false_success": bool(inner_false_success),
        "inner_stall_reason": inner_stall_reason,
        "active_violation_index": active_violation_index,
        "active_constraint_name": active_constraint_name,
        "nonfinite_candidate_evaluation": bool(nonfinite_candidate_evaluation),
        "nonfinite_candidate_fields": nonfinite_candidate_fields,
        "multiplier_cap_binding": False,
        "multiplier_cap_binding_indices": [],
    }


def _refresh_alm_history_for_penalty_update(
    entry: dict,
    updated_state: ALMIterateMeasurement,
    constraint_names: Sequence[str],
) -> None:
    routing_state = updated_state.routing_state
    signal_state = routing_state.signal_state
    entry["penalty"] = float(updated_state.penalty)
    entry["penalty_values"] = _as_float_list(
        _penalty_values(updated_state.penalty, len(constraint_names))
    )
    entry["block_penalties"] = None
    entry["max_violation"] = float(updated_state.max_feasibility_violation)
    entry["stationarity_norm"] = float(updated_state.stationarity_norm)
    entry["raw_stationarity_norm"] = _augmented_stationarity_norm(updated_state.evaluation)
    entry["kkt_stationarity_norm"] = (
        None
        if updated_state.kkt_stationarity_norm is None
        else float(updated_state.kkt_stationarity_norm)
    )
    entry["constraint_values"] = _as_float_list(updated_state.solver_constraint_values)
    entry["violation_values"] = _as_float_list(updated_state.feasibility_values)
    entry["solver_constraint_values"] = _as_float_list(
        updated_state.solver_constraint_values
    )
    entry["hard_signed_constraint_values"] = _as_float_list(
        signal_state.hard_signed_constraint_values
    )
    entry["hard_violation_values"] = _as_float_list(signal_state.hard_violation_values)
    entry["surrogate_signed_constraint_values"] = _as_float_list(
        signal_state.surrogate_signed_constraint_values
    )
    entry["hard_max_violation"] = float(routing_state.hard_max_violation)
    entry["surrogate_max_value"] = float(routing_state.surrogate_max_value)
    entry["hard_positive_shift_zero"] = bool(routing_state.hard_positive_shift_zero)
    entry["signal_mismatch_active"] = bool(updated_state.signal_mismatch_active)
    entry["feasibility_tolerance"] = float(updated_state.update_feasibility_tol)
    entry["effective_feasibility_tolerance"] = float(
        updated_state.effective_feasibility_tol
    )
    entry["stationarity_tolerance"] = float(updated_state.update_stationarity_tol)
    entry.update(_conditioning_metrics(updated_state.evaluation))
    entry[_HISTORY_DIAGNOSTICS_SOURCE_KEY] = _constraint_history_diagnostics_source(
        updated_state.evaluation,
        updated_state.multipliers,
        updated_state.penalty,
        constraint_names,
        updated_state.solver_constraint_values,
        updated_state.feasibility_values,
        routing_state,
        updated_state.effective_feasibility_tol,
    )


def _constraint_history_diagnostics_source(
    evaluation: dict,
    multipliers: np.ndarray,
    penalty,
    constraint_names: Sequence[str],
    solver_constraint_values: np.ndarray,
    feasibility_values: np.ndarray,
    routing_state: ALMConstraintRoutingState,
    feasibility_gate: float,
) -> dict:
    multiplier_array = np.asarray(multipliers, dtype=float).reshape(-1).copy()
    penalty_values = _penalty_values(penalty, multiplier_array.size)
    solver_constraint_array = (
        np.asarray(solver_constraint_values, dtype=float).reshape(-1).copy()
    )
    feasibility_array = np.asarray(feasibility_values, dtype=float).reshape(-1).copy()
    positive_shift, augmented_terms = _positive_shift_and_augmented_terms(
        evaluation,
        multiplier_array,
        penalty_values,
        solver_constraint_array,
    )
    raw_hard_violation_values = _optional_float_array(
        evaluation,
        "raw_hard_violation_values",
        None,
    )
    return {
        "constraint_names": [str(name) for name in constraint_names],
        "feasibility_values": feasibility_array,
        "raw_signed_constraint_values": _optional_float_array(
            evaluation,
            "raw_constraint_values",
            evaluation.get("raw_surrogate_signed_constraint_values"),
        ),
        "normalized_signed_constraint_values": _optional_float_array(
            evaluation,
            "normalized_signed_constraint_values",
            solver_constraint_array,
        ),
        "raw_hard_violation_values": raw_hard_violation_values,
        "normalized_feasibility_values": _optional_float_array(
            evaluation,
            "normalized_feasibility_values",
            feasibility_array,
        ),
        "constraint_scales": _optional_float_array(
            evaluation,
            "constraint_scales",
            None,
        ),
        "constraint_blocks": _optional_string_list(evaluation, "constraint_blocks"),
        "normalized_multipliers": multiplier_array,
        "raw_dual_estimates": alm_raw_dual_estimates(multiplier_array, evaluation),
        "penalty_values": penalty_values,
        "positive_shift_values": (
            np.asarray(positive_shift, dtype=float).reshape(-1).copy()
        ),
        "augmented_term_by_constraint": (
            np.asarray(augmented_terms, dtype=float).reshape(-1).copy()
        ),
        "active_pressure_by_constraint": (
            np.asarray(positive_shift, dtype=float).reshape(-1)
            * solver_constraint_array
        ),
        "surrogate_minus_hard_normalized_gap": (
            np.asarray(
                routing_state.signal_state.surrogate_signed_constraint_values,
                dtype=float,
            ).reshape(-1)
            - np.asarray(
                routing_state.signal_state.hard_signed_constraint_values,
                dtype=float,
            ).reshape(-1)
        ),
        "surrogate_hard_sign_mismatch_by_constraint": _surrogate_hard_sign_mismatch(
            routing_state.signal_state.surrogate_signed_constraint_values,
            routing_state.signal_state.hard_signed_constraint_values,
        ),
        "objective_to_augmented_term_ratio": _objective_to_augmented_term_ratio(
            evaluation,
            augmented_terms,
        ),
        "augmented_gradient_norm": float(
            np.linalg.norm(np.asarray(evaluation["grad"], dtype=float))
        ),
        "surrogate_kkt_stationarity_norm": _surrogate_kkt_stationarity_norm(
            evaluation,
            routing_state,
            feasibility_gate,
        ),
        "multiplier_interpretation": _multiplier_interpretation(evaluation),
    }


def _constraint_history_diagnostics_from_source(source: dict) -> dict:
    raw_hard_violation_values = _optional_array_to_float_list(
        source["raw_hard_violation_values"]
    )
    raw_hard_max_violation = (
        None
        if raw_hard_violation_values is None
        else _max_value(np.asarray(raw_hard_violation_values, dtype=float))
    )
    block_diagnostics = _constraint_label_history_diagnostics(
        source["constraint_names"],
        source["constraint_blocks"],
        source["feasibility_values"],
        raw_hard_violation_values,
        source["positive_shift_values"],
        source["augmented_term_by_constraint"],
    )
    return {
        "raw_signed_constraint_values": _optional_array_to_float_list(
            source["raw_signed_constraint_values"]
        ),
        "normalized_signed_constraint_values": _as_float_list(
            source["normalized_signed_constraint_values"]
        ),
        "raw_hard_violation_values": raw_hard_violation_values,
        "normalized_feasibility_values": _as_float_list(
            source["normalized_feasibility_values"]
        ),
        "constraint_scales": _optional_array_to_float_list(source["constraint_scales"]),
        "constraint_blocks": source["constraint_blocks"],
        "normalized_multipliers": _as_float_list(source["normalized_multipliers"]),
        "raw_dual_estimates": source["raw_dual_estimates"],
        "penalty_values": _as_float_list(source["penalty_values"]),
        "positive_shift_values": _as_float_list(source["positive_shift_values"]),
        "augmented_term_by_constraint": _as_float_list(
            source["augmented_term_by_constraint"]
        ),
        "active_pressure_by_constraint": _as_float_list(
            source["active_pressure_by_constraint"]
        ),
        "surrogate_minus_hard_normalized_gap": _as_float_list(
            source["surrogate_minus_hard_normalized_gap"]
        ),
        "surrogate_hard_sign_mismatch_by_constraint": source[
            "surrogate_hard_sign_mismatch_by_constraint"
        ],
        "objective_to_augmented_term_ratio": source[
            "objective_to_augmented_term_ratio"
        ],
        "augmented_gradient_norm": source["augmented_gradient_norm"],
        "surrogate_kkt_stationarity_norm": source["surrogate_kkt_stationarity_norm"],
        "multiplier_interpretation": source["multiplier_interpretation"],
        "max_raw_hard_violation": raw_hard_max_violation,
        **block_diagnostics,
    }


def _constraint_label_history_diagnostics(
    constraint_names: Sequence[str],
    constraint_blocks: Optional[List[str]],
    feasibility_values: np.ndarray,
    raw_hard_violation_values: Optional[List[float]],
    positive_shift: np.ndarray,
    augmented_terms: np.ndarray,
) -> dict:
    if constraint_blocks is None:
        return {
            "block_max_raw_hard_violation": None,
            "block_max_normalized_violation": None,
            "block_augmented_term": None,
            "block_positive_shift_norm": None,
            "blocking_constraint_name": None,
            "blocking_constraint_block": None,
        }

    block_names = list(dict.fromkeys(constraint_blocks))
    block_index_by_name = {
        block_name: index for index, block_name in enumerate(block_names)
    }
    block_indices = np.fromiter(
        (block_index_by_name[block] for block in constraint_blocks),
        dtype=int,
        count=len(constraint_blocks),
    )

    block_max_normalized_violation_array = np.zeros(len(block_names), dtype=float)
    np.maximum.at(
        block_max_normalized_violation_array,
        block_indices,
        np.asarray(feasibility_values, dtype=float),
    )
    block_augmented_term_array = np.zeros(len(block_names), dtype=float)
    np.add.at(
        block_augmented_term_array,
        block_indices,
        np.asarray(augmented_terms, dtype=float),
    )
    block_shift_square_sum_array = np.zeros(len(block_names), dtype=float)
    np.add.at(
        block_shift_square_sum_array,
        block_indices,
        np.asarray(positive_shift, dtype=float) ** 2,
    )
    raw_hard_violation_array = (
        None
        if raw_hard_violation_values is None
        else np.asarray(raw_hard_violation_values, dtype=float)
    )
    if raw_hard_violation_array is None:
        block_max_raw_hard_violation = {}
    else:
        raw_hard_violation_by_block = np.zeros(len(block_names), dtype=float)
        np.maximum.at(
            raw_hard_violation_by_block,
            block_indices,
            raw_hard_violation_array,
        )
        block_max_raw_hard_violation = {
            block: float(value)
            for block, value in zip(block_names, raw_hard_violation_by_block)
        }

    blocking_constraint_name = None
    blocking_constraint_block = None
    if feasibility_values.size > 0 and _max_value(feasibility_values) > 0.0:
        blocking_index = int(np.argmax(feasibility_values))
        blocking_constraint_name = str(constraint_names[blocking_index])
        blocking_constraint_block = constraint_blocks[blocking_index]

    return {
        "block_max_raw_hard_violation": block_max_raw_hard_violation,
        "block_max_normalized_violation": {
            block: float(value)
            for block, value in zip(block_names, block_max_normalized_violation_array)
        },
        "block_augmented_term": {
            block: float(value)
            for block, value in zip(block_names, block_augmented_term_array)
        },
        "block_positive_shift_norm": {
            block: float(np.sqrt(value))
            for block, value in zip(block_names, block_shift_square_sum_array)
        },
        "blocking_constraint_name": blocking_constraint_name,
        "blocking_constraint_block": blocking_constraint_block,
    }


def _base_objective_value(evaluation: dict) -> Optional[float]:
    base_objective = evaluation.get("base_total")
    if base_objective is None:
        base_objective = evaluation.get("base_value")
    if base_objective is None:
        return None
    return float(base_objective)


def _objective_to_augmented_term_ratio(
    evaluation: dict,
    augmented_terms: np.ndarray,
) -> Optional[float]:
    if augmented_terms.size == 0:
        return None
    augmented_term = float(np.sum(augmented_terms))
    if augmented_term == 0.0:
        return None
    base_objective = _base_objective_value(evaluation)
    if base_objective is None:
        return None
    return abs(base_objective) / abs(augmented_term)


def _surrogate_hard_sign_mismatch(
    surrogate_signed_values: np.ndarray,
    hard_signed_values: np.ndarray,
) -> List[bool]:
    surrogate_values = np.asarray(surrogate_signed_values, dtype=float)
    hard_values = np.asarray(hard_signed_values, dtype=float)
    return (surrogate_values * hard_values < 0.0).tolist()


def _lbfgsb_projected_gradient_max_norm(
    gradient,
    x,
    bounds,
) -> float:
    projected_gradient = np.asarray(gradient, dtype=float).reshape(-1).copy()
    if bounds is not None:
        x_array = np.asarray(x, dtype=float).reshape(-1)
        for index, bound in enumerate(bounds):
            lower_bound, upper_bound = bound
            coordinate = float(x_array[index])
            gradient_value = float(projected_gradient[index])
            at_lower_bound = lower_bound is not None and coordinate <= float(
                lower_bound
            )
            at_upper_bound = upper_bound is not None and coordinate >= float(
                upper_bound
            )
            if (at_lower_bound and gradient_value > 0.0) or (
                at_upper_bound and gradient_value < 0.0
            ):
                projected_gradient[index] = 0.0
    return float(np.linalg.norm(projected_gradient, ord=np.inf))
