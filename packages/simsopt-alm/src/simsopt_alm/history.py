"""Opt-in ALM history: one entry per outer-step decision of :func:`minimize_alm`.

:class:`ALMHistoryRecorder` consumes the :class:`~.events.ALMOuterStepEvent`
stream and rebuilds, from the events alone, the per-step history that
drivers and artifact writers read: 53 shared keys per entry
(`_history_entry_from_event`), the conditioning metrics, the decision
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

    Base fields describe the measured iterate; a step without an inner solve
    (the pre-inner ``converged`` shortcut) gives every inner-solve field its
    zero/None value, and inner options are the ones the last attempt ran with.
    A penalty raise refreshes the measured fields, all but the multipliers,
    to its re-evaluation at the new penalty. Decision annotations follow, then
    ``action``, ``trust_radius`` and ``outer_termination``.
    """
    measured = event.measured
    inner = event.inner
    skipped = inner is None
    options = {} if skipped or inner.inner_options is None else inner.inner_options
    hard_violation_values = measured.routing_state.signal_state.hard_violation_values
    active_violation_index = (
        None
        if skipped
        or hard_violation_values.size == 0
        or measured.routing_state.hard_max_violation <= 0.0
        else int(np.argmax(hard_violation_values))
    )
    accepted_move_norm = 0.0 if skipped else float(np.linalg.norm(inner.x - event.start_x))
    entry = {
        "outer_iteration": int(event.outer_iteration),
        "continuation_iteration": int(event.continuation_iteration),
        "constraint_names": [str(name) for name in event.constraint_names],
        "inner_iterations": 0 if skipped else int(inner.iterations),
        "inner_success": True if skipped else bool(inner.optimizer_success),
        "inner_message": (
            "ALM skipped inner solve; current iterate already satisfies the "
            "KKT stationarity gate."
            if skipped
            else str(inner.optimizer_message)
        ),
        **_measured_history_fields(measured, event.constraint_names, with_multipliers=True),
        "trust_radius": event.after.trust_radius,
        "inner_maxiter": int(options["maxiter"]) if "maxiter" in options else None,
        "inner_maxls": int(options["maxls"]) if "maxls" in options else None,
        "inner_maxfun": int(options["maxfun"]) if "maxfun" in options else None,
        "inner_profile": None if skipped else inner.inner_profile,
        "inner_lbfgsb_projected_gradient_norm": (
            None
            if skipped
            else _lbfgsb_projected_gradient_max_norm(
                measured.evaluation["grad"], inner.x, inner.bounds
            )
        ),
        "inner_attempts": 0 if skipped else int(inner.attempts),
        "accepted_move_norm": accepted_move_norm,
        "accepted_move_norm_scaled": accepted_move_norm
        / max(1.0, float(np.linalg.norm(event.start_x))),
        "infeasible_stall_move_tolerance": float(_move_tolerance(event.start_x)),
        "objective_delta": 0.0
        if skipped
        else float(event.start.evaluation["total"]) - float(measured.evaluation["total"]),
        "feasibility_delta": 0.0 if skipped else float(inner.feasibility_delta),
        "feasibility_delta_tolerance": 0.0 if skipped else float(inner.feasibility_delta_tolerance),
        "stationarity_delta": 0.0
        if skipped
        else float(event.start.stationarity_norm) - float(measured.stationarity_norm),
        "meaningful_progress": False if skipped else bool(inner.meaningful_progress),
        "feasible_stall_count": 0 if skipped else int(event.feasible_stall_count),
        "infeasible_stall": False if skipped else bool(inner.infeasible_stall),
        "inner_false_success": False if skipped else bool(inner.inner_false_success),
        "inner_stall_reason": None if skipped else inner.infeasible_stall_reason,
        "active_violation_index": active_violation_index,
        "active_constraint_name": (
            None
            if active_violation_index is None
            else str(event.constraint_names[active_violation_index])
        ),
        "nonfinite_candidate_evaluation": (
            False if skipped else bool(inner.nonfinite_candidate_evaluation)
        ),
        "nonfinite_candidate_fields": (
            None
            if skipped or inner.nonfinite_candidate_fields is None
            else list(inner.nonfinite_candidate_fields)
        ),
        "multiplier_cap_binding": False,
        "multiplier_cap_binding_indices": [],
    }
    if skipped:
        # The shortcut's action takes its key position here.
        entry["action"] = "converged"
    entry.update(_conditioning_metrics(measured.evaluation))
    if not skipped:
        entry["sufficient_decrease_measure"] = inner.sufficient_decrease_measure
    entry[_HISTORY_DIAGNOSTICS_SOURCE_KEY] = _constraint_history_diagnostics_source(
        measured, event.constraint_names
    )
    if event.dual_update is not None:
        entry["post_update_multipliers"] = _as_float_list(event.dual_update.multipliers)
        entry["multiplier_cap_binding"] = event.dual_update.multiplier_cap_binding
        entry["multiplier_cap_binding_indices"] = list(
            event.dual_update.multiplier_cap_binding_indices
        )
    if event.penalty_update is not None:
        # Existing keys keep their positions.
        penalty_update = event.penalty_update
        entry.update(
            _measured_history_fields(
                penalty_update, event.constraint_names, with_multipliers=False
            )
        )
        entry.update(_conditioning_metrics(penalty_update.evaluation))
        entry[_HISTORY_DIAGNOSTICS_SOURCE_KEY] = _constraint_history_diagnostics_source(
            penalty_update, event.constraint_names
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
        entry["penalty_hold_measure"] = inner.sufficient_decrease_measure
        entry["penalty_hold_reference_measure"] = inner.sufficient_decrease_reference
    entry["action"] = event.action
    entry["trust_radius"] = event.after.trust_radius
    if event.outer_termination is not None:
        entry["outer_termination"] = event.outer_termination
    return entry


def _measured_history_fields(
    measured: ALMIterateMeasurement,
    constraint_names: Sequence[str],
    *,
    with_multipliers: bool,
) -> dict:
    """The entry fields of one measured iterate, in recorded order; the
    ``multipliers`` and ``post_update_multipliers`` pair (``with_multipliers``)
    sits before the tolerances."""
    routing_state = measured.routing_state
    signal_state = routing_state.signal_state
    multiplier_fields = (
        {
            "multipliers": _as_float_list(measured.multipliers),
            "post_update_multipliers": _as_float_list(measured.multipliers),
        }
        if with_multipliers
        else {}
    )
    return {
        "penalty": float(measured.penalty),
        "penalty_values": _as_float_list(
            _penalty_values(measured.penalty, len(constraint_names))
        ),
        "block_penalties": None,
        "max_violation": float(measured.max_feasibility_violation),
        "stationarity_norm": float(measured.stationarity_norm),
        "raw_stationarity_norm": _augmented_stationarity_norm(measured.evaluation),
        "kkt_stationarity_norm": (
            None
            if measured.kkt_stationarity_norm is None
            else float(measured.kkt_stationarity_norm)
        ),
        # Signed g, as ``ALMResult.constraint_values``; the clipped per-row
        # violation (the evaluator's feasibility_values) is violation_values.
        "constraint_values": _as_float_list(measured.solver_constraint_values),
        "violation_values": _as_float_list(measured.feasibility_values),
        "solver_constraint_values": _as_float_list(measured.solver_constraint_values),
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
        "signal_mismatch_active": bool(measured.signal_mismatch_active),
        **multiplier_fields,
        "feasibility_tolerance": float(measured.update_feasibility_tol),
        "effective_feasibility_tolerance": float(measured.effective_feasibility_tol),
        "stationarity_tolerance": float(measured.update_stationarity_tol),
    }


def _constraint_history_diagnostics_source(
    measured: ALMIterateMeasurement, constraint_names: Sequence[str]
) -> dict:
    """What the per-constraint diagnostics of ``measured`` are built from
    when first needed (:func:`_constraint_history_diagnostics_from_source`)."""
    evaluation = measured.evaluation
    routing_state = measured.routing_state
    multiplier_array = np.asarray(measured.multipliers, dtype=float).reshape(-1).copy()
    penalty_values = _penalty_values(measured.penalty, multiplier_array.size)
    solver_constraint_array = (
        np.asarray(measured.solver_constraint_values, dtype=float).reshape(-1).copy()
    )
    feasibility_array = (
        np.asarray(measured.feasibility_values, dtype=float).reshape(-1).copy()
    )
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
            measured.effective_feasibility_tol,
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
