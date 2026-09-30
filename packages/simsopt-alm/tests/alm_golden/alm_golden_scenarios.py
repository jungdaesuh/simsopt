"""Deterministic golden trajectories of :func:`simsopt_alm.minimize_alm`.

Each scenario calls the library solver directly with a synthetic evaluator
(numpy functions of ``x`` passed through ``augmented_inequality_objective``)
and records what a caller can observe:

* ``result``: every :class:`~simsopt_alm.ALMResult` field;
* ``history``: the :class:`~simsopt_alm.history.ALMHistoryRecorder`
  entries, one per outer step (runs that pass ``on_outer_step``);
* ``events``: the ordered calls the solver makes into caller code
  (``evaluate_problem``, ``inner_callback``, ``accepted_callback``,
  ``outer_state_callback``, ``on_outer_step``, the snapshot/restore pair of a
  stateful evaluator, ``on_outer_boundary``);
* ``checkpoints``: the :class:`~simsopt_alm.checkpoint.ALMTransitionSnapshot`
  of every outer boundary (runs that pass ``on_outer_boundary``).

Values are encoded exactly: a float becomes ``float.hex`` (``~inf``, ``~-inf``,
or ``~nan:<bits>``), an array keeps its dtype and shape, and a tuple stays
distinct from a list. ``generate_alm_golden.py`` writes the encodings; the
replay test re-runs each scenario and compares the encodings for equality, key
order included.

The inner solver is SciPy's L-BFGS-B, so a replay is bitwise only with the
numpy and SciPy builds, the machine and the OpenBLAS kernels the goldens were
recorded with (``manifest.json``); run single-threaded with those kernels
pinned before numpy loads (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OPENBLAS_CORETYPE=<manifest openblas_coretype>``). Each golden
also stores its coarse outcomes (:func:`scenario_outcomes`: actions,
termination and restore reasons, flags), which should not depend on the last
bits of the iterates, and :func:`scenario_boundary_values` reads its numeric
boundary values; the replay test checks both under any numpy and SciPy.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import pickle
import platform
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Optional

import numpy as np
import scipy
import scipy.linalg  # noqa: F401  (loads SciPy's OpenBLAS)
import threadpoolctl
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult

from simsopt_alm import (
    ALMPhysics,
    ALMSettings,
    augmented_inequality_objective,
    cached_alm_evaluator,
)
from simsopt_alm import control as alm_control
from simsopt_alm.checkpoint import alm_checkpointing
from simsopt_alm.control import ALMProcessBudgetExhausted
from simsopt_alm.history import ALMHistoryRecorder

FIXTURE_DIR = Path(__file__).resolve().parent
FIXTURE_FORMAT = "alm_library_golden_trajectory_v1"


# --------------------------------------------------------------------------
# Exact encoding
# --------------------------------------------------------------------------

_FLOAT_TOKEN_PREFIXES = ("0x", "-0x", "~")


def float_token(value: float) -> str:
    """``float.hex`` for a finite value; ``~inf``, ``~-inf``, ``~nan:<bits>`` otherwise."""
    value = float(value)
    if math.isnan(value):
        return "~nan:" + struct.pack(">d", value).hex()
    if math.isinf(value):
        return "~inf" if value > 0.0 else "~-inf"
    return value.hex()


def token_to_float(token: str) -> float:
    if token.startswith("~nan:"):
        return struct.unpack(">d", bytes.fromhex(token[5:]))[0]
    if token.startswith("~"):
        return float(token[1:])
    return float.fromhex(token)


def is_float_token(value: object) -> bool:
    return isinstance(value, str) and value.startswith(_FLOAT_TOKEN_PREFIXES)


def encode(value: object) -> object:
    """Exact, JSON-safe encoding of an ALM value (see the module docstring)."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return {"$str": value} if is_float_token(value) else value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float_token(float(value))
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        return {
            "$ndarray": value.dtype.str,
            "shape": list(value.shape),
            "data": [encode(item) for item in flat],
        }
    if isinstance(value, tuple):
        return {"$tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"$set": [encode(item) for item in sorted(value)]}
    if isinstance(value, OptimizeResult):
        return {"$OptimizeResult": _encode_mapping(value)}
    if isinstance(value, LbfgsInvHessProduct):
        return {
            "$LbfgsInvHessProduct": {
                "sk": encode(np.asarray(value.sk)),
                "yk": encode(np.asarray(value.yk)),
            }
        }
    if isinstance(value, dict):
        return _encode_mapping(value)
    if isinstance(value, MappingProxyType):
        return {"$mappingproxy": _encode_mapping(value)}
    if isinstance(value, SimpleNamespace):
        return {"$namespace": _encode_mapping(vars(value))}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": type(value).__name__,
            "fields": {
                item.name: encode(getattr(value, item.name))
                for item in dataclasses.fields(value)
            },
        }
    raise TypeError(f"no golden encoding for {type(value).__name__}")


def _encode_mapping(mapping: Mapping) -> dict:
    encoded = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"golden mappings need string keys, got {key!r}")
        encoded[key] = encode(item)
    return encoded


def encoded_digest(value: object) -> str:
    """sha256 of the canonical JSON text of ``encode(value)`` (key order kept)."""
    text = json.dumps(encode(value), separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_difference(expected: object, actual: object, path: str = "$") -> Optional[str]:
    """The first path where two encodings differ, dict key order included."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys, actual_keys = list(expected), list(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}: keys differ (order counts)\n"
                f"  missing: {[k for k in expected_keys if k not in actual]}\n"
                f"  extra:   {[k for k in actual_keys if k not in expected]}\n"
                f"  expected order: {expected_keys}\n"
                f"  actual order:   {actual_keys}"
            )
        for key in expected_keys:
            found = first_difference(expected[key], actual[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(actual)} != golden {len(expected)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if type(expected) is not type(actual) or expected != actual:
        return f"{path}: {_describe(actual)} != golden {_describe(expected)}"
    return None


def _describe(value: object) -> str:
    if is_float_token(value):
        return f"{token_to_float(value)!r} ({value})"
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "..."


# --------------------------------------------------------------------------
# Recording harness
# --------------------------------------------------------------------------


class TrajectoryRecorder:
    """Ordered log of every call the solver makes into caller code."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.checkpoints: list[dict] = []

    def record(self, event: str, **payload: object) -> None:
        self.events.append(
            {"event": event, **{key: encode(value) for key, value in payload.items()}}
        )

    def wrap_evaluate(self, evaluate_problem):
        def evaluate(x, multipliers, penalty):
            call = dict(
                x=np.asarray(x, dtype=float).copy(),
                multipliers=np.asarray(multipliers, dtype=float).copy(),
                penalty=float(penalty),
            )
            evaluation = evaluate_problem(x, multipliers, penalty)
            # The evaluator calls no recorded callback, so recording after it
            # keeps the call order; the solver must survive a nonfinite total.
            self.record(
                "evaluate_problem",
                **call,
                total_finite=bool(np.isfinite(evaluation["total"])),
            )
            return evaluation

        return evaluate

    def inner_callback(self, x) -> None:
        self.record("inner_callback", x=np.asarray(x, dtype=float).copy())

    def outer_state_callback(self, outer_iteration, multipliers, penalty) -> None:
        self.record(
            "outer_state_callback",
            outer_iteration=int(outer_iteration),
            multipliers=np.asarray(multipliers, dtype=float).copy(),
            penalty=float(penalty),
        )

    def outer_step(self, event) -> None:
        # The full event is pinned by digest to keep fixtures small; the
        # history entries built from the same events are recorded in full.
        self.record(
            "on_outer_step",
            outer_iteration=int(event.outer_iteration),
            continuation_iteration=int(event.continuation_iteration),
            action=event.action,
            dual_update_penalty_reason=event.dual_update_penalty_reason,
            outer_termination=event.outer_termination,
            event_sha256=encoded_digest(event),
        )

    def outer_boundary(self, snapshot) -> None:
        self.record(
            "on_outer_boundary",
            completed_outer_iterations=int(snapshot.completed_outer_iterations),
            completed_action=snapshot.completed_action,
            termination_reason=snapshot.termination_reason,
            resume_eligible=bool(snapshot.resume_eligible),
        )
        self.checkpoints.append(encode(snapshot))


# --------------------------------------------------------------------------
# Observable outcomes
# --------------------------------------------------------------------------

HISTORY_FLAGS = (
    "dual_update_penalty_increase",
    "infeasible_stall",
    "inner_false_success",
    "multiplier_cap_binding",
    "nonfinite_candidate_evaluation",
    "signal_mismatch_active",
)


# The termination reasons of a run its spent inner maxiter budget ended: the
# latest step's action, one that ends a step without returning (the "Inner
# budget spent" table of the skill's termination.md). A best-feasible
# restore renames it max_outer_restored_best_feasible, which the outer limit
# returns too.
BUDGET_SPENT_TERMINATIONS = frozenset((
    "dual_update",
    "penalty_increase",
    "sufficient_decrease_hold",
    "infeasible_stall_penalty_increase",
    "subproblem_limit_penalty_increase",
    "signal_mismatch_penalty_increase",
    "signal_mismatch_subproblem_limit_penalty_increase",
    "subproblem_continue",
))


def _decoded_field(result: dict, name: str):
    value = result["fields"][name]
    return token_to_float(value) if is_float_token(value) else value


def run_outcomes(trajectory: dict) -> set[str]:
    """The coarse outcomes of one recorded run: the callbacks it received, its
    outer-step actions, a dual update that needed a penalty raise, a nonfinite
    evaluation it survived, the ``HISTORY_FLAGS`` it raised, an inner retry,
    its termination and restore reasons, and a spent inner budget."""
    outcomes = set()
    for event in trajectory["events"]:
        outcomes.add(f"callback:{event['event']}")
        if event["event"] == "on_outer_step":
            outcomes.add(f"action:{event['action']}")
            if event["dual_update_penalty_reason"] is not None:
                outcomes.add("dual_update_penalty_raise")
        if event["event"] == "evaluate_problem" and not event["total_finite"]:
            outcomes.add("nonfinite_evaluation")
    for entry in trajectory.get("history", []):
        outcomes.update(f"flag:{key}" for key in HISTORY_FLAGS if entry.get(key) is True)
        if entry["inner_attempts"] > 1:
            outcomes.add("inner_attempts_retried")
    result = trajectory["result"]
    if result is not None:
        outcomes.add(f"termination:{_decoded_field(result, 'termination_reason')}")
        if _decoded_field(result, "restored_best_feasible"):
            outcomes.add(
                f"restored:{_decoded_field(result, 'restored_best_feasible_reason')}"
            )
        if _decoded_field(result, "termination_reason") in BUDGET_SPENT_TERMINATIONS:
            outcomes.add("inner_maxiter_budget_spent")
    return outcomes


def scenario_outcomes(trajectory: dict) -> set[str]:
    """``run_outcomes`` of every run of a scenario, plus ``resumed`` and
    ``smoothing_changed`` when the scenario did so."""
    if "resumed" in trajectory:
        return (
            run_outcomes(trajectory["interrupted"])
            | run_outcomes(trajectory["resumed"])
            | {"resumed"}
        )
    outcomes = run_outcomes(trajectory)
    if trajectory.get("smoothing_changes"):
        outcomes.add("smoothing_changed")
    return outcomes


# --------------------------------------------------------------------------
# Numeric boundary values
# --------------------------------------------------------------------------

BOUNDARY_QUANTITIES = ("x", "objective", "max_violation", "multipliers", "penalty")


def _encoded_floats(value: object) -> Optional[tuple[float, ...]]:
    """The floats of an encoded scalar, tuple, list or ndarray; ``None`` stays."""
    if value is None:
        return None
    if is_float_token(value):
        return (token_to_float(value),)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    if isinstance(value, dict) and "$tuple" in value:
        value = value["$tuple"]
    elif isinstance(value, dict) and "$ndarray" in value:
        value = value["data"]
    if isinstance(value, list):
        return tuple(item for element in value for item in _encoded_floats(element))
    raise TypeError(f"no float view of {_describe(value)}")


def run_boundary_values(trajectory: dict, run: str = "run") -> dict[str, list]:
    """``{quantity: [(location, floats), ...]}`` of one recorded run, for each
    of ``BOUNDARY_QUANTITIES`` at every outer step (history entry), every outer
    boundary (checkpoint) and the result. ``floats`` is ``None`` where the run
    recorded none (a history entry without a base objective)."""
    values = {quantity: [] for quantity in BOUNDARY_QUANTITIES}

    def add(location, **quantities):
        for quantity, value in quantities.items():
            values[quantity].append((location, _encoded_floats(value)))

    for index, entry in enumerate(trajectory.get("history", [])):
        add(
            f"{run}.history[{index}]",
            objective=entry["conditioning_base_objective"],
            max_violation=entry["max_violation"],
            multipliers=entry["post_update_multipliers"],
            penalty=entry["penalty"],
        )
    for index, checkpoint in enumerate(trajectory["checkpoints"]):
        fields = checkpoint["fields"]
        add(
            f"{run}.checkpoints[{index}]",
            x=fields["x"],
            multipliers=fields["multipliers"],
            penalty=fields["penalty"],
        )
    result = trajectory["result"]
    if result is not None:
        fields = result["fields"]
        add(
            f"{run}.result",
            **{quantity: fields[quantity] for quantity in BOUNDARY_QUANTITIES},
        )
    return values


def scenario_boundary_values(trajectory: dict) -> dict[str, list]:
    """``run_boundary_values`` of every run of a scenario, in run order."""
    if "resumed" not in trajectory:
        return run_boundary_values(trajectory)
    interrupted = run_boundary_values(trajectory["interrupted"], "interrupted")
    resumed = run_boundary_values(trajectory["resumed"], "resumed")
    return {quantity: interrupted[quantity] + resumed[quantity] for quantity in BOUNDARY_QUANTITIES}


def boundary_deviations(expected: dict, actual: dict) -> tuple[Optional[str], dict]:
    """``(structural_difference, {quantity: max deviation})`` of two
    ``scenario_boundary_values``.

    The deviation of a value is ``|actual - expected| / max(1, |expected|)``:
    relative for large values, absolute below 1 (multipliers and violations
    that are exactly 0 at the recorded point). A structural difference (other
    boundaries, other lengths, ``None`` versus a value, non-finite values that
    differ) means the trajectory branched and deviations are not comparable.
    """
    deviations = {}
    for quantity in BOUNDARY_QUANTITIES:
        left, right = expected[quantity], actual[quantity]
        if [location for location, _ in left] != [location for location, _ in right]:
            return f"{quantity}: boundaries {len(right)} != golden {len(left)}", {}
        worst = 0.0
        for (location, golden_floats), (_, floats) in zip(left, right):
            if golden_floats is None or floats is None:
                if golden_floats is not floats:
                    return f"{location}.{quantity}: {floats} != golden {golden_floats}", {}
                continue
            if len(golden_floats) != len(floats):
                return f"{location}.{quantity}: length {len(floats)} != golden {len(golden_floats)}", {}
            for golden_value, value in zip(golden_floats, floats):
                if not (math.isfinite(golden_value) and math.isfinite(value)):
                    if struct.pack(">d", golden_value) != struct.pack(">d", value) and not (
                        math.isnan(golden_value) and math.isnan(value)
                    ):
                        return f"{location}.{quantity}: {value} != golden {golden_value}", {}
                    continue
                worst = max(worst, abs(value - golden_value) / max(1.0, abs(golden_value)))
        deviations[quantity] = worst
    return None, deviations


RESTORE_REASONS = frozenset(
    ("final_iterate_infeasible", "final_iterate_worse_than_best_feasible")
)


def result_invariant_violations(trajectory: RecordedTrajectory) -> list[str]:
    """What a fresh scenario run's final result breaks of the properties every
    ``ALMResult`` must keep, whatever path the run took: finite values,
    nonnegative multipliers, a penalty in (0, penalty_max], a restore flag
    that agrees with its reason, a ``*_restored_best_feasible`` termination
    only after a restore, and ``success`` or a restore only at a feasible
    point (``max_violation <= feasibility_tol``)."""
    if "resumed" in trajectory:
        trajectory = trajectory["resumed"]
    settings = trajectory.settings
    result = trajectory["result"]
    if result is None:
        return ["the run returned no result"]
    fields = {name: _decoded_field(result, name) for name in result["fields"]}
    violations = []
    for name in ("x", "objective", "max_violation", "multipliers", "penalty", "constraint_values"):
        floats = _encoded_floats(result["fields"][name])
        if floats is None or not all(math.isfinite(value) for value in floats):
            violations.append(f"{name} is not finite: {floats}")
    if any(value < 0.0 for value in _encoded_floats(result["fields"]["multipliers"])):
        violations.append("a multiplier is negative")
    penalty_max = math.inf if settings.penalty_max is None else settings.penalty_max
    if not 0.0 < fields["penalty"] <= penalty_max:
        violations.append(f"penalty {fields['penalty']} outside (0, {penalty_max}]")
    reason = fields["restored_best_feasible_reason"]
    restored = fields["restored_best_feasible"]
    if restored != (reason is not None):
        violations.append(f"restored_best_feasible={restored} with reason {reason!r}")
    if reason is not None and not set(reason.split(",")) <= RESTORE_REASONS:
        violations.append(f"unknown restore reason {reason!r}")
    if fields["termination_reason"].endswith("_restored_best_feasible") and not restored:
        violations.append(f"{fields['termination_reason']} without a restore")
    feasible = fields["max_violation"] <= settings.feasibility_tol
    if fields["success"] and not feasible:
        violations.append(f"success at max_violation {fields['max_violation']}")
    if restored and not feasible:
        violations.append(f"restored an infeasible point, max_violation {fields['max_violation']}")
    return violations


# --------------------------------------------------------------------------
# Synthetic physics
# --------------------------------------------------------------------------


def generic_evaluation(physics: ALMPhysics, multipliers, penalty) -> dict:
    return augmented_inequality_objective(
        physics.base_value,
        physics.base_grad,
        physics.constraint_values,
        list(physics.constraint_grads),
        multipliers,
        penalty,
    )


def hybrid_evaluation(
    surrogate: ALMPhysics,
    hard_signed_values: np.ndarray,
    multipliers,
    penalty,
    *,
    activity_tolerances: Optional[np.ndarray] = None,
) -> dict:
    """A smoothed evaluator's hybrid quartet: L on the surrogate g, the dual
    update and feasibility on the hard g."""
    evaluation = generic_evaluation(surrogate, multipliers, penalty)
    hard = np.asarray(hard_signed_values, dtype=float)
    hard_violation = np.maximum(hard, 0.0)
    evaluation["dual_update_values"] = hard.copy()
    evaluation["feasibility_values"] = hard_violation.copy()
    evaluation["max_feasibility_violation"] = float(np.max(hard_violation))
    evaluation["hard_signed_constraint_values"] = hard.copy()
    evaluation["hard_violation_values"] = hard_violation.copy()
    evaluation["surrogate_signed_constraint_values"] = np.asarray(
        surrogate.constraint_values, dtype=float
    ).copy()
    evaluation["hard_dual_update_values"] = hard.copy()
    if activity_tolerances is not None:
        evaluation["constraint_activity_tolerances"] = np.asarray(
            activity_tolerances, dtype=float
        ).copy()
    return evaluation


def toy_convex_evaluate(x, multipliers, penalty) -> dict:
    """min (x0-2)^2 + (x1-1)^2 s.t. x0 + x1 <= 2, x0 >= 0; x* = (1.5, 0.5), lambda* = (1, 0)."""
    x = np.asarray(x, dtype=float)
    physics = ALMPhysics(
        base_value=(x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        base_grad=np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        constraint_values=np.array([x[0] + x[1] - 2.0, -x[0]]),
        constraint_grads=(np.array([1.0, 1.0]), np.array([-1.0, 0.0])),
    )
    return generic_evaluation(physics, multipliers, penalty)


TOY_CONSTRAINTS = ("sum_cap", "x0_floor")


def far_target_physics(x, *, nan_above: Optional[float] = None) -> ALMPhysics:
    """min (x0-10)^2 + (x1-1)^2 s.t. x0 <= 1: the objective pulls far outside.

    ``nan_above`` makes the objective NaN past that x0 (a failed physics
    evaluation that the inner solve must sanitize)."""
    x = np.asarray(x, dtype=float)
    base_value = (x[0] - 10.0) ** 2 + (x[1] - 1.0) ** 2
    if nan_above is not None and x[0] > nan_above:
        base_value = float("nan")
    return ALMPhysics(
        base_value=base_value,
        base_grad=np.array([2.0 * (x[0] - 10.0), 2.0 * (x[1] - 1.0)]),
        constraint_values=np.array([x[0] - 1.0]),
        constraint_grads=(np.array([1.0, 0.0]),),
    )


def far_target_evaluate(x, multipliers, penalty) -> dict:
    return generic_evaluation(far_target_physics(x), multipliers, penalty)


def far_target_nan_evaluate(x, multipliers, penalty) -> dict:
    return generic_evaluation(far_target_physics(x, nan_above=5.0), multipliers, penalty)


FAR_TARGET_CONSTRAINTS = ("x0_cap",)


def hybrid_band_evaluate(*, surrogate_bias: float, activity_tolerance: Optional[float]):
    """Hard g = x0 + (x1-1)/2 - 1; the smoothed surrogate is g + bias.

    With bias > 0 the band -bias < g <= 0 is hard-feasible but
    surrogate-active; with bias < 0 and an activity tolerance the hard row is
    active while the surrogate row is not."""

    def evaluate(x, multipliers, penalty):
        x = np.asarray(x, dtype=float)
        hard = np.array([x[0] + 0.5 * (x[1] - 1.0) - 1.0])
        surrogate = ALMPhysics(
            base_value=(x[0] - 10.0) ** 2 + (x[1] - 1.0) ** 2,
            base_grad=np.array([2.0 * (x[0] - 10.0), 2.0 * (x[1] - 1.0)]),
            constraint_values=hard + surrogate_bias,
            constraint_grads=(np.array([1.0, 0.5]),),
        )
        return hybrid_evaluation(
            surrogate,
            hard,
            multipliers,
            penalty,
            activity_tolerances=(
                None if activity_tolerance is None else np.array([activity_tolerance])
            ),
        )

    return evaluate


HYBRID_CONSTRAINTS = ("gap",)


# --------------------------------------------------------------------------
# A stateful evaluator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptedIterate:
    """What the stateful evaluator snapshots: the accepted x."""

    x: tuple[float, ...]


@dataclass
class AcceptedStateGate:
    """A stateful evaluator: it tracks the accepted x and gates trial steps.

    ``accepted_x`` follows ``accepted_callback`` and restores. While
    ``step_norm_limit`` is set, a trial point farther than it from
    ``accepted_x`` is rejected (evaluated at ``accepted_x`` with an elevated
    total and ``search_step_success=False``), the way an evaluator with a
    warm-started inner solve rejects a step it cannot follow.
    ``step_norm_limits_by_outer`` changes the limit when an outer iteration
    starts (``outer_state_callback``).
    """

    accepted_x: np.ndarray
    step_norm_limit: Optional[float] = None
    step_norm_limits_by_outer: Mapping[int, Optional[float]] = field(
        default_factory=dict
    )

    def gate(self, evaluate_at: Callable[[np.ndarray, np.ndarray, float], dict]):
        def evaluate(x, multipliers, penalty):
            x_array = np.asarray(x, dtype=float)
            if self.step_norm_limit is not None:
                step = float(np.linalg.norm(x_array - self.accepted_x))
                if step > self.step_norm_limit:
                    rejected = evaluate_at(self.accepted_x, multipliers, penalty)
                    rejected["total"] = float(rejected["total"]) + max(
                        abs(float(rejected["total"])), 1.0
                    )
                    rejected["search_step_success"] = False
                    return rejected
            return evaluate_at(x_array, multipliers, penalty)

        return evaluate

    def on_outer(self, outer_iteration: int) -> None:
        if outer_iteration in self.step_norm_limits_by_outer:
            self.step_norm_limit = self.step_norm_limits_by_outer[outer_iteration]

    def snapshot(self) -> AcceptedIterate:
        return AcceptedIterate(x=tuple(float(value) for value in self.accepted_x))

    def restore(self, state: AcceptedIterate) -> None:
        self.accepted_x = np.asarray(state.x, dtype=float).copy()


# --------------------------------------------------------------------------
# Scenario runner
# --------------------------------------------------------------------------


OPTIONAL_CALLBACKS = frozenset(
    (
        "inner_callback",
        "accepted_callback",
        "outer_state_callback",
        "on_outer_step",
        "on_outer_boundary",
    )
)


@dataclass(frozen=True)
class ScenarioRun:
    """Inputs of one ``minimize_alm`` call.

    ``callbacks`` names the optional callbacks passed (each is recorded).
    ``gate`` makes the evaluator stateful: its snapshot/restore pair is passed
    and it gates trial steps. ``resume_state`` resumes from a checkpoint;
    ``pause_after_outer`` interrupts the run at that outer boundary.
    ``outer_step_hook`` receives each history entry after it is recorded.
    """

    x0: np.ndarray
    constraint_names: tuple[str, ...]
    evaluate_problem: Callable[[np.ndarray, np.ndarray, float], dict]
    settings: ALMSettings
    inner_options: dict
    gate: Optional[AcceptedStateGate] = None
    outer_step_hook: Optional[Callable[[dict], None]] = None
    accepted_hook: Optional[Callable[[np.ndarray], None]] = None
    initial_multipliers: Optional[np.ndarray] = None
    initial_penalty: Optional[float] = None
    constraint_blocks: Optional[tuple[str, ...]] = None
    base_bounds: Optional[tuple] = None
    resume_state: Optional[object] = None
    pause_after_outer: Optional[int] = None
    callbacks: frozenset[str] = OPTIONAL_CALLBACKS

    def __post_init__(self) -> None:
        unknown = self.callbacks - OPTIONAL_CALLBACKS
        if unknown:
            raise ValueError(f"unknown callbacks {sorted(unknown)}")
        required = set()
        if self.gate is not None:
            required |= {"accepted_callback", "outer_state_callback"}
        if self.accepted_hook is not None:
            required.add("accepted_callback")
        if self.outer_step_hook is not None:
            required.add("on_outer_step")
        if self.pause_after_outer is not None:
            required.add("on_outer_boundary")
        missing = required - self.callbacks
        if missing:
            raise ValueError(f"scenario needs callbacks {sorted(missing)}")


class PauseAfterBoundary(Exception):
    """Raised from the boundary callback to interrupt a run at a boundary."""


class RecordedTrajectory(dict):
    """A fresh run's trajectory (the dict a golden stores) that also carries
    the run's ``ALMSettings``, which the fixture does not store, for the result
    invariants (:func:`result_invariant_violations`)."""

    def __init__(self, trajectory: dict, settings: ALMSettings) -> None:
        super().__init__(trajectory)
        self.settings = settings


@dataclass(frozen=True)
class RunRecord:
    trajectory: dict
    result: Optional[object]
    snapshots: tuple


def execute(run: ScenarioRun) -> RunRecord:
    """Run ``minimize_alm`` once, recording every caller-visible call."""
    recorder = TrajectoryRecorder()
    gate = run.gate
    evaluate = run.evaluate_problem if gate is None else gate.gate(run.evaluate_problem)
    history = ALMHistoryRecorder.from_settings(run.settings)
    snapshots: list = []

    def accepted_callback(x):
        recorder.record("accepted_callback", x=np.asarray(x, dtype=float).copy())
        if gate is not None:
            gate.accepted_x = np.asarray(x, dtype=float).copy()
        if run.accepted_hook is not None:
            run.accepted_hook(x)

    def outer_state_callback(outer_iteration, multipliers, penalty):
        recorder.outer_state_callback(outer_iteration, multipliers, penalty)
        if gate is not None:
            gate.on_outer(int(outer_iteration))

    def on_outer_step(event):
        recorder.outer_step(event)
        history.record(event)
        if run.outer_step_hook is not None:
            run.outer_step_hook(history.history()[-1])

    def snapshot_accepted_state():
        recorder.record("snapshot_accepted_state", x=gate.accepted_x.copy())
        return gate.snapshot()

    def restore_incumbent_state(state):
        recorder.record("restore_incumbent_state", x=np.asarray(state.x).copy())
        gate.restore(state)

    def completed_outer_callback(snapshot):
        snapshots.append(snapshot)
        recorder.outer_boundary(snapshot)
        if snapshot.completed_outer_iterations == run.pause_after_outer:
            raise PauseAfterBoundary()

    checkpointing = alm_checkpointing(
        dict(run.inner_options),
        run.resume_state,
        completed_outer_callback if "on_outer_boundary" in run.callbacks else None,
    )
    optional = {
        "inner_callback": recorder.inner_callback,
        "accepted_callback": accepted_callback,
        "outer_state_callback": outer_state_callback,
        "on_outer_step": on_outer_step,
        "on_outer_boundary": checkpointing.on_outer_boundary,
    }
    kwargs = {name: optional[name] for name in sorted(run.callbacks)}
    if gate is not None:
        kwargs["snapshot_accepted_state_fn"] = snapshot_accepted_state
        kwargs["restore_incumbent_state_fn"] = restore_incumbent_state
    result = None
    try:
        # Looked up at call time, so a test can observe each library call.
        result = alm_control.minimize_alm(
            np.asarray(run.x0, dtype=float).copy(),
            list(run.constraint_names),
            recorder.wrap_evaluate(evaluate),
            run.settings,
            checkpointing.inner_options,
            initial_multipliers=run.initial_multipliers,
            initial_penalty=run.initial_penalty,
            constraint_blocks=run.constraint_blocks,
            base_bounds=run.base_bounds,
            resume_from=checkpointing.resume_from,
            **kwargs,
        )
    except PauseAfterBoundary:
        if run.pause_after_outer is None:
            raise
    trajectory = {"result": None if result is None else encode(result)}
    if "on_outer_step" in run.callbacks:
        trajectory["history"] = encode(history.history())
    trajectory["events"] = recorder.events
    trajectory["checkpoints"] = recorder.checkpoints
    return RunRecord(
        trajectory=RecordedTrajectory(trajectory, run.settings),
        result=result,
        snapshots=tuple(snapshots),
    )


def interrupted_then_resumed(make_run, pause_after_outer: int) -> dict:
    """Stop a run after an outer boundary, write its checkpoint (a pickle
    round trip, as a checkpoint file would), and resume it in a fresh run."""
    interrupted = execute(make_run(pause_after_outer=pause_after_outer))
    if interrupted.result is not None:
        raise AssertionError("the interrupted run finished before its pause boundary")
    resume_state = pickle.loads(pickle.dumps(interrupted.snapshots[-1]))
    resumed = execute(make_run(resume_state=resume_state))
    return {
        "interrupted": interrupted.trajectory,
        "resumed": resumed.trajectory,
    }


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

BOXED_INNER_OPTIONS = MappingProxyType(
    {"maxiter": 200, "maxcor": 10, "ftol": 1.0e-12, "gtol": 1.0e-12}
)


def _inner_options(maxiter: int) -> dict:
    return {**BOXED_INNER_OPTIONS, "maxiter": int(maxiter)}


def _resume_x0(default: np.ndarray, resume_state) -> np.ndarray:
    if resume_state is None:
        return default
    return np.asarray(resume_state.x, dtype=float)


def run_toy_convex() -> dict:
    settings = ALMSettings(
        max_outer_iterations=12,
        max_subproblem_continuations=2,
        penalty_init=10.0,
        penalty_scale=10.0,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
    )
    return execute(
        ScenarioRun(
            x0=np.zeros(2),
            constraint_names=TOY_CONSTRAINTS,
            evaluate_problem=toy_convex_evaluate,
            settings=settings,
            inner_options={"maxiter": 200},
            callbacks=frozenset(),
        )
    ).trajectory


def _penalty_ramp_run(*, resume_state=None, pause_after_outer=None) -> ScenarioRun:
    settings = ALMSettings(
        max_outer_iterations=14,
        max_subproblem_continuations=0,
        penalty_init=1.0,
        penalty_scale=10.0,
        penalty_max=500.0,
        feasibility_tol=1.0e-9,
        stationarity_tol=1.0e-6,
    )
    x0 = _resume_x0(np.array([8.0, 0.0]), resume_state)
    return ScenarioRun(
        x0=x0,
        constraint_names=FAR_TARGET_CONSTRAINTS,
        evaluate_problem=far_target_evaluate,
        settings=settings,
        inner_options=_inner_options(200),
        gate=AcceptedStateGate(accepted_x=x0.copy()),
        resume_state=resume_state,
        pause_after_outer=pause_after_outer,
    )


def run_penalty_ramp_to_cap() -> dict:
    return execute(_penalty_ramp_run()).trajectory


def run_resume_penalty_ramp_mid_run() -> dict:
    return interrupted_then_resumed(_penalty_ramp_run, pause_after_outer=4)


def _hybrid_band_run(
    *,
    surrogate_bias: float,
    activity_tolerance: Optional[float],
    continue_on_signal_mismatch: bool,
    max_outer_iterations: int,
    x0: tuple[float, float],
    trust_radius_init: Optional[float],
    step_norm_limit: Optional[float] = None,
) -> ScenarioRun:
    settings = ALMSettings(
        max_outer_iterations=max_outer_iterations,
        max_subproblem_continuations=3,
        penalty_init=10.0,
        penalty_scale=10.0,
        penalty_max=1.0e5,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        continue_on_signal_mismatch=continue_on_signal_mismatch,
        trust_radius_init=trust_radius_init,
        trust_radius_min=1.0e-3,
        trust_radius_grow=1.01,
        max_inner_attempts=2,
    )
    start = np.array(x0, dtype=float)
    return ScenarioRun(
        x0=start,
        constraint_names=HYBRID_CONSTRAINTS,
        evaluate_problem=hybrid_band_evaluate(
            surrogate_bias=surrogate_bias,
            activity_tolerance=activity_tolerance,
        ),
        settings=settings,
        inner_options=_inner_options(500),
        gate=AcceptedStateGate(accepted_x=start.copy(), step_norm_limit=step_norm_limit),
    )


def run_hybrid_mismatch_repair() -> dict:
    return execute(
        _hybrid_band_run(
            surrogate_bias=0.05,
            activity_tolerance=None,
            continue_on_signal_mismatch=True,
            max_outer_iterations=2,
            x0=(0.93, 1.0),
            trust_radius_init=0.02,
        )
    ).trajectory


def run_hybrid_mismatch_penalty_increase() -> dict:
    return execute(
        _hybrid_band_run(
            surrogate_bias=0.05,
            activity_tolerance=None,
            continue_on_signal_mismatch=False,
            max_outer_iterations=2,
            x0=(0.93, 1.0),
            trust_radius_init=0.02,
        )
    ).trajectory


def run_hybrid_zero_shift_split() -> dict:
    return execute(
        _hybrid_band_run(
            surrogate_bias=-0.05,
            activity_tolerance=0.02,
            continue_on_signal_mismatch=False,
            max_outer_iterations=3,
            x0=(0.93, 1.0),
            trust_radius_init=0.02,
        )
    ).trajectory


def run_constraints_inactive_stall() -> dict:
    return execute(
        _hybrid_band_run(
            surrogate_bias=0.05,
            activity_tolerance=None,
            continue_on_signal_mismatch=True,
            max_outer_iterations=3,
            x0=(0.0, 1.0),
            trust_radius_init=None,
            # The evaluator rejects every trial step (a frozen warm start).
            step_norm_limit=0.0,
        )
    ).trajectory


def _plateau_run(*, resume_state=None, pause_after_outer=None) -> ScenarioRun:
    settings = ALMSettings(
        max_outer_iterations=6,
        max_subproblem_continuations=2,
        penalty_init=1.0,
        penalty_scale=10.0,
        penalty_max=1.0e6,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        trust_radius_init=0.25,
        trust_radius_min=1.0e-3,
        trust_radius_shrink=0.5,
        trust_radius_grow=1.5,
        max_inner_attempts=3,
    )
    x0 = _resume_x0(np.array([0.95, 1.0]), resume_state)
    # Every trial step is rejected from outer 2 on (a frozen warm start).
    gate = AcceptedStateGate(accepted_x=x0.copy(), step_norm_limits_by_outer={2: 0.0})
    return ScenarioRun(
        x0=x0,
        constraint_names=FAR_TARGET_CONSTRAINTS,
        evaluate_problem=far_target_evaluate,
        settings=settings,
        inner_options=_inner_options(50),
        gate=gate,
        initial_multipliers=None if resume_state is not None else np.array([30.0]),
        resume_state=resume_state,
        pause_after_outer=pause_after_outer,
    )


def run_plateau_restore_best_feasible() -> dict:
    return execute(_plateau_run()).trajectory


def run_resume_plateau_best_feasible() -> dict:
    return interrupted_then_resumed(_plateau_run, pause_after_outer=1)


def run_trust_radius_retries() -> dict:
    settings = ALMSettings(
        max_outer_iterations=4,
        max_subproblem_continuations=2,
        penalty_init=1.0,
        penalty_scale=10.0,
        penalty_max=1.0e6,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        trust_radius_init=8.0,
        trust_radius_min=1.5,
        trust_radius_shrink=0.5,
        trust_radius_grow=1.5,
        max_inner_attempts=3,
        history_max_entries=None,
    )
    # An infeasible start (x0 <= 1 is violated): from a feasible one the run
    # would end by restoring it, not on the max-outer label it covers.
    x0 = np.array([1.25, 1.0])
    return execute(
        ScenarioRun(
            x0=x0,
            constraint_names=FAR_TARGET_CONSTRAINTS,
            evaluate_problem=far_target_nan_evaluate,
            settings=settings,
            inner_options=_inner_options(50),
            gate=AcceptedStateGate(accepted_x=x0.copy()),
        )
    ).trajectory


def run_frozen_warm_start_restore() -> dict:
    settings = ALMSettings(
        max_outer_iterations=8,
        max_subproblem_continuations=3,
        penalty_init=1.0,
        penalty_scale=10.0,
        penalty_max=1.0e6,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        trust_radius_init=0.25,
        trust_radius_min=1.0e-3,
        trust_radius_shrink=0.5,
        trust_radius_grow=1.5,
        max_inner_attempts=3,
    )
    x0 = np.zeros(2)
    return execute(
        ScenarioRun(
            x0=x0,
            constraint_names=FAR_TARGET_CONSTRAINTS,
            evaluate_problem=far_target_evaluate,
            settings=settings,
            inner_options=_inner_options(50),
            gate=AcceptedStateGate(accepted_x=x0.copy(), step_norm_limits_by_outer={4: 0.0}),
        )
    ).trajectory


SMOOTHED_CONSTRAINTS = ("spacing", "curvature")
SMOOTHING_SHRINK_RATE = 0.25
SMOOTHING_FLOOR_FRACTION = 1.0 / 8.0


@dataclass
class ShrinkingSmoothing:
    """Smoothing widths of a surrogate that an outer-step hook shrinks.

    After each outer step, every row whose surrogate and hard signals disagree
    in sign, or whose normalized surrogate-minus-hard gap exceeds the
    effective feasibility tolerance, has its width divided by
    ``1 + SMOOTHING_SHRINK_RATE`` (down to ``SMOOTHING_FLOOR_FRACTION`` of its
    initial width). Any change clears the physics cache of the evaluator,
    which reads the widths.
    """

    widths: np.ndarray
    floors: np.ndarray
    on_change: Callable[[], None]
    changes: list = field(default_factory=list)

    def after_outer_step(self, entry: dict) -> None:
        gaps = np.asarray(entry["surrogate_minus_hard_normalized_gap"], dtype=float)
        mismatches = np.asarray(
            entry["surrogate_hard_sign_mismatch_by_constraint"], dtype=bool
        )
        tolerance = float(entry["effective_feasibility_tolerance"])
        shrink = mismatches | (np.abs(gaps) > tolerance)
        widths = np.where(
            shrink,
            np.maximum(self.floors, self.widths / (1.0 + SMOOTHING_SHRINK_RATE)),
            self.widths,
        )
        if np.array_equal(widths, self.widths):
            return
        self.widths = widths
        self.changes.append(
            {
                "outer_iteration": int(entry["outer_iteration"]),
                "continuation_iteration": int(entry["continuation_iteration"]),
                "widths": widths.copy(),
            }
        )
        self.on_change()


def run_cached_physics_smoothing() -> dict:
    """A cached ``ALMPhysics`` evaluator whose surrogate an outer-step hook
    sharpens (clearing the cache on each change), with blocks and base bounds."""
    initial_widths = np.array([0.05, 0.04])
    smoothing = {}

    def physics_at(x):
        x = np.asarray(x, dtype=float)
        hard = np.array([x[0] - 1.0, x[1] - 1.0])
        surrogate = hard + smoothing["state"].widths
        hard_violation = np.maximum(hard, 0.0)
        return ALMPhysics(
            base_value=(x[0] - 0.97) ** 2 + (x[1] - 0.97) ** 2,
            base_grad=np.array([2.0 * (x[0] - 0.97), 2.0 * (x[1] - 0.97)]),
            constraint_values=surrogate,
            constraint_grads=(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            extras={
                "dual_update_values": hard,
                "feasibility_values": hard_violation,
                "max_feasibility_violation": float(np.max(hard_violation)),
                "hard_signed_constraint_values": hard,
                "hard_violation_values": hard_violation,
                "surrogate_signed_constraint_values": surrogate,
                "hard_dual_update_values": hard,
            },
        )

    evaluate_problem = cached_alm_evaluator(physics_at)
    state = ShrinkingSmoothing(
        widths=initial_widths.copy(),
        floors=initial_widths * SMOOTHING_FLOOR_FRACTION,
        on_change=evaluate_problem.cache_clear,
    )
    smoothing["state"] = state
    settings = ALMSettings(
        max_outer_iterations=6,
        max_subproblem_continuations=2,
        penalty_init=10.0,
        penalty_scale=10.0,
        penalty_max=1.0e6,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        max_inner_attempts=3,
    )
    trajectory = execute(
        ScenarioRun(
            x0=np.array([0.5, 0.5]),
            constraint_names=SMOOTHED_CONSTRAINTS,
            evaluate_problem=evaluate_problem,
            settings=settings,
            inner_options={"maxiter": 500, "maxcor": 10, "ftol": 1.0e-12, "gtol": 1.0e-12},
            outer_step_hook=state.after_outer_step,
            constraint_blocks=("spacing", "curvature"),
            base_bounds=((0.0, 2.0), (0.0, 2.0)),
            callbacks=OPTIONAL_CALLBACKS - {"on_outer_boundary"},
        )
    ).trajectory
    trajectory["smoothing_changes"] = encode(state.changes)
    return trajectory


def run_multiplier_cap_process_budget() -> dict:
    settings = ALMSettings(
        max_outer_iterations=12,
        max_subproblem_continuations=2,
        penalty_init=10.0,
        penalty_scale=10.0,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        multiplier_max=0.5,
        history_max_entries=3,
    )
    accepted = {"count": 0}

    def spend_budget(x):
        accepted["count"] += 1
        if accepted["count"] >= 5:
            raise ALMProcessBudgetExhausted()

    return execute(
        ScenarioRun(
            x0=np.zeros(2),
            constraint_names=TOY_CONSTRAINTS,
            evaluate_problem=toy_convex_evaluate,
            settings=settings,
            inner_options={"maxiter": 200},
            accepted_hook=spend_budget,
        )
    ).trajectory


def run_inner_iteration_budget() -> dict:
    settings = ALMSettings(
        max_outer_iterations=12,
        max_subproblem_continuations=2,
        penalty_init=10.0,
        penalty_scale=10.0,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
    )
    return execute(
        ScenarioRun(
            # Infeasible by less than the first subproblem's solution, so the
            # dual update needs a penalty raise; from a feasible start the
            # spent budget would end by restoring it, hiding its label.
            x0=np.array([1.52, 0.52]),
            constraint_names=TOY_CONSTRAINTS,
            evaluate_problem=toy_convex_evaluate,
            settings=settings,
            # Spent at the end of outer 2, after its dual update.
            inner_options={"maxiter": 5},
        )
    ).trajectory


def run_dual_update_penalty_cap() -> dict:
    settings = ALMSettings(
        max_outer_iterations=12,
        max_subproblem_continuations=2,
        penalty_init=10.0,
        penalty_scale=10.0,
        penalty_max=50.0,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
    )
    return execute(
        ScenarioRun(
            # Infeasible by less than the first subproblem's solution (so
            # the dual update needs a penalty raise), and not a feasible
            # start the capped run would restore.
            x0=np.array([1.52, 0.52]),
            constraint_names=TOY_CONSTRAINTS,
            evaluate_problem=toy_convex_evaluate,
            settings=settings,
            inner_options={},
        )
    ).trajectory


def run_preinner_converged() -> dict:
    settings = ALMSettings(
        max_outer_iterations=3,
        max_subproblem_continuations=1,
        penalty_init=10.0,
        penalty_scale=10.0,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
    )
    x0 = np.array([1.5, 0.5])
    return execute(
        ScenarioRun(
            x0=x0,
            constraint_names=TOY_CONSTRAINTS,
            evaluate_problem=toy_convex_evaluate,
            settings=settings,
            inner_options=_inner_options(200),
            gate=AcceptedStateGate(accepted_x=x0.copy()),
            initial_multipliers=np.array([1.0, 0.0]),
        )
    ).trajectory


def run_zero_inner_budget() -> dict:
    settings = ALMSettings(
        max_outer_iterations=3,
        max_subproblem_continuations=1,
        penalty_init=10.0,
        penalty_scale=10.0,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
    )
    return execute(
        ScenarioRun(
            x0=np.zeros(2),
            constraint_names=TOY_CONSTRAINTS,
            evaluate_problem=toy_convex_evaluate,
            settings=settings,
            inner_options={"maxiter": 0},
        )
    ).trajectory


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    run: Callable[[], dict]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "toy_convex",
        "Convex QP with no optional callbacks: dual updates to convergence.",
        run_toy_convex,
    ),
    Scenario(
        "penalty_ramp_to_cap",
        "Stateful evaluator, infeasible start: penalty raises and dual updates "
        "until the penalty cap stops the run.",
        run_penalty_ramp_to_cap,
    ),
    Scenario(
        "resume_penalty_ramp_mid_run",
        "penalty_ramp_to_cap interrupted after outer 4, its checkpoint pickled, "
        "and resumed in a fresh run.",
        run_resume_penalty_ramp_mid_run,
    ),
    Scenario(
        "hybrid_mismatch_repair",
        "Hybrid signals, continue_on_signal_mismatch=True: mismatch steps "
        "repaired by continuation, a mismatch subproblem-limit raise, and "
        "max-outer exhaustion with best-feasible restore.",
        run_hybrid_mismatch_repair,
    ),
    Scenario(
        "hybrid_mismatch_penalty_increase",
        "Hybrid signals, continue_on_signal_mismatch=False: the mismatch penalty raise.",
        run_hybrid_mismatch_penalty_increase,
    ),
    Scenario(
        "hybrid_zero_shift_split",
        "Hybrid signals, hard row active and surrogate row inactive with zero "
        "surrogate shift: no signal mismatch (no multiplier rides on the row), "
        "so the subproblem continues until the surrogate's gap leaves the "
        "iterate hard-infeasible, and max-outer exhaustion restores the best "
        "hard-feasible one.",
        run_hybrid_zero_shift_split,
    ),
    Scenario(
        "constraints_inactive_stall",
        "Hybrid signals with hard constraints inactive and every trial step "
        "rejected by the evaluator: the constraints-inactive stall termination.",
        run_constraints_inactive_stall,
    ),
    Scenario(
        "plateau_restore_best_feasible",
        "Boxed feasible continuations, then a frozen warm start: plateau stall "
        "with restore of the best-feasible incumbent.",
        run_plateau_restore_best_feasible,
    ),
    Scenario(
        "resume_plateau_best_feasible",
        "plateau_restore_best_feasible interrupted after outer 1 (best-feasible "
        "incumbent in the checkpoint), its checkpoint pickled, and resumed to "
        "the restore.",
        run_resume_plateau_best_feasible,
    ),
    Scenario(
        "trust_radius_retries",
        "Trust-region retries: shrink, min-radius/attempt exhaustion, growth, NaN "
        "trial evaluations sanitized, max-outer exhaustion.",
        run_trust_radius_retries,
    ),
    Scenario(
        "frozen_warm_start_restore",
        "Boxed steps, then every trial step rejected from outer 4 on: "
        "infeasible stalls raise the penalty to the cap (the frozen iterate "
        "is no subproblem minimizer, so no multiplier update), and the "
        "best-feasible restore at the cap.",
        run_frozen_warm_start_restore,
    ),
    Scenario(
        "cached_physics_smoothing",
        "A cached ALMPhysics evaluator whose outer-step hook sharpens the "
        "surrogate and clears the cache, with constraint blocks and base "
        "bounds; converges once the sharpened surrogate agrees with the hard "
        "signals.",
        run_cached_physics_smoothing,
    ),
    Scenario(
        "multiplier_cap_process_budget",
        "Multiplier cap binding blocks convergence, history truncation, and a "
        "caller-raised process-budget stop.",
        run_multiplier_cap_process_budget,
    ),
    Scenario(
        "inner_iteration_budget",
        "The inner maxiter budget of the call runs out at an outer boundary.",
        run_inner_iteration_budget,
    ),
    Scenario(
        "dual_update_penalty_cap",
        "A dual update that requires a penalty raise past penalty_max (inner "
        "options without maxiter).",
        run_dual_update_penalty_cap,
    ),
    Scenario(
        "preinner_converged",
        "Start at a KKT point with a stateful evaluator: the pre-inner converged "
        "shortcut with an incumbent snapshot and a terminal checkpoint.",
        run_preinner_converged,
    ),
    Scenario(
        "zero_inner_budget",
        "inner maxiter 0: the first step's inner solve is an exhausted "
        "placeholder (no move), and the spent budget ends the run after it.",
        run_zero_inner_budget,
    ),
)

SCENARIOS_BY_NAME = MappingProxyType({scenario.name: scenario for scenario in SCENARIOS})


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.json"


def load_golden(name: str) -> dict:
    return json.loads(fixture_path(name).read_text(encoding="utf-8"))


def load_manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


EPS = float(np.finfo(float).eps)


def load_sensitivity() -> dict:
    """``sensitivity.json``: each scenario's measured spread under last-bit
    noise (``measure_alm_golden_sensitivity.py``)."""
    return json.loads((FIXTURE_DIR / "sensitivity.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def openblas_coretype() -> Optional[str]:
    """The OpenBLAS kernel family this process runs: the core name
    (threadpoolctl's ``architecture``, OpenBLAS's own ``get_corename``) that
    every OpenBLAS numpy and SciPy loaded reports (``OPENBLAS_CORETYPE``, set
    before they load, pins it). None without an OpenBLAS, or when the loaded
    builds disagree."""
    names = {
        info["architecture"]
        for info in threadpoolctl.threadpool_info()
        if info["internal_api"] == "openblas"
    }
    return names.pop() if len(names) == 1 else None


def recorded_environment() -> tuple[str, str, str, str]:
    """The numpy and SciPy versions, the machine and the OpenBLAS kernel
    family the goldens were recorded with."""
    environment = load_manifest()["environment"]
    return (
        environment["numpy"],
        environment["scipy"],
        environment["machine"],
        environment["openblas_coretype"],
    )


def current_environment() -> tuple[str, str, str, Optional[str]]:
    return np.__version__, scipy.__version__, platform.machine(), openblas_coretype()


def bitwise_environment() -> bool:
    """Whether this numpy, SciPy, machine and OpenBLAS kernel family are the
    ones the goldens were recorded with (floating-point results also depend on
    the vector paths the BLAS kernels take, not only on the library
    versions)."""
    return current_environment() == recorded_environment()


def golden_difference(name: str, trajectory: dict) -> Optional[str]:
    """How a fresh run of scenario ``name`` differs from its golden: in its
    outcomes under any numpy and SciPy, bit for bit under the recorded ones."""
    golden = load_golden(name)
    outcomes = sorted(scenario_outcomes(trajectory))
    if outcomes != golden["outcomes"]:
        return f"outcomes {outcomes} != golden {golden['outcomes']}"
    if bitwise_environment():
        return first_difference(golden["trajectory"], trajectory)
    return None
