"""The evaluator's dict as the ALM loop reads it.

``evaluate_problem(x, multipliers, penalty) -> dict`` (an :class:`ALMEvaluator`
returning an :class:`ALMEvaluation`; keys in the package docstring) is the
loop's only view of the problem. This module owns what the loop assumes about
that dict: its schema and the check of it where each evaluation enters
(:func:`_contract_checked_evaluation`), which of its arrays the solver copies
to own (:data:`_OWNED_EVALUATION_ARRAY_FIELDS`,
:func:`_clone_evaluation_dict`), which fields must be finite (:func:`_nonfinite_evaluation_fields`), the
constraint metadata it attaches, the objective that ranks best-feasible
incumbents (:func:`_incumbent_objective_value`), and the measurement of an
evaluated iterate that the loop and the continuation policy read
(:func:`_measure_iterate`).
"""

# Eager annotations: on Python 3.8 a TypedDict subclass defined in another
# module resolves inherited string annotations in that module's namespace.
from typing import Callable, FrozenSet, List, Optional, Protocol, Sequence, Tuple, TypedDict, Union

import numpy as np

from .continuation import ALMIterateMeasurement
from .core import (
    _HYBRID_SIGNAL_FIELDS,
    _bound_reduced_stationarity_norm,
    _constraint_routing_state,
    _extract_constraint_state,
    _stationarity_metrics,
)
from .events import _require_acyclic_containers


class _ALMEvaluationRequired(TypedDict):
    total: float
    grad: np.ndarray
    constraint_values: np.ndarray
    feasibility_values: np.ndarray
    dual_update_values: np.ndarray
    constraint_grads: Sequence[np.ndarray]


class ALMEvaluation(_ALMEvaluationRequired, total=False):
    """The evaluator's dict: the required keys and every optional key the
    solver owns. The hybrid quartet is all-or-none, checked at run time. Any
    other key is an application diagnostic the solver keeps (no key may hold a
    container cycle).
    """

    # The objective without penalty terms, its gradient and stationarity. A
    # stationarity_norm that is not ||grad|| is used as given (no bound
    # reduction), so it must account for base_bounds itself.
    base_value: float
    base_total: float
    physics_total: float
    base_grad: np.ndarray
    metric_grad: np.ndarray
    stationarity_norm: float
    metric_stationarity_norm: float
    # What ``augmented_inequality_objective`` adds. Given both, the history
    # reports the per-row shift max(0, λ + ρg) and term (shift² − λ²)/2ρ
    # instead of recomputing them.
    max_violation: float
    max_feasibility_violation: float
    positive_shift_values: np.ndarray
    augmented_term_by_constraint: np.ndarray
    # Per-row scaling and activity.
    constraint_scales: Union[np.ndarray, Sequence[float]]
    constraint_scale_sources: Sequence[str]
    constraint_activity_tolerances: np.ndarray
    # Trial-point status.
    nonfinite_evaluation: bool
    search_step_success: bool
    # The hybrid quartet.
    hard_signed_constraint_values: np.ndarray
    hard_violation_values: np.ndarray
    surrogate_signed_constraint_values: np.ndarray
    hard_dual_update_values: np.ndarray
    # Unscaled (raw) and scaled rows: the solver copies them, and the history
    # reports the signed, violation and feasibility ones.
    raw_constraint_values: np.ndarray
    raw_solver_constraint_values: np.ndarray
    raw_dual_update_values: np.ndarray
    raw_hard_signed_constraint_values: np.ndarray
    raw_hard_violation_values: np.ndarray
    raw_surrogate_signed_constraint_values: np.ndarray
    raw_hard_dual_update_values: np.ndarray
    normalized_signed_constraint_values: np.ndarray
    normalized_feasibility_values: np.ndarray
    # More Jacobians; checkpoints omit them, as they do ``constraint_grads``.
    raw_constraint_grads: Sequence[np.ndarray]
    normalized_constraint_grads: Sequence[np.ndarray]
    # Per-row kinds; any row whose two kinds differ makes the history call the
    # multipliers search multipliers.
    gradient_value_kinds: Sequence[str]
    dual_update_value_kinds: Sequence[str]
    # Set by the solver on the dicts it keeps and hands out.
    constraint_names: Sequence[str]
    constraint_blocks: Sequence[str]
    nonfinite_fields: Sequence[str]


class ALMEvaluator(Protocol):
    """``minimize_alm``'s ``evaluate_problem``: the :class:`ALMEvaluation` at
    ``x`` for these multipliers and this penalty, a dict the solver may keep."""

    def __call__(
        self, x: np.ndarray, multipliers: np.ndarray, penalty: float
    ) -> ALMEvaluation: ...

_OWNED_EVALUATION_ARRAY_FIELDS = (
    "grad",
    "metric_grad",
    "base_grad",
    "constraint_values",
    "feasibility_values",
    "dual_update_values",
    *_HYBRID_SIGNAL_FIELDS,
    "raw_constraint_values",
    "raw_solver_constraint_values",
    "raw_dual_update_values",
    "raw_hard_signed_constraint_values",
    "raw_hard_violation_values",
    "raw_surrogate_signed_constraint_values",
    "raw_hard_dual_update_values",
    "normalized_signed_constraint_values",
    "normalized_feasibility_values",
    "constraint_scales",
    "constraint_activity_tolerances",
    "positive_shift_values",
    "augmented_term_by_constraint",
)

def _nonfinite_evaluation_fields(evaluation: dict) -> Tuple[str, ...]:
    invalid_fields: List[str] = []

    if not np.isfinite(float(evaluation["total"])):
        invalid_fields.append("total")

    grad = np.asarray(evaluation["grad"], dtype=float)
    if not np.all(np.isfinite(grad)):
        invalid_fields.append("grad")

    optional_scalar_fields = (
        "stationarity_norm",
        "metric_stationarity_norm",
        "max_violation",
        "base_value",
        "base_total",
    )
    for field_name in optional_scalar_fields:
        if field_name not in evaluation:
            continue
        if not np.isfinite(float(evaluation[field_name])):
            invalid_fields.append(field_name)

    optional_array_fields = (
        "constraint_values",
        "feasibility_values",
        "dual_update_values",
        "metric_grad",
        "base_grad",
        "constraint_activity_tolerances",
        # Explicit hybrid signal arrays participate in routing and
        # the dual update; a NaN here flows directly into multiplier
        # projection. The hybrid surrogate-vs-hard contract depends on
        # these being finite at every evaluation boundary.
        *_HYBRID_SIGNAL_FIELDS,
    )
    for field_name in optional_array_fields:
        if field_name not in evaluation:
            continue
        field_values = np.asarray(evaluation[field_name], dtype=float)
        if not np.all(np.isfinite(field_values)):
            invalid_fields.append(field_name)

    if "constraint_grads" in evaluation and evaluation["constraint_grads"] is not None:
        for grad_index, constraint_grad in enumerate(evaluation["constraint_grads"]):
            constraint_grad_array = np.asarray(constraint_grad, dtype=float)
            if not np.all(np.isfinite(constraint_grad_array)):
                invalid_fields.append(f"constraint_grads[{grad_index}]")

    return tuple(invalid_fields)

def _require_finite_evaluation(evaluation: dict, *, context: str) -> None:
    invalid_fields = _nonfinite_evaluation_fields(evaluation)
    if invalid_fields:
        invalid_summary = ", ".join(invalid_fields)
        raise ValueError(f"{context} produced non-finite ALM data: {invalid_summary}")

def _clone_evaluation_dict(
    evaluation: dict,
    *,
    owned_keys: Optional[FrozenSet[str]] = None,
) -> dict:
    """Copy evaluation arrays.

    ``owned_keys is None`` copies every ndarray and both grad lists.
    A set copies only those keys. The inner cache omits
    ``raw_constraint_grads``, so that list stays aliased.
    """
    snapshot = dict(evaluation)
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray) and (
            owned_keys is None or key in owned_keys
        ):
            snapshot[key] = value.copy()
    for grad_key in ("constraint_grads", "raw_constraint_grads"):
        if owned_keys is not None and grad_key not in owned_keys:
            continue
        grads = snapshot.get(grad_key)
        if isinstance(grads, list):
            snapshot[grad_key] = [
                grad.copy()
                if isinstance(grad, np.ndarray)
                else np.asarray(grad, dtype=float).copy()
                for grad in grads
            ]
    return snapshot

def _incumbent_objective_value(evaluation: dict) -> float:
    if "physics_total" in evaluation:
        return float(evaluation["physics_total"])
    if "base_value" in evaluation:
        return float(evaluation["base_value"])
    if "base_total" in evaluation:
        return float(evaluation["base_total"])
    return float(evaluation["total"])

def _attach_alm_constraint_metadata(
    evaluation: dict,
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
) -> dict:
    # Always shallow-copy, with or without constraint blocks, so this
    # function returns a dict the caller owns and never aliases its input.
    annotated = dict(evaluation)
    if constraint_blocks_tuple is None:
        return annotated
    annotated["constraint_names"] = constraint_names_tuple
    annotated["constraint_blocks"] = constraint_blocks_tuple
    return annotated

_REQUIRED_EVALUATION_KEYS = tuple(_ALMEvaluationRequired.__annotations__)

def _contract_checked_evaluation(
    evaluation: dict,
    *,
    x: np.ndarray,
    constraint_count: int,
    context: str,
) -> dict:
    """An owned shallow copy of the evaluator's dict at ``x``, checked where
    it enters the solver: every required key present and not None, ``grad``
    and each ``constraint_grads`` row of shape ``(x.size,)``, one row and one
    ``constraint_values`` entry per constraint, nonnegative
    ``constraint_activity_tolerances`` (optional), and ``search_step_success``
    (optional) a bool or ``numpy.bool_``, stored as a Python bool. Raises
    ``KeyError`` for a missing key and ``ValueError`` otherwise, naming
    ``context``."""
    missing = [key for key in _REQUIRED_EVALUATION_KEYS if evaluation.get(key) is None]
    if missing:
        raise KeyError(
            f"{context}: the evaluation lacks required keys (absent or None): "
            + ", ".join(missing)
        )
    constraint_grads = evaluation["constraint_grads"]
    if len(constraint_grads) != constraint_count:
        raise ValueError(
            f"{context}: constraint_grads has {len(constraint_grads)} rows for "
            f"{constraint_count} constraints"
        )
    dof_shape = (int(np.size(x)),)
    expected_shapes = (
        ("grad", evaluation["grad"], dof_shape),
        ("constraint_values", evaluation["constraint_values"], (constraint_count,)),
        *(
            (f"constraint_grads[{index}]", row, dof_shape)
            for index, row in enumerate(constraint_grads)
        ),
    )
    for field_name, value, expected in expected_shapes:
        if np.shape(value) != expected:
            raise ValueError(
                f"{context}: {field_name} has shape {np.shape(value)}, expected {expected}"
            )
    activity_tolerances = evaluation.get("constraint_activity_tolerances")
    if activity_tolerances is not None and np.any(
        np.asarray(activity_tolerances, dtype=float) < 0.0
    ):
        raise ValueError(f"{context}: constraint_activity_tolerances must be nonnegative")
    checked = dict(evaluation)
    if "search_step_success" in checked:
        flag = checked["search_step_success"]
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(
                f"{context}: search_step_success must be a bool, got "
                f"{type(flag).__name__} {flag!r}"
            )
        checked["search_step_success"] = bool(flag)
    return checked

def _search_step_rejected(evaluation: dict) -> bool:
    """Whether the evaluator rejected this trial step
    (``search_step_success`` False; absent means accepted)."""
    return not evaluation.get("search_step_success", True)

def _checked_evaluation(
    evaluate_problem: Callable[[np.ndarray, np.ndarray, object], dict],
    x: np.ndarray,
    multipliers: np.ndarray,
    penalty_argument: object,
    *,
    constraint_names_tuple: Tuple[str, ...],
    constraint_blocks_tuple: Optional[Tuple[str, ...]],
    context: str,
) -> dict:
    """The evaluation at ``(x, multipliers, penalty_argument)`` with the
    constraint metadata attached (an owned dict). Raises as
    :func:`_contract_checked_evaluation` does, and ``ValueError`` naming
    ``context`` when a field the loop reads is not finite or a container in
    it contains itself."""
    evaluation = _attach_alm_constraint_metadata(
        _contract_checked_evaluation(
            evaluate_problem(x, multipliers, penalty_argument),
            x=x,
            constraint_count=len(constraint_names_tuple),
            context=context,
        ),
        constraint_names_tuple,
        constraint_blocks_tuple,
    )
    _require_finite_evaluation(evaluation, context=context)
    _require_acyclic_containers(evaluation, context=context, path="evaluation")
    return evaluation

def _measure_iterate(
    evaluation: dict,
    *,
    x: np.ndarray,
    base_bounds: Optional[Sequence[Tuple[float, float]]],
    multipliers: np.ndarray,
    penalty: float,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    effective_feasibility_tol: float,
) -> ALMIterateMeasurement:
    """What the loop and the policy read about the iterate ``x``: its signed
    and feasibility values, routing state and stationarity norms (reduced at
    active ``base_bounds``), with the active sets judged at
    ``effective_feasibility_tol``. ``evaluation`` and ``multipliers`` are held
    by reference."""
    (
        solver_constraint_values,
        feasibility_values,
        _dual_update_values,
        max_feasibility_violation,
    ) = _extract_constraint_state(evaluation)
    routing_state = _constraint_routing_state(
        evaluation,
        multipliers,
        penalty,
        effective_feasibility_tol,
    )
    (
        stationarity_norm,
        kkt_stationarity_norm,
        signal_mismatch_active,
    ) = _stationarity_metrics(evaluation, routing_state, effective_feasibility_tol)
    stationarity_norm = _bound_reduced_stationarity_norm(evaluation, x, base_bounds)
    return ALMIterateMeasurement(
        evaluation=evaluation,
        multipliers=multipliers,
        penalty=penalty,
        solver_constraint_values=solver_constraint_values,
        feasibility_values=feasibility_values,
        max_feasibility_violation=max_feasibility_violation,
        routing_state=routing_state,
        stationarity_norm=stationarity_norm,
        kkt_stationarity_norm=kkt_stationarity_norm,
        signal_mismatch_active=signal_mismatch_active,
        update_feasibility_tol=update_feasibility_tol,
        update_stationarity_tol=update_stationarity_tol,
        effective_feasibility_tol=effective_feasibility_tol,
    )
