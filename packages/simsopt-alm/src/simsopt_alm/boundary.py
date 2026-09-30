"""The ALM loop between two outer iterations, and the rules it obeys.

:func:`~.control.minimize_alm` passes an :class:`ALMOuterBoundary` to
``on_outer_boundary`` after each completed outer iteration and once when it
returns, and resumes from a non-terminal one (``resume_from``). The rules a
boundary obeys (:func:`_validate_alm_boundary_fields`: finite typed vectors,
nonnegative multipliers, positive penalty and tolerances, consistent names,
blocks, cap indices, sufficient-decrease carriers and geometry identity) are
checked in one place for both the checkpoint snapshot
(:mod:`.checkpoint`) and ``resume_from`` (:func:`_validate_resume_boundary`).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Generic, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import (
    AcceptedStateT,
    ALMLoopState,
    _require_acyclic_containers,
    _validate_geometry_identity_pair,
)

@dataclass(frozen=True)
class ALMOuterBoundary(Generic[AcceptedStateT]):
    """The loop between two outer iterations, or where it returns.

    Passed to ``minimize_alm(on_outer_boundary=...)`` after each completed
    outer iteration and once at the end (then ``state`` holds the returned x,
    multipliers and penalty, and ``termination_reason`` is set). A boundary
    with no termination reason resumes a run as ``resume_from``, which checks
    the checkpoint rules first and keeps ``state.best_feasible`` by reference.
    ``accepted_state`` is the caller's snapshot at ``state.x``; the rest is
    read-only (arrays not writable, mappings read-only views, sequences
    tuples), as in an event. So a run resumed in-process from a published
    boundary and restored to its incumbent returns
    ``evaluation["constraint_grads"]`` as a tuple where an uninterrupted run
    returns a list (checkpoints omit that key).
    """

    completed_outer_iterations: int
    completed_action: str
    termination_reason: Optional[str]
    constraint_names: Tuple[str, ...]
    constraint_blocks: Optional[Tuple[str, ...]]
    accepted_state: Optional[AcceptedStateT]
    geometry_identity: Optional[str]
    state: ALMLoopState[AcceptedStateT]

def _strict_transition_float(value: object, context: str) -> float:
    """Validate a typed finite real before materializing it as ``float``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite real number")
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{context} must be a finite real number")
    return value_f

def _positive_transition_float(value: object, name: str) -> float:
    value_f = _strict_transition_float(value, name)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value_f

def _nonnegative_transition_float(value: object, name: str) -> float:
    value_f = _strict_transition_float(value, name)
    if value_f < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value_f

def _strict_transition_float_tuple(
    values: Sequence[float],
    context: str,
) -> Tuple[float, ...]:
    """Validate a transition vector without coercing numeric strings."""
    if isinstance(values, (str, bytes, bytearray, Mapping)) or not isinstance(
        values, Sequence
    ):
        raise ValueError(f"{context} must be a finite vector")
    return tuple(
        _strict_transition_float(value, f"{context}[{index}]")
        for index, value in enumerate(values)
    )

def _validate_transition_vectors(
    *,
    x: Sequence[float],
    multipliers: Sequence[float],
    penalty: float,
    context: str,
) -> None:
    # The strict tuples are finite 1-D vectors by construction.
    _strict_transition_float_tuple(x, f"{context} x")
    multiplier_values = _strict_transition_float_tuple(
        multipliers, f"{context} multipliers"
    )
    if any(value < 0.0 for value in multiplier_values):
        raise ValueError(f"{context} multipliers must be nonnegative")
    _positive_transition_float(penalty, f"{context} penalty")

def _validate_alm_boundary_fields(
    *,
    context: str,
    x: Sequence[float],
    multipliers: Sequence[float],
    penalty: float,
    constraint_names: Sequence[str],
    constraint_blocks: Optional[Sequence[str]],
    completed_outer_iterations: int,
    completed_action: str,
    update_feasibility_tol: float,
    update_stationarity_tol: float,
    trust_radius: Optional[float],
    last_cap_binding_active: bool,
    sufficient_decrease_measure: Optional[float],
    sufficient_decrease_multipliers: Optional[Sequence[float]],
    sufficient_decrease_penalty: Optional[float],
    total_inner_iterations: int,
    cap_binding_detected: bool,
    cap_binding_indices: Sequence[int],
    penalty_cap_reached: bool,
    penalty_cap_requested: Optional[float],
    accepted_state: Optional[object],
    geometry_identity: Optional[str],
    best_feasible_dimensions: Optional[Tuple[int, int]],
) -> None:
    """The rules a loop boundary obeys, for a checkpoint snapshot and for
    ``minimize_alm(resume_from=...)`` alike. Vectors are sequences of reals
    (not coerced); ``best_feasible_dimensions`` is the incumbent's
    ``(len(x), len(multipliers))``. Raises ``ValueError`` on the first breach.
    """
    if any(
        isinstance(index, bool) or not isinstance(index, Integral) or index < 0
        for index in cap_binding_indices
    ):
        raise ValueError("cap_binding_indices must be nonnegative integers")
    _validate_transition_vectors(
        x=x,
        multipliers=multipliers,
        penalty=penalty,
        context=context,
    )
    if len(constraint_names) != len(multipliers):
        raise ValueError(
            f"{context} constraint names and multipliers must have equal dimensions"
        )
    if any(not isinstance(name, str) or not name for name in constraint_names):
        raise ValueError(f"{context} constraint names must be non-empty strings")
    if len(set(constraint_names)) != len(constraint_names):
        raise ValueError(f"{context} constraint names must be unique")
    if constraint_blocks is not None and len(constraint_blocks) != len(
        constraint_names
    ):
        raise ValueError(
            f"{context} constraint blocks and names must have equal dimensions"
        )
    if constraint_blocks is not None and any(
        not isinstance(block, str) or not block for block in constraint_blocks
    ):
        raise ValueError(f"{context} constraint blocks must be non-empty strings")
    _positive_transition_float(update_feasibility_tol, "update_feasibility_tol")
    _positive_transition_float(update_stationarity_tol, "update_stationarity_tol")
    if trust_radius is not None:
        _positive_transition_float(trust_radius, "trust_radius")
    if isinstance(completed_outer_iterations, bool) or not isinstance(
        completed_outer_iterations, Integral
    ):
        raise ValueError("completed_outer_iterations must be an integer")
    if completed_outer_iterations < 0:
        raise ValueError("completed_outer_iterations must be nonnegative")
    if type(last_cap_binding_active) is not bool:
        raise ValueError("last_cap_binding_active must be bool")
    if not isinstance(completed_action, str) or not completed_action:
        raise ValueError("completed_action must be a non-empty string")
    if sufficient_decrease_measure is None:
        if (
            sufficient_decrease_multipliers is not None
            or sufficient_decrease_penalty is not None
        ):
            raise ValueError(
                "sufficient-decrease measurement carriers require a measure"
            )
    else:
        _nonnegative_transition_float(
            sufficient_decrease_measure, "sufficient_decrease_measure"
        )
        if sufficient_decrease_multipliers is None:
            raise ValueError(
                "sufficient_decrease_measure requires measurement multipliers"
            )
        measurement_multipliers = _strict_transition_float_tuple(
            sufficient_decrease_multipliers, "sufficient_decrease_multipliers"
        )
        if len(measurement_multipliers) != len(multipliers):
            raise ValueError(
                "sufficient-decrease measurement multipliers and names "
                "must have equal dimensions"
            )
        if any(value < 0.0 for value in measurement_multipliers):
            raise ValueError("sufficient_decrease_multipliers must be nonnegative")
        if sufficient_decrease_penalty is None:
            raise ValueError("sufficient_decrease_measure requires measurement penalty")
        _positive_transition_float(
            sufficient_decrease_penalty, "sufficient_decrease_penalty"
        )
    if (
        isinstance(total_inner_iterations, bool)
        or not isinstance(total_inner_iterations, Integral)
        or total_inner_iterations < 0
    ):
        raise ValueError("total_inner_iterations must be nonnegative integer")
    if type(cap_binding_detected) is not bool:
        raise ValueError("cap_binding_detected must be bool")
    if type(penalty_cap_reached) is not bool:
        raise ValueError("penalty_cap_reached must be bool")
    if len(set(cap_binding_indices)) != len(cap_binding_indices):
        raise ValueError("cap_binding_indices must be unique")
    if any(index >= len(multipliers) for index in cap_binding_indices):
        raise ValueError("cap_binding_indices must reference a multiplier")
    if penalty_cap_requested is not None:
        _positive_transition_float(penalty_cap_requested, "penalty_cap_requested")
    if geometry_identity is not None and (
        not isinstance(geometry_identity, str) or not geometry_identity
    ):
        raise ValueError("geometry_identity must be non-empty")
    _validate_geometry_identity_pair(
        accepted_state=accepted_state,
        geometry_identity=geometry_identity,
        context=context,
    )
    if best_feasible_dimensions is not None:
        best_x_size, best_multiplier_size = best_feasible_dimensions
        if best_x_size != len(x):
            raise ValueError(
                "ALM best-feasible x and transition x must have equal dimensions"
            )
        if best_multiplier_size != len(multipliers):
            raise ValueError(
                "ALM best-feasible multipliers and transition multipliers "
                "must have equal dimensions"
            )

def _validate_resume_boundary(resume_from: ALMOuterBoundary[AcceptedStateT]) -> None:
    """A resume boundary obeys the boundary rules, is not terminal, and its
    incumbent's evaluation is acyclic."""
    state = resume_from.state
    best = state.best_feasible
    if best is not None:
        _require_acyclic_containers(
            best.evaluation,
            context="ALM resume boundary best_feasible evaluation",
            path="evaluation",
        )
    _validate_alm_boundary_fields(
        context="ALM resume boundary",
        x=tuple(state.x),
        multipliers=tuple(state.multipliers),
        penalty=state.penalty,
        constraint_names=resume_from.constraint_names,
        constraint_blocks=resume_from.constraint_blocks,
        completed_outer_iterations=resume_from.completed_outer_iterations,
        completed_action=resume_from.completed_action,
        update_feasibility_tol=state.update_feasibility_tol,
        update_stationarity_tol=state.update_stationarity_tol,
        trust_radius=state.trust_radius,
        last_cap_binding_active=state.last_cap_binding_active,
        sufficient_decrease_measure=state.sufficient_decrease_measure,
        sufficient_decrease_multipliers=(
            None
            if state.sufficient_decrease_multipliers is None
            else tuple(state.sufficient_decrease_multipliers)
        ),
        sufficient_decrease_penalty=state.sufficient_decrease_penalty,
        total_inner_iterations=state.total_inner_iterations,
        cap_binding_detected=state.cap_binding_detected,
        cap_binding_indices=tuple(state.cap_binding_indices),
        penalty_cap_reached=state.penalty_cap_reached,
        penalty_cap_requested=state.penalty_cap_requested,
        accepted_state=resume_from.accepted_state,
        geometry_identity=resume_from.geometry_identity,
        best_feasible_dimensions=(
            None if best is None else (len(best.x), len(best.multipliers))
        ),
    )
    if resume_from.termination_reason is not None:
        raise ValueError("resume_from is terminal and cannot be resumed")
