"""Opt-in checkpoint/resume for :func:`simsopt.solve.alm.minimize_alm`.

The loop publishes an :class:`~.boundary.ALMOuterBoundary` after each completed
outer iteration and once when it returns, and resumes from one. This module
turns a boundary into an immutable :class:`ALMTransitionSnapshot` (the content
of an ``alm_transition_checkpoint_v1`` checkpoint: finite scalars, tuples and
tagged mappings, no constraint Jacobians) and a snapshot back into a boundary.
:func:`alm_checkpointing` wires both into one ``minimize_alm`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Generic, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .boundary import (
    ALMOuterBoundary,
    _strict_transition_float_tuple,
    _validate_alm_boundary_fields,
    _validate_transition_vectors,
)
from .core import _nonnegative_alm_integer
from .events import (
    AcceptedStateT,
    ALMFeasibleIncumbent,
    ALMLoopState,
    _cyclic_container_error,
    _frozen_loop_state,
    _validate_geometry_identity_pair,
)

ALMTransitionValue = Union[
    str,
    int,
    float,
    bool,
    None,
    Tuple["ALMTransitionValue", ...],
    Tuple[Tuple[str, "ALMTransitionValue"], ...],
]

ALMTransitionEvaluation = Tuple[Tuple[str, ALMTransitionValue], ...]

@dataclass(frozen=True)
class ALMFeasibleIncumbentSnapshot(Generic[AcceptedStateT]):
    """Immutable core best-feasible carrier at an outer boundary."""

    x: Tuple[float, ...]
    evaluation: ALMTransitionEvaluation
    multipliers: Tuple[float, ...]
    penalty: float
    accepted_state: Optional[AcceptedStateT]
    geometry_identity: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x",
            _strict_transition_float_tuple(self.x, "ALM best-feasible snapshot x"),
        )
        object.__setattr__(
            self,
            "multipliers",
            _strict_transition_float_tuple(
                self.multipliers,
                "ALM best-feasible snapshot multipliers",
            ),
        )
        object.__setattr__(
            self,
            "evaluation",
            _canonical_transition_evaluation(self.evaluation),
        )
        _validate_transition_vectors(
            x=self.x,
            multipliers=self.multipliers,
            penalty=self.penalty,
            context="ALM best-feasible snapshot",
        )
        if self.geometry_identity is not None and (
            not isinstance(self.geometry_identity, str) or not self.geometry_identity
        ):
            raise ValueError("ALM best-feasible geometry_identity must be non-empty")
        _validate_geometry_identity_pair(
            accepted_state=self.accepted_state,
            geometry_identity=self.geometry_identity,
            context="ALM best-feasible snapshot",
        )

@dataclass(frozen=True)
class ALMTransitionSnapshot(Generic[AcceptedStateT]):
    """Immutable state committed after one completed ALM outer iteration."""

    x: Tuple[float, ...]
    accepted_state: Optional[AcceptedStateT]
    geometry_identity: Optional[str]
    constraint_names: Tuple[str, ...]
    constraint_blocks: Optional[Tuple[str, ...]]
    completed_outer_iterations: int
    multipliers: Tuple[float, ...]
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    trust_radius: Optional[float]
    last_cap_binding_active: bool
    sufficient_decrease_measure: Optional[float]
    sufficient_decrease_multipliers: Optional[Tuple[float, ...]]
    sufficient_decrease_penalty: Optional[float]
    best_feasible: Optional[ALMFeasibleIncumbentSnapshot[AcceptedStateT]]
    completed_action: str
    termination_reason: Optional[str]
    resume_eligible: bool
    total_inner_iterations: int = 0
    cap_binding_detected: bool = False
    cap_binding_indices: Tuple[int, ...] = ()
    penalty_cap_reached: bool = False
    penalty_cap_requested: Optional[float] = None
    inner_options: Optional[Tuple[Tuple[str, ALMTransitionValue], ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "x",
            _strict_transition_float_tuple(self.x, "ALM transition snapshot x"),
        )
        object.__setattr__(
            self,
            "multipliers",
            _strict_transition_float_tuple(
                self.multipliers,
                "ALM transition snapshot multipliers",
            ),
        )
        object.__setattr__(
            self,
            "constraint_names",
            tuple(self.constraint_names),
        )
        if self.constraint_blocks is not None:
            object.__setattr__(
                self,
                "constraint_blocks",
                tuple(self.constraint_blocks),
            )
        if self.sufficient_decrease_multipliers is not None:
            object.__setattr__(
                self,
                "sufficient_decrease_multipliers",
                _strict_transition_float_tuple(
                    self.sufficient_decrease_multipliers,
                    "sufficient_decrease_multipliers",
                ),
            )
        object.__setattr__(
            self, "cap_binding_indices", tuple(self.cap_binding_indices)
        )
        if self.inner_options is not None:
            inner_option_items = tuple(self.inner_options)
            if any(
                not isinstance(key, str) or not key for key, _ in inner_option_items
            ):
                raise ValueError(
                    "ALM transition inner option keys must be non-empty strings"
                )
            if len({key for key, _ in inner_option_items}) != len(inner_option_items):
                raise ValueError("ALM transition inner option keys must be unique")
            object.__setattr__(
                self,
                "inner_options",
                tuple(
                    (key, _frozen_transition_value(value, f"inner_options[{key!r}]"))
                    for key, value in inner_option_items
                ),
            )
        _validate_alm_boundary_fields(
            context="ALM transition snapshot",
            x=self.x,
            multipliers=self.multipliers,
            penalty=self.penalty,
            constraint_names=self.constraint_names,
            constraint_blocks=self.constraint_blocks,
            completed_outer_iterations=self.completed_outer_iterations,
            completed_action=self.completed_action,
            update_feasibility_tol=self.update_feasibility_tol,
            update_stationarity_tol=self.update_stationarity_tol,
            trust_radius=self.trust_radius,
            last_cap_binding_active=self.last_cap_binding_active,
            sufficient_decrease_measure=self.sufficient_decrease_measure,
            sufficient_decrease_multipliers=self.sufficient_decrease_multipliers,
            sufficient_decrease_penalty=self.sufficient_decrease_penalty,
            total_inner_iterations=self.total_inner_iterations,
            cap_binding_detected=self.cap_binding_detected,
            cap_binding_indices=self.cap_binding_indices,
            penalty_cap_reached=self.penalty_cap_reached,
            penalty_cap_requested=self.penalty_cap_requested,
            accepted_state=self.accepted_state,
            geometry_identity=self.geometry_identity,
            best_feasible_dimensions=(
                None
                if self.best_feasible is None
                else (len(self.best_feasible.x), len(self.best_feasible.multipliers))
            ),
        )
        object.__setattr__(
            self,
            "cap_binding_indices",
            tuple(int(index) for index in self.cap_binding_indices),
        )
        if type(self.resume_eligible) is not bool:
            raise ValueError("resume_eligible must be bool")
        if self.resume_eligible and self.termination_reason is not None:
            raise ValueError("resumable ALM snapshot cannot carry a termination_reason")
        if not self.resume_eligible and (
            self.termination_reason is None or not self.termination_reason
        ):
            raise ValueError("terminal ALM snapshot requires termination_reason")
        if self.termination_reason is not None and not isinstance(
            self.termination_reason, str
        ):
            raise ValueError("termination_reason must be a string or None")
        if self.best_feasible is not None:
            _validate_frozen_transition_evaluation(self.best_feasible.evaluation)

_ALM_MAPPING_MARKER = "__alm_mapping__"

_ALM_SEQUENCE_MARKER = "__alm_sequence__"

def _transition_json_value(value: object, path: str = "value") -> ALMTransitionValue:
    """Convert a transition value to a recursively immutable finite value.

    Mapping values use an explicit tagged tuple instead of a mutable ``dict``;
    this keeps the frozen transition DTO immutable while allowing the checkpoint
    adapter to serialize it through its ordinary tuple-to-list conversion. A
    mapping, list or tuple that contains itself raises ``ValueError`` naming
    its path from ``path``; a shared subtree is encoded at each place.
    """
    return _encoded_transition_value(value, path, {})

def _encoded_transition_value(
    value: object, path: str, open_paths: Dict[int, str]
) -> ALMTransitionValue:
    """:func:`_transition_json_value` below ``path``; ``open_paths`` holds the
    containers being encoded around it (a hit is a cycle)."""
    if isinstance(value, np.ndarray):
        # ``tolist`` builds fresh lists, which cannot hold themselves.
        return (
            _ALM_SEQUENCE_MARKER,
            tuple(
                _encoded_transition_value(item, path, open_paths)
                for item in value.tolist()
            ),
        )
    if isinstance(value, np.generic):
        return _encoded_transition_value(value.item(), path, open_paths)
    if isinstance(value, (Mapping, list, tuple)):
        if id(value) in open_paths:
            raise _cyclic_container_error(
                "ALM transition value", path, open_paths[id(value)]
            )
        open_paths[id(value)] = path
        if isinstance(value, Mapping):
            entries: List[Tuple[str, ALMTransitionValue]] = []
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                if not isinstance(key, str):
                    raise TypeError(
                        "ALM transition mappings require string keys; "
                        f"received {type(key).__name__}"
                    )
                entries.append(
                    (key, _encoded_transition_value(item, f"{path}[{key!r}]", open_paths))
                )
            encoded = (_ALM_MAPPING_MARKER, tuple(entries))
        else:
            encoded = (
                _ALM_SEQUENCE_MARKER,
                tuple(
                    _encoded_transition_value(item, f"{path}[{index}]", open_paths)
                    for index, item in enumerate(value)
                ),
            )
        del open_paths[id(value)]
        return encoded
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("ALM transition values must be finite")
        return value
    raise TypeError(
        f"ALM transition evaluation contains a non-JSON value: {type(value).__name__}"
    )

def _canonical_transition_evaluation(
    evaluation: Sequence[Tuple[str, object]],
) -> ALMTransitionEvaluation:
    entries: List[Tuple[str, ALMTransitionValue]] = []
    for item in evaluation:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("ALM transition evaluation must contain key/value pairs")
        key, value = item
        if not isinstance(key, str) or not key:
            raise ValueError("ALM transition evaluation keys must be non-empty strings")
        entries.append((key, _frozen_transition_value(value, f"evaluation[{key!r}]")))
    return tuple(sorted(entries, key=lambda pair: pair[0]))

def _frozen_transition_value(value: object, path: str) -> ALMTransitionValue:
    """Encode ``value`` (named ``path`` in errors) once; an already-encoded
    value passes through.

    Snapshot constructors and checkpoint reloads both call this, so a value
    is never wrapped in a second sequence/mapping marker.
    """
    if _is_frozen_transition_value(value):
        return value
    return _transition_json_value(value, path)

def _is_frozen_transition_value(value: object) -> bool:
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return np.isfinite(value)
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    marker, payload = value
    if marker == _ALM_SEQUENCE_MARKER and isinstance(payload, tuple):
        return all(_is_frozen_transition_value(item) for item in payload)
    if marker == _ALM_MAPPING_MARKER and isinstance(payload, tuple):
        return all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and _is_frozen_transition_value(item[1])
            for item in payload
        )
    return False

def _validate_frozen_transition_value(value: object) -> None:
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("ALM transition values must be finite")
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_frozen_transition_value(item)
        return
    raise TypeError(
        "ALM transition snapshots must contain recursively immutable values; "
        f"received {type(value).__name__}"
    )

def _validate_frozen_transition_evaluation(
    evaluation: ALMTransitionEvaluation,
) -> None:
    if not isinstance(evaluation, tuple):
        raise TypeError("ALM transition evaluation must be an immutable tuple")
    seen_keys: Set[str] = set()
    for item in evaluation:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("ALM transition evaluation entries must be immutable pairs")
        key, value = item
        if not isinstance(key, str) or not key:
            raise ValueError("ALM transition evaluation keys must be non-empty strings")
        if key in seen_keys:
            raise ValueError("ALM transition evaluation keys must be unique")
        seen_keys.add(key)
        _validate_frozen_transition_value(value)

_TRANSITION_OMITTED_JACOBIAN_KEYS = frozenset(
    {
        "constraint_grads",
        "raw_constraint_grads",
        "normalized_constraint_grads",
    }
)


def _snapshot_transition_evaluation(
    evaluation: Mapping[str, object],
) -> ALMTransitionEvaluation:
    """Freeze vector fields. Constraint Jacobians stay out of the checkpoint."""
    projected = {
        key: value
        for key, value in evaluation.items()
        if key not in _TRANSITION_OMITTED_JACOBIAN_KEYS
    }
    return _canonical_transition_evaluation(tuple(projected.items()))

def _restore_transition_json_value(value: object) -> object:
    # Current checkpoints encode the marker as a JSON list after tuple
    # serialization; the dictionary form remains readable for pre-migration
    # in-process snapshots.
    if isinstance(value, Mapping) and set(value) == {_ALM_MAPPING_MARKER}:
        entries = value[_ALM_MAPPING_MARKER]
        return {str(key): _restore_transition_json_value(item) for key, item in entries}
    if isinstance(value, (tuple, list)):
        if (
            len(value) == 2
            and value[0] == _ALM_SEQUENCE_MARKER
            and isinstance(value[1], (tuple, list))
        ):
            return tuple(_restore_transition_json_value(item) for item in value[1])
        if (
            len(value) == 2
            and value[0] == _ALM_MAPPING_MARKER
            and isinstance(value[1], (tuple, list))
        ):
            return {
                str(key): _restore_transition_json_value(item) for key, item in value[1]
            }
        return [_restore_transition_json_value(item) for item in value]
    return value

def _restore_transition_evaluation(
    evaluation: ALMTransitionEvaluation,
) -> dict:
    return {
        str(key): _restore_transition_json_value(value) for key, value in evaluation
    }

def transition_snapshot(
    boundary: ALMOuterBoundary[AcceptedStateT],
    inner_options: Optional[Mapping[str, object]],
) -> ALMTransitionSnapshot[AcceptedStateT]:
    """The checkpoint of one loop boundary.

    ``inner_options`` are the options the loop was started with; a resumed
    process restores them. The best-feasible evaluation keeps no constraint
    Jacobians, and a boundary with a termination reason is not resumable.
    """
    state = boundary.state
    best = state.best_feasible
    best_snapshot = None
    if best is not None:
        best_snapshot = ALMFeasibleIncumbentSnapshot(
            x=tuple(float(value) for value in best.x),
            evaluation=_snapshot_transition_evaluation(best.evaluation),
            multipliers=tuple(float(value) for value in best.multipliers),
            penalty=float(best.penalty),
            accepted_state=best.incumbent_state,
            geometry_identity=best.geometry_identity,
        )
    measure = state.sufficient_decrease_measure
    return ALMTransitionSnapshot(
        x=tuple(float(value) for value in state.x),
        accepted_state=boundary.accepted_state,
        geometry_identity=boundary.geometry_identity,
        constraint_names=boundary.constraint_names,
        constraint_blocks=boundary.constraint_blocks,
        completed_outer_iterations=int(boundary.completed_outer_iterations),
        multipliers=tuple(float(value) for value in state.multipliers),
        penalty=float(state.penalty),
        update_feasibility_tol=float(state.update_feasibility_tol),
        update_stationarity_tol=float(state.update_stationarity_tol),
        trust_radius=(
            None if state.trust_radius is None else float(state.trust_radius)
        ),
        last_cap_binding_active=bool(state.last_cap_binding_active),
        sufficient_decrease_measure=None if measure is None else float(measure),
        sufficient_decrease_multipliers=(
            None
            if measure is None or state.sufficient_decrease_multipliers is None
            else tuple(float(value) for value in state.sufficient_decrease_multipliers)
        ),
        sufficient_decrease_penalty=(
            None
            if measure is None or state.sufficient_decrease_penalty is None
            else float(state.sufficient_decrease_penalty)
        ),
        best_feasible=best_snapshot,
        completed_action=boundary.completed_action,
        termination_reason=boundary.termination_reason,
        resume_eligible=boundary.termination_reason is None,
        total_inner_iterations=int(state.total_inner_iterations),
        cap_binding_detected=bool(state.cap_binding_detected),
        cap_binding_indices=tuple(state.cap_binding_indices),
        penalty_cap_reached=bool(state.penalty_cap_reached),
        penalty_cap_requested=(
            None
            if state.penalty_cap_requested is None
            else float(state.penalty_cap_requested)
        ),
        inner_options=(
            None
            if inner_options is None
            else tuple(
                (str(key), value)
                for key, value in sorted(
                    inner_options.items(), key=lambda pair: str(pair[0])
                )
            )
        ),
    )

def resume_boundary(
    snapshot: ALMTransitionSnapshot[AcceptedStateT],
) -> ALMOuterBoundary[AcceptedStateT]:
    """The loop boundary ``snapshot`` was taken at, for ``resume_from``: as
    read-only as a published one (sequences in the evaluation are tuples)."""
    best = snapshot.best_feasible
    best_feasible = None
    if best is not None:
        best_feasible = ALMFeasibleIncumbent(
            x=np.asarray(best.x, dtype=float),
            evaluation=_restore_transition_evaluation(best.evaluation),
            multipliers=np.asarray(best.multipliers, dtype=float),
            penalty=float(best.penalty),
            inner_result=None,
            incumbent_state=best.accepted_state,
            geometry_identity=best.geometry_identity,
        )
    return ALMOuterBoundary(
        completed_outer_iterations=snapshot.completed_outer_iterations,
        completed_action=snapshot.completed_action,
        termination_reason=snapshot.termination_reason,
        constraint_names=snapshot.constraint_names,
        constraint_blocks=snapshot.constraint_blocks,
        accepted_state=snapshot.accepted_state,
        geometry_identity=snapshot.geometry_identity,
        state=_frozen_loop_state(
            ALMLoopState(
                x=np.asarray(snapshot.x, dtype=float),
                multipliers=np.asarray(snapshot.multipliers, dtype=float),
                penalty=snapshot.penalty,
                update_feasibility_tol=snapshot.update_feasibility_tol,
                update_stationarity_tol=snapshot.update_stationarity_tol,
                trust_radius=snapshot.trust_radius,
                best_feasible=best_feasible,
                total_inner_iterations=snapshot.total_inner_iterations,
                last_cap_binding_active=snapshot.last_cap_binding_active,
                cap_binding_detected=snapshot.cap_binding_detected,
                cap_binding_indices=snapshot.cap_binding_indices,
                penalty_cap_reached=snapshot.penalty_cap_reached,
                penalty_cap_requested=snapshot.penalty_cap_requested,
                sufficient_decrease_measure=snapshot.sufficient_decrease_measure,
                sufficient_decrease_multipliers=(
                    None
                    if snapshot.sufficient_decrease_multipliers is None
                    else np.asarray(
                        snapshot.sufficient_decrease_multipliers, dtype=float
                    )
                ),
                sufficient_decrease_penalty=snapshot.sufficient_decrease_penalty,
            ),
            {},
        ),
    )

def _resumed_inner_options(
    snapshot: ALMTransitionSnapshot[AcceptedStateT],
    inner_options: Optional[Mapping[str, object]],
) -> Optional[Mapping[str, object]]:
    """The checkpointed inner options with this process's ``maxiter``."""
    if snapshot.inner_options is None:
        return inner_options
    resumed = {
        str(key): _restore_transition_json_value(value)
        for key, value in snapshot.inner_options
    }
    if inner_options is not None and "maxiter" in inner_options:
        resumed["maxiter"] = _nonnegative_alm_integer(
            "inner_options.maxiter",
            inner_options["maxiter"],
        )
    return resumed

def _report_transition_snapshot(
    completed_outer_callback: Callable[[ALMTransitionSnapshot[AcceptedStateT]], None],
    inner_options: Optional[Mapping[str, object]],
    boundary: ALMOuterBoundary[AcceptedStateT],
) -> None:
    completed_outer_callback(transition_snapshot(boundary, inner_options))

@dataclass(frozen=True)
class ALMCheckpointing(Generic[AcceptedStateT]):
    """The ``minimize_alm`` arguments of one checkpointed run:
    ``inner_options``, ``resume_from`` and ``on_outer_boundary``."""

    inner_options: Optional[Mapping[str, object]]
    resume_from: Optional[ALMOuterBoundary[AcceptedStateT]]
    on_outer_boundary: Optional[Callable[[ALMOuterBoundary[AcceptedStateT]], None]]

def alm_checkpointing(
    inner_options: Optional[Mapping[str, object]],
    resume_state: Optional[ALMTransitionSnapshot[AcceptedStateT]] = None,
    completed_outer_callback: Optional[Callable[[ALMTransitionSnapshot[AcceptedStateT]], None]] = None,
) -> ALMCheckpointing[AcceptedStateT]:
    """Checkpoint/resume for one ``minimize_alm`` call.

    ``resume_state`` restores the loop and its checkpointed inner options (with
    this call's ``maxiter``); ``completed_outer_callback`` receives a snapshot
    after each completed outer iteration and a terminal one when the run returns.
    """
    if resume_state is not None and not isinstance(resume_state, ALMTransitionSnapshot):
        raise TypeError("resume_state must be an ALMTransitionSnapshot")
    loop_inner_options = (
        inner_options
        if resume_state is None
        else _resumed_inner_options(resume_state, inner_options)
    )
    return ALMCheckpointing(
        inner_options=loop_inner_options,
        resume_from=None if resume_state is None else resume_boundary(resume_state),
        on_outer_boundary=(
            None
            if completed_outer_callback is None
            else partial(
                _report_transition_snapshot,
                completed_outer_callback,
                loop_inner_options,
            )
        ),
    )
