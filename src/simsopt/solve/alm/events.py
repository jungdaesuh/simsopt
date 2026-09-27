"""What the ALM loop hands its observers, and the read-only copy rule.

:func:`~.control.minimize_alm` passes one :class:`ALMOuterStepEvent` per
continuation-step decision to ``on_outer_step``; the event's ``after``, and
every outer boundary's ``state``, is an :class:`ALMLoopState`, whose
incumbent is an :class:`ALMFeasibleIncumbent`. Observers get owned,
read-only copies (:func:`_frozen_event_value`): arrays not writable, mappings
read-only views, lists and tuples tuples; :func:`_writable_copy` undoes that
for plain values. A continuation policy gets the same shape without the
copies (:func:`_borrowed_read_only_value`): read-only views of the loop's
arrays, borrowed for one synchronous call. Copying needs the mappings, lists and tuples to be acyclic;
a shared subtree is fine. Where such data enters the package (the loop's
evaluations, ``ALMPhysics`` extras, a resume boundary),
:func:`_require_acyclic_containers` rejects a cycle, naming its path; the
checkpoint encoder does the same in its own walk. A checkpointable accepted
state carries its geometry identity through :class:`ALMGeometryIdentityCarrier`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Callable, Dict, Generic, Mapping, Optional, Protocol, Sequence, Set, Tuple, TypeVar, runtime_checkable

import numpy as np

from .continuation import ALMInnerSolveOutcome, ALMIterateMeasurement
from .core import ALMConstraintRoutingState, ALMConstraintSignalState

AcceptedStateT = TypeVar("AcceptedStateT")

@runtime_checkable
class ALMGeometryIdentityCarrier(Protocol):
    """Accepted-state protocol for checkpointable geometry identities.

    The ALM core deliberately does not know how a regime serializes geometry.
    A checkpoint-capable regime exposes the resulting content identity through
    this small protocol so that the identity travels with the core incumbent
    instead of being recomputed by a later adapter.
    """

    @property
    def geometry_identity(self) -> str: ...

def _accepted_state_geometry_identity(
    accepted_state: Optional[AcceptedStateT],
) -> Optional[str]:
    """Read an identity only from an explicitly checkpointable carrier."""
    if accepted_state is None:
        return None
    if not isinstance(accepted_state, ALMGeometryIdentityCarrier):
        return None
    identity = accepted_state.geometry_identity
    if not isinstance(identity, str) or not identity:
        raise ValueError("checkpointable accepted-state geometry_identity must be non-empty")
    return identity

def _validate_geometry_identity_pair(
    *,
    accepted_state: Optional[AcceptedStateT],
    geometry_identity: Optional[str],
    context: str,
) -> None:
    """Keep a checkpointable accepted carrier and its identity inseparable."""
    if geometry_identity is not None and accepted_state is None:
        raise ValueError(f"{context} geometry_identity requires accepted_state")
    carrier_identity = _accepted_state_geometry_identity(accepted_state)
    if carrier_identity is not None:
        if geometry_identity is None:
            raise ValueError(
                f"{context} checkpointable accepted_state requires geometry_identity"
            )
        if geometry_identity != carrier_identity:
            raise ValueError(f"{context} geometry_identity does not match accepted_state")

@dataclass(frozen=True)
class ALMFeasibleIncumbent(Generic[AcceptedStateT]):
    x: np.ndarray
    evaluation: Mapping[str, object]
    multipliers: np.ndarray
    penalty: float
    inner_result: object
    incumbent_state: Optional[AcceptedStateT] = None
    geometry_identity: Optional[str] = None

    def __post_init__(self) -> None:
        carrier_identity = _accepted_state_geometry_identity(self.incumbent_state)
        if self.geometry_identity is None:
            object.__setattr__(self, "geometry_identity", carrier_identity)
        elif not isinstance(self.geometry_identity, str) or not self.geometry_identity:
            raise ValueError("ALM best-feasible geometry_identity must be non-empty")
        elif carrier_identity is not None and self.geometry_identity != carrier_identity:
            raise ValueError(
                "ALM best-feasible geometry_identity does not match incumbent_state"
            )
        _validate_geometry_identity_pair(
            accepted_state=self.incumbent_state,
            geometry_identity=self.geometry_identity,
            context="ALM best-feasible incumbent",
        )

@dataclass(frozen=True)
class ALMDualUpdate:
    """Projected multipliers after a dual step and the tightened tolerances."""

    multipliers: np.ndarray
    update_feasibility_tol: float
    update_stationarity_tol: float
    multiplier_cap_binding: bool
    multiplier_cap_binding_indices: Sequence[int]

@dataclass(frozen=True)
class ALMLoopState(Generic[AcceptedStateT]):
    """What the outer loop carries after a decision: the content of a checkpoint.

    ``best_feasible`` omits the inner optimizer result; its ``incumbent_state``
    is the caller's own object, and None in a skipped-inner step, whose state
    the caller snapshots only after ``on_outer_step``. ``sufficient_decrease_*``
    is the latest accepted measure with its multipliers and penalty.
    """

    x: np.ndarray
    multipliers: np.ndarray
    penalty: float
    update_feasibility_tol: float
    update_stationarity_tol: float
    trust_radius: Optional[float]
    best_feasible: Optional[ALMFeasibleIncumbent[AcceptedStateT]]
    total_inner_iterations: int
    last_cap_binding_active: bool
    cap_binding_detected: bool
    cap_binding_indices: Tuple[int, ...]
    penalty_cap_reached: bool
    penalty_cap_requested: Optional[float]
    sufficient_decrease_measure: Optional[float]
    sufficient_decrease_multipliers: Optional[np.ndarray]
    sufficient_decrease_penalty: Optional[float]

@dataclass(frozen=True)
class ALMOuterStepEvent(Generic[AcceptedStateT]):
    """One decision of the outer loop, passed to ``minimize_alm(on_outer_step=...)``.

    Emitted once per continuation step when its decision is final, before a
    terminal result is built. The package docstring lists the fields. An event
    is a read-only snapshot: arrays are not writable and mappings are
    ``MappingProxyType`` views, so it cannot be pickled or deep-copied; an
    observer that stores events converts the fields it needs. Objects the
    evaluator put in its dict that are not arrays, mappings or sequences
    (and the caller's accepted states) are passed by reference.
    """

    outer_iteration: int
    continuation_iteration: int
    constraint_names: Tuple[str, ...]
    action: str
    outer_termination: Optional[str]
    subproblem_limit_reason: Optional[str]
    signal_mismatch_repair: bool
    feasible_stall_count: int
    start_x: np.ndarray
    start: ALMIterateMeasurement
    inner: Optional[ALMInnerSolveOutcome]
    dual_update: Optional[ALMDualUpdate]
    penalty_update: Optional[ALMIterateMeasurement]
    dual_update_penalty_reason: Optional[str]
    after: ALMLoopState[AcceptedStateT]

    @property
    def measured(self) -> ALMIterateMeasurement:
        """The iterate the decision was made on (``start`` when no inner solve ran)."""
        return self.start if self.inner is None else self.inner.measured

_EVENT_CARRIERS = (
    ALMIterateMeasurement,
    ALMConstraintRoutingState,
    ALMConstraintSignalState,
    ALMInnerSolveOutcome,
    ALMDualUpdate,
)

class _FrozenList(tuple):
    """A frozen list: a tuple that :func:`_writable_copy` turns back into a list."""

    __slots__ = ()

# The containers the read-only copy rule descends into.
_DATA_CONTAINERS = (Mapping, list, tuple)

def _cyclic_container_error(context: str, path: str, earlier_path: str) -> ValueError:
    return ValueError(
        f"{context} is cyclic: {path} is {earlier_path}; mappings, lists and "
        "tuples in ALM data must not contain themselves"
    )

def _require_acyclic_containers(value: object, *, context: str, path: str) -> None:
    """Raise ``ValueError`` if a mapping, list or tuple in ``value`` contains
    itself, directly or through others; ``path`` names ``value`` in the message.

    A container reached twice without a cycle (a shared subtree) is fine.
    Other objects are not traversed.
    """
    if isinstance(value, _DATA_CONTAINERS):
        _visit_containers(value, path, {}, set(), context)

def _visit_containers(
    container: object,
    path: str,
    open_paths: Dict[int, str],
    closed: Set[int],
    context: str,
) -> None:
    """Depth-first walk: ``open_paths`` holds the containers on the current
    path (a hit is a cycle), ``closed`` the ones already walked in full."""
    open_paths[id(container)] = path
    items = container.items() if isinstance(container, Mapping) else enumerate(container)
    for key, item in items:
        if not isinstance(item, _DATA_CONTAINERS) or id(item) in closed:
            continue
        item_path = f"{path}[{key!r}]"
        if id(item) in open_paths:
            raise _cyclic_container_error(context, item_path, open_paths[id(item)])
        _visit_containers(item, item_path, open_paths, closed, context)
    del open_paths[id(container)]
    closed.add(id(container))

def _frozen_event_value(value: object, memo: Dict[int, object]) -> object:
    """An owned, read-only copy of ``value`` for an outer-step event.

    Arrays become read-only copies, mappings read-only views of frozen
    values, lists and tuples tuples (a list a :class:`_FrozenList`), and the
    solver's carriers are rebuilt field by field; anything else (scalars,
    strings, None, and any other object the evaluator returned) passes through
    by reference. ``memo`` (by ``id``) keeps shared objects shared and copies
    each once; its keys' objects must stay alive while it is in use.
    ``value`` is acyclic (checked where it entered the package).
    """
    return _read_only_value(value, memo, _read_only_array_copy)

def _borrowed_read_only_value(value: object, memo: Dict[int, object]) -> object:
    """:func:`_frozen_event_value` without the copies: arrays become
    non-writable views, so the owner's arrays keep their flags and the
    owner's later writes show through. For one synchronous call."""
    return _read_only_value(value, memo, _read_only_array_view)

def _read_only_array_copy(array: np.ndarray) -> np.ndarray:
    owned = array.copy()
    owned.setflags(write=False)
    return owned

def _read_only_array_view(array: np.ndarray) -> np.ndarray:
    borrowed = array.view()
    borrowed.setflags(write=False)
    return borrowed

def _read_only_value(
    value: object,
    memo: Dict[int, object],
    read_only_array: Callable[[np.ndarray], np.ndarray],
) -> object:
    """The read-only rule of :func:`_frozen_event_value`, with each array
    made read-only by ``read_only_array``."""
    read_only = memo.get(id(value))
    if read_only is not None:
        return read_only
    if isinstance(value, np.ndarray):
        read_only = read_only_array(value)
    elif isinstance(value, Mapping):
        read_only = MappingProxyType(
            {
                key: _read_only_value(item, memo, read_only_array)
                for key, item in value.items()
            }
        )
    elif isinstance(value, list):
        read_only = _FrozenList(
            _read_only_value(item, memo, read_only_array) for item in value
        )
    elif isinstance(value, tuple):
        read_only = tuple(_read_only_value(item, memo, read_only_array) for item in value)
    elif isinstance(value, _EVENT_CARRIERS):
        read_only = replace(
            value,
            **{
                field.name: _read_only_value(getattr(value, field.name), memo, read_only_array)
                for field in fields(value)
            },
        )
    else:
        return value
    memo[id(value)] = read_only
    return read_only

def _writable_copy(value: object) -> object:
    """The inverse of :func:`_frozen_event_value` for plain values: writable
    array copies, dicts, lists for frozen lists, tuples; others by reference.
    Container subclasses (namedtuples, ``OptimizeResult``) come back as plain
    tuples and dicts; list items are not coerced to float arrays."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _writable_copy(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_writable_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_writable_copy(item) for item in value)
    return value

def _frozen_event_incumbent(
    incumbent: Optional[ALMFeasibleIncumbent[AcceptedStateT]],
    memo: Dict[int, object],
) -> Optional[ALMFeasibleIncumbent[AcceptedStateT]]:
    """The incumbent's solver data, frozen; the caller's state passes through."""
    if incumbent is None:
        return None
    return ALMFeasibleIncumbent(
        x=_frozen_event_value(incumbent.x, memo),
        evaluation=_frozen_event_value(incumbent.evaluation, memo),
        multipliers=_frozen_event_value(incumbent.multipliers, memo),
        penalty=incumbent.penalty,
        inner_result=None,
        incumbent_state=incumbent.incumbent_state,
        geometry_identity=incumbent.geometry_identity,
    )

def _frozen_loop_state(
    state: ALMLoopState[AcceptedStateT],
    memo: Dict[int, object],
) -> ALMLoopState[AcceptedStateT]:
    """``state`` with owned read-only arrays and a frozen incumbent (its inner
    result dropped, the caller's accepted state passed through)."""
    return replace(
        state,
        x=_frozen_event_value(state.x, memo),
        multipliers=_frozen_event_value(state.multipliers, memo),
        best_feasible=_frozen_event_incumbent(state.best_feasible, memo),
        sufficient_decrease_multipliers=_frozen_event_value(
            state.sufficient_decrease_multipliers, memo
        ),
    )
