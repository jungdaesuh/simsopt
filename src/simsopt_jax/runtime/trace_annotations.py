"""Opt-in timeline annotations for changed-state GPU attribution.

This module owns the immutable phase vocabulary and the context-local tracing
state.  Numerical owners add :func:`device_scope` around existing JAX work;
benchmark code enables those scopes with :func:`trace_session`.  The default
path is disabled and does not add values, callbacks, transfers, or solver
branches to jitted computations.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, NamedTuple, Union

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.8-3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport-compatible string enum base."""


import jax

_SHA256_HEX_LENGTH = 64
_LOWER_HEX = frozenset("0123456789abcdef")


class PhaseId(StrEnum):
    """Stable phase identifiers in the timeline artifact schema."""

    HOST_H2D_SUBMIT = "host.h2d_submit"
    HOST_LINE_SEARCH_CONTROL = "host.line_search_control"
    OPTIMIZER_LIFECYCLE = "optimizer.lifecycle"
    NEWTON_WARM_START = "newton.warm_start"
    NEWTON_SOLVER_CONTROL = "newton.solver_control"
    NEWTON_RESIDUAL_JVP = "newton.residual_jvp"
    NEWTON_JACOBIAN_CONSTRUCTION = "newton.jacobian_construction"
    NEWTON_DENSE_MATERIALIZATION = "newton.dense_materialization"
    NEWTON_LINEAR_SOLVE = "newton.linear_solve"
    NEWTON_LU_FACTOR = "newton.lu_factor"
    NEWTON_REFINEMENT = "newton.refinement"
    ADJOINT_OUTER_VJP_RHS = "adjoint.outer_vjp_rhs"
    ADJOINT_DENSE_MATRIX = "adjoint.dense_matrix"
    ADJOINT_LU_FACTOR = "adjoint.lu_factor"
    ADJOINT_LU_SOLVE = "adjoint.lu_solve"
    ADJOINT_REFINEMENT = "adjoint.refinement"
    ADJOINT_IMPLICIT_COIL_VJP = "adjoint.implicit_coil_vjp"
    BIOTSAVART_FORWARD = "biotsavart.forward"
    BIOTSAVART_VJP = "biotsavart.vjp"
    HOST_D2H_MATERIALIZE = "host.d2h_materialize"
    GNTR_CURRENT_LINEARIZATION = "gntr.current_linearization"
    GNTR_CURRENT_CERTIFICATES = "gntr.current_certificates"
    GNTR_STEIHAUG = "gntr.steihaug"
    GNTR_TRIAL_EVALUATION = "gntr.trial_evaluation"
    GNTR_NONLINEAR_CORRECTION = "gntr.nonlinear_correction"
    GNTR_CORRECTED_CANDIDATE_EVALUATION = "gntr.corrected_candidate_evaluation"
    GNTR_ACCEPTANCE_RADIUS_UPDATE = "gntr.acceptance_radius_update"


GNTR_DIAGNOSTIC_PHASES = (
    PhaseId.GNTR_CURRENT_LINEARIZATION,
    PhaseId.GNTR_CURRENT_CERTIFICATES,
    PhaseId.GNTR_STEIHAUG,
    PhaseId.GNTR_TRIAL_EVALUATION,
    PhaseId.GNTR_NONLINEAR_CORRECTION,
    PhaseId.GNTR_CORRECTED_CANDIDATE_EVALUATION,
    PhaseId.GNTR_ACCEPTANCE_RADIUS_UPDATE,
)
GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        tuple(phase.value for phase in GNTR_DIAGNOSTIC_PHASES),
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class NormalizedJaxIr(NamedTuple):
    """JAX IR rendered without source locations or name-stack metadata."""

    jaxpr: str
    stablehlo: str


def normalized_jax_ir(
    function: Callable[..., object],
    *arguments: object,
) -> NormalizedJaxIr:
    """Render numerical JAX IR while stripping only debug/name metadata."""

    closed_jaxpr = jax.make_jaxpr(function)(*arguments)
    lowered = jax.jit(function).lower(*arguments)
    return NormalizedJaxIr(
        jaxpr=closed_jaxpr.pretty_print(source_info=False, name_stack=False),
        stablehlo=str(lowered.compiler_ir(dialect="stablehlo")),
    )


class EvaluationKind(StrEnum):
    """Lifecycle class for one objective/gradient evaluation."""

    INITIAL = "initial"
    TRIAL = "trial"
    FINAL_REPORTING = "final_reporting"


class EvaluationDisposition(StrEnum):
    """Post-evaluation optimizer decision for a trial candidate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class HostEvent(StrEnum):
    """Correlation points used to derive exclusive host control gaps."""

    EVALUATOR_ENTRY = "evaluator_entry"
    DEVICE_READY = "device_ready"
    EVALUATOR_RETURN = "evaluator_return"


class ProfilerBoundaryOperation(StrEnum):
    """Operation performed at an accepted-iteration profiler boundary."""

    START = "start"
    STOP = "stop"


TraceScalar = Union[str, int, float, bool]


@dataclass(frozen=True)
class HostTraceRecord:
    """One immutable monotonic-clock host interval."""

    sequence: int
    phase: PhaseId
    start_ns: int
    end_ns: int
    depth: int
    attributes: tuple[tuple[str, TraceScalar], ...]


@dataclass(frozen=True)
class EvaluationTraceContext:
    """Identity carried by every record produced during one evaluation."""

    evaluation_id: str
    parameter_sha256: str
    kind: EvaluationKind
    outer_iteration_id: int | None


@dataclass(frozen=True)
class HostEventRecord:
    """One immutable host correlation point."""

    sequence: int
    event: HostEvent
    timestamp_ns: int
    evaluation: EvaluationTraceContext
    attributes: tuple[tuple[str, TraceScalar], ...]


@dataclass(frozen=True)
class EvaluationDispositionRecord:
    """Post-evaluation optimizer decision joined without reevaluation."""

    sequence: int
    evaluation: EvaluationTraceContext
    disposition: EvaluationDisposition
    accepted_iteration_id: int | None


@dataclass(frozen=True)
class ProfilerBoundaryPauseRecord:
    """Immutable duration of one benchmark-supplied profiler boundary hook."""

    iteration_id: int
    operation: ProfilerBoundaryOperation
    start_ns: int
    end_ns: int


@dataclass
class SegmentedProfilerBoundaryAudit:
    """Own start/stop hook ordering and expose immutable timing snapshots."""

    _start_hook: Callable[[int], None]
    _stop_hook: Callable[[int], None]
    _records: list[ProfilerBoundaryPauseRecord] = field(default_factory=list)
    _iteration_ids: list[int] = field(default_factory=list)

    def _begin_iteration(self, iteration_id: int) -> None:
        if iteration_id in self._iteration_ids:
            raise RuntimeError(
                f"profiler boundary hooks cannot retry iteration {iteration_id}"
            )
        self._iteration_ids.append(iteration_id)
        self._invoke(iteration_id, ProfilerBoundaryOperation.START, self._start_hook)

    def _end_iteration(self, iteration_id: int) -> None:
        self._invoke(iteration_id, ProfilerBoundaryOperation.STOP, self._stop_hook)

    def records(self) -> tuple[ProfilerBoundaryPauseRecord, ...]:
        """Return an immutable snapshot in hook invocation order."""

        return tuple(self._records)

    def _invoke(
        self,
        iteration_id: int,
        operation: ProfilerBoundaryOperation,
        hook: Callable[[int], None],
    ) -> None:
        start_ns = time.perf_counter_ns()
        try:
            hook(iteration_id)
        finally:
            self._records.append(
                ProfilerBoundaryPauseRecord(
                    iteration_id=iteration_id,
                    operation=operation,
                    start_ns=start_ns,
                    end_ns=time.perf_counter_ns(),
                )
            )


@dataclass
class HostTraceAudit:
    """Context-local collection of completed host intervals.

    Line-search control gaps are audit-only intervals because profiler APIs
    cannot emit an annotation whose start precedes the current call.
    """

    _records: list[HostTraceRecord] = field(default_factory=list)
    _events: list[HostEventRecord] = field(default_factory=list)
    _dispositions: list[EvaluationDispositionRecord] = field(default_factory=list)
    _iteration_ids: list[int] = field(default_factory=list)

    def append(
        self,
        phase: PhaseId,
        *,
        start_ns: int,
        end_ns: int,
        depth: int,
        attributes: tuple[tuple[str, TraceScalar], ...],
    ) -> None:
        self._records.append(
            HostTraceRecord(
                sequence=len(self._records),
                phase=phase,
                start_ns=start_ns,
                end_ns=end_ns,
                depth=depth,
                attributes=attributes,
            )
        )

    def records(self) -> tuple[HostTraceRecord, ...]:
        """Return an immutable snapshot in completion order."""

        return tuple(self._records)

    def append_event(
        self,
        event: HostEvent,
        *,
        timestamp_ns: int,
        evaluation: EvaluationTraceContext,
        attributes: tuple[tuple[str, TraceScalar], ...],
    ) -> None:
        prior = tuple(
            record.event
            for record in self._events
            if record.evaluation.evaluation_id == evaluation.evaluation_id
        )
        expected = tuple(HostEvent)
        if len(prior) >= len(expected) or event is not expected[len(prior)]:
            raise RuntimeError(
                "host lifecycle events must be emitted exactly once in "
                "ENTRY, READY, RETURN order"
            )
        if (
            event is HostEvent.EVALUATOR_ENTRY
            and evaluation.kind is EvaluationKind.TRIAL
            and self._events
            and self._events[-1].event is HostEvent.EVALUATOR_RETURN
        ):
            previous = self._events[-1]
            gap_attributes = dict(attributes)
            gap_attributes.update(
                {
                    "previous_evaluation_id": previous.evaluation.evaluation_id,
                    "next_evaluation_id": evaluation.evaluation_id,
                }
            )
            self.append(
                PhaseId.HOST_LINE_SEARCH_CONTROL,
                start_ns=previous.timestamp_ns,
                end_ns=timestamp_ns,
                depth=0,
                attributes=_normalized_attributes(gap_attributes),
            )
        self._events.append(
            HostEventRecord(
                sequence=len(self._events),
                event=event,
                timestamp_ns=timestamp_ns,
                evaluation=evaluation,
                attributes=attributes,
            )
        )

    def events(self) -> tuple[HostEventRecord, ...]:
        """Return an immutable snapshot of evaluator correlation points."""

        return tuple(self._events)

    def append_disposition(
        self,
        evaluation: EvaluationTraceContext,
        *,
        disposition: EvaluationDisposition,
        accepted_iteration_id: int | None,
    ) -> None:
        if any(
            record.evaluation.evaluation_id == evaluation.evaluation_id
            for record in self._dispositions
        ):
            raise RuntimeError("an evaluation may receive only one disposition")
        self._dispositions.append(
            EvaluationDispositionRecord(
                sequence=len(self._dispositions),
                evaluation=evaluation,
                disposition=disposition,
                accepted_iteration_id=accepted_iteration_id,
            )
        )

    def dispositions(self) -> tuple[EvaluationDispositionRecord, ...]:
        """Return an immutable snapshot of optimizer decisions."""

        return tuple(self._dispositions)

    def register_iteration(self, iteration_id: int) -> None:
        """Reject duplicate accepted-iteration profiler envelopes."""

        if iteration_id in self._iteration_ids:
            raise RuntimeError(f"duplicate accepted iteration ID {iteration_id}")
        self._iteration_ids.append(iteration_id)


_TRACE_ENABLED: ContextVar[bool] = ContextVar(
    "simsopt_jax_trace_annotations_enabled", default=False
)
_DEVICE_PHASE_STACK: ContextVar[tuple[PhaseId, ...]] = ContextVar(
    "simsopt_jax_device_phase_stack", default=()
)
_HOST_TRACE_AUDIT: ContextVar[HostTraceAudit | None] = ContextVar(
    "simsopt_jax_host_trace_audit", default=None
)
_HOST_TRACE_DEPTH: ContextVar[int] = ContextVar(
    "simsopt_jax_host_trace_depth", default=0
)
_EVALUATION_CONTEXT: ContextVar[EvaluationTraceContext | None] = ContextVar(
    "simsopt_jax_timeline_evaluation_context", default=None
)
_OUTER_ITERATION_ID: ContextVar[int | None] = ContextVar(
    "simsopt_jax_timeline_outer_iteration_id", default=None
)
_SEGMENTED_PROFILER_BOUNDARY_AUDIT: ContextVar[
    SegmentedProfilerBoundaryAudit | None
] = ContextVar("simsopt_jax_segmented_profiler_boundary_audit", default=None)
_SEGMENTED_PROFILER_ACTIVE_ITERATION: ContextVar[int | None] = ContextVar(
    "simsopt_jax_segmented_profiler_active_iteration", default=None
)


def annotations_enabled() -> bool:
    """Whether source-local trace annotations are active in this context."""

    return _TRACE_ENABLED.get()


def current_device_phase_stack() -> tuple[PhaseId, ...]:
    """Return the immutable device-scope ancestry for this trace context."""

    return _DEVICE_PHASE_STACK.get() if annotations_enabled() else ()


@contextmanager
def segmented_profiler_boundaries(
    start_hook: Callable[[int], None],
    stop_hook: Callable[[int], None],
) -> Iterator[SegmentedProfilerBoundaryAudit]:
    """Install context-local hooks around accepted-iteration profiler spans.

    Hooks are benchmark-owned and receive the accepted iteration ID. Nested
    controllers are rejected so each iteration has exactly one start/stop pair.
    """

    if _SEGMENTED_PROFILER_BOUNDARY_AUDIT.get() is not None:
        raise RuntimeError("segmented profiler boundary controllers cannot nest")
    audit = SegmentedProfilerBoundaryAudit(start_hook, stop_hook)
    audit_token = _SEGMENTED_PROFILER_BOUNDARY_AUDIT.set(audit)
    active_token = _SEGMENTED_PROFILER_ACTIVE_ITERATION.set(None)
    try:
        yield audit
    finally:
        _SEGMENTED_PROFILER_ACTIVE_ITERATION.reset(active_token)
        _SEGMENTED_PROFILER_BOUNDARY_AUDIT.reset(audit_token)


@contextmanager
def trace_session() -> Iterator[HostTraceAudit]:
    """Enable device scopes and collect host intervals in this context."""

    audit = HostTraceAudit()
    enabled_token = _TRACE_ENABLED.set(True)
    device_phase_token = _DEVICE_PHASE_STACK.set(())
    audit_token = _HOST_TRACE_AUDIT.set(audit)
    depth_token = _HOST_TRACE_DEPTH.set(0)
    evaluation_token = _EVALUATION_CONTEXT.set(None)
    iteration_token = _OUTER_ITERATION_ID.set(None)
    try:
        yield audit
    finally:
        _OUTER_ITERATION_ID.reset(iteration_token)
        _EVALUATION_CONTEXT.reset(evaluation_token)
        _HOST_TRACE_DEPTH.reset(depth_token)
        _HOST_TRACE_AUDIT.reset(audit_token)
        _DEVICE_PHASE_STACK.reset(device_phase_token)
        _TRACE_ENABLED.reset(enabled_token)


@contextmanager
def evaluation_context(
    evaluation_id: str,
    parameter_sha256: str,
    kind: EvaluationKind,
) -> Iterator[EvaluationTraceContext]:
    """Correlate source-local records with one canonical parameter value."""

    normalized_evaluation_id = str(evaluation_id)
    normalized_sha256 = str(parameter_sha256)
    if not normalized_evaluation_id:
        raise ValueError("evaluation_id must be non-empty")
    if len(normalized_sha256) != _SHA256_HEX_LENGTH or any(
        character not in _LOWER_HEX for character in normalized_sha256
    ):
        raise ValueError("parameter_sha256 must be a lowercase SHA-256 digest")
    context = EvaluationTraceContext(
        evaluation_id=normalized_evaluation_id,
        parameter_sha256=normalized_sha256,
        kind=kind,
        outer_iteration_id=_OUTER_ITERATION_ID.get(),
    )
    token = _EVALUATION_CONTEXT.set(context)
    try:
        yield context
    finally:
        _EVALUATION_CONTEXT.reset(token)


@contextmanager
def accepted_iteration_span(iteration_id: int) -> Iterator[None]:
    """Emit the profiler step envelope for one accepted outer iteration."""

    if isinstance(iteration_id, bool) or not isinstance(iteration_id, int):
        raise TypeError("accepted iteration ID must be an integer")
    normalized_iteration_id = int(iteration_id)
    if normalized_iteration_id <= 0:
        raise ValueError("accepted iteration ID must be positive")
    audit = _HOST_TRACE_AUDIT.get()
    if audit is not None:
        audit.register_iteration(normalized_iteration_id)
    token = _OUTER_ITERATION_ID.set(normalized_iteration_id)
    try:
        if not annotations_enabled():
            yield
        else:
            boundary_audit = _SEGMENTED_PROFILER_BOUNDARY_AUDIT.get()
            active_iteration = _SEGMENTED_PROFILER_ACTIVE_ITERATION.get()
            if boundary_audit is None:
                with jax.profiler.StepTraceAnnotation(
                    "optimizer.accepted_iteration", step_num=normalized_iteration_id
                ):
                    yield
            else:
                if active_iteration is not None:
                    raise RuntimeError(
                        "accepted-iteration profiler boundaries cannot nest"
                    )
                active_iteration_token = _SEGMENTED_PROFILER_ACTIVE_ITERATION.set(
                    normalized_iteration_id
                )
                try:
                    boundary_audit._begin_iteration(normalized_iteration_id)
                    try:
                        with jax.profiler.StepTraceAnnotation(
                            "optimizer.accepted_iteration",
                            step_num=normalized_iteration_id,
                        ):
                            yield
                    finally:
                        boundary_audit._end_iteration(normalized_iteration_id)
                finally:
                    _SEGMENTED_PROFILER_ACTIVE_ITERATION.reset(active_iteration_token)
    finally:
        _OUTER_ITERATION_ID.reset(token)


def current_evaluation_context() -> EvaluationTraceContext | None:
    """Return the active immutable evaluation identity, if any."""

    return _EVALUATION_CONTEXT.get()


@contextmanager
def device_scope(phase: PhaseId) -> Iterator[None]:
    """Attach a stable JAX name-stack scope without changing computation."""

    if not annotations_enabled():
        yield
        return
    stack_token = _DEVICE_PHASE_STACK.set((*_DEVICE_PHASE_STACK.get(), phase))
    try:
        with jax.named_scope(phase.value):
            yield
    finally:
        _DEVICE_PHASE_STACK.reset(stack_token)


def _normalized_attributes(
    attributes: Mapping[str, TraceScalar] | None,
) -> tuple[tuple[str, TraceScalar], ...]:
    if attributes is None:
        return ()
    return tuple(sorted((str(key), value) for key, value in attributes.items()))


def _attributes_with_evaluation(
    attributes: Mapping[str, TraceScalar] | None,
) -> tuple[tuple[str, TraceScalar], ...]:
    combined = {} if attributes is None else dict(attributes)
    evaluation = current_evaluation_context()
    if evaluation is not None:
        combined.update(
            {
                "evaluation_id": evaluation.evaluation_id,
                "evaluation_kind": evaluation.kind.value,
                "parameter_sha256": evaluation.parameter_sha256,
            }
        )
        if evaluation.outer_iteration_id is not None:
            combined["outer_iteration_id"] = evaluation.outer_iteration_id
    return _normalized_attributes(combined)


@contextmanager
def host_span(
    phase: PhaseId,
    *,
    attributes: Mapping[str, TraceScalar] | None = None,
) -> Iterator[None]:
    """Emit a profiler host annotation and monotonic diagnostic interval."""

    if not annotations_enabled():
        yield
        return

    normalized = _attributes_with_evaluation(attributes)
    profiler_attributes = dict(normalized)
    audit = _HOST_TRACE_AUDIT.get()
    depth = _HOST_TRACE_DEPTH.get()
    depth_token = _HOST_TRACE_DEPTH.set(depth + 1)
    start_ns = time.perf_counter_ns()
    try:
        with jax.profiler.TraceAnnotation(phase.value, **profiler_attributes):
            yield
    finally:
        end_ns = time.perf_counter_ns()
        _HOST_TRACE_DEPTH.reset(depth_token)
        if audit is not None:
            audit.append(
                phase,
                start_ns=start_ns,
                end_ns=end_ns,
                depth=depth,
                attributes=normalized,
            )


def record_host_event(
    event: HostEvent,
    *,
    attributes: Mapping[str, TraceScalar] | None = None,
) -> None:
    """Record an evaluator correlation point in the active trace session."""

    if not annotations_enabled():
        return
    audit = _HOST_TRACE_AUDIT.get()
    evaluation = _EVALUATION_CONTEXT.get()
    if audit is None or evaluation is None:
        raise RuntimeError("host events require an active evaluation trace context")
    normalized = _attributes_with_evaluation(attributes)
    timestamp_ns = time.perf_counter_ns()
    with jax.profiler.TraceAnnotation(
        f"optimizer.lifecycle.{event.value}", **dict(normalized)
    ):
        pass
    audit.append_event(
        event,
        timestamp_ns=timestamp_ns,
        evaluation=evaluation,
        attributes=normalized,
    )


def record_evaluation_disposition(
    evaluation: EvaluationTraceContext,
    disposition: EvaluationDisposition,
    *,
    accepted_iteration_id: int | None,
) -> None:
    """Join one trial to its optimizer decision without evaluating it again."""

    if not annotations_enabled():
        return
    audit = _HOST_TRACE_AUDIT.get()
    if audit is None:
        raise RuntimeError("evaluation dispositions require an active trace session")
    if evaluation.kind is not EvaluationKind.TRIAL:
        raise ValueError("only trial evaluations may receive dispositions")
    if evaluation.outer_iteration_id is None:
        raise ValueError("trial dispositions require an outer iteration ID")
    if disposition is EvaluationDisposition.ACCEPTED and accepted_iteration_id is None:
        raise ValueError("accepted evaluations require an accepted iteration ID")
    if (
        disposition is EvaluationDisposition.ACCEPTED
        and accepted_iteration_id != evaluation.outer_iteration_id
    ):
        raise ValueError(
            "accepted iteration ID must match the evaluation outer iteration ID"
        )
    if (
        disposition is EvaluationDisposition.REJECTED
        and accepted_iteration_id is not None
    ):
        raise ValueError("rejected evaluations cannot have an accepted iteration ID")
    audit.append_disposition(
        evaluation,
        disposition=disposition,
        accepted_iteration_id=accepted_iteration_id,
    )


__all__ = (
    "GNTR_DIAGNOSTIC_PHASES",
    "GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256",
    "EvaluationDisposition",
    "EvaluationDispositionRecord",
    "EvaluationKind",
    "EvaluationTraceContext",
    "HostEvent",
    "HostEventRecord",
    "HostTraceAudit",
    "HostTraceRecord",
    "NormalizedJaxIr",
    "PhaseId",
    "ProfilerBoundaryOperation",
    "ProfilerBoundaryPauseRecord",
    "SegmentedProfilerBoundaryAudit",
    "TraceScalar",
    "accepted_iteration_span",
    "annotations_enabled",
    "current_evaluation_context",
    "device_scope",
    "evaluation_context",
    "host_span",
    "normalized_jax_ir",
    "record_evaluation_disposition",
    "record_host_event",
    "segmented_profiler_boundaries",
    "trace_session",
)
