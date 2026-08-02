"""Typed eager driver for fixed-shape custom quasi-Newton transitions.

The driver owns only transition scheduling. Algorithm modules own state,
stopping semantics, counters, and line-search behavior. Public optimizer
modules adapt callbacks into the typed sink; this module never stores a
callback in a compiled callable or cache entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Protocol, TypeVar

from simsopt_jax.runtime.host_boundary import host_transfer_phase

StateT = TypeVar("StateT")
TransitionT = TypeVar("TransitionT")
ObservationT = TypeVar("ObservationT")
HostObservationT = TypeVar("HostObservationT")
PayloadT = TypeVar("PayloadT")
InitialT = TypeVar("InitialT")
HostObservationT_contra = TypeVar(
    "HostObservationT_contra",
    contravariant=True,
)
PayloadT_contra = TypeVar("PayloadT_contra", contravariant=True)


@dataclass(frozen=True)
class RuntimeLimits:
    """Host-owned total budgets passed to each eager transition."""

    maxiter: int
    maxfun: int | None = None


class ContinueDecision(Enum):
    """Decision returned by a public-owner observation sink."""

    CONTINUE = "continue"
    STOP = "stop"


class TransitionSink(Protocol[HostObservationT_contra, PayloadT_contra]):
    """Observe one host materialization and optionally request termination."""

    def __call__(
        self,
        observation: HostObservationT_contra,
        payload: PayloadT_contra | None,
    ) -> ContinueDecision: ...


@dataclass(frozen=True)
class StepOps(
    Generic[StateT, TransitionT, ObservationT, HostObservationT, PayloadT, InitialT]
):
    """Typed algorithm adapter consumed by :func:`run_eager`.

    ``host_observation`` is the sole observation boundary for one transition.
    ``payload`` is evaluated only for the public owner's observer path; the
    null-sink path may return ``None`` without retaining accepted iterates.
    """

    initialize: Callable[[InitialT, RuntimeLimits], StateT]
    advance: Callable[[StateT, RuntimeLimits], TransitionT]
    state_of: Callable[[TransitionT], StateT]
    observe: Callable[[TransitionT], ObservationT]
    host_observation: Callable[[ObservationT], HostObservationT]
    payload: Callable[[TransitionT], PayloadT | None]
    terminal: Callable[[HostObservationT], bool]
    initial_terminal: Callable[[StateT, RuntimeLimits], bool]


def _continue_sink(
    observation: object,
    payload: object | None,
) -> ContinueDecision:
    del observation, payload
    return ContinueDecision.CONTINUE


def run_eager(
    ops: StepOps[
        StateT,
        TransitionT,
        ObservationT,
        HostObservationT,
        PayloadT,
        InitialT,
    ],
    x0: InitialT,
    limits: RuntimeLimits,
    *,
    sink: TransitionSink[HostObservationT, PayloadT] | None = None,
) -> StateT:
    """Run fixed-shape transitions until algorithm or observer termination.

    The algorithm receives total limits on every call, preserving deferred
    SciPy limit checks. The driver never performs an algorithm-specific budget
    check and materializes exactly one packed observation per transition.
    """

    with host_transfer_phase("initialization"):
        state = ops.initialize(x0, limits)
        initial_terminal = ops.initial_terminal(state, limits)
    if initial_terminal:
        return state
    observation_sink: TransitionSink[HostObservationT, PayloadT]
    if sink is None:
        observation_sink = _continue_sink
    else:
        observation_sink = sink
    while True:
        transition = ops.advance(state, limits)
        state = ops.state_of(transition)
        observation_phase = "advance" if sink is None else "callback"
        with host_transfer_phase(observation_phase):
            host_observation = ops.host_observation(ops.observe(transition))
            payload = None if sink is None else ops.payload(transition)
            decision = observation_sink(host_observation, payload)
        if ops.terminal(host_observation) or decision is ContinueDecision.STOP:
            return state
