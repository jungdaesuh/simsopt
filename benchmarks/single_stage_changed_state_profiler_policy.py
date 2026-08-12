"""Immutable profiler-capacity policy for changed-state timeline evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol

GPU_PROFILER_EVENT_CAPACITY: Final = 33_554_432
TRACE_VIEWER_MAX_EVENTS: Final = 67_108_864
TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT: Final = "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS"
CUPTI_ACTIVITY_DROP_WARNING: Final = "Already too many activity events, drop the buffer"


@dataclass(frozen=True, slots=True)
class ProfilerPolicy:
    """Exact collection policy for one profiled or unprofiled execution."""

    enabled: bool
    host_tracer_level: int | None
    python_tracer_level: int | None
    device_tracing: Literal["jax_default"] | None
    trace_viewer_max_events: int | None
    advanced_configuration: tuple[tuple[str, int], ...]


PROFILED_PROFILER_POLICY: Final = ProfilerPolicy(
    enabled=True,
    host_tracer_level=1,
    python_tracer_level=0,
    device_tracing="jax_default",
    trace_viewer_max_events=TRACE_VIEWER_MAX_EVENTS,
    advanced_configuration=(
        ("gpu_max_activity_api_events", GPU_PROFILER_EVENT_CAPACITY),
        ("gpu_max_callback_api_events", GPU_PROFILER_EVENT_CAPACITY),
    ),
)
CONTROL_PROFILER_POLICY: Final = ProfilerPolicy(
    enabled=False,
    host_tracer_level=None,
    python_tracer_level=None,
    device_tracing=None,
    trace_viewer_max_events=None,
    advanced_configuration=(),
)


class MutableProfileOptions(Protocol):
    host_tracer_level: int
    python_tracer_level: int
    advanced_configuration: dict[str, int]


def profiler_policy(profiled: bool) -> ProfilerPolicy:
    """Return the immutable policy selected by whether tracing is enabled."""

    return PROFILED_PROFILER_POLICY if profiled else CONTROL_PROFILER_POLICY


def profiler_policy_document(policy: ProfilerPolicy) -> dict[str, object]:
    """Serialize the policy into its canonical receipt representation."""

    return {
        "enabled": policy.enabled,
        "host_tracer_level": policy.host_tracer_level,
        "python_tracer_level": policy.python_tracer_level,
        "device_tracing": policy.device_tracing,
        "trace_viewer_max_events": policy.trace_viewer_max_events,
        "advanced_configuration": dict(policy.advanced_configuration),
    }


def parse_profiler_policy(document: object) -> ProfilerPolicy:
    """Accept only one of the two exact protocol policies."""

    if document == profiler_policy_document(PROFILED_PROFILER_POLICY):
        return PROFILED_PROFILER_POLICY
    if document == profiler_policy_document(CONTROL_PROFILER_POLICY):
        return CONTROL_PROFILER_POLICY
    raise ValueError("profiler policy differs from the changed-state schema")


def build_jax_profiler_options(
    factory: Callable[[], MutableProfileOptions], policy: ProfilerPolicy
) -> MutableProfileOptions:
    """Materialize exact JAX options while retaining default device tracing."""

    if policy != PROFILED_PROFILER_POLICY:
        raise ValueError("JAX profiler options require the profiled policy")
    options = factory()
    if (
        policy.host_tracer_level is None
        or policy.python_tracer_level is None
        or policy.device_tracing != "jax_default"
    ):
        raise ValueError(
            "profiled policy requires host and Python levels plus JAX-default "
            "device tracing"
        )
    options.host_tracer_level = policy.host_tracer_level
    options.python_tracer_level = policy.python_tracer_level
    options.advanced_configuration = dict(policy.advanced_configuration)
    return options
