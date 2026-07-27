"""Fail-closed contract for matched JAX fast-versus-parity benchmarks.

The artifact accepted here is performance-only evidence.  It can support a
device-specific default decision, but it can never certify native/JAX numerical
parity.  This module intentionally depends only on the Python standard library
so CI can audit a retained JSON artifact without importing JAX or SIMSOPT.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType

BENCHMARK_SCHEMA_VERSION = 3
BENCHMARK_RULE_VERSION = 2
BENCHMARK_EVIDENCE_KIND = "jax_example_execution_mode_benchmark_noncertifying"
WARM_PAIR_COUNT = 7

REPRESENTATIVE_WORKLOAD_IDS = (
    "traceable-least-squares",
    "curve-length-optimization",
    "surface-geometry-optimization",
    "coil-flux-optimization",
    "fieldline-and-particle-tracing",
)

REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES: Mapping[str, int] = MappingProxyType(
    {workload_id: 0 for workload_id in REPRESENTATIVE_WORKLOAD_IDS}
)

METRIC_OWNERS: Mapping[str, str] = MappingProxyType(
    {
        "dense_materialized_bytes": "checked_in_workload_contract",
        "elapsed_seconds": "parent_monotonic_process_wall",
        "gpu_memory": "nvidia_smi_process_poll_or_unavailable",
        "peak_host_rss_bytes": "linux_proc_process_tree_poll",
        "synchronization": "validated_json_after_process_exit",
    }
)

BENCHMARK_RULE: Mapping[str, int | float] = MappingProxyType(
    {
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 1729,
        "cold_time_ratio_max": 1.25,
        "gpu_memory_ratio_max": 1.25,
        "gpu_concurrent_memory_fraction_max": 0.05,
        "gpu_concurrent_sample_count": 5,
        "gpu_concurrent_utilization_percent_max": 5,
        "host_rss_ratio_max": 1.25,
        "warm_median_speedup_min": 1.05,
        "warm_speedup_lower_bound_min": 1.0,
        "warm_pair_count": WARM_PAIR_COUNT,
    }
)

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_PROFILE_KEYS = frozenset(("fast", "parity"))


class BenchmarkContractError(ValueError):
    """A benchmark document cannot support a device-default decision."""


@dataclass(frozen=True)
class WorkloadSummary:
    """Checked performance and memory statistics for one workload."""

    workload_id: str
    warm_median_speedup: float
    warm_speedup_lower_bound: float
    cold_time_ratio: float
    peak_host_rss_ratio: float
    peak_gpu_memory_ratio: float | None


@dataclass(frozen=True)
class PromotionDecision:
    """Immutable result of validating and evaluating one retained artifact."""

    device: str
    promoted: bool
    reasons: tuple[str, ...]
    workloads: tuple[WorkloadSummary, ...]


@dataclass(frozen=True)
class _OutcomeMetrics:
    elapsed_seconds: float | None
    peak_host_rss_bytes: int
    peak_gpu_memory_bytes: int | None
    gpu_memory_available: bool


@dataclass(frozen=True)
class _ProfileMetrics:
    cold: _OutcomeMetrics
    warmup: _OutcomeMetrics
    warm: tuple[_OutcomeMetrics, ...]

    @property
    def peak_host_rss_bytes(self) -> int:
        return max(
            outcome.peak_host_rss_bytes
            for outcome in (self.cold, self.warmup, *self.warm)
        )

    @property
    def peak_gpu_memory_bytes(self) -> int | None:
        outcomes = (self.cold, self.warmup, *self.warm)
        if not all(outcome.gpu_memory_available for outcome in outcomes):
            return None
        values = tuple(
            outcome.peak_gpu_memory_bytes
            for outcome in outcomes
            if outcome.peak_gpu_memory_bytes is not None
        )
        return max(values) if values else None


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise BenchmarkContractError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise BenchmarkContractError(f"{context} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BenchmarkContractError(
            f"{context} keys do not match schema; missing={missing}, extra={extra}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkContractError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise BenchmarkContractError(f"{context} must be at least {minimum}")
    return value


def _positive_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkContractError(f"{context} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BenchmarkContractError(f"{context} must be a positive finite number")
    return result


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != _SHA256_LENGTH or any(
        character not in _HEX_DIGITS for character in digest
    ):
        raise BenchmarkContractError(
            f"{context} must be a lowercase hexadecimal SHA-256 digest"
        )
    return digest


def _require_equal(value: object, expected: object, context: str) -> None:
    if value != expected:
        raise BenchmarkContractError(f"{context} must be {expected!r}, got {value!r}")


def _validate_top_level(document: Mapping[str, object]) -> str:
    _exact_keys(
        document,
        frozenset(
            (
                "schema_version",
                "rule_version",
                "rule",
                "evidence_kind",
                "certification_eligible",
                "metric_owners",
                "device",
                "profiles",
                "workload_ids",
                "provenance",
                "workloads",
            )
        ),
        "artifact",
    )
    _require_equal(
        document.get("schema_version"),
        BENCHMARK_SCHEMA_VERSION,
        "schema_version",
    )
    _require_equal(
        document.get("rule_version"),
        BENCHMARK_RULE_VERSION,
        "rule_version",
    )
    rule = _mapping(document.get("rule"), "rule")
    if dict(rule) != dict(BENCHMARK_RULE):
        raise BenchmarkContractError(
            "rule must match the checked-in version; threshold edits require "
            "a schema/rule version change"
        )
    _require_equal(
        document.get("evidence_kind"), BENCHMARK_EVIDENCE_KIND, "evidence_kind"
    )
    if document.get("certification_eligible") is not False:
        raise BenchmarkContractError(
            "execution-mode benchmark evidence must remain non-certifying"
        )
    metric_owners = _mapping(document.get("metric_owners"), "metric_owners")
    if dict(metric_owners) != dict(METRIC_OWNERS):
        raise BenchmarkContractError(
            "metric_owners must match the checked-in measurement ownership contract"
        )
    device = _string(document.get("device"), "device")
    if device not in ("cpu", "gpu"):
        raise BenchmarkContractError("device must be 'cpu' or 'gpu'")

    profiles = _mapping(document.get("profiles"), "profiles")
    _exact_keys(profiles, _PROFILE_KEYS, "profiles")
    for intent in ("fast", "parity"):
        _require_equal(
            profiles.get(intent), f"jax_{device}_{intent}", f"profiles.{intent}"
        )

    workload_ids = tuple(
        _string(value, f"workload_ids[{index}]")
        for index, value in enumerate(
            _sequence(document.get("workload_ids"), "workload_ids")
        )
    )
    if workload_ids != REPRESENTATIVE_WORKLOAD_IDS:
        raise BenchmarkContractError(
            "workload_ids must equal the checked-in representative inventory"
        )
    return device


def _validate_provenance(
    raw_provenance: object,
    *,
    device: str,
) -> str:
    provenance = _mapping(raw_provenance, "provenance")
    required = frozenset(
        {
            "run_id",
            "started_at_utc",
            "repo_commit",
            "source_tree_sha256",
            "manifest_sha256",
            "python_version",
            "jax_version",
            "jaxlib_version",
            "xla_version",
            "host",
            "device",
            "concurrent_load_preflight",
        }
    )
    _exact_keys(provenance, required, "provenance")
    _string(provenance.get("run_id"), "provenance.run_id")
    _string(provenance.get("started_at_utc"), "provenance.started_at_utc")
    commit = _string(provenance.get("repo_commit"), "provenance.repo_commit")
    if len(commit) != 40 or any(character not in _HEX_DIGITS for character in commit):
        raise BenchmarkContractError(
            "provenance.repo_commit must be a lowercase 40-character Git hash"
        )
    source_tree_sha256 = _sha256(
        provenance.get("source_tree_sha256"), "provenance.source_tree_sha256"
    )
    _sha256(provenance.get("manifest_sha256"), "provenance.manifest_sha256")
    for field in ("python_version", "jax_version", "jaxlib_version", "xla_version"):
        _string(provenance.get(field), f"provenance.{field}")

    host = _mapping(provenance.get("host"), "provenance.host")
    _exact_keys(host, frozenset(("hostname", "cpu_model", "os")), "provenance.host")
    for field in ("hostname", "cpu_model", "os"):
        _string(host.get(field), f"provenance.host.{field}")

    device_record = _mapping(provenance.get("device"), "provenance.device")
    _exact_keys(
        device_record,
        frozenset(("kind", "model", "uuid", "driver", "clock_policy", "power_policy")),
        "provenance.device",
    )
    _require_equal(device_record.get("kind"), device, "provenance.device.kind")
    _string(device_record.get("model"), "provenance.device.model")
    _string(device_record.get("driver"), "provenance.device.driver")
    _string(device_record.get("clock_policy"), "provenance.device.clock_policy")
    _string(device_record.get("power_policy"), "provenance.device.power_policy")
    if device == "gpu":
        _string(device_record.get("uuid"), "provenance.device.uuid")
    elif device_record.get("uuid") is not None:
        raise BenchmarkContractError("provenance.device.uuid must be null for CPU")

    preflight = _mapping(
        provenance.get("concurrent_load_preflight"),
        "provenance.concurrent_load_preflight",
    )
    _exact_keys(
        preflight,
        frozenset(("status", "detail")),
        "provenance.concurrent_load_preflight",
    )
    _require_equal(
        preflight.get("status"), "pass", "provenance.concurrent_load_preflight.status"
    )
    _string(preflight.get("detail"), "provenance.concurrent_load_preflight.detail")
    return source_tree_sha256


def _validate_gpu_memory(
    raw_gpu_memory: object,
    *,
    device: str,
    context: str,
) -> tuple[int | None, bool]:
    gpu_memory = _mapping(raw_gpu_memory, f"{context}.gpu_memory")
    _exact_keys(
        gpu_memory,
        frozenset(("status", "peak_process_bytes", "owner")),
        f"{context}.gpu_memory",
    )
    status = _string(gpu_memory.get("status"), f"{context}.gpu_memory.status")
    owner = _string(gpu_memory.get("owner"), f"{context}.gpu_memory.owner")
    value = gpu_memory.get("peak_process_bytes")
    if device == "cpu":
        _require_equal(status, "not_applicable", f"{context}.gpu_memory.status")
        _require_equal(value, None, f"{context}.gpu_memory.peak_process_bytes")
        _require_equal(owner, "not_applicable", f"{context}.gpu_memory.owner")
        return None, True
    if status == "unavailable":
        _require_equal(value, None, f"{context}.gpu_memory.peak_process_bytes")
        return None, False
    if status != "available":
        raise BenchmarkContractError(
            f"{context}.gpu_memory.status must be 'available' or 'unavailable'"
        )
    return (
        _integer(value, f"{context}.gpu_memory.peak_process_bytes", minimum=1),
        True,
    )


def _validate_outcome(
    raw_outcome: object,
    *,
    device: str,
    intent: str,
    phase: str,
    pair_index: int | None,
    order_position: int,
    input_sha256: str,
    source_tree_sha256: str,
    cache_identity: str,
    environment_sha256: str,
    max_dense_jacobian_bytes: int,
    expected_dense_materialized_bytes: int,
    context: str,
) -> _OutcomeMetrics:
    outcome = _mapping(raw_outcome, context)
    expected_profile = f"jax_{device}_{intent}"
    _require_equal(outcome.get("profile"), expected_profile, f"{context}.profile")
    _require_equal(outcome.get("intent"), intent, f"{context}.intent")
    _require_equal(outcome.get("phase"), phase, f"{context}.phase")
    _require_equal(outcome.get("pair_index"), pair_index, f"{context}.pair_index")
    _require_equal(
        outcome.get("order_position"), order_position, f"{context}.order_position"
    )
    measured = phase != "warmup"
    _require_equal(outcome.get("measured"), measured, f"{context}.measured")
    _require_equal(outcome.get("input_sha256"), input_sha256, f"{context}.input_sha256")
    _require_equal(
        outcome.get("source_tree_sha256"),
        source_tree_sha256,
        f"{context}.source_tree_sha256",
    )
    _require_equal(
        outcome.get("cache_identity"), cache_identity, f"{context}.cache_identity"
    )
    _require_equal(outcome.get("returncode"), 0, f"{context}.returncode")
    _require_equal(outcome.get("termination"), "normal", f"{context}.termination")
    _require_equal(
        outcome.get("scientific_success"), True, f"{context}.scientific_success"
    )
    _require_equal(
        outcome.get("cache_load_compatible"),
        True,
        f"{context}.cache_load_compatible",
    )
    _require_equal(
        outcome.get("runtime_environment_compatible"),
        True,
        f"{context}.runtime_environment_compatible",
    )
    _require_equal(
        outcome.get("backend_mode"), expected_profile, f"{context}.backend_mode"
    )
    _require_equal(outcome.get("platform"), device, f"{context}.platform")
    _require_equal(outcome.get("precision"), "fp64", f"{context}.precision")
    _require_equal(
        outcome.get("timing_synchronized"), True, f"{context}.timing_synchronized"
    )
    _require_equal(
        outcome.get("synchronization_owner"),
        METRIC_OWNERS["synchronization"],
        f"{context}.synchronization_owner",
    )
    elapsed_value = outcome.get("elapsed_seconds")
    diagnostic_wall_seconds = _positive_number(
        outcome.get("diagnostic_wall_seconds"),
        f"{context}.diagnostic_wall_seconds",
    )
    if measured:
        elapsed_seconds: float | None = _positive_number(
            elapsed_value, f"{context}.elapsed_seconds"
        )
        if elapsed_seconds != diagnostic_wall_seconds:
            raise BenchmarkContractError(
                f"{context}.elapsed_seconds must equal diagnostic_wall_seconds"
            )
    else:
        _require_equal(elapsed_value, None, f"{context}.warmup.elapsed_seconds")
        elapsed_seconds = None
    peak_host_rss_bytes = _integer(
        outcome.get("peak_host_rss_bytes"),
        f"{context}.peak_host_rss_bytes",
        minimum=1,
    )
    peak_gpu_memory_bytes, gpu_memory_available = _validate_gpu_memory(
        outcome.get("gpu_memory"), device=device, context=context
    )
    dense_materialized_bytes = _integer(
        outcome.get("dense_materialized_bytes"),
        f"{context}.dense_materialized_bytes",
        minimum=0,
    )
    if dense_materialized_bytes > max_dense_jacobian_bytes:
        raise BenchmarkContractError(
            f"{context}.dense_materialized_bytes exceeds max_dense_jacobian_bytes"
        )
    _require_equal(
        dense_materialized_bytes,
        expected_dense_materialized_bytes,
        f"{context}.dense_materialized_bytes workload contract",
    )
    _require_equal(
        outcome.get("environment_sha256"),
        environment_sha256,
        f"{context}.environment_sha256",
    )
    _sha256(outcome.get("stdout_sha256"), f"{context}.stdout_sha256")
    _sha256(outcome.get("stderr_sha256"), f"{context}.stderr_sha256")
    for field, suffix in (("stdout_path", ".stdout"), ("stderr_path", ".stderr")):
        path_value = _string(outcome.get(field), f"{context}.{field}")
        path = PurePosixPath(path_value)
        if path.is_absolute() or ".." in path.parts or not path_value.endswith(suffix):
            raise BenchmarkContractError(
                f"{context}.{field} must be a safe relative {suffix} path"
            )
    return _OutcomeMetrics(
        elapsed_seconds=elapsed_seconds,
        peak_host_rss_bytes=peak_host_rss_bytes,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        gpu_memory_available=gpu_memory_available,
    )


def _expected_order(workload_index: int, pair_index: int) -> tuple[str, str]:
    return (
        ("fast", "parity")
        if (workload_index + pair_index) % 2 == 0
        else ("parity", "fast")
    )


def _validate_schedule(
    raw_schedule: object, *, workload_index: int, context: str
) -> tuple[tuple[str, str], ...]:
    schedule = _sequence(raw_schedule, f"{context}.schedule")
    if len(schedule) != WARM_PAIR_COUNT:
        raise BenchmarkContractError(
            f"{context}.schedule must contain exactly seven paired entries"
        )
    orders: list[tuple[str, str]] = []
    for pair_index, raw_entry in enumerate(schedule):
        entry = _mapping(raw_entry, f"{context}.schedule[{pair_index}]")
        _exact_keys(
            entry,
            frozenset(("pair_index", "order")),
            f"{context}.schedule[{pair_index}]",
        )
        _require_equal(
            entry.get("pair_index"),
            pair_index,
            f"{context}.schedule[{pair_index}].pair_index",
        )
        order_values = tuple(
            _string(value, f"{context}.schedule[{pair_index}].order")
            for value in _sequence(
                entry.get("order"), f"{context}.schedule[{pair_index}].order"
            )
        )
        expected = _expected_order(workload_index, pair_index)
        if order_values != expected:
            raise BenchmarkContractError(
                f"{context}.schedule[{pair_index}].order must be {list(expected)!r}"
            )
        orders.append(expected)
    return tuple(orders)


def _validate_profile_runs(
    raw_profile: object,
    *,
    device: str,
    intent: str,
    workload_index: int,
    schedule: tuple[tuple[str, str], ...],
    input_sha256: str,
    source_tree_sha256: str,
    max_dense_jacobian_bytes: int,
    expected_dense_materialized_bytes: int,
    context: str,
) -> _ProfileMetrics:
    profile = _mapping(raw_profile, context)
    _exact_keys(
        profile,
        frozenset(("cache", "environment_sha256", "cold", "warmup", "warm")),
        context,
    )
    cache = _mapping(profile.get("cache"), f"{context}.cache")
    _exact_keys(
        cache,
        frozenset(("identity", "cold_initial_state", "cold_entry_count_before")),
        f"{context}.cache",
    )
    cache_identity = _string(cache.get("identity"), f"{context}.cache.identity")
    environment_sha256 = _sha256(
        profile.get("environment_sha256"), f"{context}.environment_sha256"
    )
    _require_equal(
        cache.get("cold_initial_state"), "empty", f"{context}.cache empty state"
    )
    _require_equal(
        cache.get("cold_entry_count_before"), 0, f"{context}.cache empty entry count"
    )

    cold_position = workload_index % 2 if intent == "fast" else 1 - workload_index % 2
    warmup_position = 1 - cold_position
    cold = _validate_outcome(
        profile.get("cold"),
        device=device,
        intent=intent,
        phase="cold",
        pair_index=None,
        order_position=cold_position,
        input_sha256=input_sha256,
        source_tree_sha256=source_tree_sha256,
        cache_identity=cache_identity,
        environment_sha256=environment_sha256,
        max_dense_jacobian_bytes=max_dense_jacobian_bytes,
        expected_dense_materialized_bytes=expected_dense_materialized_bytes,
        context=f"{context}.cold",
    )
    warmup = _validate_outcome(
        profile.get("warmup"),
        device=device,
        intent=intent,
        phase="warmup",
        pair_index=None,
        order_position=warmup_position,
        input_sha256=input_sha256,
        source_tree_sha256=source_tree_sha256,
        cache_identity=cache_identity,
        environment_sha256=environment_sha256,
        max_dense_jacobian_bytes=max_dense_jacobian_bytes,
        expected_dense_materialized_bytes=expected_dense_materialized_bytes,
        context=f"{context}.warmup",
    )
    warm_values = _sequence(profile.get("warm"), f"{context}.warm")
    if len(warm_values) != WARM_PAIR_COUNT:
        raise BenchmarkContractError(
            f"{context}.warm must contain exactly seven measured outcomes"
        )
    warm = tuple(
        _validate_outcome(
            raw_outcome,
            device=device,
            intent=intent,
            phase="warm",
            pair_index=pair_index,
            order_position=schedule[pair_index].index(intent),
            input_sha256=input_sha256,
            source_tree_sha256=source_tree_sha256,
            cache_identity=cache_identity,
            environment_sha256=environment_sha256,
            max_dense_jacobian_bytes=max_dense_jacobian_bytes,
            expected_dense_materialized_bytes=expected_dense_materialized_bytes,
            context=f"{context}.warm[{pair_index}]",
        )
        for pair_index, raw_outcome in enumerate(warm_values)
    )
    return _ProfileMetrics(cold=cold, warmup=warmup, warm=warm)


def deterministic_bootstrap_lower_bound(ratios: Sequence[float]) -> float:
    """Return the deterministic one-sided 95% lower bound of paired medians."""
    checked = tuple(
        _positive_number(value, f"ratios[{index}]")
        for index, value in enumerate(ratios)
    )
    if not checked:
        raise BenchmarkContractError("ratios must not be empty")
    resamples = int(BENCHMARK_RULE["bootstrap_resamples"])
    confidence = float(BENCHMARK_RULE["bootstrap_confidence"])
    generator = random.Random(int(BENCHMARK_RULE["bootstrap_seed"]))
    medians = sorted(
        statistics.median(generator.choices(checked, k=len(checked)))
        for _ in range(resamples)
    )
    lower_index = max(0, math.ceil((1.0 - confidence) * resamples) - 1)
    return float(medians[lower_index])


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator)


def _validate_workload(
    raw_workload: object,
    *,
    device: str,
    source_tree_sha256: str,
    workload_id: str,
    workload_index: int,
) -> tuple[WorkloadSummary, tuple[str, ...]]:
    context = f"workloads[{workload_index}]"
    workload = _mapping(raw_workload, context)
    _require_equal(workload.get("id"), workload_id, f"{context}.id")
    input_sha256 = _sha256(workload.get("input_sha256"), f"{context}.input_sha256")
    _sha256(workload.get("source_sha256"), f"{context}.source_sha256")
    _sha256(workload.get("command_sha256"), f"{context}.command_sha256")
    max_dense_jacobian_bytes = _integer(
        workload.get("max_dense_jacobian_bytes"),
        f"{context}.max_dense_jacobian_bytes",
        minimum=1,
    )
    _require_equal(
        workload.get("dense_materialization_owner"),
        METRIC_OWNERS["dense_materialized_bytes"],
        f"{context}.dense_materialization_owner",
    )
    expected_dense_materialized_bytes = _integer(
        workload.get("dense_materialized_bytes"),
        f"{context}.dense_materialized_bytes",
        minimum=0,
    )
    _require_equal(
        expected_dense_materialized_bytes,
        REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES[workload_id],
        f"{context}.dense_materialized_bytes checked-in value",
    )
    schedule = _validate_schedule(
        workload.get("schedule"), workload_index=workload_index, context=context
    )
    profiles = _mapping(workload.get("profiles"), f"{context}.profiles")
    _exact_keys(profiles, _PROFILE_KEYS, f"{context}.profiles")
    fast = _validate_profile_runs(
        profiles.get("fast"),
        device=device,
        intent="fast",
        workload_index=workload_index,
        schedule=schedule,
        input_sha256=input_sha256,
        source_tree_sha256=source_tree_sha256,
        max_dense_jacobian_bytes=max_dense_jacobian_bytes,
        expected_dense_materialized_bytes=expected_dense_materialized_bytes,
        context=f"{context}.profiles.fast",
    )
    parity = _validate_profile_runs(
        profiles.get("parity"),
        device=device,
        intent="parity",
        workload_index=workload_index,
        schedule=schedule,
        input_sha256=input_sha256,
        source_tree_sha256=source_tree_sha256,
        max_dense_jacobian_bytes=max_dense_jacobian_bytes,
        expected_dense_materialized_bytes=expected_dense_materialized_bytes,
        context=f"{context}.profiles.parity",
    )

    warm_ratios = tuple(
        _ratio(parity_outcome.elapsed_seconds, fast_outcome.elapsed_seconds)
        for fast_outcome, parity_outcome in zip(fast.warm, parity.warm, strict=True)
        if fast_outcome.elapsed_seconds is not None
        and parity_outcome.elapsed_seconds is not None
    )
    warm_median_speedup = float(statistics.median(warm_ratios))
    warm_speedup_lower_bound = deterministic_bootstrap_lower_bound(warm_ratios)
    if fast.cold.elapsed_seconds is None or parity.cold.elapsed_seconds is None:
        raise BenchmarkContractError(f"{context}.cold timings are missing")
    cold_time_ratio = _ratio(fast.cold.elapsed_seconds, parity.cold.elapsed_seconds)
    peak_host_rss_ratio = _ratio(fast.peak_host_rss_bytes, parity.peak_host_rss_bytes)
    fast_gpu_memory = fast.peak_gpu_memory_bytes
    parity_gpu_memory = parity.peak_gpu_memory_bytes
    peak_gpu_memory_ratio = (
        _ratio(fast_gpu_memory, parity_gpu_memory)
        if fast_gpu_memory is not None and parity_gpu_memory is not None
        else None
    )
    summary = WorkloadSummary(
        workload_id=workload_id,
        warm_median_speedup=warm_median_speedup,
        warm_speedup_lower_bound=warm_speedup_lower_bound,
        cold_time_ratio=cold_time_ratio,
        peak_host_rss_ratio=peak_host_rss_ratio,
        peak_gpu_memory_ratio=peak_gpu_memory_ratio,
    )
    reasons: list[str] = []
    if warm_median_speedup < float(BENCHMARK_RULE["warm_median_speedup_min"]):
        reasons.append(
            f"{workload_id}: warm median speedup {warm_median_speedup:.6g} is below "
            f"{BENCHMARK_RULE['warm_median_speedup_min']:.6g}"
        )
    if warm_speedup_lower_bound < float(BENCHMARK_RULE["warm_speedup_lower_bound_min"]):
        reasons.append(
            f"{workload_id}: warm speedup lower bound "
            f"{warm_speedup_lower_bound:.6g} is below "
            f"{BENCHMARK_RULE['warm_speedup_lower_bound_min']:.6g}"
        )
    if cold_time_ratio > float(BENCHMARK_RULE["cold_time_ratio_max"]):
        reasons.append(
            f"{workload_id}: cold end-to-end ratio {cold_time_ratio:.6g} exceeds "
            f"{BENCHMARK_RULE['cold_time_ratio_max']:.6g}"
        )
    if peak_host_rss_ratio > float(BENCHMARK_RULE["host_rss_ratio_max"]):
        reasons.append(
            f"{workload_id}: host RSS ratio {peak_host_rss_ratio:.6g} exceeds "
            f"{BENCHMARK_RULE['host_rss_ratio_max']:.6g}"
        )
    if device == "gpu":
        if peak_gpu_memory_ratio is None:
            reasons.append(f"{workload_id}: GPU memory unavailable for promotion")
        elif peak_gpu_memory_ratio > float(BENCHMARK_RULE["gpu_memory_ratio_max"]):
            reasons.append(
                f"{workload_id}: GPU memory ratio {peak_gpu_memory_ratio:.6g} "
                f"exceeds {BENCHMARK_RULE['gpu_memory_ratio_max']:.6g}"
            )
    return summary, tuple(reasons)


def evaluate_benchmark_artifact(raw_document: object) -> PromotionDecision:
    """Validate retained evidence and evaluate the checked-in promotion rule."""
    document = _mapping(raw_document, "artifact")
    device = _validate_top_level(document)
    source_tree_sha256 = _validate_provenance(document.get("provenance"), device=device)
    raw_workloads = _sequence(document.get("workloads"), "workloads")
    if len(raw_workloads) != len(REPRESENTATIVE_WORKLOAD_IDS):
        raise BenchmarkContractError(
            "workloads must contain the complete checked-in representative inventory"
        )
    summaries: list[WorkloadSummary] = []
    reasons: list[str] = []
    for workload_index, (raw_workload, workload_id) in enumerate(
        zip(raw_workloads, REPRESENTATIVE_WORKLOAD_IDS, strict=True)
    ):
        summary, workload_reasons = _validate_workload(
            raw_workload,
            device=device,
            source_tree_sha256=source_tree_sha256,
            workload_id=workload_id,
            workload_index=workload_index,
        )
        summaries.append(summary)
        reasons.extend(workload_reasons)
    return PromotionDecision(
        device=device,
        promoted=not reasons,
        reasons=tuple(reasons),
        workloads=tuple(summaries),
    )


__all__ = [
    "BENCHMARK_EVIDENCE_KIND",
    "BENCHMARK_RULE",
    "BENCHMARK_RULE_VERSION",
    "BENCHMARK_SCHEMA_VERSION",
    "METRIC_OWNERS",
    "REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES",
    "REPRESENTATIVE_WORKLOAD_IDS",
    "WARM_PAIR_COUNT",
    "BenchmarkContractError",
    "PromotionDecision",
    "WorkloadSummary",
    "deterministic_bootstrap_lower_bound",
    "evaluate_benchmark_artifact",
]
