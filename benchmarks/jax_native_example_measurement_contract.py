"""Fail-closed contract for matched native/JAX example measurements.

This artifact is descriptive evidence. It records precision-qualified timing
and peak-memory observations, but it neither certifies scientific parity nor
promotes a backend. Native CPU, JAX CPU, and JAX GPU resources deliberately
remain separate because host RSS and device VRAM are not commensurate.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

MEASUREMENT_SCHEMA_VERSION = 1
MEASUREMENT_EVIDENCE_KIND = "jax_native_example_measurement_noncertifying"
WARM_SAMPLE_COUNT = 7

MeasurementProfileId = Literal[
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
]
MeasurementScale = Literal["bounded", "native_default"]

MEASUREMENT_PROFILE_IDS: tuple[MeasurementProfileId, ...] = (
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
)

PROFILE_CONTRACT: Mapping[
    MeasurementProfileId,
    tuple[Literal["cpu", "gpu"], Literal["native", "fast", "parity"]],
] = MappingProxyType(
    {
        "native_cpu": ("cpu", "native"),
        "jax_cpu_fast": ("cpu", "fast"),
        "jax_cpu_parity": ("cpu", "parity"),
        "jax_gpu_fast": ("gpu", "fast"),
        "jax_gpu_parity": ("gpu", "parity"),
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_LENGTH = 64


class MeasurementContractError(ValueError):
    """A measurement artifact is incomplete, mismatched, or misleading."""


@dataclass(frozen=True)
class ProfileMeasurementSummary:
    """Validated descriptive measurements for one execution profile."""

    profile_id: MeasurementProfileId
    cold_total_seconds: float
    warm_total_seconds_median: float
    warm_total_seconds_mad: float
    peak_timing_process_tree_rss_bytes: int
    peak_allocation_gpu_process_bytes: int | None


@dataclass(frozen=True)
class MeasurementAudit:
    """Immutable result of validating one complete artifact."""

    complete: bool
    mirror_id: str
    scale: MeasurementScale
    profile_ids: tuple[MeasurementProfileId, ...]
    profiles: tuple[ProfileMeasurementSummary, ...]


@dataclass(frozen=True)
class _Sample:
    total_seconds: float | None
    peak_process_tree_rss_bytes: int


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise MeasurementContractError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MeasurementContractError(f"{context} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise MeasurementContractError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise MeasurementContractError(f"{context} must be a non-empty string")
    return value


def _literal(value: object, expected: frozenset[str], context: str) -> str:
    checked = _string(value, context)
    if checked not in expected:
        raise MeasurementContractError(
            f"{context} must be one of {sorted(expected)}, got {checked!r}"
        )
    return checked


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MeasurementContractError(
            f"{context} must be an integer of at least {minimum}"
        )
    return value


def _positive_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementContractError(f"{context} must be a positive number")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise MeasurementContractError(f"{context} must be a positive finite number")
    return checked


def _nonnegative_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementContractError(f"{context} must be nonnegative")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise MeasurementContractError(f"{context} must be finite and nonnegative")
    return checked


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != _SHA256_LENGTH or any(
        character not in _HEX_DIGITS for character in digest
    ):
        raise MeasurementContractError(
            f"{context} must be a lowercase hexadecimal SHA-256 digest"
        )
    return digest


def _require_equal(value: object, expected: object, context: str) -> None:
    if value != expected:
        raise MeasurementContractError(f"{context} must be {expected!r}, got {value!r}")


def _profile_id(value: object, context: str) -> MeasurementProfileId:
    checked = _literal(value, frozenset(MEASUREMENT_PROFILE_IDS), context)
    if checked == "native_cpu":
        return "native_cpu"
    if checked == "jax_cpu_fast":
        return "jax_cpu_fast"
    if checked == "jax_cpu_parity":
        return "jax_cpu_parity"
    if checked == "jax_gpu_fast":
        return "jax_gpu_fast"
    return "jax_gpu_parity"


def _scale(value: object, context: str) -> MeasurementScale:
    checked = _literal(value, frozenset(("bounded", "native_default")), context)
    return "bounded" if checked == "bounded" else "native_default"


def _validate_claim_policy(raw: object) -> None:
    policy = _mapping(raw, "claim_policy")
    _exact_keys(
        policy,
        frozenset(
            (
                "performance_threshold",
                "memory_threshold",
                "cross_device_speedup_claim",
                "rss_vram_ratio_claim",
            )
        ),
        "claim_policy",
    )
    _require_equal(
        policy.get("performance_threshold"),
        None,
        "claim_policy.performance_threshold",
    )
    _require_equal(
        policy.get("memory_threshold"), None, "claim_policy.memory_threshold"
    )
    if policy.get("cross_device_speedup_claim") is not False:
        raise MeasurementContractError(
            "claim_policy.cross_device_speedup_claim must be false"
        )
    if policy.get("rss_vram_ratio_claim") is not False:
        raise MeasurementContractError(
            "claim_policy.rss_vram_ratio_claim must be false; RSS and VRAM "
            "are not commensurate"
        )


def _validate_identity(raw: object) -> dict[str, str]:
    identity = _mapping(raw, "identity")
    expected = frozenset(
        (
            "input_sha256",
            "native_source_sha256",
            "jax_source_sha256",
            "scientific_comparison_sha256",
        )
    )
    _exact_keys(identity, expected, "identity")
    return {name: _sha256(identity.get(name), f"identity.{name}") for name in expected}


def _validate_provenance(raw: object) -> None:
    provenance = _mapping(raw, "provenance")
    expected = frozenset(
        (
            "repo_commit",
            "worktree_sha256",
            "python_version",
            "simsopt_version",
            "simsoptpp_sha256",
            "jax_version",
            "jaxlib_version",
            "xla_version",
            "os",
            "cpu_model",
            "cpu_count",
            "cpu_affinity",
            "ram_bytes",
            "thread_environment",
            "gpu_model",
            "gpu_uuid",
            "driver_version",
            "cuda_version",
            "monitor_interval_seconds",
        )
    )
    _exact_keys(provenance, expected, "provenance")
    commit = _string(provenance.get("repo_commit"), "provenance.repo_commit")
    if len(commit) != 40 or any(character not in _HEX_DIGITS for character in commit):
        raise MeasurementContractError(
            "provenance.repo_commit must be a lowercase 40-character Git hash"
        )
    for name in ("worktree_sha256", "simsoptpp_sha256"):
        _sha256(provenance.get(name), f"provenance.{name}")
    for name in (
        "python_version",
        "simsopt_version",
        "jax_version",
        "jaxlib_version",
        "xla_version",
        "os",
        "cpu_model",
        "gpu_model",
        "gpu_uuid",
        "driver_version",
        "cuda_version",
    ):
        _string(provenance.get(name), f"provenance.{name}")
    cpu_count = _integer(provenance.get("cpu_count"), "provenance.cpu_count", minimum=1)
    affinity = tuple(
        _integer(value, f"provenance.cpu_affinity[{index}]")
        for index, value in enumerate(
            _sequence(provenance.get("cpu_affinity"), "provenance.cpu_affinity")
        )
    )
    if not affinity or len(set(affinity)) != len(affinity):
        raise MeasurementContractError(
            "provenance.cpu_affinity must contain unique CPU indices"
        )
    if any(cpu >= cpu_count for cpu in affinity):
        raise MeasurementContractError(
            "provenance.cpu_affinity contains an index outside cpu_count"
        )
    _integer(provenance.get("ram_bytes"), "provenance.ram_bytes", minimum=1)
    thread_environment = _mapping(
        provenance.get("thread_environment"), "provenance.thread_environment"
    )
    if not thread_environment:
        raise MeasurementContractError(
            "provenance.thread_environment must record thread controls"
        )
    for name, value in thread_environment.items():
        _string(name, "provenance.thread_environment key")
        _string(value, f"provenance.thread_environment.{name}")
    _positive_number(
        provenance.get("monitor_interval_seconds"),
        "provenance.monitor_interval_seconds",
    )


def _validate_schedule(
    raw: object,
) -> tuple[
    tuple[MeasurementProfileId, ...],
    tuple[MeasurementProfileId, ...],
    tuple[tuple[MeasurementProfileId, ...], ...],
]:
    schedule = _mapping(raw, "schedule")
    _exact_keys(schedule, frozenset(("cold", "warmup", "warm")), "schedule")

    def order(value: object, context: str) -> tuple[MeasurementProfileId, ...]:
        checked: tuple[MeasurementProfileId, ...] = tuple(
            _profile_id(item, f"{context}[{index}]")
            for index, item in enumerate(_sequence(value, context))
        )
        if len(checked) != len(MEASUREMENT_PROFILE_IDS) or set(checked) != set(
            MEASUREMENT_PROFILE_IDS
        ):
            raise MeasurementContractError(
                f"{context} must contain all five profiles exactly once"
            )
        return checked

    cold = order(schedule.get("cold"), "schedule.cold")
    warmup = order(schedule.get("warmup"), "schedule.warmup")
    if warmup != tuple(reversed(cold)):
        raise MeasurementContractError("schedule.warmup must reverse schedule.cold")
    raw_warm = _sequence(schedule.get("warm"), "schedule.warm")
    if len(raw_warm) != WARM_SAMPLE_COUNT:
        raise MeasurementContractError(
            "schedule.warm must contain exactly seven balanced rounds"
        )
    warm = tuple(
        order(value, f"schedule.warm[{index}]") for index, value in enumerate(raw_warm)
    )
    for index, (previous, current) in enumerate(
        zip(warm, warm[1:], strict=False), start=1
    ):
        if current != previous[1:] + previous[:1]:
            raise MeasurementContractError(
                f"schedule.warm[{index}] must rotate the preceding order"
            )
    return cold, warmup, warm


def _validate_sample(
    raw: object,
    *,
    profile_id: MeasurementProfileId,
    phase: Literal["cold", "warmup", "warm"],
    sample_index: int | None,
    order_position: int,
    context: str,
) -> _Sample:
    sample = _mapping(raw, context)
    _exact_keys(
        sample,
        frozenset(
            (
                "phase",
                "sample_index",
                "order_position",
                "measured",
                "isolated_process",
                "returncode",
                "termination",
                "scientific_success",
                "timing_synchronized",
                "setup_compile_seconds",
                "solver_seconds",
                "total_seconds",
                "peak_process_tree_rss_bytes",
                "gpu_peak_process_bytes",
                "stdout_sha256",
                "stderr_sha256",
            )
        ),
        context,
    )
    _require_equal(sample.get("phase"), phase, f"{context}.phase")
    _require_equal(sample.get("sample_index"), sample_index, f"{context}.sample_index")
    _require_equal(
        sample.get("order_position"), order_position, f"{context}.order_position"
    )
    measured = phase != "warmup"
    _require_equal(sample.get("measured"), measured, f"{context}.measured")
    _require_equal(sample.get("isolated_process"), True, f"{context}.isolated_process")
    _require_equal(sample.get("returncode"), 0, f"{context}.returncode")
    _require_equal(sample.get("termination"), "normal", f"{context}.termination")
    _require_equal(
        sample.get("scientific_success"), True, f"{context}.scientific_success"
    )
    _require_equal(
        sample.get("timing_synchronized"), True, f"{context}.timing_synchronized"
    )
    if measured:
        setup = _positive_number(
            sample.get("setup_compile_seconds"), f"{context}.setup_compile_seconds"
        )
        solver = _positive_number(
            sample.get("solver_seconds"), f"{context}.solver_seconds"
        )
        total: float | None = _positive_number(
            sample.get("total_seconds"), f"{context}.total_seconds"
        )
        if setup + solver > total * (1.0 + 1e-12):
            raise MeasurementContractError(
                f"{context} setup/compile plus solver time exceeds total time"
            )
    else:
        for name in ("setup_compile_seconds", "solver_seconds", "total_seconds"):
            _require_equal(sample.get(name), None, f"{context}.{name}")
        total = None
    rss = _integer(
        sample.get("peak_process_tree_rss_bytes"),
        f"{context}.peak_process_tree_rss_bytes",
        minimum=1,
    )
    device = PROFILE_CONTRACT[profile_id][0]
    gpu_bytes = sample.get("gpu_peak_process_bytes")
    if device == "gpu":
        _integer(gpu_bytes, f"{context}.gpu_peak_process_bytes", minimum=1)
    else:
        _require_equal(gpu_bytes, None, f"{context}.gpu_peak_process_bytes")
    _sha256(sample.get("stdout_sha256"), f"{context}.stdout_sha256")
    _sha256(sample.get("stderr_sha256"), f"{context}.stderr_sha256")
    return _Sample(total_seconds=total, peak_process_tree_rss_bytes=rss)


def _validate_timing_environment(
    raw: object, *, profile_id: MeasurementProfileId, context: str
) -> None:
    environment = _mapping(raw, context)
    _exact_keys(
        environment,
        frozenset(
            (
                "xla_python_client_preallocate",
                "persistent_cache_policy",
                "environment_sha256",
            )
        ),
        context,
    )
    expected_preallocation = "not_applicable" if profile_id == "native_cpu" else "true"
    _require_equal(
        environment.get("xla_python_client_preallocate"),
        expected_preallocation,
        f"{context}.xla_python_client_preallocate",
    )
    _require_equal(
        environment.get("persistent_cache_policy"),
        "fresh_isolated",
        f"{context}.persistent_cache_policy",
    )
    _sha256(environment.get("environment_sha256"), f"{context}.environment_sha256")


def _validate_allocation_memory(
    raw: object, *, profile_id: MeasurementProfileId, context: str
) -> int | None:
    device = PROFILE_CONTRACT[profile_id][0]
    if device == "cpu":
        _require_equal(raw, None, context)
        return None
    sample = _mapping(raw, context)
    _exact_keys(
        sample,
        frozenset(
            (
                "isolated_process",
                "xla_python_client_preallocate",
                "returncode",
                "termination",
                "scientific_success",
                "peak_process_tree_rss_bytes",
                "gpu_peak_process_bytes",
                "gpu_counter_status",
                "monitor_owner",
                "monitor_interval_seconds",
                "concurrent_use_preflight",
            )
        ),
        context,
    )
    _require_equal(sample.get("isolated_process"), True, f"{context}.isolated_process")
    _require_equal(
        sample.get("xla_python_client_preallocate"),
        "false",
        f"{context}.xla_python_client_preallocate",
    )
    _require_equal(sample.get("returncode"), 0, f"{context}.returncode")
    _require_equal(sample.get("termination"), "normal", f"{context}.termination")
    _require_equal(
        sample.get("scientific_success"), True, f"{context}.scientific_success"
    )
    _integer(
        sample.get("peak_process_tree_rss_bytes"),
        f"{context}.peak_process_tree_rss_bytes",
        minimum=1,
    )
    _require_equal(
        sample.get("gpu_counter_status"), "available", f"{context}.gpu_counter_status"
    )
    gpu_bytes = _integer(
        sample.get("gpu_peak_process_bytes"),
        f"{context}.gpu_peak_process_bytes",
        minimum=1,
    )
    _require_equal(
        sample.get("monitor_owner"),
        "nvidia_smi_process_poll",
        f"{context}.monitor_owner",
    )
    _positive_number(
        sample.get("monitor_interval_seconds"),
        f"{context}.monitor_interval_seconds",
    )
    _require_equal(
        sample.get("concurrent_use_preflight"),
        "pass",
        f"{context}.concurrent_use_preflight",
    )
    return gpu_bytes


def _validate_profile(
    raw: object,
    *,
    profile_id: MeasurementProfileId,
    scale: MeasurementScale,
    identity: Mapping[str, str],
    cold_order: tuple[MeasurementProfileId, ...],
    warmup_order: tuple[MeasurementProfileId, ...],
    warm_orders: tuple[tuple[MeasurementProfileId, ...], ...],
) -> ProfileMeasurementSummary:
    context = f"profiles.{profile_id}"
    profile = _mapping(raw, context)
    _exact_keys(
        profile,
        frozenset(
            (
                "profile_id",
                "device",
                "intent",
                "scale",
                "input_sha256",
                "native_source_sha256",
                "jax_source_sha256",
                "scientific_comparison_sha256",
                "scientific_comparison_passed",
                "timing_environment",
                "timing_samples",
                "allocation_memory_sample",
                "summary",
            )
        ),
        context,
    )
    _require_equal(profile.get("profile_id"), profile_id, f"{context}.profile_id")
    device, intent = PROFILE_CONTRACT[profile_id]
    _require_equal(profile.get("device"), device, f"{context}.device")
    _require_equal(profile.get("intent"), intent, f"{context}.intent")
    _require_equal(profile.get("scale"), scale, f"{context}.scale")
    for name, digest in identity.items():
        _require_equal(profile.get(name), digest, f"{context}.{name}")
    _require_equal(
        profile.get("scientific_comparison_passed"),
        True,
        f"{context}.scientific_comparison_passed",
    )
    _validate_timing_environment(
        profile.get("timing_environment"),
        profile_id=profile_id,
        context=f"{context}.timing_environment",
    )
    timing = _mapping(profile.get("timing_samples"), f"{context}.timing_samples")
    _exact_keys(
        timing,
        frozenset(("cold", "warmup", "warm")),
        f"{context}.timing_samples",
    )
    cold = _validate_sample(
        timing.get("cold"),
        profile_id=profile_id,
        phase="cold",
        sample_index=None,
        order_position=cold_order.index(profile_id),
        context=f"{context}.timing_samples.cold",
    )
    warmup = _validate_sample(
        timing.get("warmup"),
        profile_id=profile_id,
        phase="warmup",
        sample_index=None,
        order_position=warmup_order.index(profile_id),
        context=f"{context}.timing_samples.warmup",
    )
    raw_warm = _sequence(timing.get("warm"), f"{context}.timing_samples.warm")
    if len(raw_warm) != WARM_SAMPLE_COUNT:
        raise MeasurementContractError(
            f"{context}.timing_samples.warm must contain exactly seven samples"
        )
    warm = tuple(
        _validate_sample(
            raw_sample,
            profile_id=profile_id,
            phase="warm",
            sample_index=index,
            order_position=warm_orders[index].index(profile_id),
            context=f"{context}.timing_samples.warm[{index}]",
        )
        for index, raw_sample in enumerate(raw_warm)
    )
    allocation_gpu_bytes = _validate_allocation_memory(
        profile.get("allocation_memory_sample"),
        profile_id=profile_id,
        context=f"{context}.allocation_memory_sample",
    )
    summary = _mapping(profile.get("summary"), f"{context}.summary")
    _exact_keys(
        summary,
        frozenset(
            (
                "cold_total_seconds",
                "warm_total_seconds_median",
                "warm_total_seconds_mad",
                "peak_timing_process_tree_rss_bytes",
                "peak_allocation_gpu_process_bytes",
            )
        ),
        f"{context}.summary",
    )
    if cold.total_seconds is None:
        raise MeasurementContractError(f"{context} cold timing is missing")
    warm_seconds = tuple(
        sample.total_seconds for sample in warm if sample.total_seconds is not None
    )
    median = float(statistics.median(warm_seconds))
    _require_equal(
        summary.get("cold_total_seconds"),
        cold.total_seconds,
        f"{context}.summary.cold_total_seconds",
    )
    _require_equal(
        summary.get("warm_total_seconds_median"),
        median,
        f"{context}.summary.warm_total_seconds_median",
    )
    mad = _nonnegative_number(
        summary.get("warm_total_seconds_mad"),
        f"{context}.summary.warm_total_seconds_mad",
    )
    peak_rss = max(
        sample.peak_process_tree_rss_bytes for sample in (cold, warmup, *warm)
    )
    _require_equal(
        summary.get("peak_timing_process_tree_rss_bytes"),
        peak_rss,
        f"{context}.summary.peak_timing_process_tree_rss_bytes",
    )
    _require_equal(
        summary.get("peak_allocation_gpu_process_bytes"),
        allocation_gpu_bytes,
        f"{context}.summary.peak_allocation_gpu_process_bytes",
    )
    return ProfileMeasurementSummary(
        profile_id=profile_id,
        cold_total_seconds=cold.total_seconds,
        warm_total_seconds_median=median,
        warm_total_seconds_mad=mad,
        peak_timing_process_tree_rss_bytes=peak_rss,
        peak_allocation_gpu_process_bytes=allocation_gpu_bytes,
    )


def validate_measurement_artifact(raw_document: object) -> MeasurementAudit:
    """Validate complete matched measurements without creating promotion claims."""
    document = _mapping(raw_document, "artifact")
    _exact_keys(
        document,
        frozenset(
            (
                "schema_version",
                "evidence_kind",
                "certification_eligible",
                "claim_policy",
                "mirror_id",
                "scale",
                "profile_ids",
                "identity",
                "provenance",
                "schedule",
                "profiles",
            )
        ),
        "artifact",
    )
    _require_equal(
        document.get("schema_version"),
        MEASUREMENT_SCHEMA_VERSION,
        "schema_version",
    )
    _require_equal(
        document.get("evidence_kind"),
        MEASUREMENT_EVIDENCE_KIND,
        "evidence_kind",
    )
    _require_equal(
        document.get("certification_eligible"), False, "certification_eligible"
    )
    _validate_claim_policy(document.get("claim_policy"))
    mirror_id = _string(document.get("mirror_id"), "mirror_id")
    scale = _scale(document.get("scale"), "scale")
    profile_ids: tuple[MeasurementProfileId, ...] = tuple(
        _profile_id(value, f"profile_ids[{index}]")
        for index, value in enumerate(
            _sequence(document.get("profile_ids"), "profile_ids")
        )
    )
    if profile_ids != MEASUREMENT_PROFILE_IDS:
        raise MeasurementContractError(
            "profile_ids must contain the exact five profiles in canonical order"
        )
    identity = _validate_identity(document.get("identity"))
    _validate_provenance(document.get("provenance"))
    cold_order, warmup_order, warm_orders = _validate_schedule(document.get("schedule"))
    profiles = _mapping(document.get("profiles"), "profiles")
    if frozenset(profiles) != frozenset(MEASUREMENT_PROFILE_IDS):
        raise MeasurementContractError("profiles must contain the exact five profiles")
    summaries = tuple(
        _validate_profile(
            profiles.get(profile_id),
            profile_id=profile_id,
            scale=scale,
            identity=identity,
            cold_order=cold_order,
            warmup_order=warmup_order,
            warm_orders=warm_orders,
        )
        for profile_id in MEASUREMENT_PROFILE_IDS
    )
    return MeasurementAudit(
        complete=True,
        mirror_id=mirror_id,
        scale=scale,
        profile_ids=profile_ids,
        profiles=summaries,
    )


__all__ = [
    "MEASUREMENT_EVIDENCE_KIND",
    "MEASUREMENT_PROFILE_IDS",
    "MEASUREMENT_SCHEMA_VERSION",
    "PROFILE_CONTRACT",
    "WARM_SAMPLE_COUNT",
    "MeasurementAudit",
    "MeasurementContractError",
    "MeasurementProfileId",
    "MeasurementScale",
    "ProfileMeasurementSummary",
    "validate_measurement_artifact",
]
