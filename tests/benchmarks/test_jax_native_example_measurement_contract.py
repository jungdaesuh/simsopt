"""Fail-closed tests for native/JAX example timing and peak-memory evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import pytest

from benchmarks.jax_native_example_measurement_contract import (
    MEASUREMENT_EVIDENCE_KIND,
    MEASUREMENT_PROFILE_IDS,
    MEASUREMENT_SCHEMA_VERSION,
    WARM_SAMPLE_COUNT,
    MeasurementContractError,
    validate_measurement_artifact,
)
from benchmarks.run_jax_native_example_measurements import (
    build_measurement_environment,
    build_measurement_schedule,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_PROFILE_IDS = (
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
)
_ProfileId = Literal[
    "native_cpu",
    "jax_cpu_fast",
    "jax_cpu_parity",
    "jax_gpu_fast",
    "jax_gpu_parity",
]


def _profile_device(profile_id: _ProfileId) -> str:
    return "gpu" if "_gpu_" in profile_id else "cpu"


def _profile_intent(profile_id: _ProfileId) -> str:
    if profile_id == "native_cpu":
        return "native"
    return "parity" if profile_id.endswith("_parity") else "fast"


def _sample(
    profile_id: _ProfileId,
    *,
    phase: str,
    sample_index: int | None,
    order_position: int,
    measured: bool,
) -> dict[str, object]:
    device = _profile_device(profile_id)
    elapsed = 1.0 + 0.01 * order_position
    return {
        "phase": phase,
        "sample_index": sample_index,
        "order_position": order_position,
        "measured": measured,
        "isolated_process": True,
        "returncode": 0,
        "termination": "normal",
        "scientific_success": True,
        "timing_synchronized": True,
        "setup_compile_seconds": elapsed * 0.25 if measured else None,
        "solver_seconds": elapsed * 0.75 if measured else None,
        "total_seconds": elapsed if measured else None,
        "peak_process_tree_rss_bytes": 100_000_000 + order_position,
        "gpu_peak_process_bytes": (
            2_000_000_000 + order_position if device == "gpu" else None
        ),
        "stdout_sha256": _SHA_A,
        "stderr_sha256": _SHA_A,
    }


def _profile(profile_id: _ProfileId, mirror_index: int) -> dict[str, object]:
    schedule = build_measurement_schedule(mirror_index)
    cold_position = schedule.cold.index(profile_id)
    warmup_position = schedule.warmup.index(profile_id)
    warm = [
        _sample(
            profile_id,
            phase="warm",
            sample_index=sample_index,
            order_position=order.index(profile_id),
            measured=True,
        )
        for sample_index, order in enumerate(schedule.warm)
    ]
    device = _profile_device(profile_id)
    return {
        "profile_id": profile_id,
        "device": device,
        "intent": _profile_intent(profile_id),
        "scale": "bounded",
        "input_sha256": _SHA_A,
        "native_source_sha256": _SHA_A,
        "jax_source_sha256": _SHA_A,
        "scientific_comparison_sha256": _SHA_A,
        "scientific_comparison_passed": True,
        "timing_environment": {
            "xla_python_client_preallocate": (
                "not_applicable" if profile_id == "native_cpu" else "true"
            ),
            "persistent_cache_policy": "fresh_isolated",
            "environment_sha256": _SHA_A,
        },
        "timing_samples": {
            "cold": _sample(
                profile_id,
                phase="cold",
                sample_index=None,
                order_position=cold_position,
                measured=True,
            ),
            "warmup": _sample(
                profile_id,
                phase="warmup",
                sample_index=None,
                order_position=warmup_position,
                measured=False,
            ),
            "warm": warm,
        },
        "allocation_memory_sample": (
            {
                "isolated_process": True,
                "xla_python_client_preallocate": "false",
                "returncode": 0,
                "termination": "normal",
                "scientific_success": True,
                "peak_process_tree_rss_bytes": 110_000_000,
                "gpu_peak_process_bytes": 1_500_000_000,
                "gpu_counter_status": "available",
                "monitor_owner": "nvidia_smi_process_poll",
                "monitor_interval_seconds": 0.05,
                "concurrent_use_preflight": "pass",
            }
            if device == "gpu"
            else None
        ),
        "summary": {
            "cold_total_seconds": 1.0 + 0.01 * cold_position,
            "warm_total_seconds_median": sorted(
                float(sample["total_seconds"]) for sample in warm
            )[len(warm) // 2],
            "warm_total_seconds_mad": 0.01,
            "peak_timing_process_tree_rss_bytes": max(
                int(sample["peak_process_tree_rss_bytes"])
                for sample in warm
            ),
            "peak_allocation_gpu_process_bytes": (
                1_500_000_000 if device == "gpu" else None
            ),
        },
    }


def _artifact() -> dict[str, object]:
    mirror_index = 0
    return {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "evidence_kind": MEASUREMENT_EVIDENCE_KIND,
        "certification_eligible": False,
        "claim_policy": {
            "performance_threshold": None,
            "memory_threshold": None,
            "cross_device_speedup_claim": False,
            "rss_vram_ratio_claim": False,
        },
        "mirror_id": "just-a-quadratic",
        "scale": "bounded",
        "profile_ids": list(MEASUREMENT_PROFILE_IDS),
        "identity": {
            "input_sha256": _SHA_A,
            "native_source_sha256": _SHA_A,
            "jax_source_sha256": _SHA_A,
            "scientific_comparison_sha256": _SHA_A,
        },
        "provenance": {
            "repo_commit": "1" * 40,
            "worktree_sha256": _SHA_A,
            "python_version": "3.11.13",
            "simsopt_version": "1.10.3.dev55",
            "simsoptpp_sha256": _SHA_A,
            "jax_version": "0.7.2",
            "jaxlib_version": "0.7.2",
            "xla_version": "jaxlib-0.7.2",
            "os": "Linux",
            "cpu_model": "test-cpu",
            "cpu_count": 32,
            "cpu_affinity": list(range(32)),
            "ram_bytes": 64 * 1024**3,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "gpu_uuid": "GPU-test",
            "driver_version": "580.65.06",
            "cuda_version": "13.0",
            "monitor_interval_seconds": 0.05,
        },
        "schedule": {
            "cold": list(build_measurement_schedule(mirror_index).cold),
            "warmup": list(build_measurement_schedule(mirror_index).warmup),
            "warm": [
                list(order)
                for order in build_measurement_schedule(mirror_index).warm
            ],
        },
        "profiles": {
            profile_id: _profile(profile_id, mirror_index)
            for profile_id in _PROFILE_IDS
        },
    }


def _profiles(artifact: dict[str, object]) -> dict[str, object]:
    profiles = artifact["profiles"]
    assert isinstance(profiles, dict)
    return profiles


def _profile_at(
    artifact: dict[str, object], profile_id: _ProfileId
) -> dict[str, object]:
    profile = _profiles(artifact)[profile_id]
    assert isinstance(profile, dict)
    return profile


def _timing_samples(profile: dict[str, object]) -> dict[str, object]:
    samples = profile["timing_samples"]
    assert isinstance(samples, dict)
    return samples


def test_complete_five_profile_artifact_is_accepted_without_promotion_claim() -> None:
    artifact = _artifact()
    before = deepcopy(artifact)

    audit = validate_measurement_artifact(artifact)

    assert audit.complete is True
    assert audit.mirror_id == "just-a-quadratic"
    assert audit.scale == "bounded"
    assert audit.profile_ids == MEASUREMENT_PROFILE_IDS
    assert artifact == before


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing_profile", "five profiles|profile"),
        ("input_mismatch", "input_sha256"),
        ("source_mismatch", "jax_source_sha256"),
        ("failed_science", "scientific_comparison_passed"),
        ("missing_warm", "seven"),
        ("missing_warmup", "warmup"),
        ("unsynchronized", "timing_synchronized"),
        ("not_isolated", "isolated_process"),
        ("nonempty_cache", "fresh_isolated"),
        ("missing_rss", "peak_process_tree_rss_bytes"),
        ("missing_vram", "gpu_counter_status|GPU"),
        ("timing_preallocation_disabled", "preallocate"),
        ("memory_preallocation_enabled", "preallocate"),
        ("profile_confusion", "intent"),
        ("scale_mismatch", "scale"),
        ("cross_device_speedup", "cross.device"),
        ("rss_vram_ratio", "RSS.*VRAM|rss_vram"),
    ),
)
def test_mutations_fail_closed(mutation: str, match: str) -> None:
    artifact = _artifact()
    cpu_fast = _profile_at(artifact, "jax_cpu_fast")
    gpu_fast = _profile_at(artifact, "jax_gpu_fast")
    if mutation == "missing_profile":
        del _profiles(artifact)["jax_gpu_parity"]
    elif mutation == "input_mismatch":
        cpu_fast["input_sha256"] = _SHA_B
    elif mutation == "source_mismatch":
        cpu_fast["jax_source_sha256"] = _SHA_B
    elif mutation == "failed_science":
        cpu_fast["scientific_comparison_passed"] = False
    elif mutation == "missing_warm":
        warm = _timing_samples(cpu_fast)["warm"]
        assert isinstance(warm, list)
        warm.pop()
    elif mutation == "missing_warmup":
        del _timing_samples(cpu_fast)["warmup"]
    elif mutation == "unsynchronized":
        warm = _timing_samples(cpu_fast)["warm"]
        assert isinstance(warm, list)
        sample = warm[0]
        assert isinstance(sample, dict)
        sample["timing_synchronized"] = False
    elif mutation == "not_isolated":
        cold = _timing_samples(cpu_fast)["cold"]
        assert isinstance(cold, dict)
        cold["isolated_process"] = False
    elif mutation == "nonempty_cache":
        environment = cpu_fast["timing_environment"]
        assert isinstance(environment, dict)
        environment["persistent_cache_policy"] = "reused"
    elif mutation == "missing_rss":
        warm = _timing_samples(cpu_fast)["warm"]
        assert isinstance(warm, list)
        sample = warm[0]
        assert isinstance(sample, dict)
        sample["peak_process_tree_rss_bytes"] = None
    elif mutation == "missing_vram":
        memory = gpu_fast["allocation_memory_sample"]
        assert isinstance(memory, dict)
        memory["gpu_counter_status"] = "unavailable"
        memory["gpu_peak_process_bytes"] = None
    elif mutation == "timing_preallocation_disabled":
        environment = gpu_fast["timing_environment"]
        assert isinstance(environment, dict)
        environment["xla_python_client_preallocate"] = "false"
    elif mutation == "memory_preallocation_enabled":
        memory = gpu_fast["allocation_memory_sample"]
        assert isinstance(memory, dict)
        memory["xla_python_client_preallocate"] = "true"
    elif mutation == "profile_confusion":
        gpu_fast["intent"] = "parity"
    elif mutation == "scale_mismatch":
        gpu_fast["scale"] = "native_default"
    else:
        policy = artifact["claim_policy"]
        assert isinstance(policy, dict)
        if mutation == "cross_device_speedup":
            policy["cross_device_speedup_claim"] = True
        else:
            policy["rss_vram_ratio_claim"] = True

    with pytest.raises(MeasurementContractError, match=match):
        validate_measurement_artifact(artifact)


def test_schedule_is_deterministic_balanced_and_has_exactly_seven_warm_rounds() -> None:
    schedule = build_measurement_schedule(mirror_index=2)

    assert schedule.cold == (
        "jax_cpu_parity",
        "jax_gpu_fast",
        "jax_gpu_parity",
        "native_cpu",
        "jax_cpu_fast",
    )
    assert schedule.warmup == tuple(reversed(schedule.cold))
    assert len(schedule.warm) == WARM_SAMPLE_COUNT == 7
    assert all(set(order) == set(MEASUREMENT_PROFILE_IDS) for order in schedule.warm)
    for first, second in zip(schedule.warm, schedule.warm[1:], strict=False):
        assert second == first[1:] + first[:1]


def test_environments_separate_production_timing_from_gpu_allocation_memory() -> None:
    inherited = {
        "PRESERVED": "yes",
        "SIMSOPT_BACKEND_MODE": "stale",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "stale",
    }

    timing = build_measurement_environment(
        "jax_gpu_fast",
        allocation_sensitive=False,
        base_environment=inherited,
        gpu_index=1,
    )
    allocation = build_measurement_environment(
        "jax_gpu_fast",
        allocation_sensitive=True,
        base_environment=inherited,
        gpu_index=1,
    )
    native = build_measurement_environment(
        "native_cpu",
        allocation_sensitive=False,
        base_environment=inherited,
        gpu_index=1,
    )

    assert timing["PRESERVED"] == "yes"
    assert timing["SIMSOPT_BACKEND_MODE"] == "jax_gpu_fast"
    assert timing["JAX_PLATFORMS"] == "cuda"
    assert timing["CUDA_VISIBLE_DEVICES"] == "1"
    assert timing["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert allocation["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "SIMSOPT_BACKEND_MODE" not in native
    assert "JAX_PLATFORMS" not in native
    assert "CUDA_VISIBLE_DEVICES" not in native
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in native


def test_native_allocation_sensitive_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="GPU"):
        build_measurement_environment(
            "native_cpu",
            allocation_sensitive=True,
            base_environment={},
        )
