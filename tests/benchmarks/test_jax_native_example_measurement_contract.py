"""Fail-closed tests for native/JAX example timing and peak-memory evidence."""

from __future__ import annotations

import os
import statistics
import sys
from copy import deepcopy
from pathlib import Path
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
    MeasurementRunnerError,
    build_collection_plan,
    build_measurement_environment,
    build_measurement_schedule,
    build_profile_command,
    classify_termination,
    execute_monitored_command,
    parse_nvidia_smi_compute_apps,
    publish_artifact_exclusive,
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
    warm_seconds = tuple(float(sample["total_seconds"]) for sample in warm)
    warm_median = float(statistics.median(warm_seconds))
    warm_mad = float(
        statistics.median(abs(value - warm_median) for value in warm_seconds)
    )
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
            "warm_total_seconds_median": warm_median,
            "warm_total_seconds_mad": warm_mad,
            "peak_timing_process_tree_rss_bytes": max(
                int(sample["peak_process_tree_rss_bytes"]) for sample in warm
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
                list(order) for order in build_measurement_schedule(mirror_index).warm
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


def test_total_only_timing_is_accepted_when_no_truthful_inner_boundary_exists() -> None:
    artifact = _artifact()
    for profile_id in _PROFILE_IDS:
        profile = _profile_at(artifact, profile_id)
        timing = _timing_samples(profile)
        for phase in ("cold", "warm"):
            raw_samples = timing[phase]
            samples = raw_samples if isinstance(raw_samples, list) else [raw_samples]
            for sample in samples:
                assert isinstance(sample, dict)
                sample["setup_compile_seconds"] = None
                sample["solver_seconds"] = None

    assert validate_measurement_artifact(artifact).complete is True


def test_partial_or_inconsistent_inner_timing_boundaries_fail_closed() -> None:
    artifact = _artifact()
    warm = _timing_samples(_profile_at(artifact, "jax_cpu_fast"))["warm"]
    assert isinstance(warm, list)
    sample = warm[0]
    assert isinstance(sample, dict)
    sample["setup_compile_seconds"] = None

    with pytest.raises(MeasurementContractError, match="together"):
        validate_measurement_artifact(artifact)


def test_summary_median_absolute_deviation_must_match_raw_samples() -> None:
    artifact = _artifact()
    summary = _profile_at(artifact, "native_cpu")["summary"]
    assert isinstance(summary, dict)
    summary["warm_total_seconds_mad"] = 99.0

    with pytest.raises(MeasurementContractError, match="warm_total_seconds_mad"):
        validate_measurement_artifact(artifact)


@pytest.mark.parametrize(
    ("profile_id", "expected_lane"),
    (
        ("native_cpu", "native-cpu"),
        ("jax_cpu_fast", "jax-cpu"),
        ("jax_cpu_parity", "jax-cpu"),
        ("jax_gpu_fast", "jax-gpu"),
        ("jax_gpu_parity", "jax-gpu"),
    ),
)
def test_profile_command_consumes_the_canonical_bundle_in_parity_child(
    profile_id: _ProfileId, expected_lane: str, tmp_path: Path
) -> None:
    command = build_profile_command(
        python_executable="/python",
        case_id="traceable-least-squares",
        profile_id=profile_id,
        input_bundle_path=tmp_path / "input_bundle.json",
        result_directory=tmp_path / "result",
        scale="bounded",
    )

    assert command == (
        "/python",
        "-m",
        "examples.jax.parity.child",
        "--case",
        "traceable-least-squares",
        "--lane",
        expected_lane,
        "--input-bundle",
        str(tmp_path / "input_bundle.json"),
        "--result-directory",
        str(tmp_path / "result"),
        "--scale",
        "bounded",
    )


@pytest.mark.parametrize(
    ("returncode", "expected"),
    (
        (0, "normal"),
        (2, "exit_2"),
        (-9, "resource_limit_or_oom"),
        (-15, "signal_15"),
    ),
)
def test_measurement_termination_is_explicit(returncode: int, expected: str) -> None:
    assert classify_termination(returncode) == expected


def test_process_attributed_gpu_memory_parser_is_fail_closed() -> None:
    assert parse_nvidia_smi_compute_apps("101, 512\n202, 1024 MiB\n") == {
        101: 512 * 1024**2,
        202: 1024 * 1024**2,
    }
    assert parse_nvidia_smi_compute_apps("No running processes found") == {}

    with pytest.raises(MeasurementRunnerError, match="malformed"):
        parse_nvidia_smi_compute_apps("not,a,valid,row")


def test_measurement_publication_is_exclusive(tmp_path: Path) -> None:
    artifact = _artifact()

    path = publish_artifact_exclusive(
        artifact,
        artifact_root=tmp_path,
        run_directory_name="run-1",
    )

    assert path.name == "artifact.json"
    assert path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        publish_artifact_exclusive(
            artifact,
            artifact_root=tmp_path,
            run_directory_name="run-1",
        )


def test_collection_plan_contains_exact_full_timing_and_memory_protocol() -> None:
    plan = build_collection_plan(mirror_index=1)

    assert len(plan) == 47
    timing = tuple(run for run in plan if not run.allocation_sensitive)
    allocation = tuple(run for run in plan if run.allocation_sensitive)
    assert len(timing) == 45
    assert len(allocation) == 2
    assert tuple(run.profile_id for run in timing[:5]) == (
        "jax_cpu_fast",
        "jax_cpu_parity",
        "jax_gpu_fast",
        "jax_gpu_parity",
        "native_cpu",
    )
    assert all(run.phase == "cold" and run.measured for run in timing[:5])
    assert tuple(run.profile_id for run in timing[5:10]) == tuple(
        reversed(tuple(run.profile_id for run in timing[:5]))
    )
    assert all(run.phase == "warmup" and not run.measured for run in timing[5:10])
    assert len(tuple(run for run in timing if run.phase == "warm")) == 35
    assert tuple(run.profile_id for run in allocation) == (
        "jax_gpu_fast",
        "jax_gpu_parity",
    )
    assert all(run.phase == "allocation_memory" for run in allocation)
    assert all(run.measured for run in allocation)


def test_collection_plan_rotates_every_warm_round() -> None:
    plan = build_collection_plan(mirror_index=0)
    warm = tuple(run for run in plan if run.phase == "warm")

    for sample_index in range(WARM_SAMPLE_COUNT):
        round_runs = tuple(run for run in warm if run.sample_index == sample_index)
        assert len(round_runs) == 5
        assert tuple(run.order_position for run in round_runs) == tuple(range(5))
        assert set(run.profile_id for run in round_runs) == set(MEASUREMENT_PROFILE_IDS)


def test_cpu_child_measurement_uses_parent_process_tree_rss_polling(
    tmp_path: Path,
) -> None:
    result = execute_monitored_command(
        command=(
            sys.executable,
            "-c",
            "payload=bytearray(16 * 1024 * 1024); print(len(payload))",
        ),
        environment={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=tmp_path,
        stdout_path=tmp_path / "child.stdout",
        stderr_path=tmp_path / "child.stderr",
        device="cpu",
        gpu_index=0,
        poll_interval_seconds=0.001,
        timeout_seconds=10.0,
    )

    assert result.returncode == 0
    assert result.termination == "normal"
    assert result.wall_seconds > 0.0
    assert result.peak_process_tree_rss_bytes >= 16 * 1024**2
    assert result.peak_gpu_process_bytes is None
    assert result.gpu_counter_status == "not_applicable"
    assert len(result.stdout_sha256) == 64
    assert len(result.stderr_sha256) == 64
    assert (tmp_path / "child.stdout").read_text(encoding="utf-8").strip() == str(
        16 * 1024**2
    )
