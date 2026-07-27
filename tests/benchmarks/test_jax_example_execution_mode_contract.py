"""Contract tests for matched JAX fast-versus-parity benchmark evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.jax_example_execution_mode_contract import (
    BENCHMARK_EVIDENCE_KIND,
    BENCHMARK_RULE,
    BENCHMARK_RULE_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    METRIC_OWNERS,
    REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES,
    REPRESENTATIVE_WORKLOAD_IDS,
    WARM_PAIR_COUNT,
    BenchmarkContractError,
    deterministic_bootstrap_lower_bound,
    evaluate_benchmark_artifact,
)
from benchmarks.run_jax_example_execution_mode_benchmark import (
    BenchmarkRunnerError,
    build_measurement_schedule,
    build_profile_environment,
    classify_termination,
    evaluate_gpu_concurrent_load_preflight,
    parse_nvidia_smi_compute_apps,
    publish_artifact_exclusive,
)

_SHA = "a" * 64
_OTHER_SHA = "b" * 64


def _gpu_memory(device: str, value: int = 100) -> dict[str, object]:
    if device == "cpu":
        return {
            "status": "not_applicable",
            "peak_process_bytes": None,
            "owner": "not_applicable",
        }
    return {
        "status": "available",
        "peak_process_bytes": value,
        "owner": "nvidia_smi_process_poll",
    }


def _outcome(
    *,
    device: str,
    intent: str,
    phase: str,
    pair_index: int | None,
    order_position: int,
    elapsed_seconds: float | None,
    input_sha256: str = _SHA,
    host_rss_bytes: int = 100,
    gpu_memory_bytes: int = 100,
) -> dict[str, object]:
    return {
        "profile": f"jax_{device}_{intent}",
        "intent": intent,
        "phase": phase,
        "pair_index": pair_index,
        "order_position": order_position,
        "measured": phase != "warmup",
        "input_sha256": input_sha256,
        "source_tree_sha256": _SHA,
        "cache_identity": f"{intent}-cache",
        "returncode": 0,
        "termination": "normal",
        "scientific_success": True,
        "cache_load_compatible": True,
        "runtime_environment_compatible": True,
        "backend_mode": f"jax_{device}_{intent}",
        "platform": device,
        "precision": "fp64",
        "timing_synchronized": True,
        "synchronization_owner": METRIC_OWNERS["synchronization"],
        "elapsed_seconds": elapsed_seconds,
        "diagnostic_wall_seconds": elapsed_seconds if elapsed_seconds else 0.5,
        "peak_host_rss_bytes": host_rss_bytes,
        "gpu_memory": _gpu_memory(device, gpu_memory_bytes),
        "dense_materialized_bytes": 0,
        "environment_sha256": _OTHER_SHA if intent == "fast" else _SHA,
        "stdout_sha256": _SHA,
        "stderr_sha256": _SHA,
        "stdout_path": f"logs/{intent}-{phase}-{pair_index}.stdout",
        "stderr_path": f"logs/{intent}-{phase}-{pair_index}.stderr",
    }


def _profile_runs(
    *,
    device: str,
    intent: str,
    workload_index: int,
) -> dict[str, object]:
    cold_elapsed = 1.10 if intent == "fast" else 1.00
    warm_elapsed = 1.00 if intent == "fast" else 1.20
    cold_order = workload_index % 2 if intent == "fast" else 1 - workload_index % 2
    warmup_order = 1 - cold_order
    warm = []
    for pair_index in range(WARM_PAIR_COUNT):
        fast_first = (workload_index + pair_index) % 2 == 0
        order_position = (
            0
            if (intent == "fast" and fast_first)
            or (intent == "parity" and not fast_first)
            else 1
        )
        warm.append(
            _outcome(
                device=device,
                intent=intent,
                phase="warm",
                pair_index=pair_index,
                order_position=order_position,
                elapsed_seconds=warm_elapsed,
                host_rss_bytes=110 if intent == "fast" else 100,
                gpu_memory_bytes=110 if intent == "fast" else 100,
            )
        )
    return {
        "cache": {
            "identity": f"{intent}-cache",
            "cold_initial_state": "empty",
            "cold_entry_count_before": 0,
        },
        "environment_sha256": _OTHER_SHA if intent == "fast" else _SHA,
        "cold": _outcome(
            device=device,
            intent=intent,
            phase="cold",
            pair_index=None,
            order_position=cold_order,
            elapsed_seconds=cold_elapsed,
            host_rss_bytes=110 if intent == "fast" else 100,
            gpu_memory_bytes=110 if intent == "fast" else 100,
        ),
        "warmup": _outcome(
            device=device,
            intent=intent,
            phase="warmup",
            pair_index=None,
            order_position=warmup_order,
            elapsed_seconds=None,
            host_rss_bytes=110 if intent == "fast" else 100,
            gpu_memory_bytes=110 if intent == "fast" else 100,
        ),
        "warm": warm,
    }


def _workload(device: str, workload_id: str, workload_index: int) -> dict[str, object]:
    schedule = []
    for pair_index in range(WARM_PAIR_COUNT):
        fast_first = (workload_index + pair_index) % 2 == 0
        schedule.append(
            {
                "pair_index": pair_index,
                "order": ["fast", "parity"] if fast_first else ["parity", "fast"],
            }
        )
    return {
        "id": workload_id,
        "input_sha256": _SHA,
        "source_sha256": _SHA,
        "command_sha256": _SHA,
        "max_dense_jacobian_bytes": 4096,
        "dense_materialization_owner": METRIC_OWNERS["dense_materialized_bytes"],
        "dense_materialized_bytes": REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES[
            workload_id
        ],
        "schedule": schedule,
        "profiles": {
            "fast": _profile_runs(
                device=device,
                intent="fast",
                workload_index=workload_index,
            ),
            "parity": _profile_runs(
                device=device,
                intent="parity",
                workload_index=workload_index,
            ),
        },
    }


def _artifact(device: str = "cpu") -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "rule_version": BENCHMARK_RULE_VERSION,
        "rule": dict(BENCHMARK_RULE),
        "evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "certification_eligible": False,
        "metric_owners": dict(METRIC_OWNERS),
        "device": device,
        "profiles": {
            "fast": f"jax_{device}_fast",
            "parity": f"jax_{device}_parity",
        },
        "workload_ids": list(REPRESENTATIVE_WORKLOAD_IDS),
        "provenance": {
            "run_id": "test-run",
            "started_at_utc": "2026-07-27T00:00:00Z",
            "repo_commit": "1" * 40,
            "source_tree_sha256": _SHA,
            "manifest_sha256": _SHA,
            "python_version": "3.11.13",
            "jax_version": "0.7.2",
            "jaxlib_version": "0.7.2",
            "xla_version": "jaxlib-0.7.2",
            "host": {
                "hostname": "benchmark-host",
                "cpu_model": "test-cpu",
                "os": "Linux",
            },
            "device": {
                "kind": device,
                "model": "RTX 5090" if device == "gpu" else "test-cpu",
                "uuid": "GPU-test" if device == "gpu" else None,
                "driver": "test-driver" if device == "gpu" else "not_applicable",
                "clock_policy": "unavailable",
                "power_policy": "unavailable",
            },
            "concurrent_load_preflight": {
                "status": "pass",
                "detail": "isolated test fixture",
            },
        },
        "workloads": [
            _workload(device, workload_id, workload_index)
            for workload_index, workload_id in enumerate(REPRESENTATIVE_WORKLOAD_IDS)
        ],
    }


def _workload_at(artifact: dict[str, object], index: int = 0) -> dict[str, object]:
    workloads = artifact["workloads"]
    assert isinstance(workloads, list)
    workload = workloads[index]
    assert isinstance(workload, dict)
    return workload


def _profile_at(
    artifact: dict[str, object], intent: str, workload_index: int = 0
) -> dict[str, object]:
    profiles = _workload_at(artifact, workload_index)["profiles"]
    assert isinstance(profiles, dict)
    profile = profiles[intent]
    assert isinstance(profile, dict)
    return profile


def _warm_at(
    artifact: dict[str, object], intent: str, pair_index: int = 0
) -> dict[str, object]:
    warm = _profile_at(artifact, intent)["warm"]
    assert isinstance(warm, list)
    outcome = warm[pair_index]
    assert isinstance(outcome, dict)
    return outcome


def test_checked_in_rule_is_versioned_and_exact() -> None:
    assert BENCHMARK_SCHEMA_VERSION == 3
    assert BENCHMARK_RULE_VERSION == 2
    assert WARM_PAIR_COUNT == 7
    assert REPRESENTATIVE_DENSE_MATERIALIZATION_BYTES == {
        workload_id: 0 for workload_id in REPRESENTATIVE_WORKLOAD_IDS
    }
    assert BENCHMARK_RULE == {
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
        "warm_pair_count": 7,
    }


@pytest.mark.parametrize("device", ("cpu", "gpu"))
def test_complete_matched_artifact_promotes(device: str) -> None:
    decision = evaluate_benchmark_artifact(_artifact(device))

    assert decision.promoted is True
    assert decision.reasons == ()
    assert tuple(summary.workload_id for summary in decision.workloads) == (
        REPRESENTATIVE_WORKLOAD_IDS
    )
    assert all(
        summary.warm_median_speedup == pytest.approx(1.2)
        for summary in decision.workloads
    )
    assert all(
        summary.warm_speedup_lower_bound == pytest.approx(1.2)
        for summary in decision.workloads
    )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("missing_profile", "profiles"),
        ("wrong_profile", "jax_cpu_fast"),
        ("workload_inventory", "workload_ids"),
        ("workload_order", "workloads"),
    ),
)
def test_rejects_missing_or_wrong_profiles_and_workloads(
    mutation: str, match: str
) -> None:
    artifact = _artifact()
    if mutation == "missing_profile":
        profiles = artifact["profiles"]
        assert isinstance(profiles, dict)
        del profiles["fast"]
    elif mutation == "wrong_profile":
        profiles = artifact["profiles"]
        assert isinstance(profiles, dict)
        profiles["fast"] = "jax_cpu_parity"
    elif mutation == "workload_inventory":
        artifact["workload_ids"] = list(REPRESENTATIVE_WORKLOAD_IDS[:-1])
    else:
        workloads = artifact["workloads"]
        assert isinstance(workloads, list)
        workloads.reverse()

    with pytest.raises(BenchmarkContractError, match=match):
        evaluate_benchmark_artifact(artifact)


def test_rejects_unmatched_inputs_and_sources() -> None:
    artifact = _artifact()
    _warm_at(artifact, "fast")["input_sha256"] = _OTHER_SHA

    with pytest.raises(BenchmarkContractError, match="input_sha256"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    _warm_at(artifact, "parity")["source_tree_sha256"] = _OTHER_SHA
    with pytest.raises(BenchmarkContractError, match="source_tree_sha256"):
        evaluate_benchmark_artifact(artifact)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("scientific_success", False, "scientific_success"),
        ("returncode", 1, "returncode"),
        ("termination", "oom", "termination"),
        ("backend_mode", "jax_cpu_parity", "backend_mode"),
        ("precision", "fp32", "precision"),
        ("platform", "gpu", "platform"),
    ),
)
def test_rejects_failed_or_wrong_profile_repetitions(
    field: str, value: object, match: str
) -> None:
    artifact = _artifact()
    _warm_at(artifact, "fast")[field] = value

    with pytest.raises(BenchmarkContractError, match=match):
        evaluate_benchmark_artifact(artifact)


def test_rejects_incompatible_persistent_cache_load() -> None:
    artifact = _artifact()
    _warm_at(artifact, "fast")["cache_load_compatible"] = False

    with pytest.raises(BenchmarkContractError, match="cache_load_compatible"):
        evaluate_benchmark_artifact(artifact)


def test_rejects_incompatible_runtime_environment() -> None:
    artifact = _artifact("gpu")
    _warm_at(artifact, "fast")["runtime_environment_compatible"] = False

    with pytest.raises(BenchmarkContractError, match="runtime_environment_compatible"):
        evaluate_benchmark_artifact(artifact)


def test_rejects_incomplete_cold_warmup_and_balanced_pair_protocol() -> None:
    artifact = _artifact()
    _profile_at(artifact, "fast")["cold"] = None
    with pytest.raises(BenchmarkContractError, match="cold"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    warmup = _profile_at(artifact, "parity")["warmup"]
    assert isinstance(warmup, dict)
    warmup["measured"] = True
    with pytest.raises(BenchmarkContractError, match="warmup.*measured"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    warm = _profile_at(artifact, "fast")["warm"]
    assert isinstance(warm, list)
    warm.pop()
    with pytest.raises(BenchmarkContractError, match="seven|7"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    schedule = _workload_at(artifact)["schedule"]
    assert isinstance(schedule, list)
    schedule[1] = {"pair_index": 1, "order": ["fast", "parity"]}
    with pytest.raises(BenchmarkContractError, match="schedule"):
        evaluate_benchmark_artifact(artifact)


def test_rejects_nonempty_cold_cache_or_cache_identity_drift() -> None:
    artifact = _artifact()
    cache = _profile_at(artifact, "fast")["cache"]
    assert isinstance(cache, dict)
    cache["cold_entry_count_before"] = 1
    with pytest.raises(BenchmarkContractError, match="empty"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    _warm_at(artifact, "fast")["cache_identity"] = "different-cache"
    with pytest.raises(BenchmarkContractError, match="cache_identity"):
        evaluate_benchmark_artifact(artifact)


def test_rejects_unsynchronized_or_nonpositive_timings_and_missing_host_rss() -> None:
    artifact = _artifact()
    _warm_at(artifact, "fast")["timing_synchronized"] = False
    with pytest.raises(BenchmarkContractError, match="timing_synchronized"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    _warm_at(artifact, "fast")["elapsed_seconds"] = 0.0
    with pytest.raises(BenchmarkContractError, match="elapsed_seconds"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact()
    _warm_at(artifact, "fast").pop("peak_host_rss_bytes")
    with pytest.raises(BenchmarkContractError, match="peak_host_rss_bytes"):
        evaluate_benchmark_artifact(artifact)


def test_gpu_unavailable_process_memory_is_diagnostic_but_cannot_promote() -> None:
    artifact = _artifact("gpu")
    gpu_memory = _warm_at(artifact, "fast")["gpu_memory"]
    assert isinstance(gpu_memory, dict)
    gpu_memory.update(
        {
            "status": "unavailable",
            "peak_process_bytes": None,
            "owner": "nvidia_smi_unavailable",
        }
    )

    decision = evaluate_benchmark_artifact(artifact)

    assert decision.promoted is False
    assert any("GPU memory unavailable" in reason for reason in decision.reasons)


def test_rejects_missing_or_malformed_gpu_memory_evidence() -> None:
    artifact = _artifact("gpu")
    _warm_at(artifact, "fast").pop("gpu_memory")
    with pytest.raises(BenchmarkContractError, match="gpu_memory"):
        evaluate_benchmark_artifact(artifact)

    artifact = _artifact("gpu")
    gpu_memory = _warm_at(artifact, "fast")["gpu_memory"]
    assert isinstance(gpu_memory, dict)
    gpu_memory["peak_process_bytes"] = 0
    with pytest.raises(BenchmarkContractError, match="peak_process_bytes"):
        evaluate_benchmark_artifact(artifact)


def test_rejects_dense_materialization_over_resolved_budget() -> None:
    artifact = _artifact()
    _warm_at(artifact, "fast")["dense_materialized_bytes"] = 4097

    with pytest.raises(BenchmarkContractError, match="max_dense_jacobian_bytes"):
        evaluate_benchmark_artifact(artifact)


def test_rejects_threshold_edit_without_rule_or_schema_change() -> None:
    artifact = _artifact()
    rule = artifact["rule"]
    assert isinstance(rule, dict)
    rule["warm_median_speedup_min"] = 1.0

    with pytest.raises(BenchmarkContractError, match="rule"):
        evaluate_benchmark_artifact(artifact)


@pytest.mark.parametrize(
    "field,value",
    (
        ("certification_eligible", True),
        ("evidence_kind", "jax_example_parity_certification"),
    ),
)
def test_rejects_fast_receipts_presented_as_parity_evidence(
    field: str, value: object
) -> None:
    artifact = _artifact()
    artifact[field] = value

    with pytest.raises(BenchmarkContractError, match="non-certifying|evidence_kind"):
        evaluate_benchmark_artifact(artifact)


def test_device_promotion_fails_closed_on_each_checked_in_threshold() -> None:
    mutations = (
        ("warm_speed", "warm median speedup"),
        ("cold", "cold end-to-end"),
        ("host_rss", "host RSS"),
        ("gpu_memory", "GPU memory"),
    )
    for mutation, expected_reason in mutations:
        device = "gpu" if mutation == "gpu_memory" else "cpu"
        artifact = _artifact(device)
        if mutation == "warm_speed":
            warm = _profile_at(artifact, "fast")["warm"]
            assert isinstance(warm, list)
            for outcome in warm:
                assert isinstance(outcome, dict)
                outcome["elapsed_seconds"] = 1.5
                outcome["diagnostic_wall_seconds"] = 1.5
        elif mutation == "cold":
            cold = _profile_at(artifact, "fast")["cold"]
            assert isinstance(cold, dict)
            cold["elapsed_seconds"] = 1.26
            cold["diagnostic_wall_seconds"] = 1.26
        elif mutation == "host_rss":
            _warm_at(artifact, "fast")["peak_host_rss_bytes"] = 126
        else:
            gpu_memory = _warm_at(artifact, "fast")["gpu_memory"]
            assert isinstance(gpu_memory, dict)
            gpu_memory["peak_process_bytes"] = 126

        decision = evaluate_benchmark_artifact(artifact)

        assert decision.promoted is False
        assert any(expected_reason in reason for reason in decision.reasons)


def test_bootstrap_lower_bound_is_deterministic_and_one_sided() -> None:
    ratios = (1.20, 1.10, 1.30, 1.15, 1.25, 1.18, 1.22)

    first = deterministic_bootstrap_lower_bound(ratios)
    second = deterministic_bootstrap_lower_bound(ratios)

    assert first == second
    assert min(ratios) <= first <= 1.20


def test_artifact_evaluation_does_not_mutate_retained_evidence() -> None:
    artifact = _artifact("gpu")
    before = deepcopy(artifact)

    evaluate_benchmark_artifact(artifact)

    assert artifact == before


def test_measurement_schedule_has_cold_warmup_and_seven_alternating_pairs() -> None:
    schedule = build_measurement_schedule(workload_index=0)

    assert len(schedule) == 18
    assert [(run.phase, run.intent, run.measured) for run in schedule[:4]] == [
        ("cold", "fast", True),
        ("cold", "parity", True),
        ("warmup", "parity", False),
        ("warmup", "fast", False),
    ]
    warm = schedule[4:]
    assert len(warm) == 2 * WARM_PAIR_COUNT
    for pair_index in range(WARM_PAIR_COUNT):
        pair = warm[2 * pair_index : 2 * pair_index + 2]
        expected = ("fast", "parity") if pair_index % 2 == 0 else ("parity", "fast")
        assert tuple(run.intent for run in pair) == expected
        assert tuple(run.order_position for run in pair) == (0, 1)
        assert all(run.pair_index == pair_index for run in pair)


def test_profile_environment_is_pinned_and_cache_specific(tmp_path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    inherited = {
        "PRESERVED": "yes",
        "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
        "JAX_COMPILATION_CACHE_DIR": "/stale/cache",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "999",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "999",
        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
    }

    profile, environment, digest = build_profile_environment(
        "cpu",
        "fast",
        cache_directory=cache_directory,
        base_environment=inherited,
        repo_root=tmp_path,
    )

    assert profile.mode == "jax_cpu_fast"
    assert environment["PRESERVED"] == "yes"
    assert environment["SIMSOPT_BACKEND_MODE"] == "jax_cpu_fast"
    assert environment["JAX_COMPILATION_CACHE_DIR"] == str(cache_directory)
    assert environment["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "0"
    assert environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
    assert "XLA_FLAGS" not in environment
    assert len(digest) == 64


def test_gpu_profile_environment_retains_only_selected_determinism_overlay(
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    _, fast, _ = build_profile_environment(
        "gpu",
        "fast",
        cache_directory=cache_directory,
        base_environment={"XLA_FLAGS": "--stale=true"},
        repo_root=tmp_path,
    )
    _, parity, _ = build_profile_environment(
        "gpu",
        "parity",
        cache_directory=cache_directory,
        base_environment={"XLA_FLAGS": "--stale=true"},
        repo_root=tmp_path,
    )

    assert "XLA_FLAGS" not in fast
    assert parity["XLA_FLAGS"] == "--xla_gpu_exclude_nondeterministic_ops=true"


def test_nvidia_smi_compute_process_parser_is_fail_closed() -> None:
    assert parse_nvidia_smi_compute_apps("123, 512\n456, 1024 MiB\n") == {
        123: 512 * 1024 * 1024,
        456: 1024 * 1024 * 1024,
    }
    assert parse_nvidia_smi_compute_apps("No running processes found") == {}

    with pytest.raises(BenchmarkRunnerError, match="nvidia-smi"):
        parse_nvidia_smi_compute_apps("malformed row")


def test_gpu_preflight_accepts_bounded_desktop_load_and_records_it() -> None:
    preflight = evaluate_gpu_concurrent_load_preflight(
        processes={101: 512 * 1024 * 1024, 202: 256 * 1024 * 1024},
        utilization_samples=((3, 1), (2, 1), (4, 2), (1, 1), (2, 1)),
        total_memory_bytes=32 * 1024 * 1024 * 1024,
    )

    assert preflight["status"] == "pass"
    assert "background_processes=101:536870912,202:268435456" in preflight["detail"]
    assert "max_gpu_utilization_percent=4" in preflight["detail"]


@pytest.mark.parametrize(
    ("processes", "samples", "match"),
    (
        ({}, ((6, 1),) * 5, "utilization"),
        ({101: 2 * 1024**3}, ((1, 1),) * 5, "memory"),
    ),
)
def test_gpu_preflight_rejects_material_concurrent_load(
    processes: dict[int, int],
    samples: tuple[tuple[int, int], ...],
    match: str,
) -> None:
    with pytest.raises(BenchmarkRunnerError, match=match):
        evaluate_gpu_concurrent_load_preflight(
            processes=processes,
            utilization_samples=samples,
            total_memory_bytes=32 * 1024 * 1024 * 1024,
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
def test_termination_classification_is_explicit(returncode: int, expected: str) -> None:
    assert classify_termination(returncode) == expected


def test_artifact_publication_is_exclusive_and_durable(tmp_path) -> None:
    document = {"run_id": "test", "status": "complete"}

    artifact_path = publish_artifact_exclusive(
        document,
        artifact_root=tmp_path,
        run_directory_name="run-test",
    )

    assert artifact_path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        publish_artifact_exclusive(
            document,
            artifact_root=tmp_path,
            run_directory_name="run-test",
        )


def test_gitignore_is_narrow_for_execution_mode_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lines = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert lines.count(".artifacts/jax-example-execution-modes/") == 1
    assert lines.count(".artifacts/jax-example-parity/") == 1
    assert ".artifacts/" not in lines
