from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

import benchmarks.custom_quasi_newton_receipts as receipt_module
import numpy as np
import pytest
from benchmarks.boozer_trial_diagnostic import (
    JoinedBoozerTrialRecord,
    LineSearchTrialEvidence,
    ObjectiveTrialEvidence,
    TrialKey,
    parameter_sha256,
    write_boozer_trial_trace,
)
from benchmarks.custom_quasi_newton_receipts import publish, validate_all


def _git_commit(root: Path) -> str:
    subprocess.run(
        ["git", "init", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "receipt-tests@example.com"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Receipt Tests"],
        check=True,
        capture_output=True,
        text=True,
    )
    marker = root / "receipt-source.txt"
    marker.write_text("synthetic receipt source\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", marker.name],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", "synthetic receipt source"],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    assert len(commit) == 40
    return commit


def _runner_directory(root: Path, *, clean: bool = True) -> Path:
    commit = _git_commit(root)
    run = root / "runner-case"
    run.mkdir()
    (run / "raw").mkdir()
    (run / "raw" / "stdout.json").write_text("{}\n", encoding="utf-8")
    (run / "measurements.json").write_text(
        json.dumps(
            {
                "git_commit": commit,
                "git_clean": clean,
                "measurements": [
                    {
                        "case": "coil47",
                        "provider": "custom",
                        "iterations": 2,
                        "status": 0,
                        "success": True,
                        "final_objective": 1.25e-12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def _runner_v7_directory(root: Path) -> Path:
    commit = _git_commit(root)
    run = root / "runner-v7"
    run.mkdir()
    device_identity = {
        "requested_device": "gpu",
        "backend": "gpu",
        "platform": "gpu",
        "jax_device": "cuda:0",
        "device_kind": "NVIDIA GeForce RTX 5090",
        "device_id": 0,
        "process_index": 0,
        "gpu_uuid": "GPU-test",
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "total_memory_bytes": 32 * 1024**3,
        "driver_version": "590.48",
        "cuda_version": "CUDA 13.0",
        "visible_devices": "GPU-test",
        "hostname": "test-host",
        "scheduler_job_id": None,
    }
    (run / "measurements.json").write_text(
        json.dumps(
            {
                "schema_version": 9,
                "git_commit": commit,
                "git_clean": True,
                "orchestrator_git_clean": True,
                "provider_child": True,
                "provider_children": [],
                "requested_device": "gpu",
                "method": "lbfgs",
                "python_version": "3.11.13",
                "jax_version": "0.10.0",
                "jaxlib_version": "0.10.0",
                "numpy_version": "2.3.1",
                "scipy_version": "1.16.0",
                "optax_version": "0.2.8",
                "runtime_environment": {
                    "JAX_PLATFORMS": "cuda",
                    "JAX_ENABLE_X64": "true",
                    "SIMSOPT_BACKEND_MODE": "jax_gpu_fast",
                    "SIMSOPT_BACKEND_STRICT": "1",
                    "SIMSOPT_PRECISION": "fp64",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": None,
                },
                "device_identity": device_identity,
                "measurements": [
                    {
                        "case": "coil47",
                        "provider": "custom",
                        "method": "lbfgs",
                        "device": "gpu",
                        "intent": "fast",
                        "solver_route": "fused_stepwise",
                        "device_identity": device_identity,
                        "maxiter": 20,
                        "iterations": 2,
                        "evaluations": 4,
                        "status": 0,
                        "success": True,
                        "stopping_reason": "converged",
                        "initial_objective": 2.0,
                        "initial_gradient_inf_norm": 1.0,
                        "final_objective": 1.25e-12,
                        "final_gradient_inf_norm": 1.0e-9,
                        "initial_parameters": [1.0],
                        "final_parameters": [0.0],
                        "fixture_build_seconds": 0.5,
                        "fixture_build_peak_rss_kib": 90,
                        "preparation_seconds": 1.0,
                        "first_execution_seconds": 2.0,
                        "cold_seconds": 3.0,
                        "warm_seconds": 0.1,
                        "solver_start_rss_kib": 100,
                        "solver_peak_rss_kib": 120,
                        "solver_peak_rss_delta_kib": 20,
                        "peak_rss_kib": 120,
                        "peak_rss_scope": "self_proc_status_phase_max",
                        "ru_maxrss_kib": 118,
                        "process_pid": 1234,
                        "peak_vram_mib": 200,
                        "inner_success": True,
                        "parameters_finite": True,
                        "observables_finite": True,
                        "constraint_norm": None,
                        "endpoint_certificate": {
                            "success": True,
                            "stopping_reason": "converged",
                            "initial_stationary": False,
                            "terminal_stationary": True,
                            "constraints_satisfied": True,
                        },
                        "scientific_observables": {},
                        "scientific_certification_seconds": 0.01,
                        "work_counters": {
                            "accepted_iterations": 2,
                            "objective_evaluations": 4,
                            "transfer_calls": 3,
                            "transfer_leaves": 6,
                            "transfer_bytes": 48,
                            "advance_observations": 0,
                        },
                        "diagnostic_artifacts": {
                            "memory_trace": "memory.json",
                            "trial_trace": None,
                        },
                        "phase_rss": [
                            {
                                "phase": "preparation",
                                "start_rss_kib": 100,
                                "peak_rss_kib": 120,
                                "end_rss_kib": 115,
                                "sample_count": 3,
                                "scope": "self_proc_status_poll_10ms",
                            },
                            {
                                "phase": "cold_solver",
                                "start_rss_kib": 115,
                                "peak_rss_kib": 118,
                                "end_rss_kib": 112,
                                "sample_count": 3,
                                "scope": "self_proc_status_poll_10ms",
                            },
                            {
                                "phase": "warm_solver",
                                "start_rss_kib": 112,
                                "peak_rss_kib": 116,
                                "end_rss_kib": 110,
                                "sample_count": 3,
                                "scope": "self_proc_status_poll_10ms",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "memory.json").write_text(
        json.dumps(
            {
                "availability": "available",
                "gpu_uuid": "GPU-test",
                "peak_used_memory_mib": 200,
                "provider_pid": 1234,
                "samples": [
                    {"sampled_at_unix_ns": 1, "used_memory_mib": 200}
                ],
                "schema_version": 1,
                "target_pid_observed": True,
                "unavailable_reason": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def _write_valid_trial_trace(
    lane: Path,
    row: dict[str, object],
    *,
    production_row: dict[str, object] | None = None,
) -> None:
    initial_parameters = np.asarray(row["initial_parameters"], dtype=np.float64)
    final_parameters = np.asarray(row["final_parameters"], dtype=np.float64)
    production = row if production_row is None else production_row
    production_final_parameters = np.asarray(
        production["final_parameters"], dtype=np.float64
    )
    initial_hash = parameter_sha256(initial_parameters)
    final_hash = parameter_sha256(final_parameters)
    evaluations = cast(int, row["evaluations"])

    def objective(*, initial: bool) -> ObjectiveTrialEvidence:
        parameter_hash = initial_hash if initial else final_hash
        objective_field = "initial_objective" if initial else "final_objective"
        gradient_field = (
            "initial_gradient_inf_norm" if initial else "final_gradient_inf_norm"
        )
        return ObjectiveTrialEvidence(
            raw_objective=cast(float, row[objective_field]),
            raw_objective_certified=True,
            filtered_objective=cast(float, row[objective_field]),
            gradient_inf_norm=cast(float, row[gradient_field]),
            gradient_finite=True,
            gradient_source="candidate",
            gradient_source_parameter_sha256=parameter_hash,
            predictor_kind=None,
            predictor_success=None,
            primal_success=True,
            adjoint_success=None,
            newton_success=True,
            newton_stop_reason_code=0,
            newton_accepted_iterations=1,
            newton_attempted_iterations=1,
            newton_last_linear_solve_success=True,
            inner_penalty_residual_l2=1.0e-12,
            inner_final_gradient_inf_norm=1.0e-12,
        )

    records: list[JoinedBoozerTrialRecord] = []
    for index in range(evaluations + 1):
        phase = (
            "initial"
            if index == 0
            else "final_refresh"
            if index == evaluations
            else "line_search"
        )
        initial = phase == "initial"
        parameter_hash = initial_hash if initial else final_hash
        parameters = initial_parameters if initial else final_parameters
        records.append(
            JoinedBoozerTrialRecord(
                key=TrialKey(index, parameter_hash),
                phase=phase,
                objective=objective(initial=initial),
                line_search=(
                    LineSearchTrialEvidence(None, None, None, None, None)
                    if phase != "line_search"
                    else LineSearchTrialEvidence(index, 0.5, -0.25, -0.1, -0.2)
                ),
                parameter_archive_key=parameter_hash,
                parameter_shape=(parameters.size,),
            )
        )
    write_boozer_trial_trace(
        lane / "trial.json",
        provider=cast(Literal["native", "custom"], row["provider"]),
        production_route=cast(str, row["solver_route"]),
        maxiter=cast(int, row["maxiter"]),
        maxls=20,
        records=tuple(records),
        parameters_by_sha256={
            initial_hash: initial_parameters,
            final_hash: final_parameters,
        },
        production_evaluations=cast(int, production["evaluations"]),
        production_final_objective=cast(float, production["final_objective"]),
        production_final_gradient_inf_norm=cast(
            float, production["final_gradient_inf_norm"]
        ),
        production_final_status=cast(int, production["status"]),
        production_final_parameters_sha256=parameter_sha256(
            production_final_parameters
        ),
        final_status=cast(int, row["status"]),
    )


def _json_object(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _first_measurement(payload: dict[str, object]) -> dict[str, object]:
    measurements = cast(list[dict[str, object]], payload["measurements"])
    return measurements[0]


def _scientific_v7_lanes(tmp_path: Path) -> list[Path]:
    run = _runner_v7_directory(tmp_path)
    payload = _json_object(run / "measurements.json")
    rtx = dict(_first_measurement(payload))
    rtx.update(
        {
            "case": "boozer",
            "method": "bfgs",
            "intent": "parity",
            "maxiter": 1000,
            "solver_route": "custom_bfgs_stepwise",
            "scientific_observables": {
                "final_boozer_residual": 1.0e-12,
                "final_non_qs": 1.0e-6,
                "final_iota": -0.2,
                "final_volume": 0.25,
            },
            "diagnostic_artifacts": {
                "memory_trace": "memory.json",
                "trial_trace": "trial.json",
            },
        }
    )
    cpu_identity = dict(cast(dict[str, object], rtx["device_identity"]))
    cpu_identity.update(
        {
            "requested_device": "cpu",
            "backend": "cpu",
            "platform": "cpu",
            "jax_device": "cpu:0",
            "device_kind": "cpu",
            "gpu_uuid": None,
            "gpu_model": None,
            "compute_capability": None,
            "total_memory_bytes": None,
            "driver_version": None,
            "cuda_version": None,
            "visible_devices": None,
        }
    )
    native = dict(rtx)
    native.update(
        {
            "provider": "native",
            "device": "cpu",
            "solver_route": "scipy_bfgs",
            "device_identity": cpu_identity,
            "peak_vram_mib": None,
        }
    )
    custom_cpu = dict(native)
    custom_cpu.update({"provider": "custom", "solver_route": "custom_bfgs_stepwise"})
    a100_identity = dict(cast(dict[str, object], rtx["device_identity"]))
    a100_identity.update(
        {
            "gpu_uuid": "GPU-a100",
            "gpu_model": "NVIDIA A100-PCIE-40GB",
            "device_kind": "NVIDIA A100-PCIE-40GB",
            "compute_capability": "8.0",
            "total_memory_bytes": 40 * 1024**3,
            "visible_devices": "GPU-a100",
        }
    )
    a100 = dict(rtx)
    a100["device_identity"] = a100_identity
    lanes: list[Path] = []
    for name, row in (
        ("native-cpu", native),
        ("custom-cpu", custom_cpu),
        ("custom-rtx", rtx),
        ("custom-a100", a100),
    ):
        lane = tmp_path / name
        lane.mkdir()
        lane_payload = dict(payload)
        runtime_environment = dict(
            cast(dict[str, object], payload["runtime_environment"])
        )
        is_gpu = row["device"] == "gpu"
        runtime_environment.update(
            {
                "JAX_PLATFORMS": "cuda" if is_gpu else "cpu",
                "SIMSOPT_BACKEND_MODE": (
                    "jax_gpu_parity" if is_gpu else "jax_cpu_parity"
                ),
            }
        )
        lane_payload.update(
            {
                "requested_device": row["device"],
                "method": row["method"],
                "runtime_environment": runtime_environment,
                "device_identity": row["device_identity"],
                "measurements": [row],
            }
        )
        _write_json(lane / "measurements.json", lane_payload)
        lane_memory = {
            "availability": "available" if is_gpu else "unavailable",
            "gpu_uuid": row["device_identity"].get("gpu_uuid"),
            "peak_used_memory_mib": row["peak_vram_mib"] if is_gpu else None,
            "provider_pid": 1234,
            "samples": (
                [{"sampled_at_unix_ns": 1, "used_memory_mib": row["peak_vram_mib"]}]
                if is_gpu
                else []
            ),
            "schema_version": 1,
            "target_pid_observed": is_gpu,
            "unavailable_reason": None if is_gpu else "cpu-device",
        }
        (lane / "memory.json").write_text(
            json.dumps(lane_memory) + "\n", encoding="utf-8"
        )
        _write_valid_trial_trace(lane, row)
        lanes.append(lane)
    return lanes


def _publish_receipt(tmp_path: Path, *, clean: bool = True) -> tuple[Path, Path]:
    run = _runner_directory(tmp_path, clean=clean)
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.11\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "receipt"
    archive = tmp_path / "archive" / "receipt"
    publish(
        (run,),
        environment_lock=lock,
        destination=destination,
        archive_uri=archive.as_uri(),
        repo_root=tmp_path,
    )
    return destination, archive


def _performance_v7_runs(
    tmp_path: Path,
    *,
    warm_pairs: tuple[tuple[float, float], ...],
    providers_by_run: tuple[tuple[str, ...], ...] | None = None,
    custom_route: str = "fused_stepwise",
    custom_intent: str = "fast",
) -> tuple[Path, ...]:
    seed = _runner_v7_directory(tmp_path)
    if providers_by_run is None:
        providers_by_run = (
            ("custom", "optax"),
            *(("custom", "optax") for _ in warm_pairs[1:]),
        )
    assert len(providers_by_run) == len(warm_pairs)
    runs: list[Path] = []
    for index, (custom_warm, optax_warm) in enumerate(warm_pairs):
        providers = providers_by_run[index]
        assert providers
        source_run = tmp_path / f"performance-{index}"
        source_run.mkdir()
        rows: list[dict[str, object]] = []
        child_provenance: list[dict[str, object]] = []
        child_payloads: list[dict[str, object]] = []
        for provider in providers:
            child_root = source_run / provider
            shutil.copytree(seed, child_root)
            child_measurements = child_root / "measurements.json"
            child_payload = _json_object(child_measurements)
            row = dict(_first_measurement(child_payload))
            row["provider"] = provider
            row["warm_seconds"] = custom_warm if provider == "custom" else optax_warm
            if provider == "custom":
                row["intent"] = custom_intent
                row["solver_route"] = custom_route
            else:
                row["intent"] = "fast"
                row["solver_route"] = "optax_lbfgs"
            child_payload["measurements"] = [row]
            _write_json(child_measurements, child_payload)
            rows.append(row)
            child_payloads.append(child_payload)
            child_provenance.append(
                {
                    "provider": provider,
                    "measurements_path": f"{provider}/measurements.json",
                    "measurements_sha256": hashlib.sha256(
                        child_measurements.read_bytes()
                    ).hexdigest(),
                    "gpu_memory_path": f"{provider}/memory.json",
                    "gpu_memory_sha256": hashlib.sha256(
                        (child_root / "memory.json").read_bytes()
                    ).hexdigest(),
                    "measurement_count": 1,
                    "git_commit": child_payload["git_commit"],
                    "git_clean": child_payload["git_clean"],
                    "runtime_environment": child_payload["runtime_environment"],
                    "requested_device": child_payload["requested_device"],
                    "method": child_payload["method"],
                    "device_identity": child_payload["device_identity"],
                }
            )
        parent_payload = dict(child_payloads[0])
        parent_payload.update(
            {
                "provider_child": False,
                "provider_children": child_provenance,
                "measurements": rows,
            }
        )
        _write_json(source_run / "measurements.json", parent_payload)
        runs.append(source_run)
    return tuple(runs)


def test_publish_performance_rejects_wrong_route_at_receipt_boundary(
    tmp_path: Path,
) -> None:
    runs = _performance_v7_runs(
        tmp_path,
        warm_pairs=((0.055, 0.1),),
        custom_route="stepwise",
    )
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            runs,
            environment_lock=lock,
            destination=tmp_path / "tracked" / "wrong-route",
            archive_uri=(tmp_path / "archive" / "wrong-route").as_uri(),
            repo_root=tmp_path,
            qualification_kind="performance",
        )

    assert str(error.value) == (
        "solver route 'stepwise' does not match provider/method 'custom'/'lbfgs'"
    )


def test_publish_performance_rejects_optax_pairing_across_source_runs(
    tmp_path: Path,
) -> None:
    runs = _performance_v7_runs(
        tmp_path,
        warm_pairs=((0.055, 0.1), (0.055, 0.1)),
        providers_by_run=(("custom",), ("optax",)),
    )
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "missing-pair"

    publish(
        runs,
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "missing-pair").as_uri(),
        repo_root=tmp_path,
        qualification_kind="performance",
    )

    qualification = cast(
        dict[str, object], _json_object(destination / "manifest.json")["qualification"]
    )
    ratio = qualification.pop("custom_to_optax_warm_ratio")
    expected_qualification = {
        "passed": False,
        "failure_reasons": [
            "performance-0:coil47:optax-count",
            "performance-1:coil47:custom-count",
        ],
        "comparison_count": 0,
        "custom_warm_seconds_median": 0.055,
        "optax_warm_seconds_median": 0.1,
        "verdict": "fail",
    }
    assert ratio == pytest.approx(0.55)
    assert qualification == expected_qualification
    assert _json_object(destination / "manifest.json")["verdict"] == "fail"
    assert validate_all(destination, repo_root=tmp_path) == 0


def test_publish_performance_rejects_warm_ratio_above_two(
    tmp_path: Path,
) -> None:
    runs = _performance_v7_runs(
        tmp_path,
        warm_pairs=((0.25, 0.1),),
    )
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "ratio-fail"

    publish(
        runs,
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "ratio-fail").as_uri(),
        repo_root=tmp_path,
        qualification_kind="performance",
    )

    qualification = _json_object(destination / "manifest.json")["qualification"]
    assert qualification["passed"] is False
    assert qualification["failure_reasons"] == [
        "custom-to-optax-warm-ratio-exceeds-2.0"
    ]
    assert qualification["comparison_count"] == 1
    assert qualification["custom_warm_seconds_median"] == 0.25
    assert qualification["optax_warm_seconds_median"] == 0.1
    assert qualification["custom_to_optax_warm_ratio"] == 2.5
    assert qualification["verdict"] == "fail"
    assert _json_object(destination / "manifest.json")["verdict"] == "fail"
    assert validate_all(destination, repo_root=tmp_path) == 0


def test_publish_performance_passes_five_real_shape_runs_at_ratio_point_five_five(
    tmp_path: Path,
) -> None:
    runs = _performance_v7_runs(
        tmp_path,
        warm_pairs=(
            (0.013, 0.024),
            (0.0135, 0.025),
            (0.014, 0.0255),
            (0.0145, 0.026),
            (0.015, 0.027),
        ),
    )
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "ratio-pass"

    publish(
        runs,
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "ratio-pass").as_uri(),
        repo_root=tmp_path,
        qualification_kind="performance",
    )

    manifest = _json_object(destination / "manifest.json")
    qualification = cast(dict[str, object], manifest["qualification"])
    assert manifest["verdict"] == "pass"
    assert qualification["passed"] is True
    assert qualification["failure_reasons"] == []
    assert qualification["comparison_count"] == 5
    assert qualification["custom_warm_seconds_median"] == 0.014
    assert qualification["optax_warm_seconds_median"] == 0.0255
    assert qualification["custom_to_optax_warm_ratio"] == pytest.approx(
        0.55, abs=0.002
    )
    assert qualification["verdict"] == "pass"
    assert validate_all(destination, repo_root=tmp_path) == 0


def test_publish_and_validate_receipt_from_a_fresh_process(tmp_path: Path) -> None:
    destination, _archive = _publish_receipt(tmp_path)

    assert validate_all(destination, repo_root=tmp_path) == 0
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "benchmarks"
                / "custom_quasi_newton_receipts.py"
            ),
            "validate-all",
            "--root",
            str(destination),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"validated": 1' in completed.stdout


def test_publish_marks_dirty_runner_diagnostic(tmp_path: Path) -> None:
    destination, _archive = _publish_receipt(tmp_path, clean=False)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_worktree_clean"] is False
    assert manifest["verdict"] == "diagnostic-pass-not-promotion"


def test_publish_rejects_missing_environment_lock(tmp_path: Path) -> None:
    run = _runner_directory(tmp_path)
    with pytest.raises(ValueError, match="environment lock does not exist"):
        publish(
            (run,),
            environment_lock=tmp_path / "missing.lock",
            destination=tmp_path / "tracked" / "receipt",
            archive_uri=(tmp_path / "archive" / "receipt").as_uri(),
            repo_root=tmp_path,
        )


def test_publish_rejects_archive_aliasing_tracked_destination(tmp_path: Path) -> None:
    run = _runner_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.11\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "receipt"

    with pytest.raises(ValueError, match="archive must be distinct"):
        publish(
            (run,),
            environment_lock=lock,
            destination=destination,
            archive_uri=destination.as_uri(),
            repo_root=tmp_path,
        )

    assert not destination.exists()


@pytest.mark.parametrize("location", ["tracked", "archive"])
def test_validate_all_rejects_artifact_tampering(tmp_path: Path, location: str) -> None:
    destination, archive = _publish_receipt(tmp_path)
    target = (destination if location == "tracked" else archive) / "metrics.json"
    target.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_all(destination, repo_root=tmp_path)


def test_validate_all_rejects_environment_lock_tampering(tmp_path: Path) -> None:
    destination, _archive = _publish_receipt(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.12\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment lock checksum mismatch"):
        validate_all(destination, repo_root=tmp_path)


def test_publish_v2_binds_runner_contract_and_archive_inventory(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "receipt-v2"
    archive = tmp_path / "archive" / "receipt-v2"

    publish(
        (run,),
        environment_lock=lock,
        destination=destination,
        archive_uri=archive.as_uri(),
        repo_root=tmp_path,
        qualification_kind="diagnostic",
        archive_storage_identity="test-archive",
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert metrics["schema_version"] == 2
    assert manifest["runner_schema_version"] == 9
    assert manifest["archive_bundle"]["storage_identity"] == "test-archive"
    assert metrics["derivations"]["sample_counts"] == {"custom": 1}
    assert manifest["verdict"] == "diagnostic-pass-not-promotion"
    assert validate_all(destination, repo_root=tmp_path) == 0
    assert (
        validate_all(
            destination,
            repo_root=tmp_path,
            archive_root=tmp_path / "archive",
        )
        == 0
    )


def test_publish_rejects_unparseable_gpu_memory_trace(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    (run / "memory.json").write_text("{}\n", encoding="utf-8")
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="GPU memory artifact"):
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "receipt-v2",
            archive_uri=(tmp_path / "archive" / "receipt-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )


def test_publish_rejects_gpu_memory_peak_mismatch(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    memory = _json_object(run / "memory.json")
    memory["peak_used_memory_mib"] = 201
    memory["samples"] = [{"sampled_at_unix_ns": 1, "used_memory_mib": 201}]
    _write_json(run / "memory.json", memory)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="GPU memory artifact peak does not match peak_vram_mib"
    ):
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "receipt-v2",
            archive_uri=(tmp_path / "archive" / "receipt-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )


def test_publish_rejects_lifetime_peak_diverging_from_phases(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    payload = _json_object(run / "measurements.json")
    row = cast(dict[str, object], cast(list[object], payload["measurements"])[0])
    row["peak_rss_kib"] = 130
    _write_json(run / "measurements.json", payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="process lifetime RSS peak does not match phase"
    ):
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "receipt-v2",
            archive_uri=(tmp_path / "archive" / "receipt-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("evaluations", 5, "production evaluations"),
        ("final_objective", 2.5, "production final objective"),
        ("final_gradient_inf_norm", 2.0e-9, "production final gradient"),
        ("status", 1, "production final status"),
        ("final_parameters", [9.0], "production final parameters"),
    ),
)
def test_publish_rejects_trial_trace_production_binding_mismatch(
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    lanes = _scientific_v7_lanes(tmp_path)
    target = lanes[0]
    payload = _json_object(target / "measurements.json")
    row = dict(_first_measurement(payload))
    bad_production_row = dict(row)
    bad_production_row[field] = bad_value
    for artifact_name in (
        "trial.json",
        "trial.records.jsonl",
        "trial.parameters.npz",
    ):
        (target / artifact_name).unlink()
    _write_valid_trial_trace(target, row, production_row=bad_production_row)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        publish(
            tuple(lanes),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "scientific-v2",
            archive_uri=(tmp_path / "archive" / "scientific-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="scientific",
        )


def test_publish_ignores_probe_trajectory_fields_for_production_binding(
    tmp_path: Path,
) -> None:
    lanes = _scientific_v7_lanes(tmp_path)
    target = lanes[0]
    payload = _json_object(target / "measurements.json")
    production_row = dict(_first_measurement(payload))
    probe_row = dict(production_row)
    probe_row.update(
        {
            "evaluations": 3,
            "final_objective": 3.0,
            "final_gradient_inf_norm": 4.0,
            "final_parameters": [9.0],
            "status": 7,
        }
    )
    for artifact_name in (
        "trial.json",
        "trial.records.jsonl",
        "trial.parameters.npz",
    ):
        (target / artifact_name).unlink()
    _write_valid_trial_trace(target, probe_row, production_row=production_row)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    destination = tmp_path / "tracked" / "scientific-v2"
    publish(
        tuple(lanes),
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "scientific-v2").as_uri(),
        repo_root=tmp_path,
        qualification_kind="scientific",
    )

    manifest = _json_object(destination / "manifest.json")
    assert manifest["verdict"] == "pass"
    assert validate_all(destination, repo_root=tmp_path) == 0


def test_publish_scientific_recomputes_endpoint_and_native_parity(
    tmp_path: Path,
) -> None:
    lanes = _scientific_v7_lanes(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "scientific-v2"

    publish(
        tuple(lanes),
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "scientific-v2").as_uri(),
        repo_root=tmp_path,
        qualification_kind="scientific",
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["verdict"] == "pass"
    qualification = manifest["qualification"]
    assert qualification["passed"] is True
    assert qualification["failure_reasons"] == []
    assert qualification["comparison_count"] == 42
    assert len(qualification["trajectory_comparisons"]) == 3
    assert validate_all(destination, repo_root=tmp_path) == 0


def test_publish_scientific_rejects_mismatched_backend_mode(
    tmp_path: Path,
) -> None:
    lanes = _scientific_v7_lanes(tmp_path)
    mismatched_lane = lanes[1] / "measurements.json"
    payload = _json_object(mismatched_lane)
    runtime_environment = cast(dict[str, object], payload["runtime_environment"])
    runtime_environment["SIMSOPT_BACKEND_MODE"] = "jax_cpu_fast"
    _write_json(mismatched_lane, payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SIMSOPT_BACKEND_MODE"):
        publish(
            tuple(lanes),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "scientific-v2",
            archive_uri=(tmp_path / "archive" / "scientific-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="scientific",
        )


def test_publish_scientific_rejects_mismatched_jax_version(tmp_path: Path) -> None:
    lanes = _scientific_v7_lanes(tmp_path)
    mismatched_lane = lanes[-1] / "measurements.json"
    payload = _json_object(mismatched_lane)
    payload["jax_version"] = "0.9.9"
    _write_json(mismatched_lane, payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="jax_version differs across lanes"):
        publish(
            tuple(lanes),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "scientific-v2",
            archive_uri=(tmp_path / "archive" / "scientific-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="scientific",
        )


def test_publish_scientific_rejects_missing_native_authority(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "scientific-v2"

    publish(
        (run,),
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "scientific-v2").as_uri(),
        repo_root=tmp_path,
        qualification_kind="scientific",
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["verdict"] == "fail"
    assert (
        "coil47:native-authority-count" in manifest["qualification"]["failure_reasons"]
    )


def test_publish_rejects_unknown_runner_schema(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    payload = json.loads((run / "measurements.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 10
    (run / "measurements.json").write_text(json.dumps(payload), encoding="utf-8")
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runner schema version"):
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "receipt-v2",
            archive_uri=(tmp_path / "archive" / "receipt-v2").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
            archive_storage_identity="test-archive",
        )


def test_publish_validates_the_copied_runner_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "snapshot"
    archive = tmp_path / "archive" / "snapshot"
    original_copy = receipt_module._copy_runner_tree

    def mutate_then_copy(source: Path, target: Path) -> None:
        if source == run:
            mutated_payload = _json_object(source / "measurements.json")
            mutated_payload["git_commit"] = "f" * 40
            _write_json(source / "measurements.json", mutated_payload)
        original_copy(source, target)

    monkeypatch.setattr(receipt_module, "_copy_runner_tree", mutate_then_copy)

    with pytest.raises(ValueError) as error:
        publish(
            (run,),
            environment_lock=lock,
            destination=destination,
            archive_uri=archive.as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "runner git_commit does not resolve to a commit in repo_root: " + "f" * 40
    )
    assert not destination.exists()
    assert not archive.exists()


def test_validate_all_rejects_duplicate_performance_source_runs(tmp_path: Path) -> None:
    runs = _performance_v7_runs(tmp_path, warm_pairs=((0.055, 0.1),))
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "duplicate-source-runs"
    publish(
        runs,
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "duplicate-source-runs").as_uri(),
        repo_root=tmp_path,
        qualification_kind="performance",
    )
    for document in ("metrics.json", "manifest.json"):
        payload = _json_object(destination / document)
        payload["source_runs"] = [runs[0].name, runs[0].name]
        _write_json(destination / document, payload)

    with pytest.raises(ValueError) as error:
        validate_all(destination, repo_root=tmp_path)

    assert str(error.value) == f"receipt source runs are duplicated: {runs[0].name}"


@pytest.mark.parametrize(
    "alias_template",
    ("./{name}", "{name}/", "{name}/.", "{name}//"),
)
def test_validate_all_rejects_aliased_source_run_names(
    tmp_path: Path,
    alias_template: str,
) -> None:
    """Path-normalized aliases of one physical run must not multiply samples."""

    runs = _performance_v7_runs(tmp_path, warm_pairs=((0.055, 0.1),))
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "aliased-source-runs"
    publish(
        runs,
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "aliased-source-runs").as_uri(),
        repo_root=tmp_path,
        qualification_kind="performance",
    )
    alias = alias_template.format(name=runs[0].name)
    for document in ("metrics.json", "manifest.json"):
        payload = _json_object(destination / document)
        payload["source_runs"] = [runs[0].name, alias]
        _write_json(destination / document, payload)

    with pytest.raises(ValueError) as error:
        validate_all(destination, repo_root=tmp_path)

    assert str(error.value) == (
        f"receipt source run is not a canonical directory name: {alias!r}"
    )


def test_publish_rejects_invalid_nested_legacy_runner_commit(tmp_path: Path) -> None:
    run = _runner_directory(tmp_path)
    nested = run / "nested"
    nested.mkdir()
    _write_json(nested / "measurements.json", {"git_commit": "invalid"})
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.11\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "legacy-nested-commit",
            archive_uri=(tmp_path / "archive" / "legacy-nested-commit").as_uri(),
            repo_root=tmp_path,
        )

    assert str(error.value) == (
        "runner git_commit must be exactly 40 lowercase hexadecimal characters"
    )
