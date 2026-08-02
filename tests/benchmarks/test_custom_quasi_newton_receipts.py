from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

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


def _runner_directory(root: Path, *, clean: bool = True) -> Path:
    run = root / "runner-case"
    run.mkdir()
    (run / "raw").mkdir()
    (run / "raw" / "stdout.json").write_text("{}\n", encoding="utf-8")
    (run / "measurements.json").write_text(
        json.dumps(
            {
                "git_commit": "abc123",
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
    run = root / "runner-v7"
    run.mkdir()
    device_identity = {
        "requested_device": "gpu",
        "backend": "gpu",
        "platform": "cuda",
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
                "schema_version": 7,
                "git_commit": "abc123",
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
                        "solver_route": "stepwise",
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
                        "peak_rss_kib": 130,
                        "peak_rss_scope": "provider_child_process_lifetime",
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
                            "advance_observations": 3,
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
    (run / "memory.json").write_text("{}\n", encoding="utf-8")
    return run


def _write_valid_trial_trace(lane: Path, row: dict[str, object]) -> None:
    initial_parameters = np.asarray(row["initial_parameters"], dtype=np.float64)
    final_parameters = np.asarray(row["final_parameters"], dtype=np.float64)
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
        (lane / "memory.json").write_text("{}\n", encoding="utf-8")
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
    assert manifest["runner_schema_version"] == 7
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
    payload["schema_version"] = 8
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
