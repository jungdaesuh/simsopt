from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest
from benchmarks.custom_quasi_newton_receipts import publish, validate_all

EndpointMutation = Literal["missing", "null", "wrong-type"]


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


def _json_object(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _first_measurement(payload: dict[str, object]) -> dict[str, object]:
    measurements = cast(list[dict[str, object]], payload["measurements"])
    return measurements[0]


def _runner_v7_directory(root: Path) -> Path:
    commit = _git_commit(root)
    run = root / "runner-v7"
    run.mkdir()
    device_identity: dict[str, object] = {
        "requested_device": "gpu",
        "backend": "gpu",
        "platform": "gpu",
        "jax_device": "cuda:0",
        "device_kind": "NVIDIA GeForce RTX 5090",
        "device_id": 0,
        "process_index": 0,
        "gpu_uuid": "GPU-0000f78e-c05d-e01c-303f-d644f4341f00",
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "total_memory_bytes": 32 * 1024**3,
        "driver_version": "590.48",
        "cuda_version": "CUDA 13.0",
        "visible_devices": "GPU-0000f78e-c05d-e01c-303f-d644f4341f00",
        "hostname": "test-host",
        "scheduler_job_id": None,
    }
    payload: dict[str, object] = {
        "schema_version": 9,
        "git_commit": commit,
        "git_clean": True,
        "orchestrator_git_clean": True,
        "provider_child": True,
        "provider_children": [],
        "requested_device": "gpu",
        "method": "lbfgs",
        "runtime_environment": {},
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
                "peak_vram_mib": 200.0,
                "process_pid": 1234,
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
                "scientific_observables": {"final_volume": 0.25},
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
    _write_json(run / "measurements.json", payload)
    (run / "memory.json").write_text(
        json.dumps(
            {
                "availability": "available",
                "gpu_uuid": "GPU-0000f78e-c05d-e01c-303f-d644f4341f00",
                "peak_used_memory_mib": 200,
                "provider_pid": 1234,
                "samples": [{"sampled_at_unix_ns": 1, "used_memory_mib": 200}],
                "schema_version": 1,
                "target_pid_observed": True,
                "unavailable_reason": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run


def _parent_runner_v7_directory(root: Path) -> Path:
    child = _runner_v7_directory(root)
    parent = root / "parent-v7"
    parent.mkdir()
    nested_child = parent / "custom"
    child.rename(nested_child)
    child_path = nested_child / "measurements.json"
    child_payload = _json_object(child_path)
    parent_payload = dict(child_payload)
    parent_payload.update(
        {
            "provider_child": False,
            "provider_children": [
                {
                    "provider": "custom",
                    "measurements_path": "custom/measurements.json",
                    "measurements_sha256": hashlib.sha256(
                        child_path.read_bytes()
                    ).hexdigest(),
                    "gpu_memory_path": "custom/memory.json",
                    "gpu_memory_sha256": hashlib.sha256(
                        (nested_child / "memory.json").read_bytes()
                    ).hexdigest(),
                    "measurement_count": 1,
                    "git_commit": child_payload["git_commit"],
                    "git_clean": child_payload["git_clean"],
                    "runtime_environment": child_payload["runtime_environment"],
                    "requested_device": child_payload["requested_device"],
                    "method": child_payload["method"],
                    "device_identity": child_payload["device_identity"],
                }
            ],
        }
    )
    _write_json(parent / "measurements.json", parent_payload)
    return parent


@pytest.fixture
def receipt_v2(tmp_path: Path) -> Path:
    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "receipt-v2"
    publish(
        (run,),
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "receipt-v2").as_uri(),
        repo_root=tmp_path,
        qualification_kind="diagnostic",
        archive_storage_identity="test-archive",
    )
    return destination


def _mutate_json(
    path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    payload = _json_object(path)
    mutation(payload)
    _write_json(path, payload)


def _mutate_retained_measurement(
    receipt: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    # Raw and flattened copies must agree so validation reaches the row contract.
    for path in (
        receipt / "raw" / "runner-v7" / "measurements.json",
        receipt / "metrics.json",
    ):
        _mutate_json(path, lambda payload: mutation(_first_measurement(payload)))


def _invalidate_field(
    row: dict[str, object],
    field: str,
    mutation: EndpointMutation,
) -> None:
    if mutation == "missing":
        row.pop(field)
    elif mutation == "null":
        row[field] = None
    else:
        row[field] = []


def test_unmodified_v2_fixture_is_valid(receipt_v2: Path, tmp_path: Path) -> None:
    assert validate_all(receipt_v2, repo_root=tmp_path) == 0


@pytest.mark.parametrize(
    ("commit", "message"),
    (
        (
            "abc123",
            "runner git_commit must be exactly 40 lowercase hexadecimal characters",
        ),
        (
            "g" * 40,
            "runner git_commit must be exactly 40 lowercase hexadecimal characters",
        ),
        (
            "0" * 40,
            "runner git_commit does not resolve to a commit in repo_root: " + "0" * 40,
        ),
    ),
)
def test_publish_rejects_unusable_runner_commit(
    tmp_path: Path,
    commit: str,
    message: str,
) -> None:
    run = _runner_v7_directory(tmp_path)
    payload = _json_object(run / "measurements.json")
    payload["git_commit"] = commit
    _write_json(run / "measurements.json", payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "invalid-commit",
            archive_uri=(tmp_path / "archive" / "invalid-commit").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == message


def test_validate_all_rejects_unresolved_retained_runner_commit(
    tmp_path: Path,
) -> None:
    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "unresolved-retained-commit"
    publish(
        (run,),
        environment_lock=lock,
        destination=destination,
        archive_uri=(tmp_path / "archive" / "unresolved-retained-commit").as_uri(),
        repo_root=tmp_path,
        qualification_kind="diagnostic",
    )

    unresolved_commit = "0123456789abcdef" * 2 + "01234567"
    retained = destination / "raw" / run.name / "measurements.json"
    payload = _json_object(retained)
    payload["git_commit"] = unresolved_commit
    _write_json(retained, payload)

    with pytest.raises(ValueError) as error:
        validate_all(destination, repo_root=tmp_path)

    assert str(error.value) == (
        "runner git_commit does not resolve to a commit in repo_root: "
        + unresolved_commit
    )


def test_publish_rejects_non_git_repo_root(tmp_path: Path) -> None:
    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    non_git_root = tmp_path.parent / f"{tmp_path.name}-not-a-repository"
    non_git_root.mkdir()

    with pytest.raises(ValueError) as error:
        publish(
            (run,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "non-git-root",
            archive_uri=(tmp_path / "archive" / "non-git-root").as_uri(),
            repo_root=non_git_root,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == f"repo_root is not a Git work tree: {non_git_root}"


@pytest.mark.parametrize(
    ("field", "message"),
    (
        (
            "gpu_memory_sha256",
            "provider child gpu_memory_path checksum mismatch",
        ),
        (
            "trial_trace_sha256",
            "provider child trial_trace_path checksum mismatch",
        ),
    ),
)
def test_publish_rejects_zeroed_child_artifact_checksum(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    payload = _json_object(parent / "measurements.json")
    children = cast(list[dict[str, object]], payload["provider_children"])
    if field == "trial_trace_sha256":
        trial_path = parent / "custom" / "trial.json"
        trial_path.write_text("synthetic trial trace\n", encoding="utf-8")
        children[0]["trial_trace_path"] = "custom/trial.json"
    children[0][field] = "0" * 64
    _write_json(parent / "measurements.json", payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "zeroed-checksum",
            archive_uri=(tmp_path / "archive" / "zeroed-checksum").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("intent", "route"),
    (
        ("fast", "stepwise"),
        ("fast", None),
        ("fast", "custom_bfgs_stepwise"),
        ("parity", "fused_stepwise"),
    ),
)
def test_v2_rejects_custom_lbfgs_route_drift_for_each_intent(
    receipt_v2: Path,
    tmp_path: Path,
    intent: str,
    route: str | None,
) -> None:
    def tamper_route(row: dict[str, object]) -> None:
        row["intent"] = intent
        if route is None:
            row.pop("solver_route")
        else:
            row["solver_route"] = route

    _mutate_retained_measurement(receipt_v2, tamper_route)

    with pytest.raises(ValueError, match="solver route"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    (
        "success",
        "iterations",
        "maxiter",
        "initial_gradient_inf_norm",
        "final_gradient_inf_norm",
        "parameters_finite",
        "observables_finite",
        "inner_success",
    ),
)
@pytest.mark.parametrize("mutation", ("missing", "null", "wrong-type"))
def test_v2_rejects_invalid_required_endpoint_inputs(
    receipt_v2: Path,
    tmp_path: Path,
    field: str,
    mutation: EndpointMutation,
) -> None:
    _mutate_retained_measurement(
        receipt_v2,
        lambda row: _invalidate_field(row, field, mutation),
    )

    with pytest.raises(TypeError, match=rf"{field} must be"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize("field", ("endpoint_certificate", "stopping_reason"))
@pytest.mark.parametrize("mutation", ("missing", "null", "wrong-type"))
def test_v2_rejects_invalid_stored_endpoint_evidence(
    receipt_v2: Path,
    tmp_path: Path,
    field: str,
    mutation: EndpointMutation,
) -> None:
    _mutate_retained_measurement(
        receipt_v2,
        lambda row: _invalidate_field(row, field, mutation),
    )

    with pytest.raises(ValueError, match=rf"stored .*{field.replace('_', ' ')}"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize("nonfinite", (math.nan, math.inf, -math.inf))
@pytest.mark.parametrize(
    "field",
    (
        "preparation_seconds",
        "first_execution_seconds",
        "cold_seconds",
        "warm_seconds",
        "initial_objective",
        "initial_gradient_inf_norm",
        "final_objective",
        "final_gradient_inf_norm",
        "solver_start_rss_kib",
        "solver_peak_rss_kib",
        "solver_peak_rss_delta_kib",
        "peak_vram_mib",
        "scientific_certification_seconds",
        "constraint_norm",
    ),
)
def test_v2_rejects_nonfinite_scalar_metrics(
    receipt_v2: Path,
    tmp_path: Path,
    field: str,
    nonfinite: float,
) -> None:
    _mutate_retained_measurement(
        receipt_v2,
        lambda row: row.__setitem__(field, nonfinite),
    )

    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("iterations", -1, "iterations must be nonnegative"),
        ("maxiter", 0, "maxiter must be positive"),
        (
            "initial_gradient_inf_norm",
            -1.0,
            "initial_gradient_inf_norm must be nonnegative",
        ),
        (
            "final_gradient_inf_norm",
            -1.0,
            "final_gradient_inf_norm must be nonnegative",
        ),
        ("constraint_norm", -1.0, "constraint_norm must be nonnegative"),
    ),
)
def test_v2_rejects_negative_endpoint_inputs(
    receipt_v2: Path,
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    _mutate_retained_measurement(
        receipt_v2,
        lambda row: row.__setitem__(field, value),
    )

    with pytest.raises(ValueError, match=message):
        validate_all(receipt_v2, repo_root=tmp_path)


def test_v2_rejects_iterations_beyond_budget(
    receipt_v2: Path,
    tmp_path: Path,
) -> None:
    def exceed_budget(row: dict[str, object]) -> None:
        row["iterations"] = 21
        work = cast(dict[str, object], row["work_counters"])
        work["accepted_iterations"] = 21

    _mutate_retained_measurement(receipt_v2, exceed_budget)

    with pytest.raises(ValueError, match="iterations cannot exceed maxiter"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize("nonfinite", (math.nan, math.inf, -math.inf))
def test_v2_rejects_nonfinite_scientific_observables(
    receipt_v2: Path,
    tmp_path: Path,
    nonfinite: float,
) -> None:
    def tamper_observable(row: dict[str, object]) -> None:
        observables = cast(dict[str, object], row["scientific_observables"])
        observables["final_volume"] = nonfinite

    _mutate_retained_measurement(receipt_v2, tamper_observable)

    with pytest.raises(
        ValueError,
        match=r"scientific_observables.final_volume must be finite",
    ):
        validate_all(receipt_v2, repo_root=tmp_path)


def test_v2_rejects_tampered_metric_derivations(
    receipt_v2: Path,
    tmp_path: Path,
) -> None:
    def tamper_derivation(metrics: dict[str, object]) -> None:
        derivations = cast(dict[str, object], metrics["derivations"])
        sample_counts = cast(dict[str, object], derivations["sample_counts"])
        sample_counts["custom"] = 2

    _mutate_json(receipt_v2 / "metrics.json", tamper_derivation)

    with pytest.raises(ValueError, match="derivations do not match raw runs"):
        validate_all(receipt_v2, repo_root=tmp_path)


def test_v2_rejects_tampered_manifest_sample_count_derivation(
    receipt_v2: Path,
    tmp_path: Path,
) -> None:
    def tamper_sample_count(manifest: dict[str, object]) -> None:
        sample_counts = cast(dict[str, object], manifest["expected_sample_counts"])
        sample_counts["custom"] = 2

    _mutate_json(receipt_v2 / "manifest.json", tamper_sample_count)

    with pytest.raises(ValueError, match="sample counts do not match raw runs"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize("document", ("metrics.json", "manifest.json"))
def test_v2_rejects_tampered_qualification(
    receipt_v2: Path,
    tmp_path: Path,
    document: str,
) -> None:
    def tamper_qualification(payload: dict[str, object]) -> None:
        qualification = cast(dict[str, object], payload["qualification"])
        qualification["passed"] = True

    _mutate_json(receipt_v2 / document, tamper_qualification)

    with pytest.raises(ValueError, match="qualification does not match raw runs"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize("document", ("metrics.json", "manifest.json"))
def test_v2_rejects_tampered_verdict(
    receipt_v2: Path,
    tmp_path: Path,
    document: str,
) -> None:
    _mutate_json(
        receipt_v2 / document,
        lambda payload: payload.__setitem__("verdict", "pass"),
    )

    with pytest.raises(ValueError, match="verdict does not match raw runs"):
        validate_all(receipt_v2, repo_root=tmp_path)


def test_validate_all_rejects_unknown_receipt_schema(
    receipt_v2: Path,
    tmp_path: Path,
) -> None:
    _mutate_json(
        receipt_v2 / "manifest.json",
        lambda manifest: manifest.__setitem__("schema_version", 999),
    )

    with pytest.raises(ValueError, match="unsupported receipt schema version: 999"):
        validate_all(receipt_v2, repo_root=tmp_path)


@pytest.mark.parametrize(
    "escaping_path",
    ("../outside.json", "traces/../../outside.json", "/tmp/outside.json"),
)
def test_v2_rejects_diagnostic_artifact_path_escape(
    receipt_v2: Path,
    tmp_path: Path,
    escaping_path: str,
) -> None:
    def tamper_artifact_reference(row: dict[str, object]) -> None:
        artifacts = cast(dict[str, object], row["diagnostic_artifacts"])
        artifacts["memory_trace"] = escaping_path

    _mutate_retained_measurement(receipt_v2, tamper_artifact_reference)

    with pytest.raises(ValueError, match="diagnostic artifact path escapes source run"):
        validate_all(receipt_v2, repo_root=tmp_path)


def test_v2_rejects_cpu_fallback_labeled_as_gpu(
    receipt_v2: Path,
    tmp_path: Path,
) -> None:
    def label_cpu_as_gpu(row: dict[str, object]) -> None:
        identity = cast(dict[str, object], row["device_identity"])
        identity["backend"] = "cpu"
        identity["platform"] = "cpu"
        identity["jax_device"] = "TFRT_CPU_0"

    _mutate_retained_measurement(receipt_v2, label_cpu_as_gpu)

    with pytest.raises(
        ValueError,
        match="provider child row identity|GPU measurement backend",
    ):
        validate_all(receipt_v2, repo_root=tmp_path)


def test_publish_rejects_provider_child_from_another_commit(tmp_path: Path) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    child_path = parent / "custom" / "measurements.json"
    child_payload = _json_object(child_path)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "user.email",
            "receipt-tests@example.com",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Receipt Tests"],
        check=True,
        capture_output=True,
        text=True,
    )
    second_marker = tmp_path / "receipt-source-second.txt"
    second_marker.write_text("second synthetic source\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", second_marker.name],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "second synthetic source"],
        check=True,
        capture_output=True,
        text=True,
    )
    second_commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    child_payload["git_commit"] = second_commit
    _write_json(child_path, child_payload)
    parent_payload = _json_object(parent / "measurements.json")
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["git_commit"] = second_commit
    children[0]["measurements_sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="commit differs from parent"):
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "mixed-commit",
            archive_uri=(tmp_path / "archive" / "mixed-commit").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )


def test_publish_requires_trial_binding_for_an_emitted_child_trial_trace(
    tmp_path: Path,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    child_path = parent / "custom" / "measurements.json"
    child_payload = _json_object(child_path)
    child_row = _first_measurement(child_payload)
    diagnostic_artifacts = cast(dict[str, object], child_row["diagnostic_artifacts"])
    diagnostic_artifacts["trial_trace"] = "trial.json"
    _write_json(child_path, child_payload)
    parent_payload = _json_object(parent / "measurements.json")
    parent_payload["measurements"] = child_payload["measurements"]
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["measurements_sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "missing-emitted-trial-binding",
            archive_uri=(
                tmp_path / "archive" / "missing-emitted-trial-binding"
            ).as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "provider child omitted trial_trace_path/trial_trace_sha256"
    )


def test_publish_rejects_trial_binding_to_a_non_emitted_child_artifact(
    tmp_path: Path,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    child_path = parent / "custom" / "measurements.json"
    child_payload = _json_object(child_path)
    child_row = _first_measurement(child_payload)
    diagnostic_artifacts = cast(dict[str, object], child_row["diagnostic_artifacts"])
    diagnostic_artifacts["trial_trace"] = "trial.json"
    _write_json(child_path, child_payload)
    parent_payload = _json_object(parent / "measurements.json")
    parent_payload["measurements"] = child_payload["measurements"]
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["measurements_sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    children[0]["trial_trace_path"] = "custom/memory.json"
    children[0]["trial_trace_sha256"] = hashlib.sha256(
        (parent / "custom" / "memory.json").read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "wrong-emitted-trial-binding",
            archive_uri=(tmp_path / "archive" / "wrong-emitted-trial-binding").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "provider child trial trace path does not bind the emitted measurement artifact"
    )


def test_publish_rejects_gpu_binding_to_a_non_emitted_child_artifact(
    tmp_path: Path,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    unrelated_memory = parent / "custom" / "unrelated-memory.json"
    unrelated_memory.write_text("unrelated GPU memory artifact\n", encoding="utf-8")
    parent_payload = _json_object(parent / "measurements.json")
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["gpu_memory_path"] = "custom/unrelated-memory.json"
    children[0]["gpu_memory_sha256"] = hashlib.sha256(
        unrelated_memory.read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "wrong-emitted-memory-binding",
            archive_uri=(
                tmp_path / "archive" / "wrong-emitted-memory-binding"
            ).as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "provider child GPU memory path does not bind the emitted measurement artifact"
    )


def test_publish_rejects_capture_without_an_emitted_child_trial_trace(
    tmp_path: Path,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    child_path = parent / "custom" / "measurements.json"
    child_payload = _json_object(child_path)
    child_payload["capture_boozer_trial_trace"] = True
    _write_json(child_path, child_payload)
    parent_payload = _json_object(parent / "measurements.json")
    parent_payload["capture_boozer_trial_trace"] = True
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["measurements_sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "missing-captured-trial",
            archive_uri=(tmp_path / "archive" / "missing-captured-trial").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "provider child trial capture requires exactly one emitted trial trace"
    )


def test_publish_rejects_trial_binding_without_an_emitted_child_trial_trace(
    tmp_path: Path,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    trial_path = parent / "custom" / "trial.json"
    trial_path.write_text("synthetic trial trace\n", encoding="utf-8")
    parent_payload = _json_object(parent / "measurements.json")
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["trial_trace_path"] = "custom/trial.json"
    children[0]["trial_trace_sha256"] = hashlib.sha256(
        trial_path.read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "binding-without-trial",
            archive_uri=(tmp_path / "archive" / "binding-without-trial").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "provider child trial binding requires exactly one emitted trial trace"
    )


def test_publish_rejects_two_rows_bound_to_the_same_child_trial_trace(
    tmp_path: Path,
) -> None:
    parent = _parent_runner_v7_directory(tmp_path)
    child_path = parent / "custom" / "measurements.json"
    child_payload = _json_object(child_path)
    first_row = _first_measurement(child_payload)
    first_artifacts = cast(dict[str, object], first_row["diagnostic_artifacts"])
    first_artifacts["trial_trace"] = "trial.json"
    child_payload["measurements"] = [first_row, dict(first_row)]
    _write_json(child_path, child_payload)
    trial_path = parent / "custom" / "trial.json"
    trial_path.write_text("synthetic trial trace\n", encoding="utf-8")
    parent_payload = _json_object(parent / "measurements.json")
    parent_payload["measurements"] = child_payload["measurements"]
    children = cast(list[dict[str, object]], parent_payload["provider_children"])
    children[0]["measurements_sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    children[0]["measurement_count"] = 2
    children[0]["trial_trace_path"] = "custom/trial.json"
    children[0]["trial_trace_sha256"] = hashlib.sha256(
        trial_path.read_bytes()
    ).hexdigest()
    _write_json(parent / "measurements.json", parent_payload)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        publish(
            (parent,),
            environment_lock=lock,
            destination=tmp_path / "tracked" / "two-rows-one-trial",
            archive_uri=(tmp_path / "archive" / "two-rows-one-trial").as_uri(),
            repo_root=tmp_path,
            qualification_kind="diagnostic",
        )

    assert str(error.value) == (
        "provider child trial binding requires exactly one emitted trial trace"
    )


@pytest.mark.parametrize(
    "forged_uuid",
    (
        "GPU-",
        "GPU-placeholder",
        "GPU-0000F78E-C05D-E01C-303F-D644F4341F00",
        "GPU-0000f78e-c05d-e01c-303f",
        "",
    ),
)
def test_validate_all_rejects_noncanonical_gpu_uuid(
    tmp_path: Path,
    forged_uuid: str,
) -> None:
    """Placeholder or malformed UUIDs must not satisfy the identity gates."""

    run = _runner_v7_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("jax==0.10.0\n", encoding="utf-8")
    receipt = tmp_path / "tracked" / "noncanonical-uuid"
    publish(
        (run,),
        environment_lock=lock,
        destination=receipt,
        archive_uri=(tmp_path / "archive" / "noncanonical-uuid").as_uri(),
        repo_root=tmp_path,
        qualification_kind="diagnostic",
    )
    for payload_path in sorted(receipt.rglob("measurements.json")):
        payload = _json_object(payload_path)
        identity = cast(dict[str, object], payload.get("device_identity"))
        if identity is not None:
            identity["gpu_uuid"] = forged_uuid
        for row in cast(list[dict[str, object]], payload.get("measurements", [])):
            row_identity = cast(dict[str, object], row["device_identity"])
            row_identity["gpu_uuid"] = forged_uuid
        _write_json(payload_path, payload)

    with pytest.raises(ValueError, match="canonical NVIDIA GPU UUID"):
        validate_all(receipt, repo_root=tmp_path)
