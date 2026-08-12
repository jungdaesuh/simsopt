from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks import run_single_stage_fullspace_gpu as runner
from benchmarks.single_stage_fullspace_bootstrap import (
    SCHEMA_VERSION as BOOTSTRAP_SCHEMA_VERSION,
)
from benchmarks.single_stage_fullspace_receipt import (
    DeviceLane,
    RunPhase,
    RunRequest,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    ImportBinding,
    RuntimeEvidence,
    RuntimeIdentity,
    RuntimeObservation,
    SourceIdentity,
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from simsopt_jax.solve.fullspace import FullSpaceRoute


def _runtime(campaign: Path, *, device_name: str = "NVIDIA RTX 5090"):
    source = SourceIdentity(
        snapshot_manifest=ArtifactRef(
            "source-snapshot/source-manifest.json",
            "a" * 64,
            10,
            SOURCE_MANIFEST_SCHEMA_VERSION,
        ),
        git_head="b" * 40,
        tracked_diff_sha256="c" * 64,
        untracked_bytes_manifest_sha256="d" * 64,
        repo_root="/source",
    )
    identity = RuntimeIdentity(
        argv=("runner.py", "--phase=first-eval"),
        cwd=str(campaign / "source-snapshot"),
        python_executable="/python",
        python_version="3.test",
        jax_version="test",
        jaxlib_version="test",
        simsopt_module_path="/snapshot/src/simsopt/__init__.py",
        simsopt_jax_module_path="/snapshot/src/simsopt_jax/__init__.py",
        native_extension_path="/snapshot/src/simsoptpp.so",
        backend="gpu",
        device_uuid="GPU-test",
        driver_version="test",
        effective_environment_sha256="e" * 64,
    )
    binding = ImportBinding("runner", "benchmarks/runner.py", 1, "f" * 64)
    evidence = RuntimeEvidence(
        source,
        RuntimeObservation(identity, binding, (binding,), (), device_name, "CUDA"),
        "a" * 64,
    )
    payload = canonical_json_bytes(evidence.to_payload())
    reference = ArtifactRef(
        "evidence/runtime-evidence.json",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        RUNTIME_EVIDENCE_SCHEMA_VERSION,
    )
    return evidence, reference


def _request(device: DeviceLane = DeviceLane.RTX5090) -> RunRequest:
    return RunRequest(RunPhase.FIRST_EVAL, FullSpaceRoute.CFS_P0, device, None, None)


def _canary_request(
    route: FullSpaceRoute = FullSpaceRoute.CFS_P0,
    *,
    steps: int = 10,
) -> RunRequest:
    return RunRequest(RunPhase.CANARY, route, DeviceLane.RTX5090, steps, None)


def test_probe_separates_compile_execution_and_records_controlled_transfers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_device = jax.devices("cpu")[0]
    bootstrap = SimpleNamespace(
        z0=jnp.asarray((1.0, 2.0), dtype=jnp.float64),
        problem=jnp.asarray((3.0,), dtype=jnp.float64),
    )

    def kernel(z: jax.Array, _problem: jax.Array) -> tuple[jax.Array, ...]:
        dtype = z.dtype
        return (
            jnp.sum(z),
            jnp.asarray(True),
            jnp.asarray(2.0, dtype=dtype),
            jnp.asarray(1.5, dtype=dtype),
            jnp.asarray(1.0e-7, dtype=dtype),
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(0.25, dtype=dtype),
            jnp.asarray(0.25, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        )

    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: [cpu_device])
    monkeypatch.setattr(runner, "_first_eval_kernel", kernel)

    numerical, timing, transfers = runner.run_first_eval_probe(bootstrap)

    assert numerical["value"] == 3.0
    assert numerical["gradient_all_finite"] is True
    assert numerical["changed_state_l2"] > 0.0
    assert timing["cold_compile_ns"] >= 0
    assert timing["cold_execution_ns"] >= 0
    assert transfers["initial_h2d_calls"] == 1
    assert transfers["hot_d2h_calls"] == 0
    assert transfers["final_d2h_calls"] == 1
    assert transfers["timed_execution_transfer_guard"] == "disallow"


def test_first_eval_artifact_is_diagnostic_provenance_bound_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    runs = campaign / "runs"
    runs.mkdir()
    runtime, runtime_ref = _runtime(campaign)
    bootstrap_bytes = canonical_json_bytes({"schema_version": BOOTSTRAP_SCHEMA_VERSION})
    bootstrap_path = campaign / "artifacts/bootstrap.json"
    bootstrap_path.parent.mkdir()
    bootstrap_path.write_bytes(bootstrap_bytes)
    bootstrap_ref = ArtifactRef(
        "artifacts/bootstrap.json",
        hashlib.sha256(bootstrap_bytes).hexdigest(),
        len(bootstrap_bytes),
        BOOTSTRAP_SCHEMA_VERSION,
    )
    monkeypatch.setattr(
        runner,
        "validate_bootstrap_artifact",
        lambda *_args, **_kwargs: {"runtime_evidence": asdict(runtime_ref)},
    )
    calls = 0

    def probe(_bootstrap: object):
        nonlocal calls
        calls += 1
        return (
            {"value": 1.0, "gradient_all_finite": True},
            {"cold_compile_ns": 2, "cold_execution_ns": 3},
            {"hot_d2h_calls": 0},
        )

    monkeypatch.setattr(runner, "run_first_eval_probe", probe)
    output = runs / "first-eval.json"

    reference = runner.publish_first_eval_evidence(
        output,
        request=_request(),
        campaign_root=campaign,
        snapshot_root=snapshot,
        runtime_evidence=runtime,
        runtime_evidence_ref=runtime_ref,
        bootstrap_artifact_ref=bootstrap_ref,
        bootstrap=SimpleNamespace(),
    )

    assert calls == 1
    assert reference.schema_version == runner.FIRST_EVAL_SCHEMA_VERSION
    assert output.stat().st_mode & 0o222 == 0
    document = load_canonical_json_bytes(output.read_bytes())
    assert isinstance(document, dict)
    assert document["endpoint_certificate_produced"] is False
    assert document["terminal_status"] == "DIAGNOSTIC_SUCCESS"
    assert document["runtime_evidence"] == asdict(runtime_ref)
    assert document["bootstrap_artifact"] == asdict(bootstrap_ref)


def test_first_eval_rejects_device_lane_mismatch_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    runs = campaign / "runs"
    runs.mkdir()
    runtime, runtime_ref = _runtime(campaign, device_name="NVIDIA A100-SXM4-40GB")
    bootstrap_bytes = canonical_json_bytes({"schema_version": BOOTSTRAP_SCHEMA_VERSION})
    bootstrap_path = campaign / "bootstrap.json"
    bootstrap_path.write_bytes(bootstrap_bytes)
    bootstrap_ref = ArtifactRef(
        "bootstrap.json",
        hashlib.sha256(bootstrap_bytes).hexdigest(),
        len(bootstrap_bytes),
        BOOTSTRAP_SCHEMA_VERSION,
    )
    monkeypatch.setattr(
        runner,
        "validate_bootstrap_artifact",
        lambda *_args, **_kwargs: {"runtime_evidence": asdict(runtime_ref)},
    )
    monkeypatch.setattr(
        runner,
        "run_first_eval_probe",
        lambda *_args: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    with pytest.raises(ValueError, match="physical GPU"):
        runner.publish_first_eval_evidence(
            runs / "first-eval.json",
            request=_request(),
            campaign_root=campaign,
            snapshot_root=snapshot,
            runtime_evidence=runtime,
            runtime_evidence_ref=runtime_ref,
            bootstrap_artifact_ref=bootstrap_ref,
            bootstrap=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    ("maximum_iterations", "execution_key"),
    ((10, "ten_step_execution_ns"), (100, "hundred_step_execution_ns")),
)
def test_cfs_p0_probe_prepares_once_runs_staged_budget_and_materializes_once(
    monkeypatch: pytest.MonkeyPatch,
    maximum_iterations: int,
    execution_key: str,
) -> None:
    cpu_device = jax.devices("cpu")[0]
    bootstrap = SimpleNamespace(
        z0=jnp.asarray((1.0, 2.0), dtype=jnp.float64),
        problem=jnp.asarray((3.0,), dtype=jnp.float64),
    )
    prepare_calls = 0
    run_calls: list[int] = []
    initial = SimpleNamespace(
        scaled_penalty_value=jnp.asarray(10.0),
        scaled_constraint_infinity_norm=jnp.asarray(2.0),
    )
    final = SimpleNamespace(
        scaled_penalty_value=jnp.asarray(8.0),
        scaled_constraint_infinity_norm=jnp.asarray(1.5),
    )
    optimizer = SimpleNamespace(
        iterations=jnp.asarray(maximum_iterations, dtype=jnp.int32),
        function_evaluations=jnp.asarray(maximum_iterations + 4, dtype=jnp.int32),
        gradient_evaluations=jnp.asarray(maximum_iterations + 4, dtype=jnp.int32),
    )
    expected_result = SimpleNamespace(
        initial_diagnostics=initial,
        final_diagnostics=final,
        initial_stationarity_infinity_norm=jnp.asarray(4.0),
        final_stationarity_infinity_norm=jnp.asarray(3.0),
        optimizer=optimizer,
        made_progress=jnp.asarray(True),
        all_finite=jnp.asarray(True),
        nonfinite_evaluation_count=jnp.asarray(0, dtype=jnp.int32),
    )

    class Prepared:
        initial_optimizer_coordinates = jnp.asarray((0.0, 0.0), dtype=jnp.float64)
        initial_merit = jnp.asarray(10.0)
        initial_gradient = jnp.asarray((4.0, 1.0))
        initial_diagnostics = initial

        def run(self, *, maximum_iterations: int):
            run_calls.append(maximum_iterations)
            return expected_result

    def prepare(*_args: object) -> Prepared:
        nonlocal prepare_calls
        prepare_calls += 1
        return Prepared()

    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: [cpu_device])
    monkeypatch.setattr(
        runner,
        "_deterministic_cfs_p0_changed_state",
        lambda z, _problem: z + jnp.asarray(1.0e-3, dtype=z.dtype),
    )
    monkeypatch.setattr(runner, "prepare_cfs_p0", prepare)

    numerical, timing, transfers = runner.run_cfs_p0_canary_probe(
        bootstrap,
        maximum_iterations=maximum_iterations,
    )

    assert prepare_calls == 1
    assert run_calls == [maximum_iterations]
    assert numerical == {
        "all_finite": True,
        "final_M0": 8.0,
        "final_scaled_feasibility_inf": 1.5,
        "final_stationarity_inf": 3.0,
        "function_evaluations": maximum_iterations + 4,
        "gradient_evaluations": maximum_iterations + 4,
        "initial_M0": 10.0,
        "initial_scaled_feasibility_inf": 2.0,
        "initial_stationarity_inf": 4.0,
        "iterations": maximum_iterations,
        "made_progress": True,
        "nonfinite_evaluation_count": 0,
    }
    assert timing["preparation_and_compile_ns"] >= 0
    assert timing[execution_key] >= 0
    assert transfers["hot_d2h_calls"] == 0
    assert transfers["final_d2h_calls"] == 1


@pytest.mark.parametrize(
    ("made_progress", "iterations", "function_evaluations", "message"),
    (
        (False, 10, 11, "progress gate"),
        (True, 9, 11, "iteration budget"),
        (True, 10, 5, "evaluation counters"),
        (True, 10, 15001, "evaluation counters"),
    ),
)
def test_cfs_p0_probe_fails_closed_on_progress_or_budget_integrity(
    monkeypatch: pytest.MonkeyPatch,
    made_progress: bool,
    iterations: int,
    function_evaluations: int,
    message: str,
) -> None:
    cpu_device = jax.devices("cpu")[0]
    bootstrap = SimpleNamespace(
        z0=jnp.asarray((1.0,), dtype=jnp.float64),
        problem=jnp.asarray((2.0,), dtype=jnp.float64),
    )
    diagnostics = SimpleNamespace(
        scaled_penalty_value=jnp.asarray(10.0),
        scaled_constraint_infinity_norm=jnp.asarray(2.0),
    )
    optimizer = SimpleNamespace(
        iterations=jnp.asarray(iterations),
        function_evaluations=jnp.asarray(function_evaluations),
        gradient_evaluations=jnp.asarray(function_evaluations),
    )
    result = SimpleNamespace(
        initial_diagnostics=diagnostics,
        final_diagnostics=diagnostics,
        initial_stationarity_infinity_norm=jnp.asarray(4.0),
        final_stationarity_infinity_norm=jnp.asarray(4.0),
        optimizer=optimizer,
        made_progress=jnp.asarray(made_progress),
        all_finite=jnp.asarray(True),
        nonfinite_evaluation_count=jnp.asarray(0),
    )
    prepared = SimpleNamespace(
        initial_optimizer_coordinates=jnp.asarray((0.0,)),
        initial_merit=jnp.asarray(10.0),
        initial_gradient=jnp.asarray((4.0,)),
        initial_diagnostics=diagnostics,
        run=lambda **_kwargs: result,
    )
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: [cpu_device])
    monkeypatch.setattr(
        runner,
        "_deterministic_cfs_p0_changed_state",
        lambda z, _problem: z + jnp.asarray(1.0e-3, dtype=z.dtype),
    )
    monkeypatch.setattr(runner, "prepare_cfs_p0", lambda *_args: prepared)

    with pytest.raises(ValueError, match=message):
        runner.run_cfs_p0_canary_probe(bootstrap, maximum_iterations=10)


def test_cfs_p0_changed_state_has_frozen_centered_optimizer_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = object()
    z = jnp.zeros((716,), dtype=jnp.float64)
    monkeypatch.setattr(
        runner,
        "fullspace_scaling_from_bootstrap",
        lambda _z, _problem: scaling,
    )

    def physical_coordinates(
        optimizer_coordinates: jax.Array,
        received_scaling: object,
    ) -> jax.Array:
        assert received_scaling is scaling
        return optimizer_coordinates

    monkeypatch.setattr(
        runner,
        "fullspace_physical_coordinates",
        physical_coordinates,
    )

    changed = runner._deterministic_cfs_p0_changed_state(z, object())

    np.testing.assert_allclose(
        jnp.linalg.norm(changed),
        runner.CFS_P0_CHANGED_OPTIMIZER_NORM,
        rtol=0.0,
        atol=2.0e-18,
    )


def test_cfs_p0_publisher_rejects_non_p0_route_before_probe(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    runs = campaign / "runs"
    runs.mkdir()
    runtime, runtime_ref = _runtime(campaign)

    with pytest.raises(ValueError, match="requires CFS-P0"):
        runner.publish_cfs_p0_canary_evidence(
            runs / "canary.json",
            request=_canary_request(FullSpaceRoute.CFS_AL1),
            campaign_root=campaign,
            snapshot_root=snapshot,
            runtime_evidence=runtime,
            runtime_evidence_ref=runtime_ref,
            bootstrap_artifact_ref=ArtifactRef(
                "missing.json", "a" * 64, 1, BOOTSTRAP_SCHEMA_VERSION
            ),
            bootstrap=SimpleNamespace(),
        )


def test_cfs_p0_100_step_publisher_uses_distinct_immutable_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    runs = campaign / "runs"
    runs.mkdir()
    runtime, runtime_ref = _runtime(campaign)
    bootstrap_bytes = canonical_json_bytes({"schema_version": BOOTSTRAP_SCHEMA_VERSION})
    bootstrap_path = campaign / "bootstrap.json"
    bootstrap_path.write_bytes(bootstrap_bytes)
    bootstrap_ref = ArtifactRef(
        "bootstrap.json",
        hashlib.sha256(bootstrap_bytes).hexdigest(),
        len(bootstrap_bytes),
        BOOTSTRAP_SCHEMA_VERSION,
    )
    monkeypatch.setattr(
        runner,
        "validate_bootstrap_artifact",
        lambda *_args, **_kwargs: {"runtime_evidence": asdict(runtime_ref)},
    )
    observed_iterations: list[int] = []

    def probe(_bootstrap: object, *, maximum_iterations: int):
        observed_iterations.append(maximum_iterations)
        return (
            {"iterations": maximum_iterations, "made_progress": True},
            {"hundred_step_execution_ns": 3},
            {"hot_d2h_calls": 0, "final_d2h_calls": 1},
        )

    monkeypatch.setattr(runner, "run_cfs_p0_canary_probe", probe)
    output = runs / "cfs-p0-canary-100.json"
    reference = runner.publish_cfs_p0_canary_evidence(
        output,
        request=_canary_request(steps=100),
        campaign_root=campaign,
        snapshot_root=snapshot,
        runtime_evidence=runtime,
        runtime_evidence_ref=runtime_ref,
        bootstrap_artifact_ref=bootstrap_ref,
        bootstrap=SimpleNamespace(),
    )

    document = load_canonical_json_bytes(output.read_bytes())
    assert isinstance(document, dict)
    assert observed_iterations == [100]
    assert reference.relative_path == runner.CFS_P0_CANARY_100_RELATIVE_PATH
    assert reference.schema_version == runner.CFS_P0_CANARY_100_SCHEMA_VERSION
    assert document["schema_version"] == runner.CFS_P0_CANARY_100_SCHEMA_VERSION
    assert output.stat().st_mode & 0o222 == 0
