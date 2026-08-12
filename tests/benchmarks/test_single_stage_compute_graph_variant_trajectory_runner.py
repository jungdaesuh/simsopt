from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import jax.numpy as jnp
import pytest
from benchmarks.single_stage_compute_graph_c0_runner import CommandResult
from benchmarks.single_stage_compute_graph_canary_runner import CanarySpec
from benchmarks.single_stage_compute_graph_isolated_launch import SnapshotModuleLaunch
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_snapshot import ManifestEntry
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    write_raw_trajectory_document,
)
from benchmarks.single_stage_compute_graph_variant_trajectory import (
    _c0_document,
    _ReplayInputs,
)
from benchmarks.single_stage_compute_graph_variant_trajectory_runner import (
    PRODUCER_MODULE,
    VariantTrajectoryLaunch,
    VariantTrajectoryRunnerError,
    _diagnostic_environment,
    launch_variant_trajectory,
    validate_variant_trajectory_launch,
)


def test_diagnostic_environment_is_derived_only_from_runtime_contract(
    tmp_path: Path,
) -> None:
    base = _spec(tmp_path)
    contract = {
        "runtime": {"jax_backend": "gpu"},
        "static_environment": {"PATH": "/bound/bin"},
        "route_environment": {},
        "policies": {"quadrature_block_sizes": [1]},
        "expected_runtime_identity_sha256": base.runtime_identity_sha256,
    }
    spec = replace(
        base,
        runtime_contract_json=json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ),
    )

    environment = _diagnostic_environment(spec)

    assert environment["PATH"] == "/bound/bin"
    assert environment["JAX_COMPILATION_CACHE_DIR"] == str(spec.cache_directory)
    assert environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY"] == (
        spec.runtime_identity_sha256
    )
    child_contract = json.loads(
        environment["SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT"]
    )
    assert child_contract["route_environment"] == {
        "JAX_COMPILATION_CACHE_DIR": str(spec.cache_directory)
    }


def _residual(values):
    return jnp.asarray(
        [
            values[0] ** 2 + 0.5 * values[1] - 2.0,
            -0.25 * values[0] + values[1] ** 2 - 3.0,
        ],
        dtype=values.dtype,
    )


def _spec(tmp_path: Path) -> CanarySpec:
    native_path = tmp_path / "native.json"
    native_path.write_bytes(
        canonical_json_bytes(
            {
                "identity": {"input_bundle_sha256": "a" * 64},
            }
        )
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(canonical_json_bytes({"test": "spec"}))
    return CanarySpec(
        variant="C1",
        solver_graph_sha256="1" * 64,
        source_state_sha256="2" * 64,
        specimen_sha256="3" * 64,
        candidate_file_sha256="4" * 64,
        parameter_sha256="5" * 64,
        device_identity_sha256="6" * 64,
        gpu_uuid="GPU-test",
        c0_gate_checkpoint_sha256="7" * 64,
        c0_warm_checkpoint_sha256="8" * 64,
        native_reference_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
        runtime_identity_sha256="9" * 64,
        input_root=tmp_path / "inputs",
        candidate_path=tmp_path / "candidate.npy",
        native_reference_path=native_path,
        snapshot_root=tmp_path / "snapshot",
        interpreter_path=Path(sys.executable),
        cache_directory=tmp_path / "cache",
        output_root=tmp_path / "output",
        snapshot_manifest_sha256="b" * 64,
        snapshot_publication_sha256="c" * 64,
        import_attestation_sha256="d" * 64,
        qualification_sha256="e" * 64,
        device_probe_sha256="f" * 64,
        runtime_provenance_sha256="0" * 64,
    )


def _raw_document() -> dict[str, object]:
    document, _ = _c0_document(
        _ReplayInputs(
            residual_fn=_residual,
            initial_state=jnp.asarray([1.25, 1.75], dtype=jnp.float64),
            maxiter=8,
            tolerance=1.0e-12,
        ),
        parameter_sha256="5" * 64,
        specimen_sha256="3" * 64,
        input_bundle_sha256="a" * 64,
        solver_graph_sha256="1" * 64,
        source_sha256="2" * 64,
    )
    return document


def _child(spec: CanarySpec, output: Path) -> SnapshotModuleLaunch:
    return SnapshotModuleLaunch(
        argv=(
            str(spec.interpreter_path),
            "-m",
            PRODUCER_MODULE,
            "--output",
            str(output),
        ),
        cwd=spec.snapshot_root,
        environment=MappingProxyType({"JAX_ENABLE_X64": "true"}),
    )


def test_launch_writes_fresh_process_receipt_bound_to_raw_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    raw_path = tmp_path / "artifacts" / "c0.raw.json"
    receipt_path = tmp_path / "artifacts" / "c0.launch.json"
    spec_path = tmp_path / "spec.json"
    launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=spec_path,
        lane="C0",
        output_path=raw_path,
        receipt_path=receipt_path,
    )
    child = _child(spec, raw_path)
    producer_entry = ManifestEntry(
        role="benchmark",
        relative_path="benchmarks/single_stage_compute_graph_variant_trajectory.py",
        size_bytes=123,
        sha256="c" * 64,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner._producer_manifest_entry",
        lambda _spec: producer_entry,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner._validate_launch_spec_binding",
        lambda _launch: None,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner.build_variant_trajectory_launch",
        lambda _launch: child,
    )

    def executor(argv, environment, cwd, timeout):
        assert tuple(argv) == child.argv
        assert environment == child.environment
        assert cwd == child.cwd
        assert timeout == 900.0
        write_raw_trajectory_document(raw_path, _raw_document())
        return CommandResult(
            returncode=0,
            stdout="",
            stderr="",
            elapsed_ns=1,
        )

    document = launch_variant_trajectory(
        launch,
        artifact_root=tmp_path / "artifacts",
        executor=executor,
    )

    assert document["state"] == "PRODUCED"
    assert document["promotion_timing"] is False
    assert document["producer"]["sha256"] == "c" * 64
    assert document["raw"]["relative_path"] == "c0.raw.json"
    assert document["launch"]["completion"] == {
        "returncode": 0,
        "timed_out": False,
        "elapsed_ns": 1,
        "stdout": "",
        "stderr": "",
    }
    assert receipt_path.read_bytes() == canonical_json_bytes(document)
    assert (
        validate_variant_trajectory_launch(
            receipt_path,
            launch,
            artifact_root=tmp_path / "artifacts",
        )
        == document
    )

    completion_tampers = (
        ("returncode", 1),
        ("timed_out", True),
        ("elapsed_ns", 0),
        ("stdout", "unexpected"),
        ("stderr", "unexpected"),
    )
    for field, value in completion_tampers:
        tampered = json.loads(json.dumps(document))
        tampered["launch"]["completion"][field] = value
        receipt_path.write_bytes(canonical_json_bytes(tampered))
        with pytest.raises(
            VariantTrajectoryRunnerError,
            match="completion record",
        ):
            validate_variant_trajectory_launch(
                receipt_path,
                launch,
                artifact_root=tmp_path / "artifacts",
            )


def test_launch_rejects_preexisting_raw_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    raw_path = tmp_path / "raw.json"
    raw_path.write_text("occupied", encoding="utf-8")
    launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=tmp_path / "spec.json",
        lane="C0",
        output_path=raw_path,
        receipt_path=tmp_path / "receipt.json",
    )

    with pytest.raises(VariantTrajectoryRunnerError, match="must not exist"):
        launch_variant_trajectory(launch, artifact_root=tmp_path)


def test_launch_rejects_output_outside_artifact_root_before_execution(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=tmp_path / "spec.json",
        lane="C0",
        output_path=tmp_path / "outside.json",
        receipt_path=tmp_path / "artifacts" / "receipt.json",
    )

    with pytest.raises(VariantTrajectoryRunnerError, match="inside artifact root"):
        launch_variant_trajectory(launch, artifact_root=tmp_path / "artifacts")

    assert not launch.receipt_path.parent.exists()


def test_launch_rejects_duplicate_destinations_before_execution(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    output_path = tmp_path / "artifacts" / "shared.json"
    launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=tmp_path / "spec.json",
        lane="C0",
        output_path=output_path,
        receipt_path=output_path,
    )

    with pytest.raises(VariantTrajectoryRunnerError, match="must be distinct"):
        launch_variant_trajectory(launch, artifact_root=tmp_path / "artifacts")

    assert not output_path.parent.exists()


def test_receipt_validation_detects_raw_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    raw_path = tmp_path / "artifacts" / "raw.json"
    receipt_path = tmp_path / "artifacts" / "receipt.json"
    raw_path.parent.mkdir()
    write_raw_trajectory_document(raw_path, _raw_document())
    launch = VariantTrajectoryLaunch(
        spec=spec,
        spec_path=tmp_path / "spec.json",
        lane="C0",
        output_path=raw_path,
        receipt_path=receipt_path,
    )
    child = _child(spec, raw_path)
    producer_entry = ManifestEntry(
        role="benchmark",
        relative_path="benchmarks/single_stage_compute_graph_variant_trajectory.py",
        size_bytes=123,
        sha256="c" * 64,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner._producer_manifest_entry",
        lambda _spec: producer_entry,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner._validate_launch_spec_binding",
        lambda _launch: None,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_variant_trajectory_runner.build_variant_trajectory_launch",
        lambda _launch: child,
    )
    receipt_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_id": "tampered",
                "launch": {
                    "completion": {
                        "returncode": 0,
                        "timed_out": False,
                        "elapsed_ns": 1,
                        "stdout": "",
                        "stderr": "",
                    }
                },
            }
        )
    )

    with pytest.raises(VariantTrajectoryRunnerError, match="differs"):
        validate_variant_trajectory_launch(
            receipt_path,
            launch,
            artifact_root=tmp_path / "artifacts",
        )
