from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import single_stage_compute_graph_complete_path as complete_path
from benchmarks.single_stage_compute_graph_complete_path import (
    CompletePathBinding,
    CompletePathEvidenceError,
    FaithfulLever,
    GapBudgetPolicyInput,
    PhaseReductionAssumption,
    ProtocolSample,
    binding_from_phase0_checkpoints,
    build_complete_path_document,
    build_complete_path_lane_environment,
    build_complete_path_plan,
    build_gap_budget_inputs_artifact,
    build_lane_snapshot_provenance_document,
    build_staged_gap_budget_timing_input,
    collect_complete_path_evidence,
    validate_gap_budget_inputs_artifact,
    write_lane_snapshot_provenance,
    write_lane_snapshot_provenance_set,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    IMPORT_ATTESTATION_SCHEMA_ID,
    MANIFEST_FILENAME,
    SOURCE_MANIFEST_SCHEMA_ID,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    canonical_json_bytes as snapshot_canonical_json_bytes,
)
from examples.jax.parity.artifacts import ArtifactValidationError

_HASH = "a" * 64


def _binding() -> CompletePathBinding:
    return CompletePathBinding(
        specimen_sha256=_HASH,
        candidate_sha256="b" * 64,
        source_sha256="c" * 64,
        runtime_identity_sha256="9" * 64,
        native_reference_sha256="d" * 64,
        gate_checkpoint_sha256="e" * 64,
        warm_checkpoint_sha256="f" * 64,
        warm_p50_ns=125.0,
        lane_id="rtx5090",
        gpu_uuid="GPU-1234",
    )


def _samples() -> list[ProtocolSample]:
    samples: list[ProtocolSample] = []
    for run in build_complete_path_plan():
        profile_id = run.profile_id
        assert profile_id in complete_path.PROFILE_IDS
        timing = 100 + 10 * tuple(complete_path.PROFILE_IDS).index(profile_id)
        parity_rows = (
            ()
            if profile_id == "native_cpu"
            else tuple(
                complete_path.measurement_runner.ParityRow(
                    observable=observable,
                    native_value=1.0,
                    lane_value=1.0,
                    tolerance=1.0e-12,
                )
                for observable, _ in (
                    complete_path.measurement_runner._SINGLE_STAGE_PARITY_OBSERVABLES
                )
            )
        )
        samples.append(
            ProtocolSample(
                profile_id=profile_id,
                phase=run.phase,
                sample_index=run.sample_index,
                optimization_wall_ns=timing,
                subprocess_wall_ns=timing + 50,
                driver=complete_path.EXPECTED_DRIVERS[profile_id],
                backend_mode=complete_path.EXPECTED_BACKENDS[profile_id],
                input_fingerprint="2" * 64,
                configuration_fingerprint="3" * 64,
                effective_construction_fingerprint="4" * 64,
                input_bundle_sha256="1" * 64,
                source_sha256="c" * 64,
                runtime_identity_sha256="9" * 64,
                nit=2,
                nfev=3,
                njev=3,
                endpoint_certificate={
                    "success": True,
                    "initial_stationary": False,
                    "terminal_stationary": False,
                    "constraints_satisfied": True,
                    "outer_status": 1,
                },
                parity_rows=parity_rows,
                snapshot_source_manifest_sha256="7" * 64,
                snapshot_import_attestation_sha256="8" * 64,
                snapshot_lane_identity_sha256="6" * 64,
                provenance={
                    "repository_commit": "commit",
                    "executed_sources": [{"path": "source.py", "sha256": _HASH}],
                },
            )
        )
    return samples


def _snapshot_evidence(tmp_path: Path) -> dict[str, Path]:
    snapshot = tmp_path / "snapshot"
    entries: list[dict[str, object]] = []
    definitions = (
        ("benchmark", "benchmarks/run.py", b"benchmark\n"),
        ("configuration", "config/input.json", b"{}\n"),
        ("execution_source", "src/simsopt.py", b"simsopt\n"),
        ("execution_source", "src/simsopt_jax.py", b"jax\n"),
        ("execution_source", "src/simsopt_jax_adapters.py", b"adapters\n"),
        ("native_extension", "src/simsoptpp.so", b"native\n"),
        ("test", "tests/test_run.py", b"test\n"),
    )
    for role, relative_path, payload in definitions:
        path = snapshot / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(
            {
                "role": role,
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    entries.sort(key=lambda entry: str(entry["relative_path"]))
    manifest = {"schema_id": SOURCE_MANIFEST_SCHEMA_ID, "entries": entries}
    manifest_path = snapshot / MANIFEST_FILENAME
    manifest_bytes = snapshot_canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    by_path = {str(entry["relative_path"]): entry for entry in entries}
    module_paths = {
        "simsopt": "src/simsopt.py",
        "simsopt_jax": "src/simsopt_jax.py",
        "simsopt_jax_adapters": "src/simsopt_jax_adapters.py",
        "simsoptpp": "src/simsoptpp.so",
    }
    attestation = {
        "schema_id": IMPORT_ATTESTATION_SCHEMA_ID,
        "state": "pass",
        "snapshot_manifest_sha256": manifest_sha256,
        "interpreter_path": sys.executable,
        "python_version": sys.version.split()[0],
        "bindings": [
            {"module": module, **by_path[path]} for module, path in module_paths.items()
        ],
    }
    for binding in attestation["bindings"]:
        binding.pop("role")
    attestation_path = tmp_path / "import-attestation.json"
    attestation_path.write_bytes(snapshot_canonical_json_bytes(attestation))
    worktree = {
        "repository_commit": "a" * 40,
        "git_status_short": [],
        "tracked_diff_sha256": "0" * 64,
        "untracked_manifest_sha256": "1" * 64,
        "source_state_sha256": _binding().source_sha256,
    }
    publication = {
        "schema_id": "single-stage-compute-graph-snapshot-publication-v1",
        "repository_root": str(tmp_path),
        "snapshot_root": str(snapshot.absolute()),
        "snapshot_manifest_sha256": manifest_sha256,
        "cross_host_source_sha256": "2" * 64,
        "native_extension": by_path["src/simsoptpp.so"],
        "worktree": worktree,
    }
    publication_path = tmp_path / "publication.json"
    publication_path.write_bytes(snapshot_canonical_json_bytes(publication))
    runtime = {
        "repository_commit": "a" * 40,
        "source_state_sha256": _binding().source_sha256,
        "interpreter_path": sys.executable,
        "runtime": {
            "python_version": sys.version.split()[0],
            "jax_version": "0.7.0",
            "jaxlib_version": "0.7.0",
        },
        "allocation": {"gpu_uuid": _binding().gpu_uuid},
        "environment": {},
    }
    runtime_path = tmp_path / "runtime-provenance.json"
    runtime_path.write_bytes(snapshot_canonical_json_bytes(runtime))
    runner_spec = {
        "schema_id": "single-stage-compute-graph-c0-runner-spec-v3",
        "lane_id": _binding().lane_id,
        "warm_sample_count": 7,
        "output_root": str(tmp_path / "output"),
        "input_root": str(snapshot / "input"),
        "candidate_path": str(snapshot / "candidate.npy"),
        "native_reference_path": str(tmp_path / "native.json"),
        "provenance": runtime,
        "receipt_template": {},
    }
    runner_path = tmp_path / "runner-spec.json"
    runner_path.write_bytes(snapshot_canonical_json_bytes(runner_spec))
    device_probe = {
        "schema_id": "single-stage-compute-graph-device-probe-v1",
        "lane_id": _binding().lane_id,
        "source_state_sha256": _binding().source_sha256,
        "runtime_identity_sha256": _binding().runtime_identity_sha256,
        "qualification_sha256": "5" * 64,
        "gpu": {
            "uuid": _binding().gpu_uuid,
            "name": "test GPU",
            "memory_bytes": 1,
        },
        "native_binary": {
            "path": str(snapshot / "src/simsoptpp.so"),
            "sha256": by_path["src/simsoptpp.so"]["sha256"],
        },
    }
    device_path = tmp_path / "device-probe.json"
    device_path.write_bytes(snapshot_canonical_json_bytes(device_probe))
    return {
        "snapshot_manifest_path": manifest_path,
        "import_attestation_path": attestation_path,
        "snapshot_publication_path": publication_path,
        "runner_spec_path": runner_path,
        "runtime_provenance_path": runtime_path,
        "device_probe_path": device_path,
    }


def test_plan_contains_exactly_one_fresh_run_per_matched_lane() -> None:
    plan = build_complete_path_plan()

    assert len(plan) == 3
    assert tuple(run.profile_id for run in plan) == complete_path.PROFILE_IDS
    assert all(run.phase == "cold" for run in plan)
    assert all(run.sample_index is None for run in plan)


def test_profile_command_accepts_snapshot_provenance_without_git_root(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    provenance_path = snapshot_root / "lane-provenance.json"
    command = complete_path.measurement_runner.build_profile_command(
        python_executable="/snapshot/python",
        case_id=complete_path.measurement_runner._SINGLE_STAGE_CASE_ID,
        profile_id="jax_gpu_fast",
        input_bundle_path=snapshot_root / "input_bundle.json",
        result_directory=snapshot_root / "result",
        scale="native_default",
        immutable_snapshot_provenance_path=provenance_path,
    )

    assert command[-2:] == (
        "--immutable-snapshot-provenance",
        str(provenance_path),
    )


def test_lane_snapshot_provenance_builder_hashes_verified_artifact_bytes(
    tmp_path: Path,
) -> None:
    evidence = _snapshot_evidence(tmp_path)

    document = build_lane_snapshot_provenance_document(
        binding=_binding(),
        profile_id="jax_gpu_fast",
        **evidence,
    )

    assert document["runtime_identity_sha256"] == "9" * 64
    assert document["source_sha256"] == "c" * 64
    assert (
        document["evidence"]["manifest"]["sha256"]
        == hashlib.sha256(evidence["snapshot_manifest_path"].read_bytes()).hexdigest()
    )
    assert (
        document["evidence"]["import_attestation"]["sha256"]
        == hashlib.sha256(evidence["import_attestation_path"].read_bytes()).hexdigest()
    )

    output = write_lane_snapshot_provenance(
        tmp_path / "evidence",
        "lanes/c0.json",
        binding=_binding(),
        profile_id="jax_gpu_fast",
        **evidence,
    )
    assert output.read_bytes() == complete_path.canonical_json_bytes(document)
    with pytest.raises(ArtifactValidationError, match="already exists"):
        write_lane_snapshot_provenance(
            tmp_path / "evidence",
            "lanes/c0.json",
            binding=_binding(),
            profile_id="jax_gpu_fast",
            **evidence,
        )


def test_lane_snapshot_provenance_rejects_drifted_source_or_attestation_bytes(
    tmp_path: Path,
) -> None:
    evidence = _snapshot_evidence(tmp_path)
    manifest_path = evidence["snapshot_manifest_path"]
    source_path = manifest_path.parent / "src/simsopt.py"
    source_path.write_bytes(b"drifted\n")

    with pytest.raises(CompletePathEvidenceError, match="manifest is invalid"):
        build_lane_snapshot_provenance_document(
            binding=_binding(),
            profile_id="jax_gpu_fast",
            **evidence,
        )


def test_snapshot_provenance_set_rejects_cross_lane_swap_and_output_collision(
    tmp_path: Path,
) -> None:
    evidence = _snapshot_evidence(tmp_path)
    output_root = tmp_path / "lane-identities"
    paths = write_lane_snapshot_provenance_set(
        output_root,
        binding=_binding(),
        **evidence,
    )

    with pytest.raises(CompletePathEvidenceError, match="profile mismatch"):
        complete_path._snapshot_provenance_identity(
            paths["jax_gpu_fast"],
            profile_id="jax_gpu_optax",
            runtime_identity_sha256=_binding().runtime_identity_sha256,
            source_sha256=_binding().source_sha256,
            gpu_uuid=_binding().gpu_uuid,
            snapshot_root=evidence["snapshot_manifest_path"].parent,
        )
    with pytest.raises(CompletePathEvidenceError, match="already exists"):
        write_lane_snapshot_provenance_set(
            output_root,
            binding=_binding(),
            **evidence,
        )


def test_snapshot_identity_rejects_runtime_device_and_referenced_byte_tampering(
    tmp_path: Path,
) -> None:
    evidence = _snapshot_evidence(tmp_path)
    device_path = evidence["device_probe_path"]
    device = json.loads(device_path.read_text(encoding="utf-8"))
    device["gpu"]["uuid"] = "GPU-wrong"
    device_path.write_bytes(snapshot_canonical_json_bytes(device))
    with pytest.raises(CompletePathEvidenceError, match="device probe binding"):
        build_lane_snapshot_provenance_document(
            binding=_binding(), profile_id="jax_gpu_fast", **evidence
        )

    evidence = _snapshot_evidence(tmp_path / "tamper")
    output = write_lane_snapshot_provenance(
        tmp_path / "output",
        "c0.json",
        binding=_binding(),
        profile_id="jax_gpu_fast",
        **evidence,
    )
    runtime_path = evidence["runtime_provenance_path"]
    runtime_path.write_bytes(runtime_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="runtime_provenance bytes changed"):
        complete_path.load_snapshot_lane_identity(output)


def test_snapshot_identity_document_has_exact_static_schema(tmp_path: Path) -> None:
    evidence = _snapshot_evidence(tmp_path)
    document = build_lane_snapshot_provenance_document(
        binding=_binding(), profile_id="native_cpu", **evidence
    )

    assert set(document) == {
        "schema_id",
        "profile_id",
        "lane",
        "backend_mode",
        "driver",
        "execution_platform",
        "runtime_identity_sha256",
        "source_sha256",
        "gpu_uuid",
        "snapshot_root",
        "static_environment",
        "evidence",
    }
    assert set(document["evidence"]) == {
        "publication",
        "manifest",
        "import_attestation",
        "runner_spec",
        "runtime_provenance",
        "device_probe",
    }
    assert "provenance" not in document
    assert document["static_environment"]["JAX_PLATFORMS"] == "cpu"
    assert "CUDA_VISIBLE_DEVICES" not in document["static_environment"]

    evidence = _snapshot_evidence(tmp_path / "second")
    attestation_path = evidence["import_attestation_path"]
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["snapshot_manifest_sha256"] = "f" * 64
    attestation_path.write_bytes(snapshot_canonical_json_bytes(attestation))
    with pytest.raises(CompletePathEvidenceError, match="does not bind"):
        build_lane_snapshot_provenance_document(
            binding=_binding(),
            profile_id="jax_gpu_fast",
            **evidence,
        )


def test_document_keeps_complete_path_and_runner_canary_timings_separate() -> None:
    document = build_complete_path_document(_binding(), _samples())

    assert document["identity"]["warm_p50_ns"] == 125.0
    assert document["matched_complete_path_reference_timings_ns"] == {
        "native": 100.0,
        "c0": 110.0,
        "optax": 120.0,
    }
    assert build_staged_gap_budget_timing_input(document) == {
        "matched_complete_path_reference_timings_ns": {
            "native": 100,
            "c0": 110,
            "optax": 120,
        },
        "c0_complete_path_value_and_gradient_evaluation_count": 3,
        "c0_complete_path_value_and_gradient_evaluation_count_semantics": (
            complete_path.GAP_BUDGET_COUNT_SEMANTICS
        ),
    }
    assert document["staged_gap_budget_timing_input"] == (
        build_staged_gap_budget_timing_input(document)
    )
    assert document["protocol"]["problem_construction_excluded"] is True
    assert document["protocol"]["closed_r5_receipt_extended"] is False
    assert document["lanes"]["native"]["gpu_uuid"] is None
    assert document["lanes"]["c0"]["gpu_uuid"] == "GPU-1234"
    assert document["lanes"]["c0"]["optimizer_counts"] == {
        "nit": 2,
        "nfev": 3,
        "njev": 3,
    }
    assert len(document["lanes"]["c0"]["parity_rows"]) == 5
    assert document["protocol"]["sample_count_per_lane"] == 1
    assert document["protocol"]["timing_claim"] == "rough_non_statistical_single_run"
    assert document["lanes"]["optax"]["raw_optimization_wall_ns"] == [120]
    assert (
        document["lanes"]["optax"]["statistical_summary"]
        == "not_produced_single_sample"
    )


def test_document_rejects_identity_drift_and_wrong_driver() -> None:
    samples = _samples()
    samples[1] = replace(samples[1], input_fingerprint="0" * 64)
    with pytest.raises(CompletePathEvidenceError, match="identity matched"):
        build_complete_path_document(_binding(), samples)

    samples = _samples()
    samples[0] = replace(samples[0], driver="wrong")
    with pytest.raises(CompletePathEvidenceError, match="driver"):
        build_complete_path_document(_binding(), samples)


def test_staged_gap_budget_builder_rejects_transcribed_timing_drift() -> None:
    document = build_complete_path_document(_binding(), _samples())
    staged = document["staged_gap_budget_timing_input"]
    assert isinstance(staged, dict)
    staged["matched_complete_path_reference_timings_ns"] = {
        "native": 100,
        "c0": 111,
        "optax": 120,
    }

    with pytest.raises(
        CompletePathEvidenceError,
        match="disagrees with authoritative timings",
    ):
        build_staged_gap_budget_timing_input(document)


def test_gap_budget_artifact_is_fully_wrapped_and_rejects_timing_drift() -> None:
    complete_document = build_complete_path_document(_binding(), _samples())
    policy = GapBudgetPolicyInput(
        phase_reduction_assumptions={
            "newton.residual_jvp": PhaseReductionAssumption(0.1, 0.2, "disjoint")
        },
        unattributed_conservative_reduction=0.0,
        unattributed_optimistic_reduction=0.1,
        faithful_levers=(FaithfulLever("dense_newton", "bounded", "6" * 64),),
    )
    artifact = build_gap_budget_inputs_artifact(complete_document, policy)

    payload = validate_gap_budget_inputs_artifact(artifact, complete_document)
    assert artifact["identity"]["source_sha256"] == "c" * 64
    assert artifact["identity"]["runtime_identity_sha256"] == "9" * 64
    assert payload["c0_complete_path_value_and_gradient_evaluation_count"] == 3

    payload["matched_complete_path_reference_timings_ns"] = {
        "native": 100,
        "c0": 111,
        "optax": 120,
    }
    with pytest.raises(CompletePathEvidenceError, match="drifted"):
        validate_gap_budget_inputs_artifact(artifact, complete_document)


def test_binding_rejects_invalid_checkpoint_identity() -> None:
    with pytest.raises(CompletePathEvidenceError, match="gate_checkpoint_sha256"):
        replace(_binding(), gate_checkpoint_sha256="not-a-hash")


def test_binding_is_recomputed_from_gate_warm_and_native_bytes(tmp_path: Path) -> None:
    native_path = tmp_path / "native.json"
    native_path.write_bytes(b'{"native":true}\n')
    native_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    gate = {
        "state": "PASSED",
        "lane_id": "rtx5090",
        "gpu_uuid": "GPU-1234",
        "specimen_sha256": _HASH,
        "parameter_sha256": "b" * 64,
        "source_state_sha256": "c" * 64,
        "runtime_identity_sha256": "9" * 64,
        "native_reference_sha256": native_sha256,
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    gate_sha256 = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    warm = {
        "state": "COMPLETE",
        "gate_checkpoint_sha256": gate_sha256,
        "lane_id": gate["lane_id"],
        "gpu_uuid": gate["gpu_uuid"],
        "specimen_sha256": gate["specimen_sha256"],
        "parameter_sha256": gate["parameter_sha256"],
        "source_state_sha256": gate["source_state_sha256"],
        "runtime_identity_sha256": gate["runtime_identity_sha256"],
        "warm_measurement": {"p50_ns": 125.0},
    }
    warm_path = tmp_path / "warm.json"
    warm_path.write_text(json.dumps(warm, sort_keys=True), encoding="utf-8")

    binding = binding_from_phase0_checkpoints(
        gate_path,
        warm_path,
        native_path,
    )

    assert binding.native_reference_sha256 == native_sha256
    assert binding.gate_checkpoint_sha256 == gate_sha256
    assert binding.warm_p50_ns == 125.0


def test_collector_uses_injected_protocol_executor_without_launching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "specimen" / "input_bundle"
    input_root.mkdir(parents=True)
    input_path = input_root / "input_bundle.json"
    input_path.write_text("{}", encoding="utf-8")
    candidate = np.linspace(0.0, 1.0, 461, dtype=np.float64)
    candidate_path = tmp_path / "specimen" / "changed_state_candidate.npy"
    np.save(candidate_path, candidate)
    candidate_sha256 = hashlib.sha256(
        np.ascontiguousarray(candidate, dtype=np.dtype("<f8")).tobytes(order="C")
    ).hexdigest()
    binding = replace(_binding(), candidate_sha256=candidate_sha256)
    specimen_path = tmp_path / "specimen" / "specimen.json"
    specimen_path.write_text(
        json.dumps(
            {
                "specimen_sha256": binding.specimen_sha256,
                "specimen": {
                    "input_bundle_sha256": hashlib.sha256(
                        input_path.read_bytes()
                    ).hexdigest(),
                    "parameter_sha256": candidate_sha256,
                },
                "input_bundle": {"relative_path": "input_bundle"},
                "candidate": {
                    "relative_path": "changed_state_candidate.npy",
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        case_id=complete_path.measurement_runner._SINGLE_STAGE_CASE_ID,
        scale="native_default",
        configuration={"outer_maxiter": 1_000},
        input_fingerprint="2" * 64,
        configuration_fingerprint="3" * 64,
    )
    monkeypatch.setattr(complete_path, "read_input_bundle", lambda root: (bundle, {}))
    monkeypatch.setattr(
        complete_path.measurement_runner,
        "build_measurement_environment",
        lambda profile_id, **kwargs: {"PROFILE": profile_id},
    )
    input_bundle_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    expected_samples = [
        replace(sample, input_bundle_sha256=input_bundle_sha256)
        for sample in _samples()
    ]
    executed: list[tuple[str, str, int | None]] = []
    snapshot_root = tmp_path / "immutable-snapshot-without-git"
    snapshot_root.mkdir()

    def executor(run, sequence_index, environment, workspace):
        del environment, workspace
        executed.append((run.profile_id, run.phase, run.sample_index))
        return expected_samples[sequence_index]

    output = collect_complete_path_evidence(
        artifact_root=tmp_path / "complete-path",
        specimen_document_path=specimen_path,
        input_bundle_path=input_path,
        candidate_path=candidate_path,
        binding=binding,
        python_executable="/unused/python",
        repo_root=snapshot_root,
        executor=executor,
    )

    assert executed == [
        (run.profile_id, run.phase, run.sample_index)
        for run in build_complete_path_plan()
    ]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_id"] == complete_path.SCHEMA_ID
    assert document["matched_complete_path_reference_timings_ns"]["c0"] == 110.0


@pytest.mark.parametrize("profile_id", complete_path.PROFILE_IDS)
def test_lane_environment_ssot_binds_route_and_numeric_policy(
    profile_id: complete_path.ProfileId, tmp_path: Path
) -> None:
    environment = build_complete_path_lane_environment(
        profile_id,
        {
            "PRESERVED": "yes",
            "CUDA_VISIBLE_DEVICES": "0",
            "JAX_COMPILATION_CACHE_DIR": "/route-only",
        },
        gpu_uuid="GPU-qualified",
        repo_root=tmp_path,
    )

    assert environment["PRESERVED"] == "yes"
    assert environment["SIMSOPT_PRECISION"] == "fp64"
    assert environment["JAX_ENABLE_X64"] == "1"
    assert (
        "JAX_COMPILATION_CACHE_DIR"
        not in complete_path.normalize_snapshot_lane_environment(environment)
    )
    if profile_id == "native_cpu":
        assert environment["JAX_PLATFORMS"] == "cpu"
        assert "CUDA_VISIBLE_DEVICES" not in environment
        assert (
            complete_path.measurement_runner._EXACT_ADJOINT_ENVIRONMENT_NAME
            not in environment
        )
    else:
        assert environment["JAX_PLATFORMS"] == "cuda"
        assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-qualified"
        assert environment["SIMSOPT_EXACT_ADJOINT_DENSE_LU"] == "1"


def test_snapshot_provenance_cli_dispatches_complete_lane_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        complete_path, "binding_from_phase0_checkpoints", lambda *paths: _binding()
    )
    monkeypatch.setattr(
        complete_path,
        "write_lane_snapshot_provenance_set",
        lambda output_root, **kwargs: {
            profile_id: output_root / f"{complete_path.LANE_IDS[profile_id]}.json"
            for profile_id in complete_path.PROFILE_IDS
        },
    )
    required = (
        "gate-checkpoint",
        "warm-checkpoint",
        "native-reference",
        "snapshot-publication",
        "snapshot-manifest",
        "import-attestation",
        "runner-spec",
        "runtime-provenance",
        "device-probe",
    )
    argv = ["snapshot-provenance"]
    for name in required:
        argv.extend((f"--{name}", str(tmp_path / name)))
    argv.extend(("--output-root", str(tmp_path / "lanes")))

    assert complete_path.main(argv) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(complete_path.PROFILE_IDS)
