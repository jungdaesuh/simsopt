from __future__ import annotations

import ast
import gzip
import hashlib
import inspect
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import benchmarks.run_single_stage_native_equivalent_quality_campaign as campaign_runner
import benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt as receipt_module
import numpy as np
import pytest
import simsoptpp
from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    ImportBinding,
    RuntimeIdentity,
    RuntimeIdentityV2,
    RuntimeObservation,
    RuntimeObservationV2,
    SnapshotEntry,
    SnapshotPublication,
    SnapshotRole,
    SourceRoot,
    WorktreeIdentity,
    build_runtime_evidence,
    build_runtime_evidence_v2,
    build_snapshot_identity,
    canonical_json_bytes,
    effective_environment,
    publish_immutable_snapshot,
    publish_runtime_evidence,
    publish_runtime_evidence_v2,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    EVIDENCE_REF_KEYS,
    FINAL_CERTIFICATE_FIELDS,
    GPU_UUID,
    HISTORY_FLOAT_FIELDS,
    HISTORY_INTEGER_FIELDS,
    HISTORY_ROW_RAW_FIELDS,
    MANIFEST_FILENAME,
    MAXIMUM_ATTEMPTS,
    PHASE_IDS,
    PHASE_SCHEMA_SHA256,
    PLAN_SHA256,
    POLICY_RAW_HASH_FIELDS,
    POLICY_RAW_SCALAR_FIELDS,
    POLICY_RAW_VECTOR_FIELDS,
    RECEIPT_FILENAME,
    TERMINAL_RAW_SCALAR_FIELDS,
    TRACE_LOOP_ENVELOPE_NAME,
    AttemptOutcome,
    DiagnosticVerdict,
    ExecutionEvidence,
    FailureStage,
    IncompleteDiagnosticReceipt,
    build_incomplete_diagnostic_receipt,
    diagnostic_artifact_manifest_payload,
    diagnostic_receipt_bytes,
    diagnostic_receipt_payload,
    execution_evidence_payload,
    history_evidence_from_arrays,
    load_and_validate_diagnostic_artifact,
    load_diagnostic_receipt_bytes,
    normalize_chrome_trace,
    policy_evidence_payload,
    terminal_numerical_payload,
    validate_diagnostic_preflight_gate,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NativeEquivalentQualityPolicy,
)


def _diag2_ref(path: Path, root: Path, schema: str) -> ArtifactRef:
    return _artifact_ref(path, root, schema)


def _diag2_raw_ref(path: Path, root: Path, schema: str, data: bytes) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _diag2_ref(path, root, schema)


def test_diag2_frozen_subset_is_receipt_owned() -> None:
    payload = receipt_module.build_diag2_frozen_numerical_subset_payload()
    assert payload["entries"] == [
        {"relative_path": path, "sha256": digest}
        for path, digest in receipt_module.DIAG2_FROZEN_NUMERICAL_ENTRIES
    ]
    with pytest.raises(ValueError, match="differ from the SSOT"):
        receipt_module.build_diag2_frozen_numerical_subset_payload(
            {"src/changed.py": "0" * 64}
        )


def test_diag2_source_authority_rejects_coherently_republished_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first, _ = _publish_test_snapshot(first_root)
    filtered = receipt_module._diag2_filtered_source_entries(first)
    monkeypatch.setattr(
        receipt_module, "DIAG2_BASELINE_FILTERED_ENTRY_COUNT", len(filtered)
    )
    monkeypatch.setattr(
        receipt_module,
        "DIAG2_BASELINE_FILTERED_ENTRIES_SHA256",
        hashlib.sha256(canonical_json_bytes(filtered)).hexdigest(),
    )
    assert receipt_module.validate_diag2_source_snapshot_authority(first_root) == first

    second_root = tmp_path / "second"
    second_root.mkdir()
    inputs = first_root.parent / "snapshot-inputs"
    changed = inputs / "src/simsopt_jax/__init__.py"
    changed.write_bytes(changed.read_bytes() + b"coherent mutation\n")
    roles = dict(receipt_module.REQUIRED_SOURCE_ROLE_BINDINGS)
    roles.update(
        {
            "src/simsopt/__init__.py": "execution_source",
            "src/simsopt_jax/__init__.py": "execution_source",
            "src/simsopt_jax_adapters/__init__.py": "execution_source",
            "src/simsoptpp.so": "native_extension",
        }
    )
    roots = [
        SourceRoot(role, inputs / relative, relative)  # type: ignore[arg-type]
        for relative, role in sorted(roles.items())
    ]
    publish_immutable_snapshot(
        second_root / "source-snapshot", roots, worktree=first.worktree
    )
    with pytest.raises(ValueError, match="non-allowlisted DIAG1 delta"):
        receipt_module.validate_diag2_source_snapshot_authority(second_root)


def test_diag2_current_production_snapshot_includes_matrix_test_authority(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    native_extension = Path(simsoptpp.__file__).resolve(strict=True)
    roots = campaign_runner._enumerated_source_roots(repository, native_extension)
    worktree = WorktreeIdentity(
        git_head="1" * 40,
        tracked_diff_sha256="2" * 64,
        untracked_bytes_manifest_sha256="3" * 64,
        repo_root=str(repository),
    )
    current_root = tmp_path / "current"
    publication = publish_immutable_snapshot(
        current_root / "source-snapshot", roots, worktree=worktree
    )
    with pytest.raises(ValueError, match="non-allowlisted DIAG1 delta"):
        receipt_module.validate_diag2_source_snapshot_authority(current_root)
    entries = {entry.relative_path: entry for entry in publication.entries}
    matrix_path = (
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py"
    )
    assert entries[matrix_path].role == "test"

    missing_root = tmp_path / "missing"
    publish_immutable_snapshot(
        missing_root / "source-snapshot",
        tuple(root for root in roots if root.relative_path != matrix_path),
        worktree=worktree,
    )
    with pytest.raises(ValueError, match="path/role binding differs"):
        receipt_module.validate_diag2_source_snapshot_authority(missing_root)

    changed_relative = "src/simsopt_jax/__init__.py"
    changed_path = tmp_path / "changed-source.py"
    changed_path.write_bytes(
        (repository / changed_relative).read_bytes() + b"coherent forbidden delta\n"
    )
    changed_roots = tuple(
        replace(root, source_path=changed_path)
        if root.relative_path == changed_relative
        else root
        for root in roots
    )
    changed_root = tmp_path / "changed"
    publish_immutable_snapshot(
        changed_root / "source-snapshot", changed_roots, worktree=worktree
    )
    with pytest.raises(ValueError, match="non-allowlisted DIAG1 delta"):
        receipt_module.validate_diag2_source_snapshot_authority(changed_root)


@pytest.mark.parametrize("complete", [False, True])
def test_diag2_manifest_preserves_source_manifest_role_and_rejects_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    root = tmp_path / "artifact"
    (root / "source-snapshot").mkdir(parents=True)
    receipt_path = root / receipt_module.DIAG2_RECEIPT_FILENAME
    receipt_path.write_bytes(canonical_json_bytes({"fixture": "receipt"}))
    source_path = root / receipt_module.DIAG2_EVIDENCE_SLOT_PATHS["source_manifest"]
    source_path.write_bytes(
        canonical_json_bytes(
            {"schema_version": receipt_module.SOURCE_MANIFEST_SCHEMA_VERSION}
        )
    )
    (root / "source-snapshot/nested.py").write_bytes(b"nested source bytes\n")
    source_ref = _diag2_ref(
        source_path, root, receipt_module.SOURCE_MANIFEST_SCHEMA_VERSION
    )
    failure = None
    if not complete:
        failure = receipt_module.StructuredFailureV2(
            receipt_module.FailureStageV2.NATIVE_REFERENCE_FAILURE,
            receipt_module.FailureReasonCodeV2.REFERENCE_INVALID,
            "5" * 64,
        )
    terminal_payload = receipt_module.build_diag2_supervisor_terminal_payload(
        disposition="COMPLETE" if complete else "INCOMPLETE",
        failure=failure,
        launched_children=("preflight", "cold") if complete else (),
        policy_authority_produced=complete,
        preflight_authorized=complete,
        cold_authorized=complete,
        staging_root=tmp_path / f"artifact.partial-{'6' * 32}",
        final_root=root,
        nonce="6" * 32,
        algorithm_route_selection=(
            receipt_module.NextRoute.RETRY_MODEL_REUSE.value
            if complete
            else "NOT_PRODUCED"
        ),
    )
    terminal_ref = _diag2_raw_ref(
        root / "supervisor-terminal.json",
        root,
        receipt_module.DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        canonical_json_bytes(terminal_payload),
    )
    slots = {
        name: receipt_module.EvidenceSlot.absent(
            receipt_module.AbsenceReason.NOT_REACHED
        )
        for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    slots["source_manifest"] = receipt_module.EvidenceSlot.present(source_ref)
    slots["supervisor_terminal"] = receipt_module.EvidenceSlot.present(terminal_ref)
    monkeypatch.setattr(receipt_module, "_diag2_receipt_slots", lambda _root: slots)

    manifest = receipt_module.diag2_artifact_manifest_payload(root)
    roles = {str(row["relative_path"]): str(row["role"]) for row in manifest["entries"]}
    assert roles[source_ref.relative_path] == "source_manifest"
    assert roles["source-snapshot/nested.py"] == "source_snapshot"
    mutated_entries = [dict(row) for row in manifest["entries"]]
    for row in mutated_entries:
        if row["relative_path"] == source_ref.relative_path:
            row["role"] = "source_snapshot"
    (root / receipt_module.DIAG2_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(
            {
                "schema_version": receipt_module.DIAG2_MANIFEST_SCHEMA_VERSION,
                "entries": mutated_entries,
            }
        )
    )
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)
    with pytest.raises(ValueError, match=r"manifest\.entries\[.*\]\.role differs"):
        receipt_module.load_and_validate_diag2_artifact(root)


@pytest.mark.parametrize("mode", ["preflight", "cold"])
@pytest.mark.parametrize("execution_status", ["COMPILE_FAILURE", "COMPILE_OOM"])
def test_diag2_compile_failure_producer_union_is_minimum_typed(
    mode: str, execution_status: str
) -> None:
    runtime = {
        "backend": "gpu",
        "device": "gpu:0",
        "device_uuid": GPU_UUID,
        "jax": "test",
        "jax_enable_x64": True,
        "jaxlib": "test",
        "python": "test",
    }
    runtime_ref = ArtifactRef(
        f"{mode}/runtime-evidence.json",
        "1" * 64,
        1,
        "single-stage-fullspace-runtime-evidence-v1",
    )
    payload = receipt_module.build_diag2_compile_failure_producer_payload(
        mode=mode,
        execution_status=execution_status,
        runtime=runtime,
        runtime_evidence=runtime_ref,
        compile_started_ns=1,
        compile_completed_ns=2,
        process_seconds_before_serialization=0.5,
        failure_reasons=("opaque failure digest",),
    )
    assert payload["execution_status"] == execution_status
    assert "policy_evidence" not in payload
    mutated = dict(payload)
    mutated["policy_evidence"] = None
    with pytest.raises(ValueError, match="keys differ"):
        receipt_module.validate_diag2_producer_payload(mutated, mode=mode)


def test_diag3_trace_failure_producer_is_additive_and_path_exact() -> None:
    def reference(relative_path: str) -> dict[str, object]:
        return {
            "relative_path": relative_path,
            "sha256": "1" * 64,
            "size_bytes": 1,
            "schema_version": "test-v1",
        }

    payload: dict[str, object] = {
        "schema_version": receipt_module.DIAG3_COLD_RESULT_SCHEMA_VERSION,
        "route": receipt_module.DIAG2_ROUTE,
        "plan_sha256": receipt_module.DIAG2_PLAN_SHA256,
        "execution_status": "TRACE_NORMALIZATION_FAILED",
        "runtime": {},
        "runtime_evidence": reference("cold/runtime-evidence.json"),
        "policy_sha256": "2" * 64,
        "phase_schema_sha256": "3" * 64,
        "history_evidence": reference("cold/numerical-result/history.json"),
        "terminal_numerical_evidence": reference(
            "cold/numerical-result/terminal-numerical.json"
        ),
        "policy_evidence": reference("cold/policy.json"),
        "raw_trace_evidence": reference(
            "cold/numerical-result/raw-trace/plugins/profile/run/trace.trace.json.gz"
        ),
        "trace_intervals_evidence": None,
        "timestamps_ns": {},
        "transfer_audit": {},
        "endpoint_audit_called": False,
        "campaign_authorized": False,
        "failure_reasons": ["TRACE_NORMALIZATION_FAILED:" + "4" * 64],
    }

    assert (
        receipt_module.validate_diag3_producer_payload(payload, mode="cold") == payload
    )
    with pytest.raises(ValueError, match="identity differs"):
        receipt_module.validate_diag2_producer_payload(payload, mode="cold")
    mutated = dict(payload)
    mutated["history_evidence"] = reference("cold/history.json")
    with pytest.raises(ValueError, match="history_evidence path differs"):
        receipt_module.validate_diag3_producer_payload(mutated, mode="cold")

    slot = receipt_module.EvidenceSlot.present(
        ArtifactRef("cold/numerical-result/history.json", "5" * 64, 1, "test-v1")
    )
    encoded = receipt_module.diag2_evidence_slot_payload(slot)
    assert (
        receipt_module.parse_diag3_evidence_slot(encoded, name="cold_history") == slot
    )
    with pytest.raises(ValueError, match="frozen layout"):
        receipt_module.parse_diag2_evidence_slot(encoded, name="cold_history")


def test_diag3_trace_failure_vector_retains_the_valid_scientific_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs: dict[str, ArtifactRef | None] = {
        name: ArtifactRef(path, "1" * 64, 1, "test-v1")
        for name, path in receipt_module.DIAG3_EVIDENCE_SLOT_PATHS.items()
    }
    refs["cold_trace_intervals"] = None
    refs["execution"] = None
    failure = receipt_module.StructuredFailureV2(
        receipt_module.FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE,
        receipt_module.FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED,
        "2" * 64,
    )
    monkeypatch.setattr(
        receipt_module,
        "_validate_diag3_slots",
        lambda *_args, **_kwargs: None,
    )

    slots = receipt_module.derive_diag3_evidence_slots(
        artifact_root=tmp_path,
        artifact_refs=refs,
        failure=failure,
    )

    for name in ("cold_history", "cold_terminal_numerical", "cold_raw_trace"):
        assert slots[name].state is receipt_module.EvidenceState.PRESENT
    assert slots["cold_trace_intervals"] == receipt_module.EvidenceSlot.absent(
        receipt_module.AbsenceReason.TRACE_NORMALIZATION_FAILED
    )
    assert slots["execution"] == receipt_module.EvidenceSlot.absent(
        receipt_module.AbsenceReason.NOT_REACHED
    )


def test_diag3_pending_directory_is_forbidden_even_when_empty(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    (root / receipt_module.DIAG3_PENDING_NUMERICAL_DIRECTORY).mkdir(parents=True)

    with pytest.raises(ValueError, match="pending numerical result"):
        receipt_module._diag3_artifact_roles(root)


def test_diag4_plan_prefix_binds_corrected_observable_wire_contract() -> None:
    plan_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md"
    )
    plan_bytes = plan_path.read_bytes()
    prefix, marker, record = plan_bytes.partition(b"## Qualification Record\n")

    assert marker == b"## Qualification Record\n"
    assert record == b""
    assert hashlib.sha256(prefix).hexdigest() == receipt_module.DIAG4_PLAN_SHA256
    assert receipt_module.DIAG4_PLAN_SHA256 == (
        "987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c"
    )
    assert b"`boozer_residual_value`" in prefix
    assert b"`observables.boozer_residual_scalar`" in prefix


def test_diag4_slot_schema_is_ordered_trace_free_and_cross_rejected() -> None:
    assert tuple(receipt_module.DIAG4_EVIDENCE_SLOT_PATHS)[20:26] == (
        "cold_history",
        "cold_terminal_numerical",
        "cold_solve_timing",
        "cold_safeguard_telemetry",
        "execution",
        "supervisor_terminal",
    )
    assert "cold_raw_trace" not in receipt_module.DIAG4_EVIDENCE_SLOT_NAMES
    assert "cold_trace_intervals" not in receipt_module.DIAG4_EVIDENCE_SLOT_NAMES
    slot = receipt_module.EvidenceSlotV4.present(
        ArtifactRef(
            "cold/numerical-result/solve-timing.json",
            "1" * 64,
            1,
            receipt_module.DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
        )
    )
    encoded = receipt_module.diag4_evidence_slot_payload(slot)
    assert (
        receipt_module.parse_diag4_evidence_slot(encoded, name="cold_solve_timing")
        == slot
    )
    with pytest.raises(ValueError, match="unknown DIAG4 evidence slot"):
        receipt_module.parse_diag4_evidence_slot(encoded, name="cold_raw_trace")
    with pytest.raises(ValueError, match="unknown DIAG2 evidence slot"):
        receipt_module.parse_diag2_evidence_slot(encoded, name="cold_solve_timing")


def _diag4_solve_timing_payload() -> dict[str, object]:
    return receipt_module.solve_timing_evidence_payload(
        child_pid=123,
        child_start_time_ticks=456,
        backend="gpu",
        gpu_uuid=GPU_UUID,
        problem_sha256="1" * 64,
        optimizer_options_sha256="2" * 64,
        base_neq_gntr1_policy_sha256="3" * 64,
        scaling_sha256="4" * 64,
        bootstrap_state_sha256="5" * 64,
        initial_physical_state_sha256="6" * 64,
        identity_sha256="7" * 64,
        source_manifest_sha256="8" * 64,
        process_started_monotonic_ns=1,
        state_ready_monotonic_ns=2,
        solve_started_monotonic_ns=3,
        solve_stopped_monotonic_ns=1_000_000_003,
        finalizer_completed_monotonic_ns=1_000_000_004,
        endpoint_audit_completed_monotonic_ns=1_000_000_005,
        serialization_started_monotonic_ns=1_000_000_006,
        hot_h2d_transfers=0,
        hot_d2h_transfers=0,
        python_callbacks=0,
        final_d2h_transfers=1,
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("profiler_enabled", True, "profiler call audit differs"),
        ("profiler_start_calls", 1, "profiler call audit differs"),
        ("solve_started_monotonic_ns", 2.5, "must be an integer"),
        ("solve_stopped_monotonic_ns", 3, "order differs"),
        ("synchronized_solve_seconds", 1.5, "arithmetic differs"),
    ),
)
def test_diag4_solve_timing_round_trip_and_mutations(
    field: str, replacement: object, message: str
) -> None:
    payload = _diag4_solve_timing_payload()
    evidence = receipt_module.validate_solve_timing_evidence_payload(payload)
    assert evidence.synchronized_solve_seconds == 1.0
    assert "process_stopped_monotonic_ns" not in payload
    mutated = dict(payload)
    mutated[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        receipt_module.validate_solve_timing_evidence_payload(mutated)


def _diag4_history(
    *,
    outcome: receipt_module.AttemptOutcome = receipt_module.AttemptOutcome.ACCEPTED,
    correction_step_ratio: float | None = 7.0e-4,
    corrected_radius_ratio: float | None = 0.9,
) -> receipt_module.HistoryEvidence:
    inactive = receipt_module.HistoryRow(
        receipt_module.AttemptOutcome.INACTIVE,
        0,
        (0, 0, 0),
        False,
        tuple(None for _ in HISTORY_FLOAT_FIELDS),
    )
    active_floats = list(inactive.floating_values)
    active_floats[HISTORY_FLOAT_FIELDS.index("correction_step_ratio")] = (
        correction_step_ratio
    )
    active_floats[HISTORY_FLOAT_FIELDS.index("corrected_radius_ratio")] = (
        corrected_radius_ratio
    )
    active_floats[HISTORY_FLOAT_FIELDS.index("trust_radius")] = (
        2.0**-14
        if outcome is receipt_module.AttemptOutcome.RETRY_STEP_BOUNDS
        else 2.0**-10
    )
    active_floats[HISTORY_FLOAT_FIELDS.index("actual_reduction")] = 1.0
    active_floats[HISTORY_FLOAT_FIELDS.index("predicted_reduction")] = 1.0
    active = replace(
        inactive,
        outcome=outcome,
        accepted_step_number=int(outcome is receipt_module.AttemptOutcome.ACCEPTED),
        floating_values=tuple(active_floats),
    )
    return receipt_module.HistoryEvidence(
        (active, *(inactive for _ in range(MAXIMUM_ATTEMPTS - 1))),
        1,
        int(outcome is receipt_module.AttemptOutcome.ACCEPTED),
        int(outcome in receipt_module.RETRY_OUTCOMES),
        receipt_module.LoopStatus.ATTEMPT_LIMIT,
        False,
        False,
        False,
        0,
        0,
    )


def _diag4_telemetry_payload(
    *,
    history: receipt_module.HistoryEvidence | None = None,
    nonlinear_corrections: tuple[int, ...] | None = None,
    maximum_individual_correction_step_ratio: tuple[float, ...] | None = None,
    correction_path_step_ratio: tuple[float, ...] | None = None,
) -> dict[str, object]:
    if history is None:
        history = _diag4_history()
    if nonlinear_corrections is None:
        nonlinear_corrections = (
            2,
            *(0 for _ in range(MAXIMUM_ATTEMPTS - 1)),
        )
    if maximum_individual_correction_step_ratio is None:
        maximum_individual_correction_step_ratio = (
            5.0e-4,
            *(float("nan") for _ in range(MAXIMUM_ATTEMPTS - 1)),
        )
    if correction_path_step_ratio is None:
        correction_path_step_ratio = (
            8.0e-4,
            *(float("nan") for _ in range(MAXIMUM_ATTEMPTS - 1)),
        )
    subtrial_count = np.zeros(MAXIMUM_ATTEMPTS, dtype="<i4")
    subtrial_count[0] = 1
    selected_subtrial_index = np.full(MAXIMUM_ATTEMPTS, -1, dtype="<i4")
    selected_subtrial_index[0] = 0
    shape = (MAXIMUM_ATTEMPTS, 3)
    subtrial_outcome = np.zeros(shape, dtype="<i4")
    subtrial_outcome[0, 0] = tuple(receipt_module.AttemptOutcome).index(
        history.rows[0].outcome
    )
    float_matrices = {
        name: np.full(shape, np.nan, dtype="<f8")
        for name in receipt_module.DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS
    }
    float_matrices["subtrial_trust_radius"][0, 0] = 2.0**-10
    float_matrices["subtrial_actual_reduction"][0, 0] = 1.0
    float_matrices["subtrial_predicted_reduction"][0, 0] = 1.0
    float_matrices["subtrial_maximum_individual_correction_step_ratio"][0, 0] = (
        maximum_individual_correction_step_ratio[0]
    )
    float_matrices["subtrial_correction_path_step_ratio"][0, 0] = (
        correction_path_step_ratio[0]
    )
    float_matrices["subtrial_corrected_radius_ratio"][0, 0] = (
        np.nan
        if history.rows[0].floating("corrected_radius_ratio") is None
        else history.rows[0].floating("corrected_radius_ratio")
    )
    integer_work = {
        name: np.zeros(shape, dtype="<i4")
        for name in receipt_module.DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS
    }
    correction_count = nonlinear_corrections[0]
    solve_call = int(
        history.rows[0].outcome is not receipt_module.AttemptOutcome.FATAL_CURRENT_STATE
    )
    trial_evaluated = int(correction_count > 0)
    integer_work["subtrial_steihaug_solve_calls"][0, 0] = solve_call
    integer_work["subtrial_total_hvp_evaluations"][0, 0] = 3 + trial_evaluated
    integer_work["subtrial_nonlinear_corrections"][0, 0] = correction_count
    integer_work["subtrial_joint_evaluations"][0, 0] = 1 + (
        2 + correction_count if trial_evaluated else 0
    )
    integer_work["subtrial_joint_linearizations"][0, 0] = 1 + correction_count
    integer_work["subtrial_joint_value_evaluations"][0, 0] = 2 if trial_evaluated else 0
    integer_work["subtrial_objective_residual_linearizations"][0, 0] = 1
    integer_work["subtrial_gram_factorizations"][0, 0] = 1 + correction_count
    integer_work["subtrial_gram_solves"][0, 0] = 2 + correction_count + solve_call * 3
    if history.rows[0].outcome is receipt_module.AttemptOutcome.RETRY_STEP_BOUNDS:
        subtrial_count[0] = 3
        selected_subtrial_index[0] = 2
        for column in range(3):
            subtrial_outcome[0, column] = tuple(receipt_module.AttemptOutcome).index(
                receipt_module.AttemptOutcome.RETRY_STEP_BOUNDS
            )
            float_matrices["subtrial_trust_radius"][0, column] = 2.0 ** (
                -10 - 2 * column
            )
            float_matrices["subtrial_actual_reduction"][0, column] = 1.0
            float_matrices["subtrial_predicted_reduction"][0, column] = 1.0
            float_matrices["subtrial_maximum_individual_correction_step_ratio"][
                0, column
            ] = maximum_individual_correction_step_ratio[0]
            float_matrices["subtrial_correction_path_step_ratio"][0, column] = (
                correction_path_step_ratio[0]
            )
            float_matrices["subtrial_corrected_radius_ratio"][0, column] = history.rows[
                0
            ].floating("corrected_radius_ratio")
            for name in receipt_module.DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS:
                integer_work[name][0, column] = integer_work[name][0, 0]
    return receipt_module.safeguard_telemetry_payload(
        history_evidence=ArtifactRef(
            "cold/numerical-result/history.json", "9" * 64, 1, "history-v1"
        ),
        problem_sha256="1" * 64,
        optimizer_options_sha256="2" * 64,
        base_neq_gntr1_policy_sha256="3" * 64,
        scaling_sha256="4" * 64,
        bootstrap_state_sha256="5" * 64,
        initial_physical_state_sha256="6" * 64,
        identity_sha256="7" * 64,
        loop_attempts=history.attempts,
        accepted_steps=history.accepted_steps,
        retryable_rejections=history.retryable_rejections,
        terminal_status=history.status.value,
        quality_latch=history.quality_latch,
        history_outcomes=tuple(row.outcome.value for row in history.rows),
        nonlinear_corrections=np.asarray(nonlinear_corrections, dtype="<i4"),
        maximum_individual_correction_step_ratio=np.asarray(
            maximum_individual_correction_step_ratio, dtype="<f8"
        ),
        correction_path_step_ratio=np.asarray(correction_path_step_ratio, dtype="<f8"),
        steihaug_solve_calls=np.asarray(
            (solve_call, *(0 for _ in range(MAXIMUM_ATTEMPTS - 1))), dtype="<i4"
        ),
        subtrial_count=subtrial_count,
        selected_subtrial_index=selected_subtrial_index,
        subtrial_outcome=subtrial_outcome,
        **float_matrices,
        **integer_work,
    )


def _diag4_cold_producer_payload() -> dict[str, object]:
    def reference(name: str, schema: str) -> dict[str, object]:
        return receipt_module._artifact_ref_payload(
            ArtifactRef(
                receipt_module.DIAG4_EVIDENCE_SLOT_PATHS[name],
                "9" * 64,
                1,
                schema,
            )
        )

    return {
        "schema_version": receipt_module.DIAG4_COLD_RESULT_SCHEMA_VERSION,
        "numerical_bundle_schema_version": (
            receipt_module.DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION
        ),
        "route": receipt_module.DIAG4_ROUTE,
        "numerical_route": receipt_module.DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": (
            receipt_module.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
        ),
        "plan_sha256": receipt_module.DIAG4_PLAN_SHA256,
        "execution_status": "COMPLETE",
        "runtime": {},
        "runtime_evidence": reference("cold_runtime", "runtime-v1"),
        "base_neq_gntr1_policy_sha256": "3" * 64,
        "policy_evidence": reference("cold_policy", "policy-v1"),
        "problem_sha256": "1" * 64,
        "optimizer_options_sha256": "2" * 64,
        "scaling_sha256": "4" * 64,
        "bootstrap_state_sha256": "5" * 64,
        "initial_physical_state_sha256": "6" * 64,
        "identity_sha256": "7" * 64,
        "source_manifest_sha256": "8" * 64,
        "history_evidence": reference("cold_history", "history-v1"),
        "terminal_numerical_evidence": reference(
            "cold_terminal_numerical",
            f"{receipt_module.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal",
        ),
        "solve_timing_evidence": reference(
            "cold_solve_timing", receipt_module.DIAG4_SOLVE_TIMING_SCHEMA_VERSION
        ),
        "safeguard_telemetry_evidence": reference(
            "cold_safeguard_telemetry",
            receipt_module.DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
        ),
        **receipt_module.diag4_profiler_call_audit_payload(),
        "endpoint_audit_called": True,
        "campaign_authorized": False,
        "failure_reasons": [],
    }


def test_diag4_telemetry_round_trip_history_join_and_mutations() -> None:
    history = _diag4_history()
    payload = _diag4_telemetry_payload()
    evidence = receipt_module.validate_safeguard_telemetry_payload(
        payload,
        history=history,
        expected_history_evidence=ArtifactRef(
            "cold/numerical-result/history.json", "9" * 64, 1, "history-v1"
        ),
    )
    assert dict(evidence.subtrial_summary)["total_nonlinear_corrections"] == 2
    mutated_summary = json.loads(json.dumps(payload))
    mutated_summary["subtrial_summary"]["total_nonlinear_corrections"] = 1
    with pytest.raises(ValueError, match="summary differs"):
        receipt_module.validate_safeguard_telemetry_payload(mutated_summary)
    mutated_padding = json.loads(json.dumps(payload))
    mutated_padding["nonlinear_corrections"]["values"][-1] = 1
    _refresh_diag4_envelope(mutated_padding, "nonlinear_corrections")
    with pytest.raises(ValueError, match="range or padding differs"):
        receipt_module.validate_safeguard_telemetry_payload(mutated_padding)
    mutated_history = replace(history, accepted_steps=0)
    with pytest.raises(ValueError, match="differs from legacy history"):
        receipt_module.validate_safeguard_telemetry_payload(
            payload, history=mutated_history
        )


@pytest.mark.parametrize(
    "mutation", ("length", "boolean", "negative", "hash", "history_ref", "attempts")
)
def test_diag4_telemetry_vector_shape_type_range_and_join_mutations(
    mutation: str,
) -> None:
    payload = json.loads(json.dumps(_diag4_telemetry_payload()))
    expected_history = ArtifactRef(
        "cold/numerical-result/history.json", "9" * 64, 1, "history-v1"
    )
    if mutation == "length":
        payload["nonlinear_corrections"]["values"].pop()
        _refresh_diag4_envelope(payload, "nonlinear_corrections")
    elif mutation == "boolean":
        payload["nonlinear_corrections"]["values"][0] = True
        _refresh_diag4_envelope(payload, "nonlinear_corrections")
    elif mutation == "negative":
        payload["nonlinear_corrections"]["values"][0] = -1
        _refresh_diag4_envelope(payload, "nonlinear_corrections")
    elif mutation == "hash":
        payload["nonlinear_corrections"]["sha256"] = "f" * 64
    elif mutation == "history_ref":
        payload["history_evidence"]["sha256"] = "e" * 64
    else:
        payload["loop_attempts"] = MAXIMUM_ATTEMPTS + 1
    with pytest.raises((TypeError, ValueError)):
        receipt_module.validate_safeguard_telemetry_payload(
            payload, expected_history_evidence=expected_history
        )


def _refresh_diag4_envelope(payload: dict[str, object], name: str) -> None:
    envelope = payload[name]
    core = {
        "dtype": envelope["dtype"],
        "shape": envelope["shape"],
        "values": envelope["values"],
    }
    envelope["sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _refresh_diag4_ratio_claims(payload: dict[str, object]) -> None:
    _refresh_diag4_envelope(payload, "maximum_individual_correction_step_ratio")
    _refresh_diag4_envelope(payload, "correction_path_step_ratio")


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_key",
        "boolean",
        "accepted_null",
        "finite_padding",
        "nonfinite_active",
        "hash",
        "summary",
        "individual_exceeds_path",
        "net_exceeds_path",
        "accepted_individual_nextafter",
        "accepted_path_nextafter",
    ),
)
def test_diag4_correction_ratio_exact_schema_bounds_and_history_mutations(
    mutation: str,
) -> None:
    history = _diag4_history()
    payload = json.loads(json.dumps(_diag4_telemetry_payload()))
    if mutation == "missing_key":
        del payload["correction_path_step_ratio"]["sha256"]
    elif mutation == "boolean":
        payload["maximum_individual_correction_step_ratio"]["values"][0] = True
        _refresh_diag4_ratio_claims(payload)
    elif mutation == "accepted_null":
        payload["correction_path_step_ratio"]["values"][0] = None
        _refresh_diag4_ratio_claims(payload)
    elif mutation == "finite_padding":
        payload["maximum_individual_correction_step_ratio"]["values"][-1] = 0.0
        _refresh_diag4_ratio_claims(payload)
    elif mutation == "nonfinite_active":
        payload["correction_path_step_ratio"]["values"][0] = float("nan")
    elif mutation == "hash":
        payload["correction_path_step_ratio"]["sha256"] = "f" * 64
    elif mutation == "summary":
        payload["subtrial_summary"]["total_nonlinear_corrections"] = 1
    elif mutation == "individual_exceeds_path":
        payload["maximum_individual_correction_step_ratio"]["values"][0] = 9.0e-4
        _refresh_diag4_ratio_claims(payload)
    elif mutation == "net_exceeds_path":
        history = _diag4_history(correction_step_ratio=9.0e-4)
    elif mutation == "accepted_individual_nextafter":
        exceeded = float(np.nextafter(1.0e-3, np.inf))
        payload["maximum_individual_correction_step_ratio"]["values"][0] = exceeded
        payload["correction_path_step_ratio"]["values"][0] = exceeded
        _refresh_diag4_ratio_claims(payload)
    else:
        payload["correction_path_step_ratio"]["values"][0] = float(
            np.nextafter(2.0e-3, np.inf)
        )
        _refresh_diag4_ratio_claims(payload)
    with pytest.raises((TypeError, ValueError)):
        receipt_module.validate_safeguard_telemetry_payload(payload, history=history)


def test_diag4_correction_ratio_builder_rejects_pattern_and_type_drift() -> None:
    nan_padding = tuple(float("nan") for _ in range(MAXIMUM_ATTEMPTS - 1))
    wrong_dtype = np.asarray((5.0e-4, *nan_padding), dtype="<f4")
    with pytest.raises(ValueError, match="dtype or shape differs"):
        receipt_module._diag4_safeguard_envelope_payload(
            wrong_dtype,
            context="test",
            dtype="<f8",
            shape=(MAXIMUM_ATTEMPTS,),
        )
    with pytest.raises(ValueError, match="zero-count value must be NaN"):
        _diag4_telemetry_payload(
            correction_path_step_ratio=(8.0e-4, *(0.0 for _ in nan_padding))
        )
    with pytest.raises(ValueError, match="correction ratios differ"):
        _diag4_telemetry_payload(
            nonlinear_corrections=(1, *(0 for _ in nan_padding)),
            maximum_individual_correction_step_ratio=(5.0e-4, *nan_padding),
            correction_path_step_ratio=(8.0e-4, *nan_padding),
        )


def test_diag4_nonfinite_correction_certificate_telemetry_is_truthful() -> None:
    history = _diag4_history(
        outcome=receipt_module.AttemptOutcome.RETRY_CORRECTION_CERTIFICATE,
        correction_step_ratio=None,
        corrected_radius_ratio=None,
    )
    nan_padding = tuple(float("nan") for _ in range(MAXIMUM_ATTEMPTS - 1))
    payload = _diag4_telemetry_payload(
        history=history,
        nonlinear_corrections=(2, *(0 for _ in nan_padding)),
        maximum_individual_correction_step_ratio=(float("nan"), *nan_padding),
        correction_path_step_ratio=(float("nan"), *nan_padding),
    )
    evidence = receipt_module.validate_safeguard_telemetry_payload(
        payload, history=history
    )
    assert evidence.maximum_individual_correction_step_ratio[0] is None
    assert evidence.correction_path_step_ratio[0] is None


def test_diag4_step_bound_retry_proves_the_failed_certificate() -> None:
    exceeded = float(np.nextafter(1.0e-3, np.inf))
    history = _diag4_history(
        outcome=receipt_module.AttemptOutcome.RETRY_STEP_BOUNDS,
        correction_step_ratio=exceeded,
    )
    nan_padding = tuple(float("nan") for _ in range(MAXIMUM_ATTEMPTS - 1))
    payload = _diag4_telemetry_payload(
        history=history,
        nonlinear_corrections=(1, *(0 for _ in nan_padding)),
        maximum_individual_correction_step_ratio=(exceeded, *nan_padding),
        correction_path_step_ratio=(exceeded, *nan_padding),
    )
    evidence = receipt_module.validate_safeguard_telemetry_payload(
        payload, history=history
    )
    assert evidence.maximum_individual_correction_step_ratio[0] == exceeded
    with pytest.raises(ValueError, match="accepted correction bounds differ"):
        _diag4_telemetry_payload(
            history=_diag4_history(correction_step_ratio=exceeded),
            nonlinear_corrections=(1, *(0 for _ in nan_padding)),
            maximum_individual_correction_step_ratio=(exceeded, *nan_padding),
            correction_path_step_ratio=(exceeded, *nan_padding),
        )


@pytest.mark.parametrize(
    "field",
    (
        *receipt_module.DIAG4_OUTER_TELEMETRY_FIELDS,
        "subtrial_count",
        "selected_subtrial_index",
        *receipt_module.DIAG4_SUBTRIAL_MATRIX_FIELDS,
    ),
)
@pytest.mark.parametrize("mutation", ("dtype", "shape", "hash"))
def test_diag4_every_typed_telemetry_envelope_is_exact(
    field: str, mutation: str
) -> None:
    payload = json.loads(json.dumps(_diag4_telemetry_payload()))
    envelope = payload[field]
    if mutation == "dtype":
        envelope["dtype"] = "<i8"
    elif mutation == "shape":
        envelope["shape"][-1] += 1
    else:
        envelope["sha256"] = "f" * 64
    with pytest.raises((TypeError, ValueError)):
        receipt_module.validate_safeguard_telemetry_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "selected_solve",
        "outcome_code",
        "steihaug_bound",
        "work",
        "recurrence",
        "summary",
    ),
)
def test_diag4_safeguard_selected_work_recurrence_and_summary_mutations(
    mutation: str,
) -> None:
    exceeded = float(np.nextafter(1.0e-3, np.inf))
    history = _diag4_history(
        outcome=receipt_module.AttemptOutcome.RETRY_STEP_BOUNDS,
        correction_step_ratio=exceeded,
    )
    padding = tuple(float("nan") for _ in range(MAXIMUM_ATTEMPTS - 1))
    payload = json.loads(
        json.dumps(
            _diag4_telemetry_payload(
                history=history,
                nonlinear_corrections=(1, *(0 for _ in padding)),
                maximum_individual_correction_step_ratio=(exceeded, *padding),
                correction_path_step_ratio=(exceeded, *padding),
            )
        )
    )
    if mutation == "selected_solve":
        payload["steihaug_solve_calls"]["values"][0] = 0
        _refresh_diag4_envelope(payload, "steihaug_solve_calls")
    elif mutation == "outcome_code":
        payload["subtrial_outcome"]["values"][0][2] = -1
        _refresh_diag4_envelope(payload, "subtrial_outcome")
    elif mutation == "steihaug_bound":
        payload["subtrial_steihaug_iterations"]["values"][0][0] = 33
        payload["subtrial_steihaug_hvp_evaluations"]["values"][0][0] = 33
        _refresh_diag4_envelope(payload, "subtrial_steihaug_iterations")
        _refresh_diag4_envelope(payload, "subtrial_steihaug_hvp_evaluations")
    elif mutation == "work":
        payload["subtrial_joint_evaluations"]["values"][0][2] += 1
        _refresh_diag4_envelope(payload, "subtrial_joint_evaluations")
    elif mutation == "recurrence":
        payload["subtrial_trust_radius"]["values"][0][1] *= 0.5
        _refresh_diag4_envelope(payload, "subtrial_trust_radius")
    else:
        payload["subtrial_summary"]["total_subtrials"] -= 1
    with pytest.raises((TypeError, ValueError)):
        receipt_module.validate_safeguard_telemetry_payload(payload, history=history)


def _diag4_test_numerical_identity() -> (
    receipt_module.NativeEquivalentNumericalIdentity
):
    return receipt_module.NativeEquivalentNumericalIdentity(
        receipt_module.DIAG4_NUMERICAL_ROUTE,
        receipt_module.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
    )


def _diag4_structural_terminal_payload() -> dict[str, object]:
    objective_terms = {
        name: 0.0 for name in receipt_module.DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS
    }
    observables = {
        name: 0.0 for name in receipt_module.DIAG4_ENDPOINT_OBSERVABLE_FIELDS
    }
    legacy = receipt_module.terminal_numerical_payload(
        arrays={
            name: {"content_sha256": "a" * 64} for name in receipt_module.ARRAY_SPECS
        },
        objective=0.0,
        objective_terms=objective_terms,
        objective_weights={name: 1.0 for name in objective_terms},
        reconstructed_objective=0.0,
        authoritative_objective=0.0,
        final_certificate={name: 0.0 for name in FINAL_CERTIFICATE_FIELDS},
        kkt_status=receipt_module.KktStatus.AVAILABLE,
        raw_kkt_inf=0.0,
        scaled_stationarity_inf=0.0,
        residual_value_defect=0.0,
        residual_gradient_defect=0.0,
        transpose_primal_dot=0.0,
        transpose_adjoint_dot=0.0,
        transpose_denominator=1.0,
        transpose_defect=0.0,
        terminal_endpoint_diagnostics_seconds=0.0,
    )
    return receipt_module.diag4_terminal_numerical_payload(
        terminal_numerical=legacy,
        numerical_identity=_diag4_test_numerical_identity(),
        endpoint_state_sha256="a" * 64,
        terminal_observables=observables,
        endpoint_objective_terms=objective_terms,
        endpoint_observables=observables,
    )


@pytest.mark.parametrize(
    "mutation",
    ("none", "state", "term", "observable", "schema", "identity", "array_bytes"),
)
def test_diag4_terminal_endpoint_evidence_round_trip_and_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    receipt = _complete_receipt(tmp_path / "terminal", monkeypatch, quality_hit=True)
    terminal_ref = dict(receipt.evidence_refs)["terminal_numerical"]
    legacy = json.loads(
        (tmp_path / "terminal" / terminal_ref.relative_path).read_bytes()
    )
    observables = {
        name: 0.0 for name in receipt_module.DIAG4_ENDPOINT_OBSERVABLE_FIELDS
    }
    payload = receipt_module.diag4_terminal_numerical_payload(
        terminal_numerical=legacy,
        numerical_identity=_diag4_test_numerical_identity(),
        endpoint_state_sha256=legacy["arrays"]["physical_state"]["content_sha256"],
        terminal_observables=observables,
        endpoint_objective_terms={
            name: legacy["objective_terms"][name]
            for name in receipt_module.DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS
        },
        endpoint_observables=observables,
    )
    payload = json.loads(canonical_json_bytes(payload))
    if mutation == "state":
        payload["endpoint_state_sha256"] = "f" * 64
    elif mutation == "term":
        payload["endpoint_objective_terms"]["non_qs"] = 1.0
    elif mutation == "observable":
        payload["endpoint_observables"]["G"] = 1.0
    elif mutation == "schema":
        payload["schema_version"] = f"{receipt_module.SCHEMA_VERSION}-terminal"
    elif mutation == "identity":
        payload["identity_sha256"] = "f" * 64
    elif mutation == "array_bytes":
        array_path = (
            tmp_path
            / "terminal"
            / legacy["arrays"]["physical_state"]["artifact"]["relative_path"]
        )
        array_path.write_bytes(array_path.read_bytes() + b"x")
    if mutation in {"none", "identity"}:
        evidence = receipt_module.validate_diag4_terminal_numerical_payload(
            tmp_path / "terminal", payload
        )
        assert tuple(name for name, _ in evidence.terminal_observables) == (
            receipt_module.DIAG4_ENDPOINT_OBSERVABLE_FIELDS
        )
        assert tuple(name for name, _ in evidence.endpoint_objective_terms) == (
            receipt_module.DIAG4_ENDPOINT_OBJECTIVE_TERM_FIELDS
        )
        if mutation == "identity":
            assert evidence.numerical_identity != _diag4_test_numerical_identity()
            monkeypatch.setattr(
                receipt_module,
                "_parse_history",
                lambda _value, **_kwargs: _diag4_history(),
            )
            with pytest.raises(receipt_module.Diag4NumericalDocumentError) as captured:
                receipt_module.validate_diag4_numerical_documents(
                    history={},
                    solve_timing=_diag4_solve_timing_payload(),
                    safeguard_telemetry=_diag4_telemetry_payload(),
                    terminal_numerical=payload,
                    producer=_diag4_cold_producer_payload(),
                    artifact_root=tmp_path / "terminal",
                )
            assert (
                captured.value.reason
                is receipt_module.FailureReasonCodeV4.NUMERICAL_IDENTITY_MISMATCH
            )
        else:
            assert evidence.numerical_identity == _diag4_test_numerical_identity()
            monkeypatch.setattr(
                receipt_module,
                "_parse_history",
                lambda _value, **_kwargs: _diag4_history(),
            )
            joined = receipt_module.validate_diag4_numerical_documents(
                history={},
                solve_timing=_diag4_solve_timing_payload(),
                safeguard_telemetry=_diag4_telemetry_payload(),
                terminal_numerical=payload,
                producer=_diag4_cold_producer_payload(),
                artifact_root=tmp_path / "terminal",
            )
            assert len(joined) == 4
    else:
        with pytest.raises(ValueError):
            receipt_module.validate_diag4_terminal_numerical_payload(
                tmp_path / "terminal", payload
            )


def test_diag4_endpoint_audit_structural_validation_does_not_require_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_audit = SimpleNamespace(passes=lambda: False)
    payload = {"schema_version": "structurally-valid-no-hit"}
    monkeypatch.setattr(
        receipt_module, "endpoint_audit_from_payload", lambda value: endpoint_audit
    )
    monkeypatch.setattr(
        receipt_module, "endpoint_audit_payload", lambda value: dict(payload)
    )
    parsed = receipt_module.validate_endpoint_audit_evidence_payload(payload)
    assert parsed is endpoint_audit
    assert not parsed.passes()
    with pytest.raises(ValueError, match="canonical round trip differs"):
        receipt_module.validate_endpoint_audit_evidence_payload(
            {"schema_version": "malformed"}
        )


@pytest.mark.parametrize(
    ("audit_passes", "expected"),
    (
        (True, receipt_module.ScientificOutcome.QUALITY_HIT),
        (False, receipt_module.ScientificOutcome.NO_HIT),
    ),
)
def test_diag4_cpu_scientific_outcome_separates_endpoint_parity_from_structure(
    monkeypatch: pytest.MonkeyPatch,
    audit_passes: bool,
    expected: receipt_module.ScientificOutcome,
) -> None:
    identity = _diag4_test_numerical_identity()
    status = receipt_module.LoopStatus.ATTEMPT_LIMIT
    history = SimpleNamespace(
        fatal=False,
        attempts=1,
        accepted_steps=1,
        retryable_rejections=0,
        status=status,
        quality_latch=True,
        first_quality_attempt=1,
        first_quality_accepted_step=1,
    )
    telemetry = SimpleNamespace(
        numerical_route=identity.numerical_route,
        numerical_result_schema_version=identity.numerical_result_schema_version,
        problem_sha256=identity.problem_sha256,
        optimizer_options_sha256=identity.optimizer_options_sha256,
        base_neq_gntr1_policy_sha256=identity.base_neq_gntr1_policy_sha256,
        scaling_sha256=identity.scaling_sha256,
        bootstrap_state_sha256=identity.bootstrap_state_sha256,
        initial_physical_state_sha256=identity.initial_physical_state_sha256,
        identity_sha256=identity.identity_sha256,
        loop_attempts=1,
        accepted_steps=1,
        retryable_rejections=0,
        terminal_status=status,
        quality_latch=True,
    )
    terminal = SimpleNamespace(numerical_identity=identity, terminal=object())
    endpoint_audit = SimpleNamespace(passes=lambda: audit_passes)
    quality = SimpleNamespace(
        residual_value_margin=0.0,
        residual_gradient_margin=0.0,
        transpose_margin=0.0,
        passes=True,
    )
    monkeypatch.setattr(
        receipt_module, "validate_terminal_endpoint_audit", lambda **_: None
    )
    monkeypatch.setattr(
        receipt_module, "_validate_terminal_raw_evidence", lambda *_: None
    )
    monkeypatch.setattr(receipt_module, "_validate_quality_replay", lambda *_: None)
    monkeypatch.setattr(receipt_module, "_terminal_semantics", lambda *_: True)
    monkeypatch.setattr(receipt_module, "_quality", lambda *_: quality)
    evidence = receipt_module.build_native_equivalent_scientific_evidence(
        history=history,
        safeguard_telemetry=telemetry,
        terminal=terminal,
        policy=object(),
        endpoint_audit=endpoint_audit,
        expected_numerical_identity=identity,
    )
    assert evidence.outcome is expected


def test_diag4_profiler_call_audit_is_one_route_owned_value() -> None:
    expected = {
        "profiler_enabled": False,
        "profiler_start_calls": 0,
        "profiler_stop_calls": 0,
        "trace_normalization_calls": 0,
    }
    assert receipt_module.diag4_profiler_call_audit_payload() == expected
    timing = _diag4_solve_timing_payload()
    assert {name: timing[name] for name in expected} == expected
    producer = _diag4_cold_producer_payload()
    assert {name: producer[name] for name in expected} == expected
    with pytest.raises(ValueError, match="frozen route"):
        receipt_module.diag4_profiler_call_audit_payload(
            receipt_module.Diag4ProfilerCallAudit(False, 1, 0, 0)
        )


def test_diag4_preflight_producer_joins_profiler_call_audit() -> None:
    producer: dict[str, object] = {
        "schema_version": receipt_module.DIAG4_PREFLIGHT_SCHEMA_VERSION,
        "route": receipt_module.DIAG4_ROUTE,
        "numerical_route": receipt_module.DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": (
            receipt_module.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
        ),
        "plan_sha256": receipt_module.DIAG4_PLAN_SHA256,
        "mode": "TRACE_FREE_COMPILE_ONLY",
        "execution_status": "SUCCESS",
        "runtime": {},
        "runtime_evidence": receipt_module._artifact_ref_payload(
            ArtifactRef("preflight/runtime-evidence.json", "1" * 64, 1, "runtime-v1")
        ),
        "base_neq_gntr1_policy_sha256": "2" * 64,
        "policy_evidence": receipt_module._artifact_ref_payload(
            ArtifactRef("preflight/policy.json", "3" * 64, 1, "policy-v1")
        ),
        "problem_sha256": "4" * 64,
        "optimizer_options_sha256": "5" * 64,
        "scaling_sha256": "6" * 64,
        "bootstrap_state_sha256": "7" * 64,
        "initial_physical_state_sha256": "8" * 64,
        "identity_sha256": "9" * 64,
        "source_manifest_sha256": "a" * 64,
        "state_size": receipt_module.STATE_SIZE,
        "equality_size": receipt_module.EQUALITY_SIZE,
        "residual_size": 2110,
        "campaign_authorized": False,
        "solver_dispatched": False,
        "finalizer_called": False,
        "endpoint_audit_called": False,
        "python_callbacks": 0,
        **receipt_module.diag4_profiler_call_audit_payload(),
        "timing": {},
        "failure_reasons": [],
    }
    assert (
        receipt_module.validate_diag4_producer_payload(producer, mode="preflight")
        == producer
    )
    producer["profiler_stop_calls"] = 1
    with pytest.raises(ValueError, match="profiler call audit differs"):
        receipt_module.validate_diag4_producer_payload(producer, mode="preflight")


@pytest.mark.parametrize(
    "mutation",
    (
        "process_pid",
        "process_stop",
        "producer_identity",
        "producer_counter",
        "execution",
    ),
)
def test_diag4_execution_join_mutations_fail_closed(mutation: str) -> None:
    timing = _diag4_solve_timing_payload()
    producer = _diag4_cold_producer_payload()
    process: dict[str, object] = {
        "child_pid": 123,
        "child_start_time_ticks": 456,
        "process_started_monotonic_ns": 1,
        "process_stopped_monotonic_ns": 1_000_000_007,
    }
    timing_bytes = canonical_json_bytes(timing)
    producer["solve_timing_evidence"] = receipt_module._artifact_ref_payload(
        ArtifactRef(
            receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["cold_solve_timing"],
            hashlib.sha256(timing_bytes).hexdigest(),
            len(timing_bytes),
            receipt_module.DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
        )
    )
    supporting = {
        name: ArtifactRef(path, "9" * 64, 1, f"{name}-v1")
        for name, path in receipt_module.DIAG4_EVIDENCE_SLOT_PATHS.items()
        if name not in {"execution", "supervisor_terminal"}
    }
    for field, name in {
        "runtime_evidence": "cold_runtime",
        "policy_evidence": "cold_policy",
        "history_evidence": "cold_history",
        "terminal_numerical_evidence": "cold_terminal_numerical",
        "solve_timing_evidence": "cold_solve_timing",
        "safeguard_telemetry_evidence": "cold_safeguard_telemetry",
    }.items():
        supporting[name] = receipt_module._artifact_ref(
            producer[field], f"test producer.{field}"
        )
    supporting["source_manifest"] = replace(
        supporting["source_manifest"], sha256="8" * 64
    )
    process_bytes = canonical_json_bytes(process)
    supporting["cold_process"] = ArtifactRef(
        receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["cold_process"],
        hashlib.sha256(process_bytes).hexdigest(),
        len(process_bytes),
        "process-v1",
    )
    producer_bytes = canonical_json_bytes(producer)
    supporting["cold_producer"] = ArtifactRef(
        receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["cold_producer"],
        hashlib.sha256(producer_bytes).hexdigest(),
        len(producer_bytes),
        receipt_module.DIAG4_COLD_RESULT_SCHEMA_VERSION,
    )
    execution = receipt_module.diag4_execution_evidence_payload(
        supporting_evidence=supporting,
        solve_timing=timing,
        producer=producer,
        process=process,
    )
    if mutation == "execution":
        changed_execution = dict(execution)
        changed_execution["profiler_start_calls"] = 1
        with pytest.raises(ValueError, match="differs from raw authorities"):
            receipt_module.validate_diag4_execution_evidence_payload(
                changed_execution,
                supporting_evidence=supporting,
                solve_timing=timing,
                producer=producer,
                process=process,
            )
        return
    changed_process = dict(process)
    changed_producer = dict(producer)
    if mutation == "process_pid":
        changed_process["child_pid"] = 18
    elif mutation == "process_stop":
        changed_process["process_stopped_monotonic_ns"] = 1_000_000_006
    elif mutation == "producer_identity":
        changed_producer["identity_sha256"] = "f" * 64
    else:
        changed_producer["profiler_start_calls"] = 1
    with pytest.raises(ValueError):
        receipt_module.diag4_execution_evidence_payload(
            supporting_evidence=supporting,
            solve_timing=timing,
            producer=changed_producer,
            process=changed_process,
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("history", "PENDING_RESULT_INVALID"),
        ("timing", "TIMING_INVALID"),
        ("telemetry", "SAFEGUARD_TELEMETRY_INVALID"),
        ("identity", "NUMERICAL_IDENTITY_MISMATCH"),
        ("terminal_identity", "NUMERICAL_IDENTITY_MISMATCH"),
        ("terminal_observable", "PENDING_RESULT_INVALID"),
    ),
)
def test_diag4_numerical_document_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch, mutation: str, reason: str
) -> None:
    history = _diag4_history()
    monkeypatch.setattr(
        receipt_module, "_parse_history", lambda _value, **_kwargs: history
    )
    timing = _diag4_solve_timing_payload()
    telemetry = _diag4_telemetry_payload()
    terminal = _diag4_structural_terminal_payload()
    producer = _diag4_cold_producer_payload()
    if mutation == "history":
        monkeypatch.setattr(
            receipt_module,
            "_parse_history",
            lambda _value, **_kwargs: (_ for _ in ()).throw(
                ValueError("history invalid")
            ),
        )
    elif mutation == "timing":
        timing["profiler_enabled"] = True
    elif mutation == "telemetry":
        telemetry["total_applied_corrections"] = 1
    elif mutation == "identity":
        producer["identity_sha256"] = "f" * 64
    elif mutation == "terminal_identity":
        terminal["identity_sha256"] = "f" * 64
    else:
        terminal["endpoint_observables"]["G"] = 1.0
    with pytest.raises(receipt_module.Diag4NumericalDocumentError) as captured:
        receipt_module.validate_diag4_numerical_documents(
            history={},
            solve_timing=timing,
            safeguard_telemetry=telemetry,
            terminal_numerical=terminal,
            producer=producer,
        )
    assert captured.value.reason.value == reason
    assert len(captured.value.detail_sha256) == 64


def test_diag4_terminal_selection_uses_stage_then_reason_precedence() -> None:
    candidates = (
        receipt_module.StructuredFailureV4(
            receipt_module.FailureStageV4.SCIENTIFIC,
            receipt_module.FailureReasonCodeV4.QUALITY_HIT,
            "1" * 64,
        ),
        receipt_module.StructuredFailureV4(
            receipt_module.FailureStageV4.NUMERICAL_COMMIT,
            receipt_module.FailureReasonCodeV4.SAFEGUARD_TELEMETRY_INVALID,
            "2" * 64,
        ),
        receipt_module.StructuredFailureV4(
            receipt_module.FailureStageV4.NUMERICAL_COMMIT,
            receipt_module.FailureReasonCodeV4.TIMING_INVALID,
            "3" * 64,
        ),
    )
    assert (
        receipt_module.select_diag4_terminal_outcome(reversed(candidates))
        == (candidates[2])
    )


def test_diag4_terminal_speed_gate_and_trace_alias_rejection(tmp_path: Path) -> None:
    final_root = tmp_path / "artifact"
    staging_root = tmp_path / f"artifact.partial-{'a' * 32}"
    no_hit = receipt_module.build_diag4_supervisor_terminal_payload(
        outcome=receipt_module.StructuredFailureV4(
            receipt_module.FailureStageV4.SCIENTIFIC,
            receipt_module.FailureReasonCodeV4.NO_HIT,
            "1" * 64,
        ),
        launched_children=("preflight", "cold"),
        staging_root=staging_root,
        final_root=final_root,
        nonce="a" * 32,
    )
    assert no_hit["next_route"] == "NOT_PRODUCED"
    assert no_hit["speed_comparison"] == "NOT_PRODUCED"
    quality_hit = receipt_module.build_diag4_supervisor_terminal_payload(
        outcome=receipt_module.StructuredFailureV4(
            receipt_module.FailureStageV4.SCIENTIFIC,
            receipt_module.FailureReasonCodeV4.QUALITY_HIT,
            "2" * 64,
        ),
        launched_children=("preflight", "cold"),
        staging_root=staging_root,
        final_root=final_root,
        nonce="a" * 32,
    )
    assert quality_hit["next_route"] == "CONDITIONAL_ENGINEERING_TIMING"
    assert quality_hit["speed_comparison"] == "CONDITIONAL_ENGINEERING_CONTEXT"
    assert receipt_module._diag4_forbidden_trace_path(
        "cold/raw-trace/plugins/profile/run/file.trace.json.gz"
    )
    assert receipt_module._diag4_forbidden_trace_path("cold/trace-intervals.json")
    assert receipt_module._diag4_forbidden_trace_path(
        "cold/numerical-result/rogue.xplane.pb"
    )
    assert not receipt_module._diag4_forbidden_trace_path(
        "cold/numerical-result/solve-timing.json"
    )


def _diag4_stage_refs(
    outcome: receipt_module.StructuredFailureV4,
) -> dict[str, ArtifactRef | None]:
    present_count = {
        receipt_module.FailureStageV4.AUTHORITY: 0,
        receipt_module.FailureStageV4.SETUP: 0,
        receipt_module.FailureStageV4.BEFORE_PREFLIGHT: 4,
        receipt_module.FailureStageV4.PREFLIGHT: (
            5
            if outcome.reason
            is receipt_module.FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED
            else 12
        ),
        receipt_module.FailureStageV4.BEFORE_COLD: 13,
        receipt_module.FailureStageV4.COLD: (
            13
            if outcome.reason is receipt_module.FailureReasonCodeV4.COLD_LAUNCH_FAILED
            else 20
        ),
        receipt_module.FailureStageV4.NUMERICAL_COMMIT: 20,
        receipt_module.FailureStageV4.RECEIPT: 25,
        receipt_module.FailureStageV4.PUBLICATION: 25,
        receipt_module.FailureStageV4.SCIENTIFIC: 25,
    }[outcome.stage]
    refs: dict[str, ArtifactRef | None] = {}
    for index, (name, path) in enumerate(
        receipt_module.DIAG4_EVIDENCE_SLOT_PATHS.items()
    ):
        refs[name] = (
            ArtifactRef(path, f"{index + 1:064x}", 1, f"{name}-v1")
            if index < present_count or name == "supervisor_terminal"
            else None
        )
    return refs


def _diag4_stage_slots(
    outcome: receipt_module.StructuredFailureV4,
) -> dict[str, receipt_module.EvidenceSlotV4]:
    refs = _diag4_stage_refs(outcome)
    slots: dict[str, receipt_module.EvidenceSlotV4] = {}
    first_absence = True
    for name, reference in refs.items():
        if reference is not None:
            slots[name] = receipt_module.EvidenceSlotV4.present(reference)
        else:
            slots[name] = receipt_module.EvidenceSlotV4.absent(
                outcome.reason if first_absence else None
            )
            first_absence = False
    return slots


def test_diag4_stage_reason_pairing_matrix_is_exact_and_closed() -> None:
    expected = {
        receipt_module.FailureStageV4.AUTHORITY: (
            "AUTHORITY_INVALID",
            "OUTPUT_ROOT_NOT_ABSENT",
            "LOCK_CLAIM_FAILED",
            "IDENTITY_REVALIDATION_FAILED",
            "AUTHORITY_ALREADY_CONSUMED",
        ),
        receipt_module.FailureStageV4.SETUP: (
            "SOURCE_PUBLICATION_FAILED",
            "FROZEN_NUMERICAL_SUBSET_INVALID",
            "NATIVE_REFERENCE_INVALID",
            "POLICY_AUTHORITY_INVALID",
            "SETUP_DEEP_LOAD_FAILED",
        ),
        receipt_module.FailureStageV4.BEFORE_PREFLIGHT: (
            "SUPERVISOR_GPU_OBSERVATION_INVALID",
            "SUPERVISOR_GPU_NONZERO",
            "AUTHORITY_CONSUMPTION_FAILED",
            "AUTHORITY_CONSUMPTION_UNCERTAIN",
        ),
        receipt_module.FailureStageV4.PREFLIGHT: (
            "PREFLIGHT_LAUNCH_FAILED",
            "PREFLIGHT_TIMEOUT",
            "PREFLIGHT_MONITOR_FAILED",
            "PREFLIGHT_EXIT_NONZERO",
            "PREFLIGHT_PROTOCOL_INVALID",
            "PREFLIGHT_PRODUCER_INVALID",
            "PREFLIGHT_GATE_FAILED",
        ),
        receipt_module.FailureStageV4.BEFORE_COLD: (
            "SUPERVISOR_GPU_OBSERVATION_INVALID",
            "SUPERVISOR_GPU_NONZERO",
            "SOURCE_REVALIDATION_FAILED",
            "IDENTITY_REVALIDATION_FAILED",
            "CONSUMPTION_MARKER_INVALID",
        ),
        receipt_module.FailureStageV4.COLD: (
            "COLD_LAUNCH_FAILED",
            "COLD_TIMEOUT",
            "COLD_MONITOR_FAILED",
            "COLD_EXIT_NONZERO",
            "COLD_PROTOCOL_INVALID",
            "COLD_PRODUCER_INVALID",
        ),
        receipt_module.FailureStageV4.NUMERICAL_COMMIT: (
            "PENDING_RESULT_ABSENT",
            "TIMING_INVALID",
            "SAFEGUARD_TELEMETRY_INVALID",
            "NUMERICAL_IDENTITY_MISMATCH",
            "QUARANTINE_FAILED",
            "PENDING_RESULT_INVALID",
            "COMMIT_COLLISION",
            "COMMIT_RENAME_FAILED",
            "COMMIT_FSYNC_FAILED",
            "COMMITTED_DEEP_LOAD_FAILED",
        ),
        receipt_module.FailureStageV4.RECEIPT: (
            "EVIDENCE_VECTOR_INVALID",
            "GROUP_PREFIX_INVALID",
            "SCIENTIFIC_RECONSTRUCTION_FAILED",
            "RECEIPT_SCHEMA_INVALID",
        ),
        receipt_module.FailureStageV4.PUBLICATION: (
            "MANIFEST_INVALID",
            "MODE_OR_LINK_INVALID",
            "STAGING_DEEP_LOAD_FAILED",
            "FINAL_COLLISION",
            "FINAL_RENAME_FAILED",
        ),
        receipt_module.FailureStageV4.SCIENTIFIC: (
            "INCOMPLETE",
            "NO_HIT",
            "QUALITY_HIT",
        ),
    }
    assert tuple(expected) == receipt_module.DIAG4_FAILURE_STAGE_ORDER
    for stage in receipt_module.FailureStageV4:
        assert (
            tuple(
                reason.value
                for reason in receipt_module.DIAG4_STAGE_REASON_ORDER[stage]
            )
            == expected[stage]
        )
        for reason in receipt_module.FailureReasonCodeV4:
            outcome = receipt_module.StructuredFailureV4(stage, reason, "f" * 64)
            if reason.value in expected[stage]:
                payload = receipt_module.diag4_terminal_outcome_payload(outcome)
                assert receipt_module.parse_diag4_terminal_outcome(payload) == outcome
            else:
                with pytest.raises(ValueError, match="stage/reason pairing differs"):
                    receipt_module.diag4_terminal_outcome_payload(outcome)


@pytest.mark.parametrize(
    "physical_failure", ("FINAL_FSYNC_FAILED", "FINAL_DEEP_LOAD_FAILED")
)
def test_diag4_physical_publication_failures_are_not_terminal_reason_codes(
    physical_failure: str,
) -> None:
    with pytest.raises(ValueError):
        receipt_module.parse_diag4_terminal_outcome(
            {
                "stage": receipt_module.FailureStageV4.PUBLICATION.value,
                "reason": {
                    "code": physical_failure,
                    "detail_sha256": "f" * 64,
                },
            }
        )


def test_diag4_every_stage_reason_has_exact_vector_and_child_prefix(
    tmp_path: Path,
) -> None:
    for stage in receipt_module.DIAG4_FAILURE_STAGE_ORDER:
        for reason in receipt_module.DIAG4_STAGE_REASON_ORDER[stage]:
            outcome = receipt_module.StructuredFailureV4(stage, reason, "f" * 64)
            slots = _diag4_stage_slots(outcome)
            receipt_module._validate_diag4_stage_vector(slots, failure=outcome)
            assert tuple(slots) == tuple(receipt_module.DIAG4_EVIDENCE_SLOT_PATHS)
            absent = [slot for slot in slots.values() if slot.artifact is None]
            if absent:
                assert absent[0].reason is reason
                assert all(slot.reason is None for slot in absent[1:])
            terminal = receipt_module.build_diag4_supervisor_terminal_payload(
                outcome=outcome,
                launched_children=receipt_module._diag4_expected_launched_children(
                    outcome
                ),
                staging_root=tmp_path.with_name(f"{tmp_path.name}.partial-{'a' * 32}"),
                final_root=tmp_path,
                nonce="a" * 32,
            )
            assert (
                receipt_module.parse_diag4_supervisor_terminal_payload(terminal)[1]
                == outcome
            )


@pytest.mark.parametrize(
    "mutation", ("order", "first_reason", "later_reason", "later_group", "subgroup")
)
def test_diag4_vector_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    outcome = receipt_module.StructuredFailureV4(
        receipt_module.FailureStageV4.COLD,
        receipt_module.FailureReasonCodeV4.COLD_TIMEOUT,
        "f" * 64,
    )
    refs = _diag4_stage_refs(outcome)
    if mutation == "order":
        reordered = {name: refs[name] for name in reversed(tuple(refs))}
        with pytest.raises(ValueError, match="frozen slot schema"):
            receipt_module.derive_diag4_evidence_slots(
                artifact_root=tmp_path, artifact_refs=reordered, outcome=outcome
            )
        return
    slots = _diag4_stage_slots(outcome)
    mutated = dict(slots)
    absent_names = [name for name, slot in mutated.items() if slot.artifact is None]
    if mutation == "first_reason":
        mutated[absent_names[0]] = receipt_module.EvidenceSlotV4.absent()
    elif mutation == "later_reason":
        mutated[absent_names[1]] = receipt_module.EvidenceSlotV4.absent(outcome.reason)
    elif mutation == "later_group":
        name = "cold_history"
        mutated[name] = receipt_module.EvidenceSlotV4.present(
            ArtifactRef(
                receipt_module.DIAG4_EVIDENCE_SLOT_PATHS[name],
                "a" * 64,
                1,
                "history-v1",
            )
        )
    else:
        for name in (
            "cold_history",
            "cold_terminal_numerical",
            "cold_solve_timing",
            "cold_safeguard_telemetry",
        ):
            mutated[name] = receipt_module.EvidenceSlotV4.absent()
        name = "cold_history"
        mutated[name] = receipt_module.EvidenceSlotV4.present(
            ArtifactRef(
                receipt_module.DIAG4_EVIDENCE_SLOT_PATHS[name],
                "a" * 64,
                1,
                "history-v1",
            )
        )
    with pytest.raises(ValueError):
        receipt_module._validate_diag4_stage_vector(mutated, failure=outcome)


@pytest.mark.parametrize(
    "legacy_schema",
    (
        receipt_module.SCHEMA_VERSION,
        receipt_module.DIAG2_SCHEMA_VERSION,
        receipt_module.DIAG3_SCHEMA_VERSION,
    ),
)
def test_diag4_loader_cross_rejects_every_legacy_schema(
    tmp_path: Path, legacy_schema: str
) -> None:
    payload = {
        "schema_version": legacy_schema,
        "route": receipt_module.DIAG4_ROUTE,
        "numerical_route": receipt_module.DIAG4_NUMERICAL_ROUTE,
        "plan_sha256": receipt_module.DIAG4_PLAN_SHA256,
        "evidence_slots": {},
        "verdict": "DIAGNOSTIC_INCOMPLETE",
        "historical_relation": "NOT_COMPARABLE_INCOMPLETE",
        "quality": None,
        "phase_attribution": {"status": "NOT_PRODUCED"},
        "next_route": "NOT_PRODUCED",
        "speed_comparison": "NOT_PRODUCED",
        "terminal_outcome": {
            "stage": "RECEIPT",
            "reason": {"code": "RECEIPT_SCHEMA_INVALID", "detail_sha256": "1" * 64},
        },
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }
    with pytest.raises(ValueError, match="identity differs"):
        receipt_module.diag4_diagnostic_receipt_from_payload(
            payload, artifact_root=tmp_path
        )


@pytest.mark.parametrize(
    "loader",
    (
        receipt_module.diagnostic_receipt_from_payload,
        receipt_module.diag2_diagnostic_receipt_from_payload,
        receipt_module.diag3_diagnostic_receipt_from_payload,
    ),
)
def test_every_legacy_loader_rejects_diag4_schema(
    tmp_path: Path, loader: Callable[..., object]
) -> None:
    payload = {
        "schema_version": receipt_module.DIAG4_SCHEMA_VERSION,
        "route": receipt_module.DIAG4_ROUTE,
        "numerical_route": receipt_module.DIAG4_NUMERICAL_ROUTE,
        "plan_sha256": receipt_module.DIAG4_PLAN_SHA256,
        "evidence_slots": {},
        "verdict": "DIAGNOSTIC_INCOMPLETE",
        "historical_relation": "NOT_COMPARABLE_INCOMPLETE",
        "quality": None,
        "phase_attribution": {"status": "NOT_PRODUCED"},
        "next_route": "NOT_PRODUCED",
        "speed_comparison": "NOT_PRODUCED",
        "terminal_outcome": {
            "stage": "RECEIPT",
            "reason": {"code": "RECEIPT_SCHEMA_INVALID", "detail_sha256": "1" * 64},
        },
        "promotion_authorized": False,
        "formal_comparison": "NOT_PRODUCED",
    }
    payload["schema_version"] = receipt_module.DIAG4_SCHEMA_VERSION
    with pytest.raises(ValueError, match="diagnostic receipt keys differ"):
        loader(payload, artifact_root=tmp_path)


def _diag4_incomplete_artifact(
    root: Path, *, staging: bool = False
) -> receipt_module.DiagnosticReceiptV4:
    root.mkdir()
    outcome = receipt_module.StructuredFailureV4(
        receipt_module.FailureStageV4.AUTHORITY,
        receipt_module.FailureReasonCodeV4.AUTHORITY_INVALID,
        "1" * 64,
    )
    terminal_path = (
        root / receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["supervisor_terminal"]
    )
    terminal_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_path.write_bytes(
        canonical_json_bytes(
            receipt_module.build_diag4_supervisor_terminal_payload(
                outcome=outcome,
                launched_children=(),
                staging_root=(
                    root
                    if staging
                    else root.with_name(f"{root.name}.partial-{'0' * 32}")
                ),
                final_root=(
                    root.with_name(root.name.removesuffix(f".partial-{'0' * 32}"))
                    if staging
                    else root
                ),
                nonce="0" * 32,
            )
        )
    )
    artifact_refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt_module.DIAG4_EVIDENCE_SLOT_PATHS
    }
    artifact_refs["supervisor_terminal"] = _artifact_ref(
        terminal_path,
        root,
        receipt_module.DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    slots = receipt_module.derive_diag4_evidence_slots(
        artifact_root=root,
        artifact_refs=artifact_refs,
        outcome=outcome,
    )
    built = receipt_module.build_diag4_diagnostic_receipt(
        artifact_root=root, evidence_slots=slots
    )
    (root / receipt_module.DIAG2_RECEIPT_FILENAME).write_bytes(
        receipt_module.diag4_diagnostic_receipt_bytes(built)
    )
    return built


def test_diag4_sealed_staging_and_final_deep_load(tmp_path: Path) -> None:
    staging = tmp_path / f"diag4.partial-{'0' * 32}"
    built = _diag4_incomplete_artifact(staging, staging=True)
    manifest_path = staging / receipt_module.DIAG2_MANIFEST_FILENAME
    manifest_path.write_bytes(
        canonical_json_bytes(receipt_module.diag4_artifact_manifest_payload(staging))
    )
    assert receipt_module.validate_diag4_writable_staging(staging) == built
    for path in staging.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(staging, 0o555)
    assert receipt_module.load_and_validate_diag4_staging(staging) == built


@pytest.mark.parametrize(
    "mutation",
    (
        "receipt_claim",
        "manifest_role",
        "manifest_digest",
        "sealed_mode",
        "extra_path",
        "relocation",
        "opaque_quarantine",
        "trace_alias",
    ),
)
def test_diag4_receipt_manifest_and_deep_loader_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "diag4-final"
    built = _diag4_incomplete_artifact(root)
    receipt_path = root / receipt_module.DIAG2_RECEIPT_FILENAME
    manifest_path = root / receipt_module.DIAG2_MANIFEST_FILENAME

    if mutation == "receipt_claim":
        payload = json.loads(receipt_path.read_bytes())
        payload["verdict"] = "DIAGNOSTIC_COMPLETE_NO_HIT"
        with pytest.raises(ValueError, match="claims differ from raw evidence"):
            receipt_module.diag4_diagnostic_receipt_from_payload(
                payload, artifact_root=root
            )
        return
    if mutation == "trace_alias":
        alias = root / "cold/raw-trace/plugins/profile/rogue.xplane.pb"
        alias.parent.mkdir(parents=True)
        alias.write_bytes(b"forbidden")
        with pytest.raises(ValueError, match="forbidden trace evidence"):
            receipt_module.diag4_artifact_manifest_payload(root)
        return
    if mutation == "extra_path":
        extra = root / "cold/numerical-result/unknown.bin"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"unknown")
        with pytest.raises(ValueError, match="unknown path"):
            receipt_module.diag4_artifact_manifest_payload(root)
        return
    if mutation == "opaque_quarantine":
        opaque = root / "cold/uncommitted-numerical-result/opaque.bin"
        opaque.parent.mkdir(parents=True)
        opaque.write_bytes(b"opaque")
        with pytest.raises(ValueError, match="contradicts terminal outcome"):
            receipt_module.diag4_artifact_manifest_payload(root)
        return

    manifest_path.write_bytes(
        canonical_json_bytes(receipt_module.diag4_artifact_manifest_payload(root))
    )
    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
    assert receipt_module.load_and_validate_diag4_artifact(root) == built
    if mutation == "relocation":
        relocated = tmp_path / "relocated"
        shutil.copytree(root, relocated)
        with pytest.raises(ValueError, match="publication.final_root"):
            receipt_module.load_and_validate_diag4_artifact(relocated)
        return
    if mutation == "sealed_mode":
        os.chmod(receipt_path, 0o644)
        with pytest.raises(ValueError, match="modes differ"):
            receipt_module.load_and_validate_diag4_artifact(root)
        return
    os.chmod(manifest_path, 0o644)
    manifest = json.loads(manifest_path.read_bytes())
    terminal_entry = next(
        entry
        for entry in manifest["entries"]
        if entry["relative_path"]
        == receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["supervisor_terminal"]
    )
    if mutation == "manifest_role":
        terminal_entry["role"] = "cold_terminal"
    else:
        terminal_entry["sha256"] = "f" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    os.chmod(manifest_path, 0o444)
    with pytest.raises(
        ValueError,
        match="role differs" if mutation == "manifest_role" else "bytes differ",
    ):
        receipt_module.load_and_validate_diag4_artifact(root)


def test_diag4_final_loader_deep_parses_present_setup_without_cold_producer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "diag4-setup-failure"
    root.mkdir()
    source_path = root / receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["source_manifest"]
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(canonical_json_bytes({"malformed": True}))
    outcome = receipt_module.StructuredFailureV4(
        receipt_module.FailureStageV4.SETUP,
        receipt_module.FailureReasonCodeV4.FROZEN_NUMERICAL_SUBSET_INVALID,
        "f" * 64,
    )
    terminal_path = (
        root / receipt_module.DIAG4_EVIDENCE_SLOT_PATHS["supervisor_terminal"]
    )
    terminal_path.write_bytes(
        canonical_json_bytes(
            receipt_module.build_diag4_supervisor_terminal_payload(
                outcome=outcome,
                launched_children=(),
                staging_root=root.with_name(f"{root.name}.partial-{'a' * 32}"),
                final_root=root,
                nonce="a" * 32,
            )
        )
    )
    slots: dict[str, receipt_module.EvidenceSlotV4] = {}
    first_absence = True
    for name in receipt_module.DIAG4_EVIDENCE_SLOT_PATHS:
        if name == "source_manifest":
            slots[name] = receipt_module.EvidenceSlotV4.present(
                _artifact_ref(source_path, root, "malformed-source-v1")
            )
        elif name == "supervisor_terminal":
            slots[name] = receipt_module.EvidenceSlotV4.present(
                _artifact_ref(
                    terminal_path,
                    root,
                    receipt_module.DIAG4_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
                )
            )
        else:
            slots[name] = receipt_module.EvidenceSlotV4.absent(
                outcome.reason if first_absence else None
            )
            first_absence = False
    receipt = receipt_module.DiagnosticReceiptV4(
        tuple(slots.items()),
        "DIAGNOSTIC_INCOMPLETE",
        "NOT_COMPARABLE_INCOMPLETE",
        None,
        {"status": "NOT_PRODUCED"},
        "NOT_PRODUCED",
        "NOT_PRODUCED",
        outcome,
    )
    (root / receipt_module.DIAG2_RECEIPT_FILENAME).write_bytes(
        receipt_module.diag4_diagnostic_receipt_bytes(receipt)
    )
    (root / receipt_module.DIAG2_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(receipt_module.diag4_artifact_manifest_payload(root))
    )
    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
    with pytest.raises(ValueError):
        receipt_module.load_and_validate_diag4_artifact(root)


def test_diag4_cold_classifier_uses_v4_failure_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = {
        name: ArtifactRef(path, "1" * 64, 1, "test-v1")
        for name, path in receipt_module.DIAG4_EVIDENCE_SLOT_PATHS.items()
    }
    monkeypatch.setattr(
        receipt_module,
        "_load_ref_json",
        lambda *_args, **_kwargs: {"not": "a producer"},
    )
    classified = receipt_module.classify_diag4_cold_evidence(
        tmp_path, artifact_refs=refs
    )
    assert isinstance(classified, receipt_module.Diag4ColdEvidenceClassification)
    assert classified.outcome == receipt_module.StructuredFailureV4(
        receipt_module.FailureStageV4.COLD,
        receipt_module.FailureReasonCodeV4.COLD_PRODUCER_INVALID,
        classified.outcome.detail_sha256 if classified.outcome is not None else "",
    )


def test_diag4_preflight_gate_binds_independent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = {
        name: ArtifactRef(path, "1" * 64, 1, "test-v1")
        for name, path in receipt_module.DIAG4_EVIDENCE_SLOT_PATHS.items()
    }
    slots = {
        name: receipt_module.EvidenceSlotV4.present(reference)
        for name, reference in refs.items()
    }
    identity = {
        "problem_sha256": "2" * 64,
        "optimizer_options_sha256": "3" * 64,
        "base_neq_gntr1_policy_sha256": "4" * 64,
        "scaling_sha256": "5" * 64,
        "bootstrap_state_sha256": "6" * 64,
        "initial_physical_state_sha256": "7" * 64,
        "identity_sha256": "8" * 64,
    }
    producer: dict[str, object] = {
        **identity,
        "runtime_evidence": receipt_module._artifact_ref_payload(
            refs["preflight_runtime"]
        ),
        "policy_evidence": receipt_module._artifact_ref_payload(
            refs["preflight_policy"]
        ),
        "source_manifest_sha256": "1" * 64,
        "runtime": {},
    }
    policy = receipt_module.PolicyEvidence(
        "4" * 64,
        "9" * 64,
        "a" * 64,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        (),
        (),
    )
    monkeypatch.setattr(
        receipt_module,
        "load_snapshot",
        lambda _root, **_kwargs: SimpleNamespace(
            manifest_sha256="1" * 64, root=tmp_path, entries=()
        ),
    )
    monkeypatch.setattr(receipt_module, "REQUIRED_SOURCE_ROLE_BINDINGS", {})
    monkeypatch.setattr(
        receipt_module,
        "validate_diag4_frozen_numerical_subset_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        receipt_module, "validate_native_equivalent_reference", lambda *_args: None
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_diag2_supervisor_zero_payload",
        lambda *_args, **_kwargs: {"captured_at_monotonic_ns": 1},
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_diag2_policy_authority_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_diag4_producer_payload",
        lambda *_args, **_kwargs: producer,
    )
    monkeypatch.setattr(receipt_module, "_parse_policy", lambda *_args: policy)
    monkeypatch.setattr(receipt_module, "_diag2_policy_evidence", lambda *_args: policy)
    monkeypatch.setattr(receipt_module, "_resolve_artifact", lambda *_args: tmp_path)
    monkeypatch.setattr(
        receipt_module,
        "validate_runtime_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(
            observation=SimpleNamespace(
                runtime_identity=SimpleNamespace(
                    backend="gpu",
                    device_uuid=GPU_UUID,
                    python_executable="/python",
                )
            )
        ),
    )
    monkeypatch.setattr(
        receipt_module,
        "_runtime_mapping",
        lambda *_args: {
            "backend": "gpu",
            "device_uuid": GPU_UUID,
            "jax_enable_x64": True,
        },
    )
    monkeypatch.setattr(
        receipt_module,
        "_diag2_child_documents",
        lambda *_args, **_kwargs: (
            {"terminal_status": "COMPLETE"},
            {
                "argv": ["/python", "worker.py"],
                "child_pid": 12,
                "child_start_time_ticks": 13,
                "process_started_monotonic_ns": 2,
            },
            "NONE",
            0,
        ),
    )
    monkeypatch.setattr(
        receipt_module, "_validate_memory", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        receipt_module,
        "_load_ref_json",
        lambda _root, reference, _context: (
            {"peak_memory_bytes": 1, "peak_memory_fraction": 0.1}
            if reference == refs["preflight_memory"]
            else {}
        ),
    )
    assert receipt_module.validate_diag4_preflight_gate(
        tmp_path,
        evidence_slots=slots,
        expected_gpu_uuid=GPU_UUID,
        physical_memory_bytes=10,
        expected_interpreter="/python",
        expected_argv=("/python", "worker.py"),
        expected_identity=identity,
        expected_frozen_numerical_entries={"src/core.py": "a" * 64},
    )
    mutated_identity = dict(identity)
    mutated_identity["problem_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from authority"):
        receipt_module.validate_diag4_preflight_gate(
            tmp_path,
            evidence_slots=slots,
            expected_gpu_uuid=GPU_UUID,
            physical_memory_bytes=10,
            expected_interpreter="/python",
            expected_argv=("/python", "worker.py"),
            expected_identity=mutated_identity,
            expected_frozen_numerical_entries={"src/core.py": "a" * 64},
        )


def test_diag4_frozen_subset_is_authority_supplied_and_source_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "benchmarks/route.py": "a" * 64,
        "src/simsopt_jax/core.py": "b" * 64,
        "tests/route_test.py": "c" * 64,
    }
    payload = receipt_module.build_diag4_frozen_numerical_subset_payload(expected)
    monkeypatch.setattr(
        receipt_module,
        "load_snapshot",
        lambda _root, **_kwargs: SimpleNamespace(
            entries=tuple(
                SimpleNamespace(
                    relative_path=path,
                    sha256=digest,
                    role=(
                        "benchmark"
                        if path.startswith("benchmarks/")
                        else "test"
                        if path.startswith("tests/")
                        else "execution_source"
                    ),
                )
                for path, digest in expected.items()
            )
        ),
    )
    assert (
        receipt_module.validate_diag4_frozen_numerical_subset_payload(
            payload, artifact_root=tmp_path, expected_entries=expected
        )
        == payload
    )
    changed_authority = dict(expected)
    changed_authority["src/simsopt_jax/core.py"] = "d" * 64
    with pytest.raises(ValueError, match="differs from authority"):
        receipt_module.validate_diag4_frozen_numerical_subset_payload(
            payload,
            artifact_root=tmp_path,
            expected_entries=changed_authority,
        )
    changed_payload = json.loads(json.dumps(payload))
    changed_payload["plan_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="not canonical"):
        receipt_module.validate_diag4_frozen_numerical_subset_payload(
            changed_payload,
            artifact_root=tmp_path,
            expected_entries=expected,
        )


def test_diag2_policy_authority_rederives_from_native_reference_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    native = np.ascontiguousarray(_copy_validated_reference(root), dtype="<f8")
    document = json.loads((root / "native-reference/reference.json").read_bytes())
    reference_volume = float(document["evidence"]["observables"]["volume"])
    volume_target = float(reference_volume - native[254])
    scale = np.empty(receipt_module.EQUALITY_SIZE, dtype="<f8")
    scale[:254] = 1.0 / np.sqrt(np.float64(254.0))
    scale[254] = 1.0 / abs(volume_target)
    reference = _artifact_ref(
        root / "native-reference/reference.json",
        root,
        document["schema_version"],
    )
    payload = receipt_module.build_diag2_policy_authority_payload(
        native_reference=reference,
        reference_volume=reference_volume,
        volume_target=volume_target,
        native_raw_equalities=native,
        constraint_inverse_scale=scale,
    )
    assert (
        receipt_module.validate_diag2_policy_authority_payload(
            payload, artifact_root=root
        )
        == payload
    )

    changed = native.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    mutated = json.loads(json.dumps(payload))
    mutated["native_raw_equalities"] = changed.tolist()
    mutated["native_raw_equalities_sha256"] = exact_numeric_tree_sha256(changed)
    identity = (
        "single-stage-native-equivalent-quality-policy-v1",
        "single-stage-fullspace-neq-gntr1-result-v1",
        receipt_module.NUMERICAL_ROUTE,
        changed,
        mutated["native_raw_equalities_sha256"],
        scale,
        receipt_module.OBJECTIVE_MAXIMUM,
        receipt_module.STATE_SIZE,
        receipt_module.EQUALITY_SIZE,
        2110,
        receipt_module.RAW_EQUALITY_ABSOLUTE_TOLERANCE,
        receipt_module.RAW_EQUALITY_RELATIVE_TOLERANCE,
        receipt_module.FEASIBILITY_MAXIMUM,
        receipt_module.RESIDUAL_VALUE_DEFECT_MAXIMUM,
        receipt_module.RESIDUAL_GRADIENT_DEFECT_MAXIMUM,
        receipt_module.TRANSPOSE_DEFECT_MAXIMUM,
        tuple(
            (name, receipt_module.FROZEN_GNTR_OPTIONS[name])
            for name in receipt_module.GNTR_OPTION_ORDER
        ),
    )
    mutated["policy_sha256"] = exact_numeric_tree_sha256(identity)
    with pytest.raises(ValueError, match="differs from NumPy reconstruction"):
        receipt_module.validate_diag2_policy_authority_payload(
            mutated, artifact_root=root
        )

    unknown = dict(payload)
    unknown["caller_extension"] = False
    with pytest.raises(ValueError, match="keys differ"):
        receipt_module.validate_diag2_policy_authority_payload(
            unknown, artifact_root=root
        )

    for field in ("reference_volume", "volume_target"):
        boundary = dict(payload)
        boundary[field] = np.nextafter(float(payload[field]), np.inf)
        with pytest.raises(ValueError, match="differs from NumPy reconstruction"):
            receipt_module.validate_diag2_policy_authority_payload(
                boundary, artifact_root=root
            )

    boolean_scalar = dict(payload)
    boolean_scalar["reference_volume"] = True
    with pytest.raises(ValueError, match="differs from NumPy reconstruction"):
        receipt_module.validate_diag2_policy_authority_payload(
            boolean_scalar, artifact_root=root
        )

    for field in ("native_raw_equalities", "constraint_inverse_scale"):
        boolean_coordinate = json.loads(json.dumps(payload))
        boolean_coordinate[field][0] = True
        with pytest.raises(ValueError, match="differs from NumPy reconstruction"):
            receipt_module.validate_diag2_policy_authority_payload(
                boolean_coordinate, artifact_root=root
            )


def test_diag2_cold_classifier_returns_first_missing_typed_slot(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    dummy = {
        "relative_path": "cold/runtime-evidence.json",
        "sha256": "1" * 64,
        "size_bytes": 1,
        "schema_version": "runtime-v1",
    }
    payload = {
        "schema_version": f"{receipt_module.SCHEMA_VERSION}-producer",
        "route": receipt_module.DIAG2_ROUTE,
        "plan_sha256": receipt_module.DIAG2_PLAN_SHA256,
        "execution_status": "COMPLETE",
        "runtime": {},
        "runtime_evidence": dummy,
        "policy_sha256": "2" * 64,
        "phase_schema_sha256": receipt_module.PHASE_SCHEMA_SHA256,
        "history_evidence": dummy,
        "terminal_numerical_evidence": dummy,
        "policy_evidence": dummy,
        "raw_trace_evidence": dummy,
        "trace_intervals_evidence": dummy,
        "timestamps_ns": {},
        "transfer_audit": {},
        "endpoint_audit_called": False,
        "campaign_authorized": False,
        "failure_reasons": [],
    }
    producer_path = root / "cold/producer.json"
    producer_path.parent.mkdir(parents=True)
    producer_path.write_bytes(canonical_json_bytes(payload))
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    refs["cold_producer"] = _artifact_ref(
        producer_path, root, f"{receipt_module.SCHEMA_VERSION}-producer"
    )
    classification = receipt_module.classify_diag2_cold_evidence(
        root, artifact_refs=refs
    )
    assert classification.typed_slots == ()
    assert classification.offending_slot == "cold_runtime"
    assert classification.failure is not None
    assert (
        classification.failure.reason
        is receipt_module.FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID
    )


def test_diag2_evidence_slot_is_an_exact_discriminated_union() -> None:
    reference = ArtifactRef(
        "cold/history.json", "1" * 64, 1, f"{receipt_module.SCHEMA_VERSION}-history"
    )
    present = receipt_module.parse_diag2_evidence_slot(
        {
            "state": "PRESENT",
            "artifact": receipt_module._artifact_ref_payload(reference),
        },
        name="cold_history",
    )
    assert present == receipt_module.EvidenceSlot.present(reference)
    absent = receipt_module.parse_diag2_evidence_slot(
        {"state": "ABSENT", "reason": "CHILD_TIMEOUT"}, name="cold_history"
    )
    assert absent == receipt_module.EvidenceSlot.absent(
        receipt_module.AbsenceReason.CHILD_TIMEOUT
    )
    with pytest.raises(ValueError, match="keys differ"):
        receipt_module.parse_diag2_evidence_slot(
            {"state": "ABSENT", "reason": "CHILD_TIMEOUT", "artifact": None},
            name="cold_history",
        )


def test_diag2_supervisor_zero_is_derived_from_raw_queries(tmp_path: Path) -> None:
    root = tmp_path / f"artifact.partial-{'0' * 32}"
    root.mkdir()
    inventory_stdout = _diag2_raw_ref(
        root / "supervisor/before-preflight-gpu-inventory.stdout.bin",
        root,
        "raw-supervisor-gpu-inventory-stdout-v1",
        f"GPU-other, 16384\n{GPU_UUID}, 33554432\n".encode(),
    )
    inventory_stderr = _diag2_raw_ref(
        root / "supervisor/before-preflight-gpu-inventory.stderr.bin",
        root,
        "raw-supervisor-gpu-inventory-stderr-v1",
        b"",
    )
    compute_stdout = _diag2_raw_ref(
        root / "supervisor/before-preflight-compute-apps.stdout.bin",
        root,
        "raw-supervisor-compute-apps-stdout-v1",
        b"",
    )
    compute_stderr = _diag2_raw_ref(
        root / "supervisor/before-preflight-compute-apps.stderr.bin",
        root,
        "raw-supervisor-compute-apps-stderr-v1",
        b"",
    )
    inventory = receipt_module.SupervisorQueryV2(
        receipt_module._DIAG2_GPU_INVENTORY_ARGV,
        "2" * 64,
        True,
        False,
        0,
        inventory_stdout,
        inventory_stderr,
    )
    compute = receipt_module.SupervisorQueryV2(
        receipt_module._DIAG2_COMPUTE_APPS_ARGV,
        "2" * 64,
        True,
        False,
        0,
        compute_stdout,
        compute_stderr,
    )
    payload = receipt_module.build_diag2_supervisor_zero_payload(
        stage="BEFORE_PREFLIGHT",
        captured_at_monotonic_ns=1,
        captured_at_unix_ns=2,
        supervisor_pid=17,
        supervisor_start_ticks=3,
        gpu_uuid=GPU_UUID,
        visible_device=GPU_UUID,
        gpu_inventory_query=inventory,
        compute_apps_query=compute,
        matching_rows=(),
    )
    assert (
        receipt_module.validate_diag2_supervisor_zero_payload(
            payload, artifact_root=root, expected_stage="BEFORE_PREFLIGHT"
        )
        == payload
    )
    swapped = json.loads(json.dumps(payload))
    swapped["gpu_inventory_query"], swapped["compute_apps_query"] = (
        swapped["compute_apps_query"],
        swapped["gpu_inventory_query"],
    )
    with pytest.raises(ValueError, match="raw query binding differs"):
        receipt_module.validate_diag2_supervisor_zero_payload(
            swapped, artifact_root=root, expected_stage="BEFORE_PREFLIGHT"
        )

    nonzero = json.loads(json.dumps(payload))
    nonzero["compute_apps_query"]["returncode"] = 1
    with pytest.raises(ValueError, match="compute-apps query did not succeed"):
        receipt_module.validate_diag2_supervisor_zero_payload(
            nonzero, artifact_root=root, expected_stage="BEFORE_PREFLIGHT"
        )

    invalid_state = json.loads(json.dumps(payload))
    invalid_state["compute_apps_query"]["timed_out"] = True
    with pytest.raises(ValueError, match="timed-out query must have null return code"):
        receipt_module.validate_diag2_supervisor_zero_payload(
            invalid_state, artifact_root=root, expected_stage="BEFORE_PREFLIGHT"
        )

    compute_stdout_path = root / compute_stdout.relative_path
    compute_stdout_path.write_text(f"17, {GPU_UUID}, 0\n", encoding="utf-8")
    changed = _diag2_ref(
        compute_stdout_path,
        root,
        "raw-supervisor-compute-apps-stdout-v1",
    )
    mutated = json.loads(json.dumps(payload))
    mutated["compute_apps_query"]["stdout"] = receipt_module._artifact_ref_payload(
        changed
    )
    with pytest.raises(ValueError, match="matching rows differ"):
        receipt_module.validate_diag2_supervisor_zero_payload(
            mutated, artifact_root=root, expected_stage="BEFORE_PREFLIGHT"
        )

    compute_stdout_path.write_bytes(b"")
    inventory_stdout_path = root / inventory_stdout.relative_path
    inventory_stdout_path.write_bytes(b"malformed-success-row\n")
    malformed = json.loads(json.dumps(payload))
    malformed["gpu_inventory_query"]["stdout"] = receipt_module._artifact_ref_payload(
        _diag2_ref(
            inventory_stdout_path,
            root,
            "raw-supervisor-gpu-inventory-stdout-v1",
        )
    )
    with pytest.raises(ValueError, match="GPU inventory row 0 is malformed"):
        receipt_module.validate_diag2_supervisor_zero_payload(
            malformed, artifact_root=root, expected_stage="BEFORE_PREFLIGHT"
        )


def test_diag2_verdict_biconditional_is_rebuilt_from_failure(tmp_path: Path) -> None:
    root = tmp_path / f"artifact.partial-{'0' * 32}"
    root.mkdir()
    final_root = tmp_path / "artifact"
    failure = receipt_module.StructuredFailureV2(
        receipt_module.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt_module.FailureReasonCodeV2.SOURCE_PRE,
        "4" * 64,
    )
    terminal_payload = receipt_module.build_diag2_supervisor_terminal_payload(
        disposition="INCOMPLETE",
        failure=failure,
        launched_children=(),
        policy_authority_produced=False,
        preflight_authorized=False,
        cold_authorized=False,
        staging_root=root,
        final_root=final_root,
        nonce="0" * 32,
        algorithm_route_selection="NOT_PRODUCED",
    )
    terminal_path = root / "supervisor-terminal.json"
    terminal_path.write_bytes(canonical_json_bytes(terminal_payload))
    terminal_ref = _diag2_ref(
        terminal_path, root, receipt_module.DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION
    )
    slots = {
        name: receipt_module.EvidenceSlot.absent(
            receipt_module.AbsenceReason.SOURCE_PRE
            if name == "source_manifest"
            else receipt_module.AbsenceReason.NOT_REACHED
        )
        for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    slots["supervisor_terminal"] = receipt_module.EvidenceSlot.present(terminal_ref)
    receipt = receipt_module.build_diag2_diagnostic_receipt(
        artifact_root=root, evidence_slots=slots
    )
    assert receipt.verdict == "DIAGNOSTIC_INCOMPLETE"
    payload = receipt_module.diag2_diagnostic_receipt_payload(receipt)
    payload["verdict"] = "DIAGNOSTIC_COMPLETE_NO_HIT"
    with pytest.raises(ValueError, match="claims differ"):
        receipt_module.diag2_diagnostic_receipt_from_payload(
            payload, artifact_root=root
        )


@pytest.mark.parametrize(
    ("quality_hit", "expected_verdict", "mutated_verdict"),
    (
        (
            False,
            "DIAGNOSTIC_COMPLETE_NO_HIT",
            "DIAGNOSTIC_COMPLETE_QUALITY_HIT",
        ),
        (
            True,
            "DIAGNOSTIC_COMPLETE_QUALITY_HIT",
            "DIAGNOSTIC_COMPLETE_NO_HIT",
        ),
    ),
)
def test_diag2_complete_fixture_recomputes_all_claim_biconditionals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality_hit: bool,
    expected_verdict: str,
    mutated_verdict: str,
) -> None:
    root = tmp_path / f"diag2.partial-{'0' * 32}"
    built = _complete_receipt(
        root,
        monkeypatch,
        quality_hit=quality_hit,
        real_authorities=True,
        diag2=True,
    )
    assert isinstance(built, receipt_module.DiagnosticReceiptV2)
    assert built.verdict == expected_verdict
    mutations: dict[str, object] = {
        "verdict": mutated_verdict,
        "failure": {
            "stage": receipt_module.FailureStageV2.SOURCE_PUBLICATION_FAILURE.value,
            "reason": {
                "code": receipt_module.FailureReasonCodeV2.SOURCE_PRE.value,
                "detail_sha256": "f" * 64,
            },
        },
        "historical_relation": "NOT_COMPARABLE_INCOMPLETE",
        "quality": None,
        "phase_attribution": None,
        "next_route": "NOT_PRODUCED",
    }
    for field, value in mutations.items():
        payload = receipt_module.diag2_diagnostic_receipt_payload(built)
        payload[field] = value
        with pytest.raises(ValueError, match="claims differ"):
            receipt_module.diag2_diagnostic_receipt_from_payload(
                payload, artifact_root=root
            )


@pytest.mark.parametrize(
    "slot_name",
    (
        "cold_producer",
        "cold_terminal",
        "frozen_numerical_subset",
        "native_reference",
        "policy_authority",
        "preflight_producer",
        "preflight_process",
        "preflight_runtime",
        "preflight_policy",
        "preflight_memory",
        "preflight_memory_samples",
        "supervisor_before_preflight",
        "supervisor_before_preflight_pid",
        "supervisor_before_preflight_start",
        "supervisor_before_preflight_rows",
        "supervisor_before_cold",
        "cold_process",
        "cold_runtime",
        "source_manifest",
        "cold_memory",
        "cold_history",
        "cold_terminal_numerical",
        "cold_raw_trace",
        "cold_trace_intervals",
        "execution",
    ),
)
def test_diag2_complete_coherent_slot_rehash_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slot_name: str,
) -> None:
    root = tmp_path / f"diag2.partial-{'0' * 32}"
    built = _complete_receipt(root, monkeypatch, real_authorities=True, diag2=True)
    assert isinstance(built, receipt_module.DiagnosticReceiptV2)
    slots = dict(built.evidence_slots)
    artifact_slot_name = (
        "supervisor_before_preflight"
        if slot_name
        in {
            "supervisor_before_preflight_pid",
            "supervisor_before_preflight_start",
            "supervisor_before_preflight_rows",
        }
        else slot_name
    )
    slot = slots[artifact_slot_name]
    assert slot.artifact is not None
    path = root / slot.artifact.relative_path
    if slot_name == "cold_raw_trace":
        data = path.read_bytes() + b"coherent-raw-trace-mutation"
    else:
        payload = json.loads(path.read_bytes())
        if slot_name == "cold_producer":
            payload["route"] = "NEQ-GNTR1-DIAG1"
        elif slot_name == "cold_terminal":
            payload["terminal_status"] = "CRASH"
        elif slot_name == "frozen_numerical_subset":
            payload["entries"][0]["sha256"] = "f" * 64
        elif slot_name == "native_reference":
            payload["evidence"]["observables"]["volume"] = np.nextafter(
                float(payload["evidence"]["observables"]["volume"]), np.inf
            )
        elif slot_name == "policy_authority":
            payload["volume_target"] = np.nextafter(
                float(payload["volume_target"]), np.inf
            )
        elif slot_name == "preflight_producer":
            payload["plan_sha256"] = "f" * 64
        elif slot_name == "preflight_process":
            payload["child_pid"] += 1
        elif slot_name == "preflight_runtime":
            payload["runtime_identity"]["backend"] = "cpu"
        elif slot_name == "preflight_policy":
            payload["objective_target"] = np.nextafter(
                float(payload["objective_target"]), np.inf
            )
        elif slot_name == "preflight_memory":
            payload["peak_memory_fraction"] = 0.2
        elif slot_name == "preflight_memory_samples":
            payload["samples"][0]["used_memory_mib"] = 2
        elif slot_name == "supervisor_before_preflight":
            payload["gpu_uuid"] = "GPU-coherent-mutation"
        elif slot_name == "supervisor_before_preflight_pid":
            payload["supervisor_pid"] += 1_000
        elif slot_name == "supervisor_before_preflight_start":
            payload["supervisor_start_ticks"] += 1
        elif slot_name == "supervisor_before_preflight_rows":
            stdout = payload["compute_apps_query"]["stdout"]
            raw = root / stdout["relative_path"]
            raw_data = (
                f"{payload['supervisor_pid']}, {payload['gpu_uuid']}, 1\n".encode()
            )
            raw.write_bytes(raw_data)
            stdout["sha256"] = hashlib.sha256(raw_data).hexdigest()
            stdout["size_bytes"] = len(raw_data)
            payload["matching_rows"] = [
                {
                    "pid": payload["supervisor_pid"],
                    "gpu_uuid": payload["gpu_uuid"],
                    "used_memory_mib": 1,
                }
            ]
        elif slot_name == "supervisor_before_cold":
            payload["visible_device"] = "GPU-coherent-mutation"
        elif slot_name == "cold_process":
            payload["child_pid"] += 1
        elif slot_name == "cold_runtime":
            payload["runtime_identity"]["backend"] = "cpu"
        elif slot_name == "source_manifest":
            payload["worktree"]["git_head"] = "f" * 40
        elif slot_name == "cold_memory":
            payload["peak_memory_fraction"] = 0.2
        elif slot_name == "cold_history":
            payload["attempts"] += 1
        elif slot_name == "cold_terminal_numerical":
            payload["objective"] = np.nextafter(float(payload["objective"]), np.inf)
        elif slot_name == "cold_trace_intervals":
            payload["phase_schema_sha256"] = "f" * 64
        elif slot_name == "execution":
            payload["child_pid"] += 1
        else:
            raise AssertionError(f"unhandled complete mutation slot: {slot_name}")
        data = canonical_json_bytes(payload)
    os.chmod(path, 0o644)
    path.write_bytes(data)
    changed = ArtifactRef(
        slot.artifact.relative_path,
        hashlib.sha256(data).hexdigest(),
        len(data),
        slot.artifact.schema_version,
    )
    slots[artifact_slot_name] = receipt_module.EvidenceSlot.present(changed)
    with pytest.raises((OSError, TypeError, ValueError)):
        receipt_module.build_diag2_diagnostic_receipt(
            artifact_root=root, evidence_slots=slots
        )


@pytest.mark.parametrize("producer_name", ("preflight_producer", "cold_producer"))
def test_diag2_complete_requires_both_producers_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer_name: str,
) -> None:
    root = tmp_path / f"diag2.partial-{'0' * 32}"
    built = _complete_receipt(root, monkeypatch, real_authorities=True, diag2=True)
    assert isinstance(built, receipt_module.DiagnosticReceiptV2)
    slots = dict(built.evidence_slots)
    producer = slots[producer_name]
    assert producer.artifact is not None
    (root / producer.artifact.relative_path).unlink()
    slots[producer_name] = receipt_module.EvidenceSlot.absent(
        receipt_module.AbsenceReason.PRODUCER_DECODE_FAILED
    )
    with pytest.raises(ValueError, match="complete DIAG2 terminal requires every slot"):
        receipt_module.build_diag2_diagnostic_receipt(
            artifact_root=root, evidence_slots=slots
        )


@pytest.mark.parametrize("mutation", ("manifest_role", "sealed_mode"))
def test_diag2_complete_sealed_manifest_role_and_mode_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / f"diag2.partial-{'0' * 32}"
    built = _complete_receipt(root, monkeypatch, real_authorities=True, diag2=True)
    assert isinstance(built, receipt_module.DiagnosticReceiptV2)
    (root / receipt_module.DIAG2_RECEIPT_FILENAME).write_bytes(
        receipt_module.diag2_diagnostic_receipt_bytes(built)
    )
    manifest_path = root / receipt_module.DIAG2_MANIFEST_FILENAME
    manifest_path.write_bytes(
        canonical_json_bytes(receipt_module.diag2_artifact_manifest_payload(root))
    )
    assert receipt_module.validate_diag2_writable_staging(root) == built
    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
    assert receipt_module.load_and_validate_diag2_staging(root) == built

    if mutation == "manifest_role":
        os.chmod(manifest_path, 0o644)
        manifest = json.loads(manifest_path.read_bytes())
        entry = next(
            item
            for item in manifest["entries"]
            if item["relative_path"] == "policy-authority.json"
        )
        entry["role"] = "cold_policy"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        os.chmod(manifest_path, 0o444)
    else:
        os.chmod(root / "policy-authority.json", 0o644)
    with pytest.raises(ValueError):
        receipt_module.load_and_validate_diag2_staging(root)


def test_diag2_stage_reason_table_covers_every_stage_and_reason() -> None:
    assert set(receipt_module.DIAG2_STAGE_REASON_CODES) == set(
        receipt_module.FailureStageV2
    )
    assert set().union(*receipt_module.DIAG2_STAGE_REASON_CODES.values()) == set(
        receipt_module.FailureReasonCodeV2
    )
    for stage, allowed in receipt_module.DIAG2_STAGE_REASON_CODES.items():
        for reason in receipt_module.FailureReasonCodeV2:
            failure_payload = {
                "stage": stage.value,
                "reason": {"code": reason.value, "detail_sha256": "5" * 64},
            }
            if reason in allowed:
                assert (
                    receipt_module._parse_diag2_failure(failure_payload).reason
                    is reason
                )
            else:
                with pytest.raises(ValueError, match="stage/reason pairing"):
                    receipt_module._parse_diag2_failure(failure_payload)


@pytest.mark.parametrize(
    ("earlier", "later"),
    tuple(
        zip(
            receipt_module.DIAG2_FAILURE_STAGE_ORDER[:-1],
            receipt_module.DIAG2_FAILURE_STAGE_ORDER[1:],
            strict=True,
        )
    ),
)
def test_diag2_failure_selector_uses_exact_adjacent_stage_precedence(
    earlier: receipt_module.FailureStageV2,
    later: receipt_module.FailureStageV2,
) -> None:
    earlier_failure = receipt_module.StructuredFailureV2(
        earlier,
        min(
            receipt_module.DIAG2_STAGE_REASON_CODES[earlier],
            key=lambda item: item.value,
        ),
        "1" * 64,
    )
    later_failure = receipt_module.StructuredFailureV2(
        later,
        min(
            receipt_module.DIAG2_STAGE_REASON_CODES[later], key=lambda item: item.value
        ),
        "2" * 64,
    )
    assert (
        receipt_module.select_diag2_failure((later_failure, earlier_failure))
        == earlier_failure
    )


def test_diag2_failure_selector_rejects_ambiguous_or_invalid_candidates() -> None:
    stage = receipt_module.FailureStageV2.PREFLIGHT_TIMEOUT
    valid = receipt_module.StructuredFailureV2(
        stage, receipt_module.FailureReasonCodeV2.CHILD_TIMEOUT, "3" * 64
    )
    assert receipt_module.select_diag2_failure((valid,)) == valid
    with pytest.raises(ValueError, match="empty"):
        receipt_module.select_diag2_failure(())
    with pytest.raises(ValueError, match="duplicate a stage"):
        receipt_module.select_diag2_failure(
            (
                valid,
                receipt_module.StructuredFailureV2(
                    stage,
                    receipt_module.FailureReasonCodeV2.CHILD_TIMEOUT,
                    "4" * 64,
                ),
            )
        )
    with pytest.raises(ValueError, match="stage/reason pairing"):
        receipt_module.select_diag2_failure(
            (
                receipt_module.StructuredFailureV2(
                    receipt_module.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
                    receipt_module.FailureReasonCodeV2.CHILD_TIMEOUT,
                    "5" * 64,
                ),
            )
        )
    with pytest.raises(ValueError, match="detail SHA"):
        receipt_module.select_diag2_failure(
            (
                receipt_module.StructuredFailureV2(
                    stage,
                    receipt_module.FailureReasonCodeV2.CHILD_TIMEOUT,
                    "not-a-sha",
                ),
            )
        )


@pytest.mark.parametrize(
    "stage",
    [
        receipt_module.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
        receipt_module.FailureStageV2.COLD_SOURCE_FAILURE,
    ],
)
def test_diag2_source_pre_is_initial_publication_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: receipt_module.FailureStageV2,
) -> None:
    invalid_payload = {
        "stage": stage.value,
        "reason": {
            "code": receipt_module.FailureReasonCodeV2.SOURCE_PRE.value,
            "detail_sha256": "5" * 64,
        },
    }
    with pytest.raises(ValueError, match="stage/reason pairing"):
        receipt_module._parse_diag2_failure(invalid_payload)

    artifact_root = tmp_path / "staging"
    artifact_root.mkdir()
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    refs["supervisor_terminal"] = ArtifactRef(
        "supervisor-terminal.json", "6" * 64, 1, "supervisor-terminal-v2"
    )
    failure = receipt_module.StructuredFailureV2(
        receipt_module.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt_module.FailureReasonCodeV2.SOURCE_PRE,
        "7" * 64,
    )
    monkeypatch.setattr(
        receipt_module, "_validate_diag2_slots", lambda *args, **kwargs: None
    )
    slots = receipt_module.derive_diag2_evidence_slots(
        artifact_root=artifact_root,
        artifact_refs=refs,
        failure=failure,
    )
    assert slots["source_manifest"] == receipt_module.EvidenceSlot.absent(
        receipt_module.AbsenceReason.SOURCE_PRE
    )
    for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES - {
        "source_manifest",
        "supervisor_terminal",
    }:
        assert slots[name] == receipt_module.EvidenceSlot.absent(
            receipt_module.AbsenceReason.NOT_REACHED
        )


def test_diag2_timeout_cannot_retain_a_producer(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    root.mkdir()

    def json_ref(relative: str, schema: str) -> ArtifactRef:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes({"schema_version": schema}))
        return _diag2_ref(path, root, schema)

    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    for name in (
        "source_manifest",
        "frozen_numerical_subset",
        "native_reference",
        "policy_authority",
        "supervisor_before_preflight",
        "preflight_terminal",
        "preflight_process",
        "supervisor_terminal",
    ):
        refs[name] = json_ref(
            receipt_module.DIAG2_EVIDENCE_SLOT_PATHS[name].replace(
                "<run>/<base>.trace.json.gz", "run/base.trace.json.gz"
            ),
            f"test-{name}-v1",
        )
    refs["preflight_producer"] = ArtifactRef(
        "preflight/producer.json",
        "7" * 64,
        1,
        "single-stage-neq-gntr1-preflight-worker-v1",
    )
    failure = receipt_module.StructuredFailureV2(
        receipt_module.FailureStageV2.PREFLIGHT_TIMEOUT,
        receipt_module.FailureReasonCodeV2.CHILD_TIMEOUT,
        "6" * 64,
    )
    with pytest.raises(ValueError, match="does not permit retained producer"):
        receipt_module.derive_diag2_evidence_slots(
            artifact_root=root, artifact_refs=refs, failure=failure
        )


def test_diag2_postlaunch_setup_drift_cannot_be_missealed(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    refs["preflight_terminal"] = ArtifactRef(
        "preflight/terminal.json", "1" * 64, 1, "terminal-v1"
    )
    failure = receipt_module.StructuredFailureV2(
        receipt_module.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt_module.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        "2" * 64,
    )
    with pytest.raises(
        ValueError, match="initial setup failure omits earlier authority"
    ):
        receipt_module.derive_diag2_evidence_slots(
            artifact_root=root, artifact_refs=refs, failure=failure
        )


@pytest.mark.parametrize(
    ("stage", "reason", "offending"),
    [
        (
            receipt_module.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            receipt_module.FailureReasonCodeV2.SOURCE_POST,
            "source_manifest",
        ),
        (
            receipt_module.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            receipt_module.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            "frozen_numerical_subset",
        ),
        (
            receipt_module.FailureStageV2.COLD_SOURCE_FAILURE,
            receipt_module.FailureReasonCodeV2.REFERENCE_INVALID,
            "native_reference",
        ),
        (
            receipt_module.FailureStageV2.COLD_SOURCE_FAILURE,
            receipt_module.FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
            "policy_authority",
        ),
    ],
)
@pytest.mark.parametrize("minimum_typed", [True, False])
def test_diag2_postlaunch_setup_drift_preserves_exact_child_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: receipt_module.FailureStageV2,
    reason: receipt_module.FailureReasonCodeV2,
    offending: str,
    minimum_typed: bool,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt_module.DIAG2_EVIDENCE_SLOT_NAMES
    }
    preserved_groups = (
        {
            "SETUP_SOURCE",
            "SETUP_REFERENCE",
            "SETUP_POLICY",
            "ZERO_PREFLIGHT",
            "PREFLIGHT",
        }
        if stage is receipt_module.FailureStageV2.PREFLIGHT_SOURCE_FAILURE
        else {
            "SETUP_SOURCE",
            "SETUP_REFERENCE",
            "SETUP_POLICY",
            "ZERO_PREFLIGHT",
            "PREFLIGHT",
            "ZERO_COLD",
            "COLD_SUPERVISION",
        }
    )
    for group, names in receipt_module._DIAG2_AUTHORITY_GROUPS:
        if group in preserved_groups:
            for name in names:
                if name != offending:
                    refs[name] = ArtifactRef(f"{name}.json", "1" * 64, 1, "test-v1")
    if minimum_typed:
        refs[offending] = ArtifactRef(
            receipt_module.DIAG2_EVIDENCE_SLOT_PATHS[offending],
            "4" * 64,
            1,
            "test-v1",
        )
    refs["supervisor_terminal"] = ArtifactRef(
        "supervisor-terminal.json", "2" * 64, 1, "terminal-v2"
    )
    monkeypatch.setattr(
        receipt_module, "_validate_diag2_slots", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(receipt_module, "_load_ref_json", lambda *_args: {})
    monkeypatch.setattr(
        receipt_module, "validate_diag2_producer_payload", lambda *_args, **_kwargs: {}
    )
    subordinate_group = (
        "PREFLIGHT"
        if stage is receipt_module.FailureStageV2.PREFLIGHT_SOURCE_FAILURE
        else "COLD_SUPERVISION"
    )
    monkeypatch.setattr(
        receipt_module,
        "_derive_diag2_subordinate_child_slots",
        lambda *_args, **_kwargs: {
            name: receipt_module.EvidenceSlot.present(refs[name])
            for name in dict(receipt_module._DIAG2_AUTHORITY_GROUPS)[subordinate_group]
            if refs[name] is not None
        },
    )
    slots = receipt_module.derive_diag2_evidence_slots(
        artifact_root=root,
        artifact_refs=refs,
        failure=receipt_module.StructuredFailureV2(stage, reason, "3" * 64),
    )
    assert slots[offending] == (
        receipt_module.EvidenceSlot.present(refs[offending])
        if minimum_typed and refs[offending] is not None
        else receipt_module.EvidenceSlot.absent(
            receipt_module.AbsenceReason(reason.value)
        )
    )
    for name, reference in refs.items():
        if name != offending and reference is not None:
            assert slots[name].state is receipt_module.EvidenceState.PRESENT
    if stage is receipt_module.FailureStageV2.COLD_SOURCE_FAILURE:
        for name in dict(receipt_module._DIAG2_AUTHORITY_GROUPS)["COLD_NUMERICAL"]:
            assert slots[name] == receipt_module.EvidenceSlot.absent(
                receipt_module.AbsenceReason.NOT_REACHED
            )


def _history_arrays() -> SimpleNamespace:
    values: dict[str, np.ndarray] = {
        "outcome": np.concatenate(
            (
                np.full(203, 1, dtype=np.int32),
                np.full(97, 2, dtype=np.int32),
            )
        ),
        "accepted_step_number": np.concatenate(
            (np.arange(1, 204, dtype=np.int32), np.zeros(97, dtype=np.int32))
        ),
        "steihaug_hit_boundary": np.zeros(MAXIMUM_ATTEMPTS, dtype=np.bool_),
    }
    for name in HISTORY_INTEGER_FIELDS:
        values[name] = np.zeros(MAXIMUM_ATTEMPTS, dtype=np.int32)
    for name in HISTORY_FLOAT_FIELDS:
        values[name] = np.ones(MAXIMUM_ATTEMPTS, dtype=np.float64)
    for name in (
        "candidate_feasibility_inf",
        "correction_norm",
        "correction_step_ratio",
        "correction_relative_residual",
        "correction_forward_error_bound",
        "trial_gram_factorization_relative_residual",
        "trial_gram_solve_relative_residual",
    ):
        values[name].fill(0.0)
    values["actual_reduction"][203:] = -1.0
    values["reduction_ratio"][203:] = -1.0
    return SimpleNamespace(**values)


def _incomplete(root: Path) -> IncompleteDiagnosticReceipt:
    return build_incomplete_diagnostic_receipt(
        artifact_root=root,
        evidence_refs={name: None for name in EVIDENCE_REF_KEYS},
    )


def _artifact_ref(path: Path, root: Path, schema: str) -> ArtifactRef:
    data = path.read_bytes()
    return ArtifactRef(
        path.relative_to(root).as_posix(),
        hashlib.sha256(data).hexdigest(),
        len(data),
        schema,
    )


def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(values), allow_pickle=False)


def _publish_test_snapshot(
    root: Path,
    *,
    diag2: bool = False,
) -> tuple[SnapshotPublication, dict[str, object]]:
    sources = root.parent / "snapshot-inputs"
    roles = dict(receipt_module.REQUIRED_SOURCE_ROLE_BINDINGS)
    roles.update(
        {
            "src/simsopt/__init__.py": "execution_source",
            "src/simsopt_jax/__init__.py": "execution_source",
            "src/simsopt_jax_adapters/__init__.py": "execution_source",
            "src/simsoptpp.so": "native_extension",
        }
    )
    if diag2:
        roles.update(
            {
                relative: roles.get(
                    relative,
                    "benchmark"
                    if relative.startswith("benchmarks/")
                    else "execution_source",
                )
                for relative, _ in receipt_module.DIAG2_FROZEN_NUMERICAL_ENTRIES
            }
        )
    roots: list[SourceRoot] = []
    repository = Path(__file__).resolve().parents[2]
    retained_source = Path(
        "/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag1-prospective/source-snapshot"
    )
    frozen_paths = dict(receipt_module.DIAG2_FROZEN_NUMERICAL_ENTRIES)
    for relative, role in sorted(roles.items()):
        path = sources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if diag2 and relative in frozen_paths:
            path.write_bytes((retained_source / relative).read_bytes())
        elif diag2 and relative != "src/simsoptpp.so":
            path.write_bytes((repository / relative).read_bytes())
        else:
            path.write_bytes(f"fixture:{relative}\n".encode())
        roots.append(SourceRoot(role, path, relative))  # type: ignore[arg-type]
    publication = publish_immutable_snapshot(
        root / "source-snapshot",
        roots,
        worktree=WorktreeIdentity(
            git_head="1" * 40,
            tracked_diff_sha256="2" * 64,
            untracked_bytes_manifest_sha256="3" * 64,
            repo_root=str(sources.resolve()),
        ),
    )
    source_identity = publication.source_identity(root)
    process_source = {
        "git_head": source_identity.git_head,
        "tracked_diff_sha256": source_identity.tracked_diff_sha256,
        "untracked_bytes_manifest_sha256": (
            source_identity.untracked_bytes_manifest_sha256
        ),
        "source_manifest_sha256": source_identity.snapshot_manifest.sha256,
        "source_manifest_size_bytes": source_identity.snapshot_manifest.size_bytes,
    }
    return publication, process_source


def _publish_test_runtime(
    root: Path,
    publication: SnapshotPublication,
    *,
    relative_path: str,
    argv: tuple[str, ...],
) -> ArtifactRef:
    entries = {entry.relative_path: entry for entry in publication.entries}
    modules = (
        ("simsopt", "src/simsopt/__init__.py"),
        ("simsopt_jax", "src/simsopt_jax/__init__.py"),
        ("simsopt_jax_adapters", "src/simsopt_jax_adapters/__init__.py"),
        ("simsoptpp", "src/simsoptpp.so"),
    )
    bindings = tuple(
        ImportBinding(module, path, entries[path].size_bytes, entries[path].sha256)
        for module, path in modules
    )
    entry = entries["benchmarks/run_single_stage_native_equivalent_quality_campaign.py"]
    entrypoint = ImportBinding(
        "__entrypoint__", entry.relative_path, entry.size_bytes, entry.sha256
    )
    environment = effective_environment(
        {
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS": "67108864",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        }
    )
    environment_sha = hashlib.sha256(
        canonical_json_bytes(dict(environment))
    ).hexdigest()
    snapshot_root = publication.root
    identity = RuntimeIdentity(
        argv=argv,
        cwd=str(snapshot_root),
        python_executable="/fixture/python",
        python_version="3.11",
        jax_version="fixture-jax",
        jaxlib_version="fixture-jaxlib",
        simsopt_module_path=str(snapshot_root / modules[0][1]),
        simsopt_jax_module_path=str(snapshot_root / modules[1][1]),
        native_extension_path=str(snapshot_root / modules[3][1]),
        backend="gpu",
        device_uuid=GPU_UUID,
        driver_version="fixture-driver",
        effective_environment_sha256=environment_sha,
    )
    evidence = build_runtime_evidence(
        snapshot_root,
        source_identity=publication.source_identity(root),
        observation=RuntimeObservation(
            identity,
            entrypoint,
            bindings,
            environment,
            "fixture-gpu",
            "fixture-platform",
        ),
    )
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_runtime_evidence(
        path, evidence, snapshot_root=snapshot_root, campaign_root=root
    )


def _copy_validated_reference(root: Path) -> np.ndarray:
    source = (
        Path(__file__).resolve().parents[2]
        / "artifacts/neq-native-reference-20260811T012049Z"
    )
    if not source.is_dir():
        pytest.fail(f"sealed native-reference fixture is absent: {source}")
    destination = root / "native-reference"
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    receipt_module.validate_native_equivalent_reference(destination)
    document = json.loads((destination / "reference.json").read_bytes())
    relative = document["evidence"]["arrays"]["raw_equalities"]["relative_path"]
    with (destination / relative).open("rb") as stream:
        return np.load(stream, allow_pickle=False)


def _ref_payload(reference: ArtifactRef) -> dict[str, object]:
    return {
        "relative_path": reference.relative_path,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "schema_version": reference.schema_version,
    }


def _write_canonical_ref(
    root: Path, relative: str, payload: dict[str, object], schema: str
) -> ArtifactRef:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return _artifact_ref(path, root, schema)


def _publish_supervision_authorities(
    root: Path,
    refs: dict[str, ArtifactRef],
    publication: SnapshotPublication,
    process_source: dict[str, object],
    policy_payload: dict[str, object],
    policy_sha256: str,
    timestamps: tuple[str, ...],
    *,
    diag2: bool = False,
    native_raw_equalities: np.ndarray | None = None,
    constraint_inverse_scale: np.ndarray | None = None,
) -> None:
    entrypoint = str(
        publication.root
        / "benchmarks/run_single_stage_native_equivalent_quality_campaign.py"
    )
    preflight_argv = (entrypoint, "--diagnostic-child", "preflight")
    cold_argv = (entrypoint, "--diagnostic-child", "cold")
    preflight_runtime = _publish_test_runtime(
        root,
        publication,
        relative_path="preflight/runtime-evidence.json",
        argv=preflight_argv,
    )
    cold_runtime = _publish_test_runtime(
        root,
        publication,
        relative_path="cold/runtime-evidence.json",
        argv=cold_argv,
    )
    refs["preflight_runtime"] = preflight_runtime
    refs["runtime"] = cold_runtime
    preflight_policy = _write_canonical_ref(
        root,
        "preflight/policy.json",
        policy_payload,
        f"{receipt_module.SCHEMA_VERSION}-policy",
    )
    refs["preflight_policy"] = preflight_policy
    if diag2:
        if native_raw_equalities is None or constraint_inverse_scale is None:
            raise ValueError("DIAG2 fixture omits parent policy inputs")
        native_document = json.loads(
            (root / "native-reference/reference.json").read_bytes()
        )
        reference_volume = float(native_document["evidence"]["observables"]["volume"])
        native_reference = _artifact_ref(
            root / "native-reference/reference.json",
            root,
            str(native_document["schema_version"]),
        )
        parent_policy = receipt_module.build_diag2_policy_authority_payload(
            native_reference=native_reference,
            reference_volume=reference_volume,
            volume_target=float.fromhex(receipt_module.DIAG2_VOLUME_TARGET_HEX),
            native_raw_equalities=native_raw_equalities,
            constraint_inverse_scale=constraint_inverse_scale,
        )
        refs["policy_authority"] = _write_canonical_ref(
            root,
            "policy-authority.json",
            parent_policy,
            receipt_module.DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION,
        )
    else:
        refs["policy_authority"] = _write_canonical_ref(
            root,
            "policy-authority.json",
            policy_payload,
            f"{receipt_module.SCHEMA_VERSION}-policy",
        )
    runtime_summary = {
        "backend": "gpu",
        "device": "fixture-gpu",
        "device_uuid": GPU_UUID,
        "jax": "fixture-jax",
        "jax_enable_x64": True,
        "jaxlib": "fixture-jaxlib",
        "python": "3.11",
    }
    preflight_payload = {
        "schema_version": "single-stage-neq-gntr1-preflight-worker-v1",
        "route": receipt_module.DIAG2_ROUTE if diag2 else receipt_module.ROUTE,
        "plan_sha256": receipt_module.DIAG2_PLAN_SHA256 if diag2 else PLAN_SHA256,
        "mode": "ANNOTATED_LOWER_COMPILE_ONLY",
        "execution_status": "SUCCESS",
        "policy_sha256": policy_sha256,
        "policy_evidence": _ref_payload(preflight_policy),
        "phase_schema_sha256": PHASE_SCHEMA_SHA256,
        "state_size": 716,
        "equality_size": 255,
        "residual_size": 2110,
        "campaign_authorized": False,
        "solver_dispatched": False,
        "finalizer_called": False,
        "endpoint_audit_called": False,
        "python_callbacks": 0,
        "runtime": runtime_summary,
        "runtime_evidence": _ref_payload(preflight_runtime),
        "timing": {
            "compile_started_ns": 1,
            "compile_completed_ns": 2,
            "process_seconds_before_serialization": 1.0,
        },
        "failure_reasons": [],
    }
    refs["preflight"] = _write_canonical_ref(
        root,
        "preflight/producer.json",
        preflight_payload,
        "single-stage-neq-gntr1-preflight-worker-v1",
    )
    cold_payload = {
        "schema_version": f"{receipt_module.SCHEMA_VERSION}-producer",
        "route": receipt_module.DIAG2_ROUTE if diag2 else receipt_module.ROUTE,
        "plan_sha256": receipt_module.DIAG2_PLAN_SHA256 if diag2 else PLAN_SHA256,
        "execution_status": "COMPLETE",
        "runtime": runtime_summary,
        "runtime_evidence": _ref_payload(cold_runtime),
        "policy_sha256": policy_sha256,
        "phase_schema_sha256": PHASE_SCHEMA_SHA256,
        "history_evidence": _ref_payload(refs["history"]),
        "terminal_numerical_evidence": _ref_payload(refs["terminal_numerical"]),
        "policy_evidence": _ref_payload(refs["policy"]),
        "raw_trace_evidence": _ref_payload(refs["raw_trace"]),
        "trace_intervals_evidence": _ref_payload(refs["trace_intervals"]),
        "timestamps_ns": {name: index + 1 for index, name in enumerate(timestamps)},
        "transfer_audit": {
            "hot_h2d_transfers": 0,
            "hot_d2h_transfers": 0,
            "python_callbacks": 0,
            "final_d2h_transfers": 1,
        },
        "endpoint_audit_called": False,
        "campaign_authorized": False,
        "failure_reasons": [],
    }
    refs["producer"] = _write_canonical_ref(
        root,
        "cold/producer.json",
        cold_payload,
        f"{receipt_module.SCHEMA_VERSION}-producer",
    )
    for prefix, argv, producer_name, pid in (
        ("preflight", preflight_argv, "preflight", 10),
        ("cold", cold_argv, "producer", 20),
    ):
        producer_ref = refs[producer_name]
        stdout_path = root / prefix / "stdout.bin"
        stdout_path.write_bytes((root / producer_ref.relative_path).read_bytes())
        stderr_path = root / prefix / "stderr.bin"
        stderr_path.write_bytes(b"")
        stdout_ref = _artifact_ref(stdout_path, root, "raw-process-stdout-v1")
        stderr_ref = _artifact_ref(stderr_path, root, "raw-process-stderr-v1")
        process_payload = {
            "schema_version": (
                receipt_module.DIAG2_PROCESS_SCHEMA_VERSION
                if diag2
                else f"{receipt_module.SCHEMA_VERSION}-process"
            ),
            "child_pid": pid,
            "child_start_time_ticks": pid + 1,
            "argv": list(argv),
            "stdout": _ref_payload(stdout_ref),
            "stderr": _ref_payload(stderr_ref),
            "process_seconds": 1.0,
            "process_diagnostics": {"returncode": 0} if diag2 else {},
            "pre_source_identity": process_source,
            "post_source_identity": process_source,
        }
        if diag2:
            process_payload.update(
                {
                    "monitor_failure_kind": "NONE",
                    "process_started_monotonic_ns": 20 if prefix == "preflight" else 50,
                    "process_stopped_monotonic_ns": 30 if prefix == "preflight" else 60,
                }
            )
        refs["preflight_process" if prefix == "preflight" else "process"] = (
            _write_canonical_ref(
                root,
                f"{prefix}/process.json",
                process_payload,
                (
                    receipt_module.DIAG2_PROCESS_SCHEMA_VERSION
                    if diag2
                    else f"{receipt_module.SCHEMA_VERSION}-process"
                ),
            )
        )
        samples_ref = _write_canonical_ref(
            root,
            f"{prefix}/gpu-memory-samples.json",
            {
                "schema_version": f"{receipt_module.SCHEMA_VERSION}-memory-samples",
                "samples": [
                    {"sampled_at_unix_ns": 1, "used_memory_mib": 1},
                    {"sampled_at_unix_ns": 2, "used_memory_mib": 1},
                ],
            },
            f"{receipt_module.SCHEMA_VERSION}-memory-samples",
        )
        argv_sha = hashlib.sha256(canonical_json_bytes(list(argv[2:]))).hexdigest()
        memory_ref = _write_canonical_ref(
            root,
            f"{prefix}/gpu-memory.json",
            {
                "schema_version": "single-stage-neq-gntr1-memory-v1",
                "monitor_scope": "whole-child-exact-pid-exact-device",
                "parent_pid": 1,
                "child_pid": pid,
                "child_start_time_ticks": pid + 1,
                "child_argv_sha256": argv_sha,
                "device_uuid": GPU_UUID,
                "sample_count": 2,
                "peak_memory_bytes": 1024 * 1024,
                "peak_memory_fraction": 0.1,
            },
            "single-stage-neq-gntr1-memory-v1",
        )
        refs[
            "preflight_memory_samples" if prefix == "preflight" else "memory_samples"
        ] = samples_ref
        refs["preflight_memory" if prefix == "preflight" else "memory"] = memory_ref
        terminal_name = (
            "preflight_child_terminal" if prefix == "preflight" else "child_terminal"
        )
        terminal_schema = (
            receipt_module.DIAG2_CHILD_TERMINAL_SCHEMA_VERSION
            if diag2
            else f"{receipt_module.SCHEMA_VERSION}-child-terminal"
        )
        terminal_payload: dict[str, object] = {
            "schema_version": terminal_schema,
            "terminal_status": "COMPLETE",
            "failure_reasons": [],
        }
        if diag2:
            terminal_payload["monitor_failure_kind"] = "NONE"
        refs[terminal_name] = _write_canonical_ref(
            root,
            f"{prefix}/terminal.json",
            terminal_payload,
            terminal_schema,
        )


def _diag2_zero_ref(root: Path, *, stage: str, monotonic_ns: int) -> ArtifactRef:
    stage_slug = "before-preflight" if stage == "BEFORE_PREFLIGHT" else "before-cold"

    def raw_ref(relative: str, schema: str, data: bytes) -> ArtifactRef:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return _artifact_ref(path, root, schema)

    inventory_stdout = raw_ref(
        f"supervisor/{stage_slug}-gpu-inventory.stdout.bin",
        "raw-supervisor-gpu-inventory-stdout-v1",
        f"{GPU_UUID}, 33554432\n".encode(),
    )
    raw_refs = {
        "inventory_stderr": raw_ref(
            f"supervisor/{stage_slug}-gpu-inventory.stderr.bin",
            "raw-supervisor-gpu-inventory-stderr-v1",
            b"",
        ),
        "compute_stdout": raw_ref(
            f"supervisor/{stage_slug}-compute-apps.stdout.bin",
            "raw-supervisor-compute-apps-stdout-v1",
            b"",
        ),
        "compute_stderr": raw_ref(
            f"supervisor/{stage_slug}-compute-apps.stderr.bin",
            "raw-supervisor-compute-apps-stderr-v1",
            b"",
        ),
    }
    query_sha = "4" * 64
    inventory = receipt_module.SupervisorQueryV2(
        receipt_module._DIAG2_GPU_INVENTORY_ARGV,
        query_sha,
        True,
        False,
        0,
        inventory_stdout,
        raw_refs["inventory_stderr"],
    )
    compute = receipt_module.SupervisorQueryV2(
        receipt_module._DIAG2_COMPUTE_APPS_ARGV,
        query_sha,
        True,
        False,
        0,
        raw_refs["compute_stdout"],
        raw_refs["compute_stderr"],
    )
    payload = receipt_module.build_diag2_supervisor_zero_payload(
        stage=stage,
        captured_at_monotonic_ns=monotonic_ns,
        captured_at_unix_ns=monotonic_ns + 1_000,
        supervisor_pid=os.getpid(),
        supervisor_start_ticks=int(
            Path(f"/proc/{os.getpid()}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()[19]
        ),
        gpu_uuid=GPU_UUID,
        visible_device=GPU_UUID,
        gpu_inventory_query=inventory,
        compute_apps_query=compute,
        matching_rows=(),
    )
    return _write_canonical_ref(
        root,
        receipt_module.DIAG2_EVIDENCE_SLOT_PATHS[
            "supervisor_before_preflight"
            if stage == "BEFORE_PREFLIGHT"
            else "supervisor_before_cold"
        ],
        payload,
        receipt_module.DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
    )


def _complete_receipt(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    quality_hit: bool = False,
    real_authorities: bool = False,
    diag2: bool = False,
) -> receipt_module.DiagnosticReceipt | receipt_module.DiagnosticReceiptV2:
    if diag2 and not real_authorities:
        raise ValueError("DIAG2 complete fixture requires real authorities")
    root.mkdir()
    publication: SnapshotPublication | None = None
    process_source: dict[str, object] | None = None
    native_authority: np.ndarray | None = None
    if real_authorities:
        publication, process_source = _publish_test_snapshot(root, diag2=diag2)
        if diag2:
            filtered = receipt_module._diag2_filtered_source_entries(publication)
            monkeypatch.setattr(
                receipt_module, "DIAG2_BASELINE_FILTERED_ENTRY_COUNT", len(filtered)
            )
            monkeypatch.setattr(
                receipt_module,
                "DIAG2_BASELINE_FILTERED_ENTRIES_SHA256",
                hashlib.sha256(canonical_json_bytes(filtered)).hexdigest(),
            )
        native_authority = _copy_validated_reference(root)
    history_arrays = _history_arrays()
    if quality_hit:
        history_arrays.outcome[:] = 0
        history_arrays.accepted_step_number[:] = 0
        history_arrays.outcome[0] = 1
        history_arrays.accepted_step_number[0] = 1
        for name in HISTORY_INTEGER_FIELDS:
            getattr(history_arrays, name)[1:] = 0
        history_arrays.steihaug_hit_boundary[1:] = False
        for name in HISTORY_FLOAT_FIELDS:
            getattr(history_arrays, name)[1:] = np.nan
    history_path = root / ("cold/history.json" if real_authorities else "history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_bytes(
        canonical_json_bytes(
            history_evidence_from_arrays(
                history_arrays,
                quality_latch=quality_hit,
                first_quality_attempt=1 if quality_hit else 0,
                first_quality_accepted_step=1 if quality_hit else 0,
            )
        )
    )
    refs: dict[str, ArtifactRef] = {
        "history": _artifact_ref(
            history_path, root, f"{receipt_module.SCHEMA_VERSION}-history"
        )
    }

    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in receipt_module.ARRAY_SPECS.items():
        arrays[name] = np.zeros(shape, dtype=np.dtype(dtype))
    if native_authority is not None:
        arrays["native_equalities"][:] = native_authority
    arrays["variable_scale"].fill(1.0)
    if diag2:
        arrays["constraint_inverse_scale"][:254] = 1.0 / np.sqrt(np.float64(254.0))
        arrays["constraint_inverse_scale"][254] = 1.0 / abs(
            float.fromhex(receipt_module.DIAG2_VOLUME_TARGET_HEX)
        )
    else:
        arrays["constraint_inverse_scale"].fill(1.0)
    accepted_rows = 2 if quality_hit else 204
    objective = 0.0 if quality_hit else 1.0
    arrays["accepted_mask"][:accepted_rows] = True
    arrays["accepted_quality_mask"][:accepted_rows] = True
    arrays["accepted_quality_objectives"].fill(objective)
    arrays["accepted_quality_coordinates_finite"].fill(True)
    arrays["accepted_quality_objective_finite"].fill(True)
    arrays["accepted_quality_raw_equalities_finite"].fill(True)
    arrays["accepted_quality_scaled_equalities_finite"].fill(True)
    arrays["accepted_quality_component_bounds_satisfied"].fill(True)
    arrays["accepted_quality_scaled_feasibility_satisfied"].fill(True)
    if quality_hit:
        arrays["accepted_quality_objective_satisfied"].fill(True)
        arrays["accepted_quality_satisfied"][:accepted_rows] = True
    if not quality_hit:
        arrays["objective_residual_vector"][0] = np.sqrt(2.0)
    arrays["transpose_equality_probe"][0] = 1.0
    arrays["transpose_state_probe"][0] = 1.0
    array_payloads: dict[str, dict[str, object]] = {}
    for name, values in arrays.items():
        path = root / ("cold/arrays" if real_authorities else "arrays") / f"{name}.npy"
        _write_npy(path, values)
        reference = _artifact_ref(path, root, f"array-{name}")
        array_payloads[name] = receipt_module.array_evidence_payload(
            reference=reference, name=name, values=values
        )
    terminal_path = root / (
        "cold/terminal-numerical.json" if real_authorities else "terminal.json"
    )
    terminal_path.write_bytes(
        canonical_json_bytes(
            terminal_numerical_payload(
                arrays=array_payloads,
                objective=objective,
                objective_terms={
                    "non_qs": 0.0,
                    "residual": objective,
                    "iota": 0.0,
                    "major_radius": 0.0,
                    "length": 0.0,
                },
                objective_weights={
                    "non_qs": 1.0,
                    "residual": 1.0,
                    "iota": 1.0,
                    "major_radius": 1.0,
                    "length": 1.0,
                },
                reconstructed_objective=(0.0 if quality_hit else 1.0000000000000002),
                authoritative_objective=objective,
                final_certificate={
                    name: (
                        (0.0 if quality_hit else 2.2204460492503126e-16)
                        if name == "residual_value_defect"
                        else 0.0
                    )
                    for name in FINAL_CERTIFICATE_FIELDS
                },
                kkt_status=receipt_module.KktStatus.AVAILABLE,
                raw_kkt_inf=0.0,
                scaled_stationarity_inf=0.0,
                residual_value_defect=(0.0 if quality_hit else 2.2204460492503126e-16),
                residual_gradient_defect=0.0,
                transpose_primal_dot=0.0,
                transpose_adjoint_dot=0.0,
                transpose_denominator=1.0,
                transpose_defect=0.0,
                terminal_endpoint_diagnostics_seconds=1.0,
            )
        )
    )
    refs["terminal_numerical"] = _artifact_ref(
        terminal_path, root, f"{receipt_module.SCHEMA_VERSION}-terminal"
    )
    native = arrays["native_equalities"]
    inverse_scale = arrays["constraint_inverse_scale"]
    native_sha = exact_numeric_tree_sha256(native)
    policy = NativeEquivalentQualityPolicy(native, native_sha, inverse_scale)
    policy_path = root / ("cold/policy.json" if real_authorities else "policy.json")
    policy_path.write_bytes(
        canonical_json_bytes(
            policy_evidence_payload(
                policy_sha256=policy.policy_sha256,
                native_raw_equalities=native,
                constraint_inverse_scale=inverse_scale,
            )
        )
    )
    refs["policy"] = _artifact_ref(
        policy_path, root, f"{receipt_module.SCHEMA_VERSION}-policy"
    )

    trace_path = root / (
        "cold/raw-trace/plugins/profile/run/host.trace.json.gz"
        if real_authorities
        else "trace.json.gz"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _write_trace(trace_path)
    if real_authorities:
        trace_path.with_name("host.xplane.pb").write_bytes(b"fixture-xplane")
    refs["raw_trace"] = _artifact_ref(trace_path, root, "jax-chrome-trace-gzip-v1")
    intervals_path = root / (
        "cold/trace-intervals.json" if real_authorities else "trace-intervals.json"
    )
    intervals_path.write_bytes(
        canonical_json_bytes(
            normalize_chrome_trace(trace_path, phase_schema_sha256=PHASE_SCHEMA_SHA256)
        )
    )
    refs["trace_intervals"] = _artifact_ref(
        intervals_path, root, f"{receipt_module.SCHEMA_VERSION}-raw-trace"
    )
    timestamps = (
        "process_started",
        "compile_started",
        "compile_completed",
        "state_ready",
        "profiler_started",
        "solve_started",
        "solve_stopped",
        "profiler_stopped",
        "finalizer_started",
        "finalizer_stopped",
        "quality_replay_started",
        "quality_replay_stopped",
        "endpoint_diagnostics_started",
        "endpoint_diagnostics_stopped",
        "final_d2h",
        "trace_exported",
        "serialized",
        "process_stopped",
    )
    if real_authorities:
        assert publication is not None and process_source is not None
        refs["source_manifest"] = publication.source_identity(root).snapshot_manifest
        refs["native_reference"] = _artifact_ref(
            root / "native-reference/reference.json",
            root,
            json.loads((root / "native-reference/reference.json").read_bytes())[
                "schema_version"
            ],
        )
        _publish_supervision_authorities(
            root,
            refs,
            publication,
            process_source,
            policy_evidence_payload(
                policy_sha256=policy.policy_sha256,
                native_raw_equalities=native,
                constraint_inverse_scale=inverse_scale,
            ),
            policy.policy_sha256,
            timestamps,
            diag2=diag2,
            native_raw_equalities=native,
            constraint_inverse_scale=inverse_scale,
        )
    else:
        for name in EVIDENCE_REF_KEYS - frozenset(
            {
                "history",
                "terminal_numerical",
                "policy",
                "raw_trace",
                "trace_intervals",
                "execution",
            }
        ):
            path = root / "support" / f"{name}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(canonical_json_bytes({"schema_version": f"test-{name}"}))
            refs[name] = _artifact_ref(path, root, f"test-{name}")
    supporting_names = EVIDENCE_REF_KEYS - frozenset(
        {"history", "terminal_numerical", "raw_trace", "trace_intervals", "execution"}
    )
    execution_path = root / "execution.json"
    if real_authorities:
        source_sha = hashlib.sha256(canonical_json_bytes(process_source)).hexdigest()
        cold_runtime_document = json.loads(
            (root / refs["runtime"].relative_path).read_bytes()
        )
        environment_sha = cold_runtime_document["runtime_identity"][
            "effective_environment_sha256"
        ]
        cold_process = json.loads((root / refs["process"].relative_path).read_bytes())
        cold_argv = cold_process["argv"]
        execution_pid = 20
        execution_start_ticks = 21
        physical_memory_bytes = 10 * 1024 * 1024
        peak_memory_bytes = 1024 * 1024
        peak_memory_fraction = 0.1
        interpreter = "/fixture/python"
        stderr_sha = hashlib.sha256(b"").hexdigest()
    else:
        source_sha = "1" * 64
        environment_sha = "2" * 64
        cold_argv = ["/python", "runner.py"]
        execution_pid = 1
        execution_start_ticks = 1
        physical_memory_bytes = 1000
        peak_memory_bytes = 500
        peak_memory_fraction = 0.5
        interpreter = "/python"
        stderr_sha = "3" * 64
    execution_path.write_bytes(
        canonical_json_bytes(
            execution_evidence_payload(
                supporting_evidence={name: refs[name] for name in supporting_names},
                preflight={
                    "status": "COMPLETE",
                    "compile_success": True,
                    "solver_dispatched": False,
                    "finalizer_called": False,
                    "endpoint_audit_called": False,
                    "campaign_authorized": False,
                    "callbacks": 0,
                },
                cold={
                    "status": "COMPLETE",
                    "child_pid": execution_pid,
                    "child_start_time_ticks": execution_start_ticks,
                    "backend": "gpu",
                    "gpu_uuid": GPU_UUID,
                    "jax_enable_x64": True,
                    "state_size": 716,
                    "equality_size": 255,
                    "residual_size": 2110,
                    "policy_sha256": policy.policy_sha256,
                    "phase_schema_sha256": PHASE_SCHEMA_SHA256,
                    "source_pre_sha256": source_sha,
                    "source_post_sha256": source_sha,
                    "runtime_environment_sha256": environment_sha,
                    "interpreter": interpreter,
                    "argv": cold_argv,
                    "physical_memory_bytes": physical_memory_bytes,
                    "peak_memory_bytes": peak_memory_bytes,
                    "peak_memory_fraction": peak_memory_fraction,
                    "hot_h2d_transfers": 0,
                    "hot_d2h_transfers": 0,
                    "python_callbacks": 0,
                    "final_d2h_transfers": 1,
                    "timestamps_ns": {
                        name: index + 1 for index, name in enumerate(timestamps)
                    },
                    "stdout_sha256": refs["producer"].sha256,
                    "stdout_size_bytes": refs["producer"].size_bytes,
                    "stderr_sha256": stderr_sha,
                    "stderr_size_bytes": 0,
                },
            )
        )
    )
    refs["execution"] = _artifact_ref(
        execution_path, root, f"{receipt_module.SCHEMA_VERSION}-execution"
    )
    if diag2:
        frozen_subset = _write_canonical_ref(
            root,
            receipt_module.DIAG2_EVIDENCE_SLOT_PATHS["frozen_numerical_subset"],
            receipt_module.build_diag2_frozen_numerical_subset_payload(),
            receipt_module.DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        )
        artifact_refs: dict[str, ArtifactRef | None] = {
            "source_manifest": refs["source_manifest"],
            "frozen_numerical_subset": frozen_subset,
            "native_reference": refs["native_reference"],
            "policy_authority": refs["policy_authority"],
            "supervisor_before_preflight": _diag2_zero_ref(
                root, stage="BEFORE_PREFLIGHT", monotonic_ns=10
            ),
            "preflight_producer": refs["preflight"],
            "preflight_terminal": refs["preflight_child_terminal"],
            "preflight_process": refs["preflight_process"],
            "preflight_memory": refs["preflight_memory"],
            "preflight_memory_samples": refs["preflight_memory_samples"],
            "preflight_runtime": refs["preflight_runtime"],
            "preflight_policy": refs["preflight_policy"],
            "supervisor_before_cold": _diag2_zero_ref(
                root, stage="BEFORE_COLD", monotonic_ns=40
            ),
            "cold_producer": refs["producer"],
            "cold_terminal": refs["child_terminal"],
            "cold_process": refs["process"],
            "cold_memory": refs["memory"],
            "cold_memory_samples": refs["memory_samples"],
            "cold_runtime": refs["runtime"],
            "cold_policy": refs["policy"],
            "cold_history": refs["history"],
            "cold_terminal_numerical": refs["terminal_numerical"],
            "cold_raw_trace": refs["raw_trace"],
            "cold_trace_intervals": refs["trace_intervals"],
            "execution": refs["execution"],
            "supervisor_terminal": None,
        }
        algorithm_route = receipt_module.derive_diag2_algorithm_route(
            artifact_root=root, artifact_refs=artifact_refs
        )
        terminal_payload = receipt_module.build_diag2_supervisor_terminal_payload(
            disposition="COMPLETE",
            failure=None,
            launched_children=("preflight", "cold"),
            policy_authority_produced=True,
            preflight_authorized=True,
            cold_authorized=True,
            staging_root=root,
            final_root=root.parent / "diag2",
            nonce=root.name.rsplit(".partial-", 1)[-1],
            algorithm_route_selection=algorithm_route,
        )
        artifact_refs["supervisor_terminal"] = _write_canonical_ref(
            root,
            receipt_module.DIAG2_EVIDENCE_SLOT_PATHS["supervisor_terminal"],
            terminal_payload,
            receipt_module.DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        )
        slots = {
            name: receipt_module.EvidenceSlot.present(reference)
            for name, reference in artifact_refs.items()
            if reference is not None
        }
        return receipt_module.build_diag2_diagnostic_receipt(
            artifact_root=root, evidence_slots=slots
        )
    if not real_authorities:
        monkeypatch.setattr(
            receipt_module, "_validate_native_equalities_authority", lambda *_args: None
        )
        monkeypatch.setattr(
            receipt_module, "_validate_execution_authorities", lambda *_args: None
        )
    return receipt_module.build_diagnostic_receipt(
        artifact_root=root, evidence_refs=refs
    )


def _sealed_incomplete(root: Path, *, partial_trace: tuple[str, ...] = ()) -> Path:
    root.mkdir()
    trace_root = root / "cold/raw-trace/plugins/profile/run"
    for suffix in partial_trace:
        path = trace_root / f"host{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained-partial-trace")
    references: dict[str, ArtifactRef | None] = {
        name: None for name in EVIDENCE_REF_KEYS
    }
    receipt = build_incomplete_diagnostic_receipt(
        artifact_root=root,
        evidence_refs=references,
    )
    receipt_path = root / RECEIPT_FILENAME
    receipt_path.write_bytes(diagnostic_receipt_bytes(receipt))
    manifest = diagnostic_artifact_manifest_payload(root)
    (root / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
    return root


def _sealed_real_complete(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> receipt_module.DiagnosticReceipt:
    receipt = _complete_receipt(root, monkeypatch, real_authorities=True)
    (root / RECEIPT_FILENAME).write_bytes(diagnostic_receipt_bytes(receipt))
    (root / MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(diagnostic_artifact_manifest_payload(root))
    )
    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
    return receipt


def _sealed_source_capture_failure(root: Path) -> Path:
    root.mkdir()
    corrupted = root / "source-snapshot/source-manifest.json"
    corrupted.parent.mkdir(parents=True)
    corrupted.write_bytes(b"retained-corrupt-snapshot-bytes")
    terminal_path = root / "cold/terminal.json"
    terminal_path.parent.mkdir()
    terminal_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": f"{receipt_module.SCHEMA_VERSION}-child-terminal",
                "terminal_status": "PROTOCOL_FAILURE",
                "failure_reasons": ["SOURCE_PRE:ValueError:" + "0" * 64],
            }
        )
    )
    references: dict[str, ArtifactRef | None] = {
        name: None for name in EVIDENCE_REF_KEYS
    }
    references["child_terminal"] = _artifact_ref(
        terminal_path,
        root,
        f"{receipt_module.SCHEMA_VERSION}-child-terminal",
    )
    receipt = build_incomplete_diagnostic_receipt(
        artifact_root=root, evidence_refs=references
    )
    assert receipt.failure_stage is FailureStage.COLD_SOURCE
    (root / RECEIPT_FILENAME).write_bytes(diagnostic_receipt_bytes(receipt))
    (root / MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(diagnostic_artifact_manifest_payload(root))
    )
    for path in root.rglob("*"):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)
    return root


def _rehash_outer_manifest(root: Path) -> None:
    manifest_path = root / MANIFEST_FILENAME
    os.chmod(manifest_path, 0o644)
    manifest = json.loads(manifest_path.read_bytes())
    for entry in manifest["entries"]:
        data = (root / entry["relative_path"]).read_bytes()
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["size_bytes"] = len(data)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    os.chmod(manifest_path, 0o444)


def _write_trace(path: Path, *, unnamed_duration_us: float = 0.25) -> None:
    events: list[dict[str, object]] = [
        {
            "ph": "X",
            "cat": "host",
            "name": TRACE_LOOP_ENVELOPE_NAME,
            "ts": 24_000.0,
            "dur": 7.25,
            "args": {},
        }
    ]
    for index, phase in enumerate(PHASE_IDS):
        events.append(
            {
                "ph": "X",
                "cat": None,
                "name": f"kernel-{index}",
                "ts": 24_000.0 + index,
                "dur": 1.0,
                "args": {
                    "name": phase,
                    "kernel_details": "grid:1,1,1 block:32,1,1",
                },
            }
        )
    events.append(
        {
            "ph": "X",
            "cat": None,
            "name": "unattributed-kernel",
            "ts": 24_007.0,
            "dur": unnamed_duration_us,
            "args": {
                "kernel_details": "grid:1,1,1 block:32,1,1",
                "cuda_graph_id": "4",
                "cuda_graph_node_id": "5",
            },
        }
    )
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"traceEvents": events}, stream, separators=(",", ":"))


def test_plan_and_phase_schema_identities_are_frozen() -> None:
    assert (
        PLAN_SHA256
        == "e6871072a7011d64e511aa8e8cf7db17d36acedbb33dbbce22b18cd0ae2c6d59"
    )
    expected = hashlib.sha256(
        json.dumps(PHASE_IDS, separators=(",", ":")).encode()
    ).hexdigest()
    assert PHASE_SCHEMA_SHA256 == expected


def test_history_array_constructor_derives_retained_aggregate() -> None:
    payload = history_evidence_from_arrays(
        _history_arrays(),
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    assert payload["attempts"] == 300
    assert payload["accepted_steps"] == 203
    assert payload["retryable_rejections"] == 97
    assert payload["status"] == "ATTEMPT_LIMIT"
    assert payload["rows"][202]["accepted_step_number"] == 203  # type: ignore[index]
    assert payload["rows"][203]["outcome"] == AttemptOutcome.RETRY_OBJECTIVE.value  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("outcome", np.zeros(300, dtype=np.bool_), TypeError),
        ("outcome", np.zeros(299, dtype=np.int32), ValueError),
        ("outcome", np.full(300, 10, dtype=np.int32), ValueError),
        ("trust_radius", np.full(300, np.inf, dtype=np.float64), ValueError),
    ),
)
def test_history_array_constructor_rejects_mutations(
    field: str, replacement: np.ndarray, error: type[Exception]
) -> None:
    history = _history_arrays()
    setattr(history, field, replacement)
    with pytest.raises(error):
        history_evidence_from_arrays(
            history,
            quality_latch=False,
            first_quality_attempt=0,
            first_quality_accepted_step=0,
        )


def test_history_latch_must_be_terminal_first_hit() -> None:
    with pytest.raises(ValueError, match="first-hit row"):
        history_evidence_from_arrays(
            _history_arrays(),
            quality_latch=True,
            first_quality_attempt=299,
            first_quality_accepted_step=203,
        )


def _single_outcome_history(outcome_code: int) -> SimpleNamespace:
    history = _history_arrays()
    history.outcome[:] = 0
    history.accepted_step_number[:] = 0
    history.outcome[0] = outcome_code
    for name in HISTORY_INTEGER_FIELDS:
        getattr(history, name)[:] = 0
    history.steihaug_hit_boundary[:] = False
    for name in HISTORY_FLOAT_FIELDS:
        getattr(history, name)[:] = np.nan
    history.trust_radius[0] = 1.0
    history.next_trust_radius[0] = 1.0
    return history


@pytest.mark.parametrize("outcome_code", (7, 8))
def test_fatal_history_stage_prefixes_are_exact(outcome_code: int) -> None:
    history = _single_outcome_history(outcome_code)
    base_fields = (
        "current_objective",
        "current_feasibility_inf",
        "current_stationarity_inf",
        "residual_value_defect",
        "residual_gradient_defect",
        "hvp_symmetry_defect",
        "probe_normalized_curvature",
        "current_projection_tangency_relative_residual",
        "current_projection_solve_relative_residual",
        "current_projection_forward_error_bound",
    )
    if outcome_code == 8:
        for name in base_fields:
            getattr(history, name)[0] = 0.0
    history_evidence_from_arrays(
        history,
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    history.candidate_objective[0] = 0.0
    with pytest.raises(ValueError, match="before its stage"):
        history_evidence_from_arrays(
            history,
            quality_latch=False,
            first_quality_attempt=0,
            first_quality_accepted_step=0,
        )


def test_retry_nonfinite_requires_complete_prior_stages_and_raw_null() -> None:
    history = _single_outcome_history(3)
    for name in HISTORY_FLOAT_FIELDS:
        getattr(history, name)[0] = 1.0
    history.candidate_objective[0] = np.nan
    history_evidence_from_arrays(
        history,
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    history.candidate_objective[0] = 1.0
    with pytest.raises(ValueError, match="no null raw telemetry"):
        history_evidence_from_arrays(
            history,
            quality_latch=False,
            first_quality_attempt=0,
            first_quality_accepted_step=0,
        )


@pytest.mark.parametrize("curvature_stage_complete", (False, True))
def test_fatal_curvature_code9_raw_stage_forms(
    curvature_stage_complete: bool,
) -> None:
    history = _single_outcome_history(9)
    base_fields = (
        HISTORY_FLOAT_FIELDS[:3]
        + HISTORY_FLOAT_FIELDS[16:20]
        + HISTORY_FLOAT_FIELDS[25:28]
    )
    step_fields = (
        HISTORY_FLOAT_FIELDS[10:11]
        + HISTORY_FLOAT_FIELDS[15:16]
        + HISTORY_FLOAT_FIELDS[20:21]
        + HISTORY_FLOAT_FIELDS[28:]
    )
    for name in base_fields:
        getattr(history, name)[0] = 0.0
    if curvature_stage_complete:
        for name in step_fields:
            getattr(history, name)[0] = 0.0
    payload = history_evidence_from_arrays(
        history,
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    assert payload["rows"][0]["outcome"] == AttemptOutcome.FATAL_CURVATURE.value  # type: ignore[index]


@pytest.mark.parametrize(
    ("outcome_code", "invalidating_field", "invalidating_value"),
    (
        (4, "correction_relative_residual", 2.0e-10),
        (5, "candidate_feasibility_inf", 2.0e-10),
        (6, "correction_norm", 2.0e-3),
    ),
)
def test_retry_certificate_codes_raw_parse(
    outcome_code: int, invalidating_field: str, invalidating_value: float
) -> None:
    history = _single_outcome_history(outcome_code)
    for name in HISTORY_FLOAT_FIELDS:
        getattr(history, name)[0] = 0.0
    history.trust_radius[0] = 1.0
    history.next_trust_radius[0] = 1.0
    history.tangent_step_norm[0] = 1.0
    history.predicted_reduction[0] = 1.0
    history.actual_reduction[0] = 1.0
    history.reduction_ratio[0] = 1.0
    getattr(history, invalidating_field)[0] = invalidating_value
    payload = history_evidence_from_arrays(
        history,
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    assert payload["rows"][0]["outcome"] == tuple(AttemptOutcome)[outcome_code].value  # type: ignore[index]


@pytest.mark.parametrize(
    "missing_trial_field",
    (
        "candidate_objective",
        "candidate_feasibility_inf",
        "actual_reduction",
        "predicted_reduction",
        "correction_norm",
        "applied_step_norm",
        "correction_step_ratio",
        "corrected_radius_ratio",
        "correction_relative_residual",
        "correction_forward_error_bound",
        "trial_gram_factorization_relative_residual",
        "trial_gram_solve_relative_residual",
    ),
)
def test_retry_nonfinite_code3_raw_stage_forms(missing_trial_field: str) -> None:
    history = _single_outcome_history(3)
    for name in HISTORY_FLOAT_FIELDS:
        getattr(history, name)[0] = 0.0
    history.trust_radius[0] = 1.0
    history.next_trust_radius[0] = 1.0
    getattr(history, missing_trial_field)[0] = np.nan
    payload = history_evidence_from_arrays(
        history,
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    assert payload["rows"][0]["outcome"] == AttemptOutcome.RETRY_NONFINITE.value  # type: ignore[index]


@pytest.mark.parametrize(("accepted", "quality_hit"), ((1, True), (256, False)))
def test_history_accept_limit_and_early_hit_have_canonical_inactive_tail(
    accepted: int, quality_hit: bool
) -> None:
    history = _history_arrays()
    history.outcome[:] = 0
    history.accepted_step_number[:] = 0
    history.outcome[:accepted] = 1
    history.accepted_step_number[:accepted] = np.arange(1, accepted + 1)
    history.actual_reduction[:accepted] = 1.0
    history.predicted_reduction[:accepted] = 1.0
    history.reduction_ratio[:accepted] = 1.0
    for name in HISTORY_INTEGER_FIELDS:
        getattr(history, name)[accepted:] = 0
    history.steihaug_hit_boundary[accepted:] = False
    for name in HISTORY_FLOAT_FIELDS:
        getattr(history, name)[accepted:] = np.nan
    payload = history_evidence_from_arrays(
        history,
        quality_latch=quality_hit,
        first_quality_attempt=accepted if quality_hit else 0,
        first_quality_accepted_step=accepted if quality_hit else 0,
    )
    assert payload["accepted_steps"] == accepted
    assert payload["bounded_complete"] is (accepted == 256)


def test_history_zero_prediction_has_exact_null_ratio_applicability() -> None:
    history = _history_arrays()
    history.predicted_reduction[203] = 0.0
    history.reduction_ratio[203] = np.nan
    history_evidence_from_arrays(
        history,
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    history.reduction_ratio[203] = 0.0
    with pytest.raises(ValueError, match="must be null for zero prediction"):
        history_evidence_from_arrays(
            history,
            quality_latch=False,
            first_quality_attempt=0,
            first_quality_accepted_step=0,
        )


def test_chrome_trace_uses_local_clock_and_counts_unnamed_cuda(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cold.trace.json.gz"
    _write_trace(path)
    payload = normalize_chrome_trace(path, phase_schema_sha256=PHASE_SCHEMA_SHA256)
    intervals = payload["device_intervals"]
    assert isinstance(intervals, list)
    assert len(intervals) == 8
    assert payload["trace_start_ns"] == 24_000_000
    assert payload["trace_stop_ns"] == 24_007_250
    assert intervals[-1]["scope_paths"] == []  # type: ignore[index]


def test_chrome_trace_does_not_count_host_scope_as_device(tmp_path: Path) -> None:
    path = tmp_path / "host-only.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "host",
                        "name": TRACE_LOOP_ENVELOPE_NAME,
                        "ts": 0,
                        "dur": 3,
                        "args": {},
                    },
                    {
                        "ph": "X",
                        "cat": "host",
                        "ts": 1,
                        "dur": 1,
                        "args": {"name": PHASE_IDS[0]},
                    },
                ]
            },
            stream,
        )
    with pytest.raises(ValueError, match="no in-envelope device intervals"):
        normalize_chrome_trace(path, phase_schema_sha256=PHASE_SCHEMA_SHA256)


def test_trace_rejects_wrong_phase_schema(tmp_path: Path) -> None:
    path = tmp_path / "cold.trace.json.gz"
    _write_trace(path)
    with pytest.raises(ValueError, match="frozen diagnostic phases"):
        normalize_chrome_trace(path, phase_schema_sha256="0" * 64)


@pytest.mark.parametrize("mutation", ("duplicate-envelope", "outside", "substring"))
def test_trace_envelope_and_exact_scope_rules(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "mutated.trace.json.gz"
    envelope = {
        "ph": "X",
        "cat": "host",
        "name": TRACE_LOOP_ENVELOPE_NAME,
        "ts": 10.0,
        "dur": 10.0,
        "args": {},
    }
    kernel = {
        "ph": "X",
        "cat": None,
        "name": "kernel",
        "ts": 11.0,
        "dur": 1.0,
        "args": {"name": PHASE_IDS[0], "kernel_details": "kernel"},
    }
    events = [envelope, kernel]
    if mutation == "duplicate-envelope":
        events.append(dict(envelope))
    elif mutation == "outside":
        kernel["ts"] = 9.5
    else:
        kernel["args"] = {
            "name": f"prefix-{PHASE_IDS[0]}-suffix",
            "kernel_details": "kernel",
        }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"traceEvents": events}, stream, separators=(",", ":"))
    if mutation == "duplicate-envelope":
        with pytest.raises(ValueError, match="exactly one loop envelope"):
            normalize_chrome_trace(path, phase_schema_sha256=PHASE_SCHEMA_SHA256)
    elif mutation == "outside":
        with pytest.raises(ValueError, match="outside the loop envelope"):
            normalize_chrome_trace(path, phase_schema_sha256=PHASE_SCHEMA_SHA256)
    else:
        payload = normalize_chrome_trace(path, phase_schema_sha256=PHASE_SCHEMA_SHA256)
        assert payload["device_intervals"][0]["scope_paths"] == []  # type: ignore[index]


def test_incomplete_receipt_round_trip_with_absent_evidence(tmp_path: Path) -> None:
    receipt = _incomplete(tmp_path)
    rebuilt = load_diagnostic_receipt_bytes(
        diagnostic_receipt_bytes(receipt), artifact_root=tmp_path
    )
    assert rebuilt == receipt
    payload = diagnostic_receipt_payload(rebuilt)
    assert payload["verdict"] == DiagnosticVerdict.INCOMPLETE.value
    assert payload["promotion_authorized"] is False
    assert all(value is None for value in payload["evidence_refs"].values())  # type: ignore[union-attr]


@pytest.mark.parametrize("encoding", ("duplicate", "pretty", "nan"))
def test_receipt_bytes_reject_noncanonical_json(tmp_path: Path, encoding: str) -> None:
    payload = diagnostic_receipt_payload(_incomplete(tmp_path))
    if encoding == "duplicate":
        data = diagnostic_receipt_bytes(_incomplete(tmp_path)).replace(
            b'{"engineering_campaign_receipt_produced"',
            b'{"schema_version":"duplicate","engineering_campaign_receipt_produced"',
            1,
        )
    elif encoding == "pretty":
        data = json.dumps(payload, indent=2).encode()
    else:
        data = diagnostic_receipt_bytes(_incomplete(tmp_path)).replace(
            b'"reuse_opportunity_estimate":null',
            b'"reuse_opportunity_estimate":NaN',
        )
    with pytest.raises((TypeError, ValueError)):
        load_diagnostic_receipt_bytes(data, artifact_root=tmp_path)


def test_deep_loader_accepts_sealed_incomplete_artifact(tmp_path: Path) -> None:
    receipt = load_and_validate_diagnostic_artifact(
        _sealed_incomplete(tmp_path / "run")
    )
    assert isinstance(receipt, IncompleteDiagnosticReceipt)
    assert receipt.failure_stage is FailureStage.PREFLIGHT


@pytest.mark.parametrize(
    "suffixes",
    ((".trace.json.gz",), (".xplane.pb",), (".trace.json.gz", ".xplane.pb")),
)
def test_sealed_incomplete_retains_only_canonical_partial_trace_roles(
    tmp_path: Path, suffixes: tuple[str, ...]
) -> None:
    root = _sealed_incomplete(tmp_path / "run", partial_trace=suffixes)
    receipt = load_and_validate_diagnostic_artifact(root)
    assert isinstance(receipt, IncompleteDiagnosticReceipt)
    manifest = json.loads((root / MANIFEST_FILENAME).read_bytes())
    roles = {entry["role"] for entry in manifest["entries"]}
    assert roles >= {
        "raw_trace_chrome" if suffix.endswith(".json.gz") else "raw_trace_xplane"
        for suffix in suffixes
    }


def test_sealed_incomplete_retains_corrupt_source_capture_as_opaque(
    tmp_path: Path,
) -> None:
    root = _sealed_source_capture_failure(tmp_path / "source-failure")
    receipt = load_and_validate_diagnostic_artifact(root)
    assert isinstance(receipt, IncompleteDiagnosticReceipt)
    assert receipt.failure_stage is FailureStage.COLD_SOURCE
    manifest = json.loads((root / MANIFEST_FILENAME).read_bytes())
    roles = {entry["relative_path"]: entry["role"] for entry in manifest["entries"]}
    assert (
        roles["source-snapshot/source-manifest.json"]
        == "source_snapshot_opaque_failure"
    )


@pytest.mark.parametrize(
    "mutation", ("tamper", "missing", "writable", "extra", "symlink", "role")
)
def test_deep_loader_rejects_artifact_tree_mutations(
    tmp_path: Path, mutation: str
) -> None:
    root = _sealed_incomplete(tmp_path / "run")
    receipt_path = root / RECEIPT_FILENAME
    if mutation == "tamper":
        os.chmod(receipt_path, 0o644)
        receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
        os.chmod(receipt_path, 0o444)
    elif mutation == "missing":
        os.chmod(root, 0o755)
        receipt_path.unlink()
        os.chmod(root, 0o555)
    elif mutation == "writable":
        os.chmod(receipt_path, 0o644)
    elif mutation == "extra":
        os.chmod(root, 0o755)
        extra = root / "extra.txt"
        extra.write_text("extra")
        os.chmod(extra, 0o444)
        os.chmod(root, 0o555)
    elif mutation == "symlink":
        os.chmod(root, 0o755)
        (root / "alias").symlink_to(RECEIPT_FILENAME)
        os.chmod(root, 0o555)
    else:
        manifest_path = root / MANIFEST_FILENAME
        os.chmod(manifest_path, 0o644)
        payload = json.loads(manifest_path.read_bytes())
        payload["entries"][0]["role"] = "fabricated_role"
        manifest_path.write_bytes(canonical_json_bytes(payload))
        os.chmod(manifest_path, 0o444)
    with pytest.raises((OSError, ValueError)):
        load_and_validate_diagnostic_artifact(root)


def test_incomplete_builder_derives_stage_and_rejects_wrong_ref_set(
    tmp_path: Path,
) -> None:
    receipt = build_incomplete_diagnostic_receipt(
        artifact_root=tmp_path,
        evidence_refs={name: None for name in EVIDENCE_REF_KEYS},
    )
    assert receipt.failure_stage is FailureStage.PREFLIGHT
    with pytest.raises(ValueError, match="frozen schema"):
        build_incomplete_diagnostic_receipt(
            artifact_root=tmp_path,
            evidence_refs={"unexpected": None},
        )


@pytest.mark.parametrize(
    ("terminal_status", "stage"),
    (
        ("TIMEOUT", FailureStage.COLD_TIMEOUT),
        ("CRASH", FailureStage.COLD_CRASH),
        ("MONITOR_FAILURE", FailureStage.COLD_MONITOR),
        ("PROTOCOL_FAILURE", FailureStage.COLD_PROTOCOL),
    ),
)
def test_incomplete_taxonomy_derives_cold_terminal_class(
    tmp_path: Path, terminal_status: str, stage: FailureStage
) -> None:
    refs: dict[str, ArtifactRef | None] = {name: None for name in EVIDENCE_REF_KEYS}
    for name in (
        "preflight",
        "preflight_process",
        "preflight_memory",
        "preflight_memory_samples",
        "preflight_runtime",
        "preflight_policy",
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(canonical_json_bytes({"schema_version": f"test-{name}"}))
        refs[name] = _artifact_ref(path, tmp_path, f"test-{name}")
    for name, status in (
        ("preflight_child_terminal", "COMPLETE"),
        ("child_terminal", terminal_status),
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(
            canonical_json_bytes(
                {"schema_version": f"test-{name}", "terminal_status": status}
            )
        )
        refs[name] = _artifact_ref(path, tmp_path, f"test-{name}")
    receipt = build_incomplete_diagnostic_receipt(
        artifact_root=tmp_path, evidence_refs=refs
    )
    assert receipt.failure_stage is stage


def _passing_execution() -> ExecutionEvidence:
    timestamp_names = (
        "process_started",
        "compile_started",
        "compile_completed",
        "state_ready",
        "profiler_started",
        "solve_started",
        "solve_stopped",
        "profiler_stopped",
        "finalizer_started",
        "finalizer_stopped",
        "quality_replay_started",
        "quality_replay_stopped",
        "endpoint_diagnostics_started",
        "endpoint_diagnostics_stopped",
        "final_d2h",
        "trace_exported",
        "serialized",
        "process_stopped",
    )
    return ExecutionEvidence(
        supporting_evidence=(),
        preflight_status="COMPLETE",
        preflight_compile_success=True,
        preflight_solver_dispatched=False,
        preflight_finalizer_called=False,
        preflight_endpoint_audit_called=False,
        preflight_campaign_authorized=False,
        preflight_callbacks=0,
        cold_status="COMPLETE",
        child_pid=10,
        child_start_time_ticks=20,
        backend="gpu",
        gpu_uuid=GPU_UUID,
        jax_enable_x64=True,
        state_size=716,
        equality_size=255,
        residual_size=2110,
        policy_sha256="1" * 64,
        phase_schema_sha256=PHASE_SCHEMA_SHA256,
        source_pre_sha256="2" * 64,
        source_post_sha256="2" * 64,
        runtime_environment_sha256="3" * 64,
        interpreter="/python",
        argv=("/python", "runner.py"),
        physical_memory_bytes=1000,
        peak_memory_bytes=500,
        reported_peak_memory_fraction=0.5,
        hot_h2d_transfers=0,
        hot_d2h_transfers=0,
        python_callbacks=0,
        final_d2h_transfers=1,
        timestamps_ns=tuple(
            (name, index + 1) for index, name in enumerate(timestamp_names)
        ),
        stdout_sha256="4" * 64,
        stdout_size_bytes=0,
        stderr_sha256="5" * 64,
        stderr_size_bytes=0,
    )


def test_execution_lifecycle_requires_strict_timestamp_order() -> None:
    execution = _passing_execution()
    assert execution.passes()
    timestamps = list(execution.timestamps_ns)
    timestamps[6] = (timestamps[6][0], timestamps[5][1])
    equal_adjacent = replace(execution, timestamps_ns=tuple(timestamps))
    assert not equal_adjacent.passes()


def test_complete_receipt_public_builder_recomputes_no_hit_and_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _complete_receipt(tmp_path / "complete", monkeypatch)
    assert receipt.verdict is DiagnosticVerdict.NO_HIT
    assert receipt.next_route is receipt_module.NextRoute.RETRY_MODEL_REUSE
    assert receipt.reuse_opportunity_estimate is not None
    assert receipt.reuse_opportunity_estimate >= 0.05


def test_complete_receipt_validates_real_deep_authorities_without_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _complete_receipt(
        tmp_path / "real-complete", monkeypatch, real_authorities=True
    )
    assert receipt.verdict is DiagnosticVerdict.NO_HIT


def test_sealed_complete_artifact_deep_loads_exact_chrome_xplane_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sealed-complete"
    receipt = _sealed_real_complete(root, monkeypatch)
    rebuilt = load_and_validate_diagnostic_artifact(root)
    assert diagnostic_receipt_payload(rebuilt) == diagnostic_receipt_payload(receipt)
    manifest = json.loads((root / MANIFEST_FILENAME).read_bytes())
    roles = {entry["role"] for entry in manifest["entries"]}
    assert {"raw_trace_chrome", "raw_trace_xplane"} <= roles


@pytest.mark.parametrize("authority", ("source", "native-reference"))
def test_complete_deep_loader_rejects_rehashed_nested_authority_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    root = tmp_path / authority
    _sealed_real_complete(root, monkeypatch)
    if authority == "source":
        target = (
            root / "source-snapshot/benchmarks/"
            "single_stage_native_equivalent_quality_diagnostic_receipt.py"
        )
    else:
        reference = json.loads((root / "native-reference/reference.json").read_bytes())
        target = (
            root
            / "native-reference"
            / reference["evidence"]["arrays"]["raw_equalities"]["relative_path"]
        )
    os.chmod(target, 0o644)
    target.write_bytes(target.read_bytes() + b"tamper")
    os.chmod(target, 0o444)
    _rehash_outer_manifest(root)
    with pytest.raises(ValueError):
        load_and_validate_diagnostic_artifact(root)


@pytest.mark.parametrize(
    ("failure_kind", "stage"),
    (
        ("source", FailureStage.COLD_SOURCE),
        ("resource", FailureStage.COLD_RESOURCE),
        ("numerical", FailureStage.NUMERICAL_EVIDENCE),
    ),
)
def test_complete_raw_refs_derive_semantic_incomplete_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    stage: FailureStage,
) -> None:
    root = tmp_path / failure_kind
    receipt = _complete_receipt(root, monkeypatch, real_authorities=True)
    refs = dict(receipt.evidence_refs)
    if failure_kind == "source":
        path = (
            root / "source-snapshot/benchmarks/"
            "single_stage_native_equivalent_quality_diagnostic_receipt.py"
        )
        os.chmod(path, 0o644)
        path.write_bytes(path.read_bytes() + b"tamper")
    elif failure_kind == "resource":
        samples_path = root / refs["memory_samples"].relative_path
        samples = json.loads(samples_path.read_bytes())
        samples["samples"][1]["used_memory_mib"] = 2
        samples_path.write_bytes(canonical_json_bytes(samples))
        refs["memory_samples"] = _artifact_ref(
            samples_path, root, refs["memory_samples"].schema_version
        )
        execution_path = root / refs["execution"].relative_path
        execution = json.loads(execution_path.read_bytes())
        execution["supporting_evidence"]["memory_samples"] = _ref_payload(
            refs["memory_samples"]
        )
        execution_path.write_bytes(canonical_json_bytes(execution))
        refs["execution"] = _artifact_ref(
            execution_path, root, refs["execution"].schema_version
        )
    else:
        array_path = root / "cold/arrays/physical_state.npy"
        with array_path.open("rb") as stream:
            values = np.load(stream, allow_pickle=False)
        values[0] = 1.0
        _write_npy(array_path, values)
        array_ref = _artifact_ref(array_path, root, "array-physical_state")
        terminal_path = root / refs["terminal_numerical"].relative_path
        terminal = json.loads(terminal_path.read_bytes())
        terminal["arrays"]["physical_state"] = receipt_module.array_evidence_payload(
            reference=array_ref, name="physical_state", values=values
        )
        terminal_path.write_bytes(canonical_json_bytes(terminal))
        refs["terminal_numerical"] = _artifact_ref(
            terminal_path, root, refs["terminal_numerical"].schema_version
        )
        producer_path = root / refs["producer"].relative_path
        producer = json.loads(producer_path.read_bytes())
        producer["terminal_numerical_evidence"] = _ref_payload(
            refs["terminal_numerical"]
        )
        producer_path.write_bytes(canonical_json_bytes(producer))
        refs["producer"] = _artifact_ref(
            producer_path, root, refs["producer"].schema_version
        )
        stdout_path = root / "cold/stdout.bin"
        stdout_path.write_bytes(producer_path.read_bytes())
        stdout_ref = _artifact_ref(stdout_path, root, "raw-process-stdout-v1")
        process_path = root / refs["process"].relative_path
        process = json.loads(process_path.read_bytes())
        process["stdout"] = _ref_payload(stdout_ref)
        process_path.write_bytes(canonical_json_bytes(process))
        refs["process"] = _artifact_ref(
            process_path, root, refs["process"].schema_version
        )
        execution_path = root / refs["execution"].relative_path
        execution = json.loads(execution_path.read_bytes())
        execution["supporting_evidence"]["producer"] = _ref_payload(refs["producer"])
        execution["supporting_evidence"]["process"] = _ref_payload(refs["process"])
        execution["stdout_sha256"] = refs["producer"].sha256
        execution["stdout_size_bytes"] = refs["producer"].size_bytes
        execution_path.write_bytes(canonical_json_bytes(execution))
        refs["execution"] = _artifact_ref(
            execution_path, root, refs["execution"].schema_version
        )
    incomplete = build_incomplete_diagnostic_receipt(
        artifact_root=root, evidence_refs=refs
    )
    assert incomplete.failure_stage is stage


def test_preflight_gate_recomputes_raw_policy_before_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "preflight-policy"
    receipt = _complete_receipt(root, monkeypatch, real_authorities=True)
    refs = dict(receipt.evidence_refs)
    policy_path = root / refs["preflight_policy"].relative_path
    policy = json.loads(policy_path.read_bytes())
    native = np.asarray(policy["native_raw_equalities"], dtype=np.float64)
    scale = np.asarray(policy["constraint_inverse_scale"], dtype=np.float64)
    scale[0] *= 2.0
    forged_policy = NativeEquivalentQualityPolicy(
        native, exact_numeric_tree_sha256(native), scale
    )
    policy_path.write_bytes(
        canonical_json_bytes(
            policy_evidence_payload(
                policy_sha256=forged_policy.policy_sha256,
                native_raw_equalities=native,
                constraint_inverse_scale=scale,
            )
        )
    )
    preflight_policy_ref = _artifact_ref(
        policy_path, root, refs["preflight_policy"].schema_version
    )
    producer_path = root / refs["preflight"].relative_path
    producer = json.loads(producer_path.read_bytes())
    producer["policy_sha256"] = forged_policy.policy_sha256
    producer["policy_evidence"] = _ref_payload(preflight_policy_ref)
    producer_path.write_bytes(canonical_json_bytes(producer))
    producer_ref = _artifact_ref(producer_path, root, refs["preflight"].schema_version)
    stdout_path = root / "preflight/stdout.bin"
    stdout_path.write_bytes(producer_path.read_bytes())
    stdout_ref = _artifact_ref(stdout_path, root, "raw-process-stdout-v1")
    process_path = root / refs["preflight_process"].relative_path
    process = json.loads(process_path.read_bytes())
    process["stdout"] = _ref_payload(stdout_ref)
    process_path.write_bytes(canonical_json_bytes(process))
    process_ref = _artifact_ref(
        process_path, root, refs["preflight_process"].schema_version
    )
    evidence = {
        "producer": producer_ref,
        "child_terminal": refs["preflight_child_terminal"],
        "process": process_ref,
        "memory": refs["preflight_memory"],
        "memory_samples": refs["preflight_memory_samples"],
        "runtime": refs["preflight_runtime"],
        "preflight_policy": preflight_policy_ref,
        "policy_authority": refs["policy_authority"],
        "source_manifest": refs["source_manifest"],
        "native_reference": refs["native_reference"],
    }
    with pytest.raises(ValueError, match="independent parent authority"):
        validate_diagnostic_preflight_gate(
            root,
            evidence_refs=evidence,
            expected_gpu_uuid=GPU_UUID,
            physical_memory_bytes=10 * 1024 * 1024,
            expected_interpreter="/fixture/python",
            expected_argv=tuple(process["argv"]),
        )


def test_complete_receipt_independently_reconstructs_quality_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _complete_receipt(tmp_path / "quality-hit", monkeypatch, quality_hit=True)
    assert receipt.verdict is DiagnosticVerdict.QUALITY_HIT
    assert receipt.history.first_quality_attempt == 1
    assert receipt.history.first_quality_accepted_step == 1


@pytest.mark.parametrize(
    "array_name",
    (
        "accepted_quality_objectives",
        "accepted_quality_raw_equalities",
        "accepted_quality_scaled_equalities",
    ),
)
def test_complete_receipt_replay_active_nonfinite_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    array_name: str,
) -> None:
    receipt = _complete_receipt(tmp_path / array_name, monkeypatch)
    evidence = receipt.terminal.array(array_name)
    values = np.array(evidence.values, copy=True)
    values[1] = np.nan
    values.flags.writeable = False
    mutated = replace(evidence, values=values)
    terminal = replace(
        receipt.terminal,
        arrays=tuple(
            (name, mutated if name == array_name else item)
            for name, item in receipt.terminal.arrays
        ),
    )
    with pytest.raises(ValueError):
        receipt_module._validate_quality_replay(
            receipt.history, terminal, receipt.policy
        )


@pytest.mark.parametrize(
    "array_name",
    (
        "accepted_quality_mask",
        "accepted_quality_coordinates_finite",
        "accepted_quality_objective_finite",
        "accepted_quality_raw_equalities_finite",
        "accepted_quality_scaled_equalities_finite",
        "accepted_quality_objective_satisfied",
        "accepted_quality_component_bounds_satisfied",
        "accepted_quality_scaled_feasibility_satisfied",
        "accepted_quality_satisfied",
    ),
)
def test_complete_receipt_recomputes_device_replay_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    array_name: str,
) -> None:
    receipt = _complete_receipt(tmp_path / array_name, monkeypatch)
    evidence = receipt.terminal.array(array_name)
    values = np.array(evidence.values, copy=True)
    values[1] = not bool(values[1])
    values.flags.writeable = False
    terminal = replace(
        receipt.terminal,
        arrays=tuple(
            (name, replace(item, values=values) if name == array_name else item)
            for name, item in receipt.terminal.arrays
        ),
    )
    with pytest.raises(ValueError, match="predicates"):
        receipt_module._validate_quality_replay(
            receipt.history, terminal, receipt.policy
        )


def test_complete_receipt_recomputes_latch_and_policy_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _complete_receipt(tmp_path / "latch", monkeypatch)
    claimed_hit = replace(
        receipt.history,
        quality_latch=True,
        first_quality_attempt=1,
        first_quality_accepted_step=1,
    )
    with pytest.raises(ValueError, match="latch/counters"):
        receipt_module._validate_quality_replay(
            claimed_hit, receipt.terminal, receipt.policy
        )
    relaxed_policy = replace(receipt.policy, objective_target=2.0)
    with pytest.raises(ValueError, match="predicates"):
        receipt_module._validate_quality_replay(
            receipt.history, receipt.terminal, relaxed_policy
        )


@pytest.mark.parametrize(
    ("array_name", "error"),
    (
        ("physical_state", r"z0 \+ S\*u"),
        ("constraint_jacobian", "transpose raw actions"),
        ("raw_stationarity", "finite KKT telemetry"),
    ),
)
def test_complete_receipt_terminal_raw_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    array_name: str,
    error: str,
) -> None:
    receipt = _complete_receipt(tmp_path / array_name, monkeypatch)
    evidence = receipt.terminal.array(array_name)
    values = np.array(evidence.values, copy=True)
    values.flat[0] = 1.0
    values.flags.writeable = False
    terminal = replace(
        receipt.terminal,
        arrays=tuple(
            (name, replace(item, values=values) if name == array_name else item)
            for name, item in receipt.terminal.arrays
        ),
    )
    with pytest.raises(ValueError, match=error):
        receipt_module._validate_terminal_raw_evidence(
            terminal, receipt.history, receipt.policy
        )


@pytest.mark.parametrize(
    ("case", "array_name", "replacement", "validator", "error"),
    (
        (
            "accepted-ledger-nan",
            "accepted_optimizer_ledger",
            np.nan,
            "quality",
            "predicates",
        ),
        (
            "terminal-optimizer",
            "optimizer_coordinates",
            1.0,
            "terminal",
            r"z0 \+ S\*u",
        ),
        (
            "accepted-ledger-value",
            "accepted_optimizer_ledger",
            1.0,
            "terminal",
            "accepted physical ledger",
        ),
        (
            "raw-equalities",
            "raw_equalities",
            1.0,
            "terminal",
            "scaled equalities",
        ),
        (
            "scaled-equalities",
            "scaled_equalities",
            1.0,
            "terminal",
            "scaled equalities",
        ),
        (
            "objective-gradient",
            "objective_gradient",
            1.0,
            "terminal",
            "objective gradients",
        ),
        (
            "scaled-multipliers",
            "multipliers",
            1.0,
            "terminal",
            "KKT telemetry",
        ),
        (
            "variable-scale",
            "variable_scale",
            0.0,
            "terminal",
            r"z0 \+ S\*u",
        ),
    ),
)
def test_rehashed_raw_terminal_vector_mutations_reach_semantic_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    array_name: str,
    replacement: float,
    validator: str,
    error: str,
) -> None:
    root = tmp_path / case
    receipt = _complete_receipt(root, monkeypatch)
    terminal_path = root / "terminal.json"
    payload = json.loads(terminal_path.read_bytes())

    def replace_array(name: str, values: np.ndarray) -> None:
        artifact = payload["arrays"][name]["artifact"]
        path = root / artifact["relative_path"]
        _write_npy(path, values)
        reference = _artifact_ref(path, root, artifact["schema_version"])
        payload["arrays"][name] = receipt_module.array_evidence_payload(
            reference=reference,
            name=name,
            values=values,
        )

    if case == "scaled-multipliers":
        jacobian = np.zeros(receipt_module.ARRAY_SPECS["constraint_jacobian"][1])
        jacobian[0, 0] = 1.0
        replace_array("constraint_jacobian", jacobian)
        jvp = np.zeros(receipt_module.ARRAY_SPECS["transpose_jvp_action"][1])
        jvp[0] = 1.0
        replace_array("transpose_jvp_action", jvp)
        vjp = np.zeros(receipt_module.ARRAY_SPECS["transpose_vjp_action"][1])
        vjp[0] = 1.0
        replace_array("transpose_vjp_action", vjp)
        payload["transpose_primal_dot"] = 1.0
        payload["transpose_adjoint_dot"] = 1.0

    artifact = payload["arrays"][array_name]["artifact"]
    array_path = root / artifact["relative_path"]
    with array_path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
    mutated = np.array(values, copy=True)
    mutated.flat[0] = replacement
    replace_array(array_name, mutated)
    terminal_path.write_bytes(canonical_json_bytes(payload))
    terminal_reference = _artifact_ref(
        terminal_path,
        root,
        f"{receipt_module.SCHEMA_VERSION}-terminal",
    )
    assert (
        terminal_reference.sha256
        == hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    )
    terminal = receipt_module._parse_terminal(
        root,
        receipt_module._load_ref_json(root, terminal_reference, "mutated terminal"),
    )
    with pytest.raises(ValueError, match=error):
        if validator == "quality":
            receipt_module._validate_quality_replay(
                receipt.history, terminal, receipt.policy
            )
        else:
            receipt_module._validate_terminal_raw_evidence(
                terminal, receipt.history, receipt.policy
            )


@pytest.mark.parametrize("mutation", ("status", "raw-value", "final-certificate"))
def test_complete_receipt_recomputes_kkt_and_final_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt = _complete_receipt(tmp_path / mutation, monkeypatch)
    terminal = receipt.terminal
    if mutation == "status":
        terminal = replace(terminal, kkt_status=receipt_module.KktStatus.NONFINITE)
    elif mutation == "raw-value":
        terminal = replace(terminal, raw_kkt_inf=1.0)
    else:
        certificate = dict(terminal.final_certificate)
        certificate["residual_gradient_defect"] = 1.0
        terminal = replace(
            terminal,
            final_certificate=tuple(certificate.items()),
            final_certificate_passes=True,
        )
    with pytest.raises(ValueError):
        receipt_module._validate_terminal_raw_evidence(
            terminal, receipt.history, receipt.policy
        )


@pytest.mark.parametrize("array_name", ("accepted_physical_ledger", "accepted_mask"))
def test_complete_receipt_recomputes_ledger_and_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    array_name: str,
) -> None:
    receipt = _complete_receipt(tmp_path / array_name, monkeypatch)
    evidence = receipt.terminal.array(array_name)
    values = np.array(evidence.values, copy=True)
    values.flat[0] = not bool(values.flat[0]) if values.dtype.kind == "b" else 1.0
    values.flags.writeable = False
    terminal = replace(
        receipt.terminal,
        arrays=tuple(
            (name, replace(item, values=values) if name == array_name else item)
            for name, item in receipt.terminal.arrays
        ),
    )
    if array_name == "accepted_mask":
        assert not receipt_module._terminal_semantics(receipt.history, terminal)
    else:
        with pytest.raises(ValueError, match="accepted physical ledger"):
            receipt_module._validate_terminal_raw_evidence(
                terminal, receipt.history, receipt.policy
            )


@pytest.mark.parametrize("mutation", ("objective-ledger", "option", "scale"))
def test_complete_receipt_parsers_reject_policy_and_objective_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / mutation
    receipt = _complete_receipt(root, monkeypatch)
    if mutation == "objective-ledger":
        payload = json.loads((root / "terminal.json").read_bytes())
        payload["objective_terms"]["residual"] = 2.0
        with pytest.raises(ValueError, match="raw term ledger"):
            receipt_module._parse_terminal(root, payload)
    else:
        payload = json.loads((root / "policy.json").read_bytes())
        if mutation == "option":
            payload["gntr_options"]["maximum_attempts"] = 299
        else:
            payload["constraint_inverse_scale"][0] = 2.0
        with pytest.raises(ValueError):
            receipt_module._parse_policy(payload, receipt.terminal)


@pytest.mark.parametrize(("coverage_ns", "produced"), ((900, True), (899, False)))
def test_phase_coverage_boundary_and_union(coverage_ns: int, produced: bool) -> None:
    intervals: list[dict[str, object]] = []
    per_phase, remainder = divmod(coverage_ns, len(PHASE_IDS))
    cursor = 0
    for index, phase in enumerate(PHASE_IDS):
        duration = per_phase + (index < remainder)
        intervals.append(
            {
                "start_ns": cursor,
                "end_ns": cursor + duration,
                "scope_paths": [[phase]],
            }
        )
        cursor += duration
    intervals.append({"start_ns": coverage_ns, "end_ns": 1000, "scope_paths": []})
    payload = {
        "schema_version": f"{receipt_module.SCHEMA_VERSION}-raw-trace",
        "phase_schema_sha256": PHASE_SCHEMA_SHA256,
        "trace_start_ns": 0,
        "trace_stop_ns": 1000,
        "device_intervals": intervals,
    }
    phases = receipt_module._parse_phases(payload, PHASE_SCHEMA_SHA256)
    assert (phases.status is receipt_module.PhaseTimingStatus.PRODUCED) is produced


def test_phase_ambiguous_deepest_owner_is_nonproduced() -> None:
    payload = {
        "schema_version": f"{receipt_module.SCHEMA_VERSION}-raw-trace",
        "phase_schema_sha256": PHASE_SCHEMA_SHA256,
        "trace_start_ns": 0,
        "trace_stop_ns": 10,
        "device_intervals": [
            {
                "start_ns": 0,
                "end_ns": 10,
                "scope_paths": [[PHASE_IDS[0]], [PHASE_IDS[1]]],
            }
        ],
    }
    phases = receipt_module._parse_phases(payload, PHASE_SCHEMA_SHA256)
    assert phases.status is receipt_module.PhaseTimingStatus.NOT_PRODUCED


def test_phase_deepest_owner_coalesces_union_and_records_pairwise_overlap() -> None:
    intervals: list[dict[str, object]] = [
        {
            "start_ns": 0,
            "end_ns": 5,
            "scope_paths": [[PHASE_IDS[0], PHASE_IDS[1]]],
        },
        {"start_ns": 4, "end_ns": 8, "scope_paths": [[PHASE_IDS[1]]]},
        {"start_ns": 6, "end_ns": 9, "scope_paths": [[PHASE_IDS[2]]]},
    ]
    for index, phase in enumerate(PHASE_IDS[3:], start=10):
        intervals.append(
            {
                "start_ns": index,
                "end_ns": index + 1,
                "scope_paths": [[phase]],
            }
        )
    payload = {
        "schema_version": f"{receipt_module.SCHEMA_VERSION}-raw-trace",
        "phase_schema_sha256": PHASE_SCHEMA_SHA256,
        "trace_start_ns": 0,
        "trace_stop_ns": 20,
        "device_intervals": intervals,
    }
    phases = receipt_module._parse_phases(payload, PHASE_SCHEMA_SHA256)
    durations = dict(phases.durations_ns)
    assert durations[PHASE_IDS[0]] == 0
    assert durations[PHASE_IDS[1]] == 8
    overlaps = {(left, right): value for left, right, value in phases.overlaps_ns}
    assert overlaps[(PHASE_IDS[1], PHASE_IDS[2])] == 2
    assert phases.status is receipt_module.PhaseTimingStatus.NOT_PRODUCED


def test_phase_missing_required_owner_is_nonproduced() -> None:
    payload = {
        "schema_version": f"{receipt_module.SCHEMA_VERSION}-raw-trace",
        "phase_schema_sha256": PHASE_SCHEMA_SHA256,
        "trace_start_ns": 0,
        "trace_stop_ns": 10,
        "device_intervals": [
            {"start_ns": 0, "end_ns": 10, "scope_paths": [[PHASE_IDS[0]]]}
        ],
    }
    phases = receipt_module._parse_phases(payload, PHASE_SCHEMA_SHA256)
    assert phases.status is receipt_module.PhaseTimingStatus.NOT_PRODUCED


@pytest.mark.parametrize(
    ("current_model_ns", "retraction_count", "expected"),
    (
        (300, 0, receipt_module.NextRoute.RETRY_MODEL_REUSE),
        (299, 0, receipt_module.NextRoute.CONDITIONING_MODEL_CHANGE),
        (299, 30, receipt_module.NextRoute.RADIUS_RETRACTION),
        (299, 29, receipt_module.NextRoute.CONDITIONING_MODEL_CHANGE),
    ),
)
def test_route_selection_exact_005_and_010_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_model_ns: int,
    retraction_count: int,
    expected: receipt_module.NextRoute,
) -> None:
    receipt = _complete_receipt(
        tmp_path / f"route-{current_model_ns}-{retraction_count}", monkeypatch
    )
    rows = list(receipt.history.rows)
    for index in range(retraction_count):
        row_index = 203 + index
        rows[row_index] = replace(
            rows[row_index], outcome=AttemptOutcome.RETRY_FEASIBILITY
        )
    history = replace(receipt.history, rows=tuple(rows))
    phases = replace(
        receipt.phases,
        device_active_ns=1940,
        total_attributed_ns=1940,
        unattributed_ns=0,
        current_model_ns=current_model_ns,
        coverage=1.0,
    )
    rebuilt = receipt_module._derive(
        receipt.evidence_refs,
        receipt.policy,
        history,
        receipt.terminal,
        phases,
        receipt.execution,
    )
    assert rebuilt.next_route is expected


def test_phase_failure_forces_incomplete_verdict_and_no_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _complete_receipt(tmp_path / "phase-incomplete", monkeypatch)
    phases = replace(
        receipt.phases, status=receipt_module.PhaseTimingStatus.NOT_PRODUCED
    )
    rebuilt = receipt_module._derive(
        receipt.evidence_refs,
        receipt.policy,
        receipt.history,
        receipt.terminal,
        phases,
        receipt.execution,
    )
    assert rebuilt.verdict is DiagnosticVerdict.INCOMPLETE
    assert rebuilt.next_route is receipt_module.NextRoute.NOT_SELECTED


def test_history_raw_field_mutation_matrix_is_schema_complete() -> None:
    payload = history_evidence_from_arrays(
        _history_arrays(),
        quality_latch=False,
        first_quality_attempt=0,
        first_quality_accepted_step=0,
    )
    row = payload["rows"][0]  # type: ignore[index]
    assert frozenset(row) == frozenset(HISTORY_ROW_RAW_FIELDS)
    for field in HISTORY_ROW_RAW_FIELDS:
        mutated = dict(row)
        mutated[field] = 0 if field == "steihaug_hit_boundary" else True
        with pytest.raises((TypeError, ValueError)):
            receipt_module._history_row(mutated, 0)


def test_terminal_raw_mutation_matrix_is_schema_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "terminal-matrix"
    _complete_receipt(root, monkeypatch)
    original = json.loads((root / "terminal.json").read_bytes())
    expected = {
        "schema_version",
        "arrays",
        "objective_terms",
        "objective_weights",
        "final_certificate",
        "kkt_status",
        *TERMINAL_RAW_SCALAR_FIELDS,
    }
    assert frozenset(original) == frozenset(expected)
    for field in TERMINAL_RAW_SCALAR_FIELDS:
        payload = json.loads((root / "terminal.json").read_bytes())
        payload[field] = True
        with pytest.raises((TypeError, ValueError)):
            receipt_module._parse_terminal(root, payload)
    payload = json.loads((root / "terminal.json").read_bytes())
    payload["kkt_status"] = "FABRICATED"
    with pytest.raises(ValueError):
        receipt_module._parse_terminal(root, payload)
    for nested in ("objective_terms", "objective_weights"):
        for field in original[nested]:
            payload = json.loads((root / "terminal.json").read_bytes())
            payload[nested][field] = True
            with pytest.raises((TypeError, ValueError)):
                receipt_module._parse_terminal(root, payload)
    assert frozenset(original["final_certificate"]) == frozenset(
        FINAL_CERTIFICATE_FIELDS
    )
    for field in FINAL_CERTIFICATE_FIELDS:
        payload = json.loads((root / "terminal.json").read_bytes())
        payload["final_certificate"][field] = True
        with pytest.raises((TypeError, ValueError)):
            receipt_module._parse_terminal(root, payload)
    assert frozenset(original["arrays"]) == frozenset(receipt_module.ARRAY_SPECS)
    for name in receipt_module.ARRAY_SPECS:
        payload = json.loads((root / "terminal.json").read_bytes())
        payload["arrays"][name]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content digest"):
            receipt_module._parse_terminal(root, payload)


def test_policy_raw_mutation_matrix_is_schema_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "policy-matrix"
    receipt = _complete_receipt(root, monkeypatch)
    original = json.loads((root / "policy.json").read_bytes())
    expected = {
        "schema_version",
        "gntr_options",
        *POLICY_RAW_HASH_FIELDS,
        *POLICY_RAW_VECTOR_FIELDS,
        *POLICY_RAW_SCALAR_FIELDS,
    }
    assert frozenset(original) == frozenset(expected)
    for field in POLICY_RAW_HASH_FIELDS:
        payload = json.loads((root / "policy.json").read_bytes())
        payload[field] = "0" * 64
        with pytest.raises(ValueError):
            receipt_module._parse_policy(payload, receipt.terminal)
    for field in POLICY_RAW_VECTOR_FIELDS:
        payload = json.loads((root / "policy.json").read_bytes())
        payload[field][0] = True
        with pytest.raises((TypeError, ValueError)):
            receipt_module._parse_policy(payload, receipt.terminal)
    for field in POLICY_RAW_SCALAR_FIELDS:
        payload = json.loads((root / "policy.json").read_bytes())
        payload[field] = True
        with pytest.raises((TypeError, ValueError)):
            receipt_module._parse_policy(payload, receipt.terminal)
    assert frozenset(original["gntr_options"]) == frozenset(
        receipt_module.FROZEN_GNTR_OPTIONS
    )
    for field in receipt_module.FROZEN_GNTR_OPTIONS:
        payload = json.loads((root / "policy.json").read_bytes())
        payload["gntr_options"][field] = True
        with pytest.raises((TypeError, ValueError)):
            receipt_module._parse_policy(payload, receipt.terminal)


def test_diag5_contract_is_independent_and_exact() -> None:
    assert receipt_module.DIAG5_PLAN_SHA256 == (
        "618cb703d507ee280e4ff861ec848e2ac76f586517e33f4f4720c04fe8e663e9"
    )
    assert len(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS) == 26
    assert tuple(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS) == tuple(
        receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS
    )
    assert receipt_module.FailureStageV5 is not receipt_module.FailureStageV4
    assert receipt_module.FailureReasonCodeV5 is not receipt_module.FailureReasonCodeV4
    assert receipt_module.EvidenceSlotV5 is not receipt_module.EvidenceSlotV4
    assert receipt_module.StructuredFailureV5 is not receipt_module.StructuredFailureV4
    assert (
        receipt_module.SolveTimingEvidenceV5 is not receipt_module.SolveTimingEvidenceV4
    )
    assert (
        receipt_module.SafeguardTelemetryV5 is not receipt_module.SafeguardTelemetryV4
    )
    expected_counts = (5, 5, 5, 7, 5, 6, 10, 4, 5, 3)
    assert (
        tuple(map(len, receipt_module.DIAG5_STAGE_REASON_ORDER.values()))
        == expected_counts
    )


def test_diag5_slot_rejects_predecessor_and_wrong_successor_schema() -> None:
    name = "cold_solve_timing"
    reference = ArtifactRef(
        relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS[name],
        sha256="1" * 64,
        size_bytes=17,
        schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[name],
    )
    payload = receipt_module.diag5_evidence_slot_payload(
        receipt_module.EvidenceSlotV5.present(reference)
    )
    parsed = receipt_module.parse_diag5_evidence_slot(payload, name=name)
    assert isinstance(parsed, receipt_module.EvidenceSlotV5)
    mutated = json.loads(json.dumps(payload))
    mutated["artifact"]["schema_version"] = (
        receipt_module.DIAG4_SOLVE_TIMING_SCHEMA_VERSION
    )
    with pytest.raises(ValueError, match="schema differs"):
        receipt_module.parse_diag5_evidence_slot(mutated, name=name)


def test_diag5_public_call_graph_does_not_dispatch_to_public_diag4_api() -> None:
    module_tree = ast.parse(inspect.getsource(receipt_module))
    offenders: list[tuple[str, str]] = []
    for node in module_tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "diag5" not in node.name:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id.startswith(
                    ("validate_diag4_", "parse_diag4_", "load_diag4_")
                )
            ):
                offenders.append((node.name, child.func.id))
    assert offenders == []


def test_diag5_compile_failure_producer_is_a_closed_union() -> None:
    runtime = {
        "backend": "gpu",
        "device": "gpu",
        "device_uuid": GPU_UUID,
        "jax": "0.0",
        "jax_enable_x64": True,
        "jaxlib": "0.0",
        "python": "3.0",
    }
    runtime_ref = ArtifactRef(
        relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS["preflight_runtime"],
        sha256="2" * 64,
        size_bytes=23,
        schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS["preflight_runtime"],
    )
    payload = receipt_module.build_diag5_compile_failure_producer_payload(
        execution_status="COMPILE_FAILURE",
        runtime=runtime,
        runtime_evidence=runtime_ref,
        compile_started_ns=10,
        compile_completed_ns=20,
        process_seconds_before_serialization=1.0,
        failure_reason="compile failed",
    )
    assert (
        receipt_module.validate_diag5_producer_payload(payload, mode="preflight")
        == payload
    )
    legacy = dict(payload)
    legacy["schema_version"] = receipt_module.DIAG4_PREFLIGHT_SCHEMA_VERSION
    with pytest.raises(ValueError):
        receipt_module.validate_diag5_producer_payload(legacy, mode="preflight")


def test_diag5_stage_vector_rejects_holes() -> None:
    slots = {
        name: receipt_module.EvidenceSlotV5.absent()
        for name in receipt_module.DIAG5_EVIDENCE_SLOT_PATHS
    }
    source = ArtifactRef(
        relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS["source_manifest"],
        sha256="3" * 64,
        size_bytes=31,
        schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS["source_manifest"],
    )
    native = ArtifactRef(
        relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS["native_reference"],
        sha256="4" * 64,
        size_bytes=41,
        schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS["native_reference"],
    )
    terminal = ArtifactRef(
        relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS["supervisor_terminal"],
        sha256="5" * 64,
        size_bytes=51,
        schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[
            "supervisor_terminal"
        ],
    )
    slots["source_manifest"] = receipt_module.EvidenceSlotV5.present(source)
    slots["native_reference"] = receipt_module.EvidenceSlotV5.present(native)
    outcome = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.SETUP,
        receipt_module.FailureReasonCodeV5.FROZEN_NUMERICAL_SUBSET_INVALID,
        "6" * 64,
    )
    slots["frozen_numerical_subset"] = receipt_module.EvidenceSlotV5.absent(
        outcome.reason
    )
    slots["supervisor_terminal"] = receipt_module.EvidenceSlotV5.present(terminal)
    with pytest.raises(receipt_module.Diag5ReceiptConstructionError) as captured:
        receipt_module._validate_diag5_stage_vector(slots, failure=outcome)
    assert (
        captured.value.reason is receipt_module.FailureReasonCodeV5.GROUP_PREFIX_INVALID
    )


def test_diag5_postmortem_rejects_semantic_mutation() -> None:
    repository = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (
            repository
            / "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json"
        ).read_bytes()
    )
    assert (
        receipt_module.validate_diag5_predecessor_postmortem_payload(payload) == payload
    )
    payload["reconstruction"]["final_root_absent"] = False
    with pytest.raises(ValueError, match="semantics differ"):
        receipt_module.validate_diag5_predecessor_postmortem_payload(payload)


def test_diag5_all_slot_schemas_reject_cross_generation_mutation() -> None:
    for index, (name, schema) in enumerate(
        receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS.items(), start=1
    ):
        reference = ArtifactRef(
            relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS[name],
            sha256=f"{index:064x}",
            size_bytes=index,
            schema_version=schema,
        )
        payload = receipt_module.diag5_evidence_slot_payload(
            receipt_module.EvidenceSlotV5.present(reference)
        )
        assert isinstance(
            receipt_module.parse_diag5_evidence_slot(payload, name=name),
            receipt_module.EvidenceSlotV5,
        )
        payload["artifact"]["schema_version"] = "predecessor-schema-v4"
        with pytest.raises(ValueError, match="schema differs"):
            receipt_module.parse_diag5_evidence_slot(payload, name=name)


def test_diag5_native_bindings_reject_each_identity_mutation() -> None:
    cpu = receipt_module.NativeBindingV5(
        "cpu", "/opt/cpu/simsoptpp.so", "7" * 64, 101, 2, 11, 13
    )
    gpu = receipt_module.NativeBindingV5(
        "gpu", "/opt/gpu/simsoptpp.so", "7" * 64, 101, 1, 17, 19
    )
    payload = receipt_module.diag5_native_bindings_payload((("cpu", cpu), ("gpu", gpu)))
    parsed = receipt_module.parse_diag5_native_bindings(payload)
    assert parsed == (("cpu", cpu), ("gpu", gpu))
    for field, value in (
        ("native_extension_sha256", "8" * 64),
        ("native_extension_size_bytes", 102),
    ):
        mutated = json.loads(json.dumps(payload))
        mutated["gpu"][field] = value
        with pytest.raises(ValueError, match="binary identity differs"):
            receipt_module.parse_diag5_native_bindings(mutated)
    mutated = json.loads(json.dumps(payload))
    mutated["gpu"]["gpu_native_extension_path"] = "relative.so"
    with pytest.raises(ValueError, match="absolute and resolved"):
        receipt_module.parse_diag5_native_bindings(mutated)


def test_diag5_stage_reason_cross_product_is_closed() -> None:
    for stage, allowed in receipt_module.DIAG5_STAGE_REASON_ORDER.items():
        disallowed = next(
            reason
            for reason in receipt_module.FailureReasonCodeV5
            if reason not in allowed
        )
        outcome = receipt_module.StructuredFailureV5(stage, disallowed, "9" * 64)
        with pytest.raises(ValueError, match="pairing differs"):
            receipt_module.diag5_terminal_outcome_payload(outcome)


def test_diag5_stage_reason_order_is_the_exact_ssot_table() -> None:
    expected = {
        "AUTHORITY": (
            "AUTHORITY_INVALID",
            "OUTPUT_ROOT_NOT_ABSENT",
            "LOCK_CLAIM_FAILED",
            "IDENTITY_REVALIDATION_FAILED",
            "AUTHORITY_ALREADY_CONSUMED",
        ),
        "SETUP": (
            "SOURCE_PUBLICATION_FAILED",
            "FROZEN_NUMERICAL_SUBSET_INVALID",
            "NATIVE_REFERENCE_INVALID",
            "POLICY_AUTHORITY_INVALID",
            "SETUP_DEEP_LOAD_FAILED",
        ),
        "BEFORE_PREFLIGHT": (
            "SUPERVISOR_GPU_OBSERVATION_INVALID",
            "SUPERVISOR_GPU_NONZERO",
            "AUTHORITY_CONSUMPTION_FAILED",
            "AUTHORITY_CONSUMPTION_UNCERTAIN",
            "SOURCE_REVALIDATION_FAILED",
        ),
        "PREFLIGHT": (
            "PREFLIGHT_LAUNCH_FAILED",
            "PREFLIGHT_TIMEOUT",
            "PREFLIGHT_MONITOR_FAILED",
            "PREFLIGHT_EXIT_NONZERO",
            "PREFLIGHT_PROTOCOL_INVALID",
            "PREFLIGHT_PRODUCER_INVALID",
            "PREFLIGHT_GATE_FAILED",
        ),
        "BEFORE_COLD": (
            "SUPERVISOR_GPU_OBSERVATION_INVALID",
            "SUPERVISOR_GPU_NONZERO",
            "SOURCE_REVALIDATION_FAILED",
            "IDENTITY_REVALIDATION_FAILED",
            "CONSUMPTION_MARKER_INVALID",
        ),
        "COLD": (
            "COLD_LAUNCH_FAILED",
            "COLD_TIMEOUT",
            "COLD_MONITOR_FAILED",
            "COLD_EXIT_NONZERO",
            "COLD_PROTOCOL_INVALID",
            "COLD_PRODUCER_INVALID",
        ),
        "NUMERICAL_COMMIT": (
            "PENDING_RESULT_ABSENT",
            "TIMING_INVALID",
            "SAFEGUARD_TELEMETRY_INVALID",
            "NUMERICAL_IDENTITY_MISMATCH",
            "QUARANTINE_FAILED",
            "PENDING_RESULT_INVALID",
            "COMMIT_COLLISION",
            "COMMIT_RENAME_FAILED",
            "COMMIT_FSYNC_FAILED",
            "COMMITTED_DEEP_LOAD_FAILED",
        ),
        "RECEIPT": (
            "EVIDENCE_VECTOR_INVALID",
            "GROUP_PREFIX_INVALID",
            "SCIENTIFIC_RECONSTRUCTION_FAILED",
            "RECEIPT_SCHEMA_INVALID",
        ),
        "PUBLICATION": (
            "MANIFEST_INVALID",
            "MODE_OR_LINK_INVALID",
            "STAGING_DEEP_LOAD_FAILED",
            "FINAL_COLLISION",
            "FINAL_RENAME_FAILED",
        ),
        "SCIENTIFIC": ("INCOMPLETE", "NO_HIT", "QUALITY_HIT"),
    }
    assert tuple(stage.value for stage in receipt_module.DIAG5_STAGE_REASON_ORDER) == (
        *expected,
    )
    assert {
        stage.value: tuple(reason.value for reason in reasons)
        for stage, reasons in receipt_module.DIAG5_STAGE_REASON_ORDER.items()
    } == expected


def test_diag5_stage_reason_prefix_contract_is_exhaustive_and_exact() -> None:
    names = tuple(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS)
    nonterminal_count = len(names) - 1
    declared_pairs = frozenset(receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES)
    expected_pairs = frozenset(
        (stage, reason)
        for stage, reasons in receipt_module.DIAG5_STAGE_REASON_ORDER.items()
        for reason in reasons
    )
    assert declared_pairs == expected_pairs
    assert receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES[
        (
            receipt_module.FailureStageV5.BEFORE_PREFLIGHT,
            receipt_module.FailureReasonCodeV5.AUTHORITY_CONSUMPTION_FAILED,
        )
    ] == (5,)
    assert receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES[
        (
            receipt_module.FailureStageV5.PREFLIGHT,
            receipt_module.FailureReasonCodeV5.PREFLIGHT_MONITOR_FAILED,
        )
    ] == (8,)
    assert receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES[
        (
            receipt_module.FailureStageV5.BEFORE_COLD,
            receipt_module.FailureReasonCodeV5.CONSUMPTION_MARKER_INVALID,
        )
    ] == (13,)
    assert receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES[
        (
            receipt_module.FailureStageV5.COLD,
            receipt_module.FailureReasonCodeV5.COLD_MONITOR_FAILED,
        )
    ] == (16,)
    assert receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES[
        (
            receipt_module.FailureStageV5.RECEIPT,
            receipt_module.FailureReasonCodeV5.EVIDENCE_VECTOR_INVALID,
        )
    ] == (24, 25)

    def reference(name: str, index: int) -> ArtifactRef:
        return ArtifactRef(
            relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS[name],
            sha256=f"{index + 1:064x}",
            size_bytes=index + 1,
            schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[name],
        )

    def slots_for(
        prefix: int, reason: receipt_module.FailureReasonCodeV5
    ) -> dict[str, receipt_module.EvidenceSlotV5]:
        return {
            name: (
                receipt_module.EvidenceSlotV5.present(reference(name, index))
                if index < prefix or index == nonterminal_count
                else receipt_module.EvidenceSlotV5.absent(
                    reason if index == prefix else None
                )
            )
            for index, name in enumerate(names)
        }

    for stage, reasons in receipt_module.DIAG5_STAGE_REASON_ORDER.items():
        for reason in reasons:
            outcome = receipt_module.StructuredFailureV5(stage, reason, "d" * 64)
            allowed = receipt_module.DIAG5_STAGE_REASON_PRESENT_PREFIXES[
                (stage, reason)
            ]
            assert allowed == tuple(sorted(frozenset(allowed)))
            for prefix in range(nonterminal_count + 1):
                slots = slots_for(prefix, reason)
                if prefix in allowed:
                    receipt_module._validate_diag5_stage_vector(slots, failure=outcome)
                else:
                    with pytest.raises(
                        receipt_module.Diag5ReceiptConstructionError
                    ) as captured:
                        receipt_module._validate_diag5_stage_vector(
                            slots, failure=outcome
                        )
                    assert captured.value.reason is (
                        receipt_module.FailureReasonCodeV5.GROUP_PREFIX_INVALID
                    )


def test_diag5_reserved_source_revalidation_details_select_exact_vectors() -> None:
    names = tuple(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS)
    nonterminal_count = len(names) - 1

    def slots_for(
        prefix: int, reason: receipt_module.FailureReasonCodeV5
    ) -> dict[str, receipt_module.EvidenceSlotV5]:
        return {
            name: (
                receipt_module.EvidenceSlotV5.present(
                    ArtifactRef(
                        receipt_module.DIAG5_EVIDENCE_SLOT_PATHS[name],
                        f"{index + 1:064x}",
                        index + 1,
                        receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[name],
                    )
                )
                if index < prefix or index == nonterminal_count
                else receipt_module.EvidenceSlotV5.absent(
                    reason if index == prefix else None
                )
            )
            for index, name in enumerate(names)
        }

    cases = (
        (
            receipt_module.FailureStageV5.PREFLIGHT,
            receipt_module.FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
            receipt_module.DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256,
            (8, 12),
            ("preflight",),
        ),
        (
            receipt_module.FailureStageV5.COLD,
            receipt_module.FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
            receipt_module.DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256,
            (16,),
            ("preflight", "cold"),
        ),
    )
    for stage, reason, detail, allowed, children in cases:
        failure = receipt_module.StructuredFailureV5(stage, reason, detail)
        assert receipt_module._diag5_allowed_present_prefixes(failure) == allowed
        assert receipt_module._diag5_expected_launched_children(failure) == children
        for prefix in range(nonterminal_count + 1):
            if prefix in allowed:
                receipt_module._validate_diag5_stage_vector(
                    slots_for(prefix, reason), failure=failure
                )
            else:
                with pytest.raises(receipt_module.Diag5ReceiptConstructionError):
                    receipt_module._validate_diag5_stage_vector(
                        slots_for(prefix, reason), failure=failure
                    )
    wrong_pair = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.COLD,
        receipt_module.FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
        receipt_module.DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256,
    )
    assert receipt_module._diag5_allowed_present_prefixes(wrong_pair) is None


@pytest.mark.parametrize(
    ("stage", "reason", "detail", "mode"),
    (
        (
            receipt_module.FailureStageV5.PREFLIGHT,
            receipt_module.FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID,
            receipt_module.DIAG5_POST_PREFLIGHT_SOURCE_REVALIDATION_DETAIL_SHA256,
            "preflight",
        ),
        (
            receipt_module.FailureStageV5.COLD,
            receipt_module.FailureReasonCodeV5.COLD_PROTOCOL_INVALID,
            receipt_module.DIAG5_POST_COLD_SOURCE_REVALIDATION_DETAIL_SHA256,
            "cold",
        ),
    ),
)
def test_diag5_reserved_source_revalidation_requires_successful_child_chain(
    stage: receipt_module.FailureStageV5,
    reason: receipt_module.FailureReasonCodeV5,
    detail: str,
    mode: str,
) -> None:
    failure = receipt_module.StructuredFailureV5(stage, reason, detail)
    terminal = {
        "terminal_status": "COMPLETE",
        "monitor_failure_kind": "NONE",
        "failure_reasons": [],
    }
    process = {"process_diagnostics": {"returncode": 0}}
    receipt_module._validate_diag5_child_outcome(
        terminal, process, mode=mode, failure=failure
    )
    receipt_module._validate_diag5_child_producer_origin(
        {"execution_status": "SUCCESS"}, failure=failure, mode=mode
    )
    with pytest.raises(ValueError, match="supervisor producer"):
        receipt_module._validate_diag5_child_producer_origin(
            {"document_origin": "PARENT_SUPERVISOR"},
            failure=failure,
            mode=mode,
        )
    for terminal_mutation, process_mutation in (
        ({"terminal_status": "PROTOCOL_FAILURE"}, {}),
        ({"monitor_failure_kind": "FINALIZATION"}, {}),
        ({"failure_reasons": [reason.value]}, {}),
        ({}, {"process_diagnostics": {"returncode": 1}}),
    ):
        with pytest.raises(ValueError, match="reserved source revalidation"):
            receipt_module._validate_diag5_child_outcome(
                {**terminal, **terminal_mutation},
                {**process, **process_mutation},
                mode=mode,
                failure=failure,
            )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("pending", receipt_module.FailureReasonCodeV5.PENDING_RESULT_INVALID),
        ("timing", receipt_module.FailureReasonCodeV5.TIMING_INVALID),
        (
            "safeguard",
            receipt_module.FailureReasonCodeV5.SAFEGUARD_TELEMETRY_INVALID,
        ),
        (
            "identity",
            receipt_module.FailureReasonCodeV5.NUMERICAL_IDENTITY_MISMATCH,
        ),
    ),
)
def test_diag5_numerical_document_failures_are_independently_typed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: receipt_module.FailureReasonCodeV5,
) -> None:
    identity_fields = (
        "problem_sha256",
        "optimizer_options_sha256",
        "base_neq_gntr1_policy_sha256",
        "scaling_sha256",
        "bootstrap_state_sha256",
        "initial_physical_state_sha256",
        "identity_sha256",
    )
    identity_values = {
        "numerical_route": receipt_module.DIAG5_NUMERICAL_ROUTE,
        "numerical_result_schema_version": (
            receipt_module.DIAG5_NUMERICAL_RESULT_SCHEMA_VERSION
        ),
        **{name: "e" * 64 for name in identity_fields},
    }
    producer = {
        **identity_values,
        "source_manifest_sha256": "f" * 64,
        "history_evidence": {
            "relative_path": "cold/numerical-result/history.json",
            "sha256": "1" * 64,
            "size_bytes": 1,
            "schema_version": receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[
                "cold_history"
            ],
        },
    }
    identity = SimpleNamespace(**identity_values)
    timing = SimpleNamespace(
        **identity_values,
        source_manifest_sha256="f" * 64,
        profiler_start_calls=0,
        profiler_stop_calls=0,
        trace_normalization_calls=0,
    )
    telemetry = SimpleNamespace(**identity_values)
    monkeypatch.setattr(
        receipt_module,
        "validate_diag5_history_evidence_payload",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_diag5_producer_payload",
        lambda *_args, **_kwargs: producer,
    )
    monkeypatch.setattr(
        receipt_module,
        "_validate_gntr3_terminal_numerical_structure",
        lambda _value: (None, identity, None, None, None, None),
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_diag5_solve_timing_evidence_payload",
        lambda _value: timing,
    )
    monkeypatch.setattr(
        receipt_module,
        "validate_diag5_safeguard_telemetry_payload",
        lambda *_args, **_kwargs: telemetry,
    )
    if mutation == "pending":
        monkeypatch.setattr(
            receipt_module,
            "validate_diag5_history_evidence_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("pending invalid")
            ),
        )
    elif mutation == "timing":
        monkeypatch.setattr(
            receipt_module,
            "validate_diag5_solve_timing_evidence_payload",
            lambda _value: (_ for _ in ()).throw(ValueError("timing invalid")),
        )
    elif mutation == "safeguard":
        monkeypatch.setattr(
            receipt_module,
            "validate_diag5_safeguard_telemetry_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("safeguard invalid")
            ),
        )
    else:
        telemetry.identity_sha256 = "0" * 64
    with pytest.raises(receipt_module.Diag5NumericalDocumentError) as captured:
        receipt_module.validate_diag5_numerical_documents(
            history={},
            solve_timing={},
            safeguard_telemetry={},
            terminal_numerical={},
            producer={},
        )
    assert captured.value.reason is reason
    assert len(captured.value.detail_sha256) == 64


def test_diag5_receipt_construction_failures_are_typed_without_message_parsing() -> (
    None
):
    names = tuple(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS)
    outcome = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.BEFORE_PREFLIGHT,
        receipt_module.FailureReasonCodeV5.AUTHORITY_CONSUMPTION_FAILED,
        "2" * 64,
    )
    slots = {
        name: receipt_module.EvidenceSlotV5.absent(
            outcome.reason if index == 0 else None
        )
        for index, name in enumerate(names)
    }
    slots["supervisor_terminal"] = receipt_module.EvidenceSlotV5.present(
        ArtifactRef(
            relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS[
                "supervisor_terminal"
            ],
            sha256="3" * 64,
            size_bytes=1,
            schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[
                "supervisor_terminal"
            ],
        )
    )
    with pytest.raises(receipt_module.Diag5ReceiptConstructionError) as captured:
        receipt_module._validate_diag5_stage_vector(slots, failure=outcome)
    assert (
        captured.value.reason is receipt_module.FailureReasonCodeV5.GROUP_PREFIX_INVALID
    )
    assert (
        receipt_module.classify_diag5_receipt_construction_error(captured.value)
        is receipt_module.FailureReasonCodeV5.GROUP_PREFIX_INVALID
    )
    assert (
        receipt_module.classify_diag5_receipt_construction_error(
            ValueError("unstructured parser failure")
        )
        is receipt_module.FailureReasonCodeV5.RECEIPT_SCHEMA_INVALID
    )


def test_diag5_manifest_rejects_broken_pending_namespace(tmp_path: Path) -> None:
    pending = tmp_path / receipt_module.DIAG5_PENDING_NUMERICAL_DIRECTORY
    pending.parent.mkdir(parents=True)
    pending.symlink_to(tmp_path / "absent-target")
    with pytest.raises(ValueError, match="pending numerical result"):
        receipt_module._diag5_artifact_roles(tmp_path)


def test_diag5_deep_load_requires_held_authority_bindings() -> None:
    required = {
        "expected_native_bindings",
        "expected_authority_sha256",
        "expected_predecessor_postmortem",
        "expected_source_snapshot_identity",
        "expected_logical_snapshot_root",
        "expected_frozen_numerical_entries",
        "expected_gpu_uuid",
        "physical_memory_bytes",
    }
    for function in (
        receipt_module.build_diag5_diagnostic_receipt,
        receipt_module.load_diag5_diagnostic_receipt_bytes,
        receipt_module.validate_diag5_writable_staging,
        receipt_module.load_and_validate_diag5_staging,
        receipt_module.load_and_validate_diag5_artifact,
        receipt_module.load_and_validate_diag5_rollback,
    ):
        signature = inspect.signature(function)
        assert required <= signature.parameters.keys()
        assert all(
            signature.parameters[name].default is inspect.Parameter.empty
            for name in required
        )
    derive_required = {
        "gpu_native_binding",
        "authority_sha256",
        "expected_source_snapshot_identity",
        "expected_logical_snapshot_root",
        "expected_frozen_numerical_entries",
        "expected_gpu_uuid",
        "physical_memory_bytes",
    }
    derive_signature = inspect.signature(receipt_module.derive_diag5_evidence_slots)
    assert derive_required <= derive_signature.parameters.keys()
    assert all(
        derive_signature.parameters[name].default is inspect.Parameter.empty
        for name in derive_required
    )
    rollback_signature = inspect.signature(
        receipt_module.load_and_validate_diag5_rollback
    )
    assert (
        rollback_signature.parameters["expected_rollback_root"].default
        is inspect.Parameter.empty
    )


def test_diag4_and_diag5_receipt_loaders_reject_each_other_before_rebuild(
    tmp_path: Path,
) -> None:
    v5_payload = {
        "schema_version": receipt_module.DIAG5_SCHEMA_VERSION,
        "route": receipt_module.DIAG5_ROUTE,
        "numerical_route": receipt_module.DIAG5_NUMERICAL_ROUTE,
        "plan_sha256": receipt_module.DIAG5_PLAN_SHA256,
    }
    with pytest.raises(ValueError, match="DIAG4 diagnostic"):
        receipt_module.diag4_diagnostic_receipt_from_payload(
            v5_payload, artifact_root=tmp_path
        )
    v4_payload = {
        "schema_version": receipt_module.DIAG4_SCHEMA_VERSION,
        "route": receipt_module.DIAG4_ROUTE,
        "numerical_route": receipt_module.DIAG4_NUMERICAL_ROUTE,
        "plan_sha256": receipt_module.DIAG4_PLAN_SHA256,
    }
    with pytest.raises(ValueError, match="DIAG5 diagnostic"):
        receipt_module.diag5_diagnostic_receipt_from_payload(
            v4_payload,
            artifact_root=tmp_path,
            expected_native_bindings={},
            expected_authority_sha256="a" * 64,
            expected_predecessor_postmortem=ArtifactRef(
                relative_path=receipt_module.DIAG5_PREDECESSOR_POSTMORTEM_PATH,
                sha256="b" * 64,
                size_bytes=1,
                schema_version=receipt_module.DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
            ),
            expected_source_snapshot_identity=SimpleNamespace(),
            expected_logical_snapshot_root=tmp_path / "source-snapshot",
            expected_frozen_numerical_entries={},
            expected_gpu_uuid=GPU_UUID,
            physical_memory_bytes=1,
        )


@pytest.mark.parametrize(
    ("stage", "reason", "prefix"),
    (
        (
            receipt_module.FailureStageV5.PREFLIGHT,
            receipt_module.FailureReasonCodeV5.PREFLIGHT_LAUNCH_FAILED,
            5,
        ),
        (
            receipt_module.FailureStageV5.COLD,
            receipt_module.FailureReasonCodeV5.COLD_LAUNCH_FAILED,
            13,
        ),
    ),
)
def test_diag5_launch_failure_vectors_have_exact_maximum_prefix(
    stage: receipt_module.FailureStageV5,
    reason: receipt_module.FailureReasonCodeV5,
    prefix: int,
) -> None:
    names = tuple(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS)
    outcome = receipt_module.StructuredFailureV5(stage, reason, "c" * 64)

    def reference(name: str, index: int) -> ArtifactRef:
        return ArtifactRef(
            relative_path=receipt_module.DIAG5_EVIDENCE_SLOT_PATHS[name],
            sha256=f"{index + 1:064x}",
            size_bytes=index + 1,
            schema_version=receipt_module.DIAG5_EVIDENCE_SLOT_SCHEMAS[name],
        )

    slots = {
        name: (
            receipt_module.EvidenceSlotV5.present(reference(name, index))
            if index < prefix or name == "supervisor_terminal"
            else receipt_module.EvidenceSlotV5.absent(
                reason if index == prefix else None
            )
        )
        for index, name in enumerate(names)
    }
    receipt_module._validate_diag5_stage_vector(slots, failure=outcome)
    slots[names[prefix]] = receipt_module.EvidenceSlotV5.present(
        reference(names[prefix], prefix)
    )
    slots[names[prefix + 1]] = receipt_module.EvidenceSlotV5.absent(reason)
    with pytest.raises(receipt_module.Diag5ReceiptConstructionError) as captured:
        receipt_module._validate_diag5_stage_vector(slots, failure=outcome)
    assert (
        captured.value.reason is receipt_module.FailureReasonCodeV5.GROUP_PREFIX_INVALID
    )


@pytest.mark.parametrize(
    ("cluster", "mutation"),
    (
        ("session", ("session_reference", "wrong")),
        ("command", ("command_text", "wrong")),
        ("exception", ("exception_class", "wrong")),
        ("count", ("copied_tree_entry_count", 603)),
        ("hash", ("execution_manifest_sha256", "0" * 64)),
    ),
)
def test_diag5_postmortem_exact_scalar_clusters_reject_mutation(
    cluster: str, mutation: tuple[str, object]
) -> None:
    del cluster
    repository = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (
            repository
            / "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json"
        ).read_bytes()
    )
    field, value = mutation
    target = payload if field == "session_reference" else payload["reconstruction"]
    target[field] = value
    with pytest.raises(ValueError, match="semantics differ"):
        receipt_module.validate_diag5_predecessor_postmortem_payload(payload)


def test_diag5_postmortem_exact_native_and_review_clusters_reject_mutation() -> None:
    repository = Path(__file__).resolve().parents[2]
    path = (
        repository
        / "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json"
    )
    native = json.loads(path.read_bytes())
    native["reconstruction"]["native_binding"]["inode"] += 1
    with pytest.raises(ValueError, match="native topology differs"):
        receipt_module.validate_diag5_predecessor_postmortem_payload(native)
    review = json.loads(path.read_bytes())
    review["reconstruction"]["prior_reviews_retracted"][0]["session"] = "wrong"
    with pytest.raises(ValueError, match="review rows differ"):
        receipt_module.validate_diag5_predecessor_postmortem_payload(review)
    order = json.loads(path.read_bytes())
    order["reconstruction"]["prior_reviews_retracted"].reverse()
    with pytest.raises(ValueError, match="review rows differ"):
        receipt_module.validate_diag5_predecessor_postmortem_payload(order)
    aggregate = json.loads(path.read_bytes())
    aggregate["reconstruction"]["retracted_reviews_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="retracted reviews differ"):
        receipt_module.validate_diag5_predecessor_postmortem_payload(aggregate)


def test_diag5_supervisor_failure_producer_is_exact_and_generation_owned() -> None:
    process = ArtifactRef(
        "preflight/process.json",
        "1" * 64,
        101,
        receipt_module.DIAG5_PROCESS_SCHEMA_VERSION,
    )
    terminal = ArtifactRef(
        "preflight/terminal.json",
        "2" * 64,
        202,
        receipt_module.DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
    )
    payload = receipt_module.build_diag5_supervisor_failure_producer_payload(
        mode="preflight",
        selected_failure_reason=(
            receipt_module.FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID
        ),
        child_pid=17,
        child_start_time_ticks=19,
        process_started_monotonic_ns=23,
        process_stopped_monotonic_ns=29,
        process_evidence=process,
        child_terminal_evidence=terminal,
    )
    assert len(payload) == 16
    assert payload["document_origin"] == "PARENT_SUPERVISOR"
    assert payload["plan_sha256"] == receipt_module.DIAG5_PLAN_SHA256
    assert (
        receipt_module.validate_diag5_supervisor_failure_producer_payload(
            payload, mode="preflight"
        )
        == payload
    )
    for field, value in (
        ("schema_version", receipt_module.DIAG4_PREFLIGHT_SCHEMA_VERSION),
        ("selected_failure_reason", "COLD_PRODUCER_INVALID"),
        ("child_pid", 0),
        ("child_start_time_ticks", -1),
        ("process_started_monotonic_ns", 30),
        (
            "child_terminal_evidence",
            {
                **payload["child_terminal_evidence"],
                "relative_path": "cold/terminal.json",
            },
        ),
    ):
        mutated = json.loads(json.dumps(payload))
        mutated[field] = value
        with pytest.raises((TypeError, ValueError)):
            receipt_module.validate_diag5_supervisor_failure_producer_payload(
                mutated, mode="preflight"
            )


def test_diag5_profiler_call_audit_is_not_a_v4_public_type() -> None:
    assert (
        receipt_module.Diag5ProfilerCallAudit
        is not receipt_module.Diag4ProfilerCallAudit
    )
    assert isinstance(
        receipt_module.DIAG5_PROFILER_CALL_AUDIT,
        receipt_module.Diag5ProfilerCallAudit,
    )
    assert not isinstance(
        receipt_module.DIAG5_PROFILER_CALL_AUDIT,
        receipt_module.Diag4ProfilerCallAudit,
    )


@pytest.mark.parametrize(
    ("reason", "status", "monitor", "returncode"),
    (
        ("PREFLIGHT_TIMEOUT", "TIMEOUT", "NONE", -15),
        ("PREFLIGHT_MONITOR_FAILED", "MONITOR_FAILURE", "BINDING", 0),
        ("PREFLIGHT_EXIT_NONZERO", "CRASH", "NONE", 7),
        ("PREFLIGHT_PROTOCOL_INVALID", "PROTOCOL_FAILURE", "NONE", 0),
        ("PREFLIGHT_PRODUCER_INVALID", "PROTOCOL_FAILURE", "NONE", 0),
    ),
)
def test_diag5_child_outcome_reason_monitor_and_process_join_is_exact(
    reason: str, status: str, monitor: str, returncode: int
) -> None:
    failure = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.PREFLIGHT,
        receipt_module.FailureReasonCodeV5(reason),
        "4" * 64,
    )
    terminal = {
        "terminal_status": status,
        "monitor_failure_kind": monitor,
        "failure_reasons": [reason],
    }
    process = {"process_diagnostics": {"returncode": returncode}}
    receipt_module._validate_diag5_child_outcome(
        terminal, process, mode="preflight", failure=failure
    )
    for mutation in (
        {"monitor_failure_kind": "NONE" if monitor != "NONE" else "FINALIZATION"},
        {"failure_reasons": []},
        {
            "terminal_status": (
                "TIMEOUT" if status == "MONITOR_FAILURE" else "MONITOR_FAILURE"
            )
        },
    ):
        with pytest.raises(ValueError, match="contradicts"):
            receipt_module._validate_diag5_child_outcome(
                {**terminal, **mutation},
                process,
                mode="preflight",
                failure=failure,
            )
    if reason != "PREFLIGHT_MONITOR_FAILED":
        invalid_return = 0 if returncode != 0 else 9
        with pytest.raises(ValueError, match="contradicts"):
            receipt_module._validate_diag5_child_outcome(
                terminal,
                {"process_diagnostics": {"returncode": invalid_return}},
                mode="preflight",
                failure=failure,
            )


def test_diag5_success_child_terminal_requires_exact_complete_closure() -> None:
    failure = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.BEFORE_COLD,
        receipt_module.FailureReasonCodeV5.SOURCE_REVALIDATION_FAILED,
        "5" * 64,
    )
    terminal = {
        "terminal_status": "COMPLETE",
        "monitor_failure_kind": "NONE",
        "failure_reasons": [],
    }
    process = {"process_diagnostics": {"returncode": 0}}
    receipt_module._validate_diag5_child_outcome(
        terminal, process, mode="preflight", failure=failure
    )
    for mutation in (
        {"terminal_status": "CRASH"},
        {"monitor_failure_kind": "FINALIZATION"},
        {"failure_reasons": ["unexpected"]},
    ):
        with pytest.raises(ValueError, match="successful termination"):
            receipt_module._validate_diag5_child_outcome(
                {**terminal, **mutation},
                process,
                mode="preflight",
                failure=failure,
            )


@pytest.mark.parametrize(
    ("reason", "monitor"),
    (
        ("PREFLIGHT_PROTOCOL_INVALID", "NONE"),
        ("PREFLIGHT_MONITOR_FAILED", "FINALIZATION"),
    ),
)
def test_diag5_parent_detected_complete_child_allows_empty_failure_reasons(
    reason: str, monitor: str
) -> None:
    failure = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.PREFLIGHT,
        receipt_module.FailureReasonCodeV5(reason),
        "9" * 64,
    )
    terminal = {
        "terminal_status": "COMPLETE",
        "monitor_failure_kind": monitor,
        "failure_reasons": [],
    }
    process = {"process_diagnostics": {"returncode": 0}}
    receipt_module._validate_diag5_child_outcome(
        terminal, process, mode="preflight", failure=failure
    )
    with pytest.raises(ValueError, match="contradicts"):
        receipt_module._validate_diag5_child_outcome(
            terminal,
            {"process_diagnostics": {"returncode": 1}},
            mode="preflight",
            failure=failure,
        )


def test_diag5_manifest_only_memory_pair_is_deep_validated(tmp_path: Path) -> None:
    mode_root = tmp_path / "preflight"
    mode_root.mkdir()
    argv = ("python", "child.py", "--mode", "preflight")
    samples = {
        "schema_version": receipt_module.DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
        "samples": [{"sampled_at_unix_ns": 11, "used_memory_mib": 1}],
    }
    memory = {
        "schema_version": receipt_module.DIAG5_MEMORY_SCHEMA_VERSION,
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "parent_pid": 3,
        "child_pid": 7,
        "child_start_time_ticks": 13,
        "child_argv_sha256": hashlib.sha256(
            canonical_json_bytes(list(argv[2:]))
        ).hexdigest(),
        "device_uuid": GPU_UUID,
        "sample_count": 1,
        "peak_memory_bytes": 1024 * 1024,
        "peak_memory_fraction": 0.25,
    }
    memory_path = mode_root / "gpu-memory.json"
    samples_path = mode_root / "gpu-memory-samples.json"
    memory_path.write_bytes(canonical_json_bytes(memory))
    samples_path.write_bytes(canonical_json_bytes(samples))
    slots = {
        f"preflight_{suffix}": receipt_module.EvidenceSlotV5.absent()
        for suffix in ("memory", "memory_samples", "runtime", "policy")
    }
    loaded = {
        "preflight_process": {
            "child_pid": 7,
            "child_start_time_ticks": 13,
            "argv": list(argv),
        }
    }
    expected_snapshot_identity = build_snapshot_identity(
        tuple(
            SnapshotEntry(role, f"{index}.bin", 1, f"{index + 1:064x}")
            for index, role in enumerate(
                sorted(receipt_module.DIAG5_GPU_SNAPSHOT_ROLES)
            )
        ),
        WorktreeIdentity("1" * 40, "2" * 64, "3" * 64, "/logical/repo"),
    )
    expected_logical_snapshot_root = Path("/logical/source-snapshot")
    receipt_module._validate_diag5_manifest_only_auxiliaries(
        tmp_path,
        slots,
        loaded,
        mode="preflight",
        authority=None,
        snapshot=None,
        expected_snapshot_identity=expected_snapshot_identity,
        expected_logical_snapshot_root=expected_logical_snapshot_root,
        gpu_native_binding=receipt_module.NativeBindingV5(
            "gpu", "/opt/native.so", "6" * 64, 1, 1, 1, 1
        ),
        expected_gpu_uuid=GPU_UUID,
        physical_memory_bytes=4 * 1024 * 1024,
    )
    for field, value in (
        ("sampled_at_unix_ns", 0),
        ("used_memory_mib", -1),
    ):
        mutated = json.loads(json.dumps(samples))
        mutated["samples"][0][field] = value
        samples_path.write_bytes(canonical_json_bytes(mutated))
        with pytest.raises(ValueError):
            receipt_module._validate_diag5_manifest_only_auxiliaries(
                tmp_path,
                slots,
                loaded,
                mode="preflight",
                authority=None,
                snapshot=None,
                expected_snapshot_identity=expected_snapshot_identity,
                expected_logical_snapshot_root=expected_logical_snapshot_root,
                gpu_native_binding=receipt_module.NativeBindingV5(
                    "gpu", "/opt/native.so", "6" * 64, 1, 1, 1, 1
                ),
                expected_gpu_uuid=GPU_UUID,
                physical_memory_bytes=4 * 1024 * 1024,
            )
    samples_path.write_bytes(canonical_json_bytes(samples))
    mutated_memory = {**memory, "device_uuid": "GPU-wrong"}
    memory_path.write_bytes(canonical_json_bytes(mutated_memory))
    with pytest.raises(ValueError, match="raw samples or execution"):
        receipt_module._validate_diag5_manifest_only_auxiliaries(
            tmp_path,
            slots,
            loaded,
            mode="preflight",
            authority=None,
            snapshot=None,
            expected_snapshot_identity=expected_snapshot_identity,
            expected_logical_snapshot_root=expected_logical_snapshot_root,
            gpu_native_binding=receipt_module.NativeBindingV5(
                "gpu", "/opt/native.so", "6" * 64, 1, 1, 1, 1
            ),
            expected_gpu_uuid=GPU_UUID,
            physical_memory_bytes=4 * 1024 * 1024,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("sampled_at_unix_ns", 0), ("used_memory_mib", -1)),
)
def test_diag5_typed_memory_sample_minima_are_generation_owned(
    field: str, value: int
) -> None:
    samples = {"samples": [{"sampled_at_unix_ns": 1, "used_memory_mib": 0}]}
    samples["samples"][0][field] = value
    with pytest.raises(ValueError):
        receipt_module._validate_diag5_memory_sample_rows(
            samples, context="DIAG5 typed"
        )


def test_diag5_invalid_producer_custody_is_reason_and_inode_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode_root = tmp_path / "preflight"
    mode_root.mkdir()
    invalid = mode_root / "invalid-producer.bin"
    invalid.write_bytes(b"invalid child bytes")
    outcome = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.PREFLIGHT,
        receipt_module.FailureReasonCodeV5.PREFLIGHT_PRODUCER_INVALID,
        "7" * 64,
    )
    producer_ref = ArtifactRef(
        "preflight/producer.json",
        "8" * 64,
        1,
        receipt_module.DIAG5_PREFLIGHT_SCHEMA_VERSION,
    )
    slots = {
        "preflight_producer": receipt_module.EvidenceSlotV5.present(producer_ref),
        **{
            f"preflight_{suffix}": receipt_module.EvidenceSlotV5.absent()
            for suffix in ("memory", "memory_samples", "runtime", "policy")
        },
    }
    producer = {
        "document_origin": "PARENT_SUPERVISOR",
        "selected_failure_reason": outcome.reason.value,
    }
    monkeypatch.setattr(receipt_module, "_load_ref_json", lambda *_args: producer)
    monkeypatch.setattr(
        receipt_module,
        "validate_diag5_producer_payload",
        lambda *_args, **_kwargs: producer,
    )
    roles: dict[str, str] = {}
    receipt_module._diag5_add_child_custody_roles(
        tmp_path, roles, slots, mode="preflight", outcome=outcome
    )
    assert roles["preflight/invalid-producer.bin"] == "preflight_invalid_producer"
    hardlink = mode_root / "invalid-copy.bin"
    os.link(invalid, hardlink)
    with pytest.raises(ValueError, match="regular inode"):
        receipt_module._diag5_add_child_custody_roles(
            tmp_path, {}, slots, mode="preflight", outcome=outcome
        )


def test_diag5_empty_quarantine_marker_is_exact_and_mutually_exclusive(
    tmp_path: Path,
) -> None:
    quarantine = tmp_path / receipt_module.DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
    quarantine.mkdir(parents=True)
    marker_path = tmp_path / receipt_module.DIAG5_EMPTY_QUARANTINE_PATH
    marker = {
        "schema_version": receipt_module.DIAG5_EMPTY_QUARANTINE_SCHEMA_VERSION,
        "route": receipt_module.DIAG5_ROUTE,
        "quarantine_relative_path": (
            receipt_module.DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
        ),
        "selected_failure_reason": "COLD_PRODUCER_INVALID",
    }
    marker_path.write_bytes(canonical_json_bytes(marker))
    outcome = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.COLD,
        receipt_module.FailureReasonCodeV5.COLD_PRODUCER_INVALID,
        "9" * 64,
    )
    roles: dict[str, str] = {}
    receipt_module._diag5_add_quarantine_roles(tmp_path, roles, outcome=outcome)
    assert roles[receipt_module.DIAG5_EMPTY_QUARANTINE_PATH] == (
        "empty_uncommitted_cold_numerical_result"
    )
    marker_path.write_bytes(
        canonical_json_bytes({**marker, "selected_failure_reason": "COLD_TIMEOUT"})
    )
    with pytest.raises(ValueError, match="marker differs"):
        receipt_module._diag5_add_quarantine_roles(tmp_path, {}, outcome=outcome)
    marker_path.write_bytes(canonical_json_bytes(marker))
    (quarantine / "retained.bin").write_bytes(b"retained")
    with pytest.raises(ValueError, match="nonempty quarantine"):
        receipt_module._diag5_add_quarantine_roles(tmp_path, {}, outcome=outcome)


def test_diag5_full_roles_admit_only_canonical_empty_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.COLD,
        receipt_module.FailureReasonCodeV5.COLD_PRODUCER_INVALID,
        "9" * 64,
    )
    terminal_path = tmp_path / "supervisor-terminal.json"
    terminal_path.write_bytes(
        canonical_json_bytes(
            receipt_module.build_diag5_supervisor_terminal_payload(
                outcome=outcome,
                launched_children=("preflight", "cold"),
                staging_root=tmp_path.with_name(f"{tmp_path.name}.partial-claim"),
                final_root=tmp_path,
            )
        )
    )
    terminal_ref = _artifact_ref(
        terminal_path,
        tmp_path,
        receipt_module.DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    slots = {
        name: receipt_module.EvidenceSlotV5.absent()
        for name in receipt_module.DIAG5_EVIDENCE_SLOT_PATHS
    }
    slots["supervisor_terminal"] = receipt_module.EvidenceSlotV5.present(terminal_ref)
    monkeypatch.setattr(receipt_module, "_diag5_receipt_slots", lambda _root: slots)
    (tmp_path / receipt_module.DIAG2_RECEIPT_FILENAME).write_bytes(b"receipt")
    predecessor = tmp_path / receipt_module.DIAG5_PREDECESSOR_POSTMORTEM_PATH
    predecessor.parent.mkdir()
    predecessor.write_bytes(b"postmortem")
    quarantine = tmp_path / receipt_module.DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
    quarantine.mkdir(parents=True)
    marker = {
        "schema_version": receipt_module.DIAG5_EMPTY_QUARANTINE_SCHEMA_VERSION,
        "route": receipt_module.DIAG5_ROUTE,
        "quarantine_relative_path": (
            receipt_module.DIAG5_UNCOMMITTED_NUMERICAL_DIRECTORY
        ),
        "selected_failure_reason": outcome.reason.value,
    }
    marker_path = tmp_path / receipt_module.DIAG5_EMPTY_QUARANTINE_PATH
    marker_path.write_bytes(canonical_json_bytes(marker))
    roles = receipt_module._diag5_artifact_roles(tmp_path)
    assert roles[receipt_module.DIAG5_EMPTY_QUARANTINE_PATH] == (
        "empty_uncommitted_cold_numerical_result"
    )
    manifest = receipt_module.diag5_artifact_manifest_payload(tmp_path)
    assert any(
        row["relative_path"] == receipt_module.DIAG5_EMPTY_QUARANTINE_PATH
        for row in manifest["entries"]
    )
    extra = tmp_path / "unknown-empty"
    extra.mkdir()
    with pytest.raises(ValueError, match="empty or alternate"):
        receipt_module._diag5_artifact_roles(tmp_path)
    extra.rmdir()
    marker_path.unlink()
    with pytest.raises(ValueError, match="omits its canonical marker"):
        receipt_module._diag5_artifact_roles(tmp_path)


def test_diag5_production_authority_artifact_reach_is_exact() -> None:
    names = tuple(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS)
    terminal = ArtifactRef(
        receipt_module.DIAG5_EVIDENCE_SLOT_PATHS["supervisor_terminal"],
        "a" * 64,
        1,
        receipt_module.DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    for reason in receipt_module.DIAG5_STAGE_REASON_ORDER[
        receipt_module.FailureStageV5.AUTHORITY
    ]:
        slots = {
            name: receipt_module.EvidenceSlotV5.absent(reason if index == 0 else None)
            for index, name in enumerate(names)
        }
        slots["supervisor_terminal"] = receipt_module.EvidenceSlotV5.present(terminal)
        failure = receipt_module.StructuredFailureV5(
            receipt_module.FailureStageV5.AUTHORITY, reason, "b" * 64
        )
        receipt_module._validate_diag5_stage_vector(slots, failure=failure)
        if reason is receipt_module.FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED:
            continue
        with pytest.raises(ValueError, match="pre-staging authority"):
            receipt_module._validate_diag5_slots(
                Path("."),
                slots,
                failure=failure,
                gpu_native_binding=receipt_module.NativeBindingV5(
                    "gpu", "/opt/native.so", "c" * 64, 1, 1, 1, 1
                ),
                authority_sha256="d" * 64,
                expected_source_snapshot_identity=SimpleNamespace(),
                expected_logical_snapshot_root=Path("/logical/source-snapshot"),
                expected_frozen_numerical_entries={},
                expected_gpu_uuid=GPU_UUID,
                physical_memory_bytes=1,
            )


def test_diag5_held_tree_rejects_path_replacement_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    file_path = root / "evidence.bin"
    file_path.write_bytes(b"held")
    with receipt_module._Diag5HeldTree(root, require_sealed=False) as held:
        assert held.file_bytes("evidence.bin") == b"held"
        token = receipt_module._DIAG5_HELD_TREE.set(held)
        reference = ArtifactRef(
            "evidence.bin", hashlib.sha256(b"held").hexdigest(), 4, "raw-v1"
        )
        replacement = root / "replacement.bin"
        replacement.write_bytes(b"replacement")
        replacement.replace(file_path)
        with pytest.raises(ValueError, match="changed while reading"):
            receipt_module._resolve_artifact(root, reference)
        with pytest.raises(ValueError, match="path changed"):
            held.revalidate_path_bindings()
        receipt_module._DIAG5_HELD_TREE.reset(token)
    file_path.unlink()
    file_path.symlink_to(root / "absent")
    with pytest.raises((OSError, ValueError)), receipt_module._Diag5HeldTree(
        root, require_sealed=False
    ):
        pass


def test_diag5_held_tree_semantic_mirror_survives_replace_restore_aba(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    nested = root / "source-snapshot"
    nested.mkdir(parents=True)
    source = nested / "source-manifest.json"
    source.write_bytes(b"held source manifest")
    with receipt_module._Diag5HeldTree(root, require_sealed=False) as held:
        token = receipt_module._DIAG5_HELD_TREE.set(held)
        held_inode = source.stat().st_ino
        saved = nested / "held-original.json"
        source.rename(saved)
        source.write_bytes(b"replacement source manifest")
        mirrored = receipt_module._diag5_held_path(
            root, "source-snapshot/source-manifest.json"
        )
        assert not mirrored.is_symlink()
        assert mirrored.read_bytes() == b"held source manifest"
        source.unlink()
        saved.rename(source)
        assert source.stat().st_ino == held_inode
        held.revalidate_path_bindings()
        receipt_module._DIAG5_HELD_TREE.reset(token)


def test_diag5_held_tree_mirror_runs_real_snapshot_and_runtime_v2_validators(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    inputs = tmp_path / "inputs"
    relative_roles: tuple[tuple[str, SnapshotRole], ...] = (
        ("benchmarks/entry.py", "benchmark"),
        ("control/source-roots.json", "execution_source_manifest"),
        ("src/simsopt/__init__.py", "execution_source"),
        ("src/simsopt_jax/__init__.py", "execution_source"),
        ("src/simsopt_jax_adapters/__init__.py", "execution_source"),
        ("src/simsoptpp.so", "native_extension"),
        ("tests/test_entry.py", "test"),
    )
    roots = []
    for relative, role in relative_roles:
        source = inputs / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"held:{relative}\n".encode())
        roots.append(SourceRoot(role, source, relative))
    worktree = WorktreeIdentity("1" * 40, "2" * 64, "3" * 64, str(inputs))
    publication = publish_immutable_snapshot(
        root / "source-snapshot",
        roots,
        worktree=worktree,
        required_roles=receipt_module.DIAG5_GPU_SNAPSHOT_ROLES,
    )
    entries = {entry.relative_path: entry for entry in publication.entries}
    installed_native = tmp_path / "installed-native.so"
    installed_native.write_bytes((inputs / "src/simsoptpp.so").read_bytes())
    native_sha = hashlib.sha256(installed_native.read_bytes()).hexdigest()
    bindings = tuple(
        ImportBinding(
            module,
            relative,
            entries[relative].size_bytes,
            entries[relative].sha256,
        )
        for module, relative in (
            ("simsopt", "src/simsopt/__init__.py"),
            ("simsopt_jax", "src/simsopt_jax/__init__.py"),
            ("simsopt_jax_adapters", "src/simsopt_jax_adapters/__init__.py"),
            ("simsoptpp", "src/simsoptpp.so"),
        )
    )
    entry = entries["benchmarks/entry.py"]
    entrypoint = ImportBinding(
        "__entrypoint__", entry.relative_path, entry.size_bytes, entry.sha256
    )
    environment = effective_environment({})
    environment_sha = hashlib.sha256(
        canonical_json_bytes(dict(environment))
    ).hexdigest()
    recorded_root = publication.root
    identity = RuntimeIdentityV2(
        argv=(str(recorded_root / "benchmarks/entry.py"),),
        cwd=str(recorded_root),
        python_executable="/fixture/python",
        python_version="3.11",
        jax_version="fixture-jax",
        jaxlib_version="fixture-jaxlib",
        simsopt_module_path=str(recorded_root / "src/simsopt/__init__.py"),
        simsopt_jax_module_path=str(recorded_root / "src/simsopt_jax/__init__.py"),
        native_extension_path=str(installed_native),
        backend="gpu",
        device_uuid=GPU_UUID,
        driver_version="fixture-driver",
        effective_environment_sha256=environment_sha,
        native_extension_sha256=native_sha,
        native_extension_size_bytes=installed_native.stat().st_size,
        native_extension_link_count=installed_native.stat().st_nlink,
    )
    evidence = build_runtime_evidence_v2(
        publication.root,
        source_identity=publication.source_identity(root),
        observation=RuntimeObservationV2(
            identity,
            entrypoint,
            bindings,
            environment,
            "fixture-gpu",
            "fixture-platform",
        ),
        expected_native_extension_path=installed_native,
        expected_native_extension_sha256=native_sha,
        expected_native_extension_size_bytes=installed_native.stat().st_size,
        expected_native_extension_link_count=installed_native.stat().st_nlink,
        required_roles=receipt_module.DIAG5_GPU_SNAPSHOT_ROLES,
    )
    runtime_path = root / "preflight/runtime-evidence.json"
    runtime_path.parent.mkdir(parents=True)
    publish_runtime_evidence_v2(
        runtime_path,
        evidence,
        snapshot_root=publication.root,
        campaign_root=root,
        expected_native_extension_path=installed_native,
        expected_native_extension_sha256=native_sha,
        expected_native_extension_size_bytes=installed_native.stat().st_size,
        expected_native_extension_link_count=installed_native.stat().st_nlink,
        required_roles=receipt_module.DIAG5_GPU_SNAPSHOT_ROLES,
    )
    with receipt_module._Diag5HeldTree(root, require_sealed=False) as held:
        token = receipt_module._DIAG5_HELD_TREE.set(held)
        mirrored_snapshot = receipt_module._diag5_held_path(root, "source-snapshot")
        receipt_module.load_snapshot(
            mirrored_snapshot,
            required_roles=receipt_module.DIAG5_GPU_SNAPSHOT_ROLES,
        )
        receipt_module.validate_diag5_runtime_evidence_v2_bytes(
            held.file_bytes("preflight/runtime-evidence.json"),
            expected_snapshot_identity=publication.identity(),
            expected_logical_campaign_root=root,
            expected_logical_snapshot_root=publication.root,
            expected_native_extension_path=installed_native,
            expected_native_extension_sha256=native_sha,
            expected_native_extension_size_bytes=installed_native.stat().st_size,
            expected_native_extension_link_count=installed_native.stat().st_nlink,
        )
        receipt_module._validate_diag5_manifest_only_auxiliaries(
            root,
            {
                f"preflight_{suffix}": receipt_module.EvidenceSlotV5.absent()
                for suffix in ("memory", "memory_samples", "runtime", "policy")
            },
            {"preflight_process": {}},
            mode="preflight",
            authority=None,
            snapshot=publication,
            expected_snapshot_identity=publication.identity(),
            expected_logical_snapshot_root=publication.root,
            gpu_native_binding=receipt_module.NativeBindingV5(
                "gpu",
                str(installed_native),
                native_sha,
                installed_native.stat().st_size,
                installed_native.stat().st_nlink,
                installed_native.stat().st_dev,
                installed_native.stat().st_ino,
            ),
            expected_gpu_uuid=GPU_UUID,
            physical_memory_bytes=1,
        )
        receipt_module._DIAG5_HELD_TREE.reset(token)


def test_diag5_held_tree_detects_content_metadata_and_namespace_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    evidence = root / "evidence.bin"
    evidence.write_bytes(b"held")
    with receipt_module._Diag5HeldTree(root, require_sealed=False) as held:
        evidence.write_bytes(b"evil")
        with pytest.raises(ValueError, match="changed while reading"):
            held.file_bytes("evidence.bin")
        evidence.write_bytes(b"held")
        evidence.chmod(0o444)
        with pytest.raises(ValueError, match="changed while reading"):
            held.file_bytes("evidence.bin")
    evidence.chmod(0o644)
    with receipt_module._Diag5HeldTree(root, require_sealed=False) as held:
        added = root / "added.bin"
        added.write_bytes(b"late")
        with pytest.raises(ValueError, match="namespace changed"):
            held.revalidate_path_bindings()
    added.unlink()
    with receipt_module._Diag5HeldTree(root, require_sealed=False) as held:
        root.chmod(0o555)
        with pytest.raises(ValueError, match="root changed"):
            held.revalidate_path_bindings()


def test_diag5_held_tree_closes_descriptors_when_binding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    held = receipt_module._Diag5HeldTree(root, require_sealed=False)

    def fail_scan(_descriptor: int, _prefix: str) -> None:
        raise RuntimeError("injected scan failure")

    monkeypatch.setattr(held, "_scan_directory", fail_scan)
    with pytest.raises(RuntimeError, match="injected"):
        held.__enter__()
    assert held.root_descriptor == -1
    assert held._descriptors == []


def test_diag5_held_tree_enforces_sealed_mode_and_link_count(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    file_path = root / "evidence.bin"
    file_path.write_bytes(b"held")
    file_path.chmod(0o444)
    root.chmod(0o555)
    with receipt_module._Diag5HeldTree(root, require_sealed=True) as held:
        assert held.file_bytes("evidence.bin") == b"held"
    root.chmod(0o755)
    link = root / "alias.bin"
    os.link(file_path, link)
    root.chmod(0o555)
    with pytest.raises(ValueError, match="hardlink"), receipt_module._Diag5HeldTree(
        root, require_sealed=True
    ):
        pass
    root.chmod(0o755)
    link.unlink()
    file_path.chmod(0o644)
    root.chmod(0o555)
    with pytest.raises(ValueError, match="file mode"), receipt_module._Diag5HeldTree(
        root, require_sealed=True
    ):
        pass
    file_path.chmod(0o444)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="root mode"), receipt_module._Diag5HeldTree(
        root, require_sealed=True
    ):
        pass


def _diag5_minimal_deep_artifact(
    staging: Path, final: Path
) -> tuple[
    receipt_module.DiagnosticReceiptV5,
    dict[str, object],
    ArtifactRef,
    object,
]:
    staging.mkdir()
    repository = Path(__file__).resolve().parents[2]
    postmortem_path = staging / receipt_module.DIAG5_PREDECESSOR_POSTMORTEM_PATH
    postmortem_path.parent.mkdir(parents=True)
    postmortem_path.write_bytes(
        (
            repository
            / "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json"
        ).read_bytes()
    )
    postmortem = _artifact_ref(
        postmortem_path,
        staging,
        receipt_module.DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
    )
    failure = receipt_module.StructuredFailureV5(
        receipt_module.FailureStageV5.AUTHORITY,
        receipt_module.FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
        "7" * 64,
    )
    terminal_path = (
        staging / receipt_module.DIAG5_EVIDENCE_SLOT_PATHS["supervisor_terminal"]
    )
    terminal_path.write_bytes(
        canonical_json_bytes(
            receipt_module.build_diag5_supervisor_terminal_payload(
                outcome=failure,
                launched_children=(),
                staging_root=staging,
                final_root=final,
            )
        )
    )
    terminal = _artifact_ref(
        terminal_path,
        staging,
        receipt_module.DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    slots = {
        name: receipt_module.EvidenceSlotV5.absent(
            failure.reason if index == 0 else None
        )
        for index, name in enumerate(receipt_module.DIAG5_EVIDENCE_SLOT_PATHS)
    }
    slots["supervisor_terminal"] = receipt_module.EvidenceSlotV5.present(terminal)
    bindings = receipt_module.diag5_native_bindings_payload(
        tuple(
            (
                role,
                receipt_module.NativeBindingV5(
                    role,
                    f"/held/{role}/native.so",
                    "8" * 64,
                    1,
                    1,
                    index + 1,
                    index + 2,
                ),
            )
            for index, role in enumerate(("cpu", "gpu"))
        )
    )
    snapshot_identity = build_snapshot_identity(
        tuple(
            SnapshotEntry(role, f"{index}.bin", 1, f"{index + 1:064x}")
            for index, role in enumerate(
                sorted(receipt_module.DIAG5_GPU_SNAPSHOT_ROLES)
            )
        ),
        WorktreeIdentity("1" * 40, "2" * 64, "3" * 64, "/held/repo"),
    )
    receipt = receipt_module.build_diag5_diagnostic_receipt(
        artifact_root=staging,
        evidence_slots=slots,
        native_bindings=bindings,
        predecessor_postmortem=postmortem,
        expected_native_bindings=bindings,
        expected_authority_sha256="9" * 64,
        expected_predecessor_postmortem=postmortem,
        expected_source_snapshot_identity=snapshot_identity,
        expected_logical_snapshot_root=staging / "source-snapshot",
        expected_frozen_numerical_entries={},
        expected_gpu_uuid=GPU_UUID,
        physical_memory_bytes=1,
    )
    (staging / receipt_module.DIAG2_RECEIPT_FILENAME).write_bytes(
        receipt_module.diag5_diagnostic_receipt_bytes(receipt)
    )
    (staging / receipt_module.DIAG2_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(receipt_module.diag5_artifact_manifest_payload(staging))
    )
    return receipt, bindings, postmortem, snapshot_identity


def test_diag5_public_deep_loaders_cover_writable_staging_final_and_rollback(
    tmp_path: Path,
) -> None:
    final = tmp_path / "diag5-final"
    staging = tmp_path / "diag5-final.partial-claim"
    receipt, bindings, postmortem, snapshot_identity = _diag5_minimal_deep_artifact(
        staging, final
    )
    arguments = {
        "expected_native_bindings": bindings,
        "expected_authority_sha256": "9" * 64,
        "expected_predecessor_postmortem": postmortem,
        "expected_source_snapshot_identity": snapshot_identity,
        "expected_logical_snapshot_root": staging / "source-snapshot",
        "expected_frozen_numerical_entries": {},
        "expected_gpu_uuid": GPU_UUID,
        "physical_memory_bytes": 1,
    }
    assert (
        receipt_module.validate_diag5_writable_staging(staging, **arguments) == receipt
    )
    for path in staging.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    staging.chmod(0o555)
    assert (
        receipt_module.load_and_validate_diag5_staging(staging, **arguments) == receipt
    )
    staging.rename(final)
    assert (
        receipt_module.load_and_validate_diag5_artifact(final, **arguments) == receipt
    )
    rollback = tmp_path / "diag5.rollback"
    final.rename(rollback)
    assert (
        receipt_module.load_and_validate_diag5_rollback(
            rollback,
            expected_rollback_root=rollback,
            expected_final_root=final,
            **arguments,
        )
        == receipt
    )
