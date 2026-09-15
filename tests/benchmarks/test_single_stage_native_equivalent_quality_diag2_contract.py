from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable

import benchmarks.run_single_stage_native_equivalent_quality_campaign as runner
import benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt as receipt
import pytest
from _diag2_fixture import (
    DIAG2_TEST_NONCE,
    artifact_ref_payload,
    current_process_start_ticks,
    frozen_numerical_source_bytes,
    rewrite_json_artifact_ref,
    seal_tree,
    write_artifact_ref,
    write_json_artifact_ref,
)
from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    SnapshotEntry,
    SnapshotRole,
    SourceRoot,
    WorktreeIdentity,
    canonical_json_bytes,
    publish_immutable_snapshot,
)

_NONCE = DIAG2_TEST_NONCE
_GROUPS = (
    ("SETUP_SOURCE", ("source_manifest", "frozen_numerical_subset")),
    ("SETUP_REFERENCE", ("native_reference",)),
    ("SETUP_POLICY", ("policy_authority",)),
    ("ZERO_PREFLIGHT", ("supervisor_before_preflight",)),
    (
        "PREFLIGHT",
        (
            "preflight_producer",
            "preflight_terminal",
            "preflight_process",
            "preflight_memory",
            "preflight_memory_samples",
            "preflight_runtime",
            "preflight_policy",
        ),
    ),
    ("ZERO_COLD", ("supervisor_before_cold",)),
    (
        "COLD_SUPERVISION",
        (
            "cold_producer",
            "cold_terminal",
            "cold_process",
            "cold_memory",
            "cold_memory_samples",
            "cold_runtime",
            "cold_policy",
        ),
    ),
    (
        "COLD_NUMERICAL",
        (
            "cold_history",
            "cold_terminal_numerical",
            "cold_raw_trace",
            "cold_trace_intervals",
            "execution",
        ),
    ),
)
_STAGE_GROUP = {
    stage: group
    for stage, group in (
        (receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE, "SETUP_SOURCE"),
        (receipt.FailureStageV2.NATIVE_REFERENCE_FAILURE, "SETUP_REFERENCE"),
        (receipt.FailureStageV2.POLICY_AUTHORITY_FAILURE, "SETUP_POLICY"),
        (receipt.FailureStageV2.GPU_ZERO_BEFORE_PREFLIGHT_FAILURE, "ZERO_PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_SUPERVISOR_FAILURE, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_TIMEOUT, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_MONITOR_FAILURE, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_PROTOCOL_FAILURE, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_COMPILE_FAILURE, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_CRASH, "PREFLIGHT"),
        (receipt.FailureStageV2.PREFLIGHT_RESOURCE_FAILURE, "PREFLIGHT"),
        (receipt.FailureStageV2.GPU_ZERO_BEFORE_COLD_FAILURE, "ZERO_COLD"),
        (receipt.FailureStageV2.COLD_SOURCE_FAILURE, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_SUPERVISOR_FAILURE, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_TIMEOUT, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_MONITOR_FAILURE, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_PROTOCOL_FAILURE, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_COMPILE_FAILURE, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_CRASH, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.COLD_RESOURCE_FAILURE, "COLD_SUPERVISION"),
        (receipt.FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE, "COLD_NUMERICAL"),
    )
}

_current_start_ticks = current_process_start_ticks
_ref = write_artifact_ref
_json_ref = write_json_artifact_ref
_ref_payload = artifact_ref_payload


def test_runner_preserves_diag2_validation_compatibility_attributes() -> None:
    compatibility = {
        "load_and_validate_diag2_artifact": receipt.load_and_validate_diag2_artifact,
        "load_and_validate_diag2_staging": receipt.load_and_validate_diag2_staging,
        "validate_diag2_writable_staging": receipt.validate_diag2_writable_staging,
    }

    for name, implementation in compatibility.items():
        assert getattr(runner, name) is implementation
        assert name not in runner.__all__


def _publish_source_authority(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    semantic_source_invalid: bool = False,
) -> tuple[ArtifactRef, ArtifactRef]:
    repository = Path(__file__).resolve().parents[2]
    input_root = root.parent / f"{root.name}-source-inputs"
    roles: dict[str, SnapshotRole] = dict(receipt.REQUIRED_SOURCE_ROLE_BINDINGS)
    roles.update(
        {
            path: (
                "benchmark" if path.startswith("benchmarks/") else "execution_source"
            )
            for path, _ in receipt.DIAG2_FROZEN_NUMERICAL_ENTRIES
        }
    )
    roles["src/simsoptpp.so"] = "native_extension"
    frozen_numerical_entries = dict(receipt.DIAG2_FROZEN_NUMERICAL_ENTRIES)
    for relative, role in sorted(roles.items()):
        source = repository / relative
        fixture = input_root / relative
        fixture.parent.mkdir(parents=True, exist_ok=True)
        if relative == "src/simsoptpp.so":
            data = b"fixture-native-extension"
        elif relative in frozen_numerical_entries:
            data = frozen_numerical_source_bytes(
                repository,
                relative,
                frozen_numerical_entries[relative],
            )
        else:
            data = source.read_bytes()
        fixture.write_bytes(data)
    sources = [
        SourceRoot(role, input_root / relative, relative)
        for relative, role in sorted(roles.items())
    ]
    clean_entries = [
        SnapshotEntry(
            role,
            relative,
            (input_root / relative).stat().st_size,
            hashlib.sha256((input_root / relative).read_bytes()).hexdigest(),
        ).to_payload()
        for relative, role in sorted(roles.items())
        if relative not in receipt.DIAG2_SOURCE_DELTA_ALLOWLIST
    ]
    if semantic_source_invalid:
        mutation_relative = next(
            relative
            for relative in sorted(roles)
            if relative not in receipt.DIAG2_SOURCE_DELTA_ALLOWLIST
            and relative != "src/simsoptpp.so"
        )
        mutation = input_root / mutation_relative
        mutation.write_bytes(mutation.read_bytes() + b"coherent-source-mutation\n")
    publication = publish_immutable_snapshot(
        root / "source-snapshot",
        sources,
        worktree=WorktreeIdentity(
            git_head="1" * 40,
            tracked_diff_sha256="2" * 64,
            untracked_bytes_manifest_sha256="3" * 64,
            repo_root=str(repository),
        ),
    )
    filtered = (
        clean_entries
        if semantic_source_invalid
        else receipt._diag2_filtered_source_entries(publication)
    )
    monkeypatch.setattr(receipt, "DIAG2_BASELINE_FILTERED_ENTRY_COUNT", len(filtered))
    monkeypatch.setattr(
        receipt,
        "DIAG2_BASELINE_FILTERED_ENTRIES_SHA256",
        hashlib.sha256(canonical_json_bytes(filtered)).hexdigest(),
    )
    source = publication.source_identity(root).snapshot_manifest
    subset = _json_ref(
        root,
        receipt.DIAG2_EVIDENCE_SLOT_PATHS["frozen_numerical_subset"],
        receipt.DIAG2_FROZEN_SUBSET_SCHEMA_VERSION,
        receipt.build_diag2_frozen_numerical_subset_payload(),
    )
    if not semantic_source_invalid:
        receipt.validate_diag2_frozen_numerical_subset_payload(
            json.loads((root / subset.relative_path).read_bytes()), artifact_root=root
        )
    return source, subset


def _publish_valid_native_reference(root: Path) -> ArtifactRef:
    source = Path(__file__).resolve().parents[2] / (
        "artifacts/neq-native-reference-20260811T012049Z"
    )
    destination = root / "native-reference"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    document = json.loads((destination / "reference.json").read_bytes())
    data = (destination / "reference.json").read_bytes()
    return ArtifactRef(
        "native-reference/reference.json",
        hashlib.sha256(data).hexdigest(),
        len(data),
        str(document["schema_version"]),
    )


def _query_ref(
    root: Path, *, stage_slug: str, query_slug: str, stream: str, data: bytes
) -> ArtifactRef:
    return _ref(
        root,
        f"supervisor/{stage_slug}-{query_slug}.{stream}.bin",
        f"raw-supervisor-{query_slug}-{stream}-v1",
        data,
    )


def _zero_ref(
    root: Path,
    *,
    stage: str,
    monotonic_ns: int,
    reason: receipt.FailureReasonCodeV2 | None = None,
) -> ArtifactRef:
    slug = "before-preflight" if stage == "BEFORE_PREFLIGHT" else "before-cold"
    inventory_stdout = _query_ref(
        root,
        stage_slug=slug,
        query_slug="gpu-inventory",
        stream="stdout",
        data=f"{receipt.GPU_UUID}, 33554432\n".encode(),
    )
    inventory_stderr = _query_ref(
        root,
        stage_slug=slug,
        query_slug="gpu-inventory",
        stream="stderr",
        data=b"",
    )
    parent_present = reason is receipt.FailureReasonCodeV2.GPU_PARENT_PID_PRESENT
    query_failed = reason is receipt.FailureReasonCodeV2.GPU_QUERY_FAILED
    compute_stdout = _query_ref(
        root,
        stage_slug=slug,
        query_slug="compute-apps",
        stream="stdout",
        data=(
            f"{os.getpid()}, {receipt.GPU_UUID}, 1\n".encode()
            if parent_present
            else b""
        ),
    )
    compute_stderr = _query_ref(
        root,
        stage_slug=slug,
        query_slug="compute-apps",
        stream="stderr",
        data=(b"query failed" if query_failed else b""),
    )
    executable_sha = "4" * 64
    inventory = receipt.SupervisorQueryV2(
        receipt._DIAG2_GPU_INVENTORY_ARGV,
        executable_sha,
        True,
        False,
        0,
        inventory_stdout,
        inventory_stderr,
    )
    compute = receipt.SupervisorQueryV2(
        receipt._DIAG2_COMPUTE_APPS_ARGV,
        executable_sha,
        True,
        False,
        1 if query_failed else 0,
        compute_stdout,
        compute_stderr,
    )
    rows = (
        (
            {
                "pid": os.getpid(),
                "gpu_uuid": receipt.GPU_UUID,
                "used_memory_mib": 1,
            },
        )
        if parent_present
        else ()
    )
    payload = receipt.build_diag2_supervisor_zero_payload(
        stage=stage,
        captured_at_monotonic_ns=monotonic_ns,
        captured_at_unix_ns=monotonic_ns + 1_000,
        supervisor_pid=os.getpid(),
        supervisor_start_ticks=_current_start_ticks(),
        gpu_uuid=receipt.GPU_UUID,
        visible_device=receipt.GPU_UUID,
        gpu_inventory_query=inventory,
        compute_apps_query=compute,
        matching_rows=rows,
    )
    return _json_ref(
        root,
        receipt.DIAG2_EVIDENCE_SLOT_PATHS[
            "supervisor_before_preflight"
            if stage == "BEFORE_PREFLIGHT"
            else "supervisor_before_cold"
        ],
        receipt.DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
        payload,
    )


def _process_ref(
    root: Path,
    mode: str,
    *,
    monitor_failure_kind: str = "NONE",
    returncode: int = 0,
) -> ArtifactRef:
    child_pid = os.getpid() + (101 if mode == "preflight" else 102)
    started = 20 if mode == "preflight" else 50
    stdout = _ref(root, f"{mode}/stdout.bin", "raw-process-stdout-v1", b"")
    stderr = _ref(root, f"{mode}/stderr.bin", "raw-process-stderr-v1", b"")
    return _json_ref(
        root,
        receipt.DIAG2_EVIDENCE_SLOT_PATHS[f"{mode}_process"],
        receipt.DIAG2_PROCESS_SCHEMA_VERSION,
        {
            "schema_version": receipt.DIAG2_PROCESS_SCHEMA_VERSION,
            "child_pid": child_pid,
            "child_start_time_ticks": (
                0 if monitor_failure_kind == "BINDING" else child_pid + 100
            ),
            "argv": ["/fixture/python", "runner.py", mode],
            "stdout": _ref_payload(stdout),
            "stderr": _ref_payload(stderr),
            "process_seconds": 1.0,
            "process_diagnostics": {"returncode": returncode},
            "monitor_failure_kind": monitor_failure_kind,
            "pre_source_identity": {},
            "post_source_identity": {},
            "process_started_monotonic_ns": started,
            "process_stopped_monotonic_ns": started + 10,
        },
    )


def _child_terminal_ref(
    root: Path,
    mode: str,
    status: str = "COMPLETE",
    *,
    monitor_failure_kind: str = "NONE",
) -> ArtifactRef:
    schema = receipt.DIAG2_CHILD_TERMINAL_SCHEMA_VERSION
    return _json_ref(
        root,
        receipt.DIAG2_EVIDENCE_SLOT_PATHS[f"{mode}_terminal"],
        schema,
        {
            "schema_version": schema,
            "terminal_status": status,
            "failure_reasons": [] if status == "COMPLETE" else ["typed-failure"],
            "monitor_failure_kind": monitor_failure_kind,
        },
    )


def _generic_ref(root: Path, name: str) -> ArtifactRef:
    if name in {"preflight_memory", "cold_memory"}:
        schema = f"fixture-{name}-v1"
        return _json_ref(
            root,
            receipt.DIAG2_EVIDENCE_SLOT_PATHS[name],
            schema,
            {
                "schema_version": schema,
                "peak_memory_fraction": 0.1,
            },
        )
    if name == "cold_terminal_numerical":
        arrays: dict[str, object] = {}
        for array_name in receipt.ARRAY_SPECS:
            array_ref = _ref(
                root,
                f"cold/arrays/{array_name}.npy",
                f"fixture-array-{array_name}-v1",
                f"fixture:{array_name}".encode(),
            )
            arrays[array_name] = {"artifact": _ref_payload(array_ref)}
        schema = f"fixture-{name}-v1"
        return _json_ref(
            root,
            receipt.DIAG2_EVIDENCE_SLOT_PATHS[name],
            schema,
            {"schema_version": schema, "arrays": arrays},
        )
    relative = receipt.DIAG2_EVIDENCE_SLOT_PATHS[name].replace(
        "<run>/<base>.trace.json.gz", "run/fixture.trace.json.gz"
    )
    schema = f"fixture-{name}-v1"
    reference = _json_ref(
        root, relative, schema, {"schema_version": schema, "slot": name}
    )
    if name == "cold_raw_trace":
        (root / relative).with_name("fixture.xplane.pb").write_bytes(b"xplane")
    return reference


def _compile_producer_ref(
    root: Path, mode: str, reason: receipt.FailureReasonCodeV2
) -> tuple[ArtifactRef, ArtifactRef]:
    runtime = _generic_ref(root, f"{mode}_runtime")
    payload = receipt.build_diag2_compile_failure_producer_payload(
        mode=mode,
        execution_status=(
            "COMPILE_OOM"
            if reason is receipt.FailureReasonCodeV2.CHILD_COMPILE_OOM
            else "COMPILE_FAILURE"
        ),
        runtime={
            "backend": "gpu",
            "device": "fixture-gpu",
            "device_uuid": receipt.GPU_UUID,
            "jax": "fixture",
            "jax_enable_x64": True,
            "jaxlib": "fixture",
            "python": "fixture",
        },
        runtime_evidence=runtime,
        compile_started_ns=1,
        compile_completed_ns=2,
        process_seconds_before_serialization=1.0,
        failure_reasons=(reason.value,),
    )
    return (
        _json_ref(
            root,
            receipt.DIAG2_EVIDENCE_SLOT_PATHS[f"{mode}_producer"],
            str(payload["schema_version"]),
            payload,
        ),
        runtime,
    )


def _success_producer_ref(
    root: Path, mode: str
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    runtime = _generic_ref(root, f"{mode}_runtime")
    policy = _generic_ref(root, f"{mode}_policy")
    if mode == "preflight":
        payload: dict[str, object] = {
            "schema_version": "single-stage-neq-gntr1-preflight-worker-v1",
            "route": receipt.DIAG2_ROUTE,
            "plan_sha256": receipt.DIAG2_PLAN_SHA256,
            "mode": "ANNOTATED_LOWER_COMPILE_ONLY",
            "execution_status": "SUCCESS",
            "policy_sha256": "5" * 64,
            "policy_evidence": _ref_payload(policy),
            "phase_schema_sha256": receipt.PHASE_SCHEMA_SHA256,
            "state_size": 716,
            "equality_size": 255,
            "residual_size": 2110,
            "campaign_authorized": False,
            "solver_dispatched": False,
            "finalizer_called": False,
            "endpoint_audit_called": False,
            "python_callbacks": 0,
            "runtime": {},
            "runtime_evidence": _ref_payload(runtime),
            "timing": {},
            "failure_reasons": [],
        }
    else:
        fake = ArtifactRef("cold/history.json", "6" * 64, 1, "fixture")
        payload = {
            "schema_version": f"{receipt.SCHEMA_VERSION}-producer",
            "route": receipt.DIAG2_ROUTE,
            "plan_sha256": receipt.DIAG2_PLAN_SHA256,
            "execution_status": "COMPLETE",
            "runtime": {},
            "runtime_evidence": _ref_payload(runtime),
            "policy_sha256": "5" * 64,
            "phase_schema_sha256": receipt.PHASE_SCHEMA_SHA256,
            "history_evidence": _ref_payload(fake),
            "terminal_numerical_evidence": _ref_payload(fake),
            "policy_evidence": _ref_payload(policy),
            "raw_trace_evidence": _ref_payload(fake),
            "trace_intervals_evidence": _ref_payload(fake),
            "timestamps_ns": {},
            "transfer_audit": {},
            "endpoint_audit_called": False,
            "campaign_authorized": False,
            "failure_reasons": [],
        }
    receipt.validate_diag2_producer_payload(payload, mode=mode)
    producer = _json_ref(
        root,
        receipt.DIAG2_EVIDENCE_SLOT_PATHS[f"{mode}_producer"],
        str(payload["schema_version"]),
        payload,
    )
    return producer, runtime, policy


def _terminal_status(reason: receipt.FailureReasonCodeV2) -> str:
    return {
        receipt.FailureReasonCodeV2.CHILD_TIMEOUT: "TIMEOUT",
        receipt.FailureReasonCodeV2.CHILD_EXIT_NONZERO: "CRASH",
        receipt.FailureReasonCodeV2.CHILD_COMPILE_FAILED: "COMPILE_FAILURE",
        receipt.FailureReasonCodeV2.CHILD_COMPILE_OOM: "COMPILE_FAILURE",
        receipt.FailureReasonCodeV2.PRODUCER_DECODE_FAILED: "PROTOCOL_FAILURE",
        receipt.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: "PROTOCOL_FAILURE",
        receipt.FailureReasonCodeV2.MONITOR_BINDING_FAILED: "MONITOR_FAILURE",
        receipt.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: "COMPLETE",
        receipt.FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED: "COMPLETE",
    }.get(reason, "PROTOCOL_FAILURE")


def _own_child_refs(
    root: Path,
    refs: dict[str, ArtifactRef | None],
    *,
    mode: str,
    reason: receipt.FailureReasonCodeV2,
) -> None:
    if reason in {
        receipt.FailureReasonCodeV2.SOURCE_PRE,
        receipt.FailureReasonCodeV2.CHILD_LAUNCH_FAILED,
    }:
        return
    monitor_failure_kind = {
        receipt.FailureReasonCodeV2.MONITOR_BINDING_FAILED: "BINDING",
        receipt.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: "FINALIZATION",
    }.get(reason, "NONE")
    refs[f"{mode}_terminal"] = _child_terminal_ref(
        root,
        mode,
        _terminal_status(reason),
        monitor_failure_kind=monitor_failure_kind,
    )
    refs[f"{mode}_process"] = _process_ref(
        root,
        mode,
        monitor_failure_kind=monitor_failure_kind,
        returncode=(
            1 if reason is receipt.FailureReasonCodeV2.CHILD_EXIT_NONZERO else 0
        ),
    )
    if reason not in {
        receipt.FailureReasonCodeV2.MONITOR_BINDING_FAILED,
        receipt.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
    }:
        refs[f"{mode}_memory"] = _generic_ref(root, f"{mode}_memory")
        if reason is receipt.FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED:
            _rewrite_ref(
                root,
                refs,
                f"{mode}_memory",
                lambda payload: payload.__setitem__("peak_memory_fraction", 0.9),
            )
        refs[f"{mode}_memory_samples"] = _generic_ref(root, f"{mode}_memory_samples")
    if reason in {
        receipt.FailureReasonCodeV2.CHILD_COMPILE_FAILED,
        receipt.FailureReasonCodeV2.CHILD_COMPILE_OOM,
    }:
        producer, runtime = _compile_producer_ref(root, mode, reason)
        refs[f"{mode}_producer"] = producer
        refs[f"{mode}_runtime"] = runtime
    elif reason in {
        receipt.FailureReasonCodeV2.SOURCE_POST,
        receipt.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
        receipt.FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
        receipt.FailureReasonCodeV2.POLICY_SCHEMA_INVALID,
        receipt.FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED,
    }:
        producer, runtime, policy = _success_producer_ref(root, mode)
        refs[f"{mode}_producer"] = producer
        refs[f"{mode}_runtime"] = runtime
        refs[f"{mode}_policy"] = policy
        if reason is receipt.FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID:
            (root / runtime.relative_path).write_bytes(b"retained-invalid-runtime")
            refs[f"{mode}_runtime"] = None
        elif reason is receipt.FailureReasonCodeV2.POLICY_SCHEMA_INVALID:
            (root / policy.relative_path).write_bytes(b"retained-invalid-policy")
            refs[f"{mode}_policy"] = None


def _publish_prior_group(
    root: Path,
    refs: dict[str, ArtifactRef | None],
    group: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_semantic_invalid: bool = False,
) -> None:
    if group == "SETUP_SOURCE":
        refs["source_manifest"], refs["frozen_numerical_subset"] = (
            _publish_source_authority(
                root,
                monkeypatch,
                semantic_source_invalid=source_semantic_invalid,
            )
        )
    elif group == "SETUP_REFERENCE":
        refs["native_reference"] = _generic_ref(root, "native_reference")
    elif group == "SETUP_POLICY":
        refs["policy_authority"] = _generic_ref(root, "policy_authority")
    elif group == "ZERO_PREFLIGHT":
        refs["supervisor_before_preflight"] = _zero_ref(
            root, stage="BEFORE_PREFLIGHT", monotonic_ns=10
        )
    elif group == "PREFLIGHT":
        producer, runtime, policy = _success_producer_ref(root, "preflight")
        refs["preflight_producer"] = producer
        refs["preflight_runtime"] = runtime
        refs["preflight_policy"] = policy
        refs["preflight_terminal"] = _child_terminal_ref(root, "preflight")
        refs["preflight_process"] = _process_ref(root, "preflight")
        refs["preflight_memory"] = _generic_ref(root, "preflight_memory")
        refs["preflight_memory_samples"] = _generic_ref(
            root, "preflight_memory_samples"
        )
    elif group == "ZERO_COLD":
        refs["supervisor_before_cold"] = _zero_ref(
            root, stage="BEFORE_COLD", monotonic_ns=40
        )
    elif group == "COLD_SUPERVISION":
        producer, runtime, policy = _success_producer_ref(root, "cold")
        refs["cold_producer"] = producer
        refs["cold_runtime"] = runtime
        refs["cold_policy"] = policy
        refs["cold_terminal"] = _child_terminal_ref(root, "cold")
        refs["cold_process"] = _process_ref(root, "cold")
        refs["cold_memory"] = _generic_ref(root, "cold_memory")
        refs["cold_memory_samples"] = _generic_ref(root, "cold_memory_samples")
    elif group == "COLD_NUMERICAL":
        for name in dict(_GROUPS)[group]:
            refs[name] = _generic_ref(root, name)


def _failure_refs(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: receipt.FailureStageV2,
    reason: receipt.FailureReasonCodeV2,
    *,
    postlaunch_untyped: bool = False,
    later_untyped: tuple[str, ...] = (),
    setup_untyped: bool = False,
    subordinate_child_reason: receipt.FailureReasonCodeV2 | None = None,
) -> tuple[dict[str, ArtifactRef | None], receipt.StructuredFailureV2]:
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt.DIAG2_EVIDENCE_SLOT_NAMES
    }
    postlaunch_drift = stage in {
        receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
        receipt.FailureStageV2.COLD_SOURCE_FAILURE,
    } and reason in {
        receipt.FailureReasonCodeV2.SOURCE_POST,
        receipt.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        receipt.FailureReasonCodeV2.REFERENCE_INVALID,
        receipt.FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
    }
    semantic_source_invalid = (
        postlaunch_drift
        and reason is receipt.FailureReasonCodeV2.SOURCE_POST
        and not postlaunch_untyped
    )
    own_group = _STAGE_GROUP[stage]
    own_index = tuple(name for name, _ in _GROUPS).index(own_group)
    for group, _ in _GROUPS[:own_index]:
        _publish_prior_group(
            root,
            refs,
            group,
            monkeypatch,
            source_semantic_invalid=(
                semantic_source_invalid and group == "SETUP_SOURCE"
            ),
        )
    if postlaunch_drift:
        extra_groups = (
            ("PREFLIGHT",)
            if stage is receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            else ("COLD_SUPERVISION",)
        )
        for group in extra_groups:
            mode = "preflight" if group == "PREFLIGHT" else "cold"
            if subordinate_child_reason is None:
                _publish_prior_group(root, refs, group, monkeypatch)
            else:
                _own_child_refs(
                    root,
                    refs,
                    mode=mode,
                    reason=subordinate_child_reason,
                )
        offending = {
            receipt.FailureReasonCodeV2.SOURCE_POST: "source_manifest",
            receipt.FailureReasonCodeV2.FROZEN_SUBSET_INVALID: (
                "frozen_numerical_subset"
            ),
            receipt.FailureReasonCodeV2.REFERENCE_INVALID: "native_reference",
            receipt.FailureReasonCodeV2.POLICY_DERIVATION_INVALID: "policy_authority",
        }[reason]
        if reason is receipt.FailureReasonCodeV2.POLICY_DERIVATION_INVALID:
            refs["native_reference"] = _publish_valid_native_reference(root)
        if postlaunch_untyped:
            refs[offending] = None
            canonical = root / receipt.DIAG2_EVIDENCE_SLOT_PATHS[offending]
            if offending == "source_manifest":
                canonical.chmod(0o644)
                canonical.write_bytes(b"retained-minimum-untyped-source-manifest")
            elif offending in {"frozen_numerical_subset", "policy_authority"}:
                canonical.write_bytes(b"retained-minimum-untyped-setup-authority")
            elif offending == "native_reference":
                canonical.write_bytes(b"retained-minimum-untyped-native-reference")
        elif reason is receipt.FailureReasonCodeV2.FROZEN_SUBSET_INVALID:
            _rewrite_ref(
                root,
                refs,
                "frozen_numerical_subset",
                lambda payload: payload["entries"][0].__setitem__("sha256", "e" * 64),
            )
        elif reason is receipt.FailureReasonCodeV2.REFERENCE_INVALID:
            (root / "native-reference/retained-array.bin").write_bytes(
                b"semantic-invalid-reference-nested-bytes"
            )
        for later in later_untyped:
            refs[later] = None
            canonical = root / receipt.DIAG2_EVIDENCE_SLOT_PATHS[later]
            if later in {"frozen_numerical_subset", "policy_authority"}:
                canonical.write_bytes(b"retained-later-minimum-untyped-authority")
            elif later == "native_reference":
                canonical.write_bytes(b"retained-later-minimum-untyped-reference")
    if own_group == "SETUP_SOURCE":
        if reason is not receipt.FailureReasonCodeV2.SOURCE_PRE:
            source, subset = _publish_source_authority(
                root,
                monkeypatch,
                semantic_source_invalid=(
                    reason is receipt.FailureReasonCodeV2.SOURCE_POST
                    and not setup_untyped
                ),
            )
            if setup_untyped and reason is receipt.FailureReasonCodeV2.SOURCE_POST:
                manifest = root / source.relative_path
                manifest.chmod(0o644)
                manifest.write_bytes(b"minimum-untyped-source-manifest")
                (root / subset.relative_path).unlink()
            elif setup_untyped:
                refs["source_manifest"] = source
                (root / subset.relative_path).write_bytes(
                    b"retained-minimum-untyped-frozen-subset"
                )
            else:
                refs["source_manifest"] = source
                if reason is receipt.FailureReasonCodeV2.SOURCE_POST:
                    (root / subset.relative_path).unlink()
                else:
                    refs["frozen_numerical_subset"] = subset
                    _rewrite_ref(
                        root,
                        refs,
                        "frozen_numerical_subset",
                        lambda payload: payload["entries"][0].__setitem__(
                            "sha256", "e" * 64
                        ),
                    )
    elif own_group == "SETUP_REFERENCE":
        if setup_untyped:
            _ref(
                root,
                receipt.DIAG2_EVIDENCE_SLOT_PATHS["native_reference"],
                "minimum-untyped-native-reference",
                b"minimum-untyped-native-reference",
            )
        else:
            refs["native_reference"] = _generic_ref(root, "native_reference")
            (root / "native-reference/retained-array.bin").write_bytes(
                b"semantic-invalid-reference-nested-bytes"
            )
    elif own_group == "SETUP_POLICY":
        refs["native_reference"] = _publish_valid_native_reference(root)
        if setup_untyped:
            (root / receipt.DIAG2_EVIDENCE_SLOT_PATHS["policy_authority"]).write_bytes(
                b"minimum-untyped-policy-authority"
            )
        else:
            refs["policy_authority"] = _generic_ref(root, "policy_authority")
    elif own_group == "ZERO_PREFLIGHT":
        refs["supervisor_before_preflight"] = _zero_ref(
            root, stage="BEFORE_PREFLIGHT", monotonic_ns=10, reason=reason
        )
    elif own_group == "ZERO_COLD":
        refs["supervisor_before_cold"] = _zero_ref(
            root, stage="BEFORE_COLD", monotonic_ns=40, reason=reason
        )
    elif own_group == "PREFLIGHT" and not postlaunch_drift:
        _own_child_refs(root, refs, mode="preflight", reason=reason)
    elif own_group == "COLD_SUPERVISION" and not postlaunch_drift:
        _own_child_refs(root, refs, mode="cold", reason=reason)
    else:
        names = dict(_GROUPS)["COLD_NUMERICAL"]
        if reason is receipt.FailureReasonCodeV2.SEMANTIC_VALIDATION_FAILED:
            for name in names:
                refs[name] = _generic_ref(root, name)
        else:
            failing_index = (
                3
                if reason is receipt.FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED
                else 0
            )
            for name in names[:failing_index]:
                refs[name] = _generic_ref(root, name)
    failure = receipt.StructuredFailureV2(stage, reason, "7" * 64)
    launched = (
        ()
        if own_index <= 3
        else (("preflight",) if own_index <= 5 else ("preflight", "cold"))
    )
    terminal_payload = receipt.build_diag2_supervisor_terminal_payload(
        disposition="INCOMPLETE",
        failure=failure,
        launched_children=launched,
        policy_authority_produced=own_index >= 3,
        preflight_authorized=own_index >= 5,
        cold_authorized=own_index >= 6,
        staging_root=root,
        final_root=root.parent / "diag2",
        nonce=_NONCE,
        algorithm_route_selection="NOT_PRODUCED",
    )
    refs["supervisor_terminal"] = _json_ref(
        root,
        receipt.DIAG2_EVIDENCE_SLOT_PATHS["supervisor_terminal"],
        receipt.DIAG2_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
        terminal_payload,
    )
    return refs, failure


def _sealed_failure_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: receipt.FailureStageV2,
    reason: receipt.FailureReasonCodeV2,
    *,
    postlaunch_untyped: bool = False,
    later_untyped: tuple[str, ...] = (),
    setup_untyped: bool = False,
    subordinate_child_reason: receipt.FailureReasonCodeV2 | None = None,
    retained_trace_suffixes: tuple[str, ...] = (),
    raw_child_stderr: bytes | None = None,
) -> tuple[Path, receipt.DiagnosticReceiptV2]:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    staging.mkdir()
    refs, failure = _failure_refs(
        staging,
        monkeypatch,
        stage,
        reason,
        postlaunch_untyped=postlaunch_untyped,
        later_untyped=later_untyped,
        setup_untyped=setup_untyped,
        subordinate_child_reason=subordinate_child_reason,
    )
    for suffix in retained_trace_suffixes:
        retained = staging / f"cold/raw-trace/plugins/profile/run/fixture{suffix}"
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(f"retained:{suffix}".encode())
    if raw_child_stderr is not None:
        process_name = (
            "cold_process" if refs["cold_process"] is not None else "preflight_process"
        )

        def rewrite_stderr(payload: dict[str, object]) -> None:
            stderr = payload["stderr"]
            assert isinstance(stderr, dict)
            raw = staging / str(stderr["relative_path"])
            raw.write_bytes(raw_child_stderr)
            stderr["sha256"] = hashlib.sha256(raw_child_stderr).hexdigest()
            stderr["size_bytes"] = len(raw_child_stderr)

        _rewrite_ref(staging, refs, process_name, rewrite_stderr)
    slots = receipt.derive_diag2_evidence_slots(
        artifact_root=staging, artifact_refs=refs, failure=failure
    )
    built = receipt.build_diag2_diagnostic_receipt(
        artifact_root=staging, evidence_slots=slots
    )
    (staging / receipt.DIAG2_RECEIPT_FILENAME).write_bytes(
        receipt.diag2_diagnostic_receipt_bytes(built)
    )
    (staging / receipt.DIAG2_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(receipt.diag2_artifact_manifest_payload(staging))
    )
    assert receipt.validate_diag2_writable_staging(staging) == built
    seal_tree(staging)
    assert receipt.load_and_validate_diag2_staging(staging) == built
    return staging, built


def _seal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: receipt.FailureStageV2,
    reason: receipt.FailureReasonCodeV2,
    *,
    postlaunch_untyped: bool = False,
    later_untyped: tuple[str, ...] = (),
    setup_untyped: bool = False,
    subordinate_child_reason: receipt.FailureReasonCodeV2 | None = None,
    retained_trace_suffixes: tuple[str, ...] = (),
    raw_child_stderr: bytes | None = None,
) -> tuple[Path, receipt.DiagnosticReceiptV2]:
    staging, built = _sealed_failure_staging(
        tmp_path,
        monkeypatch,
        stage,
        reason,
        postlaunch_untyped=postlaunch_untyped,
        later_untyped=later_untyped,
        setup_untyped=setup_untyped,
        subordinate_child_reason=subordinate_child_reason,
        retained_trace_suffixes=retained_trace_suffixes,
        raw_child_stderr=raw_child_stderr,
    )
    final = tmp_path / "diag2"
    runner._atomic_publish_diag2(runner.Diag2Publication(staging, final, _NONCE))
    return final, built


_STAGE_REASON_CASES = tuple(
    (stage, reason)
    for stage, reasons in receipt.DIAG2_STAGE_REASON_CODES.items()
    for reason in sorted(reasons, key=lambda item: item.value)
)

_SETUP_ORDER = (
    "source_manifest",
    "frozen_numerical_subset",
    "native_reference",
    "policy_authority",
)
_SETUP_OFFENDING = {
    receipt.FailureReasonCodeV2.SOURCE_PRE: "source_manifest",
    receipt.FailureReasonCodeV2.SOURCE_POST: "source_manifest",
    receipt.FailureReasonCodeV2.FROZEN_SUBSET_INVALID: "frozen_numerical_subset",
    receipt.FailureReasonCodeV2.REFERENCE_INVALID: "native_reference",
    receipt.FailureReasonCodeV2.POLICY_DERIVATION_INVALID: "policy_authority",
}


def _expected_legal_failure_slot_vector(
    stage: receipt.FailureStageV2,
    reason: receipt.FailureReasonCodeV2,
) -> dict[str, tuple[receipt.EvidenceState, receipt.AbsenceReason | None]]:
    expected = {
        name: (receipt.EvidenceState.ABSENT, receipt.AbsenceReason.NOT_REACHED)
        for name in receipt.DIAG2_EVIDENCE_SLOT_NAMES
    }
    expected["supervisor_terminal"] = (receipt.EvidenceState.PRESENT, None)

    def present_group(group: str) -> None:
        for name in dict(_GROUPS)[group]:
            expected[name] = (receipt.EvidenceState.PRESENT, None)

    initial_setup_stages = {
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureStageV2.NATIVE_REFERENCE_FAILURE,
        receipt.FailureStageV2.POLICY_AUTHORITY_FAILURE,
    }
    if stage in initial_setup_stages:
        offending = _SETUP_OFFENDING[reason]
        offending_index = _SETUP_ORDER.index(offending)
        for name in _SETUP_ORDER[:offending_index]:
            expected[name] = (receipt.EvidenceState.PRESENT, None)
        expected[offending] = (
            (
                receipt.EvidenceState.ABSENT,
                receipt.AbsenceReason.SOURCE_PRE,
            )
            if reason is receipt.FailureReasonCodeV2.SOURCE_PRE
            else (receipt.EvidenceState.PRESENT, None)
        )
        return expected

    postlaunch_setup = stage in {
        receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
        receipt.FailureStageV2.COLD_SOURCE_FAILURE,
    }
    if postlaunch_setup:
        final_group = (
            "PREFLIGHT"
            if stage is receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE
            else "COLD_SUPERVISION"
        )
        for group, _ in _GROUPS:
            present_group(group)
            if group == final_group:
                break
        return expected

    own_group = _STAGE_GROUP[stage]
    own_index = tuple(group for group, _ in _GROUPS).index(own_group)
    for group, _ in _GROUPS[:own_index]:
        present_group(group)
    direct = receipt.AbsenceReason(reason.value)
    if own_group in {"ZERO_PREFLIGHT", "ZERO_COLD"}:
        present_group(own_group)
        return expected
    if own_group in {"PREFLIGHT", "COLD_SUPERVISION"}:
        mode = "preflight" if own_group == "PREFLIGHT" else "cold"
        suffixes = {
            receipt.FailureReasonCodeV2.CHILD_LAUNCH_FAILED: frozenset(),
            receipt.FailureReasonCodeV2.CHILD_TIMEOUT: frozenset(
                {"terminal", "process", "memory", "memory_samples"}
            ),
            receipt.FailureReasonCodeV2.MONITOR_BINDING_FAILED: frozenset(
                {"terminal", "process"}
            ),
            receipt.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: frozenset(
                {"producer", "terminal", "process", "runtime", "policy"}
            ),
            receipt.FailureReasonCodeV2.PRODUCER_DECODE_FAILED: frozenset(
                {"terminal", "process", "memory", "memory_samples"}
            ),
            receipt.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: frozenset(
                {"terminal", "process", "memory", "memory_samples"}
            ),
            receipt.FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID: frozenset(
                {
                    "producer",
                    "terminal",
                    "process",
                    "memory",
                    "memory_samples",
                    "policy",
                }
            ),
            receipt.FailureReasonCodeV2.POLICY_SCHEMA_INVALID: frozenset(
                {
                    "producer",
                    "terminal",
                    "process",
                    "memory",
                    "memory_samples",
                    "runtime",
                }
            ),
            receipt.FailureReasonCodeV2.CHILD_COMPILE_FAILED: frozenset(
                {
                    "producer",
                    "terminal",
                    "process",
                    "memory",
                    "memory_samples",
                    "runtime",
                }
            ),
            receipt.FailureReasonCodeV2.CHILD_COMPILE_OOM: frozenset(
                {
                    "producer",
                    "terminal",
                    "process",
                    "memory",
                    "memory_samples",
                    "runtime",
                }
            ),
            receipt.FailureReasonCodeV2.CHILD_EXIT_NONZERO: frozenset(
                {"terminal", "process", "memory", "memory_samples"}
            ),
            receipt.FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED: frozenset(
                {
                    "producer",
                    "terminal",
                    "process",
                    "memory",
                    "memory_samples",
                    "runtime",
                    "policy",
                }
            ),
        }[reason]
        for suffix in (
            "producer",
            "terminal",
            "process",
            "memory",
            "memory_samples",
            "runtime",
            "policy",
        ):
            name = f"{mode}_{suffix}"
            if suffix in suffixes:
                expected[name] = (receipt.EvidenceState.PRESENT, None)
            else:
                expected[name] = (receipt.EvidenceState.ABSENT, direct)
        return expected
    numerical_names = dict(_GROUPS)["COLD_NUMERICAL"]
    if reason is receipt.FailureReasonCodeV2.SEMANTIC_VALIDATION_FAILED:
        for name in numerical_names:
            expected[name] = (receipt.EvidenceState.PRESENT, None)
    else:
        failing_index = (
            3 if reason is receipt.FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED else 0
        )
        for name in numerical_names[:failing_index]:
            expected[name] = (receipt.EvidenceState.PRESENT, None)
        expected[numerical_names[failing_index]] = (
            receipt.EvidenceState.ABSENT,
            direct,
        )
    return expected


_FAILURE_STAGE_ORDER = tuple(receipt.DIAG2_STAGE_REASON_CODES)
_ADJACENT_STAGE_REASON_MISMATCHES = tuple(
    (earlier, next(iter(receipt.DIAG2_STAGE_REASON_CODES[later])))
    for earlier, later in zip(
        _FAILURE_STAGE_ORDER[:-1],
        _FAILURE_STAGE_ORDER[1:],
        strict=True,
    )
)


@pytest.mark.parametrize(
    ("stage", "next_stage_reason"),
    _ADJACENT_STAGE_REASON_MISMATCHES,
    ids=lambda value: value.value,
)
def test_each_adjacent_failure_stage_rejects_the_next_stage_direct_reason(
    tmp_path: Path,
    stage: receipt.FailureStageV2,
    next_stage_reason: receipt.FailureReasonCodeV2,
) -> None:
    refs = {name: None for name in receipt.DIAG2_EVIDENCE_SLOT_NAMES}
    failure = receipt.StructuredFailureV2(stage, next_stage_reason, "8" * 64)
    with pytest.raises(ValueError, match="stage/reason pairing differs"):
        receipt.derive_diag2_evidence_slots(
            artifact_root=tmp_path,
            artifact_refs=refs,
            failure=failure,
        )


@pytest.mark.parametrize(
    ("earlier_stage", "later_stage"),
    tuple(
        zip(
            receipt.DIAG2_FAILURE_STAGE_ORDER[:-1],
            receipt.DIAG2_FAILURE_STAGE_ORDER[1:],
            strict=True,
        )
    ),
    ids=lambda value: value.value,
)
def test_each_adjacent_competing_failure_boundary_selects_frozen_earlier_stage(
    earlier_stage: receipt.FailureStageV2,
    later_stage: receipt.FailureStageV2,
) -> None:
    earlier = receipt.StructuredFailureV2(
        earlier_stage,
        min(
            receipt.DIAG2_STAGE_REASON_CODES[earlier_stage], key=lambda item: item.value
        ),
        "1" * 64,
    )
    later = receipt.StructuredFailureV2(
        later_stage,
        min(receipt.DIAG2_STAGE_REASON_CODES[later_stage], key=lambda item: item.value),
        "2" * 64,
    )
    assert receipt.select_diag2_failure((later, earlier)) == earlier


_POSTLAUNCH_REASONS = (
    receipt.FailureReasonCodeV2.SOURCE_POST,
    receipt.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
    receipt.FailureReasonCodeV2.REFERENCE_INVALID,
    receipt.FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
)

_SUBORDINATE_PREFLIGHT_CHILD_CASES = (
    (
        receipt.FailureReasonCodeV2.CHILD_TIMEOUT,
        frozenset(
            {
                "preflight_terminal",
                "preflight_process",
                "preflight_memory",
                "preflight_memory_samples",
            }
        ),
    ),
    (
        receipt.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
        frozenset(
            {
                "preflight_producer",
                "preflight_terminal",
                "preflight_process",
                "preflight_runtime",
                "preflight_policy",
            }
        ),
    ),
    (
        receipt.FailureReasonCodeV2.CHILD_COMPILE_OOM,
        frozenset(
            {
                "preflight_producer",
                "preflight_terminal",
                "preflight_process",
                "preflight_memory",
                "preflight_memory_samples",
                "preflight_runtime",
            }
        ),
    ),
    (
        receipt.FailureReasonCodeV2.CHILD_EXIT_NONZERO,
        frozenset(
            {
                "preflight_terminal",
                "preflight_process",
                "preflight_memory",
                "preflight_memory_samples",
            }
        ),
    ),
    (
        receipt.FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
        frozenset(
            {
                "preflight_terminal",
                "preflight_process",
                "preflight_memory",
                "preflight_memory_samples",
            }
        ),
    ),
    (
        receipt.FailureReasonCodeV2.MEMORY_LIMIT_EXCEEDED,
        frozenset(dict(_GROUPS)["PREFLIGHT"]),
    ),
)


@pytest.mark.parametrize(
    "mutation",
    (
        {"state": "UNKNOWN", "reason": "CHILD_TIMEOUT"},
        {"reason": "CHILD_TIMEOUT"},
        {"state": "ABSENT", "reason": "INVALID"},
        {"state": "PRESENT", "reason": "CHILD_TIMEOUT"},
        {"state": "ABSENT", "artifact": {}},
    ),
)
def test_diag2_slot_union_rejects_unknown_or_crossed_variants(
    mutation: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        receipt.parse_diag2_evidence_slot(mutation, name="cold_history")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("relative_path", "cold/not-history.json"),
        ("sha256", "not-a-sha"),
        ("size_bytes", True),
        ("schema_version", 1),
    ),
)
def test_diag2_present_slot_rejects_bad_artifact_identity(
    field: str, value: object
) -> None:
    artifact: dict[str, object] = {
        "relative_path": "cold/history.json",
        "sha256": "8" * 64,
        "size_bytes": 1,
        "schema_version": "fixture-v1",
    }
    artifact[field] = value
    with pytest.raises((TypeError, ValueError)):
        receipt.parse_diag2_evidence_slot(
            {"state": "PRESENT", "artifact": artifact}, name="cold_history"
        )


_CHILD_TERMINAL_STATUSES = (
    "COMPLETE",
    "TIMEOUT",
    "MONITOR_FAILURE",
    "CRASH",
    "PROTOCOL_FAILURE",
    "COMPILE_FAILURE",
)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("verdict", "DIAGNOSTIC_COMPLETE_NO_HIT"),
        ("historical_relation", "NO_PRIOR_HIT"),
        ("quality", {}),
        ("phase_attribution", {}),
        ("next_route", "RADIUS_RETRACTION"),
        ("promotion_authorized", True),
        ("engineering_campaign_receipt_produced", True),
        ("formal_comparison", "PRODUCED"),
    ),
)
def test_diag2_incomplete_verdict_biconditional_rejects_claim_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    final, expected = _seal_failure(
        tmp_path,
        monkeypatch,
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
    )
    payload = receipt.diag2_diagnostic_receipt_payload(expected)
    payload[field] = value
    with pytest.raises(ValueError):
        receipt.diag2_diagnostic_receipt_from_payload(payload, artifact_root=final)


@pytest.mark.parametrize(
    "identity_field",
    ("schema_version", "route", "numerical_route", "plan_sha256"),
)
def test_v1_v2_receipt_identity_is_never_aliased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    final, expected = _seal_failure(
        tmp_path,
        monkeypatch,
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
    )
    payload = receipt.diag2_diagnostic_receipt_payload(expected)
    payload[identity_field] = {
        "schema_version": receipt.SCHEMA_VERSION,
        "route": receipt.ROUTE,
        "numerical_route": receipt.ROUTE,
        "plan_sha256": receipt.PLAN_SHA256,
    }[identity_field]
    with pytest.raises(ValueError, match="identity|schema"):
        receipt.diag2_diagnostic_receipt_from_payload(payload, artifact_root=final)
    with pytest.raises(ValueError):
        receipt.load_diagnostic_receipt_bytes(
            receipt.diag2_diagnostic_receipt_bytes(expected), artifact_root=final
        )


ManifestMutation = Callable[[Path, dict[str, object]], None]


def _remove_entry(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries.pop()


def _duplicate_entry(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries.append(dict(entries[0]))


def _reverse_entries(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries.reverse()


def _bad_digest(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries[0]["sha256"] = "9" * 64


def _bad_size(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries[0]["size_bytes"] += 1


def _bad_role(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries[0]["role"] = "wrong"


def _path_traversal(_root: Path, manifest: dict[str, object]) -> None:
    entries = manifest["entries"]
    assert isinstance(entries, list)
    entries[0]["relative_path"] = "../escape"


@pytest.mark.parametrize(
    "mutation",
    (
        _remove_entry,
        _duplicate_entry,
        _reverse_entries,
        _bad_digest,
        _bad_size,
        _bad_role,
        _path_traversal,
    ),
    ids=lambda mutation: mutation.__name__,
)
def test_diag2_manifest_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: ManifestMutation,
) -> None:
    final, _ = _seal_failure(
        tmp_path,
        monkeypatch,
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
    )
    manifest_path = final / receipt.DIAG2_MANIFEST_FILENAME
    os.chmod(final, 0o755)
    os.chmod(manifest_path, 0o644)
    manifest = json.loads(manifest_path.read_bytes())
    mutation(final, manifest)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    os.chmod(manifest_path, 0o444)
    os.chmod(final, 0o555)
    with pytest.raises((OSError, TypeError, ValueError)):
        receipt.load_and_validate_diag2_artifact(final)


@pytest.mark.parametrize(
    "node_kind", ("extra", "symlink", "hardlink", "special", "mode")
)
def test_diag2_filesystem_manifest_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_kind: str,
) -> None:
    final, _ = _seal_failure(
        tmp_path,
        monkeypatch,
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
    )
    os.chmod(final, 0o755)
    terminal = final / "supervisor-terminal.json"
    if node_kind == "extra":
        (final / "extra.bin").write_bytes(b"extra")
    elif node_kind == "symlink":
        (final / "alias").symlink_to(terminal)
    elif node_kind == "hardlink":
        os.link(terminal, final / "hardlink")
    elif node_kind == "special":
        os.mkfifo(final / "fifo")
    else:
        os.chmod(terminal, 0o644)
    os.chmod(final, 0o555)
    with pytest.raises(ValueError):
        receipt.load_and_validate_diag2_artifact(final)


@pytest.mark.parametrize(
    ("stage", "launched", "policy", "preflight", "cold"),
    (
        (
            receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            (),
            True,
            False,
            False,
        ),
        (
            receipt.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            ("preflight",),
            False,
            False,
            False,
        ),
        (
            receipt.FailureStageV2.COLD_SOURCE_FAILURE,
            ("preflight",),
            True,
            True,
            True,
        ),
        (
            receipt.FailureStageV2.COLD_SOURCE_FAILURE,
            ("preflight", "cold"),
            True,
            True,
            False,
        ),
    ),
)
def test_postlaunch_setup_drift_terminal_join_is_biconditional(
    tmp_path: Path,
    stage: receipt.FailureStageV2,
    launched: tuple[str, ...],
    policy: bool,
    preflight: bool,
    cold: bool,
) -> None:
    failure = receipt.StructuredFailureV2(
        stage, receipt.FailureReasonCodeV2.SOURCE_POST, "a" * 64
    )
    with pytest.raises(ValueError, match="setup-drift terminal invariants"):
        receipt.build_diag2_supervisor_terminal_payload(
            disposition="INCOMPLETE",
            failure=failure,
            launched_children=launched,
            policy_authority_produced=policy,
            preflight_authorized=preflight,
            cold_authorized=cold,
            staging_root=tmp_path / f"diag2.partial-{_NONCE}",
            final_root=tmp_path / "diag2",
            nonce=_NONCE,
            algorithm_route_selection="NOT_PRODUCED",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema_version", "wrong"),
        ("route", "wrong"),
        ("plan_sha256", "0" * 64),
        ("disposition", "UNKNOWN"),
        ("failure_stage", None),
        ("failure_reason", None),
        ("launched_children", ["cold"]),
        ("policy_authority_produced", 1),
        ("preflight_authorized", 1),
        ("cold_authorized", 1),
        ("publication", {"staging_root": "/tmp", "final_root": "/tmp"}),
        ("engineering_campaign_receipt_produced", True),
        ("promotion_authorized", True),
        ("formal_comparison", "PRODUCED"),
        ("algorithm_route_selection", "RADIUS_RETRACTION"),
    ),
)
def test_every_supervisor_terminal_field_mutation_fails_reconstruction(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    failure = receipt.StructuredFailureV2(
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
        "a" * 64,
    )
    payload = receipt.build_diag2_supervisor_terminal_payload(
        disposition="INCOMPLETE",
        failure=failure,
        launched_children=(),
        policy_authority_produced=False,
        preflight_authorized=False,
        cold_authorized=False,
        staging_root=tmp_path / f"diag2.partial-{_NONCE}",
        final_root=tmp_path / "diag2",
        nonce=_NONCE,
        algorithm_route_selection="NOT_PRODUCED",
    )
    payload[field] = invalid_value
    with pytest.raises((TypeError, ValueError)):
        receipt._parse_diag2_supervisor_terminal(payload)


@pytest.mark.parametrize(
    ("nested_field", "invalid_value"),
    (
        ("code", receipt.FailureReasonCodeV2.REFERENCE_INVALID.value),
        ("detail_sha256", "not-a-sha256"),
    ),
)
def test_supervisor_terminal_nested_failure_reason_mutations_fail_reconstruction(
    tmp_path: Path,
    nested_field: str,
    invalid_value: str,
) -> None:
    failure = receipt.StructuredFailureV2(
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
        "a" * 64,
    )
    payload = receipt.build_diag2_supervisor_terminal_payload(
        disposition="INCOMPLETE",
        failure=failure,
        launched_children=(),
        policy_authority_produced=False,
        preflight_authorized=False,
        cold_authorized=False,
        staging_root=tmp_path / f"diag2.partial-{_NONCE}",
        final_root=tmp_path / "diag2",
        nonce=_NONCE,
        algorithm_route_selection="NOT_PRODUCED",
    )
    nested = payload["failure_reason"]
    assert isinstance(nested, dict)
    nested[nested_field] = invalid_value
    with pytest.raises((TypeError, ValueError)):
        receipt._parse_diag2_supervisor_terminal(payload)


_rewrite_ref = rewrite_json_artifact_ref


def _set_field(field: str, value: object) -> Callable[[dict[str, object]], None]:
    def mutate(payload: dict[str, object]) -> None:
        payload[field] = value

    return mutate


_DUAL_QUERY_FIELDS = (
    "launched",
    "timed_out",
    "returncode",
    "argv",
    "stdout",
    "stderr",
)


def test_diag2_seal_fsync_failure_leaves_only_partial_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    staging.mkdir()
    (staging / "raw.bin").write_bytes(b"raw")
    final = tmp_path / "diag2"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(runner.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        runner._seal_and_sync_diag2_staging(staging)
    assert staging.is_dir()
    assert not final.exists()


@pytest.mark.parametrize(
    ("failure_call", "substage"),
    ((1, "regular_file"), (3, "nested_directory"), (4, "staging_root")),
)
def test_each_seal_fsync_substage_failure_keeps_final_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
    substage: str,
) -> None:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    nested = staging / "nested"
    nested.mkdir(parents=True)
    (staging / "root.bin").write_bytes(b"root")
    (nested / "nested.bin").write_bytes(b"nested")
    final = tmp_path / "diag2"
    original_fsync = runner.os.fsync
    calls = 0

    def fail_selected(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError(f"injected {substage} fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_selected)
    with pytest.raises(OSError, match=f"injected {substage} fsync failure"):
        runner._seal_and_sync_diag2_staging(staging)
    assert staging.is_dir()
    assert not final.exists()


@pytest.mark.parametrize(
    "chmod_substage",
    ("recursive_file", "nested_directory", "staging_root"),
)
def test_each_recursive_seal_chmod_substage_failure_keeps_final_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chmod_substage: str,
) -> None:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    nested = staging / "nested"
    nested.mkdir(parents=True)
    nested_file = nested / "nested.bin"
    nested_file.write_bytes(b"nested")
    final = tmp_path / "diag2"
    target = {
        "recursive_file": nested_file,
        "nested_directory": nested,
        "staging_root": staging,
    }[chmod_substage]
    original_chmod = Path.chmod

    def fail_target(path: Path, mode: int, *args: object, **kwargs: object) -> None:
        if path == target:
            raise OSError(f"injected {chmod_substage} chmod failure")
        original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", fail_target)
    with pytest.raises(OSError, match=f"injected {chmod_substage} chmod failure"):
        runner._seal_and_sync_diag2_staging(staging)
    assert staging.is_dir()
    assert not final.exists()


@pytest.mark.parametrize("publisher_kind", ("json", "bytes"))
def test_diag2_raw_artifact_fsync_failure_never_creates_final_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher_kind: str,
) -> None:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    staging.mkdir()
    artifact = staging / f"raw.{publisher_kind}"
    final = tmp_path / "diag2"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected raw artifact fsync failure")

    monkeypatch.setattr(runner.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected raw artifact fsync failure"):
        if publisher_kind == "json":
            runner._publish_canonical_json(artifact, {"schema_version": "fixture-v1"})
        else:
            runner._publish_bytes(artifact, b"fixture")
    assert artifact.is_file()
    assert staging.is_dir()
    assert not final.exists()


def test_diag2_postrename_parent_fsync_failure_retains_reloadable_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, expected = _sealed_failure_staging(
        tmp_path,
        monkeypatch,
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
    )
    final = tmp_path / "diag2"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(runner.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected parent fsync failure"):
        runner._atomic_publish_diag2(runner.Diag2Publication(staging, final, _NONCE))
    assert not staging.exists()
    assert receipt.load_and_validate_diag2_artifact(final) == expected


def test_diag2_atomic_publish_never_replaces_existing_final(tmp_path: Path) -> None:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    staging.mkdir()
    final = tmp_path / "diag2"
    final.mkdir()
    marker = final / "owner.bin"
    marker.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        runner._atomic_publish_diag2(runner.Diag2Publication(staging, final, _NONCE))
    assert staging.is_dir()
    assert marker.read_bytes() == b"existing"


@pytest.mark.parametrize(
    "fault_point",
    (
        "after_terminal",
        "after_receipt",
        "after_manifest",
        "after_seal",
        "after_deep_load",
        "after_rename",
    ),
)
def test_terminal_to_publication_fault_points_preserve_atomic_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    staging = tmp_path / f"diag2.partial-{_NONCE}"
    staging.mkdir()
    final = tmp_path / "diag2"
    publication = runner.Diag2Publication(staging, final, _NONCE)
    refs: dict[str, ArtifactRef | None] = {
        name: None for name in receipt.DIAG2_EVIDENCE_SLOT_NAMES
    }
    failure = receipt.StructuredFailureV2(
        receipt.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        receipt.FailureReasonCodeV2.SOURCE_PRE,
        "f" * 64,
    )

    def injected(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {fault_point}")

    if fault_point == "after_terminal":
        monkeypatch.setattr(runner, "derive_diag3_evidence_slots", injected)
    elif fault_point == "after_receipt":
        original_publish_json = runner._publish_canonical_json
        calls = 0

        def fail_manifest(path: Path, payload: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected after_receipt")
            assert isinstance(payload, dict)
            original_publish_json(path, payload)

        monkeypatch.setattr(runner, "_publish_canonical_json", fail_manifest)
    elif fault_point == "after_manifest":
        monkeypatch.setattr(runner, "validate_diag3_writable_staging", injected)
    elif fault_point == "after_seal":
        monkeypatch.setattr(runner, "load_and_validate_diag3_staging", injected)
    elif fault_point == "after_deep_load":
        monkeypatch.setattr(runner, "_atomic_publish_diag2", injected)
    else:
        monkeypatch.setattr(runner, "load_and_validate_diag3_artifact", injected)
    with pytest.raises(OSError, match=f"injected {fault_point}"):
        runner._publish_diag2_terminal_and_receipt(
            publication,
            refs,
            failure=failure,
            launched_children=(),
            policy_authority_produced=False,
            preflight_authorized=False,
            cold_authorized=False,
        )
    if fault_point == "after_rename":
        assert not staging.exists()
        loaded = receipt.load_and_validate_diag3_artifact(final)
        assert loaded.failure == failure
    else:
        assert staging.is_dir()
        assert not final.exists()
