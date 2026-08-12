from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import benchmarks.single_stage_native_equivalent_quality_receipt as receipt_module
import numpy as np
import pytest
from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    SnapshotValidationError,
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from benchmarks.single_stage_native_equivalent_quality_receipt import (
    CAMPAIGN_ARTIFACT_MANIFEST_FILENAME,
    CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    NATIVE_RECEIPT_SHA256,
    NATIVE_TRAJECTORY_SHA256,
    OBJECTIVE_MAXIMUM,
    PLAN_SHA256,
    WARM_SOLVE_MAXIMUM_SECONDS,
    CampaignReceipt,
    CandidateEvidence,
    EndpointAuditEvidence,
    EngineeringDisposition,
    ExecutionStatus,
    KktTelemetry,
    KktTelemetryStatus,
    NativeBranchEvidence,
    ReferenceReceipt,
    ResourceEvidence,
    SampleName,
    SampleQuality,
    SampleReceipt,
    SourceIdentityEvidence,
    TimingEvidence,
    campaign_payload,
    campaign_receipt_from_payload,
    campaign_sha256,
    load_and_validate_campaign_artifact,
    load_campaign_receipt,
    reference_receipt_from_artifact,
)
from simsopt_jax.objectives.single_stage_fullspace import TERM_LEDGER


@dataclass(frozen=True, slots=True)
class _AuditBinding:
    accepted_step_count: int
    bootstrap_state_sha256: str


@dataclass(frozen=True, slots=True)
class _AuditContract:
    term_ledger_sha256: str


@dataclass(frozen=True, slots=True)
class _ParsedAudit:
    audited_state_sha256: str
    binding: _AuditBinding
    physics_contract: _AuditContract
    native_reference_state_sha256: str


@dataclass(frozen=True, slots=True)
class _ReferenceValidation:
    disposition: str
    failure_reasons: tuple[str, ...]


def _raw_audit_payload(*, passed: bool = True) -> dict[str, object]:
    return {
        "accepted_step_count": 1,
        "audited_state_sha256": "6" * 64,
        "bootstrap_state_sha256": "2" * 64,
        "ledger_identity_sha256": "4" * 64,
        "native_reference_state_sha256": "3" * 64,
        "semantic_passed": passed,
    }


@pytest.fixture(autouse=True)
def _endpoint_audit_mapping_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = frozenset(_raw_audit_payload())

    def parse(value: object) -> _ParsedAudit:
        if not isinstance(value, dict) or frozenset(value) != expected:
            raise ValueError("endpoint audit keys differ from the frozen schema")
        return _ParsedAudit(
            audited_state_sha256=str(value["audited_state_sha256"]),
            binding=_AuditBinding(
                int(value["accepted_step_count"]),
                str(value["bootstrap_state_sha256"]),
            ),
            physics_contract=_AuditContract(str(value["ledger_identity_sha256"])),
            native_reference_state_sha256=str(value["native_reference_state_sha256"]),
        )

    monkeypatch.setattr(receipt_module, "endpoint_audit_from_payload", parse)
    monkeypatch.setattr(
        receipt_module,
        "validate_endpoint_audit_payload",
        lambda value: isinstance(value, dict) and value.get("semantic_passed") is True,
    )


def _digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _artifact(path: str, sha256: str, schema: str, *, size: int = 100) -> ArtifactRef:
    return ArtifactRef(path, sha256, size, schema)


def _native_branch() -> NativeBranchEvidence:
    return NativeBranchEvidence(
        path_256_successful_knots=257,
        path_512_successful_knots=513,
        common_knot_count=257,
        maximum_common_knot_difference=0.0,
        maximum_scaled_feasibility_256=0.0,
        maximum_scaled_feasibility_512=0.0,
        path_256_terminal_state_sha256="3" * 64,
        path_512_terminal_state_sha256="3" * 64,
        first_failing_index=None,
    )


def _reference() -> ReferenceReceipt:
    branch = _native_branch()
    return ReferenceReceipt(
        produced=True,
        reference_evidence=_artifact(
            "reference/reference.json", "e" * 64, "native-reference-v1"
        ),
        reference_policy_sha256="1" * 64,
        native_receipt_sha256=NATIVE_RECEIPT_SHA256,
        native_trajectory_sha256=NATIVE_TRAJECTORY_SHA256,
        bootstrap_state_sha256="2" * 64,
        physical_state_sha256="3" * 64,
        raw_equalities=(0.0,) * 255,
        constraint_inverse_scale=(1.0,) * 255,
        ledger_identity_sha256="4" * 64,
        native_branch_evidence_sha256=_digest(branch.to_payload()),
        native_branch_evidence=branch,
        failure_reasons=(),
    )


def _timing(seconds: float = 1.0, *, reached: bool = True) -> TimingEvidence:
    start = 2_000_000_000
    stop = start + round(seconds * 1.0e9)
    return TimingEvidence(
        compile_completed_ns=0,
        device_state_ready_ns=1_000_000_000,
        timer_started_ns=start,
        first_hit_synchronized_ns=stop if reached else None,
        timer_stopped_ns=stop,
        audit_started_ns=stop + 1 if reached else None,
        final_transfer_ns=stop + 2 if reached else None,
        serialized_ns=stop + 3,
        synchronized_solve_seconds=seconds,
        endpoint_audit_seconds=1.0 if reached else None,
        total_process_seconds=seconds + 2.0,
    )


def _candidate(*, reached: bool = True) -> CandidateEvidence:
    return CandidateEvidence(
        reached=reached,
        first_hit_attempt=2 if reached else None,
        first_hit_accepted_step=1 if reached else None,
        accepted_step_count=1 if reached else 0,
        state_sha256="6" * 64 if reached else None,
        physical_objective=OBJECTIVE_MAXIMUM if reached else None,
        raw_equalities=(0.0,) * 255 if reached else (),
        scaled_equalities=(0.0,) * 255 if reached else (),
        scaled_feasibility_inf=0.0 if reached else None,
        state_dtype="float64" if reached else None,
        equality_dtype="float64" if reached else None,
        correction_certified=reached,
    )


def _audit() -> EndpointAuditEvidence:
    return EndpointAuditEvidence.from_payload(_raw_audit_payload())


def _resources(index: int) -> ResourceEvidence:
    source = SourceIdentityEvidence(
        git_head="8" * 40,
        tracked_diff_sha256="a" * 64,
        untracked_bytes_manifest_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        source_manifest_size_bytes=100,
    )
    return ResourceEvidence(
        source_identity_sha256=_digest(source.to_payload()),
        pre_source_identity=source,
        post_source_identity=source,
        source_manifest=_artifact(
            "evidence/source-manifest.json", "c" * 64, "source-manifest-v1"
        ),
        runtime_environment_sha256="9" * 64,
        runtime_evidence=_artifact(
            f"evidence/runtime-{index}.json",
            f"{index + 10:x}" * 64,
            "runtime-evidence-v1",
        ),
        reference_policy_sha256="1" * 64,
        backend="gpu",
        device_uuid="GPU-7951f78e-c05d-e01c-303f-d644f4341fe1",
        jax_enable_x64=True,
        child_pid=1000 + index,
        child_start_time_ticks=2000 + index,
        hot_h2d_transfers=0,
        hot_d2h_transfers=0,
        python_callbacks=0,
        final_d2h_transfers=1,
        peak_memory_fraction=0.5,
    )


def _sample(
    name: SampleName,
    index: int,
    *,
    seconds: float = 1.0,
    reached: bool = True,
) -> SampleReceipt:
    return SampleReceipt(
        sample=name,
        producer_evidence=_artifact(
            f"evidence/sample-{index}.json",
            f"{index + 1:x}" * 64,
            "sample-producer-evidence-v1",
        ),
        execution_status=ExecutionStatus.COMPLETED,
        timing=_timing(seconds, reached=reached),
        candidate=_candidate(reached=reached),
        endpoint_audit=_audit() if reached else None,
        resources=_resources(index),
        kkt_telemetry=KktTelemetry(KktTelemetryStatus.AVAILABLE, 1.0e3, 1.0e4),
        failure_reasons=(),
    )


def _complete_campaign() -> CampaignReceipt:
    names = (
        SampleName.COLD,
        SampleName.WARM_1,
        SampleName.WARM_2,
        SampleName.WARM_3,
    )
    return CampaignReceipt(
        reference=_reference(),
        samples=tuple(_sample(name, index) for index, name in enumerate(names)),
    )


def test_final_revision_identity_and_canonical_round_trip(tmp_path: Path) -> None:
    assert (
        PLAN_SHA256
        == "d082baa587b9db580ac3ef8c99a3123ed83564586b605200f7c2cfa6feb909a9"
    )
    assert WARM_SOLVE_MAXIMUM_SECONDS == 287.30421751597896
    receipt = _complete_campaign()
    payload = campaign_payload(receipt)
    encoded = canonical_json_bytes(payload)
    path = tmp_path / "campaign.json"
    path.write_bytes(encoded)

    parsed = campaign_receipt_from_payload(load_canonical_json_bytes(encoded))

    assert parsed == receipt
    assert load_campaign_receipt(path) == receipt
    assert campaign_sha256(receipt) == __import__("hashlib").sha256(encoded).hexdigest()
    assert (
        parsed.disposition() is EngineeringDisposition.ENGINEERING_SPEED_GOAL_ACHIEVED
    )


def test_claimed_terminal_disposition_is_recomputed() -> None:
    payload = campaign_payload(_complete_campaign())
    payload["terminal_disposition"] = EngineeringDisposition.NOT_PRODUCED

    with pytest.raises(ValueError, match="differs from raw evidence"):
        campaign_receipt_from_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("first_hit_attempt", 301),
        ("first_hit_accepted_step", 0),
        ("physical_objective", OBJECTIVE_MAXIMUM + 1.0e-15),
        ("scaled_feasibility_inf", 1.0001e-10),
        ("state_dtype", "float32"),
        ("equality_dtype", "float32"),
        ("correction_certified", False),
    ),
)
def test_each_scalar_candidate_gate_fails_closed(field: str, value: object) -> None:
    cold = _sample(SampleName.COLD, 0)
    candidate = replace(cold.candidate, **{field: value})
    mutated = replace(cold, candidate=candidate)

    assert mutated.quality(_reference()) is SampleQuality.QUALITY_NOT_REACHED


def test_componentwise_raw_equality_gate_is_independent() -> None:
    cold = _sample(SampleName.COLD, 0)
    equalities = list(cold.candidate.raw_equalities)
    equalities[137] = 1.0001e-12

    mutated = replace(
        cold, candidate=replace(cold.candidate, raw_equalities=tuple(equalities))
    )

    assert mutated.quality(_reference()) is SampleQuality.QUALITY_NOT_REACHED


def test_scaled_equality_vector_must_reconstruct_raw_vector_and_norm() -> None:
    cold = _sample(SampleName.COLD, 0)
    scaled = list(cold.candidate.scaled_equalities)
    scaled[12] = 1.0e-12

    inconsistent_vector = replace(
        cold,
        candidate=replace(cold.candidate, scaled_equalities=tuple(scaled)),
    )
    inconsistent_norm = replace(
        cold,
        candidate=replace(cold.candidate, scaled_feasibility_inf=1.0e-12),
    )

    assert (
        inconsistent_vector.quality(_reference()) is SampleQuality.QUALITY_NOT_REACHED
    )
    assert inconsistent_norm.quality(_reference()) is SampleQuality.QUALITY_NOT_REACHED


def test_raw_endpoint_semantic_failure_is_nonpromoting_and_retains_time() -> None:
    cold = _sample(SampleName.COLD, 0)
    mutated = replace(
        cold,
        endpoint_audit=EndpointAuditEvidence.from_payload(
            _raw_audit_payload(passed=False)
        ),
    )

    assert mutated.quality(_reference()) is SampleQuality.DEVICE_QUALITY_CANDIDATE
    campaign = CampaignReceipt(_reference(), (mutated,))
    assert (
        campaign.disposition()
        is EngineeringDisposition.ENDPOINT_AUDIT_FAILED_NONPROMOTING
    )
    assert (
        mutated.timing.synchronized_solve_seconds
        == cold.timing.synchronized_solve_seconds
    )


def test_completed_cold_without_first_hit_is_bounded_negative() -> None:
    cold = replace(
        _sample(SampleName.COLD, 0, reached=False),
        candidate=replace(_candidate(reached=False), accepted_step_count=203),
    )
    campaign = CampaignReceipt(_reference(), (cold,))
    parsed = campaign_receipt_from_payload(campaign_payload(campaign))

    assert parsed.samples[0].candidate.accepted_step_count == 203
    assert parsed.samples[0].endpoint_audit is None
    assert (
        parsed.samples[0].quality(parsed.reference) is SampleQuality.QUALITY_NOT_REACHED
    )
    assert parsed.disposition() is (
        EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
    )


@pytest.mark.parametrize("accepted_step_count", (-1, 257))
def test_no_hit_accepted_step_count_stays_within_solve_bound(
    accepted_step_count: int,
) -> None:
    cold = replace(
        _sample(SampleName.COLD, 0, reached=False),
        candidate=replace(
            _candidate(reached=False), accepted_step_count=accepted_step_count
        ),
    )

    with pytest.raises(ValueError, match="accepted-step count"):
        cold.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_policy_sha256", "a" * 64),
        ("backend", "cpu"),
        ("device_uuid", "GPU-other"),
        ("jax_enable_x64", False),
        ("child_pid", 0),
        ("child_start_time_ticks", 0),
        ("hot_h2d_transfers", 1),
        ("hot_d2h_transfers", 1),
        ("python_callbacks", 1),
        ("final_d2h_transfers", 0),
        ("peak_memory_fraction", 0.8),
    ),
)
def test_each_resource_gate_yields_not_produced(field: str, value: object) -> None:
    cold = _sample(SampleName.COLD, 0)
    mutated = replace(cold, resources=replace(cold.resources, **{field: value}))

    assert (
        CampaignReceipt(_reference(), (mutated,)).disposition()
        is EngineeringDisposition.NOT_PRODUCED
    )


def test_source_identity_is_recomputed_from_pre_post_raw_evidence() -> None:
    cold = _sample(SampleName.COLD, 0)
    changed_post = replace(
        cold.resources,
        post_source_identity=replace(
            cold.resources.post_source_identity, tracked_diff_sha256="d" * 64
        ),
    )
    fabricated_digest = replace(cold.resources, source_identity_sha256="d" * 64)
    mismatched_manifest = replace(
        cold.resources,
        source_manifest=replace(cold.resources.source_manifest, sha256="d" * 64),
    )

    for resources in (changed_post, fabricated_digest, mismatched_manifest):
        assert (
            CampaignReceipt(
                _reference(), (replace(cold, resources=resources),)
            ).disposition()
            is EngineeringDisposition.NOT_PRODUCED
        )


@pytest.mark.parametrize(
    "status", (KktTelemetryStatus.UNAVAILABLE, KktTelemetryStatus.NONFINITE)
)
def test_unavailable_and_nonfinite_kkt_are_explicit_and_non_gating(
    status: KktTelemetryStatus,
) -> None:
    campaign = _complete_campaign()
    samples = tuple(
        replace(sample, kkt_telemetry=KktTelemetry(status, None, None))
        for sample in campaign.samples
    )

    assert CampaignReceipt(campaign.reference, samples).disposition() is (
        EngineeringDisposition.ENGINEERING_SPEED_GOAL_ACHIEVED
    )


def test_nonfinite_kkt_must_not_be_serialized_as_a_number() -> None:
    campaign = _complete_campaign()
    bad = replace(
        campaign.samples[0],
        kkt_telemetry=KktTelemetry(KktTelemetryStatus.AVAILABLE, float("nan"), 1.0),
    )

    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes(
            campaign_payload(CampaignReceipt(campaign.reference, (bad,)))
        )


def test_slow_or_failed_warm_does_not_stop_three_sample_collection() -> None:
    campaign = _complete_campaign()
    slow_warm_1 = replace(
        campaign.samples[1],
        timing=_timing(288.0),
    )
    complete_slow = CampaignReceipt(
        campaign.reference,
        (campaign.samples[0], slow_warm_1, campaign.samples[2], campaign.samples[3]),
    )
    failed_warm_1 = replace(
        campaign.samples[1],
        timing=_timing(2.0, reached=False),
        candidate=_candidate(reached=False),
        endpoint_audit=None,
    )
    complete_failed = CampaignReceipt(
        campaign.reference,
        (campaign.samples[0], failed_warm_1, campaign.samples[2], campaign.samples[3]),
    )

    complete_slow.validate()
    complete_failed.validate()
    assert (
        complete_slow.disposition()
        is EngineeringDisposition.ENGINEERING_SPEED_GOAL_NOT_ACHIEVED
    )
    assert (
        complete_failed.disposition()
        is EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
    )


def test_order_replacement_runtime_reuse_and_partial_warm_prefix_fail_closed() -> None:
    campaign = _complete_campaign()
    reordered = (
        campaign.samples[0],
        campaign.samples[2],
        campaign.samples[1],
        campaign.samples[3],
    )
    duplicate = (
        campaign.samples[0],
        campaign.samples[1],
        campaign.samples[1],
        campaign.samples[3],
    )
    reused_runtime = (
        campaign.samples[0],
        replace(
            campaign.samples[1],
            resources=replace(
                campaign.samples[1].resources,
                runtime_evidence=campaign.samples[0].resources.runtime_evidence,
            ),
        ),
        campaign.samples[2],
        campaign.samples[3],
    )

    for samples in (reordered, duplicate, reused_runtime, campaign.samples[:2]):
        with pytest.raises(ValueError):
            CampaignReceipt(campaign.reference, samples).validate()


def test_reference_not_produced_and_reference_branch_gate() -> None:
    missing = ReferenceReceipt(
        produced=False,
        reference_evidence=_artifact(
            "reference/reference.json", "e" * 64, "native-reference-v1"
        ),
        reference_policy_sha256=None,
        native_receipt_sha256=NATIVE_RECEIPT_SHA256,
        native_trajectory_sha256=NATIVE_TRAJECTORY_SHA256,
        bootstrap_state_sha256=None,
        physical_state_sha256=None,
        raw_equalities=(),
        constraint_inverse_scale=(),
        ledger_identity_sha256=None,
        native_branch_evidence_sha256=None,
        native_branch_evidence=None,
        failure_reasons=("NATIVE_COMMON_KNOT_MISMATCH",),
    )

    assert (
        CampaignReceipt(missing, ()).disposition()
        is EngineeringDisposition.REFERENCE_NOT_PRODUCED
    )
    with pytest.raises(ValueError, match="256/512"):
        branch = replace(_native_branch(), common_knot_count=256)
        replace(
            _reference(),
            native_branch_evidence=branch,
            native_branch_evidence_sha256=_digest(branch.to_payload()),
        ).validate()


def test_bounded_negative_and_not_produced_execution() -> None:
    cold = _sample(SampleName.COLD, 0, reached=False)
    timeout = replace(
        cold,
        execution_status=ExecutionStatus.TIMEOUT,
        resources=replace(cold.resources, final_d2h_transfers=0),
        failure_reasons=("SOLVE_TIMEOUT",),
    )
    compile_failure = replace(
        cold,
        execution_status=ExecutionStatus.COMPILE_FAILURE,
        failure_reasons=("COMPILE_FAILURE",),
    )

    assert CampaignReceipt(_reference(), (timeout,)).disposition() is (
        EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
    )
    assert CampaignReceipt(_reference(), (compile_failure,)).disposition() is (
        EngineeringDisposition.NOT_PRODUCED
    )


def test_timeout_with_stale_candidate_and_audit_evidence_is_nonpromoting() -> None:
    campaign = _complete_campaign()
    stale_timeout = replace(
        campaign.samples[1],
        execution_status=ExecutionStatus.TIMEOUT,
        failure_reasons=("SOLVE_TIMEOUT",),
    )
    receipt = CampaignReceipt(
        campaign.reference,
        (
            campaign.samples[0],
            stale_timeout,
            campaign.samples[2],
            campaign.samples[3],
        ),
    )

    assert (
        stale_timeout.quality(campaign.reference) is SampleQuality.QUALITY_NOT_REACHED
    )
    assert receipt.disposition() is (
        EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
    )
    assert stale_timeout.timing.synchronized_solve_seconds == 1.0


def test_incomplete_endpoint_audit_is_not_produced() -> None:
    cold = _sample(SampleName.COLD, 0)
    incomplete = replace(
        cold,
        timing=replace(
            cold.timing,
            audit_started_ns=None,
            final_transfer_ns=None,
            endpoint_audit_seconds=None,
        ),
        endpoint_audit=None,
    )

    assert (
        CampaignReceipt(_reference(), (incomplete,)).disposition()
        is EngineeringDisposition.NOT_PRODUCED
    )


def test_timing_mutation_is_rejected() -> None:
    cold = _sample(SampleName.COLD, 0)
    mutated = replace(
        cold,
        timing=replace(cold.timing, audit_started_ns=cold.timing.timer_stopped_ns - 1),
    )

    with pytest.raises(ValueError, match="timing boundary"):
        mutated.validate()


def test_duplicate_keys_noncanonical_bytes_and_nonfinite_payload_are_rejected() -> None:
    with pytest.raises(SnapshotValidationError, match="duplicate key"):
        load_canonical_json_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(SnapshotValidationError, match="not canonical"):
        load_canonical_json_bytes(json.dumps({"a": 1}).encode())

    payload = campaign_payload(_complete_campaign())
    samples = payload["samples"]
    assert isinstance(samples, list)
    sample = samples[0]
    assert isinstance(sample, dict)
    candidate = sample["candidate"]
    assert isinstance(candidate, dict)
    candidate["physical_objective"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        campaign_receipt_from_payload(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("reference", "native_common_knot_gate_passed"),
        ("endpoint_audit", "audit_passed"),
        ("resources", "source_unchanged"),
    ),
)
def test_fabricated_summary_booleans_are_rejected(section: str, field: str) -> None:
    payload = campaign_payload(_complete_campaign())
    if section == "reference":
        target = payload["reference"]
    else:
        samples = payload["samples"]
        assert isinstance(samples, list)
        sample = samples[0]
        assert isinstance(sample, dict)
        target = sample[section]
    assert isinstance(target, dict)
    target[field] = True

    with pytest.raises(ValueError, match="keys differ"):
        campaign_receipt_from_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("path_256_successful_knots", 256),
        ("path_512_successful_knots", 512),
        ("common_knot_count", 256),
        ("maximum_common_knot_difference", 1.0001e-10),
        ("maximum_scaled_feasibility_256", 1.0001e-10),
        ("maximum_scaled_feasibility_512", 1.0001e-10),
        ("path_256_terminal_state_sha256", "a" * 64),
        ("path_512_terminal_state_sha256", "a" * 64),
        ("first_failing_index", 1),
    ),
)
def test_each_native_branch_gate_is_recomputed(field: str, value: object) -> None:
    branch = replace(_native_branch(), **{field: value})
    reference = replace(
        _reference(),
        native_branch_evidence=branch,
        native_branch_evidence_sha256=_digest(branch.to_payload()),
    )

    with pytest.raises(ValueError, match="256/512"):
        reference.validate()


def _write_reference_array(
    root: Path, name: str, values: np.ndarray
) -> dict[str, object]:
    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(values, dtype="<f8"), allow_pickle=False)
    payload = buffer.getvalue()
    path = root / "arrays" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "content_sha256": hashlib.sha256(
            np.ascontiguousarray(values, dtype="<f8").tobytes()
        ).hexdigest(),
        "dtype": "<f8",
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "order": "C",
        "relative_path": path.relative_to(root).as_posix(),
        "shape": list(values.shape),
        "size_bytes": len(payload),
    }


def _reference_builder_fixture(root: Path) -> ArtifactRef:
    arrays = {
        "state": _write_reference_array(root, "state", np.zeros((716,))),
        "raw_equalities": _write_reference_array(
            root, "raw-equalities", np.zeros((255,))
        ),
        "coarse_roots": _write_reference_array(root, "coarse", np.zeros((257, 255))),
        "refined_roots": _write_reference_array(root, "refined", np.zeros((513, 255))),
    }
    diagnostics = {
        "coarse_steps": [{"scaled_boozer_infinity_norm": 0.0}],
        "refined_steps": [{"scaled_boozer_infinity_norm": 0.0}],
    }
    diagnostics_bytes = canonical_json_bytes(diagnostics)
    diagnostics_path = root / "manifests" / "diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_bytes(diagnostics_bytes)
    diagnostics_ref = {
        "relative_path": diagnostics_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(diagnostics_bytes).hexdigest(),
        "size_bytes": len(diagnostics_bytes),
    }
    reference_bytes = canonical_json_bytes(
        {
            "diagnostics": diagnostics_ref,
            "evidence": {"arrays": arrays},
        }
    )
    (root / "reference.json").write_bytes(reference_bytes)
    return ArtifactRef(
        "reference/reference.json",
        hashlib.sha256(reference_bytes).hexdigest(),
        len(reference_bytes),
        "single-stage-native-equivalent-reference-v1",
    )


def test_reference_builder_recomputes_policy_bootstrap_scale_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_builder_fixture(tmp_path)
    monkeypatch.setattr(
        receipt_module,
        "validate_native_equivalent_reference",
        lambda _root: _ReferenceValidation("USABLE", ()),
    )
    bootstrap = np.zeros((716,), dtype=np.float64)
    inverse_scale = np.ones((255,), dtype=np.float64)

    receipt = reference_receipt_from_artifact(
        artifact_root=tmp_path,
        reference_evidence=reference,
        bootstrap_state=bootstrap,
        constraint_inverse_scale=inverse_scale,
    )
    changed_scale = reference_receipt_from_artifact(
        artifact_root=tmp_path,
        reference_evidence=reference,
        bootstrap_state=bootstrap,
        constraint_inverse_scale=inverse_scale * 2.0,
    )
    changed_bootstrap = reference_receipt_from_artifact(
        artifact_root=tmp_path,
        reference_evidence=reference,
        bootstrap_state=np.ones((716,), dtype=np.float64),
        constraint_inverse_scale=inverse_scale,
    )

    assert receipt.produced
    assert receipt.raw_equalities == (0.0,) * 255
    assert receipt.constraint_inverse_scale == (1.0,) * 255
    assert receipt.reference_policy_sha256 != changed_scale.reference_policy_sha256
    assert receipt.bootstrap_state_sha256 != changed_bootstrap.bootstrap_state_sha256
    ledger_payload = json.dumps(
        [asdict(row) for row in TERM_LEDGER],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert receipt.ledger_identity_sha256 == hashlib.sha256(ledger_payload).hexdigest()
    with pytest.raises(ValueError, match="bootstrap state"):
        reference_receipt_from_artifact(
            artifact_root=tmp_path,
            reference_evidence=reference,
            bootstrap_state=bootstrap.astype(np.float32),
            constraint_inverse_scale=inverse_scale,
        )


def test_reference_builder_requires_native_array_file_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_builder_fixture(tmp_path)
    monkeypatch.setattr(
        receipt_module,
        "validate_native_equivalent_reference",
        lambda _root: _ReferenceValidation("USABLE", ()),
    )
    document = load_canonical_json_bytes((tmp_path / "reference.json").read_bytes())
    assert isinstance(document, dict)
    evidence = document["evidence"]
    assert isinstance(evidence, dict)
    arrays = evidence["arrays"]
    assert isinstance(arrays, dict)
    state = arrays["state"]
    assert isinstance(state, dict)
    state["sha256"] = state.pop("file_sha256")
    reference_bytes = canonical_json_bytes(document)
    (tmp_path / "reference.json").write_bytes(reference_bytes)
    reference = replace(
        reference,
        sha256=hashlib.sha256(reference_bytes).hexdigest(),
        size_bytes=len(reference_bytes),
    )

    with pytest.raises(ValueError, match="keys differ"):
        reference_receipt_from_artifact(
            artifact_root=tmp_path,
            reference_evidence=reference,
            bootstrap_state=np.zeros((716,), dtype=np.float64),
            constraint_inverse_scale=np.ones((255,), dtype=np.float64),
        )


def test_source_git_head_is_exact_git_object_id_not_sha256() -> None:
    source = _resources(0).pre_source_identity
    source.validate()

    with pytest.raises(ValueError, match="40-hex"):
        replace(source, git_head="8" * 64).validate()


def test_reference_builder_preserves_validated_not_produced_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_builder_fixture(tmp_path)
    monkeypatch.setattr(
        receipt_module,
        "validate_native_equivalent_reference",
        lambda _root: _ReferenceValidation(
            "REFERENCE_NOT_PRODUCED", ("COMMON_KNOT_REFINEMENT",)
        ),
    )

    receipt = reference_receipt_from_artifact(
        artifact_root=tmp_path,
        reference_evidence=reference,
        bootstrap_state=np.zeros((716,), dtype=np.float64),
        constraint_inverse_scale=np.ones((255,), dtype=np.float64),
    )

    assert not receipt.produced
    assert receipt.failure_reasons == ("COMMON_KNOT_REFINEMENT",)
    assert CampaignReceipt(receipt, ()).disposition() is (
        EngineeringDisposition.REFERENCE_NOT_PRODUCED
    )


def _write_canonical(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _artifact_for(root: Path, relative_path: str, schema: str) -> ArtifactRef:
    payload = (root / relative_path).read_bytes()
    return ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=schema,
    )


def _rewrite_campaign_artifact_manifest(root: Path) -> None:
    manifest_path = root / CAMPAIGN_ARTIFACT_MANIFEST_FILENAME
    entries = []
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item != manifest_path
    ):
        payload = path.read_bytes()
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest_path.chmod(0o644)
    _write_canonical(
        manifest_path,
        {
            "schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "entries": entries,
        },
    )
    manifest_path.chmod(0o444)


def _sealed_campaign_artifact(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cold_no_hit: bool = False,
) -> CampaignReceipt:
    root.mkdir()
    reference_path = "native-reference/reference.json"
    source_path = "source-snapshot/source-manifest.json"
    _write_canonical(
        root / reference_path,
        {"schema_version": "native-reference-v1"},
    )
    _write_canonical(
        root / source_path,
        {"schema_version": "source-manifest-v1"},
    )
    reference_ref = _artifact_for(root, reference_path, "native-reference-v1")
    source_ref = _artifact_for(root, source_path, "source-manifest-v1")
    source = SourceIdentityEvidence(
        git_head="8" * 40,
        tracked_diff_sha256="a" * 64,
        untracked_bytes_manifest_sha256="b" * 64,
        source_manifest_sha256=source_ref.sha256,
        source_manifest_size_bytes=source_ref.size_bytes,
    )
    reference = replace(_reference(), reference_evidence=reference_ref)
    samples: list[SampleReceipt] = []
    names = (SampleName.COLD,) if cold_no_hit else tuple(SampleName)
    for index, name in enumerate(names):
        runtime_path = f"samples/{name.value}/runtime-evidence.json"
        _write_canonical(
            root / runtime_path,
            {"schema_version": "runtime-evidence-v1", "sample": name.value},
        )
        runtime_ref = _artifact_for(root, runtime_path, "runtime-evidence-v1")
        resources = replace(
            _resources(index),
            source_identity_sha256=_digest(source.to_payload()),
            pre_source_identity=source,
            post_source_identity=source,
            source_manifest=source_ref,
            runtime_evidence=runtime_ref,
        )
        sample = replace(_sample(name, index), resources=resources)
        if cold_no_hit:
            sample = replace(
                sample,
                timing=_timing(reached=False),
                candidate=replace(_candidate(reached=False), accepted_step_count=203),
                endpoint_audit=None,
            )
        producer_path = f"samples/{name.value}/producer.json"
        producer: dict[str, object] = {
            "schema_version": "single-stage-neq-gntr1-worker-v1",
            "route": "NEQ-GNTR1",
            "plan_sha256": PLAN_SHA256,
            "sample": name.value,
            "execution_status": "COMPLETED",
            "candidate_reached": sample.candidate.reached,
            "candidate": _candidate_payload_for_test(sample.candidate),
            "endpoint_audit": sample.endpoint_audit.to_payload()
            if sample.endpoint_audit is not None
            else None,
            "runtime_evidence": {
                "relative_path": runtime_ref.relative_path,
                "schema_version": runtime_ref.schema_version,
                "sha256": runtime_ref.sha256,
                "size_bytes": runtime_ref.size_bytes,
            },
            "runtime": {
                "backend": resources.backend,
                "device_uuid": resources.device_uuid,
                "jax_enable_x64": resources.jax_enable_x64,
            },
            "timing": {
                "compile_completed_ns": sample.timing.compile_completed_ns,
                "device_state_ready_ns": sample.timing.device_state_ready_ns,
                "timer_started_ns": sample.timing.timer_started_ns,
                "timer_stopped_ns": sample.timing.timer_stopped_ns,
                "audit_started_ns": sample.timing.audit_started_ns,
                "final_transfer_ns": sample.timing.final_transfer_ns,
                "serialized_ns": sample.timing.serialized_ns,
                "synchronized_solve_seconds": (
                    sample.timing.synchronized_solve_seconds
                ),
                "endpoint_audit_seconds": sample.timing.endpoint_audit_seconds,
            },
            "transfer_audit": {
                "hot_h2d_transfers": resources.hot_h2d_transfers,
                "hot_d2h_transfers": resources.hot_d2h_transfers,
                "python_callbacks": resources.python_callbacks,
                "final_d2h_transfers": resources.final_d2h_transfers,
            },
        }
        if name is SampleName.COLD:
            producer["reference_inputs"] = {
                "bootstrap_state": [0.0] * 716,
                "constraint_inverse_scale": [1.0] * 255,
            }
        _write_canonical(root / producer_path, producer)
        producer_ref = _artifact_for(
            root, producer_path, "single-stage-neq-gntr1-worker-v1"
        )
        sample = replace(sample, producer_evidence=producer_ref)
        samples.append(sample)
        _write_canonical(
            root / f"samples/{name.value}/terminal.json",
            {
                "schema_version": "single-stage-neq-gntr1-terminal-v1",
                "sample": name.value,
                "terminal_status": "COMPLETE",
                "child_pid": resources.child_pid,
                "child_start_time_ticks": resources.child_start_time_ticks,
                "process_seconds": sample.timing.total_process_seconds,
                "failure_reasons": [],
            },
        )
        _write_canonical(
            root / f"samples/{name.value}/gpu-memory.json",
            {
                "schema_version": "single-stage-neq-gntr1-memory-v1",
                "child_pid": resources.child_pid,
                "child_start_time_ticks": resources.child_start_time_ticks,
                "device_uuid": resources.device_uuid,
                "peak_memory_fraction": resources.peak_memory_fraction,
            },
        )
    receipt = CampaignReceipt(reference, tuple(samples))
    _write_canonical(root / "campaign.json", campaign_payload(receipt))
    manifest_path = root / CAMPAIGN_ARTIFACT_MANIFEST_FILENAME
    _write_canonical(manifest_path, {})
    _rewrite_campaign_artifact_manifest(root)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    fake_snapshot = SimpleNamespace(
        manifest_path=(root / source_path).resolve(),
        manifest_sha256=source_ref.sha256,
        worktree=SimpleNamespace(
            git_head=source.git_head,
            tracked_diff_sha256=source.tracked_diff_sha256,
            untracked_bytes_manifest_sha256=source.untracked_bytes_manifest_sha256,
        ),
    )
    runtime_identity = SimpleNamespace(
        effective_environment_sha256="9" * 64,
        backend="gpu",
        device_uuid=resources.device_uuid,
    )
    monkeypatch.setattr(receipt_module, "load_snapshot", lambda _root: fake_snapshot)
    monkeypatch.setattr(
        receipt_module,
        "validate_runtime_evidence",
        lambda _path, *, snapshot_root, campaign_root: SimpleNamespace(
            observation=SimpleNamespace(runtime_identity=runtime_identity)
        ),
    )
    monkeypatch.setattr(
        receipt_module, "validate_native_equivalent_reference", lambda _root: None
    )
    monkeypatch.setattr(
        receipt_module,
        "reference_receipt_from_artifact",
        lambda **_arguments: reference,
    )
    return receipt


def _candidate_payload_for_test(candidate: CandidateEvidence) -> dict[str, object]:
    return {
        "accepted_step_count": candidate.accepted_step_count,
        "correction_certified": candidate.correction_certified,
        "equality_dtype": candidate.equality_dtype,
        "first_hit_accepted_step": candidate.first_hit_accepted_step,
        "first_hit_attempt": candidate.first_hit_attempt,
        "physical_objective": candidate.physical_objective,
        "raw_equalities": list(candidate.raw_equalities),
        "reached": candidate.reached,
        "scaled_equalities": list(candidate.scaled_equalities),
        "scaled_feasibility_inf": candidate.scaled_feasibility_inf,
        "state_dtype": candidate.state_dtype,
        "state_sha256": candidate.state_sha256,
    }


def test_deep_campaign_artifact_loader_recomputes_sealed_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    receipt = _sealed_campaign_artifact(root, monkeypatch)

    assert load_and_validate_campaign_artifact(root) == receipt


def test_deep_loader_accepts_completed_cold_without_first_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    receipt = _sealed_campaign_artifact(root, monkeypatch, cold_no_hit=True)

    loaded = load_and_validate_campaign_artifact(root)

    assert loaded == receipt
    assert loaded.disposition() is (
        EngineeringDisposition.QUALITY_NOT_REACHED_BOUNDED_NEGATIVE
    )


@pytest.mark.parametrize(
    "mutation", ("tamper", "missing", "writable", "symlink", "extra")
)
def test_deep_campaign_artifact_loader_rejects_tree_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "campaign"
    _sealed_campaign_artifact(root, monkeypatch)
    runtime_path = root / "samples/cold/runtime-evidence.json"
    sample_root = runtime_path.parent
    if mutation == "tamper":
        runtime_path.chmod(0o644)
        _write_canonical(
            runtime_path,
            {"schema_version": "runtime-evidence-v1", "tampered": True},
        )
        runtime_path.chmod(0o444)
        _rewrite_campaign_artifact_manifest(root)
    elif mutation == "missing":
        sample_root.chmod(0o755)
        runtime_path.unlink()
        sample_root.chmod(0o555)
        _rewrite_campaign_artifact_manifest(root)
    elif mutation == "writable":
        runtime_path.chmod(0o644)
    elif mutation == "symlink":
        sample_root.chmod(0o755)
        (sample_root / "unexpected-link").symlink_to("runtime-evidence.json")
        sample_root.chmod(0o555)
    else:
        sample_root.chmod(0o755)
        extra = sample_root / "extra.json"
        _write_canonical(extra, {"schema_version": "extra-v1"})
        extra.chmod(0o444)
        sample_root.chmod(0o555)
        _rewrite_campaign_artifact_manifest(root)

    with pytest.raises((FileNotFoundError, SnapshotValidationError, ValueError)):
        load_and_validate_campaign_artifact(root)
