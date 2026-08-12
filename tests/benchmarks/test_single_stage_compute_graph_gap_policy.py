from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Literal

import pytest
from benchmarks.single_stage_compute_graph_attribution_control import (
    COMMAND_BUFFER_DISABLE_FLAG,
    AttributionAttempt,
    AttributionBinding,
    build_attribution_evidence,
)
from benchmarks.single_stage_compute_graph_complete_path import (
    GAP_BUDGET_INPUTS_SCHEMA_ID,
    build_gap_budget_inputs_artifact,
    validate_gap_budget_inputs_artifact,
)
from benchmarks.single_stage_compute_graph_gap_policy import (
    PLAN_PATH,
    GapPolicyProducerError,
    _plan_levers,
    build_phase0_gap_policy,
    produce_phase0_gap_policy,
)
from benchmarks.single_stage_compute_graph_phase0_post_gate import _load_policy
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes


def _binding() -> AttributionBinding:
    return AttributionBinding(
        candidate_sha256="a" * 64,
        specimen_sha256="b" * 64,
        input_bundle_sha256="c" * 64,
        source_sha256="d" * 64,
        production_runtime_identity_sha256="1" * 64,
        lane_id="rtx5090",
        gpu_uuid="GPU-test",
        gate_checkpoint_sha256="e" * 64,
        warm_checkpoint_sha256="f" * 64,
        warm_p50_ns=1_000.0,
    )


def _attempt(
    mode: Literal["default_control", "command_buffer_disabled"], index: int
) -> AttributionAttempt:
    disabled = mode == "command_buffer_disabled"
    return AttributionAttempt(
        mode=mode,
        attempt_index=index,
        binding=_binding(),
        runtime_identity_sha256=("2" if disabled else "1") * 64,
        xla_flag_tokens=(COMMAND_BUFFER_DISABLE_FLAG,) if disabled else (),
        compilation_cache_root=f"/cache/{mode}/{index}",
        artifact_root=f"/artifact/{mode}/{index}",
        raw_trace_path=f"trace/{mode}/{index}.json.gz",
        raw_trace_sha256="3" * 64,
        child_observation_path=f"child/{mode}/{index}.json",
        child_observation_sha256="4" * 64,
        hlo_anchor_path=f"hlo/{mode}/{index}.json",
        hlo_anchor_sha256="5" * 64,
        profile_derivation_version="compute-graph-profile-attribution-v1",
        objective=12.5,
        gradient=(1.0, -2.0, 3.0),
        solve_certificate={
            "inner_newton_success": True,
            "adjoint_success": True,
            "residual_certificates": {
                "adjoint_residual_l2": 1.0e-13,
                "adjoint_residual_relative": 2.0e-13,
            },
        },
        module_topology_identity_sha256="9" * 64,
        evaluation_envelope_ns=1_000,
        device_active_ns=800,
        phase_device_ns=(
            ("newton.linear_solve", 500),
            ("adjoint.implicit_coil_vjp", 260),
        ),
    )


def _attribution() -> dict[str, object]:
    return build_attribution_evidence(
        tuple(_attempt("default_control", index) for index in range(3)),
        tuple(_attempt("command_buffer_disabled", index) for index in range(3)),
    )


def _complete_path() -> dict[str, object]:
    binding = _binding()
    return {
        "schema_id": "single-stage-compute-graph-complete-path-v2",
        "identity": {
            "candidate_sha256": binding.candidate_sha256,
            "specimen_sha256": binding.specimen_sha256,
            "input_bundle_sha256": binding.input_bundle_sha256,
            "source_sha256": binding.source_sha256,
            "runtime_identity_sha256": binding.production_runtime_identity_sha256,
            "native_reference_sha256": "6" * 64,
            "gate_checkpoint_sha256": binding.gate_checkpoint_sha256,
            "warm_checkpoint_sha256": binding.warm_checkpoint_sha256,
            "warm_p50_ns": binding.warm_p50_ns,
            "lane_id": binding.lane_id,
            "gpu_uuid": binding.gpu_uuid,
        },
        "matched_complete_path_reference_timings_ns": {
            "native": 2_000,
            "c0": 8_000,
            "optax": 3_000,
        },
        "lanes": {"c0": {"optimizer_counts": {"nfev": 4}}},
    }


def _write_canonical(path: Path, document: object) -> None:
    path.write_bytes(canonical_json_bytes(document))


def test_policy_is_plan_derived_unmeasured_and_existing_schema_compatible(
    tmp_path: Path,
) -> None:
    complete = _complete_path()
    attribution = _attribution()
    complete_path = tmp_path / "complete.json"
    attribution_path = tmp_path / "attribution.json"
    output_path = tmp_path / "policy.json"
    _write_canonical(complete_path, complete)
    _write_canonical(attribution_path, attribution)

    produced = produce_phase0_gap_policy(
        complete_path_path=complete_path,
        attribution_evidence_path=attribution_path,
        output_path=output_path,
    )

    assert produced == output_path
    policy = _load_policy(output_path)
    assert set(policy.phase_reduction_assumptions) == {
        "adjoint.implicit_coil_vjp",
        "newton.linear_solve",
    }
    assert all(
        assumption.conservative_reduction == 0.0
        and assumption.optimistic_reduction == 1.0
        and assumption.overlap_disposition == "disjoint"
        for assumption in policy.phase_reduction_assumptions.values()
    )
    assert policy.unattributed_conservative_reduction == 0.0
    assert policy.unattributed_optimistic_reduction == 1.0
    assert tuple(lever.lever_id for lever in policy.faithful_levers) == (
        "phase-1-dense-direct-exact-newton-canaries",
        "phase-2-adjoint-assembly-and-exact-final-state-factor-handoff",
        "phase-3-fuse-the-scalar-coil-pullback",
        "phase-4-remove-measured-launch-fragmentation",
    )
    assert all(lever.disposition == "unbounded" for lever in policy.faithful_levers)
    assert tuple(lever.evidence_sha256 for lever in policy.faithful_levers) == tuple(
        lever.evidence_sha256 for lever in _plan_levers(PLAN_PATH)
    )
    artifact = build_gap_budget_inputs_artifact(complete, policy)
    assert artifact["schema_id"] == GAP_BUDGET_INPUTS_SCHEMA_ID
    validate_gap_budget_inputs_artifact(artifact, complete)
    assert output_path.read_bytes() == canonical_json_bytes(
        build_phase0_gap_policy(complete, attribution)
    )


def test_policy_rejects_tampered_attribution_summary() -> None:
    attribution = _attribution()
    tampered = copy.deepcopy(attribution)
    tampered["selected_attribution"]["phase_shares"][0][
        "selected_default_envelope_share"
    ] = 0.99

    with pytest.raises(GapPolicyProducerError, match="not promotion-safe"):
        build_phase0_gap_policy(_complete_path(), tampered)


def test_policy_rejects_complete_path_attribution_identity_mismatch() -> None:
    complete = _complete_path()
    complete["identity"]["candidate_sha256"] = "0" * 64

    with pytest.raises(GapPolicyProducerError, match="candidate_sha256"):
        build_phase0_gap_policy(complete, _attribution())


def test_policy_rejects_invalid_current_formal_target_inputs() -> None:
    complete = _complete_path()
    complete["matched_complete_path_reference_timings_ns"]["native"] = 0

    with pytest.raises(GapPolicyProducerError, match="formal inputs are invalid"):
        build_phase0_gap_policy(complete, _attribution())


def test_policy_refuses_nonfresh_output_before_reading_inputs(tmp_path: Path) -> None:
    output_path = tmp_path / "policy.json"
    original = b"owned-by-an-earlier-run\n"
    output_path.write_bytes(original)

    with pytest.raises(GapPolicyProducerError, match="must not already exist"):
        produce_phase0_gap_policy(
            complete_path_path=tmp_path / "missing-complete.json",
            attribution_evidence_path=tmp_path / "missing-attribution.json",
            output_path=output_path,
        )

    assert output_path.read_bytes() == original


def test_plan_sections_are_exact_evidence_and_plan_drift_fails_closed(
    tmp_path: Path,
) -> None:
    levers = _plan_levers(PLAN_PATH)
    assert len(levers) == 4
    assert len({lever.evidence_sha256 for lever in levers}) == 4
    assert all(
        len(lever.evidence_sha256) == hashlib.sha256().digest_size * 2
        for lever in levers
    )
    drifted = tmp_path / "plan.md"
    drifted.write_text(
        PLAN_PATH.read_text(encoding="utf-8").replace(
            "### Phase 3 — Fuse the scalar coil pullback",
            "### Phase 7 — Fuse the scalar coil pullback",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(GapPolicyProducerError, match="unique and contiguous"):
        _plan_levers(drifted)


def test_policy_rejects_noncanonical_upstream_json(tmp_path: Path) -> None:
    complete_path = tmp_path / "complete.json"
    attribution_path = tmp_path / "attribution.json"
    output_path = tmp_path / "policy.json"
    complete_path.write_text(str(_complete_path()), encoding="utf-8")
    _write_canonical(attribution_path, _attribution())

    with pytest.raises(GapPolicyProducerError, match="not valid UTF-8 JSON"):
        produce_phase0_gap_policy(
            complete_path_path=complete_path,
            attribution_evidence_path=attribution_path,
            output_path=output_path,
        )

    assert not output_path.exists()
