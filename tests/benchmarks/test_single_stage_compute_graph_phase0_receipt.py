from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Literal

import pytest
from benchmarks.single_stage_compute_graph_attribution_control import (
    AttributionAttempt,
    AttributionBinding,
    build_attribution_evidence,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    A100_LANE_ID,
    FIRST_EVALUATION_LIMIT_NS,
    FORMAL_COMPLETE_PATH_FACTOR,
    HLO_MODULE_SET_IDENTITY_SOURCE,
    LANE_AGGREGATION_POLICY,
    PHASE0_SCHEMA_ID,
    RTX_LANE_ID,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    Phase0ReceiptError,
    _validate_command_buffer,
    _validate_profile,
    canonical_hlo_module_set_identity,
    canonical_json_bytes,
    canonical_sha256,
    load_phase0_receipt,
    validate_phase0_receipt,
    write_phase0_receipt,
)


def _digest(character: str) -> str:
    return character * 64


def _manifest() -> dict[str, object]:
    return {
        "schema_id": "single-stage-compute-graph-source-manifest-v1",
        "entries": [
            {
                "role": "execution_source",
                "relative_path": "src/simsopt_jax/solver.py",
                "size_bytes": 123,
                "sha256": _digest("1"),
            },
            {
                "role": "native_extension",
                "relative_path": "build/simsoptpp.so",
                "size_bytes": 456,
                "sha256": _digest("2"),
            },
        ],
    }


def _specimen() -> dict[str, object]:
    return {
        "specimen_id": "native-default-changed-state-0",
        "input_bundle_sha256": _digest("3"),
        "parameter_sha256": _digest("4"),
        "state_dimension": 255,
        "coil_dof_count": 461,
        "grids": {
            "inner_surface_points": 169,
            "non_qs_surface_points": 1600,
            "quadrature_nodes": 250,
            "physical_coil_contributions": 18,
        },
        "weights": {"iota": 1.0, "non_qs": 0.5},
        "tolerances": {"newton": 1e-12, "adjoint": 1e-10},
        "solver_graph_id": "exact-boozer-newton-direct-adjoint",
        "solver_graph_sha256": _digest("5"),
    }


def _qualification(lane_id: str, *, blocked: bool = False) -> dict[str, object]:
    common = {
        "source_snapshot",
        "import_bindings",
        "native_extension",
        "device_identity",
        "runtime_backend",
        "fp64_policy",
        "cpu_affinity",
        "strict_transfer_smoke",
    }
    if lane_id == A100_LANE_ID:
        common |= {
            "slurm_allocation",
            "cuda_12_6_compatibility",
            "dependency_overlay",
            "resolved_cuda_libraries",
        }
    ordered = sorted(common)
    failed_check = "slurm_allocation"
    checks = [
        {
            "check_id": check_id,
            "passed": not (blocked and check_id == failed_check),
            "evidence": f"evidence for {check_id}",
        }
        for check_id in ordered
    ]
    return {
        "outcome": "blocked" if blocked else "qualified",
        "attempted_identity": {
            "hostname": "landau" if lane_id == A100_LANE_ID else "playstation",
            "requested_device": lane_id,
            "allocation_id": "123" if lane_id == A100_LANE_ID else "local",
        },
        "checks": checks,
        "blocker": (
            {
                "code": "SLURM_ALLOCATION_UNAVAILABLE",
                "check_id": failed_check,
                "reason": "no current allocation",
                "evidence_sha256": _digest("6"),
            }
            if blocked
            else None
        ),
    }


def _provenance(lane_id: str) -> dict[str, object]:
    manifest = _manifest()
    is_a100 = lane_id == A100_LANE_ID
    return {
        "repository_commit": _digest("7"),
        "source_state_sha256": _digest("8"),
        "git_status_short": [" M src/simsopt_jax/solver.py"],
        "tracked_diff_sha256": _digest("9"),
        "untracked_manifest_sha256": _digest("a"),
        "immutable_root": f"/immutable/{lane_id}",
        "immutable_tree_sha256": _digest("b"),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "interpreter_path": f"/immutable/{lane_id}/.venv/bin/python",
        "runtime": {
            "python_version": "3.13.5",
            "jax_version": "0.10.2",
            "jaxlib_version": "0.10.2",
            "cuda_runtime": "12.6",
            "cuda_driver": "570.0" if not is_a100 else "470.0",
            "jax_backend": "gpu",
            "fp64_x64_enabled": True,
            "resolved_cuda_libraries": ["libcuda.so.1", "libcudart.so.12"],
        },
        "allocation": {
            "hostname": "landau" if is_a100 else "playstation",
            "scheduler": "slurm" if is_a100 else "local",
            "allocation_id": "allocation-123" if is_a100 else "local",
            "job_id": "job-456" if is_a100 else "local",
            "gpu_name": "NVIDIA A100-PCIE-40GB"
            if is_a100
            else "NVIDIA GeForce RTX 5090",
            "gpu_uuid": "GPU-a100" if is_a100 else "GPU-rtx5090",
            "gpu_memory_bytes": 40_000_000_000 if is_a100 else 32_000_000_000,
            "cpu_affinity": "0-31",
            "cuda_compatibility_version": "12.6" if is_a100 else "native",
            "cuda_compatibility_path": "/cuda-compat-12.6"
            if is_a100
            else "not-applicable",
        },
        "import_bindings": {
            package: {
                "path": f"/immutable/{lane_id}/{package}",
                "sha256": _digest(character),
            }
            for package, character in (
                ("simsopt", "c"),
                ("simsopt_jax", "d"),
                ("simsopt_jax_adapters", "e"),
                ("simsoptpp", "f"),
            )
        },
        "package_overlay": {"lineax": "0.1.1", "numpy": "2.3.0"},
        "environment": {
            "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
        },
        "policies": {
            "dense_batch_width": 8,
            "point_chunk_size": None,
            "coil_chunk_size": None,
            "quadrature_block_sizes": [128, 122],
        },
        "compilation_cache_directory": f"/cache/{lane_id}",
    }


def _first_evaluation() -> dict[str, object]:
    gradient = [float(index) / 1000.0 for index in range(461)]
    return {
        "variant": "C0",
        "wall_time_limit_ns": FIRST_EVALUATION_LIMIT_NS,
        "elapsed_ns": 5_000_000_000,
        "completed": True,
        "objective_dtype": "float64",
        "objective": 1.25,
        "gradient_dtype": "float64",
        "gradient": gradient,
        "native_objective": 1.25,
        "native_gradient": list(gradient),
        "objective_atol": 1e-12,
        "objective_rtol": 1e-10,
        "gradient_atol": 1e-12,
        "gradient_rtol": 1e-10,
        "inner_newton_success": True,
        "adjoint_success": True,
        "residual_certificates": {
            "boozer_residual": 1e-13,
            "adjoint_residual": 1e-12,
        },
    }


def _warm_measurement() -> dict[str, object]:
    wall_times = list(range(91, 101))
    return {
        "samples": [
            {
                "sample_index": index,
                "wall_ns": wall_ns,
                "peak_process_tree_rss_bytes": 1_000_000 + index,
                "sampled_process_gpu_memory_peak_bytes": 2_000_000 + index,
                "sampled_process_gpu_memory_source": (
                    SAMPLED_PROCESS_GPU_MEMORY_SOURCE
                ),
                "profiled": False,
            }
            for index, wall_ns in enumerate(wall_times)
        ],
        "p50_ns": 95.5,
        "p95_ns": 100.0,
    }


def _profile() -> dict[str, object]:
    return {
        "evaluation_envelope_ns": 1100,
        "device_active_ns": 1000,
        "phase_interval_unions": [
            {"phase_id": "newton.residual_jvp", "intervals": [[0, 500]]},
            {"phase_id": "adjoint.dense_matrix", "intervals": [[500, 700]]},
            {"phase_id": "biotsavart.vjp", "intervals": [[700, 920]]},
        ],
        "attributed_union_ns": 920,
        "unattributed_ns": 80,
        "attribution_coverage": 0.92,
        "pjrt_execute_count": 11,
        "kernel_launch_count": 40,
        "kernel_duration_ns": [10, 20, 30],
        "inter_launch_gap_ns": 25,
        "hlo_module_set_identity": canonical_hlo_module_set_identity(("jit-c0",)),
        "hlo_module_set_identity_source": HLO_MODULE_SET_IDENTITY_SOURCE,
    }


def _attribution_control(
    lane_id: str, specimen_sha: str, *, direct_high_coverage: bool = True
) -> dict[str, object]:
    provenance = _provenance(lane_id)
    specimen = _specimen()
    allocation = provenance["allocation"]
    assert isinstance(allocation, dict)
    binding = AttributionBinding(
        candidate_sha256=str(specimen["parameter_sha256"]),
        specimen_sha256=specimen_sha,
        input_bundle_sha256=str(specimen["input_bundle_sha256"]),
        source_sha256=str(provenance["source_state_sha256"]),
        production_runtime_identity_sha256="1" * 64,
        lane_id=lane_id,
        gpu_uuid=str(allocation["gpu_uuid"]),
        gate_checkpoint_sha256=_digest("d"),
        warm_checkpoint_sha256=_digest("e"),
        warm_p50_ns=95.5,
    )

    def attempt(
        mode: Literal["default_control", "command_buffer_disabled"], index: int
    ) -> AttributionAttempt:
        disabled = mode == "command_buffer_disabled"
        return AttributionAttempt(
            mode=mode,
            attempt_index=index,
            binding=binding,
            runtime_identity_sha256=("2" if disabled else "1") * 64,
            xla_flag_tokens=("--xla_gpu_enable_command_buffer=",) if disabled else (),
            compilation_cache_root=f"/cache/attribution/{lane_id}/{mode}/{index}",
            artifact_root=f"/artifacts/attribution/{lane_id}/{mode}/{index}",
            raw_trace_path=f"post-gate/{lane_id}/{mode}/{index}/trace.gz",
            raw_trace_sha256="3" * 64,
            child_observation_path=f"post-gate/{lane_id}/{mode}/{index}/child.json",
            child_observation_sha256="4" * 64,
            hlo_anchor_path=f"post-gate/{lane_id}/{mode}/{index}/anchor.json",
            hlo_anchor_sha256="5" * 64,
            profile_derivation_version="compute-graph-profile-attribution-v1",
            objective=1.0,
            gradient=(1.0,),
            solve_certificate={
                "inner_newton_success": True,
                "adjoint_success": True,
                "residual_certificates": {"adjoint_residual_relative": 1e-15},
            },
            module_topology_identity_sha256="f" * 64,
            evaluation_envelope_ns=1100,
            device_active_ns=1000,
            phase_device_ns=(
                (
                    "newton.residual_jvp",
                    500 if disabled or direct_high_coverage else 400,
                ),
                (
                    "adjoint.dense_matrix",
                    200 if disabled or direct_high_coverage else 180,
                ),
                ("biotsavart.vjp", 220 if disabled or direct_high_coverage else 200),
            ),
        )

    return build_attribution_evidence(
        tuple(attempt("default_control", index) for index in range(3)),
        tuple(attempt("command_buffer_disabled", index) for index in range(3)),
    )


def _gap_budget() -> dict[str, object]:
    newton_share = 500 / 1100
    dense_share = 200 / 1100
    biotsavart_share = 220 / 1100
    unattributed_share = 180 / 1100
    conservative_saving = newton_share * 0.1 + dense_share * 0.1
    optimistic_saving = (
        newton_share * 0.2 + dense_share * 0.2 + unattributed_share * 0.5
    )
    candidate_c0 = 95.5
    complete_c0 = 955.0
    evaluation_count = 5
    candidate_conservative = candidate_c0 * (1.0 - conservative_saving)
    candidate_optimistic = candidate_c0 * (1.0 - optimistic_saving)
    return {
        "candidate_value_and_gradient_reference_timings_ns": {
            "c0_warm_p50": candidate_c0
        },
        "matched_complete_path_reference_timings_ns": {
            "native": 800.0,
            "c0": complete_c0,
            "optax": 900.0,
        },
        "c0_complete_path_value_and_gradient_evaluation_count": evaluation_count,
        "c0_complete_path_value_and_gradient_evaluation_count_semantics": (
            "scipy_optimize_result_nfev_for_combined_objective_and_gradient_callable_"
            "within_complete_path_boundary"
        ),
        "formal_target_factor": FORMAL_COMPLETE_PATH_FACTOR,
        "formal_target_ns": FORMAL_COMPLETE_PATH_FACTOR * 800.0,
        "projection_method": (
            "candidate_value_and_gradient_savings_subtracted_from_matched_c0_complete_path"
        ),
        "candidate_phases": [
            {
                "phase_id": "newton.residual_jvp",
                "measured_share": newton_share,
                "conservative_reduction": 0.1,
                "optimistic_reduction": 0.2,
                "overlap_disposition": "disjoint",
            },
            {
                "phase_id": "adjoint.dense_matrix",
                "measured_share": dense_share,
                "conservative_reduction": 0.1,
                "optimistic_reduction": 0.2,
                "overlap_disposition": "disjoint",
            },
            {
                "phase_id": "biotsavart.vjp",
                "measured_share": biotsavart_share,
                "conservative_reduction": 0.0,
                "optimistic_reduction": 0.0,
                "overlap_disposition": "disjoint",
            },
        ],
        "unattributed_share": unattributed_share,
        "unattributed_conservative_reduction": 0.0,
        "unattributed_optimistic_reduction": 0.5,
        "candidate_value_and_gradient_conservative_projected_ns": (
            candidate_conservative
        ),
        "candidate_value_and_gradient_optimistic_projected_ns": candidate_optimistic,
        "conservative_complete_path_projected_ns": (
            complete_c0 - evaluation_count * (candidate_c0 - candidate_conservative)
        ),
        "optimistic_complete_path_projected_ns": (
            complete_c0 - evaluation_count * (candidate_c0 - candidate_optimistic)
        ),
        "faithful_levers": [
            {
                "lever_id": "dense_newton",
                "disposition": "bounded",
                "evidence_sha256": _digest("1"),
            },
            {
                "lever_id": "scalar_pullback",
                "disposition": "stopped",
                "evidence_sha256": _digest("2"),
            },
        ],
        "all_faithful_levers_bounded": True,
        "target_reachable_optimistically": False,
        "reduction_evidence_kind": "theoretical_policy_assumptions",
        "claim_ceiling": "DIAGNOSTIC_ONLY_NO_PIVOT_AUTHORITY",
        "empirical_canary_bindings": [],
        "pivot_fired": False,
    }


def _measurement(lane_id: str, specimen_sha: str) -> dict[str, object]:
    return {
        "variant": "C0",
        "specimen_sha256": specimen_sha,
        "provenance": _provenance(lane_id),
        "first_evaluation_gate": _first_evaluation(),
        "cold_compile": {
            "wall_ns": 5_000_000,
            "peak_process_tree_rss_bytes": 1_000_000,
            "sampled_process_gpu_memory_peak_bytes": 2_000_000,
            "sampled_process_gpu_memory_source": (SAMPLED_PROCESS_GPU_MEMORY_SOURCE),
            "hlo_module_set_identity": canonical_hlo_module_set_identity(("jit-c0",)),
            "hlo_module_set_identity_source": HLO_MODULE_SET_IDENTITY_SOURCE,
        },
        "warm_measurement": _warm_measurement(),
        "profile": _profile(),
        "attribution_control": _attribution_control(lane_id, specimen_sha),
        "command_buffer": {
            "resolved_configuration": "runtime-default",
            "observed_capture_participation": True,
            "graph_launched_device_ns": 700,
            "uncaptured_device_ns": 300,
            "captured_launch_count": 3,
            "uncaptured_launch_count": 8,
            "ab_control": {
                "control_id": "command-buffer-disabled",
                "resolved_configuration": "disabled",
                "sample_wall_ns": [105, 106],
                "included_in_promotion_timing": False,
            },
        },
        "newton_telemetry": {
            "telemetry_schema_id": "single-stage-compute-graph-newton-telemetry-v2",
            "route_id": "production-exact-newton",
            "measurement_method": "device_resident_fixed_shape_exact_newton_counts",
            "host_callback_used": False,
            "raw_evidence_sha256": "7" * 64,
            "raw_evidence_relative_path": (
                f"raw-telemetry/{lane_id}/newton-telemetry.json"
            ),
            "raw_evidence_file_sha256": "8" * 64,
            "residual_evaluations": 515,
            "linear_operator_applications": 513,
            "observed_wall_ns": 110,
            "unobserved_wall_ns": 100,
            "observer_effect_ratio": 1.1,
            "collected_outside_timed_samples": True,
        },
        "gap_budget": _gap_budget(),
    }


def test_command_buffer_accepts_unset_default_xla_flags() -> None:
    command_buffer = _measurement("rtx5090", _digest("a"))["command_buffer"]
    assert isinstance(command_buffer, dict)
    command_buffer["resolved_configuration"] = ""

    _validate_command_buffer(command_buffer, "command_buffer")

    command_buffer["resolved_configuration"] = None
    with pytest.raises(Phase0ReceiptError, match="must be a string"):
        _validate_command_buffer(command_buffer, "command_buffer")


def _receipt(*, a100_blocked: bool = True) -> dict[str, object]:
    specimen = _specimen()
    specimen_sha = canonical_sha256(specimen)
    return {
        "schema_id": PHASE0_SCHEMA_ID,
        "artifact_id": "compute-graph-phase0-test",
        "evidence_kind": "compute_graph_engineering_phase0_noncampaign",
        "lane_aggregation_policy": LANE_AGGREGATION_POLICY,
        "specimen": specimen,
        "specimen_sha256": specimen_sha,
        "lanes": [
            {
                "lane_id": RTX_LANE_ID,
                "device_class": "NVIDIA GeForce RTX 5090",
                "qualification": _qualification(RTX_LANE_ID),
                "measurement": _measurement(RTX_LANE_ID, specimen_sha),
            },
            {
                "lane_id": A100_LANE_ID,
                "device_class": "NVIDIA A100",
                "qualification": _qualification(A100_LANE_ID, blocked=a100_blocked),
                "measurement": (
                    None if a100_blocked else _measurement(A100_LANE_ID, specimen_sha)
                ),
            },
        ],
    }


def _bind_raw_telemetry_artifact(
    artifact_root: Path, document: dict[str, object], lane_id: str = RTX_LANE_ID
) -> Path:
    measurement = _measured_lane(document, lane_id)["measurement"]
    telemetry = measurement["newton_telemetry"]
    attribution = measurement["attribution_control"]
    binding = attribution["production_binding"]
    raw_fields = {
        key: value
        for key, value in telemetry.items()
        if key not in {"raw_evidence_relative_path", "raw_evidence_file_sha256"}
    }
    raw_fields.pop("raw_evidence_sha256")
    raw_document = {
        "schema_id": "single-stage-compute-graph-newton-telemetry-v2",
        "state": "PRODUCED",
        "evidence_kind": "observer_bearing_exact_newton_outside_promotion_timing",
        "identity": {
            "candidate_sha256": binding["candidate_sha256"],
            "specimen_sha256": binding["specimen_sha256"],
            "input_bundle_sha256": binding["input_bundle_sha256"],
            "source_sha256": binding["source_sha256"],
            "runtime_identity_sha256": binding["production_runtime_identity_sha256"],
            "lane_id": binding["lane_id"],
            "gpu_uuid": binding["gpu_uuid"],
            "gate_checkpoint_sha256": binding["gate_checkpoint_sha256"],
            "warm_checkpoint_sha256": binding["warm_checkpoint_sha256"],
            "warm_p50_ns": binding["warm_p50_ns"],
        },
        "route_id": "production-exact-newton",
        "warmup_executions_per_lane": 1,
        "numerical_equality": {
            "objective_exact": True,
            "raw_objective_exact": True,
            "gradient_exact": True,
            "solved_state_exact": True,
            "newton_success_exact": True,
            "newton_iterations_exact": True,
        },
        "observer": {
            "api": "device_resident_fixed_shape_exact_newton_counts",
            "device_resident_fixed_shape_counts": True,
            "host_callback_used": False,
            "promotion_timing_included": False,
        },
        "newton_telemetry": raw_fields,
    }
    raw_fields["raw_evidence_sha256"] = canonical_sha256(raw_document)
    telemetry["raw_evidence_sha256"] = raw_fields["raw_evidence_sha256"]
    relative_path = Path(telemetry["raw_evidence_relative_path"])
    artifact_path = artifact_root / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = canonical_json_bytes(raw_document)
    artifact_path.write_bytes(raw_bytes)
    telemetry["raw_evidence_file_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    return artifact_path


def _measured_lane(document: dict[str, object], lane_id: str) -> dict[str, object]:
    lanes = document["lanes"]
    assert isinstance(lanes, list)
    return next(lane for lane in lanes if lane["lane_id"] == lane_id)


def test_valid_receipt_allows_machine_readable_blocked_a100_without_timing() -> None:
    audit = validate_phase0_receipt(_receipt())

    assert audit.rtx.outcome == "qualified"
    assert audit.rtx.device_uuid == "GPU-rtx5090"
    assert audit.rtx.pivot_fired is False
    assert audit.a100.outcome == "blocked"
    assert audit.a100.warm_p50_ns is None


def test_provenance_accepts_real_sha1_git_oid_and_rejects_non_oid() -> None:
    document = _receipt()
    provenance = _measured_lane(document, RTX_LANE_ID)["measurement"]["provenance"]
    provenance["repository_commit"] = "a" * 40
    validate_phase0_receipt(document)

    provenance["repository_commit"] = "not-a-git-object-id"
    with pytest.raises(Phase0ReceiptError, match="Git object ID"):
        validate_phase0_receipt(document)


def test_valid_dual_gpu_receipt_keeps_lanes_separate() -> None:
    audit = validate_phase0_receipt(_receipt(a100_blocked=False))

    assert audit.rtx.device_uuid == "GPU-rtx5090"
    assert audit.a100.device_uuid == "GPU-a100"
    assert audit.rtx.warm_p50_ns == audit.a100.warm_p50_ns == 95.5


def test_writer_publishes_canonical_bytes_exclusively_and_loader_revalidates(
    tmp_path: Path,
) -> None:
    document = _receipt()
    path = tmp_path / "phase0.json"
    _bind_raw_telemetry_artifact(tmp_path, document)

    written = write_phase0_receipt(path, document)
    loaded, audited = load_phase0_receipt(path)

    assert path.read_bytes() == canonical_json_bytes(document)
    assert loaded == document
    assert audited == written
    with pytest.raises(FileExistsError):
        write_phase0_receipt(path, document)


def test_standalone_receipt_rejects_well_formed_wrong_telemetry_digest(
    tmp_path: Path,
) -> None:
    document = _receipt()
    _bind_raw_telemetry_artifact(tmp_path, document)
    telemetry = _measured_lane(document, RTX_LANE_ID)["measurement"]["newton_telemetry"]
    telemetry["raw_evidence_sha256"] = "0" * 64

    with pytest.raises(Phase0ReceiptError, match="fields differ from bound"):
        write_phase0_receipt(tmp_path / "phase0.json", document)


def test_standalone_receipt_rejects_stale_raw_telemetry_self_digest(
    tmp_path: Path,
) -> None:
    document = _receipt()
    artifact_path = _bind_raw_telemetry_artifact(tmp_path, document)
    raw_document = json.loads(artifact_path.read_text(encoding="utf-8"))
    raw_telemetry = raw_document["newton_telemetry"]
    raw_telemetry["observed_wall_ns"] = 120
    raw_telemetry["observer_effect_ratio"] = 1.2
    raw_bytes = canonical_json_bytes(raw_document)
    artifact_path.write_bytes(raw_bytes)
    telemetry = _measured_lane(document, RTX_LANE_ID)["measurement"]["newton_telemetry"]
    telemetry["raw_evidence_file_sha256"] = hashlib.sha256(raw_bytes).hexdigest()

    with pytest.raises(Phase0ReceiptError, match="raw evidence digest"):
        write_phase0_receipt(tmp_path / "phase0.json", document)


def test_standalone_receipt_rejects_raw_telemetry_byte_tamper(tmp_path: Path) -> None:
    document = _receipt()
    artifact_path = _bind_raw_telemetry_artifact(tmp_path, document)
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")

    with pytest.raises(Phase0ReceiptError, match="byte digest differs"):
        write_phase0_receipt(tmp_path / "phase0.json", document)


@pytest.mark.parametrize(
    "relative_path", ("../newton-telemetry.json", "/tmp/newton-telemetry.json")
)
def test_standalone_receipt_rejects_unsafe_raw_telemetry_path(
    tmp_path: Path, relative_path: str
) -> None:
    document = _receipt()
    telemetry = _measured_lane(document, RTX_LANE_ID)["measurement"]["newton_telemetry"]
    telemetry["raw_evidence_relative_path"] = relative_path

    with pytest.raises(Phase0ReceiptError, match="canonical safe relative path"):
        write_phase0_receipt(tmp_path / "phase0.json", document)


def test_standalone_receipt_rejects_missing_raw_telemetry(tmp_path: Path) -> None:
    document = _receipt()

    with pytest.raises(Phase0ReceiptError, match="raw telemetry artifact is missing"):
        write_phase0_receipt(tmp_path / "phase0.json", document)


def test_relocated_artifact_tree_retains_raw_telemetry_binding(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    document = _receipt()
    _bind_raw_telemetry_artifact(source_root, document)
    write_phase0_receipt(source_root / "phase0.json", document)
    relocated_root = tmp_path / "relocated"
    shutil.copytree(source_root, relocated_root)

    loaded, audit = load_phase0_receipt(relocated_root / "phase0.json")

    assert loaded == document
    assert audit.rtx.outcome == "qualified"


def test_unknown_schema_field_fails_closed() -> None:
    document = _receipt()
    document["unexpected"] = True

    with pytest.raises(Phase0ReceiptError, match=r"extra=\['unexpected'\]"):
        validate_phase0_receipt(document)


def test_frozen_specimen_hash_mismatch_fails() -> None:
    document = _receipt()
    specimen = document["specimen"]
    assert isinstance(specimen, dict)
    specimen["parameter_sha256"] = _digest("0")

    with pytest.raises(Phase0ReceiptError, match="specimen_sha256"):
        validate_phase0_receipt(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("elapsed_ns", FIRST_EVALUATION_LIMIT_NS + 1, "did not complete"),
        ("gradient_dtype", "float32", "float64"),
        ("inner_newton_success", False, "inner Newton solve failed"),
        ("adjoint_success", False, "adjoint solve failed"),
    ],
)
def test_first_evaluation_gate_blocks_timing_on_failure(
    field: str, value: object, message: str
) -> None:
    document = _receipt()
    gate = _measured_lane(document, RTX_LANE_ID)["measurement"]["first_evaluation_gate"]
    gate[field] = value

    with pytest.raises(Phase0ReceiptError, match=message):
        validate_phase0_receipt(document)


def test_first_evaluation_requires_exactly_461_finite_gradient_entries() -> None:
    document = _receipt()
    gate = _measured_lane(document, RTX_LANE_ID)["measurement"]["first_evaluation_gate"]
    gate["gradient"] = [0.0] * 460

    with pytest.raises(Phase0ReceiptError, match="exactly 461"):
        validate_phase0_receipt(document)

    gate["gradient"] = [0.0] * 460 + [float("nan")]
    with pytest.raises(Phase0ReceiptError, match="must be finite"):
        validate_phase0_receipt(document)


def test_first_evaluation_recomputes_native_gradient_parity() -> None:
    document = _receipt()
    gate = _measured_lane(document, RTX_LANE_ID)["measurement"]["first_evaluation_gate"]
    gate["gradient"][17] += 1.0

    with pytest.raises(Phase0ReceiptError, match="gradient parity failed at index 17"):
        validate_phase0_receipt(document)


def test_warm_samples_require_ten_unprofiled_samples_and_recomputed_quantiles() -> None:
    document = _receipt()
    warm = _measured_lane(document, RTX_LANE_ID)["measurement"]["warm_measurement"]
    warm["samples"] = warm["samples"][:-1]

    with pytest.raises(Phase0ReceiptError, match="at least 10"):
        validate_phase0_receipt(document)

    document = _receipt()
    warm = _measured_lane(document, RTX_LANE_ID)["measurement"]["warm_measurement"]
    warm["samples"][0]["profiled"] = True
    with pytest.raises(Phase0ReceiptError, match="must be unprofiled"):
        validate_phase0_receipt(document)

    document = _receipt()
    warm = _measured_lane(document, RTX_LANE_ID)["measurement"]["warm_measurement"]
    warm["p95_ns"] = 99.0
    with pytest.raises(Phase0ReceiptError, match="p95_ns"):
        validate_phase0_receipt(document)


def test_capture_sources_and_module_set_identity_fail_closed() -> None:
    document = _receipt()
    measurement = _measured_lane(document, RTX_LANE_ID)["measurement"]
    measurement["cold_compile"]["sampled_process_gpu_memory_source"] = "allocator"
    with pytest.raises(Phase0ReceiptError, match="must be one of"):
        validate_phase0_receipt(document)

    document = _receipt()
    measurement = _measured_lane(document, RTX_LANE_ID)["measurement"]
    measurement["profile"]["hlo_module_set_identity"] = (
        canonical_hlo_module_set_identity(("different-module",))
    )
    with pytest.raises(Phase0ReceiptError, match="differs from cold"):
        validate_phase0_receipt(document)


def test_profile_recomputes_interval_union_and_requires_90_percent_coverage() -> None:
    document = _receipt()
    profile = _measured_lane(document, RTX_LANE_ID)["measurement"]["profile"]
    profile["attributed_union_ns"] = 919

    with pytest.raises(Phase0ReceiptError, match="interval union"):
        validate_phase0_receipt(document)

    document = _receipt()
    profile = _measured_lane(document, RTX_LANE_ID)["measurement"]["profile"]
    profile["phase_interval_unions"][-1]["intervals"] = [[700, 890]]
    profile["attributed_union_ns"] = 890
    profile["unattributed_ns"] = 110
    profile["attribution_coverage"] = 0.89
    with pytest.raises(Phase0ReceiptError, match="less than 90%"):
        validate_phase0_receipt(document)


def test_command_buffer_requires_observed_not_configured_capture_evidence() -> None:
    document = _receipt()
    command_buffer = _measured_lane(document, RTX_LANE_ID)["measurement"][
        "command_buffer"
    ]
    command_buffer["observed_capture_participation"] = False

    with pytest.raises(Phase0ReceiptError, match="contradicts trace evidence"):
        validate_phase0_receipt(document)

    command_buffer["observed_capture_participation"] = True
    command_buffer["ab_control"]["included_in_promotion_timing"] = True
    with pytest.raises(Phase0ReceiptError, match="cannot contribute"):
        validate_phase0_receipt(document)


def test_attribution_control_refuses_non_promoting_or_incomplete_transfer() -> None:
    for field, value, message in (
        ("state", "NON_PROMOTING", "explicitly non-promoting"),
        ("promotion_eligible", False, "not promotion eligible"),
        (
            "blockers",
            ["module_topology_identity_mismatch"],
            "blockers must be empty",
        ),
        ("selected_attribution", None, "internally inconsistent"),
    ):
        document = _receipt()
        attribution = _measured_lane(document, RTX_LANE_ID)["measurement"][
            "attribution_control"
        ]
        attribution[field] = value
        with pytest.raises(Phase0ReceiptError, match=message):
            validate_phase0_receipt(document)


def test_attribution_control_revalidates_runtime_and_attempt_isolation() -> None:
    document = _receipt()
    attribution = _measured_lane(document, RTX_LANE_ID)["measurement"][
        "attribution_control"
    ]
    default_attempt = attribution["direct_default_measurement"]["attempts"][0]
    disabled_attempt = attribution["attribution_replay"]["attempts"][0]
    disabled_attempt["runtime_identity_sha256"] = default_attempt[
        "runtime_identity_sha256"
    ]
    with pytest.raises(Phase0ReceiptError, match="runtime identity"):
        validate_phase0_receipt(document)

    document = _receipt()
    attribution = _measured_lane(document, RTX_LANE_ID)["measurement"][
        "attribution_control"
    ]
    attempts = attribution["direct_default_measurement"]["attempts"]
    attempts[1]["compilation_cache_root"] = attempts[0]["compilation_cache_root"]
    with pytest.raises(Phase0ReceiptError, match="roots are not isolated"):
        validate_phase0_receipt(document)


def test_attribution_control_recomputes_numerical_equivalence() -> None:
    document = _receipt()
    attribution = _measured_lane(document, RTX_LANE_ID)["measurement"][
        "attribution_control"
    ]
    attempt = attribution["attribution_replay"]["attempts"][0]
    attempt["gradient"][0] += 1.0

    with pytest.raises(Phase0ReceiptError, match="attribution evidence validation"):
        validate_phase0_receipt(document)


@pytest.mark.parametrize("field", ["method", "share"])
def test_attribution_control_recomputes_direct_selection(field: str) -> None:
    document = _receipt()
    selection = _measured_lane(document, RTX_LANE_ID)["measurement"][
        "attribution_control"
    ]["selected_attribution"]
    if field == "method":
        selection["method"] = (
            "disabled-device-fraction-times-default-device-active-share"
        )
    else:
        selection["phase_shares"][0]["selected_default_envelope_share"] += 0.01

    with pytest.raises(Phase0ReceiptError, match="selected attribution differs"):
        validate_phase0_receipt(document)


def test_attribution_control_accepts_recomputed_fallback_selection() -> None:
    document = _receipt()
    lane = _measured_lane(document, RTX_LANE_ID)
    lane["measurement"]["attribution_control"] = _attribution_control(
        RTX_LANE_ID,
        str(document["specimen_sha256"]),
        direct_high_coverage=False,
    )

    validate_phase0_receipt(document)
    assert (
        lane["measurement"]["attribution_control"]["selected_attribution"]["route"]
        == "disabled_transfer_fallback"
    )


def test_profile_gap_share_uses_evaluation_envelope() -> None:
    profile = _profile()
    profile["inter_launch_gap_ns"] = 1_050

    _, gap_share, _, _ = _validate_profile(profile, "profile")

    assert gap_share == pytest.approx(1_050 / 1_100)
    assert gap_share <= 1.0


def test_gap_budget_recomputes_formal_target_projection_and_pivot() -> None:
    for field, value, message in (
        ("formal_target_ns", 73.0, "formal_target_ns"),
        (
            "optimistic_complete_path_projected_ns",
            1.0,
            "optimistic_complete_path_projected_ns",
        ),
        ("pivot_fired", True, "pivot_fired"),
    ):
        document = _receipt()
        budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
        budget[field] = value
        with pytest.raises(Phase0ReceiptError, match=message):
            validate_phase0_receipt(document)


def test_gap_budget_phase_shares_are_recomputed_against_wall_envelope() -> None:
    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["candidate_phases"][0]["measured_share"] = 0.5

    with pytest.raises(Phase0ReceiptError, match="measured_share"):
        validate_phase0_receipt(document)

    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["unattributed_share"] = 0.08
    with pytest.raises(Phase0ReceiptError, match="unattributed_share"):
        validate_phase0_receipt(document)


def test_gap_budget_rejects_candidate_and_complete_path_timing_conflation() -> None:
    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["matched_complete_path_reference_timings_ns"]["c0"] = 95.5
    with pytest.raises(Phase0ReceiptError, match="complete path must exceed"):
        validate_phase0_receipt(document)

    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["c0_complete_path_value_and_gradient_evaluation_count"] = 11
    with pytest.raises(Phase0ReceiptError, match="evaluation envelope exceeds"):
        validate_phase0_receipt(document)

    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["candidate_value_and_gradient_reference_timings_ns"]["c0_warm_p50"] = budget[
        "matched_complete_path_reference_timings_ns"
    ]["c0"]
    with pytest.raises(Phase0ReceiptError, match="c0_warm_p50"):
        validate_phase0_receipt(document)


def test_formal_target_uses_only_native_and_optax_complete_path_references() -> None:
    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["matched_complete_path_reference_timings_ns"]["c0"] = 10_000.0
    budget["conservative_complete_path_projected_ns"] += 9_045.0
    budget["optimistic_complete_path_projected_ns"] += 9_045.0

    validate_phase0_receipt(document)
    assert budget["formal_target_ns"] == FORMAL_COMPLETE_PATH_FACTOR * 800.0

    budget["formal_target_ns"] = FORMAL_COMPLETE_PATH_FACTOR * 95.5
    with pytest.raises(Phase0ReceiptError, match="formal_target_ns"):
        validate_phase0_receipt(document)


def test_unbounded_faithful_lever_prevents_pivot() -> None:
    document = _receipt()
    budget = _measured_lane(document, RTX_LANE_ID)["measurement"]["gap_budget"]
    budget["faithful_levers"][0]["disposition"] = "unbounded"

    with pytest.raises(Phase0ReceiptError, match="all_faithful_levers_bounded"):
        validate_phase0_receipt(document)


def test_blocked_a100_cannot_contribute_timing_claim() -> None:
    document = _receipt()
    a100 = _measured_lane(document, A100_LANE_ID)
    a100["measurement"] = _measurement(A100_LANE_ID, document["specimen_sha256"])

    with pytest.raises(Phase0ReceiptError, match="must contain no timing"):
        validate_phase0_receipt(document)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("telemetry_schema_id", "old-schema", "telemetry_schema_id"),
        ("route_id", "arbitrary-route", "route_id"),
        ("measurement_method", "host-counter", "measurement_method"),
        ("host_callback_used", True, "host callback"),
        ("raw_evidence_sha256", "0", "raw_evidence_sha256"),
        ("residual_evaluations", 513, "omit initial or candidate"),
    ),
)
def test_newton_telemetry_contract_tampering_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    document = _receipt()
    telemetry = _measured_lane(document, RTX_LANE_ID)["measurement"]["newton_telemetry"]
    telemetry[field] = value

    with pytest.raises(Phase0ReceiptError, match=message):
        validate_phase0_receipt(document)


def test_qualified_a100_requires_cuda_12_6_and_separate_cache() -> None:
    document = _receipt(a100_blocked=False)
    a100_measurement = _measured_lane(document, A100_LANE_ID)["measurement"]
    a100_measurement["provenance"]["allocation"]["cuda_compatibility_version"] = "12.8"

    with pytest.raises(Phase0ReceiptError, match="CUDA 12.6"):
        validate_phase0_receipt(document)

    document = _receipt(a100_blocked=False)
    rtx_cache = _measured_lane(document, RTX_LANE_ID)["measurement"]["provenance"][
        "compilation_cache_directory"
    ]
    _measured_lane(document, A100_LANE_ID)["measurement"]["provenance"][
        "compilation_cache_directory"
    ] = rtx_cache
    with pytest.raises(Phase0ReceiptError, match="caches must remain separate"):
        validate_phase0_receipt(document)


def test_dual_gpu_receipt_requires_same_immutable_source_state() -> None:
    document = _receipt(a100_blocked=False)
    measurement = _measured_lane(document, A100_LANE_ID)["measurement"]
    measurement["provenance"]["source_state_sha256"] = _digest("0")
    measurement["attribution_control"]["production_binding"]["source_sha256"] = _digest(
        "0"
    )

    with pytest.raises(Phase0ReceiptError, match="different source states"):
        validate_phase0_receipt(document)


def test_loader_rejects_duplicate_json_keys_and_nonfinite_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_id":"a","schema_id":"b"}', encoding="utf-8")
    with pytest.raises(Phase0ReceiptError, match="duplicate JSON key"):
        load_phase0_receipt(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(Phase0ReceiptError, match="non-finite JSON constant"):
        load_phase0_receipt(nonfinite)


def test_validation_does_not_mutate_caller_document() -> None:
    document = _receipt(a100_blocked=False)
    before = copy.deepcopy(document)

    validate_phase0_receipt(document)

    assert document == before


def test_canonical_json_is_stable_and_rejects_nonfinite_values() -> None:
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"value": float("inf")})
