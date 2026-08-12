from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.single_stage_compute_graph_c0_runner import _write_exclusive_json
from benchmarks.single_stage_compute_graph_canary_profile_runner import (
    PROFILE_SCHEMA_ID,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CANARY_ARTIFACT_SCHEMA_ID,
    CanarySpec,
    _spec_identity,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_promotion_finalizer import (
    PROMOTION_ARTIFACT_SCHEMA_ID,
    PromotionFinalizerError,
    _validate_canary_artifact,
    _validate_native_trajectory_receipt,
    _validate_profile_artifact,
    _validate_variant_trajectory_receipts,
    finalize_promotion,
)
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    TrajectoryOracleAudit,
    TrajectoryOracleError,
)


def _digest(character: str) -> str:
    return character * 64


def _spec(tmp_path: Path) -> CanarySpec:
    candidate = tmp_path / "candidate.npy"
    candidate.write_bytes(b"candidate")
    native = tmp_path / "native.json"
    native.write_text("{}", encoding="utf-8")
    return CanarySpec(
        variant="C1",
        solver_graph_sha256=_digest("1"),
        source_state_sha256=_digest("2"),
        specimen_sha256=_digest("3"),
        candidate_file_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        parameter_sha256=_digest("4"),
        device_identity_sha256=_digest("5"),
        gpu_uuid="GPU-test",
        c0_gate_checkpoint_sha256=_digest("6"),
        c0_warm_checkpoint_sha256=_digest("7"),
        native_reference_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        runtime_identity_sha256=_digest("8"),
        input_root=tmp_path,
        candidate_path=candidate,
        native_reference_path=native,
        snapshot_root=tmp_path,
        interpreter_path=Path(sys.executable),
        cache_directory=tmp_path / "cache",
        output_root=tmp_path / "canary-output",
        c0_p50_ns=200.0,
        c0_p95_ns=200.0,
        c0_peak_rss_bytes=20,
        c0_peak_gpu_memory_bytes=40,
        runtime_contract_json='{"static_environment":{}}',
    )


def _gate() -> dict[str, object]:
    return {
        "objective_dtype": "float64",
        "objective": 1.0,
        "gradient_dtype": "float64",
        "gradient": [0.0] * 461,
        "inner_newton_success": True,
        "adjoint_success": True,
        "residual_certificates": {"residual": 0.0},
    }


def _base_canary(spec: CanarySpec) -> dict[str, object]:
    return {
        "schema_id": CANARY_ARTIFACT_SCHEMA_ID,
        "status": "MEASURED_NONPROMOTING",
        "identity": _spec_identity(spec),
        "gate": _gate(),
        "c0_reference": {
            "p50_ns": 200.0,
            "p95_ns": 200.0,
            "peak_process_tree_rss_bytes": 20,
            "peak_gpu_memory_bytes": 40,
        },
        "warm_measurement": {
            "sample_count": 10,
            "wall_ns": list(range(100, 110)),
            "p50_wall_ns": 104.5,
            "p95_wall_ns": 109.0,
            "peak_process_tree_rss_bytes": 10,
            "process_tree_rss_sample_counts": [2] * 10,
            "process_tree_rss_roots": [
                {"pid": 100 + index, "starttime_ticks": 200 + index}
                for index in range(10)
            ],
            "peak_gpu_memory_bytes": 20,
        },
        "performance_gates": {
            "p50_at_least_20_percent_faster": True,
            "p95_at_most_10_percent_regression": True,
            "process_tree_rss_evidence_available": True,
            "peak_process_tree_rss_at_most_10_percent_regression": True,
            "peak_gpu_memory_at_most_10_percent_regression": True,
        },
        "performance_passed": True,
        "promotion_blocker": {
            "code": "PROMOTION_FINALIZER_REQUIRED",
            "reason": "only the finalizer promotes",
        },
    }


def _write_base(spec: CanarySpec, document: dict[str, object]) -> Path:
    spec.output_root.mkdir()
    path = spec.output_root / "canary.json"
    digest = _write_exclusive_json(path, document)
    (spec.output_root / "canary.sha256").write_text(digest + "\n", encoding="utf-8")
    return path


def _hashed_file(tmp_path: Path, name: str) -> dict[str, str]:
    path = tmp_path / name
    path.write_bytes(name.encode("utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _profile(tmp_path: Path, spec: CanarySpec, canary_sha256: str) -> dict[str, object]:
    version_probe_path = tmp_path / "raw-version-probe.json"
    _write_exclusive_json(
        version_probe_path,
        {
            "returncode": 0,
            "timed_out": False,
            "elapsed_ns": 1,
            "stdout": "2026.1\n",
            "stderr": "",
        },
    )
    raw_child_path = tmp_path / "raw-child.json"
    _write_exclusive_json(
        raw_child_path,
        {
            "returncode": 0,
            "timed_out": False,
            "elapsed_ns": 1,
            "stdout": "{}",
            "stderr": "",
        },
    )
    count_path = tmp_path / "profile-counts.json"
    _write_exclusive_json(count_path, {})
    return {
        "schema_id": PROFILE_SCHEMA_ID,
        "status": "PRODUCED",
        "promotion_timing": False,
        "profile_launch": {
            "command": ["nsys", "profile"],
            "command_sha256": _digest("c"),
            "environment_sha256": _digest("d"),
            "working_directory": str(tmp_path),
            "timeout_seconds": 900.0,
        },
        "identity": {
            **_spec_identity(spec),
            "canary_artifact_sha256": canary_sha256,
        },
        "numerical_revalidation": _gate(),
        "hlo_topology": {
            "lowered_hlo_ir_sha256": _digest("a"),
            "module_name_set_identity": _digest("b"),
            "module_names": ["jit_candidate"],
            "identity_ceiling": "lowered_canonical_HLO_IR_plus_executed_trace_module_name_set",
        },
        "launches": {
            "pjrt_execute_count": 3,
            "jax_kernel_launch_count": 4,
            "nsys_kernel_activity_count": 4,
            "nsys_cuda_graph_launch_api_count": 0,
            "nsys_uncaptured_kernel_activity_count": 4,
        },
        "required_operations": {
            "residual": {"count": 3, "device_interval_union_ns": 10},
            "jacobian_construction": {"count": 1, "device_interval_union_ns": 10},
            "dense_materialization": {"count": 1, "device_interval_union_ns": 10},
            "lu_factorization": {"count": 1, "device_interval_union_ns": 10},
            "refinement": {"count": 0, "device_interval_union_ns": 0},
            "linearized_tangent_traversals": {
                "primal_traversal_count": 1,
                "tangent_batch_count": 4,
                "tangent_direction_count": 255,
            },
        },
        "raw": {
            "version_probe": {
                "path": str(version_probe_path.resolve()),
                "sha256": hashlib.sha256(version_probe_path.read_bytes()).hexdigest(),
            },
            "raw_child": {
                "path": str(raw_child_path.resolve()),
                "sha256": hashlib.sha256(raw_child_path.read_bytes()).hexdigest(),
            },
            "profile_counts": {
                "path": str(count_path.resolve()),
                "sha256": hashlib.sha256(count_path.read_bytes()).hexdigest(),
            },
            "jax_trace": _hashed_file(tmp_path, "trace.json.gz"),
            "nsys_report": _hashed_file(tmp_path, "report.nsys-rep"),
            "nsys_sqlite": _hashed_file(tmp_path, "report.sqlite"),
        },
        "tool": {
            "nsys_binary_path": _hashed_file(tmp_path, "nsys")["path"],
            "nsys_binary_sha256": hashlib.sha256(b"nsys").hexdigest(),
            "nsys_version": "2026.1",
            "nvtx_library_path": _hashed_file(tmp_path, "nvtx.so")["path"],
            "nvtx_library_sha256": hashlib.sha256(b"nvtx.so").hexdigest(),
        },
        "missing_required_source_hooks": [],
    }


def _stub_profile_rebuild(
    monkeypatch: pytest.MonkeyPatch, rebuilt: dict[str, object]
) -> None:
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._profile_child",
        lambda _result: rebuilt["numerical_revalidation"],
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.validate_profile_count_evidence",
        lambda *_args, **_kwargs: {
            "residual_evaluation_count": 3,
            "dense_primal_traversal_count": 1,
            "dense_tangent_batch_count": 4,
            "dense_tangent_direction_count": 255,
        },
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.load_trace_document",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.summarize_compute_graph_profile",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.parse_nsys_sqlite",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.build_profile_launch",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.build_profile_artifact",
        lambda **_kwargs: rebuilt,
    )


def test_base_canary_rejects_direct_promotable_status_even_with_fresh_hash(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    artifact = _base_canary(spec)
    artifact["status"] = "MEASURED_PROMOTABLE"
    path = _write_base(spec, artifact)

    with pytest.raises(PromotionFinalizerError, match="schema, status, or identity"):
        _validate_canary_artifact(path, spec)


def test_base_canary_recomputes_timing_instead_of_trusting_pass_boolean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    artifact = _base_canary(spec)
    artifact["warm_measurement"]["p50_wall_ns"] = 1.0  # type: ignore[index]
    path = _write_base(spec, artifact)
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._load_restart",
        lambda _spec: ((), tuple({} for _ in range(12))),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.build_artifact",
        lambda _spec, _observations: artifact,
    )

    with pytest.raises(PromotionFinalizerError, match="arithmetic differs"):
        _validate_canary_artifact(path, spec)


def test_base_canary_must_equal_raw_child_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    artifact = _base_canary(spec)
    path = _write_base(spec, artifact)
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._load_restart",
        lambda _spec: ((), tuple({} for _ in range(12))),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.build_artifact",
        lambda _spec, _observations: {**artifact, "performance_passed": False},
    )

    with pytest.raises(PromotionFinalizerError, match="raw-recomputed"):
        _validate_canary_artifact(path, spec)


def test_profile_identity_is_bound_to_exact_base_canary_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    canary = _base_canary(spec)
    canary_path = _write_base(spec, canary)
    canary_sha256 = hashlib.sha256(canary_path.read_bytes()).hexdigest()
    profile = _profile(tmp_path, spec, canary_sha256)
    profile_path = tmp_path / "profile-evidence.json"
    _write_exclusive_json(profile_path, profile)
    _stub_profile_rebuild(monkeypatch, profile)

    assert (
        _validate_profile_artifact(
            profile_path,
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )
        == hashlib.sha256(profile_path.read_bytes()).hexdigest()
    )
    profile["identity"]["canary_artifact_sha256"] = _digest("f")  # type: ignore[index]
    tampered = tmp_path / "profile-tampered.json"
    _write_exclusive_json(tampered, profile)
    with pytest.raises(PromotionFinalizerError, match="identity-drifted"):
        _validate_profile_artifact(
            tampered,
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )


def test_profile_summary_must_equal_raw_trace_and_nsys_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    canary = _base_canary(spec)
    canary_path = _write_base(spec, canary)
    canary_sha256 = hashlib.sha256(canary_path.read_bytes()).hexdigest()
    profile = _profile(tmp_path, spec, canary_sha256)
    path = tmp_path / "profile-evidence.json"
    _write_exclusive_json(path, profile)
    launches = dict(profile["launches"])  # type: ignore[arg-type]
    launches["pjrt_execute_count"] = 9
    _stub_profile_rebuild(monkeypatch, {**profile, "launches": launches})

    with pytest.raises(PromotionFinalizerError, match="raw-recomputed"):
        _validate_profile_artifact(
            path,
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )


def test_profile_launch_tamper_differs_from_rebuilt_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    canary = _base_canary(spec)
    canary_path = _write_base(spec, canary)
    canary_sha256 = hashlib.sha256(canary_path.read_bytes()).hexdigest()
    clean = _profile(tmp_path, spec, canary_sha256)
    tampered = {
        **clean,
        "profile_launch": {**clean["profile_launch"], "command": ["fake"]},
    }  # type: ignore[arg-type]
    path = tmp_path / "profile-evidence.json"
    _write_exclusive_json(path, tampered)
    _stub_profile_rebuild(monkeypatch, clean)

    with pytest.raises(PromotionFinalizerError, match="raw-recomputed"):
        _validate_profile_artifact(
            path,
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )


def test_profile_version_probe_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    canary = _base_canary(spec)
    canary_path = _write_base(spec, canary)
    canary_sha256 = hashlib.sha256(canary_path.read_bytes()).hexdigest()
    profile = _profile(tmp_path, spec, canary_sha256)
    binding = profile["raw"]["version_probe"]  # type: ignore[index]
    version_path = Path(binding["path"])
    version_path.write_bytes(
        canonical_json_bytes(
            {
                "elapsed_ns": 1,
                "returncode": 0,
                "stderr": "",
                "stdout": "wrong\n",
                "timed_out": False,
            }
        )
    )
    binding["sha256"] = hashlib.sha256(version_path.read_bytes()).hexdigest()
    path = tmp_path / "profile-evidence.json"
    _write_exclusive_json(path, profile)
    _stub_profile_rebuild(monkeypatch, profile)

    with pytest.raises(PromotionFinalizerError, match="version-probe completion"):
        _validate_profile_artifact(
            path,
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )


def test_profile_missing_executed_count_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    canary = _base_canary(spec)
    canary_path = _write_base(spec, canary)
    canary_sha256 = hashlib.sha256(canary_path.read_bytes()).hexdigest()
    profile = _profile(tmp_path, spec, canary_sha256)
    profile["required_operations"]["residual"]["count"] = None  # type: ignore[index]
    path = tmp_path / "profile-evidence.json"
    _write_exclusive_json(path, profile)

    with pytest.raises(PromotionFinalizerError, match="residual.count"):
        _validate_profile_artifact(
            path,
            spec=spec,
            canary=canary,
            canary_sha256=canary_sha256,
        )


def test_finalizer_is_only_writer_of_promotable_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    canary = _base_canary(spec)
    canary_path = tmp_path / "base.json"
    profile_path = tmp_path / "profile.json"
    receipt_path = tmp_path / "receipt.json"
    oracle_path = tmp_path / "oracle.json"
    spec_path = tmp_path / "canary-spec.json"
    for path in (canary_path, receipt_path, oracle_path, spec_path):
        _write_exclusive_json(path, {})
    profile_count_path = tmp_path / "profile-counts-finalizer.json"
    _write_exclusive_json(profile_count_path, {})
    _write_exclusive_json(
        profile_path,
        {
            "raw": {
                "profile_counts": {
                    "path": str(profile_count_path),
                    "sha256": hashlib.sha256(
                        profile_count_path.read_bytes()
                    ).hexdigest(),
                }
            }
        },
    )
    c0_spec_path = tmp_path / "c0-spec.json"
    _write_exclusive_json(
        c0_spec_path,
        {"receipt_template": {"specimen": {"input_bundle_sha256": _digest("9")}}},
    )
    provenance_path = tmp_path / "provenance.json"
    _write_exclusive_json(provenance_path, {"environment": {}})
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_paths = tuple(raw_root / name for name in ("native.json", "c0.json", "c1.json"))
    for path in raw_paths:
        _write_exclusive_json(path, {})
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._load_canary_spec",
        lambda _path: (
            {
                "c0_spec_path": str(c0_spec_path),
                "runtime_provenance_path": str(provenance_path),
            },
            spec,
        ),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._validate_canary_artifact",
        lambda _path, _spec: (canary, _digest("a")),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._validate_c0_receipt",
        lambda **_kwargs: _digest("b"),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._validate_profile_artifact",
        lambda _path, **_kwargs: _digest("c"),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._validate_native_trajectory_receipt",
        lambda **_kwargs: _digest("d"),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._validate_variant_trajectory_receipts",
        lambda **_kwargs: {
            "C0": {"path": "c0", "sha256": _digest("e")},
            "C1": {"path": "c1", "sha256": _digest("f")},
        },
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.require_passing_variant_trajectory_oracle",
        lambda *_args, **_kwargs: TrajectoryOracleAudit(
            variant="C1",
            parameter_sha256=spec.parameter_sha256,
            parity_passed=True,
            one_step_passed=True,
            short_replay_passed=True,
            terminal_passed=True,
        ),
    )
    destination = tmp_path / "promotion.json"

    artifact = finalize_promotion(
        canary_spec_path=spec_path,
        base_canary_artifact_path=canary_path,
        profile_evidence_path=profile_path,
        trajectory_oracle_path=oracle_path,
        trajectory_artifact_root=raw_root,
        one_step_reference_raw_path=raw_paths[0],
        trajectory_reference_raw_path=raw_paths[1],
        variant_raw_path=raw_paths[2],
        native_trajectory_receipt_path=tmp_path / "native-receipt.json",
        c0_trajectory_receipt_path=tmp_path / "c0-receipt.json",
        variant_trajectory_receipt_path=tmp_path / "c1-receipt.json",
        c0_receipt_path=receipt_path,
        destination=destination,
    )

    assert artifact["schema_id"] == PROMOTION_ARTIFACT_SCHEMA_ID
    assert artifact["status"] == "MEASURED_PROMOTABLE"
    assert destination.is_file()


def test_failed_recomputed_oracle_writes_no_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._load_canary_spec",
        lambda _path: (_ for _ in ()).throw(TrajectoryOracleError("parity failed")),
    )
    destination = tmp_path / "promotion.json"

    with pytest.raises(PromotionFinalizerError, match="parity failed"):
        finalize_promotion(
            canary_spec_path=tmp_path / "spec.json",
            base_canary_artifact_path=tmp_path / "canary.json",
            profile_evidence_path=tmp_path / "profile.json",
            trajectory_oracle_path=tmp_path / "oracle.json",
            trajectory_artifact_root=tmp_path,
            one_step_reference_raw_path=tmp_path / "native.json",
            trajectory_reference_raw_path=tmp_path / "c0.json",
            variant_raw_path=tmp_path / "c1.json",
            native_trajectory_receipt_path=tmp_path / "native-receipt.json",
            c0_trajectory_receipt_path=tmp_path / "c0-receipt.json",
            variant_trajectory_receipt_path=tmp_path / "c1-receipt.json",
            c0_receipt_path=tmp_path / "receipt.json",
            destination=destination,
        )
    assert not destination.exists()


def test_c1_requires_runner_owned_c0_launch_receipt(tmp_path: Path) -> None:
    spec = _spec(tmp_path)

    with pytest.raises(PromotionFinalizerError, match="requires a C0 launch receipt"):
        _validate_variant_trajectory_receipts(
            c0_receipt_path=None,
            variant_receipt_path=tmp_path / "c1-receipt.json",
            trajectory_reference_raw_path=tmp_path / "c0.json",
            variant_raw_path=tmp_path / "c1.json",
            profile_count_path=tmp_path / "counts.json",
            artifact_root=tmp_path,
            canary_spec_path=tmp_path / "spec.json",
            base_canary_artifact_path=tmp_path / "canary.json",
            spec=spec,
        )


def test_c2_rejects_unused_c0_launch_receipt(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    object.__setattr__(spec, "variant", "C2")

    with pytest.raises(PromotionFinalizerError, match="must not inject unused C0"):
        _validate_variant_trajectory_receipts(
            c0_receipt_path=tmp_path / "c0-receipt.json",
            variant_receipt_path=tmp_path / "c2-receipt.json",
            trajectory_reference_raw_path=tmp_path / "native.json",
            variant_raw_path=tmp_path / "c2.json",
            profile_count_path=tmp_path / "counts.json",
            artifact_root=tmp_path,
            canary_spec_path=tmp_path / "spec.json",
            base_canary_artifact_path=tmp_path / "canary.json",
            spec=spec,
        )


def test_c2_finalizer_rejects_distinct_native_reference_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    object.__setattr__(spec, "variant", "C2")
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer._load_canary_spec",
        lambda _path: ({}, spec),
    )

    with pytest.raises(PromotionFinalizerError, match="same native raw file"):
        finalize_promotion(
            canary_spec_path=tmp_path / "spec.json",
            base_canary_artifact_path=tmp_path / "canary.json",
            profile_evidence_path=tmp_path / "profile.json",
            trajectory_oracle_path=tmp_path / "oracle.json",
            trajectory_artifact_root=tmp_path,
            one_step_reference_raw_path=tmp_path / "native-one-step.json",
            trajectory_reference_raw_path=tmp_path / "native-trajectory.json",
            variant_raw_path=tmp_path / "c2.json",
            native_trajectory_receipt_path=tmp_path / "native-receipt.json",
            c0_trajectory_receipt_path=None,
            variant_trajectory_receipt_path=tmp_path / "c2-receipt.json",
            c0_receipt_path=tmp_path / "phase0.json",
            destination=tmp_path / "promotion.json",
        )


def test_native_receipt_helper_uses_published_snapshot_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    object.__setattr__(spec, "runtime_contract_json", "{}")
    native_path = tmp_path / "snapshot-native" / "simsoptpp.so"
    native_path.parent.mkdir()
    native_path.write_bytes(b"snapshot-native")
    native_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    publication_path = tmp_path / "publication.json"
    _write_exclusive_json(
        publication_path,
        {
            "native_extension": {
                "relative_path": native_path.relative_to(tmp_path).as_posix(),
                "sha256": native_sha256,
            }
        },
    )
    input_bundle_path = tmp_path / "input_bundle.json"
    input_bundle_path.write_text("{}", encoding="utf-8")
    attestation_path = tmp_path / "attestation.json"
    graph_path = tmp_path / "graph.json"
    receipt_path = tmp_path / "native-receipt.json"
    for path in (attestation_path, graph_path, receipt_path):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "examples.jax.parity.input_bundle.read_input_bundle",
        lambda _path: (
            SimpleNamespace(
                input_fingerprint=_digest("a"),
                configuration_fingerprint=_digest("b"),
            ),
            {},
        ),
    )
    captured = {}

    def validate(receipt, launch, *, artifact_root):
        captured["receipt"] = receipt
        captured["launch"] = launch
        captured["artifact_root"] = artifact_root
        return {}

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.validate_native_trajectory_launch",
        validate,
    )

    _validate_native_trajectory_receipt(
        receipt_path=receipt_path,
        raw_path=tmp_path / "native-raw.json",
        artifact_root=tmp_path,
        canary_spec_document={
            "snapshot_publication_path": str(publication_path),
            "import_attestation_path": str(attestation_path),
            "variant_solver_graph_path": str(graph_path),
        },
        spec=spec,
    )

    launch = captured["launch"]
    assert launch.binding.native_simsoptpp_path == str(native_path.resolve())
    assert launch.binding.native_simsoptpp_sha256 == native_sha256
    assert Path(launch.binding.native_simsoptpp_path).is_relative_to(
        spec.snapshot_root.resolve()
    )


def test_variant_receipt_helper_constructs_c0_and_c1_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    c0_receipt = tmp_path / "c0-receipt.json"
    c1_receipt = tmp_path / "c1-receipt.json"
    for path in (c0_receipt, c1_receipt):
        path.write_text("{}", encoding="utf-8")
    launches = []

    def validate(receipt, launch, *, artifact_root):
        del receipt, artifact_root
        launches.append(launch)
        return {}

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_promotion_finalizer.validate_variant_trajectory_launch",
        validate,
    )

    receipts = _validate_variant_trajectory_receipts(
        c0_receipt_path=c0_receipt,
        variant_receipt_path=c1_receipt,
        trajectory_reference_raw_path=tmp_path / "c0.json",
        variant_raw_path=tmp_path / "c1.json",
        profile_count_path=tmp_path / "counts.json",
        artifact_root=tmp_path,
        canary_spec_path=tmp_path / "spec.json",
        base_canary_artifact_path=tmp_path / "canary.json",
        spec=spec,
    )

    assert [launch.lane for launch in launches] == ["C0", "C1"]
    assert launches[0].profile_count_output_path is None
    assert launches[1].profile_count_output_path == tmp_path / "counts.json"
    assert set(receipts) == {"C0", "C1"}
