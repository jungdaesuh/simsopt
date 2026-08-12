from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from benchmarks.single_stage_compute_graph_canary_profile_runner import (
    validate_profile_count_evidence,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CanarySpec,
    _spec_identity,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    validate_raw_trajectory_document,
)
from benchmarks.single_stage_compute_graph_variant_trajectory import (
    PROFILE_COUNT_FIELDS,
    VariantTrajectoryError,
    _c0_document,
    _c1_document,
    _c2_document,
    _ReplayInputs,
    profile_counts_from_oracle_result,
    write_profile_count_evidence,
)


def _quadratic_residual(values: jax.Array) -> jax.Array:
    return jnp.asarray(
        [
            values[0] ** 2 + 0.5 * values[1] - 2.0,
            -0.25 * values[0] + values[1] ** 2 - 3.0,
        ],
        dtype=values.dtype,
    )


def _cubic_residual(values: jax.Array) -> jax.Array:
    return values**3 - 1.0


def _identity_kwargs() -> dict[str, str]:
    return {
        "parameter_sha256": "1" * 64,
        "specimen_sha256": "2" * 64,
        "input_bundle_sha256": "3" * 64,
        "solver_graph_sha256": "4" * 64,
        "source_sha256": "5" * 64,
    }


def test_c1_source_oracle_produces_valid_raw_trajectory() -> None:
    replay = _ReplayInputs(
        residual_fn=_quadratic_residual,
        initial_state=jnp.asarray([1.25, 1.75], dtype=jnp.float64),
        maxiter=8,
        tolerance=1.0e-12,
    )

    document, result = _c1_document(replay, **_identity_kwargs())
    normalized = validate_raw_trajectory_document(document)

    assert normalized["lane"] == "C1"
    assert normalized["one_step"] is not None
    assert len(normalized["short_replay"]) >= 2
    assert normalized["terminal"]["success"] is True
    assert (
        profile_counts_from_oracle_result("C1", result)["dense_primal_traversal_count"]
        >= 2
    )


def test_c2_source_oracle_produces_valid_rollback_trajectory() -> None:
    replay = _ReplayInputs(
        residual_fn=_cubic_residual,
        initial_state=jnp.asarray([0.1], dtype=jnp.float64),
        maxiter=2,
        tolerance=1.0e-12,
    )

    document, result = _c2_document(replay, **_identity_kwargs())
    normalized = validate_raw_trajectory_document(document)

    assert normalized["lane"] == "C2"
    assert normalized["one_step"] is not None
    assert len(normalized["short_replay"]) == 2
    assert normalized["terminal"]["rollback_taken"] is True
    assert (
        profile_counts_from_oracle_result("C2", result)["dense_tangent_batch_count"] > 0
    )


def test_c0_operator_gmres_replay_has_no_fabricated_one_step() -> None:
    replay = _ReplayInputs(
        residual_fn=_quadratic_residual,
        initial_state=jnp.asarray([1.25, 1.75], dtype=jnp.float64),
        maxiter=8,
        tolerance=1.0e-12,
    )

    document, result = _c0_document(replay, **_identity_kwargs())
    normalized = validate_raw_trajectory_document(document)

    assert result is None
    assert normalized["lane"] == "C0"
    assert normalized["one_step"] is None
    assert len(normalized["short_replay"]) >= 2


def _spec(tmp_path: Path) -> CanarySpec:
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
        native_reference_sha256="9" * 64,
        runtime_identity_sha256="a" * 64,
        input_root=tmp_path / "inputs",
        candidate_path=tmp_path / "candidate.npy",
        native_reference_path=tmp_path / "native.json",
        snapshot_root=tmp_path / "snapshot",
        interpreter_path=tmp_path / "python",
        cache_directory=tmp_path / "cache",
        output_root=tmp_path / "output",
        snapshot_manifest_sha256="b" * 64,
        snapshot_publication_sha256="c" * 64,
        import_attestation_sha256="d" * 64,
        qualification_sha256="e" * 64,
        device_probe_sha256="f" * 64,
        runtime_provenance_sha256="0" * 64,
    )


def _write_artifact(path: Path, spec: CanarySpec) -> None:
    path.write_bytes(
        canonical_json_bytes(
            {
                "identity": _spec_identity(spec),
                "status": "MEASURED_NONPROMOTING",
            }
        )
    )


def test_profile_counts_bind_validated_spec_and_final_artifact_bytes(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    artifact_path = tmp_path / "canary.json"
    evidence_path = tmp_path / "counts.json"
    _write_artifact(artifact_path, spec)
    counts = {field: index + 1 for index, field in enumerate(PROFILE_COUNT_FIELDS)}

    digest = write_profile_count_evidence(
        evidence_path,
        spec=spec,
        canary_artifact_path=artifact_path,
        counts=counts,
    )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert digest == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert (
        document["identity"]["canary_artifact_sha256"]
        == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )
    assert (
        validate_profile_count_evidence(
            document,
            spec=spec,
            canary_artifact_sha256=hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
        )
        == counts
    )


def test_profile_count_writer_rejects_artifact_identity_drift(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    artifact_path = tmp_path / "canary.json"
    identity = _spec_identity(spec)
    identity["parameter_sha256"] = "f" * 64
    artifact_path.write_bytes(
        canonical_json_bytes({"identity": identity, "status": "MEASURED_PROMOTABLE"})
    )

    with pytest.raises(VariantTrajectoryError, match="identity differs"):
        write_profile_count_evidence(
            tmp_path / "counts.json",
            spec=spec,
            canary_artifact_path=artifact_path,
            counts={field: 1 for field in PROFILE_COUNT_FIELDS},
        )


def test_profile_count_writer_rejects_post_finalization_artifact(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    artifact_path = tmp_path / "canary.json"
    artifact_path.write_bytes(
        canonical_json_bytes(
            {
                "identity": _spec_identity(spec),
                "status": "MEASURED_PROMOTABLE",
            }
        )
    )

    with pytest.raises(VariantTrajectoryError, match="pre-finalization"):
        write_profile_count_evidence(
            tmp_path / "counts.json",
            spec=spec,
            canary_artifact_path=artifact_path,
            counts={field: 1 for field in PROFILE_COUNT_FIELDS},
        )


@dataclass(frozen=True)
class _FakeCounts:
    exact_newton_variant_residual_evaluation_count: int = 1
    exact_newton_variant_dense_primal_traversal_count: int = 2
    exact_newton_variant_dense_tangent_batch_count: int = 3
    exact_newton_variant_dense_tangent_direction_count: int = 4


def test_c0_cannot_masquerade_as_dense_profile_count_evidence() -> None:
    with pytest.raises(VariantTrajectoryError, match="does not expose"):
        profile_counts_from_oracle_result("C0", _FakeCounts())
