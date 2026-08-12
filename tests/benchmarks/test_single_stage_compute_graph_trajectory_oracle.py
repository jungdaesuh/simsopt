from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    RAW_TRAJECTORY_SCHEMA_ID,
    TrajectoryOracleError,
    TrajectoryOracleIdentity,
    TrajectoryRawBindings,
    bind_raw_trajectory_inputs,
    build_variant_trajectory_oracle,
    require_passing_variant_trajectory_oracle,
    validate_raw_trajectory_document,
    validate_variant_trajectory_oracle,
    write_raw_trajectory_document,
    write_variant_trajectory_oracle,
)


def _digest(character: str) -> str:
    return character * 64


def _identity(variant: str) -> TrajectoryOracleIdentity:
    return TrajectoryOracleIdentity(
        variant=variant,  # type: ignore[arg-type]
        parameter_sha256=_digest("1"),
        specimen_sha256=_digest("2"),
        input_bundle_sha256=_digest("3"),
        solver_graph_sha256=_digest("4"),
        one_step_reference_source_sha256=_digest("5"),
        trajectory_reference_source_sha256=(
            _digest("7") if variant == "C1" else _digest("5")
        ),
        variant_source_sha256=_digest("6"),
    )


def _one_step_counters(update_count: int) -> dict[str, int]:
    return {
        "residual_evaluation_count": update_count + 1,
        "dense_materialization_count": update_count,
        "lu_factorization_count": update_count,
        "lu_solve_count": update_count * 2,
        "refinement_correction_count": update_count,
    }


def _replay_counters(lane: str, update_count: int) -> dict[str, int]:
    if lane in ("C0", "C1"):
        return {
            "residual_evaluation_count": update_count + 1,
            "attempted_iteration_count": update_count,
            "accepted_update_count": update_count,
        }
    return {
        "residual_evaluation_count": update_count + 1,
        "attempted_iteration_count": update_count,
        "applied_update_count": update_count,
        "assessed_state_count": update_count + 1,
        "rollback_recompute_count": 0,
    }


def _raw(identity: TrajectoryOracleIdentity, lane: str) -> dict[str, object]:
    document = {
        "schema_id": RAW_TRAJECTORY_SCHEMA_ID,
        "lane": lane,
        "parameter_sha256": identity.parameter_sha256,
        "specimen_sha256": identity.specimen_sha256,
        "input_bundle_sha256": identity.input_bundle_sha256,
        "solver_graph_sha256": identity.solver_graph_sha256,
        "source_sha256": (
            identity.one_step_reference_source_sha256
            if lane == "native"
            else (
                identity.trajectory_reference_source_sha256
                if lane == "C0"
                else identity.variant_source_sha256
            )
        ),
        "one_step": {
            "initial_state": [1.0, 2.0],
            "residual": [2.0, 4.0],
            "jacobian": [[8.0, 0.0], [0.0, 8.0]],
            "initial_solve": [0.25, 0.5],
            "refinement_rhs": [0.0, 0.0],
            "refinement_correction": [0.0, 0.0],
            "correction_step": [0.25, 0.5],
            "refined_residual": [0.0, 0.0],
            "next_state": [0.75, 1.5],
            "converged": False,
            "numerical_failure": False,
            "status_code": 0,
            "counters": _one_step_counters(1),
        },
        "short_replay": [
            {
                "iteration_index": 0,
                "state_before": [1.0, 2.0],
                "update": [0.25, 0.5],
                "state_after": [0.75, 1.5],
                "merit_before": 10.0,
                "merit_after": 8.0,
                "state_assessed_after": True,
                "backtracking_iteration_count": (1 if lane in ("C0", "C1") else 0),
                "accepted": True,
                "stop_decision": False,
                "status_code": 0,
                "counters": _replay_counters(lane, 1),
            },
            {
                "iteration_index": 1,
                "state_before": [0.75, 1.5],
                "update": [0.125, 0.25],
                "state_after": [0.625, 1.25],
                "merit_before": 8.0,
                "merit_after": 7.0,
                "state_assessed_after": True,
                "backtracking_iteration_count": 0,
                "accepted": True,
                "stop_decision": True,
                "status_code": 0,
                "counters": _replay_counters(lane, 2),
            },
        ],
        "terminal": {
            "success": True,
            "persist_solved_state": True,
            "rollback_taken": False,
            "returned_state": [0.625, 1.25],
            "returned_residual": [0.1, 0.2],
            "returned_jacobian": [[8.0, 0.0], [0.0, 8.0]],
            "returned_norm": 0.22360679774997896,
            "status_code": 0,
            "counters": _replay_counters(lane, 2),
        },
    }
    if lane == "C0":
        document["one_step"] = None
    return document


def _produce(
    tmp_path: Path, variant: str
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    TrajectoryOracleIdentity,
    TrajectoryRawBindings,
]:
    identity = _identity(variant)
    one_step_reference_path = tmp_path / "raw" / "native-one-step.json"
    trajectory_reference_path = tmp_path / "raw" / "trajectory-reference.json"
    variant_path = tmp_path / "raw" / "variant.json"
    write_raw_trajectory_document(one_step_reference_path, _raw(identity, "native"))
    write_raw_trajectory_document(
        trajectory_reference_path,
        _raw(identity, identity.trajectory_reference_lane),
    )
    write_raw_trajectory_document(variant_path, _raw(identity, variant))
    bindings = bind_raw_trajectory_inputs(
        artifact_root=tmp_path,
        one_step_reference_raw_path=one_step_reference_path,
        trajectory_reference_raw_path=trajectory_reference_path,
        variant_raw_path=variant_path,
    )
    document = build_variant_trajectory_oracle(
        identity=identity,
        artifact_root=tmp_path,
        one_step_reference_raw_path=one_step_reference_path,
        trajectory_reference_raw_path=trajectory_reference_path,
        variant_raw_path=variant_path,
    )
    oracle_path = tmp_path / "trajectory-oracle.json"
    write_variant_trajectory_oracle(oracle_path, document)
    return (
        oracle_path,
        one_step_reference_path,
        trajectory_reference_path,
        variant_path,
        identity,
        bindings,
    )


@pytest.mark.parametrize("variant", ["C1", "C2"])
def test_raw_recomputation_produces_passing_variant_oracle(
    tmp_path: Path, variant: str
) -> None:
    oracle_path, _, _, _, identity, bindings = _produce(tmp_path, variant)

    audit = require_passing_variant_trajectory_oracle(
        oracle_path,
        artifact_root=tmp_path,
        expected_identity=identity,
        expected_raw_bindings=bindings,
    )

    assert audit.variant == variant
    assert audit.one_step_passed is True
    assert audit.short_replay_passed is True
    assert audit.parity_passed is True


def test_c1_decision_flip_is_nonpromoting_even_with_equal_endpoint(
    tmp_path: Path,
) -> None:
    _, one_step_path, reference_path, variant_path, identity, _ = _produce(
        tmp_path, "C1"
    )
    candidate = json.loads(variant_path.read_text(encoding="utf-8"))
    candidate["short_replay"][0]["backtracking_iteration_count"] = 2  # type: ignore[index]
    variant_path.write_bytes(canonical_json_bytes(candidate))
    bindings = bind_raw_trajectory_inputs(
        artifact_root=tmp_path,
        one_step_reference_raw_path=one_step_path,
        trajectory_reference_raw_path=reference_path,
        variant_raw_path=variant_path,
    )
    document = build_variant_trajectory_oracle(
        identity=identity,
        artifact_root=tmp_path,
        one_step_reference_raw_path=one_step_path,
        trajectory_reference_raw_path=reference_path,
        variant_raw_path=variant_path,
    )
    oracle_path = tmp_path / "oracle.json"
    write_variant_trajectory_oracle(oracle_path, document)

    audit = validate_variant_trajectory_oracle(
        oracle_path,
        artifact_root=tmp_path,
        expected_identity=identity,
        expected_raw_bindings=bindings,
    )
    assert audit.one_step_passed is True
    assert audit.short_replay_passed is False
    with pytest.raises(TrajectoryOracleError, match="valid but non-promoting"):
        require_passing_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )


def test_c0_rejected_unassessed_attempt_reuses_retained_merit() -> None:
    identity = _identity("C1")
    raw = _raw(identity, "C0")
    first = raw["short_replay"][0]  # type: ignore[index]
    second = raw["short_replay"][1]  # type: ignore[index]
    first["accepted"] = False
    first["state_after"] = first["state_before"]
    first["merit_after"] = None
    first["state_assessed_after"] = False
    first["counters"]["accepted_update_count"] = 0
    second["state_before"] = first["state_after"]
    second["state_after"] = [
        before - update
        for before, update in zip(second["state_before"], second["update"])
    ]
    second["merit_before"] = first["merit_before"]
    second["counters"]["accepted_update_count"] = 1
    raw["terminal"]["returned_state"] = second["state_after"]  # type: ignore[index]
    raw["terminal"]["counters"]["accepted_update_count"] = 1  # type: ignore[index]

    validate_raw_trajectory_document(raw)

    second["merit_before"] = 9.0
    with pytest.raises(TrajectoryOracleError, match="merit chain"):
        validate_raw_trajectory_document(raw)


@pytest.mark.parametrize("field", ["update", "status_code", "counters"])
def test_c2_native_update_stop_status_and_counters_are_authoritative(
    tmp_path: Path, field: str
) -> None:
    _, one_step_path, reference_path, variant_path, identity, _ = _produce(
        tmp_path, "C2"
    )
    candidate = json.loads(variant_path.read_text(encoding="utf-8"))
    step = candidate["short_replay"][0]  # type: ignore[index]
    if field == "update":
        step[field][0] = 9.0
        step["state_after"][0] = -8.0
        candidate["short_replay"][1]["state_before"][0] = -8.0
        candidate["short_replay"][1]["state_after"][0] = -8.125
        candidate["one_step"]["correction_step"][0] = 9.0
        candidate["one_step"]["initial_solve"][0] = 9.0
        candidate["one_step"]["next_state"][0] = -8.0
        candidate["terminal"]["returned_state"][0] = -8.125
    elif field == "status_code":
        step[field] = 1
    else:
        step[field]["applied_update_count"] = 2
    variant_path.write_bytes(canonical_json_bytes(candidate))
    if field in ("update", "counters"):
        with pytest.raises(TrajectoryOracleError):
            build_variant_trajectory_oracle(
                identity=identity,
                artifact_root=tmp_path,
                one_step_reference_raw_path=one_step_path,
                trajectory_reference_raw_path=reference_path,
                variant_raw_path=variant_path,
            )
        return
    document = build_variant_trajectory_oracle(
        identity=identity,
        artifact_root=tmp_path,
        one_step_reference_raw_path=one_step_path,
        trajectory_reference_raw_path=reference_path,
        variant_raw_path=variant_path,
    )

    assert document["comparison"]["short_replay"]["passed"] is False  # type: ignore[index]
    assert document["promotion_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("stop_decision", True), ("accepted", False), ("backtracking_iteration_count", 1)],
)
def test_c2_rejects_non_native_step_control(
    tmp_path: Path, field: str, value: object
) -> None:
    _, _, _, variant_path, _, _ = _produce(tmp_path, "C2")
    candidate = json.loads(variant_path.read_text(encoding="utf-8"))
    candidate["short_replay"][0][field] = value

    with pytest.raises(
        TrajectoryOracleError, match="stop exactly|full steps|update/state relation"
    ):
        write_raw_trajectory_document(tmp_path / "invalid.json", candidate)


def test_coordinated_derived_verdict_drift_is_rejected_by_raw_rebuild(
    tmp_path: Path,
) -> None:
    oracle_path, _, _, _, identity, bindings = _produce(tmp_path, "C1")
    document = json.loads(oracle_path.read_text(encoding="utf-8"))
    document["comparison"]["one_step"]["passed"] = False
    document["comparison"]["parity_passed"] = False
    document["promotion_eligible"] = False
    oracle_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(TrajectoryOracleError, match="raw-recomputed evidence"):
        validate_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )


def test_raw_byte_change_fails_bound_sha_before_semantic_rebuild(
    tmp_path: Path,
) -> None:
    oracle_path, _, _, variant_path, identity, bindings = _produce(tmp_path, "C2")
    variant_path.write_bytes(variant_path.read_bytes() + b"\n")

    with pytest.raises(TrajectoryOracleError, match="hash mismatch"):
        validate_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )


def test_coordinated_raw_sha_update_still_requires_exact_semantic_derivation(
    tmp_path: Path,
) -> None:
    oracle_path, _, _, variant_path, identity, bindings = _produce(tmp_path, "C1")
    raw = json.loads(variant_path.read_text(encoding="utf-8"))
    raw["short_replay"][0]["accepted"] = False
    variant_path.write_bytes(canonical_json_bytes(raw))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle["raw_inputs"][2]["sha256"] = hashlib.sha256(
        variant_path.read_bytes()
    ).hexdigest()
    oracle_path.write_bytes(canonical_json_bytes(oracle))

    with pytest.raises(TrajectoryOracleError, match="raw bindings differ"):
        validate_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )


def test_fully_coordinated_raw_and_oracle_rebuild_cannot_replace_runner_binding(
    tmp_path: Path,
) -> None:
    (
        oracle_path,
        one_step_path,
        reference_path,
        variant_path,
        identity,
        bindings,
    ) = _produce(tmp_path, "C1")
    raw = json.loads(variant_path.read_text(encoding="utf-8"))
    raw["one_step"]["status_code"] = 1
    variant_path.write_bytes(canonical_json_bytes(raw))
    rebuilt = build_variant_trajectory_oracle(
        identity=identity,
        artifact_root=tmp_path,
        one_step_reference_raw_path=one_step_path,
        trajectory_reference_raw_path=reference_path,
        variant_raw_path=variant_path,
    )
    oracle_path.write_bytes(canonical_json_bytes(rebuilt))

    with pytest.raises(TrajectoryOracleError, match="raw bindings differ"):
        validate_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )


def test_artifact_cannot_widen_code_owned_tolerances(tmp_path: Path) -> None:
    oracle_path, _, _, _, identity, bindings = _produce(tmp_path, "C2")
    document = json.loads(oracle_path.read_text(encoding="utf-8"))
    document["tolerances"] = {"absolute": 2.0, "relative": 0.0}
    oracle_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(TrajectoryOracleError, match="tolerances differ from policy"):
        validate_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )


def test_raw_types_are_exact_and_reject_boolean_counter_confusion(
    tmp_path: Path,
) -> None:
    identity = _identity("C1")
    raw = _raw(identity, "C1")
    raw["one_step"]["counters"]["lu_solve_count"] = True  # type: ignore[index]

    with pytest.raises(TrajectoryOracleError, match="nonnegative integer"):
        write_raw_trajectory_document(tmp_path / "raw.json", raw)


def test_oracle_and_raw_documents_must_be_canonical(tmp_path: Path) -> None:
    oracle_path, _, _, _, identity, bindings = _produce(tmp_path, "C1")
    document = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    with pytest.raises(TrajectoryOracleError, match="not canonical JSON"):
        validate_variant_trajectory_oracle(
            oracle_path,
            artifact_root=tmp_path,
            expected_identity=identity,
            expected_raw_bindings=bindings,
        )
