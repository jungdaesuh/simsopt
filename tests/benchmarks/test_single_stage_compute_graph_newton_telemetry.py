from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.single_stage_compute_graph_c0_runner import _bound_post_gate_document
from benchmarks.single_stage_compute_graph_newton_telemetry import (
    OBSERVER_ENV,
    CandidateEvaluation,
    ExecutionCounts,
    NewtonTelemetryError,
    TelemetryIdentity,
    collect_newton_telemetry,
    main,
    validate_newton_telemetry_evidence,
    verify_input_bundle_bytes,
    write_newton_telemetry,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    _validate_newton_telemetry,
)
from simsopt_jax_adapters.geo.boozer_surface import _exact_newton_reporting_fields
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _pack_traceable_forward_result,
)


def _candidate() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 461, dtype=np.float64)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.dtype("<f8")).tobytes(order="C")
    ).hexdigest()


def _identity(candidate: np.ndarray) -> TelemetryIdentity:
    return TelemetryIdentity(
        candidate_sha256=_digest(candidate),
        specimen_sha256="b" * 64,
        input_bundle_sha256="8" * 64,
        source_sha256="c" * 64,
        runtime_identity_sha256="9" * 64,
        lane_id="rtx5090",
        gpu_uuid="GPU-rtx5090",
        gate_checkpoint_sha256="d" * 64,
        warm_checkpoint_sha256="e" * 64,
        warm_p50_ns=95.5,
    )


class _Prepared:
    def __init__(
        self,
        candidate: np.ndarray,
        *,
        observer_bearing: bool,
        perturb_observed: bool,
    ) -> None:
        self._candidate = candidate
        self._observer_bearing = observer_bearing
        self._perturb_observed = perturb_observed

    def evaluate(self) -> CandidateEvaluation:
        gradient = self._candidate.copy()
        if self._observer_bearing and self._perturb_observed:
            gradient[0] += 1.0
        return CandidateEvaluation(
            objective=1.25,
            raw_objective=1.5,
            gradient=gradient,
            solved_state=np.asarray([0.25, 0.5], dtype=np.float64),
            newton_success=True,
            newton_iterations=2,
            observer_bearing=self._observer_bearing,
            execution_counts=(
                ExecutionCounts(7, 5)
                if self._observer_bearing
                else ExecutionCounts(0, 0)
            ),
        )


def _prepare_factory(events: list[str], *, perturb_observed: bool = False):
    def prepare(candidate: np.ndarray) -> _Prepared:
        enabled = os.environ.get(OBSERVER_ENV) == "1"
        events.append("prepare-observed" if enabled else "prepare-unobserved")
        return _Prepared(
            candidate,
            observer_bearing=enabled,
            perturb_observed=perturb_observed,
        )

    return prepare


def test_collects_actual_counts_and_exact_observer_replay_outside_timing() -> None:
    candidate = _candidate()
    events: list[str] = []

    document = collect_newton_telemetry(
        _identity(candidate),
        candidate,
        _prepare_factory(events),
        clock=iter((10, 30, 100, 150)).__next__,
    )

    assert events == ["prepare-unobserved", "prepare-observed"]
    assert document["state"] == "PRODUCED"
    assert document["warmup_executions_per_lane"] == 1
    assert all(document["numerical_equality"].values())
    assert document["observer"] == {
        "api": "device_resident_fixed_shape_exact_newton_counts",
        "device_resident_fixed_shape_counts": True,
        "host_callback_used": False,
        "promotion_timing_included": False,
    }
    identity = document["identity"]
    assert identity["candidate_sha256"] == _digest(candidate)
    assert identity["specimen_sha256"] == "b" * 64
    assert identity["input_bundle_sha256"] == "8" * 64
    assert identity["source_sha256"] == "c" * 64
    assert identity["runtime_identity_sha256"] == "9" * 64
    assert identity["lane_id"] == "rtx5090"
    assert identity["gpu_uuid"] == "GPU-rtx5090"
    assert identity["gate_checkpoint_sha256"] == "d" * 64
    assert identity["warm_checkpoint_sha256"] == "e" * 64
    assert identity["warm_p50_ns"] == 95.5
    receipt = document["newton_telemetry"]
    raw_evidence_sha256 = receipt["raw_evidence_sha256"]
    assert receipt == {
        "telemetry_schema_id": "single-stage-compute-graph-newton-telemetry-v2",
        "route_id": "production-exact-newton",
        "measurement_method": "device_resident_fixed_shape_exact_newton_counts",
        "host_callback_used": False,
        "raw_evidence_sha256": raw_evidence_sha256,
        "residual_evaluations": 7,
        "linear_operator_applications": 5,
        "observed_wall_ns": 50,
        "unobserved_wall_ns": 20,
        "observer_effect_ratio": 2.5,
        "collected_outside_timed_samples": True,
    }
    assert len(raw_evidence_sha256) == 64
    _validate_newton_telemetry(receipt, "newton_telemetry")
    assert validate_newton_telemetry_evidence(document, _identity(candidate)) == receipt

    document["newton_telemetry"]["raw_evidence_sha256"] = "0" * 64
    with pytest.raises(NewtonTelemetryError, match="raw evidence digest"):
        validate_newton_telemetry_evidence(document, _identity(candidate))

    document["identity"]["warm_checkpoint_sha256"] = "f" * 64
    with pytest.raises(NewtonTelemetryError, match="differs from checkpoints"):
        validate_newton_telemetry_evidence(document, _identity(candidate))


def test_numerical_difference_fails_closed() -> None:
    candidate = _candidate()

    with pytest.raises(NewtonTelemetryError, match="gradient_exact"):
        collect_newton_telemetry(
            _identity(candidate),
            candidate,
            _prepare_factory([], perturb_observed=True),
            clock=iter((1, 2, 3, 4)).__next__,
        )


def test_missing_or_zero_execution_counts_fail_closed() -> None:
    candidate = _candidate()
    prepared = _prepare_factory([])

    class _ZeroCountPrepared:
        def evaluate(self) -> CandidateEvaluation:
            result = prepared(candidate).evaluate()
            if result.observer_bearing:
                return CandidateEvaluation(
                    objective=result.objective,
                    raw_objective=result.raw_objective,
                    gradient=result.gradient,
                    solved_state=result.solved_state,
                    newton_success=result.newton_success,
                    newton_iterations=result.newton_iterations,
                    observer_bearing=True,
                    execution_counts=ExecutionCounts(0, 0),
                )
            return result

    with pytest.raises(NewtonTelemetryError, match="device counts"):
        collect_newton_telemetry(
            _identity(candidate),
            candidate,
            lambda _candidate: _ZeroCountPrepared(),
        )


def test_identity_and_candidate_binding_precede_evaluation() -> None:
    candidate = _candidate()
    identity = _identity(candidate)
    invalid = TelemetryIdentity(
        candidate_sha256="0" * 64,
        specimen_sha256=identity.specimen_sha256,
        input_bundle_sha256=identity.input_bundle_sha256,
        source_sha256=identity.source_sha256,
        runtime_identity_sha256=identity.runtime_identity_sha256,
        lane_id=identity.lane_id,
        gpu_uuid=identity.gpu_uuid,
        gate_checkpoint_sha256=identity.gate_checkpoint_sha256,
        warm_checkpoint_sha256=identity.warm_checkpoint_sha256,
        warm_p50_ns=identity.warm_p50_ns,
    )
    events: list[str] = []

    with pytest.raises(NewtonTelemetryError, match="candidate bytes"):
        collect_newton_telemetry(
            invalid,
            candidate,
            _prepare_factory(events),
        )

    assert events == []


def test_adapter_propagates_exact_observer_token_into_fixed_shape_forward_result() -> (
    None
):
    reported = _exact_newton_reporting_fields(
        {
            "exact_newton_execution_observer_bearing": jnp.asarray(True),
            "exact_newton_residual_evaluation_count": jnp.asarray(7, dtype=jnp.int32),
            "exact_newton_linear_operator_application_count": jnp.asarray(
                5, dtype=jnp.int32
            ),
        }
    )
    packed = _pack_traceable_forward_result(
        value=jnp.asarray(1.0),
        x=jnp.asarray([1.0, 2.0]),
        sdofs=jnp.asarray([1.0]),
        iota=jnp.asarray(0.5),
        G=jnp.asarray(2.0),
        linear_solve_factors=None,
        success=jnp.asarray(True),
        primal_success=jnp.asarray(True),
        adjoint_linear_solve_available=jnp.asarray(False),
        newton_trace_capacity=0,
        exact_newton_execution_observer_bearing=reported[
            "exact_newton_execution_observer_bearing"
        ],
        exact_newton_residual_evaluation_count=reported[
            "exact_newton_residual_evaluation_count"
        ],
        exact_newton_linear_operator_application_count=reported[
            "exact_newton_linear_operator_application_count"
        ],
    )
    unobserved = _pack_traceable_forward_result(
        value=jnp.asarray(1.0),
        x=jnp.asarray([1.0, 2.0]),
        sdofs=jnp.asarray([1.0]),
        iota=jnp.asarray(0.5),
        G=jnp.asarray(2.0),
        linear_solve_factors=None,
        success=jnp.asarray(True),
        primal_success=jnp.asarray(True),
        adjoint_linear_solve_available=jnp.asarray(False),
        newton_trace_capacity=0,
    )

    assert bool(packed["exact_newton_execution_observer_bearing"])
    assert int(packed["exact_newton_residual_evaluation_count"]) == 7
    assert int(packed["exact_newton_linear_operator_application_count"]) == 5
    assert not bool(unobserved["exact_newton_execution_observer_bearing"])
    assert int(unobserved["exact_newton_residual_evaluation_count"]) == 0
    assert int(unobserved["exact_newton_linear_operator_application_count"]) == 0
    assert packed.keys() == unobserved.keys()


def test_evidence_write_is_canonical_and_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "newton-telemetry.json"
    document = {"schema_id": "test", "state": "PRODUCED", "value": 1}

    write_newton_telemetry(path, document)

    assert json.loads(path.read_text(encoding="utf-8")) == document
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_newton_telemetry(path, document)


def test_input_bundle_verification_hashes_raw_manifest_bytes(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    bundle_path = input_root / "input_bundle.json"
    raw_bytes = b'{"configuration":{"value":1}}\n'
    bundle_path.write_bytes(raw_bytes)

    assert (
        verify_input_bundle_bytes(
            input_root,
            hashlib.sha256(raw_bytes).hexdigest(),
        )
        == input_root
    )

    bundle_path.write_bytes(b'{"configuration":{"value":2}}\n')
    with pytest.raises(NewtonTelemetryError, match="raw input_bundle.json"):
        verify_input_bundle_bytes(input_root, hashlib.sha256(raw_bytes).hexdigest())


def test_evidence_is_directly_ingestible_by_staged_runner(tmp_path: Path) -> None:
    candidate = _candidate()
    identity = _identity(candidate)
    document = collect_newton_telemetry(
        identity,
        candidate,
        _prepare_factory([]),
        clock=iter((10, 30, 100, 150)).__next__,
    )
    path = tmp_path / "newton-telemetry.json"
    write_newton_telemetry(path, document)

    validated_payload = validate_newton_telemetry_evidence(document, identity)
    payload = _bound_post_gate_document(
        path,
        expected_identity=identity.to_json(),
        payload_key="newton_telemetry",
    )

    assert payload == validated_payload


def test_cli_writes_canonical_blocked_evidence_before_runtime_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.npy"
    np.save(candidate, _candidate())
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "input_bundle.json").write_bytes(b"wrong bundle bytes\n")
    output = tmp_path / "telemetry.json"
    prepare_calls: list[Path] = []

    def unexpected_prepare(input_path: Path, values: np.ndarray) -> _Prepared:
        del values
        prepare_calls.append(input_path)
        raise AssertionError("runtime preparation must follow bundle verification")

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_newton_telemetry.prepare_production_candidate",
        unexpected_prepare,
    )

    status = main(
        [
            "--input-root",
            str(input_root),
            "--candidate",
            str(candidate),
            "--candidate-sha256",
            "0" * 64,
            "--specimen-sha256",
            "b" * 64,
            "--input-bundle-sha256",
            "8" * 64,
            "--source-sha256",
            "c" * 64,
            "--runtime-identity-sha256",
            "9" * 64,
            "--lane-id",
            "rtx5090",
            "--gpu-uuid",
            "GPU-rtx5090",
            "--gate-checkpoint-sha256",
            "d" * 64,
            "--warm-checkpoint-sha256",
            "e" * 64,
            "--warm-p50-ns",
            "95.5",
            "--output",
            str(output),
        ]
    )

    assert status == 2
    blocked = json.loads(output.read_text(encoding="utf-8"))
    assert blocked["state"] == "BLOCKED"
    assert blocked["code"] == "EXACT_NEWTON_TELEMETRY_BLOCKED"
    assert blocked["identity"]["source_sha256"] == "c" * 64
    assert blocked["identity"]["input_bundle_sha256"] == "8" * 64
    assert blocked["identity"]["runtime_identity_sha256"] == "9" * 64
    assert "newton_telemetry" not in blocked
    assert prepare_calls == []
