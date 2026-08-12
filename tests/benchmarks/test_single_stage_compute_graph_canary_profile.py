from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from benchmarks.single_stage_compute_graph_c0_capture import TraceCaptureFacts
from benchmarks.single_stage_compute_graph_c0_evaluator import EvaluationResult
from benchmarks.single_stage_compute_graph_canary_evaluator import CanaryEvaluation
from benchmarks.single_stage_compute_graph_canary_profile import (
    PROFILE_CHILD_SCHEMA_ID,
    CanaryProfileError,
    _exact_hlo_ir_sha256,
    run_profile_probe,
)


def _numerical(objective: float = 1.0) -> EvaluationResult:
    return EvaluationResult(
        objective=objective,
        gradient=np.zeros(461, dtype=np.float64),
        inner_newton_success=True,
        adjoint_success=True,
        residual_certificates={"residual": 0.0},
    )


def _telemetry() -> dict[str, int | bool]:
    return {
        "exact_newton_variant_dense_linearization_used": True,
        "exact_newton_variant_linear_solve_attempt_count": 1,
        "exact_newton_variant_dense_materialization_count": 1,
        "exact_newton_variant_lu_factorization_count": 1,
        "exact_newton_variant_lu_solve_count": 1,
        "exact_newton_variant_refinement_correction_count": 0,
        "exact_newton_variant_stop_reason_code": 0,
        "exact_newton_variant_numerical_failure": False,
        "exact_newton_variant_backtracking_iteration_count": 1,
        "exact_newton_variant_stalled": False,
        "exact_newton_variant_retry_linear_solve_at_strict_cap": False,
    }


@dataclass
class _Prepared:
    evaluations: list[CanaryEvaluation]

    def evaluate_once(self) -> CanaryEvaluation:
        return self.evaluations.pop(0)


class _Nvtx:
    def __init__(self, return_value: int = 0) -> None:
        self.return_value = return_value
        self.labels: list[bytes] = []
        self.pop_count = 0

    def nvtxRangePushA(self, label: bytes) -> int:
        assert label.endswith(b"1" * 64)
        self.labels.append(label)
        return self.return_value

    def nvtxRangePop(self) -> int:
        self.pop_count += 1
        return self.return_value


def test_exact_hlo_identity_hashes_serialized_lowered_ir() -> None:
    class Hlo:
        def as_serialized_hlo_module_proto(self) -> bytes:
            return b"exact-hlo-ir"

    class Lowered:
        def compiler_ir(self, *, dialect: str):
            assert dialect == "hlo"
            return Hlo()

    class Evaluator:
        def lower(self, candidate, state):
            assert candidate == "candidate"
            assert state == "baseline-state"
            return Lowered()

    class Controller:
        current_inner_state = "baseline-state"

    class Runtime:
        incumbent_evaluator = Evaluator()

        def fresh_incumbent_controller(self):
            return Controller()

    class Prepared:
        _runtime = Runtime()
        _candidate = "candidate"

    assert _exact_hlo_ir_sha256(Prepared()) == (
        "927c954d4bbc1f67e0625cf8418dcc60b0d4756fdf6c781be31dac1e8362bc5b"
    )


def test_probe_warmup_is_outside_one_profiled_replay(monkeypatch, tmp_path) -> None:
    warm = CanaryEvaluation(_numerical(), _telemetry())
    profiled = CanaryEvaluation(_numerical(), _telemetry())
    prepared = _Prepared([warm, profiled])
    nvtx = _Nvtx(return_value=-1)

    def capture(adapter, *, parameter_sha256, trace_root):
        assert parameter_sha256 == "1" * 64
        assert trace_root == tmp_path / "trace"
        numerical = adapter.evaluate_once()
        return numerical, TraceCaptureFacts("2" * 64, ("jit_graph",), 3, 4)

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile._nvtx_library",
        lambda _path: nvtx,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile.capture_profiled_replay",
        capture,
    )
    document = run_profile_probe(
        variant="C1",
        prepared=prepared,
        parameter_sha256="1" * 64,
        trace_root=tmp_path / "trace",
        nvtx_library=tmp_path / "libnvtx.so",
        hlo_ir_identity=lambda _prepared: "3" * 64,
    )

    assert not prepared.evaluations
    assert document["schema_id"] == PROFILE_CHILD_SCHEMA_ID
    assert document["mode"] == "profile"
    assert document["capture"]["kernel_launch_count"] == 4  # type: ignore[index]
    assert len(nvtx.labels) == 1
    assert nvtx.pop_count == 1


def test_probe_propagates_capture_error_after_popping(monkeypatch, tmp_path) -> None:
    class CaptureFailure(RuntimeError):
        pass

    prepared = _Prepared([CanaryEvaluation(_numerical(), _telemetry())])
    nvtx = _Nvtx(return_value=-1)

    def capture(_adapter, **_kwargs):
        raise CaptureFailure("capture failed")

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile._nvtx_library",
        lambda _path: nvtx,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile.capture_profiled_replay",
        capture,
    )

    with pytest.raises(CaptureFailure, match="capture failed"):
        run_profile_probe(
            variant="C1",
            prepared=prepared,
            parameter_sha256="1" * 64,
            trace_root=tmp_path / "trace",
            nvtx_library=tmp_path / "libnvtx.so",
            hlo_ir_identity=lambda _prepared: "3" * 64,
        )

    assert nvtx.pop_count == 1


def test_probe_rejects_warm_profile_telemetry_drift(monkeypatch, tmp_path) -> None:
    changed = _telemetry()
    changed["exact_newton_variant_backtracking_iteration_count"] = 2
    prepared = _Prepared(
        [
            CanaryEvaluation(_numerical(), _telemetry()),
            CanaryEvaluation(_numerical(), changed),
        ]
    )

    def capture(adapter, **_kwargs):
        numerical = adapter.evaluate_once()
        return numerical, TraceCaptureFacts("2" * 64, ("jit_graph",), 3, 4)

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile._nvtx_library",
        lambda _path: _Nvtx(),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_profile.capture_profiled_replay",
        capture,
    )
    with pytest.raises(CanaryProfileError, match="telemetry differs"):
        run_profile_probe(
            variant="C1",
            prepared=prepared,
            parameter_sha256="1" * 64,
            trace_root=tmp_path / "trace",
            nvtx_library=tmp_path / "libnvtx.so",
            hlo_ir_identity=lambda _prepared: "3" * 64,
        )
