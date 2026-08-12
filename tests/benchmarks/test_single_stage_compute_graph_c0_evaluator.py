from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from benchmarks.single_stage_compute_graph_c0_evaluator import (
    CHILD_SCHEMA_ID,
    C0EvaluatorError,
    CaptureEvidence,
    ChildRequest,
    EvaluationResult,
    _canonical_candidate,
    _native_prepare,
    _request_from_environment,
    _residual_certificates,
    _validate_runtime_contract,
    _verify_snapshot_import_origins,
    build_child_observation,
)
from benchmarks.single_stage_compute_graph_c0_runner import _runtime_identity
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    HLO_MODULE_SET_IDENTITY_SOURCE,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
    canonical_hlo_module_set_identity,
)


@dataclass
class _Prepared:
    events: list[str]

    def evaluate_once(self) -> EvaluationResult:
        self.events.append("evaluate")
        return EvaluationResult(
            objective=1.25,
            gradient=np.linspace(0.0, 1.0, 461, dtype=np.float64),
            inner_newton_success=True,
            adjoint_success=True,
            residual_certificates={"adjoint_residual_relative": 1.0e-13},
        )

    def fresh_replay(self) -> _Prepared:
        self.events.append("fresh_replay")
        return self


def _parameter_sha(candidate: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(candidate, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def test_snapshot_origin_check_accepts_snapshot_namespace_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in (
        "benchmarks",
        "simsopt",
        "simsopt_jax",
        "simsopt_jax_adapters",
        "simsoptpp",
    ):
        module = ModuleType(name)
        module.__file__ = str(snapshot / f"{name}.py")
        monkeypatch.setitem(sys.modules, name, module)
    examples = ModuleType("examples")
    examples.__file__ = None
    examples.__path__ = [str(snapshot / "examples")]
    monkeypatch.setitem(sys.modules, "examples", examples)

    _verify_snapshot_import_origins(snapshot)


def test_snapshot_origin_check_rejects_namespace_path_outside_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in (
        "benchmarks",
        "simsopt",
        "simsopt_jax",
        "simsopt_jax_adapters",
        "simsoptpp",
    ):
        module = ModuleType(name)
        module.__file__ = str(snapshot / f"{name}.py")
        monkeypatch.setitem(sys.modules, name, module)
    examples = ModuleType("examples")
    examples.__file__ = None
    examples.__path__ = [str(tmp_path / "ambient" / "examples")]
    monkeypatch.setitem(sys.modules, "examples", examples)

    with pytest.raises(C0EvaluatorError, match="outside the immutable snapshot"):
        _verify_snapshot_import_origins(snapshot)


def test_exact_residual_certificates_omit_inapplicable_nan_sentinels() -> None:
    certificates = _residual_certificates(
        exact_residual=np.array([3.0, 4.0], dtype=np.float64),
        inner_penalty_residual_l2=float("nan"),
        final_gradient_inf_norm=float("nan"),
        adjoint_residual=np.array([0.0, 2.0], dtype=np.float64),
        adjoint_residual_relative=1.0e-13,
    )

    assert certificates == {
        "adjoint_residual_l2": 2.0,
        "adjoint_residual_relative": 1.0e-13,
        "boozer_exact_residual_l2": 5.0,
    }


def _capture(
    mode: str, sample_index: int | None, candidate: np.ndarray
) -> CaptureEvidence:
    return CaptureEvidence(
        mode=mode,  # type: ignore[arg-type]
        sample_index=sample_index,
        parameter_sha256=_parameter_sha(candidate),
        sampled_process_gpu_memory_peak_bytes=100,
        sampled_process_gpu_memory_source=SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
        hlo_module_set_identity=canonical_hlo_module_set_identity(("jit-c0",)),
        hlo_module_set_identity_source=HLO_MODULE_SET_IDENTITY_SOURCE,
        pjrt_execute_count=3 if mode == "profile" else None,
        kernel_launch_count=7 if mode == "profile" else None,
    )


def test_first_times_prepare_and_evaluation_before_profiled_fresh_replay() -> None:
    candidate = np.zeros(461, dtype=np.float64)
    events: list[str] = []
    prepared = _Prepared(events)

    def prepare(values: np.ndarray) -> _Prepared:
        events.append("prepare")
        return prepared

    def profile(
        replay: _Prepared, timed: EvaluationResult
    ) -> tuple[EvaluationResult, CaptureEvidence]:
        events.append("profile")
        return replay.evaluate_once(), _capture("profile", None, candidate)

    observation = build_child_observation(
        ChildRequest("profile", None),
        candidate,
        prepare,
        profile,
        clock=iter((10, 30, 70)).__next__,
        peak_rss=lambda: 200,
    )

    assert events == ["prepare", "evaluate", "fresh_replay", "profile", "evaluate"]
    assert observation["schema_id"] == CHILD_SCHEMA_ID
    assert observation["sample_index"] is None
    assert observation["parameter_sha256"] == _parameter_sha(candidate)
    assert observation["cold_compile"]["wall_ns"] == 60  # type: ignore[index]


def test_profile_compiles_with_device_scope_annotations_enabled() -> None:
    from simsopt_jax.runtime.trace_annotations import annotations_enabled

    candidate = np.zeros(461, dtype=np.float64)
    annotation_states: list[tuple[str, bool]] = []

    class _AnnotationAware(_Prepared):
        def evaluate_once(self) -> EvaluationResult:
            annotation_states.append(("evaluate", annotations_enabled()))
            return super().evaluate_once()

    prepared = _AnnotationAware([])

    def prepare(values: np.ndarray) -> _Prepared:
        del values
        annotation_states.append(("prepare", annotations_enabled()))
        return prepared

    def profile(
        replay: _Prepared, timed: EvaluationResult
    ) -> tuple[EvaluationResult, CaptureEvidence]:
        del timed
        return replay.evaluate_once(), _capture("profile", None, candidate)

    build_child_observation(
        ChildRequest("profile", None),
        candidate,
        prepare,
        profile,
        clock=iter((10, 30, 70)).__next__,
        peak_rss=lambda: 200,
    )

    assert annotation_states[:2] == [("prepare", True), ("evaluate", True)]
    assert annotations_enabled() is False


def test_warm_timing_excludes_preparation_and_never_profiles() -> None:
    candidate = np.zeros(461, dtype=np.float64)
    prepared = _Prepared([])
    observation = build_child_observation(
        ChildRequest("warm", 0),
        candidate,
        lambda values: prepared,
        None,
        clock=iter((10, 40, 65)).__next__,
        peak_rss=lambda: 200,
    )

    assert observation["wall_ns"] == 25
    assert prepared.events == ["evaluate"]


def test_gate_performs_exactly_prepare_then_one_unprofiled_evaluation() -> None:
    events: list[str] = []
    prepared = _Prepared(events)

    def prepare(values: np.ndarray) -> _Prepared:
        events.append("prepare")
        return prepared

    observation = build_child_observation(
        ChildRequest("gate", None),
        np.zeros(461, dtype=np.float64),
        prepare,
        None,
        clock=iter((10, 20, 30)).__next__,
        peak_rss=lambda: 200,
    )

    assert observation["mode"] == "gate"
    assert events == ["prepare", "evaluate"]


def test_initial_gate_emits_parameter_bound_full_numerical_evidence() -> None:
    candidate = np.linspace(-1.0, 1.0, 461, dtype=np.float64)
    prepared = _Prepared([])

    observation = build_child_observation(
        ChildRequest("initial-gate", None),
        candidate,
        lambda values: prepared,
        None,
        clock=iter((10, 20, 30)).__next__,
        peak_rss=lambda: 200,
    )

    assert observation["mode"] == "initial-gate"
    assert observation["parameter_sha256"] == _parameter_sha(candidate)
    assert observation["objective_dtype"] == "float64"
    assert len(observation["gradient"]) == 461  # type: ignore[arg-type]
    assert prepared.events == ["evaluate"]


def test_profiled_replay_must_equal_timed_result() -> None:
    candidate = np.zeros(461, dtype=np.float64)
    prepared = _Prepared([])

    def unequal(
        replay: _Prepared, timed: EvaluationResult
    ) -> tuple[EvaluationResult, CaptureEvidence]:
        result = replay.evaluate_once()
        return (
            EvaluationResult(
                objective=result.objective + 1.0,
                gradient=result.gradient,
                inner_newton_success=True,
                adjoint_success=True,
                residual_certificates=result.residual_certificates,
            ),
            _capture("profile", None, candidate),
        )

    with pytest.raises(C0EvaluatorError, match="numerically equal"):
        build_child_observation(
            ChildRequest("profile", None),
            candidate,
            lambda values: prepared,
            unequal,
            clock=iter((1, 2, 3)).__next__,
            peak_rss=lambda: 1,
        )


def test_failed_timed_solver_is_never_profiled() -> None:
    class _Failed(_Prepared):
        def evaluate_once(self) -> EvaluationResult:
            return EvaluationResult(
                objective=1.0,
                gradient=np.ones(461, dtype=np.float64),
                inner_newton_success=False,
                adjoint_success=True,
                residual_certificates={"residual": 0.0},
            )

    with pytest.raises(C0EvaluatorError, match="must both succeed"):
        build_child_observation(
            ChildRequest("gate", None),
            np.zeros(461, dtype=np.float64),
            lambda values: _Failed([]),
            lambda replay, timed: pytest.fail("failed result must not be profiled"),
            clock=iter((1, 2, 3)).__next__,
            peak_rss=lambda: 1,
        )


def test_candidate_and_request_validation(tmp_path: Path) -> None:
    candidate = np.linspace(-1.0, 1.0, 461, dtype=np.float64)
    path = tmp_path / "candidate.npy"
    np.save(path, candidate)
    loaded = _canonical_candidate(path, _parameter_sha(candidate))
    assert loaded.shape == (461,)
    assert not loaded.flags.writeable
    with pytest.raises(C0EvaluatorError, match="SHA-256"):
        _canonical_candidate(path, "0" * 64)
    assert _request_from_environment(
        {
            "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT": "C0",
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE": "warm",
            "SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX": "0",
        }
    ) == ChildRequest("warm", 0)
    assert _request_from_environment(
        {
            "SINGLE_STAGE_COMPUTE_GRAPH_VARIANT": "C0",
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE": "initial-gate",
        }
    ) == ChildRequest("initial-gate", None)


def test_c0_prepare_uses_only_the_jax_runtime() -> None:
    source = inspect.getsource(_native_prepare)

    assert "_prepare_jax_variant_runtime" in source
    assert "_prepare_native_variant_runtime" not in source


def test_runtime_contract_rejects_extra_allowlisted_static_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_environment = normalize_static_timing_environment(os.environ)
    static_environment.pop("XLA_FLAGS", None)
    policies = {
        "dense_batch_width": 8,
        "point_chunk_size": None,
        "coil_chunk_size": None,
        "quadrature_block_sizes": [128, 122],
    }
    provenance = {
        "interpreter_path": str(Path(sys.executable).absolute()),
        "runtime": {},
        "environment": static_environment,
        "policies": policies,
    }
    identity = _runtime_identity(provenance)
    contract = {
        "runtime": {},
        "static_environment": static_environment,
        "route_environment": {},
        "policies": policies,
        "expected_runtime_identity_sha256": identity,
    }
    monkeypatch.setenv("XLA_FLAGS", "--unexpected-runner-override")
    monkeypatch.setenv(
        "SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT",
        json.dumps(contract, sort_keys=True, separators=(",", ":")),
    )
    monkeypatch.setenv("SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY", identity)

    with pytest.raises(C0EvaluatorError, match="static runtime environment"):
        _validate_runtime_contract()
