from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from benchmarks.single_stage_compute_graph_native_trajectory import (
    NativeTrajectoryError,
    build_native_raw_trajectory,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    validate_raw_trajectory_document,
    write_raw_trajectory_document,
)


def _digest(character: str) -> str:
    return character * 64


def _counters(iteration: int, assessed: int) -> dict[str, int]:
    return {
        "residual_evaluation_count": assessed,
        "attempted_iteration_count": iteration,
        "applied_update_count": iteration,
        "assessed_state_count": assessed,
        "rollback_recompute_count": 0,
        "dense_materialization_count": iteration,
        "factorization_count": 2 * iteration,
        "linear_solve_count": 2 * iteration,
        "refinement_correction_count": iteration,
    }


def _assessment(
    index: int, state: list[float], residual: list[float]
) -> dict[str, object]:
    residual_array = np.asarray(residual, dtype=np.float64)
    return {
        "event": "assessment",
        "iteration_index": index,
        "state": np.asarray(state, dtype=np.float64),
        "residual": residual_array,
        "residual_norm": float(np.linalg.norm(residual_array)),
        "counters": _counters(index, index + 1),
    }


def _update(
    index: int, before: list[float], correction: list[float]
) -> dict[str, object]:
    initial = np.asarray(correction, dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)
    jacobian = np.eye(2, dtype=np.float64) * 8.0
    return {
        "event": "update",
        "iteration_index": index,
        "state_before": np.asarray(before, dtype=np.float64),
        "residual": jacobian @ initial,
        "jacobian": jacobian,
        "initial_solve": initial,
        "refinement_rhs": zero,
        "refinement_correction": zero,
        "refined_residual": zero,
        "state_after": np.asarray(before, dtype=np.float64) - initial,
        "assessed_norm": float(np.linalg.norm(jacobian @ initial)),
        "counters": _counters(index + 1, index + 1),
    }


def _events() -> list[dict[str, object]]:
    terminal_counters = _counters(2, 3)
    return [
        _assessment(0, [1.0, 2.0], [2.0, 4.0]),
        _update(0, [1.0, 2.0], [0.25, 0.5]),
        _assessment(1, [0.75, 1.5], [1.0, 2.0]),
        _update(1, [0.75, 1.5], [0.125, 0.25]),
        _assessment(2, [0.625, 1.25], [1.0e-13, 0.0]),
        {
            "event": "terminal",
            "iteration_index": 2,
            "success": True,
            "persist_solved_state": True,
            "rollback_taken": False,
            "initial_norm": 10.0,
            "returned_norm": 1.0e-13,
            "returned_state": np.asarray([0.625, 1.25], dtype=np.float64),
            "returned_residual": np.asarray([1.0e-13, 0.0], dtype=np.float64),
            "returned_jacobian": np.eye(2, dtype=np.float64) * 8.0,
            "status_code": 0,
            "counters": terminal_counters,
        },
    ]


def _document(events: list[dict[str, object]]) -> dict[str, object]:
    return build_native_raw_trajectory(
        events,
        parameter_sha256=_digest("1"),
        specimen_sha256=_digest("2"),
        input_bundle_sha256=_digest("3"),
        solver_graph_sha256=_digest("4"),
        source_sha256=_digest("5"),
        tolerance=1.0e-12,
    )


def test_native_events_build_a_valid_raw_trajectory(tmp_path: Path) -> None:
    document = _document(_events())
    normalized = validate_raw_trajectory_document(document)
    assert normalized["one_step"]["converged"] is False
    assert normalized["short_replay"][1]["stop_decision"] is True
    assert normalized["terminal"]["success"] is True

    path = tmp_path / "native-trajectory.json"
    digest = write_raw_trajectory_document(path, document)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes() == canonical_json_bytes(json.loads(path.read_bytes()))


def test_native_event_order_is_fail_closed() -> None:
    events = _events()
    events[2], events[3] = events[3], events[2]
    with pytest.raises(NativeTrajectoryError, match="must be 'assessment'"):
        _document(events)


def test_native_short_replay_requires_two_observed_updates() -> None:
    events = _events()
    shortened = [events[0], events[1], events[2], events[-1]]
    with pytest.raises(NativeTrajectoryError, match="at least two updates"):
        _document(shortened)


def test_native_terminal_status_type_is_exact() -> None:
    events = _events()
    events[-1]["status_code"] = True
    with pytest.raises(NativeTrajectoryError, match="status code"):
        _document(events)


def test_later_native_update_algebra_is_recomputed() -> None:
    events = _events()
    events[3]["refined_residual"] = np.asarray([1.0, 0.0], dtype=np.float64)
    with pytest.raises(NativeTrajectoryError, match="refined residual"):
        _document(events)


def test_native_counter_progression_is_recomputed() -> None:
    events = _events()
    events[3]["counters"]["applied_update_count"] = 7
    with pytest.raises(NativeTrajectoryError, match="counters are inconsistent"):
        _document(events)


def test_native_terminal_iteration_and_norm_are_recomputed() -> None:
    events = _events()
    events[-1]["iteration_index"] = 3
    with pytest.raises(NativeTrajectoryError, match="terminal iteration"):
        _document(events)

    events = _events()
    events[-1]["returned_norm"] = 1.0
    with pytest.raises(NativeTrajectoryError, match="residual norm"):
        _document(events)
