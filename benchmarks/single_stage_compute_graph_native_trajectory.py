"""Fresh-process native trajectory producer for C1/C2 oracle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from examples.jax.parity.cases.native_boozerqa import _prepare_native_variant_runtime
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import read_input_bundle
from simsopt.geo.boozersurface import _boozer_exact_newton_observation_context

from benchmarks.single_stage_compute_graph_native_reference import (
    NativeReferenceBinding,
    _canonical_candidate,
    _sha256,
    _sha256_path,
    _validate_runtime_binding,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    RAW_TRAJECTORY_SCHEMA_ID,
    write_raw_trajectory_document,
)

EXPECTED_PARAMETER_COUNT: Final = 461
_COUNTER_FIELDS: Final = frozenset(
    {
        "residual_evaluation_count",
        "attempted_iteration_count",
        "applied_update_count",
        "assessed_state_count",
        "rollback_recompute_count",
        "dense_materialization_count",
        "factorization_count",
        "linear_solve_count",
        "refinement_correction_count",
    }
)
_EVENT_FIELDS: Final = {
    "assessment": frozenset(
        {"event", "iteration_index", "state", "residual", "residual_norm", "counters"}
    ),
    "update": frozenset(
        {
            "event",
            "iteration_index",
            "state_before",
            "residual",
            "jacobian",
            "initial_solve",
            "refinement_rhs",
            "refinement_correction",
            "refined_residual",
            "state_after",
            "assessed_norm",
            "counters",
        }
    ),
    "terminal": frozenset(
        {
            "event",
            "iteration_index",
            "success",
            "persist_solved_state",
            "rollback_taken",
            "initial_norm",
            "returned_norm",
            "returned_state",
            "returned_residual",
            "returned_jacobian",
            "status_code",
            "counters",
        }
    ),
}


class NativeTrajectoryError(RuntimeError):
    """Native observation events do not form a complete trajectory."""


def _array(value: object, context: str, *, dimensions: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != dimensions:
        raise NativeTrajectoryError(f"{context} must be an exact FP64 array")
    if not bool(np.all(np.isfinite(array))):
        raise NativeTrajectoryError(f"{context} must contain only finite values")
    return array


def _require_array_close(
    observed: np.ndarray, expected: np.ndarray, context: str
) -> None:
    if observed.shape != expected.shape or not bool(
        np.allclose(observed, expected, rtol=1.0e-13, atol=1.0e-15)
    ):
        raise NativeTrajectoryError(f"{context} is algebraically inconsistent")


def _canonical_solver_graph_sha256(path: Path) -> str:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or raw != canonical_json_bytes(document):
        raise NativeTrajectoryError("solver graph must be a canonical JSON object")
    return hashlib.sha256(raw).hexdigest()


def _event(value: object, expected: str, index: int) -> Mapping[str, object]:
    if not isinstance(value, dict) or value.get("event") != expected:
        raise NativeTrajectoryError(
            f"native observation event {index} must be {expected!r}"
        )
    if frozenset(value) != _EVENT_FIELDS[expected]:
        raise NativeTrajectoryError(
            f"native observation event {index} has invalid {expected} fields"
        )
    return value


def _counters(event: Mapping[str, object], context: str) -> Mapping[str, int]:
    value = event.get("counters")
    if (
        not isinstance(value, dict)
        or frozenset(value) != _COUNTER_FIELDS
        or not all(
            isinstance(key, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for key, count in value.items()
        )
    ):
        raise NativeTrajectoryError(f"{context} counters are invalid")
    return value


def _native_replay_counters(event: Mapping[str, object]) -> dict[str, int]:
    counters = _counters(event, "native replay")
    return {
        field: counters[field]
        for field in (
            "residual_evaluation_count",
            "attempted_iteration_count",
            "applied_update_count",
            "assessed_state_count",
            "rollback_recompute_count",
        )
    }


def _expected_counters(
    *, iteration: int, residual_evaluations: int, assessed: int, rollback: int = 0
) -> dict[str, int]:
    return {
        "residual_evaluation_count": residual_evaluations,
        "attempted_iteration_count": iteration,
        "applied_update_count": iteration,
        "assessed_state_count": assessed,
        "rollback_recompute_count": rollback,
        "dense_materialization_count": iteration,
        "factorization_count": 2 * iteration,
        "linear_solve_count": 2 * iteration,
        "refinement_correction_count": iteration,
    }


def _require_counters(
    event: Mapping[str, object], expected: Mapping[str, int], context: str
) -> None:
    if _counters(event, context) != expected:
        raise NativeTrajectoryError(f"{context} counters are inconsistent")


def _one_step(update: Mapping[str, object], converged: bool) -> dict[str, object]:
    initial_solve = _array(update.get("initial_solve"), "initial solve", dimensions=1)
    refinement = _array(
        update.get("refinement_correction"),
        "refinement correction",
        dimensions=1,
    )
    correction = initial_solve + refinement
    counters = _counters(update, "native one-step")
    return {
        "initial_state": _array(
            update.get("state_before"), "one-step initial state", dimensions=1
        ).tolist(),
        "residual": _array(
            update.get("residual"), "one-step residual", dimensions=1
        ).tolist(),
        "jacobian": _array(
            update.get("jacobian"), "one-step Jacobian", dimensions=2
        ).tolist(),
        "initial_solve": initial_solve.tolist(),
        "refinement_rhs": _array(
            update.get("refinement_rhs"), "refinement RHS", dimensions=1
        ).tolist(),
        "refinement_correction": refinement.tolist(),
        "correction_step": correction.tolist(),
        "refined_residual": _array(
            update.get("refined_residual"), "refined residual", dimensions=1
        ).tolist(),
        "next_state": _array(
            update.get("state_after"), "one-step next state", dimensions=1
        ).tolist(),
        "converged": converged,
        "numerical_failure": False,
        "status_code": 0,
        "counters": {
            "residual_evaluation_count": counters["residual_evaluation_count"],
            "dense_materialization_count": counters["dense_materialization_count"],
            "lu_factorization_count": counters["factorization_count"],
            "lu_solve_count": counters["linear_solve_count"],
            "refinement_correction_count": counters["refinement_correction_count"],
        },
    }


def build_native_raw_trajectory(
    events: Sequence[Mapping[str, object]],
    *,
    parameter_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    solver_graph_sha256: str,
    source_sha256: str,
    tolerance: float,
) -> dict[str, object]:
    """Convert a complete exact-Newton observer stream into canonical raw data."""

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise NativeTrajectoryError("native Newton tolerance must be positive")
    if len(events) < 4:
        raise NativeTrajectoryError("native observation stream is incomplete")
    terminal = _event(events[-1], "terminal", len(events) - 1)
    body = events[:-1]
    updates: list[Mapping[str, object]] = []
    assessments: dict[int, Mapping[str, object]] = {}
    expected_kind = "assessment"
    for index, value in enumerate(body):
        event = _event(value, expected_kind, index)
        iteration = event.get("iteration_index")
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            raise NativeTrajectoryError("native iteration index is invalid")
        if expected_kind == "assessment":
            if iteration in assessments:
                raise NativeTrajectoryError("native assessment iteration is duplicated")
            assessments[iteration] = event
            expected_kind = "update"
        else:
            if iteration != len(updates):
                raise NativeTrajectoryError("native update indices are not contiguous")
            updates.append(event)
            expected_kind = "assessment"
    if len(updates) < 2:
        raise NativeTrajectoryError("native short replay requires at least two updates")
    if 0 not in assessments:
        raise NativeTrajectoryError("native initial assessment is missing")

    terminal_status = terminal.get("status_code")
    if not isinstance(terminal_status, int) or isinstance(terminal_status, bool):
        raise NativeTrajectoryError("native terminal status code is invalid")
    terminal_iteration = terminal.get("iteration_index")
    if terminal_iteration != len(updates):
        raise NativeTrajectoryError("native terminal iteration is inconsistent")
    for field in ("success", "persist_solved_state", "rollback_taken"):
        if not isinstance(terminal.get(field), bool):
            raise NativeTrajectoryError(f"native terminal {field} must be boolean")
    if terminal["persist_solved_state"] == terminal["rollback_taken"]:
        raise NativeTrajectoryError("native terminal persistence flags disagree")

    for index, assessment in assessments.items():
        state = _array(assessment.get("state"), "assessment state", dimensions=1)
        residual = _array(
            assessment.get("residual"), "assessment residual", dimensions=1
        )
        residual_norm = assessment.get("residual_norm")
        if (
            not isinstance(residual_norm, float)
            or not np.isfinite(residual_norm)
            or not np.isclose(
                residual_norm,
                np.linalg.norm(residual),
                rtol=1.0e-13,
                atol=1.0e-15,
            )
        ):
            raise NativeTrajectoryError("native assessment norm is inconsistent")
        _require_counters(
            assessment,
            _expected_counters(
                iteration=index, residual_evaluations=index + 1, assessed=index + 1
            ),
            f"native assessment {index}",
        )
        if index < len(updates):
            update = updates[index]
            if not np.array_equal(
                state,
                _array(update.get("state_before"), "update state", dimensions=1),
            ) or not np.array_equal(
                residual,
                _array(update.get("residual"), "update residual", dimensions=1),
            ):
                raise NativeTrajectoryError(
                    "native assessment and update payloads disagree"
                )
            assessed_norm = update.get("assessed_norm")
            if not isinstance(assessed_norm, float) or assessed_norm != residual_norm:
                raise NativeTrajectoryError("native update assessed norm disagrees")

    for index, update in enumerate(updates):
        _require_counters(
            update,
            _expected_counters(
                iteration=index + 1,
                residual_evaluations=index + 1,
                assessed=index + 1,
            ),
            f"native update {index}",
        )
        jacobian = _array(
            update.get("jacobian"), f"native update {index} Jacobian", dimensions=2
        )
        residual = _array(
            update.get("residual"), f"native update {index} residual", dimensions=1
        )
        initial = _array(
            update.get("initial_solve"),
            f"native update {index} initial solve",
            dimensions=1,
        )
        refinement_rhs = _array(
            update.get("refinement_rhs"),
            f"native update {index} refinement RHS",
            dimensions=1,
        )
        refinement = _array(
            update.get("refinement_correction"),
            f"native update {index} refinement correction",
            dimensions=1,
        )
        refined_residual = _array(
            update.get("refined_residual"),
            f"native update {index} refined residual",
            dimensions=1,
        )
        state_before = _array(
            update.get("state_before"),
            f"native update {index} state before",
            dimensions=1,
        )
        state_after = _array(
            update.get("state_after"),
            f"native update {index} state after",
            dimensions=1,
        )
        correction = initial + refinement
        _require_array_close(
            refinement_rhs,
            residual - jacobian @ initial,
            f"native update {index} refinement RHS",
        )
        _require_array_close(
            refined_residual,
            residual - jacobian @ correction,
            f"native update {index} refined residual",
        )
        _require_array_close(
            state_after,
            state_before - correction,
            f"native update {index} state transition",
        )
        following = assessments.get(index + 1)
        if following is not None and not np.array_equal(
            _array(update.get("state_after"), "update state after", dimensions=1),
            _array(following.get("state"), "following state", dimensions=1),
        ):
            raise NativeTrajectoryError(
                "native update/assessment state chain disagrees"
            )

    rollback_count = 1 if terminal["rollback_taken"] else 0
    assessed_count = len(assessments)
    _require_counters(
        terminal,
        _expected_counters(
            iteration=len(updates),
            residual_evaluations=len(updates) + 1 + rollback_count,
            assessed=assessed_count,
            rollback=rollback_count,
        ),
        "native terminal",
    )
    terminal_residual = _array(
        terminal.get("returned_residual"), "terminal residual", dimensions=1
    )
    terminal_norm = terminal.get("returned_norm")
    if (
        not isinstance(terminal_norm, float)
        or not np.isfinite(terminal_norm)
        or not np.isclose(
            terminal_norm,
            np.linalg.norm(terminal_residual),
            rtol=1.0e-13,
            atol=1.0e-15,
        )
    ):
        raise NativeTrajectoryError("native terminal residual norm is inconsistent")
    if not terminal["rollback_taken"] and not np.array_equal(
        _array(updates[-1].get("state_after"), "last update state", dimensions=1),
        _array(terminal.get("returned_state"), "terminal state", dimensions=1),
    ):
        raise NativeTrajectoryError("native terminal state disagrees with final update")
    replay: list[dict[str, object]] = []
    for index, update in enumerate(updates):
        assessment = assessments.get(index)
        if assessment is None:
            raise NativeTrajectoryError("native pre-update assessment is missing")
        following = assessments.get(index + 1)
        outcome_event = following if following is not None else terminal
        replay.append(
            {
                "iteration_index": index,
                "state_before": _array(
                    update.get("state_before"), "replay state before", dimensions=1
                ).tolist(),
                "update": (
                    _array(update.get("initial_solve"), "initial solve", dimensions=1)
                    + _array(
                        update.get("refinement_correction"),
                        "refinement correction",
                        dimensions=1,
                    )
                ).tolist(),
                "state_after": _array(
                    update.get("state_after"), "replay state after", dimensions=1
                ).tolist(),
                "merit_before": float(assessment["residual_norm"]),
                "merit_after": (
                    None if following is None else float(following["residual_norm"])
                ),
                "state_assessed_after": following is not None,
                "backtracking_iteration_count": 0,
                "accepted": True,
                "stop_decision": index == len(updates) - 1,
                "status_code": terminal_status if index == len(updates) - 1 else 0,
                "counters": _native_replay_counters(outcome_event),
            }
        )

    first_following = assessments.get(1)
    one_step_converged = (
        first_following is not None
        and float(first_following["residual_norm"]) <= tolerance
    )
    document = {
        "schema_id": RAW_TRAJECTORY_SCHEMA_ID,
        "lane": "native",
        "parameter_sha256": _sha256(parameter_sha256, "parameter_sha256"),
        "specimen_sha256": _sha256(specimen_sha256, "specimen_sha256"),
        "input_bundle_sha256": _sha256(input_bundle_sha256, "input_bundle_sha256"),
        "solver_graph_sha256": _sha256(solver_graph_sha256, "solver_graph_sha256"),
        "source_sha256": _sha256(source_sha256, "source_sha256"),
        "one_step": _one_step(updates[0], one_step_converged),
        "short_replay": replay,
        "terminal": {
            "success": terminal["success"],
            "persist_solved_state": terminal["persist_solved_state"],
            "rollback_taken": terminal["rollback_taken"],
            "returned_state": _array(
                terminal.get("returned_state"), "terminal state", dimensions=1
            ).tolist(),
            "returned_residual": _array(
                terminal.get("returned_residual"),
                "terminal residual",
                dimensions=1,
            ).tolist(),
            "returned_jacobian": _array(
                terminal.get("returned_jacobian"),
                "terminal Jacobian",
                dimensions=2,
            ).tolist(),
            "returned_norm": float(terminal["returned_norm"]),
            "status_code": terminal_status,
            "counters": _native_replay_counters(terminal),
        },
    }
    return document


def produce_native_raw_trajectory(
    input_root: Path,
    candidate: np.ndarray,
    parameter_sha256: str,
    binding: NativeReferenceBinding,
    *,
    solver_graph_path: Path,
) -> dict[str, object]:
    """Run one changed-state native evaluation under the exact-Newton observer."""

    bundle, arrays = read_input_bundle(input_root)
    if bundle.case_id != SPEC.case_id or bundle.scale != "native_default":
        raise NativeTrajectoryError(
            "input bundle must be the native-default single-stage specimen"
        )
    if bundle.input_fingerprint != binding.input_fingerprint:
        raise NativeTrajectoryError("input fingerprint does not match binding")
    if _sha256_path(input_root / "input_bundle.json") != binding.input_bundle_sha256:
        raise NativeTrajectoryError("input bundle bytes do not match binding")
    if bundle.configuration_fingerprint != binding.configuration_fingerprint:
        raise NativeTrajectoryError("configuration fingerprint does not match binding")
    _validate_runtime_binding(binding)
    prepared = _prepare_native_variant_runtime(bundle, arrays, SPEC)
    events: list[Mapping[str, object]] = []
    with _boozer_exact_newton_observation_context(events.append):
        prepared.evaluate_candidate(candidate)
    tolerance = float(bundle.configuration["inner_tolerance"])
    return build_native_raw_trajectory(
        events,
        parameter_sha256=parameter_sha256,
        specimen_sha256=binding.specimen_sha256,
        input_bundle_sha256=binding.input_bundle_sha256,
        solver_graph_sha256=_canonical_solver_graph_sha256(solver_graph_path),
        source_sha256=binding.source_sha256,
        tolerance=tolerance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    parser.add_argument("--input-fingerprint", required=True)
    parser.add_argument("--input-bundle-sha256", required=True)
    parser.add_argument("--configuration-fingerprint", required=True)
    parser.add_argument("--specimen-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--solver-graph", type=Path, required=True)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--interpreter-path", required=True)
    parser.add_argument("--native-simsoptpp-path", required=True)
    parser.add_argument("--native-simsoptpp-sha256", required=True)
    parser.add_argument("--runtime-contract-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        parameter_sha256 = _sha256(args.parameter_sha256, "parameter_sha256")
        candidate = _canonical_candidate(args.candidate, parameter_sha256)
        runtime_contract = json.loads(args.runtime_contract_json)
        if not isinstance(runtime_contract, dict):
            raise NativeTrajectoryError("runtime contract must be a JSON object")
        binding = NativeReferenceBinding(
            input_bundle_sha256=args.input_bundle_sha256,
            input_fingerprint=args.input_fingerprint,
            configuration_fingerprint=args.configuration_fingerprint,
            specimen_sha256=args.specimen_sha256,
            source_sha256=args.source_sha256,
            runtime_identity_sha256=args.runtime_identity_sha256,
            interpreter_path=args.interpreter_path,
            native_simsoptpp_path=args.native_simsoptpp_path,
            native_simsoptpp_sha256=args.native_simsoptpp_sha256,
            runtime_contract=runtime_contract,
        )
        document = produce_native_raw_trajectory(
            args.input_root,
            candidate,
            parameter_sha256,
            binding,
            solver_graph_path=args.solver_graph,
        )
        write_raw_trajectory_document(args.output, document)
    except (OSError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"native trajectory failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
