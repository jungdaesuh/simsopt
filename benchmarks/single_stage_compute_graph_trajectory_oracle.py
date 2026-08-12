"""Raw-evidence-backed trajectory oracle for C1 and C2 canaries.

The oracle owns the raw trajectory schema and comparison policy.  Consumers
validate provenance, reload both raw inputs, recompute every comparison, and
require exact canonical equality with the persisted oracle artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Mapping, Sequence

from benchmarks.single_stage_compute_graph_phase0_receipt import (
    canonical_json_bytes,
)

RAW_TRAJECTORY_SCHEMA_ID: Final = "single-stage-compute-graph-raw-variant-trajectory-v1"
TRAJECTORY_ORACLE_SCHEMA_ID: Final = (
    "single-stage-compute-graph-variant-trajectory-oracle-v1"
)
TRAJECTORY_DERIVATION_VERSION: Final = "variant-trajectory-oracle-v1"

Variant = Literal["C1", "C2"]
Lane = Literal["C0", "C1", "C2", "native"]


class TrajectoryOracleError(RuntimeError):
    """Raw or derived trajectory evidence violates the oracle contract."""


@dataclass(frozen=True, slots=True)
class TrajectoryOracleIdentity:
    """Provenance that binds an oracle to one candidate and source pair."""

    variant: Variant
    parameter_sha256: str
    specimen_sha256: str
    input_bundle_sha256: str
    solver_graph_sha256: str
    one_step_reference_source_sha256: str
    trajectory_reference_source_sha256: str
    variant_source_sha256: str

    @property
    def trajectory_reference_lane(self) -> Literal["C0", "native"]:
        return "C0" if self.variant == "C1" else "native"

    def to_json(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "one_step_reference_lane": "native",
            "trajectory_reference_lane": self.trajectory_reference_lane,
            "parameter_sha256": self.parameter_sha256,
            "specimen_sha256": self.specimen_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "solver_graph_sha256": self.solver_graph_sha256,
            "one_step_reference_source_sha256": self.one_step_reference_source_sha256,
            "trajectory_reference_source_sha256": (
                self.trajectory_reference_source_sha256
            ),
            "variant_source_sha256": self.variant_source_sha256,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryTolerances:
    """Coordinatewise absolute and relative tolerances for numerical parity."""

    absolute: float
    relative: float

    def to_json(self) -> dict[str, float]:
        return {"absolute": self.absolute, "relative": self.relative}


_C1_TOLERANCES: Final = TrajectoryTolerances(absolute=1.0e-12, relative=1.0e-12)
_C2_TOLERANCES: Final = TrajectoryTolerances(absolute=1.0e-10, relative=1.0e-10)


def trajectory_tolerances_for_variant(variant: Variant) -> TrajectoryTolerances:
    """Return the immutable numerical policy for one trajectory contract."""

    if variant == "C1":
        return _C1_TOLERANCES
    if variant == "C2":
        return _C2_TOLERANCES
    raise TrajectoryOracleError("variant must be C1 or C2")


@dataclass(frozen=True, slots=True)
class RawTrajectoryBinding:
    """Runner-owned relative path and digest for one raw trajectory input."""

    relative_path: str
    sha256: str

    def to_json(self, role: str) -> dict[str, str]:
        return {
            "role": role,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryRawBindings:
    """Independent bindings for the three raw inputs consumed by the oracle."""

    one_step_reference: RawTrajectoryBinding
    trajectory_reference: RawTrajectoryBinding
    variant: RawTrajectoryBinding

    def to_json(self) -> list[dict[str, str]]:
        return [
            self.one_step_reference.to_json("one_step_reference"),
            self.trajectory_reference.to_json("trajectory_reference"),
            self.variant.to_json("variant"),
        ]


@dataclass(frozen=True, slots=True)
class TrajectoryOracleAudit:
    """Result of independently rebuilding a persisted trajectory oracle."""

    variant: Variant
    parameter_sha256: str
    parity_passed: bool
    one_step_passed: bool
    short_replay_passed: bool
    terminal_passed: bool


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TrajectoryOracleError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TrajectoryOracleError(f"{context} must be an array")
    return value


def _exact_keys(
    document: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(document) != expected:
        raise TrajectoryOracleError(f"{context} has an invalid field set")


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrajectoryOracleError(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise TrajectoryOracleError(f"{context} must be a lowercase SHA-256")
    return digest


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrajectoryOracleError(f"{context} must be a nonnegative integer")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise TrajectoryOracleError(f"{context} must be boolean")
    return value


def _finite_float(value: object, context: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise TrajectoryOracleError(f"{context} must be a finite float")
    return value


def _optional_finite_float(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, context)


def _float_vector(value: object, context: str) -> tuple[float, ...]:
    values = _sequence(value, context)
    if not values:
        raise TrajectoryOracleError(f"{context} must be non-empty")
    return tuple(
        _finite_float(item, f"{context}[{index}]") for index, item in enumerate(values)
    )


def _float_matrix(value: object, context: str) -> tuple[tuple[float, ...], ...]:
    rows = _sequence(value, context)
    if not rows:
        raise TrajectoryOracleError(f"{context} must be non-empty")
    matrix = tuple(
        _float_vector(row, f"{context}[{index}]") for index, row in enumerate(rows)
    )
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise TrajectoryOracleError(f"{context} rows have inconsistent dimensions")
    return matrix


def _counters(value: object, context: str) -> dict[str, int]:
    document = _mapping(value, context)
    if not document:
        raise TrajectoryOracleError(f"{context} must be non-empty")
    counters: dict[str, int] = {}
    for name in sorted(document):
        if not isinstance(name, str) or not name:
            raise TrajectoryOracleError(f"{context} keys must be non-empty strings")
        counters[name] = _integer(document[name], f"{context}.{name}")
    return counters


def _load_canonical_object(path: Path, context: str) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                TrajectoryOracleError(
                    f"{context} contains non-finite constant {constant}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TrajectoryOracleError(f"{context} is not valid JSON") from error
    document = _mapping(value, context)
    if raw != canonical_json_bytes(document):
        raise TrajectoryOracleError(f"{context} is not canonical JSON")
    return document


def _validate_identity(identity: TrajectoryOracleIdentity) -> None:
    if identity.variant not in ("C1", "C2"):
        raise TrajectoryOracleError("variant must be C1 or C2")
    for field in (
        "parameter_sha256",
        "specimen_sha256",
        "input_bundle_sha256",
        "solver_graph_sha256",
        "one_step_reference_source_sha256",
        "trajectory_reference_source_sha256",
        "variant_source_sha256",
    ):
        _sha256(getattr(identity, field), f"identity.{field}")


def _validate_tolerances(tolerances: TrajectoryTolerances) -> None:
    for field in ("absolute", "relative"):
        value = getattr(tolerances, field)
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise TrajectoryOracleError(
                f"tolerances.{field} must be a finite nonnegative float"
            )


_RAW_FIELDS = frozenset(
    {
        "schema_id",
        "lane",
        "parameter_sha256",
        "specimen_sha256",
        "input_bundle_sha256",
        "solver_graph_sha256",
        "source_sha256",
        "one_step",
        "short_replay",
        "terminal",
    }
)
_ONE_STEP_FIELDS = frozenset(
    {
        "initial_state",
        "residual",
        "jacobian",
        "initial_solve",
        "refinement_rhs",
        "refinement_correction",
        "correction_step",
        "refined_residual",
        "next_state",
        "converged",
        "numerical_failure",
        "status_code",
        "counters",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "success",
        "persist_solved_state",
        "rollback_taken",
        "returned_state",
        "returned_residual",
        "returned_jacobian",
        "returned_norm",
        "status_code",
        "counters",
    }
)
_REPLAY_STEP_FIELDS = frozenset(
    {
        "iteration_index",
        "state_before",
        "update",
        "state_after",
        "merit_before",
        "merit_after",
        "state_assessed_after",
        "backtracking_iteration_count",
        "accepted",
        "stop_decision",
        "status_code",
        "counters",
    }
)
_ONE_STEP_COUNTER_FIELDS = frozenset(
    {
        "residual_evaluation_count",
        "dense_materialization_count",
        "lu_factorization_count",
        "lu_solve_count",
        "refinement_correction_count",
    }
)
_C1_REPLAY_COUNTER_FIELDS = frozenset(
    {
        "residual_evaluation_count",
        "attempted_iteration_count",
        "accepted_update_count",
    }
)
_C2_REPLAY_COUNTER_FIELDS = frozenset(
    {
        "residual_evaluation_count",
        "attempted_iteration_count",
        "applied_update_count",
        "assessed_state_count",
        "rollback_recompute_count",
    }
)


def _raw_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)


def _require_raw_vector_close(
    observed: Sequence[float], expected: Sequence[float], context: str
) -> None:
    if len(observed) != len(expected) or not all(
        _raw_close(left, right) for left, right in zip(observed, expected)
    ):
        raise TrajectoryOracleError(f"{context} is algebraically inconsistent")


def _matrix_vector(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    return [
        math.fsum(coefficient * value for coefficient, value in zip(row, vector))
        for row in matrix
    ]


def _validate_one_step(
    value: object, context: str, counter_fields: frozenset[str]
) -> dict[str, object]:
    step = _mapping(value, context)
    _exact_keys(step, _ONE_STEP_FIELDS, context)
    normalized = {
        "initial_state": list(
            _float_vector(step.get("initial_state"), f"{context}.initial_state")
        ),
        "residual": list(_float_vector(step.get("residual"), f"{context}.residual")),
        "jacobian": [
            list(row)
            for row in _float_matrix(step.get("jacobian"), f"{context}.jacobian")
        ],
        "initial_solve": list(
            _float_vector(step.get("initial_solve"), f"{context}.initial_solve")
        ),
        "refinement_rhs": list(
            _float_vector(step.get("refinement_rhs"), f"{context}.refinement_rhs")
        ),
        "refinement_correction": list(
            _float_vector(
                step.get("refinement_correction"),
                f"{context}.refinement_correction",
            )
        ),
        "correction_step": list(
            _float_vector(step.get("correction_step"), f"{context}.correction_step")
        ),
        "refined_residual": list(
            _float_vector(step.get("refined_residual"), f"{context}.refined_residual")
        ),
        "next_state": list(
            _float_vector(step.get("next_state"), f"{context}.next_state")
        ),
        "converged": _boolean(step.get("converged"), f"{context}.converged"),
        "numerical_failure": _boolean(
            step.get("numerical_failure"), f"{context}.numerical_failure"
        ),
        "status_code": _integer(step.get("status_code"), f"{context}.status_code"),
        "counters": _counters(step.get("counters"), f"{context}.counters"),
    }
    if frozenset(normalized["counters"]) != counter_fields:
        raise TrajectoryOracleError(f"{context}.counters has an invalid field set")
    if len(normalized["residual"]) != len(normalized["refined_residual"]):
        raise TrajectoryOracleError(f"{context} residual dimensions differ")
    if not (
        len(normalized["initial_state"])
        == len(normalized["initial_solve"])
        == len(normalized["refinement_correction"])
        == len(normalized["correction_step"])
        == len(normalized["next_state"])
    ):
        raise TrajectoryOracleError(f"{context} state dimensions differ")
    if not (
        len(normalized["jacobian"])
        == len(normalized["residual"])
        == len(normalized["refinement_rhs"])
        and all(
            len(row) == len(normalized["initial_state"])
            for row in normalized["jacobian"]
        )
    ):
        raise TrajectoryOracleError(f"{context} Jacobian dimensions differ")
    correction = [
        initial + refinement
        for initial, refinement in zip(
            normalized["initial_solve"], normalized["refinement_correction"]
        )
    ]
    initial_product = _matrix_vector(
        normalized["jacobian"], normalized["initial_solve"]
    )
    correction_product = _matrix_vector(normalized["jacobian"], correction)
    _require_raw_vector_close(
        normalized["refinement_rhs"],
        [
            residual - product
            for residual, product in zip(normalized["residual"], initial_product)
        ],
        f"{context}.refinement_rhs",
    )
    _require_raw_vector_close(
        normalized["correction_step"], correction, f"{context}.correction_step"
    )
    _require_raw_vector_close(
        normalized["refined_residual"],
        [
            residual - product
            for residual, product in zip(normalized["residual"], correction_product)
        ],
        f"{context}.refined_residual",
    )
    _require_raw_vector_close(
        normalized["next_state"],
        [
            state - delta
            for state, delta in zip(normalized["initial_state"], correction)
        ],
        f"{context}.next_state",
    )
    return normalized


def _validate_replay_step(
    value: object, context: str, counter_fields: frozenset[str]
) -> dict[str, object]:
    step = _mapping(value, context)
    _exact_keys(step, _REPLAY_STEP_FIELDS, context)
    normalized = {
        "iteration_index": _integer(
            step.get("iteration_index"), f"{context}.iteration_index"
        ),
        "state_before": list(
            _float_vector(step.get("state_before"), f"{context}.state_before")
        ),
        "update": list(_float_vector(step.get("update"), f"{context}.update")),
        "state_after": list(
            _float_vector(step.get("state_after"), f"{context}.state_after")
        ),
        "merit_before": _finite_float(
            step.get("merit_before"), f"{context}.merit_before"
        ),
        "merit_after": _optional_finite_float(
            step.get("merit_after"), f"{context}.merit_after"
        ),
        "state_assessed_after": _boolean(
            step.get("state_assessed_after"), f"{context}.state_assessed_after"
        ),
        "backtracking_iteration_count": _integer(
            step.get("backtracking_iteration_count"),
            f"{context}.backtracking_iteration_count",
        ),
        "accepted": _boolean(step.get("accepted"), f"{context}.accepted"),
        "stop_decision": _boolean(
            step.get("stop_decision"), f"{context}.stop_decision"
        ),
        "status_code": _integer(step.get("status_code"), f"{context}.status_code"),
        "counters": _counters(step.get("counters"), f"{context}.counters"),
    }
    if frozenset(normalized["counters"]) != counter_fields:
        raise TrajectoryOracleError(f"{context}.counters has an invalid field set")
    if normalized["state_assessed_after"] != (normalized["merit_after"] is not None):
        raise TrajectoryOracleError(
            f"{context} assessed-state flag and merit_after disagree"
        )
    return normalized


def _validate_terminal(
    value: object, context: str, counter_fields: frozenset[str]
) -> dict[str, object]:
    terminal = _mapping(value, context)
    _exact_keys(terminal, _TERMINAL_FIELDS, context)
    normalized = {
        "success": _boolean(terminal.get("success"), f"{context}.success"),
        "persist_solved_state": _boolean(
            terminal.get("persist_solved_state"),
            f"{context}.persist_solved_state",
        ),
        "rollback_taken": _boolean(
            terminal.get("rollback_taken"), f"{context}.rollback_taken"
        ),
        "returned_state": list(
            _float_vector(terminal.get("returned_state"), f"{context}.returned_state")
        ),
        "returned_residual": list(
            _float_vector(
                terminal.get("returned_residual"), f"{context}.returned_residual"
            )
        ),
        "returned_jacobian": [
            list(row)
            for row in _float_matrix(
                terminal.get("returned_jacobian"), f"{context}.returned_jacobian"
            )
        ],
        "returned_norm": _finite_float(
            terminal.get("returned_norm"), f"{context}.returned_norm"
        ),
        "status_code": _integer(terminal.get("status_code"), f"{context}.status_code"),
        "counters": _counters(terminal.get("counters"), f"{context}.counters"),
    }
    if frozenset(normalized["counters"]) != counter_fields:
        raise TrajectoryOracleError(f"{context}.counters has an invalid field set")
    if normalized["persist_solved_state"] == normalized["rollback_taken"]:
        raise TrajectoryOracleError(f"{context} persistence/rollback flags disagree")
    if not (
        len(normalized["returned_jacobian"]) == len(normalized["returned_residual"])
        and all(
            len(row) == len(normalized["returned_state"])
            for row in normalized["returned_jacobian"]
        )
    ):
        raise TrajectoryOracleError(f"{context} returned Jacobian dimensions differ")
    returned_norm = math.sqrt(
        math.fsum(value * value for value in normalized["returned_residual"])
    )
    if not _raw_close(normalized["returned_norm"], returned_norm):
        raise TrajectoryOracleError(f"{context}.returned_norm is inconsistent")
    return normalized


def validate_raw_trajectory_document(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize one evaluator-produced raw trajectory document."""

    _exact_keys(document, _RAW_FIELDS, "raw trajectory")
    if document.get("schema_id") != RAW_TRAJECTORY_SCHEMA_ID:
        raise TrajectoryOracleError("raw trajectory schema_id is unsupported")
    lane = _string(document.get("lane"), "raw trajectory.lane")
    if lane not in ("C0", "C1", "C2", "native"):
        raise TrajectoryOracleError("raw trajectory lane is unsupported")
    replay_counter_fields = (
        _C1_REPLAY_COUNTER_FIELDS if lane in ("C0", "C1") else _C2_REPLAY_COUNTER_FIELDS
    )
    short_replay_values = _sequence(
        document.get("short_replay"), "raw trajectory.short_replay"
    )
    if len(short_replay_values) < 2:
        raise TrajectoryOracleError("short replay must contain at least two steps")
    short_replay = [
        _validate_replay_step(
            value,
            f"raw trajectory.short_replay[{index}]",
            replay_counter_fields,
        )
        for index, value in enumerate(short_replay_values)
    ]
    if tuple(step["iteration_index"] for step in short_replay) != tuple(
        range(len(short_replay))
    ):
        raise TrajectoryOracleError("short replay iteration indices must be contiguous")
    state_size = len(short_replay[0]["state_before"])
    for index, step in enumerate(short_replay):
        if not all(
            len(step[field]) == state_size
            for field in ("state_before", "update", "state_after")
        ):
            raise TrajectoryOracleError(
                f"short replay step {index} has inconsistent state dimensions"
            )
        if index and step["state_before"] != short_replay[index - 1]["state_after"]:
            raise TrajectoryOracleError("short replay state chain is discontinuous")
        if index:
            previous = short_replay[index - 1]
            expected_merit = (
                previous["merit_before"]
                if previous["merit_after"] is None
                else previous["merit_after"]
            )
            if step["merit_before"] != expected_merit:
                raise TrajectoryOracleError("short replay merit chain is discontinuous")
        if step["accepted"]:
            expected_state_after = [
                before - update
                for before, update in zip(step["state_before"], step["update"])
            ]
        else:
            expected_state_after = step["state_before"]
        if step["state_after"] != expected_state_after:
            raise TrajectoryOracleError("short replay update/state relation is invalid")
        if step["stop_decision"] != (index == len(short_replay) - 1):
            raise TrajectoryOracleError(
                "short replay must stop exactly at its final step"
            )
        if lane in ("native", "C2") and (
            step["backtracking_iteration_count"] != 0 or not step["accepted"]
        ):
            raise TrajectoryOracleError(
                "native/C2 replay must use accepted full steps without backtracking"
            )
    if lane == "C0":
        if document.get("one_step") is not None:
            raise TrajectoryOracleError(
                "C0 raw trajectory must not fabricate one-step data"
            )
        one_step = None
    else:
        one_step = _validate_one_step(
            document.get("one_step"),
            "raw trajectory.one_step",
            _ONE_STEP_COUNTER_FIELDS,
        )
        if len(one_step["next_state"]) != state_size:
            raise TrajectoryOracleError("one-step and replay state dimensions differ")
        if (
            one_step["initial_state"] != short_replay[0]["state_before"]
            or one_step["next_state"] != short_replay[0]["state_after"]
        ):
            raise TrajectoryOracleError(
                "one-step result is not linked to replay step zero"
            )
    terminal = _validate_terminal(
        document.get("terminal"), "raw trajectory.terminal", replay_counter_fields
    )
    accepted_count = 0
    previous_residual_count = 0
    for index, step in enumerate(short_replay):
        counters = step["counters"]
        accepted_count += int(step["accepted"])
        if counters["attempted_iteration_count"] != index + 1:
            raise TrajectoryOracleError("replay attempted-iteration count is invalid")
        if counters["residual_evaluation_count"] < previous_residual_count:
            raise TrajectoryOracleError("replay residual-evaluation count regressed")
        previous_residual_count = counters["residual_evaluation_count"]
        if lane in ("C0", "C1"):
            if counters["accepted_update_count"] != accepted_count:
                raise TrajectoryOracleError("replay accepted-update count is invalid")
        elif (
            counters["applied_update_count"] != index + 1
            or counters["assessed_state_count"]
            != index + 1 + int(step["state_assessed_after"])
            or counters["rollback_recompute_count"] != 0
        ):
            raise TrajectoryOracleError("native/C2 replay counters are invalid")
    terminal_counters = terminal["counters"]
    if (
        terminal_counters["attempted_iteration_count"] != len(short_replay)
        or terminal_counters["residual_evaluation_count"] < previous_residual_count
    ):
        raise TrajectoryOracleError("terminal counters are inconsistent")
    if lane in ("C0", "C1"):
        if terminal_counters["accepted_update_count"] != accepted_count:
            raise TrajectoryOracleError("terminal accepted-update count is invalid")
    elif terminal_counters["applied_update_count"] != len(
        short_replay
    ) or terminal_counters["rollback_recompute_count"] != int(
        terminal["rollback_taken"]
    ):
        raise TrajectoryOracleError("native/C2 terminal counters are invalid")
    if (
        not terminal["rollback_taken"]
        and terminal["returned_state"] != short_replay[-1]["state_after"]
    ):
        raise TrajectoryOracleError("terminal state differs from replay terminal state")
    return {
        "schema_id": RAW_TRAJECTORY_SCHEMA_ID,
        "lane": lane,
        "parameter_sha256": _sha256(
            document.get("parameter_sha256"), "raw trajectory.parameter_sha256"
        ),
        "specimen_sha256": _sha256(
            document.get("specimen_sha256"), "raw trajectory.specimen_sha256"
        ),
        "input_bundle_sha256": _sha256(
            document.get("input_bundle_sha256"),
            "raw trajectory.input_bundle_sha256",
        ),
        "solver_graph_sha256": _sha256(
            document.get("solver_graph_sha256"),
            "raw trajectory.solver_graph_sha256",
        ),
        "source_sha256": _sha256(
            document.get("source_sha256"), "raw trajectory.source_sha256"
        ),
        "one_step": one_step,
        "short_replay": short_replay,
        "terminal": terminal,
    }


def write_raw_trajectory_document(path: Path, document: Mapping[str, object]) -> str:
    """Validate and exclusively write one canonical raw trajectory document."""

    normalized = validate_raw_trajectory_document(document)
    payload = canonical_json_bytes(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _relative_bound_path(path: Path, artifact_root: Path, context: str) -> str:
    resolved = path.resolve()
    root = artifact_root.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TrajectoryOracleError(f"{context} must be a file inside artifact_root")
    return resolved.relative_to(root).as_posix()


def _bound_path(relative_path: object, artifact_root: Path, context: str) -> Path:
    relative = PurePosixPath(_string(relative_path, context))
    if relative.is_absolute() or ".." in relative.parts:
        raise TrajectoryOracleError(f"{context} must be a safe relative path")
    path = (artifact_root.resolve() / Path(*relative.parts)).resolve()
    if not path.is_relative_to(artifact_root.resolve()) or not path.is_file():
        raise TrajectoryOracleError(f"{context} is outside artifact_root or missing")
    return path


def bind_raw_trajectory_inputs(
    *,
    artifact_root: Path,
    one_step_reference_raw_path: Path,
    trajectory_reference_raw_path: Path,
    variant_raw_path: Path,
) -> TrajectoryRawBindings:
    """Create runner-owned path/SHA bindings for the oracle's raw inputs."""

    def bind(path: Path, context: str) -> RawTrajectoryBinding:
        relative_path = _relative_bound_path(path, artifact_root, context)
        return RawTrajectoryBinding(
            relative_path=relative_path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    return TrajectoryRawBindings(
        one_step_reference=bind(
            one_step_reference_raw_path, "one_step_reference_raw_path"
        ),
        trajectory_reference=bind(
            trajectory_reference_raw_path, "trajectory_reference_raw_path"
        ),
        variant=bind(variant_raw_path, "variant_raw_path"),
    )


def _compare_vector(
    field: str,
    reference: Sequence[float],
    candidate: Sequence[float],
    tolerances: TrajectoryTolerances,
) -> dict[str, object]:
    if len(reference) != len(candidate):
        return {
            "field": field,
            "element_count": len(candidate),
            "max_absolute_error": None,
            "max_scaled_error": None,
            "passed": False,
        }
    absolute_errors = tuple(
        abs(left - right) for left, right in zip(reference, candidate)
    )
    scaled_errors = tuple(
        error
        / max(
            tolerances.absolute + tolerances.relative * abs(left),
            float.fromhex("0x0.0000000000001p-1022"),
        )
        for left, error in zip(reference, absolute_errors)
    )
    passed = all(
        error <= tolerances.absolute + tolerances.relative * abs(left)
        for left, error in zip(reference, absolute_errors)
    )
    return {
        "field": field,
        "element_count": len(candidate),
        "max_absolute_error": max(absolute_errors, default=0.0),
        "max_scaled_error": max(scaled_errors, default=0.0),
        "passed": passed,
    }


def _compare_float(
    field: str,
    reference: float,
    candidate: float,
    tolerances: TrajectoryTolerances,
) -> dict[str, object]:
    return _compare_vector(field, (reference,), (candidate,), tolerances)


def _flatten_matrix(value: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(item for row in value for item in row)


def _exact_comparison(
    field: str, reference: object, candidate: object
) -> dict[str, object]:
    return {
        "field": field,
        "passed": type(reference) is type(candidate) and reference == candidate,
    }


def _comparison_passed(comparison: Mapping[str, object]) -> bool:
    return comparison.get("passed") is True


def _compare_one_step(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    tolerances: TrajectoryTolerances,
) -> dict[str, object]:
    numerical = [
        _compare_vector(
            field,
            reference[field],
            candidate[field],
            tolerances,
        )
        for field in (
            "initial_state",
            "residual",
            "initial_solve",
            "refinement_rhs",
            "refinement_correction",
            "correction_step",
            "refined_residual",
            "next_state",
        )
    ]
    numerical.append(
        _compare_vector(
            "jacobian",
            _flatten_matrix(reference["jacobian"]),
            _flatten_matrix(candidate["jacobian"]),
            tolerances,
        )
    )
    decisions = [
        _exact_comparison(field, reference[field], candidate[field])
        for field in (
            "converged",
            "numerical_failure",
            "status_code",
        )
    ]
    return {
        "numerical": numerical,
        "decisions": decisions,
        "telemetry": {
            "reference_counters": reference["counters"],
            "candidate_counters": candidate["counters"],
        },
        "passed": all(map(_comparison_passed, (*numerical, *decisions))),
    }


def _compare_replay_step(
    variant: Variant,
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    tolerances: TrajectoryTolerances,
) -> dict[str, object]:
    numerical_fields = (
        "state_before",
        "update",
        "state_after",
    )
    numerical = [
        _compare_vector(field, reference[field], candidate[field], tolerances)
        for field in numerical_fields
    ]
    numerical.append(
        _compare_float(
            "merit_before",
            reference["merit_before"],
            candidate["merit_before"],
            tolerances,
        )
    )
    reference_merit_after = reference["merit_after"]
    candidate_merit_after = candidate["merit_after"]
    merit_after_comparison = (
        _compare_float(
            "merit_after",
            reference_merit_after,
            candidate_merit_after,
            tolerances,
        )
        if isinstance(reference_merit_after, float)
        and isinstance(candidate_merit_after, float)
        else _exact_comparison(
            "merit_after", reference_merit_after, candidate_merit_after
        )
    )
    numerical.append(merit_after_comparison)
    exact_fields = (
        (
            "iteration_index",
            "backtracking_iteration_count",
            "accepted",
            "stop_decision",
            "state_assessed_after",
            "status_code",
        )
        if variant == "C1"
        else (
            "iteration_index",
            "accepted",
            "backtracking_iteration_count",
            "stop_decision",
            "state_assessed_after",
            "status_code",
        )
    )
    decisions = [
        _exact_comparison(field, reference[field], candidate[field])
        for field in exact_fields
    ]
    counter_fields = (
        ("attempted_iteration_count", "accepted_update_count")
        if variant == "C1"
        else tuple(sorted(_C2_REPLAY_COUNTER_FIELDS))
    )
    decisions.extend(
        _exact_comparison(
            f"counters.{field}",
            reference["counters"][field],
            candidate["counters"][field],
        )
        for field in counter_fields
    )
    telemetry = {
        "reference_residual_evaluation_count": reference["counters"][
            "residual_evaluation_count"
        ],
        "candidate_residual_evaluation_count": candidate["counters"][
            "residual_evaluation_count"
        ],
    }
    return {
        "iteration_index": candidate["iteration_index"],
        "numerical": numerical,
        "decisions": decisions,
        "telemetry": telemetry,
        "passed": all(map(_comparison_passed, (*numerical, *decisions))),
    }


def _compare_terminal(
    variant: Variant,
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    tolerances: TrajectoryTolerances,
) -> dict[str, object]:
    numerical = [
        _compare_vector(field, reference[field], candidate[field], tolerances)
        for field in ("returned_state", "returned_residual")
    ]
    numerical.extend(
        (
            _compare_vector(
                "returned_jacobian",
                _flatten_matrix(reference["returned_jacobian"]),
                _flatten_matrix(candidate["returned_jacobian"]),
                tolerances,
            ),
            _compare_float(
                "returned_norm",
                reference["returned_norm"],
                candidate["returned_norm"],
                tolerances,
            ),
        )
    )
    decisions = [
        _exact_comparison(field, reference[field], candidate[field])
        for field in (
            "success",
            "persist_solved_state",
            "rollback_taken",
            "status_code",
        )
    ]
    counter_fields = (
        ("attempted_iteration_count", "accepted_update_count")
        if variant == "C1"
        else (
            "attempted_iteration_count",
            "applied_update_count",
            "assessed_state_count",
            "rollback_recompute_count",
        )
    )
    decisions.extend(
        _exact_comparison(
            f"counters.{field}",
            reference["counters"][field],
            candidate["counters"][field],
        )
        for field in counter_fields
    )
    return {
        "numerical": numerical,
        "decisions": decisions,
        "telemetry": {
            "reference_residual_evaluation_count": reference["counters"][
                "residual_evaluation_count"
            ],
            "candidate_residual_evaluation_count": candidate["counters"][
                "residual_evaluation_count"
            ],
        },
        "passed": all(map(_comparison_passed, (*numerical, *decisions))),
    }


def _validate_raw_binding(
    raw: Mapping[str, object],
    identity: TrajectoryOracleIdentity,
    *,
    expected_lane: Lane,
    expected_source_sha256: str,
) -> None:
    expected = {
        "lane": expected_lane,
        "parameter_sha256": identity.parameter_sha256,
        "specimen_sha256": identity.specimen_sha256,
        "input_bundle_sha256": identity.input_bundle_sha256,
        "solver_graph_sha256": identity.solver_graph_sha256,
        "source_sha256": expected_source_sha256,
    }
    for field, expected_value in expected.items():
        if raw[field] != expected_value:
            raise TrajectoryOracleError(f"raw trajectory {field} differs from identity")


def build_variant_trajectory_oracle(
    *,
    identity: TrajectoryOracleIdentity,
    artifact_root: Path,
    one_step_reference_raw_path: Path,
    trajectory_reference_raw_path: Path,
    variant_raw_path: Path,
) -> dict[str, object]:
    """Rebuild one C1/C2 oracle directly from canonical raw trajectories."""

    _validate_identity(identity)
    tolerances = trajectory_tolerances_for_variant(identity.variant)
    _validate_tolerances(tolerances)
    raw_bindings = bind_raw_trajectory_inputs(
        artifact_root=artifact_root,
        one_step_reference_raw_path=one_step_reference_raw_path,
        trajectory_reference_raw_path=trajectory_reference_raw_path,
        variant_raw_path=variant_raw_path,
    )
    one_step_reference = validate_raw_trajectory_document(
        _load_canonical_object(
            one_step_reference_raw_path, "one-step reference raw trajectory"
        )
    )
    trajectory_reference = validate_raw_trajectory_document(
        _load_canonical_object(
            trajectory_reference_raw_path, "trajectory reference raw trajectory"
        )
    )
    candidate = validate_raw_trajectory_document(
        _load_canonical_object(variant_raw_path, "variant raw trajectory")
    )
    _validate_raw_binding(
        one_step_reference,
        identity,
        expected_lane="native",
        expected_source_sha256=identity.one_step_reference_source_sha256,
    )
    _validate_raw_binding(
        trajectory_reference,
        identity,
        expected_lane=identity.trajectory_reference_lane,
        expected_source_sha256=identity.trajectory_reference_source_sha256,
    )
    _validate_raw_binding(
        candidate,
        identity,
        expected_lane=identity.variant,
        expected_source_sha256=identity.variant_source_sha256,
    )
    reference_replay = trajectory_reference["short_replay"]
    candidate_replay = candidate["short_replay"]
    if len(reference_replay) != len(candidate_replay):
        raise TrajectoryOracleError("reference and variant replay lengths differ")
    one_step = _compare_one_step(
        one_step_reference["one_step"], candidate["one_step"], tolerances
    )
    short_replay_steps = [
        _compare_replay_step(
            variant=identity.variant,
            reference=left,
            candidate=right,
            tolerances=tolerances,
        )
        for left, right in zip(reference_replay, candidate_replay)
    ]
    short_replay = {
        "step_count": len(short_replay_steps),
        "steps": short_replay_steps,
        "passed": all(step["passed"] is True for step in short_replay_steps),
    }
    terminal = _compare_terminal(
        identity.variant,
        trajectory_reference["terminal"],
        candidate["terminal"],
        tolerances,
    )
    parity_passed = (
        one_step["passed"] is True
        and short_replay["passed"] is True
        and terminal["passed"] is True
    )
    return {
        "schema_id": TRAJECTORY_ORACLE_SCHEMA_ID,
        "state": "PRODUCED",
        "derivation_version": TRAJECTORY_DERIVATION_VERSION,
        "identity": identity.to_json(),
        "tolerances": tolerances.to_json(),
        "raw_inputs": raw_bindings.to_json(),
        "comparison": {
            "one_step": one_step,
            "short_replay": short_replay,
            "terminal": terminal,
            "parity_passed": parity_passed,
        },
        "promotion_eligible": parity_passed,
    }


def write_variant_trajectory_oracle(path: Path, document: Mapping[str, object]) -> str:
    """Exclusively write a producer-built canonical oracle artifact."""

    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def validate_variant_trajectory_oracle(
    path: Path,
    *,
    artifact_root: Path,
    expected_identity: TrajectoryOracleIdentity,
    expected_raw_bindings: TrajectoryRawBindings,
) -> TrajectoryOracleAudit:
    """Reload raw inputs and reject any oracle not equal to a fresh derivation."""

    document = _load_canonical_object(path, "trajectory oracle")
    _exact_keys(
        document,
        frozenset(
            {
                "schema_id",
                "state",
                "derivation_version",
                "identity",
                "tolerances",
                "raw_inputs",
                "comparison",
                "promotion_eligible",
            }
        ),
        "trajectory oracle",
    )
    if (
        document.get("schema_id") != TRAJECTORY_ORACLE_SCHEMA_ID
        or document.get("state") != "PRODUCED"
        or document.get("derivation_version") != TRAJECTORY_DERIVATION_VERSION
        or document.get("identity") != expected_identity.to_json()
    ):
        raise TrajectoryOracleError("trajectory oracle schema or identity is invalid")
    tolerance_document = _mapping(document.get("tolerances"), "oracle tolerances")
    _exact_keys(
        tolerance_document, frozenset({"absolute", "relative"}), "oracle tolerances"
    )
    persisted_tolerances = TrajectoryTolerances(
        absolute=_finite_float(
            tolerance_document.get("absolute"), "absolute tolerance"
        ),
        relative=_finite_float(
            tolerance_document.get("relative"), "relative tolerance"
        ),
    )
    expected_tolerances = trajectory_tolerances_for_variant(expected_identity.variant)
    if persisted_tolerances != expected_tolerances:
        raise TrajectoryOracleError("trajectory oracle tolerances differ from policy")
    raw_inputs = _sequence(document.get("raw_inputs"), "oracle raw_inputs")
    if len(raw_inputs) != 3:
        raise TrajectoryOracleError("trajectory oracle must bind three raw inputs")
    if raw_inputs != expected_raw_bindings.to_json():
        raise TrajectoryOracleError("trajectory raw bindings differ from runner policy")
    bound: dict[str, Path] = {}
    for index, raw_input_value in enumerate(raw_inputs):
        raw_input = _mapping(raw_input_value, f"raw_inputs[{index}]")
        _exact_keys(
            raw_input,
            frozenset({"role", "relative_path", "sha256"}),
            f"raw_inputs[{index}]",
        )
        role = _string(raw_input.get("role"), f"raw_inputs[{index}].role")
        if (
            role
            not in (
                "one_step_reference",
                "trajectory_reference",
                "variant",
            )
            or role in bound
        ):
            raise TrajectoryOracleError("raw input roles are invalid or duplicated")
        raw_path = _bound_path(
            raw_input.get("relative_path"), artifact_root, f"raw_inputs[{index}]"
        )
        expected_sha256 = _sha256(
            raw_input.get("sha256"), f"raw_inputs[{index}].sha256"
        )
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != expected_sha256:
            raise TrajectoryOracleError(f"{role} raw trajectory hash mismatch")
        bound[role] = raw_path
    rebuilt = build_variant_trajectory_oracle(
        identity=expected_identity,
        artifact_root=artifact_root,
        one_step_reference_raw_path=bound["one_step_reference"],
        trajectory_reference_raw_path=bound["trajectory_reference"],
        variant_raw_path=bound["variant"],
    )
    if rebuilt != document:
        raise TrajectoryOracleError(
            "trajectory oracle differs from raw-recomputed evidence"
        )
    comparison = _mapping(document["comparison"], "oracle comparison")
    one_step = _mapping(comparison.get("one_step"), "one-step comparison")
    short_replay = _mapping(comparison.get("short_replay"), "short-replay comparison")
    terminal = _mapping(comparison.get("terminal"), "terminal comparison")
    parity_passed = comparison.get("parity_passed") is True
    if document.get("promotion_eligible") is not parity_passed:
        raise TrajectoryOracleError("promotion eligibility differs from recomputation")
    return TrajectoryOracleAudit(
        variant=expected_identity.variant,
        parameter_sha256=expected_identity.parameter_sha256,
        parity_passed=parity_passed,
        one_step_passed=one_step.get("passed") is True,
        short_replay_passed=short_replay.get("passed") is True,
        terminal_passed=terminal.get("passed") is True,
    )


def require_passing_variant_trajectory_oracle(
    path: Path,
    *,
    artifact_root: Path,
    expected_identity: TrajectoryOracleIdentity,
    expected_raw_bindings: TrajectoryRawBindings,
) -> TrajectoryOracleAudit:
    """Validate an oracle and require both one-step and replay parity."""

    audit = validate_variant_trajectory_oracle(
        path,
        artifact_root=artifact_root,
        expected_identity=expected_identity,
        expected_raw_bindings=expected_raw_bindings,
    )
    if not audit.parity_passed:
        raise TrajectoryOracleError("trajectory oracle is valid but non-promoting")
    return audit


__all__ = [
    "RAW_TRAJECTORY_SCHEMA_ID",
    "TRAJECTORY_DERIVATION_VERSION",
    "TRAJECTORY_ORACLE_SCHEMA_ID",
    "RawTrajectoryBinding",
    "TrajectoryOracleAudit",
    "TrajectoryOracleError",
    "TrajectoryOracleIdentity",
    "TrajectoryRawBindings",
    "TrajectoryTolerances",
    "bind_raw_trajectory_inputs",
    "build_variant_trajectory_oracle",
    "require_passing_variant_trajectory_oracle",
    "trajectory_tolerances_for_variant",
    "validate_raw_trajectory_document",
    "validate_variant_trajectory_oracle",
    "write_raw_trajectory_document",
    "write_variant_trajectory_oracle",
]
