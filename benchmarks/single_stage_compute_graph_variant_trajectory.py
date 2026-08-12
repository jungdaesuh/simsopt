"""Fresh-process JAX trajectory and source-owned profile-count producer.

The producer replays diagnostic-only C0/C1/C2 solver routes from the same
changed-state anchor used by the production evaluator.  Oracle payloads never
enter the timed or profiled value-and-gradient executable.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from examples.jax.parity.cases.native_boozerqa import _prepare_jax_variant_runtime
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import read_input_bundle
from simsopt_jax.geo.optimizers import optimizer as _optimizer
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _boozer_exact_residual,
    _traceable_exact_residual_kwargs,
)

from benchmarks.single_stage_compute_graph_c0_evaluator import (
    _canonical_candidate,
    _validate_runtime_contract,
    _verify_snapshot_import_origins,
)
from benchmarks.single_stage_compute_graph_c0_runner import (
    _load_canonical_json_object,
    _sha256_path,
)
from benchmarks.single_stage_compute_graph_canary_profile_runner import (
    PROFILE_COUNT_SCHEMA_ID,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CanarySpec,
    _spec_identity,
    validate_spec,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    RAW_TRAJECTORY_SCHEMA_ID,
    write_raw_trajectory_document,
)

Lane = Literal["C0", "C1", "C2"]
PROFILE_COUNT_FIELDS: Final = (
    "residual_evaluation_count",
    "dense_primal_traversal_count",
    "dense_tangent_batch_count",
    "dense_tangent_direction_count",
)


class VariantTrajectoryError(RuntimeError):
    """The requested diagnostic replay cannot produce canonical evidence."""


class _PreparedRuntime(Protocol):
    session: object

    def fresh_incumbent_controller(self) -> object: ...


class _IncumbentController(Protocol):
    current_inner_state: object


class _InnerState(Protocol):
    solved_x: object


class _Session(Protocol):
    compiled_bundle: Mapping[str, object]


class _IndexableValue(Protocol):
    def __getitem__(self, index: int) -> object: ...


class _OneStepCertificate(Protocol):
    active: object
    state: object
    residual: object
    jacobian: object
    initial_solve: object
    refinement_rhs: object
    refinement_correction: object
    refined_residual: object
    correction_step: object
    next_state: object


class _C1OracleTrace(_OneStepCertificate, Protocol):
    dense_materialization_count: object
    lu_factorization_count: object
    lu_solve_count: object
    refinement_correction_count: object
    residual_evaluation_count: object


@dataclass(frozen=True, slots=True)
class _ReplayInputs:
    residual_fn: object
    initial_state: jax.Array
    maxiter: int
    tolerance: float


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VariantTrajectoryError(f"{context} must be a lowercase SHA-256")
    return value


def _host_array(value: object, context: str, *, dimensions: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != dimensions:
        raise VariantTrajectoryError(f"{context} must be an exact FP64 array")
    if not bool(np.all(np.isfinite(array))):
        raise VariantTrajectoryError(f"{context} must contain finite values")
    return array


def _host_float(value: object, context: str) -> float:
    scalar = float(np.asarray(value))
    if not math.isfinite(scalar):
        raise VariantTrajectoryError(f"{context} must be finite")
    return scalar


def _host_int(value: object, context: str) -> int:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(
        scalar, (int, np.integer)
    ):
        raise VariantTrajectoryError(f"{context} must be an integer")
    normalized = int(scalar)
    if normalized < 0:
        raise VariantTrajectoryError(f"{context} must be nonnegative")
    return normalized


def _host_bool(value: object, context: str) -> bool:
    scalar = np.asarray(value).item()
    if not isinstance(scalar, (bool, np.bool_)):
        raise VariantTrajectoryError(f"{context} must be boolean")
    return bool(scalar)


def _exact_identity_from_spec(spec: CanarySpec) -> tuple[str, str, str, str]:
    if _sha256_path(spec.native_reference_path) != spec.native_reference_sha256:
        raise VariantTrajectoryError("native reference bytes differ from spec")
    native_reference = _load_canonical_json_object(
        spec.native_reference_path, "native reference"
    )
    identity = native_reference.get("identity")
    if not isinstance(identity, Mapping):
        raise VariantTrajectoryError("native reference identity is unavailable")
    return (
        _sha256(spec.parameter_sha256, "parameter SHA"),
        _sha256(spec.specimen_sha256, "specimen SHA"),
        _sha256(identity.get("input_bundle_sha256"), "input bundle SHA"),
        _sha256(spec.solver_graph_sha256, "solver graph SHA"),
    )


def _prepare_replay_inputs(spec: CanarySpec, lane: Lane) -> _ReplayInputs:
    bundle, arrays = read_input_bundle(spec.input_root)
    if bundle.case_id != SPEC.case_id or bundle.scale != "native_default":
        raise VariantTrajectoryError(
            "input bundle must be the native-default single-stage specimen"
        )
    runtime = cast(
        _PreparedRuntime,
        _prepare_jax_variant_runtime(
            bundle,
            arrays,
            SPEC,
            None,
            exact_newton_variant=lane,
        ),
    )
    candidate = _canonical_candidate(spec.candidate_path, spec.parameter_sha256)
    session = cast(_Session, runtime.session)
    state = session.compiled_bundle["state"]
    if not isinstance(state, Mapping):
        raise VariantTrajectoryError("traceable objective state is unavailable")
    coil_set_spec_from_dofs = state.get("coil_set_spec_from_dofs")
    objective_kwargs = state.get("objective_kwargs")
    if not callable(coil_set_spec_from_dofs) or not isinstance(
        objective_kwargs, Mapping
    ):
        raise VariantTrajectoryError("traceable residual construction is unavailable")
    candidate_device = jax.device_put(candidate, device=jax.devices()[0])
    coil_set_spec = coil_set_spec_from_dofs(candidate_device)
    exact_kwargs = _traceable_exact_residual_kwargs(objective_kwargs)

    def residual_fn(x_inner: jax.Array) -> jax.Array:
        return _boozer_exact_residual(
            x_inner,
            coil_set_spec=coil_set_spec,
            **exact_kwargs,
        )

    controller = cast(_IncumbentController, runtime.fresh_incumbent_controller())
    inner_state = cast(_InnerState, controller.current_inner_state)
    initial_state = jnp.asarray(inner_state.solved_x, dtype=jnp.float64)
    maxiter = bundle.configuration.get("inner_maxiter")
    tolerance = bundle.configuration.get("inner_tolerance")
    if (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or maxiter < 2
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise VariantTrajectoryError("exact-Newton configuration is invalid")
    return _ReplayInputs(
        residual_fn=residual_fn,
        initial_state=initial_state,
        maxiter=maxiter,
        tolerance=float(tolerance),
    )


def _one_step_document(
    certificate: _OneStepCertificate,
    *,
    index: int | None = None,
    converged: bool,
    numerical_failure: bool,
    status_code: int,
    counters: Mapping[str, int],
) -> dict[str, object]:
    def field(value: object) -> object:
        return value if index is None else cast(_IndexableValue, value)[index]

    if not _host_bool(field(certificate.active), "one-step active"):
        raise VariantTrajectoryError("one-step oracle certificate is inactive")
    return {
        "initial_state": _host_array(
            field(certificate.state), "one-step state", dimensions=1
        ).tolist(),
        "residual": _host_array(
            field(certificate.residual), "one-step residual", dimensions=1
        ).tolist(),
        "jacobian": _host_array(
            field(certificate.jacobian), "one-step Jacobian", dimensions=2
        ).tolist(),
        "initial_solve": _host_array(
            field(certificate.initial_solve),
            "one-step initial solve",
            dimensions=1,
        ).tolist(),
        "refinement_rhs": _host_array(
            field(certificate.refinement_rhs),
            "one-step refinement RHS",
            dimensions=1,
        ).tolist(),
        "refinement_correction": _host_array(
            field(certificate.refinement_correction),
            "one-step refinement correction",
            dimensions=1,
        ).tolist(),
        "correction_step": _host_array(
            field(certificate.correction_step),
            "one-step correction",
            dimensions=1,
        ).tolist(),
        "refined_residual": _host_array(
            field(certificate.refined_residual),
            "one-step refined residual",
            dimensions=1,
        ).tolist(),
        "next_state": _host_array(
            field(certificate.next_state), "one-step next state", dimensions=1
        ).tolist(),
        "converged": converged,
        "numerical_failure": numerical_failure,
        "status_code": status_code,
        "counters": dict(counters),
    }


def _one_step_counters_from_trace(trace: _C1OracleTrace, index: int) -> dict[str, int]:
    return {
        "residual_evaluation_count": _host_int(
            cast(_IndexableValue, trace.residual_evaluation_count)[index],
            "one-step residual count",
        ),
        "dense_materialization_count": _host_int(
            cast(_IndexableValue, trace.dense_materialization_count)[index],
            "one-step dense count",
        ),
        "lu_factorization_count": _host_int(
            cast(_IndexableValue, trace.lu_factorization_count)[index],
            "one-step LU count",
        ),
        "lu_solve_count": _host_int(
            cast(_IndexableValue, trace.lu_solve_count)[index],
            "one-step solve count",
        ),
        "refinement_correction_count": _host_int(
            cast(_IndexableValue, trace.refinement_correction_count)[index],
            "one-step refinement count",
        ),
    }


def _c1_document(
    replay: _ReplayInputs,
    *,
    parameter_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    solver_graph_sha256: str,
    source_sha256: str,
) -> tuple[dict[str, object], object]:
    residual_fn = cast(object, replay.residual_fn)
    runner = _optimizer._make_traceable_dense_direct_exact_newton_c1_oracle_runner(
        residual_fn,
        replay.maxiter,
        replay.tolerance,
    )
    result = runner(replay.initial_state, ())
    jax.block_until_ready(result)
    trace = result.trace
    active = np.asarray(trace.active, dtype=np.bool_)
    active_indices = tuple(int(index) for index in np.flatnonzero(active))
    if len(active_indices) < 2 or active_indices != tuple(range(len(active_indices))):
        raise VariantTrajectoryError("C1 oracle trace is incomplete or noncontiguous")
    steps: list[dict[str, object]] = []
    accepted_count = 0
    for output_index, trace_index in enumerate(active_indices):
        accepted = _host_bool(trace.accepted[trace_index], "C1 accepted")
        accepted_count += int(accepted)
        steps.append(
            {
                "iteration_index": output_index,
                "state_before": _host_array(
                    trace.state[trace_index], "C1 state", dimensions=1
                ).tolist(),
                "update": _host_array(
                    trace.correction_step[trace_index], "C1 update", dimensions=1
                ).tolist(),
                "state_after": _host_array(
                    trace.next_state[trace_index], "C1 next state", dimensions=1
                ).tolist(),
                "merit_before": _host_float(trace.norm[trace_index], "C1 norm"),
                "merit_after": _host_float(
                    trace.next_norm[trace_index], "C1 next norm"
                ),
                "state_assessed_after": True,
                "backtracking_iteration_count": _host_int(
                    trace.backtracking_iterations[trace_index],
                    "C1 backtracking iterations",
                ),
                "accepted": accepted,
                "stop_decision": output_index == len(active_indices) - 1,
                "status_code": (
                    _host_int(result.stop_reason_code, "C1 stop reason")
                    if output_index == len(active_indices) - 1
                    else 0
                ),
                "counters": {
                    "residual_evaluation_count": _host_int(
                        trace.residual_evaluation_count[trace_index],
                        "C1 residual count",
                    ),
                    "attempted_iteration_count": output_index + 1,
                    "accepted_update_count": accepted_count,
                },
            }
        )
    terminal_state = _host_array(result.x, "C1 terminal state", dimensions=1)
    terminal_residual = _host_array(
        result.residual, "C1 terminal residual", dimensions=1
    )
    terminal_jacobian = _host_array(
        jax.jacfwd(cast(object, replay.residual_fn))(result.x),
        "C1 terminal Jacobian",
        dimensions=2,
    )
    success = _host_bool(result.success, "C1 success")
    if not success:
        raise VariantTrajectoryError(
            "C1 diagnostic replay did not produce a persistable solved state"
        )
    stop_reason = _host_int(result.stop_reason_code, "C1 stop reason")
    first_index = active_indices[0]
    first_next_norm = _host_float(trace.next_norm[first_index], "C1 first next norm")
    document = {
        "schema_id": RAW_TRAJECTORY_SCHEMA_ID,
        "lane": "C1",
        "parameter_sha256": parameter_sha256,
        "specimen_sha256": specimen_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "solver_graph_sha256": solver_graph_sha256,
        "source_sha256": source_sha256,
        "one_step": _one_step_document(
            cast(_OneStepCertificate, trace),
            index=first_index,
            converged=first_next_norm <= replay.tolerance,
            numerical_failure=not _host_bool(
                trace.linear_success[first_index], "C1 linear success"
            ),
            status_code=0,
            counters=_one_step_counters_from_trace(trace, first_index),
        ),
        "short_replay": steps,
        "terminal": {
            "success": success,
            "persist_solved_state": success,
            "rollback_taken": not success,
            "returned_state": terminal_state.tolist(),
            "returned_residual": terminal_residual.tolist(),
            "returned_jacobian": terminal_jacobian.tolist(),
            "returned_norm": float(np.linalg.norm(terminal_residual)),
            "status_code": stop_reason,
            "counters": {
                "residual_evaluation_count": _host_int(
                    result.exact_newton_variant_residual_evaluation_count,
                    "C1 terminal residual count",
                ),
                "attempted_iteration_count": len(active_indices),
                "accepted_update_count": _host_int(result.nit, "C1 accepted count"),
            },
        },
    }
    return document, result


def _c2_document(
    replay: _ReplayInputs,
    *,
    parameter_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    solver_graph_sha256: str,
    source_sha256: str,
) -> tuple[dict[str, object], object]:
    residual_fn = cast(object, replay.residual_fn)
    runner = _optimizer._make_traceable_dense_direct_exact_newton_c2_oracle_runner(
        residual_fn,
        replay.maxiter,
        replay.tolerance,
    )
    result = runner(replay.initial_state, ())
    jax.block_until_ready(result)
    native = result.native
    applied_count = _host_int(native.applied_update_count, "C2 applied count")
    if applied_count < 2:
        raise VariantTrajectoryError("C2 oracle replay has fewer than two updates")
    states = np.asarray(native.applied_state_trace, dtype=np.float64)
    state_active = np.asarray(native.applied_state_trace_active, dtype=np.bool_)
    norm_trace = np.asarray(native.assessed_norm_trace, dtype=np.float64)
    norm_active = np.asarray(native.assessed_norm_trace_active, dtype=np.bool_)
    if not bool(np.all(state_active[: applied_count + 1])):
        raise VariantTrajectoryError("C2 applied-state trace is incomplete")
    assessed_count = int(np.count_nonzero(norm_active))
    steps: list[dict[str, object]] = []
    for index in range(applied_count):
        state_before = states[index]
        state_after = states[index + 1]
        assessed_after = bool(norm_active[index + 1])
        steps.append(
            {
                "iteration_index": index,
                "state_before": state_before.tolist(),
                "update": (state_before - state_after).tolist(),
                "state_after": state_after.tolist(),
                "merit_before": float(norm_trace[index]),
                "merit_after": float(norm_trace[index + 1]) if assessed_after else None,
                "state_assessed_after": assessed_after,
                "backtracking_iteration_count": 0,
                "accepted": True,
                "stop_decision": index == applied_count - 1,
                "status_code": (
                    _host_int(native.stop_reason_code, "C2 stop reason")
                    if index == applied_count - 1
                    else 0
                ),
                "counters": {
                    "residual_evaluation_count": index + 2,
                    "attempted_iteration_count": index + 1,
                    "applied_update_count": index + 1,
                    "assessed_state_count": index + 1 + int(assessed_after),
                    "rollback_recompute_count": 0,
                },
            }
        )
    success = _host_bool(native.success, "C2 success")
    rollback = _host_bool(native.rollback_branch_taken, "C2 rollback")
    terminal_residual = _host_array(
        native.residual, "C2 terminal residual", dimensions=1
    )
    first = cast(_OneStepCertificate, result.first_attempt)
    strict_cap = _optimizer._eisenstat_walker_strict_cap(
        replay.tolerance,
        dtype=jnp.asarray(first.state).dtype,
    )
    first_direction = _optimizer._dense_direct_exact_newton_direction_from_jacobian(
        jnp.asarray(first.residual),
        jnp.asarray(first.jacobian),
        tol=strict_cap,
    )
    jax.block_until_ready(first_direction)
    first_counters = {
        "residual_evaluation_count": 1,
        "dense_materialization_count": 1,
        "lu_factorization_count": _host_int(
            first_direction.status.lu_factorization_count,
            "C2 first LU count",
        ),
        "lu_solve_count": _host_int(
            first_direction.status.lu_solve_count, "C2 first solve count"
        ),
        "refinement_correction_count": _host_int(
            first_direction.status.refinement_correction_count,
            "C2 first refinement count",
        ),
    }
    document = {
        "schema_id": RAW_TRAJECTORY_SCHEMA_ID,
        "lane": "C2",
        "parameter_sha256": parameter_sha256,
        "specimen_sha256": specimen_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "solver_graph_sha256": solver_graph_sha256,
        "source_sha256": source_sha256,
        "one_step": _one_step_document(
            first,
            index=None,
            converged=bool(norm_active[1] and norm_trace[1] <= replay.tolerance),
            numerical_failure=not _host_bool(
                first_direction.status.success, "C2 linear success"
            ),
            status_code=0,
            counters=first_counters,
        ),
        "short_replay": steps,
        "terminal": {
            "success": success,
            "persist_solved_state": _host_bool(
                native.persist_solved_state, "C2 persistence"
            ),
            "rollback_taken": rollback,
            "returned_state": _host_array(
                native.x, "C2 terminal state", dimensions=1
            ).tolist(),
            "returned_residual": terminal_residual.tolist(),
            "returned_jacobian": _host_array(
                native.returned_jacobian,
                "C2 terminal Jacobian",
                dimensions=2,
            ).tolist(),
            "returned_norm": _host_float(native.returned_norm, "C2 returned norm"),
            "status_code": _host_int(native.stop_reason_code, "C2 stop reason"),
            "counters": {
                "residual_evaluation_count": _host_int(
                    result.exact_newton_variant_residual_evaluation_count,
                    "C2 terminal residual count",
                ),
                "attempted_iteration_count": _host_int(
                    native.linear_solve_attempt_count, "C2 attempted count"
                ),
                "applied_update_count": applied_count,
                "assessed_state_count": assessed_count,
                "rollback_recompute_count": _host_int(
                    native.rollback_recompute_count, "C2 rollback count"
                ),
            },
        },
    }
    return document, result


def _c0_document(
    replay: _ReplayInputs,
    *,
    parameter_sha256: str,
    specimen_sha256: str,
    input_bundle_sha256: str,
    solver_graph_sha256: str,
    source_sha256: str,
) -> tuple[dict[str, object], None]:
    residual_fn = cast(object, replay.residual_fn)
    runner = _optimizer._make_traceable_exact_newton_c0_oracle_runner(
        residual_fn,
        replay.maxiter,
        replay.tolerance,
    )
    result = runner(replay.initial_state, ())
    jax.block_until_ready(result)
    trace = result.trace
    active = np.asarray(trace.active, dtype=np.bool_)
    active_indices = tuple(int(index) for index in np.flatnonzero(active))
    if len(active_indices) < 2 or active_indices != tuple(range(len(active_indices))):
        raise VariantTrajectoryError("C0 oracle trace is incomplete or noncontiguous")
    steps: list[dict[str, object]] = []
    for output_index, trace_index in enumerate(active_indices):
        assessed_after = _host_bool(
            trace.merit_after_assessed[trace_index], "C0 assessed-after flag"
        )
        steps.append(
            {
                "iteration_index": output_index,
                "state_before": _host_array(
                    trace.state_before[trace_index], "C0 state before", dimensions=1
                ).tolist(),
                "update": _host_array(
                    trace.update[trace_index], "C0 update", dimensions=1
                ).tolist(),
                "state_after": _host_array(
                    trace.state_after[trace_index], "C0 state after", dimensions=1
                ).tolist(),
                "merit_before": _host_float(
                    trace.merit_before[trace_index], "C0 merit before"
                ),
                "merit_after": (
                    _host_float(trace.merit_after[trace_index], "C0 merit after")
                    if assessed_after
                    else None
                ),
                "state_assessed_after": assessed_after,
                "backtracking_iteration_count": _host_int(
                    trace.backtracking_iterations[trace_index],
                    "C0 backtracking iterations",
                ),
                "accepted": _host_bool(trace.accepted[trace_index], "C0 accepted"),
                "stop_decision": output_index == len(active_indices) - 1,
                "status_code": (
                    _host_int(trace.stop_reason_code[trace_index], "C0 stop reason")
                    if output_index == len(active_indices) - 1
                    else 0
                ),
                "counters": {
                    "residual_evaluation_count": _host_int(
                        trace.residual_evaluation_count[trace_index],
                        "C0 residual count",
                    ),
                    "attempted_iteration_count": _host_int(
                        trace.linear_solve_attempt_count[trace_index],
                        "C0 attempt count",
                    ),
                    "accepted_update_count": _host_int(
                        trace.accepted_update_count[trace_index],
                        "C0 accepted count",
                    ),
                },
            }
        )
    success = _host_bool(result.success, "C0 success")
    if not success:
        raise VariantTrajectoryError(
            "C0 diagnostic replay did not produce a persistable solved state"
        )
    terminal_residual = _host_array(
        result.residual, "C0 terminal residual", dimensions=1
    )
    document = {
        "schema_id": RAW_TRAJECTORY_SCHEMA_ID,
        "lane": "C0",
        "parameter_sha256": parameter_sha256,
        "specimen_sha256": specimen_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "solver_graph_sha256": solver_graph_sha256,
        "source_sha256": source_sha256,
        "one_step": None,
        "short_replay": steps,
        "terminal": {
            "success": success,
            "persist_solved_state": success,
            "rollback_taken": not success,
            "returned_state": _host_array(
                result.state, "C0 terminal state", dimensions=1
            ).tolist(),
            "returned_residual": terminal_residual.tolist(),
            "returned_jacobian": _host_array(
                result.jacobian, "C0 terminal Jacobian", dimensions=2
            ).tolist(),
            "returned_norm": _host_float(result.norm, "C0 terminal norm"),
            "status_code": _host_int(result.stop_reason_code, "C0 stop reason"),
            "counters": {
                "residual_evaluation_count": _host_int(
                    result.residual_evaluation_count, "C0 terminal residual count"
                ),
                "attempted_iteration_count": _host_int(
                    result.linear_solve_attempt_count, "C0 terminal attempt count"
                ),
                "accepted_update_count": _host_int(
                    result.accepted_update_count, "C0 terminal accepted count"
                ),
            },
        },
    }
    return document, None


def build_raw_trajectory(
    spec: CanarySpec,
    lane: Lane,
) -> tuple[dict[str, object], object | None]:
    """Run one diagnostic solver route and return its canonical raw payload."""

    parameter, specimen, input_bundle, solver_graph = _exact_identity_from_spec(spec)
    if _sha256_path(spec.input_root / "input_bundle.json") != input_bundle:
        raise VariantTrajectoryError(
            "input bundle bytes differ from the native reference"
        )
    replay = _prepare_replay_inputs(spec, lane)
    builder = {"C0": _c0_document, "C1": _c1_document, "C2": _c2_document}[lane]
    return builder(
        replay,
        parameter_sha256=parameter,
        specimen_sha256=specimen,
        input_bundle_sha256=input_bundle,
        solver_graph_sha256=solver_graph,
        source_sha256=_sha256(spec.source_state_sha256, "source state SHA"),
    )


def profile_counts_from_oracle_result(lane: Lane, result: object) -> dict[str, int]:
    """Extract source-owned dense traversal counts from a C1/C2 oracle result."""

    if lane == "C0":
        raise VariantTrajectoryError("C0 does not expose dense profile counts")
    return {
        field: _host_int(
            getattr(result, f"exact_newton_variant_{field}"),
            f"profile {field}",
        )
        for field in PROFILE_COUNT_FIELDS
    }


def write_profile_count_evidence(
    path: Path,
    *,
    spec: CanarySpec,
    canary_artifact_path: Path,
    counts: Mapping[str, int],
) -> str:
    """Exclusively bind oracle counts to validated spec and final artifact bytes."""

    artifact = _load_canonical_json_object(canary_artifact_path, "canary artifact")
    if artifact.get("identity") != _spec_identity(spec):
        raise VariantTrajectoryError("canary artifact identity differs from spec")
    if artifact.get("status") != "MEASURED_NONPROMOTING":
        raise VariantTrajectoryError(
            "profile counts require the pre-finalization measured canary artifact"
        )
    if set(counts) != set(PROFILE_COUNT_FIELDS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise VariantTrajectoryError("profile counts are incomplete or invalid")
    document = {
        "schema_id": PROFILE_COUNT_SCHEMA_ID,
        "identity": {
            **_spec_identity(spec),
            "canary_artifact_sha256": _sha256_path(canary_artifact_path),
        },
        "counts": {field: counts[field] for field in PROFILE_COUNT_FIELDS},
    }
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lane", choices=("C0", "C1", "C2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-count-output", type=Path)
    parser.add_argument("--canary-artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec_document = _load_canonical_json_object(args.spec, "canary spec")
        spec = validate_spec(spec_document)
        lane = cast(Lane, args.lane)
        if lane in ("C1", "C2") and lane != spec.variant:
            raise VariantTrajectoryError("variant lane differs from canary spec")
        if (args.profile_count_output is None) != (args.canary_artifact is None):
            raise VariantTrajectoryError(
                "profile-count output and canary artifact must be supplied together"
            )
        if args.profile_count_output is not None and lane == "C0":
            raise VariantTrajectoryError("C0 cannot produce dense profile counts")
        _verify_snapshot_import_origins(spec.snapshot_root)
        _validate_runtime_contract()
        document, result = build_raw_trajectory(spec, lane)
        write_raw_trajectory_document(args.output, document)
        if args.profile_count_output is not None:
            if result is None:
                raise VariantTrajectoryError("variant oracle result is unavailable")
            write_profile_count_evidence(
                args.profile_count_output,
                spec=spec,
                canary_artifact_path=args.canary_artifact,
                counts=profile_counts_from_oracle_result(lane, result),
            )
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"variant trajectory failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
