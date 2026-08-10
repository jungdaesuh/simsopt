"""GPU runner entry point for the single-stage full-space campaign."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from benchmarks.single_stage_fullspace_snapshot import activate_snapshot_source_imports

_SNAPSHOT_RUNTIME = (
    os.environ.get("SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256") is not None
)
if _SNAPSHOT_RUNTIME:
    activate_snapshot_source_imports(_SOURCE_ROOT)

from simsopt_jax.config import set_backend as _set_backend

if _SNAPSHOT_RUNTIME and "--preflight-only" not in sys.argv[1:]:
    _determinism_flag = "--xla_gpu_exclude_nondeterministic_ops=true"
    _xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    if _determinism_flag not in _xla_flags.split():
        os.environ["XLA_FLAGS"] = " ".join(
            token for token in (_xla_flags, _determinism_flag) if token
        )
    _set_backend(
        "jax_gpu_parity",
        strict=True,
        xla_gpu_preallocate=True,
    )

import jax
import jax.numpy as jnp
import simsoptpp
from simsopt_jax.geo.optimizers.dense_sqp import (
    DenseSQPKKTStep,
    DenseSQPStatus,
    solve_dense_sqp_kkt,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceProblem,
    fullspace_constraint_jvp,
    fullspace_constraint_vector,
    fullspace_constraint_vjp,
    fullspace_value_and_grad,
)
from simsopt_jax.solve.fullspace import (
    CfsSqp1Policy,
    FullSpaceRoute,
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
    prepare_cfs_al1,
    prepare_cfs_al2,
    prepare_cfs_p0,
    route_policy,
    sqp_route_policy,
)
from simsopt_jax.solve.fullspace_sqp import (
    CfsSqp1JointLinearization,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_linearization,
    prepare_cfs_sqp1,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    SingleStageFullSpaceBootstrap,
    build_single_stage_fullspace_bootstrap,
)

from benchmarks.single_stage_fullspace_bootstrap import (
    SCHEMA_VERSION as BOOTSTRAP_SCHEMA_VERSION,
)
from benchmarks.single_stage_fullspace_bootstrap import (
    publish_bootstrap_artifact,
    validate_bootstrap_artifact,
)
from benchmarks.single_stage_fullspace_process_gpu_monitor import (
    BoundProcessGpuMemoryMonitor,
    bound_gpu_memory_payload,
)
from benchmarks.single_stage_fullspace_receipt import (
    SCHEMA_VERSION,
    SQP_BUDGET_SHA256,
    SQP_KKT_FORWARD_ERROR_MAXIMUM,
    SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM,
    SQP_MAXIMUM_MEMORY_FRACTION,
    SQP_PLAN_SHA256,
    SQP_RESULT_SCHEMA_VERSION,
    SQP_SAMPLE_SCHEMA_VERSION,
    SQP_WARM_SOLVE_MAX_SECONDS,
    CompleteSample,
    DeviceLane,
    RunPhase,
    RunRequest,
    SqpGate,
    canonical_json_bytes,
    contract_sha256,
    contract_sha256_v2,
    load_sqp_gate_result,
    run_request_payload,
    run_request_payload_v2,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_FILENAME,
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    ArtifactRef,
    JsonValue,
    RuntimeEvidence,
    SnapshotPublication,
    SourceRoot,
    build_runtime_evidence,
    capture_worktree_identity,
    load_canonical_json_bytes,
    load_snapshot,
    observe_live_runtime,
    publish_immutable_snapshot,
    publish_runtime_evidence,
)
from benchmarks.validate_single_stage_fullspace_campaign import (
    load_sqp_sample_receipt,
)

SNAPSHOT_DIRECTORY: Final = "source-snapshot"
FIRST_EVAL_SCHEMA_VERSION: Final = "single-stage-fullspace-first-eval-v1"
FIRST_EVAL_RELATIVE_PATH: Final = "runs/first-eval.json"
CFS_P0_CANARY_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-p0-canary-v1"
CFS_P0_CANARY_10_RELATIVE_PATH: Final = "runs/cfs-p0-canary-10.json"
CFS_P0_CANARY_100_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-p0-canary-100-v1"
CFS_P0_CANARY_100_RELATIVE_PATH: Final = "runs/cfs-p0-canary-100.json"
CFS_P0_CHANGED_OPTIMIZER_NORM: Final = 1.0e-3
CFS_AL1_RESULT_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-al1-result-v1"
CFS_AL2_RESULT_SCHEMA_VERSION: Final = "single-stage-fullspace-cfs-al2-result-v1"
CFS_SQP1_DERIVATIVE_GATE_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-cfs-sqp1-derivative-gate-v1"
)
CFS_SQP1_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-cfs-sqp1-derivative-gate-receipt-v1"
)
CFS_SQP1_CANARY_1_GATE_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-cfs-sqp1-canary-1-gate-v1"
)
CFS_SQP1_CANARY_10_GATE_SCHEMA_VERSION: Final = (
    "single-stage-fullspace-cfs-sqp1-canary-10-gate-v1"
)
BOOTSTRAP_RELATIVE_PATH: Final = "artifacts/fullspace-bootstrap.json"
_SNAPSHOT_MANIFEST_ENV: Final = "SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256"
_CAMPAIGN_ROOT_ENV: Final = "SIMSOPT_FULLSPACE_CAMPAIGN_ROOT"
_SQP_CHILD_TIMEOUT_SECONDS: Final = 7_200.0
_CONFIGURATION_FILES: Final = (
    "docs/single_stage_jax_gpu_coupled_fullspace_implementation_plan.md",
    "docs/single_stage_jax_gpu_coupled_fullspace_phase0_budget.json",
    "docs/single_stage_jax_gpu_sqp_primal_dual_implementation_plan.md",
    "docs/single_stage_jax_gpu_sqp_primal_dual_phase0_budget.json",
)
_BENCHMARK_FILES: Final = (
    "benchmarks/process_gpu_monitor.py",
    "benchmarks/run_single_stage_fullspace_bootstrap.py",
    "benchmarks/run_single_stage_fullspace_gpu.py",
    "benchmarks/single_stage_fullspace_bootstrap.py",
    "benchmarks/single_stage_fullspace_endpoint_audit.py",
    "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
    "benchmarks/single_stage_fullspace_receipt.py",
    "benchmarks/single_stage_fullspace_snapshot.py",
    "benchmarks/validate_single_stage_fullspace_campaign.py",
)
_TEST_FILES: Final = (
    "tests/benchmarks/test_run_single_stage_fullspace_bootstrap.py",
    "tests/benchmarks/test_single_stage_fullspace_first_eval.py",
    "tests/benchmarks/test_single_stage_fullspace_campaign.py",
    "tests/benchmarks/test_single_stage_fullspace_campaign_v2.py",
    "tests/benchmarks/test_single_stage_fullspace_endpoint_audit.py",
    "tests/benchmarks/test_single_stage_fullspace_bootstrap_artifact.py",
    "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
    "tests/benchmarks/test_run_single_stage_fullspace_sqp.py",
    "tests/jax/adapters/test_single_stage_fullspace_bootstrap.py",
    "tests/jax/adapters/test_single_stage_fullspace_same_state_parity.py",
    "tests/jax/objectives/test_single_stage_fullspace_contract.py",
    "tests/jax/objectives/test_single_stage_fullspace_core.py",
    "tests/jax/objectives/test_single_stage_fullspace_derivatives.py",
    "tests/jax/solve/test_fullspace_route_contract.py",
    "tests/geo/test_dense_sqp.py",
    "tests/jax/solve/test_fullspace_certificate.py",
    "tests/jax/solve/test_fullspace_sqp.py",
)


class PhaseGateError(RuntimeError):
    """Raised when a requested execution phase is not implemented yet."""


@dataclass(frozen=True, slots=True)
class SnapshotChildInvocation:
    """Pinned interpreter, immutable entrypoint, cwd, and exact child environment."""

    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]


def _array_tree_size(value: object) -> tuple[int, int]:
    leaves = tuple(
        leaf
        for leaf in jax.tree_util.tree_leaves(value)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )
    return len(leaves), sum(
        int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize
        for leaf in leaves
    )


def _deterministic_changed_state(z: jax.Array) -> jax.Array:
    tangent = jnp.linspace(-1.0, 1.0, z.size, dtype=z.dtype)
    tangent = tangent / jnp.linalg.norm(tangent)
    relative_scale = jnp.maximum(jnp.abs(z), jnp.asarray(1.0, dtype=z.dtype))
    return z + jnp.asarray(1.0e-7, dtype=z.dtype) * relative_scale * tangent


def _deterministic_cfs_p0_changed_state(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    scaling = fullspace_scaling_from_bootstrap(z, problem)
    direction = jnp.linspace(-1.0, 1.0, z.size, dtype=z.dtype)
    direction = direction / jnp.linalg.norm(direction)
    optimizer_coordinates = (
        jnp.asarray(CFS_P0_CHANGED_OPTIMIZER_NORM, dtype=z.dtype) * direction
    )
    return fullspace_physical_coordinates(optimizer_coordinates, scaling)


def _first_eval_kernel(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> tuple[jax.Array, ...]:
    value, gradient = fullspace_value_and_grad(z, problem)
    tangent = jnp.linspace(-1.0, 1.0, z.size, dtype=z.dtype)
    tangent = tangent / jnp.linalg.norm(tangent)
    changed_z = _deterministic_changed_state(z)
    equality_count = fullspace_constraint_vector(changed_z, problem).size
    cotangent = jnp.linspace(0.75, -0.5, equality_count, dtype=z.dtype)
    jvp = fullspace_constraint_jvp(changed_z, tangent, problem)
    vjp = fullspace_constraint_vjp(changed_z, cotangent, problem)
    lhs = jnp.vdot(cotangent, jvp)
    rhs = jnp.vdot(vjp, tangent)
    absolute_error = jnp.abs(lhs - rhs)
    relative_error = absolute_error / jnp.maximum(
        jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)),
        jnp.asarray(jnp.finfo(z.dtype).tiny, dtype=z.dtype),
    )
    return (
        value,
        jnp.all(jnp.isfinite(gradient)),
        jnp.linalg.norm(gradient),
        jnp.max(jnp.abs(gradient)),
        jnp.linalg.norm(changed_z - z),
        jnp.all(jnp.isfinite(jvp)),
        jnp.all(jnp.isfinite(vjp)),
        lhs,
        rhs,
        absolute_error,
        relative_error,
    )


def run_first_eval_probe(
    bootstrap: SingleStageFullSpaceBootstrap,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, JsonValue]]:
    """Run one synchronized, transfer-guarded GPU derivative gate."""

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("first-eval requires exactly one JAX GPU")
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    staging_leaves, staging_bytes = _array_tree_size((host_z, host_problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))

    compiled_source = jax.jit(_first_eval_kernel)
    compile_start_ns = time.perf_counter_ns()
    compiled = compiled_source.lower(device_z, device_problem).compile()
    compile_elapsed_ns = time.perf_counter_ns() - compile_start_ns

    execute_start_ns = time.perf_counter_ns()
    with jax.transfer_guard("disallow"):
        device_result = compiled(device_z, device_problem)
        jax.block_until_ready(device_result)
    execute_elapsed_ns = time.perf_counter_ns() - execute_start_ns
    result_leaves, result_bytes = _array_tree_size(device_result)
    host_result = tuple(np.asarray(value) for value in jax.device_get(device_result))
    (
        value,
        gradient_finite,
        gradient_l2,
        gradient_inf,
        changed_state_l2,
        jvp_finite,
        vjp_finite,
        transpose_lhs,
        transpose_rhs,
        transpose_absolute_error,
        transpose_relative_error,
    ) = host_result
    scalar_values = tuple(
        float(item)
        for item in (
            value,
            gradient_l2,
            gradient_inf,
            changed_state_l2,
            transpose_lhs,
            transpose_rhs,
            transpose_absolute_error,
            transpose_relative_error,
        )
    )
    if not all(np.isfinite(scalar_values)) or not all(
        bool(item) for item in (gradient_finite, jvp_finite, vjp_finite)
    ):
        raise ValueError("first-eval value or derivative evidence is non-finite")
    if scalar_values[3] <= 0.0:
        raise ValueError("first-eval derivative evidence did not change state")
    if scalar_values[-1] > 1.0e-9 and scalar_values[-2] > 1.0e-10:
        raise ValueError("first-eval JVP/VJP transpose identity failed")

    numerical: dict[str, JsonValue] = {
        "changed_state_l2": scalar_values[3],
        "gradient_all_finite": bool(gradient_finite),
        "gradient_inf": scalar_values[2],
        "gradient_l2": scalar_values[1],
        "jvp_all_finite": bool(jvp_finite),
        "transpose_absolute_error": scalar_values[-2],
        "transpose_lhs": scalar_values[4],
        "transpose_relative_error": scalar_values[-1],
        "transpose_rhs": scalar_values[5],
        "value": scalar_values[0],
        "vjp_all_finite": bool(vjp_finite),
    }
    timing: dict[str, JsonValue] = {
        "cold_compile_ns": compile_elapsed_ns,
        "cold_execution_ns": execute_elapsed_ns,
        "synchronization": "block_until_ready",
    }
    transfer_audit: dict[str, JsonValue] = {
        "audit_scope": "first_eval_probe_after_bootstrap_publication",
        "final_d2h_bytes": result_bytes,
        "final_d2h_calls": 1,
        "final_d2h_leaves": result_leaves,
        "hot_d2h_calls": 0,
        "initial_h2d_bytes": staging_bytes,
        "initial_h2d_calls": 1,
        "initial_h2d_leaves": staging_leaves,
        "initial_host_staging_d2h_bytes": staging_bytes,
        "initial_host_staging_d2h_calls": 1,
        "timed_execution_transfer_guard": "disallow",
    }
    return numerical, timing, transfer_audit


def run_cfs_p0_canary_probe(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    maximum_iterations: int,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, JsonValue]]:
    """Prepare once and run one synchronized callback-free CFS-P0 canary."""

    if maximum_iterations not in (10, 100):
        raise ValueError("CFS-P0 canary requires exactly 10 or 100 steps")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("CFS-P0 canary requires exactly one JAX GPU")
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    staging_leaves, staging_bytes = _array_tree_size((host_z, host_problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))
    changed_state = _deterministic_cfs_p0_changed_state(device_z, device_problem)

    preparation_start_ns = time.perf_counter_ns()
    prepared = prepare_cfs_p0(
        device_problem,
        device_z,
        changed_state,
    )
    jax.block_until_ready(
        (
            prepared.initial_merit,
            prepared.initial_gradient,
            prepared.initial_diagnostics,
        )
    )
    preparation_elapsed_ns = time.perf_counter_ns() - preparation_start_ns

    execution_start_ns = time.perf_counter_ns()
    with jax.transfer_guard("disallow"):
        result = prepared.run(maximum_iterations=maximum_iterations)
        device_evidence = (
            result.initial_diagnostics.scaled_penalty_value,
            result.final_diagnostics.scaled_penalty_value,
            result.initial_diagnostics.scaled_constraint_infinity_norm,
            result.final_diagnostics.scaled_constraint_infinity_norm,
            result.initial_stationarity_infinity_norm,
            result.final_stationarity_infinity_norm,
            result.optimizer.iterations,
            result.optimizer.function_evaluations,
            result.optimizer.gradient_evaluations,
            result.made_progress,
            result.all_finite,
            result.nonfinite_evaluation_count,
        )
        jax.block_until_ready(device_evidence)
    execution_elapsed_ns = time.perf_counter_ns() - execution_start_ns
    result_leaves, result_bytes = _array_tree_size(device_evidence)
    host_evidence = tuple(
        np.asarray(value) for value in jax.device_get(device_evidence)
    )
    floating = tuple(float(value) for value in host_evidence[:6])
    iterations, function_evaluations, gradient_evaluations = (
        int(value) for value in host_evidence[6:9]
    )
    made_progress, all_finite = (bool(value) for value in host_evidence[9:11])
    nonfinite_evaluation_count = int(host_evidence[11])
    if not all(np.isfinite(floating)) or not all_finite:
        raise ValueError("CFS-P0 canary produced non-finite state or diagnostics")
    if not made_progress:
        raise ValueError("CFS-P0 canary failed the frozen progress gate")
    if nonfinite_evaluation_count != 0:
        raise ValueError("CFS-P0 canary encountered a non-finite evaluation")
    if iterations != maximum_iterations:
        raise ValueError("CFS-P0 canary did not exhaust its exact iteration budget")
    if (
        function_evaluations < iterations + 1
        or gradient_evaluations != function_evaluations
        or function_evaluations
        > route_policy(
            FullSpaceRoute.CFS_P0
        ).maximum_function_evaluations_per_inner_solve
    ):
        raise ValueError("CFS-P0 canary evaluation counters violate budget integrity")

    numerical: dict[str, JsonValue] = {
        "all_finite": all_finite,
        "final_M0": floating[1],
        "final_scaled_feasibility_inf": floating[3],
        "final_stationarity_inf": floating[5],
        "function_evaluations": function_evaluations,
        "gradient_evaluations": gradient_evaluations,
        "initial_M0": floating[0],
        "initial_scaled_feasibility_inf": floating[2],
        "initial_stationarity_inf": floating[4],
        "iterations": iterations,
        "made_progress": made_progress,
        "nonfinite_evaluation_count": nonfinite_evaluation_count,
    }
    execution_key = (
        "ten_step_execution_ns"
        if maximum_iterations == 10
        else "hundred_step_execution_ns"
    )
    timing: dict[str, JsonValue] = {
        "preparation_and_compile_ns": preparation_elapsed_ns,
        "synchronization": "block_until_ready",
        execution_key: execution_elapsed_ns,
    }
    transfers: dict[str, JsonValue] = {
        "audit_scope": "cfs_p0_canary_after_bootstrap_publication",
        "final_d2h_bytes": result_bytes,
        "final_d2h_calls": 1,
        "final_d2h_leaves": result_leaves,
        "hot_d2h_calls": 0,
        "initial_h2d_bytes": staging_bytes,
        "initial_h2d_calls": 1,
        "initial_h2d_leaves": staging_leaves,
        "initial_host_staging_d2h_bytes": staging_bytes,
        "initial_host_staging_d2h_calls": 1,
        "timed_execution_transfer_guard": "disallow",
    }
    return numerical, timing, transfers


def _optional_finite_float(value: object) -> float | None:
    scalar = float(np.asarray(value))
    return scalar if np.isfinite(scalar) else None


def _vector_payload(value: object) -> tuple[list[JsonValue], str]:
    payload = np.asarray(value).tolist()
    if not isinstance(payload, list):
        raise TypeError("SQP vector evidence must be a list")
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload, digest


def _sqp_maximum_iterations(request: RunRequest) -> int:
    request.validate_v2()
    if request.route is not FullSpaceRoute.CFS_SQP1:
        raise ValueError("SQP iteration selection requires CFS-SQP1")
    if request.phase is RunPhase.FIRST_EVAL:
        raise ValueError("SQP derivative/KKT gate has no optimizer iteration budget")
    if request.phase is RunPhase.CANARY:
        assert request.steps is not None
        return request.steps
    return 100


def _sqp_run_relative_directory(request: RunRequest) -> Path:
    """Return a collision-free campaign-local directory for one SQP seam."""

    if request.phase is RunPhase.FIRST_EVAL:
        return Path("gates/derivative")
    maximum_iterations = _sqp_maximum_iterations(request)
    if request.phase is RunPhase.CANARY:
        return Path(f"gates/canary-{maximum_iterations}")
    assert request.sample is not None
    return Path("samples") / request.sample.value


def _deterministic_cfs_sqp1_changed_state(
    bootstrap_state: jax.Array,
    problem: FullSpaceProblem,
) -> jax.Array:
    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    direction = jnp.linspace(
        -1.0, 1.0, bootstrap_state.size, dtype=bootstrap_state.dtype
    )
    direction = direction / jnp.linalg.norm(direction)
    optimizer_coordinates = jnp.asarray(1.0e-3, dtype=bootstrap_state.dtype) * direction
    return fullspace_physical_coordinates(optimizer_coordinates, scaling)


def _solve_changed_state_gate_kkt(
    changed_linearization: CfsSqp1JointLinearization,
    identity: jax.Array,
    policy: CfsSqp1Policy,
) -> tuple[DenseSQPKKTStep, jax.Array]:
    """Solve the diagnostic KKT system from the changed-state linearization."""

    constraint_jacobian = changed_linearization.constraint_jacobian
    zero_multipliers = jnp.zeros(
        changed_linearization.scaled_constraints.shape,
        dtype=changed_linearization.objective_gradient.dtype,
    )
    dual_residual = (
        changed_linearization.objective_gradient
        + constraint_jacobian.T @ zero_multipliers
    )
    step = solve_dense_sqp_kkt(
        identity,
        constraint_jacobian,
        dual_residual,
        changed_linearization.scaled_constraints,
        regularization_ladder=policy.regularization_ladder,
        relative_residual_tolerance=policy.kkt_relative_residual_tolerance,
        schur_relative_residual_tolerance=policy.schur_relative_residual_tolerance,
        kkt_forward_error_tolerance=policy.kkt_forward_error_tolerance,
        kkt_solution_scaled_residual_tolerance=(
            policy.kkt_solution_scaled_residual_tolerance
        ),
    )
    return step, dual_residual


def run_cfs_sqp1_derivative_gate(
    bootstrap: SingleStageFullSpaceBootstrap,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, JsonValue]]:
    """Materialize both derivative states and one certified changed-state KKT step."""

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("CFS-SQP1 derivative gate requires exactly one JAX GPU")
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    staging_leaves, staging_bytes = _array_tree_size((host_z, host_problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))
    policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)

    def gate_kernel(z0: jax.Array, problem: FullSpaceProblem) -> dict[str, object]:
        scaling = fullspace_scaling_from_bootstrap(z0, problem)
        bootstrap_coordinates = fullspace_optimizer_coordinates(z0, scaling)
        changed_state = _deterministic_cfs_sqp1_changed_state(z0, problem)
        changed_coordinates = fullspace_optimizer_coordinates(changed_state, scaling)
        bootstrap_linearization = cfs_sqp1_joint_linearization(
            bootstrap_coordinates, problem, scaling
        )
        changed_linearization = cfs_sqp1_joint_linearization(
            changed_coordinates, problem, scaling
        )
        direction = jnp.linspace(-0.75, 1.0, bootstrap_coordinates.size, dtype=z0.dtype)
        direction = direction / jnp.linalg.norm(direction)
        cotangent = jnp.linspace(
            0.5,
            -1.25,
            bootstrap_linearization.scaled_constraints.size,
            dtype=z0.dtype,
        )
        cotangent = cotangent / jnp.linalg.norm(cotangent)

        def state_diagnostics(linearization: object) -> dict[str, jax.Array]:
            constraint_jacobian = linearization.constraint_jacobian
            singular_values = jnp.linalg.svd(constraint_jacobian, compute_uv=False)
            sigma_maximum = jnp.max(singular_values)
            rank_cutoff = (
                jnp.asarray(policy.rank_relative_threshold, dtype=z0.dtype)
                * sigma_maximum
            )
            av = constraint_jacobian @ direction
            atw = constraint_jacobian.T @ cotangent
            lhs = jnp.vdot(cotangent, av)
            rhs = jnp.vdot(atw, direction)
            absolute_error = jnp.abs(lhs - rhs)
            relative_error = absolute_error / jnp.maximum(
                jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)),
                jnp.asarray(jnp.finfo(z0.dtype).tiny, dtype=z0.dtype),
            )
            return {
                "physical_objective": linearization.physical_objective,
                "scaled_constraints": linearization.scaled_constraints,
                "objective_gradient": linearization.objective_gradient,
                "constraint_jacobian": constraint_jacobian,
                "joint_vjp_rows": linearization.joint_vjp_rows,
                "all_finite": linearization.all_finite,
                "singular_values": singular_values,
                "numerical_rank": jnp.sum(singular_values > rank_cutoff),
                "rank_cutoff": rank_cutoff,
                "av": av,
                "atw": atw,
                "transpose_lhs": lhs,
                "transpose_rhs": rhs,
                "transpose_absolute_error": absolute_error,
                "transpose_relative_error": relative_error,
            }

        bootstrap_diagnostics = state_diagnostics(bootstrap_linearization)
        changed_diagnostics = state_diagnostics(changed_linearization)
        constraint_jacobian = changed_linearization.constraint_jacobian
        identity = jnp.eye(bootstrap_coordinates.size, dtype=z0.dtype)
        step, dual_residual = _solve_changed_state_gate_kkt(
            changed_linearization,
            identity,
            policy,
        )
        regularized_bfgs = identity + step.selected_regularization * identity
        kkt_matrix = jnp.block(
            [
                [regularized_bfgs, constraint_jacobian.T],
                [
                    constraint_jacobian,
                    jnp.zeros(
                        (
                            constraint_jacobian.shape[0],
                            constraint_jacobian.shape[0],
                        ),
                        dtype=z0.dtype,
                    ),
                ],
            ]
        )
        right_hand_side = -jnp.concatenate(
            (dual_residual, changed_linearization.scaled_constraints)
        )
        solution = jnp.concatenate((step.primal_step, step.multiplier_step))
        reconstructed_residual = kkt_matrix @ solution - right_hand_side
        reconstructed_residual_inf = jnp.linalg.norm(
            reconstructed_residual, ord=jnp.inf
        )
        reconstructed_residual_two = jnp.linalg.norm(reconstructed_residual, ord=2)
        certified_error_bound = step.kkt_forward_error_bound
        return {
            "bootstrap": bootstrap_diagnostics,
            "changed": changed_diagnostics,
            "changed_physical_state": changed_state,
            "changed_optimizer_coordinates": changed_coordinates,
            "kkt": {
                "primal_step": step.primal_step,
                "multiplier_step": step.multiplier_step,
                "valid": step.valid,
                "selected_regularization": step.selected_regularization,
                "rho_k": step.kkt_reciprocal_condition,
                "zeta_2": step.kkt_solution_scaled_residual,
                "kkt_relative_residual": step.kkt_relative_residual,
                "schur_relative_residual": step.schur_relative_residual,
                "bfgs_cholesky_relative_pivot": (step.bfgs_cholesky_relative_pivot),
                "schur_cholesky_relative_pivot": (step.schur_cholesky_relative_pivot),
                "regularization_candidates_tested": (
                    step.regularization_candidates_tested
                ),
                "reconstructed_residual_inf": reconstructed_residual_inf,
                "reconstructed_residual_two": reconstructed_residual_two,
                "certified_relative_error_bound": certified_error_bound,
                "all_finite": step.all_finite,
            },
        }

    compiled = jax.jit(gate_kernel)
    execution_start_ns = time.perf_counter_ns()
    with jax.transfer_guard("disallow"):
        device_evidence = compiled(device_z, device_problem)
        jax.block_until_ready(device_evidence)
    execution_elapsed_ns = time.perf_counter_ns() - execution_start_ns
    result_leaves, result_bytes = _array_tree_size(device_evidence)
    host = jax.device_get(device_evidence)

    def finite_array_payload(value: object) -> list[JsonValue] | None:
        array = np.asarray(value)
        return array.tolist() if np.all(np.isfinite(array)) else None

    def finite_scalar_payload(value: object) -> float | None:
        scalar = float(np.asarray(value))
        return scalar if np.isfinite(scalar) else None

    def state_payload(value: Mapping[str, object]) -> dict[str, JsonValue]:
        rows = np.asarray(value["joint_vjp_rows"])
        constraints = np.asarray(value["scaled_constraints"])
        gradient = np.asarray(value["objective_gradient"])
        jacobian = np.asarray(value["constraint_jacobian"])
        singular_values = np.asarray(value["singular_values"])
        av = np.asarray(value["av"])
        atw = np.asarray(value["atw"])
        constraints_payload = finite_array_payload(constraints)
        rows_payload = finite_array_payload(rows)
        singular_values_payload = finite_array_payload(singular_values)
        return {
            "physical_objective": finite_scalar_payload(value["physical_objective"]),
            "scaled_constraints": constraints_payload,
            "scaled_constraints_sha256": (
                None
                if constraints_payload is None
                else hashlib.sha256(
                    canonical_json_bytes(constraints_payload)
                ).hexdigest()
            ),
            "objective_gradient": finite_array_payload(gradient),
            "constraint_jacobian": finite_array_payload(jacobian),
            "joint_vjp_rows": rows_payload,
            "joint_vjp_rows_sha256": (
                None
                if rows_payload is None
                else hashlib.sha256(canonical_json_bytes(rows_payload)).hexdigest()
            ),
            "joint_vjp_rows_shape": list(rows.shape),
            "joint_vjp_rows_dtype": str(rows.dtype),
            "all_finite": bool(np.asarray(value["all_finite"])),
            "singular_values": singular_values_payload,
            "sigma_minimum": finite_scalar_payload(singular_values[-1]),
            "sigma_maximum": finite_scalar_payload(singular_values[0]),
            "numerical_rank": int(np.asarray(value["numerical_rank"])),
            "rank_relative_threshold": policy.rank_relative_threshold,
            "rank_cutoff": finite_scalar_payload(value["rank_cutoff"]),
            "av": finite_array_payload(av),
            "atw": finite_array_payload(atw),
            "transpose_lhs": finite_scalar_payload(value["transpose_lhs"]),
            "transpose_rhs": finite_scalar_payload(value["transpose_rhs"]),
            "transpose_absolute_error": finite_scalar_payload(
                value["transpose_absolute_error"]
            ),
            "transpose_relative_error": finite_scalar_payload(
                value["transpose_relative_error"]
            ),
        }

    kkt_host = host["kkt"]
    kkt_payload: dict[str, JsonValue] = {
        key: (
            bool(np.asarray(value))
            if key in ("valid", "all_finite")
            else int(np.asarray(value))
            if key == "regularization_candidates_tested"
            else finite_array_payload(value)
            if key in ("primal_step", "multiplier_step")
            else finite_scalar_payload(value)
        )
        for key, value in kkt_host.items()
    }
    bootstrap_payload = state_payload(host["bootstrap"])
    changed_payload = state_payload(host["changed"])
    failure_reasons: list[JsonValue] = []

    def audit_state(name: str, value: Mapping[str, JsonValue]) -> None:
        if value["all_finite"] is not True:
            failure_reasons.append(f"{name}_NONFINITE")
        if value["joint_vjp_rows_shape"] != [256, 716]:
            failure_reasons.append(f"{name}_ROW_SHAPE")
        if value["joint_vjp_rows_dtype"] != "float64":
            failure_reasons.append(f"{name}_DTYPE")
        if value["numerical_rank"] != 255:
            failure_reasons.append(f"{name}_RANK")
        absolute_error = value["transpose_absolute_error"]
        relative_error = value["transpose_relative_error"]
        if not isinstance(absolute_error, float) or not isinstance(
            relative_error, float
        ):
            failure_reasons.append(f"{name}_TRANSPOSE_NONFINITE")
        elif relative_error > 1.0e-9 and absolute_error > 1.0e-10:
            failure_reasons.append(f"{name}_TRANSPOSE_IDENTITY")

    audit_state("BOOTSTRAP", bootstrap_payload)
    audit_state("CHANGED", changed_payload)
    if kkt_payload["valid"] is not True:
        failure_reasons.append("KKT_INVALID")
    if kkt_payload["all_finite"] is not True:
        failure_reasons.append("KKT_NONFINITE")
    rho_k = kkt_payload["rho_k"]
    zeta_2 = kkt_payload["zeta_2"]
    if not (
        isinstance(rho_k, float)
        and isinstance(zeta_2, float)
        and np.isfinite(rho_k)
        and np.isfinite(zeta_2)
        and rho_k > zeta_2
    ):
        failure_reasons.append("KKT_RHO")
    kkt_thresholds = (
        (
            "zeta_2",
            policy.kkt_solution_scaled_residual_tolerance,
            lambda actual, limit: actual <= limit,
            "KKT_ZETA",
        ),
        (
            "kkt_relative_residual",
            policy.kkt_relative_residual_tolerance,
            lambda actual, limit: actual <= limit,
            "KKT_RESIDUAL",
        ),
        (
            "schur_relative_residual",
            policy.schur_relative_residual_tolerance,
            lambda actual, limit: actual <= limit,
            "SCHUR_RESIDUAL",
        ),
        (
            "certified_relative_error_bound",
            SQP_KKT_FORWARD_ERROR_MAXIMUM,
            lambda actual, limit: actual < limit,
            "KKT_ERROR_BOUND",
        ),
    )
    for key, limit, predicate, reason in kkt_thresholds:
        actual = kkt_payload[key]
        if not isinstance(actual, float) or not predicate(actual, limit):
            failure_reasons.append(reason)
    gate: dict[str, JsonValue] = {
        "schema_version": CFS_SQP1_DERIVATIVE_GATE_SCHEMA_VERSION,
        "gate_status": "PASS" if not failure_reasons else "FAIL",
        "failure_reasons": failure_reasons,
        "bootstrap": bootstrap_payload,
        "changed": changed_payload,
        "changed_physical_state": finite_array_payload(host["changed_physical_state"]),
        "changed_optimizer_coordinates": finite_array_payload(
            host["changed_optimizer_coordinates"]
        ),
        "kkt": kkt_payload,
        "optimizer_steps_executed": 0,
    }
    timing: dict[str, JsonValue] = {
        "synchronized_derivative_kkt_seconds": execution_elapsed_ns / 1.0e9,
        "synchronization": "block_until_ready",
    }
    transfers: dict[str, JsonValue] = {
        "audit_scope": "cfs_sqp1_derivative_kkt_gate_after_bootstrap_publication",
        "hot_h2d_calls": 0,
        "hot_d2h_calls": 0,
        "initial_h2d_calls": 1,
        "initial_h2d_bytes": staging_bytes,
        "initial_h2d_leaves": staging_leaves,
        "final_d2h_calls": 1,
        "final_d2h_bytes": result_bytes,
        "final_d2h_leaves": result_leaves,
        "timed_execution_transfer_guard": "disallow",
    }
    return gate, timing, transfers


def _prepare_cfs_sqp1_probe(
    device_problem: FullSpaceProblem,
    device_z: jax.Array,
    *,
    maximum_iterations: int,
) -> object:
    initial_physical_state = (
        _deterministic_cfs_sqp1_changed_state(device_z, device_problem)
        if maximum_iterations == 1
        else device_z
    )
    return prepare_cfs_sqp1(
        device_problem,
        device_z,
        initial_physical_state,
        maximum_iterations=maximum_iterations,
    )


def run_cfs_sqp1_probe(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    maximum_iterations: int,
    warm: bool,
) -> tuple[
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
]:
    """Run one device-resident SQP solve from immutable prepared inputs."""

    if maximum_iterations not in (1, 10, 100):
        raise ValueError("CFS-SQP1 supports exactly 1, 10, or 100 iterations")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError("CFS-SQP1 requires exactly one JAX GPU")
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    staging_leaves, staging_bytes = _array_tree_size((host_z, host_problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))
    prepared = _prepare_cfs_sqp1_probe(
        device_problem, device_z, maximum_iterations=maximum_iterations
    )
    jax.block_until_ready(
        (
            prepared.initial_optimizer_coordinates,
            prepared.initial_scaled_multipliers,
        )
    )
    initial_endpoint = cfs_sqp1_endpoint_diagnostics(
        prepared.initial_optimizer_coordinates,
        prepared.initial_scaled_multipliers,
        device_problem,
        prepared.scaling,
    )
    jax.block_until_ready(initial_endpoint)
    if warm:
        with jax.transfer_guard("disallow"):
            jax.block_until_ready(prepared.run_solver())

    solve_start_ns = time.perf_counter_ns()
    with jax.transfer_guard("disallow"):
        optimizer_result = prepared.run_solver()
        jax.block_until_ready(optimizer_result)
    solve_elapsed_ns = time.perf_counter_ns() - solve_start_ns
    finalized = prepared.finalize_result(optimizer_result)
    device_evidence = {
        "optimizer": optimizer_result,
        "initial_endpoint": initial_endpoint,
        "endpoint": finalized.endpoint,
        "route_all_finite": finalized.all_finite,
        "route_converged": finalized.converged,
    }
    jax.block_until_ready(device_evidence)
    result_leaves, result_bytes = _array_tree_size(device_evidence)
    host = jax.device_get(device_evidence)
    optimizer = host["optimizer"]
    initial_endpoint_result = host["initial_endpoint"]
    endpoint_result = host["endpoint"]
    iterations = int(np.asarray(optimizer.iterations))
    status = DenseSQPStatus(int(np.asarray(optimizer.status))).name
    history = {
        "accepted_length": iterations,
        "objective": np.asarray(optimizer.history.objective[:iterations]).tolist(),
        "feasibility_infinity_norm": np.asarray(
            optimizer.history.feasibility_infinity_norm[:iterations]
        ).tolist(),
        "stationarity_infinity_norm": np.asarray(
            optimizer.history.stationarity_infinity_norm[:iterations]
        ).tolist(),
        "step_length": np.asarray(optimizer.history.step_length[:iterations]).tolist(),
        "kkt_relative_residual": np.asarray(
            optimizer.history.kkt_relative_residual[:iterations]
        ).tolist(),
        "status": np.asarray(optimizer.history.status[:iterations]).tolist(),
    }
    physical_state, physical_state_sha = _vector_payload(endpoint_result.physical_state)
    optimizer_coordinates, optimizer_coordinates_sha = _vector_payload(
        optimizer.optimizer_coordinates
    )
    scaled_multipliers, scaled_multipliers_sha = _vector_payload(
        endpoint_result.scaled_multipliers
    )
    raw_multipliers, raw_multipliers_sha = _vector_payload(
        endpoint_result.raw_multipliers
    )
    endpoint: dict[str, JsonValue] = {
        "physical_state": physical_state,
        "physical_state_sha256": physical_state_sha,
        "optimizer_coordinates": optimizer_coordinates,
        "optimizer_coordinates_sha256": optimizer_coordinates_sha,
        "scaled_multipliers": scaled_multipliers,
        "scaled_multipliers_sha256": scaled_multipliers_sha,
        "raw_multipliers": raw_multipliers,
        "raw_multipliers_sha256": raw_multipliers_sha,
        "physical_objective": float(np.asarray(endpoint_result.physical_objective)),
        "raw_constraint_infinity_norm": float(
            np.asarray(endpoint_result.raw_constraint_infinity_norm)
        ),
        "scaled_constraint_infinity_norm": float(
            np.asarray(endpoint_result.scaled_constraint_infinity_norm)
        ),
        "raw_kkt_stationarity_infinity_norm": float(
            np.asarray(endpoint_result.raw_kkt_stationarity_infinity_norm)
        ),
        "all_finite": bool(np.asarray(endpoint_result.all_finite)),
    }
    optimizer_payload: dict[str, JsonValue] = {
        "status": status,
        "fatal": bool(np.asarray(optimizer.fatal)),
        "failed": bool(np.asarray(optimizer.failed)),
        "converged": bool(np.asarray(host["route_converged"])),
        "all_finite": bool(np.asarray(host["route_all_finite"])),
        "all_accepted_states_finite": bool(
            np.asarray(optimizer.all_accepted_states_finite)
        ),
        **{
            key: int(np.asarray(getattr(optimizer, key)))
            for key in (
                "iterations",
                "joint_evaluations",
                "derivative_builds",
                "kkt_solves",
                "line_search_evaluations",
                "rejected_nonfinite_trials",
                "bfgs_resets",
                "regularization_uses",
                "regularization_candidates_tested",
            )
        },
        **{
            key: _optional_finite_float(getattr(optimizer, key))
            for key in (
                "final_kkt_relative_residual",
                "final_kkt_reciprocal_condition",
                "final_kkt_solution_scaled_residual",
                "final_schur_relative_residual",
                "final_bfgs_cholesky_relative_pivot",
                "final_schur_cholesky_relative_pivot",
            )
        },
        "selected_regularization": _optional_finite_float(
            optimizer.selected_regularization
        ),
        "merit_penalty": float(np.asarray(optimizer.merit_penalty)),
        **{
            key: endpoint[key]
            for key in (
                "physical_objective",
                "raw_constraint_infinity_norm",
                "scaled_constraint_infinity_norm",
                "raw_kkt_stationarity_infinity_norm",
                "physical_state_sha256",
                "optimizer_coordinates_sha256",
                "scaled_multipliers_sha256",
                "raw_multipliers_sha256",
            )
        },
        "history": history,
        "history_sha256": hashlib.sha256(canonical_json_bytes(history)).hexdigest(),
        "initial_physical_objective": float(
            np.asarray(initial_endpoint_result.physical_objective)
        ),
        "initial_scaled_constraint_infinity_norm": float(
            np.asarray(initial_endpoint_result.scaled_constraint_infinity_norm)
        ),
        "initial_raw_kkt_stationarity_infinity_norm": float(
            np.asarray(initial_endpoint_result.raw_kkt_stationarity_infinity_norm)
        ),
    }
    timing: dict[str, JsonValue] = {
        "synchronized_solve_seconds": solve_elapsed_ns / 1.0e9,
        "synchronization": "block_until_ready",
        "warmup_solve_count": 1 if warm else 0,
        "pristine_inputs_restored_before_timed_solve": warm,
    }
    transfers: dict[str, JsonValue] = {
        "audit_scope": "cfs_sqp1_after_bootstrap_publication",
        "hot_h2d_calls": 0,
        "hot_d2h_calls": 0,
        "initial_h2d_calls": 1,
        "initial_h2d_bytes": staging_bytes,
        "initial_h2d_leaves": staging_leaves,
        "final_d2h_calls": 1,
        "final_d2h_bytes": result_bytes,
        "final_d2h_leaves": result_leaves,
        "timed_execution_transfer_guard": "disallow",
    }
    return optimizer_payload, endpoint, {"timing": timing, "transfers": transfers}


def _run_cfs_al_complete_probe(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    sample: CompleteSample,
    route: FullSpaceRoute,
) -> tuple[
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
]:
    """Prepare once and dispatch one complete frozen AL program once."""

    if route not in (FullSpaceRoute.CFS_AL1, FullSpaceRoute.CFS_AL2):
        raise ValueError("complete AL probe requires CFS-AL1 or CFS-AL2")
    route_name = route.value
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise ValueError(f"{route_name} complete solve requires exactly one JAX GPU")
    host_z, host_problem = jax.device_get((bootstrap.z0, bootstrap.problem))
    staging_leaves, staging_bytes = _array_tree_size((host_z, host_problem))
    device_z, device_problem = jax.device_put(
        (host_z, host_problem), device=jax.devices()[0]
    )
    jax.block_until_ready((device_z, device_problem))
    preparation_start_ns = time.perf_counter_ns()
    prepared = (
        prepare_cfs_al1(device_problem, device_z, device_z)
        if route is FullSpaceRoute.CFS_AL1
        else prepare_cfs_al2(device_problem, device_z, device_z)
    )
    jax.block_until_ready(
        (
            prepared.initial_value,
            prepared.initial_gradient,
            prepared.initial_parameters,
        )
    )
    preparation_elapsed_ns = time.perf_counter_ns() - preparation_start_ns

    execution_start_ns = time.perf_counter_ns()
    with jax.transfer_guard("disallow"):
        result = prepared.run()
        history = result.stage_history
        device_evidence = (
            result.physical_state,
            result.scaled_multipliers,
            result.raw_multipliers,
            result.next_penalty,
            result.physical_objective,
            result.raw_constraint_infinity_norm,
            result.scaled_constraint_infinity_norm,
            result.raw_kkt_stationarity_infinity_norm,
            result.completed_outer_stages,
            result.total_inner_iterations,
            result.total_function_evaluations,
            result.total_gradient_evaluations,
            result.nonfinite_evaluation_count,
            result.all_accepted_states_finite,
            result.converged,
            result.fatal,
            result.all_finite,
            history.penalty,
            history.physical_objective,
            history.scaled_feasibility_infinity_norm,
            history.augmented_stationarity_infinity_norm,
            history.optimizer_status,
            history.inner_iterations,
            history.function_evaluations,
            history.stage_completed,
        )
        jax.block_until_ready(device_evidence)
    execution_elapsed_ns = time.perf_counter_ns() - execution_start_ns
    result_leaves, result_bytes = _array_tree_size(device_evidence)
    host = tuple(np.asarray(value) for value in jax.device_get(device_evidence))
    physical_state, scaled_multipliers, raw_multipliers = host[:3]
    scalar_values = tuple(float(value) for value in host[3:8])
    completed_outer_stages, total_inner_iterations = (
        int(value) for value in host[8:10]
    )
    total_function_evaluations, total_gradient_evaluations = (
        int(value) for value in host[10:12]
    )
    nonfinite_evaluation_count = int(host[12])
    all_accepted_states_finite, converged, fatal, all_finite = (
        bool(value) for value in host[13:17]
    )
    (
        penalties,
        stage_objectives,
        stage_feasibilities,
        stage_stationarities,
        optimizer_statuses,
        inner_iterations,
        function_evaluations,
        stage_completed,
    ) = host[17:25]
    policy = route_policy(route)
    if not (
        np.all(np.isfinite(physical_state))
        and np.all(np.isfinite(scaled_multipliers))
        and np.all(np.isfinite(raw_multipliers))
        and all(np.isfinite(scalar_values))
        and np.all(np.isfinite(penalties))
        and np.all(np.isfinite(stage_objectives))
        and np.all(np.isfinite(stage_feasibilities))
        and np.all(np.isfinite(stage_stationarities))
        and all_accepted_states_finite
        and all_finite
        and not fatal
        and nonfinite_evaluation_count == 0
    ):
        raise ValueError(f"{route_name} produced non-finite or fatal solver evidence")
    if (
        completed_outer_stages != policy.maximum_outer_stages
        or not np.all(stage_completed)
        or np.any(inner_iterations < 0)
        or np.any(inner_iterations > policy.inner_iterations_per_stage)
        or np.any((optimizer_statuses != 0) & (optimizer_statuses != 1))
        or total_inner_iterations > policy.maximum_total_inner_iterations
        or int(np.sum(inner_iterations)) != total_inner_iterations
    ):
        raise ValueError(f"{route_name} violated the frozen 10-stage iteration budget")
    expected_penalties = np.minimum(
        policy.initial_penalty
        * policy.penalty_growth ** np.arange(policy.maximum_outer_stages),
        policy.maximum_penalty,
    )
    if not np.array_equal(penalties, expected_penalties):
        raise ValueError(
            f"{route_name} stage penalties differ from the frozen schedule"
        )
    if (
        np.any(function_evaluations < inner_iterations + 1)
        or np.any(
            function_evaluations > policy.maximum_function_evaluations_per_inner_solve
        )
        or int(np.sum(function_evaluations)) != total_function_evaluations
        or total_gradient_evaluations != total_function_evaluations
    ):
        raise ValueError(f"{route_name} violated per-stage evaluation-budget integrity")
    made_feasibility_progress = bool(stage_feasibilities[-1] < stage_feasibilities[0])

    numerical: dict[str, JsonValue] = {
        "all_accepted_states_finite": all_accepted_states_finite,
        "all_finite": all_finite,
        "completed_outer_stages": completed_outer_stages,
        "converged": converged,
        "fatal": fatal,
        "final_physical_objective": scalar_values[1],
        "final_raw_feasibility_inf": scalar_values[2],
        "final_raw_kkt_stationarity_inf": scalar_values[4],
        "final_scaled_feasibility_inf": scalar_values[3],
        "made_feasibility_progress": made_feasibility_progress,
        "next_penalty": scalar_values[0],
        "nonfinite_evaluation_count": nonfinite_evaluation_count,
        "total_function_evaluations": total_function_evaluations,
        "total_gradient_evaluations": total_gradient_evaluations,
        "total_inner_iterations": total_inner_iterations,
    }
    stage_history: dict[str, JsonValue] = {
        "augmented_stationarity_inf": stage_stationarities.tolist(),
        "function_evaluations": function_evaluations.tolist(),
        "inner_iterations": inner_iterations.tolist(),
        "optimizer_status": optimizer_statuses.tolist(),
        "penalty": penalties.tolist(),
        "physical_objective": stage_objectives.tolist(),
        "scaled_feasibility_inf": stage_feasibilities.tolist(),
        "stage_completed": stage_completed.tolist(),
    }
    endpoint: dict[str, JsonValue] = {
        "dtype": "float64",
        "physical_state": physical_state.tolist(),
        "raw_multipliers": raw_multipliers.tolist(),
        "scaled_multipliers": scaled_multipliers.tolist(),
    }
    solve_timing_key = (
        "cold_synchronized_solve_ns"
        if sample is CompleteSample.COLD
        else "warm_synchronized_solve_ns"
    )
    timing: dict[str, JsonValue] = {
        "preparation_and_compile_ns": preparation_elapsed_ns,
        "sample": sample.value,
        solve_timing_key: execution_elapsed_ns,
        "synchronization": "block_until_ready",
    }
    transfers: dict[str, JsonValue] = {
        "audit_scope": f"{route_name.lower().replace('-', '_')}_after_bootstrap_publication",
        "final_d2h_bytes": result_bytes,
        "final_d2h_calls": 1,
        "final_d2h_leaves": result_leaves,
        "hot_d2h_calls": 0,
        "initial_h2d_bytes": staging_bytes,
        "initial_h2d_calls": 1,
        "initial_h2d_leaves": staging_leaves,
        "initial_host_staging_d2h_bytes": staging_bytes,
        "initial_host_staging_d2h_calls": 1,
        "timed_execution_transfer_guard": "disallow",
    }
    return numerical, stage_history, endpoint, timing, transfers


def run_cfs_al1_complete_probe(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    sample: CompleteSample,
) -> tuple[
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
]:
    """Run the complete CFS-AL1 device program."""

    return _run_cfs_al_complete_probe(
        bootstrap,
        sample=sample,
        route=FullSpaceRoute.CFS_AL1,
    )


def run_cfs_al2_complete_probe(
    bootstrap: SingleStageFullSpaceBootstrap,
    *,
    sample: CompleteSample,
) -> tuple[
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
    dict[str, JsonValue],
]:
    """Run the complete CFS-AL2 device program."""

    return _run_cfs_al_complete_probe(
        bootstrap,
        sample=sample,
        route=FullSpaceRoute.CFS_AL2,
    )


def publish_first_eval_evidence(
    path: Path,
    *,
    request: RunRequest,
    campaign_root: Path,
    snapshot_root: Path,
    runtime_evidence: RuntimeEvidence,
    runtime_evidence_ref: ArtifactRef,
    bootstrap_artifact_ref: ArtifactRef,
    bootstrap: SingleStageFullSpaceBootstrap,
) -> ArtifactRef:
    """Exclusively publish a diagnostic first-eval artifact, never a solve receipt."""

    request.validate()
    if request.phase is not RunPhase.FIRST_EVAL:
        raise ValueError("first-eval evidence requires the first-eval phase")
    campaign = campaign_root.resolve(strict=True)
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.resolve(strict=True).is_relative_to(campaign):
        raise ValueError("first-eval evidence must be campaign-local")
    if bootstrap_artifact_ref.schema_version != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("first-eval bootstrap reference has the wrong schema")
    bootstrap_document = validate_bootstrap_artifact(
        bootstrap_artifact_ref.resolve_and_validate(campaign),
        campaign_root=campaign,
        snapshot_root=snapshot_root,
    )
    if bootstrap_document["runtime_evidence"] != asdict(runtime_evidence_ref):
        raise ValueError("first-eval bootstrap and runtime evidence differ")
    runtime_payload = canonical_json_bytes(runtime_evidence.to_payload())
    if runtime_evidence_ref.sha256 != hashlib.sha256(
        runtime_payload
    ).hexdigest() or runtime_evidence_ref.size_bytes != len(runtime_payload):
        raise ValueError("first-eval runtime reference differs from runtime evidence")
    requested_name = "5090" if request.device is DeviceLane.RTX5090 else "A100"
    if requested_name.lower() not in runtime_evidence.observation.device_name.lower():
        raise ValueError("requested device lane differs from the physical GPU")

    numerical, timing, transfer_audit = run_first_eval_probe(bootstrap)
    payload: dict[str, JsonValue] = {
        "bootstrap_artifact": asdict(bootstrap_artifact_ref),
        "endpoint_certificate_produced": False,
        "numerical_evidence": numerical,
        "request": asdict(request),
        "runtime_evidence": asdict(runtime_evidence_ref),
        "schema_version": FIRST_EVAL_SCHEMA_VERSION,
        "source_identity": asdict(runtime_evidence.source_identity),
        "terminal_status": "DIAGNOSTIC_SUCCESS",
        "timing": timing,
        "transfer_audit": transfer_audit,
    }
    encoded = canonical_json_bytes(payload)
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    return ArtifactRef(
        output.relative_to(campaign).as_posix(),
        hashlib.sha256(encoded).hexdigest(),
        len(encoded),
        FIRST_EVAL_SCHEMA_VERSION,
    )


def publish_cfs_p0_canary_evidence(
    path: Path,
    *,
    request: RunRequest,
    campaign_root: Path,
    snapshot_root: Path,
    runtime_evidence: RuntimeEvidence,
    runtime_evidence_ref: ArtifactRef,
    bootstrap_artifact_ref: ArtifactRef,
    bootstrap: SingleStageFullSpaceBootstrap,
) -> ArtifactRef:
    """Publish one provenance-bound diagnostic CFS-P0 10-step artifact."""

    request.validate()
    if (
        request.phase is not RunPhase.CANARY
        or request.route is not FullSpaceRoute.CFS_P0
        or request.steps not in (10, 100)
    ):
        raise ValueError("CFS-P0 canary evidence requires CFS-P0 and 10 or 100 steps")
    campaign = campaign_root.resolve(strict=True)
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.resolve(strict=True).is_relative_to(campaign):
        raise ValueError("CFS-P0 canary evidence must be campaign-local")
    if bootstrap_artifact_ref.schema_version != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("CFS-P0 bootstrap reference has the wrong schema")
    bootstrap_document = validate_bootstrap_artifact(
        bootstrap_artifact_ref.resolve_and_validate(campaign),
        campaign_root=campaign,
        snapshot_root=snapshot_root,
    )
    if bootstrap_document["runtime_evidence"] != asdict(runtime_evidence_ref):
        raise ValueError("CFS-P0 bootstrap and runtime evidence differ")
    runtime_payload = canonical_json_bytes(runtime_evidence.to_payload())
    if runtime_evidence_ref.sha256 != hashlib.sha256(
        runtime_payload
    ).hexdigest() or runtime_evidence_ref.size_bytes != len(runtime_payload):
        raise ValueError("CFS-P0 runtime reference differs from runtime evidence")
    requested_name = "5090" if request.device is DeviceLane.RTX5090 else "A100"
    if requested_name.lower() not in runtime_evidence.observation.device_name.lower():
        raise ValueError("requested device lane differs from the physical GPU")

    numerical, timing, transfer_audit = run_cfs_p0_canary_probe(
        bootstrap,
        maximum_iterations=request.steps,
    )
    if transfer_audit.get("hot_d2h_calls") != 0:
        raise ValueError("CFS-P0 canary violated the hot D2H gate")
    schema_version = (
        CFS_P0_CANARY_SCHEMA_VERSION
        if request.steps == 10
        else CFS_P0_CANARY_100_SCHEMA_VERSION
    )
    payload: dict[str, JsonValue] = {
        "bootstrap_artifact": asdict(bootstrap_artifact_ref),
        "endpoint_certificate_produced": False,
        "numerical_evidence": numerical,
        "request": asdict(request),
        "runtime_evidence": asdict(runtime_evidence_ref),
        "schema_version": schema_version,
        "source_identity": asdict(runtime_evidence.source_identity),
        "terminal_status": "DIAGNOSTIC_SUCCESS",
        "timing": timing,
        "transfer_audit": transfer_audit,
    }
    encoded = canonical_json_bytes(payload)
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    return ArtifactRef(
        output.relative_to(campaign).as_posix(),
        hashlib.sha256(encoded).hexdigest(),
        len(encoded),
        schema_version,
    )


def _cfs_al1_result_relative_path(sample: CompleteSample) -> str:
    return f"runs/cfs-al1-{sample.value}.json"


def _cfs_al2_result_relative_path(sample: CompleteSample) -> str:
    return f"runs/cfs-al2-{sample.value}.json"


def publish_cfs_al1_result(
    path: Path,
    *,
    request: RunRequest,
    campaign_root: Path,
    snapshot_root: Path,
    runtime_evidence: RuntimeEvidence,
    runtime_evidence_ref: ArtifactRef,
    bootstrap_artifact_ref: ArtifactRef,
    bootstrap: SingleStageFullSpaceBootstrap,
) -> ArtifactRef:
    """Seal an AL1 solver result that remains non-promoting until certified."""

    request.validate()
    if (
        request.phase is not RunPhase.COMPLETE
        or request.route is not FullSpaceRoute.CFS_AL1
        or request.sample is None
    ):
        raise ValueError("CFS-AL1 result requires a complete CFS-AL1 sample")
    campaign = campaign_root.resolve(strict=True)
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.resolve(strict=True).is_relative_to(campaign):
        raise ValueError("CFS-AL1 result must be campaign-local")
    if bootstrap_artifact_ref.schema_version != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("CFS-AL1 bootstrap reference has the wrong schema")
    bootstrap_document = validate_bootstrap_artifact(
        bootstrap_artifact_ref.resolve_and_validate(campaign),
        campaign_root=campaign,
        snapshot_root=snapshot_root,
    )
    if bootstrap_document["runtime_evidence"] != asdict(runtime_evidence_ref):
        raise ValueError("CFS-AL1 bootstrap and runtime evidence differ")
    runtime_payload = canonical_json_bytes(runtime_evidence.to_payload())
    if runtime_evidence_ref.sha256 != hashlib.sha256(
        runtime_payload
    ).hexdigest() or runtime_evidence_ref.size_bytes != len(runtime_payload):
        raise ValueError("CFS-AL1 runtime reference differs from runtime evidence")
    requested_name = "5090" if request.device is DeviceLane.RTX5090 else "A100"
    if requested_name.lower() not in runtime_evidence.observation.device_name.lower():
        raise ValueError("requested device lane differs from the physical GPU")

    numerical, stage_history, endpoint, timing, transfer_audit = (
        run_cfs_al1_complete_probe(bootstrap, sample=request.sample)
    )
    if transfer_audit.get("hot_d2h_calls") != 0:
        raise ValueError("CFS-AL1 violated the hot D2H gate")
    converged = numerical.get("converged") is True
    payload: dict[str, JsonValue] = {
        "bootstrap_artifact": asdict(bootstrap_artifact_ref),
        "endpoint": endpoint,
        "endpoint_certificate": None,
        "numerical_evidence": numerical,
        "promotion_eligible": False,
        "request": asdict(request),
        "runtime_evidence": asdict(runtime_evidence_ref),
        "schema_version": CFS_AL1_RESULT_SCHEMA_VERSION,
        "source_identity": asdict(runtime_evidence.source_identity),
        "stage_history": stage_history,
        "terminal_status": (
            "SOLVER_RESULT_AWAITING_ENDPOINT_CERTIFICATE"
            if converged
            else "SOLVER_RESULT_NOT_CONVERGED"
        ),
        "timing": timing,
        "trajectory_equivalence_required": False,
        "transfer_audit": transfer_audit,
    }
    encoded = canonical_json_bytes(payload)
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    return ArtifactRef(
        output.relative_to(campaign).as_posix(),
        hashlib.sha256(encoded).hexdigest(),
        len(encoded),
        CFS_AL1_RESULT_SCHEMA_VERSION,
    )


def publish_cfs_al2_result(
    path: Path,
    *,
    request: RunRequest,
    campaign_root: Path,
    snapshot_root: Path,
    runtime_evidence: RuntimeEvidence,
    runtime_evidence_ref: ArtifactRef,
    bootstrap_artifact_ref: ArtifactRef,
    bootstrap: SingleStageFullSpaceBootstrap,
) -> ArtifactRef:
    """Seal an AL2 solver result that remains non-promoting until certified."""

    request.validate()
    if (
        request.phase is not RunPhase.COMPLETE
        or request.route is not FullSpaceRoute.CFS_AL2
        or request.sample is None
    ):
        raise ValueError("CFS-AL2 result requires a complete CFS-AL2 sample")
    campaign = campaign_root.resolve(strict=True)
    output = path.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.resolve(strict=True).is_relative_to(campaign):
        raise ValueError("CFS-AL2 result must be campaign-local")
    if bootstrap_artifact_ref.schema_version != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("CFS-AL2 bootstrap reference has the wrong schema")
    bootstrap_document = validate_bootstrap_artifact(
        bootstrap_artifact_ref.resolve_and_validate(campaign),
        campaign_root=campaign,
        snapshot_root=snapshot_root,
    )
    if bootstrap_document["runtime_evidence"] != asdict(runtime_evidence_ref):
        raise ValueError("CFS-AL2 bootstrap and runtime evidence differ")
    runtime_payload = canonical_json_bytes(runtime_evidence.to_payload())
    if runtime_evidence_ref.sha256 != hashlib.sha256(
        runtime_payload
    ).hexdigest() or runtime_evidence_ref.size_bytes != len(runtime_payload):
        raise ValueError("CFS-AL2 runtime reference differs from runtime evidence")
    requested_name = "5090" if request.device is DeviceLane.RTX5090 else "A100"
    if requested_name.lower() not in runtime_evidence.observation.device_name.lower():
        raise ValueError("requested device lane differs from the physical GPU")

    numerical, stage_history, endpoint, timing, transfer_audit = (
        run_cfs_al2_complete_probe(bootstrap, sample=request.sample)
    )
    if transfer_audit.get("hot_d2h_calls") != 0:
        raise ValueError("CFS-AL2 violated the hot D2H gate")
    converged = numerical.get("converged") is True
    payload: dict[str, JsonValue] = {
        "bootstrap_artifact": asdict(bootstrap_artifact_ref),
        "endpoint": endpoint,
        "endpoint_certificate": None,
        "numerical_evidence": numerical,
        "promotion_eligible": False,
        "request": asdict(request),
        "runtime_evidence": asdict(runtime_evidence_ref),
        "schema_version": CFS_AL2_RESULT_SCHEMA_VERSION,
        "source_identity": asdict(runtime_evidence.source_identity),
        "stage_history": stage_history,
        "terminal_status": (
            "SOLVER_RESULT_AWAITING_ENDPOINT_CERTIFICATE"
            if converged
            else "SOLVER_RESULT_NOT_CONVERGED"
        ),
        "timing": timing,
        "trajectory_equivalence_required": False,
        "transfer_audit": transfer_audit,
    }
    encoded = canonical_json_bytes(payload)
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    return ArtifactRef(
        output.relative_to(campaign).as_posix(),
        hashlib.sha256(encoded).hexdigest(),
        len(encoded),
        CFS_AL2_RESULT_SCHEMA_VERSION,
    )


def execute_first_eval_snapshot_child(
    request: RunRequest,
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
) -> bytes:
    """Publish runtime, bootstrap, and first-eval evidence from the sealed child."""

    if request.phase is not RunPhase.FIRST_EVAL:
        raise PhaseGateError("only the first-eval snapshot child is implemented")
    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(campaign / SNAPSHOT_DIRECTORY)
    if (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
        or environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256
    ):
        raise ValueError("snapshot child is not bound to this campaign and manifest")
    if Path.cwd().resolve(strict=True) != publication.root:
        raise ValueError("snapshot child must execute from the immutable snapshot root")

    runtime, runtime_ref = publish_child_runtime_provenance(
        publication,
        campaign_root=campaign,
        process_argv=process_argv,
        environment=environment,
    )
    runtime_ref.resolve_and_validate(campaign).chmod(0o444)
    (campaign / "artifacts").mkdir()
    bootstrap = build_single_stage_fullspace_bootstrap()
    bootstrap_ref = publish_bootstrap_artifact(
        campaign / BOOTSTRAP_RELATIVE_PATH,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime_ref,
        bootstrap_factory=lambda: bootstrap,
    )
    (campaign / "runs").mkdir()
    first_eval_ref = publish_first_eval_evidence(
        campaign / FIRST_EVAL_RELATIVE_PATH,
        request=request,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime,
        runtime_evidence_ref=runtime_ref,
        bootstrap_artifact_ref=bootstrap_ref,
        bootstrap=bootstrap,
    )
    return first_eval_ref.resolve_and_validate(campaign).read_bytes()


def execute_cfs_p0_canary_snapshot_child(
    request: RunRequest,
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
) -> bytes:
    """Publish runtime, bootstrap, and CFS-P0 canary evidence in the sealed child."""

    if (
        request.phase is not RunPhase.CANARY
        or request.route is not FullSpaceRoute.CFS_P0
        or request.steps not in (10, 100)
    ):
        raise PhaseGateError("only the 10/100-step CFS-P0 canary child is implemented")
    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(campaign / SNAPSHOT_DIRECTORY)
    if (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
        or environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256
    ):
        raise ValueError("snapshot child is not bound to this campaign and manifest")
    if Path.cwd().resolve(strict=True) != publication.root:
        raise ValueError("snapshot child must execute from the immutable snapshot root")

    runtime, runtime_ref = publish_child_runtime_provenance(
        publication,
        campaign_root=campaign,
        process_argv=process_argv,
        environment=environment,
    )
    runtime_ref.resolve_and_validate(campaign).chmod(0o444)
    (campaign / "artifacts").mkdir()
    bootstrap = build_single_stage_fullspace_bootstrap()
    bootstrap_ref = publish_bootstrap_artifact(
        campaign / BOOTSTRAP_RELATIVE_PATH,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime_ref,
        bootstrap_factory=lambda: bootstrap,
    )
    (campaign / "runs").mkdir()
    relative_path = (
        CFS_P0_CANARY_10_RELATIVE_PATH
        if request.steps == 10
        else CFS_P0_CANARY_100_RELATIVE_PATH
    )
    canary_ref = publish_cfs_p0_canary_evidence(
        campaign / relative_path,
        request=request,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime,
        runtime_evidence_ref=runtime_ref,
        bootstrap_artifact_ref=bootstrap_ref,
        bootstrap=bootstrap,
    )
    return canary_ref.resolve_and_validate(campaign).read_bytes()


def execute_cfs_al1_snapshot_child(
    request: RunRequest,
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
) -> bytes:
    """Publish runtime, bootstrap, and sealed AL1 result in the snapshot child."""

    if (
        request.phase is not RunPhase.COMPLETE
        or request.route is not FullSpaceRoute.CFS_AL1
        or request.sample is None
    ):
        raise PhaseGateError("only complete CFS-AL1 samples are implemented")
    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(campaign / SNAPSHOT_DIRECTORY)
    if (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
        or environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256
    ):
        raise ValueError("snapshot child is not bound to this campaign and manifest")
    if Path.cwd().resolve(strict=True) != publication.root:
        raise ValueError("snapshot child must execute from the immutable snapshot root")

    runtime, runtime_ref = publish_child_runtime_provenance(
        publication,
        campaign_root=campaign,
        process_argv=process_argv,
        environment=environment,
    )
    runtime_ref.resolve_and_validate(campaign).chmod(0o444)
    (campaign / "artifacts").mkdir()
    bootstrap = build_single_stage_fullspace_bootstrap()
    bootstrap_ref = publish_bootstrap_artifact(
        campaign / BOOTSTRAP_RELATIVE_PATH,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime_ref,
        bootstrap_factory=lambda: bootstrap,
    )
    (campaign / "runs").mkdir()
    result_ref = publish_cfs_al1_result(
        campaign / _cfs_al1_result_relative_path(request.sample),
        request=request,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime,
        runtime_evidence_ref=runtime_ref,
        bootstrap_artifact_ref=bootstrap_ref,
        bootstrap=bootstrap,
    )
    return result_ref.resolve_and_validate(campaign).read_bytes()


def execute_cfs_al2_snapshot_child(
    request: RunRequest,
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
) -> bytes:
    """Publish runtime, bootstrap, and sealed AL2 result in the snapshot child."""

    if (
        request.phase is not RunPhase.COMPLETE
        or request.route is not FullSpaceRoute.CFS_AL2
        or request.sample is None
    ):
        raise PhaseGateError("only complete CFS-AL2 samples are implemented")
    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(campaign / SNAPSHOT_DIRECTORY)
    if (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
        or environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256
    ):
        raise ValueError("snapshot child is not bound to this campaign and manifest")
    if Path.cwd().resolve(strict=True) != publication.root:
        raise ValueError("snapshot child must execute from the immutable snapshot root")

    runtime, runtime_ref = publish_child_runtime_provenance(
        publication,
        campaign_root=campaign,
        process_argv=process_argv,
        environment=environment,
    )
    runtime_ref.resolve_and_validate(campaign).chmod(0o444)
    (campaign / "artifacts").mkdir()
    bootstrap = build_single_stage_fullspace_bootstrap()
    bootstrap_ref = publish_bootstrap_artifact(
        campaign / BOOTSTRAP_RELATIVE_PATH,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime_ref,
        bootstrap_factory=lambda: bootstrap,
    )
    (campaign / "runs").mkdir()
    result_ref = publish_cfs_al2_result(
        campaign / _cfs_al2_result_relative_path(request.sample),
        request=request,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime,
        runtime_evidence_ref=runtime_ref,
        bootstrap_artifact_ref=bootstrap_ref,
        bootstrap=bootstrap,
    )
    return result_ref.resolve_and_validate(campaign).read_bytes()


def run_first_eval_campaign(
    request: RunRequest,
    campaign_root: Path,
    *,
    native_extension_path: Path,
    interpreter: Path,
    environment: Mapping[str, str],
) -> bytes:
    """Snapshot the live tree, execute one isolated GPU child, and bind its bytes."""

    if request.phase is not RunPhase.FIRST_EVAL:
        raise PhaseGateError("only the first-eval campaign is implemented")
    publication = prepare_execution_snapshot(
        campaign_root,
        native_extension_path=native_extension_path,
    )
    child_output = campaign_root / FIRST_EVAL_RELATIVE_PATH
    request_argv = (
        "--phase",
        request.phase.value,
        "--route",
        request.route.value,
        "--device",
        request.device.value,
        "--output",
        str(child_output),
        "--snapshot-child",
    )
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=interpreter,
        request_argv=request_argv,
        environment=environment,
    )
    completed = subprocess.run(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot first-eval child failed with exit code "
            f"{completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    if child_output.read_bytes() != completed.stdout or (
        child_output.stat().st_mode & 0o222
    ):
        raise ValueError("persisted first-eval evidence differs or is writable")
    return completed.stdout


def run_cfs_p0_canary_campaign(
    request: RunRequest,
    campaign_root: Path,
    *,
    native_extension_path: Path,
    interpreter: Path,
    environment: Mapping[str, str],
) -> bytes:
    """Snapshot the live tree and execute one isolated staged CFS-P0 child."""

    if (
        request.phase is not RunPhase.CANARY
        or request.route is not FullSpaceRoute.CFS_P0
        or request.steps not in (10, 100)
    ):
        raise PhaseGateError("only the 10/100-step CFS-P0 canary is implemented")
    publication = prepare_execution_snapshot(
        campaign_root,
        native_extension_path=native_extension_path,
    )
    relative_path = (
        CFS_P0_CANARY_10_RELATIVE_PATH
        if request.steps == 10
        else CFS_P0_CANARY_100_RELATIVE_PATH
    )
    child_output = campaign_root / relative_path
    request_argv = (
        "--phase",
        request.phase.value,
        "--route",
        request.route.value,
        "--device",
        request.device.value,
        "--steps",
        str(request.steps),
        "--output",
        str(child_output),
        "--snapshot-child",
    )
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=interpreter,
        request_argv=request_argv,
        environment=environment,
    )
    completed = subprocess.run(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot CFS-P0 canary child failed with exit code "
            f"{completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    if child_output.read_bytes() != completed.stdout or (
        child_output.stat().st_mode & 0o222
    ):
        raise ValueError("persisted CFS-P0 canary evidence differs or is writable")
    return completed.stdout


def run_cfs_al1_campaign(
    request: RunRequest,
    campaign_root: Path,
    *,
    native_extension_path: Path,
    interpreter: Path,
    environment: Mapping[str, str],
) -> bytes:
    """Snapshot the live tree and execute one isolated complete CFS-AL1 child."""

    if (
        request.phase is not RunPhase.COMPLETE
        or request.route is not FullSpaceRoute.CFS_AL1
        or request.sample is None
    ):
        raise PhaseGateError("only complete CFS-AL1 samples are implemented")
    publication = prepare_execution_snapshot(
        campaign_root,
        native_extension_path=native_extension_path,
    )
    child_output = campaign_root / _cfs_al1_result_relative_path(request.sample)
    request_argv = (
        "--phase",
        request.phase.value,
        "--route",
        request.route.value,
        "--device",
        request.device.value,
        "--sample",
        request.sample.value,
        "--output",
        str(child_output),
        "--snapshot-child",
    )
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=interpreter,
        request_argv=request_argv,
        environment=environment,
    )
    completed = subprocess.run(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot CFS-AL1 child failed with exit code "
            f"{completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    if child_output.read_bytes() != completed.stdout or (
        child_output.stat().st_mode & 0o222
    ):
        raise ValueError("persisted CFS-AL1 result differs or is writable")
    return completed.stdout


def run_cfs_al2_campaign(
    request: RunRequest,
    campaign_root: Path,
    *,
    native_extension_path: Path,
    interpreter: Path,
    environment: Mapping[str, str],
) -> bytes:
    """Snapshot the live tree and execute one isolated complete CFS-AL2 child."""

    if (
        request.phase is not RunPhase.COMPLETE
        or request.route is not FullSpaceRoute.CFS_AL2
        or request.sample is None
    ):
        raise PhaseGateError("only complete CFS-AL2 samples are implemented")
    publication = prepare_execution_snapshot(
        campaign_root,
        native_extension_path=native_extension_path,
    )
    child_output = campaign_root / _cfs_al2_result_relative_path(request.sample)
    request_argv = (
        "--phase",
        request.phase.value,
        "--route",
        request.route.value,
        "--device",
        request.device.value,
        "--sample",
        request.sample.value,
        "--output",
        str(child_output),
        "--snapshot-child",
    )
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=interpreter,
        request_argv=request_argv,
        environment=environment,
    )
    completed = subprocess.run(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot CFS-AL2 child failed with exit code "
            f"{completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    if child_output.read_bytes() != completed.stdout or (
        child_output.stat().st_mode & 0o222
    ):
        raise ValueError("persisted CFS-AL2 result differs or is writable")
    return completed.stdout


def explicit_source_roots(
    repo_root: Path, native_extension_path: Path
) -> tuple[SourceRoot, ...]:
    """Return the finite Phase-0 source selection used by every GPU run."""

    root = repo_root.resolve(strict=True)
    native = native_extension_path.absolute()
    listed = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src/simsopt",
            "src/simsopt_jax",
            "src/simsopt_jax_adapters",
        ),
        check=False,
        capture_output=True,
    )
    if listed.returncode != 0:
        raise ValueError("failed to enumerate repository execution sources")
    source_files = tuple(
        sorted(path.decode("utf-8") for path in listed.stdout.split(b"\0")[:-1])
    )
    if not source_files:
        raise ValueError("repository execution source selection is empty")
    roots = [
        *(
            SourceRoot("execution_source", root / relative, relative)
            for relative in source_files
        ),
        *(
            SourceRoot("configuration", root / relative, relative)
            for relative in _CONFIGURATION_FILES
        ),
        *(
            SourceRoot("benchmark", root / relative, relative)
            for relative in _BENCHMARK_FILES
        ),
        *(SourceRoot("test", root / relative, relative) for relative in _TEST_FILES),
        SourceRoot("native_extension", native, f"src/{native.name}"),
    ]
    return tuple(roots)


def prepare_execution_snapshot(
    campaign_root: Path, *, native_extension_path: Path
) -> SnapshotPublication:
    """Exclusively create a campaign root and seal its exact execution tree."""

    output = campaign_root.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output path: {output}")
    if output.is_relative_to(_REPO_ROOT):
        raise ValueError("campaign output must be outside the source repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    worktree = capture_worktree_identity(_REPO_ROOT)
    output.mkdir()
    publication = publish_immutable_snapshot(
        output / SNAPSHOT_DIRECTORY,
        explicit_source_roots(_REPO_ROOT, native_extension_path),
        worktree=worktree,
    )
    if capture_worktree_identity(_REPO_ROOT) != worktree:
        raise ValueError("repository source state changed during snapshot publication")
    return publication


def prepare_or_load_execution_snapshot(
    campaign_root: Path, *, native_extension_path: Path
) -> SnapshotPublication:
    """Create one campaign snapshot or continue from its immutable bytes."""

    output = campaign_root.absolute()
    if not output.exists() and not output.is_symlink():
        return prepare_execution_snapshot(
            output, native_extension_path=native_extension_path
        )
    if output.is_symlink() or not output.is_dir():
        raise ValueError("SQP campaign root must be a real directory")
    if output.is_relative_to(_REPO_ROOT):
        raise ValueError("campaign output must be outside the source repository")
    return load_snapshot(output / SNAPSHOT_DIRECTORY)


def build_snapshot_child_invocation(
    publication: SnapshotPublication,
    *,
    campaign_root: Path,
    interpreter: Path,
    request_argv: Sequence[str],
    environment: Mapping[str, str],
    entrypoint_relative_path: str = "benchmarks/run_single_stage_fullspace_gpu.py",
) -> SnapshotChildInvocation:
    """Build the isolated re-exec that makes snapshot/src authoritative."""

    executable = Path(os.path.abspath(interpreter))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("snapshot child interpreter is not executable")
    runner = publication.root / entrypoint_relative_path
    entry = next(
        (
            item
            for item in publication.entries
            if item.relative_path == entrypoint_relative_path
        ),
        None,
    )
    if entry is None or entry.role != "benchmark" or not runner.is_file():
        raise ValueError("snapshot child runner is absent from the manifest")
    child_environment = dict(environment)
    child_environment[_SNAPSHOT_MANIFEST_ENV] = publication.manifest_sha256
    child_environment[_CAMPAIGN_ROOT_ENV] = str(campaign_root.resolve(strict=True))
    child_argv = (str(executable), "-I", str(runner), *request_argv)
    return SnapshotChildInvocation(
        argv=child_argv,
        cwd=publication.root,
        environment=child_environment,
    )


def publish_child_runtime_provenance(
    publication: SnapshotPublication,
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
    evidence_relative_directory: Path = Path("evidence"),
) -> tuple[RuntimeEvidence, ArtifactRef]:
    """Observe the snapshot child and exclusively publish its dynamic evidence."""

    campaign = campaign_root.resolve(strict=True)
    if environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256 or (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
    ):
        raise ValueError("snapshot child launch binding is absent or inconsistent")
    source_identity = publication.source_identity(campaign)
    observation = observe_live_runtime(
        publication.root,
        argv=process_argv,
        cwd=publication.root,
        environment=environment,
    )
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=source_identity,
        observation=observation,
    )
    evidence_directory = campaign / evidence_relative_directory
    evidence_directory.mkdir(parents=True)
    artifact = publish_runtime_evidence(
        evidence_directory / RUNTIME_EVIDENCE_FILENAME,
        evidence,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    return evidence, artifact


def _publish_immutable_json(
    path: Path,
    payload: Mapping[str, JsonValue],
    *,
    campaign_root: Path,
    schema_version: str,
) -> ArtifactRef:
    encoded = canonical_json_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return ArtifactRef(
        relative_path=path.relative_to(campaign_root).as_posix(),
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        schema_version=schema_version,
    )


def _sqp_canary_gate(
    request: RunRequest,
    raw: Mapping[str, JsonValue],
    memory: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Derive the immutable one- or ten-step gate verdict from raw evidence."""

    if request.phase is not RunPhase.CANARY or request.steps not in (1, 10):
        raise ValueError("SQP canary gate requires an exact 1- or 10-step request")
    optimizer = raw.get("optimizer_result")
    endpoint = raw.get("endpoint")
    transfers = raw.get("transfer_audit")
    timing = raw.get("timing")
    if not all(
        isinstance(section, dict)
        for section in (optimizer, endpoint, transfers, timing)
    ):
        raise TypeError("SQP canary raw evidence sections are malformed")
    assert isinstance(optimizer, dict)
    assert isinstance(endpoint, dict)
    assert isinstance(transfers, dict)
    assert isinstance(timing, dict)
    failure_reasons: list[JsonValue] = []
    if not (
        optimizer.get("all_finite") is True
        and optimizer.get("all_accepted_states_finite") is True
    ):
        failure_reasons.append("OPTIMIZER_NOT_FINITE")
    if endpoint.get("all_finite") is not True:
        failure_reasons.append("ENDPOINT_NOT_FINITE")
    if optimizer.get("fatal") is not False or optimizer.get("failed") is not False:
        failure_reasons.append("FATAL_STATUS")
    if optimizer.get("iterations") != request.steps:
        failure_reasons.append("ITERATION_COUNT")
    history = optimizer.get("history")
    if not isinstance(history, dict) or history.get("accepted_length") != request.steps:
        failure_reasons.append("ACCEPTED_HISTORY_LENGTH")
    if transfers.get("hot_h2d_calls") != 0 or transfers.get("hot_d2h_calls") != 0:
        failure_reasons.append("HOT_TRANSFER")
    if transfers.get("initial_h2d_calls") != 1 or transfers.get("final_d2h_calls") != 1:
        failure_reasons.append("BOUNDARY_TRANSFER")
    peak_fraction = memory.get("peak_memory_fraction")
    if (
        isinstance(peak_fraction, bool)
        or not isinstance(peak_fraction, (int, float))
        or not np.isfinite(peak_fraction)
        or float(peak_fraction) >= SQP_MAXIMUM_MEMORY_FRACTION
    ):
        failure_reasons.append("MEMORY_BUDGET")
    rho_k = optimizer.get("final_kkt_reciprocal_condition")
    zeta_2 = optimizer.get("final_kkt_solution_scaled_residual")
    kkt_relative = optimizer.get("final_kkt_relative_residual")
    schur_relative = optimizer.get("final_schur_relative_residual")
    if not (
        isinstance(rho_k, (int, float))
        and not isinstance(rho_k, bool)
        and np.isfinite(rho_k)
        and isinstance(zeta_2, (int, float))
        and not isinstance(zeta_2, bool)
        and np.isfinite(zeta_2)
        and float(rho_k) > float(zeta_2)
        and float(zeta_2) <= SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM
        and float(zeta_2) / (float(rho_k) - float(zeta_2))
        < SQP_KKT_FORWARD_ERROR_MAXIMUM
        and isinstance(kkt_relative, (int, float))
        and not isinstance(kkt_relative, bool)
        and np.isfinite(kkt_relative)
        and float(kkt_relative) <= SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM
        and isinstance(schur_relative, (int, float))
        and not isinstance(schur_relative, bool)
        and np.isfinite(schur_relative)
        and float(schur_relative) <= SQP_KKT_SOLUTION_SCALED_RESIDUAL_MAXIMUM
    ):
        failure_reasons.append("KKT_CERTIFICATE")
    gate: dict[str, JsonValue] = {
        "schema_version": (
            CFS_SQP1_CANARY_1_GATE_SCHEMA_VERSION
            if request.steps == 1
            else CFS_SQP1_CANARY_10_GATE_SCHEMA_VERSION
        ),
        "gate_status": "PASS" if not failure_reasons else "FAIL",
        "failure_reasons": failure_reasons,
        "expected_iterations": request.steps,
        "initial_state": "changed" if request.steps == 1 else "bootstrap",
    }
    if request.steps == 10:
        synchronized_seconds = timing.get("synchronized_solve_seconds")
        if (
            isinstance(synchronized_seconds, bool)
            or not isinstance(synchronized_seconds, (int, float))
            or not np.isfinite(synchronized_seconds)
        ):
            raise ValueError("SQP ten-step synchronized timing is invalid")
        projected_seconds = 10.0 * float(synchronized_seconds)
        initial_objective = optimizer.get("initial_physical_objective")
        final_objective = endpoint.get("physical_objective")
        initial_feasibility = optimizer.get("initial_scaled_constraint_infinity_norm")
        final_feasibility = endpoint.get("scaled_constraint_infinity_norm")
        initial_stationarity = optimizer.get(
            "initial_raw_kkt_stationarity_infinity_norm"
        )
        final_stationarity = endpoint.get("raw_kkt_stationarity_infinity_norm")
        progress_values = (
            initial_objective,
            final_objective,
            initial_feasibility,
            final_feasibility,
            initial_stationarity,
            final_stationarity,
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in progress_values
        ):
            failure_reasons.append("PROGRESS_EVIDENCE_NONFINITE")
        else:
            if not float(final_objective) < float(initial_objective):
                failure_reasons.append("OBJECTIVE_NOT_DECREASED")
            if not (
                float(final_feasibility) <= 1.0e-10
                or float(final_feasibility) < float(initial_feasibility)
            ):
                failure_reasons.append("FEASIBILITY_NOT_MAINTAINED_OR_DECREASED")
            if not float(final_stationarity) < float(initial_stationarity):
                failure_reasons.append("RAW_KKT_NOT_DECREASED")
        if projected_seconds >= SQP_WARM_SOLVE_MAX_SECONDS:
            failure_reasons.append("PROJECTED_TIME_EXCEEDED")
        gate.update(
            {
                "initial_physical_objective": initial_objective,
                "final_physical_objective": final_objective,
                "initial_scaled_feasibility_inf": initial_feasibility,
                "final_scaled_feasibility_inf": final_feasibility,
                "initial_raw_kkt_stationarity_inf": initial_stationarity,
                "final_raw_kkt_stationarity_inf": final_stationarity,
                "projected_100_iteration_s": projected_seconds,
                "projection_formula": "10 * synchronized_solve_seconds",
                "synchronized_solve_seconds": float(synchronized_seconds),
            }
        )
        gate["gate_status"] = "PASS" if not failure_reasons else "FAIL"
    return gate


def _sqp_gate_receipt_payload(
    request: RunRequest,
    raw: Mapping[str, JsonValue],
    raw_ref: ArtifactRef,
    memory: Mapping[str, JsonValue],
    memory_ref: ArtifactRef,
) -> dict[str, JsonValue]:
    """Bind one persisted gate verdict to its immutable raw and memory bytes."""

    if request.phase is RunPhase.FIRST_EVAL:
        derivative_gate = raw.get("derivative_kkt_gate")
        if not isinstance(derivative_gate, dict):
            raise TypeError("SQP derivative gate evidence is malformed")
        schema_version = CFS_SQP1_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION
        gate_status = derivative_gate.get("gate_status")
        failure_reasons = derivative_gate.get("failure_reasons")
        gate_detail_key = "derivative_kkt_gate"
        gate_detail = derivative_gate
    elif request.phase is RunPhase.CANARY:
        gate_detail = _sqp_canary_gate(request, raw, memory)
        schema_version = str(gate_detail["schema_version"])
        gate_status = gate_detail["gate_status"]
        failure_reasons = gate_detail["failure_reasons"]
        gate_detail_key = "canary_gate"
    else:
        raise ValueError("complete samples are not gate receipts")
    if gate_status not in ("PASS", "FAIL") or not isinstance(failure_reasons, list):
        raise ValueError("SQP gate status or failure reasons are malformed")
    return {
        "schema_version": schema_version,
        "contract_sha256": contract_sha256_v2(),
        "plan_sha256": SQP_PLAN_SHA256,
        "budget_sha256": SQP_BUDGET_SHA256,
        "request": raw["request"],
        "source_identity": raw["source_identity"],
        "runtime_evidence": raw["runtime_evidence"],
        "bootstrap_artifact": raw["bootstrap_artifact"],
        "raw_result": asdict(raw_ref),
        "gpu_memory": asdict(memory_ref),
        "gate_status": gate_status,
        "failure_reasons": failure_reasons,
        gate_detail_key: gate_detail,
    }


def _load_passed_sqp_gate_receipt(
    campaign_root: Path,
    relative_directory: Path,
    *,
    expected_schema: str,
) -> dict[str, JsonValue]:
    """Validate one persisted gate receipt and every artifact it binds."""

    path = campaign_root / relative_directory / "gate-receipt.json"
    if path.is_symlink() or not path.is_file():
        raise PhaseGateError(f"missing SQP prerequisite gate receipt: {path}")
    if path.stat().st_mode & 0o222:
        raise PhaseGateError("SQP prerequisite gate receipt is writable")
    encoded = path.read_bytes()
    payload = load_canonical_json_bytes(encoded)
    if not isinstance(payload, dict):
        raise TypeError("SQP prerequisite gate receipt must be an object")
    receipt = dict(payload)
    if receipt.get("schema_version") != expected_schema:
        raise PhaseGateError("SQP prerequisite gate receipt schema mismatch")
    if (
        receipt.get("contract_sha256") != contract_sha256_v2()
        or receipt.get("plan_sha256") != SQP_PLAN_SHA256
        or receipt.get("budget_sha256") != SQP_BUDGET_SHA256
    ):
        raise PhaseGateError("SQP prerequisite gate contract identity mismatch")
    gate = {
        CFS_SQP1_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION: SqpGate.DERIVATIVE,
        CFS_SQP1_CANARY_1_GATE_SCHEMA_VERSION: SqpGate.CANARY_1,
        CFS_SQP1_CANARY_10_GATE_SCHEMA_VERSION: SqpGate.CANARY_10,
    }[expected_schema]
    artifact = ArtifactRef(
        relative_path=path.relative_to(campaign_root).as_posix(),
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        schema_version=expected_schema,
    )
    result = load_sqp_gate_result(campaign_root, gate, artifact)
    if not result.passed or result.failure_reasons:
        raise PhaseGateError("SQP prerequisite gate did not semantically pass")
    return receipt


def _enforce_sqp_prerequisite_chain(
    request: RunRequest,
    campaign_root: Path,
) -> None:
    """Require each persisted predecessor before authorizing the next process."""

    required_gates: tuple[tuple[Path, str], ...]
    if request.phase is RunPhase.FIRST_EVAL:
        required_gates = ()
    elif request.phase is RunPhase.CANARY and request.steps == 1:
        required_gates = (
            (
                Path("gates/derivative"),
                CFS_SQP1_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION,
            ),
        )
    elif request.phase is RunPhase.CANARY and request.steps == 10:
        required_gates = (
            (
                Path("gates/derivative"),
                CFS_SQP1_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION,
            ),
            (Path("gates/canary-1"), CFS_SQP1_CANARY_1_GATE_SCHEMA_VERSION),
        )
    else:
        required_gates = (
            (
                Path("gates/derivative"),
                CFS_SQP1_DERIVATIVE_GATE_RECEIPT_SCHEMA_VERSION,
            ),
            (Path("gates/canary-1"), CFS_SQP1_CANARY_1_GATE_SCHEMA_VERSION),
            (Path("gates/canary-10"), CFS_SQP1_CANARY_10_GATE_SCHEMA_VERSION),
        )
    for relative_directory, expected_schema in required_gates:
        _load_passed_sqp_gate_receipt(
            campaign_root,
            relative_directory,
            expected_schema=expected_schema,
        )
    if request.phase is not RunPhase.COMPLETE:
        return
    assert request.sample is not None
    prior_samples = {
        CompleteSample.COLD: (),
        CompleteSample.WARM_1: (CompleteSample.COLD,),
        CompleteSample.WARM_2: (CompleteSample.COLD, CompleteSample.WARM_1),
        CompleteSample.WARM_3: (
            CompleteSample.COLD,
            CompleteSample.WARM_1,
            CompleteSample.WARM_2,
        ),
    }[request.sample]
    for prior_sample in prior_samples:
        receipt_path = (
            campaign_root / "samples" / prior_sample.value / "sample-receipt.json"
        )
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise PhaseGateError(
                "SQP warm samples require the exact certified prior sample prefix"
            )
        if receipt_path.stat().st_mode & 0o222:
            raise PhaseGateError("SQP prior sample receipt is writable")
        encoded = receipt_path.read_bytes()
        artifact = ArtifactRef(
            relative_path=receipt_path.relative_to(campaign_root).as_posix(),
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            schema_version=SQP_SAMPLE_SCHEMA_VERSION,
        )
        prior_receipt = load_sqp_sample_receipt(campaign_root, artifact)
        if (
            prior_receipt.request.sample is not prior_sample
            or not prior_receipt.promotion_eligible
            or prior_receipt.endpoint_certificate is None
        ):
            raise PhaseGateError(
                "SQP warm samples require the exact certified prior sample prefix"
            )
        prior_receipt.endpoint_certificate.resolve_and_validate(campaign_root)


def _cfs_sqp1_child_probe_payload(
    request: RunRequest,
    bootstrap: SingleStageFullSpaceBootstrap,
) -> dict[str, JsonValue]:
    """Dispatch SQP gate/solve semantics without route fallthrough."""

    if request.phase is RunPhase.FIRST_EVAL:
        gate, timing, transfers = run_cfs_sqp1_derivative_gate(bootstrap)
        timing["total_child_wall_seconds"] = None
        return {
            "schema_version": CFS_SQP1_DERIVATIVE_GATE_SCHEMA_VERSION,
            "derivative_kkt_gate": gate,
            "timing": timing,
            "transfer_audit": transfers,
            "terminal_status": "DERIVATIVE_KKT_GATE_COMPLETED",
        }
    optimizer, endpoint, execution = run_cfs_sqp1_probe(
        bootstrap,
        maximum_iterations=_sqp_maximum_iterations(request),
        warm=request.phase is RunPhase.COMPLETE
        and request.sample is not CompleteSample.COLD,
    )
    timing = execution["timing"]
    if not isinstance(timing, dict):
        raise TypeError("SQP timing evidence is malformed")
    timing["total_child_wall_seconds"] = None
    transfers = execution["transfers"]
    if not isinstance(transfers, dict):
        raise TypeError("SQP transfer evidence is malformed")
    return {
        "schema_version": SQP_RESULT_SCHEMA_VERSION,
        "optimizer_result": optimizer,
        "endpoint": endpoint,
        "timing": timing,
        "transfer_audit": transfers,
        "endpoint_certificate": None,
        "promotion_eligible": False,
        "trajectory_equivalence_required": False,
        "terminal_status": str(optimizer["status"]),
    }


def execute_cfs_sqp1_snapshot_child(
    request: RunRequest,
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
) -> bytes:
    """Produce an SQP raw-result draft; the parent seals exact wall timing."""

    request.validate_v2()
    if request.route is not FullSpaceRoute.CFS_SQP1:
        raise PhaseGateError("SQP child requires the CFS-SQP1 route")
    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(campaign / SNAPSHOT_DIRECTORY)
    if (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
        or environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256
    ):
        raise ValueError("snapshot child is not bound to this campaign and manifest")
    if Path.cwd().resolve(strict=True) != publication.root:
        raise ValueError("snapshot child must execute from the immutable snapshot root")
    _enforce_sqp_prerequisite_chain(request, campaign)
    if request.phase is RunPhase.COMPLETE:
        raise PhaseGateError(
            "CFS-SQP1 complete execution requires a production endpoint-audit "
            "authority producer; refusing before solver execution"
        )
    run_directory = _sqp_run_relative_directory(request)
    run_root = campaign / run_directory
    run_root.mkdir(parents=True)
    runtime, runtime_ref = publish_child_runtime_provenance(
        publication,
        campaign_root=campaign,
        process_argv=process_argv,
        environment=environment,
        evidence_relative_directory=run_directory / "evidence",
    )
    runtime_ref.resolve_and_validate(campaign).chmod(0o444)
    bootstrap = build_single_stage_fullspace_bootstrap()
    bootstrap_ref = publish_bootstrap_artifact(
        run_root / "bootstrap.json",
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime_ref,
        bootstrap_factory=lambda: bootstrap,
    )
    common: dict[str, JsonValue] = {
        "contract_sha256": contract_sha256_v2(),
        "plan_sha256": SQP_PLAN_SHA256,
        "budget_sha256": SQP_BUDGET_SHA256,
        "request": asdict(request),
        "source_identity": asdict(runtime.source_identity),
        "runtime_evidence": asdict(runtime_ref),
        "bootstrap_artifact": asdict(bootstrap_ref),
    }
    payload = {**common, **_cfs_sqp1_child_probe_payload(request, bootstrap)}
    return canonical_json_bytes(payload)


def _physical_gpu_memory_identity(
    environment: Mapping[str, str],
) -> tuple[str, int]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ),
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi memory query failed: {completed.stderr.strip()}"
        )
    rows: list[tuple[str, str, int]] = []
    for line in completed.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3 or not fields[2].isdecimal():
            raise ValueError("nvidia-smi emitted malformed memory identity")
        rows.append((fields[0], fields[1], int(fields[2]) * 1024 * 1024))
    visible = environment.get("CUDA_VISIBLE_DEVICES", "").strip()
    selected = rows
    if visible.startswith("GPU-"):
        selected = [row for row in rows if row[1] == visible]
    elif visible.isdecimal():
        selected = [row for row in rows if row[0] == visible]
    elif visible:
        raise ValueError("SQP runner requires exactly one visible GPU index or UUID")
    if len(selected) != 1 or not selected[0][1].startswith("GPU-"):
        raise ValueError("SQP runner did not resolve exactly one physical GPU")
    return selected[0][1], selected[0][2]


def run_cfs_sqp1_campaign(
    request: RunRequest,
    campaign_root: Path,
    *,
    native_extension_path: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    timeout_seconds: float = _SQP_CHILD_TIMEOUT_SECONDS,
) -> bytes:
    """Run one timeout-bounded SQP child and seal parent-observed evidence."""

    request.validate_v2()
    if request.route is not FullSpaceRoute.CFS_SQP1:
        raise PhaseGateError("SQP campaign requires the CFS-SQP1 route")
    _enforce_sqp_prerequisite_chain(request, campaign_root)
    if request.phase is RunPhase.COMPLETE:
        raise PhaseGateError(
            "CFS-SQP1 complete execution requires a production endpoint-audit "
            "authority producer; refusing before solver execution"
        )
    publication = prepare_or_load_execution_snapshot(
        campaign_root, native_extension_path=native_extension_path
    )
    run_root = campaign_root / _sqp_run_relative_directory(request)
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"refusing existing SQP run path: {run_root}")
    raw_path = run_root / "raw-result.json"
    request_argv: tuple[str, ...] = (
        "--phase",
        request.phase.value,
        "--route",
        request.route.value,
        "--device",
        request.device.value,
        *(() if request.steps is None else ("--steps", str(request.steps))),
        *(() if request.sample is None else ("--sample", request.sample.value)),
        "--output",
        str(raw_path),
        "--snapshot-child",
    )
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=interpreter,
        request_argv=request_argv,
        environment=environment,
    )
    gpu_uuid, physical_memory_bytes = _physical_gpu_memory_identity(environment)
    started_ns = time.perf_counter_ns()
    child = subprocess.Popen(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    monitor = BoundProcessGpuMemoryMonitor(
        gpu_uuid=gpu_uuid,
        provider_pid=child.pid,
        expected_argv=invocation.argv,
    )
    monitor.start()
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
        finished_ns = time.perf_counter_ns()
    except subprocess.TimeoutExpired as error:
        child.kill()
        child.communicate()
        raise TimeoutError("CFS-SQP1 child exceeded its frozen timeout") from error
    finally:
        memory_measurement = monitor.finish()
    child_seconds = (finished_ns - started_ns) / 1.0e9
    if child.returncode != 0:
        raise RuntimeError(
            "snapshot CFS-SQP1 child failed with exit code "
            f"{child.returncode}: {stderr.decode('utf-8', 'replace')}"
        )
    raw_value = load_canonical_json_bytes(stdout)
    if not isinstance(raw_value, dict):
        raise TypeError("SQP child result must be a canonical JSON object")
    raw = dict(raw_value)
    timing = raw.get("timing")
    if (
        not isinstance(timing, dict)
        or timing.get("total_child_wall_seconds") is not None
    ):
        raise ValueError("SQP child timing draft is malformed")
    timing["total_child_wall_seconds"] = child_seconds
    raw_schema = str(raw["schema_version"])
    if raw_schema not in (
        SQP_RESULT_SCHEMA_VERSION,
        CFS_SQP1_DERIVATIVE_GATE_SCHEMA_VERSION,
    ):
        raise ValueError("SQP child returned an unsupported result schema")
    raw_ref = _publish_immutable_json(
        raw_path,
        raw,
        campaign_root=campaign_root,
        schema_version=raw_schema,
    )
    memory = bound_gpu_memory_payload(
        monitor,
        memory_measurement,
        parent_pid=os.getpid(),
        physical_device_memory_bytes=physical_memory_bytes,
        runtime_argv=invocation.argv[2:],
    )
    memory_ref = _publish_immutable_json(
        run_root / "gpu-memory.json",
        memory,
        campaign_root=campaign_root,
        schema_version=str(memory["schema_version"]),
    )
    if request.phase is not RunPhase.COMPLETE:
        gate_receipt = _sqp_gate_receipt_payload(
            request,
            raw,
            raw_ref,
            memory,
            memory_ref,
        )
        gate_ref = _publish_immutable_json(
            run_root / "gate-receipt.json",
            gate_receipt,
            campaign_root=campaign_root,
            schema_version=str(gate_receipt["schema_version"]),
        )
        return gate_ref.resolve_and_validate(campaign_root).read_bytes()
    raise PhaseGateError(
        "CFS-SQP1 complete execution bypassed the required endpoint-audit "
        "authority gate"
    )


def executed_run_receipt_payload(
    request: RunRequest,
    *,
    runtime_evidence: RuntimeEvidence,
    runtime_evidence_ref: ArtifactRef,
    terminal_status: str,
    timing: Mapping[str, JsonValue],
    transfer_audit: Mapping[str, JsonValue],
    endpoint_certificate: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build an executed receipt from one internally consistent evidence object."""

    request.validate()
    if runtime_evidence_ref.schema_version != RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("runtime evidence reference has the wrong schema")
    evidence_payload = canonical_json_bytes(runtime_evidence.to_payload())
    if runtime_evidence_ref.size_bytes != len(evidence_payload) or (
        runtime_evidence_ref.sha256 != hashlib.sha256(evidence_payload).hexdigest()
    ):
        raise ValueError("runtime evidence reference differs from evidence bytes")
    if terminal_status in ("", "PARTIAL", "RUNNING"):
        raise ValueError("executed receipt requires a terminal status")
    if endpoint_certificate.get("certified") is not True:
        raise ValueError("executed receipt requires endpoint certification")
    return {
        "contract_sha256": contract_sha256(),
        "endpoint_certificate": dict(endpoint_certificate),
        "request": asdict(request),
        "runtime_evidence": asdict(runtime_evidence_ref),
        "runtime_identity": asdict(runtime_evidence.observation.runtime_identity),
        "schema_version": SCHEMA_VERSION,
        "source_identity": asdict(runtime_evidence.source_identity),
        "terminal_status": terminal_status,
        "timing": dict(timing),
        "trajectory_equivalence_required": False,
        "transfer_audit": dict(transfer_audit),
    }


def _reject_duplicate_options(argv: Sequence[str]) -> None:
    seen: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.partition("=")[0]
        if option in seen:
            raise ValueError(f"duplicate option: {option}")
        seen.add(option)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", required=True, choices=tuple(RunPhase))
    parser.add_argument("--route", required=True, choices=tuple(FullSpaceRoute))
    parser.add_argument("--device", required=True, choices=tuple(DeviceLane))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sample", choices=tuple(CompleteSample))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--snapshot-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_request(argv: Sequence[str]) -> tuple[RunRequest, Path, bool]:
    _reject_duplicate_options(argv)
    args = build_parser().parse_args(argv)
    request = RunRequest(
        phase=RunPhase(args.phase),
        route=FullSpaceRoute(args.route),
        device=DeviceLane(args.device),
        steps=args.steps,
        sample=CompleteSample(args.sample) if args.sample is not None else None,
    )
    if request.route is FullSpaceRoute.CFS_SQP1:
        request.validate_v2()
    else:
        request.validate()
    output = args.output
    if (
        not args.preflight_only
        and request.route is not FullSpaceRoute.CFS_SQP1
        and (output.exists() or output.is_symlink())
    ):
        raise FileExistsError(f"refusing existing output path: {output}")
    return request, output, args.preflight_only


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    request, output, preflight_only = parse_request(raw_argv)
    if preflight_only:
        payload = canonical_json_bytes(
            run_request_payload_v2(request)
            if request.route is FullSpaceRoute.CFS_SQP1
            else run_request_payload(request)
        )
    elif "--snapshot-child" in raw_argv:
        campaign_value = os.environ.get(_CAMPAIGN_ROOT_ENV)
        if campaign_value is None:
            raise ValueError("snapshot child campaign binding is absent")
        if request.route is FullSpaceRoute.CFS_SQP1:
            payload = execute_cfs_sqp1_snapshot_child(
                request,
                campaign_root=Path(campaign_value),
                process_argv=(str(Path(__file__).resolve()), *raw_argv),
                environment=os.environ,
            )
        elif (
            request.phase is RunPhase.FIRST_EVAL
            and request.route is FullSpaceRoute.CFS_P0
        ):
            payload = execute_first_eval_snapshot_child(
                request,
                campaign_root=Path(campaign_value),
                process_argv=(str(Path(__file__).resolve()), *raw_argv),
                environment=os.environ,
            )
        elif (
            request.phase is RunPhase.CANARY and request.route is FullSpaceRoute.CFS_P0
        ):
            payload = execute_cfs_p0_canary_snapshot_child(
                request,
                campaign_root=Path(campaign_value),
                process_argv=(str(Path(__file__).resolve()), *raw_argv),
                environment=os.environ,
            )
        elif request.phase is RunPhase.CANARY:
            raise PhaseGateError(
                "only the 10/100-step CFS-P0 canary child is implemented"
            )
        elif (
            request.phase is RunPhase.COMPLETE
            and request.route is FullSpaceRoute.CFS_AL1
        ):
            payload = execute_cfs_al1_snapshot_child(
                request,
                campaign_root=Path(campaign_value),
                process_argv=(str(Path(__file__).resolve()), *raw_argv),
                environment=os.environ,
            )
        elif (
            request.phase is RunPhase.COMPLETE
            and request.route is FullSpaceRoute.CFS_AL2
        ):
            payload = execute_cfs_al2_snapshot_child(
                request,
                campaign_root=Path(campaign_value),
                process_argv=(str(Path(__file__).resolve()), *raw_argv),
                environment=os.environ,
            )
        else:
            raise PhaseGateError("requested snapshot-child route/phase is unsupported")
    else:
        native_path_value = simsoptpp.__file__
        if native_path_value is None:
            raise ValueError("native extension path is unavailable")
        if request.route is FullSpaceRoute.CFS_SQP1:
            payload = run_cfs_sqp1_campaign(
                request,
                output,
                native_extension_path=Path(native_path_value),
                interpreter=Path(sys.executable),
                environment=os.environ,
            )
        elif (
            request.phase is RunPhase.FIRST_EVAL
            and request.route is FullSpaceRoute.CFS_P0
        ):
            payload = run_first_eval_campaign(
                request,
                output,
                native_extension_path=Path(native_path_value),
                interpreter=Path(sys.executable),
                environment=os.environ,
            )
        elif (
            request.phase is RunPhase.CANARY and request.route is FullSpaceRoute.CFS_P0
        ):
            payload = run_cfs_p0_canary_campaign(
                request,
                output,
                native_extension_path=Path(native_path_value),
                interpreter=Path(sys.executable),
                environment=os.environ,
            )
        elif request.phase is RunPhase.CANARY:
            raise PhaseGateError("only the 10/100-step CFS-P0 canary is implemented")
        elif (
            request.phase is RunPhase.COMPLETE
            and request.route is FullSpaceRoute.CFS_AL1
        ):
            payload = run_cfs_al1_campaign(
                request,
                output,
                native_extension_path=Path(native_path_value),
                interpreter=Path(sys.executable),
                environment=os.environ,
            )
        elif (
            request.phase is RunPhase.COMPLETE
            and request.route is FullSpaceRoute.CFS_AL2
        ):
            payload = run_cfs_al2_campaign(
                request,
                output,
                native_extension_path=Path(native_path_value),
                interpreter=Path(sys.executable),
                environment=os.environ,
            )
        else:
            raise PhaseGateError("requested campaign route/phase is unsupported")
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
