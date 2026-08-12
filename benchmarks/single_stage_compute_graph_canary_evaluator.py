"""Isolated C1/C2 production-path canary evaluator.

This artifact is intentionally separate from the authoritative C0 receipt.  A
candidate is claimable only when the production value-and-gradient graph also
exports fixed-shape variant telemetry; otherwise the evaluator reports a
machine-readable BLOCKED result after executing the real graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import numpy as np
from simsopt_jax.runtime.host_boundary import host_scalar

from benchmarks.single_stage_compute_graph_c0_evaluator import (
    EXPECTED_PARAMETER_COUNT,
    EvaluationResult,
    _canonical_candidate,
    _residual_certificates,
    _validate_result,
)

CANARY_CHILD_SCHEMA_ID: Final = "single-stage-compute-graph-canary-child-v1"
CanaryVariant = Literal["C1", "C2"]
CanaryMode = Literal["initial_gate", "gate", "warm"]

_COMMON_TELEMETRY = (
    "exact_newton_variant_dense_linearization_used",
    "exact_newton_variant_linear_solve_attempt_count",
    "exact_newton_variant_dense_materialization_count",
    "exact_newton_variant_lu_factorization_count",
    "exact_newton_variant_lu_solve_count",
    "exact_newton_variant_refinement_correction_count",
    "exact_newton_variant_stop_reason_code",
    "exact_newton_variant_numerical_failure",
)
_C1_TELEMETRY = (
    "exact_newton_variant_backtracking_iteration_count",
    "exact_newton_variant_stalled",
    "exact_newton_variant_retry_linear_solve_at_strict_cap",
)
_C2_TELEMETRY = (
    "exact_newton_variant_applied_update_count",
    "exact_newton_variant_rollback_branch_taken",
    "exact_newton_variant_rollback_recompute_count",
    "exact_newton_variant_native_persist_predicate",
    "exact_newton_variant_persist_solved_state",
    "exact_newton_variant_initial_norm",
    "exact_newton_variant_assessed_norm",
    "exact_newton_variant_returned_norm",
)


class CanaryEvaluatorError(RuntimeError):
    """The canary child cannot emit a valid observation."""


@dataclass(frozen=True, slots=True)
class CanaryEvaluation:
    """Production value/gradient result plus graph-local variant telemetry."""

    numerical: EvaluationResult
    telemetry: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class GpuMemoryEvidence:
    provider_pid: int
    gpu_uuid: str
    sample_count: int
    sample_interval_ns: int
    peak_bytes: int


class PreparedCanary(Protocol):
    def evaluate_once(self) -> CanaryEvaluation: ...


def _variant(value: object) -> CanaryVariant:
    if value == "C1":
        return "C1"
    if value == "C2":
        return "C2"
    raise CanaryEvaluatorError("canary variant must be C1 or C2")


def _validate_telemetry(
    variant: CanaryVariant, telemetry: Mapping[str, object]
) -> dict[str, int | float | bool]:
    required = _COMMON_TELEMETRY + (_C2_TELEMETRY if variant == "C2" else _C1_TELEMETRY)
    missing = tuple(key for key in required if key not in telemetry)
    if missing:
        raise CanaryEvaluatorError(
            "production value-and-gradient path did not export required "
            f"{variant} telemetry: {missing}"
        )
    normalized: dict[str, int | float | bool] = {}
    for key in required:
        value = host_scalar(telemetry[key])
        if key.endswith(
            (
                "dense_linearization_used",
                "numerical_failure",
                "branch_taken",
                "stalled",
                "retry_linear_solve_at_strict_cap",
                "native_persist_predicate",
                "persist_solved_state",
            )
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise CanaryEvaluatorError(f"{key} must be boolean")
            normalized[key] = bool(value)
        elif key.endswith(("initial_norm", "assessed_norm", "returned_norm")):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise CanaryEvaluatorError(f"{key} must be a finite number")
            normalized[key] = float(value)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise CanaryEvaluatorError(f"{key} must be an integer")
            if int(value) < 0:
                raise CanaryEvaluatorError(f"{key} must be nonnegative")
            normalized[key] = int(value)
    if normalized["exact_newton_variant_dense_materialization_count"] < 1:
        raise CanaryEvaluatorError("canary did not execute a dense materialization")
    if normalized["exact_newton_variant_numerical_failure"]:
        raise CanaryEvaluatorError("canary reported a numerical failure")
    dense_count = normalized["exact_newton_variant_dense_materialization_count"]
    lu_count = normalized["exact_newton_variant_lu_factorization_count"]
    solve_count = normalized["exact_newton_variant_lu_solve_count"]
    attempt_count = normalized["exact_newton_variant_linear_solve_attempt_count"]
    if (
        normalized["exact_newton_variant_dense_linearization_used"] is not True
        or not isinstance(dense_count, int)
        or not isinstance(lu_count, int)
        or not isinstance(solve_count, int)
        or not isinstance(attempt_count, int)
        or lu_count > dense_count
        or solve_count < lu_count
        or attempt_count < lu_count
        or normalized["exact_newton_variant_stop_reason_code"] != 0
    ):
        raise CanaryEvaluatorError("variant telemetry counters/status are inconsistent")
    if variant == "C1":
        if (
            normalized["exact_newton_variant_stalled"]
            or normalized["exact_newton_variant_retry_linear_solve_at_strict_cap"]
        ):
            raise CanaryEvaluatorError(
                "successful C1 canary reports stalled/retry state"
            )
    else:
        rollback = normalized["exact_newton_variant_rollback_branch_taken"]
        rollback_count = normalized["exact_newton_variant_rollback_recompute_count"]
        updates = normalized["exact_newton_variant_applied_update_count"]
        norms = tuple(
            normalized[key]
            for key in (
                "exact_newton_variant_initial_norm",
                "exact_newton_variant_assessed_norm",
                "exact_newton_variant_returned_norm",
            )
        )
        if (
            rollback
            or rollback_count != 0
            or normalized["exact_newton_variant_native_persist_predicate"] is not True
            or normalized["exact_newton_variant_persist_solved_state"] is not True
            or not isinstance(updates, int)
            or updates > attempt_count
            or any(not math.isfinite(float(value)) for value in norms)
        ):
            raise CanaryEvaluatorError(
                "successful C2 persistence telemetry is inconsistent"
            )
    return normalized


def evaluate_canary_once(
    *,
    variant: CanaryVariant,
    mode: CanaryMode,
    sample_index: int | None,
    parameter_sha256: str,
    prepared: PreparedCanary,
    clock=time.monotonic_ns,
    peak_rss=lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    gpu_memory: Callable[[], GpuMemoryEvidence] = lambda: GpuMemoryEvidence(
        provider_pid=1,
        gpu_uuid="GPU-test",
        sample_count=1,
        sample_interval_ns=10_000_000,
        peak_bytes=0,
    ),
) -> dict[str, object]:
    """Time one real production-path evaluation and validate its evidence."""

    if (mode == "warm") != (sample_index is not None):
        raise CanaryEvaluatorError("only warm observations carry a sample index")
    started_ns = clock()
    result = prepared.evaluate_once()
    finished_ns = clock()
    wall_ns = finished_ns - started_ns
    gpu_memory_evidence = gpu_memory()
    if wall_ns <= 0:
        raise CanaryEvaluatorError("measured wall time must be positive")
    if (
        gpu_memory_evidence.provider_pid <= 0
        or not gpu_memory_evidence.gpu_uuid
        or gpu_memory_evidence.sample_count <= 0
        or gpu_memory_evidence.sample_interval_ns <= 0
        or gpu_memory_evidence.peak_bytes < 0
    ):
        raise CanaryEvaluatorError("GPU-memory evidence is invalid")
    _validate_result(result.numerical)
    telemetry = _validate_telemetry(variant, result.telemetry)
    observation: dict[str, object] = {
        "schema_id": CANARY_CHILD_SCHEMA_ID,
        "status": "PASS",
        "variant": variant,
        "mode": mode,
        "sample_index": sample_index,
        "wall_ns": wall_ns,
        "peak_self_rss_bytes": peak_rss(),
        "gpu_memory": {
            "provider_pid": gpu_memory_evidence.provider_pid,
            "gpu_uuid": gpu_memory_evidence.gpu_uuid,
            "sample_count": gpu_memory_evidence.sample_count,
            "sample_interval_ns": gpu_memory_evidence.sample_interval_ns,
            "peak_bytes": gpu_memory_evidence.peak_bytes,
            "source": "nvidia-smi_direct_pid_gpu_uuid",
        },
        "telemetry": telemetry,
    }
    if mode in {"initial_gate", "gate"}:
        observation.update(
            {
                "parameter_sha256": parameter_sha256,
                "objective_dtype": "float64",
                "objective": result.numerical.objective,
                "gradient_dtype": "float64",
                "gradient": result.numerical.gradient.tolist(),
                "inner_newton_success": result.numerical.inner_newton_success,
                "adjoint_success": result.numerical.adjoint_success,
                "residual_certificates": dict(result.numerical.residual_certificates),
            }
        )
    return observation


class _ObservedEvaluation(Protocol):
    forward_success: object
    actual_adjoint_success: object
    forward_result: Mapping[str, object]
    adjoint_residual: object
    adjoint_residual_relative: object


class _ProductionPreparedCanary:
    def __init__(self, runtime: object, candidate: np.ndarray) -> None:
        self._runtime = runtime
        self._candidate = candidate

    def evaluate_once(self) -> CanaryEvaluation:
        from examples.jax.parity.cases.native_boozerqa import (
            _host_array,
            _host_bool,
            _host_float,
        )
        from simsopt_jax_adapters.geo.surface_objectives_traceable import (
            _accepted_incumbent_host_observation_sink,
        )

        class _Runtime(Protocol):
            def fresh_incumbent_controller(self) -> object: ...

        class _Controller(Protocol):
            def value_and_grad(
                self, parameters: np.ndarray
            ) -> tuple[float, np.ndarray]: ...

        observations: list[object] = []
        controller = cast(
            _Controller,
            cast(_Runtime, self._runtime).fresh_incumbent_controller(),
        )
        with _accepted_incumbent_host_observation_sink(observations.append):
            objective, gradient = controller.value_and_grad(self._candidate)
        if len(observations) != 1:
            raise CanaryEvaluatorError(
                "production value-and-gradient emitted an invalid observation count"
            )
        observed = cast(_ObservedEvaluation, observations[0])
        forward = observed.forward_result
        exact_residual = forward.get("residual")
        numerical = EvaluationResult(
            objective=float(objective),
            gradient=np.asarray(gradient, dtype=np.float64),
            inner_newton_success=_host_bool(observed.forward_success),
            adjoint_success=_host_bool(observed.actual_adjoint_success),
            residual_certificates=_residual_certificates(
                exact_residual=(
                    None if exact_residual is None else _host_array(exact_residual)
                ),
                inner_penalty_residual_l2=_host_float(
                    forward["inner_penalty_residual_l2"]
                ),
                final_gradient_inf_norm=_host_float(forward["final_gradient_inf_norm"]),
                adjoint_residual=_host_array(observed.adjoint_residual),
                adjoint_residual_relative=_host_float(
                    observed.adjoint_residual_relative
                ),
            ),
        )
        telemetry: dict[str, object] = {}
        for key in (*_COMMON_TELEMETRY, *_C1_TELEMETRY, *_C2_TELEMETRY):
            if key in forward:
                telemetry[key] = forward[key]
        return CanaryEvaluation(numerical=numerical, telemetry=telemetry)


def _prepare_production_canary(
    input_root: Path,
    variant: CanaryVariant,
    candidate: np.ndarray,
    mode: CanaryMode,
) -> _ProductionPreparedCanary:
    from examples.jax.parity.cases.native_boozerqa import _prepare_jax_variant_runtime
    from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
    from examples.jax.parity.input_bundle import read_input_bundle

    bundle, arrays = read_input_bundle(input_root)
    runtime = _prepare_jax_variant_runtime(
        bundle,
        arrays,
        SPEC,
        None,
        exact_newton_variant=variant,
    )
    if runtime.initial_parameters.shape != (EXPECTED_PARAMETER_COUNT,):
        raise CanaryEvaluatorError("canonical runtime does not expose 461 parameters")
    if np.array_equal(candidate, runtime.initial_parameters):
        raise CanaryEvaluatorError("candidate must be a changed state")
    parameters = runtime.initial_parameters if mode == "initial_gate" else candidate
    return _ProductionPreparedCanary(runtime, parameters)


def blocked_observation(
    variant: CanaryVariant, mode: CanaryMode, reason: str
) -> dict[str, object]:
    return {
        "schema_id": CANARY_CHILD_SCHEMA_ID,
        "status": "BLOCKED",
        "variant": variant,
        "mode": mode,
        "blocker": {
            "code": "PRODUCTION_VARIANT_TELEMETRY_UNAVAILABLE",
            "reason": reason,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("C1", "C2"), required=True)
    parser.add_argument(
        "--mode",
        choices=("initial_gate", "gate", "warm"),
        required=True,
    )
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variant = _variant(args.variant)
    mode: CanaryMode = args.mode
    try:
        from benchmarks.process_gpu_monitor import (
            ProcessGpuMemoryMonitor,
            ProcessGpuMemoryResult,
        )
        from benchmarks.single_stage_compute_graph_c0_evaluator import (
            _validate_runtime_contract,
            _verify_snapshot_import_origins,
        )

        _verify_snapshot_import_origins(args.snapshot_root)
        _validate_runtime_contract()
        candidate = _canonical_candidate(args.candidate, args.parameter_sha256)
        prepared = _prepare_production_canary(args.input_root, variant, candidate, mode)
        from benchmarks.single_stage_compute_graph_native_reference import (
            _parameter_sha256,
        )

        evaluated_parameter_sha256 = (
            _parameter_sha256(prepared._candidate)
            if mode == "initial_gate"
            else args.parameter_sha256
        )
        monitor = ProcessGpuMemoryMonitor(
            gpu_uuid=args.gpu_uuid,
            provider_pid=os.getpid(),
            interval_seconds=0.01,
        )
        monitor.start()
        monitor_result = None

        def finish_gpu_monitor() -> GpuMemoryEvidence:
            nonlocal monitor_result
            if monitor_result is None:
                monitor_result = monitor.finish()
            if not isinstance(monitor_result, ProcessGpuMemoryResult):
                raise CanaryEvaluatorError(
                    "GPU memory monitor did not observe the evaluator process"
                )
            return GpuMemoryEvidence(
                provider_pid=monitor_result.provider_pid,
                gpu_uuid=monitor_result.gpu_uuid,
                sample_count=len(monitor_result.samples),
                sample_interval_ns=10_000_000,
                peak_bytes=monitor_result.peak_used_memory_mib * 1024 * 1024,
            )

        document = evaluate_canary_once(
            variant=variant,
            mode=mode,
            sample_index=args.sample_index,
            parameter_sha256=evaluated_parameter_sha256,
            prepared=prepared,
            gpu_memory=finish_gpu_monitor,
        )
    except CanaryEvaluatorError as error:
        document = blocked_observation(variant, mode, str(error))
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0 if document["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
