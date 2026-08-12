"""Canonical isolated C0 timing-and-profile child.

One process times the frozen candidate first, then profiles a numerically equal
replay using a fresh incumbent controller from the same prepared runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import numpy as np

from benchmarks.single_stage_compute_graph_c0_runner import _runtime_identity
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_route_environment,
    normalize_static_timing_environment,
    observe_effective_numerical_policies,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    HLO_MODULE_SET_IDENTITY_SOURCE,
    SAMPLED_PROCESS_GPU_MEMORY_SOURCE,
)

CHILD_SCHEMA_ID: Final = "single-stage-compute-graph-c0-child-observation-v3"
CAPTURE_SCHEMA_ID: Final = "single-stage-compute-graph-c0-capture-evidence-v2"
EXPECTED_PARAMETER_COUNT: Final = 461
Mode = Literal["initial-gate", "gate", "profile", "warm"]


class C0EvaluatorError(RuntimeError):
    """The evaluator cannot emit a claim-eligible child observation."""


@dataclass(frozen=True, slots=True)
class ChildRequest:
    """Runner-owned invocation identity."""

    mode: Mode
    sample_index: int | None


@dataclass(frozen=True, slots=True)
class CaptureEvidence:
    """Externally captured facts unavailable from the numerical API."""

    mode: Mode
    sample_index: int | None
    parameter_sha256: str
    sampled_process_gpu_memory_peak_bytes: int
    sampled_process_gpu_memory_source: str
    hlo_module_set_identity: str
    hlo_module_set_identity_source: str
    pjrt_execute_count: int | None
    kernel_launch_count: int | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One canonical objective/gradient result and solver certificates."""

    objective: float
    gradient: np.ndarray
    inner_newton_success: bool
    adjoint_success: bool
    residual_certificates: Mapping[str, float]


class PreparedEvaluation(Protocol):
    """One prepared runtime that can mint isolated incumbent evaluations."""

    def evaluate_once(self) -> EvaluationResult: ...

    def fresh_replay(self) -> PreparedEvaluation: ...


PrepareEvaluation = Callable[[np.ndarray], PreparedEvaluation]
CaptureReplay = Callable[
    [PreparedEvaluation, EvaluationResult], tuple[EvaluationResult, CaptureEvidence]
]
Clock = Callable[[], int]
PeakRss = Callable[[], int]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise C0EvaluatorError(f"{context} must be a JSON object")
    return value


def _integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise C0EvaluatorError(f"{context} must be an integer >= {minimum}")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise C0EvaluatorError(f"{context} must be a non-empty string")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise C0EvaluatorError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _parameter_sha256(candidate: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(candidate, dtype=np.dtype("<f8")).tobytes(order="C")
    ).hexdigest()


def _request_from_environment(environment: Mapping[str, str]) -> ChildRequest:
    variant = environment.get("SINGLE_STAGE_COMPUTE_GRAPH_VARIANT")
    if variant != "C0":
        raise C0EvaluatorError("SINGLE_STAGE_COMPUTE_GRAPH_VARIANT must be C0")
    mode_value = environment.get("SINGLE_STAGE_COMPUTE_GRAPH_MODE")
    if mode_value not in ("initial-gate", "gate", "profile", "warm"):
        raise C0EvaluatorError(
            "SINGLE_STAGE_COMPUTE_GRAPH_MODE must be initial-gate, gate, profile, "
            "or warm"
        )
    mode: Mode = mode_value
    sample_value = environment.get("SINGLE_STAGE_COMPUTE_GRAPH_SAMPLE_INDEX")
    if mode != "warm":
        if sample_value is not None:
            raise C0EvaluatorError(
                "initial-gate/gate/profile mode must not define a sample index"
            )
        return ChildRequest(mode=mode, sample_index=None)
    if sample_value is None or not sample_value.isdecimal():
        raise C0EvaluatorError("warm mode requires a nonnegative decimal sample index")
    return ChildRequest(mode=mode, sample_index=int(sample_value))


def _canonical_candidate(path: Path, expected_sha256: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype != np.dtype(np.float64) or loaded.shape != (
        EXPECTED_PARAMETER_COUNT,
    ):
        raise C0EvaluatorError(
            "candidate must have exact shape (461,) and dtype float64"
        )
    candidate = np.ascontiguousarray(loaded, dtype=np.dtype("<f8"))
    if not np.all(np.isfinite(candidate)):
        raise C0EvaluatorError("candidate must contain only finite values")
    actual_sha256 = hashlib.sha256(candidate.tobytes(order="C")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise C0EvaluatorError("candidate SHA-256 does not match the frozen specimen")
    candidate.setflags(write=False)
    return candidate


def _validate_result(result: EvaluationResult) -> None:
    if not isinstance(result.objective, float) or not math.isfinite(result.objective):
        raise C0EvaluatorError("objective must be a finite Python float")
    if result.gradient.dtype != np.dtype(np.float64) or result.gradient.shape != (
        EXPECTED_PARAMETER_COUNT,
    ):
        raise C0EvaluatorError(
            "gradient must have exact shape (461,) and dtype float64"
        )
    if not np.all(np.isfinite(result.gradient)):
        raise C0EvaluatorError("gradient must contain only finite values")
    if not result.inner_newton_success or not result.adjoint_success:
        raise C0EvaluatorError("inner Newton and adjoint must both succeed")
    if not result.residual_certificates:
        raise C0EvaluatorError("residual certificates must be non-empty")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, float)
        or not math.isfinite(value)
        or value < 0.0
        for name, value in result.residual_certificates.items()
    ):
        raise C0EvaluatorError(
            "residual certificates must be finite nonnegative floats"
        )


def _require_equal_results(timed: EvaluationResult, profiled: EvaluationResult) -> None:
    if (
        timed.objective != profiled.objective
        or not np.array_equal(timed.gradient, profiled.gradient)
        or timed.inner_newton_success != profiled.inner_newton_success
        or timed.adjoint_success != profiled.adjoint_success
        or dict(timed.residual_certificates) != dict(profiled.residual_certificates)
    ):
        raise C0EvaluatorError(
            "profiled replay is not numerically equal to the timed evaluation"
        )


def build_child_observation(
    request: ChildRequest,
    candidate: np.ndarray,
    prepare: PrepareEvaluation,
    capture_replay: CaptureReplay | None,
    *,
    clock: Clock = time.monotonic_ns,
    peak_rss: PeakRss,
) -> dict[str, object]:
    """Time first, then profile an equal replay from the prepared runtime."""

    first_started_ns = clock()
    if request.mode == "profile":
        from simsopt_jax.runtime.trace_annotations import (
            EvaluationKind,
            evaluation_context,
            trace_session,
        )

        parameter_sha256 = _parameter_sha256(candidate)
        with trace_session(), evaluation_context(
            parameter_sha256,
            parameter_sha256,
            EvaluationKind.TRIAL,
        ):
            prepared = prepare(candidate)
            evaluation_started_ns = clock()
            result = prepared.evaluate_once()
            evaluation_finished_ns = clock()
    else:
        prepared = prepare(candidate)
        evaluation_started_ns = clock()
        result = prepared.evaluate_once()
        evaluation_finished_ns = clock()
    wall_ns = evaluation_finished_ns - evaluation_started_ns
    if wall_ns <= 0:
        raise C0EvaluatorError("measured wall time must be positive")
    _validate_result(result)
    common: dict[str, object] = {
        "schema_id": CHILD_SCHEMA_ID,
        "mode": request.mode,
    }
    if request.mode in ("initial-gate", "gate", "warm"):
        if capture_replay is not None:
            raise C0EvaluatorError(
                "initial-gate, gate, and warm modes must not configure profiling"
            )
        if request.mode in ("initial-gate", "gate"):
            common.update(
                {
                    "parameter_sha256": _parameter_sha256(candidate),
                    "objective_dtype": "float64",
                    "objective": result.objective,
                    "gradient_dtype": "float64",
                    "gradient": result.gradient.tolist(),
                    "inner_newton_success": result.inner_newton_success,
                    "adjoint_success": result.adjoint_success,
                    "residual_certificates": dict(result.residual_certificates),
                }
            )
        else:
            common.update(
                {
                    "sample_index": request.sample_index,
                    "wall_ns": wall_ns,
                }
            )
        common.update(
            {
                "peak_self_rss_bytes": peak_rss(),
            }
        )
        return common
    if capture_replay is None:
        raise C0EvaluatorError("profile mode requires profiling")
    profiled_result, capture = capture_replay(prepared.fresh_replay(), result)
    _validate_result(profiled_result)
    _require_equal_results(result, profiled_result)
    if capture.mode != request.mode or capture.sample_index != request.sample_index:
        raise C0EvaluatorError("capture evidence does not match the child request")
    if capture.parameter_sha256 != _parameter_sha256(candidate):
        raise C0EvaluatorError("capture evidence is bound to a different candidate")
    if capture.hlo_module_set_identity_source != HLO_MODULE_SET_IDENTITY_SOURCE:
        raise C0EvaluatorError("unsupported HLO module-set identity source")
    if capture.sampled_process_gpu_memory_source != SAMPLED_PROCESS_GPU_MEMORY_SOURCE:
        raise C0EvaluatorError("unsupported sampled process GPU-memory source")
    if capture.pjrt_execute_count is None or capture.kernel_launch_count is None:
        raise C0EvaluatorError(
            "profile capture must report execution and kernel counts"
        )
    rss_bytes = peak_rss()
    if rss_bytes <= 0:
        raise C0EvaluatorError("peak process RSS must be positive")
    if request.mode == "profile":
        common.update(
            {
                "sample_index": request.sample_index,
                "parameter_sha256": _parameter_sha256(candidate),
                "objective_dtype": "float64",
                "objective": result.objective,
                "gradient_dtype": "float64",
                "gradient": result.gradient.tolist(),
                "inner_newton_success": result.inner_newton_success,
                "adjoint_success": result.adjoint_success,
                "residual_certificates": dict(result.residual_certificates),
                "cold_compile": {
                    "wall_ns": evaluation_finished_ns - first_started_ns,
                    "peak_self_rss_bytes": rss_bytes,
                    "sampled_process_gpu_memory_peak_bytes": (
                        capture.sampled_process_gpu_memory_peak_bytes
                    ),
                    "sampled_process_gpu_memory_source": (
                        capture.sampled_process_gpu_memory_source
                    ),
                    "hlo_module_set_identity": capture.hlo_module_set_identity,
                    "hlo_module_set_identity_source": (
                        capture.hlo_module_set_identity_source
                    ),
                },
                "pjrt_execute_count": capture.pjrt_execute_count,
                "kernel_launch_count": capture.kernel_launch_count,
            }
        )
        return common
    common.update(
        {
            "sample_index": request.sample_index,
            "wall_ns": wall_ns,
            "peak_self_rss_bytes": rss_bytes,
            "sampled_process_gpu_memory_peak_bytes": (
                capture.sampled_process_gpu_memory_peak_bytes
            ),
            "sampled_process_gpu_memory_source": (
                capture.sampled_process_gpu_memory_source
            ),
            "hlo_module_set_identity": capture.hlo_module_set_identity,
            "hlo_module_set_identity_source": capture.hlo_module_set_identity_source,
            "pjrt_execute_count": capture.pjrt_execute_count,
            "kernel_launch_count": capture.kernel_launch_count,
        }
    )
    return common


class _ObservedEvaluation(Protocol):
    forward_success: object
    actual_adjoint_success: object
    forward_result: Mapping[str, object]
    adjoint_residual: object
    adjoint_residual_relative: object


def _residual_certificates(
    *,
    exact_residual: np.ndarray | None,
    inner_penalty_residual_l2: float,
    final_gradient_inf_norm: float,
    adjoint_residual: np.ndarray,
    adjoint_residual_relative: float,
) -> dict[str, float]:
    """Return route-applicable finite certificates without sentinel fields."""

    certificates = {
        "adjoint_residual_l2": float(np.linalg.norm(adjoint_residual)),
        "adjoint_residual_relative": adjoint_residual_relative,
    }
    if exact_residual is not None:
        certificates["boozer_exact_residual_l2"] = float(np.linalg.norm(exact_residual))
    for name, value in (
        ("inner_penalty_residual_l2", inner_penalty_residual_l2),
        ("final_gradient_inf_norm", final_gradient_inf_norm),
    ):
        if math.isfinite(value):
            certificates[name] = value
    return certificates


class _JaxPreparedEvaluation:
    def __init__(self, runtime: object, candidate: np.ndarray) -> None:
        self._runtime = runtime
        self._candidate = candidate

    def evaluate_once(self) -> EvaluationResult:
        from examples.jax.parity.cases.native_boozerqa import (
            _host_array,
            _host_bool,
            _host_float,
        )
        from simsopt_jax_adapters.geo.surface_objectives_traceable import (
            _accepted_incumbent_host_observation_sink,
        )

        class _Controller(Protocol):
            def value_and_grad(
                self, parameters: np.ndarray
            ) -> tuple[float, np.ndarray]: ...

        observations: list[object] = []

        class _Runtime(Protocol):
            def fresh_incumbent_controller(self) -> object: ...

        runtime = cast(_Runtime, self._runtime)
        controller = cast(_Controller, runtime.fresh_incumbent_controller())
        with _accepted_incumbent_host_observation_sink(observations.append):
            objective, gradient = controller.value_and_grad(self._candidate)
        if len(observations) != 1:
            raise C0EvaluatorError(
                "canonical evaluator did not emit exactly one observation"
            )
        observed = cast(_ObservedEvaluation, observations[0])
        adjoint_residual = _host_array(observed.adjoint_residual)
        exact_residual = observed.forward_result.get("residual")
        residual_certificates = _residual_certificates(
            exact_residual=(
                None if exact_residual is None else _host_array(exact_residual)
            ),
            inner_penalty_residual_l2=_host_float(
                observed.forward_result["inner_penalty_residual_l2"]
            ),
            final_gradient_inf_norm=_host_float(
                observed.forward_result["final_gradient_inf_norm"]
            ),
            adjoint_residual=adjoint_residual,
            adjoint_residual_relative=_host_float(observed.adjoint_residual_relative),
        )
        return EvaluationResult(
            objective=float(objective),
            gradient=np.asarray(gradient, dtype=np.float64),
            inner_newton_success=_host_bool(observed.forward_success),
            adjoint_success=_host_bool(observed.actual_adjoint_success),
            residual_certificates=residual_certificates,
        )

    def fresh_replay(self) -> _JaxPreparedEvaluation:
        return _JaxPreparedEvaluation(self._runtime, self._candidate)


def _native_prepare(input_root: Path) -> PrepareEvaluation:
    def prepare(candidate: np.ndarray) -> PreparedEvaluation:
        from examples.jax.parity.cases.native_boozerqa import (
            _prepare_jax_variant_runtime,
        )
        from examples.jax.parity.cases.native_single_stage_boozer_vacuum import (
            SPEC,
        )
        from examples.jax.parity.input_bundle import read_input_bundle

        bundle, arrays = read_input_bundle(input_root)
        prepared = _prepare_jax_variant_runtime(bundle, arrays, SPEC, None)
        if prepared.initial_parameters.shape != (EXPECTED_PARAMETER_COUNT,):
            raise C0EvaluatorError("canonical runtime does not expose 461 parameters")
        if np.array_equal(candidate, prepared.initial_parameters):
            raise C0EvaluatorError(
                "candidate must be a changed state, not the baseline"
            )
        return _JaxPreparedEvaluation(prepared, candidate)

    return prepare


def _native_initial_prepare(
    input_root: Path, expected_parameter_sha256: str
) -> tuple[np.ndarray, PrepareEvaluation]:
    """Prepare one fresh runtime and bind its canonical initial parameters."""

    from examples.jax.parity.cases.native_boozerqa import (
        _prepare_jax_variant_runtime,
    )
    from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
    from examples.jax.parity.input_bundle import read_input_bundle

    bundle, arrays = read_input_bundle(input_root)
    runtime = _prepare_jax_variant_runtime(bundle, arrays, SPEC, None)
    initial_parameters = np.asarray(runtime.initial_parameters)
    if initial_parameters.dtype != np.dtype(np.float64) or initial_parameters.shape != (
        EXPECTED_PARAMETER_COUNT,
    ):
        raise C0EvaluatorError(
            "canonical runtime initial parameters must have exact shape (461,) and "
            "dtype float64"
        )
    initial_parameters = np.ascontiguousarray(initial_parameters, dtype=np.dtype("<f8"))
    if not np.all(np.isfinite(initial_parameters)):
        raise C0EvaluatorError("canonical runtime initial parameters must be finite")
    if _parameter_sha256(initial_parameters) != expected_parameter_sha256:
        raise C0EvaluatorError(
            "canonical runtime initial parameter SHA-256 differs from native reference"
        )
    initial_parameters.setflags(write=False)

    def prepare(candidate: np.ndarray) -> PreparedEvaluation:
        if not np.array_equal(candidate, initial_parameters):
            raise C0EvaluatorError("initial-gate parameters changed before evaluation")
        return _JaxPreparedEvaluation(runtime, initial_parameters)

    return initial_parameters, prepare


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-bundle-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    parser.add_argument("--initial-parameter-sha256", required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--identity-anchor", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    return parser


def _verify_snapshot_import_origins(snapshot_root: Path) -> None:
    import simsopt
    import simsopt_jax
    import simsopt_jax_adapters
    import simsoptpp

    import benchmarks
    import examples

    snapshot_root = snapshot_root.resolve()
    for name, module in (
        ("benchmarks", benchmarks),
        ("examples", examples),
        ("simsopt", simsopt),
        ("simsopt_jax", simsopt_jax),
        ("simsopt_jax_adapters", simsopt_jax_adapters),
        ("simsoptpp", simsoptpp),
    ):
        origin = getattr(module, "__file__", None)
        if isinstance(origin, str):
            origins = (Path(origin).resolve(),)
        else:
            namespace_path = getattr(module, "__path__", ())
            origins = tuple(Path(path).resolve() for path in namespace_path)
        if not origins or any(
            not origin_path.is_relative_to(snapshot_root) for origin_path in origins
        ):
            raise C0EvaluatorError(
                f"module {name!r} resolved outside the immutable snapshot"
            )


def _validate_runtime_contract() -> str:
    raw_contract = os.environ.get("SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_CONTRACT")
    declared_identity = os.environ.get("SINGLE_STAGE_COMPUTE_GRAPH_RUNTIME_IDENTITY")
    if raw_contract is None or declared_identity is None:
        raise C0EvaluatorError("runtime contract environment is missing")
    contract = json.loads(raw_contract)
    if not isinstance(contract, dict) or frozenset(contract) != frozenset(
        {
            "runtime",
            "static_environment",
            "route_environment",
            "policies",
            "expected_runtime_identity_sha256",
        }
    ):
        raise C0EvaluatorError("runtime contract fields are invalid")
    provenance = {
        "interpreter_path": str(Path(sys.executable).absolute()),
        "runtime": contract["runtime"],
        "environment": contract["static_environment"],
        "policies": contract["policies"],
    }
    computed_identity = _runtime_identity(provenance)
    if (
        computed_identity != declared_identity
        or contract["expected_runtime_identity_sha256"] != declared_identity
    ):
        raise C0EvaluatorError("runtime contract identity is inconsistent")
    expected_environment = contract["static_environment"]
    if not isinstance(expected_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in expected_environment.items()
    ):
        raise C0EvaluatorError("runtime contract environment is invalid")
    if normalize_static_timing_environment(os.environ) != expected_environment:
        raise C0EvaluatorError("static runtime environment differs from contract")
    route_environment = contract["route_environment"]
    if not isinstance(route_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in route_environment.items()
    ):
        raise C0EvaluatorError("runtime contract route environment is invalid")
    if normalize_route_environment(os.environ) != route_environment:
        raise C0EvaluatorError("route runtime environment differs from contract")
    expected_policies = contract["policies"]
    if not isinstance(expected_policies, dict):
        raise C0EvaluatorError("runtime numerical policies are invalid")
    blocks = expected_policies.get("quadrature_block_sizes")
    if not isinstance(blocks, list) or not all(
        isinstance(value, int) for value in blocks
    ):
        raise C0EvaluatorError("quadrature policy is invalid")
    observed_policies = observe_effective_numerical_policies(sum(blocks))
    if observed_policies != expected_policies:
        raise C0EvaluatorError("observed runtime policies differ from specimen")
    import jax

    devices = jax.devices()
    runtime = contract["runtime"]
    if not isinstance(runtime, dict):
        raise C0EvaluatorError("runtime contract runtime is invalid")
    observed_runtime = {
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_backend": jax.default_backend(),
        "fp64_x64_enabled": bool(jax.config.jax_enable_x64),
        "cuda_runtime": str(getattr(devices[0].client, "platform_version", "unknown")),
    }
    for key, value in observed_runtime.items():
        if runtime.get(key) != value:
            raise C0EvaluatorError(f"observed runtime differs for {key}")
    return computed_identity


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected_input_sha256 = _sha256(args.input_bundle_sha256, "input_bundle_sha256")
        input_bundle_path = args.input_root / "input_bundle.json"
        if hashlib.sha256(input_bundle_path.read_bytes()).hexdigest() != (
            expected_input_sha256
        ):
            raise C0EvaluatorError("raw input_bundle.json SHA-256 mismatch")
        _verify_snapshot_import_origins(args.snapshot_root)
        _validate_runtime_contract()
        if os.environ.get("JAX_ENABLE_X64", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise C0EvaluatorError("JAX_ENABLE_X64 must enable strict FP64")
        if os.environ.get("JAX_TRANSFER_GUARD") != "disallow":
            raise C0EvaluatorError("JAX_TRANSFER_GUARD must be disallow")
        request = _request_from_environment(os.environ)
        parameter_sha256 = _sha256(args.parameter_sha256, "parameter_sha256")
        initial_parameter_sha256 = _sha256(
            args.initial_parameter_sha256, "initial_parameter_sha256"
        )
        from benchmarks.process_gpu_monitor import (
            ProcessGpuMemoryMonitor,
            ProcessGpuMemoryResult,
        )
        from benchmarks.single_stage_compute_graph_c0_capture import (
            _bind_identity_anchor,
            build_capture_evidence,
            capture_profiled_replay,
        )

        monitor = ProcessGpuMemoryMonitor(
            gpu_uuid=args.gpu_uuid,
            provider_pid=os.getpid(),
            interval_seconds=0.01,
        )
        monitor.start()
        monitor_finished = False

        if request.mode == "initial-gate":
            candidate, prepare = _native_initial_prepare(
                args.input_root, initial_parameter_sha256
            )
        else:
            candidate = _canonical_candidate(args.candidate, parameter_sha256)
            prepare = _native_prepare(args.input_root)

        def capture_replay(
            replay: PreparedEvaluation, timed: EvaluationResult
        ) -> tuple[EvaluationResult, CaptureEvidence]:
            del timed
            nonlocal monitor_finished
            profiled, facts = capture_profiled_replay(
                replay,
                parameter_sha256=parameter_sha256,
                trace_root=args.trace_root,
            )
            measurement = monitor.finish()
            monitor_finished = True
            if not isinstance(measurement, ProcessGpuMemoryResult):
                raise C0EvaluatorError(
                    "nvidia-smi did not observe the evaluator process on the "
                    "authenticated GPU"
                )
            _bind_identity_anchor(
                args.identity_anchor,
                request,
                facts.hlo_module_set_identity,
            )
            return profiled, build_capture_evidence(
                request, parameter_sha256, facts, measurement
            )

        if request.mode in ("initial-gate", "gate", "warm"):
            observation = build_child_observation(
                request,
                candidate,
                prepare,
                None,
                peak_rss=_peak_rss_bytes,
            )
            measurement = monitor.finish()
            monitor_finished = True
            if not isinstance(measurement, ProcessGpuMemoryResult):
                raise C0EvaluatorError(
                    "nvidia-smi did not observe the evaluator process on the "
                    "authenticated GPU"
                )
            observation["sampled_process_gpu_memory_peak_bytes"] = (
                measurement.peak_used_memory_mib * 1024 * 1024
            )
            observation["sampled_process_gpu_memory_source"] = (
                SAMPLED_PROCESS_GPU_MEMORY_SOURCE
            )
        else:
            observation = build_child_observation(
                request,
                candidate,
                prepare,
                capture_replay,
                peak_rss=_peak_rss_bytes,
            )
            if not monitor_finished:
                raise C0EvaluatorError("GPU memory monitor did not complete")
        sys.stdout.write(json.dumps(observation, sort_keys=True, separators=(",", ":")))
        sys.stdout.write("\n")
    except (OSError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"C0 evaluator failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
