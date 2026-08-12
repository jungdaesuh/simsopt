"""Observer-bearing exact-Newton telemetry for staged Phase 0 ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Final, Literal, Protocol, cast

import numpy as np

from benchmarks.single_stage_compute_graph_c0_evaluator import (
    _canonical_candidate,
    _verify_snapshot_import_origins,
)
from benchmarks.single_stage_compute_graph_c0_runner import _runtime_identity
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_route_environment,
    normalize_static_timing_environment,
    observe_effective_numerical_policies,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    Phase0ReceiptError,
    _validate_newton_telemetry,
    canonical_json_bytes,
    canonical_sha256,
)

SCHEMA_ID: Final = "single-stage-compute-graph-newton-telemetry-v2"
ROUTE_ID: Final = "production-exact-newton"
MEASUREMENT_METHOD: Final = "device_resident_fixed_shape_exact_newton_counts"
OBSERVER_ENV: Final = "SIMSOPT_TRACEABLE_EXACT_NEWTON_EXECUTION_COUNTS"
EXPECTED_PARAMETER_COUNT: Final = 461
_OBSERVER_RUN_LOCK: Final = Lock()

LaneId = Literal["rtx5090", "a100"]


class NewtonTelemetryError(RuntimeError):
    """Exact-route telemetry is incomplete, inconsistent, or contaminated."""


@dataclass(frozen=True, slots=True)
class TelemetryIdentity:
    """Immutable identities inherited from passing staged checkpoints."""

    candidate_sha256: str
    specimen_sha256: str
    input_bundle_sha256: str
    source_sha256: str
    runtime_identity_sha256: str
    lane_id: LaneId
    gpu_uuid: str
    gate_checkpoint_sha256: str
    warm_checkpoint_sha256: str
    warm_p50_ns: float

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "specimen_sha256": self.specimen_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "source_sha256": self.source_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "lane_id": self.lane_id,
            "gpu_uuid": self.gpu_uuid,
            "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
            "warm_checkpoint_sha256": self.warm_checkpoint_sha256,
            "warm_p50_ns": self.warm_p50_ns,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Host-materialized production candidate result and device-side counts."""

    objective: float
    raw_objective: float
    gradient: np.ndarray
    solved_state: np.ndarray
    newton_success: bool
    newton_iterations: int
    observer_bearing: bool
    execution_counts: ExecutionCounts


class PreparedCandidateEvaluation(Protocol):
    def evaluate(self) -> CandidateEvaluation: ...


@dataclass(frozen=True, slots=True)
class ExecutionCounts:
    """Fixed-shape execution counts returned by the compiled Newton solve."""

    residual_evaluations: int
    linear_operator_applications: int


PrepareCandidate = Callable[[np.ndarray], PreparedCandidateEvaluation]
Clock = Callable[[], int]


def _sha256(value: str, context: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise NewtonTelemetryError(f"{context} must be a lowercase SHA-256")
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NewtonTelemetryError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    document: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    if frozenset(document) != expected:
        raise NewtonTelemetryError(f"{context} has unexpected or missing fields")


def _validated_identity(identity: TelemetryIdentity) -> TelemetryIdentity:
    for field in (
        "candidate_sha256",
        "specimen_sha256",
        "input_bundle_sha256",
        "source_sha256",
        "runtime_identity_sha256",
        "gate_checkpoint_sha256",
        "warm_checkpoint_sha256",
    ):
        _sha256(str(getattr(identity, field)), field)
    if identity.lane_id not in ("rtx5090", "a100"):
        raise NewtonTelemetryError("lane_id must be 'rtx5090' or 'a100'")
    if not identity.gpu_uuid.strip():
        raise NewtonTelemetryError("gpu_uuid must be non-empty")
    if (
        isinstance(identity.warm_p50_ns, bool)
        or not isinstance(identity.warm_p50_ns, (int, float))
        or not math.isfinite(float(identity.warm_p50_ns))
        or float(identity.warm_p50_ns) <= 0.0
    ):
        raise NewtonTelemetryError("warm_p50_ns must be positive and finite")
    return identity


def verify_input_bundle_bytes(input_root: Path, expected_sha256: str) -> Path:
    """Bind production input_bundle.json bytes to the staged specimen identity."""

    expected_sha256 = _sha256(expected_sha256, "input_bundle_sha256")
    input_bundle_path = input_root / "input_bundle.json"
    actual_sha256 = hashlib.sha256(input_bundle_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise NewtonTelemetryError(
            "raw input_bundle.json SHA-256 differs from telemetry identity"
        )
    return input_root


def validate_runtime_contract(
    value: object, identity: TelemetryIdentity
) -> Mapping[str, object]:
    """Recompute the staged identity and verify the child's effective controls."""

    contract = _mapping(value, "runtime contract")
    _exact_keys(
        contract,
        frozenset(
            {
                "runtime",
                "static_environment",
                "route_environment",
                "policies",
                "expected_runtime_identity_sha256",
            }
        ),
        "runtime contract",
    )
    expected = _sha256(
        str(contract["expected_runtime_identity_sha256"]),
        "runtime contract identity",
    )
    provenance = {
        "interpreter_path": str(Path(sys.executable).absolute()),
        "runtime": contract["runtime"],
        "environment": contract["static_environment"],
        "policies": contract["policies"],
    }
    if (
        _runtime_identity(provenance) != expected
        or expected != identity.runtime_identity_sha256
    ):
        raise NewtonTelemetryError("runtime contract identity is inconsistent")
    environment = _mapping(
        contract["static_environment"], "runtime contract static environment"
    )
    if not all(isinstance(value, str) for value in environment.values()):
        raise NewtonTelemetryError(
            "runtime contract environment values must be strings"
        )
    if normalize_static_timing_environment(os.environ) != environment:
        raise NewtonTelemetryError("static runtime environment differs from contract")
    route_environment = _mapping(
        contract["route_environment"], "runtime contract route environment"
    )
    if normalize_route_environment(os.environ) != route_environment:
        raise NewtonTelemetryError("route runtime environment differs from contract")
    policies = _mapping(contract["policies"], "runtime contract policies")
    blocks = policies.get("quadrature_block_sizes")
    if not isinstance(blocks, list) or not all(
        isinstance(value, int) for value in blocks
    ):
        raise NewtonTelemetryError("runtime quadrature policy is invalid")
    if observe_effective_numerical_policies(sum(blocks)) != policies:
        raise NewtonTelemetryError("observed runtime policies differ from specimen")
    runtime = _mapping(contract["runtime"], "runtime contract runtime")
    import jax

    devices = jax.devices()
    observed_runtime = {
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_backend": jax.default_backend(),
        "fp64_x64_enabled": bool(jax.config.jax_enable_x64),
        "cuda_runtime": str(getattr(devices[0].client, "platform_version", "unknown")),
    }
    for key, observed_value in observed_runtime.items():
        if runtime.get(key) != observed_value:
            raise NewtonTelemetryError(f"observed runtime differs for {key}")
    return contract


@contextmanager
def _exact_execution_observer(enabled: bool):
    previous = os.environ.get(OBSERVER_ENV)
    if enabled:
        os.environ[OBSERVER_ENV] = "1"
    else:
        os.environ.pop(OBSERVER_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(OBSERVER_ENV, None)
        else:
            os.environ[OBSERVER_ENV] = previous


def _validate_candidate_evaluation(
    evaluation: CandidateEvaluation,
    *,
    observer_bearing: bool,
) -> None:
    if not math.isfinite(evaluation.objective) or not math.isfinite(
        evaluation.raw_objective
    ):
        raise NewtonTelemetryError("candidate objectives must be finite")
    if evaluation.gradient.dtype != np.dtype(
        np.float64
    ) or evaluation.gradient.shape != (EXPECTED_PARAMETER_COUNT,):
        raise NewtonTelemetryError("candidate gradient must be FP64 with shape (461,)")
    if evaluation.solved_state.dtype != np.dtype(np.float64):
        raise NewtonTelemetryError("solved Newton state must be FP64")
    if not np.all(np.isfinite(evaluation.gradient)) or not np.all(
        np.isfinite(evaluation.solved_state)
    ):
        raise NewtonTelemetryError("candidate result must contain only finite values")
    if not evaluation.newton_success or evaluation.newton_iterations < 0:
        raise NewtonTelemetryError("production exact Newton must succeed")
    if evaluation.observer_bearing is not observer_bearing:
        raise NewtonTelemetryError(
            "exact-Newton observer state does not match its lane"
        )
    counts = evaluation.execution_counts
    if observer_bearing != (
        counts.residual_evaluations > 0 and counts.linear_operator_applications > 0
    ):
        raise NewtonTelemetryError(
            "exact-Newton device counts are missing or unexpected"
        )
    if not observer_bearing and (
        counts.residual_evaluations != 0 or counts.linear_operator_applications != 0
    ):
        raise NewtonTelemetryError("unobserved exact-Newton counts must be zero")


def _require_exact_numerical_equality(
    unobserved: CandidateEvaluation,
    observed: CandidateEvaluation,
) -> dict[str, bool]:
    equality = {
        "objective_exact": unobserved.objective == observed.objective,
        "raw_objective_exact": unobserved.raw_objective == observed.raw_objective,
        "gradient_exact": np.array_equal(unobserved.gradient, observed.gradient),
        "solved_state_exact": np.array_equal(
            unobserved.solved_state,
            observed.solved_state,
        ),
        "newton_success_exact": (unobserved.newton_success is observed.newton_success),
        "newton_iterations_exact": (
            unobserved.newton_iterations == observed.newton_iterations
        ),
    }
    failed = tuple(name for name, matches in equality.items() if not matches)
    if failed:
        raise NewtonTelemetryError(
            "observer-bearing replay changed numerical results: " + ", ".join(failed)
        )
    return equality


def validate_newton_telemetry_evidence(
    value: object,
    expected_identity: TelemetryIdentity,
) -> Mapping[str, object]:
    """Validate persisted telemetry and bind it to staged runner checkpoints."""

    expected_identity = _validated_identity(expected_identity)
    document = _mapping(value, "newton telemetry evidence")
    _exact_keys(
        document,
        frozenset(
            {
                "schema_id",
                "state",
                "evidence_kind",
                "identity",
                "route_id",
                "warmup_executions_per_lane",
                "numerical_equality",
                "observer",
                "newton_telemetry",
            }
        ),
        "newton telemetry evidence",
    )
    if document["schema_id"] != SCHEMA_ID or document["state"] != "PRODUCED":
        raise NewtonTelemetryError("newton telemetry evidence is not produced v2 data")
    if (
        document["evidence_kind"]
        != "observer_bearing_exact_newton_outside_promotion_timing"
        or document["route_id"] != ROUTE_ID
        or document["warmup_executions_per_lane"] != 1
    ):
        raise NewtonTelemetryError("newton telemetry execution contract is invalid")
    identity = _mapping(document["identity"], "newton telemetry identity")
    if dict(identity) != expected_identity.to_json():
        raise NewtonTelemetryError("newton telemetry identity differs from checkpoints")
    equality = _mapping(document["numerical_equality"], "numerical_equality")
    expected_equality_fields = frozenset(
        {
            "objective_exact",
            "raw_objective_exact",
            "gradient_exact",
            "solved_state_exact",
            "newton_success_exact",
            "newton_iterations_exact",
        }
    )
    _exact_keys(equality, expected_equality_fields, "numerical_equality")
    if any(equality[field] is not True for field in expected_equality_fields):
        raise NewtonTelemetryError("newton telemetry numerical equality is not exact")
    observer = _mapping(document["observer"], "newton telemetry observer")
    if dict(observer) != {
        "api": "device_resident_fixed_shape_exact_newton_counts",
        "device_resident_fixed_shape_counts": True,
        "host_callback_used": False,
        "promotion_timing_included": False,
    }:
        raise NewtonTelemetryError("newton telemetry observer contract is invalid")
    receipt_fields = _mapping(document["newton_telemetry"], "newton_telemetry")
    try:
        _validate_newton_telemetry(receipt_fields, "newton_telemetry")
    except Phase0ReceiptError as error:
        raise NewtonTelemetryError(str(error)) from error
    if receipt_fields["route_id"] != ROUTE_ID:
        raise NewtonTelemetryError("newton telemetry route_id is not production exact")
    unsigned_receipt = dict(receipt_fields)
    claimed_raw_sha256 = unsigned_receipt.pop("raw_evidence_sha256")
    unsigned_document = {
        **dict(document),
        "newton_telemetry": unsigned_receipt,
    }
    if canonical_sha256(unsigned_document) != claimed_raw_sha256:
        raise NewtonTelemetryError("newton telemetry raw evidence digest is invalid")
    return receipt_fields


def _measure_candidate_lanes(
    canonical: np.ndarray,
    prepare_candidate: PrepareCandidate,
    *,
    clock: Clock,
) -> tuple[CandidateEvaluation, CandidateEvaluation, ExecutionCounts, int, int]:
    with _exact_execution_observer(False):
        unobserved_runner = prepare_candidate(canonical)
        _validate_candidate_evaluation(
            unobserved_runner.evaluate(),
            observer_bearing=False,
        )
        unobserved_started_ns = clock()
        unobserved = unobserved_runner.evaluate()
        unobserved_finished_ns = clock()
    _validate_candidate_evaluation(unobserved, observer_bearing=False)

    with _exact_execution_observer(True):
        observed_runner = prepare_candidate(canonical)
        observed_warmup = observed_runner.evaluate()
        _validate_candidate_evaluation(observed_warmup, observer_bearing=True)
        warmup_counts = observed_warmup.execution_counts
        if (
            warmup_counts.residual_evaluations <= 0
            or warmup_counts.linear_operator_applications <= 0
        ):
            raise NewtonTelemetryError(
                "observer warmup did not execute the production exact-Newton route"
            )
        observed_started_ns = clock()
        observed = observed_runner.evaluate()
        _validate_candidate_evaluation(observed, observer_bearing=True)
        counts = observed.execution_counts
        observed_finished_ns = clock()
    return (
        unobserved,
        observed,
        counts,
        unobserved_finished_ns - unobserved_started_ns,
        observed_finished_ns - observed_started_ns,
    )


def collect_newton_telemetry(
    identity: TelemetryIdentity,
    candidate: np.ndarray,
    prepare_candidate: PrepareCandidate,
    *,
    clock: Clock = time.monotonic_ns,
) -> dict[str, object]:
    """Measure one warm control and observer replay outside promotion timing."""

    identity = _validated_identity(identity)
    canonical = np.ascontiguousarray(candidate, dtype=np.dtype("<f8"))
    if canonical.dtype != np.dtype(np.float64) or canonical.shape != (
        EXPECTED_PARAMETER_COUNT,
    ):
        raise NewtonTelemetryError("candidate must be FP64 with shape (461,)")
    if not np.all(np.isfinite(canonical)):
        raise NewtonTelemetryError("candidate must contain only finite values")
    digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    if digest != identity.candidate_sha256:
        raise NewtonTelemetryError("candidate bytes differ from candidate_sha256")
    canonical.setflags(write=False)

    with _OBSERVER_RUN_LOCK:
        unobserved, observed, counts, unobserved_wall_ns, observed_wall_ns = (
            _measure_candidate_lanes(
                canonical,
                prepare_candidate,
                clock=clock,
            )
        )
    if counts.residual_evaluations <= 0 or counts.linear_operator_applications <= 0:
        raise NewtonTelemetryError(
            "observer replay produced non-positive execution counts"
        )
    if unobserved_wall_ns <= 0 or observed_wall_ns <= 0:
        raise NewtonTelemetryError("telemetry wall times must be positive")
    equality = _require_exact_numerical_equality(unobserved, observed)
    observer_effect_ratio = observed_wall_ns / unobserved_wall_ns
    receipt_fields = {
        "telemetry_schema_id": SCHEMA_ID,
        "route_id": ROUTE_ID,
        "measurement_method": MEASUREMENT_METHOD,
        "host_callback_used": False,
        "residual_evaluations": int(counts.residual_evaluations),
        "linear_operator_applications": int(counts.linear_operator_applications),
        "observed_wall_ns": observed_wall_ns,
        "unobserved_wall_ns": unobserved_wall_ns,
        "observer_effect_ratio": observer_effect_ratio,
        "collected_outside_timed_samples": True,
    }
    document = {
        "schema_id": SCHEMA_ID,
        "state": "PRODUCED",
        "evidence_kind": "observer_bearing_exact_newton_outside_promotion_timing",
        "identity": identity.to_json(),
        "route_id": ROUTE_ID,
        "warmup_executions_per_lane": 1,
        "numerical_equality": equality,
        "observer": {
            "api": "device_resident_fixed_shape_exact_newton_counts",
            "device_resident_fixed_shape_counts": True,
            "host_callback_used": False,
            "promotion_timing_included": False,
        },
        "newton_telemetry": receipt_fields,
    }
    receipt_fields["raw_evidence_sha256"] = canonical_sha256(document)
    validate_newton_telemetry_evidence(document, identity)
    return document


class _ObservedEvaluation(Protocol):
    forward_result: Mapping[str, object]


class _Controller(Protocol):
    def value_and_grad(
        self,
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]: ...


class _PreparedRuntime(Protocol):
    initial_parameters: np.ndarray

    def fresh_incumbent_controller(self) -> object: ...


@dataclass(frozen=True, slots=True)
class _ProductionCandidateEvaluation:
    runtime: _PreparedRuntime
    candidate: np.ndarray

    def evaluate(self) -> CandidateEvaluation:
        from examples.jax.parity.cases.native_boozerqa import (
            _host_array,
            _host_bool,
            _host_float,
            _host_int,
        )
        from simsopt_jax_adapters.geo.surface_objectives_traceable import (
            _accepted_incumbent_host_observation_sink,
        )

        observations: list[object] = []
        controller = cast(_Controller, self.runtime.fresh_incumbent_controller())
        with _accepted_incumbent_host_observation_sink(observations.append):
            objective, gradient = controller.value_and_grad(self.candidate)
        if len(observations) != 1:
            raise NewtonTelemetryError(
                "production candidate evaluation did not emit one observation"
            )
        observed = cast(_ObservedEvaluation, observations[0])
        forward = observed.forward_result
        return CandidateEvaluation(
            objective=float(objective),
            raw_objective=_host_float(forward["raw_value"]),
            gradient=np.asarray(gradient, dtype=np.float64),
            solved_state=_host_array(forward["x"]),
            newton_success=_host_bool(forward["newton_success"]),
            newton_iterations=_host_int(forward["newton_iterations"]),
            observer_bearing=_host_bool(
                forward["exact_newton_execution_observer_bearing"]
            ),
            execution_counts=ExecutionCounts(
                residual_evaluations=_host_int(
                    forward["exact_newton_residual_evaluation_count"]
                ),
                linear_operator_applications=_host_int(
                    forward["exact_newton_linear_operator_application_count"]
                ),
            ),
        )


def prepare_production_candidate(
    input_root: Path,
    candidate: np.ndarray,
) -> PreparedCandidateEvaluation:
    """Prepare the canonical C0 JAX runtime without duplicating its solve."""

    from examples.jax.parity.cases.native_boozerqa import (
        _prepare_jax_variant_runtime,
    )
    from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
    from examples.jax.parity.input_bundle import read_input_bundle

    bundle, arrays = read_input_bundle(input_root)
    runtime = cast(
        _PreparedRuntime,
        _prepare_jax_variant_runtime(bundle, arrays, SPEC, None),
    )
    if runtime.initial_parameters.shape != (EXPECTED_PARAMETER_COUNT,):
        raise NewtonTelemetryError("canonical runtime does not expose 461 parameters")
    if np.array_equal(candidate, runtime.initial_parameters):
        raise NewtonTelemetryError("candidate must be a changed state")
    return _ProductionCandidateEvaluation(runtime, candidate)


def write_newton_telemetry(path: Path, document: Mapping[str, object]) -> None:
    """Write one canonical evidence document without overwriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())


def _blocked_document(
    error: Exception,
    identity: TelemetryIdentity | None,
) -> dict[str, object]:
    return {
        "schema_id": SCHEMA_ID,
        "state": "BLOCKED",
        "code": "EXACT_NEWTON_TELEMETRY_BLOCKED",
        "reason": str(error),
        "identity": None if identity is None else identity.to_json(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--specimen-sha256", required=True)
    parser.add_argument("--input-bundle-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--runtime-contract-json")
    parser.add_argument("--lane-id", choices=("rtx5090", "a100"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gate-checkpoint-sha256", required=True)
    parser.add_argument("--warm-checkpoint-sha256", required=True)
    parser.add_argument("--warm-p50-ns", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    identity: TelemetryIdentity | None = None
    try:
        identity = TelemetryIdentity(
            candidate_sha256=options.candidate_sha256,
            specimen_sha256=options.specimen_sha256,
            input_bundle_sha256=options.input_bundle_sha256,
            source_sha256=options.source_sha256,
            runtime_identity_sha256=options.runtime_identity_sha256,
            lane_id=options.lane_id,
            gpu_uuid=options.gpu_uuid,
            gate_checkpoint_sha256=options.gate_checkpoint_sha256,
            warm_checkpoint_sha256=options.warm_checkpoint_sha256,
            warm_p50_ns=options.warm_p50_ns,
        )
        input_root = verify_input_bundle_bytes(
            options.input_root,
            identity.input_bundle_sha256,
        )
        if options.runtime_contract_json is None:
            raise NewtonTelemetryError("runtime contract is required")
        _verify_snapshot_import_origins(Path.cwd())
        validate_runtime_contract(json.loads(options.runtime_contract_json), identity)
        candidate = _canonical_candidate(
            options.candidate,
            identity.candidate_sha256,
        )
        document = collect_newton_telemetry(
            identity,
            candidate,
            lambda values: prepare_production_candidate(input_root, values),
        )
    except (OSError, ValueError, RuntimeError) as error:
        try:
            write_newton_telemetry(
                options.output,
                _blocked_document(error, identity),
            )
        except FileExistsError:
            pass
        sys.stderr.write(f"exact-Newton telemetry failed: {error}\n")
        return 2
    try:
        write_newton_telemetry(options.output, document)
    except OSError as error:
        sys.stderr.write(f"exact-Newton telemetry write failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
