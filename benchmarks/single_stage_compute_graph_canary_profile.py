"""Isolated, non-timing C1/C2 changed-state profile probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from benchmarks.single_stage_compute_graph_c0_capture import capture_profiled_replay
from benchmarks.single_stage_compute_graph_c0_evaluator import (
    EvaluationResult,
    _canonical_candidate,
    _require_equal_results,
    _validate_runtime_contract,
    _verify_snapshot_import_origins,
)
from benchmarks.single_stage_compute_graph_canary_evaluator import (
    CanaryEvaluation,
    CanaryEvaluatorError,
    CanaryVariant,
    _prepare_production_canary,
    _validate_telemetry,
    _variant,
)
from benchmarks.single_stage_compute_graph_command_buffer_control import (
    NVTX_RANGE_PREFIX,
    _nvtx_library,
)
from benchmarks.single_stage_compute_graph_native_reference import _parameter_sha256

PROFILE_CHILD_SCHEMA_ID: Final = "single-stage-compute-graph-canary-profile-child-v1"


class CanaryProfileError(RuntimeError):
    """The variant profile probe cannot produce claimable raw evidence."""


class _CaptureAdapter:
    def __init__(self, prepared) -> None:
        self._prepared = prepared
        self.evaluation: CanaryEvaluation | None = None

    def evaluate_once(self) -> EvaluationResult:
        self.evaluation = self._prepared.evaluate_once()
        return self.evaluation.numerical


def _exact_hlo_ir_sha256(prepared) -> str:
    runtime = prepared._runtime
    controller = runtime.fresh_incumbent_controller()
    lowered = runtime.incumbent_evaluator.lower(
        prepared._candidate, controller.current_inner_state
    )
    hlo_module = lowered.compiler_ir(dialect="hlo")
    serialized = hlo_module.as_serialized_hlo_module_proto()
    if not isinstance(serialized, bytes) or not serialized:
        raise CanaryProfileError("lowered HLO serialization is unavailable")
    return hashlib.sha256(serialized).hexdigest()


def _numerical_document(result: EvaluationResult) -> dict[str, object]:
    return {
        "objective_dtype": "float64",
        "objective": result.objective,
        "gradient_dtype": "float64",
        "gradient": result.gradient.tolist(),
        "inner_newton_success": result.inner_newton_success,
        "adjoint_success": result.adjoint_success,
        "residual_certificates": dict(result.residual_certificates),
    }


def run_profile_probe(
    *,
    variant: CanaryVariant,
    prepared,
    parameter_sha256: str,
    trace_root: Path,
    nvtx_library: Path,
    hlo_ir_identity: Callable[[object], str] = _exact_hlo_ir_sha256,
) -> dict[str, object]:
    """Warm once, then profile exactly one numerically identical replay."""

    hlo_ir_sha256 = hlo_ir_identity(prepared)
    warm = prepared.evaluate_once()
    warm_telemetry = _validate_telemetry(variant, warm.telemetry)
    adapter = _CaptureAdapter(prepared)
    nvtx = _nvtx_library(nvtx_library)
    label = (NVTX_RANGE_PREFIX + parameter_sha256).encode("ascii")
    nvtx.nvtxRangePushA(label)
    try:
        profiled_numerical, facts = capture_profiled_replay(
            adapter,
            parameter_sha256=parameter_sha256,
            trace_root=trace_root,
        )
    finally:
        nvtx.nvtxRangePop()
    if adapter.evaluation is None:
        raise CanaryProfileError("profiled replay produced no variant observation")
    profiled_telemetry = _validate_telemetry(variant, adapter.evaluation.telemetry)
    _require_equal_results(warm.numerical, profiled_numerical)
    if warm_telemetry != profiled_telemetry:
        raise CanaryProfileError("warm/profile variant telemetry differs")
    return {
        "schema_id": PROFILE_CHILD_SCHEMA_ID,
        "status": "PASS",
        "variant": variant,
        "mode": "profile",
        "parameter_sha256": parameter_sha256,
        **_numerical_document(profiled_numerical),
        "telemetry": profiled_telemetry,
        "capture": {
            "hlo_ir_sha256": hlo_ir_sha256,
            "hlo_module_set_identity": facts.hlo_module_set_identity,
            "hlo_modules": list(facts.hlo_modules),
            "pjrt_execute_count": facts.pjrt_execute_count,
            "kernel_launch_count": facts.kernel_launch_count,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("C1", "C2"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    parser.add_argument("--initial-parameter-sha256", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--nvtx-library", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variant = _variant(args.variant)
    try:
        _verify_snapshot_import_origins(args.snapshot_root)
        _validate_runtime_contract()
        candidate = _canonical_candidate(args.candidate, args.parameter_sha256)
        prepared = _prepare_production_canary(
            args.input_root, variant, candidate, "gate"
        )
        if _parameter_sha256(prepared._runtime.initial_parameters) != (
            args.initial_parameter_sha256
        ):
            raise CanaryProfileError(
                "prepared baseline state differs from initial parameter SHA"
            )
        document: Mapping[str, object] = run_profile_probe(
            variant=variant,
            prepared=prepared,
            parameter_sha256=args.parameter_sha256,
            trace_root=args.trace_root,
            nvtx_library=args.nvtx_library,
        )
    except (CanaryEvaluatorError, CanaryProfileError, ValueError) as error:
        document = {
            "schema_id": PROFILE_CHILD_SCHEMA_ID,
            "status": "BLOCKED",
            "variant": variant,
            "blocker": {"code": "PROFILE_PROBE_FAILED", "reason": str(error)},
        }
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0 if document["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
