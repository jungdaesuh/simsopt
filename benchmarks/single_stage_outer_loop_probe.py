"""Reduced convergence probe for the target single-stage outer optimizer path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.single_stage_init_parity import (
    _prefix_phase_timings,
    _run_single_stage_case,
    resolve_target_lane_compile_diagnostics,
)
from benchmarks.single_stage_backend_routing import (
    resolve_boozer_least_squares_algorithm,
    resolve_boozer_optimizer_backend,
    resolve_boozer_optimizer_method,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_OPTIMIZER_BACKEND,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
)
from benchmarks.validation_ladder_common import (
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    maybe_initialize_distributed_runtime,
    preparse_platform,
    print_provenance,
    require_x64_runtime,
    resolve_probe_lane,
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    single_stage_proof_contract,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_benchmark_compilation_cache_policy(
    "single_stage_outer_loop_probe",
    requested_platform=REQUESTED_PLATFORM,
)

import jax
import jaxlib

maybe_initialize_distributed_runtime()
jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Single-stage outer-loop probe")

LADDER_RUNG = TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG
_OUTER_LOOP_PROOF_CONTRACT = single_stage_proof_contract(LADDER_RUNG)
TARGET_OUTER_OPTIMIZER_METHOD = str(
    _OUTER_LOOP_PROOF_CONTRACT["required_outer_optimizer_method"]
)
DEFAULT_OUTER_PROOF_MAXITER = int(_OUTER_LOOP_PROOF_CONTRACT["default_maxiter"])
_MIN_ACCEPTED_ITERATIONS = int(_OUTER_LOOP_PROOF_CONTRACT["min_iterations"])
_REQUIRE_OBJECTIVE_DECREASE = bool(
    _OUTER_LOOP_PROOF_CONTRACT.get("require_objective_decrease", False)
)
_REQUIRED_RESULT_KEYS = tuple(_OUTER_LOOP_PROOF_CONTRACT["required_result_keys"])
_PHASE1_INITIAL_TRACE_REGION = "single_stage.outer_optimizer_initial_phase"
_PHASE2_TRACE_REGION = "single_stage.outer_optimizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced real single-stage target lane long enough to prove "
            "the outer optimizer takes a real multi-step descent path without "
            "entering SciPy."
        )
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured probe results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the real single-stage fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains VMEC equilibrium files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Explicit equilibrium path override.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Path to the fixed Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=DEFAULT_SMOKE_NPHI,
        help="Surface toroidal grid points.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=DEFAULT_SMOKE_NTHETA,
        help="Surface poloidal grid points.",
    )
    parser.add_argument(
        "--mpol",
        type=int,
        default=DEFAULT_SMOKE_MPOL,
        help="Surface poloidal mode count.",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=DEFAULT_SMOKE_NTOR,
        help="Surface toroidal mode count.",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=DEFAULT_VOL_TARGET,
        help="Single-stage target volume.",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=DEFAULT_IOTA_TARGET,
        help="Single-stage target iota.",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=(DEFAULT_OPTIMIZER_BACKEND,),
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX target-lane optimizer backend for the outer-loop proof.",
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=(DEFAULT_OPTIMIZER_BACKEND,),
        default=None,
        help=(
            "Optional override for the inner JAX Boozer LS backend. "
            "When provided it must stay ondevice."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_OUTER_PROOF_MAXITER,
        help="Single-stage outer-loop iteration budget for the proof rung.",
    )
    parser.add_argument(
        "--target-lane-boozer-bfgs-tol",
        type=float,
        default=None,
        help="Optional temporary Boozer LS tolerance override for target-lane trials.",
    )
    parser.add_argument(
        "--target-lane-boozer-bfgs-maxiter",
        type=int,
        default=None,
        help="Optional temporary Boozer LS iteration cap for target-lane trials.",
    )
    parser.add_argument(
        "--profile-target-lane",
        action="store_true",
        help="Record target-lane objective profiling breakdowns in the probe payload.",
    )
    parser.add_argument(
        "--profile-target-lane-only",
        action="store_true",
        help=(
            "Build and profile the target-lane runtime bundle without running the "
            "outer optimizer."
        ),
    )
    parser.add_argument(
        "--profile-target-lane-batch-size",
        type=int,
        default=1,
        help=(
            "When profiling the target lane, also record the batched seed-evaluation "
            "path over this many nearby deterministic seed points."
        ),
    )
    parser.add_argument(
        "--jax-profile-dir",
        default=None,
        help=(
            "Optional JAX/XProf trace output directory threaded through to the "
            "single-stage example subprocess."
        ),
    )
    parser.add_argument(
        "--diagnose-target-lane-scaled-phase1",
        action="store_true",
        help=(
            "Run the scaled phase-1 target-lane diagnosis instead of the full "
            "outer-loop proof so the first failing/stalling region can be "
            "localized."
        ),
    )
    parser.add_argument(
        "--record-target-lane-invalid-state-events",
        action="store_true",
        help=(
            "Record rejected target-lane L-BFGS trial states so proof failures "
            "carry a structured first-event postmortem."
        ),
    )
    parser.add_argument(
        "--enable-compile-diagnostics",
        action="store_true",
        help=(
            "Enable scoped JAX compile/cache-miss diagnostics inside the real "
            "single-stage target-lane section."
        ),
    )
    parser.add_argument(
        "--deterministic-gpu-reductions",
        action="store_true",
        help=(
            "Append --xla_gpu_deterministic_ops=true for the CUDA subprocess as "
            "a phase-1 A/B experiment."
        ),
    )
    parser.add_argument(
        "--experimental-target-lane-value-and-grad",
        action="store_true",
        help=(
            "Legacy compatibility flag. The single-stage JAX ondevice target lane "
            "now uses the fused runtime-bundle (value, grad) contract by default."
        ),
    )
    return parser.parse_args()


def _finite_result_keys(results: dict[str, Any]) -> dict[str, bool]:
    return {
        key: bool(np.isfinite(float(results.get(key, np.nan))))
        for key in _REQUIRED_RESULT_KEYS
    }


def _finite_optional_scalar(value: object) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _diagnostic_trace_region_for_phase(phase: object) -> str | None:
    if phase == "phase1":
        return _PHASE1_INITIAL_TRACE_REGION
    if phase == "phase2":
        return _PHASE2_TRACE_REGION
    return None


def _phase1_compile_behavior(
    *,
    compile_diagnostics_requested: bool,
    compile_diagnostics_enabled: bool,
    compile_diagnostics_disable_reason: str | None,
) -> dict[str, Any]:
    diagnostics_enabled = bool(compile_diagnostics_enabled)
    return {
        "diagnostics_requested": bool(compile_diagnostics_requested),
        "diagnostics_enabled": diagnostics_enabled,
        "jax_log_compiles": diagnostics_enabled,
        "jax_explain_cache_misses": diagnostics_enabled,
        "cache_reuse_evidence_valid": diagnostics_enabled,
        "disabled_reason": compile_diagnostics_disable_reason,
    }


def build_phase1_diagnostic_note(
    results: dict[str, Any],
    *,
    failures: list[str],
    compile_diagnostics_requested: bool,
    compile_diagnostics_enabled: bool,
    compile_diagnostics_disable_reason: str | None,
    deterministic_gpu_reductions: bool,
) -> dict[str, Any]:
    note: dict[str, Any] = {
        "reproduced": bool(failures),
        "trace_dir": results.get("JAX_PROFILE_DIR"),
        "termination_message": results.get("TERMINATION_MESSAGE"),
        "initial_phase_iterations": int(results.get("INITIAL_PHASE_ITERATIONS", 0)),
        "total_iterations": int(results.get("iterations", 0)),
        "target_lane_profile_recorded": "TARGET_LANE_PROFILE" in results,
        "compile_behavior": _phase1_compile_behavior(
            compile_diagnostics_requested=compile_diagnostics_requested,
            compile_diagnostics_enabled=compile_diagnostics_enabled,
            compile_diagnostics_disable_reason=compile_diagnostics_disable_reason,
        ),
        "deterministic_gpu_reductions": bool(deterministic_gpu_reductions),
        "first_bad_region": None,
        "first_bad_region_source": None,
        "first_bad_region_detail": None,
    }
    scaled_phase1 = results.get("TARGET_LANE_SCALED_PHASE1_DIAGNOSIS")
    if isinstance(scaled_phase1, dict):
        first_nonfinite_stage = scaled_phase1.get("first_nonfinite_stage")
        if isinstance(first_nonfinite_stage, str):
            note["first_bad_region"] = _PHASE1_INITIAL_TRACE_REGION
            note["first_bad_region_source"] = (
                "TARGET_LANE_SCALED_PHASE1_DIAGNOSIS.first_nonfinite_stage"
            )
            note["first_bad_region_detail"] = first_nonfinite_stage
            return note
    invalid_state = results.get("TARGET_LANE_INVALID_STATE_DIAGNOSIS")
    if isinstance(invalid_state, dict):
        events = invalid_state.get("events")
        if isinstance(events, list) and events:
            first_event = events[0]
            phase = first_event.get("phase")
            note["first_bad_region"] = (
                _diagnostic_trace_region_for_phase(phase) or str(phase)
            )
            note["first_bad_region_source"] = (
                "TARGET_LANE_INVALID_STATE_DIAGNOSIS.events[0]"
            )
            note["first_bad_region_detail"] = {
                "phase": phase,
                "iteration": first_event.get("iteration"),
                "line_search_failed": first_event.get("line_search_failed"),
                "nonfinite_step": first_event.get("nonfinite_step"),
                "stalled_step": first_event.get("stalled_step"),
                "valid_curvature": first_event.get("valid_curvature"),
                "ls_status": first_event.get("ls_status"),
            }
            return note
    if failures:
        note["first_bad_region"] = _PHASE2_TRACE_REGION
        note["first_bad_region_source"] = "probe_failures[0]"
        note["first_bad_region_detail"] = failures[0]
    return note


def evaluate_single_stage_outer_loop_probe(
    results: dict[str, Any],
    *,
    expected_boozer_optimizer_backend: str | None = None,
    expected_boozer_optimizer_method: str | None = None,
    require_accepted_step: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    summary = {
        "rung": LADDER_RUNG,
        "iterations": int(results.get("iterations", 0)),
        "boozer_optimizer_backend": results.get("boozer_optimizer_backend"),
        "boozer_optimizer_method": results.get("boozer_optimizer_method"),
        "outer_optimizer_method": str(results.get("outer_optimizer_method", "")),
        "self_intersecting": bool(results.get("SELF_INTERSECTING", False)),
        "self_intersection_check_available": bool(
            results.get("SELF_INTERSECTION_CHECK_AVAILABLE", True)
        ),
        "finite_result_keys": _finite_result_keys(results),
        "initial_objective": _finite_optional_scalar(results.get("INITIAL_OBJECTIVE")),
        "final_objective": _finite_optional_scalar(results.get("FINAL_OBJECTIVE")),
    }
    summary["objective_decrease"] = (
        None
        if summary["initial_objective"] is None or summary["final_objective"] is None
        else float(summary["initial_objective"] - summary["final_objective"])
    )
    summary["objective_decreased"] = (
        None
        if summary["objective_decrease"] is None
        else bool(summary["objective_decrease"] > 0.0)
    )

    failures: list[str] = []
    if require_accepted_step and summary["iterations"] < _MIN_ACCEPTED_ITERATIONS:
        failures.append(
            "Single-stage outer-loop probe did not complete the required "
            f"{_MIN_ACCEPTED_ITERATIONS} accepted optimizer iterations."
        )
    if summary["outer_optimizer_method"] != TARGET_OUTER_OPTIMIZER_METHOD:
        failures.append(
            "Single-stage outer-loop probe did not use the target "
            f"{TARGET_OUTER_OPTIMIZER_METHOD} method."
        )
    if (
        expected_boozer_optimizer_backend is not None
        and summary["boozer_optimizer_backend"] != expected_boozer_optimizer_backend
    ):
        failures.append(
            "Single-stage outer-loop probe did not use the requested inner "
            f"Boozer backend {expected_boozer_optimizer_backend!r}."
        )
    if (
        expected_boozer_optimizer_method is not None
        and summary["boozer_optimizer_method"] != expected_boozer_optimizer_method
    ):
        failures.append(
            "Single-stage outer-loop probe did not use the requested inner "
            f"Boozer optimizer method {expected_boozer_optimizer_method!r}."
        )
    if summary["self_intersecting"]:
        failures.append(
            "Single-stage outer-loop probe produced a self-intersecting surface."
        )
    if require_accepted_step and _REQUIRE_OBJECTIVE_DECREASE:
        if summary["initial_objective"] is None:
            failures.append(
                "Single-stage outer-loop probe did not report a finite initial objective."
            )
        elif summary["final_objective"] is None:
            failures.append(
                "Single-stage outer-loop probe did not report a finite final objective."
            )
        elif not summary["objective_decreased"]:
            failures.append(
                "Single-stage outer-loop probe did not decrease the objective."
            )
    for key, is_finite in summary["finite_result_keys"].items():
        if not is_finite:
            failures.append(
                f"Single-stage outer-loop probe produced a non-finite {key}."
            )
    return summary, failures


def main() -> None:
    args = parse_args()
    args.disable_target_lane_success_filter = True
    (
        compile_diagnostics_enabled,
        compile_diagnostics_disable_reason,
    ) = resolve_target_lane_compile_diagnostics(
        enable_compile_diagnostics=args.enable_compile_diagnostics,
        diagnose_target_lane_scaled_phase1=args.diagnose_target_lane_scaled_phase1,
        record_target_lane_invalid_state_events=(
            args.record_target_lane_invalid_state_events
        ),
    )
    bootstrap_local_simsopt()
    resolved_boozer_optimizer_backend = resolve_boozer_optimizer_backend(
        args.optimizer_backend,
        args.boozer_optimizer_backend,
    )
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Single-stage outer-loop probe",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "ladder_rung": LADDER_RUNG,
            "fixture": "real-single-stage-init",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": str(Path(args.stage2_bs_path)),
            "optimizer_backend": args.optimizer_backend,
            "boozer_optimizer_backend": resolved_boozer_optimizer_backend,
            "boozer_optimizer_backend_requested": args.boozer_optimizer_backend,
            "outer_maxiter": int(args.maxiter),
            "profile_target_lane_batch_size": int(args.profile_target_lane_batch_size),
            "diagnose_target_lane_scaled_phase1": bool(
                args.diagnose_target_lane_scaled_phase1
            ),
            "record_target_lane_invalid_state_events": bool(
                args.record_target_lane_invalid_state_events
            ),
            "compile_diagnostics_requested": bool(args.enable_compile_diagnostics),
            "compile_diagnostics_enabled": bool(compile_diagnostics_enabled),
            "compile_diagnostics_disable_reason": compile_diagnostics_disable_reason,
            "deterministic_gpu_reductions": bool(
                args.deterministic_gpu_reductions
            ),
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "compile_behavior": describe_compile_behavior(uses_subprocesses=True),
        },
    )
    print_provenance(provenance)

    case = _run_single_stage_case(
        args,
        "jax",
        platform=args.platform,
        benchmark_mode=True,
        load_surface_gamma=False,
        profile_target_lane=args.profile_target_lane,
        profile_target_lane_only=args.profile_target_lane_only,
        diagnose_target_lane_scaled_phase1=args.diagnose_target_lane_scaled_phase1,
        record_target_lane_invalid_state_events=(
            args.record_target_lane_invalid_state_events
        ),
        experimental_target_lane_value_and_grad=(
            args.experimental_target_lane_value_and_grad
        ),
        enable_compile_diagnostics=compile_diagnostics_enabled,
        deterministic_gpu_reductions=args.deterministic_gpu_reductions,
    )
    summary, failures = evaluate_single_stage_outer_loop_probe(
        case["results"],
        expected_boozer_optimizer_backend=resolved_boozer_optimizer_backend,
        expected_boozer_optimizer_method=resolve_boozer_optimizer_method(
            resolved_boozer_optimizer_backend,
            least_squares_algorithm=resolve_boozer_least_squares_algorithm(
                resolved_boozer_optimizer_backend
            ),
        ),
        require_accepted_step=not (
            args.profile_target_lane_only or args.diagnose_target_lane_scaled_phase1
        ),
    )
    payload = {
        "rung": LADDER_RUNG,
        "provenance": provenance,
        "results": case["results"],
        "probe": summary,
        "timings": {
            "jax_elapsed_s": float(case["elapsed_s"]),
            "jax_outer_elapsed_s": float(case["elapsed_s"]),
            **_prefix_phase_timings("jax", case["phase_timings"]),
        },
        "failures": failures,
        "passed": not failures,
    }
    if "TARGET_LANE_PROFILE" in case["results"]:
        payload["target_lane_profile"] = case["results"]["TARGET_LANE_PROFILE"]
    if "JAX_COMPILE_DIAGNOSTICS" in case["results"]:
        payload["compile_diagnostics"] = case["results"]["JAX_COMPILE_DIAGNOSTICS"]
    if (
        args.jax_profile_dir
        or args.enable_compile_diagnostics
        or args.deterministic_gpu_reductions
        or args.record_target_lane_invalid_state_events
        or args.diagnose_target_lane_scaled_phase1
        or "TARGET_LANE_INVALID_STATE_DIAGNOSIS" in case["results"]
        or "TARGET_LANE_SCALED_PHASE1_DIAGNOSIS" in case["results"]
    ):
        payload["phase1_diagnostic_note"] = build_phase1_diagnostic_note(
            case["results"],
            failures=failures,
            compile_diagnostics_requested=args.enable_compile_diagnostics,
            compile_diagnostics_enabled=compile_diagnostics_enabled,
            compile_diagnostics_disable_reason=compile_diagnostics_disable_reason,
            deterministic_gpu_reductions=args.deterministic_gpu_reductions,
        )
    write_json(args.output_json, payload)
    if failures:
        print("SINGLE-STAGE OUTER-LOOP PROBE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("SINGLE-STAGE OUTER-LOOP PROBE PASSED")


if __name__ == "__main__":
    main()
