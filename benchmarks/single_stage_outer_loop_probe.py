"""Reduced accepted-step probe for the target single-stage outer optimizer path."""

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
)
from benchmarks.single_stage_backend_routing import (
    resolve_boozer_least_squares_algorithm,
    resolve_boozer_optimizer_backend,
    resolve_boozer_optimizer_method,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
)
from benchmarks.validation_ladder_common import (
    apply_compilation_cache_policy,
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
apply_compilation_cache_policy()

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
_REQUIRED_RESULT_KEYS = tuple(_OUTER_LOOP_PROOF_CONTRACT["required_result_keys"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced real single-stage target lane long enough to prove "
            "the outer optimizer accepts a step without entering SciPy."
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
        choices=("scipy", "hybrid", "ondevice"),
        default="ondevice",
        help="JAX Boozer optimizer backend to prove on the target outer path.",
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default=None,
        help=(
            "Optional override for the inner JAX Boozer LS backend. "
            "Defaults to --optimizer-backend when omitted."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_OUTER_PROOF_MAXITER,
        help="Single-stage outer-loop iteration budget for the proof rung.",
    )
    parser.add_argument(
        "--profile-target-lane",
        action="store_true",
        help="Record target-lane objective profiling breakdowns in the probe payload.",
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


def evaluate_single_stage_outer_loop_probe(
    results: dict[str, Any],
    *,
    expected_boozer_optimizer_backend: str | None = None,
    expected_boozer_optimizer_method: str | None = None,
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
    }

    failures: list[str] = []
    if summary["iterations"] < _MIN_ACCEPTED_ITERATIONS:
        failures.append(
            "Single-stage outer-loop probe did not accept an optimizer step."
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
    for key, is_finite in summary["finite_result_keys"].items():
        if not is_finite:
            failures.append(
                f"Single-stage outer-loop probe produced a non-finite {key}."
            )
    return summary, failures


def main() -> None:
    args = parse_args()
    args.disable_target_lane_success_filter = True
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
        experimental_target_lane_value_and_grad=(
            args.experimental_target_lane_value_and_grad
        ),
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
    write_json(args.output_json, payload)
    if failures:
        print("SINGLE-STAGE OUTER-LOOP PROBE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("SINGLE-STAGE OUTER-LOOP PROBE PASSED")


if __name__ == "__main__":
    main()
