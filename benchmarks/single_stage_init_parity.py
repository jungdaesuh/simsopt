"""Tier 3 real single-stage init parity probe on a fixed Columbia seed."""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    find_single_file,
    gpu_proof_parity_contract,
    load_json,
    max_pointwise_geometry_drift,
    maybe_initialize_distributed_runtime,
    optimizer_drift_tolerances,
    parity_ladder_tolerances,
    preparse_platform,
    print_provenance,
    require_requested_platform_runtime,
    require_x64_runtime,
    relative_error,
    resolve_probe_lane,
    repo_pythonpath_env,
    run_python_script,
    single_stage_proof_contract,
    write_json,
)
from benchmarks.single_stage_parity_matrix import (
    LANE_CPU_SCIPY,
    LANE_JAX_CPU,
    LANE_JAX_GPU,
    _compare_optimizer_state_trace_pair,
    _file_sha256,
    _json_hash,
    _objective_config_hash_from_results,
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


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_benchmark_compilation_cache_policy(
    "single_stage_init_parity",
    requested_platform=REQUESTED_PLATFORM,
)
bootstrap_local_simsopt()

import jax
import jaxlib

maybe_initialize_distributed_runtime()
_RUNTIME_CONTEXT = "Single-stage init parity"

jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context=_RUNTIME_CONTEXT)
require_requested_platform_runtime(
    jax,
    requested_platform=REQUESTED_PLATFORM,
    context=_RUNTIME_CONTEXT,
)

_TIER3_TOLERANCES = optimizer_drift_tolerances("tier3_single_stage_init")
IOTA_ABS_TOL = _TIER3_TOLERANCES["final_iota_abs_tol"]
VOLUME_REL_TOL = _TIER3_TOLERANCES["final_volume_rel_tol"]
FIELD_ERROR_REL_TOL = _TIER3_TOLERANCES["field_error_rel_tol"]
SURFACE_GEOMETRY_REL_TOL = _TIER3_TOLERANCES["surface_geometry_rel_tol"]
TARGET_OPTIMIZER_BACKEND = DEFAULT_OPTIMIZER_BACKEND
SCIPY_JAX_OPTIMIZER_BACKEND = "scipy-jax"
SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND = "scipy-jax-fullgraph"
TARGET_OPTIMIZER_BACKENDS = (
    TARGET_OPTIMIZER_BACKEND,
    SCIPY_JAX_OPTIMIZER_BACKEND,
    SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND,
)
DEFAULT_OUTER_MAXITER = 0
TRACE_PARITY_OUTER_MAXLS = 8
_TARGET_LANE_FINAL_ONLY_SYNC = "final-only"
_TARGET_LANE_PER_ACCEPT_SYNC = "per-accept"
_OUTER_LOOP_PROOF_CONTRACT = single_stage_proof_contract(
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG
)
_OUTER_LOOP_REQUIRED_RESULT_KEYS = tuple(
    _OUTER_LOOP_PROOF_CONTRACT["required_result_keys"]
)
_TARGET_OUTER_OPTIMIZER_METHOD = str(
    _OUTER_LOOP_PROOF_CONTRACT["required_outer_optimizer_method"]
)
_TARGET_OPTIMIZER_METHOD_BY_BACKEND = {
    TARGET_OPTIMIZER_BACKEND: _TARGET_OUTER_OPTIMIZER_METHOD,
    SCIPY_JAX_OPTIMIZER_BACKEND: "lbfgs-scipy-jax",
    SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND: "lbfgs-scipy-jax-fullgraph",
}
_TARGET_LANE_COMPILE_DIAGNOSTICS_HOST_CALLBACK_REASON = (
    "compile diagnostics are disabled when Phase 1 host-callback diagnostics "
    "are enabled because that mode does not provide normal cache-reuse evidence"
)
_SAME_CANDIDATE_X_ATOL = 1e-8
_OPTIMIZER_PATH_CANDIDATE_SPLIT_ATOL = 1e-12
_SAME_CANDIDATE_SCALAR_RTOL = 1e-10
_SAME_CANDIDATE_SCALAR_ATOL = 1e-12
_SAME_CANDIDATE_GRADIENT_TOLERANCES = parity_ladder_tolerances("ls-wrapper-gradient")
_SAME_CANDIDATE_GRADIENT_RTOL = _SAME_CANDIDATE_GRADIENT_TOLERANCES["rtol"]
_SAME_CANDIDATE_GRADIENT_ATOL = _SAME_CANDIDATE_GRADIENT_TOLERANCES["atol"]
_IOTA_DECOMPOSITION_DIAGNOSTIC_ATOL = 1e-13
_IOTA_DECOMPOSITION_DIAGNOSTIC_RTOL = 1e-12
_SAME_CANDIDATE_HARDWARE_KEYS = (
    "curve_curve_min_dist",
    "curve_surface_min_dist",
    "surface_vessel_min_dist",
    "max_curvature",
)
_SAME_CANDIDATE_HARDWARE_MARGIN_KEYS = (
    "curve_curve_min_dist",
    "curve_surface_min_dist",
    "surface_vessel_min_dist",
    "max_curvature",
)
_SAME_CANDIDATE_FAILURE_SCALAR_KEYS = (
    "hardware_score",
    "solver_score",
    "penalty_multiplier",
    "penalty",
)
_SAME_CANDIDATE_FAILURE_EXACT_KEYS = (
    "reject_class",
    "intersecting",
    "solver_success",
    "failure_count",
    "search_policy",
    "donor_class",
)
_SAME_CANDIDATE_BOOZER_METADATA_EXACT_KEYS = (
    "boozer_type",
    "boozer_optimizer_backend",
    "boozer_least_squares_algorithm",
    "linearization_kind",
    "linear_solve_backend",
    "dense_newton_steps_materialized",
    "dense_linear_solve_factors_available",
    "dense_refinement_ran",
    "final_step_dense_refinement_ran",
)
_SAME_CANDIDATE_BOOZER_METADATA_SHAPE_KEYS = ("dense_hessian_shape",)
_SAME_CANDIDATE_BOOZER_METADATA_NUMERIC_KEYS = (
    "newton_tol",
    "newton_maxiter",
    "newton_iter",
    "final_gradient_norm",
    "final_gradient_inf_norm",
    "dense_hessian_bytes",
)
_IOTA_DECOMPOSITION_LAYER_FIELDS = (
    (
        "solved_state",
        (
            ("scalar", ("solved_iota",)),
            ("scalar", ("solved_G",)),
            ("vector", ("solved_surface_dofs",)),
        ),
    ),
    (
        "linear_solve_factors",
        (
            ("vector", ("linear_solve_factors", "P")),
            ("vector", ("linear_solve_factors", "L")),
            ("vector", ("linear_solve_factors", "U")),
        ),
    ),
    ("dJ_ds", (("vector", ("dJ_ds",)),)),
    ("adjoint", (("vector", ("adjoint",)),)),
    (
        "optimizer_projection_gradient",
        (("vector", ("optimizer_projection_gradient",)),),
    ),
    ("penalty_scale", (("scalar", ("penalty_scale",)),)),
    (
        "penalty_optimizer_gradient",
        (("vector", ("penalty_optimizer_gradient",)),),
    ),
    (
        "weighted_penalty_optimizer_gradient",
        (("vector", ("weighted_penalty_optimizer_gradient",)),),
    ),
)
_BOOZER_SOLVE_DECOMPOSITION_LAYER_FIELDS = (
    (
        "pre_newton_state",
        (
            ("scalar", ("pre_newton", "iota")),
            ("scalar", ("pre_newton", "G")),
            ("vector", ("pre_newton", "surface_dofs")),
            ("vector", ("pre_newton", "decision_vector")),
        ),
    ),
    (
        "pre_newton_objective_gradient",
        (
            ("scalar", ("pre_newton", "fun")),
            ("vector", ("pre_newton", "gradient")),
        ),
    ),
    (
        "final_solved_state",
        (
            ("scalar", ("final_iota",)),
            ("scalar", ("final_G",)),
            ("vector", ("final_surface_dofs",)),
            ("vector", ("final_decision_vector",)),
        ),
    ),
    ("final_objective", (("scalar", ("final_fun",)),)),
    ("final_residual", (("vector", ("final_residual",)),)),
    ("final_gradient", (("vector", ("final_gradient",)),)),
    ("final_hessian", (("vector", ("final_hessian",)),)),
    (
        "linear_solve_factors",
        (
            ("vector", ("linear_solve_factors", "P")),
            ("vector", ("linear_solve_factors", "L")),
            ("vector", ("linear_solve_factors", "U")),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real single-stage init path on CPU vs JAX and compare outcomes."
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
        help="Path to write structured comparison results.",
    )
    parser.add_argument(
        "--case-artifacts-dir",
        default=None,
        help=(
            "Directory for durable per-lane single-stage outputs. When provided, "
            "reference_outputs, target_outputs, and any compiled runtime seed spec "
            "are preserved for post-run artifact extraction."
        ),
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
        "--warm-start-run-dir",
        default=None,
        help=(
            "Optional single-stage donor directory containing surf_opt.json, "
            "results.json, and biot_savart_opt.json. Used as the reference "
            "seed source and as the source for the JAX runtime seed spec."
        ),
    )
    parser.add_argument(
        "--jax-runtime-seed-spec",
        default=None,
        help=(
            "Optional precompiled immutable JAX runtime seed spec. When omitted, "
            "the parity runner compiles one from the CPU reference run."
        ),
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=20,
        help="Number of fixed TF coils in the single-stage seed package.",
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
        choices=TARGET_OPTIMIZER_BACKENDS,
        default=TARGET_OPTIMIZER_BACKEND,
        help=(
            "JAX outer optimizer backend for the init probe. Use "
            "scipy-jax-fullgraph for CPU/SciPy-compatible host control over "
            "the full JAX value/grad graph."
        ),
    )
    parser.add_argument(
        "--reference-optimizer-method",
        choices=("lbfgs", "lbfgs-trace"),
        default="lbfgs",
        help=(
            "CPU/reference outer optimizer method. The default lbfgs is the "
            "SciPy CPU/C++ parity lane; lbfgs-trace is a non-SciPy host-core "
            "diagnostic."
        ),
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=(TARGET_OPTIMIZER_BACKEND,),
        default=None,
        help=(
            "Optional override for the inner JAX Boozer LS backend. "
            "When provided it must stay ondevice."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_OUTER_MAXITER,
        help=(
            "Single-stage outer-loop iteration budget. "
            "Use 0 to keep the historical init-only Tier 3 probe shape."
        ),
    )
    parser.add_argument(
        "--initial-step-scale",
        type=float,
        default=1.0,
        help=(
            "Initial scaled outer-phase step size passed through to the "
            "single-stage runner. This wrapper defaults it explicitly to 1.0 "
            "so CPU/C++ and JAX CPU outer runs use the same phase-2 contract."
        ),
    )
    parser.add_argument(
        "--initial-step-maxiter",
        type=int,
        default=0,
        help=(
            "Initial scaled outer-phase iteration budget passed through to the "
            "single-stage runner. This wrapper defaults it explicitly to 0 so "
            "CPU/C++ and JAX CPU compare the shared phase-2 run shape."
        ),
    )
    parser.add_argument(
        "--outer-maxls",
        type=int,
        default=TRACE_PARITY_OUTER_MAXLS,
        help=(
            "Strong-Wolfe line-search budget passed to both parity lanes. "
            "The default matches the current target-lane production budget."
        ),
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help=(
            "Request benchmark-mode target-lane execution. "
            "This skips heavy single-stage artifacts and therefore skips the "
            "surface-geometry drift check in this parity wrapper."
        ),
    )
    parser.add_argument(
        "--disable-target-lane-success-filter",
        action="store_true",
        help=(
            "Proof-only target-lane mode: bypass the outer-loop hardware "
            "success filter while preserving the JAX value/grad and optimizer "
            "execution path."
        ),
    )
    parser.add_argument(
        "--record-objective-evaluation-trace",
        action="store_true",
        help=(
            "Thread through detailed per-objective-evaluation trace recording "
            "for CPU/JAX fullgraph parity debugging."
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
    return parser.parse_args()


def _single_stage_script_path() -> Path:
    return (
        REPO_ROOT
        / "examples"
        / "single_stage_optimization"
        / "SINGLE_STAGE"
        / "single_stage_banana_example.py"
    )


def _resolve_target_lane_sync_policy(
    backend: str,
    args: argparse.Namespace,
) -> str:
    if backend != "jax":
        return _TARGET_LANE_PER_ACCEPT_SYNC
    if args.optimizer_backend != TARGET_OPTIMIZER_BACKEND:
        return _TARGET_LANE_PER_ACCEPT_SYNC
    if int(args.maxiter) <= 0:
        return _TARGET_LANE_PER_ACCEPT_SYNC
    return _TARGET_LANE_FINAL_ONLY_SYNC


def _expected_target_outer_optimizer_method(optimizer_backend: str) -> str:
    return _TARGET_OPTIMIZER_METHOD_BY_BACKEND[optimizer_backend]


def _extract_phase_timings(results: dict[str, Any]) -> dict[str, float]:
    raw_timings = results.get("TIMINGS")
    if not isinstance(raw_timings, dict):
        return {}
    timings: dict[str, float] = {}
    for key, value in raw_timings.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            timings[str(key)] = float(value)
    return timings


def _prefix_phase_timings(prefix: str, timings: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in timings.items()}


def _target_lane_label(args: argparse.Namespace, case: dict[str, Any]) -> str:
    provenance = dict(case["results"]).get("provenance", {})
    backend = str(provenance.get("backend", args.platform)).lower()
    return LANE_JAX_GPU if backend in {"cuda", "gpu"} else LANE_JAX_CPU


def _reference_lane_label(reference_backend: str) -> str:
    return LANE_JAX_CPU if reference_backend == "jax" else LANE_CPU_SCIPY


def _single_stage_full_run_family_id(
    args: argparse.Namespace,
    *,
    runtime_seed_spec_hash: str | None,
    objective_configuration_hash: str | None,
) -> str:
    return _json_hash(
        {
            "runtime_seed_spec_hash": runtime_seed_spec_hash,
            "objective_configuration_hash": objective_configuration_hash,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_bs_path": _display_path(Path(args.stage2_bs_path)),
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "vol_target": float(args.vol_target),
            "iota_target": float(args.iota_target),
            "num_tf_coils": int(getattr(args, "num_tf_coils", 20)),
            "optimizer_backend": args.optimizer_backend,
            "reference_optimizer_method": args.reference_optimizer_method,
            "initial_step_scale": float(args.initial_step_scale),
            "initial_step_maxiter": int(args.initial_step_maxiter),
            "outer_maxls": int(args.outer_maxls),
            "maxiter": int(args.maxiter),
        }
    )


def _single_stage_full_run_lane_contract(
    case: dict[str, Any],
    *,
    runtime_seed_spec_hash: str | None,
    run_family_id: str,
) -> dict[str, Any]:
    results = dict(case["results"])
    objective_hash, missing_objective_keys = _objective_config_hash_from_results(
        results
    )
    run_dir = Path(str(case["run_dir"]))
    provenance = results.get("provenance", {})
    return {
        "run_dir": str(run_dir),
        "results_json": str(run_dir / "results.json"),
        "progress_json": str(case["outer_optimizer_progress_json"]),
        "runtime_seed_spec_hash": runtime_seed_spec_hash,
        "objective_configuration_hash": objective_hash,
        "missing_objective_config_keys": missing_objective_keys,
        "run_family_id": run_family_id,
        "init_only": results.get("init_only"),
        "generated_at_utc": provenance.get("generated_at_utc"),
        "repo_sha": provenance.get("repo_sha"),
    }


def build_single_stage_full_run_artifact_contract(
    args: argparse.Namespace,
    *,
    reference_backend: str,
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
    jax_seed_spec: Path,
) -> dict[str, Any]:
    runtime_seed_spec_hash = _file_sha256(jax_seed_spec)
    reference_objective_hash, _ = _objective_config_hash_from_results(
        dict(cpu_case["results"])
    )
    target_objective_hash, _ = _objective_config_hash_from_results(
        dict(jax_case["results"])
    )
    run_family_id = _single_stage_full_run_family_id(
        args,
        runtime_seed_spec_hash=runtime_seed_spec_hash,
        objective_configuration_hash=reference_objective_hash
        if reference_objective_hash is not None
        else target_objective_hash,
    )
    return {
        "schema_version": 1,
        "runtime_seed_spec": str(jax_seed_spec),
        "runtime_seed_spec_hash": runtime_seed_spec_hash,
        "run_family_id": run_family_id,
        "lanes": {
            _reference_lane_label(reference_backend): (
                _single_stage_full_run_lane_contract(
                    cpu_case,
                    runtime_seed_spec_hash=runtime_seed_spec_hash,
                    run_family_id=run_family_id,
                )
            ),
            _target_lane_label(args, jax_case): _single_stage_full_run_lane_contract(
                jax_case,
                runtime_seed_spec_hash=runtime_seed_spec_hash,
                run_family_id=run_family_id,
            ),
        },
    }


def _append_optional_single_stage_flags(
    command: list[str],
    *,
    benchmark_mode: bool,
    profile_target_lane: bool,
    profile_target_lane_only: bool,
    diagnose_target_lane_scaled_phase1: bool,
    record_target_lane_invalid_state_events: bool,
    profile_target_lane_batch_size: int | None,
    enable_compile_diagnostics: bool,
    jax_profile_dir: str | None,
    experimental_target_lane_value_and_grad: bool,
    disable_target_lane_success_filter: bool,
    record_objective_evaluation_trace: bool,
    target_lane_boozer_bfgs_tol: float | None = None,
    target_lane_boozer_bfgs_maxiter: int | None = None,
    target_lane_boozer_newton_tol: float | None = None,
    target_lane_boozer_newton_maxiter: int | None = None,
    replay_objective_evaluation_trace: Path | None = None,
) -> None:
    if benchmark_mode:
        command.append("--benchmark-mode")
    if profile_target_lane:
        command.append("--profile-target-lane")
    if profile_target_lane_only:
        command.append("--profile-target-lane-only")
    if diagnose_target_lane_scaled_phase1:
        command.append("--diagnose-target-lane-scaled-phase1")
    if record_target_lane_invalid_state_events:
        command.append("--diagnostic-callbacks")
    if (
        profile_target_lane_batch_size is not None
        and int(profile_target_lane_batch_size) > 1
    ):
        command.extend(
            [
                "--profile-target-lane-batch-size",
                str(int(profile_target_lane_batch_size)),
            ]
        )
    effective_compile_diagnostics, _ = resolve_target_lane_compile_diagnostics(
        enable_compile_diagnostics=enable_compile_diagnostics,
        diagnose_target_lane_scaled_phase1=diagnose_target_lane_scaled_phase1,
        record_target_lane_invalid_state_events=record_target_lane_invalid_state_events,
    )
    if effective_compile_diagnostics:
        command.append("--record-jax-compile-diagnostics")
    if jax_profile_dir:
        command.extend(["--jax-profile-dir", jax_profile_dir])
    if experimental_target_lane_value_and_grad:
        command.append("--experimental-target-lane-value-and-grad")
    if disable_target_lane_success_filter:
        command.append("--disable-target-lane-success-filter")
    if record_objective_evaluation_trace:
        command.append("--record-objective-evaluation-trace")
    if replay_objective_evaluation_trace is not None:
        command.extend(
            [
                "--replay-objective-evaluation-trace",
                str(replay_objective_evaluation_trace),
            ]
        )
    if target_lane_boozer_bfgs_tol is not None:
        command.extend(
            [
                "--target-lane-boozer-bfgs-tol",
                str(float(target_lane_boozer_bfgs_tol)),
            ]
        )
    if target_lane_boozer_bfgs_maxiter is not None:
        command.extend(
            [
                "--target-lane-boozer-bfgs-maxiter",
                str(int(target_lane_boozer_bfgs_maxiter)),
            ]
        )
    if target_lane_boozer_newton_tol is not None:
        command.extend(
            [
                "--target-lane-boozer-newton-tol",
                str(float(target_lane_boozer_newton_tol)),
            ]
        )
    if target_lane_boozer_newton_maxiter is not None:
        command.extend(
            [
                "--target-lane-boozer-newton-maxiter",
                str(int(target_lane_boozer_newton_maxiter)),
            ]
        )


def resolve_target_lane_compile_diagnostics(
    *,
    enable_compile_diagnostics: bool,
    diagnose_target_lane_scaled_phase1: bool,
    record_target_lane_invalid_state_events: bool,
) -> tuple[bool, str | None]:
    """Resolve whether compile/cache diagnostics can run on this target-lane mode."""
    if not enable_compile_diagnostics:
        return False, None
    if diagnose_target_lane_scaled_phase1 or record_target_lane_invalid_state_events:
        return False, _TARGET_LANE_COMPILE_DIAGNOSTICS_HOST_CALLBACK_REASON
    return True, None


@contextlib.contextmanager
def _resolved_single_stage_output_root(
    output_root: Path | None,
    *,
    backend: str,
) -> Iterator[Path]:
    """Yield a concrete output_root, creating a temp dir only when caller omits one."""
    if output_root is not None:
        yield Path(output_root)
        return
    with tempfile.TemporaryDirectory(
        prefix=f"single-stage-init-{backend}-"
    ) as temp_dir:
        yield Path(temp_dir) / "outputs"


def _run_single_stage_case(
    args: argparse.Namespace,
    backend: str,
    *,
    platform: str,
    benchmark_mode: bool = False,
    load_surface_gamma: bool = True,
    profile_target_lane: bool = False,
    profile_target_lane_only: bool = False,
    diagnose_target_lane_scaled_phase1: bool = False,
    record_target_lane_invalid_state_events: bool = False,
    experimental_target_lane_value_and_grad: bool = False,
    enable_compile_diagnostics: bool = False,
    deterministic_gpu_reductions: bool = False,
    output_root: Path | None = None,
    jax_runtime_seed_spec: Path | None = None,
    replay_objective_evaluation_trace: Path | None = None,
) -> dict[str, Any]:
    script_path = _single_stage_script_path()
    effective_platform = platform if backend == "jax" else "cpu"
    with _resolved_single_stage_output_root(
        output_root, backend=backend
    ) as resolved_root:
        command = [
            "--backend",
            backend,
            "--output-root",
            str(resolved_root),
            "--plasma-surf-filename",
            args.plasma_surf_filename,
            "--stage2-bs-path",
            args.stage2_bs_path,
            "--nphi",
            str(args.nphi),
            "--ntheta",
            str(args.ntheta),
            "--mpol",
            str(args.mpol),
            "--ntor",
            str(args.ntor),
            "--vol-target",
            str(args.vol_target),
            "--iota-target",
            str(args.iota_target),
            "--num-tf-coils",
            str(getattr(args, "num_tf_coils", 20)),
            "--initial-step-scale",
            str(getattr(args, "initial_step_scale", 1.0)),
            "--initial-step-maxiter",
            str(getattr(args, "initial_step_maxiter", 0)),
            "--outer-maxls",
            str(getattr(args, "outer_maxls", TRACE_PARITY_OUTER_MAXLS)),
        ]
        warm_start_run_dir = getattr(args, "warm_start_run_dir", None)
        if warm_start_run_dir is not None:
            command.extend(["--warm-start-run-dir", str(warm_start_run_dir)])
        if int(args.maxiter) <= 0:
            command.append("--init-only")
        else:
            command.extend(["--maxiter", str(args.maxiter)])
        if backend == "jax":
            command.extend(["--optimizer-backend", args.optimizer_backend])
            resolved_seed_spec = (
                jax_runtime_seed_spec
                if jax_runtime_seed_spec is not None
                else getattr(args, "jax_runtime_seed_spec", None)
            )
            if resolved_seed_spec is not None:
                command.extend(["--jax-runtime-seed-spec", str(resolved_seed_spec)])
            if args.boozer_optimizer_backend is not None:
                command.extend(
                    [
                        "--boozer-optimizer-backend",
                        args.boozer_optimizer_backend,
                    ]
                )
        else:
            reference_optimizer_method = getattr(
                args,
                "reference_optimizer_method",
                "lbfgs",
            )
            if reference_optimizer_method != "lbfgs":
                command.extend(
                    [
                        "--reference-optimizer-method",
                        str(reference_optimizer_method),
                    ]
                )
        _append_optional_single_stage_flags(
            command,
            benchmark_mode=benchmark_mode,
            profile_target_lane=profile_target_lane,
            profile_target_lane_only=profile_target_lane_only,
            diagnose_target_lane_scaled_phase1=diagnose_target_lane_scaled_phase1,
            record_target_lane_invalid_state_events=(
                record_target_lane_invalid_state_events
            ),
            profile_target_lane_batch_size=getattr(
                args, "profile_target_lane_batch_size", None
            ),
            enable_compile_diagnostics=enable_compile_diagnostics,
            jax_profile_dir=getattr(args, "jax_profile_dir", None),
            experimental_target_lane_value_and_grad=(
                experimental_target_lane_value_and_grad
            ),
            disable_target_lane_success_filter=bool(
                getattr(args, "disable_target_lane_success_filter", False)
            ),
            record_objective_evaluation_trace=bool(
                getattr(args, "record_objective_evaluation_trace", False)
            ),
            target_lane_boozer_bfgs_tol=getattr(
                args, "target_lane_boozer_bfgs_tol", None
            ),
            target_lane_boozer_bfgs_maxiter=getattr(
                args, "target_lane_boozer_bfgs_maxiter", None
            ),
            target_lane_boozer_newton_tol=getattr(
                args, "target_lane_boozer_newton_tol", None
            ),
            target_lane_boozer_newton_maxiter=getattr(
                args, "target_lane_boozer_newton_maxiter", None
            ),
            replay_objective_evaluation_trace=replay_objective_evaluation_trace,
        )
        command.extend(
            [
                "--target-lane-accepted-step-sync",
                _resolve_target_lane_sync_policy(backend, args),
            ]
        )
        if args.equilibrium_path:
            command.extend(["--equilibrium-path", args.equilibrium_path])
        else:
            command.extend(["--equilibria-dir", args.equilibria_dir])

        start = time.perf_counter()
        run_python_script(
            script_path,
            command,
            env=repo_pythonpath_env(
                platform=effective_platform,
                disable_compilation_cache=(effective_platform == "cpu"),
                clear_backend_guardrails=(backend != "jax"),
                deterministic_gpu_reductions=deterministic_gpu_reductions,
            ),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        elapsed_s = time.perf_counter() - start

        results_json = find_single_file(resolved_root, "results.json")
        results = dict(load_json(results_json))
        payload = {
            "results": results,
            "surface_gamma": None,
            "elapsed_s": float(elapsed_s),
            "phase_timings": _extract_phase_timings(results),
            "run_dir": str(results_json.parent),
            "outer_optimizer_progress_json": str(
                results_json.parent / "outer_optimizer_progress.json"
            ),
        }
        if load_surface_gamma:
            surf_json = find_single_file(resolved_root, "surf_init.json")
            payload["surface_gamma"] = _load_surface_gamma_artifact(str(surf_json))
        return payload


def _compile_jax_runtime_seed_spec_from_run_dir(
    run_dir: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> Path:
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    return Path(
        single_stage_example.compile_single_stage_jax_runtime_seed_spec(
            str(run_dir),
            mpol=int(args.mpol),
            ntor=int(args.ntor),
            nphi=int(args.nphi),
            ntheta=int(args.ntheta),
            num_tf_coils=int(getattr(args, "num_tf_coils", 20)),
            output_path_or_run_dir=str(output_path),
        )
    )


def _reference_case_backend(args: argparse.Namespace) -> str:
    """Return the backend that gives the target-lane proof an apples-to-apples reference."""
    if (
        getattr(args, "jax_runtime_seed_spec", None) is not None
        or getattr(args, "warm_start_run_dir", None) is not None
    ):
        return "jax"
    return "cpu"


def _reference_case_benchmark_mode(
    args: argparse.Namespace,
    requested_benchmark_mode: bool,
) -> bool:
    """Return whether the reference can skip heavy artifacts without losing its seed."""
    return bool(requested_benchmark_mode and _reference_case_backend(args) == "jax")


def _should_compare_surface_geometry(
    args: argparse.Namespace,
    *,
    benchmark_mode: bool,
) -> bool:
    return bool(not benchmark_mode and int(args.maxiter) <= 0)


def _needs_shared_init_seed(
    args: argparse.Namespace, *, reference_backend: str
) -> bool:
    return bool(reference_backend == "cpu" and int(args.maxiter) > 0)


def _namespace_with_overrides(
    args: argparse.Namespace,
    **overrides: Any,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _run_single_stage_case_pair(
    args: argparse.Namespace,
    *,
    benchmark_mode: bool,
    reference_backend: str,
    reference_benchmark_mode: bool,
    case_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    compare_surface_geometry = _should_compare_surface_geometry(
        args,
        benchmark_mode=benchmark_mode,
    )
    seed_case = None
    same_candidate_replay_case = None
    target_args = args
    if reference_backend == "jax":
        jax_seed_spec = (
            Path(args.jax_runtime_seed_spec)
            if args.jax_runtime_seed_spec is not None
            else _compile_jax_runtime_seed_spec_from_run_dir(
                Path(args.warm_start_run_dir),
                case_root / "single_stage_jax_runtime_seed_spec.json",
                args,
            )
        )
        cpu_case = _run_single_stage_case(
            args,
            "jax",
            platform="cpu",
            benchmark_mode=reference_benchmark_mode,
            load_surface_gamma=compare_surface_geometry,
            output_root=case_root / "reference_outputs",
            jax_runtime_seed_spec=jax_seed_spec,
        )
    else:
        if _needs_shared_init_seed(args, reference_backend=reference_backend):
            seed_args = _namespace_with_overrides(args, maxiter=0)
            seed_case = _run_single_stage_case(
                seed_args,
                "cpu",
                platform="cpu",
                benchmark_mode=False,
                load_surface_gamma=False,
                output_root=case_root / "seed_outputs",
            )
            jax_seed_spec = _compile_jax_runtime_seed_spec_from_run_dir(
                Path(seed_case["run_dir"]),
                case_root / "single_stage_jax_runtime_seed_spec.json",
                args,
            )
            reference_args = _namespace_with_overrides(
                args,
                warm_start_run_dir=seed_case["run_dir"],
            )
            target_args = reference_args
        else:
            reference_args = args
        cpu_case = _run_single_stage_case(
            reference_args,
            "cpu",
            platform="cpu",
            benchmark_mode=reference_benchmark_mode,
            load_surface_gamma=compare_surface_geometry,
            output_root=case_root / "cpu_outputs",
        )
        if seed_case is None:
            jax_seed_spec = _compile_jax_runtime_seed_spec_from_run_dir(
                Path(cpu_case["run_dir"]),
                case_root / "single_stage_jax_runtime_seed_spec.json",
                args,
            )
    if (
        bool(getattr(args, "record_objective_evaluation_trace", False))
        and int(args.maxiter) > 0
    ):
        same_candidate_replay_case = _run_single_stage_case(
            target_args,
            "jax",
            platform=args.platform,
            benchmark_mode=benchmark_mode,
            load_surface_gamma=compare_surface_geometry,
            output_root=case_root / "target_same_candidate_replay_outputs",
            jax_runtime_seed_spec=jax_seed_spec,
            replay_objective_evaluation_trace=Path(
                cpu_case["outer_optimizer_progress_json"]
            ),
        )
    jax_case = _run_single_stage_case(
        target_args,
        "jax",
        platform=args.platform,
        benchmark_mode=benchmark_mode,
        load_surface_gamma=compare_surface_geometry,
        output_root=case_root / "target_outputs",
        jax_runtime_seed_spec=jax_seed_spec,
    )
    return cpu_case, jax_case, jax_seed_spec, seed_case, same_candidate_replay_case


def _load_surface_gamma_artifact(surface_json_path: str) -> np.ndarray:
    from simsopt._core.optimizable import load

    surface = load(surface_json_path)
    return np.asarray(surface.gamma(), dtype=float)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_surface_geometry_drift(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
    *,
    compare_surface_geometry: bool,
) -> tuple[float, float]:
    if not compare_surface_geometry:
        return 0.0, 0.0
    return max_pointwise_geometry_drift(
        jax_case["surface_gamma"],
        cpu_case["surface_gamma"],
    )


def _finite_required_result_keys(results: dict[str, Any]) -> dict[str, bool]:
    return {
        key: bool(np.isfinite(float(results.get(key, np.nan))))
        for key in _OUTER_LOOP_REQUIRED_RESULT_KEYS
    }


def _load_optimizer_state_trace_from_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    progress_path = Path(case["outer_optimizer_progress_json"])
    if not progress_path.exists():
        return []
    payload = load_json(progress_path)
    for event in reversed(payload.get("events", [])):
        result = event.get("result")
        if not result:
            continue
        trace = result.get("optimizer_state_trace")
        if trace:
            return list(trace)
    return []


def _load_objective_evaluation_events_from_case(
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    progress_path = Path(case["outer_optimizer_progress_json"])
    if not progress_path.exists():
        return []
    payload = load_json(progress_path)
    return [
        dict(event)
        for event in payload.get("events", [])
        if event.get("label") == "objective_evaluation"
    ]


def _summary_scalar(summary: dict[str, Any] | None) -> float | None:
    if summary is None or not bool(summary.get("finite", False)):
        return None
    value = summary.get("value")
    return None if value is None else float(value)


def _summary_vector(summary: dict[str, Any] | None) -> np.ndarray | None:
    if summary is None or not bool(summary.get("all_finite", False)):
        return None
    values = summary.get("values")
    if values is None:
        return None
    return np.asarray(values, dtype=float).reshape(-1)


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _scalar_close(left: float, right: float, *, rtol: float, atol: float) -> bool:
    return bool(abs(left - right) <= (atol + rtol * abs(right)))


def _compare_same_candidate_scalar(
    failures: list[str],
    *,
    field: str,
    cpu_value: float | None,
    jax_value: float | None,
    rtol: float = _SAME_CANDIDATE_SCALAR_RTOL,
    atol: float = _SAME_CANDIDATE_SCALAR_ATOL,
) -> float:
    if cpu_value is None or jax_value is None:
        failures.append(f"{field} missing finite CPU/JAX values.")
        return float("inf")
    diff = abs(float(jax_value) - float(cpu_value))
    if not _scalar_close(float(jax_value), float(cpu_value), rtol=rtol, atol=atol):
        failures.append(
            f"{field} mismatch: cpu={float(cpu_value):.16e}, "
            f"jax={float(jax_value):.16e}, abs_diff={diff:.3e}."
        )
    return diff


def _compare_same_candidate_vector(
    failures: list[str],
    *,
    field: str,
    cpu_vector: np.ndarray | None,
    jax_vector: np.ndarray | None,
    rtol: float = _SAME_CANDIDATE_GRADIENT_RTOL,
    atol: float = _SAME_CANDIDATE_GRADIENT_ATOL,
) -> float:
    if cpu_vector is None or jax_vector is None:
        failures.append(f"{field} missing finite CPU/JAX vectors.")
        return float("inf")
    diff = _max_abs_diff(jax_vector, cpu_vector)
    reference = 0.0 if cpu_vector.size == 0 else float(np.max(np.abs(cpu_vector)))
    if diff > (atol + rtol * reference):
        failures.append(
            f"{field} mismatch: max_abs_diff={diff:.3e}, reference={reference:.3e}."
        )
    return diff


def _compare_same_candidate_hardware(
    failures: list[str],
    *,
    cpu_status: dict[str, Any] | None,
    jax_status: dict[str, Any] | None,
) -> float:
    if cpu_status is None or jax_status is None:
        if cpu_status is not jax_status:
            failures.append("hardware_status presence mismatch.")
        return 0.0
    if bool(cpu_status.get("success")) != bool(jax_status.get("success")):
        failures.append(
            "hardware_status success mismatch: "
            f"cpu={cpu_status.get('success')}, jax={jax_status.get('success')}."
        )
    if list(cpu_status.get("violation_keys", [])) != list(
        jax_status.get("violation_keys", [])
    ):
        failures.append(
            "hardware_status violation_keys mismatch: "
            f"cpu={cpu_status.get('violation_keys')}, "
            f"jax={jax_status.get('violation_keys')}."
        )
    max_diff = 0.0
    for key in _SAME_CANDIDATE_HARDWARE_KEYS:
        if key not in cpu_status or key not in jax_status:
            continue
        diff = _compare_same_candidate_scalar(
            failures,
            field=f"hardware_status.{key}",
            cpu_value=float(cpu_status[key]),
            jax_value=float(jax_status[key]),
            rtol=1e-8,
            atol=1e-10,
        )
        max_diff = max(max_diff, diff)
    cpu_margins = cpu_status.get("threshold_margins", {})
    jax_margins = jax_status.get("threshold_margins", {})
    for key in _SAME_CANDIDATE_HARDWARE_MARGIN_KEYS:
        if key not in cpu_margins and key not in jax_margins:
            continue
        diff = _compare_same_candidate_scalar(
            failures,
            field=f"hardware_status.threshold_margins.{key}",
            cpu_value=cpu_margins.get(key),
            jax_value=jax_margins.get(key),
            rtol=1e-8,
            atol=1e-10,
        )
        max_diff = max(max_diff, diff)
    return max_diff


def _compare_same_candidate_failure(
    failures: list[str],
    *,
    cpu_failure: dict[str, Any] | None,
    jax_failure: dict[str, Any] | None,
) -> float:
    if cpu_failure is None or jax_failure is None:
        if cpu_failure is not jax_failure:
            failures.append("candidate_failure presence mismatch.")
        return 0.0
    for key in _SAME_CANDIDATE_FAILURE_EXACT_KEYS:
        if cpu_failure.get(key) != jax_failure.get(key):
            failures.append(
                f"candidate_failure.{key} mismatch: "
                f"cpu={cpu_failure.get(key)!r}, jax={jax_failure.get(key)!r}."
            )
    max_diff = 0.0
    for key in _SAME_CANDIDATE_FAILURE_SCALAR_KEYS:
        diff = _compare_same_candidate_scalar(
            failures,
            field=f"candidate_failure.{key}",
            cpu_value=float(cpu_failure[key]),
            jax_value=float(jax_failure[key]),
        )
        max_diff = max(max_diff, diff)
    if (
        cpu_failure.get("reject_class") == "solver"
        and jax_failure.get("reject_class") == "solver"
    ):
        max_diff = max(
            max_diff,
            _compare_same_candidate_scalar(
                failures,
                field="candidate_failure.residual_inf",
                cpu_value=float(cpu_failure["residual_inf"]),
                jax_value=float(jax_failure["residual_inf"]),
            ),
        )
    return max_diff


def _compare_same_candidate_exact_event_field(
    failures: list[str],
    *,
    field: str,
    cpu_event: dict[str, Any],
    jax_event: dict[str, Any],
) -> None:
    if cpu_event.get(field) != jax_event.get(field):
        failures.append(
            f"{field} mismatch: "
            f"cpu={cpu_event.get(field)!r}, jax={jax_event.get(field)!r}."
        )


def _first_boozer_solver_summary(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        metadata = event.get("boozer_solver_metadata")
        if metadata is not None:
            return dict(metadata)
    return None


def _compare_same_candidate_boozer_solver_metadata(
    failures: list[str],
    *,
    cpu_metadata: dict[str, Any] | None,
    jax_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if cpu_metadata is None or jax_metadata is None:
        if cpu_metadata is not jax_metadata:
            failures.append("boozer_solver_metadata presence mismatch.")
        return {
            "max_abs_diff": 0.0,
            "scipy_callback_trace_max_abs_diff": 0.0,
            "first_scipy_callback_split": None,
        }

    for key in _SAME_CANDIDATE_BOOZER_METADATA_EXACT_KEYS:
        if cpu_metadata.get(key) != jax_metadata.get(key):
            failures.append(
                f"boozer_solver_metadata.{key} mismatch: "
                f"cpu={cpu_metadata.get(key)!r}, jax={jax_metadata.get(key)!r}."
            )
    _compare_same_candidate_scipy_call_contract(
        failures,
        field="boozer_solver_metadata.pre_newton_scipy_call_contract",
        cpu_contract=cpu_metadata.get("pre_newton_scipy_call_contract"),
        jax_contract=jax_metadata.get("pre_newton_scipy_call_contract"),
    )
    max_diff = _compare_same_candidate_scipy_initial_call(
        failures,
        field="boozer_solver_metadata.pre_newton_scipy_initial_call",
        cpu_initial_call=cpu_metadata.get("pre_newton_scipy_initial_call"),
        jax_initial_call=jax_metadata.get("pre_newton_scipy_initial_call"),
    )
    callback_trace_summary = _compare_same_candidate_scipy_callback_trace(
        failures,
        field="boozer_solver_metadata.pre_newton_scipy_callback_trace",
        cpu_trace=cpu_metadata.get("pre_newton_scipy_callback_trace"),
        jax_trace=jax_metadata.get("pre_newton_scipy_callback_trace"),
    )
    max_diff = max(max_diff, callback_trace_summary["max_abs_diff"])
    for key in _SAME_CANDIDATE_BOOZER_METADATA_SHAPE_KEYS:
        cpu_shape = cpu_metadata.get(key)
        jax_shape = jax_metadata.get(key)
        if cpu_shape is None and jax_shape is None:
            continue
        if list(cpu_shape or []) != list(jax_shape or []):
            failures.append(
                f"boozer_solver_metadata.{key} mismatch: "
                f"cpu={cpu_shape!r}, jax={jax_shape!r}."
            )

    for key in _SAME_CANDIDATE_BOOZER_METADATA_NUMERIC_KEYS:
        cpu_value = cpu_metadata.get(key)
        jax_value = jax_metadata.get(key)
        if cpu_value is None and jax_value is None:
            continue
        diff = _compare_same_candidate_scalar(
            failures,
            field=f"boozer_solver_metadata.{key}",
            cpu_value=None if cpu_value is None else float(cpu_value),
            jax_value=None if jax_value is None else float(jax_value),
            rtol=1e-8,
            atol=1e-12,
        )
        max_diff = max(max_diff, diff)
    return {
        "max_abs_diff": max_diff,
        "scipy_callback_trace_max_abs_diff": callback_trace_summary["max_abs_diff"],
        "first_scipy_callback_split": callback_trace_summary["first_split"],
    }


def _compare_same_candidate_scipy_call_contract(
    failures: list[str],
    *,
    field: str,
    cpu_contract: dict[str, Any] | None,
    jax_contract: dict[str, Any] | None,
) -> None:
    if cpu_contract is None and jax_contract is None:
        return
    if cpu_contract is None or jax_contract is None:
        failures.append(f"{field} presence mismatch.")
        return
    exact_keys = (
        "semantic_method",
        "scipy_method",
        "scipy_options",
        "callback",
        "success",
        "status",
        "message",
        "nit",
        "nfev",
        "njev",
    )
    for key in exact_keys:
        if cpu_contract.get(key) != jax_contract.get(key):
            failures.append(
                f"{field}.{key} mismatch: "
                f"cpu={cpu_contract.get(key)!r}, "
                f"jax={jax_contract.get(key)!r}."
            )


def _compare_same_candidate_scipy_initial_call(
    failures: list[str],
    *,
    field: str,
    cpu_initial_call: dict[str, Any] | None,
    jax_initial_call: dict[str, Any] | None,
) -> float:
    if cpu_initial_call is None and jax_initial_call is None:
        return 0.0
    if cpu_initial_call is None or jax_initial_call is None:
        failures.append(f"{field} presence mismatch.")
        return float("inf")
    max_diff = _compare_same_candidate_vector(
        failures,
        field=f"{field}.decision_vector",
        cpu_vector=_summary_vector(cpu_initial_call.get("decision_vector")),
        jax_vector=_summary_vector(jax_initial_call.get("decision_vector")),
        rtol=0.0,
        atol=0.0,
    )
    max_diff = max(
        max_diff,
        _compare_same_candidate_scalar(
            failures,
            field=f"{field}.fun",
            cpu_value=_summary_scalar(cpu_initial_call.get("fun")),
            jax_value=_summary_scalar(jax_initial_call.get("fun")),
        ),
    )
    max_diff = max(
        max_diff,
        _compare_same_candidate_vector(
            failures,
            field=f"{field}.gradient",
            cpu_vector=_summary_vector(cpu_initial_call.get("gradient")),
            jax_vector=_summary_vector(jax_initial_call.get("gradient")),
        ),
    )
    return max_diff


def _same_candidate_scipy_callback_split(
    *,
    field: str,
    callback_index: int,
    cpu_entry: dict[str, Any],
    jax_entry: dict[str, Any],
    max_abs_diff: float,
) -> dict[str, Any]:
    return {
        "field": field,
        "callback_index": callback_index,
        "cpu_evaluation_index": cpu_entry.get("evaluation_index"),
        "jax_evaluation_index": jax_entry.get("evaluation_index"),
        "max_abs_diff": max_abs_diff,
    }


def _compare_same_candidate_scipy_callback_trace(
    failures: list[str],
    *,
    field: str,
    cpu_trace: list[dict[str, Any]] | None,
    jax_trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if cpu_trace is None and jax_trace is None:
        return {"max_abs_diff": 0.0, "first_split": None}
    if cpu_trace is None or jax_trace is None:
        failures.append(f"{field} presence mismatch.")
        return {
            "max_abs_diff": float("inf"),
            "first_split": {"field": field, "reason": "presence mismatch"},
        }

    max_diff = 0.0
    first_split = None
    if len(cpu_trace) != len(jax_trace):
        failures.append(
            f"{field} length mismatch: cpu={len(cpu_trace)}, jax={len(jax_trace)}."
        )
    for index, (cpu_entry, jax_entry) in enumerate(
        zip(cpu_trace, jax_trace),
        start=1,
    ):
        entry_field = f"{field}[{index}]"
        if cpu_entry.get("evaluation_index") != jax_entry.get("evaluation_index"):
            failures.append(
                f"{entry_field}.evaluation_index mismatch: "
                f"cpu={cpu_entry.get('evaluation_index')!r}, "
                f"jax={jax_entry.get('evaluation_index')!r}."
            )
            if first_split is None:
                first_split = _same_candidate_scipy_callback_split(
                    field="evaluation_index",
                    callback_index=index,
                    cpu_entry=cpu_entry,
                    jax_entry=jax_entry,
                    max_abs_diff=0.0,
                )
        cpu_x = _summary_vector(cpu_entry.get("decision_vector"))
        jax_x = _summary_vector(jax_entry.get("decision_vector"))
        decision_diff = _compare_same_candidate_vector(
            failures,
            field=f"{entry_field}.decision_vector",
            cpu_vector=cpu_x,
            jax_vector=jax_x,
            rtol=0.0,
            atol=0.0,
        )
        max_diff = max(max_diff, decision_diff)
        if first_split is None and decision_diff > 0.0:
            first_split = _same_candidate_scipy_callback_split(
                field="decision_vector",
                callback_index=index,
                cpu_entry=cpu_entry,
                jax_entry=jax_entry,
                max_abs_diff=decision_diff,
            )

        cpu_fun = _summary_scalar(cpu_entry.get("fun"))
        jax_fun = _summary_scalar(jax_entry.get("fun"))
        fun_diff = _compare_same_candidate_scalar(
            failures,
            field=f"{entry_field}.fun",
            cpu_value=cpu_fun,
            jax_value=jax_fun,
        )
        max_diff = max(max_diff, fun_diff)
        if (
            first_split is None
            and cpu_fun is not None
            and jax_fun is not None
            and not _scalar_close(
                jax_fun,
                cpu_fun,
                rtol=_SAME_CANDIDATE_SCALAR_RTOL,
                atol=_SAME_CANDIDATE_SCALAR_ATOL,
            )
        ):
            first_split = _same_candidate_scipy_callback_split(
                field="fun",
                callback_index=index,
                cpu_entry=cpu_entry,
                jax_entry=jax_entry,
                max_abs_diff=fun_diff,
            )

        cpu_gradient = _summary_vector(cpu_entry.get("gradient"))
        jax_gradient = _summary_vector(jax_entry.get("gradient"))
        gradient_diff = _compare_same_candidate_vector(
            failures,
            field=f"{entry_field}.gradient",
            cpu_vector=cpu_gradient,
            jax_vector=jax_gradient,
        )
        max_diff = max(max_diff, gradient_diff)
        gradient_reference = (
            None
            if cpu_gradient is None
            else (
                0.0 if cpu_gradient.size == 0 else float(np.max(np.abs(cpu_gradient)))
            )
        )
        if (
            first_split is None
            and gradient_reference is not None
            and gradient_diff
            > (
                _SAME_CANDIDATE_GRADIENT_ATOL
                + _SAME_CANDIDATE_GRADIENT_RTOL * gradient_reference
            )
        ):
            first_split = _same_candidate_scipy_callback_split(
                field="gradient",
                callback_index=index,
                cpu_entry=cpu_entry,
                jax_entry=jax_entry,
                max_abs_diff=gradient_diff,
            )
    if first_split is None and len(cpu_trace) != len(jax_trace):
        first_split = {
            "field": field,
            "reason": "length mismatch",
            "cpu_length": len(cpu_trace),
            "jax_length": len(jax_trace),
        }
    return {"max_abs_diff": max_diff, "first_split": first_split}


def _compare_same_candidate_objective_components(
    failures: list[str],
    *,
    cpu_components: dict[str, Any] | None,
    jax_components: dict[str, Any] | None,
    pair_index: int,
    line_search_evaluation: Any,
) -> dict[str, Any]:
    if cpu_components is None or jax_components is None:
        if cpu_components is not jax_components:
            failures.append("objective_components presence mismatch.")
        return {
            "max_slice_objective_abs_diff": 0.0,
            "max_slice_gradient_abs_diff": 0.0,
            "max_slice_objective_owner": None,
            "max_slice_gradient_owner": None,
        }
    cpu_names = set(cpu_components)
    jax_names = set(jax_components)
    if cpu_names != jax_names:
        failures.append(
            "objective_components key mismatch: "
            f"cpu={sorted(cpu_names)}, jax={sorted(jax_names)}."
        )

    max_objective = 0.0
    max_gradient = 0.0
    objective_owner = None
    gradient_owner = None
    for name in sorted(cpu_names & jax_names):
        cpu_component = cpu_components[name]
        jax_component = jax_components[name]
        objective_diff = _compare_same_candidate_scalar(
            failures,
            field=f"objective_components.{name}.weighted_objective",
            cpu_value=_summary_scalar(cpu_component.get("weighted_objective")),
            jax_value=_summary_scalar(jax_component.get("weighted_objective")),
        )
        gradient_diff = _compare_same_candidate_vector(
            failures,
            field=f"objective_components.{name}.weighted_gradient",
            cpu_vector=_summary_vector(cpu_component.get("weighted_gradient")),
            jax_vector=_summary_vector(jax_component.get("weighted_gradient")),
        )
        if objective_diff > max_objective:
            max_objective = objective_diff
            objective_owner = name
        if gradient_diff > max_gradient:
            max_gradient = gradient_diff
            gradient_owner = name
    return {
        "max_slice_objective_abs_diff": max_objective,
        "max_slice_gradient_abs_diff": max_gradient,
        "max_slice_objective_owner": objective_owner,
        "max_slice_gradient_owner": gradient_owner,
        "max_slice_pair_index": pair_index
        if objective_owner or gradient_owner
        else None,
        "max_slice_line_search_evaluation": line_search_evaluation
        if objective_owner or gradient_owner
        else None,
    }


def _nested_payload_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if value is None:
            return None
        value = value.get(key)
    return value


def _diagnostic_scalar_abs_diff(
    cpu_summary: dict[str, Any] | None,
    jax_summary: dict[str, Any] | None,
) -> float:
    if cpu_summary is None and jax_summary is None:
        return 0.0
    return _path_scalar_abs_diff(cpu_summary, jax_summary)


def _diagnostic_vector_abs_diff(
    cpu_summary: dict[str, Any] | None,
    jax_summary: dict[str, Any] | None,
) -> float:
    if cpu_summary is None and jax_summary is None:
        return 0.0
    return _path_vector_abs_diff(cpu_summary, jax_summary)


def _iota_decomposition_layer_diverged(
    layer_diff: float, layer_reference: float
) -> bool:
    return bool(
        layer_diff
        > (
            _IOTA_DECOMPOSITION_DIAGNOSTIC_ATOL
            + _IOTA_DECOMPOSITION_DIAGNOSTIC_RTOL * layer_reference
        )
    )


def _summary_reference_abs(summary: dict[str, Any] | None) -> float:
    vector = _summary_vector(summary)
    if vector is not None:
        return 0.0 if vector.size == 0 else float(np.max(np.abs(vector)))
    scalar = _summary_scalar(summary)
    return 0.0 if scalar is None else abs(float(scalar))


def _layer_decomposition_summary(
    *,
    recorded: bool,
    max_abs_diff: float = 0.0,
    max_layer: str | None = None,
    first_divergent_layer: str | None = None,
    pair_index: int | None = None,
    line_search_evaluation: Any = None,
    layer_diffs: dict[str, float] | None = None,
    layer_references: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "recorded": recorded,
        "max_abs_diff": max_abs_diff,
        "max_layer": max_layer,
        "first_divergent_layer": first_divergent_layer,
        "pair_index": pair_index,
        "line_search_evaluation": line_search_evaluation,
        "layer_diffs": {} if layer_diffs is None else layer_diffs,
        "layer_references": {} if layer_references is None else layer_references,
    }


def _compare_same_candidate_layer_decomposition(
    failures: list[str],
    *,
    field_name: str,
    layer_fields: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...],
    cpu_decomposition: dict[str, Any] | None,
    jax_decomposition: dict[str, Any] | None,
    pair_index: int,
    line_search_evaluation: Any,
) -> dict[str, Any]:
    if cpu_decomposition is None and jax_decomposition is None:
        return _layer_decomposition_summary(recorded=False)
    if cpu_decomposition is None or jax_decomposition is None:
        failures.append(f"{field_name} presence mismatch.")
        return _layer_decomposition_summary(
            recorded=False,
            max_abs_diff=float("inf"),
            max_layer="presence",
            first_divergent_layer="presence",
            pair_index=pair_index,
            line_search_evaluation=line_search_evaluation,
            layer_diffs={"presence": float("inf")},
            layer_references={"presence": 0.0},
        )

    layer_diffs = {}
    layer_references = {}
    max_abs_diff = 0.0
    max_layer = None
    first_divergent_layer = None
    for layer, fields in layer_fields:
        layer_diff = 0.0
        layer_reference = 0.0
        for kind, path in fields:
            cpu_summary = _nested_payload_value(cpu_decomposition, path)
            jax_summary = _nested_payload_value(jax_decomposition, path)
            if kind == "scalar":
                field_diff = _diagnostic_scalar_abs_diff(cpu_summary, jax_summary)
            else:
                field_diff = _diagnostic_vector_abs_diff(cpu_summary, jax_summary)
            layer_diff = max(layer_diff, field_diff)
            layer_reference = max(layer_reference, _summary_reference_abs(cpu_summary))
        layer_diffs[layer] = layer_diff
        layer_references[layer] = layer_reference
        if layer_diff > max_abs_diff:
            max_abs_diff = layer_diff
            max_layer = layer
        if first_divergent_layer is None and _iota_decomposition_layer_diverged(
            layer_diff,
            layer_reference,
        ):
            first_divergent_layer = layer

    return _layer_decomposition_summary(
        recorded=True,
        max_abs_diff=max_abs_diff,
        max_layer=max_layer,
        first_divergent_layer=first_divergent_layer,
        pair_index=pair_index if max_layer is not None else None,
        line_search_evaluation=line_search_evaluation
        if max_layer is not None
        else None,
        layer_diffs=layer_diffs,
        layer_references=layer_references,
    )


def _compare_same_candidate_iota_decomposition(
    failures: list[str],
    *,
    cpu_decomposition: dict[str, Any] | None,
    jax_decomposition: dict[str, Any] | None,
    pair_index: int,
    line_search_evaluation: Any,
) -> dict[str, Any]:
    return _compare_same_candidate_layer_decomposition(
        failures,
        field_name="iota_penalty_decomposition",
        layer_fields=_IOTA_DECOMPOSITION_LAYER_FIELDS,
        cpu_decomposition=cpu_decomposition,
        jax_decomposition=jax_decomposition,
        pair_index=pair_index,
        line_search_evaluation=line_search_evaluation,
    )


def _compare_same_candidate_boozer_solve_decomposition(
    failures: list[str],
    *,
    cpu_decomposition: dict[str, Any] | None,
    jax_decomposition: dict[str, Any] | None,
    pair_index: int,
    line_search_evaluation: Any,
) -> dict[str, Any]:
    return _compare_same_candidate_layer_decomposition(
        failures,
        field_name="boozer_solve_decomposition",
        layer_fields=_BOOZER_SOLVE_DECOMPOSITION_LAYER_FIELDS,
        cpu_decomposition=cpu_decomposition,
        jax_decomposition=jax_decomposition,
        pair_index=pair_index,
        line_search_evaluation=line_search_evaluation,
    )


def _update_parity_bug_census(
    census: dict[str, dict[str, Any]],
    *,
    family: str,
    summary: dict[str, Any],
) -> None:
    pair_index = summary["pair_index"]
    line_search_evaluation = summary["line_search_evaluation"]
    for layer, diff in summary["layer_diffs"].items():
        layer_key = f"{family}.{layer}"
        reference = summary["layer_references"].get(layer, 0.0)
        previous = census.get(layer_key)
        if previous is None or float(diff) > float(previous["max_abs_diff"]):
            census[layer_key] = {
                "family": family,
                "layer": layer,
                "max_abs_diff": diff,
                "reference_abs": reference,
                "pair_index": pair_index,
                "line_search_evaluation": line_search_evaluation,
                "diverged": _iota_decomposition_layer_diverged(diff, reference),
            }


def _finalize_parity_bug_census(
    census: dict[str, dict[str, Any]],
    *,
    first_divergence: dict[str, Any] | None,
) -> dict[str, Any]:
    layers = list(census.values())
    divergent_layers = [
        dict(entry)
        for entry in sorted(
            layers,
            key=lambda item: float(item["max_abs_diff"]),
            reverse=True,
        )
        if bool(entry["diverged"])
    ]
    return {
        "status": "recorded" if layers else "not-recorded",
        "first_divergence": first_divergence,
        "divergent_layer_count": len(divergent_layers),
        "divergent_layers": divergent_layers,
        "max_layer_diffs": {
            f"{entry['family']}.{entry['layer']}": entry["max_abs_diff"]
            for entry in layers
        },
    }


def _empirical_severity_context(
    layer_full_name: str,
    max_abs: float,
    severity_context: dict[str, Any] | None,
) -> str:
    """Return a parenthesized severity tag for inclusion in failure messages.

    Computes the drift / threshold ratio against an empirical baseline and
    classifies the result per
    `docs/parity_dual_mode_contract_2026-05-08.md` §11.5:

    - ``drift / threshold <= 1.0``: ``marginal``
    - ``1.0 < drift / threshold <= 10.0``: ``moderate``
    - ``drift / threshold > 10.0``: ``severe``

    The baseline is read from
    ``severity_context["per_layer"][layer_full_name]``. Required fields are
    ``baseline_max`` (float) and ``safety_factor`` (float, default ``5.0``).
    The reporting threshold is ``safety_factor * baseline_max``.

    Optional fields enrich the message when present:

    - ``corpus_p95`` — corpus p95 of ``max_abs_diff``
    - ``sample_size`` — number of corpus artifacts contributing
    - ``source_artifacts`` — list/iterable of corpus artifacts

    Returns an empty string when ``severity_context`` is ``None``, when its
    ``per_layer`` mapping is missing/empty (the ``INSUFFICIENT_SAMPLES``
    state pre-corpus), when the requested layer is absent from
    ``per_layer``, or when ``baseline_max``/``safety_factor`` are missing
    or zero. Callers therefore never need a guard around this helper.
    """
    if severity_context is None:
        return ""
    per_layer = severity_context.get("per_layer")
    if not isinstance(per_layer, dict) or not per_layer:
        return ""
    layer_entry = per_layer.get(layer_full_name)
    if not isinstance(layer_entry, dict):
        return ""
    baseline_raw = layer_entry.get("baseline_max")
    safety_raw = layer_entry.get("safety_factor", 5.0)
    if baseline_raw is None or safety_raw is None:
        return ""
    try:
        baseline_max = float(baseline_raw)
        safety_factor = float(safety_raw)
    except (TypeError, ValueError):
        return ""
    if baseline_max == 0.0 or safety_factor == 0.0:
        return ""
    threshold = safety_factor * baseline_max
    if threshold == 0.0:
        return ""
    ratio = float(max_abs) / threshold
    if ratio > 10.0:
        severity = "SEVERE"
    elif ratio > 1.0:
        severity = "moderate"
    else:
        severity = "marginal"
    parenthetical_bits: list[str] = []
    parenthetical_bits.append(f"{safety_factor:g}× safety factor")
    corpus_p95 = layer_entry.get("corpus_p95")
    if corpus_p95 is not None:
        try:
            corpus_p95_value = float(corpus_p95)
        except (TypeError, ValueError):
            corpus_p95_value = None
        if corpus_p95_value is not None:
            parenthetical_bits.append(f"corpus p95={corpus_p95_value:.2e}")
    sample_size = layer_entry.get("sample_size")
    if sample_size is None:
        sample_size = layer_entry.get("source_artifacts")
        if sample_size is not None:
            try:
                sample_size = len(sample_size)
            except TypeError:
                sample_size = None
    if isinstance(sample_size, bool):
        sample_size = None
    if isinstance(sample_size, int) and sample_size > 0:
        artifact_word = "artifact" if sample_size == 1 else "artifacts"
        parenthetical_bits.append(f"across {sample_size} passing {artifact_word}")
    parenthetical = ", ".join(parenthetical_bits)
    return (
        f" [{severity}: drift is {ratio:g}× empirical baseline of "
        f"{baseline_max:.2e} ({parenthetical})]"
    )


def _pre_newton_census_gate_failures(
    parity_bug_census: dict[str, Any] | None,
    *,
    severity_context: dict[str, Any] | None = None,
) -> list[str]:
    """Hard-gate: any boozer_solve.pre_newton_* divergent layer fails.

    When ``severity_context`` is provided (typically the
    ``PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]`` dict
    or a compatible structure), failure messages are augmented with
    empirical-baseline drift context (e.g. ``"drift is 100× empirical
    baseline of 4.5e-11"``). The augmented context is REPORTING ONLY —
    the gate's pass/fail decision is unchanged from the prior strict-only
    behavior. When ``severity_context`` is ``None`` or its ``per_layer``
    dict is empty / missing (the ``INSUFFICIENT_SAMPLES`` state pre-corpus),
    behavior is identical to the prior strict-only gate.

    See ``docs/parity_dual_mode_contract_2026-05-08.md`` §2.4 and §11.5
    for the contract this helper implements.
    """
    if not parity_bug_census:
        return []
    failures = []
    for entry in parity_bug_census.get("divergent_layers", []):
        family = entry.get("family")
        layer = str(entry.get("layer", ""))
        if family != "boozer_solve" or not layer.startswith("pre_newton"):
            continue
        max_abs_raw = entry.get("max_abs_diff")
        try:
            max_abs_value = float(max_abs_raw) if max_abs_raw is not None else 0.0
        except (TypeError, ValueError):
            max_abs_value = 0.0
        severity_tag = _empirical_severity_context(
            f"{family}.{layer}",
            max_abs_value,
            severity_context,
        )
        failures.append(
            "Parity bug census reported divergent "
            f"{family}.{layer}: max_abs_diff={max_abs_raw} "
            f"at pair {entry.get('pair_index')} "
            f"(line-search eval {entry.get('line_search_evaluation')})"
            f"{severity_tag}."
        )
    return failures


def _same_candidate_replay_gate_failures(
    same_candidate_replay: dict[str, Any],
) -> list[str]:
    failures = []
    if same_candidate_replay["status"] != "pass":
        first_failure = same_candidate_replay.get("first_failure_event")
        if first_failure is None:
            failures.append(
                "Same-candidate objective replay comparison did not pass: "
                f"status={same_candidate_replay['status']}."
            )
        else:
            failures.append(
                "Same-candidate objective replay comparison failed at "
                f"pair {first_failure['pair_index']} "
                f"(iteration {first_failure['accepted_iteration_target']}, "
                f"line-search eval {first_failure['line_search_evaluation']})."
            )
    parity_bug_census = same_candidate_replay.get("parity_bug_census")
    if not parity_bug_census or parity_bug_census.get("status") != "recorded":
        failures.append(
            "Same-candidate objective replay did not record a parity bug census."
        )
    failures.extend(_pre_newton_census_gate_failures(parity_bug_census))
    return failures


def compare_same_candidate_objective_replay(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
    *,
    require_exact_candidates: bool = False,
    strict_solver_contract: bool = False,
) -> dict[str, Any]:
    """Compare paired CPU/JAX objective-evaluation trace events at identical x."""
    cpu_events = _load_objective_evaluation_events_from_case(cpu_case)
    jax_events = _load_objective_evaluation_events_from_case(jax_case)
    if not cpu_events or not jax_events:
        return {
            "status": "not-recorded",
            "cpu_event_count": len(cpu_events),
            "jax_event_count": len(jax_events),
            "same_candidate_event_count": 0,
            "require_exact_candidates": bool(require_exact_candidates),
            "strict_solver_contract": bool(strict_solver_contract),
            "solver_contract_diagnostics": [],
            "failures": [],
        }
    failures: list[str] = []
    max_candidate_abs_diff = 0.0
    max_objective_abs_diff = 0.0
    max_gradient_abs_diff = 0.0
    max_hardware_abs_diff = 0.0
    max_failure_abs_diff = 0.0
    max_boozer_metadata_abs_diff = 0.0
    max_slice_objective_abs_diff = 0.0
    max_slice_gradient_abs_diff = 0.0
    max_slice_objective_owner = None
    max_slice_gradient_owner = None
    max_slice_pair_index = None
    max_slice_line_search_evaluation = None
    max_iota_decomposition_abs_diff = 0.0
    max_iota_decomposition_layer = None
    max_iota_decomposition_pair_index = None
    max_iota_decomposition_line_search_evaluation = None
    first_iota_decomposition_divergence = None
    max_iota_decomposition_layer_diffs = {}
    max_boozer_solve_decomposition_abs_diff = 0.0
    max_boozer_solve_decomposition_layer = None
    max_boozer_solve_decomposition_pair_index = None
    max_boozer_solve_decomposition_line_search_evaluation = None
    first_boozer_solve_decomposition_divergence = None
    max_boozer_solve_decomposition_layer_diffs = {}
    max_boozer_scipy_callback_abs_diff = 0.0
    first_boozer_scipy_callback_split = None
    parity_bug_census_layers: dict[str, dict[str, Any]] = {}
    first_parity_bug_census_divergence = None
    solver_contract_diagnostics: list[str] = []
    same_candidate_event_count = 0
    first_failure_event = None
    candidate_x_abs_tol = 0.0 if require_exact_candidates else _SAME_CANDIDATE_X_ATOL
    if require_exact_candidates and len(cpu_events) != len(jax_events):
        failures.append(
            "Exact objective replay event-count mismatch: "
            f"cpu={len(cpu_events)}, jax={len(jax_events)}."
        )
    for pair_index, (cpu_event, jax_event) in enumerate(
        zip(cpu_events, jax_events),
        start=1,
    ):
        cpu_x = _summary_vector(cpu_event.get("candidate_optimizer_dofs"))
        jax_x = _summary_vector(jax_event.get("candidate_optimizer_dofs"))
        if cpu_x is None or jax_x is None:
            continue
        candidate_abs_diff = _max_abs_diff(jax_x, cpu_x)
        max_candidate_abs_diff = max(max_candidate_abs_diff, candidate_abs_diff)
        if candidate_abs_diff > candidate_x_abs_tol:
            if require_exact_candidates:
                event_failures = [
                    "candidate_optimizer_dofs mismatch under exact replay: "
                    f"max_abs_diff={candidate_abs_diff:.3e}."
                ]
                if first_failure_event is None:
                    first_failure_event = {
                        "pair_index": pair_index,
                        "cpu_event_index": cpu_event.get("event_index"),
                        "jax_event_index": jax_event.get("event_index"),
                        "accepted_iteration_target": cpu_event.get(
                            "accepted_iteration_target"
                        ),
                        "line_search_evaluation": cpu_event.get(
                            "line_search_evaluation"
                        ),
                        "candidate_abs_diff": candidate_abs_diff,
                        "failures": list(event_failures),
                    }
                failures.extend(
                    f"pair {pair_index}: {failure}" for failure in event_failures
                )
            continue
        same_candidate_event_count += 1
        event_failures: list[str] = []
        _compare_same_candidate_exact_event_field(
            event_failures,
            field="native_gradient_used",
            cpu_event=cpu_event,
            jax_event=jax_event,
        )
        _compare_same_candidate_exact_event_field(
            event_failures,
            field="solver_success",
            cpu_event=cpu_event,
            jax_event=jax_event,
        )
        solver_contract_failures: list[str] = []
        boozer_metadata_summary = _compare_same_candidate_boozer_solver_metadata(
            solver_contract_failures,
            cpu_metadata=cpu_event.get("boozer_solver_metadata"),
            jax_metadata=jax_event.get("boozer_solver_metadata"),
        )
        if strict_solver_contract:
            event_failures.extend(solver_contract_failures)
        else:
            solver_contract_diagnostics.extend(
                f"pair {pair_index}: {failure}" for failure in solver_contract_failures
            )
        max_boozer_metadata_abs_diff = max(
            max_boozer_metadata_abs_diff,
            boozer_metadata_summary["max_abs_diff"],
        )
        max_boozer_scipy_callback_abs_diff = max(
            max_boozer_scipy_callback_abs_diff,
            boozer_metadata_summary["scipy_callback_trace_max_abs_diff"],
        )
        if (
            first_boozer_scipy_callback_split is None
            and boozer_metadata_summary["first_scipy_callback_split"] is not None
        ):
            first_boozer_scipy_callback_split = {
                "pair_index": pair_index,
                "cpu_event_index": cpu_event.get("event_index"),
                "jax_event_index": jax_event.get("event_index"),
                "accepted_iteration_target": cpu_event.get("accepted_iteration_target"),
                "line_search_evaluation": cpu_event.get("line_search_evaluation"),
                **boozer_metadata_summary["first_scipy_callback_split"],
            }
        compare_native_gradient_layers = bool(
            cpu_event.get("native_gradient_used")
        ) and bool(jax_event.get("native_gradient_used"))
        cpu_boozer_solve_decomposition = (
            cpu_event.get("boozer_solve_decomposition")
            if compare_native_gradient_layers
            else None
        )
        jax_boozer_solve_decomposition = (
            jax_event.get("boozer_solve_decomposition")
            if compare_native_gradient_layers
            else None
        )
        boozer_solve_decomposition_summary = (
            _compare_same_candidate_boozer_solve_decomposition(
                event_failures,
                cpu_decomposition=cpu_boozer_solve_decomposition,
                jax_decomposition=jax_boozer_solve_decomposition,
                pair_index=pair_index,
                line_search_evaluation=cpu_event.get("line_search_evaluation"),
            )
        )
        _update_parity_bug_census(
            parity_bug_census_layers,
            family="boozer_solve",
            summary=boozer_solve_decomposition_summary,
        )
        if (
            boozer_solve_decomposition_summary["max_abs_diff"]
            > max_boozer_solve_decomposition_abs_diff
        ):
            max_boozer_solve_decomposition_abs_diff = (
                boozer_solve_decomposition_summary["max_abs_diff"]
            )
            max_boozer_solve_decomposition_layer = boozer_solve_decomposition_summary[
                "max_layer"
            ]
            max_boozer_solve_decomposition_pair_index = (
                boozer_solve_decomposition_summary["pair_index"]
            )
            max_boozer_solve_decomposition_line_search_evaluation = (
                boozer_solve_decomposition_summary["line_search_evaluation"]
            )
            max_boozer_solve_decomposition_layer_diffs = dict(
                boozer_solve_decomposition_summary["layer_diffs"]
            )
        if (
            first_boozer_solve_decomposition_divergence is None
            and boozer_solve_decomposition_summary["first_divergent_layer"] is not None
        ):
            first_boozer_solve_decomposition_divergence = {
                "pair_index": boozer_solve_decomposition_summary["pair_index"],
                "line_search_evaluation": boozer_solve_decomposition_summary[
                    "line_search_evaluation"
                ],
                "layer": boozer_solve_decomposition_summary["first_divergent_layer"],
                "layer_diffs": dict(boozer_solve_decomposition_summary["layer_diffs"]),
            }
        if (
            first_parity_bug_census_divergence is None
            and boozer_solve_decomposition_summary["first_divergent_layer"] is not None
        ):
            first_parity_bug_census_divergence = {
                "family": "boozer_solve",
                "pair_index": boozer_solve_decomposition_summary["pair_index"],
                "line_search_evaluation": boozer_solve_decomposition_summary[
                    "line_search_evaluation"
                ],
                "layer": boozer_solve_decomposition_summary["first_divergent_layer"],
                "layer_diffs": dict(boozer_solve_decomposition_summary["layer_diffs"]),
            }
        max_objective_abs_diff = max(
            max_objective_abs_diff,
            _compare_same_candidate_scalar(
                event_failures,
                field="objective.value",
                cpu_value=_summary_scalar(cpu_event.get("objective")),
                jax_value=_summary_scalar(jax_event.get("objective")),
            ),
        )
        max_gradient_abs_diff = max(
            max_gradient_abs_diff,
            _compare_same_candidate_vector(
                event_failures,
                field="optimizer_gradient",
                cpu_vector=_summary_vector(cpu_event.get("optimizer_gradient")),
                jax_vector=_summary_vector(jax_event.get("optimizer_gradient")),
            ),
        )
        slice_summary = _compare_same_candidate_objective_components(
            event_failures,
            cpu_components=cpu_event.get("objective_components"),
            jax_components=jax_event.get("objective_components"),
            pair_index=pair_index,
            line_search_evaluation=cpu_event.get("line_search_evaluation"),
        )
        if slice_summary["max_slice_objective_abs_diff"] > max_slice_objective_abs_diff:
            max_slice_objective_abs_diff = slice_summary["max_slice_objective_abs_diff"]
            max_slice_objective_owner = slice_summary["max_slice_objective_owner"]
            max_slice_pair_index = slice_summary["max_slice_pair_index"]
            max_slice_line_search_evaluation = slice_summary[
                "max_slice_line_search_evaluation"
            ]
        if slice_summary["max_slice_gradient_abs_diff"] > max_slice_gradient_abs_diff:
            max_slice_gradient_abs_diff = slice_summary["max_slice_gradient_abs_diff"]
            max_slice_gradient_owner = slice_summary["max_slice_gradient_owner"]
            max_slice_pair_index = slice_summary["max_slice_pair_index"]
            max_slice_line_search_evaluation = slice_summary[
                "max_slice_line_search_evaluation"
            ]
        iota_decomposition_summary = _compare_same_candidate_iota_decomposition(
            event_failures,
            cpu_decomposition=cpu_event.get("iota_penalty_decomposition"),
            jax_decomposition=jax_event.get("iota_penalty_decomposition"),
            pair_index=pair_index,
            line_search_evaluation=cpu_event.get("line_search_evaluation"),
        )
        _update_parity_bug_census(
            parity_bug_census_layers,
            family="iota_penalty",
            summary=iota_decomposition_summary,
        )
        if iota_decomposition_summary["max_abs_diff"] > max_iota_decomposition_abs_diff:
            max_iota_decomposition_abs_diff = iota_decomposition_summary["max_abs_diff"]
            max_iota_decomposition_layer = iota_decomposition_summary["max_layer"]
            max_iota_decomposition_pair_index = iota_decomposition_summary["pair_index"]
            max_iota_decomposition_line_search_evaluation = iota_decomposition_summary[
                "line_search_evaluation"
            ]
            max_iota_decomposition_layer_diffs = dict(
                iota_decomposition_summary["layer_diffs"]
            )
        if (
            first_iota_decomposition_divergence is None
            and iota_decomposition_summary["first_divergent_layer"] is not None
        ):
            first_iota_decomposition_divergence = {
                "pair_index": iota_decomposition_summary["pair_index"],
                "line_search_evaluation": iota_decomposition_summary[
                    "line_search_evaluation"
                ],
                "layer": iota_decomposition_summary["first_divergent_layer"],
                "layer_diffs": dict(iota_decomposition_summary["layer_diffs"]),
            }
        if (
            first_parity_bug_census_divergence is None
            and iota_decomposition_summary["first_divergent_layer"] is not None
        ):
            first_parity_bug_census_divergence = {
                "family": "iota_penalty",
                "pair_index": iota_decomposition_summary["pair_index"],
                "line_search_evaluation": iota_decomposition_summary[
                    "line_search_evaluation"
                ],
                "layer": iota_decomposition_summary["first_divergent_layer"],
                "layer_diffs": dict(iota_decomposition_summary["layer_diffs"]),
            }
        if bool(cpu_event.get("native_gradient_used")) and bool(
            jax_event.get("native_gradient_used")
        ):
            max_hardware_abs_diff = max(
                max_hardware_abs_diff,
                _compare_same_candidate_vector(
                    event_failures,
                    field="boozer_surface_dofs",
                    cpu_vector=_summary_vector(cpu_event.get("boozer_surface_dofs")),
                    jax_vector=_summary_vector(jax_event.get("boozer_surface_dofs")),
                    rtol=1e-8,
                    atol=1e-10,
                ),
            )
            _compare_same_candidate_scalar(
                event_failures,
                field="boozer_iota",
                cpu_value=_summary_scalar(cpu_event.get("boozer_iota")),
                jax_value=_summary_scalar(jax_event.get("boozer_iota")),
                rtol=1e-8,
                atol=1e-10,
            )
            _compare_same_candidate_scalar(
                event_failures,
                field="boozer_G",
                cpu_value=_summary_scalar(cpu_event.get("boozer_G")),
                jax_value=_summary_scalar(jax_event.get("boozer_G")),
                rtol=1e-8,
                atol=1e-10,
            )
        max_hardware_abs_diff = max(
            max_hardware_abs_diff,
            _compare_same_candidate_hardware(
                event_failures,
                cpu_status=cpu_event.get("hardware_status"),
                jax_status=jax_event.get("hardware_status"),
            ),
        )
        max_failure_abs_diff = max(
            max_failure_abs_diff,
            _compare_same_candidate_failure(
                event_failures,
                cpu_failure=cpu_event.get("candidate_failure"),
                jax_failure=jax_event.get("candidate_failure"),
            ),
        )
        if event_failures:
            if first_failure_event is None:
                first_failure_event = {
                    "pair_index": pair_index,
                    "cpu_event_index": cpu_event.get("event_index"),
                    "jax_event_index": jax_event.get("event_index"),
                    "accepted_iteration_target": cpu_event.get(
                        "accepted_iteration_target"
                    ),
                    "line_search_evaluation": cpu_event.get("line_search_evaluation"),
                    "candidate_abs_diff": candidate_abs_diff,
                    "failures": list(event_failures),
                }
            failures.extend(
                f"pair {pair_index}: {failure}" for failure in event_failures
            )
    if same_candidate_event_count == 0:
        failures.append(
            "No paired objective-evaluation events shared the same candidate."
        )
    return {
        "status": "pass" if not failures else "fail",
        "cpu_event_count": len(cpu_events),
        "jax_event_count": len(jax_events),
        "same_candidate_event_count": same_candidate_event_count,
        "require_exact_candidates": bool(require_exact_candidates),
        "strict_solver_contract": bool(strict_solver_contract),
        "candidate_x_abs_tol": candidate_x_abs_tol,
        "max_candidate_abs_diff": max_candidate_abs_diff,
        "max_objective_abs_diff": max_objective_abs_diff,
        "max_optimizer_gradient_abs_diff": max_gradient_abs_diff,
        "max_boozer_metadata_numeric_abs_diff": max_boozer_metadata_abs_diff,
        "max_boozer_scipy_callback_abs_diff": max_boozer_scipy_callback_abs_diff,
        "first_boozer_scipy_callback_split": first_boozer_scipy_callback_split,
        "max_slice_objective_abs_diff": max_slice_objective_abs_diff,
        "max_slice_gradient_abs_diff": max_slice_gradient_abs_diff,
        "max_slice_objective_owner": max_slice_objective_owner,
        "max_slice_gradient_owner": max_slice_gradient_owner,
        "max_slice_pair_index": max_slice_pair_index,
        "max_slice_line_search_evaluation": max_slice_line_search_evaluation,
        "max_iota_decomposition_abs_diff": max_iota_decomposition_abs_diff,
        "max_iota_decomposition_layer": max_iota_decomposition_layer,
        "max_iota_decomposition_pair_index": max_iota_decomposition_pair_index,
        "max_iota_decomposition_line_search_evaluation": (
            max_iota_decomposition_line_search_evaluation
        ),
        "max_iota_decomposition_layer_diffs": max_iota_decomposition_layer_diffs,
        "first_iota_decomposition_divergence": first_iota_decomposition_divergence,
        "max_boozer_solve_decomposition_abs_diff": (
            max_boozer_solve_decomposition_abs_diff
        ),
        "max_boozer_solve_decomposition_layer": max_boozer_solve_decomposition_layer,
        "max_boozer_solve_decomposition_pair_index": (
            max_boozer_solve_decomposition_pair_index
        ),
        "max_boozer_solve_decomposition_line_search_evaluation": (
            max_boozer_solve_decomposition_line_search_evaluation
        ),
        "max_boozer_solve_decomposition_layer_diffs": (
            max_boozer_solve_decomposition_layer_diffs
        ),
        "first_boozer_solve_decomposition_divergence": (
            first_boozer_solve_decomposition_divergence
        ),
        "parity_bug_census": _finalize_parity_bug_census(
            parity_bug_census_layers,
            first_divergence=first_parity_bug_census_divergence,
        ),
        "max_hardware_metric_abs_diff": max_hardware_abs_diff,
        "max_failure_scalar_abs_diff": max_failure_abs_diff,
        "cpu_boozer_solver_summary": _first_boozer_solver_summary(cpu_events),
        "jax_boozer_solver_summary": _first_boozer_solver_summary(jax_events),
        "solver_contract_diagnostics": solver_contract_diagnostics,
        "first_failure_event": first_failure_event,
        "failures": failures,
    }


def _path_scalar_abs_diff(
    cpu_summary: dict[str, Any] | None,
    jax_summary: dict[str, Any] | None,
) -> float:
    cpu_value = _summary_scalar(cpu_summary)
    jax_value = _summary_scalar(jax_summary)
    if cpu_value is None or jax_value is None:
        return float("inf")
    return abs(float(jax_value) - float(cpu_value))


def _path_vector_abs_diff(
    cpu_summary: dict[str, Any] | None,
    jax_summary: dict[str, Any] | None,
) -> float:
    cpu_vector = _summary_vector(cpu_summary)
    jax_vector = _summary_vector(jax_summary)
    if cpu_vector is None or jax_vector is None:
        return float("inf")
    return _max_abs_diff(jax_vector, cpu_vector)


def _optimizer_path_event_diff(
    *,
    pair_index: int,
    cpu_event: dict[str, Any],
    jax_event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pair_index": int(pair_index),
        "cpu_event_index": cpu_event.get("event_index"),
        "jax_event_index": jax_event.get("event_index"),
        "cpu_accepted_iteration_target": cpu_event.get("accepted_iteration_target"),
        "jax_accepted_iteration_target": jax_event.get("accepted_iteration_target"),
        "cpu_line_search_evaluation": cpu_event.get("line_search_evaluation"),
        "jax_line_search_evaluation": jax_event.get("line_search_evaluation"),
        "candidate_abs_diff": _path_vector_abs_diff(
            cpu_event.get("candidate_optimizer_dofs"),
            jax_event.get("candidate_optimizer_dofs"),
        ),
        "objective_abs_diff": _path_scalar_abs_diff(
            cpu_event.get("objective"),
            jax_event.get("objective"),
        ),
        "optimizer_gradient_abs_diff": _path_vector_abs_diff(
            cpu_event.get("optimizer_gradient"),
            jax_event.get("optimizer_gradient"),
        ),
        "boozer_iota_abs_diff": _path_scalar_abs_diff(
            cpu_event.get("boozer_iota"),
            jax_event.get("boozer_iota"),
        ),
    }


def _max_path_event(
    current: dict[str, Any] | None,
    candidate: dict[str, Any],
    *,
    diff_key: str,
) -> dict[str, Any]:
    if current is None or float(candidate[diff_key]) > float(current[diff_key]):
        return dict(candidate)
    return current


def compare_optimizer_path_objective_evaluations(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
) -> dict[str, Any]:
    """Compare free-running CPU/JAX objective-evaluation paths.

    This is intentionally diagnostic. Same-candidate replay decides whether the
    objective contract matches at identical x; this reports where independent
    optimizer control first starts evaluating different candidates.
    """
    cpu_events = _load_objective_evaluation_events_from_case(cpu_case)
    jax_events = _load_objective_evaluation_events_from_case(jax_case)
    if not cpu_events or not jax_events:
        return {
            "status": "not-recorded",
            "cpu_event_count": len(cpu_events),
            "jax_event_count": len(jax_events),
            "paired_event_count": 0,
            "candidate_split_abs_tol": _OPTIMIZER_PATH_CANDIDATE_SPLIT_ATOL,
        }

    paired_event_count = min(len(cpu_events), len(jax_events))
    first_candidate_split_event = None
    max_candidate_event = None
    max_objective_event = None
    max_gradient_event = None
    max_iota_event = None
    for pair_index, (cpu_event, jax_event) in enumerate(
        zip(cpu_events, jax_events),
        start=1,
    ):
        event_diff = _optimizer_path_event_diff(
            pair_index=pair_index,
            cpu_event=cpu_event,
            jax_event=jax_event,
        )
        if (
            first_candidate_split_event is None
            and float(event_diff["candidate_abs_diff"])
            > _OPTIMIZER_PATH_CANDIDATE_SPLIT_ATOL
        ):
            first_candidate_split_event = dict(event_diff)
        max_candidate_event = _max_path_event(
            max_candidate_event,
            event_diff,
            diff_key="candidate_abs_diff",
        )
        max_objective_event = _max_path_event(
            max_objective_event,
            event_diff,
            diff_key="objective_abs_diff",
        )
        max_gradient_event = _max_path_event(
            max_gradient_event,
            event_diff,
            diff_key="optimizer_gradient_abs_diff",
        )
        max_iota_event = _max_path_event(
            max_iota_event,
            event_diff,
            diff_key="boozer_iota_abs_diff",
        )

    event_count_match = len(cpu_events) == len(jax_events)
    status = (
        "same-path"
        if event_count_match and first_candidate_split_event is None
        else "split"
    )
    return {
        "status": status,
        "cpu_event_count": len(cpu_events),
        "jax_event_count": len(jax_events),
        "paired_event_count": paired_event_count,
        "event_count_match": event_count_match,
        "candidate_split_abs_tol": _OPTIMIZER_PATH_CANDIDATE_SPLIT_ATOL,
        "first_candidate_split_event": first_candidate_split_event,
        "max_candidate_event": max_candidate_event,
        "max_objective_event": max_objective_event,
        "max_optimizer_gradient_event": max_gradient_event,
        "max_boozer_iota_event": max_iota_event,
    }


def _compare_case_optimizer_state_traces(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
) -> dict[str, Any]:
    return _compare_optimizer_state_trace_pair(
        _load_optimizer_state_trace_from_case(cpu_case),
        _load_optimizer_state_trace_from_case(jax_case),
    )


def _append_nonfinite_outer_loop_failures(
    failures: list[str],
    *,
    lane_label: str,
    finite_result_keys: dict[str, bool],
) -> None:
    for key, is_finite in finite_result_keys.items():
        if not is_finite:
            failures.append(
                f"{lane_label} single-stage outer-loop probe produced a non-finite {key}."
            )


def evaluate_single_stage_init_parity(
    cpu_results: dict[str, Any],
    jax_results: dict[str, Any],
    *,
    max_surface_geometry_abs: float,
    max_surface_geometry_rel: float,
    maxiter: int = DEFAULT_OUTER_MAXITER,
    expected_jax_outer_optimizer_method: str = _TARGET_OUTER_OPTIMIZER_METHOD,
) -> tuple[dict[str, Any], list[str]]:
    comparison = {
        "final_iota_abs_diff": abs(
            float(jax_results["FINAL_IOTA"]) - float(cpu_results["FINAL_IOTA"])
        ),
        "final_volume_rel_diff": relative_error(
            float(jax_results["FINAL_VOLUME"]),
            float(cpu_results["FINAL_VOLUME"]),
        ),
        "field_error_rel_diff": relative_error(
            float(jax_results["FIELD_ERROR"]),
            float(cpu_results["FIELD_ERROR"]),
        ),
        "max_curvature_rel_diff": relative_error(
            float(jax_results["MAX_CURVATURE"]),
            float(cpu_results["MAX_CURVATURE"]),
        ),
        "max_surface_pointwise_abs": max_surface_geometry_abs,
        "max_surface_pointwise_rel": max_surface_geometry_rel,
        "cpu_self_intersecting": bool(cpu_results["SELF_INTERSECTING"]),
        "jax_self_intersecting": bool(jax_results["SELF_INTERSECTING"]),
        "cpu_self_intersection_check_available": bool(
            cpu_results.get("SELF_INTERSECTION_CHECK_AVAILABLE", True)
        ),
        "jax_self_intersection_check_available": bool(
            jax_results.get("SELF_INTERSECTION_CHECK_AVAILABLE", True)
        ),
        "cpu_iterations": int(cpu_results.get("iterations", 0)),
        "jax_iterations": int(jax_results.get("iterations", 0)),
        "cpu_outer_optimizer_method": str(
            cpu_results.get("outer_optimizer_method", "lbfgs")
        ),
        "jax_outer_optimizer_method": str(
            jax_results.get("outer_optimizer_method", "lbfgs")
        ),
        "cpu_finite_result_keys": _finite_required_result_keys(cpu_results),
        "jax_finite_result_keys": _finite_required_result_keys(jax_results),
    }

    failures: list[str] = []
    if comparison["final_iota_abs_diff"] >= IOTA_ABS_TOL:
        failures.append(
            f"Final iota disagreement too large: {comparison['final_iota_abs_diff']:.2e}"
        )
    if comparison["final_volume_rel_diff"] >= VOLUME_REL_TOL:
        failures.append(
            "Final volume relative difference too large: "
            f"{comparison['final_volume_rel_diff']:.2e}"
        )
    if comparison["field_error_rel_diff"] >= FIELD_ERROR_REL_TOL:
        failures.append(
            "Final field error relative difference too large: "
            f"{comparison['field_error_rel_diff']:.2e}"
        )
    if comparison["max_surface_pointwise_rel"] >= SURFACE_GEOMETRY_REL_TOL:
        failures.append(
            "Initial Boozer surface geometry drift too large: "
            f"{comparison['max_surface_pointwise_rel']:.2e} relative"
        )
    if comparison["cpu_self_intersecting"]:
        failures.append("CPU single-stage init produced a self-intersecting surface.")
    if comparison["jax_self_intersecting"]:
        failures.append("JAX single-stage init produced a self-intersecting surface.")
    if maxiter > 0:
        if comparison["cpu_iterations"] < 1:
            failures.append(
                "CPU single-stage outer-loop probe did not accept an optimizer step."
            )
        if comparison["jax_iterations"] < 1:
            failures.append(
                "JAX single-stage outer-loop probe did not accept an optimizer step."
            )
        if (
            comparison["jax_outer_optimizer_method"]
            != expected_jax_outer_optimizer_method
        ):
            failures.append(
                "JAX target-lane outer-loop probe did not use "
                f"{expected_jax_outer_optimizer_method}."
            )
        _append_nonfinite_outer_loop_failures(
            failures,
            lane_label="CPU",
            finite_result_keys=comparison["cpu_finite_result_keys"],
        )
        _append_nonfinite_outer_loop_failures(
            failures,
            lane_label="JAX",
            finite_result_keys=comparison["jax_finite_result_keys"],
        )
    return comparison, failures


def main() -> None:
    args = parse_args()
    benchmark_mode = bool(args.benchmark_mode)
    reference_backend = _reference_case_backend(args)
    reference_benchmark_mode = _reference_case_benchmark_mode(args, benchmark_mode)
    compare_surface_geometry = _should_compare_surface_geometry(
        args,
        benchmark_mode=benchmark_mode,
    )
    stage2_bs_path = Path(args.stage2_bs_path)
    if not stage2_bs_path.exists():
        raise RuntimeError(f"Stage 2 seed fixture does not exist: {stage2_bs_path}")
    stage2_results_path = stage2_bs_path.with_name("results.json")
    if not stage2_results_path.exists():
        raise RuntimeError(
            f"Stage 2 seed results.json does not exist: {stage2_results_path}"
        )

    provenance = build_provenance(
        jax,
        jaxlib,
        title="Single-stage init parity",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "fixture": "real-single-stage-init",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": _display_path(stage2_bs_path),
            "optimizer_backend": args.optimizer_backend,
            "reference_optimizer_method": args.reference_optimizer_method,
            "boozer_optimizer_backend": args.boozer_optimizer_backend,
            "outer_maxiter": int(args.maxiter),
            "command_argv": [sys.executable, *sys.argv],
            "benchmark_mode": benchmark_mode,
            "reference_backend": reference_backend,
            "reference_platform": "cpu",
            "target_backend": "jax",
            "target_platform": args.platform,
            "reference_benchmark_mode": reference_benchmark_mode,
            "initial_step_scale": float(args.initial_step_scale),
            "initial_step_maxiter": int(args.initial_step_maxiter),
            "outer_maxls": int(args.outer_maxls),
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "iota_abs_tol": IOTA_ABS_TOL,
            "volume_rel_tol": VOLUME_REL_TOL,
            "field_error_rel_tol": FIELD_ERROR_REL_TOL,
            "surface_geometry_rel_tol": SURFACE_GEOMETRY_REL_TOL,
            "compile_behavior": describe_compile_behavior(uses_subprocesses=True),
            "optimizer_drift_tolerances": dict(_TIER3_TOLERANCES),
        },
    )
    bundle_provenance = {
        "runner": "benchmarks/single_stage_init_parity.py",
        "fake": False,
        "default_backend": provenance["backend"],
        "devices": provenance["devices"],
        "xla_flags": provenance["xla_flags"],
        "cuda_force_ptx_jit": provenance["cuda_force_ptx_jit"],
        "cuda_disable_ptx_jit": provenance["cuda_disable_ptx_jit"],
    }
    print_provenance(provenance)

    case_artifacts_dir = (
        None if args.case_artifacts_dir is None else Path(args.case_artifacts_dir)
    )
    if case_artifacts_dir is None:
        with tempfile.TemporaryDirectory(
            prefix="single-stage-init-reference-"
        ) as reference_temp_dir:
            case_root = Path(reference_temp_dir)
            (
                cpu_case,
                jax_case,
                jax_seed_spec,
                seed_case,
                same_candidate_replay_case,
            ) = _run_single_stage_case_pair(
                args,
                benchmark_mode=benchmark_mode,
                reference_backend=reference_backend,
                reference_benchmark_mode=reference_benchmark_mode,
                case_root=case_root,
            )
            case_artifacts = None
    else:
        case_artifacts_dir.mkdir(parents=True, exist_ok=True)
        (
            cpu_case,
            jax_case,
            jax_seed_spec,
            seed_case,
            same_candidate_replay_case,
        ) = _run_single_stage_case_pair(
            args,
            benchmark_mode=benchmark_mode,
            reference_backend=reference_backend,
            reference_benchmark_mode=reference_benchmark_mode,
            case_root=case_artifacts_dir,
        )
        case_artifacts = {
            "case_artifacts_dir": str(case_artifacts_dir),
            "reference_run_dir": str(cpu_case["run_dir"]),
            "target_run_dir": str(jax_case["run_dir"]),
            "reference_outer_optimizer_progress_json": cpu_case[
                "outer_optimizer_progress_json"
            ],
            "target_outer_optimizer_progress_json": jax_case[
                "outer_optimizer_progress_json"
            ],
            "jax_runtime_seed_spec": str(jax_seed_spec),
        }
        if same_candidate_replay_case is not None:
            case_artifacts["target_same_candidate_replay_run_dir"] = str(
                same_candidate_replay_case["run_dir"]
            )
            case_artifacts["target_same_candidate_replay_progress_json"] = (
                same_candidate_replay_case["outer_optimizer_progress_json"]
            )
        if seed_case is not None:
            case_artifacts["shared_seed_run_dir"] = str(seed_case["run_dir"])

    full_run_artifact_contract = build_single_stage_full_run_artifact_contract(
        args,
        reference_backend=reference_backend,
        cpu_case=cpu_case,
        jax_case=jax_case,
        jax_seed_spec=jax_seed_spec,
    )
    cpu_results = cpu_case["results"]
    jax_results = jax_case["results"]
    max_geom_abs, max_geom_rel = _resolve_surface_geometry_drift(
        cpu_case,
        jax_case,
        compare_surface_geometry=compare_surface_geometry,
    )
    comparison, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=max_geom_abs,
        max_surface_geometry_rel=max_geom_rel,
        maxiter=int(args.maxiter),
        expected_jax_outer_optimizer_method=_expected_target_outer_optimizer_method(
            args.optimizer_backend
        ),
    )
    optimizer_state_trace_parity = None
    same_candidate_replay = None
    optimizer_path_objective_evaluations = None
    if bool(args.record_objective_evaluation_trace):
        same_candidate_target_case = (
            jax_case
            if same_candidate_replay_case is None
            else same_candidate_replay_case
        )
        same_candidate_replay = compare_same_candidate_objective_replay(
            cpu_case,
            same_candidate_target_case,
            require_exact_candidates=same_candidate_replay_case is not None,
        )
        failures.extend(_same_candidate_replay_gate_failures(same_candidate_replay))
        optimizer_path_objective_evaluations = (
            compare_optimizer_path_objective_evaluations(cpu_case, jax_case)
        )
        if (
            same_candidate_replay["status"] == "pass"
            and optimizer_path_objective_evaluations["status"] == "split"
        ):
            comparison["optimizer_path_split_kind"] = (
                "optimizer_acceptance_split_after_same_candidate_parity"
            )
    if int(args.maxiter) > 0 and args.reference_optimizer_method == "lbfgs-trace":
        optimizer_state_trace_parity = _compare_case_optimizer_state_traces(
            cpu_case,
            jax_case,
        )
        if optimizer_state_trace_parity["status"] != "pass":
            failures.append(
                "CPU/C++ lbfgs-trace diagnostic vs JAX CPU optimizer_state_trace "
                "comparison "
                f"failed: {optimizer_state_trace_parity['status']}."
            )
    proof_parity = {
        **gpu_proof_parity_contract("single_stage"),
        "cpu_oracle_value": float(cpu_results["FIELD_ERROR"]),
        "gpu_value": float(jax_results["FIELD_ERROR"]),
        "value_rel_diff": float(comparison["field_error_rel_diff"]),
    }
    warnings: list[str] = []
    if not comparison["cpu_self_intersection_check_available"]:
        warnings.append(
            "CPU self-intersection parity check was skipped because the optional "
            "surface self-intersection backend is unavailable."
        )
    if not comparison["jax_self_intersection_check_available"]:
        warnings.append(
            "JAX self-intersection parity check was skipped because the optional "
            "surface self-intersection backend is unavailable."
        )
    if (
        comparison["cpu_self_intersection_check_available"]
        != comparison["jax_self_intersection_check_available"]
    ):
        warnings.append(
            "CPU and JAX lanes did not have matching self-intersection check availability."
        )
    if benchmark_mode:
        warnings.append(
            "Surface geometry drift comparison was skipped because --benchmark-mode "
            "suppresses the surf_init.json artifact."
        )
    elif not compare_surface_geometry:
        warnings.append(
            "Surface geometry drift comparison was skipped because outer-loop "
            "parity compares optimizer progress and final metrics; the JAX "
            "target lane does not emit surf_init.json in this run shape."
        )

    print(
        "CPU vs JAX: "
        f"|iota diff|={comparison['final_iota_abs_diff']:.2e}, "
        f"volume rel_diff={comparison['final_volume_rel_diff']:.2e}, "
        f"field error rel_diff={comparison['field_error_rel_diff']:.2e}, "
        f"surface rel_diff={comparison['max_surface_pointwise_rel']:.2e}"
    )
    for warning in warnings:
        print(f"NOTE: {warning}")
    if (
        optimizer_path_objective_evaluations is not None
        and optimizer_path_objective_evaluations["status"] == "split"
    ):
        first_split = optimizer_path_objective_evaluations.get(
            "first_candidate_split_event"
        )
        if first_split is not None:
            print(
                "Optimizer path split: "
                f"pair={first_split['pair_index']}, "
                f"cpu_iter={first_split['cpu_accepted_iteration_target']}, "
                f"cpu_ls={first_split['cpu_line_search_evaluation']}, "
                f"candidate_abs_diff={first_split['candidate_abs_diff']:.2e}"
            )

    payload = {
        "provenance": provenance,
        "bundle_provenance": bundle_provenance,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "comparison": comparison,
        "proof_parity": proof_parity,
        "full_run_artifact_contract": full_run_artifact_contract,
        "lanes": {
            lane: contract["run_dir"]
            for lane, contract in full_run_artifact_contract["lanes"].items()
        },
        "timings": {
            "cpu_elapsed_s": float(cpu_case["elapsed_s"]),
            "jax_elapsed_s": float(jax_case["elapsed_s"]),
            **_prefix_phase_timings("cpu", cpu_case["phase_timings"]),
            **_prefix_phase_timings("jax", jax_case["phase_timings"]),
        },
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }
    if optimizer_state_trace_parity is not None:
        payload["optimizer_state_trace_parity"] = optimizer_state_trace_parity
    if same_candidate_replay is not None:
        payload["same_candidate_replay"] = same_candidate_replay
    if optimizer_path_objective_evaluations is not None:
        payload["optimizer_path_objective_evaluations"] = (
            optimizer_path_objective_evaluations
        )
    if case_artifacts is not None:
        payload["artifacts"] = case_artifacts
    write_json(args.output_json, payload)
    if failures:
        print("SINGLE-STAGE INIT PARITY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("SINGLE-STAGE INIT PARITY PASSED")


if __name__ == "__main__":
    main()
