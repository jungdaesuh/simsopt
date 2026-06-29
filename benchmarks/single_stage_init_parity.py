"""Tier 3 real single-stage init parity probe on a fixed Columbia seed."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping
import contextlib
from dataclasses import dataclass, field as dataclass_field
import json
import os
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

from benchmarks.benchmark_timing_labels import (
    MIXED_PARITY_REFERENCE,
    benchmark_timing_label,
)
from benchmarks.validation_ladder_common import (
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    find_single_file,
    gpu_proof_parity_contract,
    isolate_parent_cuda_memory_allocator,
    load_single_stage_final_payload as load_single_stage_final_payload_from_artifact_contract,
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
    SMOKE_TEST_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
)
from benchmarks.parity_solve_quality import (
    compute_dense_operator_action_max_rel_error,
)
from simsopt_jax.geo.optimizers.single_stage_routing import (
    JAX_DECOMPOSED_SCIPY_OUTER_OPTIMIZER_METHOD,
    JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD,
    resolve_single_stage_jax_boozer_optimizer_backend as _resolve_single_stage_jax_boozer_optimizer_backend,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
CHILD_CUDA_MEMORY_ENV = isolate_parent_cuda_memory_allocator(REQUESTED_PLATFORM)
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
TARGET_OPTIMIZER_BACKEND = "ondevice"
HOST_JAX_OPTIMIZER_BACKEND = "host-jax"
SCIPY_JAX_OPTIMIZER_BACKEND = "scipy-jax"
SCIPY_JAX_DECOMPOSED_OPTIMIZER_BACKEND = "scipy-jax-decomposed"
SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND = "scipy-jax-fullgraph"
OPTAX_LBFGS_OPTIMIZER_BACKEND = "optax-lbfgs"
OPTIMISTIX_LBFGS_OPTIMIZER_BACKEND = "optimistix-lbfgs"
TARGET_NATIVE_LBFGS_OPTIMIZER_BACKENDS = (
    TARGET_OPTIMIZER_BACKEND,
    OPTAX_LBFGS_OPTIMIZER_BACKEND,
    OPTIMISTIX_LBFGS_OPTIMIZER_BACKEND,
)
TARGET_OPTIMIZER_BACKENDS = (
    *TARGET_NATIVE_LBFGS_OPTIMIZER_BACKENDS,
    HOST_JAX_OPTIMIZER_BACKEND,
    SCIPY_JAX_OPTIMIZER_BACKEND,
    SCIPY_JAX_DECOMPOSED_OPTIMIZER_BACKEND,
    SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND,
)
REFERENCE_BACKEND_AUTO = "auto"
REFERENCE_BACKEND_CPU = "cpu"
REFERENCE_BACKEND_JAX = "jax"
REFERENCE_BACKENDS = (
    REFERENCE_BACKEND_AUTO,
    REFERENCE_BACKEND_CPU,
    REFERENCE_BACKEND_JAX,
)
DEFAULT_OUTER_MAXITER = 0
MAX_COLD_SEED_OUTER_RUN_RESOLUTION = 4
HIGH_RES_SEED_MIN_ABS_IOTA = 1.0e-3
HIGH_RES_SEED_MIN_TARGET_IOTA_FRACTION = 0.5
SINGLE_STAGE_JAX_RUNTIME_SPEC_SCHEMA = "simsopt.single_stage.jax_runtime_spec"
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
    TARGET_OPTIMIZER_BACKEND: "lbfgs-ondevice",
    HOST_JAX_OPTIMIZER_BACKEND: "lbfgs",
    SCIPY_JAX_OPTIMIZER_BACKEND: _TARGET_OUTER_OPTIMIZER_METHOD,
    SCIPY_JAX_DECOMPOSED_OPTIMIZER_BACKEND: (
        JAX_DECOMPOSED_SCIPY_OUTER_OPTIMIZER_METHOD
    ),
    SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND: JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD,
    OPTAX_LBFGS_OPTIMIZER_BACKEND: "optax-lbfgs-ondevice",
    OPTIMISTIX_LBFGS_OPTIMIZER_BACKEND: "optimistix-lbfgs-ondevice",
}
_EXACT_SAME_CANDIDATE_REPLAY_BACKENDS = frozenset(
    {
        TARGET_OPTIMIZER_BACKEND,
        SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND,
        OPTAX_LBFGS_OPTIMIZER_BACKEND,
    }
)
_PUBLIC_OPTIMIZER_COMPARISON_BACKENDS = frozenset(
    {OPTAX_LBFGS_OPTIMIZER_BACKEND, OPTIMISTIX_LBFGS_OPTIMIZER_BACKEND}
)
_STRICT_TRANSFER_UNSUPPORTED_OPTIMIZER_BACKENDS = frozenset(
    {OPTIMISTIX_LBFGS_OPTIMIZER_BACKEND}
)
_TARGET_LANE_COMPILE_DIAGNOSTICS_HOST_CALLBACK_REASON = (
    "compile diagnostics are disabled when Phase 1 host-callback diagnostics "
    "are enabled because that mode does not provide normal cache-reuse evidence"
)
_TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_DEFAULT = "run"
_TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_CHOICES = (
    "default",
    "run",
    "skip",
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
        "--reference-case-artifacts-dir",
        default=None,
        help=(
            "Existing native CPU/C++ single-stage output root to use as the "
            "reference lane instead of running a new CPU reference child. This "
            "preserves the same parity JSON contract while allowing GPU pods to "
            "run only the JAX target lane after CPU-pod reference artifacts have "
            "been produced."
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
        default=str(SMOKE_TEST_STAGE2_BS_PATH),
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
        default=DEFAULT_OPTIMIZER_BACKEND,
        help=(
            "JAX outer optimizer backend for the init probe. The default "
            "scipy-jax lane uses SciPy-compatible host control over JAX "
            "objective evaluations; use host-jax for host-controlled "
            "Boozer/outer iteration with JAX kernels, ondevice, optax-lbfgs, "
            "or optimistix-lbfgs for explicit target-lane stress tests, and "
            "scipy-jax-fullgraph for host control over the full JAX value/grad "
            "graph."
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
        "--reference-backend",
        choices=REFERENCE_BACKENDS,
        default=REFERENCE_BACKEND_AUTO,
        help=(
            "Reference lane backend. auto preserves the historical warm-start "
            "behavior: warm-start/runtime-spec runs use a JAX CPU reference, "
            "while fixture runs use the native CPU/C++ reference. Use cpu to "
            "force the native CPU/C++ reference lane for seed-matrix parity."
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
        "--target-lane-boozer-bfgs-maxiter",
        type=int,
        default=None,
        help=(
            "Optional target-lane Boozer BFGS iteration cap override passed "
            "through to the single-stage runner."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-newton-polish-policy",
        choices=_TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_CHOICES,
        default=_TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_DEFAULT,
        help=(
            "Policy for the JAX/ondevice Boozer dense Newton polish in "
            "target-lane single-stage children. Defaults to 'run' (the dense "
            "polish always runs, matching the production GPU lane); pass 'skip' "
            "to bypass it and accept the least-squares solved state."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-newton-tol",
        type=float,
        default=None,
        help=(
            "Optional target-lane Boozer Newton tolerance override passed "
            "through to the single-stage runner."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-newton-maxiter",
        type=int,
        default=None,
        help=(
            "Optional target-lane Boozer Newton iteration cap override passed "
            "through to the single-stage runner."
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
        "--minimal-artifacts",
        action="store_true",
        help=(
            "Pass --minimal-artifacts to single-stage child runs so benchmark "
            "runs keep restart/JSON outputs but skip VTK and plot exports."
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
        "--record-jax-compile-diagnostics",
        action="store_true",
        help=(
            "Record named JAX compile/cache-miss diagnostics for the JAX "
            "target-lane child and write the summary into its results.json. "
            "Separates one-time compile cost from steady-state throughput and "
            "surfaces recompile counts (e.g. GPU XLA cache thrash) for the "
            "fair CPU-vs-GPU comparison. Observational only: it toggles JAX "
            "compile logging and does not change compiled code or numerics."
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
        "--single-stage-case-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional per-lane subprocess wall-clock limit. A positive value "
            "turns child stalls into structured parity-run failures instead "
            "of leaving the harness without an output JSON."
        ),
    )
    parser.add_argument(
        "--single-stage-target-case-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional JAX target/replay subprocess wall-clock limit. A positive "
            "value overrides --single-stage-case-timeout-seconds for JAX cases "
            "only, so slow CPU oracle lanes can remain uncapped."
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
    if args.optimizer_backend not in TARGET_NATIVE_LBFGS_OPTIMIZER_BACKENDS:
        return _TARGET_LANE_PER_ACCEPT_SYNC
    if int(args.maxiter) <= 0:
        return _TARGET_LANE_PER_ACCEPT_SYNC
    return _TARGET_LANE_FINAL_ONLY_SYNC


def _target_platform_uses_cuda(platform: str) -> bool:
    normalized = str(platform).strip().lower()
    if normalized == "cuda":
        return True
    if normalized != "auto":
        return False
    return str(jax.default_backend()).strip().lower() in {"cuda", "gpu"}


def _resolve_target_boozer_optimizer_backend(
    *,
    backend: str,
    platform: str,
    args: argparse.Namespace,
) -> str | None:
    if backend != "jax":
        return None
    if args.boozer_optimizer_backend is not None:
        return args.boozer_optimizer_backend
    if getattr(args, "optimizer_backend", None) == "host-jax":
        return "host-jax"
    if _target_platform_uses_cuda(platform):
        return TARGET_OPTIMIZER_BACKEND
    return None


def _normalize_target_lane_boozer_newton_polish_policy(policy: str | None) -> str:
    normalized = (
        _TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_DEFAULT
        if policy is None
        else str(policy).strip().lower()
    )
    if normalized == "default":
        normalized = _TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_DEFAULT
    if normalized not in _TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_CHOICES:
        choices = ", ".join(_TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_CHOICES)
        raise ValueError(
            f"target_lane_boozer_newton_polish_policy must be one of: {choices}."
        )
    return normalized


def _requested_target_lane_boozer_newton_polish_policy(
    args: argparse.Namespace,
) -> str:
    return _normalize_target_lane_boozer_newton_polish_policy(
        getattr(
            args,
            "target_lane_boozer_newton_polish_policy",
            _TARGET_LANE_BOOZER_NEWTON_POLISH_POLICY_DEFAULT,
        )
    )


def _resolve_target_lane_boozer_newton_polish_policy(
    *,
    backend: str,
    platform: str,
    args: argparse.Namespace,
) -> str | None:
    if backend != "jax":
        return None
    policy = _requested_target_lane_boozer_newton_polish_policy(args)
    boozer_optimizer_backend = _resolve_target_boozer_optimizer_backend(
        backend=backend,
        platform=platform,
        args=args,
    )
    effective_boozer_optimizer_backend = (
        _resolve_single_stage_jax_boozer_optimizer_backend(
            backend,
            getattr(args, "optimizer_backend", None),
            boozer_optimizer_backend,
        )
    )
    if effective_boozer_optimizer_backend != TARGET_OPTIMIZER_BACKEND:
        return None
    return policy


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


_PROGRESS_EVENT_TIMING_KEYS = {
    "initial_hardware_status_returned": "initial_hardware_status_s",
    "phase1_returned": "outer_optimizer_initial_phase_s",
    "phase2_returned": "outer_optimizer_main_s",
    "final_penalty_metrics_returned": "final_penalty_metrics_s",
    "final_hardware_metrics_returned": "final_hardware_metrics_s",
    "final_reporting_returned": "final_reporting_s",
}
_CORE_OPTIMIZER_TIMING_KEYS = (
    "outer_optimizer_initial_phase_s",
    "outer_optimizer_main_s",
)
_STATUS_PREREPORTING_TIMING_KEYS = (
    "initial_hardware_status_s",
    "target_lane_init_reporting_snapshot_s",
)
_FINAL_REPORTING_COMPONENT_TIMING_KEYS = (
    "target_lane_deferred_reporting_snapshot_s",
    "final_penalty_metrics_s",
    "final_hardware_metrics_s",
    "final_host_diagnostics_s",
    "final_artifacts_s",
)
_CASE_ARTIFACT_SOURCE_GENERATED = "generated_current_run"
_CASE_ARTIFACT_SOURCE_EXTERNAL_REFERENCE = "external_reference_artifact"


def _load_progress_event_timings(progress_path: Path) -> dict[str, float]:
    if not progress_path.exists():
        return {}
    payload = load_json(progress_path)
    timings: dict[str, float] = {}
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        key = _PROGRESS_EVENT_TIMING_KEYS.get(str(event.get("label")))
        elapsed_s = event.get("elapsed_s")
        if key is not None and isinstance(
            elapsed_s, (int, float, np.integer, np.floating)
        ):
            timings[key] = float(elapsed_s)
    return timings


def _load_progress_elapsed_s(progress_path: Path) -> float | None:
    if not progress_path.exists():
        return None
    payload = load_json(progress_path)
    event_elapsed_values = [
        float(event["event_elapsed_s"])
        for event in payload.get("events", [])
        if isinstance(event, dict)
        and isinstance(
            event.get("event_elapsed_s"), (int, float, np.integer, np.floating)
        )
    ]
    if not event_elapsed_values:
        return None
    return float(max(event_elapsed_values))


def _sum_timing_keys(
    timings: Mapping[str, float],
    keys: tuple[str, ...],
) -> float | None:
    values = [float(timings[key]) for key in keys if key in timings]
    if not values:
        return None
    return float(sum(values))


def _elapsed_remainder_s(elapsed_s: float, subtotal_s: float | None) -> float | None:
    if subtotal_s is None:
        return None
    return max(float(elapsed_s) - float(subtotal_s), 0.0)


def _case_timing_breakdown(case: dict[str, Any]) -> dict[str, Any]:
    progress_path = Path(case["outer_optimizer_progress_json"])
    progress_timings = _load_progress_event_timings(progress_path)
    phase_timings = dict(case["phase_timings"])
    combined_timings = {**progress_timings, **phase_timings}
    elapsed_s = case.get("elapsed_s")
    elapsed_source = case.get("elapsed_source")
    if elapsed_s is None:
        elapsed_s = _load_progress_elapsed_s(progress_path)
        if elapsed_s is not None:
            elapsed_source = "outer_optimizer_progress_event_elapsed_s"
    elapsed_s = None if elapsed_s is None else float(elapsed_s)
    core_optimizer_s = _sum_timing_keys(
        combined_timings,
        _CORE_OPTIMIZER_TIMING_KEYS,
    )
    status_prereporting_s = _sum_timing_keys(
        combined_timings,
        _STATUS_PREREPORTING_TIMING_KEYS,
    )
    final_reporting_s = combined_timings.get("final_reporting_s")
    if final_reporting_s is None:
        final_reporting_s = _sum_timing_keys(
            combined_timings,
            _FINAL_REPORTING_COMPONENT_TIMING_KEYS,
        )
    status_reporting_parts = [
        value
        for value in (status_prereporting_s, final_reporting_s)
        if value is not None
    ]
    status_reporting_s = (
        None if not status_reporting_parts else float(sum(status_reporting_parts))
    )
    optimizer_wall_excluding_status_reporting_s = (
        None
        if elapsed_s is None
        else _elapsed_remainder_s(elapsed_s, status_reporting_s)
    )
    return {
        "elapsed_s": elapsed_s,
        "elapsed_source": elapsed_source,
        "core_optimizer_s": core_optimizer_s,
        "optimizer_wall_excluding_status_reporting_s": (
            optimizer_wall_excluding_status_reporting_s
        ),
        "status_reporting_s": status_reporting_s,
        "status_prereporting_s": status_prereporting_s,
        "non_core_s": None
        if elapsed_s is None
        else _elapsed_remainder_s(elapsed_s, core_optimizer_s),
        "core_optimizer_fraction_of_elapsed": _timing_ratio(
            core_optimizer_s,
            elapsed_s,
        ),
        "status_reporting_fraction_of_elapsed": _timing_ratio(
            status_reporting_s,
            elapsed_s,
        ),
        "optimizer_wall_excluding_status_reporting_fraction_of_elapsed": (
            _timing_ratio(optimizer_wall_excluding_status_reporting_s, elapsed_s)
        ),
        "initial_hardware_status_s": combined_timings.get("initial_hardware_status_s"),
        "final_penalty_metrics_s": combined_timings.get("final_penalty_metrics_s"),
        "final_hardware_metrics_s": combined_timings.get("final_hardware_metrics_s"),
        "final_reporting_s": final_reporting_s,
        "outer_optimizer_main_s": combined_timings.get("outer_optimizer_main_s"),
        "outer_optimizer_initial_phase_s": combined_timings.get(
            "outer_optimizer_initial_phase_s"
        ),
        "timing_sources": {
            "phase_keys": sorted(phase_timings),
            "progress_event_keys": sorted(progress_timings),
        },
    }


def _timing_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _single_stage_pair_timing_breakdown(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
) -> dict[str, Any]:
    cpu_breakdown = _case_timing_breakdown(cpu_case)
    jax_breakdown = _case_timing_breakdown(jax_case)
    cpu_core_s = cpu_breakdown["core_optimizer_s"]
    jax_core_s = jax_breakdown["core_optimizer_s"]
    cpu_status_reporting_s = cpu_breakdown["status_reporting_s"]
    jax_status_reporting_s = jax_breakdown["status_reporting_s"]
    cpu_optimizer_wall_excluding_status_reporting_s = cpu_breakdown[
        "optimizer_wall_excluding_status_reporting_s"
    ]
    jax_optimizer_wall_excluding_status_reporting_s = jax_breakdown[
        "optimizer_wall_excluding_status_reporting_s"
    ]
    return {
        "cpu": cpu_breakdown,
        "jax": jax_breakdown,
        "jax_wall_vs_cpu_wall_ratio": _timing_ratio(
            jax_breakdown["elapsed_s"],
            cpu_breakdown["elapsed_s"],
        ),
        "jax_optimizer_wall_excluding_status_reporting_vs_cpu_ratio": (
            _timing_ratio(
                jax_optimizer_wall_excluding_status_reporting_s,
                cpu_optimizer_wall_excluding_status_reporting_s,
            )
        ),
        "jax_core_vs_cpu_core_ratio": _timing_ratio(jax_core_s, cpu_core_s),
        "jax_status_reporting_minus_cpu_s": (
            None
            if cpu_status_reporting_s is None or jax_status_reporting_s is None
            else float(jax_status_reporting_s) - float(cpu_status_reporting_s)
        ),
    }


def _single_stage_elapsed_s_and_source_from_results(
    results: dict[str, Any],
) -> tuple[float | None, str | None]:
    elapsed_s = results.get("ELAPSED_SECONDS")
    if isinstance(elapsed_s, (int, float, np.integer, np.floating)):
        return float(elapsed_s), "results_ELAPSED_SECONDS"
    timings = _extract_phase_timings(results)
    script_total_s = timings.get("script_total_s")
    if script_total_s is None:
        return None, None
    return float(script_total_s), "results_TIMINGS_script_total_s"


def _single_stage_elapsed_s_from_results(results: dict[str, Any]) -> float | None:
    elapsed_s, _elapsed_source = _single_stage_elapsed_s_and_source_from_results(
        results
    )
    return elapsed_s


def _require_loaded_single_stage_case_backend(
    results: dict[str, Any],
    *,
    expected_backend: str,
    output_root: Path,
) -> None:
    artifact_backend = results.get("backend")
    if artifact_backend != expected_backend:
        raise ValueError(
            "Loaded single-stage reference artifact backend mismatch: "
            f"expected results['backend']={expected_backend!r} under {output_root}, "
            f"found {artifact_backend!r}."
        )


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
            "target_lane_boozer_newton_polish_policy": (
                _requested_target_lane_boozer_newton_polish_policy(args)
            ),
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
    final_artifact_json = str(case["final_artifact_json"])
    final_artifact_accepted = bool(case["final_artifact_accepted"])
    artifact_source = str(case.get("artifact_source", _CASE_ARTIFACT_SOURCE_GENERATED))
    loaded_external_reference = (
        artifact_source == _CASE_ARTIFACT_SOURCE_EXTERNAL_REFERENCE
    )
    verified_runtime_seed_spec_hash = (
        None if loaded_external_reference else runtime_seed_spec_hash
    )
    verified_run_family_id = None if loaded_external_reference else run_family_id
    return {
        "run_dir": str(run_dir),
        "results_json": str(run_dir / "results.json")
        if final_artifact_accepted
        else None,
        "final_artifact_json": final_artifact_json,
        "final_artifact_accepted": final_artifact_accepted,
        "progress_json": str(case["outer_optimizer_progress_json"]),
        "artifact_source": artifact_source,
        "loaded_external_reference": bool(loaded_external_reference),
        "artifact_source_root": case.get("artifact_source_root"),
        "runtime_seed_spec_hash": verified_runtime_seed_spec_hash,
        "runtime_seed_spec_hash_verified": bool(
            verified_runtime_seed_spec_hash is not None
        ),
        "current_run_runtime_seed_spec_hash": runtime_seed_spec_hash,
        "objective_configuration_hash": objective_hash,
        "missing_objective_config_keys": missing_objective_keys,
        "run_family_id": verified_run_family_id,
        "run_family_id_verified": bool(verified_run_family_id is not None),
        "current_run_family_id": run_family_id,
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
    minimal_artifacts: bool,
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
    record_target_optimizer_state_trace: bool,
    target_lane_boozer_bfgs_tol: float | None = None,
    target_lane_boozer_bfgs_maxiter: int | None = None,
    target_lane_boozer_newton_tol: float | None = None,
    target_lane_boozer_newton_maxiter: int | None = None,
    target_lane_boozer_newton_polish_policy: str | None = None,
    replay_objective_evaluation_trace: Path | None = None,
) -> None:
    if benchmark_mode:
        command.append("--benchmark-mode")
    if minimal_artifacts:
        command.append("--minimal-artifacts")
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
    if record_target_optimizer_state_trace:
        command.append("--record-target-optimizer-state-trace")
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
    if target_lane_boozer_newton_polish_policy is not None:
        command.extend(
            [
                "--target-lane-boozer-newton-polish-policy",
                str(target_lane_boozer_newton_polish_policy),
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


def _load_single_stage_final_payload(output_root: Path) -> tuple[dict[str, Any], Path]:
    """Load the final run payload from the single-stage artifact contract."""
    return load_single_stage_final_payload_from_artifact_contract(output_root)


def _load_single_stage_case_from_output_root(
    output_root: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    load_surface_gamma: bool,
) -> dict[str, Any]:
    results, final_artifact_json = _load_single_stage_final_payload(output_root)
    _require_loaded_single_stage_case_backend(
        results,
        expected_backend=backend,
        output_root=output_root,
    )
    run_dir = final_artifact_json.parent
    progress_path = run_dir / "outer_optimizer_progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError(
            "Loaded single-stage reference artifact is missing required "
            f"outer_optimizer_progress.json: {progress_path}"
        )
    elapsed_s, elapsed_source = _single_stage_elapsed_s_and_source_from_results(results)
    if elapsed_s is None:
        elapsed_s = _load_progress_elapsed_s(progress_path)
        if elapsed_s is not None:
            elapsed_source = "outer_optimizer_progress_event_elapsed_s"
    payload = {
        "results": results,
        "surface_gamma": None,
        "elapsed_s": elapsed_s,
        "elapsed_source": elapsed_source,
        "phase_timings": _extract_phase_timings(results),
        "run_dir": str(run_dir),
        "final_artifact_json": str(final_artifact_json),
        "final_artifact_accepted": final_artifact_json.name == "results.json",
        "outer_optimizer_progress_json": str(progress_path),
        "artifact_source": _CASE_ARTIFACT_SOURCE_EXTERNAL_REFERENCE,
        "artifact_source_root": str(output_root),
    }
    if load_surface_gamma:
        if backend == "jax":
            runtime_spec_json = find_single_file(
                output_root,
                "single_stage_jax_runtime_spec.json",
            )
            payload["surface_gamma"] = _load_surface_gamma_runtime_spec(
                str(runtime_spec_json),
                args,
            )
        else:
            surf_json = find_single_file(output_root, "surf_init.json")
            payload["surface_gamma"] = _load_surface_gamma_artifact(str(surf_json))
    return payload


def _resolve_single_stage_child_platform(
    *,
    backend: str,
    platform: str,
) -> str:
    if backend != "jax":
        return "cpu"
    return platform


def _should_reuse_resolved_warm_start_solve(args: argparse.Namespace) -> bool:
    """Return whether a resolved seed state should skip setup replay."""
    if int(args.maxiter) <= 0:
        return False
    if getattr(args, "warm_start_run_dir", None) is not None:
        return True
    return bool(getattr(args, "reuse_jax_runtime_seed_solve", False)) and (
        getattr(args, "jax_runtime_seed_spec", None) is not None
    )


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
    effective_platform = _resolve_single_stage_child_platform(
        backend=backend,
        platform=platform,
    )
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
        if _should_reuse_resolved_warm_start_solve(args):
            command.append("--reuse-resolved-warm-start-solve")
        if int(args.maxiter) <= 0:
            command.append("--init-only")
        else:
            command.extend(["--maxiter", str(args.maxiter)])
        if backend == "jax":
            command.extend(["--optimizer-backend", args.optimizer_backend])
            boozer_optimizer_backend = _resolve_target_boozer_optimizer_backend(
                backend=backend,
                platform=platform,
                args=args,
            )
            resolved_seed_spec = (
                jax_runtime_seed_spec
                if jax_runtime_seed_spec is not None
                else getattr(args, "jax_runtime_seed_spec", None)
            )
            if resolved_seed_spec is not None:
                command.extend(["--jax-runtime-seed-spec", str(resolved_seed_spec)])
            if boozer_optimizer_backend is not None:
                command.extend(
                    [
                        "--boozer-optimizer-backend",
                        boozer_optimizer_backend,
                    ]
                )
            boozer_least_squares_algorithm = getattr(
                args,
                "boozer_least_squares_algorithm",
                None,
            )
            if boozer_least_squares_algorithm is not None:
                command.extend(
                    [
                        "--boozer-least-squares-algorithm",
                        str(boozer_least_squares_algorithm),
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
            minimal_artifacts=bool(getattr(args, "minimal_artifacts", False)),
            profile_target_lane=profile_target_lane,
            profile_target_lane_only=profile_target_lane_only,
            diagnose_target_lane_scaled_phase1=diagnose_target_lane_scaled_phase1,
            record_target_lane_invalid_state_events=(
                record_target_lane_invalid_state_events
            ),
            profile_target_lane_batch_size=getattr(
                args, "profile_target_lane_batch_size", None
            ),
            enable_compile_diagnostics=(
                enable_compile_diagnostics
                or (
                    backend == "jax"
                    and bool(getattr(args, "record_jax_compile_diagnostics", False))
                )
            ),
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
            record_target_optimizer_state_trace=bool(
                backend == "jax"
                and args.optimizer_backend == TARGET_OPTIMIZER_BACKEND
                and getattr(args, "reference_optimizer_method", "lbfgs")
                == "lbfgs-trace"
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
            target_lane_boozer_newton_polish_policy=(
                _resolve_target_lane_boozer_newton_polish_policy(
                    backend=backend,
                    platform=platform,
                    args=args,
                )
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
        case_env = repo_pythonpath_env(
            platform=effective_platform,
            disable_compilation_cache=(effective_platform == "cpu"),
            clear_backend_guardrails=(backend != "jax"),
            deterministic_gpu_reductions=deterministic_gpu_reductions,
            cuda_memory_env=CHILD_CUDA_MEMORY_ENV,
        )
        run_python_script(
            script_path,
            command,
            env=case_env,
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
            timeout_seconds=_single_stage_case_timeout_seconds(
                args,
                backend=backend,
                platform=platform,
            ),
        )
        elapsed_s = time.perf_counter() - start

        results, final_artifact_json = _load_single_stage_final_payload(resolved_root)
        run_dir = final_artifact_json.parent
        payload = {
            "results": results,
            "surface_gamma": None,
            "elapsed_s": float(elapsed_s),
            "elapsed_source": "measured_subprocess_wall_s",
            "phase_timings": _extract_phase_timings(results),
            "run_dir": str(run_dir),
            "final_artifact_json": str(final_artifact_json),
            "final_artifact_accepted": final_artifact_json.name == "results.json",
            "outer_optimizer_progress_json": str(
                run_dir / "outer_optimizer_progress.json"
            ),
            "artifact_source": _CASE_ARTIFACT_SOURCE_GENERATED,
            "artifact_source_root": str(resolved_root),
        }
        if load_surface_gamma:
            if backend == "jax":
                runtime_spec_json = find_single_file(
                    resolved_root,
                    "single_stage_jax_runtime_spec.json",
                )
                payload["surface_gamma"] = _load_surface_gamma_runtime_spec(
                    str(runtime_spec_json),
                    args,
                )
            else:
                surf_json = find_single_file(resolved_root, "surf_init.json")
                payload["surface_gamma"] = _load_surface_gamma_artifact(str(surf_json))
        return payload


def _single_stage_case_timeout_seconds(
    args: argparse.Namespace,
    *,
    backend: str,
    platform: str,
) -> float:
    target_timeout = float(
        getattr(args, "single_stage_target_case_timeout_seconds", 0.0)
    )
    if backend == "jax" and platform != "cpu" and target_timeout > 0.0:
        return target_timeout
    return float(getattr(args, "single_stage_case_timeout_seconds", 0.0))


def _collect_partial_single_stage_case_artifacts(case_root: Path) -> dict[str, Any]:
    """Return durable artifact pointers for a failed or timed-out case pair."""
    progress_files = sorted(case_root.rglob("outer_optimizer_progress.json"))
    final_payloads = sorted(
        [
            *case_root.rglob("results.json"),
            *case_root.rglob("REJECTED.json"),
        ]
    )
    fatal_logs = sorted(case_root.rglob("fatal_error.log"))
    return {
        "case_artifacts_dir": str(case_root),
        "outer_optimizer_progress_json": [str(path) for path in progress_files],
        "final_payload_json": [str(path) for path in final_payloads],
        "fatal_error_logs": [str(path) for path in fatal_logs],
    }


def _write_case_execution_failure_json(
    output_json: str | os.PathLike[str],
    *,
    provenance: dict[str, Any],
    bundle_provenance: dict[str, Any],
    strict_transfer_support: dict[str, Any],
    error: Exception,
    case_root: Path | None,
) -> None:
    """Write a structured failure when child case execution cannot finish."""
    failure = f"Single-stage case execution failed: {type(error).__name__}: {error}"
    payload: dict[str, Any] = {
        "provenance": provenance,
        "bundle_provenance": bundle_provenance,
        "strict_transfer_support": strict_transfer_support,
        "status": "case-execution-failed",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "warnings": [],
        "failures": [failure],
        "passed": False,
    }
    if case_root is not None:
        payload["artifacts"] = _collect_partial_single_stage_case_artifacts(case_root)
    write_json(output_json, payload)


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
    requested_backend = str(getattr(args, "reference_backend", REFERENCE_BACKEND_AUTO))
    if requested_backend != REFERENCE_BACKEND_AUTO:
        return requested_backend
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
    return bool(
        reference_backend == "cpu"
        and int(args.maxiter) > 0
        and not _has_explicit_single_stage_seed(args)
    )


def _has_explicit_single_stage_seed(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "jax_runtime_seed_spec", None) is not None
        or getattr(args, "warm_start_run_dir", None) is not None
    )


def _is_high_resolution_outer_run(args: argparse.Namespace) -> bool:
    return bool(
        int(args.maxiter) > 0
        and max(int(args.mpol), int(args.ntor)) > MAX_COLD_SEED_OUTER_RUN_RESOLUTION
    )


def _requires_continuation_seed(args: argparse.Namespace) -> bool:
    return bool(
        _is_high_resolution_outer_run(args)
        and getattr(args, "warm_start_run_dir", None) is None
    )


def _json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _require_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    source: Path,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain object field {key!r}")
    return value


def _finite_seed_float(
    value: Any,
    *,
    field_name: str,
    failures: list[str],
) -> float | None:
    if value is None:
        failures.append(f"{field_name} is missing")
        return None
    observed = float(value)
    if not np.isfinite(observed):
        failures.append(f"{field_name} is not finite")
        return None
    return observed


def _resolve_warm_start_seed_contract_G(
    args: argparse.Namespace,
    warm_start_run_dir: Path,
    results: Mapping[str, object],
) -> object:
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    del args
    surface_path, _ = single_stage_example.resolve_single_stage_warm_start_paths(
        warm_start_run_dir
    )
    return single_stage_example.resolve_single_stage_warm_start_G(results, surface_path)


def _append_seed_iota_quality_failures(
    failures: list[str],
    *,
    seed_iota: float | None,
    target_iota: float,
) -> None:
    if seed_iota is None:
        return
    target_abs = abs(float(target_iota))
    if target_abs <= HIGH_RES_SEED_MIN_ABS_IOTA:
        return
    min_abs_iota = max(
        HIGH_RES_SEED_MIN_ABS_IOTA,
        HIGH_RES_SEED_MIN_TARGET_IOTA_FRACTION * target_abs,
    )
    if abs(seed_iota) < min_abs_iota:
        failures.append(
            "seed Boozer iota is not physically relevant for the high-resolution "
            f"rung: |seed_iota|={abs(seed_iota):.6g} is below "
            f"{HIGH_RES_SEED_MIN_TARGET_IOTA_FRACTION:.3g} * "
            f"|target_iota|={target_abs:.6g}"
        )


def _validate_jax_runtime_seed_spec_contract(
    args: argparse.Namespace,
    seed_spec_path: Path,
) -> None:
    payload = _json_mapping(seed_spec_path)
    failures: list[str] = []
    if payload.get("schema") != SINGLE_STAGE_JAX_RUNTIME_SPEC_SCHEMA:
        failures.append(
            f"runtime seed spec schema is not {SINGLE_STAGE_JAX_RUNTIME_SPEC_SCHEMA!r}"
        )
    surface = _require_mapping(payload, "surface", source=seed_spec_path)
    quadrature = _require_mapping(payload, "quadrature", source=seed_spec_path)
    boozer_init = _require_mapping(payload, "boozer_init", source=seed_spec_path)
    observed_shape = (
        int(surface.get("mpol", -1)),
        int(surface.get("ntor", -1)),
        int(quadrature.get("nphi", -1)),
        int(quadrature.get("ntheta", -1)),
    )
    expected_shape = (
        int(args.mpol),
        int(args.ntor),
        int(args.nphi),
        int(args.ntheta),
    )
    if observed_shape != expected_shape:
        failures.append(
            f"runtime seed spec shape {observed_shape} does not match rung "
            f"shape {expected_shape}"
        )
    seed_iota = _finite_seed_float(
        boozer_init.get("iota"),
        field_name="boozer_init.iota",
        failures=failures,
    )
    _finite_seed_float(
        boozer_init.get("G"),
        field_name="boozer_init.G",
        failures=failures,
    )
    _append_seed_iota_quality_failures(
        failures,
        seed_iota=seed_iota,
        target_iota=float(args.iota_target),
    )
    if failures:
        raise ValueError(
            "single_stage_init_parity high-resolution JAX runtime seed contract "
            f"failed for {seed_spec_path}: " + "; ".join(failures)
        )


def _validate_warm_start_seed_contract(
    args: argparse.Namespace,
    warm_start_run_dir: Path,
) -> None:
    results_path = warm_start_run_dir / "results.json"
    results = _json_mapping(results_path)
    failures: list[str] = []
    if results.get("init_only") is True:
        failures.append("warm-start donor is init-only")
    seed_iota = _finite_seed_float(
        results.get("FINAL_IOTA"),
        field_name="FINAL_IOTA",
        failures=failures,
    )
    _append_seed_iota_quality_failures(
        failures,
        seed_iota=seed_iota,
        target_iota=float(args.iota_target),
    )
    if results.get("HARDWARE_CONSTRAINTS_OK") is False:
        failures.append("HARDWARE_CONSTRAINTS_OK is false")
    if results.get("SELF_INTERSECTING") is True:
        failures.append("SELF_INTERSECTING is true")
    seed_g = results.get("FINAL_G")
    if seed_g is None and not failures:
        seed_g = _resolve_warm_start_seed_contract_G(args, warm_start_run_dir, results)
    _finite_seed_float(
        seed_g,
        field_name="FINAL_G or derived runtime G",
        failures=failures,
    )
    if failures:
        raise ValueError(
            "single_stage_init_parity high-resolution warm-start seed contract "
            f"failed for {warm_start_run_dir}: " + "; ".join(failures)
        )


def _require_supported_single_stage_seed_contract(args: argparse.Namespace) -> None:
    if not _is_high_resolution_outer_run(args):
        return
    warm_start_run_dir = getattr(args, "warm_start_run_dir", None)
    if warm_start_run_dir is None:
        raise ValueError(
            "single_stage_init_parity high-resolution outer runs require "
            "--warm-start-run-dir from a validated continuation donor; "
            "--jax-runtime-seed-spec alone is only a runtime startup guess and "
            "does not prove continuation-branch preservation. Build the donor with "
            "examples/single_stage_optimization/SINGLE_STAGE/"
            "run_single_stage_continuation.py. "
            f"Got mpol={int(args.mpol)}, ntor={int(args.ntor)}, "
            f"maxiter={int(args.maxiter)}."
        )
    _validate_warm_start_seed_contract(args, Path(warm_start_run_dir))


def _resolve_target_jax_runtime_seed_spec(
    args: argparse.Namespace,
    *,
    case_root: Path,
) -> Path:
    if (
        _is_high_resolution_outer_run(args)
        and getattr(args, "warm_start_run_dir", None) is not None
    ):
        seed_spec_path = _compile_jax_runtime_seed_spec_from_run_dir(
            Path(args.warm_start_run_dir),
            case_root / "single_stage_jax_runtime_seed_spec.json",
            args,
        )
        _validate_jax_runtime_seed_spec_contract(args, seed_spec_path)
        return seed_spec_path
    if args.jax_runtime_seed_spec is not None:
        return Path(args.jax_runtime_seed_spec)
    return _compile_jax_runtime_seed_spec_from_run_dir(
        Path(args.warm_start_run_dir),
        case_root / "single_stage_jax_runtime_seed_spec.json",
        args,
    )


def _namespace_with_overrides(
    args: argparse.Namespace,
    **overrides: Any,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def _should_run_exact_same_candidate_replay(args: argparse.Namespace) -> bool:
    return (
        bool(getattr(args, "record_objective_evaluation_trace", False))
        and int(args.maxiter) > 0
        and args.optimizer_backend in _EXACT_SAME_CANDIDATE_REPLAY_BACKENDS
    )


def _is_public_optimizer_comparison_backend(optimizer_backend: str) -> bool:
    return optimizer_backend in _PUBLIC_OPTIMIZER_COMPARISON_BACKENDS


def _public_optimizer_trace_required(args: argparse.Namespace) -> bool:
    return int(args.maxiter) > 0 and _is_public_optimizer_comparison_backend(
        args.optimizer_backend
    )


def _target_native_trace_required(args: argparse.Namespace) -> bool:
    return (
        int(args.maxiter) > 0
        and bool(getattr(args, "record_objective_evaluation_trace", False))
        and args.optimizer_backend == TARGET_OPTIMIZER_BACKEND
    )


def _same_candidate_replay_required(args: argparse.Namespace) -> bool:
    return (
        bool(getattr(args, "record_objective_evaluation_trace", False))
        and int(args.maxiter) > 0
    )


def _trace_gated_final_metric_parity(args: argparse.Namespace) -> bool:
    return _public_optimizer_trace_required(args) or _target_native_trace_required(args)


def _strict_transfer_optimizer_support(
    args: argparse.Namespace,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    transfer_guard = provenance.get("transfer_guard")
    if transfer_guard is None:
        transfer_guard = os.environ.get("SIMSOPT_JAX_TRANSFER_GUARD")
    unsupported = bool(
        str(transfer_guard) == "disallow"
        and _target_platform_uses_cuda(args.platform)
        and args.optimizer_backend in _STRICT_TRANSFER_UNSUPPORTED_OPTIMIZER_BACKENDS
    )
    reason = None
    if unsupported:
        reason = (
            "optimistix-lbfgs is a diagnostic backend only under CUDA "
            "strict transfer guard: Optimistix/Equinox scalar predicate "
            "handling performs device-to-host transfer before SIMSOPT can "
            "hostify result metadata."
        )
    return {
        "supported": not unsupported,
        "status": "unsupported" if unsupported else "supported",
        "optimizer_backend": args.optimizer_backend,
        "platform": args.platform,
        "transfer_guard": transfer_guard,
        "reason": reason,
    }


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
    _require_supported_single_stage_seed_contract(args)
    compare_surface_geometry = _should_compare_surface_geometry(
        args,
        benchmark_mode=benchmark_mode,
    )
    reference_case_artifacts_dir = getattr(args, "reference_case_artifacts_dir", None)
    seed_case = None
    same_candidate_replay_case = None
    target_args = args
    if reference_case_artifacts_dir is not None:
        if reference_backend != "cpu":
            raise ValueError(
                "--reference-case-artifacts-dir is only valid with "
                "--reference-backend cpu."
            )
        if not _has_explicit_single_stage_seed(args):
            raise ValueError(
                "--reference-case-artifacts-dir requires --warm-start-run-dir or "
                "--jax-runtime-seed-spec so the JAX target seed is independent of "
                "the loaded CPU reference artifact."
            )
        cpu_case = _load_single_stage_case_from_output_root(
            Path(reference_case_artifacts_dir),
            args,
            backend="cpu",
            load_surface_gamma=compare_surface_geometry,
        )
        jax_seed_spec = _resolve_target_jax_runtime_seed_spec(
            args,
            case_root=case_root,
        )
    elif reference_backend == "jax":
        jax_seed_spec = _resolve_target_jax_runtime_seed_spec(
            args,
            case_root=case_root,
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
            jax_seed_spec = _resolve_target_jax_runtime_seed_spec(
                args,
                case_root=case_root,
            )
    if _should_run_exact_same_candidate_replay(args):
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


def _load_surface_gamma_runtime_spec(
    runtime_spec_path: str,
    args: argparse.Namespace,
) -> np.ndarray:
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

    payload = load_json(runtime_spec_path)
    surface_payload = payload["surface"]
    _require_runtime_spec_surface_grid(surface_payload, args)
    surface = SurfaceXYZTensorFourier(
        nfp=int(surface_payload["nfp"]),
        stellsym=bool(surface_payload["stellsym"]),
        mpol=int(surface_payload["mpol"]),
        ntor=int(surface_payload["ntor"]),
        quadpoints_phi=_runtime_spec_array(surface_payload["quadpoints_phi"]),
        quadpoints_theta=_runtime_spec_array(surface_payload["quadpoints_theta"]),
    )
    surface.x = _runtime_spec_array(surface_payload["dofs"])
    return np.asarray(surface.gamma(), dtype=float)


def _runtime_spec_array(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(payload["data"], dtype=float).reshape(tuple(payload["shape"]))


def _require_runtime_spec_surface_grid(
    surface_payload: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    observed = {
        "mpol": int(surface_payload["mpol"]),
        "ntor": int(surface_payload["ntor"]),
        "nphi": int(np.prod(surface_payload["quadpoints_phi"]["shape"])),
        "ntheta": int(np.prod(surface_payload["quadpoints_theta"]["shape"])),
    }
    expected = {
        "mpol": int(args.mpol),
        "ntor": int(args.ntor),
        "nphi": int(args.nphi),
        "ntheta": int(args.ntheta),
    }
    if observed != expected:
        raise ValueError(
            "JAX runtime surface geometry grid does not match this parity run: "
            f"observed={observed}, expected={expected}."
        )


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


def _load_optimizer_endpoint_trace_from_case(
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    progress_path = Path(case["outer_optimizer_progress_json"])
    if not progress_path.exists():
        return []
    payload = load_json(progress_path)
    return [
        dict(event)
        for event in payload.get("events", [])
        if event.get("label") == "optimizer_endpoint_trace"
    ]


def _optimizer_state_trace_entry_to_path_event(
    entry: dict[str, Any],
    *,
    event_index: int,
) -> dict[str, Any]:
    return {
        "event_index": int(event_index),
        "accepted_iteration_target": entry.get("iteration"),
        "line_search_evaluation": entry.get("nfev"),
        "candidate_optimizer_dofs": entry.get("x"),
        "objective": entry.get("fun"),
        "optimizer_gradient": entry.get("jac"),
        "optimizer_state_trace_event": True,
        "trace_event_source": "optimizer_state_trace",
    }


def _objective_events_to_accepted_path_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted_events: list[dict[str, Any]] = []
    current_iteration = object()
    current_event = None
    for event in events:
        iteration = event.get("accepted_iteration_target")
        if current_event is not None and iteration != current_iteration:
            accepted_events.append(dict(current_event))
        current_iteration = iteration
        current_event = event
    if current_event is not None:
        accepted_events.append(dict(current_event))
    return accepted_events


def _load_optimizer_path_events_pair(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cpu_objective_events = _load_objective_evaluation_events_from_case(cpu_case)
    jax_objective_events = _load_objective_evaluation_events_from_case(jax_case)
    cpu_state_trace = _load_optimizer_state_trace_from_case(cpu_case)
    jax_state_trace = _load_optimizer_state_trace_from_case(jax_case)
    cpu_endpoint_events = _load_optimizer_endpoint_trace_from_case(cpu_case)
    jax_endpoint_events = _load_optimizer_endpoint_trace_from_case(jax_case)
    metadata = {
        "cpu_objective_event_count": len(cpu_objective_events),
        "jax_objective_event_count": len(jax_objective_events),
        "cpu_optimizer_state_trace_count": len(cpu_state_trace),
        "jax_optimizer_state_trace_count": len(jax_state_trace),
        "cpu_optimizer_endpoint_trace_count": len(cpu_endpoint_events),
        "jax_optimizer_endpoint_trace_count": len(jax_endpoint_events),
    }
    if cpu_objective_events and jax_objective_events:
        return (
            cpu_objective_events,
            jax_objective_events,
            {**metadata, "event_source": "objective_evaluation"},
        )
    if cpu_objective_events and jax_state_trace:
        jax_events = [
            _optimizer_state_trace_entry_to_path_event(entry, event_index=index)
            for index, entry in enumerate(jax_state_trace, start=1)
        ]
        return (
            _objective_events_to_accepted_path_events(cpu_objective_events),
            jax_events,
            {**metadata, "event_source": "accepted_step_state_trace"},
        )
    if cpu_state_trace and jax_objective_events:
        cpu_events = [
            _optimizer_state_trace_entry_to_path_event(entry, event_index=index)
            for index, entry in enumerate(cpu_state_trace, start=1)
        ]
        return (
            cpu_events,
            _objective_events_to_accepted_path_events(jax_objective_events),
            {**metadata, "event_source": "accepted_step_state_trace"},
        )
    if cpu_state_trace and jax_state_trace:
        cpu_events = [
            _optimizer_state_trace_entry_to_path_event(entry, event_index=index)
            for index, entry in enumerate(cpu_state_trace, start=1)
        ]
        jax_events = [
            _optimizer_state_trace_entry_to_path_event(entry, event_index=index)
            for index, entry in enumerate(jax_state_trace, start=1)
        ]
        return (
            cpu_events,
            jax_events,
            {**metadata, "event_source": "optimizer_state_trace"},
        )
    if cpu_objective_events and jax_endpoint_events:
        return (
            _objective_events_to_accepted_path_events(cpu_objective_events),
            jax_endpoint_events,
            {**metadata, "event_source": "optimizer_endpoint_trace"},
        )
    if cpu_endpoint_events and jax_objective_events:
        return (
            cpu_endpoint_events,
            _objective_events_to_accepted_path_events(jax_objective_events),
            {**metadata, "event_source": "optimizer_endpoint_trace"},
        )
    if cpu_endpoint_events and jax_endpoint_events:
        return (
            cpu_endpoint_events,
            jax_endpoint_events,
            {**metadata, "event_source": "optimizer_endpoint_trace"},
        )
    return [], [], {**metadata, "event_source": "not-recorded"}


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


def _summary_matrix(summary: dict[str, Any] | None) -> np.ndarray | None:
    """Reshape a flattened ``_summarize_host_array`` payload back to 2D.

    Used by the scientific-equivalence ladder solve-quality probes (gates L4
    / E3) to recover the dense Hessian / Jacobian operator from a captured
    parity artifact. ``shape`` is required for matrix recovery; vectors and
    higher-rank tensors return ``None``. Non-finite payloads also return
    ``None`` so the probe wiring can skip cleanly without injecting NaNs.
    """
    if summary is None or not bool(summary.get("all_finite", False)):
        return None
    values = summary.get("values")
    shape = summary.get("shape")
    if values is None or shape is None:
        return None
    shape_tuple = tuple(int(dim) for dim in shape)
    if len(shape_tuple) != 2:
        return None
    return np.asarray(values, dtype=float).reshape(shape_tuple)


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _same_candidate_comparable_vectors(
    *,
    cpu_vector: np.ndarray | None,
    jax_vector: np.ndarray | None,
    target_native_replay_event: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return vector views for same-candidate comparisons.

    Target-native replay can record an objective boundary whose optimizer vector is
    the coil-DOF prefix. CPU replay records the full SIMSOPT optimizer vector. For
    that one contract, compare the CPU prefix to the target vector.
    """
    if cpu_vector is None or jax_vector is None:
        return cpu_vector, jax_vector, "full-optimizer-vector"
    if (
        target_native_replay_event
        and cpu_vector.shape != jax_vector.shape
        and cpu_vector.size >= jax_vector.size
    ):
        return (
            cpu_vector[: jax_vector.size],
            jax_vector,
            "target-native-cpu-prefix",
        )
    return cpu_vector, jax_vector, "full-optimizer-vector"


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
    if not diff <= (atol + rtol * abs(float(cpu_value))):
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


def _same_candidate_rejected_by_contract(event: dict[str, Any]) -> bool:
    return not bool(event.get("native_gradient_used"))


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


def _record_first_scipy_callback_split(
    current,
    split_field,
    index,
    cpu_entry,
    jax_entry,
    max_diff,
):
    if current is not None:
        return current
    return {
        "field": split_field,
        "callback_index": index,
        "cpu_evaluation_index": cpu_entry.get("evaluation_index"),
        "jax_evaluation_index": jax_entry.get("evaluation_index"),
        "max_abs_diff": max_diff,
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
            first_split = _record_first_scipy_callback_split(
                first_split, "evaluation_index", index, cpu_entry, jax_entry, 0.0
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
        if decision_diff > 0.0:
            first_split = _record_first_scipy_callback_split(
                first_split,
                "decision_vector",
                index,
                cpu_entry,
                jax_entry,
                decision_diff,
            )

        cpu_fun = _summary_scalar(cpu_entry.get("fun"))
        jax_fun = _summary_scalar(jax_entry.get("fun"))
        failure_count = len(failures)
        fun_diff = _compare_same_candidate_scalar(
            failures,
            field=f"{entry_field}.fun",
            cpu_value=cpu_fun,
            jax_value=jax_fun,
        )
        max_diff = max(max_diff, fun_diff)
        if (
            cpu_fun is not None
            and jax_fun is not None
            and len(failures) != failure_count
        ):
            first_split = _record_first_scipy_callback_split(
                first_split, "fun", index, cpu_entry, jax_entry, fun_diff
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
        if gradient_reference is not None and gradient_diff > (
            _SAME_CANDIDATE_GRADIENT_ATOL
            + _SAME_CANDIDATE_GRADIENT_RTOL * gradient_reference
        ):
            first_split = _record_first_scipy_callback_split(
                first_split, "gradient", index, cpu_entry, jax_entry, gradient_diff
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
            "max_slice_pair_index": None,
            "max_slice_line_search_evaluation": None,
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
    has_slice_owner = objective_owner or gradient_owner
    return {
        "max_slice_objective_abs_diff": max_objective,
        "max_slice_gradient_abs_diff": max_gradient,
        "max_slice_objective_owner": objective_owner,
        "max_slice_gradient_owner": gradient_owner,
        "max_slice_pair_index": pair_index if has_slice_owner else None,
        "max_slice_line_search_evaluation": line_search_evaluation
        if has_slice_owner
        else None,
    }


def _nested_payload_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if value is None:
            return None
        value = value.get(key)
    return value


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


@dataclass
class LayerDriftTracker:
    """Tracks drift maxima and first divergence for one layer family."""

    max_abs_diff: float = 0.0
    max_layer: str | None = None
    max_pair_index: int | None = None
    max_line_search_evaluation: Any = None
    max_layer_diffs: dict[str, float] = dataclass_field(default_factory=dict)
    first_divergence: dict[str, Any] | None = None

    def update(
        self,
        summary: dict[str, Any],
        *,
        pair_index: int,
        line_search_evaluation: Any,
    ) -> dict[str, Any] | None:
        if summary["max_abs_diff"] > self.max_abs_diff:
            self.max_abs_diff = summary["max_abs_diff"]
            self.max_layer = summary["max_layer"]
            self.max_pair_index = summary["pair_index"]
            self.max_line_search_evaluation = summary["line_search_evaluation"]
            self.max_layer_diffs = dict(summary["layer_diffs"])
        if (
            self.first_divergence is None
            and summary["first_divergent_layer"] is not None
        ):
            self.first_divergence = {
                "pair_index": pair_index,
                "line_search_evaluation": line_search_evaluation,
                "layer": summary["first_divergent_layer"],
                "layer_diffs": dict(summary["layer_diffs"]),
            }
            return self.first_divergence
        return None


def _first_parity_bug_census_divergence(current, family, divergence):
    if current is not None or divergence is None:
        return current
    payload = {"family": family, **divergence}
    payload["layer_diffs"] = dict(divergence["layer_diffs"])
    return payload


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
            if cpu_summary is None and jax_summary is None:
                field_diff = 0.0
            elif kind == "scalar":
                field_diff = _path_scalar_abs_diff(cpu_summary, jax_summary)
            else:
                field_diff = _path_vector_abs_diff(cpu_summary, jax_summary)
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


def _compute_solve_quality_probe_pair(
    *,
    cpu_decomposition: dict[str, Any] | None,
    jax_decomposition: dict[str, Any] | None,
    artifact_name: str,
) -> dict[str, float | None]:
    """Compute the LS L4 / Exact E3 operator-action max-rel-error for one pair.

    Per ``docs/parity_scientific_equivalence_contract_2026-05-09.md`` §4 and
    §2 (gates L4 + E3). Reads the captured ``final_hessian`` from each
    paired ``boozer_solve_decomposition`` (the LS branch supplies a square
    Hessian); when both lanes provide a 2D matrix of matching shape the
    deterministic probe set is applied and the maximum relative error is
    reported. Returns ``None`` for fields that cannot be computed (missing
    summary, shape mismatch, or non-2D shape) so the parity arbiter can
    record reporting-only metrics without injecting NaNs.

    Phase 1.5 is reporting-only: callers must NOT extend the failure list
    on the returned values until the calibration sweep in §10 risk register
    item 4 finishes and the §2 tolerance schedule is locked.
    """
    metrics: dict[str, float | None] = {
        "ls_hessian_action_max_rel": None,
        "exact_jacobian_action_max_rel": None,
    }
    if cpu_decomposition is None or jax_decomposition is None:
        return metrics

    cpu_hessian = _summary_matrix(cpu_decomposition.get("final_hessian"))
    jax_hessian = _summary_matrix(jax_decomposition.get("final_hessian"))
    if (
        cpu_hessian is not None
        and jax_hessian is not None
        and cpu_hessian.shape == jax_hessian.shape
        and cpu_hessian.ndim == 2
        and cpu_hessian.shape[0] == cpu_hessian.shape[1]
    ):
        metrics["ls_hessian_action_max_rel"] = (
            compute_dense_operator_action_max_rel_error(
                jax_hessian,
                cpu_hessian,
                artifact_name=f"{artifact_name}/ls_hessian",
            )
        )

    cpu_jacobian = _summary_matrix(cpu_decomposition.get("final_jacobian"))
    jax_jacobian = _summary_matrix(jax_decomposition.get("final_jacobian"))
    if (
        cpu_jacobian is not None
        and jax_jacobian is not None
        and cpu_jacobian.shape == jax_jacobian.shape
        and cpu_jacobian.ndim == 2
    ):
        metrics["exact_jacobian_action_max_rel"] = (
            compute_dense_operator_action_max_rel_error(
                jax_jacobian,
                cpu_jacobian,
                artifact_name=f"{artifact_name}/exact_jacobian",
            )
        )
    return metrics


def _aggregate_solve_quality_probes(
    aggregate: dict[str, float | None],
    *,
    pair_metrics: dict[str, float | None],
    pair_index: int,
) -> None:
    """Update the fixture-level solve-quality probe aggregate with one pair.

    The aggregate carries the *maximum* observed value per gate plus the
    pair index where that maximum was seen. Reporting-only: callers must not
    flag failures based on these aggregates until the calibration sweep
    locks the §2 tolerance schedule.
    """
    for field in ("ls_hessian_action_max_rel", "exact_jacobian_action_max_rel"):
        value = pair_metrics.get(field)
        if value is None:
            continue
        previous = aggregate.get(field)
        if previous is None or float(value) > float(previous):
            aggregate[field] = float(value)
            aggregate[f"{field}_pair_index"] = pair_index


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
    if same_candidate_replay["status"] == "not-applicable":
        return failures
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
    if same_candidate_replay.get("diagnostic_scope") == (
        "target-native-objective-gradient"
    ):
        return failures
    parity_bug_census = same_candidate_replay.get("parity_bug_census")
    if not parity_bug_census or parity_bug_census.get("status") != "recorded":
        failures.append(
            "Same-candidate objective replay did not record a parity bug census."
        )
    failures.extend(_pre_newton_census_gate_failures(parity_bug_census))
    return failures


def _same_candidate_replay_strict_gate_failure_class(
    same_candidate_replay: dict[str, Any],
) -> str | None:
    if same_candidate_replay["status"] != "pass":
        return None
    parity_bug_census = same_candidate_replay.get("parity_bug_census")
    if _pre_newton_census_gate_failures(parity_bug_census):
        return "strict_pre_newton_census_only"
    return None


def compare_same_candidate_objective_replay(
    cpu_case: dict[str, Any],
    jax_case: dict[str, Any],
    *,
    require_exact_candidates: bool = False,
    strict_solver_contract: bool = False,
    scalar_rtol: float = _SAME_CANDIDATE_SCALAR_RTOL,
    scalar_atol: float = _SAME_CANDIDATE_SCALAR_ATOL,
    gradient_rtol: float = _SAME_CANDIDATE_GRADIENT_RTOL,
    gradient_atol: float = _SAME_CANDIDATE_GRADIENT_ATOL,
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
            "scalar_rtol": float(scalar_rtol),
            "scalar_atol": float(scalar_atol),
            "gradient_rtol": float(gradient_rtol),
            "gradient_atol": float(gradient_atol),
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
    iota_decomposition_tracker = LayerDriftTracker()
    boozer_solve_decomposition_tracker = LayerDriftTracker()
    max_boozer_scipy_callback_abs_diff = 0.0
    first_boozer_scipy_callback_split = None
    # Phase 1.5 reporting-only solve-quality probe aggregate per
    # docs/parity_scientific_equivalence_contract_2026-05-09.md §2 + §4.
    # Carries fixture-level max(L4 / E3) values together with the pair index
    # where each maximum was observed. Calibration-locked enforcement is
    # gated on §10 risk register item 4 — these values stay reporting-only
    # until the calibration corpus completes.
    solve_quality_probe_aggregate: dict[str, float | None] = {
        "ls_hessian_action_max_rel": None,
        "ls_hessian_action_max_rel_pair_index": None,
        "exact_jacobian_action_max_rel": None,
        "exact_jacobian_action_max_rel_pair_index": None,
    }
    parity_bug_census_layers: dict[str, dict[str, Any]] = {}
    first_parity_bug_census_divergence = None
    solver_contract_diagnostics: list[str] = []
    target_native_rejected_contract_diagnostics: list[dict[str, Any]] = []
    same_candidate_event_count = 0
    target_native_replay_event_count = 0
    target_native_rejected_event_count = 0
    candidate_comparison_scope_counts: Counter[str] = Counter()
    gradient_comparison_scope_counts: Counter[str] = Counter()
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
        target_native_replay_event = bool(jax_event.get("target_native_replay", False))
        cpu_event_index = cpu_event.get("event_index")
        jax_event_index = jax_event.get("event_index")
        accepted_iteration_target = cpu_event.get("accepted_iteration_target")
        line_search_evaluation = cpu_event.get("line_search_evaluation")
        (
            comparable_cpu_x,
            comparable_jax_x,
            candidate_comparison_scope,
        ) = _same_candidate_comparable_vectors(
            cpu_vector=cpu_x,
            jax_vector=jax_x,
            target_native_replay_event=target_native_replay_event,
        )
        candidate_comparison_scope_counts[candidate_comparison_scope] += 1
        candidate_abs_diff = _max_abs_diff(comparable_jax_x, comparable_cpu_x)
        max_candidate_abs_diff = max(max_candidate_abs_diff, candidate_abs_diff)
        if candidate_abs_diff > candidate_x_abs_tol:
            if require_exact_candidates:
                event_failures = [
                    "candidate_optimizer_dofs mismatch under exact replay: "
                    f"max_abs_diff={candidate_abs_diff:.3e}, "
                    f"scope={candidate_comparison_scope}."
                ]
                if first_failure_event is None:
                    first_failure_event = {
                        "pair_index": pair_index,
                        "cpu_event_index": cpu_event_index,
                        "jax_event_index": jax_event_index,
                        "accepted_iteration_target": accepted_iteration_target,
                        "line_search_evaluation": line_search_evaluation,
                        "candidate_abs_diff": candidate_abs_diff,
                        "candidate_comparison_scope": candidate_comparison_scope,
                        "failures": list(event_failures),
                    }
                failures.extend(
                    f"pair {pair_index}: {failure}" for failure in event_failures
                )
            continue
        same_candidate_event_count += 1
        if target_native_replay_event:
            target_native_replay_event_count += 1
        cpu_rejected_by_contract = _same_candidate_rejected_by_contract(cpu_event)
        jax_rejected_by_contract = _same_candidate_rejected_by_contract(jax_event)
        target_native_rejected_event = (
            target_native_replay_event
            and cpu_rejected_by_contract
            and jax_rejected_by_contract
        )
        if target_native_rejected_event:
            target_native_rejected_event_count += 1
            cpu_failure = cpu_event.get("candidate_failure") or {}
            target_native_rejected_contract_diagnostics.append(
                {
                    "pair_index": pair_index,
                    "cpu_event_index": cpu_event_index,
                    "jax_event_index": jax_event_index,
                    "accepted_iteration_target": accepted_iteration_target,
                    "line_search_evaluation": line_search_evaluation,
                    "cpu_reject_class": cpu_failure.get("reject_class"),
                    "cpu_solver_success": cpu_event.get("solver_success"),
                    "jax_solver_success": jax_event.get("solver_success"),
                }
            )
        event_failures: list[str] = []
        if target_native_replay_event:
            if cpu_rejected_by_contract != jax_rejected_by_contract:
                event_failures.append(
                    "target-native replay rejection mismatch: "
                    f"cpu_native_gradient_used={cpu_event.get('native_gradient_used')}, "
                    f"jax_native_gradient_used={jax_event.get('native_gradient_used')}."
                )
        else:
            _compare_same_candidate_exact_event_field(
                event_failures,
                field="native_gradient_used",
                cpu_event=cpu_event,
                jax_event=jax_event,
            )
        if not target_native_rejected_event:
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
                "cpu_event_index": cpu_event_index,
                "jax_event_index": jax_event_index,
                "accepted_iteration_target": accepted_iteration_target,
                "line_search_evaluation": line_search_evaluation,
                **boozer_metadata_summary["first_scipy_callback_split"],
            }
        compare_native_gradient_layers = (
            not cpu_rejected_by_contract and not jax_rejected_by_contract
        )
        compare_native_gradient_diagnostics = (
            compare_native_gradient_layers and not target_native_replay_event
        )
        cpu_boozer_solve_decomposition = (
            cpu_event.get("boozer_solve_decomposition")
            if compare_native_gradient_diagnostics
            else None
        )
        jax_boozer_solve_decomposition = (
            jax_event.get("boozer_solve_decomposition")
            if compare_native_gradient_diagnostics
            else None
        )
        boozer_solve_decomposition_summary = (
            _compare_same_candidate_layer_decomposition(
                event_failures,
                field_name="boozer_solve_decomposition",
                layer_fields=_BOOZER_SOLVE_DECOMPOSITION_LAYER_FIELDS,
                cpu_decomposition=cpu_boozer_solve_decomposition,
                jax_decomposition=jax_boozer_solve_decomposition,
                pair_index=pair_index,
                line_search_evaluation=line_search_evaluation,
            )
        )
        if compare_native_gradient_diagnostics:
            solve_quality_pair_metrics = _compute_solve_quality_probe_pair(
                cpu_decomposition=cpu_boozer_solve_decomposition,
                jax_decomposition=jax_boozer_solve_decomposition,
                artifact_name=f"single_stage_init_parity/pair{pair_index}",
            )
            _aggregate_solve_quality_probes(
                solve_quality_probe_aggregate,
                pair_metrics=solve_quality_pair_metrics,
                pair_index=pair_index,
            )
        _update_parity_bug_census(
            parity_bug_census_layers,
            family="boozer_solve",
            summary=boozer_solve_decomposition_summary,
        )
        boozer_solve_divergence = boozer_solve_decomposition_tracker.update(
            boozer_solve_decomposition_summary,
            pair_index=pair_index,
            line_search_evaluation=line_search_evaluation,
        )
        first_parity_bug_census_divergence = _first_parity_bug_census_divergence(
            first_parity_bug_census_divergence,
            "boozer_solve",
            boozer_solve_divergence,
        )
        if not target_native_rejected_event:
            (
                comparable_cpu_gradient,
                comparable_jax_gradient,
                gradient_comparison_scope,
            ) = _same_candidate_comparable_vectors(
                cpu_vector=_summary_vector(cpu_event.get("optimizer_gradient")),
                jax_vector=_summary_vector(jax_event.get("optimizer_gradient")),
                target_native_replay_event=target_native_replay_event,
            )
            gradient_comparison_scope_counts[gradient_comparison_scope] += 1
            max_objective_abs_diff = max(
                max_objective_abs_diff,
                _compare_same_candidate_scalar(
                    event_failures,
                    field="objective.value",
                    cpu_value=_summary_scalar(cpu_event.get("objective")),
                    jax_value=_summary_scalar(jax_event.get("objective")),
                    rtol=scalar_rtol,
                    atol=scalar_atol,
                ),
            )
            max_gradient_abs_diff = max(
                max_gradient_abs_diff,
                _compare_same_candidate_vector(
                    event_failures,
                    field="optimizer_gradient",
                    cpu_vector=comparable_cpu_gradient,
                    jax_vector=comparable_jax_gradient,
                    rtol=gradient_rtol,
                    atol=gradient_atol,
                ),
            )
        slice_summary = _compare_same_candidate_objective_components(
            event_failures,
            cpu_components=(
                None
                if target_native_replay_event
                else cpu_event.get("objective_components")
            ),
            jax_components=(
                None
                if target_native_replay_event
                else jax_event.get("objective_components")
            ),
            pair_index=pair_index,
            line_search_evaluation=line_search_evaluation,
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
        iota_decomposition_summary = (
            _layer_decomposition_summary(recorded=False)
            if target_native_replay_event
            else _compare_same_candidate_layer_decomposition(
                event_failures,
                field_name="iota_penalty_decomposition",
                layer_fields=_IOTA_DECOMPOSITION_LAYER_FIELDS,
                cpu_decomposition=cpu_event.get("iota_penalty_decomposition"),
                jax_decomposition=jax_event.get("iota_penalty_decomposition"),
                pair_index=pair_index,
                line_search_evaluation=line_search_evaluation,
            )
        )
        _update_parity_bug_census(
            parity_bug_census_layers,
            family="iota_penalty",
            summary=iota_decomposition_summary,
        )
        iota_divergence = iota_decomposition_tracker.update(
            iota_decomposition_summary,
            pair_index=pair_index,
            line_search_evaluation=line_search_evaluation,
        )
        first_parity_bug_census_divergence = _first_parity_bug_census_divergence(
            first_parity_bug_census_divergence,
            "iota_penalty",
            iota_divergence,
        )
        if compare_native_gradient_layers:
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
        if not target_native_replay_event:
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
                    "cpu_event_index": cpu_event_index,
                    "jax_event_index": jax_event_index,
                    "accepted_iteration_target": accepted_iteration_target,
                    "line_search_evaluation": line_search_evaluation,
                    "candidate_abs_diff": candidate_abs_diff,
                    "candidate_comparison_scope": candidate_comparison_scope,
                    "failures": list(event_failures),
                }
            failures.extend(
                f"pair {pair_index}: {failure}" for failure in event_failures
            )
    if same_candidate_event_count == 0:
        failures.append(
            "No paired objective-evaluation events shared the same candidate."
        )
    diagnostic_scope = (
        "target-native-objective-gradient"
        if same_candidate_event_count > 0
        and target_native_replay_event_count == same_candidate_event_count
        else "full-objective-trace"
    )
    parity_bug_census: dict[str, Any]
    if diagnostic_scope == "target-native-objective-gradient":
        parity_bug_census = {
            "status": "not-applicable",
            "reason": "target_native_replay_records_value_gradient_and_solved_state",
            "first_divergence": None,
            "divergent_layer_count": 0,
            "divergent_layers": [],
            "max_layer_diffs": {},
        }
    else:
        parity_bug_census_records = list(parity_bug_census_layers.values())
        divergent_layers = [
            dict(entry)
            for entry in sorted(
                parity_bug_census_records,
                key=lambda item: float(item["max_abs_diff"]),
                reverse=True,
            )
            if bool(entry["diverged"])
        ]
        parity_bug_census = {
            "status": "recorded" if parity_bug_census_records else "not-recorded",
            "first_divergence": first_parity_bug_census_divergence,
            "divergent_layer_count": len(divergent_layers),
            "divergent_layers": divergent_layers,
            "max_layer_diffs": {
                f"{entry['family']}.{entry['layer']}": entry["max_abs_diff"]
                for entry in parity_bug_census_records
            },
        }
    return {
        "status": "pass" if not failures else "fail",
        "diagnostic_scope": diagnostic_scope,
        "cpu_event_count": len(cpu_events),
        "jax_event_count": len(jax_events),
        "same_candidate_event_count": same_candidate_event_count,
        "target_native_replay_event_count": target_native_replay_event_count,
        "target_native_rejected_event_count": target_native_rejected_event_count,
        "candidate_comparison_scope_counts": dict(candidate_comparison_scope_counts),
        "gradient_comparison_scope_counts": dict(gradient_comparison_scope_counts),
        "target_native_rejected_contract_diagnostics": (
            target_native_rejected_contract_diagnostics
        ),
        "require_exact_candidates": bool(require_exact_candidates),
        "strict_solver_contract": bool(strict_solver_contract),
        "scalar_rtol": float(scalar_rtol),
        "scalar_atol": float(scalar_atol),
        "gradient_rtol": float(gradient_rtol),
        "gradient_atol": float(gradient_atol),
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
        "max_iota_decomposition_abs_diff": iota_decomposition_tracker.max_abs_diff,
        "max_iota_decomposition_layer": iota_decomposition_tracker.max_layer,
        "max_iota_decomposition_pair_index": iota_decomposition_tracker.max_pair_index,
        "max_iota_decomposition_line_search_evaluation": (
            iota_decomposition_tracker.max_line_search_evaluation
        ),
        "max_iota_decomposition_layer_diffs": (
            iota_decomposition_tracker.max_layer_diffs
        ),
        "first_iota_decomposition_divergence": (
            iota_decomposition_tracker.first_divergence
        ),
        "max_boozer_solve_decomposition_abs_diff": (
            boozer_solve_decomposition_tracker.max_abs_diff
        ),
        "max_boozer_solve_decomposition_layer": (
            boozer_solve_decomposition_tracker.max_layer
        ),
        "max_boozer_solve_decomposition_pair_index": (
            boozer_solve_decomposition_tracker.max_pair_index
        ),
        "max_boozer_solve_decomposition_line_search_evaluation": (
            boozer_solve_decomposition_tracker.max_line_search_evaluation
        ),
        "max_boozer_solve_decomposition_layer_diffs": (
            boozer_solve_decomposition_tracker.max_layer_diffs
        ),
        "first_boozer_solve_decomposition_divergence": (
            boozer_solve_decomposition_tracker.first_divergence
        ),
        # Phase 1.5 reporting-only solve-quality probe slot per
        # docs/parity_scientific_equivalence_contract_2026-05-09.md §2 + §4.
        # Field-level None means "not computed" (e.g. native_gradient was
        # not used, or the lane omitted the dense Hessian / Jacobian
        # capture). Promoted to enforcing only after the §10 risk register
        # #4 calibration sweep completes and the §2 schedule is locked.
        "solve_quality_probes": dict(solve_quality_probe_aggregate),
        "parity_bug_census": parity_bug_census,
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
    cpu_events, jax_events, event_metadata = _load_optimizer_path_events_pair(
        cpu_case,
        jax_case,
    )
    if not cpu_events or not jax_events:
        return {
            "status": "not-recorded",
            "cpu_event_count": len(cpu_events),
            "jax_event_count": len(jax_events),
            "paired_event_count": 0,
            "candidate_split_abs_tol": _OPTIMIZER_PATH_CANDIDATE_SPLIT_ATOL,
            **event_metadata,
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
        **event_metadata,
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


def _metric_parity_failures(
    *,
    label: str,
    iota_abs_diff: float,
    volume_rel_diff: float,
    field_error_rel_diff: float,
) -> list[str]:
    """Tolerance check shared by the final-state and seed-state parity gates."""
    failures: list[str] = []
    if iota_abs_diff >= IOTA_ABS_TOL:
        failures.append(f"{label} iota disagreement too large: {iota_abs_diff:.2e}")
    if volume_rel_diff >= VOLUME_REL_TOL:
        failures.append(
            f"{label} volume relative difference too large: {volume_rel_diff:.2e}"
        )
    if field_error_rel_diff >= FIELD_ERROR_REL_TOL:
        failures.append(
            f"{label} field error relative difference too large: {field_error_rel_diff:.2e}"
        )
    return failures


def _final_metric_parity_failures(comparison: dict[str, Any]) -> list[str]:
    return _metric_parity_failures(
        label="Final",
        iota_abs_diff=comparison["final_iota_abs_diff"],
        volume_rel_diff=comparison["final_volume_rel_diff"],
        field_error_rel_diff=comparison["field_error_rel_diff"],
    )


def _initial_metric_parity_failures(comparison: dict[str, Any]) -> list[str]:
    """Convergence-independent seed-state parity.

    Both lanes evaluate the INITIAL (seed) surface at the identical seed DOFs,
    before any outer optimizer runs, so agreement here positively proves the JAX
    port reproduces the C++/CPU reference at a fixed state regardless of optimizer
    convergence. This is the always-armed backstop behind the (skippable)
    final-state gate.
    """
    return _metric_parity_failures(
        label="Initial seed-state",
        iota_abs_diff=comparison["initial_iota_abs_diff"],
        volume_rel_diff=comparison["initial_volume_rel_diff"],
        field_error_rel_diff=comparison["initial_field_error_rel_diff"],
    )


# scipy L-BFGS-B ``status``: 0 converged, 1 hit the iteration budget, 2 abnormal
# termination (line-search failure / no progress). Only status 2 means a lane
# aborted onto a non-optimum; status 1 is a normal full-budget stop. The
# end-state-parity gate's active path (require_final_metric_parity, i.e. the
# scipy-jax / scipy-jax-fullgraph production lanes) runs scipy L-BFGS-B.
_ABNORMAL_OPTIMIZER_STATUS = 2
_STATIONARY_NO_STEP_JAC_INF_NORM_TOL = 1.0e-12


def _optimizer_aborted_abnormally(results: dict[str, Any]) -> bool:
    """True if the lane's outer optimizer terminated abnormally (or did not run).

    ``OPTIMIZER_STATUS is None`` is the producer's marker for "optimizer did not
    run"; at maxiter>0 that is anomalous, so it is treated as not-a-usable-optimum.
    """
    status = results.get("OPTIMIZER_STATUS")
    return status is None or int(status) == _ABNORMAL_OPTIMIZER_STATUS


def _optimizer_stationary_without_accepted_step(
    results: dict[str, Any],
    *,
    iterations: int,
) -> bool:
    status = results.get("OPTIMIZER_STATUS")
    if iterations != 0 or status is None or int(status) != 0:
        return False
    jac_inf_norm = results.get("OPTIMIZER_JAC_INF_NORM")
    if jac_inf_norm is None:
        return False
    jac_inf_norm = float(jac_inf_norm)
    return bool(
        np.isfinite(jac_inf_norm)
        and jac_inf_norm <= _STATIONARY_NO_STEP_JAC_INF_NORM_TOL
    )


def _optimizer_control_split_accepts_final_metric_drift(
    *,
    same_candidate_replay: dict[str, Any] | None,
    optimizer_path_objective_evaluations: dict[str, Any] | None,
) -> bool:
    if same_candidate_replay is None or same_candidate_replay["status"] != "pass":
        return False
    if (
        optimizer_path_objective_evaluations is not None
        and optimizer_path_objective_evaluations["status"] == "split"
    ):
        return True
    return False


def evaluate_single_stage_init_parity(
    cpu_results: dict[str, Any],
    jax_results: dict[str, Any],
    *,
    max_surface_geometry_abs: float,
    max_surface_geometry_rel: float,
    maxiter: int = DEFAULT_OUTER_MAXITER,
    expected_jax_outer_optimizer_method: str = _TARGET_OUTER_OPTIMIZER_METHOD,
    require_final_metric_parity: bool = True,
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
        "initial_iota_abs_diff": abs(
            float(jax_results["INITIAL_IOTA"]) - float(cpu_results["INITIAL_IOTA"])
        ),
        "initial_volume_rel_diff": relative_error(
            float(jax_results["INITIAL_VOLUME"]),
            float(cpu_results["INITIAL_VOLUME"]),
        ),
        "initial_field_error_rel_diff": relative_error(
            float(jax_results["INITIAL_FIELD_ERROR"]),
            float(cpu_results["INITIAL_FIELD_ERROR"]),
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

    final_metric_failures = _final_metric_parity_failures(comparison)
    comparison["final_metric_parity_required"] = bool(require_final_metric_parity)
    comparison["final_metric_parity_failures"] = final_metric_failures

    # End-state metric parity compares two optimizer end-states; it is a valid
    # port-correctness signal only when each lane reached a usable optimum. A lane
    # that terminated abnormally (status 2: line-search failure / no progress --
    # the CPU reference stalling at iota~0.0035 in 39 nfev) lands on a non-optimum.
    # Skip the end-state failure ONLY when the REFERENCE aborted while the JAX
    # TARGET did not: a JAX-target abnormal stop is itself a port-relevant signal
    # and stays a hard failure, so a broken target can never hide behind this skip.
    # status 1 (hit the iteration budget) is a normal terminal state and stays
    # strictly compared. maxiter<=0 runs no outer optimizer -- nothing to gate.
    if maxiter > 0:
        comparison["cpu_optimizer_status"] = cpu_results.get("OPTIMIZER_STATUS")
        comparison["jax_optimizer_status"] = jax_results.get("OPTIMIZER_STATUS")
        comparison["cpu_optimizer_success"] = bool(cpu_results.get("OPTIMIZER_SUCCESS"))
        comparison["jax_optimizer_success"] = bool(jax_results.get("OPTIMIZER_SUCCESS"))
        comparison["cpu_stationary_no_step"] = (
            _optimizer_stationary_without_accepted_step(
                cpu_results,
                iterations=comparison["cpu_iterations"],
            )
        )
        comparison["jax_stationary_no_step"] = (
            _optimizer_stationary_without_accepted_step(
                jax_results,
                iterations=comparison["jax_iterations"],
            )
        )
        skip_for_reference_nonconvergence = _optimizer_aborted_abnormally(
            cpu_results
        ) and not _optimizer_aborted_abnormally(jax_results)
    else:
        skip_for_reference_nonconvergence = False

    failures: list[str] = []
    if require_final_metric_parity and final_metric_failures:
        if skip_for_reference_nonconvergence:
            comparison["final_metric_parity_skipped_for_nonconvergence"] = True
            comparison["skipped_final_metric_parity_failures"] = final_metric_failures
        else:
            failures.extend(final_metric_failures)

    # Convergence-independent seed-state parity backstop: ALWAYS armed (no skip,
    # not gated by require_final_metric_parity). Both lanes evaluate the seed
    # surface at identical DOFs before the outer optimizer runs, so this proves
    # the JAX port reproduces the reference at a fixed state even when the
    # final-state gate is skipped for reference non-convergence.
    initial_metric_failures = _initial_metric_parity_failures(comparison)
    comparison["initial_metric_parity_failures"] = initial_metric_failures
    failures.extend(initial_metric_failures)

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
        if comparison["cpu_iterations"] < 1 and not comparison.get(
            "cpu_stationary_no_step", False
        ):
            failures.append(
                "CPU single-stage outer-loop probe did not accept an optimizer step."
            )
        if comparison["jax_iterations"] < 1 and not comparison.get(
            "jax_stationary_no_step", False
        ):
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
            "target_lane_boozer_newton_polish_policy": (
                _requested_target_lane_boozer_newton_polish_policy(args)
            ),
            "outer_maxiter": int(args.maxiter),
            "command_argv": [sys.executable, *sys.argv],
            "benchmark_mode": benchmark_mode,
            **benchmark_timing_label(
                MIXED_PARITY_REFERENCE,
                includes_gpu_target=args.platform == "cuda",
                includes_cpu_reference=True,
                supports_performance_headline=False,
                headline_timing_classification=None,
                note=(
                    "single_stage_init_parity.py co-produces the JAX target "
                    "lane and cpp/CPU reference; whole-run timing is mixed and "
                    "not a clean GPU headline."
                ),
            ),
            "reference_backend": reference_backend,
            "reference_case_artifacts_dir": None
            if args.reference_case_artifacts_dir is None
            else _display_path(Path(args.reference_case_artifacts_dir)),
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
            "parent_cuda_memory_isolated": bool(args.platform == "cuda"),
            "child_cuda_memory_env": dict(CHILD_CUDA_MEMORY_ENV),
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
    strict_transfer_support = _strict_transfer_optimizer_support(args, provenance)
    if strict_transfer_support["status"] == "unsupported":
        warnings = [str(strict_transfer_support["reason"])]
        payload = {
            "provenance": provenance,
            "bundle_provenance": bundle_provenance,
            "strict_transfer_support": strict_transfer_support,
            "warnings": warnings,
            "failures": [],
            "passed": False,
            "status": "unsupported",
        }
        write_json(args.output_json, payload)
        print("SINGLE-STAGE INIT PARITY UNSUPPORTED")
        for warning in warnings:
            print(f"  - {warning}")
        return

    case_artifacts_dir = (
        None if args.case_artifacts_dir is None else Path(args.case_artifacts_dir)
    )
    case_root_for_failure: Path | None = None
    try:
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
            case_root_for_failure = case_artifacts_dir
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
            if args.reference_case_artifacts_dir is not None:
                case_artifacts["reference_case_artifacts_dir"] = str(
                    Path(args.reference_case_artifacts_dir)
                )
            if same_candidate_replay_case is not None:
                case_artifacts["target_same_candidate_replay_run_dir"] = str(
                    same_candidate_replay_case["run_dir"]
                )
                case_artifacts["target_same_candidate_replay_progress_json"] = (
                    same_candidate_replay_case["outer_optimizer_progress_json"]
                )
            if seed_case is not None:
                case_artifacts["shared_seed_run_dir"] = str(seed_case["run_dir"])
    except Exception as exc:
        _write_case_execution_failure_json(
            args.output_json,
            provenance=provenance,
            bundle_provenance=bundle_provenance,
            strict_transfer_support=strict_transfer_support,
            error=exc,
            case_root=case_root_for_failure,
        )
        print("SINGLE-STAGE INIT PARITY FAILED")
        print(f"  - Single-stage case execution failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

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
        require_final_metric_parity=not _trace_gated_final_metric_parity(args),
    )
    optimizer_state_trace_parity = None
    same_candidate_replay = None
    optimizer_path_objective_evaluations = None
    if _same_candidate_replay_required(args):
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
        strict_gate_failure_class = _same_candidate_replay_strict_gate_failure_class(
            same_candidate_replay
        )
        if strict_gate_failure_class is not None:
            same_candidate_replay["strict_gate_failure_class"] = (
                strict_gate_failure_class
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
    elif bool(args.record_objective_evaluation_trace):
        same_candidate_replay = {
            "status": "not-applicable",
            "reason": "maxiter=0 does not run the outer optimizer or record objective-evaluation events",
            "cpu_event_count": 0,
            "jax_event_count": 0,
            "same_candidate_event_count": 0,
            "require_exact_candidates": False,
            "strict_solver_contract": False,
            "solver_contract_diagnostics": [],
            "failures": [],
        }
        optimizer_path_objective_evaluations = {
            "status": "not-applicable",
            "reason": "maxiter=0 does not run the outer optimizer or record objective-evaluation events",
            "cpu_event_count": 0,
            "jax_event_count": 0,
            "paired_event_count": 0,
            "candidate_split_abs_tol": _OPTIMIZER_PATH_CANDIDATE_SPLIT_ATOL,
        }
    if _trace_gated_final_metric_parity(args):
        final_metric_failures = comparison["final_metric_parity_failures"]
        if not bool(args.record_objective_evaluation_trace):
            failures.append(
                "Trace-gated optimizer comparison rungs require "
                "--record-objective-evaluation-trace to record the "
                "same-candidate objective/gradient gate."
            )
            failures.extend(final_metric_failures)
        elif final_metric_failures:
            if _optimizer_control_split_accepts_final_metric_drift(
                same_candidate_replay=same_candidate_replay,
                optimizer_path_objective_evaluations=optimizer_path_objective_evaluations,
            ):
                comparison["final_metric_split_accepted"] = True
                comparison["accepted_final_metric_parity_failures"] = (
                    final_metric_failures
                )
            else:
                failures.extend(final_metric_failures)
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
        "oracle_role": "cpu_reference",
    }
    reference_lane_contract = full_run_artifact_contract["lanes"][
        _reference_lane_label(reference_backend)
    ]
    reference_artifact_contract_verified = not bool(
        reference_lane_contract.get("loaded_external_reference", False)
    ) or bool(
        reference_lane_contract.get("runtime_seed_spec_hash_verified", False)
        and reference_lane_contract.get("run_family_id_verified", False)
    )
    proof_parity["reference_artifact_source"] = reference_lane_contract.get(
        "artifact_source"
    )
    proof_parity["reference_artifact_contract_verified"] = bool(
        reference_artifact_contract_verified
    )
    active_proof_parity = proof_parity
    warnings: list[str] = []
    if not reference_artifact_contract_verified:
        warnings.append(
            "CPU reference artifact was loaded from an external output root; "
            "full_run_artifact_contract marks runtime_seed_spec_hash and "
            "run_family_id as unverified for that lane. This is a target-only "
            "replay against a loaded reference, not a self-contained full-run "
            "artifact contract proof."
        )
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
    if comparison.get("final_metric_parity_skipped_for_nonconvergence"):
        warnings.append(
            "End-state metric parity was not gated: the CPU reference optimizer "
            f"terminated abnormally (status={comparison.get('cpu_optimizer_status')}) "
            "while the JAX target did not "
            f"(status={comparison.get('jax_optimizer_status')}), so there is no "
            "converged reference optimum to compare against. The drift is recorded "
            "under skipped_final_metric_parity_failures; this run does not establish "
            "end-state port parity (a JAX-target abnormal stop would instead remain a "
            "hard failure)."
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

    lanes = {
        lane: contract["run_dir"]
        for lane, contract in full_run_artifact_contract["lanes"].items()
    }
    cpu_elapsed_s = cpu_case["elapsed_s"]
    jax_elapsed_s = jax_case["elapsed_s"]
    timing_breakdown = _single_stage_pair_timing_breakdown(cpu_case, jax_case)
    timings = {
        "cpu_elapsed_s": None if cpu_elapsed_s is None else float(cpu_elapsed_s),
        "jax_elapsed_s": None if jax_elapsed_s is None else float(jax_elapsed_s),
        "cpu_elapsed_source": cpu_case.get("elapsed_source"),
        "jax_elapsed_source": jax_case.get("elapsed_source"),
        "cpu_optimizer_wall_excluding_status_reporting_s": timing_breakdown["cpu"][
            "optimizer_wall_excluding_status_reporting_s"
        ],
        "jax_optimizer_wall_excluding_status_reporting_s": timing_breakdown["jax"][
            "optimizer_wall_excluding_status_reporting_s"
        ],
        "jax_optimizer_wall_excluding_status_reporting_vs_cpu_ratio": timing_breakdown[
            "jax_optimizer_wall_excluding_status_reporting_vs_cpu_ratio"
        ],
        **_prefix_phase_timings("cpu", cpu_case["phase_timings"]),
        **_prefix_phase_timings("jax", jax_case["phase_timings"]),
    }

    payload = {
        "provenance": provenance,
        "bundle_provenance": bundle_provenance,
        "cpu_results": cpu_results,
        "jax_results": jax_results,
        "comparison": comparison,
        "proof_parity": proof_parity,
        "active_proof_parity": active_proof_parity,
        "full_run_artifact_contract": full_run_artifact_contract,
        "lanes": lanes,
        "timings": timings,
        "timing_breakdown": timing_breakdown,
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
