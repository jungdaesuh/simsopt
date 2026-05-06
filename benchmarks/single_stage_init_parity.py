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
    if (
        diagnose_target_lane_scaled_phase1
        or record_target_lane_invalid_state_events
    ):
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


def _needs_shared_init_seed(args: argparse.Namespace, *, reference_backend: str) -> bool:
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
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any] | None]:
    compare_surface_geometry = _should_compare_surface_geometry(
        args,
        benchmark_mode=benchmark_mode,
    )
    seed_case = None
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
    jax_case = _run_single_stage_case(
        target_args,
        "jax",
        platform=args.platform,
        benchmark_mode=benchmark_mode,
        load_surface_gamma=compare_surface_geometry,
        output_root=case_root / "target_outputs",
        jax_runtime_seed_spec=jax_seed_spec,
    )
    return cpu_case, jax_case, jax_seed_spec, seed_case


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
