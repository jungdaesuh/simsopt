"""Tier 3 production-grid Boozer parity probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.benchmark_config import resolve_configs
from benchmarks.benchmark_problem import build_synthetic_boozer_problem, clone_tensor_surface
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.validation_ladder_common import (
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    peak_rss_mb,
    preparse_platform,
    print_provenance,
    query_gpu_memory_mb,
    relative_error,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


SOLVER_OPTIONS = {
    "verbose": False,
    "bfgs_maxiter": 300,
    "bfgs_tol": 1e-8,
    "newton_maxiter": 20,
    "newton_tol": 1e-9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a production-grid CPU vs JAX Boozer parity probe."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--config-label",
        default="Columbia (12 coils, 128x64)",
        help="Benchmark configuration label from benchmark_config.DEFAULT_CONFIGS.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured parity results.",
    )
    return parser.parse_args()


def build_surface_pair(config_label: str):
    """Build independent CPU/JAX Boozer problems from the shared config."""
    config = resolve_configs([config_label])[0]
    problem = build_synthetic_boozer_problem(config)
    return config, problem


def _build_cpu_solver(problem):
    from simsopt.field import BiotSavart
    from simsopt.geo import BoozerSurface, Volume

    surf = clone_tensor_surface(problem.surface)
    vol = Volume(surf)
    bs = BiotSavart(problem.coils)
    booz = BoozerSurface(
        bs,
        surf,
        vol,
        problem.vol_target,
        constraint_weight=1.0,
        options=dict(SOLVER_OPTIONS),
    )
    return booz, vol


def _build_jax_solver(problem):
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX
    from simsopt.geo import Volume

    surf = clone_tensor_surface(problem.surface)
    vol = Volume(surf)
    bs = BiotSavartJAX(problem.coils)
    booz = BoozerSurfaceJAX(
        bs,
        surf,
        vol,
        problem.vol_target,
        constraint_weight=1.0,
        options={**SOLVER_OPTIONS, "optimizer_backend": "scipy"},
    )
    return booz, vol


def _run_full_case(builder, problem, *, name: str) -> dict:
    solver, volume = builder(problem)
    start = time.perf_counter()
    result = solver.run_code(problem.iota0, problem.G0)
    elapsed = time.perf_counter() - start
    if result is None:
        raise RuntimeError(f"{name} run_code() returned None")
    return {
        "elapsed_s": elapsed,
        "success": bool(result["success"]),
        "iter": int(result["iter"]),
        "iota": float(result["iota"]),
        "G": float(result["G"]),
        "fun": summarize_result_fun(result),
        "label_error": float(abs(volume.J() - problem.vol_target)),
        "peak_rss_mb": peak_rss_mb(),
        "gpu_memory_mb": query_gpu_memory_mb(),
    }


def _run_stage_split(builder, problem) -> dict:
    solver, _ = builder(problem)
    ls_start = time.perf_counter()
    ls_result = solver.minimize_boozer_penalty_constraints_LBFGS(
        constraint_weight=solver.constraint_weight,
        iota=problem.iota0,
        G=problem.G0,
        tol=solver.options["bfgs_tol"],
        maxiter=solver.options["bfgs_maxiter"],
        verbose=solver.options["verbose"],
        limited_memory=solver.options["limited_memory"],
        weight_inv_modB=solver.options["weight_inv_modB"],
    )
    ls_elapsed = time.perf_counter() - ls_start

    solver.need_to_run_code = True
    newton_start = time.perf_counter()
    newton_result = solver.minimize_boozer_penalty_constraints_newton(
        constraint_weight=solver.constraint_weight,
        iota=ls_result["iota"],
        G=ls_result["G"],
        verbose=solver.options["verbose"],
        tol=solver.options["newton_tol"],
        maxiter=solver.options["newton_maxiter"],
        stab=solver.options.get("newton_stab", 0.0),
        weight_inv_modB=solver.options["weight_inv_modB"],
    )
    newton_elapsed = time.perf_counter() - newton_start
    return {
        "ls_s": ls_elapsed,
        "newton_s": newton_elapsed,
        "success": bool(newton_result["success"]),
        "iter": int(newton_result["iter"]),
        "fun": summarize_result_fun(newton_result),
    }


def main() -> None:
    args = parse_args()
    bootstrap_local_simsopt()
    config, problem = build_surface_pair(args.config_label)
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Production Boozer parity probe",
        extra={
            "config_label": args.config_label,
            "platform_request": args.platform,
            "grid": f"{config.nphi}x{config.ntheta}",
            "surface_modes": f"mpol={config.mpol} ntor={config.ntor}",
        },
    )
    print_provenance(provenance)

    cpu_first = _run_full_case(_build_cpu_solver, problem, name="CPU")
    cpu_stage_split = _run_stage_split(_build_cpu_solver, problem)
    jax_first = _run_full_case(_build_jax_solver, problem, name="JAX")
    jax_second = _run_full_case(_build_jax_solver, problem, name="JAX-repeat")
    jax_stage_split = _run_stage_split(_build_jax_solver, problem)

    comparison = {
        "iota_abs_diff": abs(cpu_first["iota"] - jax_first["iota"]),
        "G_rel_diff": relative_error(jax_first["G"], cpu_first["G"]),
        "fun_rel_diff": relative_error(jax_first["fun"], cpu_first["fun"]),
        "cpu_label_error": cpu_first["label_error"],
        "jax_label_error": jax_first["label_error"],
        "jax_first_call_s": jax_first["elapsed_s"],
        "jax_second_call_s": jax_second["elapsed_s"],
        "jax_compile_overhead_s": max(
            jax_first["elapsed_s"] - jax_second["elapsed_s"],
            0.0,
        ),
    }

    failures: list[str] = []
    if not cpu_first["success"]:
        failures.append("CPU BoozerSurface did not converge.")
    if not jax_first["success"]:
        failures.append("JAX BoozerSurfaceJAX did not converge.")
    if comparison["iota_abs_diff"] >= 1e-3:
        failures.append(f"Iota disagreement too large: {comparison['iota_abs_diff']:.2e}")
    if cpu_first["label_error"] >= 1e-3:
        failures.append(f"CPU label error too large: {cpu_first['label_error']:.2e}")
    if jax_first["label_error"] >= 1e-3:
        failures.append(f"JAX label error too large: {jax_first['label_error']:.2e}")
    if not np.isfinite(cpu_first["fun"]):
        failures.append("CPU final objective is non-finite.")
    if not np.isfinite(jax_first["fun"]):
        failures.append("JAX final objective is non-finite.")

    print(
        "CPU vs JAX: "
        f"|iota diff|={comparison['iota_abs_diff']:.2e}, "
        f"G rel_diff={comparison['G_rel_diff']:.2e}, "
        f"fun rel_diff={comparison['fun_rel_diff']:.2e}, "
        f"jax compile overhead~={comparison['jax_compile_overhead_s']:.2f}s"
    )

    payload = {
        "provenance": provenance,
        "cpu_first": cpu_first,
        "cpu_stage_split": cpu_stage_split,
        "jax_first": jax_first,
        "jax_second": jax_second,
        "jax_stage_split": jax_stage_split,
        "comparison": comparison,
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output_json, payload)
    if failures:
        print("PRODUCTION BOOZER PARITY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PRODUCTION BOOZER PARITY PASSED")


if __name__ == "__main__":
    main()
