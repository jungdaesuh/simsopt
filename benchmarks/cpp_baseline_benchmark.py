# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.13",
#     "cmake<3.30",
#     "ninja",
#     "scikit-build-core",
#     "pybind11<2.12",
#     "setuptools-scm",
#
# ]
# ///
"""
simsoptpp CPU baseline benchmark for Boozer run_code comparisons.

Builds simsoptpp from the current local checkout, then runs the same synthetic
Boozer problem definition used by the JAX benchmark helpers through the CPU
``BoozerSurface`` path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_config import available_config_labels, resolve_configs
from benchmarks.benchmark_problem import build_synthetic_boozer_problem


def get_git_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def build_simsopt() -> Path:
    """Build the local simsopt checkout with the simsoptpp CPU extension."""
    print("Installing build dependencies...")
    subprocess.run(
        ["apt-get", "update", "-qq"],
        capture_output=True,
    )
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "libboost-dev",
            "libeigen3-dev",
            "libboost-filesystem-dev",
        ],
        capture_output=True,
    )

    print(f"Building local simsoptpp from {REPO_ROOT} ...")
    t0 = time.perf_counter()

    uv_path = shutil.which("uv")
    if uv_path:
        cmd = ["uv", "pip", "install", "-e", ".", "-v", "--no-build-isolation"]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "-v",
            "--no-build-isolation",
        ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    build_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"Build FAILED after {build_time:.1f}s")
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        sys.exit(1)

    print(f"Build succeeded in {build_time:.1f}s")
    subprocess.run(
        [sys.executable, "-c", "import simsoptpp; print('simsoptpp OK')"],
        check=True,
    )
    return REPO_ROOT


def summarize_result_fun(res):
    fun = res.get("fun")
    if fun is not None:
        return float(fun)
    residual = res.get("residual")
    if residual is None:
        return float("nan")
    arr = np.asarray(residual)
    if arr.ndim == 0:
        return float(arr)
    return 0.5 * float(np.mean(np.square(arr)))


def make_cpu_boozer_surface(config):
    from simsopt.field import BiotSavart
    from simsopt.geo import BoozerSurface

    problem = build_synthetic_boozer_problem(config)
    bs = BiotSavart(problem.coils)
    booz = BoozerSurface(
        bs,
        problem.surface,
        problem.volume,
        problem.vol_target,
        constraint_weight=1.0,
        options={
            "verbose": False,
            "bfgs_tol": 1e-8,
            "bfgs_maxiter": 50,
            "limited_memory": False,
            "newton_tol": 1e-9,
            "newton_maxiter": 10,
            "weight_inv_modB": True,
        },
    )
    return booz, problem.iota0, problem.G0


def time_run_code(config):
    booz, iota0, G0 = make_cpu_boozer_surface(config)
    t0 = time.perf_counter()
    res = booz.run_code(iota=iota0, G=G0)
    return time.perf_counter() - t0, res


def time_run_code_stage_split(config):
    booz, iota0, G0 = make_cpu_boozer_surface(config)
    t0 = time.perf_counter()
    ls_res = booz.minimize_boozer_penalty_constraints_LBFGS(
        constraint_weight=booz.constraint_weight,
        iota=iota0,
        G=G0,
        tol=booz.options["bfgs_tol"],
        maxiter=booz.options["bfgs_maxiter"],
        verbose=booz.options["verbose"],
        limited_memory=booz.options["limited_memory"],
        weight_inv_modB=booz.options["weight_inv_modB"],
    )
    ls_time = time.perf_counter() - t0

    booz.need_to_run_code = True
    t1 = time.perf_counter()
    res = booz.minimize_boozer_penalty_constraints_newton(
        constraint_weight=booz.constraint_weight,
        iota=ls_res["iota"],
        G=ls_res["G"],
        verbose=booz.options["verbose"],
        tol=booz.options["newton_tol"],
        maxiter=booz.options["newton_maxiter"],
        stab=0.0,
        weight_inv_modB=booz.options["weight_inv_modB"],
    )
    newton_time = time.perf_counter() - t1
    return ls_time, newton_time, res


def run_benchmarks(*, configs, repeats):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    print(f"Repo SHA: {get_git_sha()}")

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"C++ run_code() benchmark: {config.label}")
        print(
            f"  grid: {config.nphi}x{config.ntheta}, surface: "
            f"mpol={config.mpol} ntor={config.ntor}, coils={config.ncoils}"
        )
        print(f"{'=' * 70}")

        first_time, res = time_run_code(config)
        print(
            f"    first call:  {first_time:.3f}s  "
            f"success={res['success']}  iter={res['iter']}"
        )
        print(
            f"    final fun:   {summarize_result_fun(res):.6e}  "
            f"iota={float(res['iota']):.6f}"
        )

        ls_time, newton_time, stage_res = time_run_code_stage_split(config)
        print(
            f"    stage split sample: LS {ls_time * 1e3:.1f}ms, "
            f"Newton {newton_time * 1e3:.1f}ms"
        )
        if not stage_res["success"]:
            print(
                "    warning: unconverged solve; treat timing as diagnostic only, "
                "not as a parity or replacement verdict"
            )

        repeat_times = []
        repeat_res = res
        for _ in range(repeats):
            elapsed, repeat_res = time_run_code(config)
            repeat_times.append(elapsed)

        times = np.asarray(repeat_times)
        print(
            f"    repeat fresh solve: {np.median(times) * 1e3:.1f}ms median, "
            f"{np.mean(times) * 1e3:.1f}ms mean ± {np.std(times) * 1e3:.1f}ms"
        )
        print(
            f"    repeat final fun: {summarize_result_fun(repeat_res):.6e}  "
            f"iota={float(repeat_res['iota']):.6f}"
        )

    print(f"\n{'=' * 70}")
    print("CPU INFO")
    print(f"{'=' * 70}")
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if any(key in line.lower() for key in ["model name", "cpu(s):", "thread"]):
                print(f"  {line.strip()}")
    except FileNotFoundError:
        pass
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        choices=available_config_labels(),
        help="Benchmark config label to run. Repeat to run multiple configs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of steady-state fresh-solve repeats per config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("simsoptpp C++ Baseline Benchmark — shared synthetic problem")
    print(f"Python: {sys.version}")
    build_simsopt()
    run_benchmarks(configs=resolve_configs(args.config), repeats=args.repeats)
    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
