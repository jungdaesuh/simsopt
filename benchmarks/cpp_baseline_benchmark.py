# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "cmake<3.30",
#     "ninja",
#     "scikit-build-core",
#     "pybind11<2.12",
#     "setuptools-scm",
#
# ]
# ///
"""
simsoptpp C++ baseline benchmark for Milestone 4 comparison.

Builds simsoptpp from source, then runs the same Boozer problem
through the CPU C++ path for a fair comparison against JAX.

Usage:
    hf jobs uv run benchmarks/cpp_baseline_benchmark.py --flavor cpu-xl --timeout 30m
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_config import available_config_labels, resolve_configs


def build_simsopt():
    """Build the local simsopt checkout with the simsoptpp CPU extension."""
    # Install boost and eigen headers via apt (if available)
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

    # Build and install the current local repo
    print(f"Building local simsoptpp from {REPO_ROOT} ...")
    t0 = time.perf_counter()

    # uv pip for UV-managed environments
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

    # Verify import
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


def run_benchmarks(*, configs, repeats):
    """Run Boozer solve through the CPU C++ path."""
    # Add the local repo to path so simsopt is importable
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, CurveXYZFourier
    from simsopt.geo.surfaceobjectives import Volume
    from simsopt.field import BiotSavart, Current, Coil

    for config in configs:
        label = config.label
        ncoils = config.ncoils
        nphi = config.nphi
        ntheta = config.ntheta
        mpol = config.mpol
        ntor = config.ntor
        nfp = config.nfp
        nquad = config.nquad
        print(f"\n{'=' * 70}")
        print(f"C++ run_code() benchmark: {label}")
        print(f"  Boozer grid: {nphi}x{ntheta}, surface: mpol={mpol} ntor={ntor}")
        print(f"{'=' * 70}")

        # Create coils (circular, z=±0.3)
        coils = []
        for k in range(ncoils):
            z_off = 0.3 * (2 * (k % 2) - 1)
            curve = CurveXYZFourier(np.linspace(0, 1, nquad, endpoint=False), order=1)
            R = 1.0
            curve.set("xc(0)", 0.0)
            curve.set("xc(1)", R)
            curve.set("yc(0)", 0.0)
            curve.set("ys(1)", R)
            curve.set("zc(0)", z_off)
            current = Current(1e5)
            current.fix_all()
            coils.append(Coil(curve, current))

        bs = BiotSavart(coils)

        # Create surface
        s = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=False,
            quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
        )
        # Simple torus: R0=1.0, r=0.1
        # DOF basis: w_m(θ)·v_n(φ) where m=0..mpol are cos, m=mpol+1..2*mpol are sin
        # DC component is (0, 0), NOT (mpol, ntor)
        s.set("x(0,0)", 1.0)  # R0 (DC)
        s.set("x(1,0)", 0.1)  # r cos(theta)
        s.set(f"z({mpol + 1},0)", 0.1)  # r sin(theta)

        vol = Volume(s)
        vol_target = vol.J()

        boozer = BoozerSurface(
            bs,
            s,
            vol,
            vol_target,
            constraint_weight=1.0,
            options={
                "verbose": False,
                "bfgs_tol": 1e-8,
                "bfgs_maxiter": 50,
                "limited_memory": False,
                "newton_tol": 1e-9,
                "newton_maxiter": 10,
            },
        )
        mu0 = 4 * np.pi * 1e-7
        G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
        iota0 = 0.3

        # First call (cold)
        print("  Running...")
        t0 = time.perf_counter()
        res = boozer.run_code(iota=iota0, G=G0)
        first_time = time.perf_counter() - t0
        if res:
            iota_val = res.get("iota")
            fun_val = summarize_result_fun(res)
            print(
                f"    first call:  {first_time:.3f}s  "
                f"converged={res.get('success', 'N/A')}  "
                f"iota={f'{iota_val:.6f}' if iota_val is not None else 'N/A'}  "
                f"fun={fun_val:.6e}"
            )

        # Stage split on a fresh solve
        boozer.need_to_run_code = True
        t0 = time.perf_counter()
        ls_res = boozer.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=boozer.constraint_weight,
            iota=iota0,
            G=G0,
            tol=boozer.options["bfgs_tol"],
            maxiter=boozer.options["bfgs_maxiter"],
            verbose=boozer.options["verbose"],
            limited_memory=boozer.options["limited_memory"],
            weight_inv_modB=boozer.options["weight_inv_modB"],
        )
        ls_time = time.perf_counter() - t0

        boozer.need_to_run_code = True
        t0 = time.perf_counter()
        boozer.minimize_boozer_penalty_constraints_newton(
            constraint_weight=boozer.constraint_weight,
            iota=ls_res["iota"],
            G=ls_res["G"],
            verbose=boozer.options["verbose"],
            tol=boozer.options["newton_tol"],
            maxiter=boozer.options["newton_maxiter"],
            stab=0.0,
            weight_inv_modB=boozer.options["weight_inv_modB"],
        )
        newton_time = time.perf_counter() - t0
        print(
            f"    stage split sample: LS {ls_time * 1e3:.1f}ms, "
            f"Newton {newton_time * 1e3:.1f}ms"
        )

        # Steady-state
        times = []
        for _ in range(repeats):
            boozer.need_to_run_code = True
            t0 = time.perf_counter()
            res = boozer.run_code(iota=iota0, G=G0)
            times.append(time.perf_counter() - t0)

        times = np.array(times)
        print(
            f"    steady:      {np.median(times) * 1e3:.1f}ms median, "
            f"{np.mean(times) * 1e3:.1f}ms mean ± {np.std(times) * 1e3:.1f}ms"
        )

    # CPU info
    print(f"\n{'=' * 70}")
    print("CPU INFO")
    print(f"{'=' * 70}")
    try:
        r = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in r.stdout.split("\n"):
            if any(k in line.lower() for k in ["model name", "cpu(s):", "thread"]):
                print(f"  {line.strip()}")
    except FileNotFoundError:
        pass
    # Thread count
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
    print("simsoptpp C++ Baseline Benchmark — Milestone 4")
    print(f"Python: {sys.version}")
    build_simsopt()
    run_benchmarks(configs=resolve_configs(args.config), repeats=args.repeats)
    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
