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

import os
import shutil
import subprocess
import sys
import time

import numpy as np


def build_simsopt():
    """Clone and build simsopt with simsoptpp C++ extension."""
    repo_dir = "/tmp/simsopt-cpp"
    if not os.path.exists(repo_dir):
        print(
            "Cloning simsopt fork (codex/hho-runner-parameterization — CPU-only branch)..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "codex/hho-runner-parameterization",
                "--depth",
                "1",
                "--recurse-submodules",
                "--shallow-submodules",
                "https://github.com/jungdaesuh/simsopt.git",
                repo_dir,
            ],
            check=True,
        )

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

    # Build and install simsopt
    print("Building simsoptpp (this takes 2-3 minutes)...")
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
        cwd=repo_dir,
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
    return repo_dir


def run_benchmarks():
    """Run Boozer solve through the CPU C++ path."""
    # Add repo to path so simsopt is importable
    sys.path.insert(0, "/tmp/simsopt-cpp/src")

    from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, CurveXYZFourier
    from simsopt.geo.surfaceobjectives import Volume
    from simsopt.field import BiotSavart, Current, Coil

    configs = [
        # (label, ncoils, nquad, nphi, ntheta, mpol, ntor)
        ("Small (4 coils, 15x15)", 4, 64, 15, 15, 2, 2),
        ("HBT-like (12 coils, 15x15)", 12, 128, 15, 15, 4, 4),
        ("Prod-grid (12 coils, 64x64)", 12, 128, 64, 64, 4, 4),
        ("Columbia (12 coils, 128x64)", 12, 200, 128, 64, 8, 6),
    ]

    for label, ncoils, nquad, nphi, ntheta, mpol, ntor in configs:
        print(f"\n{'=' * 70}")
        print(f"C++ run_code() benchmark: {label}")
        print(f"  Boozer grid: {nphi}x{ntheta}, surface: mpol={mpol} ntor={ntor}")
        print(f"{'=' * 70}")

        nfp = 1

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
                "newton_tol": 1e-12,
                "newton_maxiter": 10,
            },
        )

        # First call (cold)
        print("  Running...")
        t0 = time.perf_counter()
        res = boozer.run_code(iota=0.3, G=None)
        first_time = time.perf_counter() - t0
        if res:
            iota_val = res.get("iota")
            fun_val = res.get("fun")
            print(
                f"    first call:  {first_time:.3f}s  "
                f"converged={res.get('success', 'N/A')}  "
                f"iota={f'{iota_val:.6f}' if iota_val is not None else 'N/A'}  "
                f"fun={f'{fun_val:.6e}' if fun_val is not None else 'N/A'}"
            )

        # Steady-state
        times = []
        for _ in range(5):
            boozer.need_to_run_code = True
            t0 = time.perf_counter()
            res = boozer.run_code(iota=0.3, G=None)
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


def main():
    print("simsoptpp C++ Baseline Benchmark — Milestone 4")
    print(f"Python: {sys.version}")
    build_simsopt()
    run_benchmarks()
    print(f"\n{'=' * 70}\nBENCHMARK COMPLETE\n{'=' * 70}")


if __name__ == "__main__":
    main()
