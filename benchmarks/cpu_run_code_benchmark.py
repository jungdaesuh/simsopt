# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax",
#     "numpy",
#     "scipy",
# ]
# ///
"""
JAX CPU run_code() benchmark — CPU baseline for Milestone 4 gate comparison.

Same synthetic problem as gpu_run_code_benchmark.py but on CPU only.
This gives us the JAX-on-CPU baseline to compare against JAX-on-GPU.

For the simsoptpp C++ baseline, a separate build environment is needed.

Usage:
    hf jobs uv run benchmarks/cpu_run_code_benchmark.py --flavor cpu-xl --timeout 15m
"""

import importlib.util
import os
import subprocess
import time
from pathlib import Path

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def setup_repo():
    repo_dir = Path("/tmp/simsopt-jax")
    if not repo_dir.exists():
        print("Cloning simsopt jax-port branch...")
        subprocess.run(
            ["git", "clone", "--branch", "jax-port", "--depth", "1",
             "https://github.com/jungdaesuh/simsopt.git", str(repo_dir)],
            check=True,
        )
    return repo_dir / "src" / "simsopt"


def load_module(name, src_root, relpath, pkg_name=None):
    spec = importlib.util.spec_from_file_location(name, str(src_root / relpath))
    mod = importlib.util.module_from_spec(spec)
    if pkg_name:
        import sys
        sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_coils(ncoils, nquad=128, R=1.0):
    coils = []
    for k in range(ncoils):
        z_off = 0.3 * (2 * (k % 2) - 1)
        phi0 = 2 * np.pi * (k // 2) / max(ncoils // 2, 1)
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        gamma = np.stack([
            R * np.cos(t) * np.cos(phi0) - R * np.sin(t) * np.sin(phi0) * 0.0,
            R * np.cos(t) * np.sin(phi0) + R * np.sin(t) * np.cos(phi0) * 0.0,
            z_off * np.ones_like(t),
        ], axis=-1)
        gd = np.stack([
            -R * np.sin(t) * np.cos(phi0) * 2 * np.pi,
            -R * np.sin(t) * np.sin(phi0) * 2 * np.pi,
            np.zeros_like(t),
        ], axis=-1)
        coils.append((gamma, gd, 1e5))

    return (
        jnp.array(np.stack([c[0] for c in coils])),
        jnp.array(np.stack([c[1] for c in coils])),
        jnp.array([c[2] for c in coils]),
    )


def time_run_code(run_fn, n_runs=5, label=""):
    t0 = time.perf_counter()
    res = run_fn()
    compile_time = time.perf_counter() - t0

    for _ in range(2):
        run_fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_fn()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"  {label}")
    print(f"    compile:     {compile_time:.3f}s")
    print(f"    steady:      {np.median(times)*1e3:.1f}ms median, "
          f"{np.mean(times)*1e3:.1f}ms mean ± {np.std(times)*1e3:.1f}ms")
    print(f"    converged:   {res.get('success', 'N/A')}")
    print(f"    iota:        {res.get('iota', 'N/A')}")
    print(f"    residual:    {res.get('fun', res.get('residual', 'N/A'))}")
    return {"compile_s": compile_time, "median_ms": np.median(times) * 1e3,
            "mean_ms": np.mean(times) * 1e3, "success": res.get("success")}


def run_benchmarks(src_root):
    bs_mod = load_module("biotsavart_jax", src_root, "field/biotsavart_jax.py",
                         pkg_name="simsopt.field.biotsavart_jax")
    sf_mod = load_module("surface_fourier_jax", src_root, "geo/surface_fourier_jax.py",
                         pkg_name="simsopt.geo.surface_fourier_jax")
    br_mod = load_module("boozer_residual_jax", src_root, "geo/boozer_residual_jax.py",
                         pkg_name="simsopt.geo.boozer_residual_jax")
    opt_mod = load_module("optimizer_jax", src_root, "geo/optimizer_jax.py",
                         pkg_name="simsopt.geo.optimizer_jax")
    lbl_mod = load_module("label_constraints_jax", src_root, "geo/label_constraints_jax.py",
                         pkg_name="simsopt.geo.label_constraints_jax")

    configs = [
        ("Small (4 coils, 15x15)",   4,  64, 15, 15, 2, 2, 1),
        ("Medium (6 coils, 15x15)",  6, 128, 15, 15, 4, 4, 1),
        ("HBT-like (12 coils, 15x15)", 12, 128, 15, 15, 4, 4, 1),
    ]

    for label, ncoils, nquad, nphi, ntheta, mpol, ntor, nfp in configs:
        print(f"\n{'='*70}")
        print(f"run_code() benchmark: {label}")
        print(f"  Boozer grid: {nphi}x{ntheta}, surface: mpol={mpol} ntor={ntor}")
        print(f"{'='*70}")

        gammas, gammadashs, currents = make_coils(ncoils, nquad)

        stellsym = False
        phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)
        scatter_indices = None

        n_full = (2 * mpol + 1) * (2 * ntor + 1)
        ndofs = 3 * n_full

        rng = np.random.RandomState(42)
        dofs = jnp.array(rng.randn(ndofs) * 0.001)
        xc_idx = 0 * n_full
        dofs = dofs.at[xc_idx + mpol * (2 * ntor + 1) + ntor].set(1.0)
        dofs = dofs.at[xc_idx + (mpol + 1) * (2 * ntor + 1) + ntor].set(0.1)
        zc_idx = 2 * n_full
        dofs = dofs.at[zc_idx + (mpol - 1) * (2 * ntor + 1) + ntor].set(0.1)

        iota_init = 0.3
        G_init = float(lbl_mod.compute_G_from_currents(currents, nfp))

        def run_ls():
            return opt_mod.jax_minimize(
                lambda x: br_mod.boozer_penalty_composed(
                    x[:-2], x[-2], x[-1],
                    phis, thetas, mpol, ntor, nfp, stellsym, scatter_indices,
                    gammas, gammadashs, currents,
                    True,
                ),
                jnp.concatenate([dofs, jnp.array([iota_init, G_init])]),
                method="bfgs",
                options={"maxiter": 100, "gtol": 1e-8},
            )

        time_run_code(run_ls, n_runs=5, label="LS (BFGS, 100 iter max)")

    # System info
    print(f"\n{'='*70}")
    print("CPU INFO")
    print(f"{'='*70}")
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if any(k in line.lower() for k in ["model name", "cpu(s)", "thread", "mhz"]):
                print(f"  {line.strip()}")
    except FileNotFoundError:
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True)
            print(f"  CPU: {result.stdout.strip()}")
        except FileNotFoundError:
            pass


def main():
    print("JAX run_code() CPU Benchmark — Milestone 4 baseline")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    src_root = setup_repo()
    run_benchmarks(src_root)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
