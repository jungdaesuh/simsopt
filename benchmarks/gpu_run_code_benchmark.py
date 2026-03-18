# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax[cuda12]",
#     "numpy",
#     "scipy",
# ]
# ///
"""
JAX GPU run_code() benchmark — Milestone 4 gate.

Usage:
    hf jobs uv run benchmarks/gpu_run_code_benchmark.py --flavor a100-large --timeout 15m
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cuda")
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
        sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_coils(ncoils, nquad=128):
    """Create ncoils circular coils at z=±0.3."""
    coils = []
    for k in range(ncoils):
        z_off = 0.3 * (2 * (k % 2) - 1)
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        R = 1.0
        gamma = np.stack([R * np.cos(t), R * np.sin(t), z_off * np.ones_like(t)], axis=-1)
        gd = np.stack([-R * np.sin(t) * 2 * np.pi, R * np.cos(t) * 2 * np.pi,
                        np.zeros_like(t)], axis=-1)
        coils.append((jnp.array(gamma), jnp.array(gd), 1e5))
    gammas = jnp.stack([c[0] for c in coils])
    gammadashs = jnp.stack([c[1] for c in coils])
    currents = jnp.array([c[2] for c in coils])
    return gammas, gammadashs, currents


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
        ("Small (4 coils, 15x15)",         4,  64,  15,  15, 2, 2, 1),
        ("Medium (6 coils, 15x15)",        6, 128,  15,  15, 4, 4, 1),
        ("HBT-like (12 coils, 15x15)",    12, 128,  15,  15, 4, 4, 1),
        ("Prod-grid (12 coils, 64x64)",   12, 128,  64,  64, 4, 4, 1),
        ("Columbia (12 coils, 128x64)",   12, 200, 128,  64, 8, 6, 1),
        ("Full-HBT (22 coils, 128x64)",  22, 200, 128,  64, 8, 6, 1),
    ]

    for label, ncoils, nquad, nphi, ntheta, mpol, ntor, nfp in configs:
        print(f"\n{'='*70}")
        print(f"run_code() benchmark: {label}")
        print(f"  Boozer grid: {nphi}x{ntheta}, surface: mpol={mpol} ntor={ntor}")
        print(f"{'='*70}")

        gammas, gammadashs, currents = make_coils(ncoils, nquad)
        coil_arrays = [(gammas, gammadashs, currents)]

        stellsym = False
        phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)

        n_full = (2 * mpol + 1) * (2 * ntor + 1)
        ndofs = 3 * n_full

        # Initialize DOFs for a simple torus (R0=1.0, r=0.1)
        rng = np.random.RandomState(42)
        surf_dofs = jnp.array(rng.randn(ndofs) * 0.001)
        surf_dofs = surf_dofs.at[mpol * (2 * ntor + 1) + ntor].set(1.0)
        surf_dofs = surf_dofs.at[(mpol + 1) * (2 * ntor + 1) + ntor].set(0.1)
        surf_dofs = surf_dofs.at[2 * n_full + (mpol - 1) * (2 * ntor + 1) + ntor].set(0.1)

        iota_init = 0.3
        G_init = float(lbl_mod.compute_G_from_currents(currents))

        # Build decision vector: [surf_dofs, iota, G]
        x0 = jnp.concatenate([surf_dofs, jnp.array([iota_init, G_init])])

        # Objective function with keyword args
        def objective(x):
            return br_mod.boozer_penalty_composed(
                x,
                coil_arrays=coil_arrays,
                quadpoints_phi=phis,
                quadpoints_theta=thetas,
                mpol=mpol,
                ntor=ntor,
                nfp=nfp,
                stellsym=stellsym,
                scatter_indices=None,
                optimize_G=True,
                weight_inv_modB=True,
            )

        # Warm compile
        print("  Compiling...")
        t0 = time.perf_counter()
        res = opt_mod.jax_minimize(objective, x0, method="bfgs",
                                    maxiter=50, tol=1e-8)
        compile_time = time.perf_counter() - t0
        print(f"    first call:  {compile_time:.3f}s  "
              f"converged={res.success}  nit={res.nit}  fun={float(res.fun):.6e}")

        # Steady-state
        times = []
        for i in range(5):
            t0 = time.perf_counter()
            res = opt_mod.jax_minimize(objective, x0, method="bfgs",
                                        maxiter=50, tol=1e-8)
            times.append(time.perf_counter() - t0)

        times = np.array(times)
        print(f"    steady:      {np.median(times)*1e3:.1f}ms median, "
              f"{np.mean(times)*1e3:.1f}ms mean ± {np.std(times)*1e3:.1f}ms")
        print(f"    converged:   {res.success}  nit={res.nit}")
        print(f"    final fun:   {float(res.fun):.6e}")
        print(f"    final iota:  {float(res.x[-2]):.6f}")

        for d in jax.devices():
            try:
                stats = d.memory_stats()
                if stats:
                    print(f"    GPU peak:    {stats.get('peak_bytes_in_use', 0)/1e6:.1f} MB")
            except Exception:
                pass

    # nvidia-smi
    print(f"\n{'='*70}")
    print("GPU SUMMARY")
    print(f"{'='*70}")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True)
        print(r.stdout.strip())
    except FileNotFoundError:
        pass


def main():
    print("JAX run_code() GPU Benchmark — Milestone 4")
    print(f"JAX: {jax.__version__}  Devices: {jax.devices()}  Backend: {jax.default_backend()}")
    src_root = setup_repo()
    run_benchmarks(src_root)
    print(f"\n{'='*70}\nBENCHMARK COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()
