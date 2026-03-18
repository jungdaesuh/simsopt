# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax[cuda12]",
#     "numpy",
# ]
# ///
"""
JAX GPU Benchmark for simsopt JAX port — A100 validation.

Runs on HF Jobs:
    hf jobs uv run benchmarks/gpu_benchmark.py --flavor a100-large

Self-contained: clones simsopt jax-port branch, loads modules directly.
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def setup_repo():
    """Clone the jax-port branch if not already present."""
    repo_dir = Path("/tmp/simsopt-jax")
    if not repo_dir.exists():
        print("Cloning simsopt jax-port branch...")
        subprocess.run(
            [
                "git", "clone",
                "--branch", "jax-port",
                "--depth", "1",
                "https://github.com/jungdaesuh/simsopt.git",
                str(repo_dir),
            ],
            check=True,
        )
    return repo_dir / "src" / "simsopt"


def load_module(name, src_root, relpath):
    """Load a module without importing simsopt package."""
    spec = importlib.util.spec_from_file_location(name, str(src_root / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def print_gpu_info():
    """Print GPU and JAX device info."""
    import jax

    print(f"\n{'=' * 70}")
    print("GPU / JAX ENVIRONMENT")
    print(f"{'=' * 70}")
    print(f"JAX version:     {jax.__version__}")
    print(f"Devices:         {jax.devices()}")
    print(f"Backend:         {jax.default_backend()}")
    print(f"Float64 enabled: {jax.config.jax_enable_x64}")

    # nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        print(f"GPU:             {result.stdout.strip()}")
    except FileNotFoundError:
        print("GPU:             nvidia-smi not found")

    # JAX device memory
    for d in jax.devices():
        try:
            stats = d.memory_stats()
            if stats:
                peak = stats.get("peak_bytes_in_use", 0) / 1e9
                print(f"JAX peak mem:    {peak:.3f} GB (device {d.id})")
        except Exception:
            pass
    print()


def gpu_memory_snapshot(label=""):
    """Print current GPU memory usage."""
    import jax

    for d in jax.devices():
        try:
            stats = d.memory_stats()
            if stats:
                current = stats.get("bytes_in_use", 0) / 1e6
                peak = stats.get("peak_bytes_in_use", 0) / 1e6
                print(f"  [{label}] GPU mem: current={current:.1f} MB, peak={peak:.1f} MB")
        except Exception:
            pass


def time_fn(fn, *args, warmup=2, repeat=50):
    """Time a JIT-compiled function, separating compile from steady-state."""
    import jax

    # Compile + first call
    t0 = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    # Warm-up
    for _ in range(warmup):
        r = fn(*args)
        jax.block_until_ready(r)

    # Steady-state
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        r = fn(*args)
        jax.block_until_ready(r)
        times.append(time.perf_counter() - t0)

    return {
        "compile_s": compile_time,
        "median_ms": np.median(times) * 1e3,
        "mean_ms": np.mean(times) * 1e3,
        "std_ms": np.std(times) * 1e3,
        "min_ms": np.min(times) * 1e3,
    }


def make_coils(ncoils, nquad, R_major=1.0, r_coil=0.3):
    import jax.numpy as jnp

    gammas, gammadashs = [], []
    for k in range(ncoils):
        phi0 = 2 * np.pi * k / ncoils
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        cx = (R_major + r_coil * np.cos(t)) * np.cos(phi0)
        cy = (R_major + r_coil * np.cos(t)) * np.sin(phi0)
        cz = r_coil * np.sin(t)
        gammas.append(np.stack([cx, cy, cz], axis=-1))
        dcx = (-r_coil * np.sin(t) * np.cos(phi0)) * 2 * np.pi
        dcy = (-r_coil * np.sin(t) * np.sin(phi0)) * 2 * np.pi
        dcz = (r_coil * np.cos(t)) * 2 * np.pi
        gammadashs.append(np.stack([dcx, dcy, dcz], axis=-1))
    return (
        jnp.array(np.stack(gammas)),
        jnp.array(np.stack(gammadashs)),
        jnp.ones(ncoils) * 1e5,
    )


def make_eval_points(nphi, ntheta, nfp=2, R=1.0, r_minor=0.1):
    import jax.numpy as jnp

    phis = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi_2d, theta_2d = np.meshgrid(phis, thetas, indexing="ij")
    x = ((R + r_minor * np.cos(theta_2d)) * np.cos(phi_2d)).ravel()
    y = ((R + r_minor * np.cos(theta_2d)) * np.sin(phi_2d)).ravel()
    z = (r_minor * np.sin(theta_2d)).ravel()
    return jnp.array(np.stack([x, y, z], axis=-1))


def make_surface_data(nphi, ntheta, mpol, ntor, nfp, R=1.0, r=0.1):
    import jax.numpy as jnp

    phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)
    xc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    yc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    zc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    xc = xc.at[0, 0].set(R)
    xc = xc.at[1, 0].set(r)
    zc = zc.at[mpol + 1, 0].set(r)
    return phis, thetas, xc, yc, zc, mpol, ntor, nfp


def run_benchmarks(src_root):
    import jax
    import jax.numpy as jnp

    bs = load_module("biotsavart_jax", src_root, "field/biotsavart_jax.py")
    sf = load_module("surface_fourier_jax", src_root, "geo/surface_fourier_jax.py")
    br = load_module("boozer_residual_jax", src_root, "geo/boozer_residual_jax.py")

    configs = [
        # (label, ncoils, nquad, nphi, ntheta, mpol, ntor)
        ("Small (smoke)",     4,  128,  15,  15,  5,  5),
        ("Medium (Boozer)",  12,  200,  32,  32,  8,  6),
        ("Large (Stage 2)",  22,  200,  64,  64, 10, 10),
        ("Full (Columbia)", 22,  200, 128,  64, 10, 10),
    ]

    for label, ncoils, nquad, nphi, ntheta, mpol, ntor in configs:
        npoints = nphi * ntheta
        print(f"\n{'=' * 70}")
        print(f"{label}: {ncoils} coils, {nquad} quad, {npoints} eval pts, mpol={mpol} ntor={ntor}")
        print(f"{'=' * 70}")

        gammas, gammadashs, currents = make_coils(ncoils, nquad)
        points = make_eval_points(nphi, ntheta)

        # BiotSavart B
        t = time_fn(bs.biot_savart_B, points, gammas, gammadashs, currents)
        print(f"  BS.B:          compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # BiotSavart dB/dX
        t = time_fn(bs.biot_savart_dB_by_dX, points, gammas, gammadashs, currents)
        print(f"  BS.dB/dX:      compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # BiotSavart B+dB
        t = time_fn(bs.biot_savart_B_and_dB, points, gammas, gammadashs, currents)
        print(f"  BS.B+dB:       compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # Accuracy
        _, dB = bs.biot_savart_B_and_dB(points, gammas, gammadashs, currents)
        div_B = jnp.trace(dB, axis1=1, axis2=2)
        print(f"  div(B) max:    {float(jnp.max(jnp.abs(div_B))):.2e}")

        # Surface gamma
        phis, thetas, xc, yc, zc, _, _, nfp = make_surface_data(nphi, ntheta, mpol, ntor, 2)
        t = time_fn(sf.surface_gamma, phis, thetas, xc, yc, zc, mpol, ntor, nfp)
        print(f"  surf.gamma:    compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # Surface normal
        t = time_fn(sf.surface_normal, phis, thetas, xc, yc, zc, mpol, ntor, nfp)
        print(f"  surf.normal:   compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # Surface dgamma_by_dcoeff (needs DOF-form inputs)
        stellsym = True
        scatter_idx = sf.stellsym_scatter_indices(mpol, ntor)
        ndofs = len(scatter_idx)
        surf_dofs = jnp.zeros(ndofs).at[0].set(1.0).at[1].set(0.1)
        t = time_fn(
            sf.dgamma_by_dcoeff,
            surf_dofs, phis, thetas, mpol, ntor, nfp, stellsym, scatter_idx,
        )
        print(f"  surf.dg/dc:    compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # Boozer residual scalar
        rng = np.random.RandomState(42)
        B = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0, 0, 1.0]))
        xphi = jnp.array(rng.randn(nphi, ntheta, 3) * 0.5)
        xtheta = jnp.array(rng.randn(nphi, ntheta, 3) * 0.5)
        t = time_fn(br.boozer_residual_scalar, 1.5, 0.3, B, xphi, xtheta)
        print(f"  booz.scalar:   compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # Boozer gradient
        t = time_fn(br.boozer_residual_grad, 1.5, 0.3, B, xphi, xtheta, 0)
        print(f"  booz.grad:     compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        # Boozer Hessian
        t = time_fn(br.boozer_residual_hessian, 1.5, 0.3, B, xphi, xtheta, 0)
        print(f"  booz.hessian:  compile={t['compile_s']:.3f}s  steady={t['median_ms']:.3f}ms ±{t['std_ms']:.3f}")

        gpu_memory_snapshot(label)

    # Final GPU summary
    print(f"\n{'=' * 70}")
    print("FINAL GPU MEMORY SUMMARY")
    print(f"{'=' * 70}")
    gpu_memory_snapshot("final")

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        pass


def main():
    # Enable FP64
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    import jax
    jax.config.update("jax_enable_x64", True)

    print_gpu_info()

    src_root = setup_repo()
    run_benchmarks(src_root)

    print(f"\n{'=' * 70}")
    print("GPU BENCHMARK COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
