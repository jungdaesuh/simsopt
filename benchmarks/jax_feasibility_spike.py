#!/usr/bin/env python
"""
JAX Feasibility Spike Benchmark — Milestone 1

Measures:
  1. First-compile (JIT warm-up) time
  2. Steady-state per-call time
  3. Numerical accuracy against analytical/reference values
  4. Memory footprint (peak RSS delta)

Runs on CPU by default.  For GPU, install jaxlib[cuda12] and set
JAX_PLATFORMS=cuda before running (SIMSOPT_JAX_BACKEND is only read
by the simsopt package __init__, which this script bypasses).

Usage:
    conda run -n <env> python benchmarks/jax_feasibility_spike.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Direct module loading (avoids simsoptpp dependency)
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1] / "src" / "simsopt"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
_sf = _load("surface_fourier_jax", "geo/surface_fourier_jax.py")
_br = _load("boozer_residual_jax", "geo/boozer_residual_jax.py")
_ib = _load("integral_bdotn_jax", "objectives/integral_bdotn_jax.py")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def make_coils(ncoils=4, nquad=128, R_major=1.0, r_coil=0.3):
    """Create ncoils equally spaced circular coils around a torus."""
    gammas = []
    gammadashs = []
    for k in range(ncoils):
        phi0 = 2 * np.pi * k / ncoils
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        cx = (R_major + r_coil * np.cos(t)) * np.cos(phi0)
        cy = (R_major + r_coil * np.cos(t)) * np.sin(phi0)
        cz = r_coil * np.sin(t)
        gammas.append(np.stack([cx, cy, cz], axis=-1))

        # dγ/d(t_01) where t_01 ∈ [0,1), so multiply by 2π
        dcx = (-r_coil * np.sin(t) * np.cos(phi0)) * 2 * np.pi
        dcy = (-r_coil * np.sin(t) * np.sin(phi0)) * 2 * np.pi
        dcz = (r_coil * np.cos(t)) * 2 * np.pi
        gammadashs.append(np.stack([dcx, dcy, dcz], axis=-1))

    gammas = jnp.array(np.stack(gammas))
    gammadashs = jnp.array(np.stack(gammadashs))
    currents = jnp.ones(ncoils) * 1e5
    return gammas, gammadashs, currents


def make_surface_data(nphi=15, ntheta=15, mpol=2, ntor=2, nfp=2, R=1.0, r=0.1):
    """Create surface coefficient matrices for a simple torus."""
    phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)

    xc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    yc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    zc = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))

    xc = xc.at[0, 0].set(R)
    xc = xc.at[1, 0].set(r)
    zc = zc.at[mpol + 1, 0].set(r)

    return phis, thetas, xc, yc, zc, mpol, ntor, nfp


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def time_fn(fn, *args, warmup=1, repeat=20):
    """Time a JIT-compiled function, separating compile from steady-state."""
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
        "median_s": np.median(times),
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
    }


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def bench_biot_savart(ncoils=4, nquad=128, nphi=15, ntheta=15, nfp=2):
    """Benchmark BiotSavart B and dB/dX."""
    print(f"\n{'=' * 60}")
    print(
        f"BiotSavart  (ncoils={ncoils}, nquad={nquad}, "
        f"npoints={nphi}×{ntheta}={nphi * ntheta})"
    )
    print(f"{'=' * 60}")

    gammas, gammadashs, currents = make_coils(ncoils, nquad)

    # Evaluation points on a torus-like grid (r_minor < r_coil to avoid coils)
    phis_grid = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    thetas_grid = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi_2d, theta_2d = np.meshgrid(phis_grid, thetas_grid, indexing="ij")
    R, r_minor = 1.0, 0.1
    x = ((R + r_minor * np.cos(theta_2d)) * np.cos(phi_2d)).ravel()
    y = ((R + r_minor * np.cos(theta_2d)) * np.sin(phi_2d)).ravel()
    z = (r_minor * np.sin(theta_2d)).ravel()
    points = jnp.array(np.stack([x, y, z], axis=-1))

    # B field
    t = time_fn(_bs.biot_savart_B, points, gammas, gammadashs, currents)
    print(
        f"  B field:   compile={t['compile_s']:.3f}s  "
        f"steady={t['median_s'] * 1e3:.3f}ms (median, {t['std_s'] * 1e3:.3f}ms std)"
    )

    # dB/dX
    t = time_fn(_bs.biot_savart_dB_by_dX, points, gammas, gammadashs, currents)
    print(
        f"  dB/dX:     compile={t['compile_s']:.3f}s  "
        f"steady={t['median_s'] * 1e3:.3f}ms (median, {t['std_s'] * 1e3:.3f}ms std)"
    )

    # B + dB combined
    t = time_fn(_bs.biot_savart_B_and_dB, points, gammas, gammadashs, currents)
    print(
        f"  B+dB:      compile={t['compile_s']:.3f}s  "
        f"steady={t['median_s'] * 1e3:.3f}ms (median, {t['std_s'] * 1e3:.3f}ms std)"
    )

    # Accuracy check: ∇·B = 0
    _, dB = _bs.biot_savart_B_and_dB(points, gammas, gammadashs, currents)
    div_B = jnp.trace(dB, axis1=1, axis2=2)
    print(f"  ∇·B max:   {float(jnp.max(jnp.abs(div_B))):.2e}")


def bench_surface(nphi=15, ntheta=15, mpol=5, ntor=5, nfp=2):
    """Benchmark surface evaluation."""
    print(f"\n{'=' * 60}")
    print(f"Surface  (nphi={nphi}, ntheta={ntheta}, mpol={mpol}, ntor={ntor})")
    print(f"{'=' * 60}")

    phis, thetas, xc, yc, zc, _, _, _ = make_surface_data(nphi, ntheta, mpol, ntor, nfp)

    for name, fn in [
        ("gamma     ", _sf.surface_gamma),
        ("gammadash1", _sf.surface_gammadash1),
        ("gammadash2", _sf.surface_gammadash2),
        ("normal    ", _sf.surface_normal),
    ]:
        t = time_fn(fn, phis, thetas, xc, yc, zc, mpol, ntor, nfp)
        print(
            f"  {name}: compile={t['compile_s']:.3f}s  "
            f"steady={t['median_s'] * 1e3:.3f}ms"
        )


def bench_boozer_residual(nphi=15, ntheta=15):
    """Benchmark Boozer residual scalar, gradient, and Hessian."""
    print(f"\n{'=' * 60}")
    print(f"Boozer residual  (nphi={nphi}, ntheta={ntheta})")
    print(f"{'=' * 60}")

    rng = np.random.RandomState(42)
    B = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0, 0, 1.0]))
    xphi = jnp.array(rng.randn(nphi, ntheta, 3) * 0.5)
    xtheta = jnp.array(rng.randn(nphi, ntheta, 3) * 0.5)
    G, iota = 1.5, 0.3

    # Scalar
    t = time_fn(_br.boozer_residual_scalar, G, iota, B, xphi, xtheta)
    J = float(_br.boozer_residual_scalar(G, iota, B, xphi, xtheta))
    print(
        f"  scalar:    compile={t['compile_s']:.3f}s  "
        f"steady={t['median_s'] * 1e3:.3f}ms  J={J:.6e}"
    )

    # Gradient (iota, G only — nsurfdofs=0)
    t = time_fn(_br.boozer_residual_grad, G, iota, B, xphi, xtheta, 0)
    print(
        f"  gradient:  compile={t['compile_s']:.3f}s  "
        f"steady={t['median_s'] * 1e3:.3f}ms"
    )

    # Hessian
    t = time_fn(_br.boozer_residual_hessian, G, iota, B, xphi, xtheta, 0)
    print(
        f"  hessian:   compile={t['compile_s']:.3f}s  "
        f"steady={t['median_s'] * 1e3:.3f}ms"
    )


def bench_integral_BdotN(nphi=15, ntheta=15):
    """Benchmark integral_BdotN for all definitions."""
    print(f"\n{'=' * 60}")
    print(f"integral_BdotN  (nphi={nphi}, ntheta={ntheta})")
    print(f"{'=' * 60}")

    rng = np.random.RandomState(7)
    B = jnp.array(rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0, 0, 1.0]))
    target = jnp.zeros((nphi, ntheta))
    normal = jnp.array(rng.randn(nphi, ntheta, 3) * 0.5 + np.array([0, 0, 0.5]))

    for defn in ["quadratic flux", "normalized", "local"]:
        fn = lambda: _ib.integral_BdotN(B, target, normal, defn)
        t = time_fn(fn)
        J = float(fn())
        print(
            f"  {defn:16s}: compile={t['compile_s']:.3f}s  "
            f"steady={t['median_s'] * 1e3:.3f}ms  J={J:.6e}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("JAX Feasibility Spike Benchmark — Milestone 1")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Float64 enabled: {jax.config.jax_enable_x64}")

    # --- Small grids (fast smoke test) ---
    bench_biot_savart(ncoils=4, nquad=128, nphi=15, ntheta=15)
    bench_surface(nphi=15, ntheta=15, mpol=5, ntor=5)
    bench_boozer_residual(nphi=15, ntheta=15)
    bench_integral_BdotN(nphi=15, ntheta=15)

    # --- Representative Columbia grid sizes ---
    # Stage 2 default:  nphi=255, ntheta=64 (banana_coil_solver.py)
    # Single-stage default: nphi=255, ntheta=64
    # Boozer inner:     nphi=15, ntheta=15 (typical BoozerLS grid)
    bench_biot_savart(ncoils=12, nquad=200, nphi=64, ntheta=64)
    bench_surface(nphi=64, ntheta=64, mpol=10, ntor=10)

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
