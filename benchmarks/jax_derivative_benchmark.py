"""
Benchmark harness for M3 JAX Boozer derivative path.

Measures compile-time vs steady-state runtime for:
1. Surface coefficient Jacobians (dgamma_by_dcoeff, etc.)
2. Composed penalty scalar (forward)
3. Composed penalty gradient (VJP)
4. Composed residual Jacobian (jacfwd)
5. Composed penalty Hessian

Usage:
    conda run -n <conda-env> python benchmarks/jax_derivative_benchmark.py
"""

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from repo_bootstrap import configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(sys.argv[1:])

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_SRC = Path(__file__).resolve().parents[1] / "src" / "simsopt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    return parser.parse_known_args()[0]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Inject stub packages for lazy imports
for _pkg in ["simsopt", "simsopt.geo", "simsopt.field"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

_sf = _load("surface_fourier_jax", "geo/surface_fourier_jax.py")
sys.modules["simsopt.geo.surface_fourier_jax"] = _sf

_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
sys.modules["simsopt.field.biotsavart_jax"] = _bs

_br = _load("boozer_residual_jax", "geo/boozer_residual_jax.py")


def _make_torus_dofs(mpol, ntor, nfp, R=1.0, r=0.1):
    n_per = (2 * mpol + 1) * (2 * ntor + 1)
    dofs = np.zeros(3 * n_per)
    ncol = 2 * ntor + 1
    dofs[0 * ncol + 0] = R
    dofs[1 * ncol + 0] = r
    dofs[2 * n_per + (mpol + 1) * ncol + 0] = r
    return jnp.array(dofs)


def _make_coil_data(ncoils, nquad):
    R_coil = 1.5
    gammas = np.zeros((ncoils, nquad, 3))
    gammadashs = np.zeros((ncoils, nquad, 3))
    for i in range(ncoils):
        phi_off = 2 * np.pi * i / ncoils
        t = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
        gammas[i, :, 0] = R_coil * np.cos(t + phi_off)
        gammas[i, :, 1] = R_coil * np.sin(t + phi_off)
        gammadashs[i, :, 0] = -R_coil * np.sin(t + phi_off) * 2 * np.pi
        gammadashs[i, :, 1] = R_coil * np.cos(t + phi_off) * 2 * np.pi
    currents = np.full(ncoils, 1e5)
    return jnp.array(gammas), jnp.array(gammadashs), jnp.array(currents)


def _bench(label, fn, n_warmup=2, n_iter=10):
    """Run benchmark: separate compile from steady-state."""
    # Compile warmup
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    # Additional warmup
    for _ in range(n_warmup - 1):
        result = fn()
        jax.block_until_ready(result)

    # Steady-state timing
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    mean_t = np.mean(times)
    std_t = np.std(times)
    print(
        f"  {label:45s}  compile={compile_time:.4f}s  "
        f"steady={mean_t * 1000:.2f}ms +/- {std_t * 1000:.2f}ms"
    )
    return compile_time, mean_t


def run_benchmarks():
    # --- Parameters ---
    mpol, ntor, nfp = 2, 2, 2
    nphi, ntheta = 16, 16
    ncoils, nquad = 6, 64

    print(f"Grid: nphi={nphi}, ntheta={ntheta}, mpol={mpol}, ntor={ntor}")
    print(f"Coils: ncoils={ncoils}, nquad={nquad}")

    phis = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    thetas = jnp.linspace(0, 1.0, ntheta, endpoint=False)
    dofs = _make_torus_dofs(mpol, ntor, nfp)
    cg, cgd, ci = _make_coil_data(ncoils, nquad)

    iota, G = 0.5, 2.0
    x = jnp.concatenate([dofs, jnp.array([iota, G])])

    kwargs_penalty = dict(
        coil_gammas=cg,
        coil_gammadashs=cgd,
        coil_currents=ci,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=False,
        scatter_indices=None,
        optimize_G=True,
        weight_inv_modB=False,
    )
    kwargs_residual = dict(
        coil_gammas=cg,
        coil_gammadashs=cgd,
        coil_currents=ci,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=False,
        scatter_indices=None,
        weight_inv_modB=False,
    )

    ndofs = len(dofs)
    print(f"Surface DOFs: {ndofs},  decision vector: {len(x)}")
    print()

    # 1. Surface coefficient Jacobians
    print("--- Surface coefficient Jacobians ---")
    _bench(
        "dgamma_by_dcoeff",
        lambda: _sf.dgamma_by_dcoeff(dofs, phis, thetas, mpol, ntor, nfp, False),
    )
    _bench(
        "dgammadash1_by_dcoeff",
        lambda: _sf.dgammadash1_by_dcoeff(dofs, phis, thetas, mpol, ntor, nfp, False),
    )
    _bench(
        "dgammadash2_by_dcoeff",
        lambda: _sf.dgammadash2_by_dcoeff(dofs, phis, thetas, mpol, ntor, nfp, False),
    )
    print()

    # 2. Composed forward
    print("--- Composed penalty objective ---")
    _bench(
        "boozer_penalty_composed (forward)",
        lambda: _br.boozer_penalty_composed(x, **kwargs_penalty),
    )
    print()

    # 3. Composed gradient (VJP)
    print("--- Composed gradient (VJP) ---")
    _bench(
        "boozer_penalty_grad_composed",
        lambda: _br.boozer_penalty_grad_composed(x, **kwargs_penalty),
    )
    print()

    # 4. Composed residual Jacobian (jacfwd)
    print("--- Composed residual Jacobian (jacfwd) ---")
    _bench(
        "boozer_residual_jacobian_composed",
        lambda: _br.boozer_residual_jacobian_composed(x, **kwargs_residual),
    )
    print()

    # 5. Composed Hessian
    print("--- Composed penalty Hessian ---")
    hess_fn = jax.hessian(_br.boozer_penalty_composed)
    _bench(
        "jax.hessian(boozer_penalty_composed)",
        lambda: hess_fn(x, **kwargs_penalty),
    )
    print()

    # Summary
    n_res = 3 * nphi * ntheta
    print(f"Residual size: {n_res}")
    print(f"Jacobian shape: ({n_res}, {len(x)})")
    print(f"Hessian shape: ({len(x)}, {len(x)})")


if __name__ == "__main__":
    _parse_args()
    run_benchmarks()
