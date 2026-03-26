"""
Wave 2.2 — Evaluate default Optimistix BFGS against _minimize_bfgs_private.

Compares the two solvers on the same Boozer penalty objective from a synthetic
mock fixture (8x8 quadrature, 2 coils, 29-DOF decision vector).

Prerequisites:
    pip install optimistix==0.1.0   # benchmark-only, not in [JAX] extras

Usage:
    conda run -n columbia-jax-0.9.2 python benchmarks/optimistix_eval.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import optimistix as optx

# Deferred to _init_modules() to avoid sys.modules pollution on import.
_SRC = Path(__file__).resolve().parents[1] / "src" / "simsopt"

_boozer_penalty_objective = None
_minimize_bfgs_private = None


def _init_modules():
    """Load JAX modules via importlib (no simsoptpp) and bind module-level refs."""
    global _boozer_penalty_objective, _minimize_bfgs_private

    def _ensure_package(pkg, path):
        if pkg in sys.modules:
            return
        try:
            __import__(pkg)
        except ImportError:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m

    def _load_and_register(module_fqn, relpath):
        spec = importlib.util.spec_from_file_location(module_fqn, str(_SRC / relpath))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_fqn] = mod
        spec.loader.exec_module(mod)
        return mod

    _ensure_package("simsopt", _SRC)
    _ensure_package("simsopt.geo", _SRC / "geo")
    _ensure_package("simsopt.field", _SRC / "field")

    _load_and_register("simsopt.geo.surface_fourier_jax", "geo/surface_fourier_jax.py")
    _load_and_register("simsopt.field.biotsavart_jax", "field/biotsavart_jax.py")
    _load_and_register("simsopt.geo.boozer_residual_jax", "geo/boozer_residual_jax.py")
    _load_and_register(
        "simsopt.geo.label_constraints_jax", "geo/label_constraints_jax.py"
    )
    _opt = _load_and_register("simsopt.geo.optimizer_jax", "geo/optimizer_jax.py")
    _bsj = _load_and_register(
        "simsopt.geo.boozersurface_jax", "geo/boozersurface_jax.py"
    )

    _boozer_penalty_objective = _bsj._boozer_penalty_objective
    _minimize_bfgs_private = _opt._minimize_bfgs_private


def _make_simple_torus_coeffs(R0=1.0, r=0.1, mpol=1, ntor=1, nfp=1):
    shape = (2 * mpol + 1, 2 * ntor + 1)
    xc = np.zeros(shape)
    yc = np.zeros(shape)
    zc = np.zeros(shape)
    xc[0, 0] = R0
    xc[1, 0] = r
    zc[mpol + 1, 0] = r
    return xc, yc, zc


def _make_mock_coils(nquad=64):
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    R = 1.0
    gammas, gammadashs, currents = [], [], []
    for z_off, cur in [(0.3, 1e5), (-0.3, 1e5)]:
        g = np.stack(
            [R * np.cos(phi), R * np.sin(phi), z_off * np.ones(nquad)], axis=-1
        )
        gd = np.stack(
            [
                -R * np.sin(phi) * 2 * np.pi,
                R * np.cos(phi) * 2 * np.pi,
                np.zeros(nquad),
            ],
            axis=-1,
        )
        gammas.append(g)
        gammadashs.append(gd)
        currents.append(cur)
    return (
        jnp.array(np.stack(gammas)),
        jnp.array(np.stack(gammadashs)),
        jnp.array(currents),
    )


def _build_fixture(nphi=8, ntheta=8, mpol=1, ntor=1, nfp=1):
    """Build pure-function Boozer penalty objective + initial decision vector."""
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
    gammas, gammadashs, currents = _make_mock_coils()

    targetlabel = 2.0 * np.pi**2 * R0 * r**2
    phi_idx = jnp.arange(nphi)
    coil_arrays = [(gammas, gammadashs, currents)]

    def penalty_objective(x):
        return _boozer_penalty_objective(
            x,
            coil_arrays=coil_arrays,
            quadpoints_phi=qphi,
            quadpoints_theta=qtheta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=False,
            scatter_indices=None,
            targetlabel=targetlabel,
            constraint_weight=1.0,
            label_type="volume",
            phi_idx=phi_idx,
            optimize_G=True,
            weight_inv_modB=True,
        )

    x0 = jnp.concatenate([jnp.array(sdofs), jnp.array([0.3, 0.05])])
    return penalty_objective, x0


def run_private_bfgs(fun, x0, *, gtol=1e-10, maxiter=1500):
    """Run the private BFGS solver. Timer includes solve + final g_inf."""
    t0 = time.perf_counter()
    state = _minimize_bfgs_private(fun, x0, gtol=gtol, maxiter=maxiter)
    state.x_k.block_until_ready()
    g_final = np.asarray(state.g_k)
    g_inf = float(np.linalg.norm(g_final, ord=np.inf))
    wall = time.perf_counter() - t0

    return {
        "solver": "private_bfgs",
        "f": float(state.f_k),
        "g_inf": g_inf,
        "nit": int(state.k),
        "nfev": int(state.nfev),
        "converged": bool(state.converged),
        "failed": bool(state.failed),
        "status": int(state.status),
        "wall_s": wall,
        "x": np.asarray(state.x_k),
    }


def run_optimistix_bfgs(
    fun, x0, *, rtol=1e-10, atol=1e-10, max_steps=1500, gtol_target=None
):
    """Run Optimistix BFGS. Timer includes solve + post-solve g_inf evaluation.

    If gtol_target is set, adds a gradient-norm post-check to the result.
    """

    def fn(y, args):
        return fun(y)

    solver = optx.BFGS(rtol=rtol, atol=atol)

    t0 = time.perf_counter()
    sol = optx.minimise(fn, solver, x0, max_steps=max_steps, throw=False)
    sol.value.block_until_ready()
    f_final, g_final = jax.value_and_grad(fun)(sol.value)
    g_final = np.asarray(g_final)
    g_inf = float(np.linalg.norm(g_final, ord=np.inf))
    wall = time.perf_counter() - t0

    label = f"optimistix_bfgs(rtol={rtol:.0e}, atol={atol:.0e})"
    num_steps = int(sol.stats["num_steps"])
    num_accepted = int(sol.stats.get("num_accepted_steps", num_steps))
    result = {
        "solver": label + (" + g_norm check" if gtol_target is not None else ""),
        "f": float(f_final),
        "g_inf": g_inf,
        "nit": num_steps,
        "num_accepted": num_accepted,
        "converged": bool(sol.result == optx.RESULTS.successful),
        "result_code": str(sol.result),
        "wall_s": wall,
        "x": np.asarray(sol.value),
    }
    if gtol_target is not None:
        result["g_inf_meets_gtol"] = g_inf < gtol_target
    return result


def _print_result(r):
    print(f"  {r['solver']}")
    print(f"    f         = {r['f']:.16e}")
    print(f"    ||g||_inf = {r['g_inf']:.6e}")
    print(f"    steps     = {r['nit']}")
    if "nfev" in r:
        print(f"    nfev      = {r['nfev']}")
    if "num_accepted" in r:
        print(f"    accepted  = {r['num_accepted']}")
    print(f"    converged = {r['converged']}")
    if "failed" in r:
        print(f"    failed    = {r['failed']}  (status={r['status']})")
    if "result_code" in r:
        print(f"    result    = {r['result_code']}")
    if "g_inf_meets_gtol" in r:
        print(f"    g_inf < 1e-10 = {r['g_inf_meets_gtol']}")
    print(f"    wall      = {r['wall_s']:.3f}s")


def main():
    _init_modules()

    print("=" * 72)
    print("Wave 2.2: Optimistix BFGS Evaluation for Inner Boozer Solve")
    print("=" * 72)
    print()

    print("Building fixture...")
    fun, x0 = _build_fixture()
    print(f"  Decision vector size: {x0.shape[0]}")

    f0, g0 = jax.value_and_grad(fun)(x0)
    print(f"  Initial objective:    {float(f0):.6e}")
    print(f"  Initial ||g||_inf:    {float(jnp.linalg.norm(g0, ord=jnp.inf)):.6e}")
    print()

    print("--- Private BFGS (gtol=1e-10) ---")
    ref = run_private_bfgs(fun, x0, gtol=1e-10)
    _print_result(ref)
    print()

    print("--- Optimistix BFGS (rtol=1e-10, atol=1e-10) ---")
    r1 = run_optimistix_bfgs(fun, x0, rtol=1e-10, atol=1e-10)
    _print_result(r1)
    print()

    print("--- Optimistix BFGS (rtol=1e-13, atol=1e-13) ---")
    r2 = run_optimistix_bfgs(fun, x0, rtol=1e-13, atol=1e-13)
    _print_result(r2)
    print()

    print("--- Optimistix BFGS (rtol=1e-14, atol=1e-14) + g_norm post-check ---")
    r3 = run_optimistix_bfgs(fun, x0, rtol=1e-14, atol=1e-14, gtol_target=1e-10)
    _print_result(r3)
    print()

    print("=" * 72)
    print("COMPARISON SUMMARY")
    print("=" * 72)
    print()

    print("  NOTE: Neither solver fully converges on this synthetic fixture.")
    print(f"  Private BFGS: converged={ref['converged']}, failed={ref.get('failed')}")
    print(f"  Optimistix:   converged={r2['converged']}")
    print()

    for label, r in [
        ("optx(1e-10)", r1),
        ("optx(1e-13)", r2),
        ("optx(1e-14)+check", r3),
    ]:
        f_diff = abs(r["f"] - ref["f"])
        print(f"  {label} vs private_bfgs:")
        print(f"    |f_diff|   = {f_diff:.6e}")
        print(f"    g_inf ratio = {r['g_inf'] / max(ref['g_inf'], 1e-30):.2f}")
        print()

    print("=" * 72)
    print("TERMINATION + LINE SEARCH DIFFERENCES")
    print("=" * 72)
    print()
    print("  Private BFGS:")
    print("    Termination: ||g||_inf < gtol (gradient-norm)")
    print("    Line search: Strong Wolfe (cubic/quad zoom)")
    print()
    print("  Optimistix BFGS:")
    print(
        "    Termination: Cauchy (|y_diff| < atol+rtol*|y| AND |f_diff| < atol+rtol*|f|)"
    )
    print("    Line search: BacktrackingArmijo (halving, slope=0.1)")
    print("    Note: Optimistix TODO in source to replace BacktrackingArmijo")
    print()

    for label, r in [("optx(1e-10)", r1), ("optx(1e-13)", r2), ("optx(1e-14)", r3)]:
        status = "PASS" if r["g_inf"] < 1e-10 else "FAIL"
        print(f"  {label}: ||g||_inf = {r['g_inf']:.3e}  [{status} vs gtol=1e-10]")
    print()

    print("=" * 72)
    print("OBSERVATIONS")
    print("=" * 72)
    print()
    print("  1. On this fixture, default optx.BFGS reaches a higher final gradient")
    print("     norm (~4.2e-9) than the private BFGS (~4.5e-10) across all tested")
    print("     tolerance settings (1e-10 through 1e-14).")
    print()
    print("  2. The private BFGS ultimately fails (line search status=3) rather")
    print("     than converging. Both solvers struggle on this problem.")
    print()
    print("  3. The two solvers differ in termination criterion AND line search")
    print("     strategy simultaneously. This benchmark does not isolate which")
    print("     factor dominates. A controlled ablation (e.g., swapping only the")
    print("     line search) would be needed to establish causation.")
    print()
    print("  4. Optimistix's BacktrackingArmijo lacks the curvature condition")
    print("     that the Wolfe line search provides. The Optimistix source")
    print("     includes a TODO to replace it. A future Optimistix release with")
    print("     a Wolfe line search could change these results.")
    print()

    print("=" * 72)
    print("DECISION")
    print("=" * 72)
    print()
    print("  Keep _minimize_bfgs_private for the inner Boozer solve.")
    print()
    print("  Rationale: under default settings on this fixture, the private")
    print("  solver reaches ~10x lower gradient norm before exit. The private")
    print("  solver is the known quantity with validated behavior on production")
    print("  fixtures (Waves 2.1-2.4). Adopting Optimistix BFGS would trade a")
    print("  tested implementation for one that underperforms on the available")
    print("  benchmark without a clear corrective path today.")
    print()
    print("  Caveats:")
    print("  - This is a single synthetic fixture, not a production config.")
    print("  - Neither solver converges; comparison is between two failure modes.")
    print("  - Re-evaluate if Optimistix ships a Wolfe line search.")
    print()

    print("=" * 72)
    print("USEFUL FROM OPTIMISTIX ECOSYSTEM (Phase 3)")
    print("=" * 72)
    print()
    print("  1. ImplicitAdjoint for automatic IFT through solves (jax.custom_vjp)")
    print("  2. Lineax for GMRES/iterative linear solves (newton_polish replacement)")
    print("  Note: optimistix is a benchmark-only dep (pip install optimistix==0.1.0),")
    print("  not in the [JAX] or [JAX_GPU] public extras.")


if __name__ == "__main__":
    main()
