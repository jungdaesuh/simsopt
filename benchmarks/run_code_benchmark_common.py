"""Shared helpers for end-to-end ``BoozerSurfaceJAX.run_code()`` benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

import numpy as np

import jax
import jaxlib
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_JAX_VERSION = "0.6.2"
DEFAULT_BACKENDS = ("scipy", "ondevice", "hybrid")


@dataclass(frozen=True)
class BenchmarkConfig:
    label: str
    ncoils: int
    nphi: int
    ntheta: int
    mpol: int
    ntor: int
    nfp: int = 1


DEFAULT_CONFIGS = (
    BenchmarkConfig("Small (4 coils, 15x15)", 4, 15, 15, 2, 2),
    BenchmarkConfig("Medium (6 coils, 15x15)", 6, 15, 15, 4, 4),
    BenchmarkConfig("HBT-like (12 coils, 15x15)", 12, 15, 15, 4, 4),
    BenchmarkConfig("Prod-grid (12 coils, 64x64)", 12, 64, 64, 4, 4),
    BenchmarkConfig("Columbia (12 coils, 128x64)", 12, 128, 64, 8, 6),
    BenchmarkConfig("Full-HBT (22 coils, 128x64)", 22, 128, 64, 8, 6),
)


def _get_git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def print_provenance(title: str) -> None:
    x64_enabled = jnp.zeros(1).dtype == jnp.float64
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"repo sha:     {_get_git_sha()}")
    print(f"jax:          {jax.__version__}")
    print(f"jaxlib:       {jaxlib.__version__}")
    print(f"backend:      {jax.default_backend()}")
    print(f"devices:      {jax.devices()}")
    print(f"x64 enabled:  {x64_enabled}")
    if jax.__version__ != EXPECTED_JAX_VERSION:
        raise RuntimeError(
            f"Expected JAX {EXPECTED_JAX_VERSION} for this benchmark, found {jax.__version__}."
        )
    if not x64_enabled:
        raise RuntimeError("Expected JAX x64 mode to be enabled for this benchmark.")


def _make_boozer_surface(config: BenchmarkConfig, optimizer_backend: str):
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import (
        SurfaceRZFourier,
        SurfaceXYZTensorFourier,
        Volume,
        create_equally_spaced_curves,
    )
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

    base_curves = create_equally_spaced_curves(
        config.ncoils,
        config.nfp,
        stellsym=False,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(config.ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        config.nfp,
        stellsym=False,
    )

    quadpoints_phi = np.linspace(0.0, 1.0 / config.nfp, config.nphi, endpoint=False)
    quadpoints_theta = np.linspace(0.0, 1.0, config.ntheta, endpoint=False)

    surface = SurfaceXYZTensorFourier(
        mpol=config.mpol,
        ntor=config.ntor,
        stellsym=False,
        nfp=config.nfp,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface = SurfaceRZFourier(
        nfp=config.nfp,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    seed_surface.set_rc(0, 0, 1.0)
    seed_surface.set_rc(1, 0, 0.15)
    seed_surface.set_zs(1, 0, 0.15)
    surface.least_squares_fit(seed_surface.gamma())

    bs_jax = BiotSavartJAX(coils)
    vol_cpu = Volume(surface)
    vol_target = vol_cpu.J()

    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
    iota0 = 0.3

    booz = BoozerSurfaceJAX(
        bs_jax,
        surface,
        vol_cpu,
        vol_target,
        constraint_weight=1.0,
        options={
            "verbose": False,
            "bfgs_maxiter": 50,
            "bfgs_tol": 1e-8,
            "newton_maxiter": 10,
            "newton_tol": 1e-9,
            "optimizer_backend": optimizer_backend,
        },
    )
    return booz, iota0, G0


def _sync_result(res: dict) -> None:
    for key in ("fun", "jacobian", "hessian", "residual"):
        value = res.get(key)
        if value is not None:
            jax.block_until_ready(jnp.asarray(value))


def time_run_code(config: BenchmarkConfig, optimizer_backend: str):
    booz, iota0, G0 = _make_boozer_surface(config, optimizer_backend)
    t0 = time.perf_counter()
    res = booz.run_code(iota0, G0)
    _sync_result(res)
    return time.perf_counter() - t0, res


def benchmark_backend(
    config: BenchmarkConfig,
    optimizer_backend: str,
    *,
    repeats: int,
):
    compile_time, compile_res = time_run_code(config, optimizer_backend)
    repeat_times = []
    repeat_res = compile_res
    for _ in range(repeats):
        elapsed, repeat_res = time_run_code(config, optimizer_backend)
        repeat_times.append(elapsed)
    return compile_time, np.asarray(repeat_times), repeat_res


def run_benchmarks(
    *,
    title: str,
    backends=DEFAULT_BACKENDS,
    repeats: int = 3,
) -> None:
    summaries: dict[str, dict[str, float]] = {}

    for config in DEFAULT_CONFIGS:
        print(f"\n{'=' * 70}")
        print(f"run_code() benchmark: {config.label}")
        print(
            f"  grid: {config.nphi}x{config.ntheta}, surface: "
            f"mpol={config.mpol} ntor={config.ntor}, coils={config.ncoils}"
        )
        print(f"{'=' * 70}")

        backend_summary: dict[str, float] = {}
        for optimizer_backend in backends:
            compile_time, repeat_times, res = benchmark_backend(
                config,
                optimizer_backend,
                repeats=repeats,
            )
            backend_summary[optimizer_backend] = float(np.median(repeat_times))
            print(f"  backend={optimizer_backend}")
            print(
                f"    first call:  {compile_time:.3f}s  "
                f"success={res['success']}  iter={res['iter']}"
            )
            print(
                f"    repeat fresh solve: {np.median(repeat_times) * 1e3:.1f}ms median, "
                f"{np.mean(repeat_times) * 1e3:.1f}ms mean ± "
                f"{np.std(repeat_times) * 1e3:.1f}ms"
            )
            print(
                f"    final fun:   {float(res['fun']):.6e}  "
                f"iota={float(res['iota']):.6f}"
            )

        summaries[config.label] = backend_summary
        if "scipy" in backend_summary and "ondevice" in backend_summary:
            speedup = backend_summary["scipy"] / backend_summary["ondevice"]
            print(f"  repeat fresh-solve speedup (ondevice/scipy): {speedup:.2f}x")
        if "scipy" in backend_summary and "hybrid" in backend_summary:
            speedup = backend_summary["scipy"] / backend_summary["hybrid"]
            print(f"  repeat fresh-solve speedup (hybrid/scipy):   {speedup:.2f}x")

    if "scipy" in backends and "ondevice" in backends:
        break_even = next(
            (
                label
                for label, values in summaries.items()
                if values["ondevice"] <= values["scipy"]
            ),
            None,
        )
        print(f"\n{'=' * 70}")
        print("BREAK-EVEN SUMMARY")
        print(f"{'=' * 70}")
        if break_even is None:
            print(
                "No tested configuration reached ondevice <= scipy repeat fresh-solve time."
            )
        else:
            print(
                "First tested configuration with ondevice <= scipy "
                f"repeat fresh-solve time: {break_even}"
            )
