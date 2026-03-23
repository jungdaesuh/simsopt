"""Shared helpers for end-to-end ``BoozerSurfaceJAX.run_code()`` benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from benchmarks.benchmark_config import BenchmarkConfig, DEFAULT_CONFIGS
from benchmarks.benchmark_problem import build_synthetic_boozer_problem
from benchmarks.validation_ladder_common import (
    current_compilation_cache_metadata,
    describe_compile_behavior,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARK_JAX_VERSION = os.environ.get(
    "SIMSOPT_BENCHMARK_JAX_VERSION", "0.9.2"
)
BENCHMARK_BACKEND_CHOICES = ("scipy", "ondevice", "hybrid")
DEFAULT_PUBLIC_BACKENDS = ("scipy",)
PRIVATE_ONLY_BACKENDS = frozenset({"ondevice", "hybrid"})
SOLVER_VERBOSE = os.environ.get("SIMSOPT_BENCHMARK_SOLVER_VERBOSE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _progress(message: str) -> None:
    print(message, flush=True)


def _get_git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _current_jax_version() -> str:
    return jax.__version__


def _x64_enabled() -> bool:
    return jnp.zeros(1).dtype == jnp.float64


def _requested_private_backends(backends: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(PRIVATE_ONLY_BACKENDS.intersection(backends)))


def _resolve_runtime_lane(backends: tuple[str, ...]) -> str:
    if _requested_private_backends(backends):
        return "private-optimizer"
    return "trusted-public-reference"


def _validate_benchmark_runtime(backends: tuple[str, ...]) -> None:
    if not _x64_enabled():
        raise RuntimeError("Expected JAX x64 mode to be enabled for this benchmark.")

    version = _current_jax_version()
    private_backends = _requested_private_backends(backends)
    if version != EXPECTED_BENCHMARK_JAX_VERSION:
        requested = ", ".join(private_backends or backends)
        lane_label = f"benchmark backends {requested}" if requested else "benchmark runtime"
        raise RuntimeError(
            f"{lane_label} are configured for JAX "
            f"{EXPECTED_BENCHMARK_JAX_VERSION}; found {version}. "
            "Set SIMSOPT_BENCHMARK_JAX_VERSION only when intentionally "
            "validating a different benchmark runtime."
        )


def resolve_benchmark_backends(requested_backends=None) -> tuple[str, ...]:
    if requested_backends:
        backends = tuple(requested_backends)
    else:
        backends = DEFAULT_PUBLIC_BACKENDS
    _validate_benchmark_runtime(backends)
    return backends


def print_provenance(title: str, backends: tuple[str, ...]) -> None:
    _validate_benchmark_runtime(backends)
    compilation_cache = current_compilation_cache_metadata()
    _progress(f"\n{'=' * 70}")
    _progress(title)
    _progress(f"{'=' * 70}")
    _progress(f"repo sha:     {_get_git_sha()}")
    _progress(f"jax:          {jax.__version__}")
    _progress(f"jaxlib:       {jaxlib.__version__}")
    _progress(f"backend:      {jax.default_backend()}")
    _progress(f"devices:      {jax.devices()}")
    _progress(f"x64 enabled:  {_x64_enabled()}")
    _progress(f"lane:         {_resolve_runtime_lane(backends)}")
    _progress(f"backends:     {', '.join(backends)}")
    _progress(f"compile:      {describe_compile_behavior(uses_subprocesses=False)}")
    _progress(f"cache policy: {compilation_cache['compilation_cache_policy']}")
    if compilation_cache["compilation_cache_dir"] is not None:
        _progress(f"cache dir:    {compilation_cache['compilation_cache_dir']}")


def _make_boozer_surface(
    config: BenchmarkConfig,
    optimizer_backend: str,
    *,
    option_overrides: dict | None = None,
):
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

    problem = build_synthetic_boozer_problem(config)
    bs_jax = BiotSavartJAX(problem.coils)
    options = {
        "verbose": SOLVER_VERBOSE,
        "bfgs_maxiter": 50,
        "bfgs_tol": 1e-8,
        "newton_maxiter": 10,
        "newton_tol": 1e-9,
        "optimizer_backend": optimizer_backend,
    }
    if option_overrides:
        options.update(option_overrides)

    booz = BoozerSurfaceJAX(
        bs_jax,
        problem.surface,
        problem.volume,
        problem.vol_target,
        constraint_weight=1.0,
        options=options,
    )
    return booz, problem.iota0, problem.G0


def _sync_result(res: dict) -> None:
    if res is None:
        return
    for key in ("fun", "jacobian", "hessian", "residual"):
        value = res.get(key)
        if value is not None:
            jax.block_until_ready(jnp.asarray(value))
    info = res.get("info")
    if info is not None:
        for attr in ("x", "jac"):
            value = getattr(info, attr, None)
            if value is not None:
                jax.block_until_ready(jnp.asarray(value))


def summarize_result_fun(res: dict) -> float:
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


def time_run_code(config: BenchmarkConfig, optimizer_backend: str, *, option_overrides=None):
    _progress(f"    [{optimizer_backend}] building run_code problem")
    booz, iota0, G0 = _make_boozer_surface(
        config,
        optimizer_backend,
        option_overrides=option_overrides,
    )
    _progress(f"    [{optimizer_backend}] running full run_code()")
    t0 = time.perf_counter()
    res = booz.run_code(iota0, G0)
    _sync_result(res)
    _progress(f"    [{optimizer_backend}] full run_code() finished")
    return time.perf_counter() - t0, res


def time_run_code_stage_split(
    config: BenchmarkConfig,
    optimizer_backend: str,
    *,
    option_overrides=None,
):
    _progress(f"    [{optimizer_backend}] building stage-split problem")
    booz, iota0, G0 = _make_boozer_surface(
        config,
        optimizer_backend,
        option_overrides=option_overrides,
    )

    _progress(f"    [{optimizer_backend}] running LS stage")
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
    _sync_result(ls_res)
    ls_time = time.perf_counter() - t0

    booz.need_to_run_code = True
    _progress(f"    [{optimizer_backend}] LS stage finished; running Newton stage")
    t1 = time.perf_counter()
    res = booz.minimize_boozer_penalty_constraints_newton(
        constraint_weight=booz.constraint_weight,
        iota=ls_res["iota"],
        G=ls_res["G"],
        verbose=booz.options["verbose"],
        tol=booz.options["newton_tol"],
        maxiter=booz.options["newton_maxiter"],
        stab=booz.options["newton_stab"],
        weight_inv_modB=booz.options["weight_inv_modB"],
    )
    _sync_result(res)
    newton_time = time.perf_counter() - t1
    _progress(f"    [{optimizer_backend}] Newton stage finished")
    return ls_time, newton_time, res


def benchmark_backend(
    config: BenchmarkConfig,
    optimizer_backend: str,
    *,
    repeats: int,
    option_overrides: dict | None = None,
):
    _progress(f"  backend={optimizer_backend}")
    compile_time, compile_res = time_run_code(
        config,
        optimizer_backend,
        option_overrides=option_overrides,
    )
    ls_time, newton_time, _ = time_run_code_stage_split(
        config,
        optimizer_backend,
        option_overrides=option_overrides,
    )
    repeat_times = []
    repeat_res = compile_res
    for repeat_index in range(repeats):
        _progress(
            f"    [{optimizer_backend}] repeat fresh solve "
            f"{repeat_index + 1}/{repeats}"
        )
        elapsed, repeat_res = time_run_code(
            config,
            optimizer_backend,
            option_overrides=option_overrides,
        )
        repeat_times.append(elapsed)
    _progress(f"    [{optimizer_backend}] repeats finished")
    return compile_time, ls_time, newton_time, np.asarray(repeat_times), repeat_res


def run_benchmarks(
    *,
    title: str,
    configs=DEFAULT_CONFIGS,
    backends=DEFAULT_PUBLIC_BACKENDS,
    repeats: int = 3,
    option_overrides: dict | None = None,
) -> None:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    summaries: dict[str, dict[str, float]] = {}

    _progress(f"\n{'=' * 70}")
    _progress(title)
    _progress(f"{'=' * 70}")
    _progress(
        "Diagnostic benchmark only: short solver budgets on a synthetic problem. "
        "Use benchmarks/run_code_parity_probe.py for CPU/JAX correctness parity."
    )

    for config in configs:
        _progress(f"\n{'=' * 70}")
        _progress(f"run_code() benchmark: {config.label}")
        _progress(
            f"  grid: {config.nphi}x{config.ntheta}, surface: "
            f"mpol={config.mpol} ntor={config.ntor}, coils={config.ncoils}"
        )
        _progress(f"{'=' * 70}")

        backend_summary: dict[str, float] = {}
        for optimizer_backend in backends:
            compile_time, ls_time, newton_time, repeat_times, res = benchmark_backend(
                config,
                optimizer_backend,
                repeats=repeats,
                option_overrides=option_overrides,
            )
            backend_summary[optimizer_backend] = float(np.median(repeat_times))
            _progress(
                f"    first call:  {compile_time:.3f}s  "
                f"success={res['success']}  iter={res['iter']}"
            )
            _progress(
                f"    repeat fresh solve: {np.median(repeat_times) * 1e3:.1f}ms median, "
                f"{np.mean(repeat_times) * 1e3:.1f}ms mean ± "
                f"{np.std(repeat_times) * 1e3:.1f}ms"
            )
            _progress(
                f"    stage split sample: LS {ls_time * 1e3:.1f}ms, "
                f"Newton {newton_time * 1e3:.1f}ms"
            )
            _progress(
                f"    final fun:   {summarize_result_fun(res):.6e}  "
                f"iota={float(res['iota']):.6f}"
            )
            if not res["success"]:
                _progress(
                    "    warning: unconverged solve; treat timing as diagnostic only, "
                    "not as a parity or replacement verdict"
                )

        summaries[config.label] = backend_summary
        if "scipy" in backend_summary and "ondevice" in backend_summary:
            speedup = backend_summary["scipy"] / backend_summary["ondevice"]
            _progress(f"  repeat fresh-solve speedup (ondevice/scipy): {speedup:.2f}x")
        if "scipy" in backend_summary and "hybrid" in backend_summary:
            speedup = backend_summary["scipy"] / backend_summary["hybrid"]
            _progress(f"  repeat fresh-solve speedup (hybrid/scipy):   {speedup:.2f}x")

    if "scipy" in backends and "ondevice" in backends:
        break_even = next(
            (
                label
                for label, values in summaries.items()
                if values["ondevice"] <= values["scipy"]
            ),
            None,
        )
        _progress(f"\n{'=' * 70}")
        _progress("BREAK-EVEN SUMMARY")
        _progress(f"{'=' * 70}")
        if break_even is None:
            _progress(
                "No tested configuration reached ondevice <= scipy repeat fresh-solve time."
            )
        else:
            _progress(
                "First tested configuration with ondevice <= scipy "
                f"repeat fresh-solve time: {break_even}"
            )
