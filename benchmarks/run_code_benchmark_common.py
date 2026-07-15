"""Shared helpers for end-to-end ``BoozerSurfaceJAX.run_code()`` benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from benchmarks.benchmark_config import BenchmarkConfig, DEFAULT_CONFIGS
from benchmarks.benchmark_problem import build_synthetic_boozer_problem
from benchmarks.validation_ladder_common import (
    current_compilation_cache_metadata,
    describe_compile_behavior,
)
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARK_JAX_VERSION = os.environ.get(
    "SIMSOPT_BENCHMARK_JAX_VERSION", "0.10.0"
)
BENCHMARK_BACKEND_CHOICES = ("scipy", "ondevice")
DEFAULT_PUBLIC_BACKENDS = ("ondevice",)
PRIVATE_ONLY_BACKENDS = frozenset({"ondevice"})
SOLVER_VERBOSE = os.environ.get("SIMSOPT_BENCHMARK_SOLVER_VERBOSE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True)
class BenchmarkRepeatResult:
    """One fresh solve's wall time and final solver outcome."""

    elapsed_seconds: float
    success: bool
    iterations: int
    final_fun: float
    final_iota: float


@dataclass(frozen=True)
class BenchmarkTimingResult:
    """Every timed fresh-solve repeat retained for diagnostics."""

    repeats: tuple[BenchmarkRepeatResult, ...]

    @property
    def elapsed_seconds(self) -> tuple[float, ...]:
        return tuple(repeat.elapsed_seconds for repeat in self.repeats)

    @property
    def median_seconds(self) -> float:
        return float(np.median(self.elapsed_seconds))


@dataclass(frozen=True)
class BenchmarkBackendResult:
    """Typed timings and solver outcomes for one optimizer backend."""

    first_call: BenchmarkRepeatResult
    least_squares_seconds: float
    newton_seconds: float
    stage_split: BenchmarkRepeatResult
    timed_repeats: BenchmarkTimingResult


@dataclass(frozen=True)
class BenchmarkTimingRatioAssessment:
    """Exploratory ratio for two internally comparable timed repeat sets."""

    observed_ratio: float | None
    time_reduction_percent: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkTimingAssessment:
    """Whether one timed repeat set has internally comparable outcomes."""

    diagnostic_comparable: bool
    reasons: tuple[str, ...]


def _validate_requested_backends(backends: tuple[str, ...]) -> None:
    unknown = tuple(
        backend for backend in backends if backend not in BENCHMARK_BACKEND_CHOICES
    )
    if not unknown:
        return
    valid = ", ".join(BENCHMARK_BACKEND_CHOICES)
    raise ValueError(
        f"optimizer_backend must be one of: {valid}. "
        f"Got unknown benchmark backend(s): {', '.join(unknown)}."
    )


@lru_cache(maxsize=1)
def _jax_modules():
    import jax
    import jaxlib
    import jax.numpy as jnp

    return jax, jaxlib, jnp


def artifact_host_value(value: Any) -> Any:
    jax, _, _ = _jax_modules()
    with jax.transfer_guard_device_to_host("allow"):
        return jax.device_get(value)


def artifact_host_array(value: Any, *, dtype: object | None = None) -> np.ndarray:
    return np.asarray(artifact_host_value(value), dtype=dtype)


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
    jax, _, _ = _jax_modules()
    return jax.__version__


def _x64_enabled() -> bool:
    _, _, jnp = _jax_modules()
    return jnp.zeros(1).dtype == jnp.float64


def _requested_private_backends(backends: tuple[str, ...]) -> tuple[str, ...]:
    _validate_requested_backends(backends)
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
        lane_label = (
            f"benchmark backends {requested}" if requested else "benchmark runtime"
        )
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
    _validate_requested_backends(backends)
    _validate_benchmark_runtime(backends)
    return backends


def print_provenance(title: str, backends: tuple[str, ...]) -> None:
    jax, jaxlib, _ = _jax_modules()
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
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

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
    jax, _, jnp = _jax_modules()
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


def summarize_benchmark_repeat(
    elapsed_seconds: float,
    res: dict,
) -> BenchmarkRepeatResult:
    """Pair one fresh-solve wall time with its claim-relevant outcome."""
    return BenchmarkRepeatResult(
        elapsed_seconds=elapsed_seconds,
        success=bool(res["success"]),
        iterations=int(res["iter"]),
        final_fun=summarize_result_fun(res),
        final_iota=float(res["iota"]),
    )


def _repeat_validity_reasons(
    timing_result: BenchmarkTimingResult,
    *,
    role: str,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not timing_result.repeats:
        reasons.append(f"{role} has no timed repeats")
    for repeat_index, repeat in enumerate(timing_result.repeats, start=1):
        repeat_label = f"{role} repeat {repeat_index}"
        if not repeat.success:
            reasons.append(f"{repeat_label} solver did not converge")
        if not np.isfinite(repeat.elapsed_seconds) or repeat.elapsed_seconds <= 0.0:
            reasons.append(f"{repeat_label} wall time is not finite and positive")
        if not np.isfinite(repeat.final_fun):
            reasons.append(f"{repeat_label} final objective is not finite")
        if not np.isfinite(repeat.final_iota):
            reasons.append(f"{repeat_label} final iota is not finite")
    return tuple(reasons)


def _repeat_outcome_comparison_reasons(
    repeat_results: tuple[BenchmarkRepeatResult, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []

    tolerances = parity_ladder_tolerances("gpu-runtime")
    whole_solve_rtol = float(tolerances["whole_solve_value_rtol"])
    whole_solve_atol = float(tolerances["whole_solve_value_atol"])
    final_fun_values = tuple(repeat.final_fun for repeat in repeat_results)
    if all(np.isfinite(value) for value in final_fun_values) and not all(
        np.isclose(
            left,
            right,
            rtol=whole_solve_rtol,
            atol=whole_solve_atol,
        )
        for left, right in combinations(final_fun_values, 2)
    ):
        reasons.append(
            "repeat final objectives are not mutually comparable under the "
            "whole-solve parity tolerance"
        )

    final_iota_values = tuple(repeat.final_iota for repeat in repeat_results)
    if all(np.isfinite(value) for value in final_iota_values) and not all(
        np.isclose(
            left,
            right,
            rtol=whole_solve_rtol,
            atol=whole_solve_atol,
        )
        for left, right in combinations(final_iota_values, 2)
    ):
        reasons.append(
            "repeat final iotas are not mutually comparable under the "
            "whole-solve parity tolerance"
        )
    return tuple(reasons)


def assess_benchmark_timing(
    timing_result: BenchmarkTimingResult,
    *,
    role: str,
) -> BenchmarkTimingAssessment:
    """Validate internal outcome comparability for one diagnostic timing set."""
    reasons = _repeat_validity_reasons(timing_result, role=role)
    reasons += _repeat_outcome_comparison_reasons(timing_result.repeats)
    return BenchmarkTimingAssessment(
        diagnostic_comparable=not reasons,
        reasons=reasons,
    )


def assess_benchmark_diagnostic_ratio(
    reference: BenchmarkTimingResult,
    candidate: BenchmarkTimingResult,
) -> BenchmarkTimingRatioAssessment:
    """Return an exploratory within-process timing ratio when outcomes match.

    Every repeat must converge with finite data, and all repeat objectives and
    iotas must mutually agree under the checked-in whole-solve tolerance. This
    diagnostic does not establish a performance claim: it has no independent
    paired process blocks or confidence interval.
    """
    reasons = _repeat_validity_reasons(reference, role="reference")
    reasons += _repeat_validity_reasons(candidate, role="candidate")
    reasons += _repeat_outcome_comparison_reasons(reference.repeats + candidate.repeats)

    if reasons:
        return BenchmarkTimingRatioAssessment(
            observed_ratio=None,
            time_reduction_percent=None,
            reasons=reasons,
        )
    observed_ratio = reference.median_seconds / candidate.median_seconds
    time_reduction_percent = (
        1.0 - candidate.median_seconds / reference.median_seconds
    ) * 100.0
    return BenchmarkTimingRatioAssessment(
        observed_ratio=observed_ratio,
        time_reduction_percent=time_reduction_percent,
        reasons=(),
    )


def time_run_code(
    config: BenchmarkConfig, optimizer_backend: str, *, option_overrides=None
):
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
) -> BenchmarkBackendResult:
    _progress(f"  backend={optimizer_backend}")
    compile_time, compile_res = time_run_code(
        config,
        optimizer_backend,
        option_overrides=option_overrides,
    )
    ls_time, newton_time, stage_res = time_run_code_stage_split(
        config,
        optimizer_backend,
        option_overrides=option_overrides,
    )
    repeat_results: list[BenchmarkRepeatResult] = []
    for repeat_index in range(repeats):
        _progress(
            f"    [{optimizer_backend}] repeat fresh solve {repeat_index + 1}/{repeats}"
        )
        elapsed, repeat_res = time_run_code(
            config,
            optimizer_backend,
            option_overrides=option_overrides,
        )
        repeat_results.append(summarize_benchmark_repeat(elapsed, repeat_res))
    _progress(f"    [{optimizer_backend}] repeats finished")
    return BenchmarkBackendResult(
        first_call=summarize_benchmark_repeat(compile_time, compile_res),
        least_squares_seconds=ls_time,
        newton_seconds=newton_time,
        stage_split=summarize_benchmark_repeat(ls_time + newton_time, stage_res),
        timed_repeats=BenchmarkTimingResult(repeats=tuple(repeat_results)),
    )


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

    _progress(f"\n{'=' * 70}")
    _progress(title)
    _progress(f"{'=' * 70}")
    _progress(
        "Diagnostic benchmark only: short solver budgets on a synthetic problem. "
        "Use benchmarks/run_code_parity_probe.py for CPU/JAX correctness parity."
    )
    _progress(
        "Raw timings and matched-outcome ratios are exploratory within-process "
        "diagnostics. Performance claims require independent paired process blocks, "
        "scientific acceptance gates, and a preregistered confidence interval."
    )

    for config in configs:
        _progress(f"\n{'=' * 70}")
        _progress(f"run_code() benchmark: {config.label}")
        _progress(
            f"  grid: {config.nphi}x{config.ntheta}, surface: "
            f"mpol={config.mpol} ntor={config.ntor}, coils={config.ncoils}"
        )
        _progress(f"{'=' * 70}")

        backend_summary: dict[str, BenchmarkTimingResult] = {}
        for optimizer_backend in backends:
            backend_result = benchmark_backend(
                config,
                optimizer_backend,
                repeats=repeats,
                option_overrides=option_overrides,
            )
            timing_result = backend_result.timed_repeats
            backend_summary[optimizer_backend] = timing_result
            repeat_times = np.asarray(timing_result.elapsed_seconds)
            _progress(
                f"    first call:  {backend_result.first_call.elapsed_seconds:.3f}s  "
                f"success={backend_result.first_call.success}  "
                f"iter={backend_result.first_call.iterations}"
            )
            _progress(
                f"    repeat fresh solve: {np.median(repeat_times) * 1e3:.1f}ms median, "
                f"{np.mean(repeat_times) * 1e3:.1f}ms mean ± "
                f"{np.std(repeat_times) * 1e3:.1f}ms"
            )
            _progress(
                "    stage split sample: LS "
                f"{backend_result.least_squares_seconds * 1e3:.1f}ms, "
                f"Newton {backend_result.newton_seconds * 1e3:.1f}ms  "
                f"success={backend_result.stage_split.success}  "
                f"iter={backend_result.stage_split.iterations}"
            )
            for repeat_index, repeat in enumerate(timing_result.repeats, start=1):
                _progress(
                    f"    repeat {repeat_index}: {repeat.elapsed_seconds * 1e3:.1f}ms  "
                    f"success={repeat.success}  iter={repeat.iterations}  "
                    f"fun={repeat.final_fun:.6e}  iota={repeat.final_iota:.6f}"
                )
            if not all(repeat.success for repeat in timing_result.repeats):
                _progress(
                    "    warning: at least one timed repeat did not converge; treat "
                    "all repeat timings as diagnostic only"
                )

        if "scipy" in backend_summary and "ondevice" in backend_summary:
            comparison = assess_benchmark_diagnostic_ratio(
                backend_summary["scipy"],
                backend_summary["ondevice"],
            )
            if comparison.observed_ratio is not None:
                assert comparison.time_reduction_percent is not None
                _progress(
                    "  diagnostic timing ratio only (not a performance claim): "
                    "scipy median / ondevice median = "
                    f"{comparison.observed_ratio:.2f}x; ondevice time reduction "
                    f"vs scipy = {comparison.time_reduction_percent:.2f}%"
                )
            else:
                _progress(
                    "  diagnostic timing ratio unavailable; timed outcomes are not "
                    f"mutually comparable: {'; '.join(comparison.reasons)}"
                )
    if "scipy" in backends and "ondevice" in backends:
        _progress(f"\n{'=' * 70}")
        _progress("DIAGNOSTIC TIMING STATUS")
        _progress(f"{'=' * 70}")
        _progress(
            "No performance or break-even verdict is emitted by this within-process "
            "diagnostic benchmark."
        )
