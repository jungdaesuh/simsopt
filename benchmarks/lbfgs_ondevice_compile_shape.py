"""Compile-shape diagnostic for the stepwise L-BFGS-B control kernels."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

jax.config.update("jax_enable_x64", True)

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


from simsopt_jax.geo.optimizers._shared import (
    mark_cacheable_jit_value_and_grad as _mark_cacheable_jit_value_and_grad,
)
from simsopt_jax.geo.optimizers.private import _lbfgs as private_lbfgs
from simsopt_jax.geo.optimizers.private import _lbfgsb_scipy as lbfgsb

from benchmarks import custom_quasi_newton_runtime as runtime
from benchmarks.traceable_compile_shape import (
    lower_to_text,
    summarize_lowered_text,
)

_CompileProvider = Literal["custom", "optax"]


class _LoweredProgram(Protocol):
    def as_text(self) -> str: ...

    def compile(self, *args: object, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class _CapturedProviderCompile:
    executable: object
    stablehlo_module: str
    stablehlo_bytes: int
    compile_s: float


class _ProgressRecorder:
    """Persist atomic child checkpoints for watchdog-interrupted diagnostics."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._started_at = time.perf_counter()
        self._events: list[JsonObject] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write()

    def record(self, event: str, fields: JsonObject) -> None:
        if self._path is None:
            return
        event_payload: JsonObject = {
            "event": event,
            **fields,
            "elapsed_s": time.perf_counter() - self._started_at,
        }
        self._events.append(event_payload)
        self._write()

    def _write(self) -> None:
        if self._path is None:
            return
        payload: JsonObject = {
            "schema_version": 1,
            "events": cast(list[JsonValue], self._events),
        }
        temporary_path = self._path.with_name(f".{self._path.name}.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._path)


_LBFGS_STEPWISE_COMPILE_FRAGMENTS = (
    "lbfgs_private_value_and_grad)",
    "lbfgs_private_initial_state_solver)",
    "lbfgs_private_macro_step_solver)",
    "lbfgs_private_result_payload_solver)",
)
_LBFGS_MONOLITHIC_COMPILE_FRAGMENTS = ("lbfgs_private_monolithic_mainlb_solver)",)
_LBFGS_PRIVATE_COMPILE_FRAGMENTS = (
    *_LBFGS_STEPWISE_COMPILE_FRAGMENTS,
    *_LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
)
_KERNEL_LABELS = (
    "init_state",
    "step_from_start_to_next_observable",
    "step_from_search_to_next_observable",
    "reenter_from_new_x",
    "result_payload",
    "old_generic_step_to_next_observable",
    "old_monolithic_full_solve",
)


@dataclass(frozen=True)
class _KernelCase:
    label: str
    fn: Callable[..., object]
    args: tuple[object, ...]


class _CompileCounter(logging.Handler):
    def __init__(self, fragments: tuple[str, ...]) -> None:
        super().__init__()
        self.fragments = fragments
        self.count = 0
        self.counts_by_fragment = {fragment: 0 for fragment in fragments}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Compiling jit(" not in message:
            return
        for fragment in self.fragments:
            if fragment in message:
                self.count += 1
                self.counts_by_fragment[fragment] += 1
                return


def _sum_fragment_counts(
    counts_by_fragment: dict[str, int],
    fragments: tuple[str, ...],
) -> int:
    return sum(int(counts_by_fragment[fragment]) for fragment in fragments)


def _git_text(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _quadratic_value_and_grad(x):
    vector = jnp.asarray(x, dtype=jnp.float64)
    return 0.5 * jnp.dot(vector, vector), vector


def _objective_case(objective: str, dimension: int):
    """Build the objective and initial vector used by one diagnostic cell."""

    if objective == "quadratic":
        return (
            jnp.arange(1, dimension + 1, dtype=jnp.float64),
            _quadratic_value_and_grad,
        )
    if objective == "coil47":
        if dimension != 47:
            raise ValueError("objective='coil47' requires dimension=47")
        from benchmarks.fixtures.custom_quasi_newton import fixture

        fixture_case = fixture("coil47")
        return (
            jnp.asarray(fixture_case.initial, dtype=jnp.float64),
            jax.value_and_grad(fixture_case.objective),
        )
    raise ValueError(f"unknown compile-shape objective {objective!r}")


def _maxrss_bytes() -> int:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(maxrss)
    return int(maxrss) * 1024


def _sync(value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _summarize_bounded_result(result) -> JsonObject:
    """Classify a finite result without requiring the budget to converge."""
    objective = float(jax.device_get(result.f_k))
    gradient = np.asarray(jax.device_get(result.g_k), dtype=np.float64)
    finite = bool(np.isfinite(objective) and np.all(np.isfinite(gradient)))
    if not finite:
        raise AssertionError("bounded L-BFGS diagnostic produced a nonfinite result")
    return {
        "converged": bool(jax.device_get(result.converged)),
        "status": int(jax.device_get(result.status)),
        "iterations": int(jax.device_get(result.k)),
        "evaluations": int(jax.device_get(result.nfev)),
        "objective": objective,
        "finite": finite,
    }


def _objective_timing(
    value_and_grad,
    x0,
    *,
    progress: _ProgressRecorder | None = None,
    repeat_count: int = 3,
) -> JsonObject:
    """Measure one cold objective call and synchronized warm calls."""

    compiled = jax.jit(value_and_grad)
    if progress is not None:
        progress.record("objective_probe_start", {"repeat_count": repeat_count})
    timings: list[JsonValue] = []
    for run_index in range(repeat_count):
        started = time.perf_counter()
        value, gradient = compiled(x0)
        _sync((value, gradient))
        elapsed_s = time.perf_counter() - started
        timings.append(elapsed_s)
        if progress is not None:
            progress.record(
                "objective_probe_complete",
                {"run_index": run_index, "duration_s": elapsed_s},
            )
    return {
        "run_count": repeat_count,
        "cold_s": float(timings[0]),
        "warm_s": [float(value) for value in timings[1:]],
        "all_s": [float(value) for value in timings],
        "peak_host_rss_bytes": _maxrss_bytes(),
    }


def _device_summary() -> list[dict[str, int | str]]:
    return [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]


def _jaxpr_summary(fn, *args) -> dict[str, int]:
    closed_jaxpr = jax.make_jaxpr(fn)(*args)
    jaxpr_text = str(closed_jaxpr)
    return {
        "jaxpr_eqn_count": len(closed_jaxpr.jaxpr.eqns),
        "jaxpr_text_bytes": len(jaxpr_text.encode("utf-8")),
        "jaxpr_text_lines": len(jaxpr_text.splitlines()),
    }


def _lower_summary(label: str, fn, *args) -> dict[str, int | float | str | None]:
    measurement = lower_to_text(jax.jit(fn), *args)
    summary = summarize_lowered_text(
        label,
        measurement.lowered_text,
        lower_s=measurement.lower_s,
    )
    summary.update(_jaxpr_summary(fn, *args))
    return summary


def _result_payload(state, *, maxiter: int, maxfun: int):
    history = lbfgsb.lbfgsb_inverse_hessian_history(state)
    return private_lbfgs._lbfgsb_state_to_lbfgs_results(
        state,
        history=history,
        maxiter_limit=jnp.asarray(maxiter, dtype=jnp.int32),
        maxfun_limit=jnp.asarray(maxfun, dtype=jnp.int32),
    )


def _build_kernel_cases(
    *,
    dimension: int,
    maxcor: int,
    maxiter: int,
    maxfun: int,
    maxls: int,
    ftol: float,
    gtol: float,
    objective: str = "quadratic",
    include_legacy_kernels: bool = False,
    progress: _ProgressRecorder | None = None,
) -> list[_KernelCase]:
    x0, value_and_grad = _objective_case(objective, dimension)

    if progress is not None:
        progress.record(
            "objective_ready",
            {"objective": objective, "dimension": dimension},
        )

    def init_state(x):
        return lbfgsb.lbfgsb_initial_state(
            x,
            m=maxcor,
            bounds=None,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
        )

    state0 = init_state(x0)
    if progress is not None:
        progress.record("initial_state_ready", {"maxcor": maxcor})

    def step_kernel(state):
        return lbfgsb.lbfgsb_advance_to_next_observable(
            value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )

    def step_from_start_kernel(state):
        return lbfgsb.lbfgsb_advance_from_start_to_next_observable(
            value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
            unconstrained_fast_path=True,
        )

    state_search = lbfgsb._lbfgsb_evaluate_value_and_grad(
        value_and_grad,
        lbfgsb._lbfgsb_setulb_start(state0),
    )
    if progress is not None:
        progress.record("search_state_ready", {})

    def step_from_search_kernel(state):
        return lbfgsb.lbfgsb_advance_from_search_to_next_observable(
            value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
            unconstrained_fast_path=True,
        )

    # Re-entry lowering depends on the state pytree shapes, not on a prior
    # accepted iterate. Avoid executing a full line search while constructing
    # this diagnostic input; the real start-transition kernel is measured
    # separately below.
    state_new_x = state0
    if progress is not None:
        progress.record(
            "new_x_state_ready",
            {"source": "shape_compatible_initial_state"},
        )

    def reenter_from_new_x_kernel(state):
        return lbfgsb.lbfgsb_reenter_new_x(
            value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
            unconstrained_fast_path=True,
        )

    def result_kernel(state):
        return _result_payload(state, maxiter=maxiter, maxfun=maxfun)

    def monolithic_kernel(state):
        final_state = lbfgsb.lbfgsb_mainlb(
            value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )
        return _result_payload(final_state, maxiter=maxiter, maxfun=maxfun)

    cases = [
        _KernelCase("init_state", init_state, (x0,)),
        _KernelCase(
            "step_from_start_to_next_observable",
            step_from_start_kernel,
            (state0,),
        ),
        _KernelCase(
            "step_from_search_to_next_observable",
            step_from_search_kernel,
            (state_search,),
        ),
        _KernelCase(
            "reenter_from_new_x",
            reenter_from_new_x_kernel,
            (state_new_x,),
        ),
        _KernelCase("result_payload", result_kernel, (state0,)),
    ]
    if include_legacy_kernels:
        cases[1:1] = [
            _KernelCase(
                "old_generic_step_to_next_observable",
                step_kernel,
                (state0,),
            ),
        ]
        cases.append(
            _KernelCase("old_monolithic_full_solve", monolithic_kernel, (state0,))
        )
    return cases


def _build_summaries(
    kernel_cases: list[_KernelCase],
    *,
    progress: _ProgressRecorder | None = None,
) -> list[dict[str, int | float | str | None]]:
    summaries: list[dict[str, int | float | str | None]] = []
    for kernel_case in kernel_cases:
        if progress is not None:
            progress.record("lower_start", {"label": kernel_case.label})
        summary = _lower_summary(kernel_case.label, kernel_case.fn, *kernel_case.args)
        summaries.append(summary)
        if progress is not None:
            progress.record("lower_complete", cast(JsonObject, summary))
    return summaries


def _compile_measurement(
    kernel_case: _KernelCase,
) -> dict[str, int | float | str]:
    rss_before_bytes = _maxrss_bytes()
    started_at = time.perf_counter()
    compiled = jax.jit(kernel_case.fn).lower(*kernel_case.args).compile()
    compile_s = time.perf_counter() - started_at
    del compiled
    rss_after_bytes = _maxrss_bytes()
    return {
        "label": kernel_case.label,
        "compile_s": compile_s,
        "compiled_executable_count": 1,
        "peak_host_rss_bytes": rss_after_bytes,
        "peak_host_rss_delta_bytes": max(0, rss_after_bytes - rss_before_bytes),
    }


def _compile_measurements(
    kernel_cases: list[_KernelCase],
    *,
    progress: _ProgressRecorder | None = None,
) -> list[dict[str, int | float | str]]:
    measurements: list[dict[str, int | float | str]] = []
    for kernel_case in kernel_cases:
        if progress is not None:
            progress.record("compile_start", {"label": kernel_case.label})
        measurement = _compile_measurement(kernel_case)
        measurements.append(measurement)
        if progress is not None:
            progress.record("compile_complete", cast(JsonObject, measurement))
    return measurements


def _repeated_call_compile_summary(
    *,
    objective: str,
    dimension: int,
    maxiter: int,
    maxcor: int,
    maxls: int,
    ftol: float,
    gtol: float,
    progress: _ProgressRecorder | None = None,
):
    x0, value_and_grad = _objective_case(objective, dimension)

    cacheable_value_and_grad = _mark_cacheable_jit_value_and_grad(value_and_grad)

    def run_once() -> JsonObject:
        iteration_progress: list[JsonObject] = []
        run_started = time.perf_counter()
        previous_callback_time = run_started

        def record_iteration(iteration, objective_value, gradient_inf_norm) -> None:
            nonlocal previous_callback_time
            callback_time = time.perf_counter()
            iteration_progress.append(
                {
                    "iteration": int(iteration),
                    "objective": float(objective_value),
                    "gradient_inf_norm": float(gradient_inf_norm),
                    "solver_elapsed_s": callback_time - run_started,
                    "step_s": callback_time - previous_callback_time,
                }
            )
            previous_callback_time = callback_time
            if progress is not None:
                progress.record("solver_iteration", iteration_progress[-1])

        result = private_lbfgs._minimize_lbfgs_private_value_and_grad(
            cacheable_value_and_grad,
            x0,
            maxiter=maxiter,
            maxcor=maxcor,
            maxls=maxls,
            ftol=ftol,
            gtol=gtol,
            progress_callback=record_iteration,
        )
        summary = _summarize_bounded_result(result)
        summary["run_seconds"] = time.perf_counter() - run_started
        summary["iteration_progress"] = iteration_progress
        return summary

    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter(_LBFGS_PRIVATE_COMPILE_FRAGMENTS)
    rss_before_bytes = _maxrss_bytes()
    started_at = time.perf_counter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    run_summaries: list[JsonValue] = []
    try:
        jax.clear_caches()
        with jax.log_compiles(True):
            for run_index in range(3):
                if progress is not None:
                    progress.record(
                        "solver_run_start",
                        {"run_index": run_index},
                    )
                result_summary = run_once()
                run_summaries.append(cast(JsonObject, result_summary))
                if progress is not None:
                    progress.record(
                        "solver_run_complete",
                        {"run_index": run_index, **result_summary},
                    )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    elapsed_s = time.perf_counter() - started_at
    rss_after_bytes = _maxrss_bytes()
    expected_compile_count = 5
    stepwise_compile_count = _sum_fragment_counts(
        handler.counts_by_fragment,
        _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
    )
    monolithic_compile_count = _sum_fragment_counts(
        handler.counts_by_fragment,
        _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
    )
    return {
        "case": "private-lbfgs-repeated-call-compile-count",
        "objective": objective,
        "dimension": int(dimension),
        "run_count": 3,
        "compile_log_count": int(handler.count),
        "counts_by_fragment": dict(handler.counts_by_fragment),
        "stepwise_compile_log_count": int(stepwise_compile_count),
        "monolithic_compile_log_count": int(monolithic_compile_count),
        "expected_compile_log_count": expected_compile_count,
        "compiled_executable_count": int(handler.count),
        "recompiled_on_repeated_calls": handler.count > expected_compile_count,
        "optimizer_control_monolithic_full_run_compile": monolithic_compile_count > 0,
        "wall_s": elapsed_s,
        "peak_host_rss_bytes": rss_after_bytes,
        "peak_host_rss_delta_bytes": max(0, rss_after_bytes - rss_before_bytes),
        "result_summary": result_summary,
        "run_summaries": run_summaries,
    }


def _boozer_limited_memory_compile_summary(
    *,
    maxiter: int,
    maxcor: int,
    maxfun: int,
    maxls: int,
    ftol: float,
    gtol: float,
    progress: _ProgressRecorder | None = None,
) -> dict[str, bool | float | int | str | dict[str, int]]:
    from repo_bootstrap import bootstrap_local_simsopt

    bootstrap_local_simsopt(SRC_ROOT)

    from simsopt.geo.surfaceobjectives import Volume
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

    from benchmarks.benchmark_problem import (
        build_ls_parity_problem,
        clone_tensor_surface,
    )

    problem = build_ls_parity_problem(
        ncoils=2,
        nfp=1,
        mpol=1,
        ntor=1,
        nphi=3,
        ntheta=3,
    )
    surface = clone_tensor_surface(problem.surface)
    volume = Volume(surface)
    biot_savart = BiotSavartJAX(problem.coils)
    booz = BoozerSurfaceJAX(
        biot_savart,
        surface,
        volume,
        problem.vol_target,
        constraint_weight=1.0,
        options={
            "optimizer_backend": "ondevice",
            "limited_memory": True,
            "verbose": False,
            "bfgs_maxiter": maxiter,
            "bfgs_tol": gtol,
            "maxcor": maxcor,
            "maxfun": maxfun,
            "maxls": maxls,
            "ftol": ftol,
            "newton_maxiter": 0,
        },
    )
    decision_vector_size = int(
        np.asarray(booz._pack_decision_vector(problem.iota0, problem.G0)).size
    )

    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter(_LBFGS_PRIVATE_COMPILE_FRAGMENTS)
    rss_before_bytes = _maxrss_bytes()
    started_at = time.perf_counter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        jax.clear_caches()
        if progress is not None:
            progress.record(
                "boozer_compile_start",
                {
                    "maxiter": maxiter,
                    "maxcor": maxcor,
                    "maxfun": maxfun,
                },
            )
        with jax.log_compiles(True):
            result = booz.minimize_boozer_penalty_constraints_LBFGS(
                constraint_weight=1.0,
                iota=problem.iota0,
                G=problem.G0,
                tol=gtol,
                maxiter=maxiter,
                verbose=False,
                limited_memory=True,
                weight_inv_modB=True,
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    elapsed_s = time.perf_counter() - started_at
    rss_after_bytes = _maxrss_bytes()
    stepwise_compile_count = _sum_fragment_counts(
        handler.counts_by_fragment,
        _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
    )
    monolithic_compile_count = _sum_fragment_counts(
        handler.counts_by_fragment,
        _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
    )
    summary = {
        "case": "boozer-limited-memory-lbfgs-compile-log",
        "run_count": 1,
        "method": "lbfgs-ondevice",
        "optimizer_backend": "ondevice",
        "limited_memory": True,
        "decision_vector_size": decision_vector_size,
        "maxiter": int(maxiter),
        "maxcor": int(maxcor),
        "maxfun": int(maxfun),
        "maxls": int(maxls),
        "compile_log_count": int(handler.count),
        "counts_by_fragment": dict(handler.counts_by_fragment),
        "stepwise_compile_log_count": int(stepwise_compile_count),
        "expected_stepwise_compile_log_count": 5,
        "monolithic_compile_log_count": int(monolithic_compile_count),
        "optimizer_control_monolithic_full_run_compile": monolithic_compile_count > 0,
        "success": bool(result["success"]),
        "iterations": int(result["iter"]),
        "wall_s": elapsed_s,
        "peak_host_rss_bytes": rss_after_bytes,
        "peak_host_rss_delta_bytes": max(0, rss_after_bytes - rss_before_bytes),
    }
    if progress is not None:
        progress.record("boozer_compile_complete", cast(JsonObject, summary))
    return summary


def _comparison(summaries: list[dict[str, int | float | str | None]]) -> dict[str, int]:
    by_label = {str(row["label"]): row for row in summaries}
    monolithic = by_label["old_monolithic_full_solve"]
    old_generic_step = by_label["old_generic_step_to_next_observable"]
    specialized_steps = [
        by_label["step_from_start_to_next_observable"],
        by_label["step_from_search_to_next_observable"],
        by_label["reenter_from_new_x"],
    ]
    largest_specialized_step = max(int(row["text_bytes"]) for row in specialized_steps)
    largest_specialized_step_jaxpr = max(
        int(row["jaxpr_text_bytes"]) for row in specialized_steps
    )
    result = by_label["result_payload"]
    return {
        "old_monolithic_text_bytes": int(monolithic["text_bytes"]),
        "old_generic_step_text_bytes": int(old_generic_step["text_bytes"]),
        "largest_specialized_step_text_bytes": largest_specialized_step,
        "specialized_step_text_bytes_reduced_vs_generic": int(
            largest_specialized_step < int(old_generic_step["text_bytes"])
        ),
        "result_payload_text_bytes": int(result["text_bytes"]),
        "largest_specialized_step_plus_result_text_bytes": largest_specialized_step
        + int(result["text_bytes"]),
        "old_monolithic_jaxpr_text_bytes": int(monolithic["jaxpr_text_bytes"]),
        "old_generic_step_jaxpr_text_bytes": int(old_generic_step["jaxpr_text_bytes"]),
        "largest_specialized_step_jaxpr_text_bytes": largest_specialized_step_jaxpr,
        "specialized_step_jaxpr_text_bytes_reduced_vs_generic": int(
            largest_specialized_step_jaxpr < int(old_generic_step["jaxpr_text_bytes"])
        ),
        "result_payload_jaxpr_text_bytes": int(result["jaxpr_text_bytes"]),
    }


def _stablehlo_module_name(stablehlo_text: str) -> str:
    first_line = stablehlo_text.splitlines()[0]
    prefix = "module @"
    if not first_line.startswith(prefix):
        raise RuntimeError("provider lowering did not emit a StableHLO module")
    return first_line[len(prefix) :].split(maxsplit=1)[0]


def _provider_programs(
    provider: _CompileProvider,
    prepared: object,
) -> tuple[tuple[str, object], ...]:
    if provider == "custom":
        custom = cast(runtime._PreparedCustom, prepared)
        program = custom.program
        if program.run_mode == "fused_stepwise":
            return (
                ("initial_state", program.initial_state),
                ("value_and_grad", program.value_and_grad),
                ("fused_solve", program.fused_solve),
            )
        if program.run_mode == "stepwise":
            return (
                ("initial_state", program.initial_state),
                ("value_and_grad", program.value_and_grad),
                ("advance_from_start", program.advance_from_start),
                ("advance_from_search", program.advance_from_search),
                ("reenter_new_x", program.reenter_new_x),
                ("result_payload", program.result_payload),
            )
        raise RuntimeError(f"unsupported custom provider run mode {program.run_mode!r}")
    if provider == "optax":
        optax_prepared = cast(runtime._PreparedOptax, prepared)
        return (
            ("step", optax_prepared.step),
            ("final_value_and_grad", optax_prepared.final_value_and_grad),
        )
    raise ValueError(f"unknown compile provider {provider!r}")


def _summarize_provider_compiles(
    provider: _CompileProvider,
    prepared: object,
    captured: tuple[_CapturedProviderCompile, ...],
) -> tuple[list[JsonObject], JsonObject]:
    programs = _provider_programs(provider, prepared)
    labels_by_executable = {id(executable): label for label, executable in programs}
    if len(labels_by_executable) != len(programs):
        raise RuntimeError(
            "provider preparation reused one executable for two programs"
        )
    if len(captured) != len(programs):
        raise RuntimeError(
            f"provider {provider!r} prepared {len(programs)} programs but "
            f"performed {len(captured)} executable compiles"
        )

    summaries: list[JsonObject] = []
    observed_labels: set[str] = set()
    for record in captured:
        label = labels_by_executable.get(id(record.executable))
        if label is None:
            raise RuntimeError("provider preparation compiled an unreturned executable")
        observed_labels.add(label)
        summaries.append(
            {
                "label": label,
                "stablehlo_module": record.stablehlo_module,
                "stablehlo_bytes": record.stablehlo_bytes,
                "compile_s": record.compile_s,
                "compiled_executable_count": 1,
                "compiled_executable_count_source": "observed_lowered_compile_call",
            }
        )
    expected_labels = {label for label, _executable in programs}
    if observed_labels != expected_labels:
        raise RuntimeError("provider compile observations did not cover every program")

    aggregate: JsonObject = {
        "stablehlo_bytes": sum(
            cast(int, summary["stablehlo_bytes"]) for summary in summaries
        ),
        "compile_s": sum(cast(float, summary["compile_s"]) for summary in summaries),
        "compiled_executable_count": len(summaries),
        "compiled_executable_count_source": "observed_lowered_compile_calls",
    }
    return summaries, aggregate


def _capture_provider_preparation(
    provider: _CompileProvider,
    fixture_case: runtime.Fixture,
    x0: np.ndarray,
    *,
    maxcor: int,
    intent: str,
) -> tuple[object, tuple[_CapturedProviderCompile, ...], float]:
    if provider not in {"custom", "optax"}:
        raise ValueError(f"unknown compile provider {provider!r}")

    sample_lowered = jax.jit(lambda value: value).lower(
        jnp.asarray(0.0, dtype=jnp.float64)
    )
    lowered_type = type(sample_lowered)
    original_compile = lowered_type.compile
    captured: list[_CapturedProviderCompile] = []

    def record_compile(
        lowered: _LoweredProgram,
        *args: object,
        **kwargs: object,
    ) -> object:
        stablehlo_text = lowered.as_text()
        compile_started = time.perf_counter()
        executable = original_compile(lowered, *args, **kwargs)
        compile_s = time.perf_counter() - compile_started
        captured.append(
            _CapturedProviderCompile(
                executable=executable,
                stablehlo_module=_stablehlo_module_name(stablehlo_text),
                stablehlo_bytes=len(stablehlo_text.encode("utf-8")),
                compile_s=compile_s,
            )
        )
        return executable

    preparation_started = time.perf_counter()
    with patch.object(lowered_type, "compile", record_compile):
        if provider == "custom":
            prepared = runtime._prepare_custom(
                fixture_case,
                x0,
                maxcor=maxcor,
                run_mode=runtime._solver_route("custom", "lbfgs", intent=intent),
            )
        elif provider == "optax":
            prepared = runtime._prepare_optax(fixture_case, x0, maxcor=maxcor)
    preparation_s = time.perf_counter() - preparation_started
    return prepared, tuple(captured), preparation_s


def _provider_compile_payload(
    *,
    provider: str,
    fixture_name: str,
    device: str,
    intent: str,
    maxiter: int,
    maxcor: int,
) -> dict[str, object]:
    if provider not in {"custom", "optax"}:
        raise ValueError(f"unknown compile provider {provider!r}")
    typed_provider = cast(_CompileProvider, provider)
    runtime._validate_intent_environment(device, intent)
    if runtime.fixture_method(fixture_name) != "lbfgs":
        raise ValueError("provider compile diagnostics support only L-BFGS fixtures")

    fixture_started = time.perf_counter()
    fixture_case = runtime.fixture(fixture_name)
    x0 = np.asarray(fixture_case.initial, dtype=np.float64)
    fixture_build_s = time.perf_counter() - fixture_started
    initial_objective, initial_gradient = runtime._initial_value_and_grad(
        fixture_case,
        x0,
        provider=typed_provider,
    )
    fixture_contract = runtime._fixture_contract_payload(
        fixture_case,
        x0,
        initial_objective=initial_objective,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        method="lbfgs",
        maxiter=maxiter,
        maxcor=maxcor,
        device=device,
        intent=intent,
    )
    prepared, captured, provider_preparation_s = _capture_provider_preparation(
        typed_provider,
        fixture_case,
        x0,
        maxcor=maxcor,
        intent=intent,
    )
    programs, aggregate = _summarize_provider_compiles(
        typed_provider,
        prepared,
        captured,
    )

    git_commit, git_clean = runtime._checkout_provenance()
    candidate_sha = git_commit if git_clean else None
    device_identity = asdict(runtime._device_identity(device))
    if device == "cpu":
        for field in (
            "gpu_uuid",
            "gpu_model",
            "compute_capability",
            "total_memory_bytes",
            "driver_version",
            "cuda_version",
            "visible_devices",
        ):
            device_identity[field] = None
    gpu_identity_available = device_identity["gpu_uuid"] is not None
    provider_factory_options: JsonObject = (
        {
            "maxcor": maxcor,
            "ftol": runtime._SOLVER_FTOL,
            "gtol": runtime._SOLVER_GTOL,
            "maxls": runtime._SOLVER_MAXLS,
            "x_dtype": "float64",
        }
        if typed_provider == "custom"
        else {"memory_size": maxcor}
    )
    return {
        "schema_version": 4,
        "artifact_kind": "provider_factory_compile_shape",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "provider": typed_provider,
        "solver_route": runtime._solver_route(typed_provider, "lbfgs", intent=intent),
        "candidate_sha": candidate_sha,
        "candidate_sha_availability": (
            "available" if candidate_sha is not None else "unavailable"
        ),
        "candidate_sha_unavailable_reason": (
            None if candidate_sha is not None else "dirty_worktree"
        ),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "git_status_short": _git_text("status", "--short"),
        "jax_version": jax.__version__,
        "optax_version": runtime.optax.__version__,
        "jax_backend": jax.default_backend(),
        "platform": platform.platform(),
        "device_identity": device_identity,
        "gpu_identity_availability": (
            "available" if gpu_identity_available else "unavailable"
        ),
        "gpu_identity_unavailable_reason": (
            None if gpu_identity_available else "cpu_device"
        ),
        "runtime_environment": runtime._runtime_environment_payload(),
        "fixture": fixture_case.name,
        "fixture_build_s": fixture_build_s,
        "fixture_contract": fixture_contract,
        "dtype": str(x0.dtype),
        "parameter_shape": [int(size) for size in x0.shape],
        "options": fixture_contract["solver_options"],
        "provider_factory_options": provider_factory_options,
        "provider_preparation_s": provider_preparation_s,
        "programs": programs,
        "aggregate": aggregate,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write L-BFGS-B on-device compile-shape diagnostics."
    )
    parser.add_argument(
        "--output-json",
        default=".artifacts/lbfgs_ondevice_compile_shape_20260618.json",
        help="Path for the diagnostic JSON payload.",
    )
    parser.add_argument(
        "--progress-json",
        default=None,
        help="Optional atomic sidecar for checkpoints before watchdog termination.",
    )
    parser.add_argument(
        "--provider",
        choices=("custom-diagnostics", "custom", "optax"),
        default="custom-diagnostics",
        help=(
            "Compile the exact runtime provider factory, or retain the existing "
            "custom kernel diagnostics."
        ),
    )
    parser.add_argument(
        "--fixture",
        choices=("coil47", "rosenbrock"),
        default="coil47",
        help="Runtime-runner fixture for custom or Optax provider compilation.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--intent", choices=("fast", "parity"), default="parity")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument(
        "--objective",
        choices=("quadratic", "coil47"),
        default="quadratic",
        help="Objective graph used by the compile-shape probe.",
    )
    parser.add_argument("--maxcor", type=int)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--maxfun", type=int, default=20)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=0.0)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument(
        "--include-legacy-kernels",
        action="store_true",
        help=(
            "Include the old generic and monolithic kernels. They can require "
            "multiple GiB of host memory; run only in an externally watched "
            "process."
        ),
    )
    parser.add_argument(
        "--kernel",
        choices=("all", *_KERNEL_LABELS),
        default="all",
        help="Lower/compile only one kernel; useful for bounded phase isolation.",
    )
    parser.add_argument(
        "--skip-runtime-compile",
        action="store_true",
        help="Only emit lowering/JAXPR shape summaries; skip executable compile/RSS probes.",
    )
    parser.add_argument(
        "--skip-lowering-summary",
        action="store_true",
        help=(
            "Skip StableHLO/JAXPR text generation when a runtime/solver timing "
            "cell is being isolated."
        ),
    )
    parser.add_argument(
        "--measure-objective",
        action="store_true",
        help="Measure one cold and two warm synchronized objective calls.",
    )
    parser.add_argument(
        "--run-solver",
        action="store_true",
        help=(
            "Run three bounded solver calls and record accepted-step progress. "
            "This is separate from per-kernel lowering."
        ),
    )
    parser.add_argument(
        "--include-boozer-compile-log",
        action="store_true",
        help=(
            "Run a small Boozer limited-memory L-BFGS-B case under JAX compile "
            "logging to classify stepwise vs monolithic optimizer-control compiles."
        ),
    )
    parser.add_argument(
        "--boozer-maxiter",
        type=int,
        default=10,
        help="Iteration limit for the optional Boozer compile-log diagnostic.",
    )
    parser.add_argument(
        "--boozer-maxcor",
        type=int,
        default=10,
        help="History size for the optional Boozer compile-log diagnostic.",
    )
    args = parser.parse_args()

    provider_factory_mode = args.provider != "custom-diagnostics"
    maxcor = (
        args.maxcor if args.maxcor is not None else (10 if provider_factory_mode else 3)
    )
    maxiter = (
        args.maxiter
        if args.maxiter is not None
        else (20 if provider_factory_mode else 5)
    )

    progress = _ProgressRecorder(
        None if args.progress_json is None else Path(args.progress_json)
    )
    if provider_factory_mode:
        progress.record(
            "provider_compile_started",
            {
                "provider": args.provider,
                "fixture": args.fixture,
                "device": args.device,
                "intent": args.intent,
                "maxcor": maxcor,
                "maxiter": maxiter,
            },
        )
        provider_payload = _provider_compile_payload(
            provider=args.provider,
            fixture_name=args.fixture,
            device=args.device,
            intent=args.intent,
            maxiter=maxiter,
            maxcor=maxcor,
        )
        output_path = Path(args.output_json)
        _write_json(output_path, provider_payload)
        progress.record("payload_written", {"output_json": str(output_path)})
        print(json.dumps({"output_json": str(output_path)}, sort_keys=True))
        return

    progress.record(
        "process_started",
        {
            "objective": args.objective,
            "dimension": args.dimension,
            "maxcor": maxcor,
            "maxiter": maxiter,
            "maxfun": args.maxfun,
            "include_legacy_kernels": bool(args.include_legacy_kernels),
            "kernel": args.kernel,
        },
    )
    kernel_cases = _build_kernel_cases(
        dimension=args.dimension,
        maxcor=maxcor,
        maxiter=maxiter,
        maxfun=args.maxfun,
        maxls=args.maxls,
        ftol=args.ftol,
        gtol=args.gtol,
        objective=args.objective,
        include_legacy_kernels=args.include_legacy_kernels,
        progress=progress,
    )
    if args.kernel != "all":
        if not args.include_legacy_kernels and args.kernel.startswith("old_"):
            raise ValueError(
                "legacy kernel selection requires --include-legacy-kernels"
            )
        kernel_cases = [
            kernel_case
            for kernel_case in kernel_cases
            if kernel_case.label == args.kernel
        ]
        if not kernel_cases:
            raise ValueError(f"kernel {args.kernel!r} is not available")
    progress.record("kernel_cases_built", {"count": len(kernel_cases)})
    summaries = (
        []
        if args.skip_lowering_summary
        else _build_summaries(kernel_cases, progress=progress)
    )
    if args.skip_lowering_summary:
        progress.record("lowering_skipped", {})
    objective_timing = None
    if args.measure_objective:
        objective_x0, objective_value_and_grad = _objective_case(
            args.objective,
            args.dimension,
        )
        objective_timing = _objective_timing(
            objective_value_and_grad,
            objective_x0,
            progress=progress,
        )
    runtime_compile = None
    repeated_call_compile = None
    boozer_limited_memory_compile = None
    run_solver = not args.skip_runtime_compile and (
        args.run_solver or args.kernel == "all"
    )
    if not args.skip_runtime_compile:
        runtime_compile_summaries = _compile_measurements(
            kernel_cases,
            progress=progress,
        )
        runtime_compile = {
            "summaries": runtime_compile_summaries,
            "compiled_executable_count": sum(
                int(row["compiled_executable_count"])
                for row in runtime_compile_summaries
            ),
            "peak_host_rss_bytes": max(
                int(row["peak_host_rss_bytes"]) for row in runtime_compile_summaries
            ),
        }
        if run_solver:
            repeated_call_compile = _repeated_call_compile_summary(
                objective=args.objective,
                dimension=args.dimension,
                maxiter=maxiter,
                maxcor=maxcor,
                maxls=args.maxls,
                ftol=args.ftol,
                gtol=args.gtol,
                progress=progress,
            )
        if args.include_boozer_compile_log and args.kernel == "all":
            boozer_limited_memory_compile = _boozer_limited_memory_compile_summary(
                maxiter=args.boozer_maxiter,
                maxcor=args.boozer_maxcor,
                maxfun=args.maxfun,
                maxls=args.maxls,
                ftol=args.ftol,
                gtol=args.gtol,
                progress=progress,
            )
    payload = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_git_head": _git_text("rev-parse", "HEAD"),
        "source_git_status_short": _git_text("status", "--short"),
        "command": [sys.executable, *sys.argv],
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "platform": platform.platform(),
        "devices": _device_summary(),
        "case": {
            "objective": (
                "deterministic_quadratic"
                if args.objective == "quadratic"
                else "source_owned_coil47"
            ),
            "objective_kind": args.objective,
            "dimension": int(args.dimension),
            "maxcor": int(maxcor),
            "maxiter": int(maxiter),
            "maxfun": int(args.maxfun),
            "maxls": int(args.maxls),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "dtype": np.dtype(np.float64).str,
        },
        "summaries": summaries,
        "comparison": (
            _comparison(summaries)
            if args.include_legacy_kernels
            else {
                "status": "legacy_kernels_not_requested",
                "legacy_kernels_included": 0,
            }
        ),
        "include_legacy_kernels": bool(args.include_legacy_kernels),
        "kernel_selection": args.kernel,
        "objective_timing": objective_timing,
        "run_solver": run_solver,
        "runtime_compile": runtime_compile,
        "repeated_call_compile": repeated_call_compile,
        "boozer_limited_memory_compile": boozer_limited_memory_compile,
    }
    output_path = Path(args.output_json)
    _write_json(output_path, payload)
    progress.record("payload_written", {"output_json": str(output_path)})
    print(json.dumps({"output_json": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
