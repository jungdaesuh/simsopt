"""Compile-shape diagnostic for the stepwise L-BFGS-B control kernels."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import platform
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

jax.config.update("jax_enable_x64", True)

from benchmarks.traceable_compile_shape import (
    lower_to_text,
    summarize_lowered_text,
)
from simsopt_jax.geo.optimizers.private import _lbfgs as private_lbfgs
from simsopt_jax.geo.optimizers.private import _lbfgsb_scipy as lbfgsb
from simsopt_jax.geo.optimizers.optimizer import _mark_cacheable_jit_value_and_grad


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

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Compiling jit(" in message and any(
            fragment in message for fragment in self.fragments
        ):
            self.count += 1


def _quadratic_value_and_grad(x):
    vector = jnp.asarray(x, dtype=jnp.float64)
    return 0.5 * jnp.dot(vector, vector), vector


def _maxrss_bytes() -> int:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(maxrss)
    return int(maxrss) * 1024


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
) -> list[dict[str, int | float | str | None]]:
    x0 = jnp.arange(1, dimension + 1, dtype=jnp.float64)

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

    def step_kernel(state):
        return lbfgsb.lbfgsb_advance_to_next_observable(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )

    def step_from_start_kernel(state):
        return lbfgsb.lbfgsb_advance_from_start_to_next_observable(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )

    state_search = lbfgsb._lbfgsb_evaluate_value_and_grad(
        _quadratic_value_and_grad,
        lbfgsb._lbfgsb_setulb_start(state0),
    )

    def step_from_search_kernel(state):
        return lbfgsb.lbfgsb_advance_from_search_to_next_observable(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )

    state_new_x = step_from_start_kernel(state0).state

    def reenter_from_new_x_kernel(state):
        return lbfgsb.lbfgsb_reenter_new_x(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )

    def result_kernel(state):
        return _result_payload(state, maxiter=maxiter, maxfun=maxfun)

    def monolithic_kernel(state):
        final_state = lbfgsb.lbfgsb_mainlb(
            _quadratic_value_and_grad,
            state,
            maxiter=maxiter,
            maxfun=maxfun,
            accepted_step_callback=None,
        )
        return _result_payload(final_state, maxiter=maxiter, maxfun=maxfun)

    return [
        _KernelCase("init_state", init_state, (x0,)),
        _KernelCase("old_generic_step_to_next_observable", step_kernel, (state0,)),
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
        _KernelCase("old_monolithic_full_solve", monolithic_kernel, (state0,)),
    ]


def _build_summaries(
    kernel_cases: list[_KernelCase],
) -> list[dict[str, int | float | str | None]]:
    return [
        _lower_summary(kernel_case.label, kernel_case.fn, *kernel_case.args)
        for kernel_case in kernel_cases
    ]


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
) -> list[dict[str, int | float | str]]:
    return [_compile_measurement(kernel_case) for kernel_case in kernel_cases]


def _repeated_call_compile_summary(
    *,
    maxiter: int,
    maxcor: int,
    maxls: int,
    ftol: float,
    gtol: float,
) -> dict[str, bool | float | int | str]:
    x0 = jnp.asarray(np.array([1.0, -2.0], dtype=np.float64))

    def value_and_grad(x):
        vector = jnp.asarray(x, dtype=jnp.float64)
        return 0.5 * jnp.dot(vector, vector), vector

    cacheable_value_and_grad = _mark_cacheable_jit_value_and_grad(value_and_grad)

    def run_once() -> None:
        result = private_lbfgs._minimize_lbfgs_private_value_and_grad(
            cacheable_value_and_grad,
            x0,
            maxiter=maxiter,
            maxcor=maxcor,
            maxls=maxls,
            ftol=ftol,
            gtol=gtol,
        )
        if bool(result.converged) is not True:
            raise AssertionError("lbfgs-ondevice repeated compile probe did not converge")

    logger = logging.getLogger("jax")
    old_level = logger.level
    handler = _CompileCounter(
        (
            "lbfgs_private_macro_step_solver)",
            "lbfgs_private_result_payload_solver)",
        )
    )
    rss_before_bytes = _maxrss_bytes()
    started_at = time.perf_counter()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        jax.clear_caches()
        with jax.log_compiles(True):
            for _ in range(3):
                run_once()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    elapsed_s = time.perf_counter() - started_at
    rss_after_bytes = _maxrss_bytes()
    expected_compile_count = 4
    return {
        "case": "private-lbfgs-repeated-call-compile-count",
        "run_count": 3,
        "compile_log_count": int(handler.count),
        "expected_compile_log_count": expected_compile_count,
        "compiled_executable_count": int(handler.count),
        "recompiled_on_repeated_calls": handler.count > expected_compile_count,
        "wall_s": elapsed_s,
        "peak_host_rss_bytes": rss_after_bytes,
        "peak_host_rss_delta_bytes": max(0, rss_after_bytes - rss_before_bytes),
    }


def _comparison(summaries: list[dict[str, int | float | str | None]]) -> dict[str, int]:
    by_label = {str(row["label"]): row for row in summaries}
    monolithic = by_label["old_monolithic_full_solve"]
    old_generic_step = by_label["old_generic_step_to_next_observable"]
    specialized_steps = [
        by_label["step_from_start_to_next_observable"],
        by_label["step_from_search_to_next_observable"],
        by_label["reenter_from_new_x"],
    ]
    largest_specialized_step = max(
        int(row["text_bytes"]) for row in specialized_steps
    )
    largest_specialized_step_jaxpr = max(
        int(row["jaxpr_text_bytes"]) for row in specialized_steps
    )
    result = by_label["result_payload"]
    return {
        "old_monolithic_text_bytes": int(monolithic["text_bytes"]),
        "old_generic_step_text_bytes": int(old_generic_step["text_bytes"]),
        "largest_specialized_step_text_bytes": largest_specialized_step,
        "result_payload_text_bytes": int(result["text_bytes"]),
        "largest_specialized_step_plus_result_text_bytes": largest_specialized_step
        + int(result["text_bytes"]),
        "old_monolithic_jaxpr_text_bytes": int(monolithic["jaxpr_text_bytes"]),
        "old_generic_step_jaxpr_text_bytes": int(
            old_generic_step["jaxpr_text_bytes"]
        ),
        "largest_specialized_step_jaxpr_text_bytes": largest_specialized_step_jaxpr,
        "result_payload_jaxpr_text_bytes": int(result["jaxpr_text_bytes"]),
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
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--maxcor", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--maxfun", type=int, default=20)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=0.0)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument(
        "--skip-runtime-compile",
        action="store_true",
        help="Only emit lowering/JAXPR shape summaries; skip executable compile/RSS probes.",
    )
    args = parser.parse_args()

    kernel_cases = _build_kernel_cases(
        dimension=args.dimension,
        maxcor=args.maxcor,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
        maxls=args.maxls,
        ftol=args.ftol,
        gtol=args.gtol,
    )
    summaries = _build_summaries(kernel_cases)
    runtime_compile = None
    repeated_call_compile = None
    if not args.skip_runtime_compile:
        runtime_compile_summaries = _compile_measurements(kernel_cases)
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
        repeated_call_compile = _repeated_call_compile_summary(
            maxiter=args.maxiter,
            maxcor=args.maxcor,
            maxls=args.maxls,
            ftol=args.ftol,
            gtol=args.gtol,
        )
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "platform": platform.platform(),
        "devices": _device_summary(),
        "case": {
            "objective": "deterministic_quadratic",
            "dimension": int(args.dimension),
            "maxcor": int(args.maxcor),
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxfun),
            "maxls": int(args.maxls),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "dtype": np.dtype(np.float64).str,
        },
        "summaries": summaries,
        "comparison": _comparison(summaries),
        "runtime_compile": runtime_compile,
        "repeated_call_compile": repeated_call_compile,
    }
    output_path = Path(args.output_json)
    _write_json(output_path, payload)
    print(json.dumps({"output_json": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
