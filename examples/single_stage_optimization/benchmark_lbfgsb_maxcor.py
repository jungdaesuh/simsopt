#!/usr/bin/env python3
"""Benchmark L-BFGS-B maxcor on a fixed banana-sized quadratic fixture."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from banana_opt.lbfgsb_defaults import (
    DEFAULT_LBFGSB_MAXCOR,
    LBFGSB_MAXCOR_BENCHMARK_VALUES,
)

SCHEMA_VERSION = "banana_lbfgsb_maxcor_benchmark_v1"


def _maxrss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _fixture_parameters(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(dimension, dtype=float)
    scale = np.geomspace(1.0, 1.0e4, dimension)
    target = 0.15 * np.sin(index * 0.37)
    return scale, target


def _fixture_objective(scale: np.ndarray, target: np.ndarray):
    coupling = 0.05 * float(np.mean(scale))

    def fun(x):
        shifted = x - target
        differences = x[1:] - x[:-1]
        value = float(np.dot(scale, shifted * shifted))
        value += coupling * float(np.dot(differences, differences))

        gradient = 2.0 * scale * shifted
        gradient[:-1] -= 2.0 * coupling * differences
        gradient[1:] += 2.0 * coupling * differences
        return value, gradient

    return fun


def _lbfgsb_options(*, maxiter: int, maxcor: int) -> dict[str, float | int]:
    return {"maxiter": maxiter, "maxcor": maxcor, "ftol": 1.0e-15, "gtol": 1.0e-15}


def measure_maxcor(
    *,
    maxcor: int,
    dimension: int,
    maxiter: int,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        __file__,
        "--measure-one",
        "--maxcor",
        str(maxcor),
        "--dimension",
        str(dimension),
        "--maxiter",
        str(maxiter),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _measure_maxcor_in_process(
    *,
    maxcor: int,
    dimension: int,
    maxiter: int,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    scale, target = _fixture_parameters(dimension)
    objective = _fixture_objective(scale, target)
    x0 = np.full(dimension, 0.35, dtype=float)
    bounds = [(-1.0, 1.0)] * dimension

    for _ in range(warmup):
        minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=_lbfgsb_options(maxiter=maxiter, maxcor=maxcor),
        )

    gc.collect()
    tracemalloc.start()
    times = []
    final_objectives = []
    projected_grad_norms = []
    iterations = []
    evaluations = []
    successes = []

    for _ in range(repeat):
        start = time.perf_counter()
        result = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=_lbfgsb_options(maxiter=maxiter, maxcor=maxcor),
        )
        times.append(time.perf_counter() - start)
        final_objectives.append(float(result.fun))
        projected_grad_norms.append(float(np.linalg.norm(result.jac, ord=np.inf)))
        iterations.append(int(result.nit))
        evaluations.append(int(result.nfev))
        successes.append(bool(result.success))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "maxcor": maxcor,
        "dimension": dimension,
        "maxiter": maxiter,
        "repeat": repeat,
        "warmup": warmup,
        "seconds_min": min(times),
        "seconds_median": statistics.median(times),
        "seconds_mean": statistics.mean(times),
        "python_peak_bytes": int(peak_bytes),
        "process_peak_rss_bytes": int(_maxrss_bytes()),
        "iterations_median": statistics.median(iterations),
        "function_evaluations_median": statistics.median(evaluations),
        "final_objective_median": statistics.median(final_objectives),
        "gradient_inf_norm_median": statistics.median(projected_grad_norms),
        "success_count": sum(successes),
    }


def build_report(
    maxcor_values: Iterable[int],
    *,
    dimension: int,
    maxiter: int,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    values = tuple(maxcor_values)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "default_maxcor": DEFAULT_LBFGSB_MAXCOR,
        "benchmark_maxcor_values": list(LBFGSB_MAXCOR_BENCHMARK_VALUES),
        "dimension": dimension,
        "maxiter": maxiter,
        "repeat": repeat,
        "warmup": warmup,
        "results": [
            measure_maxcor(
                maxcor=maxcor,
                dimension=dimension,
                maxiter=maxiter,
                repeat=repeat,
                warmup=warmup,
            )
            for maxcor in values
        ],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark L-BFGS-B maxcor on a fixed banana-sized quadratic fixture."
    )
    parser.add_argument(
        "--maxcor",
        action="append",
        type=_positive_int,
        help="maxcor value to benchmark. Repeat to compare values. Defaults to the P3 set.",
    )
    parser.add_argument("--measure-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dimension", type=_positive_int, default=160)
    parser.add_argument("--maxiter", type=_positive_int, default=30)
    parser.add_argument("--repeat", type=_positive_int, default=3)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.measure_one:
        if args.maxcor is None or len(args.maxcor) != 1:
            raise ValueError("--measure-one requires exactly one --maxcor")
        print(
            json.dumps(
                _measure_maxcor_in_process(
                    maxcor=args.maxcor[0],
                    dimension=args.dimension,
                    maxiter=args.maxiter,
                    repeat=args.repeat,
                    warmup=args.warmup,
                ),
                sort_keys=True,
            )
        )
        return 0

    maxcor_values = args.maxcor if args.maxcor is not None else LBFGSB_MAXCOR_BENCHMARK_VALUES
    report = build_report(
        maxcor_values,
        dimension=args.dimension,
        maxiter=args.maxiter,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
