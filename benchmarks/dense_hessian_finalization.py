"""Benchmark dense Hessian finalization strategies.

This benchmark isolates final dense Hessian materialization after an optimizer
has already chosen a final state. It does not time the nonlinear solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(SRC_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.geo import optimizer_jax as _optimizer_jax

jax.config.update("jax_enable_x64", True)

BenchmarkFn = Callable[[jax.Array], jax.Array]
BenchmarkRow = dict[str, float | int | str | None]

CURRENT_HVP_METHOD = "current_hvp_vmap"
SKIP_METHOD = "skip_dense_finalization"
DEFAULT_SIZES = (
    16,  # mpol=2, ntor=1, fixed-G SurfaceRZFourier decision vector
    83,  # mpol=4, ntor=4, optimize-G SurfaceRZFourier decision vector
    223,  # mpol=8, ntor=6, optimize-G target-lane decision vector
)
TABLE_HEADERS = (
    "size",
    "method",
    "compile_s",
    "first_execute_s",
    "warm_median_s",
    "warm_min_s",
    "warm_max_s",
    "max_abs_diff_vs_current",
)


def _make_objective(size: int) -> BenchmarkFn:
    weights = jnp.linspace(0.25, 1.25, size, dtype=jnp.float64)
    low_rank = 0.01 * jnp.outer(weights, weights[::-1])
    matrix = jnp.diag(1.0 + weights) + 0.5 * (low_rank + low_rank.T)

    def objective(x: jax.Array) -> jax.Array:
        quadratic = 0.5 * x @ matrix @ x
        nonlinear = 1.0e-3 * jnp.sum(jnp.sin(x) ** 2)
        return quadratic + nonlinear

    return objective


def _method_fns(objective_fn: BenchmarkFn) -> tuple[tuple[str, BenchmarkFn], ...]:
    hvp_fn = _optimizer_jax._hessian_vector_product_fn(objective_fn)
    jacfwd_grad_fn = jax.jacfwd(jax.grad(objective_fn))
    hessian_fn = jax.hessian(objective_fn)

    return (
        (
            CURRENT_HVP_METHOD,
            lambda x: _optimizer_jax._materialize_dense_hessian(hvp_fn, x),
        ),
        ("jacfwd_grad_dense", jacfwd_grad_fn),
        ("jax_hessian_dense", hessian_fn),
        (SKIP_METHOD, lambda x: jnp.asarray(0.0, dtype=x.dtype)),
    )


def _time_blocking_call(fn: BenchmarkFn, x: jax.Array) -> tuple[float, jax.Array]:
    start = time.perf_counter()
    value = fn(x)
    jax.block_until_ready(value)
    return time.perf_counter() - start, value


def _measure_method(
    label: str,
    fn: BenchmarkFn,
    x: jax.Array,
    *,
    warmup: int,
    repeat: int,
) -> tuple[BenchmarkRow, jax.Array]:
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(x).compile()
    compile_s = time.perf_counter() - start

    first_execute_s, value = _time_blocking_call(compiled, x)
    for _ in range(warmup):
        _time_blocking_call(compiled, x)
    warm_times = [_time_blocking_call(compiled, x)[0] for _ in range(repeat)]

    return (
        {
            "method": label,
            "compile_s": compile_s,
            "first_execute_s": first_execute_s,
            "warm_min_s": min(warm_times),
            "warm_median_s": statistics.median(warm_times),
            "warm_mean_s": statistics.fmean(warm_times),
            "warm_max_s": max(warm_times),
            "max_abs_diff_vs_current": None,
        },
        value,
    )


def _max_abs_diff(left: jax.Array, right: jax.Array) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _run_size(size: int, *, warmup: int, repeat: int) -> list[BenchmarkRow]:
    objective_fn = _make_objective(size)
    x = jnp.linspace(-0.2, 0.2, size, dtype=jnp.float64)

    rows: list[BenchmarkRow] = []
    values: dict[str, jax.Array] = {}
    for label, fn in _method_fns(objective_fn):
        row, value = _measure_method(label, fn, x, warmup=warmup, repeat=repeat)
        row["size"] = size
        rows.append(row)
        values[label] = value

    reference = values[CURRENT_HVP_METHOD]
    for row in rows:
        label = str(row["method"])
        if label != SKIP_METHOD:
            row["max_abs_diff_vs_current"] = _max_abs_diff(values[label], reference)
    return rows


def _print_table(rows: list[BenchmarkRow]) -> None:
    print(f"jax_version={jax.__version__} backend={jax.default_backend()}")
    print(" ".join(f"{header:>24}" for header in TABLE_HEADERS))
    for row in rows:
        print(
            " ".join(
                f"{row[header]:>24.6g}"
                if isinstance(row[header], float)
                else f"{str(row[header]):>24}"
                for header in TABLE_HEADERS
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense Hessian finalization without timing the solve."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = [
        row
        for size in args.sizes
        for row in _run_size(size, warmup=args.warmup, repeat=args.repeat)
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
