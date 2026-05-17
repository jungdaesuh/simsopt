"""Benchmark RegularGridInterpolant3D skip-cell lookup strategies.

The JAX port represents skipped cells with a dense ``cell_to_row`` table whose
skipped entries redirect to the zero sentinel row in ``cell_table``. The C++
oracle keeps sparse skipped-cell state behind the pybind wrapper. This harness
builds both interpolants with the same skip predicate, checks numerical parity,
and reports hot-evaluation timings plus the explicit JAX lookup-table bytes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import simsoptpp as sopp

from simsopt.jax_core.regular_grid_interp import (
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
    build_regular_grid_interpolant_3d_device_spec,
    evaluate_batch_device,
)

_RANGE_MIN = 1.0
_RANGE_MAX = 4.0
_KEEP_MIN = 2.0
_KEEP_MAX = 3.0


def _box_skip(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    zs_arr = np.asarray(zs)
    keep = (
        (_KEEP_MIN < xs_arr)
        & (xs_arr < _KEEP_MAX)
        & (_KEEP_MIN < ys_arr)
        & (ys_arr < _KEEP_MAX)
        & (_KEEP_MIN < zs_arr)
        & (zs_arr < _KEEP_MAX)
    )
    return np.invert(keep)


def _polynomial(value_size: int) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def _evaluate(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        xs_arr = np.asarray(xs, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.float64)
        zs_arr = np.asarray(zs, dtype=np.float64)
        columns = [
            (comp + 1.0) * xs_arr
            - (comp + 2.0) * ys_arr * ys_arr
            + (comp + 3.0) * xs_arr * zs_arr
            + 0.25 * comp
            for comp in range(value_size)
        ]
        return np.stack(columns, axis=1).reshape(-1)

    return _evaluate


def _sample_points(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.uniform(_RANGE_MIN, _RANGE_MAX, size=(n_samples, 3)),
        dtype=np.float64,
    )


def _median_timing(repeats: int, fn: Callable[[], None]) -> float:
    timings = np.zeros((repeats,), dtype=np.float64)
    for idx in range(repeats):
        start = time.perf_counter()
        fn()
        timings[idx] = time.perf_counter() - start
    return float(np.median(timings))


def run_skip_cell_map_benchmark(
    *,
    n_cells: int = 30,
    degree: int = 2,
    value_size: int = 3,
    n_samples: int = 4096,
    repeats: int = 5,
    seed: int = 507,
) -> dict[str, object]:
    """Return sparse-vs-dense skip-cell lookup benchmark metrics."""

    axis_range = (_RANGE_MIN, _RANGE_MAX, n_cells)
    rule_jax = UniformInterpolationRule(degree)
    rule_cpp = sopp.UniformInterpolationRule(degree)
    function = _polynomial(value_size)

    spec = build_regular_grid_interpolant_3d(
        rule=rule_jax,
        xrange=axis_range,
        yrange=axis_range,
        zrange=axis_range,
        value_size=value_size,
        f=function,
        out_of_bounds_ok=True,
        skip=_box_skip,
    )
    device_spec = build_regular_grid_interpolant_3d_device_spec(spec)
    cpp_interpolant = sopp.RegularGridInterpolant3D(
        rule_cpp,
        axis_range,
        axis_range,
        axis_range,
        value_size,
        True,
        _box_skip,
    )
    cpp_interpolant.interpolate_batch(function)

    xyz = _sample_points(n_samples, seed)
    xyz_device = jnp.asarray(xyz)
    initial_device = jnp.zeros((n_samples, value_size), dtype=jnp.float64)

    def _jax_eval(points: jax.Array) -> jax.Array:
        return evaluate_batch_device(
            device_spec,
            points,
            initial_output=initial_device,
        )

    jax_warm = _jax_eval(xyz_device)
    jax_warm.block_until_ready()
    cpp_warm = np.zeros((n_samples, value_size), dtype=np.float64)
    cpp_interpolant.evaluate_batch(xyz, cpp_warm)

    def _time_jax() -> None:
        values = _jax_eval(xyz_device)
        values.block_until_ready()

    cpp_out = np.zeros((n_samples, value_size), dtype=np.float64)

    def _time_cpp() -> None:
        cpp_interpolant.evaluate_batch(xyz, cpp_out)

    jax_seconds = _median_timing(repeats, _time_jax)
    cpp_seconds = _median_timing(repeats, _time_cpp)

    cpp_check = np.zeros((n_samples, value_size), dtype=np.float64)
    cpp_interpolant.evaluate_batch(xyz, cpp_check)
    jax_check = np.asarray(_jax_eval(xyz_device))
    max_abs_error = float(np.max(np.abs(jax_check - cpp_check)))
    total_cells = int(n_cells**3)
    kept_cells = int(spec.cell_table.shape[0] - 1)
    skipped_cells = total_cells - kept_cells
    return {
        "benchmark": "regular_grid_skip_cell_map",
        "n_cells_per_axis": int(n_cells),
        "degree": int(degree),
        "value_size": int(value_size),
        "n_samples": int(n_samples),
        "repeats": int(repeats),
        "total_cells": total_cells,
        "kept_cells": kept_cells,
        "skipped_cells": skipped_cells,
        "skip_fraction": float(skipped_cells / total_cells),
        "jax_sentinel_cell_to_row_bytes": int(spec.cell_to_row.nbytes),
        "jax_cell_table_bytes": int(spec.cell_table.nbytes),
        "jax_median_seconds": jax_seconds,
        "cpp_unordered_map_median_seconds": cpp_seconds,
        "max_abs_error": max_abs_error,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=30)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--value-size", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=507)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_skip_cell_map_benchmark(
        n_cells=args.n_cells,
        degree=args.degree,
        value_size=args.value_size,
        n_samples=args.n_samples,
        repeats=args.repeats,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
