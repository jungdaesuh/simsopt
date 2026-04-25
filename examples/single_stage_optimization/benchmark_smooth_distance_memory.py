#!/usr/bin/env python3
"""Benchmark smooth distance constraint allocation on deterministic fixtures."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "smooth_distance_memory_v1"
DEFAULT_CASES = (
    "single-stage-curve-curve",
    "single-stage-curve-surface",
    "stage2-curve-curve",
    "stage2-curve-surface",
)


def _maxrss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _configure_imports(repo_root: Path) -> None:
    src_root = repo_root / "src"
    examples_root = repo_root / "examples" / "single_stage_optimization"
    for path in (str(examples_root), str(src_root)):
        sys.path[:] = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != Path(path).resolve()
        ]
        sys.path.insert(0, path)
    for module_name in tuple(sys.modules):
        if module_name == "simsopt" or module_name.startswith("simsopt."):
            del sys.modules[module_name]
        if module_name == "banana_opt" or module_name.startswith("banana_opt."):
            del sys.modules[module_name]
        if module_name == "alm_utils":
            del sys.modules[module_name]


def _load_functions(repo_root: Path):
    _configure_imports(repo_root)
    from simsopt._core.derivative import Derivative
    from simsopt._core.optimizable import Optimizable
    from banana_opt.single_stage_constraints import (
        smooth_min_curve_curve_signed_constraint as single_stage_curve_curve,
    )
    from banana_opt.single_stage_constraints import (
        smooth_min_curve_surface_signed_constraint as single_stage_curve_surface,
    )
    from banana_opt.stage2_objectives import (
        smooth_min_curve_surface_signed_constraint as stage2_curve_surface,
    )
    from banana_opt.stage2_objectives import (
        smooth_min_distance_signed_constraint as stage2_curve_curve,
    )

    return {
        "Derivative": Derivative,
        "Optimizable": Optimizable,
        "single_stage_curve_curve": single_stage_curve_curve,
        "single_stage_curve_surface": single_stage_curve_surface,
        "stage2_curve_curve": stage2_curve_curve,
        "stage2_curve_surface": stage2_curve_surface,
    }


def _curve_points(index: int, count: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    phase = 0.11 * float(index)
    radius = 0.22 + 0.004 * float(index % 3)
    return np.column_stack(
        (
            1.0 + 0.018 * float(index) + radius * np.cos(theta + phase),
            radius * np.sin(theta + phase),
            0.055 * np.sin(2.0 * theta + phase),
        )
    )


def _surface_points(nphi: int, ntheta: int) -> np.ndarray:
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    major_radius = 1.04
    minor_radius = 0.19
    return np.stack(
        (
            (major_radius + minor_radius * np.cos(theta_grid)) * np.cos(phi_grid),
            (major_radius + minor_radius * np.cos(theta_grid)) * np.sin(phi_grid),
            minor_radius * np.sin(theta_grid),
        ),
        axis=2,
    )


@dataclass
class GeometryFixture:
    curves: list
    surface: object
    objective: object


def _build_geometry(
    *,
    repo_root: Path,
    curve_count: int,
    curve_points: int,
    surface_phi: int,
    surface_theta: int,
) -> GeometryFixture:
    loaded = _load_functions(repo_root)
    derivative_type = loaded["Derivative"]
    optimizable_type = loaded["Optimizable"]

    class FakeGeometry(optimizable_type):
        def __init__(self, gamma: np.ndarray, name: str):
            super().__init__(x0=np.zeros(1), names=[name])
            self._gamma = np.asarray(gamma, dtype=float)

        def gamma(self) -> np.ndarray:
            return self._gamma

        def dgamma_by_dcoeff_vjp(self, point_gradient: np.ndarray):
            checksum = np.array([float(np.sum(point_gradient))], dtype=float)
            return derivative_type({self: checksum})

    curves = [
        FakeGeometry(_curve_points(index, curve_points), f"curve_{index}")
        for index in range(curve_count)
    ]
    surface = FakeGeometry(_surface_points(surface_phi, surface_theta), "surface")
    objective = optimizable_type(depends_on=[*curves, surface])
    return GeometryFixture(curves=curves, surface=surface, objective=objective)


def _build_operation(args: argparse.Namespace) -> Callable[[], tuple]:
    loaded = _load_functions(args.repo_root)
    fixture = _build_geometry(
        repo_root=args.repo_root,
        curve_count=args.curves,
        curve_points=args.curve_points,
        surface_phi=args.surface_phi,
        surface_theta=args.surface_theta,
    )
    minimum_distance = float(args.minimum_distance)
    temperature = float(args.temperature)

    if args.case == "single-stage-curve-curve":
        return lambda: loaded["single_stage_curve_curve"](
            fixture.curves,
            minimum_distance,
            temperature,
            fixture.objective,
        )
    if args.case == "single-stage-curve-surface":
        return lambda: loaded["single_stage_curve_surface"](
            fixture.curves,
            fixture.surface,
            minimum_distance,
            temperature,
            fixture.objective,
        )
    if args.case == "stage2-curve-curve":
        return lambda: loaded["stage2_curve_curve"](
            fixture.curves,
            minimum_distance,
            temperature,
            fixture.objective,
        )
    if args.case == "stage2-curve-surface":
        return lambda: loaded["stage2_curve_surface"](
            fixture.curves,
            fixture.surface,
            minimum_distance,
            temperature,
            fixture.objective,
        )
    raise ValueError(f"unknown case: {args.case}")


def _result_checksum(result: tuple) -> tuple[float, float, float]:
    signed_value = float(result[0])
    grad = np.asarray(result[1], dtype=float)
    violation = float(result[2]) if len(result) > 2 else max(0.0, signed_value)
    return signed_value, float(np.linalg.norm(grad)), violation


def measure_one(args: argparse.Namespace) -> dict[str, object]:
    operation = _build_operation(args)
    for _ in range(args.warmup):
        operation()

    tracemalloc.start()
    rss_before = _maxrss_bytes()
    times = []
    checksums = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = operation()
        times.append(time.perf_counter() - start)
        checksums.append(_result_checksum(result))
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _maxrss_bytes()

    return {
        "case": args.case,
        "repo_root": str(args.repo_root),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "curves": args.curves,
        "curve_points": args.curve_points,
        "surface_phi": args.surface_phi,
        "surface_theta": args.surface_theta,
        "minimum_distance": args.minimum_distance,
        "temperature": args.temperature,
        "seconds_min": min(times),
        "seconds_median": statistics.median(times),
        "seconds_mean": statistics.mean(times),
        "python_peak_bytes": int(python_peak),
        "process_maxrss_before_bytes": int(rss_before),
        "process_maxrss_after_bytes": int(rss_after),
        "checksum_first": checksums[0],
        "checksum_last": checksums[-1],
    }


def run_case_in_subprocess(args: argparse.Namespace, case: str) -> dict[str, object]:
    command = [
        sys.executable,
        __file__,
        "--measure-one",
        "--repo-root",
        str(args.repo_root),
        "--case",
        case,
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--curves",
        str(args.curves),
        "--curve-points",
        str(args.curve_points),
        "--surface-phi",
        str(args.surface_phi),
        "--surface-theta",
        str(args.surface_theta),
        "--minimum-distance",
        str(args.minimum_distance),
        "--temperature",
        str(args.temperature),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    cases = list(DEFAULT_CASES) if args.case == "all" else [args.case]
    results = [run_case_in_subprocess(args, case) for case in cases]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": str(args.repo_root),
        "cases": cases,
        "results": results,
    }


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure dense smooth-distance constraint allocations."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--case", choices=("all", *DEFAULT_CASES), default="all")
    parser.add_argument("--measure-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=_positive_int, default=3)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--curves", type=_positive_int, default=6)
    parser.add_argument("--curve-points", type=_positive_int, default=192)
    parser.add_argument("--surface-phi", type=_positive_int, default=96)
    parser.add_argument("--surface-theta", type=_positive_int, default=96)
    parser.add_argument("--minimum-distance", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    if args.measure_one:
        if args.case == "all":
            raise ValueError("--measure-one requires a concrete --case")
        print(json.dumps(measure_one(args), sort_keys=True))
        return 0

    payload = json.dumps(build_report(args), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
