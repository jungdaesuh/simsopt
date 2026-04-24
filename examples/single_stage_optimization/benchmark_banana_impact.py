#!/usr/bin/env python3
"""Measure banana optimization impact on fixed low-resolution fixtures.

This script is intentionally opt-in. Use it before and after a targeted
optimization patch to record wall time, Python allocation peak, and process RSS
on the same fixture.
"""

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from import_provenance import configure_local_simsopt_imports

SIMSOPT_ROOT = Path(__file__).resolve().parents[2]
configure_local_simsopt_imports(simsopt_root=str(SIMSOPT_ROOT))

from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux


SCHEMA_VERSION = "banana_impact_benchmark_v1"
BenchmarkCallable = Callable[[], float]


@dataclass(frozen=True)
class BenchmarkFixture:
    name: str
    description: str
    build: Callable[[], BenchmarkCallable]


def _maxrss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _quadpoints(count: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, count, endpoint=False)


def _surface(nfp: int = 2, nphi: int = 32, ntheta: int = 24) -> SurfaceRZFourier:
    return SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=_quadpoints(nphi),
        quadpoints_theta=_quadpoints(ntheta),
    )


def _base_curves(
    *,
    ncurves: int = 4,
    nfp: int = 2,
    order: int = 3,
    numquadpoints: int = 64,
) -> list:
    return create_equally_spaced_curves(
        ncurves,
        nfp,
        stellsym=True,
        R0=1.0,
        R1=0.28,
        order=order,
        numquadpoints=numquadpoints,
    )


def _coils_for_surface(surface: SurfaceRZFourier, ncurves: int = 4) -> list:
    curves = _base_curves(ncurves=ncurves, nfp=surface.nfp)
    currents = [Current(1.0e5) for _ in curves]
    return coils_via_symmetries(curves, currents, surface.nfp, surface.stellsym)


def _build_squared_flux() -> BenchmarkCallable:
    surface = _surface()
    biot_savart = BiotSavart(_coils_for_surface(surface))
    objective = SquaredFlux(surface, biot_savart, definition="quadratic flux")

    def run() -> float:
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=float)
        return value + 1.0e-30 * float(np.linalg.norm(gradient))

    return run


def _build_curve_surface_distance() -> BenchmarkCallable:
    surface = _surface()
    curves = _base_curves(nfp=surface.nfp)
    objective = CurveSurfaceDistance(curves, surface, minimum_distance=0.15)

    def run() -> float:
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=float)
        distance = float(objective.shortest_distance())
        return value + distance + 1.0e-30 * float(np.linalg.norm(gradient))

    return run


def _build_magnetic_field_sum() -> BenchmarkCallable:
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    points = np.column_stack(
        (
            1.0 + 0.1 * np.cos(phi),
            0.1 * np.sin(phi),
            0.05 * np.sin(2.0 * phi),
        )
    )
    field = MagneticFieldSum(
        [
            ToroidalField(1.0, 1.0),
            ToroidalField(1.2, 0.7),
            ToroidalField(0.8, 0.4),
        ]
    )
    field.set_points(points)

    def run() -> float:
        checksum = float(np.sum(field.B()))
        checksum += 1.0e-6 * float(np.sum(field.dB_by_dX()))
        checksum += 1.0e-9 * float(np.sum(field.A()))
        return checksum

    return run


def _build_biot_savart() -> BenchmarkCallable:
    surface = _surface(nphi=24, ntheta=18)
    biot_savart = BiotSavart(_coils_for_surface(surface))
    biot_savart.set_points(surface.gamma().reshape((-1, 3)))

    def run() -> float:
        checksum = float(np.sum(biot_savart.B()))
        checksum += 1.0e-6 * float(np.sum(biot_savart.dB_by_dX()))
        return checksum

    return run


FIXTURES: dict[str, BenchmarkFixture] = {
    "squared-flux": BenchmarkFixture(
        name="squared-flux",
        description="SquaredFlux J and dJ on a fixed low-resolution surface/coil set.",
        build=_build_squared_flux,
    ),
    "curve-surface-distance": BenchmarkFixture(
        name="curve-surface-distance",
        description="CurveSurfaceDistance J, dJ, and exact sampled distance.",
        build=_build_curve_surface_distance,
    ),
    "magnetic-field-sum": BenchmarkFixture(
        name="magnetic-field-sum",
        description="MagneticFieldSum B, dB_by_dX, and A accumulation.",
        build=_build_magnetic_field_sum,
    ),
    "biot-savart": BenchmarkFixture(
        name="biot-savart",
        description="BiotSavart B and dB_by_dX on a fixed surface point cloud.",
        build=_build_biot_savart,
    ),
}


def measure_operation(
    *,
    name: str,
    description: str,
    build: Callable[[], BenchmarkCallable],
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    operation = build()

    for _ in range(warmup):
        operation()

    gc.collect()
    tracemalloc.start()
    times = []
    checksums = []
    rss_before = _maxrss_bytes()

    for _ in range(repeat):
        start = time.perf_counter()
        checksums.append(float(operation()))
        times.append(time.perf_counter() - start)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _maxrss_bytes()

    return {
        "name": name,
        "description": description,
        "repeat": repeat,
        "warmup": warmup,
        "seconds_min": min(times),
        "seconds_median": statistics.median(times),
        "seconds_mean": statistics.mean(times),
        "python_peak_bytes": int(peak_bytes),
        "process_peak_rss_bytes": int(rss_after),
        "process_maxrss_before_bytes": int(rss_before),
        "process_maxrss_after_bytes": int(rss_after),
        "checksum_first": checksums[0],
        "checksum_last": checksums[-1],
    }


def measure_fixture(
    fixture_name: str,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        __file__,
        "--measure-one",
        "--fixture",
        fixture_name,
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def build_report(
    fixture_names: list[str],
    repeat: int,
    warmup: int,
    fixtures: Mapping[str, BenchmarkFixture] = FIXTURES,
) -> dict[str, object]:
    selected_fixture_names = list(fixtures) if "all" in fixture_names else fixture_names
    results = []
    for fixture_name in selected_fixture_names:
        fixture = fixtures[fixture_name]
        if fixtures is FIXTURES:
            results.append(
                measure_fixture(
                    fixture.name,
                    repeat=repeat,
                    warmup=warmup,
                )
            )
        else:
            results.append(
                measure_operation(
                    name=fixture.name,
                    description=fixture.description,
                    build=fixture.build,
                    repeat=repeat,
                    warmup=warmup,
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "repeat": repeat,
        "warmup": warmup,
        "fixtures": selected_fixture_names,
        "results": results,
    }


def _format_bytes(num_bytes: object) -> str:
    return f"{int(num_bytes) / (1024 * 1024):.3f} MiB"


def render_markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        "# Banana Impact Benchmark",
        "",
        f"- Schema: `{report['schema_version']}`",
        f"- Created UTC: `{report['created_at_utc']}`",
        f"- Repeat: `{report['repeat']}`",
        f"- Warmup: `{report['warmup']}`",
        "",
        "| Fixture | Median seconds | Mean seconds | Python peak | Process peak RSS | Checksum first | Checksum last |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        lines.append(
            "| "
            f"{result['name']} | "
            f"{float(result['seconds_median']):.9g} | "
            f"{float(result['seconds_mean']):.9g} | "
            f"{_format_bytes(result['python_peak_bytes'])} | "
            f"{_format_bytes(result['process_peak_rss_bytes'])} | "
            f"{float(result['checksum_first']):.9g} | "
            f"{float(result['checksum_last']):.9g} |"
        )
    return "\n".join(lines) + "\n"


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure banana optimization impact on fixed low-resolution fixtures."
    )
    parser.add_argument(
        "--fixture",
        action="append",
        choices=["all", *sorted(FIXTURES)],
        help="Fixture to run. Repeat the flag to run several fixtures. Defaults to all.",
    )
    parser.add_argument("--measure-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_names = args.fixture if args.fixture is not None else ["all"]
    if args.measure_one:
        if len(fixture_names) != 1 or fixture_names[0] == "all":
            raise ValueError("--measure-one requires exactly one concrete --fixture")
        fixture = FIXTURES[fixture_names[0]]
        print(
            json.dumps(
                measure_operation(
                    name=fixture.name,
                    description=fixture.description,
                    build=fixture.build,
                    repeat=args.repeat,
                    warmup=args.warmup,
                ),
                sort_keys=True,
            )
        )
        return 0

    report = build_report(fixture_names, repeat=args.repeat, warmup=args.warmup)
    if args.format == "json":
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    else:
        payload = render_markdown_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
