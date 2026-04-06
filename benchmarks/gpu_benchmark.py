#!/usr/bin/env python3
"""Legacy compatibility wrapper for the authoritative local GPU ladder.

This entrypoint used to clone an external branch and run ad hoc microbenchmarks.
It now delegates to the checked-in Tier 5 characterization, grouped-adjoint
memory probe, and standardized report renderer in the current repo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "benchmark_artifacts"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifests" / "stable_hardware_weekly_tier5.json"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the checked-in simsopt-jax GPU validation ladder from the current repo."
        )
    )
    parser.add_argument("--platform", default="cuda")
    parser.add_argument("--optimizer-backend", default="ondevice")
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--manifest-json",
        default=str(DEFAULT_MANIFEST),
        help="Manifest used when rendering the standardized markdown report.",
    )
    parser.add_argument(
        "--benchmark-mode",
        dest="benchmark_mode",
        action="store_true",
        default=True,
        help="Enable steady-state timing mode for the delegated probes.",
    )
    parser.add_argument(
        "--no-benchmark-mode",
        dest="benchmark_mode",
        action="store_false",
        help="Disable steady-state timing mode for the delegated probes.",
    )
    return parser.parse_known_args()


def _run_script(script_name: str, args: list[str]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_DIR / script_name),
            *args,
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


def main() -> None:
    args, passthrough = parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    benchmark_flag = ["--benchmark-mode"] if args.benchmark_mode else []

    tier5_json = artifacts_dir / "tier5_performance_characterization.json"
    grouped_json = artifacts_dir / "grouped_adjoint_memory_probe.json"
    grouped_profile = artifacts_dir / "grouped_adjoint_memory_profile.prof"
    report_md = artifacts_dir / "stable_hardware_weekly_report.md"

    common_args = [
        "--platform",
        args.platform,
        "--optimizer-backend",
        args.optimizer_backend,
        "--maxiter",
        str(args.maxiter),
        *benchmark_flag,
        *passthrough,
    ]

    _run_script(
        "tier5_performance_characterization.py",
        [
            *common_args,
            "--output-json",
            str(tier5_json),
        ],
    )
    _run_script(
        "grouped_adjoint_memory_probe.py",
        [
            "--platform",
            args.platform,
            "--optimizer-backend",
            args.optimizer_backend,
            "--output-json",
            str(grouped_json),
            "--device-memory-profile-out",
            str(grouped_profile),
        ],
    )
    _run_script(
        "render_benchmark_report.py",
        [
            "--manifest-json",
            str(Path(args.manifest_json).resolve()),
            "--input-json",
            str(tier5_json),
            "--output-md",
            str(report_md),
        ],
    )


if __name__ == "__main__":
    main()
