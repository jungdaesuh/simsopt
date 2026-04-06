#!/usr/bin/env python3
"""Legacy compatibility wrapper for the authoritative local GPU ladder.

This entrypoint used to clone an external branch and run ad hoc microbenchmarks.
It now delegates to the checked-in Tier 5 characterization, grouped-adjoint
memory probe, and standardized report renderer in the current repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "benchmark_artifacts"
DEFAULT_MANIFEST = BENCHMARK_DIR / "manifests" / "stable_hardware_weekly_tier5.json"


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _command_env_overrides(
    manifest: dict[str, object],
    *,
    command_name: str,
) -> dict[str, str]:
    commands = manifest.get("commands")
    if not isinstance(commands, list):
        return {}
    for command in commands:
        if not isinstance(command, dict) or command.get("name") != command_name:
            continue
        raw_env = command.get("env")
        if not isinstance(raw_env, dict):
            return {}
        return {str(key): str(value) for key, value in raw_env.items()}
    return {}


def _ensure_compilation_cache_dir(env_overrides: dict[str, str]) -> None:
    cache_dir = env_overrides.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir is None:
        return
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    cache_path.mkdir(parents=True, exist_ok=True)


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


def _run_script(
    script_name: str,
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
) -> None:
    env = dict(os.environ)
    if env_overrides:
        _ensure_compilation_cache_dir(env_overrides)
        env.update(env_overrides)
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_DIR / script_name),
            *args,
        ],
        cwd=str(REPO_ROOT),
        check=True,
        env=env,
    )


def main() -> None:
    args, passthrough = parse_args()
    manifest_path = Path(args.manifest_json).resolve()
    manifest = _load_manifest(manifest_path)
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
        env_overrides=_command_env_overrides(
            manifest,
            command_name="tier5_performance_characterization",
        ),
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
        env_overrides=_command_env_overrides(
            manifest,
            command_name="grouped_adjoint_memory_probe",
        ),
    )
    _run_script(
        "render_benchmark_report.py",
        [
            "--manifest-json",
            str(manifest_path),
            "--input-json",
            str(tier5_json),
            "--output-md",
            str(report_md),
        ],
    )


if __name__ == "__main__":
    main()
