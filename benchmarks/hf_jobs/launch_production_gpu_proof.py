#!/usr/bin/env python3
"""Launch the production GPU proof on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE2_SEED_REL = (
    "benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"
)
DEFAULT_EQUILIBRIA_REL = "examples/single_stage_optimization/equilibria"
DEFAULT_PLASMA = "wout_nfp22ginsburg_000_014417_iota15.nc"
DEFAULT_IMAGE = os.environ.get("SIMSOPT_HF_GPU_IMAGE", "python:3.11-bookworm")


def _resolve_hf_cli() -> str:
    if shutil.which("hf"):
        return "hf"
    fallback = Path("/tmp/hfhub18/bin/hf")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("Could not find the Hugging Face CLI ('hf').")


def _https_clone_url(remote_url: str) -> str:
    if remote_url.startswith("https://"):
        return remote_url
    prefix = "git@github.com:"
    if remote_url.startswith(prefix):
        return "https://github.com/" + remote_url[len(prefix) :]
    raise RuntimeError(f"Unsupported remote URL for HF clone: {remote_url}")


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_job_command(args: argparse.Namespace) -> str:
    repo_dir = "/tmp/hf-production-proof/repo"
    results_dir = "/tmp/hf-production-proof/results"
    equilibria_dir = f"{repo_dir}/{DEFAULT_EQUILIBRIA_REL}"
    stage2_seed = f"{repo_dir}/{DEFAULT_STAGE2_SEED_REL}"
    command_lines = [
        "set -euxo pipefail",
        "export PYTHONUNBUFFERED=1",
        "export HF_HUB_DISABLE_TELEMETRY=1",
        'export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compilation-cache}"',
        f'export SIMSOPT_HF_JOB_BOOTSTRAP_MODE="{args.bootstrap_mode}"',
        "unset LD_LIBRARY_PATH",
        "rm -rf /tmp/hf-production-proof",
        'mkdir -p /tmp/hf-production-proof "$JAX_COMPILATION_CACHE_DIR"',
        f"git clone --recursive {shlex.quote(args.repo_url)} {shlex.quote(repo_dir)}",
        f"cd {shlex.quote(repo_dir)}",
        f"git checkout {shlex.quote(args.repo_sha)}",
        "git submodule update --init --recursive",
        "bash benchmarks/hf_jobs/bootstrap_runtime.sh",
        "python -m pip install -v -e . --no-build-isolation",
        (
            "bash benchmarks/hf_jobs/run_production_gpu_proof.sh "
            f"--results-dir {shlex.quote(results_dir)} "
            f"--equilibria-dir {shlex.quote(equilibria_dir)} "
            f"--plasma-surf-filename {shlex.quote(DEFAULT_PLASMA)} "
            f"--stage2-bs-path {shlex.quote(stage2_seed)} "
            f"--stage2-platform {shlex.quote(args.platform)} "
            f"--single-stage-platform {shlex.quote(args.platform)} "
            f"--stage2-nphi {args.stage2_nphi} "
            f"--stage2-ntheta {args.stage2_ntheta} "
            f"--stage2-maxiter {args.stage2_maxiter} "
            f"--stage2-optimizer-backend {shlex.quote(args.stage2_optimizer_backend)} "
            f"--geometry-rel-tol {shlex.quote(args.geometry_rel_tol)} "
            f"--single-stage-nphi {args.single_stage_nphi} "
            f"--single-stage-ntheta {args.single_stage_ntheta} "
            f"--single-stage-mpol {args.single_stage_mpol} "
            f"--single-stage-ntor {args.single_stage_ntor} "
            f"--single-stage-maxiter {args.single_stage_maxiter} "
            f"--single-stage-optimizer-backend {shlex.quote(args.single_stage_optimizer_backend)} "
            f"--single-stage-boozer-optimizer-backend {shlex.quote(args.single_stage_boozer_optimizer_backend)}"
        ),
    ]
    return "\n".join(command_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the production GPU proof on Hugging Face Jobs."
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=(
            "Docker image to use for the job. Defaults to SIMSOPT_HF_GPU_IMAGE or "
            "python:3.11-bookworm."
        ),
    )
    parser.add_argument(
        "--repo-url",
        default=_https_clone_url(_git_output("remote", "get-url", "fork")),
        help="HTTPS clone URL for the repo to validate.",
    )
    parser.add_argument(
        "--repo-sha",
        default=_git_output("rev-parse", "HEAD"),
        help="Exact git SHA to validate.",
    )
    parser.add_argument(
        "--hardware",
        nargs="+",
        default=["a100-large", "h200"],
        help="HF Jobs hardware flavors to launch.",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "cuda"),
        default="cuda",
        help="JAX platform to request inside the proof jobs.",
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Runtime bootstrap mode. 'auto' reuses /opt/venv when the image "
            "already contains the heavy dependencies."
        ),
    )
    parser.add_argument("--timeout", default="8h", help="HF Jobs timeout.")
    parser.add_argument(
        "--stage2-optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default="ondevice",
    )
    parser.add_argument("--stage2-nphi", type=int, default=255)
    parser.add_argument("--stage2-ntheta", type=int, default=64)
    parser.add_argument("--stage2-maxiter", type=int, default=20)
    parser.add_argument("--geometry-rel-tol", default="5e-6")
    parser.add_argument(
        "--single-stage-optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default="ondevice",
    )
    parser.add_argument(
        "--single-stage-boozer-optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default="scipy",
    )
    parser.add_argument("--single-stage-nphi", type=int, default=255)
    parser.add_argument("--single-stage-ntheta", type=int, default=64)
    parser.add_argument("--single-stage-mpol", type=int, default=8)
    parser.add_argument("--single-stage-ntor", type=int, default=6)
    parser.add_argument("--single-stage-maxiter", type=int, default=300)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved hf jobs commands without launching them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_cli = _resolve_hf_cli()
    job_command = _build_job_command(args)
    for hardware in args.hardware:
        cli_args = [
            hf_cli,
            "jobs",
            "run",
            "--flavor",
            hardware,
            "--timeout",
            args.timeout,
            "--detach",
            "--label",
            "project=columbia",
            "--label",
            "task=production-gpu-proof",
            "--label",
            f"hardware={hardware}",
            "--label",
            f"sha={args.repo_sha}",
            args.image,
            "--",
            "bash",
            "-lc",
            job_command,
        ]
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in cli_args))
            continue
        completed = subprocess.run(cli_args, check=True, capture_output=True, text=True)
        print(f"{hardware}: {completed.stdout.strip()}")


if __name__ == "__main__":
    main()
