#!/usr/bin/env python3
"""Launch the production GPU proof on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.validation_ladder_contract import (  # noqa: E402
    build_stage2_hf_plan,
    optimizer_drift_tolerances,
    resolve_probe_lane,
)

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
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise SystemExit(stderr or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _git_optional_output(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    stdout = completed.stdout.strip()
    return stdout or None


def _list_git_remotes() -> list[str]:
    remotes = _git_optional_output("remote")
    if remotes is None:
        return []
    return [remote for remote in remotes.splitlines() if remote]


def _resolve_default_repo_url() -> str:
    branch_name = _git_optional_output("symbolic-ref", "--short", "HEAD")
    if branch_name is not None:
        upstream_remote = _git_optional_output("config", f"branch.{branch_name}.remote")
        if upstream_remote is not None:
            upstream_url = _git_optional_output("remote", "get-url", upstream_remote)
            if upstream_url is not None:
                return _https_clone_url(upstream_url)

    remote_names = _list_git_remotes()
    if len(remote_names) == 1:
        remote_url = _git_optional_output("remote", "get-url", remote_names[0])
        if remote_url is not None:
            return _https_clone_url(remote_url)

    for remote_name in ("fork", "origin"):
        if remote_name not in remote_names:
            continue
        remote_url = _git_optional_output("remote", "get-url", remote_name)
        if remote_url is not None:
            return _https_clone_url(remote_url)

    configured = ", ".join(remote_names) if remote_names else "none"
    raise SystemExit(
        "Could not infer default --repo-url from git remotes "
        f"(configured: {configured}). Pass --repo-url explicitly."
    )


def _build_optional_stage2_geometry_flag(args: argparse.Namespace) -> str:
    stage2_plan = build_stage2_hf_plan(args.stage2_maxiter, args.geometry_rel_tol)
    if stage2_plan["geometry_rel_tol"] is None:
        return ""
    return (
        "--geometry-rel-tol "
        f"{shlex.quote(str(stage2_plan['geometry_rel_tol']))} "
    )


def _build_optional_single_stage_boozer_backend_flag(
    args: argparse.Namespace,
) -> str:
    if args.single_stage_boozer_optimizer_backend is None:
        return ""
    return (
        "--single-stage-boozer-optimizer-backend "
        f"{shlex.quote(args.single_stage_boozer_optimizer_backend)}"
    )


def _resolve_repo_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.repo_url is None:
        args.repo_url = _resolve_default_repo_url()
    if args.repo_sha is None:
        args.repo_sha = _git_output("rev-parse", "HEAD")
    if args.repo_ref is None:
        args.repo_ref = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    return args


def _resolve_remote_ref(repo_url: str, repo_ref: str) -> tuple[str, str]:
    ref_candidates = (
        [repo_ref]
        if repo_ref.startswith("refs/")
        else [f"refs/heads/{repo_ref}", f"refs/tags/{repo_ref}"]
    )
    completed = subprocess.run(
        ["git", "ls-remote", "--refs", repo_url, *ref_candidates],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise SystemExit(
            stderr or f"git ls-remote failed while checking {repo_ref} on {repo_url}"
        )
    matches: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        matches.append((parts[0], parts[1]))
    if not matches:
        raise SystemExit(
            f"repo ref {repo_ref!r} does not exist on {repo_url}; "
            "HF clone --branch would fail."
        )
    if len(matches) == 1 or repo_ref.startswith("refs/"):
        return matches[0]
    preferred_order = (
        f"refs/heads/{repo_ref}",
        f"refs/tags/{repo_ref}",
    )
    for preferred_ref in preferred_order:
        for match in matches:
            if match[1] == preferred_ref:
                return match
    return matches[0]


def _verify_remote_sha_ref_contract(args: argparse.Namespace) -> tuple[str, str]:
    if args.repo_ref == "HEAD":
        raise SystemExit(
            "Detached HEAD cannot be used as --repo-ref for HF clone preflight; "
            "pass an explicit branch or tag."
        )
    resolved_ref_commit, remote_ref_name = _resolve_remote_ref(
        args.repo_url,
        args.repo_ref,
    )
    with tempfile.TemporaryDirectory(prefix="hf-proof-preflight-") as temp_dir:
        subprocess.run(
            ["git", "init", temp_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", temp_dir, "remote", "add", "origin", args.repo_url],
            check=True,
            capture_output=True,
            text=True,
        )
        fetch = subprocess.run(
            [
                "git",
                "-C",
                temp_dir,
                "fetch",
                "--quiet",
                "--filter=tree:0",
                "--no-tags",
                "origin",
                f"{remote_ref_name}:refs/remotes/origin/preflight",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if fetch.returncode != 0:
            stderr = fetch.stderr.strip()
            raise SystemExit(
                stderr
                or f"git fetch failed while checking {remote_ref_name} on {args.repo_url}"
            )
        rev_parse = subprocess.run(
            [
                "git",
                "-C",
                temp_dir,
                "rev-parse",
                "--verify",
                f"{args.repo_sha}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if rev_parse.returncode != 0:
            raise SystemExit(
                f"repo SHA {args.repo_sha} is not present on {args.repo_url} "
                f"under {remote_ref_name}; HF checkout would fail."
            )
        resolved_sha = rev_parse.stdout.strip()
        reachable = subprocess.run(
            [
                "git",
                "-C",
                temp_dir,
                "merge-base",
                "--is-ancestor",
                resolved_sha,
                "refs/remotes/origin/preflight",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if reachable.returncode != 0:
            raise SystemExit(
                f"repo SHA {resolved_sha} is not reachable from repo ref {args.repo_ref} "
                f"on {args.repo_url}; HF clone --single-branch would fail."
            )
    return resolved_sha, resolved_ref_commit


def _build_preflight_report(args: argparse.Namespace) -> dict[str, object]:
    resolved_sha, resolved_ref = _verify_remote_sha_ref_contract(args)
    stage2_tolerances = optimizer_drift_tolerances(
        "tier2_stage2_e2e",
        maxiter=args.stage2_maxiter,
    )
    try:
        stage2_plan = build_stage2_hf_plan(args.stage2_maxiter, args.geometry_rel_tol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return {
        "repo_sha": resolved_sha,
        "repo_ref": args.repo_ref,
        "repo_ref_commit": resolved_ref,
        "repo_url": args.repo_url,
        "platform": args.platform,
        "hardware": list(args.hardware),
        "stage2_lane": resolve_probe_lane(
            optimizer_backend=args.stage2_optimizer_backend
        ),
        "single_stage_lane": resolve_probe_lane(
            optimizer_backend=args.single_stage_optimizer_backend
        ),
        "stage2_rungs": list(stage2_plan["stage2_rungs"]),
        "single_stage_rungs": ["single_stage_cold", "single_stage_warm"],
        "stage2_maxiter": int(args.stage2_maxiter),
        "stage2_geometry_override": stage2_plan["geometry_rel_tol"],
        "stage2_geometry_policy": stage2_plan["geometry_policy"],
        "effective_geometry_rel_tol": stage2_plan["effective_geometry_rel_tol"],
        "effective_final_objective_rel_tol": stage2_tolerances[
            "final_objective_rel_tol"
        ],
    }


def _build_job_command(args: argparse.Namespace, *, resolved_repo_sha: str) -> str:
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
        (
            "git clone --recursive"
            f" --branch {shlex.quote(args.repo_ref)}"
            " --single-branch"
            f" {shlex.quote(args.repo_url)} {shlex.quote(repo_dir)}"
        ),
        f"cd {shlex.quote(repo_dir)}",
        f"git checkout {shlex.quote(resolved_repo_sha)}",
        "git submodule update --init --recursive",
        ". benchmarks/hf_jobs/bootstrap_runtime.sh",
        "python -m pip install -v -e .",
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
            f"{_build_optional_stage2_geometry_flag(args)}"
            f"--single-stage-nphi {args.single_stage_nphi} "
            f"--single-stage-ntheta {args.single_stage_ntheta} "
            f"--single-stage-mpol {args.single_stage_mpol} "
            f"--single-stage-ntor {args.single_stage_ntor} "
            f"--single-stage-maxiter {args.single_stage_maxiter} "
            f"--single-stage-optimizer-backend {shlex.quote(args.single_stage_optimizer_backend)} "
            f"{_build_optional_single_stage_boozer_backend_flag(args)}"
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
        default=None,
        help="HTTPS clone URL for the repo to validate.",
    )
    parser.add_argument(
        "--repo-sha",
        default=None,
        help="Exact git SHA to validate.",
    )
    parser.add_argument(
        "--repo-ref",
        default=None,
        help="Branch or ref to clone before checking out the validation SHA.",
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
    parser.add_argument("--geometry-rel-tol", type=float, default=None)
    parser.add_argument(
        "--single-stage-optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default="ondevice",
    )
    parser.add_argument(
        "--single-stage-boozer-optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default=None,
        help=(
            "Optional override for the single-stage inner Boozer LS backend. "
            "Defaults to the outer single-stage optimizer backend when omitted."
        ),
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
    args = _resolve_repo_defaults(parse_args())
    hf_cli = _resolve_hf_cli()
    preflight = _build_preflight_report(args)
    print(json.dumps({"preflight": preflight}, indent=2))
    job_command = _build_job_command(
        args,
        resolved_repo_sha=str(preflight["repo_sha"]),
    )
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
            f"sha={preflight['repo_sha']}",
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
