#!/usr/bin/env python3
"""Launch the production GPU proof on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.validation_ladder_contract import (  # noqa: E402
    build_stage2_hf_plan,
    optimizer_drift_tolerances,
    resolve_probe_lane,
)
from benchmarks.single_stage_smoke_defaults import (  # noqa: E402
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_STAGE2_BS_REL_PATH,
)

DEFAULT_STAGE2_SEED_REL = str(DEFAULT_STAGE2_BS_REL_PATH)
DEFAULT_EQUILIBRIA_REL = "examples/single_stage_optimization/equilibria"
DEFAULT_PLASMA = DEFAULT_PLASMA_SURF_FILENAME
DEFAULT_IMAGE = os.environ.get("SIMSOPT_HF_GPU_IMAGE") or None
DEFAULT_TARGET_OPTIMIZER_BACKEND = "ondevice"
TARGET_OPTIMIZER_BACKENDS = ("ondevice", "scipy-jax")
DEFAULT_EXPECTED_JAX_VERSION = "0.9.2"
DEFAULT_CUDA_LIBRARY_MODE = "bundled"
_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME = "single_stage_jax_runtime_spec.json"
_SINGLE_STAGE_WARM_START_REQUIRED_FILES = (
    "surf_opt.json",
    "results.json",
    "biot_savart_opt.json",
)
_HF_JOB_SUCCESS_STAGE = "COMPLETED"
_HF_JOB_TERMINAL_FAILURE_STAGES = {"CANCELED", "ERROR", "DELETED"}
_HF_JOB_STATUS_POLL_INTERVAL_S = 5.0
_HF_JOB_STATUS_POLL_TIMEOUT_S = 300.0


def _resolve_hf_cli() -> str:
    if shutil.which("hf"):
        return "hf"
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


def _parse_hf_job_id_line(line: str) -> str | None:
    prefix = "Job started with ID:"
    if not line.startswith(prefix):
        return None
    job_id = line[len(prefix) :].strip()
    if not job_id:
        raise RuntimeError("HF Jobs CLI emitted an empty job id.")
    return job_id


def _inspect_hf_job_stage(hf_cli: str, job_id: str) -> str:
    completed = subprocess.run(
        [hf_cli, "jobs", "inspect", "--format", "json", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    jobs = json.loads(completed.stdout)
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise RuntimeError(f"Expected one HF job inspection record for {job_id}.")
    status = jobs[0].get("status")
    if not isinstance(status, dict):
        raise RuntimeError(f"HF job {job_id} inspection is missing status.")
    stage = status.get("stage")
    if not isinstance(stage, str) or not stage:
        raise RuntimeError(f"HF job {job_id} inspection is missing status.stage.")
    return stage


def _wait_for_successful_hf_job(
    hf_cli: str,
    job_id: str,
    *,
    poll_interval_s: float = _HF_JOB_STATUS_POLL_INTERVAL_S,
    timeout_s: float = _HF_JOB_STATUS_POLL_TIMEOUT_S,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        stage = _inspect_hf_job_stage(hf_cli, job_id)
        if stage == _HF_JOB_SUCCESS_STAGE:
            return
        if stage in _HF_JOB_TERMINAL_FAILURE_STAGES:
            raise SystemExit(f"HF job {job_id} finished with stage {stage}.")
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"HF job {job_id} did not reach {_HF_JOB_SUCCESS_STAGE} "
                f"within {timeout_s:.0f}s after log streaming ended; latest stage={stage}."
            )
        time.sleep(poll_interval_s)


def _run_hf_job_foreground(hf_cli: str, cli_args: list[str]) -> None:
    job_id = None
    with subprocess.Popen(
        cli_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("HF Jobs CLI stdout pipe was not created.")
        for line in process.stdout:
            print(line, end="", flush=True)
            parsed_job_id = _parse_hf_job_id_line(line)
            if parsed_job_id is not None:
                job_id = parsed_job_id
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cli_args)
    if job_id is None:
        raise RuntimeError("Could not parse HF job id from foreground run output.")
    _wait_for_successful_hf_job(hf_cli, job_id)


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


def _build_optional_stage2_geometry_args(args: argparse.Namespace) -> list[str]:
    stage2_plan = build_stage2_hf_plan(args.stage2_maxiter, args.geometry_rel_tol)
    if stage2_plan["geometry_rel_tol"] is None:
        return []
    return ["--geometry-rel-tol", str(stage2_plan["geometry_rel_tol"])]


def _build_optional_single_stage_boozer_backend_args(
    args: argparse.Namespace,
) -> list[str]:
    if args.single_stage_boozer_optimizer_backend is None:
        return []
    return [
        "--single-stage-boozer-optimizer-backend",
        args.single_stage_boozer_optimizer_backend,
    ]


def _build_optional_single_stage_benchmark_mode_args(
    args: argparse.Namespace,
) -> list[str]:
    if not args.single_stage_benchmark_mode:
        return []
    return ["--single-stage-benchmark-mode"]


def _build_optional_single_stage_success_filter_args(
    args: argparse.Namespace,
) -> list[str]:
    if not args.single_stage_disable_target_lane_success_filter:
        return []
    return ["--single-stage-disable-target-lane-success-filter"]


def _repo_relative_seed_path(value: str, *, option_name: str) -> str:
    remote_path = PurePosixPath(value)
    if remote_path.is_absolute() or ".." in remote_path.parts:
        raise SystemExit(
            f"{option_name} must be a repo-relative path available "
            "inside the target repo."
        )
    normalized = str(remote_path)
    if not normalized or normalized == ".":
        raise SystemExit(f"{option_name} must not be empty.")
    return normalized


def _remote_repo_path(repo_dir: str, relative_path: str) -> str:
    return f"{repo_dir}/{relative_path}"


def _build_optional_single_stage_seed_args(
    args: argparse.Namespace,
    *,
    repo_dir: str,
) -> list[str]:
    args_out: list[str] = []
    if args.single_stage_warm_start_run_dir is not None:
        args_out.extend(
            [
                "--single-stage-warm-start-run-dir",
                _remote_repo_path(repo_dir, args.single_stage_warm_start_run_dir),
            ]
        )
    if args.single_stage_jax_runtime_seed_spec is not None:
        args_out.extend(
            [
                "--single-stage-jax-runtime-seed-spec",
                _remote_repo_path(repo_dir, args.single_stage_jax_runtime_seed_spec),
            ]
        )
    return args_out


def _resolve_repo_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.repo_url is None:
        args.repo_url = _resolve_default_repo_url()
    if args.repo_sha is None:
        args.repo_sha = _git_output("rev-parse", "HEAD")
    if args.repo_ref is None:
        args.repo_ref = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    return args


def _validate_runtime_contract(args: argparse.Namespace) -> argparse.Namespace:
    if not args.image:
        raise SystemExit(
            "Production GPU proof requires a prebuilt image via SIMSOPT_HF_GPU_IMAGE "
            "or --image."
        )
    if (
        args.single_stage_warm_start_run_dir is None
        and args.single_stage_jax_runtime_seed_spec is None
    ):
        raise SystemExit(
            "Production GPU proof requires a single-stage seed via "
            "--single-stage-warm-start-run-dir or "
            "--single-stage-jax-runtime-seed-spec."
        )
    if args.single_stage_warm_start_run_dir is not None:
        args.single_stage_warm_start_run_dir = _repo_relative_seed_path(
            args.single_stage_warm_start_run_dir,
            option_name="--single-stage-warm-start-run-dir",
        )
    if args.single_stage_jax_runtime_seed_spec is not None:
        args.single_stage_jax_runtime_seed_spec = _repo_relative_seed_path(
            args.single_stage_jax_runtime_seed_spec,
            option_name="--single-stage-jax-runtime-seed-spec",
        )
    return args


def _remote_single_stage_seed_contract_paths(
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    paths = []
    if args.single_stage_warm_start_run_dir is not None:
        for filename in _SINGLE_STAGE_WARM_START_REQUIRED_FILES:
            paths.append(
                (
                    "--single-stage-warm-start-run-dir",
                    f"{args.single_stage_warm_start_run_dir}/{filename}",
                )
            )
    if args.single_stage_jax_runtime_seed_spec is not None:
        paths.append(
            (
                "--single-stage-jax-runtime-seed-spec",
                args.single_stage_jax_runtime_seed_spec,
            )
        )
    return paths


def _remote_proof_seed_contract_paths(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [
        ("--stage2-bs-path", DEFAULT_STAGE2_SEED_REL),
        *_remote_single_stage_seed_contract_paths(args),
    ]


def _remote_object_type(temp_dir: str, resolved_sha: str, relative_path: str) -> str:
    object_type = subprocess.run(
        [
            "git",
            "-C",
            temp_dir,
            "cat-file",
            "-t",
            f"{resolved_sha}:{relative_path}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if object_type.returncode != 0:
        raise SystemExit(
            f"path is not present at repo SHA {resolved_sha}: {relative_path}"
        )
    return object_type.stdout.strip()


def _verify_remote_blob_path(
    temp_dir: str,
    resolved_sha: str,
    *,
    option_name: str,
    relative_path: str,
) -> None:
    try:
        object_type = _remote_object_type(temp_dir, resolved_sha, relative_path)
    except SystemExit as exc:
        raise SystemExit(f"{option_name}: {exc}") from exc
    if object_type != "blob":
        raise SystemExit(
            f"{option_name} must point to a file at repo SHA {resolved_sha}: "
            f"{relative_path}"
        )


def _verify_remote_runtime_seed_spec_path(
    temp_dir: str,
    resolved_sha: str,
    relative_path: str,
) -> None:
    object_type = _remote_object_type(temp_dir, resolved_sha, relative_path)
    if object_type == "tree":
        nested_type = _remote_object_type(
            temp_dir,
            resolved_sha,
            f"{relative_path}/{_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME}",
        )
        if nested_type != "blob":
            raise SystemExit(
                "--single-stage-jax-runtime-seed-spec directory must contain "
                f"a file named {_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME} at "
                f"repo SHA {resolved_sha}: {relative_path}"
            )
        return
    if object_type != "blob":
        raise SystemExit(
            "--single-stage-jax-runtime-seed-spec must point to a file or "
            f"directory at repo SHA {resolved_sha}: {relative_path}"
        )


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
        for option_name, relative_path in _remote_proof_seed_contract_paths(args):
            if option_name == "--single-stage-jax-runtime-seed-spec":
                _verify_remote_runtime_seed_spec_path(
                    temp_dir,
                    resolved_sha,
                    relative_path,
                )
            else:
                _verify_remote_blob_path(
                    temp_dir,
                    resolved_sha,
                    option_name=option_name,
                    relative_path=relative_path,
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
        "image": args.image,
        "expected_jax_version": DEFAULT_EXPECTED_JAX_VERSION,
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
        "single_stage_warm_start_run_dir": args.single_stage_warm_start_run_dir,
        "single_stage_jax_runtime_seed_spec": args.single_stage_jax_runtime_seed_spec,
        "stage2_maxiter": int(args.stage2_maxiter),
        "stage2_geometry_override": stage2_plan["geometry_rel_tol"],
        "stage2_geometry_policy": stage2_plan["geometry_policy"],
        "effective_geometry_rel_tol": stage2_plan["effective_geometry_rel_tol"],
        "effective_final_objective_rel_tol": stage2_tolerances[
            "final_objective_rel_tol"
        ],
    }


def _build_run_proof_argv(
    args: argparse.Namespace,
    *,
    repo_dir: str,
    results_dir: str,
    equilibria_dir: str,
    stage2_seed: str,
) -> list[str]:
    """Assemble the run_production_gpu_proof.sh argv as a token list."""
    return [
        "bash",
        "benchmarks/hf_jobs/run_production_gpu_proof.sh",
        "--results-dir",
        results_dir,
        "--equilibria-dir",
        equilibria_dir,
        "--plasma-surf-filename",
        DEFAULT_PLASMA,
        "--stage2-bs-path",
        stage2_seed,
        "--stage2-platform",
        args.platform,
        "--single-stage-platform",
        args.platform,
        "--stage2-nphi",
        str(args.stage2_nphi),
        "--stage2-ntheta",
        str(args.stage2_ntheta),
        "--stage2-maxiter",
        str(args.stage2_maxiter),
        "--stage2-optimizer-backend",
        args.stage2_optimizer_backend,
        *_build_optional_stage2_geometry_args(args),
        "--single-stage-nphi",
        str(args.single_stage_nphi),
        "--single-stage-ntheta",
        str(args.single_stage_ntheta),
        "--single-stage-mpol",
        str(args.single_stage_mpol),
        "--single-stage-ntor",
        str(args.single_stage_ntor),
        "--single-stage-maxiter",
        str(args.single_stage_maxiter),
        "--single-stage-optimizer-backend",
        args.single_stage_optimizer_backend,
        *_build_optional_single_stage_boozer_backend_args(args),
        *_build_optional_single_stage_seed_args(args, repo_dir=repo_dir),
        *_build_optional_single_stage_benchmark_mode_args(args),
        *_build_optional_single_stage_success_filter_args(args),
    ]


def _build_job_command(args: argparse.Namespace, *, resolved_repo_sha: str) -> str:
    repo_dir = "/tmp/hf-production-proof/repo"
    results_dir = "/tmp/hf-production-proof/results"
    equilibria_dir = f"{repo_dir}/{DEFAULT_EQUILIBRIA_REL}"
    stage2_seed = f"{repo_dir}/{DEFAULT_STAGE2_SEED_REL}"
    git_clone = shlex.join(
        [
            "git",
            "clone",
            "--recursive",
            "--branch",
            args.repo_ref,
            "--single-branch",
            args.repo_url,
            repo_dir,
        ]
    )
    run_proof = shlex.join(
        _build_run_proof_argv(
            args,
            repo_dir=repo_dir,
            results_dir=results_dir,
            equilibria_dir=equilibria_dir,
            stage2_seed=stage2_seed,
        )
    )
    command_lines = [
        "set -euxo pipefail",
        "export PYTHONUNBUFFERED=1",
        "export HF_HUB_DISABLE_TELEMETRY=1",
        "# Keep JAX from reserving most VRAM before the proof kernels allocate.",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        'export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"',
        f'export SIMSOPT_HF_JOB_EXPECTED_JAX_VERSION="{DEFAULT_EXPECTED_JAX_VERSION}"',
        f'export SIMSOPT_JAX_CUDA_LIBRARY_MODE="{DEFAULT_CUDA_LIBRARY_MODE}"',
        "rm -rf /tmp/hf-production-proof",
        "mkdir -p /tmp/hf-production-proof",
        git_clone,
        f"cd {shlex.quote(repo_dir)}",
        shlex.join(["git", "checkout", resolved_repo_sha]),
        "git submodule update --init --recursive",
        'export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PWD}/.artifacts/jax_compilation_cache/hf-production-proof}"',
        'mkdir -p "$JAX_COMPILATION_CACHE_DIR"',
        ". benchmarks/hf_jobs/bootstrap_runtime.sh",
        "python -m pip install -v -e .",
        run_proof,
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
            "Docker image to use for the job. Defaults to SIMSOPT_HF_GPU_IMAGE when "
            "set."
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
        choices=("cuda",),
        default="cuda",
        help="JAX platform to request inside the proof jobs.",
    )
    parser.add_argument("--timeout", default="8h", help="HF Jobs timeout.")
    parser.add_argument(
        "--stage2-optimizer-backend",
        choices=TARGET_OPTIMIZER_BACKENDS,
        default=DEFAULT_TARGET_OPTIMIZER_BACKEND,
    )
    parser.add_argument("--stage2-nphi", type=int, default=255)
    parser.add_argument("--stage2-ntheta", type=int, default=64)
    parser.add_argument("--stage2-maxiter", type=int, default=20)
    parser.add_argument("--geometry-rel-tol", type=float, default=None)
    parser.add_argument(
        "--single-stage-optimizer-backend",
        choices=TARGET_OPTIMIZER_BACKENDS,
        default=DEFAULT_TARGET_OPTIMIZER_BACKEND,
    )
    parser.add_argument(
        "--single-stage-boozer-optimizer-backend",
        choices=(DEFAULT_TARGET_OPTIMIZER_BACKEND,),
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
    parser.add_argument("--single-stage-warm-start-run-dir", default=None)
    parser.add_argument("--single-stage-jax-runtime-seed-spec", default=None)
    parser.add_argument(
        "--single-stage-benchmark-mode",
        action="store_true",
        help=(
            "Run the single-stage parity probes in benchmark/proof mode. "
            "This preserves CPU-vs-JAX comparison while using the short "
            "target-lane proof settings."
        ),
    )
    parser.add_argument(
        "--single-stage-disable-target-lane-success-filter",
        action="store_true",
        help=(
            "Thread the single-stage proof-only hardware success-filter bypass "
            "into the target-lane parity probes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved hf jobs commands without launching them.",
    )
    parser.add_argument(
        "--no-detach",
        action="store_false",
        dest="detach",
        default=True,
        help=(
            "Run each HF job in the foreground so this launcher exits nonzero "
            "when the remote proof fails."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _validate_runtime_contract(_resolve_repo_defaults(parse_args()))
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
        ]
        if args.detach:
            cli_args.append("--detach")
        cli_args.extend(
            [
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
        )
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in cli_args))
            continue
        if args.detach:
            completed = subprocess.run(
                cli_args,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"{hardware}: {completed.stdout.strip()}")
            continue
        _run_hf_job_foreground(hf_cli, cli_args)


if __name__ == "__main__":
    main()
