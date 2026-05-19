#!/usr/bin/env python3
"""Launch the production GPU proof on Lightning AI.

The proof contract is shared with the HF Jobs launcher
(:mod:`benchmarks.hf_jobs.launch_production_gpu_proof`): same git clone,
same bootstrap, same ``run_production_gpu_proof.sh`` arguments,
``--stage2-platform cuda``, identical ladder-contract probes. The two
routes therefore satisfy the same ``GPU_PROOF_PARITY_CONTRACTS`` (see
``benchmarks/validation_ladder_contract.py``) and produce parity probes
whose numerical fields are launcher-agnostic for a given SHA. Provenance
fields recorded by the run script intentionally vary by launcher
(``JAX_COMPILATION_CACHE_DIR`` subdirectory, container ``results-dir``,
container ``repo-dir``), since those describe the host environment the
proof ran in rather than the proof itself.

This launcher specialises only the Lightning-specific pieces:

* the proof results dir lives inside the ``simsopt-jax-parity-proofs`` data
  connection mounted at ``/proof`` (audit doc records that mounting a
  missing nested connection path fails before container startup, so the
  job mounts the root and creates the run-specific subdirectory itself);
* ``entrypoint="bash -lc"`` because Lightning's default ``sh -c`` shell
  rejects ``set -o pipefail`` (the failure mode that killed the first
  Nebius H200 attempt on 2026-05-12);
* ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT`` is exported before
  ``pip install -e .`` so non-PEP440 release tags like
  ``banana-surface-parity-m7-unitnormal-r1`` cannot break the install
  step (the failure mode that killed the second Nebius H200 attempt);
* the preflight JSON is written next to the existing
  ``.artifacts/lightning_h200_preflight.json`` snapshot so downstream
  audit tooling can read it the same way the HF launcher exposes its
  preflight on stdout.

Run with ``--dry-run`` to print the preflight, machine selection, and
resolved job command without contacting the Lightning control plane. Dry-runs
still require the same image and seed inputs as a launch so the preflight
matches the runnable production proof contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.hf_jobs.launch_production_gpu_proof import (  # noqa: E402
    DEFAULT_EQUILIBRIA_REL,
    DEFAULT_EXPECTED_JAX_VERSION,
    DEFAULT_STAGE2_SEED_REL,
    DEFAULT_TARGET_OPTIMIZER_BACKEND,
    TARGET_OPTIMIZER_BACKENDS,
    _build_preflight_report,
    _build_run_proof_argv,
    _resolve_repo_defaults,
    _validate_runtime_contract as _validate_hf_runtime_contract,
)


LIGHTNING_DEFAULT_CONNECTION = "simsopt-jax-parity-proofs"
LIGHTNING_CONTAINER_MOUNT_DIR = "/proof"
LIGHTNING_PROOF_ARTIFACT_PREFIX = "production-gpu-proof"
LIGHTNING_DEFAULT_MACHINE = "H200"
LIGHTNING_DEFAULT_CLOUD_PROVIDER = "nebius"
LIGHTNING_DEFAULT_MAX_RUNTIME_S = 8 * 60 * 60
LIGHTNING_DEFAULT_PREFLIGHT_PATH = (
    REPO_ROOT / ".artifacts" / "lightning_h200_preflight.json"
)
LIGHTNING_DEFAULT_ENTRYPOINT = "bash -lc"
LIGHTNING_JOB_POLL_INTERVAL_S = 30.0
LIGHTNING_JOB_POLL_TIMEOUT_S = 8 * 60 * 60.0
LIGHTNING_TERMINAL_FAILURE_STATUSES = {"Failed", "Stopped"}
LIGHTNING_TERMINAL_SUCCESS_STATUSES = {"Completed"}
DEFAULT_IMAGE = (
    os.environ.get("SIMSOPT_LIGHTNING_GPU_IMAGE")
    or os.environ.get("SIMSOPT_HF_GPU_IMAGE")
    or None
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _setuptools_scm_pretend_version(resolved_sha: str) -> str:
    if not resolved_sha:
        raise SystemExit("Resolved repo SHA is empty; cannot derive pretend version.")
    return f"0.0.0+proof.{resolved_sha[:8]}"


def _validate_lightning_runtime_contract(
    args: argparse.Namespace,
) -> argparse.Namespace:
    if not args.image:
        raise SystemExit(
            "Production GPU proof requires a prebuilt image via "
            "SIMSOPT_LIGHTNING_GPU_IMAGE, SIMSOPT_HF_GPU_IMAGE, or --image."
        )
    return _validate_hf_runtime_contract(args)


def _artifact_relative_path(resolved_sha: str, timestamp: str) -> str:
    return f"{LIGHTNING_PROOF_ARTIFACT_PREFIX}/{resolved_sha}/{timestamp}"


def _artifact_container_dir(resolved_sha: str, timestamp: str) -> str:
    relative = _artifact_relative_path(resolved_sha, timestamp)
    return f"{LIGHTNING_CONTAINER_MOUNT_DIR}/{relative}"


def _build_lightning_job_command(
    args: argparse.Namespace,
    *,
    resolved_repo_sha: str,
    results_dir: str,
) -> str:
    repo_dir = "/tmp/lightning-production-proof/repo"
    equilibria_dir = f"{repo_dir}/{DEFAULT_EQUILIBRIA_REL}"
    stage2_seed = f"{repo_dir}/{DEFAULT_STAGE2_SEED_REL}"
    pretend_version = _setuptools_scm_pretend_version(resolved_repo_sha)
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
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        'export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"',
        f'export SIMSOPT_HF_JOB_EXPECTED_JAX_VERSION="{DEFAULT_EXPECTED_JAX_VERSION}"',
        'export SIMSOPT_JAX_CUDA_LIBRARY_MODE="bundled"',
        f'export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="{pretend_version}"',
        "rm -rf /tmp/lightning-production-proof",
        "mkdir -p /tmp/lightning-production-proof",
        f"mkdir -p {shlex.quote(results_dir)}",
        git_clone,
        f"cd {shlex.quote(repo_dir)}",
        shlex.join(["git", "checkout", resolved_repo_sha]),
        "git submodule update --init --recursive",
        'export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PWD}/.artifacts/jax_compilation_cache/lightning-production-proof}"',
        'mkdir -p "$JAX_COMPILATION_CACHE_DIR"',
        ". benchmarks/hf_jobs/bootstrap_runtime.sh",
        "python -m pip install -v -e .",
        run_proof,
    ]
    return "\n".join(command_lines)


def _resolve_machine(machine_name: str) -> Any:
    try:
        from lightning_sdk import Machine
    except ImportError as exc:
        raise SystemExit(
            "lightning_sdk is required to launch Lightning jobs. "
            "Install with `pip install lightning-sdk`."
        ) from exc
    try:
        return getattr(Machine, machine_name)
    except AttributeError as exc:
        available = ", ".join(
            sorted(
                name
                for name in dir(Machine)
                if not name.startswith("_") and name.isupper()
            )
        )
        raise SystemExit(
            f"Unknown Lightning machine {machine_name!r}. Available: {available}"
        ) from exc


def _resolve_cloud_account(args: argparse.Namespace) -> str | None:
    if args.cloud_account:
        return args.cloud_account
    if not args.cloud_provider:
        return None
    return f"lightning-{args.cloud_provider}-prod"


def _resolve_cloud_provider(cloud_provider: str | None) -> Any | None:
    if cloud_provider is None:
        return None
    try:
        from lightning_sdk import CloudProvider
    except ImportError:
        return None
    try:
        return getattr(CloudProvider, cloud_provider.upper())
    except AttributeError:
        return None


def _build_lightning_preflight(
    args: argparse.Namespace,
    *,
    job_name: str,
    timestamp: str,
    machine: str,
    cloud_account: str | None,
    base_preflight: dict[str, object],
) -> dict[str, object]:
    resolved_sha = str(base_preflight["repo_sha"])
    preflight = dict(base_preflight)
    preflight.update(
        {
            "launcher": "lightning",
            "job_name": job_name,
            "hardware": [machine],
            "machine": machine,
            "cloud_provider": (args.cloud_provider or "").upper() or None,
            "cloud_account": cloud_account,
            "max_runtime_s": args.max_runtime_s,
            "artifact_connection": args.connection,
            "artifact_relative_path": _artifact_relative_path(resolved_sha, timestamp),
            "artifact_container_dir": _artifact_container_dir(resolved_sha, timestamp),
            "command_shell": LIGHTNING_DEFAULT_ENTRYPOINT,
            "setuptools_scm_pretend_version_for_simsopt": (
                _setuptools_scm_pretend_version(resolved_sha)
            ),
        }
    )
    return preflight


def _write_preflight_snapshot(preflight: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _submit_lightning_job(
    *,
    job_name: str,
    machine: Any,
    command: str,
    image: str,
    teamspace: str | None,
    cloud_account: str | None,
    cloud_provider: Any | None,
    env: dict[str, str],
    path_mappings: dict[str, str],
    max_runtime_s: int,
) -> Any:
    try:
        from lightning_sdk import Job
    except ImportError as exc:
        raise SystemExit(
            "lightning_sdk is required to launch Lightning jobs. "
            "Install with `pip install lightning-sdk`."
        ) from exc
    kwargs: dict[str, Any] = {
        "name": job_name,
        "machine": machine,
        "command": command,
        "image": image,
        "entrypoint": LIGHTNING_DEFAULT_ENTRYPOINT,
        "env": env,
        "path_mappings": path_mappings,
        "max_runtime": max_runtime_s,
    }
    if teamspace is not None:
        kwargs["teamspace"] = teamspace
    if cloud_account is not None:
        kwargs["cloud_account"] = cloud_account
    if cloud_provider is not None:
        kwargs["cloud_provider"] = cloud_provider
    return Job.run(**kwargs)


def _wait_for_terminal_status(
    job: Any,
    *,
    poll_interval_s: float,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    last_status: str | None = None
    while True:
        status = str(getattr(job, "status", "") or "").strip()
        if status and status != last_status:
            print(f"[lightning] status={status}", flush=True)
            last_status = status
        if status in LIGHTNING_TERMINAL_SUCCESS_STATUSES:
            return status
        if status in LIGHTNING_TERMINAL_FAILURE_STATUSES:
            raise SystemExit(
                f"Lightning job {job.name!r} finished with status {status}."
            )
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Lightning job {job.name!r} did not reach a terminal status "
                f"within {timeout_s:.0f}s; latest status={status or 'UNKNOWN'}."
            )
        time.sleep(poll_interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the production GPU proof on Lightning AI. Shares the "
            "proof command body with the HF Jobs launcher; differs only in "
            "data-connection mount, bash entrypoint, and setuptools-scm "
            "tag handling."
        )
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=(
            "Docker image to use. Defaults to SIMSOPT_LIGHTNING_GPU_IMAGE then "
            "SIMSOPT_HF_GPU_IMAGE when set."
        ),
    )
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--repo-sha", default=None)
    parser.add_argument("--repo-ref", default=None)
    parser.add_argument(
        "--hardware",
        default=LIGHTNING_DEFAULT_MACHINE,
        help=(
            "Lightning Machine enum name to launch (e.g. H200, H100, "
            "H200_X_8). Defaults to H200 (lit-h200x-1 on Nebius). One job "
            "per invocation; submit additional invocations for other "
            "hardware so each gets its own preflight snapshot."
        ),
    )
    parser.add_argument(
        "--cloud-provider",
        default=LIGHTNING_DEFAULT_CLOUD_PROVIDER,
        choices=("nebius", "aws", "gcp"),
        help=(
            "Cloud provider for the Lightning job (nebius, aws, gcp). Nebius "
            "exposes lit-h200x-1; AWS default exposes only lit-h200x-8."
        ),
    )
    parser.add_argument(
        "--cloud-account",
        default=None,
        help=(
            "Override the Lightning cloud account; defaults to "
            "lightning-<provider>-prod when --cloud-provider is set."
        ),
    )
    parser.add_argument(
        "--teamspace",
        default=None,
        help="Optional teamspace override; defaults to current teamspace.",
    )
    parser.add_argument(
        "--connection",
        default=LIGHTNING_DEFAULT_CONNECTION,
        help=(
            "Data connection name to mount at /proof for proof artifacts. "
            "Must already exist in the target teamspace."
        ),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Override the auto-generated Lightning job name.",
    )
    parser.add_argument(
        "--max-runtime-s",
        type=int,
        default=LIGHTNING_DEFAULT_MAX_RUNTIME_S,
        help="Job max-runtime in seconds. Default 8h.",
    )
    parser.add_argument("--platform", choices=("cuda",), default="cuda")
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
    )
    parser.add_argument(
        "--single-stage-disable-target-lane-success-filter",
        action="store_true",
    )
    parser.add_argument(
        "--preflight-path",
        type=Path,
        default=LIGHTNING_DEFAULT_PREFLIGHT_PATH,
        help=(
            "Where to write the preflight JSON snapshot. Default: "
            ".artifacts/lightning_h200_preflight.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the resolved preflight, machine, and job command without "
            "contacting the Lightning control plane."
        ),
    )
    parser.add_argument(
        "--no-detach",
        action="store_false",
        dest="detach",
        default=True,
        help=(
            "Poll Lightning job status until terminal and exit nonzero on "
            "failure (foreground). Default detaches after submission."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _validate_lightning_runtime_contract(_resolve_repo_defaults(parse_args()))
    base_preflight = _build_preflight_report(args)
    resolved_sha = str(base_preflight["repo_sha"])
    timestamp = _utc_timestamp()
    pretend_version = _setuptools_scm_pretend_version(resolved_sha)
    env = {
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "SIMSOPT_HF_JOB_EXPECTED_JAX_VERSION": DEFAULT_EXPECTED_JAX_VERSION,
        "SIMSOPT_JAX_CUDA_LIBRARY_MODE": "bundled",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT": pretend_version,
    }
    path_mappings = {LIGHTNING_CONTAINER_MOUNT_DIR: args.connection}
    cloud_account = _resolve_cloud_account(args)
    cloud_provider = _resolve_cloud_provider(args.cloud_provider)
    hardware = args.hardware
    job_name = args.job_name or (
        f"simsopt-jax-{hardware.lower().replace('_', '')}-proof-{timestamp.lower()}"
    )
    results_dir = _artifact_container_dir(resolved_sha, timestamp)
    command = _build_lightning_job_command(
        args,
        resolved_repo_sha=resolved_sha,
        results_dir=results_dir,
    )
    preflight = _build_lightning_preflight(
        args,
        job_name=job_name,
        timestamp=timestamp,
        machine=hardware,
        cloud_account=cloud_account,
        base_preflight=base_preflight,
    )
    _write_preflight_snapshot(preflight, args.preflight_path)
    print(json.dumps({"preflight": preflight}, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"[dry-run] command for {hardware}:\n{command}")
        return
    machine = _resolve_machine(hardware)
    job = _submit_lightning_job(
        job_name=job_name,
        machine=machine,
        command=command,
        image=args.image,
        teamspace=args.teamspace,
        cloud_account=cloud_account,
        cloud_provider=cloud_provider,
        env=env,
        path_mappings=path_mappings,
        max_runtime_s=args.max_runtime_s,
    )
    print(
        json.dumps(
            {
                "submitted": {
                    "job_name": job_name,
                    "machine": hardware,
                    "cloud_account": cloud_account,
                    "artifact_relative_path": _artifact_relative_path(
                        resolved_sha, timestamp
                    ),
                }
            },
            sort_keys=True,
        )
    )
    if args.detach:
        return
    _wait_for_terminal_status(
        job,
        poll_interval_s=LIGHTNING_JOB_POLL_INTERVAL_S,
        timeout_s=min(float(args.max_runtime_s), LIGHTNING_JOB_POLL_TIMEOUT_S),
    )


if __name__ == "__main__":
    main()
