"""Tests for the Lightning AI production GPU proof launcher.

The Lightning launcher shares its proof command body with the HF Jobs
launcher (:mod:`benchmarks.hf_jobs.launch_production_gpu_proof`); these
tests pin the Lightning-specific contract:

* artifacts mount inside ``simsopt-jax-parity-proofs`` at ``/proof`` and
  the run-specific subdirectory is created inside the job;
* ``entrypoint="bash -lc"`` because Lightning's default shell rejects
  ``set -o pipefail``;
* ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT`` is exported before
  ``pip install -e .`` so non-PEP440 release tags cannot break the
  install step;
* preflight JSON snapshot is durable on disk at the path the audit doc
  references.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TEST_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_REPO_ROOT))

from benchmarks.hf_jobs import launch_production_gpu_proof as hf_launcher
from benchmarks.lightning_jobs import launch_production_gpu_proof as launcher


REPO_ROOT = TEST_REPO_ROOT
LIGHTNING_LAUNCHER_SCRIPT = (
    REPO_ROOT / "benchmarks" / "lightning_jobs" / "launch_production_gpu_proof.py"
)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    return completed.stdout.strip()


def _build_remote_validation_repo(tmp_path: Path) -> tuple[str, str]:
    remote_repo = tmp_path / "remote.git"
    work_repo = tmp_path / "work"
    single_stage_seed = (
        work_repo / "benchmarks" / "fixtures" / "single_stage_seed_iota15"
    )
    subprocess.run(
        ["git", "init", "--bare", str(remote_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "init", str(work_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(work_repo, "config", "user.email", "test@example.com")
    _git(work_repo, "config", "user.name", "test")
    single_stage_seed.mkdir(parents=True)
    (work_repo / "proof.txt").write_text("main\n", encoding="utf-8")
    (single_stage_seed / "surf_opt.json").write_text("{}", encoding="utf-8")
    (single_stage_seed / "results.json").write_text("{}", encoding="utf-8")
    (single_stage_seed / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
    _git(work_repo, "add", "proof.txt")
    _git(work_repo, "add", "benchmarks/fixtures/single_stage_seed_iota15")
    _git(work_repo, "commit", "-m", "main")
    _git(work_repo, "branch", "-M", "main")
    _git(work_repo, "remote", "add", "origin", str(remote_repo))
    _git(work_repo, "push", "-u", "origin", "main")
    main_sha = _git(work_repo, "rev-parse", "HEAD")
    return str(remote_repo), main_sha


def _default_launcher_args(tmp_path: Path) -> list[str]:
    repo_url, main_sha = _build_remote_validation_repo(tmp_path)
    return [
        "--repo-url",
        repo_url,
        "--repo-ref",
        "main",
        "--repo-sha",
        main_sha,
        "--single-stage-warm-start-run-dir",
        "benchmarks/fixtures/single_stage_seed_iota15",
    ]


def _launcher_env(
    tmp_path: Path,
    *,
    image: str | None = "registry.example/simsopt-jax:cuda12-jax092",
) -> dict[str, str]:
    env = dict(os.environ)
    if image is None:
        env.pop("SIMSOPT_HF_GPU_IMAGE", None)
        env.pop("SIMSOPT_LIGHTNING_GPU_IMAGE", None)
    else:
        env["SIMSOPT_LIGHTNING_GPU_IMAGE"] = image
    return env


def test_launcher_help_works_without_site_packages():
    completed = subprocess.run(
        [sys.executable, "-S", str(LIGHTNING_LAUNCHER_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert "Launch the production GPU proof on Lightning AI" in completed.stdout


def test_setuptools_scm_pretend_version_is_deterministic_from_sha():
    assert (
        launcher._setuptools_scm_pretend_version("abcdef1234567890")
        == "0.0.0+proof.abcdef12"
    )


def test_setuptools_scm_pretend_version_rejects_empty_sha():
    with pytest.raises(SystemExit, match="Resolved repo SHA is empty"):
        launcher._setuptools_scm_pretend_version("")


def test_artifact_paths_use_connection_root_and_run_specific_subdir():
    relative = launcher._artifact_relative_path("deadbeef", "20260512T220000Z")
    container = launcher._artifact_container_dir("deadbeef", "20260512T220000Z")

    assert relative == "production-gpu-proof/deadbeef/20260512T220000Z"
    assert container == "/proof/production-gpu-proof/deadbeef/20260512T220000Z"


def test_resolve_cloud_account_prefers_explicit_override():
    args = argparse.Namespace(cloud_account="custom-account", cloud_provider="nebius")
    assert launcher._resolve_cloud_account(args) == "custom-account"


def test_resolve_cloud_account_defaults_to_provider_prod_account():
    args = argparse.Namespace(cloud_account=None, cloud_provider="nebius")
    assert launcher._resolve_cloud_account(args) == "lightning-nebius-prod"


def test_resolve_cloud_account_passes_through_none_without_provider():
    args = argparse.Namespace(cloud_account=None, cloud_provider=None)
    assert launcher._resolve_cloud_account(args) is None


def test_dry_run_emits_preflight_with_lightning_specific_fields(tmp_path):
    env = _launcher_env(tmp_path)
    launcher_args = _default_launcher_args(tmp_path)
    preflight_path = tmp_path / "lightning_h200_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(LIGHTNING_LAUNCHER_SCRIPT),
            "--dry-run",
            "--preflight-path",
            str(preflight_path),
            *launcher_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert payload["launcher"] == "lightning"
    assert payload["machine"] == launcher.LIGHTNING_DEFAULT_MACHINE
    assert payload["hardware"] == [launcher.LIGHTNING_DEFAULT_MACHINE]
    assert payload["cloud_provider"] == "NEBIUS"
    assert payload["cloud_account"] == "lightning-nebius-prod"
    assert payload["artifact_connection"] == launcher.LIGHTNING_DEFAULT_CONNECTION
    assert payload["artifact_relative_path"].startswith(
        f"{launcher.LIGHTNING_PROOF_ARTIFACT_PREFIX}/{payload['repo_sha']}/"
    )
    assert payload["artifact_container_dir"].startswith(
        f"{launcher.LIGHTNING_CONTAINER_MOUNT_DIR}/"
        f"{launcher.LIGHTNING_PROOF_ARTIFACT_PREFIX}/{payload['repo_sha']}/"
    )
    assert payload["command_shell"] == launcher.LIGHTNING_DEFAULT_ENTRYPOINT
    assert payload["setuptools_scm_pretend_version_for_simsopt"].startswith(
        "0.0.0+proof."
    )


def test_dry_run_emits_command_with_setuptools_scm_and_bash_safeguards(tmp_path):
    env = _launcher_env(tmp_path)
    launcher_args = _default_launcher_args(tmp_path)
    preflight_path = tmp_path / "lightning_h200_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(LIGHTNING_LAUNCHER_SCRIPT),
            "--dry-run",
            "--preflight-path",
            str(preflight_path),
            *launcher_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    assert "set -euxo pipefail" in stdout
    assert 'export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="0.0.0+proof.' in stdout
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in stdout
    assert (
        'export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"' in stdout
    )
    assert ". benchmarks/hf_jobs/bootstrap_runtime.sh" in stdout
    assert "python -m pip install -v -e ." in stdout
    assert "mkdir -p /proof/production-gpu-proof/" in stdout
    assert "bash benchmarks/hf_jobs/run_production_gpu_proof.sh" in stdout
    assert "--results-dir /proof/production-gpu-proof/" in stdout


def test_dry_run_requires_image_for_runnable_preflight_contract(tmp_path):
    env = _launcher_env(tmp_path, image=None)
    launcher_args = _default_launcher_args(tmp_path)
    preflight_path = tmp_path / "lightning_h200_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(LIGHTNING_LAUNCHER_SCRIPT),
            "--dry-run",
            "--preflight-path",
            str(preflight_path),
            *launcher_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "Production GPU proof requires a prebuilt image" in output
    assert "SIMSOPT_LIGHTNING_GPU_IMAGE" in output
    assert "SIMSOPT_HF_GPU_IMAGE" in output
    assert "--image" in output


def test_cloud_provider_rejects_unknown_provider(tmp_path):
    env = _launcher_env(tmp_path)
    launcher_args = _default_launcher_args(tmp_path)
    preflight_path = tmp_path / "lightning_h200_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(LIGHTNING_LAUNCHER_SCRIPT),
            "--dry-run",
            "--cloud-provider",
            "nebuis",
            "--preflight-path",
            str(preflight_path),
            *launcher_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr


def test_proof_command_shares_run_proof_argv_with_hf_launcher(tmp_path):
    """The two launchers must produce byte-identical run_production_gpu_proof.sh argv.

    Different launchers, same proof contract. If this diverges, downstream
    parity reports will silently disagree depending on which launcher
    produced them.
    """

    repo_dir = "/tmp/lightning-production-proof/repo"
    args = argparse.Namespace(
        platform="cuda",
        stage2_optimizer_backend=launcher.DEFAULT_TARGET_OPTIMIZER_BACKEND,
        stage2_nphi=255,
        stage2_ntheta=64,
        stage2_maxiter=20,
        geometry_rel_tol=None,
        single_stage_optimizer_backend=launcher.DEFAULT_TARGET_OPTIMIZER_BACKEND,
        single_stage_boozer_optimizer_backend=None,
        single_stage_nphi=255,
        single_stage_ntheta=64,
        single_stage_mpol=8,
        single_stage_ntor=6,
        single_stage_maxiter=300,
        single_stage_warm_start_run_dir=None,
        single_stage_jax_runtime_seed_spec=(
            "benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json"
        ),
        single_stage_benchmark_mode=False,
        single_stage_disable_target_lane_success_filter=False,
    )
    results_dir = "/proof/production-gpu-proof/deadbeef/20260512T220000Z"
    equilibria_dir = f"{repo_dir}/{launcher.DEFAULT_EQUILIBRIA_REL}"
    stage2_seed = f"{repo_dir}/{launcher.DEFAULT_STAGE2_SEED_REL}"

    argv = hf_launcher._build_run_proof_argv(
        args,
        repo_dir=repo_dir,
        results_dir=results_dir,
        equilibria_dir=equilibria_dir,
        stage2_seed=stage2_seed,
    )

    assert argv[0] == "bash"
    assert argv[1] == "benchmarks/hf_jobs/run_production_gpu_proof.sh"
    assert "--results-dir" in argv
    assert results_dir in argv
    assert "--stage2-platform" in argv
    cuda_indices = [i for i, item in enumerate(argv) if item == "cuda"]
    assert len(cuda_indices) >= 2


def test_main_module_imports_without_lightning_sdk():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import benchmarks.lightning_jobs.launch_production_gpu_proof as L; "
            "print(L.LIGHTNING_DEFAULT_MACHINE)",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "H200"


def test_terminal_failure_statuses_match_lightning_sdk_enum():
    """The poll loop's terminal-failure set must only contain values the SDK actually emits.

    Earlier versions of this module listed "Cancelled" and "Crashed", which are
    not members of ``lightning_sdk.Status``. A job that finishes outside the
    expected vocabulary would otherwise be classified as transitional and the
    poller would block until the 8h timeout.

    Gated on ``LIGHTNING_SDK_AVAILABLE=1``: set this env var in lightning-capable
    CI jobs so SDK enum drift is caught. If the env var is set but
    ``lightning_sdk`` cannot be imported, the test fails hard — that signals a
    CI configuration regression rather than an environment without the SDK.
    """

    if not os.environ.get("LIGHTNING_SDK_AVAILABLE"):
        pytest.skip(
            "set LIGHTNING_SDK_AVAILABLE=1 in lightning-capable CI to enforce "
            "lightning_sdk.Status enum drift detection"
        )
    # ImportError below is a CI bug: env var set but module missing.
    from lightning_sdk import Status

    sdk_status_names = {member.name for member in Status}
    assert launcher.LIGHTNING_TERMINAL_FAILURE_STATUSES <= sdk_status_names, (
        "LIGHTNING_TERMINAL_FAILURE_STATUSES contains names not present in "
        f"lightning_sdk.Status: extras="
        f"{launcher.LIGHTNING_TERMINAL_FAILURE_STATUSES - sdk_status_names}"
    )
    assert launcher.LIGHTNING_TERMINAL_SUCCESS_STATUSES <= sdk_status_names
    # "Completed" is the only terminal-success state; "Failed"/"Stopped" are
    # the terminal-failure states. Other members are transitional.
    assert launcher.LIGHTNING_TERMINAL_SUCCESS_STATUSES == {"Completed"}
    assert launcher.LIGHTNING_TERMINAL_FAILURE_STATUSES == {"Failed", "Stopped"}


def test_hardware_argument_is_single_valued(tmp_path):
    """One launcher invocation submits one job; the preflight snapshot file
    therefore cannot be silently overwritten by a multi-hardware loop."""

    env = _launcher_env(tmp_path)
    launcher_args = _default_launcher_args(tmp_path)
    preflight_path = tmp_path / "lightning_h200_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(LIGHTNING_LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "H200",
            "H100",
            "--preflight-path",
            str(preflight_path),
            *launcher_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "unrecognized arguments" in combined or "error" in combined.lower()


def test_dry_run_writes_exactly_one_preflight_snapshot(tmp_path):
    env = _launcher_env(tmp_path)
    launcher_args = _default_launcher_args(tmp_path)
    preflight_path = tmp_path / "lightning_h200_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(LIGHTNING_LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "H100",
            "--preflight-path",
            str(preflight_path),
            *launcher_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert payload["machine"] == "H100"
    assert payload["hardware"] == ["H100"]
    assert payload["job_name"].startswith("simsopt-jax-h100-proof-")
