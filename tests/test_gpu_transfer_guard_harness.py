from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import jax_gpu_failed_stale_tests_signoff as _SIGNOFF


REPO_ROOT = Path(__file__).resolve().parents[1]
STALE_SIGNOFF_SCRIPT = REPO_ROOT / "scripts" / "jax_gpu_failed_stale_tests_signoff.py"
GPU_PARITY_SCRIPT = REPO_ROOT / "scripts" / "run_gpu_parity.sh"
FULL_SUITE_TRANSFER_GUARD_ENV_VARS = (
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD_HOST_TO_DEVICE",
    "JAX_TRANSFER_GUARD_DEVICE_TO_HOST",
    "JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE",
)


def _run_stale_signoff_dry_run(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "JAX_TRANSFER_GUARD": "disallow",
        "JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE": "disallow",
        "JAX_TRANSFER_GUARD_DEVICE_TO_HOST": "disallow",
        "JAX_TRANSFER_GUARD_HOST_TO_DEVICE": "disallow",
    }
    return subprocess.run(
        [
            sys.executable,
            str(STALE_SIGNOFF_SCRIPT),
            "--repo",
            str(REPO_ROOT),
            "--dry-run",
            "--skip-clean-check",
            "--missing-path-policy=record",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_stale_signoff_dry_run_sanitizes_full_pytest_transfer_guard(tmp_path):
    result = _run_stale_signoff_dry_run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    policy = json.loads((tmp_path / "transfer_guard_env_policy.json").read_text())

    assert policy["full_pytest_env"] == {"SIMSOPT_JAX_TRANSFER_GUARD": "log"}
    assert policy["strict_probe_env"] == {
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
    }


@pytest.mark.parametrize("env_name", FULL_SUITE_TRANSFER_GUARD_ENV_VARS)
@pytest.mark.parametrize("guard_level", ("disallow", "disallow_explicit"))
def test_run_gpu_parity_rejects_full_suite_disallow_transfer_guard(
    env_name,
    guard_level,
):
    env = {
        **os.environ,
        "PLATFORM": "cuda",
        "REPO": str(REPO_ROOT),
        "PYTHON_BIN": sys.executable,
        "GPU_LOCK_ENABLED": "0",
        env_name: guard_level,
    }
    result = subprocess.run(
        ["bash", str(GPU_PARITY_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        f"full GPU parity suite uses transfer_guard=log; {env_name}={guard_level}"
        in result.stderr
    )


def test_stale_signoff_rejects_untracked_import_surface(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "sitecustomize.py").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="non-artifact untracked paths"):
        _SIGNOFF._require_clean_worktree(tmp_path)


def test_stale_signoff_allows_artifact_untracked_paths(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    artifact_path = tmp_path / ".artifacts" / "signoff" / "summary.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}\n", encoding="utf-8")

    _SIGNOFF._require_clean_worktree(tmp_path)
