from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.modules.pop("scripts", None)

from scripts import jax_gpu_failed_stale_tests_signoff as _SIGNOFF


STALE_SIGNOFF_SCRIPT = REPO_ROOT / "scripts" / "jax_gpu_failed_stale_tests_signoff.py"
GPU_PARITY_SCRIPT = REPO_ROOT / "scripts" / "run_gpu_parity.sh"
FULL_SUITE_TRANSFER_GUARD_ENV_VARS = (
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD_HOST_TO_DEVICE",
    "JAX_TRANSFER_GUARD_DEVICE_TO_HOST",
    "JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE",
)


def _run_stale_signoff_dry_run(
    tmp_path: Path,
    *,
    missing_path_policy: str | None = "record",
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "JAX_TRANSFER_GUARD": "disallow",
        "JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE": "disallow",
        "JAX_TRANSFER_GUARD_DEVICE_TO_HOST": "disallow",
        "JAX_TRANSFER_GUARD_HOST_TO_DEVICE": "disallow",
    }
    command = [
        sys.executable,
        str(STALE_SIGNOFF_SCRIPT),
        "--repo",
        str(REPO_ROOT),
        "--dry-run",
        "--skip-clean-check",
        "--results-dir",
        str(tmp_path),
    ]
    if missing_path_policy is not None:
        command.append(f"--missing-path-policy={missing_path_policy}")
    return subprocess.run(
        command,
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


def test_stale_signoff_default_dry_run_inventory_is_current(tmp_path):
    result = _run_stale_signoff_dry_run(tmp_path, missing_path_policy=None)

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["missing_integration_paths"] == 0
    assert summary["focused_abort_repro_selectors_with_missing_paths"] == []
    assert summary["focused_lane_selectors_with_missing_paths"] == []
    assert Path(summary["batch_paths_dir"]).relative_to(REPO_ROOT) == Path(
        "tests/data/jax_gpu_signoff/batches"
    )
    assert "new_failed_selector_count" not in summary
    assert not (tmp_path / "failed_selector_comparison.json").exists()
    assert not (tmp_path / "failed_selector_comparison.md").exists()
    assert summary["failures"] == []


def test_stale_signoff_dry_run_isolates_focused_abort_repros(tmp_path):
    result = _run_stale_signoff_dry_run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    focused_selectors = summary["focused_abort_repro_selectors"]
    focused_lane_selectors = summary["focused_lane_selectors"]
    focused_selectors_with_missing_paths = summary[
        "focused_abort_repro_selectors_with_missing_paths"
    ]

    composite_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestCompositeGradientPipeline::"
        "test_composite_gradient_finite_and_nonzero"
    )
    runtime_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestTraceableObjective::"
        "test_runtime_bundle_batched_value_and_grad_matches_serial"
    )
    strict_cpu_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestNonQSRatioValue::test_dj_allows_strict_transfer_guard"
    )
    strict_gpu_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestCompositeObjective::"
        "test_public_wrapper_dj_boundaries_allow_strict_transfer_guard_real_fixture"
    )
    branch_stable_ondevice_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestRealFixtureOndeviceM5Parity::"
        "test_real_fixture_ondevice_branch_stable_wrapper_values_match"
    )
    short_outer_opt_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestShortSingleStageOptRun::"
        "test_outer_opt_accepts_stationary_initial_objective"
    )
    iotas_resolve_fd_selector = (
        "tests/integration/test_single_stage_jax_cpu_reference.py::"
        "TestIotasJAXResolveFD::test_iotas_resolve_fd"
    )
    expected_missing_selectors = [
        selector
        for selector in focused_selectors
        if not (REPO_ROOT / selector.split("::", maxsplit=1)[0]).exists()
    ]
    expected_present_selectors = set(focused_selectors) - set(
        expected_missing_selectors
    )

    assert composite_selector in focused_selectors
    assert runtime_selector in focused_selectors
    assert strict_cpu_selector in focused_lane_selectors
    assert strict_gpu_selector in focused_lane_selectors
    assert branch_stable_ondevice_selector in focused_lane_selectors
    assert short_outer_opt_selector in focused_lane_selectors
    assert iotas_resolve_fd_selector in focused_lane_selectors
    assert focused_selectors_with_missing_paths == expected_missing_selectors
    assert summary["focused_lane_selectors_with_missing_paths"] == []

    batch_log = (tmp_path / "integration_batches" / "batch_012.log").read_text()
    assert f"--deselect={composite_selector}" in batch_log
    assert f"--deselect={runtime_selector}" in batch_log
    assert f"--deselect={strict_cpu_selector}" in batch_log
    assert f"--deselect={strict_gpu_selector}" in batch_log
    assert f"--deselect={branch_stable_ondevice_selector}" in batch_log
    assert f"--deselect={short_outer_opt_selector}" in batch_log
    assert f"--deselect={iotas_resolve_fd_selector}" in batch_log

    batch_deselector_paths = sorted(
        (tmp_path / "integration_batches").glob(
            "batch_*_focused_selector_deselectors.txt"
        )
    )
    batch_deselectors = [
        selector
        for path in batch_deselector_paths
        for selector in path.read_text().splitlines()
    ]
    assert (
        len(batch_deselectors)
        == summary["integration_focused_selector_deselect_count"]
    )
    assert composite_selector in batch_deselectors
    assert runtime_selector in batch_deselectors
    assert strict_cpu_selector in batch_deselectors
    assert strict_gpu_selector in batch_deselectors
    assert branch_stable_ondevice_selector in batch_deselectors
    assert short_outer_opt_selector in batch_deselectors
    assert iotas_resolve_fd_selector in batch_deselectors
    expected_present_lane_selectors = {
        selector
        for selector in focused_lane_selectors
        if (REPO_ROOT / selector.split("::", maxsplit=1)[0]).exists()
    }
    repro_deselectors = [
        selector for selector in batch_deselectors if selector in expected_present_selectors
    ]
    lane_deselectors = [
        selector
        for selector in batch_deselectors
        if selector in expected_present_lane_selectors
    ]
    assert len(repro_deselectors) == summary["integration_focused_repro_deselect_count"]
    assert len(lane_deselectors) == summary["integration_focused_lane_deselect_count"]
    assert len(batch_deselectors) == len(repro_deselectors) + len(lane_deselectors)
    assert set(batch_deselectors) <= (
        expected_present_selectors | expected_present_lane_selectors
    )

    focused_dir = tmp_path / "focused_selectors"
    assert (
        focused_dir / "batch_012_composite_gradient_finite_and_nonzero.log"
    ).is_file()
    assert (
        focused_dir / "batch_012_runtime_bundle_batched_value_and_grad.log"
    ).is_file()
    assert (
        focused_dir / "batch_012_strict_cpu_non_qs_ratio_dj_transfer_guard.log"
    ).is_file()
    assert (
        focused_dir / "batch_012_strict_gpu_public_wrapper_dj_transfer_guard.log"
    ).is_file()
    assert (
        focused_dir / "batch_012_branch_stable_ondevice_m5_values.log"
    ).is_file()
    assert (
        focused_dir / "batch_012_short_single_stage_stationary_outer_opt.log"
    ).is_file()
    assert (focused_dir / "batch_012_iotas_resolve_fd.log").is_file()


def test_stale_signoff_dry_run_cleans_stale_focused_deselector_sidecars(tmp_path):
    result = _run_stale_signoff_dry_run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    stale_sidecar = (
        tmp_path
        / "integration_batches"
        / "batch_001_focused_selector_deselectors.txt"
    )
    stale_sidecar.write_text("stale-selector\n")
    stale_focused_outputs = (
        tmp_path / "focused_selectors" / "batch_999_old.xml",
        tmp_path / "focused_selectors" / "batch_999_old.log",
        tmp_path / "focused_selectors" / "batch_999_old.log.rc",
    )
    stale_root_outputs = (
        tmp_path / "focused_abort_repro_missing_selectors.txt",
        tmp_path / "focused_lane_missing_selectors.txt",
    )
    for path in stale_focused_outputs:
        path.write_text("stale\n")
    for path in stale_root_outputs:
        path.write_text("stale\n")

    rerun = _run_stale_signoff_dry_run(tmp_path)

    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert not stale_sidecar.exists()
    assert all(not path.exists() for path in stale_focused_outputs)
    assert all(not path.exists() for path in stale_root_outputs)


def test_stale_signoff_classifies_shell_and_subprocess_sigabrt_codes():
    assert _SIGNOFF._is_sigabrt_returncode(-6)
    assert _SIGNOFF._is_sigabrt_returncode(134)
    assert not _SIGNOFF._is_sigabrt_returncode(1)


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
