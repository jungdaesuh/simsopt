from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

TEST_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_REPO_ROOT))

from benchmarks.hf_jobs import launch_production_gpu_proof as launcher
from benchmarks.validation_ladder_contract import build_stage2_hf_plan


REPO_ROOT = TEST_REPO_ROOT
RUNNER_SCRIPT = REPO_ROOT / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"
LADDER_CONTRACT = REPO_ROOT / "benchmarks" / "validation_ladder_contract.py"
LAUNCHER_SCRIPT = (
    REPO_ROOT / "benchmarks" / "hf_jobs" / "launch_production_gpu_proof.py"
)
FAKE_PROOF_SCRIPT = (
    REPO_ROOT / "tests" / "subprocess" / "hf_production_gpu_fake_runner.py"
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _copy_executable(source: Path, target: Path) -> None:
    shutil.copy2(source, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR)


def _read_call_records(call_log: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()
    ]


def _build_fake_proof_repo(
    tmp_path: Path,
    *,
    stage2_warm_mode: str = "ok",
) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    benchmarks_dir = repo_root / "benchmarks"
    hf_jobs_dir = benchmarks_dir / "hf_jobs"
    equilibria_dir = repo_root / "examples" / "single_stage_optimization" / "equilibria"
    hf_jobs_dir.mkdir(parents=True)
    equilibria_dir.mkdir(parents=True)
    shutil.copy2(RUNNER_SCRIPT, hf_jobs_dir / "run_production_gpu_proof.sh")
    shutil.copy2(LADDER_CONTRACT, benchmarks_dir / "validation_ladder_contract.py")
    call_log = repo_root / "call_log.jsonl"
    (repo_root / "fake_proof_config.json").write_text(
        json.dumps(
            {
                "call_log": str(call_log),
                "stage2_warm_mode": stage2_warm_mode,
            }
        ),
        encoding="utf-8",
    )
    _copy_executable(FAKE_PROOF_SCRIPT, benchmarks_dir / "stage2_e2e_comparison.py")
    _copy_executable(
        FAKE_PROOF_SCRIPT,
        benchmarks_dir / "single_stage_init_parity.py",
    )
    return repo_root, call_log


def test_run_production_gpu_proof_continues_after_missing_payload(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path, stage2_warm_mode="missing")
    results_dir = tmp_path / "results"
    stage2_seed = tmp_path / "stage2_seed.json"
    stage2_seed.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"),
            "--results-dir",
            str(results_dir),
            "--equilibria-dir",
            str(repo_root / "examples" / "single_stage_optimization" / "equilibria"),
            "--stage2-bs-path",
            str(stage2_seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HEARTBEAT_INTERVAL_S": "0.01"},
    )

    assert completed.returncode == 1
    assert "missing payload" in completed.stdout
    assert (results_dir / "single_stage_cold.json").is_file()
    assert (results_dir / "single_stage_warm.json").is_file()
    assert not (results_dir / "stage2_warm.json").exists()
    call_records = _read_call_records(call_log)
    stage2_calls = [
        record
        for record in call_records
        if record["output_json"].endswith("stage2_warm.json")
    ]
    assert len(stage2_calls) == 1
    assert "--geometry-rel-tol" not in stage2_calls[0]["argv"]


def test_run_production_gpu_proof_survives_corrupt_payload(tmp_path):
    repo_root, _ = _build_fake_proof_repo(tmp_path, stage2_warm_mode="corrupt")
    results_dir = tmp_path / "results"
    stage2_seed = tmp_path / "stage2_seed.json"
    stage2_seed.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"),
            "--results-dir",
            str(results_dir),
            "--equilibria-dir",
            str(repo_root / "examples" / "single_stage_optimization" / "equilibria"),
            "--stage2-bs-path",
            str(stage2_seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HEARTBEAT_INTERVAL_S": "0.01"},
    )

    assert completed.returncode == 1
    assert "corrupt payload" in completed.stdout
    assert '"corrupt_payload": true' in completed.stdout
    assert (results_dir / "single_stage_cold.json").is_file()
    assert (results_dir / "single_stage_warm.json").is_file()


def test_run_production_gpu_proof_adds_optional_repro_rung(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = tmp_path / "stage2_seed.json"
    stage2_seed.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"),
            "--results-dir",
            str(results_dir),
            "--equilibria-dir",
            str(repo_root / "examples" / "single_stage_optimization" / "equilibria"),
            "--stage2-bs-path",
            str(stage2_seed),
            "--stage2-maxiter",
            "60",
            "--geometry-rel-tol",
            "1e-6",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HEARTBEAT_INTERVAL_S": "0.01"},
    )

    assert completed.returncode == 0
    assert (results_dir / "stage2_warm_repro.json").is_file()
    call_records = _read_call_records(call_log)
    repro_calls = [
        record
        for record in call_records
        if record["output_json"].endswith("stage2_warm_repro.json")
    ]
    assert len(repro_calls) == 1
    assert "--geometry-rel-tol" in repro_calls[0]["argv"]


def test_run_production_gpu_proof_omits_boozer_override_by_default(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = tmp_path / "stage2_seed.json"
    stage2_seed.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"),
            "--results-dir",
            str(results_dir),
            "--equilibria-dir",
            str(repo_root / "examples" / "single_stage_optimization" / "equilibria"),
            "--stage2-bs-path",
            str(stage2_seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HEARTBEAT_INTERVAL_S": "0.01"},
    )

    assert completed.returncode == 0
    call_records = _read_call_records(call_log)
    single_stage_calls = [
        record
        for record in call_records
        if record["output_json"].endswith("single_stage_cold.json")
    ]
    assert len(single_stage_calls) == 1
    assert "--boozer-optimizer-backend" not in single_stage_calls[0]["argv"]


def test_run_production_gpu_proof_preserves_ld_library_path(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = tmp_path / "stage2_seed.json"
    stage2_seed.write_text("{}", encoding="utf-8")
    ld_library_path = "/cuda/lib:/driver/lib"

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"),
            "--results-dir",
            str(results_dir),
            "--equilibria-dir",
            str(repo_root / "examples" / "single_stage_optimization" / "equilibria"),
            "--stage2-bs-path",
            str(stage2_seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HEARTBEAT_INTERVAL_S": "0.01",
            "LD_LIBRARY_PATH": ld_library_path,
        },
    )

    assert completed.returncode == 0
    call_records = _read_call_records(call_log)
    assert call_records
    assert {
        record["ld_library_path"] for record in call_records
    } == {ld_library_path}
    assert {record["cuda_library_mode"] for record in call_records} == {"bundled"}


def test_build_stage2_hf_plan_keeps_smoke_jobs_geometry_report_only():
    plan = build_stage2_hf_plan(20, None)

    assert plan["stage2_rungs"] == ("stage2_cold", "stage2_warm")
    assert plan["geometry_rel_tol"] is None
    assert plan["effective_geometry_rel_tol"] is None
    assert plan["geometry_policy"] == "report-only"
    assert plan["supports_geometry_repro"] is False


def test_build_stage2_hf_plan_requires_long_run_for_explicit_geometry_repro():
    with pytest.raises(
        ValueError,
        match="Explicit --geometry-rel-tol conflicts with the maxiter=20 Stage 2 smoke contract",
    ):
        build_stage2_hf_plan(20, 1e-6)


def test_build_stage2_hf_plan_adds_repro_rung_for_long_run_override():
    plan = build_stage2_hf_plan(60, 1e-6)

    assert plan["stage2_rungs"] == (
        "stage2_cold",
        "stage2_warm",
        "stage2_warm_repro",
    )
    assert plan["geometry_rel_tol"] == pytest.approx(1e-6)
    assert plan["effective_geometry_rel_tol"] == pytest.approx(1e-6)
    assert plan["geometry_policy"] == "explicit-repro-gate"
    assert plan["supports_geometry_repro"] is True


def test_build_stage2_hf_plan_reports_default_long_run_geometry_gate():
    plan = build_stage2_hf_plan(60, None)

    assert plan["stage2_rungs"] == ("stage2_cold", "stage2_warm")
    assert plan["geometry_rel_tol"] is None
    assert plan["effective_geometry_rel_tol"] == pytest.approx(1e-6)
    assert plan["geometry_policy"] == "default-long-run-gate"
    assert plan["supports_geometry_repro"] is True


def _launcher_env(
    tmp_path: Path,
    *,
    image: str | None = "registry.example/simsopt-jax:cuda12-jax092",
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "hf",
        "#!/usr/bin/env bash\nprintf 'fake hf\\n'\n",
    )
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    if image is None:
        env.pop("SIMSOPT_HF_GPU_IMAGE", None)
    else:
        env["SIMSOPT_HF_GPU_IMAGE"] = image
    return env


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    return completed.stdout.strip()


def _build_remote_validation_repo(tmp_path: Path) -> tuple[str, str, str]:
    remote_repo = tmp_path / "remote.git"
    work_repo = tmp_path / "work"
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
    (work_repo / "proof.txt").write_text("main\n", encoding="utf-8")
    _git(work_repo, "add", "proof.txt")
    _git(work_repo, "commit", "-m", "main")
    _git(work_repo, "branch", "-M", "main")
    _git(work_repo, "remote", "add", "origin", str(remote_repo))
    _git(work_repo, "push", "-u", "origin", "main")
    main_sha = _git(work_repo, "rev-parse", "HEAD")
    _git(work_repo, "checkout", "-b", "feature")
    (work_repo / "proof.txt").write_text("feature\n", encoding="utf-8")
    _git(work_repo, "commit", "-am", "feature")
    feature_sha = _git(work_repo, "rev-parse", "HEAD")
    _git(work_repo, "push", "-u", "origin", "feature")
    return str(remote_repo), main_sha, feature_sha


def _default_remote_launcher_args(tmp_path: Path) -> list[str]:
    repo_url, main_sha, _ = _build_remote_validation_repo(tmp_path)
    return [
        "--repo-url",
        repo_url,
        "--repo-ref",
        "main",
        "--repo-sha",
        main_sha,
    ]


def test_launch_production_gpu_proof_help_works_without_site_packages():
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(LAUNCHER_SCRIPT),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert "Launch the production GPU proof on Hugging Face Jobs." in completed.stdout


def test_resolve_repo_defaults_prefers_current_branch_upstream_remote(monkeypatch):
    responses = {
        ("symbolic-ref", "--short", "HEAD"): "feature",
        ("config", "branch.feature.remote"): "collab",
        ("remote", "get-url", "collab"): "git@github.com:collab/simsopt.git",
    }

    monkeypatch.setattr(
        launcher,
        "_git_optional_output",
        lambda *args: responses.get(args),
    )
    monkeypatch.setattr(
        launcher,
        "_git_output",
        lambda *args: {
            ("rev-parse", "HEAD"): "deadbeef",
            ("rev-parse", "--abbrev-ref", "HEAD"): "feature",
        }[args],
    )

    args = launcher.argparse.Namespace(repo_url=None, repo_sha=None, repo_ref=None)
    resolved = launcher._resolve_repo_defaults(args)

    assert resolved.repo_url == "https://github.com/collab/simsopt.git"
    assert resolved.repo_sha == "deadbeef"
    assert resolved.repo_ref == "feature"


def test_resolve_default_repo_url_rejects_ambiguous_nonstandard_remotes(monkeypatch):
    responses = {
        ("symbolic-ref", "--short", "HEAD"): None,
        ("remote",): "alice\nbob",
    }

    monkeypatch.setattr(
        launcher,
        "_git_optional_output",
        lambda *args: responses.get(args),
    )

    with pytest.raises(SystemExit, match="Pass --repo-url explicitly"):
        launcher._resolve_default_repo_url()


def test_launch_production_gpu_proof_dry_run_omits_smoke_geometry_override(tmp_path):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert '"effective_geometry_rel_tol": null' in completed.stdout
    assert '"stage2_geometry_policy": "report-only"' in completed.stdout
    assert "--geometry-rel-tol" not in completed.stdout
    assert "--single-stage-boozer-optimizer-backend" not in completed.stdout
    assert "unset LD_LIBRARY_PATH" not in completed.stdout
    assert 'SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC="jax[cuda12]==0.9.2"' in completed.stdout
    assert 'SIMSOPT_JAX_CUDA_LIBRARY_MODE="bundled"' in completed.stdout
    assert "stage2_warm_repro" not in completed.stdout


def test_launch_production_gpu_proof_requires_explicit_image_or_env(tmp_path):
    env = _launcher_env(tmp_path, image=None)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "requires a prebuilt image via SIMSOPT_HF_GPU_IMAGE or --image" in completed.stderr


def test_launch_production_gpu_proof_rejects_fallback_image_without_always_bootstrap(
    tmp_path,
):
    env = _launcher_env(tmp_path, image=None)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--image",
            launcher.DEFAULT_FALLBACK_IMAGE,
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "requires --bootstrap-mode always" in completed.stderr


def test_launch_production_gpu_proof_allows_explicit_fallback_image_with_always_bootstrap(
    tmp_path,
):
    env = _launcher_env(tmp_path, image=None)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--image",
            launcher.DEFAULT_FALLBACK_IMAGE,
            "--bootstrap-mode",
            "always",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert launcher.DEFAULT_FALLBACK_IMAGE in completed.stdout


def test_launch_production_gpu_proof_reports_default_long_run_geometry_gate(tmp_path):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--stage2-maxiter",
            "60",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert '"effective_geometry_rel_tol": 1e-06' in completed.stdout
    assert '"stage2_geometry_policy": "default-long-run-gate"' in completed.stdout
    assert "--geometry-rel-tol" not in completed.stdout
    assert "stage2_warm_repro" not in completed.stdout


def test_launch_production_gpu_proof_rejects_smoke_geometry_override(tmp_path):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--stage2-maxiter",
            "20",
            "--geometry-rel-tol",
            "1e-6",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "maxiter=20 Stage 2 smoke contract" in completed.stderr


def test_launch_production_gpu_proof_rejects_remote_sha_not_on_repo_ref(tmp_path):
    env = _launcher_env(tmp_path)
    repo_url, _, feature_sha = _build_remote_validation_repo(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--repo-url",
            repo_url,
            "--repo-ref",
            "main",
            "--repo-sha",
            feature_sha,
            "--hardware",
            "a100-large",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert (
        "is not present on" in completed.stderr
        or "is not reachable from repo ref" in completed.stderr
    )


def test_launch_production_gpu_proof_accepts_matching_remote_repo_ref_and_sha(tmp_path):
    env = _launcher_env(tmp_path)
    repo_url, main_sha, _ = _build_remote_validation_repo(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--repo-url",
            repo_url,
            "--repo-ref",
            "main",
            "--repo-sha",
            main_sha,
            "--hardware",
            "a100-large",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert f'"repo_sha": "{main_sha}"' in completed.stdout


def test_launch_production_gpu_proof_allows_explicit_long_run_geometry_rung(tmp_path):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--stage2-maxiter",
            "60",
            "--geometry-rel-tol",
            "1e-6",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert '"stage2_rungs": [' in completed.stdout
    assert '"stage2_geometry_policy": "explicit-repro-gate"' in completed.stdout
    assert "stage2_warm_repro" in completed.stdout
    assert "--geometry-rel-tol 1e-06" in completed.stdout
