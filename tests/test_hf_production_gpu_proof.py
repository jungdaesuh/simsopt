from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Sequence

import pytest

TEST_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_REPO_ROOT))

from benchmarks.hf_jobs import launch_production_gpu_proof as launcher
from benchmarks.validation_ladder_contract import build_stage2_hf_plan


REPO_ROOT = TEST_REPO_ROOT
RUNNER_SCRIPT = REPO_ROOT / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"
BOOTSTRAP_SCRIPT = REPO_ROOT / "benchmarks" / "hf_jobs" / "bootstrap_runtime.sh"
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


def _call_records_for_output(
    call_log: Path,
    output_name: str,
) -> list[dict[str, object]]:
    return [
        record
        for record in _read_call_records(call_log)
        if str(record["output_json"]).endswith(output_name)
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
    _copy_executable(
        FAKE_PROOF_SCRIPT,
        hf_jobs_dir / "cuda_pytest_probe.py",
    )
    return repo_root, call_log


def _single_stage_seed_args(tmp_path: Path) -> list[str]:
    seed_spec = tmp_path / "single_stage_seed_spec.json"
    seed_spec.write_text("{}", encoding="utf-8")
    return ["--single-stage-jax-runtime-seed-spec", str(seed_spec)]


def _write_stage2_seed(tmp_path: Path) -> Path:
    stage2_seed = tmp_path / "stage2_seed.json"
    stage2_seed.write_text("{}", encoding="utf-8")
    return stage2_seed


def _fake_runner_env(**extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "HEARTBEAT_INTERVAL_S": "0.01",
        "SIMSOPT_FAKE_GPU": "1",
        **extra,
    }


def _run_production_gpu_proof(
    repo_root: Path,
    results_dir: Path,
    stage2_seed: Path,
    extra_args: Sequence[str] = (),
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(repo_root / "benchmarks" / "hf_jobs" / "run_production_gpu_proof.sh"),
            "--results-dir",
            str(results_dir),
            "--equilibria-dir",
            str(repo_root / "examples" / "single_stage_optimization" / "equilibria"),
            "--stage2-bs-path",
            str(stage2_seed),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_fake_runner_env() if env is None else env,
    )


def test_bootstrap_runtime_writes_gpu_smoke_payload_and_fails_closed():
    script = BOOTSTRAP_SCRIPT.read_text(encoding="utf-8")

    assert "BOOTSTRAP_JAX_SMOKE_JSON" in script
    assert "Production GPU proof requires a prebuilt runtime" in script
    assert "\"default_backend\": default_backend" in script
    assert "\"devices\": [str(device) for device in jax.devices()]" in script
    assert "Expected GPU JAX backend during HF proof bootstrap" in script
    assert "apt-get" not in script
    assert "pip install" not in script
    assert "BOOTSTRAP_MODE" not in script


def test_run_production_gpu_proof_has_ptx_and_cubin_cuda_canaries():
    script = RUNNER_SCRIPT.read_text(encoding="utf-8")

    expected_snippets = (
        "run_cuda_canary ptx 1 0",
        "run_cuda_canary cubin 0 1",
        "CUDA_FORCE_PTX_JIT",
        "CUDA_DISABLE_PTX_JIT",
        "CUDA_CACHE_DISABLE=1",
        "CUDA_CACHE_PATH=\"${cuda_cache_dir}\"",
        "cuda_driver_cache_${mode}",
        "JAX_COMPILATION_CACHE_DIR=\"${canary_cache_dir}\"",
        "cuda_canary_cache_${mode}",
        "class CanaryPayload(TypedDict):",
        "def canary_kernel(x: jax.Array) -> jax.Array:",
        "payload: CanaryPayload",
        "value.block_until_ready()",
        "CUDA canary",
    )
    for snippet in expected_snippets:
        assert snippet in script


def test_run_production_gpu_proof_requires_single_stage_seed(tmp_path):
    repo_root, _ = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(repo_root, results_dir, stage2_seed)

    assert completed.returncode == 1
    assert "requires --single-stage-warm-start-run-dir" in completed.stderr
    assert not results_dir.exists()


def test_run_production_gpu_proof_continues_after_missing_payload(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path, stage2_warm_mode="missing")
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 1
    assert "missing payload" in completed.stdout
    assert (results_dir / "single_stage_cold.json").is_file()
    assert (results_dir / "single_stage_warm.json").is_file()
    assert not (results_dir / "stage2_warm.json").exists()
    stage2_calls = _call_records_for_output(call_log, "stage2_warm.json")
    assert len(stage2_calls) == 1
    assert "--geometry-rel-tol" not in stage2_calls[0]["argv"]


def test_run_production_gpu_proof_survives_corrupt_payload(tmp_path):
    repo_root, _ = _build_fake_proof_repo(tmp_path, stage2_warm_mode="corrupt")
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 1
    assert "corrupt payload" in completed.stdout
    assert '"corrupt_payload": true' in completed.stdout
    assert (results_dir / "single_stage_cold.json").is_file()
    assert (results_dir / "single_stage_warm.json").is_file()


def test_run_production_gpu_proof_adds_optional_repro_rung(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        [
            "--stage2-maxiter",
            "60",
            "--geometry-rel-tol",
            "1e-6",
            *_single_stage_seed_args(tmp_path),
        ],
    )

    assert completed.returncode == 0
    assert (results_dir / "stage2_warm_repro.json").is_file()
    repro_calls = _call_records_for_output(call_log, "stage2_warm_repro.json")
    assert len(repro_calls) == 1
    assert "--geometry-rel-tol" in repro_calls[0]["argv"]


def test_run_production_gpu_proof_omits_boozer_override_by_default(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 0
    single_stage_calls = _call_records_for_output(call_log, "single_stage_cold.json")
    assert len(single_stage_calls) == 1
    assert "--boozer-optimizer-backend" not in single_stage_calls[0]["argv"]


def test_run_production_gpu_proof_rejects_cpu_runtime_without_fake_gpu(tmp_path):
    repo_root, _ = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)
    env = _fake_runner_env()
    env.pop("SIMSOPT_FAKE_GPU", None)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
        env=env,
    )

    assert completed.returncode == 1
    assert "CUDA canary ptx expected GPU backend" in completed.stderr


def test_run_production_gpu_proof_rejects_non_cuda_platform_without_fake_gpu(tmp_path):
    repo_root, _ = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)
    env = _fake_runner_env()
    env.pop("SIMSOPT_FAKE_GPU", None)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        [
            "--stage2-platform",
            "cpu",
            "--single-stage-platform",
            "cuda",
            *_single_stage_seed_args(tmp_path),
        ],
        env=env,
    )

    assert completed.returncode == 1
    assert (
        "Production GPU proof requires --stage2-platform cuda and "
        "--single-stage-platform cuda"
    ) in completed.stderr


def test_run_production_gpu_proof_preserves_explicit_proof_parity_schema(tmp_path):
    repo_root, _ = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 0
    assert '"proof_parity"' in completed.stdout
    proof_parity = json.loads(
        (results_dir / "stage2_cold.json").read_text(encoding="utf-8")
    )["proof_parity"]
    assert {
        "cpu_oracle_value",
        "gpu_value",
        "value_rtol",
        "gradient_rtol",
        "value_lane",
        "gradient_lane",
    } <= set(proof_parity)
    boozer_parity = json.loads(
        (results_dir / "boozer_well_conditioned_adjoint.json").read_text(
            encoding="utf-8"
        )
    )["proof_parity"]
    reduction_parity = json.loads(
        (results_dir / "reduction_cancellation_stress.json").read_text(
            encoding="utf-8"
        )
    )["proof_parity"]
    assert boozer_parity["lane"] == "exact_well_conditioned_adjoint"
    assert reduction_parity["lane"] == "reduction_cpu_gpu"


def test_run_production_gpu_proof_rejects_proof_rtol_above_ladder(tmp_path):
    repo_root, _ = _build_fake_proof_repo(
        tmp_path,
        stage2_warm_mode="invalid_proof_rtol",
    )
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 1
    assert "proof_parity.value_rtol exceeds ladder contract" in completed.stdout


def test_run_production_gpu_proof_rejects_stale_value_rel_diff(tmp_path):
    repo_root, _ = _build_fake_proof_repo(
        tmp_path,
        stage2_warm_mode="invalid_value_rel_diff",
    )
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 1
    assert "proof_parity.value_rel_diff does not match CPU/GPU values" in completed.stdout


def test_run_production_gpu_proof_rejects_gradient_rel_diff_above_ladder(tmp_path):
    repo_root, _ = _build_fake_proof_repo(
        tmp_path,
        stage2_warm_mode="invalid_gradient_rel_diff",
    )
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
    )

    assert completed.returncode == 1
    assert "proof_parity.gradient_rel_diff" in completed.stdout
    assert "exceeds gradient_rtol" in completed.stdout


def test_run_production_gpu_proof_threads_single_stage_seed_contract(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)
    warm_start_run_dir = tmp_path / "single_stage_seed"
    jax_runtime_seed_spec = tmp_path / "single_stage_seed_spec.json"

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        [
            "--single-stage-warm-start-run-dir",
            str(warm_start_run_dir),
            "--single-stage-jax-runtime-seed-spec",
            str(jax_runtime_seed_spec),
        ],
    )

    assert completed.returncode == 0
    single_stage_calls = _call_records_for_output(call_log, "single_stage_cold.json")
    assert len(single_stage_calls) == 1
    assert "--warm-start-run-dir" in single_stage_calls[0]["argv"]
    assert str(warm_start_run_dir) in single_stage_calls[0]["argv"]
    assert "--jax-runtime-seed-spec" in single_stage_calls[0]["argv"]
    assert str(jax_runtime_seed_spec) in single_stage_calls[0]["argv"]
    assert "--case-artifacts-dir" in single_stage_calls[0]["argv"]
    artifacts_flag_index = single_stage_calls[0]["argv"].index("--case-artifacts-dir")
    assert (
        single_stage_calls[0]["argv"][artifacts_flag_index + 1]
        == str(results_dir / "artifacts" / "single_stage_cold")
    )


def test_run_production_gpu_proof_threads_single_stage_benchmark_mode(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        [
            "--single-stage-benchmark-mode",
            *_single_stage_seed_args(tmp_path),
        ],
    )

    assert completed.returncode == 0
    single_stage_calls = _call_records_for_output(call_log, "single_stage_cold.json")
    assert len(single_stage_calls) == 1
    assert "--benchmark-mode" in single_stage_calls[0]["argv"]


def test_run_production_gpu_proof_threads_single_stage_success_filter_bypass(
    tmp_path,
):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        [
            "--single-stage-disable-target-lane-success-filter",
            *_single_stage_seed_args(tmp_path),
        ],
    )

    assert completed.returncode == 0
    single_stage_calls = _call_records_for_output(call_log, "single_stage_cold.json")
    assert len(single_stage_calls) == 1
    assert "--disable-target-lane-success-filter" in single_stage_calls[0]["argv"]


def test_run_production_gpu_proof_preserves_ld_library_path(tmp_path):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)
    ld_library_path = "/cuda/lib:/driver/lib"

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
        env=_fake_runner_env(LD_LIBRARY_PATH=ld_library_path),
    )

    assert completed.returncode == 0
    call_records = _read_call_records(call_log)
    assert call_records
    assert {
        record["ld_library_path"] for record in call_records
    } == {ld_library_path}
    assert {record["cuda_library_mode"] for record in call_records} == {"bundled"}
    assert all(
        "--xla_gpu_deterministic_ops=true" in str(record["xla_flags"]).split()
        for record in call_records
    )


def test_run_production_gpu_proof_uses_repo_artifact_compilation_cache_by_default(
    tmp_path,
):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)
    env = _fake_runner_env()
    env.pop("JAX_COMPILATION_CACHE_DIR", None)

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
        env=env,
    )

    assert completed.returncode == 0
    expected_cache_dir = (
        repo_root / ".artifacts" / "jax_compilation_cache" / "hf-production-proof"
    )
    call_records = _read_call_records(call_log)
    assert {record["jax_compilation_cache_dir"] for record in call_records} == {
        str(expected_cache_dir)
    }
    assert expected_cache_dir.is_dir()


def test_run_production_gpu_proof_enforces_xla_python_client_preallocate_false(
    tmp_path,
):
    repo_root, call_log = _build_fake_proof_repo(tmp_path)
    results_dir = tmp_path / "results"
    stage2_seed = _write_stage2_seed(tmp_path)
    env = _fake_runner_env(XLA_PYTHON_CLIENT_PREALLOCATE="true")

    completed = _run_production_gpu_proof(
        repo_root,
        results_dir,
        stage2_seed,
        _single_stage_seed_args(tmp_path),
        env=env,
    )

    assert completed.returncode == 0
    call_records = _read_call_records(call_log)
    assert {record["xla_python_client_preallocate"] for record in call_records} == {
        "false"
    }


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
    return _build_remote_validation_repo_with_seed(tmp_path)


def _build_remote_validation_repo_with_seed(
    tmp_path: Path,
    *,
    include_stage2_seed: bool = True,
    include_runtime_seed_spec: bool = False,
    runtime_spec_tree: bool = False,
) -> tuple[str, str, str]:
    remote_repo = tmp_path / "remote.git"
    work_repo = tmp_path / "work"
    single_stage_seed = work_repo / "benchmarks" / "fixtures" / "single_stage_seed_iota15"
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
    if include_stage2_seed:
        (single_stage_seed / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
    if include_runtime_seed_spec:
        (single_stage_seed / "single_stage_jax_runtime_spec.json").write_text(
            "{}",
            encoding="utf-8",
        )
    if runtime_spec_tree:
        runtime_spec_tree_path = single_stage_seed / "single_stage_jax_runtime_spec.json"
        runtime_spec_tree_path.mkdir()
        (runtime_spec_tree_path / "placeholder.txt").write_text("", encoding="utf-8")
    _git(work_repo, "add", "proof.txt")
    _git(work_repo, "add", "benchmarks/fixtures/single_stage_seed_iota15")
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
        "--single-stage-warm-start-run-dir",
        "benchmarks/fixtures/single_stage_seed_iota15",
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


def test_resolve_hf_cli_requires_hf_on_path(monkeypatch):
    def forbidden_path_lookup(_path):
        raise AssertionError("HF CLI resolution must not consult hard-coded paths")

    monkeypatch.setattr(launcher.shutil, "which", lambda _name: None)
    monkeypatch.setattr(launcher, "Path", forbidden_path_lookup)

    with pytest.raises(RuntimeError, match="Could not find the Hugging Face CLI"):
        launcher._resolve_hf_cli()


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
    assert "--detach" in completed.stdout
    assert '"effective_geometry_rel_tol": null' in completed.stdout
    assert '"stage2_geometry_policy": "report-only"' in completed.stdout
    assert "--geometry-rel-tol" not in completed.stdout
    assert "--single-stage-boozer-optimizer-backend" not in completed.stdout
    assert "unset LD_LIBRARY_PATH" not in completed.stdout
    assert 'export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"' in completed.stdout
    assert (
        'export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-'
        '${PWD}/.artifacts/jax_compilation_cache/hf-production-proof}"'
        in completed.stdout
    )
    assert "/tmp/jax-compilation-cache" not in completed.stdout
    assert "SIMSOPT_HF_JOB_BOOTSTRAP_MODE" not in completed.stdout
    assert "SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC" not in completed.stdout
    assert 'SIMSOPT_JAX_CUDA_LIBRARY_MODE="bundled"' in completed.stdout
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in completed.stdout
    assert (
        "--single-stage-warm-start-run-dir "
        "/tmp/hf-production-proof/repo/benchmarks/fixtures/single_stage_seed_iota15"
        in completed.stdout
    )
    assert "stage2_warm_repro" not in completed.stdout


def test_launch_production_gpu_proof_no_detach_streams_remote_result(tmp_path):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--no-detach",
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
    assert "--detach" not in completed.stdout
    assert "--hardware" not in completed.stdout
    assert "hf jobs run --flavor a100-large" in completed.stdout


def test_run_hf_job_foreground_inspects_terminal_status(monkeypatch):
    waited: list[tuple[str, str]] = []

    class FakeProcess:
        stdout = iter(["Job started with ID: job-123\n", "remote log\n"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        launcher,
        "_wait_for_successful_hf_job",
        lambda hf_cli, job_id: waited.append((hf_cli, job_id)),
    )

    launcher._run_hf_job_foreground("hf", ["hf", "jobs", "run"])

    assert waited == [("hf", "job-123")]


def test_run_hf_job_foreground_rejects_missing_job_id(monkeypatch):
    class FakeProcess:
        stdout = iter(["remote log without job id\n"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    with pytest.raises(RuntimeError, match="Could not parse HF job id"):
        launcher._run_hf_job_foreground("hf", ["hf", "jobs", "run"])


def test_inspect_hf_job_stage_requests_json_format(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps([{"status": {"stage": "COMPLETED"}}]),
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher._inspect_hf_job_stage("hf", "job-123") == "COMPLETED"
    assert calls == [["hf", "jobs", "inspect", "--format", "json", "job-123"]]


def test_wait_for_successful_hf_job_rejects_remote_failure(monkeypatch):
    monkeypatch.setattr(launcher, "_inspect_hf_job_stage", lambda _hf_cli, _job_id: "ERROR")

    with pytest.raises(SystemExit, match="finished with stage ERROR"):
        launcher._wait_for_successful_hf_job("hf", "job-123", poll_interval_s=0.0)


def test_wait_for_successful_hf_job_accepts_completed_status(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "_inspect_hf_job_stage",
        lambda _hf_cli, _job_id: "COMPLETED",
    )

    launcher._wait_for_successful_hf_job("hf", "job-123", poll_interval_s=0.0)


def test_launch_production_gpu_proof_dry_run_threads_single_stage_benchmark_mode(
    tmp_path,
):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--single-stage-benchmark-mode",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert "--single-stage-benchmark-mode" in completed.stdout


def test_launch_production_gpu_proof_dry_run_threads_single_stage_success_filter_bypass(
    tmp_path,
):
    env = _launcher_env(tmp_path)
    remote_args = _default_remote_launcher_args(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--single-stage-disable-target-lane-success-filter",
            *remote_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert "--single-stage-disable-target-lane-success-filter" in completed.stdout


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


def test_launch_production_gpu_proof_requires_single_stage_seed(tmp_path):
    env = _launcher_env(tmp_path)
    repo_url, main_sha, _ = _build_remote_validation_repo(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--dry-run",
            "--hardware",
            "a100-large",
            "--repo-url",
            repo_url,
            "--repo-ref",
            "main",
            "--repo-sha",
            main_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "requires a single-stage seed" in completed.stderr


def test_launch_production_gpu_proof_rejects_bootstrap_mode_override():
    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER_SCRIPT),
            "--bootstrap-mode",
            "always",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "unrecognized arguments: --bootstrap-mode" in completed.stderr


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
            "--single-stage-warm-start-run-dir",
            "benchmarks/fixtures/single_stage_seed_iota15",
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
            "--single-stage-warm-start-run-dir",
            "benchmarks/fixtures/single_stage_seed_iota15",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert f'"repo_sha": "{main_sha}"' in completed.stdout


def test_launch_production_gpu_proof_accepts_repo_runtime_seed_spec(tmp_path):
    env = _launcher_env(tmp_path)
    repo_url, main_sha, _ = _build_remote_validation_repo_with_seed(
        tmp_path,
        include_runtime_seed_spec=True,
    )
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
            "--single-stage-mpol",
            "10",
            "--single-stage-ntor",
            "10",
            "--single-stage-jax-runtime-seed-spec",
            "benchmarks/fixtures/single_stage_seed_iota15",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode == 0
    assert (
        '"single_stage_jax_runtime_seed_spec": '
        '"benchmarks/fixtures/single_stage_seed_iota15"'
    ) in completed.stdout
    assert (
        "--single-stage-jax-runtime-seed-spec "
        "/tmp/hf-production-proof/repo/benchmarks/fixtures/single_stage_seed_iota15"
    ) in completed.stdout


def test_launch_production_gpu_proof_rejects_host_absolute_seed_path(tmp_path):
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
            "--single-stage-jax-runtime-seed-spec",
            str(tmp_path / "single_stage_seed_spec.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "repo-relative path" in completed.stderr


def test_launch_production_gpu_proof_rejects_seed_path_missing_from_target_sha(
    tmp_path,
):
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
            "--single-stage-warm-start-run-dir",
            "missing/single-stage-seed",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "path is not present at repo SHA" in completed.stderr


def test_launch_production_gpu_proof_rejects_stage2_seed_missing_from_target_sha(
    tmp_path,
):
    env = _launcher_env(tmp_path)
    repo_url, main_sha, _ = _build_remote_validation_repo_with_seed(
        tmp_path,
        include_stage2_seed=False,
        include_runtime_seed_spec=True,
    )
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
            "--single-stage-jax-runtime-seed-spec",
            "benchmarks/fixtures/single_stage_seed_iota15",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "--stage2-bs-path" in completed.stderr
    assert "path is not present at repo SHA" in completed.stderr


def test_launch_production_gpu_proof_rejects_runtime_seed_directory_without_spec(
    tmp_path,
):
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
            "--single-stage-jax-runtime-seed-spec",
            "benchmarks/fixtures/single_stage_seed_iota15",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "single_stage_jax_runtime_spec.json" in completed.stderr


def test_launch_production_gpu_proof_rejects_runtime_seed_spec_tree(
    tmp_path,
):
    env = _launcher_env(tmp_path)
    repo_url, main_sha, _ = _build_remote_validation_repo_with_seed(
        tmp_path,
        runtime_spec_tree=True,
    )
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
            "--single-stage-jax-runtime-seed-spec",
            "benchmarks/fixtures/single_stage_seed_iota15",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    assert completed.returncode != 0
    assert "directory must contain a file named" in completed.stderr


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
