import argparse
import json
import os
from pathlib import Path
import sys
import math
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.adjoint_fd_validation import (
    ADJOINT_RESIDUAL_REL_TOL,
    FIXED_SURFACE_FD_ABS_TOL,
    FIXED_SURFACE_FD_REL_TOL,
    FULL_RESOLVE_FD_ABS_TOL,
    FULL_RESOLVE_FD_REL_TOL,
    RECOMPOSED_TOTAL_REL_TOL,
    evaluate_adjoint_validation,
)
from benchmarks.adjoint_probe_common import compute_derivative_l2_metrics
from benchmarks.single_stage_backend_routing import (
    resolve_boozer_limited_memory,
    resolve_boozer_optimizer_method,
)
from benchmarks.grouped_adjoint_memory_probe import (
    _build_grouped_adjoint_payload,
    evaluate_grouped_adjoint_memory_probe,
)
import benchmarks.single_stage_init_parity as single_stage_init_parity_module
from benchmarks.single_stage_outer_loop_probe import (
    TARGET_OUTER_OPTIMIZER_METHOD,
    evaluate_single_stage_outer_loop_probe,
)
from benchmarks.benchmark_config import DEFAULT_CONFIGS, resolve_configs
from benchmarks.benchmark_problem import (
    build_ls_parity_problem,
    build_synthetic_boozer_problem,
    clone_tensor_surface,
)
import benchmarks.run_code_benchmark_common as run_code_benchmark_common
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.single_stage_init_parity import (
    DEFAULT_OUTER_MAXITER,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    FIELD_ERROR_REL_TOL,
    IOTA_ABS_TOL,
    SURFACE_GEOMETRY_REL_TOL,
    VOLUME_REL_TOL,
    evaluate_single_stage_init_parity,
)
from benchmarks.stage2_e2e_comparison import (
    build_stage2_e2e_payload,
    evaluate_stage2_e2e_comparison,
)
import benchmarks.stage2_e2e_comparison as stage2_e2e_comparison_module
from benchmarks.tier5_performance_characterization import (
    safe_speedup,
    summarize_pair_probe,
    summarize_single_lane_probe,
)
from benchmarks.validation_ladder_common import (
    _JAX_COMPILATION_CACHE_ENV_VAR,
    _SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR,
    apply_compilation_cache_policy,
    build_provenance,
    describe_compile_behavior,
    max_pointwise_geometry_drift,
    optimizer_drift_tolerances,
    repo_pythonpath_env,
    require_x64_runtime,
    resolve_probe_lane,
    run_python_script,
    short_run_geometry_rel_tolerance,
)


def test_resolve_configs_defaults_to_all_configs():
    assert resolve_configs(None) == DEFAULT_CONFIGS


def test_resolve_configs_preserves_requested_order():
    labels = [
        "Columbia (12 coils, 128x64)",
        "Small (4 coils, 15x15)",
    ]
    configs = resolve_configs(labels)
    assert [config.label for config in configs] == labels


def test_resolve_configs_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unknown benchmark config"):
        resolve_configs(["does-not-exist"])


def test_summarize_result_fun_prefers_fun():
    assert summarize_result_fun({"fun": np.float64(1.25)}) == 1.25


def test_summarize_result_fun_falls_back_to_residual_norm():
    residual = np.array([1.0, 2.0, 3.0])
    expected = 0.5 * float(np.mean(np.square(residual)))
    assert summarize_result_fun({"residual": residual}) == expected


def test_summarize_result_fun_returns_nan_without_fun_or_residual():
    assert math.isnan(summarize_result_fun({}))


def test_build_synthetic_boozer_problem_uses_requested_grid():
    config = DEFAULT_CONFIGS[0]
    problem = build_synthetic_boozer_problem(config)

    assert len(problem.surface.quadpoints_phi) == config.nphi
    assert len(problem.surface.quadpoints_theta) == config.ntheta
    assert problem.surface.stellsym is False
    assert problem.surface.nfp == config.nfp
    assert problem.iota0 == pytest.approx(0.3)
    assert problem.G0 > 0.0


def test_build_ls_parity_problem_matches_known_good_fixture_shape():
    problem = build_ls_parity_problem()

    assert problem.surface.stellsym is True
    assert problem.surface.nfp == 2
    assert len(problem.surface.quadpoints_phi) == 5
    assert len(problem.surface.quadpoints_theta) == 5
    assert problem.iota0 == pytest.approx(0.3)
    assert problem.G0 > 0.0


def test_clone_tensor_surface_is_independent():
    problem = build_ls_parity_problem()
    surface_copy = clone_tensor_surface(problem.surface)

    np.testing.assert_allclose(surface_copy.get_dofs(), problem.surface.get_dofs())
    new_dofs = surface_copy.get_dofs().copy()
    new_dofs[0] += 1.0
    surface_copy.set_dofs(new_dofs)

    assert surface_copy is not problem.surface
    assert surface_copy.get_dofs()[0] != pytest.approx(problem.surface.get_dofs()[0])


def test_max_pointwise_geometry_drift_matches_expected_scale():
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    actual = np.array([[1.0, 0.0, 0.0], [0.0, 2.0003, 0.0]])

    max_abs, max_rel = max_pointwise_geometry_drift(actual, reference)

    assert max_abs == pytest.approx(3e-4)
    assert max_rel == pytest.approx(1.5e-4)


def test_short_run_geometry_rel_tolerance_disables_20_iter_smoke_gate():
    assert short_run_geometry_rel_tolerance(20) is None
    assert short_run_geometry_rel_tolerance(21) == pytest.approx(1e-6)
    assert short_run_geometry_rel_tolerance(20, explicit_tol=2.5e-6) == pytest.approx(
        2.5e-6
    )


def test_single_stage_init_fixture_files_are_vendored():
    assert DEFAULT_STAGE2_BS_PATH.is_file()
    assert DEFAULT_STAGE2_BS_PATH.with_name("results.json").is_file()


def test_single_stage_init_fixture_results_include_required_seed_metadata():
    results = json.loads(DEFAULT_STAGE2_BS_PATH.with_name("results.json").read_text())

    assert results["MAJOR_RADIUS"] > 0.0
    assert results["TOROIDAL_FLUX"] > 0.0
    assert results["banana_surf_radius"] > 0.0
    assert results["order"] >= 1


def test_single_stage_init_defaults_to_reduced_grid_smoke_fixture(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["single_stage_init_parity.py", "--output-json", "/tmp/out.json"],
    )

    args = single_stage_init_parity_module.parse_args()

    assert args.nphi == DEFAULT_SMOKE_NPHI
    assert args.ntheta == DEFAULT_SMOKE_NTHETA
    assert args.mpol == DEFAULT_SMOKE_MPOL
    assert args.ntor == DEFAULT_SMOKE_NTOR
    assert (
        args.optimizer_backend
        == single_stage_init_parity_module.TARGET_OPTIMIZER_BACKEND
    )
    assert args.boozer_optimizer_backend is None
    assert args.maxiter == DEFAULT_OUTER_MAXITER


def test_repo_pythonpath_env_sets_all_platform_selectors(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = repo_pythonpath_env(platform="cpu")

    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["SIMSOPT_JAX_PLATFORM"] == "cpu"
    assert env["SIMSOPT_JAX_BACKEND"] == "cpu"


def test_repo_pythonpath_env_auto_clears_inherited_platform_selectors(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("SIMSOPT_JAX_PLATFORM", "cuda")
    monkeypatch.setenv("SIMSOPT_JAX_BACKEND", "cuda")
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing")

    env = repo_pythonpath_env(platform="auto")

    assert "JAX_PLATFORMS" not in env
    assert "SIMSOPT_JAX_PLATFORM" not in env
    assert "SIMSOPT_JAX_BACKEND" not in env
    assert env["PYTHONPATH"].endswith("/tmp/existing")


def test_apply_compilation_cache_policy_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(_JAX_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)

    metadata = apply_compilation_cache_policy()

    assert metadata == {
        "compilation_cache_enabled": False,
        "compilation_cache_dir": None,
        "compilation_cache_policy": "disabled",
    }
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in os.environ


def test_apply_compilation_cache_policy_honors_explicit_cache_dir(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)

    metadata = apply_compilation_cache_policy(tmp_path / "jax-cache")

    assert metadata["compilation_cache_enabled"] is True
    assert metadata["compilation_cache_policy"] == "explicit"
    assert metadata["compilation_cache_dir"] == str(tmp_path / "jax-cache")
    assert Path(metadata["compilation_cache_dir"]).is_dir()


def test_apply_compilation_cache_policy_honors_disable_flag(monkeypatch):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")
    monkeypatch.setenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, "1")

    metadata = apply_compilation_cache_policy()

    assert metadata == {
        "compilation_cache_enabled": False,
        "compilation_cache_dir": None,
        "compilation_cache_policy": "disabled",
    }
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in os.environ


def test_optimizer_drift_tolerances_tier2_geometry_gate_tracks_iteration_budget():
    tol_20 = optimizer_drift_tolerances("tier2_stage2_e2e", maxiter=20)
    tol_21 = optimizer_drift_tolerances("tier2_stage2_e2e", maxiter=21)

    assert tol_20["geometry_rel_tol"] is None
    assert tol_21["geometry_rel_tol"] == pytest.approx(1e-6)
    assert "geometry_rel_tol_20_iter" not in tol_20
    assert "geometry_rel_tol_default" not in tol_20


def test_describe_compile_behavior_tracks_cache_state(monkeypatch):
    monkeypatch.delenv(_JAX_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)
    apply_compilation_cache_policy()
    assert (
        describe_compile_behavior(uses_subprocesses=True)
        == "persistent compilation cache disabled; subprocess timings include first-call compilation"
    )

    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)
    apply_compilation_cache_policy()
    assert (
        describe_compile_behavior(uses_subprocesses=False)
        == "persistent compilation cache enabled; cached executables may reduce first-call compile cost"
    )


def test_resolve_probe_lane_tracks_private_optimizer_backends():
    assert resolve_probe_lane() == "trusted-public-reference"
    assert resolve_probe_lane(optimizer_backend="scipy") == "trusted-public-reference"
    assert resolve_probe_lane(optimizer_backend="hybrid") == "private-optimizer"
    assert resolve_probe_lane(optimizer_backend="ondevice") == "private-optimizer"


def test_run_code_benchmark_runtime_lane_matches_ladder_vocabulary(monkeypatch):
    monkeypatch.setattr(
        run_code_benchmark_common,
        "_current_jax_version",
        lambda: run_code_benchmark_common.EXPECTED_BENCHMARK_JAX_VERSION,
    )
    assert (
        run_code_benchmark_common._resolve_runtime_lane(("scipy",))
        == "trusted-public-reference"
    )
    assert (
        run_code_benchmark_common._resolve_runtime_lane(("hybrid",))
        == "private-optimizer"
    )


def test_require_x64_runtime_rejects_float32_runtime():
    fake_jax = types.SimpleNamespace(
        numpy=types.SimpleNamespace(
            zeros=lambda n: np.zeros(n, dtype=np.float32),
            float64=np.float64,
        )
    )

    with pytest.raises(RuntimeError, match="Tier 5 requires jax_enable_x64=True"):
        require_x64_runtime(fake_jax, context="Tier 5")


def test_build_provenance_includes_compilation_cache_metadata(monkeypatch):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/probe-cache")
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.get_git_sha",
        lambda: "abc123",
    )
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.peak_rss_mb",
        lambda: 12.5,
    )
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.query_gpu_memory_mb",
        lambda: None,
    )
    fake_jax = types.SimpleNamespace(
        __version__=run_code_benchmark_common.EXPECTED_BENCHMARK_JAX_VERSION,
        default_backend=lambda: "cpu",
        devices=lambda: ["cpu:0"],
        numpy=types.SimpleNamespace(
            zeros=lambda n: np.zeros(n, dtype=np.float64),
            float64=np.float64,
        ),
    )
    fake_jaxlib = types.SimpleNamespace(
        __version__=run_code_benchmark_common.EXPECTED_BENCHMARK_JAX_VERSION
    )

    provenance = build_provenance(
        fake_jax,
        fake_jaxlib,
        title="Probe",
        extra={"lane": "private-optimizer", "compile_behavior": "cold+warm"},
    )

    assert provenance["lane"] == "private-optimizer"
    assert provenance["compile_behavior"] == "cold+warm"
    assert provenance["compilation_cache_enabled"] is True
    assert provenance["compilation_cache_dir"] == "/tmp/probe-cache"


def test_run_python_script_streams_and_captures_output(tmp_path, capsys):
    script = tmp_path / "echo_child.py"
    script.write_text(
        "import sys\n"
        "print('stdout-line', flush=True)\n"
        "print('stderr-line', file=sys.stderr, flush=True)\n",
        encoding="utf-8",
    )

    result = run_python_script(
        script,
        [],
        cwd=tmp_path,
        stream_output=True,
    )

    captured = capsys.readouterr()
    assert "stdout-line" in captured.out
    assert "stderr-line" in captured.err
    assert "stdout-line" in result.stdout
    assert "stderr-line" in result.stderr


def test_run_python_script_stream_output_preserves_failure_details(tmp_path, capsys):
    script = tmp_path / "fail_child.py"
    script.write_text(
        "import sys\n"
        "print('before-fail', flush=True)\n"
        "print('boom', file=sys.stderr, flush=True)\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="Subprocess failed with exit code 3"
    ) as excinfo:
        run_python_script(
            script,
            [],
            cwd=tmp_path,
            stream_output=True,
        )

    captured = capsys.readouterr()
    assert "before-fail" in captured.out
    assert "boom" in captured.err
    assert "stdout:\nbefore-fail" in str(excinfo.value)
    assert "stderr:\nboom" in str(excinfo.value)


def test_single_stage_init_parity_accepts_small_real_fixture_differences():
    cpu_results = {
        "FINAL_IOTA": 0.1500,
        "FINAL_VOLUME": 0.1000000,
        "FIELD_ERROR": 0.0030,
        "MAX_CURVATURE": 12.0,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
    }
    jax_results = {
        "FINAL_IOTA": 0.1505,
        "FINAL_VOLUME": 0.10000005,
        "FIELD_ERROR": 0.0030002,
        "MAX_CURVATURE": 12.1,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
    }

    comparison, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=1e-6,
        max_surface_geometry_rel=5e-6,
    )

    assert comparison["final_iota_abs_diff"] < IOTA_ABS_TOL
    assert comparison["final_volume_rel_diff"] < VOLUME_REL_TOL
    assert comparison["field_error_rel_diff"] < FIELD_ERROR_REL_TOL
    assert comparison["max_surface_pointwise_rel"] < SURFACE_GEOMETRY_REL_TOL
    assert failures == []
    assert comparison["cpu_self_intersection_check_available"] is True
    assert comparison["jax_self_intersection_check_available"] is True


def test_single_stage_init_parity_reports_real_gate_failures():
    cpu_results = {
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.10,
        "FIELD_ERROR": 0.003,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
    }
    jax_results = {
        "FINAL_IOTA": 0.17,
        "FINAL_VOLUME": 0.101,
        "FIELD_ERROR": 0.004,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": True,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
    }

    _, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=1e-4,
        max_surface_geometry_rel=2e-5,
    )

    assert any("Final iota disagreement too large" in failure for failure in failures)
    assert any(
        "Final volume relative difference too large" in failure for failure in failures
    )
    assert any(
        "Final field error relative difference too large" in failure
        for failure in failures
    )
    assert any(
        "Initial Boozer surface geometry drift too large" in failure
        for failure in failures
    )
    assert any("self-intersecting" in failure for failure in failures)


def test_single_stage_init_parity_tracks_self_intersection_check_availability():
    cpu_results = {
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.10,
        "FIELD_ERROR": 0.003,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": False,
    }
    jax_results = {
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.10,
        "FIELD_ERROR": 0.003,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
    }

    comparison, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=1e-6,
        max_surface_geometry_rel=5e-6,
    )

    assert failures == []
    assert comparison["cpu_self_intersection_check_available"] is False
    assert comparison["jax_self_intersection_check_available"] is True


def test_single_stage_init_case_loads_surface_before_tempdir_cleanup(
    monkeypatch, tmp_path
):
    args = argparse.Namespace(
        plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
        stage2_bs_path=str(DEFAULT_STAGE2_BS_PATH),
        nphi=63,
        ntheta=32,
        mpol=4,
        ntor=4,
        vol_target=0.1,
        iota_target=0.15,
        optimizer_backend="scipy",
        boozer_optimizer_backend=None,
        maxiter=0,
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        lambda *args, **kwargs: argparse.Namespace(stdout="", stderr=""),
    )

    def fake_find_single_file(root: str | Path, pattern: str) -> Path:
        path = Path(root) / pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(
        single_stage_init_parity_module, "find_single_file", fake_find_single_file
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "load_json",
        lambda path: {
            "FINAL_IOTA": 0.15,
            "FINAL_VOLUME": 0.1,
            "FIELD_ERROR": 0.003,
            "MAX_CURVATURE": 10.0,
            "SELF_INTERSECTING": False,
        },
    )

    observed_paths: list[Path] = []

    def fake_load_surface_gamma_artifact(surface_json_path: str) -> np.ndarray:
        path = Path(surface_json_path)
        observed_paths.append(path)
        assert path.exists()
        return np.zeros((2, 2, 3))

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_load_surface_gamma_artifact",
        fake_load_surface_gamma_artifact,
    )

    payload = single_stage_init_parity_module._run_single_stage_case(
        args,
        "cpu",
        platform="cpu",
    )

    assert observed_paths
    np.testing.assert_allclose(payload["surface_gamma"], np.zeros((2, 2, 3)))
    assert payload["results"]["SELF_INTERSECTING"] is False


def test_single_stage_init_case_threads_optimizer_backend_to_jax_lane(
    monkeypatch, tmp_path
):
    args = argparse.Namespace(
        plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
        stage2_bs_path=str(DEFAULT_STAGE2_BS_PATH),
        nphi=63,
        ntheta=32,
        mpol=4,
        ntor=4,
        vol_target=0.1,
        iota_target=0.15,
        optimizer_backend="ondevice",
        boozer_optimizer_backend="scipy",
        maxiter=1,
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    observed_commands: list[list[str]] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )

    def fake_run_python_script(_script_path, command, **_kwargs):
        observed_commands.append(list(command))
        return argparse.Namespace(stdout="", stderr="")

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        fake_run_python_script,
    )

    def fake_find_single_file(root: str | Path, pattern: str) -> Path:
        path = Path(root) / pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(
        single_stage_init_parity_module, "find_single_file", fake_find_single_file
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "load_json",
        lambda _path: {
            "FINAL_IOTA": 0.15,
            "FINAL_VOLUME": 0.1,
            "FIELD_ERROR": 0.003,
            "MAX_CURVATURE": 10.0,
            "SELF_INTERSECTING": False,
        },
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_load_surface_gamma_artifact",
        lambda _surface_json_path: np.zeros((2, 2, 3)),
    )

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cpu",
    )

    assert observed_commands
    assert "--init-only" not in observed_commands[0]
    maxiter_flag_index = observed_commands[0].index("--maxiter")
    assert observed_commands[0][maxiter_flag_index + 1] == "1"
    optimizer_flag_index = observed_commands[0].index("--optimizer-backend")
    assert observed_commands[0][optimizer_flag_index + 1] == "ondevice"
    boozer_optimizer_flag_index = observed_commands[0].index(
        "--boozer-optimizer-backend"
    )
    assert observed_commands[0][boozer_optimizer_flag_index + 1] == "scipy"


def _single_stage_probe_results(**overrides):
    results = {
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.1,
        "FIELD_ERROR": 0.003,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
        "iterations": 1,
        "boozer_optimizer_backend": "scipy",
        "boozer_optimizer_method": "bfgs",
        "outer_optimizer_method": "lbfgs",
    }
    results.update(overrides)
    return results


def test_single_stage_init_parity_requires_accepted_step_on_outer_loop_probe():
    cpu_results = _single_stage_probe_results()
    jax_results = _single_stage_probe_results(
        iterations=0,
        outer_optimizer_method="lbfgs-ondevice",
    )

    _, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=0.0,
        max_surface_geometry_rel=0.0,
        maxiter=1,
    )

    assert any("did not accept an optimizer step" in failure for failure in failures)


def test_single_stage_outer_loop_probe_accepts_finite_target_lane_result():
    summary, failures = evaluate_single_stage_outer_loop_probe(
        _single_stage_probe_results(
            FINAL_IOTA=0.01,
            FIELD_ERROR=0.004,
            MAX_CURVATURE=32.0,
            SELF_INTERSECTION_CHECK_AVAILABLE=False,
            outer_optimizer_method=TARGET_OUTER_OPTIMIZER_METHOD,
        ),
        expected_boozer_optimizer_backend="scipy",
        expected_boozer_optimizer_method="bfgs",
    )

    assert failures == []
    assert summary["iterations"] == 1
    assert summary["boozer_optimizer_backend"] == "scipy"
    assert summary["outer_optimizer_method"] == TARGET_OUTER_OPTIMIZER_METHOD
    assert summary["self_intersection_check_available"] is False


def test_single_stage_outer_loop_probe_rejects_missing_step_or_wrong_method():
    _, failures = evaluate_single_stage_outer_loop_probe(
        _single_stage_probe_results(
            iterations=0,
            boozer_optimizer_backend="ondevice",
            boozer_optimizer_method="bfgs-ondevice",
            SELF_INTERSECTING=True,
            FINAL_IOTA=np.nan,
            FIELD_ERROR=0.004,
            MAX_CURVATURE=32.0,
        ),
        expected_boozer_optimizer_backend="scipy",
        expected_boozer_optimizer_method="bfgs",
    )

    assert any("did not accept an optimizer step" in failure for failure in failures)
    assert any("requested inner Boozer backend" in failure for failure in failures)
    assert any(
        "requested inner Boozer optimizer method" in failure for failure in failures
    )
    assert any(TARGET_OUTER_OPTIMIZER_METHOD in failure for failure in failures)
    assert any("self-intersecting surface" in failure for failure in failures)
    assert any("non-finite FINAL_IOTA" in failure for failure in failures)


def _grouped_adjoint_memory_metrics(*, snapshots, **overrides):
    metrics = {
        "adjoint_residual_rel": 1e-12,
        "adjoint_norm": 1.0,
        "adjoint_finite": True,
        "implicit_gradient_finite": True,
        "implicit_gradient_norm": 1.0,
        "snapshots": snapshots,
    }
    metrics.update(overrides)
    return metrics


def _grouped_adjoint_snapshot(label, rss_mb):
    return {
        "label": label,
        "elapsed_s": rss_mb * 0.5,
        "rss_mb": rss_mb,
        "gpu_memory_mb": None,
    }


def _complete_grouped_adjoint_snapshots():
    return [
        _grouped_adjoint_snapshot("start", 1.0),
        _grouped_adjoint_snapshot("after_stage2_results_load", 1.25),
        _grouped_adjoint_snapshot("after_biotsavart_load", 1.5),
        _grouped_adjoint_snapshot("after_surface_seed_setup", 1.75),
        _grouped_adjoint_snapshot("after_boozer_surface_fit", 2.0),
        _grouped_adjoint_snapshot("after_boozer_setup", 2.25),
        _grouped_adjoint_snapshot("after_boozer_lbfgs", 2.5),
        _grouped_adjoint_snapshot("before_boozer_newton", 2.625),
        _grouped_adjoint_snapshot("after_boozer_newton", 2.75),
        _grouped_adjoint_snapshot("after_boozer_solve", 3.0),
        _grouped_adjoint_snapshot("after_boozer_postprocess", 3.25),
        _grouped_adjoint_snapshot("after_fixture", 3.5),
        _grouped_adjoint_snapshot("after_objective", 4.5),
        _grouped_adjoint_snapshot("after_adjoint_solve", 5.5),
        _grouped_adjoint_snapshot("before_grouped_adjoint_vjp", 6.0),
        _grouped_adjoint_snapshot("after_grouped_adjoint_vjp_first_group", 6.25),
        _grouped_adjoint_snapshot("after_grouped_adjoint_vjp_end", 6.5),
        _grouped_adjoint_snapshot("after_derivative_projection", 6.75),
        _grouped_adjoint_snapshot("after_norm_metrics", 7.0),
    ]


def test_compute_derivative_l2_metrics_ignores_missing_dependency_keys():
    class _DepOpt:
        __slots__ = ("local_dofs_free_status",)
        __hash__ = object.__hash__

        def __init__(self, free_status):
            self.local_dofs_free_status = np.asarray(free_status, dtype=bool)

    dep_present = _DepOpt([True, False])
    dep_missing = _DepOpt([True])
    lineage = types.SimpleNamespace(
        dofs_free_status=np.array([True], dtype=bool),
        local_dof_size=1,
        dofs=types.SimpleNamespace(dep_opts=lambda: [dep_present, dep_missing]),
    )
    derivative = types.SimpleNamespace(
        data={dep_present: np.array([3.0, 4.0], dtype=float)}
    )

    norm, finite = compute_derivative_l2_metrics(
        derivative,
        types.SimpleNamespace(unique_dof_lineage=[lineage]),
    )

    assert finite is True
    assert norm == pytest.approx(3.0)
    assert dep_missing not in derivative.data


def test_grouped_adjoint_memory_probe_requires_complete_finite_metrics():
    failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(snapshots=_complete_grouped_adjoint_snapshots())
    )

    assert failures == []


def test_grouped_adjoint_memory_probe_rejects_missing_snapshots_or_nonfinite_gradient():
    failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(
            adjoint_residual_rel=np.inf,
            implicit_gradient_finite=False,
            implicit_gradient_norm=0.0,
            snapshots=[_grouped_adjoint_snapshot("start", 1.0)],
        )
    )

    assert any("required snapshots" in failure for failure in failures)
    assert any("non-finite implicit gradient" in failure for failure in failures)


def test_grouped_adjoint_memory_payload_records_limited_memory_route():
    snapshots = _complete_grouped_adjoint_snapshots()
    metrics = _grouped_adjoint_memory_metrics(snapshots=snapshots)

    payload = _build_grouped_adjoint_payload(
        provenance={
            "title": "Grouped adjoint memory probe",
            "optimizer_backend": "ondevice",
            "boozer_limited_memory_requested": True,
            "boozer_optimizer_backend": "ondevice",
            "boozer_limited_memory": True,
        },
        fixture={
            "equilibrium_path": Path("/tmp/equilibrium.nc"),
            "stage2_bs_path": Path("/tmp/biot_savart_opt.json"),
            "boozer_optimizer_backend": "ondevice",
        },
        base_result={
            "success": True,
            "optimizer_method": "lbfgs-ondevice",
        },
        metrics=metrics,
        failures=[],
        snapshots=snapshots,
        boozer_limited_memory=True,
        boozer_limited_memory_requested=True,
    )

    assert payload["provenance"]["boozer_limited_memory"] is True
    assert payload["provenance"]["boozer_optimizer_backend"] == "ondevice"
    assert payload["baseline"]["boozer_optimizer_backend"] == "ondevice"
    assert payload["baseline"]["optimizer_method"] == "lbfgs-ondevice"
    assert payload["baseline"]["boozer_limited_memory"] is True
    assert payload["baseline"]["boozer_limited_memory_requested"] is True


def test_grouped_adjoint_memory_payload_records_requested_but_inactive_limited_memory():
    snapshots = _complete_grouped_adjoint_snapshots()
    metrics = _grouped_adjoint_memory_metrics(snapshots=snapshots)

    payload = _build_grouped_adjoint_payload(
        provenance={
            "title": "Grouped adjoint memory probe",
            "optimizer_backend": "ondevice",
            "boozer_limited_memory_requested": True,
            "boozer_optimizer_backend": "scipy",
            "boozer_limited_memory": False,
        },
        fixture={
            "equilibrium_path": Path("/tmp/equilibrium.nc"),
            "stage2_bs_path": Path("/tmp/biot_savart_opt.json"),
            "boozer_optimizer_backend": "scipy",
        },
        base_result={
            "success": True,
            "optimizer_method": "bfgs",
        },
        metrics=metrics,
        failures=[],
        snapshots=snapshots,
        boozer_limited_memory=False,
        boozer_limited_memory_requested=True,
    )

    assert payload["provenance"]["boozer_limited_memory"] is False
    assert payload["provenance"]["boozer_limited_memory_requested"] is True
    assert payload["baseline"]["boozer_optimizer_backend"] == "scipy"
    assert payload["baseline"]["optimizer_method"] == "bfgs"
    assert payload["baseline"]["boozer_limited_memory"] is False
    assert payload["baseline"]["boozer_limited_memory_requested"] is True


def test_grouped_adjoint_memory_resolves_effective_limited_memory_route():
    assert resolve_boozer_limited_memory("ondevice", True) is True
    assert resolve_boozer_limited_memory("scipy", True) is False
    assert resolve_boozer_limited_memory("ondevice", False) is False


def test_single_stage_outer_loop_probe_resolves_expected_boozer_method():
    assert resolve_boozer_optimizer_method("scipy") == "bfgs"
    assert resolve_boozer_optimizer_method("scipy", limited_memory=True) == "lbfgs"
    assert resolve_boozer_optimizer_method("hybrid") == "bfgs-hybrid"
    with pytest.raises(ValueError, match="does not support limited_memory=True"):
        resolve_boozer_optimizer_method("hybrid", limited_memory=True)
    assert resolve_boozer_optimizer_method("ondevice") == "bfgs-ondevice"
    assert (
        resolve_boozer_optimizer_method("ondevice", limited_memory=True)
        == "lbfgs-ondevice"
    )


def _stage2_e2e_comparison_case(**overrides):
    comparison = {
        "optimizer_backend": "scipy",
        "final_objective_rel_diff": 1e-7,
        "field_error_rel_diff": 1e-7,
        "field_error_rel_tol": 1e-4,
        "max_geometry_pointwise_rel": 2e-6,
        "geometry_rel_tol": 5e-6,
        "cpu_trajectory_finite": True,
        "jax_trajectory_finite": True,
        "cpu_trajectory_improves": True,
        "jax_trajectory_improves": True,
        "matched_cpu_state": {
            "objective_rel_diff": 1e-12,
            "field_error_rel_diff": 1e-12,
            "gradient_allclose": True,
            "gradient_l2_rel_diff": 1e-12,
        },
        "matched_jax_state": {
            "objective_rel_diff": 1e-12,
            "field_error_rel_diff": 1e-12,
            "gradient_allclose": True,
            "gradient_l2_rel_diff": 1e-12,
        },
    }
    comparison.update(overrides)
    return comparison


def _stage2_probe_payload_case(**overrides):
    payload = {
        "composite": {
            "J": 1.0,
            "mean_abs_relBfinal_norm": 0.01,
            "dJ": [0.5, -0.25],
        }
    }
    payload.update(overrides)
    return payload


def _stage2_gradient_term_case(
    *,
    objective_rel_diff=1e-12,
    gradient_allclose=True,
    gradient_componentwise_allclose=True,
    gradient_global_scale_match=True,
    gradient_l2_rel_diff=1e-12,
    gradient_max_abs_diff=1e-12,
    gradient_scaled_atol=1e-12,
):
    return {
        "objective_rel_diff": objective_rel_diff,
        "gradient_allclose": gradient_allclose,
        "gradient_componentwise_allclose": gradient_componentwise_allclose,
        "gradient_global_scale_match": gradient_global_scale_match,
        "gradient_l2_rel_diff": gradient_l2_rel_diff,
        "gradient_max_abs_diff": gradient_max_abs_diff,
        "gradient_scaled_atol": gradient_scaled_atol,
    }


def _stage2_e2e_results_case(**overrides):
    results = {
        "FINAL_OBJECTIVE": 1.0,
        "FIELD_ERROR": 0.01,
        "FINAL_BANANA_GAMMA": [[[1.0, 0.0, 0.0]]],
        "FINAL_CURVE_LENGTH": 1.0,
        "FINAL_CC_DISTANCE": 0.2,
        "LENGTH_TARGET": 1.75,
        "CC_THRESHOLD": 0.05,
        "CURVATURE_THRESHOLD": 40.0,
        "MAX_CURVATURE": 39.5,
        "SELF_INTERSECTING": False,
        "iterations": 3,
    }
    results.update(overrides)
    return results


def _stage2_ondevice_quality_case(**overrides):
    comparison = {
        "optimizer_backend": "ondevice",
        "jax_objective_not_worse_than_cpu": True,
        "jax_field_error_not_worse_than_cpu": True,
        "jax_curve_length_within_target": True,
        "jax_cc_distance_within_threshold": True,
        "jax_curvature_not_worse_than_cpu": True,
        "jax_self_intersecting": False,
    }
    comparison.update(overrides)
    return _stage2_e2e_comparison_case(**comparison)


def test_stage2_e2e_comparison_keeps_field_error_as_hard_gate():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_e2e_comparison_case(
            matched_cpu_state={
                "objective_rel_diff": 1e-12,
                "field_error_rel_diff": 2e-4,
                "gradient_allclose": True,
                "gradient_l2_rel_diff": 1e-12,
            }
        )
    )

    assert any(
        "Matched CPU-final field diagnostic parity too large" in failure
        for failure in failures
    )


def test_stage2_e2e_comparison_disables_short_run_geometry_gate_when_matched_state_checks_pass():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_e2e_comparison_case(
            max_geometry_pointwise_rel=3e-6,
            geometry_rel_tol=None,
        )
    )

    assert failures == []


def test_stage2_e2e_comparison_keeps_geometry_report_only_for_short_run_smoke():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_e2e_comparison_case(
            max_geometry_pointwise_rel=6e-6,
            geometry_rel_tol=None,
        )
    )

    assert failures == []


def test_stage2_e2e_comparison_still_rejects_large_geometry_drift_once_gate_is_enabled():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_e2e_comparison_case(
            max_geometry_pointwise_rel=6e-6,
            geometry_rel_tol=5e-6,
        )
    )

    assert any(
        "Final banana-coil geometry drift too large" in failure for failure in failures
    )


def test_stage2_e2e_comparison_rejects_matched_state_gradient_mismatch():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_e2e_comparison_case(
            matched_jax_state={
                "objective_rel_diff": 1e-12,
                "field_error_rel_diff": 1e-12,
                "gradient_allclose": False,
                "gradient_l2_rel_diff": 1e-3,
                "worst_gradient_term": {
                    "name": "curvature_barrier",
                    **_stage2_gradient_term_case(
                        gradient_allclose=False,
                        gradient_componentwise_allclose=False,
                        gradient_global_scale_match=False,
                        gradient_l2_rel_diff=1e-3,
                        gradient_max_abs_diff=2e-4,
                        gradient_scaled_atol=5e-12,
                    ),
                },
            }
        )
    )

    assert any(
        "Matched JAX-final gradient parity failed" in failure
        and "curvature_barrier" in failure
        for failure in failures
    )


def test_stage2_e2e_comparison_accepts_ondevice_solution_quality_without_geometry_parity():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(geometry_rel_tol=None)
    )

    assert failures == []


def test_stage2_e2e_comparison_labels_ondevice_failures_against_cpu_ondevice_lane():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            jax_objective_not_worse_than_cpu=False,
            jax_final_objective=1.2,
            cpu_final_objective=1.0,
            cpu_lane_label="CPU ondevice lane",
        )
    )

    assert any("CPU ondevice lane" in failure for failure in failures)


def test_stage2_e2e_comparison_rejects_ondevice_matched_state_gradient_mismatch():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            geometry_rel_tol=None,
            matched_jax_state={
                "objective_rel_diff": 1e-12,
                "field_error_rel_diff": 1e-12,
                "gradient_allclose": False,
                "gradient_l2_rel_diff": 1e-3,
                "worst_gradient_term": {
                    "name": "curvature_barrier",
                    **_stage2_gradient_term_case(
                        gradient_allclose=False,
                        gradient_componentwise_allclose=False,
                        gradient_global_scale_match=False,
                        gradient_l2_rel_diff=1e-3,
                        gradient_max_abs_diff=2e-4,
                        gradient_scaled_atol=5e-12,
                    ),
                },
            },
        )
    )

    assert any(
        "Matched JAX-final gradient parity failed" in failure
        and "curvature_barrier" in failure
        for failure in failures
    )


def test_stage2_e2e_comparison_rejects_ondevice_geometry_drift_when_explicit_gate_enabled():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            max_geometry_pointwise_rel=6e-6,
            geometry_rel_tol=5e-6,
        )
    )

    assert any(
        "Final banana-coil geometry drift too large" in failure for failure in failures
    )


def test_stage2_e2e_comparison_rejects_ondevice_constraint_violation():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            jax_cc_distance_within_threshold=False,
            jax_final_cc_distance=0.04,
            cc_threshold=0.05,
        )
    )

    assert any("configured threshold" in failure for failure in failures)


def test_stage2_gradient_parity_accepts_global_scale_match_near_barrier():
    cpu_grad = np.asarray([0.0, 3.0e7, -4.0e7], dtype=float)
    jax_grad = np.asarray([1.55e-5, 3.0e7, -4.0e7], dtype=float)

    metrics = stage2_e2e_comparison_module._build_gradient_parity_metrics(
        cpu_grad,
        jax_grad,
    )

    assert metrics["gradient_componentwise_allclose"] is False
    assert metrics["gradient_global_scale_match"] is True
    assert metrics["gradient_allclose"] is True
    assert metrics["gradient_l2_rel_diff"] < 1e-9


def test_stage2_gradient_parity_rejects_material_global_mismatch():
    cpu_grad = np.asarray([0.0, 3.0e7, -4.0e7], dtype=float)
    jax_grad = np.asarray([0.0, 3.0e7 + 1.0, -4.0e7], dtype=float)

    metrics = stage2_e2e_comparison_module._build_gradient_parity_metrics(
        cpu_grad,
        jax_grad,
    )

    assert metrics["gradient_global_scale_match"] is False
    assert metrics["gradient_allclose"] is False
    assert metrics["gradient_max_abs_diff"] == pytest.approx(1.0)


def test_stage2_e2e_probe_threads_optimizer_backend_to_both_probe_lanes(
    monkeypatch,
    tmp_path,
):
    args = argparse.Namespace(
        optimizer_backend="ondevice",
        nphi=31,
        ntheta=16,
        equilibrium_path=None,
        plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
        equilibria_dir=str(tmp_path / "equilibria"),
    )
    observed_commands: list[list[str]] = []

    monkeypatch.setattr(
        stage2_e2e_comparison_module,
        "_stage2_script_path",
        lambda: tmp_path / "driver.py",
    )
    monkeypatch.setattr(
        stage2_e2e_comparison_module,
        "write_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stage2_e2e_comparison_module,
        "load_json",
        lambda _path: {},
    )

    def fake_run_python_script(_script_path, command, **_kwargs):
        observed_commands.append(list(command))
        return argparse.Namespace(stdout="", stderr="")

    monkeypatch.setattr(
        stage2_e2e_comparison_module,
        "run_python_script",
        fake_run_python_script,
    )

    stage2_e2e_comparison_module._run_stage2_probe(
        args,
        "cpu",
        platform="cpu",
        dofs=[0.1, -0.2],
    )
    stage2_e2e_comparison_module._run_stage2_probe(
        args,
        "jax",
        platform="cpu",
        dofs=[0.1, -0.2],
    )

    assert len(observed_commands) == 2
    for command in observed_commands:
        optimizer_flag_index = command.index("--optimizer-backend")
        assert command[optimizer_flag_index + 1] == "ondevice"


def test_stage2_e2e_ondevice_endpoint_lane_uses_jax_cpu_reference():
    assert stage2_e2e_comparison_module._resolve_stage2_endpoint_cpu_lane(
        "ondevice"
    ) == (
        "jax",
        "cpu",
        "cpu-ondevice",
    )
    assert stage2_e2e_comparison_module._resolve_stage2_endpoint_cpu_lane("scipy") == (
        "cpu",
        "auto",
        "cpu-reference",
    )


def test_stage2_e2e_matched_state_probes_follow_resolved_cpu_lane(monkeypatch):
    args = argparse.Namespace()
    observed_calls: list[tuple[str, str, tuple[float, ...]]] = []

    def fake_run_stage2_probe(_args, backend, *, platform, dofs):
        observed_calls.append((backend, platform, tuple(dofs)))
        return {"backend": backend, "platform": platform}

    monkeypatch.setattr(
        stage2_e2e_comparison_module,
        "_run_stage2_probe",
        fake_run_stage2_probe,
    )

    probes = stage2_e2e_comparison_module._run_stage2_matched_state_probes(
        args,
        cpu_backend="jax",
        cpu_platform="cpu",
        jax_platform="cuda",
        dofs=[0.1, -0.2],
    )

    assert probes["cpu"] == {"backend": "jax", "platform": "cpu"}
    assert probes["jax"] == {"backend": "jax", "platform": "cuda"}
    assert observed_calls == [
        ("jax", "cpu", (0.1, -0.2)),
        ("jax", "cuda", (0.1, -0.2)),
    ]


def test_stage2_e2e_payload_preserves_trajectory_and_timing_artifacts():
    provenance = {"title": "Stage 2 end-to-end comparison"}
    cpu_case = {
        "results": _stage2_e2e_results_case(),
        "trajectory": [
            {
                "J": 2.0,
                "Jf": 0.2,
                "mean_abs_relBfinal_norm": 0.2,
                "curve_length": 1.0,
                "coil_coil_distance": 1.0,
                "curvature": 1.0,
                "grad_norm": 1.0,
            },
            {
                "J": 1.0,
                "Jf": 0.1,
                "mean_abs_relBfinal_norm": 0.1,
                "curve_length": 1.0,
                "coil_coil_distance": 1.0,
                "curvature": 1.0,
                "grad_norm": 0.5,
            },
        ],
        "elapsed_s": 12.5,
    }
    jax_case = {
        "results": _stage2_e2e_results_case(
            FINAL_OBJECTIVE=1.0 + 1e-7,
            FIELD_ERROR=0.01 + 1e-7,
            optimizer_backend="ondevice",
        ),
        "optimizer_timings": {
            "cold_run_s": 9.5,
            "warm_run_s": 3.25,
            "compile_overhead_s": 6.25,
        },
        "trajectory": [
            {
                "J": 2.0,
                "Jf": 0.2,
                "mean_abs_relBfinal_norm": 0.2,
                "curve_length": 1.0,
                "coil_coil_distance": 1.0,
                "curvature": 1.0,
                "grad_norm": 1.0,
            },
            {
                "J": 1.0 + 1e-7,
                "Jf": 0.1,
                "mean_abs_relBfinal_norm": 0.1,
                "curve_length": 1.0,
                "coil_coil_distance": 1.0,
                "curvature": 1.0,
                "grad_norm": 0.5,
            },
        ],
        "elapsed_s": 12.75,
    }

    payload = build_stage2_e2e_payload(
        provenance,
        cpu_case,
        jax_case,
        {
            "cpu": _stage2_probe_payload_case(),
            "jax": _stage2_probe_payload_case(),
        },
        {
            "cpu": _stage2_probe_payload_case(),
            "jax": _stage2_probe_payload_case(),
        },
        cpu_lane_kind="cpu-ondevice",
        geometry_rel_tol=5e-6,
    )

    assert payload["passed"] is True
    assert payload["status"] == "passed"
    assert payload["cpu"] == {
        "elapsed_s": pytest.approx(12.5),
        "iterations": 3,
    }
    assert payload["jax"] == {
        "elapsed_s": pytest.approx(9.5),
        "iterations": 3,
        "optimizer_backend": "ondevice",
    }
    assert payload["ondevice_metrics"]["jax_final_objective"] == pytest.approx(
        1.0 + 1e-7
    )
    assert payload["cpu_trajectory"] == cpu_case["trajectory"]
    assert payload["jax_trajectory"] == jax_case["trajectory"]
    assert payload["comparison"]["cpu_elapsed_s"] == pytest.approx(12.5)
    assert payload["comparison"]["jax_elapsed_s"] == pytest.approx(9.5)
    assert payload["comparison"]["cpu_lane_kind"] == "cpu-ondevice"
    assert payload["comparison"]["cpu_lane_label"] == "CPU ondevice lane"
    assert payload["comparison"]["matched_cpu_state"]["gradient_allclose"] is True
    assert payload["comparison"]["matched_jax_state"]["gradient_allclose"] is True
    assert payload["timings"]["cpu_outer_elapsed_s"] == pytest.approx(12.5)
    assert payload["timings"]["jax_outer_elapsed_s"] == pytest.approx(12.75)
    assert payload["timings"]["jax_primary_elapsed_s"] == pytest.approx(9.5)
    assert payload["timings"]["jax_optimizer_cold_run_s"] == pytest.approx(9.5)
    assert payload["timings"]["jax_optimizer_warm_run_s"] == pytest.approx(3.25)
    assert payload["timings"]["jax_optimizer_compile_overhead_s"] == pytest.approx(6.25)


def test_stage2_e2e_payload_allows_intentional_barrier_rejection_entries():
    provenance = {"title": "Stage 2 end-to-end comparison"}
    barrier_entry = {
        "J": np.inf,
        "Jf": 0.2,
        "mean_abs_relBfinal_norm": 0.2,
        "curve_length": 1.0,
        "coil_coil_distance": 0.04,
        "curvature": 1.0,
        "grad_norm": np.nan,
        "distance_constraint_violated": True,
    }
    converged_entry = {
        "J": 1.0,
        "Jf": 0.1,
        "mean_abs_relBfinal_norm": 0.1,
        "curve_length": 1.0,
        "coil_coil_distance": 0.06,
        "curvature": 1.0,
        "grad_norm": 0.5,
        "distance_constraint_violated": False,
    }
    cpu_case = {
        "results": _stage2_e2e_results_case(),
        "trajectory": [barrier_entry, converged_entry],
        "elapsed_s": 12.5,
    }
    jax_case = {
        "results": _stage2_e2e_results_case(
            FINAL_OBJECTIVE=1.0 + 1e-7,
            FIELD_ERROR=0.01 + 1e-7,
            optimizer_backend="ondevice",
        ),
        "optimizer_timings": {
            "cold_run_s": 9.5,
            "warm_run_s": 3.25,
            "compile_overhead_s": 6.25,
        },
        "trajectory": [converged_entry],
        "elapsed_s": 12.75,
    }

    payload = build_stage2_e2e_payload(
        provenance,
        cpu_case,
        jax_case,
        {
            "cpu": _stage2_probe_payload_case(),
            "jax": _stage2_probe_payload_case(),
        },
        {
            "cpu": _stage2_probe_payload_case(),
            "jax": _stage2_probe_payload_case(),
        },
        cpu_lane_kind="cpu-reference",
        geometry_rel_tol=5e-6,
    )

    assert payload["comparison"]["cpu_trajectory_finite"] is True
    assert payload["comparison"]["jax_trajectory_finite"] is True


def test_safe_speedup_returns_ratio_for_positive_times():
    assert safe_speedup(10.0, 2.0) == pytest.approx(5.0)


def test_safe_speedup_rejects_missing_or_nonpositive_candidate():
    assert safe_speedup(10.0, None) is None
    assert safe_speedup(10.0, 0.0) is None


def test_summarize_pair_probe_records_speedup():
    summary = summarize_pair_probe(
        name="tier2_stage2_e2e",
        payload={"passed": True},
        outer_elapsed_s=50.0,
        cpu_elapsed_s=20.0,
        lane_elapsed_s=5.0,
        lane_label="jax-cuda",
    )

    assert summary["passed"] is True
    assert summary["outer_elapsed_s"] == pytest.approx(50.0)
    assert summary["speedup_vs_cpu"] == pytest.approx(4.0)


def test_summarize_single_lane_probe_keeps_outer_elapsed():
    summary = summarize_single_lane_probe(
        name="tier4_adjoint_fd",
        payload={"passed": True},
        outer_elapsed_s=12.0,
        lane_label="jax-cpu",
    )

    assert summary == {
        "name": "tier4_adjoint_fd",
        "passed": True,
        "outer_elapsed_s": pytest.approx(12.0),
        "lane_label": "jax-cpu",
    }


def _adjoint_validation_metrics(**overrides):
    metrics = {
        "adjoint_residual_rel": ADJOINT_RESIDUAL_REL_TOL / 10.0,
        "implicit_gradient_finite": True,
        "implicit_gradient_norm": 1.0,
        "total_gradient_finite": True,
        "total_gradient_norm": 2.0,
        "recomposed_total_rel": RECOMPOSED_TOTAL_REL_TOL / 10.0,
        "fd_samples": [
            {
                "sample_index": 0,
                "accepted": True,
                "rel_err": FIXED_SURFACE_FD_REL_TOL / 10.0,
                "abs_err": FIXED_SURFACE_FD_ABS_TOL / 10.0,
            }
        ],
        "stable_resolve_fd_samples": 1,
        "min_stable_resolve_fd_samples": 1,
        "full_resolve_fd_samples": [
            {
                "sample_index": 0,
                "stable": True,
                "accepted": True,
                "rel_err": FULL_RESOLVE_FD_REL_TOL / 10.0,
                "abs_err": FULL_RESOLVE_FD_ABS_TOL / 10.0,
            }
        ],
    }
    metrics.update(overrides)
    return metrics


def test_evaluate_adjoint_validation_accepts_stable_metrics():
    failures = evaluate_adjoint_validation(_adjoint_validation_metrics())

    assert failures == []


def test_evaluate_adjoint_validation_reports_real_contract_failures():
    failures = evaluate_adjoint_validation(
        _adjoint_validation_metrics(
            adjoint_residual_rel=ADJOINT_RESIDUAL_REL_TOL * 10.0,
            implicit_gradient_finite=False,
            implicit_gradient_norm=0.0,
            total_gradient_finite=False,
            total_gradient_norm=0.0,
            recomposed_total_rel=RECOMPOSED_TOTAL_REL_TOL * 10.0,
            fd_samples=[
                {
                    "sample_index": 0,
                    "accepted": False,
                    "rel_err": FIXED_SURFACE_FD_REL_TOL * 10.0,
                    "abs_err": FIXED_SURFACE_FD_ABS_TOL * 10.0,
                }
            ],
            stable_resolve_fd_samples=1,
            min_stable_resolve_fd_samples=2,
            full_resolve_fd_samples=[
                {
                    "sample_index": 0,
                    "stable": True,
                    "accepted": False,
                    "rel_err": FULL_RESOLVE_FD_REL_TOL * 10.0,
                    "abs_err": FULL_RESOLVE_FD_ABS_TOL * 10.0,
                },
                {
                    "sample_index": 1,
                    "stable": False,
                    "accepted": False,
                    "plus_reason": "branch_switch",
                    "minus_reason": "branch_switch",
                },
            ],
        )
    )

    assert any("Adjoint solve residual too large" in failure for failure in failures)
    assert any(
        "Implicit correction produced NaN/inf" in failure for failure in failures
    )
    assert any(
        "Implicit correction produced zero gradient" in failure for failure in failures
    )
    assert any(
        "Total reduced gradient produced NaN/inf" in failure for failure in failures
    )
    assert any("Total reduced gradient is zero" in failure for failure in failures)
    assert any(
        "Direct-minus-implicit recomposition drift too large" in failure
        for failure in failures
    )
    assert any(
        "Fixed-surface FD sample 0 exceeded tolerance" in failure
        for failure in failures
    )
    assert any(
        "Only 1 stable full re-solve FD samples were found; need at least 2." in failure
        for failure in failures
    )
    assert any(
        "Full re-solve FD sample 0 exceeded tolerance" in failure
        for failure in failures
    )


def test_evaluate_adjoint_validation_rejects_empty_fd_samples():
    failures = evaluate_adjoint_validation(
        _adjoint_validation_metrics(
            fd_samples=[],
            stable_resolve_fd_samples=0,
            min_stable_resolve_fd_samples=2,
            full_resolve_fd_samples=[],
        )
    )

    assert "No fixed-surface FD samples were evaluated." in failures
    assert "No full re-solve FD samples were evaluated." in failures
    assert (
        "Only 0 stable full re-solve FD samples were found; need at least 2."
        in failures
    )
