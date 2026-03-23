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
    RECOMPOSED_TOTAL_REL_TOL,
    evaluate_adjoint_validation,
)
import benchmarks.single_stage_init_parity as single_stage_init_parity_module
from benchmarks.benchmark_config import DEFAULT_CONFIGS, resolve_configs
from benchmarks.benchmark_problem import (
    build_ls_parity_problem,
    build_synthetic_boozer_problem,
    clone_tensor_surface,
)
import benchmarks.run_code_benchmark_common as run_code_benchmark_common
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.single_stage_init_parity import (
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


def test_short_run_geometry_rel_tolerance_relaxes_20_iter_smoke_runs():
    assert short_run_geometry_rel_tolerance(20) == pytest.approx(5e-6)
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


def test_apply_compilation_cache_policy_honors_explicit_cache_dir(monkeypatch, tmp_path):
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

    assert tol_20["geometry_rel_tol"] == pytest.approx(5e-6)
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

    with pytest.raises(RuntimeError, match="Subprocess failed with exit code 3") as excinfo:
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
    }
    jax_results = {
        "FINAL_IOTA": 0.1505,
        "FINAL_VOLUME": 0.10000005,
        "FIELD_ERROR": 0.0030002,
        "MAX_CURVATURE": 12.1,
        "SELF_INTERSECTING": False,
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


def test_single_stage_init_parity_reports_real_gate_failures():
    cpu_results = {
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.10,
        "FIELD_ERROR": 0.003,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": False,
    }
    jax_results = {
        "FINAL_IOTA": 0.17,
        "FINAL_VOLUME": 0.101,
        "FIELD_ERROR": 0.004,
        "MAX_CURVATURE": 10.0,
        "SELF_INTERSECTING": True,
    }

    _, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=1e-4,
        max_surface_geometry_rel=2e-5,
    )

    assert any("Final iota disagreement too large" in failure for failure in failures)
    assert any("Final volume relative difference too large" in failure for failure in failures)
    assert any("Final field error relative difference too large" in failure for failure in failures)
    assert any("Initial Boozer surface geometry drift too large" in failure for failure in failures)
    assert any("self-intersecting" in failure for failure in failures)


def test_single_stage_init_case_loads_surface_before_tempdir_cleanup(monkeypatch, tmp_path):
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
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    monkeypatch.setattr(single_stage_init_parity_module, "_single_stage_script_path", lambda: tmp_path / "driver.py")
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

    monkeypatch.setattr(single_stage_init_parity_module, "find_single_file", fake_find_single_file)
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

    def fake_load_surface_artifacts(surface_json_path: str) -> tuple[np.ndarray, bool]:
        path = Path(surface_json_path)
        observed_paths.append(path)
        assert path.exists()
        return np.zeros((2, 2, 3)), True

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_load_surface_artifacts",
        fake_load_surface_artifacts,
    )

    payload = single_stage_init_parity_module._run_single_stage_case(
        args,
        "cpu",
        platform="cpu",
    )

    assert observed_paths
    np.testing.assert_allclose(payload["surface_gamma"], np.zeros((2, 2, 3)))
    assert payload["results"]["SELF_INTERSECTING"] is True


def test_stage2_e2e_comparison_keeps_field_error_as_hard_gate():
    failures = evaluate_stage2_e2e_comparison(
        {
            "final_objective_rel_diff": 1e-7,
            "field_error_rel_diff": 2e-4,
            "field_error_rel_tol": 1e-4,
            "max_geometry_pointwise_rel": 2e-6,
            "geometry_rel_tol": 5e-6,
            "cpu_trajectory_finite": True,
            "jax_trajectory_finite": True,
            "cpu_trajectory_improves": True,
            "jax_trajectory_improves": True,
        }
    )

    assert any("Final field error relative difference too large" in failure for failure in failures)


def test_stage2_e2e_comparison_relaxes_short_run_geometry_gate():
    failures = evaluate_stage2_e2e_comparison(
        {
            "final_objective_rel_diff": 1e-7,
            "field_error_rel_diff": 1e-7,
            "field_error_rel_tol": 1e-4,
            "max_geometry_pointwise_rel": 3e-6,
            "geometry_rel_tol": 5e-6,
            "cpu_trajectory_finite": True,
            "jax_trajectory_finite": True,
            "cpu_trajectory_improves": True,
            "jax_trajectory_improves": True,
        }
    )

    assert failures == []


def test_stage2_e2e_comparison_still_rejects_large_geometry_drift():
    failures = evaluate_stage2_e2e_comparison(
        {
            "final_objective_rel_diff": 1e-7,
            "field_error_rel_diff": 1e-7,
            "field_error_rel_tol": 1e-4,
            "max_geometry_pointwise_rel": 6e-6,
            "geometry_rel_tol": 5e-6,
            "cpu_trajectory_finite": True,
            "jax_trajectory_finite": True,
            "cpu_trajectory_improves": True,
            "jax_trajectory_improves": True,
        }
    )

    assert any("Final banana-coil geometry drift too large" in failure for failure in failures)


def test_stage2_e2e_payload_preserves_trajectory_and_timing_artifacts():
    provenance = {"title": "Stage 2 end-to-end comparison"}
    cpu_case = {
        "results": {
            "FINAL_OBJECTIVE": 1.0,
            "FIELD_ERROR": 0.01,
            "FINAL_BANANA_GAMMA": [[[1.0, 0.0, 0.0]]],
            "iterations": 3,
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
        "results": {
            "FINAL_OBJECTIVE": 1.0 + 1e-7,
            "FIELD_ERROR": 0.01 + 1e-7,
            "FINAL_BANANA_GAMMA": [[[1.0, 0.0, 0.0]]],
            "iterations": 3,
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
        "elapsed_s": 8.75,
    }

    payload = build_stage2_e2e_payload(
        provenance,
        cpu_case,
        jax_case,
        geometry_rel_tol=5e-6,
    )

    assert payload["passed"] is True
    assert payload["cpu_trajectory"] == cpu_case["trajectory"]
    assert payload["jax_trajectory"] == jax_case["trajectory"]
    assert payload["comparison"]["cpu_elapsed_s"] == pytest.approx(12.5)
    assert payload["comparison"]["jax_elapsed_s"] == pytest.approx(8.75)


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


def test_evaluate_adjoint_validation_accepts_stable_metrics():
    failures = evaluate_adjoint_validation(
        {
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
        }
    )

    assert failures == []


def test_evaluate_adjoint_validation_reports_real_contract_failures():
    failures = evaluate_adjoint_validation(
        {
            "adjoint_residual_rel": ADJOINT_RESIDUAL_REL_TOL * 10.0,
            "implicit_gradient_finite": False,
            "implicit_gradient_norm": 0.0,
            "total_gradient_finite": False,
            "total_gradient_norm": 0.0,
            "recomposed_total_rel": RECOMPOSED_TOTAL_REL_TOL * 10.0,
            "fd_samples": [
                {
                    "sample_index": 0,
                    "accepted": False,
                    "rel_err": FIXED_SURFACE_FD_REL_TOL * 10.0,
                    "abs_err": FIXED_SURFACE_FD_ABS_TOL * 10.0,
                }
            ],
        }
    )

    assert any("Adjoint solve residual too large" in failure for failure in failures)
    assert any("Implicit correction produced NaN/inf" in failure for failure in failures)
    assert any("Implicit correction produced zero gradient" in failure for failure in failures)
    assert any("Total reduced gradient produced NaN/inf" in failure for failure in failures)
    assert any("Total reduced gradient is zero" in failure for failure in failures)
    assert any(
        "Direct-minus-implicit recomposition drift too large" in failure
        for failure in failures
    )
    assert any(
        "Fixed-surface FD sample 0 exceeded tolerance" in failure
        for failure in failures
    )


def test_evaluate_adjoint_validation_rejects_empty_fd_samples():
    failures = evaluate_adjoint_validation(
        {
            "adjoint_residual_rel": ADJOINT_RESIDUAL_REL_TOL / 10.0,
            "implicit_gradient_finite": True,
            "implicit_gradient_norm": 1.0,
            "total_gradient_finite": True,
            "total_gradient_norm": 2.0,
            "recomposed_total_rel": RECOMPOSED_TOTAL_REL_TOL / 10.0,
            "fd_samples": [],
        }
    )

    assert failures == ["No fixed-surface FD samples were evaluated."]
