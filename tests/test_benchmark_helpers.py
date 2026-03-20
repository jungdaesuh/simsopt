import argparse
import json
from pathlib import Path
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmarks.single_stage_init_parity as single_stage_init_parity_module
from benchmarks.benchmark_config import DEFAULT_CONFIGS, resolve_configs
from benchmarks.benchmark_problem import (
    build_ls_parity_problem,
    build_synthetic_boozer_problem,
    clone_tensor_surface,
)
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.single_stage_init_parity import (
    DEFAULT_STAGE2_BS_PATH,
    FIELD_ERROR_REL_TOL,
    IOTA_ABS_TOL,
    SURFACE_GEOMETRY_REL_TOL,
    VOLUME_REL_TOL,
    evaluate_single_stage_init_parity,
)
from benchmarks.stage2_e2e_comparison import evaluate_stage2_e2e_comparison
from benchmarks.validation_ladder_common import (
    max_pointwise_geometry_drift,
    repo_pythonpath_env,
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

    def fake_load_surface_gamma(surface_json_path: str) -> np.ndarray:
        path = Path(surface_json_path)
        observed_paths.append(path)
        assert path.exists()
        return np.zeros((2, 2, 3))

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_load_surface_gamma",
        fake_load_surface_gamma,
    )

    payload = single_stage_init_parity_module._run_single_stage_case(
        args,
        "cpu",
        platform="cpu",
    )

    assert observed_paths
    np.testing.assert_allclose(payload["surface_gamma"], np.zeros((2, 2, 3)))


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
