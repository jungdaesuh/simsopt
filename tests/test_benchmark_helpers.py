import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import math
import types

import numpy as np
import pytest
from simsopt.geo.surface import Surface
from simsopt.geo.surfaceobjectives_jax import _canonicalize_traceable_exact_quadrature

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.adjoint_fd_validation import (
    ADJOINT_RESIDUAL_REL_TOL,
    FIXED_SURFACE_FD_ABS_TOL,
    FIXED_SURFACE_FD_REL_TOL,
    FULL_RESOLVE_FD_ABS_TOL,
    FULL_RESOLVE_FD_REL_TOL,
    RECOMPOSED_TOTAL_REL_TOL,
    compute_direct_and_total_gradients,
    evaluate_adjoint_validation,
)
import benchmarks.adjoint_probe_common as adjoint_probe_common
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
import benchmarks.single_stage_outer_loop_probe as single_stage_outer_loop_probe
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
import benchmarks.stage2_value_gradient_parity as stage2_value_gradient_parity_module
import benchmarks.tier5_performance_characterization as tier5_performance_characterization
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_PLASMA_SURF_FILENAME,
)
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
    build_tier5_performance_contract,
    safe_speedup,
    summarize_informational_pair_probe,
    summarize_pair_probe,
    summarize_stage2_e2e_performance_probe,
    summarize_single_lane_probe,
)
from benchmarks.validation_ladder_common import (
    _JAX_COMPILATION_CACHE_ENV_VAR,
    _SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR,
    _SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR,
    _TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR,
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    apply_compilation_cache_policy,
    build_provenance,
    describe_compile_behavior,
    evaluate_tier5_performance_budget,
    grouped_adjoint_memory_budget,
    max_pointwise_geometry_drift,
    optimizer_drift_tolerances,
    repo_pythonpath_env,
    require_requested_platform_runtime,
    require_x64_runtime,
    resolve_probe_lane,
    run_python_script,
    short_run_stage2_final_objective_rel_tolerance,
    short_run_geometry_rel_tolerance,
    single_stage_proof_contract,
    tier5_performance_budget,
)


def _load_benchmark_module(name: str, relpath: str):
    module_path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


biot_savart_kernel_scaling = _load_benchmark_module(
    "biot_savart_kernel_scaling",
    "benchmarks/biot_savart_kernel_scaling.py",
)
gpu_benchmark_module = _load_benchmark_module(
    "gpu_benchmark_module",
    "benchmarks/gpu_benchmark.py",
)
jax_ci_contract = _load_benchmark_module(
    "jax_ci_contract",
    "scripts/jax_ci_contract.py",
)


def _single_stage_cuda_runtime_available() -> bool:
    try:
        devices = single_stage_init_parity_module.jax.devices("gpu")
    except RuntimeError:
        return False
    return bool(devices)


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


def test_biot_savart_kernel_scaling_payload_includes_tuning(monkeypatch):
    monkeypatch.setattr(
        biot_savart_kernel_scaling,
        "_make_fixture",
        lambda case, seed: (
            np.zeros((case.npoints, 3)),
            np.zeros((case.ncoils, case.nquad, 3)),
            np.zeros((case.ncoils, case.nquad, 3)),
            np.ones((case.ncoils,)),
        ),
    )
    monkeypatch.setattr(
        biot_savart_kernel_scaling,
        "_measure_kernel",
        lambda fn, *args, warmup, repeat: {
            "compile_s": 0.1,
            "median_ms": 0.2,
            "mean_ms": 0.3,
            "shape": list(np.shape(fn(*args))),
        },
    )
    monkeypatch.setattr(
        biot_savart_kernel_scaling,
        "build_provenance",
        lambda jax_module, jaxlib_module, *, title, extra=None: {
            "title": title,
            "repo_sha": "deadbeef",
            **(extra or {}),
        },
    )

    payload = biot_savart_kernel_scaling.build_biotsavart_kernel_scaling_payload(
        title="kernel scaling",
        mode="jax_gpu_fast",
        warmup=0,
        repeat=1,
        seed=0,
        cases=(biot_savart_kernel_scaling.KernelScalingCase("mini", 2, 4, 3),),
    )

    assert payload["provenance"]["backend_mode"] == "jax_gpu_fast"
    assert payload["provenance"]["chunk_policy"] == "performance_tuned"
    assert payload["provenance"]["coil_chunk_size"] == 64
    assert payload["provenance"]["quadrature_block_size"] == 64
    assert payload["cases"][0]["label"] == "mini"
    assert payload["cases"][0]["B"]["shape"] == [3, 3]
    assert payload["cases"][0]["dA_by_dX"]["shape"] == [3, 3, 3]


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


def test_short_run_stage2_final_objective_rel_tolerance_tracks_smoke_budget():
    assert short_run_stage2_final_objective_rel_tolerance(20) == pytest.approx(5e-4)
    assert short_run_stage2_final_objective_rel_tolerance(21) == pytest.approx(1e-4)


def test_jax_ci_contract_ratchet_rel_tol_tightens_without_loosening():
    assert jax_ci_contract.ratchet_rel_tol(1e-12, 1e-14, factor=10.0) == pytest.approx(
        1e-13
    )
    assert jax_ci_contract.ratchet_rel_tol(1e-12, 1e-10, factor=10.0) == pytest.approx(
        1e-12
    )


def _mock_gpu_reduction_sum() -> np.float64:
    return np.nextafter(1.0, 2.0)


def test_jax_ci_contract_reduction_order_probe_tracks_ulp_distance(monkeypatch):
    monkeypatch.setattr(
        jax_ci_contract,
        "_cpu_reduction_sum_via_subprocess",
        lambda sample_size: 1.0,
    )
    monkeypatch.setattr(
        jax_ci_contract,
        "_sum_on_backend",
        lambda values, *, backend: _mock_gpu_reduction_sum(),
    )

    result = jax_ci_contract.probe_reduction_order(
        1000,
        target_backend="gpu",
        max_ulp=10,
    )

    assert result["cpu_sum"] == pytest.approx(1.0)
    assert result["backend_sum"] == pytest.approx(_mock_gpu_reduction_sum())
    assert result["ulp_distance"] == 1
    assert result["rel_err"] == pytest.approx(
        abs(_mock_gpu_reduction_sum() - 1.0) / 1.0
    )
    assert result["passed"] is True


def test_jax_ci_contract_reduction_order_probe_uses_cpu_subprocess_oracle(monkeypatch):
    seen_sample_sizes: list[int] = []

    monkeypatch.setattr(
        jax_ci_contract,
        "_cpu_reduction_sum_via_subprocess",
        lambda sample_size: seen_sample_sizes.append(sample_size) or 1.0,
    )
    monkeypatch.setattr(
        jax_ci_contract,
        "_sum_on_backend",
        lambda values, *, backend: (
            _mock_gpu_reduction_sum()
            if backend == "gpu"
            else pytest.fail("probe_reduction_order should not request cpu in-process")
        ),
    )

    result = jax_ci_contract.probe_reduction_order(
        2048,
        target_backend="gpu",
        max_ulp=10,
    )

    assert seen_sample_sizes == [2048]
    assert result["cpu_sum"] == pytest.approx(1.0)
    assert result["backend_sum"] == pytest.approx(_mock_gpu_reduction_sum())


def test_jax_ci_contract_same_device_probe_requires_bitwise_identity(monkeypatch):
    monkeypatch.setattr(
        jax_ci_contract,
        "_reproducibility_output",
        lambda *, backend, seed, sample_size: np.array([1.0, 2.0, 3.0]),
    )

    result = jax_ci_contract.probe_same_device_bitwise_reproducibility(
        backend="gpu",
        seed=1729,
        sample_size=1000,
    )

    assert result["bitwise_equal"] is True
    assert result["rel_err"] == pytest.approx(0.0)
    assert result["passed"] is True


def test_jax_ci_contract_payload_tracks_ratchet_and_pass_state(monkeypatch):
    monkeypatch.setattr(
        jax_ci_contract,
        "ci_reproducibility_contract",
        lambda: {
            "gpu_reduction_order_max_ulp": 10,
            "gpu_reduction_order_rel_tol": 1e-12,
            "gpu_reduction_order_sample_size": 1000,
            "gpu_reproducibility_seed": 1729,
            "gpu_reproducibility_sample_size": 1000,
            "tolerance_ratchet_factor": 10.0,
        },
    )
    monkeypatch.setattr(
        jax_ci_contract,
        "build_provenance",
        lambda jax_module, jaxlib_module, *, title, extra=None: {
            "title": title,
            "repo_sha": "deadbeef",
            "backend": "gpu",
            "devices": ["gpu:0"],
            **(extra or {}),
        },
    )
    monkeypatch.setattr(
        jax_ci_contract,
        "probe_reduction_order",
        lambda sample_size, *, target_backend, max_ulp: {
            "sample_size": sample_size,
            "cpu_sum": 1.0,
            "backend_sum": np.nextafter(1.0, 2.0),
            "rel_err": 1e-14,
            "ulp_distance": 1,
            "passed": True,
        },
    )
    monkeypatch.setattr(
        jax_ci_contract,
        "probe_same_device_bitwise_reproducibility",
        lambda *, backend, seed, sample_size: {
            "seed": seed,
            "sample_size": sample_size,
            "first": [1.0, 2.0, 3.0],
            "second": [1.0, 2.0, 3.0],
            "bitwise_equal": True,
            "rel_err": 0.0,
            "passed": True,
        },
    )

    payload = jax_ci_contract.build_ci_contract_payload(requested_platform="cuda")

    assert payload["passed"] is True
    assert payload["reduction_order"]["ulp_distance"] == 1
    assert payload["tolerance_drift"]["ratcheted_rel_tol"] == pytest.approx(1e-13)
    assert payload["same_device_bitwise_reproducibility"]["bitwise_equal"] is True


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


@pytest.mark.skipif(
    not _single_stage_cuda_runtime_available(),
    reason="requires a real CUDA JAX runtime",
)
def test_single_stage_init_parity_passes_on_real_cuda_runtime(tmp_path):
    output_json = tmp_path / "single-stage-init-cuda.json"

    run_python_script(
        Path(single_stage_init_parity_module.__file__),
        [
            "--platform",
            "cuda",
            "--optimizer-backend",
            "scipy",
            "--output-json",
            str(output_json),
        ],
        cwd=single_stage_init_parity_module.REPO_ROOT,
        env=repo_pythonpath_env(platform="cuda"),
        bootstrap_repo=True,
        stream_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["provenance"]["platform_request"] == "cuda"
    assert str(payload["provenance"]["backend"]).lower() in {"gpu", "cuda"}


@pytest.mark.skipif(
    not _single_stage_cuda_runtime_available(),
    reason="requires a real CUDA JAX runtime",
)
def test_single_stage_init_parity_ondevice_passes_on_real_cuda_runtime(tmp_path):
    output_json = tmp_path / "single-stage-init-ondevice-cuda.json"

    run_python_script(
        Path(single_stage_init_parity_module.__file__),
        [
            "--platform",
            "cuda",
            "--optimizer-backend",
            single_stage_init_parity_module.TARGET_OPTIMIZER_BACKEND,
            "--benchmark-mode",
            "--output-json",
            str(output_json),
        ],
        cwd=single_stage_init_parity_module.REPO_ROOT,
        env=repo_pythonpath_env(platform="cuda"),
        bootstrap_repo=True,
        stream_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["comparison"]["jax_outer_optimizer_method"] == TARGET_OUTER_OPTIMIZER_METHOD
    assert payload["comparison"]["jax_self_intersecting"] is False


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


def test_repo_pythonpath_env_clears_inherited_target_lane_sync(monkeypatch):
    monkeypatch.setenv(_TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR, "final-only")

    env = repo_pythonpath_env(platform="cpu")

    assert _TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR not in env


def test_repo_pythonpath_env_can_disable_compilation_cache(monkeypatch):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")

    env = repo_pythonpath_env(platform="cpu", disable_compilation_cache=True)

    assert _JAX_COMPILATION_CACHE_ENV_VAR not in env
    assert env[_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR] == "1"
    assert env[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] == "disabled"


def test_repo_pythonpath_env_clears_stale_disable_flags_when_cache_is_enabled(
    monkeypatch,
):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")
    monkeypatch.setenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, "1")
    monkeypatch.setenv(_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR, "disabled")

    env = repo_pythonpath_env(platform="cuda")

    assert env[_JAX_COMPILATION_CACHE_ENV_VAR] == "/tmp/jax-cache"
    assert _SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR not in env
    assert _SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR not in env


def test_repo_pythonpath_env_clears_backend_guardrails_when_requested(monkeypatch):
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")

    env = repo_pythonpath_env(platform="cpu", clear_backend_guardrails=True)

    assert "SIMSOPT_BACKEND_MODE" not in env
    assert "SIMSOPT_BACKEND_STRICT" not in env
    assert "SIMSOPT_JAX_TRANSFER_GUARD" not in env


def test_repo_pythonpath_env_preserves_backend_guardrails_by_default(monkeypatch):
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")

    env = repo_pythonpath_env(platform="cuda")

    assert env["SIMSOPT_BACKEND_MODE"] == "jax_gpu_fast"
    assert env["SIMSOPT_BACKEND_STRICT"] == "1"
    assert env["SIMSOPT_JAX_TRANSFER_GUARD"] == "disallow"


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

    assert tol_20["final_objective_rel_tol"] == pytest.approx(5e-4)
    assert tol_21["final_objective_rel_tol"] == pytest.approx(1e-4)
    assert tol_20["geometry_rel_tol"] is None
    assert tol_21["geometry_rel_tol"] == pytest.approx(1e-6)
    assert "final_objective_rel_tol_20_iter" not in tol_20
    assert "final_objective_rel_tol_default" not in tol_20
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


def test_require_x64_runtime_prefers_config_flag_without_array_probe():
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(jax_enable_x64=True),
        numpy=types.SimpleNamespace(
            zeros=lambda n: (_ for _ in ()).throw(AssertionError("should not probe")),
            float64=np.float64,
        ),
    )

    require_x64_runtime(fake_jax, context="Tier 5")


def test_build_provenance_includes_compilation_cache_metadata(monkeypatch):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/probe-cache")
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")
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
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common._current_sharding_metadata",
        lambda: {
            "sharding_strategy": "hybrid",
            "sharding_active": True,
            "sharding_axis_name": "d",
            "sharding_device_count": 4,
            "sharding_local_device_count": 2,
            "sharding_min_points_to_shard": 128,
            "sharding_min_pairwise_rows_to_shard": 8,
            "distributed_enabled": True,
            "distributed_initialized": True,
            "distributed_process_count": 2,
            "distributed_process_id": 1,
            "distributed_coordinator_address": "127.0.0.1:12345",
            "distributed_local_device_ids": [0, 1],
        },
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
    assert provenance["backend_mode"] == "jax_gpu_fast"
    assert provenance["backend_strict"] is True
    assert provenance["transfer_guard"] == "disallow"
    assert provenance["compilation_cache_enabled"] is True
    assert provenance["compilation_cache_dir"] == "/tmp/probe-cache"
    assert provenance["sharding_strategy"] == "hybrid"
    assert provenance["sharding_active"] is True
    assert provenance["sharding_min_pairwise_rows_to_shard"] == 8
    assert provenance["distributed_initialized"] is True


def _fake_jax_runtime(*, backend: str, devices: list[str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        default_backend=lambda: backend,
        devices=lambda: devices,
    )


def test_require_requested_platform_runtime_accepts_cuda_backend_alias():
    fake_jax = _fake_jax_runtime(
        backend="gpu",
        devices=["CudaDevice(id=0)"],
    )

    require_requested_platform_runtime(
        fake_jax,
        requested_platform="cuda",
        context="Tier 3",
    )


def test_require_requested_platform_runtime_rejects_cpu_fallback_for_cuda():
    fake_jax = _fake_jax_runtime(
        backend="cpu",
        devices=["CpuDevice(id=0)"],
    )

    with pytest.raises(
        RuntimeError,
        match="requested JAX platform 'cuda' but initialized backend 'cpu'",
    ):
        require_requested_platform_runtime(
            fake_jax,
            requested_platform="cuda",
            context="Tier 3",
        )


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
            "TIMINGS": {
                "boozer_total_s": np.float64(3.5),
                "outer_optimizer_s": np.float64(1.25),
            },
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
    assert payload["phase_timings"] == {
        "boozer_total_s": pytest.approx(3.5),
        "outer_optimizer_s": pytest.approx(1.25),
    }


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

    observed_invocations: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_invocations.append((list(command), dict(kwargs["env"])))
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
            "TIMINGS": {"boozer_total_s": 2.0},
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

    assert len(observed_invocations) == 1
    command, env = observed_invocations[0]
    assert "--init-only" not in command
    maxiter_flag_index = command.index("--maxiter")
    assert command[maxiter_flag_index + 1] == "1"
    optimizer_flag_index = command.index("--optimizer-backend")
    assert command[optimizer_flag_index + 1] == "ondevice"
    boozer_optimizer_flag_index = command.index("--boozer-optimizer-backend")
    assert command[boozer_optimizer_flag_index + 1] == "scipy"
    target_lane_sync_flag_index = command.index("--target-lane-accepted-step-sync")
    assert command[target_lane_sync_flag_index + 1] == "final-only"
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in env
    assert env[_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR] == "1"
    assert env[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] == "disabled"


def test_prefix_phase_timings_adds_lane_prefix():
    assert single_stage_init_parity_module._prefix_phase_timings(
        "jax",
        {"boozer_total_s": 2.5, "outer_optimizer_s": 1.0},
    ) == {
        "jax_boozer_total_s": pytest.approx(2.5),
        "jax_outer_optimizer_s": pytest.approx(1.0),
    }


def test_single_stage_init_case_benchmark_mode_skips_surface_gamma_artifact(
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
        boozer_optimizer_backend=None,
        maxiter=1,
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    observed_command: list[str] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_command[:] = list(command)
        return argparse.Namespace(stdout="", stderr="")

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        fake_run_python_script,
    )

    def fake_find_single_file(root: str | Path, pattern: str) -> Path:
        if pattern == "surf_init.json":
            raise AssertionError("benchmark_mode should not require surf_init.json")
        path = Path(root) / pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "find_single_file",
        fake_find_single_file,
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

    payload = single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cpu",
        benchmark_mode=True,
        load_surface_gamma=False,
    )

    assert "--benchmark-mode" in observed_command
    assert payload["surface_gamma"] is None


def test_single_stage_init_case_threads_profile_target_lane_flag(monkeypatch, tmp_path):
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
        boozer_optimizer_backend=None,
        maxiter=1,
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    observed_command: list[str] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_command[:] = list(command)
        return argparse.Namespace(stdout="", stderr="")

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        fake_run_python_script,
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "find_single_file",
        lambda root, pattern: Path(root) / pattern,
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
        lambda _path: np.zeros((2, 2, 3)),
    )

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cpu",
        profile_target_lane=True,
    )

    assert "--profile-target-lane" in observed_command


def test_single_stage_init_case_threads_experimental_target_lane_flag(
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
        boozer_optimizer_backend=None,
        maxiter=1,
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    observed_command: list[str] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_command[:] = list(command)
        return argparse.Namespace(stdout="", stderr="")

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        fake_run_python_script,
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "find_single_file",
        lambda root, pattern: Path(root) / pattern,
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
        lambda _path: np.zeros((2, 2, 3)),
    )

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cpu",
        experimental_target_lane_value_and_grad=True,
    )

    assert "--experimental-target-lane-value-and-grad" in observed_command


def test_single_stage_init_case_preserves_target_lane_value_and_grad_result(
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
        boozer_optimizer_backend=None,
        maxiter=1,
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
        lambda *_args, **_kwargs: argparse.Namespace(stdout="", stderr=""),
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "find_single_file",
        lambda root, pattern: Path(root) / pattern,
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
            "target_lane_value_and_grad": True,
        },
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_load_surface_gamma_artifact",
        lambda _path: np.zeros((2, 2, 3)),
    )

    payload = single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cpu",
    )

    assert payload["results"]["target_lane_value_and_grad"] is True


def test_single_stage_init_case_pins_default_target_lane_sync_for_cpu_lane(
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
        maxiter=1,
        equilibrium_path=None,
        equilibria_dir=str(tmp_path / "equilibria"),
    )

    observed_invocations: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )
    monkeypatch.setenv(_TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR, "final-only")

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_invocations.append((list(command), dict(kwargs["env"])))
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
        "cpu",
        platform="cpu",
    )

    assert len(observed_invocations) == 1
    command, env = observed_invocations[0]
    target_lane_sync_flag_index = command.index("--target-lane-accepted-step-sync")
    assert command[target_lane_sync_flag_index + 1] == "per-accept"
    assert _TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR not in env


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


def _grouped_adjoint_snapshot(label, rss_mb, gpu_memory_mb=None):
    return {
        "label": label,
        "elapsed_s": rss_mb * 0.5,
        "rss_mb": rss_mb,
        "gpu_memory_mb": gpu_memory_mb,
    }


def _grouped_adjoint_budget(platform: str) -> dict[str, float | None]:
    return grouped_adjoint_memory_budget(
        fixture="real-single-stage-init",
        platform=platform,
    )


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


def _weekly_tier5_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "manifests"
        / "stable_hardware_weekly_tier5.json"
    )


def _workflow_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / ".github" / "workflows" / filename


def _weekly_tier5_workflow_path() -> Path:
    return _workflow_path("jax_benchmark_reporting.yml")


def _smoke_workflow_path() -> Path:
    return _workflow_path("jax_smoke.yml")


def _gpu_parity_workflow_path() -> Path:
    return _workflow_path("jax_gpu_parity.yml")


def _assert_named_benchmark_env_bootstrap(
    workflow_text: str, *, verify_python: bool = False
) -> None:
    assert "BENCHMARK_ENV_NAME: jax-0.9.2" in workflow_text
    assert "BENCHMARK_ENV_FILE: envs/jax-0.9.2.yml" in workflow_text
    assert 'grep -Fxq "$BENCHMARK_ENV_NAME"' in workflow_text
    assert (
        'conda env create -n "$BENCHMARK_ENV_NAME" -f "$BENCHMARK_ENV_FILE"'
        in workflow_text
    )
    assert (
        'conda env update -n "$BENCHMARK_ENV_NAME" -f "$BENCHMARK_ENV_FILE" --prune'
        in workflow_text
    )
    assert 'python -m pip install -e ".[JAX_GPU,dev]"' in workflow_text
    if verify_python:
        assert 'conda run --no-capture-output -n "$BENCHMARK_ENV_NAME" python -V' in workflow_text


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


def test_grouped_adjoint_helpers_require_streaming_vjp_groups():
    jr_jax = types.SimpleNamespace(
        boozer_surface=types.SimpleNamespace(
            res={
                "iota": 0.1,
                "G": 0.2,
                "vjp_groups": None,
                "vjp": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("legacy full VJP fallback should stay disabled")
                ),
            }
        )
    )

    with pytest.raises(RuntimeError, match="legacy full-pytree adjoint fallback"):
        list(adjoint_probe_common.iter_grouped_adjoint_cotangents(jr_jax, np.ones(2)))


def test_compute_adjoint_state_uses_device_native_forward_backward(monkeypatch):
    recorded = {}

    def fake_forward_backward_jax(P, L, U, rhs, iterative_refinement=False):
        recorded["args"] = (P, L, U, rhs)
        recorded["iterative_refinement"] = iterative_refinement
        return rhs

    monkeypatch.setattr(
        adjoint_probe_common,
        "forward_backward_jax",
        fake_forward_backward_jax,
    )
    p_mat = np.eye(2)
    l_mat = np.eye(2)
    u_mat = np.eye(2)
    rhs = np.array([1.0, -2.0])
    expected_spec = object()

    def fake_compute_dJ_ds(coil_set_spec, iota, G, weight_inv_modB):
        recorded["compute_dJ_ds_args"] = (coil_set_spec, iota, G, weight_inv_modB)
        return rhs

    jr_jax = types.SimpleNamespace(
        boozer_surface=types.SimpleNamespace(
            res={
                "PLU": (p_mat, l_mat, u_mat),
                "iota": 0.1,
                "G": 0.2,
                "weight_inv_modB": False,
            }
        ),
        biotsavart=types.SimpleNamespace(
            x=np.array([3.0, 4.0]),
            coil_set_spec_from_dofs=lambda coil_dofs: (
                recorded.setdefault("coil_dofs", coil_dofs.copy()),
                expected_spec,
            )[1],
        ),
        _compute_dJ_ds=fake_compute_dJ_ds,
    )

    adjoint, residual_rel = adjoint_probe_common.compute_adjoint_state(jr_jax)

    np.testing.assert_allclose(np.asarray(adjoint), rhs)
    assert residual_rel == pytest.approx(0.0)
    assert recorded["args"] == (p_mat, l_mat, u_mat, rhs)
    assert recorded["iterative_refinement"] is True
    np.testing.assert_allclose(recorded["coil_dofs"], np.array([3.0, 4.0]))
    assert recorded["compute_dJ_ds_args"] == (expected_spec, 0.1, 0.2, False)


def test_accumulate_grouped_adjoint_derivative_uses_biotsavart_projection_api(
    monkeypatch,
):
    recorded = {}

    class FakeDerivative:
        def __init__(self, blocks):
            self.blocks = tuple(blocks)

        def __iadd__(self, other):
            self.blocks = self.blocks + other.blocks
            return self

    def fake_projection(d_coil_arrays, coil_indices):
        recorded.setdefault("projection_args", []).append((d_coil_arrays, coil_indices))
        blocks = ((tuple(map(tuple, d_coil_arrays)), tuple(map(tuple, coil_indices))),)
        return FakeDerivative(blocks)

    bs_jax = types.SimpleNamespace(coil_cotangents_to_derivative=fake_projection)
    grouped = [
        (np.array([[1.0, 2.0, 3.0]]), [0, 2]),
        (np.array([[4.0, 5.0, 6.0]]), [1]),
    ]

    from simsopt._core import derivative as derivative_module

    monkeypatch.setattr(derivative_module, "Derivative", FakeDerivative)
    derivative = adjoint_probe_common.accumulate_grouped_adjoint_derivative(
        bs_jax,
        iter(grouped),
    )

    assert len(recorded["projection_args"]) == 2
    np.testing.assert_allclose(recorded["projection_args"][0][0][0], grouped[0][0])
    assert recorded["projection_args"][0][1] == [[0, 2]]
    np.testing.assert_allclose(recorded["projection_args"][1][0][0], grouped[1][0])
    assert recorded["projection_args"][1][1] == [[1]]
    assert len(derivative.blocks) == 2


def test_compute_direct_and_total_gradients_uses_live_boozer_g(monkeypatch):
    direct_gradient = np.array([7.0, 11.0])
    total_gradient = np.array([5.0, 9.0])
    implicit_correction = np.array([2.0, 2.0])
    recorded = {}

    class FakeDerivative:
        def __call__(self, _optim):
            return direct_gradient

    def fake_value_and_direct_coil_derivative(
        biotsavart,
        objective_value_and_grad,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    ):
        recorded["value_and_direct_args"] = (
            biotsavart,
            objective_value_and_grad,
            coil_dofs.copy(),
            x_inner.copy(),
            optimize_G,
            weight_inv_modB,
        )
        return 0.0, FakeDerivative()

    booz_jax = types.SimpleNamespace(
        res={"iota": 0.1, "G": 0.2, "weight_inv_modB": False},
        _get_surface_dofs=lambda: np.array([0.3, 0.4]),
    )

    def fake_inner_objective_state(iota, G, sdofs=None):
        recorded["inner_state_args"] = (iota, G, sdofs.copy())
        return np.array([1.0, 2.0, 3.0]), True

    jr_jax = types.SimpleNamespace(
        boozer_surface=booz_jax,
        dJ=lambda: total_gradient,
        _inner_objective_state=fake_inner_objective_state,
        _direct_objective_value_and_grad=object(),
    )
    bs_jax = types.SimpleNamespace(x=np.array([9.0, 8.0]))

    monkeypatch.setattr(
        "simsopt.geo.surfaceobjectives_jax._value_and_direct_coil_derivative",
        fake_value_and_direct_coil_derivative,
    )
    direct, total, recomposed_rel = compute_direct_and_total_gradients(
        jr_jax,
        bs_jax,
        implicit_correction,
    )

    np.testing.assert_allclose(direct, direct_gradient)
    np.testing.assert_allclose(total, total_gradient)
    iota, G, sdofs = recorded["inner_state_args"]
    assert iota == 0.1
    assert G == 0.2
    np.testing.assert_allclose(sdofs, np.array([0.3, 0.4]))
    (
        called_bs_jax,
        objective_value_and_grad,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    ) = recorded["value_and_direct_args"]
    assert called_bs_jax is bs_jax
    assert objective_value_and_grad is jr_jax._direct_objective_value_and_grad
    np.testing.assert_allclose(coil_dofs, np.array([9.0, 8.0]))
    np.testing.assert_allclose(x_inner, np.array([1.0, 2.0, 3.0]))
    assert optimize_G is True
    assert weight_inv_modB is False
    assert recomposed_rel == pytest.approx(0.0)


def test_grouped_adjoint_memory_probe_requires_complete_finite_metrics():
    failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(snapshots=_complete_grouped_adjoint_snapshots()),
        budget=_grouped_adjoint_budget("cpu"),
    )

    assert failures == []


def test_grouped_adjoint_memory_probe_rejects_missing_snapshots_or_nonfinite_gradient():
    failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(
            adjoint_residual_rel=np.inf,
            implicit_gradient_finite=False,
            implicit_gradient_norm=0.0,
            snapshots=[_grouped_adjoint_snapshot("start", 1.0)],
        ),
        budget=_grouped_adjoint_budget("cpu"),
    )

    assert any("required snapshots" in failure for failure in failures)
    assert any("non-finite implicit gradient" in failure for failure in failures)


def test_grouped_adjoint_memory_probe_rejects_budget_regressions():
    snapshots = _complete_grouped_adjoint_snapshots()
    snapshots[-1]["rss_mb"] = 9000.0
    snapshots[-1]["gpu_memory_mb"] = 13000.0

    failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(snapshots=snapshots),
        budget=_grouped_adjoint_budget("cuda"),
    )

    assert any("peak RSS" in failure for failure in failures)
    assert any("peak GPU memory" in failure for failure in failures)


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
        memory_budget=_grouped_adjoint_budget("cuda"),
        failures=[],
        snapshots=snapshots,
        boozer_limited_memory=True,
        boozer_limited_memory_requested=True,
        device_memory_profile_path="/tmp/grouped.prof",
    )

    assert payload["provenance"]["boozer_limited_memory"] is True
    assert payload["provenance"]["boozer_optimizer_backend"] == "ondevice"
    assert payload["baseline"]["boozer_optimizer_backend"] == "ondevice"
    assert payload["baseline"]["optimizer_method"] == "lbfgs-ondevice"
    assert payload["baseline"]["boozer_limited_memory"] is True
    assert payload["baseline"]["boozer_limited_memory_requested"] is True
    assert payload["memory"]["budget"]["max_peak_gpu_memory_mb"] == pytest.approx(12288.0)
    assert payload["memory"]["device_memory_profile_path"] == "/tmp/grouped.prof"


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
        memory_budget=_grouped_adjoint_budget("cpu"),
        failures=[],
        snapshots=snapshots,
        boozer_limited_memory=False,
        boozer_limited_memory_requested=True,
        device_memory_profile_path=None,
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


def test_grouped_adjoint_memory_budget_accepts_gpu_platform_alias():
    assert _grouped_adjoint_budget("gpu") == _grouped_adjoint_budget("cuda")


def test_tier5_single_stage_probe_args_thread_benchmark_mode():
    args = argparse.Namespace(
        platform="cuda",
        equilibrium_path=None,
        plasma_surf_filename="fixture.nc",
        equilibria_dir="/tmp/equilibria",
        stage2_bs_path="/tmp/biot_savart_opt.json",
        single_stage_nphi=63,
        single_stage_ntheta=32,
        mpol=4,
        ntor=4,
        vol_target=0.1,
        iota_target=0.15,
        optimizer_backend="ondevice",
        benchmark_mode=True,
    )

    command = tier5_performance_characterization._single_stage_init_probe_args(args)

    assert "--optimizer-backend" in command
    optimizer_backend_idx = command.index("--optimizer-backend")
    assert command[optimizer_backend_idx + 1] == "ondevice"
    assert "--benchmark-mode" in command


def test_tier5_stage2_e2e_probe_args_thread_optimizer_backend():
    args = argparse.Namespace(
        platform="cuda",
        equilibrium_path=None,
        plasma_surf_filename="fixture.nc",
        equilibria_dir="/tmp/equilibria",
        stage2_nphi=255,
        stage2_ntheta=64,
        maxiter=20,
        optimizer_backend="ondevice",
    )

    command = tier5_performance_characterization._stage2_e2e_probe_args(args)

    assert "--optimizer-backend" in command
    optimizer_backend_idx = command.index("--optimizer-backend")
    assert command[optimizer_backend_idx + 1] == "ondevice"


def test_stage2_benchmark_scripts_default_to_repo_fixture_equilibria_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage2_value_gradient_parity.py",
            "--output-json",
            str(tmp_path / "stage2-tier1.json"),
        ],
    )
    stage2_value_gradient_args = stage2_value_gradient_parity_module.parse_args()
    assert stage2_value_gradient_args.plasma_surf_filename == DEFAULT_PLASMA_SURF_FILENAME
    assert stage2_value_gradient_args.equilibria_dir == str(DEFAULT_EQUILIBRIA_DIR)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage2_e2e_comparison.py",
            "--output-json",
            str(tmp_path / "stage2-tier2.json"),
        ],
    )
    stage2_e2e_args = stage2_e2e_comparison_module.parse_args()
    assert stage2_e2e_args.plasma_surf_filename == DEFAULT_PLASMA_SURF_FILENAME
    assert stage2_e2e_args.equilibria_dir == str(DEFAULT_EQUILIBRIA_DIR)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_outer_loop_probe.py",
            "--output-json",
            str(tmp_path / "single-stage-outer-loop.json"),
        ],
    )
    single_stage_outer_loop_args = single_stage_outer_loop_probe.parse_args()
    assert (
        single_stage_outer_loop_args.plasma_surf_filename
        == DEFAULT_PLASMA_SURF_FILENAME
    )
    assert single_stage_outer_loop_args.equilibria_dir == str(DEFAULT_EQUILIBRIA_DIR)


def test_stage2_value_gradient_real_fixture_preserves_sharding_summaries(
    tmp_path,
    monkeypatch,
):
    cpu_payload = {
        "dof_count": 2,
        "equilibrium_path": "cpu.nc",
        "squared_flux": {
            "J": 1.0,
            "dJ": [0.5, -0.25],
            "grad_norm": 0.5590169943749475,
        },
        "sharding_summaries": {"field": {"kind": "SingleDeviceSharding"}},
    }
    jax_payload = {
        "dof_count": 2,
        "equilibrium_path": "jax.nc",
        "squared_flux": {
            "J": 1.0,
            "dJ": [0.5, -0.25],
            "grad_norm": 0.5590169943749475,
        },
        "sharding_summaries": {
            "pairwise_penalty": {
                "dynamic_self": {
                    "left": {"gammas": {"kind": "NamedSharding"}},
                }
            }
        },
    }

    perf_counter_values = iter((10.0, 11.0, 20.0, 22.5))
    monkeypatch.setattr(
        stage2_value_gradient_parity_module.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        stage2_value_gradient_parity_module,
        "repo_pythonpath_env",
        lambda **_kwargs: {},
    )

    def _fake_run_python_script(_script, argv, **_kwargs):
        payload = cpu_payload if argv[1] == "cpu" else jax_payload
        export_path = Path(argv[3])
        export_path.write_text(json.dumps(payload), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        stage2_value_gradient_parity_module,
        "run_python_script",
        _fake_run_python_script,
    )

    args = argparse.Namespace(
        nphi=31,
        ntheta=16,
        equilibrium_path="dummy.nc",
        plasma_surf_filename="unused.nc",
        equilibria_dir=str(tmp_path),
        platform="cpu",
    )

    results = stage2_value_gradient_parity_module.run_real_fixture(args)

    assert results["cpu"]["elapsed_s"] == pytest.approx(1.0)
    assert results["jax"]["elapsed_s"] == pytest.approx(2.5)
    assert results["cpu"]["sharding_summaries"] == cpu_payload["sharding_summaries"]
    assert results["jax"]["sharding_summaries"] == jax_payload["sharding_summaries"]


def test_weekly_tier5_manifest_targets_ondevice_benchmark_mode():
    manifest = json.loads(_weekly_tier5_manifest_path().read_text(encoding="utf-8"))
    args = manifest["commands"][0]["args"]

    assert "--optimizer-backend" in args
    optimizer_backend_idx = args.index("--optimizer-backend")
    assert args[optimizer_backend_idx + 1] == "ondevice"
    assert "--benchmark-mode" in args
    assert manifest["runtime_contract"]["backend_mode"] == "jax_gpu_fast"
    assert manifest["runtime_contract"]["strict_backend"] is True
    assert manifest["runtime_contract"]["transfer_guard"] == "log"
    assert manifest["performance_budget"]["profile"] == "stable_hardware_weekly"
    assert (
        manifest["performance_budget"]["tier2_stage2_e2e"]["min_outer_speedup_vs_cpu"]
        == pytest.approx(1.25)
    )
    assert (
        manifest["performance_budget"]["tier2_stage2_e2e"]["max_compile_overhead_s"]
        == pytest.approx(60.0)
    )
    assert (
        manifest["memory_budget"]["max_peak_gpu_memory_mb"] == pytest.approx(12288.0)
    )


def test_weekly_tier5_manifest_includes_grouped_adjoint_memory_probe_command():
    manifest = json.loads(_weekly_tier5_manifest_path().read_text(encoding="utf-8"))
    grouped_command = manifest["commands"][1]

    assert grouped_command["name"] == "grouped_adjoint_memory_probe"
    assert grouped_command["script"] == "benchmarks/grouped_adjoint_memory_probe.py"
    assert grouped_command["env"] == {
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "JAX_COMPILATION_CACHE_DIR": (
            "benchmark_artifacts/jax-compilation-cache/jax_gpu_fast-dense-audit"
        ),
    }
    assert grouped_command["args"][-2:] == [
        "--device-memory-profile-out",
        "benchmark_artifacts/grouped_adjoint_memory_profile.prof",
    ]


def test_weekly_tier5_workflow_sets_cache_and_ondevice_contract():
    workflow_text = _weekly_tier5_workflow_path().read_text(encoding="utf-8")

    _assert_named_benchmark_env_bootstrap(workflow_text, verify_python=True)
    assert "JAX_COMPILATION_CACHE_DIR" in workflow_text
    assert "SIMSOPT_BACKEND_MODE: jax_gpu_fast" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: log" in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert "jax_gpu_fast-dense-audit" in workflow_text
    assert "setuptools_scm" not in workflow_text
    assert "--optimizer-backend ondevice" in workflow_text
    assert "--benchmark-mode" in workflow_text
    assert "continue-on-error: true" in workflow_text
    assert "benchmarks/grouped_adjoint_memory_probe.py" in workflow_text
    assert "--device-memory-profile-out benchmark_artifacts/grouped_adjoint_memory_profile.prof" in workflow_text
    assert "if-no-files-found: ignore" in workflow_text
    assert "Fail on benchmark gate regressions" in workflow_text


def test_gpu_parity_workflow_enforces_strict_transfer_guard_contract():
    workflow_text = _gpu_parity_workflow_path().read_text(encoding="utf-8")

    _assert_named_benchmark_env_bootstrap(workflow_text)
    assert "SIMSOPT_BACKEND_MODE: jax_gpu_parity" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert "setuptools_scm" not in workflow_text
    assert "benchmarks/stage2_value_gradient_parity.py" in workflow_text
    assert "--fixture real" in workflow_text
    assert "benchmarks/single_stage_outer_loop_probe.py" in workflow_text
    assert "--optimizer-backend ondevice" in workflow_text
    assert "benchmark_artifacts/stage2_value_gradient_parity_real_cuda.json" in workflow_text
    assert "benchmark_artifacts/single_stage_outer_loop_cuda.json" in workflow_text


def test_smoke_workflow_adds_cuda_e2e_target_lane_gate():
    workflow_text = _smoke_workflow_path().read_text(encoding="utf-8")
    required_paths = (
        "benchmarks/stage2_value_gradient_parity.py",
        "benchmarks/single_stage_outer_loop_probe.py",
        "benchmarks/grouped_adjoint_memory_probe.py",
        "benchmarks/tier5_performance_characterization.py",
        "benchmarks/render_benchmark_report.py",
        "benchmarks/manifests/stable_hardware_weekly_tier5.json",
        ".github/workflows/jax_gpu_parity.yml",
        ".github/workflows/jax_benchmark_reporting.yml",
    )

    _assert_named_benchmark_env_bootstrap(workflow_text)
    assert "jax-gpu-e2e:" in workflow_text
    assert "name: JAX GPU e2e smoke (CUDA, ondevice)" in workflow_text
    assert "runs-on: [self-hosted, gpu]" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert 'JAX_ENABLE_X64: "1"' in workflow_text
    assert "benchmarks/stage2_e2e_comparison.py" in workflow_text
    assert "benchmarks/single_stage_init_parity.py" in workflow_text
    assert "--platform cuda" in workflow_text
    assert "--optimizer-backend ondevice" in workflow_text
    for required_path in required_paths:
        assert required_path in workflow_text


def test_legacy_gpu_benchmark_wrapper_delegates_to_local_validation_ladder():
    wrapper_text = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_benchmark.py"
    ).read_text(encoding="utf-8")

    assert "tier5_performance_characterization.py" in wrapper_text
    assert "grouped_adjoint_memory_probe.py" in wrapper_text
    assert "render_benchmark_report.py" in wrapper_text
    assert "git clone" not in wrapper_text
    assert "https://github.com" not in wrapper_text


def test_legacy_gpu_benchmark_wrapper_applies_grouped_probe_env_override(
    monkeypatch, tmp_path
):
    artifacts_dir = tmp_path / "artifacts"
    dense_audit_cache_dir = tmp_path / "jax-compilation-cache" / "jax_gpu_fast-dense-audit"
    manifest_path = tmp_path / "stable_hardware_weekly_tier5.json"
    manifest_path.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "name": "tier5_performance_characterization",
                        "script": "benchmarks/tier5_performance_characterization.py",
                        "args": [],
                    },
                    {
                        "name": "grouped_adjoint_memory_probe",
                        "script": "benchmarks/grouped_adjoint_memory_probe.py",
                        "env": {
                            "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
                            "JAX_COMPILATION_CACHE_DIR": str(dense_audit_cache_dir),
                        },
                        "args": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        platform="cuda",
        optimizer_backend="ondevice",
        maxiter=20,
        artifacts_dir=str(artifacts_dir),
        manifest_json=str(manifest_path),
        benchmark_mode=True,
    )
    observed_invocations: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "log")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_gpu_fast-cuda")
    monkeypatch.setattr(gpu_benchmark_module, "parse_args", lambda: (args, []))

    def fake_run(command, *, cwd, check, env):
        del cwd, check
        observed_invocations.append((list(command), dict(env)))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(gpu_benchmark_module.subprocess, "run", fake_run)

    gpu_benchmark_module.main()

    assert len(observed_invocations) == 3
    tier5_command, tier5_env = observed_invocations[0]
    grouped_command, grouped_env = observed_invocations[1]
    report_command, report_env = observed_invocations[2]
    assert "tier5_performance_characterization.py" in tier5_command[1]
    assert tier5_env["SIMSOPT_JAX_TRANSFER_GUARD"] == "log"
    assert tier5_env["JAX_COMPILATION_CACHE_DIR"] == "/tmp/jax_gpu_fast-cuda"
    assert "grouped_adjoint_memory_probe.py" in grouped_command[1]
    assert grouped_env["SIMSOPT_JAX_TRANSFER_GUARD"] == "disallow"
    assert grouped_env["JAX_COMPILATION_CACHE_DIR"] == str(dense_audit_cache_dir)
    assert "render_benchmark_report.py" in report_command[1]
    assert report_env["SIMSOPT_JAX_TRANSFER_GUARD"] == "log"
    assert dense_audit_cache_dir.is_dir()


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


def test_single_stage_outer_loop_contract_matches_probe_defaults():
    contract = single_stage_proof_contract()

    assert contract["default_maxiter"] == 1
    assert contract["min_iterations"] == 1
    assert contract["required_outer_optimizer_method"] == TARGET_OUTER_OPTIMIZER_METHOD
    assert contract["required_result_keys"] == (
        "FINAL_IOTA",
        "FINAL_VOLUME",
        "FIELD_ERROR",
        "MAX_CURVATURE",
    )
    assert TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG == "tier3_single_stage_outer_loop"


def test_canonicalize_traceable_exact_quadrature_rewrites_shifted_half_period_grid():
    booz = types.SimpleNamespace(
        mpol=2,
        ntor=2,
        nfp=5,
        stellsym=True,
        quadpoints_phi=Surface.get_phi_quadpoints(nphi=31, range="half period", nfp=5),
        quadpoints_theta=Surface.get_theta_quadpoints(ntheta=16),
    )

    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz)
    )

    np.testing.assert_allclose(
        np.asarray(quadpoints_phi),
        np.linspace(0.0, 1.0 / (2.0 * booz.nfp), booz.ntor + 1, endpoint=False),
    )
    np.testing.assert_allclose(
        np.asarray(quadpoints_theta),
        np.linspace(0.0, 1.0, 2 * booz.mpol + 1, endpoint=False),
    )
    assert mask_indices.ndim == 1
    assert mask_indices.size > 0


def test_canonicalize_traceable_exact_quadrature_preserves_exact_half_period_grid():
    booz = types.SimpleNamespace(
        mpol=2,
        ntor=2,
        nfp=5,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 1.0 / (2.0 * 5.0), 3, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
    )

    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz)
    )

    np.testing.assert_allclose(np.asarray(quadpoints_phi), booz.quadpoints_phi)
    np.testing.assert_allclose(np.asarray(quadpoints_theta), booz.quadpoints_theta)
    assert mask_indices.ndim == 1
    assert mask_indices.size > 0


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
            "curvature": 39.0,
            "curvature_threshold": 40.0,
            "curvature_margin": 1.0,
            "curvature_threshold_edge_active": False,
            "gradient_allclose": True,
            "gradient_l2_rel_diff": 1e-12,
        },
        "matched_jax_state": {
            "objective_rel_diff": 1e-12,
            "field_error_rel_diff": 1e-12,
            "curvature": 39.0,
            "curvature_threshold": 40.0,
            "curvature_margin": 1.0,
            "curvature_threshold_edge_active": False,
            "gradient_allclose": True,
            "gradient_l2_rel_diff": 1e-12,
        },
    }
    comparison.update(overrides)
    return comparison


def _stage2_probe_payload_case(**overrides):
    payload = {
        "curvature_threshold": 40.0,
        "curvature_margin": 39.0,
        "composite": {
            "J": 1.0,
            "mean_abs_relBfinal_norm": 0.01,
            "curvature": 1.0,
            "dJ": [0.5, -0.25],
        },
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
        "final_objective_rel_tol": 1e-4,
        "cpu_final_objective": 1.0,
        "jax_final_objective": 1.0,
        "jax_objective_not_worse_than_cpu": True,
        "cpu_field_error": 0.01,
        "jax_field_error": 0.01,
        "jax_field_error_not_worse_than_cpu": True,
        "length_target": 1.75,
        "jax_final_curve_length": 1.0,
        "jax_curve_length_within_target": True,
        "cc_threshold": 0.05,
        "jax_final_cc_distance": 0.2,
        "jax_cc_distance_within_threshold": True,
        "curvature_threshold": 40.0,
        "cpu_max_curvature": 39.5,
        "jax_max_curvature": 39.5,
        "cpu_curvature_margin": 0.5,
        "jax_curvature_margin": 0.5,
        "cpu_curvature_threshold_edge_active": False,
        "jax_curvature_threshold_edge_active": False,
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
                    "name": "curvature_penalty",
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
        and "curvature_penalty" in failure
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
                "curvature": 39.0,
                "curvature_threshold": 40.0,
                "curvature_margin": 1.0,
                "curvature_threshold_edge_active": False,
                "gradient_allclose": False,
                "gradient_l2_rel_diff": 1e-3,
                "worst_gradient_term": {
                    "name": "curvature_penalty",
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
        and "curvature_penalty" in failure
        for failure in failures
    )


def test_stage2_e2e_comparison_accepts_threshold_edge_curvature_gradient_portability():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            geometry_rel_tol=None,
            matched_jax_state={
                "objective_rel_diff": 1e-12,
                "field_error_rel_diff": 1e-12,
                "curvature": 39.9999995,
                "curvature_threshold": 40.0,
                "curvature_margin": 5e-7,
                "curvature_threshold_edge_active": True,
                "gradient_allclose": False,
                "gradient_l2_rel_diff": 5e-6,
                "worst_gradient_term": {
                    "name": "curvature_penalty",
                    **_stage2_gradient_term_case(
                        gradient_allclose=False,
                        gradient_componentwise_allclose=False,
                        gradient_global_scale_match=False,
                        gradient_l2_rel_diff=5e-6,
                        gradient_max_abs_diff=1.3e-1,
                        gradient_scaled_atol=1.3e-4,
                    ),
                },
            },
        )
    )

    assert failures == []


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


def test_stage2_e2e_comparison_accepts_threshold_edge_objective_drift():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            final_objective_rel_diff=2.8e-4,
            final_objective_rel_tol=5e-4,
            jax_objective_not_worse_than_cpu=True,
            cpu_max_curvature=39.9999995,
            jax_max_curvature=39.9999994,
            curvature_threshold=40.0,
            cpu_curvature_margin=5e-7,
            jax_curvature_margin=6e-7,
            cpu_curvature_threshold_edge_active=True,
            jax_curvature_threshold_edge_active=True,
        )
    )

    assert failures == []


def test_stage2_e2e_comparison_accepts_curvature_exactly_at_threshold():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            cpu_max_curvature=40.0,
            jax_max_curvature=40.0,
            curvature_threshold=40.0,
            cpu_curvature_margin=0.0,
            jax_curvature_margin=0.0,
            cpu_curvature_threshold_edge_active=True,
            jax_curvature_threshold_edge_active=True,
        )
    )

    assert failures == []


def test_stage2_e2e_comparison_accepts_machine_scale_curvature_drift_when_cpu_already_violates_threshold():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            cpu_max_curvature=41.599461283993726,
            jax_max_curvature=41.59946134324458,
            curvature_threshold=40.0,
            cpu_curvature_margin=-1.5994612839937261,
            jax_curvature_margin=-1.5994613432445774,
            jax_curvature_not_worse_than_cpu=True,
        )
    )

    assert failures == []


def test_stage2_e2e_comparison_rejects_large_threshold_edge_objective_drift():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            jax_objective_not_worse_than_cpu=False,
            jax_final_objective=6009.5,
            cpu_final_objective=6000.0,
            final_objective_rel_tol=5e-4,
            cpu_max_curvature=39.9999995,
            jax_max_curvature=39.9999994,
            curvature_threshold=40.0,
            cpu_curvature_margin=5e-7,
            jax_curvature_margin=6e-7,
            cpu_curvature_threshold_edge_active=True,
            jax_curvature_threshold_edge_active=True,
        )
    )

    assert any("Final objective is worse" in failure for failure in failures)


def test_stage2_e2e_comparison_rejects_ondevice_constraint_violation():
    failures = evaluate_stage2_e2e_comparison(
        _stage2_ondevice_quality_case(
            jax_cc_distance_within_threshold=False,
            jax_final_cc_distance=0.04,
            cc_threshold=0.05,
        )
    )

    assert any("configured threshold" in failure for failure in failures)


def test_stage2_gradient_parity_accepts_global_scale_match_near_threshold():
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
    observed_invocations: list[tuple[list[str], dict[str, str]]] = []

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

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_invocations.append((list(command), dict(kwargs["env"])))
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

    assert len(observed_invocations) == 2
    for command, env in observed_invocations:
        optimizer_flag_index = command.index("--optimizer-backend")
        assert command[optimizer_flag_index + 1] == "ondevice"
        assert _JAX_COMPILATION_CACHE_ENV_VAR not in env
        assert env[_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR] == "1"
        assert env[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] == "disabled"


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


def test_stage2_e2e_probe_keeps_compilation_cache_for_cuda_lane(
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
    observed_envs: list[dict[str, str]] = []
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")

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

    def fake_run_python_script(_script_path, command, **kwargs):
        observed_envs.append(dict(kwargs["env"]))
        return argparse.Namespace(stdout="", stderr="")

    monkeypatch.setattr(
        stage2_e2e_comparison_module,
        "run_python_script",
        fake_run_python_script,
    )

    stage2_e2e_comparison_module._run_stage2_probe(
        args,
        "jax",
        platform="cuda",
        dofs=[0.1, -0.2],
    )

    assert len(observed_envs) == 1
    env = observed_envs[0]
    assert env[_JAX_COMPILATION_CACHE_ENV_VAR] == "/tmp/jax-cache"
    assert _SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR not in env
    assert _SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR not in env


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
        final_objective_rel_tol=1e-4,
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


def test_stage2_e2e_payload_allows_intentional_threshold_violation_entries():
    provenance = {"title": "Stage 2 end-to-end comparison"}
    threshold_violation_entry = {
        "J": 5.0,
        "Jf": 0.2,
        "mean_abs_relBfinal_norm": 0.2,
        "curve_length": 1.0,
        "coil_coil_distance": 0.04,
        "curvature": 1.0,
        "grad_norm": 0.75,
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
        "trajectory": [threshold_violation_entry, converged_entry],
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
        final_objective_rel_tol=1e-4,
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


def test_summarize_informational_pair_probe_marks_timing_as_non_headline():
    summary = summarize_informational_pair_probe(
        name="tier1b_real_stage2",
        payload={"passed": True},
        outer_elapsed_s=50.0,
        cpu_elapsed_s=20.0,
        lane_elapsed_s=5.0,
        lane_label="jax-cuda",
        timing_semantics="correctness_probe_only",
        recommended_question="parity",
    )

    assert summary["speedup_vs_cpu"] == pytest.approx(4.0)
    assert summary["timing_semantics"] == "correctness_probe_only"
    assert summary["recommended_question"] == "parity"
    assert summary["supports_performance_headline"] is False
    assert "headline_metric" not in summary


def _stage2_e2e_performance_probe_payload(
    *, warm_run_s: float | None = None, compile_overhead_s: float | None = None
):
    timings = {
        "cpu_outer_elapsed_s": 20.0,
        "jax_outer_elapsed_s": 25.0,
    }
    if warm_run_s is not None:
        timings["jax_optimizer_warm_run_s"] = warm_run_s
    if compile_overhead_s is not None:
        timings["jax_optimizer_compile_overhead_s"] = compile_overhead_s
    return {
        "passed": True,
        "comparison": {
            "cpu_elapsed_s": 20.0,
            "jax_elapsed_s": 10.0,
        },
        "timings": timings,
    }


def test_summarize_stage2_e2e_performance_probe_separates_cold_outer_and_warm():
    payload = _stage2_e2e_performance_probe_payload(
        warm_run_s=5.0, compile_overhead_s=5.0
    )

    summary = summarize_stage2_e2e_performance_probe(
        payload=payload,
        outer_elapsed_s=55.0,
        lane_label="jax-cuda",
    )

    assert summary["name"] == "tier2_stage2_e2e"
    assert summary["speedup_vs_cpu"] == pytest.approx(2.0)
    assert summary["outer_speedup_vs_cpu"] == pytest.approx(0.8)
    assert summary["warm_speedup_vs_cpu"] == pytest.approx(4.0)
    assert summary["headline_metric"] == "warm_speedup_vs_cpu"
    assert summary["headline_speedup_vs_cpu"] == pytest.approx(4.0)
    assert summary["supports_performance_headline"] is True
    assert summary["timing_semantics"] == "separate_cold_end_to_end_and_warm_steady_state"


def test_summarize_stage2_e2e_performance_probe_falls_back_to_cold_without_warm_timing():
    payload = _stage2_e2e_performance_probe_payload()

    summary = summarize_stage2_e2e_performance_probe(
        payload=payload,
        outer_elapsed_s=55.0,
        lane_label="jax-cuda",
    )

    assert summary["warm_speedup_vs_cpu"] is None
    assert summary["lane_warm_elapsed_s"] is None
    assert summary["headline_metric"] == "speedup_vs_cpu"
    assert summary["headline_speedup_vs_cpu"] == pytest.approx(2.0)


def test_build_tier5_performance_contract_routes_parity_and_headline_sources():
    summary = [
        {
            "name": "tier1b_real_stage2",
            "timing_semantics": "correctness_probe_only",
            "supports_performance_headline": False,
        },
        {
            "name": "tier2_stage2_e2e",
            "headline_metric": "warm_speedup_vs_cpu",
            "headline_speedup_vs_cpu": 4.0,
            "warm_speedup_vs_cpu": 4.0,
            "outer_speedup_vs_cpu": 0.8,
            "supports_performance_headline": True,
        },
        {
            "name": "tier3_single_stage_init",
            "timing_semantics": "initialization_probe_only",
            "supports_performance_headline": False,
        },
        {
            "name": "tier4_adjoint_fd",
            "supports_performance_headline": False,
        },
    ]

    contract = build_tier5_performance_contract(summary)

    assert contract["parity_source"]["rung"] == "tier1b_real_stage2"
    assert contract["cold_end_to_end_source"]["rung"] == "tier2_stage2_e2e"
    assert contract["cold_end_to_end_source"]["metric_path"] == (
        "summary_by_name.tier2_stage2_e2e.outer_speedup_vs_cpu"
    )
    assert contract["cold_end_to_end_source"]["speedup_vs_cpu"] == pytest.approx(0.8)
    assert contract["warm_steady_state_source"]["metric_path"] == (
        "summary_by_name.tier2_stage2_e2e.warm_speedup_vs_cpu"
    )
    assert contract["warm_steady_state_source"]["speedup_vs_cpu"] == pytest.approx(4.0)
    assert contract["headline_performance_source"]["metric_path"] == (
        "summary_by_name.tier2_stage2_e2e.warm_speedup_vs_cpu"
    )
    assert contract["sharding_source"]["active_path"] == (
        "rungs.tier2_stage2_e2e.provenance.sharding_active"
    )
    assert contract["sharding_source"]["strategy_path"] == (
        "rungs.tier2_stage2_e2e.provenance.sharding_strategy"
    )
    assert contract["do_not_use_for_performance_headline"] == [
        "tier1b_real_stage2",
        "tier3_single_stage_init",
    ]


def test_evaluate_tier5_sharding_contract_rejects_inactive_multi_device_lane():
    failures = tier5_performance_characterization.evaluate_tier5_sharding_contract(
        {
            "sharding_strategy": "hybrid",
            "sharding_active": False,
            "sharding_device_count": 4,
        }
    )

    assert failures
    assert "sharding_active" in failures[0]


def test_evaluate_tier5_performance_budget_rejects_stage2_speed_regressions():
    budget = tier5_performance_budget(profile="stable_hardware_weekly")
    failures = evaluate_tier5_performance_budget(
        {
            "tier2_stage2_e2e": {
                "outer_speedup_vs_cpu": 0.1,
                "warm_speedup_vs_cpu": 0.25,
                "lane_compile_overhead_s": 75.0,
            }
        },
        budget,
    )

    assert any("cold end-to-end speedup" in failure for failure in failures)
    assert any("warm steady-state speedup" in failure for failure in failures)
    assert any("compile overhead" in failure for failure in failures)


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
