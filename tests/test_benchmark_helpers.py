import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import math
import subprocess
import types

import numpy as np
import pytest
from simsopt.geo.surface import Surface
import simsopt.geo.surfaceobjectives_jax as surfaceobjectives_jax_module
from simsopt.geo.surfaceobjectives_jax import _canonicalize_traceable_exact_quadrature
from conftest import _parity_device_for_lane

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
import benchmarks.adjoint_fd_validation as adjoint_fd_validation_module
import benchmarks.adjoint_probe_common as adjoint_probe_common
from benchmarks.adjoint_probe_common import (
    compute_derivative_l2_metrics,
    compute_gradient_l2_metrics,
)
from benchmarks.single_stage_backend_routing import (
    resolve_boozer_least_squares_algorithm,
    resolve_boozer_optimizer_backend,
    resolve_boozer_limited_memory,
    resolve_boozer_optimizer_method,
)
from benchmarks.grouped_adjoint_memory_probe import (
    _GroupedVJPTimingRecorder,
    _build_grouped_adjoint_baseline_comparison,
    _build_grouped_adjoint_payload,
    _representative_run_wall_s,
    evaluate_grouped_adjoint_memory_probe,
)
import benchmarks.single_stage_init_parity as single_stage_init_parity_module
import benchmarks.single_stage_parity_matrix as single_stage_parity_matrix
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
import benchmarks.production_boozer_parity_probe as production_boozer_parity_probe_module
import benchmarks.run_code_parity_probe as run_code_parity_probe_module
import benchmarks.stage2_value_gradient_parity as stage2_value_gradient_parity_module
import benchmarks.tier5_performance_characterization as tier5_performance_characterization
from benchmarks.traceable_compile_shape import summarize_lowered_text
from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_PLASMA_SURF_FILENAME,
    default_optimizer_backend_for_backend,
)
from benchmarks.single_stage_smoke_defaults import DEFAULT_STAGE2_RESULTS_PATH
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
    summarize_single_stage_outer_loop_performance_probe,
    summarize_stage2_e2e_performance_probe,
    summarize_single_lane_probe,
)
from benchmarks.validation_ladder_common import (
    _JAX_COMPILATION_CACHE_ENV_VAR,
    _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR,
    _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR,
    _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR,
    _SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR,
    _SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR,
    _TARGET_LANE_ACCEPTED_STEP_SYNC_ENV_VAR,
    PARITY_LADDER_TOLERANCES,
    TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
    apply_benchmark_compilation_cache_policy,
    apply_compilation_cache_policy,
    benchmark_compilation_cache_dir,
    build_provenance,
    describe_compile_behavior,
    evaluate_tier5_performance_budget,
    grouped_adjoint_memory_budget,
    max_pointwise_geometry_drift,
    optimizer_drift_tolerances,
    parity_ladder_ratchet_rel_tol,
    parity_ladder_tolerances,
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


def test_summarize_lowered_text_counts_control_flow_tokens():
    lowered_text = "\n".join(
        (
            "module @shape {",
            "  stablehlo.while {...}",
            "  stablehlo.case {...}",
            "  stablehlo.if {...}",
            "  mhlo.while {...}",
            "}",
        )
    )

    summary = summarize_lowered_text(
        "unit",
        lowered_text,
        lower_s=0.125,
    )

    assert summary["label"] == "unit"
    assert summary["lower_s"] == pytest.approx(0.125)
    assert summary["text_lines"] == 6
    assert summary["stablehlo_while_count"] == 1
    assert summary["stablehlo_case_count"] == 1
    assert summary["stablehlo_if_count"] == 1
    assert summary["mhlo_while_count"] == 1
    assert summary["mhlo_case_count"] == 0


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


def test_build_ls_parity_problem_honors_requested_scale():
    problem = build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)

    assert len(problem.coils) == 4 * problem.surface.nfp * 2
    assert len(problem.surface.quadpoints_phi) == 16
    assert len(problem.surface.quadpoints_theta) == 8


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


def test_parity_ladder_ratchet_rel_tol_respects_lane_contracts():
    assert parity_ladder_ratchet_rel_tol(
        "direct-kernel",
        1e-10,
        1e-12,
    ) == pytest.approx(1e-11)
    assert parity_ladder_ratchet_rel_tol(
        "direct-kernel",
        1e-10,
        2e-11,
    ) == pytest.approx(1e-10)
    assert parity_ladder_ratchet_rel_tol(
        "exact-ill-conditioned-adjoint",
        1e-6,
        1e-12,
    ) == pytest.approx(1e-6)
    assert parity_ladder_ratchet_rel_tol(
        "branch-stable-resolve",
        1e-6,
        1e-12,
        branch_divergent=True,
    ) == pytest.approx(1e-6)


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
    results = json.loads(DEFAULT_STAGE2_RESULTS_PATH.read_text())

    assert results["MAJOR_RADIUS"] > 0.0
    assert results["TOROIDAL_FLUX"] > 0.0
    assert results["banana_surf_radius"] > 0.0
    assert results["order"] >= 1
    assert results["TF_CURRENT_A"] > 0.0
    assert results["NUM_TF_COILS"] > 0


def test_single_stage_init_fixture_tf_current_contract_is_valid():
    from simsopt._core.optimizable import load

    from examples.single_stage_optimization.banana_opt.current_contracts import (
        resolve_loaded_tf_current_A,
    )

    results = json.loads(DEFAULT_STAGE2_RESULTS_PATH.read_text())
    bs = load(str(DEFAULT_STAGE2_BS_PATH))

    tf_current_A = resolve_loaded_tf_current_A(
        results["TF_CURRENT_A"],
        bs.coils[: int(results["NUM_TF_COILS"])],
    )

    assert tf_current_A == pytest.approx(80000.0)


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
    assert args.reference_optimizer_method == "lbfgs"
    assert args.initial_step_scale == pytest.approx(1.0)
    assert args.initial_step_maxiter == 0
    assert args.outer_maxls == single_stage_init_parity_module.TRACE_PARITY_OUTER_MAXLS


def test_single_stage_init_accepts_reference_trace_optimizer(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_init_parity.py",
            "--output-json",
            "/tmp/out.json",
            "--reference-optimizer-method",
            "lbfgs-trace",
        ],
    )

    args = single_stage_init_parity_module.parse_args()

    assert args.reference_optimizer_method == "lbfgs-trace"


def test_single_stage_init_accepts_fullgraph_scipy_control_optimizer(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_init_parity.py",
            "--output-json",
            "/tmp/out.json",
            "--optimizer-backend",
            "scipy-jax-fullgraph",
        ],
    )

    args = single_stage_init_parity_module.parse_args()

    assert args.optimizer_backend == "scipy-jax-fullgraph"


def test_single_stage_init_accepts_objective_evaluation_trace(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_stage_init_parity.py",
            "--output-json",
            "/tmp/out.json",
            "--record-objective-evaluation-trace",
        ],
    )

    args = single_stage_init_parity_module.parse_args()

    assert args.record_objective_evaluation_trace


def test_single_stage_init_surface_geometry_gate_is_init_only(tmp_path):
    args = _single_stage_case_args(tmp_path)
    args.maxiter = 0

    assert single_stage_init_parity_module._should_compare_surface_geometry(
        args,
        benchmark_mode=False,
    )

    args.maxiter = 1
    assert not single_stage_init_parity_module._should_compare_surface_geometry(
        args,
        benchmark_mode=False,
    )
    assert not single_stage_init_parity_module._should_compare_surface_geometry(
        args,
        benchmark_mode=True,
    )


def test_single_stage_init_cpu_parity_requests_shared_seed_for_outer_runs(tmp_path):
    args = _single_stage_case_args(tmp_path)
    args.maxiter = 1
    args.reference_optimizer_method = "lbfgs-trace"

    assert single_stage_init_parity_module._needs_shared_init_seed(
        args,
        reference_backend="cpu",
    )

    args.reference_optimizer_method = "lbfgs"
    assert single_stage_init_parity_module._needs_shared_init_seed(
        args,
        reference_backend="cpu",
    )


def test_single_stage_fixture_optimizer_backend_defaults_by_backend():
    assert default_optimizer_backend_for_backend("jax") == "ondevice"
    assert default_optimizer_backend_for_backend("cpu") == "scipy"


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


def test_repo_pythonpath_env_keeps_cpu_visible_for_cuda_callbacks(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = repo_pythonpath_env(platform="cuda")

    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert env["SIMSOPT_JAX_PLATFORM"] == "cuda"
    assert env["SIMSOPT_JAX_BACKEND"] == "cuda"


def _assert_benchmark_module_import_bootstraps_local_simsopt(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(repo_root)!r}); "
                f"sys.path.insert(0, {str(repo_root / 'src')!r}); "
                f"import {module_name}; "
                "import simsopt; "
                "print(simsopt.__file__)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=repo_pythonpath_env(platform="cpu"),
    )

    assert completed.returncode == 0, completed.stderr
    assert str(repo_root / "src" / "simsopt" / "__init__.py") in completed.stdout.strip()


def test_single_stage_init_parity_import_bootstraps_local_simsopt():
    _assert_benchmark_module_import_bootstraps_local_simsopt(
        "benchmarks.single_stage_init_parity"
    )


def test_stage2_e2e_comparison_import_bootstraps_local_simsopt():
    _assert_benchmark_module_import_bootstraps_local_simsopt(
        "benchmarks.stage2_e2e_comparison"
    )


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


def test_parity_device_for_lane_rejects_unknown_lane():
    jax_module = types.SimpleNamespace(devices=lambda: ())

    with pytest.raises(ValueError, match="expected 'cpu' or 'gpu'"):
        _parity_device_for_lane(jax_module, "tpu")


def test_repo_pythonpath_env_can_disable_compilation_cache(monkeypatch):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR, "0")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR, "-1")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR, "all")

    env = repo_pythonpath_env(platform="cpu", disable_compilation_cache=True)

    assert _JAX_COMPILATION_CACHE_ENV_VAR not in env
    assert _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR not in env
    assert _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR not in env
    assert _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR not in env
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


def test_repo_pythonpath_env_adds_detected_cuda_toolchain_root(monkeypatch, tmp_path):
    cuda_root = tmp_path / "cuda"
    bin_dir = cuda_root / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ptxas").touch()
    monkeypatch.setattr(
        "repo_bootstrap._DEFAULT_CUDA_TOOLCHAIN_ROOT",
        cuda_root,
    )
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    env = repo_pythonpath_env(platform="cuda")

    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert (
        env["XLA_FLAGS"].split()[0]
        == f"--xla_gpu_cuda_data_dir={cuda_root}"
    )


def test_repo_pythonpath_env_respects_explicit_cuda_data_dir_flag(monkeypatch, tmp_path):
    cuda_root = tmp_path / "cuda"
    bin_dir = cuda_root / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ptxas").touch()
    explicit_flag = "--xla_gpu_cuda_data_dir=/already/set"
    monkeypatch.setattr(
        "repo_bootstrap._DEFAULT_CUDA_TOOLCHAIN_ROOT",
        cuda_root,
    )
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("XLA_FLAGS", f"{explicit_flag} --other-flag=1")

    env = repo_pythonpath_env(platform="cuda")

    assert env["PATH"].split(os.pathsep)[0] == str(cuda_root / "bin")
    assert env["XLA_FLAGS"] == f"{explicit_flag} --other-flag=1"


def test_repo_pythonpath_env_prefers_active_env_cuda_toolchain_root(
    monkeypatch, tmp_path
):
    active_root = tmp_path / "active-env"
    active_bin_dir = active_root / "bin"
    active_bin_dir.mkdir(parents=True)
    (active_bin_dir / "nvlink").touch()
    (active_bin_dir / "ptxas").touch()
    active_nvjitlink_dir = (
        active_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "nvjitlink"
        / "lib"
    )
    active_nvjitlink_dir.mkdir(parents=True)
    (active_nvjitlink_dir / "libnvJitLink.so.12").touch()

    default_root = tmp_path / "default-cuda"
    default_bin_dir = default_root / "bin"
    default_bin_dir.mkdir(parents=True)
    (default_bin_dir / "nvlink").touch()
    (default_bin_dir / "ptxas").touch()

    monkeypatch.setattr(
        "repo_bootstrap._DEFAULT_CUDA_TOOLCHAIN_ROOT",
        default_root,
    )
    monkeypatch.setattr("repo_bootstrap.sys.prefix", str(active_root))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    env = repo_pythonpath_env(platform="cuda")

    assert env["PATH"].split(os.pathsep)[0] == str(active_bin_dir)
    assert (
        env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(active_nvjitlink_dir)
    )
    assert (
        env["XLA_FLAGS"].split()[0]
        == f"--xla_gpu_cuda_data_dir={active_root}"
    )


def test_repo_pythonpath_env_detects_target_arch_nvjitlink_dir(monkeypatch, tmp_path):
    active_root = tmp_path / "active-env"
    active_bin_dir = active_root / "bin"
    active_bin_dir.mkdir(parents=True)
    (active_bin_dir / "nvlink").touch()
    (active_bin_dir / "ptxas").touch()
    active_target_nvjitlink_dir = active_root / "targets" / "sbsa-linux" / "lib"
    active_target_nvjitlink_dir.mkdir(parents=True)
    (active_target_nvjitlink_dir / "libnvJitLink.so.12").touch()

    default_root = tmp_path / "default-cuda"
    default_bin_dir = default_root / "bin"
    default_bin_dir.mkdir(parents=True)
    (default_bin_dir / "nvlink").touch()
    (default_bin_dir / "ptxas").touch()

    monkeypatch.setattr(
        "repo_bootstrap._DEFAULT_CUDA_TOOLCHAIN_ROOT",
        default_root,
    )
    monkeypatch.setattr("repo_bootstrap.sys.prefix", str(active_root))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    env = repo_pythonpath_env(platform="cuda")

    assert env["PATH"].split(os.pathsep)[0] == str(active_bin_dir)
    assert (
        env["LD_LIBRARY_PATH"].split(os.pathsep)[0]
        == str(active_target_nvjitlink_dir)
    )
    assert (
        env["XLA_FLAGS"].split()[0]
        == f"--xla_gpu_cuda_data_dir={active_root}"
    )


def test_repo_pythonpath_env_bundled_cuda_clears_local_toolchain_overrides(
    monkeypatch,
):
    monkeypatch.setenv("SIMSOPT_JAX_CUDA_LIBRARY_MODE", "bundled")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cuda/lib:/driver/lib")
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_gpu_cuda_data_dir=/tmp/fake-cuda --xla_gpu_deterministic_ops=true",
    )

    env = repo_pythonpath_env(platform="cuda")

    assert env["PATH"] == "/usr/bin"
    assert "LD_LIBRARY_PATH" not in env
    assert env["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true"
    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert env["SIMSOPT_JAX_PLATFORM"] == "cuda"
    assert env["SIMSOPT_JAX_BACKEND"] == "cuda"


def test_apply_compilation_cache_policy_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(_JAX_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR, raising=False)
    monkeypatch.delenv(_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR, raising=False)
    monkeypatch.delenv(_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR, raising=False)

    metadata = apply_compilation_cache_policy()

    assert metadata == {
        "compilation_cache_enabled": False,
        "compilation_cache_dir": None,
        "compilation_cache_policy": "disabled",
    }
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR not in os.environ


def test_apply_compilation_cache_policy_honors_explicit_cache_dir(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)

    metadata = apply_compilation_cache_policy(tmp_path / "jax-cache")

    assert metadata["compilation_cache_enabled"] is True
    assert metadata["compilation_cache_policy"] == "explicit"
    assert metadata["compilation_cache_dir"] == str(tmp_path / "jax-cache")
    assert Path(metadata["compilation_cache_dir"]).is_dir()
    assert os.environ[_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR] == "0"
    assert os.environ[_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR] == "-1"
    assert os.environ[_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR] == "all"


def test_apply_compilation_cache_policy_honors_disable_flag(monkeypatch):
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/jax-cache")
    monkeypatch.setenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, "1")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR, "0")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR, "-1")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR, "all")

    metadata = apply_compilation_cache_policy()

    assert metadata == {
        "compilation_cache_enabled": False,
        "compilation_cache_dir": None,
        "compilation_cache_policy": "disabled",
    }
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR not in os.environ


def test_benchmark_compilation_cache_dir_uses_repo_artifacts_root(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.REPO_ROOT",
        tmp_path,
    )

    cache_dir = benchmark_compilation_cache_dir("single_stage_outer_loop_probe")

    assert cache_dir == (
        tmp_path
        / ".artifacts"
        / "jax_compilation_cache"
        / "single_stage_outer_loop_probe"
    )


def test_benchmark_compilation_cache_dir_scopes_cuda_runs_by_gpu_name(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.REPO_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common._benchmark_cuda_cache_target_suffix",
        lambda: "cuda-nvidia-geforce-rtx-5090",
    )

    cache_dir = benchmark_compilation_cache_dir(
        "single_stage_outer_loop_probe",
        requested_platform="cuda",
    )

    assert cache_dir == (
        tmp_path
        / ".artifacts"
        / "jax_compilation_cache"
        / "single_stage_outer_loop_probe-cuda-nvidia-geforce-rtx-5090"
    )


def test_apply_benchmark_compilation_cache_policy_uses_explicit_cache_dir(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.REPO_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common._benchmark_cuda_cache_target_suffix",
        lambda: "cuda-nvidia-geforce-rtx-5090",
    )
    monkeypatch.delenv(_JAX_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)

    metadata = apply_benchmark_compilation_cache_policy(
        "single_stage_outer_loop_probe",
        requested_platform="cuda",
    )

    assert metadata["compilation_cache_enabled"] is True
    assert metadata["compilation_cache_policy"] == "explicit"
    assert metadata["compilation_cache_dir"] == str(
        tmp_path
        / ".artifacts"
        / "jax_compilation_cache"
        / "single_stage_outer_loop_probe-cuda-nvidia-geforce-rtx-5090"
    )
    assert os.environ[_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR] == "0"
    assert os.environ[_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR] == "-1"
    assert os.environ[_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR] == "all"


def test_apply_benchmark_compilation_cache_policy_honors_shared_cache_env(
    monkeypatch, tmp_path
):
    shared_cache_dir = tmp_path / "shared-jax-cache"
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, str(shared_cache_dir))
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)

    metadata = apply_benchmark_compilation_cache_policy(
        "single_stage_outer_loop_probe",
        requested_platform="cuda",
    )

    assert metadata["compilation_cache_enabled"] is True
    assert metadata["compilation_cache_policy"] == "explicit"
    assert metadata["compilation_cache_dir"] == str(shared_cache_dir)
    assert shared_cache_dir.is_dir()


def test_apply_benchmark_compilation_cache_policy_keeps_cpu_only_runs_disabled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "benchmarks.validation_ladder_common.REPO_ROOT",
        tmp_path,
    )
    monkeypatch.delenv(_JAX_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR, raising=False)
    monkeypatch.delenv(_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR, raising=False)
    monkeypatch.delenv(_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR, raising=False)
    monkeypatch.delenv(_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR, raising=False)

    metadata = apply_benchmark_compilation_cache_policy(
        "single_stage_outer_loop_probe",
        requested_platform="cpu",
    )

    assert metadata == {
        "compilation_cache_enabled": False,
        "compilation_cache_dir": None,
        "compilation_cache_policy": "disabled",
    }
    assert _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR not in os.environ
    assert _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR not in os.environ


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


def test_optimizer_drift_tolerances_include_optimizer_state_parity_lane():
    tolerances = optimizer_drift_tolerances("optimizer_state_parity")

    assert tolerances == {
        "x_rtol": pytest.approx(1e-6),
        "x_atol": pytest.approx(1e-8),
        "objective_rel_tol": pytest.approx(1e-6),
        "gradient_rtol": pytest.approx(1e-6),
        "gradient_atol": pytest.approx(1e-8),
        "jac_norm_inf_abs_tol": pytest.approx(1e-8),
    }


def _single_stage_parity_report(*, termination_message="ok"):
    return {
        "jax_cpu_vs_h100_value_grad": {
            "jax_cpu_objective": 1.0,
            "h100_gpu_objective": 1.0 + 1.0e-15,
            "objective_abs_delta": 1.0e-15,
            "objective_rel_delta": 1.0e-15,
            "jax_cpu_grad_inf_norm": 0.5,
            "h100_gpu_grad_inf_norm": 0.5,
            "grad_inf_norm_abs_delta": 0.0,
            "grad_max_abs_delta": 1.0e-12,
            "grad_max_abs_delta_index": 0,
            "grad_max_abs_delta_jax_cpu_component": 0.5,
            "grad_max_abs_delta_h100_gpu_component": 0.5 + 1.0e-12,
            "grad_allclose_rtol_1e-10_atol_1e-12": True,
        },
        "same_seed_no_optimizer_metrics": {
            "INITIAL_VOLUME": {
                "values": {
                    "cpu_scipy": 1.0,
                    "jax_cpu": 1.0,
                    "h100_gpu": 1.0,
                }
            },
            "TERMINATION_MESSAGE": {
                "values": {
                    "cpu_scipy": termination_message,
                    "jax_cpu": termination_message,
                    "h100_gpu": termination_message,
                }
            },
        },
    }


def _release_gate_parity_report(*, termination_message="ok"):
    lanes = ("cpu_scipy", "jax_cpu", "jax_gpu")
    metrics = {
        metric_name: {"values": {lane: 1.0 for lane in lanes}}
        for metric_name in (
            "INITIAL_VOLUME",
            "INITIAL_IOTA",
            "INITIAL_FIELD_ERROR",
            "INITIAL_MAX_CURVATURE",
            "FINAL_VOLUME",
            "FINAL_IOTA",
            "FIELD_ERROR",
            "MAX_CURVATURE",
            "CURVE_CURVE_MIN_DIST",
            "CURVE_SURFACE_MIN_DIST",
            "SURFACE_VESSEL_MIN_DIST",
        )
    }
    metrics["TERMINATION_MESSAGE"] = {
        "values": {lane: termination_message for lane in lanes}
    }
    return {
        "jax_cpu_vs_jax_gpu_value_grad": {
            "objective_abs_delta": 1.0e-15,
            "objective_rel_delta": 1.0e-15,
            "jax_cpu_grad_inf_norm": 0.5,
            "grad_max_abs_delta": 1.0e-12,
            "grad_allclose_rtol_1e-10_atol_1e-12": True,
        },
        "same_seed_no_optimizer_metrics": metrics,
        "full_run_artifact_contract": {
            "run_family_id": "same-run-family",
            "lanes": {
                lane: {
                    "runtime_seed_spec_hash": "same-runtime-seed",
                    "objective_configuration_hash": "same-objective",
                    "run_family_id": "same-run-family",
                    "init_only": False,
                }
                for lane in lanes
            },
        },
    }


def _release_gate_hashes(value="same-hash"):
    return {
        "equilibrium_hash": value,
        "runtime_seed_spec_hash": value,
        "biot_savart_json_hash": value,
        "objective_configuration_hash": value,
        "active_dof_mask_hash": value,
        "fixed_dof_mask_hash": value,
        "frozen_dof_mask_hash": value,
    }


def _release_gate_assembled_outputs():
    return {
        "total_objective": 1.0,
        "objective_components": {"field": 1.0},
        "full_optimizer_basis_gradient": [0.0],
        "gradient_inf_norm": 0.0,
        "gradient_l2_norm": 0.0,
        "field_error": 0.0,
        "iota": 0.1,
        "volume": 0.1,
        "max_curvature": 1.0,
        "coil_coil_min_distance": 0.1,
        "coil_plasma_min_distance": 0.1,
        "plasma_vessel_min_distance": 0.1,
        "self_intersection": {
            "available": True,
            "self_intersecting": False,
        },
        "hardware_constraints": {"status": "pass"},
    }


def _release_gate_operator_outputs():
    return {
        "biot_savart_B": [0.0],
        "surface_gamma": [[0.0, 0.0, 0.0]],
        "integral_BdotN": 0.0,
        "boozer_residual_vector": [0.0],
        "boozer_residual_norm": 0.0,
        "boozer_residual_max_norm": 0.0,
        "first_derivative_kernel_samples": {"status": "pass"},
        "boozer_residual_jacobian_metadata": {"status": "pass"},
        "boozer_jvp": [0.0],
        "boozer_vjp": [0.0],
        "boozer_adjoint_solve": {"status": "pass", "residual": 0.0},
    }


def _release_gate_lane(*, backend, gpu_memory_mb=None, include_gpu_facts=True):
    provenance = {
        "repo_sha": "abc123",
        "jax": "0.0.0",
        "jaxlib": "0.0.0",
        "backend": backend,
        "devices": [backend],
        "x64_enabled": True,
        "peak_rss_mb": 128.0,
        "xla_flags": "--xla_gpu_deterministic_ops=true",
        "compilation_cache_policy": "disabled",
    }
    if gpu_memory_mb is not None:
        provenance["gpu_memory_mb"] = gpu_memory_mb
    if include_gpu_facts:
        provenance["nvidia_smi_gpus"] = [
            {
                "name": "H100",
                "driver_version": "0.0",
                "memory_total_mb": 81920.0,
            }
        ]
    return {
        "hashes": _release_gate_hashes(),
        "assembled_outputs": _release_gate_assembled_outputs(),
        "operator_outputs": _release_gate_operator_outputs(),
        "provenance": provenance,
        "timings": {"compile_time_s": 1.0, "run_time_s": 2.0},
    }


def _release_gate_fixed_state_artifact(*, comparison_status="pass", gpu_memory_mb=1024.0):
    return {
        "schema_version": 1,
        "lanes": {
            "cpp_cpu": _release_gate_lane(backend="cpu", include_gpu_facts=False),
            "jax_cpu": _release_gate_lane(backend="cpu", include_gpu_facts=False),
            "jax_gpu": _release_gate_lane(
                backend="cuda",
                gpu_memory_mb=gpu_memory_mb,
            ),
        },
        "comparisons": {
            "cpp_cpu_vs_jax_cpu": {"status": comparison_status},
            "cpp_cpu_vs_jax_gpu": {"status": "pass"},
            "jax_cpu_vs_jax_gpu": {"status": "pass"},
        },
        "performance_summary_by_name": {
            "tier2_stage2_e2e": {
                "outer_speedup_vs_cpu": 2.0,
                "warm_speedup_vs_cpu": 2.0,
                "lane_compile_overhead_s": 1.0,
            }
        },
        "passed": comparison_status == "pass",
        "failures": [] if comparison_status == "pass" else ["cpp_cpu_vs_jax_cpu drift"],
    }


def _release_gate_parity_report_with_timings():
    report = _release_gate_parity_report()
    report["timings"] = {
        "cpu_scipy": {
            "script_total_s": 500.0,
        },
        "jax_gpu": {
            "outer_optimizer_s": 250.0,
            "outer_optimizer_main_s": 210.0,
            "script_total_s": 300.0,
        },
    }
    return report


def _release_gate_coordinate_mapping_artifact():
    return {
        "schema_version": 1,
        "status": "pass",
        "inputs": {},
        "mapping": {"status": "pass"},
        "active_indices": [0],
        "frozen_indices": [1],
        "state_reconstruction": {"status": "pass"},
        "gradient_projection": {"status": "pass"},
        "finite_difference_checks": [{"status": "pass"}],
        "failures": [],
    }


def _optimizer_trace_entry(*, iteration=1, trial_x=None):
    vector = lambda values: {"values": list(values)}
    scalar = lambda value: {"value": float(value)}
    if trial_x is None:
        trial_x = [0.875, 2.0625]
    return {
        "iteration": iteration,
        "x": vector([1.0, 2.0]),
        "fun": scalar(1.0),
        "jac": vector([0.5, -0.25]),
        "jac_inf_norm": scalar(0.5),
        "search_direction": vector([-0.5, 0.25]),
        "search_direction_dot_grad": scalar(-0.3125),
        "step_scale": scalar(0.25),
        "step": vector([-0.125, 0.0625]),
        "trial_x": vector(trial_x),
        "trial_fun": scalar(0.5),
        "trial_jac": vector([0.1, -0.2]),
        "trial_jac_inf_norm": scalar(0.2),
        "nfev": 3,
        "njev": 3,
        "line_search_status": 0,
        "valid_curvature": True,
        "accepted": True,
        "converged": False,
    }


def _write_optimizer_trace_progress(path: Path, entries=None, *, message="ok"):
    if entries is None:
        entries = [_optimizer_trace_entry()]
    if isinstance(entries, dict):
        entries = [entries]
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "label": "phase2_returned",
                        "result": {
                            "message": message,
                            "optimizer_state_trace": list(entries),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_single_stage_init_compares_case_optimizer_state_traces(tmp_path):
    cpu_progress = tmp_path / "cpu_progress.json"
    jax_progress = tmp_path / "jax_progress.json"
    _write_optimizer_trace_progress(cpu_progress)
    _write_optimizer_trace_progress(jax_progress)

    result = single_stage_init_parity_module._compare_case_optimizer_state_traces(
        {"outer_optimizer_progress_json": str(cpu_progress)},
        {"outer_optimizer_progress_json": str(jax_progress)},
    )

    assert result["status"] == "pass"
    assert result["entry_count"] == 1


def test_single_stage_init_blocks_missing_case_optimizer_state_trace(tmp_path):
    cpu_progress = tmp_path / "cpu_progress.json"
    cpu_progress.write_text(json.dumps({"events": []}), encoding="utf-8")

    result = single_stage_init_parity_module._compare_case_optimizer_state_traces(
        {"outer_optimizer_progress_json": str(cpu_progress)},
        {"outer_optimizer_progress_json": str(tmp_path / "missing.json")},
    )

    assert result["status"] == "blocked"


def test_single_stage_parity_matrix_blocks_unmatched_trajectory_artifacts():
    report = _single_stage_parity_report()
    report["same_seed_no_optimizer_metrics"]["TERMINATION_MESSAGE"]["values"][
        "h100_gpu"
    ] = "optimized"

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(report)

    assert matrix["comparisons"]["jax_cpu_vs_h100_same_state_value_grad"][
        "status"
    ] == "pass"
    assert matrix["comparisons"]["full_trajectory_parity"]["status"] == "blocked"
    assert matrix["passed"] is False


def test_single_stage_parity_matrix_reports_absolute_deltas():
    report = _single_stage_parity_report()
    report["jax_cpu_vs_h100_value_grad"]["objective_abs_delta"] = -1.0e-15
    report["jax_cpu_vs_h100_value_grad"]["objective_rel_delta"] = -1.0e-15
    report["jax_cpu_vs_h100_value_grad"]["grad_max_abs_delta"] = -1.0e-12
    report["same_seed_no_optimizer_metrics"]["INITIAL_VOLUME"]["values"][
        "jax_cpu"
    ] = 0.999999999999

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(report)

    same_state = matrix["comparisons"]["jax_cpu_vs_h100_same_state_value_grad"]
    metric = matrix["comparisons"]["cpu_scipy_vs_jax_cpu_same_seed_metrics"][
        "metrics"
    ]["INITIAL_VOLUME"]
    assert same_state["objective_abs_delta"] >= 0.0
    assert same_state["objective_rel_delta"] >= 0.0
    assert same_state["grad_max_abs_delta"] >= 0.0
    assert metric["abs_delta"] >= 0.0


def test_single_stage_parity_matrix_accepts_matched_optimizer_state_traces(tmp_path):
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    for path in (jax_cpu_progress, gpu_progress):
        _write_optimizer_trace_progress(path)

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _single_stage_parity_report(),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
    )

    comparisons = matrix["comparisons"]
    assert comparisons["optimizer_state_trace_pairs"]["status"] == "pass"
    assert comparisons["full_trajectory_parity"]["status"] == "pass"
    assert matrix["passed"] is True


def test_single_stage_parity_matrix_uses_progress_terminations(tmp_path):
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    _write_optimizer_trace_progress(jax_cpu_progress, message="matched")
    _write_optimizer_trace_progress(gpu_progress, message="matched")
    report = _single_stage_parity_report()
    report["same_seed_no_optimizer_metrics"]["TERMINATION_MESSAGE"]["values"][
        "h100_gpu"
    ] = "stale"

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        report,
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
    )

    full_trajectory = matrix["comparisons"]["full_trajectory_parity"]
    assert full_trajectory["status"] == "pass"
    assert full_trajectory["termination_messages"]["jax_cpu"] == "matched"
    assert full_trajectory["termination_messages"]["h100_gpu"] == "matched"


def test_single_stage_parity_matrix_accepts_cpu_scipy_progress_without_trace(tmp_path):
    cpu_progress = tmp_path / "cpu_progress.json"
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    cpu_progress.write_text(
        json.dumps({"events": [{"result": {"message": "matched"}}]}),
        encoding="utf-8",
    )
    _write_optimizer_trace_progress(jax_cpu_progress, message="matched")
    _write_optimizer_trace_progress(gpu_progress, message="matched")

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _single_stage_parity_report(),
        cpu_progress_json=str(cpu_progress),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
    )

    trace_pairs = matrix["comparisons"]["optimizer_state_trace_pairs"]["pairs"]
    assert "cpu_cpp_trace_vs_jax_cpu" not in trace_pairs
    assert (
        matrix["comparisons"]["full_trajectory_parity"]["termination_messages"][
            "cpu_scipy"
        ]
        == "matched"
    )
    assert matrix["comparisons"]["full_trajectory_parity"]["status"] == "pass"


def test_single_stage_parity_matrix_reports_drifted_optimizer_state_traces(tmp_path):
    cpu_progress = tmp_path / "cpu_progress.json"
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    drifted_gpu_entry = _optimizer_trace_entry()
    drifted_gpu_entry["trial_x"] = {"values": [0.5, 2.0625]}
    _write_optimizer_trace_progress(cpu_progress)
    _write_optimizer_trace_progress(jax_cpu_progress)
    _write_optimizer_trace_progress(gpu_progress, drifted_gpu_entry)

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _single_stage_parity_report(),
        cpu_progress_json=str(cpu_progress),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
    )

    comparisons = matrix["comparisons"]
    trace_pairs = comparisons["optimizer_state_trace_pairs"]
    assert trace_pairs["status"] == "drift"
    assert comparisons["full_trajectory_parity"]["status"] == "drift"
    assert (
        trace_pairs["pairs"]["jax_cpu_vs_h100_gpu"]["status"] == "drift"
    )


def test_single_stage_parity_matrix_compares_later_trace_entries(tmp_path):
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    jax_entries = [
        _optimizer_trace_entry(iteration=1),
        _optimizer_trace_entry(iteration=2),
    ]
    gpu_entries = [
        _optimizer_trace_entry(iteration=1),
        _optimizer_trace_entry(iteration=2, trial_x=[0.5, 2.0625]),
    ]
    _write_optimizer_trace_progress(jax_cpu_progress, jax_entries)
    _write_optimizer_trace_progress(gpu_progress, gpu_entries)

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _single_stage_parity_report(),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
    )

    pair = matrix["comparisons"]["optimizer_state_trace_pairs"]["pairs"][
        "jax_cpu_vs_h100_gpu"
    ]
    assert pair["status"] == "drift"
    assert pair["entry_count"] == 2
    assert pair["first_mismatch"]["iteration_index"] == 1
    assert matrix["passed"] is False


def test_release_gate_requires_direct_cpp_fixed_state_comparisons(tmp_path):
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    _write_optimizer_trace_progress(jax_cpu_progress)
    _write_optimizer_trace_progress(gpu_progress)

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report(),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
        fixed_state_artifact=_release_gate_fixed_state_artifact(
            comparison_status="drift"
        ),
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    fixed_state = matrix["buckets"]["fixed_state_physics_parity"]
    assert fixed_state["status"] == "drift"
    assert (
        fixed_state["comparisons"]["cpp_cpu_vs_jax_cpu_fixed_state"]["status"]
        == "drift"
    )
    assert matrix["release_gate_passed"] is False
    assert "fixed_state_physics_parity" in matrix["blocking_buckets"]


def test_release_gate_blocks_mixed_full_run_objective_contract(tmp_path):
    cpu_progress = tmp_path / "cpu_progress.json"
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    _write_optimizer_trace_progress(cpu_progress)
    _write_optimizer_trace_progress(jax_cpu_progress)
    _write_optimizer_trace_progress(gpu_progress)
    report = _release_gate_parity_report()
    report["full_run_artifact_contract"]["lanes"]["jax_gpu"][
        "objective_configuration_hash"
    ] = "different-objective"

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        report,
        cpu_progress_json=str(cpu_progress),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
        fixed_state_artifact=_release_gate_fixed_state_artifact(),
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    bucket = matrix["buckets"]["full_run_artifact_contract"]
    assert bucket["status"] == "blocked"
    assert "full_run_artifact_contract" in matrix["blocking_buckets"]
    assert any("objective_configuration_hash" in failure for failure in bucket["failures"])
    assert matrix["comparisons"]["cpu_scipy_vs_jax_cpu_same_seed_metrics"][
        "status"
    ] == "blocked"
    assert matrix["first_divergence"]["stage"] == "run_contract"
    assert matrix["release_gate_passed"] is False


def test_release_gate_missing_cpp_cpu_blocks_fixed_state_bucket():
    fixed_state = _release_gate_fixed_state_artifact()
    fixed_state["lanes"].pop("cpp_cpu")

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report(),
        fixed_state_artifact=fixed_state,
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    bucket = matrix["buckets"]["fixed_state_physics_parity"]
    assert bucket["status"] == "blocked"
    assert "cpp_cpu" in bucket["missing_lanes"]
    assert matrix["release_gate_passed"] is False


def test_release_gate_coordinate_mapping_artifact_is_required():
    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report(),
        fixed_state_artifact=_release_gate_fixed_state_artifact(),
    )

    bucket = matrix["buckets"]["coordinate_mapping_parity"]
    assert bucket["status"] == "blocked"
    assert "coordinate_mapping_parity" in matrix["blocking_buckets"]
    assert matrix["release_gate_passed"] is False


def test_release_gate_rejects_legacy_h100_keys():
    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _single_stage_parity_report(),
        fixed_state_artifact=_release_gate_fixed_state_artifact(),
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    same_state = matrix["comparisons"]["jax_cpu_vs_jax_gpu_same_state_value_grad"]
    assert same_state["status"] == "blocked"
    assert same_state["legacy_keys"]
    assert matrix["passed"] is False


def test_release_gate_blocks_missing_cuda_provenance():
    fixed_state = _release_gate_fixed_state_artifact()
    fixed_state["lanes"]["jax_gpu"]["provenance"].pop("nvidia_smi_gpus")

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report(),
        fixed_state_artifact=fixed_state,
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    bucket = matrix["buckets"]["performance_memory_report"]
    assert bucket["status"] == "blocked"
    assert any("NVIDIA GPU facts are missing" in failure for failure in bucket["failures"])


def test_release_gate_fails_when_gpu_memory_exceeds_checked_in_budget():
    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report(),
        fixed_state_artifact=_release_gate_fixed_state_artifact(
            gpu_memory_mb=13000.0,
        ),
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    bucket = matrix["buckets"]["performance_memory_report"]
    assert bucket["status"] == "drift"
    assert any("peak GPU memory" in failure for failure in bucket["failures"])
    assert matrix["release_gate_passed"] is False


def test_release_gate_derives_performance_budget_from_report_timings():
    fixed_state = _release_gate_fixed_state_artifact()
    fixed_state.pop("performance_summary_by_name")

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report_with_timings(),
        fixed_state_artifact=fixed_state,
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    bucket = matrix["buckets"]["performance_memory_report"]
    summary = bucket["performance_summary_by_name"]["tier2_stage2_e2e"]
    assert bucket["status"] == "drift"
    assert bucket["performance_summary_source"] == "parity_report.timings"
    assert summary["outer_speedup_vs_cpu"] == pytest.approx(2.0)
    assert summary["lane_compile_overhead_s"] == pytest.approx(40.0)
    assert "performance summary is missing" not in bucket["failures"]
    assert any("warm steady-state speedup" in failure for failure in bucket["failures"])


def test_release_gate_final_metric_drift_has_structured_first_divergence(tmp_path):
    cpu_progress = tmp_path / "cpu_progress.json"
    jax_cpu_progress = tmp_path / "jax_cpu_progress.json"
    gpu_progress = tmp_path / "gpu_progress.json"
    _write_optimizer_trace_progress(cpu_progress)
    _write_optimizer_trace_progress(jax_cpu_progress)
    _write_optimizer_trace_progress(gpu_progress)
    report = _release_gate_parity_report()
    report["same_seed_no_optimizer_metrics"]["FINAL_VOLUME"]["values"][
        "jax_gpu"
    ] = 1.5

    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        report,
        cpu_progress_json=str(cpu_progress),
        jax_cpu_progress_json=str(jax_cpu_progress),
        gpu_progress_json=str(gpu_progress),
        fixed_state_artifact=_release_gate_fixed_state_artifact(),
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )

    divergence = matrix["first_divergence"]
    assert divergence["stage"] == "final_sync"
    assert divergence["stage"] in single_stage_parity_matrix.FIRST_DIVERGENCE_STAGES
    assert any(
        "cpu_scipy_vs_jax_gpu_final_metrics" in failure
        for failure in matrix["buckets"]["final_metric_envelope"]["failures"]
    )
    assert matrix["release_gate_passed"] is False


def test_release_gate_markdown_report_includes_required_summary_tables():
    matrix = single_stage_parity_matrix.build_single_stage_parity_matrix(
        _release_gate_parity_report(),
        fixed_state_artifact=_release_gate_fixed_state_artifact(
            comparison_status="drift"
        ),
        coordinate_mapping_artifact=_release_gate_coordinate_mapping_artifact(),
    )
    matrix["fixed_state_artifact"] = _release_gate_fixed_state_artifact(
        comparison_status="drift"
    )
    matrix["artifact_paths"] = {
        "fixed_state_parity_json": ".artifacts/parity/fixed-state.json",
        "coordinate_mapping_json": ".artifacts/parity/coordinate.json",
    }

    markdown = single_stage_parity_matrix.build_release_gate_markdown_report(matrix)

    assert "# Single-Stage Release Gate: FAIL" in markdown
    assert "fixed_state_physics_parity failed because" in markdown
    assert "## Bucket Status" in markdown
    assert "## Fixed-State Deltas" in markdown
    assert "## Full-Run Public Behavior Deltas" in markdown
    assert "## Memory Table" in markdown
    assert "## Memory And Performance Budgets" in markdown
    assert "## Device And Version Table" in markdown
    assert "## Git Status" in markdown
    assert "cpp_cpu_vs_jax_cpu_fixed_state" in markdown
    assert "## Runtime Table" in markdown
    assert ".artifacts/parity/fixed-state.json" in markdown


def test_parity_ladder_tolerances_capture_precision_lanes():
    expected_lanes = {
        "direct_kernel",
        "ls_wrapper_gradient",
        "derivative_heavy",
        "exact_well_conditioned_adjoint",
        "exact_ill_conditioned_adjoint",
        "branch_stable_resolve",
        "fd_gradient",
        "gpu_runtime",
        "reduction_cpu_gpu",
    }
    assert set(PARITY_LADDER_TOLERANCES) == expected_lanes

    direct = parity_ladder_tolerances("direct-kernel")
    assert direct["rtol"] == pytest.approx(1e-10)
    assert direct["atol"] == pytest.approx(1e-12)
    assert direct["requires_direct_cpp_oracle"] is True

    ls_wrapper = parity_ladder_tolerances("ls-wrapper-gradient")
    assert ls_wrapper["rtol"] == pytest.approx(1e-10)
    assert ls_wrapper["atol"] == pytest.approx(1e-12)
    assert ls_wrapper["requires_same_state"] is True

    derivative = parity_ladder_tolerances("derivative-heavy")
    assert derivative["first_derivative_rtol"] == pytest.approx(1e-8)
    assert derivative["first_derivative_atol"] == pytest.approx(1e-10)
    assert derivative["second_derivative_rtol"] == pytest.approx(1e-6)
    assert derivative["second_derivative_atol"] == pytest.approx(1e-8)

    exact_well_conditioned = parity_ladder_tolerances(
        "exact-well-conditioned-adjoint"
    )
    assert exact_well_conditioned["adjoint_rtol"] == pytest.approx(1e-6)
    assert exact_well_conditioned["adjoint_atol"] == pytest.approx(1e-8)
    assert exact_well_conditioned["residual_rel_tol"] == pytest.approx(1e-10)
    assert exact_well_conditioned["vector_parity_required"] is True

    exact_ill_conditioned = parity_ladder_tolerances(
        "exact-ill-conditioned-adjoint"
    )
    assert exact_ill_conditioned["adjoint_rtol"] is None
    assert exact_ill_conditioned["residual_rel_tol"] == pytest.approx(1e-10)
    assert exact_ill_conditioned["operator_failure_allowed"] is True
    assert exact_ill_conditioned["vector_parity_required"] is False

    branch_stable = parity_ladder_tolerances("branch-stable-resolve")
    assert branch_stable["core_value_rtol"] == pytest.approx(1e-6)
    assert branch_stable["core_value_atol"] == pytest.approx(1e-7)
    assert branch_stable["derived_value_rtol"] == pytest.approx(5e-5)
    assert branch_stable["derived_value_atol"] == pytest.approx(1e-7)

    fd_gradient = parity_ladder_tolerances("FD-gradient")
    assert fd_gradient["directional_fd_rtol"] == pytest.approx(1e-5)
    assert fd_gradient["directional_fd_atol"] == pytest.approx(1e-7)
    assert fd_gradient["central_fd_error_rate"] == pytest.approx(0.4)

    gpu_runtime = parity_ladder_tolerances("GPU-runtime")
    assert gpu_runtime["same_state_forward_rtol"] == pytest.approx(1e-10)
    assert gpu_runtime["same_state_gradient_rtol"] == pytest.approx(1e-8)
    assert gpu_runtime["whole_solve_value_rtol"] == pytest.approx(1e-6)
    assert gpu_runtime["whole_solve_value_atol"] == pytest.approx(1e-7)
    assert gpu_runtime["requires_runtime_metadata"] is True

    reduction_cpu_gpu = parity_ladder_tolerances("reduction-cpu-gpu")
    assert reduction_cpu_gpu["rtol"] == pytest.approx(1e-12)
    assert reduction_cpu_gpu["atol"] == pytest.approx(1e-12)
    assert reduction_cpu_gpu["requires_cpu_gpu_devices"] is True


def test_parity_ladder_tolerances_return_independent_copy():
    direct = parity_ladder_tolerances("direct_kernel")
    direct["rtol"] = 1.0

    assert parity_ladder_tolerances("direct_kernel")["rtol"] == pytest.approx(
        1e-10
    )


def test_parity_ladder_tolerances_reject_unknown_lane():
    with pytest.raises(ValueError, match="Unknown parity ladder lane"):
        parity_ladder_tolerances("exact-dense-plu-parity")


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
    assert resolve_probe_lane(optimizer_backend="ondevice") == "private-optimizer"
    assert resolve_probe_lane(optimizer_backend="scipy-jax") == "target-scipy-control"
    assert (
        resolve_probe_lane(optimizer_backend="scipy-jax-fullgraph")
        == "target-scipy-fullgraph-control"
    )
    with pytest.raises(ValueError, match="optimizer_backend must be one of"):
        resolve_probe_lane(optimizer_backend="hybrid")


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
    with pytest.raises(ValueError, match="optimizer_backend must be one of"):
        run_code_benchmark_common._resolve_runtime_lane(("hybrid",))


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
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    monkeypatch.setenv("CUDA_FORCE_PTX_JIT", "1")
    monkeypatch.setenv("CUDA_DISABLE_PTX_JIT", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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
        "benchmarks.validation_ladder_common.query_nvidia_smi_facts",
        lambda: {"nvidia_smi_gpus": [{"name": "test-gpu"}]},
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
    assert provenance["xla_flags"] == "--xla_gpu_deterministic_ops=true"
    assert provenance["cuda_force_ptx_jit"] == "1"
    assert provenance["cuda_disable_ptx_jit"] == "0"
    assert provenance["cuda_env"] == {
        "CUDA_FORCE_PTX_JIT": "1",
        "CUDA_DISABLE_PTX_JIT": "0",
    }
    assert provenance["nvidia_smi_gpus"] == [{"name": "test-gpu"}]
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
        "FINAL_IOTA": 0.15000000005,
        "FINAL_VOLUME": 0.100000000005,
        "FIELD_ERROR": 0.003000000012,
        "MAX_CURVATURE": 12.1,
        "SELF_INTERSECTING": False,
        "SELF_INTERSECTION_CHECK_AVAILABLE": True,
    }

    comparison, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=1e-12,
        max_surface_geometry_rel=5e-10,
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
        max_surface_geometry_abs=1e-12,
        max_surface_geometry_rel=5e-10,
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
        boozer_optimizer_backend="ondevice",
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
    assert command[boozer_optimizer_flag_index + 1] == "ondevice"
    target_lane_sync_flag_index = command.index("--target-lane-accepted-step-sync")
    assert command[target_lane_sync_flag_index + 1] == "final-only"
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in env
    assert env[_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR] == "1"
    assert env[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] == "disabled"


def test_single_stage_init_case_threads_fullgraph_optimizer_backend_to_jax_lane(
    monkeypatch, tmp_path
):
    args = _single_stage_case_args(tmp_path)
    args.optimizer_backend = "scipy-jax-fullgraph"
    args.record_objective_evaluation_trace = True
    observed_invocations = _observe_single_stage_case_invocations(monkeypatch, tmp_path)
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

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cpu",
        load_surface_gamma=False,
    )

    assert len(observed_invocations) == 1
    command, _env = observed_invocations[0]
    optimizer_flag_index = command.index("--optimizer-backend")
    assert command[optimizer_flag_index + 1] == "scipy-jax-fullgraph"
    assert "--boozer-optimizer-backend" not in command
    assert "--record-objective-evaluation-trace" in command
    target_lane_sync_flag_index = command.index("--target-lane-accepted-step-sync")
    assert command[target_lane_sync_flag_index + 1] == "per-accept"


def test_single_stage_init_case_threads_profile_target_lane_only_flag(
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
        jax_profile_dir=str(tmp_path / "xprof"),
        profile_target_lane_batch_size=4,
    )

    observed_command: list[str] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        lambda _script_path, command, **kwargs: observed_command.extend(command)
        or argparse.Namespace(stdout="", stderr=""),
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

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cuda",
        benchmark_mode=True,
        load_surface_gamma=False,
        profile_target_lane=True,
        profile_target_lane_only=True,
    )

    assert "--profile-target-lane" in observed_command
    assert "--profile-target-lane-only" in observed_command
    batch_flag_index = observed_command.index("--profile-target-lane-batch-size")
    assert observed_command[batch_flag_index + 1] == "4"
    profile_flag_index = observed_command.index("--jax-profile-dir")
    assert observed_command[profile_flag_index + 1] == str(tmp_path / "xprof")


def _single_stage_case_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
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
        jax_profile_dir=None,
        profile_target_lane_batch_size=1,
        reference_optimizer_method="lbfgs",
        initial_step_scale=1.0,
        initial_step_maxiter=0,
        outer_maxls=single_stage_init_parity_module.TRACE_PARITY_OUTER_MAXLS,
    )


def _observe_single_stage_case_invocations(monkeypatch, tmp_path: Path):
    observed_invocations: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_single_stage_script_path",
        lambda: tmp_path / "driver.py",
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "run_python_script",
        lambda _script_path, command, **kwargs: observed_invocations.append(
            (list(command), dict(kwargs["env"]))
        )
        or argparse.Namespace(stdout="", stderr=""),
    )
    return observed_invocations


def test_single_stage_init_case_pair_threads_shared_seed_to_jax_fullgraph_lane(
    monkeypatch,
    tmp_path,
):
    args = _single_stage_case_args(tmp_path)
    args.optimizer_backend = "scipy-jax-fullgraph"
    args.maxiter = 3
    args.platform = "cpu"
    args.jax_runtime_seed_spec = None
    args.warm_start_run_dir = None
    calls = []

    def fake_run_single_stage_case(
        case_args,
        backend,
        *,
        platform,
        benchmark_mode,
        load_surface_gamma,
        output_root,
        jax_runtime_seed_spec=None,
    ):
        del platform, benchmark_mode, load_surface_gamma, jax_runtime_seed_spec
        run_dir = tmp_path / f"{len(calls)}_{backend}"
        run_dir.mkdir()
        calls.append(
            {
                "backend": backend,
                "warm_start_run_dir": getattr(case_args, "warm_start_run_dir", None),
                "output_root": Path(output_root),
            }
        )
        progress_json = run_dir / "outer_optimizer_progress.json"
        progress_json.write_text("[]", encoding="utf-8")
        return {
            "run_dir": str(run_dir),
            "results": _single_stage_contract_results(),
            "outer_optimizer_progress_json": str(progress_json),
        }

    def fake_compile_seed_spec(run_dir, output_path, _args):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text('{"seed": true}', encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_run_single_stage_case",
        fake_run_single_stage_case,
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "_compile_jax_runtime_seed_spec_from_run_dir",
        fake_compile_seed_spec,
    )

    single_stage_init_parity_module._run_single_stage_case_pair(
        args,
        benchmark_mode=False,
        reference_backend="cpu",
        reference_benchmark_mode=False,
        case_root=tmp_path / "case",
    )

    seed_run_dir = calls[0]["warm_start_run_dir"]
    assert calls[0]["backend"] == "cpu"
    assert seed_run_dir is None
    shared_seed_run_dir = str(tmp_path / "0_cpu")
    assert calls[1]["backend"] == "cpu"
    assert calls[1]["warm_start_run_dir"] == shared_seed_run_dir
    assert calls[2]["backend"] == "jax"
    assert calls[2]["warm_start_run_dir"] == shared_seed_run_dir


def test_single_stage_init_case_threads_phase1_diagnostic_flags_and_env(
    monkeypatch, tmp_path
):
    args = _single_stage_case_args(tmp_path)
    observed_invocations = _observe_single_stage_case_invocations(monkeypatch, tmp_path)

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

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cuda",
        benchmark_mode=True,
        load_surface_gamma=False,
        diagnose_target_lane_scaled_phase1=True,
        record_target_lane_invalid_state_events=True,
        enable_compile_diagnostics=True,
        deterministic_gpu_reductions=True,
    )

    assert len(observed_invocations) == 1
    command, env = observed_invocations[0]
    assert "--diagnose-target-lane-scaled-phase1" in command
    assert "--diagnostic-callbacks" in command
    assert "--record-target-lane-invalid-state-events" not in command
    assert "--record-jax-compile-diagnostics" not in command
    assert "--xla_gpu_deterministic_ops=true" in env["XLA_FLAGS"].split()


def test_single_stage_init_case_threads_compile_diagnostics_without_host_callbacks(
    monkeypatch, tmp_path
):
    args = _single_stage_case_args(tmp_path)
    observed_invocations = _observe_single_stage_case_invocations(monkeypatch, tmp_path)
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "find_single_file",
        lambda root, pattern: Path(root) / pattern,
    )
    monkeypatch.setattr(
        single_stage_init_parity_module,
        "load_json",
        lambda _path: {},
    )

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "jax",
        platform="cuda",
        benchmark_mode=True,
        load_surface_gamma=False,
        enable_compile_diagnostics=True,
    )

    assert len(observed_invocations) == 1
    command, _env = observed_invocations[0]
    assert "--record-jax-compile-diagnostics" in command


def test_prefix_phase_timings_adds_lane_prefix():
    assert single_stage_init_parity_module._prefix_phase_timings(
        "jax",
        {"boozer_total_s": 2.5, "outer_optimizer_s": 1.0},
    ) == {
        "jax_boozer_total_s": pytest.approx(2.5),
        "jax_outer_optimizer_s": pytest.approx(1.0),
    }


def _single_stage_contract_results(**overrides):
    results = {
        "CONSTRAINT_METHOD": "penalty",
        "CONSTRAINT_WEIGHT": 1.0,
        "ALM_FORMULATION": None,
        "TARGET_VOLUME": 0.1,
        "TARGET_IOTA": 0.15,
        "NON_QS_WEIGHT": 1.0,
        "RES_WEIGHT": 1.0,
        "IOTAS_WEIGHT": 1.0,
        "LENGTH_WEIGHT": 1.0,
        "LENGTH_TARGET": 1.0,
        "CC_DIST": 0.1,
        "CC_WEIGHT": 1.0,
        "CS_DIST": 0.1,
        "CS_WEIGHT": 1.0,
        "SS_DIST": 0.1,
        "SURF_DIST_WEIGHT": 1.0,
        "CURVATURE_THRESHOLD": 100.0,
        "CURVATURE_WEIGHT": 1.0,
        "BANANA_CURRENT_MAX_A": 80000.0,
        "STAGE2_TF_CURRENT_A": 80000.0,
        "STAGE2_TF_CURRENT_LIMIT_ENFORCED": True,
        "TF_CURRENT_LIMIT_A": 80000.0,
        "COIL_VESSEL_MIN_DIST_M": 0.01,
        "init_only": False,
        "provenance": {
            "generated_at_utc": "2026-05-02T00:00:00Z",
            "repo_sha": "abc123",
        },
    }
    results.update(overrides)
    return results


def _single_stage_contract_case(tmp_path: Path, lane_name: str, results):
    run_dir = tmp_path / lane_name
    run_dir.mkdir()
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    progress_json = run_dir / "outer_optimizer_progress.json"
    _write_optimizer_trace_progress(progress_json)
    return {
        "results": results,
        "run_dir": str(run_dir),
        "outer_optimizer_progress_json": str(progress_json),
    }


def test_single_stage_init_full_run_contract_records_reportable_lane_state(
    tmp_path,
):
    args = _single_stage_case_args(tmp_path)
    args.platform = "cpu"
    seed_spec = tmp_path / "single_stage_jax_runtime_seed_spec.json"
    seed_spec.write_text('{"seed": 7}', encoding="utf-8")
    cpu_case = _single_stage_contract_case(
        tmp_path,
        "cpu",
        _single_stage_contract_results(),
    )
    jax_case = _single_stage_contract_case(
        tmp_path,
        "jax",
        _single_stage_contract_results(),
    )

    contract = (
        single_stage_init_parity_module.build_single_stage_full_run_artifact_contract(
            args,
            reference_backend="cpu",
            cpu_case=cpu_case,
            jax_case=jax_case,
            jax_seed_spec=seed_spec,
        )
    )

    lanes = contract["lanes"]
    assert set(lanes) == {"cpu_scipy", "jax_cpu"}
    assert contract["runtime_seed_spec_hash"] == single_stage_parity_matrix._file_sha256(
        seed_spec
    )
    assert lanes["cpu_scipy"]["runtime_seed_spec_hash"] == contract[
        "runtime_seed_spec_hash"
    ]
    assert lanes["jax_cpu"]["runtime_seed_spec_hash"] == contract[
        "runtime_seed_spec_hash"
    ]
    assert lanes["cpu_scipy"]["objective_configuration_hash"] == lanes["jax_cpu"][
        "objective_configuration_hash"
    ]
    assert lanes["cpu_scipy"]["missing_objective_config_keys"] == []
    assert lanes["cpu_scipy"]["run_family_id"] == contract["run_family_id"]
    assert lanes["jax_cpu"]["run_family_id"] == contract["run_family_id"]
    assert lanes["cpu_scipy"]["init_only"] is False
    assert lanes["cpu_scipy"]["results_json"] == str(
        Path(cpu_case["run_dir"]) / "results.json"
    )
    assert lanes["jax_cpu"]["progress_json"] == jax_case[
        "outer_optimizer_progress_json"
    ]


def test_single_stage_init_full_run_contract_preserves_objective_mismatch(
    tmp_path,
):
    args = _single_stage_case_args(tmp_path)
    args.platform = "auto"
    seed_spec = tmp_path / "single_stage_jax_runtime_seed_spec.json"
    seed_spec.write_text('{"seed": 7}', encoding="utf-8")
    cpu_case = _single_stage_contract_case(
        tmp_path,
        "cpu",
        _single_stage_contract_results(CURVATURE_THRESHOLD=100.0),
    )
    jax_case = _single_stage_contract_case(
        tmp_path,
        "jax",
        _single_stage_contract_results(
            CURVATURE_THRESHOLD=40.0,
            provenance={
                "backend": "gpu",
                "generated_at_utc": "2026-05-02T00:00:00Z",
                "repo_sha": "abc123",
            },
        ),
    )

    contract = (
        single_stage_init_parity_module.build_single_stage_full_run_artifact_contract(
            args,
            reference_backend="cpu",
            cpu_case=cpu_case,
            jax_case=jax_case,
            jax_seed_spec=seed_spec,
        )
    )

    lanes = contract["lanes"]
    assert set(lanes) == {"cpu_scipy", "jax_gpu"}
    assert lanes["cpu_scipy"]["run_family_id"] == lanes["jax_gpu"]["run_family_id"]
    assert (
        lanes["cpu_scipy"]["objective_configuration_hash"]
        != lanes["jax_gpu"]["objective_configuration_hash"]
    )
    assert lanes["cpu_scipy"]["missing_objective_config_keys"] == []
    assert lanes["jax_gpu"]["missing_objective_config_keys"] == []


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


def test_single_stage_init_reference_case_keeps_restart_artifacts_for_seed_compile():
    args = types.SimpleNamespace(jax_runtime_seed_spec=None, warm_start_run_dir=None)

    assert (
        single_stage_init_parity_module._reference_case_benchmark_mode(args, True)
        is False
    )
    assert single_stage_init_parity_module._reference_case_backend(args) == "cpu"


def test_single_stage_init_reference_case_uses_benchmark_mode_with_explicit_seed():
    args = types.SimpleNamespace(
        jax_runtime_seed_spec="/tmp/seed-spec.json",
        warm_start_run_dir=None,
    )

    assert (
        single_stage_init_parity_module._reference_case_benchmark_mode(args, True)
        is True
    )
    assert single_stage_init_parity_module._reference_case_backend(args) == "jax"


def test_single_stage_init_reference_case_uses_jax_cpu_with_warm_start_seed():
    args = types.SimpleNamespace(
        jax_runtime_seed_spec=None,
        warm_start_run_dir="/tmp/single-stage-run",
    )

    assert (
        single_stage_init_parity_module._reference_case_benchmark_mode(args, True)
        is True
    )
    assert single_stage_init_parity_module._reference_case_backend(args) == "jax"


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


def test_single_stage_init_case_threads_disable_target_lane_success_filter_flag(
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
        disable_target_lane_success_filter=True,
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
    )

    assert "--disable-target-lane-success-filter" in observed_command


def test_single_stage_init_case_threads_target_lane_boozer_trial_overrides(
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
        target_lane_boozer_bfgs_tol=1e-6,
        target_lane_boozer_bfgs_maxiter=64,
        target_lane_boozer_newton_tol=1e-9,
        target_lane_boozer_newton_maxiter=3,
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
    )

    assert "--target-lane-boozer-bfgs-tol" in observed_command
    assert "--target-lane-boozer-bfgs-maxiter" in observed_command
    assert "--target-lane-boozer-newton-tol" in observed_command
    assert "--target-lane-boozer-newton-maxiter" in observed_command


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
        "iterations": 10,
        "boozer_optimizer_backend": "ondevice",
        "boozer_optimizer_method": "bfgs-ondevice",
        "outer_optimizer_method": TARGET_OUTER_OPTIMIZER_METHOD,
        "INITIAL_OBJECTIVE": 10.0,
        "FINAL_OBJECTIVE": 8.0,
    }
    results.update(overrides)
    return results


def test_single_stage_init_parity_requires_accepted_step_on_outer_loop_probe():
    cpu_results = _single_stage_probe_results()
    jax_results = _single_stage_probe_results(
        iterations=0,
        outer_optimizer_method=TARGET_OUTER_OPTIMIZER_METHOD,
    )

    _, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=0.0,
        max_surface_geometry_rel=0.0,
        maxiter=1,
    )

    assert any("did not accept an optimizer step" in failure for failure in failures)


def test_single_stage_init_parity_accepts_fullgraph_target_optimizer_method():
    cpu_results = _single_stage_probe_results()
    jax_results = _single_stage_probe_results(
        outer_optimizer_method="lbfgs-scipy-jax-fullgraph",
    )

    _, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=0.0,
        max_surface_geometry_rel=0.0,
        maxiter=1,
        expected_jax_outer_optimizer_method="lbfgs-scipy-jax-fullgraph",
    )

    assert failures == []


def test_single_stage_init_parity_rejects_non_finite_outer_loop_results():
    cpu_results = _single_stage_probe_results()
    jax_results = _single_stage_probe_results(
        FINAL_IOTA=np.nan,
        MAX_CURVATURE=np.inf,
        outer_optimizer_method=TARGET_OUTER_OPTIMIZER_METHOD,
    )

    comparison, failures = evaluate_single_stage_init_parity(
        cpu_results,
        jax_results,
        max_surface_geometry_abs=0.0,
        max_surface_geometry_rel=0.0,
        maxiter=1,
    )

    assert comparison["jax_finite_result_keys"]["FINAL_IOTA"] is False
    assert comparison["jax_finite_result_keys"]["MAX_CURVATURE"] is False
    assert any("non-finite FINAL_IOTA" in failure for failure in failures)
    assert any("non-finite MAX_CURVATURE" in failure for failure in failures)


def test_single_stage_outer_loop_probe_accepts_finite_target_lane_result():
    summary, failures = evaluate_single_stage_outer_loop_probe(
        _single_stage_probe_results(
            FINAL_IOTA=0.01,
            FIELD_ERROR=0.004,
            MAX_CURVATURE=32.0,
            SELF_INTERSECTION_CHECK_AVAILABLE=False,
        ),
        expected_boozer_optimizer_backend="ondevice",
        expected_boozer_optimizer_method="bfgs-ondevice",
    )

    assert failures == []
    assert summary["iterations"] == 10
    assert summary["boozer_optimizer_backend"] == "ondevice"
    assert summary["outer_optimizer_method"] == TARGET_OUTER_OPTIMIZER_METHOD
    assert summary["self_intersection_check_available"] is False
    assert summary["objective_decrease"] == pytest.approx(2.0)
    assert summary["objective_decreased"] is True


def test_single_stage_outer_loop_probe_rejects_missing_step_or_wrong_method():
    _, failures = evaluate_single_stage_outer_loop_probe(
        _single_stage_probe_results(
            iterations=0,
            boozer_optimizer_backend="ondevice",
            boozer_optimizer_method="lm-ondevice",
            outer_optimizer_method="bfgs",
            SELF_INTERSECTING=True,
            FINAL_IOTA=np.nan,
            FINAL_OBJECTIVE=10.5,
            FIELD_ERROR=0.004,
            MAX_CURVATURE=32.0,
        ),
        expected_boozer_optimizer_backend="ondevice",
        expected_boozer_optimizer_method="bfgs-ondevice",
    )

    assert any("required 10 accepted optimizer iterations" in failure for failure in failures)
    assert any(
        "requested inner Boozer optimizer method" in failure for failure in failures
    )
    assert any("self-intersecting surface" in failure for failure in failures)
    assert any("did not decrease the objective" in failure for failure in failures)
    assert any("non-finite FINAL_IOTA" in failure for failure in failures)


def test_single_stage_outer_loop_probe_profile_only_allows_zero_iterations():
    summary, failures = evaluate_single_stage_outer_loop_probe(
        _single_stage_probe_results(
            iterations=0,
            boozer_optimizer_backend="ondevice",
            boozer_optimizer_method="bfgs-ondevice",
            outer_optimizer_method=TARGET_OUTER_OPTIMIZER_METHOD,
        ),
        expected_boozer_optimizer_backend="ondevice",
        expected_boozer_optimizer_method="bfgs-ondevice",
        require_accepted_step=False,
    )

    assert failures == []
    assert summary["iterations"] == 0


def test_single_stage_outer_loop_probe_builds_phase1_note_from_scaled_phase1_diagnosis():
    disable_reason = (
        single_stage_init_parity_module._TARGET_LANE_COMPILE_DIAGNOSTICS_HOST_CALLBACK_REASON
    )
    note = single_stage_outer_loop_probe.build_phase1_diagnostic_note(
        {
            "iterations": 0,
            "INITIAL_PHASE_ITERATIONS": 3,
            "TERMINATION_MESSAGE": "diagnose_target_lane_scaled_phase1",
            "JAX_PROFILE_DIR": "/tmp/xprof",
            "TARGET_LANE_SCALED_PHASE1_DIAGNOSIS": {
                "first_nonfinite_stage": "steepest_descent_trial"
            },
        },
        failures=["Single-stage outer-loop probe did not decrease the objective."],
        compile_diagnostics_requested=True,
        compile_diagnostics_enabled=False,
        compile_diagnostics_disable_reason=disable_reason,
        deterministic_gpu_reductions=False,
    )

    assert note["reproduced"] is True
    assert note["trace_dir"] == "/tmp/xprof"
    assert note["first_bad_region"] == "single_stage.outer_optimizer_initial_phase"
    assert (
        note["first_bad_region_source"]
        == "TARGET_LANE_SCALED_PHASE1_DIAGNOSIS.first_nonfinite_stage"
    )
    assert note["first_bad_region_detail"] == "steepest_descent_trial"
    assert note["compile_behavior"] == {
        "diagnostics_requested": True,
        "diagnostics_enabled": False,
        "jax_log_compiles": False,
        "jax_explain_cache_misses": False,
        "cache_reuse_evidence_valid": False,
        "disabled_reason": disable_reason,
    }


def test_single_stage_outer_loop_probe_builds_phase1_note_from_invalid_state_events():
    note = single_stage_outer_loop_probe.build_phase1_diagnostic_note(
        {
            "iterations": 4,
            "TARGET_LANE_INVALID_STATE_DIAGNOSIS": {
                "events": [
                    {
                        "phase": "phase2",
                        "iteration": 2,
                        "line_search_failed": True,
                        "nonfinite_step": False,
                        "stalled_step": True,
                        "valid_curvature": True,
                        "ls_status": 3,
                    }
                ]
            },
        },
        failures=["Single-stage outer-loop probe produced a non-finite FINAL_IOTA."],
        compile_diagnostics_requested=False,
        compile_diagnostics_enabled=False,
        compile_diagnostics_disable_reason=None,
        deterministic_gpu_reductions=True,
    )

    assert note["reproduced"] is True
    assert note["first_bad_region"] == "single_stage.outer_optimizer"
    assert (
        note["first_bad_region_source"]
        == "TARGET_LANE_INVALID_STATE_DIAGNOSIS.events[0]"
    )
    assert note["first_bad_region_detail"] == {
        "phase": "phase2",
        "iteration": 2,
        "line_search_failed": True,
        "nonfinite_step": False,
        "stalled_step": True,
        "valid_curvature": True,
        "ls_status": 3,
    }
    assert note["deterministic_gpu_reductions"] is True


def _grouped_adjoint_memory_metrics(*, snapshots, **overrides):
    metrics = {
        "adjoint_residual_rel": 1e-12,
        "adjoint_norm": 1.0,
        "adjoint_finite": True,
        "implicit_gradient_finite": True,
        "implicit_gradient_norm": 1.0,
        "snapshots": snapshots,
        "grouped_vjp_timings": _complete_grouped_vjp_timings(),
        "cache_stability": _complete_grouped_vjp_cache_stability(),
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
        _grouped_adjoint_snapshot("after_dofs_gradient_projection", 6.75),
        _grouped_adjoint_snapshot("after_norm_metrics", 7.0),
    ]


def _complete_grouped_vjp_timings():
    return {
        "requested_stream_pass_count": 3,
        "stream_pass_count": 3,
        "group_count": 2,
        "first_stream_s": 0.4,
        "first_compile_time_s": 0.25,
        "first_compile_time_note": (
            "First grouped-VJP group call measured with block_until_ready; "
            "includes compilation plus first group execution."
        ),
        "first_stream_steady_state_group_median_s": 0.1,
        "warm_stream_times_s": [0.4, 0.44],
        "steady_state_grouped_vjp_time_s": 0.42,
        "steady_state_grouped_vjp_per_group_s": 0.2,
        "total_representative_run_wall_s": 3.5,
        "steady_state_grouped_vjp_wall_fraction": 0.12,
        "passes": [],
    }


def _grouped_vjp_timings_for_fraction(steady_state_s: float, total_wall_s: float):
    timings = _complete_grouped_vjp_timings()
    timings["warm_stream_times_s"] = [steady_state_s]
    timings["steady_state_grouped_vjp_time_s"] = steady_state_s
    timings["total_representative_run_wall_s"] = total_wall_s
    timings["steady_state_grouped_vjp_wall_fraction"] = steady_state_s / total_wall_s
    return timings


def _complete_grouped_vjp_cache_stability():
    return {
        "diagnostics_requested": True,
        "diagnostics_enabled": True,
        "jax_log_compiles": True,
        "jax_explain_cache_misses": True,
        "warm_pass_count": 2,
        "warm_compile_event_count": 0,
        "warm_cache_miss_count": 0,
        "unexpected_steady_state_recompile": False,
        "production_pass": {
            "compile_event_count": 1,
            "cache_miss_count": 1,
            "compile_messages": ["Compiling production grouped VJP"],
            "cache_miss_messages": ["cache miss production grouped VJP"],
        },
        "warm_passes": {
            "compile_event_count": 0,
            "cache_miss_count": 0,
            "compile_messages": [],
            "cache_miss_messages": [],
        },
    }


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


def _h200_production_proof_workflow_path() -> Path:
    return _workflow_path("jax_h200_production_proof.yml")


def _workflow_job_section(
    workflow_text: str,
    job_name: str,
    next_job_name: str | None = None,
) -> str:
    section = workflow_text.split(f"  {job_name}:", maxsplit=1)[1]
    if next_job_name is None:
        return section
    return section.split(f"  {next_job_name}:", maxsplit=1)[0]


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
            get_adjoint_runtime_state=lambda: (_ for _ in ()).throw(
                RuntimeError("legacy full-pytree adjoint fallback should stay disabled")
            )
        )
    )

    with pytest.raises(RuntimeError, match="legacy full-pytree adjoint fallback"):
        list(adjoint_probe_common.iter_grouped_adjoint_cotangents(jr_jax, np.ones(2)))


def test_compute_adjoint_state_uses_runtime_adjoint_state():
    recorded = {}
    rhs = np.array([1.0, -2.0])
    expected_spec = object()

    def fake_compute_dJ_ds(coil_set_spec, iota, G, weight_inv_modB):
        recorded["compute_dJ_ds_args"] = (coil_set_spec, iota, G, weight_inv_modB)
        return rhs

    def fake_solve_transpose_with_status(passed_rhs):
        recorded["solve_transpose_rhs"] = np.asarray(passed_rhs)
        return passed_rhs, True

    def fake_apply_transpose(passed_adjoint):
        recorded["apply_transpose_adjoint"] = np.asarray(passed_adjoint)
        return passed_adjoint

    jr_jax = types.SimpleNamespace(
        boozer_surface=types.SimpleNamespace(
            get_adjoint_runtime_state=lambda: types.SimpleNamespace(
                solved_state=types.SimpleNamespace(
                    iota=0.1,
                    G=0.2,
                    weight_inv_modB=False,
                ),
                solve_transpose=fake_solve_transpose_with_status,
                solve_transpose_with_status=fake_solve_transpose_with_status,
                apply_transpose=fake_apply_transpose,
            )
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
    np.testing.assert_allclose(recorded["solve_transpose_rhs"], rhs)
    np.testing.assert_allclose(recorded["apply_transpose_adjoint"], rhs)
    np.testing.assert_allclose(recorded["coil_dofs"], np.array([3.0, 4.0]))
    assert recorded["compute_dJ_ds_args"] == (expected_spec, 0.1, 0.2, False)


def test_compute_adjoint_state_raises_when_runtime_operator_solve_fails():
    jr_jax = types.SimpleNamespace(
        boozer_surface=types.SimpleNamespace(
            get_adjoint_runtime_state=lambda: types.SimpleNamespace(
                solved_state=types.SimpleNamespace(
                    iota=0.1,
                    G=0.2,
                    weight_inv_modB=False,
                ),
                linearization_kind="hessian",
                solve_transpose_with_status=lambda rhs: (rhs, False),
            )
        ),
        biotsavart=types.SimpleNamespace(
            x=np.array([3.0, 4.0]),
            coil_set_spec_from_dofs=lambda _coil_dofs: object(),
        ),
        _compute_dJ_ds=lambda *_args: np.array([1.0, -2.0]),
    )

    with pytest.raises(RuntimeError, match="operator-backed transpose solve"):
        adjoint_probe_common.compute_adjoint_state(jr_jax)


def test_accumulate_grouped_adjoint_dofs_gradient_uses_biotsavart_projection_api():
    recorded = {}

    def fake_projection(d_coil_arrays, coil_indices, *, coil_dofs):
        recorded.setdefault("projection_args", []).append((d_coil_arrays, coil_indices))
        projection_index = len(recorded["projection_args"])
        return np.asarray(coil_dofs, dtype=float) + float(projection_index)

    bs_jax = types.SimpleNamespace(
        x=np.array([3.0, 4.0]),
        coil_cotangents_to_dofs_gradient=fake_projection,
    )
    grouped = [
        (np.array([[1.0, 2.0, 3.0]]), [0, 2]),
        (np.array([[4.0, 5.0, 6.0]]), [1]),
    ]

    gradient = adjoint_probe_common.accumulate_grouped_adjoint_dofs_gradient(
        bs_jax,
        iter(grouped),
    )

    assert len(recorded["projection_args"]) == 2
    np.testing.assert_allclose(recorded["projection_args"][0][0][0], grouped[0][0])
    assert recorded["projection_args"][0][1] == [[0, 2]]
    np.testing.assert_allclose(recorded["projection_args"][1][0][0], grouped[1][0])
    assert recorded["projection_args"][1][1] == [[1]]
    np.testing.assert_allclose(np.asarray(gradient), np.array([9.0, 11.0]))


def test_compute_gradient_l2_metrics_matches_flat_gradient_norm():
    norm, finite = compute_gradient_l2_metrics(np.array([3.0, 4.0]))

    assert norm == pytest.approx(5.0)
    assert finite is True


def test_grouped_vjp_timing_recorder_streams_without_materializing_all_groups():
    yielded_groups = [
        (np.array([[1.0, 2.0, 3.0]]), [0]),
        (np.array([[4.0, 5.0, 6.0]]), [1]),
    ]
    pull_count = 0

    def stream_group_vjps(_adjoint):
        nonlocal pull_count
        for entry in yielded_groups:
            pull_count += 1
            yield entry

    jr_jax = types.SimpleNamespace(
        boozer_surface=types.SimpleNamespace(
            get_adjoint_runtime_state=lambda: types.SimpleNamespace(
                stream_group_vjps=stream_group_vjps,
            )
        )
    )
    recorder = _GroupedVJPTimingRecorder(requested_stream_pass_count=2)
    first_stream = recorder.timed_cotangents(
        jr_jax,
        np.ones(2),
        label="production_dofs_gradient_projection",
    )

    first_entry = next(first_stream)
    assert pull_count == 1
    np.testing.assert_allclose(first_entry[0], yielded_groups[0][0])
    second_entry = next(first_stream)
    assert pull_count == 2
    np.testing.assert_allclose(second_entry[0], yielded_groups[1][0])
    with pytest.raises(StopIteration):
        next(first_stream)

    summary = recorder.summary(total_representative_run_wall_s=1.25)

    assert summary["requested_stream_pass_count"] == 2
    assert summary["stream_pass_count"] == 1
    assert summary["group_count"] == 2
    assert summary["first_compile_time_s"] >= 0.0
    assert summary["total_representative_run_wall_s"] == pytest.approx(1.25)


def test_compute_direct_and_total_gradients_uses_live_boozer_g(monkeypatch):
    direct_gradient = np.array([7.0, 11.0])
    total_gradient = np.array([5.0, 9.0])
    implicit_correction = np.array([2.0, 2.0])
    recorded = {}

    def fake_value_and_direct_coil_gradient(
        objective_value_and_grad,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    ):
        recorded["value_and_direct_args"] = (
            objective_value_and_grad,
            coil_dofs.copy(),
            x_inner.copy(),
            optimize_G,
            weight_inv_modB,
        )
        return 0.0, direct_gradient

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
        surfaceobjectives_jax_module,
        "_value_and_direct_coil_gradient",
        fake_value_and_direct_coil_gradient,
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
        objective_value_and_grad,
        coil_dofs,
        x_inner,
        optimize_G,
        weight_inv_modB,
    ) = recorded["value_and_direct_args"]
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


def test_grouped_adjoint_memory_probe_rejects_missing_timings_or_warm_recompile():
    failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(
            snapshots=_complete_grouped_adjoint_snapshots(),
            grouped_vjp_timings={},
            cache_stability={
                **_complete_grouped_vjp_cache_stability(),
                "unexpected_steady_state_recompile": True,
            },
        ),
        budget=_grouped_adjoint_budget("cpu"),
    )

    assert any("required timing fields" in failure for failure in failures)
    assert any("steady-state grouped-VJP recompilation" in failure for failure in failures)


def test_grouped_adjoint_memory_probe_rejects_unrepresentative_grouped_vjp():
    cold_path_failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(
            snapshots=_complete_grouped_adjoint_snapshots(),
            grouped_vjp_timings=_grouped_vjp_timings_for_fraction(0.04, 1.0),
        ),
        budget=_grouped_adjoint_budget("cpu"),
    )
    gray_zone_failures = evaluate_grouped_adjoint_memory_probe(
        _grouped_adjoint_memory_metrics(
            snapshots=_complete_grouped_adjoint_snapshots(),
            grouped_vjp_timings=_grouped_vjp_timings_for_fraction(0.08, 1.0),
        ),
        budget=_grouped_adjoint_budget("cpu"),
    )

    assert any("below 5%" in failure for failure in cold_path_failures)
    assert any("below the 10%" in failure for failure in gray_zone_failures)


def test_grouped_adjoint_representative_wall_time_ignores_warm_passes():
    snapshots = _complete_grouped_adjoint_snapshots()
    snapshots.extend(
        [
            _grouped_adjoint_snapshot("warm_stream_1_end", 20.0),
            _grouped_adjoint_snapshot("after_norm_metrics", 21.0),
        ]
    )

    assert _representative_run_wall_s(snapshots) == pytest.approx(3.375)


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


def test_grouped_adjoint_baseline_comparison_enforces_ship_gate():
    snapshots = _complete_grouped_adjoint_snapshots()
    snapshots[-1]["gpu_memory_mb"] = 600.0
    baseline_payload = {
        "timings": {"steady_state_grouped_vjp_time_s": 1.0},
        "memory": {"peak_gpu_memory_mb": 1000.0},
    }
    passing_metrics = _grouped_adjoint_memory_metrics(
        snapshots=snapshots,
        grouped_vjp_timings=_grouped_vjp_timings_for_fraction(0.75, 3.0),
    )
    passing_metrics["baseline_comparison"] = (
        _build_grouped_adjoint_baseline_comparison(
            current_metrics=passing_metrics,
            baseline_payload=baseline_payload,
            baseline_json="/tmp/baseline.json",
        )
    )

    assert passing_metrics["baseline_comparison"]["speedup_gate_passed"] is True
    assert passing_metrics["baseline_comparison"]["memory_gate_passed"] is True
    assert (
        evaluate_grouped_adjoint_memory_probe(
            passing_metrics,
            budget=_grouped_adjoint_budget("cpu"),
        )
        == []
    )

    snapshots = _complete_grouped_adjoint_snapshots()
    snapshots[-1]["gpu_memory_mb"] = 800.0
    failing_metrics = _grouped_adjoint_memory_metrics(
        snapshots=snapshots,
        grouped_vjp_timings=_grouped_vjp_timings_for_fraction(0.9, 3.0),
    )
    failing_metrics["baseline_comparison"] = (
        _build_grouped_adjoint_baseline_comparison(
            current_metrics=failing_metrics,
            baseline_payload=baseline_payload,
            baseline_json="/tmp/baseline.json",
        )
    )

    failures = evaluate_grouped_adjoint_memory_probe(
        failing_metrics,
        budget=_grouped_adjoint_budget("cpu"),
    )

    assert any("ship gate missed" in failure for failure in failures)


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
    assert payload["timings"]["first_compile_time_s"] == pytest.approx(0.25)
    assert payload["baseline_comparison"] is None
    assert payload["cache_stability"]["warm_compile_event_count"] == 0
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


def test_tier5_single_stage_outer_loop_probe_args_thread_benchmark_mode():
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
        single_stage_outer_loop_maxiter=3,
        profile_target_lane_batch_size=8,
    )

    command = tier5_performance_characterization._single_stage_outer_loop_probe_args(
        args
    )

    assert "--maxiter" in command
    maxiter_idx = command.index("--maxiter")
    assert command[maxiter_idx + 1] == "3"
    assert "--optimizer-backend" in command
    optimizer_backend_idx = command.index("--optimizer-backend")
    assert command[optimizer_backend_idx + 1] == "ondevice"
    assert "--benchmark-mode" in command
    batch_size_idx = command.index("--profile-target-lane-batch-size")
    assert command[batch_size_idx + 1] == "8"


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
    assert stage2_e2e_args.optimizer_backend == "ondevice"

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
    assert single_stage_outer_loop_args.optimizer_backend == "ondevice"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adjoint_fd_validation.py",
            "--output-json",
            str(tmp_path / "adjoint-tier4.json"),
        ],
    )
    adjoint_fd_args = adjoint_fd_validation_module.parse_args()
    assert adjoint_fd_args.plasma_surf_filename == DEFAULT_PLASMA_SURF_FILENAME
    assert adjoint_fd_args.equilibria_dir == str(DEFAULT_EQUILIBRIA_DIR)
    assert adjoint_fd_args.optimizer_backend == "ondevice"


def test_production_boozer_probe_defaults_jax_lane_to_ondevice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_boozer_parity_probe.py",
            "--output-json",
            str(tmp_path / "production-boozer.json"),
        ],
    )

    args = production_boozer_parity_probe_module.parse_args()

    assert args.optimizer_backend == "ondevice"


def test_run_code_parity_probe_defaults_jax_lane_to_ondevice(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_code_parity_probe.py"])

    args = run_code_parity_probe_module.parse_args()

    assert args.optimizer_backend == "ondevice"


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
    assert "--record-jax-compile-diagnostics" in grouped_command["args"]
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
    assert "--record-jax-compile-diagnostics" in workflow_text
    assert "--device-memory-profile-out benchmark_artifacts/grouped_adjoint_memory_profile.prof" in workflow_text
    assert "if-no-files-found: ignore" in workflow_text
    assert "Fail on benchmark gate regressions" in workflow_text


def test_gpu_parity_workflow_enforces_strict_transfer_guard_contract():
    workflow_text = _gpu_parity_workflow_path().read_text(encoding="utf-8")

    _assert_named_benchmark_env_bootstrap(workflow_text)
    assert "SIMSOPT_BACKEND_MODE: jax_gpu_parity" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert 'XLA_FLAGS: --xla_gpu_deterministic_ops=true' in workflow_text
    assert "setuptools_scm" not in workflow_text
    assert "benchmarks/stage2_value_gradient_parity.py" in workflow_text
    assert "--fixture real" in workflow_text
    assert "benchmarks/single_stage_outer_loop_probe.py" in workflow_text
    assert "--optimizer-backend ondevice" in workflow_text
    assert "tests/geo/test_boozer_residual_jax.py \\" in workflow_text
    assert "-k gpu_parity" in workflow_text
    assert "benchmark_artifacts/stage2_value_gradient_parity_real_cuda.json" in workflow_text
    assert "benchmark_artifacts/single_stage_outer_loop_cuda.json" in workflow_text


def test_gpu_parity_workflow_adds_full_suite_disallow_lane():
    workflow_text = _gpu_parity_workflow_path().read_text(encoding="utf-8")

    _assert_named_benchmark_env_bootstrap(workflow_text)
    assert "gpu-full-suite-disallow:" in workflow_text
    assert "name: GPU full suite (CUDA, transfer_guard=disallow)" in workflow_text
    assert "runs-on: [self-hosted, gpu]" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert 'XLA_FLAGS: --xla_gpu_deterministic_ops=true' in workflow_text
    assert 'JAX_ENABLE_X64: "1"' in workflow_text
    assert 'PYTHONUNBUFFERED: "1"' in workflow_text
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE: "false"' in workflow_text
    assert "Run full pytest suite under CUDA strict transfer guard" in workflow_text
    assert "python -m pytest tests \\" in workflow_text
    assert "--capture=tee-sys" in workflow_text
    assert "-o log_cli=true" in workflow_text
    assert "-o log_cli_level=INFO" in workflow_text


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
        ".github/workflows/jax_h200_production_proof.yml",
    )

    _assert_named_benchmark_env_bootstrap(workflow_text)
    assert "jax-gpu-e2e:" in workflow_text
    assert "name: JAX GPU e2e smoke (CUDA, ondevice)" in workflow_text
    assert "runs-on: [self-hosted, gpu]" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert 'XLA_FLAGS: --xla_gpu_deterministic_ops=true' in workflow_text
    assert 'JAX_ENABLE_X64: "1"' in workflow_text
    gpu_e2e = _workflow_job_section(
        workflow_text,
        "jax-gpu-e2e",
        "jax-gpu-strict-purity",
    )
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE: "false"' in gpu_e2e
    assert "benchmarks/stage2_e2e_comparison.py" in workflow_text
    assert "benchmarks/single_stage_init_parity.py" in workflow_text
    assert "--platform cuda" in workflow_text
    assert "--optimizer-backend ondevice" in workflow_text
    for required_path in required_paths:
        assert required_path in workflow_text


def test_smoke_workflow_adds_cuda_strict_transfer_guard_pytest_lane():
    workflow_text = _smoke_workflow_path().read_text(encoding="utf-8")
    _assert_named_benchmark_env_bootstrap(workflow_text)
    assert "tests/integration/test_single_stage_jax_cpu_reference.py" in workflow_text
    assert "tests/integration/test_single_stage_physics_parity.py" in workflow_text
    assert "jax-gpu-strict-purity:" in workflow_text
    assert "name: JAX GPU strict purity (CUDA, transfer_guard=disallow)" in workflow_text
    assert "runs-on: [self-hosted, gpu]" in workflow_text
    assert 'SIMSOPT_BACKEND_STRICT: "1"' in workflow_text
    assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in workflow_text
    assert 'XLA_FLAGS: --xla_gpu_deterministic_ops=true' in workflow_text
    assert 'JAX_ENABLE_X64: "1"' in workflow_text
    strict_purity = _workflow_job_section(workflow_text, "jax-gpu-strict-purity")
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE: "false"' in strict_purity
    assert "tests/test_jax_import_smoke.py" in workflow_text
    assert "gpu_ondevice_loops_with_host_constants" in workflow_text
    assert "grouped_biot_savart_gpu_spec_eval" in workflow_text
    assert "grouped_biot_savart_gpu_current_arrays" in workflow_text
    assert "stage2_target_objective_ondevice_entry" in workflow_text
    assert "tests/geo/test_boozersurface_jax.py" in workflow_text
    assert "run_code_traceable_exact_executes_inner_solve_on_gpu" in workflow_text
    assert "run_code_traceable_lm_ondevice_executes_inner_solve_on_gpu" in workflow_text
    assert "tests/integration/test_single_stage_jax_cpu_reference.py" in workflow_text
    assert "TestRealFixtureGpuM5Parity" in workflow_text
    assert "test_ls_solve_parity_production_scale_gpu_under_disallow" in workflow_text
    assert "tests/integration/test_single_stage_physics_parity.py" in workflow_text
    assert "TestSingleStageOuterLoopGpuProof" in workflow_text


def test_smoke_workflow_runs_accessibility_with_simsoptpp_lane():
    workflow_text = _smoke_workflow_path().read_text(encoding="utf-8")
    public_unit = workflow_text.split("  jax-public-guardrails:", maxsplit=1)[0]
    private_optimizer = workflow_text.split(
        "  jax-private-optimizer:",
        maxsplit=1,
    )[1]

    assert "src/simsopt/geo/accessibility.py" in workflow_text
    assert "tests/geo/test_accessibility.py" in workflow_text
    assert "Run accessibility FD/Hessian tests" in private_optimizer
    assert "python -m pytest tests/geo/test_accessibility.py -v --tb=short" in private_optimizer
    assert "tests/geo/test_accessibility.py \\" not in public_unit


def test_h200_production_proof_workflow_launches_real_h200_hf_job():
    workflow_text = _h200_production_proof_workflow_path().read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow_text
    assert "image:" in workflow_text
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" in workflow_text
    assert 'python -m pip install --upgrade "huggingface_hub>=1.12.0"' in workflow_text
    assert "hf jobs hardware | tee hf_jobs_hardware.txt" in workflow_text
    assert "grep -E '^h200[[:space:]]' hf_jobs_hardware.txt" in workflow_text
    assert "benchmarks/hf_jobs/launch_production_gpu_proof.py" in workflow_text
    assert "--hardware h200" in workflow_text
    assert "--platform cuda" in workflow_text
    assert "--no-detach" in workflow_text
    assert "--repo-url \"https://github.com/${{ github.repository }}.git\"" in workflow_text
    assert "--repo-ref \"${{ github.ref_name }}\"" in workflow_text
    assert "--repo-sha \"${{ github.sha }}\"" in workflow_text
    assert "--single-stage-warm-start-run-dir" in workflow_text
    assert "--single-stage-jax-runtime-seed-spec" in workflow_text
    assert "Set exactly one single-stage seed input." in workflow_text


def test_smoke_workflow_pins_jax_ci_contract_ratchet_gate():
    workflow_text = _smoke_workflow_path().read_text(encoding="utf-8")

    assert "Run CI contract helper tests" in workflow_text
    assert "tests/test_benchmark_helpers.py \\" in workflow_text
    assert (
        'jax_ci_contract_ratchet_rel_tol_tightens_without_loosening'
        in workflow_text
    )
    assert "jax_ci_contract_reduction_order_probe_tracks_ulp_distance" in workflow_text
    assert (
        "jax_ci_contract_same_device_probe_requires_bitwise_identity"
        in workflow_text
    )
    assert "jax_ci_contract_payload_tracks_ratchet_and_pass_state" in workflow_text


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
        grouped_adjoint_baseline_json=None,
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
    assert "--baseline-json" not in grouped_command
    assert grouped_env["SIMSOPT_JAX_TRANSFER_GUARD"] == "disallow"
    assert grouped_env["JAX_COMPILATION_CACHE_DIR"] == str(dense_audit_cache_dir)
    assert "render_benchmark_report.py" in report_command[1]
    assert report_env["SIMSOPT_JAX_TRANSFER_GUARD"] == "log"
    assert dense_audit_cache_dir.is_dir()


def test_single_stage_outer_loop_probe_resolves_expected_boozer_method():
    assert resolve_boozer_least_squares_algorithm("scipy") == "quasi-newton"
    with pytest.raises(ValueError, match="require boozer_optimizer_backend"):
        resolve_boozer_least_squares_algorithm("hybrid")
    assert resolve_boozer_least_squares_algorithm("ondevice") == "quasi-newton"
    assert resolve_boozer_optimizer_method("scipy") == "scipy"
    assert resolve_boozer_optimizer_method("scipy", limited_memory=True) == "scipy"
    with pytest.raises(ValueError, match="require boozer_optimizer_backend"):
        resolve_boozer_optimizer_method("hybrid")
    with pytest.raises(ValueError, match="require boozer_optimizer_backend"):
        resolve_boozer_optimizer_method("hybrid", limited_memory=True)
    assert resolve_boozer_optimizer_method("ondevice") == "bfgs-ondevice"
    assert (
        resolve_boozer_optimizer_method(
            "ondevice",
            least_squares_algorithm="quasi-newton",
        )
        == "bfgs-ondevice"
    )
    assert (
        resolve_boozer_optimizer_method(
            "ondevice",
            limited_memory=True,
            least_squares_algorithm="quasi-newton",
        )
        == "lbfgs-ondevice"
    )
    with pytest.raises(ValueError, match="least_squares_algorithm='lm'"):
        resolve_boozer_optimizer_method(
            "ondevice",
            limited_memory=True,
            least_squares_algorithm="lm",
        )


def test_single_stage_outer_loop_probe_resolves_boozer_backend():
    assert resolve_boozer_optimizer_backend("ondevice", None) == "ondevice"
    assert resolve_boozer_optimizer_backend("scipy-jax", None) == "ondevice"
    assert resolve_boozer_optimizer_backend("scipy-jax-fullgraph", None) == "scipy"
    assert resolve_boozer_optimizer_backend("ondevice", "scipy") == "scipy"
    with pytest.raises(ValueError, match="require boozer_optimizer_backend"):
        resolve_boozer_optimizer_backend("hybrid", None)


def test_single_stage_outer_loop_contract_matches_probe_defaults():
    contract = single_stage_proof_contract()

    assert contract["default_maxiter"] == 10
    assert contract["min_iterations"] == 10
    assert contract["require_objective_decrease"] is True
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


def test_stage2_e2e_trajectory_improves_accepts_machine_scale_roundoff_drift():
    trajectory = [
        {"J": 0.001435769751044976},
        {"J": 0.0014357697510449764},
    ]

    assert stage2_e2e_comparison_module._trajectory_improves(trajectory)


def test_stage2_e2e_trajectory_improves_rejects_real_objective_regression():
    trajectory = [
        {"J": 0.001435769751044976},
        {"J": 0.001435769752044976},
    ]

    assert not stage2_e2e_comparison_module._trajectory_improves(trajectory)


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
        maxiter=60,
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
    assert payload["proof_parity"]["cpu_oracle_value"] == pytest.approx(1.0)
    assert payload["proof_parity"]["gpu_value"] == pytest.approx(1.0 + 1e-7)
    assert payload["proof_parity"]["value_rtol"] == pytest.approx(1e-4)
    assert payload["proof_parity"]["gradient_rtol"] == pytest.approx(1e-9)
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
        maxiter=60,
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
    assert summary["headline_metric"] == "outer_speedup_vs_cpu"
    assert summary["headline_speedup_vs_cpu"] == pytest.approx(0.8)
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
    assert summary["headline_metric"] == "outer_speedup_vs_cpu"
    assert summary["headline_speedup_vs_cpu"] == pytest.approx(0.8)


def test_summarize_single_stage_outer_loop_performance_probe_records_speedup():
    payload = {
        "passed": True,
        "timings": {
            "cpu_elapsed_s": 20.0,
            "jax_elapsed_s": 5.0,
        },
    }

    summary = summarize_single_stage_outer_loop_performance_probe(
        payload=payload,
        outer_elapsed_s=30.0,
        lane_label="jax-cuda",
    )

    assert summary["name"] == TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG
    assert summary["speedup_vs_cpu"] == pytest.approx(4.0)
    assert summary["timing_semantics"] == "short_outer_loop_probe_with_cpu_reference"
    assert summary["supports_performance_headline"] is False
    assert summary["counts_toward_phase_pass"] is False


def _failed_outer_loop_probe_payload() -> dict[str, object]:
    return {
        "passed": False,
        "timings": {"cpu_elapsed_s": 20.0, "jax_elapsed_s": 5.0},
        "failures": ["probe failed"],
    }


def test_timed_probe_accepts_written_json_from_failed_informational_probe(
    monkeypatch, tmp_path
):
    observed_envs: list[dict[str, str]] = []

    def fake_run_python_script(script_path, args, **kwargs):
        output_json = Path(args[args.index("--output-json") + 1])
        output_json.write_text(
            json.dumps(_failed_outer_loop_probe_payload()),
            encoding="utf-8",
        )
        observed_envs.append(dict(kwargs["env"]))
        raise RuntimeError("informational probe failed")

    monkeypatch.setattr(
        tier5_performance_characterization,
        "run_python_script",
        fake_run_python_script,
    )

    payload, outer_elapsed_s = tier5_performance_characterization._timed_probe(
        tmp_path / "probe.py",
        ["--platform", "cuda"],
        platform="cuda",
        accept_failed_output_json=True,
    )

    assert payload["passed"] is False
    assert payload["failures"] == ["probe failed"]
    assert outer_elapsed_s >= 0.0
    assert observed_envs


def test_timed_probe_disables_compilation_cache_for_cpu_children(monkeypatch, tmp_path):
    observed_envs: list[dict[str, str]] = []

    def fake_run_python_script(script_path, args, **kwargs):
        output_json = Path(args[args.index("--output-json") + 1])
        output_json.write_text(json.dumps({"passed": True}), encoding="utf-8")
        observed_envs.append(dict(kwargs["env"]))

    monkeypatch.setattr(
        tier5_performance_characterization,
        "run_python_script",
        fake_run_python_script,
    )
    monkeypatch.setenv(_JAX_COMPILATION_CACHE_ENV_VAR, "/tmp/parent-cache")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR, "0")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR, "-1")
    monkeypatch.setenv(_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR, "all")

    payload, outer_elapsed_s = tier5_performance_characterization._timed_probe(
        tmp_path / "probe.py",
        ["--platform", "cpu"],
        platform="cpu",
    )

    assert payload["passed"] is True
    assert outer_elapsed_s >= 0.0
    assert len(observed_envs) == 1
    env = observed_envs[0]
    assert _JAX_COMPILATION_CACHE_ENV_VAR not in env
    assert _JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_ENV_VAR not in env
    assert _JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_ENV_VAR not in env
    assert _JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES_ENV_VAR not in env
    assert env[_SIMSOPT_DISABLE_COMPILATION_CACHE_ENV_VAR] == "1"
    assert env[_SIMSOPT_COMPILATION_CACHE_POLICY_ENV_VAR] == "disabled"


def test_single_stage_init_case_threads_reference_trace_optimizer_to_cpu_lane(
    monkeypatch,
    tmp_path,
):
    args = _single_stage_case_args(tmp_path)
    args.reference_optimizer_method = "lbfgs-trace"
    observed_invocations = _observe_single_stage_case_invocations(monkeypatch, tmp_path)

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

    single_stage_init_parity_module._run_single_stage_case(
        args,
        "cpu",
        platform="cpu",
        benchmark_mode=True,
        load_surface_gamma=False,
    )

    assert len(observed_invocations) == 1
    command, _env = observed_invocations[0]
    reference_method_flag_index = command.index("--reference-optimizer-method")
    assert command[reference_method_flag_index + 1] == "lbfgs-trace"
    initial_step_scale_flag_index = command.index("--initial-step-scale")
    assert command[initial_step_scale_flag_index + 1] == "1.0"
    initial_step_maxiter_flag_index = command.index("--initial-step-maxiter")
    assert command[initial_step_maxiter_flag_index + 1] == "0"
    outer_maxls_flag_index = command.index("--outer-maxls")
    assert command[outer_maxls_flag_index + 1] == "8"
    assert "--optimizer-backend" not in command


def test_tier5_provenance_extra_uses_real_single_stage_fixture():
    args = argparse.Namespace(
        platform="cuda",
        optimizer_backend="ondevice",
        plasma_surf_filename="fixture.nc",
        stage2_bs_path="/tmp/seed.json",
        stage2_nphi=255,
        stage2_ntheta=64,
        single_stage_nphi=255,
        single_stage_ntheta=64,
        mpol=8,
        ntor=6,
        maxiter=20,
        single_stage_outer_loop_maxiter=1,
        samples=3,
        eps=1e-4,
        phase="gpu",
    )

    provenance_extra = tier5_performance_characterization._tier5_provenance_extra(
        args,
        benchmark_mode=False,
    )

    assert provenance_extra["fixture"] == "real-single-stage-init"
    assert provenance_extra["lane"] == resolve_probe_lane(
        optimizer_backend="ondevice"
    )
    assert provenance_extra["phase"] == "gpu"


def _set_summary_rung_passed(
    summary: list[dict[str, object]],
    rung_name: str,
    *,
    passed: bool,
) -> None:
    for item in summary:
        if item["name"] == rung_name:
            item["passed"] = passed
            return
    raise AssertionError(f"Missing rung in summary: {rung_name}")


def test_build_tier5_performance_contract_routes_parity_and_headline_sources():
    summary = [
        {
            "name": "tier1b_real_stage2",
            "timing_semantics": "correctness_probe_only",
            "supports_performance_headline": False,
        },
        {
            "name": "tier2_stage2_e2e",
            "headline_metric": "outer_speedup_vs_cpu",
            "headline_speedup_vs_cpu": 0.8,
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
            "name": TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
            "timing_semantics": "short_outer_loop_probe_with_cpu_reference",
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
        "summary_by_name.tier2_stage2_e2e.outer_speedup_vs_cpu"
    )
    assert contract["headline_performance_source"]["speedup_vs_cpu"] == pytest.approx(0.8)
    assert contract["sharding_source"]["active_path"] == (
        "rungs.tier2_stage2_e2e.provenance.sharding_active"
    )
    assert contract["sharding_source"]["strategy_path"] == (
        "rungs.tier2_stage2_e2e.provenance.sharding_strategy"
    )
    assert contract["do_not_use_for_performance_headline"] == [
        "tier1b_real_stage2",
        "tier3_single_stage_init",
        TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
        "tier4_adjoint_fd",
    ]


def _tier5_gpu_phase_payload(**overrides):
    payload = {
        "phase": "gpu",
        "provenance": {"phase": "gpu", "platform_request": "cuda"},
        "rungs": {
            "tier1b_real_stage2": {"passed": True, "provenance": {"platform": "cuda"}},
            "tier2_stage2_e2e": {
                "passed": True,
                "provenance": {
                    "sharding_active": False,
                    "sharding_strategy": "none",
                    "sharding_device_count": 1,
                },
            },
            "tier3_single_stage_init": {"passed": True},
            TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG: {"passed": True},
            "tier4_adjoint_fd_lane": {"passed": True},
        },
        "summary": [
            {
                "name": "tier1b_real_stage2",
                "passed": True,
                "outer_elapsed_s": 3.0,
                "cpu_elapsed_s": 1.0,
                "lane_elapsed_s": 0.5,
                "lane_label": "jax-cuda",
                "timing_semantics": "correctness_probe_only",
                "supports_performance_headline": False,
            },
            {
                "name": "tier2_stage2_e2e",
                "passed": True,
                "outer_elapsed_s": 9.0,
                "cpu_elapsed_s": 20.0,
                "lane_elapsed_s": 10.0,
                "lane_outer_elapsed_s": 10.0,
                "lane_warm_elapsed_s": 5.0,
                "warm_speedup_vs_cpu": 4.0,
                "outer_speedup_vs_cpu": 2.0,
                "lane_compile_overhead_s": 5.0,
                "headline_metric": "outer_speedup_vs_cpu",
                "headline_speedup_vs_cpu": 2.0,
                "lane_label": "jax-cuda",
                "timing_semantics": "separate_cold_end_to_end_and_warm_steady_state",
                "supports_performance_headline": True,
            },
            {
                "name": "tier3_single_stage_init",
                "passed": True,
                "outer_elapsed_s": 7.0,
                "cpu_elapsed_s": 3.0,
                "lane_elapsed_s": 2.0,
                "lane_label": "jax-cuda",
                "timing_semantics": "initialization_probe_only",
                "supports_performance_headline": False,
            },
            {
                "name": TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
                "passed": True,
                "outer_elapsed_s": 8.0,
                "cpu_elapsed_s": 4.0,
                "lane_elapsed_s": 2.0,
                "speedup_vs_cpu": 2.0,
                "lane_label": "jax-cuda",
                "timing_semantics": "short_outer_loop_probe_with_cpu_reference",
                "supports_performance_headline": False,
                "counts_toward_phase_pass": False,
            },
        ],
        "probe_timings": {
            "tier1b_real_stage2": 3.0,
            "tier2_stage2_e2e": 9.0,
            "tier3_single_stage_init": 7.0,
            TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG: 8.0,
            "tier4_adjoint_fd_lane": 11.0,
        },
        "aggregate": {
            "lane_label": "jax-cuda",
            "pending_rungs": ["tier4_adjoint_fd"],
        },
    }
    payload.update(overrides)
    return payload


def _tier5_cpu_phase_payload(**overrides):
    payload = {
        "phase": "cpu",
        "provenance": {"phase": "cpu", "platform_request": "cpu"},
        "rungs": {
            "tier4_adjoint_fd_cpu": {"passed": True},
        },
        "summary": [],
        "probe_timings": {
            "tier4_adjoint_fd_cpu": 13.0,
        },
        "aggregate": {
            "lane_label": "jax-cuda",
            "pending_rungs": [
                "tier1b_real_stage2",
                "tier2_stage2_e2e",
                "tier3_single_stage_init",
                TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
                "tier4_adjoint_fd",
            ],
        },
    }
    payload.update(overrides)
    return payload


def test_build_aggregate_payload_merges_gpu_and_cpu_phase_artifacts():
    payload = tier5_performance_characterization._build_aggregate_payload(
        gpu_payload=_tier5_gpu_phase_payload(),
        cpu_payload=_tier5_cpu_phase_payload(),
    )

    assert payload["phase"] == "aggregate"
    assert payload["aggregate"]["complete"] is True
    assert payload["aggregate"]["pending_rungs"] == []
    assert payload["aggregate"]["phase_passed"] is True
    assert payload["aggregate"]["passed"] is True
    assert payload["summary_by_name"]["tier4_adjoint_fd"]["cpu_elapsed_s"] == pytest.approx(
        13.0
    )
    assert payload["summary_by_name"]["tier4_adjoint_fd"]["lane_elapsed_s"] == pytest.approx(
        11.0
    )
    assert payload["summary_by_name"]["tier4_adjoint_fd"]["outer_elapsed_s"] == pytest.approx(
        24.0
    )
    assert payload["summary_by_name"][TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG][
        "speedup_vs_cpu"
    ] == pytest.approx(2.0)
    assert payload["aggregate"]["performance_contract"]["headline_performance_source"][
        "metric_path"
    ] == "summary_by_name.tier2_stage2_e2e.outer_speedup_vs_cpu"
    assert payload["phase_inputs"]["gpu"]["phase"] == "gpu"
    assert payload["phase_inputs"]["cpu"]["phase"] == "cpu"


def test_build_aggregate_payload_ignores_failed_informational_outer_loop_rung():
    gpu_payload = _tier5_gpu_phase_payload()
    gpu_payload["rungs"][TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG] = {"passed": False}
    _set_summary_rung_passed(
        gpu_payload["summary"],
        TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG,
        passed=False,
    )

    payload = tier5_performance_characterization._build_aggregate_payload(
        gpu_payload=gpu_payload,
        cpu_payload=_tier5_cpu_phase_payload(),
    )

    assert payload["summary_by_name"][TIER3_SINGLE_STAGE_OUTER_LOOP_RUNG]["passed"] is False
    assert payload["aggregate"]["phase_passed"] is True
    assert payload["aggregate"]["passed"] is True


def test_build_aggregate_payload_rejects_wrong_phase_inputs():
    with pytest.raises(ValueError, match="Expected 'gpu' phase artifact"):
        tier5_performance_characterization._build_aggregate_payload(
            gpu_payload=_tier5_cpu_phase_payload(),
            cpu_payload=_tier5_cpu_phase_payload(),
        )


def test_partial_phase_payload_tracks_probe_only_wall_time():
    gpu_payload = tier5_performance_characterization._combine_phase_payload(
        provenance={"phase": "gpu"},
        lane_label="jax-cuda",
        phase="gpu",
        rungs=_tier5_gpu_phase_payload()["rungs"],
        summary=_tier5_gpu_phase_payload()["summary"],
        probe_timings=_tier5_gpu_phase_payload()["probe_timings"],
    )
    cpu_payload = tier5_performance_characterization._combine_phase_payload(
        provenance={"phase": "cpu"},
        lane_label="jax-cuda",
        phase="cpu",
        rungs=_tier5_cpu_phase_payload()["rungs"],
        summary=[],
        probe_timings=_tier5_cpu_phase_payload()["probe_timings"],
    )

    assert gpu_payload["aggregate"]["total_outer_elapsed_s"] == pytest.approx(38.0)
    assert cpu_payload["aggregate"]["total_outer_elapsed_s"] == pytest.approx(13.0)


def test_tier5_aggregate_main_skips_runtime_initialization(monkeypatch, tmp_path):
    output_json = tmp_path / "tier5.json"
    aggregate_payload = {
        "summary": [],
        "aggregate": {
            "total_outer_elapsed_s": 0.0,
            "performance_failures": [],
            "sharding_failures": [],
        },
    }
    observed_writes: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        tier5_performance_characterization,
        "parse_args",
        lambda: argparse.Namespace(
            phase="aggregate",
            gpu_input_json=str(tmp_path / "gpu.json"),
            cpu_input_json=str(tmp_path / "cpu.json"),
            output_json=str(output_json),
            platform="cuda",
            optimizer_backend="ondevice",
            benchmark_mode=False,
            plasma_surf_filename="fixture.nc",
            equilibria_dir=str(tmp_path),
            equilibrium_path=None,
            stage2_bs_path=str(tmp_path / "seed.json"),
            stage2_nphi=255,
            stage2_ntheta=64,
            single_stage_nphi=255,
            single_stage_ntheta=64,
            mpol=8,
            ntor=6,
            maxiter=20,
            vol_target=1.0,
            iota_target=0.5,
            samples=3,
            eps=1e-4,
        ),
    )
    monkeypatch.setattr(
        tier5_performance_characterization,
        "_runtime_modules",
        lambda: (_ for _ in ()).throw(AssertionError("runtime should not initialize")),
    )
    monkeypatch.setattr(
        tier5_performance_characterization,
        "load_json",
        lambda path: {"phase": "gpu"} if str(path).endswith("gpu.json") else {"phase": "cpu"},
    )
    monkeypatch.setattr(
        tier5_performance_characterization,
        "_build_aggregate_payload",
        lambda **_: aggregate_payload,
    )
    monkeypatch.setattr(
        tier5_performance_characterization,
        "write_json",
        lambda path, payload: observed_writes.append((str(path), payload)),
    )

    tier5_performance_characterization.main()

    assert observed_writes == [(str(output_json), aggregate_payload)]


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

    assert any("outer first-run wall-clock speedup" in failure for failure in failures)
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
