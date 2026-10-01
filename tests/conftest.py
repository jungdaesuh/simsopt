"""Root pytest configuration: repository import path and collection markers.

It imports neither JAX nor simsopt and leaves ``XLA_FLAGS`` and the JAX config
alone, so upstream's native tests run as they do without it. JAX-dependent test
modules take the JAX test runtime from ``jax_test_support`` instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

# Rewrite asserts in the helpers that JAX test modules import from there.
pytest.register_assert_rewrite("jax_test_support")

_STRICT_GPU_EXCLUDED_MARKERS = {
    "adapter_boundary": "adapter-boundary test is not part of strict jax_gpu_parity",
    "native_cpu_reference": "native CPU/reference test is not part of strict jax_gpu_parity",
    "build_compat_native": "native build-compat test is not part of strict jax_gpu_parity",
}

_ADAPTER_BOUNDARY_TEST_FILES = frozenset(
    {
        "field/test_selffieldforces.py",
        "geo/test_curvexyzfouriersymmetries_spec_jax.py",
        "geo/test_qfmsurface_jax.py",
        "geo/test_surface_fourier_jax.py",
        "geo/test_surface_garabedian_jax.py",
        "geo/test_surface_henneberg_jax.py",
        "geo/test_surface_objectives_jax.py",
        "geo/test_surface_rzfourier_jax.py",
        "geo/test_surface_rzfourier_jax_non_stellsym_production_scale.py",
        "geo/test_surface_rzfourier_transfer_guard_jax.py",
        "geo/test_surface_xyz_tensor_clamped_jax.py",
        "jax/native_unit_parity/test_curve_subclasses_parity.py",
        "jax/native_unit_parity/test_force_parity.py",
        "jax/native_unit_parity/test_strain_parity.py",
        "objectives/test_fluxobjective_jax_parity.py",
    }
)

_NATIVE_CPU_REFERENCE_TEST_FILES: frozenset[str] = frozenset()

_NATIVE_CPU_REFERENCE_NODE_PREFIXES = frozenset(
    {
        "jax/solve/test_value_grad_contract.py::test_simsopt_bfgs_uses_explicit_value_grad_under_strict_transfer_guard",
        "jax/objectives/test_force_stage_two.py::test_force_stage_two_diagnostics_match_native_force_objectives",
        "geo/test_boozer_derivatives_jax.py::TestBoozerDirectCppOracles::",
        "geo/test_boozer_derivatives_jax.py::TestBoozerResidualJacobianComposed::test_jacobian_matches_cpp_dresidual_dc_oracle",
        "geo/test_boozer_derivatives_jax.py::TestBoozerHessianComposed::test_hessian_matches_cpp_boozer_residual_ds2_oracle",
        "geo/test_boozer_residual_jax.py::TestBoozerResidualScalar::test_scalar_matches_cpp_oracle",
        "geo/test_boozer_residual_jax.py::TestBoozerResidualParityStress::test_vector_reduction_near_tolerance_floor_matches_cpp_scalar",
        "geo/test_boozer_residual_jax.py::TestBoozerResidualParityStress::test_scalar_residual_norm_near_tolerance_floor_matches_cpp_oracle",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_scipy_bfgs_strips_limited_memory_and_callback_options",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_reference_scipy_adapters_materialize_host_contract",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_reference_scipy_adapter_rejects_non_scalar_objective",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_scipy_minimize_does_not_cache_unmarked_objective",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_quadratic",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_refines_nontrivial_gmres_residual",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_dense_steps_use_materialized_solve",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_backtracks_finite_norm_increasing_operator_steps",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_backtracks_nonfinite_value_candidate",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_rejects_nonfinite_gradient_norm_candidate",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_nonfinite_operator_step_fails_without_dense_fallback",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_exact_linear_system",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_exact_materializes_dense_jacobian_once_at_final_iterate",
        "geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_exact_skips_dense_jacobian_when_ceiling_is_exceeded",
        "geo/test_boozersurface_jax.py::TestNewtonPolishBoozer::",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_backend_mutation_does_not_rewrite_dense_linearization_default",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_instantiation_defaults_optimizer_backend_from_runtime_contract",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_reference_ls_reuses_cached_scipy_value_and_grad_transform",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_lbfgs_public_api_uses_options_default_when_limited_memory_omitted",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_host_scipy_quasi_newton_uses_cpu_ordered_value_grad_without_trace",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_ls_api_accepts_weight_inv_modB_override",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_manual_ls_api_supports_baseline_demo_sequence",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_restores_cpu_api",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_stays_native_without_root",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_uses_full_jacobian_solve",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_newton_api_routes_without_legacy_vectorize_kwarg",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_private_options_rejected_with_scipy_backend",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_routes_backend_contract_to_expected_method",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_scipy_bfgs_pre_newton_contract_uses_cpu_call_shape",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_reference_lm_forwards_least_squares_options",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_jax_least_squares_solves_simple_structured_problem",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_reference_minimize_supports_structured_explicit_value_and_grad",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_jax_least_squares_pytree_hot_path_skips_flattening_adapter",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_same_input_same_result_without_res_identity",
        "geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_emits_newton_progress_updates",
        "geo/test_boozersurface_jax.py::TestMixedQuadratureBoozer::test_run_code_ls_converges",
        "geo/test_boozersurface_jax.py::TestMixedQuadratureBoozer::test_penalty_matches_uniform",
    }
)

_BUILD_COMPAT_NATIVE_TEST_FILES = frozenset(
    {
        "geo/test_boozersurface.py",
    }
)


def _item_nodeid_matches(item, prefixes: frozenset[str]) -> bool:
    nodeid = item.nodeid
    test_root_relative_nodeid = nodeid.removeprefix("tests/")
    return any(
        nodeid.startswith(prefix) or test_root_relative_nodeid.startswith(prefix)
        for prefix in prefixes
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark heavy JAX suites so they can be sharded consistently."""
    tests_root = Path(__file__).resolve().parent
    strict_gpu_mode = os.environ.get("SIMSOPT_BACKEND_MODE") == "jax_gpu_parity"
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            relpath = path.relative_to(tests_root)
        except ValueError:
            continue

        relpath_str = relpath.as_posix()
        if relpath.parts and relpath.parts[0] == "jax":
            item.add_marker(pytest.mark.jax_gpu_pure)
        if relpath.parts and relpath.parts[0] == "integration":
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.jax_gpu_integration)
            item.add_marker(pytest.mark.slow)
        if relpath_str in _ADAPTER_BOUNDARY_TEST_FILES:
            item.add_marker(pytest.mark.adapter_boundary)
        if relpath_str in _NATIVE_CPU_REFERENCE_TEST_FILES or _item_nodeid_matches(
            item, _NATIVE_CPU_REFERENCE_NODE_PREFIXES
        ):
            item.add_marker(pytest.mark.native_cpu_reference)
        if relpath_str in _BUILD_COMPAT_NATIVE_TEST_FILES:
            item.add_marker(pytest.mark.build_compat_native)
        if relpath_str in {
            "integration/test_single_stage_jax.py",
        }:
            item.add_marker(pytest.mark.single_stage)
            item.add_marker(pytest.mark.slow)
        if relpath_str in {
            "integration/test_stage2_target_lane_purity.py",
        }:
            item.add_marker(pytest.mark.stage2)
            item.add_marker(pytest.mark.slow)
        if relpath_str in {
            "geo/test_boozersurface_jax.py",
            "geo/test_boozersurface_jax_private.py",
        }:
            item.add_marker(pytest.mark.boozer)
            item.add_marker(pytest.mark.slow)
        if strict_gpu_mode:
            for marker_name, reason in _STRICT_GPU_EXCLUDED_MARKERS.items():
                if any(item.iter_markers(marker_name)):
                    item.add_marker(pytest.mark.skip(reason=reason))
                    break
