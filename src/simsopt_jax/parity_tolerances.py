"""Owned production contract for parity-validation tolerance lanes.

This module is the canonical owner of the parity tolerance ladder shared by
user-facing examples, the example parity harness, and benchmark orchestration.
``benchmarks.validation_ladder_contract`` re-exports these names for its
existing consumers; production and example code must import them from here so
the ``src`` and ``examples`` trees never depend on benchmark modules.
"""

from __future__ import annotations

from typing import Union

ParityToleranceValue = Union[float, bool, None]

PARITY_LADDER_TOLERANCES: dict[str, dict[str, ParityToleranceValue]] = {
    "native_workflow": {
        "same_state_value_rtol": 1e-10,
        "same_state_value_atol": 1e-12,
        "same_state_derivative_rtol": 1e-8,
        "same_state_derivative_atol": 1e-10,
        "whole_solve_value_rtol": 1e-6,
        "whole_solve_value_atol": 1e-7,
        "terminal_relative_reduction": 1e-12,
        "terminal_constraint_norm_atol": 1e-10,
        "terminal_orthonormality_atol": 1e-12,
        "terminal_stationarity_atol": 1e-7,
        "requires_same_input": True,
        "requires_native_workflow_oracle": True,
        "requires_direct_cpp_oracle": False,
    },
    "direct_kernel": {
        "rtol": 1e-10,
        "atol": 1e-12,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "vector_parity_required": True,
    },
    "relaxed_kernel": {
        "rtol": 1e-6,
        "atol": 1e-8,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "documents_reduction_order_drift": True,
    },
    "float32_smoke": {
        "rtol": 1e-5,
        "atol": 1e-6,
        "objective_rtol": 1e-4,
        "objective_atol": 1e-6,
        "gradient_rtol": 1e-3,
        "gradient_atol": 1e-5,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "runtime_dtype_float32": True,
        "smoke_only": True,
        "production_parity": False,
        "gradient_diagnostic_only": True,
    },
    "ls_wrapper_gradient": {
        "rtol": 1e-10,
        "atol": 1e-12,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "vector_parity_required": True,
    },
    "derivative_heavy": {
        "scalar_value_rtol": 1e-10,
        "scalar_value_atol": 1e-12,
        "first_derivative_rtol": 1e-8,
        "first_derivative_atol": 1e-10,
        "second_derivative_rtol": 1e-6,
        "second_derivative_atol": 1e-8,
        "requires_same_input": True,
        "requires_direct_cpp_oracle": True,
        "fd_validation_secondary": True,
    },
    "reporting_contract": {
        "scalar_value_rtol": 1e-10,
        "scalar_value_atol": 1e-12,
        "distance_rtol": 1e-10,
        "distance_atol": 1e-12,
        "requires_same_state": True,
        "host_materialization_allowed": True,
    },
    "direct_hessian_oracle": {
        "second_derivative_rtol": 1e-8,
        "second_derivative_atol": 1e-10,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "full_matrix_required": True,
    },
    "exact_well_conditioned_adjoint": {
        "adjoint_rtol": 1e-6,
        "adjoint_atol": 1e-8,
        "gradient_rtol": 1e-6,
        "gradient_atol": 1e-8,
        "residual_rel_tol": 1e-10,
        "requires_same_state": True,
        "requires_well_conditioned_jacobian": True,
        "vector_parity_required": True,
    },
    "exact_ill_conditioned_adjoint": {
        "adjoint_rtol": None,
        "adjoint_atol": None,
        "gradient_rtol": None,
        "gradient_atol": None,
        "residual_rel_tol": 1e-10,
        "requires_same_state": True,
        "requires_well_conditioned_jacobian": False,
        "operator_failure_allowed": True,
        "vector_parity_required": False,
        # Raw-vector parity is intentionally disabled because near-singular
        # Jacobians admit infinitely many adjoints that satisfy the residual
        # gate. Action-level (range-space) parity remains well-defined:
        # project both adjoints onto ``U_well`` (the columns of ``U`` whose
        # singular values exceed ``σ_max * 1e-8``) and compare there. The
        # canonical action-level threshold is ``1e-6`` (one order looser
        # than ``exact_well_conditioned_adjoint``) and is enforced by
        # ``tests/geo/test_boozersurface_jax.py::``
        # ``TestBoozerSurfaceJAXClass::``
        # ``test_exact_ill_conditioned_operator_adjoint_action_level_parity``.
        "action_level_rtol": 1e-6,
    },
    "branch_stable_resolve": {
        "core_value_rtol": 1e-6,
        "core_value_atol": 1e-7,
        "derived_value_rtol": 5e-5,
        "derived_value_atol": 1e-7,
        "requires_branch_stable_state": True,
        "branch_divergence_downgrades_to_health_only": True,
    },
    "fd_gradient": {
        "directional_fd_rtol": 1e-5,
        "directional_fd_atol": 1e-7,
        "directional_derivative_floor": 1e-12,
        "central_fd_error_rate": 0.4,
        "central_fd_min_stable_eps": 3,
        "direction_seed": 1729,
        "direction_count": 5,
        "max_direction_rejection_fraction": 0.2,
        "requires_branch_stable_state": True,
        "compares_directional_derivative": True,
    },
    "gpu_runtime": {
        "same_state_forward_rtol": 1e-10,
        "same_state_forward_atol": 1e-12,
        "same_state_gradient_rtol": 1e-8,
        "same_state_gradient_atol": 1e-10,
        "whole_solve_value_rtol": 1e-6,
        "whole_solve_value_atol": 1e-7,
        "requires_x64": True,
        "requires_fixed_seed": True,
        "requires_runtime_metadata": True,
    },
    # Exact one-to-one example mirrors retain their source-owned thresholds
    # here so manifest routes never embed or silently relax tolerances.
    "mirror_boozer_value": {"rtol": 1e-3, "atol": 1e-8},
    "mirror_boozer_parameters": {"rtol": 0.0, "atol": 2e-3},
    "mirror_single_stage_initial_objective": {
        "rtol": 1e-12,
        "atol": 1e-15,
    },
    "mirror_single_stage_initial_gradient": {
        "rtol": 2e-9,
        "atol": 2e-12,
    },
    "mirror_single_stage_final_value": {"rtol": 2e-8, "atol": 2e-12},
    "mirror_single_stage_final_parameters": {
        "rtol": 2e-8,
        "atol": 2e-10,
    },
    "mirror_single_stage_terminal_gradient": {"rtol": 0.0, "atol": 1e-7},
    "mirror_single_stage_terminal_constraint": {"rtol": 0.0, "atol": 1e-10},
    "mirror_optimization_5e2": {"rtol": 5e-2, "atol": 1e-9},
    "mirror_optimization_3e2": {"rtol": 3e-2, "atol": 1e-9},
    "mirror_optimization_2e2": {"rtol": 2e-2, "atol": 1e-9},
    "mirror_optimization_5e3": {"rtol": 5e-3, "atol": 1e-10},
    "mirror_optimization_1e1": {"rtol": 1e-1, "atol": 1e-9},
    "mirror_pmqa_final": {"rtol": 5e-4, "atol": 0.0},
    "mirror_qfm_value": {"rtol": 5e-5, "atol": 1e-7},
    "mirror_qfm_parameters": {"rtol": 2e-3, "atol": 2e-4},
    "mirror_qfm_persistence": {"rtol": 2e-2, "atol": 1e-5},
    "mirror_surface_invariant": {"rtol": 1e-8, "atol": 1e-10},
    "mirror_trace_ncsx_time": {"rtol": 0.0, "atol": 2e-2},
    "mirror_trace_ncsx_state": {"rtol": 0.0, "atol": 3e-2},
    "mirror_trace_qa_time": {"rtol": 0.0, "atol": 6e-3},
    "mirror_trace_qa_state": {"rtol": 0.0, "atol": 2e-3},
    "mirror_trace_qa_poincare": {"rtol": 0.0, "atol": 7e-3},
    "mirror_trace_particle_time": {"rtol": 0.0, "atol": 2e-6},
    "mirror_trace_particle_state": {"rtol": 0.0, "atol": 2e-3},
    "reduction_cpu_gpu": {
        "rtol": 1e-12,
        "atol": 1e-12,
        "requires_x64": True,
        "requires_cpu_gpu_devices": True,
        "uses_cancellation_stress": True,
    },
    # Scientific-equivalence ladder lanes per
    # docs/parity_scientific_equivalence_contract_2026-05-09.md §2 + §9.
    # These lanes are reporting-only at Phase 0/1: the parity arbiter does
    # not yet enforce these thresholds, and the existing
    # ``linear_solve_factors`` byte-parity probe remains authoritative.
    # ``*_condition_estimate_present`` is True because the JAX-native
    # Hager–Higham helper now populates dense compatibility solves.
    # Individual solve results may still emit ``None`` when their dense
    # compatibility operator is intentionally unavailable.
    "ls_solve_quality": {
        "ls_hessian_symmetry_rel_tol": 1e-10,
        "ls_hessian_action_max_rel_tol": 1e-8,
        "ls_newton_linear_residual_rel_tol": 1e-8,
        "ls_newton_step_abs_diff_rel_tol": 1e-8,
        "ls_condition_estimate_present": True,
        "requires_same_state": True,
        "reporting_only": True,
    },
    "exact_solve_quality": {
        "exact_jacobian_action_max_rel_tol": 1e-8,
        "exact_newton_linear_residual_rel_tol": 1e-8,
        "exact_refinement_correction_rel_tol": 1e-9,
        "exact_adjoint_solve_residual_rel_tol": 1e-8,
        "exact_condition_estimate_present": True,
        "requires_same_state": True,
        "reporting_only": True,
    },
    "pm_mwpgp_fixed_step": {
        "rtol": 1e-9,
        "atol": 1e-11,
        "state_trace_rtol": 1e-9,
        "state_trace_atol": 1e-11,
        "optimality_atol": 1e-9,
        "monotonicity_rtol": 1e-12,
        "single_step_rtol": 1e-12,
        "single_step_atol": 1e-14,
        "requires_same_state": True,
        "requires_direct_cpp_oracle": True,
        "vector_parity_required": True,
    },
    # Event-time tracing lane: adaptive RK + bracketed event localization
    # accuracy contract. Loose tolerances reflect that floating-point step
    # control and root bisection accumulate drift over a trajectory; the
    # state-vector comparison after N integration steps cannot match the
    # direct-kernel lane. Used by items 14 (RK path) and 16 (tracing
    # wrappers).
    "event_time_tracing": {
        "state_vector_rtol": 1e-6,
        "state_vector_atol": 1e-8,
        "event_time_rtol": 1e-7,
        "event_time_atol": 1e-9,
        "step_count_max_ratio": 1.25,
        "requires_branch_stable_state": True,
        "requires_event_localization": True,
        "requires_x64": True,
    },
}


def normalize_contract_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def parity_ladder_tolerances(lane: str) -> dict[str, ParityToleranceValue]:
    """Return the precision contract for a named parity-validation lane."""
    lane_key = normalize_contract_key(lane)
    if lane_key not in PARITY_LADDER_TOLERANCES:
        valid = ", ".join(sorted(PARITY_LADDER_TOLERANCES))
        raise ValueError(
            f"Unknown parity ladder lane {lane!r}. Expected one of: {valid}."
        )
    return dict(PARITY_LADDER_TOLERANCES[lane_key])
