"""Preserve the field/leaf order of the 53 former compute NamedTuples."""

from dataclasses import FrozenInstanceError, fields

import jax
import numpy as np
import pytest
import simsopt_jax.core.interpolated_field as interpolated_field_records
import simsopt_jax.core.qfm_solver as qfm_records
import simsopt_jax.core.tracing as tracing_records
import simsopt_jax.geo.optimizers.linear_solve as linear_solve_records
import simsopt_jax.geo.optimizers.native_ls_newton as native_ls_newton_records
import simsopt_jax.geo.optimizers.optimizer as optimizer_records
import simsopt_jax.geo.optimizers.private._bfgs as bfgs_records
import simsopt_jax.geo.optimizers.private._lbfgsb_scipy as lbfgsb_records
import simsopt_jax.geo.optimizers.private._types as solver_records
import simsopt_jax_adapters.geo.surface_objectives_traceable as traceable_records


COMPUTE_RECORDS = (
    (interpolated_field_records.InterpolatedFieldCylCache, ("B_cyl", "GradAbsB_cyl")),
    (
        qfm_records.QfmAugmentedLagrangianInfo,
        (
            "success",
            "status",
            "fun",
            "gradient",
            "nit",
            "nfev",
            "njev",
            "label_value",
            "label_residual",
            "qfm_value",
            "augmented_value",
            "multiplier",
            "penalty_weight",
        ),
    ),
    (
        qfm_records.QfmExactKktInfo,
        (
            "feasibility_abs",
            "stationarity_inf",
            "lagrange_multiplier",
            "qfm_gradient_inf",
            "label_gradient_norm",
        ),
    ),
    (
        qfm_records.QfmPenaltySolveInfo,
        (
            "success",
            "status",
            "fun",
            "gradient",
            "nit",
            "nfev",
            "njev",
            "label_value",
            "label_residual",
            "qfm_value",
            "penalty_value",
        ),
    ),
    (
        qfm_records._BFGSResult,
        ("x", "success", "status", "fun", "jac", "nit", "nfev", "njev"),
    ),
    (
        qfm_records._BFGSState,
        ("x", "fun", "grad", "hess_inv", "nit", "nfev", "njev", "status"),
    ),
    (qfm_records._QfmMetrics, ("qfm_value", "label_value", "label_residual")),
    (
        tracing_records._Dopri5AdaptiveStep,
        (
            "h_clamped",
            "y_new",
            "accepted",
            "nonfinite_state",
            "h_next",
            "t_next",
            "y_next",
            "k_next",
            "stages",
        ),
    ),
    (
        tracing_records._Dopri5Stages,
        ("y_new", "y_err", "k1", "k3", "k4", "k5", "k6", "k7", "field_cache"),
    ),
    (
        linear_solve_records._DenseJacobianMaterialization,
        ("residual", "jacobian", "telemetry"),
    ),
    (
        linear_solve_records._DenseJacobianMaterializationTelemetry,
        (
            "assembler_code",
            "residual_evaluation_count",
            "primal_traversal_count",
            "tangent_batch_count",
            "tangent_direction_count",
            "batch_width",
            "tail_width",
        ),
    ),
    (
        linear_solve_records._ExactFinalLinearizationValidation,
        (
            "success",
            "identity_valid",
            "orientation_valid",
            "factorization_residual",
            "factor_metadata_valid",
            "factorization_valid",
            "factorization_reconstruction_count",
        ),
    ),
    (
        linear_solve_records._LinearSolveStatus,
        (
            "success",
            "residual",
            "residual_relative",
            "iterations",
            "residual_scale",
            "requested_tolerance",
            "effective_tolerance",
            "dense_materialization_count",
            "lu_factorization_count",
            "lu_solve_count",
            "refinement_correction_count",
        ),
    ),
    (
        linear_solve_records._RetainedJacobianTransposeSolve,
        (
            "solution",
            "correction",
            "residual",
            "condition_estimate",
            "payload_validation",
            "status",
        ),
    ),
    (
        native_ls_newton_records._NativeLsNewtonState,
        ("x", "fun", "grad", "hessian", "grad_norm", "nit"),
    ),
    (
        optimizer_records._DenseExactNewtonDirection,
        (
            "residual",
            "jacobian",
            "lu",
            "pivots",
            "initial_solve",
            "refinement_rhs",
            "direction",
            "correction",
            "linear_residual",
            "condition_estimate",
            "status",
        ),
    ),
    (
        optimizer_records._NativeDenseExactNewtonC2Result,
        (
            "x",
            "residual",
            "returned_jacobian",
            "iteration_count",
            "applied_update_count",
            "success",
            "numerical_failure",
            "stop_reason_code",
            "rollback_branch_taken",
            "native_persist_predicate",
            "persist_solved_state",
            "initial_norm",
            "assessed_norm",
            "returned_norm",
            "applied_state_trace",
            "applied_state_trace_active",
            "assessed_norm_trace",
            "assessed_norm_trace_active",
            "exact_newton_linear_residual_rel",
            "exact_refinement_correction_rel",
            "linear_solve_attempt_count",
            "dense_materialization_count",
            "lu_factorization_count",
            "lu_solve_count",
            "refinement_correction_count",
            "rollback_recompute_count",
        ),
    ),
    (
        bfgs_records._BFGSObservation,
        ("terminal", "accepted", "iteration", "nfev", "ngev"),
    ),
    (
        bfgs_records._BFGSObserverObservation,
        ("terminal", "accepted", "iteration", "nfev", "ngev", "x_k", "f_k", "g_k"),
    ),
    (bfgs_records._BFGSTransition, ("state", "accepted")),
    (lbfgsb_records.LbfgsbActiveResult, ("x", "iwhere", "prjctd", "cnstnd", "boxed")),
    (lbfgsb_records.LbfgsbBmvResult, ("p", "info")),
    (
        lbfgsb_records.LbfgsbCauchyResult,
        ("iorder", "iwhere", "t", "d", "xcp", "p", "c", "wbp", "v", "nseg", "info"),
    ),
    (lbfgsb_records.LbfgsbCmprlbResult, ("r", "wa", "info")),
    (lbfgsb_records.LbfgsbDcsrchResult, ("stp", "task", "task_msg", "isave", "dsave")),
    (
        lbfgsb_records.LbfgsbDcstepResult,
        ("stx", "fx", "dx", "sty", "fy", "dy", "stp", "brackt"),
    ),
    (lbfgsb_records.LbfgsbFormkResult, ("wn", "wn1", "info")),
    (lbfgsb_records.LbfgsbFormtResult, ("wt", "info")),
    (
        lbfgsb_records.LbfgsbFreevResult,
        ("nfree", "idx", "nenter", "ileave", "idx2", "wrk"),
    ),
    (lbfgsb_records.LbfgsbHpsolbResult, ("t", "iorder")),
    (lbfgsb_records.LbfgsbInverseHessianHistory, ("s", "y", "n_corrs")),
    (
        lbfgsb_records.LbfgsbLnsrlbResult,
        (
            "x",
            "fold",
            "gd",
            "gdold",
            "g",
            "r",
            "t",
            "stp",
            "dnorm",
            "dtd",
            "xstep",
            "stpmx",
            "ifun",
            "iback",
            "nfgv",
            "info",
            "task",
            "task_msg",
            "isave",
            "dsave",
            "temp_task",
            "temp_task_msg",
        ),
    ),
    (
        lbfgsb_records.LbfgsbMacroStepResult,
        ("state", "accepted_new_x", "terminal", "entry_kind"),
    ),
    (
        lbfgsb_records.LbfgsbMatupdResult,
        ("ws", "wy", "sy", "ss", "itail", "col", "head", "theta"),
    ),
    (
        lbfgsb_records.LbfgsbState,
        (
            "m",
            "x",
            "l",
            "u",
            "nbd",
            "f",
            "g",
            "factr",
            "pgtol",
            "maxls",
            "workspace",
            "n_iterations",
            "nfev",
            "njev",
            "evaluated_nonfinite_count",
            "all_accepted_states_finite",
        ),
    ),
    (lbfgsb_records.LbfgsbSubsmResult, ("x", "d", "xp", "iword", "wv", "info")),
    (
        lbfgsb_records.LbfgsbWorkspace,
        ("wa", "iwa", "task", "ln_task", "lsave", "isave", "dsave"),
    ),
    (
        solver_records._BFGSResults,
        (
            "converged",
            "failed",
            "k",
            "nfev",
            "ngev",
            "nhev",
            "x_k",
            "f_k",
            "g_k",
            "H_k",
            "old_old_fval",
            "status",
            "line_search_status",
        ),
    ),
    (
        solver_records._LBFGSInvalidStepRecord,
        (
            "recorded",
            "iteration",
            "step_scale",
            "line_search_failed",
            "nonfinite_step",
            "ls_status",
            "failure_reason",
            "curvature_margin_measured",
            "curvature_margin",
        ),
    ),
    (
        solver_records._LBFGSResults,
        (
            "converged",
            "failed",
            "k",
            "nfev",
            "ngev",
            "x_k",
            "f_k",
            "g_k",
            "s_history",
            "y_history",
            "rho_history",
            "gamma",
            "status",
            "ls_status",
            "evaluated_nonfinite_count",
            "all_accepted_states_finite",
            "invalid_step_record",
            "optimizer_state_trace",
            "hess_inv_s",
            "hess_inv_y",
            "hess_inv_n_corrs",
            "task",
        ),
    ),
    (
        solver_records._LineSearchResults,
        ("failed", "nit", "nfev", "ngev", "k", "a_k", "f_k", "g_k", "status"),
    ),
    (
        solver_records._LineSearchState,
        (
            "done",
            "failed",
            "i",
            "a_i2",
            "phi_i2",
            "dphi_i2",
            "g_i2",
            "a_i1",
            "phi_i1",
            "dphi_i1",
            "g_i1",
            "best_a",
            "best_phi",
            "best_dphi",
            "best_g",
            "nfev",
            "ngev",
            "a_star",
            "phi_star",
            "dphi_star",
            "g_star",
        ),
    ),
    (
        solver_records._ZoomState,
        (
            "done",
            "failed",
            "j",
            "a_lo",
            "phi_lo",
            "dphi_lo",
            "g_lo",
            "a_hi",
            "phi_hi",
            "dphi_hi",
            "g_hi",
            "has_rec",
            "a_rec",
            "phi_rec",
            "dphi_rec",
            "g_rec",
            "a_star",
            "phi_star",
            "dphi_star",
            "g_star",
            "best_a",
            "best_phi",
            "best_dphi",
            "best_g",
            "nfev",
            "ngev",
        ),
    ),
    (
        traceable_records.TraceableObjectiveExecutionCounts,
        (
            "newton_iteration_count",
            "dense_materialization_count",
            "lu_factorization_count",
            "lu_solve_count",
            "refinement_correction_count",
            "adjoint_execution_count",
        ),
    ),
    (
        traceable_records.TraceableObjectiveInnerState,
        ("coil_dofs", "solved_x", "objective_value", "eligible"),
    ),
    (
        traceable_records._TraceableAdjointExecutionEvidence,
        ("adjoint_output", "residual", "residual_relative"),
    ),
    (
        traceable_records._TraceableExactPayloadFusedConsumerResult,
        (
            "success",
            "value",
            "gradient",
            "retained_solve",
            "dynamic_inputs_match",
            "live_residual_matches_payload",
            "returned_state_residual_success",
            "finite_outputs",
            "live_residual",
            "producer_residual_inf_norm",
        ),
    ),
    (
        traceable_records._TraceableExactPayloadFusedEvidence,
        (
            "producer_residual",
            "live_residual",
            "producer_residual_inf_norm",
            "adjoint",
            "consumer_reuse_counts",
            "full_graph_counts",
            "factorization_reconstruction_count",
            "linearization_primal_traversal_count",
        ),
    ),
    (
        traceable_records._TraceableExactPayloadFusedResult,
        ("value", "gradient", "status", "evidence"),
    ),
    (
        traceable_records._TraceableExactPayloadFusedStatus,
        (
            "success",
            "returned_state_solve_success",
            "producer_solve_success",
            "returned_state_residual_success",
            "dynamic_inputs_match",
            "live_residual_matches_payload",
            "payload_validation_success",
            "factorization_valid",
            "adjoint_solve_success",
            "finite_outputs",
        ),
    ),
    (traceable_records._TraceableExactReturnedState, ("solved_state", "solve_success")),
    (
        traceable_records._TraceablePLURefinement,
        (
            "solution",
            "residual",
            "status",
            "residual_relative_trace",
            "contraction_ratio_trace",
            "residual_relative_trace_length",
            "contraction_finite",
            "contraction_monotone",
            "stagnated",
        ),
    ),
    (
        traceable_records._TraceableSuppliedFactorSolveStatus,
        (
            "success",
            "residual",
            "residual_relative",
            "iterations",
            "residual_scale",
            "requested_tolerance",
            "effective_tolerance",
            "supplied_factor_residual_relative_trace",
            "supplied_factor_residual_relative_trace_length",
            "supplied_factor_contraction_ratio_trace",
            "supplied_factor_contraction_finite",
            "supplied_factor_contraction_monotone",
            "supplied_factor_stagnated",
            "fp64_rebuild_count",
            "fp64_rebuild_residual_relative_trace",
            "fp64_rebuild_residual_relative_trace_length",
        ),
    ),
)


@pytest.mark.parametrize(
    "record_type, original_fields",
    COMPUTE_RECORDS,
    ids=[record_type.__name__ for record_type, _ in COMPUTE_RECORDS],
)
def test_compute_record_preserves_namedtuple_field_and_leaf_order(
    record_type, original_fields
):
    # Distinct sentinels expose dropped, static, or permuted fields independently
    # of the registration declaration. Domain inputs are covered by kernel tests.
    values = tuple(float(index + 1) for index in range(len(original_fields)))
    record = record_type(*values)
    assert tuple(field.name for field in fields(record)) == original_fields
    assert jax.tree_util.tree_leaves(record) == list(values)
    with pytest.raises(FrozenInstanceError):
        setattr(record, original_fields[0], -1.0)

    leaves, treedef = jax.tree_util.tree_flatten(record)
    restored = jax.tree_util.tree_unflatten(
        treedef, [value + 100.0 for value in leaves]
    )
    assert isinstance(restored, record_type)
    assert tuple(getattr(restored, name) for name in original_fields) == tuple(
        value + 100.0 for value in values
    )

    compiled = jax.jit(lambda item: item)(record)
    assert isinstance(compiled, record_type)
    for name, value in zip(original_fields, values, strict=True):
        np.testing.assert_array_equal(getattr(compiled, name), value)

    gradient = jax.grad(
        lambda item: sum(value**2 for value in jax.tree_util.tree_leaves(item))
    )(record)
    assert isinstance(gradient, record_type)
    for name, value in zip(original_fields, values, strict=True):
        np.testing.assert_array_equal(getattr(gradient, name), 2.0 * value)
