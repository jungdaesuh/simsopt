# ALM Test-Coverage Adequacy Audit

Repo: `/Users/suhjungdae/code/columbia/simsopt-surrogate`
Subject: `examples/single_stage_optimization/alm_utils.py` (4637 LOC), entry `minimize_alm`
Audit date: 2026-05-08
Method: each test in the listed files inspected and classified as **shape**, **numerical pin**, **property**, or **end-to-end**. Properties demanded by the user (Q1-Q10) checked against full test set.

---

## Executive summary

The ALM driver has 118 tests in `tests/geo/test_alm_utils.py` (4447 LOC) plus thin auxiliary suites. The vast majority are **shape** + **structural** tests (closure-free, schema, history-key inventory, frozen dataclass, action-string ordering). Most "behavioral" tests use `patch.object(module, "minimize", side_effect=fake_minimize)` to fake-fix the inner solver and assert downstream history actions / classification labels. **Only one true end-to-end test** drives the ALM into a real success state (`test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint` and the scale-1 mirror).

### Tests cover
- Action-string sequencing of the outer loop on a faked inner solver: `dual_update | penalty_increase | infeasible_stall_penalty_increase | subproblem_continue | constraints_inactive_converged | constraints_inactive_stall | signal_mismatch_penalty_increase | converged`.
- Multiplier projection to non-negative, multiplier cap, dual-update tolerance shrink, tolerance floor.
- Penalty-cap saturation and the `_next_penalty` overflow guard.
- Stall reasoning (`_classify_infeasible_inner_stall`) for the four SciPy message arms.
- History entry schema (52 shared keys), history truncation count, history immutability against callback mutation.
- Restoration of best-feasible incumbent under `final_iterate_worse_than_best_feasible`.
- Constraint-routing diagnostics (block grouping, signal mismatch, surrogate-vs-hard sign mismatch).
- Surrogate-KKT stationarity helper computation and gating.
- Constraint-normalization arithmetic (scaling values / grads / tolerances) and shape rejection.
- Skipped-inner shortcut: SciPy `minimize` is **not** invoked when the iterate already meets tolerances.
- One `test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint` E2E success on a scalar problem.

### Tests do NOT cover (the load-bearing physics/math gaps)
- **No KKT verification at termination.** No test asserts on the final iterate that `||c(x)||_inf <= eps_feas` AND `||grad L_A||_inf <= eps_stat` together. The single E2E success path checks only `result.x[0] <= 1.0+1e-6` and that `final_kkt_stationarity_norm == 0.0` — but the scalar problem has the trivial KKT point at the constraint boundary with `lambda=1`, so a sign flip / off-by-one in either the constraint sign or the augmented gradient could still pass.
- **No multiplier-sign property test.** No test on a multi-constraint problem asserts `lambda_i >= 0 forall i` after a non-trivial dual update at the final iterate. (`_project_nonnegative_multipliers_with_diagnostics` is unit-tested at the helper level only.)
- **No "penalty grows under stalled feasibility" property test.** `test_minimize_alm_increases_penalty_only_when_feasibility_is_bad` runs at most 2 outer iterations and checks `result.penalty == 100.0` (i.e. exactly two scalings). No test confirms that across N stalled outers `mu` grows monotonically by `penalty_scale` until the cap.
- **No stall-classification end-to-end coverage of all three classes simultaneously.** `382d7a082` ("pin ALM stall classification and skipped-inner shortcut") split the helper-level classification into 4 unit tests on `_classify_infeasible_inner_stall` and one "skipped-inner shortcut" full-driver test. The unit tests are textual SciPy-message dispatch; they never drive the full ALM into "progress" -> "infeasible_stall" -> "skipped_inner" within one run. There is no test that asserts the `_ALMOuterDecision` enum value on the outer loop for each stall class.
- **No normalization-invariance property test.** `test_scale_one_minimize_alm_matches_scalar_history_and_convergence` only confirms the *trivial* identity `scale=1.0`; it does not test scale != 1.0 invariance. `test_scaled_inequality_fixture_preserves_raw_feasibility_and_dual_conversion` and `test_alm_fixture_benchmarking::test_fixture_raw_and_normalized_constraints_share_feasible_set` check arithmetic at a single fixture point but never compare two end-to-end ALM solves at different scales.
- **No sign-flipped-constraint regression test.** Nothing in the suite would catch flipping `signed_constraint_value = c(x) - upper` to `upper - c(x)`. Both `augmented_inequality_objective` tests and the E2E test only run with the correct sign convention.
- **No NaN/Inf in c(x).** `test_minimize_alm_sanitizes_nonfinite_candidate_evaluations` only injects NaN into `total` and `grad`. Constraint-value NaN is never injected — and the sanitizer's behavior on NaN constraint_values is therefore undefined by tests.
- **No best-feasible monotonicity property test.** `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective` only checks the *final* restoration; no test asserts that at every outer iteration `best_feasible.objective <= prev_best_feasible.objective`.
- **No block-penalty-update test.** `block_penalties` is wired through history/diagnostics but the tests only assert it is `None` or contains a single block. No test confirms per-block penalty growth.
- **No success-mode E2E with > 1 active constraint** (the 12-constraint test is a smoke that mocks the inner solver).
- **No equality-constraint test.** ALM driver supports inequality only; if the contract grows, no protective test exists.
- **No second-order test.** The directional Taylor test is a property-test for evaluator gradient consistency; no test runs it against the full augmented problem to catch a missing penalty-gradient term.

### Net judgment
Test surface area looks large, but the **load-bearing math contracts** (KKT, dual sign, normalization invariance, monotonic best-feasible, sign-of-constraint correctness) are pinned only at the unit-helper level on toy 1-D inputs. The suite is **structural-heavy**: a refactor that swaps two signs in a critical augmented-gradient term, or that produces `lambda_i < 0` for a scenario the projection helper doesn't cover, would not necessarily fail any existing test.

---

## Test-by-test classification

Notation: **S**=shape, **N**=numerical pin (specific value/tol), **P**=property (invariant), **E**=end-to-end ALM run that asserts solver-correctness outcome. "Faked-inner" means SciPy `minimize` is replaced by `fake_minimize` returning hand-crafted `SimpleNamespace`, so the inner gradient descent never runs — all asserted behavior is outer-loop bookkeeping.

### `tests/geo/test_alm_utils.py`

| Test | Cat | Asserted | NOT asserted |
|---|---|---|---|
| `test_bound_residuals_clamp_satisfied_constraints` | N | `upper_bound_residual`/`lower_bound_residual` clamping at four points | sign convention; only positive-side residuals are checked |
| `test_augmented_inequality_objective_uses_projected_multiplier_shift` | N | total/grad/positive-shift/feasibility values for a fixed 2-constraint input | gradient via finite-difference; sign-flip would still pass at the chosen non-trivial values only by coincidence |
| `test_extract_constraint_state_requires_dual_update_values` | S | KeyError raised when contract is incomplete | semantic contract |
| `test_meaningful_progress_*` (3 tests) | N+P | acceptance/rejection of progress at numeric edges | does not invariant-test "if both feasibility and stationarity strictly improve, progress is True" |
| `test_augmented_inequality_objective_accepts_vector_penalty` | N | vector-penalty arithmetic equality with scalar penalty | full-driver vector-penalty path |
| `test_augmented_inequality_objective_one_block_vector_matches_scalar` | N | one-block vector-penalty == scalar | per-block penalty escalation |
| `test_project_nonnegative_multipliers_uses_vector_penalty` | N | vector-penalty projection | sign violation under non-finite penalty |
| `test_minimize_alm_records_constraint_blocks_as_diagnostics_only` | S/E | result.block_penalties is None and history fields are populated | per-block penalty *update* (because block penalties are diagnostic only, this is not actually tested anywhere) |
| `test_minimize_alm_rejects_mismatched_constraint_block_metadata` | S | ValueError when constraint_blocks length mismatches | semantic |
| `test_build_constraint_metadata_tuples_stringifies_each_input_once` | S/P | call-count invariant | semantic correctness of stringification |
| `test_attach_constraint_metadata_*_does_not_scale_with_inner_iterations` (2) | P | helper invocation is O(1) per outer iteration, not O(inner) | semantic correctness |
| `test_normalize_alm_constraints_scales_*` | N | scaling arithmetic at a single fixture | scale-invariance of full ALM solve |
| `test_normalize_alm_constraints_accepts_empty_constraint_set` | S | empty arrays | semantic |
| `test_normalize_alm_constraints_rejects_nonfinite_or_nonpositive_scales` | S | ValueError | semantic |
| `test_normalize_alm_constraints_rejects_shape_mismatch` | S | ValueError x4 shape mismatch types | semantic |
| `test_constraint_history_diagnostics_groups_values_by_block` | N | block grouping arithmetic at fixed fixture | per-block penalty update path |
| `test_objective_to_augmented_term_ratio_requires_explicit_base_objective` | S | KeyError | semantic |
| `test_alm_summary_uses_summary_diagnostics_without_full_history_payload` | S/P | summary mutation safety | numeric correctness of summary |
| `test_incumbent_objective_value_prefers_promoted_physics_total` | N | helper preference at fixed input | property "ranking is monotone in physics_total" |
| `test_multiplier_interpretation_marks_mixed_value_sources_as_search_multipliers` | S | string label | semantic |
| `test_surrogate_kkt_stationarity_uses_surrogate_feasibility_gate` | N | helper output at fixed input | full-driver KKT residual at convergence |
| `test_lbfgsb_projected_gradient_max_norm_uses_projected_infinity_norm` | N | projection norm at fixed input | invariant under bound activation |
| `test_augmented_inequality_objective_exposes_solver_constraint_metadata` | S | constraint metadata passthrough | semantic |
| `test_project_nonnegative_multipliers_enforces_nonnegativity_and_cap` | N | projection at single fixture (lambda=1.0) | property `forall lambda, projected >= 0` |
| `test_scaled_inequality_fixture_preserves_raw_feasibility_and_dual_conversion` | N | scale arithmetic | end-to-end scale invariance |
| `test_scale_one_fixture_matches_existing_scalar_alm_behavior` | N | `scale=1.0` identity | non-trivial scale |
| `test_minimize_alm_public_entrypoint_is_small_and_closure_free` | S | LOC <= 500, no nested defs | semantic |
| `test_extracted_alm_state_carriers_exist` | S | hasattr | semantic |
| `test_minimize_alm_impl_orchestrator_is_small_and_closure_free` | S | LOC <= 500 | semantic |
| `test_minimize_alm_impl_phase4_helpers_exist_and_closure_free` | S | hasattr + closure-free | semantic |
| `test_phase4_result_carriers_are_frozen_dataclasses_without_mutable_defaults` | S | frozen + no mutable defaults | algorithmic correctness |
| `test_dual_update_projects_multipliers_and_tightens_tolerances` | N+P | tolerance shrink, multipliers >= 0 (single 2-vector) | non-degenerate KKT verification |
| `test_dual_update_floors_tolerance_at_settings_value` | N | tolerance floor | shrinking property invariant |
| `test_handle_alm_penalty_cap_termination_forwards_all_keyword_arguments` | S | call-arg pass-through | semantic |
| `test_normalize_rejects_*` (6 tests) | S | settings validation | semantic |
| `test_build_alm_history_entry_returns_all_52_shared_keys` | S | 52 keys | numeric |
| `test_build_alm_history_entry_skipped_inner_payload_matches_post_inner_shape` | S | shape parity | numeric |
| `test_select_inner_solve_profile_uses_explicit_boxed_feasible_continuation_profile` | N | profile dispatch at fixed input | profile correctness |
| `test_zero_trust_radius_disables_box_bounds_and_boxed_profile` | S/P | bounds=None and profile name | semantic |
| `test_minimize_alm_rejects_asymmetric_incumbent_hooks` | S | TypeError | semantic |
| `test_minimize_alm_reports_constraint_blocks_from_evaluation_metadata` | S/E | metadata routing | algorithmic |
| `test_minimize_alm_keeps_positional_snapshot_hook_compatibility` | S | call-signature compat | algorithmic |
| `test_build_inner_options_caps_*` (2) | N | inner option caps | invariant under continuation |
| `test_classify_infeasible_inner_stall_scales_move_tolerance_with_iterate_norm` | N+P | move-tolerance scales with `||x||` | semantic for very large `||x||` |
| `test_conditioning_metrics_*` (3) | N | metric arithmetic | semantic |
| `test_minimize_alm_sanitizes_nonfinite_candidate_evaluations` | E (faked-inner) | NaN total/grad sanitization, history.action="penalty_increase" | NaN in `constraint_values`, NaN in `dual_update_values`, Inf, mixed NaN/finite gradients |
| `test_candidate_is_acceptable_allows_near_equal_feasible_trial` | N | acceptance at fixed input | invariant |
| `test_directional_taylor_test_passes_for_consistent_gradient` | E | Taylor pass on quartic | not run on full augmented problem (only the evaluator) |
| `test_directional_taylor_test_flags_inconsistent_gradient` | E | Taylor fails on inconsistent gradient | same |
| **`test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint`** | E | `x <= 1+1e-6`, `c(x) <= 1e-6`, `final_kkt_stationarity_norm == 0` | does NOT assert success=True (asserts `assertFalse(result.success)`!), does NOT assert `||grad L_A||_inf <= eps_stat`, does NOT assert `lambda >= 0` |
| `test_scale_one_minimize_alm_matches_scalar_history_and_convergence` | E | `scale=1.0` identity passes converged | scale != 1.0 invariance |
| `test_minimize_alm_requires_signed_constraint_activity_for_boundary_kkt_success` | E | `x ~= 1.0`, `multipliers == [0.0]` | success at boundary with lambda > 0 |
| `test_minimize_alm_keeps_current_iterate_after_all_trials_are_rejected` | E (faked-inner) | x not corrupted by rejected trial | algorithmic |
| `test_minimize_alm_retries_with_smaller_trust_radius_after_abnormal_step` | E (faked-inner) | trust-radius shrink | algorithmic |
| `test_minimize_alm_reports_feasibility_values_separately_from_solver_residuals` | S | reporting separation | semantic |
| `test_stationarity_metrics_uses_raw_norm_when_stage2_signals_disagree` | N | metric routing at fixed input | invariant |
| `test_constraint_routing_state_flags_boundary_mismatch_when_surrogate_shift_is_live` | N | routing flag at fixed input | invariant |
| `test_minimize_alm_returns_constraints_inactive_converged_for_stage2_zero_shift` | E (faked-inner) | termination_reason and history actions | KKT verification (no `lambda` checked, no `||grad L_A||`) |
| `test_minimize_alm_returns_constraints_inactive_stall_after_repeat_without_progress` | E (faked-inner) | termination_reason="constraints_inactive_stall" after 3 stalls | KKT verification |
| `test_minimize_alm_escalates_penalty_after_repeated_stage2_signal_mismatch` | E (faked-inner) | history actions | semantic |
| `test_minimize_alm_increases_penalty_only_when_feasibility_is_bad` | E (faked-inner) | `result.penalty == 100.0` after 2 outers | property "mu grows monotonically across N infeasible outers until cap" (only 2 outers) |
| `test_minimize_alm_reports_history_truncation_count` | S/E | truncation count | algorithmic |
| `test_history_diagnostic_materialization_is_pure_and_snapshots_are_owned` | P | diagnostic snapshot is owned by caller | semantic |
| `test_minimize_alm_short_circuits_zero_step_infeasible_stall` | E (faked-inner) | `infeasible_stall_penalty_increase` after 1 fake call | KKT termination |
| `test_sanitize_nonfinite_evaluation_copies_only_owned_gradient_arrays` | P | aliasing safety | semantic correctness of sanitized values |
| `test_minimize_alm_classifies_relative_reduction_false_success_as_infeasible_stall` | E (faked-inner) | history `inner_false_success=True`, `inner_stall_reason=...` | actual termination of full-driver run |
| `test_minimize_alm_updates_duals_without_growing_penalty_after_meaningful_progress` | E (faked-inner) | first action is `dual_update`, second multipliers value | property "no penalty growth iff feasibility decreased" |
| `test_minimize_alm_uses_current_progress_for_three_dual_updates_before_penalty` | E (faked-inner) | sequence `[dual, dual, dual, penalty_increase]` | invariant for arbitrary stationarity sequences |
| `test_minimize_alm_tolerances_nonincreasing_across_dual_penalty_dual` | P | `feasibility_tolerance` and `stationarity_tolerance` are non-increasing | invariant under early termination |
| `test_minimize_alm_applies_multiplier_cap_on_dual_update` | E (faked-inner) | post-cap multiplier == 0.2 | property "multipliers <= multiplier_max" for arbitrary inputs |
| `test_minimize_alm_applies_multiplier_cap_in_normalized_units` | E (faked-inner) | scaled cap | invariant under scale |
| `test_minimize_alm_result_preserves_raw_signed_constraint_values` | S | passthrough | semantic |
| `test_minimize_alm_history_entry_has_stable_schema` | S | schema | semantic |
| `test_minimize_alm_tracks_active_constraint_switching` | E (faked-inner) | active constraint name across 2 outers | semantic correctness of "active" definition |
| `test_minimize_alm_handles_high_dimension_many_constraints_smoke` | S/E (faked-inner) | result.x shape, history values length | KKT, multiplier sign, penalty growth |
| `test_minimize_alm_caps_relaxed_feasibility_gate_before_dual_update` | E (faked-inner) | gate cap | invariant |
| `test_minimize_alm_dual_updates_after_zero_work_feasible_stall_*` (3) | E (faked-inner) | history dispatch under feasible-stall variants | KKT |
| `test_minimize_alm_uses_full_outer_budget_after_tolerance_tightens` | E (faked-inner) | outer count after tightening | invariant |
| `test_minimize_alm_keeps_outer_loop_running_after_feasible_stall` | E (faked-inner) | does not early-terminate | invariant |
| `test_minimize_alm_stops_on_long_feasible_no_progress_plateau` | E (faked-inner) | `termination_reason=plateau_stall` | when does plateau actually represent KKT vs not |
| `test_minimize_alm_escalates_penalty_for_material_violation_above_capped_gate` | E (faked-inner) | history action sequence | KKT |
| **`test_minimize_alm_accepts_kkt_stationarity_at_nearly_active_inequality_boundary`** | E (faked-inner) | `kkt_stationarity_norm=0`, `raw_stationarity_norm=1` (i.e. raw-but-not-projected) | does NOT assert termination_reason="converged" — asserts `max_outer_after_dual_update`! KKT is only asserted on a single history entry, not as a termination contract |
| `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective` | E (faked-inner) | `restored_best_feasible=True`, `result.x = best.x` | property "best_feasible.objective is monotonically non-increasing across outers" |
| `test_minimize_alm_restores_best_feasible_solver_owned_incumbent_state` | E (faked-inner) | snapshot/restore hook is called | invariant |
| `test_minimize_alm_interrupts_inner_solver_when_kkt_gate_is_hit_in_callback` | E (faked-inner) | callback raises early-stop | KKT verification at termination |
| `test_minimize_alm_skips_inner_solver_when_current_iterate_already_converged` | E (faked-inner) | inner_iterations=0 (but FAILS to assert `success=True`!) | actual converged termination |
| `test_minimize_alm_reports_inner_and_accepted_callbacks_separately` | E (faked-inner) | callback list | semantic |
| `test_minimize_alm_uses_metric_gradient_for_convergence_diagnostics` | E (faked-inner) | metric grad routing | semantic |
| `test_next_penalty_caps_requested_growth` | N | helper at fixed input | invariant under arbitrary mu |
| `test_next_penalty_caps_on_overflow_when_no_max` | N | overflow saturation | invariant |
| `test_minimize_alm_stops_when_penalty_cap_blocks_further_growth` | E (faked-inner) | `termination_reason=penalty_cap_reached` | KKT residual at termination |
| `test_minimize_alm_rejects_initial_penalty_above_cap` | S | ValueError | semantic |
| `test_continuation_step_short_circuits_when_iterate_is_already_converged` | E | termination_reason="converged" via continuation step (calls real inner solver — but trivially because the iterate is already feasible+stationary) | non-trivial converged termination |
| `test_continuation_step_breaks_outer_after_infeasible_stall_penalty_increase` | E (faked-inner) | `_ALMContinuationDecision.BREAK_OUTER` | three stall classes simultaneously |
| `test_outer_iteration_propagates_return_decision_from_continuation_step` | S+E | enum dispatch via mock | algorithmic |
| `test_outer_iteration_signals_exhaust_when_final_outer_and_no_return` | S+E | enum dispatch | algorithmic |
| `test_outer_iteration_signals_next_outer_when_not_final_and_no_return` | S+E | enum dispatch | algorithmic |
| `test_outer_iteration_emits_outer_state_callback_exactly_once_before_continuation` | P | callback count == 1 | callback ordering w/ multiple continuations |
| `test_outer_iteration_dispatches_until_continuation_budget_exhausted` | E (mocked-step) | continuation budget loop | algorithmic |
| `test_relative_reduction_of_f_with_no_feasibility_gain_is_false_success` | N | `_classify_infeasible_inner_stall` arm | full-driver dispatch into this arm |
| `test_other_success_message_is_generic_false_success` | N | classification arm | full-driver dispatch |
| `test_failed_inner_without_feasibility_gain_is_failed_stall` | N | classification arm | full-driver dispatch |
| `test_movement_above_tolerance_is_not_stall` | N | non-stall arm | full-driver dispatch |
| `test_candidate_within_feasibility_gate_is_not_stall` | N | non-stall arm | full-driver dispatch |
| **`test_minimize_alm_skips_inner_solve_when_already_converged`** | E (mocked SciPy with `AssertionError`) | scipy never called, `termination_reason="converged"`, `success=True` | This is one of only TWO tests that actually assert `success=True`+`converged` — but it does so only when the iterate is *already* trivially KKT-satisfying with `c=-1.0` (strict-interior, lambda=0) |

### `tests/geo/test_alm_benchmarking.py` (10 tests)

| Test | Cat | Asserted | NOT asserted |
|---|---|---|---|
| `test_alm_relevance_prefers_explicit_constraint_method` | S | DB-row classification | algorithmic |
| `test_baseline_row_detects_nested_normalized_alm_fields` | S | normalized field detection | algorithmic |
| `test_build_baseline_summary_reads_registry_ledger_artifacts_and_seeds` | S | DB schema | algorithmic |
| `test_autoresearch_root_requires_explicit_arg_or_env` | S | error path | algorithmic |
| `test_registry_rows_raise_on_missing_database` | S | error path | algorithmic |
| `test_write_baseline_outputs_preserves_comparison_schema` | S | schema | algorithmic |
| `test_comparison_rows_join_ledger_rows_by_row_index` | S | join shape | algorithmic |
| `test_comparison_rows_join_run_artifacts_by_source_path` | S | join shape | algorithmic |
| `test_write_after_outputs_writes_after_rows_and_joined_comparison` | S | schema | algorithmic |

(All shape-only — these test the autoresearch baseline-comparison registry, NOT the ALM driver.)

### `tests/geo/test_alm_fixture_benchmarking.py` (6 tests)

| Test | Cat | Asserted | NOT asserted |
|---|---|---|---|
| `test_fixture_raw_and_normalized_constraints_share_feasible_set` | N | feasible at upper-bound point | KKT |
| `test_run_fixture_benchmark_emits_raw_and_normalized_rows` | S/E | payload schema (calls real ALM but only checks output shape) | KKT, multiplier sign, success |
| `test_run_fixture_benchmark_preserves_explicit_empty_inner_options` | S | empty options pass-through | algorithmic |
| `test_cli_does_not_default_to_user_specific_autoresearch_root` | S | CLI arg parsing | algorithmic |
| `test_evaluate_fixture_rejects_unknown_formulation` | S | ValueError | algorithmic |
| `test_write_fixture_benchmark_writes_json_payload` | S | JSON schema | algorithmic |

### `tests/geo/test_single_stage_alm_integration.py` (~80 tests)

These tests are dominated by **CLI argument schema** (parse_args, validation, wrapper-command construction) and **constraint metadata payload** schemas. They do not exercise `minimize_alm`. Selected examples:

| Test | Cat | Asserted | NOT asserted |
|---|---|---|---|
| `test_single_stage_parse_args_exposes_alm_trust_region_controls` | S | CLI arg presence | algorithmic |
| `test_single_stage_builds_bounded_alm_settings` | S | settings dataclass field values | algorithmic |
| `test_single_stage_zero_trust_radius_disables_bounds_in_settings` | S | bounds=None | invariant |
| `test_hardware_constraint_schema_declares_expected_targets` | S | constraint name list | algorithmic |
| `test_alm_metadata_rejects_block_incompatible_value_kinds` | S | ValueError | algorithmic |
| `test_alm_metadata_accepts_physics_raw_value_kind_contract` | S | accept | algorithmic |
| `test_single_stage_alm_constraint_names_follow_shared_schema` | S | name list shape | algorithmic |
| `test_stage2_alm_wrapper_*` (~30 tests) | S | CLI wrapper construction, schema rejection / acceptance | algorithmic |
| `test_stage2_results_contract_records_*` | S | results dict schema | algorithmic |
| `test_stage2_seed_loader_reuses_saved_biot_savart_configuration` | S | seed file format | algorithmic |
| `test_single_stage_thresholded_physics_rerun_wrapper_*` (~20 tests) | S | wrapper CLI | algorithmic |

(All of these are config-shape tests. None drive `minimize_alm`.)

### `tests/geo/test_stage2_track_b_wrappers.py` (8 tests)

All shape-only / CLI-wrapper tests. None drive ALM.

### `tests/geo/test_single_stage_workflow_helpers.py` (127 tests)

Workflow-level CLI/artifact tests; checked via grep that no test calls `minimize_alm` or asserts KKT. All shape.

### `tests/geo/test_banana_objective_modules.py` (88 tests)

Two references to `alm_stationarity_tol`, both in CLI wrapper construction. None drive `minimize_alm`.

### `tests/geo/test_surface_mode_contracts.py` (12 tests)

CLI/schema tests; do not drive ALM.

### `tests/geo/test_ishw_deliverables.py` (37 tests)

ISHW deliverable JSON-schema tests; do not drive ALM.

---

## Per-question answers

### Q1: KKT verification at termination?
**No.** No test asserts both `||c(x)||_inf <= feasibility_tol` AND `||grad L_A||_inf <= stationarity_tol` together at the **final** iterate. The closest tests:
- `test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint` (line 1712) asserts `result.x[0] <= 1+1e-6` AND `result.constraint_values[0] <= 1e-6` AND `final_kkt_stationarity_norm ~= 0` — but it asserts `assertFalse(result.success)` (because of how `success` interacts with `final_kkt_stationarity_norm` flag), and `final_raw_stationarity_norm` is asserted to be 1.0 (i.e. NOT zero). This is a contradiction with strict KKT.
- `test_minimize_alm_skips_inner_solve_when_already_converged` (line 4397) asserts `success=True` AND `termination_reason="converged"` — but only because the iterate trivially has `c=-1.0` (strict interior with the surrogate `stationarity_norm=0` already met). It does not test a non-trivial KKT solution.
- `test_minimize_alm_accepts_kkt_stationarity_at_nearly_active_inequality_boundary` (line 3451) asserts `kkt_stationarity_norm==0`/`raw_stationarity_norm==1.0` at one history entry — but the termination is `max_outer_after_dual_update`, not "converged". This pins the kkt-norm helper, not the driver's stop condition.

A driver bug that returns a non-stationary point (e.g. forgets to update `state.final_eval` before the converged check) would be caught only by the trivial skipped-inner path.

### Q2: Multiplier sign (lambda_i >= 0 for inequalities)?
**Helper-level only.** `test_project_nonnegative_multipliers_enforces_nonnegativity_and_cap` (line 785) and `test_project_nonnegative_multipliers_uses_vector_penalty` (line 235) confirm the projection helper. `test_dual_update_projects_multipliers_and_tightens_tolerances` checks `np.all(result.multipliers >= 0.0)` for one fixed 2-vector input. **No test runs the full driver and asserts at termination that all final multipliers are non-negative.**

### Q3: Penalty schedule grows on stalled feasibility?
**Partial.** `test_minimize_alm_increases_penalty_only_when_feasibility_is_bad` (line 2210) only confirms a single x10 escalation across 2 outer iterations (`penalty_init=1.0` -> `result.penalty=100.0`). `test_minimize_alm_uses_current_progress_for_three_dual_updates_before_penalty` (line 2542) confirms a 4-step `[dual, dual, dual, penalty_increase]` sequence. No test asserts the **monotone-non-decreasing** property `mu_{k+1} >= mu_k forall k`, and no test asserts saturation at the `penalty_max` (only the cap-termination is tested, not the schedule arithmetic across many cycles).

### Q4: Stall classification — does `382d7a082` actually drive all three classes?
**No.** Inspection of `382d7a082` (commit-introduced tests):
- `_classify_infeasible_inner_stall` is unit-tested by 5 helper tests (`tests/geo/test_alm_utils.py:4283-4383`): `relative_reduction_false_success`, `other_success_message`, `failed_inner_without_feasibility_gain`, `movement_above_tolerance_is_not_stall`, `candidate_within_feasibility_gate_is_not_stall`. These are **textual SciPy-message dispatch tests** on `SimpleNamespace` instances, never running the ALM.
- The full-driver tests (`test_minimize_alm_classifies_relative_reduction_false_success_as_infeasible_stall` at line 2442; `test_minimize_alm_short_circuits_zero_step_infeasible_stall` at line 2323) drive only the "infeasible_stall" arm with a faked SciPy `minimize`. **They never exercise "progress" (meaningful) -> "skipped_inner" within a single end-to-end run with a real inner solver.**
- The **skipped-inner-shortcut** test (`test_minimize_alm_skips_inner_solve_when_already_converged` at line 4397) is good (it patches scipy `minimize` to raise `AssertionError`), but it does not assert the `_ALMOuterDecision` enum value of the outer loop — it only asserts `result.termination_reason="converged"` at the public level. The decision-enum dispatch is tested separately at lines 4155-4256, but with a mocked `_run_alm_continuation_step`, so the enum-emission path is decoupled from any real algorithm execution.

Verdict: stall classification is pinned **piecewise** at the unit level, not as an integrated three-class behavior.

### Q5: Phase-4 result-carrier shape and infeasible-stall arm (`531a33c3c`)?
**Locked but shape-only.** `test_phase4_result_carriers_are_frozen_dataclasses_without_mutable_defaults` (line 905) asserts `frozen=True` and `not isinstance(default, (dict,list,set))`. This is structural. The infeasible-stall arm is locked at `test_continuation_step_breaks_outer_after_infeasible_stall_penalty_increase` (line 4041), which uses a faked-inner `_make_fake_minimize()` and asserts `decision == BREAK_OUTER`, `penalty == settings.penalty_init * settings.penalty_scale`, and history `action == "infeasible_stall_penalty_increase"`. This is good arm-level pinning but does not check the full numerical state of the carrier.

### Q6: Normalization invariance?
**No.** The two normalization tests (`test_scale_one_minimize_alm_matches_scalar_history_and_convergence` at line 1750 and `test_scaled_inequality_fixture_preserves_raw_feasibility_and_dual_conversion` at line 797) only confirm `scale=1.0` identity or single-fixture arithmetic. **No test runs `minimize_alm` end-to-end with `scales = [s, s]` for two non-trivial values of `s` and asserts that final iterates and (post-scale-undoing) multipliers transform consistently.** A regression where the dual-update step forgets to multiply/divide by `s` would not be caught.

### Q7: Sign-flipped constraint?
**No protective test.** All constraint-bearing E2E tests use the documented sign convention `signed = c(x) - upper`. If a regression flipped this sign (e.g. swapped `upper - c(x)`), `augmented_inequality_objective` would still produce a finite-but-wrong gradient, and the trivial scalar problem at line 1712 would converge to the wrong feasibility region or fail to converge at all. The test at line 1812 (`test_minimize_alm_requires_signed_constraint_activity_for_boundary_kkt_success`) is close but only asserts `multipliers == [0.0]` (i.e. inactive). No test asserts the **sign** of the multiplier for an *active* boundary constraint.

### Q8: NaN/Inf robustness?
**Partial.** `test_minimize_alm_sanitizes_nonfinite_candidate_evaluations` (line 1587) injects NaN into `total` and `grad` for a *candidate* evaluation and confirms the driver falls back. **It does not test:**
- NaN in `constraint_values` (the most likely physics-side regression).
- NaN in `dual_update_values`.
- Inf in any field.
- NaN propagated through several outer iterations (i.e. that state corruption does not accumulate).
- NaN at the *initial* iterate (which would be a calling-convention bug).

The fallback test `test_sanitize_nonfinite_evaluation_copies_only_owned_gradient_arrays` (line 2405) is an aliasing test, not a numerical-robustness test.

### Q9: Feasible-incumbent monotonicity?
**No.** `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective` (line 3496) confirms the *restoration* and that the best-feasible objective is preferred over a worse final iterate. **It does not assert the property that** `best_feasible.objective` is non-increasing **across all outer iterations**. The hand-crafted `evaluate_problem` exercises a 4-region piecewise function; if a regression replaced "improves" with "differs" in the comparator at line 3984-3986, this test would still pass (because the test only checks the *one* restoration outcome, not the per-outer monotonicity).

### Q10: Block-penalty correctness?
**Diagnostics-only.** `test_minimize_alm_records_constraint_blocks_as_diagnostics_only` (line 251) explicitly asserts `result.block_penalties is None` — i.e. block-aware penalty escalation is **explicitly not implemented** as of the current contract. Block grouping is tested at the diagnostic helper level (`test_constraint_history_diagnostics_groups_values_by_block` at line 537). No per-block penalty *update* test exists, because no per-block penalty update is implemented. If a future PR adds per-block penalties, there is no protective test to confirm correctness.

---

## Top-10 missing tests, ranked by risk

### 1. End-to-end KKT verification at termination (HIGH)
**Bug it would catch:** any regression that returns a non-stationary or infeasible point as `success=True`. E.g. a missing `update_feasibility_tol` floor leak, a swapped argument in `_handle_alm_dual_update_transition`, or a forgotten `penalty * c(x)` term in the augmented gradient.

**Sketch:** Solve a 5-D quadratic with two active inequality constraints (one on a coordinate sum, one on the L_2 norm) using the **real** inner solver. Assert: (a) `result.success is True`, (b) `np.max(np.abs(result.constraint_values)) <= settings.feasibility_tol`, (c) `np.linalg.norm(grad_L_aug, np.inf) <= settings.stationarity_tol` where `grad_L_aug` is reconstructed from the final `evaluate_problem(result.x, result.multipliers, result.penalty)["grad"]`, (d) `result.multipliers >= -1e-12` elementwise, (e) for each active constraint i, `result.multipliers[i] * result.constraint_values[i] <= 1e-9` (complementary slackness).

### 2. Multi-constraint multiplier sign at termination (HIGH)
**Bug it would catch:** a sign-flip in the dual update that would otherwise drive multipliers negative (then projection masks it but with corrupted iterates).

**Sketch:** Use a 3-D problem with 5 inequality constraints, three active. Assert at the final iterate `np.all(result.multipliers >= 0.0)` and `np.any(result.multipliers > 1e-6)` (i.e. at least one strictly positive). Also assert `np.all(result.history[-1]["post_update_multipliers"]) >= 0.0`.

### 3. Normalization invariance under scale != 1 (HIGH)
**Bug it would catch:** a regression where the dual update forgets to convert between normalized and physical units (most likely site: gradient passthrough or `_handle_alm_penalty_*`).

**Sketch:** Same scalar problem as `test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint`, but evaluated under `scales = [16000.0]` (physical) and `scales = [1.0]` (raw). Run both end-to-end. Assert `np.allclose(scaled_result.x, raw_result.x, atol=1e-6)`, `np.allclose(scaled_result.multipliers / 16000.0, raw_result.multipliers, atol=1e-6)`, and identical `termination_reason`.

### 4. Constraint sign-flip detection (HIGH)
**Bug it would catch:** silent sign flips in `signed_constraint_value = c(x) - upper`.

**Sketch:** Use a strictly convex objective `f(x) = (x-2)^2/2` with inequality `x <= 1` (must be active at `x=1, lambda=1`). Assert `result.success is True` AND `0.99 <= result.x[0] <= 1.01` AND `0.99 <= result.multipliers[0] <= 1.01`. Then add a second test that does the same but with a *flipped*-sign hand-crafted `evaluate_problem` — and assert that the flipped version either (a) fails `result.success` or (b) lands on the wrong feasibility region (`x[0] < 0.5`). The second test pins the *failure mode* under a known-bad sign convention; if it ever passes, a sign-related semantic shift has happened.

### 5. NaN in constraint_values robustness (MEDIUM)
**Bug it would catch:** physics-side NaN propagation through the constraint pipeline (likely site: constraint normalization with division by ill-conditioned scale, or finite-difference gradient mis-step).

**Sketch:** `evaluate_problem` returns `constraint_values=np.array([np.nan])` once at iteration 2, then valid values after. Assert: (a) the driver does not produce a NaN final `result.x`, (b) `result.history[k]["nonfinite_candidate_evaluation"]` is True for the bad iteration, (c) no subsequent history entry has `np.nan` in `constraint_values` or `multipliers`, (d) the driver eventually terminates with a defined `termination_reason`.

### 6. Best-feasible monotonicity invariant (MEDIUM)
**Bug it would catch:** a regression where the comparator at lines 3984-3986 (`<` vs `<=` vs `>`) is corrupted, leading to non-monotone improvement.

**Sketch:** Solve a multi-modal feasible problem (e.g. `f(x) = (x[0]^2 - 1)^2 + 0.1*(x[1]-x[0]^3)^2` with a single inequality) with `max_outer_iterations=10`. Use a `history_callback` to record `state.best_feasible.evaluation["base_value"]` at each outer iteration. Assert that the recorded sequence is non-increasing.

### 7. Penalty-schedule monotonicity over many infeasible outers (MEDIUM)
**Bug it would catch:** a regression where `_next_penalty` returns a non-increasing value (e.g. `min` instead of `max` in the cap clamp).

**Sketch:** With `max_outer_iterations=8`, a faked-inner that always returns infeasible+stalled, and `penalty_init=1.0, penalty_scale=10.0, penalty_max=1e6`. Assert that `[entry["penalty"] for entry in result.history]` is sorted ascending and that the final `result.penalty == min(penalty_init * scale^N, penalty_max)`.

### 8. Three-class stall classification in one full-driver run (MEDIUM)
**Bug it would catch:** the outer loop's `_ALMOuterDecision` dispatch could miss one of the three classes if continuation-step or outer-iteration dispatch is refactored.

**Sketch:** Hand-craft a `evaluate_problem` whose responses cycle: outer 1 makes meaningful progress (-> RETURN/NEXT_OUTER), outer 2 returns infeasible-stall (-> BREAK_OUTER + penalty_increase), outer 3 returns iterate already-KKT (-> RETURN with `converged`). Use a real (or carefully-faked) inner solver and capture the `_ALMOuterDecision` enum value via `outer_state_callback` or by patching `_run_alm_outer_iteration` to record. Assert all three enum values appear in the captured sequence in the expected order.

### 9. Equality-constraint contract reject (LOW)
**Bug it would catch:** if a future change adds equality-constraint plumbing without updating tests, the dual-update sign convention would silently corrupt.

**Sketch:** Pass an `evaluate_problem` whose `constraint_values` semantics imply equality (e.g. `c(x) = 0` line). Without explicit equality plumbing, the test asserts that the driver either rejects with a clear `ValueError` or produces an inequality-style result with `multipliers >= 0`. This protects the contract scope.

### 10. Augmented-gradient numerical-Taylor consistency at convergence (LOW)
**Bug it would catch:** any regression in the assembled augmented gradient that goes undetected because the inner solver does not run gradient checks.

**Sketch:** After running a converged 2-D ALM solve, call `run_directional_taylor_test` on a closure that wraps `evaluate_problem` with the *final* multipliers and penalty (so the test is on the augmented Lagrangian, not the base objective). Assert `result["passed"] is True` and `result["max_ratio"] < 1e-3`.

---

## Strengths

1. **Phase-4 closure-free orchestrator** structural pinning (`test_minimize_alm_public_entrypoint_is_small_and_closure_free`, `test_minimize_alm_impl_orchestrator_is_small_and_closure_free`) is a real architectural guard — it would catch a refactor that re-introduces nested closures.
2. **History entry schema lock** (`_HISTORY_ENTRY_SHARED_KEYS` with 52 explicit keys) prevents accidental field renames or drops.
3. **Frozen-dataclass + no-mutable-default invariant** (`test_phase4_result_carriers_are_frozen_dataclasses_without_mutable_defaults`) catches a class of aliasing bugs.
4. **Aliasing/ownership tests** (`test_history_diagnostic_materialization_is_pure_and_snapshots_are_owned`, `test_sanitize_nonfinite_evaluation_copies_only_owned_gradient_arrays`) defend against shared-mutable-state bugs which are subtle.
5. **Stall-classification arm dispatch** at the helper level (5 tests against `_classify_infeasible_inner_stall`) pins SciPy message-text contracts that the repo cannot otherwise version-pin (`scipy>=1.5.4`).
6. **Skipped-inner shortcut** test patches `minimize` to `AssertionError` — this is the *correct* way to assert "the inner solver was not called", and is a strong test.
7. **Tolerance non-increasing** test (`test_minimize_alm_tolerances_nonincreasing_across_dual_penalty_dual`) is one of the few **invariant** tests in the suite.
8. **Multiplier-cap normalization-units test** (`test_minimize_alm_applies_multiplier_cap_in_normalized_units`) is a rare scale-aware test that the suite needs more of.
9. **Constraint-routing diagnostic helper** has good fixture-based coverage of block-grouping, signal mismatch, and surrogate-vs-hard sign disagreement — these are non-trivial cases where the diagnostic is itself the contract.
10. **Active-constraint switching** test (`test_minimize_alm_tracks_active_constraint_switching`) is a small but meaningful integration check on the multi-constraint dispatch.
