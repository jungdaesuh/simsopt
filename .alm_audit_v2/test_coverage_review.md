# ALM Test Coverage Audit v2 — 2026-05-08

Repo: `/Users/suhjungdae/code/columbia/simsopt-surrogate`
HEAD: `e7b836464` (`surrogate-confinement-v2`)
Subject: production `alm_utils.py` (4847 LOC) + 3 runner modules + 4 test files (5519 LOC of tests)
Method: every test in `tests/geo/test_alm_utils.py` (140 tests), `tests/geo/test_single_stage_alm_integration.py` (74 tests), `tests/geo/test_alm_benchmarking.py` (9 tests), `tests/geo/test_alm_fixture_benchmarking.py` (6 tests) was read and mapped to `examples/single_stage_optimization/alm_utils.py` line numbers; the post-bf936a0a4 fix list (M2, M3.a, M3.b, M4, M5, M6, M8, M9, L1-L5) was reverse-mapped to its production sites and matched against test assertions.

## Summary

The `bf936a0a4` "harden driver boundary" commit added 3 new test classes (~20 tests) and 3 high-level integration tests that materially improve coverage of M3.b, M8, H1, T1.a, and M1 (deterministic termination). Together with prior history-shape pinning and helper-level unit tests, the suite is **structurally robust against accidental contract drift** but has **5 critical gaps** where a previously-fixed bug could regress without any test catching it:

- **M4 (cap-active gates both success arms): no negative test exists.** The cap-binding diagnostic is asserted (`multiplier_cap_binding=True`), but no test asserts that the converged labels are *blocked* when cap binding is active. `not run_state.last_cap_binding_active` at `alm_utils.py:4273` and `alm_utils.py:4320` could be deleted without breaking any test.
- **M5 (post-inner routing uses effective_feasibility_tol): no direct test.** A single comment at `tests/geo/test_alm_utils.py:3867-3870` references M5, but no assertion compares pre-inner vs post-inner routing tolerance values.
- **M6 (`_kkt_stationarity_norm` `nnls` `RuntimeError` only): no test for the catch arm.** Adding `OverflowError`, `Exception`, or removing the catch entirely would not break any test.
- **M9 (KKT diagnostic uses `base_grad`, not `metric_grad`): no test.** The augmented-vs-bare gradient distinction at `alm_utils.py:2228-2231` is not asserted anywhere.
- **L1 (no-blocks lane shallow-copy): no test.** Asymmetric ownership between lanes at `alm_utils.py:2360-2364` is not pinned.

Beyond the v1 fix verification, the suite has **structural weaknesses** that v1 already identified (no end-to-end KKT for non-trivial multi-constraint problem, no normalization invariance under non-trivial scale, no NaN-in-constraint-values robustness, no per-outer best-feasible-monotonicity invariant, smoke-only 12-constraint test) — none of these were closed by `bf936a0a4`. The new T1.a test (line 4304) is a substantive 2-D KKT integration test, but it is the *only* multi-dimensional non-trivial KKT integration test.

Severity tally: 5 CRITICAL, 9 HIGH, 7 MEDIUM, 2 LOW.

---

## Methodology

1. Read prior audit `.alm_audit/test_coverage_review.md` for v1 findings (test_count=139 then; now 140 in test_alm_utils.py; 229 total in 4 files).
2. Read `bf936a0a4` commit message + `.alm_audit/FIX_PLAN.md` for the M/L fix inventory.
3. Mapped each fix to its production site by `grep`-ing production code for the fix-comment markers (`# M2:`, `# M3.a:`, `# M4:`, `# M9:`, `# L1:`, etc.).
4. For each fix site, searched test files for assertions on the post-fix behavior using `grep` on:
   - The fix marker (`grep -n "M2\|M3.a"`),
   - The action label / history field (`grep -n "subproblem_limit_penalty_increase"`),
   - The runtime predicate (`grep -n "last_cap_binding_active"`).
5. Read the actual assertion blocks (not just the comment) to verify whether the test would catch a regression of the fix.
6. Cross-referenced the v1 audit's "tests do NOT cover" list to determine which gaps were closed and which persist.

---

## Findings

### F1: M4 cap-active predicate has no negative test [CRITICAL]
- Production code: `examples/single_stage_optimization/alm_utils.py:4264-4296` (converged arm gated by `not run_state.last_cap_binding_active`); `alm_utils.py:4316-4346` (constraints_inactive_converged arm with same gate); `alm_utils.py:4434` (predicate update).
- Test status: **no negative test**. `test_minimize_alm_applies_multiplier_cap_on_dual_update` (`tests/geo/test_alm_utils.py:2949-2996`) asserts `result.history[0]["multiplier_cap_binding"] == True` and `result.multiplier_cap_binding == True`, but it does not assert that the converged label is *blocked*. The test runs only 2 outer iterations, never reaches converged tolerances, and asserts `result.success is False` for an unrelated reason (`max_outer_iterations=2` exhaustion).
- Suggested test: `test_minimize_alm_blocks_converged_label_when_multiplier_cap_is_binding`. Build a fixture where (a) an outer iteration reaches `feasibility_tol` AND `stationarity_tol` AND `signal_mismatch_active=False`, (b) the dual update binds at `multiplier_max`. Assert `result.termination_reason != "converged"` AND `result.success is False`. Then run a parallel fixture where the same conditions hold but the cap is large enough to avoid binding. Assert `result.termination_reason == "converged"` AND `result.success is True`.
- Why it matters: the cap-active gate is the M4 fix's *only* runtime effect. Deleting `and not run_state.last_cap_binding_active` from line 4273 (or replacing it with a stale `cap_binding_detected` sticky flag) would be a silent regression of the v1 audit's S1 finding (which spawned M4).

### F2: M5 post-inner `effective_feasibility_tol` consistency has no assertion [CRITICAL]
- Production code: `examples/single_stage_optimization/alm_utils.py:4118-4124` (post-inner `_constraint_routing_state` uses clamped `effective_feasibility_tol`); compare with pre-inner usage at `_pre_inner_constraint_routing_state` site.
- Test status: **smoke + comment only**. The only M5 reference is a comment at `tests/geo/test_alm_utils.py:3867-3870` inside `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective`, which says "M5: post-inner routing now uses the clamped effective_feasibility_tol consistent with pre-inner". The test asserts `minimize_calls["count"] == 4`, but does not extract pre-inner vs post-inner routing tolerance values from history.
- Suggested test: `test_minimize_alm_post_inner_routing_uses_same_effective_feasibility_tol_as_pre_inner`. Capture two history entries via `history_callback` — one before the inner solve fires, one after. Assert `entry_pre["effective_feasibility_tolerance"] == entry_post["effective_feasibility_tolerance"]` for the same outer iteration when `relaxed_feasibility_gate_cap` is set.
- Why it matters: M5 prevents tolerance divergence within a single outer step. A regression that re-introduces the divergence would silently route different constraint signals through pre-inner versus post-inner branches, giving incoherent dispatch decisions. The fix is functionally invisible without a paired comparison test.

### F3: M6 `_kkt_stationarity_norm` `nnls` `RuntimeError` catch is unguarded [CRITICAL]
- Production code: `examples/single_stage_optimization/alm_utils.py:2170-2188`. `try: nnls(active_matrix, -total_grad_array, maxiter=10*active_matrix.shape[1]) except RuntimeError as exc: warnings.warn(...); return None`.
- Test status: **no test**. `_kkt_stationarity_norm` is never called directly in tests — only `_surrogate_kkt_stationarity_norm` (the wrapper) is exercised at `tests/geo/test_alm_utils.py:743`, but that test path provides a tiny consistent active set where `nnls` converges easily. The `RuntimeError`-catching arm at line 2181 is dead-code from the test suite's perspective.
- Suggested test: `test_kkt_stationarity_norm_catches_nnls_runtime_error`. Use `unittest.mock.patch.object(module, "nnls", side_effect=RuntimeError("too many iterations"))` and call `_kkt_stationarity_norm` with a non-empty active set. Assert (a) the helper returns `None`, (b) a `RuntimeWarning` is emitted via `warnings.catch_warnings()`, (c) shape errors (`ValueError`) still propagate (separate test arm).
- Why it matters: M6 was added because production runs hit `nnls` "too many iterations" on ill-conditioned active Jacobians. If the catch is broken (e.g., catches `Exception` masking real errors, or fails to set `kkt_stationarity_norm=None`), an entire ALM run could abort with a diagnostic-helper exception. No test would catch this regression.

### F4: M9 KKT diagnostic uses `base_grad` not `metric_grad` — no test [CRITICAL]
- Production code: `examples/single_stage_optimization/alm_utils.py:2221-2231`. Comment "M9: KKT stationarity is `‖∇f + Σλ_i∇c_i‖`, NOT `‖∇L_A + Σλ_i∇c_i‖`. Augmented gradient `∇L_A` already contains the active-constraint contribution `(λ + μc)∇c`, so feeding it to nnls' active-set projection collapses to ~0 once the inner solve converges and hides multiplier-quality defects." Falls back to `metric_grad` only when `base_grad` is missing.
- Test status: **no test**. The T1.a test at `tests/geo/test_alm_utils.py:4304` returns `base_grad` in the evaluation, but does not assert that swapping `base_grad` for `metric_grad` would break the test (because `metric_grad == base_grad + augmented_grad` with magnitude 1.0 at the converged minimizer for that fixture, the test could pass with either gradient feed).
- Suggested test: `test_kkt_stationarity_uses_base_grad_not_augmented_grad`. Build a 2-D fixture where the augmented gradient ≠ bare base gradient at the iterate (e.g., active constraint with non-zero multiplier shift). Run two evaluations: one with `base_grad` set explicitly, one omitting it. Assert that `kkt_stationarity_norm` differs in the second case (where `metric_grad` is used) and that the `base_grad` value matches the analytical bare ∇f.
- Why it matters: M9 silently masks multiplier-quality defects in the augmented-gradient case. A regression that re-routes `metric_grad` into `_kkt_stationarity_norm` would zero-out the diagnostic at convergence (because the augmented gradient is already minimized) and hide infeasibility-on-active-constraints failures.

### F5: L1 no-blocks lane shallow-copy contract has no test [CRITICAL]
- Production code: `examples/single_stage_optimization/alm_utils.py:2352-2365`. `_attach_alm_constraint_metadata` always shallow-copies; "Previously the no-blocks lane returned the caller's dict (alias) while the blocks lane shallow-copied. Uniform ownership: this function returns a dict the caller owns."
- Test status: **no test**. No test asserts that `_attach_alm_constraint_metadata(evaluation, names_tuple, None)` returns a dict that is `not evaluation` (i.e., a copy). The L2 history-list-copy is well-tested at `tests/geo/test_alm_utils.py:2673-2696`, but the L1 metadata-attach contract is untested.
- Suggested test: `test_attach_alm_constraint_metadata_returns_owned_dict_in_no_blocks_lane`. `evaluation = {"total": 1.0, "grad": np.zeros(1)}; result = module._attach_alm_constraint_metadata(evaluation, ("c0",), None); self.assertIsNot(result, evaluation); result["total"] = 99.0; self.assertEqual(evaluation["total"], 1.0)`.
- Why it matters: aliasing bugs are the most insidious source of accepted-state corruption. L1 was added precisely because the no-blocks lane returned the caller's dict; if a refactor reverts to `return evaluation`, downstream snapshot caches would be silently corrupted by subsequent mutations.

### F6: ALMSettings `__post_init__` validation has 11 untested rejection arms [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:37-77` validates 11 fields. Tests cover `penalty_max <= 0`, `penalty_max < penalty_init`, `history_max_entries <= 0`. **No test** for: `max_outer_iterations <= 0`, `max_subproblem_continuations < 0`, `penalty_init <= 0`, `penalty_scale <= 1`, `feasibility_tol <= 0`, `stationarity_tol <= 0`, `trust_radius_init < 0`, `trust_radius_min <= 0`, `trust_radius_shrink not in (0,1)`, `trust_radius_grow <= 1`, `max_inner_attempts <= 0`, `multiplier_max <= 0`.
- Test status: **smoke / partial**. Of 14 fields, only 3 have rejection tests at `tests/geo/test_alm_utils.py:1135-1162`.
- Suggested tests: 11 small `subTest` cases under `AlmSettingsPostInitValidationTests`. Each constructs `ALMSettings(field=invalid_value)` and asserts the matching `ValueError` regex. L5's docstring-claim that programmatic construction "cannot bypass the CLI-level guards" is currently a partial claim.
- Why it matters: L5 was added as defense against `ALMSettings(trust_radius_grow=0.5)` silently shrinking. The unguarded fields are silent-corruption surfaces.

### F7: `validate_alm_cli_args` has no rejection tests [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:389-432`. Validates 14 CLI argument fields with custom error messages.
- Test status: **patched out**. The only reference is `tests/geo/test_single_stage_example.py:11108` which patches `validate_alm_cli_args` to a no-op. No test asserts any of the 14 rejection arms (e.g., `--alm-max-outer-iters=0`, `--alm-penalty-init=-1.0`, `--alm-trust-radius-shrink=2.0`).
- Suggested test class: `ValidateAlmCliArgsTests` with 14 subTest cases mirroring the 14 raise sites at lines 390-432.
- Why it matters: The error messages are operator-facing; a regression that silently accepts `--alm-penalty-scale=0.5` would corrupt the schedule without raising. Operator-visible CLI guards are a documented contract surface.

### F8: end-to-end KKT verification still asserts contradictory expectations [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:4264-4296` (converged arm).
- Test status: **inadequate**. `test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint` (`tests/geo/test_alm_utils.py:1861-1897`) asserts `result.x[0] <= 1+1e-6`, `result.constraint_values[0] <= 1e-6`, `result.final_kkt_stationarity_norm == 0.0` BUT also asserts `assertFalse(result.success)` AND `result.final_raw_stationarity_norm == 1.0`. This is contradictory: KKT residual is zero but raw stationarity is 1.0 (because the augmented gradient at the boundary still has the multiplier shift). The test does not pin success at the converged labels.
- The new T1.a test (`tests/geo/test_alm_utils.py:4304-4374`) is a **substantive** 2-D KKT integration test that asserts `success=True`, `termination_reason="converged"`, `multipliers ≈ [0.5]`, `multipliers >= 0`. This is the strongest KKT pin in the suite. It was added by `bf936a0a4`.
- Suggested test: extend T1.a-style coverage with (a) a 5-D problem with 3 active constraints and complementary slackness `λ_i * c_i ≈ 0` checked, (b) a problem where the optimal multiplier is at the cap (testing M4 negative path simultaneously), (c) a problem with one active inequality and one strictly inactive (asserting `multipliers[inactive] == 0.0`). Also: add an assertion to the line-1861 test that *explicitly* documents the partial-KKT outcome (e.g., "this fixture terminates with `max_outer` because the augmented gradient is non-zero at the boundary; replace with T1.a-style fixture for full KKT").
- Why it matters: a single 2-D fixture pins the integral fix but does not catch sign-flips or off-by-ones in higher dimensions or under cap binding.

### F9: Multi-constraint multiplier-sign property has no end-to-end test [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:2988-2995` (`_handle_alm_dual_update_transition` calls `_project_nonnegative_multipliers_with_diagnostics`).
- Test status: **helper-level only**. `test_dual_update_projects_multipliers_and_tightens_tolerances` at `tests/geo/test_alm_utils.py:969-1001` asserts `np.all(result.multipliers >= 0.0)` for one fixed 2-vector input. T1.a asserts `result.multipliers[0] >= 0.0` for a 1-multiplier 2-D problem. **No test** drives `minimize_alm` end-to-end with ≥ 3 constraints (multiple active) and asserts `np.all(result.multipliers >= 0.0)`.
- Suggested test: `test_minimize_alm_final_multipliers_are_all_nonnegative_under_multiple_active_constraints`. 3-D problem, 5 inequalities, 3 active. Use real (not faked) inner solver. Assert `np.all(result.multipliers >= -1e-12)` AND `np.any(result.multipliers > 1e-6)`. Also assert `np.all([entry["post_update_multipliers"][i] >= 0 for entry in result.history for i in range(5)])`.
- Why it matters: a sign-flip in `_project_nonnegative_multipliers_with_diagnostics` could go unnoticed if the projection only zeroes negatives. A multi-constraint fixture is the only environment where the per-element signs can be exercised meaningfully.

### F10: NaN/Inf robustness coverage is incomplete [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:1324-1345` (`_sanitize_nonfinite_inner_evaluation`); used at multiple sanitize-boundary call sites.
- Test status: **partial**. `test_minimize_alm_sanitizes_nonfinite_candidate_evaluations` at line 1736 injects NaN into `total` and `grad` of a candidate. It does **not** test:
  - NaN in `constraint_values` (the most likely physics-side failure mode).
  - NaN in `dual_update_values` (would corrupt multiplier projection).
  - Inf (vs NaN) in any field.
  - NaN at the *initial* iterate (calling-convention bug).
  - NaN propagation across multiple outer iterations.
- Suggested tests:
  - `test_minimize_alm_sanitizes_nonfinite_constraint_values`.
  - `test_minimize_alm_sanitizes_nonfinite_dual_update_values`.
  - `test_minimize_alm_rejects_nonfinite_initial_iterate` (should raise at `_normalize_alm_run_inputs`).
  - `test_minimize_alm_sanitizes_inf_total_and_grad`.
- Why it matters: physics evaluators returning NaN under adverse geometry are the most common production failure mode. The current single test covers only the simplest path.

### F11: Penalty-schedule monotonicity invariant has no test [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:_next_penalty` and `_emit_alm_penalty_increase_arm`.
- Test status: **point-test only**. `test_minimize_alm_increases_penalty_only_when_feasibility_is_bad` (`tests/geo/test_alm_utils.py:2493-2524`) only confirms `result.penalty == 100.0` after 2 outer iterations (single x10 escalation). `test_next_penalty_caps_requested_growth` (line 4148) and `test_next_penalty_caps_on_overflow_when_no_max` (line 4161) test the helper at single fixture inputs.
- Suggested test: `test_minimize_alm_penalty_schedule_is_monotonically_nondecreasing_under_repeated_infeasible_stalls`. With `max_outer_iterations=8`, faked-inner that always returns `infeasible_stall`, `penalty_init=1.0`, `penalty_scale=10.0`, `penalty_max=1e6`. Assert (a) `[entry["penalty"] for entry in result.history]` is sorted-ascending, (b) `result.penalty == min(penalty_init * scale^N, penalty_max)`, (c) consecutive entries differ by exactly `penalty_scale`.
- Why it matters: a regression where `_next_penalty` returns `min` instead of `max` against the cap, or a sign error in the schedule, would silently halve the schedule.

### F12: Best-feasible monotonicity invariant has no test [HIGH]
- Production code: best-feasible incumbent comparison in `_minimize_alm_impl` (the comparator at lines 3984-3986 in v1; now restructured in `_run_alm_outer_iteration`).
- Test status: **endpoint-only**. `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective` (`tests/geo/test_alm_utils.py:3789-3885`) asserts the *final* restoration. **No test** asserts the per-outer property `best_feasible.objective[k+1] <= best_feasible.objective[k]`.
- Suggested test: `test_minimize_alm_best_feasible_objective_is_monotonically_nonincreasing`. Use `outer_state_callback` to capture `(iteration, multipliers, penalty)` at each outer; correlate with `result.history[k]["base_value"]` filtered by the feasibility predicate. Assert non-increasing across the feasible subsequence.
- Why it matters: a regression where the comparator becomes `<` (strict) or flips to `>` would not be caught by the current endpoint test.

### F13: Normalization invariance under non-trivial scale has no test [HIGH]
- Production code: `examples/single_stage_optimization/alm_utils.py:normalize_alm_constraints`, downstream multiplier-cap at line 2998 (raw vs normalized).
- Test status: **scale=1 identity only**. `test_scale_one_minimize_alm_matches_scalar_history_and_convergence` (line 1899-1959) confirms `scale=1.0` identity. **No test** runs `minimize_alm` end-to-end with `scales=[s, s]` for two non-trivial values of `s` and asserts equivalent solutions. `test_minimize_alm_applies_multiplier_cap_in_normalized_units` at line 2998 tests a single point of the scaling logic with a faked-inner.
- Suggested test: `test_minimize_alm_normalization_invariance_under_nontrivial_scale`. Same problem as T1.a, evaluated at `scales=[16000.0]` (physical) and `scales=[1.0]` (raw). Assert `np.allclose(scaled.x, raw.x)`, `np.allclose(scaled.multipliers / 16000.0, raw.multipliers)`, identical termination_reason, identical history actions.
- Why it matters: a regression that forgets to convert between normalized and physical units in the dual update would corrupt the multiplier sequence for any production run.

### F14: Empty constraint set has no end-to-end test [MEDIUM]
- Production code: `examples/single_stage_optimization/alm_utils.py:minimize_alm` (no explicit guard against empty constraint list).
- Test status: **helper-level only**. `test_normalize_alm_constraints_accepts_empty_constraint_set` at line 474 confirms helper behavior. **No test** drives `minimize_alm(x0, constraint_names=[], evaluate_problem, settings, inner_options)`.
- Suggested test: `test_minimize_alm_unconstrained_problem_short_circuits_to_inner_solve`. Build a 1-D unconstrained quadratic. Assert `result.success is True`, `result.multipliers.shape == (0,)`, `result.constraint_values.shape == (0,)`, `result.history` reports `action="converged"` after one outer iteration.
- Why it matters: empty-constraint is a degenerate but valid mode; a refactor that assumes ≥ 1 constraint somewhere in the dispatch would silently break.

### F15: All-feasible / all-violating starts have no edge-case tests [MEDIUM]
- Production code: dispatch in `_run_alm_continuation_step` branches on `hard_feasible_for_update` (line 4411), `constraints_inactive_candidate` (line 4316). Both edges (all-feasible from start, all-violating from start) are valid starting conditions for production runs.
- Test status: **none specific**. `test_minimize_alm_returns_constraints_inactive_converged_for_stage2_zero_shift` (line 2194) covers the all-feasible-from-start path, but uses faked-inner. **No test** covers all-violating-from-start with a non-faked inner solver.
- Suggested tests:
  - `test_minimize_alm_all_violating_initial_iterate_drives_inner_solve_into_feasibility`. Real inner solver, x0 violates both constraints in the T1.a-style problem.
  - `test_minimize_alm_all_strictly_feasible_initial_iterate_terminates_constraints_inactive`. x0 deeply interior; assert `termination_reason="constraints_inactive_converged"` after real inner solve.
- Why it matters: production runs frequently start outside or on the boundary; the dispatch behavior differs at each entry point.

### F16: 12-constraint smoke test asserts only shape [MEDIUM]
- Production code: full `minimize_alm` for high-dimension multi-constraint problem.
- Test status: **smoke only**. `test_minimize_alm_handles_high_dimension_many_constraints_smoke` at `tests/geo/test_alm_utils.py:3286-3328` runs 128-D / 12-constraint with faked-inner. Asserts only `result.x.shape == (128,)` and `len(result.history[0]["constraint_values"]) == 12`.
- Suggested upgrade: add assertions on `result.history[0]["multiplier_cap_binding_indices"]`, `np.all(result.multipliers >= 0)`, KKT consistency for the active subset, and `result.termination_reason in {"converged", "max_outer", ...}` (an explicit allowlist).
- Why it matters: marked as "smoke" but hides as a coverage entry. Promoting to a property test costs ~5 lines.

### F17: Three-class stall classification not driven in single end-to-end run [MEDIUM]
- Production code: `_classify_infeasible_inner_stall` at `alm_utils.py:_classify_infeasible_inner_stall`; outer-loop dispatch into `_ALMOuterDecision`.
- Test status: **piecewise only**. 5 helper tests at `tests/geo/test_alm_utils.py:4712-4814` pin the SciPy-message-text dispatch arms in isolation. The full driver tests exercise *one* arm per test (e.g., `test_minimize_alm_classifies_relative_reduction_false_success_as_infeasible_stall` at line 2735). **No test** drives "meaningful progress -> infeasible stall -> skipped inner" within a single end-to-end run.
- Suggested test: `test_minimize_alm_three_class_stall_classification_in_single_run`. Hand-craft an evaluator whose responses cycle: outer 1 progress (RETURN/NEXT_OUTER), outer 2 infeasible-stall (BREAK_OUTER + penalty_increase), outer 3 skipped-inner (RETURN converged). Capture `_ALMOuterDecision` enum via `outer_state_callback` patch. Assert all three values appear in expected order.
- Why it matters: outer-loop dispatch refactors could miss one of the three classes; the piecewise tests would not catch that the integration is broken.

### F18: `_emit_alm_*` retry arms have no direct unit tests [MEDIUM]
- Production code: 4 emit helpers at `alm_utils.py:3610` (`_emit_alm_subproblem_continue`), `alm_utils.py:3646` (`_emit_alm_stall_failure_step`), `alm_utils.py:3696` (`_emit_alm_converged_step`), `alm_utils.py:3750` (`_emit_alm_penalty_increase_arm`). Each owns a slice of the history-entry mutation contract (e.g., `outer_termination="max_outer"` annotation, `subproblem_limit_reason` annotation).
- Test status: **indirect only**. Each is exercised through `minimize_alm` integration tests; none has a direct unit test.
- Suggested tests: 4 small unit tests, each calling the helper with a hand-built `_ContinuationStepState` and asserting the history-entry mutations. This would pin the emit-arm contract independently of the driver.
- Why it matters: M2 (`outer_termination="max_outer"` annotation) lives at `_emit_alm_subproblem_continue`. A regression that forgets the annotation in *one* of the two call sites (signal-mismatch retry vs feasible-update retry) would only be caught by a happens-to-cover test. Direct unit tests would lock the annotation contract.

### F19: M2 covers only the signal-mismatch retry call site, not the feasible-update retry [MEDIUM]
- Production code: `_emit_alm_subproblem_continue` is called from two sites: `alm_utils.py:4403` (after signal-mismatch retry) and `alm_utils.py:4534` (after feasible-update retry). Both should annotate `outer_termination="max_outer"` when `is_final_outer=True`.
- Test status: **partial**. `test_minimize_alm_escalates_penalty_after_repeated_stage2_signal_mismatch` (line 2286) reaches the signal-mismatch retry path. The feasible-update retry path is exercised by `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective` (line 3789), which DOES assert `result.history[2]["action"] == "subproblem_continue"` AND `result.history[3]["outer_termination"] == "max_outer"`. But this assertion at line 3885 is on `subproblem_limit_penalty_increase` (the M3.a arm), not on `subproblem_continue` itself.
- Look at line 3525-3588 — `test_minimize_alm_uses_full_outer_budget_after_tolerance_tightens` asserts `result.history[0]["outer_termination"] == "max_outer"`, but the action is `dual_update`, not `subproblem_continue`.
- Suggested test: `test_minimize_alm_subproblem_continue_annotates_max_outer_on_feasible_update_retry`. Build a fixture that drives the feasible-update retry path on the final outer iteration and asserts `result.history[k]["action"] == "subproblem_continue"` AND `result.history[k]["outer_termination"] == "max_outer"`.
- Why it matters: M2 was specifically introduced because *both* call sites needed the annotation. The current tests cover the annotation generically but do not pin both call paths simultaneously.

### F20: Run-to-run determinism only tested under sustained mismatch [MEDIUM]
- Production code: ALM is deterministic given fixed seed and evaluator; the only stochastic input is the directional Taylor test seed.
- Test status: **mismatch-only**. `test_alm_terminates_deterministically_under_sustained_signal_mismatch` (line 2359-2491) asserts run-to-run identical history for the mismatch fixture. **No test** asserts run-to-run determinism on a converging fixture.
- Suggested test: `test_minimize_alm_is_deterministic_on_same_fixture`. Run T1.a fixture twice. Assert `np.array_equal(run1.x, run2.x)`, `run1.history == run2.history`, identical multipliers and penalty.
- Why it matters: determinism is an undocumented but assumed property; a regression that introduces nondeterministic dispatch (e.g., via dict-iteration order in metadata construction) would fail reproducibility audits silently.

### F21: alm_fixture_benchmarking does not assert numerical equivalence [MEDIUM]
- Production code: `examples/single_stage_optimization/banana_opt/alm_fixture_benchmarking.py:run_fixture_case`.
- Test status: **schema only**. `test_run_fixture_benchmark_emits_raw_and_normalized_rows` at `tests/geo/test_alm_fixture_benchmarking.py:33-80` asserts payload schema (formulations present, schema_version, comparison fields). Does NOT assert that raw and normalized formulations land at the same minimizer or share the same multipliers (post-scale-undoing).
- Suggested upgrade: add assertions `np.allclose(raw_row["x"], normalized_row["x"], atol=1e-5)`, `np.allclose(raw_row["multipliers"] / fixture.scale_array(), normalized_row["multipliers"], atol=1e-5)`, `raw_row["termination_reason"] == normalized_row["termination_reason"]`.
- Why it matters: the fixture benchmark is the closest thing to a regression-ALM-trajectory test in the suite. Adding 3 lines turns it into one.

### F22: Penalty-cap-reached path has no KKT assertion [MEDIUM]
- Production code: `examples/single_stage_optimization/alm_utils.py:_handle_alm_penalty_cap_termination` at line 3123.
- Test status: **termination-reason only**. `test_minimize_alm_stops_when_penalty_cap_blocks_further_growth` (line 4174) asserts `result.termination_reason == "penalty_cap_reached"`. Does NOT assert any property of the final iterate (e.g., the violation should be at-or-near `feasibility_tol`, the multiplier should be at the cap, the message should reference the cap value).
- Suggested upgrade: add assertions on `result.x`, `result.constraint_values`, `result.multipliers`, `result.message`.

### F23: Constraint sign-flip detection has no protective test [LOW]
- Production code: `signed_constraint_value = c(x) - upper` convention threaded through ALM.
- Test status: **detection-only via numeric pin**. `test_augmented_inequality_objective_uses_projected_multiplier_shift` (line 103) asserts specific values on a 2-constraint fixture; a sign flip would change them. T1.a (line 4304) would also detect a sign flip but only because the new minimizer would fall on the wrong side of the constraint. Neither test makes the sign-detection contract explicit.
- Suggested test: a "negative-control" test that constructs a *flipped*-sign hand-crafted `evaluate_problem` and asserts the driver reaches the wrong region (`result.x` lands on the infeasible side). This locks the contract by asserting the failure mode.
- Why it matters: cosmetic — the sign would be caught by existing tests, but only indirectly.

### F24: Normalization-invariance edge: empty scales [LOW]
- Production code: `normalize_alm_constraints([], [], [], [], [])`.
- Test status: tested at line 474. Edge fully covered.
- No issue. (Confirms strength.)

---

## v1 Fix Verification Coverage

Per-fix mapping. "Pinned" = a test exists whose assertion would fail if the fix were reverted; "diagnostic-only" = test asserts the diagnostic but not the runtime effect.

| Fix | Production site | Test? | Strength | Notes |
|---|---|---|---|---|
| **M2** (subproblem_continue annotates `outer_termination="max_outer"` on both call paths) | `alm_utils.py:3610-3643` | Partial | MEDIUM | F19. The annotation is asserted at multiple history-entry checks (e.g., line 2042, 2524, 2562, 2692, 3381, 3525, 3588, 3885), but none specifically pins both call paths simultaneously. |
| **M3.a** (max_subproblem_continuations exhaustion → penalty_increase arm) | `alm_utils.py:4500-4527` | Pinned | STRONG | `test_minimize_alm_restores_best_feasible_incumbent_by_base_objective` (line 3878-3884) asserts `action="subproblem_limit_penalty_increase"` and `subproblem_limit_reason="max_subproblem_continuations"`. Strong coverage. |
| **M3.b** (gtol ratchet removed; user anchor preserved) | `alm_utils.py:4170-4204` | Pinned | STRONG | `test_minimize_alm_preserves_user_gtol_anchor_across_outer_iterations` (line 4251-4302). Strong: directly observes inner-solve gtol values across iterations. |
| **M4** (cap-binding gates both success arms) | `alm_utils.py:4264-4296`, `4316-4346`, `4434` | **No** | MISSING | F1. Only diagnostic (`multiplier_cap_binding=True`) is asserted; no negative test that the converged label is *blocked*. CRITICAL gap. |
| **M5** (post-inner routing uses effective_feasibility_tol) | `alm_utils.py:4118-4124` | **No** | MISSING | F2. Only a comment in one test references M5. CRITICAL gap. |
| **M6** (`nnls` `RuntimeError` only) | `alm_utils.py:2170-2188` | **No** | MISSING | F3. Catch arm is dead-code from test perspective. CRITICAL gap. |
| **M7** (scale floor provenance) | hardware-constraint schema layer | Pinned | STRONG | `test_physics_alm_metadata_records_scale_floor_when_threshold_below_floor` and 4 variants in `tests/geo/test_banana_objective_modules.py:2656-4644`. |
| **M8** (driver-boundary multiplier validation) | `alm_utils.py:_normalize_alm_run_inputs` + `validate_initial_multipliers` | Pinned | STRONG | `ValidateInitialMultipliersTests` (8 tests at line 1284-1333) + `AlmNormalizeRunInputsValidationTests` extensions (line 1188-1240). Strong coverage. |
| **M9** (KKT diagnostic uses `base_grad`) | `alm_utils.py:2221-2231` | **No** | MISSING | F4. Augmented-vs-bare gradient distinction not asserted. CRITICAL gap. |
| **L1** (`_attach_alm_constraint_metadata` always shallow-copies) | `alm_utils.py:2352-2365` | **No** | MISSING | F5. No-blocks-lane shallow-copy contract not pinned. CRITICAL gap. |
| **L2** (history snapshot is defensive copy) | `alm_utils.py:2521-` (history snapshot helper) | Pinned | STRONG | `test_minimize_alm_short_circuits_zero_step_infeasible_stall` (line 2606-2696) explicitly tests `assertIsNot(snapshot["history"], result.history)` and mutation isolation. Strong. |
| **L3, L4** (cache + sanitize boundaries clone evaluation dict; `_build_augmented_evaluation` `.copy()`s every ndarray) | `alm_utils.py:1306-1321`, `555-603` | Partial | MEDIUM | `test_sanitize_nonfinite_evaluation_copies_only_owned_gradient_arrays` (line 2698-2733) is a strong aliasing test for the sanitize boundary. **No test** for the cache boundary or for `_build_augmented_evaluation`'s ndarray copy contract. |
| **L5** (`ALMSettings.__post_init__` validates 14 fields) | `alm_utils.py:37-77` | Partial | WEAK | F6. Only 3 of 14 fields have rejection tests. |
| **T1.a** (gold-standard KKT integration) | tests | Pinned | STRONG | `test_minimize_alm_converges_to_kkt_on_active_linear_inequality` (line 4304-4374). 2-D quadratic with `x+y<=1`, λ=0.5, success=True, multipliers >= 0, KKT feasibility checked. Strong. |
| **T1.e** (signal-mismatch deterministic termination) | tests | Pinned | STRONG | `test_alm_terminates_deterministically_under_sustained_signal_mismatch` (line 2359-2491). Strong: re-runs same fixture, asserts identical termination_reason, identical action sequence. |
| **H1** (`require_positive_alm_threshold`) | `alm_utils.py:require_positive_alm_threshold` | Pinned | STRONG | `RequirePositiveAlmThresholdTests` (6 tests at line 1243-1281). Strong. |

Net: **5 of 13 driver-fixes are NOT pinned by post-fix regression tests** (M4, M5, M6, M9, L1). 1 is partial-only (M2 indirect, L5). 7 are strongly pinned.

---

## Test Quality Issues

### TQ1: Smoke-only labeled "tests" [MEDIUM]
- `test_minimize_alm_handles_high_dimension_many_constraints_smoke` (`tests/geo/test_alm_utils.py:3286`). Asserts only shape (`result.x.shape`, `len(history[0]["constraint_values"])`). The "_smoke" suffix is honest, but the test could cheaply add multiplier-sign and termination-allowlist assertions.

### TQ2: Tests with mock-too-aggressive [MEDIUM]
- `test_outer_iteration_propagates_return_decision_from_continuation_step` (`tests/geo/test_alm_utils.py:4584`) and 4 sibling tests at lines 4604-4685 mock `_run_alm_continuation_step` entirely. They exercise enum-dispatch routing only, never real algorithm execution. They are correct as arm-dispatch tests but should be paired with at least one integration test that doesn't mock the inner step. Only `test_outer_iteration_emits_outer_state_callback_exactly_once_before_continuation` (line 4635) is partly real (does not mock the callback contract).

### TQ3: Trivial assertions on dataclass typing [LOW]
- `test_dual_update_projects_multipliers_and_tightens_tolerances` (line 1000-1001) asserts `isinstance(result.multiplier_cap_binding, bool)` and `isinstance(result.multiplier_cap_binding_indices, list)`. These pass trivially due to dataclass typing. Either remove or replace with a runtime invariant.

### TQ4: Tests asserting `assertFalse(result.success)` as the success criterion [HIGH]
- `test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint` (line 1893) asserts `assertFalse(result.success)` while also asserting KKT-style numerical pins. This is internally contradictory and undermines the purpose of the test. The post-bf936a0a4 T1.a test (line 4358) correctly asserts `assertTrue(result.success)`. The line-1893 test should be either retired or refactored.

### TQ5: History-equality test as the only frozen-trajectory regression test [MEDIUM]
- `test_scale_one_minimize_alm_matches_scalar_history_and_convergence` (line 1899-1959) at line 1959 asserts `raw_result.history == scale_one_result.history`. This is the *only* test that pins a full ALM history trajectory. It only does so on the trivial `scale=1.0` identity. Adding a frozen-trajectory regression test for T1.a-style problems would catch refactor-induced drift.

### TQ6: `alm_fixture_benchmarking` numerical equivalence not asserted [MEDIUM]
- See F21. The benchmarking infrastructure runs ALM end-to-end at two formulations but only asserts payload schema. Adding 3 numerical assertions would convert the test into a normalization-invariance regression suite.

---

## Coverage by Category

| Category | Coverage | Notes |
|---|---|---|
| Outer driver (`_minimize_alm_impl`, `_run_alm_outer_iteration`) | STRONG | Structural pins (line 856-902), dispatch tests (line 4584-4685), integration via T1.a (line 4304). |
| Continuation step (`_run_alm_continuation_step`) | MEDIUM | 2 dedicated tests (line 4437-4502); covers converged short-circuit and infeasible-stall arm. Other arms covered indirectly only. |
| Each retry arm (`_emit_alm_*`) | WEAK | F18: no direct unit tests. M2 only partially covered by both-call-paths assertion. |
| Math primitives (multiplier projection, augmented terms) | STRONG | `test_project_nonnegative_multipliers_*` (3 tests), `test_augmented_inequality_objective_*` (4 tests). |
| Normalization (block-penalty removal, scale floors) | STRONG | 5 dedicated tests (line 446-498). M7 strongly covered in `test_banana_objective_modules.py`. |
| Diagnostics (history schema, conditioning_metrics) | STRONG | History schema lock with 52 keys (line 1383); conditioning metrics 3 tests (line 1688-1734). |
| KKT verification at termination | MEDIUM | T1.a (line 4304) is the strongest test. F8 documents the contradiction in line 1861. |
| Multiplier sign invariant | WEAK | F9. Only helper-level tests; no end-to-end multi-constraint sign assertion. |
| NaN/Inf robustness | WEAK | F10. Only `total` and `grad` NaN; no `constraint_values`, `dual_update_values`, Inf, initial-iterate NaN. |
| Penalty schedule monotonicity | WEAK | F11. Only single-step escalation tested. |
| Best-feasible monotonicity | WEAK | F12. Only endpoint restoration tested. |
| Normalization invariance under non-trivial scale | WEAK | F13. Only `scale=1.0` identity. |
| Empty / edge constraint sets | WEAK | F14, F15. No end-to-end tests. |
| Three-class stall classification (single-run) | WEAK | F17. Only piecewise. |
| `validate_alm_cli_args` rejection arms | MISSING | F7. Not tested at all. |
| ALMSettings `__post_init__` rejection arms | WEAK | F6. 3 of 14 fields tested. |
| Determinism (run-to-run) | WEAK | F20. Only mismatch-fixture tested. |
| End-to-end runner test on small problem | MEDIUM | `test_run_fixture_benchmark_emits_raw_and_normalized_rows` (`tests/geo/test_alm_fixture_benchmarking.py:33`) drives ALM end-to-end but only asserts schema. |
| Hybrid contract (surrogate vs hard signal) | STRONG | T1.e + 6 routing-state tests + signal_mismatch_active tests. |
| Documentation tests (CONTRACT.md examples mirrored) | MEDIUM | The hybrid contract document explicitly cites `test_alm_terminates_deterministically_under_sustained_signal_mismatch` as its sole verification — that single test backs the contract. No CONTRACT.md examples for M2/M4/M5/M9 fixes. |

### Performance/timing
- No `@pytest.mark.slow` or skip markers in any test file. All tests run unconditionally. The full T1.a integration test (`test_minimize_alm_converges_to_kkt_on_active_linear_inequality`) likely takes < 1s; no need for slow-marker. **No issue.**

---

## Verdict

Confidence in coverage: **MEDIUM**. The bf936a0a4 commit added 3 substantive new tests (T1.a KKT integration, T1.e mismatch determinism, M3.b gtol-anchor) that materially close v1 audit gaps. M3.a, M3.b, M7, M8, T1.a, T1.e, H1, L2 are now strongly pinned. However, **5 driver-fix arms (M4, M5, M6, M9, L1) added in the same commit lack post-fix regression tests** — these CRITICAL gaps mean the fixes themselves can be reverted without any test failing.

The structural-test surface is unusually strong (closure-free orchestrator pins, frozen-dataclass invariants, 52-key history schema, defensive-copy assertions for L2). The integration-test surface is unusually weak in several specific dimensions: end-to-end multi-constraint multiplier sign, normalization invariance under non-trivial scale, NaN-in-constraint-values, best-feasible monotonicity. None of these were closed by `bf936a0a4`.

### Top 5 missing tests, ranked by risk

1. **`test_minimize_alm_blocks_converged_label_when_multiplier_cap_is_binding`** (F1, M4) — pins the only runtime effect of M4. CRITICAL.

2. **`test_kkt_stationarity_norm_catches_nnls_runtime_error`** (F3, M6) — keeps the M6 catch arm alive. CRITICAL.

3. **`test_kkt_stationarity_uses_base_grad_not_augmented_grad`** (F4, M9) — protects multiplier-quality diagnostic from silent collapse. CRITICAL.

4. **`test_minimize_alm_post_inner_routing_uses_same_effective_feasibility_tol_as_pre_inner`** (F2, M5) — locks the M5 invariant. CRITICAL.

5. **`test_attach_alm_constraint_metadata_returns_owned_dict_in_no_blocks_lane`** (F5, L1) — pins the L1 ownership contract. CRITICAL.

Beyond these CRITICAL entries:

6. **`test_minimize_alm_sanitizes_nonfinite_constraint_values`** (F10) — closes the most-likely production-failure NaN path. HIGH.

7. **`ValidateAlmCliArgsTests`** class with 14 subTest cases (F7) — pins the operator-facing CLI contract. HIGH.

8. **`AlmSettingsPostInitValidationTests`** extension to cover all 14 fields (F6, L5) — closes the "programmatic construction can bypass CLI guards" risk that L5 was meant to address. HIGH.

9. **`test_minimize_alm_normalization_invariance_under_nontrivial_scale`** (F13) — protects the entire scale-aware production pipeline. HIGH.

10. **`test_minimize_alm_final_multipliers_are_all_nonnegative_under_multiple_active_constraints`** (F9) — extends T1.a coverage to ≥ 3 active constraints. HIGH.
