# ALM Algorithm/Control-Flow Audit v2 — 2026-05-08

## Summary

The v1 fix commit `bf936a0a4` lands most of the M2/M3.a/M3.b/M4/M5/M6/M8/M9
fixes correctly, but **three new control-flow defects survive** plus several
diagnostic-consistency gaps. The most serious is **F1**: the M4
`last_cap_binding_active` predicate is gated on the post-inner converged arm
and the constraints-inactive arm, but the **pre-inner skipped-inner shortcut
at `_run_alm_continuation_step` L3968-L4057 is not gated** — so a run whose
previous outer iteration ended with `dual_update.multiplier_cap_binding=True`
can still emit `success=True` with `termination_reason="converged"` on the
next outer iteration's continuation 0, in clear violation of the M4 contract.

Two MEDIUM gaps survive in M5: post-inner `_attach_alm_history_diagnostics`
still uses the unclamped `state.update_feasibility_tol` (L4250), and the
sibling diagnostic helper `_surrogate_kkt_stationarity_norm` still feeds
augmented `metric_grad` (not `base_grad`) into `_kkt_stationarity_norm`,
mirroring the exact M9 bug that was fixed only in `_stationarity_metrics`.
The M2 fix is correct at the helper level but has **no test pinning the
feasible-update retry call site** (audit FIX_PLAN_REVIEW.md flagged this in
v1 as INSUFFICIENT-TESTS; the fix landed without that test).

`_termination_reason_from_history` does not have a specific case for the
new `subproblem_limit_penalty_increase` action, so on final-outer the
returned label is the bare `"max_outer"` rather than a more informative
`max_outer_after_subproblem_limit_penalty_increase`. This is observable but
non-load-bearing.

The remaining items (initial multiplier upper bound vs `multiplier_max`,
`x0` finiteness validation, `_termination_reason_from_history` ordering)
are LOW.

## Methodology

- Read `.alm_audit/algorithm_review.md`, `FIX_PLAN.md`, `FIX_PLAN_REVIEW.md`,
  and `ADVERSARIAL_REVIEW.md` to anchor on prior findings.
- Read `git show bf936a0a4 -- examples/single_stage_optimization/alm_utils.py`
  (full diff, 658 lines) to see what changed under each fix label.
- Re-read the entire driver
  (`examples/single_stage_optimization/alm_utils.py`, 4847 lines) end to end,
  with explicit attention to the audit checklist (terminator labels,
  `subproblem_limit` routing, `gtol` persistence, cap-binding gating,
  effective tolerance consistency, `nnls` catch scope, multiplier validation,
  retry-arm precedence/race, history append/snapshot ordering, off-by-one,
  and the `_made_meaningful_inner_progress` semantics).
- Cross-referenced `tests/geo/test_alm_utils.py` (4877 lines) for
  characterization coverage of each fix.
- Hand-traced retry-arm dispatch precedence and state ownership across
  `_run_alm_continuation_step → _run_alm_outer_iteration → _minimize_alm_impl`.

## Findings

### F1: M4 cap-binding gate omits the pre-inner skipped-inner converged arm [HIGH]

- File: `examples/single_stage_optimization/alm_utils.py:3968-4057`
- Code:
  ```python
  if (
      current_max_feasibility_violation <= settings.feasibility_tol
      and current_stationarity_norm <= settings.stationarity_tol
      and not current_constraints_inactive_candidate
      and not current_signal_mismatch_active
  ):
      # ... append history "converged", set best_feasible, return RETURN
      # with success=True via _build_alm_converged_step_result(...,
      # termination_reason="converged")
  ```
- Bug: This is the third success arm in the driver — the pre-inner
  "skipped-inner shortcut" — but the M4 fix only added
  `and not run_state.last_cap_binding_active` at L4273 (post-inner converged)
  and L4320 (`constraints_inactive_converged`). The skipped-inner predicate
  at L3968 still emits `success=True` with `termination_reason="converged"`
  even when `run_state.last_cap_binding_active` is True from a prior outer
  iteration's clamped dual update.
- Why: The commit message for `bf936a0a4` says "M4: non-sticky
  `run_state.last_cap_binding_active` predicate gates **both** `converged`
  and `constraints_inactive_converged` success arms." But
  `_build_alm_converged_step_result` at L3456 has three call sites: L4041
  (skipped-inner), L4280 (post-inner converged via `_emit_alm_converged_step`,
  M4-gated), and L4327 (constraints-inactive via `_emit_alm_converged_step`,
  M4-gated). Site L4041 is the skipped-inner path and is not gated.
- Impact: Reachable wrong-success label. Concrete sequence:
  1. Outer iteration `k` reaches dual-update arm at L4411-L4452. The
     projection clamps the new multiplier at `settings.multiplier_max`, so
     `dual_update.multiplier_cap_binding = True` and L4434 sets
     `run_state.last_cap_binding_active = True`. BREAK_OUTER fires (no `x`
     change).
  2. Outer iteration `k+1` continuation 0 starts. `run_state.x` is unchanged
     from end of iter `k`. The augmented stationarity at the unchanged iterate
     is whatever it was at the end of iter `k`, which was `<=
     state.update_stationarity_tol` (the gate that triggered the dual update
     at L4411). After enough dual updates with `penalty_scale > 1`,
     `update_stationarity_tol` has tightened to `settings.stationarity_tol`
     (the floor at L3115-3116 in `_handle_alm_dual_update_transition`).
     Therefore `current_stationarity_norm <= settings.stationarity_tol` holds.
  3. If `current_max_feasibility_violation <= settings.feasibility_tol` and
     `not current_constraints_inactive_candidate` and
     `not current_signal_mismatch_active`, the L3968 gate fires, and the
     skipped-inner path returns RETURN with `success=True` and
     `termination_reason="converged"`.
  4. The result is mislabeled "converged" while
     `result.multiplier_cap_binding=True` and the underlying KKT residual is
     held artificially small by the cap, exactly the failure mode M4 was
     written to prevent.
- Repro: Construct a 1D inequality problem with a small enough
  `settings.multiplier_max` that the projection clamps on the first dual
  update; let the inner solve drive the iterate to user-strict feasibility
  and stationarity simultaneously. The first outer iteration will dual-update
  with cap; the second outer iteration's first continuation step will hit the
  L3968 gate and emit "converged" success. The post-inner gate at L4264 has
  the M4 guard, so it would block this — but the skipped-inner path runs
  *before* the inner solve and so reaches the success arm without traversing
  the M4-guarded L4264.
- Suggested fix: Add `and not run_state.last_cap_binding_active` to the
  L3968-L3973 predicate, mirroring the post-inner gate at L4273.
  Alternatively, factor the predicate into a shared helper
  `_alm_converged_arm_predicate(...)` that all three sites call. Add a
  characterization test that drives outer iter `k` to a clamped dual update
  and asserts that outer iter `k+1` does NOT label success via the
  skipped-inner path.

### F2: M5 fix is incomplete at `_attach_alm_history_diagnostics` [MEDIUM]

- File: `examples/single_stage_optimization/alm_utils.py:4241-4251`
- Code:
  ```python
  _attach_alm_history_diagnostics(
      history_entry,
      state.final_eval,
      state.multipliers,
      penalty_argument,
      constraint_names,
      solver_constraint_values,
      feasibility_values,
      routing_state,
      state.update_feasibility_tol,   # <-- still the unclamped tol
  )
  ```
- Bug: The M5 fix at L4098-L4112 changed the post-inner
  `_constraint_routing_state` and `_stationarity_metrics` calls to use
  `effective_feasibility_tol`. But the immediately-following
  `_attach_alm_history_diagnostics` call at L4250 still passes
  `state.update_feasibility_tol`. The diagnostics helper (via
  `_constraint_history_diagnostics_source` at L887-L975) uses this
  `feasibility_gate` argument as the active-set gate inside
  `_surrogate_kkt_stationarity_norm` at L851. Different gates → different
  active-set selections → different diagnostic numbers, in the same outer
  iteration.
- Why: The FIX_PLAN.md M5 §"Files to change" only lists `alm_utils.py:3936-3941`
  (the routing/stationarity calls). It does not list the
  `_attach_alm_history_diagnostics` call site. The diagnostic remained on
  the un-fixed side of the gate.
- Impact: The history entry's `surrogate_kkt_stationarity_norm` field
  (and any consumer-side reasoning that mixes it with the routing-state
  flags) is computed against a different active set than the routing
  decisions that lit the same arm. This is a diagnostic inconsistency, not
  a control-flow bug — but it is exactly the divergence M5 was meant to
  remove.
- Repro: Run an outer iteration where `state.update_feasibility_tol >
  effective_feasibility_tol` (i.e., on early ALM iterations before the
  schedule has tightened). Inspect `result.history[-1][
  "surrogate_kkt_stationarity_norm"]`; it will use a different active mask
  than the routing flags `signal_mismatch_active`, `hard_positive_shift_zero`
  in the same entry.
- Suggested fix: Replace `state.update_feasibility_tol` at L4250 with
  `effective_feasibility_tol`. Verify by also adding a unit test that
  asserts the history entry's `surrogate_kkt_stationarity_norm` matches the
  value computed against `effective_feasibility_tol`.

### F3: M9 fix not propagated to `_surrogate_kkt_stationarity_norm` [MEDIUM]

- File: `examples/single_stage_optimization/alm_utils.py:835-852`
- Code:
  ```python
  def _surrogate_kkt_stationarity_norm(
      evaluation: dict,
      routing_state: ALMConstraintRoutingState,
      feasibility_gate: float,
  ) -> float | None:
      metric_grad = np.asarray(
          evaluation.get("metric_grad", evaluation["grad"]),
          dtype=float,
      )
      surrogate_values = routing_state.signal_state.surrogate_signed_constraint_values
      return _kkt_stationarity_norm(
          metric_grad,                      # <-- augmented gradient
          evaluation.get("constraint_grads"),
          ...
      )
  ```
- Bug: The M9 fix at L2228-L2231 in `_stationarity_metrics` correctly
  switched the KKT diagnostic to use `evaluation.get("base_grad", metric_grad)`
  (bare ∇f) so the residual reflects true KKT and not inner-solve
  convergence. The sibling diagnostic helper at L835-L852,
  `_surrogate_kkt_stationarity_norm`, still uses `metric_grad` (the
  augmented gradient ∇L_A) as the input to `_kkt_stationarity_norm`. This
  is the exact pre-M9 bug for the surrogate-side diagnostic.
- Why: The commit message for `bf936a0a4` says only that
  `_stationarity_metrics` was changed. The audit's M9 reasoning ("Augmented
  gradient ∇L_A already contains the active-constraint contribution
  (λ + μc)∇c, so feeding it to nnls' active-set projection collapses to ~0
  once the inner solve converges and hides multiplier-quality defects")
  applies identically to the surrogate-side helper, but the fix did not
  propagate.
- Impact: The history field `surrogate_kkt_stationarity_norm` (and the
  result-level `final_surrogate_kkt_stationarity_norm` in the
  `alm_summary` payload at L1158-L1160) is structurally the same incorrect
  diagnostic that M9 fixed in the hard signal path. Anyone using
  `surrogate_kkt_stationarity_norm` to gauge multiplier quality on the
  surrogate side is reading a misleading number.
- Repro: Compute the field for a pair of evaluations: one with `metric_grad
  != base_grad`, one identical apart from a multiplier offset. Both will
  report `surrogate_kkt_stationarity_norm ≈ 0`, even when the multipliers
  are not at the true active-set Lagrange values.
- Suggested fix: In `_surrogate_kkt_stationarity_norm`, replace the
  `metric_grad` argument to `_kkt_stationarity_norm` with
  `np.asarray(evaluation.get("base_grad", metric_grad), dtype=float)`,
  mirroring L2228-L2231 in `_stationarity_metrics`. Add a test that
  exercises the surrogate-side diagnostic with `base_grad ≠ metric_grad`
  and asserts the residual reflects bare ∇f.

### F4: M2 feasible-update retry call site has no test [MEDIUM]

- File: `examples/single_stage_optimization/alm_utils.py:4534-4540` and
  `tests/geo/test_alm_utils.py` (no test for this site)
- Code:
  ```python
  return _emit_alm_subproblem_continue(
      state=state,
      run_state=run_state,
      history_entry=history_entry,
      history_callback=history_callback,
      is_final_outer=is_final_outer,   # <-- M2 wiring is correct
  )
  ```
- Bug: The M2 fix correctly threads `is_final_outer` into both call sites
  of `_emit_alm_subproblem_continue` (L4408 signal-mismatch retry and
  L4534 feasible-update retry). The helper at L3610-L3643 sets
  `history_entry["outer_termination"] = "max_outer"` when
  `is_final_outer=True`, which is correct. However the test suite only
  exercises the signal-mismatch path
  (`test_minimize_alm_escalates_penalty_after_repeated_stage2_signal_mismatch`
  at `tests/geo/test_alm_utils.py:2286` and
  `test_alm_terminates_deterministically_under_sustained_signal_mismatch`
  at `tests/geo/test_alm_utils.py:2359`). There is no test that exercises
  the feasible-update retry path's `is_final_outer=True` branch.
- Why: FIX_PLAN_REVIEW.md flagged this in v1 as INSUFFICIENT-TESTS:
  "Keep the helper signature change, but add a second final-outer test
  that reaches the feasible-update retry path at `alm_utils.py:4325-4330`."
  The recommendation was not followed in `bf936a0a4`.
- Impact: A regression in `_run_alm_continuation_step`'s feasible-update
  arm that drops the `is_final_outer` argument (e.g., during a future
  refactor) would not be caught by the existing test suite. The fix is
  correct as written, but unprotected.
- Repro: Construct a fixture where `signal_mismatch_active=False`,
  `hard_feasible_for_update=True`, `stationarity_norm > update_stationarity_tol`,
  `made_inner_progress=True`, `continuation_iteration=0 <
  max_subproblem_continuations`, and `is_final_outer=True`. The L4534
  helper fires; assert `result.history[-1]["outer_termination"] ==
  "max_outer"` and `result.termination_reason.startswith("max_outer")`.
- Suggested fix: Add a characterization test that drives the feasible-update
  retry path with `is_final_outer=True` and asserts both the
  `outer_termination` annotation and the resulting
  `result.termination_reason`.

### F5: `_termination_reason_from_history` lacks the `subproblem_limit_penalty_increase` case [MEDIUM]

- File: `examples/single_stage_optimization/alm_utils.py:1779-1795`
- Code:
  ```python
  if outer_termination == "max_outer":
      if restored_best_feasible:
          return "max_outer_restored_best_feasible"
      if latest_action == "dual_update":
          return "max_outer_after_dual_update"
      if latest_action == "subproblem_limit":
          return "max_outer_after_subproblem_limit"
      if latest_action == "infeasible_stall_penalty_increase":
          return "max_outer_after_infeasible_stall"
      if latest_action == "penalty_increase":
          return "max_outer_after_penalty_increase"
      return "max_outer"
  ```
- Bug: M3.a introduced a new history action string,
  `"subproblem_limit_penalty_increase"` (set at L4526), but
  `_termination_reason_from_history` does not have a specific case for it.
  When a final-outer iteration triggers M3.a's path without cap-hit, the
  result's `termination_reason` is the bare `"max_outer"` rather than a
  more informative `"max_outer_after_subproblem_limit_penalty_increase"`
  (or a logically grouped `"max_outer_after_subproblem_limit"`).
- Why: The new action label was added to the continuation arm but the
  reverse-mapping helper was not updated.
- Impact: Operators inspecting `result.termination_reason` to triage a
  failed run cannot distinguish "exhausted outer budget after final
  iteration's subproblem-limit penalty escalation" from a plain max_outer
  with no preceding action. The history entry at `result.history[-1]`
  carries `action="subproblem_limit_penalty_increase"` and
  `subproblem_limit_reason="max_subproblem_continuations"`, so the
  information is recoverable from the history, but the top-level summary
  drops it.
- Repro: Drive an ALM run with `max_outer_iterations=1`,
  `max_subproblem_continuations=0`, a feasible-but-stationarity-failing
  iterate. The single outer iteration's single continuation hits
  `max_continuations_hit=True` and routes through the M3.a penalty arm.
  `result.termination_reason == "max_outer"` (not
  `"max_outer_after_subproblem_limit_penalty_increase"`).
- Suggested fix: Add a case to L1779-L1790:
  ```python
  if latest_action == "subproblem_limit_penalty_increase":
      return "max_outer_after_subproblem_limit_penalty_increase"
  ```
  Pair it with the existing `subproblem_limit` case at L1784-L1785 since
  semantically both arose from the same "ran out of subproblem budget"
  cause. Update any consumer-side parsers.

### F6: `validate_initial_multipliers` does not validate against `multiplier_max` [LOW]

- File: `examples/single_stage_optimization/alm_utils.py:2272-2295`
- Code:
  ```python
  def validate_initial_multipliers(multipliers, n_constraints: int) -> np.ndarray:
      arr = np.asarray(multipliers, dtype=float)
      if arr.shape != (n_constraints,):
          raise ValueError(...)
      if not np.isfinite(arr).all():
          raise ValueError(...)
      if (arr < 0.0).any():
          raise ValueError(...)
      return arr.copy()
  ```
- Bug: M8 added shape, finiteness, and non-negativity checks. But it does
  not check against `settings.multiplier_max`. Caller can pass
  `initial_multipliers=[1e10]` with `settings.multiplier_max=1e6`; the
  vector is silently accepted, and on the first projection at L1660-L1674
  in `_project_nonnegative_multipliers_with_diagnostics` the entries are
  clipped to the cap and `cap_binding=True` is set. This makes the very
  first dual update appear to be at the cap.
- Why: The original audit (M8) framed the validation as boundary checks
  for the contract. The cap is enforced silently downstream, but a caller
  passing an out-of-cap initial multiplier is signalling that the cap may
  be wrong for the run, and a clean upfront error is more appropriate.
- Impact: Hard to reach in practice (resume JSON typically writes
  in-cap values). When reachable, the run starts in `last_cap_binding_active=True`
  state, which now (post-M4 fix) blocks both post-inner converged paths
  (L4264 and L4316) until a future un-clamped dual update. The skipped-inner
  shortcut at L3968 still fires in this state — see F1.
- Repro: Call `minimize_alm(...,
  settings=ALMSettings(multiplier_max=1.0e6),
  initial_multipliers=np.array([1.0e10]))`. The validator returns the
  caller's vector unchanged; the first dual update clamps to 1e6 and sets
  `cap_binding=True`.
- Suggested fix: In `validate_initial_multipliers`, accept
  `multiplier_max: float | None` (passed in from `_normalize_alm_run_inputs`
  using `settings.multiplier_max`) and raise if any entry exceeds the cap.
  Or, less strict: warn but accept. The first option is more in line with
  the project posture (no silent recovery).

### F7: `_normalize_alm_run_inputs` does not validate `x0` finiteness [LOW]

- File: `examples/single_stage_optimization/alm_utils.py:2318`
- Code:
  ```python
  x = np.asarray(x0, dtype=float).copy()
  ```
- Bug: No NaN/Inf check. If a caller passes `x0=[NaN]`, the value
  propagates into `evaluate_problem`, which presumably returns NaN
  evaluations, which are caught by `_require_finite_evaluation` at L3934
  with a less actionable error.
- Why: M8 added validation for `initial_multipliers`, `initial_penalty`,
  and the asymmetric incumbent-hook pair, but not for `x0`.
- Impact: Confusing error message ("ALM outer iterate evaluation produced
  non-finite ALM data: total, grad") when the actual fault is a bad
  caller-supplied `x0`.
- Repro: `minimize_alm(np.array([np.nan]), ...)` — the run fails inside
  the first inner solve with a non-finite-evaluation error pointing at the
  evaluator rather than the input.
- Suggested fix: After L2318, add:
  ```python
  if not np.isfinite(x).all():
      bad = np.where(~np.isfinite(x))[0].tolist()
      raise ValueError(f"ALM x0 non-finite at indices {bad}")
  ```

## Verification of v1 Fixes

| Fix | Status | Notes |
|---|---|---|
| M2 (subproblem_continue terminator label) | confirmed (correct), test gap | Both call sites correctly thread `is_final_outer`. Helper at L3610-L3643 sets `outer_termination="max_outer"` when `is_final_outer=True`. Signal-mismatch path is tested. Feasible-update path is **not tested** — see F4. |
| M3.a (max_subproblem_continuations exhaustion routes to penalty arm) | confirmed | At L4500-L4527 the new path: sets `subproblem_limit_reason="max_subproblem_continuations"`, then calls `_emit_alm_penalty_increase_arm(..., action="subproblem_limit_penalty_increase")`. Plateau-stall preserved at L4462-L4499. Action-string consumer-side: see F5 (missing case in `_termination_reason_from_history`). |
| M3.b (gtol does not ratchet looser across outer iters) | confirmed | The L4001 line was removed; `state.inner_options` is the user's anchor and `_build_inner_options(state.inner_options, ...)` re-derives the staged value fresh per call. Pinned by `test_minimize_alm_preserves_user_gtol_anchor_across_outer_iterations` at `tests/geo/test_alm_utils.py:4251`. |
| M4 (cap-binding gates both success arms) | **incomplete** | The L4273 (post-inner converged) and L4320 (constraints-inactive) gates correctly check `not run_state.last_cap_binding_active`. The pre-inner skipped-inner shortcut at L3968 — also a success arm — is **not gated**. See F1. |
| M5 (post-inner routing uses effective_feasibility_tol) | **incomplete** | Routing/stationarity at L4098-L4112 correctly use `effective_feasibility_tol`. The immediately-following `_attach_alm_history_diagnostics` call at L4250 still uses the unclamped `state.update_feasibility_tol`. See F2. |
| M6 (nnls RuntimeError catch) | confirmed | L2175-L2188 catches only `RuntimeError`, passes explicit `maxiter=10*n`, returns `None` on failure. Shape errors propagate as `ValueError`. |
| M8 (validate_initial_multipliers) | confirmed (with gap) | Shape, finite, non-negative checks at L2272-L2295. Wired into `_normalize_alm_run_inputs` at L2319-L2323. **Does not check against `settings.multiplier_max`** — see F6. **Does not check `x0` finiteness** — see F7. |
| M9 (KKT diagnostic uses base_grad) | **incomplete** | Fixed in `_stationarity_metrics` at L2228-L2231 (correct: uses `evaluation.get("base_grad", metric_grad)`). The sibling helper `_surrogate_kkt_stationarity_norm` at L835-L852 still uses `metric_grad` directly. See F3. |
| L1 (always shallow-copy in `_attach_alm_constraint_metadata`) | confirmed | L2360-L2365: `annotated = dict(evaluation)` then conditional augmentation. Both lanes return an owned dict. |
| L2 (history-list copy in callback) | confirmed | `_emit_alm_history_snapshot` at L2535-L2540 passes `list(history)` (shallow copy). The `latest_entry` is a deep snapshot via `_snapshot_history_entry`. Pinned at `tests/geo/test_alm_utils.py:2680`. |
| L3, L4 (evaluation dict cloning at boundaries) | confirmed | `_clone_evaluation_dict` at L1306-L1321; used in `_ALMInnerAttemptEvaluator.fun` at L239 and in `_sanitize_nonfinite_inner_evaluation` at L1334. `_build_augmented_evaluation` at L575-L603 calls `.copy()` on every stored ndarray. |
| L5 (ALMSettings.__post_init__ validation) | confirmed | L37-L77 mirrors `validate_alm_cli_args`. Accepts `max_subproblem_continuations=0` per docstring. |
| H1 (require_positive_alm_threshold) | confirmed (out of audit scope) | Helper at L373-L386. Driver-side wiring is in `single_stage_banana_example.py` and `hardware_constraint_schema.py`, outside the algorithm-correctness audit. |

## Confirmed-Correct Items

The following items from the audit checklist were re-verified against the
current `bf936a0a4` tree and are correct:

- **Outer-loop control flow** (`_minimize_alm_impl` L4683-L4813,
  `_run_alm_outer_iteration` L4585-L4680). The three `_ALMOuterDecision`
  arms are exhaustively dispatched at L4773-L4780. The `final_eval is
  None or last_result is None` defensive raise at L4782-L4785 catches the
  impossible "no continuation step ran" case (since
  `range(max_subproblem_continuations + 1)` always has ≥1 iteration after
  L5 validation accepts 0).
- **Continuation step state ownership** (`_run_alm_continuation_step`
  L3886-L4557). All 13 returns flow through `_finalize_continuation_step`
  at L3412-L3433, which packages the mutable `_ContinuationStepState`
  carrier into the frozen `_ALMContinuationStepResult`. Every field is
  sourced from the carrier; no field is silently default-initialized.
- **`feasible_stall_count` resetting**. Reset to 0 at the start of each
  outer iteration (L4614). Reset to 0 inside the dual-update arm (L4412)
  and signal-mismatch CONTINUE arm (L4401). Inside the penalty-increase
  helper, `_apply_continuation_penalty_increase` resets it to 0 via
  `ALMPenaltyIncreaseResult.feasible_stall_count=0` at L2724.
- **Retry arm precedence is well-defined**. ARMs (using audit numbering):
  B (post-inner converged) > C (forced_infeasible) > D (constraints_inactive)
  > E (signal_mismatch+hard_feasible_strict) > F (dual_update) > G
  (hard_feasible_for_update without dual gate) > H (default penalty_increase).
  Mutual exclusion is enforced through gate predicates: ARM B requires
  `not constraints_inactive_candidate and not signal_mismatch_active`;
  ARM D requires `constraints_inactive_candidate`; ARM E requires
  `signal_mismatch_active and hard_feasible_strict`; etc.
- **`inner_options` ownership**. The user's `inner_options` dict is never
  mutated (every `_build_inner_options` call at L1733 creates a fresh
  shallow copy). M3.b removed the persistence assignment at the previous
  L4001, so `state.inner_options` stays as the user's anchor across
  outer iterations.
- **Penalty cap termination is clean**. `_handle_alm_penalty_cap_termination`
  at L3123-L3163 returns a single failure result with
  `termination_reason="penalty_cap_reached"`, building from the
  post-bump evaluation. No infinite loop is possible because
  `_next_penalty` saturates at `penalty_max` and `cap_hit=True` is sticky.
- **Skipped-inner shortcut prerequisites** (L3968-L3973) **except for the
  M4 cap gate**. Excludes the `constraints_inactive_candidate` and
  `signal_mismatch_active` arms correctly. Does NOT exclude the cap-binding
  case (see F1).
- **History append-only invariant**. Entries are appended at L4234 (and
  L3974 for the skipped-inner path). After append, the entry is mutated
  in-place by `_attach_alm_history_diagnostics`,
  `_refresh_alm_history_for_penalty_update`, and the arm-specific action
  annotation. The snapshot delivered to `history_callback` is deep-copied
  via `_snapshot_history_entry` at L1354-L1355, so external observers see
  a consistent point-in-time view (per L2 fix). The list itself is
  shallow-copied per L2 fix at L2535-L2540.
- **`_made_meaningful_inner_progress`** (L1416-L1458). Returns True if any
  of: moved x, improved objective (with relative tolerance), improved
  stationarity, or improved feasibility. Specifically handles the case
  where inner iterations improve constraint feasibility but worsen
  objective: `improved_feasibility=True` makes the function return True
  even if `improved_objective=False`.
- **Off-by-one in the outer loop**. `for outer_iteration in range(1,
  settings.max_outer_iterations + 1)` runs `max_outer_iterations`
  iterations. `is_final_outer = (outer_iteration ==
  settings.max_outer_iterations)` correctly identifies the last iteration.
- **Off-by-one in the continuation loop**. `for continuation_iteration in
  range(settings.max_subproblem_continuations + 1)` runs
  `max_subproblem_continuations + 1` iterations. With the M3.a fix, the
  last iteration's `max_continuations_hit = (continuation_iteration ==
  settings.max_subproblem_continuations)` is True and routes through
  penalty escalation rather than naturally exiting via CONTINUE.
- **`_EarlyStopInnerSolve` handling**. Catch at L3273-L3281 in
  `_run_alm_inner_attempts`. Sets `result.success=True`, `nit=1`. Then
  `_candidate_is_acceptable` at L3289 evaluates the early-stop iterate.
  `_classify_infeasible_inner_stall` returns False because the candidate
  satisfies `<= effective_feasibility_tol`. The candidate is accepted via
  the `acceptable and not infeasible_inner_stall` branch.
- **Dual-update arm semantics** (L4411-L4452). Updates multipliers and
  tolerances via `_handle_alm_dual_update_transition`. The
  `multiplier_cap_binding` flag from the projection is propagated to
  `run_state.last_cap_binding_active` (L4434) and to
  `run_state.cap_binding_detected` (L4436). BREAK_OUTER fires; the next
  outer iteration starts with the new multipliers but the same `x`.
- **Trust-radius growth** (L3589-L3607). Centralized in
  `_grow_continuation_trust_radius`. Only fires when
  `run_state.trust_radius is not None`. Bounded below by
  `settings.trust_radius_min`.

## Verdict

The v1 fix commit `bf936a0a4` is a substantial improvement over the prior
state and lands the bulk of the M2/M3.a/M3.b/M4/M5/M6/M8/M9 audit items.
Three control-flow gaps survive:

1. **F1 (HIGH)**: M4 cap-binding gate is missing on the pre-inner
   skipped-inner success arm at L3968. A run that ends an outer iteration
   with a clamped dual update can mislabel the next outer iteration's
   first-continuation iterate as `success=True / "converged"` despite the
   M4 contract.
2. **F2 (MEDIUM)**: M5's effective_feasibility_tol consistency was applied
   to routing/stationarity but not to the post-inner
   `_attach_alm_history_diagnostics` call at L4250.
3. **F3 (MEDIUM)**: M9's base_grad fix was applied to
   `_stationarity_metrics` but not to the sibling helper
   `_surrogate_kkt_stationarity_norm` at L835-L852.

Lower-severity items: F4 (M2 feasible-update retry call site has no test),
F5 (`_termination_reason_from_history` lacks the
`subproblem_limit_penalty_increase` case), F6 (initial-multipliers
validator does not check `multiplier_max`), F7 (no `x0` finiteness check).

After F1-F3 are landed and pinned by tests, the algorithm/control-flow
contract for the ALM driver is consistent with the audit. F4-F7 are
low-priority diagnostic and validation polish.
