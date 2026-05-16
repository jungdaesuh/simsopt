# ALM Math Audit v2 — 2026-05-08

## Summary

Reviewed the post-`bf936a0a4` ALM driver in `examples/single_stage_optimization/alm_utils.py` (4847 LOC, HEAD `e7b836464`). The core augmented-Lagrangian math (objective, gradient, dual update, projection, penalty schedule, gtol-anchor preservation) is all correct and matches Bertsekas (1982, eq. 4.69) under the project's `c ≤ 0` feasibility convention. The recent v1 fixes mostly land cleanly: M3.b (gtol-ratchet removal), M5 (effective gate consistency), M6 (`nnls` `RuntimeError` only), and M9 (KKT residual uses `base_grad`) all check out at the cited sites.

**Findings**: 1 HIGH, 1 MEDIUM, 1 LOW.

- **F1 (HIGH)**: M4's cap-active gate is missing from the third success path — the start-of-outer skipped-inner shortcut at `alm_utils.py:3968-3973`. A run that enters a fresh outer iteration with cap-bound multipliers from the previous dual update can be silently labelled `converged` despite the cap distorting the Lagrangian. M4 only gated the post-inner-solve `converged` arm and the `constraints_inactive_converged` arm.
- **F2 (MEDIUM)**: M9 fix was applied to `_kkt_stationarity_norm` consumer in `_stationarity_metrics` (`alm_utils.py:2229`), but the parallel diagnostic `_surrogate_kkt_stationarity_norm` (`alm_utils.py:840-841`) **still feeds the augmented gradient** instead of `base_grad`. Same defect class; same fix not applied. The reported `final_surrogate_kkt_stationarity_norm` collapses to ≈0 once the inner solve converges and is meaningless as a multiplier-quality indicator.
- **F3 (LOW)**: `_kkt_stationarity_norm` declares `feasibility_gate` and `feasibility_values` parameters that it never uses (`alm_utils.py:2144-2146` vs body L2155-2167). Dead-arg API noise; no math impact.

The hybrid surrogate-vs-hard signal split documented in `docs/alm_hybrid_signal_contract_2026-05-08.md` is honored throughout the verified call paths: inner objective at `stage2_objectives.py:1951` consumes surrogate; dual update at `alm_utils.py:3102` consumes `preferred_dual_update_values` (which `_extract_stage2_constraint_signal_state` ties to `hard_dual_update_values`); the converged gate carries the `not signal_mismatch_active` guard at `alm_utils.py:4268`. No new leakage.

## Methodology

Files read line-by-line against the v1 audit (`.alm_audit/math_review.md`) and the v1 fix plan (`.alm_audit/FIX_PLAN.md`):

- `examples/single_stage_optimization/alm_utils.py` — full file (4847 lines), with focused trace of:
  - L435-487 `augmented_inequality_objective`, L555-642 `_build_augmented_evaluation` / `_augmented_terms` / `_positive_shift_and_augmented_terms`
  - L835-852 `_surrogate_kkt_stationarity_norm`
  - L855-873 `_lbfgsb_projected_gradient_max_norm`
  - L1627-1675 multiplier projection helpers
  - L1677-1725 penalty / tolerance schedule helpers
  - L1727-1762 `_build_inner_options` (gtol staging)
  - L1947-2137 constraint-state extraction and routing
  - L2140-2244 `_kkt_stationarity_norm` + `_stationarity_metrics`
  - L2272-2349 multiplier validation and run-input normalization
  - L2543-2730 penalty-state evaluation and penalty-increase
  - L3090-3120 `_handle_alm_dual_update_transition`
  - L3211-3383 `_run_alm_inner_attempts`
  - L3886-4557 `_run_alm_continuation_step` (every branch)
  - L4683-4813 `_minimize_alm_impl`
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1655-2050` — signal-construction call site for `augmented_inequality_objective`
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:325-580, 1060-1140` — the parallel single-stage call site, including `evaluate_base_objective` for both `thresholded_physics` and `weighted_sum` modes
- `docs/alm_hybrid_signal_contract_2026-05-08.md` — the contract under test
- Commit `bf936a0a4` diff for M2/M3/M4/M5/M6/M9 to confirm the *applied* code matches the *promised* fix

Cross-checks performed:

1. Algebraic verification that `_augmented_terms` `0.5 * (s - λ)(s + λ) / ρ` ≡ Bertsekas `(1/(2ρ))[max(0, λ + ρc)]² − λ²/(2ρ)`.
2. Trace of the three success-emit paths (skipped-inner shortcut at L3968, post-inner converged at L4264, constraints-inactive converged at L4316) against the M4 gate set.
3. Trace of the 8 `_run_alm_continuation_step` exit branches to confirm `signal_mismatch_active`, `last_cap_binding_active`, `forced_infeasible_penalty_cycle` are never bypassed when they should fire.
4. Diff-grep for `metric_grad` vs `base_grad` consumers to confirm M9 was applied uniformly.
5. Sign-convention spot-check of `upper_bound_residual` against `smooth_min_distance_signed_constraint` (`stage2_objectives.py:1524`) and `smooth_max_curvature_signed_constraint` (`stage2_objectives.py:1452`) — both produce `c ≤ 0 means feasible` correctly.

## Findings

### F1: M4 cap-active gate missing from start-of-outer skipped-inner shortcut [HIGH]

- **File**: `examples/single_stage_optimization/alm_utils.py:3968-3973`
- **Code**:
  ```python
  if (
      current_max_feasibility_violation <= settings.feasibility_tol
      and current_stationarity_norm <= settings.stationarity_tol
      and not current_constraints_inactive_candidate
      and not current_signal_mismatch_active
  ):
      ...
      return _finalize_continuation_step(
          state,
          _ALMContinuationDecision.RETURN,
          _build_alm_converged_step_result(
              ...
              termination_reason="converged",
              ...
          ),
      )
  ```
- **Bug**: This is the third path that emits `termination_reason="converged"` (with `success=True`). The other two converged-emitters were patched by M4 to add `and not run_state.last_cap_binding_active`:
  - `alm_utils.py:4264-4274` (post-inner converged): `... and not run_state.last_cap_binding_active`
  - `alm_utils.py:4316-4321` (constraints-inactive converged): `... and not run_state.last_cap_binding_active`
  This skipped-inner shortcut at L3968 was missed. The flag `run_state.last_cap_binding_active` is non-sticky and updated to `bool(dual_update.multiplier_cap_binding)` at L4434 inside the dual-update arm; it carries forward to the next outer iteration's start-of-outer evaluation unchanged through penalty-increase / continuation arms.
- **Why it's wrong**: The M4 invariant (per its own comment at L4269-4272: "cap-binding multipliers mean the dual update was clamped; KKT residual is being held artificially small by the cap, not by genuine convergence") is the contract: a result that satisfies the converged tolerances **only because the multiplier was clamped at `multiplier_max`** does not satisfy true KKT. The standard Bertsekas / Conn-Gould-Toint termination theorems require the multiplier sequence to converge from the unconstrained dual update; clamping discards gradient information from the dual problem. With this skipped-inner shortcut, the following sequence triggers a false `converged` label:
  1. Outer K: inner converges feasible/stationary, dual update clamps → `last_cap_binding_active = True`, BREAK_OUTER.
  2. Outer K+1: start-of-outer evaluation re-evaluates at the same x with the clamped multipliers. If the iterate still meets feasibility/stationarity gates (extremely likely — x didn't move, multipliers only scaled by clamping at the cap), the shortcut at L3968 fires.
  3. The shortcut emits `termination_reason="converged"` with `result.success=True`. The cap-binding flag survives in `result.multiplier_cap_binding` (sticky) and `result.multiplier_cap_binding_indices`, but `result.success` is the user-facing boolean and it lies.
- **Impact**: Silent wrong-answer at the result-success boundary. Same severity ladder as the original M4 finding: the converged label asserts KKT to within `stationarity_tol`, but the stationarity norm at this iterate is held small by the clamp, not by `λ` actually equalling the optimal multiplier. Downstream consumers (`run_single_stage_*` campaign drivers, autoresearch promotion, frontier reporting) gate on `result.success` and accept the design as a feasible KKT point. Mathematically this is a non-KKT point of the original problem (the true λ exceeds `multiplier_max`).
- **Suggested fix**: Add `and not run_state.last_cap_binding_active` to the gate at L3968-3973, mirroring the post-inner gate at L4273:
  ```python
  if (
      current_max_feasibility_violation <= settings.feasibility_tol
      and current_stationarity_norm <= settings.stationarity_tol
      and not current_constraints_inactive_candidate
      and not current_signal_mismatch_active
      and not run_state.last_cap_binding_active
  ):
      ...
  ```
  Add a regression test mirroring the missing M4 tests from the v1 fix plan (`.alm_audit/FIX_PLAN.md` M4 §): `test_current_cap_active_blocks_skipped_inner_shortcut_label` driving `outer K` → cap-binding dual update → `outer K+1` → start-of-outer feasible/stationary → assert `result.success is False`. None of the four planned M4 tests in the v1 plan actually exist in `tests/geo/test_alm_utils.py` (verified via `grep -n "current_cap_active\|cap_active_blocks" tests/geo/test_alm_utils.py` — no hits).

### F2: M9 fix not applied to `_surrogate_kkt_stationarity_norm` diagnostic [MEDIUM]

- **File**: `examples/single_stage_optimization/alm_utils.py:835-852`
- **Code**:
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
          metric_grad,
          evaluation.get("constraint_grads"),
          ...
      )
  ```
- **Bug**: The function feeds `metric_grad` (the augmented Lagrangian gradient `∇L_A = ∇f + Σ s_i ∇c_i`) to `_kkt_stationarity_norm` and asks `nnls` to minimize `||metric_grad + A_active μ||` over `μ ≥ 0`. Once the inner solve converges, `metric_grad ≈ 0`, so the residual collapses to ≈0 with `μ_opt = 0`. The reported value is therefore ≈0 regardless of multiplier quality.
- **Why it's wrong**: The KKT stationarity residual is `min_{λ ≥ 0} ‖∇f + A_active λ‖` (Bertsekas eq. 4.5; Nocedal-Wright Thm. 12.1). The input must be `∇f` (the *base* objective gradient), not `∇L_A`. This is exactly the same bug that M9 fixed for `_stationarity_metrics` at `alm_utils.py:2228-2231`. The M9 fix is explicitly NOT applied here despite the function name promising a KKT diagnostic.
- **Impact**: The fields `surrogate_kkt_stationarity_norm` (in every history entry, `alm_utils.py:969`, `alm_utils.py:1099`) and `final_surrogate_kkt_stationarity_norm` (in `result.alm_summary` at `alm_utils.py:2898-2900`) are systematically misleading: they always converge to ≈0 once the inner solve converges, regardless of whether the surrogate-side multipliers are anywhere near a true KKT solution of the surrogate problem. Operators reading the saved results to gauge "how close is the surrogate side to true KKT" are getting 0.0 always — useless. This is **diagnostic-only** (the field is never consumed by any termination gate — verified by `grep "surrogate_kkt_stationarity_norm" alm_utils.py` shows reporting-only consumption), so it does NOT cause silent wrong answers in `result.success`. That is why the severity is MEDIUM, not HIGH.
- **Suggested fix**: Mirror the M9 pattern in `_stationarity_metrics` (`alm_utils.py:2228-2231`):
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
      kkt_grad = np.asarray(
          evaluation.get("base_grad", metric_grad),
          dtype=float,
      )
      surrogate_values = routing_state.signal_state.surrogate_signed_constraint_values
      return _kkt_stationarity_norm(
          kkt_grad,
          evaluation.get("constraint_grads"),
          surrogate_values,
          np.maximum(surrogate_values, 0.0),
          _constraint_activity_tolerances(evaluation, surrogate_values),
          feasibility_gate,
      )
  ```
  Add a regression test analogous to the (currently-missing) M9 test described in the v1 fix plan: pin that the surrogate KKT residual is non-zero at an iterate where `∇f + Σλ∇c ≠ 0` despite the inner solve having driven `∇L_A ≈ 0`.

### F3: `_kkt_stationarity_norm` has dead `feasibility_gate` and `feasibility_values` parameters [LOW]

- **File**: `examples/single_stage_optimization/alm_utils.py:2140-2190`
- **Code**:
  ```python
  def _kkt_stationarity_norm(
      total_grad,
      constraint_grads,
      constraint_values: np.ndarray,
      feasibility_values: np.ndarray,        # never read
      activity_tolerances: np.ndarray,
      feasibility_gate: float,                # never read
  ) -> float | None:
      ...
      for constraint_grad, constraint_value, _feasibility_value, activity_tolerance in zip(
          constraint_grads,
          constraint_values,
          feasibility_values,
          activity_tolerances,
      ):
          if float(constraint_value) < -float(activity_tolerance):
              continue
          active_constraint_grads.append(...)
      ...
  ```
- **Bug**: `feasibility_gate` is in the signature but never read inside the function body (only the activity-tolerance check at L2162 affects which constraints are included). `feasibility_values` is unpacked but bound to a `_feasibility_value` discard variable in the loop. Both are passed at every call site (`_stationarity_metrics` L2237, `_surrogate_kkt_stationarity_norm` L851).
- **Why it's wrong**: Not wrong mathematically. The KKT active-set is correctly identified by `constraint_value >= -activity_tolerance` (the standard Bertsekas-style activity band). But the function lies about what it consumes; future maintainers may add a real feasibility-gate filter without realizing the gate is silently a no-op today, and the dead-arg increases the call-site burden.
- **Impact**: Cosmetic / API hygiene. No math output change.
- **Suggested fix**: Drop the two dead parameters:
  ```python
  def _kkt_stationarity_norm(
      total_grad,
      constraint_grads,
      constraint_values: np.ndarray,
      activity_tolerances: np.ndarray,
  ) -> float | None:
      ...
      for constraint_grad, constraint_value, activity_tolerance in zip(
          constraint_grads,
          constraint_values,
          activity_tolerances,
      ):
          if float(constraint_value) < -float(activity_tolerance):
              continue
          active_constraint_grads.append(...)
      ...
  ```
  Update both call sites to drop the corresponding arguments. No behavior change — the existing values were never read.

## Confirmed-Correct Items

The following items from the v1 audit and the `bf936a0a4` fix set were re-verified line-by-line and are mathematically correct in the current code:

1. **Augmented-Lagrangian objective formula** (`alm_utils.py:606-616`, `alm_utils.py:447-487`). `_augmented_terms` returns `0.5 * (s − λ) * (s + λ) / ρ` which expands to `(s² − λ²)/(2ρ)` — exactly Bertsekas (1982, eq. 4.69). Both branches `λ + ρc > 0` and `λ + ρc ≤ 0` reduce to the textbook expressions `λc + (ρ/2)c²` and `−λ²/(2ρ)` respectively.

2. **Augmented-Lagrangian gradient** (`alm_utils.py:466-473`). `total_grad = ∇f + Σ max(0, λ + ρc_i) ∇c_i` is the correct `∂ψ_i/∂c_i = max(0, λ + ρc_i)` chain rule.

3. **Multiplier projection** (`alm_utils.py:1642-1675`). `λ_{k+1} = min(cap, max(0, λ_k + ρ_k c_k))` matches the canonical safeguarded inequality update.

4. **Penalty schedule monotonicity** (`alm_utils.py:1708-1724`, `alm_utils.py:2684-2691`). `_next_penalty` enforces `μ_{k+1} = β μ_k` with `β > 1` (CLI + `__post_init__` both validated at L399, L51), capped at `penalty_max`. The post-penalty-increase `next_feasibility_tol = min(prev, schedule_tol(next_penalty))` is monotonically tightening (see `_penalty_schedule_tolerance`).

5. **Dual-update tolerance tightening** (`alm_utils.py:3110-3117`). After a dual update, `update_feasibility_tol /= penalty_scale` floored at `settings.feasibility_tol`, same for stationarity. Monotonic.

6. **Outer convergence test (post-inner arm)** (`alm_utils.py:4264-4274`). `max_feasibility_violation ≤ feasibility_tol` AND `stationarity_norm ≤ stationarity_tol` AND `not constraints_inactive_candidate` AND `not signal_mismatch_active` AND `not run_state.last_cap_binding_active` — all four required guards in place.

7. **Constraints-inactive converged test** (`alm_utils.py:4316-4321`). Same M4 cap-active gate present here. ✓

8. **M4 non-sticky cap-binding update** (`alm_utils.py:4434`). `run_state.last_cap_binding_active = bool(dual_update.multiplier_cap_binding)` — unconditional write (resets to False on a non-clamping dual update).

9. **Sticky `cap_binding_detected` diagnostic preserved** (`alm_utils.py:4435-4439`). Set True only when current dual update clamped; only used in `result.multiplier_cap_binding` (`alm_utils.py:2918`). Diagnostic-only, not gate-active. Matches the M4 contract.

10. **M3.b gtol-anchor preservation** (`alm_utils.py:4163-4174`). The comment block makes the contract explicit; `state.inner_options` is never written. Each call to `_build_inner_options(state.inner_options, update_stationarity_tol, profile=...)` re-derives the staged `gtol` from the user's untouched base. Verified by the regression test at `tests/geo/test_alm_utils.py:4251-4302`.

11. **M5 effective-gate consistency** (`alm_utils.py:4095-4112`). Post-inner `_constraint_routing_state` and `_stationarity_metrics` both consume `effective_feasibility_tol` (the clamped value computed at L3935-3937), matching the pre-inner pass at L3944-3958.

12. **M6 nnls failure isolation** (`alm_utils.py:2175-2188`). `try / except RuntimeError` only — `ValueError` (shape mismatch) propagates. Explicit `maxiter=10 * active_matrix.shape[1]`. The `RuntimeWarning` keeps the failure operator-visible.

13. **M9 KKT diagnostic uses `base_grad`** in `_stationarity_metrics` (`alm_utils.py:2228-2231`). Correctly falls back to `metric_grad` when `evaluation["base_grad"]` is absent, but every Stage-2 / single-stage path verified to set `base_grad` (L578 in `_build_augmented_evaluation` plus the `evaluation.update(alm_eval)` flow in `single_stage_objectives.py:1088` and `stage2_objectives.py:1959`).

14. **Hybrid signal contract honored** (`alm_utils.py:1979-2037`, `:3090-3120`):
    - Inner objective is fed surrogate (`stage2_objectives.py:1951-1958` passes `normalized_surrogate_signed_constraint_values`).
    - Dual update uses `routing_state.signal_state.preferred_dual_update_values` which is `evaluation["hard_dual_update_values"]` when explicit Stage-2 signals exist (`alm_utils.py:2014`).
    - `signal_mismatch_active` is required-False in both converged gates (`alm_utils.py:4268`, `:4321`). Constraint forbidden by `docs/alm_hybrid_signal_contract_2026-05-08.md` is enforced.

15. **`hard_dual_update_values` strict-extract** (`alm_utils.py:1998-2006`). `KeyError` raised if `explicit_stage2_signals` and any of the four required fields missing. Matches the contract's "must not be loosened" guarantee.

16. **Penalty-cap termination flow** (`alm_utils.py:3123-3163`, `:2680-2730`). On `_next_penalty` cap-hit, `_apply_alm_penalty_increase` returns `cap_hit=True` which `_apply_continuation_penalty_increase` (L3542-3556) routes through `_handle_alm_penalty_cap_termination` (`termination_reason="penalty_cap_reached"`). Clean exit.

17. **L-BFGS-B projected gradient diagnostic** (`alm_utils.py:855-873`). `at_lower_bound and grad > 0` → zero out (descent direction would push x below bound); `at_upper_bound and grad < 0` → zero out. Standard projected-gradient max-norm.

18. **Sign-convention consistency** (`stage2_objectives.py:1524`, `:1452`). Distance constraints: `signed_value = minimum_distance − smooth_min`, so feasible iff `signed ≤ 0`. Curvature: `signed_value = smooth_max − threshold`, feasible iff `signed ≤ 0`. All match `c ≤ 0` convention used in `alm_utils.py:447`.

19. **`validate_initial_multipliers` driver-boundary check** (`alm_utils.py:2272-2295`). Shape, finiteness, non-negativity all enforced; `arr.copy()` ensures owned return. Routed from `_normalize_alm_run_inputs:2319-2323`.

20. **`ALMSettings.__post_init__` programmatic validation** (`alm_utils.py:37-77`). All numeric guardrails from `validate_alm_cli_args` mirrored — programmatic construction can no longer bypass.

## Verdict

The ALM driver is mathematically equivalent to Bertsekas (1982) and Conn-Gould-Toint LANCELOT (Algorithm 4.1 of the safeguarded multiplier method) under the hybrid surrogate-vs-hard signal contract documented at `docs/alm_hybrid_signal_contract_2026-05-08.md`. The classical convergence theorems are forfeited under sustained `signal_mismatch_active` (per the contract); engineering safeguards (mismatch-detection + converged-gate guard) prevent false `success` labelling under that regime, which we re-verified.

**One real bug remains** (F1, HIGH): the M4 cap-active gate is missing from the start-of-outer skipped-inner shortcut at `alm_utils.py:3968-3973`. This is the same severity class as the original M4 finding and lets a clamped-multiplier iterate label `result.success=True`. The fix is one line and a regression test.

**One real diagnostic-only bug remains** (F2, MEDIUM): `_surrogate_kkt_stationarity_norm` at `alm_utils.py:835-852` was missed by the M9 sweep and still consumes `metric_grad` instead of `base_grad`. The reported `final_surrogate_kkt_stationarity_norm` is therefore ≈0 at every converged run regardless of multiplier quality. No effect on `result.success`, but a misleading diagnostic.

**One cosmetic finding** (F3, LOW): two dead parameters in `_kkt_stationarity_norm`. No math impact.

After F1 is fixed, the driver is bug-free at the math level: every documented invariant in the FIX_PLAN's Confirmed-Correct list and the v1 audit's What-I-Verified list still holds; M2/M3.a/M3.b/M4 (partial)/M5/M6/M8/M9 land at their cited line numbers and behave per their docstrings. The hybrid-signal contract test (`test_alm_terminates_deterministically_under_sustained_signal_mismatch`) and the gold-standard KKT integration test (`test_minimize_alm_converges_to_kkt_on_active_linear_inequality`, λ=0.5) are present and the math invariants they pin are honored by the implementation.
