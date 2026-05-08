# ALM Algorithm Review

Date: 2026-05-08
Branch: surrogate-confinement-v2
File audited: `examples/single_stage_optimization/alm_utils.py` (4637 LOC)

## Executive Summary

The ALM driver's outer/continuation/inner state machine is structurally sound:
arm dispatch is disjoint, dual updates fire only after inner-solve convergence,
the penalty-cap path terminates cleanly, and the recent decomposition (commits
`52ff64d7f`/`a31a3e9f4`/`80e518337`) preserves callback ordering and history-
schema invariants. Two real correctness gaps remain: (a) a narrow EXHAUST
edge case where a final-outer `subproblem_continue` step produces a misleading
`termination_reason`, and (b) inconsistent feasibility-gate plumbing between
`current_routing_state` (uses `effective_feasibility_tol`) and the post-inner
`routing_state` (uses `update_feasibility_tol`). Several smaller findings
relate to dead defensive code, redundant best-feasible overwrites, and
non-idiomatic mutate-in-list history bookkeeping that is functionally correct
but easy to misread.

## Findings

| Severity | File:Line | Symptom | Root Cause | Recommended Fix |
|---|---|---|---|---|
| HIGH | `alm_utils.py:4454-4470, 4577-4581, 3457-3484` | When `signal_mismatch_active` + `made_inner_progress` + `continuation_iteration == 0 == max_subproblem_continuations` on the final outer iteration, `_emit_alm_subproblem_continue` returns `CONTINUE_CONTINUATION` and the inner loop natural-exits via `EXHAUST`. The latest history entry is `action="subproblem_continue"` with no `outer_termination` annotation. `_termination_reason_from_history` falls through to `latest_action` → returns `"subproblem_continue"`, masking the real "max_outer" cause. | `_emit_alm_subproblem_continue` does not accept or honor `is_final_outer`, so it never tags `outer_termination = "max_outer"` on the terminal continuation. Every other BREAK_OUTER/RETURN emitter does this via `_annotate_break_outer_history` / explicit set at L4285. | Thread `is_final_outer` into `_emit_alm_subproblem_continue` and set `history_entry["outer_termination"] = "max_outer"` when both `is_final_outer` AND `continuation_iteration == settings.max_subproblem_continuations`. Add a unit test exercising `max_subproblem_continuations=0` + signal_mismatch on the final outer iteration that asserts `result.termination_reason.startswith("max_outer")`. |
| MEDIUM | `alm_utils.py:3776-3799 vs 3936-3950` | `current_routing_state` is built with `effective_feasibility_tol` (clamped to `[settings.feasibility_tol, relaxed_feasibility_gate_cap]`); the post-inner `routing_state` is built with `state.update_feasibility_tol` (unbounded above). Their `hard_activity_mask`, `surrogate_activity_mask`, and `signal_mismatch_active` flags can therefore disagree when `update_feasibility_tol > relaxed_feasibility_gate_cap` (i.e., on early ALM iterations when penalty is still small). | Two different gate values flow into `_constraint_routing_state` at the two call sites. The decision to feed `update_feasibility_tol` to the post-inner routing was not paired with a matching change to `current_routing_state`, which still uses the bounded gate. | Pick one gate per audit pass. The natural choice is `effective_feasibility_tol` everywhere — it is exactly the value used by `hard_feasible_for_update` at L4081 and matches the pre-existing `current_routing_state` site. Replace `state.update_feasibility_tol` with `effective_feasibility_tol` at L3941 and L3949. Re-run `tests/geo/test_alm_utils.py` to confirm no characterization test regresses. |
| MEDIUM | `alm_utils.py:4271-4317, 4527-4570` | When `hit_stall_limit` fires due to `continuation_iteration == max_subproblem_continuations` (NOT plateau-stall), the arm returns `BREAK_OUTER` without bumping penalty or updating duals. The next outer iteration re-runs the same subproblem with identical (μ, λ). For runs where this state recurs, ALM burns up to `max_outer_iterations` of inner-solve work without changing μ or λ — bounded but wasteful, and obscures the real "ran out of subproblem budget" signal in the result. | The `subproblem_limit (max_subproblem_continuations)` arm intentionally does not escalate to penalty-increase; the design relies on the next outer iteration to "try again". But there is no schedule change between iterations. | If a subproblem hits `max_subproblem_continuations` while feasible-for-update without progress, escalate to `_emit_alm_penalty_increase_arm` (same as the default fall-through at L4332) so the next outer iteration sees a tighter penalty. Preserve plateau_stall RETURN as is. Add a regression test that drives a problem that hits `max_subproblem_continuations` repeatedly and asserts μ grows between outer iterations. |
| MEDIUM | `alm_utils.py:1628-1663, 4001` | `state.inner_options = inner_attempt.last_inner_options` carries the previous attempt's `gtol = max(base_gtol, staged_gtol)` into the next continuation. Because `_build_inner_options` then computes `staged_gtol` from the (possibly tighter) `update_stationarity_tol` and takes `max(base_gtol, staged_gtol)` again, `gtol` can never decrease across continuations — even as ALM tolerances tighten. The inner solve runs at a looser `gtol` than `update_stationarity_tol` would imply later in the run. | The "inner_options memo" pattern is preserving the largest seen gtol so the user's explicit setting wins over the stager. But propagating the *staged* gtol back into `base_gtol` makes the floor monotone non-decreasing. | Stop persisting `staged_gtol` into `last_inner_options`. Either (a) compute `staged_gtol` afresh per attempt from `update_stationarity_tol` and the user's original `inner_options["gtol"]`, or (b) memo only the user-facing keys (`maxiter`, `maxls`, `maxfun`) and recompute `gtol`/`ftol` per call. This is a performance/effectiveness fix, not a soundness fix. |
| MEDIUM | `alm_utils.py:2548, 4002-4078, 2370-2387` | History entries are appended at L4061 and then mutated in-place by `_attach_alm_history_diagnostics`, `_refresh_alm_history_for_penalty_update`, `_annotate_break_outer_history`, etc. The entry passed to `history_callback` is deep-copied, so external observers are safe; but the in-list dict's `action`/`penalty`/`stationarity_norm`/diagnostics-source are *overwritten* between the L4061 append and the eventual `_emit_alm_history_snapshot` at the arm boundary. This violates the auditor's "append-only, no in-place edit" expectation; subtle bugs can hide if a future arm forgets to overwrite a stale field. | The append-then-mutate pattern is intentional (single-entry-per-step, fields settle progressively). Each arm assumes it owns the entry through to its `_emit_alm_history_snapshot`. | (a) Document the append-then-mutate contract at the top of `_run_alm_continuation_step` and note it explicitly on `_append_alm_history_entry`'s docstring; (b) add a unit test that fakes a `history_callback` and asserts the snapshot dict reflects the post-arm state for each emitter (penalty_increase, dual_update, subproblem_continue, subproblem_limit, converged, stall_failure). |
| LOW | `alm_utils.py:3205-3213` | The defensive fallback `if accepted_result is None or accepted_eval is None or accepted_x is None:` is unreachable. The `for attempt_index in range(1, max_inner_attempts + 1)` loop always sets `accepted_*` before `break` on every code path (acceptable, infeasible_inner_stall, attempt_radius is None, exhausted, or trust_radius_min). `max_inner_attempts > 0` is enforced by `validate_alm_cli_args` so the loop always runs at least once. | Defensive programming carried over from earlier refactors. | Replace the block with an `assert accepted_result is not None and accepted_eval is not None and accepted_x is not None, "internal: every inner-attempt branch must set accepted_*"` and drop the `last_attempt_result is None` branch. |
| LOW | `alm_utils.py:3862-3873` | Inside the skipped-inner shortcut, `state.best_feasible` is overwritten unconditionally at L3862. Other sites (L3987-3998) gate the update on a strict objective improvement. If a previous outer iteration captured a feasible iterate with a strictly lower base objective, this overwrite breaks the "monotonic best-feasible-so-far" invariant. | The shortcut path was written assuming "if we are at full KKT convergence, this iterate IS the answer", which is a reasonable design choice but inconsistent with the strict `<` gating elsewhere. | Either (a) add the same `_incumbent_objective_value(current_eval) < _incumbent_objective_value(state.best_feasible.evaluation)` guard, or (b) document at the L3862 site that the shortcut returns RETURN immediately, so the subsequent `_restore_alm_best_feasible_on_failure` machinery is never consulted on this path; therefore the unconditional overwrite is observationally inert. Option (a) is safer. |
| LOW | `alm_utils.py:2167-2202` | `initial_multipliers` is accepted as-is with no validation. Negative multipliers, NaN multipliers, and shape mismatches against `len(constraint_names)` are silently propagated into the augmented Lagrangian. The first inner solve runs with whatever the caller passed; subsequent iterations get clamped by `_project_nonnegative_multipliers_with_diagnostics`. | The function asserts only on `penalty` finiteness/positivity. | Add: `if multipliers.shape != (len(constraint_names),): raise ValueError(...)`, `if not np.all(np.isfinite(multipliers)): raise ValueError(...)`, `if np.any(multipliers < 0.0): raise ValueError("inequality-ALM multipliers must be nonnegative")`. |
| LOW | `alm_utils.py:1296-1403` | `_acceptable_total_upper_bound` allows candidates whose augmented total is up to `_ACCEPTANCE_TOTAL_RTOL = 1e-3` (0.1%) larger than `current_total` (plus 1e-10 absolute). 0.1% relative slack is generous for a quasi-Newton step that should reduce the augmented objective. | Tolerance was chosen to absorb numerical noise / line-search artifacts. | Audit downstream: confirm whether 1e-3 is the *intended* tolerance (and document why) or whether it can be tightened to e.g. 1e-6 without breaking the existing characterization tests. No code change unless the answer is "tighten". |
| NIT | `alm_utils.py:2937-2967, 4238` | After a dual update, `state.multipliers` is the new λ but `state.final_multipliers`/`state.final_penalty` are still the pre-update values (set at L3968/3969). Both flow into the result carrier. If the run hits `max_outer` immediately after a dual update, the final result reports the *pre-update* λ as `multipliers` while the next-outer-iteration trajectory used the *post-update* λ. The two views are individually consistent (final_eval was computed at pre-update λ) but easy to misread. | Two parallel "current" and "final" multiplier slots in `_ContinuationStepState`/`_ALMOuterIterationResult`/`_ALMContinuationStepResult`. | Document the semantic split (`multipliers` = next-iteration starting point; `final_multipliers` = multipliers at the last fully-evaluated `final_eval`) at the dataclass definition sites at L3025-3055. |
| NIT | `alm_utils.py:4138-4180` | The `constraints_inactive_candidate` arm and the regular `converged` arm at L4091 both produce a `success=True` result; the only difference is `termination_reason` ("constraints_inactive_converged" vs "converged") and message. The convergence test at L4091 explicitly excludes constraints_inactive_candidate so the two arms are mutually exclusive. | Two parallel converged arms, the second carrying additional state. | Either fold both into a single `_emit_alm_converged_step` call site that picks the termination_reason via a small predicate, or document why they remain separate (probably for downstream telemetry to distinguish the two basin shapes). |
| NIT | `alm_utils.py:62-77, 1900-1938` | `preferred_dual_update_values` is selected based on whether the evaluation has explicit Stage-2 signal fields (`hard_*`, `surrogate_*`). If the evaluator forgets one signal field, `_extract_stage2_constraint_signal_state` raises `KeyError` from L1903-1907 — not the `ValueError` used elsewhere for ALM contract failures. | Inconsistent exception type for evaluator contract failures. | Either standardize on `ValueError` for evaluator contract failures (recommended) or document that signal-field absence is reported via `KeyError`. |

## Verified Correct

The following were verified against the audit checklist and current tree:

- **Outer-loop control flow (audit item 1):** `_minimize_alm_impl` (L4473-4603)
  routes through `_run_alm_outer_iteration` (L4375-4470) and handles the three
  `_ALMOuterDecision` arms exhaustively: `RETURN` returns immediately,
  `EXHAUST` breaks to the post-loop failure builder, `NEXT_OUTER` continues
  the for. The explicit `AssertionError` guard at L4567-4570 (added in
  `80e518337`) catches future enum extensions.
- **Continuation-step state machine (audit item 2):** All thirteen returns
  from `_run_alm_continuation_step` flow through `_finalize_continuation_step`
  (L3259-3280), which builds `_ALMContinuationStepResult` from the
  `_ContinuationStepState` carrier — every field is sourced explicitly from
  the carrier; no field is silently default-initialized at the call sites.
- **Stall classification (audit item 3):** `_classify_infeasible_inner_stall`
  (L1425-1465) enforces all four short-circuits in order: `moved_norm > tol` →
  not a stall; `candidate <= feasibility_gate` → not a stall; feasibility
  improved → not a stall; objective improved → not a stall. The four
  classification arms (`relative_objective_termination_*`,
  `successful_inner_solve_*`, `failed_inner_solve_*`) match the unit tests
  added in `382d7a082`.
- **Dual-update timing (audit item 4):** Multipliers are updated only at
  L4228-4265, which fires *after* `_run_alm_inner_attempts` at L3900 has
  produced a candidate AND `state.final_eval` has been refreshed at L3926
  AND the convergence-against-`update_stationarity_tol` predicate has been
  evaluated. Pre-inner-solve evaluations never touch `state.multipliers`.
- **Penalty cap & escape (audit item 5):** `_apply_alm_penalty_increase`
  (L2506-2577) is the single penalty-bump path. When `_next_penalty` returns
  `cap_hit=True`, `_apply_continuation_penalty_increase` (L3340-3404) returns
  a populated `cap_result` which `_emit_alm_penalty_increase_arm` returns as
  RETURN, terminating cleanly via `_handle_alm_penalty_cap_termination` with
  `termination_reason="penalty_cap_reached"`. No infinite-loop hazard:
  `cap_hit` is sticky (once `requested > penalty_max`, it stays True every
  call), so the next attempt yields the same RETURN.
- **Inner-solve retry (audit item 6):** `_run_alm_inner_attempts` (L3058-3230)
  retries with shrinking trust radius only; it never bumps μ. The penalty-
  bump decision is delegated to the outer driver via the
  `forced_infeasible_penalty_cycle` flag (set at L3177) which arm 6 (L4120)
  honors. Acceptance and stall classification are checked before any retry.
- **Feasible-incumbent monotonicity (audit item 7):** `state.best_feasible`
  is updated at L3987-3998 with a strict `<` objective improvement guard.
  `_restore_alm_best_feasible_on_failure` (L2777-2829) restores the incumbent
  whenever the final iterate is infeasible OR worse than `best_feasible`.
  (See LOW finding above for the L3862 unconditional-overwrite edge case.)
- **History append-only invariant (audit item 8):** Entries are appended at
  L2338 inside `_append_alm_history_entry` and never re-ordered or removed
  (history truncation only drops oldest entries when over capacity at
  L2340-2343). Entries ARE mutated in-place after append (see MEDIUM
  finding), but the snapshot delivered to `history_callback` is deep-copied
  via `_snapshot_history_entry` at L2384, so external observers see a
  consistent point-in-time view.
- **Off-spec escape hatches (audit item 9):** No `os.environ.get(...)`,
  `getenv(...)`, or `ACCEPT_OFFSPEC*` references inside `alm_utils.py`. The
  off-spec hatch removal in `d61648f50` was applied to constraint-contract /
  Stage-2 wrappers, not to the ALM driver.
- **Skipped-inner shortcut (audit item 10):** L3809-3898 fires only when the
  iterate is at full user-strict convergence (`settings.feasibility_tol` AND
  `settings.stationarity_tol` both met) AND not in the constraints_inactive
  or signal_mismatch arms. This is the full KKT predicate, so dual update is
  *correctly* skipped — there is no further gradient to drive λ. Returns
  `_ALMContinuationDecision.RETURN` with a `converged` history action; outer
  loop terminates immediately. Unit-tested by
  `SkippedInnerShortcutTests::test_minimize_alm_skips_inner_solve_when_already_converged`.
- **Disjoint dispatch arms:** Arms 5 (converged), 6 (forced_infeasible),
  7 (constraints_inactive), 8 (signal_mismatch), 9 (dual_update),
  10 (subproblem_continue/limit), 11 (default penalty_increase) are
  mutually exclusive given their conditions. Arm 6 cannot fire on a
  feasible iterate because `_classify_infeasible_inner_stall` returns False
  when `candidate_max_feasibility_violation <= feasibility_gate`.
- **Trust-radius schedule:** `_grow_continuation_trust_radius` (L3436-3454)
  is shared between arm 8 (signal_mismatch progress) and arm 10b
  (hard_feasible_for_update progress). Shrink schedule lives in
  `_run_alm_inner_attempts` at L3189-3202.

## Control-Flow Trace (Pseudocode)

```
minimize_alm(x0, constraint_names, evaluate_problem, settings, ...):
    return _minimize_alm_impl(...)

_minimize_alm_impl:
    normalized = _normalize_alm_run_inputs(x0, ..., settings, ...)
    run_state = ALMRunState(x=normalized.x, history=[], penalty_cap_*=False, ...)
    multipliers = normalized.multipliers
    penalty = normalized.penalty
    update_feasibility_tol, update_stationarity_tol = penalty-schedule tols
    final_eval = last_result = best_feasible = None
    inner_options_state = inner_options

    for outer_iteration in 1 .. max_outer_iterations:
        is_final_outer = (outer_iteration == max_outer_iterations)
        outcome = _run_alm_outer_iteration(...)
        # propagate (multipliers, penalty, update_*_tol, last_result,
        #            final_eval, final_*, best_feasible, inner_options) from outcome

        if outcome.decision == RETURN:    return outcome.result
        if outcome.decision == EXHAUST:   break
        if outcome.decision == NEXT_OUTER: continue
        else: raise AssertionError

    if final_eval is None or last_result is None: raise RuntimeError  # defensive
    termination_reason = _termination_reason_from_history(history)
    return _build_alm_failure_result_with_optional_restore(
        ..., termination_reason=termination_reason,
        final_max_feasibility_violation=feasibility(final_eval))


_run_alm_outer_iteration:
    if outer_state_callback: outer_state_callback(outer_iteration, multipliers, penalty)
    feasible_stall_count = 0

    for continuation_iteration in 0 .. max_subproblem_continuations:
        step = _run_alm_continuation_step(...)
        # propagate state from step

        if step.decision == RETURN:               return finalize(step, RETURN, step.result)
        if step.decision == BREAK_OUTER:          break
        if step.decision == CONTINUE_CONTINUATION: continue
        else: raise AssertionError

    # Inner loop natural exit: only reachable when the last continuation
    # returned CONTINUE_CONTINUATION. The post-inner-solve state machine
    # makes this a narrow edge case (signal_mismatch + made_progress +
    # continuation==0 == max_subproblem_continuations). See HIGH finding.
    return _ALMOuterIterationResult(
        decision=EXHAUST if is_final_outer else NEXT_OUTER,
        ...)


_run_alm_continuation_step:
    state = _ContinuationStepState(multipliers, penalty, ..., best_feasible, inner_options)
    current_eval = evaluate_problem(run_state.x, state.multipliers, penalty)
    require_finite(current_eval)
    effective_feasibility_tol = clamp(update_feasibility_tol, [feasibility_tol, 1e-2])
    current_routing_state = build_routing(current_eval, ..., effective_feasibility_tol)

    # ARM A: Skipped-inner shortcut (full KKT convergence at user-strict tols)
    if current_max_feas <= settings.feasibility_tol
       and current_stationarity <= settings.stationarity_tol
       and not constraints_inactive_candidate
       and not signal_mismatch_active:
        update best_feasible and final_eval (unconditional overwrite — see LOW finding)
        emit history "converged"
        return finalize(state, RETURN, build_converged_result(...))

    # Inner solve
    inner_attempt = _run_alm_inner_attempts(...)
    update run_state.x, run_state.trust_radius, run_state.total_inner_iterations
    state.final_eval = inner_attempt.evaluation
    routing_state = build_routing(state.final_eval, ..., update_feasibility_tol)  # MEDIUM: gate mismatch
    compute hard_feasible_strict, hard_feasible_for_update, constraints_inactive_candidate
    made_inner_progress = check movement / objective-drop / feasibility-drop / stationarity-drop
    if feasible: update best_feasible (strict <)
    append history entry; attach diagnostics
    state.final_multipliers = state.multipliers.copy(); state.final_penalty = state.penalty

    # ARM B (5): KKT-converged at user-strict tols, post-inner
    if max_feas <= settings.feasibility_tol
       and stationarity <= settings.stationarity_tol
       and not constraints_inactive_candidate
       and not signal_mismatch_active:
        return emit_converged_step(...)  # RETURN

    # ARM C (6): Inner solver false-success / no movement on infeasible iterate
    if inner_attempt.forced_infeasible_penalty_cycle:
        return emit_penalty_increase_arm(action="infeasible_stall_penalty_increase")
        # → cap_hit ? RETURN(cap_result) : BREAK_OUTER

    # ARM D (7): Constraints inactive + Stage-2 signals consistent
    if constraints_inactive_candidate:
        if stationarity <= settings.stationarity_tol:
            return emit_converged_step(termination="constraints_inactive_converged")
        if not made_inner_progress and continuation_iteration > 0:
            return emit_stall_failure_step(termination="constraints_inactive_stall")
        # else fall through

    # ARM E (8): Signal mismatch with hard-feasible iterate
    if signal_mismatch_active and hard_feasible_strict:
        if not made_inner_progress or continuation_iteration > 0:
            if surrogate_positive_shift_zero:
                return emit_stall_failure_step(termination="signal_mismatch_stall")
            return emit_penalty_increase_arm(action="signal_mismatch_penalty_increase")
        feasible_stall_count = 0
        grow_trust_radius()
        return emit_subproblem_continue(...)  # CONTINUE_CONTINUATION

    # ARM F (9): Inner subproblem converged at update tol → DUAL UPDATE
    if hard_feasible_for_update and stationarity <= update_stationarity_tol:
        feasible_stall_count = 0
        dual_update = _handle_alm_dual_update_transition(state.multipliers, ...)
        state.multipliers = dual_update.multipliers       # NEW λ
        state.update_*_tol = dual_update.update_*_tol     # tighten by penalty_scale
        # state.final_multipliers / state.final_penalty NOT updated — see NIT
        if multiplier_cap_binding: run_state.cap_binding_* update
        annotate "dual_update" history (sets outer_termination on final outer)
        return finalize(state, BREAK_OUTER, None)

    # ARM G (10): Hard-feasible but stationarity not met → continue or limit
    if hard_feasible_for_update:
        feasible_stall_count = 0 if made_inner_progress else feasible_stall_count + 1
        hit_stall_limit = (continuation_iteration == max_subproblem_continuations
                           or feasible_stall_count >= _PLATEAU_STALL_LIMIT=2)
        if hit_stall_limit:
            annotate "subproblem_limit" history
            if feasible_stall_count >= _PLATEAU_STALL_LIMIT:
                return finalize(state, RETURN, plateau_stall_failure)
            return finalize(state, BREAK_OUTER, None)  # MEDIUM: no μ bump
        grow_trust_radius()
        if not made_inner_progress: tighten update_stationarity_tol toward 0.5 * stationarity_norm
        return emit_subproblem_continue(...)  # CONTINUE_CONTINUATION

    # ARM H (11): Default fall-through — infeasible, bump penalty
    return emit_penalty_increase_arm(action="penalty_increase")


_run_alm_inner_attempts:
    accepted_* = None
    attempt_radius = trust_radius
    for attempt_index in 1 .. max_inner_attempts:
        profile = _select_inner_solve_profile(attempt_radius, continuation, feasible_enough)
        inner_options = _build_inner_options(request.inner_options, ..., profile)  # MEDIUM: gtol monotone
        bounds = _build_box_bounds(x, attempt_radius)
        try:
            result = scipy.minimize(L-BFGS-B, jac=True, callback=evaluator.callback)
            candidate_x, candidate_eval = ...
        except _EarlyStopInnerSolve as early_stop:
            result = (success=True), candidate_x = early_stop.x, candidate_eval = ...

        moved_norm = ||candidate_x - x||
        acceptable = _candidate_is_acceptable(...)             # ATOL=1e-10, RTOL=1e-3
        infeasible_inner_stall, false_success, reason =
            _classify_infeasible_inner_stall(...)              # 4-arm classification

        if acceptable and not infeasible_inner_stall:
            accept; if moved >= 0.5 * attempt_radius: trust_radius *= grow
            break
        if infeasible_inner_stall:
            accept current_eval; forced_infeasible_penalty_cycle = True; break
        if attempt_radius is None:
            accept current_eval; break  # unbounded, no trust region
        if attempt_radius <= trust_radius_min or last_attempt:
            accept current_eval; break
        attempt_radius *= shrink
        continue

    # LOW: The defensive "if accepted_result is None" fallback at L3205
    # is unreachable given max_inner_attempts > 0.
    return ALMInnerAttemptResult(...)
```

## How To Reproduce The HIGH Finding

```python
# pseudo-test
settings = ALMSettings(
    max_outer_iterations=1,
    max_subproblem_continuations=0,
    penalty_init=1.0,
    feasibility_tol=1e-6,
    stationarity_tol=1e-6,
)
# evaluate_problem returns a stage-2 evaluation that:
#   - keeps hard_max_violation <= settings.feasibility_tol (hard-feasible strict)
#   - has signal_mismatch_active = True (surrogate disagrees with hard signals)
#   - moves x and reduces base objective on the inner solve
result = minimize_alm(x0, names, evaluate_problem, settings, {})
assert result.termination_reason.startswith("max_outer")  # FAILS today: returns "subproblem_continue"
```

## Suggested Closeout

1. Patch `_emit_alm_subproblem_continue` to accept and honor `is_final_outer`
   plus `continuation_iteration`/`max_subproblem_continuations` so the final
   continuation tags `outer_termination = "max_outer"`. Update both call
   sites (signal-mismatch arm at L4220-4226 and feasible-update arm at
   L4319-4330).
2. Unify the routing-state gate. Either pass `effective_feasibility_tol`
   to both call sites at L3791 and L3941, or document why the post-inner
   site uses `update_feasibility_tol`. Adding a comment is acceptable;
   silently letting the gate diverge is not.
3. Decide whether `subproblem_limit (max_subproblem_continuations)` should
   bump μ before BREAK_OUTER. If yes, route through
   `_emit_alm_penalty_increase_arm` like the default fall-through.
4. Stop persisting `staged_gtol` into `state.inner_options` so subsequent
   continuations can tighten `gtol` as `update_stationarity_tol` shrinks.
5. Document the append-then-mutate history pattern at the top of
   `_run_alm_continuation_step` and on `_append_alm_history_entry`.
6. Add `initial_multipliers` shape/sign/finite validation in
   `_normalize_alm_run_inputs`.

End of report.
