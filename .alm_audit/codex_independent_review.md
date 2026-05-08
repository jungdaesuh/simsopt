# Sections Audited and Why

I audited three high-risk ALM seams rather than the whole 4.6k-line driver:
signed constraint normalization/routing, dual-update and scalar penalty math, and the continuation/outer state machine. These are the areas most likely to hide a real math or algorithm bug after the recent signed-semantics, normalization, penalty-cap, and continuation refactors.

Focused validation run:

```bash
python -m pytest -q tests/geo/test_alm_utils.py -k 'DualUpdate or stage2_signal or signal_mismatch or ContinuationStep or OuterIteration or skipped_inner or penalty_cap'
# 16 passed, 102 deselected
```

## 1. Signed constraint normalization and hard-vs-surrogate routing

### Math Contract

The ALM convention should be `g_i(x) <= 0`; positive signed residuals are violations. For smoothed geometry constraints, the objective may use a differentiable surrogate signed value, but feasibility certification and multiplier updates must use the chosen hard/certification signal. Normalization must divide both signed residuals and gradients by the same positive scale.

### Code Walk

The generic inequality objective implements the standard projected-inequality ALM formula:

```text
examples/single_stage_optimization/alm_utils.py:385-390
penalty_values = _penalty_values(penalty, constraint_values.size)
positive_shift = np.maximum(0.0, multipliers + penalty_values * constraint_values)
augmented_terms = _augmented_terms(positive_shift, multipliers, penalty_values)
```

The value term is the equivalent of
`(max(0, lambda + rho*g)^2 - lambda^2) / (2*rho)` and the gradient is
`base_grad + max(0, lambda + rho*g) * grad(g)`.

The sign convention in the geometry helpers matches `g <= 0`:

```text
examples/single_stage_optimization/banana_opt/single_stage_constraints.py:130-135
signed_value = float(minimum_distance) - float(smooth_min)
hard_signed_value = float(minimum_distance) - float(hard_min)
result = (signed_value, -grad, max(0.0, signed_value))
```

Stage 2 uses the same lower-bound distance sign and returns the hard signed sidecar:

```text
examples/single_stage_optimization/banana_opt/stage2_objectives.py:1514-1518
# grad = d(smooth_min)/dx, but signed_value = min_dist - smooth_min,
# so d(signed_value)/dx = -d(smooth_min)/dx = -grad.
signed_value = float(minimum_distance) - smooth_min
hard_signed_value = float(minimum_distance) - float(hard_min)
```

Stage 2 metadata explicitly routes surrogate geometry into the objective/gradient while using hard values for dual update and feasibility:

```text
examples/single_stage_optimization/banana_opt/stage2_objectives.py:741-749
uses_surrogate = constraint_name in surrogate_hardware_names
objective_value_kind="surrogate" if uses_surrogate else "hard",
gradient_value_kind="surrogate" if uses_surrogate else "hard",
dual_update_value_kind="hard",
feasibility_value_kind="hard",
```

The evaluator then normalizes the surrogate objective signal and the chosen hard/surrogate update signal separately:

```text
examples/single_stage_optimization/banana_opt/stage2_objectives.py:1893-1905
normalized_payload = normalize_alm_constraints(...)
normalized_surrogate_signed_constraint_values = normalized_payload["normalized_signed_values"]
normalized_hard_violation_values = normalized_payload["normalized_feasibility_values"]
```

```text
examples/single_stage_optimization/banana_opt/stage2_objectives.py:1918-1936
raw_dual_update_values, raw_feasibility_values = _stage2_alm_signal_values(...)
normalized_signal_payload = normalize_alm_constraint_signals(...)
normalized_dual_update_values = normalized_signal_payload["normalized_signed_values"]
normalized_feasibility_values = normalized_signal_payload["normalized_feasibility_values"]
```

The final evaluation preserves this split:

```text
examples/single_stage_optimization/banana_opt/stage2_objectives.py:1955-1963
"dual_update_values": normalized_dual_update_values,
"feasibility_values": normalized_feasibility_values,
"hard_signed_constraint_values": normalized_hard_signed_constraint_values,
"surrogate_signed_constraint_values": normalized_surrogate_signed_constraint_values,
"hard_dual_update_values": normalized_hard_signed_constraint_values,
```

The core solver enforces that explicit Stage 2 signal mode is all-or-nothing:

```text
examples/single_stage_optimization/alm_utils.py:1883-1907
stage2_signal_fields = (...)
if missing_fields:
    raise KeyError("ALM stage-2 signal evaluation missing required fields: " + ...)
```

Routing uses the hard signal for certification/update pressure and separately computes surrogate pressure to detect mismatch:

```text
examples/single_stage_optimization/alm_utils.py:2007-2016
hard_positive_shift = _positive_shift(... preferred_dual_update_values)
surrogate_positive_shift = _positive_shift(... surrogate_signed_constraint_values)
```

Tests explicitly pin signed/normalized sidecars and mismatch behavior:

```text
tests/geo/test_alm_utils.py:1983-2016
test_stationarity_metrics_uses_raw_norm_when_stage2_signals_disagree
...
self.assertTrue(mismatch)
self.assertIsNone(kkt_stationarity_norm)
```

### Verdict

No bug found in the audited signed-semantics path. The constraint builders, normalizers, metadata, and solver routing agree on `g <= 0`. The hard-vs-surrogate split is explicit and the tests are not just shape tests; they assert mismatch behavior and sidecar values.

Residual note: I did not audit every possible geometry edge case. One non-ALM edge case worth a separate pass is empty curve/surface collections in helper functions, because Stage 2 and single-stage helpers do not all handle those degenerate inputs identically.

## 2. Dual update, multiplier projection, and scalar penalty cap math

### Math Contract

For inequality constraints `g_i(x) <= 0`, the multiplier update should be
`lambda_i^+ = max(0, lambda_i + rho_i * g_i(x))`, optionally clipped by a configured multiplier cap. Penalty growth should multiply `rho` by `penalty_scale`, stop at `penalty_max`, and not silently exceed the cap.

### Code Walk

Multiplier update uses the expected projected formula:

```text
examples/single_stage_optimization/alm_utils.py:1548-1553
penalty_values = _penalty_values(penalty, dual_update_array.size)
return np.maximum(0.0, np.asarray(multipliers, dtype=float) + penalty_values * dual_update_array)
```

Clipping is diagnostic, not silent:

```text
examples/single_stage_optimization/alm_utils.py:1567-1575
cap_binding_mask = updated > cap
return (np.minimum(updated, cap), bool(np.any(cap_binding_mask)), ...)
```

The dual-update transition uses the routing state's preferred dual-update values, so Stage 2 hard signals are used where the metadata chooses them:

```text
examples/single_stage_optimization/alm_utils.py:2946-2952
_project_nonnegative_multipliers_with_diagnostics(
    multipliers,
    routing_state.signal_state.preferred_dual_update_values,
    penalty_argument,
```

Penalty growth is scalar in the current contract and stops at the cap:

```text
examples/single_stage_optimization/alm_utils.py:1615-1625
requested_penalty = penalty * penalty_scale
if not np.isfinite(requested_penalty) or requested_penalty > penalty_max:
    return penalty_max, True, requested_penalty
return requested_penalty, False, requested_penalty
```

Cap termination records the requested value and builds a failure result rather than continuing with an over-cap penalty:

```text
examples/single_stage_optimization/alm_utils.py:2523-2530
next_penalty, cap_hit, requested_penalty = _next_penalty(...)
if cap_hit:
    history_entry["action"] = "penalty_cap_reached"
```

The current branch intentionally removed live block-penalty control. That is not a hidden missing branch in this audit scope; it is documented and tested as the current contract:

```text
docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md:13-15
- [x] Scalar ALM penalty control only; block-penalty state/settings/helpers deleted.
- [x] `constraint_blocks` retained as diagnostic labels only.
- [x] Legacy `block_penalties` / `ALM_BLOCK_PENALTIES` fields retained as nullable `None`.
```

The corresponding test confirms diagnostic-only blocks and scalar penalty values:

```text
tests/geo/test_alm_utils.py:284-290
self.assertTrue(result.success)
self.assertEqual(penalty_arguments, [1.0])
self.assertIsNone(result.block_penalties)
self.assertEqual(result.penalty_values, [1.0, 1.0])
```

### Verdict

No bug found in the audited dual update or scalar penalty cap path. The arithmetic matches projected inequality ALM, multiplier caps are surfaced, and scalar penalty cap termination is explicit. The old "phase 4/5 block penalty" risk is stale for this branch because the current source of truth intentionally makes blocks diagnostic-only.

One design limitation remains, but it is not a bug under the current contract: all live penalty growth is scalar, so a single blocking constraint can raise the global penalty for every block.

## 3. Continuation-step / outer-iteration state machine and skipped-inner shortcut

### Algorithm Contract

Each outer iteration may run multiple continuation attempts. A continuation step should either return a final ALM result, break to the next outer iteration after a penalty or dual update, or continue the subproblem. The skipped-inner shortcut is only valid when the current iterate already satisfies the strict user feasibility/stationarity gates and is not in a Stage 2 hard/surrogate mismatch arm.

### Code Walk

The skipped-inner condition is strict and excludes inactive/mismatch Stage 2 cases:

```text
examples/single_stage_optimization/alm_utils.py:3809-3814
if (
    current_max_feasibility_violation <= settings.feasibility_tol
    and current_stationarity_norm <= settings.stationarity_tol
    and not current_constraints_inactive_candidate
    and not current_signal_mismatch_active
):
```

When the shortcut fires, it writes a converged history entry with zero inner iterations and returns through the same result builder as normal convergence:

```text
examples/single_stage_optimization/alm_utils.py:3815-3823
history_entry, run_state.history_truncated_count = _append_alm_history_entry(...)
...
_build_skipped_inner_history_entry(...)
```

The post-inner state machine handles the high-risk arms in a deterministic order:

```text
examples/single_stage_optimization/alm_utils.py:4091-4096
if max_feasibility_violation <= settings.feasibility_tol
    and stationarity_norm <= settings.stationarity_tol
    and not constraints_inactive_candidate
    and not signal_mismatch_active:
```

Then infeasible stall, inactive-hard-constraint, signal-mismatch, dual-update, feasible plateau, and penalty-increase arms are mutually ordered:

```text
examples/single_stage_optimization/alm_utils.py:4120-4136
if inner_attempt.forced_infeasible_penalty_cycle:
    return _emit_alm_penalty_increase_arm(...)
```

```text
examples/single_stage_optimization/alm_utils.py:4182-4218
if signal_mismatch_active and hard_feasible_strict:
    ...
    return _emit_alm_penalty_increase_arm(...)
```

```text
examples/single_stage_optimization/alm_utils.py:4228-4265
if hard_feasible_for_update and stationarity_norm <= state.update_stationarity_tol:
    dual_update = _handle_alm_dual_update_transition(...)
    ...
    return _finalize_continuation_step(... BREAK_OUTER ...)
```

The outer dispatcher copies every sticky field out of the continuation result before checking the decision, so updated multipliers, penalty, tolerances, final evaluation, best feasible incumbent, and inner options are not dropped:

```text
examples/single_stage_optimization/alm_utils.py:4433-4443
multipliers = step.multipliers
penalty = step.penalty
update_feasibility_tol = step.update_feasibility_tol
...
inner_options = step.inner_options
```

Unhandled continuation and outer decisions fail fast:

```text
examples/single_stage_optimization/alm_utils.py:4450-4453
if step.decision != _ALMContinuationDecision.CONTINUE_CONTINUATION:
    raise AssertionError(f"unhandled continuation decision {step.decision!r}")
```

The tests pin both direct continuation dispatch and the full skipped-inner shortcut:

```text
tests/geo/test_alm_utils.py:4008-4039
test_continuation_step_short_circuits_when_iterate_is_already_converged
...
self.assertEqual(result.result.termination_reason, "converged")
```

```text
tests/geo/test_alm_utils.py:4386-4394
SkippedInnerShortcutTests
...
the ALM driver must skip the L-BFGS-B inner solve
```

### Verdict

No bug found in the audited continuation/outer state machine. I specifically looked for dropped transition fields, stale final-evaluation threading, skipped-inner overreach, and missing fail-fast handling. The state carrier and dispatcher preserve the updated fields, and the targeted tests exercise the same branch classes I audited.

One caveat: the skipped-inner test is intentionally non-Stage-2. Stage 2 mismatch/inactive branches are covered elsewhere in `test_alm_utils.py`, but I did not construct a new end-to-end stellarator fixture run in this audit.

## Sections Explicitly NOT Audited

I did not audit the full Stage 2 or single-stage entrypoint wiring, real equilibrium fixture behavior, Boozer/iota physics objective correctness, all geometry degeneracies, benchmarking/ledger export code, or every frontier-related dirty worktree file. The breadth pass should still cover those. I also treated scalar penalty control as the current contract rather than re-litigating the deleted block-penalty feature.
