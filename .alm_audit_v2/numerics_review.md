# ALM Numerics Audit v2 — 2026-05-08

## Summary

Audit of numerical stability surfaces in `examples/single_stage_optimization/alm_utils.py`
on `surrogate-confinement-v2` HEAD `e7b836464`, after the M2-M9 / H1 / M7 fixes
landed in commits `bf936a0a4` and `a169f296a`. Compared against the v1 audit
(`.alm_audit/numerics_review.md`).

The v1 audit's MED/LOW findings remain in the regression set (1.1 nnls is
addressed by M6; 1.2 attach-metadata aliasing addressed by L1; 1.3-1.5 partly
addressed by L4 `_clone_evaluation_dict`). However, this re-audit surfaces
**three HIGH-severity surviving defects** and several MEDIUM/LOW items that
were not in the v1 scope:

- **F1 (HIGH)**: `_surrogate_kkt_stationarity_norm` was not updated when M9
  fixed `_stationarity_metrics`. The surrogate KKT diagnostic still feeds the
  augmented gradient into `nnls`, so `surrogate_kkt_stationarity_norm`
  reported in history and `final_surrogate_kkt_stationarity_norm` reported in
  the result silently collapse to ~0 once L-BFGS-B converges, regardless of
  multiplier quality.
- **F2 (HIGH)**: Stage-2 signal fields (`hard_signed_constraint_values`,
  `hard_violation_values`, `surrogate_signed_constraint_values`,
  `hard_dual_update_values`) participate in routing/dual-update decisions but
  are **not** in `_nonfinite_evaluation_fields`'s array-validation list. NaN
  in `hard_dual_update_values` flows directly into the dual-update projection,
  silently producing NaN multipliers.
- **F3 (HIGH)**: `_project_nonnegative_multipliers_with_diagnostics` cap
  comparison `updated > cap` evaluates to False when `updated` contains NaN,
  so cap-binding diagnostics lie about NaN multipliers and `np.minimum(NaN, cap)`
  yields NaN. Combined with F2 this is a complete silent-NaN-propagation path.

Together, F1, F2, F3 create a path where a NaN out of a Stage-2 evaluator
silently corrupts multipliers and flatters the KKT residual diagnostic, with
the only crash deferred to the next outer iteration's
`_require_finite_evaluation` call.

MEDIUM findings: NaN-permissive `ALMSettings.__post_init__` validation (F4);
hardware scale floor at machine-epsilon allows numerically-destructive
amplification (F5). LOW findings: divide-by-zero in Taylor diagnostic (F6);
`alm_raw_dual_estimates` lacks scale-positivity validation (F7); semantic
inconsistency in inactive-set definition between `_kkt_stationarity_norm` and
`_constraint_activity_mask` (F8); `_normalize_trust_radius` accepts `+inf`
silently changing profile selection (F9).

## Methodology

1. Read v1 audit (`.alm_audit/numerics_review.md`) and `FIX_PLAN.md` to baseline
   prior findings.
2. `git show bf936a0a4 -- examples/single_stage_optimization/alm_utils.py` and
   `git show a169f296a` to see what changed since v1.
3. Read every function called out in the task (L1247-1354 finite checks,
   L1363 conditioning, L1461 acceptance bound, L1505-1564 stall metrics,
   L1567-1614 trust radius / profile, L1655-1705 multiplier / penalty
   schedule, L1727-1762 inner options, L1798-1823 Taylor directions,
   L2140-2244 KKT and stationarity).
4. Trace each candidate finding back to a producing call site to verify
   reachability, and forward to a consumer to confirm the visible effect.
5. Verify NumPy NaN semantics empirically (`np.any([NaN > 1.0])`,
   `nnls(A, b_with_nan)`) instead of trusting documentation.
6. Cross-check upstream guards: `_require_finite_evaluation`,
   `_sanitize_nonfinite_inner_evaluation`, `validate_initial_multipliers`,
   `_penalty_values`.

## Findings

### F1: `_surrogate_kkt_stationarity_norm` did not receive the M9 fix [HIGH]
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
        metric_grad,
        evaluation.get("constraint_grads"),
        surrogate_values,
        np.maximum(surrogate_values, 0.0),
        _constraint_activity_tolerances(evaluation, surrogate_values),
        feasibility_gate,
    )
```
- Bug: M9 (commit `bf936a0a4`) corrected `_stationarity_metrics` to use
  `evaluation.get("base_grad", metric_grad)` instead of the augmented
  `metric_grad`/`grad`, with the rationale (verbatim from L2221-2227):
  `"Augmented gradient ∇L_A already contains the active-constraint
  contribution (λ + μc)∇c, so feeding it to nnls' active-set projection
  collapses to ~0 once the inner solve converges and hides multiplier-quality
  defects."` The same defect lives in `_surrogate_kkt_stationarity_norm`,
  which still feeds `metric_grad` (which falls through to `evaluation["grad"]`
  = augmented gradient when `metric_grad` is absent) into `_kkt_stationarity_norm`.
- Why: L-BFGS-B drives `‖∇L_A‖` to ≤ gtol. The active-set projection of the
  augmented gradient onto the constraint normals is therefore ≤ gtol by
  construction. The "surrogate KKT residual" reported in
  `_constraint_history_diagnostics_source` (L969) and surfaced in
  `_alm_summary` (L1158-1160) and `final_surrogate_kkt_stationarity_norm`
  (L2898) thus collapses to numerical noise on every history entry whose
  inner solve converged, **regardless of whether the multipliers are
  garbage**.
- Impact: `surrogate_kkt_stationarity_norm` is a *certification* metric used
  by post-run analyses (e.g. champion-archive promotion). Operators reading
  ~0 in this field will conclude "Stage 2 surrogate KKT satisfied" while in
  fact the diagnostic is uninformative. Identical to the symptom M9 was
  written to fix, but on the surrogate path.
- Repro: Any ALM run with `explicit_stage2_signals=True` reaching inner-solve
  convergence (`stationarity_norm <= gtol`) — the surrogate KKT will read
  ~`gtol` regardless of `‖∇f + Σλ_i ∇c_i‖`. Concretely, take any current
  `tests/geo/test_alm_utils.py` fixture that produces explicit hard signals,
  sample `surrogate_kkt_stationarity_norm` at convergence, and compare against
  `‖base_grad + Σ_active mult_active * constraint_grad_active‖`.
- Suggested fix (single-line change mirroring M9):
```python
    kkt_grad = np.asarray(
        evaluation.get("base_grad", evaluation.get("metric_grad", evaluation["grad"])),
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

### F2: Stage-2 signal fields excluded from `_nonfinite_evaluation_fields` [HIGH]
- File: `examples/single_stage_optimization/alm_utils.py:1271-1284`
- Code:
```python
    optional_array_fields = (
        "constraint_values",
        "feasibility_values",
        "dual_update_values",
        "metric_grad",
        "base_grad",
        "constraint_activity_tolerances",
    )
    for field_name in optional_array_fields:
        if field_name not in evaluation:
            continue
        field_values = np.asarray(evaluation[field_name], dtype=float)
        if not np.all(np.isfinite(field_values)):
            invalid_fields.append(field_name)
```
- Bug: The Stage-2 explicit-signal fields `hard_signed_constraint_values`,
  `hard_violation_values`, `surrogate_signed_constraint_values`, and
  `hard_dual_update_values` are listed in `_OWNED_EVALUATION_ARRAY_FIELDS`
  (L326-350) — i.e. ALM treats them as evaluation outputs that need ownership
  guarantees — but they are **not** in this finiteness whitelist. NaN/Inf in
  any of them passes `_require_finite_evaluation` silently.
- Why: `_extract_stage2_constraint_signal_state` (L1979-2037) reads these
  fields without validation when `explicit_stage2_signals=True`.
  `routing_state.signal_state.preferred_dual_update_values` then equals
  `hard_dual_update_values`, which feeds directly into
  `_handle_alm_dual_update_transition`'s
  `_project_nonnegative_multipliers_with_diagnostics(... preferred_dual_update_values, ...)`
  call at L3099-3105. NaN propagates through `_updated_nonnegative_multipliers`
  (NaN passes `np.maximum(0.0, NaN) = NaN`), and the cap check at L1669
  `cap_binding_mask = updated > cap` is False for NaN (verified empirically:
  `np.any(np.array([nan]) > 1.0) == False`).
- Impact: Stage-2 evaluators that hit a numerical edge case (e.g., LCFS
  reconstruction failing on a topologically-degenerate surface; Boozer
  residual blowing up) and emit a NaN in `hard_dual_update_values` will
  silently corrupt the next iteration's multipliers. Crash is deferred to
  the next `_require_finite_evaluation` call (`evaluate_problem` returning
  NaN gradient because input multipliers are NaN), which surfaces as an
  ALM run termination with provenance pointing at "outer iterate
  evaluation" — misattributing root cause.
- Repro: Construct a Stage-2 evaluation where `hard_dual_update_values =
  np.array([1.0, np.nan])`, all other fields finite. Pass to
  `_handle_alm_dual_update_transition`; observe new multipliers contain NaN
  and `multiplier_cap_binding=False`.
- Suggested fix:
```python
    optional_array_fields = (
        "constraint_values",
        "feasibility_values",
        "dual_update_values",
        "metric_grad",
        "base_grad",
        "constraint_activity_tolerances",
        # F2: stage-2 explicit signal fields participate in routing /
        # dual update; finiteness must be enforced at the same boundary as
        # `dual_update_values`.
        "hard_signed_constraint_values",
        "hard_violation_values",
        "surrogate_signed_constraint_values",
        "hard_dual_update_values",
    )
```

### F3: `_project_nonnegative_multipliers_with_diagnostics` mishandles NaN at the cap [HIGH]
- File: `examples/single_stage_optimization/alm_utils.py:1655-1674`
- Code:
```python
def _project_nonnegative_multipliers_with_diagnostics(
    multipliers: np.ndarray,
    dual_update_values: np.ndarray,
    penalty,
    multiplier_max: float | None,
) -> tuple[np.ndarray, bool, list[int]]:
    updated = _updated_nonnegative_multipliers(
        multipliers,
        dual_update_values,
        penalty,
    )
    if multiplier_max is None:
        return updated, False, []
    cap = float(multiplier_max)
    cap_binding_mask = updated > cap
    return (
        np.minimum(updated, cap),
        bool(np.any(cap_binding_mask)),
        np.flatnonzero(cap_binding_mask).tolist(),
    )
```
- Bug: When `updated` contains NaN, `updated > cap` returns False at those
  positions (NaN comparison semantics). The cap is therefore not "applied" —
  `np.minimum(NaN, cap)` returns NaN, not `cap` — and the diagnostic
  `multiplier_cap_binding` reports False at exactly those indices that
  silently passed through as NaN.
- Why: NumPy semantics: `nan > x = False`, `np.minimum(nan, x) = nan`.
  Empirical: `python3 -c "import numpy as np; print(np.any(np.array([np.nan]) > 1.0)); print(np.minimum(np.nan, 1.0))"` ⇒ `False  nan`.
  v1 audit Finding 1.5 already noted this as defense-in-depth-only. Combined
  with F2 (which removes the upstream finite guard), this becomes a real
  reachable path.
- Impact: NaN multipliers flow into next outer iteration. The
  `last_cap_binding_active` predicate at L4434 is set to `False` (because
  `dual_update.multiplier_cap_binding == False`), so the converged-success
  gate at L4273 / L4320 is *not* blocked by cap-binding. A run with NaN
  multipliers could in principle hit the converged-success arm (depends on
  whether the next eval's `total/grad` are sanitized to non-NaN — they would
  be, by `_sanitize_nonfinite_inner_evaluation` upstream of dual update —
  but the `stationarity_norm` reported with NaN multipliers is still wrong).
- Repro: F2 repro plus the projection step.
- Suggested fix (defense-in-depth even after F2 fixes the upstream gap):
```python
def _project_nonnegative_multipliers_with_diagnostics(
    multipliers: np.ndarray,
    dual_update_values: np.ndarray,
    penalty,
    multiplier_max: float | None,
) -> tuple[np.ndarray, bool, list[int]]:
    updated = _updated_nonnegative_multipliers(
        multipliers,
        dual_update_values,
        penalty,
    )
    if not np.all(np.isfinite(updated)):
        bad = np.flatnonzero(~np.isfinite(updated)).tolist()
        raise ValueError(
            f"ALM dual update produced non-finite multipliers at indices {bad}"
        )
    if multiplier_max is None:
        return updated, False, []
    cap = float(multiplier_max)
    cap_binding_mask = updated > cap
    return (
        np.minimum(updated, cap),
        bool(np.any(cap_binding_mask)),
        np.flatnonzero(cap_binding_mask).tolist(),
    )
```

### F4: `ALMSettings.__post_init__` and `validate_alm_cli_args` accept NaN silently [MEDIUM]
- File: `examples/single_stage_optimization/alm_utils.py:37-77` and `389-432`
- Code (L41-77, condensed):
```python
    if self.max_outer_iterations <= 0:
        raise ValueError(...)
    if self.penalty_init <= 0.0:
        raise ValueError(...)
    if self.penalty_scale <= 1.0:
        raise ValueError(...)
    if self.penalty_max is not None and self.penalty_max <= 0.0:
        raise ValueError(...)
    if self.feasibility_tol <= 0.0:
        raise ValueError(...)
    if self.stationarity_tol <= 0.0:
        raise ValueError(...)
    if self.trust_radius_init is not None and self.trust_radius_init < 0.0:
        raise ValueError(...)
    if self.trust_radius_min <= 0.0:
        raise ValueError(...)
    if not (0.0 < self.trust_radius_shrink < 1.0):
        raise ValueError(...)
    if self.trust_radius_grow <= 1.0:
        raise ValueError(...)
    if self.multiplier_max is not None and self.multiplier_max <= 0.0:
        raise ValueError(...)
```
- Bug: Every `<=` / `<` / `>` test against a constant returns False when one
  operand is NaN. So `ALMSettings(penalty_max=float('nan'))`,
  `feasibility_tol=float('nan')`, `multiplier_max=float('nan')`, etc. all pass
  validation. The single test that would correctly reject NaN is
  `0.0 < self.trust_radius_shrink < 1.0` (L68) — chained comparisons return
  False for NaN, but the reject condition is `not (...)` so NaN reaches
  through.
- Why: NumPy + Python: `nan <= 0.0 = False`, `nan > 1.0 = False`,
  `0.0 < nan < 1.0 = False`. Empirical: `python3 -c "print(float('nan') <=
  0.0)"` ⇒ `False`.
- Impact:
  - `penalty_init=NaN`: defers to L1685 `_penalty_values` raise
    (`ALM penalty values must be finite and positive`) on first invocation —
    safe, but error message points at the inner call, not the constructor.
  - `feasibility_tol=NaN`: comparisons like `max_violation <= feasibility_tol`
    silently return False forever; ALM never converges and runs to
    `max_outer_iterations`. Wastes inner solves; no clear error.
  - `penalty_max=NaN`: `_next_penalty` (L1722) `requested > NaN = False`, so
    the cap is never enforced. Penalty grows unboundedly until float64
    overflow.
  - `multiplier_max=NaN`: same — cap never enforced.
- Repro:
```python
ALMSettings(feasibility_tol=float('nan'))  # accepts; will silently misbehave
```
- Suggested fix: replace each comparison with an explicit positive-finite
  guard helper:
```python
def _require_positive_finite(name: str, value: float) -> None:
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
```
  and call from both `ALMSettings.__post_init__` and `validate_alm_cli_args`
  for each validated knob. This is the same shape as the existing
  `require_positive_alm_threshold` (L373-386) — extend its use.

### F5: `ALM_PHYSICAL_SCALE_FLOOR = sys.float_info.epsilon` is too low for inequality conditioning [MEDIUM]
- File: `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py:30`
- Code:
```python
ALM_PHYSICAL_SCALE_FLOOR = sys.float_info.epsilon
```
  used at L301:
```python
return resolve_alm_scale_with_provenance(
    candidate, ALM_PHYSICAL_SCALE_FLOOR, base_source
)
```
- Bug: A user-supplied threshold of (e.g.) `1e-15` m passes
  `require_positive_alm_threshold` and is then floored at
  `~2.22e-16`. Subsequent normalization divides constraint values by this
  ~`eps` scale, amplifying values by a factor of ~`4.5e15`. Combined with
  `multiplier_max=1e6` and `penalty_max=1e8`, the per-coordinate
  `positive_shift = max(0, λ + μ·c_normalized)` reaches
  `1e6 + 1e8 × 4.5e15 = 4.5e23`. The augmented term `(s² − λ²)/(2μ)`
  is then `~1e39` per constraint — within float64 range, but the gradient
  contribution `positive_shift × ∇c` has coordinates `~4.5e23`, dwarfing
  `base_grad` and rendering the L-BFGS-B step direction garbage.
- Why: A floor of `eps` only prevents division-by-zero, not "division-by-
  numerically-meaningless-quantity". The objective floor
  `ALM_OBJECTIVE_SCALE_FLOOR = 1.0e-12`
  (`hardware_constraint_schema.py:31`) is applied to physics constraints
  (`single_stage_objectives.py:505`); the hardware floor is ~10⁴ × smaller
  by design ("physical scales are SI") but `eps` is not a meaningful physical
  scale.
- Impact: Soft failure mode. The ALM run will not crash but will fail to
  converge, with `multiplier_cap_binding=True` saturating early. Currently
  guarded *only* by the assumption that no Stage-2 hardware threshold is
  ever set below `1e-9` m or `1e-9` A (informal) — there is no contract.
- Repro:
```python
HardwareConstraintSpec(name="x", kind="lower_bound", threshold=1e-15, ...)
# scale floors to 2.22e-16; normalized constraint values are amplified by ~4.5e15
```
- Suggested fix: pick `1e-12` (matching objective floor) or compute a
  problem-relative floor (e.g. `max(eps, 1e-6 × characteristic_length_m)`).
  At minimum: log a warning when `scale_floor_applied=True` AND the floor
  amplifies signals by more than a configured factor (e.g. 1e10).

### F6: `_directional_taylor_result` divides by zero / NaN-propagates on epsilon=0 [LOW]
- File: `examples/single_stage_optimization/alm_utils.py:1849-1851`
- Code:
```python
    for epsilon in taylor_epsilons:
        step = float(epsilon) * unit_direction
        plus_eval = evaluate_problem(x + step, multiplier_array, penalty)
        minus_eval = evaluate_problem(x - step, multiplier_array, penalty)
        central_estimate = (
            float(plus_eval["total"]) - float(minus_eval["total"])
        ) / (2.0 * float(epsilon))
```
- Bug: `run_directional_taylor_test` validates `len(taylor_epsilons) > 0`
  (L1904) but does not validate per-element finiteness or non-zero. With
  `epsilon = 0.0`, `step = 0`, `plus_eval == minus_eval`, numerator = 0,
  denominator = 0 ⇒ `central_estimate = NaN`. With `epsilon = inf`, denominator
  is inf, central_estimate = 0 (assuming finite numerator). Both produce
  meaningless ratios that silently flow through to `direction_result`.
- Impact: Diagnostic-only — `run_directional_taylor_test` is the gradient
  Taylor test, not part of the ALM solve. But the test will return
  `passed=True` for an obviously broken epsilon list.
- Repro:
```python
run_directional_taylor_test(evaluator, x0, multipliers, penalty=1.0,
                            epsilons=[0.0, 1e-3])
# returns NaN errors with passed=True
```
- Suggested fix:
```python
    if len(taylor_epsilons) == 0:
        raise ValueError("epsilons must be non-empty")
    if not all(np.isfinite(eps) and eps > 0.0 for eps in taylor_epsilons):
        raise ValueError("epsilons must all be finite and positive")
```

### F7: `alm_raw_dual_estimates` divides by `constraint_scales` without finite/positive validation [LOW]
- File: `examples/single_stage_optimization/alm_utils.py:696-704`
- Code:
```python
def alm_raw_dual_estimates(multipliers, evaluation: dict) -> list[float] | None:
    constraint_scales = evaluation.get("constraint_scales")
    if constraint_scales is None:
        return None
    scales = np.asarray(constraint_scales, dtype=float)
    multiplier_array = np.asarray(multipliers, dtype=float)
    if scales.shape != multiplier_array.shape:
        raise ValueError("constraint_scales shape must match multipliers")
    return _as_float_list(multiplier_array / scales)
```
- Bug: Unlike `normalize_alm_constraint_signals` (L516-518) and
  `normalize_alm_constraint_grads` (L536-538), this function does not check
  `np.any(~np.isfinite(scale_array)) or np.any(scale_array <= 0.0)`. If
  `evaluation["constraint_scales"]` (sourced from
  `alm_constraint_metadata_payload`) contains any `0.0` or negative value, the
  division silently produces ±inf or 0/0 = NaN, which is then encoded into the
  history snapshot.
- Why: The upstream constructors guard scale positivity (the M7 commit
  `a169f996a` added `require_positive_alm_threshold` and the schema-level
  `_validate_alm_metadata`). But `alm_raw_dual_estimates` is the public API
  consumer — defense-in-depth requires validating at the boundary even if
  callers are believed safe.
- Impact: LOW. With current call sites, scales are always validated upstream.
  But this is the documented public surface, and a future refactor that
  introduces an unvalidated path (e.g., reading scales from a JSON sidecar)
  would silently emit garbage in `raw_dual_estimates`.
- Suggested fix: extend the existing validator (L517-518) into a shared helper:
```python
def _validate_constraint_scales_in_place(scales: np.ndarray) -> None:
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("ALM constraint scales must be finite and positive")
```
  and call it from L700.

### F8: `_kkt_stationarity_norm` activity rule diverges from `_constraint_activity_mask` [LOW]
- File: `examples/single_stage_optimization/alm_utils.py:2156-2167` and `2053-2063`
- Code (`_kkt_stationarity_norm` activity rule):
```python
    for constraint_grad, constraint_value, _feasibility_value, activity_tolerance in zip(
        constraint_grads,
        constraint_values,
        feasibility_values,
        activity_tolerances,
    ):
        if float(constraint_value) < -float(activity_tolerance):
            continue
        active_constraint_grads.append(np.asarray(constraint_grad, dtype=float).reshape(-1))
```
  vs (`_constraint_activity_mask`):
```python
    return np.logical_and(
        np.asarray(feasibility_values, dtype=float) <= float(feasibility_gate),
        np.asarray(constraint_values, dtype=float)
        >= -np.asarray(activity_tolerances, dtype=float),
    )
```
- Bug: The KKT-residual function ignores both `feasibility_values` and
  `feasibility_gate` in its activity test (despite receiving them as
  parameters), while `_constraint_activity_mask` (used by the routing layer
  at L2090-2103) requires *both* feasibility-gate-passing and
  activity-tolerance-passing. So a constraint that is far above the
  feasibility gate is *not* in the routing's "active" set but *is* in the
  KKT diagnostic's "active" set. The two diagnostics tell inconsistent
  stories about the same iterate.
- Why: Possibly intentional ("KKT residual considers all not-strictly-inactive
  constraints"), but the prior audit noted no design rationale and the
  unused parameters (`feasibility_values`, `feasibility_gate`) suggest a
  refactor that left the parameters in place after removing the test.
- Impact: LOW. Diagnostic inconsistency, not a numerical fault. The KKT
  residual will be **larger** than the routing-active version for any
  constraint that's marginally violated above the feasibility gate. This
  could trigger false-positive "KKT not converged" reads.
- Suggested fix: either remove the unused parameters and document the
  "all-not-strictly-inactive" rule, or align both definitions:
```python
    for constraint_grad, constraint_value, feasibility_value, activity_tolerance in zip(
        constraint_grads, constraint_values, feasibility_values, activity_tolerances,
    ):
        feasible_under_gate = float(feasibility_value) <= float(feasibility_gate)
        active_band = float(constraint_value) >= -float(activity_tolerance)
        if not (feasible_under_gate and active_band):
            continue
        active_constraint_grads.append(np.asarray(constraint_grad, dtype=float).reshape(-1))
```

### F9: `_normalize_trust_radius` accepts `+inf`, silently routes through "boxed" inner profile [LOW]
- File: `examples/single_stage_optimization/alm_utils.py:1567-1573` and `1604-1614`
- Code:
```python
def _normalize_trust_radius(trust_radius: float | None) -> float | None:
    if trust_radius is None:
        return None
    normalized = float(trust_radius)
    if normalized <= 0.0:
        return None
    return normalized
```
  ```python
def _select_inner_solve_profile(
    *,
    trust_radius: float | None,
    continuation_iteration: int,
    feasible_enough: bool,
) -> ALMInnerSolveProfile:
    if _normalize_trust_radius(trust_radius) is None:
        return _UNBOUNDED_INNER_PROFILE
    return _BOXED_INNER_PROFILES[(bool(feasible_enough), bool(continuation_iteration > 0))]
```
- Bug: `+inf` passes the `<= 0.0` guard. `_select_inner_solve_profile` then
  selects a "boxed" profile (with `maxiter_cap`, `maxls_cap`, `ftol_floor`)
  even though `_build_box_bounds` produces effectively unbounded bounds
  (`(value - inf, value + inf) = (-inf, +inf)`, which scipy treats as
  unbounded). The boxed profile's caps are tighter than the unbounded
  profile, so an `inf` trust radius silently constrains iteration counts
  and `ftol` more aggressively than `None` does — the opposite of the
  user's intent.
- Why: `__post_init__` only checks `trust_radius_init < 0.0` (L64), not
  `not np.isfinite(trust_radius_init)`.
- Impact: LOW. Users who pass `inf` trust radius are uncommon, and the
  symptom is "tighter caps than expected", not numerical incorrectness.
- Repro:
```python
ALMSettings(trust_radius_init=float('inf'))  # accepted; later boxed_* profile applied
```
- Suggested fix:
```python
def _normalize_trust_radius(trust_radius: float | None) -> float | None:
    if trust_radius is None:
        return None
    normalized = float(trust_radius)
    if not np.isfinite(normalized) or normalized <= 0.0:
        return None
    return normalized
```
  and add `if not np.isfinite(self.trust_radius_init):` to
  `ALMSettings.__post_init__`.

## Confirmed-Correct Items

The following surfaces were specifically inspected and found to behave
correctly under the current contract:

1. **M6 nnls `RuntimeError` catch** (L2175-2188). `nnls` is wrapped with
   `try/except RuntimeError`, `maxiter=10*active_matrix.shape[1]` is
   passed, and the failure mode returns `None` (caller-tolerated). However,
   `nnls` raises `ValueError` (not `RuntimeError`) when the inputs contain
   NaN/Inf (verified empirically); the catch does not cover that mode.
   Today this is unreachable because `_require_finite_evaluation` and
   `_sanitize_nonfinite_inner_evaluation` guarantee finite inputs upstream
   of every `_kkt_stationarity_norm` call site, but this is the same
   defense-in-depth gap as F2/F3.

2. **`_lbfgsb_projected_gradient_max_norm` projection rule** (L855-873).
   The boundary projection sets `projected_gradient[i] = 0` only when the
   coordinate is **at or beyond** the bound and the gradient pushes
   *outward* (positive at lower bound, negative at upper bound). `None`
   bounds correctly skip projection (`at_lower_bound = lower_bound is not
   None and ...` short-circuits). Test
   `test_lbfgsb_projected_gradient_max_norm_uses_projected_infinity_norm`
   (test_alm_utils.py:751-761) pins this exact behavior with mixed
   `(0, 1)` and `(None, None)` bounds. **No bug.**

3. **`_penalty_values` finiteness guard** (L1685-1686). `np.any(~np.isfinite(values)) or np.any(values <= 0.0)` correctly rejects NaN, Inf,
   zero, and negatives. The `<= 0.0` test still has NaN-permissive behavior
   (`np.any([NaN <= 0.0]) == np.any([False]) == False`), but the
   `np.any(~np.isfinite(values))` is checked first and *is* NaN-correct
   (`np.isfinite(NaN) == False`, so `~np.isfinite(NaN) == True`). The
   composite check therefore rejects NaN. **No bug.**

4. **`_next_penalty` overflow handling** (L1714-1724). `np.isfinite(requested_penalty)`
   correctly catches multiplication overflow (`finite * finite_scale = inf`)
   and returns the input penalty unchanged with `cap_hit=True`. With
   `penalty_max=None`, also handled. With finite `penalty_max`, overflow
   returns `(penalty_max, True, requested_penalty)`. **No bug.**

5. **`_acceptable_total_upper_bound` scale defense** (L1461-1465). Using
   `total_scale = max(eps, |current_total|)` prevents the relative-tolerance
   term from collapsing to zero when `current_total ≈ 0`. With `current_total
   = NaN`, the sum yields NaN and `candidate <= NaN` is False, so the
   candidate is rejected (conservative). **No bug.**

6. **`_elevated_rejection_total` strict inflation** (L1302-1303). The
   elevated total `current + max(|current|, 1.0) + ATOL` is strictly larger
   than `_acceptable_total_upper_bound(current)`, so a sanitize fallback is
   guaranteed to be rejected. With `current = +inf`, `inf + inf + ATOL = inf`
   (still rejected because `inf <= inf` would be True; but
   `_require_finite_evaluation` prevents `current = inf`). **No bug.**

7. **`validate_initial_multipliers` boundary guard** (L2272-2295). Shape,
   finiteness (`np.isfinite(arr).all()`), and non-negativity (`(arr <
   0.0).any()`) are checked. `np.isfinite` is NaN-correct. `arr < 0.0`
   silently passes NaN (NaN < 0 is False), but the prior `np.isfinite`
   check already rejected NaN. **No bug.**

8. **`_normalize_alm_run_inputs` initial penalty validation** (L2329-2335).
   `not np.isfinite(penalty) or penalty <= 0.0` correctly rejects NaN
   (because `np.isfinite(NaN) == False`). **No bug.**

9. **Penalty schedule tolerance arithmetic** (L1697-1705). Receives a
   scalar penalty (validated upstream as `>= settings.penalty_init > 0`).
   `1.0 / penalty` cannot divide by zero. Returned tolerance is `max(input_tol,
   1.0/penalty) >= input_tol > 0`. **No bug.**

10. **`_build_inner_options` gtol/ftol clamps** (L1735-1739, L1752).
    `staged_gtol = max(eps, min(1e-4, 0.1 * tol))` clamps below at `eps`
    even when `tol` is denormal. `options["ftol"] = max(base_ftol,
    profile.ftol_floor)` ensures the floor is applied. **No bug.**

11. **`_normalized_taylor_directions` zero-direction rejection** (L1819-1821).
    `direction_norm <= np.finfo(float).eps` correctly rejects zero and
    near-zero norms. With `direction = [inf, 0, 0]`, `np.linalg.norm = inf`,
    `inf <= eps` is False, so the function returns `direction / inf` =
    `[NaN, 0, 0]` — but this only matters for the diagnostic Taylor test
    (F6 covers the related epsilon issue). The norm guard itself is correct
    for finite inputs.

12. **`_run_alm_inner_attempts` NaN-x handling** (L3258-3284). When
    `result.x` from scipy contains NaN, `np.array_equal(NaN_x, cached_x)`
    returns False (NumPy default `equal_nan=False`), so cache is bypassed
    and `evaluate_problem(NaN_x, ...)` is called, returning a non-finite
    evaluation that is sanitized to a fallback with elevated total. The
    candidate is then rejected by `_candidate_is_acceptable`, classified as
    `infeasible_inner_stall`, and the inner-attempt loop accepts
    `request.x` (the previous finite iterate). **No bug.**

13. **`_clone_evaluation_dict` ownership semantics** (L1306-1321). Every
    `np.ndarray` value is `.copy()`-ed; the `constraint_grads` list is
    rebuilt with each element copied. `_sanitize_nonfinite_inner_evaluation`
    (L1324-1351) calls this on the no-invalid-field fast path (L1334) and
    rebuilds owned copies on the sanitize-fallback path. **No bug.**

14. **`_build_box_bounds` width construction** (L1589-1601). `widths =
    trust_radius * np.maximum(1.0, |center|)` correctly scales the box width
    by the per-coordinate magnitude, with a floor of `trust_radius * 1.0`.
    With `trust_radius = +inf`, widths are inf and bounds are `(-inf, +inf)`
    — see F9. With `trust_radius = NaN`, `_normalize_trust_radius` does not
    catch NaN (also F9-adjacent: `NaN <= 0.0` is False, so NaN passes).
    Today this is unreachable because no caller produces NaN trust radius.

## Verdict

3 HIGH (F1, F2, F3), 2 MEDIUM (F4, F5), 4 LOW (F6, F7, F8, F9). The HIGH
items form a single causally-linked path: Stage-2 evaluator emits NaN in
a field excluded from the finite-check (F2) → NaN propagates through the
multiplier projection without tripping the cap diagnostic (F3) → KKT
diagnostic (F1) reads near-zero regardless. Recommend fixing F2 + F3
together (one whitelist entry plus one finiteness assert), and fixing F1
in the same change as M9's mirror (L835-852). F4 should add a single
shared positive-finite helper used at both `__post_init__` and CLI args
boundaries. F5 is a contract-level decision and warrants discussion with
hardware constraints owners. F6-F9 are independent and can be addressed
opportunistically.
