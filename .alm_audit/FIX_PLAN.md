# ALM Mode — Confirmed Issues & Fix Plan (v2)

**Audit lineage**:
- 8-agent breadth audit (2026-05-08) producing `.alm_audit/{math,algorithm,physics,numerics,regression,normalization,test_coverage,codex_independent}_review.md`.
- Codex `gpt-5.5` `xhigh` adversarial review of audit findings → `.alm_audit/ADVERSARIAL_REVIEW.md` (gpt-5.5 confirmed via `codex debug models`).
- Claude Code independent code-walk verification of every cited file:line.
- Adversarial review of FIX_PLAN.md v1 → `.alm_audit/FIX_PLAN_REVIEW.md` (Opus 4.7 fallback after codex network was blocked; 5 of 5 substantive findings independently spot-checked by Claude Code against current tree).
- v2 incorporates all v1 review revisions.

**Scope**: Augmented Lagrangian Method driver `examples/single_stage_optimization/alm_utils.py` + immediate consumers.
**Branch**: `surrogate-confinement-v2`.
**Repo guardrails honored**: SSOT, DRY, SOLID, KISS, YAGNI; no dynamic imports; no `cast(Any)`; no defensive try/except beyond what is required by a real crash surface; production-grade; no silent config rewrites.

## What changed v1 → v2

| ID | v1 → v2 |
|---|---|
| H1 | Reframed: `validate_single_stage_alm_formulation_args` already rejects negative for `thresholded_physics` (`single_stage_banana_example.py:2964-2970`); live bugs are zero-acceptance, three independent metadata-constructor floor sites, and `weighted_sum` mode skipping the formulation validator. |
| S1 | Scope narrowed to `weighted_sum + alm` only. Auto-zero replaced with strict raise — no hidden config rewrite. |
| M2 | Tests extended to cover both call paths of `_emit_alm_subproblem_continue` (signal-mismatch + feasible-update retry). |
| M3 | Dual-update-before-BREAK_OUTER (which violated the existing stationarity gate at `alm_utils.py:4228`) replaced with: route `max_subproblem_continuations` exhaustion through existing penalty-increase arm. gtol-ratchet fix preserved. |
| M4 | Sticky `run_state.cap_binding_detected` replaced with current-iterate cap-active predicate. Both success arms (`converged` and `constraints_inactive_converged`) gated. |
| M6 | Catch narrowed to `RuntimeError` only. Explicit `maxiter` added. Test uses monkeypatched `nnls` to force the actual error class. |
| M7 | Fix surface extended to `hardware_constraint_schema.py:273-277, 330-334`. Dataclass field default appended at end for backward-compat. |
| M8 | Validation moved to driver boundary `_normalize_alm_run_inputs` (alm_utils.py:~2171). Resume validator delegates to the same helper. |
| L2 | "Drop the mutation" replaced with explicit contract decision: `_emit_alm_history_snapshot` passes a copy; identity test at `test_alm_utils.py:2390` updated. |
| T1 | KKT math corrected (λ = 0.5). Normalization-invariance assertion corrected (raw = λ_norm / scale). Sign-flip test strengthened to verify the final point under the original-orientation constraint. |
| **NEW** | M9 (was math F4): `_kkt_stationarity_norm` uses `metric_grad`, not bare `base_grad` — diagnostic semantics fix. |
| **NEW** | M10 (was physics F2): Stage 2 `--stage2-iota-tolerance` vs single-stage `--alm-iota-penalty-threshold` operator-facing unit mismatch. |
| **NEW** | L4 (was numerics 1.3, 1.4): mutable-alias contracts in `_sanitize_nonfinite_inner_evaluation` and `_build_augmented_evaluation`. |
| **NEW** | L5 (was regression L21): `trust_radius_grow` validated CLI-only; programmatic `ALMSettings(trust_radius_grow=0.5)` silently shrinks. |
| **NEW** | T2: best-feasible monotonicity, penalty-schedule monotonicity over many infeasible outers, full-driver stall-class coverage, augmented-gradient Taylor consistency (test_coverage items 6-8, 10). |

## Severity ladder (v2)

| ID | Title | Severity | Effort (revised) |
|---|---|---|---|
| H1 | Zero/floor in ALM threshold validation | HIGH | 1.5-2 hr |
| S1 | `weighted_sum` + ALM coil-length double-feed | SUB-MAJOR | 3-5 hr |
| M1 | Surrogate-vs-hard hybrid signal | MEDIUM | ~3 hr (design) |
| M2 | Final-outer termination-reason mislabel | MEDIUM | ~45 min |
| M3 | Subproblem-limit `max_continuations` exhaustion routed to penalty arm + gtol ratchet | MEDIUM | 3-5 hr |
| M4 | Cap-active predicate gates both success arms | MEDIUM | 2-4 hr |
| M5 | Routing tolerance divergence within outer step | MEDIUM | ~45 min |
| M6 | `_kkt_stationarity_norm` `nnls` `RuntimeError` only | MEDIUM | 1-2 hr |
| M7 | Scale floor recorded in metadata across all sites | MEDIUM | 2-4 hr |
| M8 | ALM driver-boundary multiplier validation | MEDIUM | 1.5-3 hr |
| M9 | `_kkt_stationarity_norm` uses `base_grad`, not `metric_grad` | MEDIUM | ~30 min |
| M10 | Iota threshold operator-facing units consistency | MEDIUM | ~1 hr |
| L1 | `_attach_alm_constraint_metadata` ownership symmetry | LOW | ~15 min |
| L2 | History callback ownership contract | LOW | 2-4 hr |
| L3 | Inner-evaluator cache copy at boundary | LOW | ~30 min |
| L4 | Aliasing in `_sanitize_nonfinite_inner_evaluation` and `_build_augmented_evaluation` | LOW | ~45 min |
| L5 | `ALMSettings(trust_radius_grow)` programmatic guard | LOW | ~15 min |
| T1 | KKT/invariance/sign-flip/NaN/mismatch tests | STRUCTURAL | 4-8 hr |
| T2 | Monotonicity + stall-class + Taylor-consistency tests | STRUCTURAL | 3-5 hr |

## Recommended fix order (v2 — corrected)

The v1 order was wrong: T1 last meant the very tests needed to catch regressions in S1, M3, M4 weren't available when those landed. v2 lands characterization tests first.

1. **Characterization tests (corrected math)** — these protect every later fix:
   - T1.a: KKT fixture with corrected λ = 0.5.
   - T1.b: normalization-invariance with corrected raw = λ_norm / scale.
   - T1.c: sign-flip with original-orientation feasibility check.
   - T1.d: NaN-in-constraint deterministic exit.
   - T1.e: signal-mismatch deterministic termination.
   - M2 dual-call-path test.
   - M3 subproblem-limit characterization.
   - M4 cap-active characterization (both success arms).
   - L2 history-callback contract characterization.
   - T2 monotonicity + stall-class + Taylor characterizations.
2. **Validation/provenance**: H1, M7, M8, M10, L1, L3, L4, L5.
3. **Control-flow + diagnostics**: M5, M6, M9, M2, M4, corrected M3.
4. **Design-call fixes (after user input)**: S1, M1.
5. **Cross-cutting validation suite**.

---

## H1. Threshold validation — zero acceptance + floor sites + weighted_sum gap

**Severity**: HIGH

### Evidence (verified)

`single_stage_banana_example.py:2935, 2964-2970`:
```python
def validate_single_stage_alm_formulation_args(args):
    ...
    negative_thresholds = [
        flag_name for flag_name, value in required_thresholds.items() if float(value) < 0.0
    ]
    if negative_thresholds:
        raise ValueError(...)
```
Validator exists. `< 0.0` admits **zero**. And only fires for `thresholded_physics`.

`single_stage_banana_example.py:8083`: validator is called inside the `thresholded_physics` branch only.

`single_stage_objectives.py:503`: `scale=max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR)`.

`stage2_objectives.py:724`: `scale=max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR)`.

`hardware_constraint_schema.py:273-277`:
```python
def _resolved_alm_scale(spec, raw_threshold):
    return max(
        raw_threshold if spec.alm_scale is None else float(spec.alm_scale),
        ALM_PHYSICAL_SCALE_FLOOR,
    )
```
Third independent floor site missed by v1.

`single_stage_banana_example.py:1383-1413`: bare `type=float` CLI args, no per-arg validator.

### Root cause

Threshold positivity is the contract precondition (a normalization scale base must be strictly positive). Three issues stack:
1. Existing validator uses `< 0.0` instead of `<= 0.0` — zero passes, then is silently floored to 1e-12 ⇒ ~10¹² signal blow-up.
2. Existing validator only runs for `thresholded_physics`. `weighted_sum + alm` skips it; ALM threshold flags can still feed metadata constructors and floor.
3. Three independent floor sites (`single_stage_objectives.py`, `stage2_objectives.py`, `hardware_constraint_schema.py`) each apply the floor without enforcing positivity.

### Solution

Single shared helper in `alm_utils.py` enforces positivity at every boundary. Floor remains a defense-in-depth guarantee, never a silent recovery.

```python
# in alm_utils.py
def require_positive_alm_threshold(name: str, value: float | None) -> float | None:
    """None means constraint disabled. Otherwise must be strictly > 0."""
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"ALM threshold {name!r} must be a finite positive value; got {value!r}")
    return value_f
```

### Implementation checklist

- [ ] Add `require_positive_alm_threshold(name, value)` helper in `alm_utils.py`.
- [ ] In `validate_single_stage_alm_formulation_args` (`single_stage_banana_example.py:2935`): replace the inline `negative_thresholds` block with calls to the shared helper for each of the four flags.
- [ ] Add a top-level CLI validator hook in `validate_alm_cli_args` (`alm_utils.py:318`) that runs the helper for the four ALM threshold flags **regardless of formulation** (so `weighted_sum + alm` is also covered).
- [ ] In `_physics_alm_metadata` (`single_stage_objectives.py:495`): call helper before computing `scale = max(...)`. Floor becomes a no-op for valid input.
- [ ] In `_stage2_alm_constraint_metadata` (`stage2_objectives.py:702`): call helper for `iota_penalty_threshold`.
- [ ] In `_resolved_alm_scale` (`hardware_constraint_schema.py:273`): the `raw_threshold` argument cannot be None here; require `> 0`. The `spec.alm_scale` override path needs the same check.
- [ ] Document the floor as defense-in-depth only, not a recovery mechanism, in the helper docstring.
- [ ] Update CLI `--help` text on the four flags: "must be a finite positive value when set."

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | new helper near L318 | shared helper + integrate into `validate_alm_cli_args` |
| `single_stage_banana_example.py` | 2935-2970 | route through helper |
| `single_stage_banana_example.py` | 1383-1413 | help text refresh |
| `single_stage_objectives.py` | 495-513 | call helper |
| `stage2_objectives.py` | 694-734 | call helper |
| `hardware_constraint_schema.py` | 273-277 | call helper |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_require_positive_alm_threshold_rejects_zero_negative_nan_inf`.
- [ ] `tests/geo/test_alm_utils.py::test_require_positive_alm_threshold_accepts_none`.
- [ ] `tests/geo/test_alm_utils.py::test_validate_alm_cli_args_rejects_zero_thresholds_in_weighted_sum_mode`.
- [ ] `tests/geo/test_single_stage_example.py::test_validate_single_stage_alm_formulation_args_rejects_zero_thresholds`.
- [ ] `tests/geo/test_banana_objective_modules.py::test_physics_alm_metadata_rejects_zero_threshold`.
- [ ] Stage 2 mirror.
- [ ] Hardware schema mirror.

### Risks

- Saved configs with literal `0.0` thresholds become load-error after fix. Acceptable: they were yielding 10¹² inflated residuals.
- Float comparison: `0.0` parses cleanly. Subnormal positive values pass `> 0`; that's correct (real geometry tolerances can be ~1e-9).

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "threshold" -v
pytest tests/geo/test_banana_objective_modules.py -k "scale" -v
```

---

## S1. `weighted_sum` + ALM double-feeds coil-length penalty (strict raise, scoped)

**Severity**: SUB-MAJOR

### Evidence (verified)

`single_stage_banana_example.py:1472-1473`: `--length-weight` defaults to `1`.
`single_stage_banana_example.py:8332`: `LENGTH_WEIGHT = args.length_weight`.
`single_stage_banana_example.py:3706`: `JCurveLength = QuadraticPenalty(curvelength, length_target, "max")`.
`single_stage_banana_example.py:3720-3729`: `JF = build_total_objective(..., LENGTH_WEIGHT, JCurveLength, ...)`.
`single_stage_banana_example.py:4266-4267`: same `curvelength` and `length_target` passed into ALM constraint setup.

`single_stage_objectives.py:417-432`: `LENGTH_WEIGHT * JCurveLength` enters base objective only when `alm_formulation == "weighted_sum"`. For `thresholded_physics`, `evaluate_base_objective` returns `total=0.0`, `grad=0` — LENGTH_WEIGHT is irrelevant.

### Root cause

Coil-length penalty enters two paths simultaneously **only in `weighted_sum + alm` mode**:
- Soft term in base objective: `LENGTH_WEIGHT * QuadraticPenalty(curvelength, length_target, "max")`.
- ALM inequality: `length ≤ length_target` at the same threshold.

Both push toward the same boundary; the saved λ no longer represents the marginal cost of the ALM constraint.

`thresholded_physics + alm` is unaffected because LENGTH_WEIGHT does not enter `JF` in that mode.

### Solution

Strict raise, scoped to the actual buggy mode. No auto-zero, no warning, no hidden rewrite.

```python
# in validate_single_stage_alm_formulation_args
if (
    args.constraint_method == "alm"
    and args.alm_formulation == "weighted_sum"
    and float(args.length_weight) != 0.0
):
    raise ValueError(
        "ALM weighted_sum formulation owns the coil-length constraint; "
        "--length-weight must be 0 in this mode (got {args.length_weight}). "
        "Set --length-weight 0 explicitly, or use --alm-formulation thresholded_physics."
    )
```

### Implementation checklist

- [ ] Add the scoped raise in `validate_single_stage_alm_formulation_args` (`single_stage_banana_example.py:2935`).
- [ ] Audit wrappers/runners that programmatically construct args (`run_single_stage_frontier_campaign.py`, `run_single_stage_thresholded_physics_alm.py`, `workflow_runner_common.py`, `workflow_helpers.py`) for sites that set `length_weight` while selecting `weighted_sum + alm`. Update those sites to set `length_weight=0.0` explicitly.
- [ ] Default-value resolution: do **not** change the global default (1.0). Change only the conjunctive-mode requirement.
- [ ] Document in `examples/single_stage_optimization/README.md` that `--length-weight 0` is required for `--alm-formulation weighted_sum`.

### Files to change

| Path | Change |
|---|---|
| `single_stage_banana_example.py:2935` | scoped raise |
| `examples/single_stage_optimization/run_single_stage_frontier_campaign.py` | audit/fix |
| `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py` | not affected (thresholded_physics) — confirm |
| `examples/single_stage_optimization/workflow_runner_common.py` | audit/fix |
| `examples/single_stage_optimization/workflow_helpers.py` | audit/fix |
| `examples/single_stage_optimization/README.md` | document |

### Tests to add

- [ ] `tests/geo/test_single_stage_example.py::test_weighted_sum_alm_rejects_nonzero_length_weight`.
- [ ] `..._accepts_zero_length_weight`.
- [ ] `..._thresholded_physics_alm_unaffected_by_length_weight` — `--alm-formulation thresholded_physics` accepts any `length_weight`.
- [ ] `..._weighted_sum_no_alm_unaffected` — pure weighted_sum (no ALM) keeps the soft penalty.

### Risks

- **Run identity**: saved runs produced under `weighted_sum + alm + length_weight=1.0` had biased solutions. Their saved λ values for `coil_length_upper_bound` are misinterpretable. Flag in handoff so consumers (frontier reporting, autoresearch baselines) don't rely on those multipliers.
- **Backward-compat**: existing CLIs that set `length_weight=1.0 weighted_sum alm` will now fail. This is intentional — they were broken.

### Validation

```bash
pytest tests/geo/test_single_stage_example.py -k "length_weight" -v
# Smoke: existing weighted_sum alm runs with length_weight=0 still produce results
```

---

## M1. Surrogate-vs-hard hybrid signal

**Severity**: MEDIUM (unchanged from v1; APPROVED)

### Evidence (verified)

`stage2_objectives.py:1943-1972`: surrogate signed values feed augmented inequality objective; hard signed values stored as `hard_dual_update_values`.
`alm_utils.py:1915, 2946`: dual update consumes hard values.
`alm_utils.py:2005, 2020`: signal-mismatch detection.
`alm_utils.py:4091-4096`: `not signal_mismatch_active` is required for `converged`. False-success label blocked.

### Root cause

Inner solver minimizes augmented Lagrangian on surrogate (smoothed) signal; dual update uses hard (true) signal. Classical convergence theory does not apply. Engineering safety rail is the mismatch guard. Failure-labeling under sustained mismatch is not theory-backed.

### Solution

Document and accept (Option A). Inner objective on hard signals would be non-smooth; the surrogate exists for solver gradient stability. The safeguard is sufficient. Strengthen by adding a deterministic-termination property test under sustained mismatch.

### Implementation checklist

- [ ] Write `docs/alm_hybrid_signal_contract_2026-05-08.md`: specify which convergence theorems are forfeited, why the project accepts the trade-off (geometric constraints + smoothed gradients), the safeguards (mismatch detection + converged-gate guard), and the residual risk class (failure-label chatter).
- [ ] Add property test that runs ALM with sustained `signal_mismatch_active` and asserts: deterministic exit, deterministic termination reason, no chatter between continuation arms, no false-success label.

### Files to change

| Path | Change |
|---|---|
| `docs/alm_hybrid_signal_contract_2026-05-08.md` | new doc |
| `tests/geo/test_alm_utils.py` | property test |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_alm_terminates_deterministically_under_sustained_signal_mismatch`.

### Risks

- Documentation drift: future ALM refactors may forget the hybrid contract. The property test pins the behavior.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "signal_mismatch" -v
```

---

## M2. Final-outer termination-reason mislabel — both call paths

**Severity**: MEDIUM

### Evidence (verified)

`alm_utils.py:3457-3484`: `_emit_alm_subproblem_continue` takes no `is_final_outer` argument; sets `action="subproblem_continue"`; does not set `outer_termination`.

Two call sites (verified by codex review):
- `alm_utils.py:4221-4226` — signal-mismatch progress path.
- `alm_utils.py:4325-4330` — feasible-update retry path.

`alm_utils.py:1685-1696`: `_termination_reason_from_history` returns latest action verbatim if no `outer_termination` set.

### Solution

Add `is_final_outer: bool` parameter; set `outer_termination = "max_outer"` when true. Update both call sites.

### Implementation checklist

- [ ] Modify `_emit_alm_subproblem_continue` signature to accept `is_final_outer: bool`.
- [ ] In the helper, before snapshot emission: `if is_final_outer: history_entry["outer_termination"] = "max_outer"`.
- [ ] Update call site at L4221-4226 to pass `is_final_outer`.
- [ ] Update call site at L4325-4330 to pass `is_final_outer`.
- [ ] Verify via `rg "_emit_alm_subproblem_continue"` no third call site exists.

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 3457-3484 | add param |
| `alm_utils.py` | 4221-4226, 4325-4330 | pass param |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_subproblem_continue_signal_mismatch_path_final_outer_labels_max_outer`.
- [ ] `tests/geo/test_alm_utils.py::test_subproblem_continue_feasible_update_path_final_outer_labels_max_outer`.
- [ ] `tests/geo/test_alm_utils.py::test_subproblem_continue_midrun_label_unchanged`.

### Risks

- Saved-run analysis filtering on `termination_reason == "subproblem_continue"` may include or exclude runs differently. Acceptable: those runs were mislabeled.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "subproblem_continue" -v
```

---

## M3. `max_subproblem_continuations` exhaustion → penalty-increase + gtol-ratchet fix

**Severity**: MEDIUM

### Evidence (verified)

`alm_utils.py:4228`: dual-update arm requires `hard_feasible_for_update and stationarity_norm <= state.update_stationarity_tol`.
`alm_utils.py:4267-4317`: subproblem-limit branch is reached when stationarity gate failed; currently returns `BREAK_OUTER` without state advancement (multipliers unchanged, penalty unchanged).
`alm_utils.py:1640`: `options["gtol"] = max(base_gtol, staged_gtol)`.
`alm_utils.py:4001`: `state.inner_options = inner_attempt.last_inner_options` — staged value persists into `state`, becomes new `base` floor.

The v1 plan's "dual update before BREAK_OUTER" was wrong: it would update multipliers without the stationarity gate that the rest of the algorithm treats as mandatory.

### Solution

Two sub-fixes, independent:

**M3.a — `max_subproblem_continuations` routing**: When subproblem-limit triggers due to `max_subproblem_continuations` (not `_PLATEAU_STALL_LIMIT`), the inner subproblem is unsolved at requested tolerance — bumping the penalty (existing penalty-increase arm) is the correct ALM action. Plateau-stall continues to terminate.

**M3.b — gtol ratchet**: persist all of `last_inner_options` *except* `gtol` (and any other staged keys), so each outer iteration computes `staged_gtol` from a stable `base_gtol` baseline.

### Implementation checklist

#### M3.a

- [ ] In `alm_utils.py:4267-4317`, distinguish the two reasons:
  - If `state.feasible_stall_count >= _PLATEAU_STALL_LIMIT` ⇒ existing plateau-stall failure path (unchanged).
  - Else (i.e., `continuation_iteration == settings.max_subproblem_continuations`) ⇒ route to the existing penalty-increase arm. Use the same helper invoked elsewhere for penalty escalation (`rg "penalty_increase|_handle_alm_penalty" alm_utils.py` to identify exact name).
- [ ] Document the new label/action in `_emit_alm_history_snapshot` so the operator can see "subproblem_continuations exhausted → penalty bumped to X." Add a new history action string `"subproblem_limit_penalty_increase"`.
- [ ] Verify the penalty cap (`alm_utils.py:2970+ _handle_alm_penalty_cap_termination`) is reached cleanly when this new path repeatedly bumps μ.

#### M3.b

- [ ] In `alm_utils.py:_build_inner_options` (~L1628), document `gtol` as a "staged key."
- [ ] At persistence point `alm_utils.py:4001`, use:
  ```python
  state.inner_options = {
      k: v for k, v in inner_attempt.last_inner_options.items() if k != "gtol"
  }
  ```
- [ ] Audit other keys that `_build_inner_options` writes (`maxls`, `maxiter`, `maxfun`, `ftol`) — confirm whether they are staged or stable. Currently L1640 stages only `gtol`; L1641 stages `maxls` (`max(1, int(...))`); L1646-L1648 set `maxiter`/`maxfun` from caps. Drop staged keys uniformly.

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 4267-4317 | route to penalty-increase on `max_subproblem_continuations`; preserve plateau termination |
| `alm_utils.py` | 4001 | drop staged keys from persistence |
| `alm_utils.py` | 1628-1648 | identify staged keys in helper docstring |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_subproblem_limit_max_continuations_routes_to_penalty_increase` — drive ALM with `max_subproblem_continuations=1`, hard-feasible iterate, stationarity gate fails. Assert: penalty changed, `_PLATEAU_STALL_LIMIT` not triggered.
- [ ] `..._plateau_stall_terminates_unchanged` — drive plateau-stall path; assert termination as today.
- [ ] `..._gtol_does_not_ratchet_across_continuations` — sequence of inner solves; assert `gtol` follows the schedule, not bounded below by prior staged value.
- [ ] `..._penalty_cap_reached_cleanly_via_new_path` — drive enough penalty bumps to hit cap; assert clean failure with `termination_reason="penalty_cap_reached"`.

### Risks

- New penalty-increase calls may exhaust the penalty cap faster than today. That's correct behavior: the previous code was burning outer budget without progress. Operators may need to raise `--alm-penalty-max` for problems that legitimately need many subproblem continuations.
- The new history action `"subproblem_limit_penalty_increase"` adds a value to the action enum / set. Audit consumers (`rg '"subproblem_limit"|action.*subproblem' .`).

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "subproblem_limit or penalty_increase or gtol" -v
```

---

## M4. Multiplier-cap-active gates both success arms

**Severity**: MEDIUM

### Evidence (verified)

`alm_utils.py:4091-4118`: regular `converged` gate.
`alm_utils.py:4138-4161`: `constraints_inactive_converged` gate (second success arm).
`alm_utils.py:4241-4251`: dual-update produces `multiplier_cap_binding` (current iteration) and updates sticky `run_state.cap_binding_detected`.

The v1 plan only addressed the regular gate and used the sticky run-level flag, which would falsely fail later iterations after the cap unbound.

### Solution

Define the *current* cap-active predicate from the **last dual update's `multiplier_cap_binding`**, not the sticky run-state flag. Gate both success arms.

A multiplier is "currently cap-binding" iff (a) its value is at the cap **and** (b) the most recent attempted dual update tried to push it further positive (was clamped). The `dual_update.multiplier_cap_binding` flag at L4244 captures (b); (a) follows from the projection.

### Implementation checklist

- [ ] Carry `dual_update.multiplier_cap_binding` (the current-iteration flag) into the converged-gate scope. It is already attached to `history_entry["multiplier_cap_binding"]` at L4244 — propagate to the gate evaluation.
- [ ] At L4091-4096 add: `and not current_multiplier_cap_binding`.
- [ ] At L4138-4161 add the same gate.
- [ ] Add `_emit_alm_multiplier_cap_failure_step` (or extend an existing failure helper) returning result with `termination_reason="multiplier_cap_active"` and message listing binding indices.
- [ ] Sticky `run_state.cap_binding_detected` remains a diagnostic only (not used for gating).

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 4091-4118 | extend regular gate |
| `alm_utils.py` | 4138-4161 | extend constraints-inactive gate |
| `alm_utils.py` | new helper | failure path |
| `alm_utils.py` | gate-scope plumbing | propagate current flag |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_current_cap_active_blocks_regular_converged_label`.
- [ ] `..._blocks_constraints_inactive_converged_label`.
- [ ] `..._historical_cap_then_unbound_does_not_block_convergence` — multiplier was at cap at iter k, dual update at iter k+1 unclamped it. Convergence at iter k+2 should be allowed.
- [ ] `..._sticky_cap_diagnostic_remains` — `result.cap_binding_detected` should still report historical cap binding for post-run analysis.

### Risks

- Runs that previously labeled `converged` while at the cap will now label as failure. Correct behavior; flag for handoff.
- The constraints-inactive gate path is rare; verify the test fixture actually exercises it (otherwise the gate could regress).

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "cap_active or cap_binding" -v
```

---

## M5. Routing tolerance divergence within one outer iteration

**Severity**: MEDIUM (unchanged; APPROVED)

### Evidence (verified)

`alm_utils.py:3776-3790`: pre-inner uses **clamped** `effective_feasibility_tol`.
`alm_utils.py:3936-3941`: post-inner uses **raw** `state.update_feasibility_tol`.

### Solution

SSOT — compute the gate once per outer iteration; pass to both routing calls.

### Implementation checklist

- [ ] Confirm `_effective_feasibility_gate` is idempotent (calling on already-clamped value returns same value). If not, refactor.
- [ ] Replace `state.update_feasibility_tol` at L3940 with `effective_feasibility_tol` (the value computed at L3776-3778).

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 3936-3941 | use `effective_feasibility_tol` |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_routing_state_uses_effective_feasibility_gate_consistently`.

### Risks

- Behavior change in early iterations only.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "routing" -v
```

---

## M6. `_kkt_stationarity_norm` `nnls` `RuntimeError` only

**Severity**: MEDIUM

### Evidence (verified)

`alm_utils.py:2070-2073`:
```python
multipliers, _residual_norm = nnls(active_matrix, -total_grad_array)
residual = total_grad_array + active_matrix @ multipliers
return float(np.linalg.norm(residual))
```

SciPy `nnls` raises `RuntimeError("too many iterations")` on max-iter exhaustion. It does NOT raise `ValueError` on rank deficiency — it returns a least-squares answer.

The v1 plan caught `(RuntimeError, ValueError)`. Catching `ValueError` swallows shape/contract bugs.

### Solution

Catch only `RuntimeError`. Add explicit `maxiter` so the iteration cap is bounded and reproducible. Test by monkeypatching `nnls` to raise `RuntimeError`, plus a separate test ensuring shape errors still propagate.

### Implementation checklist

- [ ] Replace L2071 with:
  ```python
  try:
      multipliers, _residual_norm = nnls(
          active_matrix,
          -total_grad_array,
          maxiter=10 * active_matrix.shape[1],
      )
  except RuntimeError as exc:
      logger.warning(
          "kkt_stationarity_nnls_failure: shape=%s, exc=%s",
          active_matrix.shape, exc,
      )
      return None
  ```
- [ ] Add narrow comment justifying the `RuntimeError` catch (real crash surface in diagnostic helper, not defensive coding).
- [ ] Confirm `_stationarity_metrics` already handles `None` from this function (codex review confirmed it does).

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 2070-2073 | guarded nnls + maxiter |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_kkt_stationarity_norm_returns_none_on_nnls_runtime_error` — monkeypatch `alm_utils.nnls` to raise `RuntimeError("too many iterations")`; assert return is `None`, no raise.
- [ ] `..._shape_mismatch_still_raises` — pass `active_matrix` with wrong shape; assert `ValueError` propagates (not swallowed).

### Risks

- `maxiter` value: `10 * n_active_constraints` is generous. Tune if observed cap-hits indicate it's too tight.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "kkt_stationarity_norm" -v
```

---

## M7. Scale floor recorded across all metadata sites

**Severity**: MEDIUM

### Evidence (verified)

`single_stage_objectives.py:495-513` (physics metadata) — first floor site.
`stage2_objectives.py:702-734` (Stage 2 iota) — second floor site.
`hardware_constraint_schema.py:273-277` (`_resolved_alm_scale`) — third floor site, missed by v1.
`hardware_constraint_schema.py:330-334, 383-408` — emits source/payload without floor flag.
`hardware_constraint_schema.py:46-58` — `ALMConstraintMetadata` dataclass definition.

Direct constructors of `ALMConstraintMetadata` exist in tests and production; adding a required field would break them.

### Solution

One shared helper computes `(scale, floor_applied, source)`. Dataclass field added at end with default `False` for backward-compat.

```python
# In hardware_constraint_schema.py near ALMConstraintMetadata:
def resolve_alm_scale_with_provenance(
    raw_value: float,
    floor: float,
    base_source: str,
) -> tuple[float, bool, str]:
    floor_applied = raw_value < floor
    scale = floor if floor_applied else raw_value
    source = f"{base_source}:floored" if floor_applied else base_source
    return scale, floor_applied, source
```

### Implementation checklist

- [ ] Add `scale_floor_applied: bool = False` at the **end** of `ALMConstraintMetadata` dataclass (`hardware_constraint_schema.py:46-58`).
- [ ] Add `resolve_alm_scale_with_provenance(raw_value, floor, base_source)` helper (or two helpers if `ALM_OBJECTIVE_SCALE_FLOOR` and `ALM_PHYSICAL_SCALE_FLOOR` need different defaults; KISS prefers one helper with the floor passed in).
- [ ] Refactor `_physics_alm_metadata` (`single_stage_objectives.py:495`) to call helper.
- [ ] Refactor `_stage2_alm_constraint_metadata` (`stage2_objectives.py:702`) iota branch to call helper.
- [ ] Refactor `_resolved_alm_scale` (`hardware_constraint_schema.py:273-277`) to use helper. Update the surrounding metadata-emission code at L330-334, L383-408 to include `scale_floor_applied` and floored source.
- [ ] Update artifact payload emission to include the `scale_floor_applied` field where consumers read it.
- [ ] Audit run-summary printers (`rg "alm_scale|scale_floor" examples/single_stage_optimization/`) for surface in operator-facing reports.

### Files to change

| Path | Lines | Change |
|---|---|---|
| `hardware_constraint_schema.py` | 46-58 | add field with default |
| `hardware_constraint_schema.py` | new helper | shared `resolve_alm_scale_with_provenance` |
| `hardware_constraint_schema.py` | 273-277 | call helper |
| `hardware_constraint_schema.py` | 330-334, 383-408 | emit field |
| `single_stage_objectives.py` | 495-513 | call helper |
| `stage2_objectives.py` | 702-734 | call helper |
| run-summary printers | various | surface field |

### Tests to add

- [ ] `tests/geo/test_banana_objective_modules.py::test_physics_alm_metadata_records_scale_floor_application`.
- [ ] Stage 2 mirror.
- [ ] Hardware schema mirror.
- [ ] Backward-compat: existing `ALMConstraintMetadata(...)` constructors that do not pass `scale_floor_applied` still work and default to `False`.

### Risks

- Schema change: pickle/JSON readers must default `scale_floor_applied=False` on unpickle. Audit `rg "ALMConstraintMetadata.*pickle\|json" .`.

### Validation

```bash
pytest tests/geo/test_banana_objective_modules.py -k "scale_floor or scale or floor" -v
```

---

## M8. ALM driver-boundary multiplier validation

**Severity**: MEDIUM

### Evidence (verified)

`alm_utils.py:2171-2175` (`_normalize_alm_run_inputs`): copies `initial_multipliers` directly or initializes zeros — no validation.
`alm_utils.py:2160-2164`: already validates `penalty_max` here — same boundary, precedent for value validation.
`single_stage_banana_example.py:4342-4366` (`validate_resume_alm_state`): checks shape/names; not values.
`single_stage_banana_example.py:8661-8664, 8951-8954`: route checkpoint-resume state through that helper.

Two callers: resume JSON path AND direct `minimize_alm(initial_multipliers=...)` path. The v1 plan only fixed the first.

### Solution

Single helper at the ALM driver boundary. Resume validator delegates to it.

```python
# in alm_utils.py near _normalize_alm_run_inputs
def validate_initial_multipliers(multipliers, n_constraints: int) -> np.ndarray:
    arr = np.asarray(multipliers, dtype=float)
    if arr.shape != (n_constraints,):
        raise ValueError(
            f"ALM initial_multipliers shape {arr.shape} != ({n_constraints},)"
        )
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise ValueError(f"ALM initial_multipliers non-finite at indices {bad}")
    if (arr < 0).any():
        bad = np.where(arr < 0)[0].tolist()
        raise ValueError(f"ALM initial_multipliers negative at indices {bad}")
    return arr.copy()
```

### Implementation checklist

- [ ] Add `validate_initial_multipliers(multipliers, n_constraints)` in `alm_utils.py`.
- [ ] Add `validate_initial_penalty(penalty)` requiring finite `> 0`.
- [ ] Call both from `_normalize_alm_run_inputs` (`alm_utils.py:2171-2175`) replacing the unchecked `np.copy(initial_multipliers)`.
- [ ] Refactor `validate_resume_alm_state` (`single_stage_banana_example.py:4342-4366`) to delegate to the same helpers (after shape/names check, value validation goes through the new helpers).

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | new helpers + 2171-2175 | driver-boundary validation |
| `single_stage_banana_example.py` | 4342-4366 | delegate |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_minimize_alm_rejects_nan_initial_multipliers`.
- [ ] `..._rejects_negative_initial_multipliers`.
- [ ] `..._rejects_inf_initial_multipliers`.
- [ ] `..._rejects_wrong_shape_initial_multipliers`.
- [ ] `..._rejects_nonpositive_penalty`.
- [ ] `..._rejects_nonfinite_penalty`.
- [ ] `tests/geo/test_single_stage_example.py::test_validate_resume_alm_state_rejects_nan_multipliers` (delegated path).
- [ ] `..._rejects_negative_multipliers`.
- [ ] `..._rejects_nonpositive_penalty`.

### Risks

- Existing resume files with legitimately-zero multipliers (slack inequality fully inactive) must pass — `>= 0` check accepts them.
- Saved runs with corruption become un-resumable. Acceptable.

### Validation

```bash
pytest tests/geo/test_alm_utils.py tests/geo/test_single_stage_example.py -k "initial_multipliers or validate_resume_alm_state" -v
```

---

## M9. `_kkt_stationarity_norm` uses `metric_grad`, not bare `base_grad`

**Severity**: MEDIUM (was math `F4` finding; missed by v1)

### Evidence

`alm_utils.py:2041-2073` (`_kkt_stationarity_norm`): function consumes `total_grad` argument.
Per `math_review.md F4`: this is fed `metric_grad = ∇L_A` rather than `base_grad = ∇f`. As a result, near convergence the function collapses to `‖∇L_A‖` (which is zero when the inner solve has converged), masking the true KKT residual `‖∇f + Σλ_i∇c_i‖`.

### Root cause

KKT stationarity at the ALM solution is `‖∇f + Σλ_i∇c_i‖ = 0`. With augmented Lagrangian `L_A = f + Σλ_i c_i + (μ/2) Σ c_i²`, we have `∇L_A = ∇f + Σ(λ_i + μ c_i)∇c_i`. At an inner-converged iterate `∇L_A ≈ 0` regardless of multiplier quality. Using `∇L_A` as the input to `nnls` therefore gives a meaningless residual and does not measure KKT.

### Solution

Pass bare `∇f` (`base_grad`), not `∇L_A` (`metric_grad`), to `_kkt_stationarity_norm`.

### Implementation checklist

- [ ] Identify the call site of `_kkt_stationarity_norm` in `_stationarity_metrics` (`alm_utils.py:~2103-2111`).
- [ ] Confirm whether `base_grad` is available in scope (the evaluation dict carries `grad` for the augmented total; need a separate `base_grad` field).
- [ ] If `base_grad` is not currently exposed: extend the evaluation contract to include it. The objective evaluators (`stage2_objectives.py`, `single_stage_objectives.py`) compute `∇f` separately from the augmented term; ensure it is recorded in the evaluation dict.
- [ ] Update `_stationarity_metrics` to pass `base_grad` to `_kkt_stationarity_norm`.
- [ ] Confirm the diagnostic value reported in result + history actually shrinks toward zero only when KKT is satisfied (not just when inner-solve converged).

### Files to change

| Path | Lines | Change |
|---|---|---|
| Objective evaluators | various | record `base_grad` in evaluation dict |
| `alm_utils.py` | ~2103-2111 | pass `base_grad` |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_kkt_stationarity_norm_uses_base_grad_not_augmented` — construct an iterate where `∇L_A ≈ 0` but `∇f + Σλ∇c ≠ 0` (e.g., wrong multipliers); assert `_kkt_stationarity_norm` returns the larger value.

### Risks

- Result-format change: the diagnostic value will now be larger in many runs; flag for handoff so saved-run consumers re-baseline expectations.
- T1's KKT integration test (below) depends on this fix — order matters.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "kkt_stationarity" -v
```

---

## M10. Iota threshold operator-facing units consistency

**Severity**: MEDIUM (was physics `F2` finding; missed by v1)

### Evidence

Per `physics_review.md Finding 2`: `--stage2-iota-tolerance` and `--alm-iota-penalty-threshold` use different operator-facing semantics (deviation vs squared-penalty units). A user who expects equivalent numerical scale across modes is misled.

### Root cause

Stage 2 expresses iota constraint as `|iota - iota_target| ≤ tolerance` (deviation, dimensionless) — direct geometric meaning.
Single-stage ALM expresses it as `Jiota_penalty ≤ threshold` where `Jiota_penalty = (iota - iota_target)² * weight` (squared with weight) — magnitude depends on weight.

Setting `--alm-iota-penalty-threshold = 1e-3` does not correspond to a 1e-3 iota deviation.

### Solution

Two paths:

**Option A**: Express both in the same operator-facing unit (deviation). Internally Stage 2 keeps deviation, single-stage converts threshold to deviation by inverting the squared-penalty formula at metadata-construction time. Simpler operator UX.

**Option B**: Document the discrepancy clearly. Change CLI help to state the unit explicitly. Add operator-facing converter helper or example in README.

Option A is more user-friendly but requires understanding the squared-penalty weight convention. Pick A unless the weight depends on per-run state (in which case A becomes ambiguous).

### Implementation checklist (Option A)

- [ ] Decide option with user.
- [ ] If A: in `_physics_alm_metadata` (single-stage iota branch), convert operator-supplied deviation to internal squared-penalty threshold using the known weight. Update CLI help.
- [ ] If B: update CLI help text on both flags to state the unit; add README converter example.
- [ ] Add test that documents the chosen semantics.

### Files to change

| Path | Change |
|---|---|
| `single_stage_banana_example.py` | CLI help text |
| `single_stage_objectives.py` | iota metadata builder |
| `examples/single_stage_optimization/README.md` | document |

### Tests to add

- [ ] `tests/geo/test_single_stage_example.py::test_alm_iota_penalty_threshold_matches_stage2_iota_tolerance` — for the same target deviation, both modes produce the same active/inactive verdict.

### Risks

- Backward-compat: changing the unit interpretation means existing CLI invocations must be re-mapped.

### Validation

```bash
pytest tests/geo/test_single_stage_example.py -k "iota_penalty or iota_tolerance" -v
```

---

## L1. `_attach_alm_constraint_metadata` ownership symmetry

**Severity**: LOW (unchanged; APPROVED)

### Evidence (verified)

`alm_utils.py:2204-2214`: no-blocks lane returns original dict; blocks lane shallow-copies.

### Solution

Always shallow-copy.

### Implementation checklist

- [ ] Replace L2210 `return evaluation` with `return dict(evaluation)`.

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_attach_alm_constraint_metadata_always_returns_independent_dict`.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "attach_alm_constraint_metadata" -v
```

---

## L2. History callback ownership contract — explicit decision

**Severity**: LOW

### Evidence (verified)

`alm_utils.py:2379-2385`: docstring claims "history is borrowed/read-only ALM state."
`alm_utils.py:2151-2168`: ALM internals mutate `history_entry` in place (in flight).
`single_stage_banana_example.py:9015-9021`: callback writes back into `history[-1]`.
`single_stage_banana_example.py:6507-6535`: emits both `latest_history_entry` and full `history` in partial-state payloads — downstream payloads need `smoothing_changed` on history entries, not just on `latest_history_entry`.
`tests/geo/test_alm_utils.py:2390`: asserts callback `history` is the live `result.history` object.

The v1 plan's "drop the line" approach silently dropped `smoothing_changed` from emitted partial-state payloads.

### Solution

**Strict option (chosen)**: change `_emit_alm_history_snapshot` to pass a defensive copy of the history list. Update the identity test. Callbacks may mutate their copy. Single-stage callback's payload-emission code keeps its own owned partial-history copy with `smoothing_changed` written into the latest entry.

Rationale: ALM internal `result.history` should not be mutable through any callback — it's the authoritative run record.

### Implementation checklist

- [ ] Modify `_emit_alm_history_snapshot` (`alm_utils.py:~2379`) to pass `list(history)` instead of the live list. (Shallow list copy; entries within are still snapshotted at emission time via `_snapshot_history_entry`.)
- [ ] Update the docstring at L2379 to "history is a defensive shallow copy; mutations do not affect ALM internal state."
- [ ] Update test `tests/geo/test_alm_utils.py:2390` from identity-equality (`is`) to value-equality (`==`).
- [ ] In `single_stage_banana_example.py:9015-9021`, restructure to maintain `alm_partial_state["history"]` as an owned copy that includes the `smoothing_changed` augmentation. The existing `latest_history_entry["smoothing_changed"]` still flows through; the partial-state copy adds it to the latest entry of the owned history.
- [ ] Verify partial-state payload emission at `single_stage_banana_example.py:6507-6535` reads from the owned copy (still has `smoothing_changed`).

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 2379-2385 | defensive list copy |
| `single_stage_banana_example.py` | 9015-9021 | owned partial-history copy |
| `tests/geo/test_alm_utils.py` | 2390 | identity → value equality |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_history_callback_cannot_mutate_alm_internal_history` — callback attempts mutation of `history[-1]`; assert ALM internal `result.history` unchanged.
- [ ] `tests/geo/test_single_stage_example.py::test_partial_state_history_includes_smoothing_changed` — full pipeline; assert emitted partial-state payload's history entries carry `smoothing_changed`.

### Risks

- Performance: list copy per outer iteration. Trivial.
- Existing callers that rely on identity (`callback_history is result.history`) will break. The only known caller relying on identity is the test at L2390; after the test update, no other reliance.

### Validation

```bash
pytest tests/geo/test_alm_utils.py tests/geo/test_single_stage_example.py -k "history" -v
```

---

## L3. Inner-evaluator cache copy at boundary

**Severity**: LOW (unchanged; APPROVED)

### Evidence (verified)

`alm_utils.py:184`: `self.cached_evaluation = evaluation` (reference, not copy).
`alm_utils.py:495-510`: `np.asarray` non-copy semantics.
`alm_utils.py:2209-2210` (after L1 fix this is uniform copy).
`alm_utils.py:3105-3111, 3157-3161`: candidate-reuse path.

### Solution

Defensive copy at cache boundary.

### Implementation checklist

- [ ] Add `_clone_evaluation_for_cache(evaluation: dict) -> dict` in `alm_utils.py`: copy dict shell, deep-copy `np.ndarray` values (one level deep — KISS).
- [ ] Replace L184 `self.cached_evaluation = evaluation` with cloned version.

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_inner_evaluator_cache_does_not_alias_caller_buffers`.

### Risks

- Per-evaluation copy cost — non-trivial for large arrays. Profile if hot path slows.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "cache" -v
```

---

## L4. Mutable-alias contracts in `_sanitize_nonfinite_inner_evaluation` and `_build_augmented_evaluation`

**Severity**: LOW (was numerics `1.3, 1.4`; missed by v1)

### Evidence

Per `numerics_review.md`:
- `_sanitize_nonfinite_inner_evaluation` (`alm_utils.py:1228-1252`): shallow-copy whitelist for owned arrays; non-array fields are aliased.
- `_build_augmented_evaluation` (`alm_utils.py:495-525`): aliases caller's input arrays via `np.asarray`.

### Solution

For the shallow-copy whitelist, unify with the same `_clone_evaluation_for_cache` helper introduced in L3. For `_build_augmented_evaluation`, copy input arrays explicitly when they are stored in the returned evaluation.

### Implementation checklist

- [ ] In `_sanitize_nonfinite_inner_evaluation` (~L1228-1252): use the L3 helper for the dict shell.
- [ ] In `_build_augmented_evaluation` (~L495-525): for arrays that are stored in the returned dict (not just passed through to `np.linalg.norm` etc.), copy via `.copy()` after `np.asarray`.

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 1228-1252, 495-525 | uniform copy semantics |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_sanitize_nonfinite_returns_independent_dict`.
- [ ] `..._build_augmented_evaluation_does_not_alias_caller_arrays`.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "sanitize or build_augmented" -v
```

---

## L5. `ALMSettings(trust_radius_grow)` programmatic guard

**Severity**: LOW (was regression `21`; missed by v1)

### Evidence

`alm_utils.py:352-353`: `if trust_radius_grow is not None and trust_radius_grow <= 1.0: raise ValueError("--alm-trust-radius-grow must be greater than 1")` — CLI only.
`alm_utils.py:_grow_continuation_trust_radius`: relies on `trust_radius_grow > 1`. Programmatic `ALMSettings(trust_radius_grow=0.5)` silently shrinks instead of growing.

### Solution

Add `__post_init__` guard in `ALMSettings` (`alm_utils.py:16`).

### Implementation checklist

- [ ] In `ALMSettings.__post_init__` (or add one), validate `self.trust_radius_grow > 1.0` (when `trust_radius_grow` attribute exists; check the dataclass definition for whether it's optional).
- [ ] Validate other settings the CLI checks but the dataclass does not: `trust_radius_min > 0`, `trust_radius_shrink in (0, 1)`, `max_inner_attempts > 0`, `curvature_smoothing > 0`, `distance_smoothing > 0`, `penalty_init > 0`, `penalty_scale > 1`, `feasibility_tol > 0`, `stationarity_tol > 0`, `max_outer_iterations > 0`, `max_subproblem_continuations > 0`. Mirror the CLI validator at L318-359.

### Files to change

| Path | Lines | Change |
|---|---|---|
| `alm_utils.py` | 16 (ALMSettings) | `__post_init__` validation |

### Tests to add

- [ ] `tests/geo/test_alm_utils.py::test_alm_settings_rejects_invalid_trust_radius_grow`.
- [ ] `..._rejects_invalid_penalty_scale`.
- [ ] (etc. — one per validated field, parameterized.)

### Risks

- Existing programmatic constructors that pass an invalid value (currently no-op or silently broken) will now raise. Audit `rg "ALMSettings\(" .` for callers.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "alm_settings" -v
```

---

## T1. KKT / invariance / sign-flip / NaN / mismatch tests (corrected math)

**Severity**: STRUCTURAL

### Evidence

Original gap from `test_coverage_review.md`: no test asserts KKT on a `success=True` final iterate with active constraints. v1 had wrong KKT math (λ=1.0 instead of 0.5) and inverted normalization-invariance assertion.

### Solution

Single small problem with analytically known KKT solution that exercises all of: active inequality, non-zero `∇f`, non-zero `λ`, non-trivial penalty schedule.

### Fixture (corrected)

Minimize `f(x, y) = 0.5 * ((x-1)² + (y-1)²)` subject to `c(x, y) = x + y - 1 ≤ 0`.

KKT analysis:
- `∇f = (x-1, y-1)`.
- Active constraint at `x + y = 1`. Stationarity: `∇f + λ∇c = 0` ⇒ `(x-1, y-1) + λ(1, 1) = 0` ⇒ `x = y = 1 - λ`.
- Constraint active: `x + y = 1` ⇒ `2(1 - λ) = 1` ⇒ **λ = 0.5**.
- Minimizer: `(0.5, 0.5)`.

### Implementation checklist

- [ ] Implement the fixture as a real evaluator (no mock) in `tests/geo/test_alm_utils.py`. Exposes `f`, `∇f`, `c`, `∇c`, base+augmented evaluation contract that ALM expects.
- [ ] Run with `feasibility_tol=1e-8`, `stationarity_tol=1e-8`, `max_outer_iterations=20`, `penalty_init=1.0`, `penalty_scale=10.0`.
- [ ] Assert:
  - `result.success is True`.
  - `result.termination_reason == "converged"`.
  - `||x_final - (0.5, 0.5)||_∞ < 1e-6`.
  - `|λ_final - 0.5| < 1e-6`.
  - `λ_final >= 0`.
  - `result.kkt_stationarity_norm < 1e-7` (after M9 fix, this is bare-grad KKT).
  - `c(x_final) <= feasibility_tol`.

### T1.b — Normalization invariance (corrected)

Convention (per `normalization_review.md`): `λ_norm = λ_raw * scale` ⇒ raw multiplier = `λ_norm / scale`.

- [ ] Run the fixture with `scale_a = 0.1` and `scale_b = 10.0`. (Either via constraint normalization or by re-parameterizing the constraint.)
- [ ] Assert: `result_a.x ≈ result_b.x` (primal invariance).
- [ ] Assert: `result_a.multipliers / scale_a ≈ result_b.multipliers / scale_b` (raw-comparable invariance).

### T1.c — Sign-flip detection (strengthened)

A flipped constraint (`-c ≤ 0` instead of `c ≤ 0`) inverts the feasible region. The run can still "succeed" — but to the wrong feasible set.

- [ ] Run the fixture with the constraint hand-flipped: `c_flipped(x, y) = -(x + y - 1)`. Feasible region is now `x + y ≥ 1`; minimizer at `(0.5, 0.5)` is on the boundary but the fixture's inner solver may converge to a different point.
- [ ] Assert one of:
  - The run does not label `success`, OR
  - The run labels `success` but the final point **violates the original-orientation constraint**: i.e., assert `final_x + final_y > 1 + tolerance` (which proves the original `c ≤ 0` is violated).
- [ ] Document this as a sign-detector smoke test.

### T1.d — NaN-in-constraint deterministic exit

- [ ] Drive an evaluation that returns `NaN` in `constraint_values` once at iter 3.
- [ ] Assert run terminates with a deterministic NaN-detected `termination_reason`.
- [ ] Assert no NaN propagates to multipliers or x.

### T1.e — Signal-mismatch deterministic termination

- [ ] Drive an evaluator that returns surrogate vs hard signed values that disagree for all iterations.
- [ ] Assert run exits in O(`max_outer_iterations`) with a deterministic termination reason.
- [ ] Assert reason is stable across re-runs (no chatter).

### Files to change

| Path | Change |
|---|---|
| `tests/geo/test_alm_utils.py` | T1.a-e |
| `tests/geo/conftest.py` if needed | helper fixture |

### Tests to add

- [ ] `test_minimize_alm_converges_to_kkt_on_active_linear_inequality_lambda_half`.
- [ ] `test_minimize_alm_normalization_invariance_two_scales_raw_division`.
- [ ] `test_minimize_alm_sign_flipped_constraint_either_fails_or_violates_original`.
- [ ] `test_minimize_alm_handles_nan_in_constraint_values_deterministic_exit`.
- [ ] `test_minimize_alm_signal_mismatch_termination_deterministic`.

### Risks

- Tolerance pinning depends on actual ALM convergence rate. Tune if too tight.
- Per memory: shared test helpers per 2026-04-01 refactor; importlib only for strict isolation.
- T1.a depends on M9 (bare-grad KKT) — order matters.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "kkt or invariance or sign_flipped or nan_in_constraint or signal_mismatch_termination" -v
```

---

## T2. Monotonicity + stall-class + Taylor-consistency tests

**Severity**: STRUCTURAL (was `test_coverage_review.md` items 6-8, 10; missed by v1)

### Evidence

Per `test_coverage_review.md`:
- Item 6: best-feasible monotonicity not tested.
- Item 7: penalty-schedule monotonicity over many infeasible outers not tested.
- Item 8: full-driver stall-class coverage (all three: progress, infeasible, skipped-inner) not exercised in one integrated run.
- Item 10: augmented-gradient Taylor consistency (`(L_A(x+εd) - L_A(x))/ε ≈ ∇L_A(x)·d`) not tested.

### Implementation checklist

#### T2.a — Best-feasible monotonicity

- [ ] Drive a multi-outer-iteration run on a problem where the iterate path crosses feasible/infeasible boundaries.
- [ ] Track `result.best_feasible.objective` across outer iterations.
- [ ] Assert sequence is monotonic non-increasing.

#### T2.b — Penalty-schedule monotonicity over many infeasible outers

- [ ] Drive an infeasible-only run (e.g., `max_outer_iterations=10`, evaluator always reports infeasible, no progress).
- [ ] Assert `μ` is monotonic non-decreasing across outer iterations.
- [ ] Assert `μ` reaches the cap and termination is `penalty_cap_reached`.

#### T2.c — Full-driver stall-class coverage

- [ ] Single integrated run that exercises all three stall classes: progress (constraint norm flat), infeasible (no feasible direction), skipped-inner (already converged).
- [ ] Assert each class fires at least once and produces the expected `_ALMOuterDecision` enum value.

#### T2.d — Augmented-gradient Taylor consistency

- [ ] Pick a point `x`, direction `d`, compute `L_A(x)`, `L_A(x + ε d)`, `∇L_A(x)·d` for `ε ∈ {1e-3, 1e-4, 1e-5}`.
- [ ] Assert `|(L_A(x + ε d) - L_A(x))/ε - ∇L_A(x)·d|` is `O(ε)` (linear in ε).

### Files to change

| Path | Change |
|---|---|
| `tests/geo/test_alm_utils.py` | T2.a-d |

### Tests to add

- [ ] `test_minimize_alm_best_feasible_monotonic_nonincreasing`.
- [ ] `test_minimize_alm_penalty_monotonic_to_cap_on_infeasible`.
- [ ] `test_minimize_alm_full_driver_exercises_all_stall_classes`.
- [ ] `test_minimize_alm_augmented_gradient_taylor_consistency`.

### Validation

```bash
pytest tests/geo/test_alm_utils.py -k "monotonic or stall or taylor" -v
```

---

## Cross-cutting validation (after all fixes)

```bash
# Targeted ALM tests
pytest tests/geo/test_alm_utils.py tests/geo/test_alm_benchmarking.py tests/geo/test_alm_fixture_benchmarking.py tests/geo/test_single_stage_alm_integration.py -v

# Single-stage integration
pytest tests/geo/test_single_stage_example.py -v

# Banana objective modules
pytest tests/geo/test_banana_objective_modules.py -v

# Full geo suite
pytest tests/geo/ -v

# Lint
ruff check examples/single_stage_optimization/alm_utils.py examples/single_stage_optimization/banana_opt/

# Smoke run end-to-end
python examples/single_stage_optimization/run_alm_normalization_fixture_benchmark.py --quick
```

## Out-of-scope (deferred)

- The 4 known test failures + 3 parity gaps tracked in `project_known_issues.md`.
- Strict-CUDA / GPU-purity work on `gpu-purity-stage2-20260405` — this branch is pure NumPy/SciPy in the ALM path; no JAX dependencies in fixes.
- Block-penalty subsystem — phased out per `docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md`; do not extend or modify.

## Open design calls (require user input)

- **S1**: scoped raise as written (preferred), or option C (separate soft/hard length targets — only if physics demands).
- **M1**: Option A (document/accept hybrid) as written, or Option B (align inner objective to hard at iteration K).
- **M10**: Option A (unify operator-facing units) or Option B (document discrepancy).

## Cross-cutting correctness

| Pair | Coordination |
|---|---|
| H1 / M7 | Share scale-resolution helper contract — H1 enforces positivity, M7 records floor application. Cannot drift. |
| H1 / M8 | Both add boundary validation; reuse pattern (helper + delegate). |
| M2 / M3 | Both touch continuation/outer-loop dispatch. M2 is labeling fix on natural exhaustion; M3 is state-advancement on subproblem-limit. Independent but adjacent — implement separately. |
| M4 success arms | Must cover both `converged` and `constraints_inactive_converged`. |
| S1 / T1 | T1's KKT and normalization tests are the regression net for S1. Land T1 first. |
| M9 / T1 | T1.a's KKT assertion depends on M9 (bare-grad KKT). M9 lands before T1.a. |

## Effort summary (revised)

Total estimated: **35-65 hours** for the full set. v1 estimated ~15 hours, which was materially low due to missing items and underestimated complexity on S1, M3, M4, M6, M7, M8, L2, T1.

## References

- v1 audit reports: `.alm_audit/{math,algorithm,physics,numerics,regression,normalization,test_coverage,codex_independent}_review.md`.
- Adversarial review of audit findings: `.alm_audit/ADVERSARIAL_REVIEW.md` (codex `gpt-5.5` `xhigh`).
- Adversarial review of FIX_PLAN.md v1: `.alm_audit/FIX_PLAN_REVIEW.md` (Opus 4.7 fallback; spot-checked by Claude Code).
- Recent ALM commits (April-May 2026): `git log --oneline --all --since="2026-04-15" -- examples/single_stage_optimization/alm_utils.py`.
- Plans: `docs/alm_*.md`.
- Memory: `~/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/{project_known_issues,project_jax_port_status,feedback_test_importlib_pattern}.md`.
