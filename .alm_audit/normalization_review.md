# ALM Constraint Normalization & Block-Penalty Review

**Audit date:** 2026-05-08
**Repo:** `/Users/suhjungdae/code/columbia/simsopt-surrogate`
**Branch HEAD:** `surrogate-confinement-v2` (working tree)
**Scope:** scalar ALM normalization path, block-penalty residue, signed semantics, scale floors, diagnostics labels.

## Executive Summary

The ALM normalization rework lands **mostly correctly**. Block-penalty *control* code is fully removed (per the 2026-05-06 hardening plan); only legacy nullable schema fields (`block_penalties=None`, `ALM_BLOCK_PENALTIES=None`) and diagnostic-only `constraint_blocks` labels remain. The scalar penalty path is the single solver-decision path; `_penalty_values` still accepts a vector argument but the only production call site passes a scalar (vector branch retained for tests/`augmented_inequality_objective` flexibility, no runtime can reach it from `minimize_alm`).

The normalization contract is internally consistent for the project's chosen convention: solver math (objective, gradient, multipliers, dual update, KKT, stationarity, feasibility tolerance, activity tolerances) operates on `c_norm = c_raw / s` with `λ_norm = λ_raw · s`; `ρ` is **not** rescaled with `s²`. This is *not* full mathematical scale-invariance — it is "all internal arithmetic lives in normalized units" — but it is consistent end-to-end. Raw certification fields are preserved as explicit `raw_*` sidecars and never silently mixed with normalized solver values.

Three findings were identified; none are critical, but two are real correctness/diagnosability gaps that should be closed:

- **HIGH** — Negative physics-threshold inputs (`--alm-qs-threshold`, `--alm-boozer-threshold`, `--alm-iota-penalty-threshold`, `--alm-length-penalty-threshold`) pass through CLI/`_physics_alm_metadata` without validation and silently floor scale to `1e-12`, producing extreme blow-up in normalized signals.
- **MEDIUM** — When the scale floor wins (`raw_threshold < eps` or `raw_threshold ≤ 1e-12`), `ALMConstraintMetadata.source` continues to read `"threshold:<name>"`, hiding the floor decision from artifacts. The plan explicitly required the source to surface this.
- **LOW** — `_normalize_alm_run_inputs` does not check that `initial_multipliers.shape == (len(constraint_names),)`. The resume path (`validate_resume_alm_state`) does, but defense-in-depth at the ALM boundary is missing.

The benchmark fixture at `examples/single_stage_optimization/run_alm_normalization_fixture_benchmark.py` runs raw vs normalized formulations on two fixtures, but does **not** pin numerical AL values, multipliers, or dual estimates. It would not catch a silent normalization regression where `c̃ = c/s` survives but `λ̃ = λ·s` is broken (e.g. wrong sign of `s`, wrong scale-source picked). This is mentioned as an observation, not a finding.

## Findings

### F1 (HIGH) — Negative physics threshold passes silently and degrades scale to floor

**Files:**
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1383-1413` — CLI args `--alm-qs-threshold`, `--alm-boozer-threshold`, `--alm-iota-penalty-threshold`, `--alm-length-penalty-threshold` are typed as `float` with no positivity check.
- `examples/single_stage_optimization/alm_utils.py:318-359` — `validate_alm_cli_args` validates penalty/tol/trust-radius args but **not** physics thresholds.
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:495-513` — `_physics_alm_metadata` accepts any `threshold: float`; `scale = max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR)` floors negative or zero values without raising or marking the source.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:716-735` — same pattern for Stage 2 iota_penalty (Stage 2 elsewhere validates `iota_tolerance > 0` via `stage2_iota_penalty_threshold`, but the metadata builder itself does not enforce the sign of its input).

**Symptom:** Set `--alm-qs-threshold=-1.0`. `_physics_alm_metadata` produces `ALMConstraintMetadata(scale=1e-12, raw_threshold=-1.0, …)`. Then `c_norm = c_raw / 1e-12` is up to 1e12× the physical magnitude, blowing up the AL value, multiplier update, and KKT residual. Optimizer fails (or worse: succeeds but on the wrong problem) without a clear error pointing back to the user input.

**Root cause:** No positivity gate at any layer between argparse and `_resolved_alm_scale`. The floor was designed to handle "zero or tiny explicit thresholds" (per `docs/alm_constraint_normalization_block_penalty_impl_plan_2026-04-26.md` line 269), not negative values.

**Fix:** add `validate_alm_cli_args` checks for the four physics thresholds when `--alm-formulation=thresholded_physics`:
```python
for arg_name, value in (
    ("--alm-qs-threshold", args.alm_qs_threshold),
    ("--alm-boozer-threshold", args.alm_boozer_threshold),
    ("--alm-iota-penalty-threshold", args.alm_iota_penalty_threshold),
    ("--alm-length-penalty-threshold", args.alm_length_penalty_threshold),
):
    if value is not None and value <= 0.0:
        raise ValueError(f"{arg_name} must be positive when provided")
```
Belt-and-suspenders: also raise in `_physics_alm_metadata` and `_stage2_alm_constraint_metadata` when `raw_threshold < 0` (zero is intentional and gets the floor; negative is a config error).

### F2 (MEDIUM) — Scale source label hides floor decisions

**Files:**
- `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py:273-277` — `_resolved_alm_scale` returns `max(raw_threshold or spec.alm_scale, ALM_PHYSICAL_SCALE_FLOOR)` but no boolean indicating which branch won.
- `hardware_constraint_schema.py:330-334` — `source = "threshold:{schema_name}" if spec.alm_scale is None else "schema:{schema_name}.alm_scale"`.
- `single_stage_objectives.py:507`, `stage2_objectives.py:728` — physics metadata always sets `source = f"threshold:{name}"`.

**Symptom:** When `raw_threshold < ALM_PHYSICAL_SCALE_FLOOR` (= `sys.float_info.epsilon`) or `raw_threshold ≤ ALM_OBJECTIVE_SCALE_FLOOR` (= `1e-12`), the resulting metadata reports `scale=eps` (or `1e-12`) and `raw_threshold=<original>` but `source` continues to read `threshold:<name>` rather than `floored:<name>`. Artifacts and `ALM_CONSTRAINT_SCALE_SOURCES` therefore cannot distinguish a "real-world tiny threshold" run from a "user error floored to eps" run.

**Root cause:** The plan called this out (`docs/alm_constraint_normalization_block_penalty_impl_plan_2026-04-26.md` line 260: "Record the scale source in metadata so small positive floors are visible in artifacts"). The implementation correctly applies the floor but never sets the `source` to a floor label.

**Fix:** in both `_resolved_alm_scale` (return value plus a `floor_applied` bool) and `_physics_alm_metadata`, branch:
```python
chosen_scale = raw_threshold if spec.alm_scale is None else float(spec.alm_scale)
if chosen_scale <= ALM_PHYSICAL_SCALE_FLOOR:
    return ALM_PHYSICAL_SCALE_FLOOR, True   # (scale, floor_applied)
return chosen_scale, False
```
and propagate `source = "floored:<name>"` (or `"threshold:<name>+floor"`) when the floor wins. Update the metadata `source` enum-style validator to accept the new value.

### F3 (LOW) — `_normalize_alm_run_inputs` skips length check on `initial_multipliers`

**Files:** `examples/single_stage_optimization/alm_utils.py:2144-2201`.

**Symptom:** A caller passes `initial_multipliers=np.zeros(N)` where `N != len(constraint_names)`. `_normalize_alm_run_inputs` copies the array as-is (line 2171-2175). The mismatch surfaces later inside `augmented_inequality_objective` or the dual update via `_penalty_values(penalty, multipliers.size)` and broadcast errors — but the error stack is far from the offending input.

**Root cause:** The shape check is in `validate_resume_alm_state` (`single_stage_banana_example.py:4361-4366`) and not in the ALM-side normalization. Production code goes through `validate_resume_alm_state`, so this is currently unreachable from CLI but is a footgun for tests/scripts that build `minimize_alm` calls directly.

**Fix:** at line 2175 add:
```python
if multipliers.shape != (len(constraint_names),):
    raise ValueError(
        f"initial_multipliers length {multipliers.shape[0]} does not match "
        f"constraint count {len(constraint_names)}"
    )
```
Same defense-in-depth as `_normalize_alm_run_inputs`'s existing finite/positive penalty checks.

## Normalization Invariance Worked Example

Pick the banana_current constraint (scale `s = BANANA_CURRENT_HARD_LIMIT_A`, default `16000.0 A`, schema-floored). Suppose at iterate `x`:
- `c_raw(x) = |I| - I_max = 1000 A` (i.e. `I_max + 1000`).
- Surrogate gradient `g_raw = ∂c_raw/∂x = e_I` (unit vector along the current control coordinate).
- Current normalized multiplier `λ_norm = 4.0` (so `λ_raw = λ_norm/s = 2.5e-4`).
- Scalar penalty `ρ = 1.0`.

### Path through code (single-stage)

1. `evaluate_alm_objective` (`single_stage_objectives.py:599`) builds `signed_value = c_raw = 1000.0` and `grad = g_raw = e_I` for each active constraint (line 960-961).
2. `_single_stage_alm_constraint_metadata` (line 516) returns `ALMConstraintMetadata(scale=16000.0, block="current", source="threshold:banana_current_upper_bound", …)`. Validators (`hardware_constraint_schema.py:287-313`) confirm `scale > 0`, `activity_tolerance ≥ 0`, value-kind tuple compatible with the `current` block.
3. `normalize_alm_constraint_grads` and `normalize_alm_constraint_signals` (line 1039-1054) divide:
   - `c_norm = 1000 / 16000 = 0.0625`,
   - `g_norm = e_I / 16000`,
   - `feasibility_norm = max(c_norm, 0) = 0.0625`,
   - `activity_tolerance_norm = (16000 · 1e-3) / 16000 = 1e-3`.
   `normalize_alm_constraint_signals` (alm_utils.py:437-459) rejects `s ≤ 0` or non-finite (line 444-445), enforces shape match on every array, and never touches `λ` or `ρ`.
4. `augmented_inequality_objective` (alm_utils.py:374-414) consumes `(c_norm, g_norm, λ_norm, ρ)`:
   - `positive_shift = max(0, λ_norm + ρ · c_norm) = max(0, 4 + 0.0625) = 4.0625`.
   - `augmented_term = (4.0625² − 4²) / (2·1) = (16.504 − 16) / 2 = 0.252`.
   - Total grad contribution: `positive_shift · g_norm = 4.0625 · e_I / 16000 = 2.539e-4 · e_I`.
5. Solver advances using these normalized values; `feasibility_tol` is interpreted in normalized units; `_kkt_stationarity_norm` (alm_utils.py:2041-2073) uses `g_norm`.
6. Dual update (`_handle_alm_dual_update_transition`, alm_utils.py:2937-2967) projects `λ_norm ← max(0, λ_norm + ρ · c_norm) = 4.0625`. New `λ_raw = 4.0625/16000 = 2.539e-4`.
7. Final reporting (`_build_alm_result`, line 2580):
   - `multipliers = [4.0625]` (normalized, line 2720).
   - `raw_dual_estimates = λ/s = [2.539e-4]` (line 2715, calls `alm_raw_dual_estimates` → `multiplier_array / scales`, line 626).
   - `constraint_values = feasibility_norm = [0.0625]` (line 2658, normalized clipped).
   - `raw_constraint_values = [1000.0]` (line 2664, from `_explicit_raw_signed_constraint_values`).
   - `raw_hard_violation_values = [1000.0]` (line 2682).
   - `_alm_summary` (line 1067) reports `max_normalized_violation = 0.0625`, `max_raw_hard_violation_by_constraint = {"banana_current_upper_bound": 1000.0}`.

### Now repeat with `s' = 32000` (e.g. an off-spec override)

To preserve raw certification (`c_raw = 1000` should still register as a 1000 A overshoot):
- `c_norm' = 1000/32000 = 0.03125`. `g_norm' = e_I / 32000`.
- The corresponding equivalent `λ_norm'` if the optimizer reached the same raw dual estimate is `λ_norm' = λ_raw · s' = 2.5e-4 · 32000 = 8.0`.
- With `ρ = 1.0` (unchanged), `positive_shift' = max(0, 8 + 0.03125) = 8.03125`. Augmented term = `(8.03125² − 8²)/(2) = 0.2505`.
- Gradient contribution: `8.03125 · e_I / 32000 = 2.510e-4 · e_I` — within roundoff of the original `2.539e-4`.

The two paths produce **the same raw dual estimate and almost the same gradient contribution at infinite-precision invariance**, because the raw ρ-equivalent scaling is hidden inside positive_shift's structure (`λ + ρ·c → λ + ρ·c/s` vs `λ·s + ρ·c`). The small mismatch comes from `ρ` not transforming with `s²`; the convention is "ρ is a normalized-space tunable", not "ρ has raw-unit semantics".

This worked example confirms the project's contract:
- Raw certification (`c_raw`, `c_raw_hard_violation`) is invariant by construction (raw fields are preserved sidecars).
- Optimizer feasibility (`feasibility_tol` ≤ `c_norm`) is invariant.
- Multiplier physical interpretation (`λ_raw = λ_norm / s`) is invariant.
- AL value and gradient at a fixed iterate are *not* exactly invariant under a scale change with `ρ` held fixed, but the difference is bounded by the `ρ·c_norm` term, which is exactly the mechanism that adapts `λ` to the constraint scale across outer iterations.

## Verified Correct

- `normalize_alm_constraint_signals` and `normalize_alm_constraint_grads` reject `s ≤ 0` and non-finite scales (`alm_utils.py:444-445, 464-465`); negative *threshold* still slips through earlier (F1) but cannot reach normalization with `s ≤ 0`.
- Block-penalty *control* removal: `ALMBlockPenaltyState`, `_initial_block_penalty_state`, `_block_penalty_vector`, `_next_block_penalty_state`, and `block_penalties_enabled` are absent from `alm_utils.py`. Legacy nullable fields (`ALMResult.block_penalties=None`, `block_penalty_cap_reached=None`, `block_penalty_cap_requested=None`) and diagnostic `constraint_blocks` labels remain (`alm_utils.py:2725-2727, 2275, 2459`). `ALM_BLOCK_PENALTIES` artifact key still emitted as `None` for legacy parser compatibility (`alm_utils.py:1155`).
- `_penalty_values` accepts both scalar and matched-vector penalty (`alm_utils.py:1578-1588`); only scalar reaches it from `minimize_alm` (vector branch retained for direct unit-test calls and future-compat).
- Single penalty per constraint: `_handle_alm_dual_update_transition` uses one `penalty_argument` against `preferred_dual_update_values` (`alm_utils.py:2937-2967`); no per-block ρ.
- Dual update is inequality-style (clip-to-zero) only: `_updated_nonnegative_multipliers` and `_project_nonnegative_multipliers_with_diagnostics` apply `max(0, λ + ρ·c)` with optional cap (`alm_utils.py:1543-1574`). The equality `augmented_objective` helper was deleted (per Phase 7); only `tests/geo/test_single_stage_alm_integration.py:658` retains a regression assertion.
- Inequality-tag dispatch is at the constraint-source level: every `ALMConstraintMetadata` carries `dual_update_value_kind ∈ {"surrogate", "hard"}` and the routing layer (`_extract_stage2_constraint_signal_state`, `_constraint_routing_state`) selects `preferred_dual_update_values` from explicit `hard_dual_update_values` when stage-2 signals are present (alm_utils.py:1880-1925). No heuristic fallback after the 2026-05-06 hardening (Phase 5).
- Signed-semantics fix (`bfd4b5195`): `_explicit_raw_signed_constraint_values` (alm_utils.py:600-608) sources `raw_constraint_values` (signed) — never `raw_feasibility_values` (clipped). `_build_alm_result` writes `raw_constraint_values` to `result.raw_constraint_values` directly without falling back to clipped feasibility (alm_utils.py:2618, 2664). Single-stage and Stage 2 both populate `raw_constraint_values` from the unsanitized signed values (`single_stage_objectives.py:1106`, `stage2_objectives.py:1965`).
- Diagnostics labelling (per `b75c47e91`, `0df39cef3`, `30e6c06c9`):
  - `_constraint_label_history_diagnostics` (alm_utils.py:633-721) groups `feasibility_values` (normalized) by constraint label, reports `block_max_normalized_violation` from normalized space and `block_max_raw_hard_violation` from explicit raw sidecar. Confusion between normalized and raw is impossible because the two values come from different evaluation keys.
  - `_alm_summary` (alm_utils.py:1038-1102) emits both `max_normalized_violation` (normalized solver gate) and `max_raw_hard_violation_by_constraint` (raw certification map), labeled clearly.
  - `_multiplier_interpretation` (alm_utils.py:798-806) returns `"search_multipliers"` whenever `gradient_value_kind != dual_update_value_kind`, preventing downstream consumers from treating mixed-source multipliers as physical KKT duals.
- Activity tolerances flow normalized-only into `_kkt_stationarity_norm` and `_constraint_activity_mask` (alm_utils.py:1941-1964, 2041-2073). Tolerances are `scale · fraction` raw → `fraction` after normalization, uniform across constraints (e.g. `1e-3` for hardware constraints with `ALM_ACTIVITY_TOLERANCE_FRACTION`).
- KKT residual (`_kkt_stationarity_norm`) includes any constraint with `c_value ≥ -tolerance` — i.e. strongly violated constraints are kept in the active set (line 2063-2064). This matches Phase 4's contract that "the active-set exclusion that skips strongly violated constraints" was removed.
- ALM evaluation finiteness check (`_nonfinite_evaluation_fields`, alm_utils.py:1169-1214) covers `total`, `grad`, `constraint_values`, `feasibility_values`, `dual_update_values`, `metric_grad`, `base_grad`, `constraint_activity_tolerances`, and every `constraint_grads[i]`. `constraint_scales` non-finiteness is caught at `normalize_alm_constraint_signals` (alm_utils.py:444-445) instead — different layer but no gap.
- `_extract_constraint_state` is fail-fast: missing `feasibility_values` or `dual_update_values` raises `KeyError` (alm_utils.py:1856-1859); shape mismatches against `constraint_values` raise `ValueError` (line 1862-1865). The Phase 5 strict-signal contract is enforced.
- `_extract_stage2_constraint_signal_state` requires the full quartet `{hard_signed_constraint_values, hard_violation_values, surrogate_signed_constraint_values, hard_dual_update_values}` whenever any one is present (alm_utils.py:1899-1907). No silent fallback.
- Resume contract: `validate_resume_alm_state` (`single_stage_banana_example.py:4343-4367`) enforces exact `constraint_names` element-wise match and `multipliers.shape == (len(constraint_names),)`. The ALM checkpoint state serializes `constraint_names`, `multipliers`, and `penalty`. Unrelated checkpoints can no longer be reused after constraint-name rewiring (Phase 6).
- `workflow_runner_common.run_command` strips inherited `ALM_*` env vars (per Phase 8) so a parent shell `ALM_PENALTY_INIT=1e6` cannot silently override an explicit `--alm-penalty-init=1.0` flag in a child run.
- `validate_alm_cli_args` (alm_utils.py:318-359) covers penalty/tolerance/trust-radius positivity and `penalty_max ≥ penalty_init`. Missing only the physics-threshold positivity (F1).
- Public `ALM_FINAL_CONSTRAINT_VALUES` artifact field reads `raw_constraint_values` first (`single_stage_banana_example.py:6044-6048`, `stage2_objectives.py:1015-1019`); the `constraint_values` fallback in `stage2_objectives.py:1018` is dead code today because every Stage 2/single-stage evaluation populates `raw_constraint_values` (single_stage_objectives.py:1106, stage2_objectives.py:1965). Worth deleting eventually but harmless now.
- The fixture benchmark at `run_alm_normalization_fixture_benchmark.py` runs both raw-units and normalized-units formulations on two fixtures and emits a comparison row (`alm_fixture_benchmarking.py:88-112, 274-293`). Tests assert that both formulations share the same feasible set at the upper-bound iterate and that both rows are emitted (`tests/geo/test_alm_fixture_benchmarking.py:16-77`). It does **not** pin numerical AL values, multipliers, or dual estimates — a silent regression that breaks `λ̃ = λ·s` while preserving feasibility could pass this benchmark. Recommend pinning expected `multipliers`, `final_objective`, and `final_normalized_max_violation` for at least the `two_scale_hardware_boundary` fixture under deterministic settings.

## Closing Notes

The April–May 2026 normalization rework is well-engineered: the field contract is explicit, raw vs normalized labels are honored, every solver decision uses normalized values, and every certification/reporting decision uses raw values. Block-penalty removal is clean — only legacy schema fields and label-only `constraint_blocks` survived, exactly as planned. The two real findings (F1, F2) are entry-boundary and observability gaps that don't break the math but make user errors harder to surface. F3 is purely defensive.
