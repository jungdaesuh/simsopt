# ALM Constraint Normalization and Block Penalty Implementation Plan

Date: 2026-04-26
Status: Phases 0 through 3 implemented; Phase 4 still conditional
Scope: `examples/single_stage_optimization/alm_utils.py`, `banana_opt/hardware_constraint_schema.py`, `banana_opt/stage2_objectives.py`, `banana_opt/single_stage_objectives.py`, ALM result reporting, and focused ALM tests

## Review Verdict

Approve Phases 0 through 3 for implementation. Treat Phase 4 as conditional until the normalized scalar ALM path has tests and at least dry-run evidence.

The current ALM engine is already a real projected inequality augmented Lagrangian, not a simple penalty wrapper. It has L-BFGS-B inner solves, boxed trust-radius subproblems, nonfinite candidate rejection, multiplier and penalty caps, hard-vs-surrogate routing, KKT-style stationarity handling, history callbacks, and best-feasible restore.

The validated remaining problem is scale control. Stage 2 and single-stage ALM still pass raw signed residuals into the same scalar penalty/multiplier update path. Those residuals mix meters, inverse meters, amperes, and physics objective units. That makes one penalty parameter do too many jobs and makes multiplier magnitudes difficult to compare across constraint families.

Do not start with solver replacement, multi-surface ALM, or independent-current ALM. First make the existing ALM contract unit-aware and diagnostic. The key required edit before coding is to make the normalized-vs-raw field contract explicit inside `minimize_alm`, because current hard/surrogate fields are not just report fields; they affect multiplier updates, activity masks, stationarity, positive-shift logic, and signal-mismatch penalty decisions.

## Implementation Status

Current implementation status:

- [x] Phase 0 baseline artifact scan and scale-invariance tests landed.
- [x] Phase 1 metadata sidecars landed for Stage 2 and single-stage ALM constraints.
- [x] Phase 2 normalized scalar ALM landed for Stage 2 and single-stage objective construction.
- [x] Public result writers preserve raw `ALM_FINAL_*` fields and add normalized fields plus `ALM_SCHEMA_VERSION`.
- [x] `--alm-feas-tol` help text now identifies the tolerance as dimensionless and normalized.
- [x] ALM history now emits actionable raw/normalized sidecars, projected positive shifts, augmented terms, active pressure, constraint scales, blocks, normalized multipliers, raw dual estimates, surrogate-hard gap arrays, sign mismatch arrays, objective-to-augmented-term ratio, separated gradient/stationarity labels, and block summaries.
- [x] Final ALM result payloads now emit `ALM_SUMMARY`, `ALM_MULTIPLIER_INTERPRETATION`, `ALM_FINAL_AUGMENTED_GRADIENT_NORM`, and `ALM_FINAL_SURROGATE_KKT_STATIONARITY_NORM`.
- [ ] Phase 4 block penalties are not implemented.

Baseline artifact scan generated:

```text
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports/alm_normalization_benchmarks/baseline_20260426T042650Z.jsonl
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports/alm_normalization_benchmarks/comparison_20260426T042650Z.csv
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports/alm_normalization_benchmarks/comparison_20260426T042650Z.md
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports/alm_normalization_benchmarks/fixture_manifest_20260426T042650Z.json
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports/alm_normalization_benchmarks/skipped_artifacts_20260426T042650Z.json
```

Baseline scan counts:

```text
baseline_rows: 1412
harvested_seed_fixtures: 52
ledger_rows: 72
registry_rows: 43
run_artifact_rows: 1297
skipped_artifacts: 2
```

## Current Code Evidence

### Raw ALM constraint vectors

Stage 2 builds hard and surrogate signed values, then calls `augmented_inequality_objective(...)` with the surrogate signed values directly. The payload also stores raw hard values and raw hard violations for feasibility routing.

Relevant code:

- `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
- `evaluate_stage2_alm_problem(...)`
- raw fields: `hard_signed_constraint_values`, `surrogate_signed_constraint_values`, `hard_violation_values`, `dual_update_values`

Single-stage ALM similarly builds hardware and physics constraint tuples and passes raw `constraint_values` and gradients into `augmented_inequality_objective(...)`.

Relevant code:

- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`
- `evaluate_alm_objective(...)`
- raw fields: `dual_update_values`, `feasibility_values`, `constraint_grads`

### Schema has thresholds but no ALM scale contract

`HardwareConstraintSpec` currently owns:

```text
name
kind
threshold
applies_to
traversal_policy
```

It does not own an ALM scale, block label, or ALM activity tolerance. Thresholds are enough for pass/fail and signed-value construction, but not enough to normalize residuals consistently.

### ALM has scalar penalty state

`ALMSettings` owns one penalty:

```text
penalty_init
penalty_scale
penalty_max
```

Multiplier projection uses:

```text
lambda_i <- max(0, lambda_i + penalty * constraint_i)
```

There is no per-constraint or per-block penalty state.

### Hard-vs-surrogate support already exists

The code already detects hard/surrogate disagreement and exposes it in history/result payloads. Preserve this. The missing part is scale-aware gap reporting and automatic tuning policy, not the basic signal split.

## Target Contract

Every ALM constraint has two representations:

```text
raw_signed_value        physical/reporting units
raw_grad                gradient of raw_signed_value
raw_violation           max(raw_signed_value, 0) or exact hard violation
raw_activity_tolerance  physical/reporting activity tolerance
scale                   positive normalization scale
normalized_signed_value raw_signed_value / scale
normalized_grad         raw_grad / scale
normalized_violation    raw_violation / scale
normalized_activity_tolerance raw_activity_tolerance / scale
block                   geometry | current | physics | surface
```

ALM math should consume normalized signed values and normalized gradients. Artifacts should preserve raw values.

Field contract decision: existing ALM-internal fields become normalized after Phase 2. Raw values move to explicit `raw_*` names. This keeps `minimize_alm` simple and prevents accidental mixing of raw and normalized signals.

Normalized ALM-internal fields:

```text
constraint_values
dual_update_values
feasibility_values
hard_signed_constraint_values
hard_violation_values
surrogate_signed_constraint_values
hard_dual_update_values
constraint_grads
constraint_activity_tolerances
```

Raw sidecar fields:

```text
raw_constraint_values
raw_dual_update_values
raw_feasibility_values
raw_hard_signed_constraint_values
raw_hard_violation_values
raw_surrogate_signed_constraint_values
raw_hard_dual_update_values
raw_constraint_grads
raw_constraint_activity_tolerances
```

Value-source contract:

```text
objective_value_kind      surrogate | hard | raw_physics
gradient_value_kind       surrogate | hard | raw_physics
dual_update_value_kind    surrogate | hard
feasibility_value_kind    surrogate | hard
certification_value_kind  hard for hardware constraints
```

Each ALM constraint must declare which value source drives the differentiable ALM objective, gradient, multiplier update, optimizer feasibility, and raw certification. For smooth geometry constraints, the objective and gradient may come from a differentiable surrogate while raw certification comes from the exact hard check.

KKT-style stationarity is only a stationarity statement for the value and gradient pair used in the differentiable ALM objective. If `dual_update_value_kind != gradient_value_kind`, report the multipliers as search multipliers, not physical or KKT duals.

Public result fields that users reasonably expect to be in physical units should remain raw and backward-compatible. Add explicit normalized fields and an ALM schema/contract version instead of silently changing those public units.

Optimizer feasibility and certification feasibility are separate:

```text
ALM convergence:
    max_normalized_feasibility_violation <= ALMSettings.feasibility_tol

Hardware certification:
    max_raw_hard_violation_by_constraint <= raw engineering tolerance
```

Multiplier units after this change are normalized ALM units. For a raw constraint `c_raw(x) <= 0` and normalized constraint `c_norm(x) = c_raw(x) / scale`:

```text
lambda_norm * c_norm = (lambda_norm / scale) * c_raw
lambda_raw = lambda_norm / scale
lambda_norm = lambda_raw * scale
```

Emit both `normalized_multipliers` and `raw_dual_estimates = normalized_multipliers / constraint_scales` where useful.

## Design Invariants

- [x] Raw hard and surrogate values remain available in history and final artifacts.
- [x] ALM objective value and gradient are computed from normalized values only.
- [x] ALM multiplier updates, hard/surrogate routing, activity masks, and KKT stationarity operate on normalized values and normalized activity tolerances.
- [x] Every constraint declares value-source metadata for objective, gradient, dual update, feasibility, and certification signals.
- [x] Stationarity labels distinguish differentiable surrogate stationarity from physical/KKT dual interpretation.
- [x] `max_feasibility_violation` is normalized for optimizer convergence.
- [x] Raw hard pass/fail remains the certification source.
- [x] Public raw result fields stay raw; normalized result fields are added explicitly.
- [x] Constraint names stay stable for artifact consumers.
- [x] Existing penalty-mode behavior is unchanged.
- [x] Stage 2 and single-stage use the same scale source for shared hardware constraints.
- [x] No fallback remapping from unknown constraints; unknown names fail fast.
- [x] No dynamic imports or broad defensive wrappers are introduced.

## Scale Defaults

Use schema-owned defaults where possible.

| Constraint | Scale source |
| --- | --- |
| `coil_length_upper_bound` | Active ALM length threshold, such as per-run `length_target`; fall back to `COIL_LENGTH_HARD_LIMIT_M` only when it is the active constraint. |
| `coil_coil_spacing` | Active coil-coil distance threshold, such as per-run `cc_threshold` or `COIL_COIL_MIN_DIST_M`. |
| `coil_surface_spacing` | Active coil-surface distance threshold, such as per-run `cs_threshold` or `COIL_PLASMA_MIN_DIST_M`. |
| `surface_vessel_spacing` | Active surface-vessel threshold, such as `PLASMA_VESSEL_MIN_DIST_M` or the run override. |
| `max_curvature` | Active curvature threshold, such as per-run `curvature_threshold` or `MAX_CURVATURE_INV_M`. |
| `banana_current_upper_bound` | Actual ALM current threshold used by the constraint. Do not silently cap the scale at `BANANA_CURRENT_HARD_LIMIT_A` when an offspec override intentionally uses a larger threshold. |
| `iota_penalty` | Explicit penalty threshold as primary scale source: `scale = max(explicit_threshold, ALM_OBJECTIVE_SCALE_FLOOR)`. Raw feasibility still uses the original explicit threshold. |
| `qs_error` | Explicit `alm_qs_threshold`; fail fast when missing in `thresholded_physics`. |
| `boozer_residual` | Explicit `alm_boozer_threshold`; fail fast when missing in `thresholded_physics`. |
| `length_penalty` | Explicit length-penalty threshold or `ALM_OBJECTIVE_SCALE_FLOOR`, emitted in metadata. |

Use `max(scale, np.finfo(float).eps)` at the scale-construction boundary, not scattered through call sites. Record the scale source in metadata so small positive floors are visible in artifacts.

Named scale floors:

```python
ALM_PHYSICAL_SCALE_FLOOR = np.finfo(float).eps
ALM_OBJECTIVE_SCALE_FLOOR = 1.0e-12
```

Use `ALM_PHYSICAL_SCALE_FLOOR` only for unit-bearing hardware scales after threshold resolution. Use `ALM_OBJECTIVE_SCALE_FLOOR` for dimensionless or objective-like constraints such as iota penalty, Boozer residual, QS error, and length penalty. In `thresholded_physics`, missing QS, Boozer, iota, or length thresholds should fail fast; floors protect against zero or tiny explicit thresholds, not missing configuration.

## Implementation Plan

### Phase 0: Lock current behavior and add scale fixtures

- [x] Add a before/after benchmark collector that reads ALM history/result artifacts and emits a stable summary table.
- [x] Add a scale-invariance toy fixture:

```text
Given c_raw(x) <= 0 and c_norm(x) = c_raw(x) / scale, verify:
    lambda_raw = lambda_norm / scale
    raw feasibility is unchanged
    normalized feasibility changes as expected
    scale = 1 preserves existing scalar ALM behavior
```

- [x] Replace raw Stage 2 characterization expectations with normalized Phase 2 assertions.
- [x] Replace raw single-stage characterization expectations with normalized Phase 2 assertions.
- [x] Update hard/surrogate routing tests to assert normalized ALM routing plus raw sidecars.
- [x] Add schema tests for scale lookup on hardware constraints.
- [x] Add tests for physics threshold scales in `thresholded_physics` mode.

Expected files:

- `tests/geo/test_banana_objective_modules.py`
- `tests/geo/test_alm_utils.py`
- `tests/geo/test_single_stage_alm_integration.py`
- `examples/single_stage_optimization/banana_opt/alm_benchmarking.py` or `scripts/alm_compare_runs.py`

### Phase 1: Add ALM scale metadata without changing solver math

- [x] Extend `HardwareConstraintSpec` with `alm_scale: float | None`, `alm_block: ALMBlock | None`, and schema-owned ALM activity tolerance metadata where applicable.
- [x] Add helper:

```python
ALMBlock = Literal["geometry", "current", "physics", "surface"]
ALMValueKind = Literal["surrogate", "hard", "raw_physics"]


@dataclass(frozen=True)
class ALMConstraintMetadata:
    scale: float
    block: ALMBlock
    activity_tolerance: float
    raw_threshold: float | None
    source: str
    objective_value_kind: ALMValueKind
    gradient_value_kind: ALMValueKind
    dual_update_value_kind: Literal["surrogate", "hard"]
    feasibility_value_kind: Literal["surrogate", "hard"]
    certification_value_kind: Literal["hard"]


def hardware_constraint_alm_metadata(
    name: str,
    *,
    threshold_overrides: Mapping[str, float] | None = None,
) -> ALMConstraintMetadata:
    ...
```

- [x] Keep small compatibility helpers if useful, implemented from the metadata SSOT:

```python
def hardware_constraint_alm_scale(name: str, *, threshold_overrides: Mapping[str, float] | None = None) -> float:
    ...


def hardware_constraint_alm_block(name: str) -> str:
    ...
```

- [x] Emit ALM metadata and raw sidecars in Stage 2 and single-stage evaluations:

```text
constraint_scales
constraint_blocks
constraint_scale_sources
objective_value_kinds
gradient_value_kinds
dual_update_value_kinds
feasibility_value_kinds
certification_value_kinds
raw_dual_update_values
raw_feasibility_values
raw_constraint_grads
raw_constraint_activity_tolerances
```

- [x] Validate metadata at construction/lookup time:

```text
scale is finite and positive
activity_tolerance is finite and nonnegative
block is known
constraint name is known
scale source is nonempty
value-source kinds are compatible with the constraint family
```

At the end of Phase 1, optimizer behavior should be unchanged. Only payloads and tests change.

### Phase 2: Normalize ALM objective inputs

- [x] Add a small pure helper in `alm_utils.py`:

```python
def normalize_alm_constraints(
    signed_values,
    constraint_grads,
    feasibility_values,
    activity_tolerances,
    scales,
):
    ...
```

Return:

```text
normalized_signed_values
normalized_constraint_grads
normalized_feasibility_values
normalized_activity_tolerances
```

- [x] Normalize all ALM-consumed arrays at the Stage 2 and single-stage evaluation boundaries:

```python
def normalize_alm_payload(payload, scales):
    ...
```

- [x] Stage 2 should call `augmented_inequality_objective(...)` with normalized signed values and normalized gradients.
- [x] Single-stage should call `augmented_inequality_objective(...)` with normalized signed values and normalized gradients.
- [x] `dual_update_values`, `feasibility_values`, `hard_dual_update_values`, `hard_violation_values`, and hard/surrogate signed values consumed by ALM routing should become normalized values for ALM math.
- [x] `_constraint_activity_tolerances(...)`, `_constraint_activity_mask(...)`, and `_kkt_stationarity_norm(...)` should operate on normalized values and normalized tolerances.
- [x] Preserve raw values under explicit raw field names.
- [x] Keep public raw result fields raw where artifact consumers expect physical units. Add normalized fields explicitly.
- [x] Route public serialization through one result-building function so raw/normalized field decisions are not scattered through call sites.
- [x] Add an ALM schema/contract version to result payloads.
- [x] Update CLI/help text for `--alm-feas-tol`: after Phase 2 it is a dimensionless normalized ALM tolerance, not meters, amperes, inverse meters, or objective units.

Compatibility fields:

```text
ALM_FINAL_CONSTRAINT_VALUES                    raw, backward-compatible
ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES          new
ALM_FINAL_RAW_HARD_VIOLATION_BY_CONSTRAINT      new or clarified
ALM_FINAL_MAX_NORMALIZED_VIOLATION              new optimizer convergence value
ALM_SCHEMA_VERSION                              new
```

Acceptance tests:

- [x] A banana-current violation of `1000 A` with scale `16000 A` contributes as `0.0625`.
- [x] A curvature violation of `1 1/m` with scale `40 1/m` contributes as `0.025`.
- [x] Hard feasibility in raw units still fails/passes exactly as before.
- [x] Multiplier cap tests still pass in normalized units.
- [x] Convergence decisions use normalized feasibility only.
- [x] Hard feasibility failures are still reported in raw physical units.
- [x] Directional Taylor tests pass after normalization for Stage 2 and single-stage constraints.
- [x] Empty constraint sets, scalar constraints, vector constraints, gradient shape mismatches, nonfinite scales, and `scale=1` behavior are covered.

### Phase 3: Make ALM history actionable

- [x] Add initial per-constraint history fields:

```text
raw_signed_constraint_values
normalized_signed_constraint_values
raw_hard_violation_values
normalized_feasibility_values
constraint_scales
constraint_blocks
normalized_multipliers
raw_dual_estimates
positive_shift_values
augmented_term_by_constraint
active_pressure_by_constraint
surrogate_minus_hard_normalized_gap
surrogate_hard_sign_mismatch_by_constraint
objective_to_augmented_term_ratio
inner_lbfgsb_projected_gradient_norm
augmented_gradient_norm
surrogate_kkt_stationarity_norm
```

Use the projected inequality ALM term names rather than generic penalty contribution names. The implemented inequality objective is:

```text
positive_shift_i = max(0, lambda_i + rho * c_i)
augmented_term_i = (positive_shift_i**2 - lambda_i**2) / (2 * rho)
gradient += positive_shift_i * grad(c_i)
```

`augmented_term_i` can be negative for inactive constraints with nonzero multipliers, so do not label it as `penalty_contribution`.

- [x] Add complete derived block summaries:

```text
block_max_normalized_violation
block_max_raw_hard_violation
block_augmented_term
block_positive_shift_norm
blocking_constraint_name
blocking_constraint_block
```

- [x] Add a compact final `ALM_SUMMARY` artifact field for Stage 2 and single-stage:

```text
termination_reason
max_normalized_violation
max_raw_hard_violation_by_constraint
stationarity_norm
penalty
penalty_cap_reached
multiplier_cap_binding
signal_mismatch_active
blocking_constraint_name
blocking_constraint_block
```

Do not add automatic recommendations in this phase. Make the data sufficient first.

Keep the L-BFGS-B projected-gradient infinity norm, matching SciPy's `gtol`
projected-gradient max-component contract, separate from augmented-gradient norm
and surrogate KKT-style stationarity. Surrogate KKT-style stationarity uses the
differentiable surrogate signed values and surrogate feasibility gate; raw hard
feasibility remains a certification signal, not the active-set gate for that
diagnostic. These are different diagnostics and should not be collapsed into one
`stationarity_norm` without labels.

### Phase 4: Add block penalties after normalized scalar path is stable

Only start this phase after Phase 2 and Phase 3 tests pass.

- [ ] Add `ALMBlockPenaltyState` or equivalent internal state:

```text
block -> penalty
block -> penalty_cap_reached
block -> requested_penalty
```

- [ ] Extend `ALMSettings` with optional block penalty controls while preserving scalar defaults:

```text
block_penalties_enabled: bool = False
block_penalty_init: dict[str, float] | None = None
block_penalty_scale: dict[str, float] | None = None
block_penalty_max: dict[str, float] | None = None
```

- [ ] Refactor multiplier update to use the penalty for each constraint's block.
- [ ] Refactor augmented inequality objective to accept a vector penalty or block penalty vector.
- [ ] Use the explicit vector-penalty formula:

```text
rho_i = penalty for constraint i's block
s_i = max(0, lambda_i + rho_i * c_i)
augmented_term_i = (s_i**2 - lambda_i**2) / (2 * rho_i)
gradient += s_i * grad(c_i)
lambda_i <- max(0, lambda_i + rho_i * c_dual_update_i)
```

- [ ] Penalty growth should apply only to blocks whose normalized violation fails to improve.
- [ ] Add hysteresis to prevent block penalty growth from tiny numerical fluctuations:

```text
improvement_fraction = 0.9
patience = 1

if block_violation > max(feas_tol, previous_block_violation * improvement_fraction):
    grow that block's penalty
else:
    keep that block's penalty
```

Use the patience counter so a single tiny oscillation does not trigger penalty growth.

- [ ] Preserve scalar penalty behavior as the default path until explicit block-penalty tests and real-run fixtures pass.
- [ ] Do not expose block penalty CLI flags immediately. Keep controls internal until real-run validation proves defaults are stable.
- [ ] Before enabling this path, prove:

```text
block_penalties_enabled = False gives the same behavior as scalar ALM
all constraints in one block reproduce scalar ALM
different blocks only change rho_i, not scale, sign, or gradient semantics
block penalty caps are reported per block
```

Block defaults:

```text
geometry: coil length, coil-coil, coil-surface, max curvature
current: banana current
physics: qs_error, boozer_residual, iota_penalty, length_penalty
surface: surface-vessel spacing
```

### Phase 5: Revisit feature expansion

Do not implement these until normalization and diagnostics are stable.

- [ ] Independent banana-current ALM: one normalized `abs(I_i) - I_max <= 0` constraint per control current.
- [ ] Multi-surface ALM: surface spacing constraints only at first; topology remains a gate/certification signal.
- [ ] Adaptive smoothing: driven by normalized hard/surrogate gap counts.
- [ ] Distance acceleration: KD-tree candidate pruning for ALM surrogate gradients, while keeping exact hard checks.

## Test Matrix

CI-compatible runner:

The repository workflows run geo tests with `unittest discover`:

```bash
python -m unittest discover -t tests -v -s tests/geo
coverage run --source=simsopt -m unittest discover -t tests -v -s tests/geo
```

New ALM tests must be discoverable by `unittest`. Do not rely on pytest-only tests unless the CI workflows are also updated. Pytest commands are acceptable for local convenience only.

Run after Phase 1:

```bash
python3 -m unittest discover -t tests -v -s tests/geo -p "test_alm_utils.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_banana_objective_modules.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_single_stage_alm_integration.py"
```

Run after Phase 2 and later:

```bash
python3 -m unittest discover -t tests -v -s tests/geo -p "test_alm_utils.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_banana_objective_modules.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_single_stage_alm_integration.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_single_stage_example.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_stage2_single_stage_handoff.py"
python3 -m unittest discover -t tests -v -s tests/geo -p "test_single_stage_workflow_helpers.py"
```

Optional real-run validation:

```bash
python3 examples/single_stage_optimization/run_stage2_alm.py --dry-run
python3 examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py --dry-run
```

Wrapper dry runs validate command construction and marker/summary contracts only.
They intentionally report `output_contract=dry_run_summary_only` and
`contains_solver_outputs=false`, so final raw/normalized ALM payload fields are
validated through materialized result paths, production payload assembly, and
focused result-builder tests.

Required validation before merging Phase 2:

- [x] Unit tests confirm `1000 A / 16000 A = 0.0625`.
- [x] Unit tests confirm `1 1/m / 40 1/m = 0.025`.
- [x] Directional Taylor tests pass after normalization for Stage 2 and single-stage constraints.
- [x] Stage 2 result payloads emit both raw and normalized values.
- [x] Single-stage result payloads emit both raw and normalized values.
- [x] A hard feasibility failure is still reported in raw physical units.
- [x] A convergence decision uses normalized feasibility only.
- [x] Multiplier cap tests are updated to normalized units.
- [x] History clearly shows the same blocking constraint in both raw and normalized forms.
- [x] No penalty-mode behavior changes.
- [x] Hard-vs-surrogate mismatch tests still pass with normalized ALM routing and raw certification sidecars.
- [x] Value-source metadata is emitted for every ALM constraint.
- [x] Multipliers are labeled as search multipliers when dual-update values and gradient values come from different sources.
- [x] The plain CI runner executes the new tests through `unittest discover`.

## Before/After Impact Measurement

Measure the implementation as an optimizer-contract change. The first success criterion is not a lower final objective; it is that normalized ALM routing improves stability and diagnostics while preserving raw hardware certification.

### Artifact source contract

Executing this plan must record both the baseline and after-implementation impact. The benchmark collector should not rely only on fresh local runs; it should first snapshot the existing autoresearch artifacts and use them as the baseline source of truth where applicable.

Baseline artifact roots:

```text
/Users/suhjungdae/code/columbia/autoresearch/registry
/Users/suhjungdae/code/columbia/autoresearch/runs
/Users/suhjungdae/code/columbia/autoresearch/results_surrogate_legacy_vmec.jsonl
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports
/Users/suhjungdae/code/columbia/autoresearch/harvested_seeds
```

Required collector behavior:

```text
1. Read existing run metadata from the autoresearch registry and ledgers.
2. Identify ALM-relevant baseline runs by explicit run fields first, then artifact contents.
3. Link each baseline row to its source run directory, ledger row, registry row, and exported artifacts when present.
4. Derive benchmark fixtures from harvested seeds and prior successful or near-miss runs.
5. Save a frozen baseline summary before changing ALM behavior.
6. Run the after-implementation fixtures with the same seed/config contracts.
7. Save the after summary with the same schema as the baseline summary.
8. Emit a joined before/after comparison table.
```

Recommended output location:

```text
/Users/suhjungdae/code/columbia/autoresearch/artifact_exports/alm_normalization_benchmarks/
    baseline_YYYYMMDD.jsonl
    after_YYYYMMDD.jsonl
    comparison_YYYYMMDD.csv
    comparison_YYYYMMDD.md
    fixture_manifest_YYYYMMDD.json
```

Baseline rows should preserve provenance fields:

```text
source_kind                  registry | ledger | run_artifact | harvested_seed
source_path
run_id
ledger_file
ledger_row_index
registry_database
registry_table
artifact_export_path
seed_artifact_path
solver_checkout
solver_commit
created_at
```

If an older artifact does not contain normalized ALM fields, set normalized fields to null and mark:

```text
normalization_contract_version = "pre_normalization"
normalized_fields_available = false
```

Do not mutate the existing registry, ledger, run directories, artifact exports, or harvested seeds during baseline capture. The collector should create new benchmark summaries only.

### Benchmark cases

Use fixed seeds/configs and identical CLI arguments for baseline and implementation runs.

```text
Stage 2:
    easy feasible seed
    mildly infeasible geometry seed
    current-limited seed
    curvature-limited seed

Single-stage weighted_sum:
    feasible-ish seed
    geometry-stressed seed
    current-stressed seed

Single-stage thresholded_physics:
    QS/Boozer threshold-stressed seed
    iota/length threshold-stressed seed
```

If the run path is deterministic, one run per fixture is enough for contract validation. If any stochastic step is enabled, run at least three replicates per fixture and report median plus worst case.

### Metrics to collect

Convergence reliability:

```text
hard_feasible_success
normalized_alm_feasible_success
restored_best_feasible
outer_iterations_to_first_hard_feasible
outer_iterations_to_final_feasible
```

Constraint behavior:

```text
max_raw_hard_violation_by_constraint
max_normalized_violation_by_constraint
blocking_constraint_name
blocking_constraint_block
hard_surrogate_mismatch_count
multiplier_cap_hit_count
penalty_cap_hit_count
```

Optimizer stiffness:

```text
final_penalty
penalty_growth_event_count
stationarity_norm_initial
stationarity_norm_final
stationarity_norm_best
penalty_gradient_norm_over_base_gradient_norm
inner_stop_reason_counts
```

Performance:

```text
wall_time_s
objective_eval_count
gradient_eval_count
boozer_eval_count
biot_savart_eval_count
distance_eval_count
time_to_first_hard_feasible_s
time_to_best_feasible_objective_s
```

Solution quality:

```text
final_base_objective
best_feasible_base_objective
final_qs_error
final_boozer_residual
final_iota_metric
raw_engineering_margin_by_constraint
```

### Summary table

The benchmark collector should emit one row per case:

```text
case
before_success
after_success
before_best_feasible_objective
after_best_feasible_objective
before_max_raw_hard_violation
after_max_raw_hard_violation
before_max_normalized_violation
after_max_normalized_violation
before_outer_iters
after_outer_iters
before_evals
after_evals
before_wall_s
after_wall_s
before_penalty_cap_hits
after_penalty_cap_hits
before_multiplier_cap_hits
after_multiplier_cap_hits
blocking_constraint_before
blocking_constraint_after
```

### Phase-specific pass criteria

Phase 2 normalization passes when:

```text
hard-feasible success rate does not regress
best feasible objective does not regress beyond tolerance
raw certification decisions are unchanged for equivalent final candidates
normalized feasibility drives ALM convergence decisions
penalty and multiplier cap hits decrease or stay flat
history identifies the same blocking raw constraint in normalized form
```

Phase 4 block penalties pass when:

```text
penalty growth events decrease
final penalty decreases or stays localized to the blocking block
inner line-search and maxiter failures decrease
time to first hard-feasible point improves
best feasible objective improves under the same evaluation budget
```

Treat wall-clock speed as a secondary metric for Phases 2 and 4. Direct speedups should be expected only after caching and distance-acceleration work.

## Expected Impact

Before:

- One scalar penalty balances meters, inverse meters, amps, and objective values.
- A large raw amp violation can dominate normalized-looking geometry residuals.
- Multiplier values are hard to compare across constraints.
- Penalty cap and multiplier cap events are harder to interpret.

After Phase 2:

- Feasibility tolerances operate on dimensionless ALM units.
- Constraint multipliers are comparable across constraint families.
- Raw dual estimates are available through `normalized_multipliers / constraint_scales`.
- Existing scalar penalty tuning becomes less sensitive to physical units.
- Run history can separate raw certification failure from normalized optimizer pressure.

After Phase 4:

- Penalty growth can target only the blocking block.
- Geometry problems should stop over-stiffening current/physics constraints.
- Current or physics failures should stop globally inflating geometry penalties.

The likely first payoff is reliability and diagnosis quality, not raw wall-clock speed. Direct speed improvements should come later from caching and distance acceleration.

## Non-Goals

- [ ] Do not replace ALM with `trust-constr`.
- [ ] Do not add multi-surface ALM in the same patch.
- [ ] Do not add independent banana-current ALM before normalized scalar ALM lands.
- [ ] Do not change penalty-mode hardware traversal behavior.
- [ ] Do not remove hard feasibility certification fields.
- [ ] Do not make topology a differentiable ALM constraint.

## Resolved Decisions and Open Questions

- [x] Public result fields named `ALM_FINAL_CONSTRAINT_VALUES` should remain raw for backward compatibility. Add normalized fields explicitly.
- [x] `ALMSettings.feasibility_tol` should be documented as normalized-only after Phase 2.
- [x] Block penalties should not be exposed through CLI immediately.
- [x] Stage 2 iota penalty scale should use the explicit penalty threshold as the primary scale source, with `ALM_OBJECTIVE_SCALE_FLOOR` applied only to the scale.
- [x] Use `ALM_PHYSICAL_SCALE_FLOOR = np.finfo(float).eps` and `ALM_OBJECTIVE_SCALE_FLOOR = 1.0e-12` as the named floor constants.
